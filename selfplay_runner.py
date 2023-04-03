import time
from matplotlib.pyplot import axis
import torch
import logging
import numpy as np
from gym import spaces
from typing import List
from torch.utils.tensorboard import SummaryWriter
from algorithm.utils.buffer import SharedReplayBuffer
import os
import json


def _t2n(x):
    return x.detach().cpu().numpy()


class ShareRunner:
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.test_envs = config['test_envs']
        self.device = config['device']

        # parameters
        self.env_name = self.all_args.env_name
        self.num_env_steps = int(self.all_args.num_env_steps)
        self.n_rollout_threads = self.all_args.n_rollout
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout
        self.n_test_rollout_threads = self.all_args.n_test_rollout
        self.buffer_size = self.all_args.buffer_size

        # interval
        self.save_interval = self.all_args.save_interval
        self.log_interval = self.all_args.log_interval
        self.test_interval = self.all_args.test_interval
        self.test_episodes = self.all_args.test_episodes
        # eval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.eval_episodes = self.all_args.eval_episodes
        self.latest_elo = self.all_args.init_elo

        # dir
        self.log_dir = config["log_dir"] / self.all_args.selfplay_algorithm
        self.run_dir = config["run_dir"]
        self.save_dir = self.run_dir / self.all_args.selfplay_algorithm
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.writer = SummaryWriter(self.log_dir)
        self.load()

    def load(self):
        self.obs_space = self.envs.observation_space
        self.share_obs_space = self.envs.share_observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        # policy & algorithm

        from algorithm.mappo.ppo_trainer import PPOTrainer as Trainer
        from algorithm.mappo.ppo_policy import PPOPolicy as Policy

        self.policy = Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)
        self.buffer = SharedReplayBuffer(self.all_args, self.num_agents // 2, self.obs_space, self.share_obs_space,
                                         self.act_space)

        # [Selfplay] allocate memory for opponent policy/data in training
        from algorithm.utils.selfplay import get_algorithm
        self.selfplay_algo = get_algorithm(self.all_args.selfplay_algorithm)

        assert self.all_args.n_choose_opponents <= self.n_rollout_threads, \
            "Number of different opponents({}) must less than or equal to number of training threads({})!" \
                .format(self.all_args.n_choose_opponents, self.n_rollout_threads)

        self.policy_pool = {'0': self.all_args.init_elo}  # type: dict[str, float]
        self.opponent_policy = [
            Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
            for _ in range(self.all_args.n_choose_opponents)]
        self.opponent_env_split = np.array_split(np.arange(self.n_rollout_threads), len(self.opponent_policy))

        self.opponent_obs = np.zeros_like(self.buffer.obs[0])
        self.opponent_rnn_states = np.zeros_like(self.buffer.rnn_states_actor[0])
        self.opponent_masks = np.ones_like(self.buffer.masks[0])
        self.max_policy_size = self.all_args.n_choose_opponents

        self.eval_opponent_policy = Policy(self.all_args, self.obs_space, self.share_obs_space, self.act_space,
                                           device=self.device)

        if len(os.listdir(self.save_dir)) > 0:
            self.restore()

        logging.info("\n Load selfplay opponents: Algo {}, num_opponents {}.\n"
                     .format(self.all_args.selfplay_algorithm, self.all_args.n_choose_opponents))

    def restore(self):
        policy_actor_state_dict = torch.load(str(self.save_dir) + '/actor_latest.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = torch.load(str(self.save_dir) + '/critic_latest.pt')
        self.policy.critic.load_state_dict(policy_critic_state_dict)
        if os.path.exists(str(self.save_dir) + '/elo.txt'):
            with open(str(self.save_dir) + '/elo.txt', encoding="utf-8") as file:
                self.policy_pool = json.load(file)
            keylist = []
            for i in self.policy_pool.keys():
                keylist.append(int(i))
            keylist.sort()
            self.latest_elo = self.policy_pool[str(keylist[-1])]
            self.reset_opponent()

    def train(self):
        self.policy.prep_training()
        train_infos = self.trainer.train(self.policy, self.buffer)
        self.buffer.after_update()
        return train_infos

    def run(self):
        self.warmup()

        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads
        already_trained_episodes = -1
        if len(os.listdir(str(self.log_dir))) > 1:
            from tensorboard.backend.event_processing import event_accumulator
            # ea = event_accumulator.EventAccumulator(str(self.log_dir / str(self.model_dir).split('\\')[-1]))
            ea = event_accumulator.EventAccumulator(str(self.log_dir))
            ea.Reload()
            # assert len(ea.scalars.Keys()) != 0, 'log data should not none!'
            already_trained_episodes = len(ea.scalars.Items('train/average_episode_rewards'))
            logging.info(f'already trained {already_trained_episodes} episodes')

        for episode in range(already_trained_episodes + 1, episodes):
            start = time.time()
            for step in range(self.buffer_size):
                # Sample actions
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos = self.envs.step(actions)
                data = obs, share_obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            # self.render()
            # post process
            self.total_num_steps = episode * self.buffer_size * self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                logging.info(
                    "\n updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(episode,
                                episodes,
                                self.total_num_steps,
                                self.num_env_steps,
                                int(self.buffer_size * self.n_rollout_threads / (end - start))))

                train_infos["average_episode_rewards"] = self.buffer.rewards[:, :, :2, :].sum() / (
                        (self.buffer.masks == False).sum() + 1)
                logging.info("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_info(train_infos, self.total_num_steps)

            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)

            if episode % self.eval_interval == 0 and episode != 0 and self.use_eval:
                self.eval(self.total_num_steps)

            if episode % self.test_interval == 0 and episode != 0:
                self.test(self.total_num_steps)

            # save model

    def warmup(self):
        # reset env
        obs, share_obs = self.envs.reset()
        self.opponent_obs = obs[:, self.num_agents // 2:, ...]
        obs = obs[:, :self.num_agents // 2, ...]
        share_obs = share_obs[:, :self.num_agents // 2, ...]
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                      np.concatenate(self.buffer.obs[step]),
                                      np.concatenate(self.buffer.rnn_states_actor[step]),
                                      np.concatenate(self.buffer.rnn_states_critic[step]),
                                      np.concatenate(self.buffer.masks[step]))
        # split parallel data [N*M, shape] => [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # [Selfplay] get actions of opponent policy
        opponent_actions = np.zeros_like(actions)
        for policy_idx, policy in enumerate(self.opponent_policy):
            env_idx = self.opponent_env_split[policy_idx]
            opponent_action, opponent_rnn_states \
                = policy.act(np.concatenate(self.opponent_obs[env_idx]),
                             np.concatenate(self.opponent_rnn_states[env_idx]),
                             np.concatenate(self.opponent_masks[env_idx]))
            opponent_actions[env_idx] = np.array(np.split(_t2n(opponent_action), len(env_idx)))
            self.opponent_rnn_states[env_idx] = np.array(np.split(_t2n(opponent_rnn_states), len(env_idx)))
        actions = np.concatenate((actions, opponent_actions), axis=1)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    @torch.no_grad()
    def compute(self):
        self.policy.prep_rollout()
        next_values = self.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                             np.concatenate(self.buffer.rnn_states_critic[-1]),
                                             np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.buffer.n_rollout_threads))
        self.buffer.compute_returns(next_values)

    def insert(self, data: List[np.ndarray]):
        obs, share_obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data
        rewards = np.expand_dims(rewards, axis=-1)
        dones = np.ones([self.n_rollout_threads, self.num_agents]) * dones.reshape(-1, 1)
        dones_env = np.all(dones, axis=-1)

        rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]),
                                                       dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]),
                                                        dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # [Selfplay] divide ego/opponent of collecting data TODO: shared_obs

        self.opponent_obs = obs[:, self.num_agents // 2:, ...]
        self.opponent_masks = masks[:, self.num_agents // 2:, ...]

        obs = obs[:, :self.num_agents // 2, ...]
        share_obs = share_obs[:, :self.num_agents // 2, ...]
        actions = actions[:, :self.num_agents // 2, ...]
        rewards = rewards[:, :self.num_agents // 2, ...]
        masks = masks[:, :self.num_agents // 2, ...]

        self.buffer.insert(obs, share_obs, actions, rewards, masks, action_log_probs, values, \
                           rnn_states_actor, rnn_states_critic)

    @torch.no_grad()
    def eval(self, total_num_steps):
        logging.info("\nStart evaluation...")
        self.policy.prep_rollout()
        total_episodes = 0
        episode_rewards, opponent_episode_rewards = [], []
        cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)
        opponent_cumulative_rewards = np.zeros_like(cumulative_rewards)

        # [Selfplay] Choose opponent policy for evaluation
        if len(self.policy_pool.keys()) == self.max_policy_size:
            eval_choose_opponents = list(self.policy_pool.keys())
        else:
            eval_choose_opponents = [self.selfplay_algo.choose(self.policy_pool) for _ in range(self.max_policy_size)]
        eval_each_episodes = self.eval_episodes // self.max_policy_size
        logging.info(f" Choose opponents {eval_choose_opponents} for evaluation")

        eval_cur_opponent_idx = 0
        while total_episodes < self.eval_episodes:

            # [Selfplay] Load opponent policy
            if total_episodes >= eval_cur_opponent_idx * eval_each_episodes:
                policy_idx = eval_choose_opponents[eval_cur_opponent_idx]
                self.eval_opponent_policy.actor.load_state_dict(
                    torch.load(str(self.save_dir) + f'/actor_{policy_idx}.pt'))
                self.eval_opponent_policy.prep_rollout()
                eval_cur_opponent_idx += 1
                logging.info(f" Load opponent {policy_idx} for evaluation ({total_episodes}/{self.eval_episodes})")

                # reset obs/rnn/mask
                obs, _ = self.eval_envs.reset()

                opponent_obs = obs[:, self.num_agents // 2:, ...]
                obs = obs[:, :self.num_agents // 2, ...]

                masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
                opponent_masks = np.ones_like(masks, dtype=np.float32)
                rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]),
                                      dtype=np.float32)
                opponent_rnn_states = np.zeros_like(rnn_states, dtype=np.float32)

            # [Selfplay] get actions
            actions, rnn_states = self.policy.act(np.concatenate(obs),
                                                  np.concatenate(rnn_states),
                                                  np.concatenate(masks), deterministic=False)
            actions = np.array(np.split(_t2n(actions), self.n_eval_rollout_threads))
            rnn_states = np.array(np.split(_t2n(rnn_states), self.n_eval_rollout_threads))

            opponent_actions, opponent_rnn_states \
                = self.eval_opponent_policy.act(np.concatenate(opponent_obs),
                                                np.concatenate(opponent_rnn_states),
                                                np.concatenate(opponent_masks), deterministic=False)
            opponent_rnn_states = np.array(np.split(_t2n(opponent_rnn_states), self.n_eval_rollout_threads))
            opponent_actions = np.array(np.split(_t2n(opponent_actions), self.n_eval_rollout_threads))
            actions = np.concatenate((actions, opponent_actions), axis=1)

            # Obser reward and next obs
            obs, _, eval_rewards, dones, eval_infos = self.eval_envs.step(actions)
            eval_rewards = np.expand_dims(eval_rewards, axis=-1)
            dones = np.ones([self.n_eval_rollout_threads, self.num_agents]) * dones.reshape(-1, 1)
            dones_env = np.all(dones, axis=-1)

            total_episodes += np.sum(dones_env)

            # [Selfplay] Reset obs, masks, rnn_states
            opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            masks[dones_env == True] = np.zeros(((dones_env == True).sum(), *masks.shape[1:]), dtype=np.float32)
            rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states.shape[1:]),
                                                     dtype=np.float32)

            opponent_masks[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), *opponent_masks.shape[1:]), dtype=np.float32)
            opponent_rnn_states[dones_env == True] = \
                np.zeros(((dones_env == True).sum(), *opponent_rnn_states.shape[1:]), dtype=np.float32)

            # [Selfplay] Get rewards
            opponent_rewards = eval_rewards[:, self.num_agents // 2:, ...]
            opponent_cumulative_rewards += opponent_rewards
            opponent_episode_rewards.append(opponent_cumulative_rewards[dones_env == True])
            opponent_cumulative_rewards[dones_env == True] = 0

            eval_rewards = eval_rewards[:, :self.num_agents // 2, ...]
            cumulative_rewards += eval_rewards
            episode_rewards.append(cumulative_rewards[dones_env == True])
            cumulative_rewards[dones_env == True] = 0

        # Compute average episode rewards
        # np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)
        episode_rewards = np.concatenate(episode_rewards)  # shape (self.eval_episodes, self.num_agents, 1)
        episode_rewards = episode_rewards.squeeze(-1).mean(axis=-1)  # shape: (self.eval_episodes,)
        eval_average_episode_rewards = np.array(np.split(episode_rewards, self.max_policy_size)).mean(
            axis=-1)  # shape (self.num_opponents,)

        opponent_episode_rewards = np.concatenate(opponent_episode_rewards)
        opponent_episode_rewards = opponent_episode_rewards.squeeze(-1).mean(axis=-1)
        opponent_average_episode_rewards = np.array(np.split(opponent_episode_rewards, self.max_policy_size)).mean(
            axis=-1)

        # Update elo
        ego_elo = np.array([self.latest_elo for _ in range(self.max_policy_size)])
        opponent_elo = np.array([self.policy_pool[key] for key in eval_choose_opponents])
        expected_score = 1 / (1 + 10 ** ((ego_elo - opponent_elo) / 400))

        actual_score = np.zeros_like(expected_score)
        diff = opponent_average_episode_rewards - eval_average_episode_rewards
        actual_score[diff >= 0.8] = 1  # win
        actual_score[abs(diff) <= 0.8] = 0.5  # tie
        actual_score[diff <= -0.8] = 0  # lose
        elo_gain = 32 * (actual_score - expected_score)
        update_opponent_elo = opponent_elo + elo_gain
        for i, key in enumerate(eval_choose_opponents):
            self.policy_pool[key] = update_opponent_elo[i]
        ego_elo = ego_elo - elo_gain
        self.latest_elo = ego_elo.mean()

        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = eval_average_episode_rewards.mean()
        # eval_infos['latest_elo'] = self.latest_elo
        logging.info(" eval average episode rewards: " + str(eval_infos['eval_average_episode_rewards']))
        logging.info(" latest elo score: " + str(self.latest_elo))
        self.log_info(eval_infos, total_num_steps, mode='eval')
        logging.info("...End evaluation")

        with open(str(self.save_dir) + '/elo.txt', 'w') as file:
            file.write(json.dumps(self.policy_pool))
            file.close()
        self.reset_opponent()

    def test(self, total_num_steps):
        logging.info("\nStart test...")
        self.policy.prep_rollout()
        obs, _ = self.test_envs.reset()
        masks = np.ones((self.n_test_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
        rnn_states = np.zeros((self.n_test_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]),
                              dtype=np.float32)

        rewards = np.zeros([self.n_test_rollout_threads, 1])
        epo = 0
        while epo < self.test_episodes:
            actions, rnn_states = self.policy.act(np.concatenate(obs),
                                                  np.concatenate(rnn_states),
                                                  np.concatenate(masks), deterministic=False)
            actions = np.array(np.split(_t2n(actions), self.n_test_rollout_threads))
            rnn_states = np.array(np.split(_t2n(rnn_states), self.n_test_rollout_threads))
            obs, _, eval_rewards, done, eval_infos = self.test_envs.step(actions)
            if done.all():
                obs, _ = self.test_envs.reset()
                masks = np.ones((self.n_test_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
                rnn_states = np.zeros((self.n_test_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]),
                                      dtype=np.float32)
                epo += 1
            else:
                rewards = rewards + eval_rewards.sum(axis=-1, keepdims=True)
        test_infos = {}
        for i in range(self.n_test_rollout_threads):
            test_infos[f'reward_{i + 1}'] = rewards[i] / self.test_episodes
        test_infos['reward_mean'] = rewards.sum() / self.test_episodes / self.n_test_rollout_threads
        self.log_info(test_infos, total_num_steps, 'test')
        logging.info("\nEnd test...")

    def save(self, episode):
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_latest.pt')
        policy_critic_state_dict = self.policy.critic.state_dict()
        torch.save(policy_critic_state_dict, str(self.save_dir) + '/critic_latest.pt')
        # [Selfplay] save policy & performance
        torch.save(policy_actor_state_dict, str(self.save_dir) + f'/actor_{episode}.pt')
        if self.max_policy_size <= len(self.policy_pool.keys()):
            if self.all_args.selfplay_algorithm == 'sp':
                keylist = []
                for i in self.policy_pool.keys():
                    keylist .append(int(i))
                keylist.sort()
                del self.policy_pool[str(keylist[0])]
                self.policy_pool[str(episode)] = self.latest_elo
            elif self.all_args.selfplay_algorithm == 'fsp':
                keylist = []
                for i in self.policy_pool.keys():
                    keylist.append(i)
                del self.policy_pool[keylist[np.random.choice(self.max_policy_size)]]
                self.policy_pool[str(episode)] = self.latest_elo
        else:
            self.policy_pool[str(episode)] = self.latest_elo

    def reset_opponent(self):
        choose_opponents = []
        for policy in self.opponent_policy:
            choose_idx = self.selfplay_algo.choose(self.policy_pool)
            choose_opponents.append(choose_idx)
            policy.actor.load_state_dict(torch.load(str(self.save_dir) + f'/actor_{choose_idx}.pt'))
            policy.prep_rollout()
        logging.info(f" Choose opponents {choose_opponents} for training")

        # clear buffer
        self.buffer.clear()
        self.opponent_obs = np.zeros_like(self.opponent_obs)
        self.opponent_rnn_states = np.zeros_like(self.opponent_rnn_states)
        self.opponent_masks = np.ones_like(self.opponent_masks)

        # reset env
        obs, share_obs = self.envs.reset()
        if self.all_args.n_choose_opponents > 0:
            self.opponent_obs = obs[:, self.num_agents // 2:, ...]
            obs = obs[:, :self.num_agents // 2, ...]
            share_obs = share_obs[:, :self.num_agents // 2, ...]
        self.buffer.obs[0] = obs.copy()
        self.buffer.share_obs[0] = share_obs.copy()

    def log_info(self, infos, total_num_steps, mode='train'):
        assert mode in ['train', 'eval', 'test']
        for k, v in infos.items():
            self.writer.add_scalar(mode + '/' + k, v, total_num_steps)
