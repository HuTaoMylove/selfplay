import torch
import numpy as np
from typing import Union, List
from abc import ABC, abstractmethod
from algorithm.utils.utils import get_shape_from_space


class SharedReplayBuffer:
    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        # env config
        self.num_agents = num_agents
        self.n_rollout_threads = args.n_rollout
        # buffer config
        self.gamma = args.gamma
        self.buffer_size = args.buffer_size
        self.use_gae = args.use_gae
        self.gae_lambda = args.gae_lambda
        # rnn config
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers

        obs_shape = get_shape_from_space(obs_space)
        share_obs_shape = get_shape_from_space(share_obs_space)
        act_shape = get_shape_from_space(act_space)
        self.act_shape = act_shape
        # (o_0, s_0, a_0, r_0, d_0, ..., o_T, s_T)
        self.obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *obs_shape),
                            dtype=np.float32)
        self.share_obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *share_obs_shape),
                                  dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape),
                                dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # NOTE: masks[t] = 1 - dones[t-1], which represents whether obs[t] is a terminal state .... same for all agents
        self.masks = np.ones((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # pi(a)
        self.action_log_probs = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape),
                                         dtype=np.float32)
        # V(o), R(o) while advantage = returns - value_preds
        self.value_preds = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1),
                                    dtype=np.float32)
        self.returns = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # rnn
        self.rnn_states_actor = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents,
                                          self.recurrent_hidden_layers, self.recurrent_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states_actor)

        self.step = 0

    @staticmethod
    def _cast(x: np.ndarray):
        return x.transpose(1, 2, 0, *range(3, x.ndim)).reshape(-1, *x.shape[3:])

    @staticmethod
    def _flatten(T: int, N: int, x: np.ndarray):
        return x.reshape(T * N, *x.shape[2:])

    def clear(self):
        self.step = 0
        self.obs = np.zeros_like(self.obs, dtype=np.float32)
        self.actions = np.zeros_like(self.actions, dtype=np.float32)
        self.rewards = np.zeros_like(self.rewards, dtype=np.float32)
        self.masks = np.ones_like(self.masks, dtype=np.float32)
        self.action_log_probs = np.zeros_like(self.action_log_probs, dtype=np.float32)
        self.value_preds = np.zeros_like(self.value_preds, dtype=np.float32)
        self.returns = np.zeros_like(self.returns, dtype=np.float32)
        self.rnn_states_actor = np.zeros_like(self.rnn_states_critic)
        self.rnn_states_critic = np.zeros_like(self.rnn_states_actor)
        self.share_obs = np.zeros_like(self.share_obs)

    @property
    def advantages(self) -> np.ndarray:
        advantages = self.returns[:-1] - self.value_preds[:-1]  # type: np.ndarray
        return (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def insert(self,
               obs: np.ndarray,
               share_obs: np.ndarray,
               actions: np.ndarray,
               rewards: np.ndarray,
               masks: np.ndarray,
               action_log_probs: np.ndarray,
               value_preds: np.ndarray,
               rnn_states_actor: np.ndarray,
               rnn_states_critic: np.ndarray,
               **kwargs):
        """Insert numpy data.
        Args:
            obs:                o_{t+1}
            actions:            a_{t}
            rewards:            r_{t}
            masks:              mask[t+1] = 1 - done_{t}
            action_log_probs:   log_prob(a_{t})
            value_preds:        value(o_{t})
            rnn_states_actor:   ha_{t+1}
            rnn_states_critic:  hc_{t+1}
        """

        self.obs[self.step + 1] = obs.copy()
        self.share_obs[self.step + 1] = share_obs.copy()
        self.actions[self.step] = actions.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rnn_states_actor[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.step = (self.step + 1) % self.buffer_size

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.rnn_states_actor[0] = self.rnn_states_actor[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()

    def recurrent_generator(self, advantages: np.ndarray, num_mini_batch: int, data_chunk_length: int):
        """
        A recurrent generator that yields training data for chunked RNN training arranged in mini batches.
        This generator shuffles the data by sequences.

        Args:
            advantages (np.ndarray): advantage estimates.
            num_mini_batch (int): number of minibatches to split the batch into.
            data_chunk_length (int): length of sequence chunks with which to train RNN.

        Returns:
            (obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, \
                old_action_log_probs_batch, advantages_batch, returns_batch, value_preds_batch, \
                rnn_states_actor_batch, rnn_states_critic_batch)
        """
        assert self.n_rollout_threads * self.buffer_size >= data_chunk_length, (
            "PPO requires the number of processes ({}) * buffer size ({}) "
            "to be greater than or equal to the number of data chunk length ({}).".format(
                self.n_rollout_threads, self.buffer_size, data_chunk_length))

        # Transpose and reshape parallel data into sequential data
        obs = self._cast(self.obs[:-1])
        share_obs = self._cast(self.share_obs[:-1])
        actions = self._cast(self.actions)
        masks = self._cast(self.masks[:-1])
        old_action_log_probs = self._cast(self.action_log_probs)
        advantages = self._cast(advantages)
        returns = self._cast(self.returns[:-1])
        value_preds = self._cast(self.value_preds[:-1])
        rnn_states_actor = self._cast(self.rnn_states_actor[:-1])
        rnn_states_critic = self._cast(self.rnn_states_critic[:-1])

        # Get mini-batch size and shuffle chunk data
        data_chunks = self.n_rollout_threads * self.buffer_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            obs_batch = []
            share_obs_batch = []
            actions_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            advantages_batch = []
            returns_batch = []
            value_preds_batch = []
            rnn_states_actor_batch = []
            rnn_states_critic_batch = []

            for index in indices:
                ind = index * data_chunk_length
                # size [T+1, N, M, Dim] => [T, N, M, Dim] => [N, M, T, Dim] => [N * M * T, Dim] => [L, Dim]
                obs_batch.append(obs[ind:ind + data_chunk_length])
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(old_action_log_probs[ind:ind + data_chunk_length])
                advantages_batch.append(advantages[ind:ind + data_chunk_length])
                returns_batch.append(returns[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                # size [T+1, N, M, Dim] => [T, N, M, Dim] => [N, M, T, Dim] => [N * M * T, Dim] => [1, Dim]
                rnn_states_actor_batch.append(rnn_states_actor[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)
            obs_batch = np.stack(obs_batch, axis=1)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            advantages_batch = np.stack(advantages_batch, axis=1)
            returns_batch = np.stack(returns_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_actor_batch = np.stack(rnn_states_actor_batch).reshape(N, *self.rnn_states_actor.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = self._flatten(L, N, obs_batch)
            share_obs_batch = self._flatten(L, N, share_obs_batch)
            actions_batch = self._flatten(L, N, actions_batch)
            masks_batch = self._flatten(L, N, masks_batch)
            old_action_log_probs_batch = self._flatten(L, N, old_action_log_probs_batch)
            advantages_batch = self._flatten(L, N, advantages_batch)
            returns_batch = self._flatten(L, N, returns_batch)
            value_preds_batch = self._flatten(L, N, value_preds_batch)

            yield obs_batch, share_obs_batch, actions_batch, masks_batch, \
                  old_action_log_probs_batch, advantages_batch, returns_batch, value_preds_batch, \
                  rnn_states_actor_batch, rnn_states_critic_batch

    def compute_returns(self, next_value: np.ndarray):
        """
        Compute returns either as discounted sum of rewards, or using GAE.

        Args:
            next_value(np.ndarray): value predictions for the step after the last episode step.
        """
        self.value_preds[-1] = next_value
        self.rewards = np.expand_dims(np.repeat(self.rewards.squeeze(-1).mean(-1, keepdims=True), self.num_agents, -1),
                                      -1)
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            td_delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                       self.value_preds[step]
            gae = td_delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]


class HierSharedReplayBuffer:
    def __init__(self, args, num_agents, obs_space, share_obs_space, act_space):
        # env config
        self.num_agents = num_agents
        self.n_rollout_threads = args.n_rollout
        # buffer config
        self.gamma = args.gamma
        self.buffer_size = args.buffer_size
        self.use_gae = args.use_gae
        self.gae_lambda = args.gae_lambda
        # rnn config
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers

        obs_shape = get_shape_from_space(obs_space)
        share_obs_shape = get_shape_from_space(share_obs_space)
        act_shape = get_shape_from_space(act_space)
        self.act_shape = act_shape
        # (o_0, s_0, a_0, r_0, d_0, ..., o_T, s_T)
        self.obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *obs_shape),
                            dtype=np.float32)
        self.share_obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *share_obs_shape),
                                  dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape),
                                dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # NOTE: masks[t] = 1 - dones[t-1], which represents whether obs[t] is a terminal state .... same for all agents
        self.masks = np.ones((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # pi(a)
        self.action_log_probs = [np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape),
                                          dtype=np.float32) for _ in range(5)]
        # V(o), R(o) while advantage = returns - value_preds
        self.value_preds = [np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1),
                                     dtype=np.float32) for _ in range(5)]
        self.returns = [np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1),
                                 dtype=np.float32) for _ in range(5)]
        # rnn
        self.rnn_states_actor = [np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents,
                                           self.recurrent_hidden_layers, i), dtype=np.float32) for i in
                                 [48, 64, 72, 96, 128]]
        self.rnn_states_critic = [np.zeros_like(i) for i in self.rnn_states_actor]

        self.step = 0

    @staticmethod
    def _cast(x: np.ndarray):
        return x.transpose(1, 2, 0, *range(3, x.ndim)).reshape(-1, *x.shape[3:])

    @staticmethod
    def _flatten(T: int, N: int, x: np.ndarray):
        return x.reshape(T * N, *x.shape[2:])

    def clear(self):
        self.step = 0
        self.obs = np.zeros_like(self.obs, dtype=np.float32)
        self.actions = np.zeros_like(self.actions, dtype=np.float32)
        self.rewards = np.zeros_like(self.rewards, dtype=np.float32)
        self.masks = np.ones_like(self.masks, dtype=np.float32)
        for i in range(5):
            self.action_log_probs[i] = np.zeros_like(self.action_log_probs[i], dtype=np.float32)
            self.value_preds[i] = np.zeros_like(self.value_preds[i], dtype=np.float32)
            self.returns[i] = np.zeros_like(self.returns[i], dtype=np.float32)
            self.rnn_states_actor[i] = np.zeros_like(self.rnn_states_critic[i])
            self.rnn_states_critic[i] = np.zeros_like(self.rnn_states_actor[i])
        self.share_obs = np.zeros_like(self.share_obs)

    @property
    def advantages(self) -> np.ndarray:
        advantages = []
        for i in range(5):
            a = self.returns[i][:-1] - self.value_preds[i][:-1]  # type: np.ndarray
            a = (a - a.mean()) / (a.std() + 1e-5)
            advantages.append(a.copy())
        return advantages

    def insert(self,
               obs: np.ndarray,
               share_obs: np.ndarray,
               actions: np.ndarray,
               rewards: np.ndarray,
               masks: np.ndarray,
               action_log_probs: np.ndarray,
               value_preds: np.ndarray,
               rnn_states_actor: np.ndarray,
               rnn_states_critic: np.ndarray,
               **kwargs):
        """Insert numpy data.
        Args:
            obs:                o_{t+1}
            actions:            a_{t}
            rewards:            r_{t}
            masks:              mask[t+1] = 1 - done_{t}
            action_log_probs:   log_prob(a_{t})
            value_preds:        value(o_{t})
            rnn_states_actor:   ha_{t+1}
            rnn_states_critic:  hc_{t+1}
        """

        self.obs[self.step + 1] = obs.copy()
        self.share_obs[self.step + 1] = share_obs.copy()
        self.actions[self.step] = actions.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        for i in range(len(action_log_probs)):
            self.action_log_probs[i][self.step] = action_log_probs[i].copy()
            self.value_preds[i][self.step] = value_preds[i].copy()
            self.rnn_states_actor[i][self.step + 1] = rnn_states_actor[i].copy()
            self.rnn_states_critic[i][self.step + 1] = rnn_states_critic[i].copy()
        self.step = (self.step + 1) % self.buffer_size

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        for i in range(5):
            self.rnn_states_actor[i][0]=self.rnn_states_actor[i][-1].copy()
            self.rnn_states_critic[i][0]=self.rnn_states_critic[i][-1].copy()


    def recurrent_generator(self, advantages: np.ndarray, num_mini_batch: int, data_chunk_length: int):
        """
        A recurrent generator that yields training data for chunked RNN training arranged in mini batches.
        This generator shuffles the data by sequences.

        Args:
            advantages (np.ndarray): advantage estimates.
            num_mini_batch (int): number of minibatches to split the batch into.
            data_chunk_length (int): length of sequence chunks with which to train RNN.

        Returns:
            (obs_batch, share_obs_batch, actions_batch, masks_batch, active_masks_batch, \
                old_action_log_probs_batch, advantages_batch, returns_batch, value_preds_batch, \
                rnn_states_actor_batch, rnn_states_critic_batch)
        """
        assert self.n_rollout_threads * self.buffer_size >= data_chunk_length, (
            "PPO requires the number of processes ({}) * buffer size ({}) "
            "to be greater than or equal to the number of data chunk length ({}).".format(
                self.n_rollout_threads, self.buffer_size, data_chunk_length))

        # Transpose and reshape parallel data into sequential data
        obs = self._cast(self.obs[:-1])
        share_obs = self._cast(self.share_obs[:-1])
        actions = self._cast(self.actions)
        masks = self._cast(self.masks[:-1])

        old_action_log_probs, returns, value_preds, rnn_states_actor, rnn_states_critic = [], [], [], [], []

        for i in range(5):
            old_action_log_probs.append(self._cast(self.action_log_probs[i]).copy())
            advantages[i]=self._cast(advantages[i])
            returns.append(self._cast(self.returns[i][:-1]).copy())
            value_preds.append(self._cast(self.value_preds[i][:-1]).copy())
            rnn_states_actor.append(self._cast(self.rnn_states_actor[i][:-1]).copy())
            rnn_states_critic.append(self._cast(self.rnn_states_critic[i][:-1]).copy())

        # Get mini-batch size and shuffle chunk data
        data_chunks = self.n_rollout_threads * self.buffer_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        for indices in sampler:
            p_obs_batch = []
            p_share_obs_batch = []
            p_actions_batch = []
            p_masks_batch = []
            p_old_action_log_probs_batch = []
            p_advantages_batch = []
            p_returns_batch = []
            p_value_preds_batch = []
            p_rnn_states_actor_batch = []
            p_rnn_states_critic_batch = []

            for i in range(5):
                obs_batch = []
                share_obs_batch = []
                actions_batch = []
                masks_batch = []
                old_action_log_probs_batch = []
                advantages_batch = []
                returns_batch = []
                value_preds_batch = []
                rnn_states_actor_batch = []
                rnn_states_critic_batch = []

                for index in indices:
                    ind = index * data_chunk_length
                    # size [T+1, N, M, Dim] => [T, N, M, Dim] => [N, M, T, Dim] => [N * M * T, Dim] => [L, Dim]
                    obs_batch.append(obs[ind:ind + data_chunk_length].copy())
                    share_obs_batch.append(share_obs[ind:ind + data_chunk_length].copy())
                    actions_batch.append(actions[ind:ind + data_chunk_length].copy())
                    masks_batch.append(masks[ind:ind + data_chunk_length].copy())
                    old_action_log_probs_batch.append(old_action_log_probs[i][ind:ind + data_chunk_length].copy())
                    advantages_batch.append(advantages[i][ind:ind + data_chunk_length].copy())
                    returns_batch.append(returns[i][ind:ind + data_chunk_length].copy())
                    value_preds_batch.append(value_preds[i][ind:ind + data_chunk_length].copy())
                    # size [T+1, N, M, Dim] => [T, N, M, Dim] => [N, M, T, Dim] => [N * M * T, Dim] => [1, Dim]
                    rnn_states_actor_batch.append(rnn_states_actor[i][ind].copy())
                    rnn_states_critic_batch.append(rnn_states_critic[i][ind].copy())

                L, N = data_chunk_length, mini_batch_size

                # These are all from_numpys of size (L, N, Dim)
                obs_batch = np.stack(obs_batch, axis=1)
                share_obs_batch = np.stack(share_obs_batch, axis=1)
                actions_batch = np.stack(actions_batch, axis=1)
                masks_batch = np.stack(masks_batch, axis=1)
                old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
                advantages_batch = np.stack(advantages_batch, axis=1)
                returns_batch = np.stack(returns_batch, axis=1)
                value_preds_batch = np.stack(value_preds_batch, axis=1)

                # States is just a (N, -1) from_numpy
                rnn_states_actor_batch = np.stack(rnn_states_actor_batch).reshape(N, *self.rnn_states_actor[i].shape[3:])
                rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic[i].shape[3:])

                # Flatten the (L, N, ...) from_numpys to (L * N, ...)
                obs_batch = self._flatten(L, N, obs_batch)
                share_obs_batch = self._flatten(L, N, share_obs_batch)
                actions_batch = self._flatten(L, N, actions_batch)
                masks_batch = self._flatten(L, N, masks_batch)
                old_action_log_probs_batch = self._flatten(L, N, old_action_log_probs_batch)
                advantages_batch = self._flatten(L, N, advantages_batch)
                returns_batch = self._flatten(L, N, returns_batch)
                value_preds_batch = self._flatten(L, N, value_preds_batch)

                p_obs_batch.append(obs_batch.copy())
                p_share_obs_batch.append(share_obs_batch.copy())
                p_actions_batch.append(actions_batch.copy())
                p_masks_batch.append(masks_batch.copy())
                p_old_action_log_probs_batch.append(old_action_log_probs_batch.copy())
                p_advantages_batch.append(advantages_batch.copy())
                p_returns_batch.append(returns_batch.copy())
                p_value_preds_batch.append(value_preds_batch.copy())
                p_rnn_states_actor_batch.append(rnn_states_actor_batch.copy())
                p_rnn_states_critic_batch.append(rnn_states_critic_batch.copy())

            yield p_obs_batch, p_share_obs_batch, p_actions_batch, p_masks_batch, \
                  p_old_action_log_probs_batch, p_advantages_batch, p_returns_batch, p_value_preds_batch, \
                  p_rnn_states_actor_batch, p_rnn_states_critic_batch

    def compute_returns(self, next_value: np.ndarray):
        """
        Compute returns either as discounted sum of rewards, or using GAE.

        Args:
            next_value(np.ndarray): value predictions for the step after the last episode step.
        """
        for i in range(5):
            self.value_preds[i][-1] = next_value[i]
        self.rewards = np.expand_dims(np.repeat(self.rewards.squeeze(-1).mean(-1, keepdims=True), self.num_agents, -1),
                                      -1)
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            for i in range(5):
                td_delta = self.rewards[step] + self.gamma * self.value_preds[i][step + 1] * self.masks[step + 1] - \
                           self.value_preds[i][step]
                gae = td_delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                self.returns[i][step] = gae + self.value_preds[i][step]
