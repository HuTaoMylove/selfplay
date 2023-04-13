import torch
import torch.nn as nn
import numpy as np
from typing import Union, List
from .ppo_policy import PPOPolicy
from ..utils.buffer import SharedReplayBuffer
from ..utils.utils import check, get_gard_norm


class PPOTrainer:
    def __init__(self, args, device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        # ppo config
        self.args = args
        self.ppo_epoch = args.ppo_epoch
        self.clip_param = args.clip_param
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        # rnn configs
        self.data_chunk_length = args.data_chunk_length

    def ppo_update(self, policy: PPOPolicy, sample):

        if not self.args.use_hierarchical_network:
            obs_batch, share_obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample
            old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
            advantages_batch = check(advantages_batch).to(**self.tpdv)
            returns_batch = check(returns_batch).to(**self.tpdv)
            value_preds_batch = check(value_preds_batch).to(**self.tpdv)
            values, action_log_probs, dist_entropy = policy.evaluate_actions(share_obs_batch,
                                                                             obs_batch,
                                                                             rnn_states_actor_batch,
                                                                             rnn_states_critic_batch,
                                                                             actions_batch,
                                                                             masks_batch)

            # Obtain the loss function
            ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
            policy_loss = torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
            policy_loss = -policy_loss.mean()

            if self.use_clipped_value_loss:
                value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                            self.clip_param)
                value_losses = (values - returns_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
            else:
                value_loss = 0.5 * (returns_batch - values).pow(2)
            value_loss = value_loss.mean()

            policy_entropy_loss = -dist_entropy.mean()

            loss = policy_loss + value_loss * self.value_loss_coef + policy_entropy_loss * self.entropy_coef

            # Optimize the loss function
            policy.optimizer.zero_grad()
            loss.backward()
            if self.use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(policy.actor.parameters(), self.max_grad_norm).item()
                critic_grad_norm = nn.utils.clip_grad_norm_(policy.critic.parameters(), self.max_grad_norm).item()
            else:
                actor_grad_norm = get_gard_norm(policy.actor.parameters())
                critic_grad_norm = get_gard_norm(policy.critic.parameters())
            policy.optimizer.step()

            return policy_loss, value_loss, policy_entropy_loss, ratio, actor_grad_norm, critic_grad_norm
        else:

            p_obs_batch, p_share_obs_batch, p_actions_batch, p_masks_batch, \
            p_old_action_log_probs_batch, p_advantages_batch, p_returns_batch, p_value_preds_batch, \
            p_rnn_states_actor_batch, p_rnn_states_critic_batch = sample

            for obs_batch, share_obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
                returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch, p in zip(p_obs_batch, \
                                                                                                            p_share_obs_batch,
                                                                                                            p_actions_batch,
                                                                                                            p_masks_batch, \
                                                                                                            p_old_action_log_probs_batch,
                                                                                                            p_advantages_batch, \
                                                                                                            p_returns_batch,
                                                                                                            p_value_preds_batch, \
                                                                                                            p_rnn_states_actor_batch,
                                                                                                            p_rnn_states_critic_batch,
                                                                                                            policy):
                old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
                advantages_batch = check(advantages_batch).to(**self.tpdv)
                returns_batch = check(returns_batch).to(**self.tpdv)
                value_preds_batch = check(value_preds_batch).to(**self.tpdv)
                values, action_log_probs, dist_entropy = p.evaluate_actions(share_obs_batch,
                                                                            obs_batch,
                                                                            rnn_states_actor_batch,
                                                                            rnn_states_critic_batch,
                                                                            actions_batch,
                                                                            masks_batch)

                # Obtain the loss function
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
                policy_loss = torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                policy_loss = -policy_loss.mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = (values - returns_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                    value_loss = 0.5 * torch.min(value_losses, value_losses_clipped)
                else:
                    value_loss = 0.5 * (returns_batch - values).pow(2)

                value_loss = value_loss.mean()

                policy_entropy_loss = -dist_entropy.mean()

                loss = policy_loss + value_loss * self.value_loss_coef + policy_entropy_loss * self.entropy_coef

                # Optimize the loss function
                p.optimizer.zero_grad()
                loss.backward()
                if self.use_max_grad_norm:
                    actor_grad_norm = nn.utils.clip_grad_norm_(p.actor.parameters(), self.max_grad_norm).item()
                    critic_grad_norm = nn.utils.clip_grad_norm_(p.critic.parameters(), self.max_grad_norm).item()
                else:
                    actor_grad_norm = get_gard_norm(p.actor.parameters())
                    critic_grad_norm = get_gard_norm(p.critic.parameters())
                p.optimizer.step()

            return policy_loss, value_loss, policy_entropy_loss, ratio, actor_grad_norm, critic_grad_norm

    def train(self, policy: PPOPolicy, buffer: SharedReplayBuffer, prob=None):
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_entropy_loss'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        if prob is not None:
            softmax = prob / prob.sum()
            for i in range(len(softmax)):
                lr = np.clip(self.args.lr if softmax[i] > 0.6 else self.args.lr * (softmax[i] + 0.01) / 0.41, 1e-4,
                             1e-3).item()

                for params in policy[i].optimizer.param_groups:  # 遍历Optimizer中的每一组参数
                    params['lr'] = lr

        for _ in range(self.ppo_epoch):
            data_generator = buffer.recurrent_generator(buffer.advantages, self.num_mini_batch, self.data_chunk_length)

            for sample in data_generator:
                policy_loss, value_loss, policy_entropy_loss, ratio, \
                actor_grad_norm, critic_grad_norm = self.ppo_update(policy, sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_entropy_loss'] += policy_entropy_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
