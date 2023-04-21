import copy

import gym
import numpy as np
import torch
import torch.nn as nn

from ..utils.mlp import MLPBase
from ..utils.gru import GRULayer
from ..utils.act import ACTLayer
from ..utils.utils import check
from algorithm.utils.selfattention import SelfAttention


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(PPOActor, self).__init__()
        # network config
        self.gain = args.gain
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.algo = args.selfplay_algorithm

        self.tpdv = dict(dtype=torch.float32, device=device)
        # if self.algo == 'hsp':
        #     self.id = nn.Linear(5, 96)
        #     self.mode = nn.Linear(7, 96)
        #     self.pos = nn.Linear(2, 96)
        #     self.dir = nn.Linear(2, 96)
        #
        #     self.attention = SelfAttention(96, 96, 96)
        #     input_size = 99
        # else:
        #     self.base = MLPBase(obs_space, self.hidden_size, self.activation_id)
        #     input_size = self.base.output_size
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id)
        input_size = self.base.output_size
        self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
        input_size = self.rnn.output_size
        # (3) act module
        self.act = ACTLayer(act_space, input_size, self.act_hidden_size, self.activation_id, self.gain)
        self.to(device)

    def forward(self, obs, rnn_states, masks, att_mode=2, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self.algo == 'hsp':
            # id = self.id(obs[:, :5]).unsqueeze(1)
            # mode = self.mode(obs[:, 5:12]).unsqueeze(1)
            # pos = self.pos(obs[:, 12:22].reshape(-1, 5, 2))
            # dir = self.dir(obs[:, 22:].reshape(-1, 5, 2))
            # full = torch.cat([id, mode, pos, dir], dim=1)
            # actor_features = self.attention(full, att_mode)
            if att_mode == 0:
                obs[:, 18:] = -1
            elif att_mode == 1:
                obs[:, 22:] = -1
            actor_features = self.base(obs)
        else:
            actor_features = self.base(obs)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        actions, action_log_probs = self.act(actor_features, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, att_mode=2, return_rnn=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self.algo == 'hsp':
            # id = self.id(obs[:, :5]).unsqueeze(1)
            # mode = self.mode(obs[:, 5:12]).unsqueeze(1)
            # pos = self.pos(obs[:, 12:22].reshape(-1, 5, 2))
            # dir = self.dir(obs[:, 22:].reshape(-1, 5, 2))
            # full = torch.cat([id, mode, pos, dir], dim=1)
            # actor_features = self.attention(full, att_mode)
            if att_mode == 0:
                obs[:, 18:] = -1
            elif att_mode == 1:
                obs[:, 22:] = -1
            actor_features = self.base(obs)
        else:
            actor_features = self.base(obs)

        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action)
        if return_rnn:
            return action_log_probs, dist_entropy, rnn_states
        else:
            return action_log_probs, dist_entropy
