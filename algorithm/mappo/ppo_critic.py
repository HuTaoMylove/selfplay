import torch
import torch.nn as nn
import gym
from ..utils.mlp import MLPBase, MLPLayer
from ..utils.gru import GRULayer
from ..utils.utils import check, init
from ..utils.conv import ConvBase
import numpy as np
from algorithm.utils.selfattention import SelfAttention


class PPOCritic(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super(PPOCritic, self).__init__()
        # network config

        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id

        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.hidden_size = args.hidden_size
        self.algo = args.selfplay_algorithm
        # (1) feature extraction module
        # self.base = ConvBase(observation_space=obs_space, output_size=256)

        if self.algo == 'hsp':
            self.id = nn.Linear(3, 96)
            self.mode = nn.Linear(7, 96)
            self.pos = nn.Linear(2, 96)
            self.dir = nn.Linear(2, 96)

            self.attention = SelfAttention(96, 96, 96)
            input_size = 96
        else:
            self.base = MLPBase(obs_space, self.hidden_size, self.activation_id)
            input_size = self.base.output_size

        self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
        input_size = self.rnn.output_size
        if len(self.act_hidden_size) > 0:
            self.mlp = MLPLayer(input_size, self.act_hidden_size, self.activation_id)
        input_size = self.mlp.output_size
        # self.value_out = self.init_(nn.Linear(input_size, 1))
        self.value_out = nn.Linear(input_size, 1)
        self.to(device)

    def init_(self, m):
        return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), 0.01)

    def forward(self, obs, rnn_states, masks, att_mode=0):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if self.algo == 'hsp':
            id = self.id(obs[:, :3]).unsqueeze(1)
            mode = self.mode(obs[:, 3:10]).unsqueeze(1)
            pos = self.pos(obs[:, 10:20].reshape(-1, 5, 2))
            dir = self.dir(obs[:, 20:].reshape(-1, 5, 2))
            full = torch.cat([id, mode, pos, dir], dim=1)
            critic_features = self.attention(full, att_mode)
        else:
            critic_features = self.base(obs)

        critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        if len(self.act_hidden_size) > 0:
            critic_features = self.mlp(critic_features)

        values = self.value_out(critic_features)

        return values, rnn_states
