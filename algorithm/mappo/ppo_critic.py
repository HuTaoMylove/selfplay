import torch
import torch.nn as nn
import gym
from ..utils.mlp import MLPBase, MLPLayer
from ..utils.gru import GRULayer
from ..utils.utils import check
from ..utils.conv import ConvBase
import numpy as np

class PPOCritic(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super(PPOCritic, self).__init__()
        # network config

        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id

        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.hidden_size=args.hidden_size
        # (1) feature extraction module
        # self.base = ConvBase(observation_space=obs_space, output_size=256)
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id)
        input_size = self.base.output_size
        self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
        input_size = self.rnn.output_size
        if len(self.act_hidden_size) > 0:
            self.mlp = MLPLayer(input_size, self.act_hidden_size, self.activation_id)
        input_size = self.mlp.output_size
        self.value_out = nn.Linear(input_size, 1)
        self.to(device)

    def forward(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(obs)

        critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        if len(self.act_hidden_size) > 0:
            critic_features = self.mlp(critic_features)

        values = self.value_out(critic_features)

        return values, rnn_states
