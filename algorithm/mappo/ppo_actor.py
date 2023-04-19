import gym
import numpy as np
import torch
import torch.nn as nn

from ..utils.mlp import MLPBase
from ..utils.gru import GRULayer
from ..utils.act import ACTLayer
from ..utils.utils import check
from ..utils.conv import ConvBase


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(PPOActor, self).__init__()
        # network config
        self.gain = args.gain
        self.obs_version = args.obs_version
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.args = args
        self.tpdv = dict(dtype=torch.float32, device=device)

        if self.obs_version == 'v1':
            # leftpos ballpos
            obs_space = gym.spaces.Box(low=np.array([-np.inf] * 19), high=np.array([np.inf] * 19), dtype=float)
        elif self.obs_version == 'v2':
            # leftpos rightpos ballpos
            obs_space = gym.spaces.Box(low=np.array([-np.inf] * 27), high=np.array([np.inf] * 27), dtype=float)
        elif self.obs_version == 'v3':
            # leftpos leftdir rightpos ballpos balldir
            obs_space = gym.spaces.Box(low=np.array([-np.inf] * 30), high=np.array([np.inf] * 30), dtype=float)
        elif self.obs_version == 'v4':
            # leftpos leftdir rightpos rightdir ballpos balldir
            obs_space = gym.spaces.Box(low=np.array([-np.inf] * 42), high=np.array([np.inf] * 42), dtype=float)
        elif self.obs_version == 'v5':
            # leftpos leftdir rightpos rightdir ballpos balldir ballrotate busy (full obs)
            obs_space = gym.spaces.Box(low=np.array([-np.inf] * 38), high=np.array([np.inf] * 38), dtype=float)

        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id)
        input_size = self.base.output_size
        self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
        input_size = self.rnn.output_size
        # (3) act module
        self.act = ACTLayer(act_space, input_size, self.act_hidden_size, self.activation_id, self.gain)
        self.to(device)

    def forward(self, obs, rnn_states, masks, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self.obs_version == 'v1':
            # leftpos ballpos
            obs = torch.cat([obs[..., :14], obs[..., 26:29], obs[..., -2:]], dim=-1)
        elif self.obs_version == 'v2':
            # leftpos rightpos ballpos
            obs = torch.cat([obs[..., :22], obs[..., 34:37], obs[..., -2:]], dim=-1)
        elif self.obs_version == 'v3':
            # leftpos leftdir rightpos ballpos balldir
            obs = torch.cat([obs[..., :22], obs[..., 26:32], obs[..., -2:]], dim=-1)
        elif self.obs_version == 'v4':
            # leftpos leftdir rightpos rightdir ballpos balldir
            obs = torch.cat([obs[..., :40], obs[..., -2:]], dim=-1)

        actor_features = self.base(obs)
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        actions, action_log_probs = self.act(actor_features, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, return_rnn=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if self.obs_version == 'v1':
            # leftpos ballpos
            obs = torch.cat([obs[..., :14], obs[..., 26:29], obs[..., -2:]], dim=-1)
        elif self.obs_version == 'v2':
            # leftpos rightpos ballpos
            obs = torch.cat([obs[..., :22], obs[..., 34:37], obs[..., -2:]], dim=-1)
        elif self.obs_version == 'v3':
            # leftpos leftdir rightpos ballpos balldir
            obs = torch.cat([obs[..., :22], obs[..., 26:32], obs[..., -2:]], dim=-1)
        elif self.obs_version == 'v4':
            # leftpos leftdir rightpos rightdir ballpos balldir
            obs = torch.cat([obs[..., :40], obs[..., -2:]], dim=-1)
        actor_features = self.base(obs)

        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action)
        if return_rnn:
            return action_log_probs, dist_entropy, rnn_states
        else:
            return action_log_probs, dist_entropy
