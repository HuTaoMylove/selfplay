from arguments import get_args
from selfplay_runner import ShareRunner
import gfootball.env as football_env
# from utils import *
from multiagent_setup import get_env, get_eval_env, get_test_env
import os
import traceback
import torch
import random
import logging
import numpy as np
from pathlib import Path
import copy
from multiagent_setup import FootballEnv

args=get_args()


class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '64 64'
        self.act_hidden_size = '64 64'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 64
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = False
        self.lr = 0.001
        self.obs_version = 'v1'

from algorithm.mappo.ppo_trainer import PPOTrainer as Trainer
from algorithm.mappo.ppo_policy import PPOPolicy as Policy

env=FootballEnv(args)
policy = Policy(Args(), env.observation_space , env.share_observation_space,env.action_space)



def change(a):
    b=copy.copy(a)
    b.rewards='asdfasdf'
change(a)
print(a.rewards)