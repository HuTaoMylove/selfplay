import time

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
import gym
from algorithm.mappo.ppo_actor import PPOActor

args = get_args()


def _t2n(x):
    return x.detach().cpu().numpy()


class TestEnv:
    def create_single_football_env(self, name):
        env = football_env.create_environment(
            env_name=name,
            rewards='scoring',
            stacked=False,
            write_goal_dumps=False, write_full_episode_dumps=False,
            dump_frequency=0,
            render=True,
            representation='simple115',
            number_of_left_players_agent_controls=2,
            number_of_right_players_agent_controls=0,
        )
        return env

    def __init__(self, name):
        self.env = self.create_single_football_env(name)
        self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
        self.num_agents = 8
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            dtype=self.env.observation_space.dtype)
        self.share_observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[0][:-4],
            high=self.env.observation_space.high[0][:-4],
            dtype=float)

    def reset(self):
        o = self.env.reset()
        share_o = np.concatenate([np.expand_dims(o[0][:-4], 0)] * self.num_agents)
        return o, share_o

    def step(self, action: np.ndarray):
        o, r, d, i = self.env.step(action.reshape(self.num_agents, ))
        share_o = np.concatenate([np.expand_dims(o[0][:-4], 0)] * self.num_agents)
        return o, share_o, r, d, i

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)


env = TestEnv('5_vs_5')
model = PPOActor(args, env.observation_space, env.action_space)
model.load_state_dict(torch.load('results/5_vs_5/sp' + '/actor_30.pt'))
obs, _ = env.reset()
obs = np.expand_dims(obs, 0)
masks = np.ones((1, args.num_agent//2, 1), dtype=np.float32)
rnn_states = np.zeros((1, args.num_agent//2, 1, 128),
                      dtype=np.float32)

t = 0
rs = np.zeros(2)
with torch.no_grad():
    while t < 3001:
        actions, _, rnn_states = model.forward(np.concatenate(obs),
                                               np.concatenate(rnn_states),
                                               np.concatenate(masks), deterministic=False)
        actions = np.array(np.split(_t2n(actions), 1))

        # actions = np.random.randint(0, 19, [4])
        if t % 100 == 0:
            print(t)
        t += 1
        rnn_states = np.array(np.split(_t2n(rnn_states), 1))
        obs, _, r, done, eval_infos = env.step(actions)
        rs += r

        obs = np.expand_dims(obs, 0)
        if done:
            break
    print(rs)
