"""A simple example of setting up a multi-agent version of GFootball with rllib.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Dict, Any, Tuple
import argparse
import gfootball.env as football_env
import gym
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
import numpy as np
from arguments import get_args


class FootballEnv:
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def create_single_football_env(self, args):
        assert args.num_agent == args.left_agent + args.right_agent
        env = football_env.create_environment(
            env_name=args.env_name,
            representation='simple115',
            stacked=False,
            rewards=args.rewards,
            write_goal_dumps=False, write_full_episode_dumps=False,
            render=args.render,
            dump_frequency=0,
            number_of_left_players_agent_controls=args.left_agent,
            number_of_right_players_agent_controls=args.right_agent,
        )
        return env

    def n_agents(self):
        return self.num_agents

    def __init__(self, args):
        self.env = self.create_single_football_env(args)
        self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
        self.num_agents = args.num_agent
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            dtype=float)
        self.share_observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[0][:-2],
            high=self.env.observation_space.high[0][:-2],
            dtype=float)

    def reset(self):
        o = self.env.reset()
        share_o = o[:, 2:]
        return o, share_o

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        o, r, d, i = self.env.step(action.reshape(self.num_agents, ))
        share_o = o[:, 2:]
        return o, share_o, r, d, i

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)


class TestEnv:
    def create_single_football_env(self, name):
        env = football_env.create_environment(
            env_name=name,
            stacked=False,
            write_goal_dumps=False, write_full_episode_dumps=False,
            dump_frequency=0,
            representation='simple115',
            number_of_left_players_agent_controls=2,
            number_of_right_players_agent_controls=0,
        )
        return env

    def __init__(self, name):
        self.env = self.create_single_football_env(name)
        self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
        self.num_agents = 2
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            dtype=self.env.observation_space.dtype)
        self.share_observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[0][:-2],
            high=self.env.observation_space.high[0][:-2],
            dtype=float)

    def reset(self):
        o = self.env.reset()
        share_o = o[:, 2:]
        return o, share_o

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        o, r, d, i = self.env.step(action.reshape(self.num_agents, ))
        share_o = o[:, 2:]
        return o, share_o, r, d, i

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)


def get_env(args):
    def get_env_fn(rank):
        def init_env():
            env = FootballEnv(args)
            env.seed(args.seed + rank * 1000)
            return env

        return init_env

    if args.n_rollout == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout)])


import copy


def get_eval_env(arg):
    args = copy.copy(arg)
    args.rewards = 'scoring'

    def get_env_fn(rank):
        def init_env():
            env = FootballEnv(args)
            env.seed(args.seed * 5000 + rank * 1000)
            return env

        return init_env

    if args.n_eval_rollout == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(args.n_eval_rollout)])


def get_test_env(name=None):
    if name is None:
        name = ['test_1', 'test_2', 'test_3', 'test_4', 'test_5']

    def get_env_fn(names):
        def init_env():
            env = TestEnv(names)
            env.seed(1000)
            return env

        return init_env

    if len(name) == 1:
        return ShareDummyVecEnv([get_env_fn(name[0])])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in name])
