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
            stacked=False,
            logdir='/tmp/rllib_test',
            rewards=args.rewards,
            write_goal_dumps=False, write_full_episode_dumps=False,
            render=args.render,
            dump_frequency=0,
            number_of_left_players_agent_controls=args.left_agent,
            number_of_right_players_agent_controls=args.right_agent,
            channel_dimensions=(42, 42))
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
            dtype=self.env.observation_space.dtype)
        share_obs_shape = np.array(self.observation_space.shape)
        share_obs_shape[-1] = share_obs_shape[-1] + self.num_agents // 2 - 1
        self.share_observation_space = gym.spaces.Box(
            low=np.zeros(share_obs_shape),
            high=np.ones(share_obs_shape) * 255,
            dtype=self.env.observation_space.dtype)

    def reset(self):
        o = self.env.reset()
        half = int(self.num_agents // 2)
        left_half = np.concatenate([o[0], np.concatenate(np.expand_dims(o[1:half, ..., -1], -1), -1)], axis=-1)
        left_half = np.expand_dims(left_half, axis=0)
        left_half = np.concatenate([left_half] * half, axis=0)
        right_half = np.concatenate([o[half], np.concatenate(np.expand_dims(o[half + 1:half * 2, ..., -1], -1), -1)],
                                    axis=-1)
        right_half = np.expand_dims(right_half, axis=0)
        right_half = np.concatenate([right_half] * half, axis=0)
        share_o = np.concatenate([left_half, right_half], axis=0)
        return o, share_o

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        o, r, d, i = self.env.step(action.reshape(self.num_agents, ))
        half = int(self.num_agents // 2)
        left_half = np.concatenate([o[0], np.concatenate(np.expand_dims(o[1:half, ..., -1], -1), -1)], axis=-1)
        left_half = np.expand_dims(left_half, axis=0)
        left_half = np.concatenate([left_half] * half, axis=0)
        right_half = np.concatenate([o[half], np.concatenate(np.expand_dims(o[half + 1:half * 2, ..., -1], -1), -1)],
                                    axis=-1)
        right_half = np.expand_dims(right_half, axis=0)
        right_half = np.concatenate([right_half] * half, axis=0)
        share_o = np.concatenate([left_half, right_half], axis=0)
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


def get_eval_env(args):
    def get_env_fn(rank):
        def init_env():
            env = FootballEnv(args)
            env.seed(args.seed*5000 + rank * 1000)
            return env

        return init_env

    if args.n_eval_rollout == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(args.n_eval_rollout)])

def get_test_env(args):
    def get_env_fn(rank):
        def init_env():
            env = FootballEnv(args)
            env.seed(args.seed*5000 + rank * 1000)
            return env

        return init_env

    return ShareSubprocVecEnv([get_env_fn(i) for i in range(5)])
