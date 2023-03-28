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
from env_wrappers import SubprocVecEnv, DummyVecEnv
import numpy as np
from arguments import get_args

parser = argparse.ArgumentParser()

parser.add_argument('--num-agents', type=int, default=3)
parser.add_argument('--num-policies', type=int, default=3)
parser.add_argument('--num-iters', type=int, default=100000)
parser.add_argument('--simple', action='store_true')


# class RllibGFootball(MultiAgentEnv):


class FootballEnv:
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def create_single_football_env(self, args):
        env = football_env.create_environment(
            env_name=args.env_name, stacked=True,
            logdir='/tmp/rllib_test',
            write_goal_dumps=False, write_full_episode_dumps=False, render=args.render,
            dump_frequency=0,
            number_of_left_players_agent_controls=args.left_agent,
            number_of_right_players_agent_controls=args.right_agent,
            channel_dimensions=(42, 42))
        return env

    def __init__(self, args):
        self.env = self.create_single_football_env(args)
        self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            dtype=self.env.observation_space.dtype)
        self.num_agents = args.num_agent

    def reset(self):
        return self.env.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        o, r, d, i = self.env.step(action.reshape(self.num_agents, ))
        return o, r, d, i

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

    return SubprocVecEnv([get_env_fn(i) for i in range(args.n_rollout)])


if __name__ == '__main__':
    args = get_args()
    # env = FootballEnv(args)
    env = get_env(args)
    # env = create_single_football_env(args)

    obs = env.reset()
    for _ in range(int(1e3)):
        flag = 0
        action = np.random.randint(0, env.action_space.n, 8).reshape(2, 4)
        # action_dict = {}
        # for i in range(args.num_agent):
        #     action_dict["agent_{}".format(i)] = action[i]
        obs_next, reward, done, info = env.step(action)
        if done['__all__']:
            print("done")
            obs_next = env.reset()
        for i in range(args.num_agent):
            if not np.array_equal(obs["agent_{}".format(i)], obs_next["agent_{}".format(i)]):
                print('obs changes')
                print(np.sum(obs["agent_{}".format(i)] - obs_next["agent_{}".format(i)]))
                print(obs["agent_{}".format(i)].shape)
        # if not np.array_equal(obs, obs_next):
        #     flag += 1
        obs = obs_next
