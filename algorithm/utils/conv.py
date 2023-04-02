import sys

import gym

sys.path.append('..')

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ConvBase(nn.Module):
    def __init__(self, observation_space, output_size=256):
        super(ConvBase, self).__init__()
        self.obs_width, self.obs_height, self.obs_channel = observation_space.shape
        self.action_size = output_size
        self.conv1 = nn.Conv2d(self.obs_channel, self.obs_channel * 2, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.obs_channel * 2, self.obs_channel * 3, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(self.obs_channel * 3, self.obs_channel * 3, 3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(self.obs_channel * 2)
        self.bn2 = nn.BatchNorm2d(self.obs_channel * 3)

        obs_channel = self.obs_channel * 3
        obs_width = int(((self.obs_width + 1) // 2 + 1) // 2)
        obs_height = int(((self.obs_height + 1) // 2 + 1) // 2)
        self.fc1 = nn.Linear(obs_channel * obs_height * obs_width, 256)
        self.fc_bn1 = nn.LayerNorm(256)

        self.fc2 = nn.Linear(256, self.action_size)
        self.flatten = nn.Flatten()

    def forward(self, s):
        # s: batch_size x board_x x board_y
        s = s.view(-1, self.obs_channel, self.obs_width, self.obs_height)
        s = F.relu(self.bn1(self.conv1(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.conv3(s))  # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = self.flatten(s)
        s = F.relu(self.fc_bn1(self.fc1(s)))
        return self.fc2(s)


if __name__ == '__main__':
    o = gym.spaces.Box(np.zeros([22, 22, 4]), np.zeros([22, 22, 4]))
    a = gym.spaces.Discrete(19)
    net = ConvBase(o, 256)
    input = torch.arange(42 * 42 * 4 * 10, dtype=torch.float32).reshape(-1, 42, 42, 4)
    out = net(input)
