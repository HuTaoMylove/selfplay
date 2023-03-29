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

e = get_test_env()
o, so = e.reset()
print(o)
