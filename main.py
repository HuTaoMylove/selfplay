from arguments import get_args
from selfplay_runner import Runner
import gfootball.env as football_env
from gfootball_net import NNetWrapper as nn
from utils import *
from multiagent_setup import FootballEnv,get_env
import numpy as np
import torch
import sys
import os
import traceback
import wandb
import socket
import torch
import random
import logging
import numpy as np
from pathlib import Path
import setproctitle

if __name__ == '__main__':
    all_args = get_args()
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    envs = get_env(all_args)

    run_dir = Path("./results") \
              / all_args.env_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    log_dir = Path("./log") / all_args.env_name
    if not log_dir.exists():
        os.makedirs(str(log_dir))

    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")  # use cude mask to control using which GPU
        torch.cuda.empty_cache()
        torch.set_num_threads(all_args.n_training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training)

    config = {
        "all_args": all_args,
        "envs": envs,
        "device": device,
        "run_dir": run_dir,
        "log_dir": log_dir
    }

    runner = Runner(config)
    try:
        runner.run()
    except BaseException:
        traceback.print_exc()
    finally:
        # post process
        envs.close()


    nnet = nn(env, args)
    c = Coach(env, nnet, args)
    vloss_hist, ploss_hist = c.learn()
    vloss_hist = np.array(vloss_hist)
    ploss_hist = np.array(ploss_hist)
    np.save('ploss_hist.npy', ploss_hist)
    np.save('vloss_hist.npy', vloss_hist)
    env.close()

    # if is_train == False:
    #     model_path = '11_vs_11_competition.pth'
    #     args = get_args()
    #     args.render = False  # change to True if you want to see the play
    #     args.left_agent = 2
    #     args.right_agent = 2
    #     args.num_agent = 4
    #     env = RllibGFootball(args)
    #     # network = nn(env, args)
    #     # network.load_state_dict(model_path)
    #     obs = env.reset()
    #     for _ in range(int(1e4)):
    #         action = {}
    #         for i in range(args.num_agent):
    #             # with torch.no_grad():
    #             #     pi, _ = network.predict(obs['agent_{}'.format(i)])
    #             # action['agent_{}'.format(i)] = np.random.choice(len(pi), p=pi)
    #             action['agent_{}'.format(i)] = int(np.random.rand(1) * 19)
    #         obs, reward, done, _ = env.step(action)
    #         if done:
    #             obs = env.reset()
    #     env.close()
