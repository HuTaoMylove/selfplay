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

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    all_args = get_args()

    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    envs = get_env(all_args)
    eval_envs = get_eval_env(all_args) if all_args.use_eval else None
    test_envs = get_test_env()

    run_dir = Path("./results") \
              / all_args.env_name / all_args.selfplay_algorithm
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    log_dir = Path("./log") / all_args.env_name / all_args.selfplay_algorithm
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

    import setproctitle
    setproctitle.setproctitle(str(all_args.selfplay_algorithm)
                              +"@" + str('yemingzhi'))
    config = {
        "all_args": all_args,
        "envs": envs,
        'eval_envs': eval_envs,
        'test_envs': test_envs,
        "device": device,
        "run_dir": run_dir,
        "log_dir": log_dir
    }

    runner = ShareRunner(config)
    try:
        runner.run()
    except BaseException:
        traceback.print_exc()
    finally:
        # post process
        envs.close()
