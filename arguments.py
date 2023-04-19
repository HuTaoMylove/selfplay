import argparse


def get_args():
    parse = argparse.ArgumentParser()

    """
    env para
    """
    parse.add_argument('--env-name', type=str, default='5_vs_5')
    parse.add_argument('--num-env-steps', type=int, default=3e8, help='the steps to collect samples')
    parse.add_argument('--cuda', action='store_true', default=False, help='use cuda do the training')
    parse.add_argument('--num-agent', type=int, default=4)
    parse.add_argument('--left-agent', type=int, default=2)
    parse.add_argument('--right-agent', type=int, default=2)
    parse.add_argument('--rewards', type=str, default='scoring,checkpoints')
    parse.add_argument('--render', type=bool, default=False, help='show nr not')

    """
    selfplay para
    """
    parse.add_argument('--n-rollout', type=int, default=5, help='the number of para env')
    parse.add_argument("--selfplay-algorithm", type=str, default='hsp', choices=["hsp", "fsp", "pfsp"],
                       help="Specifiy the selfplay algorithm (default 'sp')")
    parse.add_argument('--n-choose-opponents', type=int, default=5,
                       help="number of different opponents chosen for rollout. (default 1)")
    parse.add_argument('--init-elo', type=float, default=1000.0,
                       help="initial ELO for policy performance. (default 1000.0)")

    """
    train para
    """

    parse.add_argument("--buffer-size", type=int, default=2000,
                       help="maximum storage in the buffer.")
    parse.add_argument("--data-chunk-length", type=int, default=8,
                       help="Time length of chunks used to train a recurrent_policy (default 16)")
    parse.add_argument("--num-mini-batch", type=int, default=4,
                       help='number of batches for ppo (default: 1)')
    parse.add_argument("--gamma", type=float, default=0.99,
                       help='discount factor for rewards (default: 0.99)')
    parse.add_argument("--ppo-epoch", type=int, default=4,
                       help='number of ppo epochs (default: 10)')
    parse.add_argument('--lr', type=float, default=0.001, help='learning rate of the algorithm')

    parse.add_argument("--use-gae", action='store_false', default=True,
                       help='Whether to use generalized advantage estimation')
    parse.add_argument("--gae-lambda", type=float, default=0.95,
                       help='gae lambda parameter (default: 0.95)')
    parse.add_argument("--clip-param", type=float, default=0.2,
                       help='ppo clip parameter (default: 0.2)')
    parse.add_argument("--use-clipped-value-loss", action='store_true', default=False,
                       help="By default false. If set, clip value loss.")
    parse.add_argument("--value-loss-coef", type=float, default=1,
                       help='ppo value loss coefficient (default: 1)')
    parse.add_argument("--entropy-coef", type=float, default=0.01,
                       help='entropy term coefficient (default: 0.01)')
    parse.add_argument("--use-max-grad-norm", action='store_false', default=True,
                       help="By default, use max norm of gradients. If set, do not use.")
    parse.add_argument("--max-grad-norm", type=float, default=2,
                       help='max norm of gradients (default: 2)')

    """
    eval para
    """

    parse.add_argument("--use-eval", action='store_true', default=False,
                       help="by default, do not start evaluation. If set, start evaluation alongside with training.")
    # parse.add_argument("--num-opponents", type=int, default=3,
    #                    help="Number of parallel envs for evaluating rollout (default 1)")
    parse.add_argument("--n-eval-rollout", type=int, default=4,
                       help="Number of parallel envs for evaluating rollout (default 1)")
    parse.add_argument("--eval-episodes", type=int, default=20,
                       help="number of episodes of a single evaluation. (default 32)")

    """
    
    test para
    
    """
    parse.add_argument("--n-test-rollout", type=int, default=5,
                       help="number of episodes of a single evaluation. (default 32)")
    parse.add_argument("--test-episodes", type=int, default=4,
                       help="number of episodes of a single evaluation. (default 32)")

    """
    interval para
    """

    parse.add_argument('--save-interval', type=int, default=5, help='the number of save')
    parse.add_argument('--log-interval', type=int, default=1, help='the number of log')
    parse.add_argument("--eval-interval", type=int, default=1, help="time duration between contiunous twice "
                                                                    "evaluation progress. (default 25)")
    parse.add_argument("--test-interval", type=int, default=5, help="time duration between contiunous twice "
                                                                    "evaluation progress. (default 25)")
    parse.add_argument('--seed', type=int, default=123, help='the random seeds')
    parse.add_argument('--n-training', type=int, default=4, help='the number of workers to collect samples')

    # args = parse.parse_args()
    args, unknown = parse.parse_known_args()

    return args
