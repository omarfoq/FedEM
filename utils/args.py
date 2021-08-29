import os
import argparse


def args_to_string(args):
    """
    Transform experiment's arguments into a string
    :param args:
    :return: string
    """
    if args.decentralized:
        return f"{args.experiment}_decentralized"

    args_string = ""

    args_to_show = ["experiment", "method"]
    for arg in args_to_show:
        args_string = os.path.join(args_string, str(getattr(args, arg)))

    if args.locally_tune_clients:
        args_string += "_adapt"

    return args_string


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'experiment',
        help='name of experiment',
        type=str
    )
    parser.add_argument(
        'method',
        help='the method to be used;'
             ' possible are `FedAvg`, `FedEM`, `local`, `FedProx`, `L2SGD`,'
             ' `pFedMe`, `AFL`, `FFL` and `clustered`;',
        type=str
    )
    parser.add_argument(
        '--decentralized',
        help='if chosen decentralized version is used,'
             'client are connected via an erdos-renyi graph of parameter p=0.5,'
             'the mixing matrix is obtained via FMMC (Fast Mixin Markov Chain),'
             'see https://web.stanford.edu/~boyd/papers/pdf/fmmc.pdf);'
             'can be combined with `method=FedEM`, in that case it is equivalent to `D-EM`;'
             'can not be used when method is `AFL` or `FFL`, in that case a warning is raised'
             'and decentralized is set to `False`;'
             'in all other cases D-SGD is used;',
        action='store_true'
    )
    parser.add_argument(
        '--sampling_rate',
        help='proportion of clients to be used at each round; default is 1.0',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--input_dimension',
        help='the dimension of one input sample; only used for synthetic dataset',
        type=int,
        default=None
    )
    parser.add_argument(
        '--output_dimension',
        help='the dimension of output space; only used for synthetic dataset',
        type=int,
        default=None
    )
    parser.add_argument(
        '--n_learners',
        help='number of learners_ensemble to be used with `FedEM`; ignored if method is not `FedEM`; default is 3',
        type=int,
        default=3
    )
    parser.add_argument(
        '--n_rounds',
        help='number of communication rounds; default is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--bz',
        help='batch_size; default is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--local_steps',
        help='number of local steps before communication; default is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--log_freq',
        help='frequency of writing logs; defaults is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--device',
        help='device to use, either cpu or cuda; default is cpu',
        type=str,
        default="cpu"
    )
    parser.add_argument(
        '--optimizer',
        help='optimizer to be used for the training; default is sgd',
        type=str,
        default="sgd"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help='learning rate; default is 1e-3',
        default=1e-3
    )
    parser.add_argument(
        "--lr_lambda",
        type=float,
        help='learning rate for clients weights; only used for agnostic FL; default is 0.',
        default=0.
    )
    parser.add_argument(
        "--lr_scheduler",
        help='learning rate decay scheme to be used;'
             ' possible are "sqrt", "linear", "cosine_annealing", "multi_step" and "constant" (no learning rate decay);'
             'default is "constant"',
        type=str,
        default="constant"
    )
    parser.add_argument(
        "--mu",
        help='proximal / penalty term weight, used when --optimizer=`prox_sgd` also used with L2SGD; default is `0.`',
        type=float,
        default=0
    )
    parser.add_argument(
        "--communication_probability",
        help='communication probability, only used with L2SGD',
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--q",
        help='fairness hyper-parameter, ony used for FFL client; default is 1.',
        type=float,
        default=1.
    )
    parser.add_argument(
        "--locally_tune_clients",
        help='if selected, clients are tuned locally for one epoch before writing logs;',
        action='store_true'
    )
    parser.add_argument(
        '--validation',
        help='if chosen the validation part will be used instead of test part;'
             ' make sure to use `val_frac > 0` in `generate_data.py`;',
        action='store_true'
    )
    parser.add_argument(
        "--verbose",
        help='verbosity level, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`;',
        type=int,
        default=0
    )
    parser.add_argument(
        "--logs_root",
        help='root path to write logs; if not passed, it is set using arguments',
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--save_path",
        help='directory to save checkpoints once the training is over; if not specified checkpoints are not saved',
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--seed",
        help='random seed',
        type=int,
        default=1234
    )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    return args
