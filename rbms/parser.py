import argparse
from typing import Any

import numpy as np
import torch


def add_args_pytorch(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add an argument group to the parser for pytorch device and dtype

    Args:
        parser (argparse.ArgumentParser): argparse.ArgumentParser:
    """
    pytorch_args = parser.add_argument_group("PyTorch")
    pytorch_args.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="(Defaults to cuda). The device to use in PyTorch.",
    )
    pytorch_args.add_argument(
        "--dtype",
        type=str,
        choices=["int", "half", "float", "double"],
        default="float",
        help="(Defaults to float). The dtype to use in PyTorch.",
    )
    return parser


def add_args_saves(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add an argument group to the parser for the input-output during training

    Args:
        parser (argparse.ArgumentParser): argparse.ArgumentParser:
    """
    save_args = parser.add_argument_group("Save")
    save_args.add_argument(
        "-o",
        "--filename",
        type=str,
        default=None,
        help="(Defaults to RBM.h5). Path to the file where to save the model or load if training is restored.",
    )
    save_args.add_argument(
        "--n_save",
        type=int,
        default=50,
        help="(Defaults to 50). Number of models to save during the training.",
    )

    save_args.add_argument(
        "--spacing",
        type=str,
        default="exp",
        help="(Defaults to exp). Spacing to save models.",
        choices=["exp", "linear"],
    )
    save_args.add_argument(
        "--log", default=False, action="store_true", help="Log metrics during training."
    )
    save_args.add_argument(
        "--overwrite",
        default=True,
        action="store_true",
        help="(Defaults to False). Force overwrite of save file if it already exists without asking for confirmation.",
    )
    return parser


def add_args_init_rbm(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    rbm_args = parser.add_argument_group("RBM")
    rbm_args.add_argument(
        "--num_hiddens",
        type=int,
        default=None,
        help="(Defaults to 100). Number of hidden units.",
    )
    rbm_args.add_argument(
        "--num_chains",
        type=int,
        default=None,
        help="(Defaults to 2000). Number of parallel chains.",
    )
    rbm_args.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="(Defaults to None). Model to use. If None is provided, will be a RBM with the same visible type as the dataset and binary hiddens. If restore, this argument is ignored.",
    )
    rbm_args.add_argument(
    "--energy_type",
    type=str,
    default=None,
    choices=[
        "mlp",
        "mlp_no_w2",
        "mlp_silu_no_w2",
        "mlp_sigmoid_no_w2",
        "rbm",
        "gaussian",
    ],
    help="Energy type to use when model_type is BEBM or CEBM.",
    )
    return parser


def add_sampling_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    sampling_args = parser.add_argument_group("Sampling")
    sampling_args.add_argument(
        "--gibbs_steps",
        type=int,
        default=None,
        help="(Defaults to 100). Number of gibbs steps to perform for each gradient update.",
    )
    sampling_args.add_argument(
        "--beta",
        default=None,
        type=float,
        help="(Defaults to 1.0). The inverse temperature of the RBM",
    )
    return parser


def add_grad_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    grad_args = parser.add_argument_group("Gradient")
    grad_args.add_argument(
        "--L1",
        default=None,
        type=float,
        help="(Defaults to 0.0). Lambda parameter for the L1 regularization.",
    )
    grad_args.add_argument(
        "--L2",
        default=None,
        type=float,
        help="(Defaults to 0.0). Lambda parameter for the L2 regularization.",
    )
    grad_args.add_argument(
        "--no_center",
        default=False,
        action="store_true",
        help="(Defaults to False). Use the non-centered gradient.",
    )
    grad_args.add_argument(
        "--max_norm_grad",
        default=None,
        type=float,
        help="(Defaults to None). Maximum norm of the gradient before update.",
    )
    grad_args.add_argument(
        "--normalize_grad",
        default=False,
        action="store_true",
        help="(Defaults to False). Normalize the gradient before update.",
    )
    return parser


def add_args_train(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    train_args = parser.add_argument_group("Train")
    train_args.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="(Defaults to 2000). Minibatch size.",
    )
    train_args.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="(Defaults to 0.01). Learning rate.",
    )
    train_args.add_argument(
        "--num_updates",
        default=None,
        type=int,
        help="(Defaults to 10 000). Number of gradient updates to perform.",
    )
    train_args.add_argument(
        "--optim", default=None, type=str, help="(Defaults to sgd). Optimizer to use."
    )
    train_args.add_argument(
        "--mult_optim",
        action="store_true",
        default=False,
        help="(Defaults to False). Use a different optimizer for each param group.",
    )
    train_args.add_argument(
        "--training_type",
        type=str,
        default=None,
        help="(Defaults to 'pcd'). Type of the training, should be one of {'pcd', 'cd', 'rdm'}.",
    )
    train_args.add_argument(
        "--max_lr",
        type=float,
        default=None,
        help="(Defaults to 10). Maximum learning rate when adaptative learning rate is used.",
    )
    train_args.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Set it to scale learning rate with the number of variables of the system",
    )
    train_args.add_argument(
        "--restore",
        default=False,
        action="store_true",
        help="(Defaults to False). Restore the training",
    )
    train_args.add_argument(
        "--update",
        default=None,
        type=int,
        help="(Defaults to None). Update to restore from, if None the last is selected.",
    )
    return parser


def add_args_regularization(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    reg_args = parser.add_argument_group("Regularization")
    reg_args.add_argument(
        "--L1",
        default=None,
        type=float,
        help="(Defaults to 0.0). Lambda parameter for the L1 regularization.",
    )
    reg_args.add_argument(
        "--L2",
        default=None,
        type=float,
        help="(Defaults to 0.0). Lambda parameter for the L2 regularization.",
    )
    return parser


def remove_argument(parser, arg):
    """Args:
    parser
    arg
    """
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

    for action in parser._action_groups:
        for group_action in action._group_actions:
            opts = group_action.option_strings
            if (opts and opts[0] == arg) or group_action.dest == arg:
                action._group_actions.remove(group_action)
                return


def match_args_dtype(args: dict[str, Any]) -> dict[str, Any]:
    match args["dtype"]:
        case "int":
            args["dtype"] = torch.int64
        case "half":
            args["dtype"] = torch.float16
        case "float":
            args["dtype"] = torch.float32
        case "double":
            args["dtype"] = torch.float64
    return args


default_args: dict[str, Any] = {
    "filename": "RBM.h5",
    "n_save": 50,
    "acc_ptt": 0.25,
    "acc_ll": 0.7,
    "spacing": "exp",
    "log": True,
    "overwrite": True,
    "num_hiddens": 100,
    "batch_size": 2000,
    "gibbs_steps": 100,
    "learning_rate": 0.01,
    "num_chains": 2000,
    "num_updates": 10000,
    "beta": 1.0,
    "restore": False,
    "seed": np.random.randint(0, 1000000000000),
    "no_center": False,
    "L1": 0.0,
    "L2": 0.0,
    "max_norm_grad": -1,
    "optim": "sgd",
    "max_lr": 10,
    "training_type": "pcd",
    "energy_type": "mlp",
}


def set_args_default(
    args: dict[str, Any], default_args: dict[str, Any]
) -> dict[str, Any]:
    for k, v in args.items():
        if v is None and k in default_args.keys():
            args[k] = default_args[k]
    return args
