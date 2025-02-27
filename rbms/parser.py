import argparse
from typing import Any

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
        default="RBM.h5",
        help="(Defaults to RBM.h5). Path to the file where to save the model or load if training is restored.",
    )
    save_args.add_argument(
        "--n_save",
        type=int,
        default=50,
        help="(Defaults to 50). Number of models to save during the training.",
    )
    save_args.add_argument(
        "--acc_ptt",
        type=float,
        default=0.25,
        help="(Defaults to 0.25). Minimum PTT acceptance to save configurations for ptt file.",
    )
    save_args.add_argument(
        "--acc_ll",
        type=float,
        default=0.7,
        help="(Defaults to 0.75). Minimum PTT acceptance to save configurations for ll file.",
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


def add_args_rbm(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add an argument group to the parser for the general hyperparameters of a RBM

    Args:
        parser (argparse.ArgumentParser): argparse.ArgumentParser:
    """
    rbm_args = parser.add_argument_group("RBM")
    rbm_args.add_argument(
        "--num_hiddens",
        type=int,
        default=100,
        help="(Defaults to 100). Number of hidden units.",
    )
    rbm_args.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="(Defaults to 2000). Minibatch size.",
    )
    rbm_args.add_argument(
        "--gibbs_steps",
        type=int,
        default=100,
        help="(Defaults to 100). Number of gibbs steps to perform for each gradient update.",
    )
    rbm_args.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="(Defaults to 0.01). Learning rate.",
    )
    rbm_args.add_argument(
        "--num_chains",
        type=int,
        default=2000,
        help="(Defaults to 2000). Number of parallel chains.",
    )
    rbm_args.add_argument(
        "--num_updates",
        default=10_000,
        type=int,
        help="(Defaults to 10 000). Number of gradient updates to perform.",
    )
    rbm_args.add_argument(
        "--beta",
        default=1.0,
        type=float,
        help="(Defaults to 1.0). The inverse temperature of the RBM",
    )
    rbm_args.add_argument(
        "--restore",
        default=False,
        action="store_true",
        help="(Defaults to False). Restore the training",
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
