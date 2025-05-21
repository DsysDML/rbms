import argparse

import torch

from rbms.dataset import load_dataset
from rbms.dataset.parser import add_args_dataset
from rbms.map_model import map_model
from rbms.parser import (
    add_args_pytorch,
    add_args_rbm,
    add_args_saves,
    add_args_regularization,
    match_args_dtype,
    remove_argument,
    default_args,
)
from rbms.training.pcd import train
from rbms.training.utils import get_checkpoints


def create_parser():
    parser = argparse.ArgumentParser(description="Train a Restricted Boltzmann Machine")
    parser = add_args_dataset(parser)
    parser = add_args_rbm(parser)
    parser = add_args_regularization(parser)
    parser = add_args_saves(parser)
    parser = add_args_pytorch(parser)
    remove_argument(parser, "use_torch")
    return parser


def train_rbm(args: dict):
    if args["num_updates"] is None:
        args["num_updates"] = default_args["num_updates"]
    checkpoints = get_checkpoints(
        num_updates=args["num_updates"], n_save=args["n_save"], spacing=args["spacing"]
    )
    train_dataset, test_dataset = load_dataset(
        dataset_name=args["dataset"],
        test_dataset_name=args["test_dataset"],
        subset_labels=args["subset_labels"],
        use_weights=args["use_weights"],
        alphabet=args["alphabet"],
        binarize=args["binarize"],
        device=args["device"],
        dtype=args["dtype"],
    )
    print(train_dataset)
    if train_dataset.is_binary:
        model_type = "BBRBM"
    else:
        model_type = "PBRBM"
    train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_type=model_type,
        args=args,
        dtype=args["dtype"],
        checkpoints=checkpoints,
        map_model=map_model,
        default_args=default_args,
    )


def main():
    torch.backends.cudnn.benchmark = True
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)
    args = match_args_dtype(args)
    train_rbm(args=args)


if __name__ == "__main__":
    main()
