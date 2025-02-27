import argparse

import torch

from rbms.dataset import load_dataset
from rbms.dataset.parser import add_args_dataset
from rbms.map_model import map_model
from rbms.parser import (
    add_args_pytorch,
    add_args_rbm,
    add_args_saves,
    match_args_dtype,
    remove_argument,
)
from rbms.training.pcd import train
from rbms.training.utils import get_checkpoints


def create_parser():
    parser = argparse.ArgumentParser(description="Train a Restricted Boltzmann Machine")
    parser = add_args_dataset(parser)
    parser = add_args_rbm(parser)
    parser = add_args_saves(parser)
    parser = add_args_pytorch(parser)
    remove_argument(parser, "use_torch")
    return parser


def train_rbm(args: dict):
    checkpoints = get_checkpoints(
        num_updates=args["num_updates"], n_save=args["n_save"], spacing=args["spacing"]
    )
    train_dataset, test_dataset = load_dataset(
        dataset_name=args["data"],
        subset_labels=args["subset_labels"],
        use_weights=args["use_weights"],
        alphabet=args["alphabet"],
        binarize=args["binarize"],
        train_size=args["train_size"],
        test_size=args["test_size"],
        device=args["device"],
        dtype=args["dtype"],
    )
    print(train_dataset)
    if train_dataset.is_binary:
        model_type = "BBRBM"
    else:
        model_type = "PBRBM"
    train(
        dataset=train_dataset,
        test_dataset=test_dataset,
        model_type=model_type,
        args=args,
        dtype=args["dtype"],
        checkpoints=checkpoints,
        map_model=map_model,
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
