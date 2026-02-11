import argparse

import torch

from rbms.dataset import load_dataset
from rbms.dataset.parser import add_args_dataset
from rbms.map_model import map_model
from rbms.optim import setup_optim
from rbms.parser import (
    add_args_init_rbm,
    add_args_pytorch,
    add_args_saves,
    add_args_train,
    add_grad_args,
    add_sampling_args,
    default_args,
    match_args_dtype,
    remove_argument,
    set_args_default,
)
from rbms.training.pcd import train
from rbms.training.utils import get_checkpoints, init_training, restore_training


def create_parser():
    parser = argparse.ArgumentParser(description="Train a Restricted Boltzmann Machine")
    parser = add_args_dataset(parser)
    parser = add_args_init_rbm(parser)
    parser = add_args_train(parser)
    parser = add_sampling_args(parser)
    parser = add_grad_args(parser)
    parser = add_args_saves(parser)
    parser = add_args_pytorch(parser)
    remove_argument(parser, "use_torch")
    return parser


def process_args(args: dict):
    args_torch = {"device": args["device"], "dtype": args["dtype"]}
    args_dataset = {
        "dataset_name": args["dataset"],
        "test_dataset_name": args["test_dataset"],
        "train_size": args["train_size"],
        "test_size": args["test_size"],
        "subset_labels": args["subset_labels"],
        "use_weights": args["use_weights"],
        "alphabet": args["alphabet"],
        "remove_duplicates": args["remove_duplicates"],
        "seed": args["seed"],
    }
    key_args_grad = [
        "no_center",
        "normalize_grad",
        "max_norm_grad",
        "L1",
        "L2",
    ]
    key_args_sampling = [
        "gibbs_steps",
        "beta",
    ]
    key_args_train = [
        "optim",
        "learning_rate",
        "batch_size",
        "num_updates",
        "mult_optim",
        "training_type",
        "max_lr",
        "scale_lr",
    ]
    key_args_save = [
        "filename",
        "n_save",
        "spacing",
        "overwrite",
    ]
    key_args_init = [
        "num_chains",
        "num_hiddens",
        "model_type",
    ]

    args_grad = {}
    args_sampling = {}
    args_train = {}
    args_save = {}
    args_init = {}
    all_target = [args_grad, args_sampling, args_train, args_save, args_init]
    all_keys = [
        key_args_grad,
        key_args_sampling,
        key_args_train,
        key_args_save,
        key_args_init,
    ]
    for target, keys in zip(all_target, all_keys):
        for k in keys:
            target[k] = args[k]

    return (
        args_dataset,
        args_save,
        args_train,
        args_grad,
        args_sampling,
        args_torch,
        args_init,
    )


def main():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)
    args = set_args_default(args, default_args=default_args)
    args = match_args_dtype(args)
    (
        args_dataset,
        args_save,
        args_train,
        args_grad,
        args_sampling,
        args_torch,
        args_init,
    ) = process_args(args)
    checkpoints = get_checkpoints(
        num_updates=args_train["num_updates"],
        n_save=args_save["n_save"],
        spacing=args_save["spacing"],
    )
    train_dataset, test_dataset = load_dataset(
        dataset_name=args_dataset["dataset_name"],
        test_dataset_name=args_dataset["test_dataset_name"],
        subset_labels=args_dataset["subset_labels"],
        use_weights=args["use_weights"],
        alphabet=args["alphabet"],
        remove_duplicates=args["remove_duplicates"],
        **args_torch,
    )
    flags = ["checkpoint"]
    init_training(
        args_save,
        args_train,
        args_grad,
        args_sampling,
        args_init,
        args_dataset,
        args_torch,
        train_dataset,
        flags,
    )
    args_train["update"] = 1
    (
        params,
        parallel_chains,
        target_update,
        elapsed_time,
        train_dataset,
        test_dataset,
    ) = restore_training(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        args_save=args_save,
        args_train=args_train,
        args_dataset=args_dataset,
        args_torch=args_torch,
        map_model=map_model,
    )
    optimizer = setup_optim(args_train["optim"], args_train, params)
    train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        params=params,
        parallel_chains=parallel_chains,
        optimizer=optimizer,
        curr_update=target_update,
        elapsed_time=elapsed_time,
        checkpoints=checkpoints,
        args_save=args_save,
        args_train=args_train,
        args_grad=args_grad,
        args_sampling=args_sampling,
    )


if __name__ == "__main__":
    main()
