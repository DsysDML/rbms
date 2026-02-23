import argparse

import h5py
import torch

from rbms import get_saved_updates
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
from rbms.sampler import CD, PCD, RDM
from rbms.training.implement import _init_training, _restore_training
from rbms.training.pcd import train
from rbms.training.utils import get_checkpoints


def create_parser() -> argparse.ArgumentParser:
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


def main(args, map_model=map_model):
    checkpoints = get_checkpoints(
        num_updates=args["num_updates"],
        n_save=args["n_save"],
        spacing=args["spacing"],
    )
    train_dataset, test_dataset = load_dataset(
        dataset_name=args["dataset"],
        test_dataset_name=args["test_dataset"],
        subset_labels=args["subset_labels"],
        use_weights=args["use_weights"],
        alphabet=args["alphabet"],
        remove_duplicates=args["remove_duplicates"],
        device=args["device"],
        dtype=args["dtype"],
    )
    flags = ["checkpoint"]
    if not args["restore"]:
        args = set_args_default(args, default_args=default_args)
        _init_training(
            train_dataset=train_dataset,
            seed=args["seed"],
            train_size=args["train_size"],
            test_size=1 - args["train_size"],
            num_hiddens=args["num_hiddens"],
            num_chains=args["num_chains"],
            model_type=args["model_type"],
            filename=args["filename"],
            n_save=args["n_save"],
            spacing=args["spacing"],
            batch_size=args["batch_size"],
            optim=args["optim"],
            mult_optim=args["mult_optim"],
            training_type=args["training_type"],
            learning_rate=args["learning_rate"],
            max_lr=args["max_lr"],
            gibbs_steps=args["gibbs_steps"],
            beta=args["beta"],
            centered=not (args["no_center"]),
            L1=args["L1"],
            L2=args["L2"],
            normalize_grad=args["normalize_grad"],
            max_norm_grad=args["max_norm_grad"],
            subset_labels=args["subset_labels"],
            use_weights=args["use_weights"],
            alphabet=args["alphabet"],
            remove_duplicates=args["remove_duplicates"],
            dtype=args["dtype"],
            device=args["device"],
            flags=flags,
            map_model=map_model,
        )
        args["update"] = 1
    print(args)

    args = load_args_from_filename(args)
    print(args)
    args = set_args_default(args, default_args)
    if args["update"] is None:
        args["update"] = get_saved_updates(args["filename"])[-1]
    (
        params,
        parallel_chains,
        target_update,
        elapsed_time,
        train_dataset,
        test_dataset,
    ) = _restore_training(
        filename=args["filename"],
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_updates=args["num_updates"],
        target_update=args["update"],
        seed=args["seed"],
        train_size=args["train_size"],
        test_size=args["test_size"],
        device=args["device"],
        dtype=args["dtype"],
        map_model=map_model,
    )

    optimizer = setup_optim(args["optim"], args, params)
    from rbms.pre_grad import build_pre_grad_update

    pre_grad_update = build_pre_grad_update(
        optimizer=optimizer,
        lambda_l1=args["L1"],
        lambda_l2=args["L2"],
        normalize_grad=args["normalize_grad"],
        max_grad_norm=args["max_norm_grad"],
    )

    match args["training_type"]:
        case "pcd":
            sampler = PCD(
                params=params,
                chains=parallel_chains,
                num_steps=args["gibbs_steps"],
                beta=args["beta"],
            )
        case "cd":
            sampler = CD(params=params, num_steps=args["gibbs_steps"], beta=args["beta"])
        case "rdm":
            sampler = RDM(
                params=params,
                num_chains=parallel_chains["visible"].shape[0],
                num_steps=args["gibbs_steps"],
                beta=args["beta"],
            )

        case _:
            raise ValueError(f"No training type {args['training_type']} supported.")

    train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        params=params,
        sampler=sampler,
        optimizer=optimizer,
        batch_size=args["batch_size"],
        centered=not (args["no_center"]),
        curr_update=args["update"],
        pre_grad_update=pre_grad_update,
        elapsed_time=elapsed_time,
        checkpoints=checkpoints,
        num_updates=args["num_updates"],
        filename=args["filename"],
    )


def load_args_from_filename(args: dict):
    with h5py.File(args["filename"], "r") as f:
        if args["gibbs_steps"] is None:
            args["gibbs_steps"] = f["sampling_args"]["gibbs_steps"][()].item()
        if args["beta"] is None:
            args["beta"] = f["sampling_args"]["beta"][()].item()
        if args["optim"] is None:
            args["optim"] = str(f["train_args"]["optim"][()])
        if args["batch_size"] is None:
            args["batch_size"] = f["train_args"]["batch_size"][()].item()
        if args["training_type"] is None:
            args["training_type"] = str(f["train_args"]["training_type"][()])
        args["no_center"] = f["grad_args"]["no_center"][()].item()
        args["seed"] = f["dataset_args"]["seed"][()].item()
        args["train_size"] = f["dataset_args"]["train_size"][()].item()
        args["test_size"] = f["dataset_args"]["test_size"][()].item()
        if args["L1"] is None:
            args["L1"] = f["grad_args"]["L1"][()].item()
        if args["L2"] is None:
            args["L2"] = f["grad_args"]["L2"][()].item()
        if args["normalize_grad"] is None:
            args["normalize_grad"] = f["grad_args"]["normalize_grad"][()].item()
        if args["max_norm_grad"] is None:
            args["max_norm_grad"] = f["grad_args"]["max_norm_grad"][()].item()

    return args


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)
    # args = set_args_default(args, default_args=default_args)
    args = match_args_dtype(args)
    main(args=args)
