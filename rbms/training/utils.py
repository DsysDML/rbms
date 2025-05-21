import pathlib
import time
from typing import Any, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from rbms.classes import EBM
from rbms.const import LOG_FILE_HEADER
from rbms.dataset.dataset_class import RBMDataset
from rbms.io import load_model, save_model
from rbms.map_model import map_model
from rbms.parser import default_args, set_args_default
from rbms.potts_bernoulli.classes import PBRBM
from rbms.potts_bernoulli.utils import ensure_zero_sum_gauge
from rbms.utils import get_saved_updates


def setup_training(
    args: dict,
    train_dataset: RBMDataset,
    test_dataset: Optional[RBMDataset] = None,
    map_model: dict[str, EBM] = map_model,
) -> Tuple[
    EBM,
    dict[str, Tensor],
    dict[str, Any],
    float,
    int,
    float,
    float,
    pathlib.Path,
    tqdm,
    RBMDataset,
    RBMDataset,
]:
    # Retrieve the the number of training updates already performed on the model
    updates = get_saved_updates(filename=args["filename"])
    num_updates = updates[-1]
    if args["num_updates"] <= num_updates:
        raise RuntimeError(
            f"The parameter /'num_updates/' ({args['num_updates']}) must be greater than the previous number of updates ({num_updates})."
        )

    params, parallel_chains, elapsed_time, hyperparameters = load_model(
        args["filename"],
        num_updates,
        device=args["device"],
        dtype=args["dtype"],
        restore=True,
        map_model=map_model,
    )

    # Hyperparameters
    for k, v in hyperparameters.items():
        if args[k] is None:
            args[k] = v

    if test_dataset is None:
        train_dataset, test_dataset = train_dataset.split_train_test(
            rng=np.random.default_rng(args["seed"]),
            train_size=args["train_size"],
            test_size=args["test_size"],
        )

    # Open the log file if it exists
    log_filename = pathlib.Path(args["filename"]).parent / pathlib.Path(
        f"log-{pathlib.Path(args['filename']).stem}.csv"
    )
    args["log"] = log_filename.exists()

    # Progress bar
    pbar = tqdm(
        initial=num_updates,
        total=args["num_updates"],
        colour="red",
        dynamic_ncols=True,
        ascii="-#",
    )
    pbar.set_description(f"Training {params.name}")

    # Initialize gradients for the parameters
    for p in params.parameters():
        p.grad = torch.zeros_like(p)

    # Start recording training time
    start = time.time()

    return (
        params,
        parallel_chains,
        args,
        num_updates,
        start,
        elapsed_time,
        log_filename,
        pbar,
        train_dataset,
        test_dataset,
    )


def create_machine(
    filename: str,
    params: EBM,
    num_visibles: int,
    num_hiddens: int,
    num_chains: int,
    batch_size: int,
    gibbs_steps: int,
    learning_rate: float,
    train_size: float,
    log: bool,
    flags: List[str],
    seed: int,
    L1: float,
    L2: float,
) -> None:
    """Create a RBM and save it to a new file.

    Args:
        filename (str): The name of the file to save the RBM.
        params (RBM): Initialized parameters.
        num_visibles (int): Number of visible units.
        num_hiddens (int): Number of hidden units.
        num_chains (int): Number of parallel chains for gradient computation.
        batch_size (int): Size of the data batch.
        gibbs_steps (int): Number of Gibbs steps to perform.
        learning_rate (float): Learning rate for training.
        log (bool): Whether to enable logging.
        L1 (float): Lambda parameter for L1 regularization.
        L2 (float): Lambda parameter for L2 regularization.
    """
    # Permanent chains
    parallel_chains = params.init_chains(num_samples=num_chains)
    parallel_chains = params.sample_state(chains=parallel_chains, n_steps=gibbs_steps)
    with h5py.File(filename, "w") as file_model:
        hyperparameters = file_model.create_group("hyperparameters")
        hyperparameters["num_hiddens"] = num_hiddens
        hyperparameters["num_visibles"] = num_visibles
        hyperparameters["num_chains"] = num_chains
        hyperparameters["batch_size"] = batch_size
        hyperparameters["gibbs_steps"] = gibbs_steps
        hyperparameters["filename"] = str(filename)
        hyperparameters["learning_rate"] = learning_rate
        hyperparameters["train_size"] = train_size
        hyperparameters["seed"] = seed
        hyperparameters["L1"] = L1
        hyperparameters["L2"] = L2

    save_model(
        filename=filename,
        params=params,
        chains=parallel_chains,
        num_updates=1,
        time=0.0,
        flags=flags,
    )
    if log:
        filename = pathlib.Path(filename)
        log_filename = filename.parent / pathlib.Path(f"log-{filename.stem}.csv")
        with open(log_filename, "w", encoding="utf-8") as log_file:
            log_file.write(",".join(LOG_FILE_HEADER) + "\n")


def get_checkpoints(num_updates: int, n_save: int, spacing: str = "exp") -> np.ndarray:
    """Select the list of training times (ages) at which to save the model.

    Args:
        num_updates (int): Number of gradient updates to perform during training.
        n_save (int): Number of models to save.
        spacing (str, optional): Spacing method, either "linear" ("lin") or "exponential" ("exp"). Defaults to "exp".

    Returns:
        np.ndarray: Array of checkpoint indices.
    """
    match spacing:
        case "exp":
            checkpoints = []
            xi = num_updates
            for _ in range(n_save):
                checkpoints.append(xi)
                xi = xi / num_updates ** (1 / n_save)
            checkpoints = np.unique(np.array(checkpoints, dtype=np.int32))
        case "linear":
            checkpoints = np.linspace(1, num_updates, n_save).astype(np.int32)
        case _:
            raise ValueError(
                f"spacing should be one of ('exp', 'linear'), got {spacing}"
            )
    checkpoints = np.unique(np.append(checkpoints, num_updates))
    return checkpoints


def initialize_model_archive(
    args: dict,
    model_type: str,
    train_dataset: RBMDataset,
    test_dataset: Optional[RBMDataset],
    dtype: torch.dtype,
):
    num_visibles = train_dataset.get_num_visibles()
    args = set_args_default(args=args, default_args=default_args)
    rng = np.random.default_rng(args["seed"])
    if test_dataset is None:
        train_dataset, _ = train_dataset.split_train_test(
            rng, args["train_size"], args["test_size"]
        )
    params = map_model[model_type].init_parameters(
        num_hiddens=args["num_hiddens"],
        dataset=train_dataset,
        device=args["device"],
        dtype=dtype,
    )

    if isinstance(params, PBRBM):
        ensure_zero_sum_gauge(params)
    create_machine(
        filename=args["filename"],
        params=params,
        num_visibles=num_visibles,
        num_hiddens=args["num_hiddens"],
        num_chains=args["num_chains"],
        batch_size=args["batch_size"],
        gibbs_steps=args["gibbs_steps"],
        learning_rate=args["learning_rate"],
        train_size=args["train_size"],
        log=args["log"],
        flags=["checkpoint"],
        seed=args["seed"],
        L1=args["L1"],
        L2=args["L2"],
    )
