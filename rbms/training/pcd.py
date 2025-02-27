import time
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.optim import SGD

from rbms.classes import RBM
from rbms.dataset.dataset_class import RBMDataset
from rbms.io import save_model
from rbms.map_model import map_model
from rbms.potts_bernoulli.classes import PBRBM
from rbms.potts_bernoulli.utils import ensure_zero_sum_gauge
from rbms.sampling.gibbs import sample_state
from rbms.training.utils import create_machine, setup_training
from rbms.utils import check_file_existence, log_to_csv


def fit_batch_pcd(
    batch: Tuple[Tensor, Tensor],
    parallel_chains: dict[str, Tensor],
    params: RBM,
    gibbs_steps: int,
    beta: float,
    centered: bool = True,
) -> Tuple[dict[str, Tensor], dict]:
    """Sample the RBM and compute the gradient.

    Args:
        batch (Tuple[Tensor, Tensor]): Dataset samples and associated weights.
        parallel_chains (dict[str, Tensor]): Parallel chains used for gradient computation.
        params (RBM): Parameters of the RBM.
        gibbs_steps (int): Number of Gibbs steps to perform.
        beta (float): Inverse temperature.

    Returns:
        Tuple[dict[str, Tensor], dict]: A tuple containing the updated chains and the logs.
    """
    v_data, w_data = batch
    # Initialize batch
    curr_batch = params.init_chains(
        num_samples=v_data.shape[0],
        weights=w_data,
        start_v=v_data,
    )
    # sample permanent chains
    parallel_chains = sample_state(
        gibbs_steps=gibbs_steps,
        chains=parallel_chains,
        params=params,
        beta=beta,
    )
    params.compute_gradient(data=curr_batch, chains=parallel_chains, centered=centered)
    logs = {}
    return parallel_chains, logs


def train(
    dataset: RBMDataset,
    test_dataset: RBMDataset,
    model_type: str,
    args: dict,
    dtype: torch.dtype,
    checkpoints: np.ndarray,
    map_model: dict[str, RBM] = map_model,
) -> None:
    """Train the Bernoulli-Bernoulli RBM model.

    Args:
        dataset (RBMDataset): The training dataset.
        test_dataset (RBMDataset): The test dataset (not used).
        model_type (str): Type of RBM used (BBRBM or PBRBM)
        args (dict): A dictionary of training arguments.
        dtype (torch.dtype): The data type for the parameters.
        checkpoints (np.ndarray): An array of checkpoints for saving model states.
    """
    filename = args["filename"]
    if not (args["overwrite"]):
        check_file_existence(filename)

    num_visibles = dataset.get_num_visibles()

    # Create a first archive with the initialized model
    if not (args["restore"]):
        params = map_model[model_type].init_parameters(
            num_hiddens=args["num_hiddens"],
            dataset=dataset,
            device=args["device"],
            dtype=dtype,
        )
        create_machine(
            filename=filename,
            params=params,
            num_visibles=num_visibles,
            num_hiddens=args["num_hiddens"],
            num_chains=args["num_chains"],
            batch_size=args["batch_size"],
            gibbs_steps=args["gibbs_steps"],
            learning_rate=args["learning_rate"],
            log=args["log"],
            flags=["checkpoint"],
        )

    (
        params,
        parallel_chains,
        args,
        learning_rate,
        num_updates,
        start,
        elapsed_time,
        log_filename,
        pbar,
    ) = setup_training(args, map_model=map_model)

    optimizer = SGD(params.parameters(), lr=learning_rate, maximize=True)

    for k, v in args.items():
        print(f"{k} : {v}")

    # Continue the training
    with torch.no_grad():
        for idx in range(num_updates + 1, args["num_updates"] + 1):
            rand_idx = torch.randperm(len(dataset))[: args["batch_size"]]
            batch = (dataset.data[rand_idx], dataset.weights[rand_idx])

            optimizer.zero_grad(set_to_none=False)
            parallel_chains, logs = fit_batch_pcd(
                batch=batch,
                parallel_chains=parallel_chains,
                params=params,
                gibbs_steps=args["gibbs_steps"],
                beta=args["beta"],
            )
            optimizer.step()
            if isinstance(params, PBRBM):
                ensure_zero_sum_gauge(params)

            # Save current model if necessary
            if idx in checkpoints:
                curr_time = time.time() - start
                save_model(
                    filename=args["filename"],
                    params=params,
                    chains=parallel_chains,
                    num_updates=idx,
                    time=curr_time + elapsed_time,
                    flags=["checkpoint"],
                )

            if args["log"]:
                log_to_csv(logs, log_file=log_filename)

            # Update progress bar
            pbar.update(1)
