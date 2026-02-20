import time

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from tqdm.autonotebook import tqdm

from rbms.classes import EBM, Sampler
from rbms.dataset.dataset_class import RBMDataset
from rbms.io import save_model, save_sampler
from rbms.potts_bernoulli.classes import PBRBM
from rbms.potts_bernoulli.utils import ensure_zero_sum_gauge


def fit_batch_pcd(
    batch: tuple[Tensor, Tensor],
    parallel_chains: dict[str, Tensor],
    params: EBM,
    gibbs_steps: int,
    beta: float,
    centered: bool = True,
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
    normalize_grad: bool = True,
    max_norm_grad: float = -1,
) -> dict[str, Tensor]:
    """Sample the EBM and compute the gradient.

    Args:
        batch (Tuple[Tensor, Tensor]): Dataset samples and associated weights.
        parallel_chains (dict[str, Tensor]): Parallel chains used for gradient computation.
        params (EBM): Parameters of the EBM.
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
    parallel_chains = params.sample_state(
        chains=parallel_chains, n_steps=gibbs_steps, beta=beta
    )
    params.compute_gradient(
        data=curr_batch,
        chains=parallel_chains,
        centered=centered,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
    )
    if normalize_grad:
        params.normalize_grad()
    if max_norm_grad > 0:
        params.clip_grad(max_norm=max_norm_grad)
    return parallel_chains


@torch.compile
@torch.no_grad
def train(
    train_dataset: RBMDataset,
    test_dataset: RBMDataset,
    params: EBM,
    parallel_chains: dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    curr_update: int,
    elapsed_time: float,
    checkpoints: np.ndarray,
    args_save: dict[str, str],
    args_train: dict[str, int | float],
    args_grad: dict[str, bool | float],
    args_sampling: dict[str, int | float],
):
    """Train an EBM.

    Args:
        dataset (RBMDataset): The training dataset.
        test_dataset (RBMDataset): The test dataset (not used).
        model_type (str): Type of RBM used (BBRBM or PBRBM)
        args (dict): A dictionary of training arguments.
        dtype (torch.dtype): The data type for the parameters.
        checkpoints (np.ndarray): An array of checkpoints for saving model states.
    """
    # Sampling
    gibbs_steps: int = args_sampling["gibbs_steps"]
    beta: float = args_sampling["beta"]

    # Grad
    centered: bool = not (args_grad["no_center"])
    L1: float = args_grad["L1"]
    L2: float = args_grad["L2"]
    normalize_grad: bool = args_grad["normalize_grad"]
    max_norm_grad: float = args_grad["max_norm_grad"]

    # train
    batch_size: int = args_train["batch_size"]
    num_updates: int = args_train["num_updates"]
    training_type: str = args_train["training_type"]

    # save
    filename: str = args_save["filename"]

    # _train(
    #     params=params,
    #     parallel_chains=parallel_chains,
    #     optimizer=optimizer,
    #     train_dataset=train_dataset,
    #     checkpoints=checkpoints,
    #     curr_update=curr_update,
    #     num_updates=num_updates,
    #     batch_size=batch_size,
    #     training_type=training_type,
    #     gibbs_steps=gibbs_steps,
    #     beta=beta,
    #     centered=centered,
    #     L1=L1,
    #     L2=L2,
    #     normalize_grad=normalize_grad,
    #     max_norm_grad=max_norm_grad,
    #     filename=filename,
    #     elapsed_time=elapsed_time,
    # )
    # pbar
    pbar = tqdm(
        initial=curr_update,
        total=num_updates,
        colour="red",
        dynamic_ncols=True,
        ascii="-#",
    )
    pbar.set_description(f"Training {params.name}")

    start = time.perf_counter()

    for idx in range(curr_update + 1, num_updates + 1):
        batch = train_dataset.batch(batch_size)
        data, weights = batch["data"], batch["weights"]
        if training_type == "rdm":
            parallel_chains = params.init_chains(parallel_chains["visible"].shape[0])
        elif training_type == "cd":
            parallel_chains = params.init_chains(
                data.shape[0],
                weights=weights,
                start_v=data,
            )
        for opt in optimizer:
            opt.zero_grad(set_to_none=False)

        parallel_chains = fit_batch_pcd(
            batch=(data, weights),
            parallel_chains=parallel_chains,
            params=params,
            gibbs_steps=gibbs_steps,
            beta=beta,
            centered=centered,
            lambda_l1=L1,
            lambda_l2=L2,
            normalize_grad=normalize_grad,
            max_norm_grad=max_norm_grad,
        )
        for opt in optimizer:
            opt.step()

        if isinstance(params, PBRBM):
            ensure_zero_sum_gauge(params)

        # Save current model if necessary
        if idx in checkpoints or idx == num_updates:
            curr_time = time.perf_counter() - start
            learning_rate = torch.tensor([opt.param_groups[0]["lr"] for opt in optimizer])
            save_model(
                filename=filename,
                params=params,
                chains=parallel_chains,
                num_updates=idx,
                time=curr_time + elapsed_time,
                learning_rate=learning_rate,
                flags=["checkpoint"],
            )

        pbar.set_postfix_str(f"lr: {optimizer[0].param_groups[0]['lr']:.6f}")
        # Update progress bar
        pbar.update(1)


@torch.no_grad
@torch.compile
def train_v2(
    train_dataset: RBMDataset,
    test_dataset: RBMDataset,
    params: EBM,
    sampler: Sampler,
    optimizer: list[Optimizer],
    batch_size: int,
    centered: bool,
    curr_update: int,
    pre_grad_update: torch.nn.Sequential,
    elapsed_time: float,
    checkpoints: np.ndarray,
    num_updates: int,
    filename: str,
):
    pbar = tqdm(
        initial=curr_update,
        total=num_updates,
        colour="red",
        dynamic_ncols=True,
        ascii="-#",
    )
    pbar.set_description(f"Training {params.name}")

    start = time.perf_counter()

    for idx in range(curr_update + 1, num_updates + 1):
        batch = train_dataset.batch(batch_size)
        data, weights = batch["data"], batch["weights"]

        for opt in optimizer:
            opt.zero_grad(set_to_none=False)

        # Initialize batch
        curr_batch = params.init_chains(
            num_samples=data.shape[0],
            weights=weights,
            start_v=data,
        )
        parallel_chains = sampler.get_conf_grad(batch=data)

        params.compute_gradient(
            data=curr_batch,
            chains=parallel_chains,
            centered=centered,
        )
        # Do a bunch of modification on the gradient

        pre_grad_update(input=None)

        for opt in optimizer:
            opt.step()

        params.post_grad_update()
        sampler.post_grad_update(params=params)

        # Get flags for save
        flags = []
        flags = params.save_flags(flags)
        flags = sampler.save_flags(flags)
        if idx in checkpoints or idx == num_updates:
            flags.append("checkpoint")

        if len(flags) > 0:
            curr_time = time.perf_counter() - start
            learning_rate = torch.tensor([opt.param_groups[0]["lr"] for opt in optimizer])
            save_model(
                filename=filename,
                params=params,
                chains=parallel_chains,
                num_updates=idx,
                time=curr_time + elapsed_time,
                learning_rate=learning_rate,
                flags=flags,
            )

            save_sampler(filename, sampler)
        pbar.update(1)
