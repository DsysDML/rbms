from typing import Generator, Tuple

import numpy as np
import torch
from torch import Tensor

from rbms.classes import EBM


def update_weights_ais(
    prev_params: EBM,
    curr_params: EBM,
    chains: dict[str, Tensor],
    log_weights: Tensor,
    n_steps: int = 1,
) -> Tuple[Tensor, dict[str, Tensor]]:
    """Update the weights used during Annealed Importance Sampling.

    Args:
        prev_params (RBM): The previous parameters of the RBM.
        curr_params (RBM): The current parameters of the RBM.
        chains (dict[str, Tensor]): The parallel chains used for sampling.
        log_weights (Tensor): The log weights used in the sampling process.

    Returns:
        Tuple[Tensor, dict[str, Tensor]]: A tuple containing the updated log weights and the updated chains.
    """
    chains = prev_params.sample_state(n_steps=n_steps, chains=chains)
    energy_prev = prev_params.compute_energy_visibles(v=chains["visible"])
    energy_curr = curr_params.compute_energy_visibles(v=chains["visible"])
    log_weights += -energy_curr + energy_prev
    return log_weights, chains


def interpolate_ebm(
    params_1: EBM, params_2: EBM, steps: Tensor
) -> Generator[EBM, None, None]:
    """Interpolates between two RBMs"""
    for step in steps:
        yield params_1 * (1 - step) + params_2 * step


def compute_partition_function_ais(
    num_chains: int, num_beta: int, params: EBM
) -> float:
    """Compute the log partition function using Annealed Importance Sampling with temperature.

    Args:
        num_chains (int): Number of parallel chains for sampling.
        num_beta (int): Number of temperature steps.
        params (RBM): Parameters of the RBM.
        vbias_ref (Optional[Tensor], optional): Reference visible bias. Defaults to None.

    Returns:
        float: The computed log partition function.
    """
    device = params.device

    all_betas = torch.linspace(start=0, end=1, steps=num_beta)

    # Compute the reference log partition function
    ## Here the case where all the weights are 0

    log_z_init = params.ref_log_z()
    params_ref = params.independent_model()

    chains = params_ref.init_chains(num_samples=num_chains)

    log_weights = torch.zeros(num_chains, device=device)

    interpolator = interpolate_ebm(params_ref, params, steps=all_betas)

    curr_params = next(interpolator)
    for i in range(len(all_betas) - 1):
        # interpolate between true distribution and ref distribution
        prev_params = curr_params.clone()
        curr_params = next(interpolator)
        log_weights, chains = update_weights_ais(
            prev_params=prev_params,
            curr_params=curr_params,
            chains=chains,
            log_weights=log_weights,
        )
    log_z = torch.logsumexp(log_weights, 0) - np.log(num_chains) + log_z_init
    return log_z.item()
