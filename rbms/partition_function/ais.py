from typing import Generator

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
) -> tuple[Tensor, dict[str, Tensor]]:
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


def compute_partition_function_ais(num_chains: int, num_beta: int, params: EBM) -> float:
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

    log_z_init = params.ref_log_z
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



def _make_energy_interpolated_model(
    params_ref: EBM,
    params: EBM,
    beta: float,
) -> EBM:
    """Build an EBM whose energy is (1 - beta) E_ref + beta E_model.

    This is the mathematically correct AIS bridge for neural-network EBMs,
    where parameter interpolation is not equivalent to energy interpolation.
    """

    from rbms.EBM_binary.classes import BEBM
    from rbms.EBM_binary.energies import InterpolatedEnergy

    if not hasattr(params_ref, "energy") or not hasattr(params, "energy"):
        raise TypeError("Energy interpolation AIS requires EBM objects exposing `.energy`.")

    energy = InterpolatedEnergy(
        energy_0=params_ref.energy,
        energy_1=params.energy,
        beta=beta,
    )

    return BEBM(
        energy=energy,
        num_visibles=params.num_visibles,
        device=params.device,
        dtype=params.dtype,
    )


def _init_reference_chains(
    params_ref: EBM,
    num_chains: int,
) -> dict[str, Tensor]:
    """Initialize AIS chains from the independent reference when available."""

    visible_field = getattr(params_ref.energy, "visible_field", None)

    if visible_field is None:
        return params_ref.init_chains(num_samples=num_chains)

    probabilities = torch.sigmoid(visible_field)
    visible = torch.bernoulli(probabilities.expand(num_chains, -1))

    return params_ref.init_chains(
        num_samples=num_chains,
        start_v=visible,
    )


def compute_partition_function_ais_ebm(
    num_chains: int,
    num_beta: int,
    params: EBM,
    n_steps: int = 1,
    kernel: str | None = None,
    kernel_params: dict | None = None,
) -> float:
    """Compute log Z for an energy-wrapped EBM using energy-space AIS.

    Unlike `compute_partition_function_ais`, this function does not rely on
    `params_ref * (1 - beta) + params * beta`. That parameter-space
    interpolation is correct for RBMs, whose energies are linear in the
    parameters, but it is not correct for generic neural-network EBMs.

    The annealing path is instead

        E_beta(v) = (1 - beta) E_ref(v) + beta E_model(v).
    """

    if kernel_params is None:
        kernel_params = {}

    device = params.device
    dtype = getattr(params, "dtype", torch.get_default_dtype())

    all_betas = torch.linspace(
        start=0.0,
        end=1.0,
        steps=num_beta,
        device=device,
        dtype=dtype,
    )

    log_z_init = params.ref_log_z
    params_ref = params.independent_model()
    chains = _init_reference_chains(params_ref=params_ref, num_chains=num_chains)
    log_weights = torch.zeros(num_chains, device=device, dtype=dtype)

    sample_kwargs = dict(kernel_params=kernel_params)
    if kernel is not None:
        sample_kwargs["kernel"] = kernel

    beta_prev = float(all_betas[0].item())
    curr_params = _make_energy_interpolated_model(
        params_ref=params_ref,
        params=params,
        beta=beta_prev,
    )

    for beta_next_tensor in all_betas[1:]:
        beta_next = float(beta_next_tensor.item())
        next_params = _make_energy_interpolated_model(
            params_ref=params_ref,
            params=params,
            beta=beta_next,
        )

        chains = curr_params.sample_state(
            chains=chains,
            n_steps=n_steps,
            **sample_kwargs,
        )

        energy_prev = curr_params.compute_energy_visibles(v=chains["visible"])
        energy_next = next_params.compute_energy_visibles(v=chains["visible"])
        log_weights += -energy_next.detach() + energy_prev.detach()

        curr_params = next_params

    log_z = torch.logsumexp(log_weights, 0) - np.log(num_chains) + log_z_init
    return log_z.item()
