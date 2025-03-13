from typing import List, Optional

import h5py
import torch
from torch import Tensor

from rbms.classes import EBM
from rbms.utils import swap_chains


def swap_configurations(
    chains: List[dict[str, Tensor]],
    params: EBM,
    inverse_temperatures: Tensor,
    index: Optional[List[Tensor]] = None,
):
    """
    Swap configurations between adjacent chains based on their energy and inverse temperatures.

    Args:
        chains (List[dict[str, Tensor]]): List of dict[str, Tensor] objects, where each dict[str, Tensor] contains visible and hidden states.
        params (RBM): RBM parameters used to compute the energy of the configurations.
        inverse_temperatures (Tensor): Inverse temperatures for each swap.
        index (Optional[List[Tensor]]): Starting inverse temperature indices of the chains.

    Returns:
        Tuple[List[dict[str, Tensor]], Tensor, Optional[List[Tensor]]]:
            - Updated list of dict[str, Tensor] objects after swapping.
            - Tensor of acceptance rates for each swap.
            - Updated list of indices if provided, otherwise None.
    """
    n_chains, L = chains[0]["visible"].shape
    acc_rate = torch.zeros(inverse_temperatures.shape[0] - 1)
    for idx in range(inverse_temperatures.shape[0] - 1):
        energy_0 = params.compute_energy(v=chains[idx].visible, h=chains[idx].hidden)
        energy_1 = params.compute_energy(
            v=chains[idx + 1].visible, h=chains[idx + 1].hidden
        )

        delta_energy = (
            -energy_1 * inverse_temperatures[idx]
            + energy_0 * inverse_temperatures[idx]
            + energy_1 * inverse_temperatures[idx + 1]
            - energy_0 * inverse_temperatures[idx + 1]
        )
        swap = torch.exp(delta_energy) > torch.rand(
            size=(n_chains,), device=delta_energy.device
        )
        if index is not None:
            swapped_index_0 = torch.where(swap, index[idx + 1], index[idx])
            swapped_index_1 = torch.where(swap, index[idx], index[idx + 1])
            index[idx] = swapped_index_0
            index[idx + 1] = swapped_index_1

        acc_rate[idx] = (swap.sum() / n_chains).cpu()

        chains[idx], chains[idx + 1] = swap_chains(chains[idx], chains[idx + 1], swap)

    return chains, acc_rate, index


def find_inverse_temperatures(target_acc_rate: float, params: EBM) -> Tensor:
    """
    Finds a sequence of inverse temperatures for a given target acceptance rate.

    This function generates a sequence of inverse temperatures that are used in
    parallel tempering to achieve a specified target acceptance rate. It starts
    with an initial temperature of 0 and iteratively finds the next temperature
    by performing Gibbs sampling and configuration swapping.

    Args:
        target_acc_rate (float): The target acceptance rate for the swaps.
        params (RBM): An instance of the RBM class containing the model parameters.

    Returns:
        Tensor: A tensor containing the selected inverse temperatures.
    """
    inverse_temperatures = torch.linspace(0, 1, 1000)
    selected_temperatures = [0]
    n_chains = 100
    prev_chains = params.init_chains(num_samples=n_chains)
    new_chains = params.init_chains(num_samples=n_chains)

    for i in range(len(inverse_temperatures) - 1):
        prev_chains = params.sample_state(
            n_steps=10,
            chains=prev_chains,
            beta=selected_temperatures[-1],
        )
        new_chains = params.sample_state(
            n_steps=10,
            chains=new_chains,
            beta=inverse_temperatures[i],
        )

        _, acc_rate, _ = swap_configurations(
            chains=[prev_chains, new_chains],
            params=params,
            inverse_temperatures=torch.tensor(
                [selected_temperatures[-1], inverse_temperatures[i]]
            ),
        )
        if acc_rate[-1] < target_acc_rate + 0.1:
            selected_temperatures.append(inverse_temperatures[i])
            prev_chains = new_chains.clone()
    if selected_temperatures[-1] != 1.0:
        selected_temperatures.append(1)
    return torch.tensor(selected_temperatures)


def pt_sampling(
    it_mcmc: int,
    increment: int,
    target_acc_rate: float,
    num_chains: int,
    params: EBM,
    out_file: str,
    save_index: bool,
):
    """
    Parallel Tempering (PT) sampling for a Restricted Boltzmann Machine (RBM).

    Args:
        it_mcmc (int): Total number of MCMC iterations.
        increment (int): Number of Gibbs steps between each swap attempt.
        target_acc_rate (float): Target acceptance rate for temperature swaps.
        num_chains (int): Number of parallel chains to run.
        params (RBM): The RBM model parameters.
        out_file (str): Path to the output file where indices will be saved.

    Returns:
        list: List of final chains after sampling.
        torch.Tensor: Inverse temperatures used for sampling.
        list: Indices of the chains.
    """
    inverse_temperatures = find_inverse_temperatures(target_acc_rate, params)
    list_chains = []
    for i in range(inverse_temperatures.shape[0]):
        list_chains.append(params.init_chains(num_samples=num_chains))

    # Annealing to initialize the chains
    index = None
    if save_index:
        index = []
    for i in range(inverse_temperatures.shape[0]):
        for j in range(i, inverse_temperatures.shape[0]):
            list_chains[j] = params.sample_state(
                n_steps=increment,
                chains=list_chains[j],
                beta=inverse_temperatures[i],
            )
        if save_index:
            index.append(
                torch.ones(
                    list_chains[i].visible.shape[0], device=list_chains[i].device
                )
                * i
            )

    counts = 0
    while counts < it_mcmc:
        counts += increment
        # Iterate chains
        for i in range(len(list_chains)):
            list_chains[i] = params.clonesample_state(
                n_steps=increment,
                chains=list_chains[i],
                beta=inverse_temperatures[i],
            )

        # Swap chains
        list_chains, acc_rate, index = swap_configurations(
            chains=list_chains,
            params=params,
            inverse_temperatures=inverse_temperatures,
            index=index,
        )
        if save_index:
            with h5py.File(out_file, "a") as f:
                f[f"index_{counts}"] = torch.vstack(index).cpu().numpy()

    return list_chains, inverse_temperatures, index
