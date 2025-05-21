from typing import Optional

import numpy as np
import torch
from torch import Tensor

from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.bernoulli_bernoulli.implement import (
    _compute_energy,
    _compute_energy_hiddens,
    _compute_energy_visibles,
    _compute_gradient,
    _init_chains,
    _init_parameters,
    _sample_hiddens,
    _sample_visibles,
)
from rbms.dataset.dataset_class import RBMDataset


def sample_hiddens(
    chains: dict[str, Tensor], params: BBRBM, beta: float = 1.0
) -> dict[str, Tensor]:
    """Sample the hidden layer conditionally to the visible one.

    Args:
        chains (dict[str, Tensor]): The parallel chains used for sampling.
        params (BBRBM): The parameters of the RBM.
        beta (float, optional): The inverse temperature. Defaults to 1.0.

    Returns:
        dict[str, Tensor]: The updated chains with sampled hidden states.
    """
    chains["hidden"], chains["hidden_mag"] = _sample_hiddens(
        v=chains["visible"],
        weight_matrix=params.weight_matrix,
        hbias=params.hbias,
        beta=beta,
    )
    return chains


def sample_visibles(
    chains: dict[str, Tensor], params: BBRBM, beta: float = 1.0
) -> dict[str, Tensor]:
    """Sample the visible layer conditionally to the hidden one.

    Args:
        chains (dict[str, Tensor]): The parallel chains used for sampling.
        params (BBRBM): The parameters of the RBM.
        beta (float, optional): The inverse temperature. Defaults to 1.0.

    Returns:
        dict[str, Tensor]: The updated chains with sampled visible states.
    """
    chains["visible"], chains["visible_mag"] = _sample_visibles(
        h=chains["hidden"],
        weight_matrix=params.weight_matrix,
        vbias=params.vbias,
        beta=beta,
    )
    return chains


def compute_energy(
    v: Tensor,
    h: Tensor,
    params: BBRBM,
) -> Tensor:
    """Compute the energy of the RBM on the visible and hidden variables.

    Args:
        v (Tensor): Visible configurations.
        h (Tensor): Hidden configurations.
        params (BBRBM): Parameters of the RBM.

    Returns:
        Tensor: The computed energy.
    """
    return _compute_energy(
        v=v,
        h=h,
        vbias=params.vbias,
        hbias=params.hbias,
        weight_matrix=params.weight_matrix,
    )


def compute_energy_visibles(v: Tensor, params: BBRBM) -> Tensor:
    """Returns the marginalized energy of the model computed on the visible configurations

    Args:
        v (Tensor): Visible configurations
        params (BBRBM): Parameters of the RBM

    Returns:
        Tensor: The computed energy.
    """
    return _compute_energy_visibles(
        v=v, vbias=params.vbias, hbias=params.hbias, weight_matrix=params.weight_matrix
    )


def compute_energy_hiddens(h: Tensor, params: BBRBM) -> Tensor:
    """Returns the marginalized energy of the model computed on hidden configurations

    Args:
        h (Tensor): Hidden configurations.
        params (BBRBM): Parameters of the RBM.
    """
    return _compute_energy_hiddens(
        h=h, vbias=params.vbias, hbias=params.hbias, weight_matrix=params.weight_matrix
    )


def compute_gradient(
    data: dict[str, Tensor],
    chains: dict[str, Tensor],
    params: BBRBM,
    centered: bool = True,
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
) -> None:
    """Compute the gradient for each of the parameters and attach it.

    Args:
        data (dict[str, Tensor]): The data state.
        chains (dict[str, Tensor]): The parallel chains used for gradient computation.
        params (BBRBM): The parameters of the RBM.
        centered (bool, optional): Whether to use centered gradients. Defaults to True.
        lambda_l1 (float, optional): factor for the L1 regularization. Defaults to 0.
        lambda_l2 (float, optional): factor for the L2 regularization. Defaults to 0.
    """
    _compute_gradient(
        v_data=data["visible"],
        mh_data=data["hidden_mag"],
        w_data=data["weights"],
        v_chain=chains["visible"],
        h_chain=chains["hidden"],
        w_chain=chains["weights"],
        vbias=params.vbias,
        hbias=params.hbias,
        weight_matrix=params.weight_matrix,
        centered=centered,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
    )


def init_chains(
    num_samples: int,
    params: BBRBM,
    weights: Optional[Tensor] = None,
    start_v: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Initialize a Markov chain for the RBM by sampling a uniform distribution on the visible layer
    and sampling the hidden layer according to the visible one.

    Args:
        num_samples (int): The number of samples to initialize.
        params (BBRBM): The parameters of the RBM.
        weights (Tensor, optional): The weights of each configuration. Defaults to None.
        start_v (Optional[Tensor], optional): The initial visible states. Defaults to None.

    Returns:
        dict[str, Tensor]: The initialized Markov chain.

    Notes:
        - If start_v is specified, its number of samples will override the num_samples argument.
    """
    visible, hidden, mean_visible, mean_hidden = _init_chains(
        num_samples=num_samples,
        weight_matrix=params.weight_matrix,
        hbias=params.hbias,
        start_v=start_v,
    )
    if weights is None:
        weights = torch.ones(
            visible.shape[0], device=visible.device, dtype=visible.dtype
        )
    return dict(
        visible=visible,
        hidden=hidden,
        visible_mag=mean_visible,
        hidden_mag=mean_hidden,
        weights=weights,
    )


def init_parameters(
    num_hiddens: int,
    dataset: RBMDataset,
    device: torch.device,
    dtype: torch.dtype,
    var_init: float = 1e-4,
) -> BBRBM:
    """Initialize the parameters of the RBM.

    Args:
        num_hiddens (int): Number of hidden units.
        dataset (RBMDataset): Training dataset.
        device (torch.device): PyTorch device for the parameters.
        dtype (torch.dtype): PyTorch dtype for the parameters.
        var_init (float, optional): Variance of the weight matrix. Defaults to 1e-4.

    Notes:
        - The number of visible units is induced from the dataset provided.
        - Hidden biases are set to 0.
        - Visible biases are set to the frequencies of the dataset.
        - The weight matrix is initialized with a Gaussian distribution of variance `var_init`.
    """
    data = dataset.data
    # Convert to torch Tensor if necessary
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(dataset.data).to(device=device, dtype=dtype)
    vbias, hbias, weight_matrix = _init_parameters(
        num_hiddens=num_hiddens,
        data=data,
        device=device,
        dtype=dtype,
        var_init=var_init,
    )
    return BBRBM(weight_matrix=weight_matrix, vbias=vbias, hbias=hbias)
