import numpy as np
import torch
from torch import Tensor

# --- switched to Bernoulli–Gaussian (Gaussian hidden) ---
from rbms.bernoulli_gaussian.classes import BGRBM
from rbms.bernoulli_gaussian.implement import (
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
    chains: dict[str, Tensor], params: BGRBM, beta: float = 1.0
) -> dict[str, Tensor]:
    """Sample h|v(Gaussian hidden with fixed var = 1/Nv)"""
    chains["hidden"], chains["hidden_mag"] = _sample_hiddens(
        v=chains["visible"],
        weight_matrix=params.weight_matrix,
        hbias=params.hbias,
        beta=beta,
    )
    return chains


def sample_visibles(
    chains: dict[str, Tensor], params: BGRBM, beta: float = 1.0
) -> dict[str, Tensor]:
    """Sample v|h Bernoulli"""
    chains["visible"], chains["visible_mag"] = _sample_visibles(
        h=chains["hidden"],
        weight_matrix=params.weight_matrix,
        vbias=params.vbias,
        beta=beta,
    )
    return chains


def compute_energy(v: Tensor, h: Tensor, params: BGRBM) -> Tensor:
    return _compute_energy(
        v=v,
        h=h,
        vbias=params.vbias,
        hbias=params.hbias,
        weight_matrix=params.weight_matrix,
    )


def compute_energy_visibles(v: Tensor, params: BGRBM) -> Tensor:
    """Marginalized energy over h"""
    return _compute_energy_visibles(
        v=v,
        vbias=params.vbias,
        hbias=params.hbias,
        weight_matrix=params.weight_matrix,
        const=params.const,
    )


def compute_energy_hiddens(h: Tensor, params: BGRBM) -> Tensor:
    """Energy marginalized over v"""
    return _compute_energy_hiddens(
        h=h,
        vbias=params.vbias,
        hbias=params.hbias,
        weight_matrix=params.weight_matrix,
    )


def compute_gradient(
    data: dict[str, Tensor],
    chains: dict[str, Tensor],
    params: BGRBM,
    centered: bool,
) -> None:
    _compute_gradient(
        v_data=data["visible"],
        h_data=data["hidden_mag"],  # use conditional mean for positive phase
        w_data=data["weights"],
        v_chain=chains["visible"],
        h_chain=chains["hidden_mag"],  # negative phase from chain samples
        w_chain=chains["weights"],
        vbias=params.vbias,
        hbias=params.hbias,
        weight_matrix=params.weight_matrix,
        centered=centered,
    )


def init_chains(
    num_samples: int,
    params: BGRBM,
    weights: Tensor | None = None,
    start_v: Tensor | None = None,
) -> dict[str, Tensor]:
    visible, hidden, mean_visible, mean_hidden = _init_chains(
        num_samples=num_samples,
        weight_matrix=params.weight_matrix,
        hbias=params.hbias,
        start_v=start_v,
    )
    if weights is None:
        weights = torch.ones(visible.shape[0], device=visible.device, dtype=visible.dtype)
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
) -> BGRBM:
    data = dataset.data
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(dataset.data).to(device=device, dtype=dtype)
    vbias, hbias, weight_matrix = _init_parameters(
        num_hiddens=num_hiddens, data=data, device=device, dtype=dtype, var_init=var_init
    )
    return BGRBM(
        weight_matrix=weight_matrix, vbias=vbias, hbias=hbias, device=device, dtype=dtype
    )
