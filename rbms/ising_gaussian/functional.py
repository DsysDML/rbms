from typing import Optional

import numpy as np
import torch
from torch import Tensor

from rbms.ising_gaussian.classes import IGRBM
from rbms.ising_gaussian.implement import (
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
    chains: dict[str, Tensor], params: IGRBM, beta: float = 1.0
) -> dict[str, Tensor]:
    chains["hidden"], chains["hidden_mag"] = _sample_hiddens(
        v=chains["visible"],
        weight_matrix=params.weight_matrix,
        hbias=params.hbias,
        beta=beta,
    )
    return chains


def sample_visibles(
    chains: dict[str, Tensor], params: IGRBM, beta: float = 1.0
) -> dict[str, Tensor]:
    chains["visible"], chains["visible_mag"] = _sample_visibles(
        h=chains["hidden"],
        weight_matrix=params.weight_matrix,
        vbias=params.vbias,
        beta=beta,
    )
    return chains


def compute_energy(v: Tensor, h: Tensor, params: IGRBM) -> Tensor:
    return _compute_energy(
        v=v,
        h=h,
        vbias=params.vbias,
        hbias=params.hbias,
        weight_matrix=params.weight_matrix,
    )


def compute_energy_visibles(v: Tensor, params: IGRBM) -> Tensor:
    return _compute_energy_visibles(
        v=v,
        vbias=params.vbias,
        hbias=params.hbias,
        weight_matrix=params.weight_matrix,
        const=params.const,
    )


def compute_energy_hiddens(h: Tensor, params: IGRBM) -> Tensor:
    return _compute_energy_hiddens(
        h=h,
        vbias=params.vbias,
        hbias=params.hbias,
        weight_matrix=params.weight_matrix,
    )


def compute_gradient(
    data: dict[str, Tensor],
    chains: dict[str, Tensor],
    params: IGRBM,
    centered: bool,
) -> None:
    _compute_gradient(
        v_data=data["visible"],
        mh_data=data["hidden_mag"],
        w_data=data["weights"],
        v_chain=chains["visible"],
        h_chain=chains["hidden_mag"],
        w_chain=chains["weights"],
        vbias=params.vbias,
        hbias=params.hbias,
        weight_matrix=params.weight_matrix,
        centered=centered,
    )


def init_chains(
    num_samples: int,
    params: IGRBM,
    weights: Optional[Tensor] = None,
    start_v: Optional[Tensor] = None,
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
) -> IGRBM:
    data = dataset.data
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(dataset.data).to(device=device, dtype=dtype)
    vbias, hbias, weight_matrix = _init_parameters(
        num_hiddens=num_hiddens, data=data, device=device, dtype=dtype, var_init=var_init
    )
    return IGRBM(
        weight_matrix=weight_matrix, vbias=vbias, hbias=hbias, device=device, dtype=dtype
    )
