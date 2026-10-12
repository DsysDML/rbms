import torch
from torch import Tensor

from rbms.dataset.dataset_class import RBMDataset
from rbms.truncated_gaussian_bernoulli.classes import TGBRBM


def sample_hiddens(
    chains: dict[str, Tensor], params: TGBRBM, beta: float = 1.0
) -> dict[str, Tensor]:
    """Sample h|v (binary hiddens)."""
    return params.sample_hiddens(chains=chains, beta=beta)


def sample_visibles(
    chains: dict[str, Tensor], params: TGBRBM, beta: float = 1.0
) -> dict[str, Tensor]:
    """Sample v|h (truncated Gaussian visibles)."""
    return params.sample_visibles(chains=chains, beta=beta)


def compute_energy(v: Tensor, h: Tensor, params: TGBRBM) -> Tensor:
    return params.compute_energy(v=v, h=h)


def compute_energy_visibles(v: Tensor, params: TGBRBM) -> Tensor:
    """Energy marginalized over h."""
    return params.compute_energy_visibles(chains={"visible": v})


def compute_energy_hiddens(h: Tensor, params: TGBRBM) -> Tensor:
    """Energy marginalized over v."""
    return params.compute_energy_hiddens(h=h)


def compute_gradient(
    data: dict[str, Tensor],
    chains: dict[str, Tensor],
    params: TGBRBM,
    centered: bool,
) -> None:
    params.compute_gradient(data=data, chains=chains, centered=centered)


def init_chains(
    num_samples: int,
    params: TGBRBM,
    weights: Tensor | None = None,
    start_v: Tensor | None = None,
) -> dict[str, Tensor]:
    return params.init_chains(num_samples=num_samples, weights=weights, start_v=start_v)


def init_parameters(
    num_hiddens: int,
    dataset: RBMDataset,
    device: torch.device,
    dtype: torch.dtype,
    var_init: float = 1e-4,
) -> TGBRBM:
    return TGBRBM.init_parameters(
        num_hiddens=num_hiddens,
        dataset=dataset,
        device=device,
        dtype=dtype,
        var_init=var_init,
    )
