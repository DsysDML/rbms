from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class MLPEnergy(torch.nn.Module):
    """Binary visible-state energy represented by an MLP.

    The module maps a batch of visible configurations v in {0, 1}^N to one
    scalar energy per sample. A fixed visible field h can be added as

        E(v) = E_MLP(v) - v^T h,

    which is the Bernoulli analogue of the external field in an Ising model.
    """

    def __init__(
        self,
        num_visibles: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        visible_field: Tensor | None = None,
    ):
        super().__init__()
        self.num_visibles = num_visibles
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if visible_field is None:
            visible_field = torch.zeros(num_visibles)
        self.visible_field = torch.nn.Parameter(torch.zeros_like(visible_field))

        layers = []
        in_dim = num_visibles
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.SiLU())
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, 1))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, v: Tensor) -> Tensor:
        return self.net(v).view(-1) - v @ self.visible_field


def get_visible_field_from_data(
    data: Tensor,
    weights: Tensor | None = None,
    eps: float = 1e-4,
) -> Tensor:
    """Return h_i = log(p_i / (1 - p_i)) for binary variables v_i in {0, 1}."""

    if weights is None:
        p = data.mean(dim=0)
    else:
        weights = weights.to(device=data.device, dtype=data.dtype).view(-1)
        p = (data * weights[:, None]).sum(dim=0) / weights.sum()

    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


class RBMEnergy(torch.nn.Module):
    """Bernoulli-Bernoulli RBM visible free energy.

    The joint RBM energy is

        E(v, h) = -v^T W h - a^T v - b^T h,

    with binary visible units v and binary hidden units h.

    Marginalizing over h gives the visible energy

        E(v) = -a^T v - sum_j log(1 + exp(b_j + (v W)_j)).

    This class implements exactly that visible energy.
    """

    def __init__(
        self,
        num_visibles: int,
        hidden_dim: int = 256,
        weight_scale: float = 1e-2,
        visible_bias: Tensor | None = None,
    ):
        super().__init__()
        self.num_visibles = num_visibles
        self.hidden_dim = hidden_dim

        self.weight = torch.nn.Parameter(
            weight_scale * torch.randn(num_visibles, hidden_dim)
        )

        if visible_bias is None:
            visible_bias = torch.zeros(num_visibles)

        self.vbias = torch.nn.Parameter(visible_bias.clone())
        self.hbias = torch.nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, v: Tensor) -> Tensor:
        hidden_field = v @ self.weight + self.hbias
        hidden_term = torch.nn.functional.softplus(hidden_field).sum(dim=1)
        visible_term = v @ self.vbias
        return -visible_term - hidden_term


ENERGY_MAP: dict[str, type[torch.nn.Module]] = {
    "mlp": MLPEnergy,
    "rbm": RBMEnergy,
}


def build_energy(
    energy_type: str,
    num_visibles: int,
    device: torch.device | str,
    dtype: torch.dtype,
    **energy_kwargs,
) -> torch.nn.Module:
    if energy_type not in ENERGY_MAP:
        raise ValueError(
            f"Unknown energy type '{energy_type}'. "
            f"Available energy types: {list(ENERGY_MAP.keys())}."
        )

    energy = ENERGY_MAP[energy_type](
        num_visibles=num_visibles,
        **energy_kwargs,
    )
    return energy.to(device=device, dtype=dtype)


def restore_energy(
    named_params: dict[str, np.ndarray],
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Restore an energy module from saved parameter arrays.

    The HDF5 archive stores only the energy state_dict. This function identifies
    which energy class produced that state_dict, rebuilds the module, and loads
    the saved tensors.
    """

    energy_type = identify_energy_type(named_params)

    match energy_type:
        case "rbm":
            energy = restore_rbm_energy(named_params)

        case "mlp":
            energy = restore_mlp_energy(named_params)

        case _:
            raise ValueError(
                f"Cannot restore unknown energy type '{energy_type}'. "
                f"Available parameter keys: {list(named_params.keys())}."
            )

    state_dict = {
        name: torch.as_tensor(array, device=device, dtype=dtype)
        for name, array in named_params.items()
    }
    if "visible_field" in energy.state_dict() and "visible_field" not in state_dict:
        state_dict["visible_field"] = torch.zeros(
            energy.num_visibles,
            device=device,
            dtype=dtype,
        )

    energy.load_state_dict(state_dict)
    return energy.to(device=device, dtype=dtype)


def identify_energy_type(named_params: dict[str, np.ndarray]) -> str:
    """Infer the energy type from the state_dict keys."""

    keys = set(named_params)

    match keys:
        case keys if {"weight", "vbias", "hbias"} <= keys:
            return "rbm"

        case keys if any(name.startswith("net.") for name in keys):
            return "mlp"

        case _:
            raise ValueError(
                "Could not identify BEBM energy type from saved parameters. "
                f"Available keys: {list(named_params.keys())}."
            )


def restore_rbm_energy(
    named_params: dict[str, np.ndarray],
) -> RBMEnergy:
    """Rebuild an RBMEnergy from its saved parameter shapes."""

    num_visibles, hidden_dim = named_params["weight"].shape

    return RBMEnergy(
        num_visibles=num_visibles,
        hidden_dim=hidden_dim,
    )


def restore_mlp_energy(
    named_params: dict[str, np.ndarray],
) -> MLPEnergy:
    """Rebuild an MLPEnergy from its saved parameter shapes."""

    weight_keys = sorted(
        [name for name in named_params if name.endswith(".weight")],
        key=lambda name: int(name.split(".")[1]),
    )

    if len(weight_keys) == 0:
        raise ValueError("Cannot restore MLPEnergy without weight tensors.")

    first_weight = named_params[weight_keys[0]]
    num_visibles = first_weight.shape[1]
    hidden_dim = first_weight.shape[0]
    num_layers = len(weight_keys) - 1

    return MLPEnergy(
        num_visibles=num_visibles,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )


class IndependentBernoulliEnergy(torch.nn.Module):
    """Independent Bernoulli visible energy.

    E(v) = -v^T h

    with h_i = log(p_i / (1 - p_i)).
    """

    def __init__(self, visible_field: Tensor):
        super().__init__()
        self.num_visibles = visible_field.shape[0]
        self.register_buffer("visible_field", visible_field.clone())

    def forward(self, v: Tensor) -> Tensor:
        return -v @ self.visible_field



class InterpolatedEnergy(torch.nn.Module):
    """Energy-space interpolation between two binary visible energies.

    For AIS with a generic EBM, the correct bridge is

        E_beta(v) = (1 - beta) E_0(v) + beta E_1(v),

    not an interpolation of neural-network parameters.
    """

    def __init__(
        self,
        energy_0: torch.nn.Module,
        energy_1: torch.nn.Module,
        beta: float,
    ):
        super().__init__()
        self.energy_0 = energy_0
        self.energy_1 = energy_1
        self.beta = beta
        self.num_visibles = energy_1.num_visibles

    def forward(self, v: Tensor) -> Tensor:
        return (1.0 - self.beta) * self.energy_0(v) + self.beta * self.energy_1(v)
