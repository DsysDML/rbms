from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class GaussianBaseEnergy(torch.nn.Module):
    """Independent Gaussian reference energy for continuous visibles."""

    def __init__(self, data_mean: Tensor, data_std: Tensor):
        super().__init__()
        self.num_visibles = data_mean.shape[0]
        self.register_buffer("data_mean", data_mean.clone())
        self.register_buffer("data_std", data_std.clone().clamp_min(1e-4))

    def forward(self, x: Tensor) -> Tensor:
        z = (x - self.data_mean) / self.data_std
        return 0.5 * z.square().sum(dim=1)

    @property
    def log_z(self) -> Tensor:
        log_two_pi = torch.log(
            torch.tensor(2.0 * torch.pi, device=self.data_std.device, dtype=self.data_std.dtype)
        )
        return 0.5 * self.num_visibles * log_two_pi + torch.log(self.data_std).sum()


class MLPEnergy(torch.nn.Module):
    """Continuous visible-state energy represented by an MLP plus Gaussian tails."""

    def __init__(
        self,
        num_visibles: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        data_mean: Tensor | None = None,
        data_std: Tensor | None = None,
    ):
        super().__init__()
        self.num_visibles = num_visibles
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if data_mean is None:
            data_mean = torch.zeros(num_visibles)
        if data_std is None:
            data_std = torch.ones(num_visibles)

        self.base = GaussianBaseEnergy(data_mean=data_mean, data_std=data_std)

        layers = []
        in_dim = num_visibles
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.SiLU())
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, 1))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).view(-1) + self.base(x)


ENERGY_MAP: dict[str, type[torch.nn.Module]] = {
    "mlp": MLPEnergy,
    "gaussian": GaussianBaseEnergy,
}


def get_gaussian_base_from_data(
    data: Tensor,
    weights: Tensor | None = None,
    eps: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Return weighted mean and standard deviation for continuous data."""

    if weights is None:
        mean = data.mean(dim=0)
        var = data.var(dim=0, unbiased=False)
    else:
        weights = weights.to(device=data.device, dtype=data.dtype).view(-1)
        norm_weights = weights / weights.sum()
        mean = (data * norm_weights[:, None]).sum(dim=0)
        var = ((data - mean).square() * norm_weights[:, None]).sum(dim=0)

    return mean, var.sqrt().clamp_min(eps)


def build_energy(
    energy_type: str,
    num_visibles: int,
    device: torch.device | str,
    dtype: torch.dtype,
    **energy_kwargs,
) -> torch.nn.Module:
    if energy_type not in ENERGY_MAP:
        raise ValueError(
            f"Unknown continuous EBM energy type '{energy_type}'. "
            f"Available energy types: {list(ENERGY_MAP.keys())}."
        )

    if energy_type == "gaussian":
        energy = GaussianBaseEnergy(
            data_mean=energy_kwargs["data_mean"],
            data_std=energy_kwargs["data_std"],
        )
    else:
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
    energy_type = identify_energy_type(named_params)

    match energy_type:
        case "gaussian":
            energy = restore_gaussian_energy(named_params)
        case "mlp":
            energy = restore_mlp_energy(named_params)
        case _:
            raise ValueError(
                f"Cannot restore unknown continuous energy type '{energy_type}'. "
                f"Available parameter keys: {list(named_params.keys())}."
            )

    state_dict = {
        name: torch.as_tensor(array, device=device, dtype=dtype)
        for name, array in named_params.items()
    }
    energy.load_state_dict(state_dict)
    return energy.to(device=device, dtype=dtype)


def identify_energy_type(named_params: dict[str, np.ndarray]) -> str:
    keys = set(named_params)

    match keys:
        case keys if any(name.startswith("net.") for name in keys):
            return "mlp"
        case keys if {"data_mean", "data_std"} <= keys:
            return "gaussian"
        case _:
            raise ValueError(
                "Could not identify continuous EBM energy type from saved parameters. "
                f"Available keys: {list(named_params.keys())}."
            )


def restore_gaussian_energy(named_params: dict[str, np.ndarray]) -> GaussianBaseEnergy:
    return GaussianBaseEnergy(
        data_mean=torch.as_tensor(named_params["data_mean"]),
        data_std=torch.as_tensor(named_params["data_std"]),
    )


def restore_mlp_energy(named_params: dict[str, np.ndarray]) -> MLPEnergy:
    weight_keys = sorted(
        [name for name in named_params if name.startswith("net.") and name.endswith(".weight")],
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
        data_mean=torch.as_tensor(named_params["base.data_mean"]),
        data_std=torch.as_tensor(named_params["base.data_std"]),
    )
