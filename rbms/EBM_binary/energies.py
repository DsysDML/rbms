from __future__ import annotations

import torch
from torch import Tensor


class MLPEnergy(torch.nn.Module):
    """A binary visible-state energy E_theta(v) represented by an MLP.

    The module maps a batch of visible configurations v in {0, 1}^N to one
    scalar energy per sample.
    """

    def __init__(
        self,
        num_visibles: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_visibles = num_visibles
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        in_dim = num_visibles
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.SiLU())
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, 1))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, v: Tensor) -> Tensor:
        return self.net(v).view(-1)


ENERGY_MAP: dict[str, type[torch.nn.Module]] = {
    "mlp": MLPEnergy,
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
