from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import Tensor

from rbms.classes import RBM
from rbms.custom_fn import check_keys_dict, log2cosh
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


class IGRBM(RBM):
    """Ising-Gaussian RBM with fixed hidden variance = 1/Nv, +- 1 visibles, without any bias"""

    visible_type: str = "ising"

    def __init__(
        self,
        weight_matrix: Tensor,
        vbias: Tensor,
        hbias: Tensor,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        if device is None:
            device = weight_matrix.device
        if dtype is None:
            dtype = weight_matrix.dtype

        self.device, self.dtype = device, dtype
        self.weight_matrix = weight_matrix.to(device=self.device, dtype=self.dtype)
        self.vbias = vbias.to(device=self.device, dtype=self.dtype)
        self.hbias = hbias.to(device=self.device, dtype=self.dtype)

        log_two_pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=dtype, device=device))
        const = (
            0.5
            * float(self.weight_matrix.shape[1])
            * (
                -torch.log(
                    torch.tensor(
                        float(self.weight_matrix.shape[0]), dtype=dtype, device=device
                    )
                )
                + log_two_pi
            )
        )
        self.const = const
        self.name = "IGRBM"
        self.flags = []

    def __add__(self, other):
        return IGRBM(
            weight_matrix=self.weight_matrix + other.weight_matrix,
            vbias=self.vbias + other.vbias,
            hbias=self.hbias + other.hbias,
            device=self.device,
            dtype=self.dtype,
        )

    def __mul__(self, other):
        return IGRBM(
            weight_matrix=self.weight_matrix * other,
            vbias=self.vbias * other,
            hbias=self.hbias * other,
            device=self.device,
            dtype=self.dtype,
        )

    def clone(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        return IGRBM(
            weight_matrix=self.weight_matrix.clone(),
            vbias=self.vbias.clone(),
            hbias=self.hbias.clone(),
            device=device,
            dtype=dtype,
        )

    def compute_energy(self, v: Tensor, h: Tensor) -> Tensor:
        return _compute_energy(
            v=v, h=h, vbias=self.vbias, hbias=self.hbias, weight_matrix=self.weight_matrix
        )

    def compute_energy_hiddens(self, h: Tensor) -> Tensor:
        return _compute_energy_hiddens(
            h=h, vbias=self.vbias, hbias=self.hbias, weight_matrix=self.weight_matrix
        )

    def compute_energy_visibles(self, v: Tensor) -> Tensor:
        return _compute_energy_visibles(
            v=v,
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
            const=self.const,
        )

    def compute_gradient(self, data, chains, centered=True):
        _compute_gradient(
            v_data=data["visible"],
            mh_data=data["hidden_mag"],
            w_data=data["weights"],
            v_chain=chains["visible"],
            h_chain=chains["hidden_mag"],
            w_chain=chains["weights"],
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
            centered=centered,
        )

    def independent_model(self):
        return IGRBM(
            weight_matrix=torch.zeros_like(self.weight_matrix),
            vbias=self.vbias,
            hbias=self.hbias,  # torch.zeros_like(self.hbias),
            device=self.device,
            dtype=self.dtype,
        )

    def init_chains(self, num_samples, weights=None, start_v=None):
        visible, hidden, mean_visible, mean_hidden = _init_chains(
            num_samples=num_samples,
            weight_matrix=self.weight_matrix,
            hbias=self.hbias,
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

    @staticmethod
    def init_parameters(num_hiddens, dataset, device, dtype, var_init=0.0001):
        data = dataset.data
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(dataset.data).to(device=device, dtype=dtype)
        vbias, hbias, weight_matrix = _init_parameters(
            num_hiddens=num_hiddens,
            data=data,
            device=device,
            dtype=dtype,
            var_init=var_init,
        )
        return IGRBM(
            weight_matrix=weight_matrix,
            vbias=vbias,
            hbias=hbias,
            device=device,
            dtype=dtype,
        )

    def named_parameters(self):
        return {
            "weight_matrix": self.weight_matrix.cpu().numpy(),
            "vbias": self.vbias.cpu().numpy(),
            "hbias": self.hbias.cpu().numpy(),
        }

    @property
    def num_hiddens(self):
        return self.hbias.shape[0]

    @property
    def num_visibles(self):
        return self.vbias.shape[0]

    def parameters(self) -> List[Tensor]:
        return [self.weight_matrix, self.vbias, self.hbias]

    @property
    def ref_log_z(self):
        K = self.num_hiddens
        # logZ_v = torch.log1p(torch.exp(self.vbias)).sum()
        logZ_v = log2cosh(self.vbias).sum()
        quad = 0.5 * torch.dot(self.hbias, self.hbias) / float(self.num_visibles)
        log_norm = 0.5 * K * np.log(2.0 * np.pi) - 0.5 * K * np.log(
            float(self.num_visibles)
        )
        return (logZ_v + quad + log_norm).item()

    def sample_hiddens(self, chains: dict[str, Tensor], beta=1) -> dict[str, Tensor]:
        chains["hidden"], chains["hidden_mag"] = _sample_hiddens(
            v=chains["visible"],
            weight_matrix=self.weight_matrix,
            hbias=self.hbias,
            beta=beta,
        )
        return chains

    def sample_visibles(self, chains: dict[str, Tensor], beta=1) -> dict[str, Tensor]:
        chains["visible"], chains["visible_mag"] = _sample_visibles(
            h=chains["hidden"],
            weight_matrix=self.weight_matrix,
            vbias=self.vbias,
            beta=beta,
        )
        return chains

    @staticmethod
    def set_named_parameters(
        named_params: dict[str, np.ndarray],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> IGRBM:
        names = ["vbias", "hbias", "weight_matrix"]
        for k in names:
            if k not in named_params:
                raise ValueError(
                    f"""Dictionary params missing key '{k}'\n Provided keys : {named_params.keys()}\n Expected keys: {names}"""
                )
        params = IGRBM(
            weight_matrix=torch.from_numpy(named_params.pop("weight_matrix")).to(
                device=device, dtype=dtype
            ),
            vbias=torch.from_numpy(named_params.pop("vbias")).to(
                device=device, dtype=dtype
            ),
            hbias=torch.from_numpy(named_params.pop("hbias")).to(
                device=device, dtype=dtype
            ),
        )
        if len(named_params) > 0:
            raise ValueError(
                f"Too many keys in params dictionary. Remaining keys: {named_params.keys()}"
            )
        return params

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "IGRBM":
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.weight_matrix = self.weight_matrix.to(device=self.device, dtype=self.dtype)
        self.vbias = self.vbias.to(device=self.device, dtype=self.dtype)
        self.hbias = self.hbias.to(device=self.device, dtype=self.dtype)
        return self

    def get_metrics(self, metrics):
        return metrics

    def post_grad_update(self):
        pass

    def pre_grad_update(self):
        pass
