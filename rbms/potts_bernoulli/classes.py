from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from rbms.classes import RBM
from rbms.potts_bernoulli.implement import (
    _compute_energy,
    _compute_energy_hiddens,
    _compute_energy_visibles,
    _compute_gradient,
    _init_chains,
    _init_parameters,
    _sample_hiddens,
    _sample_visibles,
)


class PBRBM(RBM):
    """Parameters of the Potts-Bernoulli RBM"""

    def __init__(
        self,
        weight_matrix: Tensor,
        vbias: Tensor,
        hbias: Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize the parameters of the Potts-Bernoulli RBM.

        Args:
            weight_matrix (Tensor): The weight matrix of the RBM.
            vbias (Tensor): The visible bias of the RBM.
            hbias (Tensor): The hidden bias of the RBM.
            device (Optional[torch.device], optional): The device for the parameters.
                Defaults to the device of `weight_matrix`.
            dtype (Optional[torch.dtype], optional): The data type for the parameters.
                Defaults to the data type of `weight_matrix`.
        """
        if device is None:
            device = weight_matrix.device
        if dtype is None:
            dtype = weight_matrix.dtype
        self.device = device
        self.dtype = dtype
        self.weight_matrix = weight_matrix.to(device=self.device, dtype=self.dtype)
        self.vbias = vbias.to(device=self.device, dtype=self.dtype)
        self.hbias = hbias.to(device=self.device, dtype=self.dtype)
        self.name = "PBRBM"

    def __add__(self, other):
        return PBRBM(
            weight_matrix=self.weight_matrix + other.weight_matrix,
            vbias=self.vbias + other.vbias,
            hbias=self.hbias + other.hbias,
        )

    def __mul__(self, other):
        return PBRBM(
            weight_matrix=self.weight_matrix * other,
            vbias=self.vbias * other,
            hbias=self.hbias * other,
        )

    @torch.jit.export
    def clone(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        return PBRBM(
            weight_matrix=self.weight_matrix.clone(),
            vbias=self.vbias.clone(),
            hbias=self.hbias.clone(),
            device=device,
            dtype=dtype,
        )

    def compute_energy(self, v, h):
        return _compute_energy(
            v=v,
            h=h,
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
        )

    def compute_energy_hiddens(self, h):
        return _compute_energy_hiddens(
            h=h,
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
        )

    def compute_energy_visibles(self, v):
        return _compute_energy_visibles(
            v=v,
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
        )

    def compute_gradient(
        self, data, chains, centered=True, lambda_l1=0.0, lambda_l2=0.0
    ):
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
            lambda_l1=0.0,
            lambda_l2=lambda_l2,
        )

    def independent_model(self):
        return PBRBM(
            weight_matrix=torch.zeros_like(self.weight_matrix),
            vbias=torch.zeros_like(self.vbias),
            hbias=torch.zeros_like(self.hbias),
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
        params = PBRBM(weight_matrix=weight_matrix, vbias=vbias, hbias=hbias)
        return params

    def named_parameters(self):
        return {
            "weight_matrix": self.weight_matrix,
            "vbias": self.vbias,
            "hbias": self.hbias,
        }

    def num_hiddens(self):
        return self.hbias.shape[0]

    def num_states(self) -> int:
        """Number of colors for the Potts variables"""
        return self.weight_matrix.shape[1]

    def num_visibles(self):
        return self.vbias.shape[0]

    def parameters(self) -> List[Tensor]:
        return [self.weight_matrix, self.vbias, self.hbias]

    def ref_log_z(self):
        return (
            self.num_hiddens() * np.log(2)
            + self.num_visibles() * np.log(self.num_states())
        ).item()

    def sample_hiddens(self, chains, beta=1):
        chains["hidden"], chains["hidden_mag"] = _sample_hiddens(
            chains["visible"], self.weight_matrix, self.hbias, beta=beta
        )
        return chains

    def sample_visibles(self, chains, beta=1):
        chains["visible"], chains["visible_mag"] = _sample_visibles(
            chains["hidden"], self.weight_matrix, self.vbias, beta=beta
        )
        return chains

    @staticmethod
    def set_named_parameters(named_params):
        names = ["vbias", "hbias", "weight_matrix"]
        for k in names:
            if k not in named_params.keys():
                raise ValueError(
                    f"""Dictionary params missing key '{k}'\n Provided keys : {named_params.keys()}\n Expected keys: {names}"""
                )
        params = PBRBM(
            weight_matrix=named_params.pop("weight_matrix"),
            vbias=named_params.pop("vbias"),
            hbias=named_params.pop("hbias"),
        )
        if len(named_params.keys()) > 0:
            raise ValueError(
                f"Too many keys in params dictionary. Remaining keys: {named_params.keys()}"
            )
        return params

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.weight_matrix = self.weight_matrix.to(device=self.device, dtype=self.dtype)
        self.vbias = self.vbias.to(device=self.device, dtype=self.dtype)
        self.hbias = self.hbias.to(device=self.device, dtype=self.dtype)
        return self
