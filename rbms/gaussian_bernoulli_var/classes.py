from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from rbms.classes import RBM
from rbms.custom_fn import check_keys_dict
from rbms.gaussian_bernoulli_var.implement import (
    _compute_energy,
    _compute_energy_hiddens,
    _compute_energy_visibles,
    _compute_gradient,
    _enforce_positive_precision,
    _init_chains,
    _init_parameters,
    _sample_hiddens,
    _sample_visibles,
)

PARAM_NAMES = [
    "weight_matrix",
    "weight_matrix_2",
    "vbias",
    "hbias",
    "precision",
    "min_precision",
]


class GBRBM_VAR(RBM):
    """Parameters of the Gaussian-Bernoulli RBM with trainable visible variances.

    Gaussian visibles and binary {0, 1} hiddens. The precision of each visible,
    P_i(h) = precision_i - sum_a W2_ia h_a, is trained and depends on the hidden
    configuration through `weight_matrix_2` (coupling between v^2 / 2 and h).

    After each gradient step, `precision` is raised if needed so that
    P_i(h) >= `min_precision` for all h (otherwise p(v|h) is not normalizable).
    """

    visible_type: str = "continuous"

    def __init__(
        self,
        weight_matrix: Tensor,
        weight_matrix_2: Tensor,
        vbias: Tensor,
        hbias: Tensor,
        precision: Tensor,
        min_precision: Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the parameters of the Gaussian-Bernoulli RBM with trainable variances.

        Args:
            weight_matrix (Tensor): Coupling between v and h.
            weight_matrix_2 (Tensor): Coupling between v^2 / 2 and h.
            vbias (Tensor): The visible bias of the RBM.
            hbias (Tensor): The hidden bias of the RBM.
            precision (Tensor): Precision (inverse variance) of the visibles when h = 0.
            min_precision (Tensor, optional): Lower bound on the conditional precision
                of the visibles. Defaults to 1e-1 * precision.
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
        self.weight_matrix_2 = weight_matrix_2.to(device=self.device, dtype=self.dtype)
        self.vbias = vbias.to(device=self.device, dtype=self.dtype)
        self.hbias = hbias.to(device=self.device, dtype=self.dtype)
        self.precision = precision.to(device=self.device, dtype=self.dtype)
        if min_precision is None:
            min_precision = 1e-1 * self.precision
        self.min_precision = min_precision.to(device=self.device, dtype=self.dtype)
        self.name = "GBRBM_VAR"
        self.flags = []

    def _with_trained(self, weight_matrix, weight_matrix_2, vbias, hbias, precision):
        # New instance sharing the non-trained attributes
        return GBRBM_VAR(
            weight_matrix=weight_matrix,
            weight_matrix_2=weight_matrix_2,
            vbias=vbias,
            hbias=hbias,
            precision=precision,
            min_precision=self.min_precision,
        )

    def __add__(self, other):
        return self._with_trained(
            weight_matrix=self.weight_matrix + other.weight_matrix,
            weight_matrix_2=self.weight_matrix_2 + other.weight_matrix_2,
            vbias=self.vbias + other.vbias,
            hbias=self.hbias + other.hbias,
            precision=self.precision + other.precision,
        )

    def __mul__(self, other):
        return self._with_trained(
            weight_matrix=self.weight_matrix * other,
            weight_matrix_2=self.weight_matrix_2 * other,
            vbias=self.vbias * other,
            hbias=self.hbias * other,
            precision=self.precision * other,
        )

    def clone(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        return GBRBM_VAR(
            weight_matrix=self.weight_matrix.clone(),
            weight_matrix_2=self.weight_matrix_2.clone(),
            vbias=self.vbias.clone(),
            hbias=self.hbias.clone(),
            precision=self.precision.clone(),
            min_precision=self.min_precision.clone(),
            device=device,
            dtype=dtype,
        )

    def compute_energy(self, v: Tensor, h: Tensor) -> Tensor:
        return _compute_energy(
            v=v,
            h=h,
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
            weight_matrix_2=self.weight_matrix_2,
            precision=self.precision,
        )

    def compute_energy_hiddens(self, h: Tensor) -> Tensor:
        return _compute_energy_hiddens(
            h=h,
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
            weight_matrix_2=self.weight_matrix_2,
            precision=self.precision,
        )

    def compute_energy_visibles(self, chains: dict[str, Tensor]) -> Tensor:
        return _compute_energy_visibles(
            v=chains["visible"],
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
            weight_matrix_2=self.weight_matrix_2,
            precision=self.precision,
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
            weight_matrix_2=self.weight_matrix_2,
            precision=self.precision,
            centered=centered,
        )

    def independent_model(self):
        return self._with_trained(
            weight_matrix=torch.zeros_like(self.weight_matrix),
            weight_matrix_2=torch.zeros_like(self.weight_matrix_2),
            vbias=self.vbias,
            hbias=torch.zeros_like(self.hbias),
            precision=self.precision,
        )

    def init_chains(self, num_samples, weights=None, start_v=None):
        visible, hidden, mean_visible, mean_hidden = _init_chains(
            num_samples=num_samples,
            weight_matrix=self.weight_matrix,
            weight_matrix_2=self.weight_matrix_2,
            vbias=self.vbias,
            hbias=self.hbias,
            precision=self.precision,
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
        (
            vbias,
            hbias,
            weight_matrix,
            weight_matrix_2,
            precision,
            min_precision,
        ) = _init_parameters(
            num_hiddens=num_hiddens,
            data=data,
            device=device,
            dtype=dtype,
            var_init=var_init,
        )
        return GBRBM_VAR(
            weight_matrix=weight_matrix,
            weight_matrix_2=weight_matrix_2,
            vbias=vbias,
            hbias=hbias,
            precision=precision,
            min_precision=min_precision,
        )

    def named_parameters(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k).cpu().numpy() for k in PARAM_NAMES}

    @property
    def num_hiddens(self):
        return self.hbias.shape[0]

    @property
    def num_visibles(self):
        return self.vbias.shape[0]

    def parameters(self) -> list[Tensor]:
        # min_precision is not trained
        return [
            self.weight_matrix,
            self.weight_matrix_2,
            self.vbias,
            self.hbias,
            self.precision,
        ]

    @property
    def ref_log_z(self):
        return (
            0.5
            * (
                self.vbias.square() / self.precision
                + np.log(2 * np.pi)
                - torch.log(self.precision)
            ).sum()
            + self.num_hiddens * np.log(2)
        ).item()

    def sample_hiddens(self, chains: dict[str, Tensor], beta=1) -> dict[str, Tensor]:
        chains["hidden"], chains["hidden_mag"] = _sample_hiddens(
            v=chains["visible"],
            weight_matrix=self.weight_matrix,
            weight_matrix_2=self.weight_matrix_2,
            hbias=self.hbias,
            beta=beta,
        )
        return chains

    def sample_visibles(self, chains: dict[str, Tensor], beta=1) -> dict[str, Tensor]:
        chains["visible"], chains["visible_mag"] = _sample_visibles(
            h=chains["hidden"],
            weight_matrix=self.weight_matrix,
            weight_matrix_2=self.weight_matrix_2,
            vbias=self.vbias,
            precision=self.precision,
            beta=beta,
        )
        return chains

    @staticmethod
    def set_named_parameters(
        named_params: dict[str, np.ndarray],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> GBRBM_VAR:
        check_keys_dict(d=named_params, names=PARAM_NAMES)
        params = GBRBM_VAR(
            **{
                k: torch.from_numpy(named_params.pop(k)).to(device=device, dtype=dtype)
                for k in PARAM_NAMES
            }
        )
        if len(named_params.keys()) > 0:
            raise ValueError(
                f"Too many keys in params dictionary. Remaining keys: {named_params.keys()}"
            )
        return params

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        for k in PARAM_NAMES:
            setattr(self, k, getattr(self, k).to(device=self.device, dtype=self.dtype))
        return self

    def get_metrics(self, metrics):
        return metrics

    def post_grad_update(self):
        _enforce_positive_precision(
            precision=self.precision,
            weight_matrix_2=self.weight_matrix_2,
            min_precision=self.min_precision,
        )

    def pre_grad_update(self):
        pass
