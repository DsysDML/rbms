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
from rbms.gaussian_bernoulli.implement import _drop_identity_normalization

PARAM_NAMES = [
    "weight_matrix",
    "weight_matrix_2",
    "vbias",
    "hbias",
    "log_precision",
    "min_precision",
    "max_precision",
]


class GBRBM_VAR(RBM):
    """Parameters of the Gaussian-Bernoulli RBM with trainable visible variances.

    Gaussian visibles and binary {0, 1} hiddens. The precision of each visible,
    P_i(h) = precision_i - sum_a W2_ia h_a, is trained and depends on the hidden
    configuration through `weight_matrix_2` (coupling between v^2 / 2 and h).
    `weight_matrix_2` is constrained to be <= 0: the hiddens can only increase the
    precision, and precision_i is the smallest one (h = 0).

    The trained parameter is `log_precision` (precision = exp(log_precision)): a
    gradient step changes the precision by a factor instead of an absolute amount,
    which keeps it stable when the data have a large scale (the gradient of the
    precision involves v^2).

    After each gradient step, W2 is clipped to <= 0 and the precision is raised if
    needed to `min_precision`, so that P_i(h) >= `min_precision` for all h (otherwise
    p(v|h) is not normalizable). The precision is also lowered if needed to
    `max_precision` (the constant visibles would otherwise sharpen forever).
    """

    visible_type: str = "continuous"

    def __init__(
        self,
        weight_matrix: Tensor,
        weight_matrix_2: Tensor,
        vbias: Tensor,
        hbias: Tensor,
        precision: Tensor | None = None,
        min_precision: Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        log_precision: Tensor | None = None,
        max_precision: Tensor | None = None,
    ):
        """Initialize the parameters of the Gaussian-Bernoulli RBM with trainable variances.

        Args:
            weight_matrix (Tensor): Coupling between v and h.
            weight_matrix_2 (Tensor): Coupling between v^2 / 2 and h.
            vbias (Tensor): The visible bias of the RBM.
            hbias (Tensor): The hidden bias of the RBM.
            precision (Tensor, optional): Precision (inverse variance) of the visibles
                when h = 0. Either `precision` or `log_precision` must be given.
            min_precision (Tensor, optional): Lower bound on the conditional precision
                of the visibles. Defaults to 0.25 * precision.
            device (Optional[torch.device], optional): The device for the parameters.
                Defaults to the device of `weight_matrix`.
            dtype (Optional[torch.dtype], optional): The data type for the parameters.
                Defaults to the data type of `weight_matrix`.
            log_precision (Tensor, optional): Log of the precision (trained parameter).
            max_precision (Tensor, optional): Upper bound on the precision at h = 0.
                Defaults to no bound.
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
        if log_precision is None:
            if precision is None:
                raise ValueError("Either precision or log_precision must be given")
            log_precision = torch.log(precision)
        self.log_precision = log_precision.to(device=self.device, dtype=self.dtype)
        if min_precision is None:
            min_precision = 0.25 * self.precision
        self.min_precision = min_precision.to(device=self.device, dtype=self.dtype)
        if max_precision is None:
            max_precision = torch.full_like(self.log_precision, torch.inf)
        self.max_precision = max_precision.to(device=self.device, dtype=self.dtype)
        self.name = "GBRBM_VAR"
        self.flags = []

    @property
    def precision(self) -> Tensor:
        return self.log_precision.exp()

    def _with_trained(self, weight_matrix, weight_matrix_2, vbias, hbias, log_precision):
        # New instance sharing the non-trained attributes
        return GBRBM_VAR(
            weight_matrix=weight_matrix,
            weight_matrix_2=weight_matrix_2,
            vbias=vbias,
            hbias=hbias,
            log_precision=log_precision,
            min_precision=self.min_precision,
            max_precision=self.max_precision,
        )

    def __add__(self, other):
        return self._with_trained(
            weight_matrix=self.weight_matrix + other.weight_matrix,
            weight_matrix_2=self.weight_matrix_2 + other.weight_matrix_2,
            vbias=self.vbias + other.vbias,
            hbias=self.hbias + other.hbias,
            log_precision=self.log_precision + other.log_precision,
        )

    def __mul__(self, other):
        return self._with_trained(
            weight_matrix=self.weight_matrix * other,
            weight_matrix_2=self.weight_matrix_2 * other,
            vbias=self.vbias * other,
            hbias=self.hbias * other,
            log_precision=self.log_precision * other,
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
            log_precision=self.log_precision.clone(),
            min_precision=self.min_precision.clone(),
            max_precision=self.max_precision.clone(),
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
        precision = self.precision
        precision_correction = _compute_gradient(
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
            precision=precision,
            centered=centered,
            weight_mask=self.weight_mask,
        )
        # Chain rule: dL/dlog(precision) = precision * dL/dprecision, except for the
        # centering correction. The correction compensates the change of the precision
        # at the mean hidden configuration due to the step of W2: it is an absolute
        # change of the precision, lr * correction, which in log scale is
        # lr * correction / precision. Multiplying it by the precision instead makes
        # the compensation precision^2 times too large: the precision then grows even
        # when the model is already too narrow, faster and faster (dlambda/dt ~ lambda^2).
        self.log_precision.grad = (
            precision.grad - precision_correction
        ) * precision + precision_correction / precision

    def independent_model(self):
        return self._with_trained(
            weight_matrix=torch.zeros_like(self.weight_matrix),
            weight_matrix_2=torch.zeros_like(self.weight_matrix_2),
            vbias=self.vbias,
            hbias=torch.zeros_like(self.hbias),
            log_precision=self.log_precision,
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
    def init_parameters(
        num_hiddens, dataset, device, dtype, var_init=0.0001, min_precision_factor=0.25
    ):
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
            max_precision,
        ) = _init_parameters(
            num_hiddens=num_hiddens,
            data=data,
            device=device,
            dtype=dtype,
            var_init=var_init,
            min_precision_factor=min_precision_factor,
        )
        return GBRBM_VAR(
            weight_matrix=weight_matrix,
            weight_matrix_2=weight_matrix_2,
            vbias=vbias,
            hbias=hbias,
            precision=precision,
            min_precision=min_precision,
            max_precision=max_precision,
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
        # min_precision and max_precision are not trained
        return [
            self.weight_matrix,
            self.weight_matrix_2,
            self.vbias,
            self.hbias,
            self.log_precision,
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
        if "log_precision" not in named_params and "precision" in named_params:
            # Archives saved when the precision itself was the trained parameter
            named_params["log_precision"] = np.log(named_params.pop("precision"))
        if "max_precision" not in named_params:
            # Archives saved before the upper bound existed: no bound
            named_params["max_precision"] = np.full_like(
                named_params["log_precision"], np.inf
            )
        _drop_identity_normalization(named_params)
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

    @torch.no_grad()
    def post_grad_update(self):
        precision = self.precision
        projected = precision.clone()
        _enforce_positive_precision(
            precision=projected,
            weight_matrix_2=self.weight_matrix_2,
            min_precision=self.min_precision,
            max_precision=self.max_precision,
        )
        # Only the projected entries are changed (no round trip exp/log on the others)
        self.log_precision.copy_(
            torch.where(projected != precision, torch.log(projected), self.log_precision)
        )

    def pre_grad_update(self):
        pass
