from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from rbms.classes import RBM
from rbms.custom_fn import check_keys_dict
from rbms.gaussian_bernoulli.implement import (
    _compute_energy,
    _compute_energy_hiddens,
    _compute_energy_visibles,
    _compute_gradient,
    _init_chains,
    _init_parameters,
    _sample_hiddens,
    _sample_visibles,
)


class GBRBM(RBM):
    """Parameters of the Gaussian-Bernoulli RBM.

    Gaussian visibles (unbounded) with fixed (non-trained) standard deviations `sigma`
    and binary {0, 1} hiddens.
    """

    visible_type: str = "continuous"
    # Trained parameters
    _trained_names: tuple[str, ...] = ("weight_matrix", "vbias", "hbias")
    # Fixed (non-trained) attributes saved with the model. Subclasses can extend it.
    _fixed_names: tuple[str, ...] = ("sigma",)

    def __init__(
        self,
        weight_matrix: Tensor,
        vbias: Tensor,
        hbias: Tensor,
        sigma: Tensor,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the parameters of the Gaussian-Bernoulli RBM.

        Args:
            weight_matrix (Tensor): The weight matrix of the RBM.
            vbias (Tensor): The visible bias of the RBM.
            hbias (Tensor): The hidden bias of the RBM.
            sigma (Tensor): The (fixed) standard deviation of each visible unit.
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
        self.sigma = sigma.to(device=self.device, dtype=self.dtype)
        self.name = "GBRBM"
        self.flags = []

    def _fixed(self) -> dict[str, Tensor]:
        return {k: getattr(self, k) for k in self._fixed_names}

    def __add__(self, other):
        return type(self)(
            weight_matrix=self.weight_matrix + other.weight_matrix,
            vbias=self.vbias + other.vbias,
            hbias=self.hbias + other.hbias,
            **self._fixed(),
        )

    def __mul__(self, other):
        return type(self)(
            weight_matrix=self.weight_matrix * other,
            vbias=self.vbias * other,
            hbias=self.hbias * other,
            **self._fixed(),
        )

    def clone(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        return type(self)(
            weight_matrix=self.weight_matrix.clone(),
            vbias=self.vbias.clone(),
            hbias=self.hbias.clone(),
            **{k: v.clone() for k, v in self._fixed().items()},
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
            sigma=self.sigma,
        )

    def compute_energy_hiddens(self, h: Tensor) -> Tensor:
        return _compute_energy_hiddens(
            h=h,
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
            sigma=self.sigma,
        )

    def compute_energy_visibles(self, chains: dict[str, Tensor]) -> Tensor:
        return _compute_energy_visibles(
            v=chains["visible"],
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
            sigma=self.sigma,
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
        return type(self)(
            weight_matrix=torch.zeros_like(self.weight_matrix),
            vbias=self.vbias,
            hbias=torch.zeros_like(self.hbias),
            **self._fixed(),
        )

    def init_chains(self, num_samples, weights=None, start_v=None):
        visible, hidden, mean_visible, mean_hidden = _init_chains(
            num_samples=num_samples,
            weight_matrix=self.weight_matrix,
            vbias=self.vbias,
            hbias=self.hbias,
            sigma=self.sigma,
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

    @classmethod
    def init_parameters(cls, num_hiddens, dataset, device, dtype, var_init=0.0001):
        data = dataset.data
        # Convert to torch Tensor if necessary
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(dataset.data).to(device=device, dtype=dtype)
        vbias, hbias, weight_matrix, sigma = _init_parameters(
            num_hiddens=num_hiddens,
            data=data,
            device=device,
            dtype=dtype,
            var_init=var_init,
        )
        return cls(weight_matrix=weight_matrix, vbias=vbias, hbias=hbias, sigma=sigma)

    def named_parameters(self) -> dict[str, np.ndarray]:
        return {
            k: getattr(self, k).cpu().numpy()
            for k in self._trained_names + self._fixed_names
        }

    @property
    def num_hiddens(self):
        return self.hbias.shape[0]

    @property
    def num_visibles(self):
        return self.vbias.shape[0]

    def parameters(self) -> list[Tensor]:
        # sigma (and the other fixed attributes) are not trained
        return [getattr(self, k) for k in self._trained_names]

    @property
    def ref_log_z(self):
        var = self.sigma.square()
        return (
            0.5 * (var * self.vbias.square() + torch.log(2 * np.pi * var)).sum()
            + self.num_hiddens * np.log(2)
        ).item()

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
            sigma=self.sigma,
            beta=beta,
        )
        return chains

    @classmethod
    def set_named_parameters(
        cls,
        named_params: dict[str, np.ndarray],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> GBRBM:
        names = list(cls._trained_names + cls._fixed_names)
        check_keys_dict(d=named_params, names=names)
        params = cls(
            **{
                k: torch.from_numpy(named_params.pop(k)).to(device=device, dtype=dtype)
                for k in names
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
        for k in self._trained_names + self._fixed_names:
            setattr(self, k, getattr(self, k).to(device=self.device, dtype=self.dtype))
        return self

    def get_metrics(self, metrics):
        return metrics

    def post_grad_update(self):
        pass

    def pre_grad_update(self):
        pass
