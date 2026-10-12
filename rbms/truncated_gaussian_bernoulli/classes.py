from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from rbms.gaussian_bernoulli.classes import GBRBM
from rbms.truncated_gaussian_bernoulli.implement import (
    _compute_energy_hiddens,
    _init_vbias,
    _log_partition_visibles,
    _sample_truncated_normal,
    _sample_visibles,
)


class TGBRBM(GBRBM):
    """Parameters of the truncated Gaussian-Bernoulli RBM.

    Same energy and parameters as the GBRBM (fixed `sigma`, trained `weight_matrix`,
    `vbias`, `hbias`), but the visibles live in the box [vmin, vmax]: given h, each visible
    follows a Gaussian of location sigma^2 (vbias + W h) and scale sigma truncated to
    [vmin, vmax]. The generated configurations are therefore always inside the range of
    the data.

    With infinite bounds it reduces to the GBRBM.
    """

    _fixed_names: tuple[str, ...] = ("sigma", "vmin", "vmax")

    def __init__(
        self,
        weight_matrix: Tensor,
        vbias: Tensor,
        hbias: Tensor,
        sigma: Tensor,
        vmin: Tensor | None = None,
        vmax: Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the parameters of the truncated Gaussian-Bernoulli RBM.

        Args:
            weight_matrix (Tensor): The weight matrix of the RBM.
            vbias (Tensor): The visible bias of the RBM.
            hbias (Tensor): The hidden bias of the RBM.
            sigma (Tensor): The (fixed) scale of each visible unit.
            vmin (Tensor, optional): Lower bound of the visible variables. Defaults to -inf.
            vmax (Tensor, optional): Upper bound of the visible variables. Defaults to +inf.
            device (Optional[torch.device], optional): The device for the parameters.
                Defaults to the device of `weight_matrix`.
            dtype (Optional[torch.dtype], optional): The data type for the parameters.
                Defaults to the data type of `weight_matrix`.
        """
        super().__init__(
            weight_matrix=weight_matrix,
            vbias=vbias,
            hbias=hbias,
            sigma=sigma,
            device=device,
            dtype=dtype,
        )
        if vmin is None:
            vmin = torch.full_like(self.vbias, -torch.inf)
        if vmax is None:
            vmax = torch.full_like(self.vbias, torch.inf)
        self.vmin = vmin.to(device=self.device, dtype=self.dtype)
        self.vmax = vmax.to(device=self.device, dtype=self.dtype)
        self.name = "TGBRBM"

    def compute_energy_hiddens(self, h: Tensor) -> Tensor:
        return _compute_energy_hiddens(
            h=h,
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
            sigma=self.sigma,
            vmin=self.vmin,
            vmax=self.vmax,
        )

    def sample_visibles(self, chains: dict[str, Tensor], beta=1) -> dict[str, Tensor]:
        chains["visible"], chains["visible_mag"] = _sample_visibles(
            h=chains["hidden"],
            weight_matrix=self.weight_matrix,
            vbias=self.vbias,
            sigma=self.sigma,
            vmin=self.vmin,
            vmax=self.vmax,
            beta=beta,
        )
        return chains

    def init_chains(self, num_samples, weights=None, start_v=None):
        if start_v is None:
            # Sample from the independent model (W = 0)
            mu = (self.sigma.square() * self.vbias).expand(num_samples, -1)
            start_v, _ = _sample_truncated_normal(
                mu=mu, std=self.sigma.expand_as(mu), vmin=self.vmin, vmax=self.vmax
            )
        return super().init_chains(num_samples, weights=weights, start_v=start_v)

    @property
    def ref_log_z(self):
        return (
            _log_partition_visibles(self.vbias, self.sigma, self.vmin, self.vmax).sum()
            + self.num_hiddens * np.log(2)
        ).item()

    @classmethod
    def init_parameters(cls, num_hiddens, dataset, device, dtype, var_init=0.0001):
        params = super().init_parameters(
            num_hiddens=num_hiddens,
            dataset=dataset,
            device=device,
            dtype=dtype,
            var_init=var_init,
        )
        data = dataset.data
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        data = data.to(device=device, dtype=dtype)
        # Common range of the visible variables (e.g. [0, 1] for pixels)
        params.vmin = torch.full_like(params.vbias, data.min().item())
        params.vmax = torch.full_like(params.vbias, data.max().item())
        # Match the mean of the data with the truncated independent model
        params.vbias = _init_vbias(
            data_mean=data.mean(0), sigma=params.sigma, vmin=params.vmin, vmax=params.vmax
        )
        return params
