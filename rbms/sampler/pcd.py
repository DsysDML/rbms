from __future__ import annotations

import h5py
import numpy as np
import torch
from torch import Tensor

from rbms.classes import EBM, Sampler


class PCD(Sampler):
    def __init__(
        self,
        params: EBM,
        chains: dict[str, Tensor],
        num_steps: int,
        beta: float = 1,
        **kwargs,
    ):
        self.name = "PCD"
        self.chains = chains
        self.params = params
        self.beta = beta
        self.num_steps = num_steps
        self.flags = []

    def get_conf_grad(self, batch: Tensor):
        self.sample(num_steps=None)
        return self.chains

    def sample(self, num_steps: int | None, **kwargs):
        self.chains = self.params.sample_state(
            chains=self.chains, n_steps=self.num_steps, beta=self.beta
        )

    @torch.compiler.disable
    def named_parameters(self):
        params_dict = self.params.named_parameters()
        params_dict["model_type"] = np.asarray(self.params.name, dtype="T")
        params_dict["sampler_type"] = np.asarray(self.name, dtype="T")
        params_dict["parallel_chains"] = self.chains["visible"].cpu().numpy()
        params_dict["beta"] = np.asarray(self.beta)
        params_dict["num_steps"] = np.asarray(self.num_steps)
        return params_dict

    @staticmethod
    def set_named_parameters(
        named_params: dict[str, np.ndarray],
        map_model: dict[str, EBM],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> PCD:
        names = ["model_type", "chains", "beta", "num_steps"]
        for k in names:
            if k not in named_params.keys():
                raise ValueError(
                    f"""Dictionary params missing key '{k}'\n Provided keys : {named_params.keys()}\n Expected keys: {names}"""
                )
        model_type = str(named_params.pop("model_type"))
        chains_visible = torch.from_numpy(named_params.pop("parallel_chains")).to(
            device=device, dtype=dtype
        )
        beta = float(named_params.pop("beta"))
        num_steps = int(named_params.pop("num_steps"))
        # There should only remain the keys for the model loading
        params = map_model[model_type].set_named_parameters(
            named_params=named_params, device=device, dtype=dtype
        )
        chains = params.init_chains(chains_visible.shape[0], start_v=chains_visible)

        return PCD(params=params, chains=chains, num_steps=num_steps, beta=beta)

    def post_grad_update(self, params: EBM):
        self.params = params

    def get_metrics(self, metrics, **kwargs):
        return metrics

    def pre_grad_update(self):
        pass
