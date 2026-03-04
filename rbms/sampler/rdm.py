import numpy as np
import torch
from torch import Tensor

from rbms.classes import EBM, Sampler


class RDM(Sampler):
    def __init__(
        self, params: EBM, num_chains: int, num_steps: int, beta: float = 1, **kwargs
    ):
        self.name = "RDM"
        self.params = params
        self.beta = beta
        self.num_chains = num_chains
        self.num_steps = num_steps
        self.chains = self.params.init_chains(num_chains)
        self.flags = []

    def sample(self, num_steps: int | None, **kwargs):
        chains = self.params.init_chains(num_samples=self.num_chains)
        chains = self.params.sample_state(
            chains=chains, n_steps=self.num_steps, beta=self.beta
        )
        return chains

    def get_conf_grad(self, batch: Tensor):
        self.sample(num_steps=None)
        return self.chains

    @torch.compiler.disable
    def named_parameters(self):
        params_dict = self.params.named_parameters()
        params_dict["model_type"] = np.asarray(self.params.name, dtype="T")
        params_dict["sampler_type"] = np.asarray(self.name, dtype="T")
        params_dict["num_chains"] = np.asarray(self.num_chains)
        params_dict["beta"] = np.asarray(self.beta)
        params_dict["num_steps"] = np.asarray(self.num_steps)
        match self.params.visible_type:
            case "bernoulli":
                chains_save = self.chains["visible"].bool().cpu().numpy()
            case "ising" | "categorical":
                chains_save = self.chains["visible"].to(torch.int16).cpu().numpy()
            case _:
                chains_save = self.chains["visible"].cpu().numpy()
        params_dict["parallel_chains"] = chains_save
        return params_dict

    @staticmethod
    def set_named_parameters(
        named_params: dict[str, np.ndarray],
        map_model: dict[str, type[EBM]],
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        names = ["model_type", "num_chains", "beta", "num_steps"]
        for k in names:
            if k not in named_params.keys():
                raise ValueError(
                    f"""Dictionary params missing key '{k}'\n Provided keys : {named_params.keys()}\n Expected keys: {names}"""
                )
        model_type = str(named_params.pop("model_type"))
        num_chains = int(named_params.pop("num_chains"))
        beta = float(named_params.pop("beta"))
        num_steps = int(named_params.pop("num_steps"))
        chains_visible = torch.from_numpy(named_params.pop("parallel_chains")).to(
            device=device, dtype=dtype
        )
        # There should only remain the keys for the model loading
        params = map_model[model_type].set_named_parameters(
            named_params=named_params, device=device, dtype=dtype
        )
        chains = params.init_chains(chains_visible.shape[0], start_v=chains_visible)
        sampler = RDM(
            params=params, num_chains=num_chains, num_steps=num_steps, beta=beta
        )
        sampler.chains = chains
        return sampler

    def post_grad_update(self, params: EBM):
        self.params = params

    def get_metrics_display(self, metrics, **kwargs):
        return metrics

    def get_metrics_save(self):
        return None

    def pre_grad_update(self):
        pass
