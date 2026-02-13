import h5py
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
        self.chains = None
        self.flags = []

    def sample(self, batch: Tensor):
        chains = self.params.init_chains(num_samples=self.num_chains)
        chains = self.params.sample_state(
            chains=chains, n_steps=self.num_steps, beta=self.beta
        )
        return chains

    def save(self, filename):
        if self.chains is not None:
            with h5py.File(filename, "a") as f:
                if "parallel_chains" in f.keys():
                    f["parallel_chains"][...] = self.chains["visible"].cpu().numpy()
                else:
                    f["parallel_chains"] = self.chains["visible"].cpu().numpy()

    def named_parameters(self):
        params_dict = self.params.named_parameters()
        params_dict["model_type"] = self.params.name
        params_dict["num_chains"] = self.num_chains
        params_dict["beta"] = self.beta
        params_dict["num_steps"] = self.num_steps
        return params_dict

    def set_named_parameters(named_params, map_model: dict[str, EBM]):
        names = ["model_type", "num_chains", "beta", "num_steps"]
        for k in names:
            if k not in named_params.keys():
                raise ValueError(
                    f"""Dictionary params missing key '{k}'\n Provided keys : {named_params.keys()}\n Expected keys: {names}"""
                )
        model_type = named_params.pop("model_type")
        num_chains = named_params.pop("num_chains")
        beta = named_params.pop("beta")
        num_steps = named_params.pop("num_steps")
        # There should only remain the keys for the model loading
        params = map_model[model_type].set_named_parameters(named_params=named_params)

        return RDM(params=params, num_chains=num_chains, num_steps=num_steps, beta=beta)

    def post_grad_update(self, params: EBM):
        self.params = params
