from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from rbms.bm.implement import _mask_self_couplings, _sample_one_visible_potts
from rbms.bm.utils import get_freq_single_point, get_freq_two_points
from rbms.classes import EBM
from rbms.custom_fn import check_keys_dict, one_hot
from rbms.dataset.dataset_class import RBMDataset


class PBM(EBM):
    """Potts Boltzmann Machine (bmDCA)"""

    visible_type: str = "categorical"

    def __init__(
        self,
        weight_matrix: Tensor,
        bias: Tensor,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        pseudo_count: float = 0.0,
    ):
        if device is None:
            device = weight_matrix.device
        if dtype is None:
            dtype = weight_matrix.dtype
        self.device = device
        self.dtype = dtype
        self.weight_matrix = weight_matrix.to(device=self.device, dtype=self.dtype)
        self.bias = bias.to(device=self.device, dtype=self.dtype)
        # Pseudo count applied to the data statistics in the gradient
        self.pseudo_count = pseudo_count
        self.name = "PBM"
        self.flags = []

    def __add__(self, other):
        """Add the parameters of two EBMs. Useful for interpolation"""
        return PBM(
            weight_matrix=self.weight_matrix + other.weight_matrix,
            bias=self.bias + other.bias,
            pseudo_count=self.pseudo_count,
        )

    def __mul__(self, other):
        """Multiplies the parameters of the EBM by a float."""
        return PBM(
            weight_matrix=self.weight_matrix * other,
            bias=self.bias * other,
            pseudo_count=self.pseudo_count,
        )

    def sample_one_visible(self, chains: dict[str, Tensor], beta: float = 1.0):
        one_hot_gen = one_hot(chains["visible"].long(), self.num_states)
        res = _sample_one_visible_potts(one_hot_gen, self.weight_matrix, self.bias, beta)
        chains["visible"] = res.argmax(-1).to(self.dtype)
        return chains

    def sample_visibles(
        self, chains: dict[str, Tensor], beta: float = 1.0
    ) -> dict[str, Tensor]:
        """Sample L randomly selected .

        Args:
            chains (dict[str, Tensor]): The parallel chains used for sampling.
            beta (float, optional): The inverse temperature. Defaults to 1.0.

        Returns:
            dict[str, Tensor]: The updated chains with sampled hidden states.
        """
        one_hot_gen = one_hot(chains["visible"].long(), self.num_states)
        for _ in range(self.num_visibles):
            one_hot_gen = _sample_one_visible_potts(
                one_hot_gen, self.weight_matrix, self.bias, beta
            )
        chains["visible"] = one_hot_gen.argmax(-1).to(self.dtype)
        return chains

    def compute_energy_visibles(self, v: Tensor) -> Tensor:
        """Returns the marginalized energy of the model computed on the visible configurations

        Args:
            v (Tensor): Visible configurations

        Returns:
            Tensor: The computed energy.
        """
        L, q = self.bias.shape
        batch_size = v.shape[0]
        x_flat = (
            one_hot(v.long(), num_classes=self.num_states)
            .to(self.dtype)
            .view(batch_size, -1)
        )
        bias_flat = self.bias.view(-1)
        couplings_flat = self.weight_matrix.reshape(L * q, L * q)
        bias_term = x_flat @ bias_flat
        coupling_term = torch.sum(x_flat * (x_flat @ couplings_flat), dim=1)
        energy = -bias_term - 0.5 * coupling_term

        return energy

    def init_chains(
        self,
        num_samples: int,
        weights: Tensor | None = None,
        start_v: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Initialize Markov chains by sampling the profile model (local fields only).

        Args:
            num_samples (int): The number of samples to initialize.
            weights (Tensor, optional): The weights of the chains. Defaults to None (all ones).
            start_v (Tensor, optional): The initial visible states. Defaults to None.

        Returns:
            dict[str, Tensor]: The initialized Markov chain.

        Notes:
            - If start_v is specified, its number of samples will override the num_samples argument.
        """
        if num_samples <= 0:
            if start_v is not None:
                num_samples = start_v.shape[0]
            else:
                raise ValueError(f"Got negative num_samples arg: {num_samples}")

        # Marginals of the profile model, shape (num_visibles, num_states)
        profile = torch.softmax(self.bias, dim=-1)
        if start_v is None:
            v = (
                torch.multinomial(profile, num_samples=num_samples, replacement=True)
                .T.contiguous()
                .to(self.dtype)
            )
        else:
            v = start_v.to(device=self.device, dtype=self.dtype)
        mv = profile.expand(v.shape[0], -1, -1)

        # Initialize chains
        if weights is None:
            weights = torch.ones(v.shape[0], device=v.device, dtype=v.dtype)
        return dict(
            visible=v,
            visible_mag=mv,
            weights=weights,
        )

    def compute_gradient(
        self,
        data: dict[str, Tensor],
        chains: dict[str, Tensor],
        centered: bool = True,
    ) -> None:
        """Compute the gradient for each of the parameters and attach it.

        Args:
            data (dict[str, Tensor]): The data state.
            chains (dict[str, Tensor]): The parallel chains used for gradient computation.
            centered (bool, optional): Unused. Defaults to True.

        Notes:
            - The pseudo count is only applied to the data statistics, so that the model is
              pulled towards frequencies that are non-zero for every state.
            - The couplings between the states of a same site are not trained.
        """
        one_hot_data = one_hot(data["visible"].long(), self.num_states).to(self.dtype)
        one_hot_gen = one_hot(chains["visible"].long(), self.num_states).to(self.dtype)
        fi_data = get_freq_single_point(one_hot_data, data["weights"], self.pseudo_count)
        fi_gen = get_freq_single_point(one_hot_gen, chains["weights"])
        fij_data = get_freq_two_points(one_hot_data, data["weights"], self.pseudo_count)
        fij_gen = get_freq_two_points(one_hot_gen, chains["weights"])
        self.bias.grad = fi_data - fi_gen
        self.weight_matrix.grad = fij_data - fij_gen
        _mask_self_couplings(self.weight_matrix.grad)

    def parameters(self) -> list[Tensor]:
        """Returns a list containing the parameters of the RBM.

        Returns:
            List[Tensor]: A list containing the weight matrix, visible bias, and hidden bias.
        """
        return [self.weight_matrix, self.bias]

    def named_parameters(self) -> dict[str, np.ndarray]:
        return {
            "weight_matrix": self.weight_matrix.cpu().numpy(),
            "bias": self.bias.cpu().numpy(),
            "pseudo_count": np.asarray(self.pseudo_count),
        }

    @staticmethod
    def set_named_parameters(
        named_params: dict[str, np.ndarray],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> EBM:
        names = ["bias", "weight_matrix"]
        check_keys_dict(d=named_params, names=names)
        params = PBM(
            weight_matrix=torch.from_numpy(named_params.pop("weight_matrix")).to(
                device=device, dtype=dtype
            ),
            bias=torch.from_numpy(named_params.pop("bias")).to(
                device=device, dtype=dtype
            ),
            # Archives saved before the pseudo count was stored have none
            pseudo_count=float(named_params.pop("pseudo_count", 0.0)),
        )
        if len(named_params.keys()) > 0:
            raise ValueError(
                f"Too many keys in params dictionary. Remaining keys: {named_params.keys()}"
            )
        return params

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Move the parameters to the specified device and/or convert them to the specified data type.

        Args:
            device (Optional[torch.device], optional): The device to move the parameters to.
                Defaults to None.
            dtype (Optional[torch.dtype], optional): The data type to convert the parameters to.
                Defaults to None.

        Returns:
            RBM: The modified RBM instance.
        """
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.weight_matrix = self.weight_matrix.to(device=self.device, dtype=self.dtype)
        self.bias = self.bias.to(device=self.device, dtype=self.dtype)
        return self

    def clone(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> EBM:
        """Create a clone of the RBM instance.

        Args:
            device (Optional[torch.device], optional): The device for the cloned parameters.
                Defaults to the current device.
            dtype (Optional[torch.dtype], optional): The data type for the cloned parameters.
                Defaults to the current data type.

        Returns:
            RBM: A new RBM instance with cloned parameters.
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        return PBM(
            self.weight_matrix.clone(),
            self.bias.clone(),
            device=device,
            dtype=dtype,
            pseudo_count=self.pseudo_count,
        )

    @staticmethod
    def init_parameters(
        num_hiddens: int,
        dataset: RBMDataset,
        device: torch.device | str,
        dtype: torch.dtype,
        var_init: float = 1e-4,
        pseudo_count: float | None = None,
    ) -> PBM:
        """Initialize the parameters of the Potts BM.

        Args:
            num_hiddens (int): Unused, the model has no hidden units.
            dataset (RBMDataset): Training dataset.
            device (torch.device): PyTorch device for the parameters.
            dtype (torch.dtype): PyTorch dtype for the parameters.
            var_init (float, optional): Unused, the couplings are initialized to 0.
            pseudo_count (float, optional): Pseudo count applied to the data statistics.
                Defaults to 1 / Meff, with Meff the effective number of samples of the dataset.

        Notes:
            - The number of visible units is induced from the dataset provided.
            - Biases are set to the log of the frequencies of the dataset, with the pseudo count.
            - Couplings are set to 0.
        """
        if pseudo_count is None:
            pseudo_count = 1.0 / dataset.weight_sum
        fi = get_freq_single_point(
            one_hot(dataset.data.long(), dataset.get_num_states()).to(
                device=device, dtype=dtype
            ),
            dataset.weights.to(device=device, dtype=dtype),
            pseudo_count,
        )
        num_visibles, num_states = fi.shape
        return PBM(
            weight_matrix=torch.zeros(
                (num_visibles, num_states, num_visibles, num_states),
                device=device,
                dtype=dtype,
            ),
            bias=torch.log(fi),
            device=device,
            dtype=dtype,
            pseudo_count=pseudo_count,
        )

    @property
    def num_visibles(self) -> int:
        """Number of visible units"""
        return self.weight_matrix.shape[0]

    @property
    def ref_log_z(self) -> float:
        """Reference log partition function with weights set to 0 (except for the visible bias)."""
        return torch.logsumexp(self.bias, dim=1).sum().item()

    def independent_model(self) -> PBM:
        """Independent model where only local fields are preserved."""
        return PBM(
            torch.zeros_like(self.weight_matrix),
            self.bias.clone(),
            device=self.device,
            dtype=self.dtype,
            pseudo_count=self.pseudo_count,
        )

    def sample_state(
        self, chains: dict[str, Tensor], n_steps: int, beta: float = 1.0
    ) -> dict[str, Tensor]:
        """Sample the model for n_steps

        Args:
            chains (): The starting position of the chains.
            n_steps (int): The number of sampling steps.
            beta (float, optional): The inverse temperature. Defaults to 1.0

        Returns:
            dict[str, Tensor]: The updated chains after n_steps of sampling.
        """
        chains_mutate = {
            "visible": chains["visible"].clone(),
            "weights": chains["weights"].clone(),
        }  # avoids to modify the chains inplace

        for _ in torch.arange(n_steps):
            chains_mutate = self.sample_visibles(chains_mutate, beta)

        return chains_mutate

    def get_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        return metrics

    def pre_grad_update(self) -> None:
        pass

    def post_grad_update(self) -> None:
        # The Gibbs sampler assumes no coupling between the states of a same site
        _mask_self_couplings(self.weight_matrix)

    @property
    def effective_number_variables(self) -> float:
        return 1

    @property
    def num_states(self) -> int:
        return self.weight_matrix.shape[1]
