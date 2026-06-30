from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from rbms.bm.implement import _sample_one_visible_potts
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
    ):
        if device is None:
            device = weight_matrix.device
        if dtype is None:
            dtype = weight_matrix.dtype
        self.device = device
        self.dtype = dtype
        self.weight_matrix = weight_matrix.to(device=self.device, dtype=self.dtype)
        self.bias = bias.to(device=self.device, dtype=self.dtype)
        self.name = "PBM"
        self.flags = []

    def __add__(self, other):
        """Add the parameters of two EBMs. Useful for interpolation"""
        return PBM(
            weight_matrix=self.weight_matrix + other.weight_matrix,
            bias=self.bias + other.bias,
        )

    def __mul__(self, other):
        """Multiplies the p ofsy a float."""
        return PBM(
            weight_matrix=self.weight_matrix * other,
            bias=self.bias * other,
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

    def compute_energy_visibles(self, chains: dict[str, Tensor]) -> Tensor:
        """Returns the marginalized energy of the model computed on the visible configurations

        Args:
            v (Tensor): Visible configurations

        Returns:
            Tensor: The computed energy.
        """
        v = chains["visible"]
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
        """Initialize a Markov chain for the EBM by sampling a uniform distribution on the visible layer
        and sampling the hidden layer according to the visible one.

        Args:
            num_samples (int): The number of samples to initialize.
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

        if start_v is None:
            # Dummy mean visible
            mv = (
                torch.ones(
                    size=(num_samples, self.num_visibles),
                    device=self.device,
                    dtype=self.dtype,
                )
                / 2
            )
            v = torch.bernoulli(mv)
        else:
            # Dummy mean visible
            mv = torch.ones_like(start_v, device=self.device, dtype=self.dtype) / 2
            v = start_v.to(device=self.device, dtype=self.dtype)

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
            centered (bool, optional): Whether to use centered gradients. Defaults to True.
            lambda_l1 (float, optional): factor for the L1 regularization. Defaults to 0.
            lambda_l2 (float, optional): factor for the L2 regularization. Defaults to 0.
        """
        pseudo_count = 1e-4
        one_hot_data = one_hot(data["visible"].long(), self.num_states).to(self.dtype)
        one_hot_gen = one_hot(chains["visible"].long(), self.num_states).to(self.dtype)
        fi_data = get_freq_single_point(one_hot_data, data["weights"], pseudo_count)
        fi_gen = get_freq_single_point(one_hot_gen, chains["weights"], pseudo_count)
        fij_data = get_freq_two_points(one_hot_data, data["weights"], pseudo_count)
        fij_gen = get_freq_two_points(one_hot_gen, chains["weights"], pseudo_count)
        self.bias.grad = fi_data - fi_gen
        self.weight_matrix.grad = fij_data - fij_gen

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
        )

    @staticmethod
    def init_parameters(
        num_hiddens: int,
        dataset: RBMDataset,
        device: torch.device | str,
        dtype: torch.dtype,
        var_init: float = 1e-4,
    ) -> PBM:
        """Initialize the parameters of the RBM.

        Args:
            num_hiddens (int): Number of hidden units.
            dataset (RBMDataset): Training dataset.
            device (torch.device): PyTorch device for the parameters.
            dtype (torch.dtype): PyTorch dtype for the parameters.
            var_init (float, optional): Variance of the weight matrix. Defaults to 1e-4.

        Notes:
            - The number of visible units is induced from the dataset provided.
            - Hidden biases are set to 0.
            - Visible biases are set to the frequencies of the dataset.
            - The weight matrix is initialized with a Gaussian distribution of variance `var_init`.
        """

        fi = get_freq_single_point(
            one_hot(dataset.data.long(), dataset.get_num_states()), dataset.weights, 1e-4
        )
        return PBM(
            weight_matrix=torch.zeros(
                (
                    dataset.get_num_visibles(),
                    dataset.get_num_states(),
                    dataset.get_num_visibles(),
                    dataset.get_num_states(),
                ),
                device=fi.device,
                dtype=fi.dtype,
            ),
            bias=torch.log(fi),
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
        pass

    @property
    def effective_number_variables(self) -> float:
        return 1

    @property
    def num_states(self) -> int:
        return self.weight_matrix.shape[1]
