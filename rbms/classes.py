from abc import ABC, abstractmethod
from typing import List, Optional, Self

import torch
from torch import Tensor

from rbms.dataset.dataset_class import RBMDataset


class EBM(ABC):
    """An abstract class representing the parameters of an Energy-Based Model."""

    device: torch.device

    @abstractmethod
    def __init__(self): ...

    @abstractmethod
    def __add__(self, other: Self) -> Self:
        """Add the parameters of two RBMs. Useful for interpolation"""
        ...

    @abstractmethod
    def __mul__(self, other: float) -> Self:
        """Multiplies the parameters of the RBM by a float."""
        ...

    @abstractmethod
    def sample_visibles(
        self, chains: dict[str, Tensor], beta: float = 1.0
    ) -> dict[str, Tensor]:
        """Sample the visible layer conditionally to the hidden one.

        Args:
            chains (dict[str, Tensor]): The parallel chains used for sampling.
            beta (float, optional): The inverse temperature. Defaults to 1.0.

        Returns:
            dict[str, Tensor]: The updated chains with sampled hidden states.
        """
        ...

    @abstractmethod
    def compute_energy_visibles(self, v: Tensor) -> Tensor:
        """Returns the marginalized energy of the model computed on the visible configurations

        Args:
            v (Tensor): Visible configurations

        Returns:
            Tensor: The computed energy.
        """
        ...

    @abstractmethod
    def init_chains(
        self,
        num_samples: int,
        weights: Optional[Tensor] = None,
        start_v: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Initialize a Markov chain for the RBM by sampling a uniform distribution on the visible layer
        and sampling the hidden layer according to the visible one.

        Args:
            num_samples (int): The number of samples to initialize.
            start_v (Tensor, optional): The initial visible states. Defaults to None.

        Returns:
            dict[str, Tensor]: The initialized Markov chain.

        Notes:
            - If start_v is specified, its number of samples will override the num_samples argument.
        """
        ...

    @abstractmethod
    def compute_gradient(
        self,
        data: dict[str, Tensor],
        chains: dict[str, Tensor],
        centered: bool = True,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
    ) -> None:
        """Compute the gradient for each of the parameters and attach it.

        Args:
            data (dict[str, Tensor]): The data state.
            chains (dict[str, Tensor]): The parallel chains used for gradient computation.
            centered (bool, optional): Whether to use centered gradients. Defaults to True.
            lambda_l1 (float, optional): factor for the L1 regularization. Defaults to 0.
            lambda_l2 (float, optional): factor for the L2 regularization. Defaults to 0.
        """
        ...

    @abstractmethod
    def parameters(self) -> List[Tensor]:
        """Returns a list containing the parameters of the RBM.

        Returns:
            List[Tensor]: A list containing the weight matrix, visible bias, and hidden bias.
        """
        ...

    @abstractmethod
    def named_parameters(self) -> dict[str, Tensor]: ...

    @staticmethod
    @abstractmethod
    def set_named_parameters(named_params: dict[str, Tensor]) -> Self: ...

    @abstractmethod
    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Self:
        """Move the parameters to the specified device and/or convert them to the specified data type.

        Args:
            device (Optional[torch.device], optional): The device to move the parameters to.
                Defaults to None.
            dtype (Optional[torch.dtype], optional): The data type to convert the parameters to.
                Defaults to None.

        Returns:
            RBM: The modified RBM instance.
        """
        ...

    @abstractmethod
    def clone(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> Self:
        """Create a clone of the RBM instance.

        Args:
            device (Optional[torch.device], optional): The device for the cloned parameters.
                Defaults to the current device.
            dtype (Optional[torch.dtype], optional): The data type for the cloned parameters.
                Defaults to the current data type.

        Returns:
            RBM: A new RBM instance with cloned parameters.
        """
        ...

    @staticmethod
    @abstractmethod
    def init_parameters(
        num_hiddens: int,
        dataset: RBMDataset,
        device: torch.device,
        dtype: torch.dtype,
        var_init: float = 1e-4,
    ) -> Self:
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
        ...

    @abstractmethod
    def num_visibles(self) -> int:
        """Number of visible units"""
        ...

    @abstractmethod
    def ref_log_z(self) -> float:
        """Reference log partition function with weights set to 0 (except for the visible bias)."""
        ...

    @abstractmethod
    def independent_model(self) -> Self:
        """Independent model where only local fields are preserved."""

    @abstractmethod
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
        ...


class RBM(EBM):
    """An abstract class representing the parameters of a RBM."""

    @abstractmethod
    def sample_hiddens(
        self, chains: dict[str, Tensor], beta: float = 1.0
    ) -> dict[str, Tensor]:
        """Sample the hidden layer conditionally to the visible one.

        Args:
            chains (dict[str, Tensor]): The parallel chains used for sampling.
            beta (float, optional): The inverse temperature. Defaults to 1.0.

        Returns:
            dict[str, Tensor]: The updated chains with sampled hidden states.
        """
        ...

    @abstractmethod
    def compute_energy(self, v: Tensor, h: Tensor) -> Tensor:
        """Compute the energy of the RBM on the visible and hidden variables.

        Args:
            v (Tensor): Visible configurations.
            h (Tensor): Hidden configurations.

        Returns:
            Tensor: The computed energy.
        """
        ...

    @abstractmethod
    def compute_energy_hiddens(self, h: Tensor) -> Tensor:
        """Returns the marginalized energy of the model computed on hidden configurations

        Args:
            h (Tensor): The computed energy
        """
        ...

    @abstractmethod
    def num_hiddens(self) -> int:
        """Number of hidden units"""
        ...

    def sample_state(self, chains, n_steps, beta=1.0):
        new_chains = {
            "visible": chains["visible"].clone(),
            "weights": chains["weights"].clone(),
        }
        for _ in range(n_steps):
            new_chains = self.sample_hiddens(chains=new_chains, beta=beta)
            new_chains = self.sample_visibles(chains=new_chains, beta=beta)
        new_chains = self.sample_hiddens(chains=new_chains, beta=beta)
        return new_chains
