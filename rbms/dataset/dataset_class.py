import gzip
import textwrap
from typing import Dict, Union, Self, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RBMDataset(Dataset):
    """A dataset class for RBM training and evaluation."""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        names: np.ndarray,
        dataset_name: str,
        is_binary: bool,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # names should stay as a np array as its dtype is object
        self.names = names
        self.dataset_name = dataset_name
        self.device = device
        self.dtype = dtype
        self.is_binary = is_binary
        self.data = torch.from_numpy(data).to(device=self.device, dtype=self.dtype)
        # Weights should have shape n_visibles
        self.weights = (
            torch.from_numpy(weights).view(-1).to(device=self.device, dtype=self.dtype)
        )
        # Labels are int
        self.labels = torch.from_numpy(labels).to(device=self.device, dtype=torch.int32)

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Get a sample from the dataset.

        Args:
            index (int): The index of the sample.

        Returns:
            Dict[str, Union[np.ndarray, torch.Tensor]]: A dictionary containing the sample data, labels, weights, and names.
        """
        return {
            "data": self.data[index],
            "labels": self.labels[index],
            "weights": self.weights[index],
            "names": self.names[index],
        }

    def __str__(self) -> str:
        """Get a string representation of the dataset.

        Returns:
            str: The string representation of the dataset.
        """
        return textwrap.dedent(
            f"""
        Dataset: {self.dataset_name}
        Variable type: {"Bernoulli" if self.is_binary else "Potts"}
        Number of samples: {self.data.shape[0]}
        Number of features: {self.data.shape[1]}
        """
        )

    def get_num_visibles(self) -> int:
        """Get the number of visible units.

        Returns:
            int: The number of visible units.
        """
        return self.data.shape[1]

    def get_num_states(self) -> int:
        """Get the number of states.

        Returns:
            int: The number of states.
        """
        return int(self.data.max() + 1)

    def get_effective_size(self) -> int:
        """Get the effective size of the dataset.

        Returns:
            int: The effective size of the dataset.
        """
        return int(self.weights.sum())

    def get_gzip_entropy(self, mean_size: int = 50, num_samples: int = 100):
        """Compute the gzip entropy of the dataset.

        Args:
            mean_size (int, optional): The number of samples to average over. Defaults to 50.
            num_samples (int, optional): The number of samples to use for each entropy calculation. Defaults to 100.

        Returns:
            float: The computed gzip entropy.
        """

        pbar = tqdm(range(mean_size))
        pbar.set_description("Compute entropy gzip")
        en = np.zeros(mean_size)
        for i in pbar:
            en[i] = len(
                gzip.compress(
                    (
                        self.data[torch.randperm(self.data.shape[0])[:num_samples]]
                    ).astype(int)
                )
            )
        return np.mean(en)

    def split_train_test(
        self,
        rng: np.random.Generator,
        train_size: float,
        test_size: Optional[float] = None,
    ) -> Tuple[Self, Self | None]:
        num_samples = self.data.shape[0]
        if test_size is None:
            test_size = 1.0 - train_size

        # Shuffle dataset
        permutation_index = rng.permutation(num_samples)
        train_size = int(train_size * num_samples)
        test_size = int(test_size * num_samples)

        train_dataset = RBMDataset(
            data=self.data[permutation_index[:train_size]].cpu().numpy(),
            labels=self.labels[permutation_index[:train_size]].cpu().numpy(),
            weights=self.weights[permutation_index[:train_size]].cpu().numpy(),
            names=self.names[permutation_index[:train_size]],
            dataset_name=self.dataset_name,
            is_binary=self.is_binary,
            device=self.device,
            dtype=self.dtype,
        )
        test_dataset = None
        if test_size > 0:
            test_dataset = RBMDataset(
                data=self.data[permutation_index[train_size : train_size + test_size]]
                .cpu()
                .numpy(),
                labels=self.labels[
                    permutation_index[train_size : train_size + test_size]
                ]
                .cpu()
                .numpy(),
                weights=self.weights[
                    permutation_index[train_size : train_size + test_size]
                ]
                .cpu()
                .numpy(),
                names=self.names[
                    permutation_index[train_size : train_size + test_size]
                ],
                dataset_name=self.dataset_name,
                is_binary=self.is_binary,
                device=self.device,
                dtype=self.dtype,
            )
        return train_dataset, test_dataset
