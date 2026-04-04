from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from rbms.custom_fn import one_hot


def get_subset_labels(
    data: np.ndarray, labels: np.ndarray, subset_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Select subset of dataset w.r.t. labels
    dataset_select = []
    labels_select = []
    for label in subset_labels:
        mask = labels == label
        dataset_select.append(np.array(data[mask], dtype=float))
        labels_select.append(np.array(labels[mask]))
    data = np.concatenate(dataset_select)
    labels = np.concatenate(labels_select)
    return data, labels


@torch.jit.script
def get_unique_indices(input_dataset: Tensor) -> Tensor:
    """
    Given a dataset, return the first index of every unique sample of the dataset. Useful to remove duplicates.

    Args:
        input_dataset (str): Dataset to get unique indices from.

    Returns:
        Tensor: Indices of the first appearance of each unique value.
    """
    _, idx, counts = torch.unique(
        input_dataset, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
    unique_ind = ind_sorted[cum_sum]
    return unique_ind


def bernoulli_to_ising(x):
    return x * 2 - 1


def ising_to_bernoulli(x):
    return (x + 1) / 2


def categorical_to_bernoulli(x):
    return one_hot(x.long()).reshape(x.shape[0], -1)


convert_data: dict[str, dict[str, Callable[[Tensor], Tensor]]] = {
    "bernoulli": {
        "bernoulli": (lambda x: x),
        "ising": (lambda x: bernoulli_to_ising(x)),
        "categorical": (lambda x: x),
        "continuous": (lambda x: x),
    },
    "ising": {
        "bernoulli": (lambda x: ising_to_bernoulli(x)),
        "ising": (lambda x: x),
        "categorical": (lambda x: ising_to_bernoulli(x)),
    },
    "categorical": {
        "bernoulli": (lambda x: categorical_to_bernoulli(x)),
        "ising": (lambda x: bernoulli_to_ising(categorical_to_bernoulli(x))),
        "categorical": (lambda x: x),
    },
    "continuous": {"bernoulli": (lambda x: x), "continuous": (lambda x: x)},
}


def get_covariance_matrix(
    data: Tensor,
    weights: Tensor | None = None,
    num_extract: int | None = None,
    center: bool = True,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Returns the covariance matrix of the data. If weights is specified, the weighted covariance matrix is computed.

    Args:
        data (Tensor): Data.
        weights (Tensor, optional): Weights of the data. Defaults to None.
        num_extract (int, optional): Number of data to extract to compute the covariance matrix. Defaults to None.
        center (bool): Center the data. Defaults to True.
        device (torch.device): Device. Defaults to 'cpu'.
        dtype (torch.dtype): DType. Defaults to torch.float32.

    Returns:
        Tensor: Covariance matrix of the dataset.
    """
    num_data = len(data)
    num_classes = int(data.max().item() + 1)
    dtype = data.dtype

    if weights is None:
        weights = torch.ones(num_data)
    weights = weights.to(device=device, dtype=dtype)

    if num_extract is not None:
        idxs = np.random.choice(a=np.arange(num_data), size=(num_extract,), replace=False)
        data = data[idxs]
        weights = weights[idxs]
        num_data = num_extract

    if num_classes != 2:
        data = data.to(device=device, dtype=torch.int32)
        data_oh = one_hot(data, num_classes=num_classes).reshape(num_data, -1)
    else:
        data_oh = data.to(device=device, dtype=dtype)

    norm_weights = weights.reshape(-1, 1) / weights.sum()
    data_mean = (data_oh * norm_weights).sum(0, keepdim=True)
    cov_matrix = ((data_oh * norm_weights).mT @ data_oh) - int(center) * (
        data_mean.mT @ data_mean
    )
    return cov_matrix
