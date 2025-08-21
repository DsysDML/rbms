from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def get_subset_labels(
    data: np.ndarray, labels: np.ndarray, subset_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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
    _, idx, counts = torch.unique(input_dataset,
        dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
    unique_ind = ind_sorted[cum_sum]
    return unique_ind