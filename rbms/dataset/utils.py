from typing import Tuple

import numpy as np


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
