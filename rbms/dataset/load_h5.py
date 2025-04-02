from pathlib import Path
from typing import Tuple

import h5py
import numpy as np


def load_HDF5(
    filename: str | Path, binarize: bool = True
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Load a dataset from an HDF5 file.

    Args:
        filename (str): The name of the HDF5 file to load.
        binarize (str, optional): Binarize the dataset. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The dataset and labels.
    """
    labels = None
    with h5py.File(filename, "r") as f:
        if "samples" not in f.keys():
            raise ValueError(
                f"Could not find 'samples' key if hdf5 file keys: {f.keys()}"
            )
        dataset = np.array(f["samples"][()])
        if "labels" in f.keys():
            labels = np.array(f["labels"][()])
            if labels.shape[0] != dataset.shape[0]:
                print(
                    f"Ignoring labels since its dimension ({labels.shape[0]}) does not match the number of samples ({dataset.shape[0]})."
                )
                labels = None
    if "cont" not in str(filename.resolve()):
        unique_values = np.unique(dataset)
        is_ising = np.all(unique_values == np.array([-1, 1]))
        is_bernoulli = np.all(unique_values == np.array([0, 1]))
    else:
        unique_values = [0, 1]
        is_bernoulli = np.bool_(True)
        is_ising = np.bool_(False)
    if len(unique_values) != 2:
        raise ValueError(
            f"The dataset should be binary valued but got {len(unique_values)} different unique values: {unique_values}"
        )
    if not (is_ising or is_bernoulli):
        raise ValueError(
            f"The dataset should have either [0, 1] or [-1, 1] values, got {unique_values}"
        )
    if binarize:
        if is_ising:
            dataset = (dataset + 1) / 2
    return dataset, labels
