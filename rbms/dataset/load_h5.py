from pathlib import Path

import h5py
import numpy as np
import torch

from rbms.dataset.fasta_utils import compute_weights


def load_HDF5(
    filename: str | Path,
    use_weights: bool = False,
    device: torch.device | str = "cuda",
) -> tuple[np.ndarray, np.ndarray | None, str, np.ndarray]:
    """Load a dataset from an HDF5 file.

    Args:
        filename (str): The name of the HDF5 file to load.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The dataset and labels.
    """
    labels = None
    variable_type = "bernoulli"
    with h5py.File(filename, "r") as f:
        if "samples" not in f.keys():
            raise ValueError(
                f"Could not find 'samples' key in hdf5 file keys: {f.keys()}"
            )
        dataset = np.array(f["samples"][()])
        if "variable_type" not in f.keys():
            print(
                f"No variable_type found in the hdf5 file keys: {f.keys()}. Assuming 'bernoulli'."
            )
            print(
                "Set a 'variable_type' with value 'bernoulli', 'ising', 'categorical' or 'continuous' in the hdf5 archive to remove this message"
            )
        else:
            variable_type = f["variable_type"][()].decode()
        weights = np.ones(dataset.shape[0])
        if use_weights:
            if variable_type != "categorical":
                print("Ignoring compute weights since data is not categorical")
            else:
                weights = compute_weights(data=dataset, device=device)

        if "labels" in f.keys():
            labels = np.array(f["labels"][()])
            if labels.shape[0] != dataset.shape[0]:
                print(
                    f"Ignoring labels since its dimension ({labels.shape[0]}) does not match the number of samples ({dataset.shape[0]})."
                )
                labels = None
    return dataset, labels, variable_type, weights
