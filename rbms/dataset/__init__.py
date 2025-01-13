from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from rbms.dataset.dataset_class import RBMDataset
from rbms.dataset.load_fasta import load_FASTA
from rbms.dataset.load_h5 import load_HDF5
from rbms.dataset.utils import get_subset_labels


def load_dataset(
    dataset_name: str,
    subset_labels: Optional[List[int]] = None,
    use_weights: bool = False,
    train_size: float = 0.6,
    test_size: Optional[float] = None,
    binarize: bool = False,
    alphabet="protein",
    seed: int = 19023741073419046239412739401234901,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    rng = np.random.default_rng(seed)
    data = None
    weights = None
    names = None
    labels = None
    is_binary = True

    dataset_name = Path(dataset_name)

    match dataset_name.suffix:
        case ".h5":
            data, labels = load_HDF5(filename=dataset_name, binarize=binarize)
        case ".fasta":
            data, weights, names = load_FASTA(
                filename=dataset_name,
                binarize=binarize,
                use_weights=use_weights,
                alphabet=alphabet,
                device=device,
            )
            if not binarize:
                is_binary = False
        case _:
            raise ValueError(
                """
            Dataset could not be loaded as the type is not recognized.
            It should be either:
                - '.h5',
                - '.fasta'
            """
            )
    # Select subset of dataset w.r.t. labels
    if subset_labels is not None and labels is not None:
        data, labels = get_subset_labels(data, labels, subset_labels)

    if weights is None:
        weights = np.ones(data.shape[0])
    if names is None:
        names = np.arange(data.shape[0])
    if labels is None:
        labels = -np.ones(data.shape[0])

    # Shuffle dataset
    permutation_index = rng.permutation(data.shape[0])

    # Split train/test
    train_size = int(train_size * data.shape[0])
    if test_size is not None:
        test_size = int(test_size * data.shape[0])
    else:
        test_size = data.shape[0] - train_size

    train_dataset = RBMDataset(
        data=data[permutation_index[:train_size]],
        labels=labels[permutation_index[:train_size]],
        weights=weights[permutation_index[:train_size]],
        names=names[permutation_index[:train_size]],
        dataset_name=dataset_name,
        is_binary=is_binary,
        device=device,
        dtype=dtype,
    )
    test_dataset = None
    if test_size > 0:
        test_dataset = RBMDataset(
            data=data[permutation_index[train_size : train_size + test_size]],
            labels=labels[permutation_index[train_size : train_size + test_size]],
            weights=weights[permutation_index[train_size : train_size + test_size]],
            names=names[permutation_index[train_size : train_size + test_size]],
            dataset_name=dataset_name,
            is_binary=is_binary,
            device=device,
            dtype=dtype,
        )
    return train_dataset, test_dataset
