from pathlib import Path

import numpy as np
import torch

from rbms.dataset.dataset_class import RBMDataset
from rbms.dataset.load_fasta import load_FASTA
from rbms.dataset.load_h5 import load_HDF5
from rbms.dataset.utils import get_subset_labels, get_unique_indices


def load_dataset(
    dataset_name: str,
    test_dataset_name: str | None = None,
    subset_labels: list[int] | None = None,
    use_weights: bool = False,
    alphabet="protein",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[RBMDataset, RBMDataset | None]:
    return_datasets = []
    for dset_name in [dataset_name, test_dataset_name]:
        data = None
        labels = None
        weights = None
        names = None

        if dset_name is not None:
            dset_name = Path(dset_name)
            print(f"Reading dataset from {str(dset_name)}...")
            match dset_name.suffix:
                case ".h5":
                    data, labels, variable_type, weights = load_HDF5(
                        filename=dset_name,
                        use_weights=use_weights,
                        device=device,
                    )
                case _:
                    data, weights, names = load_FASTA(
                        filename=dset_name,
                        use_weights=use_weights,
                        alphabet=alphabet,
                        device=device,
                    )
                    variable_type = "categorical"
            # Select subset of dataset w.r.t. labels
            if subset_labels is not None and labels is not None:
                data, labels = get_subset_labels(data, labels, subset_labels)

            if weights is None:
                weights = np.ones(data.shape[0])
            if names is None:
                names = np.arange(data.shape[0])
            if labels is None:
                labels = -np.ones(data.shape[0])

            # Remove duplicates and internally shuffle the dataset
            unique_ind = get_unique_indices(torch.from_numpy(data)).cpu().numpy()

            idx = torch.randperm(unique_ind.shape[0])
            data = data[unique_ind[idx]]
            labels = labels[unique_ind[idx]]
            weights = weights[unique_ind[idx]]
            names = names[unique_ind[idx]]

            return_datasets.append(
                RBMDataset(
                    data=data,
                    labels=labels,
                    weights=weights,
                    names=names,
                    dataset_name=dataset_name,
                    variable_type=variable_type,
                    device=device,
                    dtype=dtype,
                )
            )
            print("    Done")
        else:
            return_datasets.append(None)
    return tuple(return_datasets)
