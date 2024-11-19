from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from rbms.dataset.fasta_utils import (
    compute_weights,
    encode_sequence,
    get_tokens,
    import_from_fasta,
    validate_alphabet,
)


def load_FASTA(
    filename: str | Path,
    binarize: bool = False,
    use_weights: bool = False,
    alphabet: str = "protein",
    device="cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset from a FASTA file.

    Args:
        filename (str): The name of the FASTA file to load.
        binarize (bool, optional): Binarize the dataset to [0,1]. Defaults to "Potts".
        use_weights (bool, optional): Whether to use weights in the dataset. Defaults to False.
        alphabet (str, optional): The alphabet used in the dataset. Defaults to "protein".
        device (str, optional): The device to use for PyTorch tensors. Defaults to "cuda".

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray] The dataset, weights and names.
    """
    # Select the proper encoding
    tokens = get_tokens(alphabet)
    names, sequences = import_from_fasta(filename)
    validate_alphabet(sequences=sequences, tokens=tokens)
    names = np.array(names)
    dataset = np.vectorize(
        encode_sequence, excluded=["tokens"], signature="(), () -> (n)"
    )(sequences, tokens)

    num_data = len(dataset)
    if use_weights:
        print("Automatically computing the sequence weights...")
        weights = compute_weights(dataset, device=device)
    else:
        weights = np.ones((num_data, 1), dtype=np.float32)

    weights = weights.squeeze(-1)
    if binarize:
        num_categories = len(np.unique(dataset))
        num_visibles = dataset.shape[1]
        categories = (
            np.arange(num_categories)
            .repeat(num_visibles, axis=0)
            .reshape(-1, num_visibles)
            .T
        )
        enc = OneHotEncoder(categories=categories.tolist())
        dataset = enc.fit_transform(dataset).toarray()
    return dataset, weights, names
