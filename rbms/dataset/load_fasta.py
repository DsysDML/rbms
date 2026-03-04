from pathlib import Path

import numpy as np
import torch

# from rbms.custom_fn import one_hot
from rbms.dataset.fasta_utils import (
    compute_weights,
    encode_sequence,
    get_tokens,
    import_from_fasta,
    validate_alphabet,
)


def load_FASTA(
    filename: str | Path,
    use_weights: bool = False,
    alphabet: str = "protein",
    device: torch.device | str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset from a FASTA file.

    Args:
        filename (str): The name of the FASTA file to load.
        use_weights (bool, optional): Whether to use weights in the dataset. Defaults to False.
        alphabet (str, optional): The alphabet used in the dataset. Defaults to "protein".
        device (str, optional): The device to use for PyTorch tensors. Defaults to "cuda".

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray] The dataset, weights and names.
    """
    # Select the proper encoding
    tokens = get_tokens(alphabet)
    names, sequences = import_from_fasta(filename)
    if len(sequences) == 0:
        raise ValueError(
            f"The input dataset is empty. Check that the alphabet is correct. Current alphabet: {alphabet}"
        )
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
    return dataset, weights, names
