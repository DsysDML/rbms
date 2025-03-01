from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch

ArrayLike = Tuple[np.ndarray, List]

TOKENS_PROTEIN = "-ACDEFGHIKLMNPQRSTVWY"
TOKENS_RNA = "-ACGU"
TOKENS_DNA = "-ACGT"


def get_tokens(alphabet: str):
    """Load the vocabulary associated to the alphabet type.
    Args:
        alphabet (str): alphabet type (one of 'protein', 'rna', 'dna').

    Returns:
        str: Vocabulary.
    """
    assert isinstance(alphabet, str), "Argument 'alphabet' must be of type str"
    if alphabet == "protein":
        return TOKENS_PROTEIN
    elif alphabet == "rna":
        return TOKENS_RNA
    elif alphabet == "dna":
        return TOKENS_DNA
    else:
        return alphabet


def encode_sequence(sequence: str, tokens: str) -> np.ndarray:
    """Takes a string sequence in input an returns the numeric encoding.

    Args:
        sequence (str): Sequence.
        tokens (str): Vocabulary.

    Returns:
        list: Encoded sequence.
    """
    letter_map = {letter: number for number, letter in enumerate(tokens)}
    return np.array([letter_map[letter] for letter in sequence])


def decode_sequence(sequence: ArrayLike, tokens: str) -> str:
    """Takes a numeric sequence in input an returns the string encoding.

    Args:
        sequence (ArrayLike): Encoded sequence.
        tokens (str): Vocabulary.

    Returns:
        list: Decoded sequence.
    """
    return "".join([tokens[aa] for aa in sequence])


def import_from_fasta(fasta_name: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Import data from a fasta file.

    Args:
        fasta_name (Union[str, Path]): Path to the fasta file.

    Raises:
        RuntimeError: The file is not in fasta format.

    Returns:
        Tuple[list, list]: headers, sequences.
    """
    sequences = []
    names = []
    seq = ""
    with open(fasta_name, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line.startswith(">"):
            raise RuntimeError(f"The file {fasta_name} is not in a fasta format.")
        f.seek(0)
        for line in f:
            if not line.strip():
                continue
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                header = line[1:].strip().replace(" ", "_")
                names.append(header)
                seq = ""
            else:
                seq += line.strip()
    if seq:
        sequences.append(seq)
    return np.array(names), np.array(sequences)


def write_fasta(
    fname: str,
    headers: ArrayLike,
    sequences: ArrayLike,
    numeric_input: bool = False,
    remove_gaps: bool = False,
    alphabet: str = "protein",
):
    """Generate a fasta file with the input sequences.

    Args:
        fname (str): Name of the output fasta file.
        headers (ArrayLike): List of sequences' headers.
        sequences (ArrayLike): List of sequences.
        numeric_input (bool, optional): Whether the sequences are in numeric (encoded) format or not. Defaults to False.
        remove_gaps (bool, optional): If True, removes the gap from the alignment. Defaults to False.
        alphabet (str, optional): Selects the type of sequences. Possible chooses are ("protein", "rna"). Defaults to "protein".

    Raises:
        RuntimeError: The alphabet is not a valid choice.
    """
    tokens = get_tokens(alphabet)

    if numeric_input:
        # Decode the sequences
        seqs_decoded = np.vectorize(decode_sequence, signature="(m), () -> ()")(
            sequences, tokens
        )
    else:
        seqs_decoded = sequences.copy()
    if remove_gaps:
        seqs_decoded = np.vectorize(lambda s: s.replace("-", ""))(seqs_decoded)

    with open(fname, "w", encoding="utf-8") as f:
        for name_seq, seq in zip(headers, seqs_decoded):
            f.write(">" + name_seq + "\n")
            f.write(seq)
            f.write("\n")


def compute_weights(
    data: ArrayLike, th: float = 0.8, device: torch.device = "cpu"
) -> np.ndarray:
    """Computes the weight to be assigned to each sequence 's' in 'data' as 1 / n_clust, where 'n_clust' is the number of sequences
    that have a sequence identity with 's' >= th.

    Args:
        data (ArrayLike): Encoded input dataset.
        th (float, optional): Sequence identity threshold for the clustering. Defaults to 0.8.
        device (torch.device): Device.

    Returns:
        np.ndarray: Array with the weights of the sequences.
    """
    device = torch.device(device)
    data = torch.tensor(data, device=device)
    assert len(data.shape) == 2, "'data' must be a 2-dimensional array"
    _, L = data.shape

    def get_sequence_weight(s: torch.Tensor, data: torch.Tensor, L: int, th: float):
        seq_id = torch.sum(s == data, dim=1) / L
        n_clust = torch.sum(seq_id >= th)
        return 1.0 / n_clust

    weights = torch.vstack([get_sequence_weight(s, data, L, th) for s in data])
    return weights.cpu().numpy()


def validate_alphabet(sequences: ArrayLike, tokens: str):
    all_char = "".join(sequences)
    tokens_data = "".join(sorted(set(all_char)))
    sorted_tokens = "".join(sorted(tokens))
    if sorted_tokens != tokens_data:
        raise KeyError(
            f"The chosen alphabet is incompatible with the Multi-Sequence Alignment. The missing tokens are: {[c for c in tokens_data if c not in sorted_tokens]}"
        )
