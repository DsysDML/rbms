from typing import Tuple

import torch
from torch import Tensor


def compute_aats(
    sample_data: Tensor, sample_gen: Tensor, n_sample: int, dist: str = "euclid"
) -> Tuple[float, float]:
    """
    Compute the nearest neighbor Adversarial Accuracy (AATS).

    This function calculates the AATS score for two sample sets by determining
    the probability that the nearest neighbor of a sample in one set belongs
    to the same set.

    Args:
        sample_data (Tensor): First sample set.
        sample_gen (Tensor): Second sample set
        n_sample (int): Subset cardinal on which to compute both AATS score
        dist (str, optional): Distance to use. Can be ("euclid", "hamming")
    Returns:
        Tuple[float, float]:
            AAtruth (float): AATS score for the first sample set.
            AAsyn (float): AATS score for the second sample set.
    """
    # Concatenate data
    full_data = torch.cat((sample_data[:n_sample], sample_gen[:n_sample]), 0)
    # Compute distance matrix
    match dist:
        case "euclid":
            distance_matrix = torch.cdist(full_data, full_data, p=2.0)
        case "hamming":
            distance_matrix = torch.cdist(full_data, full_data, p=0.0)
        case _:
            raise ValueError(
                f"'dist' arg should be one of ('euclid', 'hamming'), got '{dist}'"
            )
    torch.diagonal(distance_matrix).fill_(float("inf"))

    # switch to numpy to handle negative indexing
    distance_matrix = distance_matrix.cpu().numpy()

    # the next line is use to transform the matrix into
    #  d_TT d_TF   INTO d_TF- d_TT-  where the minus indicate a reverse order of the columns
    #  d_FT d_FF        d_FT  d_FF
    distance_matrix[: int(distance_matrix.shape[0] / 2), :] = distance_matrix[
        : int(distance_matrix.shape[0] / 2), ::-1
    ]
    closest = distance_matrix.argmin(axis=1)
    n = int(closest.shape[0] / 2)

    # for a true sample, proba that the closest is in the set of true samples
    aa_truth = (closest[:n] >= n).sum() / n
    # for a fake sample, proba that the closest is in the set of fake samples
    aa_syn = (closest[n:] >= n).sum() / n

    return aa_truth, aa_syn
