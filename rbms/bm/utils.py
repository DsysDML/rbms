import torch
from torch import Tensor

from rbms.bm.implement import _get_freq_single_point, _get_freq_two_points


def get_freq_two_points(
    data: Tensor,
    weights: Tensor | None = None,
    pseudo_count: float = 0.0,
) -> Tensor:
    """
    Computes the 2-points statistics of the input MSA.

    Args:
        data (torch.Tensor): One-hot encoded data array.
        weights (Optional[torch.Tensor], optional): Array of weights to assign to the sequences of shape.
        pseudo_count (float, optional): Pseudo count for the single and two points statistics. Acts as a regularization. Defaults to 0.0.

    Raises:
        ValueError: If the input data is not a 3D tensor.

    Returns:
        torch.Tensor: Matrix of two-point frequencies of shape (L, q, L, q).
    """
    if data.dim() != 3:
        raise ValueError(
            f"Expected data to be a 3D tensor, but got {data.dim()}D tensor instead"
        )

    M = len(data)
    if weights is not None:
        norm_weights = weights.reshape(M, 1) / weights.sum()
    else:
        norm_weights = torch.ones((M, 1), device=data.device, dtype=data.dtype) / M

    return _get_freq_two_points(data, norm_weights, pseudo_count)


def get_freq_single_point(
    data: Tensor,
    weights: Tensor | None = None,
    pseudo_count: float = 0.0,
) -> Tensor:
    """Computes the single point frequencies of the input MSA.
    Args:
        data (torch.Tensor): One-hot encoded data array.
        weights (Optional[torch.Tensor], optional): Weights of the sequences.
        pseudo_count (float, optional): Pseudo count to be added to the frequencies. Defaults to 0.0.

    Raises:
        ValueError: If the input data is not a 3D tensor.

    Returns:
        torch.Tensor: Single point frequencies.
    """
    if data.dim() != 3:
        raise ValueError(
            f"Expected data to be a 3D tensor, but got {data.dim()}D tensor instead"
        )
    M = len(data)
    if weights is not None:
        norm_weights = weights.reshape(M, 1, 1) / weights.sum()
    else:
        norm_weights = torch.ones((M, 1, 1), device=data.device, dtype=data.dtype) / M

    return _get_freq_single_point(data, norm_weights, pseudo_count)
