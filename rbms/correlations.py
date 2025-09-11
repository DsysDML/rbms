from typing import Optional

import torch
from torch import Tensor


def compute_1b_freq(data: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    # Uniform weights
    if weights is None:
        weights = torch.ones(data.shape[0], device=data.device)
    # Normalize weights
    weights /= weights.sum()
    return (data * weights.unsqueeze(1)).sum(0)


@torch.jit.script
def _2b_batched(centered_data: Tensor, weights: Tensor, batch_size: int) -> Tensor:
    n_dim = centered_data.shape[1]
    res = torch.zeros(n_dim, n_dim, device=centered_data.device)
    for i in range(0, n_dim, batch_size):
        for j in range(i, n_dim, batch_size):
            tmp = torch.einsum(
                "mi ,mj -> ij",
                centered_data[:, i : i + batch_size] * weights,
                centered_data[:, j : j + batch_size],
            )
            res[i : i + batch_size, j : j + batch_size] = tmp
    return res


def compute_2b_correlations(
    data: Tensor,
    weights: Optional[Tensor] = None,
    batch_size: Optional[int] = None,
    full_mat=False,
) -> Tensor:
    # Uniform weights
    if weights is None:
        weights = torch.ones(data.shape[0], device=data.device)
    # Normalize weights
    weights /= weights.sum()

    batched = batch_size is not None
    centered_data = data - (data * weights.unsqueeze(1)).sum(0)
    if batched:
        res = _2b_batched(
            centered_data=centered_data,
            weights=weights.unsqueeze(1),
            batch_size=batch_size,
        )
        if full_mat:
            res = torch.triu(res, 1) + torch.tril(res).T
        return res / torch.sqrt(
            torch.diag(res).unsqueeze(1) @ torch.diag(res).unsqueeze(0)
        )
    return torch.corrcoef(data)


@torch.jit.script
def _3b_batched(centered_data: Tensor, weights: Tensor, batch_size: int) -> Tensor:
    n_dim = centered_data.shape[1]
    res = torch.zeros(n_dim, n_dim, n_dim, device=centered_data.device)
    for i in range(0, n_dim, batch_size):
        for j in range(i, n_dim, batch_size):
            for k in range(j, n_dim, batch_size):
                tmp = torch.einsum(
                    "mi ,mj, mk -> ijk",
                    centered_data[:, i : i + batch_size] * weights,
                    centered_data[:, j : j + batch_size],
                    centered_data[:, k : k + batch_size],
                )
                res[i : i + batch_size, j : j + batch_size, k : k + batch_size] = tmp
    return res


@torch.jit.script
def _3b_full_mat(res: Tensor) -> Tensor:
    n_dim = res.shape[0]
    for i in range(n_dim):
        for j in range(i, n_dim):
            res[i, :, j] = res[i, j]
            res[j, i, :] = res[i, j]
            res[j, :, i] = res[i, j]
            res[:, i, j] = res[i, j]
            res[:, j, i] = res[i, j]
    return res


def compute_3b_correlations(
    data: Tensor,
    weights: Optional[Tensor] = None,
    batch_size: Optional[int] = None,
    full_mat: bool = False,
) -> Tensor:
    # Uniform weights
    if weights is None:
        weights = torch.ones(data.shape[0], device=data.device)
    # Normalize weights
    weights /= weights.sum()

    batched = batch_size is not None
    centered_data = data - (data * weights.unsqueeze(1)).sum(0)
    if batched:
        res = _3b_batched(
            centered_data=centered_data,
            weights=weights.unsqueeze(1),
            batcu_size=batch_size,
        )
        if full_mat:
            res = _3b_full_mat(res)
        return res / data.shape[0]
    return (
        torch.einsum(
            "mi, mj, mk -> ijk",
            centered_data * weights.unsqueeze(1),
            centered_data,
            centered_data,
        )
        / data.shape[0]
    )
