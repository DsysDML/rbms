from typing import Optional

import torch
from torch import Tensor


@torch.jit.script
def _2b_batched(centered_data: Tensor, batch_size: int):
    n_dim = centered_data.shape[1]
    res = torch.zeros(n_dim, n_dim, device=centered_data.device)
    for i in range(0, n_dim, batch_size):
        for j in range(i, n_dim, batch_size):
            tmp = torch.einsum(
                "mi ,mj -> ij",
                centered_data[:, i : i + batch_size],
                centered_data[:, j : j + batch_size],
            )
            res[i : i + batch_size, j : j + batch_size] = tmp
    return res


def compute_2b_correlations(
    data: Tensor, batch_size: Optional[int] = None, full_mat=False
):
    batched = batch_size is not None
    centered_data = data - data.mean(0)
    if batched:
        res = _2b_batched(centered_data=centered_data, batch_size=batch_size)
        if full_mat:
            res = torch.triu(res, 1) + torch.tril(res).T
        return res / torch.sqrt(
            torch.diag(res).unsqueeze(1) @ torch.diag(res).unsqueeze(0)
        )
    return torch.corrcoef(data)


@torch.jit.script
def _3b_batched(centered_data: Tensor, batch_size: int):
    n_dim = centered_data.shape[1]
    res = torch.zeros(n_dim, n_dim, n_dim, device=centered_data.device)
    for i in range(0, n_dim, batch_size):
        for j in range(i, n_dim, batch_size):
            for k in range(j, n_dim, batch_size):
                tmp = torch.einsum(
                    "mi ,mj, mk -> ijk",
                    centered_data[:, i : i + batch_size],
                    centered_data[:, j : j + batch_size],
                    centered_data[:, k : k + batch_size],
                )
                res[i : i + batch_size, j : j + batch_size, k : k + batch_size] = tmp
    return res


@torch.jit.script
def _3b_full_mat(res: Tensor):
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
    data: Tensor, batch_size: Optional[int] = None, full_mat: bool = False
):
    batched = batch_size is not None
    centered_data = data - data.mean(0)
    if batched:
        res = _3b_batched(centered_data, batch_size)
        if full_mat:
            res = _3b_full_mat(res)
        return res / data.shape[0]
    return (
        torch.einsum("mi, mj, mk -> ijk", centered_data, centered_data, centered_data)
        / data.shape[0]
    )
