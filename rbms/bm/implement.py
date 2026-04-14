import torch
from torch import Tensor

from rbms.custom_fn import one_hot


@torch.jit.script
def _get_freq_single_point(
    data: Tensor,
    weights: Tensor,
    pseudo_count: float,
) -> torch.Tensor:
    _, _, q = data.shape
    frequencies = (data * weights).sum(dim=0)
    # Set to zero the negative frequencies. Used for the reintegration.
    torch.clamp_(frequencies, min=0.0)

    return (1.0 - pseudo_count) * frequencies + (pseudo_count / q)


@torch.jit.script
def _get_freq_two_points(
    data: torch.Tensor,
    weights: torch.Tensor,
    pseudo_count: float,
) -> torch.Tensor:

    M, L, q = data.shape
    data_oh = data.reshape(M, q * L)

    fij = (data_oh * weights).T @ data_oh
    # Set to zero the negative frequencies. Used for the reintegration.
    torch.clamp_(fij, min=0.0)
    # Apply the pseudo count
    fij = (1.0 - pseudo_count) * fij + (pseudo_count / q**2)
    # Diagonal terms must represent the single point frequencies
    fij_diag = _get_freq_single_point(
        data, weights.reshape(M, 1, 1), pseudo_count
    ).ravel()
    # Set the diagonal terms of fij to the single point frequencies
    fij = torch.diagonal_scatter(fij, fij_diag, dim1=0, dim2=1)

    return fij.reshape(L, q, L, q)


@torch.jit.script
def _sample_one_visible_potts(
    v: Tensor, weight_matrix: Tensor, bias: Tensor, beta: float
) -> Tensor:
    device = v.device
    dtype = weight_matrix.dtype
    N, L, q = v.shape
    idx = torch.randint(0, L, (1,), device=device)[0]
    couplings_residue = weight_matrix[idx].reshape(q, L * q)
    logit_residue = beta * (
        bias[idx].unsqueeze(0) + v.reshape(N, L * q).to(dtype) @ couplings_residue.T
    )  # (N, q)
    new_residues = one_hot(
        torch.multinomial(torch.softmax(logit_residue, dim=-1), num_samples=1).squeeze(
            -1
        ),
        num_classes=q,
    )
    v[:, idx] = new_residues
    return v
