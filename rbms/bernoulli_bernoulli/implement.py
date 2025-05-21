from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import softmax


@torch.jit.script
def _sample_hiddens(
    v: Tensor, weight_matrix: Tensor, hbias: Tensor, beta: float = 1.0
) -> Tuple[Tensor, Tensor]:
    mh = torch.sigmoid(beta * (hbias + (v @ weight_matrix)))
    h = torch.bernoulli(mh)
    return h, mh


@torch.jit.script
def _sample_visibles(
    h: Tensor, weight_matrix: Tensor, vbias: Tensor, beta: float = 1.0
) -> Tuple[Tensor, Tensor]:
    mv = torch.sigmoid(beta * (vbias + (h @ weight_matrix.T)))
    v = torch.bernoulli(mv)
    return v, mv


@torch.jit.script
def _compute_energy(
    v: Tensor,
    h: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    weight_matrix: Tensor,
) -> Tensor:
    fields = torch.tensordot(vbias, v, dims=[[0], [1]]) + torch.tensordot(
        hbias, h, dims=[[0], [1]]
    )
    interaction = torch.multiply(
        v, torch.tensordot(h, weight_matrix, dims=[[1], [1]])
    ).sum(1)

    return -fields - interaction


@torch.jit.script
def _compute_energy_visibles(
    v: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
) -> Tensor:
    field = v @ vbias
    exponent = hbias + (v @ weight_matrix)
    log_term = torch.where(
        exponent < 10, torch.log(1.0 + torch.exp(exponent)), exponent
    )
    return -field - log_term.sum(1)


@torch.jit.script
def _compute_energy_hiddens(
    h: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
) -> Tensor:
    field = h @ hbias
    exponent = vbias + (h @ weight_matrix.T)
    log_term = torch.where(
        exponent < 10, torch.log(1.0 + torch.exp(exponent)), exponent
    )
    return -field - log_term.sum(1)


@torch.jit.script
def _compute_gradient(
    v_data: Tensor,
    mh_data: Tensor,
    w_data: Tensor,
    v_chain: Tensor,
    h_chain: Tensor,
    w_chain: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    weight_matrix: Tensor,
    centered: bool = True,
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
) -> None:
    w_data = w_data.view(-1, 1)
    w_chain = w_chain.view(-1, 1)
    # Turn the weights of the chains into normalized weights
    chain_weights = softmax(-w_chain, dim=0)
    w_data_norm = w_data.sum()

    # Averages over data and generated samples
    v_data_mean = (v_data * w_data).sum(0) / w_data_norm
    torch.clamp_(v_data_mean, min=1e-7, max=(1.0 - 1e-7))
    h_data_mean = (mh_data * w_data).sum(0) / w_data_norm
    v_gen_mean = (v_chain * chain_weights).sum(0)
    torch.clamp_(v_gen_mean, min=1e-7, max=(1.0 - 1e-7))
    h_gen_mean = (h_chain * chain_weights).sum(0)

    if centered:
        # Centered variables
        v_data_centered = v_data - v_data_mean
        h_data_centered = mh_data - h_data_mean
        v_gen_centered = v_chain - v_data_mean
        h_gen_centered = h_chain - h_data_mean

        # Gradient
        grad_weight_matrix = (
            (v_data_centered * w_data).T @ h_data_centered
        ) / w_data_norm - ((v_gen_centered * chain_weights).T @ h_gen_centered)
        grad_vbias = v_data_mean - v_gen_mean - (grad_weight_matrix @ h_data_mean)
        grad_hbias = h_data_mean - h_gen_mean - (v_data_mean @ grad_weight_matrix)
    else:
        v_data_centered = v_data
        h_data_centered = mh_data
        v_gen_centered = v_chain
        h_gen_centered = h_chain

        # Gradient
        grad_weight_matrix = ((v_data * w_data).T @ mh_data) / w_data_norm - (
            (v_chain * chain_weights).T @ h_chain
        )
        grad_vbias = v_data_mean - v_gen_mean
        grad_hbias = h_data_mean - h_gen_mean

    if lambda_l1 > 0:
        grad_weight_matrix -= 2 * lambda_l2 * torch.sign(weight_matrix)
        grad_vbias -= 2 * lambda_l2 * torch.sign(vbias)
        grad_hbias -= 2 * lambda_l2 * torch.sign(hbias)

    if lambda_l2 > 0:
        grad_weight_matrix -= 2 * lambda_l2 * weight_matrix
        grad_vbias -= 2 * lambda_l2 * vbias
        grad_hbias -= 2 * lambda_l2 * hbias

    # Attach to the parameters

    weight_matrix.grad.set_(grad_weight_matrix)
    vbias.grad.set_(grad_vbias)
    hbias.grad.set_(grad_hbias)


@torch.jit.script
def _init_chains(
    num_samples: int,
    weight_matrix: Tensor,
    hbias: Tensor,
    start_v: Optional[Tensor] = None,
):
    num_visibles, _ = weight_matrix.shape
    device = weight_matrix.device
    dtype = weight_matrix.dtype
    # Handle negative number of samples
    if num_samples <= 0:
        if start_v is not None:
            num_samples = start_v.shape[0]
        else:
            raise ValueError(f"Got negative num_samples arg: {num_samples}")

    if start_v is None:
        # Dummy mean visible
        mv = (
            torch.ones(size=(num_samples, num_visibles), device=device, dtype=dtype) / 2
        )
        v = torch.bernoulli(mv)
    else:
        # Dummy mean visible
        mv = torch.ones_like(start_v, device=device, dtype=dtype) / 2
        v = start_v.to(device=device, dtype=dtype)

    # Initialize chains

    h, mh = _sample_hiddens(v=v, weight_matrix=weight_matrix, hbias=hbias)
    return v, h, mv, mh


def _init_parameters(
    num_hiddens: int,
    data: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    var_init: float = 1e-4,
) -> Tuple[Tensor, Tensor, Tensor]:
    _, num_visibles = data.shape
    eps = 1e-4
    weight_matrix = (
        torch.randn(size=(num_visibles, num_hiddens), device=device, dtype=dtype)
        * var_init
    )
    frequencies = data.mean(0)
    frequencies = torch.clamp(frequencies, min=eps, max=(1.0 - eps))
    vbias = (torch.log(frequencies) - torch.log(1.0 - frequencies)).to(
        device=device, dtype=dtype
    )
    hbias = torch.zeros(num_hiddens, device=device, dtype=dtype)
    return vbias, hbias, weight_matrix
