from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import softmax

from rbms.custom_fn import one_hot


@torch.jit.script
def _sample_hiddens(
    v: Tensor, weight_matrix: Tensor, hbias: Tensor, beta: float = 1.0
) -> Tuple[Tensor, Tensor]:
    dtype = weight_matrix.dtype
    num_visibles, num_states, num_hiddens = weight_matrix.shape
    weight_matrix_oh = weight_matrix.view(num_visibles * num_states, num_hiddens)
    v_oh = one_hot(v.to(torch.int32), num_classes=num_states, dtype=dtype).view(
        -1, num_visibles * num_states
    )
    mh = torch.sigmoid(beta * (hbias + v_oh @ weight_matrix_oh))
    h = torch.bernoulli(mh).to(weight_matrix.dtype)
    return h, mh


@torch.jit.script
def _sample_visibles(
    h: Tensor, weight_matrix: Tensor, vbias: Tensor, beta: float = 1.0
) -> Tuple[Tensor, Tensor]:
    num_visibles, num_states, _ = weight_matrix.shape
    mv = torch.softmax(
        beta * (vbias + torch.tensordot(h, weight_matrix, dims=[[1], [2]])),
        dim=-1,
    )
    v = (
        torch.multinomial(mv.view(-1, num_states), 1)
        .view(-1, num_visibles)
        .to(weight_matrix.dtype)
    )
    return v, mv


@torch.jit.script
def _compute_energy(
    v: Tensor, h: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
):
    dtype = weight_matrix.dtype
    num_visibles, num_states, num_hiddens = weight_matrix.shape
    v_oh = one_hot(v.to(torch.int32), num_classes=num_states, dtype=dtype).view(
        -1, num_visibles * num_states
    )
    vbias_oh = vbias.flatten()
    weight_matrix_oh = weight_matrix.view(num_visibles * num_states, num_hiddens)
    fields = (v_oh @ vbias_oh) + (h @ hbias)
    interaction = ((v_oh @ weight_matrix_oh) * h).sum(1)
    return -fields - interaction


@torch.jit.script
def _compute_energy_visibles(
    v: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
):
    dtype = weight_matrix.dtype
    num_visibles, num_states, num_hiddens = weight_matrix.shape
    v_oh = one_hot(v.to(torch.int32), num_classes=num_states, dtype=dtype).view(
        -1, num_visibles * num_states
    )

    vbias_oh = vbias.flatten()
    weight_matrix_oh = weight_matrix.view(num_visibles * num_states, num_hiddens)
    field = v_oh @ vbias_oh
    exponent = hbias + (v_oh @ weight_matrix_oh)
    log_term = torch.where(
        exponent < 10, torch.log(1.0 + torch.exp(exponent)), exponent
    )
    return -field - log_term.sum(1)


@torch.jit.script
def _compute_energy_hiddens(
    h: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
):
    field = h @ hbias
    arg_lse = vbias + torch.tensordot(h, weight_matrix, dims=[[1], [2]])
    lse = torch.logsumexp(arg_lse, dim=2).sum(1)
    return -field - lse


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
):
    w_data = w_data.view(-1, 1, 1)
    w_chain = w_chain.view(-1, 1, 1)
    int_dtype = torch.int32
    dtype = weight_matrix.dtype
    num_states = weight_matrix.shape[1]

    # One-hot representation of the data
    v_data_one_hot = one_hot(v_data.to(int_dtype), num_classes=num_states, dtype=dtype)
    v_gen_one_hot = one_hot(v_chain.to(int_dtype), num_classes=num_states, dtype=dtype)

    # Turn the weights of the chains into normalized weights
    chain_weights = softmax(-w_chain, dim=0)
    w_chain_norm = chain_weights.sum()
    w_data_norm = w_data.sum()
    # Averages over data and generated samples
    v_data_mean = (v_data_one_hot * w_data).sum(0) / w_data_norm
    h_data_mean = (mh_data * w_data.view(-1, 1)).sum(0) / w_data_norm
    v_gen_mean = (v_gen_one_hot * chain_weights).sum(0) / w_chain_norm
    h_gen_mean = (h_chain * chain_weights.view(-1, 1)).sum(0) / w_chain_norm
    torch.clamp_(v_data_mean, min=1e-7, max=(1.0 - 1e-7))
    torch.clamp_(v_gen_mean, min=1e-7, max=(1.0 - 1e-7))
    if centered:
        # Centered variables
        v_data_centered = v_data_one_hot - v_data_mean
        h_data_centered = mh_data - h_data_mean
        v_gen_centered = v_gen_one_hot - v_data_mean
        h_gen_centered = h_chain - h_data_mean

        # Gradient
        grad_weight_matrix = (
            torch.tensordot(
                v_data_centered,
                h_data_centered,
                dims=[[0], [0]],
            )
            / v_data.shape[0]
            - torch.tensordot(
                v_gen_centered,
                h_gen_centered,
                dims=[[0], [0]],
            )
            / v_chain.shape[0]
        )
        grad_vbias = (
            v_data_mean
            - v_gen_mean
            - torch.tensordot(grad_weight_matrix, h_data_mean, dims=[[2], [0]])
        )
        grad_hbias = (
            h_data_mean
            - h_gen_mean
            - torch.tensordot(v_data_mean, grad_weight_matrix, dims=[[0, 1], [0, 1]])
        )
    else:
        v_data_centered = v_data_one_hot
        h_data_centered = mh_data
        v_gen_centered = v_gen_one_hot
        h_gen_centered = h_chain

        # Gradient
        grad_weight_matrix = (
            torch.tensordot(
                v_data_centered,
                h_data_centered,
                dims=[[0], [0]],
            )
            / v_data.shape[0]
            - torch.tensordot(
                v_gen_centered,
                h_gen_centered,
                dims=[[0], [0]],
            )
            / v_chain.shape[0]
        )

        grad_vbias = v_data_mean - v_gen_mean
        grad_hbias = h_data_mean - h_gen_mean

    if lambda_l1 > 0:
        grad_weight_matrix -= 2 * lambda_l2 * torch.sign(weight_matrix)
        grad_vbias -= 2 * lambda_l2 * torch.sign(vbias)
        grad_hbias -= 2 * lambda_l2 * torch.sign(hbias)

    if lambda_l2 != 0:
        grad_weight_matrix -= 2 * lambda_l2 * weight_matrix
        grad_vbias -= 2 * lambda_l2 * vbias
        grad_hbias -= 2 * lambda_l2 * hbias
    weight_matrix.grad.set_(grad_weight_matrix)
    vbias.grad.set_(grad_vbias)
    hbias.grad.set_(grad_hbias)


def _init_chains(
    num_samples: int,
    weight_matrix: Tensor,
    hbias: Tensor,
    start_v: Optional[Tensor] = None,
):
    num_visibles, num_states, num_hiddens = weight_matrix.shape
    if start_v is None:
        v = torch.randint(
            0,
            num_states,
            size=(num_samples, num_visibles),
            device=weight_matrix.device,
            dtype=weight_matrix.dtype,
        )
    else:
        v = start_v.to(weight_matrix.dtype)
    weight_matrix_oh = weight_matrix.view(num_visibles * num_states, num_hiddens)
    v_oh = one_hot(
        v.to(torch.int32), num_classes=num_states, dtype=weight_matrix_oh.dtype
    ).view(-1, num_visibles * num_states)
    mv = torch.zeros(v.shape[0], v.shape[1], num_states)
    mh = torch.sigmoid(hbias + v_oh @ weight_matrix_oh)
    h = torch.bernoulli(mh)
    return v, h, mv, mh


def _init_parameters(
    num_hiddens: int,
    data: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    var_init: float = 1e-4,
) -> Tuple[Tensor, Tensor, Tensor]:
    _, num_visibles = data.shape
    eps = 1e-7
    num_states = int(torch.max(data) + 1)
    all_states = torch.arange(num_states).reshape(-1, 1, 1).to(data.device)
    frequencies = (data == all_states).type(torch.float32).mean(1).to(device)
    frequencies = torch.clamp(frequencies, min=eps, max=(1.0 - eps))
    vbias = (
        (
            torch.log(frequencies)
            - 1.0 / num_states * torch.sum(torch.log(frequencies), 0)
        )
        .to(device=device, dtype=dtype)
        .T
    )
    hbias = torch.zeros(num_hiddens, device=device, dtype=dtype)
    weight_matrix = (
        torch.randn(
            size=(num_visibles, num_states, num_hiddens), device=device, dtype=dtype
        )
        * var_init
    )
    # print(torch.svd(weight_matrix.reshape(-1, weight_matrix.shape[-1])).S)
    return vbias, hbias, weight_matrix
