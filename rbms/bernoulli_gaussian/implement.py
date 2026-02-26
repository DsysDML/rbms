import torch
from torch import Tensor


def _sample_hiddens(
    v: Tensor, weight_matrix: Tensor, hbias: Tensor, beta: float = 1.0
) -> tuple[Tensor, Tensor]:
    mh = hbias + (v @ weight_matrix)
    h = torch.randn_like(mh) + mh
    return h, mh


def _sample_visibles(
    h: Tensor, weight_matrix: Tensor, vbias: Tensor, beta: float = 1.0
) -> tuple[Tensor, Tensor]:
    mv = torch.sigmoid(vbias + h @ weight_matrix.T)
    v = torch.bernoulli(mv)
    return v, mv


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
    num_visibles = weight_matrix.shape[0]
    quad = 0.5 * num_visibles * (h * h).sum(1)
    return -fields - interaction + quad


def _compute_energy_visibles(
    v: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    weight_matrix: Tensor,
    const: Tensor,
) -> Tensor:
    field = v @ vbias
    t = hbias + (v @ weight_matrix)
    num_visibles = weight_matrix.shape[0]
    quad_term = 0.5 * (t * t).sum(1) / num_visibles
    return -field - quad_term + const


def _compute_energy_hiddens(
    h: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
) -> Tensor:
    field = h @ hbias
    exponent = vbias + (h @ weight_matrix.T)
    log_term = torch.where(exponent < 10, torch.log1p(torch.exp(exponent)), exponent)
    num_visibles = weight_matrix.shape[0]
    quad = 0.5 * (h * h).sum(1) * num_visibles
    return -field - log_term.sum(1) + quad


def _compute_gradient(
    v_data: Tensor,
    h_data: Tensor,
    w_data: Tensor,
    v_chain: Tensor,
    h_chain: Tensor,
    w_chain: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    weight_matrix: Tensor,
    centered: bool,
) -> None:
    w_data = w_data.view(-1, 1)
    w_chain = w_chain.view(-1, 1)
    chain_weights = w_chain / w_chain.sum()
    w_data_norm = w_data.sum()

    v_data_mean = (v_data * w_data).sum(0) / w_data_norm
    torch.clamp_(v_data_mean, min=1e-4, max=(1.0 - 1e-4))
    h_data_mean = (h_data * w_data).sum(0) / w_data_norm
    v_gen_mean = (v_chain * chain_weights).sum(0)
    torch.clamp_(v_gen_mean, min=1e-4, max=(1.0 - 1e-4))
    h_gen_mean = (h_chain * chain_weights).sum(0)

    if centered:
        # Centered variables
        v_data_centered = v_data - v_data_mean
        h_data_centered = h_data - h_data_mean
        v_gen_centered = v_chain - v_data_mean
        h_gen_centered = h_chain - h_data_mean

        # Gradient
        grad_weight_matrix = (
            (v_data_centered * w_data).T @ h_data_centered
        ) / w_data_norm - ((v_gen_centered * chain_weights).T @ h_gen_centered)
        grad_vbias = v_data_mean - v_gen_mean - (grad_weight_matrix @ h_data_mean)
        grad_hbias = h_data_mean - h_gen_mean - (v_data_mean @ grad_weight_matrix)
    else:
        # Gradient: h_data instead of mh_data
        grad_weight_matrix = ((v_data * w_data).T @ h_data) / w_data_norm - (
            (v_chain * chain_weights).T @ h_chain
        )
        grad_vbias = v_data_mean - v_gen_mean
        grad_hbias = h_data_mean - h_gen_mean

    # Attach to the parameters
    weight_matrix.grad = grad_weight_matrix
    vbias.grad = grad_vbias
    hbias.grad = grad_hbias


def _init_chains(
    num_samples: int,
    weight_matrix: Tensor,
    hbias: Tensor,
    start_v: Tensor | None = None,
):
    num_visibles, num_hiddens = weight_matrix.shape
    device = weight_matrix.device
    dtype = weight_matrix.dtype
    if num_samples <= 0:
        if start_v is not None:
            num_samples = start_v.shape[0]
        else:
            raise ValueError(f"Got negative num_samples arg: {num_samples}")

    if start_v is None:
        mv = torch.ones(size=(num_samples, num_visibles), device=device, dtype=dtype) / 2
        v = torch.bernoulli(mv)
    else:
        mv = torch.zeros_like(start_v, device=device, dtype=dtype)
        v = start_v.to(device=device, dtype=dtype)

    h, mh = _sample_hiddens(v=v, weight_matrix=weight_matrix, hbias=hbias)
    return v, h, mv, mh


def _init_parameters(
    num_hiddens: int,
    data: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    var_init: float = 1e-6,
):
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
