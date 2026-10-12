import math

import torch
from torch import Tensor

# Energy of the Gaussian-Bernoulli RBM with hidden-dependent visible variances:
#   E(v, h) = sum_i lambda_i v_i^2 / 2 - sum_i b_i v_i - sum_a c_a h_a
#             - sum_ia v_i W_ia h_a - sum_ia (v_i^2 / 2) W2_ia h_a
# Given h, the visibles are independent Gaussians with precision
#   P_i(h) = lambda_i - sum_a W2_ia h_a   (kept > 0, see `_enforce_positive_precision`)
# and mean (b_i + sum_a W_ia h_a) / P_i(h).
# p(h_a = 1 | v) = sigmoid(c_a + sum_i v_i W_ia + sum_i (v_i^2 / 2) W2_ia).
# At inverse temperature beta the whole energy is multiplied by beta.


@torch.jit.script
def _sample_hiddens(
    v: Tensor,
    weight_matrix: Tensor,
    weight_matrix_2: Tensor,
    hbias: Tensor,
    beta: float = 1.0,
) -> tuple[Tensor, Tensor]:
    mh = torch.sigmoid(
        beta * (hbias + (v @ weight_matrix) + (0.5 * v.square()) @ weight_matrix_2)
    )
    h = torch.bernoulli(mh)
    return h, mh


@torch.jit.script
def _sample_visibles(
    h: Tensor,
    weight_matrix: Tensor,
    weight_matrix_2: Tensor,
    vbias: Tensor,
    precision: Tensor,
    beta: float = 1.0,
) -> tuple[Tensor, Tensor]:
    prec_h = precision - h @ weight_matrix_2.T
    mv = (vbias + (h @ weight_matrix.T)) / prec_h
    v = mv + torch.randn_like(mv) / torch.sqrt(beta * prec_h)
    return v, mv


@torch.jit.script
def _compute_energy(
    v: Tensor,
    h: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    weight_matrix: Tensor,
    weight_matrix_2: Tensor,
    precision: Tensor,
) -> Tensor:
    half_sq = 0.5 * v.square()
    fields = v @ vbias - half_sq @ precision + h @ hbias
    interaction = ((v @ weight_matrix + half_sq @ weight_matrix_2) * h).sum(1)
    return -fields - interaction


@torch.jit.script
def _compute_energy_visibles(
    v: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    weight_matrix: Tensor,
    weight_matrix_2: Tensor,
    precision: Tensor,
) -> Tensor:
    half_sq = 0.5 * v.square()
    field = v @ vbias - half_sq @ precision
    exponent = hbias + (v @ weight_matrix) + half_sq @ weight_matrix_2
    log_term = torch.nn.functional.softplus(exponent)
    return -field - log_term.sum(1)


@torch.jit.script
def _compute_energy_hiddens(
    h: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    weight_matrix: Tensor,
    weight_matrix_2: Tensor,
    precision: Tensor,
) -> Tensor:
    field = h @ hbias
    prec_h = precision - h @ weight_matrix_2.T
    gaussian_term = 0.5 * (vbias + (h @ weight_matrix.T)).square() / prec_h + 0.5 * (
        math.log(2 * math.pi) - torch.log(prec_h)
    )
    return -field - gaussian_term.sum(1)


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
    weight_matrix_2: Tensor,
    precision: Tensor,
    centered: bool = True,
) -> None:
    w_data = w_data.view(-1, 1)
    w_chain = w_chain.view(-1, 1)
    # Turn the weights of the chains into normalized weights
    chain_weights = w_chain / w_chain.sum()
    w_data_norm = w_data.sum()

    # The statistics conjugated to W2 and -precision are s = v^2 / 2
    s_data = 0.5 * v_data.square()
    s_chain = 0.5 * v_chain.square()

    # Averages over data and generated samples
    v_data_mean = (v_data * w_data).sum(0) / w_data_norm
    s_data_mean = (s_data * w_data).sum(0) / w_data_norm
    h_data_mean = (mh_data * w_data).sum(0) / w_data_norm
    v_gen_mean = (v_chain * chain_weights).sum(0)
    s_gen_mean = (s_chain * chain_weights).sum(0)
    h_gen_mean = (h_chain * chain_weights).sum(0)

    if centered:
        # Centered variables
        v_data_centered = v_data - v_data_mean
        s_data_centered = s_data - s_data_mean
        h_data_centered = mh_data - h_data_mean
        v_gen_centered = v_chain - v_data_mean
        s_gen_centered = s_chain - s_data_mean
        h_gen_centered = h_chain - h_data_mean

        # Gradient
        grad_weight_matrix = (
            (v_data_centered * w_data).T @ h_data_centered
        ) / w_data_norm - ((v_gen_centered * chain_weights).T @ h_gen_centered)
        grad_weight_matrix_2 = (
            (s_data_centered * w_data).T @ h_data_centered
        ) / w_data_norm - ((s_gen_centered * chain_weights).T @ h_gen_centered)
        grad_vbias = v_data_mean - v_gen_mean - (grad_weight_matrix @ h_data_mean)
        grad_precision = -(
            s_data_mean - s_gen_mean - (grad_weight_matrix_2 @ h_data_mean)
        )
        grad_hbias = (
            h_data_mean
            - h_gen_mean
            - (v_data_mean @ grad_weight_matrix)
            - (s_data_mean @ grad_weight_matrix_2)
        )
    else:
        # Gradient
        grad_weight_matrix = ((v_data * w_data).T @ mh_data) / w_data_norm - (
            (v_chain * chain_weights).T @ h_chain
        )
        grad_weight_matrix_2 = ((s_data * w_data).T @ mh_data) / w_data_norm - (
            (s_chain * chain_weights).T @ h_chain
        )
        grad_vbias = v_data_mean - v_gen_mean
        grad_precision = -(s_data_mean - s_gen_mean)
        grad_hbias = h_data_mean - h_gen_mean

    # Attach to the parameters
    weight_matrix.grad = grad_weight_matrix
    weight_matrix_2.grad = grad_weight_matrix_2
    vbias.grad = grad_vbias
    hbias.grad = grad_hbias
    precision.grad = grad_precision


@torch.no_grad()
def _enforce_positive_precision(
    precision: Tensor, weight_matrix_2: Tensor, min_precision: Tensor
) -> None:
    """Raise (in-place) the precision so that P_i(h) = lambda_i - sum_a W2_ia h_a
    is at least `min_precision` for every hidden configuration."""
    precision.copy_(
        torch.maximum(precision, weight_matrix_2.clamp(min=0).sum(1) + min_precision)
    )


def _init_chains(
    num_samples: int,
    weight_matrix: Tensor,
    weight_matrix_2: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    precision: Tensor,
    start_v: Tensor | None = None,
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
        # Sample from the independent model (W = W2 = 0)
        mv = (vbias / precision).expand(num_samples, num_visibles).clone()
        v = mv + torch.randn_like(mv) / torch.sqrt(precision)
    else:
        v = start_v.to(device=device, dtype=dtype)
        mv = (vbias / precision).expand_as(v).clone()

    # Initialize chains
    h, mh = _sample_hiddens(
        v=v, weight_matrix=weight_matrix, weight_matrix_2=weight_matrix_2, hbias=hbias
    )
    return v, h, mv, mh


def _init_parameters(
    num_hiddens: int,
    data: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    var_init: float = 1e-4,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    _, num_visibles = data.shape
    data = data.to(device=device, dtype=dtype)
    weight_matrix = (
        torch.randn(size=(num_visibles, num_hiddens), device=device, dtype=dtype)
        * var_init
    )
    weight_matrix_2 = torch.zeros(
        size=(num_visibles, num_hiddens), device=device, dtype=dtype
    )
    # Constant visibles (e.g. border pixels) have zero std: put a floor on sigma
    min_sigma = max(1e-1 * data.std().item(), 1e-6)
    sigma = data.std(0).clamp(min=min_sigma)
    precision = sigma.square().reciprocal()
    # The conditional std can grow at most to sqrt(10) times the initial one
    min_precision = 0.1 * precision
    # Match the mean of the data with the independent model
    vbias = data.mean(0) * precision
    hbias = torch.zeros(num_hiddens, device=device, dtype=dtype)
    return vbias, hbias, weight_matrix, weight_matrix_2, precision, min_precision
