import numpy as np
import torch
from torch import Tensor

# Energy of the Gaussian-Bernoulli RBM (fixed visible variances sigma_i^2):
#   E(v, h) = sum_i v_i^2 / (2 sigma_i^2) - sum_i b_i v_i - sum_a c_a h_a - sum_ia v_i W_ia h_a
# so that p(v_i | h) = N(sigma_i^2 (b_i + sum_a W_ia h_a), sigma_i^2)
# and     p(h_a = 1 | v) = sigmoid(c_a + sum_i v_i W_ia).
# At inverse temperature beta the whole energy is multiplied by beta.


@torch.jit.script
def _sample_hiddens(
    v: Tensor, weight_matrix: Tensor, hbias: Tensor, beta: float = 1.0
) -> tuple[Tensor, Tensor]:
    mh = torch.sigmoid(beta * (hbias + (v @ weight_matrix)))
    h = torch.bernoulli(mh)
    return h, mh


@torch.jit.script
def _sample_visibles(
    h: Tensor, weight_matrix: Tensor, vbias: Tensor, sigma: Tensor, beta: float = 1.0
) -> tuple[Tensor, Tensor]:
    mv = sigma.square() * (vbias + (h @ weight_matrix.T))
    v = mv + torch.randn_like(mv) * (sigma / beta**0.5)
    return v, mv


@torch.jit.script
def _compute_energy(
    v: Tensor,
    h: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    weight_matrix: Tensor,
    sigma: Tensor,
) -> Tensor:
    quad = 0.5 * (v / sigma).square().sum(1)
    fields = v @ vbias + h @ hbias
    interaction = ((v @ weight_matrix) * h).sum(1)
    return quad - fields - interaction


@torch.jit.script
def _compute_energy_visibles(
    v: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor, sigma: Tensor
) -> Tensor:
    quad = 0.5 * (v / sigma).square().sum(1)
    field = v @ vbias
    exponent = hbias + (v @ weight_matrix)
    log_term = torch.nn.functional.softplus(exponent)
    return quad - field - log_term.sum(1)


def _compute_energy_visibles_gradient(
    v: Tensor, hbias: Tensor, weight_matrix: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    # F(v) = sum_i v_i^2 / (2 sigma_i^2) - v.vbias - sum_a softplus(hbias_a + (v W)_a): sigma is
    # fixed, the derivatives with respect to the trained parameters are those of the BBRBM
    mean_hidden = torch.sigmoid(hbias + (v @ weight_matrix))
    grad_vbias = -v
    grad_hbias = -mean_hidden
    grad_weight_matrix = -v.T @ mean_hidden
    return grad_vbias, grad_hbias, grad_weight_matrix


@torch.jit.script
def _compute_energy_hiddens(
    h: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor, sigma: Tensor
) -> Tensor:
    field = h @ hbias
    var = sigma.square()
    gaussian_term = 0.5 * var * (vbias + (h @ weight_matrix.T)).square() + 0.5 * torch.log(
        2 * np.pi * var
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
    centered: bool = True,
    weight_mask: Tensor | None = None,
) -> None:
    w_data = w_data.view(-1, 1)
    w_chain = w_chain.view(-1, 1)
    # Turn the weights of the chains into normalized weights
    chain_weights = w_chain / w_chain.sum()
    w_data_norm = w_data.sum()

    # Averages over data and generated samples
    v_data_mean = (v_data * w_data).sum(0) / w_data_norm
    h_data_mean = (mh_data * w_data).sum(0) / w_data_norm
    v_gen_mean = (v_chain * chain_weights).sum(0)
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
        if weight_mask is not None:
            grad_weight_matrix = grad_weight_matrix * weight_mask
        grad_vbias = v_data_mean - v_gen_mean - (grad_weight_matrix @ h_data_mean)
        grad_hbias = h_data_mean - h_gen_mean - (v_data_mean @ grad_weight_matrix)
    else:
        # Gradient
        grad_weight_matrix = ((v_data * w_data).T @ mh_data) / w_data_norm - (
            (v_chain * chain_weights).T @ h_chain
        )
        if weight_mask is not None:
            grad_weight_matrix = grad_weight_matrix * weight_mask
        grad_vbias = v_data_mean - v_gen_mean
        grad_hbias = h_data_mean - h_gen_mean

    # Attach to the parameters
    weight_matrix.grad = grad_weight_matrix
    vbias.grad = grad_vbias
    hbias.grad = grad_hbias


def _init_chains(
    num_samples: int,
    weight_matrix: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    sigma: Tensor,
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
        # Sample from the independent model (W = 0)
        mv = (sigma.square() * vbias).expand(num_samples, num_visibles).clone()
        v = mv + torch.randn_like(mv) * sigma
    else:
        v = start_v.to(device=device, dtype=dtype)
        mv = (sigma.square() * vbias).expand_as(v).clone()

    # Initialize chains
    h, mh = _sample_hiddens(v=v, weight_matrix=weight_matrix, hbias=hbias)
    return v, h, mv, mh


def _init_parameters(
    num_hiddens: int,
    data: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    var_init: float = 1e-4,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _, num_visibles = data.shape
    data = data.to(device=device, dtype=dtype)
    weight_matrix = (
        torch.randn(size=(num_visibles, num_hiddens), device=device, dtype=dtype)
        * var_init
    )
    # Constant visibles (e.g. border pixels) have zero std: put a floor on sigma
    min_sigma = max(1e-1 * data.std().item(), 1e-6)
    sigma = data.std(0).clamp(min=min_sigma)
    # Match the mean of the data with the independent model
    vbias = data.mean(0) / sigma.square()
    hbias = torch.zeros(num_hiddens, device=device, dtype=dtype)
    return vbias, hbias, weight_matrix, sigma


def _drop_identity_normalization(named_params: dict[str, np.ndarray]) -> None:
    """Remove (in-place) the keys `v_mean` and `v_std` of archives saved while the
    Gaussian RBMs standardized the visibles internally (feature removed). Only an
    identity transformation (data already standardized) can be loaded without it."""
    if "v_mean" not in named_params and "v_std" not in named_params:
        return
    v_mean = named_params.pop("v_mean")
    v_std = named_params.pop("v_std")
    if not (np.allclose(v_mean, 0, atol=1e-3) and np.allclose(v_std, 1, atol=1e-3)):
        raise ValueError(
            "This model was trained with the visibles standardized internally "
            "(v_mean, v_std), which is no longer supported: train it again."
        )
