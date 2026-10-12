import math

import torch
from torch import Tensor
from torch.special import log_ndtr, ndtri

from rbms.truncated_gaussian_bernoulli.fused import (
    can_use_fused,
    sample_truncated_normal_fused,
)

# Truncated Gaussian-Bernoulli RBM: same energy as the GBRBM,
#   E(v, h) = sum_i v_i^2 / (2 sigma_i^2) - sum_i b_i v_i - sum_a c_a h_a - sum_ia v_i W_ia h_a,
# but the visibles live in the box [vmin_i, vmax_i] (E = +inf outside). Hence
#   p(v_i | h) = TruncatedNormal(mu_i = sigma_i^2 (b_i + sum_a W_ia h_a), sigma_i^2, [vmin_i, vmax_i])
# and p(h_a = 1 | v) = sigmoid(c_a + sum_i v_i W_ia) as in the GBRBM.
# At inverse temperature beta the std becomes sigma / sqrt(beta): at beta = 0 the visibles
# are uniform in the box (finite bounds), so tempering down to beta = 0 is well defined.

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
LOG_2PI = math.log(2 * math.pi)
INV_SQRT2 = 1 / math.sqrt(2)
SQRT_2_OVER_PI = math.sqrt(2 / math.pi)


def _flip_to_left_tail(alpha: Tensor, beta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Use Phi(b) - Phi(a) = Phi(-a) - Phi(-b) so that the interval [alpha, beta] lies
    mostly in the left tail, where Phi is computed accurately in log scale."""
    flip = (alpha + beta) > 0
    left = torch.where(flip, -beta, alpha)
    right = torch.where(flip, -alpha, beta)
    sign = torch.where(flip, -1.0, 1.0).to(alpha.dtype)
    return left, right, sign


def _log_ndtr_diff(alpha: Tensor, beta: Tensor) -> Tensor:
    """log(Phi(beta) - Phi(alpha)) for alpha <= beta, stable in the tails."""
    left, right, _ = _flip_to_left_tail(alpha, beta)
    log_phi_left = log_ndtr(left)
    log_phi_right = log_ndtr(right)
    return log_phi_right + torch.log1p(-torch.exp(log_phi_left - log_phi_right))


def _truncated_offset_left(
    left: Tensor, right: Tensor, log_phi_left: Tensor, log_phi_right: Tensor
) -> tuple[Tensor, Tensor]:
    """E[z] - right for z ~ N(0, 1) truncated to [left, right], with the interval in the
    left tail (left + right <= 0), from the already computed log Phi(left), log Phi(right).
    Also returns E[z] computed directly, to be used when `right` is infinite.

    The mean is returned relative to the edge `right`: when the interval is far in the
    tail, E[z] ~ right and computing mu + std * E[z] directly loses all precision in
    float32. Far in the tail (right < -1) we use the Mills ratios phi(x) / Phi(x), written
    with erfcx so that no large exponent appears.
    """
    # Bulk: phi(x) / Z in log scale (phi(-inf) = 0)
    log_z = log_phi_right + torch.log1p(-torch.exp(log_phi_left - log_phi_right))
    pdf_left = torch.exp(-0.5 * left.square() - LOG_SQRT_2PI - log_z)
    pdf_right = torch.exp(-0.5 * right.square() - LOG_SQRT_2PI - log_z)
    pdf_left = torch.where(torch.isinf(left), 0.0, pdf_left)
    mean_direct = pdf_left - pdf_right
    # Tail: E[z] = (q - M(right)) / (1 - r), with M(x) = phi(x) / Phi(x),
    # r = Phi(left) / Phi(right) and q = phi(left) / Phi(right) = M(left) r
    r_tail = torch.clamp(right, max=-1.0)
    l_tail = torch.minimum(left, r_tail)
    erfcx_r = torch.special.erfcx(-r_tail * INV_SQRT2)
    erfcx_l = torch.special.erfcx(-l_tail * INV_SQRT2)
    mills_r = SQRT_2_OVER_PI / erfcx_r
    ratio = torch.exp(
        torch.log(erfcx_l / erfcx_r) - 0.5 * (l_tail - r_tail) * (l_tail + r_tail)
    )
    q = torch.where(ratio > 0, SQRT_2_OVER_PI / erfcx_l * ratio, 0.0)
    tail = (q - mills_r - r_tail * (1 - ratio)) / (1 - ratio)
    return torch.where(right < -1, tail, mean_direct - right), mean_direct


def _truncated_normal_moments(alpha: Tensor, beta: Tensor) -> Tensor:
    """Mean of a standard normal truncated to [alpha, beta]."""
    left, right, sign = _flip_to_left_tail(alpha, beta)
    offset, mean_direct = _truncated_offset_left(left, right, log_ndtr(left), log_ndtr(right))
    return sign * torch.where(torch.isinf(right), mean_direct, right + offset)


def _ndtri_log(log_p: Tensor) -> Tensor:
    """Inverse of log Phi: z such that log Phi(z) = log_p.

    torch.special.ndtri(exp(log_p)) loses everything once exp(log_p) underflows
    (log_p < -87 in float32). Below a threshold we start from the asymptotic expansion
    log Phi(z) ~ -z^2 / 2 - log(-z) - log(2 pi) / 2 and refine with two Newton steps on
    log Phi, which only needs log_ndtr and is accurate in any precision.
    """
    threshold = math.log(torch.finfo(log_p.dtype).tiny) + 10.0
    # For p > 1/2 use Phi^-1(p) = -Phi^-1(1 - p), with 1 - p = -expm1(log_p) accurate
    z_bulk = torch.where(
        log_p < math.log(0.5), ndtri(torch.exp(log_p)), -ndtri(-torch.expm1(log_p))
    )
    # Clamp so that the unused branch stays finite
    log_p_tail = torch.clamp(log_p, max=threshold)
    t = -2.0 * log_p_tail
    z = -torch.sqrt(t - torch.log(t) - LOG_2PI)
    for _ in range(2):
        log_phi = log_ndtr(z)
        # Newton on f(z) = log Phi(z) - log_p, with f'(z) = phi(z) / Phi(z)
        z = z - (log_phi - log_p_tail) * torch.exp(log_phi + 0.5 * z.square() + LOG_SQRT_2PI)
    z = torch.where(log_p < threshold, z, z_bulk)
    # log_p = -inf (p = 0) would give inf - inf in the tail branch
    return torch.where(torch.isneginf(log_p), -torch.inf, z)


def _sample_truncated_normal(
    mu: Tensor, std: Tensor, vmin: Tensor, vmax: Tensor
) -> tuple[Tensor, Tensor]:
    """Sample N(mu, std^2) truncated to [vmin, vmax] by inverse CDF.

    The interval is moved to the left tail (see `_flip_to_left_tail`), where log Phi is
    accurate, and the inverse CDF is computed in log scale (see `_ndtri_log`), so that
    the computation is accurate in the working precision (float32), also when the
    interval is many standard deviations away from mu.

    Returns:
        tuple[Tensor, Tensor]: samples and mean of the truncated normal.
    """
    alpha = (vmin - mu) / std
    beta = (vmax - mu) / std
    left, right, sign = _flip_to_left_tail(alpha, beta)

    log_phi_left = log_ndtr(left)
    log_phi_right = log_ndtr(right)
    # u in (0, 1]: torch.rand can return exactly 0, which is not rare in float32 (2^-24)
    u = 1 - torch.rand_like(mu)
    # Phi(z) = Phi(left) + u (Phi(right) - Phi(left)), in log scale
    log_phi_z = log_phi_right + torch.log(
        u + (1 - u) * torch.exp(log_phi_left - log_phi_right)
    )
    z = _ndtri_log(log_phi_z)
    z = torch.maximum(torch.minimum(z, right), left)

    v = mu + std * sign * z
    v = torch.maximum(torch.minimum(v, vmax), vmin)
    # Mean relative to the edge of the box that corresponds to `right`
    edge = torch.where(sign > 0, vmax, vmin)
    offset, mean_direct = _truncated_offset_left(left, right, log_phi_left, log_phi_right)
    mean = torch.where(
        torch.isinf(right), mu + std * sign * mean_direct, edge + std * sign * offset
    )
    mean = torch.maximum(torch.minimum(mean, vmax), vmin)
    return v, mean


def _sample_visibles(
    h: Tensor,
    weight_matrix: Tensor,
    vbias: Tensor,
    sigma: Tensor,
    vmin: Tensor,
    vmax: Tensor,
    beta: float = 1.0,
) -> tuple[Tensor, Tensor]:
    mu = sigma.square() * (vbias + (h @ weight_matrix.mT))
    if beta == 0:
        # Uniform in the box
        v = vmin + torch.rand_like(mu) * (vmax - vmin)
        return v, (0.5 * (vmin + vmax)).expand_as(mu)
    if can_use_fused(mu, sigma, vmin, vmax):
        # Single Triton kernel on GPU (same algorithm)
        return sample_truncated_normal_fused(mu, sigma, vmin, vmax, scale=beta**-0.5)
    return _sample_truncated_normal(
        mu=mu, std=(sigma / beta**0.5).expand_as(mu), vmin=vmin, vmax=vmax
    )


def _log_partition_visibles(
    field: Tensor, sigma: Tensor, vmin: Tensor, vmax: Tensor
) -> Tensor:
    """log int_{vmin}^{vmax} exp(-v^2 / (2 sigma^2) + field v) dv, element-wise."""
    mu = sigma.square() * field
    alpha = (vmin - mu) / sigma
    beta = (vmax - mu) / sigma
    return (
        0.5 * sigma.square() * field.square()
        + torch.log(sigma)
        + LOG_SQRT_2PI
        + _log_ndtr_diff(alpha, beta)
    )


def _compute_energy_hiddens(
    h: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    weight_matrix: Tensor,
    sigma: Tensor,
    vmin: Tensor,
    vmax: Tensor,
) -> Tensor:
    field = vbias + (h @ weight_matrix.T)
    log_z_visibles = _log_partition_visibles(field, sigma, vmin, vmax)
    return -(h @ hbias) - log_z_visibles.sum(1)


def _init_vbias(
    data_mean: Tensor, sigma: Tensor, vmin: Tensor, vmax: Tensor, num_iter: int = 100
) -> Tensor:
    """Visible bias such that the mean of the independent model (W = 0) matches the
    mean of the data. The truncated mean is increasing in mu: bisection on mu."""
    # The data mean must be strictly inside the box for a finite mu
    eps = 1e-3 * (vmax - vmin)
    target = torch.minimum(torch.maximum(data_mean, vmin + eps), vmax - eps)
    width = (vmax - vmin) + 50 * sigma
    lo, hi = vmin - width, vmax + width
    for _ in range(num_iter):
        mid = 0.5 * (lo + hi)
        mean = mid + sigma * _truncated_normal_moments(
            (vmin - mid) / sigma, (vmax - mid) / sigma
        )
        above = mean > target
        hi = torch.where(above, mid, hi)
        lo = torch.where(above, lo, mid)
    return 0.5 * (lo + hi) / sigma.square()
