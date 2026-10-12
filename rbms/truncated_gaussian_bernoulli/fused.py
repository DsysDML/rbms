"""Fused (single kernel) sampling of the truncated normal visibles of the TGBRBM on GPU.

Same algorithm as `implement._sample_truncated_normal` (left-tail flip, inverse CDF in
log scale, Newton refinement in the deep tail, mean of the truncated normal), but
written as one Triton kernel: each element is read and written once instead of once
per element-wise operation, which is what limits the PyTorch version on GPU.
"""

import math

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
LOG_2PI = math.log(2 * math.pi)
INV_SQRT2 = 1 / math.sqrt(2)
SQRT_2_OVER_PI = math.sqrt(2 / math.pi)
LOG_HALF = math.log(0.5)
# exp(log_p) is a normal float32 above this value (log of the smallest normal + margin)
LOG_P_TAIL = math.log(torch.finfo(torch.float32).tiny) + 10.0

if HAS_TRITON:
    # Globals used inside Triton kernels must be constexpr
    _LOG_SQRT_2PI = tl.constexpr(LOG_SQRT_2PI)
    _LOG_2PI = tl.constexpr(LOG_2PI)
    _INV_SQRT2 = tl.constexpr(INV_SQRT2)
    _SQRT_2_OVER_PI = tl.constexpr(SQRT_2_OVER_PI)
    _LOG_HALF = tl.constexpr(LOG_HALF)
    _LOG_P_TAIL = tl.constexpr(LOG_P_TAIL)

    @triton.jit
    def _log_ndtr(x):
        # log Phi(x): erfcx form in the left tail (no underflow), erfc form elsewhere
        left = tl.log(0.5 * libdevice.erfcx(-x * _INV_SQRT2)) - 0.5 * x * x
        right = libdevice.log1p(-0.5 * libdevice.erfc(x * _INV_SQRT2))
        return tl.where(x < -1.0, left, right)

    # The seed changes at every call: do not specialize the kernel on its value
    @triton.jit(do_not_specialize=["seed"])
    def _truncated_normal_kernel(
        mu_ptr,
        sigma_ptr,
        vmin_ptr,
        vmax_ptr,
        v_ptr,
        mean_ptr,
        scale,
        n_elements,
        num_visibles,
        seed,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        col = offs % num_visibles
        mu = tl.load(mu_ptr + offs, mask=mask, other=0.0)
        std = tl.load(sigma_ptr + col, mask=mask, other=1.0) * scale
        lo = tl.load(vmin_ptr + col, mask=mask, other=0.0)
        hi = tl.load(vmax_ptr + col, mask=mask, other=1.0)

        alpha = (lo - mu) / std
        beta = (hi - mu) / std
        # Move the interval to the left tail: Phi(b) - Phi(a) = Phi(-a) - Phi(-b)
        flip = (alpha + beta) > 0
        a = tl.where(flip, -beta, alpha)
        b = tl.where(flip, -alpha, beta)
        sign = tl.where(flip, -1.0, 1.0)

        log_phi_a = _log_ndtr(a)
        log_phi_b = _log_ndtr(b)
        ratio = tl.exp(log_phi_a - log_phi_b)
        # u in (0, 1]
        u = 1.0 - tl.rand(seed, offs)
        log_p = log_phi_b + tl.log(u + (1.0 - u) * ratio)

        # Inverse of log Phi
        z_low = libdevice.normcdfinv(tl.exp(log_p))
        z_high = -libdevice.normcdfinv(-libdevice.expm1(log_p))
        z_bulk = tl.where(log_p < _LOG_HALF, z_low, z_high)
        # Deep tail: asymptotic start + 2 Newton steps on log Phi
        log_p_tail = tl.minimum(log_p, _LOG_P_TAIL)
        t = -2.0 * log_p_tail
        z_tail = -tl.sqrt(t - tl.log(t) - _LOG_2PI)
        lp = _log_ndtr(z_tail)
        z_tail = z_tail - (lp - log_p_tail) * tl.exp(lp + 0.5 * z_tail * z_tail + _LOG_SQRT_2PI)
        lp = _log_ndtr(z_tail)
        z_tail = z_tail - (lp - log_p_tail) * tl.exp(lp + 0.5 * z_tail * z_tail + _LOG_SQRT_2PI)
        z = tl.where(log_p < _LOG_P_TAIL, z_tail, z_bulk)
        z = tl.where(z != z, a, z)  # NaN guard (never expected)
        z = tl.minimum(tl.maximum(z, a), b)

        v = mu + std * sign * z
        v = tl.minimum(tl.maximum(v, lo), hi)

        # Mean of the truncated normal, relative to the edge corresponding to b
        # (see implement._truncated_offset_left)
        log_z = log_phi_b + libdevice.log1p(-ratio)
        pdf_a = tl.exp(-0.5 * a * a - _LOG_SQRT_2PI - log_z)
        pdf_b = tl.exp(-0.5 * b * b - _LOG_SQRT_2PI - log_z)
        pdf_a = tl.where(a == -float("inf"), 0.0, pdf_a)
        mean_direct = pdf_a - pdf_b
        b_t = tl.minimum(b, -1.0)
        a_t = tl.minimum(a, b_t)
        erfcx_b = libdevice.erfcx(-b_t * _INV_SQRT2)
        erfcx_a = libdevice.erfcx(-a_t * _INV_SQRT2)
        r = tl.exp(tl.log(erfcx_a / erfcx_b) - 0.5 * (a_t - b_t) * (a_t + b_t))
        q = tl.where(r > 0, _SQRT_2_OVER_PI / erfcx_a * r, 0.0)
        tail = (q - _SQRT_2_OVER_PI / erfcx_b - b_t * (1.0 - r)) / (1.0 - r)
        offset = tl.where(b < -1.0, tail, mean_direct - b)
        edge = tl.where(flip, lo, hi)
        # An infinite edge (unbounded box) cannot be the reference: direct formula
        mean = tl.where(
            b == float("inf"), mu + std * sign * mean_direct, edge + std * sign * offset
        )
        mean = tl.minimum(tl.maximum(mean, lo), hi)

        tl.store(v_ptr + offs, v, mask=mask)
        tl.store(mean_ptr + offs, mean, mask=mask)


def can_use_fused(mu: Tensor, sigma: Tensor, vmin: Tensor, vmax: Tensor) -> bool:
    """The fused kernel handles float32 CUDA tensors with mu of shape
    (num_samples, num_visibles) and per-visible sigma, vmin, vmax."""
    return (
        HAS_TRITON
        and mu.is_cuda
        and mu.dtype == torch.float32
        and mu.dim() == 2
        and all(
            x.dim() == 1 and x.shape[0] == mu.shape[1] and x.dtype == torch.float32
            for x in (sigma, vmin, vmax)
        )
    )


def sample_truncated_normal_fused(
    mu: Tensor, sigma: Tensor, vmin: Tensor, vmax: Tensor, scale: float = 1.0
) -> tuple[Tensor, Tensor]:
    """Sample N(mu, (scale * sigma)^2) truncated to [vmin, vmax] in one kernel.

    Args:
        mu (Tensor): Locations, shape (num_samples, num_visibles).
        sigma (Tensor): Per-visible scale, shape (num_visibles,).
        vmin (Tensor): Per-visible lower bound, shape (num_visibles,).
        vmax (Tensor): Per-visible upper bound, shape (num_visibles,).
        scale (float): Global factor on sigma (1 / sqrt(beta)).

    Returns:
        tuple[Tensor, Tensor]: samples and mean of the truncated normal.
    """
    mu = mu.contiguous()
    v = torch.empty_like(mu)
    mean = torch.empty_like(mu)
    n = mu.numel()
    # Seed of the Philox generator of the kernel, drawn from the torch generator so
    # that torch.manual_seed controls the samples
    seed = int(torch.randint(0, 2**31 - 1, (1,), device="cpu").item())
    block = 1024
    # Launch on the device of the tensors (Triton uses the current device)
    with torch.cuda.device(mu.device):
        _truncated_normal_kernel[(triton.cdiv(n, block),)](
            mu,
            sigma.contiguous(),
            vmin.contiguous(),
            vmax.contiguous(),
            v,
            mean,
            float(scale),
            n,
            mu.shape[1],
            seed,
            BLOCK=block,
        )
    return v, mean
