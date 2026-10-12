import numpy as np
import torch
from scipy.stats import truncnorm

from rbms.truncated_gaussian_bernoulli.fused import (
    HAS_TRITON,
    can_use_fused,
    sample_truncated_normal_fused,
)
from rbms.truncated_gaussian_bernoulli.implement import _truncated_normal_moments

try:
    import pytest

    pytestmark = pytest.mark.skipif(
        not (HAS_TRITON and torch.cuda.is_available()), reason="needs triton and a GPU"
    )
except ImportError:
    pass
DEVICE = "cuda"


def test_mean_matches_float64_reference():
    # Locations from far below to far above the box [0, 1], several scales
    sigma = torch.tensor([0.01, 0.03, 0.1, 0.3, 1.0, 3.0], device=DEVICE)
    mu = torch.linspace(-3, 4, 20001, device=DEVICE).unsqueeze(1).expand(-1, len(sigma)).contiguous()
    vmin, vmax = torch.zeros(len(sigma), device=DEVICE), torch.ones(len(sigma), device=DEVICE)
    _, mean = sample_truncated_normal_fused(mu, sigma, vmin, vmax)
    mu64, s64 = mu.double(), sigma.double()
    ref = mu64 + s64 * _truncated_normal_moments((0 - mu64) / s64, (1 - mu64) / s64)
    assert torch.isfinite(mean).all()
    assert (mean >= 0).all() and (mean <= 1).all()
    torch.testing.assert_close(mean.double(), ref, atol=2e-5, rtol=1e-4)


INTERVALS = [
    (-1.0, 2.0),
    (8.0, 9.0),
    (-50.0, -40.0),
    (30.0, 31.0),
    (-1e-3, 1e-3),
    (-3.0, 60.0),
    (-60.0, 3.0),
    (0.5, np.inf),
    (-np.inf, -3.0),
    (-np.inf, np.inf),
]


def test_samples_match_scipy():
    torch.manual_seed(0)
    n = 400000
    loc, scale = 0.3, 0.7
    vmin = torch.tensor([loc + scale * x for x, _ in INTERVALS], device=DEVICE)
    vmax = torch.tensor([loc + scale * y for _, y in INTERVALS], device=DEVICE)
    sigma = torch.full((len(INTERVALS),), scale, device=DEVICE)
    mu = torch.full((n, len(INTERVALS)), loc, device=DEVICE)
    v, mean = sample_truncated_normal_fused(mu, sigma, vmin, vmax)
    assert torch.isfinite(v).all()
    assert (v >= vmin).all() and (v <= vmax).all()
    v = v.double().cpu()
    for j, (x, y) in enumerate(INTERVALS):
        ref = truncnorm(x, y, loc=loc, scale=scale)
        np.testing.assert_allclose(mean[0, j].item(), ref.mean(), rtol=1e-4, atol=1e-6)
        tol = 5 * ref.std() / np.sqrt(n) + 1e-6 * abs(ref.mean())
        assert abs(v[:, j].mean().item() - ref.mean()) < tol, (x, y)
        np.testing.assert_allclose(v[:, j].std().item(), ref.std(), rtol=2e-2, atol=1e-7)
        # Quantiles (inverse CDF accuracy)
        qs = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        np.testing.assert_allclose(
            np.quantile(v[:, j].numpy(), qs), ref.ppf(qs), atol=0.02 * ref.std() + 1e-6
        )


def test_stress_and_seed():
    torch.manual_seed(1)
    n, nv = 100000, 1000
    mu = (torch.randn(n, nv, device=DEVICE) * 3).contiguous()
    sigma = torch.rand(nv, device=DEVICE) * 0.5 + 0.01
    vmin, vmax = torch.zeros(nv, device=DEVICE), torch.ones(nv, device=DEVICE)
    assert can_use_fused(mu, sigma, vmin, vmax)
    v, mean = sample_truncated_normal_fused(mu, sigma, vmin, vmax)
    assert torch.isfinite(v).all() and torch.isfinite(mean).all()
    assert (v >= 0).all() and (v <= 1).all()
    torch.manual_seed(2)
    v1, _ = sample_truncated_normal_fused(mu[:10], sigma, vmin, vmax)
    torch.manual_seed(2)
    v2, _ = sample_truncated_normal_fused(mu[:10], sigma, vmin, vmax)
    torch.testing.assert_close(v1, v2)
