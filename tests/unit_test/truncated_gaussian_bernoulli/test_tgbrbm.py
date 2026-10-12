import numpy as np
import torch
from scipy.special import log_ndtr, ndtri
from scipy.stats import truncnorm

from rbms.dataset.dataset_class import RBMDataset
from rbms.gaussian_bernoulli.classes import GBRBM
from rbms.partition_function.exact import compute_partition_function
from rbms.truncated_gaussian_bernoulli.classes import TGBRBM
from rbms.truncated_gaussian_bernoulli.implement import (
    _log_ndtr_diff,
    _ndtri_log,
    _sample_truncated_normal,
    _truncated_normal_moments,
)
from rbms.utils import get_categorical_configurations

NUM_VISIBLES = 4
NUM_HIDDENS = 3
DTYPE = torch.float64


def make_params(seed=0, bounded=True):
    torch.manual_seed(seed)
    vmin = torch.zeros(NUM_VISIBLES, dtype=DTYPE) if bounded else None
    vmax = torch.ones(NUM_VISIBLES, dtype=DTYPE) if bounded else None
    return TGBRBM(
        weight_matrix=torch.randn(NUM_VISIBLES, NUM_HIDDENS, dtype=DTYPE),
        vbias=torch.randn(NUM_VISIBLES, dtype=DTYPE),
        hbias=torch.randn(NUM_HIDDENS, dtype=DTYPE),
        sigma=0.3 * torch.rand(NUM_VISIBLES, dtype=DTYPE) + 0.2,
        vmin=vmin,
        vmax=vmax,
    )


def all_hiddens():
    return get_categorical_configurations(n_states=2, n_dim=NUM_HIDDENS, dtype=DTYPE)


# Standardized truncation intervals: central, one-sided, far tails on both sides,
# very narrow, and unbounded
INTERVALS = [
    (-1.0, 2.0),
    (0.5, np.inf),
    (-np.inf, -3.0),
    (8.0, 9.0),
    (-50.0, -40.0),
    (30.0, 31.0),
    (-1e-3, 1e-3),
    (-np.inf, np.inf),
]


def test_log_ndtr_diff_and_mean_match_scipy():
    a = torch.tensor([i[0] for i in INTERVALS], dtype=DTYPE)
    b = torch.tensor([i[1] for i in INTERVALS], dtype=DTYPE)
    # log(Phi(b) - Phi(a)) with scipy, using the tail where Phi is accurate
    expected_log_z = np.array(
        [
            np.log(-np.expm1(log_ndtr(x) - log_ndtr(y))) + log_ndtr(y)
            if x + y <= 0
            else np.log(-np.expm1(log_ndtr(-y) - log_ndtr(-x))) + log_ndtr(-x)
            for x, y in INTERVALS
        ]
    )
    np.testing.assert_allclose(_log_ndtr_diff(a, b).numpy(), expected_log_z, rtol=1e-10)
    expected_mean = np.array([truncnorm(x, y).mean() for x, y in INTERVALS])
    np.testing.assert_allclose(
        _truncated_normal_moments(a, b).numpy(), expected_mean, rtol=1e-6, atol=1e-10
    )


def test_ndtri_log_matches_scipy():
    # From the bulk to very deep tails, in float32 and float64
    log_p = np.concatenate([-np.logspace(-6, np.log10(5000), 300), [np.log(0.5)]])
    # Reference: ndtri(exp(log_p)) in the bulk, Newton in float64 in the deep tail
    z_ref = np.where(log_p > -700, ndtri(np.exp(np.maximum(log_p, -700))), 0.0)
    deep = log_p <= -700
    z = -np.sqrt(-2 * log_p[deep])
    for _ in range(50):
        lp = log_ndtr(z)
        z = z - (lp - log_p[deep]) * np.exp(lp + 0.5 * z**2 + 0.5 * np.log(2 * np.pi))
    z_ref[deep] = z
    for dtype, rtol in [(torch.float32, 2e-5), (torch.float64, 1e-9)]:
        z = _ndtri_log(torch.tensor(log_p, dtype=dtype)).double().numpy()
        np.testing.assert_allclose(z, z_ref, rtol=rtol, atol=rtol)


def test_sampling_extreme_uniforms(monkeypatch=None):
    # torch.rand can return exactly 0 (probability 2^-24 in float32) or values close to 1
    orig = torch.rand_like
    mu = torch.tensor([[-0.73, 0.5, 1.4, -5.0]], dtype=torch.float32).expand(4, -1)
    std = torch.tensor([0.105, 0.3, 0.2, 0.03], dtype=torch.float32).expand(4, -1)
    vmin, vmax = torch.zeros(4), torch.ones(4)
    assert torch.isneginf(_ndtri_log(torch.tensor([-torch.inf])))
    try:
        for value in [0.0, 1 - 2**-24, 0.5]:
            torch.rand_like = lambda x, value=value: torch.full_like(x, value)
            v, mean = _sample_truncated_normal(mu, std, vmin, vmax)
            assert torch.isfinite(v).all() and torch.isfinite(mean).all(), value
            assert (v >= vmin).all() and (v <= vmax).all()
    finally:
        torch.rand_like = orig


def test_sampling_matches_scipy():
    n = 200000
    for dtype, rtol_mean in [(torch.float64, 1e-6), (torch.float32, 1e-4)]:
        torch.manual_seed(1)
        for x, y in INTERVALS:
            mu = torch.full((n, 1), 0.3, dtype=dtype)
            std = torch.full((n, 1), 0.7, dtype=dtype)
            vmin = torch.tensor([0.3 + 0.7 * x], dtype=dtype)
            vmax = torch.tensor([0.3 + 0.7 * y], dtype=dtype)
            v, mean = _sample_truncated_normal(mu, std, vmin, vmax)
            assert v.dtype == dtype and mean.dtype == dtype
            assert torch.isfinite(v).all()
            assert (v >= vmin).all() and (v <= vmax).all()
            ref = truncnorm(x, y, loc=0.3, scale=0.7)
            np.testing.assert_allclose(mean[0].item(), ref.mean(), rtol=rtol_mean, atol=1e-7)
            # Monte Carlo accuracy (+ float32 resolution of the location)
            tol = 5 * ref.std() / np.sqrt(n) + (1e-12 if dtype == torch.float64 else 1e-6 * abs(ref.mean()))
            assert abs(v.double().mean().item() - ref.mean()) < tol, (dtype, x, y)
            np.testing.assert_allclose(v.double().std().item(), ref.std(), rtol=2e-2, atol=1e-7)


def test_sampling_float32_and_beta():
    params = make_params()
    params = params.to(dtype=torch.float32)
    h = torch.bernoulli(torch.full((50000, NUM_HIDDENS), 0.5))
    for beta in [1.0, 0.3, 1e-8, 0.0]:
        chains = params.sample_visibles({"hidden": h.clone()}, beta=beta)
        v = chains["visible"]
        assert v.dtype == torch.float32
        assert torch.isfinite(v).all()
        assert (v >= 0).all() and (v <= 1).all()
    # beta = 0: uniform in the box
    torch.testing.assert_close(v.mean(0), torch.full((NUM_VISIBLES,), 0.5), atol=1e-2, rtol=0)
    torch.testing.assert_close(v.var(0), torch.full((NUM_VISIBLES,), 1 / 12), atol=1e-2, rtol=0)


def test_conditional_moments():
    params = make_params()
    n = 200000
    h = torch.bernoulli(torch.full((1, NUM_HIDDENS), 0.5, dtype=DTYPE)).expand(n, -1)
    beta = 0.6
    chains = params.sample_visibles({"hidden": h}, beta=beta)
    mu = params.sigma.square() * (params.vbias + h[0] @ params.weight_matrix.T)
    std = params.sigma / beta**0.5
    a, b = (0 - mu) / std, (1 - mu) / std
    ref = [truncnorm(a[i].item(), b[i].item(), loc=mu[i].item(), scale=std[i].item()) for i in range(NUM_VISIBLES)]
    np.testing.assert_allclose(chains["visible_mag"][0].numpy(), [r.mean() for r in ref], rtol=1e-8)
    np.testing.assert_allclose(chains["visible"].mean(0).numpy(), [r.mean() for r in ref], atol=5e-3)
    np.testing.assert_allclose(chains["visible"].var(0).numpy(), [r.var() for r in ref], rtol=3e-2)


def test_energy_visibles_is_marginal_over_hiddens():
    params = make_params()
    v = torch.rand(10, NUM_VISIBLES, dtype=DTYPE)
    energy = torch.stack([params.compute_energy(v, hh.expand(10, -1)) for hh in all_hiddens()])
    torch.testing.assert_close(
        params.compute_energy_visibles({"visible": v}), -torch.logsumexp(-energy, 0)
    )


def test_energy_hiddens_is_marginal_over_visibles():
    # Numerical integration over the box [0, 1] for each visible
    params = make_params()
    h = all_hiddens()
    n_grid = 200001
    grid = torch.linspace(0, 1, n_grid, dtype=DTYPE).view(1, 1, -1)
    log_w = torch.full((n_grid,), np.log(1 / (n_grid - 1)), dtype=DTYPE)
    log_w[0] = log_w[-1] = np.log(0.5 / (n_grid - 1))  # trapezoidal rule
    fields = params.vbias + h @ params.weight_matrix.T
    log_integrand = (
        -0.5 * grid**2 / params.sigma.square().view(1, -1, 1)
        + fields.unsqueeze(-1) * grid
        + log_w
    )
    expected = -(h @ params.hbias) - torch.logsumexp(log_integrand, -1).sum(1)
    torch.testing.assert_close(params.compute_energy_hiddens(h), expected)


def test_ref_log_z():
    params = make_params().independent_model()
    assert isinstance(params, TGBRBM)
    log_z = compute_partition_function(params, all_hiddens())
    np.testing.assert_allclose(log_z, params.ref_log_z)


def test_unbounded_reduces_to_gbrbm():
    params = make_params(bounded=False)
    gb = GBRBM(params.weight_matrix, params.vbias, params.hbias, params.sigma)
    h = all_hiddens()
    torch.testing.assert_close(params.compute_energy_hiddens(h), gb.compute_energy_hiddens(h))
    np.testing.assert_allclose(params.ref_log_z, gb.ref_log_z)


def test_gradient_matches_autograd_of_free_energy():
    params = make_params()
    data = params.init_chains(30, start_v=torch.rand(30, NUM_VISIBLES, dtype=DTYPE))
    chains = params.init_chains(40)
    params.compute_gradient(data, chains, centered=False)
    leaves = [p.clone().requires_grad_() for p in params.parameters()]
    p = TGBRBM(*leaves, sigma=params.sigma, vmin=params.vmin, vmax=params.vmax)
    objective = -(
        p.compute_energy_visibles(data).mean() - p.compute_energy_visibles(chains).mean()
    )
    objective.backward()
    for expected, param in zip(leaves, params.parameters()):
        torch.testing.assert_close(param.grad, expected.grad)


def test_init_parameters_matches_data_mean():
    torch.manual_seed(0)
    n = 5000
    data = torch.rand(n, NUM_VISIBLES, dtype=DTYPE) ** torch.tensor([1.0, 3.0, 0.3, 8.0], dtype=DTYPE)
    data[:, 0] = 0.0  # constant pixel at the lower bound
    dataset = RBMDataset(
        data=data.numpy(),
        labels=np.zeros(n),
        weights=np.ones(n),
        names=np.arange(n),
        dataset_name="test",
        variable_type="continuous",
        device="cpu",
        dtype=DTYPE,
    )
    params = TGBRBM.init_parameters(NUM_HIDDENS, dataset, "cpu", DTYPE)
    assert isinstance(params, TGBRBM)
    chains = params.independent_model().init_chains(200000)
    assert (chains["visible"] >= 0).all() and (chains["visible"] <= 1).all()
    torch.testing.assert_close(chains["visible"].mean(0)[1:], data.mean(0)[1:], atol=5e-3, rtol=0)
    # The constant pixel is as close to 0 as allowed by the floor on the target mean
    assert chains["visible"].mean(0)[0] < 2e-3


def test_named_parameters_round_trip_and_type():
    params = make_params()
    loaded = TGBRBM.set_named_parameters(params.named_parameters(), device="cpu", dtype=DTYPE)
    assert isinstance(loaded, TGBRBM) and loaded == params
    assert isinstance(params * 0.5 + params, TGBRBM)
    assert isinstance(params.clone(), TGBRBM)
    vbias = params.vbias.clone()
    params.post_grad_update()
    torch.testing.assert_close(params.vbias, vbias)

