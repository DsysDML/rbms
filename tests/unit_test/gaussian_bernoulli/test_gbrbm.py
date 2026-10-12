import numpy as np
import torch

from rbms.gaussian_bernoulli.classes import GBRBM
from rbms.partition_function.exact import compute_partition_function
from rbms.utils import get_categorical_configurations

NUM_VISIBLES = 4
NUM_HIDDENS = 3
DTYPE = torch.float64


def make_params(seed=0):
    torch.manual_seed(seed)
    return GBRBM(
        weight_matrix=torch.randn(NUM_VISIBLES, NUM_HIDDENS, dtype=DTYPE),
        vbias=torch.randn(NUM_VISIBLES, dtype=DTYPE),
        hbias=torch.randn(NUM_HIDDENS, dtype=DTYPE),
        sigma=torch.rand(NUM_VISIBLES, dtype=DTYPE) + 0.5,
    )


def all_hiddens():
    return get_categorical_configurations(n_states=2, n_dim=NUM_HIDDENS, dtype=DTYPE)


def test_energy_visibles_is_marginal_over_hiddens():
    params = make_params()
    v = torch.randn(10, NUM_VISIBLES, dtype=DTYPE)
    h = all_hiddens()
    energy = torch.stack([params.compute_energy(v, hh.expand(10, -1)) for hh in h])
    expected = -torch.logsumexp(-energy, 0)
    torch.testing.assert_close(params.compute_energy_visibles({"visible": v}), expected)


def test_energy_hiddens_is_marginal_over_visibles():
    # 1D numerical integration on each visible (they are independent given h)
    params = make_params()
    h = all_hiddens()
    grid = torch.linspace(-30, 30, 200001, dtype=DTYPE)
    dx = grid[1] - grid[0]
    var = params.sigma.square()
    fields = params.vbias + h @ params.weight_matrix.T  # (2^Nh, Nv)
    # -E(v_i | h) for each visible on the grid
    log_integrand = (
        -0.5 * grid.view(1, 1, -1) ** 2 / var.view(1, -1, 1)
        + fields.unsqueeze(-1) * grid.view(1, 1, -1)
    )
    log_int = torch.logsumexp(log_integrand, -1) + torch.log(dx)
    expected = -(h @ params.hbias) - log_int.sum(1)
    torch.testing.assert_close(params.compute_energy_hiddens(h), expected)


def test_ref_log_z():
    params = make_params().independent_model()
    log_z = compute_partition_function(params, all_hiddens())
    np.testing.assert_allclose(log_z, params.ref_log_z)


def test_conditional_visible_moments():
    params = make_params()
    n = 200000
    h = torch.bernoulli(torch.full((n, NUM_HIDDENS), 0.5, dtype=DTYPE))
    h[:] = h[0]
    beta = 0.7
    chains = params.sample_visibles({"hidden": h}, beta=beta)
    mean = params.sigma.square() * (params.vbias + h[0] @ params.weight_matrix.T)
    torch.testing.assert_close(chains["visible_mag"][0], mean)
    torch.testing.assert_close(chains["visible"].mean(0), mean, atol=2e-2, rtol=0)
    torch.testing.assert_close(
        chains["visible"].var(0), params.sigma.square() / beta, atol=2e-2, rtol=2e-2
    )


def test_gradient_vanishes_when_data_equals_chains():
    params = make_params()
    v = torch.randn(50, NUM_VISIBLES, dtype=DTYPE)
    chains = params.init_chains(50, start_v=v)
    for centered in [True, False]:
        params.compute_gradient(chains, chains, centered=centered)
        for p in params.parameters():
            torch.testing.assert_close(p.grad, torch.zeros_like(p))


def test_gradient_matches_autograd_of_log_likelihood():
    params = make_params()
    v = torch.randn(30, NUM_VISIBLES, dtype=DTYPE)
    # Exact model averages by enumerating the hiddens
    h = all_hiddens()
    log_p_h = -params.compute_energy_hiddens(h)
    p_h = torch.softmax(log_p_h, 0)
    mv_h = params.sigma.square() * (params.vbias + h @ params.weight_matrix.T)
    model_vh = (p_h.view(-1, 1, 1) * mv_h.unsqueeze(-1) * h.unsqueeze(1)).sum(0)
    data = params.init_chains(30, start_v=v)
    # Build fake chains that reproduce the exact model averages
    chains = {
        "visible": mv_h,
        "hidden_mag": h,
        "weights": p_h,
    }
    params.compute_gradient(data, chains, centered=False)
    expected_w = (v.T @ data["hidden_mag"]) / 30 - model_vh

    # Autograd of the exact log likelihood
    W = params.weight_matrix.clone().requires_grad_()
    b = params.vbias.clone().requires_grad_()
    c = params.hbias.clone().requires_grad_()
    p = GBRBM(W, b, c, params.sigma)
    ll = -p.compute_energy_visibles({"visible": v}).mean() - torch.logsumexp(
        -p.compute_energy_hiddens(h), 0
    )
    ll.backward()
    torch.testing.assert_close(W.grad, expected_w)
    torch.testing.assert_close(params.weight_matrix.grad, W.grad)
    torch.testing.assert_close(params.vbias.grad, b.grad)
    torch.testing.assert_close(params.hbias.grad, c.grad)


def test_post_grad_update_does_not_change_parameters():
    params = make_params()
    before = [p.clone() for p in params.parameters()]
    params.post_grad_update()
    for b, p in zip(before, params.parameters()):
        torch.testing.assert_close(p, b)


def test_named_parameters_round_trip():
    params = make_params()
    named = params.named_parameters()
    assert set(named) == {"weight_matrix", "vbias", "hbias", "sigma"}
    loaded = GBRBM.set_named_parameters(named, device="cpu", dtype=DTYPE)
    assert loaded == params

