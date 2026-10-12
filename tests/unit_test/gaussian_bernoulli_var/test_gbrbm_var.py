import numpy as np
import torch

from rbms.gaussian_bernoulli_var.classes import GBRBM_VAR
from rbms.partition_function.exact import compute_partition_function
from rbms.utils import get_categorical_configurations

NUM_VISIBLES = 4
NUM_HIDDENS = 3
DTYPE = torch.float64


def make_params(seed=0):
    torch.manual_seed(seed)
    params = GBRBM_VAR(
        weight_matrix=torch.randn(NUM_VISIBLES, NUM_HIDDENS, dtype=DTYPE),
        weight_matrix_2=0.3 * torch.randn(NUM_VISIBLES, NUM_HIDDENS, dtype=DTYPE),
        vbias=torch.randn(NUM_VISIBLES, dtype=DTYPE),
        hbias=torch.randn(NUM_HIDDENS, dtype=DTYPE),
        precision=torch.rand(NUM_VISIBLES, dtype=DTYPE) + 0.5,
    )
    # Make sure the conditional precisions are positive
    params.post_grad_update()
    return params


def all_hiddens():
    return get_categorical_configurations(n_states=2, n_dim=NUM_HIDDENS, dtype=DTYPE)


def conditional_precision(params, h):
    return params.precision - h @ params.weight_matrix_2.T


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
    grid = torch.linspace(-60, 60, 400001, dtype=DTYPE).view(1, 1, -1)
    dx = grid[0, 0, 1] - grid[0, 0, 0]
    prec_h = conditional_precision(params, h)  # (2^Nh, Nv)
    fields = params.vbias + h @ params.weight_matrix.T
    log_integrand = -0.5 * prec_h.unsqueeze(-1) * grid**2 + fields.unsqueeze(-1) * grid
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
    h = torch.bernoulli(torch.full((1, NUM_HIDDENS), 0.5, dtype=DTYPE)).expand(n, -1)
    beta = 0.7
    chains = params.sample_visibles({"hidden": h}, beta=beta)
    prec = conditional_precision(params, h[0])
    mean = (params.vbias + h[0] @ params.weight_matrix.T) / prec
    torch.testing.assert_close(chains["visible_mag"][0], mean)
    torch.testing.assert_close(chains["visible"].mean(0), mean, atol=3e-2, rtol=0)
    torch.testing.assert_close(
        chains["visible"].var(0), 1 / (beta * prec), atol=3e-2, rtol=3e-2
    )


def test_gradient_vanishes_when_data_equals_chains():
    params = make_params()
    v = torch.randn(50, NUM_VISIBLES, dtype=DTYPE)
    chains = params.init_chains(50, start_v=v)
    for centered in [True, False]:
        params.compute_gradient(chains, chains, centered=centered)
        for p in params.parameters():
            torch.testing.assert_close(p.grad, torch.zeros_like(p))


def test_gradient_matches_autograd_of_free_energy():
    # Non-centered gradient = -d/dtheta [<F(v)>_data - <F(v)>_chains]
    params = make_params()
    data = params.init_chains(30, start_v=torch.randn(30, NUM_VISIBLES, dtype=DTYPE))
    chains = params.init_chains(40, start_v=torch.randn(40, NUM_VISIBLES, dtype=DTYPE))
    params.compute_gradient(data, chains, centered=False)

    leaves = [p.clone().requires_grad_() for p in params.parameters()]
    p = GBRBM_VAR(*leaves)
    objective = -(
        p.compute_energy_visibles(data).mean() - p.compute_energy_visibles(chains).mean()
    )
    objective.backward()
    for expected, param in zip(leaves, params.parameters()):
        torch.testing.assert_close(param.grad, expected.grad)


def test_positive_precision():
    params = make_params()
    params.weight_matrix_2 += 5.0
    params.post_grad_update()
    prec = conditional_precision(params, all_hiddens())
    assert (prec >= params.min_precision - 1e-12).all()
    # Equality is reached for the worst hidden configuration
    torch.testing.assert_close(prec.min(0).values, params.min_precision)


def test_named_parameters_round_trip():
    params = make_params()
    loaded = GBRBM_VAR.set_named_parameters(
        params.named_parameters(), device="cpu", dtype=DTYPE
    )
    assert loaded == params
