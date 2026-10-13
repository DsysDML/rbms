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
    W, W2, b, c, log_precision = leaves
    p = GBRBM_VAR(W, W2, b, c, log_precision=log_precision, min_precision=params.min_precision)
    objective = -(
        p.compute_energy_visibles(data).mean() - p.compute_energy_visibles(chains).mean()
    )
    objective.backward()
    for expected, param in zip(leaves, params.parameters()):
        torch.testing.assert_close(param.grad, expected.grad)


def test_positive_precision():
    params = make_params()
    params.weight_matrix_2[0] += 5.0
    params.log_precision[1] = -20.0
    params.post_grad_update()
    # The hiddens can only increase the precision
    assert (params.weight_matrix_2 <= 0).all()
    prec = conditional_precision(params, all_hiddens())
    assert (prec >= params.min_precision - 1e-12).all()
    # The smallest precision is reached at h = 0, raised to the floor when needed
    torch.testing.assert_close(prec.min(0).values, params.precision)
    torch.testing.assert_close(params.precision[1], params.min_precision[1])


def test_max_precision():
    params = make_params()
    params.max_precision = torch.full_like(params.precision, 2.0)
    params.log_precision[2] = 10.0
    params.post_grad_update()
    assert (params.precision <= 2.0 + 1e-12).all()
    torch.testing.assert_close(params.precision[2], torch.tensor(2.0, dtype=DTYPE))


def test_named_parameters_round_trip():
    params = make_params()
    loaded = GBRBM_VAR.set_named_parameters(
        params.named_parameters(), device="cpu", dtype=DTYPE
    )
    assert loaded == params


def test_log_precision_parametrization():
    params = make_params()
    names = [next(k for k in params.named_parameters() if getattr(params, k) is q) for q in params.parameters()]
    assert names == ["weight_matrix", "weight_matrix_2", "vbias", "hbias", "log_precision"]
    torch.testing.assert_close(params.precision, params.log_precision.exp())
    # dL/dlog(lambda) = lambda * dL/dlambda
    data = params.init_chains(30, start_v=torch.randn(30, NUM_VISIBLES, dtype=DTYPE))
    chains = params.init_chains(40, start_v=torch.randn(40, NUM_VISIBLES, dtype=DTYPE))
    params.compute_gradient(data, chains, centered=False)
    lam = params.precision.clone().requires_grad_()
    W, W2, b, c = (x.clone() for x in params.parameters()[:4])
    p = GBRBM_VAR(W, W2, b, c, precision=lam)
    (-(p.compute_energy_visibles(data).mean() - p.compute_energy_visibles(chains).mean())).backward()
    torch.testing.assert_close(params.log_precision.grad, lam.grad * lam.detach())
    # AIS interpolation stays between the two models in log space
    mid = params.independent_model() * 0.5 + params * 0.5
    torch.testing.assert_close(mid.precision, params.precision)


def test_load_old_archive_with_precision():
    params = make_params()
    named = params.named_parameters()
    named["precision"] = np.exp(named.pop("log_precision"))
    loaded = GBRBM_VAR.set_named_parameters(named, device="cpu", dtype=DTYPE)
    torch.testing.assert_close(loaded.precision, params.precision)
    assert loaded == params

    # Archives saved before the upper bound on the precision: no bound
    named = params.named_parameters()
    named.pop("max_precision")
    loaded = GBRBM_VAR.set_named_parameters(named, device="cpu", dtype=DTYPE)
    assert torch.isinf(loaded.max_precision).all()



def test_centered_step_with_w2_bound():
    # The centered gradient keeps the centered hidden bias c' = c + mu_v.W + mu_s.W2
    # (mu: data means of v and s = v^2 / 2) moving as h_data - h_model. The entries of W2
    # at the bound (0) pushed up by the gradient are clipped after the step, so they must
    # not enter the centering corrections: after the step and the projection, c' must
    # still have moved by lr * (h_data - h_model).
    params = make_params()
    assert (params.weight_matrix_2 == 0).any()
    v = 3 * torch.randn(200, NUM_VISIBLES, dtype=DTYPE)
    data = params.sample_hiddens(params.init_chains(200, start_v=v))
    chains = params.init_chains(300)
    mu_v, mu_s = v.mean(0), 0.5 * v.square().mean(0)

    def centered_hbias(p):
        return p.hbias + mu_v @ p.weight_matrix + mu_s @ p.weight_matrix_2

    params.compute_gradient(data, chains, centered=False)
    expected_change = params.hbias.grad.clone()
    params.compute_gradient(data, chains, centered=True)
    frozen = params.weight_matrix_2 >= 0
    assert (params.weight_matrix_2.grad[frozen] <= 0).all()
    before = centered_hbias(params)
    lr = 1e-3
    with torch.no_grad():
        for p in params.parameters():
            p += lr * p.grad
    params.post_grad_update()
    torch.testing.assert_close(centered_hbias(params) - before, lr * expected_change)


def test_centered_step_of_log_precision():
    # The centered gradient trains the centered precision lambda' = lambda - W2 @ mu_h
    # (mu_h: data mean of h), whose plain log-scale step is lr * lambda * dL/dlambda,
    # i.e. lambda' moves by lr * lambda^2 * dL/dlambda. The centering correction of
    # lambda only compensates the step of W2 (absolute change of lambda): it must not be
    # multiplied by lambda as the rest of the log-scale gradient. With precisions far
    # from 1 the compensation was lambda^2 times too large and lambda blew up.
    params = make_params()
    params.log_precision += 2.0  # precisions far from 1
    params.weight_matrix_2.clamp_(max=-0.1)  # no entry at the bound
    v = 0.3 * torch.randn(200, NUM_VISIBLES, dtype=DTYPE)
    data = params.sample_hiddens(params.init_chains(200, start_v=v))
    chains = params.init_chains(300)
    mu_h = data["hidden_mag"].mean(0)

    def centered_precision(p):
        return p.precision - p.weight_matrix_2 @ mu_h

    params.compute_gradient(data, chains, centered=False)
    lam = params.precision
    expected_change = lam * params.log_precision.grad  # lambda^2 dL/dlambda
    params.compute_gradient(data, chains, centered=True)
    before = centered_precision(params)
    lr = 1e-6
    with torch.no_grad():
        for p in params.parameters():
            p += lr * p.grad
    params.post_grad_update()
    torch.testing.assert_close(
        (centered_precision(params) - before) / lr, expected_change, rtol=1e-3, atol=1e-6
    )
