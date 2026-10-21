import numpy as np
import torch

from rbms.gaussian_bernoulli_var.classes import GBRBM_VAR
from rbms.optim import PTTSpeedController, SGD_cossim_ptt, limit_step_scale, setup_optim

DTYPE = torch.float64


class FakeSampler:
    """Chains of the current model (only what the controller uses)."""

    def __init__(self, params, num_chains=500):
        self.chains = params.init_chains(num_chains)

    def get_curr_conf(self):
        return self.chains


def make_setup(lr, target, mult_optim=True, relative=0.0):
    torch.manual_seed(0)
    params = GBRBM_VAR(
        weight_matrix=0.1 * torch.randn(6, 4, dtype=DTYPE),
        weight_matrix_2=-0.1 * torch.rand(6, 4, dtype=DTYPE),
        vbias=torch.randn(6, dtype=DTYPE),
        hbias=torch.randn(4, dtype=DTYPE),
        precision=torch.ones(6, dtype=DTYPE),
    )
    sampler = FakeSampler(params)
    args = {
        "learning_rate": lr,
        "max_lr": 10,
        "scale_lr": False,
        "mult_optim": mult_optim,
        "optim": "cossim_ptt",
        "ptt_speed": target,
        "ptt_relative": relative,
    }
    optimizer = setup_optim("cossim_ptt", args, params, sampler)
    data = params.init_chains(500, start_v=2 * torch.randn(500, 6, dtype=DTYPE) + 1)
    return params, sampler, optimizer, data


def run(params, sampler, optimizer, data, num_updates):
    for _ in range(num_updates):
        for opt in optimizer:
            opt.zero_grad(set_to_none=False)
        data = params.sample_hiddens(data)
        params.compute_gradient(data, sampler.get_curr_conf(), centered=True)
        for opt in optimizer:
            opt.step()
        params.post_grad_update()


def lrs(optimizer):
    return np.array([opt.param_groups[0]["lr"] for opt in optimizer])


def test_setup_shares_one_controller():
    params, sampler, optimizer, _ = make_setup(lr=0.01, target=0.01)
    assert len(optimizer) == len(params.parameters())
    assert all(isinstance(opt, SGD_cossim_ptt) for opt in optimizer)
    controller = optimizer[0].controller
    assert isinstance(controller, PTTSpeedController)
    assert all(opt.controller is controller for opt in optimizer)
    assert controller.target == 0.01


def speed_of_update(params, sampler, optimizer, data):
    """Displacement std_x[F_new(x) - F(x)] of one update on the chains of the sampler."""
    before = params.clone()
    run(params, sampler, optimizer, data, 1)
    chains = sampler.get_curr_conf()
    delta = params.compute_energy_visibles(chains) - before.compute_energy_visibles(chains)
    return delta.std().item()


def test_large_lr_is_limited_at_each_update():
    # Too large lr: every update moves the model by at most the target, and the limited lr
    # becomes the lr of the training
    for mult_optim in [True, False]:
        params, sampler, optimizer, data = make_setup(lr=1.0, target=0.01, mult_optim=mult_optim)
        for _ in range(20):
            speed = speed_of_update(params, sampler, optimizer, data)
            assert speed <= 0.01 * (1 + 1e-9)
            assert np.isclose(optimizer[0].controller.last_speed, speed)
        assert (lrs(optimizer) < 1.0).all()
        # The limit acts at the first update and the lr follows it: the displacement stays
        # close to the target (bisection up to a factor 1.05 on the step)
        assert speed > 0.01 / 1.2


def test_small_lr_grows_until_the_limit():
    # Small lr: the limit does not act and the lr grows by lr_growth per update
    params, sampler, optimizer, data = make_setup(lr=1e-4, target=0.01)
    growth = optimizer[0].controller.lr_growth
    run(params, sampler, optimizer, data, 10)
    np.testing.assert_allclose(lrs(optimizer), 1e-4 * growth**10, rtol=1e-5)
    # Until the displacement reaches the target
    run(params, sampler, optimizer, data, 1000)
    assert lrs(optimizer)[0] < 1e-4 * growth**1010
    assert optimizer[0].controller.last_speed > 0.01 / 1.2


def test_limit_step_scale():
    params, sampler, optimizer, data = make_setup(lr=1.0, target=0.01, mult_optim=False)
    data = params.sample_hiddens(data)
    params.compute_gradient(data, sampler.get_curr_conf(), centered=True)
    chains = sampler.get_curr_conf()
    steps = [(i, p.grad, 1.0) for i, p in enumerate(params.parameters())]
    factor, std = limit_step_scale(params, chains, steps, 0.01)
    assert 0 < factor < 1 and std <= 0.01
    # A step 5% larger exceeds the limit
    steps = [(i, p.grad, 1.05 * factor) for i, p in enumerate(params.parameters())]
    assert limit_step_scale(params, chains, steps, 0.01)[0] < 1
    # A step that satisfies the limit is not changed
    steps = [(i, p.grad, 0.5 * factor) for i, p in enumerate(params.parameters())]
    assert limit_step_scale(params, chains, steps, 0.01)[0] == 1.0


def test_relative_learning_rates():
    # Large target: the limit does not act, the lr of all the parameters grow by lr_growth
    # and the distribution between the parameters follows exp(rate * (cos_i - mean cos))
    rate = 0.5
    params, sampler, optimizer, data = make_setup(lr=0.01, target=1e6, relative=rate)
    controller = optimizer[0].controller
    growth = controller.lr_growth
    run(params, sampler, optimizer, data, 1)  # first update: previous gradient stored
    np.testing.assert_allclose(lrs(optimizer), 0.01 * growth, rtol=1e-5)
    for _ in range(20):
        before = lrs(optimizer)
        run(params, sampler, optimizer, data, 1)
        cos = controller.last_cosines
        assert cos.shape == (len(optimizer),) and (np.abs(cos) <= 1 + 1e-12).all()
        np.testing.assert_allclose(
            lrs(optimizer), before * growth * np.exp(rate * (cos - cos.mean())), rtol=1e-5
        )
    np.testing.assert_allclose(np.log(lrs(optimizer)).mean(), np.log(0.01 * growth**21), rtol=1e-5)
    assert lrs(optimizer).std() > 0
