import numpy as np
import torch

from rbms.gaussian_bernoulli_var.classes import GBRBM_VAR
from rbms.optim import PTTSpeedController, SGD_cossim_ptt, setup_optim

DTYPE = torch.float64


class FakeSampler:
    """Chains of the current model (only what the controller uses)."""

    def __init__(self, params, num_chains=500):
        self.chains = params.init_chains(num_chains)

    def get_curr_conf(self):
        return self.chains


def make_setup(lr, target, window=5, mult_optim=True, relative=0.0):
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
        "ptt_window": window,
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
    assert controller.window == 5 and controller.target == 0.01


def test_learning_rate_follows_measured_displacement():
    params, sampler, optimizer, data = make_setup(lr=0.01, target=0.01, window=5)
    controller = optimizer[0].controller
    # First call: snapshot of the initial model. The lr stays fixed during the window
    before = params.clone()
    run(params, sampler, optimizer, data, 5)
    np.testing.assert_allclose(lrs(optimizer), 0.01)
    # Next call, before its step: adjustment with the displacement of the 5 updates
    after = params.clone()
    run(params, sampler, optimizer, data, 1)
    # The adjustment uses the displacement since the snapshot, on the sampler chains
    delta = after.compute_energy_visibles(sampler.chains) - before.compute_energy_visibles(
        sampler.chains
    )
    speed = delta.std().item() / controller.window
    assert np.isclose(controller.last_speed, speed)
    factor = np.clip((0.01 / speed) ** 0.5, 0.5, 2.0)
    np.testing.assert_allclose(lrs(optimizer), 0.01 * factor)


def test_learning_rate_converges_to_target_speed():
    # Too large lr: it goes down; too small: it goes up. Both towards the same speed
    final = []
    for lr in [1.0, 1e-5]:
        params, sampler, optimizer, data = make_setup(lr=lr, target=0.01, window=5)
        run(params, sampler, optimizer, data, 200)
        final.append(lrs(optimizer)[0])
        assert 0.3 < optimizer[0].controller.last_speed / 0.01 < 3
    assert final[0] < 1.0 and final[1] > 1e-5


def test_relative_learning_rates():
    # Window longer than the run: only the distribution between the parameters acts.
    # The factors are centered: the geometric mean of the lr does not change, and the
    # lr of each parameter follows exp(rate * (cos_i - mean cos)) at each update
    rate = 0.5
    params, sampler, optimizer, data = make_setup(lr=0.01, target=0.01, window=10_000, relative=rate)
    controller = optimizer[0].controller
    run(params, sampler, optimizer, data, 1)  # first update: previous gradient stored
    np.testing.assert_allclose(lrs(optimizer), 0.01)
    for _ in range(20):
        before = lrs(optimizer)
        run(params, sampler, optimizer, data, 1)
        cos = controller.last_cosines
        assert cos.shape == (len(optimizer),) and (np.abs(cos) <= 1 + 1e-12).all()
        np.testing.assert_allclose(lrs(optimizer), before * np.exp(rate * (cos - cos.mean())))
    np.testing.assert_allclose(np.log(lrs(optimizer)).mean(), np.log(0.01))
    assert lrs(optimizer).std() > 0
    # With a single optimizer there is nothing to distribute
    params, sampler, optimizer, data = make_setup(
        lr=0.01, target=0.01, window=10_000, relative=rate, mult_optim=False
    )
    run(params, sampler, optimizer, data, 5)
    np.testing.assert_allclose(lrs(optimizer), 0.01)
