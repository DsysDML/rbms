import pytest
import torch

from rbms.potts_bernoulli.functional import (
    compute_energy,
    compute_energy_hiddens,
    compute_energy_visibles,
    compute_gradient,
    init_chains,
    init_parameters,
    sample_hiddens,
    sample_visibles,
)


# Test sample_hiddens function
def test_sample_hiddens(sample_params_class_pbrbm, sample_chains_pbrbm):
    params = sample_params_class_pbrbm
    chains = sample_chains_pbrbm

    updated_chains = sample_hiddens(chains, params)

    assert updated_chains["hidden"].shape == (pytest.NUM_CHAINS, pytest.NUM_HIDDENS)
    assert updated_chains["hidden_mag"].shape == (pytest.NUM_CHAINS, pytest.NUM_HIDDENS)


# Test sample_visibles function
def test_sample_visibles(sample_params_class_pbrbm, sample_chains_pbrbm):
    params = sample_params_class_pbrbm
    chains = sample_chains_pbrbm

    updated_chains = sample_visibles(chains, params)

    assert updated_chains["visible"].shape == (pytest.NUM_CHAINS, pytest.NUM_VISIBLES)
    assert updated_chains["visible_mag"].shape == (
        pytest.NUM_CHAINS,
        pytest.NUM_VISIBLES,
        pytest.NUM_STATES,
    )


# Test compute_energy function
def test_compute_energy(sample_params_class_pbrbm):
    params = sample_params_class_pbrbm
    v = torch.randn(pytest.NUM_SAMPLES, pytest.NUM_VISIBLES)
    h = torch.randn(pytest.NUM_SAMPLES, pytest.NUM_HIDDENS)

    energy = compute_energy(v, h, params)

    assert energy.shape == (pytest.NUM_SAMPLES,)


# Test compute_energy_visibles function
def test_compute_energy_visibles(sample_params_class_pbrbm):
    params = sample_params_class_pbrbm
    v = torch.randn(pytest.NUM_SAMPLES, pytest.NUM_VISIBLES)

    energy = compute_energy_visibles(v, params)

    assert energy.shape == (pytest.NUM_SAMPLES,)


# Test compute_energy_hiddens function
def test_compute_energy_hiddens(sample_params_class_pbrbm):
    params = sample_params_class_pbrbm
    h = torch.randn(pytest.NUM_SAMPLES, pytest.NUM_HIDDENS)

    energy = compute_energy_hiddens(h, params)

    assert energy.shape == (pytest.NUM_SAMPLES,)


# Test compute_gradient function
def test_compute_gradient(
    sample_params_class_pbrbm, sample_chains_pbrbm, sample_data_pbrbm
):
    params = sample_params_class_pbrbm
    chains = sample_chains_pbrbm
    data = sample_data_pbrbm
    for param in params.parameters():
        param.grad = torch.zeros_like(param)
    compute_gradient(data, chains, params)

    assert params.vbias.grad is not None
    assert params.hbias.grad is not None
    assert params.weight_matrix.grad is not None


# Test init_chains function
def test_init_chains(sample_params_class_pbrbm):
    params = sample_params_class_pbrbm

    chain = init_chains(pytest.NUM_SAMPLES, params)

    assert chain["visible"].shape == (pytest.NUM_SAMPLES, pytest.NUM_VISIBLES)
    assert chain["hidden"].shape == (pytest.NUM_SAMPLES, pytest.NUM_HIDDENS)
    assert chain["visible_mag"].shape == (
        pytest.NUM_SAMPLES,
        pytest.NUM_VISIBLES,
        pytest.NUM_STATES,
    )
    assert chain["hidden_mag"].shape == (pytest.NUM_SAMPLES, pytest.NUM_HIDDENS)


# Test init_parameters function
def test_init_parameters(sample_dataset_pbrbm):
    dataset = sample_dataset_pbrbm

    bb_rbm = init_parameters(
        pytest.NUM_HIDDENS,
        dataset,
        torch.device("cpu"),
        torch.float32,
    )

    assert bb_rbm.weight_matrix.shape == (
        pytest.NUM_VISIBLES,
        pytest.NUM_STATES,
        pytest.NUM_HIDDENS,
    )
    assert bb_rbm.vbias.shape == (pytest.NUM_VISIBLES, pytest.NUM_STATES)
    assert bb_rbm.hbias.shape == (pytest.NUM_HIDDENS,)
