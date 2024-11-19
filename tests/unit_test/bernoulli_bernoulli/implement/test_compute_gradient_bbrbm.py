import pytest
import torch

from rbms.bernoulli_bernoulli.implement import _compute_gradient


def test_compute_gradient_not_centered(
    sample_params_bbrbm,
    sample_binary_v_chains,
    sample_binary_h_chains,
    sample_binary_v_samples,
    sample_binary_h_samples,
):
    v_chains, mv_chains = sample_binary_v_chains
    v_data, mv_data = sample_binary_v_samples
    h_chains, mh_chains = sample_binary_h_chains
    h_data, mh_data = sample_binary_h_samples
    vbias, hbias, weight_matrix = sample_params_bbrbm
    w_data = torch.ones(pytest.NUM_SAMPLES, 1)
    w_chains = torch.ones(pytest.NUM_CHAINS, 1)

    # Initialize grad to zero
    vbias.grad = torch.zeros_like(vbias)
    hbias.grad = torch.zeros_like(hbias)
    weight_matrix.grad = torch.zeros_like(weight_matrix)

    _compute_gradient(
        v_data,
        mh_data,
        w_data,
        v_chains,
        h_chains,
        w_chains,
        vbias,
        hbias,
        weight_matrix,
        centered=False,
    )

    assert vbias.grad.norm() > 0
    assert hbias.grad.norm() > 0
    assert weight_matrix.grad.norm() > 0
    assert vbias.grad.shape == vbias.shape
    assert hbias.grad.shape == hbias.shape
    assert weight_matrix.grad.shape == weight_matrix.shape
    assert vbias.grad.device == vbias.device
    assert hbias.grad.device == hbias.device
    assert weight_matrix.grad.device == weight_matrix.device
    assert vbias.grad.dtype == vbias.dtype
    assert hbias.grad.dtype == hbias.dtype
    assert weight_matrix.grad.dtype == weight_matrix.dtype


def test_compute_gradient_centered(
    sample_params_bbrbm,
    sample_binary_v_chains,
    sample_binary_h_chains,
    sample_binary_v_samples,
    sample_binary_h_samples,
):
    v_chains, mv_chains = sample_binary_v_chains
    v_data, mv_data = sample_binary_v_samples
    h_chains, mh_chains = sample_binary_h_chains
    h_data, mh_data = sample_binary_h_samples
    vbias, hbias, weight_matrix = sample_params_bbrbm
    w_data = torch.ones(pytest.NUM_SAMPLES, 1)
    w_chains = torch.ones(pytest.NUM_CHAINS, 1)

    # Initialize grad to zero
    vbias.grad = torch.zeros_like(vbias)
    hbias.grad = torch.zeros_like(hbias)
    weight_matrix.grad = torch.zeros_like(weight_matrix)

    _compute_gradient(
        v_data,
        mh_data,
        w_data,
        v_chains,
        h_chains,
        w_chains,
        vbias,
        hbias,
        weight_matrix,
        centered=True,
    )

    assert vbias.grad.norm() > 0
    assert hbias.grad.norm() > 0
    assert weight_matrix.grad.norm() > 0
    assert vbias.grad.shape == vbias.shape
    assert hbias.grad.shape == hbias.shape
    assert weight_matrix.grad.shape == weight_matrix.shape
    assert vbias.grad.device == vbias.device
    assert hbias.grad.device == hbias.device
    assert weight_matrix.grad.device == weight_matrix.device
    assert vbias.grad.dtype == vbias.dtype
    assert hbias.grad.dtype == hbias.dtype
    assert weight_matrix.grad.dtype == weight_matrix.dtype
