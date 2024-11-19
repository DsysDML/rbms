import pytest
import torch

from rbms.potts_bernoulli.implement import _init_chains


def test_init_chains_random(sample_params_pbrbm):
    vbias, hbias, weight_matrix = sample_params_pbrbm

    v, h, mv, mh = _init_chains(pytest.NUM_SAMPLES, weight_matrix, hbias, start_v=None)

    assert v.shape == (pytest.NUM_SAMPLES, pytest.NUM_VISIBLES)
    assert mv.shape == (pytest.NUM_SAMPLES, pytest.NUM_VISIBLES, pytest.NUM_STATES)
    assert h.shape == (pytest.NUM_SAMPLES, pytest.NUM_HIDDENS)
    assert mh.shape == (pytest.NUM_SAMPLES, pytest.NUM_HIDDENS)

    assert v.dtype == weight_matrix.dtype
    assert mv.dtype == weight_matrix.dtype
    assert h.dtype == weight_matrix.dtype
    assert mh.dtype == weight_matrix.dtype

    assert v.device == weight_matrix.device
    assert mv.device == weight_matrix.device
    assert h.device == weight_matrix.device
    assert mh.device == weight_matrix.device


def test_init_chains_from_conf(sample_params_pbrbm, sample_potts_v_samples):
    vbias, hbias, weight_matrix = sample_params_pbrbm
    v_init = sample_potts_v_samples

    v, h, mv, mh = _init_chains(pytest.NUM_CHAINS, weight_matrix, hbias, start_v=v_init)

    assert v.shape == (pytest.NUM_SAMPLES, pytest.NUM_VISIBLES)
    assert mv.shape == (pytest.NUM_SAMPLES, pytest.NUM_VISIBLES, pytest.NUM_STATES)
    assert h.shape == (pytest.NUM_SAMPLES, pytest.NUM_HIDDENS)
    assert mh.shape == (pytest.NUM_SAMPLES, pytest.NUM_HIDDENS)

    assert v.dtype == weight_matrix.dtype
    assert mv.dtype == weight_matrix.dtype
    assert h.dtype == weight_matrix.dtype
    assert mh.dtype == weight_matrix.dtype

    assert v.device == weight_matrix.device
    assert mv.device == weight_matrix.device
    assert h.device == weight_matrix.device
    assert mh.device == weight_matrix.device

    assert torch.equal(v, v_init)
