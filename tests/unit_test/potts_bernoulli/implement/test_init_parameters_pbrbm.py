import numpy as np
import pytest
import torch

from rbms.potts_bernoulli.implement import _init_parameters


def test_init_parameters_cpu(sample_potts_v_samples):
    # We test here with default device / dtype to be able to compute statistics
    # on the parameters.
    data = sample_potts_v_samples
    device = torch.device("cpu")
    dtype = torch.float32
    var_init = 1e-4

    vbias, hbias, weight_matrix = _init_parameters(
        pytest.NUM_HIDDENS,
        data,
        device,
        dtype,
        var_init,
    )

    assert np.allclose(weight_matrix.std().item(), var_init, atol=var_init)

    assert vbias.shape == (pytest.NUM_VISIBLES, pytest.NUM_STATES)
    assert hbias.shape == (pytest.NUM_HIDDENS,)
    assert weight_matrix.shape == (
        pytest.NUM_VISIBLES,
        pytest.NUM_STATES,
        pytest.NUM_HIDDENS,
    )

    assert vbias.device == device
    assert hbias.device == device
    assert weight_matrix.device == device

    assert vbias.dtype == dtype
    assert hbias.dtype == dtype
    assert weight_matrix.dtype == dtype


def test_init_parameters_dummy_device(sample_potts_v_samples):
    # We test now with the dummy device + different dtype
    data = sample_potts_v_samples
    device = pytest.DEVICE_CUDA
    dtype = torch.float64
    var_init = 1e-4

    vbias, hbias, weight_matrix = _init_parameters(
        pytest.NUM_HIDDENS,
        data,
        device,
        dtype,
        var_init,
    )

    assert vbias.shape == (pytest.NUM_VISIBLES, pytest.NUM_STATES)
    assert hbias.shape == (pytest.NUM_HIDDENS,)
    assert weight_matrix.shape == (
        pytest.NUM_VISIBLES,
        pytest.NUM_STATES,
        pytest.NUM_HIDDENS,
    )

    assert vbias.device.type == device
    assert hbias.device.type == device
    assert weight_matrix.device.type == device

    assert vbias.dtype == dtype
    assert hbias.dtype == dtype
    assert weight_matrix.dtype == dtype
