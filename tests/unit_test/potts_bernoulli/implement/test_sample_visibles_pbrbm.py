import pytest
import torch

from rbms.potts_bernoulli.implement import _sample_visibles


def test_sample_visibles(
    sample_binary_h_samples, sample_weight_matrix_pbrbm, sample_vbias_pbrbm
):
    # Arrange
    h, _ = sample_binary_h_samples
    weight_matrix = sample_weight_matrix_pbrbm
    vbias = sample_vbias_pbrbm
    beta = 1.0

    # Act
    v, mv = _sample_visibles(h, weight_matrix, vbias, beta)

    # Assert
    assert v.shape == (pytest.NUM_SAMPLES, pytest.NUM_VISIBLES)
    assert mv.shape == (pytest.NUM_SAMPLES, pytest.NUM_VISIBLES, pytest.NUM_STATES)
    assert torch.allclose(
        v.unique(), torch.arange(pytest.NUM_STATES, device=v.device, dtype=v.dtype)
    )
    assert torch.all(mv >= 0) and torch.all(mv <= 1)
    assert h.dtype == v.dtype
