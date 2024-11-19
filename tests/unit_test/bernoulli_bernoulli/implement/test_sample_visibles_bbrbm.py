import pytest
import torch

from rbms.bernoulli_bernoulli.implement import _sample_visibles


def test_sample_visibles(
    sample_binary_h_samples, sample_weight_matrix_bbrbm, sample_vbias_bbrbm
):
    # Arrange
    h, _ = sample_binary_h_samples
    weight_matrix = sample_weight_matrix_bbrbm
    vbias = sample_vbias_bbrbm
    beta = 1.0

    # Act
    v, mv = _sample_visibles(h, weight_matrix, vbias, beta)

    # Assert
    assert v.shape == (pytest.NUM_SAMPLES, pytest.NUM_VISIBLES)
    assert mv.shape == (pytest.NUM_SAMPLES, pytest.NUM_VISIBLES)
    assert torch.allclose(
        v.unique(), torch.tensor([0, 1], device=h.device, dtype=h.dtype)
    )
    assert torch.all(mv >= 0) and torch.all(mv <= 1)
    assert h.dtype == v.dtype
