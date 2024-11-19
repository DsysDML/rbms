import pytest
import torch

from rbms.bernoulli_bernoulli.implement import _sample_hiddens


def test_sample_hiddens(
    sample_binary_v_samples, sample_weight_matrix_bbrbm, sample_hbias_bbrbm
):
    # Arrange
    v, _ = sample_binary_v_samples
    weight_matrix = sample_weight_matrix_bbrbm
    hbias = sample_hbias_bbrbm
    beta = 1.0

    # Act
    h, mh = _sample_hiddens(v, weight_matrix, hbias, beta)

    # Assert
    assert h.shape == (pytest.NUM_SAMPLES, pytest.NUM_HIDDENS)
    assert mh.shape == (pytest.NUM_SAMPLES, pytest.NUM_HIDDENS)
    assert torch.allclose(
        h.unique(), torch.tensor([0, 1], device=h.device, dtype=h.dtype)
    )
    assert torch.all(mh >= 0) and torch.all(mh <= 1)
    assert h.dtype == v.dtype
