import pytest
import torch

from rbms.potts_bernoulli.implement import _sample_hiddens


def test_sample_hiddens(
    sample_potts_v_samples, sample_weight_matrix_pbrbm, sample_hbias_pbrbm
):
    # Arrange
    v = sample_potts_v_samples
    weight_matrix = sample_weight_matrix_pbrbm
    hbias = sample_hbias_pbrbm
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
