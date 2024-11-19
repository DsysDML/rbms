import pytest

from rbms.bernoulli_bernoulli.implement import _compute_energy_hiddens


def test_compute_energy_hiddens(sample_binary_h_samples, sample_params_bbrbm):
    # Arrange
    h, _ = sample_binary_h_samples
    vbias, hbias, weight_matrix = sample_params_bbrbm

    # Act
    energy = _compute_energy_hiddens(h, vbias, hbias, weight_matrix)

    # Assert
    assert len(energy.shape) == 1
    assert energy.shape[0] == pytest.NUM_SAMPLES
