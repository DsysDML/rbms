import pytest

from rbms.bernoulli_bernoulli.implement import _compute_energy_visibles


def test_compute_energy_visibles(sample_binary_v_samples, sample_params_bbrbm):
    # Arrange
    v, _ = sample_binary_v_samples
    vbias, hbias, weight_matrix = sample_params_bbrbm

    # Act
    energy = _compute_energy_visibles(v, vbias, hbias, weight_matrix)

    # Assert
    assert len(energy.shape) == 1
    assert energy.shape[0] == pytest.NUM_SAMPLES
