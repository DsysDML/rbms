import pytest

from rbms.potts_bernoulli.implement import _compute_energy


def test_compute_energy(
    sample_potts_v_samples, sample_binary_h_samples, sample_params_pbrbm
):
    # Arrange
    v = sample_potts_v_samples
    h, _ = sample_binary_h_samples
    vbias, hbias, weight_matrix = sample_params_pbrbm

    # Act
    energy = _compute_energy(v, h, vbias, hbias, weight_matrix)

    # Assert
    assert len(energy.shape) == 1
    assert energy.shape[0] == pytest.NUM_SAMPLES
