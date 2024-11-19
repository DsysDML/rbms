import pytest

from rbms.metrics.aats import compute_aats


def test_compute_aats_euclid(sample_binary_v_chains, sample_binary_v_samples):
    v_data = sample_binary_v_samples[0]
    v_gen = sample_binary_v_chains[0]
    aa_truth, aa_syn = compute_aats(
        v_data,
        v_gen,
        n_sample=min(v_data.shape[0], v_gen.shape[0]),
        dist="euclid",
    )
    assert isinstance(aa_truth, float)
    assert isinstance(aa_syn, float)
    assert 0.0 <= aa_truth <= 1.0
    assert 0.0 <= aa_syn <= 1.0


def test_compute_aats_hamming(sample_binary_v_samples, sample_binary_v_chains):
    v_data = sample_binary_v_samples[0]
    v_gen = sample_binary_v_chains[0]
    aa_truth, aa_syn = compute_aats(
        v_data,
        v_gen,
        n_sample=min(v_data.shape[0], v_gen.shape[0]),
        dist="hamming",
    )
    assert isinstance(aa_truth, float)
    assert isinstance(aa_syn, float)
    assert 0.0 <= aa_truth <= 1.0
    assert 0.0 <= aa_syn <= 1.0


def test_compute_aats_invalid_dist(sample_binary_v_samples, sample_binary_v_chains):
    v_data = sample_binary_v_samples[0]
    v_gen = sample_binary_v_chains[0]
    with pytest.raises(ValueError):
        compute_aats(v_data, v_gen, n_sample=2, dist="invalid")
