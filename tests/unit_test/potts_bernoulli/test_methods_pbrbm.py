import pytest
import torch

from rbms.potts_bernoulli.classes import PBRBM


# Test BBRBM class
def test_pb_rbm_initialization(sample_params_class_pbrbm):
    pb_rbm = sample_params_class_pbrbm

    assert pb_rbm.weight_matrix.shape == (
        pytest.NUM_VISIBLES,
        pytest.NUM_STATES,
        pytest.NUM_HIDDENS,
    )
    assert pb_rbm.vbias.shape == (pytest.NUM_VISIBLES, pytest.NUM_STATES)
    assert pb_rbm.hbias.shape == (pytest.NUM_HIDDENS,)


def test_pb_rbm_sample_hiddens(sample_params_class_pbrbm, sample_chains_pbrbm):
    pb_rbm = sample_params_class_pbrbm
    chains = sample_chains_pbrbm

    updated_chain = pb_rbm.sample_hiddens(chains)

    assert updated_chain["hidden"].shape == (pytest.NUM_CHAINS, pytest.NUM_HIDDENS)
    assert updated_chain["hidden_mag"].shape == (pytest.NUM_CHAINS, pytest.NUM_HIDDENS)


def test_pb_rbm_sample_visibles(sample_params_class_pbrbm, sample_chains_pbrbm):
    pb_rbm = sample_params_class_pbrbm
    chains = sample_chains_pbrbm

    updated_chain = pb_rbm.sample_visibles(chains)

    assert updated_chain["visible"].shape == (pytest.NUM_CHAINS, pytest.NUM_VISIBLES)
    assert updated_chain["visible_mag"].shape == (
        pytest.NUM_CHAINS,
        pytest.NUM_VISIBLES,
        pytest.NUM_STATES,
    )


def test_pb_rbm_compute_energy(
    sample_params_class_pbrbm, sample_potts_v_samples, sample_binary_h_samples
):
    pb_rbm = sample_params_class_pbrbm
    v = sample_potts_v_samples
    h, _ = sample_binary_h_samples

    energy = pb_rbm.compute_energy(v, h)

    assert energy.shape == (pytest.NUM_SAMPLES,)


def test_pb_rbm_compute_energy_visibles(
    sample_params_class_pbrbm, sample_potts_v_samples
):
    pb_rbm = sample_params_class_pbrbm
    v = sample_potts_v_samples

    energy = pb_rbm.compute_energy_visibles(v)

    assert energy.shape == (pytest.NUM_SAMPLES,)


def test_pb_rbm_compute_energy_hiddens(
    sample_params_class_pbrbm, sample_binary_h_samples
):
    pb_rbm = sample_params_class_pbrbm
    h, _ = sample_binary_h_samples

    energy = pb_rbm.compute_energy_hiddens(h)

    assert energy.shape == (pytest.NUM_SAMPLES,)


def test_pb_rbm_compute_gradient(
    sample_params_class_pbrbm, sample_chains_pbrbm, sample_data_pbrbm
):
    pb_rbm = sample_params_class_pbrbm
    chains = sample_chains_pbrbm
    data = sample_data_pbrbm

    # We need to initialize the gradient
    for params in pb_rbm.parameters():
        params.grad = torch.zeros_like(params)

    # For realistic values
    chains = pb_rbm.sample_state(chains, 1)

    pb_rbm.compute_gradient(data, chains)

    # If the value has been updated, the norm should be != 0
    assert pb_rbm.vbias.grad.norm() != 0
    assert pb_rbm.hbias.grad.norm() != 0
    assert pb_rbm.weight_matrix.grad.norm() != 0


def test_pb_rbm_parameters(sample_params_class_pbrbm):
    pb_rbm = sample_params_class_pbrbm

    params = pb_rbm.parameters()

    assert len(params) == 3
    assert torch.equal(params[0], pb_rbm.weight_matrix)
    assert torch.equal(params[1], pb_rbm.vbias)
    assert torch.equal(params[2], pb_rbm.hbias)


def test_pb_rbm_init_chains(sample_params_class_pbrbm):
    pb_rbm = sample_params_class_pbrbm

    chain = pb_rbm.init_chains(pytest.NUM_CHAINS)

    assert chain["visible"].shape == (pytest.NUM_CHAINS, pytest.NUM_VISIBLES)
    assert chain["hidden"].shape == (pytest.NUM_CHAINS, pytest.NUM_HIDDENS)
    assert chain["visible_mag"].shape == (
        pytest.NUM_CHAINS,
        pytest.NUM_VISIBLES,
        pytest.NUM_STATES,
    )
    assert chain["hidden_mag"].shape == (pytest.NUM_CHAINS, pytest.NUM_HIDDENS)


def test_pb_rbm_named_parameters(sample_params_class_pbrbm):
    pb_rbm = sample_params_class_pbrbm

    named_params = pb_rbm.named_parameters()

    assert len(named_params) == 3
    assert torch.equal(named_params["weight_matrix"], pb_rbm.weight_matrix)
    assert torch.equal(named_params["vbias"], pb_rbm.vbias)
    assert torch.equal(named_params["hbias"], pb_rbm.hbias)


def test_pb_rbm_set_named_parameters(
    sample_weight_matrix_pbrbm, sample_vbias_pbrbm, sample_hbias_pbrbm
):
    new_weight_matrix = sample_weight_matrix_pbrbm
    new_vbias = sample_vbias_pbrbm
    new_hbias = sample_hbias_pbrbm

    new_pb_rbm = PBRBM.set_named_parameters(
        {
            "weight_matrix": new_weight_matrix,
            "vbias": new_vbias,
            "hbias": new_hbias,
        }
    )

    assert torch.equal(new_pb_rbm.weight_matrix, new_weight_matrix)
    assert torch.equal(new_pb_rbm.vbias, new_vbias)
    assert torch.equal(new_pb_rbm.hbias, new_hbias)


def test_pb_rbm_to(sample_params_class_pbrbm):
    pb_rbm = sample_params_class_pbrbm

    pb_rbm.to(device=pytest.DEVICE_CUDA, dtype=torch.float64)

    assert pb_rbm.weight_matrix.device.type == pytest.DEVICE_CUDA
    assert pb_rbm.vbias.device.type == pytest.DEVICE_CUDA
    assert pb_rbm.hbias.device.type == pytest.DEVICE_CUDA
    assert pb_rbm.weight_matrix.dtype == torch.float64
    assert pb_rbm.vbias.dtype == torch.float64
    assert pb_rbm.hbias.dtype == torch.float64


def test_pb_rbm_clone(sample_params_class_pbrbm):
    pb_rbm = sample_params_class_pbrbm

    cloned_pb_rbm = pb_rbm.clone()

    assert torch.equal(cloned_pb_rbm.weight_matrix, pb_rbm.weight_matrix)
    assert torch.equal(cloned_pb_rbm.vbias, pb_rbm.vbias)
    assert torch.equal(cloned_pb_rbm.hbias, pb_rbm.hbias)


def test_pb_rbm_init_parameters(sample_dataset_pbrbm):
    dataset = sample_dataset_pbrbm

    pb_rbm = PBRBM.init_parameters(
        pytest.NUM_HIDDENS,
        dataset,
        torch.device("cpu"),
        torch.float32,
    )

    assert pb_rbm.weight_matrix.shape == (
        pytest.NUM_VISIBLES,
        pytest.NUM_STATES,
        pytest.NUM_HIDDENS,
    )
    assert pb_rbm.vbias.shape == (pytest.NUM_VISIBLES, pytest.NUM_STATES)
    assert pb_rbm.hbias.shape == (pytest.NUM_HIDDENS,)


def test_pb_rbm_num_visibles(sample_params_class_pbrbm):
    pb_rbm = sample_params_class_pbrbm

    assert pb_rbm.num_visibles() == pytest.NUM_VISIBLES


def test_pb_rbm_num_hiddens(sample_params_class_pbrbm):
    pb_rbm = sample_params_class_pbrbm

    assert pb_rbm.num_hiddens() == pytest.NUM_HIDDENS


def test_pb_rbm_ref_log_z(sample_params_class_pbrbm):
    pb_rbm = sample_params_class_pbrbm

    ref_log_z = pb_rbm.ref_log_z()

    assert isinstance(ref_log_z, float)


def test_pb_rbm_add(sample_params_class_pbrbm):
    pb_rbm1 = sample_params_class_pbrbm
    pb_rbm2 = sample_params_class_pbrbm

    added_pb_rbm = pb_rbm1 + pb_rbm2

    assert torch.equal(
        added_pb_rbm.weight_matrix, pb_rbm1.weight_matrix + pb_rbm2.weight_matrix
    )
    assert torch.equal(added_pb_rbm.vbias, pb_rbm1.vbias + pb_rbm2.vbias)
    assert torch.equal(added_pb_rbm.hbias, pb_rbm1.hbias + pb_rbm2.hbias)


def test_pb_rbm_mul(sample_params_class_pbrbm):
    pb_rbm = sample_params_class_pbrbm

    scalar = 2.0
    multiplied_pb_rbm = pb_rbm * scalar

    assert torch.equal(multiplied_pb_rbm.weight_matrix, pb_rbm.weight_matrix * scalar)
    assert torch.equal(multiplied_pb_rbm.vbias, pb_rbm.vbias * scalar)
    assert torch.equal(multiplied_pb_rbm.hbias, pb_rbm.hbias * scalar)


def test_pb_rbm_independent_model(sample_params_class_pbrbm):
    pb_rbm = sample_params_class_pbrbm

    independent_pb_rbm = pb_rbm.independent_model()

    assert torch.equal(
        independent_pb_rbm.weight_matrix, torch.zeros_like(pb_rbm.weight_matrix)
    )
    assert torch.equal(independent_pb_rbm.vbias, torch.zeros_like(pb_rbm.vbias))
    assert torch.equal(independent_pb_rbm.hbias, torch.zeros_like(pb_rbm.hbias))
