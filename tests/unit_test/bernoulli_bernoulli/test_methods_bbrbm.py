import pytest
import torch

from rbms.bernoulli_bernoulli.classes import BBRBM


# Test BBRBM class
def test_bb_rbm_initialization(sample_params_class_bbrbm):
    bb_rbm = sample_params_class_bbrbm

    assert bb_rbm.weight_matrix.shape == (pytest.NUM_VISIBLES, pytest.NUM_HIDDENS)
    assert bb_rbm.vbias.shape == (pytest.NUM_VISIBLES,)
    assert bb_rbm.hbias.shape == (pytest.NUM_HIDDENS,)


def test_bb_rbm_sample_hiddens(sample_params_class_bbrbm, sample_chains_bbrbm):
    bb_rbm = sample_params_class_bbrbm
    chains = sample_chains_bbrbm

    updated_chain = bb_rbm.sample_hiddens(chains)

    assert updated_chain["hidden"].shape == (pytest.NUM_CHAINS, pytest.NUM_HIDDENS)
    assert updated_chain["hidden_mag"].shape == (pytest.NUM_CHAINS, pytest.NUM_HIDDENS)


def test_bb_rbm_sample_visibles(sample_params_class_bbrbm, sample_chains_bbrbm):
    bb_rbm = sample_params_class_bbrbm
    chains = sample_chains_bbrbm

    updated_chain = bb_rbm.sample_visibles(chains)

    assert updated_chain["visible"].shape == (pytest.NUM_CHAINS, pytest.NUM_VISIBLES)
    assert updated_chain["visible_mag"].shape == (
        pytest.NUM_CHAINS,
        pytest.NUM_VISIBLES,
    )


def test_bb_rbm_compute_energy(sample_params_class_bbrbm):
    bb_rbm = sample_params_class_bbrbm
    v = torch.randn(pytest.NUM_SAMPLES, pytest.NUM_VISIBLES)
    h = torch.randn(pytest.NUM_SAMPLES, pytest.NUM_HIDDENS)

    energy = bb_rbm.compute_energy(v, h)

    assert energy.shape == (pytest.NUM_SAMPLES,)


def test_bb_rbm_compute_energy_visibles(sample_params_class_bbrbm):
    bb_rbm = sample_params_class_bbrbm
    v = torch.randn(pytest.NUM_SAMPLES, pytest.NUM_VISIBLES)

    energy = bb_rbm.compute_energy_visibles(v)

    assert energy.shape == (pytest.NUM_SAMPLES,)


def test_bb_rbm_compute_energy_hiddens(sample_params_class_bbrbm):
    bb_rbm = sample_params_class_bbrbm
    h = torch.randn(pytest.NUM_SAMPLES, pytest.NUM_HIDDENS)

    energy = bb_rbm.compute_energy_hiddens(h)

    assert energy.shape == (pytest.NUM_SAMPLES,)


def test_bb_rbm_compute_gradient(
    sample_params_class_bbrbm, sample_chains_bbrbm, sample_data_bbrbm
):
    bb_rbm = sample_params_class_bbrbm
    chains = sample_chains_bbrbm
    data = sample_data_bbrbm

    # We need to initialize the gradient
    for params in bb_rbm.parameters():
        params.grad = torch.zeros_like(params)

    # For realistic values
    chains = bb_rbm.sample_state(chains, 1)

    bb_rbm.compute_gradient(data, chains)

    # If the value has been updated, the norm should be != 0
    assert bb_rbm.vbias.grad.norm() != 0
    assert bb_rbm.hbias.grad.norm() != 0
    assert bb_rbm.weight_matrix.grad.norm() != 0


def test_bb_rbm_parameters(sample_params_class_bbrbm):
    bb_rbm = sample_params_class_bbrbm

    params = bb_rbm.parameters()

    assert len(params) == 3
    assert torch.equal(params[0], bb_rbm.weight_matrix)
    assert torch.equal(params[1], bb_rbm.vbias)
    assert torch.equal(params[2], bb_rbm.hbias)


def test_bb_rbm_init_chains(sample_params_class_bbrbm):
    bb_rbm = sample_params_class_bbrbm

    chain = bb_rbm.init_chains(pytest.NUM_CHAINS)

    assert chain["visible"].shape == (pytest.NUM_CHAINS, pytest.NUM_VISIBLES)
    assert chain["hidden"].shape == (pytest.NUM_CHAINS, pytest.NUM_HIDDENS)
    assert chain["visible_mag"].shape == (pytest.NUM_CHAINS, pytest.NUM_VISIBLES)
    assert chain["hidden_mag"].shape == (pytest.NUM_CHAINS, pytest.NUM_HIDDENS)


def test_bb_rbm_named_parameters(sample_params_class_bbrbm):
    bb_rbm = sample_params_class_bbrbm

    named_params = bb_rbm.named_parameters()

    assert len(named_params) == 3
    assert torch.equal(named_params["weight_matrix"], bb_rbm.weight_matrix)
    assert torch.equal(named_params["vbias"], bb_rbm.vbias)
    assert torch.equal(named_params["hbias"], bb_rbm.hbias)


def test_bb_rbm_set_named_parameters():
    new_weight_matrix = torch.randn(pytest.NUM_VISIBLES, pytest.NUM_HIDDENS)
    new_vbias = torch.randn(pytest.NUM_VISIBLES)
    new_hbias = torch.randn(pytest.NUM_HIDDENS)

    new_bb_rbm = BBRBM.set_named_parameters(
        {
            "weight_matrix": new_weight_matrix,
            "vbias": new_vbias,
            "hbias": new_hbias,
        }
    )

    assert torch.equal(new_bb_rbm.weight_matrix, new_weight_matrix)
    assert torch.equal(new_bb_rbm.vbias, new_vbias)
    assert torch.equal(new_bb_rbm.hbias, new_hbias)


def test_bb_rbm_to(sample_params_class_bbrbm):
    bb_rbm = sample_params_class_bbrbm

    bb_rbm.to(device=pytest.DEVICE_CUDA, dtype=torch.float64)

    assert bb_rbm.weight_matrix.device.type == pytest.DEVICE_CUDA
    assert bb_rbm.vbias.device.type == pytest.DEVICE_CUDA
    assert bb_rbm.hbias.device.type == pytest.DEVICE_CUDA
    assert bb_rbm.weight_matrix.dtype == torch.float64
    assert bb_rbm.vbias.dtype == torch.float64
    assert bb_rbm.hbias.dtype == torch.float64


def test_bb_rbm_clone(sample_params_class_bbrbm):
    bb_rbm = sample_params_class_bbrbm

    cloned_bb_rbm = bb_rbm.clone()

    assert torch.equal(cloned_bb_rbm.weight_matrix, bb_rbm.weight_matrix)
    assert torch.equal(cloned_bb_rbm.vbias, bb_rbm.vbias)
    assert torch.equal(cloned_bb_rbm.hbias, bb_rbm.hbias)


def test_bb_rbm_init_parameters(sample_dataset_bbrbm):
    dataset = sample_dataset_bbrbm

    bb_rbm = BBRBM.init_parameters(
        pytest.NUM_HIDDENS,
        dataset,
        torch.device("cpu"),
        torch.float32,
    )

    assert bb_rbm.weight_matrix.shape == (pytest.NUM_VISIBLES, pytest.NUM_HIDDENS)
    assert bb_rbm.vbias.shape == (pytest.NUM_VISIBLES,)
    assert bb_rbm.hbias.shape == (pytest.NUM_HIDDENS,)


def test_bb_rbm_num_visibles(sample_params_class_bbrbm):
    bb_rbm = sample_params_class_bbrbm

    assert bb_rbm.num_visibles() == pytest.NUM_VISIBLES


def test_bb_rbm_num_hiddens(sample_params_class_bbrbm):
    bb_rbm = sample_params_class_bbrbm

    assert bb_rbm.num_hiddens() == pytest.NUM_HIDDENS


def test_bb_rbm_ref_log_z(sample_params_class_bbrbm):
    bb_rbm = sample_params_class_bbrbm

    ref_log_z = bb_rbm.ref_log_z()

    assert isinstance(ref_log_z, float)


def test_bb_rbm_add(sample_params_class_bbrbm):
    bb_rbm1 = sample_params_class_bbrbm
    bb_rbm2 = sample_params_class_bbrbm

    added_bb_rbm = bb_rbm1 + bb_rbm2

    assert torch.equal(
        added_bb_rbm.weight_matrix, bb_rbm1.weight_matrix + bb_rbm2.weight_matrix
    )
    assert torch.equal(added_bb_rbm.vbias, bb_rbm1.vbias + bb_rbm2.vbias)
    assert torch.equal(added_bb_rbm.hbias, bb_rbm1.hbias + bb_rbm2.hbias)


def test_bb_rbm_mul(sample_params_class_bbrbm):
    bb_rbm = sample_params_class_bbrbm

    scalar = 2.0
    multiplied_bb_rbm = bb_rbm * scalar

    assert torch.equal(multiplied_bb_rbm.weight_matrix, bb_rbm.weight_matrix * scalar)
    assert torch.equal(multiplied_bb_rbm.vbias, bb_rbm.vbias * scalar)
    assert torch.equal(multiplied_bb_rbm.hbias, bb_rbm.hbias * scalar)


def test_bb_rbm_independent_model(sample_params_class_bbrbm):
    bb_rbm = sample_params_class_bbrbm

    independent_bb_rbm = bb_rbm.independent_model()

    assert torch.equal(
        independent_bb_rbm.weight_matrix, torch.zeros_like(bb_rbm.weight_matrix)
    )
    assert torch.equal(independent_bb_rbm.vbias, bb_rbm.vbias)
    assert torch.equal(independent_bb_rbm.hbias, torch.zeros_like(bb_rbm.hbias))
