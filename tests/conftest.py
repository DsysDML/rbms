import numpy as np
import pytest
import torch

from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.dataset.dataset_class import RBMDataset
from rbms.potts_bernoulli.classes import PBRBM

pytest.NUM_HIDDENS = 3
pytest.NUM_STATES = 5
pytest.NUM_VISIBLES = 7
pytest.NUM_SAMPLES = 11
pytest.NUM_CHAINS = 13
pytest.BATCH_SIZE = 17
pytest.GIBBS_STEPS = 19
pytest.SEED = 42
pytest.TRAIN_SIZE = 0.6

pytest.DEVICE_CUDA = "meta"
pytest.LEARNING_RATE = 0.03


@pytest.fixture
def sample_params_class_bbrbm():
    weight_matrix = torch.randn(pytest.NUM_VISIBLES, pytest.NUM_HIDDENS)
    vbias = torch.randn(pytest.NUM_VISIBLES)
    hbias = torch.randn(pytest.NUM_HIDDENS)
    params = BBRBM(weight_matrix, vbias, hbias)
    return params


@pytest.fixture
def sample_chains_bbrbm():
    mean_visible = torch.ones(pytest.NUM_CHAINS, pytest.NUM_VISIBLES) / 2
    mean_hidden = torch.ones(pytest.NUM_CHAINS, pytest.NUM_HIDDENS) / 2
    visible = torch.bernoulli(mean_visible)
    hidden = torch.bernoulli(mean_hidden)
    chains = dict(
        visible=visible,
        hidden=hidden,
        visible_mag=mean_visible,
        hidden_mag=mean_hidden,
        weights=torch.ones(pytest.NUM_CHAINS, 1),
    )
    return chains


@pytest.fixture
def sample_data_bbrbm():
    mean_visible = torch.ones(pytest.NUM_SAMPLES, pytest.NUM_VISIBLES) / 2
    mean_hidden = torch.ones(pytest.NUM_SAMPLES, pytest.NUM_HIDDENS) / 2
    visible = torch.bernoulli(mean_visible)
    hidden = torch.bernoulli(mean_hidden)
    chains = dict(
        visible=visible,
        hidden=hidden,
        visible_mag=mean_visible,
        hidden_mag=mean_hidden,
        weights=torch.ones(pytest.NUM_SAMPLES, 1),
    )
    return chains


@pytest.fixture
def sample_filename(tmp_path):
    return tmp_path / "test_model.h5"


@pytest.fixture
def sample_dataset_bbrbm():
    data = (np.random.rand(pytest.NUM_SAMPLES, pytest.NUM_VISIBLES) > 0.5).astype(float)
    weights = np.ones(pytest.NUM_SAMPLES, dtype=np.float32) / pytest.NUM_SAMPLES
    dataset = RBMDataset(
        data=data,
        labels=np.ones(pytest.NUM_SAMPLES),
        weights=weights,
        names=-np.ones(pytest.NUM_SAMPLES),
        dataset_name="test",
        device=torch.device("cpu"),
        dtype=torch.float32,
        is_binary=True,
    )
    return dataset


@pytest.fixture
def sample_binary_v_samples():
    mv = torch.ones(pytest.NUM_SAMPLES, pytest.NUM_VISIBLES) / 2
    v = torch.bernoulli(mv).float()
    return v, mv


@pytest.fixture
def sample_binary_v_chains():
    mv = torch.ones(pytest.NUM_CHAINS, pytest.NUM_VISIBLES) / 2
    v = torch.bernoulli(mv).float()
    return v, mv


@pytest.fixture
def sample_binary_h_samples():
    mh = torch.ones(pytest.NUM_SAMPLES, pytest.NUM_HIDDENS) / 2
    h = torch.bernoulli(mh).float()
    return h, mh


@pytest.fixture
def sample_binary_h_chains():
    mh = torch.ones(pytest.NUM_CHAINS, pytest.NUM_HIDDENS) / 2
    h = torch.bernoulli(mh).float()
    return h, mh


@pytest.fixture
def sample_weight_matrix_bbrbm():
    return torch.randn(pytest.NUM_VISIBLES, pytest.NUM_HIDDENS)


@pytest.fixture
def sample_vbias_bbrbm():
    return torch.randn(pytest.NUM_VISIBLES)


@pytest.fixture
def sample_hbias_bbrbm():
    return torch.randn(pytest.NUM_HIDDENS)


@pytest.fixture
def sample_params_bbrbm(
    sample_vbias_bbrbm, sample_hbias_bbrbm, sample_weight_matrix_bbrbm
):
    return sample_vbias_bbrbm, sample_hbias_bbrbm, sample_weight_matrix_bbrbm


@pytest.fixture
def sample_args(tmp_path):
    filename = tmp_path / "test_model.h5"
    return {
        "learning_rate": pytest.LEARNING_RATE,
        "batch_size": pytest.BATCH_SIZE,
        "num_chains": pytest.NUM_CHAINS,
        "num_hiddens": pytest.NUM_HIDDENS,
        "gibbs_steps": pytest.GIBBS_STEPS,
        "log": True,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
        "num_updates": 3,
        "filename": filename,
        "beta": 1.0,
        "overwrite": True,
        "seed": pytest.SEED,
        "train_size": 0.6,
        "test_size": None,
        "no_center": False,
        "L1": 0.0,
        "L2": 1.0,
    }


@pytest.fixture
def sample_potts_v_samples():
    v = torch.randint(
        0,
        pytest.NUM_STATES,
        (pytest.NUM_SAMPLES, pytest.NUM_VISIBLES),
        dtype=torch.float32,
    )
    return v


@pytest.fixture
def sample_potts_v_chains():
    v = torch.randint(
        0,
        pytest.NUM_STATES,
        (pytest.NUM_CHAINS, pytest.NUM_VISIBLES),
        dtype=torch.float32,
    )
    return v


@pytest.fixture
def sample_chains_pbrbm():
    mean_visible = (
        torch.ones(pytest.NUM_CHAINS, pytest.NUM_VISIBLES, pytest.NUM_STATES) / 2
    )
    mean_hidden = torch.ones(pytest.NUM_CHAINS, pytest.NUM_HIDDENS) / 2
    visible = torch.randint(
        0,
        pytest.NUM_STATES,
        (pytest.NUM_CHAINS, pytest.NUM_VISIBLES),
        dtype=torch.float32,
    )
    hidden = torch.randint(
        0,
        2,
        (pytest.NUM_CHAINS, pytest.NUM_HIDDENS),
        dtype=torch.float32,
    )
    weights = torch.ones(pytest.NUM_CHAINS, dtype=torch.float32)
    chains = dict(
        visible=visible,
        hidden=hidden,
        visible_mag=mean_visible,
        hidden_mag=mean_hidden,
        weights=weights,
    )
    return chains


@pytest.fixture
def sample_data_pbrbm():
    mean_visible = (
        torch.ones(pytest.NUM_SAMPLES, pytest.NUM_VISIBLES, pytest.NUM_STATES) / 2
    )
    mean_hidden = torch.ones(pytest.NUM_SAMPLES, pytest.NUM_HIDDENS) / 2
    visible = torch.randint(
        0, pytest.NUM_STATES, (pytest.NUM_SAMPLES, pytest.NUM_VISIBLES)
    )
    hidden = torch.randint(0, 2, (pytest.NUM_SAMPLES, pytest.NUM_HIDDENS))
    weights = torch.ones(pytest.NUM_SAMPLES, dtype=torch.float32)
    chains = dict(
        visible=visible,
        hidden=hidden,
        visible_mag=mean_visible,
        hidden_mag=mean_hidden,
        weights=weights,
    )
    return chains


@pytest.fixture
def sample_dataset_pbrbm():
    data = np.random.randint(
        0, pytest.NUM_STATES, (pytest.NUM_SAMPLES, pytest.NUM_VISIBLES)
    )

    weights = np.ones(pytest.NUM_SAMPLES, dtype=np.float32) / pytest.NUM_SAMPLES
    dataset = RBMDataset(
        data=data,
        labels=np.ones(pytest.NUM_SAMPLES),
        weights=weights,
        names=-np.ones(pytest.NUM_SAMPLES),
        dataset_name="test",
        device=torch.device("cpu"),
        dtype=torch.float32,
        is_binary=True,
    )
    return dataset


@pytest.fixture
def sample_weight_matrix_pbrbm():
    return torch.randn(pytest.NUM_VISIBLES, pytest.NUM_STATES, pytest.NUM_HIDDENS)


@pytest.fixture
def sample_hbias_pbrbm():
    return torch.randn(pytest.NUM_HIDDENS)


@pytest.fixture
def sample_vbias_pbrbm():
    return torch.randn(pytest.NUM_VISIBLES, pytest.NUM_STATES)


@pytest.fixture
def sample_params_pbrbm(
    sample_vbias_pbrbm, sample_hbias_pbrbm, sample_weight_matrix_pbrbm
):
    return sample_vbias_pbrbm, sample_hbias_pbrbm, sample_weight_matrix_pbrbm


@pytest.fixture
def sample_params_class_pbrbm(
    sample_vbias_pbrbm, sample_hbias_pbrbm, sample_weight_matrix_pbrbm
):
    return PBRBM(
        weight_matrix=sample_weight_matrix_pbrbm,
        vbias=sample_vbias_pbrbm,
        hbias=sample_hbias_pbrbm,
    )
