from pathlib import Path

import pytest
import torch

from rbms.const import LOG_FILE_HEADER
from rbms.io import load_model
from rbms.potts_bernoulli.classes import PBRBM
from rbms.training.utils import create_machine


# Helper function to create a temporary HDF5 file for testing
def create_temp_hdf5_file(tmp_path, sample_params, sample_chains):
    filename = tmp_path / "test_model.h5"
    create_machine(
        filename, params=sample_params, chains=sample_chains, num_updates=1, time=0
    )
    return filename


def test_create_load_machine(tmp_path, sample_params_class_pbrbm):
    filename = tmp_path / "test_model.h5"
    device = torch.device("cpu")
    dtype = torch.float32
    create_machine(
        filename=str(filename),
        params=sample_params_class_pbrbm,
        num_visibles=pytest.NUM_VISIBLES,
        num_hiddens=pytest.NUM_HIDDENS,
        num_chains=pytest.NUM_CHAINS,
        batch_size=pytest.BATCH_SIZE,
        gibbs_steps=pytest.GIBBS_STEPS,
        learning_rate=pytest.LEARNING_RATE,
        log=True,
        flags=["test"],
    )

    # Check if the file was created
    assert filename.exists()

    # Check if the log file was created
    log_filename = filename.parent / Path(f"log-{filename.stem}.csv")
    assert log_filename.exists()

    # Check the contents of the log file
    with open(log_filename, "r", encoding="utf-8") as log_file:
        header = log_file.readline().strip()
        assert header == ",".join(LOG_FILE_HEADER)

    params, chains, start, hyperparameters = load_model(
        filename=str(filename),
        index=1,
        device=device,
        dtype=dtype,
        restore=False,
    )
    assert isinstance(params, PBRBM)
    assert isinstance(chains, dict)
    assert isinstance(start, float)
    assert isinstance(hyperparameters, dict)
    assert hyperparameters["batch_size"] == pytest.BATCH_SIZE
    assert hyperparameters["gibbs_steps"] == pytest.GIBBS_STEPS
    assert hyperparameters["learning_rate"] == pytest.LEARNING_RATE


def test_create_load_machine_dtype(tmp_path, sample_params_class_pbrbm):
    filename = tmp_path / "test_model.h5"
    device = torch.device("cpu")
    dtype = torch.float64
    create_machine(
        filename=str(filename),
        params=sample_params_class_pbrbm,
        num_visibles=pytest.NUM_VISIBLES,
        num_hiddens=pytest.NUM_HIDDENS,
        num_chains=pytest.NUM_CHAINS,
        batch_size=pytest.BATCH_SIZE,
        gibbs_steps=pytest.GIBBS_STEPS,
        learning_rate=pytest.LEARNING_RATE,
        log=True,
        flags=["test"],
    )

    # Check if the file was created
    assert filename.exists()

    # Check if the log file was created
    log_filename = filename.parent / Path(f"log-{filename.stem}.csv")
    assert log_filename.exists()

    # Check the contents of the log file
    with open(log_filename, "r", encoding="utf-8") as log_file:
        header = log_file.readline().strip()
        assert header == ",".join(LOG_FILE_HEADER)

    params, chains, start, hyperparameters = load_model(
        filename=str(filename),
        index=1,
        device=device,
        dtype=dtype,
        restore=False,
    )
    assert isinstance(params, PBRBM)
    assert isinstance(chains, dict)
    assert isinstance(start, float)
    assert isinstance(hyperparameters, dict)
    assert hyperparameters["batch_size"] == pytest.BATCH_SIZE
    assert hyperparameters["gibbs_steps"] == pytest.GIBBS_STEPS
    assert hyperparameters["learning_rate"] == pytest.LEARNING_RATE
