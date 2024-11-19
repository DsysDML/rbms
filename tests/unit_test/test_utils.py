import h5py
import numpy as np
import pytest
import torch

from rbms.const import LOG_FILE_HEADER
from rbms.io import load_params, save_model
from rbms.partition_function.ais import update_weights_ais
from rbms.partition_function.exact import compute_partition_function
from rbms.sampling.gibbs import sample_state
from rbms.training.utils import get_checkpoints
from rbms.utils import (
    check_file_existence,
    compute_log_likelihood,
    get_categorical_configurations,
    get_eigenvalues_history,
    get_flagged_updates,
    get_saved_updates,
    log_to_csv,
    query_yes_no,
    restore_rng_state,
    swap_chains,
)


# Helper function to create a temporary HDF5 file for testing
def create_temp_hdf5_file(tmp_path, num_visibles, num_hiddens, num_chains):
    filename = tmp_path / "test_model.h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("parallel_chains", data=np.random.rand(num_chains, num_visibles))
        hyperparameters = f.create_group("hyperparameters")
        hyperparameters.create_dataset("batch_size", data=pytest.BATCH_SIZE)
        hyperparameters.create_dataset("gibbs_steps", data=pytest.GIBBS_STEPS)
        hyperparameters.create_dataset("learning_rate", data=pytest.LEARNING_RATE)
        update_group = f.create_group("update_1")
        params_group = update_group.create_group("params")

        params_group["weight_matrix"] = np.random.rand(
            pytest.NUM_VISIBLES, pytest.NUM_HIDDENS
        )
        update_group.create_dataset("time", data=0.0)
    return filename


# Test get_checkpoints function
def test_get_checkpoints():
    num_updates = 100
    n_save = 5

    checkpoints_exp = get_checkpoints(num_updates, n_save, spacing="exp")
    checkpoints_lin = get_checkpoints(num_updates, n_save, spacing="linear")

    assert len(checkpoints_exp) == n_save
    assert len(checkpoints_lin) == n_save
    assert np.all(checkpoints_exp >= 1)
    assert np.all(checkpoints_lin >= 1)


# Test get_eigenvalues_history function
def test_get_eigenvalues_history(tmp_path):
    filename = create_temp_hdf5_file(
        tmp_path, pytest.NUM_VISIBLES, pytest.NUM_HIDDENS, pytest.NUM_CHAINS
    )
    with h5py.File(filename, "a") as f:
        for i in range(2, 6):
            update_group = f.create_group(f"update_{i}")
            weight_matrix = np.random.rand(pytest.NUM_VISIBLES, pytest.NUM_HIDDENS)
            params_group = update_group.create_group("params")
            params_group["weight_matrix"] = weight_matrix

    gradient_updates, eigenvalues = get_eigenvalues_history(filename)

    assert len(gradient_updates) == 5
    assert eigenvalues.shape[1] == pytest.NUM_HIDDENS
    assert eigenvalues.shape[0] == 5


# Test get_saved_updates function
def test_get_saved_updates(tmp_path):
    filename = create_temp_hdf5_file(
        tmp_path, pytest.NUM_VISIBLES, pytest.NUM_HIDDENS, pytest.NUM_CHAINS
    )
    with h5py.File(filename, "a") as f:
        for i in range(2, 6):
            f.create_group(f"update_{i}")

    updates = get_saved_updates(filename)

    assert len(updates) == 5
    assert np.all(updates == np.arange(5) + 1)


# Test get_categorical_configurations function
def test_get_categorical_configurations():
    n_dim = 3
    n_states = 2
    binary_configs = get_categorical_configurations(n_states=n_states, n_dim=n_dim)

    assert binary_configs.shape == (2**n_dim, n_dim)
    assert torch.all(
        binary_configs
        == torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.float32,
        )
    )


# Test query_yes_no function
def test_query_yes_no(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda: "yes")
    assert query_yes_no("Do you agree?") is True

    monkeypatch.setattr("builtins.input", lambda: "no")
    assert query_yes_no("Do you agree?") is False


# Test check_file_existence function
def test_check_file_existence(tmp_path, monkeypatch):
    filename = tmp_path / "test_file.txt"
    filename.touch()

    monkeypatch.setattr("builtins.input", lambda: "yes")
    check_file_existence(str(filename))
    assert not filename.exists()

    filename.touch()
    monkeypatch.setattr("builtins.input", lambda: "no")
    with pytest.raises(SystemExit):
        check_file_existence(str(filename))


# Test restore_rng_state function
def test_restore_rng_state(tmp_path):
    filename = create_temp_hdf5_file(
        tmp_path, pytest.NUM_VISIBLES, pytest.NUM_HIDDENS, pytest.NUM_CHAINS
    )

    with h5py.File(filename, "a") as f:
        update_group = f["update_1"]
        torch_rng_state = torch.get_rng_state()
        update_group.create_dataset("torch_rng_state", data=torch.get_rng_state().numpy())
        np_rng_state = np.random.get_state()
        update_group.create_dataset("numpy_rng_arg0", data=np_rng_state[0])
        update_group.create_dataset("numpy_rng_arg1", data=np_rng_state[1])
        update_group.create_dataset("numpy_rng_arg2", data=np_rng_state[2])
        update_group.create_dataset("numpy_rng_arg3", data=np_rng_state[3])
        update_group.create_dataset("numpy_rng_arg4", data=np_rng_state[4])

    restore_rng_state(filename, 1)

    assert np.array_equal(np.random.get_state()[0], np_rng_state[0])
    assert np.array_equal(np.random.get_state()[1], np_rng_state[1])
    assert np.array_equal(np.random.get_state()[2], np_rng_state[2])
    assert np.array_equal(np.random.get_state()[3], np_rng_state[3])
    assert np.array_equal(np.random.get_state()[4], np_rng_state[4])
    assert torch.equal(torch.get_rng_state(), torch_rng_state)


# Test log_to_csv function
def test_log_to_csv(tmp_path):
    log_file = tmp_path / "test_log.csv"
    logs = {key: float(i) for i, key in enumerate(LOG_FILE_HEADER)}

    log_to_csv(logs, str(log_file))

    with open(log_file, "r") as f:
        log_content = f.read().strip()
        assert log_content == ",".join(map(str, logs.values()))


# Test compute_log_likelihood function
def test_compute_log_likelihood(sample_params_class_bbrbm, sample_binary_v_samples):
    params = sample_params_class_bbrbm
    v_data, _ = sample_binary_v_samples
    w_data = torch.ones(v_data.shape[0], dtype=v_data.dtype, device=v_data.device)
    log_z = compute_partition_function(
        params,
        get_categorical_configurations(
            n_states=2, n_dim=min(params.num_hiddens(), params.num_visibles())
        ),
    )

    log_likelihood = compute_log_likelihood(v_data, w_data, params, log_z)

    assert isinstance(log_likelihood, float)
    assert log_likelihood < 0


# Test compute_partition_function function
def test_compute_partition_function(sample_params_class_bbrbm):
    params = sample_params_class_bbrbm
    all_config = get_categorical_configurations(n_states=2, n_dim=pytest.NUM_VISIBLES)

    log_z = compute_partition_function(params, all_config)

    assert isinstance(log_z, float)


# Test update_weights_ais function
def test_update_weights_ais(sample_params_class_bbrbm, sample_chains_bbrbm):
    prev_params = sample_params_class_bbrbm
    curr_params = sample_params_class_bbrbm
    chains = sample_chains_bbrbm
    log_weights = torch.zeros(pytest.NUM_CHAINS)

    updated_log_weights, updated_chains = update_weights_ais(
        prev_params, curr_params, chains, log_weights
    )

    assert updated_log_weights.shape == (pytest.NUM_CHAINS,)
    assert isinstance(updated_chains, dict)


# Test save_model function
def test_save_model(tmp_path, sample_params_class_bbrbm, sample_chains_bbrbm):
    filename = tmp_path / "test_model.h5"
    params = sample_params_class_bbrbm
    chains = sample_chains_bbrbm
    num_updates = 1
    time = 0.0

    save_model(str(filename), params, chains, num_updates, time, ["flag_1", "flag_2"])

    with h5py.File(filename, "r") as f:
        assert "update_1" in f.keys()
        assert "parallel_chains" in f.keys()
        assert "time" in f["update_1"].keys()
        assert "flag_1" in f["update_1"]["flags"]
        assert "flag_2" in f["update_1"]["flags"]


# Test load_params function
def test_load_params(tmp_path, sample_params_class_bbrbm):
    filename = create_temp_hdf5_file(
        tmp_path, pytest.NUM_VISIBLES, pytest.NUM_HIDDENS, pytest.NUM_CHAINS
    )
    params = sample_params_class_bbrbm.named_parameters()
    with h5py.File(filename, "a") as f:
        f["model_type"] = "BBRBM"
        # params_group = update_group.create_group("params")
        params_group = f["update_1"]["params"]
        for name, param in params.items():
            if name in params_group.keys():
                del params_group[name]
            params_group.create_dataset(name, data=param.detach().cpu().numpy())

    loaded_params = load_params(filename, 1, torch.device("cpu"), torch.float32)

    for name, param in params.items():
        assert torch.equal(loaded_params.named_parameters()[name], param)


# Test load_params function
def test_load_params_pbrbm(tmp_path, sample_params_class_pbrbm):
    filename = create_temp_hdf5_file(
        tmp_path, pytest.NUM_VISIBLES, pytest.NUM_HIDDENS, pytest.NUM_CHAINS
    )
    params = sample_params_class_pbrbm.named_parameters()
    with h5py.File(filename, "a") as f:
        f["model_type"] = "PBRBM"
        # params_group = update_group.create_group("params")
        params_group = f["update_1"]["params"]
        for name, param in params.items():
            if name in params_group.keys():
                del params_group[name]
            params_group.create_dataset(name, data=param.detach().cpu().numpy())

    loaded_params = load_params(filename, 1, torch.device("cpu"), torch.float32)

    for name, param in params.items():
        assert torch.equal(loaded_params.named_parameters()[name], param)


# Test swap_chains function
def test_swap_chains_bernoulli(sample_chains_bbrbm, sample_params_class_bbrbm):
    import copy

    chain_1 = sample_chains_bbrbm
    chain_2 = sample_state(1, copy.deepcopy(chain_1), sample_params_class_bbrbm)

    assert not (torch.all(chain_1["visible"] == chain_2["visible"]))

    idx = torch.randint(low=0, high=2, size=(pytest.NUM_CHAINS,), dtype=torch.bool)

    saved_chain_1 = copy.deepcopy(chain_1)
    saved_chain_2 = copy.deepcopy(chain_2)

    swapped_chain_1, swapped_chain_2 = swap_chains(chain_1, chain_2, idx)

    assert swapped_chain_1["visible"].shape == saved_chain_1["visible"].shape
    assert swapped_chain_2["visible"].shape == saved_chain_2["visible"].shape
    assert swapped_chain_1["visible_mag"].shape == saved_chain_1["visible_mag"].shape
    assert swapped_chain_2["visible_mag"].shape == saved_chain_2["visible_mag"].shape
    assert swapped_chain_1["hidden"].shape == saved_chain_1["hidden"].shape
    assert swapped_chain_2["hidden"].shape == saved_chain_2["hidden"].shape
    assert swapped_chain_1["hidden_mag"].shape == saved_chain_1["hidden_mag"].shape
    assert swapped_chain_2["hidden_mag"].shape == saved_chain_2["hidden_mag"].shape

    # Test the swaps correspond
    assert torch.all(swapped_chain_1["visible"][idx] == saved_chain_2["visible"][idx])
    assert torch.all(swapped_chain_2["visible"][idx] == saved_chain_1["visible"][idx])

    assert torch.all(
        swapped_chain_1["visible_mag"][idx] == saved_chain_2["visible_mag"][idx]
    )
    assert torch.all(
        swapped_chain_2["visible_mag"][idx] == saved_chain_1["visible_mag"][idx]
    )

    assert torch.all(swapped_chain_1["hidden"][idx] == saved_chain_2["hidden"][idx])
    assert torch.all(swapped_chain_2["hidden"][idx] == saved_chain_1["hidden"][idx])

    assert torch.all(
        swapped_chain_1["hidden_mag"][idx] == saved_chain_2["hidden_mag"][idx]
    )
    assert torch.all(
        swapped_chain_2["hidden_mag"][idx] == saved_chain_1["hidden_mag"][idx]
    )

    # Test the non swaps correspond
    assert torch.all(
        swapped_chain_1["visible"][torch.logical_not(idx)]
        == saved_chain_1["visible"][torch.logical_not(idx)]
    )
    assert torch.all(
        swapped_chain_2["visible"][torch.logical_not(idx)]
        == saved_chain_2["visible"][torch.logical_not(idx)]
    )

    assert torch.all(
        swapped_chain_1["visible_mag"][torch.logical_not(idx)]
        == saved_chain_1["visible_mag"][torch.logical_not(idx)]
    )
    assert torch.all(
        swapped_chain_2["visible_mag"][torch.logical_not(idx)]
        == saved_chain_2["visible_mag"][torch.logical_not(idx)]
    )

    assert torch.all(
        swapped_chain_1["hidden"][torch.logical_not(idx)]
        == saved_chain_1["hidden"][torch.logical_not(idx)]
    )
    assert torch.all(
        swapped_chain_2["hidden"][torch.logical_not(idx)]
        == saved_chain_2["hidden"][torch.logical_not(idx)]
    )

    assert torch.all(
        swapped_chain_1["hidden_mag"][torch.logical_not(idx)]
        == saved_chain_1["hidden_mag"][torch.logical_not(idx)]
    )
    assert torch.all(
        swapped_chain_2["hidden_mag"][torch.logical_not(idx)]
        == saved_chain_2["hidden_mag"][torch.logical_not(idx)]
    )


# Test swap_chains function
def test_swap_chains_potts(sample_chains_pbrbm, sample_params_class_pbrbm):
    chain_1 = sample_chains_pbrbm
    import copy

    chain_2 = copy.deepcopy(chain_1)
    chain_2 = sample_state(1, chain_2, sample_params_class_pbrbm)

    assert not (torch.all(chain_1["visible"] == chain_2["visible"]))

    idx = torch.randint(low=0, high=2, size=(pytest.NUM_CHAINS,), dtype=torch.bool)

    saved_chain_1 = copy.deepcopy(chain_1)
    saved_chain_2 = copy.deepcopy(chain_2)

    swapped_chain_1, swapped_chain_2 = swap_chains(chain_1, chain_2, idx)

    assert swapped_chain_1["visible"].shape == saved_chain_1["visible"].shape
    assert swapped_chain_2["visible"].shape == saved_chain_2["visible"].shape
    assert swapped_chain_1["visible_mag"].shape == saved_chain_1["visible_mag"].shape
    assert swapped_chain_2["visible_mag"].shape == saved_chain_2["visible_mag"].shape
    assert swapped_chain_1["hidden"].shape == saved_chain_1["hidden"].shape
    assert swapped_chain_2["hidden"].shape == saved_chain_2["hidden"].shape
    assert swapped_chain_1["hidden_mag"].shape == saved_chain_1["hidden_mag"].shape
    assert swapped_chain_2["hidden_mag"].shape == saved_chain_2["hidden_mag"].shape

    # Test the swaps correspond
    assert torch.all(swapped_chain_1["visible"][idx] == saved_chain_2["visible"][idx])
    assert torch.all(swapped_chain_2["visible"][idx] == saved_chain_1["visible"][idx])

    assert torch.all(
        swapped_chain_1["visible_mag"][idx] == saved_chain_2["visible_mag"][idx]
    )
    assert torch.all(
        swapped_chain_2["visible_mag"][idx] == saved_chain_1["visible_mag"][idx]
    )

    assert torch.all(swapped_chain_1["hidden"][idx] == saved_chain_2["hidden"][idx])
    assert torch.all(swapped_chain_2["hidden"][idx] == saved_chain_1["hidden"][idx])

    assert torch.all(
        swapped_chain_1["hidden_mag"][idx] == saved_chain_2["hidden_mag"][idx]
    )
    assert torch.all(
        swapped_chain_2["hidden_mag"][idx] == saved_chain_1["hidden_mag"][idx]
    )

    # Test the non swaps correspond
    assert torch.all(
        swapped_chain_1["visible"][torch.logical_not(idx)]
        == saved_chain_1["visible"][torch.logical_not(idx)]
    )
    assert torch.all(
        swapped_chain_2["visible"][torch.logical_not(idx)]
        == saved_chain_2["visible"][torch.logical_not(idx)]
    )

    assert torch.all(
        swapped_chain_1["visible_mag"][torch.logical_not(idx)]
        == saved_chain_1["visible_mag"][torch.logical_not(idx)]
    )
    assert torch.all(
        swapped_chain_2["visible_mag"][torch.logical_not(idx)]
        == saved_chain_2["visible_mag"][torch.logical_not(idx)]
    )

    assert torch.all(
        swapped_chain_1["hidden"][torch.logical_not(idx)]
        == saved_chain_1["hidden"][torch.logical_not(idx)]
    )
    assert torch.all(
        swapped_chain_2["hidden"][torch.logical_not(idx)]
        == saved_chain_2["hidden"][torch.logical_not(idx)]
    )

    assert torch.all(
        swapped_chain_1["hidden_mag"][torch.logical_not(idx)]
        == saved_chain_1["hidden_mag"][torch.logical_not(idx)]
    )
    assert torch.all(
        swapped_chain_2["hidden_mag"][torch.logical_not(idx)]
        == saved_chain_2["hidden_mag"][torch.logical_not(idx)]
    )


# Test get_flagged_updates function
def test_get_flagged_updates(tmp_path):
    filename = create_temp_hdf5_file(
        tmp_path, pytest.NUM_VISIBLES, pytest.NUM_HIDDENS, pytest.NUM_CHAINS
    )
    with h5py.File(filename, "a") as f:
        for i in range(1, 6):
            if f"update_{i}" not in f.keys():
                update_group = f.create_group(f"update_{i}")
            else:
                update_group = f[f"update_{i}"]
            flags_group = update_group.create_group("flags")
            flags_group.create_dataset("flag_1", data=(i % 2 == 0))
            flags_group.create_dataset("flag_2", data=(i % 2 != 0))

    flagged_updates_flag_1 = get_flagged_updates(filename, "flag_1")
    flagged_updates_flag_2 = get_flagged_updates(filename, "flag_2")

    assert np.array_equal(flagged_updates_flag_1, np.array([2, 4]))
    assert np.array_equal(flagged_updates_flag_2, np.array([1, 3, 5]))
