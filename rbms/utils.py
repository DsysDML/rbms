import itertools
import pathlib
import sys
from typing import Tuple

import h5py
import numpy as np
import torch
from torch import Tensor

from rbms.classes import EBM
from rbms.const import LOG_FILE_HEADER


def get_eigenvalues_history(filename: str):
    """
    Extracts the history of eigenvalues of the RBM's weight matrix.

    Args:
        filename (str): Path to the HDF5 training archive.

    Returns:
        tuple: A tuple containing two elements:
            - gradient_updates (np.ndarray): Array of gradient update steps.
            - eigenvalues (np.ndarray): Eigenvalues along training.
    """
    with h5py.File(filename, "r") as f:
        gradient_updates = []
        eigenvalues = []
        for key in f.keys():
            if "update_" in key:
                weight_matrix = f[key]["params"]["weight_matrix"][()]
                weight_matrix = weight_matrix.reshape(-1, weight_matrix.shape[-1])
                eig = np.linalg.svd(weight_matrix, compute_uv=False)
                eigenvalues.append(eig.reshape(*eig.shape, 1))
                gradient_updates.append(int(key.split("_")[1]))

        # Sort the results
        sorting = np.argsort(gradient_updates)
        gradient_updates = np.array(gradient_updates)[sorting]
        eigenvalues = np.array(np.hstack(eigenvalues).T)[sorting]

    return gradient_updates, eigenvalues


def get_saved_updates(filename: str) -> np.ndarray:
    """
    Extracts the saved index from an RBM training archive

    Args:
        filename (str): The path to the HDF5 file from which to extract update indices.

    Returns:
        np.ndarray: Sorted array of update indices.
    """
    updates = []
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            if "update_" in key:
                update = int(key.replace("update_", ""))
                updates.append(update)
    return np.sort(np.array(updates))


def get_categorical_configurations(
    n_states: int,
    n_dim: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Generate all possible categorical configurations for a given number of states and dimensions.

    Args:
        n_states (int): Number of possible states for each dimension.
        n_dim (int): Number of dimensions.
        device (torch.device, optional): Device on which to place the tensor. Default is CPU.
        dtype (torch.dtype, optional): Data type of the returned tensor. Default is torch.float32.

    Returns:
        Tensor: A tensor containing all possible categorical configurations.

    Raises:
        ValueError: If the number of dimensions exceeds the maximum allowed (20).
    """
    max_dim = 20
    if n_dim > max_dim:
        raise ValueError(
            f"The number of dimension for the configurations exceeds the maximum number of dimension: {max_dim}"
        )
    return torch.from_numpy(
        np.array(list(itertools.product(range(n_states), repeat=n_dim)))
    ).to(device=device, dtype=dtype)


def query_yes_no(question: str, default: str = "yes") -> bool:
    """Ask a yes/no question via raw_input() and return their answer.

    Args:
        question (str): Question asked
        default (str, optional): Default answer to the question. Defaults to

    Returns:
        bool: True if yes, False if no

    Notes:
        Credits to "https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input"
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def check_file_existence(filename: str):
    """
    Checks if a file exists at the given filename path. If the file exists, prompts the user
    with a yes/no question on whether to override the file. If the user agrees, the file is deleted.
    If the user disagrees, the program exits.

    Args:
        filename (str): The path to the file to check.

    Raises:
        SystemExit: If the file exists and the user chooses not to override it.
    """
    if pathlib.Path(filename).exists():
        question = f"File: {filename} exists. Do you want to override it ?"
        if query_yes_no(question=question, default="yes"):
            print(f"Deleting {filename}.")
            pathlib.Path(filename).unlink()
        else:
            print("No overriding.")
            sys.exit(0)


def restore_rng_state(filename: str, index: int):
    """
    Restores the random number generator (RNG) states for both PyTorch and NumPy from a RBM training archive.

    Args:
        filename (str): Path to the HDF5 file.
        index (int): Training index to load.

    Raises:
        KeyError: If the specified index does not exist in the HDF5 file.
        OSError: If the file cannot be opened or read.
        ValueError: If the RNG state data is not in the expected format.
    """
    last_file_key = f"update_{index}"
    with h5py.File(filename, "r") as f:
        torch.set_rng_state(torch.tensor(np.array(f[last_file_key]["torch_rng_state"])))
        np_rng_state = tuple(
            [
                f[last_file_key]["numpy_rng_arg0"][()].decode("utf-8"),
                f[last_file_key]["numpy_rng_arg1"][()],
                f[last_file_key]["numpy_rng_arg2"][()],
                f[last_file_key]["numpy_rng_arg3"][()],
                f[last_file_key]["numpy_rng_arg4"][()],
            ]
        )
        np.random.set_state(np_rng_state)


def log_to_csv(logs: dict[str, float], log_file: str) -> None:
    """
    Append log data to a CSV file.

    Parameters:
        logs (dict[str, float]): Dictionary containing log data where keys are the log names and values are the log values.
        log_file (str): Path to the CSV file.
    """
    to_write = ""
    with open(log_file, "a") as f:
        for i, k in enumerate(LOG_FILE_HEADER):
            if k in logs.keys():
                if i == 0:
                    to_write += f"{logs[k]}"
                else:
                    to_write += f",{logs[k]}"
            else:
                if i == 0:
                    to_write += ""
                else:
                    to_write += ","
        f.write(to_write + "\n")


def compute_log_likelihood(
    v_data: Tensor, w_data: Tensor, params: EBM, log_z: float
) -> float:
    """Compute the log likelihood of the RBM on the data, given its log partition function.

    Args:
        v_data (Tensor): Data to estimate the log likelihood.
        w_data (Tensor): Weights associated to the samples.
        params (RBM): Parameters of the RBM.
        log_z (float): Log partition function.

    Returns:
        float: Log Likelihood.
    """
    w_normalized = w_data / w_data.sum()
    return -(params.compute_energy_visibles(v=v_data) @ w_normalized).item() - log_z


@torch.jit.script
def swap_chains(
    chain_1: dict[str, Tensor], chain_2: dict[str, Tensor], idx: Tensor
) -> Tuple[dict[str, Tensor], dict[str, Tensor]]:
    """
    Swap elements between two dict[str, Tensor]s at specified indices.

    Args:
        chain_1 (dict[str, Tensor]): First chain.
        chain_2 (dict[str, Tensor]): Second chain.
        idx (Tensor): Tensor of indices specifying which elements to swap between the chains.

    Returns:
        Tuple[dict[str, Tensor], dict[str, Tensor]]: Modified chains after swapping.
    """
    new_chain_1 = dict()
    new_chain_2 = dict()

    new_chain_1["weights"] = torch.where(
        idx, chain_2["weights"].squeeze(), chain_1["weights"].squeeze()
    ).unsqueeze(-1)
    new_chain_2["weights"] = torch.where(
        idx, chain_1["weights"].squeeze(), chain_2["weights"].squeeze()
    ).unsqueeze(-1)

    idx_vis = idx.unsqueeze(1).repeat(1, chain_1["visible"].shape[1])

    if len(chain_1["visible_mag"].shape) > len(chain_1["visible"].shape):
        idx_vis_mean = idx_vis.repeat(1, chain_1["visible_mag"].shape[2]).reshape(
            chain_1["visible_mag"].shape
        )
    else:
        idx_vis_mean = idx_vis
    idx_hid = idx.unsqueeze(1).repeat(1, chain_1["hidden"].shape[1])

    new_chain_1["visible"] = torch.where(
        idx_vis, chain_2["visible"], chain_1["visible"]
    )
    new_chain_2["visible"] = torch.where(
        idx_vis, chain_1["visible"], chain_2["visible"]
    )

    new_chain_1["visible_mag"] = torch.where(
        idx_vis_mean, chain_2["visible_mag"], chain_1["visible_mag"]
    )
    new_chain_2["visible_mag"] = torch.where(
        idx_vis_mean, chain_1["visible_mag"], chain_2["visible_mag"]
    )

    new_chain_1["hidden"] = torch.where(idx_hid, chain_2["hidden"], chain_1["hidden"])
    new_chain_2["hidden"] = torch.where(idx_hid, chain_1["hidden"], chain_2["hidden"])

    new_chain_1["hidden_mag"] = torch.where(
        idx_hid, chain_2["hidden_mag"], chain_1["hidden_mag"]
    )
    new_chain_2["hidden_mag"] = torch.where(
        idx_hid, chain_1["hidden_mag"], chain_2["hidden_mag"]
    )

    return new_chain_1, new_chain_2


def get_flagged_updates(filename: str, flag: str) -> np.ndarray:
    """
    Retrieve a sorted list of update indices from an HDF5 file that have a specific flag set.

    Args:
        filename (str): Path to the HDF5 file.
        flag (str): Flag to check for in the updates.

    Returns:
        np.ndarray: Sorted array of update indices that have the specified flag set.
    """
    flagged_updates = []
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            if "update_" in key:
                update = int(key.replace("update_", ""))
                if flag in f[key]["flags"]:
                    if f[key]["flags"][flag][()]:
                        flagged_updates.append(update)
    flagged_updates = np.sort(np.array(flagged_updates))
    return flagged_updates
