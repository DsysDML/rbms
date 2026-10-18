import itertools
import pathlib
import sys

import h5py
import numpy as np
import torch
from torch import Tensor

from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.classes import EBM
from rbms.const import LOG_FILE_HEADER
from rbms.ising_ising.classes import IIRBM


def get_eigenvalues_history(filename: str, backend="cpu"):
    """
    Extracts the history of eigenvalues of the RBM's weight matrix.

    Args:
        filename (str): Path to the HDF5 training archive.

    Returns:
        tuple: A tuple containing two elements:
            - gradient_updates (np.ndarray): Array of gradient update steps.
            - eigenvalues (np.ndarray): Eigenvalues along training.
    """
    saved_updates = get_saved_updates(filename)
    eigenvalues = []
    for upd in saved_updates:
        compute = False
        with h5py.File(filename, "a") as f:
            if "singular_values" not in f[f"update_{upd}"]:
                compute = True
                weight_matrix = f[f"update_{upd}"]["params"]["weight_matrix"][()]

        if compute:
            weight_matrix = weight_matrix.reshape(-1, weight_matrix.shape[-1])
            if backend == "gpu":
                eig = (
                    torch.svd(
                        torch.from_numpy(weight_matrix).to(device="cuda"),
                        compute_uv=False,
                    )
                    .S.cpu()
                    .numpy()
                )
            else:
                eig = np.linalg.svd(weight_matrix, compute_uv=False)
            with h5py.File(filename, "a") as f:
                f[f"update_{upd}"]["singular_values"] = eig

        with h5py.File(filename, "a") as f:
            eig = f[f"update_{upd}"]["singular_values"][()]
            eigenvalues.append(eig.reshape(*eig.shape, 1))
    eigenvalues = np.array(np.hstack(eigenvalues).T)
    return saved_updates, eigenvalues


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
    max_dim = 25
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
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


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
    return (
        -(params.compute_energy_visibles({"visible": v_data}) @ w_normalized).item()
        - log_z
    )


@torch.jit.script
def swap_chains(
    chain_1: dict[str, Tensor], chain_2: dict[str, Tensor], idx: Tensor
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
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

    new_chain_1["visible"] = torch.where(idx_vis, chain_2["visible"], chain_1["visible"])
    new_chain_2["visible"] = torch.where(idx_vis, chain_1["visible"], chain_2["visible"])

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
    flagged_updates = np.sort(np.array(flagged_updates, dtype=int))
    return flagged_updates


def bernoulli_to_ising(params: BBRBM) -> IIRBM:
    weight_matrix = 0.25 * params.weight_matrix
    vbias = 0.5 * params.vbias + weight_matrix.sum(dim=1)
    hbias = 0.5 * params.hbias + weight_matrix.sum(dim=0)
    return IIRBM(vbias=vbias, hbias=hbias, weight_matrix=weight_matrix)


def ising_to_bernoulli(params: IIRBM) -> BBRBM:
    weight_matrix = 4.0 * params.weight_matrix
    vbias = 2.0 * params.vbias - 2.0 * params.weight_matrix.sum(dim=1)
    hbias = 2.0 * params.hbias - 2.0 * params.weight_matrix.sum(dim=0)
    return BBRBM(vbias=vbias, hbias=hbias, weight_matrix=weight_matrix)


def local_square_mask(
    centers: Tensor, side: int, height: int, width: int
) -> Tensor:
    """(height * width, num_centers) mask of the pixels in the square of side `side` around
    each center (flattened index of a pixel), cut at the borders of the image. The square
    spans the rows i - (side - 1) // 2, ..., i + side // 2 (same for the columns): for an odd
    side 2 r + 1, the pixels at a distance at most r in each direction."""
    rows, cols = centers // width, centers % width
    lo, hi = (side - 1) // 2, side // 2
    i = torch.arange(height).unsqueeze(1)
    j = torch.arange(width).unsqueeze(1)
    in_rows = (i >= rows - lo) & (i <= rows + hi)  # (height, num_centers)
    in_cols = (j >= cols - lo) & (j <= cols + hi)  # (width, num_centers)
    return (in_rows.unsqueeze(1) & in_cols.unsqueeze(0)).reshape(height * width, -1)


def make_weight_mask(
    weight_matrix: Tensor, sparsity: float, seed: int, mask_type: str = "random"
) -> tuple[Tensor, dict]:
    """Mask of the couplings for a sparse training: a fraction `sparsity` (rho) of the
    couplings between a visible and a hidden unit is trained, the others are fixed to zero.

    mask_type:
        - "random": exactly round(rho * Nv * Nh) couplings chosen at random.
        - "local": the visibles are the pixels of a square image. Each hidden unit is coupled
          to the pixels of a square around its center (cut at the borders). The centers are
          the pixels in random order, each one used once before any is repeated (with
          Nh = Nv every pixel is the center of one hidden unit). The side of the squares is
          the one whose fraction of couplings is the closest to rho.

    Returns the mask, of shape (Nv, 1, ..., Nh), broadcastable to the weight matrix (for Potts
    visibles, all the states of a visible are coupled or not to a hidden unit), and a dict
    with the details of the mask to save in the archive (fraction of trained couplings, side
    of the squares).
    """
    if not 0 < sparsity <= 1:
        raise ValueError(f"The sparsity must be in (0, 1], got {sparsity}")
    num_visibles, num_hiddens = weight_matrix.shape[0], weight_matrix.shape[-1]
    num_couplings = num_visibles * num_hiddens
    generator = torch.Generator().manual_seed(int(seed))
    info = {}
    match mask_type:
        case "random":
            kept = torch.randperm(num_couplings, generator=generator)[
                : round(sparsity * num_couplings)
            ]
            mask = torch.zeros(num_couplings, dtype=torch.bool)
            mask[kept] = True
        case "local":
            side_image = int(round(num_visibles**0.5))
            if side_image**2 != num_visibles:
                raise ValueError(
                    f"Local mask: the {num_visibles} visibles are not a square image"
                )
            num_perms = -(-num_hiddens // num_visibles)
            centers = torch.cat(
                [torch.randperm(num_visibles, generator=generator) for _ in range(num_perms)]
            )[:num_hiddens]
            # Up to 2 L - 1 the square covers the whole image from any center
            masks = [
                local_square_mask(centers, side, side_image, side_image)
                for side in range(1, 2 * side_image)
            ]
            densities = torch.tensor([m.float().mean().item() for m in masks])
            best = int((densities - sparsity).abs().argmin())
            mask = masks[best]
            info["mask_side"] = best + 1
            info["mask_centers"] = centers.numpy()
        case _:
            raise ValueError(f"Unknown mask type '{mask_type}', should be 'random' or 'local'")
    info["mask_density"] = mask.float().mean().item()
    shape = (num_visibles, *[1] * (weight_matrix.dim() - 2), num_hiddens)
    mask = mask.reshape(shape).to(device=weight_matrix.device, dtype=weight_matrix.dtype)
    return mask, info
