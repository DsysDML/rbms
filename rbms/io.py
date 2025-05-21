from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor

from rbms.classes import EBM
from rbms.map_model import map_model
from rbms.utils import restore_rng_state


def save_model(
    filename: str,
    params: EBM,
    chains: dict[str, Tensor],
    num_updates: int,
    time: float,
    flags: List[str] = [],
) -> None:
    """Save the current state of the model.

    Args:
        filename (str): The name of the file to save the model state.
        params (RBM): The parameters of the RBM.
        chains (dict[str, Tensor]): The parallel chains used for sampling.
        num_updates (int): The number of updates performed.
        time (float): Elapsed time.
        flags (List[str]): flags for the current update. Defaults to []
    """

    named_params = params.named_parameters()
    name = params.name
    with h5py.File(filename, "a") as f:
        checkpoint = f.create_group(f"update_{num_updates}")

        # Save the parameters of the model
        params_ckpt = checkpoint.create_group("params")
        for n, p in named_params.items():
            params_ckpt[n] = p.detach().cpu().numpy()
            # This is for retrocompatibility purpose
            checkpoint[n] = params_ckpt[n]
        # Save current random state
        checkpoint["torch_rng_state"] = torch.get_rng_state()
        checkpoint["numpy_rng_arg0"] = np.random.get_state()[0]
        checkpoint["numpy_rng_arg1"] = np.random.get_state()[1]
        checkpoint["numpy_rng_arg2"] = np.random.get_state()[2]
        checkpoint["numpy_rng_arg3"] = np.random.get_state()[3]
        checkpoint["numpy_rng_arg4"] = np.random.get_state()[4]
        checkpoint["time"] = time

        # Update the parallel chains to resume training
        if "parallel_chains" in f.keys():
            f["parallel_chains"][...] = chains["visible"].cpu().numpy()
        else:
            f["parallel_chains"] = chains["visible"].cpu().numpy()

        if "model_type" not in f.keys():
            f["model_type"] = name
        flag = checkpoint.create_group("flags")
        for fl in flags:
            flag[fl] = True
            # This is for retrocompatibility purpose
            checkpoint[f"save_{fl}"] = True


def load_params(
    filename: str,
    index: int,
    device: torch.device,
    dtype: torch.dtype,
    map_model: dict[str, EBM] = map_model,
) -> EBM:
    """Load the parameters of the RBM from the specified archive at the given update index.

    Args:
        filename (str): The name of the file containing the RBM parameters.
        index (int): The update index from which to load the parameters.
        device (torch.device): The device to move the parameters to.
        dtype (torch.dtype): The data type to convert the parameters to.

    Returns:
        RBM: The loaded RBM parameters.
    """
    last_file_key = f"update_{index}"
    params = {}
    with h5py.File(filename, "r") as f:
        for k in f[last_file_key]["params"].keys():
            params[k] = torch.from_numpy(f[last_file_key]["params"][k][()]).to(
                device=device, dtype=dtype
            )
            model_type = f["model_type"][()].decode()
    return map_model[model_type].set_named_parameters(params)


def load_model(
    filename: str,
    index: int,
    device: torch.device,
    dtype: torch.dtype,
    restore: bool = False,
    map_model: dict[str, EBM] = map_model,
) -> Tuple[EBM, dict[str, Tensor], float, dict]:
    """Load a RBM from a h5 archive.

    Args:
        filename (str): The name of the file containing the RBM model.
        index (int): The update index from which to load the model.
        device (torch.device): The device to move the model to.
        dtype (torch.dtype): The data type to convert the model to.
        restore (bool, optional): Whether to restore the random state at the given update.
            Useful for restoring training. Defaults to False.

    Returns:
        Tuple[EBM, dict[str, Tensor], float, dict]: A tuple containing the loaded RBM parameters,
        the parallel chains, the time taken, and the model's hyperparameters.
    """
    last_file_key = f"update_{index}"
    hyperparameters = dict()
    with h5py.File(filename, "r") as f:
        visible = torch.from_numpy(f["parallel_chains"][()]).to(
            device=device, dtype=dtype
        )
        # Elapsed time
        start = np.array(f[last_file_key]["time"]).item()

        # Hyperparameters
        hyperparameters["batch_size"] = int(f["hyperparameters"]["batch_size"][()])
        hyperparameters["gibbs_steps"] = int(f["hyperparameters"]["gibbs_steps"][()])
        hyperparameters["learning_rate"] = float(
            f["hyperparameters"]["learning_rate"][()]
        )
        hyperparameters["L1"] = float(f["hyperparameters"]["L1"][()])
        hyperparameters["L2"] = float(f["hyperparameters"]["L2"][()])
        if "seed" in f["hyperparameters"].keys():
            hyperparameters["seed"] = int(f["hyperparameters"]["seed"][()])
        if "train_size" in f["hyperparameters"].keys():
            hyperparameters["train_size"] = float(
                f["hyperparameters"]["train_size"][()]
            )
    params = load_params(
        filename=filename, index=index, device=device, dtype=dtype, map_model=map_model
    )
    perm_chains = params.init_chains(visible.shape[0], start_v=visible)

    if restore:
        restore_rng_state(filename=filename, index=index)
    return (params, perm_chains, start, hyperparameters)
