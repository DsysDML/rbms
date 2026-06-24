import h5py
import numpy as np
import torch
from torch import Tensor

from rbms.classes import EBM, Sampler
from rbms.map_model import map_model
from rbms.utils import restore_rng_state


@torch.compiler.disable
def save_model(
    filename: str,
    params: EBM,
    chains: dict[str, Tensor],
    num_updates: int,
    time: float,
    learning_rate: Tensor,
    effective_time: np.ndarray,
    flags: list[str] = [],
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
            params_ckpt[n] = p
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
        checkpoint["effective_time"] = effective_time
        checkpoint["learning_rate"] = learning_rate.cpu().numpy()
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
    device: torch.device | str,
    dtype: torch.dtype,
    map_model: dict[str, type[EBM]] = map_model,
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
            params[k] = f[last_file_key]["params"][k][()]
            model_type = f["model_type"][()].decode()
    return map_model[model_type].set_named_parameters(params, device=device, dtype=dtype)


def load_model(
    filename: str,
    index: int,
    device: torch.device | str,
    dtype: torch.dtype,
    restore: bool = False,
    map_model: dict[str, type[EBM]] = map_model,
) -> tuple[EBM, dict[str, Tensor], float]:
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
        the parallel chains and the time taken
    """
    last_file_key = f"update_{index}"
    with h5py.File(filename, "r") as f:
        visible = torch.from_numpy(f["parallel_chains"][()]).to(
            device=device, dtype=dtype
        )
        # Elapsed time
        start = np.array(f[last_file_key]["time"]).item()

    params = load_params(
        filename=filename, index=index, device=device, dtype=dtype, map_model=map_model
    )
    perm_chains = params.init_chains(visible.shape[0], start_v=visible)

    if restore:
        restore_rng_state(filename=filename, index=index)
    return (params, perm_chains, start)


def save_sampler(filename: str, sampler: Sampler, update: int):
    named_params = sampler.named_parameters()
    metrics = sampler.get_metrics_save()
    name = sampler.name
    with h5py.File(filename, "a") as f:
        if "sampler" not in f.keys():
            f.create_group("sampler")
            f["sampler"]["name"] = name

        # Save the parameters of the model
        for n, p in named_params.items():
            if n in f["sampler"].keys():
                f["sampler"][n][...] = p
            else:
                f["sampler"][n] = p

        if metrics is not None:
            for n, p in metrics.items():
                f[f"update_{update}"][n] = p
