import numpy as np
import torch
from torch import Tensor

from rbms.classes import EBM
from rbms.dataset.dataset_class import RBMDataset
from rbms.map_model import map_model
from rbms.training.implement import _init_training, _restore_training


def get_checkpoints(num_updates: int, n_save: int, spacing: str = "exp") -> np.ndarray:
    """Select the list of training times (ages) at which to save the model.

    Args:
        num_updates (int): Number of gradient updates to perform during training.
        n_save (int): Number of models to save.
        spacing (str, optional): Spacing method, either "linear" ("lin") or "exponential" ("exp"). Defaults to "exp".

    Returns:
        np.ndarray: Array of checkpoint indices.
    """
    match spacing:
        case "exp":
            checkpoints = []
            xi = num_updates
            for _ in range(n_save):
                checkpoints.append(xi)
                xi = xi / num_updates ** (1 / n_save)
            checkpoints = np.unique(np.array(checkpoints, dtype=np.int32))
        case "linear":
            checkpoints = np.linspace(1, num_updates, n_save).astype(np.int32)
        case _:
            raise ValueError(f"spacing should be one of ('exp', 'linear'), got {spacing}")
    checkpoints = np.unique(np.append(checkpoints, num_updates))
    return checkpoints


def init_training(
    args_save: dict[str, str | int],
    args_train: dict,
    args_grad: dict,
    args_sampling: dict,
    args_init: dict,
    args_dataset: dict,
    args_torch: dict[str, str | torch.dtype],
    train_dataset: RBMDataset,
    flags: list[str] = ["checkpoint"],
    map_model: dict[str, EBM] = map_model,
):
    # Torch
    device: str = args_torch["device"]
    dtype: torch.dtype = args_torch["dtype"]

    # Sampling
    gibbs_steps: int = args_sampling["gibbs_steps"]
    beta: float = args_sampling["beta"]

    # Grad
    centered: bool = not (args_grad["no_center"])
    L1: float = args_grad["L1"]
    L2: float = args_grad["L2"]
    normalize_grad: bool = args_grad["normalize_grad"]
    max_norm_grad: float = args_grad["max_norm_grad"]

    # train
    batch_size: int = args_train["batch_size"]
    num_updates: int = args_train["num_updates"]
    optim: str = args_train["optim"]
    mult_optim: bool = args_train["mult_optim"]
    training_type: str = args_train["training_type"]
    learning_rate: float = args_train["learning_rate"]
    max_lr: float = args_train["max_lr"]

    # save
    filename: str = args_save["filename"]
    n_save: int = args_save["n_save"]
    spacing: str = args_save["spacing"]

    # dataset
    seed: int = args_dataset["seed"]
    train_size: float = args_dataset["train_size"]
    test_size: float = args_dataset["test_size"]
    if test_size is None:
        test_size = 1 - train_size
    subset_labels: list = args_dataset["subset_labels"]
    use_weights: bool = args_dataset["use_weights"]
    alphabet: str = args_dataset["alphabet"]
    remove_duplicates: bool = args_dataset["remove_duplicates"]

    # init
    num_hiddens: int = args_init["num_hiddens"]
    num_chains: int = args_init["num_chains"]
    model_type: str = args_init["model_type"]

    _init_training(
        train_dataset=train_dataset,
        seed=seed,
        train_size=train_size,
        test_size=test_size,
        num_hiddens=num_hiddens,
        num_chains=num_chains,
        model_type=model_type,
        filename=filename,
        n_save=n_save,
        spacing=spacing,
        batch_size=batch_size,
        optim=optim,
        mult_optim=mult_optim,
        training_type=training_type,
        learning_rate=learning_rate,
        max_lr=max_lr,
        gibbs_steps=gibbs_steps,
        beta=beta,
        centered=centered,
        L1=L1,
        L2=L2,
        normalize_grad=normalize_grad,
        max_norm_grad=max_norm_grad,
        subset_labels=subset_labels,
        use_weights=use_weights,
        alphabet=alphabet,
        remove_duplicates=remove_duplicates,
        dtype=dtype,
        device=device,
        flags=flags,
        map_model=map_model,
    )


def restore_training(
    train_dataset: RBMDataset,
    test_dataset: RBMDataset,
    args_save: dict[str, str],
    args_train: dict[str, int | float],
    args_dataset,
    args_torch: dict[str, str | torch.dtype],
    map_model: dict[str, EBM],
) -> tuple[
    EBM,
    dict[str, Tensor],
    int,
    float,
    RBMDataset,
    RBMDataset,
]:
    target_update = args_train["update"]
    filename = args_save["filename"]
    num_updates: int = args_train["num_updates"]

    # Torch
    device: str = args_torch["device"]
    dtype: torch.dtype = args_torch["dtype"]

    # dataset
    seed: int = args_dataset["seed"]
    train_size: float = args_dataset["train_size"]
    test_size: float = args_dataset["test_size"]

    return _restore_training(
        filename=filename,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_updates=num_updates,
        target_update=target_update,
        seed=seed,
        train_size=train_size,
        test_size=test_size,
        device=device,
        dtype=dtype,
    )
