import h5py
import numpy as np
import torch
from torch import Tensor

from rbms.classes import EBM
from rbms.dataset.dataset_class import RBMDataset
from rbms.io import load_model, save_model
from rbms.map_model import map_model
from rbms.utils import get_saved_updates


def _init_training(
    train_dataset: RBMDataset,
    seed: int,
    train_size: float,
    test_size: float,
    num_hiddens: int,
    num_chains: int,
    model_type: str,
    filename: str,
    n_save: int,
    spacing: str,
    batch_size: int,
    optim: str,
    mult_optim: bool,
    training_type: str,
    learning_rate: float,
    max_lr: float,
    gibbs_steps: int,
    beta: float,
    centered: bool,
    L1: float,
    L2: float,
    normalize_grad: bool,
    max_norm_grad: float,
    subset_labels: list,
    use_weights: bool,
    alphabet: str,
    remove_duplicates: bool,
    dtype: torch.dtype,
    device: torch.device | str,
    flags: list[str],
    map_model: dict[str, type[EBM]] = map_model,
):
    if model_type is None:
        match train_dataset.variable_type:
            case "bernoulli":
                model_type = "BBRBM"
            case "categorical":
                model_type = "PBRBM"
            case "ising":
                model_type = "IIRBM"
            case _:
                raise NotImplementedError()

    train_dataset.match_model_variable_type(map_model[model_type].visible_type)
    # Setup dataset
    num_visibles = train_dataset.get_num_visibles()

    # Setup RBM
    params = map_model[model_type].init_parameters(
        num_hiddens=num_hiddens,
        dataset=train_dataset,
        device=device,
        dtype=dtype,
    )

    # Permanent chains
    parallel_chains = params.init_chains(num_samples=num_chains)
    parallel_chains = params.sample_state(chains=parallel_chains, n_steps=gibbs_steps)

    # Save hyperparameters
    if mult_optim:
        lr = torch.tensor([learning_rate] * len(params.parameters()))
    else:
        lr = torch.tensor([learning_rate])

    with h5py.File(filename, "w") as file_model:
        hyperparameters = file_model.create_group("hyperparameters")
        hyperparameters["num_visibles"] = num_visibles
        hyperparameters["num_hiddens"] = num_hiddens
        hyperparameters["num_chains"] = num_chains
        hyperparameters["filename"] = str(filename)
    effective_time = torch.zeros_like(lr).cpu().numpy()
    save_model(
        filename=filename,
        params=params,
        chains=parallel_chains,
        num_updates=1,
        time=0.0,
        flags=flags,
        effective_time=effective_time,
        learning_rate=lr,
    )

    with h5py.File(filename, "a") as f:
        dataset = f.create_group("dataset_args")
        if subset_labels is not None:
            dataset["subset_labels"] = subset_labels
        dataset["use_weights"] = use_weights
        dataset["train_size"] = train_size
        dataset["test_size"] = test_size
        dataset["alphabet"] = np.asarray(alphabet, dtype="T")
        dataset["remove_duplicates"] = remove_duplicates
        dataset["seed"] = seed

        grad = f.create_group("grad_args")
        grad["no_center"] = not (centered)
        grad["normalize_grad"] = normalize_grad
        grad["max_norm_grad"] = max_norm_grad
        grad["L1"] = L1
        grad["L2"] = L2

        sampling = f.create_group("sampling_args")
        sampling["gibbs_steps"] = gibbs_steps
        sampling["beta"] = beta

        train_args = f.create_group("train_args")
        train_args["optim"] = np.asarray(optim, dtype="T")
        train_args["batch_size"] = batch_size
        train_args["learning_rate"] = lr
        train_args["training_type"] = np.asarray(training_type, dtype="T")
        train_args["max_lr"] = max_lr

        save_args = f.create_group("save_args")
        save_args["n_save"] = n_save
        save_args["spacing"] = np.asarray(spacing, dtype="T")


def _restore_training(
    filename: str,
    train_dataset: RBMDataset,
    test_dataset: RBMDataset | None,
    num_updates: int,
    target_update: int,
    seed: int,
    train_size: float,
    test_size: float,
    device: str,
    dtype: torch.dtype,
    map_model: dict[str, type[EBM]] = map_model,
) -> tuple[EBM, dict[str, Tensor], int, float, RBMDataset, RBMDataset]:
    # Retrieve the the number of training updates already performed on the model
    print(f"Restoring training from update {target_update}")

    if num_updates <= target_update:
        raise RuntimeError(
            f"The parameter /'num_updates/' ({num_updates}) must be greater than the previous number of updates ({target_update})."
        )

    params, parallel_chains, elapsed_time = load_model(
        filename,
        target_update,
        device=device,
        dtype=dtype,
        restore=True,
        map_model=map_model,
    )

    # Delete all updates after the current one
    saved_updates = get_saved_updates(filename)
    if saved_updates[-1] > target_update:
        to_delete = saved_updates[saved_updates > target_update]
        with h5py.File(filename, "a") as f:
            print("Deleting:")
            for upd in to_delete:
                print(f" - {upd}")
                del f[f"update_{upd}"]

    if test_dataset is None:
        print("Splitting dataset")
        train_dataset, test_dataset = train_dataset.split_train_test(
            rng=np.random.default_rng(seed),
            train_size=train_size,
            test_size=test_size,
        )
        print("Train dataset:")
        print(train_dataset)
        print("Test dataset:")
        print(test_dataset)

    # Initialize gradients for the parameters
    params.init_grad()

    train_dataset.match_model_variable_type(params.visible_type)
    test_dataset.match_model_variable_type(params.visible_type)
    return (
        params,
        parallel_chains,
        target_update,
        elapsed_time,
        train_dataset,
        test_dataset,
    )
