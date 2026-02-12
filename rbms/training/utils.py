import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer

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

    # if model_type is None:
    #     match train_dataset.variable_type:
    #         case "bernoulli":
    #             model_type = "BBRBM"
    #         case "categorical":
    #             model_type = "PBRBM"
    #         case "ising":
    #             model_type = "IIRBM"
    #         case _:
    #             raise NotImplementedError()

    # # Setup dataset
    # num_visibles = train_dataset.get_num_visibles()

    # # Setup RBM
    # params = map_model[model_type].init_parameters(
    #     num_hiddens=num_hiddens,
    #     dataset=train_dataset,
    #     device=device,
    #     dtype=dtype,
    # )
    # if isinstance(params, PBRBM):
    #     ensure_zero_sum_gauge(params)

    # # Permanent chains
    # parallel_chains = params.init_chains(num_samples=num_chains)
    # parallel_chains = params.sample_state(chains=parallel_chains, n_steps=gibbs_steps)

    # # Save hyperparameters
    # if mult_optim:
    #     learning_rate = torch.tensor([learning_rate] * len(params.parameters()))
    # else:
    #     learning_rate = torch.tensor([learning_rate])

    # with h5py.File(filename, "w") as file_model:
    #     hyperparameters = file_model.create_group("hyperparameters")
    #     hyperparameters["num_visibles"] = num_visibles
    #     hyperparameters["num_hiddens"] = num_hiddens
    #     hyperparameters["num_chains"] = num_chains
    #     hyperparameters["filename"] = str(filename)

    # save_model(
    #     filename=filename,
    #     params=params,
    #     chains=parallel_chains,
    #     num_updates=1,
    #     time=0.0,
    #     flags=flags,
    #     learning_rate=learning_rate,
    # )

    # with h5py.File(filename, "a") as f:
    #     dataset = f.create_group("dataset_args")
    #     if subset_labels is not None:
    #         dataset["subset_labels"] = subset_labels
    #     dataset["use_weights"] = use_weights
    #     dataset["train_size"] = train_size
    #     dataset["test_size"] = test_size
    #     dataset["alphabet"] = alphabet
    #     dataset["remove_duplicates"] = remove_duplicates
    #     dataset["seed"] = seed

    #     grad = f.create_group("grad_args")
    #     grad["no_center"] = not (centered)
    #     grad["normalize_grad"] = normalize_grad
    #     grad["max_norm_grad"] = max_norm_grad
    #     grad["L1"] = L1
    #     grad["L2"] = L2

    #     sampling = f.create_group("sampling_args")
    #     sampling["gibbs_steps"] = gibbs_steps
    #     sampling["beta"] = beta

    #     train_args = f.create_group("train_args")
    #     train_args["optim"] = optim
    #     train_args["batch_size"] = batch_size
    #     train_args["learning_rate"] = learning_rate
    #     train_args["training_type"] = training_type
    #     train_args["max_lr"] = max_lr

    #     save_args = f.create_group("save_args")
    #     save_args["n_save"] = n_save
    #     save_args["spacing"] = spacing


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

    # # Retrieve the the number of training updates already performed on the model
    # print(f"Restoring training from update {target_update}")

    # if num_updates <= target_update:
    #     raise RuntimeError(
    #         f"The parameter /'num_updates/' ({num_updates}) must be greater than the previous number of updates ({target_update})."
    #     )

    # params, parallel_chains, elapsed_time = load_model(
    #     filename,
    #     target_update,
    #     device=device,
    #     dtype=dtype,
    #     restore=True,
    #     map_model=map_model,
    # )

    # # Delete all updates after the current one
    # saved_updates = get_saved_updates(filename)
    # if saved_updates[-1] > target_update:
    #     to_delete = saved_updates[saved_updates > target_update]
    #     with h5py.File(filename, "a") as f:
    #         print("Deleting:")
    #         for upd in to_delete:
    #             print(f" - {upd}")
    #             del f[f"update_{upd}"]

    # if test_dataset is None:
    #     print("Splitting dataset")
    #     train_dataset, test_dataset = train_dataset.split_train_test(
    #         rng=np.random.default_rng(seed),
    #         train_size=train_size,
    #         test_size=test_size,
    #     )
    #     print("Train dataset:")
    #     print(train_dataset)
    #     print("Test dataset:")
    #     print(test_dataset)

    # # Initialize gradients for the parameters
    # params.init_grad()

    # train_dataset.match_model_variable_type(params.visible_type)
    # test_dataset.match_model_variable_type(params.visible_type)
    # return (
    #     params,
    #     parallel_chains,
    #     target_update,
    #     elapsed_time,
    #     train_dataset,
    #     test_dataset,
    # )


def pre_grad_update(
    optimizer: list[Optimizer],
    normalize_grad: bool,
    max_grad_norm: float,
    lambda_l1: float,
    lambda_l2: float,
):
    for opt in optimizer:
        if normalize_grad:
            norm_grad = torch.nn.utils.get_total_norm(
                [p.grad for p in opt.param_groups[0]["params"] if p.grad is not None]
            )
            for p in opt.param_groups[0]["params"]:
                p.grad /= norm_grad
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"])
        for p in opt.param_groups[0]["params"]:
            if lambda_l1 > 0:
                p.grad -= lambda_l1 * torch.sign(p)
            if lambda_l2 > 0:
                p.grad -= lambda_l2 * p
