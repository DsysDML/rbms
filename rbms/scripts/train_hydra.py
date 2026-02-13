import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rbms import get_saved_updates
from rbms.classes import EBM
from rbms.dataset.dataset_class import RBMDataset
from rbms.training.implement import _init_training, _restore_training, _train
from rbms.training.pcd import train_v2
from rbms.training.utils import (
    get_checkpoints,
)
from torch import Tensor
from torch.optim import Optimizer


def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


dtype_lookup = {
    "int": torch.int64,
    "half": torch.float16,
    "float": torch.float32,
    "double": torch.float64,
}


def extract_optim_from_hydra_conf(cfg: DictConfig):
    if cfg.optim._target_.split(".")[-1] == "SGD_cossim":
        return "cossim"
    if 'nesterov' in cfg.optim.keys():
        return "nag"
    return "sgd"


def init_training_hydra(cfg: DictConfig, train_dataset: RBMDataset, flags: list[str]):
    print(train_dataset)
    if cfg.rbm.model_type is None:
        match train_dataset.variable_type:
            case "bernoulli":
                OmegaConf.update(
                    cfg, "rbm.model_type", value="BBRBM", force_add=True, merge=False
                )
            case "ising":
                OmegaConf.update(cfg, "rbm.model_type", value="IIRBM")
            case "categorical":
                OmegaConf.update(cfg, "rbm.model_type", value="PBRBM")
            case _:
                pass
    _init_training(
        train_dataset=train_dataset,
        seed=cfg.seed,
        train_size=cfg.train_size,
        test_size=cfg.test_size,
        num_hiddens=cfg.rbm.num_hiddens,
        num_chains=cfg.num_chains,
        model_type=cfg.rbm.model_type,
        filename=cfg.filename,
        n_save=cfg.n_save,
        spacing=cfg.spacing,
        batch_size=cfg.batch_size,
        optim=extract_optim_from_hydra_conf(cfg),
        mult_optim=cfg.mult_optim,
        training_type=cfg.training_type,
        learning_rate=cfg.optim.lr,
        max_lr=cfg.optim.max_lr if "max_lr" in cfg.optim.keys() else cfg.optim.lr,
        gibbs_steps=cfg.sampler.num_steps,
        beta=cfg.sampler.beta,
        centered=cfg.rbm.center,
        L1=cfg.grad.lambda_l1,
        L2=cfg.grad.lambda_l2,
        normalize_grad=cfg.grad.normalize_grad,
        max_norm_grad=cfg.grad.max_grad_norm,
        subset_labels=cfg.dataset.subset_labels,
        use_weights=cfg.dataset.use_weights,
        alphabet=cfg.dataset.alphabet,
        remove_duplicates=cfg.dataset.remove_duplicates,
        dtype=dtype_lookup[cfg.dtype],
        device=cfg.device,
        flags=flags,
    )

import pathlib
def init_training_v2(cfg: DictConfig, train_dataset: RBMDataset, map_model: dict[str, EBM]):
    save_folder = pathlib.Path(cfg.save.folder)  
    # Initialize Model
    params = map_model[cfg.rbm.model_type].init_parameters(
        num_hiddens=cfg.rbm.num_hiddens,
        dataset=train_dataset,
        device=cfg.device,
        dtype=dtype_lookup[cfg.dtype],
    )
    # Initialize Sampler
    parallel_chains = params.init_chains(cfg.rbm.num_chains)
    sampler = hydra.utils.instantiate(cfg.sampler, params=params, chains=parallel_chains)

    # Learning rate

    # Save model
    save_model(
        filename=save_folder / "model.h5",
        params=params,
        chains=parallel_chains,
        num_updates=1,
        time=0.0,
        flags=flags,
        learning_rate=learning_rate,
    )
    # Save Sampler

    # Save updated config 
    OmegaConf.save(cfg, save_folder / "config.yaml")


def restore_training_hydra(
    cfg: DictConfig, train_dataset: RBMDataset, test_dataset: RBMDataset | None
):
    return _restore_training(
        filename=cfg.filename,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_updates=cfg.num_updates,
        target_update=cfg.restore_update,
        seed=cfg.seed,
        train_size=cfg.train_size,
        test_size=cfg.test_size,
        device=cfg.device,
        dtype=dtype_lookup[cfg.dtype],
    )



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.seed is None:
        OmegaConf.update(cfg, "seed", np.random.randint(0, 1000000000000))
    if cfg.test_size is None:
        OmegaConf.update(cfg, "test_size", 1 - cfg.train_size)
    if cfg.num_chains is None:
        cfg.num_chains = cfg.batch_size
    checkpoints = get_checkpoints(
        num_updates=cfg.num_updates, n_save=cfg.n_save, spacing=cfg.spacing
    )
    train_dataset, test_dataset = hydra.utils.call(
        cfg.dataset, device=cfg.device, dtype=dtype_lookup[cfg.dtype]
    )
    print(train_dataset)
    print(test_dataset)

    # Init training
    flags = ["checkpoint"]
    init_training_hydra(cfg=cfg, train_dataset=train_dataset, flags=flags)
    if cfg.restore_update is None:
        OmegaConf.update(
            cfg, "restore_update", int(get_saved_updates(cfg.filename)[-1])
        )
    # Restore training
    (
        params,
        parallel_chains,
        curr_update,
        elapsed_time,
        train_dataset,
        test_dataset,
    ) = restore_training_hydra(
        cfg=cfg, train_dataset=train_dataset, test_dataset=test_dataset
    )
    sampler = hydra.utils.instantiate(
        cfg.sampler, params=params, chains=parallel_chains
    )
    # Setup optimizer
    from omegaconf import ListConfig
    if cfg.mult_optim:
        if not isinstance(cfg.optim.lr, ListConfig):
            OmegaConf.update(cfg, "optim.lr",[cfg.optim.lr] * len(params.parameters()))
        optimizer = [
            hydra.utils.instantiate(cfg.optim, params=params.parameters(), lr=cfg.optim.lr[i])
            for i, p in enumerate(params.parameters())
        ]
    else:
        optimizer = [hydra.utils.instantiate(cfg.optim, params=params.parameters())]

    pre_grad_update = hydra.utils.call(cfg.grad, optimizer=optimizer)
    # Save and clean
    train_v2(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        params=params,
        sampler=sampler,
        optimizer=optimizer,
        batch_size=cfg.batch_size,
        centered=cfg.rbm.center,
        curr_update=curr_update,
        pre_grad_update=pre_grad_update,
        elapsed_time=elapsed_time,
        checkpoints=checkpoints,
        num_updates=cfg.num_updates,
        filename=cfg.filename,
    )



if __name__ == "__main__":
    main()
