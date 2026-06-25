import time

import numpy as np
import torch
from torch.optim import Optimizer
from tqdm.autonotebook import tqdm

from rbms.classes import EBM, Sampler
from rbms.dataset.dataset_class import RBMDataset
from rbms.io import save_model, save_sampler


# @torch.no_grad
def train(
    train_dataset: RBMDataset,
    test_dataset: RBMDataset,
    params: EBM,
    sampler: Sampler,
    optimizer: list[Optimizer],
    # early_stopper: EarlyStopper | None,
    batch_size: int,
    centered: bool,
    curr_update: int,
    pre_grad_update: torch.nn.Sequential,
    elapsed_time: float,
    checkpoints: np.ndarray,
    num_updates: int,
    filename: str,
):
    pbar = tqdm(
        initial=curr_update,
        total=num_updates,
        colour="red",
        dynamic_ncols=True,
        ascii="-#",
    )
    pbar.set_description(f"Training {params.name}")

    start = time.perf_counter()
    effective_time = np.zeros(len(optimizer))
    for idx in range(curr_update + 1, num_updates + 1):
        batch = train_dataset.batch(batch_size)
        data, weights = batch["data"], batch["weights"]

        for opt in optimizer:
            opt.zero_grad(set_to_none=False)

        # Initialize batch
        curr_batch = params.init_chains(
            num_samples=data.shape[0],
            weights=weights,
            start_v=data,
        )
        parallel_chains = sampler.get_conf_grad(batch=data)

        params.compute_gradient(
            data=curr_batch,
            chains=parallel_chains,
            centered=centered,
        )
        # Do a bunch of modification on the gradient

        pre_grad_update(input=None)
        params.pre_grad_update()
        sampler.pre_grad_update()

        for opt in optimizer:
            opt.step()

        params.post_grad_update()
        sampler.post_grad_update(params=params)
        learning_rates = np.asarray([opt.param_groups[0]["lr"] for opt in optimizer])
        effective_time += learning_rates
        # Get flags for save
        flags = []
        flags = params.save_flags(flags)
        flags = sampler.save_flags(flags)
        if idx in checkpoints or idx == num_updates:
            flags.append("checkpoint")

        if len(flags) > 0:
            names_params = (
                list(params.named_parameters().keys()) if len(optimizer) > 1 else ["all"]
            )

            metrics = {}
            metrics = sampler.get_metrics_display(
                metrics, train_dataset=train_dataset, test_dataset=test_dataset
            )
            pbar.write(f"=========== Update {idx} ===========")
            for k, v in metrics.items():
                pbar.write(f"{k}: {v}")
            pbar.write("learning rate :")
            for i in range(len(optimizer)):
                pbar.write(f"    - {names_params[i]} : {learning_rates[i]:.6f}")

            # pbar.write(metrics)
            curr_time = time.perf_counter() - start
            learning_rate = torch.tensor([opt.param_groups[0]["lr"] for opt in optimizer])
            save_model(
                filename=filename,
                params=params,
                chains=parallel_chains,
                num_updates=idx,
                time=curr_time + elapsed_time,
                learning_rate=learning_rate,
                effective_time=effective_time,
                flags=flags,
            )

            save_sampler(filename, sampler, idx)
        pbar.update(1)
