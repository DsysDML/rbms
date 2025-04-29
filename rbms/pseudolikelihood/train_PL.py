import time
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import Subset

from rbms.classes import RBM
from rbms.bernoulli_bernoulli_BM.bm import BBBM
from rbms.ising.ising_rbm import IsingRBM
from rbms.dataset.dataset_class import RBMDataset
from rbms.io import save_model
from rbms.map_model import map_model
from rbms.potts_bernoulli.classes import PBRBM
from rbms.potts_bernoulli.utils import ensure_zero_sum_gauge
from rbms.sampling.gibbs import sample_state
from rbms.training.utils import create_machine, setup_training
from rbms.utils import check_file_existence, log_to_csv


def step_PL2(
    batch: Tuple[Tensor, Tensor],
    params: RBM,
    l: float,
)->  dict:
    v_data, w_data = batch 
    curr_batch = params.init_chains(
        num_samples=v_data.shape[0],
        weights=w_data,
        start_v=v_data,
    )
    params.compute_gradient_PL2(data=curr_batch, l=l)
    logs = {}
    return logs
    

def train_PL1(
    dataset: RBMDataset,
    test_dataset: RBMDataset,
    model_type: str,
    args: dict,
    dtype: torch.dtype,
    checkpoints: np.ndarray,
    map_model: dict[str, RBM] = map_model,
) -> None:
    """Train the Bernoulli-Bernoulli RBM model.

    Args:
        dataset (RBMDataset): The training dataset.
        test_dataset (RBMDataset): The test dataset (not used).
        model_type (str): Type of RBM used (BBRBM or PBRBM)
        args (dict): A dictionary of training arguments.
        dtype (torch.dtype): The data type for the parameters.
        checkpoints (np.ndarray): An array of checkpoints for saving model states.
    """
    filename = args["filename"]
    if not (args["overwrite"]):
        check_file_existence(filename)
        
    if args["gibbs_steps_init"]:   #MODIFICATION DONE DUE TO BM SLOW DYNAMICS
        gibbs_steps_init = args["gibbs_steps_init"]
    else:
        gibbs_steps_init = 1000

    num_visibles = dataset.get_num_visibles()

    # Create a first archive with the initialized model
    if not (args["restore"]):
        params = map_model[model_type].init_parameters(
            num_hiddens=args["num_hiddens"],
            dataset=dataset,
            device=args["device"],
            dtype=dtype,
            beta=args["beta"],
            use_fields=args["use_fields"]
        )
        create_machine(
            filename=filename,
            params=params,
            num_visibles=num_visibles,
            num_hiddens=args["num_hiddens"],
            num_chains=args["num_chains"],
            batch_size=args["batch_size"],
            gibbs_steps=args["gibbs_steps"],
            learning_rate=args["learning_rate"],
            log=args["log"],
            flags=["checkpoint"],
            gibbs_steps_init=gibbs_steps_init
        )

    (
        params,
        parallel_chains,
        args,
        learning_rate,
        num_updates,
        start,
        elapsed_time,
        log_filename,
        pbar,
    ) = setup_training(args, map_model=map_model)
    
    use_fields = args["use_fields"]
    use_hfield = args["use_hfield"]
    normalize_fields = args["normalize_fields"]
    normalize_K1 = args["normalize_K1"]
    if use_fields==False:
        params.vbias = torch.zeros_like(params.vbias)
        
    params.hbias = torch.zeros_like(params.hbias)
    
    params.K1 = torch.nn.Parameter(params.K1)          ###############
    params.vbias = torch.nn.Parameter(params.vbias) 
    params.hbias = torch.nn.Parameter(params.hbias) 
    

    # for p in params.parameters():
    #     p.grad = torch.zeros_like(p)

    optimizer = AdamW(params.parameters(), lr=learning_rate)#, weight_decay=0.001)#, maximize=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args["lr_arr"], gamma=args["lr_factor"])

    for k, v in args.items():
        print(f"{k} : {v}")

    logs = {}

    # Continue the training
    #with torch.no_grad():
    
    max_norm=50
    
    for idx in range(num_updates + 1, args["num_updates"] + 1):
        rand_idx = torch.randperm(len(dataset))[: args["batch_size"]]
        batch = (dataset.data[rand_idx], dataset.weights[rand_idx])
        #optimizer.zero_grad(set_to_none=False)
        #logs = step_PL2(batch, params, l=args["lambda"]) 
        
            
        loss = params.compute_loss_PL1(batch[0], l=args["lambda"], use_fields=use_fields, use_hfield=use_hfield)
        

        if (args["verbose"]==True) and (idx%10 == 1):
            print("Update: ", idx, "Loss:", loss.item(), "lr:", lr_scheduler.get_last_lr(), "J_norm:", torch.norm(params.K1).item(), "v_norm:", torch.norm(params.vbias).item(), "h_norm:", torch.norm(params.hbias).item())
        
        if (torch.isnan(loss).any() == False) and (torch.isinf(loss).any() == False):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm)
            optimizer.step()
        '''
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            lr_scheduler.base_lrs = [group['lr'] for group in optimizer.param_groups]
        '''
        lr_scheduler.step()
       
        if normalize_K1==True:
            params.normalize_K1()
        if normalize_fields==True:
            params.normalize_v()
        
        # if isinstance(params, BM):
        #     with torch.no_grad():
        #         params.K2.data.fill_diagonal_(0.)
        if isinstance(params, PBRBM):
            ensure_zero_sum_gauge(params)
            
        
        # Save current model if necessary
        if idx in checkpoints:
            curr_time = time.time() - start
            save_model(
                filename=args["filename"],
                params=params,
                chains=parallel_chains,
                num_updates=idx,
                time=curr_time + elapsed_time,
                flags=["checkpoint"],
            )
        if args["log"]:
            log_to_csv(logs, log_file=log_filename)
        # Update progress bar
        pbar.update(1)


def train_PL2(
    dataset: RBMDataset,
    test_dataset: RBMDataset,
    model_type: str,
    args: dict,
    dtype: torch.dtype,
    checkpoints: np.ndarray,
    map_model: dict[str, RBM] = map_model,
) -> None:
    """Train the Bernoulli-Bernoulli RBM model.

    Args:
        dataset (RBMDataset): The training dataset.
        test_dataset (RBMDataset): The test dataset (not used).
        model_type (str): Type of RBM used (BBRBM or PBRBM)
        args (dict): A dictionary of training arguments.
        dtype (torch.dtype): The data type for the parameters.
        checkpoints (np.ndarray): An array of checkpoints for saving model states.
    """
    filename = args["filename"]
    if not (args["overwrite"]):
        check_file_existence(filename)
        
    if args["gibbs_steps_init"]:   #MODIFICATION DONE DUE TO BM SLOW DYNAMICS
        gibbs_steps_init = args["gibbs_steps_init"]
    else:
        gibbs_steps_init = 1000

    num_visibles = dataset.get_num_visibles()

    # Create a first archive with the initialized model
    if not (args["restore"]):
        params = map_model[model_type].init_parameters(
            num_hiddens=args["num_hiddens"],
            dataset=dataset,
            device=args["device"],
            dtype=dtype,
            beta=args["beta"],
            use_fields=args["use_fields"]
        )
        create_machine(
            filename=filename,
            params=params,
            num_visibles=num_visibles,
            num_hiddens=args["num_hiddens"],
            num_chains=args["num_chains"],
            batch_size=args["batch_size"],
            gibbs_steps=args["gibbs_steps"],
            learning_rate=args["learning_rate"],
            log=args["log"],
            flags=["checkpoint"],
            gibbs_steps_init=gibbs_steps_init
        )

    (
        params,
        parallel_chains,
        args,
        learning_rate,
        num_updates,
        start,
        elapsed_time,
        log_filename,
        pbar,
    ) = setup_training(args, map_model=map_model)
    
    use_fields = args["use_fields"]
    use_hfield = args["use_hfield"]
    normalize_fields = args["normalize_fields"]
    normalize_K2 = args["normalize_K2"]
    if use_fields==False:
        params.vbias = torch.zeros_like(params.vbias)
        
    params.hbias = torch.zeros_like(params.hbias)
    
    params.K2 = torch.nn.Parameter(params.K2)          ###############
    params.vbias = torch.nn.Parameter(params.vbias) 
    params.hbias = torch.nn.Parameter(params.hbias) 
    

    # for p in params.parameters():
    #     p.grad = torch.zeros_like(p)

    optimizer = Adam(params.parameters(), lr=learning_rate)#, weight_decay=0.001)#, maximize=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args["lr_arr"], gamma=args["lr_factor"])

    for k, v in args.items():
        print(f"{k} : {v}")

    logs = {}

    # Continue the training
    #with torch.no_grad():
    
    max_norm=50
    
    for idx in range(num_updates + 1, args["num_updates"] + 1):
        rand_idx = torch.randperm(len(dataset))[: args["batch_size"]]
        batch = (dataset.data[rand_idx], dataset.weights[rand_idx])
        #optimizer.zero_grad(set_to_none=False)
        #logs = step_PL2(batch, params, l=args["lambda"]) 
        
            
        loss = params.compute_loss_PL2(batch[0], l=args["lambda"], use_fields=use_fields, use_hfield=use_hfield)
        

        if (args["verbose"]==True) and (idx%10 == 1):
            print("Update: ", idx, "Loss:", loss.item(), "lr:", lr_scheduler.get_last_lr(), "J_norm:", torch.norm(params.K2).item(), "v_norm:", torch.norm(params.vbias).item(), "h_norm:", torch.norm(params.hbias).item())
        
        if (torch.isnan(loss).any() == False) and (torch.isinf(loss).any() == False):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm)
            optimizer.step()
        '''
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            lr_scheduler.base_lrs = [group['lr'] for group in optimizer.param_groups]
        '''
        lr_scheduler.step()
       
        if normalize_K2==True:
            params.normalize_K2()
        if normalize_fields==True:
            params.normalize_v()
        
        # if isinstance(params, BM):
        #     with torch.no_grad():
        #         params.K2.data.fill_diagonal_(0.)
        if isinstance(params, PBRBM):
            ensure_zero_sum_gauge(params)
            
        
        # Save current model if necessary
        if idx in checkpoints:
            curr_time = time.time() - start
            save_model(
                filename=args["filename"],
                params=params,
                chains=parallel_chains,
                num_updates=idx,
                time=curr_time + elapsed_time,
                flags=["checkpoint"],
            )
        if args["log"]:
            log_to_csv(logs, log_file=log_filename)
        # Update progress bar
        pbar.update(1)
