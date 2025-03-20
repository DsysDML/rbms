from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import softmax


@torch.jit.script
def _sample_hiddens(
    v: Tensor, weight_matrix: Tensor, hbias: Tensor, beta: float = 1.0
) -> Tuple[Tensor, Tensor]:
    dtype = torch.float32
    return v, v    #nothing happens to visible after dummy hidden passage


@torch.jit.script
def _sample_visibles(
    v: Tensor, h: Tensor, weight_matrix: Tensor, vbias: Tensor, beta: float = 1.0
) -> Tuple[Tensor, Tensor]:
    x = v.clone()
    num_chains, num_units = x.shape
    for i in range(num_units):   #it has to be sequential
        h_i = torch.einsum("j,bj->b",weight_matrix[i],x)      #diagonal is already zero               
        probs = torch.sigmoid(beta*h_i)
        x[:,i] = torch.bernoulli(probs)*2-1
    return x, x


@torch.jit.script
def _compute_energy(
    v: Tensor,
    vbias: Tensor,
    weight_matrix: Tensor,
) -> Tensor:
    x = v.clone()
    fields = (x * vbias).sum(dim=1)
    interaction = 0.5 * torch.einsum("bi,ij,bj->b", x, weight_matrix, x)

    return -fields - interaction


@torch.jit.script
def _compute_energy_visibles(
    v: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
) -> Tensor:
    x = v.clone()
    fields = (x * vbias).sum(dim=1)
    interaction = 0.5 * torch.einsum("bi,ij,bj->b", x, weight_matrix, x)

    return -fields - interaction


@torch.jit.script
def _compute_energy_hiddens(
    h: Tensor, vbias: Tensor, hbias: Tensor, weight_matrix: Tensor
) -> Tensor:
    dtype = torch.float32
    return torch.tensor([0], device=weight_matrix.device, dtype=dtype)


@torch.jit.script
def _compute_gradient(
    v_data: Tensor,
    v_chain: Tensor,
    weight_matrix: Tensor,
    vbias: Tensor,
    centered: bool = False,
) -> None:
    dtype = torch.float32
    
    x_data = v_data
    x_model = v_chain

    if centered:
        pass
    else:
        pass
    
    # Empirical averages
    data_mean = x_data.mean(dim=0)                # shape (num_units,)
    data_corr = (x_data.unsqueeze(2) * x_data.unsqueeze(1)).mean(dim=0)  # shape (num_units, num_units)
    # Model (chain) averages
    model_mean = x_model.mean(dim=0)  
    model_corr = (x_model.unsqueeze(2) * x_model.unsqueeze(1)).mean(dim=0)
    # Bias gradient
    grad_weight_matrix = data_corr - model_corr 
    grad_weight_matrix.fill_diagonal_(0.0)                                  
    grad_vbias = torch.tensor([0], device=weight_matrix.device, dtype=dtype)
    #grad_hbias = torch.tensor([0], device=weight_matrix.device, dtype=weight_matrix.type)

    # Attach to the parameters

    weight_matrix.grad.set_(grad_weight_matrix)
    vbias.grad.set_(grad_vbias)
    #hbias.grad.set_(grad_hbias)

@torch.jit.script
def _init_chains(
    num_samples: int,
    weight_matrix: Tensor,
    hbias: Tensor,
    start_v: Optional[Tensor] = None,
):
    num_visibles = weight_matrix.shape[0]
    device = weight_matrix.device
    dtype = torch.float32
    # Handle negative number of samples
    if num_samples <= 0:
        if start_v is not None:
            num_samples = start_v.shape[0]
        else:
            raise ValueError(f"Got negative num_samples arg: {num_samples}")

    if start_v is None:
        # Dummy mean visible
        v = torch.randint(low=0, high=2, size=(num_samples, num_visibles), device=device, dtype=torch.float32)*2-1
    else:
        v = start_v.to(device=device, dtype=dtype)
    return v


def _init_parameters(
    data: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    var_init: float = 1e-2,
) -> Tuple[Tensor, Tensor, Tensor]:
    _, num_visibles = data.shape
    eps = 1e-4
    weight_matrix = (
        torch.randn(size=(num_visibles, num_visibles), device=device, dtype=dtype)
        * var_init
    )
    weight_matrix.fill_diagonal_(0.0)
    weight_matrix = 0.5*(weight_matrix+weight_matrix.T)
    '''
    frequencies = data.mean(0)
    frequencies = torch.clamp(frequencies, min=eps, max=(1.0 - eps))
    vbias = (torch.log(frequencies) - torch.log(1.0 - frequencies)).to(
        device=device, dtype=dtype
    )
    hbias = torch.zeros(num_hiddens, device=device, dtype=dtype)
    '''
    return weight_matrix
