import rbms
from rbms.classes import RBM

from rbms.bernoulli_bernoulli_BM.implement import (
    _compute_energy,
    _compute_energy_hiddens,
    _compute_energy_visibles,
    _compute_gradient,
    _init_chains,
    _init_parameters,
    _sample_hiddens,
    _sample_visibles,
)

from typing import List, Optional, Self
import torch.nn.functional as F


import numpy as np
import torch
from torch import Tensor

class BBBM(RBM):
    """Parameters of the Bernoulli-Bernoulli RBM"""

    def __init__(
        self,
        weight_matrix: Tensor,
        vbias: Tensor,
        hbias: Tensor,
        K1: Tensor,
        K2: Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize the parameters of the Bernoulli-Bernoulli RBM.

        Args:
            weight_matrix (Tensor): The weight matrix of the RBM.
            vbias (Tensor): The visible bias of the RBM.
            hbias (Tensor): The hidden bias of the RBM.
            device (Optional[torch.device], optional): The device for the parameters.
                Defaults to the device of `weight_matrix`.
            dtype (Optional[torch.dtype], optional): The data type for the parameters.
                Defaults to the data type of `weight_matrix`.
        """
        if device is None:
            device = weight_matrix.device
        if dtype is None:
            dtype = weight_matrix.dtype
        self.device = device
        self.dtype = dtype
        self.weight_matrix = weight_matrix.to(device=self.device, dtype=self.dtype)
        self.w_norm_0 = torch.norm(weight_matrix)
        self.vbias = vbias.to(device=self.device, dtype=self.dtype)
        self.v_norm_0 = torch.norm(vbias)
        self.hbias = hbias.to(device=self.device, dtype=self.dtype)
        self.name = "BBBM"
        self.N = len(weight_matrix[0])
        self.K1 = K1.to(device=self.device, dtype=self.dtype)
        self.K2 = K2.to(device=self.device, dtype=self.dtype)
        self.K2_norm_0 = torch.norm(K2)
        self.mask = torch.ones_like(self.weight_matrix, device=self.device)  # Shape [N, N]
        self.mask.fill_diagonal_(0)  # Set diagonal to 0

    def __add__(self, other):
        return BBBM(
            weight_matrix=self.weight_matrix + other.weight_matrix,
            vbias=self.vbias + other.vbias,
            hbias=self.hbias + other.hbias,
        )

    def __mul__(self, other):
        return BBBM(
            weight_matrix=self.weight_matrix * other,
            vbias=self.vbias * other,
            hbias=self.hbias * other,
        )

    def clone(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        return BBBM(
            weight_matrix=self.weight_matrix.clone(),
            vbias=self.vbias.clone(),
            hbias=self.hbias.clone(),
            device=device,
            dtype=dtype,
        )

    def compute_energy(self, v: Tensor, h: Tensor) -> Tensor:
        return _compute_energy(
            v=v,
            vbias=self.vbias,
            weight_matrix=self.weight_matrix,
        )

    def compute_energy_hiddens(self, h: Tensor) -> Tensor:
        return _compute_energy_hiddens(
            h=h,
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
        )

    def compute_energy_visibles(self, v: Tensor) -> Tensor:
        return _compute_energy_visibles(
            v=v,
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
        )

    def compute_gradient(self, data, chains, use_fields, centered=True):
        _compute_gradient(
            v_data=data["visible"],
            v_chain=chains["visible"],
            weight_matrix=self.weight_matrix,
            vbias=self.vbias,
            use_fields=use_fields,
            centered=centered,
        )

    def independent_model(self):
        return BBBM(
            weight_matrix=torch.zeros_like(self.weight_matrix),
            vbias=self.vbias,
            hbias=torch.zeros_like(self.hbias),
        )

    def init_chains(self, num_samples, weights=None, start_v=None):
        visible = _init_chains(
            num_samples=num_samples,
            weight_matrix=self.weight_matrix,
            hbias=self.hbias,
            start_v=start_v,
        )
        if weights is None:
            weights = torch.ones(
                visible.shape[0], device=visible.device, dtype=visible.dtype
            )
        return dict(
            visible=visible,
            hidden=visible,
            visible_mag=visible,
            hidden_mag=visible,
            weights=weights,
        )

    @staticmethod
    def init_parameters(num_hiddens, dataset, device, dtype, beta, use_fields, var_init=0.1):
        data = dataset.data
        # Convert to torch Tensor if necessary
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(dataset.data).to(device=device, dtype=dtype)
        weight_matrix, vbias = _init_parameters(
            data=data,
            device=device,
            dtype=dtype,
            var_init=var_init,
            beta=beta
        )
        num_visible = len(data[0,:])
        if use_fields==False:
            vbias = torch.zeros_like(weight_matrix[0], device=device, dtype=dtype)
        
        hbias = torch.zeros_like(weight_matrix[0], device=device, dtype=dtype)
        K1 = torch.randn_like(weight_matrix, device=device, dtype=dtype)/np.sqrt(float(num_visible))
        K2 = torch.randn_like(weight_matrix, device=device, dtype=dtype)/np.sqrt(float(num_visible))
        K1 = (K1.T+K1)/2
        K2 = (K2.T+K2)/2
        K1.fill_diagonal_(0.0) 
        K2.fill_diagonal_(0.0)
        
        return BBBM(weight_matrix=weight_matrix, vbias=vbias, hbias=hbias, K1=K1, K2=K2)

    def named_parameters(self):
        return {
            "weight_matrix": self.weight_matrix,
            "vbias": self.vbias,
            "hbias": self.hbias,
            "K1": self.K1,
            "K2": self.K2,
        }

    def num_hiddens(self):
        return self.hbias.shape[0]

    def num_visibles(self):
        return self.weight_matrix.shape[0]

    def parameters(self) -> List[Tensor]:
        return [self.weight_matrix, self.vbias, self.hbias, self.K1, self.K2]

    def ref_log_z(self):
        return (
            self.num_visibles() * np.log(2)
        ).item()

    def sample_hiddens(self, chains: dict[str, Tensor], beta=1) -> dict[str, Tensor]:
        chains["hidden"], chains["hidden_mag"] = _sample_hiddens(
            v=chains["visible"],
            weight_matrix=self.weight_matrix,
            hbias=self.hbias,
            beta=beta,
        )
        return chains

    def sample_visibles(self, chains: dict[str, Tensor], beta=1) -> dict[str, Tensor]:
        chains["visible"], chains["visible_mag"] = _sample_visibles(
            v=chains["visible"],
            h=chains["hidden"],
            weight_matrix=self.weight_matrix,
            vbias=self.vbias,
            beta=beta,
        )
        return chains

    @staticmethod
    def set_named_parameters(named_params: dict[str, Tensor]) -> Self:
        names = ["vbias", "hbias", "weight_matrix", "K1", "K2"]
        for k in names:
            if k not in named_params.keys():
                raise ValueError(
                    f"""Dictionary params missing key '{k}'\n Provided keys : {named_params.keys()}\n Expected keys: {names}"""
                )
        params = BBBM(
            weight_matrix=named_params.pop("weight_matrix"),
            vbias=named_params.pop("vbias"),
            hbias=named_params.pop("hbias"),
            K1=named_params.pop("K1"),
            K2=named_params.pop("K2")
        )
        if len(named_params.keys()) > 0:
            raise ValueError(
                f"Too many keys in params dictionary. Remaining keys: {named_params.keys()}"
            )
        return params

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.weight_matrix = self.weight_matrix.to(device=self.device, dtype=self.dtype)
        self.vbias = self.vbias.to(device=self.device, dtype=self.dtype)
        self.hbias = self.hbias.to(device=self.device, dtype=self.dtype)
        return self
    
    def Z_i_mu_func(self, h, l):
        Z_i_mu = 2*torch.cosh(l*h) 
        return Z_i_mu
    
    def compute_pseudolikelihood_matrix(self, matrix, data, l):
        h = torch.einsum('ij,mj->mi', matrix*self.mask, data)
        x_J_x = torch.einsum('mi,mi->mi', data, h)
        energy_i_mu = -x_J_x + (1 / l) * torch.log(self.Z_i_mu_func(h,l)+1e-9)
        PL = energy_i_mu.mean()
        return PL
    
    def compute_loss_PL1(self, data, l, use_fields, use_hfield=False):
        x = data                              # [M,N] each entry ∈{0,1}
        h = torch.einsum('ij,mj->mi', self.K1*self.mask.to(self.K1.device), x)   # local field Σ_j K_ij x_j
        if use_fields:                        # optional visible bias
            h = h + self.vbias.unsqueeze(0)

        logZ = F.softplus(l*h)                # log(1+e^{λ h}) – numerically stable
        e_i  = -x*h + (1./l)*logZ             # −log P(x_i|x_{¬i}) / λ
        return e_i.mean()                     # scalar loss


    # ---------- second–order pseudolikelihood (pairwise) ----------
    def compute_loss_PL2(self, data, l, use_fields, use_hfield=False):
        x  = data                             # [M,N]
        K  = self.K2*self.mask                # interaction matrix with zero diagonal
    
        # global fields for every unit
        h  = torch.einsum('ik,mk->mi', K, x)                       # [M,N]
        diff_term = torch.einsum('ik,mk->mik', K, x)               # [M,N,N]
    
        if use_fields:
            h = h + self.vbias.unsqueeze(0)                        # add biases only once
    
        # “leave-one-out’’ effective fields
        h_i_eff = h.unsqueeze(2) - diff_term                       # h_i − K_ij x_j   [M,N,N]
        h_j_eff = h.unsqueeze(1) - diff_term                       # h_j − K_ij x_i   [M,N,N]
    
        x_i = x.unsqueeze(2)                                       # x_i            [M,N,1]
        x_j = x.unsqueeze(1)                                       # x_j            [M,1,N]
    
        # energy part actually observed: −(K_ij x_i x_j + h_i_eff x_i + h_j_eff x_j)
        E_pair  = K * x_i * x_j
        E_field = h_i_eff * x_i + h_j_eff * x_j
    
        # partition function for the {0,1}×{0,1} pair
        Z_xx = (
            1.0                                                     # (0,0)
            + torch.exp(l * h_i_eff)                                # (1,0)
            + torch.exp(l * h_j_eff)                                # (0,1)
            + torch.exp(l * (K + h_i_eff + h_j_eff))                # (1,1)
        )
    
        e_ij = -E_pair - E_field + (1./l)*torch.log(Z_xx + 1e-9)    # −log P(x_i,x_j|rest)/λ
        return e_ij.mean()

        
    def normalize_w(self):
        with torch.no_grad():
            norm = torch.norm(self.weight_matrix.data)
            self.weight_matrix.data = self.weight_matrix.data * self.w_norm_0 / (norm+1e-9)
            
    def normalize_K2(self):
        with torch.no_grad():
            norm = torch.norm(self.K2.data)
            self.K2.data = self.K2.data * self.K2_norm_0 / (norm+1e-9)
            
    def normalize_v(self):
        with torch.no_grad():
            norm = torch.norm(self.vbias.data)
            self.vbias.data = self.vbias.data * self.v_norm_0 / (norm+1e-9)
    
    
    
        