import rbms
from rbms.classes import RBM

from rbms.bm.implement import (
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

import numpy as np
import torch
from torch import Tensor

class BM(RBM):
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
        self.name = "BM"
        self.N = len(weight_matrix[0])
        self.K1 = K1.to(device=self.device, dtype=self.dtype)
        self.K2 = K2.to(device=self.device, dtype=self.dtype)
        self.K2_norm_0 = torch.norm(K2)
        self.mask = torch.ones_like(self.weight_matrix, device=self.device)  # Shape [N, N]
        self.mask.fill_diagonal_(0)  # Set diagonal to 0

    def __add__(self, other):
        return BM(
            weight_matrix=self.weight_matrix + other.weight_matrix,
            vbias=self.vbias + other.vbias,
            hbias=self.hbias + other.hbias,
        )

    def __mul__(self, other):
        return BM(
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
        return BM(
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
        return BM(
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
        
        return BM(weight_matrix=weight_matrix, vbias=vbias, hbias=hbias, K1=K1, K2=K2)

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
        params = BM(
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
    
    '''
    def compute_pseudolikelihood_K1(self, data, l):
        h = torch.einsum('ij,mj->mi', self.K_matrix*self.mask, data)
        x_J_x = torch.einsum('mi,mi->mi', data, h)
        energy_i_mu = -x_J_x + (1 / l) * torch.log(self.Z_i_mu_func(h,l))
        PL = energy_i_mu.mean()
        return PL
    '''
    def compute_gradient_PL1(self, data):
        pass
    
    '''
    def compute_gradient_PL2(self, data, l):
        x = data["visible"]
        h = torch.einsum("ij,mj->mi",self.K2*self.mask, x)
        print("h", h.mean())
        
        #h_a = h.unsqueeze(2)+h.unsqueeze(1)
        #h_b = h.unsqueeze(2)-h.unsqueeze(1)
        #i_term = torch.einsum("ij,mi->mij", self.K2, x)
        #j_term = torch.einsum("ij,mj->mij", self.K2, x)
        #
        #h_a = h_a - i_term - j_term
        #h_b = h_b - i_term + j_term
        #
        #grad_term_1 = torch.exp(self.K2)*torch.cosh(h_a)
        #grad_term_2 = torch.exp(-self.K2)*torch.cosh(h_b)
        #
        #print("grad1", grad_term_1.mean())
        #print("grad2", grad_term_2.mean())

        #grad_K2 = ((grad_term_1-grad_term_2)/(grad_term_1+grad_term_2+1e-9)).mean(0)
        
        
        i_term = torch.einsum("ij,mi->mij", self.K2, x)
        j_term = torch.einsum("ij,mj->mij", self.K2, x)
        h_i_eff = h.unsqueeze(2)-i_term
        h_j_eff = h.unsqueeze(1)-j_term
        
        data_corr = (x.unsqueeze(2) * x.unsqueeze(1)).mean(dim=0)
        
        grad_K2 =  data_corr - torch.tanh(l*self.K2+0.5*torch.log(torch.cosh(h_i_eff+h_j_eff)+1e-9)-torch.log(torch.cosh(h_i_eff-h_j_eff)+1e-9)).mean(0)
        
        grad_K2 = (grad_K2+grad_K2.T)/2
        
        print(grad_K2.mean())
        
        grad_K2.fill_diagonal_(0.0)
        
        
        self.K2.grad.set_(grad_K2)
    '''    
    
    def comppute_loss_PL1(self, data, l, use_fields, use_hfield=False):
          # [M, N]
        x=data
        J_x = torch.einsum('ijab,mjb->mia', self.K1 * self.mask.to(self.K1.device), x)   # [M, d]
        y_i_mu = J_x.norm(dim=-1)  # Taking the norm over the last dimension -> [M,N]
        x_J_x = torch.einsum('mia,mia->mi', x, J_x)  # [M, N]
        Z_i_mu = 2*torch.cosh(l*y_i_mu)
        # Compute the energy term for each mu: - dot_product + lam^-1 * log(Z_i_mu)
                    # Compute the energy term for each mu: - dot_product + lam^-1 * log(Z_i_mu)
        e_i = -x_J_x + (1 / l) * torch.log(Z_i_mu+1e-9)  # [M,N]

        return e_i.mean()
    
    def compute_loss_PL2(self, data, l, use_fields, use_hfield=False):
        x = data#["visible"]
        h = torch.einsum("ik,mk->mi",self.K2*self.mask, x)
        diff_term = torch.einsum("ik,mk->mik", self.K2*self.mask, x)
        #j_term = torch.einsum("ik,mk->mi", self.K2*self.mask, x)
        if use_fields == True:
            #fields_x = torch.einsum("i,mi->mi", self.vbias, x)
            h = h+self.vbias.unsqueeze(0)
        h_i_eff = h.unsqueeze(2)-diff_term   #[M,N,1]
        h_j_eff = h.unsqueeze(1)-diff_term    #[M,1,N]
        
        J_xx = torch.einsum("mi,ij,mj->mij", x, self.K2*self.mask, x)
        h_xx = h_i_eff*x.unsqueeze(2)+h_j_eff*x.unsqueeze(1)
        Z_xx = 2.*(torch.exp(l*self.K2*self.mask)*torch.cosh(l*h_i_eff+l*h_j_eff)+torch.exp(-l*self.K2*self.mask)*torch.cosh(l*h_i_eff-l*h_j_eff))
        
        e_ij = -J_xx-h_xx+1./l*torch.log(Z_xx+(1-self.mask)+1e-9)
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
    
    
    
        