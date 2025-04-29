from typing import List, Optional, Self

import numpy as np
import torch
from torch import Tensor

from rbms.ising.implement import (
    _compute_energy,
    _compute_energy_hiddens,
    _compute_energy_visibles,
    _compute_gradient,
    _init_chains,
    _init_parameters,
    _sample_hiddens,
    _sample_visibles,
)
from rbms.classes import RBM


class IsingRBM(RBM):
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
        self.K1 = K1.to(device=self.device, dtype=self.dtype)
        self.K2 = K2.to(device=self.device, dtype=self.dtype)
        self.K2_norm_0 = torch.norm(K2)
        self.name = "IsingRBM"

    def __add__(self, other):
        return IsingRBM(
            weight_matrix=self.weight_matrix + other.weight_matrix,
            vbias=self.vbias + other.vbias,
            hbias=self.hbias + other.hbias,
        )

    def __mul__(self, other):
        return IsingRBM(
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
        return IsingRBM(
            weight_matrix=self.weight_matrix.clone(),
            vbias=self.vbias.clone(),
            hbias=self.hbias.clone(),
            device=device,
            dtype=dtype,
        )

    def compute_energy(self, v: Tensor, h: Tensor) -> Tensor:
        return _compute_energy(
            v=v,
            h=h,
            vbias=self.vbias,
            hbias=self.hbias,
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
            mh_data=data["hidden_mag"],
            w_data=data["weights"],
            v_chain=chains["visible"],
            h_chain=chains["hidden"],
            w_chain=chains["weights"],
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
            centered=False
        )

    def independent_model(self):
        return IsingRBM(
            weight_matrix=torch.zeros_like(self.weight_matrix),
            vbias=self.vbias,
            hbias=torch.zeros_like(self.hbias),
        )

    def init_chains(self, num_samples, weights=None, start_v=None):
        visible, hidden, mean_visible, mean_hidden = _init_chains(
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
            hidden=hidden,
            visible_mag=mean_visible,
            hidden_mag=mean_hidden,
            weights=weights,
        )

    @staticmethod
    def init_parameters(num_hiddens, dataset, device, dtype, beta, use_fields, var_init=0.0001):
        data = dataset.data
        # Convert to torch Tensor if necessary
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(dataset.data).to(device=device, dtype=dtype)
        vbias, hbias, weight_matrix = _init_parameters(
            num_hiddens=num_hiddens,
            data=data,
            device=device,
            dtype=dtype,
            var_init=var_init,
            beta=beta
        )
        num_visible = len(data[0,:])
        K1 = torch.randn_like(weight_matrix, device=device, dtype=dtype)/np.sqrt(float(num_hiddens))
        K2 = torch.randn_like(weight_matrix, device=device, dtype=dtype)/np.sqrt(float(num_hiddens))

        return IsingRBM(weight_matrix=weight_matrix, vbias=vbias, hbias=hbias, K1=K1, K2=K2)

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
        return self.vbias.shape[0]

    def parameters(self) -> List[Tensor]:
        return [self.weight_matrix, self.vbias, self.hbias, self.K1, self.K2]

    def ref_log_z(self):
        return (
            torch.log1p(torch.exp(self.vbias)).sum() + self.num_hiddens() * np.log(2)
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
        params = IsingRBM(
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
        self.K1 = self.K1.to(device=self.device, dtype=self.dtype)
        self.K2 = self.K2.to(device=self.device, dtype=self.dtype)
        return self
    
    def compute_loss_PL1(self, data, l, use_fields=True, use_hfield=True):
        x = data  # [M, N]

        with torch.no_grad():                                                
            h_pre = torch.einsum("ia,mi->ma", self.K1, x)                    
            if use_hfield:                                                   
                h_pre = h_pre + self.hbias
            h = torch.tanh(l * h_pre)                                        

        F = torch.einsum("ja,ma->mj", self.K1, h)                            
        if use_fields:
            F = F + self.vbias                                              
        
        xF  = torch.einsum("mi,mi->mi", x, F)                               
        Z_i = 2*torch.cosh(l * F)                                       
        e_i = -xF + (1.0 / l) * torch.log(Z_i + 1e-9)                    

        return e_i.mean()   

    def compute_loss_PL2(self, data, l, use_fields, use_hfield):
        x = data
        with torch.no_grad():
            if use_fields == True and use_hfield ==True:   
                h = torch.tanh(l*(self.hbias+torch.einsum("ia,mi->ma", self.K2, x)))
            else:
                h = torch.tanh(l*(torch.einsum("ia,mi->ma", self.K2, x)))
        b = torch.einsum("ja,ma->mj",self.K2, h)
        c = torch.einsum("ja,mj->ma",self.K2, x)
        
        if use_fields == True:
            b = b+self.vbias
            if use_hfield==True:
                c = c+self.hbias
            
        j_term = torch.einsum("ja,ma->mja", self.K2, h)
        a_term = torch.einsum("ja,mj->mja", self.K2, x)

        b_i_eff = b.unsqueeze(2)-j_term   #[M,N,1]
        c_a_eff = c.unsqueeze(1)-a_term    #[M,1,N]
        
        w_ai = torch.einsum("ma,ja,mj->mja", h, self.K2, x)
        h_ai = b_i_eff*x.unsqueeze(2)+c_a_eff*h.unsqueeze(1)
        Z_ai = 2.*(torch.exp(l*self.K2)*torch.cosh(l*b_i_eff+l*c_a_eff)+torch.exp(-l*self.K2)*torch.cosh(l*b_i_eff-l*c_a_eff))

        e_ij = -w_ai-h_ai+1./l*torch.log(Z_ai+1e-9)
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