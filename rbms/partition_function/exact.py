import torch
from torch import Tensor

from rbms.classes import RBM


def compute_partition_function(params: RBM, all_config: Tensor) -> float:
    """Compute the exact log partition function.

    Args:
        params (RBM): Parameters of the RBM.
        all_config (Tensor): Tensor containing the enumeration of all possible states of one of the layers.

    Returns:
        float: Exact log partition function.
    """
    n_dim_config = all_config.shape[1]
    n_visible, n_hidden = params.vbias.shape[0], params.hbias.shape[0]
    if n_dim_config == n_hidden:
        energy = params.compute_energy_hiddens(h=all_config)
    elif n_dim_config == n_visible:
        energy = params.compute_energy_visibles(v=all_config)
    else:
        raise ValueError(
            f"The number of dimension for the configurations '{n_dim_config}' does not match the number of visible '{n_visible}' or the number of hidden '{n_hidden}'"
        )
    return torch.logsumexp(-energy, 0).item()
