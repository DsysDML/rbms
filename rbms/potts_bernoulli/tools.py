from typing import Optional

import numpy as np
import torch
from torch import Tensor

from rbms.custom_fn import one_hot


def get_covariance_matrix(
    data: Tensor,
    weights: Optional[Tensor] = None,
    num_extract: Optional[int] = None,
    center: bool = True,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Returns the covariance matrix of the data. If weights is specified, the weighted covariance matrix is computed.

    Args:
        data (Tensor): Data.
        weights (Tensor, optional): Weights of the data. Defaults to None.
        num_extract (int, optional): Number of data to extract to compute the covariance matrix. Defaults to None.
        center (bool): Center the data. Defaults to True.
        device (torch.device): Device. Defaults to 'cpu'.
        dtype (torch.dtype): DType. Defaults to torch.float32.

    Returns:
        Tensor: Covariance matrix of the dataset.
    """
    num_data = len(data)
    num_classes = int(data.max().item() + 1)

    if weights is None:
        weights = torch.ones(num_data)
    weights = weights.to(device=device, dtype=torch.float32)

    if num_extract is not None:
        idxs = np.random.choice(a=np.arange(num_data), size=(num_extract,), replace=False)
        data = data[idxs]
        weights = weights[idxs]
        num_data = num_extract

    if num_classes != 2:
        data = data.to(device=device, dtype=torch.int32)
        data_oh = one_hot(data, num_classes=num_classes).reshape(num_data, -1)
    else:
        data_oh = data.to(device=device, dtype=torch.float32)

    norm_weights = weights.reshape(-1, 1) / weights.sum()
    data_mean = (data_oh * norm_weights).sum(0, keepdim=True)
    cov_matrix = ((data_oh * norm_weights).mT @ data_oh) - int(center) * (
        data_mean.mT @ data_mean
    )
    return cov_matrix
