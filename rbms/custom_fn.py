import h5py
import numpy as np
import torch
from torch import Tensor


def one_hot(
    x: Tensor, num_classes: int = -1, dtype: torch.dtype = torch.float32
) -> Tensor:
    """A one-hot encoding function faster than the PyTorch one working with torch.int32 and returning a float Tensor

    Args:
        x (Tensor): Input tensor.
        num_classes (int, optional): Number of classes for the one_hot encoding. If negative, then the number of classes is automatically detected.
        dtype (torch.dtype, optional): Dtype of the returned tensor. Defaults to torch.float32

    Returns
        Tensor: One-hot encoded version of the input tensor.
    """
    if num_classes < 0:
        num_classes = int(x.max().item()) + 1
    res = torch.zeros(x.shape[0], x.shape[1], num_classes, device=x.device, dtype=dtype)
    tmp = torch.meshgrid(
        torch.arange(x.shape[0], device=x.device),
        torch.arange(x.shape[1], device=x.device),
        indexing="ij",
    )
    index = (tmp[0], tmp[1], x)
    values = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=dtype)
    res.index_put_(index, values)
    return res


def log2cosh(x: Tensor) -> Tensor:
    """Numerically stable version of log(2*cosh(x)).

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor.
    """
    return torch.abs(x) + torch.log1p(torch.exp(-2 * torch.abs(x)))


def check_keys_dict(d: dict, names: list[str]):
    for k in names:
        if k not in d.keys():
            raise ValueError(
                f"""Dictionary params missing key '{k}'\n Provided keys : {d.keys()}\n Expected keys: {names}"""
            )


def load_string(f: h5py.Dataset, k: str | bytes) -> str:
    # Fix 1: Ensure key is a string
    # key = k.decode("utf-8") if isinstance(k, bytes) else k
    val = np.asarray(f[k])
    # Fix 2: Ensure string values (like 'Reservoir') are strings, not bytes
    if val.dtype.kind in ["S", "V", "O"]:  # Bytes, Void, or Object (StringDType)
        val = val.astype(str)
    return str(val)
