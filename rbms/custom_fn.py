import h5py
import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import one_hot as one_hot


def one_hot_old(
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


def clone_dict(d: dict[str, Tensor]) -> dict[str, Tensor]:
    res = {}
    for k in d.keys():
        res[k] = d[k].clone()
    return res


# @torch.compile(fullgraph=True)
def swap_tensor(
    v1: Tensor, v2: Tensor, swap_mask: Tensor, swap_only_v1: bool = False
) -> tuple[Tensor, Tensor]:
    """
    Swap configurations between v_1 and v_2 on the first axis according to the boolean mask swap_mask

    Args:
        v1 (Tensor): shape (n, d)
        v2 (Tensor): shape (n, d)
        swap_mask (Tensor): shape (n, )

    Returns:
        tuple[Tensor, Tensor] v1, v2
    """
    swap_mask = swap_mask.view(-1, 1).repeat(1, v1.shape[1])
    if not (swap_only_v1):
        v_save = v1.clone()
        v1 = torch.where(swap_mask, v2, v_save)
        v2 = torch.where(swap_mask, v_save, v2)
    else:
        v1 = torch.where(swap_mask, v2, v1)
    return v1, v2
