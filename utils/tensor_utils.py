"""Utilities for converting between torch tensors and numpy arrays.

Tensor contract (used throughout this codebase):
- All *public* functions accept and return np.ndarray.
- Internally, torch tensors are used for GPU-accelerated computation.
- Use to_torch() to enter torch space and to_numpy() to leave it.
"""
from typing import Any

import numpy as np
import torch


def to_numpy(x: Any) -> np.ndarray:
    """Convert any tensor/array-like to a pure numpy ndarray.

    Handles: numpy arrays (no-op), torch CPU/CUDA tensors,
    himalaya arrays, and any object implementing __array__.
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    # himalaya arrays and other array-likes: np.asarray goes through __array__,
    # then np.array forces a copy to break any reference to the underlying tensor.
    arr = np.asarray(x)
    return np.array(arr, dtype=arr.dtype)


def to_torch(
    x: Any,
    dtype: torch.dtype = torch.float32,
    device: str | None = None,
) -> torch.Tensor:
    """Convert a numpy array (or array-like) to a torch tensor.

    Always calls to_numpy() first, so passing an existing tensor is safe and
    avoids the 'copy construct from tensor' UserWarning.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.tensor(to_numpy(x), dtype=dtype, device=device)
