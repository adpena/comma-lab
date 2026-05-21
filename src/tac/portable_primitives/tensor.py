# SPDX-License-Identifier: MIT
"""PortableTensor wrapper for cross-backend numerical conversion.

The wrapper is intentionally THIN — callers operate on backend-native
tensors most of the time; the wrapper exists to (a) carry the backend tag
so misroutings raise loudly, (b) provide canonical to_numpy / from_numpy
conversion for the MLX -> PyTorch export pipeline + tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.portable_primitives.backend import Backend, resolve_backend

__all__ = [
    "PortableTensor",
    "to_numpy",
    "from_numpy",
]


@dataclass
class PortableTensor:
    """Backend-tagged tensor wrapper.

    The ``data`` field is a backend-native tensor:
    - For ``backend=Backend.MLX``: ``mlx.core.array``
    - For ``backend=Backend.PYTORCH``: ``torch.Tensor``

    The wrapper does NOT auto-convert on operations; callers route through
    backend-specific code paths via the primitives in :mod:`.nn`. Use
    :func:`to_numpy` for backend-agnostic readout (tests, export pipeline).
    """

    data: Any
    backend: Backend

    def __post_init__(self) -> None:
        # Resolve to make sure backend is actually available.
        self.backend = resolve_backend(self.backend)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def dtype(self) -> str:
        return str(self.data.dtype)


def to_numpy(tensor: Any, backend: Backend | str | None = None) -> np.ndarray:
    """Convert a backend-native tensor to numpy (always materialized).

    Supports:
    - :class:`PortableTensor` (backend inferred from wrapper)
    - ``mlx.core.array`` (auto-detected if backend not specified)
    - ``torch.Tensor`` (auto-detected if backend not specified)
    - already-numpy arrays (no-op)
    """

    if isinstance(tensor, PortableTensor):
        return to_numpy(tensor.data, tensor.backend)

    if isinstance(tensor, np.ndarray):
        return tensor

    if backend is not None:
        kind = resolve_backend(backend)
        if kind is Backend.MLX:
            import mlx.core as mx

            mx.eval(tensor)  # materialize lazy graph
            return np.array(tensor)
        if kind is Backend.PYTORCH:
            import torch

            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().numpy()

    # Auto-detect.
    try:
        import torch

        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
    except ImportError:
        pass

    try:
        import mlx.core as mx

        if isinstance(tensor, mx.array):
            mx.eval(tensor)
            return np.array(tensor)
    except ImportError:
        pass

    raise TypeError(f"cannot convert {type(tensor).__name__} to numpy")


def from_numpy(arr: np.ndarray, backend: Backend | str) -> Any:
    """Convert a numpy array to a backend-native tensor."""

    kind = resolve_backend(backend)
    if kind is Backend.MLX:
        import mlx.core as mx

        return mx.array(arr)
    if kind is Backend.PYTORCH:
        import torch

        return torch.from_numpy(arr.copy())  # copy to detach numpy ownership
    raise ValueError(f"unsupported backend: {kind}")
