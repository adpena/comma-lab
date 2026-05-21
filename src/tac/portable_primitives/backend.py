# SPDX-License-Identifier: MIT
"""Backend enum + capability detection for portable primitives.

Per OVERNIGHT-WW: canonical backend resolution so callers don't reach into
import machinery. Mirrors :mod:`tac.local_acceleration.mlx_integration`
sister pattern at the portable-primitives surface.
"""

from __future__ import annotations

import enum
import importlib.util

__all__ = [
    "Backend",
    "BackendUnavailableError",
    "is_mlx_available",
    "is_pytorch_available",
    "resolve_backend",
]


class Backend(str, enum.Enum):
    """Canonical backend identifiers.

    MLX = Apple Silicon Metal GPU (non-promotable per Catalog #1 + #192).
    PYTORCH = canonical PyTorch (promotion-capable on CUDA Linux x86_64).
    """

    MLX = "mlx"
    PYTORCH = "pytorch"


class BackendUnavailableError(RuntimeError):
    """Raised when caller requests a backend that isn't importable on this host."""


def is_mlx_available() -> bool:
    """Check whether MLX framework is importable + Metal device available.

    Returns False on non-Apple-Silicon hosts, on hosts without MLX installed,
    or when Metal compute is unavailable. Sister of
    :func:`tac.local_acceleration.mlx_integration.is_mlx_available`.
    """

    if importlib.util.find_spec("mlx") is None:
        return False
    if importlib.util.find_spec("mlx.core") is None:
        return False
    try:
        import mlx.core as mx

        return bool(mx.metal.is_available())
    except (ImportError, AttributeError, RuntimeError):
        return False


def is_pytorch_available() -> bool:
    """Check whether PyTorch is importable."""

    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_backend(backend: str | Backend) -> Backend:
    """Resolve a backend identifier and verify it's available on this host.

    Raises :class:`BackendUnavailableError` if the requested backend isn't
    importable. Use :func:`is_mlx_available` / :func:`is_pytorch_available`
    to check availability without raising.
    """

    if isinstance(backend, Backend):
        kind = backend
    else:
        try:
            kind = Backend(backend.lower())
        except ValueError as exc:
            raise ValueError(
                f"unknown backend: {backend!r} (expected one of {[b.value for b in Backend]})"
            ) from exc

    if kind is Backend.MLX and not is_mlx_available():
        raise BackendUnavailableError(
            "MLX backend requested but mlx framework not importable or Metal "
            "device unavailable; install via `pip install mlx` on Apple Silicon "
            "host or pass backend='pytorch' for cross-platform fallback"
        )
    if kind is Backend.PYTORCH and not is_pytorch_available():
        raise BackendUnavailableError(
            "PyTorch backend requested but torch not importable; install via "
            "`pip install torch` for CPU/CUDA/MPS support"
        )
    return kind
