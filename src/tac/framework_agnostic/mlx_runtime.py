# SPDX-License-Identifier: MIT
"""Canonical MLX runtime acquisition helpers.

MLX is a first-class local training substrate in this repo, but direct
``import mlx.*`` blocks scattered across substrates create three recurring
problems: inconsistent failure messages, inconsistent optional-import behavior,
and audit noise that hides the real unique math in each method. This module is
the small shared boundary for MLX module acquisition and forced evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tac.framework_agnostic.backend import BackendUnavailableError


@dataclass(frozen=True)
class MlxRuntime:
    """Imported MLX module bundle requested by a caller."""

    mx: Any
    nn: Any | None = None
    optimizers: Any | None = None
    utils: Any | None = None


def _mlx_import_error(component: str, exc: BaseException) -> BackendUnavailableError:
    return BackendUnavailableError(
        f"MLX {component} is unavailable. Install `mlx` on macOS Apple Silicon "
        "or route through the numpy/PyTorch portability path for this substrate."
    )


def require_mlx_core() -> Any:
    """Return ``mlx.core`` or fail closed with a canonical message."""
    try:
        import mlx.core as mx
    except Exception as exc:  # pragma: no cover - platform dependent
        raise _mlx_import_error("core", exc) from exc
    return mx


def require_mlx_nn() -> Any:
    """Return ``mlx.nn`` or fail closed with a canonical message."""
    try:
        import mlx.nn as nn
    except Exception as exc:  # pragma: no cover - platform dependent
        raise _mlx_import_error("nn", exc) from exc
    return nn


def require_mlx_optimizers() -> Any:
    """Return ``mlx.optimizers`` or fail closed with a canonical message."""
    try:
        import mlx.optimizers as optimizers
    except Exception as exc:  # pragma: no cover - platform dependent
        raise _mlx_import_error("optimizers", exc) from exc
    return optimizers


def require_mlx_utils() -> Any:
    """Return ``mlx.utils`` or fail closed with a canonical message."""
    try:
        import mlx.utils as utils
    except Exception as exc:  # pragma: no cover - platform dependent
        raise _mlx_import_error("utils", exc) from exc
    return utils


def require_mlx_runtime(
    *,
    nn: bool = False,
    optimizers: bool = False,
    utils: bool = False,
) -> MlxRuntime:
    """Return a requested MLX module bundle."""
    return MlxRuntime(
        mx=require_mlx_core(),
        nn=require_mlx_nn() if nn else None,
        optimizers=require_mlx_optimizers() if optimizers else None,
        utils=require_mlx_utils() if utils else None,
    )


def optional_mlx_runtime(
    *,
    nn: bool = False,
    optimizers: bool = False,
    utils: bool = False,
) -> MlxRuntime | None:
    """Return a requested MLX bundle, or ``None`` on non-MLX hosts."""
    try:
        return require_mlx_runtime(nn=nn, optimizers=optimizers, utils=utils)
    except BackendUnavailableError:
        return None


def is_mlx_runtime_available() -> bool:
    """Return whether ``mlx.core`` can be imported on this host."""
    return optional_mlx_runtime() is not None


def mlx_eval(*values: Any) -> None:
    """Canonical wrapper for ``mx.eval`` used by MLX training loops."""
    mx = require_mlx_core()
    mx.eval(*values)


def mlx_array(value: Any, *, dtype: Any | None = None) -> Any:
    """Canonical wrapper for ``mx.array`` with optional dtype."""
    mx = require_mlx_core()
    if dtype is None:
        return mx.array(value)
    return mx.array(value, dtype=dtype)


def mlx_compile(fn: Any | None = None, **kwargs: Any) -> Any:
    """Canonical wrapper/decorator for ``mx.compile``."""
    mx = require_mlx_core()
    if fn is None:
        return lambda inner: mx.compile(inner, **kwargs)
    return mx.compile(fn, **kwargs)


__all__ = [
    "MlxRuntime",
    "is_mlx_runtime_available",
    "mlx_array",
    "mlx_compile",
    "mlx_eval",
    "optional_mlx_runtime",
    "require_mlx_core",
    "require_mlx_nn",
    "require_mlx_optimizers",
    "require_mlx_runtime",
    "require_mlx_utils",
]
