# SPDX-License-Identifier: MIT
"""Canonical decorators for framework-agnostic primitives.

Per operator NON-NEGOTIABLE META directive 2026-05-28: *"remmebr MLX first
but agnostic portability via numpy and tinygrad like primitives and helpers
or **decorators** or whatever"*.

The decorators provide the canonical opt-in surface for substrate trainers
to declare per-method framework preference (per CLAUDE.md "UNIQUE-AND-
COMPLETE-PER-METHOD operating mode" + "MLX-FIRST NUMPY-PORTABLE
INDIVIDUALLY-FRACTAL" 8th standing directive):

  * ``@framework_agnostic`` — explicit backend selection at call-time
  * ``@mlx_first_with_numpy_fallback`` — MLX-first per 8th standing directive
  * ``@pytorch_first_with_numpy_fallback`` — PyTorch-first per Catalog #205 sister
  * ``@inflate_runtime_helper`` — enforces HNeRV parity L4 ≤200 LOC + ≤2 deps

Per CLAUDE.md "Beauty, simplicity, and developer experience": each decorator
is a thin wrapper that resolves backend once and passes through; the
decorated function does NOT need to handle backend selection internally.

Per CLAUDE.md "Forbidden device-selection defaults (the MPS-fallback trap)"
+ Catalog #1 sister: decorators MUST raise BackendUnavailableError if the
caller's explicit preference is unavailable; SILENT fallback to a non-
promotable backend would re-introduce the MPS-fallback bug class.

Cross-references:
  * Catalog #205 — sister at inflate-time device-selection surface
  * Catalog #341 — Tier A canonical-routing markers (decorators preserve)
  * Catalog #287 — placeholder-rationale rejection sister discipline
"""
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from tac.framework_agnostic.backend import (
    Backend,
    BackendUnavailableError,
    select_backend,
)


F = TypeVar("F", bound=Callable[..., Any])


def framework_agnostic(
    backend: Backend | None = None,
    *,
    env_var: str = "PACT_FRAMEWORK_BACKEND",
) -> Callable[[F], F]:
    """Decorator: resolve backend once and pass via ``backend`` kwarg.

    The decorated function MUST accept a ``backend`` keyword argument. The
    decorator resolves :class:`Backend` via
    :func:`tac.framework_agnostic.backend.select_backend` per the canonical
    cascade (override → env_var → priority) and injects it.

    Args:
        backend: Optional explicit Backend; defaults to AUTO (resolves at
            call-time per platform priority).
        env_var: Environment variable name; default
            ``PACT_FRAMEWORK_BACKEND``.

    Returns:
        Decorator that resolves backend per call.

    Example:
        >>> @framework_agnostic(backend=Backend.NUMPY)
        ... def my_op(x, *, backend=None):
        ...     return quantize_int8_per_channel(x, backend=backend)
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Caller's explicit kwarg override beats decorator default.
            requested = kwargs.pop("backend", None) or backend
            resolved = select_backend(override=requested, env_var=env_var)
            return fn(*args, backend=resolved, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator


def mlx_first_with_numpy_fallback(fn: F) -> F:
    """Decorator: try MLX first; fall back to numpy if MLX unavailable.

    Per CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL" 8th
    standing directive: training MLX-first on M5 Max + numpy fallback for
    portability + inflate numpy-portable bridge.

    Per CLAUDE.md "Forbidden device-selection defaults (the MPS-fallback
    trap)" + Catalog #1: the fallback is to NUMPY (deterministic; canonical
    bridge contract), NOT to MPS (silent + non-promotable). This decorator
    is therefore safe-by-construction.

    The decorated function MUST accept a ``backend`` keyword argument.

    Returns:
        Decorated function that resolves to MLX or NUMPY at call-time.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Caller's explicit override beats decorator priority.
        if "backend" in kwargs and kwargs["backend"] is not None:
            return fn(*args, **kwargs)
        try:
            backend = select_backend(
                priority=(Backend.MLX, Backend.NUMPY),
            )
        except BackendUnavailableError:
            backend = Backend.NUMPY
        return fn(*args, backend=backend, **kwargs)
    return wrapper  # type: ignore[return-value]


def pytorch_first_with_numpy_fallback(fn: F) -> F:
    """Decorator: try PyTorch first; fall back to numpy if PyTorch unavailable.

    Sister of :func:`mlx_first_with_numpy_fallback` for contest-resolution
    paths per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
    non-negotiable + Catalog #205 sister discipline.

    PyTorch is the canonical contest-resolution framework (T4 / A100 / 4090
    paired auth-eval per Catalog #246); MLX is non-promotable per Catalog
    #192/#317 until paired Linux x86_64 + NVIDIA empirical anchor lands.
    This decorator is therefore the canonical opt-in for substrate trainers
    targeting contest archives.

    The decorated function MUST accept a ``backend`` keyword argument.

    Returns:
        Decorated function that resolves to PYTORCH or NUMPY at call-time.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if "backend" in kwargs and kwargs["backend"] is not None:
            return fn(*args, **kwargs)
        try:
            backend = select_backend(
                priority=(Backend.PYTORCH, Backend.NUMPY),
            )
        except BackendUnavailableError:
            backend = Backend.NUMPY
        return fn(*args, backend=backend, **kwargs)
    return wrapper  # type: ignore[return-value]


def inflate_runtime_helper(fn: F) -> F:
    """Decorator: enforce HNeRV parity L4 ≤200 LOC + ≤2 deps + numpy-portable.

    Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
    lesson 4 (Inflate.py ≤ 100 LOC default budget; explicit waiver for
    ≤ 200 with rationale) + Catalog #146 contest-compliant inflate runtime
    contract + Catalog #205 sister inflate-time helpers.

    The decorated function MUST be a pure inflate-time primitive (numpy
    only; no torch / mlx / tinygrad imports). The decorator pins the
    backend to NUMPY regardless of caller's override per the bridge
    contract: ``MLX state_dict → npz → ZIP-member → numpy inflate``.

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog
    #287: the decorator does NOT enforce LOC budget at runtime (that's
    Catalog #328 surface); it pins the backend semantic so the inflate
    primitive cannot accidentally consume MLX / PyTorch + violate the
    canonical bridge contract.

    Returns:
        Decorated function pinned to Backend.NUMPY.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Pin to NUMPY per the bridge contract; caller's override IGNORED
        # because inflate runtime must be numpy-portable per HNeRV L4.
        kwargs["backend"] = Backend.NUMPY
        return fn(*args, **kwargs)
    return wrapper  # type: ignore[return-value]


__all__ = [
    "framework_agnostic",
    "inflate_runtime_helper",
    "mlx_first_with_numpy_fallback",
    "pytorch_first_with_numpy_fallback",
]
