# SPDX-License-Identifier: MIT
"""Canonical FrameworkAgnosticTensor Protocol per Catalog #335 sister discipline.

Defines the minimum interface every backend's tensor must satisfy so the
canonical primitives in :mod:`tac.framework_agnostic.operations` can route
operations per :class:`tac.framework_agnostic.backend.Backend` without
backend-specific branching at every call site.

Per CLAUDE.md "Beauty, simplicity, and developer experience" + "tac stays
clean" non-negotiables: the Protocol is intentionally minimal so future
substrate trainers can adopt without per-substrate API surgery.

Per CLAUDE.md "Forbidden score claims": Tensor.as_numpy() preserves canonical
Provenance per Catalog #323 — backend-specific tensors are converted via
canonical paths so downstream consumers (autopilot ranker, sensitivity map,
bit-allocator) inherit axis_tag + hardware_substrate + evidence_grade without
silent provenance loss.

The Protocol is ``runtime_checkable`` so the auto-discovery loop + the
sister cathedral consumer can verify candidate tensors satisfy the contract
before threading them through canonical primitives.

Cross-references:
  * Catalog #335 — sister Protocol pattern at the cathedral consumer surface
  * Catalog #205 — sister at inflate-device-selection surface
  * Catalog #323 — canonical Provenance umbrella
  * tinygrad's `Tensor` class (~3000 LOC reference; this Protocol is the
    lighter-weight ~50 LOC contract)
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FrameworkAgnosticTensor(Protocol):
    """Minimum tensor interface every backend must satisfy.

    All canonical primitives in :mod:`tac.framework_agnostic.operations`
    accept any object satisfying this Protocol. Backend-specific tensor
    classes (``torch.Tensor`` / ``mx.array`` / ``numpy.ndarray`` /
    ``tinygrad.Tensor``) all satisfy structurally.

    For numpy arrays: ``shape`` returns ``tuple[int, ...]``; ``dtype``
    returns ``numpy.dtype``. For torch tensors: ``shape`` is
    ``torch.Size``; ``dtype`` is ``torch.dtype``. The Protocol intentionally
    does not constrain dtype/shape types so concrete backends can use their
    canonical types.

    Canonical conversion methods route through
    :func:`tac.framework_agnostic.helpers.coerce_to_backend` so per-backend
    tensor-class adaptation is centralized.
    """

    @property
    def shape(self) -> Sequence[int]: ...

    @property
    def dtype(self) -> Any: ...


def shape_of(tensor: Any) -> tuple[int, ...]:
    """Return canonical tuple shape regardless of backend tensor class.

    Per CLAUDE.md "Beauty, simplicity, and developer experience": shapes
    are reported as plain int tuples so downstream consumers (canonical
    Provenance, autopilot ranker, sensitivity map) don't need backend-
    specific introspection.

    Raises:
        TypeError: if tensor has no .shape attribute or shape is not iterable.
    """
    if not hasattr(tensor, "shape"):
        raise TypeError(
            f"shape_of expected FrameworkAgnosticTensor with .shape attribute, "
            f"got {type(tensor).__name__}"
        )
    raw = tensor.shape
    try:
        return tuple(int(d) for d in raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"shape_of expected iterable of int dimensions, got {raw!r} "
            f"on {type(tensor).__name__}"
        ) from exc


def dtype_name(tensor: Any) -> str:
    """Return canonical dtype name (str) regardless of backend tensor class.

    Canonical dtype names used across the framework_agnostic primitives:
    ``"float32"``, ``"float16"``, ``"int8"``, ``"int32"``, ``"uint8"``,
    ``"bool"``. Backend-specific dtypes (torch.float16 vs mlx.float16 vs
    numpy.float16) all reduce to ``"float16"``.

    Raises:
        TypeError: if tensor has no .dtype attribute.
    """
    if not hasattr(tensor, "dtype"):
        raise TypeError(
            f"dtype_name expected FrameworkAgnosticTensor with .dtype, "
            f"got {type(tensor).__name__}"
        )
    raw = tensor.dtype
    # torch.dtype / numpy.dtype / mlx.Dtype all str() to canonical names
    name = str(raw)
    # numpy reports as "<class 'numpy.float32'>" sometimes; normalize.
    for token in ("torch.", "numpy.", "mlx.", "mx.", "<class '", "'>"):
        name = name.replace(token, "")
    return name


__all__ = [
    "FrameworkAgnosticTensor",
    "dtype_name",
    "shape_of",
]
