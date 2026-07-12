# SPDX-License-Identifier: MIT
"""Runtime-checkable protocols for Lens Engine adapters and lenses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .core import T, TypedResult


@runtime_checkable
class Lens(Protocol):
    """A named, typed operation set over a :class:`~tac.lens_engine.core.T`."""

    name: str
    operations: frozenset[str]

    def apply(self, complex_: T, op: str, **args: Any) -> TypedResult[Any]:
        """Apply one supported operation to ``complex_``."""
        ...


@runtime_checkable
class ComplexAdapter(Protocol):
    """An authority-preserving source adapter that exposes its data as ``T``."""

    name: str

    def to_complex(self) -> T:
        """Return the validated typed-attributed-complex view."""
        ...
