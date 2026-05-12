"""Reusable loss-function primitives for tac trainers.

This subpackage holds substrate-independent training losses that are
documented, tested, and tagged with their `target_substrate_hint` for
downstream wiring discipline.
"""
from __future__ import annotations

from tac.losses.cat_entropy_v2 import (
    CatEntropyV2Config,
    cat_entropy_v2,
)

__all__ = [
    "CatEntropyV2Config",
    "cat_entropy_v2",
]
