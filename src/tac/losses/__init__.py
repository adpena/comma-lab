"""Reusable loss-function primitives for tac trainers.

The historical single-file ``tac.losses`` implementation now lives in
``tac.losses.core`` so newer loss submodules can coexist without shadowing
legacy imports such as ``from tac.losses import scorer_loss``.
"""
from __future__ import annotations

from . import core as _core
from .core import *  # noqa: F403
from tac.losses.cat_entropy_v2 import (
    CatEntropyV2Config,
    cat_entropy_v2,
)

_core_exports = [name for name in dir(_core) if not name.startswith("__")]
for _name in _core_exports:
    if _name not in globals():
        globals()[_name] = getattr(_core, _name)

__all__ = sorted(set(_core_exports + [
    "CatEntropyV2Config",
    "cat_entropy_v2",
]))
