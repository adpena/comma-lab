# SPDX-License-Identifier: MIT
"""Canonical Tier-1 training-optimization helpers.

This package lands the three canonical helpers identified in
``.omx/research/optimization_opportunities_audit_20260514.md`` as Tier-1
(highest EV/$) opportunities:

* O1 — :mod:`tac.training_optimization.scorer_cache` — GT-scorer-output
  caching for ``score_pair_components``. Removes the GT scorer forward
  from every training step (the target video + scorer weights are
  invariant during training, so the GT forward is invariant too).
  Reference impl: ``experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py``.

* O2 — :mod:`tac.training_optimization.autocast_helper` — canonical
  ``torch.autocast`` context wrapper honoring substrate trainers'
  ``--enable-autocast-fp16`` argparse flag (Catalog #172). Backportable
  to every substrate trainer in a 5-LOC patch.

* O3 — :mod:`tac.training_optimization.compile_helper` — canonical
  ``torch.compile`` wrapper with graceful fallback honoring substrate
  trainers' ``--enable-torch-compile`` flag (Catalog #179).

These helpers do NOT change substrate score semantics. They are
opt-in via existing CLI flags. The cache helper is mathematically
identical to the un-cached path (target tensors are detached + frozen);
autocast and compile carry well-documented numerical drift bounded by
PyTorch literature.

Per CLAUDE.md "Apples-to-apples evidence discipline" — every speedup
estimate in module docstrings is tagged ``[derived]`` (first-principles),
``[literature-extrapolation]`` (PyTorch / HuggingFace benchmarks), or
``[would-need-empirical]``. No score claims live in these helpers.
"""

from __future__ import annotations

from tac.training_optimization.autocast_helper import (
    AutocastConfig,
    autocast_aware_forward,
    resolve_autocast_dtype,
)
from tac.training_optimization.compile_helper import (
    CompileConfig,
    compile_with_fallback,
)
from tac.training_optimization.scorer_cache import (
    GTScorerCache,
    GTScorerCacheError,
    build_gt_scorer_cache,
)

__all__ = [
    "AutocastConfig",
    "CompileConfig",
    "GTScorerCache",
    "GTScorerCacheError",
    "autocast_aware_forward",
    "build_gt_scorer_cache",
    "compile_with_fallback",
    "resolve_autocast_dtype",
]
