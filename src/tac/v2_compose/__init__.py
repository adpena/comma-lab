# SPDX-License-Identifier: MIT
"""tac.v2_compose — the v2 hybrid-witness composition layer (the ~30% missing CPU plumbing).

Composes already-MEASURED, already-BUILT components into the v2 hybrid witness:
deterministic STORE+GENERATE bulk (keyframes + per-class stratified warp) + a small LEARN
residual INR (the d_seg-floor closer) + a dual-use STORE pose sidecar -> a 4-section byte-close.

Pipeline spec: ``.omx/research/v2_coherent_automated_composition_pipeline_spec_20260630T201213Z.md``.
The single automated entry point is ``tools/compose_witness_archive.py`` (PHASE-A CPU/$0 +
PHASE-B byte-close), which delegates to this package (CLAUDE.md "tac stays clean").

NO-FAKE / means != ends: this is composition PLUMBING (a MEANS). The frontier pointer is UNMOVED
0.19110; it moves ONLY when the residual-INR GPU run lands + the 4-section archive byte-closes +
``upstream/evaluate.py`` (CPU + CUDA) returns below 0.19110. Every number here is advisory.

The KNOWN deep-math split (ENCODED, not discovered): STORE/GENERATE = pose-recoverable classes
(Road=ground-homography / hood=identity / sky=rotation-only) + keyframes + dual-use pose; LEARN =
the Lane-survival residual + small movables (the binding wall). See ``store_learn_split``.
"""

from __future__ import annotations

from tac.v2_compose.store_learn_split import (
    ClassAssignment,
    ClassRecoverability,
    StoreLearnAssignment,
    encode_known_split,
    load_reach_kstar,
    load_warp_recoverability_from_grok,
)

__all__ = [
    "ClassAssignment",
    "ClassRecoverability",
    "StoreLearnAssignment",
    "encode_known_split",
    "load_reach_kstar",
    "load_warp_recoverability_from_grok",
]
