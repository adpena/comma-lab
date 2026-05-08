"""Scorer introspection and CUDA-vs-CPU drift diagnostics.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
HARDWARE" and "MPS auth eval is NOISE": this package provides observational
tooling to "light up" SegNet and PoseNet internals like neon-dye injection in
biology — see exactly which layers compute what, where precision is lost, and
where CUDA-vs-CPU diverge. NEVER produces a score claim. All numerical outputs
are tagged `[diagnostic-not-score]`.

Public API:
- `ScorerIntrospector` — observational forward-hook recorder.
- `IntrospectionRecord` — per-layer activation/attention dump container.
- `LayerStats` / `AttentionFingerprint` — fingerprint dataclasses.
- `compute_layer_drift` — CUDA-vs-CPU drift comparator.
- `DriftMetrics` — per-layer drift summary.
- `compounding_factor` — geometric (1+ε)^L compounding test.
"""

from __future__ import annotations

from .scorer_introspection import (
    AttentionFingerprint,
    IntrospectionRecord,
    LayerStats,
    ScorerIntrospector,
    fingerprint_tensor,
)
from .cuda_cpu_drift import (
    DriftMetrics,
    compounding_factor,
    compute_layer_drift,
)
from .decoder_drift_introspection import (
    DecoderDriftIntrospector,
    DriftReport,
    FrameByteFingerprint,
    lipschitz_pose_drift_prediction,
    quantify_drift,
)

__all__ = [
    "AttentionFingerprint",
    "DecoderDriftIntrospector",
    "DriftMetrics",
    "DriftReport",
    "FrameByteFingerprint",
    "IntrospectionRecord",
    "LayerStats",
    "ScorerIntrospector",
    "compounding_factor",
    "compute_layer_drift",
    "fingerprint_tensor",
    "lipschitz_pose_drift_prediction",
    "quantify_drift",
]
