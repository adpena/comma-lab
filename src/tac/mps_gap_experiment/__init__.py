# SPDX-License-Identifier: MIT
"""MPS-train CUDA-score gap experiment infrastructure.

DIAGNOSTIC PACKAGE — research-only. Empirically answers the question:
**Do MPS-trained weights survive CUDA scoring on REAL CONTEST FRAMES?**

NOT a contest substrate. NO archive grammar. NO score claims. Every artifact
emitted by this package is tagged ``[MPS-research-signal]`` or
``[diagnostic-CUDA Modal A10G]`` per CLAUDE.md "MPS auth eval is NOISE" +
Catalog #1 + Catalog #192 + Catalog #317.

Empirical context (per `lane_mps_conv2d_wrap_fix_plus_real_frame_gap_experiment_20260518`
commit ``24278cf06``): the Conv2d-cliff wrapper hypothesis was empirically
falsified (per-layer drift 2.205e-3 with vs without wrapper) and the SegNet
end-to-end output drift on synthetic noise was 7.6e-5 — already below the
1e-3 cumulative threshold the Phase B gate was conditional on. The wrapper is
unnecessary. THE remaining open question this package answers: does a real
*trained* MPS model produce a checkpoint whose CUDA forward components agree
with the MPS forward components within a usable tolerance?

Verdict thresholds (per the landing memo):

* ``gap_relative_aggregate < 5%`` → Local-MPS-train viable for substrate
  training; recommend Catalog #317 scope-narrowing for MPS opt-in.
* ``gap_relative_aggregate 5-20%`` → Local-MPS-train viable for advisory
  ranking; not promotion-grade.
* ``gap_relative_aggregate > 20%`` → Local-MPS-train NOT viable; pivot to
  MLX or VideoToolbox-decode + CUDA-train.

Public surfaces (narrow):

* :class:`TinyRenderer` — ~10K param FiLM-conditioned RGB renderer for
  reconstructing 2-frame pair from a pose vector.
* :func:`train_on_mps_real_frames` — local MPS training loop (canonical
  scorer-loss + EMA(0.997) + eval_roundtrip=True).
* :func:`compute_gap_components` — re-runs the trained checkpoint on a
  target device and emits the canonical gap manifest.

Sister of :mod:`tac.mps_diagnostic.layerwise_drift` (the layerwise diagnostic
that already established the per-layer drift) and
:mod:`tac.optimization.mps_research_signal` (the canonical advisory-manifest
emitter). This package is the END-TO-END counterpart: does the drift survive
a full training loop on real frames?
"""

from __future__ import annotations

from .tiny_renderer import TinyRenderer, build_tiny_renderer, count_params
from .train_on_mps import (
    TrainingMetrics,
    train_on_mps_real_frames,
)
from .harvest_and_verdict import (
    ComponentGap,
    GapManifest,
    classify_verdict,
    compute_gap_components,
    compute_local_mps_reference_components,
    compute_target_cuda_components,
    diff_components_and_classify_verdict,
)

__all__ = (
    "TinyRenderer",
    "build_tiny_renderer",
    "count_params",
    "TrainingMetrics",
    "train_on_mps_real_frames",
    "ComponentGap",
    "GapManifest",
    "classify_verdict",
    "compute_gap_components",
    "compute_local_mps_reference_components",
    "compute_target_cuda_components",
    "diff_components_and_classify_verdict",
)
