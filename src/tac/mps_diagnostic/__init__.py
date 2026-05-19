# SPDX-License-Identifier: MIT
"""MPS local-compute frontier diagnostic package.

Canonical helpers for layerwise drift measurement across PyTorch backends
(MPS / CPU / CUDA), used to identify the mechanism behind the well-documented
23x/2x/2.5x drift on PoseNet/SegNet/score that CLAUDE.md "MPS auth eval is
NOISE" records empirically but does NOT explain mechanistically.

This package is DIAGNOSTIC infrastructure. Every artifact it produces is
non-promotable per CLAUDE.md "MPS auth eval is NOISE" non-negotiable + Catalog
#192 (`check_macos_cpu_advisory_not_promoted_without_linux_verification`):

  - evidence_grade = "macOS-MPS-diagnostic" (or "macOS-CPU-PyTorch-reference"
    for the bit-exact PyTorch CPU comparison axis)
  - score_claim = False
  - promotion_eligible = False
  - axis tag = "[macOS-MPS-PyTorch]" / "[macOS-CPU-PyTorch]" /
    "[advisory only]"

These markers are STRUCTURAL protection. They must NEVER be removed because
they prevent local-Mac measurements from leaking into the canonical
[contest-CUDA] / [contest-CPU] posterior.

Lane: lane_mps_local_compute_frontier_diagnostic_20260518 (Catalog #126).
Operator standing directive 2026-05-18: "leveraging local compute especially
is a top priority of all".

Public API:
    - measure_layerwise_drift(model, sample_input, backends, ...)
        Run identical input through each backend; capture per-module output
        via forward hooks; emit per-layer L_inf, L_2, mean-relative drift.
    - emit_drift_table_markdown(drift_data, output_path)
        Sort by depth; flag first-divergence-layer; emit markdown table.
    - identify_drift_cliff_layer(drift_data, threshold=1e-3)
        Return name of first layer whose backend-pair L_inf drift exceeds
        threshold.
    - DRIFT_EVIDENCE_GRADE = "macOS-MPS-diagnostic"
    - DRIFT_AXIS_TAG_MPS = "[macOS-MPS-PyTorch]"
    - DRIFT_AXIS_TAG_CPU = "[macOS-CPU-PyTorch]"
    - DRIFT_AXIS_TAG_CUDA = "[contest-CUDA-PyTorch-reference]"
"""

from tac.mps_diagnostic.layerwise_drift import (
    DRIFT_AXIS_TAG_CPU,
    DRIFT_AXIS_TAG_CUDA,
    DRIFT_AXIS_TAG_MPS,
    DRIFT_EVIDENCE_GRADE,
    LayerDriftRecord,
    emit_drift_table_markdown,
    identify_drift_cliff_layer,
    measure_layerwise_drift,
)

__all__ = [
    "DRIFT_AXIS_TAG_CPU",
    "DRIFT_AXIS_TAG_CUDA",
    "DRIFT_AXIS_TAG_MPS",
    "DRIFT_EVIDENCE_GRADE",
    "LayerDriftRecord",
    "emit_drift_table_markdown",
    "identify_drift_cliff_layer",
    "measure_layerwise_drift",
]
