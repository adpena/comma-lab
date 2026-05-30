# SPDX-License-Identifier: MIT
"""Local-leverage acceleration helpers for Apple Silicon (M5 Max + MLX + Metal + MPS).

Per operator directive 2026-05-21 verbatim *"Let's make sure we are leveraging
local cpu and mps and metal and mlx as much as possible"* + Carmack MVP-first
phasing 5-step amplification per CLAUDE.md non-negotiable.

This package operationalizes the M5 Max 128GB unified memory + Metal GPU +
MLX framework as a FIRST-CLASS local pre-dispatch research substrate for:

- Substrate prototyping (paradigm validation BEFORE paid Modal/Vast.ai spend)
- Codec primitive acceleration (Metal compute shaders / MLX kernels)
- Premise verification at 100% recipe-exact fidelity (no Modal worker setup)
- Conditional local candidate ranking after cache identity, parity, and score
  calibration gates; never standalone contest-axis authority

Per CLAUDE.md non-negotiables PRESERVED:
- **MPS auth eval is NOISE** (Catalog #1; never authoritative score path)
- **macOS-CPU is `[macOS-CPU advisory]`** (Catalog #192; non-promotable
  without paired Linux x86_64)
- **Submission auth eval — BOTH CPU AND CUDA** (paid Linux x86_64 + NVIDIA
  remains contest-axis promotion gate)
- **Apples-to-apples evidence discipline** (every signal carries canonical
  axis_tag + hardware_substrate + evidence_grade per Catalog #287/#323)

Per CLAUDE.md "MPS auth eval is NOISE" empirical exception (commit
``c8d51ebb5`` 2026-05-19 META-ASSUMPTION review): the 23x MPS-vs-CUDA
universality assumption was HARD-EARNED-NUANCED → on current archives +
current PoseNet/SegNet the drift is <1% (MPS-VIABLE probe outcome PROCEED
at 0.072% gap, 69x below 5% threshold). This package's helpers leverage
that finding for dev-velocity acceleration WITHOUT relaxing the
authoritative-axis gates.

Canonical surfaces in this package:

- :mod:`tac.local_acceleration.routability_audit` — per-substrate
  classification of local-routability across the 75 substrate trainer
  surface (LOCAL-MLX-TRAINABLE / LOCAL-MPS-TRAINABLE / LOCAL-CPU-PROXY /
  PAID-ONLY) per M5 Max 128GB unified memory + Metal GPU + MLX framework.
- :mod:`tac.local_acceleration.mlx_integration` — MLX framework canonical
  training-loop scaffold + utilities for substrate-class neural net
  prototyping on Apple Silicon GPU (Metal-backed). Non-promotable by
  construction per Catalog #1/#192/#317 sister discipline.
- :mod:`tac.local_acceleration.metal_kernels` — Metal compute shader hooks
  for codec primitive acceleration (entropy coding inner loops, byte
  manipulation, lossless transforms). Reserved API surface; current
  implementation falls back to PyTorch MPS for kernels Metal doesn't
  beat.

Sister of:
- :mod:`tac.optimization.mps_research_signal` (Catalog #192 sister; MPS
  proxy curve discovery)
- :mod:`tac.optimization.macos_cpu_advisory_signal` (Catalog #192 sister;
  macOS-CPU advisory proxy)
- :mod:`tac.cathedral_consumers.mps_viable_prescreen_consumer` (Catalog
  #341 sister; routing recommendation)
- :mod:`tac.cathedral_consumers.cpu_axis_optimal_consumer` (sister at
  CPU-axis pre-screen surface)
"""

from __future__ import annotations

__all__ = [
    "EVIDENCE_GRADE_METAL",
    "EVIDENCE_GRADE_MLX",
    "EVIDENCE_TAG_METAL",
    "EVIDENCE_TAG_MLX",
    "MLX_ACQUISITION_BATCH_OPERATION_SET_SCHEMA",
    "MLX_ACQUISITION_BATCH_SCHEMA",
    "MLX_ACQUISITION_REPRESENTATION_CONTRACT_SCHEMA",
    "PROVENANCE_EVIDENCE_GRADE_MLX",
    "SCHEMA_VERSION",
    "build_mlx_research_signal_provenance",
]

SCHEMA_VERSION = "local_acceleration.v1"

# Per Catalog #287 + #323 canonical Provenance: every local signal carries
# explicit non-promotable evidence_grade + axis_tag. Sister of
# tac.optimization.mps_research_signal.EVIDENCE_GRADE +
# tac.optimization.macos_cpu_advisory_signal.EVIDENCE_GRADE.
EVIDENCE_GRADE_MLX = "macOS-MLX-research-signal"
EVIDENCE_TAG_MLX = "[macOS-MLX research-signal]"
PROVENANCE_EVIDENCE_GRADE_MLX = "macos_mlx_research_signal"

EVIDENCE_GRADE_METAL = "macOS-Metal-research-signal"
EVIDENCE_TAG_METAL = "[macOS-Metal research-signal]"

MLX_ACQUISITION_BATCH_SCHEMA = "mlx_acquisition_batch.v1"
MLX_ACQUISITION_BATCH_OPERATION_SET_SCHEMA = "mlx_acquisition_operation_set.v1"
MLX_ACQUISITION_REPRESENTATION_CONTRACT_SCHEMA = (
    "mlx_acquisition_representation_contract.v1"
)


def build_mlx_research_signal_provenance(
    *,
    artifact_sha256: str,
    source_path: str,
    captured_at_utc: str | None = None,
):
    """Build canonical Provenance for a local MLX research-signal artifact.

    ``EVIDENCE_GRADE_MLX`` remains the legacy row-display grade consumed by
    older queues. This helper emits the canonical enum value
    ``macos_mlx_research_signal`` through ``tac.provenance`` so new rows can
    carry both without vocabulary drift.
    """

    from tac.provenance import build_provenance_for_macos_mlx_research_signal

    return build_provenance_for_macos_mlx_research_signal(
        artifact_sha256=artifact_sha256,
        source_path=source_path,
        captured_at_utc=captured_at_utc,
    )
