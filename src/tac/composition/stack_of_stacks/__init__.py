"""Stack-of-stacks composition (beat-PR95 design Idea 3).

The stack-of-stacks composer turns the empirically-measured primitives
landed 2026-05-13 (`A1`, `A1+LAPose`, `A1+wavelet_residual`, `φ1 SABOR`,
`φ3 S2SBS`, `LangevinOptimizer`) into ONE coherent compose-and-dispatch
pipeline per the design memo at
``.omx/research/beat_pr95_curriculum_substrate_training_design_20260513.md``
(see Idea 3 — Stack-of-Stacks Composition).

Three composition layers (`Inner`, `Middle`, `Outer`) map to:

* :class:`InnerStack` — substrate base + φ1 SABOR boundary atoms + φ3 S2SBS
  HF byte-stuffing sidecar + score-gradient-aware residuals.  Composes into
  one substrate-tagged archive trailer (D4.B magic-byte pattern).
* :class:`MiddleStack` — cross-substrate composition.  Combines distinct
  inner-stacked substrates (e.g. A1 + LAPose, A1 + wavelet) with explicit
  rate-budget partition + score-aware mixing.
* :class:`OuterStack` — K-checkpoint ensemble.  Stores per-pair-best-of-K
  selector at inflate time; selector is 1 byte per pair (600 B for 600
  pairs) + the K-1 alternative checkpoints' delta payloads.

The full composer top-level API is :func:`compose_stack_of_stacks`; the
canonical inflate-time runtime is :mod:`tac.composition.stack_of_stacks.inflate`.

Score-claim discipline (NON-NEGOTIABLE per CLAUDE.md)
-----------------------------------------------------
* This module produces ARCHIVE BYTES; the canonical archive contains the
  base substrate trailer plus the stack-of-stacks D4.B magic-byte sidecar
  (``SOS1`` magic).
* Until paired ``[contest-CUDA]`` + ``[contest-CPU]`` anchors land for a
  specific compose spec, every result is ``score_claim=False``,
  ``promotion_eligible=False``, ``ready_for_exact_eval_dispatch=False``.
* The composer does NOT modify the inner archive grammar — every primitive
  retains its own magic-byte trailer; the stack-of-stacks sidecar wraps
  the ensemble per-pair selector ONLY.

Cross-references
----------------
- Design memo: ``.omx/research/beat_pr95_curriculum_substrate_training_design_20260513.md``
- SABOR audit: ``.omx/research/sabor_boundary_audit_20260513.md``
- S2SBS audit: ``.omx/research/s2sbs_blindspot_audit_20260513.md``
- LangevinOptimizer: ``tac.optimization.langevin_optimizer``
- A1+LAPose: ``tac.substrates.a1_plus_lapose``
- A1+wavelet: ``tac.substrates.a1_plus_wavelet_residual``
- Composition registry (sister surface): ``tac.composition.registry``
- Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline":
  this is a substrate-engineering lane; LOC budget per HNeRV parity
  lesson L7.
"""

from __future__ import annotations

from tac.composition.stack_of_stacks.compose import (
    SCHEMA_VERSION,
    SOS_SIDECAR_MAGIC,
    SOS_SIDECAR_VERSION,
    BoundaryAtomSpec,
    HFSidecarSpec,
    InnerStack,
    InnerStackSpec,
    MiddleStack,
    MiddleStackSpec,
    OuterStack,
    OuterStackSpec,
    ResidualSpec,
    StackOfStacksError,
    compose_stack_of_stacks,
    decompose_stack_of_stacks,
    score_aware_mixing_weights,
    validate_byte_budget,
)

__all__ = [
    "SCHEMA_VERSION",
    "SOS_SIDECAR_MAGIC",
    "SOS_SIDECAR_VERSION",
    "BoundaryAtomSpec",
    "HFSidecarSpec",
    "InnerStack",
    "InnerStackSpec",
    "MiddleStack",
    "MiddleStackSpec",
    "OuterStack",
    "OuterStackSpec",
    "ResidualSpec",
    "StackOfStacksError",
    "compose_stack_of_stacks",
    "decompose_stack_of_stacks",
    "score_aware_mixing_weights",
    "validate_byte_budget",
]
