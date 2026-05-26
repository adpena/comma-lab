# SPDX-License-Identifier: MIT
"""Hinton-distilled scorer surrogate substrate (MLX foundation surface).

This package operationalises the **TOP-1** substrate-class-shift candidate
per the DQS1-ASYMPTOTIC-FLOOR ranking
(``.omx/research/dqs1_asymptotic_floor_substrate_class_shift_prioritization_20260525.md``):
Hinton-Vinyals-Dean 2014 distillation paradigm (KL T=2.0 between teacher
SegNet logits and a smaller-capacity student) extended to the FULL training
loop per the Quantizr canonical 0.33 [contest-CUDA] anchor and AAA T4 §6.5
+ §12.4 Tier 2A spec.

Slot 1 (commit ``acf1661ca``) landed the canonical PR95 MLX long-training
infrastructure (``tac.local_acceleration.pr95_hnerv_mlx_long_training``)
including ``SubstrateAdapterScaffold`` whose ``custom_loss_fn`` field
matches the signature ``(bundle, indices, targets_batch) -> mx.array``.
This package provides the canonical Hinton KL T=2.0 ``custom_loss_fn`` for
that scaffold so the operator can drive $0 macOS-MLX long-training
validation BEFORE any paid Modal / Vast.ai / HF Jobs Tier-2 dispatch fires.

**Axis discipline**: every artifact produced via this surface is
``[macOS-MLX research-signal]`` per CLAUDE.md "MLX portable-local-substrate
authority" non-negotiable + Catalog #192 + Catalog #1. ``score_claim``,
``promotion_eligible``, ``rank_or_kill_eligible``, and
``ready_for_exact_eval_dispatch`` are all ``False`` by construction;
paid CPU + CUDA paired auth-eval on 1:1 contest-compliant hardware
remains required before any contest-axis claim per CLAUDE.md "Submission
auth eval — BOTH CPU AND CUDA" non-negotiable.

**Sister scope**: Slot 3 (in-flight at landing time) handles paid dispatch
packaging (recipe + trainer + symposium). THIS package is sister-DISJOINT
at the file-level: only NEW files under
``src/tac/substrates/hinton_distilled_scorer_surrogate/*`` and an executor
CLI at ``tools/run_hinton_mlx_long_training_smoke.py``. No mutation of the
Slot 1 canonical infrastructure or sister codex
``mlx_scorer_adapters.py``.

Operationally the two waves PAIR: this $0 MLX validation gates the next
local queue-owned scorer-teacher proof step. ``CONVERGES_CONSISTENTLY`` →
``LOCAL_MLX_QUEUE_READY``;
``DIVERGES`` / ``OSCILLATES`` / ``SUB_PARADIGM`` → DEFER per Catalog #307
(implementation-level falsification) + Catalog #308 (alternative reducer
enumeration) + Catalog #325 (per-substrate symposium reactivation).
"""

from __future__ import annotations

from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
    DEFAULT_DISTILLATION_TEMPERATURE,
    DEFAULT_SEGNET_CLASSES,
    HintonDistilledKLLossResult,
    HintonMlxCustomLossFnConfig,
    LearnableConv1x1StudentHead,
    MockTeacherLogitsProvider,
    RealSegNetTeacherLogitsCache,
    TeacherLogitsProvider,
    build_learnable_student_head,
    build_real_segnet_teacher_cache,
    hinton_distilled_kl_t2_loss,
    kl_divergence_between_softmax,
    make_hinton_custom_loss_fn,
    softmax_with_temperature,
)

__all__ = [
    "DEFAULT_DISTILLATION_TEMPERATURE",
    "DEFAULT_SEGNET_CLASSES",
    "HintonDistilledKLLossResult",
    "HintonMlxCustomLossFnConfig",
    "LearnableConv1x1StudentHead",
    "MockTeacherLogitsProvider",
    "RealSegNetTeacherLogitsCache",
    "TeacherLogitsProvider",
    "build_learnable_student_head",
    "build_real_segnet_teacher_cache",
    "hinton_distilled_kl_t2_loss",
    "kl_divergence_between_softmax",
    "make_hinton_custom_loss_fn",
    "softmax_with_temperature",
]
