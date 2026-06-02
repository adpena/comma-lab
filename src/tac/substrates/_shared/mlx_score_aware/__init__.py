# SPDX-License-Identifier: MIT
"""Canonical MLX-first score-aware training harness — composable typed package.

MLX-SCORE-AWARE-HARNESS-REFACTOR-WAVE 2026-05-27. Production-hardened,
separation-of-concerns refactor of the prior monolith
``tac.substrates._shared.mlx_score_aware_full_main`` (commit ``9635ca39a``).
The MLX-FIRST sister of the PyTorch-only ``pact_nerv_full_main.py``: the
substrate-AGNOSTIC MLX score-aware training loop that extinguishes the
``NotImplementedError`` for substrates whose distinguishing primitive is an
MLX-native trainable renderer.

This package is the local-training half of the MLX-first / numpy-portable
contract: MLX is used for encoder-side training + advisory research signal,
while contest inflate stays numpy-portable and exact score authority stays with
CPU/CUDA auth evaluation.

## Separation of concerns (one responsibility per module)

- ``device_gate`` — fail-closed MLX device gate (no CPU/CUDA leak; Catalog
  #1 + #317 + #325) + the shared ``MlxScoreAwareHarnessError``.
- ``targets`` — real-video target decode (Catalog #114; ``decode_video``).
- ``bundle`` — the ``RendererBundle`` substrate-UNIQUE axis + ``MlxRenderer``
  Protocol (the canonical-vs-unique boundary, Catalog #290).
- ``loss`` — gradient-reachable score-aware Lagrangian (reconstruction MSE +
  Hinton-KL T=2.0 scorer surrogate; Catalog #164 + eval_roundtrip).
- ``adapter`` — the Style-B ``MlxScoreAwareAdapter`` bridging the bundle into
  the canonical L2 harness; EMA / loop / watchdog DELEGATED to
  ``run_long_training`` (NOT duplicated).
- ``portability`` — numpy-portable inflate static verifier (8th directive;
  HNeRV parity L4).
- ``harness`` — the substrate-AGNOSTIC ``_full_main`` orchestrator composing
  the above + routing through ``run_long_training``.

Non-promotable by construction per CLAUDE.md "MLX portable-local-substrate
authority" + Catalog #127/#192/#317/#341: ``score_claim=False``,
``promotion_eligible=False``, ``ready_for_exact_eval_dispatch=False``.

[verified-against: tac.training.long_training_canonical.run_long_training canonical L2 harness]
[verified-against: tac.substrates._shared.numpy_portable_inflate LANDED numpy-portable bridge]
"""
from __future__ import annotations

from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
from tac.substrates._shared.mlx_score_aware.bundle import (
    FORWARD_CONVENTIONS,
    MlxRenderer,
    PoseScorerTeacherProvider,
    RendererBundle,
    ScorerTeacherProvider,
)
from tac.substrates._shared.mlx_score_aware.carrier_training_plan import (
    CANONICAL_SCORE_AWARE_DECODER_TRAINING_STACK,
    CarrierTrainingPlanError,
    CarrierTrainingThresholds,
    build_score_aware_carrier_training_plan,
)
from tac.substrates._shared.mlx_score_aware.coder_qat import (
    CODER_QAT_AUTHORITY,
    CoderAwareQATConfig,
    build_decoder_coder_qat_terms,
    coder_qat_loss_weights,
    coder_qat_metadata,
)
from tac.substrates._shared.mlx_score_aware.device_gate import (
    MlxScoreAwareHarnessError,
    is_mlx_available,
    require_mlx_for_harness,
)
from tac.substrates._shared.mlx_score_aware.harness import (
    CONTEST_NORMALIZER,
    MLX_EVIDENCE_GRADE,
    run_mlx_score_aware_full_main,
)
from tac.substrates._shared.mlx_score_aware.loss import (
    build_mlx_posenet_pair_teacher,
    build_mlx_segnet_pair_teacher,
    decode_frames_nhwc01,
    pose_student_inputs_nhwc,
    score_aware_loss,
)
from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    CONTEST_BYTE_PRICE_SCORE,
    MODEL_SIZE_BUDGET_PLAN_SCHEMA,
    ModelSizeBudgetPlanError,
    build_modelsize_budget_plan,
    build_modelsize_budget_plan_from_iterable,
)
from tac.substrates._shared.mlx_score_aware.portability import (
    FORBIDDEN_INFLATE_IMPORT_ROOTS,
    assert_numpy_portable_inflate,
)
from tac.substrates._shared.mlx_score_aware.targets import (
    N_PAIRS_FULL,
    decode_mlx_targets,
)

__all__ = [
    "CANONICAL_SCORE_AWARE_DECODER_TRAINING_STACK",
    "CODER_QAT_AUTHORITY",
    "CONTEST_BYTE_PRICE_SCORE",
    "CONTEST_NORMALIZER",
    "FORBIDDEN_INFLATE_IMPORT_ROOTS",
    "FORWARD_CONVENTIONS",
    "MLX_EVIDENCE_GRADE",
    "MODEL_SIZE_BUDGET_PLAN_SCHEMA",
    "N_PAIRS_FULL",
    "CarrierTrainingPlanError",
    "CarrierTrainingThresholds",
    "CoderAwareQATConfig",
    "MlxRenderer",
    "MlxScoreAwareAdapter",
    "MlxScoreAwareHarnessError",
    "ModelSizeBudgetPlanError",
    "PoseScorerTeacherProvider",
    "RendererBundle",
    "ScorerTeacherProvider",
    "assert_numpy_portable_inflate",
    "build_decoder_coder_qat_terms",
    "build_mlx_posenet_pair_teacher",
    "build_mlx_segnet_pair_teacher",
    "build_modelsize_budget_plan",
    "build_modelsize_budget_plan_from_iterable",
    "build_score_aware_carrier_training_plan",
    "coder_qat_loss_weights",
    "coder_qat_metadata",
    "decode_frames_nhwc01",
    "decode_mlx_targets",
    "is_mlx_available",
    "pose_student_inputs_nhwc",
    "require_mlx_for_harness",
    "run_mlx_score_aware_full_main",
    "score_aware_loss",
]
