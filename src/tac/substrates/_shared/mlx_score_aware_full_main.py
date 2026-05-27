# SPDX-License-Identifier: MIT
"""Backwards-compat facade for the MLX-first score-aware harness.

# NO_GRAD_WAIVED:MLX_substrate_harness_uses_mlx_value_and_grad_lazy_eval_no_pytorch_autograd_per_mlx_first_canonical_doctrine_8th_standing_directive
# AUTOCAST_FP16_WAIVED:MLX_harness_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_8th_standing_directive

MLX-SCORE-AWARE-HARNESS-REFACTOR-WAVE 2026-05-27: the original ~780 LOC monolith
(commit ``9635ca39a``) is refactored into the composable typed package
``tac.substrates._shared.mlx_score_aware`` (separation of concerns:
device_gate / targets / bundle / loss / adapter / portability / harness — each
unit-tested + 30s-reviewable). This module remains a THIN re-export facade so
the prior import path keeps working for the 2 landed trainers
(``dreamer_v3_rssm`` / ``z8_*``) + their tests + any sister consumer.

New code SHOULD import from the package directly::

    from tac.substrates._shared.mlx_score_aware import (
        RendererBundle, decode_mlx_targets, run_mlx_score_aware_full_main,
    )

This facade is kept (not deleted) per CLAUDE.md "Beauty, simplicity, and
developer experience" — public API stability for OSS consumers; the canonical
implementation lives in the package.

[verified-against: tac.substrates._shared.mlx_score_aware package (canonical impl)]
"""
from __future__ import annotations

from tac.substrates._shared.mlx_score_aware import (
    CONTEST_NORMALIZER,
    FORBIDDEN_INFLATE_IMPORT_ROOTS,
    FORWARD_CONVENTIONS,
    MLX_EVIDENCE_GRADE,
    N_PAIRS_FULL,
    MlxRenderer,
    MlxScoreAwareAdapter,
    MlxScoreAwareHarnessError,
    PoseScorerTeacherProvider,
    RendererBundle,
    ScorerTeacherProvider,
    assert_numpy_portable_inflate,
    build_mlx_posenet_pair_teacher,
    build_mlx_segnet_pair_teacher,
    decode_frames_nhwc01,
    decode_mlx_targets,
    is_mlx_available,
    require_mlx_for_harness,
    run_mlx_score_aware_full_main,
    score_aware_loss,
)

__all__ = [
    "CONTEST_NORMALIZER",
    "FORBIDDEN_INFLATE_IMPORT_ROOTS",
    "FORWARD_CONVENTIONS",
    "MLX_EVIDENCE_GRADE",
    "N_PAIRS_FULL",
    "MlxRenderer",
    "MlxScoreAwareAdapter",
    "MlxScoreAwareHarnessError",
    "PoseScorerTeacherProvider",
    "RendererBundle",
    "ScorerTeacherProvider",
    "assert_numpy_portable_inflate",
    "build_mlx_posenet_pair_teacher",
    "build_mlx_segnet_pair_teacher",
    "decode_frames_nhwc01",
    "decode_mlx_targets",
    "is_mlx_available",
    "require_mlx_for_harness",
    "run_mlx_score_aware_full_main",
    "score_aware_loss",
]
