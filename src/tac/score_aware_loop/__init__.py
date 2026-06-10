# SPDX-License-Identifier: MIT
"""The WORKING score-aware loop (task #76 fix for the INERT loop).

The PR95-faithful, reusable, well-conditioned score-aware training surface:
direct margin/CE through the LIVE frozen SegNet (no learnable student head, no
recon-MSE), GT-SegNet-argmax targets, the canonical eval roundtrip in the inner
loop, ``100*seg + 1*pose`` aggregation, AdamW + cosine + grad-clip-1.0, and an
EMA shadow inference checkpoint.

Public surface:
    - :class:`~tac.score_aware_loop.trainer.ScoreAwareTrainer`
    - :class:`~tac.score_aware_loop.trainer.ScoreAwareLoopConfig`
    - :mod:`~tac.score_aware_loop.live_segnet_loss` (the PR95 stage losses)
    - :class:`~tac.score_aware_loop.tiny_carrier.TinyPairCarrier`

See ``.omx/research/inert_loop_fix_*.md`` for the head-to-head verdict.
"""

from tac.score_aware_loop.live_segnet_loss import (
    STAGE_SEG_LOSS_FNS,
    ce_seg_loss,
    exact_d_seg_from_logits,
    l7_softplus_seg_loss,
    pose_loss,
    smooth_disagreement_seg_loss,
    tau_softplus_seg_loss,
)
from tac.score_aware_loop.tiny_carrier import TinyPairCarrier
from tac.score_aware_loop.trainer import (
    SCORER_HW,
    ScoreAwareLoopConfig,
    ScoreAwareTrainer,
)

__all__ = [
    "SCORER_HW",
    "STAGE_SEG_LOSS_FNS",
    "ScoreAwareLoopConfig",
    "ScoreAwareTrainer",
    "TinyPairCarrier",
    "ce_seg_loss",
    "exact_d_seg_from_logits",
    "l7_softplus_seg_loss",
    "pose_loss",
    "smooth_disagreement_seg_loss",
    "tau_softplus_seg_loss",
]
