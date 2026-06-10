# SPDX-License-Identifier: MIT
"""The 1:1 MLX PORT of PR95 (``hnerv_muon``) — parity-gated, train + export.

The clean, faithful MLX port the operator asked for: a *1:1 MLX port of PR95's
proven loop*, defined by a torch-parity GATE (bit-/score-exact vs the PR95 torch
reference per component), NOT a "PR95-inspired" harness with its own (wrong)
defaults. See ``.omx/research/why_substrate_work_was_broken_derivatives_and_the_
redirect_20260610.md`` for why the prior shared harness was a single point of
failure, and ``.omx/research/full_stack_audit_*.md`` (#81) for the C1-C9 config
defects this port fixes by construction.

The package binds the three VERIFIED MLX kernels through the score-aware loop:

- decoder: :class:`tac.local_acceleration.pr95_hnerv_mlx.HNeRVDecoderMLX`
  (bit-exact vs ``torch.nn.PixelShuffle`` + the from-scratch torch decoder, #81).
- NS-Muon: :func:`tac.local_acceleration.pr95_hnerv_mlx.zeropower_via_newtonschulz5_mlx`.
- optimizer step: :func:`tac.local_acceleration.pr95_hnerv_mlx.apply_pr95_mlx_optimizer_step`.

This package adds the missing pieces:

- :mod:`~tac.mlx_pr95_port.mlx_losses` — the 1:1 MLX port of PR95 ``losses.py``
  (4 stage seg losses + pose + exact d_seg), parity-gated vs torch.
- :mod:`~tac.mlx_pr95_port.score_bridge` — the torch-frozen-scorer <-> ``mx.vjp``
  gradient bridge (faithful score-aware loss; no second-order MLX scorer NaN).
- :mod:`~tac.mlx_pr95_port.mlx_trainer` — the score-aware loop (Muon-throughout,
  C1-C9 fixed) whose LIVE MLX render d_seg descends.

Authority: ``[macOS-MLX research-signal]`` / ``[macOS-CPU advisory]`` — the
scorer math runs on torch CPU (the exact authority path; NO MPS); the decoder
runs on MLX. Non-promotable per Catalog #192; a contest score requires
``upstream/evaluate.py`` on paired CUDA + Linux-x86_64 CPU.
"""

from tac.mlx_pr95_port.mlx_losses import (
    STAGE_SEG_LOSS_FNS_MLX,
    ce_seg_loss_mlx,
    exact_d_seg_from_logits_mlx,
    l7_softplus_seg_loss_mlx,
    pose_loss_mlx,
    smooth_disagreement_seg_loss_mlx,
    tau_softplus_seg_loss_mlx,
)
from tac.mlx_pr95_port.mlx_trainer import (
    MlxScoreAwareConfig,
    MlxScoreAwareTrainer,
)
from tac.mlx_pr95_port.score_bridge import (
    CAMERA_HW,
    SCORER_HW,
    ScoreBridgeResult,
    TorchScorerBridge,
)

__all__ = [
    "CAMERA_HW",
    "SCORER_HW",
    "STAGE_SEG_LOSS_FNS_MLX",
    "MlxScoreAwareConfig",
    "MlxScoreAwareTrainer",
    "ScoreBridgeResult",
    "TorchScorerBridge",
    "ce_seg_loss_mlx",
    "exact_d_seg_from_logits_mlx",
    "l7_softplus_seg_loss_mlx",
    "pose_loss_mlx",
    "smooth_disagreement_seg_loss_mlx",
    "tau_softplus_seg_loss_mlx",
]
