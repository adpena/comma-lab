# SPDX-License-Identifier: MIT
"""D4 substrate architecture — frame_1 + motion + residual → frame_0.

Per deep-math memo §3.5 + §6 D4 the WynerZivFrame0Substrate composes:

1. A MotionModelModule (SE3 or OPTICAL_FLOW per probe-disambiguator).
2. A PhotometricResidualParameter that holds the per-pair residual at the
   chosen coarse resolution.

Frame_1 is provided EXTERNALLY at training time (decoded from
``upstream/videos/0.mkv``) and at inflate time (reconstructed by the BASE
substrate referenced via the archive's BASE_SHA section).

This substrate is INTENTIONALLY simple per HNeRV parity discipline lesson L12
(single-LOC-per-LOC review discipline): the score gains come from the
score-aware Wyner-Ziv loss + the rate term saved by not encoding frame_0
directly, not from architectural complexity.

Catalog #124 archive-grammar 8 fields are declared inline in __init__.py at
module level so the AST walker observes them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tac.substrates.d4_wyner_ziv_frame_0.frame0_synthesis import synthesize_frame_0
from tac.substrates.d4_wyner_ziv_frame_0.motion_model import (
    EVAL_HW,
    MotionModelMode,
    MotionModelModule,
    OpticalFlowField,
    SE3MotionParams,
)

NUM_PAIRS: int = 600
"""Contest pair count."""

PER_PAIR_SE3_PARAMS: int = 6
"""SE(3) parameter count per pair (3 translation + 3 axis-angle)."""

# Archive byte targets (substrate-engineering scope, paired with an external
# base substrate). The D4 sidecar archive carries motion + residual + base sha
# only.
TOTAL_ARCHIVE_TARGET_BYTES_MIN: int = 30_000
TOTAL_ARCHIVE_TARGET_BYTES_MAX: int = 80_000

BASE_SHA_HEX_LEN: int = 64
"""Length of the base substrate archive sha256 hex stored in the archive."""


@dataclass(frozen=True)
class WynerZivFrame0Config:
    """Static design-time parameters for the D4 substrate.

    Args:
        motion_mode: SE3_PARAMETRIC or OPTICAL_FLOW per probe-disambiguator.
        num_pairs: contest pair count (default 600).
        output_height: scorer-resolution height (default 384).
        output_width: scorer-resolution width (default 512).
        flow_grid_h: optical-flow coarse grid height (only used in OPTICAL_FLOW
            mode; default 12).
        flow_grid_w: optical-flow coarse grid width (default 16).
        residual_coarse_h: photometric residual coarse height (default 48 =
            1/8 scorer height).
        residual_coarse_w: photometric residual coarse width (default 64 =
            1/8 scorer width).
        residual_loss_weight: scalar Lagrangian weight on the per-pixel
            residual L² penalty (kept small; the dominant training signal
            comes from the scorer through the reconstructed pair).
        base_substrate_id: human-readable identifier of the intended base
            substrate (e.g. ``"a1_hnerv_ft_microcodec"``, ``"pr101_lc_v2"``,
            ``"hdm8"``). Stored in archive meta for forensic provenance; does
            NOT pin the base substrate at runtime (the BASE_SHA hex stored
            separately is the cryptographic custody anchor).
    """

    motion_mode: MotionModelMode = MotionModelMode.SE3_PARAMETRIC
    num_pairs: int = NUM_PAIRS
    output_height: int = EVAL_HW[0]
    output_width: int = EVAL_HW[1]
    flow_grid_h: int = 12
    flow_grid_w: int = 16
    residual_coarse_h: int = 48
    residual_coarse_w: int = 64
    residual_loss_weight: float = 1.0
    base_substrate_id: str = "external_base_substrate_v0"

    @property
    def output_hw(self) -> tuple[int, int]:
        return (self.output_height, self.output_width)


class WynerZivFrame0Substrate(nn.Module):
    """The D4 substrate: motion model + per-pair photometric residual.

    Composition contract with an external base substrate:

    * At TRAINING time, ``frame_1`` is provided as the ground-truth video
      pair second-frame (decoded from ``upstream/videos/0.mkv`` via
      ``tac.substrates._shared.trainer_skeleton.decode_real_pairs``). The
      D4 trainer minimizes the joint loss
      ``L = L_score(score_pair_components(synthesize_frame_0(frame_1, ...),
      frame_1; gt_frame_0, gt_frame_1)) + λ_res · ||residual||²``.

    * At INFLATE time, ``frame_1`` is provided by the base substrate's
      ``inflate_one_video(...)`` output. The D4 inflate runtime loads the
      base substrate's archive, verifies its sha matches the WZF01 header's
      BASE_SHA section, inflates frame_1, then synthesizes frame_0 via the
      D4 motion + residual.

    The substrate does NOT carry a renderer (the base substrate carries the
    full RGB renderer); it only carries the frame-0 derivation primitives.
    """

    def __init__(self, cfg: WynerZivFrame0Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.motion = MotionModelModule(
            mode=cfg.motion_mode,
            num_pairs=cfg.num_pairs,
            flow_grid_h=cfg.flow_grid_h,
            flow_grid_w=cfg.flow_grid_w,
            output_hw=cfg.output_hw,
        )
        # Per-pair photometric residual at coarse resolution (3 channels, RGB).
        # Initialized at zero (= frame_0 == warped_frame_1, i.e. assume the
        # motion model alone is sufficient on cycle 0).
        self.residual_coarse = nn.Parameter(
            torch.zeros(cfg.num_pairs, 3, cfg.residual_coarse_h, cfg.residual_coarse_w)
        )

    def reconstruct_pair(
        self,
        frame_1: torch.Tensor,
        pair_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct (frame_0, frame_1) given frame_1.

        Args:
            frame_1: RGB tensor in unit range.

                * If ``pair_indices`` is ``None``: shape must be
                  ``(cfg.num_pairs, 3, H, W)`` — backward-compat full-batch
                  reconstruction (forbidden for cfg.num_pairs >= 256 on T4
                  per the 2026-05-14 OOM anchor; use mini-batched indexing
                  instead).
                * If ``pair_indices`` is provided: shape must be
                  ``(len(pair_indices), 3, H, W)`` and ``pair_indices`` is a
                  1-D long tensor selecting which trainable motion params +
                  residual rows to use. This is the canonical path for
                  training and OOM-bounded inflate batches.

            pair_indices: optional ``(B,)`` long tensor in ``[0, cfg.num_pairs)``
                selecting the per-pair motion params + residual to use for
                this mini-batch. When provided the full ``cfg.num_pairs ==
                frame_1.shape[0]`` invariant is replaced with the per-index
                selection invariant. Gradients flow into the selected rows
                of ``motion.se3_flat`` / ``motion.flow_uv`` /
                ``residual_coarse`` only.

        Returns:
            (frame_0_reconstructed, frame_1_unchanged). Frame_1 is passed
            through untouched per the SegNet-frame-1-byte-identical
            invariant (SegNet sees only frame_1).

        OOM safety: per the D4 OOM fix (Catalog #209 sister gate +
        lane_d4_oom_fix_minibatch_reconstruct_20260514), callers that
        operate on the full 600-pair contest dataset MUST mini-batch via
        ``pair_indices`` because the full forward (warp + residual
        upsample 48x64 -> 384x512) requires ~13 GB of activation memory
        which exceeds T4 14.56 GB capacity.
        """
        if pair_indices is None:
            # Backward-compat path: full-batch reconstruction.
            if frame_1.shape[0] != self.cfg.num_pairs:
                raise ValueError(
                    f"frame_1 batch {frame_1.shape[0]} != cfg.num_pairs "
                    f"{self.cfg.num_pairs}; pass pair_indices for "
                    f"mini-batched reconstruction"
                )
            sel_se3_flat = (
                self.motion.se3_flat
                if self.motion.se3_flat is not None
                else None
            )
            sel_flow_uv = (
                self.motion.flow_uv
                if self.motion.flow_uv is not None
                else None
            )
            sel_residual = self.residual_coarse
        else:
            if pair_indices.dim() != 1:
                raise ValueError(
                    f"pair_indices must be 1-D; got shape "
                    f"{tuple(pair_indices.shape)}"
                )
            if pair_indices.numel() == 0:
                raise ValueError("pair_indices must be non-empty")
            if frame_1.shape[0] != pair_indices.shape[0]:
                raise ValueError(
                    f"frame_1 batch {frame_1.shape[0]} != "
                    f"len(pair_indices) {pair_indices.shape[0]}"
                )
            # Validate the index range fails loud with a sane error before
            # torch.index_select raises (which produces a less actionable
            # CUDA-side assertion).
            min_idx = int(pair_indices.min().item())
            max_idx = int(pair_indices.max().item())
            if min_idx < 0 or max_idx >= self.cfg.num_pairs:
                raise ValueError(
                    f"pair_indices range [{min_idx}, {max_idx}] outside "
                    f"[0, {self.cfg.num_pairs})"
                )
            # torch.index_select preserves gradient flow into the selected
            # rows of the parent nn.Parameter; sliced rows accumulate
            # gradients via autograd's scatter-add path.
            sel_se3_flat = (
                self.motion.se3_flat.index_select(0, pair_indices)
                if self.motion.se3_flat is not None
                else None
            )
            sel_flow_uv = (
                self.motion.flow_uv.index_select(0, pair_indices)
                if self.motion.flow_uv is not None
                else None
            )
            sel_residual = self.residual_coarse.index_select(0, pair_indices)

        # Build motion structs from the selected (or full) parameter rows.
        if self.cfg.motion_mode == MotionModelMode.SE3_PARAMETRIC:
            assert sel_se3_flat is not None
            se3_params = SE3MotionParams.from_flat(sel_se3_flat)
            flow_field = None
        else:
            assert sel_flow_uv is not None
            se3_params = None
            flow_field = OpticalFlowField(
                flow_uv=sel_flow_uv,
                grid_h=self.cfg.flow_grid_h,
                grid_w=self.cfg.flow_grid_w,
            )
        frame_0 = synthesize_frame_0(
            frame_1=frame_1,
            motion_mode=self.cfg.motion_mode,
            se3_params=se3_params,
            flow_field=flow_field,
            residual=sel_residual,
            output_hw=self.cfg.output_hw,
            clamp_unit=True,
        )
        return frame_0, frame_1


__all__ = [
    "BASE_SHA_HEX_LEN",
    "EVAL_HW",
    "NUM_PAIRS",
    "PER_PAIR_SE3_PARAMS",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "WynerZivFrame0Config",
    "WynerZivFrame0Substrate",
]
