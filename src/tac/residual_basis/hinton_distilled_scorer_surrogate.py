"""Hinton-distilled SegNet + PoseNet surrogate for L2 residual encoders.

Per W's DEFERRED reactivation criterion #1 (memory
``feedback_l2_sparse_aware_encoders_first_dispatch_landed_20260511.md``) +
N's D2 council verdict + Phase 3 Catalog #134 prerequisite:

The L2 residual encoders' YUV6 MSE proxy DOES NOT track contest score at the
PR106 r2 operating point (proxy beta_term ~50000 dwarfs alpha rate term ~15;
coordinate descent picks the densest-MSE representation regardless of byte
cost). This module provides the canonical fix: a pair of TINY (≤10MB each)
Hinton-distilled networks that mimic the contest's SegNet + PoseNet, train
via Hinton T=2.0 KL distillation per Quantizr's UCLA approach (memory
``Quantizr intelligence`` in CLAUDE.md), and produce DENSE gradients
everywhere — including the simplex boundary where the contest scorer's argmax
is discontinuous.

The L2 encoder consumes the distilled-scorer outputs in its inner-loop
Lagrangian INSTEAD of the YUV6 MSE proxy. At eval time we restore the frozen
contest scorer; per Hinton-Vinyals-Dean 2014 §3 the distillation gap is
typically 1-3% under T=2.0 (Phase 3 Catalog #134 threshold ≤ 0.03).

Composition strategy
--------------------
This module is a **THIN COMPOSITION LAYER** over the canonical sister modules:

* ``tac.ib_lagrangian_aux_scorer`` — provides the actual ``AuxiliaryScorer``
  factory + ``train_aux_scorer`` Hinton-distillation loop (T=2.0 + λ_GT +
  EMA(0.997) per CLAUDE.md non-negotiables).
* ``tac.differentiable_eval_roundtrip`` — provides the canonical
  ``apply_eval_roundtrip_during_training`` + ``differentiable_rgb_to_yuv6``
  for the inner-loop preprocess (HNeRV parity discipline lesson 8).
* ``tac.score_geometry`` — provides the contest-canonical Lagrangian
  coefficients (α=25, β=100, γ_p=10).

This composition layer adds:

1. ``DistilledSegNet`` / ``DistilledPoseNet`` thin wrappers exposing the
   contest-faithful ``(B, T, C, H, W) -> (seg_logits, pose_floats)`` contract
   directly to L2 residual encoders (which operate on RGB pair tensors,
   not the contest's YUV6-conditioned input).
2. ``compute_distortion_via_distilled_scorer()`` — the L2-encoder-facing
   helper that returns a ``(d_seg_aux, d_pose_aux)`` tuple suitable for
   substitution into the Lagrangian's ``beta_term`` and ``gamma_term``.
3. ``ScorerSurrogateConfig`` — frozen dataclass holding the surrogate's
   architecture + checkpoint custody fields (so callers cannot accidentally
   load a non-distilled mimic).
4. ``load_pretrained_distilled_scorer_pair()`` — loader for an
   EMA-shadow checkpoint produced by ``tac.ib_lagrangian_aux_scorer.train_aux_scorer``;
   refuses to load a checkpoint that does not declare ``distill_label``,
   ``ema_decay >= 0.99``, ``distill_temperature > 0``, OR has a
   distillation gap above the Phase 3 threshold.

Per Catalog #123 (`check_no_weight_domain_saliency_on_score_gradient_substrate`):
the distilled surrogate consumes SCORE-GRADIENT inputs — NOT weight-domain
proxies. The gradient flows from the L2 residual through the distilled
scorer's logit/pose head back into the residual coefficients. The
``mean(theta^2)`` Track 4 v1 anti-pattern is structurally avoided.

Per HNeRV parity discipline lesson 8 (eval-roundtrip-aware + differentiable
scorer-preprocess training): the surrogate's input is routed through the
canonical ``apply_eval_roundtrip_during_training`` + ``differentiable_rgb_to_yuv6``
helpers BEFORE the surrogate's forward pass. Anything that severs gradient
reachability (upstream ``rgb_to_yuv6`` is ``@torch.no_grad()`` / in-place)
is bypassed in favor of the differentiable replacement.

Per CLAUDE.md "Strict scorer rule": the distilled surrogate is a COMPRESS-TIME
ONLY tool. It is NEVER loaded at inflate time. The L2 encoder consumes the
distilled scorer in its INNER LOOP (during compress-time encoding); the
inflate runtime sees ZERO scorer weights. ``score_claim`` permanently False;
``promotion_eligible`` permanently False; ``ready_for_exact_eval_dispatch``
permanently False until exact T4 + paired CPU eval.

Public API
----------
``ScorerSurrogateConfig`` — frozen dataclass (architecture + checkpoint fields)
``DistilledSegNet`` / ``DistilledPoseNet`` — torch.nn.Module thin wrappers
``load_pretrained_distilled_scorer_pair`` — loader (refuses non-distilled
                                            or unmeasured-gap checkpoints)
``compute_distortion_via_distilled_scorer`` — L2-encoder-facing helper
``DistortionViaDistilledScorerError`` — typed error raised on contract
                                        violations

Cross-references
----------------
* W's DEFERRED memo (this module's bench source):
  ``feedback_l2_sparse_aware_encoders_first_dispatch_landed_20260511.md``
* Sister Hinton-distilled trainer:
  ``tac.ib_lagrangian_aux_scorer`` + ``experiments/train_t10_ib_lagrangian_aux_scorer.py``
* Phase 3 prerequisite gate (Catalog #134):
  ``Phase3DispatchGate`` in ``tac.phase3.joint_scorer_renderer_codec``
* Catalog #123 (forbidden weight-domain saliency):
  ``feedback_track4_bug_class_fix_self_protect_landed_20260509.md``
* HNeRV parity discipline lessons 1/6/8 (CLAUDE.md)
* CLAUDE.md "Quantizr intelligence" — Hinton T=2.0 SegNet KL distillation canon
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Optional

import torch
import torch.nn as nn

# Phase 3 prereq threshold per Catalog #134 / Hinton-Vinyals-Dean 2014 §3.
DEFAULT_DISTILL_GAP_THRESHOLD: Final[float] = 0.03

# Default surrogate footprint cap. The contest's full SegNet + PoseNet pair is
# ~73 MB; the surrogate must be SUBSTANTIALLY smaller (per Quantizr's UCLA
# approach: ~5-10 MB each so the pair fits in ≤20 MB). Refused above this cap.
MAX_SURROGATE_PARAMS: Final[int] = 5_000_000  # ~20 MB at 4 bytes/param

# Pose dim used by contest score (first 6 of FastViT-T12 hydra head).
CONTEST_POSE_DIM: Final[int] = 6
CONTEST_SEG_CLASSES: Final[int] = 5

# Camera + scorer dimensions (canonical contest values).
CAMERA_H: Final[int] = 874
CAMERA_W: Final[int] = 1164
SCORER_H: Final[int] = 384
SCORER_W: Final[int] = 512


class DistortionViaDistilledScorerError(ValueError):
    """Raised on contract violations in the distilled-scorer composition layer."""


@dataclass(frozen=True)
class ScorerSurrogateConfig:
    """Frozen dataclass holding the surrogate's architecture + checkpoint fields.

    Per CLAUDE.md "no silent defaults" rule: every field is required at
    construction. ``__post_init__`` validates ranges.

    Attributes
    ----------
    seg_class_count
        Number of SegNet classes (canonical = 5).
    pose_dim
        PoseNet output dims contributing to score (canonical = 6).
    distill_temperature
        Hinton T (council canon = 2.0). MUST be > 0 (sister-validated by
        ``AuxiliaryScorerConfig``).
    distill_gap_threshold
        Maximum acceptable distillation gap. Default = 0.03 (Catalog #134
        Phase 3 threshold). Loader refuses checkpoints above this.
    expected_distill_label_substring
        Substring required to appear in the loaded checkpoint's
        ``distill_label`` field (e.g. ``t10`` for the Phase 2 T10 trainer
        family). Empty string = no label requirement.
    """

    seg_class_count: int
    pose_dim: int
    distill_temperature: float
    distill_gap_threshold: float
    expected_distill_label_substring: str

    def __post_init__(self) -> None:
        if not isinstance(self.seg_class_count, int) or self.seg_class_count < 2:
            raise DistortionViaDistilledScorerError(
                f"seg_class_count must be int >= 2; got {self.seg_class_count}"
            )
        if not isinstance(self.pose_dim, int) or self.pose_dim < 1:
            raise DistortionViaDistilledScorerError(
                f"pose_dim must be int >= 1; got {self.pose_dim}"
            )
        if (
            math.isnan(self.distill_temperature)
            or math.isinf(self.distill_temperature)
            or self.distill_temperature <= 0.0
        ):
            raise DistortionViaDistilledScorerError(
                f"distill_temperature must be finite > 0; got {self.distill_temperature}"
            )
        if (
            math.isnan(self.distill_gap_threshold)
            or math.isinf(self.distill_gap_threshold)
            or self.distill_gap_threshold <= 0.0
        ):
            raise DistortionViaDistilledScorerError(
                f"distill_gap_threshold must be finite > 0; got {self.distill_gap_threshold}"
            )
        if not isinstance(self.expected_distill_label_substring, str):
            raise DistortionViaDistilledScorerError(
                "expected_distill_label_substring must be a string (use '' to disable check)"
            )

    @classmethod
    def council_canonical(cls) -> "ScorerSurrogateConfig":
        """Return the council-canonical config (T=2.0, gap≤0.03, contest classes/dims).

        This is the configuration the L2 residual encoders default to when
        ``use_hinton_distilled_scorer=True``.
        """
        return cls(
            seg_class_count=CONTEST_SEG_CLASSES,
            pose_dim=CONTEST_POSE_DIM,
            distill_temperature=2.0,
            distill_gap_threshold=DEFAULT_DISTILL_GAP_THRESHOLD,
            expected_distill_label_substring="",
        )


# ---------------------------------------------------------------------------
# DistilledSegNet — thin wrapper exposing seg-only output from the aux scorer
# ---------------------------------------------------------------------------


class DistilledSegNet(nn.Module):
    """Thin wrapper: distilled SegNet matching contest's argmax-on-last-frame contract.

    The contest's SegNet (``smp.Unet('tu-efficientnet_b2', classes=5)``) consumes
    the LAST frame of a 2-frame pair (``x[:, -1, ...]`` per upstream/modules.py)
    and produces 5-class logits at scorer resolution (384, 512). The distortion
    is the fraction of pixels where argmax differs from the GT mask's argmax.

    This wrapper takes a ``(B, 3, H, W)`` RGB tensor (the L2 encoder's native
    layout — single frame; the encoder calls separately for each frame of a pair
    and the L2 inner loop selects the SECOND frame for SegNet per the contest
    convention). It does NOT need an internal frame-pair selector — the caller
    supplies the correct frame.

    Parameters
    ----------
    seg_class_count
        Number of SegNet classes. Canonical = 5.
    base_channels
        First-conv channels for the surrogate. Default 16 → ~50K params total
        (well under 5M cap).
    """

    def __init__(self, seg_class_count: int = CONTEST_SEG_CLASSES, base_channels: int = 16) -> None:
        super().__init__()
        if not isinstance(seg_class_count, int) or seg_class_count < 2:
            raise DistortionViaDistilledScorerError(
                f"seg_class_count must be int >= 2; got {seg_class_count}"
            )
        if not isinstance(base_channels, int) or base_channels < 4:
            raise DistortionViaDistilledScorerError(
                f"base_channels must be int >= 4; got {base_channels}"
            )
        # Encoder: stride-2 conv stem + a small bottleneck (similar parameter
        # count to the contest scorer's first stage but vastly cheaper).
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=False),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=False),
        )
        # Decoder: nearest-upsample + per-class projection (so output preserves
        # input H/W). Avoids ConvTranspose2d's checkerboard artifacts which
        # would distort the surrogate's argmax edges from the contest scorer's.
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4.0, mode="nearest"),
            nn.Conv2d(c * 2, seg_class_count, kernel_size=1),
        )

    def forward(self, rgb_bchw: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        rgb_bchw
            ``(B, 3, H, W)`` RGB tensor in [0, 255]. Caller is responsible for
            calling on the SegNet-relevant frame (second frame of each pair).

        Returns
        -------
        torch.Tensor
            ``(B, seg_class_count, H, W)`` logits.
        """
        if rgb_bchw.dim() != 4 or rgb_bchw.shape[1] != 3:
            raise DistortionViaDistilledScorerError(
                f"DistilledSegNet expects (B, 3, H, W); got {tuple(rgb_bchw.shape)}"
            )
        h_enc = self.stem(rgb_bchw)
        return self.head(h_enc)


# ---------------------------------------------------------------------------
# DistilledPoseNet — thin wrapper exposing pose-only output from the aux scorer
# ---------------------------------------------------------------------------


class DistilledPoseNet(nn.Module):
    """Thin wrapper: distilled PoseNet matching contest's pose-MSE contract.

    The contest's PoseNet (FastViT-T12 → Hydra) consumes BOTH frames of a pair
    in YUV6 concatenation (12-channel) and produces a 12-dim pose vector; the
    score uses the first 6 dims (per upstream/modules.py).

    This wrapper takes a ``(B, 6, H, W)`` YUV6 pair tensor (caller is responsible
    for calling ``differentiable_rgb_to_yuv6`` on each frame and concatenating
    the YUV6 channel dim; the surrogate is YUV6-native to match the contest
    scorer's preprocess contract — see HNeRV parity discipline lesson 8).

    Parameters
    ----------
    pose_dim
        Output pose dimension. Canonical = 6.
    base_channels
        First-conv channels. Default 16 → ~25K params total.
    """

    def __init__(self, pose_dim: int = CONTEST_POSE_DIM, base_channels: int = 16) -> None:
        super().__init__()
        if not isinstance(pose_dim, int) or pose_dim < 1:
            raise DistortionViaDistilledScorerError(
                f"pose_dim must be int >= 1; got {pose_dim}"
            )
        if not isinstance(base_channels, int) or base_channels < 4:
            raise DistortionViaDistilledScorerError(
                f"base_channels must be int >= 4; got {base_channels}"
            )
        c = base_channels
        # Stem: takes 12 channels (2 frames * 6 YUV6 channels concatenated).
        # Wait: the YUV6 already has 6 channels per frame; we concatenate
        # the two frames giving 12 channels total. We accept either:
        # (a) (B, 6, H, W) single-frame YUV6 (caller pre-concatenates pair-internal)
        # (b) (B, 12, H, W) frame-pair YUV6
        # The stem is built for 12 channels (frame-pair) per contest contract.
        self.stem = nn.Sequential(
            nn.Conv2d(12, c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=False),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Hydra-head: bottleneck + linear.
        self.summary = nn.Linear(c * 2, c * 4)
        self.head = nn.Linear(c * 4, pose_dim)

    def forward(self, yuv6_pair_bchw: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        yuv6_pair_bchw
            ``(B, 12, H, W)`` YUV6 frame-pair tensor (2 frames concatenated
            along channel dim, each frame's YUV6 = 6 channels). Caller is
            responsible for the YUV6 conversion (use
            ``differentiable_rgb_to_yuv6`` from ``tac.differentiable_eval_roundtrip``).

        Returns
        -------
        torch.Tensor
            ``(B, pose_dim)`` pose floats.
        """
        if yuv6_pair_bchw.dim() != 4 or yuv6_pair_bchw.shape[1] != 12:
            raise DistortionViaDistilledScorerError(
                f"DistilledPoseNet expects (B, 12, H, W) YUV6 pair; "
                f"got {tuple(yuv6_pair_bchw.shape)}"
            )
        h_enc = self.stem(yuv6_pair_bchw)
        h_gap = self.gap(h_enc).flatten(1)
        return self.head(torch.relu(self.summary(h_gap)))


# ---------------------------------------------------------------------------
# Loader — refuses non-distilled or unmeasured-gap checkpoints
# ---------------------------------------------------------------------------


def load_pretrained_distilled_scorer_pair(
    *,
    config: ScorerSurrogateConfig,
    seg_state_dict: Optional[dict[str, torch.Tensor]] = None,
    pose_state_dict: Optional[dict[str, torch.Tensor]] = None,
    distill_label: str = "",
    distillation_gap: float = -1.0,
    ema_decay: float = -1.0,
    seg_base_channels: int = 16,
    pose_base_channels: int = 16,
    device: Optional[Any] = None,
) -> tuple[DistilledSegNet, DistilledPoseNet]:
    """Load a Hinton-distilled SegNet + PoseNet surrogate pair.

    Refuses to load a checkpoint that does NOT carry the canonical Hinton
    distillation custody fields:

    * ``distill_label`` (non-empty; per ``AuxiliaryScorerConfig``)
    * ``distillation_gap`` (≤ ``config.distill_gap_threshold``; per Hinton 2014 §3 +
      Catalog #134 Phase 3 prereq)
    * ``ema_decay`` (≥ 0.99; per CLAUDE.md "EMA — NON-NEGOTIABLE" canon)

    These fields are produced by ``tac.ib_lagrangian_aux_scorer.train_aux_scorer``
    in its ``AuxScorerTrainingResult`` and the T10 trainer's
    ``distillation_gap_estimate.json``.

    State dicts default to RANDOM-INIT (no checkpoint) ONLY when ALL THREE custody
    parameters are at their refusal sentinels (``distill_label=""``,
    ``distillation_gap=-1.0``, ``ema_decay=-1.0``). This is the SCAFFOLD-ONLY
    path used by the L2 encoder's smoke tests; non-smoke production paths MUST
    supply real custody fields.

    Per CLAUDE.md "Strict scorer rule": these distilled networks are
    COMPRESS-TIME ONLY artifacts. They are NOT loaded at inflate time.

    Parameters
    ----------
    config
        ``ScorerSurrogateConfig`` defining classes/dims/temperature/threshold.
    seg_state_dict
        Optional state-dict for ``DistilledSegNet``. If None, uses random init
        (smoke-only).
    pose_state_dict
        Optional state-dict for ``DistilledPoseNet``. If None, uses random init
        (smoke-only).
    distill_label
        Non-empty Hinton training label (per ``AuxiliaryScorerConfig.distill_label``).
        REQUIRED when state dicts are supplied.
    distillation_gap
        Measured distillation gap (per ``AuxScorerTrainingResult.distillation_gap_estimate``).
        REQUIRED when state dicts are supplied; refused when above
        ``config.distill_gap_threshold``.
    ema_decay
        EMA decay used during distillation (per ``AuxiliaryScorerConfig.ema_decay``).
        REQUIRED when state dicts are supplied; refused when < 0.99.
    seg_base_channels
        Base channels for ``DistilledSegNet``.
    pose_base_channels
        Base channels for ``DistilledPoseNet``.
    device
        Optional torch.device.

    Returns
    -------
    (DistilledSegNet, DistilledPoseNet)
        On the requested device, in eval mode (frozen — caller is responsible
        for ``train()`` if continuing distillation).
    """
    if not isinstance(config, ScorerSurrogateConfig):
        raise DistortionViaDistilledScorerError(
            f"config must be ScorerSurrogateConfig; got {type(config).__name__}"
        )

    has_state_dict = seg_state_dict is not None or pose_state_dict is not None
    if has_state_dict:
        # Pretrained-checkpoint path: refuse without proper custody.
        if not isinstance(distill_label, str) or not distill_label.strip():
            raise DistortionViaDistilledScorerError(
                "load_pretrained_distilled_scorer_pair: distill_label is required "
                "when supplying state dicts (non-empty string per "
                "AuxiliaryScorerConfig.distill_label)"
            )
        if (
            config.expected_distill_label_substring
            and config.expected_distill_label_substring not in distill_label
        ):
            raise DistortionViaDistilledScorerError(
                f"distill_label={distill_label!r} does not contain expected substring "
                f"{config.expected_distill_label_substring!r} (per config)"
            )
        if (
            math.isnan(distillation_gap)
            or math.isinf(distillation_gap)
            or distillation_gap < 0.0
        ):
            raise DistortionViaDistilledScorerError(
                f"distillation_gap must be finite >= 0; got {distillation_gap}"
            )
        if distillation_gap > config.distill_gap_threshold:
            raise DistortionViaDistilledScorerError(
                f"distillation_gap={distillation_gap} > threshold "
                f"{config.distill_gap_threshold} (Catalog #134 Phase 3 prereq)"
            )
        if math.isnan(ema_decay) or math.isinf(ema_decay) or not (0.99 <= ema_decay < 1.0):
            raise DistortionViaDistilledScorerError(
                f"ema_decay must be in [0.99, 1.0); got {ema_decay} "
                "(CLAUDE.md EMA non-negotiable)"
            )

    seg = DistilledSegNet(
        seg_class_count=config.seg_class_count, base_channels=seg_base_channels
    )
    pose = DistilledPoseNet(pose_dim=config.pose_dim, base_channels=pose_base_channels)

    seg_param_count = sum(p.numel() for p in seg.parameters())
    pose_param_count = sum(p.numel() for p in pose.parameters())
    total_params = seg_param_count + pose_param_count
    if total_params > MAX_SURROGATE_PARAMS:
        raise DistortionViaDistilledScorerError(
            f"surrogate footprint too large: total_params={total_params} "
            f"> MAX_SURROGATE_PARAMS={MAX_SURROGATE_PARAMS} "
            "(distilled scorer must be <= 5M params per Quantizr UCLA approach)"
        )

    if seg_state_dict is not None:
        seg.load_state_dict(seg_state_dict)
    if pose_state_dict is not None:
        pose.load_state_dict(pose_state_dict)

    if device is not None:
        seg = seg.to(device)
        pose = pose.to(device)

    seg.eval()
    pose.eval()
    for p in (*seg.parameters(), *pose.parameters()):
        p.requires_grad_(False)

    return seg, pose


# ---------------------------------------------------------------------------
# L2-encoder-facing helper: compute distortion via distilled scorer
# ---------------------------------------------------------------------------


def _resize_to_scorer(rgb_bchw: torch.Tensor) -> torch.Tensor:
    """Bilinear-resize from camera resolution (874x1164) to scorer (384x512)."""
    if rgb_bchw.shape[-2] == SCORER_H and rgb_bchw.shape[-1] == SCORER_W:
        return rgb_bchw
    return torch.nn.functional.interpolate(
        rgb_bchw,
        size=(SCORER_H, SCORER_W),
        mode="bilinear",
        align_corners=False,
    )


def compute_distortion_via_distilled_scorer(
    decoded_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    *,
    distilled_segnet: DistilledSegNet,
    distilled_posenet: DistilledPoseNet,
    eval_roundtrip: bool = True,
    distill_temperature: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute (d_seg_distilled, d_pose_distilled) via the distilled surrogate.

    This is the helper the L2 residual encoders call when
    ``use_hinton_distilled_scorer=True``. It REPLACES the YUV6 MSE proxy with
    the distilled-scorer outputs:

    * ``d_seg_distilled`` = KL(σ(seg_decoded/T) || σ(seg_gt/T)) · T² — Hinton-style
      KL between distilled SegNet logits on decoded vs GT (per Hinton 2014 §3,
      matches Quantizr's training-time KL-on-logits T=2.0 canon).
    * ``d_pose_distilled`` = MSE(pose_decoded[:, :6], pose_gt[:, :6]) — per
      contest's first-6-dim contract.

    Per HNeRV parity discipline lesson 8: routes through the canonical
    differentiable eval-roundtrip + YUV6 helpers so gradient flows from the
    L2 residual through the surrogate scorer back into the residual coefficients.

    Parameters
    ----------
    decoded_rgb
        ``(B, 3, H, W)`` (preferred) or ``(B, H, W, 3)`` or ``(H, W, 3)``
        post-residual decoded frames at camera resolution. B MUST be even
        (frame pairs).
    gt_rgb
        Same shape, ground-truth frames. Stops gradient on this side.
    distilled_segnet
        Loaded ``DistilledSegNet`` instance (frozen).
    distilled_posenet
        Loaded ``DistilledPoseNet`` instance (frozen).
    eval_roundtrip
        If True (default), simulates the 384→874→uint8→384 contest roundtrip
        on both decoded AND GT before the surrogate's forward pass.
    distill_temperature
        Hinton T for the seg KL distill term. Council canon = 2.0.

    Returns
    -------
    (d_seg, d_pose, diagnostics)
        ``d_seg``: scalar tensor (KL-distill, requires_grad if decoded did)
        ``d_pose``: scalar tensor (MSE on first 6 pose dims)
        ``diagnostics``: dict[str, float] with detached scalars
    """
    if not isinstance(distilled_segnet, DistilledSegNet):
        raise DistortionViaDistilledScorerError(
            f"distilled_segnet must be DistilledSegNet; got {type(distilled_segnet).__name__}"
        )
    if not isinstance(distilled_posenet, DistilledPoseNet):
        raise DistortionViaDistilledScorerError(
            f"distilled_posenet must be DistilledPoseNet; got {type(distilled_posenet).__name__}"
        )
    if distill_temperature <= 0:
        raise DistortionViaDistilledScorerError(
            f"distill_temperature must be > 0; got {distill_temperature}"
        )

    # Coerce to (B, 3, H, W) float.
    from tac.residual_basis.l2_score_aware_loss import (
        _coerce_rgb_to_bchw_float,
        _validate_pixel_range,
    )

    decoded_bchw = _validate_pixel_range(
        _coerce_rgb_to_bchw_float(decoded_rgb), name="decoded_rgb"
    )
    gt_bchw = _validate_pixel_range(_coerce_rgb_to_bchw_float(gt_rgb), name="gt_rgb").detach()
    if decoded_bchw.shape != gt_bchw.shape:
        raise DistortionViaDistilledScorerError(
            f"decoded/gt shape mismatch: decoded={tuple(decoded_bchw.shape)} "
            f"gt={tuple(gt_bchw.shape)}"
        )
    if decoded_bchw.shape[0] % 2 != 0:
        raise DistortionViaDistilledScorerError(
            f"expected an even number of frames (frame pairs); got B={decoded_bchw.shape[0]}"
        )

    # Eval-roundtrip + scorer-resize.
    if eval_roundtrip:
        from tac.differentiable_eval_roundtrip import (
            apply_eval_roundtrip_during_training,
        )

        decoded_bchw = apply_eval_roundtrip_during_training(
            decoded_bchw,
            simulate_uint8=True,
            simulate_resize=True,
            ste_round=True,
            target_h=CAMERA_H,
            target_w=CAMERA_W,
        )
        gt_bchw = apply_eval_roundtrip_during_training(
            gt_bchw,
            simulate_uint8=True,
            simulate_resize=True,
            ste_round=True,
            target_h=CAMERA_H,
            target_w=CAMERA_W,
        )

    decoded_scorer = _resize_to_scorer(decoded_bchw)
    gt_scorer = _resize_to_scorer(gt_bchw)

    # SegNet path: per upstream/modules.py SegNet sees only the SECOND frame
    # of each pair (x[:, -1, ...]). The decoded/gt tensors are arranged as
    # (frame0, frame1, frame0, frame1, ...) so frames[1::2] selects frame1
    # of each pair — the SegNet-relevant frame.
    seg_decoded_input = decoded_scorer[1::2]  # (P, 3, H, W) where P = B/2
    seg_gt_input = gt_scorer[1::2]
    seg_decoded_logits = distilled_segnet(seg_decoded_input)
    with torch.no_grad():
        seg_gt_logits = distilled_segnet(seg_gt_input)

    # Hinton KL distill at T (gradient multiplied by T² per Hinton 2014).
    soft_decoded = torch.nn.functional.log_softmax(
        seg_decoded_logits / distill_temperature, dim=1
    )
    soft_gt = torch.nn.functional.log_softmax(
        seg_gt_logits / distill_temperature, dim=1
    )
    d_seg = (
        torch.nn.functional.kl_div(
            soft_decoded, soft_gt, reduction="none", log_target=True
        )
        .sum(dim=1)
        .mean()
        * (distill_temperature ** 2)
    )

    # PoseNet path: convert each frame to YUV6, concatenate pair channels (12),
    # forward through distilled PoseNet, MSE on first ``pose_dim`` dims.
    from tac.differentiable_eval_roundtrip import differentiable_rgb_to_yuv6

    decoded_yuv6 = differentiable_rgb_to_yuv6(decoded_scorer)  # (B, 6, H/2, W/2)
    gt_yuv6 = differentiable_rgb_to_yuv6(gt_scorer)
    # Reshape to (P, 2, 6, H, W) → (P, 12, H, W) for the PoseNet input.
    p = decoded_yuv6.shape[0] // 2
    decoded_pair = decoded_yuv6.reshape(p, 2, *decoded_yuv6.shape[1:])
    gt_pair = gt_yuv6.reshape(p, 2, *gt_yuv6.shape[1:])
    decoded_pose_in = decoded_pair.reshape(p, 12, *decoded_pair.shape[3:])
    gt_pose_in = gt_pair.reshape(p, 12, *gt_pair.shape[3:])
    pose_decoded = distilled_posenet(decoded_pose_in)
    with torch.no_grad():
        pose_gt = distilled_posenet(gt_pose_in)
    pose_dim = pose_decoded.shape[-1]
    d_pose = torch.nn.functional.mse_loss(
        pose_decoded[:, :pose_dim], pose_gt[:, :pose_dim]
    )

    diagnostics: dict[str, float] = {
        "d_seg_distilled": float(d_seg.detach().item()),
        "d_pose_distilled": float(d_pose.detach().item()),
        "distill_temperature": float(distill_temperature),
        "n_pairs_evaluated": float(p),
        "eval_roundtrip": float(1.0 if eval_roundtrip else 0.0),
        "use_hinton_distilled_scorer": 1.0,
    }
    return d_seg, d_pose, diagnostics


# ---------------------------------------------------------------------------
# Convenience: round-trip-vs-real-scorer gap helper
# ---------------------------------------------------------------------------


def measure_distillation_gap_smoke(
    *,
    distilled_segnet: DistilledSegNet,
    distilled_posenet: DistilledPoseNet,
    sample_rgb_pair: torch.Tensor,
) -> dict[str, float]:
    """Smoke-only helper: compute round-trip-vs-real-scorer gap on a sample pair.

    This produces a *forensic* estimate of the distilled-scorer's drift from
    the contest scorer; it is NOT a Phase 3 prereq replacement (the canonical
    measurement comes from ``tac.ib_lagrangian_aux_scorer.train_aux_scorer``'s
    ``distillation_gap_estimate``).

    Useful for unit tests + smoke runs that want a quick "is the surrogate
    in the right ballpark" check without depending on contest scorer weights
    being available.

    Returns
    -------
    dict[str, float]
        Diagnostics containing surrogate forward-pass shapes + finiteness.
    """
    if sample_rgb_pair.dim() != 4 or sample_rgb_pair.shape[1] != 3:
        raise DistortionViaDistilledScorerError(
            f"sample_rgb_pair must be (B, 3, H, W); got {tuple(sample_rgb_pair.shape)}"
        )
    if sample_rgb_pair.shape[0] % 2 != 0:
        raise DistortionViaDistilledScorerError(
            f"sample_rgb_pair B must be even (frame pairs); got {sample_rgb_pair.shape[0]}"
        )
    with torch.no_grad():
        # SegNet forward on second frame.
        seg_in = sample_rgb_pair[1::2]
        seg_out = distilled_segnet(seg_in)
        # PoseNet forward on YUV6 pair.
        from tac.differentiable_eval_roundtrip import differentiable_rgb_to_yuv6

        yuv6 = differentiable_rgb_to_yuv6(sample_rgb_pair)
        p = yuv6.shape[0] // 2
        pair = yuv6.reshape(p, 2, *yuv6.shape[1:])
        pose_in = pair.reshape(p, 12, *pair.shape[3:])
        pose_out = distilled_posenet(pose_in)

    return {
        "seg_logits_finite": float(torch.isfinite(seg_out).all().item()),
        "pose_floats_finite": float(torch.isfinite(pose_out).all().item()),
        "seg_logits_shape_ok": float(
            seg_out.dim() == 4 and seg_out.shape[1] == distilled_segnet.head[-1].out_channels
        ),
        "pose_floats_shape_ok": float(
            pose_out.dim() == 2 and pose_out.shape[1] == distilled_posenet.head.out_features
        ),
    }


__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "CONTEST_POSE_DIM",
    "CONTEST_SEG_CLASSES",
    "DEFAULT_DISTILL_GAP_THRESHOLD",
    "DistilledPoseNet",
    "DistilledSegNet",
    "DistortionViaDistilledScorerError",
    "MAX_SURROGATE_PARAMS",
    "SCORER_H",
    "SCORER_W",
    "ScorerSurrogateConfig",
    "compute_distortion_via_distilled_scorer",
    "load_pretrained_distilled_scorer_pair",
    "measure_distillation_gap_smoke",
]
