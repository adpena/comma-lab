# SPDX-License-Identifier: MIT
"""Resumable curriculum driver for the P2 torch-vehicle (vendored PR95 HNeRV-Muon).

This is the ADAPTER's heart. The vendored ``stages/common.py:train_stage`` is a
monolithic loop with NO resume, NO telemetry, and a HARDCODED ``base_channels=36``
(in 3 call sites) — it cannot run the base_ch=20 rate-win config and a death
loses the whole in-flight stage. We do NOT edit the pristine vendored source
(CLAUDE.md "Forbidden in-place edits to public PR intake clones"); instead we
RE-DRIVE the vendored PRIMITIVES (``HNeRVDecoder`` / ``Muon`` /
``partition_params_for_muon`` / the seg-loss fns / ``ema_update`` / ``apply_qat``
/ ``cat_entropy_v2`` / ``build_archive`` / ``parse_archive``) unchanged, with:

* ``base_channels`` threaded into ``HNeRVDecoder(...)`` (the FINDING-1 fix —
  parametrization lives in the driver, NOT in the pristine source);
* COMPLETE checkpoint/resume (decoder + latents + EMA shadow + AdamW state +
  Muon momentum + LR-scheduler state + RNG + curriculum position), so a death
  costs ≤1 checkpoint interval — verified bit-identical by the kill+restart
  test (``tests/test_driver_resume.py``);
* per-epoch telemetry to durable JSONL (the "Max observability" non-negotiable);
* BEST-checkpoint-by-canonical-score tracking from the EMA shadow (the EMA
  non-negotiable — inference/export bytes are the shadow).

The per-step math is a 1:1 port of ``common.py`` (eval_roundtrip bicubic↑→
bilinear↓→uint8-STE, joint clip, ``ema_update`` after each step, sigma weight
noise via QAT, C1a entropy) — verified faithful in ``tests/test_driver_faithful.py``.

The driver is SCORER-INJECTABLE: production wires the real frozen SegNet/PoseNet
via :class:`RealScorerContext` (``precompute_targets`` + ``evaluate_decoder``);
tests inject a tiny synthetic frozen scorer so the resume round-trip (which is
architecture-AGNOSTIC) is fast + deterministic, exactly as the MLX
``test_checkpoint`` does.

Authority: torch-CPU TRUSTED (CLAUDE.md "local CPU + MLX GPU good"); NO MPS. The
in-loop d_seg/d_pose are ``[contest-CPU advisory]`` NON-PROMOTABLE until the
byte-closed archive is run through ``upstream/evaluate.py``.
"""

from __future__ import annotations

import io
import math
import struct
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.torch_vehicle.checkpoint import (
    TorchCheckpointPosition,
    checkpoint_exists,
    load_checkpoint,
    read_manifest,
    save_checkpoint,
    write_done_marker,
)
from tac.torch_vehicle.telemetry import EpochRecord, TelemetryWriter

if TYPE_CHECKING:  # StageSpec used only in annotations at module level
    from tac.torch_vehicle.curriculum import StageSpec

_EVAL_H, _EVAL_W = 384, 512
_SEG_NUM_CLASSES = 5  # contest SegNet head (upstream/modules.py: smp.Unet classes=5)
_TRACK_A_D2_CONSERVATIVE_BYTE_TARGET = 2731.0
# Lever-3 v2 FiLM learning-rate CAP (the #118-SEALED stability fix). The residual
# pose-FiLM (``pose_mlp`` + ``film_resid``) is given a DEDICATED AdamW param group at
# ``min(stage.adamw_lr, _FILM_LR_CAP)`` and is EXCLUDED from the Muon partition — so the
# FiLM never trains at the full ``adamw_lr`` (the lr=1e-2 transient-overshoot v2's #118
# review caught) NOR under Muon's orthogonalized SGD (which would bypass the cap entirely,
# the §A finding of from0_deployment_fullstack_adversarial_review_20260613). Only applied
# when ``pose_film_enabled`` → the levers-OFF basin is byte-identical.
_FILM_LR_CAP = 1e-3
# The wrapper-level FiLM param-name prefixes per pose-FiLM version (v1 stem-injection vs
# v2 residual-rgb0). Used to (a) route FiLM params to the capped AdamW group and (b) split
# the EMA shadow at export. v1 = the single ``_PoseFiLM`` module; v2 = the cond MLP + the
# rgb_0 residual FiLM.
_FILM_PARAM_PREFIXES = {1: ("pose_film.",), 2: ("pose_mlp.", "film_resid.")}


def _segnet_logit_margin_map(seg_out: torch.Tensor) -> torch.Tensor:
    """Per-pixel SegNet ``top1 − top2`` logit margin of the PREDICTION logits — the
    Lever-5 boundary sensitivity map (detached: it is a per-pixel WEIGHT, a Catalog
    #341 Tier A proxy, NOT a score claim).

    ``seg_out`` is ``(B, 5, H, W)`` SegNet logits of the DECODED frame (already
    forwarded for the seg loss — no extra scorer pass). Small margin = decision
    frontier (argmax-prone-to-flip); large margin = confident interior. Mirrors
    ``tac.substrates.d1_segnet_margin_polytope.margin_map.compute_logit_margin_map``
    (top2 = the two largest logits per pixel) but reads the already-computed
    prediction logits in-place (the canonical helper re-runs the scorer; we reuse)."""
    top2, _ = torch.topk(seg_out.detach(), k=2, dim=1, largest=True, sorted=True)
    return (top2[:, 0] - top2[:, 1]).clamp_min(0.0)  # (B, H, W) >= 0


def _warmup_wrap(
    cosine_lambda: Callable[[int], float],
    *,
    warmup_frac: float,
    stage_epochs: int,
    start_ratio: float,
) -> Callable[[int], float]:
    """Wrap a per-stage cosine ``lr_lambda`` with a leading LINEAR warmup (the E#5
    stage-transition pose-kick fix — warmup-after-restart).

    DEFAULT-PRESERVING: ``warmup_frac <= 0`` returns ``cosine_lambda`` UNCHANGED (the
    same object) so the LambdaLR sees the byte-identical legacy schedule — every stage
    starts at the cosine peak exactly as before.

    When ``warmup_frac > 0``: the first ``w = ceil(warmup_frac · stage_epochs)`` epochs
    (>= 1) ramp the LR multiplier LINEARLY from ``start_ratio`` (× peak) at epoch 0 up to
    ``cosine_lambda(w)`` (the cosine value at the warmup-end epoch) at epoch ``w``. From
    epoch ``w`` onward the wrapped lambda IS ``cosine_lambda`` — so the warmup ends
    exactly where the cosine would be at that epoch (C0-continuous; no jump). The ramp is
    a multiplier on the SAME ``peak_lr`` the LambdaLR base captured, so it eases the
    shared trunk in at the boundary instead of slamming it to peak. ``w`` is clamped to
    ``stage_epochs - 1`` so at least one cosine epoch remains (warmup_frac <= 0.5 already
    guarantees this for stage_epochs >= 2; the clamp covers a 1-epoch stage).
    """
    if warmup_frac <= 0.0:
        return cosine_lambda
    w = max(1, math.ceil(warmup_frac * stage_epochs))
    w = min(w, max(1, stage_epochs - 1))
    target_at_w = cosine_lambda(w)  # the cosine value the warmup ramps UP to (no jump)

    def wrapped(epoch: int) -> float:
        if epoch >= w:
            return cosine_lambda(epoch)
        # linear ramp start_ratio → target_at_w over [0, w]
        frac = epoch / w
        return start_ratio + (target_at_w - start_ratio) * frac

    return wrapped


def _seg_loss_for_spec(
    spec: StageSpec,
    seg_out: torch.Tensor,
    seg_targets_hard: torch.Tensor,
    *,
    temperature: float | None = None,
) -> torch.Tensor:
    """Score-domain seg loss router (Lever 2 + the L2 anneal hook + Lever 5 margin
    weight) — DEFAULT-PRESERVING.

    ``spec.seg_surrogate is None`` (the default for every vendored-projected
    ``StageSpec``) returns ``spec.seg_loss_fn(seg_out, seg_targets_hard)`` EXACTLY
    as the legacy non-routed call did — byte-for-byte identical, so a basin that
    resumes onto this code is unchanged (verified by
    ``test_default_seg_surrogate_is_byte_identical_to_vendored_call`` +
    ``test_default_seg_surrogate_gradient_is_byte_identical`` in
    ``tests/test_seg_surrogate_lever.py``). The ``temperature`` / margin-weight
    extensions are NO-OPS on this default path (the vendored CE path returns a
    scalar, not a per-pixel tensor — there is no temperature or per-pixel handle).

    When ``spec.seg_surrogate`` is a surrogate name (``"soft_cosine"`` /
    ``"fisher_rao"`` / ``"sinkhorn"``) the seg term is routed through the
    differentiable score-domain d_seg surrogate
    :func:`tac.losses.core.segnet_surrogate_per_pixel`. The contest d_seg is the
    per-pixel SegNet ARGMAX-DISAGREEMENT rate; the surrogate concentrates the
    gradient where the argmax actually flips (HNeRV parity L6), unlike CE which
    spends capacity on confident-interior pixels the argmax already gets right.

    Lever-2 anneal (``temperature``): when the driver passes a per-epoch
    ``temperature`` (the cosine anneal from ``spec.seg_temperature`` toward
    ``spec.seg_temperature_end``) it OVERRIDES the static ``spec.seg_temperature``
    for the PREDICTION softmax. ``temperature is None`` (the default) uses the static
    ``spec.seg_temperature`` — so a caller that does not pass it is unchanged.

    Lever-5 margin weight (``spec.margin_weight_tau``): when set, the per-pixel
    surrogate is weighted by ``exp(−margin/τ)`` (the prediction's SegNet logit
    margin) BEFORE the mean — boundary-prone pixels (small margin) get more gradient,
    confident-interior pixels ~0. ``None`` (the default) leaves the surrogate
    unweighted (Lever-2 baseline).

    GT side-information: the curriculum caches the HARD per-pixel target argmax
    (``seg_targets_hard``, ``(B, 384, 512)`` int64), not GT logits. The score-domain
    surrogate needs a GT class distribution; we build SHARP one-hot GT LOGITS from
    the hard target (argmax(GT logits) == the cached hard class), so the GT stays
    hard while only the PRED is temperature-softened (see the inline comment below).
    With one-hot GT, soft-cosine ``1 - Σ_c softmax(pred/T)_c · onehot_c =
    1 - softmax(pred/T)[gt]`` is EXACTLY the per-pixel "pred's probability mass NOT
    on the GT-argmax class" — the differentiable argmax-flip surrogate the memo
    specifies, cache-free (argmax is all the surrogate's per-pixel target needs).
    """
    if spec.seg_surrogate is None:
        # Legacy vendored path — UNCHANGED. Do not touch (basin-resume safety). The
        # anneal/margin extensions never touch this branch (the default path takes
        # temperature=None and margin_weight_tau=None and returns the raw CE scalar).
        return spec.seg_loss_fn(seg_out, seg_targets_hard)

    if spec.seg_surrogate == "lovasz":
        # Lovász-softmax (Berman 2018): the TIGHTEST CONVEX UPPER BOUND on the contest's
        # argmax-disagreement (IoU) metric — a more direct d_seg surrogate than soft_cosine
        # (which is margin-resonant frame fidelity). Returns a SCALAR multi-class hinge
        # (mean one-vs-rest), so it is NOT per-pixel: the Lever-5 margin weight does not
        # apply (Lovász already concentrates on the argmax-boundary by construction). T=1
        # standard softmax (the convex-envelope geometry is defined on the plain simplex;
        # temperature-sharpening would distort it). Pose/FiLM are unchanged — this swaps
        # ONLY the seg term.
        from tac.lovasz_hinge import lovasz_hinge_mask_distortion

        pred_probs = torch.softmax(seg_out, dim=1)  # (B, 5, H, W) standard simplex
        gt_probs = (
            F.one_hot(seg_targets_hard.long(), num_classes=_SEG_NUM_CLASSES)
            .permute(0, 3, 1, 2)
            .to(seg_out.dtype)
        )  # (B, 5, H, W) one-hot
        return lovasz_hinge_mask_distortion(
            pred_probs, gt_probs, num_classes=_SEG_NUM_CLASSES
        )

    from tac.losses.core import segnet_surrogate_per_pixel

    # ACCELERATOR PROBE 1 — the flip-targeting margin-hinge. It is defined on RAW
    # logits (the contest d_seg is a hard argmax; there is no softmax/temperature to
    # tune), so it skips the one-hot-GT-logit + temperature plumbing the soft-cosine /
    # fisher-rao / sinkhorn surrogates need. It needs only the HARD GT class index,
    # which the curriculum already caches (``seg_targets_hard``).
    if spec.seg_surrogate == "margin_hinge":
        from tac.losses.core import segnet_margin_hinge_per_pixel

        per_pixel = segnet_margin_hinge_per_pixel(
            seg_out,
            seg_targets_hard,
            margin_target=float(spec.seg_margin_hinge_target),
            num_classes=_SEG_NUM_CLASSES,
        )  # (B, H, W)
        per_pixel = _apply_seg_levers(spec, per_pixel, seg_out, seg_targets_hard)
        return per_pixel

    temp = float(temperature) if temperature is not None else float(spec.seg_temperature)
    # seg_out: (B, 5, H, W) logits. seg_targets_hard: (B, H, W) int64 class indices.
    # Build SHARP one-hot GT LOGITS (B, 5, H, W): the GT-argmax class gets a large
    # positive logit, all others ~0. ``softmax(gt_logits / T)`` is then ~one-hot for
    # ANY reasonable T (at the memo's coldest T=0.05, 30/0.05=600 → exactly one-hot),
    # so the GT is NOT temperature-softened (it is hard, as the contest d_seg is a
    # hard argmax) while ``temp`` softens ONLY the PREDICTION. With one-hot GT,
    # soft_cosine reduces to ``1 - softmax(pred/T)[gt]`` EXACTLY — the differentiable
    # argmax-flip surrogate the memo specifies (verified bit-equal to a hand-computed
    # reference at T∈{1.0, 0.5, 0.1} in the lever test). Passing GT as LOGITS
    # (``gt_already_probs=False``) — NOT cached probs — is what lets the surrogate
    # honor non-unit ``temperature`` (the helper refuses cached probs at T≠1 because
    # cached probs are softmax@T=1; a one-hot logit has no such constraint). This
    # generalizes to fisher_rao / sinkhorn (same one-hot GT logits).
    _ONEHOT_GT_LOGIT = 30.0  # softmax(30·onehot/T) ≈ one-hot for all T ≥ ~0.02
    gt_onehot = F.one_hot(
        seg_targets_hard.long(), num_classes=_SEG_NUM_CLASSES
    )  # (B, H, W, 5)
    gt_logits = (
        gt_onehot.permute(0, 3, 1, 2).to(seg_out.dtype) * _ONEHOT_GT_LOGIT
    )  # (B, 5, H, W) sharp one-hot logits
    per_pixel = segnet_surrogate_per_pixel(
        seg_out,
        gt_logits,
        surrogate=spec.seg_surrogate,
        temperature=temp,
        gt_already_probs=False,
        num_classes=_SEG_NUM_CLASSES,
    )  # (B, H, W)
    return _apply_seg_levers(spec, per_pixel, seg_out, seg_targets_hard)


def _apply_seg_levers(
    spec: StageSpec,
    per_pixel: torch.Tensor,
    seg_out: torch.Tensor,
    seg_targets_hard: torch.Tensor,
) -> torch.Tensor:
    """Apply the per-pixel seg levers (road↔lane class emphasis + Lever-5 margin
    weight) and reduce to a scalar. Shared by every per-pixel surrogate path
    (soft_cosine / fisher_rao / sinkhorn / margin_hinge).

    Road↔lane emphasis (ACCELERATOR PROBE 1): Probe E found ~64% of d_seg flips are
    road↔lane; ``spec.road_lane_emphasis > 1`` multiplies the per-pixel loss by a
    (mean-1-normalised) per-class weight emphasising road (0) + lane (1), concentrating
    the flip-targeting gradient on the dominant flip class-pair. ``1.0`` (default) is
    a no-op.

    Lever-5 margin weight: ``exp(−margin/τ)`` over the DECODED-frame top1−top2 logit
    margin — boundary-prone pixels get MORE gradient, confident interior ~0.
    """
    if getattr(spec, "road_lane_emphasis", 1.0) and float(spec.road_lane_emphasis) != 1.0:
        from tac.losses.core import _apply_class_weights, road_lane_emphasis_class_weights

        cw = road_lane_emphasis_class_weights(
            emphasis=float(spec.road_lane_emphasis),
            num_classes=_SEG_NUM_CLASSES,
            device=per_pixel.device,
            dtype=per_pixel.dtype,
        )
        # GT-argmax binned per-pixel weight (L1-normalised to mean 1 internally, so
        # absolute magnitude is preserved). seg_targets_hard is (B, H, W) int64; build
        # the (B, C, H, W) one-hot the helper's argmax expects.
        gt_onehot = F.one_hot(
            seg_targets_hard.long(), num_classes=_SEG_NUM_CLASSES
        ).permute(0, 3, 1, 2)
        per_pixel = _apply_class_weights(per_pixel, gt_onehot, cw, gt_already_probs=False)
    if spec.margin_weight_tau is not None:
        # Lever 5: weight per-pixel by exp(-margin/tau) (boundary-prone pixels get
        # more gradient). The margin is a DETACHED per-pixel weight (Tier A proxy);
        # it shapes WHERE the surrogate's gradient lands, it is not itself optimized.
        margin = _segnet_logit_margin_map(seg_out)  # (B, H, W) detached
        tau = max(float(spec.margin_weight_tau), 1e-6)
        weight = torch.exp(-margin / tau)  # (B, H, W) in (0, 1], 1 at the boundary
        if getattr(spec, "margin_weight_renorm", False):
            # WS-A/M7 fix (opt-in): renormalize by Σweight → a TRUE weighted mean that
            # REDISTRIBUTES gradient to the boundary WITHOUT shrinking total magnitude as
            # margins grow (stops the compounding decay vs T-anneal + LR-decay). Off below.
            return (per_pixel * weight).sum() / weight.sum().clamp_min(1e-6)
        # Mean of (per_pixel · weight): a weighted average that concentrates the loss
        # mass on small-margin pixels. (Not re-normalized by Σweight — the absolute
        # magnitude scaling is folded into the seg_weight Lagrangian coefficient.)
        return (per_pixel * weight).mean()
    return per_pixel.mean()


def _adaptive_do_pose(
    epoch: int,
    pose_floor: float | None,
    last_pose_mse: float | None,
    hist: list[float],
    *,
    tol: float,
    k_max: int,
    last_pose_epoch: int,
) -> bool:
    """Pure APGC decision: should the (expensive) pose path be COMPUTED this epoch?

    A closed-loop controller that holds d_pose at its (moving) floor with minimum
    gradient spend, grounded in the equimarginal/constraint principle: pay the pose
    path only when the marginal score harm of leaving pose un-corrected exceeds the
    wall-clock saved by skipping. The decision (NO side effects — testable in isolation):

      * No floor yet / first epoch  → COMPUTE (establish the floor).
      * ``dev = last_pose_mse / pose_floor`` (relative deviation from the running floor).
      * ``rising`` = the trend over ``hist`` is positive (hist[-1] > hist[0]) — the
        derivative term that catches drift BEFORE it breaches the band.
      * dev > 1+tol (out of the deadband) OR rising  → DRIFT-ARREST: compute every epoch.
      * else (solidly at floor): widen the cadence proportionally toward ``k_max`` —
        ``slack`` is 1 at the exact floor and 0 at the band edge, so ``k`` ramps
        1 → k_max as pose sits deeper inside the band; compute only on the cadence epoch.
      * MEASUREMENT-FLOOR: if ``epoch - last_pose_epoch >= k_max`` force a compute
        regardless — a skipped epoch does NOT measure pose (the pose forward is fused
        with the backward), so this bounds how long the controller may run blind.

    Returns True (compute pose) / False (skip, flow the SegNet-only cotangent)."""
    if pose_floor is None or last_pose_mse is None:
        do_pose = True
    else:
        # Running floor is the MIN of all computed pose_mse, so dev >= 1 by construction
        # for the current sample; a fresh new minimum gives dev == 1.0 (deepest in band).
        dev = last_pose_mse / pose_floor if pose_floor > 0.0 else float("inf")
        rising = len(hist) >= 2 and hist[-1] > hist[0]
        if dev > 1.0 + tol or rising:
            do_pose = True  # drift-arrest — correct every epoch until back at floor
        else:
            slack = (1.0 + tol - dev) / tol  # 1.0 at floor → 0.0 at the band edge
            slack = max(0.0, min(1.0, slack))
            k = max(1, min(k_max, round(1 + slack * (k_max - 1))))
            do_pose = (epoch % k) == 0
    # Measurement-floor: never run blind longer than k_max epochs between pose computes.
    if (epoch - last_pose_epoch) >= k_max:
        do_pose = True
    return do_pose


def _split_three_section_archive(archive: bytes) -> tuple[bytes, bytes, bytes, bytes]:
    """Return ``(meta_brotli, decoder_blob, latents_brotli, trailing)``.

    The vendored archive grammar is exactly three length-prefixed sections. D2 uses
    this to keep the meta and latent sections byte-for-byte while replacing only the
    decoder blob with the variable-level codec output.
    """
    buf = io.BytesIO(archive)
    sections: list[bytes] = []
    for _ in range(3):
        raw_len = buf.read(4)
        if len(raw_len) != 4:
            raise ValueError("truncated vendored archive while reading section length")
        sec_len = struct.unpack("<I", raw_len)[0]
        section = buf.read(sec_len)
        if len(section) != sec_len:
            raise ValueError("truncated vendored archive while reading section payload")
        sections.append(section)
    trailing = buf.read()
    return sections[0], sections[1], sections[2], trailing


def _join_three_section_archive(
    meta_brotli: bytes, decoder_blob: bytes, latents_brotli: bytes, trailing: bytes = b""
) -> bytes:
    out = io.BytesIO()
    for section in (meta_brotli, decoder_blob, latents_brotli):
        out.write(struct.pack("<I", len(section)))
        out.write(section)
    out.write(trailing)
    return out.getvalue()


def _normalize_variable_level_rd_table(
    rd_table: dict[str, Any],
) -> dict[str, dict[int, tuple[float, float]]]:
    """Normalize persisted JSON RD tables to the allocator's typed shape."""
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for tensor, raw_curve in rd_table.items():
        if not isinstance(raw_curve, dict):
            raise ValueError(f"RD curve for {tensor!r} must be an object")
        curve: dict[int, tuple[float, float]] = {}
        for raw_level, raw_pair in raw_curve.items():
            level = int(raw_level)
            if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
                raise ValueError(f"RD point {tensor!r}/{raw_level!r} must be [bytes, dist]")
            curve[level] = (float(raw_pair[0]), float(raw_pair[1]))
        if 127 not in curve:
            raise ValueError(f"RD curve for {tensor!r} is missing the 127 baseline")
        if curve[127] != (0.0, 0.0):
            raise ValueError(f"RD curve for {tensor!r} has nonzero 127 baseline: {curve[127]}")
        out[str(tensor)] = curve
    if not out:
        raise ValueError("variable-level waterfill RD table is empty")
    return out


def _sensitivity_for_codec_weight_keys(
    sensitivity: dict[str, float] | None,
    weight_keys: list[str],
) -> dict[str, float] | None:
    """Re-key the online score-sensitivity EMA onto the codec's WEIGHT state-dict keys.

    THE SPINE RECONCILIATION (the single-source contract). ``accumulate_tensor_
    sensitivity`` keys the EMA by ``decoder.named_modules()`` MODULE names (e.g.
    ``stem`` / ``blocks.0``, or ``decoder.stem`` / ``decoder.blocks.0`` when the
    decoder is the FiLM wrapper). ``apply_score_aware_qat`` (the training-time QAT
    bits) looks up by those SAME module names, so QAT consumes the EMA correctly. But
    the variable-level EXPORT codec checks the WEIGHT state-dict keys
    (``<module>.weight``) — a DIFFERENT namespace. Without this translation the codec
    lookup ``sensitivity.get("blocks.0.weight")`` MISSES the EMA key ``"blocks.0"`` →
    every tensor reads 0.0 → uniform → the rate-attack SILENTLY no-ops even when
    enabled (the QAT trained a coarse grid the export never reproduced). This rebinds
    so the SAME ``||∂S/∂w_t||`` tensor drives BOTH the QAT bits AND the codec levels —
    the spine the bind-all spec calls for ("compute sensitivity ONCE and fan out").

    For each codec weight key ``<module>.weight`` the EMA value is resolved by trying,
    in order: the exact key (already codec-keyed — the test-API convention), the
    module name ``<module>`` (the no-FiLM EMA convention), and the FiLM-wrapper-
    prefixed module name ``decoder.<module>`` (the FiLM EMA convention). Returns
    ``None`` when ``sensitivity`` is ``None`` OR no codec weight key resolves to any
    EMA entry (so the codec's own ``None`` → uniform → byte-identical-to-vendored
    default-preserving path is preserved exactly — NEVER a fabricated allocation).
    """
    if sensitivity is None:
        return None
    out: dict[str, float] = {}
    for wkey in weight_keys:
        module = wkey[: -len(".weight")] if wkey.endswith(".weight") else wkey
        for candidate in (wkey, module, f"decoder.{module}"):
            if candidate in sensitivity:
                out[wkey] = float(sensitivity[candidate])
                break
    return out or None


class _SimulatedDeath(Exception):
    """Raised by the test-only ``_stop_after_global_epoch`` hook AFTER a checkpoint
    lands, to simulate a SIGKILL/OOM mid-run for the in-process resume test."""


# ---------------------------------------------------------------------------
# Scorer context protocol: production = real frozen SegNet/PoseNet; test = synthetic
# ---------------------------------------------------------------------------
class ScorerContext(Protocol):
    """The per-run frozen-scorer + GT-targets surface the driver trains against.

    Production binds the REAL contest scorer (``precompute_targets`` →
    ``distortion_net``, ``seg_targets_hard``, ``pose_targets``; GT via
    ``frame_utils.yuv420_to_rgb``). Tests bind a synthetic frozen scorer with the
    same interface so the resume round-trip is fast.
    """

    n_pairs: int
    seg_targets_hard: torch.Tensor  # (n_pairs, 384, 512) int64
    pose_targets: torch.Tensor  # (n_pairs, 6) float32

    def seg_pose_forward(
        self, decoded_bhwc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the frozen scorer on roundtripped decoder output.

        ``decoded_bhwc`` is (B, 2, 384, 512, 3) float in [0,255] (post eval-
        roundtrip). Returns ``(seg_logits, pose_pred6)`` where ``seg_logits`` is
        (B*?, 5, 384, 512) matching ``seg_targets_hard`` indexing and
        ``pose_pred6`` is (B, 6) — so the driver's loss matches ``common.py``.
        """
        ...

    def exact_eval(
        self, ema_decoder: nn.Module, ema_latents: torch.Tensor, archive_bytes: int
    ) -> dict[str, float]:
        """Canonical d_seg / d_pose / rate / score on the EMA shadow (the score
        the BEST checkpoint tracks). Production routes through the real
        ``evaluate_decoder`` + ``compute_score`` on the parse-back archive."""
        ...


@dataclass
class TorchVehicleConfig:
    """Run-level config (architecture + budget + IO); the per-stage schedule
    comes from :func:`tac.torch_vehicle.curriculum.build_curriculum`."""

    base_channels: int = 20
    latent_dim: int = 28
    out_dir: Path = Path("experiments/results/torch_vehicle_run")
    checkpoint_every_epochs: int = 1  # a death costs <= this many epochs
    total_epoch_budget: int | None = None
    ema_decay: float = 0.999
    eval_every: int | None = None
    # ``device`` is the AUTHORITY/eval device — the exact d_seg/d_pose that pick
    # the BEST checkpoint and seed the telemetry MUST be CPU-TRUSTED (or CUDA);
    # MPS is FORBIDDEN here (CLAUDE.md "MPS auth eval is NOISE": 23x pose drift).
    device: str = "cpu"
    # ``train_device`` is the GRADIENT backend ONLY — the per-step forward+backward
    # of the decoder→roundtrip→frozen-scorer graph. MPS is ALLOWED here (and is the
    # whole point: ~104x faster fwd+bwd than torch-CPU per the bench). The MPS
    # gradient is research-signal until it passes the descent-equivalence acceptance
    # gate (BOTH d_seg AND d_pose at the real n); the EXACT metric is always re-run
    # on ``device`` (CPU authority). Defaults to ``device`` (single-device legacy).
    train_device: str | None = None
    # ``split_by_head`` (the pose-axis SALVAGE): run the SegNet-path forward+backward
    # on ``train_device`` (MPS — the 90x lever, validated bit-identical on d_seg) and
    # the PoseNet-path forward+backward on the CPU AUTHORITY ``device`` (zero pose
    # drift). The two per-head frame cotangents are summed at the frame tensor, so the
    # combined decoder gradient is descent-equivalent on BOTH terms BY CONSTRUCTION
    # (the pose part IS the authority gradient). The prior full-MPS gradient REJECTED
    # the descent-equivalence gate on the POSE axis (d_seg PASS / d_pose REJECT, gap
    # 0.06->7.02). Requires a split device. Defaults False (full train_device forward).
    split_by_head: bool = False
    # ``async_eval`` (the throughput salvage): run the CPU AUTHORITY exact eval
    # (byte-close + 600-pair d_seg/d_pose/rate/score) in a BACKGROUND THREAD off a
    # POINT-IN-TIME CPU snapshot of the EMA shadow, so the GPU/MPS training loop is
    # NOT blocked by the ~13-min CPU eval. The eval is the IDENTICAL eval on the
    # IDENTICAL snapshot — only non-blocking — so the authority numbers are
    # bit-for-bit the same as the sync path (proved by the no-regression test).
    # At most ONE eval is in-flight: a new eval epoch arriving while the prior eval
    # is still running is SKIPPED (logged) — the cadence self-throttles. On run
    # completion the in-flight thread is JOINED so the final BEST + last eval row
    # land before exit. Defaults False (the sync path is byte-identical unchanged).
    async_eval: bool = False
    # ``pose_film_enabled`` (Lever 3 — the Quantizr STORE-pose lesson): wrap the
    # vendored ``HNeRVDecoder`` with a stored-pose FiLM at the stem (channels[0])
    # and store the 6 GT pose scalars per pair as side information (Wyner-Ziv), so
    # d_pose collapses toward the stored-pose quant floor and the decoder capacity
    # is freed for d_seg + rate. The stored pose is range-coded into an ADDITIVE
    # archive section (~1 KB charged; the vendored codec stays pristine). DEFAULT
    # FALSE → the driver builds the vendored decoder unchanged and adds NO pose
    # section, so the archive is BYTE-IDENTICAL to today (the live basin is
    # unaffected if it resumes onto this code — proved by the byte-identity test).
    pose_film_enabled: bool = False
    # FiLM MLP bottleneck width (only consulted when ``pose_film_enabled``).
    pose_film_hidden: int = 8
    # Pose-section CODEC (only consulted when ``pose_film_enabled``; the REOPENED
    # #1 finding — the ego trajectory is ~1-DOF). DEFAULT ``"iid"`` = byte-identical
    # to today (the legacy per-pair codec). ``"lowrank"`` selects the low-rank SVD
    # codec (``pose_film.encode_pose_section_lowrank``).
    #
    # HONEST FINDING (Pass-1 adversarial review): a NAIVE max-byte-cut (rank-2/254 →
    # ~1.14 KB) is NET-NEGATIVE on the full score, because the legacy iid codec
    # already stores the pose nearly losslessly (MSE ≈ 2.9e-5) and the pose term
    # ``sqrt(10·d_pose)`` is nonlinear, so the fidelity loss costs more than the bytes
    # save. The DEFAULT low-rank operating point is therefore the Pareto-DOMINANT
    # rank-4/511 (~2.56 KB vs ~3.09 KB = smaller, AND MSE 2.7e-5 ≤ iid's 2.9e-5 =
    # lower), a modest unambiguous byte win (~-0.0004 rate) where the pose term cannot
    # worsen. The net win is ``[contest-CPU advisory]`` NON-PROMOTABLE — the rate
    # saving is exact but the exact net magnitude needs a byte-closed
    # ``upstream/evaluate.py`` on the real archive. The codec choice is INVISIBLE to
    # inflate (parse_pose_section auto-dispatches on the section magic), so a low-rank
    # archive round-trips through the SAME inflate.
    pose_section_codec: str = "iid"
    # Low-rank codec rank/levels (only consulted when ``pose_section_codec ==
    # "lowrank"``). The DEFAULT rank=4/levels=511 is the Pareto-DOMINANT operating
    # point on the real GT pose (smaller than iid AND lower-MSE → improves rate, pose
    # term cannot worsen); a more aggressive cut (rank-2/254) is net-negative on the
    # full score (the encode + round-trip MEASURE both bytes and MSE; the net win is
    # advisory until byte-closed exact eval).
    pose_section_lowrank_rank: int = 4
    pose_section_lowrank_levels: int = 511
    # Lever-3 pose-FiLM VERSION (only consulted when ``pose_film_enabled``):
    #   1 = v1 stem-injection (``pose_film.PoseFiLMHNeRVWrapper``) — feeds BOTH heads,
    #       so pose couples into d_seg (the v1 instability; legacy/default).
    #   2 = v2 OPTIMAL Quantizr port (``pose_film_v2.PoseFiLMHNeRVWrapperV2``) —
    #       residual FiLM on the ``rgb_0`` head ONLY, leaving ``rgb_1`` (the SegNet
    #       frame) FiLM-clean → EXACT d_seg/d_pose decoupling (#118 SEALED). v2 routes
    #       the FiLM params (``pose_mlp`` + ``film_resid``) to a DEDICATED capped-LR
    #       AdamW group (``_FILM_LR_CAP``), excluded from Muon (the §A review fix).
    # DEFAULT 1 (legacy); the from-0 decisive run sets 2. Byte-identity is unaffected
    # by the version when ``pose_film_enabled=False`` (no FiLM either way).
    pose_film_version: int = 1
    # FiLM-v2 TRUNK DECOUPLING (the EXACT ∂d_seg/∂pose=0 completion). v2 already makes
    # the FiLM HEAD decoupled (rgb_1 is FiLM-clean → ∂d_seg/∂(FiLM-params)=0), but the
    # POSE LOSS gradient still LEAKS into the SHARED decoder: pose reads BOTH frames
    # (PoseNet is a 2-frame net), so pose grad flows pose_loss → f1 → rgb_1/trunk/latents
    # AND pose_loss → f0 → rgb_0/trunk/latents → and the SAME trunk+latents produce f1,
    # so training to reduce d_pose perturbs the shared trunk, which moves f1, which moves
    # d_seg. Measured symptom (the leaked synergy): under the oomph seg loss d_pose drifts
    # UP monotonically (0.000335→0.000417 over ep10-59); with ∂S/∂d_pose ≈ 86% of
    # ∂S/∂d_seg this is score-costly. DEFAULT FALSE → the pose cotangent flows into the
    # whole shared graph exactly as today (byte-identical gradient; the live A/B is
    # unaffected). When TRUE (requires pose_film_enabled AND split_by_head), the POSE
    # loss gradient updates ONLY the FiLM pose path (``pose_mlp`` + ``film_resid``) and is
    # STOP-GRADIENT on the shared trunk + latents + rgb_0/rgb_1 heads → ∂d_seg/∂(pose-
    # objective)=0 EXACTLY and the two objectives are orthogonal (trunk+latents trained by
    # SEG only, the FiLM pose path by POSE only). Mechanism: in the split-by-head backward
    # the seg + pose cotangents are NO LONGER fused — seg backprops first (trains the whole
    # graph), the non-FiLM ``.grad`` is snapshotted, pose backprops (accumulates onto all
    # params), then the snapshot is RESTORED onto every non-FiLM param + latents (removing
    # the pose contribution there) — proved EXACT by the gradient-routing test (non-FiLM
    # grad == seg-only; FiLM grad == pose). MEASURED-QUESTION CAVEAT: freezing the trunk
    # w.r.t. pose means the FiLM head + ~6 stored scalars/pair must carry ALL the pose
    # signal — this MIGHT hold d_pose WORSE (if the scalars are insufficient) OR better (no
    # seg/pose tug-of-war). The Quantizr design (store-6-pose + FiLM) suggests it suffices,
    # but it is EMPIRICAL — this is an OPT-IN mode for a GPU A/B (complete-decoupling vs
    # current-v2), NOT a forced default. NOT an authority/score knob; the exact d_seg/d_pose
    # that pick BEST still run the full scorer on the CPU authority.
    pose_film_trunk_stopgrad: bool = False
    # FiLM-v2 rgb_0 DECOUPLING REFINEMENT (the "pose can train the frame-0 head too" fix).
    # ``pose_film_trunk_stopgrad`` (above) restores ALL non-FiLM params — INCLUDING the
    # ``rgb_0`` head — to their seg-only grad, which FREEZES ``rgb_0`` w.r.t. the pose
    # objective. But the contest SegNet reads ONLY frame-1 (``rgb_1``), so d_seg is
    # INDEPENDENT of ``rgb_0`` (``∂d_seg/∂(rgb_0 params) = 0`` exactly — rgb_0 only writes
    # f0, which SegNet never sees). Therefore the POSE loss CAN train ``rgb_0`` (the
    # frame-0 head, which IS pose-conditioned via the FiLM residual on its input) WITHOUT
    # any d_seg cost — giving the pose objective strictly MORE capacity (the rgb_0 head +
    # the FiLM path) to hold d_pose, while keeping the EXACT ∂d_seg/∂(pose-objective)=0
    # decoupling (trunk + skips + blocks + refine + rgb_1 + latents stay seg-only). DEFAULT
    # FALSE → ``rgb_0`` is restored to seg-only exactly as the base trunk-stopgrad does
    # (byte-identical to the trunk-stopgrad A/B today). When TRUE (requires
    # ``pose_film_trunk_stopgrad=True``), ``rgb_0``'s params are EXCLUDED from the seg-only-
    # restore set, so the pose backward's contribution to them is KEPT → pose trains
    # {FiLM path + rgb_0}, seg trains {trunk + skips + blocks + refine + rgb_1 + latents}.
    # PROOF (test): ∂d_seg/∂(pose-objective) STILL = 0 (trunk+latents+rgb_1 grad bit-
    # identical to seg-only) AND rgb_0 now carries the pose gradient (was zero under the
    # base trunk-stopgrad). NOT an authority/score knob; the exact d_seg/d_pose that pick
    # BEST still run the full scorer on the CPU authority.
    pose_film_rgb0_pose_trainable: bool = False
    # Track-A DISTORTION finishing-kit (PR98 bias / T10 affine / S12 mask) — an
    # inflate-side, zero/near-zero-byte postproc on the rendered frames + a ~54-byte
    # distortion archive section. DEFAULT None → NO kit, NO section, BYTE-IDENTICAL
    # (the live basin/distortion arm resuming onto this code is unaffected — proved
    # by the kit no-op test). It is a POST-CONVERGENCE finishing pass: set on the
    # FINAL export of a converged checkpoint, never during the descending basin. The
    # value is a ``tac.torch_vehicle.distortion_finishing_kit.DistortionKitConfig``.
    distortion_kit: Any = None
    # Track-A Item B / D2 rate path: math-optimal variable-level decoder codec.
    # DEFAULT FALSE -> the no-FiLM branch is exactly ``self.v.build_archive(...)`` and
    # remains byte-identical to the vendored/base_ch20 basin. When enabled, this is
    # intentionally conservative and base_ch20-only: it consumes the MEASURED RD table
    # from ``experiments/probe_variable_level_waterfill_net.py`` and solves
    # ``byte_target=2731, net_stop=False``. It composes with the FiLM archive branch by
    # replacing only the decoder blob and preserving the additive pose section. The
    # falsified aggressive ``net_stop`` path is not exposed as a driver knob.
    variable_level_waterfill_enabled: bool = False
    variable_level_waterfill_rd_table: dict[str, Any] | None = None
    variable_level_waterfill_byte_target: float = _TRACK_A_D2_CONSERVATIVE_BYTE_TARGET
    # Lever-4↔variable-level-export UNIFICATION (R14 contest-optimality finding,
    # operator directive "completely engineer + implement + adversarially review +
    # harden"). DEFAULT FALSE -> the export is byte-identical to today (the no-FiLM
    # branch is exactly ``self.v.build_archive(...)``; the FiLM branch the additive
    # pose archive). When TRUE, the EXPORT decoder blob is built at a VARIABLE per-
    # tensor INT8 grid derived from Lever-4's ONLINE score-sensitivity EMA
    # (``||∂S/∂w_t||``) via ``levels_from_sensitivity_for_codec`` (the SAME rank-norm
    # band the score-aware QAT trained the decoder to be robust at) — capturing the
    # full reverse-waterfill byte saving (R14: ~36% decoder-blob on a coarse map)
    # WITHOUT a separate offline RD-table measurement (the unification of Lever-4's
    # online EMA with the variable-level codec). This is the contest-OPTIMAL byte-half
    # of Lever-4. Mutually exclusive with ``variable_level_waterfill_enabled`` (two
    # distinct level sources for the SAME export blob); ``__post_init__`` refuses both.
    # The byte saving is REAL + measurable; the NET-score win is advisory until a
    # 600-pair byte-closed dual CPU/CUDA eval (no score claim from the flag alone).
    lever4_variable_level_export_enabled: bool = False
    # Pose-gradient THROTTLE (the "pose is solved → stop paying for it" speed lever).
    # The split-by-head PoseNet path costs ~51% of the epoch (CPU FastViT fwd+bwd, measured
    # 10.2s of a 20s epoch @96 pairs) yet d_pose converges to ~0.0015 early — computing a
    # near-noise √(10·d_pose) gradient every epoch is the waste. ``pose_grad_every_k``
    # computes the pose cotangent only every k-th global epoch (k=1 = EVERY epoch =
    # byte-identical DEFAULT; k>1 = skip the pose path on off-epochs, flowing the
    # SegNet-only cotangent into the decoder). ``pose_grad_resume_threshold``>0 is BOTH
    # the warm-up AND the self-protect guard: while the last-computed training pose_mse
    # exceeds it (pose still converging, or drifted), pose is force-computed EVERY epoch;
    # once it drops below (solved) the every-k cadence takes over — so pose can never be
    # silently starved while it matters. Scoped to split_by_head; the non-split path is
    # unchanged. DEFAULT k=1 / threshold=0.0 → the gradient is BYTE-IDENTICAL to today
    # (live A/B unaffected; apples-to-apples preserved until ALL arms restart with the
    # same k). NOT an authority/score knob — it only changes train-time wall-clock; the
    # exact d_seg/d_pose that pick BEST still run the full scorer on the CPU authority.
    pose_grad_every_k: int = 1
    pose_grad_resume_threshold: float = 0.0
    # ``pose_grad_on_train_device`` (the FULL-MPS-ALL-LEVERS unbundle — the speed lever
    # that does NOT touch score). The base split-by-head SALVAGE runs the SegNet path on
    # ``train_device`` (MPS) but HARDCODES the PoseNet forward+backward onto the CPU
    # AUTHORITY ``device`` — conflating the AUTHORITY rule (MPS never SCORES — correct,
    # forever) with the GRADIENT (the MPS pose gradient is per-step faithful: the full-MPS
    # base_ch20 basin trained on MPS reached CPU-authority d_pose=0.00034 — the gradient
    # IS good). That CPU-side pose backward is ~51% of the epoch (CPU FastViT fwd+bwd),
    # making split-by-head ~7x slower than full-MPS. When TRUE, the PoseNet head's
    # forward+backward run on ``train_device`` (MPS) via the SAME train_distortion_net the
    # SegNet head uses (patch_scorer_for_mps + differentiable-yuv6 already applied when that
    # net is built on MPS) with the train-device pose targets — recovering the full-MPS
    # speed while KEEPING the per-axis cotangent SPLIT (the seg cotangent and the pose
    # cotangent stay distinct tensors, so the equimarginal controller (Lever A) + per-dim
    # Mahalanobis (Lever C) + FiLM-v2 trunk-stopgrad ALL keep firing on their separable
    # per-axis norms). The AUTHORITY is UNCHANGED: ``exact_eval`` (the d_seg/d_pose/score
    # that pick BEST) ALWAYS runs on ``device`` (CPU/CUDA) — this flag moves ONLY the
    # training GRADIENT to MPS, never the score (CLAUDE.md "MPS IS a valid TRAINING-GRADIENT
    # device" + "MPS auth eval is NOISE"). The documented risk is OPTIMIZER CHAOS (Muon +
    # a weakly-driven pose term at high LR) — the descent-equivalence is verified by the
    # chaos-check smoke (monotone d_pose), NOT assumed. Requires ``split_by_head=True`` (the
    # per-axis split is the substrate this rides on). DEFAULT FALSE → the PoseNet path runs
    # on the CPU authority EXACTLY as today (byte-identical gradient; the live split-by-head
    # A/B is unaffected — proved by the device-routing test). NOT an authority/score knob.
    pose_grad_on_train_device: bool = False
    # Adaptive Pose-Gradient Controller (APGC) — the closed-loop replacement for the
    # static k/threshold THROTTLE above. The static throttle is OPEN-loop: it fixes a
    # cadence k and a fixed resume threshold, so when d_pose DRIFTS (measured monotonic
    # 0.000335→0.000408 over ep10-40 under the oomph seg crank, as the shared decoder
    # trunk re-tunes toward seg) the fixed threshold (0.001, ~2.5-3× the actual ~0.0004
    # pose_mse) NEVER fires and pose is corrected only 1-in-k epochs WHILE drifting. The
    # calculus is why that matters: ∂S/∂d_pose = 5/√(10·d_pose) = 85.5 at d_pose≈0.0004
    # ≈ 86% of ∂S/∂d_seg = 100 — so an un-arrested pose drift is nearly as score-costly
    # as the d_seg we optimize (the measured ep10-40 drift already cost +0.006 S).
    #
    # APGC holds d_pose at its (moving) FLOOR with minimum gradient spend, via the
    # equimarginal / constraint principle: spend the expensive pose path ONLY when the
    # marginal score harm of NOT correcting (pose above the deadband, or rising) exceeds
    # the wall-clock saved by skipping. It tracks a RUNNING-MIN ``_pose_floor`` (the
    # adaptive floor, not a guessed constant), holds pose within ``floor·(1+tol)`` (a
    # ``floor_tol``=0.08 deadband ≈ a 0.0023 S slack at the frontier operating point),
    # arrests drift the moment the deviation breaches the band OR the recent trend rises,
    # otherwise widens the cadence proportionally toward ``k_max`` as pose sits solidly at
    # floor, and NEVER goes blind longer than ``k_max`` epochs (the measurement-floor — a
    # skipped epoch does not MEASURE pose since the pose forward is fused with the backward
    # here, so the measurement-floor bounds drift-blindness; a future refinement is a
    # cheap forward-only pose probe decoupled from the backward).
    #
    # DEFAULT False → the EXISTING static k/threshold branch above runs UNCHANGED
    # (byte-identical to today; the live A/B is unaffected). When True, the static k/
    # threshold are IGNORED and the controller governs ``do_pose``. Scoped to
    # split_by_head (the throttle is split-only); ``__post_init__`` refuses adaptive on
    # the non-split path. NOT an authority/score knob — it only changes train-time
    # wall-clock cadence; the exact d_seg/d_pose that pick BEST still run the full scorer
    # on the CPU authority.
    pose_grad_adaptive: bool = False
    # APGC deadband: hold pose ≤ floor·(1+tol). 0.08 ⇒ tolerate an 8% deviation above the
    # running floor before drift-arrest fires (≈ 0.0023 S slack at the frontier d_pose).
    pose_grad_floor_tol: float = 0.08
    # APGC sparsest cadence (when pose is solidly at floor) AND the measurement-floor: the
    # max number of epochs the controller may go without a pose MEASUREMENT before forcing
    # a compute. Caps drift-blindness on skipped epochs.
    pose_grad_k_max: int = 8
    # APGC trend window: the number of most-recent COMPUTED pose_mse values the derivative
    # (slope) term inspects. ``rising`` = hist[-1] > hist[0] over this window ⇒ arrest even
    # inside the deadband (the trend caught drift before it breached the band).
    pose_grad_trend_window: int = 3
    # -- Lever A (EQUIMARGINAL pose-weight controller) — DEFAULT-PRESERVING ----
    # ``pose_equimarginal_enabled is False`` (the default) leaves ``spec.pose_weight`` UNMODIFIED every
    # epoch — byte-identical gradient. When True (split-by-head only), an EMA-smoothed deadbanded controller
    # (``tac.torch_vehicle.equimarginal_pose_weight``) scales ``pose_weight`` per epoch so the measured
    # per-axis frame-cotangent-norm ratio ``‖cot_pose‖/‖cot_seg‖`` tracks ``pose_equimarginal_rho`` — the
    # WEIGHT analogue of the APGC cadence. Because ``pose_l = sqrt(10·pose_mse)`` is ALREADY the contest
    # pose-term in score units, balancing the cotangent norms balances the per-step SCORE pull (the
    # equimarginal point). It TRACKS ``5/sqrt(10·d_pose)`` (auto-lowers w_pose as d_pose falls) and RESTORES
    # the balance the oomph w_seg×1.5 overlay breaks. Scoped to split_by_head (needs the separable per-axis
    # cotangent norms the split backward computes); ``__post_init__`` refuses it on the non-split path.
    # NOT an authority/score knob — it changes only the train-time weight; the exact d_seg/d_pose that pick
    # BEST still run the full scorer on the CPU authority. Checkpointed (the controller state continues a
    # resume).
    pose_equimarginal_enabled: bool = False
    pose_equimarginal_rho: float = 1.0  # target pose:seg score-marginal ratio (1.0 = true equimarginal)
    pose_equimarginal_decay: float = 0.9  # EMA decay for the measured ratio
    pose_equimarginal_tol: float = 0.15  # deadband half-width as a fraction of rho
    pose_equimarginal_bound_lo: float = 0.25  # accumulated w_pose floor as a fraction of w_pose0
    pose_equimarginal_bound_hi: float = 4.0  # accumulated w_pose ceiling as a fraction of w_pose0
    # -- Lever C (per-dim pose Mahalanobis / AIL weighting) — DEFAULT-PRESERVING
    # ``pose_dim_weights is None`` (the default) leaves the pose loss as the UNIFORM
    # ``MSE(pose_pred6, target6)`` — byte-identical (no multiply, no renorm). A length-6 tuple routes the
    # squared error through ``tac.torch_vehicle.pose_dim_weights.weighted_pose_mse`` (renormalised to mean 1.0
    # so the overall pose-loss scale — and thus ``pose_weight``'s calibration — is preserved; only the per-dim
    # balance tilts). The weights are MEASURED on the basin (per-dim target variance, inverse-variance
    # Mahalanobis) by ``measure_pose_dim_weights_from_targets``, NOT hand-set. Applies to BOTH the split and
    # fused pose paths. SegNet reads only rgb_1 so this costs ZERO d_seg + ZERO bytes.
    pose_dim_weights: tuple[float, ...] | None = None
    # WS-A/M3 Muon LR-floor fix (DEFAULT-OFF → byte-identical). False = Muon shares the
    # AdamW lr_lambda (its floor mis-keyed to adamw_lr → never anneals below 0.5× at stage
    # 8). True = Muon gets its own cosine floor keyed to muon_lr. Opt-in for the scaled run.
    muon_lr_floor_fix: bool = False
    # WARM-START fine-tune (the capacity-vs-loss-wall disambiguator + general fine-tuning
    # entry point). DEFAULT None → stage-0 latents are the random init (byte-identical from-0
    # path). When set to a ``best/`` dir holding ``best_ema_decoder.pt`` + ``best_ema_latents.pt``
    # (the canonical converged-checkpoint layout this driver writes), a FRESH run (no resume
    # checkpoint in out_dir) builds the stage-0 decoder, LOADS those weights into it, and uses
    # the stored latents as the stage-0 init INSTEAD of the random draw — so training CONTINUES
    # from a converged basin rather than from scratch. This is an apples-to-apples fine-tune
    # primitive: two runs with the SAME ``warm_start_dir`` + SAME ``seed`` share their init
    # bit-for-bit and differ only by their curriculum (e.g. an oomph-soft_cosine arm vs a
    # CE-continue arm). REQUIRES ``pose_film_enabled=False`` (the saved vendored state_dict has
    # no FiLM params) and a matching ``base_channels`` / ``latent_dim`` (the load is strict).
    # Resume always wins: if a checkpoint already exists in out_dir, the warm-start is ignored
    # (the run is resuming its OWN trajectory, not re-priming). NOT consulted on any non-stage-0
    # carry path. Byte-identical when None.
    warm_start_dir: Path | None = None
    # KD-WARM-START (the wall-clock resolution — the bind-all linchpin). DEFAULT None →
    # byte-identical (no KD; the stage-0 init is the random/warm_start path above). The full
    # PR95 29k-epoch from-scratch curriculum is ~weeks on MPS — INFEASIBLE. The basin
    # (stage-1 CE) captures most of that VALUE, but the solved-taper's DIFFERENT channel
    # shapes BLOCK a strict-decoder warm-start of the vendored basin. KD-warm-start resolves
    # it: when set to a ``best/`` dir holding the converged VENDORED-taper basin's
    # ``best_ema_decoder.pt`` + ``best_ema_latents.pt`` AND ``taper_channels`` is set (the
    # student is the solved taper), a FRESH run (no resume checkpoint):
    #   * LATENTS: loads the basin latents (n_pairs, latent_dim) DIRECTLY as the stage-0 init
    #     (taper-INDEPENDENT — only the decoder channels change), and
    #   * DECODER: builds a FROZEN teacher = the basin's vendored-taper decoder, then runs a
    #     KD WARM-UP phase (the first ``kd_warm_epochs`` of stage 0) whose loss is the
    #     frame-MSE between the student (solved-taper) frames and the teacher frames rendered
    #     on the SAME latents — distilling the basin's rendered pairs into the re-tapered
    #     student. After the warm-up the normal score-aware curriculum continues from the
    #     distilled student.
    # REQUIRES ``taper_channels`` set (the student is the re-taper; a same-taper warm-start
    # should use ``warm_start_dir`` directly, no KD needed) and a matching ``base_channels``
    # / ``latent_dim`` (the teacher load is strict). Mutually exclusive with
    # ``warm_start_dir`` (two distinct stage-0 init sources). Resume always wins (a run that
    # owns a checkpoint continues its own trajectory). Byte-identical when None.
    kd_warm_start_dir: Path | None = None
    # KD warm-up phase length: the number of stage-0 epochs that run the frame-MSE
    # distillation (the basin teacher → solved-taper student) BEFORE the normal score-aware
    # curriculum continues. Only consulted when ``kd_warm_start_dir`` is set. Must be >= 1
    # (the phase is a PREFIX of stage 0; 0 would be a no-op KD that should just be a plain
    # warm-start) and <= stage-0 epochs (else the whole of stage 0 is KD with no score-aware
    # continuation — refused so the bind-all contract holds).
    kd_warm_epochs: int = 300
    # KD warm-up learning rate (the distillation AdamW lr). Only consulted when
    # ``kd_warm_start_dir`` is set. Defaults to a moderate fine-tune lr.
    kd_warm_lr: float = 1e-3
    # KD warm-up: co-train the (shared) latents during distillation. DEFAULT True (the
    # student adapts the code to its own taper while chasing the teacher frames, recomputed
    # each step from the current latents). Only consulted when ``kd_warm_start_dir`` is set.
    kd_warm_train_latents: bool = True
    # EMA WARMUP (the EMA-shadow-lag fix for SHORT runs; sister of the capstone curriculum fix).
    # DEFAULT False → the EMA decay is the constant ``spec.ema_decay`` every step (byte-identical
    # legacy path). The constant 0.999 has a ~1000-step window; on a short fine-tune (≤ a few
    # hundred epochs) the shadow stays frozen near its init and HIDES the trajectory — so the
    # exact d_seg/d_pose read off the shadow never reflect the fine-tuning. When True, the
    # effective per-step decay is ``min(spec.ema_decay, (t+1)/(t+10))`` where ``t`` is the
    # driver's global EMA step counter (0 at the first step of the run) — the standard
    # bias-corrected warmup: decay ramps 0.10 → 0.99 → ``spec.ema_decay`` so the shadow TRACKS
    # the live weights early and converges to the faithful decay once enough steps accumulate.
    # Scoped to the EMA update only; the optimizer/loss/RNG paths are untouched. The faithful
    # long-run basin is byte-identical when this is off; opt-in for warm-start fine-tunes.
    ema_warmup: bool = False
    # CONFIGURABLE TAPER (the #1 structural lever — d_seg-aware capacity reallocation).
    # DEFAULT None → the decoder is the vendored ``HNeRVDecoder`` with the hardcoded HNeRV
    # taper (byte-identical). When set to an explicit 7-stage channel schedule, the decoder is
    # the ``ConfigurableTaperHNeRVDecoder`` (a faithful generalization of the vendored decoder
    # whose ONLY difference is the channel schedule) — reallocating capacity from the
    # insensitive low-res early stages to the d_seg-critical mid-late stages (gate-2
    # sensitivity map) at a ~byte-matched param count. The vendored codec is schedule-agnostic
    # so build_archive/parse_archive/partition_params_for_muon/apply_qat all round-trip the
    # different shapes unchanged. Requires pose_film_enabled=False (the taper carrier has no
    # FiLM wrapper here). A resume into an out_dir trained with a DIFFERENT taper fails closed
    # (strict load shape mismatch) — never a silent cross-architecture resume. Byte-identical
    # when None (the live basin / from-0 A/B are unaffected).
    taper_channels: list[int] | None = None
    # WEIGHT-ENTROPY PENALTY (the Ballé end-to-end rate-distortion lever — the
    # un-integrated VCM term). DEFAULT 0.0 → BYTE-IDENTICAL (the term is never
    # computed, the penalty module is never built, its params never enter the
    # optimizer; the live basin resuming onto this code is unaffected — proved by
    # the byte-identity test). When > 0, the driver builds ONE
    # ``tac.torch_vehicle.weight_entropy_penalty.WeightEntropyPenalty`` per run
    # (a per-output-channel Ballé factorized logistic-CDF prior on each decoder
    # Conv2d/Linear weight tensor's CODEC-GRID symbols), adds its LEARNABLE prior
    # params to the AdamW group (so the prior trains), and adds
    # ``λ · rate_term`` (the expected codelength on the contest rate scale) to the
    # training loss in the SAME stages C1a applies (mirroring ``cat_lambda``'s
    # 0.01→0.02 late-stage schedule via ``weight_entropy_penalty_stage_min``). The
    # contest rate term is ``25·archive_bytes/N`` and ``archive_bytes ≈ Σ H(symbol)
    # ·numel/8``; the post-hoc coder is already at the lossless floor, so the ONLY
    # rate lever is lowering ``H`` itself — which is set by TRAINING. The Ballé term
    # pulls the weight-symbol distribution toward low entropy → lower ``H`` → lower
    # byte floor. STRONGER than the memoryless C1a shadow (a learned per-channel
    # prior, not a fixed soft histogram). NOT an authority/score knob — it changes
    # train-time dynamics; the exact d_seg/d_pose that pick BEST + the REAL archive
    # bytes still come from the byte-closed codec on the CPU authority. The net-score
    # win is ``[contest-CPU advisory]`` until a byte-closed paired CPU/CUDA eval (no
    # score claim from the flag alone; the empirical bit-spend proof is the λ-on/off
    # A/B measuring real ``archive_bytes`` at equal d_seg/d_pose per Catalog #304).
    weight_entropy_penalty_lambda: float = 0.0
    # The FIRST curriculum stage index (0-based) at which the weight-entropy penalty
    # is ACTIVE. Mirrors C1a's "late-stage only" schedule (C1a ramps in at stage 5);
    # the decoder-rate lever is most useful once the coarse structure is learned. A
    # stage at index < this contributes NO penalty term (byte-identical there) even
    # when ``weight_entropy_penalty_lambda`` > 0. Default 0 = active from stage 0
    # (the smoke/A-B convention so a tiny 1-stage run exercises it). Only consulted
    # when ``weight_entropy_penalty_lambda`` > 0.
    weight_entropy_penalty_stage_min: int = 0
    # Initial logistic scale for each per-channel Ballé prior (only consulted when
    # the penalty is active). The default 10.0 starts broad (the codec-grid symbols
    # span ~±127 early) and tightens as the prior learns.
    weight_entropy_penalty_init_scale: float = 10.0
    # WATERFILL the weight-entropy penalty across tensors (the KKT reverse-water-fill
    # ALLOCATION option). DEFAULT FALSE = UNIFORM λ on every tensor (the legacy
    # byte-identical loss term). When TRUE (and λ>0), the per-tensor penalty weight is
    # ``byte_share_t / (sensitivity_t + eps)`` (normalized to the same aggregate budget),
    # so the rate pressure concentrates on big-byte / low-d_seg-d_pose-sensitivity
    # tensors and protects high-sensitivity ones. The sensitivity source is the Lever-4
    # ``tensor_sensitivity_ema`` when it is populated (score-aware-QAT runs), else
    # byte-share-only (still a non-uniform allocation). Re-derived each epoch from the
    # CURRENT weights + sensitivity EMA. NOT consulted when λ=0 (penalty never built).
    weight_entropy_penalty_waterfill: bool = False
    # SUPERSEDE C1a when the penalty is active (the DOUBLE-COUNT fix). PR95's C1a
    # (``cat_entropy_v2`` via ``spec.cat_lambda``) and this penalty BOTH penalize the
    # SAME quantity — the size-weighted codec-grid symbol entropy of ``w/(max|w|/127)``,
    # bits/weight. C1a is a fixed-bandwidth Gaussian soft-histogram; this is a LEARNED
    # per-channel logistic-prior expected codelength. MEASURED (probe
    # ``experiments/probe_balle_c1a_qat_interaction.py``): stacking BOTH reaches a HIGHER
    # (worse) entropy than EITHER alone (−1.38 vs penalty −1.72 / C1a −1.57) — the two
    # same-quantity estimators interfere, so stacking is NET-NEGATIVE. DEFAULT TRUE: when
    # ``weight_entropy_penalty_lambda > 0`` AND this stage's penalty is active, the C1a
    # term is ZEROED (the learned-prior penalty SUPERSEDES the memoryless shadow). Set
    # FALSE to stack both (NOT recommended; the probe shows it is worse). BYTE-IDENTICAL
    # on the default λ=0 path (the penalty is off → C1a is untouched → the live basin is
    # unaffected). Only consulted when ``weight_entropy_penalty_lambda > 0``.
    weight_entropy_penalty_supersedes_c1a: bool = True
    # PER-STAGE LR WARMUP (the E#5 stage-transition pose-kick fix). DEFAULT 0.0 →
    # BYTE-IDENTICAL: every stage's LambdaLR starts at the cosine PEAK (epoch 0 →
    # 0.5·(1+cos(0)) = 1.0 × peak_lr) exactly as the legacy faithful path, so the live
    # basin resuming onto this code is unaffected (proved by
    # ``test_stage_lr_warmup_frac_zero_is_byte_identical``). PROBLEM (MEASURED, E#5): at
    # each stage→stage boundary PR95 resets the optimizer + the cosine restarts to PEAK
    # (1e-3) and SLAMS the shared trunk → d_pose spiked 0.00021→0.00142 (6.8×) before
    # recovering, and d_seg 0.00224→0.00287, at the stage-1→2 boundary; 6 more transitions
    # remain in the full curriculum. FIX (standard warmup-after-restart): when > 0, the
    # FIRST ``ceil(stage_lr_warmup_frac · spec.epochs)`` epochs of EACH stage LINEARLY
    # ramp the LR from a small floor (``stage_lr_warmup_start_ratio`` × peak) up to the
    # cosine value at the warmup-end epoch, then the normal cosine continues — so the
    # trunk is eased in at the boundary instead of slammed. Applied to BOTH the AdamW and
    # Muon lr_lambdas (the two trunk-touching groups). Must be in [0.0, 0.5]; > 0.5 would
    # leave < half the stage for the cosine descent (refused in __post_init__). The
    # latent + FiLM + penalty groups ride the same AdamW lambda (one schedule). Resume:
    # the LambdaLR ``last_epoch`` is checkpointed, so a death mid-warmup resumes the same
    # ramp position bit-for-bit. ``[contest-CPU advisory]`` — the win is the MEASURED
    # boundary-kick reduction (the headline test), not a score claim from the flag alone.
    stage_lr_warmup_frac: float = 0.0
    # The LR floor (as a fraction of the stage peak) the warmup ramp STARTS from at the
    # first epoch of each stage. Only consulted when ``stage_lr_warmup_frac`` > 0. 0.1 =
    # start at 10% of peak then linearly ramp to the cosine value at warmup-end (a gentle
    # ease-in). Must be in (0.0, 1.0].
    stage_lr_warmup_start_ratio: float = 0.1
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if str(self.device).lower().startswith("mps"):
            raise ValueError(
                "MPS is NEVER trusted as the AUTHORITY device (CLAUDE.md): corrupts "
                "SegNet/PoseNet argmax (23x pose drift). The exact d_seg/d_pose that "
                "pick BEST must be CPU/CUDA. Set device='cpu' and train_device='mps' "
                "to use the Apple GPU as the GRADIENT backend only (split-device)."
            )
        if self.train_device is None:
            self.train_device = self.device
        if self.split_by_head and self.train_device == self.device:
            raise ValueError(
                "split_by_head requires train_device != device (the SegNet path runs "
                "on train_device, the PoseNet path on the CPU authority device). Set "
                "device='cpu' and train_device='mps'."
            )
        if self.variable_level_waterfill_enabled:
            if self.base_channels != 20:
                raise ValueError(
                    "Track-A D2 variable-level waterfill is only certified for the "
                    "base_ch20 production adapter. Set base_channels=20 or leave "
                    "variable_level_waterfill_enabled=False."
                )
            if not self.variable_level_waterfill_rd_table:
                raise ValueError(
                    "variable_level_waterfill_enabled requires a measured RD table "
                    "from probe_variable_level_waterfill_net.py; closed-form or "
                    "missing RD data is forbidden."
                )
            if float(self.variable_level_waterfill_byte_target) <= 0:
                raise ValueError("variable_level_waterfill_byte_target must be positive")
        if int(self.pose_film_version) not in (1, 2):
            raise ValueError(
                f"pose_film_version must be 1 (v1 stem) or 2 (v2 residual-rgb0); "
                f"got {self.pose_film_version}"
            )
        if not (0.0 <= float(self.stage_lr_warmup_frac) <= 0.5):
            raise ValueError(
                "stage_lr_warmup_frac must be in [0.0, 0.5] (0.0 = byte-identical no "
                "warmup; > 0.5 would leave < half the stage for the cosine descent); got "
                f"{self.stage_lr_warmup_frac}"
            )
        if not (0.0 < float(self.stage_lr_warmup_start_ratio) <= 1.0):
            raise ValueError(
                "stage_lr_warmup_start_ratio must be in (0.0, 1.0]; got "
                f"{self.stage_lr_warmup_start_ratio}"
            )
        if self.pose_section_codec not in ("iid", "lowrank"):
            raise ValueError(
                f"pose_section_codec must be 'iid' (legacy, byte-identical) or "
                f"'lowrank' (low-rank SVD pose codec); got {self.pose_section_codec!r}"
            )
        if self.pose_section_codec == "lowrank":
            if not (1 <= int(self.pose_section_lowrank_rank) <= 6):
                raise ValueError(
                    "pose_section_lowrank_rank must be in [1, 6] (pose_dim=6); got "
                    f"{self.pose_section_lowrank_rank}"
                )
            if int(self.pose_section_lowrank_levels) < 1:
                raise ValueError(
                    "pose_section_lowrank_levels must be >= 1; got "
                    f"{self.pose_section_lowrank_levels}"
                )
        if self.pose_film_trunk_stopgrad:
            # The trunk-stopgrad routing only EXISTS when there is a FiLM pose path to
            # route the pose gradient INTO, and is implemented in the split-by-head
            # backward (which is the only path that holds a separable pose cotangent).
            # Refuse a mis-config so the flag can never silently no-op (a silent no-op
            # would let the operator believe the decoupling is active when it is not).
            if not self.pose_film_enabled:
                raise ValueError(
                    "pose_film_trunk_stopgrad requires pose_film_enabled=True: the pose "
                    "gradient is routed INTO the FiLM pose path (pose_mlp + film_resid); "
                    "with no FiLM there is no pose path to route to."
                )
            if int(self.pose_film_version) != 2:
                raise ValueError(
                    "pose_film_trunk_stopgrad requires pose_film_version=2 (the v2 "
                    "residual-rgb0 wrapper whose pose params are pose_mlp + film_resid). "
                    "v1 stem-injection feeds both heads and has no decoupled pose path."
                )
            if not self.split_by_head:
                raise ValueError(
                    "pose_film_trunk_stopgrad requires split_by_head=True: the trunk-"
                    "stopgrad routing lives in the split-by-head backward (the only path "
                    "with a separable pose cotangent); the non-split path computes both "
                    "heads in one fused graph (no separable pose gradient to mask)."
                )
        if self.pose_film_rgb0_pose_trainable and not self.pose_film_trunk_stopgrad:
            # The rgb_0-decoupling refinement only EXISTS as a modification of the
            # trunk-stopgrad seg-only-restore set (it EXCLUDES rgb_0 from that set). With
            # the fused backward (trunk-stopgrad OFF) there is no separate seg-only-restore
            # to refine — the pose grad already flows into rgb_0 (and everything else).
            # Refuse a mis-config so the flag can never silently no-op.
            raise ValueError(
                "pose_film_rgb0_pose_trainable requires pose_film_trunk_stopgrad=True: it "
                "refines the trunk-stopgrad seg-only-restore set by EXCLUDING rgb_0 (so the "
                "pose loss trains the frame-0 head, which SegNet never reads). With the "
                "fused backward there is no seg-only-restore set to refine."
            )
        if int(self.pose_grad_every_k) < 1:
            raise ValueError(
                f"pose_grad_every_k must be >= 1 (1 = compute pose every epoch = "
                f"byte-identical); got {self.pose_grad_every_k}"
            )
        if self.pose_grad_on_train_device and not self.split_by_head:
            # The on-train-device pose path is a MODIFICATION of the split-by-head
            # backward (it only chooses WHICH device the separable PoseNet head runs on);
            # the non-split fused path computes both heads in one graph on the train
            # device already. Refuse a mis-config so the flag can never silently no-op.
            raise ValueError(
                "pose_grad_on_train_device requires split_by_head=True: it selects the "
                "train device for the SEPARABLE PoseNet head of the split-by-head backward; "
                "the non-split fused path already runs both heads on train_device."
            )
        if float(self.pose_grad_resume_threshold) < 0.0:
            raise ValueError(
                f"pose_grad_resume_threshold must be >= 0.0; got "
                f"{self.pose_grad_resume_threshold}"
            )
        if self.pose_grad_adaptive:
            # The APGC is a split-only THROTTLE replacement (the pose path it cadences
            # only exists on the split-by-head backward); refuse it on the non-split path
            # so a mis-config can never silently no-op.
            if not self.split_by_head:
                raise ValueError(
                    "pose_grad_adaptive (APGC) requires split_by_head=True: the adaptive "
                    "controller cadences the split-by-head PoseNet path; the non-split "
                    "path computes both heads in one graph (nothing to throttle)."
                )
            if float(self.pose_grad_floor_tol) <= 0.0:
                raise ValueError(
                    f"pose_grad_floor_tol must be > 0.0 (the deadband half-width); got "
                    f"{self.pose_grad_floor_tol}"
                )
            if int(self.pose_grad_k_max) < 1:
                raise ValueError(
                    f"pose_grad_k_max must be >= 1 (the sparsest cadence / measurement "
                    f"floor); got {self.pose_grad_k_max}"
                )
            if int(self.pose_grad_trend_window) < 2:
                raise ValueError(
                    f"pose_grad_trend_window must be >= 2 (the slope needs >=2 samples); "
                    f"got {self.pose_grad_trend_window}"
                )
        if self.pose_equimarginal_enabled:
            # Lever A needs the SEPARABLE per-axis frame-cotangent norms; only the split-by-head backward
            # computes ``cot_seg`` + ``cot_pose`` as distinct tensors. Refuse on the non-split path so the
            # flag can never silently no-op (a silent no-op would let the operator believe the equimarginal
            # controller is active when it is not). The constructor (build_curriculum runtime) instantiates
            # the controller from these validated params.
            if not self.split_by_head:
                raise ValueError(
                    "pose_equimarginal_enabled requires split_by_head=True: the controller balances the "
                    "per-axis frame-cotangent norms (‖cot_seg‖, ‖cot_pose‖) the split-by-head backward "
                    "computes separately; the non-split fused path has no separable per-axis cotangent."
                )
            if not (float(self.pose_equimarginal_rho) > 0.0):
                raise ValueError(
                    f"pose_equimarginal_rho must be > 0 (got {self.pose_equimarginal_rho})"
                )
            if not (0.0 <= float(self.pose_equimarginal_decay) < 1.0):
                raise ValueError(
                    f"pose_equimarginal_decay must be in [0,1) (got {self.pose_equimarginal_decay})"
                )
            if not (float(self.pose_equimarginal_tol) > 0.0):
                raise ValueError(
                    f"pose_equimarginal_tol must be > 0 (got {self.pose_equimarginal_tol})"
                )
            if not (
                0.0 < float(self.pose_equimarginal_bound_lo) <= 1.0 <= float(self.pose_equimarginal_bound_hi)
            ):
                raise ValueError(
                    "require 0 < pose_equimarginal_bound_lo <= 1 <= pose_equimarginal_bound_hi (got "
                    f"{self.pose_equimarginal_bound_lo}, {self.pose_equimarginal_bound_hi})"
                )
        if self.pose_dim_weights is not None:
            # Lever C: validate + renormalise the length-6 weights at construction so a mis-shaped/negative
            # weight fails CLOSED here, not silently inside the loss. The normalised tuple (mean 1.0) is
            # stored so uniform-after-norm is byte-identical and the loss path can use it directly.
            from tac.torch_vehicle.pose_dim_weights import normalise_pose_dim_weights

            self.pose_dim_weights = normalise_pose_dim_weights(self.pose_dim_weights)
        if self.lever4_variable_level_export_enabled and self.variable_level_waterfill_enabled:
            raise ValueError(
                "lever4_variable_level_export_enabled and variable_level_waterfill_enabled "
                "are mutually exclusive: both replace the SAME export decoder blob with a "
                "variable-level grid, but from DIFFERENT level sources (Lever-4's online "
                "score-sensitivity EMA vs the offline measured RD table). Enable exactly one."
            )
        if float(self.weight_entropy_penalty_lambda) < 0.0:
            raise ValueError(
                "weight_entropy_penalty_lambda must be >= 0.0 (0.0 = OFF / byte-identical); "
                f"got {self.weight_entropy_penalty_lambda}"
            )
        if int(self.weight_entropy_penalty_stage_min) < 0:
            raise ValueError(
                "weight_entropy_penalty_stage_min must be >= 0 (the first stage index the "
                f"penalty is active); got {self.weight_entropy_penalty_stage_min}"
            )
        if float(self.weight_entropy_penalty_init_scale) <= 0.0:
            raise ValueError(
                "weight_entropy_penalty_init_scale must be > 0 (the per-channel prior's "
                f"initial logistic scale); got {self.weight_entropy_penalty_init_scale}"
            )
        if self.warm_start_dir is not None:
            self.warm_start_dir = Path(self.warm_start_dir)
            # FiLM warm-start IS supported (the decoupled-oomph path): _load_warm_start_into
            # detects the pose-FiLM wrapper and loads the vendored ckpt into its inner
            # ``.decoder`` submodule, leaving the FiLM at identity-init (f0 == vendored rgb_0).
            # So a converged no-FiLM ckpt warm-starts a FiLM decoder cleanly.
        if self.taper_channels is not None:
            self.taper_channels = [int(c) for c in self.taper_channels]
            if len(self.taper_channels) != 7:
                raise ValueError(
                    f"taper_channels must have 7 stages (the HNeRV 6x8→384x512 ladder); "
                    f"got {len(self.taper_channels)}: {self.taper_channels}"
                )
            if any(c <= 0 for c in self.taper_channels):
                raise ValueError(f"taper_channels must all be positive; got {self.taper_channels}")
            if self.taper_channels[-1] < 2:
                raise ValueError(
                    "taper_channels final stage must be >= 2 (the refine block uses "
                    f"final//2 as its width); got {self.taper_channels[-1]}"
                )
            # taper + FiLM COMPOSITION (the bind-all production combo: solved taper +
            # FiLM-v2 pose decouple). The FiLM-v2 wrapper (``PoseFiLMHNeRVWrapperV2``) wraps
            # the INNER decoder via ``.decoder`` and reads only its public surface
            # (``channels[-1]`` / ``stem`` / ``blocks`` / ``skips`` / ``ps`` / ``refine`` /
            # ``rgb_0`` / ``rgb_1`` / ``base_h`` / ``base_w``) — ALL of which the
            # ``ConfigurableTaperHNeRVDecoder`` exposes identically to the vendored decoder —
            # so v2 composes with the taper carrier cleanly (``_new_decoder`` already wraps
            # whatever ``_new_vendored_decoder`` builds, taper or not). v1 stem-injection is
            # NOT supported on a taper (it injects on the shared stem channel ``channels[0]``
            # and feeds both heads — untested on a re-taper AND it couples d_pose into d_seg,
            # the v1 instability v2 fixes), so taper + FiLM is restricted to v2.
            if self.pose_film_enabled and int(self.pose_film_version) != 2:
                raise ValueError(
                    "taper_channels + pose_film requires pose_film_version=2: the v2 "
                    "residual-rgb0 wrapper composes with the configurable-taper carrier "
                    "(it reads only the decoder's public surface); v1 stem-injection is not "
                    "supported on a re-taper (set pose_film_version=2 or disable FiLM)."
                )
        if self.kd_warm_start_dir is not None:
            self.kd_warm_start_dir = Path(self.kd_warm_start_dir)
            # KD-warm-start is the RE-TAPER warm-start path: it REQUIRES taper_channels (the
            # student is the solved taper; a same-taper warm-start uses warm_start_dir
            # directly with no KD). Mutually exclusive with warm_start_dir (two distinct
            # stage-0 init sources for the SAME stage-0 decoder/latents).
            if self.taper_channels is None:
                raise ValueError(
                    "kd_warm_start_dir requires taper_channels (the student is the solved "
                    "re-taper that the basin teacher is distilled into; a same-taper "
                    "warm-start should use warm_start_dir directly — no KD needed)."
                )
            if self.warm_start_dir is not None:
                raise ValueError(
                    "kd_warm_start_dir and warm_start_dir are mutually exclusive: both set "
                    "the stage-0 init, by DIFFERENT mechanisms (KD distillation of the "
                    "basin into a re-taper vs a strict same-taper decoder load). Use exactly "
                    "one."
                )
            if int(self.kd_warm_epochs) < 1:
                raise ValueError(
                    f"kd_warm_epochs must be >= 1 (the KD warm-up is a non-empty PREFIX of "
                    f"stage 0); got {self.kd_warm_epochs}"
                )
            if float(self.kd_warm_lr) <= 0.0:
                raise ValueError(f"kd_warm_lr must be > 0.0; got {self.kd_warm_lr}")
        self.out_dir = Path(self.out_dir)


@dataclass
class _EvalSnapshot:
    """An immutable POINT-IN-TIME CPU copy of the EMA shadow the eval scores.

    Captured in the MAIN thread (cheap), then either evaluated inline (sync) or
    handed to a background worker (async). Because it is a deep CPU copy, the
    training loop may keep mutating the live (MPS) EMA shadow without racing the
    eval — guaranteeing the async numbers equal the sync numbers on the SAME
    snapshot (the no-regression contract)."""

    ema_sd: dict[str, torch.Tensor]
    ema_latents: torch.Tensor
    # Lever-4 online score-sensitivity EMA (``||∂S/∂w_t||``), snapshotted alongside
    # the weights so the EXPORT can build a variable-level decoder blob from it when
    # ``cfg.lever4_variable_level_export_enabled`` (the R14 unification). ``None`` when
    # the lever/flag is off — the export then stays byte-identical to the vendored path.
    tensor_sensitivity_ema: dict[str, float] | None = None


@dataclass
class _StageRuntime:
    """Mutable per-stage training objects (decoder/optimizers/sched/EMA)."""

    decoder: nn.Module
    latents: nn.Parameter
    ema_decoder: nn.Module
    ema_latents: torch.Tensor
    adamw_opt: torch.optim.Optimizer
    muon_opt: Any  # vendored Muon or None
    adamw_sched: Any
    muon_sched: Any
    muon_params: list
    # The grad-CLIP set for the AdamW side: the decoder AdamW-group params PLUS the
    # FiLM params (which live in a SEPARATE capped-LR AdamW group but must be clipped
    # together). NOT "the AdamW-only group" — it is the union clipped at each step.
    adamw_clip_params: list
    # Lever 4 (score-aware QAT) per-tensor sensitivity EMA ``s_t = ||∂S/∂w_t||``,
    # accumulated from the score-domain loss backward (the EMA smooths early-train
    # noise). Empty until the first backward seeds it — while empty the score-aware
    # QAT falls back to the vendored uniform 127-level grid (bit-identical), so the
    # first steps of a score-aware-QAT stage are unchanged from uniform QAT.
    tensor_sensitivity_ema: dict[str, float] = field(default_factory=dict)


class TorchVehicleDriver:
    """Drives the faithful PR95 8-stage curriculum with resume + telemetry.

    Usage (production)::

        ctx = RealScorerContext(video_path, device)
        driver = TorchVehicleDriver(cfg, scorer=ctx, vendored=import_vendored_bundle())
        driver.run()  # resumes from cfg.out_dir if a checkpoint is present
    """

    def __init__(
        self,
        cfg: TorchVehicleConfig,
        *,
        scorer: ScorerContext,
        vendored: VendoredBundle,
        curriculum: list[StageSpec] | None = None,
    ):
        self.cfg = cfg
        self.scorer = scorer
        self.v = vendored
        # AUTHORITY/eval device (CPU-TRUSTED or CUDA) — the exact d_seg/d_pose live
        # here. ``train_device`` is the GRADIENT backend (may be MPS for the 104x).
        self.device = torch.device(cfg.device)
        self.train_device = torch.device(cfg.train_device or cfg.device)
        self.split_device = self.train_device != self.device
        # Split-by-HEAD: SegNet grad on train_device (MPS), PoseNet grad on the CPU
        # authority, summed at the frame tensor. Honored only when the scorer context
        # exposes the per-head forwards (RealScorerContext / SyntheticScorerContext).
        self.split_by_head = bool(cfg.split_by_head) and getattr(
            scorer, "split_by_head", False
        )
        self.n_pairs = int(scorer.n_pairs)
        if curriculum is None:
            from tac.torch_vehicle.curriculum import build_curriculum

            curriculum = build_curriculum(
                total_epoch_budget=cfg.total_epoch_budget,
                ema_decay=cfg.ema_decay,
                eval_every=cfg.eval_every,
            )
        self.curriculum = curriculum
        self.telemetry = TelemetryWriter(
            cfg.out_dir,
            run_meta={
                "base_channels": cfg.base_channels,
                "latent_dim": cfg.latent_dim,
                "n_pairs": self.n_pairs,
                "total_epoch_budget": cfg.total_epoch_budget,
                "ema_decay": cfg.ema_decay,
                "device": cfg.device,
            },
        )
        self.best_score = self.telemetry.best_score
        self.best_ep = self.telemetry.best_ep
        self.best_stage = self.telemetry.best_stage
        self._global_epoch = 0
        # EMA warmup step counter (only consulted when ``cfg.ema_warmup``): the number of
        # EMA updates applied so far this run (0 at the first optimizer step). Drives the
        # bias-corrected warmup decay ``min(spec.ema_decay, (t+1)/(t+10))``. Left at 0 and
        # never read on the default (constant-decay) path — byte-identical.
        self._ema_step = 0
        # Pose-throttle state: the last-computed training pose_mse (None until the first
        # epoch that computes the pose head). Drives both the resume-threshold guard and
        # the telemetry carry on skipped epochs. See ``pose_grad_every_k``.
        self._last_pose_mse: float | None = None
        # APGC (Adaptive Pose-Gradient Controller) state — only consulted when
        # ``cfg.pose_grad_adaptive``; persisted across resume for trajectory correctness.
        #   _pose_floor       — running MIN of the computed training pose_mse (the
        #                       adaptive floor the deadband is anchored to). None until
        #                       the first pose compute.
        #   _pose_mse_hist    — the last ``trend_window`` computed pose_mse values (the
        #                       derivative/slope term). Trimmed to the window.
        #   _last_pose_epoch  — the global epoch of the last pose COMPUTE (the
        #                       measurement-floor reference: never go blind > k_max).
        #                       Init -1 (a SENTINEL meaning "no compute yet") so the
        #                       once-per-epoch bookkeeping gate fires at the very first
        #                       epoch (global_epoch 0); a real compute stamps the actual
        #                       epoch >= 0.
        # All three left at their default on the static/non-adaptive path → byte-identical
        # (round-trip as the same defaults; the controller branch never reads them there).
        self._pose_floor: float | None = None
        self._pose_mse_hist: list[float] = []
        self._last_pose_epoch: int = -1
        # -- Lever A: the EQUIMARGINAL pose-weight controller (only consulted when
        # ``cfg.pose_equimarginal_enabled``; persisted across resume so the w_pose trajectory continues).
        # None on the default path → the pose path uses ``spec.pose_weight`` unmodified (byte-identical).
        self._equimarginal_ctrl = None
        if bool(cfg.pose_equimarginal_enabled):
            from tac.torch_vehicle.equimarginal_pose_weight import (
                EquimarginalPoseWeightController,
            )

            self._equimarginal_ctrl = EquimarginalPoseWeightController(
                rho=float(cfg.pose_equimarginal_rho),
                decay=float(cfg.pose_equimarginal_decay),
                tol=float(cfg.pose_equimarginal_tol),
                bound_lo=float(cfg.pose_equimarginal_bound_lo),
                bound_hi=float(cfg.pose_equimarginal_bound_hi),
            )
        # The most-recent equimarginal telemetry row (observability #305); None until the controller runs.
        self._last_equimarginal_telemetry: dict | None = None
        # Test-only hook: simulate a death after this many global epochs (the
        # checkpoint has already landed when the death fires). None in production.
        self._stop_after_global_epoch: int | None = None
        # -- async-eval state (CLAUDE.md "async authority eval" throughput salvage) --
        # The async eval runs in a BACKGROUND THREAD; ``_eval_lock`` serializes the
        # two shared surfaces it touches with the main loop: the BEST-tracker fields
        # (best_score/best_ep/best_stage + the best/ dir write) AND the single
        # TelemetryWriter (which is NOT thread-safe — the main loop also records
        # non-eval rows). ``_eval_thread`` holds the at-most-one in-flight worker so
        # the cadence self-throttles (a new eval epoch arriving while the prior eval
        # is alive is SKIPPED). ``_skipped_evals`` counts the skips for the report.
        self._async_eval = bool(cfg.async_eval)
        self._eval_lock = threading.Lock()
        self._eval_thread: threading.Thread | None = None
        self._skipped_evals = 0
        self._inflight_snapshot_epoch: int | None = None
        # WEIGHT-ENTROPY PENALTY (Ballé rate lever) state. Built LAZILY on the first
        # ``_build_stage_runtime`` when ``cfg.weight_entropy_penalty_lambda > 0`` (the
        # decoder it sizes against is created in ``run()``), then PERSISTED across
        # stages (the learned per-channel prior carries, like the weight EMA). Its
        # params are re-added to each stage's AdamW group (PR95 rebuilds the optimizer
        # per stage). ``None`` on the default path → byte-identical (never built,
        # never read). ``_cur_stage_index`` lets ``_weight_regularizers`` honor the
        # ``weight_entropy_penalty_stage_min`` schedule (mirrors C1a's late-stage ramp).
        self._weight_entropy_penalty: nn.Module | None = None
        self._cur_stage_index: int = 0
        # The current-stage Lever-4 sensitivity EMA (set in _train_one_epoch); read by
        # the weight-entropy WATERFILL allocation. Empty default = byte-share-only
        # waterfill (or, with waterfill off, ignored entirely).
        self._cur_tensor_sensitivity_ema: dict[str, float] = {}
        # The final LIVE (non-EMA) decoder, set at each stage boundary in ``run()`` (None
        # before the first stage completes). Post-run inspection surface for the
        # weight-entropy λ-on/off A/B; NOT a score surface (BEST/export use the EMA shadow).
        self._final_decoder: nn.Module | None = None

    # -- architecture construction (base_ch threaded — the FINDING-1 fix) -----
    def _new_vendored_decoder(self, device: torch.device | None = None) -> nn.Module:
        """Build a fresh base_ch-threaded VENDORED decoder (no FiLM).

        When ``cfg.taper_channels`` is set, build the ``ConfigurableTaperHNeRVDecoder``
        (a faithful generalization of the vendored decoder whose ONLY difference is the
        channel schedule) instead — the #1 structural lever. DEFAULT (None) returns the
        vendored decoder unchanged (byte-identical)."""
        dev = device if device is not None else self.train_device
        if self.cfg.taper_channels is not None:
            from tac.torch_vehicle.configurable_taper_decoder import (
                ConfigurableTaperHNeRVDecoder,
            )

            return ConfigurableTaperHNeRVDecoder(
                latent_dim=self.cfg.latent_dim,
                base_channels=self.cfg.base_channels,
                eval_size=(_EVAL_H, _EVAL_W),
                channels=self.cfg.taper_channels,
            ).to(dev)
        return self.v.HNeRVDecoder(
            latent_dim=self.cfg.latent_dim,
            base_channels=self.cfg.base_channels,
            eval_size=(_EVAL_H, _EVAL_W),
        ).to(dev)

    def _new_decoder(self, device: torch.device | None = None) -> nn.Module:
        """Build a fresh base_ch-threaded decoder on ``device`` (default: the
        TRAIN device — the gradient backend). The parse-back EVAL decoder is built
        on the AUTHORITY device explicitly (``device=self.device``).

        Lever 3 (``cfg.pose_film_enabled``): wrap the vendored decoder in a
        :class:`~tac.torch_vehicle.pose_film.PoseFiLMHNeRVWrapper` (stem-injection
        FiLM, identity at init) and SEED its ``stored_pose`` buffer from the GT
        PoseNet pose (``scorer.pose_targets``). DEFAULT-OFF returns the vendored
        decoder unchanged (byte-identical)."""
        dev = device if device is not None else self.train_device
        if not self.cfg.pose_film_enabled:
            return self._new_vendored_decoder(dev)
        if int(self.cfg.pose_film_version) == 2:
            from tac.torch_vehicle.pose_film_v2 import (
                PoseFiLMHNeRVWrapperV2 as _PoseFiLMWrapper,
            )
        else:
            from tac.torch_vehicle.pose_film import (
                PoseFiLMHNeRVWrapper as _PoseFiLMWrapper,
            )

        vendored = self._new_vendored_decoder(dev)
        # RNG-NEUTRAL FiLM construction (the from-scratch A/B fix): the
        # ``_PoseFiLM`` module init draws from the global torch RNG. In a FRESH
        # (non-resume) run the very next RNG consumer is the stage-1 RANDOM LATENT
        # draw (``run()``: ``torch.randn(n_pairs, latent_dim)``). If the FiLM
        # construction were allowed to advance the global stream, the FiLM-on arm's
        # initial latents would DIVERGE from the no-FiLM basin's (same distribution,
        # different realization) — breaking the bit-shared-init contract the
        # from-scratch decisive A/B depends on. We SNAPSHOT the CPU RNG state before
        # the FiLM build and RESTORE it after, so building FiLM consumes NET-ZERO
        # global RNG: the subsequent latent draw lands at the SAME stream position the
        # no-FiLM basin draws from → bit-identical initial latents. (The FiLM params
        # are still randomly initialized — from the snapshotted state — and are
        # immediately overwritten to IDENTITY by ``_PoseFiLM``'s zero-init ``fc2``, so
        # gamma=1/beta=0 at init regardless; only the global-stream POSITION is
        # preserved.) BASIN-SAFE: the basin never enters this branch
        # (``pose_film_enabled=False``) so its trajectory is byte-identical. The
        # device RNG (e.g. MPS) is NOT touched here because the latent draw is on CPU
        # (``run()`` draws on CPU then moves), so the CPU snapshot/restore is the
        # complete fix for the reproducible-init contract.
        _rng_state = torch.get_rng_state()
        try:
            wrapper = _PoseFiLMWrapper(
                vendored,
                n_pairs=self.n_pairs,
                film_hidden=self.cfg.pose_film_hidden,
            ).to(dev)
        finally:
            torch.set_rng_state(_rng_state)
        # Seed the STORED pose from the GT PoseNet pose (side-info, not learned).
        wrapper.set_stored_pose(self.scorer.pose_targets[: self.n_pairs].to(dev))
        return wrapper

    # -- warm-start fine-tune (the disambiguator + general fine-tune entry point) --
    def _load_warm_start_into(self, decoder: nn.Module) -> nn.Parameter:
        """Load the converged decoder weights from ``cfg.warm_start_dir`` INTO
        ``decoder`` (strict) and return the stored latents as a fresh stage-0
        ``nn.Parameter`` on the TRAIN device. Replaces the random from-0 init so a
        run CONTINUES from a converged basin. STRICT by construction: a shape/key
        mismatch (wrong base_ch / latent_dim / FiLM wrapper) raises here rather than
        silently mis-loading — the apples-to-apples fine-tune contract depends on the
        two arms loading the SAME bytes into the SAME architecture."""
        wd = Path(self.cfg.warm_start_dir)  # type: ignore[arg-type]
        dec_path = wd / "best_ema_decoder.pt"
        lat_path = wd / "best_ema_latents.pt"
        if not dec_path.exists() or not lat_path.exists():
            raise FileNotFoundError(
                f"warm_start_dir={wd} must hold best_ema_decoder.pt + best_ema_latents.pt "
                f"(the canonical converged-checkpoint layout this driver writes); missing "
                f"{'best_ema_decoder.pt' if not dec_path.exists() else 'best_ema_latents.pt'}"
            )
        # The driver saves a BARE state_dict / BARE tensor (torch.save(ema_sd, ...) /
        # torch.save(ema_latents, ...)); tolerate a dict-wrapped variant defensively.
        sd = torch.load(dec_path, map_location="cpu", weights_only=False)
        # A bare HNeRV state_dict has param-name keys (e.g. "rgb_1.weight") — never a
        # literal "state_dict" key — so this unwrap only fires on a dict-wrapped save.
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        lat = torch.load(lat_path, map_location="cpu", weights_only=False)
        if isinstance(lat, dict):
            if not lat:
                raise ValueError(f"warm-start latents dict at {lat_path} is empty")
            lat = lat.get("latents", next(iter(lat.values())))
        # STRICT load — a mismatch is a misconfigured warm-start, not a silent partial load.
        # FiLM-WARM-START (the decoupled-oomph path): when the decoder is a pose-FiLM wrapper
        # (children = decoder/pose_mlp/film_resid), the saved vendored ckpt has the INNER
        # decoder's keys (e.g. "rgb_1.weight"), not the wrapper's prefixed keys. Load into the
        # inner ``.decoder`` submodule (strict) and LEAVE the FiLM at its identity-init
        # (proj/beta zero → film_resid==0 → f0 renders bit-equal to the vendored rgb_0). So the
        # converged 0.00359 trunk/heads carry over AND the d_seg⊥pose decoupling is active —
        # the math-optimal start for cranking the seg (oomph) loss without a pose cost.
        target = decoder
        # FiLM-wrapper detect: BOTH v1 (children: decoder/pose_film) and v2 (decoder/pose_mlp/
        # film_resid) hold the vendored decoder in a ``.decoder`` submodule. A plain vendored
        # OR ConfigurableTaper decoder has NO ``.decoder`` child (its children are stem/blocks/
        # …), so ``.decoder`` being a Module uniquely identifies a FiLM wrapper — robust for v1
        # AND v2 (keying on pose_mlp would silently miss v1 → a broken v1 warm-start).
        if isinstance(getattr(decoder, "decoder", None), nn.Module):
            target = decoder.decoder
        target.load_state_dict({k: v.to(self.train_device) for k, v in sd.items()})
        if int(lat.shape[0]) != self.n_pairs or int(lat.shape[1]) != self.cfg.latent_dim:
            raise ValueError(
                f"warm-start latents shape {tuple(lat.shape)} != (n_pairs={self.n_pairs}, "
                f"latent_dim={self.cfg.latent_dim}); cannot warm-start a different basis"
            )
        return nn.Parameter(lat.detach().clone().to(self.train_device))

    # -- KD warm-start (the wall-clock resolution: distill the basin into the re-taper) --
    def _run_kd_warm_up(self, rt: _StageRuntime, spec: StageSpec) -> None:
        """Run the KD WARM-UP phase: distill the FROZEN basin teacher (the converged
        VENDORED-taper decoder from ``cfg.kd_warm_start_dir``) into the re-tapered student
        (``rt.decoder``) on the (directly warm-started) latents (``rt.latents``), for
        ``cfg.kd_warm_epochs`` epochs of frame-MSE distillation. After the distillation the
        student's EMA shadow is RE-SYNCED to the distilled state (so the shadow tracks the
        distilled student, not the random init it deep-copied at ``_build_stage_runtime``).

        The teacher is the basin's OWN architecture (the vendored taper — the basin had no
        ``taper_channels``), so it is the plain vendored ``HNeRVDecoder`` at the basin's
        ``base_channels``/``latent_dim`` (NOT the configurable-taper student). It is FROZEN
        (eval + requires_grad False + rendered under no_grad) — the KD step can never train
        it (the NO-FAKE contract).

        ``kd_warm_epochs`` MUST be <= the stage-0 epoch budget (else the whole of stage 0 is
        KD with no score-aware continuation — refused so the bind-all contract holds).
        Records a KD-warm-up telemetry row (the first/last frame-MSE — ``last < first`` is
        the proof the distillation ran). Only called when ``do_kd_warm_up`` (stage-0 fresh
        KD init); a no-op for any other path."""
        from tac.torch_vehicle.kd_warm_start import build_frozen_teacher, kd_warm_up_decoder

        kd_epochs = int(self.cfg.kd_warm_epochs)
        if kd_epochs > spec.epochs:
            raise ValueError(
                f"kd_warm_epochs={kd_epochs} must be <= the stage-0 epoch budget "
                f"({spec.epochs}); the KD warm-up is a PREFIX of stage 0 and must leave "
                f"room for the score-aware curriculum to continue after it."
            )
        # The teacher is the basin's vendored-taper decoder (NO FiLM, NO configurable taper).
        teacher = build_frozen_teacher(
            self.cfg.kd_warm_start_dir,
            vendored_decoder_cls=self.v.HNeRVDecoder,
            latent_dim=self.cfg.latent_dim,
            base_channels=self.cfg.base_channels,
            device=self.train_device,
            eval_size=(_EVAL_H, _EVAL_W),
        )
        stats = kd_warm_up_decoder(
            student=rt.decoder,
            teacher=teacher,
            latents=rt.latents,
            n_pairs=self.n_pairs,
            epochs=kd_epochs,
            batch_size=spec.batch_size,
            lr=float(self.cfg.kd_warm_lr),
            train_latents=bool(self.cfg.kd_warm_train_latents),
            latent_lr_mult=spec.latent_lr_mult,
            seed=self.cfg.seed,
            device=self.train_device,
            pose_film_enabled=self.cfg.pose_film_enabled,
        )
        # RE-SYNC the EMA shadow to the distilled student (the shadow deep-copied the random
        # init at _build_stage_runtime; after KD it must reflect the distilled state so the
        # subsequent EMA tracking + the BEST-from-shadow eval start from the distilled basin).
        rt.ema_decoder.load_state_dict(rt.decoder.state_dict())
        rt.ema_latents = rt.latents.detach().clone()
        # Telemetry: a KD-warm-up row (NOT an eval row — TRAIN-time priming). first/last
        # frame-MSE prove the distillation actually lowered the student-vs-teacher distortion.
        self.telemetry.record(
            EpochRecord(
                stage_index=0,
                stage_name=f"{spec.name}__kd_warm_up",
                epoch_in_stage=kd_epochs,
                global_epoch=self._global_epoch,
                loss=stats["last_loss"],
                pose_mse=float("nan"),
                adamw_lr=float(self.cfg.kd_warm_lr),
                muon_lr=None,
                grad_norm_adamw=None,
                grad_norm_muon=None,
                evaluated=False,
            )
        )

    def _film_param_ids(self, decoder: nn.Module) -> set[int]:
        """The ``id()`` set of the pose-FiLM params on ``decoder`` (the wrapper), by
        version-keyed name prefix. EMPTY when ``pose_film_enabled`` is False (no FiLM
        params exist → the optimizer/clip paths are byte-identical to the vendored
        baseline). Used by :meth:`_build_stage_runtime` to route the FiLM to a capped
        AdamW group + exclude it from Muon (the §A review fix)."""
        if not self.cfg.pose_film_enabled:
            return set()
        prefixes = _FILM_PARAM_PREFIXES[int(self.cfg.pose_film_version)]
        return {
            id(p)
            for n, p in decoder.named_parameters()
            if any(n.startswith(pre) for pre in prefixes)
        }

    def _rgb0_param_ids(self, decoder: nn.Module) -> set[int]:
        """The ``id()`` set of the ``rgb_0`` head params on ``decoder``. Used ONLY by the
        ``pose_film_rgb0_pose_trainable`` refinement to EXCLUDE rgb_0 from the seg-only-
        restore set (so the pose loss may train the frame-0 head, which SegNet never reads).

        rgb_0 lives on the INNER vendored/taper decoder (``decoder.rgb_0`` on the FiLM
        wrapper, or ``decoder.rgb_0`` directly on a bare decoder). We resolve the actual
        ``rgb_0`` Module — robust to the wrapper vs bare layout AND any future rename of the
        wrapper's submodule attribute — and key on its parameter ``id()``s (not name
        strings, so it is decoupled from the named-parameter prefix). EMPTY when the
        decoder exposes no ``rgb_0`` (defensive: never silently masks the wrong tensor)."""
        inner = decoder
        # The FiLM wrapper holds the vendored/taper decoder in ``.decoder``; the head lives
        # there. A bare vendored/taper decoder has ``rgb_0`` directly. Resolve whichever.
        if not hasattr(inner, "rgb_0") and isinstance(
            getattr(inner, "decoder", None), nn.Module
        ):
            inner = inner.decoder
        rgb0 = getattr(inner, "rgb_0", None)
        if not isinstance(rgb0, nn.Module):
            return set()
        return {id(p) for p in rgb0.parameters() if p.requires_grad}

    def _non_film_grad_params(
        self, decoder: nn.Module, latents: torch.Tensor
    ) -> list[torch.Tensor]:
        """The SHARED-decoder params (trunk + skips + blocks + refine + rgb_0/rgb_1 heads)
        PLUS the latents — i.e. every trainable tensor the POSE objective must NOT update
        under ``pose_film_trunk_stopgrad``. The FiLM pose path (``pose_mlp`` + ``film_resid``)
        is EXCLUDED (it is the one place the pose gradient is allowed to land).

        rgb_0 REFINEMENT (``cfg.pose_film_rgb0_pose_trainable``): when ON, the ``rgb_0``
        head params are ALSO excluded from this set, so the pose backward's contribution to
        them is KEPT (not restored to seg-only). This is EXACT-decoupling-preserving: SegNet
        reads ONLY frame-1 (``rgb_1``), so ``∂d_seg/∂(rgb_0 params) = 0`` — training rgb_0
        with the pose loss costs NO d_seg, while giving the pose objective strictly more
        capacity (rgb_0 IS the pose-conditioned frame-0 head). DEFAULT (flag OFF) keeps
        rgb_0 in the seg-only-restore set, byte-identical to the base trunk-stopgrad.

        Used by :meth:`_split_by_head_backward` to snapshot+restore the non-FiLM ``.grad``
        across the (separated) pose backward, so the pose contribution is removed from the
        shared graph and ∂(trunk,latents,rgb_1[,rgb_0])/∂(pose-objective)=0 EXACTLY. The
        latents are an ``nn.Parameter`` so they are masked too (the pose loss must not move
        the shared latent code that produces the seg frame f1)."""
        film_ids = self._film_param_ids(decoder)
        excluded_ids = set(film_ids)
        if self.cfg.pose_film_rgb0_pose_trainable:
            excluded_ids |= self._rgb0_param_ids(decoder)
        params = [
            p
            for p in decoder.parameters()
            if p.requires_grad and id(p) not in excluded_ids
        ]
        if isinstance(latents, torch.Tensor) and latents.requires_grad:
            params.append(latents)
        return params

    def _build_stage_runtime(
        self,
        spec: StageSpec,
        *,
        decoder: nn.Module,
        latents: nn.Parameter,
        ema_decoder: nn.Module | None,
        ema_latents: torch.Tensor | None,
    ) -> _StageRuntime:
        """Build fresh optimizers + cosine schedulers for a stage (faithful: PR95
        resets the optimizer per stage; weights/EMA carry)."""
        if ema_decoder is None:
            ema_decoder = deepcopy(decoder)
        if ema_latents is None:
            ema_latents = latents.data.clone()

        # Lever-3 FiLM param routing (the §A review fix; #118 SEALED). When the
        # pose-FiLM is on, its params (``pose_mlp`` + ``film_resid`` for v2 / ``pose_film``
        # for v1) get a DEDICATED AdamW group at the capped LR and are EXCLUDED from
        # Muon (whose orthogonalized SGD would bypass the cap). They STAY in the
        # clip set (``adamw_clip_params``) so grad-clip still covers them. Empty when
        # FiLM is off → the basin/control optimizer is byte-identical to the legacy path.
        film_ids = self._film_param_ids(decoder)
        film_params = (
            [p for _n, p in decoder.named_parameters() if id(p) in film_ids and p.requires_grad]
            if film_ids
            else []
        )
        film_lr = min(spec.adamw_lr, _FILM_LR_CAP)

        # WEIGHT-ENTROPY PENALTY (Ballé rate lever) param routing — DEFAULT-PRESERVING.
        # When ``cfg.weight_entropy_penalty_lambda > 0``, build the penalty ONCE (sized to
        # THIS decoder; persisted across stages so the learned per-channel prior carries)
        # and add its LEARNABLE prior params (loc / raw_scale / raw_shape per coded weight
        # tensor) to a DEDICATED AdamW group at ``spec.adamw_lr`` so the prior trains. Empty
        # list when the lever is off → ``adamw_groups`` is byte-identical to the legacy path
        # (no extra group, no optimizer-state change). The penalty's params are NOT in the
        # clip set or the Muon partition (they are prior params, not decoder weights — they
        # carry no frame gradient and must not be orthogonalized by Muon).
        penalty_params: list[torch.Tensor] = []
        if float(self.cfg.weight_entropy_penalty_lambda) > 0.0:
            if self._weight_entropy_penalty is None:
                from tac.torch_vehicle.weight_entropy_penalty import WeightEntropyPenalty

                self._weight_entropy_penalty = WeightEntropyPenalty(
                    decoder,
                    init_scale=float(self.cfg.weight_entropy_penalty_init_scale),
                ).to(self.train_device)
            penalty_params = [
                p for p in self._weight_entropy_penalty.parameters() if p.requires_grad
            ]

        if spec.use_muon:
            muon_params, adamw_params = self.v.partition_params_for_muon(decoder)
            if film_ids:
                muon_params = [p for p in muon_params if id(p) not in film_ids]
                adamw_params = [p for p in adamw_params if id(p) not in film_ids]
            muon_opt = self.v.Muon(
                muon_params,
                lr=spec.muon_lr,
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                weight_decay=spec.muon_weight_decay,
            )
            adamw_groups = [
                {"params": adamw_params, "lr": spec.adamw_lr},
                {"params": [latents], "lr": spec.adamw_lr * spec.latent_lr_mult},
            ]
            if film_params:
                adamw_groups.append({"params": film_params, "lr": film_lr})
            if penalty_params:
                adamw_groups.append({"params": penalty_params, "lr": spec.adamw_lr})
            adamw_opt = torch.optim.AdamW(adamw_groups, weight_decay=0.0)
        else:
            muon_opt = None
            muon_params = []
            all_params = list(decoder.parameters())
            adamw_params = (
                [p for p in all_params if id(p) not in film_ids] if film_ids else all_params
            )
            adamw_groups = [
                {"params": adamw_params, "lr": spec.adamw_lr},
                {"params": [latents], "lr": spec.adamw_lr * spec.latent_lr_mult},
            ]
            if film_params:
                adamw_groups.append({"params": film_params, "lr": film_lr})
            if penalty_params:
                adamw_groups.append({"params": penalty_params, "lr": spec.adamw_lr})
            adamw_opt = torch.optim.AdamW(adamw_groups, weight_decay=0.0)
        # The clip set = the decoder AdamW params PLUS the FiLM params (so grad-clip at
        # ``_train_one_epoch`` covers the capped FiLM group too). == adamw_params when
        # FiLM is off (byte-identical clip).
        adamw_clip_params = adamw_params + film_params

        eta_min_ratio = max(spec.lr_floor_ratio / spec.adamw_lr, 1e-3)

        def lr_lambda(epoch: int) -> float:
            return max(0.5 * (1 + math.cos(math.pi * epoch / spec.epochs)), eta_min_ratio)

        # E#5 per-stage LR warmup (DEFAULT-OFF → returns lr_lambda unchanged =
        # byte-identical). When on, the first warmup_frac·epochs ramp LR floor→cosine.
        warmup_frac = float(getattr(self.cfg, "stage_lr_warmup_frac", 0.0))
        warmup_start_ratio = float(getattr(self.cfg, "stage_lr_warmup_start_ratio", 0.1))
        lr_lambda = _warmup_wrap(
            lr_lambda,
            warmup_frac=warmup_frac,
            stage_epochs=spec.epochs,
            start_ratio=warmup_start_ratio,
        )

        adamw_sched = torch.optim.lr_scheduler.LambdaLR(adamw_opt, lr_lambda)
        # WS-A/M3 Muon LR-floor fix (opt-in, default-off → byte-identical). The shared
        # lr_lambda floors at eta_min_ratio = lr_floor_ratio/ADAMW_lr (=0.5 at stage 8 →
        # Muon LR never anneals below 50% of peak). When muon_lr_floor_fix is on, Muon gets
        # its OWN floor keyed to muon_lr (the intended absolute floor). Off → Muon shares
        # the AdamW lambda exactly as before.
        if muon_opt is None:
            muon_sched = None
        elif getattr(self.cfg, "muon_lr_floor_fix", False) and spec.muon_lr:
            muon_eta_min = max(spec.lr_floor_ratio / spec.muon_lr, 1e-3)

            def muon_lr_lambda(epoch: int) -> float:
                return max(0.5 * (1 + math.cos(math.pi * epoch / spec.epochs)), muon_eta_min)

            # E#5 warmup wraps the Muon floor-fix lambda too (DEFAULT-OFF → unchanged).
            muon_lr_lambda = _warmup_wrap(
                muon_lr_lambda,
                warmup_frac=warmup_frac,
                stage_epochs=spec.epochs,
                start_ratio=warmup_start_ratio,
            )
            muon_sched = torch.optim.lr_scheduler.LambdaLR(muon_opt, muon_lr_lambda)
        else:
            # Shares the (already warmup-wrapped) adamw lr_lambda → Muon also warms up.
            muon_sched = torch.optim.lr_scheduler.LambdaLR(muon_opt, lr_lambda)
        return _StageRuntime(
            decoder=decoder,
            latents=latents,
            ema_decoder=ema_decoder,
            ema_latents=ema_latents,
            adamw_opt=adamw_opt,
            muon_opt=muon_opt,
            adamw_sched=adamw_sched,
            muon_sched=muon_sched,
            muon_params=muon_params,
            adamw_clip_params=adamw_clip_params,
        )

    # -- one faithful training epoch (1:1 with common.py) --------------------
    def _train_one_epoch(
        self, rt: _StageRuntime, spec: StageSpec, *, epoch_in_stage: int = 0
    ) -> tuple[float, float, float, float]:
        """Run one epoch; returns (mean_loss, mean_pose_mse, last_grad_adamw, last_grad_muon).

        ``epoch_in_stage`` (0-based) drives the Lever-2 per-epoch seg-temperature
        ANNEAL hook (the OPTIMIZE part). DEFAULT-PRESERVING: when
        ``spec.seg_temperature_end is None`` (the default) the annealed temperature is
        the static ``spec.seg_temperature`` for EVERY epoch, so the call is unchanged;
        the default ``epoch_in_stage=0`` keeps legacy callers (tests) bit-identical.
        """
        from tac.torch_vehicle.curriculum import seg_temperature_for_epoch

        # Lever 2 anneal: the PREDICTION softmax temperature for THIS epoch. NO-OP on
        # the default path (seg_temperature_end is None → returns the static T; and
        # the vendored CE path ignores temperature entirely).
        epoch_temperature = seg_temperature_for_epoch(spec, epoch_in_stage)

        decoder, latents = rt.decoder, rt.latents
        # Expose the CURRENT-stage Lever-4 sensitivity EMA to ``_weight_regularizers`` so
        # the weight-entropy WATERFILL allocation (when enabled) can read it. Default path
        # (waterfill off OR empty EMA) ignores it → byte-identical loss term.
        self._cur_tensor_sensitivity_ema = rt.tensor_sensitivity_ema
        n_pairs = self.n_pairs
        bs = spec.batch_size
        # torch.randperm honors the global torch RNG (captured/restored on resume).
        # Build on CPU then move so the RNG draw is device-independent (MPS has its
        # own RNG stream) — this keeps the resume bit-identical regardless of the
        # train device, and keeps the permutation reproducible vs a CPU-arm A/B.
        pair_indices = torch.randperm(n_pairs).to(self.train_device)
        epoch_loss = 0.0
        epoch_pose = 0.0
        nb = 0
        last_gn_adamw = None
        last_gn_muon = None
        # APGC decides the pose cadence ONCE per epoch (the controller reads the
        # PRE-epoch state — floor/last_pose_mse/trend — so the decision is stable across
        # all batches of this epoch; without this the per-batch state mutation below would
        # make batch 2's decision differ from batch 1's). Mirrors the static throttle,
        # whose ``_global_epoch % _k`` term is also epoch-constant. None on the static
        # path; the controller is consulted only inside the split-by-head branch.
        apgc_do_pose: bool | None = None
        if self.split_by_head and self.cfg.pose_grad_adaptive:
            apgc_do_pose = _adaptive_do_pose(
                self._global_epoch,
                self._pose_floor,
                self._last_pose_mse,
                self._pose_mse_hist,
                tol=float(self.cfg.pose_grad_floor_tol),
                k_max=int(self.cfg.pose_grad_k_max),
                last_pose_epoch=self._last_pose_epoch,
            )

        for batch_start in range(0, n_pairs, bs):
            idx = pair_indices[batch_start : batch_start + bs]
            B = len(idx)

            if spec.use_qat:
                # Lever 4 (score-aware QAT): DEFAULT-PRESERVING. ``score_aware_qat is
                # False`` (the default) uses the vendored UNIFORM 127-level fake-quant
                # — byte-identical to today. When True, the per-tensor INT8 grid is a
                # function of the score-sensitivity EMA (high-sensitivity tensors get a
                # finer grid → argmax boundary protected; low-sensitivity ones a coarser
                # grid → fewer brotli bytes). While the EMA is still empty (early in the
                # stage, before the first backward seeds it) the sensitivity is None →
                # the score-aware path quantizes EVERY tensor at the base 127 levels,
                # bit-identical to the vendored uniform QAT.
                if spec.score_aware_qat:
                    from tac.torch_vehicle.score_aware_qat import apply_score_aware_qat

                    sens = rt.tensor_sensitivity_ema or None
                    originals = apply_score_aware_qat(decoder, sens)
                else:
                    originals = self.v.apply_qat(decoder)
            # Lever 3: when pose-FiLM is on the decoder is a PoseFiLMHNeRVWrapper
            # whose forward needs the per-pair index to look up the stored pose.
            # When off, the vendored decoder forward takes only the latents.
            decoded_pair = (
                decoder(latents[idx], idx)
                if self.cfg.pose_film_enabled
                else decoder(latents[idx])
            )
            if spec.use_qat:
                if spec.score_aware_qat:
                    from tac.torch_vehicle.score_aware_qat import restore_score_aware_qat

                    restore_score_aware_qat(decoder, originals)
                else:
                    self.v.restore_qat(decoder, originals)

            flat = decoded_pair.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
            up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
            down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
            decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)

            decoded_clamped = decoded_bhwc.clamp(0, 255)
            decoded_rounded = decoded_clamped.round()
            decoded_bhwc = decoded_clamped + (decoded_rounded - decoded_clamped).detach()

            rt.adamw_opt.zero_grad()
            if rt.muon_opt is not None:
                rt.muon_opt.zero_grad()

            if self.split_by_head:
                if apgc_do_pose is not None:
                    # APGC (closed-loop): the static k/threshold are IGNORED; the
                    # controller decided ONCE per epoch above (stable across batches) —
                    # holds d_pose at its moving floor with minimum spend (drift-arrest on
                    # band breach / rising trend, proportional cadence at floor,
                    # measurement-floor every k_max). See ``_adaptive_do_pose``.
                    do_pose = apgc_do_pose
                else:
                    # Static pose-throttle (the "pose is solved → stop paying for it"
                    # OPEN-loop speed lever): compute the pose cotangent EVERY epoch
                    # (k<=1, byte-identical default), while pose is still converging/
                    # drifted (last pose_mse > resume_threshold), the FIRST time (never
                    # computed), or on the every-k cadence epoch.
                    _k = int(self.cfg.pose_grad_every_k)
                    _thr = float(self.cfg.pose_grad_resume_threshold)
                    do_pose = (
                        _k <= 1
                        or self._last_pose_mse is None
                        or (_thr > 0.0 and self._last_pose_mse > _thr)
                        or (self._global_epoch % _k == 0)
                    )
                loss_val, pose_mse_val = self._split_by_head_backward(
                    decoded_bhwc, idx, spec, temperature=epoch_temperature,
                    compute_pose=do_pose, decoder=decoder, latents=latents,
                )
                # APGC bookkeeping: when pose was COMPUTED this epoch, advance the
                # controller state off the just-measured pose_mse — update the running
                # floor, append to the trend window (trimmed), and stamp the measurement
                # epoch (the measurement-floor reference). Gated to ONCE per epoch
                # (``_last_pose_epoch != _global_epoch``) so a multi-batch epoch advances
                # the floor/trend exactly once (off the first batch's measure), keeping the
                # cadence semantics at EPOCH granularity. Skipped epochs leave the state
                # stale BY DESIGN (the measurement-floor bounds the staleness). Guarded by
                # the adaptive flag so the static/non-split paths are byte-identical.
                if (
                    apgc_do_pose
                    and self._last_pose_epoch != self._global_epoch
                ):
                    _pm = self._last_pose_mse  # updated inside the backward on compute
                    if _pm is not None:
                        self._pose_floor = (
                            _pm if self._pose_floor is None
                            else min(self._pose_floor, _pm)
                        )
                        self._pose_mse_hist.append(float(_pm))
                        _win = max(2, int(self.cfg.pose_grad_trend_window))
                        if len(self._pose_mse_hist) > _win:
                            del self._pose_mse_hist[:-_win]
                        self._last_pose_epoch = self._global_epoch
                # The categorical-entropy regularizer + Lever-1 rate surrogate do NOT
                # depend on the frames (they read the decoder weights / the full latent
                # tensor), so they backprop straight to the decoder/latents — add them
                # as a separate scalar backward (accumulates into the same .grad buffers
                # the frame-cotangent backward populated).
                reg = self._weight_regularizers(decoder, latents, spec)
                if reg is not None:
                    reg.backward()
                    loss_val += float(reg.item())
            else:
                seg_out, pose_pred6 = self.scorer.seg_pose_forward(decoded_bhwc)

                seg_l = _seg_loss_for_spec(
                    spec, seg_out, self.scorer.seg_targets_hard[idx],
                    temperature=epoch_temperature,
                )
                # Lever C: per-dim weighting on the FUSED (non-split) pose path too (default None → plain MSE,
                # byte-identical). The equimarginal controller (Lever A) is split-only (refused on the fused
                # path by __post_init__), so the fused path keeps ``spec.pose_weight`` unchanged.
                pdw = self.cfg.pose_dim_weights
                if pdw is None:
                    pose_mse = F.mse_loss(pose_pred6, self.scorer.pose_targets[idx])
                else:
                    from tac.torch_vehicle.pose_dim_weights import weighted_pose_mse

                    pose_mse = weighted_pose_mse(
                        pose_pred6, self.scorer.pose_targets[idx], pdw
                    )
                pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)

                loss = spec.seg_weight * seg_l + spec.pose_weight * pose_l
                reg = self._weight_regularizers(decoder, latents, spec)
                if reg is not None:
                    loss = loss + reg
                loss.backward()
                loss_val = float(loss.item())
                pose_mse_val = float(pose_mse.item())

            # Lever 4: accumulate the per-tensor sensitivity EMA from the just-computed
            # score-domain ``w.grad`` (BEFORE the optimizer.step() zeroes nothing — grads
            # persist until the next zero_grad). NO-OP unless score_aware_qat is on (we
            # only pay the norm cost when the lever consumes it).
            if spec.use_qat and spec.score_aware_qat:
                from tac.torch_vehicle.score_aware_qat import accumulate_tensor_sensitivity

                accumulate_tensor_sensitivity(
                    decoder, rt.tensor_sensitivity_ema, decay=spec.qat_sensitivity_decay
                )
            gn_adamw = torch.nn.utils.clip_grad_norm_(
                [*rt.adamw_clip_params, latents], spec.grad_clip
            )
            last_gn_adamw = float(gn_adamw)
            if rt.muon_opt is not None and spec.grad_clip_muon is not None:
                gn_muon = torch.nn.utils.clip_grad_norm_(rt.muon_params, spec.grad_clip_muon)
                last_gn_muon = float(gn_muon)
            rt.adamw_opt.step()
            if rt.muon_opt is not None:
                rt.muon_opt.step()

            # EMA update after each step (the EMA non-negotiable; faithful position).
            # Default: the constant ``spec.ema_decay`` (byte-identical). WARMUP (opt-in,
            # ``cfg.ema_warmup``): the effective decay ramps via the bias-corrected
            # ``min(spec.ema_decay, (t+1)/(t+10))`` so the shadow TRACKS the live weights
            # on a short fine-tune instead of staying frozen near init (the EMA-shadow-lag
            # fix). ``t`` is the global EMA-step counter; it advances ONLY on the warmup
            # path so the default counter stays 0 and unread.
            ema_decay = spec.ema_decay
            if self.cfg.ema_warmup:
                ema_decay = min(spec.ema_decay, (self._ema_step + 1) / (self._ema_step + 10))
                self._ema_step += 1
            self.v.ema_update(
                rt.ema_decoder, decoder, rt.ema_latents, latents, decay=ema_decay
            )

            epoch_loss += loss_val
            epoch_pose += pose_mse_val
            nb += 1

        rt.adamw_sched.step()
        if rt.muon_sched is not None:
            rt.muon_sched.step()
        return epoch_loss / max(nb, 1), epoch_pose / max(nb, 1), last_gn_adamw, last_gn_muon

    def _weight_regularizers(
        self, decoder: nn.Module, latents: torch.Tensor, spec: StageSpec
    ) -> torch.Tensor | None:
        """The weight-domain regularizers added to the loss: the vendored C1a
        categorical entropy (``cat_lambda``) + the Lever-1 differentiable brotli-rate
        surrogate (``rate_lambda_w · H(W_i|W_{i-1}) + rate_lambda_lat · H(Δlatent)``).

        Returns ``None`` when NO regularizer is active (``cat_lambda == 0`` AND both
        ``rate_lambda_*`` == 0) — the DEFAULT for every vendored stage except the C1a
        stages, so the loss is byte-identical to the pre-lever path. Returning ``None``
        (vs a 0.0 tensor) preserves the EXACT legacy control flow: when only C1a is
        active this returns ``spec.cat_lambda * cat_entropy_v2(...)`` — the SAME tensor
        the legacy code added — so a basin that resumes onto this code is unchanged.

        Lever 1 is DEFAULT-OFF: both ``rate_lambda_w`` and ``rate_lambda_lat`` default
        to 0.0, so the rate surrogate is never even computed on the default path (no
        extra cost, no gradient change). When enabled, the rate term is a true brotli
        lower bound (order-1 conditional weight entropy + latent temporal-delta
        entropy) — see ``tac.losses.rate_surrogate``."""
        terms: list[torch.Tensor] = []
        # C1a DOUBLE-COUNT GUARD: when the learned-prior weight-entropy penalty is ACTIVE
        # for this stage AND ``weight_entropy_penalty_supersedes_c1a`` (default True),
        # ZERO the C1a term (the two penalize the same H; stacking is MEASURED net-negative).
        # Byte-identical on the default λ=0 path (penalty inactive → C1a unchanged).
        _penalty_active = (
            float(self.cfg.weight_entropy_penalty_lambda) > 0.0
            and self._weight_entropy_penalty is not None
            and self._cur_stage_index >= int(self.cfg.weight_entropy_penalty_stage_min)
        )
        _c1a_superseded = _penalty_active and bool(
            self.cfg.weight_entropy_penalty_supersedes_c1a
        )
        if spec.cat_lambda > 0 and not _c1a_superseded:
            ent = self.v.cat_entropy_v2(
                decoder, sigma=spec.cat_sigma, sample_size=2000, device=self.train_device
            )
            terms.append(spec.cat_lambda * ent)
        if spec.rate_lambda_w > 0 or spec.rate_lambda_lat > 0:
            from tac.losses.rate_surrogate import (
                RateSurrogateConfig,
                brotli_rate_surrogate,
            )

            # The latent rate term is GLOBAL (the codec delta-codes the full latent
            # sequence), so pass the ENTIRE latent tensor — not the batch slice. The
            # weight term reads the decoder weights (also global).
            #
            # MED-1 FIX (probe ``experiments/probe_lever1_entropy_vs_real_brotli.py``):
            # use ``codec_scan_order=True`` so the conditional entropy is computed over
            # the FULL ``state_dict()`` (weights AND biases, in state-dict order) as ONE
            # concatenated stream — the EXACT density the vendored codec
            # ``encode_decoder(quantize_state_dict(sd))`` brotli-compresses. The probe
            # measured Spearman 0.90 / Pearson 0.999 between this and real brotli decoder
            # bytes, vs Spearman -0.14 for the legacy per-tensor-weights-only mode, so
            # TRAIN-TIME rate now tracks DEPLOY-TIME bytes (the full-stack-synergy
            # requirement). The FiLM ``pose_film.*`` params ship in the decoder blob too,
            # so the full-state_dict walk regularizes them as well.
            rate_cfg = RateSurrogateConfig(codec_scan_order=True)
            lat_arg = latents if spec.rate_lambda_lat > 0 else None
            h_cond, r_lat = brotli_rate_surrogate(
                decoder, lat_arg, rate_cfg, device=self.train_device
            )
            if spec.rate_lambda_w > 0:
                terms.append(spec.rate_lambda_w * h_cond)
            if spec.rate_lambda_lat > 0:
                terms.append(spec.rate_lambda_lat * r_lat)
        # WEIGHT-ENTROPY PENALTY (the Ballé end-to-end rate-distortion lever) —
        # DEFAULT-PRESERVING. ``cfg.weight_entropy_penalty_lambda == 0.0`` (the default)
        # adds NO term (and the penalty module was never built — see _build_stage_runtime),
        # so the loss is byte-identical. When > 0 AND the current stage is at/after
        # ``weight_entropy_penalty_stage_min`` (the C1a-style late-stage schedule), add
        # ``λ · rate_term`` where ``rate_term`` is the expected codelength of the decoder's
        # CURRENT codec-grid weight symbols under the LEARNED per-channel Ballé prior,
        # mapped onto the contest rate scale (25·bits/8/N). The term carries gradient to
        # BOTH the decoder weights (pulling the symbol distribution toward low entropy →
        # lower deployed bytes) AND the prior params (which adapt). The penalty module is
        # built when the lever is enabled; if it is still None here the lever is off.
        lam_we = float(self.cfg.weight_entropy_penalty_lambda)
        if (
            lam_we > 0.0
            and self._weight_entropy_penalty is not None
            and self._cur_stage_index >= int(self.cfg.weight_entropy_penalty_stage_min)
        ):
            # WATERFILL allocation (default OFF → uniform per-tensor weight → byte-identical
            # loss term). When ON, derive ``byte_share_t / (sensitivity_t+eps)`` multipliers
            # from the CURRENT weights + the Lever-4 sensitivity EMA (byte-share-only when the
            # EMA is empty). The reported rate_term stays UN-weighted (comparable across A/Bs);
            # only the WEIGHTED total_bits steers the per-tensor allocation. We minimize the
            # WEIGHTED bits, so use total_bits (mapped to the contest scale) as the loss term
            # under waterfill, and the un-weighted rate_term under uniform.
            if bool(self.cfg.weight_entropy_penalty_waterfill):
                ww = self._weight_entropy_penalty.compute_waterfill_weights(
                    decoder, self._cur_tensor_sensitivity_ema or None
                )
                total_bits, _rate_term = self._weight_entropy_penalty.rate_bits(
                    decoder, per_tensor_weights=ww
                )
                weighted_rate = total_bits / 8.0 / 37_545_489.0 * 25.0
                terms.append(lam_we * weighted_rate)
            else:
                _total_bits, rate_term = self._weight_entropy_penalty.rate_bits(decoder)
                terms.append(lam_we * rate_term)
        if not terms:
            return None
        reg = terms[0]
        for t in terms[1:]:
            reg = reg + t
        return reg

    # -- split-by-head combined-gradient backward (the pose-axis salvage) -----
    def _split_by_head_backward(
        self,
        decoded_bhwc: torch.Tensor,
        idx: torch.Tensor,
        spec: StageSpec,
        *,
        temperature: float | None = None,
        compute_pose: bool = True,
        decoder: nn.Module | None = None,
        latents: torch.Tensor | None = None,
    ) -> tuple[float, float]:
        """Compute the COMBINED frame-gradient (SegNet path on the fast train device,
        PoseNet path on the CPU authority) and inject it into the decoder graph via a
        single ``decoded_bhwc.backward(gradient=combined_cotangent)``.

        The math (why this is descent-equivalent on BOTH terms BY CONSTRUCTION):
        the full loss ``L = w_seg * seg_l(F) + w_pose * pose_l(F)`` is a sum of two
        terms that each depend on the SAME frame tensor ``F = decoded_bhwc``. By the
        chain rule, ``dL/dF = w_seg * d(seg_l)/dF + w_pose * d(pose_l)/dF`` — the two
        per-head frame cotangents SUM at the frame tensor, and ``dL/dtheta`` (decoder)
        follows from ``dL/dF`` by the SAME vjp regardless of how dL/dF was assembled.
        So we:
          1. detach two frame leaves that share F's values (one on train_device for
             SegNet, one on the CPU authority for PoseNet),
          2. backprop each head's weighted loss to its own leaf (``leaf.grad`` = that
             head's frame cotangent),
          3. sum the two cotangents on the train device (move the CPU pose cotangent
             over — a value transfer, the gradient is already computed on the CPU
             authority PoseNet so it carries ZERO MPS pose drift),
          4. ``decoded_bhwc.backward(gradient=combined)`` — flows the exact combined
             gradient into the decoder.

        The SegNet cotangent is the (validated bit-identical on d_seg) MPS gradient;
        the PoseNet cotangent is the CPU AUTHORITY gradient (zero drift). Returns
        ``(loss_value, pose_mse_value)`` for telemetry (recomputed from the detached
        head losses; not authority — telemetry only).

        FiLM-v2 TRUNK DECOUPLING (``cfg.pose_film_trunk_stopgrad``): when ON, the seg +
        pose cotangents are NO LONGER fused into one ``combined`` backward. Instead, the
        pose contribution is removed from every SHARED-decoder param (trunk + latents +
        rgb_0/rgb_1 heads) so ∂(shared)/∂(pose-objective)=0 EXACTLY, leaving the pose
        gradient ONLY on the FiLM pose path (``pose_mlp`` + ``film_resid``):
          a. ``decoded_bhwc.backward(gradient=cot_seg, retain_graph=True)`` — the SEG
             cotangent trains the WHOLE graph (the FiLM pose params get ~0 here: SegNet
             reads only the FiLM-clean f1, so the seg cotangent on f0 is ~0).
          b. SNAPSHOT ``.grad`` of every non-FiLM param + the latents (the seg-only grad).
          c. ``decoded_bhwc.backward(gradient=cot_pose)`` — the POSE cotangent ACCUMULATES
             onto ALL params (trunk/latents/heads AND the FiLM pose params).
          d. RESTORE the snapshot onto every non-FiLM param + the latents — removing the
             pose contribution from the shared graph. The FiLM pose params are NOT
             restored, so they keep the pose gradient (and seg's ~0 contribution).
        Net: trunk+latents+heads are trained by SEG only; the FiLM pose path by POSE only.
        Requires ``decoder`` + ``latents`` (the param handles to snapshot/restore); the
        ``__post_init__`` guard ensures the flag is only on when both are passed (split +
        FiLM v2). On the default (flag OFF) path the fused ``combined`` backward runs
        EXACTLY as before — byte-identical gradient."""
        # --- SegNet path on the TRAIN device (MPS) ---
        frames_seg = decoded_bhwc.detach().requires_grad_(True)  # train_device leaf
        seg_out = self.scorer.seg_forward_train(frames_seg)
        seg_targets = self.scorer.seg_targets_hard[idx]
        seg_l = _seg_loss_for_spec(spec, seg_out, seg_targets, temperature=temperature)
        (spec.seg_weight * seg_l).backward()
        cot_seg = frames_seg.grad  # d(w_seg*seg_l)/dF on train_device

        # --- PoseNet path: CPU AUTHORITY (default) OR the TRAIN device (the full-MPS
        #     unbundle, cfg.pose_grad_on_train_device). The AUTHORITY rule is UNCHANGED
        #     either way — the exact d_pose that picks BEST runs through exact_eval on the
        #     CPU authority. This switch is GRADIENT-only: it chooses which device computes
        #     the (always-summed-at-the-frame) pose COTANGENT. On the default (flag OFF)
        #     path this is byte-identical to before (CPU authority pose grad, zero drift).
        # Pose-throttle: when ``compute_pose`` is False (pose is solved + this is an
        # off-cadence epoch), SKIP the expensive FastViT fwd+bwd entirely and flow
        # ONLY the SegNet cotangent into the decoder. ``compute_pose`` defaults True so
        # every existing caller (k=1, tests) is byte-identical. The pose term re-engages
        # on the cadence epoch or if it drifts above the resume threshold (caller logic).
        cot_pose = None
        if compute_pose:
            on_train = bool(self.cfg.pose_grad_on_train_device)
            pose_dev = self.train_device if on_train else self.device
            frames_pose = (
                decoded_bhwc.detach().to(pose_dev).requires_grad_(True)
            )  # pose-device leaf, same values
            if on_train:
                # MPS (or train-device) pose head — the SAME frozen train net the SegNet
                # head uses; gradient-only, never an authority score. Targets already live
                # on train_device (the scorer holds them there for the per-step loss).
                pose_pred6 = self.scorer.pose_forward_train(frames_pose)
                pose_targets = self.scorer.pose_targets[idx.to(pose_dev)]
            else:
                pose_pred6 = self.scorer.pose_forward_authority(frames_pose)
                pose_targets = self._pose_targets_authority()[idx.to(pose_dev)]
            # Lever C: per-dim Mahalanobis/AIL weighting of the 6 scored dims (renormalised to mean 1.0 so
            # ``cfg.pose_dim_weights is None`` AND uniform-after-norm are byte-identical to the plain MSE).
            pdw = self.cfg.pose_dim_weights
            if pdw is None:
                pose_mse = F.mse_loss(pose_pred6, pose_targets)
            else:
                from tac.torch_vehicle.pose_dim_weights import weighted_pose_mse

                pose_mse = weighted_pose_mse(pose_pred6, pose_targets, pdw)
            pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)
            (spec.pose_weight * pose_l).backward()
            cot_pose = frames_pose.grad  # d(w_pose*pose_l)/dF, on pose_dev
            # Fail-closed: a pose forward that severs the pose Jacobian (e.g. an
            # un-patched, no_grad/in-place yuv6 on the train device) yields an all-zero
            # pose cotangent → the pose objective would silently train nothing. Refuse it
            # so the operator never believes the pose head is active when it is not. Only
            # checked when a positive pose weight is requested (a zero weight legitimately
            # zeroes the cotangent).
            if (
                on_train
                and float(spec.pose_weight) > 0.0
                and not bool(cot_pose.detach().abs().any().item())
            ):
                raise RuntimeError(
                    "pose_grad_on_train_device: the train-device pose cotangent is "
                    "IDENTICALLY ZERO — the pose Jacobian was severed (likely an "
                    "un-patched/no_grad rgb_to_yuv6 on the train device). The pose "
                    "objective would train nothing. Ensure load_frozen_distortion_net "
                    "applied patch_upstream_yuv6_globally + patch_scorer_for_mps for "
                    "the train device, or disable pose_grad_on_train_device."
                )
            cot_pose = cot_pose.to(cot_seg.device)  # value transfer to the train device
            pose_mse_val = float(pose_mse.item())
            self._last_pose_mse = pose_mse_val
            # Lever A: the EQUIMARGINAL pose-weight controller (default OFF → no-op). The cotangents above
            # were computed with ``spec.pose_weight`` folded in; ``cot_pose`` is LINEAR in the pose weight, so
            # rescaling it by ``w_pose_eff / spec.pose_weight`` is EXACTLY the cotangent at the controller's
            # effective weight (no extra backward). The controller measures the per-axis frame-cotangent norms
            # at the base weighting and drives ``‖cot_pose‖/‖cot_seg‖`` toward ``rho`` (the equimarginal point;
            # see ``tac.torch_vehicle.equimarginal_pose_weight``). Telemetry is logged for observability (#305).
            pose_weight_eff = float(spec.pose_weight)
            if self._equimarginal_ctrl is not None and float(spec.pose_weight) > 0.0:
                cot_seg_norm = float(cot_seg.detach().norm().item())
                cot_pose_norm = float(cot_pose.detach().norm().item())
                pose_weight_eff = self._equimarginal_ctrl.update(
                    cot_seg_norm, cot_pose_norm, w_pose_base=float(spec.pose_weight)
                )
                scale = pose_weight_eff / float(spec.pose_weight)
                if scale != 1.0:
                    cot_pose = cot_pose * scale  # exact: cot_pose is linear in the pose weight
                self._last_equimarginal_telemetry = self._equimarginal_ctrl.telemetry(
                    cot_seg_norm=cot_seg_norm, cot_pose_norm=cot_pose_norm, w_pose_eff=pose_weight_eff
                )
            loss_val = float((spec.seg_weight * seg_l).item()) + float(
                (pose_weight_eff * pose_l).item()
            )
        else:
            # SegNet-only cotangent (pose throttled this epoch). loss_val is the seg term
            # only; pose_mse_val carries the last-computed value for telemetry continuity.
            pose_mse_val = (
                float("nan") if self._last_pose_mse is None else self._last_pose_mse
            )
            loss_val = float((spec.seg_weight * seg_l).item())

        # --- inject the frame cotangent(s) into the decoder ---
        # The trunk-stopgrad routing applies ONLY when the FiLM-v2 decoupling flag is on
        # AND there is an actual pose cotangent this epoch (a throttled epoch flows the
        # seg cotangent alone, identically in both modes). When it does NOT apply we keep
        # the EXACT legacy fused backward (byte-identical gradient on the default path).
        trunk_stopgrad = (
            bool(self.cfg.pose_film_trunk_stopgrad)
            and cot_pose is not None
            and decoder is not None
        )
        if not trunk_stopgrad:
            combined = cot_seg if cot_pose is None else cot_seg + cot_pose
            decoded_bhwc.backward(gradient=combined)
            return loss_val, pose_mse_val

        # FiLM-v2 trunk decoupling: route the POSE gradient ONLY to the FiLM pose path.
        # (a) SEG cotangent trains the whole graph (FiLM pose params get ~0 from seg).
        decoded_bhwc.backward(gradient=cot_seg, retain_graph=True)
        # (b) snapshot the seg-only grad of every SHARED (non-FiLM) param + latents.
        shared = self._non_film_grad_params(decoder, latents)
        snapshot = [
            (None if p.grad is None else p.grad.detach().clone()) for p in shared
        ]
        # (c) POSE cotangent accumulates onto ALL params (shared + FiLM pose path).
        decoded_bhwc.backward(gradient=cot_pose)
        # (d) restore the seg-only grad on the SHARED params, removing the pose
        #     contribution there → ∂(shared)/∂(pose-objective)=0 EXACTLY. The FiLM pose
        #     path keeps the accumulated pose grad (seg's contribution to it is ~0).
        for p, snap in zip(shared, snapshot, strict=True):
            if snap is None:
                p.grad = None
            else:
                p.grad.copy_(snap)
        return loss_val, pose_mse_val

    def _pose_targets_authority(self) -> torch.Tensor:
        """The pose targets on the AUTHORITY device (cached). For split-by-head the
        PoseNet path runs on the CPU authority, so its targets must live there too
        (the scorer holds them on train_device for the non-split path)."""
        cached = getattr(self, "_pose_targets_authority_cache", None)
        if cached is None or cached.device != self.device:
            cached = self.scorer.pose_targets.to(self.device)
            self._pose_targets_authority_cache = cached
        return cached

    # -- checkpoint state capture / restore ----------------------------------
    def _capture_state(self, rt: _StageRuntime, spec: StageSpec) -> dict[str, Any]:
        return {
            "decoder": {k: v.detach().cpu().clone() for k, v in rt.decoder.state_dict().items()},
            "latents": rt.latents.detach().cpu().clone(),
            "ema_decoder": {k: v.detach().cpu().clone() for k, v in rt.ema_decoder.state_dict().items()},
            "ema_latents": rt.ema_latents.detach().cpu().clone(),
            "adamw": rt.adamw_opt.state_dict(),
            "muon": rt.muon_opt.state_dict() if rt.muon_opt is not None else None,
            "adamw_sched": rt.adamw_sched.state_dict(),
            "muon_sched": rt.muon_sched.state_dict() if rt.muon_sched is not None else None,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            # Lever 4: persist the per-tensor score-sensitivity EMA so a
            # score-aware-QAT resume continues the SAME quant-grid trajectory
            # (else it resets to empty → uniform-127 fallback for the post-resume
            # steps). A plain dict copy (JSON/torch-safe). Empty/default on the
            # vendored path → round-trips as an empty dict (no behavior change).
            "tensor_sensitivity_ema": dict(rt.tensor_sensitivity_ema),
            # Ballé weight-entropy lever: persist the LEARNED per-channel prior
            # params (loc / raw_scale / raw_shape per coded weight tensor) so a
            # λ>0 RESUME continues the SAME adapted prior instead of rebuilding a
            # fresh one (which would jolt the rate term for the post-resume steps —
            # the prior re-learns the symbol distribution from init_scale). ``None``
            # on the default λ=0 path (the penalty module is never built) →
            # round-trips as None → byte-identical resume for the live basin.
            "weight_entropy_penalty": (
                {k: v.detach().cpu().clone() for k, v in self._weight_entropy_penalty.state_dict().items()}
                if self._weight_entropy_penalty is not None
                else None
            ),
            # EMA-warmup step counter — persisted so a resume CONTINUES the warmup decay
            # schedule (else _ema_step resets to 0 → the decay snaps back to 0.1 and the
            # shadow takes a spurious jolt). Always 0 on the default (ema_warmup off) path,
            # so it round-trips as 0 → byte-identical for the faithful basin.
            "ema_step": int(self._ema_step),
            # APGC controller state — persisted so a resume CONTINUES the same adaptive
            # cadence (else the floor/trend/measurement-epoch reset → pose is recomputed
            # every epoch for the first k_max epochs post-resume, a spurious cost spike,
            # and the floor re-establishes from a possibly-drifted sample). All default on
            # the non-adaptive path (floor None, empty hist, epoch 0) → round-trips as the
            # same defaults → byte-identical for the static/faithful basin.
            "pose_floor": self._pose_floor,
            "pose_mse_hist": list(self._pose_mse_hist),
            "last_pose_epoch": int(self._last_pose_epoch),
            # Lever A: persist the equimarginal controller state (ratio EMA + accumulated w_pose fraction +
            # step count) so a resume CONTINUES the same w_pose trajectory (else it snaps back to w_pose0 and
            # the pose/seg balance jolts). None on the default path (controller off) → round-trips as None →
            # byte-identical for the faithful basin.
            "equimarginal_ctrl": (
                None if self._equimarginal_ctrl is None else self._equimarginal_ctrl.state_dict()
            ),
            "base_channels": self.cfg.base_channels,
            "latent_dim": self.cfg.latent_dim,
            "n_pairs": self.n_pairs,
            # Persist the taper schedule so resume fails closed on a taper change (the
            # codec stays schedule-agnostic; this makes the resume-safety EXPLICIT).
            "taper_channels": self.cfg.taper_channels,
            # Persist the stage-8 Muon own-floor flag so a resume that TOGGLES it fails
            # closed (sister of the taper guard). A checkpoint trained with floor-fix=X
            # carries a Muon LambdaLR step-count tuned to X's eta_min; resuming under !X
            # would silently apply that step-count to a different lambda at stage 8.
            "muon_lr_floor_fix": bool(self.cfg.muon_lr_floor_fix),
            # Persist the E#5 per-stage LR warmup fraction + start-ratio so a resume that
            # CHANGES them fails closed (sister of the floor-fix guard). A mid-stage
            # checkpoint carries a LambdaLR step-count tuned to THIS warmup shape; resuming
            # under a different shape would silently mis-schedule the LR within the stage.
            "stage_lr_warmup_frac": float(self.cfg.stage_lr_warmup_frac),
            "stage_lr_warmup_start_ratio": float(self.cfg.stage_lr_warmup_start_ratio),
            "stage_name": spec.name,
            "ema_decay": spec.ema_decay,
            "best_score": self.best_score,
            "best_ep": self.best_ep,
            "best_stage": self.best_stage,
        }

    def _restore_into(self, rt: _StageRuntime, merged: dict[str, Any]) -> None:
        # The TRAIN runtime (decoder/latents/EMA shadow) lives on the train device
        # (the gradient backend, possibly MPS); the checkpoint is CPU-resident.
        td = self.train_device
        rt.decoder.load_state_dict({k: v.to(td) for k, v in merged["decoder"].items()})
        rt.latents.data = merged["latents"].to(td)
        rt.ema_decoder.load_state_dict(
            {k: v.to(td) for k, v in merged["ema_decoder"].items()}
        )
        rt.ema_latents = merged["ema_latents"].to(td)
        # Restore the EMA-warmup step counter (backward-compatible: old checkpoints lack
        # the key → 0, which is also the value on any ema_warmup-off run → no change).
        self._ema_step = int(merged.get("ema_step", 0))
        # Restore the APGC controller state (backward-compatible: a legacy/pre-APGC
        # checkpoint lacks these keys → the defaults, which are also the values on any
        # non-adaptive run → no change). ``pose_floor`` may be None (no compute yet).
        _pf = merged.get("pose_floor")
        self._pose_floor = None if _pf is None else float(_pf)
        self._pose_mse_hist = [float(x) for x in merged.get("pose_mse_hist", [])]
        self._last_pose_epoch = int(merged.get("last_pose_epoch", -1))
        # Lever A: restore the equimarginal controller trajectory (backward-compatible — a checkpoint without
        # the key, or a run with the controller off, leaves the controller at its fresh defaults → no change).
        _eq = merged.get("equimarginal_ctrl")
        if self._equimarginal_ctrl is not None and _eq is not None:
            self._equimarginal_ctrl.load_state_dict(_eq)
        # Optimizer/scheduler restore is SKIPPED for a FORK-SEED checkpoint (the
        # Lever-3 A/B seeds the WEIGHTS + EMA from the immutable basin forkpoint but
        # cannot transfer the AdamW/Muon momentum across an architecture change —
        # FiLM adds new params, so the param groups differ). A fork-seed checkpoint
        # sets ``adamw=None`` to signal "fresh optimizers from these weights at this
        # curriculum position". Production checkpoints ALWAYS carry optimizer state,
        # so the default (full) resume is unchanged.
        if merged.get("adamw") is not None:
            rt.adamw_opt.load_state_dict(merged["adamw"])
            if rt.muon_opt is not None and merged.get("muon") is not None:
                rt.muon_opt.load_state_dict(merged["muon"])
            rt.adamw_sched.load_state_dict(merged["adamw_sched"])
            if rt.muon_sched is not None and merged.get("muon_sched") is not None:
                rt.muon_sched.load_state_dict(merged["muon_sched"])
        # Lever 4: restore the per-tensor score-sensitivity EMA (so a
        # score-aware-QAT resume continues the SAME quant grid, not a uniform-127
        # reset). Backward-compatible: a legacy/fork-seed checkpoint with no key
        # (or ``None``) leaves the freshly-built empty EMA — exactly today's
        # behavior on the default path (the EMA re-seeds from the first backward).
        sens = merged.get("tensor_sensitivity_ema")
        if sens:
            rt.tensor_sensitivity_ema.clear()
            rt.tensor_sensitivity_ema.update({str(k): float(v) for k, v in sens.items()})

        # Ballé weight-entropy lever: restore the LEARNED per-channel prior params so a
        # λ>0 resume continues the adapted prior (the penalty module is built lazily in
        # _build_stage_runtime BEFORE this restore when λ>0). Backward-compatible: a
        # legacy/λ=0 checkpoint has the key absent or ``None`` and/or the penalty is
        # ``None`` → no restore → byte-identical (the default path never built one). The
        # decoder it sizes against is the SAME architecture (channel counts), so the
        # state_dict shapes match by construction; a mismatch (architecture drift across
        # resume) surfaces as a torch load_state_dict error rather than silent skew.
        we_state = merged.get("weight_entropy_penalty")
        if we_state and self._weight_entropy_penalty is not None:
            self._weight_entropy_penalty.load_state_dict(
                {k: v.to(self.train_device) for k, v in we_state.items()}
            )

        # RNG restore (so the next epoch's randperm matches the uninterrupted run).
        from tac.torch_vehicle.checkpoint import restore_rng

        restore_rng(merged)

    # -- eval + BEST tracking (EMA shadow is the export bytes) ---------------
    def _snapshot_ema(self, rt: _StageRuntime) -> _EvalSnapshot:
        """Capture a POINT-IN-TIME CPU copy of the EMA shadow the eval will score.

        This is the ONLY part of the eval that touches the live training runtime,
        and it runs in the MAIN thread (cheap — a state_dict + latents deep-copy to
        CPU). After it returns, training may keep mutating the live (MPS) EMA shadow
        without racing the eval: the eval operates exclusively on this immutable
        snapshot. The async path snapshots HERE (main thread), then hands the
        snapshot to a background worker; the sync path snapshots + evals inline. The
        eval math (:meth:`_eval_snapshot`) is IDENTICAL for both — so the authority
        numbers are bit-for-bit the same (the no-regression guarantee)."""
        ema_sd = {
            k: v.detach().cpu().clone() for k, v in rt.ema_decoder.state_dict().items()
        }
        ema_latents = rt.ema_latents.detach().cpu().clone()
        # Snapshot Lever-4's online sensitivity EMA ONLY when the unification export is
        # enabled (else None → byte-identical vendored export; no behavior change).
        sens = (
            dict(rt.tensor_sensitivity_ema)
            if self.cfg.lever4_variable_level_export_enabled and rt.tensor_sensitivity_ema
            else None
        )
        return _EvalSnapshot(
            ema_sd=ema_sd, ema_latents=ema_latents, tensor_sensitivity_ema=sens
        )

    def _build_archive_with_optional_variable_waterfill(
        self,
        archive_decoder_sd: dict[str, torch.Tensor],
        meta_dict: dict[str, Any],
        build_base_archive: Callable[[dict[str, Any]], bytes],
    ) -> tuple[bytes, dict[str, torch.Tensor], torch.Tensor]:
        """Build a vendored archive, optionally replacing only the decoder section.

        D2 is intentionally default-off and conservative: when disabled this returns
        exactly ``build_base_archive(meta_dict)`` and the vendored parse-back state. When
        enabled it solves the measured RD table with ``byte_target`` and
        ``net_stop=False`` (the bankable operating point), then swaps the decoder blob
        while leaving meta/latents and any additive trailing section intact.

        """
        if not self.cfg.variable_level_waterfill_enabled:
            archive = build_base_archive(meta_dict)
            eval_decoder_sd, eval_latents, _meta = self.v.parse_archive(archive)
            return archive, eval_decoder_sd, eval_latents

        from tac.losses.variable_level_codec import (
            build_decoder_blob_variable_or_vendored,
            decode_decoder_variable,
        )
        from tac.losses.variable_level_waterfill_allocator import (
            solve_waterfill_allocation,
            verify_kkt_marginal_equalization,
        )

        rd_table = _normalize_variable_level_rd_table(
            self.cfg.variable_level_waterfill_rd_table or {}
        )
        missing = sorted(set(rd_table).difference(archive_decoder_sd))
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise ValueError(
                "variable_level_waterfill_rd_table contains tensors absent from the "
                f"archive decoder state dict: {preview}{suffix}"
            )
        byte_target = float(self.cfg.variable_level_waterfill_byte_target)
        alloc = solve_waterfill_allocation(
            rd_table, byte_target=byte_target, net_stop=False
        )
        kkt_holds, kkt_msg = verify_kkt_marginal_equalization(alloc)
        levels = {k: int(v) for k, v in alloc.levels.items()}
        decoder_blob, is_variable_format = build_decoder_blob_variable_or_vendored(
            archive_decoder_sd, levels
        )
        active_levels = {k: v for k, v in sorted(levels.items()) if v < 127}
        meta_dict["decoder_codec"] = "variable_level_waterfill.v1"
        meta_dict["variable_level_waterfill"] = {
            "schema": "track_a_item_b_d2_variable_level_waterfill.v1",
            "enabled": True,
            "byte_target": byte_target,
            "net_stop": False,
            "falsified_path": "net_stop",
            "source": "Track-A Item B conservative byte-target operating point",
            "rd_table_tensors": len(rd_table),
            "n_coarsened": int(alloc.n_coarsened),
            "total_byte_saving_predicted": float(alloc.total_byte_saving),
            "total_dist_cost_predicted": float(alloc.total_dist_cost),
            "net_score_delta_predicted_from_rd_table": float(alloc.net_score_delta),
            "kkt_marginal_equalization_holds": bool(kkt_holds),
            "kkt_explanation": kkt_msg,
            "levels": active_levels,
            "decoder_blob_is_variable_format": bool(is_variable_format),
            "decoder_blob_bytes": len(decoder_blob),
            "score_claim": False,
            "authority": "[macOS-CPU advisory] NON-PROMOTABLE until dual CPU/CUDA exact eval",
        }
        base_archive = build_base_archive(meta_dict)
        eval_decoder_sd_vendored, eval_latents, _meta = self.v.parse_archive(base_archive)
        meta_brotli, _vendored_decoder_blob, latents_brotli, trailing = (
            _split_three_section_archive(base_archive)
        )
        archive = _join_three_section_archive(
            meta_brotli, decoder_blob, latents_brotli, trailing
        )
        meta_dict["variable_level_waterfill"]["archive_bytes"] = len(archive)
        eval_decoder_sd = (
            decode_decoder_variable(decoder_blob)
            if is_variable_format
            else eval_decoder_sd_vendored
        )
        return archive, eval_decoder_sd, eval_latents

    def _build_archive_with_optional_sensitivity_variable_levels(
        self,
        archive_decoder_sd: dict[str, torch.Tensor],
        meta_dict: dict[str, Any],
        build_base_archive: Callable[[dict[str, Any]], bytes],
        sensitivity: dict[str, float] | None,
    ) -> tuple[bytes, dict[str, torch.Tensor], torch.Tensor]:
        """The Lever-4↔variable-level-export UNIFICATION (R14 contest-optimality finding).

        Build a vendored archive, optionally replacing ONLY the decoder section with a
        VARIABLE per-tensor INT8 grid derived from Lever-4's ONLINE score-sensitivity
        EMA (``sensitivity[name] = ||∂S/∂w_t||``). This is the contest-OPTIMAL byte-half
        of Lever-4: the SAME per-tensor sensitivity it computes online to shape the
        training-time QAT grid now also drives the EXPORT grid (the reverse-waterfill
        allocation — high-sensitivity tensors keep 127 levels, low-sensitivity ones
        coarsen toward ``min_abs_levels``), capturing the full byte saving WITHOUT a
        separate offline RD-table sweep (the gap R14 measured: Lever-4 alone delivered
        only the ~-4.4% brotli-compressibility saving on the uniform export, leaving the
        ~-36% variable-level saving on the table).

        DEFAULT-PRESERVING (the daemon-safety guard): disabled (flag off) OR
        ``sensitivity is None`` / uniform → returns EXACTLY ``build_base_archive(meta_
        dict)`` + the vendored parse-back state (byte-identical to today). The level map
        is built by ``levels_from_sensitivity_for_codec`` — the SAME rank-norm band the
        score-aware QAT (Lever 4) trained the decoder to be robust at, so the export grid
        matches the trained grid (mathematically/algebraically/geometrically consistent
        per the reverse-waterfill KKT — Cover & Thomas Ch.10). When non-uniform, the
        variable decoder blob is spliced into the 3-section archive (meta/latents/
        trailing preserved) and ``decode_decoder_variable`` reads it back.

        Score-claim discipline: the BYTE saving is real + measurable; the NET-score win
        is advisory until a 600-pair byte-closed dual CPU/CUDA exact eval. NO score is
        claimed from enabling the flag alone (sister of the D2 waterfill discipline).
        """
        if not self.cfg.lever4_variable_level_export_enabled:
            archive = build_base_archive(meta_dict)
            eval_decoder_sd, eval_latents, _meta = self.v.parse_archive(archive)
            return archive, eval_decoder_sd, eval_latents

        from tac.losses.variable_level_codec import (
            build_decoder_blob_variable_or_vendored,
            decode_decoder_variable,
            levels_from_sensitivity_for_codec,
        )

        # Build the per-tensor level map from the online sensitivity EMA. THE SPINE
        # RECONCILIATION: the EMA is keyed by ``decoder.named_modules()`` MODULE names
        # (``blocks.0`` no-FiLM / ``decoder.blocks.0`` FiLM-wrapper) — the SAME keys
        # ``apply_score_aware_qat`` consumes for the training-time bits — but the codec
        # checks the WEIGHT state-dict keys (``blocks.0.weight``). ``_sensitivity_for_
        # codec_weight_keys`` rebinds the module-keyed EMA onto the codec weight keys so
        # the SAME ``||∂S/∂w_t||`` tensor drives BOTH the QAT grid AND the export grid
        # (the single-source spine). Without it the codec lookup would miss every EMA
        # entry → uniform → the rate-attack would silently no-op (the QAT trained a
        # coarse grid the export never reproduced). Tensors absent from the EMA (biases,
        # FiLM params, etc.) get the base 127 levels.
        weight_keys = [k for k in archive_decoder_sd if k.endswith(".weight")]
        codec_sensitivity = _sensitivity_for_codec_weight_keys(sensitivity, weight_keys)
        levels_by_weight = levels_from_sensitivity_for_codec(codec_sensitivity, weight_keys)
        # The codec maps over ALL sd keys; non-weight keys default to base (127).
        levels = {k: int(levels_by_weight.get(k, 127)) for k in archive_decoder_sd}
        decoder_blob, is_variable_format = build_decoder_blob_variable_or_vendored(
            archive_decoder_sd, levels
        )
        if not is_variable_format:
            # Uniform/near-uniform sensitivity → vendored byte-identical (no coarsening).
            # meta_dict is NOT mutated → the base archive is bit-identical to vendored.
            archive = build_base_archive(meta_dict)
            eval_decoder_sd, eval_latents, _meta = self.v.parse_archive(archive)
            return archive, eval_decoder_sd, eval_latents
        # ADVERSARIAL-REVIEW FIX (R14, ordering): the ``decoder_codec`` flag + metadata
        # MUST be written into ``meta_dict`` BEFORE ``build_base_archive`` so the meta
        # section embedded in ``meta_brotli`` carries the flag — otherwise the inflate
        # side cannot know to use ``decode_decoder_variable`` (mirrors the D2 method's
        # order; the prior order built the base archive with UNmutated meta = the flag
        # never reached the emitted bytes).
        active_levels = {k: v for k, v in sorted(levels.items()) if v < 127}
        meta_dict["decoder_codec"] = "lever4_sensitivity_variable_level.v1"
        meta_dict["lever4_variable_level_export"] = {
            "schema": "lever4_online_sensitivity_variable_level_export.v1",
            "enabled": True,
            "source": "Lever-4 online score-sensitivity EMA (||dS/dw||) -> reverse-waterfill",
            "n_coarsened": len(active_levels),
            "levels": active_levels,
            "decoder_blob_is_variable_format": True,
            "decoder_blob_bytes": len(decoder_blob),
            "score_claim": False,
            "authority": "[contest-CPU advisory] NON-PROMOTABLE until dual CPU/CUDA exact eval",
        }
        base_archive = build_base_archive(meta_dict)
        eval_decoder_sd_vendored, eval_latents, _meta = self.v.parse_archive(base_archive)
        meta_brotli, _vendored_decoder_blob, latents_brotli, trailing = (
            _split_three_section_archive(base_archive)
        )
        archive = _join_three_section_archive(
            meta_brotli, decoder_blob, latents_brotli, trailing
        )
        meta_dict["lever4_variable_level_export"]["archive_bytes"] = len(archive)
        eval_decoder_sd = decode_decoder_variable(decoder_blob)
        return archive, eval_decoder_sd, eval_latents

    def _build_archive_and_eval_decoder(
        self,
        ema_sd: dict[str, torch.Tensor],
        ema_latents: torch.Tensor,
        meta_dict: dict[str, Any],
        sensitivity: dict[str, float] | None = None,
    ) -> tuple[bytes, nn.Module, torch.Tensor]:
        """Build the byte-closed archive from the EMA shadow + return the PARSE-BACK
        eval decoder + parsed latents (on the AUTHORITY device).

        DEFAULT (no FiLM): the vendored 3-section archive + a vendored decoder
        rebuilt from the int8-dequantized parse-back — UNCHANGED from the legacy
        path (byte-identical).

        Lever 3 (``pose_film_enabled``): the EMA shadow is the WRAPPER state dict
        (``decoder.*`` + ``pose_film.*`` + ``stored_pose``). We (1) split the FiLM
        weights into the codec-compatible decoder blob (bare vendored keys + the
        ``pose_film.*`` keys) and the ``stored_pose`` buffer into the ADDITIVE pose
        section, (2) build ``vendored_archive + encode_pose_section`` (pristine
        codec untouched), (3) parse BOTH back and rebuild the FiLM wrapper + a
        cursor-based eval adapter so the exact eval renders the SAME FiLM-conditioned
        frames the inflate path produces (byte-closed faithful)."""
        if not self.cfg.pose_film_enabled:
            if self.cfg.lever4_variable_level_export_enabled:
                archive, eval_decoder_sd, eval_latents = (
                    self._build_archive_with_optional_sensitivity_variable_levels(
                        ema_sd,
                        meta_dict,
                        lambda md: self.v.build_archive(ema_sd, ema_latents, meta_dict=md),
                        sensitivity,
                    )
                )
            else:
                archive, eval_decoder_sd, eval_latents = (
                    self._build_archive_with_optional_variable_waterfill(
                        ema_sd,
                        meta_dict,
                        lambda md: self.v.build_archive(
                            ema_sd, ema_latents, meta_dict=md
                        ),
                    )
                )
            eval_dec = self._new_vendored_decoder(device=self.device)
            eval_dec.load_state_dict(
                {k: v.to(self.device) for k, v in eval_decoder_sd.items()}
            )
            eval_dec.eval()
            return archive, eval_dec, eval_latents

        # Version-aware FiLM export: the additive pose-section grammar
        # (``build_archive_with_pose`` / ``parse_pose_section`` / ``wrapper_sd_to_archive_
        # decoder_sd``) + the cursor eval adapter (``_FiLMEvalDecoder``) are SHARED (v2
        # re-exports them verbatim); only the wrapper class + the FiLM-key prefixes differ.
        if int(self.cfg.pose_film_version) == 2:
            from tac.torch_vehicle.pose_film_v2 import (
                PoseFiLMHNeRVWrapperV2 as _PoseFiLMWrapper,
            )
            from tac.torch_vehicle.pose_film_v2 import (
                _FiLMEvalDecoder,
                build_archive_with_pose,
                parse_pose_section,
                wrapper_sd_to_archive_decoder_sd,
            )
        else:
            from tac.torch_vehicle.pose_film import (
                PoseFiLMHNeRVWrapper as _PoseFiLMWrapper,
            )
            from tac.torch_vehicle.pose_film import (
                _FiLMEvalDecoder,
                build_archive_with_pose,
                parse_pose_section,
                wrapper_sd_to_archive_decoder_sd,
            )
        _film_prefixes = _FILM_PARAM_PREFIXES[int(self.cfg.pose_film_version)]

        # (1) Split the wrapper state dict: codec decoder blob (vendored + FiLM
        # weights) and the stored-pose buffer (additive pose section).
        archive_decoder_sd = wrapper_sd_to_archive_decoder_sd(ema_sd)
        stored_pose = ema_sd["stored_pose"]
        # (2) Build the additive archive (pristine vendored build_archive + pose),
        # with optional D2 (RD-table) OR Lever-4 (online-EMA) variable-level
        # replacement of ONLY the decoder section (the additive pose section + meta +
        # latents are preserved either way; the two level sources are mutually
        # exclusive per __post_init__).
        _base_archive_builder = lambda md: build_archive_with_pose(  # noqa: E731
            self.v.build_archive,
            archive_decoder_sd,
            ema_latents,
            md,
            stored_pose,
            pose_codec=self.cfg.pose_section_codec,
            lowrank_rank=self.cfg.pose_section_lowrank_rank,
            lowrank_levels=self.cfg.pose_section_lowrank_levels,
        )
        if self.cfg.lever4_variable_level_export_enabled:
            archive, eval_decoder_sd, eval_latents = (
                self._build_archive_with_optional_sensitivity_variable_levels(
                    archive_decoder_sd, meta_dict, _base_archive_builder, sensitivity
                )
            )
        else:
            archive, eval_decoder_sd, eval_latents = (
                self._build_archive_with_optional_variable_waterfill(
                    archive_decoder_sd, meta_dict, _base_archive_builder
                )
            )
        # (3) Parse BOTH sections back and rebuild the FiLM wrapper for eval. The FiLM
        # keys keep their wrapper-level prefixes (``pose_film.*`` v1 / ``pose_mlp.*`` +
        # ``film_resid.*`` v2); the rest are the bare vendored decoder keys.
        parsed_pose = parse_pose_section(archive, self.v.parse_archive)
        film_sd = {
            k: v
            for k, v in eval_decoder_sd.items()
            if any(k.startswith(p) for p in _film_prefixes)
        }
        dec_sd = {
            k: v
            for k, v in eval_decoder_sd.items()
            if not any(k.startswith(p) for p in _film_prefixes)
        }
        vendored = self._new_vendored_decoder(device=self.device)
        vendored.load_state_dict({k: v.to(self.device) for k, v in dec_sd.items()})
        wrapper = _PoseFiLMWrapper(
            vendored,
            n_pairs=self.n_pairs,
            film_hidden=self.cfg.pose_film_hidden,
        ).to(self.device)
        # Load the FiLM submodule weights over the freshly-built (identity-init) FiLM,
        # keeping the vendored ``decoder.*`` keys already loaded. strict=False because the
        # merged dict is the FULL wrapper state (no missing/unexpected keys in practice).
        wrapper.load_state_dict(
            {
                **wrapper.state_dict(),
                **{k: v.to(self.device) for k, v in film_sd.items()},
            },
            strict=False,
        )
        if parsed_pose is not None:
            wrapper.set_stored_pose(parsed_pose.to(self.device))
        wrapper.eval()
        eval_dec = _FiLMEvalDecoder(wrapper)
        eval_dec.eval()  # resets the cursor to pair 0
        return archive, eval_dec, eval_latents

    def _eval_snapshot(
        self, snap: _EvalSnapshot, spec: StageSpec, stage_index: int, snapshot_epoch: int
    ) -> dict[str, Any]:
        """Build the archive from the SNAPSHOT EMA shadow, exact-eval the PARSE-BACK
        (int8-dequantized) decoder + latents, track BEST, and RECORD the telemetry
        row — all keyed to ``snapshot_epoch`` (the epoch the snapshot was taken at).

        FIDELITY (1:1 with vendored ``common.py:228-238``): the score that picks
        the BEST checkpoint is the score of the **archive the contest sees**, i.e.
        the int8-quantized decoder + delta-coded latents AFTER ``parse_archive``,
        NOT the raw float EMA shadow (which over-estimates quality and would pick
        the wrong checkpoint). We reconstruct the eval decoder from the parse-back
        ``state_dict`` and eval THAT.

        THREAD-SAFETY: this method may run in a BACKGROUND thread (async path) or
        inline (sync path). The pure-eval part (build_archive / parse_archive /
        exact_eval) touches NO shared state — it reads only the immutable snapshot.
        The shared surfaces (best-tracker fields, the ``best/`` dir write, and the
        single non-thread-safe :class:`TelemetryWriter`) are mutated ONLY under
        ``self._eval_lock`` so they never race the main loop's non-eval records.

        Returns the eval dict (d_seg / d_pose / rate / score / archive_bytes /
        is_best). The BEST EMA shadow + archive are written to ``out_dir/best/``.
        """
        ema_sd = snap.ema_sd
        meta_dict = {
            "n_pairs": self.n_pairs,
            "latent_dim": self.cfg.latent_dim,
            "base_channels": self.cfg.base_channels,  # FINDING: base_ch in meta (was 36)
            "eval_size": [_EVAL_H, _EVAL_W],
        }
        archive, eval_dec, eval_latents = self._build_archive_and_eval_decoder(
            ema_sd, snap.ema_latents, meta_dict,
            sensitivity=snap.tensor_sensitivity_ema,
        )
        archive_bytes = len(archive)
        # Parse-back: the int8-dequantized decoder + delta-decoded latents — the
        # contest-visible artifact (faithful to common.py). The exact eval runs on
        # the AUTHORITY device (CPU-TRUSTED / CUDA), NEVER the train device — even
        # when training on MPS the score that picks BEST is the CPU authority
        # (CLAUDE.md "MPS auth eval is NOISE"). So the eval decoder is built on
        # self.device and the scorer's exact_eval routes through its authority net.
        ev = self.scorer.exact_eval(eval_dec, eval_latents.to(self.device), archive_bytes)
        score = float(ev["score"])
        d_seg = ev.get("seg_distortion", ev.get("d_seg"))
        d_pose = ev.get("pose_distortion", ev.get("d_pose"))
        rate = ev.get("rate")
        # --- shared-state mutation under the lock (BEST tracker + telemetry) ------
        with self._eval_lock:
            is_best = score < self.best_score
            if is_best:
                self.best_score = score
                self.best_ep = snapshot_epoch
                self.best_stage = stage_index
                best_dir = self.cfg.out_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                (best_dir / "best_archive.bin").write_bytes(archive)
                torch.save(ema_sd, best_dir / "best_ema_decoder.pt")
                torch.save(snap.ema_latents, best_dir / "best_ema_latents.pt")
                variable_level_meta = meta_dict.get("variable_level_waterfill")
                (best_dir / "best_meta.json").write_text(
                    _json_dumps(
                        {
                            "stage_index": stage_index,
                            "stage_name": spec.name,
                            "global_epoch": snapshot_epoch,
                            "score": score,
                            "d_seg": d_seg,
                            "d_pose": d_pose,
                            "rate": rate,
                            "archive_bytes": archive_bytes,
                            "base_channels": self.cfg.base_channels,
                            "decoder_codec": meta_dict.get("decoder_codec", "vendored"),
                            "variable_level_waterfill": variable_level_meta,
                            "authority": "[contest-CPU advisory] NON-PROMOTABLE",
                        }
                    )
                )
        out = {
            "d_seg": d_seg,
            "d_pose": d_pose,
            "rate": rate,
            "score": score,
            "archive_bytes": archive_bytes,
            "is_best": is_best,
        }
        if "variable_level_waterfill" in meta_dict:
            out["decoder_codec"] = meta_dict.get("decoder_codec", "vendored")
            out["variable_level_waterfill"] = meta_dict["variable_level_waterfill"]
        return out

    def _record_eval_row(
        self,
        ev: dict[str, Any],
        spec: StageSpec,
        stage_index: int,
        snapshot_epoch: int,
    ) -> None:
        """Append the EVAL telemetry row, tagged with ``snapshot_epoch`` (the epoch
        the eval's snapshot came from — which may be a few epochs behind the current
        training epoch on the async path). Written under ``_eval_lock`` because the
        single :class:`TelemetryWriter` is not thread-safe and the main loop also
        records non-eval rows."""
        with self._eval_lock:
            self.telemetry.record(
                EpochRecord(
                    stage_index=stage_index,
                    stage_name=spec.name,
                    epoch_in_stage=snapshot_epoch
                    - sum(self.curriculum[i].epochs for i in range(stage_index)),
                    global_epoch=snapshot_epoch,
                    loss=float("nan"),
                    pose_mse=float("nan"),
                    adamw_lr=float("nan"),
                    muon_lr=None,
                    evaluated=True,
                    d_seg=ev.get("d_seg"),
                    d_pose=ev.get("d_pose"),
                    rate=ev.get("rate"),
                    score=ev.get("score"),
                    archive_bytes=ev.get("archive_bytes"),
                    is_best=ev.get("is_best", False),
                    extra={"async_eval_row": True, "snapshot_epoch": snapshot_epoch},
                )
            )

    # -- async eval scheduling (the throughput salvage) ----------------------
    def _async_eval_in_flight(self) -> bool:
        """True iff a background eval worker is currently running."""
        return self._eval_thread is not None and self._eval_thread.is_alive()

    def _schedule_async_eval(
        self, rt: _StageRuntime, spec: StageSpec, stage_index: int, snapshot_epoch: int
    ) -> bool:
        """Snapshot the EMA shadow NOW (main thread) and spawn ONE background worker
        to run the IDENTICAL exact eval off that snapshot, then return immediately so
        training continues. At most ONE eval is in-flight: if a prior worker is still
        running, SKIP (log + count) — the cadence self-throttles (a 13-min eval over-
        runs the 150s eval interval early; once training-only epochs are cheap the
        eval naturally lands within an interval). Returns True if scheduled, False if
        skipped."""
        if self._async_eval_in_flight():
            self._skipped_evals += 1
            print(
                f"[async-eval] SKIP @ global_ep={snapshot_epoch}: prior eval "
                f"(snapshot_ep={self._inflight_snapshot_epoch}) still running "
                f"(total skipped={self._skipped_evals})",
                flush=True,
            )
            return False
        snap = self._snapshot_ema(rt)  # cheap, main-thread, point-in-time
        self._inflight_snapshot_epoch = snapshot_epoch

        def _worker() -> None:
            t0 = time.time()
            try:
                ev = self._eval_snapshot(snap, spec, stage_index, snapshot_epoch)
                self._record_eval_row(ev, spec, stage_index, snapshot_epoch)
                print(
                    f"[async-eval] DONE snapshot_ep={snapshot_epoch} "
                    f"score={ev['score']:.5f} d_seg={ev['d_seg']} d_pose={ev['d_pose']} "
                    f"({time.time() - t0:.0f}s){' *BEST*' if ev['is_best'] else ''}",
                    flush=True,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                # An eval failure must NOT take down the training loop (it runs in a
                # daemon thread off the main loop). Log loudly; do NOT write a
                # telemetry row (a failed eval has no authority numbers — a row would
                # be misleading). Training continues; the next eval epoch re-schedules
                # off a fresh snapshot.
                print(
                    f"[async-eval] FAILED snapshot_ep={snapshot_epoch}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        self._eval_thread = threading.Thread(
            target=_worker, name=f"async-eval-ep{snapshot_epoch}", daemon=True
        )
        self._eval_thread.start()
        return True

    def _join_async_eval(self, timeout: float | None = None) -> None:
        """JOIN any in-flight eval worker so the final BEST + last eval row land
        before the run exits (the DONE-marker contract). Called at run completion."""
        if self._eval_thread is not None and self._eval_thread.is_alive():
            print("[async-eval] JOIN: waiting for in-flight eval to finish...", flush=True)
            self._eval_thread.join(timeout=timeout)
        self._eval_thread = None

    # -- the resumable run ----------------------------------------------------
    def run(self) -> dict[str, Any]:
        """Run (or resume) the full curriculum. Idempotent on a DONE marker.

        Resume contract: if ``out_dir`` holds a checkpoint, the run restores the
        complete state and continues from ``(stage_index, epoch_in_stage)``; a
        death costs at most ``checkpoint_every_epochs`` epochs. On completion a
        DONE marker is written (marker-on-exit).
        """
        from tac.torch_vehicle.checkpoint import is_done

        if is_done(self.cfg.out_dir):
            return {"status": "already_done", "best_score": self.best_score}

        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        # Resume position (default: start of stage 0).
        resume_pos = TorchCheckpointPosition(0, 0)
        merged: dict[str, Any] | None = None
        if checkpoint_exists(self.cfg.out_dir):
            man = read_manifest(self.cfg.out_dir)
            if int(man["n_pairs"]) != self.n_pairs:
                raise ValueError(
                    f"checkpoint n_pairs={man['n_pairs']} != trainer n_pairs={self.n_pairs}; "
                    "cannot resume a different basis"
                )
            if int(man["base_channels"]) != self.cfg.base_channels:
                raise ValueError(
                    f"checkpoint base_channels={man['base_channels']} != "
                    f"cfg.base_channels={self.cfg.base_channels}; cannot resume a different basis"
                )
            # latent_dim resume guard (sister of base_channels/n_pairs): a latent_dim change
            # alters the stem Linear(latent_dim, ...) shape → fail closed HERE with a clear
            # message rather than a cryptic load_state_dict size-mismatch in _restore_into.
            # Backward-compatible: pre-key checkpoints default to cfg.latent_dim (pass).
            if int(man.get("latent_dim", self.cfg.latent_dim)) != self.cfg.latent_dim:
                raise ValueError(
                    f"checkpoint latent_dim={man.get('latent_dim')} != "
                    f"cfg.latent_dim={self.cfg.latent_dim}; cannot resume a different basis"
                )
            # EXPLICIT taper-resume guard (do not rely on an accidental state_dict shape
            # mismatch): a checkpoint trained with a different taper schedule is a different
            # architecture. Normalize None (vendored) so old checkpoints (no key) match a
            # vendored-cfg resume. A list-vs-list / list-vs-None mismatch fails closed here.
            _man_taper = man.get("taper_channels")
            _man_taper = list(_man_taper) if _man_taper is not None else None
            _cfg_taper = list(self.cfg.taper_channels) if self.cfg.taper_channels is not None else None
            if _man_taper != _cfg_taper:
                raise ValueError(
                    f"checkpoint taper_channels={_man_taper} != cfg.taper_channels={_cfg_taper}; "
                    "cannot resume a different taper (different architecture)"
                )
            # muon_lr_floor_fix resume guard (sister of the taper guard; Lens-D 2026-06-19
            # finding). The flag only changes the STAGE-8 Muon LambdaLR floor. Toggling it is
            # SAFE before stage 8 (Muon scheduler does not exist yet — built fresh at stage 8),
            # so a stages-1-7 checkpoint (has_muon=False) may resume under either flag value.
            # It is UNSAFE only once a Muon scheduler step-count is checkpointed (has_muon=True =
            # at/into stage 8): resuming under a toggled flag would apply that step-count to a
            # different lambda → fail closed EXPLICITLY. Backward-compatible: old checkpoints
            # lack the key → read as False; pre-stage-8 they pass regardless of cfg.
            if (
                bool(man.get("has_muon", False))
                and bool(man.get("muon_lr_floor_fix", False)) != bool(self.cfg.muon_lr_floor_fix)
            ):
                raise ValueError(
                    f"checkpoint muon_lr_floor_fix={man.get('muon_lr_floor_fix', False)} != "
                    f"cfg.muon_lr_floor_fix={self.cfg.muon_lr_floor_fix} at a Muon-stage "
                    "checkpoint (has_muon=True); cannot resume with a toggled stage-8 Muon "
                    "floor (would mis-schedule the Muon LR)"
                )
            # E#5 stage-LR-warmup resume guard (sister of the floor-fix guard). The warmup
            # SHAPES every stage's LambdaLR. Changing warmup_frac/start_ratio is SAFE only at
            # a clean stage boundary (epoch_in_stage == 0 → the scheduler is rebuilt fresh for
            # the next stage). It is UNSAFE mid-stage (epoch_in_stage > 0 → a LambdaLR
            # step-count is checkpointed tuned to the OLD warmup shape; resuming under a NEW
            # shape would apply that step-count to a differently-shaped lambda → mis-schedule
            # the LR within the stage). Backward-compatible: old checkpoints lack the keys →
            # read as the cfg value (pass). float compare with a tight tolerance.
            _man_wf = float(man.get("stage_lr_warmup_frac", self.cfg.stage_lr_warmup_frac))
            _man_sr = float(
                man.get("stage_lr_warmup_start_ratio", self.cfg.stage_lr_warmup_start_ratio)
            )
            _mid_stage = int(man.get("epoch_in_stage", 0)) > 0
            if _mid_stage and (
                abs(_man_wf - float(self.cfg.stage_lr_warmup_frac)) > 1e-12
                or abs(_man_sr - float(self.cfg.stage_lr_warmup_start_ratio)) > 1e-12
            ):
                raise ValueError(
                    f"checkpoint stage_lr_warmup_frac={_man_wf}/start_ratio={_man_sr} != "
                    f"cfg {self.cfg.stage_lr_warmup_frac}/{self.cfg.stage_lr_warmup_start_ratio} "
                    f"at a MID-STAGE checkpoint (epoch_in_stage={man.get('epoch_in_stage')}); "
                    "cannot resume with a changed warmup shape (would mis-schedule the stage LR). "
                    "Change the warmup only at a clean stage boundary (epoch_in_stage=0)."
                )
            merged = load_checkpoint(self.cfg.out_dir, map_location=self.cfg.device)
            resume_pos = merged["position"]
            self.best_score = float(man["best_score"])
            self.best_ep = int(man["best_ep"])
            self.best_stage = int(man["best_stage"])

        # Carry decoder/latents/EMA across stages.
        carry_decoder: nn.Module | None = None
        carry_latents: nn.Parameter | None = None
        carry_ema_decoder: nn.Module | None = None
        carry_ema_latents: torch.Tensor | None = None
        # Lever-4 (score-aware QAT) per-tensor sensitivity EMA carry across stages.
        # The PR95 schedule has FIVE consecutive QAT stages (3-7); with score-aware
        # QAT on, the sensitivity ``s_t = ||dS/dw_t||`` is a property of the CARRIED
        # decoder, so it belongs to the "weights/EMA carry" side of the boundary
        # (line 460), NOT the "optimizer resets per stage" side. Without this carry,
        # each QAT->QAT boundary resets the EMA to empty -> the new stage's QAT falls
        # back to uniform-127 for its first hundreds of steps (the SAME defect R2
        # fixed for resume, manifesting at the normal stage boundary). DEFAULT-SAFE:
        # on any non-score-aware-QAT path the prior stage's EMA is always empty, so
        # the carry is empty and the new stage is byte-identical to today.
        carry_sensitivity_ema: dict[str, float] = {}

        # Recompute the global-epoch base for stages already completed.
        self._global_epoch = sum(
            self.curriculum[i].epochs for i in range(resume_pos.stage_index)
        ) + resume_pos.epoch_in_stage

        for stage_index in range(resume_pos.stage_index, len(self.curriculum)):
            spec = self.curriculum[stage_index]
            start_epoch = resume_pos.epoch_in_stage if stage_index == resume_pos.stage_index else 0
            # The weight-entropy penalty (Ballé lever) honors a late-stage schedule
            # (``weight_entropy_penalty_stage_min``) like C1a; ``_weight_regularizers``
            # reads this to gate the term per stage. No-op on the default path (λ=0).
            self._cur_stage_index = stage_index

            # Build decoder/latents for this stage (carry from prior stage, or init).
            resuming_into_this_stage = (
                merged is not None and stage_index == resume_pos.stage_index
            )
            # KD-warm-start applies ONLY at the stage-0 init of a FRESH run (no resume,
            # no carry). When it applies, the KD warm-up phase distills the basin teacher
            # into the (just-built, still-random) re-tapered student AFTER the runtime is
            # built — recorded here as ``do_kd_warm_up`` so the post-runtime block fires it.
            do_kd_warm_up = False
            if carry_decoder is None:
                decoder = self._new_decoder()  # train-device (gradient backend)
                if (
                    self.cfg.kd_warm_start_dir is not None
                    and spec.init_latents_random
                    and not resuming_into_this_stage
                ):
                    # KD-WARM-START: load the basin's latents DIRECTLY as the stage-0 init
                    # (taper-INDEPENDENT (n_pairs, latent_dim) — only the decoder channels
                    # change), and FLAG the KD warm-up phase (it runs after the runtime is
                    # built, distilling the frozen basin teacher into this re-tapered
                    # student). The decoder stays at its FRESH random init here; the KD
                    # warm-up is what loads the basin's knowledge into it. Resume/carry win
                    # (the `carry_decoder is None and not resuming_into_this_stage` guards):
                    # a run that owns a checkpoint continues its own trajectory.
                    from tac.torch_vehicle.kd_warm_start import load_kd_warm_start_latents

                    latents = nn.Parameter(
                        load_kd_warm_start_latents(
                            self.cfg.kd_warm_start_dir,
                            n_pairs=self.n_pairs,
                            latent_dim=self.cfg.latent_dim,
                        ).to(self.train_device)
                    )
                    do_kd_warm_up = True
                elif (
                    self.cfg.warm_start_dir is not None
                    and spec.init_latents_random
                    and not resuming_into_this_stage
                ):
                    # WARM-START fine-tune: load the converged decoder weights + stored
                    # latents from a prior run's best/ dir, REPLACING the random from-0
                    # init. The from-0 random draw is SKIPPED (its RNG is not consumed) —
                    # this is its own init contract: two arms with the SAME warm_start_dir
                    # + seed are bit-identical here and diverge only in their curriculum.
                    # Resume wins (the `not resuming_into_this_stage` guard): a run that
                    # already owns a checkpoint continues its own trajectory.
                    latents = self._load_warm_start_into(decoder)
                elif spec.init_latents_random:
                    # Draw on CPU (deterministic via the global seed) then move to the
                    # train device — MPS has its own RNG stream, so a CPU draw keeps
                    # the init reproducible vs a CPU-arm descent-equivalence A/B.
                    latents = nn.Parameter(
                        (torch.randn(self.n_pairs, self.cfg.latent_dim) * 0.1).to(
                            self.train_device
                        )
                    )
                elif resuming_into_this_stage:
                    # The latents will be overwritten by ``_restore_into`` below; a
                    # placeholder of the right shape is sufficient (faithful: the
                    # vendored loop loads ``final_latents.pt`` here — we load from
                    # the checkpoint).
                    latents = nn.Parameter(
                        torch.zeros(self.n_pairs, self.cfg.latent_dim, device=self.train_device)
                    )
                else:
                    # Fail closed (faithful to vendored common.py:122): a non-stage-1
                    # stage with neither an in-memory carry NOR a resume checkpoint
                    # has no latents to start from — this is a misconfigured run, not
                    # a silent zeros-init.
                    raise ValueError(
                        f"stage {spec.name} (index {stage_index}) has init_latents_random="
                        "False but no prior-stage carry and no resume checkpoint to load "
                        "latents from — cannot start a non-stage-1 stage from scratch."
                    )
            else:
                decoder = carry_decoder
                latents = carry_latents  # type: ignore[assignment]

            rt = self._build_stage_runtime(
                spec,
                decoder=decoder,
                latents=latents,
                ema_decoder=carry_ema_decoder,
                ema_latents=carry_ema_latents,
            )
            # Seed the Lever-4 sensitivity EMA from the PRIOR stage (carried, like the
            # weight EMA). Empty on the default path -> no-op (byte-identical). A
            # resume INTO this stage overwrites it via ``_restore_into`` below (which
            # clears+updates when its checkpointed EMA is non-empty), so the carry
            # never clobbers a more-specific resume-restored EMA.
            if carry_sensitivity_ema:
                rt.tensor_sensitivity_ema.update(carry_sensitivity_ema)

            # If resuming INTO this stage, restore the mid-stage state now.
            if merged is not None and stage_index == resume_pos.stage_index:
                self._restore_into(rt, merged)
                merged = None  # consumed

            # KD WARM-UP phase (the bind-all wall-clock resolution): a non-empty PREFIX of
            # stage 0 that distills the FROZEN basin teacher into the re-tapered student on
            # the (directly-warm-started) latents, BEFORE the normal score-aware curriculum
            # continues. Fires only when ``do_kd_warm_up`` (stage-0 fresh-run KD init). The
            # student's EMA shadow is RE-SYNCED to the distilled state afterward so the
            # shadow tracks the distilled student (not the random init it deep-copied at
            # ``_build_stage_runtime``). No-op (and byte-identical) when kd_warm_start_dir
            # is None.
            if do_kd_warm_up:
                self._run_kd_warm_up(rt, spec)

            for epoch in range(start_epoch, spec.epochs):
                # ``epoch`` is the 0-based epoch within this stage — drives the Lever-2
                # seg-temperature anneal (NO-OP unless seg_temperature_end is set).
                mean_loss, mean_pose_mse, gn_adamw, gn_muon = self._train_one_epoch(
                    rt, spec, epoch_in_stage=epoch
                )
                self._global_epoch += 1
                epoch_in_stage = epoch + 1  # 1-based completed

                # Eval cadence (faithful: eval_every).
                evaluated = (epoch_in_stage % spec.eval_every) == 0
                ev: dict[str, Any] = {}
                # SYNC path (default; byte-identical to the legacy run): eval inline
                # off a snapshot and emit ONE combined train+eval row. The snapshot
                # round-trip is a no-op vs reading rt directly — same archive, same
                # exact numbers — so the sync output is unchanged.
                # ASYNC path (--async-eval): record the TRAIN-ONLY row now and SPAWN
                # the (identical) eval off a point-in-time snapshot in a background
                # thread; the eval row lands LATER, tagged with this snapshot epoch.
                # Training continues immediately — the ~13-min CPU eval no longer
                # blocks the MPS loop.
                if evaluated and not self._async_eval:
                    snap = self._snapshot_ema(rt)
                    ev = self._eval_snapshot(snap, spec, stage_index, self._global_epoch)

                self.telemetry.record(
                    EpochRecord(
                        stage_index=stage_index,
                        stage_name=spec.name,
                        epoch_in_stage=epoch_in_stage,
                        global_epoch=self._global_epoch,
                        loss=mean_loss,
                        pose_mse=mean_pose_mse,
                        adamw_lr=float(rt.adamw_opt.param_groups[0]["lr"]),
                        muon_lr=(
                            float(rt.muon_opt.param_groups[0]["lr"])
                            if rt.muon_opt is not None
                            else None
                        ),
                        grad_norm_adamw=gn_adamw,
                        grad_norm_muon=gn_muon,
                        # In ASYNC mode this row is train-only (the eval row is
                        # emitted separately by the worker, tagged with the snapshot
                        # epoch); ``evaluated`` here reflects the SYNC inline eval.
                        evaluated=evaluated and not self._async_eval,
                        d_seg=ev.get("d_seg"),
                        d_pose=ev.get("d_pose"),
                        rate=ev.get("rate"),
                        score=ev.get("score"),
                        archive_bytes=ev.get("archive_bytes"),
                        is_best=ev.get("is_best", False),
                    )
                )

                # ASYNC eval: schedule AFTER the train row is recorded so the
                # snapshot is taken at this epoch's completed EMA shadow. At most one
                # in-flight; over-cadence evals self-throttle (skip + log).
                if evaluated and self._async_eval:
                    self._schedule_async_eval(rt, spec, stage_index, self._global_epoch)

                # Checkpoint cadence (a death costs <= this many epochs).
                if (epoch_in_stage % self.cfg.checkpoint_every_epochs) == 0:
                    self._checkpoint(rt, spec, stage_index, epoch_in_stage)

                # Test-only simulated death: raise AFTER the checkpoint has landed
                # so a resumed run continues from this exact point. Production
                # never sets this; a real death (SIGKILL/OOM/preempt) is the
                # equivalent and is covered by the subprocess kill+restart test.
                if (
                    self._stop_after_global_epoch is not None
                    and self._global_epoch >= self._stop_after_global_epoch
                ):
                    raise _SimulatedDeath(self._global_epoch)

            # End of stage: ensure a checkpoint at the boundary + carry forward.
            self._checkpoint(rt, spec, stage_index, spec.epochs)
            carry_decoder = rt.decoder
            # Expose the final LIVE (non-EMA) decoder for post-run inspection (the
            # weight-entropy A/B reads the trained weights the rate lever shaped). NOT a
            # score surface — BEST/export use the EMA shadow as always. Set every stage so
            # it holds the last stage's live decoder at run end.
            self._final_decoder = rt.decoder
            carry_latents = rt.latents
            carry_ema_decoder = rt.ema_decoder
            carry_ema_latents = rt.ema_latents
            # Carry the Lever-4 sensitivity EMA forward (empty on the default path).
            carry_sensitivity_ema = dict(rt.tensor_sensitivity_ema)
            resume_pos = TorchCheckpointPosition(stage_index + 1, 0)

        # JOIN any in-flight async eval so the final BEST + last eval row land
        # before the DONE marker (the marker-on-exit contract). No-op in sync mode.
        self._join_async_eval()

        summary = {
            "status": "complete",
            "best_score": self.best_score,
            "best_ep": self.best_ep,
            "best_stage": self.best_stage,
            "n_stages": len(self.curriculum),
            "base_channels": self.cfg.base_channels,
            "async_eval": self._async_eval,
            "skipped_async_evals": self._skipped_evals,
            "authority": "[contest-CPU advisory] NON-PROMOTABLE — exact via upstream/evaluate.py",
        }
        write_done_marker(self.cfg.out_dir, summary)
        return summary

    def _checkpoint(self, rt: _StageRuntime, spec: StageSpec, stage_index: int, epoch_in_stage: int) -> None:
        state = self._capture_state(rt, spec)
        save_checkpoint(
            state,
            self.cfg.out_dir,
            TorchCheckpointPosition(stage_index, epoch_in_stage),
        )

    def export_production_archive(
        self,
        ema_sd: dict[str, torch.Tensor],
        ema_latents: torch.Tensor,
        *,
        sensitivity: dict[str, float] | None = None,
        score_finished: bool = False,
    ) -> dict[str, Any]:
        """The FINAL production export — bind the rate-attack + L3 finishing-kit into
        ONE byte-closed packet from a CONVERGED checkpoint (the P2 export entry point).

        This is the POST-CONVERGENCE export pass (NEVER an in-loop perturbation — the
        descending basin is untouched). It composes the two production levers that are
        applied at export time:

        * RATE-ATTACK (decoder-blob variable-level codec): ``_build_archive_and_eval_
          decoder`` already builds the rate-attacked base archive when
          ``cfg.lever4_variable_level_export_enabled`` is set, driven by the SAME online
          score-sensitivity EMA (``sensitivity``, module-keyed) the QAT consumed — the
          single-source SPINE (re-keyed onto the codec weight keys by ``_sensitivity_
          for_codec_weight_keys``). Pass the converged ``rt.tensor_sensitivity_ema`` (or
          the ``best_meta`` sensitivity snapshot) as ``sensitivity``.
        * L3 DISTORTION FINISHING-KIT (PR98 bias / T10 affine / S12): ``cfg.distortion_
          kit`` (a ``DistortionKitConfig``) is appended as a ≤54-byte trailing section
          via :func:`finish_checkpoint_with_distortion_kit`. The substrate ``inflate.sh``
          reads the section and runs ``apply_distortion_kit_to_raw_frames`` after the
          vendored ``inflate.py``.

        DEFAULT-PRESERVING (the byte-identical contract):
        ``cfg.lever4_variable_level_export_enabled`` OFF (or uniform sensitivity) AND
        ``cfg.distortion_kit`` None/disabled → the returned ``finished_archive`` is
        BYTE-IDENTICAL to the pristine vendored ``build_archive`` (no rate-attack
        splice, no distortion section). Each lever flips on only when its config is
        non-trivially set.

        ``score_finished=True`` re-scores the finished packet with the kit applied
        POST-round on the AUTHORITY device via :func:`kit_aware_exact_eval` — the HONEST
        L3 score (the +54 B rate cost AND the kit's d_seg/d_pose effect are both
        measured; never a bytes-only-without-distortion fake). Requires the scorer to
        expose ``distortion_net`` + ``video_path`` (the real scorer context). Off by
        default ($0, no scorer streaming); the BYTE deltas are returned regardless.

        Score-claim discipline: the BYTE saving (rate-attack + the ≤54-B section) is
        real + measurable; any returned score is ``[contest-CPU advisory]`` NON-
        PROMOTABLE until a 600-pair byte-closed dual CPU/CUDA exact eval (G3). NO score
        is claimed from calling this method.
        """
        meta_dict: dict[str, Any] = {
            "n_pairs": self.n_pairs,
            "latent_dim": self.cfg.latent_dim,
            "base_channels": self.cfg.base_channels,
            "eval_size": [_EVAL_H, _EVAL_W],
        }
        # (1) rate-attack base archive (the SPINE: module-keyed EMA -> codec levels) +
        # the parse-back eval decoder/latents (byte-closed faithful).
        base_archive, eval_dec, eval_latents = self._build_archive_and_eval_decoder(
            ema_sd, ema_latents, meta_dict, sensitivity=sensitivity
        )
        # (2) L3 finishing-kit section (default-OFF -> byte-identical no-op).
        finish = finish_checkpoint_with_distortion_kit(
            base_archive, self.cfg.distortion_kit, vendored=self.v
        )
        finished_archive: bytes = finish["finished_archive"]
        result: dict[str, Any] = {
            "finished_archive": finished_archive,
            "base_archive_bytes": finish["base_archive_bytes"],
            "section_bytes": finish["section_bytes"],
            "added_bytes": finish["added_bytes"],
            "finished_archive_bytes": len(finished_archive),
            "rate_attack_enabled": bool(self.cfg.lever4_variable_level_export_enabled),
            "decoder_codec": meta_dict.get("decoder_codec", "vendored"),
            "distortion_section_present": finish["section_bytes"] > 0,
            "is_byte_identical_to_vendored_base": finish["is_byte_identical"]
            and meta_dict.get("decoder_codec", "vendored") == "vendored",
            "authority": "[contest-CPU advisory] NON-PROMOTABLE until dual CPU/CUDA exact eval",
            "score_claim": False,
        }
        if "lever4_variable_level_export" in meta_dict:
            result["lever4_variable_level_export"] = meta_dict["lever4_variable_level_export"]
        if "variable_level_waterfill" in meta_dict:
            result["variable_level_waterfill"] = meta_dict["variable_level_waterfill"]
        if not score_finished:
            return result
        # (3) HONEST kit-aware re-score (the L3 effect MEASURED, not faked). The rate
        # term uses the FINISHED packet bytes (base + section); the distortion uses the
        # kit applied POST-round on the AUTHORITY device.
        distortion_net = getattr(self.scorer, "distortion_net", None)
        video_path = getattr(self.scorer, "video_path", None)
        score_helper = getattr(self.scorer, "_score", None)
        if distortion_net is None or video_path is None or score_helper is None:
            raise ValueError(
                "score_finished=True requires the real scorer context (distortion_net "
                "+ video_path + total_video_bytes); the synthetic scorer cannot kit-aware "
                "re-score. Leave score_finished=False for byte-only export."
            )
        tvb = score_helper.total_video_bytes(video_path)
        ev = kit_aware_exact_eval(
            eval_dec,
            eval_latents.to(self.device),
            distortion_net,
            video_path,
            distortion_kit=self.cfg.distortion_kit,
            archive_bytes=len(finished_archive),
            total_video_bytes=tvb,
            device=str(self.device),
        )
        result["finished_score"] = ev
        return result


@dataclass
class VendoredBundle:
    """The vendored PR95 primitives the driver re-drives (NO source edits).

    Built via :func:`import_vendored_bundle`; kept as an explicit handle so the
    driver's vendored-dependency surface is auditable in one place.
    """

    HNeRVDecoder: type
    Muon: type
    partition_params_for_muon: Callable
    ema_update: Callable
    apply_qat: Callable
    restore_qat: Callable
    cat_entropy_v2: Callable
    build_archive: Callable
    parse_archive: Callable


def import_vendored_bundle() -> VendoredBundle:
    """Import the vendored PR95 primitives WITHOUT the challenge-scorer dependency.

    ``model`` / ``optim`` / ``losses`` / ``codec`` import cleanly (no challenge
    repo needed); only ``data`` / ``score`` pull in the frozen scorer (those are
    used by :class:`RealScorerContext`, not here). So this bundle is importable on
    a host without the challenge weights — the resume/checkpoint machinery is
    architecture-agnostic.
    """
    from tac.torch_vehicle.vendored_imports import import_vendored

    model = import_vendored("model")
    optim = import_vendored("optim")
    losses = import_vendored("losses")
    codec = import_vendored("codec")
    return VendoredBundle(
        HNeRVDecoder=model.HNeRVDecoder,
        Muon=optim.Muon,
        partition_params_for_muon=optim.partition_params_for_muon,
        ema_update=losses.ema_update,
        apply_qat=losses.apply_qat,
        restore_qat=losses.restore_qat,
        cat_entropy_v2=losses.cat_entropy_v2,
        build_archive=codec.build_archive,
        parse_archive=codec.parse_archive,
    )


def _json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Track-A DISTORTION finishing-kit — the inflate-side post-convergence pass.
#
# These are the production wiring of the kit (PR98 bias / T10 affine / S12
# certification) onto a CONVERGED checkpoint's exported archive. They are kept as
# standalone functions (operate on any ``best/`` archive + an injected scorer) so
# the live basin loop is untouched (the kit is a FINAL export pass, never an
# in-loop perturbation — ``cfg.distortion_kit`` defaults None → byte-identical).
# ---------------------------------------------------------------------------
@torch.inference_mode()
def kit_aware_exact_eval(
    eval_decoder: nn.Module,
    eval_latents: torch.Tensor,
    distortion_net: Any,
    video_path: Any,
    *,
    distortion_kit: Any = None,
    archive_bytes: int,
    total_video_bytes: int,
    batch_pairs: int = 8,
    device: str = "cpu",
) -> dict[str, float]:
    """Faithful kit-aware exact eval — MIRRORS the vendored ``score.evaluate_decoder``
    (render → bicubic↑ camera → clamp/round/uint8 → ``compute_distortion``) but
    INSERTS the distortion-kit postproc on the POST-ROUND uint8 frames — EXACTLY where
    the substrate's ``inflate.sh`` runs ``apply_distortion_kit_to_raw_frames`` after
    the vendored ``inflate.py``. So the BEST-tracked score equals what the FINISHED
    contest packet produces (the production inflate path), not a pre-round
    approximation (the ≤1 ULP gap matters for a ±1-bias fit). NO vendored edit.

    A ``distortion_kit=None`` (or disabled/identity) is the byte-identical no-op:
    the frames pass through unchanged so the score equals the vendored eval.
    """
    import av
    from frame_utils import yuv420_to_rgb

    from tac.torch_vehicle.distortion_finishing_kit import (
        DistortionKitConfig,
        apply_distortion_kit_to_raw_frames,
    )

    _EVAL_H, _EVAL_W = 384, 512
    _CAM_H, _CAM_W = 874, 1164
    if distortion_kit is None:
        distortion_kit = DistortionKitConfig(enabled=False)

    eval_decoder.eval()
    n_pairs = int(eval_latents.shape[0])
    dev = torch.device(device)
    container = av.open(str(video_path))
    frames_iter = container.decode(container.streams.video[0])
    seg_total = 0.0
    pose_total = 0.0
    count = 0

    def _next_pair() -> torch.Tensor | None:
        f0 = None
        for frame in frames_iter:
            rgb = yuv420_to_rgb(frame)
            if f0 is None:
                f0 = rgb
                continue
            return torch.stack([f0, rgb])
        return None

    pair_idx = 0
    while pair_idx < n_pairs:
        batch_gt = []
        for _ in range(min(batch_pairs, n_pairs - pair_idx)):
            pair = _next_pair()
            if pair is None:
                break
            batch_gt.append(pair)
        if not batch_gt:
            break
        gt = torch.stack(batch_gt).to(dev)  # (B,2,H,W,3) uint8
        b = gt.shape[0]
        idx = torch.arange(pair_idx, pair_idx + b, device=dev)
        z = eval_latents[idx].to(dev)
        decoded = eval_decoder(z)  # (B,2,3,384,512) float
        flat = decoded.reshape(b * 2, 3, _EVAL_H, _EVAL_W)
        up = torch.nn.functional.interpolate(
            flat, size=(_CAM_H, _CAM_W), mode="bicubic", align_corners=False
        )
        cam_float = up.reshape(b, 2, 3, _CAM_H, _CAM_W).permute(0, 1, 3, 4, 2)
        # Vendored-inflate-faithful: the contest packet rounds/casts to uint8 in the
        # vendored inflate.py, THEN the substrate's inflate.sh runs the numpy raw-frame
        # kit postproc. So the eval applies the kit POST-round (on the uint8 frames)
        # to match the PRODUCTION inflate path EXACTLY (not pre-round; the ≤1 ULP gap
        # matters for a ±1-bias fit). NO vendored edit — this MIRRORS the substrate's
        # own inflate.sh chain (vendored inflate -> apply_distortion_kit_to_raw_frames).
        cand = cam_float.clamp(0, 255).round().to(torch.uint8)
        if not (distortion_kit is None or distortion_kit.is_identity):
            # Flatten (B,2,...) -> (2B,...) so frame PARITY matches the raw-frame layout
            # (pair k -> frames 2k, 2k+1), apply the numpy kit, reshape back.
            raw = cand.reshape(b * 2, _CAM_H, _CAM_W, 3).cpu().numpy()
            finished = apply_distortion_kit_to_raw_frames(raw, distortion_kit)
            cand = torch.from_numpy(finished).reshape(b, 2, _CAM_H, _CAM_W, 3).to(dev)
        pose_d, seg_d = distortion_net.compute_distortion(gt, cand)
        seg_total += float(seg_d.sum().item())
        pose_total += float(pose_d.sum().item())
        count += b
        pair_idx += b
    container.close()

    d_seg = seg_total / max(count, 1)
    d_pose = pose_total / max(count, 1)
    rate = archive_bytes / float(total_video_bytes)
    score = 100.0 * d_seg + (10.0 * d_pose) ** 0.5 + 25.0 * rate
    return {"seg_distortion": d_seg, "pose_distortion": d_pose, "rate": rate, "score": score}


def finish_checkpoint_with_distortion_kit(
    best_archive_bytes: bytes,
    distortion_kit: Any,
    vendored: VendoredBundle | None = None,
) -> dict[str, Any]:
    """Append the distortion-kit section to a converged ``best/best_archive.bin``.

    The finished packet = the vendored archive (UNCHANGED bytes) + the ~54-byte
    distortion section appended. A DISABLED/identity kit appends NOTHING → the
    finished bytes are BYTE-IDENTICAL to the input (the default-OFF contract).

    Returns ``{finished_archive, base_archive_bytes, section_bytes, added_bytes,
    is_byte_identical}``. The substrate's ``inflate.sh`` reads the trailing section
    (if present) and runs ``apply_distortion_kit_to_raw_frames`` after the vendored
    ``inflate.py`` — the vendored runtime is NOT edited."""
    from tac.torch_vehicle.distortion_finishing_kit import (
        DISTORTION_SECTION_MAGIC,
        DistortionKitConfig,
        serialize_distortion_section,
    )

    if distortion_kit is None:
        distortion_kit = DistortionKitConfig(enabled=False)
    # Refuse double-appends (idempotent fail-closed): the base must not already
    # carry a trailing distortion section.
    if best_archive_bytes[-54:][:4] == DISTORTION_SECTION_MAGIC:
        raise ValueError(
            "base archive already carries a distortion section (double-finish "
            "forbidden — finish the PRISTINE converged archive)"
        )
    section = serialize_distortion_section(distortion_kit)
    finished = best_archive_bytes + section
    return {
        "finished_archive": finished,
        "base_archive_bytes": len(best_archive_bytes),
        "section_bytes": len(section),
        "added_bytes": len(finished) - len(best_archive_bytes),
        "is_byte_identical": finished == best_archive_bytes,
    }


def split_finished_archive(finished_archive: bytes) -> tuple[bytes, Any]:
    """Inverse of :func:`finish_checkpoint_with_distortion_kit`: split a finished
    packet into ``(base_archive_bytes, DistortionKitConfig)``.

    A packet WITHOUT a trailing section → ``(finished_archive, disabled config)``.
    The substrate's numpy inflate uses this to recover the base ``0.bin`` (handed to
    the vendored ``inflate.py``) + the kit (applied to the raw frames after)."""
    from tac.torch_vehicle.distortion_finishing_kit import (
        _SECTION_STRUCT,
        DISTORTION_SECTION_MAGIC,
        DistortionKitConfig,
        parse_distortion_section,
    )

    seclen = _SECTION_STRUCT.size
    if len(finished_archive) >= seclen and finished_archive[-seclen:][:4] == DISTORTION_SECTION_MAGIC:
        base = finished_archive[:-seclen]
        cfg = parse_distortion_section(finished_archive[-seclen:])
        return base, cfg
    return finished_archive, DistortionKitConfig(enabled=False)


__all__ = [
    "ScorerContext",
    "TorchVehicleConfig",
    "TorchVehicleDriver",
    "VendoredBundle",
    "finish_checkpoint_with_distortion_kit",
    "import_vendored_bundle",
    "kit_aware_exact_eval",
    "split_finished_archive",
]
