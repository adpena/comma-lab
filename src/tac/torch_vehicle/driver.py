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
        if int(self.pose_grad_every_k) < 1:
            raise ValueError(
                f"pose_grad_every_k must be >= 1 (1 = compute pose every epoch = "
                f"byte-identical); got {self.pose_grad_every_k}"
            )
        if float(self.pose_grad_resume_threshold) < 0.0:
            raise ValueError(
                f"pose_grad_resume_threshold must be >= 0.0; got "
                f"{self.pose_grad_resume_threshold}"
            )
        if self.lever4_variable_level_export_enabled and self.variable_level_waterfill_enabled:
            raise ValueError(
                "lever4_variable_level_export_enabled and variable_level_waterfill_enabled "
                "are mutually exclusive: both replace the SAME export decoder blob with a "
                "variable-level grid, but from DIFFERENT level sources (Lever-4's online "
                "score-sensitivity EMA vs the offline measured RD table). Enable exactly one."
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
            if self.pose_film_enabled:
                raise ValueError(
                    "taper_channels requires pose_film_enabled=False: the configurable-taper "
                    "carrier has no FiLM wrapper here (combine them as a separate step)."
                )
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
        if hasattr(decoder, "pose_mlp") and isinstance(getattr(decoder, "decoder", None), nn.Module):
            target = decoder.decoder
        target.load_state_dict({k: v.to(self.train_device) for k, v in sd.items()})
        if int(lat.shape[0]) != self.n_pairs or int(lat.shape[1]) != self.cfg.latent_dim:
            raise ValueError(
                f"warm-start latents shape {tuple(lat.shape)} != (n_pairs={self.n_pairs}, "
                f"latent_dim={self.cfg.latent_dim}); cannot warm-start a different basis"
            )
        return nn.Parameter(lat.detach().clone().to(self.train_device))

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
            adamw_opt = torch.optim.AdamW(adamw_groups, weight_decay=0.0)
        # The clip set = the decoder AdamW params PLUS the FiLM params (so grad-clip at
        # ``_train_one_epoch`` covers the capped FiLM group too). == adamw_params when
        # FiLM is off (byte-identical clip).
        adamw_clip_params = adamw_params + film_params

        eta_min_ratio = max(spec.lr_floor_ratio / spec.adamw_lr, 1e-3)

        def lr_lambda(epoch: int) -> float:
            return max(0.5 * (1 + math.cos(math.pi * epoch / spec.epochs)), eta_min_ratio)

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

            muon_sched = torch.optim.lr_scheduler.LambdaLR(muon_opt, muon_lr_lambda)
        else:
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
                # Pose-throttle (the "pose is solved → stop paying for it" speed lever):
                # compute the pose cotangent EVERY epoch (k<=1, byte-identical default),
                # while pose is still converging/drifted (last pose_mse > resume_threshold),
                # the FIRST time (never computed), or on the every-k cadence epoch.
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
                    compute_pose=do_pose,
                )
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
                pose_mse = F.mse_loss(pose_pred6, self.scorer.pose_targets[idx])
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
        if spec.cat_lambda > 0:
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
        head losses; not authority — telemetry only)."""
        # --- SegNet path on the TRAIN device (MPS) ---
        frames_seg = decoded_bhwc.detach().requires_grad_(True)  # train_device leaf
        seg_out = self.scorer.seg_forward_train(frames_seg)
        seg_targets = self.scorer.seg_targets_hard[idx]
        seg_l = _seg_loss_for_spec(spec, seg_out, seg_targets, temperature=temperature)
        (spec.seg_weight * seg_l).backward()
        cot_seg = frames_seg.grad  # d(w_seg*seg_l)/dF on train_device

        # --- PoseNet path on the CPU AUTHORITY (zero pose drift) ---
        # Pose-throttle: when ``compute_pose`` is False (pose is solved + this is an
        # off-cadence epoch), SKIP the expensive CPU FastViT fwd+bwd entirely and flow
        # ONLY the SegNet cotangent into the decoder. ``compute_pose`` defaults True so
        # every existing caller (k=1, tests) is byte-identical. The pose term re-engages
        # on the cadence epoch or if it drifts above the resume threshold (caller logic).
        if compute_pose:
            frames_pose = (
                decoded_bhwc.detach().to(self.device).requires_grad_(True)
            )  # authority-device leaf, same values
            pose_pred6 = self.scorer.pose_forward_authority(frames_pose)
            pose_targets = self._pose_targets_authority()[idx.to(self.device)]
            pose_mse = F.mse_loss(pose_pred6, pose_targets)
            pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)
            (spec.pose_weight * pose_l).backward()
            cot_pose = frames_pose.grad  # d(w_pose*pose_l)/dF on the CPU authority
            combined = cot_seg + cot_pose.to(cot_seg.device)
            pose_mse_val = float(pose_mse.item())
            self._last_pose_mse = pose_mse_val
            loss_val = float((spec.seg_weight * seg_l).item()) + float(
                (spec.pose_weight * pose_l).item()
            )
        else:
            # SegNet-only cotangent (pose throttled this epoch). loss_val is the seg term
            # only; pose_mse_val carries the last-computed value for telemetry continuity.
            combined = cot_seg
            pose_mse_val = (
                float("nan") if self._last_pose_mse is None else self._last_pose_mse
            )
            loss_val = float((spec.seg_weight * seg_l).item())

        # --- inject the combined frame cotangent into the decoder ---
        decoded_bhwc.backward(gradient=combined)
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
            # EMA-warmup step counter — persisted so a resume CONTINUES the warmup decay
            # schedule (else _ema_step resets to 0 → the decay snaps back to 0.1 and the
            # shadow takes a spurious jolt). Always 0 on the default (ema_warmup off) path,
            # so it round-trips as 0 → byte-identical for the faithful basin.
            "ema_step": int(self._ema_step),
            "base_channels": self.cfg.base_channels,
            "latent_dim": self.cfg.latent_dim,
            "n_pairs": self.n_pairs,
            # Persist the taper schedule so resume fails closed on a taper change (the
            # codec stays schedule-agnostic; this makes the resume-safety EXPLICIT).
            "taper_channels": self.cfg.taper_channels,
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

        # Build the per-tensor level map from the online sensitivity EMA, keyed by the
        # WEIGHT state-dict keys the codec checks (``<module>.weight``). Tensors absent
        # from the sensitivity map (biases, FiLM params, etc.) get the base 127 levels.
        weight_keys = [k for k in archive_decoder_sd if k.endswith(".weight")]
        levels_by_weight = levels_from_sensitivity_for_codec(sensitivity, weight_keys)
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
            self.v.build_archive, archive_decoder_sd, ema_latents, md, stored_pose
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

            # Build decoder/latents for this stage (carry from prior stage, or init).
            resuming_into_this_stage = (
                merged is not None and stage_index == resume_pos.stage_index
            )
            if carry_decoder is None:
                decoder = self._new_decoder()  # train-device (gradient backend)
                if (
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
