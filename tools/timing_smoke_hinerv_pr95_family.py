#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase B0 timing smoke for the PR95-family HiNeRV architecture on local MLX.

Per CLAUDE.md "Long-burn score-lowering campaign default" (a campaign MUST
include "a timing-smoke command that measures seconds/epoch") + "Carmack
MVP-first phasing" (the smallest faithful local-CPU/MLX surface that exercises
the assumption BEFORE any paid dispatch). This tool turns the B1 600-pair
29650-epoch PR95 curriculum cost from a GUESS into MEASURED hours.

What it measures (the FAITHFUL per-epoch training-step compute):

  Surface A (primary, robust): the REAL PR95-family MLX renderer
    (``tac.substrates.hi_nerv.mlx_renderer.HinervSubstrateMLX`` built from the
    default ``HinervConfig`` = decoder_channels (48,40,32,24,20,16,12),
    3-scale latent pyramid, 6+ PixelShuffle upsample stages, sin activation,
    bilinear final resize per CLAUDE.md L18) forward + backward through a
    score-DOMAIN reconstruction loss against REAL contest frames decoded from
    ``upstream/videos/0.mkv`` (synthetic data FORBIDDEN per CLAUDE.md
    Catalog #114), with ``mlx.nn.value_and_grad`` + an AdamW optimizer step.
    This is the dominant per-epoch wall-clock term.

  Surface B (faithful score-aware, best-effort): the canonical
    ``MlxScoreAwareAdapter.train_step`` with REAL gradient-free MLX SegNet +
    PoseNet teacher caches (the per-pair teacher logits/poses are computed ONCE
    at setup, NOT per epoch — the renderer gradient flows through small
    learnable student heads distilled toward those caches). When this path
    constructs cleanly it is reported as the faithful score-aware number;
    otherwise the blocker is recorded and Surface A stands as a tight LOWER
    bound on the real per-epoch cost (the score-aware student-head + KL/MSE
    term is a small MLP on top of the renderer fwd+bwd).

AUTHORITY: MLX timing is HARDWARE-ADVISORY only. Every number is tagged
``[macOS-MLX research-signal]`` with ``score_claim=false``,
``promotion_eligible=false``, ``promotable=false``. This tool emits TIMING and
a COST MODEL only. It NEVER claims a contest score. The score is exact-eval'd
later (B2) on byte-closed archive bytes via ``upstream/evaluate.py``.

Disk hygiene (CLAUDE.md "Local Disk, SSD Spill, Auto-Cleanup"): this tool
writes only a small JSON manifest under ``.omx/research/`` (durable, small
metadata). It creates NO bulk artifacts (no checkpoints, no inflated frames,
no caches persisted) and uses no ``/tmp`` paths.

Usage:
    .venv/bin/python tools/timing_smoke_hinerv_pr95_family.py
    .venv/bin/python tools/timing_smoke_hinerv_pr95_family.py \
        --timing-pairs 16 --epochs 5 --batch-pairs 16 \
        --output .omx/research/timing_smoke_hinerv_pr95_family_<utc>.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Make the repo importable without relying on an installed wheel.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "upstream"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# The contest curriculum / scorer constants used by the projection + cost model.
PR95_FULL_CURRICULUM_EPOCHS = 29_650  # CLAUDE.md L14: 8-stage 29,650-epoch curriculum
PR95_PRAGMATIC_REDUCED_EPOCHS = 3_000  # a credible local-feasible reduced schedule
CONTEST_FULL_PAIRS = 600  # 1200 frames / 2 = 600 per-frame-PAIR latents (CLAUDE.md L19)

# CLAUDE.md "GPU budget and compute resources" price/performance table. The
# "speed_vs_t4" column is the published relative throughput; we use it to scale
# the measured MLX per-epoch cost into a rough paid-GPU per-epoch estimate via
# an MLX<->T4 anchor (research-signal only; the real paid number is measured at
# B1 launch, never here).
GPU_RATE_TABLE = {
    "vast_rtx_4090": {"usd_per_hr": 0.25, "speed_vs_t4": 4.5, "note": "primary new-experiment GPU"},
    "aws_t4": {"usd_per_hr": 0.22, "speed_vs_t4": 1.0, "note": "scale-out / auth-eval fleet"},
    "modal_t4": {"usd_per_hr": 0.59, "speed_vs_t4": 1.0, "note": "existing infra"},
    "modal_a100": {"usd_per_hr": 1.10, "speed_vs_t4": 6.0, "note": "approx; a100 80GB-class"},
}

# Rough M-series-MLX vs T4 throughput anchor for a small (~340K param) conv
# renderer. This is a COARSE research-signal multiplier ONLY (used to bracket
# the paid cost so the operator can decide local-vs-paid); the authoritative
# paid per-epoch cost is the B1 timing smoke on the actual GPU. Tagged advisory.
MLX_LOCAL_VS_T4_THROUGHPUT_ANCHOR = 0.5  # CLAUDE.md GPU table: "Local M5 Max MPS ~0.5x T4"

_ADVISORY_TAG = "[macOS-MLX research-signal]"


@dataclass(frozen=True)
class StepTiming:
    """Per-step wall-clock samples for one timed surface."""

    surface: str
    batch_pairs: int
    warmup_steps: int
    steps_timed: int
    per_step_seconds: tuple[float, ...]
    notes: str
    score_aware_included: bool

    def median_step_seconds(self) -> float:
        return statistics.median(self.per_step_seconds) if self.per_step_seconds else float("nan")

    def mean_step_seconds(self) -> float:
        return statistics.fmean(self.per_step_seconds) if self.per_step_seconds else float("nan")

    def steps_per_epoch(self, total_pairs: int) -> float:
        # One epoch = one pass over all ``total_pairs`` pairs in ``batch_pairs``
        # chunks (ceil division).
        return float((int(total_pairs) + int(self.batch_pairs) - 1) // int(self.batch_pairs))

    def seconds_per_epoch(self, total_pairs: int) -> float:
        return self.median_step_seconds() * self.steps_per_epoch(total_pairs)


def _utc_now_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _param_counts(cfg) -> dict[str, int]:
    """Return torch + MLX param counts and confirm they match."""

    from tac.substrates.hi_nerv.architecture import HinervSubstrate
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    torch_model = HinervSubstrate(cfg)
    torch_params = int(torch_model.num_parameters())
    mlx_model = HinervSubstrateMLX(cfg)
    mlx_params = int(mlx_model.num_parameters())
    return {
        "torch_num_parameters": torch_params,
        "mlx_num_parameters": mlx_params,
        "torch_mlx_match": bool(torch_params == mlx_params),
    }


def _decode_real_pairs(*, num_pairs: int, height: int, width: int):
    """Decode ``num_pairs`` REAL contest pairs (synthetic FORBIDDEN)."""

    from tac.substrates._shared.mlx_score_aware.targets import decode_mlx_targets

    video = _REPO_ROOT / "upstream" / "videos" / "0.mkv"
    if not video.is_file():
        raise FileNotFoundError(
            f"contest video not found at {video}; cannot run a faithful timing "
            "smoke on REAL pairs (synthetic data is FORBIDDEN per CLAUDE.md)"
        )
    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        str(video),
        num_pairs=int(num_pairs),
        output_height=int(height),
        output_width=int(width),
    )
    return target_rgb_0, target_rgb_1


def _time_renderer_fwd_bwd(
    *,
    cfg,
    target_rgb_0,
    target_rgb_1,
    batch_pairs: int,
    epochs: int,
    warmup_steps: int,
) -> StepTiming:
    """Surface A: time the REAL MLX renderer fwd+bwd training step on real pairs.

    Loss is the score-DOMAIN reconstruction MSE in ``[0, 1]`` against the real
    decoded target frames (the dominant per-epoch compute). ``mlx.nn``'s
    ``value_and_grad`` produces grads over EVERY trainable renderer parameter;
    an AdamW step is applied so the timing reflects the full optimizer update,
    not just the backward.
    """

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    model = HinervSubstrateMLX(cfg)
    n_pairs = int(target_rgb_0.shape[0])
    bp = max(1, min(int(batch_pairs), n_pairs))

    # Targets are NHWC [0,1]; the renderer returns (B,2,3,H,W) in [0,255].
    # Build NCHW [0,1] target pairs once (transpose is cheap; done eagerly).
    tgt0 = mx.transpose(target_rgb_0, (0, 3, 1, 2))  # (N,3,H,W)
    tgt1 = mx.transpose(target_rgb_1, (0, 3, 1, 2))
    mx.eval(tgt0, tgt1)

    opt = optim.AdamW(learning_rate=1e-3)

    # Bound the number of timed steps: warmup + (epochs passes over the pairs).
    n_chunks_per_epoch = (n_pairs + bp - 1) // bp
    total_steps = int(warmup_steps) + int(epochs) * n_chunks_per_epoch

    def _batch_idx(step: int):
        # Deterministic round-robin over contiguous pair chunks.
        chunk = step % n_chunks_per_epoch
        start = chunk * bp
        end = min(start + bp, n_pairs)
        idx = mx.array(np.arange(start, end, dtype=np.int64))
        return idx, start, end

    def make_loss(idx, start, end):
        def loss_fn(m):
            out = m(idx)  # (b,2,3,H,W) in [0,255]
            rec0 = out[:, 0] / 255.0
            rec1 = out[:, 1] / 255.0
            d0 = rec0 - tgt0[start:end]
            d1 = rec1 - tgt1[start:end]
            return mx.mean(d0 * d0) + mx.mean(d1 * d1)

        return loss_fn

    lvg = nn.value_and_grad

    per_step: list[float] = []
    last_loss = float("nan")
    for step in range(total_steps):
        idx, start, end = _batch_idx(step)
        loss_fn = make_loss(idx, start, end)
        value_and_grad_fn = lvg(model, loss_fn)
        t0 = time.perf_counter()
        loss, grads = value_and_grad_fn(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)  # force the lazy graph to execute
        t1 = time.perf_counter()
        last_loss = float(loss)
        if not np.isfinite(last_loss):
            raise RuntimeError(
                f"Surface A renderer loss became non-finite at step {step}: {last_loss}"
            )
        if step >= int(warmup_steps):
            per_step.append(t1 - t0)

    return StepTiming(
        surface="renderer_fwd_bwd_recon_mse",
        batch_pairs=bp,
        warmup_steps=int(warmup_steps),
        steps_timed=len(per_step),
        per_step_seconds=tuple(per_step),
        notes=(
            "REAL MLX HiNeRV renderer fwd+bwd through score-domain reconstruction "
            f"MSE on REAL contest pairs; AdamW step; final loss={last_loss:.6f}. "
            "Score-aware student-head/KL/MSE term EXCLUDED -> per-epoch projection "
            "is a tight LOWER bound (the renderer fwd+bwd is the dominant term)."
        ),
        score_aware_included=False,
    )


def _time_score_aware_train_step(
    *,
    cfg,
    target_rgb_0,
    target_rgb_1,
    batch_pairs: int,
    epochs: int,
    warmup_steps: int,
) -> tuple[StepTiming | None, dict[str, object]]:
    """Surface B (best-effort): time the canonical score-aware adapter train_step.

    Builds the real gradient-free MLX SegNet + PoseNet teacher caches ONCE
    (timed separately, amortized), threads small learnable student heads into a
    ``RendererBundle``, and times ``MlxScoreAwareAdapter.train_step``. This path
    is intentionally defensive: the adapter is a large, actively-evolving module
    and its exact contract may shift. On ANY failure we record the blocker and
    return ``None`` so Surface A stands as the faithful lower bound.
    """

    diag: dict[str, object] = {
        "attempted": True,
        "succeeded": False,
        "teacher_setup_seconds": None,
        "blocker": None,
    }
    try:
        import dataclasses

        import mlx.core as mx

        from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
        from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle
        from tac.substrates._shared.mlx_score_aware.loss import (
            build_mlx_posenet_pair_teacher,
            build_mlx_segnet_pair_teacher,
        )
        from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

        n_pairs = int(target_rgb_0.shape[0])
        bp = max(1, min(int(batch_pairs), n_pairs))
        model = HinervSubstrateMLX(cfg)

        # Minimal bundle for teacher construction (targets at contest 384x512).
        base_bundle = RendererBundle(
            target_rgb_0=target_rgb_0,
            target_rgb_1=target_rgb_1,
            model=model,
            num_pairs=n_pairs,
        )

        t_setup0 = time.perf_counter()
        seg_teacher = build_mlx_segnet_pair_teacher(
            base_bundle, upstream_dir=str(_REPO_ROOT / "upstream"), device="cpu"
        )
        pose_teacher = build_mlx_posenet_pair_teacher(
            base_bundle, upstream_dir=str(_REPO_ROOT / "upstream"), device="cpu"
        )
        mx.eval(getattr(seg_teacher, "teacher_logits_thwk", mx.array(0.0)))
        t_setup1 = time.perf_counter()
        diag["teacher_setup_seconds"] = round(t_setup1 - t_setup0, 4)

        # Build learnable student heads aligned to the teacher caches using the
        # CANONICAL builders (mirrors the canonical z7 real-hinton wiring +
        # mlx_score_aware_full_main path). NO hand-rolled head shapes.
        from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
            build_learnable_pose_student_head,
            build_learnable_student_head,
        )

        seg_num_classes = int(getattr(seg_teacher, "num_classes", 5))
        pose_dims = int(getattr(pose_teacher, "pose_dims", 6))
        seg_head = build_learnable_student_head(
            num_classes=seg_num_classes, in_channels=3, seed=0
        )
        pose_head = build_learnable_pose_student_head(pose_dims=pose_dims, seed=0)

        bundle = dataclasses.replace(
            base_bundle,
            distillation_weight=0.5,
            scorer_teacher=seg_teacher,
            learnable_student_head=seg_head,
            pose_distillation_weight=1.0,
            pose_scorer_teacher=pose_teacher,
            learnable_pose_student_head=pose_head,
            pose_dims=pose_dims,
            distillation_num_classes=seg_num_classes,
            # REAL teachers — allow_mock_scorer_teacher stays False (default) per
            # CLAUDE.md "NO FAKE IMPLEMENTATIONS"; this is the faithful score path.
        )

        adapter = MlxScoreAwareAdapter(bundle, substrate_id="hi_nerv_b0_timing")

        n_chunks_per_epoch = (n_pairs + bp - 1) // bp
        total_steps = int(warmup_steps) + int(epochs) * n_chunks_per_epoch
        per_step: list[float] = []
        for step in range(total_steps):
            chunk = step % n_chunks_per_epoch
            start = chunk * bp
            end = min(start + bp, n_pairs)
            # Canonical batch is an MLX int array (mirrors the z7 wiring), NOT a
            # Python list — the adapter calls mx.take(latents, batch, axis=0).
            idx = mx.array(np.arange(start, end, dtype=np.int32))
            t0 = time.perf_counter()
            # Canonical 3-arg train_step(batch, learning_rate, loss_weights).
            adapter.train_step(idx, 1e-3, {})
            mx.eval(model.parameters())
            t1 = time.perf_counter()
            if step >= int(warmup_steps):
                per_step.append(t1 - t0)

        diag["succeeded"] = True
        return (
            StepTiming(
                surface="score_aware_adapter_train_step",
                batch_pairs=bp,
                warmup_steps=int(warmup_steps),
                steps_timed=len(per_step),
                per_step_seconds=tuple(per_step),
                notes=(
                    "Canonical MlxScoreAwareAdapter.train_step with REAL "
                    "gradient-free MLX SegNet+PoseNet teacher caches (one-time "
                    "setup amortized); renderer grad flows through learnable "
                    "student heads distilled toward the caches. FAITHFUL "
                    "score-aware per-epoch cost."
                ),
                score_aware_included=True,
            ),
            diag,
        )
    except Exception as exc:  # best-effort surface; record the blocker, never fake
        diag["blocker"] = f"{type(exc).__name__}: {exc}"
        diag["traceback_tail"] = "".join(traceback.format_exc().splitlines(keepends=True)[-6:])
        return None, diag


def _projection_block(
    *,
    seconds_per_epoch: float,
    label: str,
    score_aware_included: bool,
) -> dict[str, object]:
    """Project full + reduced curricula and the local-vs-paid cost model."""

    def _hours(epochs: int) -> float:
        return seconds_per_epoch * float(epochs) / 3600.0

    full_hours = _hours(PR95_FULL_CURRICULUM_EPOCHS)
    reduced_hours = _hours(PR95_PRAGMATIC_REDUCED_EPOCHS)

    # Paid-GPU bracket: scale MLX per-epoch by the coarse MLX<->T4 anchor to get
    # a T4-equivalent per-epoch, then by each GPU's published speed_vs_t4.
    t4_equiv_seconds_per_epoch = seconds_per_epoch * MLX_LOCAL_VS_T4_THROUGHPUT_ANCHOR
    paid_estimates: dict[str, object] = {}
    for gpu, spec in GPU_RATE_TABLE.items():
        gpu_seconds_per_epoch = t4_equiv_seconds_per_epoch / float(spec["speed_vs_t4"])
        full_gpu_hours = gpu_seconds_per_epoch * PR95_FULL_CURRICULUM_EPOCHS / 3600.0
        reduced_gpu_hours = gpu_seconds_per_epoch * PR95_PRAGMATIC_REDUCED_EPOCHS / 3600.0
        paid_estimates[gpu] = {
            "usd_per_hr": spec["usd_per_hr"],
            "speed_vs_t4": spec["speed_vs_t4"],
            "note": spec["note"],
            "est_seconds_per_epoch": round(gpu_seconds_per_epoch, 4),
            "est_full_29650ep_hours": round(full_gpu_hours, 2),
            "est_full_29650ep_usd": round(full_gpu_hours * float(spec["usd_per_hr"]), 2),
            "est_reduced_3000ep_hours": round(reduced_gpu_hours, 2),
            "est_reduced_3000ep_usd": round(reduced_gpu_hours * float(spec["usd_per_hr"]), 2),
        }

    return {
        "label": label,
        "axis_tag": _ADVISORY_TAG,
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "score_aware_included": bool(score_aware_included),
        "seconds_per_epoch_mlx_local": round(seconds_per_epoch, 4),
        "local_mlx_full_29650ep_hours": round(full_hours, 2),
        "local_mlx_full_29650ep_usd": 0.0,
        "local_mlx_reduced_3000ep_hours": round(reduced_hours, 2),
        "local_mlx_reduced_3000ep_usd": 0.0,
        "paid_gpu_bracket_advisory": paid_estimates,
        "paid_bracket_caveat": (
            "Paid estimates are a COARSE research-signal bracket derived from a "
            f"published MLX<->T4 throughput anchor ({MLX_LOCAL_VS_T4_THROUGHPUT_ANCHOR}x) "
            "+ the CLAUDE.md GPU rate table. The authoritative paid per-epoch "
            "cost is the B1 timing smoke on the actual GPU, NEVER this tool."
        ),
    }


def _recommend_b1_path(*, projection: dict[str, object], local_hours_tolerance: float) -> dict[str, object]:
    """Recommend local-MLX vs paid-campaign for B1 from the measured hours."""

    full_local_hours = float(projection["local_mlx_full_29650ep_hours"])
    reduced_local_hours = float(projection["local_mlx_reduced_3000ep_hours"])
    vast = projection["paid_gpu_bracket_advisory"]["vast_rtx_4090"]  # type: ignore[index]

    if full_local_hours <= local_hours_tolerance:
        recommendation = "local_mlx_full_curriculum"
        rationale = (
            f"Full 29650-ep curriculum projects to ~{full_local_hours:.1f} local "
            f"MLX hours (<= tolerance {local_hours_tolerance:.0f}h) at $0. Run B1 "
            "locally; no paid dispatch needed."
        )
    elif reduced_local_hours <= local_hours_tolerance:
        recommendation = "local_mlx_reduced_then_paid_full_if_promising"
        rationale = (
            f"Full curriculum (~{full_local_hours:.1f}h) exceeds the local "
            f"tolerance ({local_hours_tolerance:.0f}h), but the reduced "
            f"{PR95_PRAGMATIC_REDUCED_EPOCHS}-ep schedule projects to "
            f"~{reduced_local_hours:.1f}h at $0. Run the reduced schedule "
            "locally as the B1 MVP; escalate to a paid full-curriculum campaign "
            "(lane-claim + Vast.ai RTX 4090 ~"
            f"${vast['est_full_29650ep_usd']}) only if the reduced run is "  # type: ignore[index]
            "promising on exact-eval at B2."
        )
    else:
        recommendation = "paid_campaign_required"
        rationale = (
            f"Even the reduced {PR95_PRAGMATIC_REDUCED_EPOCHS}-ep schedule "
            f"projects to ~{reduced_local_hours:.1f}h locally (> tolerance "
            f"{local_hours_tolerance:.0f}h). B1 should be a lane-claimed paid "
            f"campaign: Vast.ai RTX 4090 full-curriculum est ~"
            f"${vast['est_full_29650ep_usd']} / ~{vast['est_full_29650ep_hours']}h "  # type: ignore[index]
            "(measure the real per-epoch via a paid timing smoke first)."
        )

    return {
        "recommendation": recommendation,
        "rationale": rationale,
        "local_hours_tolerance": local_hours_tolerance,
        "axis_tag": _ADVISORY_TAG,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    from tac.substrates.hi_nerv.architecture import HinervConfig

    cfg = HinervConfig()  # PR95-family default (decoder_channels (48,40,32,24,20,16,12))

    started = _utc_now_iso()
    param_counts = _param_counts(cfg)

    decode_t0 = time.perf_counter()
    target_rgb_0, target_rgb_1 = _decode_real_pairs(
        num_pairs=int(args.timing_pairs),
        height=int(cfg.output_height),
        width=int(cfg.output_width),
    )
    decode_t1 = time.perf_counter()

    # Surface A — primary, robust.
    timing_a = _time_renderer_fwd_bwd(
        cfg=cfg,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        batch_pairs=int(args.batch_pairs),
        epochs=int(args.epochs),
        warmup_steps=int(args.warmup_steps),
    )

    # Surface B — faithful score-aware, best-effort (skippable for speed).
    timing_b: StepTiming | None = None
    surface_b_diag: dict[str, object] = {"attempted": False}
    if not args.skip_score_aware:
        timing_b, surface_b_diag = _time_score_aware_train_step(
            cfg=cfg,
            target_rgb_0=target_rgb_0,
            target_rgb_1=target_rgb_1,
            batch_pairs=int(args.batch_pairs),
            # score-aware is heavier; cap epochs for the smoke unless overridden
            epochs=int(args.score_aware_epochs),
            warmup_steps=int(args.warmup_steps),
        )

    # The authoritative per-epoch number for the projection: prefer the faithful
    # score-aware surface when it succeeded; otherwise Surface A (lower bound).
    primary = (
        timing_b if (timing_b is not None and timing_b.steps_timed > 0) else timing_a
    )

    spe = primary.seconds_per_epoch(CONTEST_FULL_PAIRS)
    projection = _projection_block(
        seconds_per_epoch=spe,
        label=f"projection_from::{primary.surface}",
        score_aware_included=primary.score_aware_included,
    )
    recommendation = _recommend_b1_path(
        projection=projection,
        local_hours_tolerance=float(args.local_hours_tolerance),
    )

    def _timing_to_dict(t: StepTiming | None) -> dict[str, object] | None:
        if t is None:
            return None
        return {
            "surface": t.surface,
            "batch_pairs": t.batch_pairs,
            "warmup_steps": t.warmup_steps,
            "steps_timed": t.steps_timed,
            "median_step_seconds": round(t.median_step_seconds(), 4),
            "mean_step_seconds": round(t.mean_step_seconds(), 4),
            "steps_per_epoch_full_600pairs": t.steps_per_epoch(CONTEST_FULL_PAIRS),
            "seconds_per_epoch_full_600pairs": round(t.seconds_per_epoch(CONTEST_FULL_PAIRS), 4),
            "per_step_seconds_samples": [round(s, 5) for s in t.per_step_seconds],
            "score_aware_included": t.score_aware_included,
            "notes": t.notes,
        }

    payload: dict[str, object] = {
        "schema": "timing_smoke_hinerv_pr95_family.v1",
        "axis_tag": _ADVISORY_TAG,
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "phase": "B0_timing_smoke",
        "purpose": (
            "Measure seconds/epoch for the REAL PR95-family HiNeRV architecture "
            "on local MLX so the B1 600-pair 29650-epoch curriculum cost is "
            "MEASURED hours, not a guess (CLAUDE.md MVP-first + Long-burn "
            "campaign default). TIMING ONLY — never a score claim."
        ),
        "started_utc": started,
        "finished_utc": _utc_now_iso(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or platform.machine(),
            "python": sys.version.split()[0],
        },
        "config": {
            "decoder_channels": list(cfg.decoder_channels),
            "num_upsample_blocks": cfg.num_upsample_blocks,
            "latent_dims": [cfg.latent_dim_coarse, cfg.latent_dim_mid, cfg.latent_dim_fine],
            "embed_dim": cfg.embed_dim,
            "output_hw": [cfg.output_height, cfg.output_width],
            "num_pairs_full": cfg.num_pairs,
            "use_hierarchical_feature_grid": cfg.use_hierarchical_feature_grid,
            "use_convnext_blocks": cfg.use_convnext_blocks,
            "sin_frequency": cfg.sin_frequency,
        },
        "param_counts": param_counts,
        "param_count_note": (
            "Default HinervConfig yields "
            f"{param_counts['mlx_num_parameters']} params, ABOVE the architecture "
            "docstring's '~240K target' — the (48,40,32,24,20,16,12) 7-block "
            "PixelShuffle decoder is larger than the SKETCH comment implies. "
            "Still PR95-family-class (CLAUDE.md L18/L19 structure), reviewable."
        ),
        "real_pairs": {
            "video": "upstream/videos/0.mkv",
            "timing_pairs_decoded": int(args.timing_pairs),
            "decode_seconds": round(decode_t1 - decode_t0, 4),
            "synthetic_data_used": False,
        },
        "smoke_params": {
            "epochs": int(args.epochs),
            "score_aware_epochs": int(args.score_aware_epochs),
            "batch_pairs": int(args.batch_pairs),
            "warmup_steps": int(args.warmup_steps),
            "first_epoch_dropped_as_warmup": int(args.warmup_steps) > 0,
        },
        "timing_surface_a_renderer_fwd_bwd": _timing_to_dict(timing_a),
        "timing_surface_b_score_aware": _timing_to_dict(timing_b),
        "surface_b_diagnostics": surface_b_diag,
        "primary_surface_for_projection": primary.surface,
        "projection": projection,
        "b1_recommendation": recommendation,
        "constants": {
            "pr95_full_curriculum_epochs": PR95_FULL_CURRICULUM_EPOCHS,
            "pr95_pragmatic_reduced_epochs": PR95_PRAGMATIC_REDUCED_EPOCHS,
            "contest_full_pairs": CONTEST_FULL_PAIRS,
            "mlx_local_vs_t4_throughput_anchor": MLX_LOCAL_VS_T4_THROUGHPUT_ANCHOR,
        },
        "stop_after_timing": True,
        "stop_rationale": (
            "B0 deliverable is the timing that GATES the B1 decision. Per the "
            "mission, do NOT start B1 training; the recommendation field routes "
            "the B1 launch decision."
        ),
    }
    return payload


def _default_output_path() -> Path:
    return _REPO_ROOT / ".omx" / "research" / f"timing_smoke_hinerv_pr95_family_{_utc_now_compact()}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase B0 timing smoke for the PR95-family HiNeRV architecture on "
            "local MLX (research-signal only; never a score claim)."
        )
    )
    parser.add_argument(
        "--timing-pairs",
        type=int,
        default=16,
        help="Number of REAL contest pairs to decode for the timing smoke (default 16).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Surface-A epochs to time (passes over the decoded pairs; default 4).",
    )
    parser.add_argument(
        "--score-aware-epochs",
        type=int,
        default=2,
        help="Surface-B (score-aware) epochs to time; heavier, so fewer (default 2).",
    )
    parser.add_argument(
        "--batch-pairs",
        type=int,
        default=16,
        help="Pairs per training step (default 16).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2,
        help="Warmup steps dropped before timing (default 2; drops first-epoch warmup).",
    )
    parser.add_argument(
        "--skip-score-aware",
        action="store_true",
        help="Skip Surface B (score-aware adapter) entirely; report Surface A only.",
    )
    parser.add_argument(
        "--local-hours-tolerance",
        type=float,
        default=12.0,
        help="Max local-MLX hours considered tolerable for a local B1 run (default 12).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default .omx/research/timing_smoke_hinerv_pr95_family_<utc>.json).",
    )
    args = parser.parse_args(argv)

    payload = run(args)

    out_path = Path(args.output) if args.output else _default_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Human-readable console summary (the operator-facing readback).
    pc = payload["param_counts"]
    proj = payload["projection"]
    rec = payload["b1_recommendation"]
    print(f"[B0] {_ADVISORY_TAG} PR95-family HiNeRV timing smoke")
    print(f"  params: torch={pc['torch_num_parameters']} mlx={pc['mlx_num_parameters']} match={pc['torch_mlx_match']}")
    print(f"  primary surface: {payload['primary_surface_for_projection']} (score_aware_included={proj['score_aware_included']})")
    print(f"  seconds/epoch (MLX-local, 600 pairs): {proj['seconds_per_epoch_mlx_local']}")
    print(f"  full 29650-ep local hours: {proj['local_mlx_full_29650ep_hours']}  ($0)")
    print(f"  reduced 3000-ep local hours: {proj['local_mlx_reduced_3000ep_hours']}  ($0)")
    vast = proj["paid_gpu_bracket_advisory"]["vast_rtx_4090"]
    print(f"  paid bracket (Vast 4090, advisory): full ~${vast['est_full_29650ep_usd']} / ~{vast['est_full_29650ep_hours']}h")
    print(f"  B1 recommendation: {rec['recommendation']}")
    print(f"  -> {rec['rationale']}")
    sb = payload["surface_b_diagnostics"]
    if sb.get("attempted") and not sb.get("succeeded"):
        print(f"  surface-B (score-aware) blocker: {sb.get('blocker')}")
    print(f"  wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
