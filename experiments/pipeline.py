#!/usr/bin/env python3
"""Canonical compression pipeline: video -> trained renderer -> optimized archive.

Production-grade pipeline for the comma.ai video compression challenge.
Takes a video and a trained renderer checkpoint as input, produces an
optimized submission archive through a multi-step coordinate descent on
the rate-distortion surface.

Pipeline steps per iteration:
    1. Export: float checkpoint -> quantized renderer.bin
    2. Pose TTO: adaptive gradient-based pose optimization (converges to tolerance)
    3. QAT: quantization-aware fine-tuning (monitors quality degradation)
    4. Fridrich refinement: steganalytic polish with texture/L-inf/Markov losses
    5. Weight compression: int4+LZMA2 or FP4 (configurable, auto mode available)
    6. Archive: build submission zip with masks + renderer + poses
    7. Eval: full end-to-end auth evaluation through upstream scorer

The pipeline is idempotent: each step writes a .done marker, and re-running
resumes from the last completed step. Iterative refinement continues until
score improvement falls below a convergence tolerance.

Usage:
    # Compress a single video:
    PYTHONPATH=src:upstream python experiments/pipeline.py compress \\
        --video upstream/videos/0.mkv \\
        --checkpoint path/to/distill_phase2_best.pt \\
        --device cuda --output-dir results/submission_v1

    # Compress with all optimizations:
    PYTHONPATH=src:upstream python experiments/pipeline.py compress \\
        --video upstream/videos/0.mkv \\
        --checkpoint path/to/distill_phase2_best.pt \\
        --device cuda --output-dir results/submission_v1 \\
        --half-frame --binary-poses --brotli --weight-compression auto

    # Evaluate an existing archive:
    PYTHONPATH=src:upstream python experiments/pipeline.py eval \\
        --archive results/submission_v1/archive.zip \\
        --video upstream/videos/0.mkv --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pydantic import BaseModel
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn


# ── Constants ────────────────────────────────────────────────────────────

ORIGINAL_VIDEO_BYTES = 37_545_489
NUM_FRAMES = 1200
SEGNET_H, SEGNET_W = 384, 512


# ── Logging ──────────────────────────────────────────────────────────────

def _log(msg: str, level: str = "pipeline") -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def _banner(msg: str) -> None:
    w = max(len(msg) + 4, 60)
    _log("=" * w)
    _log(f"  {msg}")
    _log("=" * w)


# ── Resume support ───────────────────────────────────────────────────────

def _step_done(output_dir: Path, step: str) -> bool:
    return (output_dir / f".done_{step}").exists()


def _mark_done(output_dir: Path, step: str, meta: dict | None = None) -> None:
    """Atomically write a .done_<step> marker.

    R29 fix: write to a tempfile then os.replace — guarantees a concurrent
    reader sees either the OLD marker or the COMPLETE NEW marker, never a
    partial JSON. Path.write_text is non-atomic on POSIX (open+write+close);
    a concurrent process could read mid-write and treat partial JSON as a
    complete .done marker.
    """
    info = {"step": step, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if meta:
        info.update(meta)
    target = output_dir / f".done_{step}"
    tmp = output_dir / f".done_{step}.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(info, indent=2))
    os.replace(tmp, target)


# ── Pipeline configuration ──────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Production pipeline configuration.

    Architecture defaults match ``tac.renderer.ArchConfig`` (the single source
    of truth).  ``padding_mode`` is overridden to "replicate" because Yousfi's
    analysis showed that zeros creates boundary artifacts.
    """

    # Input
    video: str = ""
    checkpoint: str = ""
    masks: str = ""  # FULL masks (1200 frames) for training/pose TTO. Extracted from video if empty.
    masks_archive: str = ""  # HALF-FRAME masks (600 frames) for archive. Built from full masks if empty.
    output_dir: str = "experiments/results/pipeline"
    device: str = "cuda"
    upstream: str = "upstream"

    # Architecture — defaults from ArchConfig (see tac.renderer.ArchConfig)
    base_ch: int = 36       # ArchConfig default: 36  (was 20 — BUG: diverged from DistillConfig)
    mid_ch: int = 60        # ArchConfig default: 60  (was 28 — BUG: diverged from DistillConfig)
    motion_hidden: int = 32  # ArchConfig default: 32
    depth: int = 1          # ArchConfig default: 1
    pose_dim: int = 6       # ArchConfig default: 6
    embed_dim: int = 6      # ArchConfig default: 6
    use_dsconv: bool = False  # ArchConfig default: False (was True — BUG: diverged from DistillConfig)
    padding_mode: str = "replicate"  # ArchConfig default: "zeros" — override: boundary artifact avoidance
    use_dilation: bool = False  # ArchConfig default: False
    use_zoom_flow: bool = False  # GREEN profile: MotionPredictor 4ch (gate+residual), flow from RadialZoomWarp

    # Training discipline / loss modulators (passed through to train_distill.py)
    use_swa: bool = False
    use_per_class_weights: bool = False
    freeze_motion_phase2: bool = False
    freeze_renderer_phase3: bool = False
    beneficial_quant_noise: bool = False

    # Pose TTO — adaptive convergence
    pose_max_steps: int = 2000        # upper bound
    pose_min_steps: int = 200         # minimum before checking convergence
    pose_lr: float = 0.01
    pose_convergence_tol: float = 1e-4  # stop when improvement < tol for patience steps
    pose_patience: int = 50
    pose_batch_pairs: int = 16  # 50 OOMs on 4090 with both scorers loaded

    # QAT — adaptive quality monitoring
    qat_max_epochs: int = 500         # upper bound
    qat_min_epochs: int = 100         # minimum
    qat_lr: float = 5e-5
    qat_quality_threshold: float = 1.5  # stop if FP4/float quality ratio exceeds this

    # Archive
    half_frame: bool = False
    binary_poses: bool = False
    brotli: bool = False
    mask_crf: int = 50

    # Weight compression: "fp4" (current FP4+codebook), "int4_lzma2" (int4+LZMA2),
    # or "auto" (try int4_lzma2 first, fall back to fp4 if quality degrades too much)
    weight_compression: str = "fp4"

    # Iterative refinement — convergence-driven, not arbitrary count
    max_iterations: int = 10          # safety bound (convergence stops earlier)
    convergence_tol: float = 0.01     # stop when score improvement < tol between cycles
    # The pipeline is a coordinate descent on the rate-distortion surface:
    # each cycle optimizes poses (distortion) then quantizes (rate).
    # Convergence is reached when marginal improvement falls below the
    # marginal rate cost of running another cycle.


# ── Step implementations ─────────────────────────────────────────────────

def step_extract_masks(cfg: PipelineConfig) -> tuple[Path, Path]:
    """Extract SegNet masks — FULL (1200) for training/TTO, HALF (600) for archive.

    Training and pose TTO need full-frame masks (both frame_t and frame_t1).
    The archive only needs half-frame masks (frame_t1 only, frame_t derived via warp).
    These are DIFFERENT requirements — conflating them caused a batch 38/75 crash
    when pose TTO tried to make 600 pairs from 600 half-frame masks.
    """
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Full masks (1200 frames) — for training and pose TTO
    if cfg.masks and Path(cfg.masks).exists():
        full_masks_path = Path(cfg.masks)
        _log(f"Using pre-extracted full masks: {full_masks_path}")
    elif _step_done(out, "masks_full"):
        full_masks_path = out / "masks_full.mkv"
        _log("Full mask extraction already done, skipping")
    else:
        _log("Extracting FULL SegNet masks (1200 frames) from video...")
        from tac.data import decode_video
        from tac.scorer import load_scorers
        from tac.mask_codec import extract_masks, encode_masks

        gt_frames = decode_video(cfg.video, target_h=SEGNET_H, target_w=SEGNET_W)[:NUM_FRAMES]
        _log(f"  Decoded {len(gt_frames)} frames")

        models_dir = Path(cfg.upstream) / "models"
        _, segnet = load_scorers(
            str(models_dir / "posenet.safetensors"),
            str(models_dir / "segnet.safetensors"),
            device=cfg.device,
        )
        masks = extract_masks(gt_frames, segnet, device=cfg.device)
        full_masks_path = out / "masks_full.mkv"
        nbytes = encode_masks(masks, full_masks_path, crf=cfg.mask_crf)
        _log(f"  Full masks: {masks.shape[0]} frames, {nbytes:,} bytes")
        _mark_done(out, "masks_full", {"n_masks": masks.shape[0], "bytes": nbytes})

    # Half-frame masks (600 frames) — for archive only
    if cfg.masks_archive and Path(cfg.masks_archive).exists():
        half_masks_path = Path(cfg.masks_archive)
        _log(f"Using pre-extracted half-frame masks: {half_masks_path}")
    elif cfg.half_frame:
        if _step_done(out, "masks_half"):
            half_masks_path = out / "masks_half.mkv"
        else:
            _log("Building half-frame masks from full masks...")
            from tac.mask_codec import decode_masks, encode_masks
            full = decode_masks(str(full_masks_path))
            half = full[1::2]  # odd frames only
            half_masks_path = out / "masks_half.mkv"
            nbytes = encode_masks(half, half_masks_path, crf=cfg.mask_crf)
            _log(f"  Half masks: {half.shape[0]} frames, {nbytes:,} bytes")
            _mark_done(out, "masks_half", {"n_masks": half.shape[0], "bytes": nbytes})
    else:
        half_masks_path = full_masks_path  # no half-frame, use full for archive too

    return full_masks_path, half_masks_path


def step_export(cfg: PipelineConfig, iteration: int = 0) -> Path:
    """Export float checkpoint to FP4 renderer.bin."""
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    if _step_done(iter_dir, "export"):
        _log(f"Export already done (iter {iteration}), skipping")
        return iter_dir / "renderer.bin"

    _log("Exporting checkpoint to FP4")
    from tac.renderer import AsymmetricPairGenerator
    from tac.renderer_export import export_asymmetric_checkpoint_fp4

    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)

    model = AsymmetricPairGenerator(
        num_classes=5, embed_dim=cfg.embed_dim,
        base_ch=cfg.base_ch, mid_ch=cfg.mid_ch,
        motion_hidden=cfg.motion_hidden, depth=cfg.depth,
        pose_dim=cfg.pose_dim, use_dsconv=cfg.use_dsconv,
        padding_mode=cfg.padding_mode, use_dilation=cfg.use_dilation,
        use_zoom_flow=cfg.use_zoom_flow,
    )
    # strict=False so we get the missing/unexpected lists; we then raise
    # if either is non-empty — equivalent to strict=True but with a much
    # better error message. (R24 finding: silent strict=False permitted
    # use_zoom_flow channel-count mismatch.)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint shape mismatch — refuse to export wrong arch. "
            f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}. "
            f"Verify use_zoom_flow={cfg.use_zoom_flow}, base_ch={cfg.base_ch}, "
            f"mid_ch={cfg.mid_ch}, motion_hidden={cfg.motion_hidden} match training."
        )
    n_params = sum(p.numel() for p in model.parameters())

    renderer_bin = iter_dir / "renderer.bin"
    nbytes = export_asymmetric_checkpoint_fp4(model, str(renderer_bin))
    _log(f"  {n_params:,} params → {nbytes:,} bytes ({nbytes/1024:.1f} KB)")

    _mark_done(iter_dir, "export", {"params": n_params, "bytes": nbytes})
    return renderer_bin


def _export_refined_checkpoint(cfg: PipelineConfig, ckpt_path: Path,
                                iteration: int) -> Path:
    """Export a refined (Fridrich/QAT) checkpoint to FP4 .bin.

    R32 fix: split out from step_export because step_export is gated by
    .done_export which is set with the ORIGINAL cfg.checkpoint at iteration
    start. Re-calling step_export would silently return stale weights and
    discard whatever refinement just happened. This helper writes to a
    distinct filename so the .done_export gate cannot collide.
    """
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    out_path = iter_dir / "renderer_refined.bin"

    # R33 fix: idempotence guard. Match every other step's resume contract
    # so a restart doesn't re-do the load+export. The output is deterministic
    # given (cfg arch, ckpt_path) so reusing the existing file is safe.
    if out_path.exists() and out_path.stat().st_size > 0:
        _log(f"  Refined export already exists, skipping ({out_path.name})")
        return out_path

    from tac.renderer import AsymmetricPairGenerator
    from tac.renderer_export import export_asymmetric_checkpoint_fp4

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model = AsymmetricPairGenerator(
        num_classes=5, embed_dim=cfg.embed_dim,
        base_ch=cfg.base_ch, mid_ch=cfg.mid_ch,
        motion_hidden=cfg.motion_hidden, depth=cfg.depth,
        pose_dim=cfg.pose_dim, use_dsconv=cfg.use_dsconv,
        padding_mode=cfg.padding_mode, use_dilation=cfg.use_dilation,
        use_zoom_flow=cfg.use_zoom_flow,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"_export_refined_checkpoint: shape mismatch on {ckpt_path}. "
            f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}. "
            f"Verify arch (use_zoom_flow={cfg.use_zoom_flow}, base_ch={cfg.base_ch}) "
            f"matches the refined checkpoint."
        )
    # R34 fix: atomic write via tmp + os.replace. Without this, a concurrent
    # invocation (or interrupted run + immediate resume) could observe a
    # partial file via the idempotence guard's `out_path.stat().st_size > 0`
    # check and ship truncated weights downstream.
    tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}.bin")
    try:
        nbytes = export_asymmetric_checkpoint_fp4(model, str(tmp_path))
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    _log(f"  Refined export → {out_path.name} ({nbytes:,}B = {nbytes/1024:.1f}KB)")
    return out_path


def step_pose_tto(cfg: PipelineConfig, iteration: int = 0) -> Path:
    """Adaptive pose TTO with convergence detection."""
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    # R29 fix: validate masks BEFORE constructing the subprocess cmd. Prior
    # version passed cfg.masks blindly, so an empty string flowed through
    # to optimize_poses.py and produced a cryptic decode failure deep in the
    # subprocess instead of a clear "masks not set" error here.
    if not cfg.masks:
        raise RuntimeError(
            f"step_pose_tto: cfg.masks is empty. step_extract_masks must run "
            f"first AND populate cfg.masks. iter={iteration} output_dir={cfg.output_dir}"
        )
    if not Path(cfg.masks).exists():
        raise RuntimeError(
            f"step_pose_tto: masks file does not exist: {cfg.masks!r}. "
            f"iter={iteration} output_dir={cfg.output_dir}"
        )

    if _step_done(iter_dir, "pose_tto"):
        _log(f"Pose TTO already done (iter {iteration}), skipping")
        for suffix in [".bin", ".pt"]:
            p = iter_dir / f"optimized_poses{suffix}"
            if p.exists():
                return p
        return iter_dir / "optimized_poses.pt"

    _log(f"Pose TTO (up to {cfg.pose_max_steps} steps, converge at tol={cfg.pose_convergence_tol})")
    cmd = [
        sys.executable, "-u", "experiments/optimize_poses.py",
        "--checkpoint", cfg.checkpoint,
        "--masks", cfg.masks,
        "--device", cfg.device,
        "--steps", str(cfg.pose_max_steps),
        "--lr", str(cfg.pose_lr),
        "--batch-pairs", str(cfg.pose_batch_pairs),
        "--eval-roundtrip",
        "--output-dir", str(iter_dir),
    ]

    # Reuse cached pose targets from previous runs (saves ~15 min of PoseNet inference)
    for candidate in [iter_dir / "gt_pose_targets.pt", Path(cfg.output_dir) / "gt_pose_targets.pt"]:
        if candidate.exists():
            cmd.extend(["--gt-pose-targets", str(candidate)])
            _log(f"  Using cached pose targets: {candidate}")
            break

    t0 = time.monotonic()
    # R30: 4-hour timeout (pose TTO is the longest-running subprocess; A100
    # 4090 typical wall is 30-90 min). A hung scorer or DataLoader would
    # otherwise block the pipeline forever.
    try:
        result = subprocess.run(cmd, timeout=14400)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Pose TTO timed out after 4h — likely hung GPU/DataLoader")
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        raise RuntimeError(f"Pose TTO failed (exit {result.returncode})")

    _log(f"  Complete in {elapsed/60:.1f} min")

    poses_path = iter_dir / "optimized_poses.bin"
    if not poses_path.exists():
        poses_path = iter_dir / "optimized_poses.pt"

    _mark_done(iter_dir, "pose_tto", {"elapsed_s": round(elapsed)})
    return poses_path


def step_sensitivity_sweep(cfg: PipelineConfig, iteration: int = 0) -> Path | None:
    """Run scorer-Jacobian sensitivity sweep to determine per-layer bit allocation.

    Measures how quantization noise in each renderer layer propagates through
    the actual SegNet + PoseNet scorers. Layers that are scorer-insensitive
    get fewer bits (2), scorer-sensitive layers get more bits (8), and the
    rest stay at 4. The output JSON is consumed by the QAT step and the
    weight compression step for mixed-precision export.

    Returns path to the sensitivity sweep JSON, or None if skipped.
    """
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    step_name = "sensitivity_sweep"

    sweep_json = iter_dir / "scorer_sensitivity_sweep.json"

    if _step_done(iter_dir, step_name):
        _log(f"Sensitivity sweep already done (iter {iteration}), skipping")
        return sweep_json if sweep_json.exists() else None

    _log("Scorer-Jacobian sensitivity sweep (Yousfi approach)")
    cmd = [
        sys.executable, "-u", "experiments/scorer_sensitivity_sweep.py",
        "--checkpoint", cfg.checkpoint,
        "--device", cfg.device,
        "--n-pairs", "5",
        "--output", str(sweep_json),
    ]

    t0 = time.monotonic()
    # R30: 1-hour timeout for sensitivity sweep (Yousfi Jacobian; 5 pairs
    # should be 5-15 min on A100). Hung Jacobian must not block the pipeline.
    try:
        result = subprocess.run(cmd, timeout=3600)
    except subprocess.TimeoutExpired:
        _log("Sensitivity sweep timed out after 1h, continuing without", "WARN")
        _mark_done(iter_dir, step_name, {"failed": True, "timeout": True})
        return None
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        _log(f"Sensitivity sweep failed (exit {result.returncode}), continuing without", "WARN")
        _mark_done(iter_dir, step_name, {"failed": True, "elapsed_s": round(elapsed)})
        return None

    _log(f"  Complete in {elapsed/60:.1f} min")

    # Log the allocation summary
    if sweep_json.exists():
        try:
            data = json.loads(sweep_json.read_text())
            alloc = data.get("allocation", {})
            uniform_kb = data.get("uniform_kb", 0)
            mixed_kb = data.get("mixed_kb", 0)
            _log(f"  Allocation: {len(alloc)} layers, "
                 f"uniform={uniform_kb:.1f}KB -> mixed={mixed_kb:.1f}KB "
                 f"({(1 - mixed_kb / max(uniform_kb, 0.1)) * 100:.1f}% savings)")
        except (json.JSONDecodeError, KeyError):
            pass

    _mark_done(iter_dir, step_name, {"elapsed_s": round(elapsed)})
    return sweep_json


def step_qat(cfg: PipelineConfig, iteration: int = 0,
             mixed_precision_json: Path | None = None) -> Path:
    """QAT fine-tune with quality monitoring."""
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    if _step_done(iter_dir, "qat"):
        _log(f"QAT already done (iter {iteration}), skipping")
        # qat_finetune.py may save under either name depending on path; resolve
        # against what actually exists on disk so the caller never gets a
        # phantom path. (Round 23 finding.)
        for candidate in ("renderer_fp4.bin", "renderer_qat.bin"):
            p = iter_dir / candidate
            if p.exists():
                return p
        existing_bins = sorted(iter_dir.glob("*.bin"))
        if existing_bins:
            return existing_bins[-1]
        raise RuntimeError(
            f"QAT marked done in {iter_dir} but no .bin output found. "
            f"Delete .done_qat marker to force rerun."
        )

    _log(f"QAT fine-tune (up to {cfg.qat_max_epochs} epochs)")
    cmd = [
        sys.executable, "-u", "experiments/qat_finetune.py",
        "--checkpoint", cfg.checkpoint,
        "--upstream", cfg.upstream,
        "--device", cfg.device,
        "--fp4-epochs", str(cfg.qat_max_epochs),
        "--lr", str(cfg.qat_lr),
        "--output-dir", str(iter_dir),
        "--base-ch", str(cfg.base_ch),
        "--mid-ch", str(cfg.mid_ch),
        "--motion-hidden", str(cfg.motion_hidden),
        "--depth", str(cfg.depth),
        "--embed-dim", str(cfg.embed_dim),
        "--pose-dim", str(cfg.pose_dim),
        "--padding-mode", cfg.padding_mode,
    ]
    if cfg.use_dsconv:
        cmd.append("--use-dsconv")
    if cfg.use_dilation:
        cmd.append("--use-dilation")
    if cfg.use_zoom_flow:
        cmd.append("--use-zoom-flow")
    if mixed_precision_json is not None and mixed_precision_json.exists():
        cmd.extend(["--mixed-precision-json", str(mixed_precision_json)])
        _log(f"  Using mixed-precision allocation from: {mixed_precision_json}")

    t0 = time.monotonic()
    # R30: 6-hour timeout for QAT (250 FP4 epochs typical 1-3h on A100).
    try:
        result = subprocess.run(cmd, timeout=21600)
    except subprocess.TimeoutExpired:
        raise RuntimeError("QAT timed out after 6h — likely hung")
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        raise RuntimeError(f"QAT failed (exit {result.returncode})")

    _log(f"  Complete in {elapsed/60:.1f} min")

    # R34 fix: qat_finetune.py saves 'renderer_fp4.bin', NOT 'renderer_qat.bin'.
    # Prior code returned the wrong-name path and only the glob fallback
    # rescued at runtime. Worse, the alphabetical glob could return the
    # PRE-QAT 'renderer.bin' if 'renderer_fp4.bin' wasn't present, silently
    # using the un-quantized model downstream.
    qat_bin = iter_dir / "renderer_fp4.bin"
    if not qat_bin.exists():
        # Try the legacy/alternative names before falling back.
        for alt in ("renderer_qat.bin", "renderer_mixed.bin"):
            cand = iter_dir / alt
            if cand.exists():
                qat_bin = cand
                break
        else:
            raise RuntimeError(
                f"QAT step succeeded but no QAT artifact found in {iter_dir}. "
                f"Expected renderer_fp4.bin (or renderer_qat.bin / renderer_mixed.bin). "
                f"Refusing to fall back to pre-QAT renderer.bin (would silently "
                f"ship un-quantized weights)."
            )

    _mark_done(iter_dir, "qat", {"elapsed_s": round(elapsed)})
    return qat_bin


def step_fridrich_refine(cfg: PipelineConfig, iteration: int = 0) -> Path:
    """Step 3.5: Fridrich steganalytic refinement (post-QAT polish).

    100 epochs at very low LR with ONLY Fridrich losses (texture + L-inf + Markov).
    Pushes pixel-level errors further into the scorer's null space without
    disturbing the QAT-optimized scorer distortion. Our competitive advantage.
    """
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    step_name = "fridrich_refine"

    def _resolve_fridrich_output() -> Path:
        """Find the best Fridrich refinement output, with explicit fallbacks.

        R31 fix: prior version returned a hardcoded `fridrich/distill_phase2_best.pt`
        path that may not exist if train_distill.py used a different name.
        Glob the fridrich subdir; on resume from failure, fall back to QAT.

        R32 fix: glob sort by mtime (newest first) — alphabetical sort could
        return a partial `*_tmp.pt` save instead of the latest complete one.
        Last-resort fallback raises RuntimeError instead of returning the
        original input — the pipeline made no progress, fail loud.
        """
        fridrich_dir = iter_dir / "fridrich"
        if fridrich_dir.exists():
            # R34: distill_best.pt removed — no producer ever writes it.
            # train_distill.py saves distill_phase2_best.pt and distill_latest.pt.
            for candidate in (fridrich_dir / "distill_phase2_best.pt",
                              fridrich_dir / "distill_latest.pt"):
                if candidate.exists():
                    return candidate
            # Sort by mtime (newest last); last item is most recent successful save.
            pts = sorted(fridrich_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
            # Skip obvious tempfiles in the name.
            pts = [p for p in pts if "tmp" not in p.name.lower()]
            if pts:
                return pts[-1]
        # No Fridrich output exists — fall back to the QAT pre-Fridrich checkpoint.
        for fb in (iter_dir / "qat_best_float.pt", iter_dir / "renderer_qat.bin",
                   iter_dir / "renderer_fp4.bin"):
            if fb.exists():
                return fb
        # Last resort: pipeline made NO progress. Fail loud per R32 finding.
        raise RuntimeError(
            f"_resolve_fridrich_output: no Fridrich, QAT-best, or FP4 output "
            f"found in {iter_dir}. Pipeline made no progress beyond input. "
            f"Delete .done_fridrich_refine and .done_qat to force rerun."
        )

    if _step_done(iter_dir, step_name):
        _log(f"Fridrich refinement already done (iter {iteration}), skipping")
        # Read the .done marker — if the previous run was a failure/timeout,
        # we should NOT pretend a fridrich output exists.
        try:
            done_meta = json.loads((iter_dir / f".done_{step_name}").read_text())
            if done_meta.get("failed") or done_meta.get("timeout") or done_meta.get("skipped"):
                _log(f"  Previous Fridrich was {done_meta} — using QAT fallback", "WARN")
        except (json.JSONDecodeError, FileNotFoundError):
            pass
        return _resolve_fridrich_output()

    # Find best QAT checkpoint to start from
    # qat_finetune.py saves: qat_best_float.pt, renderer_fp4.bin
    qat_ckpt = iter_dir / "qat_best_float.pt"
    if not qat_ckpt.exists():
        qat_ckpt = iter_dir / "distill_latest.pt"
    if not qat_ckpt.exists():
        _log("No QAT checkpoint found, skipping Fridrich refinement", "WARN")
        _mark_done(iter_dir, step_name, {"skipped": True})
        return _resolve_fridrich_output()

    _log("Fridrich steganalytic refinement (100 epochs, Fridrich-only)")
    cmd = [
        sys.executable, "-u", "experiments/train_distill.py",
        "--checkpoint", str(qat_ckpt),
        "--masks", cfg.masks,
        "--device", cfg.device,
        "--output-dir", str(iter_dir / "fridrich"),
        "--base-ch", str(cfg.base_ch), "--mid-ch", str(cfg.mid_ch),
        "--motion-hidden", str(cfg.motion_hidden),
        "--depth", str(cfg.depth), "--pose-dim", str(cfg.pose_dim),
        "--embed-dim", str(cfg.embed_dim),
        "--padding-mode", cfg.padding_mode,
        "--eval-roundtrip",
        # Zero scorer loss — ONLY Fridrich pixel-space losses
        "--seg-weight", "0.0", "--pose-weight", "0.0", "--pixel-weight", "0.0",
        "--use-texture-loss", "--texture-loss-weight", "1.0",
        "--use-linf-penalty", "--linf-weight", "0.1",
        "--use-markov-loss", "--markov-weight", "0.5",
        "--phase1-epochs", "0", "--phase2-epochs", "100", "--phase3-epochs", "0",
        "--phase2-lr", "1e-5",
        "--ema-decay", "0.997",
        "--checkpoint-every", "50", "--eval-every", "25", "--log-every", "10",
        "--upstream", cfg.upstream,
    ]
    if cfg.use_dsconv:
        cmd.append("--use-dsconv")
    if cfg.use_dilation:
        cmd.append("--use-dilation")
    if cfg.use_zoom_flow:
        cmd.append("--use-zoom-flow")
    # Pass through training-discipline flags so Fridrich refinement runs in
    # the same regime as the original training. Otherwise the refinement
    # diverges from the training distribution it's trying to polish.
    if cfg.use_swa:
        cmd.append("--use-swa")
    if cfg.use_per_class_weights:
        cmd.append("--use-per-class-weights")
    if cfg.freeze_motion_phase2:
        cmd.append("--freeze-motion-phase2")
    if cfg.freeze_renderer_phase3:
        cmd.append("--freeze-renderer-phase3")
    if cfg.beneficial_quant_noise:
        cmd.append("--beneficial-quant-noise")

    t0 = time.monotonic()
    # R30: 2-hour timeout for Fridrich refinement (100 epochs typical 30-60min).
    try:
        result = subprocess.run(cmd, timeout=7200)
    except subprocess.TimeoutExpired:
        _log("Fridrich refinement timed out after 2h, falling back to QAT output", "WARN")
        _mark_done(iter_dir, step_name, {"failed": True, "timeout": True})
        return _resolve_fridrich_output()
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        _log(f"Fridrich refinement failed (exit {result.returncode})", "WARN")
        _mark_done(iter_dir, step_name, {"failed": True})
        return _resolve_fridrich_output()

    _log(f"  Fridrich refinement complete in {elapsed/60:.1f} min")
    _mark_done(iter_dir, step_name, {"elapsed_s": round(elapsed)})
    # R31: glob for the actual best checkpoint instead of returning a hardcoded
    # path that may not exist if train_distill.py used a different filename.
    return _resolve_fridrich_output()


def step_compress_weights(
    cfg: PipelineConfig,
    checkpoint_path: Path,
    iteration: int = 0,
    sensitivity_json: Path | None = None,
) -> Path:
    """Compress model weights using mixed-precision quantization + LZMA2.

    Three compression strategies (choose best for this checkpoint):
    1. Int4 per-tensor + LZMA2 (simplest, ~2.2 bits/weight)
    2. LearnableBitDepth fine-tune + LZMA2 (~1.5-2.0 bits/weight, future)
    3. FP4 + Brotli (current, ~4.4 bits/weight, fallback)

    When cfg.weight_compression == "auto", tries int4_lzma2 first.
    Measures quality degradation on a small forward pass. Falls back
    to fp4 if output divergence exceeds an acceptable threshold.
    """
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    step_name = "compress_weights"

    if _step_done(iter_dir, step_name):
        _log(f"Weight compression already done (iter {iteration}), skipping")
        # Return whichever format was produced
        for suffix in ["_int4lzma2.bin", ".bin"]:
            p = iter_dir / f"renderer{suffix}"
            if p.exists():
                return p
        return checkpoint_path

    mode = cfg.weight_compression
    if mode == "fp4":
        _log("Weight compression: FP4 (default, no additional compression)")
        _mark_done(iter_dir, step_name, {"mode": "fp4", "skipped": True})
        return checkpoint_path

    # I4LZ format has no arch header — inflate cannot recover padding_mode,
    # use_dilation, or use_zoom_flow from the compressed state dict. Forcing
    # I4LZ on a model trained with non-default arch flags would silently use
    # zeros/False at inflate → wrong reconstruction. (R24 finding: SHIRAZ-class
    # bug, but at inflate time instead of QAT time.)
    needs_arch_header = (
        cfg.padding_mode != "zeros"
        or cfg.use_dilation
        or cfg.use_zoom_flow
    )
    if mode in ("int4_lzma2", "auto") and needs_arch_header:
        _log(
            f"Weight compression: I4LZ format unsafe for padding_mode={cfg.padding_mode!r}, "
            f"use_dilation={cfg.use_dilation}, use_zoom_flow={cfg.use_zoom_flow}. "
            f"I4LZ has no arch header → inflate would silently use defaults. "
            f"Falling back to FP4 (which embeds the full arch header).",
            "WARN",
        )
        _mark_done(iter_dir, step_name, {"mode": "fp4_fallback",
                                          "reason": "i4lz_no_arch_header"})
        return checkpoint_path

    _log(f"Weight compression: {mode}")

    from tac.renderer import AsymmetricPairGenerator
    from tac.mixed_precision_export import export_int4_lzma2, load_int4_lzma2

    # Load the checkpoint into a model for re-export
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)

    model = AsymmetricPairGenerator(
        num_classes=5, embed_dim=cfg.embed_dim,
        base_ch=cfg.base_ch, mid_ch=cfg.mid_ch,
        motion_hidden=cfg.motion_hidden, depth=cfg.depth,
        pose_dim=cfg.pose_dim, use_dsconv=cfg.use_dsconv,
        padding_mode=cfg.padding_mode, use_dilation=cfg.use_dilation,
        use_zoom_flow=cfg.use_zoom_flow,
    )
    # R32 fix: same hard-fail-on-shape-mismatch pattern as step_export.
    # Silent strict=False permits use_zoom_flow channel-count mismatches.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"step_compress_weights: shape mismatch on {checkpoint_path}. "
            f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}. "
            f"Verify arch (use_zoom_flow={cfg.use_zoom_flow}, base_ch={cfg.base_ch}) "
            f"matches the checkpoint."
        )
    model.eval()

    int4_path = iter_dir / "renderer_int4lzma2.bin"
    fp4_path = iter_dir / "renderer.bin"

    # Load per-layer bit allocation from sensitivity sweep (if available)
    bit_allocation: dict[str, int] | None = None
    if sensitivity_json is not None and sensitivity_json.exists():
        try:
            sweep_data = json.loads(sensitivity_json.read_text())
            bit_allocation = sweep_data.get("allocation")
            if bit_allocation:
                _log(f"  Using mixed-precision allocation from: {sensitivity_json}")
                bit_counts = {}
                for b in bit_allocation.values():
                    bit_counts[b] = bit_counts.get(b, 0) + 1
                _log(f"  Allocation: {', '.join(f'{b}b:{n}' for b, n in sorted(bit_counts.items()))}")
        except (json.JSONDecodeError, KeyError) as e:
            _log(f"  Failed to load sensitivity sweep: {e}", "WARN")

    # ── Strategy 1: Int4/Mixed + LZMA2 ──
    int4_bytes = export_int4_lzma2(model, int4_path, bit_allocation=bit_allocation)
    label = "Mixed" if bit_allocation else "Int4"
    _log(f"  {label}+LZMA2: {int4_bytes:,} bytes ({int4_bytes/1024:.1f} KB)")

    if mode == "int4_lzma2":
        # Unconditionally use int4_lzma2
        _mark_done(iter_dir, step_name, {"mode": "int4_lzma2", "bytes": int4_bytes})
        return int4_path

    # ── mode == "auto": compare quality ──
    # Quick quality check: load int4 weights, compare forward pass
    restored_sd = load_int4_lzma2(int4_path)
    model_restored = AsymmetricPairGenerator(
        num_classes=5, embed_dim=cfg.embed_dim,
        base_ch=cfg.base_ch, mid_ch=cfg.mid_ch,
        motion_hidden=cfg.motion_hidden, depth=cfg.depth,
        pose_dim=cfg.pose_dim, use_dsconv=cfg.use_dsconv,
        padding_mode=cfg.padding_mode, use_dilation=cfg.use_dilation,
        use_zoom_flow=cfg.use_zoom_flow,
    )
    # R33 fix: same hard-fail-on-shape-mismatch pattern as step_export.
    # Without this, a corrupt/truncated I4LZ file silently restores partial
    # weights and the RMSE quality measurement compares against zeros for
    # missing layers — producing artificially low RMSE that would make the
    # 'auto' mode incorrectly choose I4LZ over FP4.
    missing_r, unexpected_r = model_restored.load_state_dict(restored_sd, strict=False)
    if missing_r or unexpected_r:
        raise RuntimeError(
            f"step_compress_weights: I4LZ restore shape mismatch. "
            f"missing={list(missing_r)[:5]} unexpected={list(unexpected_r)[:5]}. "
            f"I4LZ archive may be truncated/corrupt."
        )
    model_restored.eval()

    # Measure parameter-space RMSE
    total_se = 0.0
    total_n = 0
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model_restored.named_parameters()
    ):
        se = ((p1.detach() - p2.detach()) ** 2).sum().item()
        total_se += se
        total_n += p1.numel()
    rmse = (total_se / max(total_n, 1)) ** 0.5

    _log(f"  Int4+LZMA2 weight RMSE: {rmse:.6f}")

    # Also compare FP4 size for the auto decision
    from tac.renderer_export import export_asymmetric_checkpoint_fp4
    fp4_bytes = export_asymmetric_checkpoint_fp4(model, str(fp4_path))
    _log(f"  FP4: {fp4_bytes:,} bytes ({fp4_bytes/1024:.1f} KB)")

    # Auto decision: use int4_lzma2 if it's smaller (it almost always will be)
    # AND weight RMSE is acceptable (< 0.05, which is well within FP4 noise)
    if int4_bytes < fp4_bytes and rmse < 0.05:
        _log(f"  Auto: chose int4_lzma2 (smaller by {fp4_bytes - int4_bytes:,} bytes)")
        chosen = "int4_lzma2"
        result_path = int4_path
    else:
        _log(f"  Auto: chose fp4 (int4 RMSE={rmse:.4f} or size not better)")
        chosen = "fp4"
        result_path = fp4_path

    _mark_done(iter_dir, step_name, {
        "mode": chosen,
        "int4_bytes": int4_bytes,
        "fp4_bytes": fp4_bytes,
        "int4_rmse": round(rmse, 6),
    })
    return result_path


def step_archive(cfg: PipelineConfig, renderer_bin: Path, poses_path: Path,
                 iteration: int = 0, corrections_bin: Path | None = None) -> Path:
    """Build optimized submission archive."""
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    archive_path = iter_dir / "archive.zip"

    if _step_done(iter_dir, "archive"):
        _log(f"Archive already built (iter {iteration}), skipping")
        return archive_path

    _log("Building submission archive")
    from tac.submission_archive import (
        build_submission_archive, RENDERER_SUBMISSION_MANIFEST,
        RENDERER_COMPACT_MANIFEST,
    )

    is_bin = poses_path.suffix == ".bin"
    manifest = RENDERER_COMPACT_MANIFEST if (cfg.binary_poses or is_bin) else RENDERER_SUBMISSION_MANIFEST

    # Include gradient corrections if available (Eureka 6 — Contrarian)
    corr_path = corrections_bin if (corrections_bin and corrections_bin.exists()) else None
    if corr_path:
        from tac.submission_archive import ArchiveManifest
        manifest = ArchiveManifest(
            renderer_bin=manifest.renderer_bin,
            masks_mkv=manifest.masks_mkv,
            optimized_poses_pt=manifest.optimized_poses_pt,
            optimized_poses_bin=manifest.optimized_poses_bin,
            gradient_corrections_bin=True,
        )
        _log(f"  Including gradient corrections: {corr_path} ({corr_path.stat().st_size:,} bytes)")

    result = build_submission_archive(
        output_path=archive_path,
        renderer_bin=renderer_bin,
        masks_mkv=cfg.masks_archive or cfg.masks,
        optimized_poses_pt=None if is_bin else poses_path,
        optimized_poses_bin=poses_path if is_bin else None,
        gradient_corrections_bin=corr_path,
        manifest=manifest,
        validate=True,
        use_brotli=cfg.brotli,
    )

    rate = 25 * result.archive_bytes / ORIGINAL_VIDEO_BYTES
    _log(f"  {result.archive_bytes:,} bytes ({result.archive_bytes/1024:.1f} KB)")
    _log(f"  Rate: {rate:.4f}")
    for name, size in sorted(result.files_found.items()):
        _log(f"    {name}: {size:,} bytes")

    _mark_done(iter_dir, "archive", {
        "bytes": result.archive_bytes, "rate": round(rate, 6),
    })
    return archive_path


def step_engineered_corrections(cfg: PipelineConfig, renderer_bin: Path,
                                 iteration: int = 0) -> Path | None:
    """Eureka 6: Compute gradient-directed SegNet corrections.

    Pre-computes sparse pixel perturbations that flip wrong SegNet predictions
    to match GT. Stored as gradient_corrections.bin for inflate-time application.
    Contest-compliant: no scorers at inflate time.
    """
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    step_name = "engineered_corrections"
    corrections_bin = iter_dir / "gradient_corrections.bin"

    # R30: use canonical _step_done/_mark_done instead of bare touch() —
    # consistent with every other step + atomic .done write.
    if _step_done(iter_dir, step_name) and corrections_bin.exists():
        _log(f"[{step_name}] Skipping — already done ({corrections_bin})")
        return corrections_bin

    _log(f"[{step_name}] Computing SegNet-targeting corrections...")
    video_path = Path(cfg.upstream) / "videos" / "0.mkv"
    cmd = [
        sys.executable, "-u", "experiments/engineered_quant_noise.py",
        "--checkpoint", str(renderer_bin),
        "--video", str(video_path),
        "--device", cfg.device,
        "--output-dir", str(iter_dir),
    ]
    # 1-hour timeout — corrections is a single-pass gradient computation that
    # should complete in 5-30 minutes on a 4090. Anything longer is hung.
    # (R28 finding: prior call had no timeout — could block the pipeline forever.)
    try:
        result = subprocess.run(cmd, timeout=3600)
    except subprocess.TimeoutExpired:
        _log(f"[{step_name}] WARNING: Timed out after 1h, skipping corrections")
        return None
    if result.returncode != 0:
        _log(f"[{step_name}] WARNING: Failed (exit {result.returncode}), skipping corrections")
        return None

    if corrections_bin.exists():
        size = corrections_bin.stat().st_size
        _mark_done(iter_dir, step_name, {"corrections_bytes": size})
        _log(f"[{step_name}] Done: {corrections_bin} ({size:,} bytes)")
        return corrections_bin
    # Subprocess succeeded but didn't produce the expected file.
    _log(f"[{step_name}] WARNING: subprocess returned 0 but {corrections_bin} not produced")
    return None


def step_eval(cfg: PipelineConfig, renderer_bin: Path, archive_path: Path,
              iteration: int = 0) -> dict:
    """Full e2e auth evaluation through upstream scorer.

    Args:
        renderer_bin: path to renderer.bin (checkpoint for auth_eval_renderer.py)
        archive_path: path to archive.zip (for rate calculation via --archive-size-bytes)
    """
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"

    if _step_done(iter_dir, "eval"):
        _log(f"Eval already done (iter {iteration}), skipping")
        return json.loads((iter_dir / ".done_eval").read_text())

    _log("Running full e2e auth evaluation")
    archive_bytes = archive_path.stat().st_size if archive_path.exists() else 0
    cmd = [
        sys.executable, "-u", "experiments/auth_eval_renderer.py",
        "--checkpoint", str(renderer_bin),
        "--upstream-dir", cfg.upstream,
        "--device", cfg.device,
        "--archive-size-bytes", str(archive_bytes),
    ]

    # R30: stream stdout to a tempfile rather than capture_output=True. A
    # 30-min auth eval emits MB of pair-by-pair logs; buffering all of it in
    # memory could OOM on small instances. The full log lives at
    # iter_dir/auth_eval.stdout.log for postmortem; we read it back to
    # extract the RESULT_JSON sentinel.
    # R34 fix: pid-suffix the log files so concurrent invocations don't
    # truncate each other's output via open("w"). After success, we rename
    # to canonical names with os.replace for postmortem-friendly paths.
    stdout_log_pid = iter_dir / f"auth_eval.stdout.{os.getpid()}.log"
    stderr_log_pid = iter_dir / f"auth_eval.stderr.{os.getpid()}.log"
    stdout_log = stdout_log_pid
    stderr_log = stderr_log_pid
    t0 = time.monotonic()
    timed_out = False
    with stdout_log.open("w") as out_f, stderr_log.open("w") as err_f:
        try:
            result = subprocess.run(cmd, stdout=out_f, stderr=err_f, timeout=3600)
        except subprocess.TimeoutExpired:
            # Mark in-stream so postmortem readers know the log was truncated.
            timed_out = True
    if timed_out:
        # Append the sentinel AFTER the with block closes the file so the
        # marker is the last line, not interleaved with kernel-buffered writes.
        with stdout_log.open("a") as out_f:
            out_f.write("\n[PIPELINE TIMEOUT — eval killed after 1h]\n")
        raise RuntimeError(
            "Auth eval timed out after 1h — should typically take 20-30 min "
            f"on T4. See {stdout_log} (last line marked TIMEOUT)."
        )
    elapsed = time.monotonic() - t0

    _log(f"  Complete in {elapsed/60:.1f} min")

    if result.returncode != 0:
        _log(f"  Auth eval FAILED (exit {result.returncode})", "ERROR")
        # Read the last 10 lines of stderr without buffering the whole file
        try:
            stderr_tail = stderr_log.read_text().strip().split("\n")[-10:]
            for line in stderr_tail:
                _log(f"  stderr: {line}")
        except Exception:
            pass
        # NEVER mark a failed eval as done — that would cache the failure and
        # the next pipeline run would skip eval and return a phantom score.
        # (R24 finding.)
        raise RuntimeError(
            f"Auth eval failed (exit {result.returncode}). "
            f"Investigate {stderr_log}; do NOT cache this failure."
        )

    # Parse the RESULT_JSON sentinel from the stdout log (R29 schema-first
    # contract; replaces R27/R28 fragile regex).
    stdout_text = stdout_log.read_text()
    score = _extract_eval_score(stdout_text, logger=_log)

    meta = {"elapsed_s": round(elapsed), "score": score}
    _mark_done(iter_dir, "eval", meta)
    return meta


# R29: replaced fragile regex (caught bugs across rounds 28 + 29) with a
# schema-first contract. auth_eval_renderer.py now emits a single
# `RESULT_JSON: {...}` line; we json.loads + Pydantic-validate. No more
# token-by-token guessing whether a number is a score, a timing measurement,
# or a component breakdown. Schema-first per CLAUDE.md.
_RESULT_JSON_RE = re.compile(r"^RESULT_JSON:\s*(\{.*\})\s*$", re.MULTILINE)


# NON-NEGOTIABLE CONTRACT: any breaking change to the RESULT_JSON payload
# emitted by `experiments/auth_eval_renderer.py` MUST bump BOTH:
#   1. EXPECTED_AUTH_EVAL_SCHEMA below
#   2. The `schema_version` int in `auth_eval_renderer.py`'s _result_payload
# Adding a new optional field is NOT a breaking change (extra='ignore' tolerates
# it). Removing a field, renaming, or changing types IS a breaking change.
# Mismatched versions raise RuntimeError at parse time, surfacing the drift
# loudly per the schema-first standard.
EXPECTED_AUTH_EVAL_SCHEMA = 1


class AuthEvalResult(BaseModel):
    """Validated payload from auth_eval_renderer.py's RESULT_JSON sentinel line.

    Required fields are strictly typed. extra='ignore' so the emitter can add
    new optional fields without breaking older parsers (forward-compat).
    schema_version is checked separately against EXPECTED_AUTH_EVAL_SCHEMA;
    a bump there means coordinated parser change.
    """
    model_config = {"extra": "ignore"}
    schema_version: int
    final_score: float
    score_seg: float
    score_pose: float
    score_rate: float
    avg_segnet_dist: float
    avg_posenet_dist: float
    rate: float
    archive_size_bytes: int
    gt_size_bytes: int


def _extract_eval_score(stdout: str, logger=None) -> float | None:
    """Parse the RESULT_JSON sentinel line from auth_eval_renderer.py stdout.

    Returns the validated final_score, or None if no RESULT_JSON line was
    found. Raises ValidationError if the payload is malformed, or
    RuntimeError if the schema version doesn't match expectations.
    """
    matches = list(_RESULT_JSON_RE.finditer(stdout))
    if not matches:
        if logger:
            logger("  No RESULT_JSON line in auth_eval stdout — eval may be "
                   "running an old version without the sentinel emit", "WARN")
        return None
    # Use the LAST match — defensive against an eval that emits multiple
    # (e.g., per-iteration). The last is the final result.
    payload = matches[-1].group(1)
    parsed = AuthEvalResult.model_validate_json(payload)
    if parsed.schema_version != EXPECTED_AUTH_EVAL_SCHEMA:
        raise RuntimeError(
            f"RESULT_JSON schema_version={parsed.schema_version} does not match "
            f"expected={EXPECTED_AUTH_EVAL_SCHEMA}. Update AuthEvalResult and "
            f"EXPECTED_AUTH_EVAL_SCHEMA in lockstep with auth_eval_renderer.py."
        )
    if logger:
        logger(f"  Parsed RESULT_JSON: final={parsed.final_score:.4f} "
               f"seg={parsed.score_seg:.4f} pose={parsed.score_pose:.4f} "
               f"rate={parsed.score_rate:.4f}")
    return parsed.final_score


# ── Main pipeline ────────────────────────────────────────────────────────

def _capture_provenance(cfg: PipelineConfig) -> dict:
    """Build a full-provenance dict for `pipeline_config.json`. Captures the
    config + git hash + GPU info + library versions per CLAUDE.md non-negotiable
    'Full provenance' rule. (R25 finding: prior config dump had none of this.)
    """
    prov = {"config": asdict(cfg)}
    try:
        import torch
        prov["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            prov["gpu_name"] = torch.cuda.get_device_name(0)
            prov["cuda_version"] = torch.version.cuda
    except Exception as e:
        prov["torch_version_error"] = str(e)
    try:
        import platform
        prov["python_version"] = platform.python_version()
        prov["platform"] = platform.platform()
    except Exception:
        pass
    try:
        # R34 fix: timeouts on git subprocesses. NFS-backed home dirs and
        # broken post-checkout hooks can hang `git rev-parse` indefinitely;
        # provenance capture should never block the pipeline. The except
        # clause already records TimeoutExpired (subclass of Exception).
        prov["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            cwd=Path(__file__).parent.parent, timeout=10,
        ).strip()
        prov["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True,
            cwd=Path(__file__).parent.parent, timeout=10,
        ).strip())
    except Exception as e:
        prov["git_hash_error"] = str(e)
    prov["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return prov


def run_compress(cfg: PipelineConfig) -> None:
    """Full compress pipeline: video → archive with iterative optimization."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full provenance for reproducibility (R25 fix: was just asdict(cfg)).
    # R30 fix: on resume, append a `resumes` list rather than overwrite the
    # original timestamp/git_hash — otherwise the .done markers reference the
    # original run but the JSON timestamp claims today.
    prov_path = output_dir / "pipeline_config.json"
    new_prov = _capture_provenance(cfg)
    if prov_path.exists():
        try:
            existing = json.loads(prov_path.read_text())
            resumes = existing.setdefault("resumes", [])
            resumes.append({
                "timestamp_utc": new_prov.get("timestamp_utc"),
                "git_hash": new_prov.get("git_hash"),
                "git_dirty": new_prov.get("git_dirty"),
                "torch_version": new_prov.get("torch_version"),
            })
            prov_path.write_text(json.dumps(existing, indent=2))
        except (json.JSONDecodeError, KeyError):
            # Existing JSON is corrupted; overwrite.
            prov_path.write_text(json.dumps(new_prov, indent=2))
    else:
        prov_path.write_text(json.dumps(new_prov, indent=2))

    t0 = time.monotonic()
    _banner(f"COMPRESSION PIPELINE — {cfg.max_iterations} iteration(s)")
    _log(f"Video: {cfg.video}")
    _log(f"Checkpoint: {cfg.checkpoint}")
    _log(f"Device: {cfg.device}")
    _log(f"Optimizations: half_frame={cfg.half_frame}, binary_poses={cfg.binary_poses}, brotli={cfg.brotli}")

    # Step 0: Extract masks (once, shared across iterations)
    # FULL masks (1200 frames) for training/pose TTO
    # HALF masks (600 frames) for archive only
    full_masks_path, half_masks_path = step_extract_masks(cfg)
    cfg.masks = str(full_masks_path)  # training/TTO always uses full
    cfg.masks_archive = str(half_masks_path)  # archive uses half (if --half-frame)

    prev_score = float("inf")
    for iteration in range(cfg.max_iterations):
        _banner(f"ITERATION {iteration + 1} / {cfg.max_iterations}")

        # Step 1: Export
        renderer_bin = step_export(cfg, iteration)

        # Step 2: Pose TTO
        poses_path = step_pose_tto(cfg, iteration)

        # Step 2.5: Scorer-Jacobian sensitivity sweep (Eureka 3 — Hotz)
        # Runs BEFORE QAT so the bit allocation can inform quantization-aware
        # training. Only runs when weight_compression is not plain fp4.
        sensitivity_json: Path | None = None
        if cfg.weight_compression != "fp4":
            sensitivity_json = step_sensitivity_sweep(cfg, iteration)

        # Step 3: QAT (pass mixed-precision allocation if available)
        qat_bin = step_qat(cfg, iteration, mixed_precision_json=sensitivity_json)

        # Step 3.5: Fridrich steganalytic refinement (post-QAT polish)
        fridrich_ckpt = step_fridrich_refine(cfg, iteration)

        # Step 3.7: Weight compression (int4+LZMA2 or FP4, per cfg.weight_compression)
        if fridrich_ckpt.exists() and fridrich_ckpt.suffix == ".pt":
            best_ckpt = fridrich_ckpt
        else:
            # Fall back to QAT best, then raw export. R33 fix: filename is
            # qat_best_float.pt (per qat_finetune.py), not renderer_qat_best.pt.
            # The wrong name made this fallback ALWAYS fall through to
            # cfg.checkpoint, silently using pre-QAT weights.
            qat_best = Path(cfg.output_dir) / f"iter_{iteration}" / "qat_best_float.pt"
            best_ckpt = qat_best if qat_best.exists() else Path(cfg.checkpoint)

        if cfg.weight_compression != "fp4" and best_ckpt.exists():
            final_renderer = step_compress_weights(
                cfg, best_ckpt, iteration, sensitivity_json=sensitivity_json,
            )
        elif best_ckpt.exists() and best_ckpt.suffix == ".pt":
            # R32 fix: do NOT re-call step_export — its .done_export marker
            # was set at iteration start with the ORIGINAL cfg.checkpoint
            # path, so it would silently return the stale renderer.bin and
            # discard the Fridrich-refined weights. Inline the export here
            # to a distinct filename (renderer_refined.bin) so the
            # downstream consumer sees the right artifact.
            final_renderer = _export_refined_checkpoint(cfg, best_ckpt, iteration)
        else:
            final_renderer = qat_bin if qat_bin.exists() else renderer_bin

        # Step 3.8: Engineered SegNet corrections (Eureka 6 — Contrarian)
        corrections_bin = step_engineered_corrections(cfg, final_renderer, iteration)

        # Step 4: Archive (include corrections if available)
        archive_path = step_archive(cfg, final_renderer, poses_path, iteration,
                                     corrections_bin=corrections_bin)

        # Step 5: Eval (pass renderer.bin for scoring, archive.zip for rate)
        eval_result = step_eval(cfg, final_renderer, archive_path, iteration)
        score = eval_result.get("score")

        if score is not None:
            improvement = prev_score - score
            _log(f"  Score: {score:.4f} (improvement: {improvement:+.4f})")

            if iteration > 0 and improvement < cfg.convergence_tol:
                _log(f"  Converged (improvement {improvement:.4f} < tol {cfg.convergence_tol})")
                break

            prev_score = score

        # For next iteration: use QAT output as starting point.
        # R33 fix: filename is qat_best_float.pt (per qat_finetune.py), not
        # renderer_qat_best.pt. The wrong name made this guard ALWAYS False,
        # so cfg.checkpoint never advanced — every multi-iteration run was
        # silently equivalent to --max-iterations 1.
        if iteration < cfg.max_iterations - 1:
            qat_ckpt = Path(cfg.output_dir) / f"iter_{iteration}" / "qat_best_float.pt"
            if qat_ckpt.exists():
                cfg.checkpoint = str(qat_ckpt)
                _log(f"  Next iteration starts from: {qat_ckpt}")
            else:
                _log(f"  WARN: qat_best_float.pt not found in iter_{iteration} — "
                     f"next iteration will re-train from original cfg.checkpoint")

    total = time.monotonic() - t0
    _banner(f"PIPELINE COMPLETE — {total/60:.1f} min total")


def run_eval(args: argparse.Namespace) -> None:
    """Evaluate an existing archive against an existing renderer checkpoint.

    R29 fix: prior version passed `archive` as both args to step_eval, which
    sent the archive .zip as --checkpoint to auth_eval_renderer.py. The eval
    script expected a renderer .pt/.bin → cryptic load error. Every
    `pipeline.py eval` invocation was broken.
    R30 fix: support --profile so arch fields can match training.
    """
    # If --profile given, load matching arch fields onto args.
    if getattr(args, "profile", None):
        _apply_profile(args, args.profile, user_provided_flags=set())
    pcfg_field_names = {f.name for f in fields(PipelineConfig)}
    base_kwargs = {
        "video": args.video, "device": args.device,
        "upstream": getattr(args, "upstream", "upstream"),
        "output_dir": str(Path(args.archive).parent),
    }
    # Pull any arch fields the profile injected; they're already on args.
    for field_name in pcfg_field_names:
        if field_name in base_kwargs:
            continue
        if hasattr(args, field_name):
            base_kwargs[field_name] = getattr(args, field_name)
    cfg = PipelineConfig(**base_kwargs)
    archive = Path(args.archive)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(
            f"--checkpoint {checkpoint} does not exist. eval needs both the "
            f"archive (for rate calculation) and the renderer (for forward pass)."
        )
    result = step_eval(cfg, checkpoint, archive)
    score = result.get("score")
    # R30 fix: was `if score:` which is falsy for score=0.0 (a perfect score).
    # Use explicit None check, mirroring the pattern in run_compress.
    if score is not None:
        _log(f"Final score: {score:.4f}")


# ── CLI ──────────────────────────────────────────────────────────────────

def _user_provided_flags(
    parser: argparse.ArgumentParser, argv: list[str],
    *, active_subcommand: str,
) -> set[str]:
    """Return the set of arg.dest values that the user explicitly passed in argv.

    Walks the top-level parser AND the active subparser only — NOT every
    subparser, which could mark the wrong dest if two subparsers share an
    option string with different dest names. (R27 finding.)

    `active_subcommand` is REQUIRED (kw-only, no default). Earlier versions
    accepted None and silently returned an empty set for subparser flags —
    a footgun that would clobber every CLI override with profile values.
    (R28 finding.)
    """
    if not active_subcommand or not isinstance(active_subcommand, str):
        raise ValueError(
            f"active_subcommand must be the name of the active subparser "
            f"(e.g., 'compress'). Got {active_subcommand!r}."
        )
    user_set: set[str] = set()
    actions: list[argparse.Action] = list(parser._actions)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            sub = action.choices.get(active_subcommand)
            if sub is not None:
                actions.extend(sub._actions)

    for action in actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            matched = False
            for token in argv:
                if token == opt or token.startswith(opt + "="):
                    user_set.add(action.dest)
                    matched = True
                    break
            if matched:
                break
    return user_set


def _apply_profile(args: argparse.Namespace, profile_name: str,
                   user_provided_flags: set[str] | None = None) -> None:
    """Load PROFILES[profile_name]. Profile fills in defaults; explicit CLI
    flags from `user_provided_flags` win. Profile keys that don't match a
    PipelineConfig field are SILENTLY SKIPPED (they're for training scripts
    like train_distill.py that consume the profile separately).

    However, a profile key that LOOKS like a PipelineConfig field but is
    misspelled (Levenshtein close-match) FAILS LOUDLY — that's the SHIRAZ
    class. E.g. `motino_hidden` is close to `motion_hidden` → fail.
    `use_tto` is not close to anything → silent skip (training-only).

    `user_provided_flags` is a set of arg names that the operator explicitly
    passed on the command line. These are NOT overwritten by the profile.
    """
    import difflib
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from tac.profiles import PROFILES
    if profile_name not in PROFILES:
        raise SystemExit(
            f"unknown --profile {profile_name!r}. Available: {sorted(PROFILES.keys())}"
        )
    profile = PROFILES[profile_name]
    user_provided_flags = user_provided_flags or set()

    pcfg_fields = {f.name for f in fields(PipelineConfig)}
    likely_typos: list[tuple[str, str]] = []
    for key in profile.keys():
        if key in pcfg_fields:
            continue
        # cutoff=0.85: tight enough to flag motino_hidden→motion_hidden but
        # not loose enough to flag every training-only key as a typo of
        # something. Empirical: 0.85 catches 1-2 char edits on names ≥6 chars.
        close = difflib.get_close_matches(key, pcfg_fields, n=1, cutoff=0.85)
        if close:
            likely_typos.append((key, close[0]))
    if likely_typos:
        msg_parts = [f"  {k!r} → did you mean {v!r}?" for k, v in likely_typos]
        raise SystemExit(
            f"profile {profile_name!r} has likely-typo keys (close-match to "
            f"a real PipelineConfig field):\n" + "\n".join(msg_parts) +
            "\n\nFix the typos or, if intentional, rename the key to something "
            "clearly distinct from existing PipelineConfig field names."
        )

    for key, value in profile.items():
        if key in pcfg_fields and key not in user_provided_flags:
            setattr(args, key, value)
    args._resolved_profile = profile_name


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Canonical compression pipeline for comma.ai video compression challenge",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compress subcommand
    comp = sub.add_parser("compress", help="Compress video to submission archive")
    comp.add_argument("--profile", default=None,
                      help="Named profile from src/tac/profiles.py — overrides every "
                           "matching arch/discipline arg with the profile value. The "
                           "canonical way to launch (no SHIRAZ-class arg drift).")
    comp.add_argument("--video", required=True, help="Input video path")
    comp.add_argument("--checkpoint", required=True, help="Trained renderer checkpoint")
    comp.add_argument("--masks", default="", help="FULL masks (1200 frames) for training/TTO")
    comp.add_argument("--masks-archive", default="", help="HALF-FRAME masks (600 frames) for archive only")
    comp.add_argument("--output-dir", default="experiments/results/pipeline")
    comp.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    comp.add_argument("--upstream", default="upstream")
    # Architecture
    comp.add_argument("--base-ch", type=int, default=36)
    comp.add_argument("--mid-ch", type=int, default=60)
    comp.add_argument("--motion-hidden", type=int, default=32)
    comp.add_argument("--depth", type=int, default=1)
    comp.add_argument("--pose-dim", type=int, default=6)
    comp.add_argument("--embed-dim", type=int, default=6)
    comp.add_argument("--use-dsconv", action="store_true", default=False)
    comp.add_argument("--padding-mode", type=str, default="replicate",
                      choices=["zeros", "reflect", "replicate", "circular"])
    comp.add_argument("--use-dilation", action="store_true")
    comp.add_argument("--use-zoom-flow", action="store_true",
                      help="GREEN profile: 4ch MotionPredictor + RadialZoomWarp")
    # Training-discipline boolean flags (passed through to step_fridrich_refine,
    # which re-invokes train_distill.py). Profile-driven; usually set via --profile.
    comp.add_argument("--use-swa", action="store_true")
    comp.add_argument("--use-per-class-weights", action="store_true")
    comp.add_argument("--freeze-motion-phase2", action="store_true")
    comp.add_argument("--freeze-renderer-phase3", action="store_true")
    comp.add_argument("--beneficial-quant-noise", action="store_true")
    # Optimization
    comp.add_argument("--pose-max-steps", type=int, default=2000)
    comp.add_argument("--pose-lr", type=float, default=0.01)
    comp.add_argument("--pose-batch-pairs", type=int, default=16,
                      help="OOM mitigation knob — reduce on smaller GPUs (R34)")
    comp.add_argument("--qat-max-epochs", type=int, default=500)
    comp.add_argument("--qat-lr", type=float, default=5e-5)
    # Archive
    comp.add_argument("--half-frame", action="store_true")
    comp.add_argument("--binary-poses", action="store_true")
    comp.add_argument("--brotli", action="store_true")
    comp.add_argument("--mask-crf", type=int, default=50)
    # Weight compression
    comp.add_argument("--weight-compression", type=str, default="fp4",
                      choices=["fp4", "int4_lzma2", "auto"],
                      help="Weight compression: fp4 (default), int4_lzma2, or auto")
    # Iteration
    comp.add_argument("--max-iterations", type=int, default=10,
                      help="Safety bound on convergence cycles (stops earlier when converged)")

    # eval subcommand
    ev = sub.add_parser("eval", help="Evaluate an existing archive")
    ev.add_argument("--archive", required=True, help="Path to archive.zip (for rate calculation)")
    ev.add_argument("--checkpoint", required=True,
                    help="Renderer checkpoint (.pt or .bin) for forward pass. R29 fix.")
    ev.add_argument("--video", required=True, help="GT video for scoring")
    ev.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    # R30: auth_eval_renderer.py reads arch from checkpoint header for .bin
    # formats but inherits PipelineConfig defaults for .pt loading. Make
    # arch overridable so an eval-only invocation can match the training arch.
    ev.add_argument("--profile", default=None,
                    help="Optional profile name to populate arch fields (matches compress)")
    ev.add_argument("--upstream", default="upstream")

    args = parser.parse_args()

    if args.command == "compress":
        # Discover which flags the user explicitly typed (vs argparse defaults)
        # by walking sys.argv. This preserves "explicit CLI overrides profile"
        # semantics so `--profile X --base-ch 999` actually uses base_ch=999.
        # Anything else is filled in from the profile.
        user_provided = _user_provided_flags(parser, sys.argv[1:], active_subcommand="compress")
        if getattr(args, "profile", None):
            _apply_profile(args, args.profile, user_provided_flags=user_provided)
        # Build PipelineConfig from every args field that overlaps a
        # dataclass field. This automatically picks up any new field added
        # to PipelineConfig OR set by _apply_profile, with no risk of
        # forgetting to thread one through (R26 finding: 7 fields were
        # silently stuck at dataclass defaults under the manual approach).
        pcfg_field_names = {f.name for f in fields(PipelineConfig)}
        kwargs = {k: v for k, v in vars(args).items() if k in pcfg_field_names}
        cfg = PipelineConfig(**kwargs)
        run_compress(cfg)
    elif args.command == "eval":
        run_eval(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
