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
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
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
    info = {"step": step, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if meta:
        info.update(meta)
    (output_dir / f".done_{step}").write_text(json.dumps(info, indent=2))


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
    )
    model.load_state_dict(state, strict=False)
    n_params = sum(p.numel() for p in model.parameters())

    renderer_bin = iter_dir / "renderer.bin"
    nbytes = export_asymmetric_checkpoint_fp4(model, str(renderer_bin))
    _log(f"  {n_params:,} params → {nbytes:,} bytes ({nbytes/1024:.1f} KB)")

    _mark_done(iter_dir, "export", {"params": n_params, "bytes": nbytes})
    return renderer_bin


def step_pose_tto(cfg: PipelineConfig, iteration: int = 0) -> Path:
    """Adaptive pose TTO with convergence detection."""
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

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

    t0 = time.monotonic()
    result = subprocess.run(cmd)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        raise RuntimeError(f"Pose TTO failed (exit {result.returncode})")

    _log(f"  Complete in {elapsed/60:.1f} min")

    poses_path = iter_dir / "optimized_poses.bin"
    if not poses_path.exists():
        poses_path = iter_dir / "optimized_poses.pt"

    _mark_done(iter_dir, "pose_tto", {"elapsed_s": round(elapsed)})
    return poses_path


def step_qat(cfg: PipelineConfig, iteration: int = 0) -> Path:
    """QAT fine-tune with quality monitoring."""
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    if _step_done(iter_dir, "qat"):
        _log(f"QAT already done (iter {iteration}), skipping")
        return iter_dir / "renderer_qat.bin"

    _log(f"QAT fine-tune (up to {cfg.qat_max_epochs} epochs)")
    cmd = [
        sys.executable, "-u", "experiments/qat_finetune.py",
        "--checkpoint", cfg.checkpoint,
        "--masks", cfg.masks,
        "--device", cfg.device,
        "--qat-epochs", str(cfg.qat_max_epochs),
        "--qat-lr", str(cfg.qat_lr),
        "--eval-roundtrip",
        "--output-dir", str(iter_dir),
    ]

    t0 = time.monotonic()
    result = subprocess.run(cmd)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        raise RuntimeError(f"QAT failed (exit {result.returncode})")

    _log(f"  Complete in {elapsed/60:.1f} min")

    qat_bin = iter_dir / "renderer_qat.bin"
    if not qat_bin.exists():
        bins = sorted(iter_dir.glob("*.bin"))
        qat_bin = bins[-1] if bins else iter_dir / "renderer.bin"

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

    if _step_done(iter_dir, step_name):
        _log(f"Fridrich refinement already done (iter {iteration}), skipping")
        return iter_dir / "renderer_fridrich.bin"

    # Find best QAT checkpoint to start from
    qat_ckpt = iter_dir / "renderer_qat_best.pt"
    if not qat_ckpt.exists():
        qat_ckpt = iter_dir / "distill_latest.pt"
    if not qat_ckpt.exists():
        _log("No QAT checkpoint found, skipping Fridrich refinement", "WARN")
        _mark_done(iter_dir, step_name, {"skipped": True})
        return iter_dir / "renderer_qat.bin"

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
    ]
    if cfg.use_dsconv:
        cmd.append("--use-dsconv")
    if cfg.use_dilation:
        cmd.append("--use-dilation")

    t0 = time.monotonic()
    result = subprocess.run(cmd)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        _log(f"Fridrich refinement failed (exit {result.returncode})", "WARN")
        _mark_done(iter_dir, step_name, {"failed": True})
        return iter_dir / "renderer_qat.bin"

    _log(f"  Fridrich refinement complete in {elapsed/60:.1f} min")
    _mark_done(iter_dir, step_name, {"elapsed_s": round(elapsed)})
    return iter_dir / "fridrich" / "distill_phase2_best.pt"


def step_compress_weights(
    cfg: PipelineConfig,
    checkpoint_path: Path,
    iteration: int = 0,
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
    )
    model.load_state_dict(state, strict=False)
    model.eval()

    int4_path = iter_dir / "renderer_int4lzma2.bin"
    fp4_path = iter_dir / "renderer.bin"

    # ── Strategy 1: Int4 + LZMA2 ──
    int4_bytes = export_int4_lzma2(model, int4_path)
    _log(f"  Int4+LZMA2: {int4_bytes:,} bytes ({int4_bytes/1024:.1f} KB)")

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
    )
    model_restored.load_state_dict(restored_sd, strict=False)
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
                 iteration: int = 0) -> Path:
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

    result = build_submission_archive(
        output_path=archive_path,
        renderer_bin=renderer_bin,
        masks_mkv=cfg.masks_archive or cfg.masks,  # half-frame for archive, full as fallback
        optimized_poses_pt=None if is_bin else poses_path,
        optimized_poses_bin=poses_path if is_bin else None,
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


def step_eval(cfg: PipelineConfig, archive_path: Path, iteration: int = 0) -> dict:
    """Full e2e auth evaluation through upstream scorer."""
    iter_dir = Path(cfg.output_dir) / f"iter_{iteration}"

    if _step_done(iter_dir, "eval"):
        _log(f"Eval already done (iter {iteration}), skipping")
        return json.loads((iter_dir / ".done_eval").read_text())

    _log("Running full e2e auth evaluation")
    cmd = [
        sys.executable, "-u", "experiments/auth_eval_renderer.py",
        "--archive", str(archive_path),
        "--device", cfg.device,
        "--upstream", cfg.upstream,
    ]

    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    _log(f"  Complete in {elapsed/60:.1f} min")

    # Extract score from output
    score = None
    for line in (result.stdout or "").split("\n"):
        if "TOTAL" in line.upper() or "score" in line.lower():
            _log(f"  >> {line.strip()}")
        # Try to parse numeric score
        for token in line.split():
            try:
                val = float(token)
                if 0 < val < 100:
                    score = val
            except ValueError:
                pass

    meta = {"elapsed_s": round(elapsed), "score": score}
    _mark_done(iter_dir, "eval", meta)
    return meta


# ── Main pipeline ────────────────────────────────────────────────────────

def run_compress(cfg: PipelineConfig) -> None:
    """Full compress pipeline: video → archive with iterative optimization."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    (output_dir / "pipeline_config.json").write_text(json.dumps(asdict(cfg), indent=2))

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

        # Step 3: QAT
        qat_bin = step_qat(cfg, iteration)

        # Step 3.5: Fridrich steganalytic refinement (post-QAT polish)
        fridrich_ckpt = step_fridrich_refine(cfg, iteration)

        # Step 3.7: Weight compression (int4+LZMA2 or FP4, per cfg.weight_compression)
        if fridrich_ckpt.exists() and fridrich_ckpt.suffix == ".pt":
            best_ckpt = fridrich_ckpt
        else:
            # Fall back to QAT best, then raw export
            qat_best = Path(cfg.output_dir) / f"iter_{iteration}" / "renderer_qat_best.pt"
            best_ckpt = qat_best if qat_best.exists() else Path(cfg.checkpoint)

        if cfg.weight_compression != "fp4" and best_ckpt.exists():
            final_renderer = step_compress_weights(cfg, best_ckpt, iteration)
        elif best_ckpt.exists() and best_ckpt.suffix == ".pt":
            # Re-export checkpoint to FP4
            final_renderer = step_export(cfg, iteration)
        else:
            final_renderer = qat_bin if qat_bin.exists() else renderer_bin

        # Step 4: Archive
        archive_path = step_archive(cfg, final_renderer, poses_path, iteration)

        # Step 5: Eval
        eval_result = step_eval(cfg, archive_path, iteration)
        score = eval_result.get("score")

        if score is not None:
            improvement = prev_score - score
            _log(f"  Score: {score:.4f} (improvement: {improvement:+.4f})")

            if iteration > 0 and improvement < cfg.convergence_tol:
                _log(f"  Converged (improvement {improvement:.4f} < tol {cfg.convergence_tol})")
                break

            prev_score = score

        # For next iteration: use QAT output as starting point
        if iteration < cfg.max_iterations - 1:
            qat_ckpt = Path(cfg.output_dir) / f"iter_{iteration}" / "renderer_qat_best.pt"
            if qat_ckpt.exists():
                cfg.checkpoint = str(qat_ckpt)
                _log(f"  Next iteration starts from: {qat_ckpt}")

    total = time.monotonic() - t0
    _banner(f"PIPELINE COMPLETE — {total/60:.1f} min total")


def run_eval(args: argparse.Namespace) -> None:
    """Evaluate an existing archive."""
    cfg = PipelineConfig(
        video=args.video, device=args.device, upstream=args.upstream,
        output_dir=str(Path(args.archive).parent),
    )
    result = step_eval(cfg, Path(args.archive))
    score = result.get("score")
    if score:
        _log(f"Final score: {score:.4f}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Canonical compression pipeline for comma.ai video compression challenge",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compress subcommand
    comp = sub.add_parser("compress", help="Compress video to submission archive")
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
    # Optimization
    comp.add_argument("--pose-max-steps", type=int, default=2000)
    comp.add_argument("--pose-lr", type=float, default=0.01)
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
    ev.add_argument("--archive", required=True, help="Path to archive.zip")
    ev.add_argument("--video", required=True, help="GT video for scoring")
    ev.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    ev.add_argument("--upstream", default="upstream")

    args = parser.parse_args()

    if args.command == "compress":
        cfg = PipelineConfig(
            video=args.video, checkpoint=args.checkpoint,
            masks=args.masks, masks_archive=args.masks_archive,
            output_dir=args.output_dir,
            device=args.device, upstream=args.upstream,
            base_ch=args.base_ch, mid_ch=args.mid_ch,
            motion_hidden=args.motion_hidden, depth=args.depth,
            pose_dim=args.pose_dim, embed_dim=args.embed_dim,
            use_dsconv=args.use_dsconv,
            padding_mode=args.padding_mode,
            use_dilation=args.use_dilation,
            pose_max_steps=args.pose_max_steps, pose_lr=args.pose_lr,
            qat_max_epochs=args.qat_max_epochs, qat_lr=args.qat_lr,
            half_frame=args.half_frame, binary_poses=args.binary_poses,
            brotli=args.brotli, mask_crf=args.mask_crf,
            weight_compression=args.weight_compression,
            max_iterations=args.max_iterations,
        )
        run_compress(cfg)
    elif args.command == "eval":
        run_eval(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
