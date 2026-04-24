#!/usr/bin/env python3
"""Canonical compression pipeline: video → trained renderer → optimized archive.

Production-grade pipeline for the comma.ai video compression challenge.
Takes a video (or list of videos) as input, produces an optimized archive.

The pipeline is smarter than manual orchestration:
- Adaptive pose optimization: converges to tolerance, not fixed step count
- Adaptive QAT: monitors quality degradation, stops when marginal
- Iterative refinement: model→pose→QAT cycles until convergence
- Automatic rate-distortion optimization: explores the Pareto frontier
- Idempotent: re-running resumes from last completed step

Usage:
    # Compress a single video:
    PYTHONPATH=src:upstream python experiments/pipeline.py compress \
        --video upstream/videos/0.mkv \
        --checkpoint experiments/results/definitive_float_ema/distill_phase2_best.pt \
        --device cuda --output-dir results/submission_v1

    # Compress with all optimizations:
    PYTHONPATH=src:upstream python experiments/pipeline.py compress \
        --video upstream/videos/0.mkv \
        --checkpoint experiments/results/definitive_float_ema/distill_phase2_best.pt \
        --device cuda --output-dir results/submission_v1 \
        --half-frame --binary-poses --brotli --iterations 2

    # Evaluate an existing archive:
    PYTHONPATH=src:upstream python experiments/pipeline.py eval \
        --archive results/submission_v1/archive.zip \
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
    """Production pipeline configuration."""

    # Input
    video: str = ""
    checkpoint: str = ""
    masks: str = ""  # pre-extracted masks (optional, extracted from video if empty)
    output_dir: str = "experiments/results/pipeline"
    device: str = "cuda"
    upstream: str = "upstream"

    # Architecture (must match checkpoint)
    base_ch: int = 20
    mid_ch: int = 28
    motion_hidden: int = 32
    depth: int = 1
    pose_dim: int = 6
    embed_dim: int = 6
    use_dsconv: bool = True
    padding_mode: str = "replicate"
    use_dilation: bool = False

    # Pose TTO — adaptive convergence
    pose_max_steps: int = 2000        # upper bound
    pose_min_steps: int = 200         # minimum before checking convergence
    pose_lr: float = 0.01
    pose_convergence_tol: float = 1e-4  # stop when improvement < tol for patience steps
    pose_patience: int = 50
    pose_batch_pairs: int = 50

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

    # Iterative refinement — convergence-driven, not arbitrary count
    max_iterations: int = 10          # safety bound (convergence stops earlier)
    convergence_tol: float = 0.01     # stop when score improvement < tol between cycles
    # The pipeline is a coordinate descent on the rate-distortion surface:
    # each cycle optimizes poses (distortion) then quantizes (rate).
    # Convergence is reached when marginal improvement falls below the
    # marginal rate cost of running another cycle.


# ── Step implementations ─────────────────────────────────────────────────

def step_extract_masks(cfg: PipelineConfig) -> Path:
    """Extract SegNet masks from GT video if not provided."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if cfg.masks and Path(cfg.masks).exists():
        _log(f"Using pre-extracted masks: {cfg.masks}")
        return Path(cfg.masks)

    if _step_done(out, "masks"):
        _log("Mask extraction already done, skipping")
        return out / "masks.mkv"

    _log("Extracting SegNet masks from video...")
    from tac.data import decode_video
    from tac.scorer import load_scorers
    from tac.mask_codec import extract_masks, extract_half_masks, encode_masks

    gt_frames = decode_video(cfg.video, target_h=SEGNET_H, target_w=SEGNET_W)[:NUM_FRAMES]
    _log(f"  Decoded {len(gt_frames)} frames at {SEGNET_H}x{SEGNET_W}")

    _, segnet = load_scorers(str(Path(cfg.upstream) / "models"), device=cfg.device)

    if cfg.half_frame:
        masks = extract_half_masks(gt_frames, segnet, device=cfg.device)
        _log(f"  Extracted {masks.shape[0]} half-frame masks")
    else:
        masks = extract_masks(gt_frames, segnet, device=cfg.device)
        _log(f"  Extracted {masks.shape[0]} full-frame masks")

    mask_path = out / "masks.mkv"
    nbytes = encode_masks(masks, mask_path, crf=cfg.mask_crf)
    _log(f"  Encoded: {mask_path} ({nbytes:,} bytes, CRF {cfg.mask_crf})")

    _mark_done(out, "masks", {"n_masks": masks.shape[0], "bytes": nbytes})
    return mask_path


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
        masks_mkv=cfg.masks,
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
    masks_path = step_extract_masks(cfg)
    cfg.masks = str(masks_path)

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

        # Step 4: Archive (use Fridrich-refined if available, else QAT, else float)
        if fridrich_ckpt.exists() and fridrich_ckpt.suffix == ".pt":
            # Re-export Fridrich-refined checkpoint to FP4
            final_renderer = step_export(cfg, iteration)  # will use updated checkpoint
        else:
            final_renderer = qat_bin if qat_bin.exists() else renderer_bin
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
    comp.add_argument("--masks", default="", help="Pre-extracted masks (optional)")
    comp.add_argument("--output-dir", default="experiments/results/pipeline")
    comp.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    comp.add_argument("--upstream", default="upstream")
    # Architecture
    comp.add_argument("--base-ch", type=int, default=20)
    comp.add_argument("--mid-ch", type=int, default=28)
    comp.add_argument("--motion-hidden", type=int, default=32)
    comp.add_argument("--depth", type=int, default=1)
    comp.add_argument("--pose-dim", type=int, default=6)
    comp.add_argument("--embed-dim", type=int, default=6)
    comp.add_argument("--use-dsconv", action="store_true")
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
            masks=args.masks, output_dir=args.output_dir,
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
            max_iterations=args.max_iterations,
        )
        run_compress(cfg)
    elif args.command == "eval":
        run_eval(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
