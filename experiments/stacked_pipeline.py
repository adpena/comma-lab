#!/usr/bin/env python3
"""Stacked Pipeline: end-to-end chain of all generation methods.

Chains the following stages:
  1. GT-Sparse TTO (start from GT, perturb insensitive patches)
  2. Per-pair ensemble selection (if multiple candidates available)
  3. Post-filter (CNN from submissions/robust_current/postfilter_int8.pt)
  4. Proxy + auth eval

The pipeline is modular: each stage can be skipped or configured independently.
Stage outputs are checkpointed for resume support.

Usage::

    # Smoke test:
    PYTHONPATH=src:upstream python experiments/stacked_pipeline.py \
        --device mps --smoke

    # Full run with GT-sparse TTO + postfilter:
    PYTHONPATH=src:upstream python experiments/stacked_pipeline.py \
        --device cuda --n-frames 1200

    # Skip TTO, just evaluate existing frames:
    PYTHONPATH=src:upstream python experiments/stacked_pipeline.py \
        --device cuda --frames-path experiments/results/gt_sparse_tto_*/sparse_tto_frames.pt
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch


from tac.scorer import compute_proxy_score
from tac.utils import find_project_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stacked Pipeline: GT-TTO + ensemble + postfilter + eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--n-frames", type=int, default=1200, help="Number of frames")
    p.add_argument("--upstream", type=str, default=None, help="Path to upstream repo")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p.add_argument("--video", type=str, default=None, help="Path to GT video")

    # Stage control
    p.add_argument("--frames-path", type=str, default=None,
                   help="Path to precomputed frames .pt (skip GT-TTO stage)")
    p.add_argument("--skip-postfilter", action="store_true",
                   help="Skip post-filter stage")
    p.add_argument("--postfilter-path", type=str, default=None,
                   help="Path to postfilter model (default: submissions/robust_current/postfilter_int8.pt)")

    # GT-TTO config (used if --frames-path is not provided)
    p.add_argument("--tto-patches", type=int, default=50, help="Patches for sparse TTO")
    p.add_argument("--tto-restarts", type=int, default=5, help="Restarts for sparse TTO")
    p.add_argument("--tto-steps", type=int, default=200, help="Steps per restart")
    p.add_argument("--sensitivity-map", type=str, default=None,
                   help="Precomputed sensitivity map for TTO")

    # CLAUDE.md non-negotiable: eval_roundtrip ALWAYS True. Removed
    # `--no-eval-roundtrip` flag; only escape hatch is TAC_ALLOW_NO_ROUNDTRIP=1.
    p.add_argument("--eval-roundtrip", action="store_true", default=True,
                   help="Simulate official scorer resolution round-trip. "
                        "ALWAYS True; disabling requires TAC_ALLOW_NO_ROUNDTRIP=1.")
    p.add_argument("--smoke", action="store_true", help="Smoke test")
    return p.parse_args()


def apply_postfilter(
    frames: torch.Tensor,
    postfilter_path: str | Path,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Apply the CNN post-filter to frames.

    The post-filter is a small CNN trained to minimize scorer distortion.
    It operates on individual frames (not pairs).

    Args:
        frames: (N, H, W, 3) float tensor in [0, 255].
        postfilter_path: path to postfilter .pt file.
        device: computation device.
        batch_size: frames per batch.

    Returns:
        (N, H, W, 3) float tensor of filtered frames in [0, 255].
    """
    postfilter = torch.jit.load(str(postfilter_path), map_location=device)
    postfilter.eval()

    N = frames.shape[0]
    filtered = torch.zeros_like(frames)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = frames[start:end].to(device)

        # Post-filter expects (B, 3, H, W) in [0, 1]
        batch_chw = batch.permute(0, 3, 1, 2) / 255.0

        with torch.no_grad():
            out = postfilter(batch_chw)

        # Back to (B, H, W, 3) in [0, 255]
        out_hwc = (out.permute(0, 2, 3, 1) * 255.0).round().clamp(0, 255)
        filtered[start:end] = out_hwc.cpu()

    del postfilter
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return filtered



def _enforce_eval_roundtrip(args) -> None:
    """CLAUDE.md non-negotiable: eval_roundtrip ALWAYS True; only escape hatch
    is TAC_ALLOW_NO_ROUNDTRIP=1 env var with loud banner.

    2026-04-27 codex R5-4 #4: delegated to the centralised
    `tac.eval_roundtrip_gate.enforce_eval_roundtrip` helper. The previous
    per-script copies were sticky — they only printed the warning when
    `args.eval_roundtrip` was already False, so a leftover env var in a
    shell / tmux session silently relaxed later runs without acknowledgement.
    The centralised helper warns whenever the env var is present and
    records it in run provenance.
    """
    from tac.eval_roundtrip_gate import enforce_eval_roundtrip
    output_dir = getattr(args, "output_dir", None)
    enforce_eval_roundtrip(args, output_dir=output_dir, write_provenance=output_dir is not None)


def main():
    args = parse_args()

    if args.smoke:
        args.n_frames = 20
        args.tto_patches = 20
        args.tto_restarts = 1
        args.tto_steps = 10
        print("[smoke] Smoke test mode")

    args.n_frames = args.n_frames - (args.n_frames % 2)

    root = find_project_root()
    device = torch.device(args.device)
    upstream = Path(args.upstream) if args.upstream else root / "upstream"

    if args.output_dir is None:
        ts = time.strftime("%Y%m%dT%H%M%S")
        args.output_dir = str(root / "experiments" / "results" / f"stacked_pipeline_{ts}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # codex R5-r6 #3: gate AFTER output_dir resolution so sidecar lands.
    _enforce_eval_roundtrip(args)

    video_path = args.video or str(upstream / "videos" / "0.mkv")

    if args.postfilter_path is None:
        args.postfilter_path = str(root / "submissions" / "robust_current" / "postfilter_int8.pt")

    t_total_start = time.monotonic()
    stage_results = {}

    # ── Load scorers ─────────────────────────────────────────────────────
    print("[setup] Loading scorers...")
    from tac.scorer import load_differentiable_scorers
    posenet, segnet = load_differentiable_scorers(upstream, device=str(device))

    # ── Decode GT video ──────────────────────────────────────────────────
    print(f"[setup] Decoding GT video ({args.n_frames} frames)...")
    from tac.data import load_gt_video

    gt_frames = load_gt_video(video_path, n_frames=args.n_frames)
    args.n_frames = len(gt_frames)

    # GT baseline
    gt_tensor = torch.stack(gt_frames).float()
    gt_score = compute_proxy_score(gt_tensor, gt_frames, posenet, segnet, device,
                                   eval_roundtrip=args.eval_roundtrip)
    stage_results["gt_baseline"] = gt_score
    print(f"[GT baseline] score={gt_score['score']:.6f}")

    # ── Stage 1: Get frames (GT-TTO or precomputed) ─────────────────────
    if args.frames_path and Path(args.frames_path).exists():
        print(f"\n[stage 1] Loading precomputed frames: {args.frames_path}")
        current_frames = torch.load(args.frames_path, map_location="cpu", weights_only=True).float()
        # Truncate to n_frames if needed
        current_frames = current_frames[:args.n_frames]
    else:
        print("\n[stage 1] Running GT-Sparse TTO...")
        t0 = time.monotonic()

        # Import GT-sparse TTO components. Ensure project root is importable.
        import sys
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        from experiments.gt_sparse_tto import (
            compute_sensitivity_map_live,
            extract_gt_masks,
            extract_gt_pose_targets,
            run_sparse_tto_batch,
            select_patches_for_frames,
        )

        # Sensitivity map
        if args.sensitivity_map and Path(args.sensitivity_map).exists():
            sensitivity_map = torch.load(args.sensitivity_map, map_location="cpu", weights_only=True)
        else:
            sensitivity_map = compute_sensitivity_map_live(gt_frames, posenet, device, batch_pairs=4)
            torch.save(sensitivity_map, output_dir / "sensitivity_map.pt")

        # Patch selection
        patch_locations = select_patches_for_frames(
            sensitivity_map, n_patches=args.tto_patches, patch_size=7,
        )

        # GT targets
        gt_masks = extract_gt_masks(gt_frames, segnet, device)
        gt_pose = extract_gt_pose_targets(gt_frames, posenet, device)

        # Run TTO in batches
        N = args.n_frames
        P = N // 2
        batch_pairs = 50 if not args.smoke else 10
        n_batches = math.ceil(P / batch_pairs)
        current_frames = torch.zeros_like(gt_tensor)

        for batch_idx in range(n_batches):
            pair_start = batch_idx * batch_pairs
            pair_end = min(pair_start + batch_pairs, P)
            frame_start = 2 * pair_start
            frame_end = 2 * pair_end

            batch_result = run_sparse_tto_batch(
                gt_frames_batch=gt_tensor[frame_start:frame_end],
                patch_locations=patch_locations,
                posenet=posenet,
                segnet=segnet,
                gt_pose_targets=gt_pose[pair_start:pair_end],
                gt_seg_masks=gt_masks[frame_start:frame_end],
                device=device,
                steps_per_restart=args.tto_steps,
                n_restarts=args.tto_restarts,
            )
            current_frames[frame_start:frame_end] = batch_result

        dt = time.monotonic() - t0
        print(f"[stage 1] GT-Sparse TTO done in {dt:.1f}s")

    # Score after stage 1
    s1_score = compute_proxy_score(current_frames, gt_frames, posenet, segnet, device,
                                   eval_roundtrip=args.eval_roundtrip)
    stage_results["after_tto"] = s1_score
    print(f"[stage 1] score={s1_score['score']:.6f} | "
          f"pose={s1_score['pose']:.6f} | seg={s1_score['seg']:.6f}")

    # Save checkpoint
    torch.save(current_frames.to(torch.uint8), output_dir / "after_tto.pt")

    # ── Stage 2: Post-filter ─────────────────────────────────────────────
    if not args.skip_postfilter and Path(args.postfilter_path).exists():
        print("\n[stage 2] Applying post-filter...")
        t0 = time.monotonic()
        current_frames = apply_postfilter(current_frames, args.postfilter_path, device)
        dt = time.monotonic() - t0
        print(f"[stage 2] Post-filter done in {dt:.1f}s")

        s2_score = compute_proxy_score(current_frames, gt_frames, posenet, segnet, device,
                                       eval_roundtrip=args.eval_roundtrip)
        stage_results["after_postfilter"] = s2_score
        print(f"[stage 2] score={s2_score['score']:.6f} | "
              f"pose={s2_score['pose']:.6f} | seg={s2_score['seg']:.6f}")

        torch.save(current_frames.to(torch.uint8), output_dir / "after_postfilter.pt")
    else:
        if args.skip_postfilter:
            print("\n[stage 2] Post-filter SKIPPED (--skip-postfilter)")
        else:
            print(f"\n[stage 2] Post-filter SKIPPED (not found: {args.postfilter_path})")

    # ── Final summary ────────────────────────────────────────────────────
    t_total = time.monotonic() - t_total_start

    print(f"\n{'=' * 72}")
    print("STACKED PIPELINE RESULTS")
    print(f"{'=' * 72}")
    for stage_name, score_dict in stage_results.items():
        print(f"  {stage_name:25s}: score={score_dict['score']:.6f} | "
              f"pose={score_dict['pose']:.6f} | seg={score_dict['seg']:.6f}")
    print(f"  Total time: {t_total:.1f}s")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 72}")

    # Save final frames and results
    torch.save(current_frames.to(torch.uint8), output_dir / "final_frames.pt")
    results = {
        "stages": stage_results,
        "config": {
            "n_frames": args.n_frames,
            "tto_patches": args.tto_patches,
            "tto_restarts": args.tto_restarts,
            "tto_steps": args.tto_steps,
            "skip_postfilter": args.skip_postfilter,
            "eval_roundtrip": args.eval_roundtrip,
        },
        "timing": {"total_s": round(t_total, 2)},
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
