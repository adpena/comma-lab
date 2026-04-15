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
import torch.nn.functional as F


def find_project_root() -> Path:
    """Walk up from this file to find the project root (contains src/)."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "src").is_dir() and (p / "upstream").is_dir():
            return p
        p = p.parent
    raise RuntimeError("Cannot find project root (expected src/ and upstream/ dirs)")


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

    p.add_argument("--simulate-resize", action="store_true",
                   help="Simulate official scorer resolution round-trip")
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

    return filtered


def compute_proxy_score(
    frames: torch.Tensor,
    gt_frames: list[torch.Tensor],
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    rate: float = 0.0,
    batch_size: int = 16,
    simulate_resize: bool = False,
) -> dict:
    """Compute proxy score matching official scorer formula."""
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W

    N = frames.shape[0]
    P = N // 2
    total_pose, total_seg, n_pairs = 0.0, 0.0, 0

    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)

        cand_pairs, gt_pairs = [], []
        for k in range(start, end):
            cand_pairs.append(torch.stack([frames[2 * k], frames[2 * k + 1]], dim=0))
            gt_pairs.append(torch.stack([
                gt_frames[2 * k].float(), gt_frames[2 * k + 1].float(),
            ], dim=0))

        cand_t = torch.stack(cand_pairs).to(device)
        gt_t = torch.stack(gt_pairs).to(device)

        cand_chw = cand_t.permute(0, 1, 4, 2, 3).contiguous()
        gt_chw = gt_t.permute(0, 1, 4, 2, 3).contiguous()

        B, T, C, H, W = cand_chw.shape
        if H != SEGNET_INPUT_H or W != SEGNET_INPUT_W:
            cand_flat = cand_chw.reshape(B * T, C, H, W)
            gt_flat = gt_chw.reshape(B * T, C, H, W)
            cand_flat = F.interpolate(cand_flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                                      mode="bilinear", align_corners=False)
            gt_flat = F.interpolate(gt_flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                                    mode="bilinear", align_corners=False)
            cand_chw = cand_flat.reshape(B, T, C, SEGNET_INPUT_H, SEGNET_INPUT_W)
            gt_chw = gt_flat.reshape(B, T, C, SEGNET_INPUT_H, SEGNET_INPUT_W)

        cand_chw = cand_chw.round().clamp(0, 255)

        if simulate_resize:
            flat = cand_chw.reshape(-1, *cand_chw.shape[2:])
            flat = F.interpolate(flat, size=(874, 1164), mode="bilinear", align_corners=False)
            flat = flat.round().clamp(0, 255)
            flat = F.interpolate(flat, size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
                                 mode="bilinear", align_corners=False)
            cand_chw = flat.reshape(B, T, *flat.shape[1:])

        with torch.no_grad():
            fp_in = posenet.preprocess_input(cand_chw)
            gp_in = posenet.preprocess_input(gt_chw)
            fp_out = posenet(fp_in)
            gp_out = posenet(gp_in)
            pose_mse = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean(dim=-1)
            total_pose += pose_mse.sum().item()

            fs_in = segnet.preprocess_input(cand_chw)
            gs_in = segnet.preprocess_input(gt_chw)
            fs_out = segnet(fs_in)
            gs_out = segnet(gs_in)
            diff = (fs_out.argmax(dim=1) != gs_out.argmax(dim=1)).float()
            seg_disagree = diff.mean(dim=tuple(range(1, diff.ndim)))
            total_seg += seg_disagree.sum().item()
            n_pairs += B

    avg_pose = total_pose / max(n_pairs, 1)
    avg_seg = total_seg / max(n_pairs, 1)
    score = 100.0 * avg_seg + math.sqrt(10.0 * avg_pose) + 25.0 * rate

    return {
        "score": score, "pose": avg_pose, "seg": avg_seg, "rate": rate,
        "pose_contribution": math.sqrt(10.0 * avg_pose),
        "seg_contribution": 100.0 * avg_seg,
        "rate_contribution": 25.0 * rate,
        "n_pairs": n_pairs,
    }


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

    video_path = args.video or str(upstream / "videos" / "0.mkv")

    if args.postfilter_path is None:
        args.postfilter_path = str(root / "submissions" / "robust_current" / "postfilter_int8.pt")

    t_total_start = time.monotonic()
    stage_results = {}

    # ── Load scorers ─────────────────────────────────────────────────────
    print("[setup] Loading scorers...")
    from tac.scorer import load_scorers
    posenet, segnet = load_scorers(
        posenet_path=upstream / "models" / "posenet.safetensors",
        segnet_path=upstream / "models" / "segnet.safetensors",
        device=str(device),
        upstream_dir=str(upstream),
    )

    # ── Decode GT video ──────────────────────────────────────────────────
    print(f"[setup] Decoding GT video ({args.n_frames} frames)...")
    from tac.data import decode_video
    from tac.camera import SEGNET_INPUT_H, SEGNET_INPUT_W

    gt_frames_full = decode_video(video_path, target_h=SEGNET_INPUT_H, target_w=SEGNET_INPUT_W)
    gt_frames = gt_frames_full[:args.n_frames]
    args.n_frames = len(gt_frames) - (len(gt_frames) % 2)
    gt_frames = gt_frames[:args.n_frames]

    # GT baseline
    gt_tensor = torch.stack(gt_frames).float()
    gt_score = compute_proxy_score(gt_tensor, gt_frames, posenet, segnet, device,
                                   simulate_resize=args.simulate_resize)
    stage_results["gt_baseline"] = gt_score
    print(f"[GT baseline] score={gt_score['score']:.6f}")

    # ── Stage 1: Get frames (GT-TTO or precomputed) ─────────────────────
    if args.frames_path and Path(args.frames_path).exists():
        print(f"\n[stage 1] Loading precomputed frames: {args.frames_path}")
        current_frames = torch.load(args.frames_path, map_location="cpu", weights_only=True).float()
        # Truncate to n_frames if needed
        current_frames = current_frames[:args.n_frames]
    else:
        print(f"\n[stage 1] Running GT-Sparse TTO...")
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
                                   simulate_resize=args.simulate_resize)
    stage_results["after_tto"] = s1_score
    print(f"[stage 1] score={s1_score['score']:.6f} | "
          f"pose={s1_score['pose']:.6f} | seg={s1_score['seg']:.6f}")

    # Save checkpoint
    torch.save(current_frames.to(torch.uint8), output_dir / "after_tto.pt")

    # ── Stage 2: Post-filter ─────────────────────────────────────────────
    if not args.skip_postfilter and Path(args.postfilter_path).exists():
        print(f"\n[stage 2] Applying post-filter...")
        t0 = time.monotonic()
        current_frames = apply_postfilter(current_frames, args.postfilter_path, device)
        dt = time.monotonic() - t0
        print(f"[stage 2] Post-filter done in {dt:.1f}s")

        s2_score = compute_proxy_score(current_frames, gt_frames, posenet, segnet, device,
                                       simulate_resize=args.simulate_resize)
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
            "simulate_resize": args.simulate_resize,
        },
        "timing": {"total_s": round(t_total, 2)},
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
