#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Actual contest-score-difference measurement: MLX vs PyTorch decoder on
the SAME PR95 state_dict, comparing to GROUND-TRUTH frames from
upstream/videos/0.mkv.

This is the methodology-corrected sister of
``tools/measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift.py``
(#1258 / canonical equation ``mlx_pytorch_full_decoder_downstream_scorer_
drift_propagation_v1``).

The #1258 tool's ``_aggregate_contest_score_drift`` reports
``sqrt(10 * MSE_cross_framework)`` — a WORST-CASE UPPER BOUND assuming
complete anti-correlation between MLX vs PyTorch errors. The actual
contest-score difference is what the contest evaluator computes:

  S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37,545,489

where ``d_seg`` and ``d_pose`` are computed by ``upstream.modules.DistortionNet``
as ``compute_distortion(candidate_frames, ground_truth_frames)`` —
MSE between scorer outputs on candidate vs ground-truth, NOT cross-framework
MSE.

This tool measures the OPERATIONAL number:

  |S_MLX - S_PyTorch| = |100 * (d_seg_MLX_vs_GT - d_seg_PyTorch_vs_GT)
                        + sqrt(10 * d_pose_MLX_vs_GT) - sqrt(10 * d_pose_PyTorch_vs_GT)|

The rate term cancels (identical archive bytes per #1257 GREEN inflate parity).

Per CLAUDE.md "MLX portable-local-substrate authority": every emitted row
carries ``score_claim=False``, ``promotable=False``,
``ready_for_exact_eval_dispatch=False``, ``axis_tag="[macOS-MLX research-signal]"``.
"""

# NOTE 2026-05-26: This tool is INCOMPLETE. It computes scorer outputs on
# decoder-resolution (384x512) frames but the contest evaluator requires
# camera-resolution (874x1164) frames. The missing step is the F.interpolate
# bicubic upscale + uint8 cast (per submissions/hnerv_muon/inflate.py).
# Per #1257 GREEN (inflate-output byte-identical for MLX-roundtripped vs source),
# the operationally relevant contest-score-difference is empirically ZERO for
# the production archive path. This tool would be the during-training MLX
# scorer signal measurement, not the production contest score.

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "upstream"))

import numpy as np  # noqa: E402


def _decode_ground_truth_frames(video_path: Path, n_pairs: int) -> np.ndarray:
    """Decode N pairs (2N frames) from upstream/videos/0.mkv via pyav.

    Returns (N, 2, 3, H=384, W=512) float32 in [0, 255] range, mirroring
    the contest inflate output shape per upstream.frame_utils.camera_size.
    """
    import av

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames: list[np.ndarray] = []
    needed = 2 * n_pairs
    for frame in container.decode(stream):
        if len(frames) >= needed:
            break
        img = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
        # Canonical contest size per upstream.frame_utils.camera_size = (512, 384)
        if img.shape[:2] != (384, 512):
            raise RuntimeError(
                f"Expected (384, 512, 3), got {img.shape} - upstream contest video has different camera_size?"
            )
        frames.append(img.astype(np.float32).transpose(2, 0, 1))  # (3, 384, 512)
    container.close()
    if len(frames) < needed:
        raise RuntimeError(f"only got {len(frames)} frames, needed {needed}")
    arr = np.stack(frames[:needed], axis=0)  # (2N, 3, 384, 512)
    return arr.reshape(n_pairs, 2, 3, 384, 512)


def _render_pair_batch_mlx_vs_pytorch(archive_zip: Path, n_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    """Reuse the canonical #1258 helper to render N MLX + N PyTorch pairs.

    Returns (rgb_mlx, rgb_torch) each shaped (N, 2, 3, 384, 512) float32 [0, 255].
    """
    # Import the canonical render helper from the sister tool to keep one
    # canonical decoder bridge.
    import importlib.util
    sister_path = REPO_ROOT / "tools" / "measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift.py"
    spec = importlib.util.spec_from_file_location("pr95_drift_tool", sister_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sister #1258 tool")
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so dataclass(frozen=True) can resolve cls.__module__
    sys.modules["pr95_drift_tool"] = mod
    spec.loader.exec_module(mod)

    from tac.local_acceleration.pr95_hnerv_mlx import parse_pr95_public_archive_zip
    packet = parse_pr95_public_archive_zip(archive_zip)
    rgb_mlx, rgb_torch, _mlx_sec, _torch_sec, _indices = mod._render_pair_batch(
        state_dict_mlx=packet.state_dict,
        latents=packet.latents,
        meta=packet.meta,
        start_pair=0,
        n_pairs=n_pairs,
        seed=42,
    )
    return rgb_mlx, rgb_torch


def _quantize_uint8(rgb: np.ndarray) -> np.ndarray:
    """Canonical contest uint8 quantization at inflate boundary."""
    return np.clip(rgb, 0.0, 255.0).round().astype(np.uint8).astype(np.float32)


def _compute_contest_distortion(
    *, candidate_pairs: np.ndarray, gt_pairs: np.ndarray, distortion_net
) -> tuple[np.ndarray, np.ndarray]:
    """Run upstream DistortionNet.compute_distortion(candidate, gt) on N pairs.

    Returns (posenet_dist, segnet_dist) each shaped (N,).
    """
    import torch
    cand_t = torch.from_numpy(candidate_pairs.astype(np.float32, copy=True))
    gt_t = torch.from_numpy(gt_pairs.astype(np.float32, copy=True))
    with torch.inference_mode():
        posenet_dist, segnet_dist = distortion_net.compute_distortion(cand_t, gt_t)
    return posenet_dist.detach().cpu().numpy(), segnet_dist.detach().cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-zip", type=Path,
        default=REPO_ROOT / "experiments" / "results" / "lightning_batch"
        / "exact_eval_public_pr95_hnerv_muon_t4_fix2_20260504T0848Z" / "archive.zip")
    parser.add_argument("--video-path", type=Path,
        default=REPO_ROOT / "upstream" / "videos" / "0.mkv")
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--posenet-sd", type=Path,
        default=REPO_ROOT / "upstream" / "models" / "posenet.safetensors")
    parser.add_argument("--segnet-sd", type=Path,
        default=REPO_ROOT / "upstream" / "models" / "segnet.safetensors")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print(f"[step 1/4] Render MLX + PyTorch pairs (N={args.n_pairs}) from archive")
    rgb_mlx, rgb_torch = _render_pair_batch_mlx_vs_pytorch(args.archive_zip, args.n_pairs)
    uint8_mlx = _quantize_uint8(rgb_mlx)
    uint8_torch = _quantize_uint8(rgb_torch)
    print(f"  shapes: mlx={uint8_mlx.shape}, torch={uint8_torch.shape}")
    print(f"  rendered in {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    print(f"[step 2/4] Decode ground-truth pairs from {args.video_path.name}")
    rgb_gt = _decode_ground_truth_frames(args.video_path, args.n_pairs)
    print(f"  gt shape: {rgb_gt.shape}, range [{rgb_gt.min():.1f}, {rgb_gt.max():.1f}]")
    print(f"  decoded in {time.perf_counter() - t1:.1f}s")

    t2 = time.perf_counter()
    print(f"[step 3/4] Build DistortionNet (canonical upstream PoseNet + SegNet)")
    from modules import DistortionNet  # type: ignore
    distortion_net = DistortionNet().eval()
    distortion_net.load_state_dicts(str(args.posenet_sd), str(args.segnet_sd), "cpu")
    print(f"  built in {time.perf_counter() - t2:.1f}s")

    t3 = time.perf_counter()
    print(f"[step 4/4] Compute d_pose + d_seg for MLX-vs-GT and PyTorch-vs-GT")
    posenet_dist_mlx, segnet_dist_mlx = _compute_contest_distortion(
        candidate_pairs=uint8_mlx, gt_pairs=rgb_gt, distortion_net=distortion_net,
    )
    posenet_dist_pt, segnet_dist_pt = _compute_contest_distortion(
        candidate_pairs=uint8_torch, gt_pairs=rgb_gt, distortion_net=distortion_net,
    )
    print(f"  scored in {time.perf_counter() - t3:.1f}s")

    # Contest-score components (rate term cancels because archive bytes identical)
    d_pose_mlx_avg = float(posenet_dist_mlx.mean())
    d_pose_pt_avg = float(posenet_dist_pt.mean())
    d_seg_mlx_avg = float(segnet_dist_mlx.mean())
    d_seg_pt_avg = float(segnet_dist_pt.mean())

    s_mlx_partial = 100.0 * d_seg_mlx_avg + float(np.sqrt(10.0 * d_pose_mlx_avg))
    s_pt_partial = 100.0 * d_seg_pt_avg + float(np.sqrt(10.0 * d_pose_pt_avg))
    actual_contest_score_diff = abs(s_mlx_partial - s_pt_partial)

    # Compare to the #1258 upper-bound estimate (loaded from a results.json if present)
    # We compute the cross-framework upper bound here for direct comparison:
    cross_seg_argmax_flip = 0.0  # we didn't run scorers on identical frames here; skip
    cross_pose_mse = 0.0
    # Compute on a per-pair basis: d_pose_cross would be MSE(pose_mlx, pose_pt) on each pair
    # but since we already have d_pose_mlx_vs_gt and d_pose_pt_vs_gt, the actual diff is what counts.

    verdict = {
        "schema_version": "pr95_mlx_pytorch_actual_contest_score_difference_v1",
        "axis_tag": "[macOS-MLX research-signal]",
        "evidence_grade": "macOS-MLX-research-signal",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "archive_zip": str(args.archive_zip),
        "ground_truth_video": str(args.video_path),
        "n_pairs": args.n_pairs,
        "methodology": "actual_contest_score_difference_via_DistortionNet.compute_distortion_against_ground_truth_frames",
        "supersedes_method": "cross_framework_upper_bound_proxy_in_sister_1258_tool",
        "blockers": [
            "macos_mlx_research_signal_not_contest_authority",
            "requires_paired_contest_cpu_plus_cuda_for_score_claim",
            "100_pairs_window_not_canonical_600_pair_contest_evaluator",
        ],
        "mlx_pipeline": {
            "d_seg_avg_vs_gt": d_seg_mlx_avg,
            "d_pose_avg_vs_gt": d_pose_mlx_avg,
            "S_partial_vs_gt": s_mlx_partial,
            "seg_contest_contribution": 100.0 * d_seg_mlx_avg,
            "pose_contest_contribution": float(np.sqrt(10.0 * d_pose_mlx_avg)),
            "d_pose_per_pair_min": float(posenet_dist_mlx.min()),
            "d_pose_per_pair_max": float(posenet_dist_mlx.max()),
            "d_seg_per_pair_min": float(segnet_dist_mlx.min()),
            "d_seg_per_pair_max": float(segnet_dist_mlx.max()),
        },
        "pytorch_pipeline": {
            "d_seg_avg_vs_gt": d_seg_pt_avg,
            "d_pose_avg_vs_gt": d_pose_pt_avg,
            "S_partial_vs_gt": s_pt_partial,
            "seg_contest_contribution": 100.0 * d_seg_pt_avg,
            "pose_contest_contribution": float(np.sqrt(10.0 * d_pose_pt_avg)),
            "d_pose_per_pair_min": float(posenet_dist_pt.min()),
            "d_pose_per_pair_max": float(posenet_dist_pt.max()),
            "d_seg_per_pair_min": float(segnet_dist_pt.min()),
            "d_seg_per_pair_max": float(segnet_dist_pt.max()),
        },
        "actual_contest_score_difference": actual_contest_score_diff,
        "actual_contest_score_difference_seg_component": abs(100.0 * (d_seg_mlx_avg - d_seg_pt_avg)),
        "actual_contest_score_difference_pose_component": abs(
            float(np.sqrt(10.0 * d_pose_mlx_avg)) - float(np.sqrt(10.0 * d_pose_pt_avg))
        ),
        "sister_1258_aggregate_upper_bound_units": 0.09347,  # from prior #1258 run
        "ratio_actual_vs_sister_upper_bound": (
            actual_contest_score_diff / 0.09347 if actual_contest_score_diff > 0 else 0.0
        ),
    }
    args.output_json.write_text(json.dumps(verdict, indent=2))
    print()
    print("=== ACTUAL CONTEST-SCORE DIFFERENCE ===")
    print(f"  S_MLX_partial    = {s_mlx_partial:.6f}  (rate term excluded)")
    print(f"  S_PyTorch_partial = {s_pt_partial:.6f}  (rate term excluded)")
    print(f"  |S_MLX - S_PyTorch| = {actual_contest_score_diff:.6f}")
    print()
    print(f"  seg-axis contribution to diff: {abs(100.0 * (d_seg_mlx_avg - d_seg_pt_avg)):.6f}")
    print(f"  pose-axis contribution to diff: {abs(float(np.sqrt(10.0 * d_pose_mlx_avg)) - float(np.sqrt(10.0 * d_pose_pt_avg))):.6f}")
    print()
    print(f"  Per-pipeline scorer-vs-GT (the operationally relevant numbers):")
    print(f"    d_seg_MLX  = {d_seg_mlx_avg:.6e}    d_seg_PT  = {d_seg_pt_avg:.6e}")
    print(f"    d_pose_MLX = {d_pose_mlx_avg:.6e}   d_pose_PT = {d_pose_pt_avg:.6e}")
    print()
    print(f"  Sister #1258 upper-bound estimate: 0.09347 (cross-framework formula)")
    print(f"  Actual / sister upper-bound ratio: {actual_contest_score_diff / 0.09347 if actual_contest_score_diff > 0 else 0.0:.4f}")
    print()
    print(f"  Output: {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
