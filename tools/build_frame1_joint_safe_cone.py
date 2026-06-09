#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the frame1 JOINT SAFE CONE for N real pairs from upstream/videos/0.mkv.

For every non-overlapping frame pair (frame0, frame1) this decodes the GT, loads
the REAL upstream SegNet + PoseNet (CPU, $0), measures the SegNet frame1 boundary
margin/slope (the seg-safe half-cone) and the differentiable PoseNet frame1
pixel-Jacobian (the pose-null half-cone), and intersects them into the per-pixel
joint cone radius + per-SegNet-class-region aggregates. It writes per-pair cone
maps (compressed .npz) to the durable SSD tier plus a summary JSON, and runs the
behavioral falsification (perturb INSIDE the cone -> d_seg stable + small d_pose;
perturb OUTSIDE the fragile set -> measurable d_seg/d_pose movement).

REUSE (orphan inventory 2026-06-09): the two half-cones are reused from
``tac.optimization.frame1_joint_safe_cone`` (which reuses the verified z8
``segnet_boundary_pixel_saliency`` / ``posenet_pixel_jacobian_norm``). No
half-cone is rebuilt here; this is the thin actuator.

ALL outputs are ``[macOS-CPU advisory]`` / mechanism-only, non-promotable. Local
macOS-CPU is NOT 1:1 contest hardware; the cone proposes radii, the full-video
exact DistortionNet replay ratifies. $0 local, NO cloud, NO paid GPU.

Auto-cleanup: per-pair .npz maps land on the SSD tier with a committed manifest
(path + bytes + sha256); they are deterministically rebuildable from this exact
command, so they may be cold-stored/deleted with the manifest preserved.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

UPSTREAM = REPO_ROOT / "upstream"
DEFAULT_VIDEO = UPSTREAM / "videos" / "0.mkv"

# Durable SSD tiers in operator priority order (CLAUDE.md storage waterfall).
_SSD_TIERS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _default_output_dir() -> Path:
    stamp = _utc_stamp()
    for tier in _SSD_TIERS:
        if tier.is_dir():
            return tier / f"frame1_joint_safe_cone_{stamp}"
    return REPO_ROOT / "experiments" / "results" / f"frame1_joint_safe_cone_{stamp}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--num-pairs", type=int, default=8, help="number of frame pairs")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="durable output dir (default: SSD tier)")
    parser.add_argument("--d-pose", type=float, default=3.4e-5,
                        help="operating-point d_pose for the pose AIL coupling gain")
    parser.add_argument("--seg-margin-tol", type=float, default=0.5)
    parser.add_argument("--pose-response-tol", type=float, default=1e-3)
    parser.add_argument("--fragile-radius-threshold", type=float, default=0.5)
    parser.add_argument("--no-validate", action="store_true",
                        help="skip the behavioral inside/outside falsification")
    parser.add_argument("--save-maps", action="store_true",
                        help="write per-pair .npz cone maps (default: summary only)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    import torch

    from tac.data import decode_video
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.optimization.frame1_joint_safe_cone import (
        Frame1ConeConfig,
        compute_frame1_joint_safe_cone,
        validate_cone_behaviorally,
    )
    from tac.repo_io import json_text, sha256_bytes, write_json_artifact

    args = parse_args(argv)
    video = _resolve(args.video)
    if not video.is_file():
        print(f"FATAL: video not found: {video}", file=sys.stderr)
        return 2
    n = int(args.num_pairs)
    if n < 1:
        print("FATAL: --num-pairs must be >= 1", file=sys.stderr)
        return 2

    out_dir = _resolve(args.output_dir) if args.output_dir else _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load real scorers ($0 CPU) + make PoseNet YUV6 gradient-reachable.
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    patch_upstream_yuv6_globally()
    from modules import DistortionNet, PoseNet, SegNet  # type: ignore[import-not-found]
    from safetensors.torch import load_file

    seg = SegNet().eval()
    seg.load_state_dict(load_file(str(UPSTREAM / "models" / "segnet.safetensors"), device="cpu"))
    pose = PoseNet().eval()
    pose.load_state_dict(load_file(str(UPSTREAM / "models" / "posenet.safetensors"), device="cpu"))
    dn = None
    if not args.no_validate:
        dn = DistortionNet().eval()
        dn.load_state_dicts(
            str(UPSTREAM / "models" / "posenet.safetensors"),
            str(UPSTREAM / "models" / "segnet.safetensors"),
            "cpu",
        )

    frames = decode_video(str(video), target_h=384, target_w=512, max_frames=2 * n)
    if len(frames) < 2 * n:
        print(f"FATAL: decoded {len(frames)} frames; need {2 * n}", file=sys.stderr)
        return 2

    cfg = Frame1ConeConfig(
        d_pose=float(args.d_pose),
        seg_margin_tol=float(args.seg_margin_tol),
        pose_response_tol=float(args.pose_response_tol),
        fragile_radius_threshold=float(args.fragile_radius_threshold),
    )

    t0 = time.time()
    per_pair: list[dict] = []
    map_manifest: list[dict] = []
    for pair_idx in range(n):
        f0 = frames[2 * pair_idx].numpy()
        f1 = frames[2 * pair_idx + 1].numpy()
        gt = np.stack([f0, f1], axis=0)  # (2, H, W, C) uint8 [0, 255]
        pair = torch.from_numpy(gt[None]).float()  # (1, 2, H, W, C)
        cone = compute_frame1_joint_safe_cone(
            segnet=seg, posenet=pose, pair_btchwc_unit255=pair, config=cfg
        )
        row = {
            "pair_index": pair_idx,
            "summary": cone.summary,
            "per_region": {str(k): v for k, v in cone.per_region.items()},
        }
        if not args.no_validate and dn is not None:
            row["validation"] = validate_cone_behaviorally(
                distortion_net=dn, gt_pair_btchwc_unit255=pair, cone=cone
            )
        per_pair.append(row)

        if args.save_maps:
            buf = _npz_bytes(
                joint_cone_radius=cone.joint_cone_radius.astype(np.float32),
                seg_margin=cone.seg_margin.astype(np.float32),
                seg_margin_budget=cone.seg_margin_budget.astype(np.float32),
                pose_jacobian_norm=cone.pose_jacobian_norm.astype(np.float32),
                pose_budget=cone.pose_budget.astype(np.float32),
                joint_sensitivity=cone.joint_sensitivity.astype(np.float32),
                fragile_cone_mask=cone.fragile_cone_mask,
                seg_argmax_class=cone.seg_argmax_class.astype(np.int16),
            )
            map_path = out_dir / f"cone_pair_{pair_idx:05d}.npz"
            if map_path.exists() and not args.overwrite:
                print(f"FATAL: refusing to overwrite {map_path}", file=sys.stderr)
                return 2
            map_path.write_bytes(buf)
            map_manifest.append({
                "pair_index": pair_idx,
                "path": str(map_path),
                "bytes": len(buf),
                "sha256": sha256_bytes(buf),
            })

    elapsed = time.time() - t0

    # Aggregate over pairs.
    def _mean(key: str) -> float:
        vals = [p["summary"][key] for p in per_pair if key in p["summary"]]
        return float(sum(vals) / len(vals)) if vals else 0.0

    agg = {
        "mean_usable_budget_fraction": _mean("usable_budget_fraction"),
        "mean_empty_cone_fraction": _mean("empty_cone_fraction"),
        "mean_pose_binds_fraction": _mean("pose_binds_fraction"),
        "mean_seg_binds_fraction": _mean("seg_binds_fraction"),
        "mean_pose_null_fraction": _mean("pose_null_fraction"),
        "pose_ail_gain": _mean("pose_ail_gain"),
    }
    val_agg = None
    if not args.no_validate:
        disc = [p["validation"]["cone_discriminates"] for p in per_pair if "validation" in p]
        seg_ratios = [p["validation"]["seg_discrimination_ratio"] for p in per_pair if "validation" in p]
        pose_ratios = [p["validation"]["pose_discrimination_ratio"] for p in per_pair if "validation" in p]
        in_seg = [p["validation"]["inside_seg_delta"] for p in per_pair if "validation" in p]
        out_seg = [p["validation"]["outside_seg_delta"] for p in per_pair if "validation" in p]
        in_pose = [p["validation"]["inside_pose_delta"] for p in per_pair if "validation" in p]
        out_pose = [p["validation"]["outside_pose_delta"] for p in per_pair if "validation" in p]
        val_agg = {
            "n_pairs_validated": len(disc),
            "n_pairs_discriminate": int(sum(disc)),
            "all_pairs_discriminate": bool(all(disc)) if disc else False,
            "median_seg_discrimination_ratio": float(np.median(seg_ratios)) if seg_ratios else 0.0,
            "median_pose_discrimination_ratio": float(np.median(pose_ratios)) if pose_ratios else 0.0,
            "mean_inside_seg_delta": float(np.mean(in_seg)) if in_seg else 0.0,
            "mean_outside_seg_delta": float(np.mean(out_seg)) if out_seg else 0.0,
            "mean_inside_pose_delta": float(np.mean(in_pose)) if in_pose else 0.0,
            "mean_outside_pose_delta": float(np.mean(out_pose)) if out_pose else 0.0,
        }

    summary_payload = {
        "schema": "frame1_joint_safe_cone_cli_result.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "video": str(video),
        "num_pairs": n,
        "elapsed_seconds": round(elapsed, 2),
        "config": {
            "d_pose": cfg.d_pose,
            "seg_margin_tol": cfg.seg_margin_tol,
            "pose_response_tol": cfg.pose_response_tol,
            "fragile_radius_threshold": cfg.fragile_radius_threshold,
        },
        "aggregate": agg,
        "validation_aggregate": val_agg,
        "per_pair": per_pair,
        "map_manifest": map_manifest,
        "evidence_grade": "macOS-CPU advisory",
        "axis_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    summary_path = out_dir / "frame1_joint_safe_cone_summary.json"
    exp_sha = None
    if summary_path.is_file() and args.overwrite:
        from tac.repo_io import sha256_file

        exp_sha = sha256_file(summary_path)
    write_json_artifact(
        summary_path, summary_payload,
        allow_overwrite=bool(args.overwrite), expected_existing_sha256=exp_sha,
    )

    print(json_text({
        "schema": "frame1_joint_safe_cone_cli_stdout.v1",
        "summary_path": str(summary_path),
        "num_pairs": n,
        "maps_written": len(map_manifest),
        "aggregate": agg,
        "validation_aggregate": val_agg,
        "score_claim": False,
        "promotion_eligible": False,
    }), end="")
    return 0


def _npz_bytes(**arrays) -> bytes:
    import io

    import numpy as np

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


if __name__ == "__main__":
    raise SystemExit(main())
