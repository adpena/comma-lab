# SPDX-License-Identifier: MIT
"""Build the EVALUATOR RESPONSE ATLAS over the full 600-pair contest video.

For every non-overlapping frame pair ``(frame0, frame1)`` of
``upstream/videos/0.mkv`` this:

  1. Decodes the GT pair (PyAV).
  2. Computes the #35 JOINT SAFE CONE with the REAL CPU-torch scorers
     (SegNet argmax-flip margin field + PoseNet frame1-channel pixel-Jacobian +
     joint cone radius) — the per-pair scorer-sensitivity fields. NEVER MPS.
  3. Summarises the cone into one typed :class:`AtlasPairRow` (an INDEX row:
     pointers + reduced stats; no tensor copy).
  4. Writes the per-pixel cone-map ``.npz`` to the durable SSD tier (the spatial
     budget surface the LF waterfiller reads) + a sha-cited manifest entry.

Then it reduces the 600 per-pair rows into the cross-video headline (MLX
unified-memory kernel, numpy reference fallback) and persists the atlas JSONL
index on the SSD tier.

CRASH-RESUME: per-pair progress is appended to a durable progress JSONL on the
SSD tier; re-running with ``--resume`` skips pairs already computed (read back
from the progress log + on-disk cone maps). Run detached (nohup) for the full
600-pair sweep (~18 min on the M5 Max CPU path).

EVIDENCE: ``[macOS-CPU advisory]`` (scorer forwards CPU-torch) +
``[macOS-MLX research-signal]`` (cross-video reduce). Non-promotable. $0 local,
NO cloud, NO paid GPU, NO MPS.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"
DEFAULT_VIDEO = UPSTREAM / "videos" / "0.mkv"

# Durable SSD tiers in operator priority order (CLAUDE.md storage waterfall).
_SSD_TIERS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (REPO / p)


def _default_output_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for tier in _SSD_TIERS:
        if tier.is_dir():
            return tier / f"evaluator_response_atlas_{stamp}"
    # explicit local fallback under .omx/tmp (never /tmp)
    return REPO / ".omx" / "tmp" / f"evaluator_response_atlas_{stamp}"


def _npz_bytes(**arrays) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--num-pairs", type=int, default=600,
                        help="number of frame pairs (default: 600 = full contest video)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="durable output dir (default: SSD tier)")
    parser.add_argument("--d-pose", type=float, default=3.4e-5,
                        help="operating-point d_pose for the pose AIL gain")
    parser.add_argument("--seg-margin-tol", type=float, default=0.5)
    parser.add_argument("--pose-response-tol", type=float, default=1e-3)
    parser.add_argument("--fragile-radius-threshold", type=float, default=0.5)
    parser.add_argument("--no-save-maps", action="store_true",
                        help="do not write per-pair .npz cone maps (index only)")
    parser.add_argument("--resume", action="store_true",
                        help="skip pairs already in the progress log + on disk")
    parser.add_argument("--prefer-numpy-reduce", action="store_true",
                        help="use numpy reduction (default: MLX unified memory)")
    parser.add_argument("--progress-every", type=int, default=10,
                        help="emit a progress line every N pairs")
    return parser.parse_args(argv)


def _load_progress(progress_path: Path) -> set[int]:
    done: set[int] = set()
    if not progress_path.is_file():
        return done
    for line in progress_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("status") == "done" and "pair_index" in obj:
            done.add(int(obj["pair_index"]))
    return done


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    import torch

    from tac.data import decode_video
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.optimization.evaluator_response_atlas import (
        AtlasPairRow,
        build_atlas,
        build_atlas_row_from_cone,
    )
    from tac.optimization.frame1_joint_safe_cone import (
        Frame1ConeConfig,
        compute_frame1_joint_safe_cone,
    )
    from tac.repo_io import sha256_bytes, write_json_artifact

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
    maps_dir = out_dir / "cone_maps"
    if not args.no_save_maps:
        maps_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "atlas_progress.jsonl"

    done = _load_progress(progress_path) if args.resume else set()
    if done:
        print(f"[resume] {len(done)} pairs already done; skipping them", flush=True)

    # Load real scorers ($0 CPU) + make PoseNet YUV6 gradient-reachable. NEVER MPS.
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    patch_upstream_yuv6_globally()
    from modules import PoseNet, SegNet  # type: ignore[import-not-found]
    from safetensors.torch import load_file

    seg = SegNet().eval()
    seg.load_state_dict(load_file(str(UPSTREAM / "models" / "segnet.safetensors"), device="cpu"))
    pose = PoseNet().eval()
    pose.load_state_dict(load_file(str(UPSTREAM / "models" / "posenet.safetensors"), device="cpu"))

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

    rows: list[AtlasPairRow] = []
    map_manifest: list[dict] = []
    # Load already-computed rows from disk (resume).
    rows_index_path = out_dir / "atlas_rows_index.jsonl"
    if args.resume and rows_index_path.is_file():
        for line in rows_index_path.read_text().splitlines():
            if line.strip():
                rows.append(AtlasPairRow.from_json_obj(json.loads(line)))

    t0 = time.time()
    rows_fh = rows_index_path.open("a", encoding="utf-8")
    progress_fh = progress_path.open("a", encoding="utf-8")
    try:
        for pair_idx in range(n):
            if pair_idx in done:
                continue
            f0 = frames[2 * pair_idx].numpy()
            f1 = frames[2 * pair_idx + 1].numpy()
            gt = np.stack([f0, f1], axis=0)  # (2, H, W, C) uint8 [0, 255]
            pair = torch.from_numpy(gt[None]).float()  # (1, 2, H, W, C)
            cone = compute_frame1_joint_safe_cone(
                segnet=seg, posenet=pose, pair_btchwc_unit255=pair, config=cfg
            )

            cone_map_path = None
            cone_map_sha = None
            if not args.no_save_maps:
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
                map_path = maps_dir / f"cone_pair_{pair_idx:05d}.npz"
                map_path.write_bytes(buf)
                cone_map_path = str(map_path)
                cone_map_sha = sha256_bytes(buf)
                map_manifest.append({
                    "pair_index": pair_idx,
                    "path": cone_map_path,
                    "bytes": len(buf),
                    "sha256": cone_map_sha,
                })

            row = build_atlas_row_from_cone(
                pair_index=pair_idx,
                cone=cone,
                cone_map_path=cone_map_path,
                cone_map_sha256=cone_map_sha,
                compute_path="cpu_torch",
            )
            rows.append(row)
            rows_fh.write(json.dumps(row.to_json_obj(), sort_keys=True) + "\n")
            rows_fh.flush()
            progress_fh.write(json.dumps({
                "status": "done",
                "pair_index": pair_idx,
                "pair_budget": row.joint_cone_summary.pair_budget,
                "fragile_fraction": row.seg_margin_field_stats.fragile_fraction,
                "pose_binds_fraction": row.joint_cone_summary.pose_binds_fraction,
                "written_at_utc": datetime.now(UTC).isoformat(),
            }) + "\n")
            progress_fh.flush()

            if (pair_idx + 1) % int(args.progress_every) == 0 or pair_idx == n - 1:
                el = time.time() - t0
                ncomp = pair_idx + 1 - len(done)
                rate = el / max(ncomp, 1)
                eta = rate * (n - pair_idx - 1)
                print(
                    f"[atlas] pair {pair_idx + 1}/{n} "
                    f"budget={row.joint_cone_summary.pair_budget:.1f} "
                    f"fragile={row.seg_margin_field_stats.fragile_fraction:.3f} "
                    f"pose_binds={row.joint_cone_summary.pose_binds_fraction:.3f} "
                    f"| {rate:.2f}s/pair eta={eta/60:.1f}min",
                    flush=True,
                )
    finally:
        rows_fh.close()
        progress_fh.close()

    elapsed = time.time() - t0

    # Cross-video reduction (MLX-first unified memory; numpy reference fallback).
    atlas = build_atlas(rows, prefer_mlx=not args.prefer_numpy_reduce)
    headline = atlas.headline

    # Persist the atlas JSONL index (header + per-pair rows; pointers, no tensors).
    atlas_jsonl_path = out_dir / "evaluator_response_atlas.jsonl"
    atlas_jsonl_path.write_text("\n".join(atlas.to_jsonl_lines()) + "\n", encoding="utf-8")

    summary_payload = {
        "schema": "evaluator_response_atlas_cli_result.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "video": str(video),
        "num_pairs": len(rows),
        "elapsed_seconds": round(elapsed, 2),
        "scorer_compute_path": "cpu_torch",  # NEVER mps
        "reduce_path": headline.get("reduce_path"),
        "config": {
            "d_pose": cfg.d_pose,
            "seg_margin_tol": cfg.seg_margin_tol,
            "pose_response_tol": cfg.pose_response_tol,
            "fragile_radius_threshold": cfg.fragile_radius_threshold,
        },
        "headline": headline,
        "atlas_jsonl_path": str(atlas_jsonl_path),
        "rows_index_path": str(rows_index_path),
        "n_cone_maps": len(map_manifest),
        "evidence_grade": "macOS-CPU advisory",
        "axis_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    summary_path = out_dir / "evaluator_response_atlas_summary.json"
    write_json_artifact(summary_path, summary_payload)

    # Map manifest (sha-cited producer artifacts).
    if map_manifest:
        write_json_artifact(out_dir / "cone_map_manifest.json", {
            "schema": "evaluator_response_atlas_cone_map_manifest.v1",
            "maps": map_manifest,
        })

    print("\n=== ATLAS HEADLINE (600-pair) ===", flush=True)
    print(f"  n_pairs: {headline['n_pairs']}", flush=True)
    print(f"  reduce_path: {headline['reduce_path']}", flush=True)
    print(f"  video_usable_budget_fraction: {headline['video_usable_budget_fraction']:.4f}", flush=True)
    print(f"  video_pose_binds_fraction: {headline['video_pose_binds_fraction']:.4f}", flush=True)
    print(f"  fragile_mass_gini: {headline['fragile_mass_gini']:.4f}", flush=True)
    print(f"  budget_concentration_gini: {headline['budget_concentration_gini']:.4f}", flush=True)
    print(f"  total_free_budget: {headline['total_free_budget']:.1f}", flush=True)
    print(f"  top10_budget_pair_indices: {headline['top10_budget_pair_indices']}", flush=True)
    print(f"  top10_fragile_pair_indices: {headline['top10_fragile_pair_indices']}", flush=True)
    print(f"\nsummary: {summary_path}", flush=True)
    print(f"atlas index: {atlas_jsonl_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
