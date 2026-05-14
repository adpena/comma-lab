#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Per-pair sensitivity diagnostic — macOS-CPU advisory only.

Operator directive 2026-05-13 LOCAL HARDWARE MAXIMIZATION SWEEP Stream 2.

For a given archive, inflate via the contest's inflate.sh + run upstream
DistortionNet pair-by-pair to capture per-pair (PoseNet, SegNet) distortions.
Identifies top-50 highest-leverage pairs (largest score-marginal contribution)
and bottom-50 lowest-leverage pairs (free-bytes candidates).

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192:
- evidence_grade = "macOS-CPU-advisory"
- score_claim = False, promotion_eligible = False
- ranking_only = True

The macOS-CPU axis correlates with [contest-CPU] within ~2e-5 per Catalog #192.
Per-pair sensitivity feeds bit-allocator + sensitivity-map hooks (Catalog #125).

Usage:
    .venv/bin/python tools/diagnose_per_pair_sensitivity.py \\
        --archive submissions/a1/archive.zip \\
        --archive-id a1_baseline \\
        --output-dir experiments/results/lane_local_hardware_maximization_sweep_20260513_<UTC>
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "upstream"))

EVIDENCE_GRADE = "macOS-CPU-advisory"
EVIDENCE_TAG = "[macOS-CPU advisory only]"
LANE_ID = "lane_local_hardware_maximization_sweep_20260513"


def _utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _inflate_archive(archive: Path, work_dir: Path, inflate_sh: Path) -> Path:
    """Run inflate.sh archive_dir output_dir file_list -> returns inflated dir.

    archive_dir contains the unpacked contents of archive.zip (per contract).
    """
    archive_dir = work_dir / "archive_dir"
    archive_dir.mkdir(parents=True, exist_ok=True)
    inflated_dir = work_dir / "inflated"
    inflated_dir.mkdir(parents=True, exist_ok=True)
    # Unpack archive.zip into archive_dir (contest contract)
    import zipfile
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(archive_dir)
    # Build file_list: contest expects per-video filenames (no extension)
    file_list_path = work_dir / "file_list.txt"
    # Use the contest's canonical video list
    pub_test_names = (
        REPO_ROOT / "upstream" / "public_test_video_names.txt"
    )
    if pub_test_names.is_file():
        names = [
            ln.strip()
            for ln in pub_test_names.read_text().splitlines()
            if ln.strip()
        ]
    else:
        names = ["0"]
    file_list_path.write_text("\n".join(names) + "\n")
    cmd = [
        "bash",
        str(inflate_sh),
        str(archive_dir),
        str(inflated_dir),
        str(file_list_path),
    ]
    env = os.environ.copy()
    # Use the current interpreter (venv python) for inflate.py
    env["PYTHON"] = sys.executable
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), env=env
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"inflate.sh failed rc={proc.returncode}\nstdout={proc.stdout[:500]}\n"
            f"stderr={proc.stderr[:1500]}"
        )
    return inflated_dir


def _compute_per_pair_distortions(
    inflated_dir: Path, max_pairs: int = 600
) -> dict:
    """Compute per-pair PoseNet + SegNet distortions on macOS-CPU.

    Uses the contest's seq_len=2 non-overlapping pair batching (matches
    upstream/evaluate.py contract).
    """
    import torch
    from frame_utils import (
        AVVideoDataset,
        TensorVideoDataset,
    )
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    device = torch.device("cpu")
    # Load DistortionNet on CPU
    distortion_net = DistortionNet()
    distortion_net.load_state_dicts(
        str(REPO_ROOT / posenet_sd_path),
        str(REPO_ROOT / segnet_sd_path),
        device,
    )
    distortion_net.eval()
    distortion_net.to(device)

    # Build datasets — GT from upstream/videos/0.mkv, comp from inflated_dir
    # Match the contest's public_test_video_names.txt exactly ("0.mkv")
    test_video_names_path = REPO_ROOT / "upstream" / "public_test_video_names.txt"
    if test_video_names_path.is_file():
        test_video_names = [
            ln.strip() for ln in test_video_names_path.read_text().splitlines() if ln.strip()
        ]
    else:
        test_video_names = ["0.mkv"]
    videos_dir = REPO_ROOT / "upstream" / "videos"
    batch_size = 1  # per-pair granularity
    ds_gt = AVVideoDataset(
        test_video_names,
        data_dir=videos_dir,
        batch_size=batch_size,
        device=device,
        num_threads=1,
        seed=0,
    )
    ds_gt.prepare_data()
    ds_comp = TensorVideoDataset(
        test_video_names,
        data_dir=inflated_dir,
        batch_size=batch_size,
        device=device,
        num_threads=1,
        seed=0,
    )
    ds_comp.prepare_data()
    dl_gt = torch.utils.data.DataLoader(ds_gt, batch_size=None, num_workers=0)
    dl_comp = torch.utils.data.DataLoader(ds_comp, batch_size=None, num_workers=0)

    pose_per_pair = []
    seg_per_pair = []
    pair_count = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for (_, _, batch_gt), (_, _, batch_comp) in zip(dl_gt, dl_comp, strict=False):
            batch_gt = batch_gt.to(device)
            batch_comp = batch_comp.to(device)
            pose_d, seg_d = distortion_net.compute_distortion(batch_gt, batch_comp)
            # shape (B,) with B=1
            for i in range(pose_d.shape[0]):
                pose_per_pair.append(float(pose_d[i].item()))
                seg_per_pair.append(float(seg_d[i].item()))
                pair_count += 1
            if pair_count >= max_pairs:
                break
    elapsed = time.perf_counter() - started
    return {
        "pose_per_pair": pose_per_pair,
        "seg_per_pair": seg_per_pair,
        "pair_count": pair_count,
        "elapsed_seconds": elapsed,
    }


def _analyze_per_pair(
    archive: Path,
    pose_per_pair: list,
    seg_per_pair: list,
    archive_bytes: int,
) -> dict:
    """Compute per-pair marginal score contribution + top/bottom rankings.

    Score formula: total = 100*seg_avg + sqrt(10*pose_avg) + 25*rate
    Per-pair marginal contributions:
        seg_marginal_per_pair = 100 / N
        pose_marginal_per_pair = (5 / N) / sqrt(10*pose_avg)
        rate_share_per_pair = (25*rate / N)  [uniform — bytes are global]
    """
    N = len(pose_per_pair)
    pose_avg = sum(pose_per_pair) / N
    seg_avg = sum(seg_per_pair) / N

    seg_term = 100 * seg_avg
    pose_term = math.sqrt(10 * pose_avg) if pose_avg > 0 else 0.0
    seg_marginal_factor = 100.0 / N  # marginal d(seg_term)/d(seg_per_pair)
    pose_marginal_factor = 5.0 / N / math.sqrt(10 * pose_avg) if pose_avg > 0 else 0.0

    # Per-pair total marginal: sum of normalized contributions
    pair_data = []
    for i in range(N):
        seg_contrib = seg_per_pair[i] * seg_marginal_factor
        pose_contrib = pose_per_pair[i] * pose_marginal_factor
        total_marginal = seg_contrib + pose_contrib
        pair_data.append({
            "pair_index": i,
            "seg_distortion": seg_per_pair[i],
            "pose_distortion": pose_per_pair[i],
            "seg_score_contribution": seg_contrib,
            "pose_score_contribution": pose_contrib,
            "total_score_contribution": total_marginal,
        })

    # Sort by total contribution
    pair_data_sorted = sorted(
        pair_data, key=lambda r: r["total_score_contribution"], reverse=True
    )
    top50 = pair_data_sorted[:50]
    bottom50 = pair_data_sorted[-50:]

    # Compute concentration metrics
    total_marginal_sum = sum(p["total_score_contribution"] for p in pair_data)
    top50_share = sum(p["total_score_contribution"] for p in top50) / total_marginal_sum
    bottom50_share = sum(p["total_score_contribution"] for p in bottom50) / total_marginal_sum

    return {
        "N": N,
        "pose_avg": pose_avg,
        "seg_avg": seg_avg,
        "seg_term_100x": seg_term,
        "pose_term_sqrt10x": pose_term,
        "seg_marginal_factor_per_pair": seg_marginal_factor,
        "pose_marginal_factor_per_pair": pose_marginal_factor,
        "total_marginal_sum": total_marginal_sum,
        "top50_concentration_share": top50_share,
        "bottom50_concentration_share": bottom50_share,
        "top50_pairs": top50,
        "bottom50_pairs": bottom50,
        "archive_bytes": archive_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--archive-id", type=str, required=True)
    p.add_argument(
        "--inflate-sh",
        type=Path,
        default=None,
        help="Path to inflate.sh that matches archive grammar. "
             "If omitted, defaults to <archive>.parent/inflate.sh "
             "(adjacent inflate.sh, contest contract).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output dir under experiments/results/<lane>_<UTC>/",
    )
    p.add_argument("--max-pairs", type=int, default=600)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan + verdict skeleton without running inflate or eval.",
    )
    args = p.parse_args(argv)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    out_str = str(out)
    if "/tmp/" in out_str or "/private/tmp/" in out_str or "/var/tmp/" in out_str:
        print(f"FATAL: refusing /tmp persisted output: {out_str}", file=sys.stderr)
        return 2

    archive = args.archive.resolve()
    archive_bytes = archive.stat().st_size
    archive_sha = _sha256_of(archive)

    if args.dry_run:
        plan = {
            "lane_id": LANE_ID,
            "archive": str(archive),
            "archive_id": args.archive_id,
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha,
            "max_pairs": args.max_pairs,
            "output_dir": str(out),
            "dry_run": True,
            "estimated_wall_clock_seconds": 600,  # ~10 min on macOS-CPU
        }
        print(json.dumps(plan, indent=2))
        return 0

    started = time.perf_counter()

    # 1. Inflate archive
    work_dir = out / f"work_{args.archive_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"[diag] inflating {archive} -> {work_dir}")
    inflate_sh = args.inflate_sh
    if inflate_sh is None:
        adj = archive.parent / "inflate.sh"
        if adj.is_file():
            inflate_sh = adj
        else:
            print(
                f"FATAL: --inflate-sh not provided and no adjacent inflate.sh at {adj}",
                file=sys.stderr,
            )
            return 2
    inflated_dir = _inflate_archive(archive, work_dir, inflate_sh)

    # 2. Compute per-pair distortions
    print(f"[diag] computing per-pair distortions (N={args.max_pairs}) on macOS-CPU")
    per_pair = _compute_per_pair_distortions(inflated_dir, max_pairs=args.max_pairs)

    # 3. Analyze
    analysis = _analyze_per_pair(
        archive,
        per_pair["pose_per_pair"],
        per_pair["seg_per_pair"],
        archive_bytes,
    )

    total_elapsed = time.perf_counter() - started

    # 4. Persist
    output = {
        "schema": "per_pair_sensitivity_v1",
        "lane_id": LANE_ID,
        "archive_id": args.archive_id,
        "archive_path": str(archive),
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_tag": EVIDENCE_TAG,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ranking_only": True,
        "platform": {
            "node": platform.node(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "elapsed_seconds": total_elapsed,
        "per_pair_elapsed_seconds": per_pair["elapsed_seconds"],
        "pair_count": per_pair["pair_count"],
        "stamped_at_utc": _utc_stamp(),
        "analysis": analysis,
        "raw_per_pair": {
            "pose_per_pair": per_pair["pose_per_pair"],
            "seg_per_pair": per_pair["seg_per_pair"],
        },
    }
    out_path = out / f"per_pair_sensitivity_{args.archive_id}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[diag] wrote {out_path}")
    print(
        f"[diag] N={analysis['N']}  pose_avg={analysis['pose_avg']:.6e}  "
        f"seg_avg={analysis['seg_avg']:.6e}"
    )
    print(
        f"[diag] top50_share={analysis['top50_concentration_share']:.4f}  "
        f"bottom50_share={analysis['bottom50_concentration_share']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
