# SPDX-License-Identifier: MIT
"""Build the SHARED GT cache for the MLX through-R witness FLEET (operator 2026-06-25).

The MLX trainer ``experiments/train_witness_realized_through_R_mlx.py`` calls
``precompute_gt(n_pairs)`` per-process: it decodes the GT video + runs the FROZEN
CPU-torch SegNet (argmax + margin) and PoseNet (GT pose) over ``n_pairs`` pairs.
At n600 that is ~480 s of CPU work PER ARM. A parallel fleet that each recomputes
GT would THRASH the CPU and serialize on the decode/scorer forward -- defeating
the GPU+unified-memory saturation goal.

This builder runs ``precompute_gt`` ONCE and serializes the complete GTData
(gt_f0, gt_f1 uint8 frames + lstars int64 + margins f32 + gt_poses f64) to a
single npz the fleet arms mmap-load via ``--gt-cache``. NO-FAKE: the cached
arrays are the EXACT outputs of the same frozen CPU-torch authority the trainer
uses inline -- no surrogate, byte-identical to the inline path.

Disk: writes to the SSD/repo results tier, NEVER /tmp durable (CLAUDE.md hygiene).
The npz is rebuildable (this command + the frozen scorers + the GT video), so it
is safely externalizable; the cache path itself is the provenance.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def build(n_pairs: int, out_path: Path) -> dict:
    from experiments.train_witness_realized_through_R_mlx import precompute_gt

    t0 = time.time()
    gt, _seg_cpu, _posenet_cpu = precompute_gt(n_pairs)
    secs = time.time() - t0

    P = gt.n_pairs
    # Stack into single dense arrays (mmap-friendly). Frames are (P,H,W,3) uint8.
    gt_f0 = np.stack([np.asarray(a, dtype=np.uint8) for a in gt.gt_f0], axis=0)
    gt_f1 = np.stack([np.asarray(a, dtype=np.uint8) for a in gt.gt_f1], axis=0)
    lstars = np.stack([np.asarray(a, dtype=np.int64) for a in gt.lstars], axis=0)
    margins = np.stack([np.asarray(a, dtype=np.float32) for a in gt.margins], axis=0)
    gt_poses = np.stack([np.asarray(a, dtype=np.float64) for a in gt.gt_poses], axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        n_pairs=np.int64(P),
        gt_f0=gt_f0,
        gt_f1=gt_f1,
        lstars=lstars,
        margins=margins,
        gt_poses=gt_poses,
    )
    nbytes = out_path.stat().st_size
    return {
        "n_pairs": int(P),
        "build_secs": round(secs, 1),
        "out_path": str(out_path),
        "bytes": int(nbytes),
        "shapes": {
            "gt_f0": list(gt_f0.shape),
            "gt_f1": list(gt_f1.shape),
            "lstars": list(lstars.shape),
            "margins": list(margins.shape),
            "gt_poses": list(gt_poses.shape),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the shared GT cache npz for the MLX through-R fleet")
    ap.add_argument("--num-pairs", type=int, required=True)
    ap.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="npz path; defaults to experiments/results/mlx_fleet_gt_cache/gt_n<N>.npz",
    )
    args = ap.parse_args(argv)
    out_path = args.out_path or (REPO / "experiments/results/mlx_fleet_gt_cache" / f"gt_n{args.num_pairs}.npz")
    _refuse_tmp(out_path)
    info = build(args.num_pairs, out_path)
    print(json.dumps(info, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
