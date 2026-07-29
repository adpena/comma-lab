#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pb1 P1 — receiver-realized full-n600 base verdict on the t3 TR1 endpoint archive.

Renders every pair through the COMMITTED TR1 receiver
(``tac.optimization.ddm_tr1_runtime``: ``parse_archive`` ->
``render_frame1_camera_uint8``; frame0 = zeros, the executable meaning of the
inert pose stub), verdicts with the frozen CPU-torch SegNet against the shared
GT cache (``lstars``) and, with ``--pose``, the frozen CPU-torch PoseNet
against the banked pose targets (``rows[i]["center"]``).

This is the DEPLOYED-bytes base: the numbers here are what the shipped archive
realizes, not the trainer's fp32 EMA confirm.  Per-pair rows are cached on SSD
so downstream steps (QDBS incremental-exact evaluation, rung-2 pricing) can
recompute full-population means after single-pair edits without re-verdicting
untouched pairs.

Outputs (resumable chunks; atomic tmp+rename):
  <out>/p1_chunks/chunk_SSSS_EEEE.npz   per-pair d_seg, per-class flips,
                                        token-cell flip maps, optional d_pose
  <out>/p1_receiver_realized_verdict.json   (--finalize)

Axis: [macOS-CPU advisory]. score_claim=false. Frozen CPU-torch scorers are the
verdict authority (never MPS/MLX) per CLAUDE.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/adpena/projects/pact")
SCHEMA_CHUNK = "ddm_pb1_p1_verdict_chunk.v1"
SCHEMA_RECEIPT = "ddm_pb1_p1_receiver_realized_verdict.v1"
N_CLASSES = 5
GRID_H = 24
GRID_W = 32


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--gt-cache", required=True, type=Path)
    ap.add_argument("--pose-targets", type=Path, default=None,
                    help="pose_metric_n600_batch32.json (rows[i].center 6-vec)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=120,
                    help="pairs per saved chunk (scorer-slot cap <=120)")
    ap.add_argument("--seg-batch", type=int, default=12)
    ap.add_argument("--pose", action="store_true", help="also verdict d_pose")
    ap.add_argument("--pose-batch", type=int, default=6)
    ap.add_argument("--finalize", action="store_true",
                    help="merge chunks and write the receipt JSON")
    args = ap.parse_args()
    if args.chunk > 120:
        ap.error("--chunk must be <= 120 (n600 scorer-slot discipline)")
    if args.pose and args.pose_targets is None:
        ap.error("--pose requires --pose-targets")
    return args


def _load_scorers(need_pose: bool):
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "upstream"))
    from tac.boundary_math.seg_core import load_real_segnet

    seg_cpu = load_real_segnet("cpu")
    posenet_cpu = None
    if need_pose:
        import torch
        from modules import DistortionNet, posenet_sd_path, segnet_sd_path

        dn = DistortionNet().eval()
        dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
        posenet_cpu = dn.posenet
        for p in posenet_cpu.parameters():
            p.requires_grad = False
    return seg_cpu, posenet_cpu


def _cell_flip_map(flip_mask: np.ndarray) -> np.ndarray:
    h, w = flip_mask.shape
    if h % GRID_H != 0 or w % GRID_W != 0:
        raise SystemExit(f"argmax grid {h}x{w} not divisible by token grid "
                         f"{GRID_H}x{GRID_W}")
    ch, cw = h // GRID_H, w // GRID_W
    return (flip_mask.reshape(GRID_H, ch, GRID_W, cw)
            .sum(axis=(1, 3)).astype(np.uint32))


def run_chunks(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "experiments"))
    from train_witness_realized_through_R_mlx import (
        cpu_verdict_d_pose_batch,
        cpu_verdict_d_seg_argmax_batch,
    )

    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    from tac.optimization.ddm_tr1_runtime import (
        parse_archive,
        render_frame1_camera_uint8,
    )

    archive_bytes = args.archive.read_bytes()
    parsed = parse_archive(archive_bytes)
    lstars = open_stored_npy_memmap(args.gt_cache, "lstars")
    seg_cpu, posenet_cpu = _load_scorers(args.pose)
    pose_targets = None
    if args.pose:
        rows = json.loads(args.pose_targets.read_text())["rows"]
        pose_targets = [np.asarray(r["center"], dtype=np.float64) for r in rows]

    chunk_dir = args.out_dir / "p1_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for c0 in range(args.start, args.end, args.chunk):
        c1 = min(c0 + args.chunk, args.end)
        out_path = chunk_dir / f"chunk_{c0:04d}_{c1:04d}.npz"
        if out_path.exists():
            print(f"[skip] {out_path.name} exists", flush=True)
            continue
        t0 = time.time()
        idxs = list(range(c0, c1))
        frames = [render_frame1_camera_uint8(parsed.packet, i) for i in idxs]
        t_render = time.time() - t0

        dsegs = np.zeros(len(idxs), dtype=np.float64)
        class_flips = np.zeros((len(idxs), N_CLASSES), dtype=np.int64)
        class_gt = np.zeros((len(idxs), N_CLASSES), dtype=np.int64)
        cell_flips = np.zeros((len(idxs), GRID_H, GRID_W), dtype=np.uint32)
        t1 = time.time()
        for b0 in range(0, len(idxs), args.seg_batch):
            b1 = min(b0 + args.seg_batch, len(idxs))
            gts = [np.asarray(lstars[i], dtype=np.int64) for i in idxs[b0:b1]]
            ds, realized = cpu_verdict_d_seg_argmax_batch(
                seg_cpu, frames[b0:b1], gts)
            for j, (d, gt) in enumerate(zip(ds, gts, strict=True)):
                k = b0 + j
                dsegs[k] = d
                flip = realized[j] != gt
                cell_flips[k] = _cell_flip_map(flip)
                for c in range(N_CLASSES):
                    gm = gt == c
                    class_gt[k, c] = int(gm.sum())
                    class_flips[k, c] = int((gm & flip).sum())
        t_seg = time.time() - t1

        dposes = np.full(len(idxs), np.nan, dtype=np.float64)
        t2 = time.time()
        if args.pose:
            zeros = np.zeros_like(frames[0])
            for b0 in range(0, len(idxs), args.pose_batch):
                b1 = min(b0 + args.pose_batch, len(idxs))
                dp = cpu_verdict_d_pose_batch(
                    posenet_cpu,
                    [zeros] * (b1 - b0),
                    frames[b0:b1],
                    [pose_targets[i] for i in idxs[b0:b1]],
                )
                dposes[b0:b1] = dp
        t_pose = time.time() - t2

        import io

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            schema=np.frombuffer(SCHEMA_CHUNK.encode(), dtype=np.uint8),
            idxs=np.asarray(idxs, dtype=np.int64),
            dsegs=dsegs,
            class_flips=class_flips,
            class_gt=class_gt,
            cell_flips=cell_flips,
            dposes=dposes,
            walls=np.asarray([t_render, t_seg, t_pose], dtype=np.float64),
        )
        _atomic_write_bytes(out_path, buf.getvalue())
        print(f"[chunk {c0}:{c1}] render {t_render:.1f}s seg {t_seg:.1f}s "
              f"pose {t_pose:.1f}s dseg_mean {dsegs.mean():.7f}", flush=True)


def finalize(args: argparse.Namespace) -> None:
    chunk_dir = args.out_dir / "p1_chunks"
    paths = sorted(chunk_dir.glob("chunk_*.npz"))
    idxs, dsegs, cflips, cgt, dposes, walls = [], [], [], [], [], []
    for p in paths:
        z = np.load(p)
        idxs.append(z["idxs"])
        dsegs.append(z["dsegs"])
        cflips.append(z["class_flips"])
        cgt.append(z["class_gt"])
        dposes.append(z["dposes"])
        walls.append(z["walls"])
    idx = np.concatenate(idxs)
    order = np.argsort(idx)
    idx = idx[order]
    if not np.array_equal(idx, np.arange(args.end - args.start) + args.start):
        raise SystemExit(f"coverage gap: have {idx.size} pairs, expected "
                         f"[{args.start},{args.end})")
    ds = np.concatenate(dsegs)[order]
    cf = np.concatenate(cflips)[order]
    cg = np.concatenate(cgt)[order]
    dp = np.concatenate(dposes)[order]
    total_px = float(cg.sum())
    per_class = [float(cf[:, c].sum()) / total_px for c in range(N_CLASSES)]
    d_seg_mean = float(ds.mean())
    if abs(sum(per_class) - d_seg_mean) > 1e-9:
        raise SystemExit(
            f"per-class decomposition {sum(per_class)} != mean {d_seg_mean}")
    pose_done = bool(np.isfinite(dp).all())
    receipt = {
        "schema": SCHEMA_RECEIPT,
        "evidence_axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "archive_path": str(args.archive),
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "gt_cache": str(args.gt_cache),
        "n_pairs": int(idx.size),
        "d_seg_mean": d_seg_mean,
        "d_seg_max": float(ds.max()),
        "d_seg_max_pair": int(idx[int(ds.argmax())]),
        "per_class_d_seg": per_class,
        "per_class_convention": (
            "per_class[c] = sum over pairs of |{px: gt==c and realized!=gt}| /"
            " total_px; sum over classes == d_seg_mean"),
        "d_pose_mean": (float(dp.mean()) if pose_done else None),
        "d_pose_max": (float(dp.max()) if pose_done else None),
        "d_pose_note": (
            "frame0 = zeros (inert pose stub executable meaning); targets ="
            " banked rows[i].center from pose_metric_n600_batch32.json"),
        "wall_seconds_total": float(np.asarray(walls).sum()),
        "chunks": [p.name for p in paths],
        "receiver_module_sha256": _sha256_file(
            REPO / "src/tac/optimization/ddm_tr1_runtime.py"),
        "generated_by": "tools/pb1_receiver_realized_verdict.py",
    }
    out = args.out_dir / "p1_receiver_realized_verdict.json"
    _atomic_write_bytes(out, (json.dumps(receipt, indent=1, sort_keys=True)
                              + "\n").encode())
    print(json.dumps({k: receipt[k] for k in (
        "d_seg_mean", "d_seg_max", "per_class_d_seg", "d_pose_mean",
        "n_pairs", "wall_seconds_total")}, indent=1))
    print(f"receipt: {out}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.finalize:
        finalize(args)
    else:
        run_chunks(args)


if __name__ == "__main__":
    main()
