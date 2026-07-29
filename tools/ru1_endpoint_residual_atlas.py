#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_ru1 Tier-A — endpoint residual atlas over the t3 TR1 endpoint archive.

One instrumented pass over the SAME receiver-realized path pb1's P1 verdict
used (``tac.optimization.ddm_tr1_runtime``: ``parse_archive`` ->
``render_frame1_camera_uint8``; frozen CPU-torch SegNet), extended to CAPTURE
per-flip records instead of only counts:

  per flipped pixel: (pair, y, x, gt_class, realized_class,
                      m_def   = z[realized] - z[gt]      logit deficit to close,
                      gap12   = z[top1] - z[top2]        realized runner-up gap,
                      gt_margin                          cached GT margin,
                      dist_bin in {0 on-GT-boundary, 1 within-3px, 2 interior},
                      gt_flicker  1 if GT argmax differs at this px vs adjacent pair)

Outputs resumable atomic chunks (<= --chunk pairs each, scorer-slot cap 120):
  <out>/atlas_chunks/atlas_SSSS_EEEE.npz

Axis: [macOS-CPU advisory]. score_claim=false, promotion_eligible=false.
Consumer: ddm_ru1 deliverables 1-3 (residual atlas / upstream typing / floor
slice); downstream pb1 P2 targeting. Pointer 0.1910828242 [contest-CPU] UNMOVED.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/adpena/projects/pact")
SCHEMA_CHUNK = "ddm_ru1_atlas_chunk.v1"
N_CLASSES = 5

# Processes that indicate the pb1 scorer slot (or the trainer) is live.
SLOT_TOKENS = ("pb1_receiver_realized_verdict", "train_levelset_witness",
               "train_witness_realized", "pb1_qdbs", "pb1_p2")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--gt-cache", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=120,
                    help="pairs per saved chunk (scorer-slot cap <=120)")
    ap.add_argument("--seg-batch", type=int, default=12)
    ap.add_argument("--skip-slot-check", action="store_true",
                    help="skip the pb1-liveness refusal (tests only)")
    args = ap.parse_args()
    if args.chunk > 120:
        ap.error("--chunk must be <= 120 (n600 scorer-slot discipline)")
    return args


def slot_is_live() -> bool:
    """True if a pb1/trainer scorer job appears in the process table."""
    out = subprocess.run(["ps", "-axo", "command"], capture_output=True,
                         text=True, check=False).stdout
    return any(tok in line for line in out.splitlines()
               for tok in SLOT_TOKENS if "ru1_endpoint_residual_atlas" not in line)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def _gt_boundary_distbin(gt: np.ndarray) -> np.ndarray:
    """0 = on a GT inter-class boundary (4-neighbor disagreement),
    1 = within ~3 px of one (two 3x3 dilations), 2 = interior."""
    b = np.zeros(gt.shape, dtype=bool)
    b[:-1, :] |= gt[:-1, :] != gt[1:, :]
    b[1:, :] |= gt[1:, :] != gt[:-1, :]
    b[:, :-1] |= gt[:, :-1] != gt[:, 1:]
    b[:, 1:] |= gt[:, 1:] != gt[:, :-1]
    near = b.copy()
    for _ in range(3):  # 3 dilations of radius 1 => within ~3 px
        d = near.copy()
        d[:-1, :] |= near[1:, :]
        d[1:, :] |= near[:-1, :]
        d[:, :-1] |= near[:, 1:]
        d[:, 1:] |= near[:, :-1]
        near = d
    dist = np.full(gt.shape, 2, dtype=np.uint8)
    dist[near] = 1
    dist[b] = 0
    return dist


def run_chunks(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "upstream"))
    import torch

    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    from tac.boundary_math.seg_core import load_real_segnet
    from tac.optimization.ddm_tr1_runtime import (
        parse_archive,
        render_frame1_camera_uint8,
    )

    archive_bytes = args.archive.read_bytes()
    parsed = parse_archive(archive_bytes)
    lstars = open_stored_npy_memmap(args.gt_cache, "lstars")
    gt_margins = open_stored_npy_memmap(args.gt_cache, "margins")
    seg_cpu = load_real_segnet("cpu")

    chunk_dir = args.out_dir / "atlas_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_total = int(np.asarray(lstars.shape[0]))

    for c0 in range(args.start, args.end, args.chunk):
        c1 = min(c0 + args.chunk, args.end)
        out_path = chunk_dir / f"atlas_{c0:04d}_{c1:04d}.npz"
        if out_path.exists():
            print(f"[skip] {out_path.name} exists", flush=True)
            continue
        if not args.skip_slot_check and slot_is_live():
            raise SystemExit(
                f"[refuse] pb1/trainer scorer job live before chunk {c0}:{c1} "
                f"- ru1 top-ups only when the slot is idle (charter)")
        t0 = time.time()
        idxs = list(range(c0, c1))
        frames = [render_frame1_camera_uint8(parsed.packet, i) for i in idxs]
        t_render = time.time() - t0

        rec_pair, rec_y, rec_x = [], [], []
        rec_gtc, rec_rlc = [], []
        rec_mdef, rec_gap12, rec_gtm = [], [], []
        rec_dist, rec_flick = [], []
        dsegs = np.zeros(len(idxs), dtype=np.float64)

        t1 = time.time()
        for b0 in range(0, len(idxs), args.seg_batch):
            b1 = min(b0 + args.seg_batch, len(idxs))
            arr = np.stack([np.asarray(f)[None] for f in frames[b0:b1]], axis=0)
            xp = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous().float()
            with torch.inference_mode():
                logits = seg_cpu(seg_cpu.preprocess_input(xp))  # (n,5,h,w)
            logits_np = logits.cpu().numpy().astype(np.float32)
            for j in range(b1 - b0):
                i = idxs[b0 + j]
                z = logits_np[j]                       # (5,h,w)
                realized = z.argmax(axis=0).astype(np.int64)
                gt = np.asarray(lstars[i], dtype=np.int64)
                flip = realized != gt
                dsegs[b0 + j] = float(flip.sum()) / gt.size
                ys, xs = np.nonzero(flip)
                if ys.size == 0:
                    continue
                gtc = gt[ys, xs]
                rlc = realized[ys, xs]
                z_at = z[:, ys, xs]                    # (5,K)
                m_def = z_at[rlc, np.arange(ys.size)] - z_at[gtc, np.arange(ys.size)]
                top2 = np.partition(z_at, -2, axis=0)[-2:, :]
                gap12 = top2[1] - top2[0]
                gtm = np.asarray(gt_margins[i])[ys, xs]
                dist = _gt_boundary_distbin(gt)[ys, xs]
                nb = i + 1 if i + 1 < n_total else i - 1
                flick = (np.asarray(lstars[nb], dtype=np.int64)[ys, xs] != gtc)
                rec_pair.append(np.full(ys.size, i, dtype=np.int32))
                rec_y.append(ys.astype(np.uint16))
                rec_x.append(xs.astype(np.uint16))
                rec_gtc.append(gtc.astype(np.uint8))
                rec_rlc.append(rlc.astype(np.uint8))
                rec_mdef.append(m_def.astype(np.float32))
                rec_gap12.append(gap12.astype(np.float32))
                rec_gtm.append(gtm.astype(np.float32))
                rec_dist.append(dist.astype(np.uint8))
                rec_flick.append(flick.astype(np.uint8))
        t_seg = time.time() - t1

        def cat(chunks, dtype):
            return (np.concatenate(chunks) if chunks
                    else np.zeros(0, dtype=dtype))

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            schema=np.frombuffer(SCHEMA_CHUNK.encode(), dtype=np.uint8),
            idxs=np.asarray(idxs, dtype=np.int64),
            dsegs=dsegs,
            pair=cat(rec_pair, np.int32),
            y=cat(rec_y, np.uint16),
            x=cat(rec_x, np.uint16),
            gt_class=cat(rec_gtc, np.uint8),
            realized_class=cat(rec_rlc, np.uint8),
            m_def=cat(rec_mdef, np.float32),
            gap12=cat(rec_gap12, np.float32),
            gt_margin=cat(rec_gtm, np.float32),
            dist_bin=cat(rec_dist, np.uint8),
            gt_flicker=cat(rec_flick, np.uint8),
            walls=np.asarray([t_render, t_seg], dtype=np.float64),
        )
        _atomic_write_bytes(out_path, buf.getvalue())
        n_flips = sum(int(a.size) for a in rec_pair)
        print(f"[chunk {c0}:{c1}] render {t_render:.1f}s seg {t_seg:.1f}s "
              f"flips {n_flips} dseg_mean {dsegs.mean():.7f}", flush=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_chunks(args)


if __name__ == "__main__":
    main()
