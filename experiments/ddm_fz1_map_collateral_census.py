# SPDX-License-Identifier: MIT
"""ddm_fz1 -- n600 advisory collateral census of the tz1 margin-coupled map on the pu2 vehicle.

The map coarsens token codes (rate -113,648 B gross); the renders change, so BOTH scorer legs
need the advisory measurement BEFORE the one n600 evaluator spend:

  seg:  argmax(mapped f_1) vs the CACHED GT argmax (gt_argmax_n600.npy) -> d_seg_mapped;
        also vs the CACHED live argmax (cx1_argmax_n600.npy) -> flip attribution.
        Break-even (tz1 #1): pays iff delta d_seg < 7.56e-4.
  pose: d_pose(mapped pair) vs GT pose targets -> the map's pose collateral + the per-pair
        base for the selective frame_0 repair (stale-partner law: solves run on MAPPED renders).

Advisory only ([macOS-CPU frozen-scorer advisory]); the exact row is the evaluator's.
GT decode ONLY via frame_utils.yuv420_to_rgb.  score_claim=False.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from ddm_js1_staging_discriminator import (
    CAM_H,
    CAM_W,
    N_PAIRS_TOTAL,
    Scorer,
    seq_len,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True, help="mapped inflated 0.raw")
    ap.add_argument("--gt-mkv", type=Path, required=True)
    ap.add_argument("--gt-argmax", type=Path, required=True)
    ap.add_argument("--live-argmax", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--flush-every", type=int, default=25)
    args = ap.parse_args()

    t0 = time.time()
    raw = np.memmap(args.raw, dtype=np.uint8, mode="r",
                    shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3))
    gtc = np.load(args.gt_argmax, mmap_mode="r")     # (600,384,512) uint8
    cx1 = np.load(args.live_argmax, mmap_mode="r")
    sc = Scorer(args.threads)

    rows: list[dict] = []
    if args.out.exists():
        rows = json.loads(args.out.read_text()).get("rows", [])
    done = {int(r["pair"]) for r in rows}

    def flush() -> None:
        n = len(rows)
        fl = np.array([r["flips_vs_gt"] for r in rows], dtype=np.int64)
        dp = np.array([r["d_pose_mapped"] for r in rows])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "schema": "ddm_fz1_map_collateral_census.v1",
                "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
                "score_claim": False,
                "n": n,
                "d_seg_mapped_running": float(fl.sum() / (n * 384 * 512)) if n else None,
                "d_seg_live_reference": 0.00431179,
                "seg_break_even_delta": 7.56e-4,
                "d_pose_mapped_running_mean": float(dp.mean()) if n else None,
                "d_pose_live_reference": 0.00154517,
                "rows": rows}, f, indent=1)

    import av
    from ddm_sq1_eta_seg_realization import yuv420_to_rgb

    container = av.open(str(args.gt_mkv))
    stream = container.streams.video[0]
    buf: dict[int, np.ndarray] = {}
    for idx, frame in enumerate(container.decode(stream)):
        buf[idx] = yuv420_to_rgb(frame).numpy()
        if idx % 2 == 1:
            p = idx // 2
            if p not in done:
                gt = np.stack([buf[idx - 1], buf[idx]])
                dec = np.stack([raw[idx - 1], raw[idx]]).astype(np.uint8)
                lam = sc.seg_argmax(dec)
                pose_gt = sc.pose_out(gt)
                dp = sc.d_pose(pose_gt, sc.pose_out(dec))
                rows.append({
                    "pair": int(p),
                    "flips_vs_gt": int((lam != np.asarray(gtc[p])).sum()),
                    "flips_vs_live": int((lam != np.asarray(cx1[p])).sum()),
                    "d_pose_mapped": float(dp),
                })
                if len(rows) % args.flush_every == 0:
                    flush()
                    print(f"[fz1-m] {len(rows)}/600 t={time.time()-t0:.0f}s", flush=True)
            buf.clear()
        if idx >= N_PAIRS_TOTAL * seq_len - 1:
            break
    container.close()
    rows.sort(key=lambda r: r["pair"])
    flush()
    fl = np.array([r["flips_vs_gt"] for r in rows], dtype=np.int64)
    dp = np.array([r["d_pose_mapped"] for r in rows])
    dseg = fl.sum() / (600 * 384 * 512)
    print(f"[fz1-m] DONE n={len(rows)} d_seg_mapped={dseg:.8f} "
          f"(live 0.00431179, delta {dseg-0.00431179:+.2e}, break-even +7.56e-4) "
          f"d_pose_mapped_mean={dp.mean():.8f} (live 0.00154517) t={time.time()-t0:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
