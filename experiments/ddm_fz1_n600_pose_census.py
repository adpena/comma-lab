# SPDX-License-Identifier: MIT
"""ddm_fz1 n600 pose-damage census on the cr2_ep854 delivered frames (charter step 1).

Per-pair d_pose of the DELIVERED pair vs the GT pose targets, all 600 pairs -- PoseNet forwards
only, no solves.  This is the damage-magnitude distribution bz1 named as the unknown, and it is
the stratum-weight input the mixed-k waterfill needs (m96: never extrapolate a pose verdict from
a prefix; the strata make the extrapolation honest).

Cross-check: the mean over 600 must reproduce the evaluator's n600 `Average PoseNet Distortion
37.87713242` (report.txt) -- an end-to-end control on GT decode + pose path (advisory-vs-exact
gap expected O(1e-4) relative, cr1 §7-class).

GT decode ONLY via frame_utils.yuv420_to_rgb.  Axis: [macOS-CPU frozen-scorer advisory]
NON-PROMOTABLE.  score_claim=False.
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
    ap.add_argument("--sub-dir", type=Path, required=True)
    ap.add_argument("--gt-mkv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--flush-every", type=int, default=25)
    args = ap.parse_args()

    t0 = time.time()
    raw = np.memmap(args.sub_dir / "inflated" / "0.raw", dtype=np.uint8, mode="r",
                    shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3))
    sc = Scorer(args.threads)

    rows: list[dict] = []
    if args.out.exists():
        rows = json.loads(args.out.read_text()).get("rows", [])
    done = {int(r["pair"]) for r in rows}

    def flush() -> None:
        d = np.array([r["d_pose_delivered"] for r in rows])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "schema": "ddm_fz1_n600_pose_census.v1",
                "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
                "score_claim": False,
                "n": len(rows),
                "mean": float(d.mean()) if len(rows) else None,
                "evaluator_n600_mean_control": 37.87713242,
                "quantiles": {q: float(np.quantile(d, float(q)))
                              for q in ("0.1", "0.25", "0.5", "0.75", "0.9", "0.99")}
                if len(rows) >= 20 else {},
                "strata_counts": {
                    "light_lt5": int((d < 5).sum()),
                    "mid_5_30": int(((d >= 5) & (d < 30)).sum()),
                    "heavy_ge30": int((d >= 30).sum()),
                } if len(rows) else {},
                "rows": rows}, f, indent=1)

    # incremental GT decode: stream the container once, score pairs as their frames arrive
    import av
    from ddm_sq1_eta_seg_realization import yuv420_to_rgb  # canonical decode only

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
                pose_gt = sc.pose_out(gt)
                dp = sc.d_pose(pose_gt, sc.pose_out(dec))
                rows.append({"pair": int(p), "d_pose_delivered": float(dp)})
                if len(rows) % args.flush_every == 0:
                    flush()
                    print(f"[fz1-c] {len(rows)}/600 t={time.time()-t0:.0f}s "
                          f"running_mean={np.mean([r['d_pose_delivered'] for r in rows]):.3f}",
                          flush=True)
            buf.clear()
        if idx >= N_PAIRS_TOTAL * seq_len - 1:
            break
    container.close()
    rows.sort(key=lambda r: r["pair"])
    flush()
    d = np.array([r["d_pose_delivered"] for r in rows])
    print(f"[fz1-c] DONE n={len(rows)} mean={d.mean():.6f} (control 37.87713242) "
          f"median={np.median(d):.4f} t={time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
