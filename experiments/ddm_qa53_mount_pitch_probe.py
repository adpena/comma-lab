#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_qa53 — mount-pitch global-refinement transfer probe (pm1 mine rank 5).

The pfs1 receiver ships the ground normal at pitch=0 (n=[0,-1,0]); openpilot
calibration custody says the EON mount pitch is ~-0.02 rad (window [-0.09,0.17],
calibrationd.py; #145 extrinsics pitch=-0.02).  A non-zero pitch tilts the
ground plane the homography assumes.  This is a ~0-byte global constant (one
derived-or-swept value shared by all pairs, all frames).

$0 TRANSFER probe (no re-solve): at each pair's qa43 GT-optimized p_two_star,
sweep a GLOBAL pitch on the GROUND homography (far stays H_inf) and re-evaluate
d_pose.  If some pitch != 0 improves d_pose across the tail beyond noise, the
ground plane is mis-calibrated at 0 and a free global-pitch amendment helps ALL
pairs; if the minimum sits at pitch=0, pitch=0 is optimal (INSTANCE-scope null).
The base = the qa45 realizable static two-plane (far = rows < v=437 -> H_inf,
ground = rows >= v=437 -> full H at pitch p, hood -> identity).

`tac` HIJACK control-guarded (QD15): PYTHONPATH="$PWD/src:$PWD/upstream:$PWD/experiments".
Axis: [macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE.  score_claim=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")

import numpy as np

np.seterr(all="ignore")

sys.path.insert(0, "experiments")
import ddm_pfs1_ep_warp_pose_solve as d2m

QA43_JL = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qa43_20260729/two_plane_probe_v2.partial.jsonl")
D2_JL = Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d2/d2_ep_solve.partial.jsonl")
OUT = Path("/Volumes/VertigoDataTier/pact/ddm_qa53_20260730")
JL = OUT / "mount_pitch_probe.partial.jsonl"

DERIVED_HORIZON = 437
PITCHES = [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04]  # rad; 0.0 == receiver ship value


def q16(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, np.float64).astype(np.float16).astype(np.float64)


def load_jsonl_by_pair(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[int(r["pair"])] = r
    return rows


def main() -> None:
    import torch
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)

    qa43 = load_jsonl_by_pair(QA43_JL)
    d2 = load_jsonl_by_pair(D2_JL)
    targets = sorted(p for p in qa43 if p in d2)
    print(f"[qa53] tail pairs: {len(targets)}; pitch sweep {PITCHES}", flush=True)

    oracle = d2m.WarpPoseOracle(s_r=1.0)
    recv, p3v2 = oracle.recv, oracle.p3v2
    K, Kinv, grid = oracle.K, oracle.Kinv, oracle.grid
    ch, cw = recv.CAMERA_H, recv.CAMERA_W
    rows_col = np.arange(ch)[:, None]
    m_far = (rows_col < DERIVED_HORIZON) & np.ones((1, cw), bool)

    done = set()
    if JL.exists():
        for ln in JL.read_text().splitlines():
            if ln.strip():
                done.add(int(json.loads(ln)["pair"]))
    todo = [p for p in targets if p not in done]
    print(f"[qa53] todo={len(todo)} (done {len(done)})", flush=True)
    if not todo:
        return

    t_all = time.time()
    for pidx in todo:
        t0 = time.time()
        row = d2[pidx]
        s_t = float(row["s_t"])
        tp = oracle.targets64[pidx].copy()
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        theta = q16(np.asarray(qa43[pidx]["p_two_star"], np.float64))
        # far field is pitch-independent (H_inf uses s_t=0 -> n drops out); warp once
        hf = recv.pose_to_homography(theta, K, Kinv, 0.0, 1.0, 0.0)
        warp_f = recv.warp_rgb(f1_f, hf, grid)

        res: dict[str, float] = {}
        for p_rad in PITCHES:
            hg = recv.pose_to_homography(theta, K, Kinv, s_t, 1.0, float(p_rad))
            warp_g = recv.warp_rgb(f1_f, hg, grid)
            f0 = np.where(m_far[..., None], warp_f, warp_g)
            p6 = p3v2.pose6_u8(oracle.posenet, recv._to_uint8(f0), f1_u8)
            res[f"p{p_rad:+.2f}"] = float(np.mean((p6 - tp) ** 2))
        best_key = min(res, key=res.get)
        rec = {
            "pair": int(pidx),
            "d_single_cached": float(qa43[pidx]["d_single_solved_cached"]),
            "pitch_dpose": res,
            "best_pitch": best_key, "best_dpose": float(res[best_key]),
            "d_pitch0": float(res["p+0.00"]),
            "wall_s": round(time.time() - t0, 1),
        }
        with JL.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[qa53] pair {pidx}: pitch0 {rec['d_pitch0']:.4f} best {best_key} "
              f"{rec['best_dpose']:.4f} {rec['wall_s']}s", flush=True)

    print(f"[qa53] done in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
