#!/usr/bin/env python
"""ddm_rs2 — n600 DRIVE comparison of the two BUILT A/B arms (base vs A vs B).

WHY THIS AND NOT THE FULL 384-CELL SWEEP
----------------------------------------
The full per-cell DRIVE sweep died silently at group 24/36 with no receipt (its
own design flaw: loop-end-only saving, which CLAUDE.md explicitly forbids).  Its
per-cell map was a nice-to-have.  The number that actually DECIDES the queued gate
is the DRIVE of the two arms against each other at n600 -- the drive-side half of
the flip-damage prediction in the memo's section 2.4.  That is 3 x 600 renders,
~8 minutes, and it is checkpointed per chunk so a kill costs one chunk.

DRIVE is exact and scorer-free: SegNet reads only D(f1), so the L1 of
D(cam_arm) - D(cam_base) over (384,512,3) is precisely the perturbation the scorer
sees, and a pixel can only flip if its scorer input changed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SSD = Path("/Volumes/VertigoDataTier/pact")
RT = SSD / "ddm_v4d_20260731"
PF = SSD / "ddm_pfs1_20260729/d1/eval_root/submissions/pfs1"
WORK = SSD / "ddm_rs2_20260803"
OUT = WORK / "rs2_arm_drive_n600.json"
CKPT = WORK / "rs2_arm_drive_n600.partial.jsonl"

for _p in ("src", str(PF), str(RT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import inflate_runner_v4d as IR  # noqa: E402

from tac.optimization.ddm_ll1_window_solve import window_geometry  # noqa: E402

P = 600
CHUNK = 50
THRESH = (0.0, 1.0, 2.0, 4.0, 8.0)
DIRS = {
    "base": WORK / "cx1_dir",
    "A_gr1_drop63": WORK / "ab_kA_gr1_drop63_dir",
    "B_rs2_rfkey": WORK / "ab_kB_rs2_rfkey_bytematched_dir",
}


def make_D():
    ys, xs, wy, wx = window_geometry()
    ys, xs = np.asarray(ys), np.asarray(xs)
    wy = np.asarray(wy, dtype=np.float32)
    wx = np.asarray(wx, dtype=np.float32)

    def D(cam):
        v = np.asarray(cam, dtype=np.float32)
        r = v[ys[:, 0]] * wy[:, 0, None, None] + v[ys[:, 1]] * wy[:, 1, None, None]
        return r[:, xs[:, 0]] * wx[None, :, 0, None] + r[:, xs[:, 1]] * wx[None, :, 1, None]

    return D


def main() -> int:
    t0 = time.time()
    D = make_D()
    decs = {k: IR.Decoder(v) for k, v in DIRS.items()}
    arms = [k for k in DIRS if k != "base"]
    acc = {a: {"L1": 0.0, "Linf": 0.0, "px": np.zeros(len(THRESH))} for a in arms}
    ab_L1 = 0.0
    done = 0
    with CKPT.open("w") as ck:
        for c0 in range(0, P, CHUNK):
            ids = range(c0, min(c0 + CHUNK, P))
            for p in ids:
                base = D(decs["base"].f1(p))
                planes = {}
                for a in arms:
                    d = np.abs(D(decs[a].f1(p)) - base)
                    planes[a] = d
                    m = d.max(axis=2)
                    acc[a]["L1"] += float(d.sum())
                    acc[a]["Linf"] = max(acc[a]["Linf"], float(m.max()))
                    for i, t in enumerate(THRESH):
                        acc[a]["px"][i] += float((m > t).sum())
                ab_L1 += float(np.abs(planes[arms[0]] - planes[arms[1]]).sum())
                done += 1
            ck.write(json.dumps({
                "pairs_done": done,
                "elapsed_s": round(time.time() - t0, 1),
                **{a: {"L1": acc[a]["L1"], "px_gt0": acc[a]["px"][0]} for a in arms},
            }) + "\n")
            ck.flush()
            print(f"pairs {done}/{P}  {time.time() - t0:.0f}s", flush=True)

    rep = {
        "axis": "[byte-closed, scorer-free]", "score_claim": False,
        "promotion_eligible": False, "n_pairs": done,
        "baseline": "cx1 v4d_composed_cx1_pj2ix2_archive.zip S 0.8264972 353,808 B",
        "arms": {a: {"drive_L1": acc[a]["L1"], "drive_Linf": acc[a]["Linf"],
                     **{f"px_gt_{t:g}LSB": acc[a]["px"][i] for i, t in enumerate(THRESH)}}
                 for a in arms},
        "elapsed_s": round(time.time() - t0, 1),
    }
    a0, a1 = arms
    rep["ratios_B_over_A"] = {
        "drive_L1": acc[a1]["L1"] / acc[a0]["L1"],
        **{f"px_gt_{t:g}LSB": (acc[a1]["px"][i] / acc[a0]["px"][i]) if acc[a0]["px"][i] else None
           for i, t in enumerate(THRESH)},
    }
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True))
    print(json.dumps(rep, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
