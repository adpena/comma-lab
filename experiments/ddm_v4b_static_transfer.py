#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_v4b stage-1 — STATIC-mask transfer check on the Knee-A base (task #776).

THE MEASUREMENT (ck1 §2/§5 + qa45): every two-plane d_pose ck1 solved on the
Knee-A base used the GT ``lstars`` class mask (far cls2 -> H_inf, ground -> full
H, hood cls4 -> identity).  GT masks are ILLEGAL at inflate.  qa45 proved (on the
FULL base) that re-scoring the GT-solved ``p_two_star`` through a STATIC horizon
mask (far = rows < v, v = K-derived ground-plane vanishing row = round(cy) = 437)
BEATS the GT upper bound.  The SHIPPING archive is the KNEE base, so this arm
re-runs that transfer on the KNEE-base f1 (ck1.build_kneeA_oracle) with ck1's
KNEE-base ``p_two_star`` -> the HONEST realized shipping pose table the v4b gate
must reproduce.

Per pair (112 tail): ship selection = min(d_single_kneeA[cached], d_two_static).
selector = 1 iff two@static < single (monotone-safe: a degraded static pair falls
back to its single-plane fallback).  ship pose = p_two_star if selector else
p_single_kneeA.  Non-tail (488) ship single-plane p_single_kneeA (selector 0).

POSITIVE CONTROL (substrate identity): reproduce ck1's cached ``d_two_solved``
(GT masks) at p_two_star to ~0 -> the ck1/main-tac render+PoseNet == the
instrument that produced ck1_solve.  If it fails, STOP (confound).

QA50 rider ($0): SVD the post-two-plane tail correction (p_two_star - warp-start)
over the two-win pairs -> name the next systematic pose axis.

Axis: [macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE. score_claim=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.  GT numbers UPPER BOUND; static
numbers REALIZABLE.  The n600 evaluate gate (MAIN fires) is the authority.
"""
from __future__ import annotations

import argparse
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
import ddm_ck1_pose_resolve_kneeA as ck1  # build_kneeA_oracle + q16

CK1_SOLVE = Path("/Volumes/VertigoDataTier/pact/ddm_ck1_20260729/ck1_solve.partial.jsonl")
GT_NPZ = Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
OUT = Path("/Volumes/VertigoDataTier/pact/ddm_v4b_20260730")
JL = OUT / "v4b_static_transfer.partial.jsonl"


def _utc() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def q16(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, np.float64).astype(np.float16).astype(np.float64)


def contribution(dpm: float) -> float:
    return float(np.sqrt(10.0 * float(dpm)))


def geometric_horizon_row(K: np.ndarray) -> float:
    """Ground-plane vanishing row v for pitch=0 (n=[0,-1,0]): l = K^{-T} n,
    v = -l2/l1 -> cy for the EON intrinsics (437).  Pure code, 0 bytes."""
    n = np.array([0.0, -1.0, 0.0], np.float64)
    ln = np.linalg.inv(K).T @ n
    return float(-ln[2] / ln[1])


def load_ck1(path: Path) -> dict[int, dict]:
    return {int(json.loads(x)["pair"]): json.loads(x)
            for x in path.read_text().splitlines() if x.strip()}


def load_done(path: Path) -> set[int]:
    done: set[int] = set()
    if path.exists():
        for x in path.read_text().splitlines():
            if x.strip():
                done.add(int(json.loads(x)["pair"]))
    return done


def run_transfer() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ck = load_ck1(CK1_SOLVE)
    tail = sorted(p for p in ck if ck[p].get("in_tail"))
    print(f"[v4b] tail pairs: {len(tail)}", flush=True)

    oracle = ck1.build_kneeA_oracle(s_r=1.0)
    recv, p3v2 = oracle.recv, oracle.p3v2
    ch, cw = recv.CAMERA_H, recv.CAMERA_W
    K = oracle.K
    v = geometric_horizon_row(K)
    v_row = round(v)
    print(f"[v4b] derived horizon row v = {v:.4f} -> row {v_row}", flush=True)

    lstars = np.load(GT_NPZ)["lstars"]
    yi = np.minimum((np.arange(ch) * 384) // ch, 383)
    xi = np.minimum((np.arange(cw) * 512) // cw, 511)
    rows = np.arange(ch)[:, None]            # (H,1) for the static horizon
    static_far = (rows < v_row) & np.ones((1, cw), bool)   # (H,W) bool

    done = load_done(JL)
    todo = [p for p in tail if p not in done]
    print(f"[v4b] todo={len(todo)} (done {len(done)})", flush=True)
    if not todo:
        return

    t_all = time.time()
    for pidx in todo:
        t0 = time.time()
        r = ck[pidx]
        s_t = float(r["s_t"])
        tp = oracle.targets64[pidx].copy()
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        theta = q16(np.asarray(r["p_two_star"], np.float64))

        # SHARED warps at the shipped f16 two-plane pose (s_r=1.0)
        hg = recv.pose_to_homography(theta, K, oracle.Kinv, s_t, 1.0, 0.0)
        hf = recv.pose_to_homography(theta, K, oracle.Kinv, 0.0, 1.0, 0.0)
        warp_g = recv.warp_rgb(f1_f, hg, oracle.grid)
        warp_f = recv.warp_rgb(f1_f, hf, oracle.grid)

        def _mse(f0f: np.ndarray, *, _f1_u8=f1_u8, _tp=tp) -> float:
            p6 = p3v2.pose6_u8(oracle.posenet, recv._to_uint8(f0f), _f1_u8)
            return float(np.mean((p6 - _tp) ** 2))

        # REALIZABLE static compose (the shipping f0 for a two-plane pair)
        f0_static = np.where(static_far[..., None], warp_f, warp_g)
        d_two_static = _mse(f0_static)

        # POSITIVE CONTROL: GT-mask compose must reproduce ck1 d_two_solved
        cls = lstars[pidx][np.ix_(yi, xi)]
        gt_far = (cls == 2)[..., None]
        gt_hood = cls == 4
        f0_gt = np.where(gt_far, warp_f, warp_g)
        f0_gt[gt_hood] = f1_f[gt_hood]
        d_two_gt = _mse(f0_gt)

        d_single = float(r["d_single_kneeA"])
        d_two_gt_cached = float(r["d_two_solved"])
        selector = 1 if d_two_static < d_single else 0
        rec = {
            "pair": int(pidx), "s_t": s_t,
            "d_single_kneeA": d_single,
            "d_two_static": float(d_two_static),
            "d_two_gt": float(d_two_gt),
            "d_two_gt_cached": d_two_gt_cached,
            "gt_ctrl_abs_delta": abs(d_two_gt - d_two_gt_cached),
            "selector": int(selector),
            "d_ship": float(min(d_single, d_two_static)),
            "ship_kind": "two" if selector else "single",
            "p_two_star": [float(x) for x in r["p_two_star"]],
            "p_single_kneeA": [float(x) for x in r["p_single_kneeA"]],
            "wall_s": round(time.time() - t0, 1),
        }
        with JL.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(f"[v4b {pidx:3d}] single {d_single:.4f} two@static {d_two_static:.4f} "
              f"(gt {d_two_gt:.4f} cached {d_two_gt_cached:.4f} "
              f"d={rec['gt_ctrl_abs_delta']:.2e}) sel={selector} "
              f"ship {rec['d_ship']:.4f} {rec['wall_s']}s", flush=True)
    print(f"[v4b] transfer done in {time.time()-t_all:.0f}s", flush=True)


def summarize() -> None:
    """Build the FULL 600 shipping table + aggregate + QA50 SVD rider."""
    ck = load_ck1(CK1_SOLVE)
    xfer = load_ck1(JL)  # keyed by pair
    tail = sorted(p for p in ck if ck[p].get("in_tail"))
    n_tail_done = sum(1 for p in tail if p in xfer)
    if n_tail_done != len(tail):
        print(f"[v4b] WARNING: {n_tail_done}/{len(tail)} tail transferred; "
              "summary is partial", flush=True)

    # 600 shipping vector: nontail = single-plane p_single_kneeA (selector 0);
    # tail = the min-selection measured through the static compose.
    ship = {}
    for p in range(600):
        if ck[p].get("in_tail") and p in xfer:
            x = xfer[p]
            ship[p] = {"selector": int(x["selector"]),
                       "p": x["p_two_star"] if x["selector"] else x["p_single_kneeA"],
                       "d": float(x["d_ship"]), "kind": x["ship_kind"]}
        else:
            ship[p] = {"selector": 0, "p": ck[p]["p_single_kneeA"],
                       "d": float(ck[p]["d_single_kneeA"]), "kind": "single"}

    d600 = np.array([ship[p]["d"] for p in range(600)])
    sel = np.array([ship[p]["selector"] for p in range(600)])
    seg = 0.00553676
    import os as _os
    kb = _os.path.getsize(
        "/Volumes/VertigoDataTier/pact/ddm_wr1_20260729/wr1_kneeA_safe_274k_archive.zip")
    rate = 25 * kb / 37_545_489
    pose_contrib = contribution(float(d600.mean()))
    predicted_S = 100 * seg + pose_contrib + rate

    # GT baselines for the delta
    dbest = np.array([ck[p]["d_best_kneeA"] for p in range(600)])
    dsingle = np.array([ck[p]["d_single_kneeA"] for p in range(600)])

    # QA50: SVD of the post-two-plane tail correction (shipped two-star vs
    # warp-start p0 = t_p with rotation dims zeroed) over the selector=1 pairs.
    two_pairs = [p for p in tail if p in xfer and xfer[p]["selector"] == 1]
    qa50 = {}
    if two_pairs:
        # correction = shipped two-plane pose MINUS the single-plane pose it
        # replaced (the systematic direction the two-plane solve moves the ship).
        M = np.array([np.asarray(xfer[p]["p_two_star"], np.float64)
                      - np.asarray(xfer[p]["p_single_kneeA"], np.float64)
                      for p in two_pairs])
        Mc = M - M.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
        energy = (S ** 2) / max(float((S ** 2).sum()), 1e-30)
        qa50 = {
            "n_two_win": len(two_pairs),
            "singular_values": [float(s) for s in S],
            "energy_fraction": [float(e) for e in energy],
            "top_axis_right_singular_vector": [float(x) for x in Vt[0]],
            "mean_correction_6dof": [float(x) for x in M.mean(0)],
            "note": "correction = shipped two-plane pose - single-plane pose it "
                    "replaced, over selector=1 tail pairs; SVD names the dominant "
                    "systematic pose axis. dims [dz,dy,dx (t_p order), rx,ry,rz].",
        }

    receipt = {
        "schema": "ddm_v4b_static_transfer.v1", "utc": _utc(),
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE; "
                "static REALIZABLE, GT UPPER BOUND; gate is authority",
        "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "n_tail_transferred": n_tail_done, "n_selector_two": int(sel.sum()),
        "gt_ctrl_max_abs_delta": max(
            (float(xfer[p]["gt_ctrl_abs_delta"]) for p in xfer), default=None),
        "gt_ctrl_mean_abs_delta": float(np.mean(
            [xfer[p]["gt_ctrl_abs_delta"] for p in xfer])) if xfer else None,
        "kneeA_bytes": kb, "rate_term": rate, "seg_d": seg, "seg_contribution": 100 * seg,
        "ship_mean_d_pose_600": float(d600.mean()),
        "ship_pose_contribution": pose_contrib,
        "predicted_composed_S": predicted_S,
        "gt_best_mean_d_pose_600": float(dbest.mean()),
        "gt_best_contribution": contribution(float(dbest.mean())),
        "gt_best_composed_S": 100 * seg + contribution(float(dbest.mean())) + rate,
        "single_only_mean_d_pose_600": float(dsingle.mean()),
        "single_only_composed_S": 100 * seg + contribution(float(dsingle.mean())) + rate,
        "tail_static_mean": float(np.mean([xfer[p]["d_two_static"] for p in xfer]))
        if xfer else None,
        "tail_single_mean": float(np.mean([xfer[p]["d_single_kneeA"] for p in xfer]))
        if xfer else None,
        "tail_ship_mean": float(np.mean([xfer[p]["d_ship"] for p in xfer]))
        if xfer else None,
        "qa50_svd_rider": qa50,
    }
    (OUT / "v4b_static_transfer_receipt.json").write_text(
        json.dumps(receipt, indent=1) + "\n")

    # shipping table the build tool consumes
    ship_rows = [{"pair": p, **ship[p]} for p in range(600)]
    (OUT / "v4b_ship_table.json").write_text(
        json.dumps({"schema": "ddm_v4b_ship_table.v1", "n": 600,
                    "n_selector_two": int(sel.sum()), "rows": ship_rows}, indent=0) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("transfer", "summarize"), default="transfer")
    args = ap.parse_args()
    if args.mode == "transfer":
        run_transfer()
    else:
        summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
