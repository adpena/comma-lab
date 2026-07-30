#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_qa45 stage-1 — REALIZABLE-mask transfer probe for the two-plane pose warp.

THE PROBLEM (ck1 §2/§5 + qa43 §1): every two-plane d_pose measured so far uses
the GT ``lstars`` class mask (frame_1 argmax, nearest-upsampled 384x512 ->
874x1164) to route far (class 2 -> H_inf) vs ground (full H) vs hood (class 4 ->
identity).  GT masks are ILLEGAL at inflate (no scorers at decode; GT not
shipped) -> the qa43/ck1 "selection 41.357" is an UPPER BOUND.  A realizable
receiver may only use: (a) a STATIC geometric mask (horizon-row split at the
K-derived ground-plane vanishing row v = cy = 437, plus a static hood region;
rule-118 FREE, 0 bytes), or (b) the decoded partition the archive ships (0
marginal bytes, needs the seg-cell decoder wired -> v4b build), or (c) a small
SHIPPED mask refinement (counted bytes, last resort).

STAGE 1 ($0 transfer, ~8 PoseNet fwd/pair): for each of the 112 tail pairs take
the ALREADY-SOLVED p_two_star (from qa43 two_plane_probe_v2) and re-evaluate
d_pose under candidate REALIZABLE masks WITHOUT re-solving:
  * gt_control        -- reproduce the qa43 GT-mask compose at p_two_star; MUST
                         match d_two_solved to the digit (substrate identity).
  * static horizon    -- ladder of static-global horizon rows {291,350,400,437,
                         480}; rows < h -> far (H_inf), else ground (full H).
  * hood variants     -- vdmaj (video-derived majority class-4 bitmap, COUNTED
                         bytes measured) / rect (free bottom-region rectangle) /
                         none (isolate the hood's contribution).

Selection metric (mirrors qa43): selection_total = sum_p min(d_single[p],
d_candidate[p]) where d_single = cached single-plane d_pose_solved.  DECISION
RULE: if a static candidate's selection_total <= ~1.15x the GT selection (41.357
-> 47.56), static wins outright (0 bytes).

Axis: [macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE.  score_claim=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.  GT-mask numbers are UPPER BOUND.
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
import ddm_pfs1_ep_warp_pose_solve as d2m  # WarpPoseOracle + oracle machinery

QA43_JL = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qa43_20260729/two_plane_probe_v2.partial.jsonl")
D2_JL = Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d2/d2_ep_solve.partial.jsonl")
GT_NPZ = Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
OUT = Path("/Volumes/VertigoDataTier/pact/ddm_qa45_20260730")
JL = OUT / "realizable_mask_probe.partial.jsonl"

HORIZONS = [291, 350, 400, 437, 480]  # 437 = derived cy vanishing row; 291 = ledger latent-8 prior
DERIVED_HORIZON = 437
HOOD_RECT_FRAC = 0.80  # free generic bottom-region prior (rows >= 0.80*H)


def q16(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, np.float64).astype(np.float16).astype(np.float64)


def load_jsonl_by_pair(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[int(r["pair"])] = r
    return rows


def geometric_horizon_row(K: np.ndarray) -> float:
    """Ground-plane vanishing row v for pitch=0 (n=[0,-1,0]): l = K^{-T} n,
    v = -l2/l1.  Returns cy for the EON intrinsics (437)."""
    n = np.array([0.0, -1.0, 0.0], np.float64)
    l = np.linalg.inv(K).T @ n
    return float(-l[2] / l[1])


def build_static_hood(lstars: np.ndarray, yi: np.ndarray, xi: np.ndarray,
                      ch: int, cw: int) -> tuple[np.ndarray, int, int]:
    """Video-derived STATIC hood = majority(class-4) over all frames.

    Returns (hood_874 bool HxW, brotli_bytes of the 384-res packed bitmap,
    hood_top_row_874).  The bitmap is video-derived (class-4 frequency) -> if
    shipped it is COUNTED; the byte cost is the brotli of the 384-res mask.
    """
    import brotli
    maj384 = (lstars == 4).mean(axis=0) >= 0.5  # (384,512) bool, static hood
    packed = np.packbits(maj384.astype(np.uint8).ravel())
    nbytes = len(brotli.compress(packed.tobytes(), quality=11))
    hood874 = maj384[np.ix_(yi, xi)]
    rows_with_hood = np.where(hood874.any(axis=1))[0]
    top = int(rows_with_hood.min()) if rows_with_hood.size else ch
    return hood874, nbytes, top


def main() -> None:
    import torch
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)

    qa43 = load_jsonl_by_pair(QA43_JL)
    d2 = load_jsonl_by_pair(D2_JL)
    targets = sorted(qa43)  # the 112 tail pairs
    print(f"[qa45] tail pairs: {len(targets)}", flush=True)

    oracle = d2m.WarpPoseOracle(s_r=1.0)
    recv, p3v2 = oracle.recv, oracle.p3v2
    ch, cw = recv.CAMERA_H, recv.CAMERA_W
    K = oracle.K
    gh = geometric_horizon_row(K)
    print(f"[qa45] geometric horizon row (cy vanishing) = {gh:.2f}", flush=True)

    lstars = np.load(GT_NPZ)["lstars"]
    yi = np.minimum((np.arange(ch) * 384) // ch, 383)
    xi = np.minimum((np.arange(cw) * 512) // cw, 511)
    hood_vdmaj, hood_bytes, hood_top = build_static_hood(lstars, yi, xi, ch, cw)
    hood_rect_top = int(round(HOOD_RECT_FRAC * ch))
    rows = np.arange(ch)[:, None]  # (H,1) for horizon masks
    hood_rect = np.zeros((ch, cw), bool)
    hood_rect[hood_rect_top:, :] = True
    print(f"[qa45] hood: vdmaj top_row={hood_top} bytes(brotli 384-res)={hood_bytes} "
          f"| rect top_row={hood_rect_top} (frac {HOOD_RECT_FRAC})", flush=True)

    # candidate list: (name, horizon_row or None-for-GT, hood_kind)
    cands: list[tuple[str, int | None, str]] = [("gt_control", None, "gt")]
    for h in HORIZONS:
        cands.append((f"h{h}_hood_vdmaj", h, "vdmaj"))
    # at the derived horizon, isolate hood contribution + free-rect variant
    cands.append((f"h{DERIVED_HORIZON}_hood_rect", DERIVED_HORIZON, "rect"))
    cands.append((f"h{DERIVED_HORIZON}_hood_none", DERIVED_HORIZON, "none"))

    done = set()
    if JL.exists():
        for ln in JL.read_text().splitlines():
            if ln.strip():
                done.add(int(json.loads(ln)["pair"]))
    todo = [p for p in targets if p not in done]
    print(f"[qa45] todo={len(todo)} (done {len(done)})", flush=True)
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

        # SHARED warps: ground (full H) + far (H_inf, s_t=0), both s_r=1.0
        hg = recv.pose_to_homography(theta, K, oracle.Kinv, s_t, 1.0, 0.0)
        hf = recv.pose_to_homography(theta, K, oracle.Kinv, 0.0, 1.0, 0.0)
        warp_g = recv.warp_rgb(f1_f, hg, oracle.grid)
        warp_f = recv.warp_rgb(f1_f, hf, oracle.grid)

        cls = lstars[pidx][np.ix_(yi, xi)]
        gt_far = (cls == 2)
        gt_hood = (cls == 4)

        def compose_eval(m_far: np.ndarray, m_hood: np.ndarray | None) -> float:
            f0 = np.where(m_far[..., None], warp_f, warp_g)
            if m_hood is not None:
                f0[m_hood] = f1_f[m_hood]
            p6 = p3v2.pose6_u8(oracle.posenet, recv._to_uint8(f0), f1_u8)
            return float(np.mean((p6 - tp) ** 2))

        rec: dict[str, float | int] = {"pair": int(pidx)}
        for name, h, hood_kind in cands:
            if h is None:  # gt control
                m_far, m_hood = gt_far, gt_hood
            else:
                m_far = (rows < h) & np.ones((1, cw), bool)
                m_hood = {"vdmaj": hood_vdmaj, "rect": hood_rect,
                          "none": None}[hood_kind]
            rec[name] = compose_eval(m_far, m_hood)
        rec["d_two_solved_gt_cached"] = float(qa43[pidx]["d_two_solved"])
        rec["d_single_solved_cached"] = float(qa43[pidx]["d_single_solved_cached"])
        rec["gt_ctrl_abs_delta"] = abs(rec["gt_control"] - rec["d_two_solved_gt_cached"])
        rec["wall_s"] = round(time.time() - t0, 1)
        with JL.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[qa45] pair {pidx}: gt_ctrl {rec['gt_control']:.4f} "
              f"(cached {rec['d_two_solved_gt_cached']:.4f} d={rec['gt_ctrl_abs_delta']:.2e}) "
              f"h437 {rec[f'h{DERIVED_HORIZON}_hood_vdmaj']:.4f} "
              f"single {rec['d_single_solved_cached']:.4f} {rec['wall_s']}s",
              flush=True)

    print(f"[qa45] sweep done in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
