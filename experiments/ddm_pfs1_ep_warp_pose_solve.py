#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pfs1 D2 — the e_p pose-field production solve ON the warp base.

THE REALIZATION (sc1 e_p ∘ p3v2 warp, composed on this vehicle): the D1 receiver
reconstructs frame_0 as ``warp(f1, H(p_ship; s_t))`` where ``p_ship`` is the
shipped per-pair 6-vector.  The evaluator scores PoseNet(f0,f1) against LIVE GT
— nothing forces ``p_ship == t_p``.  So the shipped 6-vector is a FREE per-pair
control: solve ``p*_i = argmin_p d_pose_i(warp(f1_i, H(p; s_t_i)))`` through the
REAL frozen CPU-torch PoseNet6 (STE-uint8 camera-res, the exact receiver path)
at IDENTICAL stream bytes to D1.  The pose-field ``δ_i = p*_i − t_p_i`` IS the
e_p field on this vehicle (sc1 machinery: rank-measured, AR/SMEVR-priced), and
the actuation is the se(3) ground-plane screw through the homography — a
DERIVED carrier direction (operator law: PoseNet-Jacobian/se(3)-projected, never
generic DCT/PCA; the generic-basis controls are already measured dead in p3v2
§2/§5).

Solver: per-pair damped Gauss-Newton on r(θ) = pose6(warp(θ)) − t_p ∈ R^6 with
θ = [δ ∈ R^6, ds_t] (7 DOF), forward-difference Jacobian (8 PoseNet forwards
per relinearization) + backtracking line search, budgeted relins.  Every
accepted point is scored at the SHIPPED quantization (float16 warp pose,
s_t on the continuous grid value) — realized points, never ceilings.

Modes:
  --mode solve  n600 chunked resumable full solve; JSONL rows carry
                {pair, p_star(f16-rounded), s_t, d_pose_warp, d_pose_solved}.
  --mode price  the (contribution, bytes) ladder: warp-only / 6dof-f16 /
                rank-k int8 projections (k in --ranks, SVD directions from the
                solved field itself), each REALIZED (fresh PoseNet forwards on
                the quantized reconstruction) + SMEVR/brotli-priced.

Axis: [macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE. score_claim=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED. One n600 scorer job at a time.
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

REPO = Path("/Users/adpena/projects/pact")
for _p in (REPO, REPO / "src", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SSD_OUT = Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729")
D1_WORK = SSD_OUT / "d1"
CAMERA_H, CAMERA_W = 874, 1164
ST_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.044, 0.06, 0.08, 0.12, 0.16, 0.24]
# forward-difference steps per warp-pose dim, scaled to t_p per-dim spread
# (std [1.256, .036, .030, .010, .007, .029]; dim0 = forward speed dominates).
FD_STEPS = np.array([0.08, 0.004, 0.004, 0.0015, 0.0015, 0.004], np.float64)
FD_STEP_ST = 0.004
FREE_FLOOR_CONTRIB = 0.0327  # p3v2 §0 unpriced free-frame_0 upper bound
FALSIFIER_BYTES = 4096
FALSIFIER_CONTRIB = 0.5


def _utc() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def contribution(d_pose_mean: float) -> float:
    return float(np.sqrt(10.0 * float(d_pose_mean)))


def load_solved_st(n: int) -> np.ndarray:
    jl = D1_WORK / "d1_warp_solve.partial.jsonl"
    per: dict[int, int] = {}
    for ln in jl.read_text().splitlines():
        if ln.strip():
            rr = json.loads(ln)
            per[int(rr["pair"])] = int(rr["s_t_idx"])
    missing = [i for i in range(n) if i not in per]
    if missing:
        raise SystemExit(f"D1 solve jsonl missing {len(missing)} pairs")
    return np.asarray([per[i] for i in range(n)], dtype=np.int64)


class WarpPoseOracle:
    """d_pose of the SHIPPED reconstruction for a candidate warp pose.

    Reuses the p3v2 frozen-PoseNet authority + the vendored receiver warp
    primitives (byte-identical to the engine, proven in the D1 build positive
    control).  ROTATION ACTIVATION (D2 finding): the D1 receiver ships
    ``s_r=0`` (R=I), leaving pose dims 3-5 INERT.  This oracle composes the
    SAME vendored primitives with ``s_r`` as a parameter (default 1.0) so all
    6 shipped dims act through the full plane-induced homography
    H = K(R - t n^T/d)K^{-1}; ``s_r=0`` reproduces the D1 receiver exactly.
    A D2-winning stream therefore implies a one-line receiver amendment
    (grammar v4, generic code), NOT built/eval'd in this arm.
    """

    def __init__(self, s_r: float = 1.0) -> None:
        import ddm_p3v2_optimal_form_pose_resolve as p3v2

        from tac.optimization import ddm_tr1_runtime as rt
        self.p3v2 = p3v2
        self.rt = rt
        self.s_r = float(s_r)
        self.posenet, _ = p3v2.load_posenet()
        seg = Path("/Volumes/VertigoDataTier/pact/ddm_pb1_20260729/p2c_aimed_archive.zip")
        parsed = rt.parse_archive(seg.read_bytes())
        self.packet = rt.parse_packet(rt.reemit_packet(parsed.packet))
        sys.path.insert(0, str(D1_WORK / "submission"))
        import pfs1_warp_receiver as recv
        self.recv = recv
        self.K = recv.intrinsics_native()
        self.Kinv = np.linalg.inv(self.K)
        self.grid = recv._target_grid(recv.CAMERA_H, recv.CAMERA_W)
        self.targets64 = p3v2.load_targets(600)

    def f1(self, pidx: int) -> np.ndarray:
        return self.rt.render_frame1_camera_uint8(self.packet, pidx)

    def d_pose(self, pidx: int, f1: np.ndarray, warp_pose: np.ndarray,
               s_t: float) -> float:
        H = self.recv.pose_to_homography(np.asarray(warp_pose, np.float64),
                                         self.K, self.Kinv, float(s_t),
                                         self.s_r, 0.0)
        f0 = self.recv._to_uint8(self.recv.warp_rgb(
            np.asarray(f1, np.float64), H, self.grid))
        return self.p3v2.d_pose_u8(self.posenet, f0, f1, self.targets64[pidx])

    def d_pose_shipped(self, pidx: int, f1: np.ndarray, warp_pose: np.ndarray,
                       s_t: float) -> float:
        """Score at the SHIPPED quantization: float16 pose (receiver decode dtype)."""
        q = np.asarray(warp_pose, np.float64).astype(np.float16).astype(np.float64)
        return self.d_pose(pidx, f1, q, s_t)


def solve_pair_gn(oracle: WarpPoseOracle, pidx: int, s_t0: float, *,
                  relins: int, lm0: float = 1.0) -> dict:
    """Damped GN over θ=[warp_pose(6), s_t] from (t_p, s_t0)."""
    tp = oracle.targets64[pidx].copy()
    f1 = oracle.f1(pidx)

    def pose6_of(warp_pose, s_t):
        f0 = oracle.recv.warp_base_f0(f1, warp_pose, float(s_t))
        return oracle.p3v2.pose6_u8(oracle.posenet, f0, f1)

    theta_p = tp.copy()
    theta_st = float(s_t0)
    cur6 = pose6_of(theta_p, theta_st)
    cur = float(np.mean((cur6 - tp) ** 2))
    d0 = cur
    lm = lm0
    n_fwd = 1
    for _ in range(relins):
        # numerical Jacobian: 7 forwards
        J = np.zeros((6, 7), np.float64)
        for k in range(6):
            dp = theta_p.copy()
            dp[k] += FD_STEPS[k]
            J[:, k] = (pose6_of(dp, theta_st) - cur6) / FD_STEPS[k]
        st_hi = min(max(theta_st + FD_STEP_ST, 0.0), 0.32)
        dst = st_hi - theta_st
        if dst > 1e-9:
            J[:, 6] = (pose6_of(theta_p, st_hi) - cur6) / dst
        n_fwd += 7
        r = cur6 - tp
        accepted = False
        for _damp in range(4):
            A = J.T @ J + lm * np.diag(np.maximum(np.diag(J.T @ J), 1e-8))
            try:
                step = np.linalg.solve(A, -(J.T @ r))
            except np.linalg.LinAlgError:
                break
            for scale in (1.0, 0.5):
                cp = theta_p + scale * step[:6]
                cst = float(np.clip(theta_st + scale * step[6], 0.0, 0.32))
                c6 = pose6_of(cp, cst)
                n_fwd += 1
                cval = float(np.mean((c6 - tp) ** 2))
                if cval < cur:
                    theta_p, theta_st, cur6, cur = cp, cst, c6, cval
                    lm = max(lm * 0.3, 1e-4)
                    accepted = True
                    break
            if accepted:
                break
            lm *= 8.0
        if not accepted or cur < 1e-6:
            break
    d_shipped = oracle.d_pose_shipped(pidx, f1, theta_p, theta_st)
    n_fwd += 1
    return {"pair": pidx, "d_pose_warp": d0, "d_pose_solved": cur,
            "d_pose_shipped_f16": float(d_shipped),
            "p_star": [float(x) for x in theta_p], "s_t": theta_st,
            "n_forwards": n_fwd}


def run_solve(args: argparse.Namespace) -> None:
    import torch
    torch.set_num_threads(4)
    oracle = WarpPoseOracle(s_r=float(args.s_r))
    n = int(args.n_pairs)
    st_idx = load_solved_st(n)
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    jl = work / "d2_ep_solve.partial.jsonl"
    cache: dict[int, dict] = {}
    if jl.exists() and args.resume:
        for ln in jl.read_text().splitlines():
            if ln.strip():
                rr = json.loads(ln)
                cache[int(rr["pair"])] = rr
        print(f"[d2-solve] resume: {len(cache)} cached", flush=True)
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    for i in range(n):
        if i in cache:
            continue
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            print(f"[d2-solve] --max-seconds at pair {i}; {len(cache)} done; "
                  "re-run --resume", flush=True)
            fj.close()
            raise SystemExit(2)
        row = solve_pair_gn(oracle, i, ST_GRID[int(st_idx[i])],
                            relins=int(args.relins))
        fj.write(json.dumps(row) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[i] = row
        if i % 10 == 0 or i == n - 1:
            ws = np.asarray([cache[k]["d_pose_warp"] for k in cache])
            ss = np.asarray([cache[k]["d_pose_shipped_f16"] for k in cache])
            print(f"  [d2 {i:3d}] warp={row['d_pose_warp']:.4f} -> "
                  f"solved(f16)={row['d_pose_shipped_f16']:.5f} "
                  f"fwd={row['n_forwards']} (means {ws.mean():.4f} -> "
                  f"{ss.mean():.5f}) ({time.time()-t0:.0f}s)", flush=True)
    fj.close()
    ws = np.asarray([cache[i]["d_pose_warp"] for i in range(n)])
    ss = np.asarray([cache[i]["d_pose_shipped_f16"] for i in range(n)])
    receipt = {
        "schema": "ddm_pfs1_d2_solve.v1", "utc": _utc(), "n_pairs": n,
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "relins": int(args.relins), "s_r": float(args.s_r),
        "start_note": "d_pose_warp rows = the (t_p, s_t_D1) start WITH s_r active "
                      "(differs from the D1 s_r=0 receiver point when s_r != 0)",
        "d_pose_warp_mean": float(ws.mean()),
        "d_pose_solved_f16_mean": float(ss.mean()),
        "d_pose_solved_f16_median": float(np.median(ss)),
        "d_pose_solved_f16_max": float(ss.max()),
        "contribution_warp": contribution(float(ws.mean())),
        "contribution_solved_f16": contribution(float(ss.mean())),
        "mean_forwards_per_pair": float(np.mean(
            [cache[i]["n_forwards"] for i in range(n)])),
    }
    (work / "d2_solve_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)


# --------------------------------------------------------------------------- #
# price: the (contribution, bytes) ladder — every point REALIZED at shipped quant
# --------------------------------------------------------------------------- #
def _code_bytes_f16(mat_f16: np.ndarray) -> int:
    import brotli
    return len(brotli.compress(np.ascontiguousarray(mat_f16, np.float16).tobytes(),
                               quality=11))


def _st_stream_bytes(st_idx: np.ndarray) -> int:
    from ddm_r7_token_coder import encode_token_codes
    codes = np.ascontiguousarray(st_idx, np.uint8).reshape(len(st_idx), 1, 1, 1)
    return len(encode_token_codes(codes, levels=len(ST_GRID), codec="auto"))


def run_price(args: argparse.Namespace) -> None:
    import torch
    torch.set_num_threads(4)
    work = args.work_dir
    jl = work / "d2_ep_solve.partial.jsonl"
    rows: dict[int, dict] = {}
    for ln in jl.read_text().splitlines():
        if ln.strip():
            rr = json.loads(ln)
            rows[int(rr["pair"])] = rr
    n = int(args.n_pairs)
    if len([i for i in range(n) if i in rows]) != n:
        raise SystemExit(f"d2 solve incomplete: {len(rows)}/{n}")
    oracle = WarpPoseOracle(s_r=float(args.s_r))
    st_idx_d1 = load_solved_st(n)
    tp64 = oracle.targets64[:n]
    p_star = np.asarray([rows[i]["p_star"] for i in range(n)], np.float64)
    st_star = np.asarray([rows[i]["s_t"] for i in range(n)], np.float64)
    d_warp = np.asarray([rows[i]["d_pose_warp"] for i in range(n)], np.float64)
    d_6dof = np.asarray([rows[i]["d_pose_shipped_f16"] for i in range(n)], np.float64)

    # s_t re-snap: shipped s_t returns to the 11-value grid index (nearest);
    # rank points are REALIZED at that snapped s_t (receiver format unchanged).
    grid = np.asarray(ST_GRID, np.float64)
    st_snap_idx = np.array([int(np.argmin(np.abs(grid - s))) for s in st_star])
    st_snap = grid[st_snap_idx]

    # the e_p field (reported in BOTH charts): delta over t_p = the sc1-style
    # residual field; the SHIPPED code is the SVD of p_star DIRECTLY (the
    # receiver needs ONLY the warp pose — t_p never ships on the rank rungs).
    delta = p_star - tp64
    pc = p_star - p_star.mean(0, keepdims=True)
    _u, S, Vt = np.linalg.svd(pc, full_matrices=False)
    energy = (S ** 2) / (S ** 2).sum()
    _du, dS, _dvt = np.linalg.svd(delta - delta.mean(0, keepdims=True),
                                  full_matrices=False)
    delta_energy = (dS ** 2) / (dS ** 2).sum()

    import brotli

    st_bytes = _st_stream_bytes(st_snap_idx.astype(np.uint8))

    # --- build every candidate warp-pose matrix + its stream bytes UP FRONT;
    #     then ONE per-pair pass (render f1 once, score all candidates).
    candidates: dict[str, dict] = {}
    p_star_f16_bytes = _code_bytes_f16(p_star.astype(np.float16))
    candidates["warp_plus_ep_6dof_f16"] = {
        "poses": p_star, "stream_bytes": p_star_f16_bytes + st_bytes,
        "note": "full 6-DOF solved warp pose, f16, SAME receiver format as D1"}
    mean_p = p_star.mean(0)
    for k in [int(x) for x in args.ranks]:
        Vk = Vt[:k]                       # (k,6) shipped once, f16
        C = pc @ Vk.T                     # (n,k)
        for qtag, qmax, qdt in (("int8", 127.0, np.int8),
                                ("int16", 32767.0, np.int16)):
            scale = np.abs(C).max(0) / qmax + 1e-12
            q = np.clip(np.round(C / scale), -qmax, qmax).astype(qdt)
            Cq = q.astype(np.float64) * scale
            stream = (len(brotli.compress(np.ascontiguousarray(q).tobytes(),
                                          quality=11))
                      + st_bytes
                      + len(np.asarray(Vk, np.float16).tobytes())
                      + len(np.asarray(scale, np.float16).tobytes())
                      + len(mean_p.astype(np.float16).tobytes()))
            candidates[f"warp_plus_ep_rank{k}_{qtag}"] = {
                "poses": mean_p[None, :] + Cq @ Vk, "stream_bytes": int(stream),
                "note": "SVD-of-p_star code: coeffs + f16 dirs/mean/scale + s_t; "
                        "t_p never ships"}

    # --- one resumable per-pair pass over all candidates
    jl2 = work / "d2_price_realize.partial.jsonl"
    cache: dict[int, dict] = {}
    if jl2.exists() and args.resume:
        for ln in jl2.read_text().splitlines():
            if ln.strip():
                rr = json.loads(ln)
                cache[int(rr["pair"])] = rr
        print(f"[d2-price] resume: {len(cache)} cached", flush=True)
    fj = open(jl2, "a")  # noqa: SIM115
    t0 = time.time()
    names = list(candidates)
    for i in range(n):
        if i in cache:
            continue
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            print(f"[d2-price] --max-seconds at pair {i}; {len(cache)} done; "
                  "re-run --resume", flush=True)
            fj.close()
            raise SystemExit(2)
        f1 = oracle.f1(i)
        row = {"pair": i}
        for name_ in names:
            row[name_] = float(oracle.d_pose_shipped(
                i, f1, candidates[name_]["poses"][i], float(st_snap[i])))
        fj.write(json.dumps(row) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[i] = row
        if i % 40 == 0 or i == n - 1:
            print(f"  [price {i:3d}] 6dof={row['warp_plus_ep_6dof_f16']:.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    fj.close()

    ladder: list[dict] = []
    tp_bytes = _code_bytes_f16(tp64.astype(np.float16))
    warp_only_bytes = tp_bytes + _st_stream_bytes(st_idx_d1.astype(np.uint8))
    d1r = D1_WORK / "d1_solve_receipt.json"
    if d1r.exists():
        d1_receipt = json.loads(d1r.read_text())
        ladder.append({
            "rung": "warp_only_tp_f16_sr0_D1",
            "d_pose_mean": d1_receipt["d_pose_mean"],
            "contribution": d1_receipt["pose_contribution"],
            "stream_bytes": warp_only_bytes,
            "note": "the D1 shipped point (s_r=0 receiver), CITED from d1_solve_receipt"})
    ladder.append({
        "rung": "warp_start_tp_sr_active", "d_pose_mean": float(d_warp.mean()),
        "contribution": contribution(float(d_warp.mean())),
        "stream_bytes": warp_only_bytes,
        "note": "the (t_p, s_t_D1) start point WITH s_r active (solve-row starts)"})
    for name_ in names:
        vals = np.asarray([cache[i][name_] for i in range(n)], np.float64)
        entry = {"rung": name_, "d_pose_mean": float(vals.mean()),
                 "contribution": contribution(float(vals.mean())),
                 "stream_bytes": candidates[name_]["stream_bytes"],
                 "note": candidates[name_]["note"]}
        if name_ == "warp_plus_ep_6dof_f16":
            entry["gn_unsnapped_mean_cite"] = float(d_6dof.mean())
        ladder.append(entry)

    best4k = [r for r in ladder if r["stream_bytes"] <= FALSIFIER_BYTES]
    falsifier_fired = not any(r["contribution"] < FALSIFIER_CONTRIB for r in best4k)
    receipt = {
        "schema": "ddm_pfs1_d2_price.v1", "utc": _utc(), "n_pairs": n,
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "s_r": float(args.s_r),
        "receiver_note": "rungs are priced for the s_r-active grammar v4 receiver "
                         "(one-line generic-code amendment); the D1 archive ships v3 s_r=0",
        "p_star_svd_energy_frac": [float(x) for x in energy],
        "e_p_delta_svd_energy_frac": [float(x) for x in delta_energy],
        "e_p_per_dim_std": [float(x) for x in delta.std(0)],
        "free_floor_contribution_cite": FREE_FLOOR_CONTRIB,
        "falsifier": {"budget_bytes": FALSIFIER_BYTES,
                      "bar_contribution": FALSIFIER_CONTRIB,
                      "fired": bool(falsifier_fired)},
        "ladder": ladder,
    }
    (work / "d2_price_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("solve", "price"), required=True)
    ap.add_argument("--work-dir", type=Path, default=SSD_OUT / "d2")
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--relins", type=int, default=4)
    ap.add_argument("--s-r", type=float, default=1.0,
                    help="rotation activation scale (0.0 = D1 receiver exactly)")
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    if args.mode == "solve":
        run_solve(args)
    else:
        run_price(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
