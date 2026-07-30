#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_qa44 — PHOTOMETRIC-PHYSICS rungs on the realizable two-plane pose warp.

Operator 07-29: "physics and photometrics interact... in which order... simple is
sometimes most potent."  The image-formation ORDER is
    motion -> per-depth projection -> photometric response -> rolling-shutter -> uint8
and the shipped receiver implements ONLY the projection stage (the two-plane warp).
QA44 measures the three next stages as $0 rungs, each aimed first at the tail's
hard core and validated against the deepest wins as controls.  PoseNet reads YUV6
of BOTH frames, so every rung is scorer-visible by construction.

BASE = the REALIZABLE static two-plane (qa45): far = rows < v_horizon (H_inf,
s_t=0), ground = rows >= v_horizon (full H), hood = identity.  v_horizon = the
K-derived ground-plane vanishing row cy=437 (rule-118 FREE, 0 bytes).  Pose per
pair = the qa43 GT-optimized ``p_two_star`` (advisory; a static re-solve is owed
at v4b, so absolute totals here are an UPPER BOUND on the base — the RUNG DELTAS
are the measurement).

Rungs (each measured as a d_pose delta vs the base ctrl on the SAME pair):
  A rolling-shutter row-shear: rotation scaled linearly across rows,
    rot(v) = ((1 - b/2) at top .. (1 + b/2) at bottom), pure SHEAR (mean rotation
    preserved).  b = a FIXED global physical constant (readout/inter-frame ratio),
    0 counted bytes; we SWEEP b to find what PoseNet reads.  b=0 == base ctrl.
  B auto-exposure gain/bias: f0 := a*warp(f1) + b applied AFTER warp BEFORE uint8
    (camerad AE changes between consecutive frames).  (a,b) GN-solved at fixed
    pose, ~4 B/pair f16.  No re-warp (post-multiply the cached composite).
  C Movable third plane: cars are NOT on the ground plane.  Route the Movable
    region (GT class-3 mask, UB) through a full H at a per-pair effective depth
    s_t_mov (GN-solved 1 param, ~2 B/pair).  s_t_mov = s_t (ground) reproduces
    the base for the Movable pixels.

Per-rung falsifier: no target pair improves > noise -> that physics is not what
PoseNet reads at this base (INSTANCE scope).  Controls: the 8 deepest wins must
not be DEGRADED by a global-constant rung (else the rung needs a selector).

`tac` import HIJACK control-guarded (QD15): run with
    PYTHONPATH="$PWD/src:$PWD/upstream:$PWD/experiments"
so `import tac` resolves to main src, NOT the eg1 codex worktree.  The GT-control
substrate-identity check (ctrl vs qa45 cached h437) is the positive control.

Axis: [macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE.  score_claim=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.  One n600 scorer job at a time (this
is PoseNet-only, tail-25 pairs — well within the PoseNet-bounded envelope).
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
OUT = Path("/Volumes/VertigoDataTier/pact/ddm_qa44_20260730")
JL = OUT / "photometric_rungs_probe.partial.jsonl"

# 17 hard-core two-plane losses (dt >= ds) + 8 deepest wins (controls), from qa43 v2.
HARDCORE17 = [16, 58, 62, 67, 70, 72, 74, 77, 78, 80, 90, 140, 150, 222, 512, 523, 590]
DEEPWINS8 = [19, 25, 46, 50, 51, 175, 280, 355]
TARGETS = sorted(set(HARDCORE17 + DEEPWINS8))

DERIVED_HORIZON = 437  # K-derived ground-plane vanishing row (cy), rule-118 free
RS_BETAS = [0.0, 0.5, 1.0, -0.5, -1.0]  # rolling-shutter shear sweep (b=0 == ctrl)
GAIN_FD, BIAS_FD = 0.02, 2.0  # rung-B FD steps (a near 1.0, b in 0..255 units)
STMOV_FD = 0.01               # rung-C FD step on the Movable-plane translation scale
GN_RELINS = 4
MOV_MIN_FRAC = 0.005          # skip rung C where Movable < 0.5% of pixels


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
    # default = the 25 aimed pairs (17 hard-core losses + 8 deep-win controls);
    # QA44_FULL_TAIL=1 widens to ALL 112 tail pairs (resumable — skips the 25).
    if os.environ.get("QA44_FULL_TAIL"):
        base = sorted(p for p in qa43 if p in d2)
    else:
        base = [p for p in TARGETS if p in qa43 and p in d2]
    targets = base
    print(f"[qa44] targets={len(targets)} (hardcore {len(HARDCORE17)} + "
          f"wins {len(DEEPWINS8)}; full_tail={bool(os.environ.get('QA44_FULL_TAIL'))})",
          flush=True)

    oracle = d2m.WarpPoseOracle(s_r=1.0)
    recv, p3v2 = oracle.recv, oracle.p3v2
    K, Kinv, grid = oracle.K, oracle.Kinv, oracle.grid
    ch, cw = recv.CAMERA_H, recv.CAMERA_W

    lstars = np.load(GT_NPZ)["lstars"]
    yi = np.minimum((np.arange(ch) * 384) // ch, 383)
    xi = np.minimum((np.arange(cw) * 512) // cw, 511)
    rows_col = np.arange(ch)[:, None]
    m_far_static = (rows_col < DERIVED_HORIZON) & np.ones((1, cw), bool)  # (H,W)
    alpha_row = (np.arange(ch) / (ch - 1.0))[:, None, None]  # (H,1,1) row blend

    done = set()
    if JL.exists():
        for ln in JL.read_text().splitlines():
            if ln.strip():
                done.add(int(json.loads(ln)["pair"]))
    todo = [p for p in targets if p not in done]
    print(f"[qa44] todo={len(todo)} (done {len(done)})", flush=True)
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

        cls = lstars[pidx][np.ix_(yi, xi)]
        m_hood = cls == 4
        m_mov = cls == 3
        mov_frac = float(m_mov.mean())

        # closures bind per-pair loop vars via default args (avoid late-binding;
        # matches the sister qa43 probe idiom) — all invoked within this iteration.
        def warps_at(rot_scale: float, *, _theta=theta, _s_t=s_t,
                     _f1_f=f1_f) -> tuple[np.ndarray, np.ndarray]:
            hg = recv.pose_to_homography(_theta, K, Kinv, _s_t, rot_scale, 0.0)
            hf = recv.pose_to_homography(_theta, K, Kinv, 0.0, rot_scale, 0.0)
            return (recv.warp_rgb(_f1_f, hg, grid), recv.warp_rgb(_f1_f, hf, grid))

        # cache the base (rot_scale=1) warps: rungs B/C reuse them
        wg1, wf1 = warps_at(1.0)

        def composite(wg: np.ndarray, wf: np.ndarray, *,
                      gain: float = 1.0, bias: float = 0.0,
                      wmov: np.ndarray | None = None,
                      _m_mov=m_mov, _m_hood=m_hood, _f1_f=f1_f) -> np.ndarray:
            f0 = np.where(m_far_static[..., None], wf, wg)
            if wmov is not None:
                f0 = np.where(_m_mov[..., None], wmov, f0)
            f0[_m_hood] = _f1_f[_m_hood]
            if gain != 1.0 or bias != 0.0:
                f0 = gain * f0 + bias
            return f0

        def score(f0_f: np.ndarray, *, _f1_u8=f1_u8, _tp=tp) -> float:
            p6 = p3v2.pose6_u8(oracle.posenet, recv._to_uint8(f0_f), _f1_u8)
            return float(np.mean((p6 - _tp) ** 2))

        n_fwd = 0

        # --- base ctrl (static two-plane at p_two_star) --------------------
        d_ctrl = score(composite(wg1, wf1))
        n_fwd += 1

        # --- RUNG A: rolling-shutter row-shear sweep -----------------------
        rungA: dict[str, float] = {}
        best_A, best_b = d_ctrl, 0.0
        for b in RS_BETAS:
            if b == 0.0:
                rungA["b0.0"] = d_ctrl
                continue
            wg_t, wf_t = warps_at(1.0 - b / 2.0)
            wg_b, wf_b = warps_at(1.0 + b / 2.0)
            f0_top = composite(wg_t, wf_t)
            f0_bot = composite(wg_b, wf_b)
            f0 = (1.0 - alpha_row) * f0_top + alpha_row * f0_bot
            dv = score(f0)
            n_fwd += 1
            rungA[f"b{b}"] = dv
            if dv < best_A:
                best_A, best_b = dv, b

        # --- RUNG B: auto-exposure gain/bias GN (a,b) at fixed pose --------
        a_p, b_p = 1.0, 0.0
        curB = d_ctrl
        cur6B = p3v2.pose6_u8(oracle.posenet,
                              recv._to_uint8(composite(wg1, wf1, gain=a_p, bias=b_p)),
                              f1_u8)
        n_fwd += 1
        lmB = 1.0
        for _ in range(GN_RELINS):
            # FD Jacobian over (a,b): 2 forwards
            p6_a = p3v2.pose6_u8(
                oracle.posenet,
                recv._to_uint8(composite(wg1, wf1, gain=a_p + GAIN_FD, bias=b_p)),
                f1_u8)
            p6_b = p3v2.pose6_u8(
                oracle.posenet,
                recv._to_uint8(composite(wg1, wf1, gain=a_p, bias=b_p + BIAS_FD)),
                f1_u8)
            n_fwd += 2
            Jb = np.stack([(p6_a - cur6B) / GAIN_FD, (p6_b - cur6B) / BIAS_FD], 1)
            r = cur6B - tp
            accepted = False
            for _damp in range(4):
                A = Jb.T @ Jb + lmB * np.diag(np.maximum(np.diag(Jb.T @ Jb), 1e-8))
                try:
                    step = np.linalg.solve(A, -(Jb.T @ r))
                except np.linalg.LinAlgError:
                    break
                for scale in (1.0, 0.5):
                    ca = a_p + scale * step[0]
                    cb = b_p + scale * step[1]
                    c6 = p3v2.pose6_u8(
                        oracle.posenet,
                        recv._to_uint8(composite(wg1, wf1, gain=ca, bias=cb)), f1_u8)
                    n_fwd += 1
                    cv = float(np.mean((c6 - tp) ** 2))
                    if cv < curB:
                        a_p, b_p, cur6B, curB = ca, cb, c6, cv
                        lmB = max(lmB * 0.3, 1e-4)
                        accepted = True
                        break
                if accepted:
                    break
                lmB *= 8.0
            if not accepted:
                break

        # --- RUNG C: Movable third plane GN (s_t_mov) ----------------------
        curC = d_ctrl
        stmov = s_t
        if mov_frac >= MOV_MIN_FRAC:
            wmov0 = recv.warp_rgb(
                f1_f, recv.pose_to_homography(theta, K, Kinv, stmov, 1.0, 0.0), grid)
            cur6C = p3v2.pose6_u8(
                oracle.posenet, recv._to_uint8(composite(wg1, wf1, wmov=wmov0)), f1_u8)
            curC = float(np.mean((cur6C - tp) ** 2))
            n_fwd += 1
            lmC = 1.0
            for _ in range(GN_RELINS):
                wmov_d = recv.warp_rgb(
                    f1_f,
                    recv.pose_to_homography(theta, K, Kinv, stmov + STMOV_FD, 1.0, 0.0),
                    grid)
                p6d = p3v2.pose6_u8(
                    oracle.posenet, recv._to_uint8(composite(wg1, wf1, wmov=wmov_d)),
                    f1_u8)
                n_fwd += 1
                Jc = ((p6d - cur6C) / STMOV_FD)[:, None]
                r = cur6C - tp
                accepted = False
                for _damp in range(4):
                    A = Jc.T @ Jc + lmC * np.maximum(Jc.T @ Jc, 1e-8)
                    try:
                        stp = float(np.linalg.solve(A, -(Jc.T @ r))[0])
                    except np.linalg.LinAlgError:
                        break
                    for scale in (1.0, 0.5):
                        cand = stmov + scale * stp
                        wmovc = recv.warp_rgb(
                            f1_f,
                            recv.pose_to_homography(theta, K, Kinv, cand, 1.0, 0.0), grid)
                        c6 = p3v2.pose6_u8(
                            oracle.posenet,
                            recv._to_uint8(composite(wg1, wf1, wmov=wmovc)), f1_u8)
                        n_fwd += 1
                        cv = float(np.mean((c6 - tp) ** 2))
                        if cv < curC:
                            stmov, cur6C, curC = cand, c6, cv
                            lmC = max(lmC * 0.3, 1e-4)
                            accepted = True
                            break
                    if accepted:
                        break
                    lmC *= 8.0
                if not accepted:
                    break

        d_single = float(qa43[pidx]["d_single_solved_cached"])
        rec = {
            "pair": int(pidx),
            "is_win_control": bool(pidx in DEEPWINS8),
            "d_single_cached": d_single,
            "d_two_gt_cached": float(qa43[pidx]["d_two_solved"]),
            "d_ctrl_static": float(d_ctrl),
            "rungA_rs": {k: float(v) for k, v in rungA.items()},
            "rungA_best": float(best_A), "rungA_best_beta": float(best_b),
            "rungB_gainbias": float(curB), "rungB_a": float(a_p), "rungB_b": float(b_p),
            "rungC_movplane": float(curC), "rungC_stmov": float(stmov),
            "rungC_mov_frac": mov_frac, "s_t": s_t,
            "n_fwd": int(n_fwd), "wall_s": round(time.time() - t0, 1),
        }
        with JL.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[qa44] pair {pidx}: ctrl {d_ctrl:.4f} | A {best_A:.4f}(b{best_b:+.1f}) "
              f"| B {curB:.4f}(a{a_p:.3f},b{b_p:+.1f}) | C {curC:.4f}"
              f"(mov{mov_frac:.3f}) | single {d_single:.4f} "
              f"fwd {n_fwd} {rec['wall_s']}s", flush=True)

    print(f"[qa44] sweep done in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
