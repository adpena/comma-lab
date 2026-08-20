#!/usr/bin/env python3
"""ddm_qa43 stage-1a — TWO-PLANE (near/far parallax) warp probe on the pose tail.

Operator pointer 2026-07-29: "Openpilot has polytope and other related to yaw and
rotation on multiple axes and near vs far and parallax."

Mechanism under test (uh1 §2/§3): the shipped warp reconstructs frame_0 from
frame_1 with ONE plane-induced homography H = K(R - t n^T/d)K^-1, which assigns
ground-depth parallax to FAR content (Undrivable = sky/far scenery, ~49% of
area). On the 112-pair turn tail the dense single-plane 6-DOF solve removes only
12.8% of residual and its correction is rank-1 along dim-0 (forward speed,
+32.6 near-constant) — the classic rotation/translation aliasing of a single
planar homography. The derived cure: per-class MULTI-PLANE warp —
  far   (class 2, Undrivable)     -> H_inf = K R K^-1   (s_t = 0, d -> inf)
  ground(classes 0,1,3 Road/Lane/Movable) -> full H at s_t (d = camera height)
  hood  (class 4, MyCar, static)  -> identity (f0 := f1)
Class masks come from the ALREADY-SHIPPED partition (probe stand-in: GT lstars,
frame_1 argmax, nearest-upsampled 384x512 -> 874x1164; the receiver would derive
the mask from the decoded partition — generic code, rule-118 FREE, ~zero
marginal counted bytes; probe-vs-decoded mask delta noted in the receipt).

Protocol (k worst tail pairs by cached d_pose_solved, resumable JSONL):
  ctrl        single-plane d_pose at cached p_star (f16) — must reproduce the
              cached d_pose_solved (substrate-identity check; ch1 discipline)
  two@p_star  two-plane generator at the CACHED single-plane solution
  two solved  damped-GN re-solve (mirrors ddm_pfs1_ep_warp_pose_solve.solve_pair_gn:
              start = t_p with rotation dims zeroed, f16-quantized acceptance,
              FD Jacobian, LM damping) through the two-plane compose

Axis: [macOS-CPU advisory; frozen PoseNet oracle; probe not a score claim].
Pointer: 0.1910828242 [contest-CPU] UNMOVED.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments")
import ddm_pfs1_ep_warp_pose_solve as d2m  # oracle + FD_STEPS

D2_JL = Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d2/d2_ep_solve.partial.jsonl")
GT_NPZ = Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
OUT = Path("/Volumes/VertigoDataTier/pact/ddm_qa43_20260729")
# v2 (multi-start): GN from BOTH starts {p0 rotation-zeroed, q16(p_star)} — the
# v1 run measured the zero-rotation start POISONED for the two-plane compose
# (H_far=identity freezes the far field; pairs 82/71/72 stuck). v1 file kept.
JL = OUT / "two_plane_probe_v2.partial.jsonl"

K_PAIRS = 112
RELINS = 4


def q16(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, np.float64).astype(np.float16).astype(np.float64)


def load_d2() -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for line in D2_JL.read_text().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[int(r["pair"])] = r
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    solve = load_d2()
    order = sorted(solve, key=lambda p: -solve[p]["d_pose_solved"])
    targets = order[:K_PAIRS]

    done = set()
    if JL.exists():
        for line in JL.read_text().splitlines():
            if line.strip():
                done.add(int(json.loads(line)["pair"]))
    todo = [p for p in targets if p not in done]
    print(f"[qa43-2p] k={K_PAIRS} todo={todo}", flush=True)
    if not todo:
        return

    oracle = d2m.WarpPoseOracle(s_r=1.0)
    recv, p3v2 = oracle.recv, oracle.p3v2
    lstars = np.load(GT_NPZ)["lstars"]
    ch, cw = recv.CAMERA_H, recv.CAMERA_W
    yi = np.minimum((np.arange(ch) * 384) // ch, 383)
    xi = np.minimum((np.arange(cw) * 512) // cw, 511)

    for pidx in todo:
        t0 = time.time()
        row = solve[pidx]
        s_t = float(row["s_t"])
        tp = oracle.targets64[pidx].copy()
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        cls = lstars[pidx][np.ix_(yi, xi)]
        m_far = (cls == 2)[..., None]
        m_hood = cls == 4

        def compose_two(theta: np.ndarray, *, _s_t=s_t, _m_far=m_far,
                        _m_hood=m_hood, _f1_f=f1_f) -> np.ndarray:
            hg = recv.pose_to_homography(np.asarray(theta, np.float64),
                                         oracle.K, oracle.Kinv, _s_t, 1.0, 0.0)
            hf = recv.pose_to_homography(np.asarray(theta, np.float64),
                                         oracle.K, oracle.Kinv, 0.0, 1.0, 0.0)
            f0 = np.where(_m_far,
                          recv.warp_rgb(_f1_f, hf, oracle.grid),
                          recv.warp_rgb(_f1_f, hg, oracle.grid))
            f0[_m_hood] = _f1_f[_m_hood]
            return recv._to_uint8(f0)

        def pose6_two(theta: np.ndarray, *, _compose=compose_two,
                      _f1_u8=f1_u8) -> np.ndarray:
            return p3v2.pose6_u8(oracle.posenet, _compose(theta), _f1_u8)

        def mse(p6: np.ndarray, *, _tp=tp) -> float:
            return float(np.mean((p6 - _tp) ** 2))

        n_fwd = 0
        p_star = np.asarray(row["p_star"], np.float64)
        ctrl = oracle.d_pose_shipped(pidx, f1_u8, p_star, s_t)
        two_at_pstar = mse(pose6_two(q16(p_star)))
        n_fwd += 1

        def gn_from(theta_init: np.ndarray, *, _pose6=pose6_two,
                    _tp=tp, _mse=mse) -> tuple[float, np.ndarray, int]:
            """Damped GN through the two-plane compose (mirrors solve_pair_gn);
            monotone acceptance at shipped f16 quantization."""
            nf = 0
            theta = q16(theta_init)
            cur6 = _pose6(theta)
            cur = _mse(cur6)
            nf += 1
            lm = 1.0
            for _ in range(RELINS):
                jac = np.zeros((6, 6), np.float64)
                for k in range(6):
                    dp = theta.copy()
                    dp[k] += d2m.FD_STEPS[k]
                    jac[:, k] = (_pose6(dp) - cur6) / d2m.FD_STEPS[k]
                nf += 6
                r = cur6 - _tp
                accepted = False
                for _damp in range(4):
                    a = jac.T @ jac + lm * np.diag(
                        np.maximum(np.diag(jac.T @ jac), 1e-8))
                    try:
                        step = np.linalg.solve(a, -(jac.T @ r))
                    except np.linalg.LinAlgError:
                        break
                    for scale in (1.0, 0.5):
                        cand = q16(theta + scale * step)
                        c6 = _pose6(cand)
                        nf += 1
                        cv = _mse(c6)
                        if cv < cur:
                            theta, cur6, cur = cand, c6, cv
                            accepted = True
                            break
                    if accepted:
                        lm = max(lm * 0.33, 1e-3)
                        break
                    lm *= 4.0
                if not accepted:
                    break
            return cur, theta, nf

        # MULTI-START: p0 (rotation-zeroed t_p) + p_star (the single-plane optimum)
        p0 = tp.copy()
        p0[3:] = 0.0
        cur_a, th_a, nf_a = gn_from(p0)
        cur_b, th_b, nf_b = gn_from(p_star)
        n_fwd += nf_a + nf_b
        cur, theta = (cur_a, th_a) if cur_a <= cur_b else (cur_b, th_b)

        rec = {
            "pair": int(pidx),
            "d_pose_warp": float(row["d_pose_warp"]),
            "d_single_solved_cached": float(row["d_pose_solved"]),
            "d_single_ctrl": float(ctrl),
            "d_two_at_pstar": float(two_at_pstar),
            "d_two_from_p0": float(cur_a),
            "d_two_from_pstar": float(cur_b),
            "d_two_solved": float(cur),
            "p_two_star": [float(v) for v in theta],
            "n_fwd": int(n_fwd),
            "wall_s": round(time.time() - t0, 1),
        }
        with JL.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[qa43-2p] pair {pidx}: warp {rec['d_pose_warp']:.4f} "
              f"single {rec['d_single_solved_cached']:.4f} (ctrl {ctrl:.4f}) "
              f"two@p* {two_at_pstar:.4f} two-solved {cur:.4f} "
              f"fwd {n_fwd} {rec['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
