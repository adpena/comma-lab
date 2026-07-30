# SPDX-License-Identifier: MIT
"""ddm_bc1 §3.5 — QA77-LITE composed-S stage-exit verdict for the QA24 seg re-burn.

The QA24 coarse grid DROPS the sky (grid rows 0-4) + lower-hood (rows 20-23) cells from
birth (sg1 §2), which FREEZES far-field content. The co9 bidirectional law prices that on
the POSE axis (Knee-A: sky/hood freeze costs pose). This module prices it: at the burn's
stage exits, on a BOUNDED subset (the pose-mass TAIL, MAIN QA66: top-17 = 74.3% of pose
mass), it runs a bounded WARP-CONSTRAINED terminal pose solve (6-dim pose + 2-param
photometric) against the burn's rendered frame_1 and reports COMPOSED S = 100*d_seg +
sqrt(10*d_pose) + 25*bytes/37_545_489 so stage/endpoint decisions see the pose cost.

VERDICT-LEVEL ONLY — NEVER differentiated through the burn's training graph and NEVER changes
the trained tokens/weights or shipped bytes (the burn optimizes seg-only; the full pose
re-solve is MAIN's post-burn charter). It is an EARLY-WARNING instrument.

THE SOLVER (MAIN recall steer 2026-07-30) — a composed-S VERDICT needs only SOLVED d_pose
VALUES (no gradients through any receiver), so there is NO v4c archive / decode-grammar
coupling. The realized pose landscape is RAZOR-SHARP (tt1 FD: +/-1e-3 in pose[3] doubles d)
so first-order Adam does NOT converge; the PROVEN converger is a damped Levenberg-Marquardt
Gauss-Newton with a FINITE-DIFFERENCE Jacobian (the eg1 E3 / tt1 approach: ~6-8 PoseNet
forwards per relinearization, 2-3 relins, seconds/pair), realized-acceptance line search.
The warp is the FIXED EON/openpilot two-plane geometry (intrinsics_native + horizon row 437 +
camera height 1.22 m; a static-global constant, NOT a video-derived payload) from the pfs1
receiver; frame_0 = warp(frame_1, pose6) so the burn's frame_1 freeze cost propagates into
the solved d_pose. PoseNet is the frozen CPU authority (ddm_p3v2). All numpy + CPU forwards.

Axis: [macOS-CPU advisory]. score_claim=false. Pointer 0.1910828242 [contest-CPU] UNMOVED.
FAILS GRACEFULLY (available=False, reason set) if the pfs1 geometry / PoseNet / targets are
unavailable — an advisory verdict must never crash the burn.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CAMERA_H, CAMERA_W = 874, 1164
SCORE_RATE_DENOM = 37_545_489.0  # contest rate denominator (25*bytes/denom)

_REPO = Path(__file__).resolve().parents[1]


def _find_pfs1_dir() -> Path | None:
    """Locate the pfs1 warp-receiver source tree (fixed geometry: intrinsics_native etc.).
    Prefers the tt1-twin's canonical path; falls back to any SSD submissions/pfs1 dir."""
    cand = Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/pfs1")
    if (cand / "pfs1_warp_receiver.py").is_file():
        return cand
    for base in ("/Volumes/VertigoDataTier/pact", "/Volumes/APDataStore/pact"):
        b = Path(base)
        if not b.is_dir():
            continue
        for p in b.glob("**/submissions/pfs1/pfs1_warp_receiver.py"):
            return p.parent
    return None


class ComposedSVerdict:
    """Warp-constrained FD-Jacobian LM-GN pose solver on the burn's frame_1 -> composed S.

    Construction loads the fixed geometry + CPU PoseNet + targets ONCE. ``available`` is
    False (with ``reason`` set) when any dependency is missing; callers skip the verdict.
    """

    def __init__(self, num_pairs: int) -> None:
        self.available = False
        self.reason = ""
        self.num_pairs = int(num_pairs)
        pfs1 = _find_pfs1_dir()
        if pfs1 is None:
            self.reason = "pfs1 warp-receiver geometry tree not found on SSD"
            return
        for p in (str(_REPO / "src"), str(_REPO / "experiments"), str(_REPO / "upstream"),
                  str(pfs1)):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            import pfs1_warp_receiver as pw  # fixed EON geometry (0-byte generic code)
            from ddm_p3v2_optimal_form_pose_resolve import (
                d_pose_u8,
                load_posenet,
                load_targets,
                pose6_u8,
            )
        except Exception as exc:
            self.reason = f"geometry/posenet import failed: {exc}"
            return
        self._pw = pw
        self._pose6_u8 = pose6_u8
        self._d_pose_u8 = d_pose_u8
        self.posenet, _ = load_posenet()          # CPU, frozen (realized authority)
        self.targets = load_targets(self.num_pairs)
        self.K = pw.intrinsics_native()
        self.Kinv = np.linalg.inv(self.K)
        self.grid = pw._target_grid(CAMERA_H, CAMERA_W)
        self.v_row = round(float(self.K[1, 2]))   # geometric horizon row (=437 for EON)
        self._far = (np.arange(CAMERA_H)[:, None] < self.v_row) & np.ones((1, CAMERA_W), bool)
        self.available = True

    # ---- numpy realized warp (two-plane static compose; mirrors v4c Decoder._warp_pair) --- #
    def _warp_f0(self, f1_u8: np.ndarray, pose: np.ndarray, a: float, b: float,
                 quantize: bool) -> np.ndarray:
        """Realized frame_0 = two-plane warp(frame_1, pose6) then photometric a*.+b, then uint8.
        ``quantize`` True applies the SHIPPED f16 pose/(a,b) quantization (final realized eval);
        False keeps full precision (the FD-Jacobian solve, so f16 granularity never swamps eps)."""
        pw = self._pw
        p = np.asarray(pose, np.float64)
        if quantize:
            p = p.astype(np.float16).astype(np.float64)
            a, b = float(np.float16(a)), float(np.float16(b))
        f1_f = f1_u8.astype(np.float64)
        hg = pw.pose_to_homography(p, self.K, self.Kinv, 1.0, 1.0, 0.0)
        wg = pw.warp_rgb(f1_f, hg, self.grid)
        hf = pw.pose_to_homography(p, self.K, self.Kinv, 0.0, 1.0, 0.0)  # far plane (s_t=0)
        wf = pw.warp_rgb(f1_f, hf, self.grid)
        f0f = np.where(self._far[..., None], wf, wg)
        if a != 1.0 or b != 0.0:
            f0f = a * f0f + b
        return pw._to_uint8(f0f)

    def solve_d_pose(self, pair_idx: int, f1_cam_u8: np.ndarray, relins: int = 3,
                     fd_eps: float = 1.5e-3, lm_lambda: float = 1e-3, backtracks: int = 6) -> dict:
        """Damped LM Gauss-Newton over (pose6, a, b) from zero-init, FINITE-DIFFERENCE Jacobian
        of the 6-dim realized residual r(theta) = PoseNet6(warp(f1,pose),f1) - target. Returns
        the REALIZED d_pose (shipped f16 quantization) at the best accepted step + trajectory."""
        net = self.posenet
        f1_u8 = f1_cam_u8.astype(np.uint8)
        target = np.asarray(self.targets[pair_idx], np.float64)

        def resid(theta: np.ndarray) -> np.ndarray:
            f0 = self._warp_f0(f1_u8, theta[:6], 1.0 + theta[6], theta[7], quantize=False)
            return self._pose6_u8(net, f0, f1_u8) - target  # (6,)

        theta = np.zeros(8, dtype=np.float64)
        r = resid(theta)
        cost = float(np.mean(r ** 2))
        best_theta, best_cost = theta.copy(), cost
        traj = [cost]
        lam = float(lm_lambda)
        for _ in range(int(relins)):
            J = np.zeros((6, 8), dtype=np.float64)  # FD Jacobian (8 forwards)
            for k in range(8):
                tp = theta.copy()
                tp[k] += fd_eps
                J[:, k] = (resid(tp) - r) / fd_eps
            JtJ = J.T @ J
            g = J.T @ r
            accepted = False
            for _bt in range(int(backtracks)):
                try:
                    delta = -np.linalg.solve(JtJ + lam * np.eye(8), g)
                except np.linalg.LinAlgError:
                    lam = min(lam * 4.0, 1e4)
                    continue
                nt = theta + delta
                nr = resid(nt)
                nc = float(np.mean(nr ** 2))
                if nc < cost:
                    theta, r, cost = nt, nr, nc
                    lam = max(lam * 0.5, 1e-7)
                    accepted = True
                    if nc < best_cost:
                        best_theta, best_cost = nt.copy(), nc
                    break
                lam = min(lam * 2.0, 1e4)
            traj.append(cost)
            if not accepted:
                break
        # realized d_pose at the SHIPPED f16 quantization (the frozen CPU-PoseNet acceptor).
        f0 = self._warp_f0(f1_u8, best_theta[:6], 1.0 + best_theta[6], best_theta[7],
                           quantize=True)
        d_realized = float(self._d_pose_u8(net, f0, f1_u8, target))
        return {"d_pose": d_realized, "residual_mse_traj": traj, "relins_run": len(traj) - 1}

    def composed_s(self, subset_ids: list[int], cams_u8: list[np.ndarray],
                   dseg_subset: float, total_counted_bytes: int, relins: int = 3) -> dict:
        """Composed-S verdict over a bounded subset. ``cams_u8`` are the burn's camera-res
        (H,W,3) uint8 frame_1 renders for ``subset_ids`` (the same the seg gate produced)."""
        solves = [self.solve_d_pose(i, cam, relins=relins)
                  for i, cam in zip(subset_ids, cams_u8, strict=True)]
        dposes = [s["d_pose"] for s in solves]
        d_pose_mean = float(np.mean(dposes))
        rate = 25.0 * float(total_counted_bytes) / SCORE_RATE_DENOM
        composed = 100.0 * float(dseg_subset) + float(np.sqrt(10.0 * d_pose_mean)) + rate
        return {
            "composed_s": composed,
            "seg_contrib": 100.0 * float(dseg_subset),
            "pose_contrib": float(np.sqrt(10.0 * d_pose_mean)),
            "rate_contrib": rate,
            "d_seg_subset": float(dseg_subset),
            "d_pose_solved_mean": d_pose_mean,
            "d_pose_per_pair": [float(x) for x in dposes],
            "relins_run_per_pair": [int(s["relins_run"]) for s in solves],
            "n_subset": len(subset_ids),
            "solver": "warp_pose6_fd_lm_gn",  # damped LM-GN, FD Jacobian (eg1/tt1 proven)
            "score_claim": False,
            "axis": "[macOS-CPU advisory]",
            "note": "QA77-lite: warp-constrained FD-LM-GN pose solve on the burn frame_1 tail "
                    "subset; VERDICT-level (never differentiated through); prices co9 sky/hood-"
                    "freeze pose cost. Full v4c pose re-solve is MAIN's post-burn charter.",
        }
