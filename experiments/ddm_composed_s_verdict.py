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

THE SOLVER — a composed-S VERDICT needs only SOLVED d_pose VALUES (no gradients through any
receiver), so there is NO v4c archive / decode-grammar coupling. The realized pose landscape
is RAZOR-SHARP (tt1 FD: +/-1e-3 in pose[3] doubles d), so this uses a damped Levenberg-
Marquardt Gauss-Newton with the ANALYTIC (STE) Jacobian (torch autograd through the STE
round; the FD Jacobian is uint8-quantization-noise-limited, first-order Adam does not
converge). The warp is the FIXED EON/openpilot two-plane geometry (intrinsics_native +
horizon row 437 + camera height 1.22 m; a static-global constant, NOT a video-derived
payload); frame_0 = warp(frame_1, pose6). PoseNet is the frozen CPU authority (ddm_p3v2).

MEASURED CAVEAT (ddm_bc1 2026-07-30, decisive across FOUR solvers — Adam / FD-GN / analytic-GN
warp-pose6 / p3v2 cosine6-f0): a BOUNDED stage-exit ABSOLUTE pose solve on the burn's
seg-optimized frame_1 PLATEAUS at d_pose ~10-38 (all four), FAR above the trustworthy
post-burn ~0.0016. Diagnostic proof: GT_f0+GT_f1 -> d_pose ~1e-11 (target reachable), but
warp/basis pose-recovery from a single seg-render frame plateaus at ~10-38. The trustworthy
d_pose comes from the FULL jointly-optimized post-burn re-solve (pose + photometric + TTO,
MAIN's ~3.5h charter), NOT a bounded stage-exit solve. => the ABSOLUTE composed-S is NOT a
trustworthy endpoint acceptance number; the correct in-burn instrument is the DEGRADED
DIRECTIONAL DELTA (d_pose vs the un-dropped baseline at a fixed reference), a Knee-A externality
sign, which is an operator-GO'd downgrade of the all-or-nothing contract. This module is the
correct machinery for either MAIN's completion (the joint re-solve) or the delta instrument.

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
            import torch
            from ddm_p3v2_optimal_form_pose_resolve import (
                d_pose_u8,
                load_posenet,
                load_targets,
                pose6_u8,
            )
            from ddm_tt1_twin import _expmap_so3_torch  # differentiable Rodrigues (reuse)
        except Exception as exc:
            self.reason = f"geometry/posenet import failed: {exc}"
            return
        self.torch = torch
        self._expmap = _expmap_so3_torch
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
        # torch geometry (fp32 CPU) for the differentiable STE warp analytic Jacobian.
        f32 = torch.float32
        self.tK = torch.tensor(self.K, dtype=f32)
        self.tKinv = torch.tensor(self.Kinv, dtype=f32)
        self.tgrid = torch.tensor(self.grid, dtype=f32)
        self.tn = torch.tensor([0.0, -1.0, 0.0], dtype=f32)
        far = np.zeros(CAMERA_H * CAMERA_W, dtype=bool)
        far[: self.v_row * CAMERA_W] = True
        self.tfar = torch.tensor(far)
        self.camera_height_m = 1.22
        self.available = True

    # ---- differentiable STE two-plane warp -> pose6 out (analytic Jacobian; tt1 approach) --- #
    def _pose_to_H_t(self, pose6, s_t: float):
        torch = self.torch
        t = s_t * torch.stack([pose6[2], pose6[1], pose6[0]])
        R = self._expmap(torch.stack([pose6[3], pose6[4], pose6[5]]))
        M = R - torch.outer(t, self.tn) / self.camera_height_m
        return self.tK @ M @ self.tKinv

    def _warp_t(self, f1_flat, H):
        torch = self.torch
        src = torch.linalg.inv(H) @ self.tgrid
        z = src[2]
        su = src[0] / z
        sv = src[1] / z
        W1, H1 = CAMERA_W - 1, CAMERA_H - 1
        valid = (torch.isfinite(su) & torch.isfinite(sv) & (z > 0)
                 & (su >= 0) & (su <= W1) & (sv >= 0) & (sv <= H1))
        su_c = su.clamp(0.0, float(W1))
        sv_c = sv.clamp(0.0, float(H1))
        x0 = torch.floor(su_c).long()
        y0 = torch.floor(sv_c).long()
        x1 = torch.clamp(x0 + 1, max=W1)
        y1 = torch.clamp(y0 + 1, max=H1)
        wx = (su_c - x0.to(su_c.dtype)).unsqueeze(1)
        wy = (sv_c - y0.to(sv_c.dtype)).unsqueeze(1)
        Ia = f1_flat[y0 * CAMERA_W + x0]
        Ib = f1_flat[y0 * CAMERA_W + x1]
        Ic = f1_flat[y1 * CAMERA_W + x0]
        Id = f1_flat[y1 * CAMERA_W + x1]
        top = Ia * (1.0 - wx) + Ib * wx
        bot = Ic * (1.0 - wx) + Id * wx
        sampled = top * (1.0 - wy) + bot * wy
        return torch.where(valid.unsqueeze(1), sampled, f1_flat)

    def _pose6_diff(self, f1_flat, theta):
        """Differentiable 6-dim PoseNet output for theta=(pose6,a-1,b): two-plane warp + STE
        uint8 + PoseNet6. The STE round gives a smooth backward => an ANALYTIC Jacobian that
        the razor-sharp uint8 landscape needs (FD is quantization-noise-limited)."""
        torch = self.torch
        pose6, a, b = theta[:6], 1.0 + theta[6], theta[7]
        warp_g = self._warp_t(f1_flat, self._pose_to_H_t(pose6, 1.0))
        warp_f = self._warp_t(f1_flat, self._pose_to_H_t(pose6, 0.0))
        f0 = torch.where(self.tfar.unsqueeze(1), warp_f, warp_g)
        f0 = (a * f0 + b).reshape(CAMERA_H, CAMERA_W, 3).clamp(0.0, 255.0)
        f0 = f0 + (torch.round(f0) - f0).detach()  # STE uint8
        f1_cam = f1_flat.reshape(CAMERA_H, CAMERA_W, 3)
        x = torch.stack([f0.permute(2, 0, 1), f1_cam.permute(2, 0, 1)]).unsqueeze(0)
        out = self.posenet(self.posenet.preprocess_input(x))
        pose = out["pose"] if isinstance(out, dict) else out
        return pose[0, :6]

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

    def solve_d_pose(self, pair_idx: int, f1_cam_u8: np.ndarray, relins: int = 12,
                     lm_lambda: float = 1e-2, backtracks: int = 8) -> dict:
        """Damped LM Gauss-Newton over (pose6, a, b) from zero-init with the ANALYTIC (STE)
        Jacobian of the 6-dim residual r(theta) = PoseNet6(warp(f1,pose),f1) - target. Returns
        the REALIZED d_pose (shipped f16 quantization) at the best accepted step + trajectory.
        The analytic Jacobian (torch autograd through the STE round) converges the razor-sharp
        uint8 landscape where FD (quantization-noise-limited) and first-order Adam do not."""
        torch = self.torch
        net = self.posenet
        f1_u8 = f1_cam_u8.astype(np.uint8)
        target = np.asarray(self.targets[pair_idx], np.float64)
        f1_flat = torch.tensor(f1_u8.astype(np.float32)).reshape(-1, 3)
        target_t = torch.tensor(target.astype(np.float32))

        def resid_np(theta_np: np.ndarray):
            th = torch.tensor(theta_np.astype(np.float32))
            with torch.no_grad():
                r = (self._pose6_diff(f1_flat, th) - target_t).numpy().astype(np.float64)
            return r

        def jac_and_resid(theta_np: np.ndarray):
            th = torch.tensor(theta_np.astype(np.float32), requires_grad=True)
            p6 = self._pose6_diff(f1_flat, th)          # (6,) differentiable
            r = (p6.detach() - target_t).numpy().astype(np.float64)
            J = np.zeros((6, 8), dtype=np.float64)       # analytic Jacobian (6 vjp)
            for k in range(6):
                g = torch.autograd.grad(p6[k], th, retain_graph=(k < 5))[0]
                J[k] = g.numpy().astype(np.float64)
            return J, r

        theta = np.zeros(8, dtype=np.float64)
        r = resid_np(theta)
        cost = float(np.mean(r ** 2))
        best_theta, best_cost = theta.copy(), cost
        traj = [cost]
        lam = float(lm_lambda)
        for _ in range(int(relins)):
            J, r = jac_and_resid(theta)
            JtJ = J.T @ J
            g = J.T @ r
            accepted = False
            for _bt in range(int(backtracks)):
                try:
                    delta = -np.linalg.solve(JtJ + lam * np.eye(8), g)
                except np.linalg.LinAlgError:
                    lam = min(lam * 4.0, 1e6)
                    continue
                nt = theta + delta
                nc = float(np.mean(resid_np(nt) ** 2))
                if nc < cost:
                    theta, cost = nt, nc
                    lam = max(lam * 0.5, 1e-8)
                    accepted = True
                    if nc < best_cost:
                        best_theta, best_cost = nt.copy(), nc
                    break
                lam = min(lam * 3.0, 1e6)
            traj.append(cost)
            if not accepted or cost < 1e-6:
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
            "solver": "warp_pose6_analytic_lm_gn",  # damped LM-GN, analytic STE Jacobian
            "absolute_solve_trustworthy": False,  # MEASURED: bounded stage-exit solve plateaus
            # at d_pose ~10-38 on the seg-render f1 (4 solvers) vs post-burn ~0.0016; the
            # absolute composed_s is directional-only. Read residual_mse_traj for convergence.
            "score_claim": False,
            "axis": "[macOS-CPU advisory]",
            "note": "QA77-lite: warp-constrained analytic-LM-GN pose solve on the burn frame_1 "
                    "tail subset; VERDICT-level (never differentiated through). ABSOLUTE d_pose "
                    "is NOT trustworthy (bounded stage-exit solve plateaus far above the joint "
                    "post-burn re-solve, MAIN's charter); the trustworthy in-burn signal is the "
                    "directional delta vs the un-dropped baseline (operator-GO'd downgrade).",
        }
