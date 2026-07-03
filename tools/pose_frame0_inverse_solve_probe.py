#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""POSE FRAME0 INVERSE-SOLVE PROBE (#249) — the $0 DEEP-MATH pose gate for the P-E / P-F ladder rungs.

THE DECISIVE QUESTION (n600, real gt, NO-FAKE): SegNet reads only ``x[:,-1,...]`` = frame1, so **frame0
is SEG-FREE** (upstream/modules.py:108). PoseNet reads BOTH frames -> d_pose = MSE(PoseNet6(gen_pair) -
PoseNet6(orig_pair)). R1 (the trained store-nothing carrier) floored at d_pose ~0.0011 because it
constrained frame0 to a RIGID warp of the non-photoreal witness render. Does a FREE frame0, solved
against the FROZEN PoseNet, beat 0.0011 (toward ~0)? And is that win a CHEAP COUNTED parameterization
(P-F legal) or only an EXPENSIVE/ADVERSARIAL frame0 (the eval-hack fake)?

FOUR MEASUREMENTS (frozen CPU-torch PoseNet authority, NEVER MPS; through-R witness render f1):
  0. AFFINE CALIBRATION (step 1): fit PoseNet6(warp(f1,xi),f1) - PoseNet6(f1,f1) ~= A.xi over a handful
     of pairs (LS). A (6x6), its conditioning, and the LS R^2 characterize the physical-xi <-> PoseNet-6
     map (LDM Thm1 affine). This is the deep-math initializer for the warp base + the calibration the
     ladder assumes -- MEASURED, not guessed.
  1. P-E FREE-FRAME0 SOLVE: per pair, Levenberg-Marquardt Gauss-Newton on frame0 in the 6-dim pose
     subspace: f0 <- f0 - Jt (J Jt + lam I)^-1 r, J = d PoseNet6 / d frame0 (6xN, frame_slot=0). The
     min-norm step lives in the rank-<=6 row space of J (the SMALLEST frame0 change that hits the 6-dim
     target). Measured d_pose reported TWO ways: (a) gradient-space (work-res float) and (b) FROZEN
     AUTHORITY (camera-res uint8 roundtrip via cpu_verdict_d_pose) -- the (a)->(b) gap = does the
     adversarial precision survive the eval roundtrip? Plus ||delta||, per-pixel RMS, out-of-[0,255]
     fraction (legality).
  2. P-F (a) RANK-k J-BASIS SWEEP: truncated-pinv rank-k min-norm step for k=1..6 -> d_pose(k). The
     achievable floor per pose-DOF. Its basis (rows of J) is NOT decoder-reproducible (needs the frozen
     net) -> the EXPENSIVE/adversarial side of the rate ledger.
  3. P-F (b) GENERIC DCT-k SWEEP: represent the free-solve residual delta = f0_free - f0_warp in a GENERIC
     low-frequency 2D-DCT basis (decoder-REPRODUCIBLE, rule-118 free), keep k coeffs/channel, reconstruct,
     measure d_pose(k) through the frozen authority + the honest counted byte rate (3k int8 + zlib). The
     CHEAP counted side. Comparing (a) vs (b): if (b) needs vastly more coeffs than (a) for the same
     d_pose, the free-frame0 win is BASIS-ADVERSARIAL (needs the net) -> P-F floors near the warp; if (b)
     reaches the floor with few generic coeffs, it is CHEAPLY COUNTED -> P-F viable.

FIREWALL (NO-FAKE #6/#8 + rule-118 + EdgeBench App C eval-hack mirror): this MEASURES the achievable
d_pose + the counted-delta rate; it is NOT a production carrier. The legal carrier = generic warp (FREE,
deterministic) + COUNTED delta; a per-pair OPTIMIZED-adversarial frame0 stored as a table/"code" is the
eval-hack FAKE. The tool reports, HONESTLY, whether the free-frame0 solve is realizable as a cheap
COUNTED parameterization (DCT-k Pareto) or only as an expensive/adversarial one (rank-k J-basis gap), and
whether f0 stays a LEGAL image surviving the uint8/YUV6 roundtrip.

AUTHORITY: ``[macOS-CPU advisory / CPU-torch research-signal]`` NON-PROMOTABLE. Frozen CPU-torch PoseNet,
numpy/torch fp32/fp64; NEVER MPS; no CUDA; no paid eval. Frontier pointer 0.19110 UNMOVED. score_claim /
promotable = False. A DIAGNOSTIC that gates the grand-council optimal-form symposium (#250); NOT a score
(a real score needs the byte-close #238/#202). GT decodes via ``frame_utils.yuv420_to_rgb`` only (the
GT cache). NO-FAKE self-check asserts PoseNet(GT pair)==gt_poses (<1e-4) before any solve.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

np.seterr(all="ignore")

# 4-thread CPU (do NOT contend with any running MLX-GPU training). Set BEFORE torch import.
for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _refuse_tmp(path: Path, field: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field}={path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _advisory_axis() -> str:
    import platform

    if platform.system() == "Linux" and platform.machine().lower() in ("x86_64", "amd64"):
        return "[contest-CPU advisory] NON-PROMOTABLE"
    return "[macOS-CPU advisory / CPU-torch research-signal] NON-PROMOTABLE"


# ---------------------------------------------------------------------------
# torch helpers: differentiable PoseNet6(f0_work, f1_cam) with f0 as the free (3,work) variable.
# ---------------------------------------------------------------------------
def _posenet6(posenet, f0_work, f1_cam_t, camera_h, camera_w, quantize=True):
    """PoseNet6 of (f0_work upsampled to camera, f1_cam). f0_work: (3,wh,ww) torch, requires_grad.
    f1_cam_t: (3,Hc,Wc) torch. Returns (6,) torch pose6 with graph to f0_work.

    ``quantize`` (default True): apply the EVAL-FAITHFUL uint8 roundtrip to the camera-res frame0 via
    straight-through round+clamp (forward = round(clamp(.,0,255)); backward = identity) so the solve
    MINIMIZES the actual frozen-authority (uint8) d_pose and finds a uint8-ROBUST delta — not a sub-grey
    min-norm perturbation the eval roundtrip would destroy. This mirrors the trainer's through-R STE round."""
    import torch
    import torch.nn.functional as F

    f0_cam = F.interpolate(f0_work.unsqueeze(0), size=(camera_h, camera_w), mode="bilinear", align_corners=False)[0]
    if quantize:
        c = f0_cam.clamp(0.0, 255.0)
        f0_cam = c + (torch.round(c) - c).detach()  # STE round: forward uint8-quantized, backward identity
    x = torch.stack([f0_cam, f1_cam_t]).unsqueeze(0)  # (1,2,3,Hc,Wc)
    pin = posenet.preprocess_input(x)
    out = posenet(pin)
    pose = out["pose"] if isinstance(out, dict) else out
    return pose[0, :6].reshape(-1)


def _jacobian6(posenet, f0_work, f1_cam_t, camera_h, camera_w, quantize=True):
    """Return (pose6 np (6,), J np (6,N)) — the 6xN Jacobian d PoseNet6 / d f0_work at f0_work."""
    import torch

    f0_work = f0_work.detach().clone().requires_grad_(True)
    pose6 = _posenet6(posenet, f0_work, f1_cam_t, camera_h, camera_w, quantize=quantize)
    n = f0_work.numel()
    jac = np.zeros((6, n), dtype=np.float64)
    for k in range(6):
        g = torch.autograd.grad(pose6[k], f0_work, retain_graph=(k < 5), create_graph=False)[0]
        jac[k] = g.detach().reshape(-1).cpu().numpy().astype(np.float64)
    return pose6.detach().cpu().numpy().astype(np.float64), jac


def _frozen_dpose(posenet, f0_work_np, f1_cam_uint8, target, camera_h, camera_w):
    """The FROZEN AUTHORITY d_pose: upsample f0_work (3,wh,ww) -> camera, clip+round to uint8, run the
    exact per-pair cpu_verdict_d_pose (inference_mode, the eval-roundtrip authority). Returns
    (d_pose float, frac_out_of_range float, f0_uint8 (Hc,Wc,3))."""
    import torch
    import torch.nn.functional as F

    import train_witness_realized_through_R_mlx as twr

    f0w = torch.as_tensor(np.asarray(f0_work_np, dtype=np.float32))  # (3,wh,ww)
    f0_cam = F.interpolate(f0w.unsqueeze(0), size=(camera_h, camera_w), mode="bilinear", align_corners=False)[0]
    f0_cam = f0_cam.permute(1, 2, 0).numpy()  # (Hc,Wc,3) float
    oob = float(np.mean((f0_cam < 0.0) | (f0_cam > 255.0)))
    f0_uint8 = np.clip(np.round(f0_cam), 0, 255).astype(np.uint8)
    dp = twr.cpu_verdict_d_pose(posenet, f0_uint8, np.asarray(f1_cam_uint8, dtype=np.uint8), np.asarray(target))
    return float(dp), oob, f0_uint8


# ---------------------------------------------------------------------------
# render (reuse the canonical numpy-fp32 oracle through-R render; NO pose_carrier -> f1 = clean witness).
# ---------------------------------------------------------------------------
def render_all_witness(params, manifest, n_pairs, lbc):
    code_full = np.asarray(params["code"], np.float32)[: 2 * n_pairs]
    frames, _argmax = lbc.numpy_oracle_reference_frames(params, code_full, manifest, n_pairs)
    # f1 (the seg-scored frame) per pair. (f0 is the witness render too but we re-solve it freely.)
    f1s = [frames[2 * j + 1] for j in range(n_pairs)]
    f0_render = [frames[2 * j] for j in range(n_pairs)]  # the witness-render frame0 (init reference)
    return f1s, f0_render


# ---------------------------------------------------------------------------
# 0. affine A,b calibration: PoseNet6(warp(f1,xi),f1) - PoseNet6(f1,f1) ~= A.xi  (LS over cal pairs).
# ---------------------------------------------------------------------------
def calibrate_affine(posenet, f1s, gt, cal_idx, camera_h, camera_w):
    from tac.boundary_math.warp_real_luma_frame0 import GroundHomographyGeom, homography_from_xi_numpy
    from tac.boundary_math.keyframe_codec import warp_by_homography
    import train_witness_realized_through_R_mlx as twr

    geom = GroundHomographyGeom.eon(native_hw=(camera_h, camera_w))
    # xi perturbation design: 6 axes x SMALL local magnitudes (translation-first convention rho,omega).
    # LDM Thm1 affine identifiability is a LOCAL statement, so we linearize at xi=0 with small steps:
    # A = local d PoseNet6 / d xi through the render warp. (Large-xi extrapolation is handled by the
    # separate s_t-fit warp base, not by A.)
    axes_mags = []
    for j in range(6):
        for m in [-0.1, -0.03, 0.03, 0.1]:
            xi = np.zeros(6, np.float64); xi[j] = m
            axes_mags.append(xi)
    XI, DP = [], []
    per_pair = []
    for pidx in cal_idx:
        f1 = np.asarray(f1s[pidx], np.uint8)
        p0 = twr._cpu_pose_raw(posenet, f1, f1)  # PoseNet6(f1,f1) zero-motion
        for xi in axes_mags:
            H = homography_from_xi_numpy(xi, geom)
            f0w = warp_by_homography(f1, H, out_hw=(camera_h, camera_w))
            p = twr._cpu_pose_raw(posenet, f0w, f1)
            XI.append(xi); DP.append(p - p0)
        per_pair.append({"pair": int(pidx), "p0_zero_motion": [float(x) for x in p0]})
    XI = np.asarray(XI, np.float64); DP = np.asarray(DP, np.float64)  # (M,6),(M,6)
    # LS: DP = XI @ A^T  -> A^T = pinv(XI) @ DP ; A (6x6): dPose/dxi.
    At, *_ = np.linalg.lstsq(XI, DP, rcond=None)
    A = At.T
    pred = XI @ A.T
    ss_res = float(((DP - pred) ** 2).sum())
    ss_tot = float(((DP - DP.mean(0)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sv = np.linalg.svd(A, compute_uv=False)
    return {
        "A_6x6": A.tolist(),
        "A_singular_values": [float(x) for x in sv],
        "A_condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        "affine_fit_R2": float(r2),
        "n_cal_pairs": len(cal_idx),
        "n_samples": int(XI.shape[0]),
        "note": "PoseNet6(warp(f1,xi),f1)-PoseNet6(f1,f1) ~= A.xi ; A=dPose/dxi (LDM affine, per-pair b "
                "= PoseNet6(f1,f1)). High R^2 => the physical-xi<->PoseNet-6 map is affine as assumed.",
    }, A, geom


# ---------------------------------------------------------------------------
# 1. P-E free-frame0 LM Gauss-Newton solve.
# ---------------------------------------------------------------------------
def pe_free_solve(posenet, f0_init_work, f1_cam_t, target, camera_h, camera_w,
                  max_iter=8, recompute_J_every=1, clip=True):
    """LM-GN on f0_work to minimize ||PoseNet6(f0,f1)-target||^2. Returns dict with the solve trace +
    the final f0_work np + the final Jacobian J (6xN) at the solution (for the rank-k sweep)."""
    import torch

    tgt = np.asarray(target, np.float64)
    f0 = f0_init_work.detach().clone()
    lam = 1e-8
    n = f0.numel()
    J = None
    trace = []
    pose6, J = _jacobian6(posenet, f0, f1_cam_t, camera_h, camera_w)
    r = pose6 - tgt
    dpose_grad = float((r * r).mean())
    trace.append({"iter": 0, "dpose_grad": dpose_grad, "lam": lam})
    for it in range(1, max_iter + 1):
        JJt = J @ J.T
        max_eig = float(np.linalg.eigvalsh(JJt)[-1]) if n else 0.0
        accepted = False
        for _ls in range(6):  # LM damping line search
            try:
                coef = np.linalg.solve(JJt + lam * max_eig * np.eye(6) + 1e-30 * np.eye(6), r)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(JJt + lam * max_eig * np.eye(6), r, rcond=None)[0]
            delta = -(J.T @ coef)  # (N,) min-norm step
            f0_try = f0 + torch.as_tensor(delta.reshape(f0.shape), dtype=torch.float32)
            if clip:
                f0_try = f0_try.clamp(0.0, 255.0)
            with torch.no_grad():
                pt = _posenet6(posenet, f0_try.requires_grad_(False), f1_cam_t, camera_h, camera_w)
            rt = pt.detach().cpu().numpy().astype(np.float64) - tgt
            dpose_try = float((rt * rt).mean())
            if dpose_try < dpose_grad:
                f0 = f0_try
                r = rt
                dpose_grad = dpose_try
                lam = max(lam * 0.3, 1e-12)
                accepted = True
                break
            lam = min(lam * 5.0, 1e3)
        trace.append({"iter": it, "dpose_grad": dpose_grad, "lam": lam, "accepted": accepted,
                      "max_eig_JJt": max_eig})
        if not accepted:
            break
        if dpose_grad < 1e-9:
            break
        if it % recompute_J_every == 0 and it < max_iter:
            pose6, J = _jacobian6(posenet, f0, f1_cam_t, camera_h, camera_w)
            r = pose6 - tgt
    return {"f0_work": f0.detach().cpu().numpy().astype(np.float32), "J": J, "trace": trace,
            "dpose_grad_final": dpose_grad}


# ---------------------------------------------------------------------------
# 2. WARP BASE (P-A / rigid anchor): per-pair s_t-fit ground-homography warp of the witness render f1
#    (the FREE, rule-118 generic warp — reuses the PROVEN regime_homography + warp_rgb + s_t grid fit).
# ---------------------------------------------------------------------------
_ST_GRID = [0.005, 0.01, 0.02, 0.03, 0.044, 0.06, 0.08, 0.12, 0.16, 0.24]


def warp_base_fit(posenet, f1_cam, target, K_nat, Kinv_nat, grid_nat, camera_h, camera_w, work_h, work_w):
    """Fit the single translation-scale s_t (positive grid + s_t=0 null) that minimizes the FROZEN
    d_pose of the ground-homography warp of the witness render f1 (regime='ground'). Returns
    (best_s_t, dpose_warp_frozen, f0_warp_work np (3,wh,ww)). This is the P-A/R1-class rigid floor
    MEASURED in-harness on the NON-photoreal render (a fair generic-warp baseline)."""
    import torch
    import torch.nn.functional as F

    from tools.measure_pose_warp_dseg import regime_homography
    from tools.measure_screw_warp_through_R import _to_uint8, warp_rgb

    def _dp(s_t):
        H = regime_homography(np.asarray(target, np.float64), K_nat, Kinv_nat, (float(s_t), 0.0, 0.0), "ground")
        f0w_cam = _to_uint8(warp_rgb(np.asarray(f1_cam, np.float64), H, grid_nat))
        f0w = F.interpolate(torch.as_tensor(f0w_cam, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0),
                            size=(work_h, work_w), mode="bilinear", align_corners=False)[0].numpy()
        dp, _oob, _ = _frozen_dpose(posenet, f0w, f1_cam, target, camera_h, camera_w)
        return dp, f0w

    best_dp, best_f0 = _dp(0.0)
    best_st = 0.0
    for s in _ST_GRID:
        dp, f0w = _dp(s)
        if dp < best_dp:
            best_st, best_dp, best_f0 = float(s), dp, f0w
    return best_st, float(best_dp), best_f0


# ---------------------------------------------------------------------------
# 3. P-F: GENERIC-basis compressibility of the free residual delta = f0_free - f0_base.
#    The cheap-vs-adversarial verdict: how many bytes (in a DECODER-REPRODUCIBLE generic basis) to
#    retain the free-solve d_pose gain? THREE generic bases bracket compressibility:
#      (i)   largest-magnitude 2D-DCT-k / ch (DCT-sparse; k values + k uint32 indices counted)
#      (ii)  low-frequency 2D-DCT-k / ch    (smooth/low-freq; k values, indices IMPLICIT/free)
#      (iii) per-channel low-rank-r SVD     (low-rank; r*(H+W) fp16 counted)
#    If a modest budget reaches the free floor -> CHEAPLY COUNTED (P-F viable). If ALL need a large
#    budget -> the pose signal is high-frequency/edge-concentrated -> EXPENSIVE/ADVERSARIAL (needs the
#    non-reproducible J basis; a per-pair table would be the eval-hack fake).
# ---------------------------------------------------------------------------
def _dct2(x):
    from scipy.fft import dctn
    return dctn(x, type=2, norm="ortho", axes=(-2, -1))


def _idct2(x):
    from scipy.fft import idctn
    return idctn(x, type=2, norm="ortho", axes=(-2, -1))


def _lowfreq_mask(h, w, k):
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    order = np.lexsort((jj.ravel(), ii.ravel(), (ii + jj).ravel()))
    mask = np.zeros(h * w, bool); mask[order[:k]] = True
    return mask.reshape(h, w)


def pf_generic_compress(posenet, f0_free_work, f0_base_work, f1_cam, target, camera_h, camera_w,
                        k_list, r_list):
    """delta = f0_free - f0_base (3,wh,ww). Frozen d_pose + honest counted bytes for three generic
    decoder-reproducible representations of delta, swept over budget."""
    base = np.asarray(f0_base_work, np.float64)
    delta = np.asarray(f0_free_work, np.float64) - base  # (3,wh,ww)
    C, H, W = delta.shape
    out = {"dct_largest_mag": {}, "dct_lowfreq": {}, "lowrank_svd": {}}
    D = _dct2(delta)  # (3,H,W)

    for k in k_list:
        k = min(int(k), H * W)
        # (i) largest-magnitude coeffs (positions counted)
        recon_lm = np.zeros_like(D); bytes_lm = 0
        for c in range(C):
            flat = D[c].reshape(-1)
            idx = np.argpartition(np.abs(flat), -k)[-k:]
            m = np.zeros(H * W, bool); m[idx] = True
            recon_lm[c] = (flat * m).reshape(H, W)
            vals = flat[idx]; scale = np.abs(vals).max() + 1e-9
            q = np.clip(np.round(vals / scale * 127.0), -127, 127).astype(np.int8)
            raw = q.tobytes() + idx.astype(np.uint32).tobytes() + np.float32(scale).tobytes()
            bytes_lm += len(zlib.compress(raw, 9))
        f0lm = np.clip((base + _idct2(recon_lm)).astype(np.float32), 0, 255)
        dp_lm, oob_lm, _ = _frozen_dpose(posenet, f0lm, f1_cam, target, camera_h, camera_w)
        out["dct_largest_mag"][f"k{k}"] = {"d_pose_frozen": dp_lm, "coeffs_per_ch": int(k),
                                           "bytes_zlib": int(bytes_lm), "oob_frac": oob_lm}
        # (ii) low-frequency coeffs (positions implicit/free)
        mask = _lowfreq_mask(H, W, k)
        recon_lf = D * mask[None]
        bytes_lf = 0
        for c in range(C):
            vals = D[c][mask]; scale = np.abs(vals).max() + 1e-9
            q = np.clip(np.round(vals / scale * 127.0), -127, 127).astype(np.int8)
            bytes_lf += len(zlib.compress(q.tobytes() + np.float32(scale).tobytes(), 9))
        f0lf = np.clip((base + _idct2(recon_lf)).astype(np.float32), 0, 255)
        dp_lf, oob_lf, _ = _frozen_dpose(posenet, f0lf, f1_cam, target, camera_h, camera_w)
        out["dct_lowfreq"][f"k{k}"] = {"d_pose_frozen": dp_lf, "coeffs_per_ch": int(k),
                                       "bytes_zlib": int(bytes_lf), "oob_frac": oob_lf}

    for r in r_list:
        recon = np.zeros_like(delta); bytes_r = 0
        for c in range(C):
            U, s, Vt = np.linalg.svd(delta[c], full_matrices=False)
            rr = min(int(r), s.size)
            recon[c] = (U[:, :rr] * s[:rr]) @ Vt[:rr]
            Uf = (U[:, :rr] * np.sqrt(s[:rr])).astype(np.float16)
            Vf = (Vt[:rr].T * np.sqrt(s[:rr])).astype(np.float16)
            bytes_r += len(zlib.compress(Uf.tobytes() + Vf.tobytes(), 9))
        f0r = np.clip((base + recon).astype(np.float32), 0, 255)
        dp_r, oob_r, _ = _frozen_dpose(posenet, f0r, f1_cam, target, camera_h, camera_w)
        out["lowrank_svd"][f"r{r}"] = {"d_pose_frozen": dp_r, "rank": int(r),
                                       "bytes_zlib": int(bytes_r), "oob_frac": oob_r}
    return out


# ---------------------------------------------------------------------------
def run(args) -> dict:
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(4)
    t0 = time.time()
    authority = _advisory_axis()

    import tools.levelset_byte_close_and_eval as lbc
    import train_witness_realized_through_R_mlx as twr
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # upstream
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally

    ckpt_dir = Path(args.ckpt_dir)
    params, cfg = lbc._load_levelset_ckpt(ckpt_dir, args.npz_name)
    so = lbc.detect_self_orient(cfg, {"freq_across": 32.0, "freq_along": 4.0, "tau": 4.0, "iters": 4})
    blob, _bd = lbc.build_levelset_blob(params, cfg, so, None)
    manifest, _b, _c, _p, _lane, _pc = lbc._read_blob_bytes(blob)
    n_total = int(cfg["n_pairs"])
    n_pairs = min(int(args.n_pairs), n_total)
    camera_h, camera_w = lbc.CAMERA_H, lbc.CAMERA_W
    work_h, work_w = int(args.work_h), int(args.work_w)

    # GT + frozen PoseNet (patched yuv6 so the Jacobian is not severed)
    gt, _seg_cpu, _pn = twr.load_gt_from_cache(Path(args.gt_cache), n_total)
    patch_upstream_yuv6_globally()
    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet = dn.posenet
    for p_ in posenet.parameters():
        p_.requires_grad = False

    # NO-FAKE self-check: PoseNet(GT pair) == gt_poses under the patch (proves the differentiable yuv6 is
    # numerically faithful so the verdict d_pose authority is unchanged).
    sc = {"pairs": min(4, n_total), "pose_max_abs_err": 0.0}
    for p in range(sc["pairs"]):
        dp0 = twr.cpu_verdict_d_pose(posenet, gt.gt_f0[p], gt.gt_f1[p], gt.gt_poses[p])
        sc["pose_max_abs_err"] = max(sc["pose_max_abs_err"], float(dp0))
    sc["PASS"] = bool(sc["pose_max_abs_err"] < 1e-4)
    if not sc["PASS"]:
        raise SystemExit(f"NO-FAKE self-check FAILED under yuv6 patch: {sc}")
    print(f"[probe] NO-FAKE self-check PASS (pose_err={sc['pose_max_abs_err']:.2e})  {authority}", flush=True)

    # render ALL n witness pairs through R (f1 = clean witness render; f0 re-solved freely). Cache the
    # rendered f1 frames to npz so RESUME chunks (Bash caps a call at 10 min -> we run in --max-seconds
    # chunks) skip the ~2-4 min re-render.
    # witness-frame cache on the SSD tier (bulky rebuildable artifact per CLAUDE.md disk-hygiene rule).
    _ssd = next((Path(p) for p in ("/Volumes/VertigoDataTier/pact", "/Volumes/APDataStore/pact")
                 if Path(p).is_dir()), _REPO / "experiments/results")
    fcache = _ssd / f"pose_frame0_probe_cache/{Path(args.out).stem}.witness_f1.npz"
    fcache.parent.mkdir(parents=True, exist_ok=True)
    _refuse_tmp(fcache, "witness_cache")
    if fcache.exists() and not args.fresh:
        print(f"[probe] loading cached witness f1 frames from {fcache}", flush=True)
        z = np.load(fcache)
        f1s = [z["f1"][i] for i in range(min(n_pairs, z["f1"].shape[0]))]
        if len(f1s) < n_pairs:
            raise SystemExit(f"cache has {len(f1s)} < {n_pairs} frames; delete {fcache} to re-render")
    else:
        print(f"[probe] rendering {n_pairs} witness pairs through R (render {cfg['render_h']}x{cfg['render_w']}, "
              f"self_orient={so.get('self_orient')})...", flush=True)
        f1s, _f0_render = render_all_witness(params, manifest, n_pairs, lbc)
        np.savez_compressed(fcache, f1=np.stack([np.asarray(f, np.uint8) for f in f1s], 0))
        print(f"[probe] rendered {n_pairs} pairs in {time.time()-t0:.0f}s (cached -> {fcache.name})", flush=True)

    # native intrinsics + target grid for the ground-homography warp base (reuse the proven engine)
    from tools.measure_pose_warp_dseg import _target_grid, intrinsics_at
    K_nat = intrinsics_at(camera_w, camera_h)
    Kinv_nat = np.linalg.inv(K_nat)
    grid_nat = _target_grid(camera_h, camera_w)

    # 0. affine calibration (REPORTED characterization: how affine is the render-warp xi->PoseNet map?)
    cal_idx = list(range(0, n_pairs, max(1, n_pairs // args.n_cal)))[: args.n_cal]
    print(f"[probe] affine A,b calibration on {len(cal_idx)} pairs...", flush=True)
    affine, _A, _geom = calibrate_affine(posenet, f1s, gt, cal_idx, camera_h, camera_w)
    print(f"[probe] affine fit R2={affine['affine_fit_R2']:.4f} cond(A)={affine['A_condition_number']:.3e}", flush=True)

    # per-pair solve
    rows = []
    jsonl = Path(args.out).with_suffix(".partial.jsonl")
    _refuse_tmp(jsonl, "partial_jsonl")
    done = set()
    if jsonl.exists() and not args.fresh:
        for ln in jsonl.read_text().splitlines():
            try:
                r = json.loads(ln); done.add(int(r["pair_index"])); rows.append(r)
            except Exception:
                pass
    fjl = open(jsonl, "a")
    k_pf = [16, 64, 256, 1024, 4096]
    r_pf = [1, 2, 4, 8, 16]
    coarse_grids = [(24, 32), (48, 64), (96, 128)]
    loop_t0 = time.time()

    for pidx in range(n_pairs):
        if pidx in done:
            continue
        if args.max_seconds and (time.time() - loop_t0) > args.max_seconds:
            print(f"[probe] --max-seconds {args.max_seconds} reached at pair {pidx} "
                  f"({len(rows)}/{n_pairs} done); exiting cleanly (resume via partial JSONL).", flush=True)
            break
        tp = time.time()
        f1 = np.asarray(f1s[pidx], np.uint8)
        target = np.asarray(gt.gt_poses[pidx], np.float64)
        f1_cam_t = torch.as_tensor(f1, dtype=torch.float32).permute(2, 0, 1)
        # init: f0_work = downsample(f1) (zero-motion generic init)
        f0_init_work = F.interpolate(f1_cam_t.unsqueeze(0), size=(work_h, work_w), mode="bilinear",
                                     align_corners=False)[0].clone()

        # zero-motion baseline d_pose (frozen)
        dpose_zero, _, _ = _frozen_dpose(posenet, f0_init_work.numpy(), f1, target, camera_h, camera_w)

        # WARP BASE (P-A anchor): per-pair s_t-fit ground-homography warp of the witness render f1 (FREE).
        # This is the generic-warp start the P-E free solve refines FROM (f0=f1 is too far for the large
        # forward-motion targets -> LM-GN stalls; the warp already carries the bulk of the ego motion).
        best_st, dpose_warp, f0_warp_work = warp_base_fit(
            posenet, f1, target, K_nat, Kinv_nat, grid_nat, camera_h, camera_w, work_h, work_w)

        # P-E free solve (FULL work res), INITIALIZED from the warp base (delta = f0_free - f0_warp).
        # quantization-aware (STE round) so it minimizes the ACTUAL uint8 frozen d_pose (the eval floor).
        f0_warp_t = torch.as_tensor(f0_warp_work, dtype=torch.float32)
        sol = pe_free_solve(posenet, f0_warp_t, f1_cam_t, target, camera_h, camera_w,
                            max_iter=args.max_iter)
        f0_free = sol["f0_work"]
        dpose_free_frozen, oob_free, _ = _frozen_dpose(posenet, f0_free, f1, target, camera_h, camera_w)
        delta_tot = f0_free - f0_warp_work
        delta_rms = float(np.sqrt((delta_tot ** 2).mean()))
        delta_max = float(np.abs(delta_tot).max())

        # P-F CHEAP-COUNTED test: coarse-resolution free solves. A frame0 solved at a COARSE grid is a
        # LOW-DOF, decoder-reproducible (bilinear-upsampled), cheaply-STORABLE delta. If a coarse free
        # solve reaches low frozen d_pose, the pose signal is SMOOTH -> cheaply counted (P-F viable). If
        # only the full-res solve works, the signal is high-freq/edge-concentrated -> adversarial.
        coarse = {}
        if args.pf:
            for (ch, cw) in coarse_grids:
                f0_warp_c = F.interpolate(torch.as_tensor(f0_warp_work, dtype=torch.float32).unsqueeze(0),
                                          size=(ch, cw), mode="bilinear", align_corners=False)[0]
                solc = pe_free_solve(posenet, f0_warp_c, f1_cam_t, target, camera_h, camera_w,
                                     max_iter=max(4, args.max_iter - 2))
                f0c = solc["f0_work"]
                # frozen d_pose of the coarse solve (upsample coarse->camera->uint8 is inside _frozen_dpose,
                # but here f0c is (3,ch,cw); _frozen_dpose upsamples work->camera regardless of work size).
                dpc, oobc, _ = _frozen_dpose(posenet, f0c, f1, target, camera_h, camera_w)
                # counted bytes: the coarse delta over the coarse warp base, int8-quantized + zlib.
                dc = (f0c - f0_warp_c.numpy())
                sc = np.abs(dc).max() + 1e-9
                q = np.clip(np.round(dc / sc * 127.0), -127, 127).astype(np.int8)
                by = len(zlib.compress(q.tobytes() + np.float32(sc).tobytes(), 9))
                coarse[f"{ch}x{cw}"] = {"d_pose_frozen": float(dpc), "dof": int(3 * ch * cw),
                                        "bytes_zlib": int(by), "oob_frac": float(oobc)}

        # P-F: generic-basis compressibility of the free residual over the WARP base (the real P-F:
        # generic warp free + counted delta) AND over the f1-init base (conservative: no warp).
        pf_over_warp, pf_over_init = {}, {}
        if args.pf_generic:
            pf_over_warp = pf_generic_compress(posenet, f0_free, f0_warp_work, f1, target,
                                               camera_h, camera_w, k_pf, r_pf)
            pf_over_init = pf_generic_compress(posenet, f0_free, f0_init_work.numpy(), f1, target,
                                               camera_h, camera_w, k_pf, r_pf)

        row = {
            "pair_index": int(pidx),
            "target_pose6": [float(x) for x in target],
            "d_pose_zero_motion": dpose_zero,
            "d_pose_free_frozen": dpose_free_frozen,
            "d_pose_free_grad": sol["dpose_grad_final"],
            "d_pose_warp_base": dpose_warp,
            "warp_s_t": best_st,
            "delta_rms_grey": delta_rms,
            "delta_max_grey": delta_max,
            "oob_frac_free": oob_free,
            "n_iter": len(sol["trace"]) - 1,
            "pf_coarse_free_solve": coarse,
            "pf_over_warp": pf_over_warp,
            "pf_over_init": pf_over_init,
            "secs": round(time.time() - tp, 2),
        }
        rows.append(row)
        fjl.write(json.dumps(row) + "\n"); fjl.flush()
        if pidx < 8 or pidx % 50 == 0:
            print(f"  [{pidx+1}/{n_pairs}] zero={dpose_zero:.3g} warp={dpose_warp:.3g}(st={best_st:.3g}) "
                  f"free_grad={sol['dpose_grad_final']:.3g} free_frozen={dpose_free_frozen:.3g} "
                  f"dRMS={delta_rms:.2f} oob={oob_free:.3g} it={row['n_iter']} ({row['secs']:.1f}s)", flush=True)
    fjl.close()

    # aggregate
    def _stat(key, sub=None):
        vals = []
        for r in rows:
            v = r
            for kk in (sub or []):
                v = v.get(kk, {}) if isinstance(v, dict) else {}
            v = v.get(key) if isinstance(v, dict) else None
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        a = np.asarray(vals, np.float64)
        if a.size == 0:
            return {"n": 0}
        return {"mean": float(a.mean()), "median": float(np.median(a)), "min": float(a.min()),
                "max": float(a.max()), "n": int(a.size)}

    def _dpose_contrib(d):
        return float(np.sqrt(10.0 * d)) if (d is not None and d >= 0) else None

    zero_s = _stat("d_pose_zero_motion")
    warp_s = _stat("d_pose_warp_base")
    free_grad_s = _stat("d_pose_free_grad")
    free_frozen_s = _stat("d_pose_free_frozen")
    drms_s = _stat("delta_rms_grey")
    oob_s = _stat("oob_frac_free")

    # P-F curves: median d_pose + median bytes per budget, for each (base, method).
    def _pf_curve(base_field, method, labels):
        curve = {}
        for lab in labels:
            dvals, bvals = [], []
            for r in rows:
                cell = r.get(base_field, {}).get(method, {}).get(lab) if isinstance(r.get(base_field), dict) else None
                if cell and cell.get("d_pose_frozen") is not None and np.isfinite(cell["d_pose_frozen"]):
                    dvals.append(float(cell["d_pose_frozen"]))
                    if cell.get("bytes_zlib") is not None:
                        bvals.append(float(cell["bytes_zlib"]))
            if dvals:
                curve[lab] = {"d_pose_median": float(np.median(dvals)),
                              "contrib_median": _dpose_contrib(float(np.median(dvals))),
                              "bytes_median": (float(np.median(bvals)) if bvals else None), "n": len(dvals)}
        return curve

    klab = [f"k{k}" for k in [16, 64, 256, 1024, 4096]]
    rlab = [f"r{r}" for r in [1, 2, 4, 8, 16]]
    pf = {}
    coarse_curve = {}
    if args.pf_generic:
        for base in ("pf_over_warp", "pf_over_init"):
            pf[base] = {
                "dct_largest_mag": _pf_curve(base, "dct_largest_mag", klab),
                "dct_lowfreq": _pf_curve(base, "dct_lowfreq", klab),
                "lowrank_svd": _pf_curve(base, "lowrank_svd", rlab),
            }
    if args.pf:
        # coarse-resolution free-solve curve (the fair SMOOTH cheaply-counted delta)
        for lab in ["24x32", "48x64", "96x128"]:
            dvals = [r["pf_coarse_free_solve"][lab]["d_pose_frozen"] for r in rows
                     if r.get("pf_coarse_free_solve", {}).get(lab)]
            bvals = [r["pf_coarse_free_solve"][lab]["bytes_zlib"] for r in rows
                     if r.get("pf_coarse_free_solve", {}).get(lab)]
            dof = next((r["pf_coarse_free_solve"][lab]["dof"] for r in rows
                        if r.get("pf_coarse_free_solve", {}).get(lab)), None)
            if dvals:
                coarse_curve[lab] = {"d_pose_median": float(np.median(dvals)),
                                     "contrib_median": _dpose_contrib(float(np.median(dvals))),
                                     "bytes_median": float(np.median(bvals)), "dof": dof, "n": len(dvals)}

    R1_FLOOR = 0.0011
    ANCESTOR = 3.4e-5
    free_med = free_frozen_s.get("median")
    beats_r1 = bool(free_med is not None and free_med < R1_FLOOR)
    warp_med = warp_s.get("median")

    # RATE-AWARE cheap-vs-adversarial verdict. THE decisive accounting: a per-pair frame0 residual is
    # stored for EACH of the n_pairs pairs -> its rate contribution is 25 * bytes_per_pair * n_pairs /
    # 37_545_489 (the archive-byte term). To keep the pose carrier's rate NEGLIGIBLE (< ~0.01 of S) the
    # per-pair store must be ~ (0.01 * 37_545_489 / 25 / n_pairs) bytes ~= 25 bytes/pair at n600 -- i.e.
    # a ~6-scalar (xi / pose-residual) store, NOT an image-space field. So "cheaply counted" means the
    # NET (pose contribution + rate contribution) BEATS R1's cheap pose floor (contribution 0.105 at
    # ~0 added rate, R1's stored-xi mechanism).
    ARCHIVE_BYTES = 37_545_489.0
    R1_CONTRIB = float(np.sqrt(10.0 * R1_FLOOR))  # ~0.105

    def _rate_contrib(bytes_per_pair):
        return 25.0 * float(bytes_per_pair) * n_pairs / ARCHIVE_BYTES

    cheap_verdict = None
    if args.pf and coarse_curve:
        # net S impact of each image-space (coarse) carrier = pose contribution + rate contribution.
        coarse_net = {}
        best_net = None  # (net, grid, pose_contrib, rate_contrib)
        for lab, v in coarse_curve.items():
            pc = v["contrib_median"]; rc = _rate_contrib(v["bytes_median"])
            net = (pc or 0.0) + rc
            coarse_net[lab] = {"pose_contrib": pc, "rate_contrib_at_n600_scale": rc, "net_S": net,
                               "bytes_per_pair": v["bytes_median"], "d_pose": v["d_pose_median"]}
            if best_net is None or net < best_net[0]:
                best_net = (net, lab, pc, rc)
        # a cheaply-counted image-space carrier must BEAT R1's cheap pose floor NET of its rate cost.
        cheap_realizable = bool(best_net is not None and best_net[0] < R1_CONTRIB)
        cheap_verdict = {
            "free_solve_floor_median_dpose": free_med,
            "free_solve_floor_contrib": _dpose_contrib(free_med),
            "warp_base_floor_median_dpose": warp_med,
            "R1_cheap_pose_floor_contrib": R1_CONTRIB,
            "note_free_solve_is_adversarial": (
                "the full-res free-frame0 floor (~%.2e) is reached by a tiny (sub-grey) but SPATIALLY-"
                "PRECISE high-frequency delta; it is NOT decoder-reproducible and per-pair image storage "
                "is rate-prohibitive." % (free_med if free_med else 0.0)),
            "coarse_carrier_NET_S_at_n%d_scale" % n_pairs: coarse_net,
            "cheapest_NET_coarse_carrier": (
                {"grid": best_net[1], "net_S": best_net[0], "pose_contrib": best_net[2],
                 "rate_contrib": best_net[3]} if best_net else None),
            "generic_basis_curves_if_run": pf.get("pf_over_warp", {}),
            "CHEAPLY_COUNTED_REALIZABLE": bool(cheap_realizable),
            "interpretation": (
                "CHEAPLY COUNTED: a smooth low-DOF (coarse) free frame0 carrier BEATS R1's cheap pose "
                "floor NET of its per-pair rate cost -> a legal image-space pose carrier improves S." if
                cheap_realizable else
                "EXPENSIVE / ADVERSARIAL / RATE-PROHIBITIVE: the full-res free-frame0 floor (~%.0e, contrib "
                "%.4f) is real but reached only by a high-frequency sub-grey adversarial delta (NOT "
                "generic-codeable). A SMOOTH coarse free frame0 CAN reach low d_pose but its per-pair "
                "image store is rate-prohibitive at n%d scale (25*bytes*%d/37.5M) -> NET S worse than R1's "
                "cheap 0.105 floor. So the free-frame0 win is NOT cheaply legal-realizable. The LEGAL cheap "
                "pose floor stays ~R1's 0.0011 (contrib 0.105): warp base (rule-118 free) + a ~6-scalar "
                "POSE-SPACE stored residual (xi), NOT an image-space frame0. A per-pair optimized frame0 "
                "stored as a table would be the eval-hack fake (NO-FAKE #6/#8)." % (
                    free_med or 0.0, _dpose_contrib(free_med) or 0.0, n_pairs, n_pairs)),
        }

    report = {
        "tool": "tools/pose_frame0_inverse_solve_probe.py",
        "gate": "#249 $0 deep-math pose frame0 inverse-solve probe (P-E / P-F)",
        "utc": _utc(),
        "authority": authority,
        "score_claim": False, "promotion_eligible": False, "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False, "promotable": False,
        "frontier_pointer": "UNMOVED 0.19110 (advisory diagnostic; not a contest score)",
        "ckpt_dir": str(ckpt_dir), "npz_name": cfg.get("npz_name"),
        "config": {k: cfg.get(k) for k in ("hidden_dim", "n_hidden", "mod_dim", "activation",
                                           "render_h", "render_w", "chroma")},
        "n_pairs": n_pairs, "work_resolution": [work_h, work_w], "self_orient": so.get("self_orient"),
        "no_fake_selfcheck": sc,
        "0_affine_calibration": affine,
        "1_P_E_free_frame0_solve": {
            "d_pose_zero_motion": zero_s, "contrib_zero": _dpose_contrib(zero_s.get("median")),
            "d_pose_warp_base_P_A": warp_s, "contrib_warp": _dpose_contrib(warp_s.get("median")),
            "d_pose_free_gradient_space": free_grad_s, "contrib_free_grad": _dpose_contrib(free_grad_s.get("median")),
            "d_pose_free_FROZEN_AUTHORITY": free_frozen_s, "contrib_free_frozen": _dpose_contrib(free_frozen_s.get("median")),
            "delta_rms_grey_levels": drms_s, "oob_frac": oob_s,
            "R1_rigid_warp_floor_anchor": R1_FLOOR, "ancestor_rgb_anchor": ANCESTOR,
            "BEATS_R1_0.0011": beats_r1,
            "grad_vs_frozen_gap": (free_frozen_s.get("median", 0) - free_grad_s.get("median", 0)),
        },
        "warp_s_t_median": _stat("warp_s_t").get("median"),
        "2_3_P_F_rate_pareto": {
            "coarse_free_solve_SMOOTH_cheaply_counted": coarse_curve,
            "over_warp_base_GENERIC_free_smooth_plus_counted_delta": pf.get("pf_over_warp", {}),
            "over_f1_init_CONSERVATIVE_whole_motion_counted": pf.get("pf_over_init", {}),
            "cheap_vs_adversarial_verdict": cheap_verdict,
        },
        "verdicts": {
            "free_frame0_beats_R1_rigid_warp": beats_r1,
            "free_frozen_d_pose_median": free_med,
            "free_frozen_contrib_median": _dpose_contrib(free_med),
            "warp_base_d_pose_median": warp_med,
            "affine_map_R2": affine["affine_fit_R2"],
            "delta_survives_uint8_roundtrip": bool(free_frozen_s.get("median") is not None and
                                                   free_grad_s.get("median") is not None and
                                                   (free_frozen_s["median"] - free_grad_s["median"]) < 1e-3),
            "cheaply_counted_realizable": (cheap_verdict or {}).get("CHEAPLY_COUNTED_REALIZABLE"),
        },
        "caveats": [
            "SAMPLE: n=%d pairs (contiguous). THROUGH-R witness render f1 (self-orient shippable render); "
            "frame0 solved at work res %dx%d, verdict at camera %dx%d uint8 (eval-roundtrip)." % (
                n_pairs, work_h, work_w, camera_h, camera_w),
            "GRADIENT vs FROZEN: the LM-GN minimizes the work-res float d_pose; the FROZEN AUTHORITY "
            "d_pose re-measures at camera-res uint8 (inference_mode cpu_verdict_d_pose). The gap is the "
            "eval-roundtrip cost of the solved frame0 (small gap = robust, not knife-edge adversarial).",
            "FIREWALL: P-E is the ACHIEVABLE floor (per-pair adversarial-capable free frame0). P-F measures "
            "whether a DECODER-REPRODUCIBLE generic-basis (DCT / low-rank) counted delta over the FREE "
            "generic warp base reaches that floor cheaply. A per-pair optimized frame0 stored as a table "
            "is the eval-hack fake (NO-FAKE #6/#8); only the counted generic-basis delta + generic warp is legal.",
            "WARP BASE: per-pair s_t-fit ground-homography warp of the witness render f1 (rule-118 FREE, "
            "reuses the proven regime_homography+warp_rgb). This is the P-A/R1-class rigid floor MEASURED "
            "in-harness on the NON-photoreal render; R1's trained store-nothing 0.0011 is a different "
            "(trained per-pair residual) mechanism. The AFFINE A,b calibration is REPORTED as a "
            "characterization of how affine the render-warp xi->PoseNet map is (low R^2 => weakly affine).",
            "AUTHORITY: frozen CPU-torch PoseNet + differentiable yuv6 patch (NO-FAKE self-check "
            "PoseNet(GT)==gt_poses <1e-4). NEVER MPS. This DIAGNOSTIC does NOT move the pointer.",
        ],
        "elapsed_secs": round(time.time() - t0, 1),
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", type=Path,
                    default=_REPO / "experiments/results/levelset_n600_R1_storenothing_descent_ev1_20260703T004906Z")
    ap.add_argument("--npz-name", type=str, default="levelset_witness_ema_BEST.npz")
    ap.add_argument("--gt-cache", type=str,
                    default=str(_REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"))
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--n-cal", type=int, default=8, help="pairs for the affine A,b calibration")
    ap.add_argument("--max-iter", type=int, default=8, help="LM-GN iterations for the free solve")
    ap.add_argument("--work-h", type=int, default=384)
    ap.add_argument("--work-w", type=int, default=512)
    ap.add_argument("--pf", action="store_true", help="run the P-F coarse-resolution free-solve rate sweep")
    ap.add_argument("--pf-generic", action="store_true",
                    help="ALSO run the (slower) generic DCT/low-rank compressibility sweep of the full delta")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop the per-pair loop after this many seconds (resume via partial JSONL); "
                         "0 = no limit. Use to chunk the n600 run under the 10-min Bash cap.")
    ap.add_argument("--fresh", action="store_true", help="ignore any existing partial JSONL + frame cache")
    ap.add_argument("--out", type=Path,
                    default=_REPO / "reports/pose_frame0_inverse_solve_probe.json")
    args = ap.parse_args(argv)
    _refuse_tmp(Path(args.out), "out")
    report = run(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("\n=== POSE FRAME0 INVERSE-SOLVE VERDICT ===", flush=True)
    print(json.dumps(report["verdicts"], indent=2), flush=True)
    print(f"[report] wrote {args.out}  {report['authority']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
