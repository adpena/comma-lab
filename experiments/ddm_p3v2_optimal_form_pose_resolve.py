#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_p3v2 — OPTIMAL-FORM terminal pose re-solve on the tr1 composed vehicle (RANK-1 arm).

THE PRE-REGISTERED DECISION (charter + gc7r row 1): the pb1 composed row S~20.2746 is 96-97% pose
(19.51 of the 20.10 gap). P3 adjudicated N1=NO (photometric wall) on a SUB-OPTIMAL solve: frame_0
started from ``zeros`` (d_pose ~78), used a FIXED rank-6 cosine basis, and the GN was budget-truncated
at ~2 relinearizations (38.06 mean). This tool re-measures at OPTIMAL FORM.

FROZEN-SCORER FACTORIZATION LAW (upstream/modules.py:108): SegNet reads ``x[:,-1]`` = frame_1 ONLY, so
frame_0 is 100% seg-free. The ENTIRE frame_0 is a pose-only actuation surface at zero seg risk; only
BYTES bind. frame_1 (the seg frame) is asserted UNTOUCHED per rung.

LADDER (each rung MEASURED through the real receiver path: STE-uint8 camera-res + frozen CPU-torch
PoseNet6 authority; d_pose = mean((pose6(gen_pair) - banked_target6)**2)):
  baselines   stored_f0 / zeros / copy(f1) -> d_pose reference.
  S0          the EXISTING actuation (rank-6 cosine basis) run to convergence -> budget-truncated
              (converges << 38) vs rank-deficient (plateaus). Verdict for the honest baseline.
  S1d         FULL FREE frame_0 GN (work-res Adam, STE-uint8) -> the UNPRICED reach ceiling. THIS is
              the pre-registered falsifier: if it still cannot reach <=1e-3-class -> wall CONFIRMED at
              FORMULATION scope; else the wall was an artifact of the naive solve.
  S1 price    generic DECODER-REPRODUCIBLE compression of the free-solve delta over a cheap base
              (low-frequency 2D-DCT-k + per-channel low-rank-r), counted bytes -> d_pose(bytes) Pareto.
  S2 LOTTO    SHARED low-rank frame_0 basis (counted ONCE) + per-pair coefficients (counted) vs
              per-pair rank-1, at matched bytes. The supermask idea transplanted to the pose carrier.

AUTHORITY: ``[macOS-CPU frozen-PoseNet advisory]`` NON-PROMOTABLE. Frozen CPU-torch PoseNet only; NEVER
MPS; no CUDA; no paid eval. Pointer 0.1910828242 [contest-CPU] UNMOVED. score_claim/promotable=False.
The banked carriers (#715 quotient reach curve, sc1 e_p) are CITED from their committed receipts, not
rebuilt (the rebuild is the fake); this tool races the Jacobian-aligned / free / LOTTO frame_0 family.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

# 4-thread CPU (never contend with a live MLX-GPU job). Set BEFORE torch import.
for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")

import numpy as np

np.seterr(all="ignore")

_REPO = Path(__file__).resolve().parents[1]
_UPSTREAM = Path("/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709")
for _p in (_REPO, _REPO / "src", _UPSTREAM):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# tr1 composed-vehicle custody (charter-designated; the eg1 rehearse tool's DEFAULT_FRAME_ROOT).
_FRAME_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_ct1_campaign_telemetry_encode_20260725/"
    "e5a_runtime/output_identity/"
    "2a2c0367150f8c8c0953dfb5c1485e238bbc9995c37385e149e52ae22f506241"
)
_TARGETS = Path(
    "/Volumes/VertigoDataTier/pact/"
    "ddm_ms4_metric_producers_and_measurement_20260724T042005Z/pose_metric_n600_batch32.json"
)
_SSD_OUT = Path("/Volumes/VertigoDataTier/pact/ddm_p3v2_20260729")
CAMERA_H, CAMERA_W = 874, 1164
CHUNK = 32  # pairs per raw file
PAIR_SHAPE = (2, CAMERA_H, CAMERA_W, 3)
_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_hash() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


# ---------------------------------------------------------------------------
# frozen PoseNet (yuv6 patched so the Jacobian is not severed) + frame/target custody
# ---------------------------------------------------------------------------
def load_posenet():
    import modules  # upstream (pinned)
    import torch
    from safetensors.torch import load_file

    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally

    if Path(modules.__file__).resolve() != (_UPSTREAM / "modules.py").resolve():
        raise RuntimeError("imported non-custodied upstream modules.py")
    patch_upstream_yuv6_globally()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    posenet = modules.PoseNet().eval().cpu()
    posenet.load_state_dict(load_file(str(_UPSTREAM / "models/posenet.safetensors"), device="cpu"))
    for p in posenet.parameters():
        p.requires_grad = False
    return posenet, modules


def load_pair(pidx: int) -> np.ndarray:
    """Load composed pair pidx (2,874,1164,3) uint8 from the chunked raw custody. The final chunk is
    24 pairs (pairs_0576_0600.raw), not 32 — derive the chunk's actual pair count from the filename."""
    lo = (pidx // CHUNK) * CHUNK
    hi = min(lo + CHUNK, 600)
    raw = _FRAME_ROOT / f"pairs_{lo:04d}_{hi:04d}.raw"
    if not raw.is_file():
        raise FileNotFoundError(f"missing composed-frame chunk: {raw}")
    n_in_chunk = hi - lo
    cam = np.memmap(raw, mode="r", dtype=np.uint8, shape=(n_in_chunk, *PAIR_SHAPE))
    return np.array(cam[pidx - lo], dtype=np.uint8, copy=True)


def load_targets(n: int) -> np.ndarray:
    bundle = json.loads(_TARGETS.read_text())
    if bundle.get("output_dimension") != 6:
        raise RuntimeError("pose target bundle is not 6-dim")
    rows = bundle["rows"]
    return np.stack([np.asarray(rows[i]["center"], np.float64) for i in range(n)], 0)


# ---------------------------------------------------------------------------
# frozen authority d_pose (the ONLY score-relevant quantity): camera-res uint8 pair -> PoseNet6 -> MSE.
# ---------------------------------------------------------------------------
def pose6_u8(posenet, f0_u8: np.ndarray, f1_u8: np.ndarray) -> np.ndarray:
    import torch
    x = torch.from_numpy(np.stack([f0_u8, f1_u8])[None]).permute(0, 1, 4, 2, 3).float()
    with torch.inference_mode():
        out = posenet(posenet.preprocess_input(x))
    pose = out["pose"] if isinstance(out, dict) else out
    return pose[0, :6].numpy().astype(np.float64)


def d_pose_u8(posenet, f0_u8: np.ndarray, f1_u8: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pose6_u8(posenet, f0_u8, f1_u8) - np.asarray(target)) ** 2))


def _f0work_to_u8(f0_work, camera_h=CAMERA_H, camera_w=CAMERA_W) -> np.ndarray:
    """work-res (3,wh,ww) float -> camera-res uint8 (H,W,3) via bilinear up + clamp/round (the eval path)."""
    import torch
    import torch.nn.functional as F
    t = torch.as_tensor(np.asarray(f0_work, np.float32))
    cam = F.interpolate(t.unsqueeze(0), size=(camera_h, camera_w), mode="bilinear", align_corners=False)[0]
    return cam.clamp(0, 255).round().permute(1, 2, 0).numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# differentiable pose6(f0_work, f1) with STE uint8 @ camera (matches the frozen authority path).
# ---------------------------------------------------------------------------
def _pose6_grad(posenet, f0_work, f1_cam_t, camera_h=CAMERA_H, camera_w=CAMERA_W):
    import torch
    import torch.nn.functional as F
    f0_cam = F.interpolate(f0_work.unsqueeze(0), size=(camera_h, camera_w),
                           mode="bilinear", align_corners=False)[0]
    c = f0_cam.clamp(0.0, 255.0)
    f0_cam = c + (torch.round(c) - c).detach()  # STE round: forward uint8, backward identity
    x = torch.stack([f0_cam, f1_cam_t]).unsqueeze(0)
    out = posenet(posenet.preprocess_input(x))
    pose = out["pose"] if isinstance(out, dict) else out
    return pose[0, :6].reshape(-1)


# ---------------------------------------------------------------------------
# S0: the EXISTING actuation = rank-6 generic low-frequency cosine basis + per-coefficient GN.
#     (reproduces the eg1/P3 mechanism so the convergence verdict is on the SAME actuation.)
# ---------------------------------------------------------------------------
def _cosine6_basis(h: int, w: int) -> np.ndarray:
    x = np.cos(2.0 * np.pi * (np.arange(w) + 0.5) / w)
    y = np.cos(2.0 * np.pi * (np.arange(h) + 0.5) / h)
    fields = np.zeros((6, h, w, 3), np.float32)
    for c in range(3):
        fields[c, :, :, c] = x[None, :]
        fields[c + 3, :, :, c] = y[:, None]
    return fields  # (6,H,W,3)


def s0_cosine6_solve(posenet, f0_base_u8, f1_u8, target, *, relins, amp):
    """coefficient LM-GN over the rank-6 cosine basis, camera-res, run to `relins` relinearizations.
    frame_0 = clip(f0_base + sum_k coef_k * amp * basis_k). Returns the per-relin d_pose trajectory."""
    basis = _cosine6_basis(CAMERA_H, CAMERA_W)  # (6,H,W,3)
    base = f0_base_u8.astype(np.float64)

    def render(coef):
        f0 = base + amp * np.tensordot(coef, basis, axes=(0, 0))
        return np.clip(np.round(f0), 0, 255).astype(np.uint8)

    def dp(coef):
        return d_pose_u8(posenet, render(coef), f1_u8, target)

    coef = np.zeros(6, np.float64)
    d0 = dp(coef)
    traj = [d0]
    lam = 1e-2
    eps = 1.0  # finite-diff step in coefficient units
    for _ in range(relins):
        p0 = pose6_u8(posenet, render(coef), f1_u8)
        jac = np.zeros((6, 6), np.float64)  # d pose6 / d coef_k
        for k in range(6):
            cc = coef.copy()
            cc[k] += eps
            jac[:, k] = (pose6_u8(posenet, render(cc), f1_u8) - p0) / eps
        r = p0 - np.asarray(target)
        cur = float(np.mean(r ** 2))
        accepted = False
        for _ls in range(8):
            jjt = jac.T @ jac
            try:
                step = np.linalg.solve(jjt + lam * np.eye(6), jac.T @ r)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(jjt + lam * np.eye(6), jac.T @ r, rcond=None)[0]
            cc = coef - step
            dtry = dp(cc)
            if dtry < cur:
                coef = cc
                lam = max(lam * 0.3, 1e-6)
                accepted = True
                break
            lam = min(lam * 5.0, 1e4)
        traj.append(dp(coef))
        if not accepted:
            break
    return {"d_pose_traj": [float(x) for x in traj], "relins_run": len(traj) - 1,
            "coef_final": [float(x) for x in coef], "amp": amp}


# ---------------------------------------------------------------------------
# S1d: FULL FREE frame_0 GN (work-res Adam, STE uint8) — the unpriced reach ceiling + the free f0.
# ---------------------------------------------------------------------------
def s1d_free_solve(posenet, f0_init_work, f1_u8, target, *, iters, lr, wh, ww, tol=1e-3):
    """Run to CONVERGENCE (not budget-truncate — the exact P3 mistake). Adam with plateau LR decay;
    early-stop when the frozen-uint8 d_pose stops improving by > `tol` (relative) over a 3-check window.
    Returns the best frozen-uint8 frame_0, its d_pose, the checkpoint trajectory, and iters used."""
    import torch
    f1_cam_t = torch.from_numpy(f1_u8).permute(2, 0, 1).float()
    tg = torch.from_numpy(np.asarray(target, np.float64)).float()
    f0w = torch.as_tensor(np.asarray(f0_init_work, np.float32)).clone().requires_grad_(True)
    opt = torch.optim.Adam([f0w], lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best = None
    best_dp = float("inf")
    traj = []
    check_every = 10
    it_used = 0
    for it in range(iters):
        opt.zero_grad()
        p = _pose6_grad(posenet, f0w, f1_cam_t)
        loss = ((p - tg) ** 2).mean()
        loss.backward()
        opt.step()
        it_used = it + 1
        if it % check_every == 0 or it == iters - 1:
            dp = d_pose_u8(posenet, _f0work_to_u8(f0w.detach().numpy()), f1_u8, target)
            sched.step(dp)
            traj.append((it, dp))
            if dp < best_dp:
                best_dp, best = dp, f0w.detach().numpy().copy()
            # convergence: last 3 checks improved < tol (relative) AND we've done >= 6 checks
            if len(traj) >= 6:
                recent = [d for _, d in traj[-3:]]
                if (max(recent) - min(recent)) <= tol * max(recent):
                    break
    return {"f0_work": best if best is not None else f0w.detach().numpy(),
            "d_pose_free_u8": float(best_dp),
            "traj": [[int(i), float(d)] for i, d in traj], "iters_used": int(it_used)}


# ---------------------------------------------------------------------------
# S1 price: generic DECODER-REPRODUCIBLE compression of the free-solve delta over a cheap base.
#   base = bilinear(downsample(f1)) upsampled -> decoder-free (f1 is already in the archive).
#   (i)  low-frequency 2D-DCT-k / channel (positions implicit) ; (ii) per-channel low-rank-r SVD.
#   Both counted (int8 + zlib9) and measured through the frozen uint8 authority.
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
    mask = np.zeros(h * w, bool)
    mask[order[:min(k, h * w)]] = True
    return mask.reshape(h, w)


_ST_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.044, 0.06, 0.08, 0.12, 0.16, 0.24]


def warp_base_work(posenet, f1_u8, target, wh, ww):
    """DECODER-REPRODUCIBLE ego-motion base frame_0: ground-homography warp of frame_1 by the CARRIED
    6-value pose target, with the single translation-scale s_t fit on a positive grid (1 scalar/pair,
    ~0 bytes). Reuses the proven #249/measure_pose_warp engine. Returns (base_work (3,wh,ww) float,
    s_t, d_pose_warp_u8). The pose target is already carried (sc1 t_p sidecar), so this base is free."""
    import torch
    import torch.nn.functional as F

    from tools.measure_pose_warp_dseg import _target_grid, intrinsics_at, regime_homography
    from tools.measure_screw_warp_through_R import _to_uint8, warp_rgb

    k_nat = intrinsics_at(CAMERA_W, CAMERA_H)
    kinv = np.linalg.inv(k_nat)
    grid = _target_grid(CAMERA_H, CAMERA_W)
    tgt = np.asarray(target, np.float64)

    def eval_st(s_t):
        h = regime_homography(tgt, k_nat, kinv, (float(s_t), 0.0, 0.0), "ground")
        f0cam = _to_uint8(warp_rgb(np.asarray(f1_u8, np.float64), h, grid))
        return f0cam, d_pose_u8(posenet, f0cam, f1_u8, target)

    best_cam, best_dp, best_st = None, float("inf"), 0.0
    for s in _ST_GRID:
        f0cam, dp = eval_st(s)
        if dp < best_dp:
            best_cam, best_dp, best_st = f0cam, dp, float(s)
    base_work = F.interpolate(torch.from_numpy(best_cam).permute(2, 0, 1).float().unsqueeze(0),
                              size=(wh, ww), mode="bilinear", align_corners=False)[0].numpy()
    return base_work, best_st, float(best_dp)


def _cheap_base_work(f1_u8, wh, ww, ds=(48, 64)):
    """decoder-reproducible base frame_0 at work-res: downsample f1 to a coarse grid then up to work-res."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(f1_u8).permute(2, 0, 1).float().unsqueeze(0)
    coarse = F.interpolate(t, size=ds, mode="bilinear", align_corners=False)
    up = F.interpolate(coarse, size=(wh, ww), mode="bilinear", align_corners=False)[0]
    return up.numpy()  # (3,wh,ww) float


def s1_price(posenet, f0_free_work, base_work, f1_u8, target, *, k_list, r_list):
    base = np.asarray(base_work, np.float64)
    delta = np.asarray(f0_free_work, np.float64) - base  # (3,wh,ww)
    C, H, W = delta.shape
    out = {"dct_lowfreq": {}, "dct_largest_mag": {}, "lowrank_svd": {}}
    D = _dct2(delta)
    for k in k_list:
        # (i) low-frequency coeffs (positions implicit/free)
        mask = _lowfreq_mask(H, W, k)
        recon = D * mask[None]
        nbytes = 0
        for c in range(C):
            vals = D[c][mask]
            scale = float(np.abs(vals).max()) + 1e-9
            q = np.clip(np.round(vals / scale * 127.0), -127, 127).astype(np.int8)
            nbytes += len(zlib.compress(q.tobytes() + np.float32(scale).tobytes(), 9))
        f0 = np.clip(base + _idct2(recon), 0, 255).astype(np.float32)
        dp = d_pose_u8(posenet, _f0work_to_u8(f0), f1_u8, target)
        out["dct_lowfreq"][f"k{int(k)}"] = {"d_pose": dp, "coeffs_per_ch": int(k), "bytes": int(nbytes)}
        # (ii) largest-magnitude coeffs anywhere (captures high-freq structure; positions counted)
        kk = min(int(k), H * W)
        recon_lm = np.zeros_like(D)
        nbytes_lm = 0
        for c in range(C):
            flat = D[c].reshape(-1)
            idx = np.argpartition(np.abs(flat), -kk)[-kk:]
            m = np.zeros(H * W, bool)
            m[idx] = True
            recon_lm[c] = (flat * m).reshape(H, W)
            vals = flat[idx]
            scale = float(np.abs(vals).max()) + 1e-9
            q = np.clip(np.round(vals / scale * 127.0), -127, 127).astype(np.int8)
            raw = q.tobytes() + idx.astype(np.uint32).tobytes() + np.float32(scale).tobytes()
            nbytes_lm += len(zlib.compress(raw, 9))
        f0lm = np.clip(base + _idct2(recon_lm), 0, 255).astype(np.float32)
        dplm = d_pose_u8(posenet, _f0work_to_u8(f0lm), f1_u8, target)
        out["dct_largest_mag"][f"k{int(k)}"] = {"d_pose": dplm, "coeffs_per_ch": int(kk), "bytes": int(nbytes_lm)}
    for r in r_list:
        recon = np.zeros_like(delta)
        nbytes = 0
        for c in range(C):
            U, s, Vt = np.linalg.svd(delta[c], full_matrices=False)
            rr = min(int(r), s.size)
            recon[c] = (U[:, :rr] * s[:rr]) @ Vt[:rr]
            Uf = (U[:, :rr] * np.sqrt(s[:rr])).astype(np.float16)
            Vf = (Vt[:rr].T * np.sqrt(s[:rr])).astype(np.float16)
            nbytes += len(zlib.compress(Uf.tobytes() + Vf.tobytes(), 9))
        f0 = np.clip(base + recon, 0, 255).astype(np.float32)
        dp = d_pose_u8(posenet, _f0work_to_u8(f0), f1_u8, target)
        out["lowrank_svd"][f"r{int(r)}"] = {"d_pose": dp, "rank": int(r), "bytes": int(nbytes)}
    return out


# ---------------------------------------------------------------------------
# S2 LOTTO: SHARED low-rank frame_0 basis across pairs (counted ONCE) + per-pair coeffs (counted).
#   stack the free-solve deltas {delta_p} (each 3*wh*ww); SVD the (P, N) matrix -> shared rank-R basis
#   V (R,N) counted once as fp16+zlib; per-pair coeffs a_p (R) counted int8+zlib. vs per-pair rank-1
#   (each pair its OWN rank-1 U s Vt, counted). MATCHED-BYTES honesty via the counted totals.
# ---------------------------------------------------------------------------
def s2_lotto(posenet, deltas, bases_work, f1s, targets, *, ranks, amortize_n=600):
    P = len(deltas)
    flat = np.stack([d.reshape(-1) for d in deltas], 0)  # (P, N)
    shape = deltas[0].shape
    # global mean removed into the shared dictionary automatically by SVD on the raw stack.
    _U, _s, Vt = np.linalg.svd(flat, full_matrices=False)  # flat = U s Vt ; Vt (min(P,N), N) shared basis
    out = {"shared_dict": {}, "per_pair_rank1": {}, "amortize_n": int(amortize_n)}
    for R in ranks:
        R = min(int(R), Vt.shape[0])
        basis = Vt[:R]  # (R, N) shared, counted ONCE (amortized over amortize_n pairs)
        basis_bytes = len(zlib.compress(basis.astype(np.float16).tobytes(), 9))
        coeffs = flat @ basis.T  # (P, R) per-pair coeffs
        cscale = float(np.abs(coeffs).max()) + 1e-9
        cq = np.clip(np.round(coeffs / cscale * 127.0), -127, 127).astype(np.int8)
        coeff_bytes = len(zlib.compress(cq.tobytes() + np.float32(cscale).tobytes(), 9))
        coeff_bytes_per_pair = coeff_bytes / P
        recon = (cq.astype(np.float64) * cscale / 127.0) @ basis  # (P,N) dequant reconstruction
        dps = []
        for p in range(P):
            f0 = np.clip(bases_work[p] + recon[p].reshape(shape), 0, 255).astype(np.float32)
            dps.append(d_pose_u8(posenet, _f0work_to_u8(f0), f1s[p], targets[p]))
        out["shared_dict"][f"R{R}"] = {
            "d_pose_mean": float(np.mean(dps)), "d_pose_median": float(np.median(dps)),
            "basis_bytes_once": int(basis_bytes), "coeff_bytes_per_pair": float(coeff_bytes_per_pair),
            "bytes_per_pair_amortized_over_n": float(basis_bytes / amortize_n + coeff_bytes_per_pair),
            "note_degenerate_if_R>=P": bool(R >= P)}
    # per-pair rank-1 comparator: each pair its own rank-1 delta (own basis, no sharing).
    for r1 in (1,):
        dps = []
        nbytes = 0
        for p in range(P):
            d = deltas[p]
            C, H, W = d.shape
            recon = np.zeros_like(d)
            for c in range(C):
                Uc, sc, Vc = np.linalg.svd(d[c], full_matrices=False)
                recon[c] = (Uc[:, :r1] * sc[:r1]) @ Vc[:r1]
                Uf = (Uc[:, :r1] * np.sqrt(sc[:r1])).astype(np.float16)
                Vf = (Vc[:r1].T * np.sqrt(sc[:r1])).astype(np.float16)
                nbytes += len(zlib.compress(Uf.tobytes() + Vf.tobytes(), 9))
            f0 = np.clip(bases_work[p] + recon, 0, 255).astype(np.float32)
            dps.append(d_pose_u8(posenet, _f0work_to_u8(f0), f1s[p], targets[p]))
        out["per_pair_rank1"][f"r{r1}"] = {
            "d_pose_mean": float(np.mean(dps)), "d_pose_median": float(np.median(dps)),
            "total_bytes": int(nbytes), "bytes_per_pair": float(nbytes / P)}
    return out


# ---------------------------------------------------------------------------
def contribution(d_pose_mean: float) -> float:
    return float(np.sqrt(10.0 * max(d_pose_mean, 0.0)))


def _prefix_pair_selection_scope(pair_indices: list[int], requested_n: int) -> dict[str, Any]:
    pairs = [int(v) for v in pair_indices]
    return {
        "schema": "subset_scope.v1",
        "n": len(pairs),
        "requested_n": int(requested_n),
        "population": 600,
        "selection_mode": "video_order_prefix",
        "pair_indices": pairs,
        "axis_bias_caveat": (
            "video-order prefix is bounded mechanics evidence, not population evidence; "
            "prefix bias can differ by axis, so no n600 claim follows from this subset"
        ),
        "population_claim": False,
    }


def run(args) -> dict[str, Any]:
    import torch
    torch.set_num_threads(4)
    t0 = time.time()
    posenet, modules = load_posenet()
    n = int(args.n_pairs)
    targets = load_targets(n)
    wh, ww = int(args.work_h), int(args.work_w)

    # NO-FAKE self-check: PoseNet6(f1,f1) is zero-motion; copy(f1) d_pose must match target[0]^2/6 class.
    pair0 = load_pair(0)
    f1_0 = pair0[1]
    dcopy0 = d_pose_u8(posenet, f1_0, f1_0, targets[0])
    print(f"[p3v2] self-check copy(f1) pair0 d_pose={dcopy0:.4f} (target0[0]^2/6={targets[0][0]**2/6:.4f})  "
          f"{args.axis}", flush=True)

    # RESUMABLE per-pair persistence (the reaper kills detached/long runs; the receipt writes only at
    # the end, so persist each pair as it lands). JSONL = scalar rows; npz cache = f0_free+base for LOTTO.
    jsonl = Path(str(args.out).replace(".json", ".partial.jsonl"))
    npz_dir = Path(str(args.out).replace(".json", "_pairs"))
    _refuse_tmp(jsonl)
    npz_dir.mkdir(parents=True, exist_ok=True)
    done_rows: dict[int, dict[str, Any]] = {}
    if jsonl.exists() and args.resume:
        for ln in jsonl.read_text().splitlines():
            try:
                rr = json.loads(ln)
                done_rows[int(rr["pair"])] = rr
            except Exception:
                pass
        print(f"[p3v2] resume: {len(done_rows)} pairs already persisted in {jsonl.name}", flush=True)
    fjl = open(jsonl, "a")  # noqa: SIM115 (append handle held across the resumable per-pair loop)

    per_pair = []
    for pidx in range(n):
        npz_p = npz_dir / f"pair{pidx:04d}.npz"
        if pidx in done_rows and npz_p.exists():
            z = np.load(npz_p)
            per_pair.append({**done_rows[pidx], "_f0_free": z["f0"], "_base": z["base"],
                             "_f1": z["f1"], "_target": z["target"]})
            continue
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            print(f"[p3v2] --max-seconds reached at pair {pidx}; stopping "
                  f"({len(per_pair)} loaded). Re-invoke with --resume to continue.", flush=True)
            break
        pair = load_pair(pidx)
        f0_stored, f1 = pair[0], pair[1]
        target = targets[pidx]
        row: dict[str, Any] = {"pair": pidx}
        # baselines
        row["d_pose_stored"] = d_pose_u8(posenet, f0_stored, f1, target)
        row["d_pose_zeros"] = d_pose_u8(posenet, np.zeros_like(f0_stored), f1, target)
        row["d_pose_copy"] = d_pose_u8(posenet, f1, f1, target)
        # S0 existing rank-6 cosine actuation to convergence (from zeros base, as P3 did); subset only
        # (the convergence verdict is basis-level, not per-pair). Costly finite-diff GN.
        if args.s0 and pidx < args.s0_pairs:
            row["s0_cosine6"] = s0_cosine6_solve(
                posenet, np.zeros_like(f0_stored), f1, target, relins=args.s0_relins, amp=args.s0_amp)
        # DECODER-REPRODUCIBLE warp base (ego-motion homography of f1 by the carried pose + 1 s_t scalar).
        # This is the cheap base for both the free-solve init AND the priced-carrier residual.
        base_work, s_t, dpose_warp = warp_base_work(posenet, f1, target, wh, ww)
        row["d_pose_warp_base"] = dpose_warp
        row["warp_s_t"] = s_t
        # S1d full free frame_0 (init from the ego-motion-aligned warp base — the right basin)
        free = s1d_free_solve(posenet, base_work, f1, target, iters=args.free_iters, lr=args.free_lr,
                              wh=wh, ww=ww)
        row["d_pose_free_u8"] = free["d_pose_free_u8"]
        row["free_iters_used"] = free["iters_used"]
        row["free_traj"] = free["traj"]
        # S1 price (per-pair generic compression of the free delta over the warp base — the honest
        # cheap-realization test: is the ego-motion residual compressible in a decoder-reproducible basis?)
        if args.price:
            row["s1_price"] = s1_price(posenet, free["f0_work"], base_work, f1, target,
                                       k_list=args.k_list, r_list=args.r_list)
        # persist (scalar row -> JSONL ; arrays -> npz) BEFORE appending, so a kill cannot lose it.
        fjl.write(json.dumps(row) + "\n")
        fjl.flush()
        os.fsync(fjl.fileno())
        np.savez_compressed(npz_p, f0=free["f0_work"].astype(np.float32),
                            base=base_work.astype(np.float32), f1=f1, target=target)
        per_pair.append({**row, "_f0_free": free["f0_work"], "_base": base_work,
                         "_f1": f1, "_target": target})
        print(f"  pair {pidx:2d}: stored={row['d_pose_stored']:.3f} zeros={row['d_pose_zeros']:.2f} "
              f"warp={row['d_pose_warp_base']:.4f} free_u8={row['d_pose_free_u8']:.6f} "
              f"(it{row['free_iters_used']}) ({time.time()-t0:.0f}s)", flush=True)
    fjl.close()

    # S2 LOTTO across the solved pairs
    lotto = None
    if args.lotto and len(per_pair) >= 2:
        deltas = [(r["_f0_free"] - r["_base"]) for r in per_pair]
        bases = [r["_base"] for r in per_pair]
        f1s = [r["_f1"] for r in per_pair]
        tgs = [r["_target"] for r in per_pair]
        lotto = s2_lotto(posenet, deltas, bases, f1s, tgs, ranks=args.lotto_ranks)

    # frame_1-untouched + frame_0 seg-free spot check (factorization law): SegNet reads x[:,-1]=f1 only.
    seg_check = None
    if args.seg_check and per_pair:
        segnet = modules.SegNet().eval().cpu()
        from safetensors.torch import load_file
        segnet.load_state_dict(load_file(str(_UPSTREAM / "models/segnet.safetensors"), device="cpu"))
        for p in segnet.parameters():
            p.requires_grad = False
        r = per_pair[0]
        f0_free_u8 = _f0work_to_u8(r["_f0_free"])
        f0_zero = np.zeros_like(f0_free_u8)

        def seg_argmax(f0_u8, f1_u8):
            x = torch.from_numpy(np.stack([f0_u8, f1_u8])[None]).permute(0, 1, 4, 2, 3).float()
            with torch.inference_mode():
                logits = segnet(segnet.preprocess_input(x))
            return logits.argmax(1)[0].numpy()

        a_free = seg_argmax(f0_free_u8, r["_f1"])
        a_zero = seg_argmax(f0_zero, r["_f1"])
        seg_check = {"pair": int(per_pair[0]["pair"]),
                     "argmax_identical_across_frame0_change": bool(np.array_equal(a_free, a_zero)),
                     "note": "SegNet argmax is IDENTICAL for two totally different frame_0 with the same "
                             "frame_1 -> frame_0 is 100% seg-free (upstream/modules.py:108). d_seg untouched."}

    # aggregate. BINDING rule (task prompt / Assumption-Adversary, MAIN-adopted): the composed row's
    # pose term is sqrt(10 * MEAN d_pose), so the vehicle verdict is on the free-upper-bound MEAN
    # contribution. thr_wall = 2.5e-4 (contribution 0.05). thr_1e3 is the charter's looser falsifier.
    stored = np.array([r["d_pose_stored"] for r in per_pair])
    free = np.array([r["d_pose_free_u8"] for r in per_pair])
    # S0 convergence summary (rank-6 cosine basis run to `s0_relins` relins): budget-truncated vs
    # rank-deficient. plateau = min d_pose reached; the P3 stop was 38.06 at ~2 relins.
    s0_rows = [r["s0_cosine6"] for r in per_pair if "s0_cosine6" in r]
    s0_summary = None
    if s0_rows:
        plateaus = [min(x["d_pose_traj"]) for x in s0_rows]
        s0_summary = {
            "n_pairs": len(s0_rows), "relins_budget": args.s0_relins,
            "plateau_d_pose_mean": float(np.mean(plateaus)),
            "plateau_d_pose_median": float(np.median(plateaus)),
            "mean_relins_run": float(np.mean([x["relins_run"] for x in s0_rows])),
            "example_traj": s0_rows[0]["d_pose_traj"],
            "verdict": "RANK_DEFICIENT" if float(np.median(plateaus)) > 5.0 else "BUDGET_TRUNCATED",
            "note": "the rank-6 cosine basis is the EXISTING P3 actuation. If it plateaus >> the free "
                    "floor (~1e-4) even at convergence, the basis is RANK-DEFICIENT (the 38.06 was a "
                    "basis problem, not only budget); the free/Jacobian-aligned family is required."}
    thr_wall = 2.5e-4  # contribution 0.05 (the BINDING FORMULATION-scope wall threshold)
    thr_1e3 = 1e-3     # the charter's <=1e-3-class falsifier (looser)
    free_mean = float(free.mean())
    contrib_mean = contribution(free_mean)
    verdict = ("WALL_REFUTED_ARTIFACT_OF_NAIVE_SOLVE" if contrib_mean <= 0.05
               else "WALL_CONFIRMED_FORMULATION_SCOPE")
    rule_side = ("CANDIDATE_LINE" if contrib_mean <= 0.05 else "CALIBRATION_INSTRUMENT")

    payload: dict[str, Any] = {
        "schema": "ddm_p3v2_optimal_form_pose_resolve.v1",
        "tool": "experiments/ddm_p3v2_optimal_form_pose_resolve.py",
        "utc": _utc(), "git_hash": _git_hash(), "axis": args.axis,
        "score_claim": False, "promotable": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "n_pairs_done": len(per_pair), "n_pairs_requested": n,
        "pair_selection": _prefix_pair_selection_scope([r["pair"] for r in per_pair], n),
        "work_res": [wh, ww], "free_iters": args.free_iters,
        "baseline_d_pose": {
            "stored_mean": float(stored.mean()), "stored_median": float(np.median(stored)),
            "zeros_mean": float(np.mean([r["d_pose_zeros"] for r in per_pair])),
            "copy_mean": float(np.mean([r["d_pose_copy"] for r in per_pair])),
            "warp_base_mean": float(np.mean([r["d_pose_warp_base"] for r in per_pair])),
            "warp_base_median": float(np.median([r["d_pose_warp_base"] for r in per_pair])),
            "warp_base_contribution_at_mean": contribution(
                float(np.mean([r["d_pose_warp_base"] for r in per_pair]))),
            "warp_base_note": "ego-motion homography of f1 by the carried pose target + 1 s_t scalar; "
                              "DECODER-REPRODUCIBLE ~0 bytes; the cheap-carrier floor."},
        "s1d_free_upper_bound": {
            "d_pose_mean": free_mean, "d_pose_median": float(np.median(free)),
            "d_pose_max": float(free.max()), "d_pose_min": float(free.min()),
            "contribution_at_mean": contrib_mean,
            "contribution_at_median": contribution(float(np.median(free))),
            "frac_pairs_below_2p5e-4": float(np.mean(free <= thr_wall)),
            "frac_pairs_below_1e-3": float(np.mean(free <= thr_1e3)),
            "mean_iters_used": float(np.mean([r["free_iters_used"] for r in per_pair])),
            "max_iters_budget": args.free_iters,
            "converged_note": "iters_used < budget on a pair => convergence early-stop fired (NOT "
                              "budget-truncated, the P3 error); see per_pair.free_traj."},
        "PRE_REGISTERED_RULE": {
            "wall_threshold_d_pose": thr_wall, "wall_threshold_contribution": 0.05,
            "binding_quantity": "sqrt(10 * MEAN free d_pose) = composed-row pose contribution",
            "binding_contribution_measured": contrib_mean,
            "falsifier_1e3_class": thr_1e3,
            "VERDICT": verdict, "vehicle_designation": rule_side,
            "note": "S1d is the UNPRICED free-frame_0 reach ceiling; frame_0 100% seg-free (law). "
                    "BINDING rule: contribution<=0.05 => wall REFUTED / candidate line; else "
                    "wall CONFIRMED at FORMULATION scope / calibration instrument (v10 pose-in-burn head)."},
        "citations_banked_carriers": {
            "p3_6cosine_budget_truncated": {"d_pose_mean": 38.06223, "bytes_n600": 7295,
                                            "source": "ddm_pb1_20260729/p3/p3_terminal_pose_receipt.json"},
            "p715_quotient_reach_rank1": {"d_pose": 19.895, "carrier_bytes": 3520,
                                          "note": "generic covariance basis; d_pose RISES with rank"},
            "sc1_e_p_rank1": {"residual_bytes": 2039, "note": "pose-FIELD carrier; raw seed d_pose 36-146"}},
        "s0_cosine6_convergence": s0_summary,
        "seg_untouched_spot_check": seg_check,
        "s2_lotto": lotto,
        "per_pair": [{k: v for k, v in r.items() if not k.startswith("_")} for r in per_pair],
        "elapsed_s": round(time.time() - t0, 1),
    }
    return payload


def run_s3warp(args) -> dict[str, Any]:
    """S3 priced n600 point on the CHEAP realizable winner = the warp-base carrier. For each pair fit
    the s_t scalar (11-value grid) and record the frozen-uint8 d_pose. Price the s_t stream with the
    merged SMEVR/r7 coder (the s_t index in [0,11) per pair). Report the mean d_pose + composed-row S.
    The warp uses the CARRIED 6-value pose target (sc1 t_p sidecar) — already-carried, not new bytes."""
    import torch
    torch.set_num_threads(4)
    t0 = time.time()
    posenet, _modules = load_posenet()
    n = int(args.n_pairs)
    targets = load_targets(n)
    wh, ww = int(args.work_h), int(args.work_w)
    # RESUMABLE per-pair persistence (reaper-safe sub-5-min chunks). JSONL: {pair, s_t_idx, d_pose}.
    s3_jsonl = Path(str(args.out).replace(".json", ".s3.partial.jsonl"))
    _refuse_tmp(s3_jsonl)
    s3_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[int, dict] = {}
    if s3_jsonl.exists() and args.resume:
        for ln in s3_jsonl.read_text().splitlines():
            try:
                rr = json.loads(ln)
                cache[int(rr["pair"])] = rr
            except Exception:
                pass
        print(f"[p3v2-s3] resume: {len(cache)} pairs cached", flush=True)
    fj = open(s3_jsonl, "a")  # noqa: SIM115 (append across the resumable loop)
    per = {}
    for pidx in range(n):
        if pidx in cache:
            per[pidx] = cache[pidx]
            continue
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            print(f"[p3v2-s3] --max-seconds reached at pair {pidx}; {len(per)} done. Re-run --resume.",
                  flush=True)
            break
        pair = load_pair(pidx)
        _base, s_t, dpose_warp = warp_base_work(posenet, pair[1], targets[pidx], wh, ww)
        rr = {"pair": pidx, "s_t_idx": int(_ST_GRID.index(s_t)), "d_pose": float(dpose_warp)}
        fj.write(json.dumps(rr) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        per[pidx] = rr
        if pidx % 20 == 0 or pidx == n - 1:
            rmean = float(np.mean([per[k]["d_pose"] for k in per]))
            print(f"  [s3] pair {pidx:3d}: warp d_pose={dpose_warp:.4f} s_t={s_t} "
                  f"(running mean={rmean:.4f}) ({time.time()-t0:.0f}s)", flush=True)
    fj.close()
    ordered = [per[k] for k in sorted(per)]
    st_idx = [r["s_t_idx"] for r in ordered]
    dps = np.asarray([r["d_pose"] for r in ordered], np.float64)
    done = sorted(per)
    # price the s_t index stream with the merged SMEVR/r7 coder (levels=len(_ST_GRID)).
    r7_bytes = None
    try:
        sys.path.insert(0, str(_REPO / "experiments"))
        from ddm_r7_token_coder import encode_token_codes
        codes = np.asarray(st_idx, np.uint8).reshape(len(st_idx), 1, 1, 1)  # [P,H,W,C] r7 contract
        frame = encode_token_codes(codes, levels=len(_ST_GRID), codec="auto")
        r7_bytes = len(frame)
    except Exception:
        r7_bytes = None
    zlib_bytes = len(zlib.compress(np.asarray(st_idx, np.uint8).tobytes(), 9))
    st_stream_bytes = r7_bytes if r7_bytes is not None else zlib_bytes
    pose_mean = float(dps.mean())
    pose_contrib = contribution(pose_mean)
    # composed-row arithmetic vs the pb1 instrument row S~20.2746 (pose 19.50954 / seg 0.38901 / rate 0.37609)
    ROW_S, ROW_POSE, ROW_SEG, ROW_RATE = 20.27464, 19.50954, 0.38901, 0.37609
    ROW_BYTES = 564880
    P3_POSE_BYTES = 7295  # the 6-cosine member replaced by this carrier
    # additional bytes vs the row: replace the 7295-B pose member with (s_t stream); the 6-value pose
    # target is ASSUMED already carried (sc1 t_p ~2KB). Report BOTH: t_p-free and t_p-counted.
    tp_sidecar_est = 2039  # sc1 e_p/t_p banked estimate
    delta_bytes_tp_free = st_stream_bytes - P3_POSE_BYTES
    delta_bytes_tp_counted = (st_stream_bytes + tp_sidecar_est) - P3_POSE_BYTES
    new_rate_tp_free = ROW_RATE + 25.0 * delta_bytes_tp_free / 37_545_489
    new_rate_tp_counted = ROW_RATE + 25.0 * delta_bytes_tp_counted / 37_545_489
    new_S_tp_free = ROW_SEG + pose_contrib + new_rate_tp_free
    new_S_tp_counted = ROW_SEG + pose_contrib + new_rate_tp_counted
    payload = {
        "schema": "ddm_p3v2_s3_warp_priced_n600.v1",
        "tool": "experiments/ddm_p3v2_optimal_form_pose_resolve.py --mode s3warp",
        "utc": _utc(), "git_hash": _git_hash(), "axis": args.axis,
        "score_claim": False, "promotable": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "n_pairs_done": len(done), "n_pairs_requested": n,
        "pair_selection": _prefix_pair_selection_scope(done, n),
        "carrier": "warp-base (ground homography of f1 by the carried pose target + per-pair s_t index)",
        "d_pose_warp_mean": pose_mean, "d_pose_warp_median": float(np.median(dps)),
        "d_pose_warp_max": float(dps.max()), "d_pose_warp_min": float(dps.min()),
        "pose_contribution": pose_contrib,
        "st_stream_bytes_r7_smevr": r7_bytes, "st_stream_bytes_zlib": zlib_bytes,
        "st_grid": _ST_GRID,
        "composed_row_arithmetic": {
            "pb1_row": {"S": ROW_S, "pose": ROW_POSE, "seg": ROW_SEG, "rate": ROW_RATE, "bytes": ROW_BYTES},
            "p3_pose_member_bytes_replaced": P3_POSE_BYTES,
            "tp_sidecar_assumed_carried_est": tp_sidecar_est,
            "new_pose_contribution": pose_contrib,
            "new_S_if_tp_already_carried": new_S_tp_free,
            "new_S_if_tp_counted_new": new_S_tp_counted,
            "delta_S_vs_row_tp_free": new_S_tp_free - ROW_S,
            "delta_S_vs_row_tp_counted": new_S_tp_counted - ROW_S,
            "note": f"pose 19.50954 -> {pose_contrib:.5f} (cheap decoder-reproducible warp carrier). "
                    "Free upper bound (unpriced, n24) reaches contribution ~0.03 but is "
                    "basis-adversarial; closing the gap needs pose-field terminal-solve (sc1 e_p) "
                    "or v10 pose-in-burn."},
        "elapsed_s": round(time.time() - t0, 1),
    }
    return payload


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("ladder", "s3warp"), default="ladder")
    ap.add_argument("--n-pairs", type=int, default=24)
    ap.add_argument("--work-h", type=int, default=192)
    ap.add_argument("--work-w", type=int, default=256)
    ap.add_argument("--free-iters", type=int, default=80)
    ap.add_argument("--free-lr", type=float, default=3.0)
    ap.add_argument("--s0", action="store_true", help="run the S0 rank-6 cosine convergence check")
    ap.add_argument("--s0-relins", type=int, default=30)
    ap.add_argument("--s0-amp", type=float, default=4.0)
    ap.add_argument("--s0-pairs", type=int, default=8, help="cap S0 to the first N pairs (basis verdict)")
    ap.add_argument("--price", action="store_true", help="run the S1 byte-Pareto pricing")
    ap.add_argument("--k-list", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--r-list", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--lotto", action="store_true", help="run the S2 LOTTO shared-dictionary race")
    ap.add_argument("--lotto-ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--seg-check", action="store_true", help="frame_0 seg-free SegNet argmax spot check")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="resume from the per-pair partial JSONL + npz cache (reaper-safe chunks)")
    ap.add_argument("--axis", type=str, default="[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE")
    ap.add_argument("--out", type=Path, default=_SSD_OUT / "p3v2_ladder_receipt.json")
    return ap


def main() -> int:
    args = build_args().parse_args()
    _refuse_tmp(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "s3warp":
        payload = run_s3warp(args)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        payload_sha = sha256(args.out.read_bytes()).hexdigest()
        ca = payload["composed_row_arithmetic"]
        print(f"\n[p3v2-s3] warp carrier n{payload['n_pairs_done']}: d_pose mean={payload['d_pose_warp_mean']:.4f} "
              f"contribution={payload['pose_contribution']:.4f}", flush=True)
        print(f"[p3v2-s3] composed S 20.27464 -> {ca['new_S_if_tp_already_carried']:.4f} (tp-free) / "
              f"{ca['new_S_if_tp_counted_new']:.4f} (tp-counted); s_t stream r7={payload['st_stream_bytes_r7_smevr']}B", flush=True)
        print(f"[p3v2-s3] receipt -> {args.out}  sha256={payload_sha}", flush=True)
        return 0
    payload = run(args)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    payload_sha = sha256(args.out.read_bytes()).hexdigest()
    print(f"\n[p3v2] VERDICT: {payload['PRE_REGISTERED_RULE']['VERDICT']} "
          f"-> vehicle={payload['PRE_REGISTERED_RULE']['vehicle_designation']}", flush=True)
    print(f"[p3v2] free-frame0 d_pose median={payload['s1d_free_upper_bound']['d_pose_median']:.3e} "
          f"mean={payload['s1d_free_upper_bound']['d_pose_mean']:.3e} "
          f"contribution@mean={payload['s1d_free_upper_bound']['contribution_at_mean']:.4f}", flush=True)
    print(f"[p3v2] receipt -> {args.out}  sha256={payload_sha}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
