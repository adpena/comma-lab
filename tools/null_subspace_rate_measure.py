#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""null_subspace_rate_measure — task #519 STAGE-1: quantify the two score-invariant null
subspaces in a frozen level-set witness checkpoint and the rate they cost. $0, CPU, read-only
on checkpoints.

THE TWO NULL SUBSPACES (derived from the ACTUAL witness head form — see below):

1. GAUGE (class-constant direction of the out_sdf head).
   The witness render is  RGB = sigmoid( softmax(phi/T) @ palette + tex ) * 255  with
   phi = out_sdf(h) a plain K=5 linear head (LVLS1 shared-head form). softmax is EXACTLY
   invariant to phi -> phi + c(p)*1 for any per-pixel constant c(p), so the class-MEAN
   component of out_sdf ( W = 1 x w_bar + W_perp, b = b_bar*1 + b_perp ) is a true gauge
   orbit of the WHOLE render (stronger than argmax-invariance; the self-orient fixed point
   consumes argmax(phi) and is therefore also invariant). Projecting it out is exactly
   render-invariant in fp32 algebra; ONLY the int8 per-tensor symmetric quantization
   (scale = max|W|/127) can break it — which leg 4 measures through the REAL decode.

   A SECOND exact gauge in the same head family: shifting every palette row by a constant
   per-channel vector v adds v to soft@palette (softmax rows sum to 1) and is exactly
   cancelled by out_tex.bias -= v is wrong sign — palette' = palette - 1 x v,
   bias' = bias + v leaves base + tex INVARIANT. Canonicalizing v = per-channel palette
   mean moves that energy into 3 bias scalars.

2. BLIND (ker(A) of the exact shared scorer resize).
   Both frozen scorers resize the camera frame (874,1164) -> (384,512) with the SAME
   bilinear align_corners=False antialias=False kernel A (see
   src/tac/through_r/blind_coordinate.py). A is separable: A(X) = Dr @ X @ Dc^T. The
   orthogonal projector onto ker(A)^perp = range(A^T) is P_r (x) P_c with
   P_r = Q_r Q_r^T (Q_r from QR of Dr^T). We measure (a) the energy fraction of the
   DECODED camera frames lying in ker(A), and (b) the fraction of OUTPUT-LAYER
   (out_tex / out_sdf) weight-direction energy whose Jacobian-pushforward image
   (linearized past the uint8 round: d rgb -> bicubic-up U -> A) lies in ker(A).
   HONEST STRUCTURAL NOTE: ker(A) lives in image space; it does NOT pull back to a
   weight-space linear subspace through the nonlinear render, so there is NO exact
   weight-space projection for it — its direct rate lever on a pure-generator witness is
   0 bytes (the blob stores no camera pixels; cf. blind_coordinate.py SCOPE). Leg 2 is a
   capacity/waste observation, not a rate projection.

AUTHORITY: every scorer number here is the frozen CPU-torch scorer on the byte-closed
decode — [macOS-CPU advisory] NON-PROMOTABLE, no score claims. 1-thread per the standard.

Usage:
  OMP_NUM_THREADS=1 .venv/bin/python tools/null_subspace_rate_measure.py \
      --ckpt-dir experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z \
      --npz-name levelset_witness_ema_BEST.npz \
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
      --pairs 32 --jacobian-pairs 2 --out reports/null_subspace_rate_measure_519.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream", _REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import levelset_byte_close_and_eval as bc  # noqa: E402  (REUSE the real byte-close/decode path)

AXIS = "[macOS-CPU advisory] NON-PROMOTABLE"


# ----------------------------------------------------------------------------------
# leg 1 — gauge component magnitudes (pure fp32 tensor algebra on the checkpoint)
# ----------------------------------------------------------------------------------
def gauge_stats(params: dict[str, np.ndarray]) -> dict[str, Any]:
    W = np.asarray(params["out_sdf.weight"], np.float64)   # (K, H)
    b = np.asarray(params["out_sdf.bias"], np.float64)     # (K,)
    K = W.shape[0]
    w_bar = W.mean(axis=0)                                 # (H,)
    b_bar = float(b.mean())
    gauge_w_norm = float(np.sqrt(K) * np.linalg.norm(w_bar))   # ||1 (x) w_bar||_F
    gauge_b_norm = float(np.sqrt(K) * abs(b_bar))
    Wn, bn = float(np.linalg.norm(W)), float(np.linalg.norm(b))
    pal = np.asarray(params["palette"], np.float64)        # (K, 3)
    v = pal.mean(axis=0)                                   # (3,) channel mean
    pal_gauge_norm = float(np.sqrt(K) * np.linalg.norm(v))
    return {
        "head_form": "phi = out_sdf(h) linear K=5 -> softmax(phi/T) -> @palette (+tex) -> sigmoid",
        "gauge_direction": "class-constant (all-ones across the 5 rows) of out_sdf.weight/bias",
        "out_sdf.weight": {
            "fro_norm": Wn,
            "gauge_component_norm": gauge_w_norm,
            "gauge_fraction_of_norm": gauge_w_norm / Wn,
            "gauge_fraction_of_energy": (gauge_w_norm / Wn) ** 2,
        },
        "out_sdf.bias": {
            "norm": bn,
            "gauge_component_norm": gauge_b_norm,
            "gauge_fraction_of_norm": (gauge_b_norm / bn) if bn > 0 else 0.0,
            "bias_values": [float(x) for x in b],
        },
        "palette_gauge": {
            "fro_norm": float(np.linalg.norm(pal)),
            "channel_mean": [float(x) for x in v],
            "gauge_component_norm": pal_gauge_norm,
            "gauge_fraction_of_norm": pal_gauge_norm / float(np.linalg.norm(pal)),
            "cancellation_partner": "out_tex.bias += channel_mean (exact, softmax rows sum to 1)",
        },
    }


def project_gauge(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {k: np.array(v, np.float32, copy=True) for k, v in params.items()}
    W = out["out_sdf.weight"]
    out["out_sdf.weight"] = (W - W.mean(axis=0, keepdims=True)).astype(np.float32)
    b = out["out_sdf.bias"]
    out["out_sdf.bias"] = (b - b.mean()).astype(np.float32)
    return out


def project_palette_gauge(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {k: np.array(v, np.float32, copy=True) for k, v in params.items()}
    pal = out["palette"]
    v = pal.mean(axis=0, keepdims=True)                    # (1,3)
    out["palette"] = (pal - v).astype(np.float32)
    out["out_tex.bias"] = (out["out_tex.bias"] + v[0]).astype(np.float32)
    return out


# ----------------------------------------------------------------------------------
# leg 2 — exact shared-resize kernel machinery (separable bilinear, torch-exact)
# ----------------------------------------------------------------------------------
def _resize_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Dr (384,874), Dc (512,1164): the EXACT torch bilinear align_corners=False
    antialias=False downsample as separable row/col matrices (impulse-probed, fp64)."""
    import torch

    ch, cw = bc.CAMERA_H, bc.CAMERA_W
    sh, sw = 384, 512
    with torch.inference_mode():
        eye_r = torch.eye(ch, dtype=torch.float64)[None, None]           # (1,1,874,874)
        Dr = torch.nn.functional.interpolate(eye_r, size=(sh, ch), mode="bilinear",
                                             align_corners=False)[0, 0].numpy()  # (384,874)
        eye_c = torch.eye(cw, dtype=torch.float64)[None, None]           # (1,1,1164,1164)
        DcT = torch.nn.functional.interpolate(eye_c, size=(cw, sw), mode="bilinear",
                                              align_corners=False)[0, 0].numpy()  # (1164,512)
    return Dr, DcT.T


def _ortho_range(D: np.ndarray) -> np.ndarray:
    """Orthonormal basis Q of range(D^T) (columns), so P = Q Q^T projects onto ker(D)^perp."""
    Q, R = np.linalg.qr(D.T)                                # D^T (n_in, n_out)
    if float(np.abs(np.diag(R)).min()) <= 0:
        raise RuntimeError("resize matrix not full row rank — projector invalid")
    return np.ascontiguousarray(Q)                          # (n_in, n_out)


def _blind_fraction(X: np.ndarray, Qr: np.ndarray, Qc: np.ndarray) -> float:
    """1 - ||Q_r^T X Q_c||^2 / ||X||^2 for a single-channel camera field X (874,1164). f64."""
    X = np.asarray(X, np.float64)
    tot = float(np.sum(X * X))
    if tot == 0.0:
        return 0.0
    S = (Qr.T @ X) @ Qc
    return 1.0 - float(np.sum(S * S)) / tot


def _projector_self_test(Dr: np.ndarray, Dc: np.ndarray, Qr: np.ndarray, Qc: np.ndarray) -> dict[str, float]:
    """Verify P = Qr Qr^T (x) Qc Qc^T is the exact projector onto range(A^T):
    (1) A(X - P X) == 0 (ker residual invisible); (2) a seen-space element has blind fraction 0;
    (3) an image supported ONLY on the exactly-blind rows/cols has blind fraction 1."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((Dr.shape[1], Dc.shape[1]))
    PX = Qr @ (Qr.T @ X @ Qc) @ Qc.T
    r1 = float(np.abs(Dr @ (X - PX) @ Dc.T).max())          # should be ~0
    Z = rng.standard_normal((Dr.shape[0], Dc.shape[0]))
    Y = Dr.T @ Z @ Dc                                        # in range(A^T)
    r2 = float(_blind_fraction(Y, Qr, Qc))                   # should be ~0
    blind_rows = np.abs(Dr).sum(axis=0) == 0.0               # camera rows with exact zero weight
    Xb = np.zeros_like(X)
    Xb[blind_rows, :] = rng.standard_normal((int(blind_rows.sum()), X.shape[1]))
    r3 = float(_blind_fraction(Xb, Qr, Qc))                  # should be 1
    return {"max_A_of_ker_residual": r1, "seen_space_blind_fraction": r2,
            "blind_row_image_blind_fraction": r3,
            "n_exact_blind_rows": int(blind_rows.sum()),
            "n_exact_blind_cols": int((np.abs(Dc).sum(axis=0) == 0.0).sum())}


def blind_output_stats(frames: list[np.ndarray], Qr: np.ndarray, Qc: np.ndarray) -> dict[str, Any]:
    """Energy fraction of decoded uint8 camera frames in ker(A), raw + mean-removed."""
    raw, ac = [], []
    for f in frames:
        X = np.asarray(f, np.float64)
        for c in range(3):
            xc = X[..., c]
            raw.append(_blind_fraction(xc, Qr, Qc))
            ac.append(_blind_fraction(xc - xc.mean(), Qr, Qc))
    return {
        "n_channel_fields": len(raw),
        "ker_dim_fraction_of_camera_space": 1.0 - (384 * 512) / float(bc.CAMERA_H * bc.CAMERA_W),
        "raw_energy_fraction_in_kerA": {"mean": float(np.mean(raw)), "min": float(np.min(raw)),
                                        "max": float(np.max(raw))},
        "ac_energy_fraction_in_kerA": {"mean": float(np.mean(ac)), "min": float(np.min(ac)),
                                       "max": float(np.max(ac))},
    }


# ----------------------------------------------------------------------------------
# leg 2b — output-layer weight-direction blindness through the linearized decode Jacobian
# ----------------------------------------------------------------------------------
def _feats_for_pair(m: dict[str, Any], params: dict[str, np.ndarray], code: np.ndarray,
                    pi: int) -> np.ndarray:
    """Mirror of numpy_oracle_reference_frames' self-orient fixed point (legacy Fourier arm)."""
    rh, rw = int(m["render_h"]), int(m["render_w"])
    coords = bc._canon_coords_grid(rh, rw)
    B = bc._canon_curvelet_B(m["bank_n_scales"], m["bank_n_orient0"], m["bank_f0"],
                             m["bank_base"], m["bank_n_iso"], m["max_bank_freq"])
    curv = bc._canon_curvelet_feats(coords, B)
    if not bool(m["self_orient"]):
        return curv
    from tac.boundary_math.lever_b_levelset_generator import levelset_rgb_forward_numpy
    fwd_kw = {
        "n_hidden": int(m["n_hidden"]), "hidden_dim": int(m["hidden_dim"]),
        "n_classes": int(m["n_classes"]), "activation": str(m["activation"]),
        "softmax_temp": float(m["softmax_temp"]), "wire_w0": float(m["wire_w0"]),
        "wire_s0": float(m["wire_s0"]), "hosc_beta": float(m["hosc_beta"]),
        "hosc_omega": float(m["hosc_omega"]), "chroma": bool(m["chroma"]),
    }
    ndf = int(m["n_dir_freqs"])
    dirf = np.zeros((curv.shape[0], 4 * ndf), np.float32)
    prev = None
    for _ in range(int(m["so_iters"])):
        feats = np.concatenate([curv, dirf], axis=-1)
        _rgb, phi = levelset_rgb_forward_numpy(params, feats, code[2 * pi + 1], **fwd_kw)
        am = phi.argmax(-1).reshape(rh, rw).astype(np.int64)
        if prev is not None and np.array_equal(am, prev):
            break
        dirf = bc._canon_dir_feats(coords, am, ndf, float(m["so_freq_along"]),
                                   float(m["so_freq_across"]), float(m["so_tau"]))
        prev = am
    return np.concatenate([curv, dirf], axis=-1)


def _forward_intermediates(m: dict[str, Any], params: dict[str, np.ndarray],
                           feats: np.ndarray, code_row: np.ndarray):
    """fp64 trunk forward returning (h, soft, base, tex) — mirrors levelset_rgb_forward_numpy."""
    from tac.boundary_math.lever_b_levelset_generator import _act
    p = {k: np.asarray(v, np.float64) for k, v in params.items()}
    akw = {"w0": float(m["wire_w0"]), "s0": float(m["wire_s0"]),
           "beta": float(m["hosc_beta"]), "omega": float(m["hosc_omega"])}
    act = str(m["activation"])
    nH, hd = int(m["n_hidden"]), int(m["hidden_dim"])
    cr = np.asarray(code_row, np.float64)
    h = _act(np.asarray(feats, np.float64) @ p["in_proj.weight"].T + p["in_proj.bias"], act, **akw)
    film = (cr @ p["film.weight"].T + p["film.bias"]).reshape(nH, 2, hd)
    for li in range(nH):
        pre = (h @ p[f"hidden.{li}.weight"].T + p[f"hidden.{li}.bias"]) * (1.0 + film[li, 0]) + film[li, 1]
        h = _act(pre, act, **akw)
    phi = h @ p["out_sdf.weight"].T + p["out_sdf.bias"]
    tex = h @ p["out_tex.weight"].T + p["out_tex.bias"]
    z = phi / float(m["softmax_temp"])
    z = z - z.max(axis=-1, keepdims=True)
    soft = np.exp(z)
    soft = soft / soft.sum(axis=-1, keepdims=True)
    base = soft @ p["palette"]
    return h, soft, base, tex


def _bicubic_up_batch(fields: np.ndarray) -> np.ndarray:
    """(N, C, rh, rw) f32 -> (N, C, 874, 1164) via bicubic align_corners=False (linearized _R)."""
    import torch

    with torch.inference_mode():
        t = torch.from_numpy(np.ascontiguousarray(fields))
        up = torch.nn.functional.interpolate(t, size=(bc.CAMERA_H, bc.CAMERA_W),
                                             mode="bicubic", align_corners=False)
    return up.numpy()


def jacobian_blindness(m: dict[str, Any], params: dict[str, np.ndarray], code: np.ndarray,
                       pair_indices: list[int], Qr: np.ndarray, Qc: np.ndarray,
                       chunk: int = 24) -> dict[str, Any]:
    """Per output-layer weight direction: fraction of the pushforward image energy in ker(A),
    aggregated weight-energy-weighted. Linearized past the uint8 round (advisory structural)."""
    rh, rw = int(m["render_h"]), int(m["render_w"])
    W_tex = np.asarray(params["out_tex.weight"], np.float64)   # (3, H)
    W_sdf = np.asarray(params["out_sdf.weight"], np.float64)   # (K, H)
    pal = np.asarray(params["palette"], np.float64)            # (K, 3)
    T = float(m["softmax_temp"])
    K, H = W_sdf.shape
    per_pair: list[dict[str, float]] = []
    for pi in pair_indices:
        feats = _feats_for_pair(m, params, code, pi)
        h, soft, base, tex = _forward_intermediates(m, params, feats, code[2 * pi + 1])
        sig = 1.0 / (1.0 + np.exp(-(base + tex)))              # (P,3)
        sprime = (255.0 * sig * (1.0 - sig)).astype(np.float32)  # (P,3)
        h32 = h.astype(np.float32)
        # --- out_tex directions (r, j): field = sprime[:, r] * h[:, j], single channel r ---
        tex_num = tex_den = 0.0
        tex_fracs = np.zeros((3, H), np.float64)
        for r in range(3):
            for j0 in range(0, H, chunk):
                j1 = min(j0 + chunk, H)
                flds = (sprime[:, r][None, :] * h32[:, j0:j1].T).reshape(j1 - j0, 1, rh, rw)
                up = _bicubic_up_batch(flds)[:, 0]              # (n, 874, 1164)
                for i in range(up.shape[0]):
                    fr = _blind_fraction(up[i], Qr, Qc)
                    tex_fracs[r, j0 + i] = fr
                    w2 = float(W_tex[r, j0 + i]) ** 2
                    tex_num += w2 * fr
                    tex_den += w2
        # --- out_sdf directions (k, j): 3-channel field sprime[:,r]*D_k[:,r]*h[:,j] ---
        sdf_num = sdf_den = 0.0
        sdf_fracs = np.zeros((K, H), np.float64)
        for k in range(K):
            Dk = ((1.0 / T) * soft[:, k:k + 1] * (pal[k][None, :] - base)).astype(np.float32)  # (P,3)
            amp = sprime * Dk                                   # (P,3)
            for j0 in range(0, H, chunk):
                j1 = min(j0 + chunk, H)
                flds = (amp.T[None, :, :] * h32[:, j0:j1].T[:, None, :]).reshape(
                    j1 - j0, 3, rh, rw)                         # (n,3,rh,rw)
                up = _bicubic_up_batch(flds)                    # (n,3,874,1164)
                for i in range(up.shape[0]):
                    u64 = up[i].astype(np.float64)
                    tot = float(np.sum(u64 * u64))
                    if tot == 0.0:
                        fr = 0.0
                    else:
                        seen = 0.0
                        for c in range(3):
                            S = (Qr.T @ u64[c]) @ Qc
                            seen += float(np.sum(S * S))
                        fr = 1.0 - seen / tot
                    sdf_fracs[k, j0 + i] = fr
                    w2 = float(W_sdf[k, j0 + i]) ** 2
                    sdf_num += w2 * fr
                    sdf_den += w2
        per_pair.append({
            "pair": pi,
            "out_tex_weight_energy_weighted_blind_fraction": tex_num / max(tex_den, 1e-30),
            "out_tex_unweighted_mean_blind_fraction": float(tex_fracs.mean()),
            "out_sdf_weight_energy_weighted_blind_fraction": sdf_num / max(sdf_den, 1e-30),
            "out_sdf_unweighted_mean_blind_fraction": float(sdf_fracs.mean()),
        })
    agg = {k: float(np.mean([pp[k] for pp in per_pair]))
           for k in per_pair[0] if k != "pair"}
    return {"pairs": pair_indices, "per_pair": per_pair, "mean": agg,
            "note": "linearized past uint8 round; Jacobian = d(render)/d(output-layer weight) "
                    "pushed through bicubic-up then projected against ker(A) — advisory structural"}


# ----------------------------------------------------------------------------------
# legs 3+4 — real byte-close packing + decode + frozen CPU-torch scorer
# ----------------------------------------------------------------------------------
def build_blob_for(params: dict[str, np.ndarray], cfg: dict[str, Any],
                   so: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    blob, breakdown = bc.build_levelset_blob(params, cfg, so, None)
    return blob, breakdown


def decode_frames(blob: bytes, n_pairs: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    m, dq_params, dq_code, lane_pairs, pcar = bc._dequant_blob(blob)
    return bc.numpy_oracle_reference_frames(dq_params, dq_code, m, n_pairs,
                                            lane_pairs=lane_pairs, pose_carrier=pcar)


def load_gt_subset(gt_cache: Path, n_pairs: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    z = np.load(gt_cache, allow_pickle=False)
    lstars = np.asarray(z["lstars"][:n_pairs]).astype(np.int64)
    gt_poses = np.asarray(z["gt_poses"][:n_pairs]).astype(np.float64)
    return [lstars[i] for i in range(n_pairs)], [gt_poses[i] for i in range(n_pairs)]


def load_scorers():
    import torch

    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # upstream

    from tac.boundary_math.seg_core import load_real_segnet
    seg_cpu = load_real_segnet("cpu")
    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet_cpu = dn.posenet
    for p in posenet_cpu.parameters():
        p.requires_grad = False
    return seg_cpu, posenet_cpu


def score_frames(frames: list[np.ndarray], lstars, gt_poses, seg_cpu, posenet_cpu,
                 batch: int = 8) -> dict[str, Any]:
    import train_witness_realized_through_R_mlx as twr
    P = len(frames) // 2
    f0s = [frames[2 * i] for i in range(P)]
    f1s = [frames[2 * i + 1] for i in range(P)]
    d_segs: list[float] = []
    d_poses: list[float] = []
    for i0 in range(0, P, batch):
        i1 = min(i0 + batch, P)
        d_segs += twr.cpu_verdict_d_seg_batch(seg_cpu, f1s[i0:i1], lstars[i0:i1])
        d_poses += twr.cpu_verdict_d_pose_batch(posenet_cpu, f0s[i0:i1], f1s[i0:i1],
                                                gt_poses[i0:i1])
    return {"pairs": P, "d_seg": float(np.mean(d_segs)), "d_pose": float(np.mean(d_poses)),
            "d_seg_per_pair_max": float(np.max(d_segs))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--npz-name", default=None)
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--pairs", type=int, default=32,
                    help="pairs decoded+scored per variant (first N; n600 = full)")
    ap.add_argument("--jacobian-pairs", type=int, default=2)
    ap.add_argument("--blind-frame-pairs", type=int, default=8)
    ap.add_argument("--skip-jacobian", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--so-iters", type=int, default=4)
    ap.add_argument("--so-tau", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    t0 = time.time()
    ckpt_dir = Path(args.ckpt_dir)
    params, cfg = bc._load_levelset_ckpt(ckpt_dir, args.npz_name)
    # self-orient overrides: prefer the PERSISTED trainer values from the npz
    z = np.load(ckpt_dir / (args.npz_name or "levelset_witness_ema_BEST.npz"), allow_pickle=False)
    fa = float(z["__cfg_freq_across"]) if "__cfg_freq_across" in z.files else 32.0
    fl = float(z["__cfg_freq_along"]) if "__cfg_freq_along" in z.files else 4.0
    so = bc.detect_self_orient(cfg, {"freq_across": fa, "freq_along": fl,
                                     "tau": args.so_tau, "iters": args.so_iters})
    report: dict[str, Any] = {
        "task": "#519 STAGE-1 null-subspace rate measurement",
        "axis": AXIS,
        "ckpt": str(ckpt_dir / (args.npz_name or "")),
        "epoch": int(z["__epoch"]) if "__epoch" in z.files else None,
        "self_orient_overrides": {"freq_across": fa, "freq_along": fl,
                                  "tau": args.so_tau, "iters": args.so_iters},
    }

    # -- leg 1: gauge magnitudes --------------------------------------------------
    report["leg1_gauge"] = gauge_stats(params)
    print(json.dumps({"leg1_gauge": report["leg1_gauge"]}, indent=2), flush=True)

    # -- leg 3: real byte-close packing of original vs projected variants ---------
    variants = {
        "orig": params,
        "gauge_proj": project_gauge(params),
        "palette_canon": project_palette_gauge(params),
        "both_proj": project_palette_gauge(project_gauge(params)),
    }
    blobs: dict[str, bytes] = {}
    report["leg3_rate"] = {}
    for name, pv in variants.items():
        blob, bd = build_blob_for(pv, cfg, so)
        blobs[name] = blob
        report["leg3_rate"][name] = {
            "total_0bin_bytes": bd["total_0bin_bytes"],
            "base_int8_brotli_bytes": bd["base_int8_brotli_bytes"],
            "code_int8_brotli_bytes": bd["code_int8_brotli_bytes"],
        }
    base = report["leg3_rate"]["orig"]["total_0bin_bytes"]
    for name in variants:
        report["leg3_rate"][name]["delta_total_bytes_vs_orig"] = (
            report["leg3_rate"][name]["total_0bin_bytes"] - base)
    print(json.dumps({"leg3_rate": report["leg3_rate"]}, indent=2), flush=True)

    # -- decode variants through the REAL receiver oracle -------------------------
    n_pairs = int(args.pairs)
    frames_by: dict[str, list[np.ndarray]] = {}
    argmax_by: dict[str, list[np.ndarray]] = {}
    for name in variants:
        tD = time.time()
        fr, am = decode_frames(blobs[name], n_pairs)
        frames_by[name] = fr
        argmax_by[name] = am
        print(f"[decode] {name}: {n_pairs} pairs in {time.time()-tD:.1f}s", flush=True)

    # frame bit-identity vs orig (exact-invariance check BEFORE scorers)
    report["leg4_frame_identity"] = {}
    for name in variants:
        if name == "orig":
            continue
        same = sum(int(np.array_equal(a, b)) for a, b in zip(frames_by["orig"], frames_by[name]))
        diffpx = [int(np.count_nonzero(a != b)) for a, b in zip(frames_by["orig"], frames_by[name])]
        am_same = sum(int(np.array_equal(a, b)) for a, b in zip(argmax_by["orig"], argmax_by[name]))
        report["leg4_frame_identity"][name] = {
            "frames_bit_identical": same, "frames_total": len(frames_by[name]),
            "max_diff_px_per_frame": int(max(diffpx)) if diffpx else 0,
            "mean_diff_px_per_frame": float(np.mean(diffpx)) if diffpx else 0.0,
            "render_argmax_identical_frames": am_same, "argmax_total": len(argmax_by[name]),
        }
    print(json.dumps({"leg4_frame_identity": report["leg4_frame_identity"]}, indent=2), flush=True)

    # -- leg 2a: blind (ker A) energy of the decoded output -----------------------
    Dr, Dc = _resize_matrices()
    Qr, Qc = _ortho_range(Dr), _ortho_range(Dc)
    report["leg2_projector_self_test"] = _projector_self_test(Dr, Dc, Qr, Qc)
    print(json.dumps({"leg2_projector_self_test": report["leg2_projector_self_test"]}), flush=True)
    bf = frames_by["orig"][: 2 * int(args.blind_frame_pairs)]
    report["leg2_blind_output"] = blind_output_stats(bf, Qr, Qc)
    print(json.dumps({"leg2_blind_output": report["leg2_blind_output"]}, indent=2), flush=True)

    # -- leg 2b: output-layer weight-direction blindness (Jacobian pushforward) ---
    if not args.skip_jacobian:
        m, dq_params, dq_code, _lp, _pc = bc._dequant_blob(blobs["orig"])
        jp = [0, max(0, n_pairs // 2)][: max(1, int(args.jacobian_pairs))]
        tJ = time.time()
        report["leg2_blind_weights"] = jacobian_blindness(m, dq_params, dq_code, jp, Qr, Qc)
        print(f"[jacobian] {time.time()-tJ:.1f}s", flush=True)
        print(json.dumps({"leg2_blind_weights": report["leg2_blind_weights"]["mean"]}, indent=2),
              flush=True)

    # -- leg 4: frozen CPU-torch scorer through the real decode --------------------
    if not args.skip_score:
        lstars, gt_poses = load_gt_subset(Path(args.gt_cache), n_pairs)
        seg_cpu, posenet_cpu = load_scorers()  # SCORER_LOADER_ORDER_OK: local wrapper (this file :379) returns (seg_cpu, posenet_cpu); unpack matches its verified order
        report["leg4_score"] = {}
        for name in variants:
            tS = time.time()
            report["leg4_score"][name] = score_frames(frames_by[name], lstars, gt_poses,
                                                      seg_cpu, posenet_cpu)
            print(f"[score] {name}: {time.time()-tS:.1f}s", flush=True)
        d0 = report["leg4_score"]["orig"]
        for name in variants:
            if name == "orig":
                continue
            report["leg4_score"][name]["delta_d_seg_vs_orig"] = (
                report["leg4_score"][name]["d_seg"] - d0["d_seg"])
            report["leg4_score"][name]["delta_d_pose_vs_orig"] = (
                report["leg4_score"][name]["d_pose"] - d0["d_pose"])
        print(json.dumps({"leg4_score": report["leg4_score"]}, indent=2), flush=True)

    report["elapsed_seconds"] = time.time() - t0
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2))
        print(f"[out] {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
