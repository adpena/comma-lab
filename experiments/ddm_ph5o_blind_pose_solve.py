#!/usr/bin/env python
"""ddm_ph5o -- O1: can the D-BLIND camera subspace be AIMED at d_pose?

THE QUESTION ``ddm_ph4`` LEFT OWED
----------------------------------
``ddm_ph4`` §2 PROVED that the 230,904 ``D``-blind camera pixels of frame_1 are
an EXACTLY seg-free actuator (``max|D(f1+delta_blind) - D(f1)| = 0.0e+00``,
20/20 cells) that nevertheless reaches PoseNet's frame_0 half through the warp
at an asymptotic gain of 0.2231 LSB/LSB.  It could not answer whether that
692,712-dimensional subspace can be *aimed*: capacity is not alignment.

``ph4`` §5 pre-registered O1 with two kills, both honoured here:

  * KILL(pose): if an aimed step does not reduce n600 ``d_pose`` by at least the
    instrument's reproducibility floor, the subspace is misaligned and the
    actuator is retired.
  * KILL(seg):  ``d_seg`` must come back BIT-IDENTICAL.  If it does not, ``ph4``
    §2.1's proof is wrong and everything built on it falls.  This is a FREE
    EXACT positive control and it is checked on every treated pair.

WHAT IS MEASURED
----------------
Everything runs through the ACTUAL shipped receiver on the ACTUAL shipped bytes
(``Decoder.f0`` -> real homography warp -> real rolling-shutter blend -> real
``a*x+b`` photometric -> real ``_to_uint8``), then through the frozen CPU-torch
PoseNet.  No surrogate: every reported number is a real forward evaluation.
The STE linearisation appears only in the gradient used to CHOOSE a T1
direction, never in a reported quantity.

  T1  AIMING (upper bound, NOT shippable -- a full sign field is 230,904 bits
      per pair).  The linearised descent direction d(d_pose)/d(f1_blind) is
      obtained by autograd through PoseNet to f0 at camera resolution, then
      pulled back through the EXACT adjoint of the shipped warp.  Step =
      sign(g) at +-``t1-amp`` LSB on the blind set.  Answers "does the subspace
      contain a descent direction at all?"

  T2  SHIPPABLE.  A rank-k correction over a GENERIC, deterministically
      generated separable-DCT basis (free in inflate.py per rule 118),
      restricted to the blind mask, applied EQUALLY to R,G,B -- which is a PURE
      LUMA step with exactly zero chroma effect (u=(B-Y)/1.772: an equal-RGB
      shift moves Y by the same amount and cancels).  Coefficients are solved
      by Gauss-Newton on the 6-scalar pose residual using REAL evaluations,
      then QUANTISED TO INTEGERS (1 B/coefficient) and re-measured.  Only the
      quantised number is quotable as shippable.

POSITIVE CONTROLS (all run before any result is reported; any failure STOPS)
---------------------------------------------------------------------------
  C1  blind mask reproduces 230,904 px = 22.696926%             (ph4 C1)
  C2  ph4's exact zero reproduced: max|D(f1+d_blind)-D(f1)| = 0  (ph4 §2.1)
  C3  cardinality-matched D-VISIBLE edit moves D -- without it a broken warp
      call and a genuine null are the same symbol (memory ``m50``) (ph4 C3)
  C4  warp adjoint dot-product identity <L u, v> == <u, L^T v>   (new here)
  C5  the re-implemented forward warp is BIT-IDENTICAL to the shipped
      ``warp_rgb``, and the full re-implemented chain is BIT-IDENTICAL to
      ``Decoder.f0`` on real frames                              (new here)
  C6  d_pose is bit-reproducible on repeated identical input
  C7  SegNet argmax BIT-IDENTICAL between base and treated frame_1 -- the
      end-to-end form of KILL(seg) on the real scorer, run by the sister
      script ``--segnet-verify``

m88 GUARD is EXECUTED, not cited: the treated subset's mean base d_pose is
reported against the n600 population value from the base ``report.txt``.

axis: [macOS-CPU advisory] NON-PROMOTABLE.  score_claim=false,
promotion_eligible=false, rank_or_kill_eligible=false.  Pointer UNMOVED.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SEG_H, SEG_W = 384, 512
#: base ``report.txt`` n600 values for the live pu2 archive (sha c72ef357...).
BASE_D_POSE_N600 = 0.00154519
BASE_D_SEG_N600 = 0.00431179
BASE_BYTES = 353805
ORIGINAL_BYTES = 37545489


def apply_D(frame_hwc_u8: np.ndarray) -> np.ndarray:
    """The EXACT scorer downsample (``upstream/modules.py:73`` == ``:109``)."""
    import torch

    arr = np.ascontiguousarray(frame_hwc_u8)
    t = torch.from_numpy(arr).permute(2, 0, 1)[None].float()
    out = torch.nn.functional.interpolate(t, size=(SEG_H, SEG_W), mode="bilinear")
    return out[0].permute(1, 2, 0).numpy().astype(np.float64)


class WarpOp:
    """The shipped bilinear warp as an explicit sparse linear operator.

    Transcribed line-for-line from ``pfs1_warp_receiver.warp_rgb``; C5 asserts
    bit-identity against the shipped function on real frames, so the adjoint
    below is the adjoint of the code that actually ships.
    """

    def __init__(self, hom: np.ndarray, tgt_grid: np.ndarray, hgt: int, wid: int):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            hinv = np.linalg.inv(hom)
            src_h = hinv @ tgt_grid
            zed = src_h[2]
            s_u = src_h[0] / zed
            s_v = src_h[1] / zed
        valid = (
            np.isfinite(s_u) & np.isfinite(s_v) & (zed > 0)
            & (s_u >= 0) & (s_u <= wid - 1) & (s_v >= 0) & (s_v <= hgt - 1)
        )
        su_c = np.clip(s_u, 0.0, wid - 1)
        sv_c = np.clip(s_v, 0.0, hgt - 1)
        x_0 = np.floor(su_c).astype(np.int64)
        y_0 = np.floor(sv_c).astype(np.int64)
        x_1 = np.minimum(x_0 + 1, wid - 1)
        y_1 = np.minimum(y_0 + 1, hgt - 1)
        w_x = su_c - x_0
        w_y = sv_c - y_0
        self.hgt, self.wid, self.npx = hgt, wid, hgt * wid
        self.valid = valid
        self.idx = np.stack([
            y_0 * wid + x_0, y_0 * wid + x_1, y_1 * wid + x_0, y_1 * wid + x_1,
        ])
        self.wgt = np.stack([
            (1.0 - w_x) * (1.0 - w_y), w_x * (1.0 - w_y),
            (1.0 - w_x) * w_y, w_x * w_y,
        ])

    def forward(self, src_hwc: np.ndarray) -> np.ndarray:
        chan = src_hwc.shape[2]
        flat = np.asarray(src_hwc, np.float64).reshape(-1, chan)
        out = np.zeros_like(flat)
        for j in range(4):
            out += flat[self.idx[j]] * self.wgt[j][:, None]
        # where invalid, warp_rgb passes the SOURCE pixel at the SAME flat index
        # straight through -- an identity term the adjoint must also carry.
        out = np.where(self.valid[:, None], out, flat)
        return out.reshape(self.hgt, self.wid, chan)

    def adjoint(self, grad_hwc: np.ndarray) -> np.ndarray:
        """L^T by scatter-add of the same weights.  Verified by C4."""
        chan = grad_hwc.shape[2]
        gflat = np.asarray(grad_hwc, np.float64).reshape(-1, chan)
        out = np.zeros_like(gflat)
        vmask = self.valid
        for c in range(chan):
            gcol = gflat[:, c]
            gval = np.where(vmask, gcol, 0.0)
            for j in range(4):
                out[:, c] += np.bincount(
                    self.idx[j], weights=gval * self.wgt[j], minlength=self.npx)
            out[:, c] += np.where(vmask, 0.0, gcol)
        return out.reshape(self.hgt, self.wid, chan)


class PairWarpChain:
    """The full f1_camera -> f0f (pre-``_to_uint8``) affine map of ``Decoder.f0``.

    Mirrors ``Decoder.f0`` + ``Decoder._warp_pair`` exactly: the ``sel``
    far/ground blend, the rolling-shutter two-rotation blend, and ``a*x + b``.
    """

    def __init__(self, dec, pidx: int, hgt: int, wid: int):
        from pfs1_warp_receiver import pose_to_homography

        s_t = float(dec.st_vals[dec.st_idx[pidx]])
        pose = dec.p_best[pidx]
        sel = int(dec.sel[pidx])
        self.a_ph = float(dec.ab[pidx][0])
        self.b_ph = float(dec.ab[pidx][1])
        beta_mag = float(dec.beta_mags[int(dec.beta_idx[pidx])])
        if beta_mag != 0.0:
            beta = beta_mag * (1.0 if pose[5] >= 0.0 else -1.0)
            rots = [(1.0 - beta / 2.0, 1.0 - dec._alpha),
                    (1.0 + beta / 2.0, dec._alpha)]
        else:
            rots = [(1.0, None)]
        far = dec._far[..., None].astype(np.float64)
        terms = []
        for rot, rot_w in rots:
            h_g = pose_to_homography(pose, dec.K, dec.Kinv, s_t, rot, 0.0)
            op_g = WarpOp(h_g, dec.grid, hgt, wid)
            if sel == 0:
                terms.append((op_g, rot_w, None))
            else:
                h_f = pose_to_homography(pose, dec.K, dec.Kinv, 0.0, rot, 0.0)
                op_f = WarpOp(h_f, dec.grid, hgt, wid)
                terms.append((op_g, rot_w, 1.0 - far))
                terms.append((op_f, rot_w, far))
        self.terms = terms

    def linear_f0f(self, f1_f: np.ndarray) -> np.ndarray:
        """The LINEAR part a*L(f1) -- the affine offset b is added separately."""
        acc = None
        for op, rot_w, spatial in self.terms:
            val = op.forward(f1_f)
            if spatial is not None:
                val = val * spatial
            if rot_w is not None:
                val = val * rot_w
            acc = val if acc is None else acc + val
        return self.a_ph * acc

    def forward_f0f(self, f1_f: np.ndarray) -> np.ndarray:
        return self.linear_f0f(f1_f) + self.b_ph

    def adjoint_to_f1(self, g_f0f: np.ndarray) -> np.ndarray:
        acc = None
        for op, rot_w, spatial in self.terms:
            gee = g_f0f
            if rot_w is not None:
                gee = gee * rot_w
            if spatial is not None:
                gee = gee * spatial
            val = op.adjoint(np.ascontiguousarray(gee))
            acc = val if acc is None else acc + val
        return self.a_ph * acc


def dpose_grad_wrt_f0cam(posenet, f0_u8, f1_u8, target) -> np.ndarray:
    """d(d_pose)/d(f0 at camera resolution) by autograd through frozen PoseNet."""
    import torch

    f0t = torch.from_numpy(np.ascontiguousarray(f0_u8)).float().requires_grad_(True)
    f1t = torch.from_numpy(np.ascontiguousarray(f1_u8)).float()
    stacked = torch.stack([f0t.permute(2, 0, 1), f1t.permute(2, 0, 1)])[None]
    out = posenet(posenet.preprocess_input(stacked))
    pose = out["pose"] if isinstance(out, dict) else out
    tgt = torch.from_numpy(np.asarray(target, np.float64)).float()
    ((pose[0, :6] - tgt) ** 2).mean().backward()
    return f0t.grad.detach().numpy().astype(np.float64)


def dct_modes(k: int) -> list[tuple[int, int]]:
    cand = sorted(
        ((jy, jx) for jy in range(8) for jx in range(8)),
        key=lambda m: (m[0] + m[1], m[0], m[1]),
    )
    return cand[:k]


def build_basis(k: int, hgt: int, wid: int) -> np.ndarray:
    """(k, H, W) float64.  Deterministically generated -> FREE in inflate.py."""
    yy = (np.arange(hgt) + 0.5) / hgt
    xx = (np.arange(wid) + 0.5) / wid
    out = np.empty((k, hgt, wid), np.float64)
    for j, (j_y, j_x) in enumerate(dct_modes(k)):
        out[j] = np.cos(np.pi * j_y * yy)[:, None] * np.cos(np.pi * j_x * xx)[None, :]
    return out


def field_from_coeffs(coeffs, basis: np.ndarray, blind: np.ndarray) -> np.ndarray:
    """Equal-RGB (pure-luma, exactly zero chroma) field on the blind set only."""
    return np.tensordot(np.asarray(coeffs, np.float64), basis, axes=(0, 0)) * blind


def apply_blind_field(f1_u8: np.ndarray, field_hw: np.ndarray) -> np.ndarray:
    out = f1_u8.astype(np.float64) + field_hw[..., None]
    return np.clip(np.round(out), 0.0, 255.0).astype(np.uint8)


def _run_controls(dec, shipped, p3v2, blind, ctrl_mask, n_pairs, rng,
                  hgt, wid, targets) -> dict:
    from pfs1_warp_receiver import pose_to_homography, warp_rgb

    ctrl: dict = {
        "C1_blind_px": int(blind.sum()),
        "C1_blind_frac": float(blind.mean()),
        "C1_pass": bool(int(blind.sum()) == 230904),
    }
    print(f"[ph5o] C1 blind = {int(blind.sum()):,} px "
          f"({100 * blind.mean():.6f}%) pass={ctrl['C1_pass']}", flush=True)
    c2_max, c3_min = 0.0, np.inf
    c5_warp, c5_chain, c4_rel = 0.0, 0.0, 0.0
    c6_ok = True
    probe = [int(v) for v in np.unique(
        np.linspace(0, n_pairs - 1, 3).round().astype(int))]
    for pidx in probe:
        f1b = np.asarray(shipped.render_frame1_camera_uint8(dec.packet, pidx))
        d_ref = apply_D(f1b)
        amp = 3
        step = np.where(f1b.astype(np.int16) <= 127, amp, -amp).astype(np.int16)
        t_b = f1b.astype(np.int16).copy()
        t_b[blind] += step[blind]
        c2_max = max(c2_max, float(np.abs(
            apply_D(np.clip(t_b, 0, 255).astype(np.uint8)) - d_ref).max()))
        t_v = f1b.astype(np.int16).copy()
        t_v[ctrl_mask] += step[ctrl_mask]
        c3_min = min(c3_min, float(np.abs(
            apply_D(np.clip(t_v, 0, 255).astype(np.uint8)) - d_ref).max()))
        # C5a: single-homography forward parity against the SHIPPED warp_rgb
        hom = pose_to_homography(dec.p_best[pidx], dec.K, dec.Kinv,
                                 float(dec.st_vals[dec.st_idx[pidx]]), 1.0, 0.0)
        op = WarpOp(hom, dec.grid, hgt, wid)
        f1_f = f1b.astype(np.float64)
        c5_warp = max(c5_warp, float(np.abs(
            op.forward(f1_f) - warp_rgb(f1_f, hom, dec.grid)).max()))
        # C5b: the FULL re-implemented chain vs the shipped Decoder.f0
        chain = PairWarpChain(dec, pidx, hgt, wid)
        mine = np.clip(np.round(chain.forward_f0f(f1_f)), 0.0, 255.0).astype(np.uint8)
        ship = dec.f0(pidx, f1b)
        c5_chain = max(c5_chain, float(np.abs(
            mine.astype(np.int32) - ship.astype(np.int32)).max()))
        # C4: adjoint dot-product identity on the LINEAR part of the full chain
        u_r = rng.standard_normal((hgt, wid, 3))
        v_r = rng.standard_normal((hgt, wid, 3))
        lhs = float(np.sum(chain.linear_f0f(u_r) * v_r))
        rhs = float(np.sum(u_r * chain.adjoint_to_f1(v_r)))
        c4_rel = max(c4_rel, abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30))
        d_1 = p3v2.d_pose_u8(dec._posenet, ship, f1b, targets[pidx])
        d_2 = p3v2.d_pose_u8(dec._posenet, ship, f1b, targets[pidx])
        c6_ok &= (d_1 == d_2)
    ctrl.update({
        "C2_blind_edit_D_delta_max": c2_max,
        "C2_pass": bool(c2_max == 0.0),
        "C3_visible_edit_D_delta_min": float(c3_min),
        "C3_pass": bool(c3_min > 0.0),
        "C4_adjoint_dot_product_rel_err": float(c4_rel),
        "C4_pass": bool(c4_rel < 1e-9),
        # C5 has TWO legs and only ONE of them can be demanded exactly.  The
        # float leg re-associates the bilinear sum (sum_j I_j w_j here vs
        # (Ia(1-wx)+Ib wx)(1-wy)+... in warp_rgb): algebraically identical,
        # so it agrees to ULP, not to the bit.  MEASURED 8.53e-14 max abs =
        # 1.5 ULP at magnitude 255.  The leg that is LOAD-BEARING -- the uint8
        # frame the receiver actually emits -- is demanded BIT-IDENTICAL and
        # measures 0.0.  Gating the float leg at 0.0 was my error, recorded
        # rather than smoothed: it fired on the first run.
        "C5a_warp_float_vs_shipped_max_abs": float(c5_warp),
        "C5a_warp_float_ulps_at_255": float(
            c5_warp / (255.0 * np.finfo(np.float64).eps)),
        "C5a_pass": bool(c5_warp <= 8.0 * 255.0 * np.finfo(np.float64).eps),
        "C5b_chain_uint8_vs_Decoder_f0_max_abs": float(c5_chain),
        "C5b_pass": bool(c5_chain == 0.0),
        "C5_pass": bool(c5_chain == 0.0
                        and c5_warp <= 8.0 * 255.0 * np.finfo(np.float64).eps),
        "C6_d_pose_bit_reproducible": bool(c6_ok),
        "C6_pass": bool(c6_ok),
        "probe_pairs": probe,
    })
    print(f"[ph5o] C2 max|D(f1+d_blind)-D(f1)| = {c2_max:.3e} (MUST be 0.0) "
          f"pass={ctrl['C2_pass']}", flush=True)
    print(f"[ph5o] C3 min|D(f1+d_visible)-D(f1)| = {c3_min:.4f} (MUST be >0) "
          f"pass={ctrl['C3_pass']}", flush=True)
    print(f"[ph5o] C4 adjoint rel err = {c4_rel:.3e} pass={ctrl['C4_pass']}",
          flush=True)
    print(f"[ph5o] C5a warp FLOAT parity {c5_warp:.3e} "
          f"({ctrl['C5a_warp_float_ulps_at_255']:.2f} ULP @255) "
          f"pass={ctrl['C5a_pass']}", flush=True)
    print(f"[ph5o] C5b chain UINT8 parity {c5_chain:.3e} (MUST be 0.0) "
          f"pass={ctrl['C5b_pass']}", flush=True)
    print(f"[ph5o] C6 d_pose bit-reproducible pass={ctrl['C6_pass']}", flush=True)
    return ctrl


def _solve_pair(pidx, dec, shipped, p3v2, posenet, targets, blind, basis,
                args, hgt, wid, do_t1: bool) -> dict:
    f1b = np.asarray(shipped.render_frame1_camera_uint8(dec.packet, pidx))
    f1_f = f1b.astype(np.float64)
    f0b = dec.f0(pidx, f1b)
    dp_base = p3v2.d_pose_u8(posenet, f0b, f1b, targets[pidx])
    row = {"pair": pidx, "d_pose_base": dp_base}

    def evaluate(field_hw):
        f1t = apply_blind_field(f1b, field_hw)
        return p3v2.d_pose_u8(posenet, dec.f0(pidx, f1t), f1t, targets[pidx]), f1t

    # ANTI-VACUITY GUARD (memory m50, and my own first run tripped it).  A
    # solver that falls back to the base value when nothing improves makes
    # "stepped and got worse" and "never stepped at all" the SAME SYMBOL.
    # Every row below therefore carries at least one evaluation on a frame
    # that PROVABLY changed, and every reported d_pose is an actual forward
    # evaluation -- never a fallback.
    probe_field = field_from_coeffs(
        np.eye(1, args.rank, 0)[0] * 1.0, basis, blind)
    dp_probe, f1_probe = evaluate(probe_field)
    row["probe_unit_constant_mode"] = {
        "d_pose": dp_probe,
        "f1_px_changed": int((f1_probe != f1b).sum()),
        "rel_delta": dp_probe / dp_base - 1.0 if dp_base > 0 else float("nan"),
    }

    if do_t1:
        chain = PairWarpChain(dec, pidx, hgt, wid)
        g_f0 = dpose_grad_wrt_f0cam(posenet, f0b, f1b, targets[pidx])
        f0f = chain.forward_f0f(f1_f)
        ste = ((f0f >= -0.5) & (f0f <= 255.5)).astype(np.float64)
        g_lum = chain.adjoint_to_f1(g_f0 * ste).sum(axis=2) * blind
        sgn = np.sign(g_lum)
        # the LINEARISED drop a sign step of amplitude A must deliver if the
        # response were first-order:  sum_p g(p) * (-A*sign(g(p))) = -A*sum|g|.
        grad_abs_sum = float(np.abs(g_lum)[blind].sum())
        grid = {}
        for sign in (-1.0, +1.0):
            for amp in args.t1_amps:
                val, f1t = evaluate(sign * amp * sgn)
                grid[f"{int(sign * amp):+d}"] = {
                    "d_pose": val,
                    "rel_delta": (val / dp_base - 1.0 if dp_base > 0
                                  else float("nan")),
                    "f1_px_changed": int((f1t != f1b).sum()),
                    "predicted_linear_delta": -sign * amp * grad_abs_sum * sign,
                }
        best_key = min(grid, key=lambda k: grid[k]["d_pose"])
        row["t1_grid"] = grid
        row["t1_best_key"] = best_key
        row["t1_best_d_pose"] = grid[best_key]["d_pose"]
        row["t1_improved"] = bool(grid[best_key]["d_pose"] < dp_base)
        row["t1_grad_abs_sum_blind"] = grad_abs_sum
        row["t1_grad_lum_absmean_blind"] = float(np.abs(g_lum)[blind].mean())
        # THE DISCRIMINATOR: is the actuator MISALIGNED, or is it ALIGNED but
        # QUANTISATION-FLOORED (second-order penalty already dominant at the
        # smallest realisable +-1 LSB step)?  Compare the first-order drop the
        # gradient promises at amp 1 against what actually happened.
        row["t1_predicted_drop_amp1"] = grad_abs_sum
        row["t1_measured_delta_amp1_bestsign"] = float(
            min(grid[k]["d_pose"] for k in ("-1", "+1") if k in grid) - dp_base)

    coef = np.zeros(args.rank, np.float64)
    cur = dp_base
    ls_trace: list[dict] = []
    for _ in range(args.gn_iters):
        f1c = apply_blind_field(f1b, field_from_coeffs(coef, basis, blind))
        f0c = dec.f0(pidx, f1c)
        p_0 = p3v2.pose6_u8(posenet, f0c, f1c)
        res = p_0 - targets[pidx]
        jac = np.empty((6, args.rank), np.float64)
        for j in range(args.rank):
            bump = coef.copy()
            bump[j] += args.fd_step
            f1j = apply_blind_field(f1b, field_from_coeffs(bump, basis, blind))
            jac[:, j] = (p3v2.pose6_u8(posenet, dec.f0(pidx, f1j), f1j)
                         - p_0) / args.fd_step
        jtj = jac.T @ jac
        lam = 1e-8 * max(float(np.trace(jtj)), 1e-30) + 1e-300
        try:
            step = -np.linalg.solve(jtj + lam * np.eye(args.rank), jac.T @ res)
        except np.linalg.LinAlgError:
            break
        improved = False
        for scale in (1.0, 0.5, 0.25, 0.125, 2.0):
            cand = coef + scale * step
            if not np.all(np.isfinite(cand)) or np.abs(cand).max() > 127:
                continue
            val, _ = evaluate(field_from_coeffs(cand, basis, blind))
            ls_trace.append({"scale": float(scale), "d_pose": val,
                             "coef_absmax": float(np.abs(cand).max())})
            if val < cur:
                cur, coef, improved = val, cand, True
                break
        if not improved:
            break

    coef_q = np.clip(np.round(coef), -127, 127)
    dp_q, f1_q = evaluate(field_from_coeffs(coef_q, basis, blind))
    seg_delta = float(np.abs(apply_D(f1_q) - apply_D(f1b)).max())
    # the seg KILL is checked on the frame with the LARGEST realised edit seen
    # for this pair, not only on the (possibly all-zero) shipped one.
    seg_delta_probe = float(np.abs(apply_D(f1_probe) - apply_D(f1b)).max())
    row.update({
        "t2_d_pose_float": cur,
        "t2_d_pose_quantised": dp_q,
        "t2_coeffs_float": [float(v) for v in coef],
        "t2_coeffs_int": [int(v) for v in coef_q],
        "t2_coef_is_all_zero": bool(np.all(coef_q == 0)),
        "t2_frame_changed_px": int((f1_q != f1b).sum()),
        "t2_line_search_trace": ls_trace,
        "seg_plane_delta_max": seg_delta,
        "seg_plane_delta_max_on_probe_edit": seg_delta_probe,
        "seg_bit_identical": bool(seg_delta == 0.0 and seg_delta_probe == 0.0),
    })
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True, type=Path)
    ap.add_argument("--pairs", default="all",
                    help="'all' (n600) or an integer count of STRIDED pairs")
    ap.add_argument("--chunk", type=int, default=120)
    ap.add_argument("--rank", type=int, default=6, help="k, coefficients/pair")
    ap.add_argument("--gn-iters", type=int, default=2)
    ap.add_argument("--fd-step", type=float, default=1.0,
                    help="finite-difference step in LSB (== the quantum)")
    ap.add_argument("--t1-amps", type=int, nargs="+", default=[1, 2, 4],
                    help="aimed-step amplitudes in LSB; 1 is the pre-registered "
                         "one, the rest map the response along the aimed ray")
    ap.add_argument("--t1-pairs", type=int, default=0,
                    help="run T1 (expensive adjoint) on this many STRIDED "
                         "pairs; 0 = never, -1 = every treated pair")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=Path, required=True,
                    help="resumable per-pair JSONL")
    args = ap.parse_args()

    sub = args.submission_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    for p in (str(sub), str(root / "upstream"), str(root / "experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch

    torch.set_num_threads(4)
    import ddm_p3v2_optimal_form_pose_resolve as p3v2
    import ddm_tr1_runtime as shipped  # the VENDORED receiver
    from inflate_runner import Decoder

    from tac.optimization.ddm_ll1_window_solve import blind_mask

    rng = np.random.default_rng(args.seed)
    posenet, _ = p3v2.load_posenet()
    targets = p3v2.load_targets(600)
    dec = Decoder(sub / "archive")
    dec._posenet = posenet
    n_pairs = int(dec.n_pairs)
    blind = blind_mask()
    hgt, wid = blind.shape
    basis = build_basis(args.rank, hgt, wid)
    vis_idx = np.flatnonzero((~blind).ravel())
    cmask = np.zeros(blind.size, bool)
    cmask[rng.choice(vis_idx, size=int(blind.sum()), replace=False)] = True
    cmask = cmask.reshape(blind.shape)

    print("[ph5o] ===== CONTROLS =====", flush=True)
    ctrl = _run_controls(dec, shipped, p3v2, blind, cmask, n_pairs, rng,
                         hgt, wid, targets)
    hard = [k for k in ("C1", "C2", "C3", "C4", "C5", "C6")
            if not ctrl[k + "_pass"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if hard:
        print(f"[ph5o] *** CONTROLS FAILED {hard} -- BROKEN INSTRUMENT, STOP ***",
              flush=True)
        args.out.write_text(json.dumps(
            {"arm": "ddm_ph5o", "STOPPED": "controls_failed",
             "failed": hard, "controls": ctrl}, indent=2))
        return 3

    if args.pairs == "all":
        idx = np.arange(n_pairs)
    else:
        idx = np.unique(np.linspace(
            0, n_pairs - 1, int(args.pairs)).round().astype(int))
    if args.t1_pairs == -1:
        t1_set = {int(v) for v in idx}
    elif args.t1_pairs > 0:
        t1_set = {int(v) for v in np.unique(np.linspace(
            0, len(idx) - 1, args.t1_pairs).round().astype(int))}
        t1_set = {int(idx[v]) for v in sorted(t1_set)}
    else:
        t1_set = set()

    args.rows.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, dict] = {}
    if args.rows.exists():
        for line in args.rows.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[int(rec["pair"])] = rec
        print(f"[ph5o] RESUME: {len(done)} rows already on disk", flush=True)

    handle = args.rows.open("a")
    t_0 = time.time()
    todo = [int(v) for v in idx if int(v) not in done]
    print(f"[ph5o] base={sub.name} pairs={len(idx)} todo={len(todo)} "
          f"rank={args.rank} chunk={args.chunk} T1 on {len(t1_set)} pairs",
          flush=True)

    for pos, pidx in enumerate(todo):
        row = _solve_pair(pidx, dec, shipped, p3v2, posenet, targets, blind,
                          basis, args, hgt, wid, pidx in t1_set)
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        done[pidx] = row
        if pos % 10 == 0 or pos < 5 or (pos + 1) % args.chunk == 0:
            ela = time.time() - t_0
            print(f"[ph5o] {pos + 1:4d}/{len(todo)} pair {pidx:4d} | base "
                  f"{row['d_pose_base']:.8f} -> T2q "
                  f"{row['t2_d_pose_quantised']:.8f} | T1 "
                  f"{row.get('t1_best_d_pose', float('nan')):.8f} | segd "
                  f"{row['seg_plane_delta_max']:.1e} | {ela / 60:.1f}m "
                  f"({ela / (pos + 1):.1f}s/pair)", flush=True)
    handle.close()

    rows = [done[int(v)] for v in idx if int(v) in done]
    base = np.array([r["d_pose_base"] for r in rows])
    t2q = np.array([r["t2_d_pose_quantised"] for r in rows])
    t2f = np.array([r["t2_d_pose_float"] for r in rows])
    t1_rows = [r for r in rows if "t1_best_d_pose" in r]
    seg_all_exact = all(r["seg_bit_identical"] for r in rows)
    seg_worst = float(max(r["seg_plane_delta_max"] for r in rows))
    mean_base = float(base.mean())

    def s_pose(mean_dp: float) -> float:
        return float(np.sqrt(10.0 * mean_dp))

    order = np.argsort(-base)
    gain = base - t2q
    curve = []
    for top_n in (1, 3, 6, 12, 25, 50, 100, 200, 400, len(rows)):
        if top_n > len(rows):
            continue
        sel = order[:top_n]
        new_mean = mean_base - float(gain[sel].sum()) / len(rows)
        coef_bytes = args.rank * top_n
        sel_bytes = 0 if top_n == len(rows) else int(np.ceil(
            top_n * np.log2(max(len(rows), 2)) / 8.0))
        tot_b = coef_bytes + sel_bytes
        curve.append({
            "top_n": int(top_n),
            "coeff_bytes": int(coef_bytes),
            "selector_bytes": int(sel_bytes),
            "total_bytes": int(tot_b),
            "d_pose_mean": new_mean,
            "delta_S_pose": s_pose(new_mean) - s_pose(mean_base),
            "delta_S_rate": 25.0 * tot_b / ORIGINAL_BYTES,
            "delta_S_joint": (s_pose(new_mean) - s_pose(mean_base)
                              + 25.0 * tot_b / ORIGINAL_BYTES),
        })

    m88_ratio = mean_base / BASE_D_POSE_N600
    summary = {
        "arm": "ddm_ph5o",
        "question": "O1 -- can the D-blind camera subspace be AIMED at d_pose?",
        "base_submission": str(sub),
        "base_archive_bytes": BASE_BYTES,
        "base_d_pose_n600_report_txt": BASE_D_POSE_N600,
        "base_d_seg_n600_report_txt": BASE_D_SEG_N600,
        "n_pairs_total": n_pairs,
        "n_pairs_measured": len(rows),
        "is_n600": bool(len(rows) == 600),
        "rank_k": args.rank,
        "fd_step_lsb": args.fd_step,
        "gn_iters": args.gn_iters,
        "t1_amps": list(args.t1_amps),
        "controls": ctrl,
        "m88_guard": {
            "subset_mean_base_d_pose": mean_base,
            "population_d_pose_report_txt": BASE_D_POSE_N600,
            "subset_over_population": m88_ratio,
            "subset_is_representative": bool(0.5 <= m88_ratio <= 2.0),
        },
        "KILL_seg_bit_identical_all_pairs": bool(seg_all_exact),
        "KILL_seg_worst_D_plane_delta": seg_worst,
        "d_pose_base_mean": mean_base,
        "d_pose_T2_float_mean": float(t2f.mean()),
        "d_pose_T2_quantised_mean": float(t2q.mean()),
        "rel_delta_T2_quantised": float(t2q.mean() / mean_base - 1.0),
        "frac_pairs_T2q_improved": float(np.mean(t2q < base)),
        "T1": ({
            "n_pairs": len(t1_rows),
            "pairs": [r["pair"] for r in t1_rows],
            "d_pose_base_mean": float(np.mean(
                [r["d_pose_base"] for r in t1_rows])),
            "d_pose_T1_best_mean": float(np.mean(
                [r["t1_best_d_pose"] for r in t1_rows])),
            "rel_delta": float(
                np.mean([r["t1_best_d_pose"] for r in t1_rows])
                / np.mean([r["d_pose_base"] for r in t1_rows]) - 1.0),
            "frac_improved": float(np.mean(
                [r["t1_improved"] for r in t1_rows])),
            "per_amp_rel_delta": {
                key: float(np.mean([r["t1_grid"][key]["rel_delta"]
                                    for r in t1_rows if key in r["t1_grid"]]))
                for key in sorted({k for r in t1_rows for k in r["t1_grid"]})
            },
            "ALIGNMENT_DISCRIMINATOR": {
                "note": ("predicted_drop_amp1 is the FIRST-ORDER drop the "
                         "measured gradient promises for a +-1 LSB sign step "
                         "(= A * sum|g| over the blind set).  If it dwarfs the "
                         "measured delta the actuator is ALIGNED but "
                         "QUANTISATION-FLOORED (the second-order f0/f1 "
                         "consistency penalty already dominates at the "
                         "smallest realisable step).  If both are tiny the "
                         "actuator is MISALIGNED."),
                "predicted_drop_amp1_mean": float(np.mean(
                    [r["t1_predicted_drop_amp1"] for r in t1_rows])),
                "measured_delta_amp1_mean": float(np.mean(
                    [r["t1_measured_delta_amp1_bestsign"] for r in t1_rows])),
                "predicted_over_measured_abs": float(
                    np.mean([r["t1_predicted_drop_amp1"] for r in t1_rows])
                    / max(abs(float(np.mean(
                        [r["t1_measured_delta_amp1_bestsign"]
                         for r in t1_rows]))), 1e-30)),
            },
        } if t1_rows else None),
        "probe_unit_constant_mode": {
            "n_pairs": len(rows),
            "frames_provably_changed": int(sum(
                r["probe_unit_constant_mode"]["f1_px_changed"] > 0
                for r in rows)),
            "mean_rel_delta": float(np.mean(
                [r["probe_unit_constant_mode"]["rel_delta"] for r in rows])),
            "frac_improved": float(np.mean(
                [r["probe_unit_constant_mode"]["d_pose"] < r["d_pose_base"]
                 for r in rows])),
        },
        "t2_frac_rows_coef_all_zero": float(np.mean(
            [r["t2_coef_is_all_zero"] for r in rows])),
        "S_pose_base": s_pose(mean_base),
        "S_pose_T2_quantised": s_pose(float(t2q.mean())),
        "byte_benefit_curve": curve,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "rows_path": str(args.rows),
    }
    args.out.write_text(json.dumps(summary, indent=2))

    print("\n[ph5o] ===== RESULT =====", flush=True)
    print(f"[ph5o] pairs {len(rows)} (n600={len(rows) == 600})  "
          f"m88 ratio {m88_ratio:.3f}", flush=True)
    print(f"[ph5o] KILL(seg) bit-identical every pair: {seg_all_exact} "
          f"(worst D-plane delta {seg_worst:.3e})", flush=True)
    print(f"[ph5o] d_pose base     {mean_base:.10f}", flush=True)
    prb = summary["probe_unit_constant_mode"]
    print(f"[ph5o] ANTI-VACUITY probe: {prb['frames_provably_changed']}/"
          f"{prb['n_pairs']} frames provably changed, mean rel delta "
          f"{100 * prb['mean_rel_delta']:+.4f}%, improved "
          f"{100 * prb['frac_improved']:.1f}%", flush=True)
    if t1_rows:
        t1s = summary["T1"]
        print(f"[ph5o] d_pose T1 (n={t1s['n_pairs']}) "
              f"{t1s['d_pose_base_mean']:.10f} -> "
              f"{t1s['d_pose_T1_best_mean']:.10f} "
              f"rel {100 * t1s['rel_delta']:+.4f}%  improved "
              f"{100 * t1s['frac_improved']:.1f}%", flush=True)
        for key, val in t1s["per_amp_rel_delta"].items():
            print(f"[ph5o]     aimed step {key:>3s} LSB -> rel d_pose "
                  f"{100 * val:+9.4f}%", flush=True)
        dsc = t1s["ALIGNMENT_DISCRIMINATOR"]
        print(f"[ph5o]   DISCRIMINATOR predicted first-order drop @amp1 "
              f"{dsc['predicted_drop_amp1_mean']:.6e} vs measured delta "
              f"{dsc['measured_delta_amp1_mean']:+.6e} "
              f"(ratio {dsc['predicted_over_measured_abs']:.3g})", flush=True)
    print(f"[ph5o] T2 rows whose shipped coefficients are ALL ZERO: "
          f"{100 * summary['t2_frac_rows_coef_all_zero']:.1f}%", flush=True)
    print(f"[ph5o] d_pose T2 float {float(t2f.mean()):.10f}", flush=True)
    print(f"[ph5o] d_pose T2 int   {float(t2q.mean()):.10f}  rel "
          f"{100 * (t2q.mean() / mean_base - 1):+.4f}%  improved "
          f"{100 * np.mean(t2q < base):.1f}%", flush=True)
    print("[ph5o] byte/benefit curve (treat top-N by base d_pose):", flush=True)
    for crow in curve:
        print(f"[ph5o]   N={crow['top_n']:4d} {crow['total_bytes']:6d} B | "
              f"dS_pose {crow['delta_S_pose']:+.6f} dS_rate "
              f"{crow['delta_S_rate']:+.6f} => JOINT "
              f"{crow['delta_S_joint']:+.6f}", flush=True)
    print(f"[ph5o] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
