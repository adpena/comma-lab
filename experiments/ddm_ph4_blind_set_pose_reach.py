#!/usr/bin/env python
"""ddm_ph4 -- is the D-blind camera set a POSE actuator, or is it dead bits?

THE CONTRADICTION THIS SETTLES
------------------------------
Two arms wrote opposite things about the same 230,904 camera pixels in the
same 48 hours:

  * ``ddm_rz1`` §1 R1(c):   "22.6969% of camera pixels (230,904) are blind to
    BOTH scorers and must receive zero bits."  Ranked in its PREDICTED-NULL
    table as "null by construction -- read by neither scorer", "do not spend".

  * ``ddm_ra1`` §6:  "The blind set is blind to ``D`` but **not** to the warp."

They cannot both be right.  The difference matters a great deal, because if
``ra1`` is right then this set is an actuator with a property nothing else in
the vehicle has:

  SegNet reads ``x[:, -1, ...]`` -- frame_1 ONLY -- and then ``D``
  (``upstream/modules.py:107-109``).  ``D`` is bilinear point-sampling at
  stride 2.276 > 2, so it reads 768 of 874 rows and 1024 of 1164 columns and
  NEVER touches the remaining 230,904 camera pixels.  SegNet therefore has
  exactly ONE path to frame_1, and that path cannot see them.

  => an edit confined to the D-blind camera pixels of frame_1 changes d_seg by
     EXACTLY ZERO.  Not approximately; not to within a quantization residue.
     Exactly, and with no lattice caveat, because SegNet has only one lattice.

PoseNet, by contrast, reads BOTH frames, and in this vehicle frame_0 is
manufactured by warping frame_1's CAMERA raster
(``inflate_runner.py:307-327``: ``_warp_pair`` at sub-pixel homography
offsets, then ``a*f0 + b``).  That warp samples frame_1 at positions that have
nothing to do with ``D``'s lattice.  ``ddm_pz1`` §7.1 established the general
law -- *null-space membership does not survive a change of lattice* -- and
measured a camera->scorer attenuation of only 1.662x for one particular
D-null field.  The same law applied to the blind set predicts ``ra1`` is right
and ``rz1``'s R1(c) is too strong.

WHAT IS MEASURED (all $0, ZERO scorer forward passes)
-----------------------------------------------------
For STRIDED pairs (never a prefix, per memory ``m88``), on the live shipped
receiver and the live shipped bytes:

  * SEG-FREEDOM (the load-bearing half):  ``D(f1+delta) - D(f1)`` for a
    blind-only delta.  MUST be exactly 0.0 -- this is a proof, not a statistic.
  * POSE-REACH  (the contested half):  ``D(f0(f1+delta)) - D(f0(f1))``, i.e.
    the delta in the literal tensor PoseNet's frame_0 half consumes, plus the
    yuv6 and PoseNet-normalised (``/63.75``) forms.
  * LEVERAGE:  the blind set's share of the warp's total sampling weight,
    measured as the ratio of the f0 response to a blind-only unit step against
    the f0 response to an ALL-pixel unit step.  22.6969% would mean the blind
    set carries exactly its population share of the warp's attention.

    ** READ THE AMPLITUDE SWEEP, NOT THE SINGLE-AMPLITUDE RATIO. **  At amp 1
    this ratio reads 0.32, which looks like a 1.4x over-weighting and is NOT.
    ``_to_uint8``'s round acts as a THRESHOLD at sub-LSB response, inflating
    the smallest step by 1.204x.  Run ``--amps 1 2 4 8 16``: the gain decays
    monotonically to 0.2231 against the population share 0.226969 (1.7%).
    Measured 2026-08-03; it refuted this arm's own first reading.

  * ``--edit-set visible`` measures the SAME pass-through for the D-VISIBLE
    complement -- the set a Q3 isoluminant chroma edit actually writes to.
    Measured 0.8902, which is 4.0x the blind set's and five-to-six orders of
    magnitude above the <=0.9-LSB uint8 residue ``ddm_rz1`` §3.2 pre-registered
    as the expected break in its "exactly pose-free by construction" claim.
    NOTE the two do NOT add (0.2231 + 0.8902 = 1.1133 > 0.9809 all-pixel):
    rms does not add across spatially correlated response fields, so neither
    may be inferred from the other by interpolation.

POSITIVE CONTROLS (run before any number is reported)
-----------------------------------------------------
  C1  blind-set size reproduces the independently-known 230,904 (22.696926%)
  C2  the probe can return the negative: a NULL delta reports exactly 0
      everywhere, on every channel, including the f0 path
  C3  the probe is not inert: a delta on the COMPLEMENT (D-visible) set of the
      same cardinality MUST move ``D(f1)``.  Without C3 a broken warp call and
      a genuine null are indistinguishable (memory ``m50``).
  C4  the f0 chain is live: an ALL-pixel unit step must move ``D(f0)`` by
      ~1 LSB (the warp interpolates, so its weights sum to 1 up to ``a``).

axis: [macOS-CPU advisory] NON-PROMOTABLE.  score_claim=false,
promotion_eligible=false, rank_or_kill_eligible=false.  Pointer UNMOVED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SEG_H, SEG_W = 384, 512


def _install_submission_path(sub: Path) -> None:
    for p in (str(sub), str(Path(__file__).resolve().parents[1] / "upstream")):
        if p not in sys.path:
            sys.path.insert(0, p)


def apply_D(frame_hwc_u8: np.ndarray) -> np.ndarray:
    """The EXACT scorer downsample, from ``upstream/modules.py:73`` and ``:109``.

    Both PoseNet.preprocess_input and SegNet.preprocess_input call the identical
    ``F.interpolate(x, size=(384, 512), mode='bilinear')`` -- verified at source
    by ``ddm_pz1`` §0(4).  align_corners and antialias stay at torch defaults.
    """
    arr = np.ascontiguousarray(frame_hwc_u8)
    t = torch.from_numpy(arr).permute(2, 0, 1)[None].float()
    out = torch.nn.functional.interpolate(t, size=(SEG_H, SEG_W), mode="bilinear")
    return out[0].permute(1, 2, 0).numpy().astype(np.float64)


def rgb_to_yuv6_np(rgb_hwc: np.ndarray) -> np.ndarray:
    """``upstream/frame_utils.py:51-78``, transcribed for HWC float64 input."""
    hgt, wid = rgb_hwc.shape[0], rgb_hwc.shape[1]
    h2, w2 = hgt // 2, wid // 2
    rgb = rgb_hwc[: 2 * h2, : 2 * w2, :]
    chan_r, chan_g, chan_b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = np.clip(chan_r * 0.299 + chan_g * 0.587 + chan_b * 0.114, 0.0, 255.0)
    u_pl = np.clip((chan_b - lum) / 1.772 + 128.0, 0.0, 255.0)
    v_pl = np.clip((chan_r - lum) / 1.402 + 128.0, 0.0, 255.0)
    u_sub = (u_pl[0::2, 0::2] + u_pl[1::2, 0::2]
             + u_pl[0::2, 1::2] + u_pl[1::2, 1::2]) * 0.25
    v_sub = (v_pl[0::2, 0::2] + v_pl[1::2, 0::2]
             + v_pl[0::2, 1::2] + v_pl[1::2, 1::2]) * 0.25
    return np.stack([lum[0::2, 0::2], lum[1::2, 0::2],
                     lum[0::2, 1::2], lum[1::2, 1::2], u_sub, v_sub], axis=0)


def stats(delta: np.ndarray) -> dict:
    mag = np.abs(delta)
    return {
        "rms": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
        "max": float(mag.max()),
        "mean_abs": float(mag.mean()),
        "frac_changed": float(np.mean(mag > 1e-12)),
        "frac_over_half_lsb": float(np.mean(mag > 0.5)),
    }


def signed_step(frame: np.ndarray, amp: int = 1) -> np.ndarray:
    """A deterministic in-range step: ``+amp`` on dark pixels, ``-amp`` on bright.

    Never saturates for ``amp <= 128``, so ``clip`` cannot silently swallow the
    perturbation and turn a live actuator into a false null.
    """
    return np.where(frame.astype(np.int16) <= 127, amp, -amp).astype(np.int16)


def perturb(frame: np.ndarray, mask: np.ndarray, step: np.ndarray) -> np.ndarray:
    out = frame.astype(np.int16).copy()
    out[mask] += step[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True, type=Path)
    ap.add_argument("--pairs", type=int, default=6,
                    help="number of STRIDED pairs (never a prefix, per m88)")
    ap.add_argument("--amps", type=int, nargs="+", default=[1],
                    help="blind-step amplitudes in LSB; >1 measures the "
                         "rounding-mediated DYNAMICS law")
    ap.add_argument("--edit-set", choices=("blind", "visible"), default="blind",
                    help="which camera pixels the treatment step writes to. "
                         "'visible' is the D-VISIBLE complement -- the set that "
                         "carries a Q3 isoluminant chroma edit -- and measures "
                         "its warp pass-through directly instead of inferring it")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sub = args.submission_dir.resolve()
    _install_submission_path(sub)

    import ddm_tr1_runtime as shipped  # the VENDORED receiver
    from inflate_runner import Decoder

    from tac.optimization.ddm_ll1_window_solve import blind_mask

    blind = blind_mask()
    n_blind = int(blind.sum())
    visible = ~blind
    # C3's complement set, cardinality-matched to the blind set so the control
    # differs from the treatment ONLY in WHICH pixels move, never how many.
    vis_idx = np.flatnonzero(visible.ravel())
    rng = np.random.default_rng(20260803)
    ctrl_flat = rng.choice(vis_idx, size=n_blind, replace=False)
    ctrl = np.zeros(blind.size, dtype=bool)
    ctrl[ctrl_flat] = True
    ctrl = ctrl.reshape(blind.shape)
    allpx = np.ones_like(blind)
    # The TREATMENT set.  'blind' proves seg-freedom and measures the pose
    # actuator; 'visible' measures the warp pass-through of the complement --
    # the set a Q3 isoluminant chroma edit actually writes to (ph4 §O3).
    treat = blind if args.edit_set == "blind" else visible

    dec = Decoder(sub / "archive")
    n_pairs = int(dec.n_pairs)
    idx = np.unique(np.linspace(0, n_pairs - 1, args.pairs).round().astype(int))

    print(f"[ph4] base={sub.name} n_pairs={n_pairs} strided={list(idx)}", flush=True)
    print(f"[ph4] C1 blind set = {n_blind:,} px ({100.0 * blind.mean():.6f}%) "
          f"| control set = {int(ctrl.sum()):,} px (D-VISIBLE, cardinality-matched)",
          flush=True)
    print(f"[ph4] TREATMENT set = '{args.edit_set}' -> {int(treat.sum()):,} px "
          f"({100.0 * treat.mean():.6f}%)", flush=True)

    rows: list[dict] = []
    c2_ok = True
    c3_ok = True
    c4_ok = True
    seg_exactly_free = True

    amp_rows: list[dict] = []
    for i in idx:
        f1 = np.asarray(shipped.render_frame1_camera_uint8(dec.packet, int(i)))
        d_f1_ref = apply_D(f1)
        f0_ref = dec.f0(int(i), f1)
        d_f0_ref = apply_D(f0_ref)
        # --- the DYNAMICS law: response vs blind-step amplitude -------------
        for amp in sorted({int(a) for a in args.amps}):
            stp = signed_step(f1, amp)
            f1_a = perturb(f1, treat, stp)
            seg_a = float(np.abs(apply_D(f1_a) - d_f1_ref).max())
            seg_exactly_free &= (seg_a == 0.0)
            f0_a = dec.f0(int(i), f1_a)
            cam_a = f0_a.astype(np.float64) - f0_ref.astype(np.float64)
            sco_a = apply_D(f0_a) - d_f0_ref
            amp_rows.append({
                "pair": int(i), "amp": amp,
                "seg_plane_delta_max": seg_a,
                "camera_f0": stats(cam_a),
                "scorer_f0": stats(sco_a),
            })
            print(f"[ph4] amp {amp:4d} pair {i:4d} | SEG {seg_a:.1e} | "
                  f"CAM f0 rms {stats(cam_a)['rms']:.4f} "
                  f"chg {100 * stats(cam_a)['frac_changed']:.1f}% | "
                  f"SCORER f0 rms {stats(sco_a)['rms']:.4f}", flush=True)

        step = signed_step(f1, 1)

        f1_treat = perturb(f1, treat, step)
        f1_ctrl = perturb(f1, ctrl, step)
        f1_all = perturb(f1, allpx, step)
        f1_null = f1.copy()

        d_f1 = apply_D(f1)
        d_f1_treat = apply_D(f1_treat)
        d_f1_ctrl = apply_D(f1_ctrl)

        # --- THE PROOF: blind-only edits are invisible to D, hence to SegNet ---
        seg_delta = d_f1_treat - d_f1
        seg_max = float(np.abs(seg_delta).max())
        seg_exactly_free &= (seg_max == 0.0)

        # --- C3: the same-size D-VISIBLE edit MUST move D(f1) ---
        ctrl_seg_max = float(np.abs(d_f1_ctrl - d_f1).max())
        c3_ok &= (ctrl_seg_max > 0.0)

        # --- the measurement: does the warp carry the blind edit to PoseNet? ---
        f0 = dec.f0(int(i), f1)
        f0_treat = dec.f0(int(i), f1_treat)
        f0_all = dec.f0(int(i), f1_all)
        f0_null = dec.f0(int(i), f1_null)

        cam_f0 = f0_treat.astype(np.float64) - f0.astype(np.float64)
        sco_f0 = apply_D(f0_treat) - apply_D(f0)
        sco_f0_all = apply_D(f0_all) - apply_D(f0)

        yuv_b = np.concatenate([rgb_to_yuv6_np(apply_D(f0_treat)),
                                rgb_to_yuv6_np(d_f1_treat)], axis=0)
        yuv_0 = np.concatenate([rgb_to_yuv6_np(apply_D(f0)),
                                rgb_to_yuv6_np(d_f1)], axis=0)
        yuv_d = yuv_b - yuv_0

        # --- C2: null delta must report exactly 0 through the SAME chain ---
        c2_max = float(np.abs(
            apply_D(f0_null) - apply_D(f0)).max()) + float(np.abs(
                apply_D(f1_null) - d_f1).max())
        c2_ok &= (c2_max == 0.0)

        # --- C4: the all-pixel step must move D(f0) by ~1 LSB ---
        all_rms = float(np.sqrt(np.mean(sco_f0_all ** 2)))
        c4_ok &= (all_rms > 0.1)

        leverage = (float(np.sqrt(np.mean(sco_f0 ** 2))) / all_rms
                    if all_rms > 0 else 0.0)

        row = {
            "pair": int(i),
            "seg_plane_delta_max_ABS_treatment_edit": seg_max,
            "C3_control_visible_edit_seg_plane_max": ctrl_seg_max,
            "C2_null_delta_max": c2_max,
            "camera_f0_delta": stats(cam_f0),
            "scorer_f0_delta": stats(sco_f0),
            "scorer_f0_delta_ALL_PIXEL_STEP": stats(sco_f0_all),
            "yuv6_pair_delta": stats(yuv_d),
            "posenet_normalised_delta": stats(yuv_d / 63.75),
            "treat_share_of_warp_weight": leverage,
        }
        rows.append(row)
        print(
            f"[ph4] pair {i:4d} | SEG delta max {seg_max:.1e} "
            f"(ctrl {ctrl_seg_max:.3f}) | CAM f0 rms "
            f"{row['camera_f0_delta']['rms']:.4f} chg "
            f"{100 * row['camera_f0_delta']['frac_changed']:.1f}% | "
            f"SCORER f0 rms {row['scorer_f0_delta']['rms']:.4f} "
            f"max {row['scorer_f0_delta']['max']:.2f} chg "
            f"{100 * row['scorer_f0_delta']['frac_changed']:.1f}% | "
            f"leverage {leverage:.4f}", flush=True)

    def agg(key: str, field: str) -> float:
        return float(np.mean([r[key][field] for r in rows]))

    summary = {
        "arm": "ddm_ph4",
        "base": sub.name,
        "n_pairs_total": n_pairs,
        "pairs_measured": [int(v) for v in idx],
        "edit_set": args.edit_set,
        "treat_px": int(treat.sum()),
        "treat_frac": float(treat.mean()),
        "blind_px": n_blind,
        "blind_frac": float(blind.mean()),
        "controls": {
            "C1_blind_set_size_matches_230904": n_blind == 230904,
            "C2_probe_returns_exact_zero_on_null": bool(c2_ok),
            "C3_cardinality_matched_visible_edit_moves_D": bool(c3_ok),
            "C4_all_pixel_step_moves_D_f0": bool(c4_ok),
        },
        "SEG_EXACTLY_FREE_all_pairs": bool(seg_exactly_free),
        "camera_f0_delta_rms": agg("camera_f0_delta", "rms"),
        "camera_f0_delta_frac_changed": agg("camera_f0_delta", "frac_changed"),
        "scorer_f0_delta_rms": agg("scorer_f0_delta", "rms"),
        "scorer_f0_delta_max": float(np.max(
            [r["scorer_f0_delta"]["max"] for r in rows])),
        "scorer_f0_delta_frac_changed": agg("scorer_f0_delta", "frac_changed"),
        "scorer_f0_delta_ALL_PIXEL_rms": agg("scorer_f0_delta_ALL_PIXEL_STEP", "rms"),
        "treat_share_of_warp_weight": float(np.mean(
            [r["treat_share_of_warp_weight"] for r in rows])),
        "posenet_normalised_delta_rms": agg("posenet_normalised_delta", "rms"),
        "posenet_normalised_delta_max": float(np.max(
            [r["posenet_normalised_delta"]["max"] for r in rows])),
        "amplitude_sweep": amp_rows,
        "amplitude_law": [
            {
                "amp": a,
                "camera_f0_rms": float(np.mean(
                    [r["camera_f0"]["rms"] for r in amp_rows if r["amp"] == a])),
                "camera_f0_frac_changed": float(np.mean(
                    [r["camera_f0"]["frac_changed"]
                     for r in amp_rows if r["amp"] == a])),
                "scorer_f0_rms": float(np.mean(
                    [r["scorer_f0"]["rms"] for r in amp_rows if r["amp"] == a])),
                "scorer_f0_frac_changed": float(np.mean(
                    [r["scorer_f0"]["frac_changed"]
                     for r in amp_rows if r["amp"] == a])),
                "seg_plane_delta_max": float(np.max(
                    [r["seg_plane_delta_max"]
                     for r in amp_rows if r["amp"] == a])),
            }
            for a in sorted({r["amp"] for r in amp_rows})
        ],
        "rows": rows,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print("\n[ph4] ==== CONTROLS ====", flush=True)
    for key, val in summary["controls"].items():
        print(f"[ph4]   {key}: {val}", flush=True)
    print(f"\n[ph4] SEG EXACTLY FREE on every pair: {seg_exactly_free}", flush=True)
    print(f"[ph4] POSE reach (scorer-plane f0 rms, unit {args.edit_set} step): "
          f"{summary['scorer_f0_delta_rms']:.6f} LSB", flush=True)
    print(f"[ph4] {args.edit_set} share of warp sampling weight: "
          f"{summary['treat_share_of_warp_weight']:.6f} "
          f"(population share {treat.mean():.6f})", flush=True)
    if len(summary["amplitude_law"]) > 1:
        print("[ph4] AMPLITUDE LAW (gain = scorer-plane f0 rms per LSB of step):",
              flush=True)
        for law in summary["amplitude_law"]:
            print(f"[ph4]   amp {law['amp']:4d} -> gain "
                  f"{law['scorer_f0_rms'] / law['amp']:.4f} "
                  f"| seg-plane delta max {law['seg_plane_delta_max']:.4g}",
                  flush=True)
    print(f"[ph4] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
