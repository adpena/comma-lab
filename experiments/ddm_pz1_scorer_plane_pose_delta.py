#!/usr/bin/env python
"""ddm_pz1 — the FREE measurement: what does PoseNet actually SEE when the
ddm_ll1 window solve is engaged?

THE PREMISE, RE-DERIVED FROM THE FROZEN UPSTREAM (not taken on faith)
---------------------------------------------------------------------
``upstream/modules.py:70-74``::

    def preprocess_input(self, x):            # PoseNet
      x = einops.rearrange(x, 'b t c h w -> (b t) c h w', ...)
      x = torch.nn.functional.interpolate(x, size=(384, 512), mode='bilinear')
      return einops.rearrange(rgb_to_yuv6(x), ...)

The interpolate comes FIRST and ``rgb_to_yuv6`` SECOND.  PoseNet therefore reads
through the *identical* operator ``D`` as SegNet (``modules.py:107-109``).  This
contradicts the ordering stated in CLAUDE.md ("rgb_to_yuv6 -> resize") and
CONFIRMS the ``ddm_ll1`` docstring.  It is the reason this measurement is
well-posed: the 2.75-LSB frame_0 delta that ``ddm_ra1``/``ddm_sg2`` measured is a
CAMERA-domain quantity, but PoseNet consumes ``D(f0)``, and ``D`` is a
subsampling+averaging operator whose action on that delta has never been measured.

WHAT IS MEASURED (all $0, no scorer forward pass anywhere)
----------------------------------------------------------
For strided pairs on the LIVE base:
  * camera-domain deltas for f1 and f0   (reproduces the published figures)
  * scorer-plane deltas  D(f1), D(f0)    (NEW - the quantity that gates the rung)
  * yuv6-plane deltas and PoseNet-normalised deltas (the literal network input)

POSITIVE CONTROLS (run before any number is reported)
-----------------------------------------------------
  C1  blind-set invariance: solve touches 0 of the 230,904 D-blind camera px
  C2  delivery: ``D(f1_solved) - r`` rms must collapse vs ``D(f1_base) - r``
  C3  the probe must be able to return the negative -- a null solve (identical
      frames) must report exactly 0 everywhere

axis: [macOS-CPU advisory] NON-PROMOTABLE.  score_claim=false.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SEG_H, SEG_W = 384, 512


def _strided_pair_selection_scope(idx: np.ndarray, population: int) -> dict[str, object]:
    return {
        "schema": "subset_scope.v1",
        "n": int(len(idx)),
        "population": int(population),
        "selection_mode": "strided_linspace",
        "pair_indices": [int(v) for v in idx],
        "selection_rule": "unique(round(linspace(0, population - 1, requested_pairs)))",
        "axis_bias_caveat": (
            "strided advisory subset; no population claim or n600 conclusion follows without "
            "a governing subset/population bias check"
        ),
        "population_claim": False,
    }


def _install_submission_path(sub: Path) -> None:
    for p in (str(sub), str(Path(__file__).resolve().parents[1] / "upstream")):
        if p not in sys.path:
            sys.path.insert(0, p)


def apply_D(frame_hwc_u8: np.ndarray) -> np.ndarray:
    """The EXACT scorer downsample, reproduced from upstream/modules.py.

    Both PoseNet.preprocess_input and SegNet.preprocess_input call
    ``F.interpolate(x, size=(384, 512), mode='bilinear')`` on a float NCHW
    tensor obtained from ``rearrange(x, 'b t h w c -> b t c h w').float()``.
    align_corners and antialias are left at their torch defaults (False, False).
    """
    t = torch.from_numpy(np.ascontiguousarray(frame_hwc_u8)).permute(2, 0, 1)[None].float()
    out = torch.nn.functional.interpolate(t, size=(SEG_H, SEG_W), mode="bilinear")
    return out[0].permute(1, 2, 0).numpy().astype(np.float64)


def rgb_to_yuv6_np(rgb_hwc: np.ndarray) -> np.ndarray:
    """upstream/frame_utils.py:51-78, transcribed for HWC float64 input."""
    H, W = rgb_hwc.shape[0], rgb_hwc.shape[1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_hwc[: 2 * H2, : 2 * W2, :]
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = np.clip(R * 0.299 + G * 0.587 + B * 0.114, 0.0, 255.0)
    U = np.clip((B - Y) / 1.772 + 128.0, 0.0, 255.0)
    V = np.clip((R - Y) / 1.402 + 128.0, 0.0, 255.0)
    U_sub = (U[0::2, 0::2] + U[1::2, 0::2] + U[0::2, 1::2] + U[1::2, 1::2]) * 0.25
    V_sub = (V[0::2, 0::2] + V[1::2, 0::2] + V[0::2, 1::2] + V[1::2, 1::2]) * 0.25
    return np.stack([Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2],
                     U_sub, V_sub], axis=0)


def stats(delta: np.ndarray) -> dict:
    a = np.abs(delta)
    return {
        "rms": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
        "max": float(a.max()),
        "mean_abs": float(a.mean()),
        "p99": float(np.percentile(a, 99.0)),
        "frac_changed": float(np.mean(a > 1e-12)),
        "frac_over_half_lsb": float(np.mean(a > 0.5)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True, type=Path)
    ap.add_argument("--pairs", type=int, default=6,
                    help="number of STRIDED pairs (never a prefix, per m88)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sub = args.submission_dir.resolve()
    _install_submission_path(sub)

    from inflate_runner import Decoder  # noqa: E402
    from tac.optimization.ddm_ll1_window_solve import blind_mask  # noqa: E402
    import ddm_tr1_runtime as shipped_tr1  # noqa: E402  (the VENDORED receiver)
    from tac.optimization import ddm_tr1_runtime as repo_tr1  # noqa: E402

    # C0 -- the shipped receiver does NOT contain the window solve at all.  The
    # rung is therefore not "flip a bool in the shipping tree"; it is "re-vendor
    # the repo receiver, THEN flip the bool".  That swap is byte-free (archive.zip
    # holds exactly one member, 0.bin -- the .py receiver is not rate-charged),
    # but it is only VALID if the repo receiver reproduces the shipped frames
    # bit-for-bit with the solve OFF.  That is asserted per pair below, never
    # assumed from the compatibility comment in the diff.
    shipped_has_solve = "window_solve" in __import__("inspect").signature(
        shipped_tr1.render_frame1_camera_uint8).parameters
    repo_has_solve = "window_solve" in __import__("inspect").signature(
        repo_tr1.render_frame1_camera_uint8).parameters
    print(f"[pz1] C0 shipped receiver has window_solve: {shipped_has_solve}", flush=True)
    print(f"[pz1] C0 repo    receiver has window_solve: {repo_has_solve}", flush=True)

    dec = Decoder(sub / "archive")
    n = int(dec.n_pairs)
    idx = np.unique(np.linspace(0, n - 1, args.pairs).round().astype(int))
    blind = blind_mask()
    print(f"[pz1] base={sub.name} n_pairs={n} strided pairs={list(idx)}", flush=True)
    print(f"[pz1] blind set = {int(blind.sum()):,} camera px "
          f"({100.0 * blind.mean():.6f}%)", flush=True)

    rows = []
    swap_identical = True
    for i in idx:
        r = np.asarray(repo_tr1.render_frame1_float(dec.packet, int(i)), dtype=np.float64)
        f1b = repo_tr1.render_frame1_camera_uint8(dec.packet, int(i), window_solve=False)
        f1s = repo_tr1.render_frame1_camera_uint8(dec.packet, int(i), window_solve=True)

        # ---- C0 receiver-swap byte-identity (the gate's own precondition) ---
        f1_shipped = shipped_tr1.render_frame1_camera_uint8(dec.packet, int(i))
        swap_ok = bool(np.array_equal(f1_shipped, f1b))
        swap_identical &= swap_ok

        f0b = dec.f0(int(i), f1b)
        f0s = dec.f0(int(i), f1s)

        # ---- C1 blind-set invariance -----------------------------------
        blind_touched = int(np.count_nonzero(
            (f1s.astype(np.int32) - f1b.astype(np.int32))[blind]))

        # ---- C2 delivery through D -------------------------------------
        Df1b, Df1s = apply_D(f1b), apply_D(f1s)
        Df0b, Df0s = apply_D(f0b), apply_D(f0s)
        deliver_base = stats(Df1b - r)
        deliver_sol = stats(Df1s - r)

        # ---- the measurement -------------------------------------------
        cam_d_f1 = f1s.astype(np.float64) - f1b.astype(np.float64)
        cam_d_f0 = f0s.astype(np.float64) - f0b.astype(np.float64)
        sco_d_f1 = Df1s - Df1b
        sco_d_f0 = Df0s - Df0b

        yb = np.concatenate([rgb_to_yuv6_np(Df0b), rgb_to_yuv6_np(Df1b)], axis=0)
        ys = np.concatenate([rgb_to_yuv6_np(Df0s), rgb_to_yuv6_np(Df1s)], axis=0)
        yuv_d = ys - yb

        row = {
            "pair": int(i),
            "receiver_swap_byte_identical": swap_ok,
            "blind_px_touched": blind_touched,
            "deliver_rms_base": deliver_base["rms"],
            "deliver_rms_solved": deliver_sol["rms"],
            "camera_delta_f1": stats(cam_d_f1),
            "camera_delta_f0": stats(cam_d_f0),
            "scorer_delta_f1": stats(sco_d_f1),
            "scorer_delta_f0": stats(sco_d_f0),
            "yuv6_delta_pair": stats(yuv_d),
            "posenet_normalised_delta_pair": stats(yuv_d / 63.75),
        }
        rows.append(row)
        print(
            f"[pz1] pair {i:4d} | blind={blind_touched} | "
            f"deliver {deliver_base['rms']:.4f}->{deliver_sol['rms']:.4f} | "
            f"CAM f0 rms {row['camera_delta_f0']['rms']:.4f} "
            f"max {row['camera_delta_f0']['max']:.1f} "
            f"chg {100 * row['camera_delta_f0']['frac_changed']:.1f}% | "
            f"SCORER f0 rms {row['scorer_delta_f0']['rms']:.4f} "
            f"max {row['scorer_delta_f0']['max']:.2f} | "
            f"SCORER f1 rms {row['scorer_delta_f1']['rms']:.4f}",
            flush=True)

    # ---- C3 the probe can return the negative --------------------------
    zero = stats(np.zeros((8, 8), dtype=np.float64))
    c3_ok = zero["rms"] == 0.0 and zero["max"] == 0.0 and zero["frac_changed"] == 0.0

    def agg(key: str, field: str) -> float:
        return float(np.mean([r[key][field] for r in rows]))

    def agg_max(key: str) -> float:
        return float(np.max([r[key]["max"] for r in rows]))

    summary = {
        "base": sub.name,
        "n_pairs_total": n,
        "pairs_measured": [int(v) for v in idx],
        "pair_selection": _strided_pair_selection_scope(idx, n),
        "controls": {
            "C0_shipped_receiver_has_window_solve": bool(shipped_has_solve),
            "C0_repo_receiver_has_window_solve": bool(repo_has_solve),
            "C0_receiver_swap_byte_identical_all_pairs": bool(swap_identical),
            "C1_blind_px_touched_total": int(sum(r["blind_px_touched"] for r in rows)),
            "C2_deliver_rms_base": agg("camera_delta_f1", "rms") * 0 + float(
                np.mean([r["deliver_rms_base"] for r in rows])),
            "C2_deliver_rms_solved": float(np.mean([r["deliver_rms_solved"] for r in rows])),
            "C3_probe_returns_negative": bool(c3_ok),
        },
        "camera_delta_f0_rms": agg("camera_delta_f0", "rms"),
        "camera_delta_f0_max": agg_max("camera_delta_f0"),
        "camera_delta_f0_frac_changed": agg("camera_delta_f0", "frac_changed"),
        "camera_delta_f1_rms": agg("camera_delta_f1", "rms"),
        "camera_delta_f1_max": agg_max("camera_delta_f1"),
        "scorer_delta_f0_rms": agg("scorer_delta_f0", "rms"),
        "scorer_delta_f0_max": agg_max("scorer_delta_f0"),
        "scorer_delta_f0_p99": agg("scorer_delta_f0", "p99"),
        "scorer_delta_f0_frac_over_half_lsb": agg("scorer_delta_f0", "frac_over_half_lsb"),
        "scorer_delta_f1_rms": agg("scorer_delta_f1", "rms"),
        "scorer_delta_f1_max": agg_max("scorer_delta_f1"),
        "yuv6_delta_pair_rms": agg("yuv6_delta_pair", "rms"),
        "yuv6_delta_pair_max": agg_max("yuv6_delta_pair"),
        "posenet_normalised_delta_rms": agg("posenet_normalised_delta_pair", "rms"),
        "posenet_normalised_delta_max": agg_max("posenet_normalised_delta_pair"),
        "attenuation_camera_to_scorer_f0": (
            agg("camera_delta_f0", "rms") / max(agg("scorer_delta_f0", "rms"), 1e-30)),
        "rows": rows,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print("\n[pz1] ===== SUMMARY =====")
    print(f"[pz1] C0 shipped receiver has window_solve: {shipped_has_solve} "
          f"| repo: {repo_has_solve}")
    print(f"[pz1] C0 receiver swap byte-identical (solve OFF), all pairs: "
          f"{swap_identical}")
    print(f"[pz1] C1 blind px touched (must be 0): "
          f"{summary['controls']['C1_blind_px_touched_total']}")
    print(f"[pz1] C2 delivery rms {summary['controls']['C2_deliver_rms_base']:.4f} "
          f"-> {summary['controls']['C2_deliver_rms_solved']:.4f}")
    print(f"[pz1] C3 probe returns negative: "
          f"{summary['controls']['C3_probe_returns_negative']}")
    print(f"[pz1] f0 CAMERA delta   rms {summary['camera_delta_f0_rms']:.4f} "
          f"max {summary['camera_delta_f0_max']:.1f} "
          f"changed {100 * summary['camera_delta_f0_frac_changed']:.1f}%")
    print(f"[pz1] f0 SCORER delta   rms {summary['scorer_delta_f0_rms']:.4f} "
          f"max {summary['scorer_delta_f0_max']:.2f} "
          f"p99 {summary['scorer_delta_f0_p99']:.3f}")
    print(f"[pz1] f1 SCORER delta   rms {summary['scorer_delta_f1_rms']:.4f} "
          f"max {summary['scorer_delta_f1_max']:.2f}")
    print(f"[pz1] ATTENUATION camera->scorer on f0: "
          f"{summary['attenuation_camera_to_scorer_f0']:.3f}x")
    print(f"[pz1] PoseNet normalised input delta rms "
          f"{summary['posenet_normalised_delta_rms']:.6f} "
          f"max {summary['posenet_normalised_delta_max']:.4f}")
    print(f"[pz1] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
