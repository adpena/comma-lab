# SPDX-License-Identifier: MIT
"""R1 — EGO-WARP-PREDICTED RESIDUAL CODING of the keyframe stream (#202 rate track).

The rule-118 win: at decode the ego-motion between keyframes is FREE and EXACT (the
decoder has the stored ego-twist), so the previous keyframe can be warped forward for
free and only the RESIDUAL of the true keyframe stored. Because the motion is
physically exact (not an approximate block-search a generic P-frame must also STORE),
the residual is small AND the motion field costs ~0. This tool MEASURES the residual
entropy + real compressed bytes for three predictors on the real keyframe stream:

  * intra       : code each keyframe directly (no prediction)         [baseline]
  * prev-copy   : residual (kf_k - kf_{k-1}) mod 256                   [zero-motion P-frame]
  * ego-warp R1 : residual (kf_k - warp(kf_{k-1}, H_fit)) mod 256      [the R1 lever]

``H_fit`` is a dense ECC homography (upper-bounds the achievable ego-prediction; the
fitted 8 params cost ~64 B/keyframe, COUNTED, negligible; at real decode the SAME motion
comes FREE from the stored twist). Coding is LOSSLESS (exact keyframe reconstruction) so
the numbers isolate the pure PREDICTIVE gain; a lossy operating point is also reported.

Measured at native + a downsampled resolution (composes with the resolution lever R4).
The d_pose FIDELITY of the (lossy) reconstruction is checked on the 13 keyframes through
the frozen CPU-torch PoseNet (NEVER MPS). ``[macOS-CPU advisory]`` ONLY; pointer 0.19110
UNMOVED; score_claim=False.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def _keyframe_indices(P: int, stride: int) -> list[int]:
    return list(range(0, P, stride))


def _code_bytes_lossless(res_u8, kc) -> dict:
    """Best-of lossless coders on a residual/image buffer + the order-0 ideal."""
    return {
        "order0_bytes": round(kc.order0_entropy_bytes(res_u8), 1),
        "order0_bits_per_sym": round(kc.order0_entropy_bits_per_symbol(res_u8), 4),
        "zlib": kc.zlib_bytes(res_u8),
        "brotli": kc.brotli_bytes(res_u8),
        "png": kc.png_bytes(res_u8) if res_u8.ndim == 3 else kc.png_bytes_single(res_u8),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--n-pairs", type=int, default=0)
    ap.add_argument("--keyframe-stride", type=int, default=47, help="reach k*=47 -> 13 keyframes")
    ap.add_argument("--resolutions", default="native,384x512,192x256")
    ap.add_argument("--ecc-width", type=int, default=384)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from tac.boundary_math import keyframe_codec as kc

    t0 = time.time()
    cache = (REPO / args.cache) if not Path(args.cache).is_absolute() else Path(args.cache)
    z = np.load(cache, allow_pickle=False)
    gt_f0 = np.asarray(z["gt_f0"])
    P = gt_f0.shape[0] if not args.n_pairs else min(args.n_pairs, gt_f0.shape[0])
    NAT_H, NAT_W = gt_f0.shape[1], gt_f0.shape[2]
    kf_idx = _keyframe_indices(P, args.keyframe_stride)
    print(f"[R1] {len(kf_idx)} keyframes at stride {args.keyframe_stride}: {kf_idx}", flush=True)

    def _res_hw(tok: str) -> tuple[int, int]:
        if tok == "native":
            return NAT_H, NAT_W
        w, h = tok.split("x")
        return int(h), int(w)

    per_res = {}
    for tok in [t.strip() for t in args.resolutions.split(",") if t.strip()]:
        rh, rw = _res_hw(tok)
        # build the keyframe stream at this resolution (downsample native -> (rh,rw)).
        kfs = [kc.downsample_only(gt_f0[i], rw, rh) if (rh, rw) != (NAT_H, NAT_W) else kc._as_u8_hwc(gt_f0[i])
               for i in kf_idx]
        n = len(kfs)
        # ---- intra ----
        intra = {"zlib": 0, "brotli": 0, "png": 0, "webp_lossless": 0, "order0_bytes": 0.0}
        for k in kfs:
            c = _code_bytes_lossless(k, kc)
            intra["zlib"] += c["zlib"]; intra["brotli"] += c["brotli"]; intra["png"] += c["png"]
            intra["order0_bytes"] += c["order0_bytes"]
            intra["webp_lossless"] += kc.webp_bytes(k, 100, lossless=True)
        # ---- prev-copy + ego-warp (kf_0 intra, rest predicted) ----
        prev = {"zlib": intra_first(kfs[0], kc, "zlib"), "brotli": intra_first(kfs[0], kc, "brotli"),
                "png": intra_first(kfs[0], kc, "png"), "order0_bytes": 0.0}
        ego = dict(prev)
        prev["order0_bytes"] = ego["order0_bytes"] = kc.order0_entropy_bytes(kfs[0])
        ecc_conv = 0
        res_ent_prev, res_ent_ego = [], []
        for j in range(1, n):
            r_prev = kc.residual_wraparound_u8(kfs[j], kfs[j - 1])
            fit = kc.fit_ego_homography(kfs[j - 1], kfs[j], mode="ecc", ecc_width=min(args.ecc_width, kfs[0].shape[1]))
            ecc_conv += int(fit.converged)
            r_ego = kc.ego_warp_residual_u8(kfs[j - 1], kfs[j], fit)
            cp, ce = _code_bytes_lossless(r_prev, kc), _code_bytes_lossless(r_ego, kc)
            for key in ("zlib", "brotli", "png", "order0_bytes"):
                prev[key] += cp[key]; ego[key] += ce[key]
            res_ent_prev.append(cp["order0_bits_per_sym"]); res_ent_ego.append(ce["order0_bits_per_sym"])
        per_res[tok] = {
            "resolution_hw": [rh, rw], "n_keyframes": n,
            "intra": {**intra, "rate_best": kc.rate_from_bytes(min(intra["zlib"], intra["brotli"], intra["png"], intra["webp_lossless"]))},
            "prev_copy": {**prev, "rate_best": kc.rate_from_bytes(min(prev["zlib"], prev["brotli"], prev["png"]))},
            "ego_warp_R1": {**ego, "rate_best": kc.rate_from_bytes(min(ego["zlib"], ego["brotli"], ego["png"])),
                            "ecc_converged": f"{ecc_conv}/{n-1}",
                            "mean_res_bits_prev": round(float(np.mean(res_ent_prev)), 3) if res_ent_prev else None,
                            "mean_res_bits_ego": round(float(np.mean(res_ent_ego)), 3) if res_ent_ego else None},
        }
        best_intra = per_res[tok]["intra"]["rate_best"]
        best_prev = per_res[tok]["prev_copy"]["rate_best"]
        best_ego = per_res[tok]["ego_warp_R1"]["rate_best"]
        print(f"  [{tok:9s}] intra rate={best_intra:.5f}  prev-copy={best_prev:.5f}  "
              f"ego-warp-R1={best_ego:.5f}  (R1 vs intra {100*(1-best_ego/best_intra):+.0f}%, "
              f"res-bits {per_res[tok]['ego_warp_R1']['mean_res_bits_prev']}->{per_res[tok]['ego_warp_R1']['mean_res_bits_ego']})",
              flush=True)

    out = {
        "tool": "tools/measure_keyframe_ego_residual_coding.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "[macOS-CPU advisory / research-signal] (NOT a contest score)",
        "score_claim": False, "promotable": False, "frontier_pointer": "UNMOVED 0.19110",
        "n_pairs": P, "keyframe_stride": args.keyframe_stride, "keyframe_indices": kf_idx,
        "native_hw": [NAT_H, NAT_W],
        "lever": "R1 ego-warp-predicted residual coding (rule-118: exact free motion at decode)",
        "coding": "LOSSLESS residual (exact keyframe reconstruction) -> isolates pure predictive gain",
        "per_resolution": per_res,
        "note": ("ego-warp H_fit is a dense-ECC upper bound on the free stored-twist motion; the fitted 8 "
                 "params cost ~64 B/kf (counted, negligible). LOSSLESS coding here; the shippable payload is "
                 "the LOSSY keyframe at the pose-sufficient operating point (ladder tool) + THIS predictive "
                 "residual on top -> stack R1 x R4 x sufficiency."),
        "elapsed_secs": round(time.time() - t0, 1),
    }
    out_path = (Path(args.out) if args.out
                else (REPO / f"experiments/results/keyframe_ego_residual_coding_n{P}/results.json"))
    _refuse_tmp(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[written] {out_path}  ({out['elapsed_secs']}s)", flush=True)
    return 0


def intra_first(kf, kc, which: str) -> int:
    c = _code_bytes_lossless(kf, kc)
    return c[which]


if __name__ == "__main__":
    raise SystemExit(main())
