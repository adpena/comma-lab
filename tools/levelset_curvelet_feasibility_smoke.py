# SPDX-License-Identifier: MIT
"""$0 CPU feasibility: softmax-of-SDF + curvelet vs sine/spectral on REAL L* partitions.

The DECISIVE deep-math measure (NO GPU, NO training): on real SegNet L* partitions, does the
softmax-of-SDF representation R-SURVIVE better than the spectral (sine/Fourier-feature) one,
and does the GENERIC curvelet front-end represent the partition better than the isotropic
Fourier basis at EQUAL feature budget?

  (1) R-SURVIVAL: build SDF fields (1-Lipschitz) AND spectral low-pass fields (Gibbs) that both
      reproduce L* pre-R; apply the contest R (bicubic up to camera -> uint8 @ camera ->
      bilinear down to 384x512); report pre/post-R argmax disagreement, R-ADDED flips, and the
      OFF-boundary fraction of those R-added flips (Gibbs aliasing = off-boundary; SDF =
      boundary-local monotone shift).
  (2) FRONT-END: least-squares fit the per-class SDF target with the curvelet bank vs the
      isotropic Fourier basis at EQUAL feature count; report argmax-fit disagreement.

VERDICT axis: [macOS-CPU advisory] field-level R-survival PROXY (NOT the SegNet-authority
realized d_seg — that needs the training arm; a flat palette through an untrained SegNet is
not informative). promotion_eligible=False. Authority = the frozen CPU-torch SegNet for L*.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.boundary_math.lever_b_levelset_generator import (  # noqa: E402
    CurveletBankConfig,
    build_coords,
    curvelet_directional_B,
    front_end_fit_disagreement,
    isotropic_fourier_B,
    r_survival_flip_fraction,
    signed_distance_fields,
    spectral_lowpass_fields,
)


def _agg(stats_list):
    return {
        "pre_R_disagree": float(np.mean([s.pre_R_disagree for s in stats_list])),
        "post_R_disagree": float(np.mean([s.post_R_disagree for s in stats_list])),
        "r_added_flips": float(np.mean([s.r_added_flips for s in stats_list])),
        "r_added_off_boundary_frac": float(np.mean([s.r_added_off_boundary_frac for s in stats_list])),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="level-set + curvelet $0 CPU feasibility")
    ap.add_argument("--num-pairs", type=int, default=6)
    ap.add_argument("--n-classes", type=int, default=5)
    ap.add_argument("--cutoff-frac", type=float, default=0.12,
                    help="spectral control: radial FFT cutoff (fraction of Nyquist).")
    ap.add_argument("--out-json", type=Path,
                    default=REPO / ".omx" / "research" / "levelset_curvelet_feasibility.json")
    args = ap.parse_args(argv)

    from tac.boundary_math.seg_core import (
        decode_gt_frame1_pairs,
        load_real_segnet,
        segnet_argmax_and_margin,
    )

    seg = load_real_segnet("cpu")
    lstars: list[np.ndarray] = []
    for _idx, _f0, f1 in decode_gt_frame1_pairs(n_pairs=args.num_pairs):
        lstar, _m = segnet_argmax_and_margin(seg, np.asarray(f1))
        lstars.append(np.asarray(lstar).astype(np.int64))
    print(json.dumps({"stage": "L*_extracted", "n_pairs": len(lstars),
                      "shape": list(lstars[0].shape)}), flush=True)

    sdf_stats, spec_stats = [], []
    curv_fit, iso_fit = [], []
    # EQUAL-budget bases: curvelet bank vs isotropic Gaussian Fourier of the SAME column count.
    bank = CurveletBankConfig(n_scales=4, n_orient0=6, f0=2.0, base=2.0, n_iso=4)
    B_curv = curvelet_directional_B(bank)
    n_feats = B_curv.shape[1]
    B_iso = isotropic_fourier_B(n_feats, sigma=8.0)
    print(json.dumps({"stage": "bank", "curvelet_cols": int(n_feats),
                      "isotropic_cols": int(B_iso.shape[1])}), flush=True)

    for i, lstar in enumerate(lstars):
        h, w = lstar.shape
        sdf = signed_distance_fields(lstar, args.n_classes)
        spec = spectral_lowpass_fields(lstar, args.n_classes, cutoff_frac=args.cutoff_frac)
        sdf_stats.append(r_survival_flip_fraction(sdf, lstar))
        spec_stats.append(r_survival_flip_fraction(spec, lstar))
        coords = build_coords(h, w)
        curv_fit.append(front_end_fit_disagreement(coords, sdf, B_curv, h, w))
        iso_fit.append(front_end_fit_disagreement(coords, sdf, B_iso, h, w))
        print(json.dumps({"stage": "pair", "i": i,
                          "sdf_postR": round(sdf_stats[-1].post_R_disagree, 6),
                          "spec_postR": round(spec_stats[-1].post_R_disagree, 6),
                          "curv_fit": round(curv_fit[-1], 6),
                          "iso_fit": round(iso_fit[-1], 6)}), flush=True)

    sdf_agg = _agg(sdf_stats)
    spec_agg = _agg(spec_stats)
    result = {
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "axis": "[macOS-CPU advisory] field-level R-survival proxy; promotion_eligible=false",
        "n_pairs": len(lstars),
        "r_survival": {"sdf_levelset": sdf_agg, "spectral_sine": spec_agg},
        "front_end_fit_argmax_disagree": {
            "curvelet": float(np.mean(curv_fit)),
            "isotropic": float(np.mean(iso_fit)),
            "curvelet_vs_isotropic_ratio": float(np.mean(curv_fit) / max(np.mean(iso_fit), 1e-9)),
        },
        "verdict": {
            "sdf_postR_vs_spectral_postR_ratio": sdf_agg["post_R_disagree"] / max(spec_agg["post_R_disagree"], 1e-9),
            "sdf_r_added_vs_spectral_r_added_ratio": sdf_agg["r_added_flips"] / max(spec_agg["r_added_flips"], 1e-9),
            "sdf_off_boundary_frac": sdf_agg["r_added_off_boundary_frac"],
            "spectral_off_boundary_frac": spec_agg["r_added_off_boundary_frac"],
        },
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("\n=== LEVEL-SET + CURVELET FEASIBILITY (field-level R-survival proxy) ===")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
