"""$0 measurement: manifold-aware lane-SDF component — precise + contained? (FEED-dm)

Decisive isolation test for the operator's lane-edge refinement 2026-06-27. Substitute
the level-set class-1 field phi_1 with a STRUCTURED lane SDF (signed-distance-to-
ground-polynomial-band-with-dash, ~7 floats/line) while keeping the OTHER classes'
IDEAL SDFs (from the frozen CPU-torch L*); recompute the argmax partition; decompose
the disagreement vs L* into:
  * lane FN  (true lane missed -> shape fidelity; target ~0.00046 per FEED-dj),
  * class-0 d_seg (road disturbed -> CONTAINMENT; ideal-SDF baseline = 0),
  * other-class d_seg.

Compares 3 variants: ideal (baseline, class0=0), continuous band (no dash), dash-gated
band. The decisive question: does the lane-SDF give the lane to ~0.00046 (PRECISE)
WHILE leaving class-0 ~unchanged (CONTAINED), and is the ground-frame dash gate the
containment knob?

CPU-only, $0. Reads ONLY ``lstars`` from gt_n96.npz (~150 MB, no GPU/MPS, no scorer
fleet). Frozen CPU-torch SegNet argmax (cached, bit-exact). [macOS-CPU advisory]
research-signal: score_claim=false, promotable=false, NOT a byte-closed row.

Usage:
  .venv/bin/python experiments/measure_lane_sdf_containment.py --n 48
  .venv/bin/python experiments/measure_lane_sdf_containment.py --n 8 --r-survival
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tac.boundary_math.lane_sdf_component import (
    build_structured_lane_sdf,
    decompose_argmax_disagreement,
    inject_lane_sdf,
)
from tac.boundary_math.lever_b_levelset_generator import signed_distance_fields

_GT = Path("experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
_N_CLASSES = 5
_LANE = 1
_ROAD = 0


def _mean_decomp(decomps: list) -> dict:
    keys = ["total_dseg", "lane_fn", "lane_fp_from_road", "lane_fp_from_other",
            "class0_dseg", "other_dseg", "lane_attributable"]
    return {k: float(np.mean([getattr(d, k) for d in decomps])) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=48, help="frames (<=96).")
    ap.add_argument("--centerline-deg", type=int, default=3)
    ap.add_argument("--r-survival", action="store_true",
                    help="also measure post-R (round-trip) disagreement on the first --r-n frames.")
    ap.add_argument("--r-n", type=int, default=6)
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    if not _GT.exists():
        raise SystemExit(f"GT cache missing: {_GT}")
    t0 = time.time()
    # Lazy NpzFile: access ONLY lstars (do not touch gt_f0/gt_f1 frames -> keep memory light).
    npz = np.load(_GT)
    lstars = npz["lstars"]  # (96,384,512) int64
    P = min(int(args.n), lstars.shape[0])
    print(f"[lane-sdf $0] n={P} frames; reading lstars only ({lstars.shape}) ...", flush=True)

    rows_cont, rows_dash, rows_ideal = [], [], []
    meta_cont, meta_dash = [], []
    for i in range(P):
        L = np.asarray(lstars[i]).astype(np.int64)
        phi_ideal = signed_distance_fields(L, _N_CLASSES)  # (H,W,K); argmax == L exactly
        # baseline sanity (ideal): should be 0 everywhere
        pred_ideal = phi_ideal.argmax(-1)
        rows_ideal.append(decompose_argmax_disagreement(pred_ideal, L, lane_cls=_LANE, road_cls=_ROAD))

        # continuous band (no dash gate) -> exposes the dash-gap leakage into class-0
        phi1_c, mc = build_structured_lane_sdf(L, lane_cls=_LANE, dash_gate=False,
                                               centerline_deg=args.centerline_deg)
        pred_c = inject_lane_sdf(phi_ideal, phi1_c, lane_cls=_LANE, mode="replace").argmax(-1)
        rows_cont.append(decompose_argmax_disagreement(pred_c, L, lane_cls=_LANE, road_cls=_ROAD))
        meta_cont.append(mc)

        # dash-gated band (2-param ground-frame dash) -> the containment knob
        phi1_d, md = build_structured_lane_sdf(L, lane_cls=_LANE, dash_gate=True,
                                               centerline_deg=args.centerline_deg)
        pred_d = inject_lane_sdf(phi_ideal, phi1_d, lane_cls=_LANE, mode="replace").argmax(-1)
        rows_dash.append(decompose_argmax_disagreement(pred_d, L, lane_cls=_LANE, road_cls=_ROAD))
        meta_dash.append(md)
        if (i + 1) % 12 == 0:
            print(f"  ... {i+1}/{P}", flush=True)

    ideal = _mean_decomp(rows_ideal)
    cont = _mean_decomp(rows_cont)
    dash = _mean_decomp(rows_dash)
    floats_c = float(np.mean([m["total_floats"] for m in meta_cont]))
    floats_d = float(np.mean([m["total_floats"] for m in meta_dash]))
    lines_c = float(np.mean([m["n_lines"] for m in meta_cont]))
    dash_modeled = float(np.mean([m["n_dash_modeled"] for m in meta_dash]))

    def _pr(tag, d):
        print(f"\n=== {tag} ===")
        print(f"  total_dseg          {d['total_dseg']:.6f}")
        print(f"  lane_attributable   {d['lane_attributable']:.6f}")
        print(f"    lane_fn (shape)   {d['lane_fn']:.6f}   (target <= 0.00087; FEED-dj 0.00046)")
        print(f"    lane_fp<-road     {d['lane_fp_from_road']:.6f}   (dash-gap leak INTO class-0)")
        print(f"    lane_fp<-other    {d['lane_fp_from_other']:.6f}")
        print(f"  class0_dseg         {d['class0_dseg']:.6f}   (CONTAINMENT; ideal baseline 0)")
        print(f"  other_dseg          {d['other_dseg']:.6f}")

    _pr("IDEAL SDF (baseline; argmax==L*)", ideal)
    _pr("CONTINUOUS band (no dash)", cont)
    _pr("DASH-GATED band (2-param ground dash)", dash)
    print(f"\n[manifold] lines/frame ~{lines_c:.1f}; floats/frame cont~{floats_c:.0f} "
          f"dash~{floats_d:.0f}; dash-modeled lines/frame ~{dash_modeled:.1f}")
    print(f"[containment delta vs ideal] class0: continuous {cont['class0_dseg']:.6f} -> "
          f"dash {dash['class0_dseg']:.6f}  (recovered {cont['class0_dseg']-dash['class0_dseg']:.6f})")

    out = {
        "n": P, "centerline_deg": args.centerline_deg,
        "ideal": ideal, "continuous": cont, "dash": dash,
        "floats_per_frame_cont": floats_c, "floats_per_frame_dash": floats_d,
        "lines_per_frame": lines_c, "dash_modeled_lines_per_frame": dash_modeled,
        "authority": "macOS-CPU advisory", "score_claim": False, "promotable": False,
        "byte_closed_row": False, "elapsed_s": round(time.time() - t0, 1),
    }

    if args.r_survival:
        out["r_survival"] = _measure_r_survival(lstars, min(int(args.r_n), P), args.centerline_deg)

    if args.json_out:
        jp = Path(args.json_out)
        if "/tmp" in str(jp):
            raise SystemExit("refuse /tmp json-out (durable evidence rule)")
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(out, indent=2))
        print(f"\n[json] {jp}")
    print(f"\n[done] {out['elapsed_s']}s  [macOS-CPU advisory] research-signal; pointer UNMOVED 0.19110")


def _measure_r_survival(lstars, rn: int, deg: int) -> dict:
    """post-R disagreement: does the structured lane SDF survive R as well as ideal?"""

    from tac.boundary_math.lever_b_levelset_generator import apply_R_to_fields

    print(f"\n[R-survival] n={rn} (bicubic up -> uint8 @camera -> bilinear down) ...", flush=True)
    ideal_post, cont_post, dash_post = [], [], []
    for i in range(rn):
        L = np.asarray(lstars[i]).astype(np.int64)
        phi_ideal = signed_distance_fields(L, _N_CLASSES)
        phi1_c, _ = build_structured_lane_sdf(L, lane_cls=_LANE, dash_gate=False, centerline_deg=deg)
        phi1_d, _ = build_structured_lane_sdf(L, lane_cls=_LANE, dash_gate=True, centerline_deg=deg)
        phi_cont = inject_lane_sdf(phi_ideal, phi1_c, lane_cls=_LANE, mode="replace")
        phi_dash = inject_lane_sdf(phi_ideal, phi1_d, lane_cls=_LANE, mode="replace")
        # L* is 384x512 == scorer res; apply_R returns scorer res -> compare argmax directly
        ideal_post.append(float(np.mean(apply_R_to_fields(phi_ideal).argmax(-1) != L)))
        cont_post.append(float(np.mean(apply_R_to_fields(phi_cont).argmax(-1) != L)))
        dash_post.append(float(np.mean(apply_R_to_fields(phi_dash).argmax(-1) != L)))
    res = {"ideal_post_R_dseg": float(np.mean(ideal_post)),
           "continuous_post_R_dseg": float(np.mean(cont_post)),
           "dash_post_R_dseg": float(np.mean(dash_post)), "r_n": rn}
    print(f"  post-R d_seg: ideal {res['ideal_post_R_dseg']:.6f}  "
          f"continuous {res['continuous_post_R_dseg']:.6f}  dash-struct {res['dash_post_R_dseg']:.6f}")
    return res


if __name__ == "__main__":
    main()
