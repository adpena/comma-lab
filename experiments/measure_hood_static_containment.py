"""$0 measurement: static ego-hood SDF component — precise + contained? (FEED-du)

The NEXT per-lever optimal-form treatment after the lane-SDF (FEED-dm), applying the
VALIDATED lane-SDF isolation template to the EGO-HOOD class. Substitute the level-set
hood field phi_hood with a SINGLE frame-shared static-mask SDF while keeping the OTHER
classes' IDEAL per-frame SDFs (from the frozen CPU-torch L*); recompute the argmax
partition; decompose the disagreement vs L* into:
  * hood FN   (true hood missed -> the per-frame hood beyond the static mask; PRECISE?),
  * containment leak (non-hood -> hood; ideal-SDF baseline = 0; CONTAINED?),
  * other-class d_seg.

Compares: ideal (baseline 0), majority-vote static mask (recommended), intersection
static mask (max containment / more FN — the analog of the lane dash-gate over-gating).

NO-FAKE: the static ego-hood class is DETECTED from the data (``identify_static_hood_class``)
not hardcoded — it is class 4 in gt_n96.npz (static IoU 0.963, bottom rows), NOT the
FEED-dn label class 2 (which is the static TOP sky region). The mask is a REAL majority
vote over the REAL cached lstars; the SDF is a REAL scipy EDT; the argmax + disagreement
are REAL vs the REAL L* (bit-exact). No stub, no surrogate.

CPU-only, $0. Reads ONLY ``lstars`` from gt_n96.npz (~150 MB, no GPU/MPS, no scorer
fleet). [macOS-CPU advisory] research-signal: score_claim=false, promotable=false, NOT a
byte-closed row.

Usage:
  .venv/bin/python experiments/measure_hood_static_containment.py --n 96
  .venv/bin/python experiments/measure_hood_static_containment.py --n 96 --r-survival
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tac.boundary_math.hood_static_component import (
    build_static_hood_sdf,
    compute_static_hood_mask,
    decompose_argmax_disagreement,
    hood_mask_byte_cost,
    identify_static_hood_class,
    inject_hood_sdf,
)
from tac.boundary_math.lever_b_levelset_generator import signed_distance_fields

_GT = Path("experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
_N_CLASSES = 5
_ROAD = 0


def _mean_decomp(decomps: list) -> dict:
    keys = ["total_dseg", "lane_fn", "lane_fp_from_road", "lane_fp_from_other",
            "class0_dseg", "other_dseg", "lane_attributable"]
    out = {k: float(np.mean([getattr(d, k) for d in decomps])) for k in keys}
    # relabel lane_* -> hood_* for hood semantics (same math; class-agnostic decompose)
    return {
        "total_dseg": out["total_dseg"],
        "hood_fn": out["lane_fn"],
        "hood_fp_from_road": out["lane_fp_from_road"],
        "hood_fp_from_other": out["lane_fp_from_other"],
        "containment_leak": out["lane_fp_from_road"] + out["lane_fp_from_other"],
        "road_disturbed": out["class0_dseg"],
        "other_dseg": out["other_dseg"],
        "hood_attributable": out["lane_attributable"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=96, help="frames (<=96).")
    ap.add_argument("--n-frames-amortize", type=int, default=600,
                    help="frame count to amortize the single mask over (contest 600).")
    ap.add_argument("--r-survival", action="store_true",
                    help="also measure post-R (round-trip) disagreement on --r-n frames.")
    ap.add_argument("--r-n", type=int, default=8)
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    if not _GT.exists():
        raise SystemExit(f"GT cache missing: {_GT}")
    t0 = time.time()
    npz = np.load(_GT)
    lstars = npz["lstars"]  # (96,384,512) int64 — read ONLY this member (keep memory light)
    P = min(int(args.n), lstars.shape[0])
    Lall = np.asarray(lstars[:P]).astype(np.int64)
    print(f"[hood-static $0] n={P} frames; reading lstars only ({lstars.shape}) ...", flush=True)

    # --- NO-FAKE: detect the static ego-hood class from the data ---
    hood_cls, ev = identify_static_hood_class(Lall, n_classes=_N_CLASSES)
    print(f"[NO-FAKE class detect] static ego-hood class = {hood_cls} "
          f"(FEED-dn label said 2; detected from data)")
    for e in sorted(ev, key=lambda x: x.cls):
        print(f"  cls{e.cls}: static_IoU={e.static_iou:.4f} frac={e.frac_of_frame:.4f} "
              f"bottom_share={e.bottom_share:.4f} maj_rows={e.maj_row_span}")

    # --- the SINGLE canonical static mask(s) (computed once, shared by all frames) ---
    maj = compute_static_hood_mask(Lall, hood_cls=hood_cls, agg="majority")
    inter = compute_static_hood_mask(Lall, hood_cls=hood_cls, agg="intersection")
    print(f"\n[static mask] hood_cls={hood_cls}")
    print(f"  majority:     px={maj.px} frac={maj.frac_of_frame:.4f} rows={maj.row_span} "
          f"mean_frame_IoU={maj.mean_frame_iou:.4f} min={maj.min_frame_iou:.4f}")
    print(f"  intersection: px={inter.px} frac={inter.frac_of_frame:.4f} rows={inter.row_span} "
          f"mean_frame_IoU={inter.mean_frame_iou:.4f}")

    phi_maj = build_static_hood_sdf(maj.mask)      # ONE field, all frames
    phi_int = build_static_hood_sdf(inter.mask)

    rows_ideal, rows_maj, rows_int = [], [], []
    for i in range(P):
        L = Lall[i]
        phi_ideal = signed_distance_fields(L, _N_CLASSES)  # (H,W,K); argmax == L exactly
        rows_ideal.append(decompose_argmax_disagreement(phi_ideal.argmax(-1), L,
                                                         lane_cls=hood_cls, road_cls=_ROAD))
        pred_m = inject_hood_sdf(phi_ideal, phi_maj, lane_cls=hood_cls, mode="replace").argmax(-1)
        rows_maj.append(decompose_argmax_disagreement(pred_m, L, lane_cls=hood_cls, road_cls=_ROAD))
        pred_i = inject_hood_sdf(phi_ideal, phi_int, lane_cls=hood_cls, mode="replace").argmax(-1)
        rows_int.append(decompose_argmax_disagreement(pred_i, L, lane_cls=hood_cls, road_cls=_ROAD))
        if (i + 1) % 24 == 0:
            print(f"  ... {i+1}/{P}", flush=True)

    ideal = _mean_decomp(rows_ideal)
    majd = _mean_decomp(rows_maj)
    intd = _mean_decomp(rows_int)

    def _pr(tag, d):
        print(f"\n=== {tag} ===")
        print(f"  total_dseg          {d['total_dseg']:.6f}")
        print(f"  hood_attributable   {d['hood_attributable']:.6f}")
        print(f"    hood_fn (shape)   {d['hood_fn']:.6f}   (per-frame hood beyond static mask)")
        print(f"    containment_leak  {d['containment_leak']:.6f}   (non-hood -> hood; ideal 0)")
        print(f"      fp<-road        {d['hood_fp_from_road']:.6f}")
        print(f"      fp<-other       {d['hood_fp_from_other']:.6f}")
        print(f"  other_dseg          {d['other_dseg']:.6f}")

    _pr("IDEAL SDF (baseline; argmax==L*)", ideal)
    _pr("MAJORITY static mask (recommended)", majd)
    _pr("INTERSECTION static mask (max-contain)", intd)

    byte = hood_mask_byte_cost(maj.mask, n_frames=int(args.n_frames_amortize))
    print(f"\n[byte cost] single majority mask: best_counted={byte['best_counted_bytes']} B "
          f"(row-span-brotli {byte['row_span_brotli_bytes']} / bitmap-brotli {byte['bitmap_brotli_bytes']}) "
          f"-> {byte['amortized_bytes_per_frame']:.4f} B/frame over {byte['n_frames_amortized']} "
          f"-> rate {byte['score_rate_contribution']:.8f}")

    out = {
        "n": P, "hood_cls_detected": int(hood_cls), "feed_dn_label_was": 2,
        "class_evidence": [vars(e) for e in sorted(ev, key=lambda x: x.cls)],
        "static_mask_majority": {k: v for k, v in vars(maj).items() if k != "mask"},
        "static_mask_intersection": {k: v for k, v in vars(inter).items() if k != "mask"},
        "ideal": ideal, "majority": majd, "intersection": intd, "byte_cost": byte,
        "authority": "macOS-CPU advisory", "score_claim": False, "promotable": False,
        "byte_closed_row": False, "elapsed_s": round(time.time() - t0, 1),
    }
    # numpy-typed tuples -> json-safe
    out = json.loads(json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else list(o)))

    if args.r_survival:
        out["r_survival"] = _measure_r_survival(Lall, min(int(args.r_n), P), hood_cls, phi_maj)

    if args.json_out:
        jp = Path(args.json_out)
        if "/tmp" in str(jp):
            raise SystemExit("refuse /tmp json-out (durable evidence rule)")
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(out, indent=2))
        print(f"\n[json] {jp}")
    print(f"\n[done] {out['elapsed_s']}s  [macOS-CPU advisory] research-signal; pointer UNMOVED 0.19110")


def _measure_r_survival(Lall, rn: int, hood_cls: int, phi_hood: np.ndarray) -> dict:
    """post-R disagreement: does the single static hood SDF survive R as well as ideal?"""

    from tac.boundary_math.lever_b_levelset_generator import apply_R_to_fields

    print(f"\n[R-survival] n={rn} (bicubic up -> uint8 @camera -> bilinear down) ...", flush=True)
    ideal_post, maj_post = [], []
    for i in range(rn):
        L = Lall[i]
        phi_ideal = signed_distance_fields(L, _N_CLASSES)
        phi_m = inject_hood_sdf(phi_ideal, phi_hood, lane_cls=hood_cls, mode="replace")
        ideal_post.append(float(np.mean(apply_R_to_fields(phi_ideal).argmax(-1) != L)))
        maj_post.append(float(np.mean(apply_R_to_fields(phi_m).argmax(-1) != L)))
    res = {"ideal_post_R_dseg": float(np.mean(ideal_post)),
           "majority_static_post_R_dseg": float(np.mean(maj_post)), "r_n": rn}
    print(f"  post-R d_seg: ideal {res['ideal_post_R_dseg']:.6f}  "
          f"majority-static {res['majority_static_post_R_dseg']:.6f}")
    return res


if __name__ == "__main__":
    main()
