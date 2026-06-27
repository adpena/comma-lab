"""$0 measurement: sky/undrivable + road-as-complement components AND the FIRST
JOINT-CONTAINMENT measure of the wired structured-component codec (FEED-dw).

Completes the structured-region decomposition (lane FEED-ds + hood FEED-dv + THIS:
sky + road) and begins the FEED-dt integration-seal — the highest-risk item: do the
structured components COMPOSE without antagonism in ONE argmax?

Per the validated isolation template (build structured field -> inject as phi_k ->
recompute argmax vs the frozen CPU-torch L* -> decompose), measures:
  1. SKY/UNDRIVABLE (class detected): static-mask SDF vs per-frame horizon-line SDF
     (precise + contained + bytes; which captures the static TOP region better?).
  2. ROAD-AS-COMPLEMENT: inject lane+sky+hood, set phi_road = const, measure how much
     road d_seg is captured FOR FREE (is a dedicated road SDF even needed?).
  3. THE JOINT-CONTAINMENT MEASURE: inject ALL structured components TOGETHER (road-
     complement + lane + sky + hood), keep ONLY learned Movable ideal, recompute the
     joint argmax, and report:
       (a) joint total d_seg vs the SUM of per-component isolation d_segs (ADD or ANTAGONIZE?),
       (b) per-class containment in the joint (full confusion: does class-A leak into class-B?),
       (c) the RESIDUAL d_seg the witness must LEARN (Movable mass + fine boundary).

NO-FAKE: classes are DETECTED from the data (classify_segnet_regions), NOT hardcoded.
Masks/lines are REAL fits to the REAL cached lstars; SDFs are REAL scipy EDTs; the joint
argmax + full per-class disagreement are REAL vs the REAL cached L* (bit-exact). No stub,
no surrogate. CPU-only, $0; reads ONLY `lstars` from gt_n96.npz (~150 MB; no GPU/MPS, no
scorer fleet). [macOS-CPU advisory] research-signal: score_claim=false, promotable=false,
NOT a byte-closed row.

Usage:
  .venv/bin/python experiments/measure_road_horizon_joint_containment.py --n 96
  .venv/bin/python experiments/measure_road_horizon_joint_containment.py --n 96 --r-survival \
      --json-out experiments/results/road_horizon_joint_FEED-dw/n96.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tac.boundary_math.lane_sdf_component import build_structured_lane_sdf, inject_lane_sdf
from tac.boundary_math.hood_static_component import (
    build_static_hood_sdf,
    compute_static_hood_mask,
)
from tac.boundary_math.road_horizon_component import (
    build_sky_line_sdf,
    build_static_sky_sdf,
    classify_segnet_regions,
    decompose_full_confusion,
    mean_full_decomp,
    road_complement_byte_cost,
    road_complement_field,
    sky_line_byte_cost,
    static_mask_byte_cost,
)
from tac.boundary_math.lever_b_levelset_generator import signed_distance_fields

_GT = Path("experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
_N_CLASSES = 5


def _assemble(phi_ideal: np.ndarray, replacements: dict) -> np.ndarray:
    """Reuse inject_lane_sdf (class-agnostic, mode=replace) to substitute each structured
    channel into the ideal stack."""

    out = phi_ideal
    for k, field in replacements.items():
        out = inject_lane_sdf(out, field, lane_cls=int(k), mode="replace")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=96, help="frames (<=96).")
    ap.add_argument("--n-frames-amortize", type=int, default=600)
    ap.add_argument("--horizon-deg", type=int, default=1, help="horizon-line poly degree (1=tilt).")
    ap.add_argument("--r-survival", action="store_true")
    ap.add_argument("--r-n", type=int, default=8)
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    if not _GT.exists():
        raise SystemExit(f"GT cache missing: {_GT}")
    t0 = time.time()
    npz = np.load(_GT)
    lstars = npz["lstars"]  # (96,384,512) int64 — read ONLY this member (memory-light)
    P = min(int(args.n), lstars.shape[0])
    Lall = np.asarray(lstars[:P]).astype(np.int64)
    print(f"[road-horizon-joint $0] n={P} frames; reading lstars only ({lstars.shape}) ...", flush=True)

    # --- NO-FAKE: detect ALL 5 roles from the data ---
    roles = classify_segnet_regions(Lall, n_classes=_N_CLASSES)
    R, LN, SK, MV, HD = roles.road, roles.lane, roles.sky, roles.movable, roles.hood
    print(f"\n[NO-FAKE region detect] road={R} lane={LN} sky={SK} movable={MV} hood={HD}")
    for e in roles.evidence:
        print(f"  cls{e.cls}: iou={e.static_iou:.3f} frac={e.frac_of_frame:.3f} "
              f"top={e.top_share:.3f} bot={e.bottom_share:.3f} mid={e.mid_share:.3f} "
              f"lines={e.n_lane_lines} rows={e.maj_row_span}")

    # --- static fields computed ONCE (shared by all frames) ---
    sky_sm = compute_static_hood_mask(Lall, hood_cls=SK, agg="majority")
    phi_sky_static, _ = build_static_sky_sdf(Lall, sky_cls=SK, agg="majority")
    hood_sm = compute_static_hood_mask(Lall, hood_cls=HD, agg="majority")
    phi_hood = build_static_hood_sdf(hood_sm.mask)
    phi_road = road_complement_field(Lall.shape[1], Lall.shape[2], level=0.0)
    print(f"\n[static] sky majority: px={sky_sm.px} frac={sky_sm.frac_of_frame:.4f} "
          f"rows={sky_sm.row_span} mean_IoU={sky_sm.mean_frame_iou:.4f} min={sky_sm.min_frame_iou:.4f}")
    print(f"[static] hood majority: px={hood_sm.px} frac={hood_sm.frac_of_frame:.4f} "
          f"rows={hood_sm.row_span} mean_IoU={hood_sm.mean_frame_iou:.4f}")

    # variant accumulators (lists of FullDecomp)
    V = {k: [] for k in [
        "ideal", "iso_lane", "iso_hood", "iso_sky_mask", "iso_sky_line", "iso_road",
        "joint_mask", "joint_line", "joint_line_no_movable", "joint_mask_roadideal",
    ]}
    horizon_coeffs: list = []
    horizon_rms: list = []
    lane_floats: list = []

    for i in range(P):
        L = Lall[i]
        phi_ideal = signed_distance_fields(L, _N_CLASSES)        # argmax==L exactly
        # lane OPTIMAL FORM = continuous band (dash_gate=False) per FEED-ds (the dash gate over-gates)
        phi_lane, lmeta = build_structured_lane_sdf(L, lane_cls=LN, dash_gate=False)
        lane_floats.append(int(lmeta.get("total_floats", 0)))
        phi_sky_line, smeta = build_sky_line_sdf(L, sky_cls=SK, deg=int(args.horizon_deg))
        if smeta.get("fit"):
            horizon_coeffs.append(smeta["coeffs"])
            horizon_rms.append(float(smeta["rms_row"]))

        def dec(stack):
            return decompose_full_confusion(stack.argmax(-1), L, n_classes=_N_CLASSES)

        # baseline
        V["ideal"].append(dec(phi_ideal.copy()))
        # isolation (replace ONE channel; others ideal)
        V["iso_lane"].append(dec(_assemble(phi_ideal.copy(), {LN: phi_lane})))
        V["iso_hood"].append(dec(_assemble(phi_ideal.copy(), {HD: phi_hood})))
        V["iso_sky_mask"].append(dec(_assemble(phi_ideal.copy(), {SK: phi_sky_static})))
        V["iso_sky_line"].append(dec(_assemble(phi_ideal.copy(), {SK: phi_sky_line})))
        V["iso_road"].append(dec(_assemble(phi_ideal.copy(), {R: phi_road})))
        # JOINT (all structured together; Movable ideal)
        V["joint_mask"].append(dec(_assemble(phi_ideal.copy(),
                                              {R: phi_road, LN: phi_lane, SK: phi_sky_static, HD: phi_hood})))
        V["joint_line"].append(dec(_assemble(phi_ideal.copy(),
                                             {R: phi_road, LN: phi_lane, SK: phi_sky_line, HD: phi_hood})))
        # JOINT but road kept IDEAL (isolates structured mutual antagonism from the complement)
        V["joint_mask_roadideal"].append(dec(_assemble(phi_ideal.copy(),
                                                       {LN: phi_lane, SK: phi_sky_static, HD: phi_hood})))
        # JOINT with Movable REMOVED (phi_movable deep-negative) -> movable-must-learn magnitude
        phi_no_mv = phi_ideal.copy()
        phi_no_mv[..., MV] = -float(max(L.shape))
        V["joint_line_no_movable"].append(dec(_assemble(phi_no_mv,
                                                        {R: phi_road, LN: phi_lane, SK: phi_sky_line, HD: phi_hood})))
        del phi_ideal, phi_lane, phi_sky_line, phi_no_mv
        if (i + 1) % 24 == 0:
            print(f"  ... {i+1}/{P}", flush=True)

    means = {k: mean_full_decomp(v, n_classes=_N_CLASSES) for k, v in V.items()}
    role_names = {R: "road", LN: "lane", SK: "sky", MV: "movable", HD: "hood"}

    def show(tag, m):
        print(f"\n=== {tag} ===  total_dseg={m['total_dseg']:.6f}")
        for k in range(_N_CLASSES):
            print(f"   {role_names[k]:8s}(c{k}) true={m['true_frac'][k]:.4f} "
                  f"FN={m['fn'][k]:.6f} FP={m['fp'][k]:.6f}")

    show("IDEAL (all 5 ideal SDF; harness baseline)", means["ideal"])
    show("ISO lane (phi_lane, others ideal)", means["iso_lane"])
    show("ISO hood (phi_hood static, others ideal)", means["iso_hood"])
    show("ISO sky-MASK (static, others ideal)", means["iso_sky_mask"])
    show("ISO sky-LINE (per-frame horizon, others ideal)", means["iso_sky_line"])
    show("ISO road-COMPLEMENT (phi_road=const, others ideal)", means["iso_road"])

    # pick the better sky variant for the headline joint
    sky_mask_iso = means["iso_sky_mask"]["total_dseg"]
    sky_line_iso = means["iso_sky_line"]["total_dseg"]
    sky_winner = "line" if sky_line_iso < sky_mask_iso else "mask"
    joint_key = "joint_line" if sky_winner == "line" else "joint_mask"

    show("JOINT (road-IDEAL + lane + sky-MASK + hood; structured mutual only)", means["joint_mask_roadideal"])
    show("JOINT (road-compl + lane + sky-MASK + hood; movable ideal)", means["joint_mask"])
    show("JOINT (road-compl + lane + sky-LINE + hood; movable ideal)", means["joint_line"])

    # antagonism: joint vs sum of isolations (using the chosen sky variant)
    sky_iso_used = sky_line_iso if sky_winner == "line" else sky_mask_iso
    iso_sum = (means["iso_lane"]["total_dseg"] + means["iso_hood"]["total_dseg"]
               + sky_iso_used + means["iso_road"]["total_dseg"])
    joint_total = means[joint_key]["total_dseg"]
    antagonism = joint_total - iso_sum
    # disambiguate: structured mutual antagonism (road IDEAL) vs the road-complement's marginal cost
    joint_mutual = means["joint_mask_roadideal"]["total_dseg"]
    road_complement_marginal = means["joint_mask"]["total_dseg"] - joint_mutual
    # residual the witness must LEARN
    movable_true = means[joint_key]["true_frac"][MV]            # movable mass (given ideal in joint)
    joint_no_mv = means["joint_line_no_movable"]["total_dseg"]  # what happens if movable not given
    movable_learn = joint_no_mv - joint_total                   # movable-induced d_seg if unlearned
    structured_floor = joint_total                              # residual at structured-class boundaries

    print("\n" + "=" * 72)
    print("JOINT-CONTAINMENT VERDICT (FEED-dw integration-seal first measure)")
    print("=" * 72)
    print(f"  sky winner (lower isolation d_seg): {sky_winner.upper()}  "
          f"(mask {sky_mask_iso:.6f} vs line {sky_line_iso:.6f})")
    print(f"  SUM of isolations  (lane+hood+sky[{sky_winner}]+road) = {iso_sum:.6f}")
    print(f"  JOINT total d_seg  (all structured, movable ideal)    = {joint_total:.6f}")
    print(f"  ANTAGONISM (joint - sum) = {antagonism:+.6f}  "
          f"(ratio joint/sum = {joint_total / iso_sum if iso_sum else float('nan'):.3f})")
    print(f"    -> {'ANTAGONIZE (compound)' if antagonism > 1e-4 else ('SYNERGY' if antagonism < -1e-4 else 'ADDITIVE/~no-interaction')}")
    print(f"  attribution: structured-MUTUAL joint (road IDEAL) = {joint_mutual:.6f}; "
          f"road-complement marginal = {road_complement_marginal:+.6f}")
    print(f"  RESIDUAL the witness must LEARN:")
    print(f"    structured-class fine-boundary floor (movable given) = {structured_floor:.6f}")
    print(f"    Movable class mass (true_frac, must be learned)      = {movable_true:.6f}")
    print(f"    Movable-induced d_seg if NOT given (joint_no_mv-joint) = {movable_learn:.6f}")
    print(f"    => witness learned-residual ~ structured_floor + movable = "
          f"{structured_floor + movable_learn:.6f} (vs full-ideal {means['ideal']['total_dseg']:.6f})")

    # byte costs
    sky_mask_bytes = static_mask_byte_cost(sky_sm.mask, n_frames=int(args.n_frames_amortize))
    hcoef = np.asarray(horizon_coeffs, np.float64) if horizon_coeffs else np.zeros((1, args.horizon_deg + 1))
    sky_line_bytes = sky_line_byte_cost(hcoef, n_frames=int(args.n_frames_amortize))
    road_bytes = road_complement_byte_cost()
    print(f"\n[byte] sky-MASK static: {sky_mask_bytes['best_counted_bytes']} B total "
          f"({sky_mask_bytes['amortized_bytes_per_frame']:.4f} B/frame, rate {sky_mask_bytes['score_rate_contribution']:.2e})")
    print(f"[byte] sky-LINE per-frame: {sky_line_bytes['best_counted_bytes']} B total "
          f"({sky_line_bytes['bytes_per_frame']:.4f} B/frame, rate {sky_line_bytes['score_rate_contribution']:.2e}); "
          f"horizon rms {float(np.mean(horizon_rms)) if horizon_rms else -1:.2f} px")
    print(f"[byte] road-COMPLEMENT: {road_bytes['counted_bytes']} B ({road_bytes['note']})")

    out = {
        "n": P, "horizon_deg": int(args.horizon_deg),
        "roles": roles.as_dict(),
        "region_evidence": [vars(e) for e in roles.evidence],
        "sky_static_mask": {k: v for k, v in vars(sky_sm).items() if k != "mask"},
        "hood_static_mask": {k: v for k, v in vars(hood_sm).items() if k != "mask"},
        "variants": means,
        "sky_winner": sky_winner,
        "antagonism": {
            "sum_of_isolations": iso_sum, "joint_total": joint_total,
            "antagonism_abs": antagonism,
            "ratio_joint_over_sum": float(joint_total / iso_sum) if iso_sum else None,
            "verdict": ("ANTAGONIZE" if antagonism > 1e-4 else ("SYNERGY" if antagonism < -1e-4 else "ADDITIVE")),
            "structured_mutual_joint_road_ideal": joint_mutual,
            "road_complement_marginal": road_complement_marginal,
        },
        "witness_residual": {
            "structured_fine_boundary_floor": structured_floor,
            "movable_true_frac": movable_true,
            "movable_induced_if_unlearned": movable_learn,
            "approx_learned_residual": structured_floor + movable_learn,
        },
        "byte_cost": {"sky_mask": sky_mask_bytes, "sky_line": sky_line_bytes, "road": road_bytes},
        "lane_floats_per_frame_mean": float(np.mean(lane_floats)) if lane_floats else 0.0,
        "horizon_rms_row_mean": float(np.mean(horizon_rms)) if horizon_rms else -1.0,
        "authority": "macOS-CPU advisory", "score_claim": False, "promotable": False,
        "byte_closed_row": False, "elapsed_s": round(time.time() - t0, 1),
    }
    out = json.loads(json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else list(o)))

    if args.r_survival:
        out["r_survival"] = _measure_r_survival(
            Lall, min(int(args.r_n), P), R, LN, SK, HD, phi_road, phi_hood, phi_sky_static)

    if args.json_out:
        jp = Path(args.json_out)
        if "/tmp" in str(jp):
            raise SystemExit("refuse /tmp json-out (durable evidence rule)")
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(out, indent=2))
        print(f"\n[json] {jp}")
    print(f"\n[done] {out['elapsed_s']}s  [macOS-CPU advisory] research-signal; pointer UNMOVED 0.19110")


def _measure_r_survival(Lall, rn, R, LN, SK, HD, phi_road, phi_hood, phi_sky_static) -> dict:
    """post-R disagreement: does the full structured JOINT (sky-MASK, the winner) survive R?
    Reports road-const (recommended-complement) AND road-ideal (the clean structured floor)."""

    from tac.boundary_math.lever_b_levelset_generator import apply_R_to_fields

    print(f"\n[R-survival] n={rn} (bicubic up -> uint8 @camera -> bilinear down) ...", flush=True)
    ideal_post, jc_post, ji_post = [], [], []
    for i in range(rn):
        L = Lall[i]
        phi_ideal = signed_distance_fields(L, _N_CLASSES)
        phi_lane, _ = build_structured_lane_sdf(L, lane_cls=LN, dash_gate=False)
        joint_const = _assemble(phi_ideal.copy(),
                                {R: phi_road, LN: phi_lane, SK: phi_sky_static, HD: phi_hood})
        joint_ideal = _assemble(phi_ideal.copy(),
                                {LN: phi_lane, SK: phi_sky_static, HD: phi_hood})  # road ideal
        ideal_post.append(float(np.mean(apply_R_to_fields(phi_ideal).argmax(-1) != L)))
        jc_post.append(float(np.mean(apply_R_to_fields(joint_const).argmax(-1) != L)))
        ji_post.append(float(np.mean(apply_R_to_fields(joint_ideal).argmax(-1) != L)))
    res = {"ideal_post_R_dseg": float(np.mean(ideal_post)),
           "joint_roadconst_skymask_post_R_dseg": float(np.mean(jc_post)),
           "joint_roadideal_skymask_post_R_dseg": float(np.mean(ji_post)), "r_n": rn}
    print(f"  post-R d_seg: ideal {res['ideal_post_R_dseg']:.6f}  "
          f"joint(road-const,sky-mask) {res['joint_roadconst_skymask_post_R_dseg']:.6f}  "
          f"joint(road-ideal,sky-mask) {res['joint_roadideal_skymask_post_R_dseg']:.6f}")
    return res


if __name__ == "__main__":
    main()
