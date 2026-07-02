# SPDX-License-Identifier: MIT
"""Register the canonical equation for the AA-SDF observation-map render (triality: EQUATIONS leg).

Reads the n600 AA verify JSON (frozen CPU-torch SegNet through the ACTUAL contest R) and registers
``aa_sdf_observation_footprint_render_dseg_v1`` with an EmpiricalAnchor (DESIGN lower bounds from
the committed levelset gate vs the measured n600 curve). Idempotent (append-only 'registered'
event keyed by equation_id).

Usage:
    .venv/bin/python tools/register_aa_sdf_observation_render_equation.py \
        --result reports/aa_sdf_observation_render_verify_n600_20260701.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.canonical_equations.equation import (  # noqa: E402
    CanonicalEquation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
)
from tac.canonical_equations.registry import register_canonical_equation  # noqa: E402
from tac.provenance.builders import (  # noqa: E402
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

# DESIGN lower bounds (the law the equation codifies, from the committed gate FEED-ly):
# footprint-integrated (supersample->box) rendering vs point-sampling at a FIXED render grid
# raises class-1 (lane) recall by >= this, at ~0 rate (decode-time deterministic op).
PRED_RECALL_LIFT_G192 = 0.30       # gate measured +0.38 at 192; conservative lower bound
PRED_AA_DSEG_FLOOR_G384 = 0.00120  # oracle-R AA floor at 384 (gate: 0.00091); upper bound target


def _latest_result(pattern: str) -> Path:
    hits = sorted(glob.glob(str(REPO / pattern)) + glob.glob(pattern))
    n600 = [h for h in hits if "_n600_" in h]
    return Path((n600 or hits)[-1])


def build_equation(result: dict, result_path: str) -> CanonicalEquation:
    curve = result["render_grid_curve"]
    n = int(result["n_pairs"])
    ss = int(result["supersample_ss"])
    # Binding anchor points: recall lift at the coarsest grid (aliasing strongest) + AA d_seg floor
    # at 384 (the LEVELSET default render grid).
    g_lift = "192" if "192" in curve else min(curve, key=lambda k: int(k))
    g_floor = "384" if "384" in curve else max(curve, key=lambda k: int(k))
    emp_recall_lift = float(curve[g_lift]["real"]["aa_recall_lift"])
    emp_aa_dseg_floor = float(curve[g_floor]["real"]["aa"]["d_seg"]["mean"])
    # residual: normalized shortfall of measured vs the design bound (0 if measured beats the bound).
    res_lift = max(0.0, (PRED_RECALL_LIFT_G192 - emp_recall_lift) / max(PRED_RECALL_LIFT_G192, 1e-6))
    res_floor = max(0.0, (emp_aa_dseg_floor - PRED_AA_DSEG_FLOOR_G384) / max(PRED_AA_DSEG_FLOOR_G384, 1e-6))
    residual = float(0.5 * (res_lift + res_floor))
    # full per-grid curve (real signal) for the anchor payload.
    real_curve = {g: {"point_dseg": c["real"]["point"]["d_seg"]["mean"],
                      "aa_dseg": c["real"]["aa"]["d_seg"]["mean"],
                      "point_recall": c["real"]["point"]["lane_recall"]["mean"],
                      "aa_recall": c["real"]["aa"]["lane_recall"]["mean"],
                      "aa_recall_lift": c["real"]["aa_recall_lift"],
                      "aa_dseg_gain": c["real"]["aa_dseg_gain"]}
                  for g, c in curve.items()}

    anchor = EmpiricalAnchor(
        anchor_id=f"aa_sdf_observation_render_n{n}_ss{ss}_frozen_cpu_torch_segnet",
        measurement_utc=result.get("utc", "2026-07-01T00:00:00Z"),
        inputs={"n_pairs": n, "supersample_ss": ss, "grids": list(curve.keys()),
                "signal": "real_frame_achievable_through_R (confound-free upper bound)"},
        predicted_output={"recall_lift_g192_lower_bound": PRED_RECALL_LIFT_G192,
                          "aa_dseg_floor_g384_upper_bound": PRED_AA_DSEG_FLOOR_G384},
        empirical_output={"recall_lift_g192": emp_recall_lift,
                          "aa_dseg_floor_g384": emp_aa_dseg_floor,
                          "render_grid_curve_real": real_curve},
        residual=residual,
        source_artifact=result_path,
        measurement_method="frozen_cpu_torch_segnet_argmax_point_vs_supersample_box_through_contest_R",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=result_path,
            reactivation_criteria="byte-closed upstream/evaluate.py exact row through the AA-wired witness render",
            measurement_axis="[contest-CPU advisory]",
            hardware_substrate="m5_max_cpu_torch",
        ),
    )
    return CanonicalEquation(
        equation_id="aa_sdf_observation_footprint_render_dseg_v1",
        name="AA-SDF observation-map (footprint-integrated) render lowers d_seg at fixed rate",
        one_line_summary=(
            "Footprint-integrated (supersample->box / mip-NeRF IPE) render recovers finest-scale "
            "lanes that point-sampling erases through R, lowering d_seg at ~0 rate."
        ),
        latex_form=(
            r"\hat{c}(p)=\frac{1}{|F_p|}\int_{F_p} g(x)\,dx \approx \frac{1}{s^2}\sum_{s\times s} g "
            r"\;\Rightarrow\; d_{\mathrm{seg}}(\mathrm{AA}) \le d_{\mathrm{seg}}(\mathrm{point});\;"
            r"\mathbb{E}[\sin(2\pi b\cdot x)]=\sin(2\pi b\cdot\mu)e^{-2\pi^2 b^\top\Sigma b}"
        ),
        python_callable_module_path="tac.boundary_math.aa_sdf_observation_render:box_downsample_np",
        domain_of_validity={"scorer": ["frozen_cpu_torch_segnet"], "R": ["contest_faithful_uint8_at_camera"],
                            "aa_mode": ["supersample_box", "mip_nerf_ipe"],
                            "render_grid": [int(g) for g in curve], "n_pairs_range": [1, 600]},
        units_in={"witness_rgb": "0_255", "supersample_ss": "integer", "footprint_sigma": "coord_units"},
        units_out={"d_seg": "fraction", "lane_recall": "fraction"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"recall_lift": res_lift, "aa_dseg_floor": res_floor},
        last_calibration_utc=result.get("utc", "2026-07-01T00:00:00Z"),
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "experiments.train_LEVELSET_witness_realized_through_R_mlx",
            "tac.boundary_math.aa_sdf_observation_render",
        ),
        canonical_producers=(
            "tools/aa_sdf_observation_render_verify_n600.py",
            "tools/register_aa_sdf_observation_render_equation.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="aa_sdf_observation_footprint_render_dseg.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[predicted]",
            hardware_substrate="unknown",
        ),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", default="reports/aa_sdf_observation_render_verify_n600_*.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    rp = _latest_result(args.result)
    result = json.loads(rp.read_text())
    eq = build_equation(result, str(rp))
    eo = eq.empirical_anchors[0].empirical_output
    print(json.dumps({"equation_id": eq.equation_id, "anchor": eq.empirical_anchors[0].anchor_id,
                      "residual": eq.empirical_anchors[0].residual,
                      "recall_lift_g192": eo["recall_lift_g192"],
                      "aa_dseg_floor_g384": eo["aa_dseg_floor_g384"],
                      "result": str(rp)}, indent=2))
    if not args.dry_run:
        register_canonical_equation(eq, subagent_id="aa-sdf-observation-render",
                                    notes="FEED-ma AA-SDF footprint render; n600 frozen-CPU-torch through-R anchor")
        print(json.dumps({"stage": "registered", "equation_id": eq.equation_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
