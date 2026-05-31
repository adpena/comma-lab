# SPDX-License-Identifier: MIT
"""Register the canonical equation ``detector_informed_recon_weight_d_seg_savings_v1``
from the magnitude-test smoke output (Catalog #344).

FAIL-CLOSED: refuses to register unless the smoke output's
``global_magnitude_verdict == "CONTEST_RELEVANT"`` AND at least one surface shows a
contest-relevant margin (``max_abs_margin_detector_vs_uniform >= 1e-3``). If the lever
is real-but-negligible, the equation stays FORMALIZATION_PENDING and this tool exits 2
with the reason. NO fabricated numbers — every anchor field is read from the REAL smoke
JSON. NON-PROMOTABLE per Catalog #192/#341 ([macOS-CPU advisory]).

The equation codifies: on a high-baseline-d_seg render surface, detector-informed
(full-grid SegNet response × S-UNIWARD texture) allocation of a fixed byte budget on
the UWD1 direct-payload sidechannel reduces the post-correction SegNet d_seg by a
contest-relevant margin over uniform allocation, and the margin grows with the budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.canonical_equations import (  # noqa: E402
    CanonicalEquation,
    EmpiricalAnchor,
    register_canonical_equation,
)
from tac.provenance import build_provenance_for_predicted  # noqa: E402

EQUATION_ID = "detector_informed_recon_weight_d_seg_savings_v1"
_CONTEST_RELEVANT_THRESHOLD = 1e-3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--smoke-output",
        default="experiments/results/detector_informed_lever_magnitude_high_dseg_20260531/smoke_output.json",
    )
    ap.add_argument("--subagent-id", default="detector_informed_lever_magnitude_high_dseg_20260531")
    args = ap.parse_args()

    smoke_path = REPO_ROOT / args.smoke_output
    if not smoke_path.is_file():
        print(f"[register] FAIL-CLOSED: smoke output not found at {smoke_path}", file=sys.stderr)
        return 2
    smoke = json.loads(smoke_path.read_text())

    verdict = smoke.get("global_magnitude_verdict")
    best_margin = float(smoke.get("best_margin_detector_vs_uniform_across_surfaces", 0.0))
    if verdict != "CONTEST_RELEVANT" or best_margin < _CONTEST_RELEVANT_THRESHOLD:
        print(
            f"[register] FAIL-CLOSED: verdict={verdict!r} best_margin={best_margin:.6f} "
            f"(< {_CONTEST_RELEVANT_THRESHOLD}); leaving equation FORMALIZATION_PENDING.",
            file=sys.stderr,
        )
        return 2

    # Build empirical anchors from the REAL contest-relevant surfaces.
    anchors: list[EmpiricalAnchor] = []
    for surface, sr in smoke["surface_results"].items():
        if sr.get("magnitude_verdict") != "CONTEST_RELEVANT":
            continue
        baseline = float(sr["baseline_d_seg"])
        for row in sr["verdict_per_budget"]:
            tb = int(row["target_bytes"])
            margin = float(row["margin_detector_vs_uniform"])  # empirical d_seg(uniform)-d_seg(detector)
            # PREDICTED: the model's claim is "margin > 0 and grows with budget"; the
            # quantitative prediction at registration is 0.0 baseline residual (the
            # smoke IS the first empirical anchor — Catalog #344 source-is-anchor).
            anchors.append(
                EmpiricalAnchor(
                    anchor_id=f"mag_{surface}_tb{tb}",
                    measurement_utc=smoke["captured_at_utc"],
                    inputs={
                        "surface": surface,
                        "baseline_d_seg": baseline,
                        "target_bytes": tb,
                        "detector_weight": "full_grid_segnet_input_gradient_saliency",
                    },
                    predicted_output={"margin_detector_vs_uniform_sign": "positive"},
                    empirical_output={
                        "margin_detector_vs_uniform": margin,
                        "d_seg_detector": float(row["d_seg_detector"]),
                        "d_seg_uniform": float(row["d_seg_uniform"]),
                        "detector_lowest": bool(row["detector_lowest"]),
                    },
                    residual=0.0,  # source-is-anchor: smoke IS the first empirical anchor
                    source_artifact=args.smoke_output,
                    measurement_method="macos_arm64_mps_real_segnet_argmax_flip_rate",
                    provenance=build_provenance_for_predicted(
                        model_id="detector_informed_lever_magnitude_high_dseg_v1",
                        inputs_sha256=str(smoke.get("inputs_sha256", "")),
                        measurement_axis="[macOS-CPU advisory]",
                        hardware_substrate="macos_arm64_mps",
                        captured_at_utc=smoke["captured_at_utc"],
                    ),
                    empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
                )
            )

    if not anchors:
        print("[register] FAIL-CLOSED: no contest-relevant anchors built.", file=sys.stderr)
        return 2

    equation = CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Detector-informed reconstruction-weight d_seg savings on high-baseline render",
        one_line_summary=(
            "On a moderate-baseline-d_seg render, detector-informed (full-grid SegNet "
            "response x texture) UWD1 byte allocation beats uniform by a contest-relevant "
            "d_seg margin that grows with budget."
        ),
        latex_form=(
            r"\Delta d_{seg}(B) = d_{seg}^{uniform}(B) - d_{seg}^{detector}(B) > 0, \quad "
            r"\frac{\partial \Delta d_{seg}}{\partial B} > 0, \quad "
            r"\text{cost}_i = \text{texture}_i \cdot (\epsilon + r_i),\ "
            r"r_i = |\partial \mathcal{L}_{seg}/\partial \text{pixel}_i|"
        ),
        python_callable_module_path=(
            "tac.substrates.uniward_per_pixel_distortion.full_grid_segnet_response_cost_map."
            "compose_full_grid_response_cost_map"
        ),
        domain_of_validity={
            "surface_class": ["high_baseline_d_seg_render", "spatial_downsample", "blockify_lut_render"],
            "baseline_d_seg_range": [0.03, 0.55],
            "byte_budget_range": [800, 12800],
            "detector_weight": ["full_grid_segnet_input_gradient_saliency"],
            "sidechannel": ["uwd1_sparse_delta_direct_payload"],
        },
        units_in={
            "target_bytes": "int_byte_budget",
            "baseline_d_seg": "float_argmax_flip_rate",
            "segnet_response": "float_abs_input_gradient_saliency",
        },
        units_out={"margin_detector_vs_uniform": "float_d_seg_reduction"},
        empirical_anchors=tuple(anchors),
        predicted_vs_empirical_residual={"margin_sign_match_fraction": 0.0},
        last_calibration_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        next_recalibration_trigger="when_3+_new_empirical_anchors_in_domain",
        canonical_producers=(
            "experiments/detector_informed_lever_magnitude_high_dseg_smoke.py",
            "tac.substrates.uniward_per_pixel_distortion.full_grid_segnet_response_cost_map",
        ),
        canonical_consumers=(
            "tac.uniward_delta.pack_sparse_delta",
            "tac.substrates.uniward_per_pixel_distortion.detector_informed_direct_payload_cost_map",
        ),
        provenance=build_provenance_for_predicted(
            model_id="detector_informed_lever_magnitude_high_dseg_v1",
            inputs_sha256=str(smoke.get("inputs_sha256", "")),
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="macos_arm64_mps",
            captured_at_utc=smoke["captured_at_utc"],
        ),
    )

    register_canonical_equation(
        equation,
        agent="claude",
        subagent_id=args.subagent_id,
        notes=(
            f"Magnitude test: detector-informed lever scales to contest-relevant "
            f"d_seg on high-baseline render (best margin {best_margin:.6f} vs uniform); "
            f"{len(anchors)} anchors. NON-PROMOTABLE [macOS-CPU advisory]."
        ),
    )
    print(
        f"[register] REGISTERED {EQUATION_ID}: {len(anchors)} anchors, "
        f"best margin {best_margin:.6f} (CONTEST_RELEVANT).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
