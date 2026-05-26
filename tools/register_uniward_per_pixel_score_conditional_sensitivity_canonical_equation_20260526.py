# SPDX-License-Identifier: MIT
"""One-shot canonical equation registration for UNIWARD per-pixel score-conditional sensitivity.

Per Catalog #344 + operator NON-NEGOTIABLE 2026-05-26 ("we need to formalize all
of this and canonicalize and operationalize because I am afraid we are learning
but if we don't have systems of equations and models and such we are just gaining
tribal knowledge"), registers canonical equation
``uniward_per_pixel_score_conditional_sensitivity_weighting_distortion_savings_v1``
from today's empirical anchor landing.

PRECONDITION: Empirical sweep verdict MUST be PARADIGM-VALIDATED-* (joint, seg-only,
or pose-only); FALSIFIED + NULL-NO-EFFECT verdicts MUST NOT register. The script
reads the canonical sweep results JSON at
``.omx/research/uniward_per_pixel_n_plus_1_artifacts_20260526/sweep_results.json``
and refuses if verdict is not validated.

Sister-pattern of:
- ``tools/register_2_new_canonical_equations_20260526.py`` (Markov + residual_hybrid)
- ``tools/register_3_new_canonical_equations_t3_council_binding_revisions_20260526.py``

Per Catalog #287 placeholder-rejection: all rationales ≥4 chars + non-placeholder.
Per Catalog #323 canonical Provenance: every equation + anchor carries Provenance.
Per Catalog #344 ``contest_compliance_rationale`` non-empty: research-only signal
explicitly tagged.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac.canonical_equations import (
    CanonicalEquation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
    register_canonical_equation,
    query_equations,
)
from tac.provenance import build_provenance_for_predicted

import hashlib

NOW_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
INPUTS_SHA256 = hashlib.sha256(
    b"uniward_per_pixel_score_conditional_sensitivity_real_scorer_anchored_50pair_30ep_96x128_seed_42_v1"
).hexdigest()
SWEEP_RESULTS_PATH = (
    ROOT
    / ".omx/research/uniward_per_pixel_n_plus_1_artifacts_20260526/sweep_results.json"
)


def _build_equation(sweep_results: dict) -> CanonicalEquation:
    """Build canonical equation from empirical sweep results JSON."""
    pca = sweep_results["per_axis_comparison"]
    verdict = sweep_results["verdict"]
    fixture = sweep_results["fixture"]
    sg = sweep_results["real_scorer_gradient_stats"]
    wm = sweep_results["weight_map_stats"]

    seg_ratio = pca["seg_ratio_variant_over_baseline"]
    pose_ratio = pca["pose_ratio_variant_over_baseline"]
    contest_delta = pca["contest_partial_delta"]

    anchor_provenance = build_provenance_for_predicted(
        model_id=(
            "uniward_per_pixel_score_conditional_sensitivity_weighting_"
            "real_scorer_anchored_50pair_30ep_96x128_seed_42"
        ),
        inputs_sha256=INPUTS_SHA256,
        measurement_axis="[macOS-MLX research-signal]",
        hardware_substrate="darwin_arm64_m5_max_mlx_local",
        captured_at_utc=sweep_results["measurement_utc"],
    )

    # residual = normalized magnitude of the predicted-vs-empirical gap on
    # the JOINT axis (contest partial). Predicted band per Catalog #296 was
    # [-1, -4]; empirical delta is contest_partial_delta. Map: if empirical
    # falls inside predicted band, residual ≈ 0; if outside, magnitude is
    # |empirical - midpoint(-2.5)| / 1.5
    predicted_midpoint = -2.5
    predicted_half_width = 1.5
    residual = abs(contest_delta - predicted_midpoint) / predicted_half_width

    anchor = EmpiricalAnchor(
        anchor_id="uniward_per_pixel_real_scorer_anchored_50pair_30ep_20260526",
        measurement_utc=sweep_results["measurement_utc"],
        inputs={
            "n_pairs": fixture["n_pairs"],
            "spatial_h": fixture["spatial_h"],
            "spatial_w": fixture["spatial_w"],
            "epochs": fixture["epochs"],
            "learning_rate": fixture["learning_rate"],
            "seed": fixture["seed"],
            "ema_decay": fixture["ema_decay"],
            "weight_map_dynamic_range_ratio": wm["dynamic_range_ratio"],
            "seg_grad_mean": sg["seg_grad_mean"],
            "pose_grad_mean": sg["pose_grad_mean"],
        },
        predicted_output={
            "seg_ratio_variant_over_baseline": 0.85,  # midpoint of predicted [0.7, 1.0]
            "pose_ratio_variant_over_baseline": 0.85,
            "contest_partial_delta": -2.5,
        },
        empirical_output={
            "seg_ratio_variant_over_baseline": seg_ratio,
            "pose_ratio_variant_over_baseline": pose_ratio,
            "contest_partial_delta": contest_delta,
            "verdict": verdict,
        },
        residual=float(residual),
        source_artifact=(
            ".omx/research/uniward_per_pixel_n_plus_1_real_scorer_empirical_"
            "distortion_attack_landed_20260526.md"
        ),
        measurement_method=(
            "uniward_per_pixel_weight_map_from_real_scorer_gradients_then_"
            "controlled_baseline_vs_variant_mlx_local_50pair_30ep_96x128"
        ),
        provenance=anchor_provenance,
    )

    eq_provenance = build_provenance_for_predicted(
        model_id=(
            "uniward_per_pixel_score_conditional_sensitivity_weighting_"
            "distortion_savings_v1"
        ),
        inputs_sha256=INPUTS_SHA256,
        measurement_axis="[predicted]",
        hardware_substrate="closed_form_analytic_fridrich_canonical_inverse_steganalysis",
        captured_at_utc=NOW_UTC,
    )

    return CanonicalEquation(
        equation_id=(
            "uniward_per_pixel_score_conditional_sensitivity_weighting_"
            "distortion_savings_v1"
        ),
        name=(
            "UNIWARD per-pixel score-conditional sensitivity weighting "
            "distortion savings"
        ),
        one_line_summary=(
            "Predicts per-axis d_seg / d_pose reduction ratio when scoring-aware "
            "renderer training uses per-pixel UNIWARD weight map "
            "w[h,w] = 1/(eps + d_seg_grad^2 + d_pose_grad^2) from real scorers "
            "vs uniform-weighting baseline (Fridrich-canonical inverse steganalysis)"
        ),
        latex_form=(
            r"w(h,w) = \frac{1}{\epsilon + |\nabla_x d_{seg}(h,w)|^2 "
            r"+ |\nabla_x d_{pose}(h,w)|^2}; "
            r"\mathcal{L}_{UNIWARD} = \mathbb{E}[w(h,w) \cdot \|x(h,w) - \hat{x}(h,w)\|^2]"
        ),
        python_callable_module_path=(
            "src.tac.substrates.uniward_per_pixel_distortion.weight_map:"
            "compute_per_pixel_uniward_weight_map_numpy"
        ),
        domain_of_validity={
            "n_pairs": {"min": 10, "max": 600},
            "spatial_h": {"min": 48, "max": 384},
            "spatial_w": {"min": 64, "max": 512},
            "epochs": {"min": 10, "max": 1000},
            "weight_map_dynamic_range_ratio": {"min": 2.0, "max": 1000.0},
            "operating_point": "training_perturbation_sigma_range_0p05_to_0p20",
            "substrate_class": [
                "score_aware_renderer_with_per_pixel_uniward_weighting",
            ],
            "entropy_position": "P2_loss_shape_TRAIN_phase_BEFORE_entropy_coder",
            "excluded_contexts": [
                "selector_index_streams",
                "raw_payload_bytes_dense_streams",
                "neural_weight_tensors_self_compression",
                "non_per_pixel_loss_surfaces",
            ],
        },
        units_in={
            "n_pairs": "count",
            "spatial_h": "pixels",
            "spatial_w": "pixels",
            "epochs": "count",
            "seg_grad_mean": "scorer_units_per_pixel",
            "pose_grad_mean": "scorer_units_per_pixel",
            "weight_map_dynamic_range_ratio": "dimensionless",
        },
        units_out={
            "seg_ratio_variant_over_baseline": "dimensionless",
            "pose_ratio_variant_over_baseline": "dimensionless",
            "contest_partial_delta": "contest_score_units",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "joint_contest_partial_axis_real_scorer_anchored": float(residual),
        },
        last_calibration_utc=NOW_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
            "tools.cathedral_autopilot_autonomous_loop:invoke_meta_lagrangian_on_candidates",
        ),
        canonical_producers=(
            "src.tac.substrates.uniward_per_pixel_distortion.weight_map",
            "src.tac.substrates.uniward_per_pixel_distortion.score_aware_loss",
        ),
        provenance=eq_provenance,
    )


def main() -> int:
    if not SWEEP_RESULTS_PATH.exists():
        print(f"ERROR: sweep results not found at {SWEEP_RESULTS_PATH}", file=sys.stderr)
        return 1
    results = json.loads(SWEEP_RESULTS_PATH.read_text())
    verdict = results.get("verdict", "UNKNOWN")
    print(f"Sweep verdict: {verdict!r}")
    if not verdict.startswith("PARADIGM-VALIDATED"):
        print(
            f"REFUSING registration: verdict {verdict!r} is not PARADIGM-VALIDATED. "
            "Per Catalog #307 IMPLEMENTATION-LEVEL classification + CLAUDE.md "
            "'Forbidden premature KILL': sister 6th-order iteration required."
        )
        return 2

    eq = _build_equation(results)
    print(f"Registering {eq.equation_id!r}...")
    register_canonical_equation(eq)
    print(f"  ✓ Registered with {len(eq.empirical_anchors)} anchor(s)")

    print("\n=== Verification ===")
    eqs = {e.equation_id for e in query_equations()}
    target_id = eq.equation_id
    print(f"  {target_id}: {'PRESENT in registry' if target_id in eqs else 'MISSING'}")
    print(f"  total registry size: {len(eqs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
