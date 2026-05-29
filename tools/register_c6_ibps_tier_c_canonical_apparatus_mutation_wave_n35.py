#!/usr/bin/env -S uv run --quiet python
# SPDX-License-Identifier: MIT
"""Wave N+35 canonical apparatus mutation per RANK 4 highest-EV-shortest-WC slot directive.

Registers BOTH:
- Canonical equation `c6_ibps_post_training_tier_c_within_class_density_v1` per Catalog #344
  with the empirical anchor density=0.9711040488 from the 2026-05-20 Tier-C re-measurement on
  C6 IBPS landed archive sha=be06a4b0972e6cedcfe64ebc69bde394b72dab3cc5d372a887ecf185d8a2dbec.

- Canonical anti-pattern `random_init_tier_c_density_predicts_post_training_class_shift_v1`
  per Catalog #344 anti-pattern sister + Catalog #324 bug class anchor (the 22x miss) with the
  EmpiricalFalsification capturing IMPLEMENTATION_LEVEL_CONFIRMATION per Catalog #307.

Sister of Catalog #313 probe_outcomes ledger row (already landed 2026-05-20). This canonical
mutation closes the gap where the empirical result was in probe_outcomes but NOT in the
canonical equations + anti-patterns registries per the operator 2026-05-28 NON-NEGOTIABLE
"memos must be acted upon".
"""
from __future__ import annotations

from datetime import datetime, timezone

import tac.canonical_anti_patterns as cap
import tac.canonical_equations as ce
from tac.provenance import (
    ProvenanceEvidenceGrade,
    build_provenance_for_archive_member,
    build_provenance_for_predicted,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


ARCHIVE_PATH = (
    "experiments/results/lane_substrate_c6_e4_mdl_ibps_modal_t4_dispatch_"
    "20260517T230751Z__smoke__50ep_modal/harvested_artifacts/archive.zip"
)
ARCHIVE_SHA = (
    "be06a4b0972e6cedcfe64ebc69bde394b72dab3cc5d372a887ecf185d8a2dbec"
)
EVIDENCE_PATH = (
    "experiments/results/c6_ibps_post_training_tier_c_remeasurement_"
    "20260520T020900Z/c6_ibps_50ep_be06a4b09_post_training_mdl_ablation.json"
)
PROBE_ID = (
    "c6_e4_mdl_ibps_post_training_tier_c_remeasurement_landed_archive_"
    "be06a4b09_20260519"
)


def register_equation() -> ce.CanonicalEquation:
    """Register the C6 IBPS post-training Tier-C within-class density canonical equation."""
    measurement_utc = "2026-05-20T02:13:06Z"

    # Provenance for the equation itself (predicted-from-model registration anchor)
    eq_provenance = build_provenance_for_predicted(
        model_id="c6_ibps_post_training_tier_c_within_class_density_v1",
        inputs_sha256=ARCHIVE_SHA,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
        captured_at_utc=_utc_now(),
    )

    # Provenance for the empirical anchor (canonical archive-member custody)
    # The C6 IBPS archive uses canonical monolithic single-file packet '0.bin'
    # per HNeRV parity discipline lesson 3.
    anchor_provenance = build_provenance_for_archive_member(
        archive_zip_path=ARCHIVE_PATH,
        member_name="0.bin",
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="macos_arm64",
        evidence_grade=ProvenanceEvidenceGrade.MACOS_CPU_ADVISORY,
        captured_at_utc=measurement_utc,
    )

    empirical_anchor = ce.EmpiricalAnchor(
        anchor_id=PROBE_ID,
        measurement_utc=measurement_utc,
        inputs={
            "archive_sha256": ARCHIVE_SHA,
            "archive_size_bytes": 225157,
            "grammar": "ibps1",
            "device": "cpu",
            "pair_samples": 30,
            "noise_sigma_relative_range": [0.001, 1.0],
            "tier_c_targets": ["state_dict", "latents"],
            "substrate_config": {
                "latent_dim": 24,
                "beta_ib": 0.01,
                "epochs": 50,
            },
            "in_domain_context": "c6_ibps_v1_post_training_continuous_gaussian_ib_24_dim_bottleneck",
        },
        predicted_output={
            "predicted_band_lower": 0.113,
            "predicted_band_upper": 0.163,
            "random_init_pre_training_density": 2.67e-5,
            "claimed_class": "across_class",
            "claim_origin": "tier_c_measured_on_random_init_pre_training_archive",
        },
        empirical_output={
            "mdl_tier_c_density_estimate": 0.9711040487603954,
            "mdl_tier_c_substrate_class_verdict": "within_class",
            "mdl_tier_c_curve_knee_signal": 251.7860907662281,
            "mdl_tier_c_latent_sigma1_delta": 1.0365474278190026,
            "final_score": 3.04,
            "ratio_vs_random_init_claim": 36400.0,
            "axis_tag": "[macOS-CPU advisory only]",
            "evidence_grade": "MDL-ablation-cpu",
        },
        residual=0.9711040487603954 - 0.0,  # canonical equation predicts 0 (random_init claim was ACROSS_CLASS = density ~ 0)
        source_artifact=EVIDENCE_PATH,
        measurement_method="tools/mdl_scorer_conditional_ablation.py --tier c --pair-samples 30 (cpu; 493s wall-clock; $0 spend)",
        provenance=anchor_provenance,
    )

    equation = ce.CanonicalEquation(
        equation_id="c6_ibps_post_training_tier_c_within_class_density_v1",
        name="C6 IBPS post-training Tier-C density empirically within-class",
        one_line_summary=(
            "C6 IBPS post-training Tier-C density=0.9711 WITHIN_CLASS supersedes random-init "
            "across_class claim (2.67e-5); 36,400x amplification per Catalog #324."
        ),
        latex_form=(
            r"D_{tier\_c}(\theta_{post-train}) "
            r"= \frac{1}{2}\left(\sigma\left(\frac{|\Delta_{latents,\sigma=1.0}|}{|\Delta_{latents,\sigma=1.0}| + 0.05}\right) "
            r"+ \sigma\left(\frac{|\Delta_{sd,\sigma=0.1}/\Delta_{sd,\sigma=0.01}|}{|\Delta_{sd,\sigma=0.1}/\Delta_{sd,\sigma=0.01}| + 3.0}\right)\right) "
            r"\approx 0.97 \gg 0.30 \Rightarrow \text{WITHIN\_CLASS}"
        ),
        python_callable_module_path="tac.analysis.scorer_conditional_mdl:compute_tier_c_density",
        domain_of_validity={
            "substrate_class": "continuous_gaussian_information_bottleneck",
            "architecture_class": "ib_24_dim_continuous_gaussian_posterior_3_hidden_layer_64_unit_encoder",
            "epochs_range": [50, 100],
            "beta_ib_range": [0.001, 0.1],
            "latent_dim_range": [12, 48],
            "axis": "tier_c_density_post_training",
            "regime": "near_converged_smoke_post_training",
            "relevance_tokens": [
                "c6", "c6_e4_mdl_ibps", "c6_ibps", "ibps", "information_bottleneck",
                "continuous_gaussian", "ib_24_dim", "tier_c", "post_training",
                "within_class", "shannon_zen_floor", "tishby_ib_principle",
                "predicted_band_phantom_random_init", "catalog_324_anchor",
                "catalog_307_implementation_level_falsification",
                "catalog_325_per_substrate_symposium_evidence",
                "catalog_313_probe_outcomes_ledger_defer",
            ],
            "supersedes_random_init_density_claim": True,
        },
        units_in={
            "noise_sigma_relative": "fraction_of_parameter_std",
            "pair_samples": "count",
            "archive_size_bytes": "bytes",
            "latent_dim": "continuous_dimensions",
        },
        units_out={
            "mdl_tier_c_density_estimate": "dimensionless_unit_interval",
            "ratio_vs_random_init_claim": "dimensionless_ratio",
        },
        empirical_anchors=(empirical_anchor,),
        predicted_vs_empirical_residual={
            "median_residual": 0.9711040487603954,
            "max_abs_residual": 0.9711040487603954,
            "amplification_factor": 36400.0,
            "predicted_band_miss_ratio": 22.0,
        },
        last_calibration_utc=measurement_utc,
        next_recalibration_trigger=ce.RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
            "tools/audit_predicted_band_provenance.py",
            "tac.optimization.tier_c_density_post_training_validator",
        ),
        canonical_producers=(
            "tools/mdl_scorer_conditional_ablation.py",
            "tac.analysis.scorer_conditional_mdl",
        ),
        provenance=eq_provenance,
    )

    return ce.register_canonical_equation(
        equation,
        agent="claude",
        subagent_id="wave_n35_c6_ibps_landed_archive_tier_c_re_measurement_20260528",
        notes=(
            "Wave N+35 RANK 4 highest-EV-shortest-WC canonical apparatus mutation. "
            "Empirical anchor from 2026-05-20 Tier-C re-measurement on C6 IBPS landed archive "
            "be06a4b0972e6c. Density=0.9711 WITHIN_CLASS supersedes random-init pre-training "
            "across_class claim (density 2.67e-5; 36,400x off post-training). Per CLAUDE.md "
            "Forbidden premature KILL: paradigm INTACT (Tishby IB principle valid); C6 IBPS v1 "
            "implementation-level falsified per Catalog #307. Reactivation queue per "
            "probe_outcomes row: Path B2 DreamerV3 RSSM categorical posterior (Path A theoretical "
            "derivation already landed via canonical equation #2 categorical_posterior_capacity_"
            "vs_continuous_gaussian_v1 2026-05-20T13:18:15Z); Path B1 hierarchical IB; Path B3 "
            "beta-tuned sweep; Path B4 combined."
        ),
    )


def register_anti_pattern() -> cap.AntiPattern:
    """Register the canonical anti-pattern that the Catalog #324 META-fix structurally extincts."""
    measurement_utc = "2026-05-20T02:24:31Z"

    ap_provenance = build_provenance_for_predicted(
        model_id="random_init_tier_c_density_predicts_post_training_class_shift_v1",
        inputs_sha256=ARCHIVE_SHA,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
        captured_at_utc=_utc_now(),
    )

    fals_provenance = build_provenance_for_archive_member(
        archive_zip_path=ARCHIVE_PATH,
        member_name="0.bin",
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="macos_arm64",
        evidence_grade=ProvenanceEvidenceGrade.MACOS_CPU_ADVISORY,
        captured_at_utc=measurement_utc,
    )

    empirical_falsification = cap.EmpiricalFalsification(
        anti_pattern_id="random_init_tier_c_density_predicts_post_training_class_shift_v1",
        falsification_id="c6_ibps_v1_22x_miss_pre_to_post_training_tier_c_density_amplification_36400x_20260520",
        measurement_method=(
            "Catalog #324 post-training Tier-C re-measurement via "
            "tools/mdl_scorer_conditional_ablation.py --tier c --pair-samples 30 on landed "
            "archive be06a4b0972e6c vs random-init pre-training Tier-C anchor (density 2.67e-5 "
            "ACROSS_CLASS) baked into substrate recipe predicted_band [0.113, 0.163]"
        ),
        empirical_artifact_path=EVIDENCE_PATH,
        empirical_output={
            "post_training_tier_c_density": 0.9711040487603954,
            "post_training_class_verdict": "within_class",
            "random_init_tier_c_density_claim": 2.67e-5,
            "random_init_class_claim": "across_class",
            "amplification_factor_post_vs_pre": 36400.0,
            "predicted_band": [0.113, 0.163],
            "empirical_final_score": 3.04,
            "predicted_band_miss_ratio": 22.0,
            "modal_call_id": "fc-01KRW353MJJ9A6QW8H99QWZEMH",
            "wall_clock_seconds": 493.45,
            "gpu_spend_usd": 0.0,
            "axis_tag": "[macOS-CPU advisory only]",
        },
        falsification_residual=0.9711040487603954 - 2.67e-5,
        captured_at_utc=measurement_utc,
        canonical_provenance=fals_provenance,
        incident_classification=cap.INCIDENT_IMPLEMENTATION_LEVEL_CONFIRMATION,
        severity_observed=cap.SEVERITY_OBSERVED_HIGH,
        operator_routable_unwind_path=(
            "Per CLAUDE.md 'Forbidden premature KILL without research exhaustion' + Catalog #307 "
            "paradigm-vs-implementation discipline: paradigm (Tishby IB principle) INTACT; C6 IBPS "
            "v1 implementation (beta_ib=0.01 + latent_dim=24 + continuous Gaussian 24-dim "
            "bottleneck) FALSIFIED at the IMPLEMENTATION level. Reactivation queue per "
            "probe_outcomes ledger row "
            "c6_e4_mdl_ibps_post_training_tier_c_remeasurement_landed_archive_be06a4b09_20260519: "
            "(1) Path B2 DreamerV3 RSSM categorical posterior smoke (Path A theoretical derivation "
            "already landed via canonical equation categorical_posterior_capacity_vs_continuous_"
            "gaussian_v1 2026-05-20T13:18:15Z; converged MLX advisory training 500ep landed "
            "2026-05-27 with archive sha af5d78b92a0d8086 awaiting paired CUDA/CPU + post-training "
            "Tier-C re-measurement per Catalog #324); (2) Path B1 hierarchical IB; (3) Path B3 "
            "beta-tuned sweep; (4) Path B4 combined. Each variant requires fresh per-substrate "
            "symposium per Catalog #325 + post-training Tier-C re-measurement on the variant "
            "first-anchor archive per Catalog #324."
        ),
    )

    anti_pattern = cap.AntiPattern(
        anti_pattern_id="random_init_tier_c_density_predicts_post_training_class_shift_v1",
        description=(
            "Substrate recipe declares predicted_band derived from Tier-C density measured on "
            "RANDOM_INIT pre-training archive AS IF it predicts post-training class-shift; "
            "empirically the post-training density can be 4-5 orders of magnitude higher than the "
            "random-init claim, producing 22x predicted_band miss and DEFER status per Catalog "
            "#313. The random-init Tier-C does NOT predict post-training class membership because "
            "training pulls the parameter manifold INTO a within-class basin even when the random "
            "init occupies an across-class shell."
        ),
        forbidden_pattern_predicate=(
            "Substrate recipe declares predicted_band field AND Tier-C density is the source of "
            "the band AND the Tier-C measurement was performed on random_init pre-training archive "
            "AND the recipe lacks predicted_band_validation_status: post_training_* OR pending_"
            "post_training waiver per Catalog #324"
        ),
        falsification_band={
            "post_training_to_random_init_density_amplification_min": 1000.0,
            "post_training_to_random_init_density_amplification_max": 100000.0,
            "post_training_to_random_init_density_amplification_observed": 36400.0,
            "predicted_band_miss_ratio_min": 5.0,
            "predicted_band_miss_ratio_observed": 22.0,
        },
        recurrence_conditions=(
            "any_substrate_recipe_declaring_predicted_band_derived_from_random_init_tier_c_without_post_training_validation_status",
            "any_substrate_recipe_treating_random_init_tier_c_density_as_post_training_class_shift_predictor",
            "any_substrate_design_memo_claiming_class_shift_at_pre_training_without_post_training_re_measurement_pinned_reactivation_criterion",
        ),
        canonical_source_anchor=(
            "Catalog #324 META-fix landing memo + 2026-05-20T02:24:31Z probe_outcomes ledger "
            "DEFER verdict on C6 IBPS landed archive be06a4b0972e6c (density 0.9711 WITHIN_CLASS "
            "vs random-init density 2.67e-5 ACROSS_CLASS claim; 36,400x amplification)"
        ),
        canonical_unwind_path=(
            "Per Catalog #324 META-fix: every substrate recipe with predicted_band field MUST "
            "declare predicted_band_validation_status in {validated_post_training (with artifact "
            "path), pending_post_training (with reactivation criteria pinned), research_only=true, "
            "dispatch_enabled=false}. Per Catalog #325: every paid-dispatch substrate needs a "
            "per-substrate symposium per the 6-step canonical contract. Per CLAUDE.md 'Forbidden "
            "premature KILL': implementation-level falsification per Catalog #307 does NOT kill "
            "the paradigm; the next reactivation routes per probe_outcomes reactivation_queue "
            "(Path B2 DreamerV3 RSSM categorical posterior first per the C6 IBPS reactivation row)."
        ),
        canonical_producers=(
            "tools/mdl_scorer_conditional_ablation.py",
            "tac.analysis.scorer_conditional_mdl",
            "tac.optimization.tier_c_density_post_training_validator",
        ),
        canonical_consumers=(
            "tools/audit_predicted_band_provenance.py",
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
            "tools/operator_authorize.py",
        ),
        paradigm_class=cap.PARADIGM_DIAGNOSIS,
        severity=cap.SEVERITY_HIGH,
        provenance=ap_provenance,
        empirical_falsifications=(empirical_falsification,),
        last_recalibration_utc=measurement_utc,
        next_recalibration_trigger=cap.RECALIBRATE_ON_NEW_FALSIFICATIONS,
    )

    return cap.register_anti_pattern(
        anti_pattern,
        agent="claude",
        subagent_id="wave_n35_c6_ibps_landed_archive_tier_c_re_measurement_20260528",
        notes=(
            "Wave N+35 sister canonical anti-pattern registration. The Catalog #324 META-fix "
            "(landed 2026-05-17) structurally extincts this anti-pattern at the recipe-emit "
            "surface; this registration adds the canonical anti-pattern entry per Catalog #344 + "
            "operator 2026-05-28 'memos must be acted upon' canonical apparatus mutation enforcement "
            "standing directive. EmpiricalFalsification captures the 2026-05-20 22x predicted_band "
            "miss + 36,400x density amplification on C6 IBPS landed archive be06a4b0972e6c as "
            "IMPLEMENTATION_LEVEL_CONFIRMATION per Catalog #307 (paradigm intact; specific "
            "configuration falsified)."
        ),
    )


if __name__ == "__main__":
    print("[wave-n+35] Registering canonical equation c6_ibps_post_training_tier_c_within_class_density_v1...")
    equation = register_equation()
    print(f"  ✓ equation_id={equation.equation_id}")
    print(f"  ✓ empirical_anchors={len(equation.empirical_anchors)}")
    print(f"  ✓ canonical_producers={len(equation.canonical_producers)}")
    print(f"  ✓ canonical_consumers={len(equation.canonical_consumers)}")
    print()
    print("[wave-n+35] Registering canonical anti-pattern random_init_tier_c_density_predicts_post_training_class_shift_v1...")
    anti_pattern = register_anti_pattern()
    print(f"  ✓ anti_pattern_id={anti_pattern.anti_pattern_id}")
    print(f"  ✓ empirical_falsifications={len(anti_pattern.empirical_falsifications)}")
    print(f"  ✓ paradigm_class={anti_pattern.paradigm_class}")
    print(f"  ✓ severity={anti_pattern.severity}")
    print()
    print("[wave-n+35] Canonical apparatus mutation COMPLETE per RANK 4 highest-EV-shortest-WC slot directive.")
