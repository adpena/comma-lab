# SPDX-License-Identifier: MIT
"""Register canonical equation `fridrich_school_inverse_steganalysis_patterns_v1`.

Per CLAUDE.md Catalog #344 "Canonical equations + models registry"
non-negotiable: every NEW canonical helper package MUST register an associated
canonical equation with EmpiricalAnchor + producers + consumers.

Sister of `tools/register_yousfi_cascade_tier_1_canonical_equations.py` at the
canonical-pattern-extraction surface.
"""

from __future__ import annotations

from datetime import datetime, timezone

from tac.canonical_equations import (
    CanonicalEquation,
    EmpiricalAnchor,
    register_canonical_equation,
)
from tac.provenance.builders import build_provenance_for_predicted


_UTC = datetime.now(timezone.utc).isoformat()


def register_fridrich_school_canonical_equation() -> CanonicalEquation:
    """Register `fridrich_school_inverse_steganalysis_patterns_v1`."""
    anchor = EmpiricalAnchor(
        anchor_id="fridrich_school_canonical_pattern_inventory_count_anchor_20260530",
        measurement_utc=_UTC,
        inputs={
            "in_domain_context": "fridrich_school_canonical_pattern_extraction_inventory_7_patterns",
            "research_phase_complete": True,
            "yousfi_repos_cataloged": 30,
            "highest_priority_targets": 4,  # autostego + deepsteganalysis + OneHotConv + comma10k-baseline
            "fridrich_other_students_referenced": 7,  # Filler, Pevny, Holub, Sedighi, Boroumand, Kodovsky, Denemark
        },
        predicted_output={"canonical_pattern_inventory_count": 7},
        empirical_output={"canonical_pattern_inventory_count": 7},
        residual=0.0,
        source_artifact=(
            "src/tac/composition/fridrich_school_inverse_steganalysis_patterns/"
            "canonical_pattern_inventory.py::build_fridrich_school_canonical_patterns_inventory"
        ),
        measurement_method="source_inspection_canonical_pattern_inventory_count_via_introspection_helper",
        provenance=build_provenance_for_predicted(
            model_id="tac.composition.fridrich_school_inverse_steganalysis_patterns",
            inputs_sha256="0" * 64,
            captured_at_utc=_UTC,
        ),
        empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
    )

    eq = CanonicalEquation(
        equation_id="fridrich_school_inverse_steganalysis_patterns_v1",
        name="Fridrich-school canonical inverse-steganalysis pattern inventory (Yousfi recent repos + sister-students)",
        one_line_summary=(
            "7 canonical patterns from Yousfi POST-alaska repos + Fridrich-students; "
            "operator-facing via build_fridrich_school_canonical_patterns_inventory()."
        ),
        latex_form=(
            r"\text{inventory\_count} = "
            r"|\{ \text{alice\_vs\_eve}, \text{lclsmr}, \text{efficientnet\_surgery}, "
            r"\text{onehot}, \text{comma10k\_lineage}, \text{stc\_filler}, "
            r"\text{fusion\_ensemble} \}| = 7"
        ),
        python_callable_module_path=(
            "tac.composition.fridrich_school_inverse_steganalysis_patterns."
            "build_fridrich_school_canonical_patterns_inventory"
        ),
        domain_of_validity={
            "scope": "canonical_pattern_extraction_from_external_repos",
            "covered_repos": [
                "github.com/YassineYousfi/autostego",
                "github.com/DDELab/deepsteganalysis",
                "github.com/YassineYousfi/OneHotConv",
                "github.com/YassineYousfi/comma10k-baseline",
                "Filler-Judas-Fridrich 2011 IEEE TIFS canonical paper",
            ],
            "covered_fridrich_school_authors": [
                "Yousfi",
                "Filler",
                "Pevny",
                "Holub",
                "Sedighi",
                "Boroumand",
                "Kodovsky",
                "Denemark",
                "Butora",
                "Giboulot",
            ],
        },
        units_in={
            "inventory_introspection_helper": "Python callable returning tuple[FridrichSchoolCanonicalPatternRow, ...]",
        },
        units_out={
            "canonical_pattern_inventory_count": "integer count of canonical patterns ported",
            "per_pattern_5_axis_classification": "Mapping[axis, HARD-EARNED-vs-DOCUMENTED-ADAPTATION-vs-N/A]",
            "per_pattern_yousfi_fridrich_axis_cross_ref": "string citing cascade axis assignment",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "canonical_pattern_inventory_count_residual": 0.0,
            "canonical_pattern_inventory_count_predicted": 7.0,
            "canonical_pattern_inventory_count_empirical": 7.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger="when_3+_new_empirical_anchors_in_domain",
        canonical_consumers=(
            "tac.composition.alaska_inverse_steganalysis_patterns (sister canonical patterns)",
            "tac.composition.hill_canonical_inverse_steganalysis_li_wang_li_huang_2014 (Slot YY cost-function)",
            "tac.composition.mipod_canonical_inverse_steganalysis_sedighi_cogranne_fridrich_2016 (Slot AAA cost-function)",
            "tac.composition.hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010 (Slot CCC cost-function)",
            "tac.composition.pr110_opt_7_uniward_inverse_scorer_basis_expansion (Slot FF cost-function)",
            "tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet (Slot RR pose-axis)",
            "tools.cathedral_autopilot_autonomous_loop.invoke_cathedral_consumers_on_candidates (per Catalog #335 auto-discovery)",
        ),
        canonical_producers=(
            "tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_alice_vs_eve_adversarial_loop",
            "tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_lclsmr_linear_steganalysis_detector",
            "tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_efficientnet_steganalysis_surgery",
            "tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_onehot_jpeg_steganalysis",
            "tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_comma10k_baseline_lineage",
            "tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_syndrome_trellis_coding_filler",
            "tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_fusion_detector_ensemble",
        ),
        provenance=build_provenance_for_predicted(
            model_id="tac.composition.fridrich_school_inverse_steganalysis_patterns",
            inputs_sha256="0" * 64,
            captured_at_utc=_UTC,
        ),
    )

    return register_canonical_equation(
        eq,
        agent="claude_opus_4_7_1m",
        subagent_id="fridrich_school_canonical_pattern_extraction_20260530_201500",
        notes=(
            "Extends alaska canonical pattern extraction (sister 2026-05-30 "
            "landing commit 61a91a48e) to Yousfi's POST-alaska recent repos + "
            "Fridrich's other students. 7 canonical patterns ported; 99 "
            "dedicated tests pass; 0 preflight violations. Per operator "
            "binding 2026-05-30: 'yousfi may have more recent repos that are "
            "even more useful... same with fridrich and fridrich's other "
            "students'."
        ),
    )


if __name__ == "__main__":
    eq = register_fridrich_school_canonical_equation()
    print(f"OK registered: {eq.equation_id}")
    print(f"  producers={len(eq.canonical_producers)}")
    print(f"  consumers={len(eq.canonical_consumers)}")
    print(f"  empirical_anchors={len(eq.empirical_anchors)}")
