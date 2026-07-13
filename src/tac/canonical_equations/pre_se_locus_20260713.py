# SPDX-License-Identifier: MIT
"""Canonical successor laws for the Round-5 PRE-SE feature-locus probe."""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.replace_round5_deeper_nonlinear_20260713 import (
    post_se_sparse_teacher_economics,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "pre_se_locus_tileability_and_localization_v1"
DAG_FEED = ".omx/research/pre_se_locus_DAG_FEED_20260713.md"
AXIS = "[macOS-CPU advisory; NumPy-fp64 convex; CPU-torch exact costates]"
RECEIPT = "experiments/results/pre_se_locus_20260713/receipt.json"
RECEIPT_SHA256 = "660a5763831539715d8593df0ba40a0f50f660af93c0e5bcd1d399ea340d1abb"
MEASUREMENT_UTC = "2026-07-13T22:09:26.355160Z"


def strict_tileability_from_global_dependencies(
    *, upstream_global_reductions: int, own_global_reduction_applied: bool
) -> dict[str, bool | int]:
    """A cut is strictly RGB-tileable only when no global dependency precedes it."""

    if (
        isinstance(upstream_global_reductions, bool)
        or not isinstance(upstream_global_reductions, int)
        or upstream_global_reductions < 0
    ):
        raise ValueError("upstream_global_reductions must be a nonnegative integer")
    if not isinstance(own_global_reduction_applied, bool):
        raise ValueError("own_global_reduction_applied must be boolean")
    count = upstream_global_reductions + int(own_global_reduction_applied)
    return {
        "upstream_global_reductions": upstream_global_reductions,
        "own_global_reduction_applied": own_global_reduction_applied,
        "global_dependency_count": count,
        "strict_end_to_end_independently_tileable_from_rgb": count == 0,
    }


def build_pre_se_locus_tileability_and_localization_v1() -> CanonicalEquation:
    """Build the structural tileability law and its n600 retained-mass anchor."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=DAG_FEED,
        reactivation_criteria=(
            "recompute for a deep extractor with zero upstream global reductions, a charged "
            "full-frame gate-donation scheme, a multi-source/dense-label localizer, or a "
            "different replay distribution"
        ),
        measurement_axis=AXIS,
        hardware_substrate="apple_macos_arm64_cpu_numpy_fp64_and_cpu_torch_teacher",
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="pre_se_locus_v9_n600_seed455_20260713",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "n_states": 600,
            "inherited_exact_train_targets": 480,
            "fresh_exact_heldout_targets": 120,
            "selected_prefix_cells_per_state": 2311,
            "prefix_cells_per_state": 49152,
            "realized_area_fraction": 0.047017415364583336,
            "block2_feature_count": 188,
            "block3_feature_count": 332,
            "ordered_class_pair_count": 20,
            "nonlinear_seeds": [455, 456, 457],
            "receipt_sha256": RECEIPT_SHA256,
        },
        predicted_output={
            "heldout_retained_mass_fraction_minimum": 0.47,
            "strict_end_to_end_tileability_required": True,
        },
        empirical_output={
            "verdict": "WIDER-FAMILY-KILL",
            "verdict_scope": (
                "FAMILY x TESTED-SINGLE-SOURCE-LOCI x FIXED-REPLAY x "
                "STRICT-END-TO-END-RGB-TILEABILITY"
            ),
            "block2_convex_retained_mass_fraction": 0.20233024422907497,
            "block2_nonlinear_retained_mass_fraction": 0.2736871496424692,
            "block3_convex_retained_mass_fraction": 0.09314654496850622,
            "block3_nonlinear_retained_mass_fraction": 0.31323809443347944,
            "oracle_retained_mass_fraction": 0.5278150212253758,
            "block2_cut_fraction": 0.03785634855148739,
            "block3_cut_fraction": 0.0670083252029248,
            "block2_upstream_global_reductions": 4,
            "block3_upstream_global_reductions": 7,
            "own_se_applied_to_capture": False,
            "block2_strict_tileability": False,
            "block3_strict_tileability": False,
            "campaign_honest_teacher_starts": 600,
            "teacher_retries": 0,
            "score_claim": False,
            "pointer_moved": False,
        },
        residual=0.15676190556652054,
        source_artifact=RECEIPT,
        measurement_method=(
            "preregistered fixed n600 replay; immutable Round-5 exact train-target reuse; "
            "fresh untouched 120-state exact CPU-SegNet heldout costates; separate last-MBConv "
            "block2/block3 SE-forward-pre charts; twenty pair-specific RankRLS Moore-Penrose "
            "heads and three deterministic pair-gated MLP seeds per locus"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="PRE-SE strict tileability and single-source localization",
        one_line_summary=(
            "A pre-own-SE tap is RGB-tileable only with zero upstream global reductions; "
            "block2/block3 have 4/7 and all four localizer rungs miss the 0.47 mass gate."
        ),
        latex_form=(
            r"T_{strict}(c)=\mathbb{1}[N_{global}^{up}(c)+N_{global}^{own}(c)=0],"
            r"\qquad \rho_k=\frac{\sum_{i\in\operatorname{TopK}(s,k)}\|\lambda_i\|_2^2}"
            r"{\sum_i\|\lambda_i\|_2^2}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.pre_se_locus_20260713:"
            "strict_tileability_from_global_dependencies"
        ),
        domain_of_validity={
            "scope_level": (
                "family x tested-single-source-loci x fixed replay x strict RGB tileability"
            ),
            "included": [
                "fixed V9 n600 seed455 replay with the Round-5 split and exact targets",
                "last-MBConv block2/block3 SE-forward-pre inputs",
                "twenty pair-specific convex heads and three-seed width-32 MLP per locus",
                "strict end-to-end tileability from the RGB input",
            ],
            "excluded": [
                "SE-free or local-attention deep extractors",
                "charged cached/donated global-gate broadcasts",
                "multi-source, dense-label, or larger attention localizers",
                "other replay distributions or seeds",
                "evaluator score, archive bytes, contest CPU/CUDA, or promotion authority",
            ],
            "research_only": True,
            "review_status": "self-audited-UNREVIEWED_BY_MAIN",
            "authority": AXIS,
        },
        units_in={
            "global_dependency_count": "count",
            "costate_mass": "squared SegNet input-gradient units",
        },
        units_out={
            "strict_tileability": "boolean",
            "retained_mass_fraction": "dimensionless",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "best_pre_se_retained_mass_shortfall": 0.15676190556652054,
            "block2_global_dependency_excess": 4,
            "block3_global_dependency_excess": 7,
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.scorer_surrogate.pre_se_locus_20260713",
            "tools.probe_pre_se_locus_20260713",
            "tac.witness_dsl.pre_se_locus_policy_20260713",
        ),
        canonical_producers=("tools.probe_pre_se_locus_20260713",),
        provenance=provenance,
    )


def populate_pre_se_locus_tileability_and_localization_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Append through the locked helper; main review owns shared registration."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_pre_se_locus_tileability_and_localization_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="pre-se-locus; strict-tileability; wider-family-kill; research-only",
    )
    return equation


__all__ = [
    "AXIS",
    "DAG_FEED",
    "EQUATION_ID",
    "MEASUREMENT_UTC",
    "RECEIPT",
    "RECEIPT_SHA256",
    "build_pre_se_locus_tileability_and_localization_v1",
    "populate_pre_se_locus_tileability_and_localization_v1",
    "post_se_sparse_teacher_economics",
    "strict_tileability_from_global_dependencies",
]
