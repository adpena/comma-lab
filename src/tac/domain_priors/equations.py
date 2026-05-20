# SPDX-License-Identifier: MIT
"""Canonical equations for tac.domain_priors — Catalog #344 registrations.

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 4 Step 4.2: this module registers 3
canonical equations in the :mod:`tac.canonical_equations` registry per
Catalog #344, codifying the domain-prior predictions so future agents +
cathedral autopilot + research subagents can audit residuals + trigger
auto-recalibration via canonical helpers (not tribal-knowledge prose).

The 3 equations:

  1. ``per_frame_difficulty_atlas_v1`` — per-pair → per-frame difficulty
     projection. Validates against future per-frame breakdowns from
     ``upstream/evaluate.py`` with per-frame mode flag (operator-routable
     next-wave dispatch).

  2. ``ego_motion_concentration_prior_v1`` — Atick-Redlich 1990 +
     Rao-Ballard 1999 cooperative-receiver / predictive-coding alignment
     applied to dashcam pose-vector or affine-flow per-pair signal.
     Validates against Comma2k19 + contest-video ego-motion distribution
     comparison.

  3. ``per_segnet_class_chroma_priors_v1`` — SegNet 5-class per-class
     pixel-count + chroma-variance + motion-magnitude prior. Validates
     against the canonical openpilot mask prior contract. Sister of
     Catalog #354 exploit #5 ``per_segnet_class_chroma_consumer``.

Each equation has empty initial empirical_anchors (no contest dispatch
yet) AND non-empty canonical_producers + canonical_consumers per the
orphan-equation invariant.

Cross-references:
  * CATHEDRAL-SMARTER-DESIGN-MEMO Dim 4 Step 4.2 (lines 409-413)
  * Catalog #344 ``check_empirical_finding_memo_references_canonical_equation``
  * Catalog #323 canonical Provenance umbrella
  * Catalog #354 master-gradient exploit consumer bundle (sister consumers)
  * Catalog #213 Comma2k19 canonical local-chunk cache
  * Catalog #209 contest-video-leakage non-negotiable
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    CanonicalEquation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
    RECALIBRATE_ON_RESIDUAL_DRIFT,
)
from tac.canonical_equations.registry import register_canonical_equation
from tac.provenance.builders import build_provenance_for_predicted


DOMAIN_PRIORS_EQUATION_IDS: tuple[str, ...] = (
    "per_frame_difficulty_atlas_v1",
    "ego_motion_concentration_prior_v1",
    "per_segnet_class_chroma_priors_v1",
)
"""Canonical equation IDs registered by this module per Catalog #344."""


_DOMAIN_PRIORS_DESIGN_PLACEHOLDER_SHA = "0" * 64


def _design_provenance(model_id: str) -> Any:
    """Build a PREDICTED Provenance for design-only domain-prior equation registration.

    Mirrors ``tac.canonical_equations.builtins._design_provenance`` so
    every domain-prior equation registers with a non-promotable PREDICTED
    grade until a paired CUDA+CPU anchor lands per CLAUDE.md "Submission
    auth eval — BOTH CPU AND CUDA" non-negotiable.
    """
    return build_provenance_for_predicted(
        model_id=model_id,
        inputs_sha256=_DOMAIN_PRIORS_DESIGN_PLACEHOLDER_SHA,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
    )


def build_per_frame_difficulty_atlas_v1() -> CanonicalEquation:
    """Equation 1: per-pair → per-frame difficulty projection formula.

    The canonical formula (mean-over-incident-pairs aggregator, the default):

        difficulty_frame(t) = (1 / |incident_pairs(t)|) * sum_{p in incident_pairs(t)} difficulty_pair(p)

    where ``incident_pairs(t)`` is the set of pair indices the frame
    participates in (canonical adjacent construction: {t-1, t} for
    interior frames, {0} or {N-1} for boundary frames).

    The 3 valid aggregator operators are documented in
    :data:`tac.domain_priors.PER_FRAME_DIFFICULTY_AGGREGATOR_VALID`
    (mean / max / sum). Auto-recalibration via the canonical equations
    registry refits the choice once paired CUDA+CPU anchors land with
    per-frame breakdowns.

    Per the CARGO-CULT audit in CATHEDRAL-SMARTER-DESIGN-MEMO Dim 4:
    "Per-pair → per-frame aggregation requires choice of aggregation
    operator" → CARGO-CULTED-PENDING-EMPIRICAL. This equation registers
    the mean aggregator as the design-time choice; empirical anchors
    will validate / refute.
    """
    return CanonicalEquation(
        equation_id="per_frame_difficulty_atlas_v1",
        name="Per-frame difficulty atlas — per-pair → per-frame projection",
        one_line_summary=(
            "Project per-pair master-gradient difficulty to per-frame via "
            "mean-over-incident-pairs aggregation (canonical default)."
        ),
        latex_form=(
            r"D_{\text{frame}}(t) = \frac{1}{|P_t|} \sum_{p \in P_t} D_{\text{pair}}(p), "
            r"\quad P_t = \{t-1, t\} \cap [0, N_{\text{pairs}})"
        ),
        python_callable_module_path=(
            "tac.domain_priors.per_frame_difficulty:build_per_frame_difficulty_from_per_pair_atlas"
        ),
        domain_of_validity={
            "video_total_frames_range": [2, 2400],
            "pair_construction": ["adjacent", "non_overlap"],
            "aggregator_operators": [
                "mean_over_incident_pairs",
                "max_over_incident_pairs",
                "sum_over_incident_pairs",
            ],
        },
        units_in={
            "per_pair_difficulty_vector": "non_negative_float_per_pair_gradient_l2_norm",
            "archive_sha256": "hex_string_64_chars",
        },
        units_out={
            "per_frame_difficulty": "non_negative_float_per_frame_aggregated_gradient_norm",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-05-20T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_consumers.per_pair_difficulty_atlas_consumer",
            "tac.cathedral_consumers.bit_allocator_per_pair_consumer",
            "tools.cathedral_autopilot_autonomous_loop:invoke_cathedral_consumers_on_candidates",
        ),
        canonical_producers=(
            "tac.domain_priors.per_frame_difficulty:build_per_frame_difficulty_from_per_pair_atlas",
            "tac.master_gradient_consumers:per_pair_difficulty_atlas",
        ),
        provenance=_design_provenance("tac.domain_priors.per_frame_difficulty_atlas.v1"),
    )


def build_ego_motion_concentration_prior_v1() -> CanonicalEquation:
    """Equation 2: ego-motion concentration prior (Atick-Redlich / Rao-Ballard).

    Per Atick-Redlich 1990 cooperative-receiver framing + Rao-Ballard 1999
    predictive-coding hierarchical-Bayesian structure: ego-motion magnitude
    + concentration drives the canonical cooperative-receiver signal-axis
    decomposition for dashcam video.

    Pose-anchor magnitude formula:

        magnitude_pair(p) = ||pose_vector(p)||_2  (canonical 6-DOF)
        magnitude_frame(t) = mean over incident pairs

    Affine-flow concentration formula:

        concentration_pair(p) = (rotation+skew) / (translation+rotation+skew)
        score_frame(t) = magnitude * (1 - 0.5 * concentration)

    Per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand
    council symposium" Catalog #311: ego-motion-conditioning is the
    canonical predictive-coding form for dashcam video. The Z6 / Z7 / Z8
    substrate design memos all bind ego-motion to the predictive-coding
    lineage.

    Initial design choice (cargo-cult-audit-marked HARD-EARNED per Dim 4
    audit): the Atick-Redlich cooperative-receiver alignment IS the
    canonical framing. Validation against comma2k19 vs contest-video
    ego-motion distribution comparison is the next-wave anchor.
    """
    return CanonicalEquation(
        equation_id="ego_motion_concentration_prior_v1",
        name="Ego-motion concentration prior — Atick-Redlich / Rao-Ballard alignment",
        one_line_summary=(
            "Per-frame ego-motion magnitude + flow-concentration from per-pair pose / "
            "affine-flow rows; aligns with Atick-Redlich / Rao-Ballard predictive coding."
        ),
        latex_form=(
            r"S_{\text{ego}}(t) = M(t) \cdot (1 - \tfrac{1}{2} C(t)), \quad "
            r"M(t) = \frac{1}{|P_t|}\sum_{p \in P_t} \|\theta_{\text{pose}}(p)\|_2"
        ),
        python_callable_module_path=(
            "tac.domain_priors.ego_motion_concentration:build_ego_motion_concentration_from_pose_anchors"
        ),
        domain_of_validity={
            "video_kind": ["dashcam_forward_facing"],
            "pose_dof": [6],
            "anchor_kinds": ["pose_vector", "affine_flow", "raft_optical_flow"],
            "predictive_coding_lineage": [
                "atick_redlich_1990",
                "rao_ballard_1999",
                "tishby_zaslavsky_2015",
            ],
        },
        units_in={
            "pose_vector_per_pair": "float_6_dof_pose_change_rad_and_meters",
            "affine_flow_per_pair": "float_6_element_2x3_affine_matrix_a_b_tx_c_d_ty",
        },
        units_out={
            "ego_motion_score_per_frame": "non_negative_float_dimensionless",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-05-20T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_consumers.per_pair_lora_supervision_signal_consumer",
            "tools.cathedral_autopilot_autonomous_loop:invoke_cathedral_consumers_on_candidates",
        ),
        canonical_producers=(
            "tac.domain_priors.ego_motion_concentration:build_ego_motion_concentration_from_pose_anchors",
            "tac.domain_priors.ego_motion_concentration:build_ego_motion_concentration_from_affine_flow",
            "tac.ego_flow",
        ),
        provenance=_design_provenance("tac.domain_priors.ego_motion_concentration_prior.v1"),
    )


def build_per_segnet_class_chroma_priors_v1() -> CanonicalEquation:
    """Equation 3: SegNet 5-class per-class pixel + chroma + motion priors.

    Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 4 line 412: per-class statistical
    priors compose with the canonical openpilot mask prior contract +
    Catalog #354 exploit #5 ``per_segnet_class_chroma_consumer`` to drive
    per-class chroma allocation in the bit-allocator.

    Canonical per-class triple:

        prior_class(c) = (pixel_count_fraction(c), chroma_variance(c), motion_magnitude_mean(c))

    Sum invariant:

        sum_{c in [0, 5)} pixel_count_fraction(c) == 1.0

    The 5 SegNet classes:

        0: background_sky_road
        1: vehicle
        2: pedestrian
        3: lane_marking
        4: other_foreground

    Per Dim 4 cargo-cult audit: per-class statistical priors compose with
    per-class chroma allocation per Catalog #354 exploit #5 IS HARD-EARNED.
    The canonical mechanism is already wired; this equation formalizes the
    prior into a typed contract.
    """
    return CanonicalEquation(
        equation_id="per_segnet_class_chroma_priors_v1",
        name="Per-SegNet-class chroma priors — pixel + chroma + motion triple",
        one_line_summary=(
            "Per-SegNet-class (5-class) pixel_count_fraction + chroma_variance + "
            "motion_magnitude triple; drives per-class chroma allocation."
        ),
        latex_form=(
            r"\text{prior}_c = (f_c, \sigma^2_{\text{chroma}, c}, |\bar{m}_c|), "
            r"\quad \sum_{c=0}^{4} f_c = 1, \quad c \in \{0, 1, 2, 3, 4\}"
        ),
        python_callable_module_path=(
            "tac.domain_priors.per_class_statistical:build_per_class_statistical_priors_from_scorer_output"
        ),
        domain_of_validity={
            "scorer_class_count": [5],
            "scorer_kind": [
                "segnet_5_class_argmax",
                "segnet_5_class_softmax",
                "openpilot_mask_prior",
            ],
            "video_kind": ["dashcam_forward_facing"],
            "canonical_class_names": [
                "background_sky_road",
                "vehicle",
                "pedestrian",
                "lane_marking",
                "other_foreground",
            ],
        },
        units_in={
            "pixel_count_fractions": "fraction_per_class_summing_to_1",
            "chroma_variances": "variance_per_class_non_negative_float",
            "motion_magnitudes": "mean_pose_magnitude_per_class_non_negative_float",
        },
        units_out={
            "per_class_prior_triple": "tuple_of_5_PerClassPrior_dataclass_rows",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-05-20T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_consumers.per_segnet_class_chroma_consumer",
            "tac.cathedral_consumers.bit_allocator_per_pair_consumer",
            "tools.cathedral_autopilot_autonomous_loop:invoke_cathedral_consumers_on_candidates",
        ),
        canonical_producers=(
            "tac.domain_priors.per_class_statistical:build_per_class_statistical_priors_from_scorer_output",
            "tac.categorical_substrate",
            "tac.categorical_label_atoms",
        ),
        provenance=_design_provenance("tac.domain_priors.per_segnet_class_chroma_priors.v1"),
    )


def build_all_domain_prior_equations() -> list[CanonicalEquation]:
    """Build all 3 canonical domain-prior equations as a list."""
    return [
        build_per_frame_difficulty_atlas_v1(),
        build_ego_motion_concentration_prior_v1(),
        build_per_segnet_class_chroma_priors_v1(),
    ]


def register_domain_prior_canonical_equations(
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
    notes: str | None = None,
) -> list[CanonicalEquation]:
    """Register all 3 canonical domain-prior equations.

    Idempotent: equations are keyed by ``equation_id`` and the registry's
    APPEND-ONLY semantics mean re-running this helper appends additional
    ``registered`` events (the canonical re-registration trail per
    Catalog #110/#113 HISTORICAL_PROVENANCE).

    Args:
        path: optional registry path override; defaults to canonical
            ``.omx/state/canonical_equations_registry.jsonl``.
        lock_path: optional lock path override.
        agent: optional agent identifier for the registration event row.
        subagent_id: optional subagent identifier.
        notes: optional notes for the registration event row.

    Returns:
        List of the 3 registered :class:`CanonicalEquation` instances (in
        canonical order: per_frame / ego_motion / per_class).
    """
    equations = build_all_domain_prior_equations()
    for eq in equations:
        register_canonical_equation(
            eq,
            path=path,
            lock_path=lock_path,
            agent=agent or "tac.domain_priors",
            subagent_id=subagent_id,
            notes=notes or "Catalog #344 domain-prior canonical-equation registration (Step 4.2)",
        )
    return equations


__all__ = [
    "DOMAIN_PRIORS_EQUATION_IDS",
    "build_per_frame_difficulty_atlas_v1",
    "build_ego_motion_concentration_prior_v1",
    "build_per_segnet_class_chroma_priors_v1",
    "build_all_domain_prior_equations",
    "register_domain_prior_canonical_equations",
]
