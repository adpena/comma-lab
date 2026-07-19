# SPDX-License-Identifier: MIT
"""Canonical right-level coordinate/quotient law for evaluator-equivalent witnesses.

This module is the equations leg of the 2026-07-18 right-problem synthesis.  It
does not introduce a score, a measurement, or a new empirical fit.  It composes
five existing canonical laws into one fail-closed optimization contract:

* represent spatial state by Morse--Smale / winner-rival hyperplane cells,
  their separatrices, and their tie loci;
* represent predictable pair motion by ``xi`` and count every video-derived
  gauge/event payload;
* make the remaining trainable coordinate an explicit class/cell quotient
  ``T[class_id, cell_id]``;
* solve or project the coordinates that admit an exact construction before
  training, then train only the residual quotient;
* judge membership only after the exact realized-through-R and uint8-lattice
  pullback into the frozen evaluator cells.

Generic deterministic generator code is free under the repository's rule-118
accounting convention.  A video-derived seed, weight, codebook, gauge, event,
or quotient payload is always counted.  In particular, this law does *not*
assume that the current FiLM parameterization is isomorphic to the frozen
SegNet head; that relation remains an empirical pullback question.

The research sidecar is deliberately not required to exist at import/build
time.  ``build_provenance_for_research_sidecar`` records a non-promotable
placeholder SHA until the parent landing creates the sidecar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

if TYPE_CHECKING:
    from tac.provenance.contract import Provenance

EQUATION_ID = "witness_right_level_coordinate_quotient_law_v1"
SIDECAR_PATH = ".omx/research/solve_the_right_problem_kolmogorov_sweep_20260718.md"

CALIBRATION_UTC = "2026-07-18T04:04:58Z"
RATE_DENOMINATOR_BYTES = 37_545_489
RATE_WEIGHT_NUMERATOR = 25
SEG_WEIGHT = 100.0
POSE_RADICAND_WEIGHT = 10.0

COMPOSED_EVIDENCE_IDS = (
    "fullstack_unique_home_assignment_v1",
    "witness_general_covariance_totality_v1",
    "optimal_metric_unification_v1",
    "witness_own_residual_decomposition_v1",
    "necessity_generator_seed_dseg_calibration_v1",
)


@dataclass(frozen=True)
class RightLevelCoordinateQuotient:
    """Machine-readable structural state selected by the cross-cut law."""

    spatial_state: tuple[str, ...]
    temporal_coordinate: str
    counted_irregular_coordinates: tuple[str, ...]
    residual_quotient: str
    generic_generator_accounting: str
    video_derived_accounting: str
    solve_project_before_training: tuple[str, ...]
    train_only: str
    realization_authority: tuple[str, ...]
    film_to_frozen_head_relation: str


def _require_int(value: int, *, field_name: str, nonnegative: bool) -> int:
    """Validate byte counts without silently accepting bool or rounded floats."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer byte count")
    if nonnegative and value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def _require_nonnegative_finite(value: float | int, *, field_name: str) -> float:
    """Validate score components before applying the frozen score equation."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    realized = float(value)
    if not math.isfinite(realized):
        raise ValueError(f"{field_name} must be finite")
    if realized < 0.0:
        raise ValueError(f"{field_name} must be >= 0")
    return realized


def rate_score_term_exact(archive_bytes: int) -> Fraction:
    """Return the exact contest rate term ``25 * bytes / 37_545_489``.

    ``Fraction`` is intentional: the byte term has exact rational arithmetic,
    so callers need not lose the frontier-sized delta to premature rounding.
    """

    archive_bytes = _require_int(
        archive_bytes, field_name="archive_bytes", nonnegative=True
    )
    return Fraction(RATE_WEIGHT_NUMERATOR * archive_bytes, RATE_DENOMINATOR_BYTES)


def rate_score_delta_exact(delta_archive_bytes: int) -> Fraction:
    """Return the exact signed score delta caused by an integer byte delta."""

    delta_archive_bytes = _require_int(
        delta_archive_bytes, field_name="delta_archive_bytes", nonnegative=False
    )
    return Fraction(
        RATE_WEIGHT_NUMERATOR * delta_archive_bytes, RATE_DENOMINATOR_BYTES
    )


def contest_score(*, d_seg: float | int, d_pose: float | int, archive_bytes: int) -> float:
    """Evaluate the frozen score formula from realized component values.

    This arithmetic helper confers no authority on its inputs.  Callers must
    supply ``d_seg`` and ``d_pose`` measured through the exact R/uint8/frozen-
    scorer chain on the archive bytes being scored.
    """

    d_seg_f = _require_nonnegative_finite(d_seg, field_name="d_seg")
    d_pose_f = _require_nonnegative_finite(d_pose, field_name="d_pose")
    return (
        SEG_WEIGHT * d_seg_f
        + math.sqrt(POSE_RADICAND_WEIGHT * d_pose_f)
        + float(rate_score_term_exact(archive_bytes))
    )


def right_level_coordinate_quotient_law() -> RightLevelCoordinateQuotient:
    """Return the typed state/accounting/optimization contract.

    The tuple tokens are intentionally stable: planner and regression surfaces
    can inspect the law without parsing prose or LaTeX.
    """

    return RightLevelCoordinateQuotient(
        spatial_state=(
            "morse_smale_cells",
            "winner_rival_hyperplane_cells",
            "separatrices",
            "tie_loci",
        ),
        temporal_coordinate="xi",
        counted_irregular_coordinates=("gauge", "events"),
        residual_quotient="T[class_id,cell_id]",
        generic_generator_accounting=(
            "FREE_ONLY_IF_GENERIC_DETERMINISTIC_AND_NOT_VIDEO_DERIVED"
        ),
        video_derived_accounting=(
            "COUNT_EVERY_VIDEO_DERIVED_SEED_WEIGHT_CODEBOOK_GAUGE_EVENT_OR_T_PAYLOAD"
        ),
        solve_project_before_training=(
            "project_exact_morse_smale_and_hyperplane_cell_constraints",
            "solve_exact_separatrix_and_tie_locus_constraints_where_identified",
            "solve_or_bank_predictable_xi",
        ),
        train_only="residual_class_cell_quotient_after_exact_projection",
        realization_authority=(
            "exact_inflate_bytes",
            "uint8_lattice",
            "exact_R_resize_round_chain",
            "frozen_SegNet_and_PoseNet_cells",
        ),
        film_to_frozen_head_relation=(
            "NOT_ASSUMED_ISOMORPHIC_REQUIRES_EXACT_PULLBACK_EMPIRICAL_GATE"
        ),
    )


def _sidecar_provenance(reactivation_criteria: str) -> Provenance:
    return build_provenance_for_research_sidecar(
        sidecar_path=SIDECAR_PATH,
        reactivation_criteria=reactivation_criteria,
        measurement_axis="[DERIVED source-inspected composition; no new measurement]",
        hardware_substrate="not_applicable_structural_composition",
        captured_at_utc=CALIBRATION_UTC,
    )


def build_witness_right_level_coordinate_quotient_law_v1() -> CanonicalEquation:
    """Build the non-promotable canonical equation for the right-level law."""

    law = right_level_coordinate_quotient_law()
    anchor = EmpiricalAnchor(
        anchor_id="right_level_quotient_composed_source_inspection_20260718",
        measurement_utc=CALIBRATION_UTC,
        inputs={
            "evidence_ids": list(COMPOSED_EVIDENCE_IDS),
            "evidence_label": "DERIVED_FROM_COMPOSED_EXISTING_ANCHORS",
            "new_measurement": False,
            "new_score_claim": False,
        },
        predicted_output={
            "right_state": list(law.spatial_state),
            "temporal_coordinate": law.temporal_coordinate,
            "counted_irregular_coordinates": list(law.counted_irregular_coordinates),
            "residual_quotient": law.residual_quotient,
            "train_only": law.train_only,
        },
        empirical_output={
            "source_inspection_result": (
                "the five named canonical laws jointly support the structural composition"
            ),
            "evidence_label": "DERIVED_SOURCE_INSPECTION_NOT_A_MEASUREMENT",
            "new_measurement": False,
            "new_score_claim": False,
            "pointer_moved": False,
        },
        residual=0.0,
        source_artifact=SIDECAR_PATH,
        measurement_method=(
            "source inspection and logical composition of five existing canonical equations; "
            "residual=0 is composition closure, not a measured score residual"
        ),
        provenance=_sidecar_provenance(
            "replace composition-only status only after exact through-R/uint8 archive replay "
            "and a registered empirical anchor on the same contest axis"
        ),
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Witness right-level coordinate quotient: solve exact evaluator-cell geometry, "
            "then train only the counted class/cell residual"
        ),
        one_line_summary=(
            "Use cell/separatrix/tie-locus state plus xi; count gauge/events and video payload; "
            "project exact structure, then train only T[class,cell] through exact R/uint8."
        ),
        latex_form=(
            r"z_{right}=(\mathcal M_{MS/H},\partial\mathcal M,\mathcal L_{tie},\xi,q_{g,e}),\ "
            r"W=\mathcal R_{8}\!\left[G_{generic}(z_{right})+T_{c,k}\right],\ "
            r"K(G_{generic})=0,\ K(q_{g,e},seed,weights,T)>0,\ "
            r"\theta_{train}\in\mathcal X/\operatorname{Im}(\Pi_{solve}),\ "
            r"\mathrm{FiLM}\not\cong h_{SegNet}\ \text{without exact pullback evidence}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.witness_right_level_coordinate_quotient_20260718:"
            "right_level_coordinate_quotient_law"
        ),
        domain_of_validity={
            "law_status": "DERIVED_FROM_COMPOSED_EXISTING_ANCHORS",
            "measurement_axis": [
                "DERIVED source-inspected composition; no new measurement"
            ],
            "promotion_eligible": False,
            "score_claim_valid": False,
            "pointer_moved": False,
            "research_only": True,
            "verdict_scope": (
                "STRUCTURAL DESIGN law only; no instance, formulation, family, or paradigm "
                "score verdict"
            ),
            "right_problem_level": [
                "semantic class",
                "Morse-Smale / winner-rival cell",
                "separatrix / tie locus",
                "frame-pair temporal coordinate",
                "archive byte accounting",
            ],
            "required_authority_chain": [
                "exact archive bytes",
                "inflate",
                "uint8 lattice",
                "exact R resize/round chain",
                "frozen SegNet/PoseNet evaluator cells",
            ],
            "charged_free_boundary": {
                "free": "generic deterministic generator program only",
                "counted": (
                    "every video-derived seed, weight, codebook, gauge, event, and T payload"
                ),
            },
            "explicit_non_assumption": (
                "current FiLM is not assumed isomorphic to the frozen SegNet head"
            ),
            "composed_evidence_ids": list(COMPOSED_EVIDENCE_IDS),
            "excluded": [
                "proxy-loss-only acceptance",
                "continuous-field membership before uint8/R realization",
                "free accounting for any video-derived constant",
                "FiLM-to-frozen-head isomorphism without exact empirical pullback",
            ],
        },
        units_in={
            "d_seg": "dimensionless frozen-SegNet cell debt",
            "d_pose": "dimensionless frozen-PoseNet pair debt",
            "archive_bytes": "bytes",
            "state": "class/cell/boundary/pair coordinates",
        },
        units_out={
            "score": "contest score units",
            "rate_delta": "exact rational contest score units",
            "quotient": "counted residual class/cell coordinate",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"structural_composition_closure": 0.0},
        last_calibration_utc=CALIBRATION_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_autoconfig (future right-level projection/train planner)",
            "tac.canonical_equations.registry",
        ),
        canonical_producers=(*COMPOSED_EVIDENCE_IDS, SIDECAR_PATH),
        provenance=_sidecar_provenance(
            "promote only after a concrete implementation is archive-byte-closed and its exact "
            "realized-through-R/uint8 Seg/Pose outcome is registered on a contest authority axis"
        ),
    )


def populate_witness_right_level_coordinate_quotient_law_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the law to a canonical registry without requiring the sidecar."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_witness_right_level_coordinate_quotient_law_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "right-level coordinate/quotient structural law; DERIVED from five existing "
            "canonical equations; no new measurement, score claim, or pointer movement"
        ),
    )
    return equation


__all__ = [
    "CALIBRATION_UTC",
    "COMPOSED_EVIDENCE_IDS",
    "EQUATION_ID",
    "POSE_RADICAND_WEIGHT",
    "RATE_DENOMINATOR_BYTES",
    "RATE_WEIGHT_NUMERATOR",
    "SEG_WEIGHT",
    "SIDECAR_PATH",
    "RightLevelCoordinateQuotient",
    "build_witness_right_level_coordinate_quotient_law_v1",
    "contest_score",
    "populate_witness_right_level_coordinate_quotient_law_equation",
    "rate_score_delta_exact",
    "rate_score_term_exact",
    "right_level_coordinate_quotient_law",
]
