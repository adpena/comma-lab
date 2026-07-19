# SPDX-License-Identifier: MIT
"""Focused tests for ``witness_right_level_coordinate_quotient_law_v1``.

The tests lock the exact rate arithmetic, fail-closed byte validation, typed
right-level state, charged/free boundary, exact realization gate, honest
composition label, missing-sidecar build behavior, and registry round trip.
They perform no scorer invocation and make no score claim.
"""
from __future__ import annotations

import json
from fractions import Fraction

import pytest

from tac.canonical_equations import witness_right_level_coordinate_quotient_20260718 as law_module
from tac.canonical_equations.equation import (
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
)
from tac.canonical_equations.registry import get_equation_by_id
from tac.canonical_equations.witness_right_level_coordinate_quotient_20260718 import (
    COMPOSED_EVIDENCE_IDS,
    EQUATION_ID,
    RATE_DENOMINATOR_BYTES,
    RightLevelCoordinateQuotient,
    build_witness_right_level_coordinate_quotient_law_v1,
    contest_score,
    populate_witness_right_level_coordinate_quotient_law_equation,
    rate_score_delta_exact,
    rate_score_term_exact,
    right_level_coordinate_quotient_law,
)


def test_exact_rate_term_and_signed_delta() -> None:
    assert rate_score_term_exact(0) == Fraction(0, 1)
    assert rate_score_term_exact(RATE_DENOMINATOR_BYTES) == Fraction(25, 1)
    assert rate_score_term_exact(1) == Fraction(25, RATE_DENOMINATOR_BYTES)
    assert rate_score_delta_exact(-1) == Fraction(-25, RATE_DENOMINATOR_BYTES)


@pytest.mark.parametrize("bad", [True, 1.0, "1", None])
def test_rate_helpers_refuse_non_integer_byte_counts(bad: object) -> None:
    with pytest.raises(TypeError):
        rate_score_term_exact(bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        rate_score_delta_exact(bad)  # type: ignore[arg-type]


def test_rate_term_refuses_negative_archive_size_but_delta_is_signed() -> None:
    with pytest.raises(ValueError):
        rate_score_term_exact(-1)
    assert rate_score_delta_exact(-RATE_DENOMINATOR_BYTES) == -25


def test_contest_score_uses_frozen_formula() -> None:
    assert contest_score(
        d_seg=0.01,
        d_pose=0.4,
        archive_bytes=RATE_DENOMINATOR_BYTES,
    ) == pytest.approx(1.0 + 2.0 + 25.0)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"d_seg": -1.0, "d_pose": 0.0, "archive_bytes": 0}, ValueError),
        ({"d_seg": 0.0, "d_pose": -1.0, "archive_bytes": 0}, ValueError),
        ({"d_seg": float("nan"), "d_pose": 0.0, "archive_bytes": 0}, ValueError),
        ({"d_seg": 0.0, "d_pose": float("inf"), "archive_bytes": 0}, ValueError),
        ({"d_seg": True, "d_pose": 0.0, "archive_bytes": 0}, TypeError),
    ],
)
def test_contest_score_refuses_non_realized_components(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        contest_score(**kwargs)  # type: ignore[arg-type]


def test_structural_law_uses_the_right_state_and_explicit_quotient() -> None:
    law = right_level_coordinate_quotient_law()
    assert isinstance(law, RightLevelCoordinateQuotient)
    assert law.spatial_state == (
        "morse_smale_cells",
        "winner_rival_hyperplane_cells",
        "separatrices",
        "tie_loci",
    )
    assert law.temporal_coordinate == "xi"
    assert law.counted_irregular_coordinates == ("gauge", "events")
    assert law.residual_quotient == "T[class_id,cell_id]"


def test_structural_law_counts_video_data_and_trains_only_residual() -> None:
    law = right_level_coordinate_quotient_law()
    assert law.generic_generator_accounting.startswith("FREE_ONLY_IF_GENERIC_DETERMINISTIC")
    for token in ("SEED", "WEIGHT", "GAUGE", "EVENT", "T_PAYLOAD"):
        assert token in law.video_derived_accounting
    assert law.solve_project_before_training
    assert law.train_only == "residual_class_cell_quotient_after_exact_projection"


def test_structural_law_requires_exact_pullback_and_denies_film_isomorphism() -> None:
    law = right_level_coordinate_quotient_law()
    assert law.realization_authority == (
        "exact_inflate_bytes",
        "uint8_lattice",
        "exact_R_resize_round_chain",
        "frozen_SegNet_and_PoseNet_cells",
    )
    assert law.film_to_frozen_head_relation.startswith("NOT_ASSUMED_ISOMORPHIC")


def test_canonical_equation_is_honest_composed_research_only_law() -> None:
    equation = build_witness_right_level_coordinate_quotient_law_v1()
    assert isinstance(equation, CanonicalEquation)
    assert equation.equation_id == EQUATION_ID
    assert equation.domain_of_validity["law_status"] == (
        "DERIVED_FROM_COMPOSED_EXISTING_ANCHORS"
    )
    assert equation.domain_of_validity["composed_evidence_ids"] == list(
        COMPOSED_EVIDENCE_IDS
    )
    assert equation.domain_of_validity["score_claim_valid"] is False
    assert equation.domain_of_validity["pointer_moved"] is False
    assert equation.provenance.promotion_eligible is False
    assert equation.provenance.score_claim_valid is False


def test_composed_anchor_is_source_inspected_not_a_new_measurement() -> None:
    equation = build_witness_right_level_coordinate_quotient_law_v1()
    assert len(equation.empirical_anchors) == 1
    anchor = equation.empirical_anchors[0]
    assert anchor.empirical_verification_status == VERIFIED_VIA_SOURCE_INSPECTION
    assert anchor.inputs["evidence_ids"] == list(COMPOSED_EVIDENCE_IDS)
    assert anchor.inputs["new_measurement"] is False
    assert anchor.inputs["new_score_claim"] is False
    assert anchor.empirical_output["pointer_moved"] is False
    assert "not a measured score residual" in anchor.measurement_method


def test_build_does_not_require_research_sidecar(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_sidecar = tmp_path / "not_yet_landed.md"
    monkeypatch.setattr(law_module, "SIDECAR_PATH", str(missing_sidecar))
    equation = law_module.build_witness_right_level_coordinate_quotient_law_v1()
    assert not missing_sidecar.exists()
    assert equation.provenance.source_path == str(missing_sidecar)
    assert equation.provenance.source_sha256 == "0" * 64


def test_equation_serializes_to_json() -> None:
    payload = build_witness_right_level_coordinate_quotient_law_v1().to_dict()
    encoded = json.dumps(payload, sort_keys=True)
    assert EQUATION_ID in encoded
    assert "DERIVED_SOURCE_INSPECTION_NOT_A_MEASUREMENT" in encoded


def test_populate_round_trips_through_registry_without_sidecar(tmp_path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    lock = tmp_path / "canonical_equations.lock"
    equation = populate_witness_right_level_coordinate_quotient_law_equation(
        path=registry,
        lock_path=lock,
        agent="test",
        subagent_id="right-problem-law-test",
    )
    assert equation.equation_id == EQUATION_ID
    restored = get_equation_by_id(EQUATION_ID, path=registry)
    assert restored is not None
    assert restored.equation_id == EQUATION_ID
    assert restored.domain_of_validity["composed_evidence_ids"] == list(
        COMPOSED_EVIDENCE_IDS
    )
