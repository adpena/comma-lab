from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.ddm_min_description_contract import (
    HEADLINE_SCHEMA,
    MinimumDescriptionContractError,
    build_minimum_description_headline,
)


def _row(**updates: object) -> dict:
    values = {
        "stored_problem_bytes": 100,
        "stored_problem_sha256": "a" * 64,
        "exception_bytes": 20,
        "exception_sha256": "b" * 64,
        "realized_d_seg": 0.001,
        "realized_d_pose": 0.0001,
        "stored_problem_own_lineage": True,
        "donor_conditioned": False,
        "expansion_receiver_closed": True,
        "pose_tube_active": True,
        "realized_uint8_r_frozen_scorers": True,
        "quotient_coordinates_only": True,
        "scorer_metric_active": True,
        "alternating_typed_subproblems": True,
        "typed_blocks_active": True,
        "per_dimension_quanta_active": True,
    }
    values.update(updates)
    return build_minimum_description_headline(**values)  # type: ignore[arg-type]


def test_headline_requires_complete_own_lineage_joint_receipt() -> None:
    row = _row()
    assert row["schema"] == HEADLINE_SCHEMA
    assert row["headline_eligible"] is True
    assert row["decision_triple"] == {
        "total_counted_bytes": 120,
        "realized_d_seg": 0.001,
        "realized_d_pose": 0.0001,
    }


def test_donor_conditioning_is_inadmissible_even_with_complete_bytes() -> None:
    row = _row(donor_conditioned=True)
    assert row["status"] == "INADMISSIBLE_DONOR_CONDITIONING"
    assert row["headline_eligible"] is False
    assert row["decision_triple"]["total_counted_bytes"] is None
    assert "DONOR_CONDITIONING_INADMISSIBLE" in row["blockers"]


def test_missing_problem_and_pose_custody_yield_exact_blockers() -> None:
    row = _row(
        stored_problem_bytes=None,
        stored_problem_sha256=None,
        pose_tube_active=False,
    )
    assert row["headline_eligible"] is False
    assert row["blockers"] == [
        "STORED_PROBLEM_BYTE_CUSTODY_MISSING",
        "POSE_TUBE_NOT_ACTIVE_IN_SOLVE",
    ]
    assert row["diagnostic_distortions"]["realized_d_seg"] == 0.001


def test_flat_untyped_solve_withholds_headline_with_five_exact_blockers() -> None:
    row = _row(
        quotient_coordinates_only=False,
        scorer_metric_active=False,
        alternating_typed_subproblems=False,
        typed_blocks_active=False,
        per_dimension_quanta_active=False,
    )
    assert row["recursive_solve_typing"]["blockers"] == [
        "GAUGE_COORDINATES_NOT_DROPPED",
        "SCORER_METRIC_NOT_ACTIVE",
        "TYPED_SUBPROBLEM_ALTERNATION_NOT_ACTIVE",
        "TYPED_BLOCK_ATLAS_NOT_ACTIVE",
        "PER_DIMENSION_EFFECTIVE_QUANTA_NOT_ACTIVE",
    ]
    assert row["decision_triple"]["total_counted_bytes"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stored_problem_bytes", True),
        ("exception_bytes", -1),
        ("stored_problem_sha256", "not-a-sha"),
        ("realized_d_pose", float("nan")),
    ],
)
def test_malformed_headline_fields_fail_closed(field: str, value: object) -> None:
    with pytest.raises(MinimumDescriptionContractError):
        _row(**{field: value})


def test_n600_repo_receipt_rebuilds_the_fail_closed_campaign_headline() -> None:
    root = Path(__file__).resolve().parents[2]
    receipt = json.loads(
        (
            root
            / ".omx/research/ddm_ms1_min_description_lattice_solve_20260724_receipt.json"
        ).read_text()
    )
    expected = build_minimum_description_headline(
        stored_problem_bytes=None,
        stored_problem_sha256=None,
        exception_bytes=731622325,
        exception_sha256=None,
        realized_d_seg=0.0001519690619574653,
        realized_d_pose=0.00010184327939026322,
        stored_problem_own_lineage=False,
        donor_conditioned=False,
        expansion_receiver_closed=False,
        pose_tube_active=False,
        realized_uint8_r_frozen_scorers=True,
        quotient_coordinates_only=False,
        scorer_metric_active=False,
        alternating_typed_subproblems=False,
        typed_blocks_active=False,
        per_dimension_quanta_active=False,
    )
    # The receipt gives the diagnostic exception a more specific historical
    # role while preserving every authority-bearing contract field.
    actual = receipt["campaign_headline"]
    assert actual["solve_mandated_exceptions"]["conditional_coding_role"] != (
        expected["solve_mandated_exceptions"]["conditional_coding_role"]
    )
    actual = json.loads(json.dumps(actual))
    actual["solve_mandated_exceptions"]["conditional_coding_role"] = (
        expected["solve_mandated_exceptions"]["conditional_coding_role"]
    )
    assert actual == expected
