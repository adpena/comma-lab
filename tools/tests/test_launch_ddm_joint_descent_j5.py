from __future__ import annotations

from pathlib import Path

import pytest

from tac.optimization.direct_description_joint_descent import (
    DirectDescriptionJointDescentTypedConfigV1,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tools.launch_ddm_joint_descent import (
    _assert_worst_geometry_receipt,
    _c1_bucket_delta,
    _opening_exact_admitted,
    _seg_lexicographic_attempt_key,
    _write_structural_proposal_rejection,
)

REPO = Path(__file__).resolve().parents[2]
TICKET = REPO / ".omx/research/configs/ddm_j5_366_realized_acceptance_warmstart_20260723.json"


def _verdict(*, role_errors: int, residual_errors: int) -> dict:
    return {
        "c1_debt_buckets": {
            "role_correction_owned": {"errors": role_errors},
            "residual_trunk_owned": {
                "errors": residual_errors,
                "errors_above_target_allowance": max(residual_errors - 136_839, 0),
            },
        }
    }


def test_c1_bucket_delta_is_fixed_partition_and_candidate_minus_reference() -> None:
    row = _c1_bucket_delta(
        _verdict(role_errors=726_416, residual_errors=2_514_112),
        _verdict(role_errors=726_420, residual_errors=2_514_100),
    )

    assert row == {
        "role_correction_owned_delta_errors": 4,
        "residual_trunk_owned_delta_errors": -12,
        "residual_trunk_debt_delta_errors": -12,
        "global_delta_errors": -8,
        "residual_bucket_descended": True,
        "sign_convention": "candidate_minus_reference; negative removes errors",
    }


def test_c1_bucket_delta_accepts_legacy_per_class_verdict_for_resume() -> None:
    def legacy(*, road: int, lane: int, undrivable: int, movable: int, mycar: int) -> dict:
        return {
            "per_class": {
                "Road": {"errors": road},
                "Lane": {"errors": lane},
                "Undrivable": {"errors": undrivable},
                "Movable": {"errors": movable},
                "MyCar": {"errors": mycar},
            }
        }

    row = _c1_bucket_delta(
        legacy(road=2_000_000, lane=300_000, undrivable=400_000, movable=426_416, mycar=114_112),
        legacy(road=1_999_991, lane=300_003, undrivable=399_998, movable=426_417, mycar=114_111),
    )

    assert row["role_correction_owned_delta_errors"] == 4
    assert row["residual_trunk_owned_delta_errors"] == -12
    assert row["residual_trunk_debt_delta_errors"] == -12
    assert row["residual_bucket_descended"] is True


def test_worst_geometry_receipt_must_bind_all_52_stage3_secants() -> None:
    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(TICKET)
    contract = config.worst_geometry_memory_contract
    geometry = {
        "pair_start": 498,
        "pair_ids": [498, 499, 500, 501],
        "active_groups": [
            "island_worldsheet",
            "lane_program",
            "shared_template_dof",
        ],
        "island_secants": 28,
        "lane_secants": 24,
        "total_secants": 52,
        "derived_basis_gib": 4.72976016998291,
    }
    _assert_worst_geometry_receipt(geometry, contract)
    geometry["total_secants"] = 8
    with pytest.raises(DirectDescriptionError, match="sealed worst geometry"):
        _assert_worst_geometry_receipt(geometry, contract)


@pytest.mark.parametrize(
    ("priced", "component", "residual", "expected"),
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
    ],
)
def test_campaign_opening_requires_joint_component_and_residual_gates(
    priced: bool,
    component: bool,
    residual: bool,
    expected: bool,
) -> None:
    assert (
        _opening_exact_admitted(
            policy="campaign_component_safe_exact_n600",
            pure_priced_accepted=priced,
            component_safe=component,
            cumulative_fire_green=residual,
        )
        is expected
    )


def test_seg_lexicographic_order_keeps_rung_primary_and_seg_safe_first() -> None:
    attempts = [
        {
            "multiplier_index": 0,
            "candidate_index": 0,
            "metrics": {"seg_ce_margin": 1.01},
        },
        {
            "multiplier_index": 1,
            "candidate_index": 1,
            "metrics": {"seg_ce_margin": 0.80},
        },
        {
            "multiplier_index": 0,
            "candidate_index": 2,
            "metrics": {"seg_ce_margin": 0.99},
        },
    ]
    attempts.sort(
        key=lambda attempt: _seg_lexicographic_attempt_key(
            attempt,
            reference_seg_proxy=1.0,
        )
    )
    assert [attempt["candidate_index"] for attempt in attempts] == [2, 0, 1]


def test_structural_proposal_rejection_is_immutable_and_keeps_exact_authority_false(
    tmp_path: Path,
) -> None:
    row = _write_structural_proposal_rejection(
        out_dir=tmp_path,
        candidate_id="worldsheet_joint_active_y_-1",
        global_step=1,
        multiplier=32.0,
        multiplier_index=0,
        proposal_staging="camera_q8",
        reason="G1 Movable polygon escaped scorer geometry",
    )

    assert row["verdict"] == "REJECT_AND_SHRINK"
    assert row["exact_replay_executed"] is False
    assert row["score_claim"] is False
    assert "INSTANCE proposal geometry only" in row["verdict_scope"]
