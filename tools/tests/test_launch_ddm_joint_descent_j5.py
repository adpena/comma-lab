from __future__ import annotations

from tools.launch_ddm_joint_descent import _c1_bucket_delta


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
