# SPDX-License-Identifier: MIT
"""Fail-closed headline contract for the DDM inverse-solve campaign.

The campaign's decision number is not a residual-code diagnostic in isolation.
It is the own-lineage stored problem plus solve-mandated exceptions, measured
after deterministic expansion through uint8, the real resize, and both frozen
scorers.  Donor-conditioned rows are structurally inadmissible.
"""

from __future__ import annotations

import math
from typing import Any

HEADLINE_SCHEMA = "ddm_min_description_headline.v1"


class MinimumDescriptionContractError(ValueError):
    """A malformed byte, lineage, or realized-acceptance declaration."""


def _optional_bytes(value: int | None, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MinimumDescriptionContractError(
            f"{field} must be a nonnegative exact integer or null"
        )
    return value


def _optional_sha256(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MinimumDescriptionContractError(
            f"{field} must be a lowercase SHA-256 or null"
        )
    return value


def _distortion(value: float, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise MinimumDescriptionContractError(
            f"{field} must be finite and nonnegative"
        )
    return result


def build_minimum_description_headline(
    *,
    stored_problem_bytes: int | None,
    stored_problem_sha256: str | None,
    exception_bytes: int | None,
    exception_sha256: str | None,
    realized_d_seg: float,
    realized_d_pose: float,
    stored_problem_own_lineage: bool,
    donor_conditioned: bool,
    expansion_receiver_closed: bool,
    pose_tube_active: bool,
    realized_uint8_r_frozen_scorers: bool,
) -> dict[str, Any]:
    """Build the only row eligible to headline minimum-description progress.

    A diagnostic row is still returned when custody is incomplete so a failed
    rung remains useful system intelligence.  Its decision triple is withheld,
    and blockers state exactly which authority edge is absent.
    """

    problem_bytes = _optional_bytes(stored_problem_bytes, "stored_problem_bytes")
    exceptions = _optional_bytes(exception_bytes, "exception_bytes")
    problem_sha = _optional_sha256(stored_problem_sha256, "stored_problem_sha256")
    exceptions_sha = _optional_sha256(exception_sha256, "exception_sha256")
    d_seg = _distortion(realized_d_seg, "realized_d_seg")
    d_pose = _distortion(realized_d_pose, "realized_d_pose")
    declarations = (
        stored_problem_own_lineage,
        donor_conditioned,
        expansion_receiver_closed,
        pose_tube_active,
        realized_uint8_r_frozen_scorers,
    )
    if any(not isinstance(value, bool) for value in declarations):
        raise MinimumDescriptionContractError(
            "lineage and acceptance declarations must be exact booleans"
        )

    blockers: list[str] = []
    if donor_conditioned:
        blockers.append("DONOR_CONDITIONING_INADMISSIBLE")
    if not stored_problem_own_lineage:
        blockers.append("OWN_LINEAGE_STORED_PROBLEM_NOT_PROVEN")
    if problem_bytes is None or problem_sha is None:
        blockers.append("STORED_PROBLEM_BYTE_CUSTODY_MISSING")
    if exceptions is None or exceptions_sha is None:
        blockers.append("SOLVE_EXCEPTION_BYTE_CUSTODY_MISSING")
    if not expansion_receiver_closed:
        blockers.append("STORED_PROBLEM_EXPANSION_NOT_RECEIVER_CLOSED")
    if not pose_tube_active:
        blockers.append("POSE_TUBE_NOT_ACTIVE_IN_SOLVE")
    if not realized_uint8_r_frozen_scorers:
        blockers.append("REALIZED_UINT8_R_FROZEN_SCORER_ACCEPTANCE_MISSING")

    eligible = not blockers
    total_bytes = (
        int(problem_bytes + exceptions)
        if eligible and problem_bytes is not None and exceptions is not None
        else None
    )
    return {
        "schema": HEADLINE_SCHEMA,
        "campaign": "inverse_solve_minimum_description_witness",
        "status": (
            "HEADLINE_ELIGIBLE"
            if eligible
            else (
                "INADMISSIBLE_DONOR_CONDITIONING"
                if donor_conditioned
                else "HEADLINE_BLOCKED"
            )
        ),
        "headline_eligible": eligible,
        "stored_problem": {
            "bytes": problem_bytes,
            "sha256": problem_sha,
            "own_lineage": stored_problem_own_lineage,
            "receiver_expansion_closed": expansion_receiver_closed,
        },
        "solve_mandated_exceptions": {
            "bytes": exceptions,
            "sha256": exceptions_sha,
            "conditional_coding_role": (
                "exceptions conditioned only on deterministic expansion of the "
                "counted own-lineage stored problem"
            ),
        },
        "joint_constraints": {
            "pose_tube_active": pose_tube_active,
            "realized_uint8_r_frozen_scorers": realized_uint8_r_frozen_scorers,
        },
        "donor_conditioned": donor_conditioned,
        "decision_triple": {
            "total_counted_bytes": total_bytes,
            "realized_d_seg": d_seg if eligible else None,
            "realized_d_pose": d_pose if eligible else None,
        },
        "diagnostic_distortions": {
            "realized_d_seg": d_seg,
            "realized_d_pose": d_pose,
        },
        "blockers": blockers,
        "score_claim": False,
        "promotion_eligible": False,
    }


__all__ = [
    "HEADLINE_SCHEMA",
    "MinimumDescriptionContractError",
    "build_minimum_description_headline",
]
