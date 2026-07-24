# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_pa2_zero_byte_decode_family_20260724 import (
    select_strict_conditional,
    zero_byte_conditional_score,
)


def test_score_law_matches_rate_normalization() -> None:
    value = zero_byte_conditional_score(
        seg_errors=0,
        scored_pixels=600 * 384 * 512,
        d_pose=0.0,
        archive_bytes=37_545_489,
        decoded_frame_only=True,
        archive_identity_exact=True,
        n600_batch32_complete=True,
    )
    assert value == 25.0


@pytest.mark.parametrize(
    "field",
    ("decoded_frame_only", "archive_identity_exact", "n600_batch32_complete"),
)
def test_score_law_fails_closed_on_missing_gate(field: str) -> None:
    row = {
        "seg_errors": 0,
        "scored_pixels": 1,
        "d_pose": 0.0,
        "archive_bytes": 0,
        "decoded_frame_only": True,
        "archive_identity_exact": True,
        "n600_batch32_complete": True,
    }
    row[field] = False
    with pytest.raises(ValueError, match="requires"):
        zero_byte_conditional_score(**row)


def test_strict_conditional_selects_fresh_minimum() -> None:
    value = select_strict_conditional(
        current_score=28.00173925293584,
        candidates=(
            {
                "member": "spatial",
                "score": 30.478287855584014,
                "archive_byte_delta": 0,
            },
            {
                "member": "frame1_xihat",
                "score": 25.244396496399435,
                "archive_byte_delta": 0,
            },
        ),
    )
    assert value["selected_member"] == "frame1_xihat"
    assert value["strict_improvement"] is True
    assert value["conditional_delta_score"] == pytest.approx(-2.7573427565364064)


def test_strict_conditional_refuses_tie_and_byte_drift() -> None:
    tied = select_strict_conditional(
        current_score=1.0,
        candidates=(
            {"member": "neutral", "score": 1.0, "archive_byte_delta": 0},
        ),
    )
    assert tied["selected_member"] is None
    with pytest.raises(ValueError, match="changed archive"):
        select_strict_conditional(
            current_score=1.0,
            candidates=(
                {"member": "counted", "score": 0.0, "archive_byte_delta": 1},
            ),
        )
