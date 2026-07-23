# SPDX-License-Identifier: MIT
"""Tests for the DDM M4 rate-floor canonical laws."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from tac.canonical_equations.ddm_m4_rate_floor_20260723 import (
    DELEGATED_MAX_D_POSE,
    DELEGATED_MAX_D_SEG,
    SETTLED_D_POSE,
    SETTLED_D_SEG,
    ReceiverRow,
    lever_pool_map,
    minimum_admissible_receiver_row,
    score_terms,
    strict_archive_cap_bytes,
    uint8_unrecovered_scheduled_debt,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def _row(
    row_id: str,
    archive_bytes: int,
    d_seg: str,
    d_pose: str,
    *,
    n_pairs: int = 600,
    receiver_closed: bool = True,
) -> ReceiverRow:
    return ReceiverRow(
        row_id=row_id,
        archive_bytes=archive_bytes,
        d_seg=Decimal(d_seg),
        d_pose=Decimal(d_pose),
        n_pairs=n_pairs,
        receiver_closed=receiver_closed,
        evidence_axis="[test]",
    )


def test_settled_sub015_cap_is_154524_bytes() -> None:
    assert strict_archive_cap_bytes(SETTLED_D_SEG, SETTLED_D_POSE) == 154_524
    assert score_terms(SETTLED_D_SEG, SETTLED_D_POSE, 154_524)["total"] < Decimal("0.15")
    assert score_terms(SETTLED_D_SEG, SETTLED_D_POSE, 154_525)["total"] > Decimal("0.15")


def test_minimum_admissible_row_rejects_lower_byte_distortion_failure() -> None:
    rows = (
        _row("admissible", 177_169, "0.0005453067355847452", "0.00002930838566754801"),
        _row("int7_n600", 174_061, "0.001537", "0.000222"),
        _row("ddm_v19b", 137_825, "0.026594424778", "163.061176604795"),
    )
    got = minimum_admissible_receiver_row(rows, max_d_seg=DELEGATED_MAX_D_SEG, max_d_pose=DELEGATED_MAX_D_POSE)
    assert got.row_id == "admissible"
    assert got.archive_bytes == 177_169


def test_minimum_requires_n600_and_receiver_closure() -> None:
    rows = (
        _row("n48_only", 175_801, "0.000975", "0.000033", n_pairs=48),
        _row(
            "description_only",
            154_000,
            "0.0001",
            "0.0001",
            receiver_closed=False,
        ),
    )
    with pytest.raises(ValueError, match="no admissible"):
        minimum_admissible_receiver_row(rows)


def test_uint8_debt_is_not_mislabeled_as_realized_gain() -> None:
    assert uint8_unrecovered_scheduled_debt() == Decimal("0.01605162")


def test_pool_partition_is_exactly_the_delegated_seven_levers() -> None:
    mapping = lever_pool_map()
    assert set(mapping) == {
        "multicoefficient-solve",
        "correction-synergy",
        "frame-separation",
        "ker(A)-hide",
        "context-arithmetic-code",
        "xi-once-for-pose",
        "chart-canonicalization",
    }
    assert mapping["multicoefficient-solve"] == mapping["correction-synergy"]
    assert mapping["xi-once-for-pose"] == mapping["chart-canonicalization"]
    assert mapping["context-arithmetic-code"] == mapping["chart-canonicalization"]


def test_durable_receipt_keeps_three_floor_notions_separate() -> None:
    receipt_path = REPO_ROOT / ".omx/research/ddm_m4_rate_floor_einstein_avenue_20260723_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    rate_floor = receipt["rate_floor"]
    relaxed = rate_floor["audited_relaxed_receiver_floor"]
    exact_c1 = rate_floor["audited_exact_c1_receiver_floor"]

    assert rate_floor["universal_lower_bound_bytes"] == 0
    assert relaxed["archive_bytes"] == 177_169
    assert relaxed["global_minimum_claim"] is False
    assert exact_c1["archive_bytes"] == 409_526_925
    assert exact_c1["global_minimum_claim"] is False
    assert rate_floor["strict_sub015_at_settled_c1"]["max_archive_bytes"] == 154_524
    assert rate_floor["decisive_gap_bytes"] == 22_645
    assert receipt["rule_118_partition"]["measured_current_free_reclassification_reduction_bytes"] == 0
    assert receipt["ker_A"]["measured_counted_bytes_hideable_for_free"] == 0
    assert receipt["main_landing_review"]["required"] is True
