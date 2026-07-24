# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tools.summarize_ddm_ms6_receiver_support import (
    _distribution,
    _g3_coverage,
    _signed_asymmetry,
)


def test_signed_asymmetry_preserves_direction_and_zero_support() -> None:
    assert _signed_asymmetry(10, 30) == pytest.approx(0.5)
    assert _signed_asymmetry(30, 10) == pytest.approx(-0.5)
    assert _signed_asymmetry(0, 0) == 0.0


def test_distribution_reports_direction_counts_and_quantiles() -> None:
    value = _distribution([-1.0, 0.0, 0.5, 1.0])
    assert value["count"] == 4
    assert value["negative_dominant_count"] == 1
    assert value["exact_tie_count"] == 1
    assert value["positive_dominant_count"] == 2
    assert value["median"] == pytest.approx(0.25)


def test_g3_coverage_requires_exact_pair_in_joined_pair_ids() -> None:
    rows = [
        {
            "bucket_id": "a",
            "pf2_membership_pair_ids": [1, 2],
            "pair_ids": [1],
        },
        {
            "bucket_id": "b",
            "pf2_membership_pair_ids": [1],
            "pair_ids": [1],
        },
        {
            "bucket_id": "c",
            "pf2_membership_pair_ids": [2],
            "pair_ids": [],
        },
    ]
    value = _g3_coverage(rows, [1, 2])
    assert value["coverage_proven"] is False
    assert value["fully_joined_pair_count"] == 1
    assert value["missing_blocks"] == [{"pair_id": 2, "bucket_id": "a"}, {"pair_id": 2, "bucket_id": "c"}]


def test_g3_coverage_ignores_buckets_without_hard_pair_membership() -> None:
    rows = [
        {
            "bucket_id": "a",
            "pf2_membership_pair_ids": [7],
            "pair_ids": [7],
        },
        {
            "bucket_id": "unrelated",
            "pf2_membership_pair_ids": [9],
            "pair_ids": [],
        },
    ]
    value = _g3_coverage(rows, [7])
    assert value["coverage_proven"] is True
    assert value["missing_block_count"] == 0
