# SPDX-License-Identifier: MIT
"""Binary64 parity guards for the pinned upstream evaluator score DAG."""

from __future__ import annotations

import math

import pytest

from tac.contest_score import (
    UNCOMPRESSED_SIZE_BYTES,
    compute_contest_score,
    rate_term,
)


def _pinned_upstream_score(d_seg: float, d_pose: float, archive_nbytes: int) -> float:
    """Literal operation order from upstream/evaluate.py:64-65,92."""

    rate = archive_nbytes / UNCOMPRESSED_SIZE_BYTES
    return 100 * d_seg + math.sqrt(d_pose * 10) + 25 * rate


@pytest.mark.parametrize(
    "archive_nbytes",
    (1, 2, 3, 7, 127, 128, 1_000, 80_238, 187_562, 37_545_488, 37_545_489),
)
def test_rate_term_is_bit_identical_to_pinned_upstream_order(
    archive_nbytes: int,
) -> None:
    expected_rate = archive_nbytes / UNCOMPRESSED_SIZE_BYTES
    assert rate_term(archive_nbytes).hex() == (25 * expected_rate).hex()


def test_guard_fixture_detects_the_old_algebraic_reassociation() -> None:
    upstream_order = 25 * (1 / UNCOMPRESSED_SIZE_BYTES)
    reassociated = 25 * 1 / UNCOMPRESSED_SIZE_BYTES
    assert reassociated != upstream_order
    assert abs(reassociated - upstream_order) == math.ulp(upstream_order)


@pytest.mark.parametrize(
    ("d_seg", "d_pose", "archive_nbytes"),
    (
        (0.0, 0.0, 1),
        (0.0001519690619574653, 0.00010184327939026322, 80_238),
        (0.0035127171, 127.3595581, 81_027),
        (1.0, 1.0, 37_545_489),
    ),
)
def test_full_score_is_bit_identical_to_pinned_upstream_order(
    d_seg: float,
    d_pose: float,
    archive_nbytes: int,
) -> None:
    actual = compute_contest_score(d_seg, d_pose, archive_nbytes)
    expected = _pinned_upstream_score(d_seg, d_pose, archive_nbytes)
    assert actual.hex() == expected.hex()
