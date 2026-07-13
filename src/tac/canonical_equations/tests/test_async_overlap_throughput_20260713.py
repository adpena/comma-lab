# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.canonical_equations.async_overlap_throughput_20260713 import (
    EQUATION_ID,
    account_async_overlap,
    async_only_identifiability,
    derive_incremental_vjp_s,
)


def test_matched_overlap_separates_tail_from_contention() -> None:
    row = account_async_overlap(
        train_solo_s=100.0,
        train_concurrent_s=112.0,
        async_service_s=47.0,
    )
    assert EQUATION_ID.endswith("_v1")
    assert row.exposed_async_tail_s == 0.0
    assert row.contention_penalty_fraction == pytest.approx(0.12)
    assert row.contention_free_speedup_x == pytest.approx(1.12)
    assert row.score_claim is False
    assert row.pointer_moved is False


def test_exposed_tail_is_not_mislabeled_as_contention() -> None:
    row = account_async_overlap(
        train_solo_s=100.0,
        train_concurrent_s=100.0,
        async_service_s=125.0,
    )
    assert row.exposed_async_tail_s == 25.0
    assert row.contention_penalty_fraction == 0.0
    assert row.contention_free_speedup_x == 1.0


def test_async_only_log_refuses_contention_inference() -> None:
    row = async_only_identifiability(cadence_miss_count=0)
    assert row["no_cadence_miss_measured"] is True
    assert row["contention_identified"] is False
    assert row["contention_penalty_fraction"] is None


@pytest.mark.parametrize("bad", [-1, 1.2, True])
def test_async_only_log_requires_nonnegative_integer_miss_count(bad: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        async_only_identifiability(cadence_miss_count=bad)  # type: ignore[arg-type]


def test_inclusive_backward_requires_subtraction_and_refuses_negative_increment() -> None:
    assert derive_incremental_vjp_s(forward_s=0.6, backward_inclusive_s=0.9) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="incremental VJP is unresolved"):
        derive_incremental_vjp_s(forward_s=0.9, backward_inclusive_s=0.6)


@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
def test_nonphysical_timings_fail_closed(bad: float) -> None:
    with pytest.raises(ValueError):
        account_async_overlap(train_solo_s=bad, train_concurrent_s=1.0, async_service_s=1.0)
