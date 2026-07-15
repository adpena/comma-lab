# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.canonical_equations.micro_batch_memory_guard_20260715 import (
    EQUATION_ID,
    MICRO_BATCH_EXTRA_PAIR_GUARD_GIB,
    build_guard_receipt,
    micro_batch_guard_gib,
)


def test_guard_is_zero_for_serial_and_full_b4_process_peak_for_b2():
    assert micro_batch_guard_gib(1) == 0.0
    assert micro_batch_guard_gib(2) == MICRO_BATCH_EXTRA_PAIR_GUARD_GIB


def test_guard_scales_monotonically_per_extra_pair():
    assert micro_batch_guard_gib(4) == 3 * MICRO_BATCH_EXTRA_PAIR_GUARD_GIB


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_guard_refuses_invalid_batch_values(bad):
    with pytest.raises(ValueError, match="micro_batch_pairs"):
        micro_batch_guard_gib(bad)


def test_receipt_keeps_actual_rss_and_score_authority_empty():
    receipt = build_guard_receipt(2).to_dict()
    assert receipt["equation_id"] == EQUATION_ID
    assert receipt["guard_gib"] == 5.78
    assert receipt["actual_current_v9_b2_n600_rss_gib"] is None
    assert receipt["score_claim"] is False
    assert receipt["pointer_moved"] is False
    assert "UNMEASURED" in receipt["evidence_label"]
