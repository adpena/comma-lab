# SPDX-License-Identifier: MIT
"""Tests for the gap-decomposition denominator law (ddm_cv1, 2026-08-02).

The REGRESSION this pins: on its first run the module contradicted a figure MAIN had
already published to MEMORY.md ("1% of gap = 11,892 B"; the correct value is 10,907 B at
the corrected floor). That is the whole point of making the denominator executable.

TWICE-CORRECTED, and the second correction came from a different direction than the first.
Run 1 caught MAIN's arithmetic (11,892 -> 10,908). Then `ddm_na1` caught the INPUT: the
PR130 floor is **191,052 B, not 190,952** — 190,952 yields floor 0.1720751, which does not
reproduce PR130's published 0.172141, while 191,052 yields 0.1721417, which does. Corrected
gap 0.7262358; 1% = 10,907 B. The equation was right both times; its inputs were not. That
is the argument for sourcing every field and refusing unsourced rows.
"""
from __future__ import annotations

import math

import pytest

from tac.canonical_equations.gap_decomposition_against_floor_20260802 import (
    GapDecomposition,
    MeasuredScoreTriple,
)

_DEN = 37_545_489


def _ours() -> MeasuredScoreTriple:
    """dc1_fold, 2026-08-02, n600 upstream/evaluate.py rc=0."""
    return MeasuredScoreTriple(
        d_seg=0.00431179,
        d_pose=0.00516578,
        archive_bytes=360_309,
        rate_denominator_bytes=_DEN,
        source_artifact="dc1_fold n600 evaluate.py rc=0 2026-08-02",
        axis_tag="[macOS-CPU advisory exact n600]",
    )


def _floor() -> MeasuredScoreTriple:
    """PR130 external demonstrated row."""
    return MeasuredScoreTriple(
        d_seg=0.0002966,
        d_pose=2.3311e-5,
        archive_bytes=191_052,
        rate_denominator_bytes=_DEN,
        source_artifact="PR130 external row (191,052 B — CORRECTED by ddm_na1 2026-08-02; the prior 190,952 gives floor 0.1720751, which does not reproduce the published 0.172141)",
        axis_tag="[contest-CUDA]",
    )


def test_total_recomputed_from_components_not_a_rounded_field():
    """S must come from the three terms; evaluate.py's 2-dp 'Final score' lies."""
    assert _ours().total == pytest.approx(0.8983775, abs=5e-7)


def test_per_axis_gaps_and_ordering():
    g = GapDecomposition(ours=_ours(), floor=_floor())
    gaps = g.per_axis()
    assert gaps["seg"] == pytest.approx(0.401519, abs=1e-6)
    assert gaps["pose"] == pytest.approx(0.2120156, abs=1e-6)
    assert gaps["rate"] == pytest.approx(0.1127013, abs=1e-6)
    # The ordering is a MEASURED OUTPUT. If a future row changes it, this test should
    # fail loudly rather than let a stale "seg is biggest" assumption ride.
    assert g.rank_by_gap() == ("seg", "pose", "rate")


def test_shares_sum_to_one_and_seg_is_the_majority():
    g = GapDecomposition(ours=_ours(), floor=_floor())
    shares = g.shares()
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-12)
    assert shares["seg"] == pytest.approx(0.553, abs=0.001)


def test_fraction_of_gap_sign_convention_and_the_dc1_row():
    """A score-LOWERING delta returns a POSITIVE fraction of gap closed."""
    g = GapDecomposition(ours=_ours(), floor=_floor())
    assert g.fraction_of_gap(-0.0000560) == pytest.approx(7.71e-5, rel=1e-2)
    assert g.fraction_of_gap(+0.0000560) < 0.0  # a regression closes negative gap


def test_bytes_per_percent_regression_the_published_figure_was_wrong():
    """PINNED: 10,907 B (at the CORRECTED 191,052 B floor), not the 11,892 B MAIN
    published before this module existed. The figure moved 10,908 -> 10,907 when ddm_na1
    corrected the PR130 byte count; both refute 11,892 by three orders of the tolerance."""
    g = GapDecomposition(ours=_ours(), floor=_floor())
    got = g.bytes_per_percent_of_gap()
    assert got == pytest.approx(10_907, rel=1e-3)
    assert abs(got - 11_892) > 900, "the superseded figure must not silently pass"


def test_mismatched_rate_denominators_refuse():
    """Catalog #812: evaluate.py sums videos/ dynamically. Two rows measured against
    different directory contents are not comparable on the rate axis."""
    other = MeasuredScoreTriple(
        d_seg=0.0002966,
        d_pose=2.3311e-5,
        archive_bytes=191_052,
        rate_denominator_bytes=_DEN + 4096,  # a stray ._* file
        source_artifact="PR130 with a polluted videos/ dir",
        axis_tag="[contest-CUDA]",
    )
    with pytest.raises(ValueError, match="rate denominators differ"):
        GapDecomposition(ours=_ours(), floor=other)


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"d_seg": -1e-9}, ValueError),
        ({"d_pose": math.inf}, ValueError),
        ({"archive_bytes": 0}, ValueError),
        ({"rate_denominator_bytes": -1}, ValueError),
        ({"archive_bytes": 360_309.0}, TypeError),   # a float byte count is a bug
        ({"source_artifact": "  "}, ValueError),
        ({"axis_tag": ""}, ValueError),
        ({"status": "DERIVED"}, ValueError),          # only MEASURED may anchor a gap
    ],
)
def test_fail_closed_on_unsourced_or_nonmeasured_inputs(kwargs, exc):
    base = dict(
        d_seg=0.00431179,
        d_pose=0.00516578,
        archive_bytes=360_309,
        rate_denominator_bytes=_DEN,
        source_artifact="x",
        axis_tag="[advisory]",
    )
    base.update(kwargs)
    with pytest.raises(exc):
        MeasuredScoreTriple(**base)


def test_at_the_floor_shares_and_fraction_refuse_rather_than_divide_by_zero():
    row = _floor()
    g = GapDecomposition(ours=row, floor=row)
    assert g.total_gap == pytest.approx(0.0, abs=1e-12)
    for call in (g.shares, lambda: g.fraction_of_gap(-0.01), g.bytes_per_percent_of_gap):
        with pytest.raises(ValueError):
            call()


def test_negative_gap_is_reported_not_clipped():
    """If we ever BEAT the floor on an axis, that must show as a negative share --
    clipping it would silently inflate the remaining axes."""
    strong = MeasuredScoreTriple(
        d_seg=0.0001,            # better than the floor's 0.0002966
        d_pose=0.00516578,
        archive_bytes=360_309,
        rate_denominator_bytes=_DEN,
        source_artifact="hypothetical seg-beating row",
        axis_tag="[advisory]",
    )
    g = GapDecomposition(ours=strong, floor=_floor())
    assert g.per_axis()["seg"] < 0.0
    assert g.shares()["seg"] < 0.0
