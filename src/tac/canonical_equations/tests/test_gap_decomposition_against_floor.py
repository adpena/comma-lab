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
import warnings

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


# ---------------------------------------------------------------------------
# ddm_op3 (2026-08-03) -- CUSTODY READERS AND TRANSFORMATION LAWS.
#
# POSITIVE CONTROLS ARE THE DELIVERABLE HERE, not a checkbox.  Two single-file
# detectors written the previous day both PASSED their own positive control while
# being unable to return the negative at all.  So every clause below is exercised
# twice: a known-BAD case that must be REFUSED, and a known-GOOD case that must
# pass.  A test that only shows the good case cannot distinguish a working
# instrument from an inert one.
# ---------------------------------------------------------------------------

import json as _json
from pathlib import Path as _Path

from tac.canonical_equations.gap_decomposition_against_floor_20260802 import (
    EXACT_INVARIANT,
    PR130_FLOOR_ARCHIVE_BYTES,
    PR130_FLOOR_PUBLISHED_TOTAL,
    RESTATEABLE,
    SKIP_POINTER_CROSSCHECK,
    EvaluatorReportParseError,
    LiveOperatingPoint,
    demonstrated_floor_pr130,
    live_operating_point,
    marginals,
    parse_evaluator_report,
    restate_pose_delta_at,
    seg_rate_exchange_bytes_per_flip,
    triple_from_evaluator_report,
)
from tac.canonical_equations.gap_decomposition_against_floor_20260802 import (
    GapDecomposition as _GapDecomposition,
)

# A byte-faithful copy of the real cx1 receipt (the live own-vehicle best on
# 2026-08-03).  Inlined rather than read from /Volumes so the control still runs
# when the SSD is unmounted -- a test that silently skips is the vacuity class.
_CX1_REPORT = """=== Evaluation config ===
  batch_size: 16
  device: cpu
  num_threads: 2
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00255143
  Average SegNet Distortion: 0.00431179
  Submission file size: 353,808 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00942345
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 0.83
"""


def _write(tmp_path: _Path, text: str, name: str = "report.txt") -> _Path:
    d = tmp_path / "v4d_cx1_pj2ix2"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(text, encoding="utf-8")
    return p


# --- CONTROL PAIR 1: the comma-truncation regression --------------------------


def test_POSITIVE_CONTROL_good_receipt_parses_every_field(tmp_path):
    """KNOWN-GOOD: the real cx1 receipt must parse and recompute to 0.8264972."""
    parsed = parse_evaluator_report(_write(tmp_path, _CX1_REPORT))
    assert parsed.archive_bytes == 353_808, "the comma must be consumed, not truncated at"
    assert parsed.rate_denominator_bytes == 37_545_489
    assert parsed.n_samples == 600
    assert parsed.total == pytest.approx(0.8264972, abs=5e-7)


def test_POSITIVE_CONTROL_comma_truncation_cannot_pass_quietly(tmp_path):
    """KNOWN-BAD: the exact regression ddm_qd2 hit -- a tolerant regex read '353,808'
    as '353' and produced a plausible, entirely wrong table.  Here the file itself is
    truncated mid-number; the anchored pattern demands the 'bytes' suffix and the end of
    line, so a partial match must REFUSE rather than return 353."""
    broken = _CX1_REPORT.replace("Submission file size: 353,808 bytes", "Submission file size: 353")
    with pytest.raises(EvaluatorReportParseError, match="missing required field"):
        parse_evaluator_report(_write(tmp_path, broken))


# --- CONTROL PAIR 2: population (the prefix-vs-n600 genus axis) ---------------


def test_POSITIVE_CONTROL_full_population_accepted(tmp_path):
    t = triple_from_evaluator_report(
        _write(tmp_path, _CX1_REPORT), axis_tag="[macOS-CPU advisory exact n600]"
    )
    assert t.total == pytest.approx(0.8264972, abs=5e-7)


def test_POSITIVE_CONTROL_subset_population_refused(tmp_path):
    """KNOWN-BAD: a 73-sample prefix measured mean d_pose 5.1x the population's, so a
    prefix is a scene block, not a sample.  A subset receipt must not be read as a
    smaller measurement of the same population."""
    subset = _CX1_REPORT.replace("over 600 samples", "over 73 samples")
    with pytest.raises(EvaluatorReportParseError, match="different population"):
        triple_from_evaluator_report(_write(tmp_path, subset), axis_tag="[advisory]")


def test_POSITIVE_CONTROL_missing_population_header_refused(tmp_path):
    """The population is an ARGUMENT of the claim; it may not be assumed."""
    headless = _CX1_REPORT.replace("=== Evaluation results over 600 samples ===", "=== results ===")
    with pytest.raises(EvaluatorReportParseError, match="POPULATION"):
        parse_evaluator_report(_write(tmp_path, headless))


# --- CONTROL PAIR 3: the floor (the number every ranking divides by) ---------


def test_POSITIVE_CONTROL_floor_reproduces_its_published_row():
    """KNOWN-GOOD: 191,052 B reproduces PR130's published 0.172141."""
    floor = demonstrated_floor_pr130(_DEN, frontier_pointer_path=SKIP_POINTER_CROSSCHECK)
    assert floor.archive_bytes == PR130_FLOOR_ARCHIVE_BYTES
    assert floor.total == pytest.approx(PR130_FLOOR_PUBLISHED_TOTAL, abs=5e-7)


def test_POSITIVE_CONTROL_the_superseded_190952_would_not_reproduce():
    """KNOWN-BAD, stated as arithmetic: the transposed 190,952 gives 0.1720747, which
    misses the published row by 130x the tolerance the loader enforces.  This is the
    check that actually decided the correction -- pinned so it cannot silently revert."""
    wrong = MeasuredScoreTriple(
        d_seg=0.0002966,
        d_pose=2.331e-5,
        archive_bytes=190_952,
        rate_denominator_bytes=_DEN,
        source_artifact="the transposed PR130 byte count (payload 'p', not archive.zip)",
        axis_tag="[contest-CUDA]",
    )
    assert abs(wrong.total - PR130_FLOOR_PUBLISHED_TOTAL) > 5e-5


def test_POSITIVE_CONTROL_floor_refuses_when_the_pointer_bar_has_moved(tmp_path):
    """KNOWN-BAD: if the leaderboard moves, ranking against the old floor must REFUSE,
    not silently continue.  This is the 'formulation/floor' axis of the genus, and a
    class verdict has already flipped on dividing by the wrong floor."""
    ptr = tmp_path / "pointer.json"
    ptr.write_text(_json.dumps({"effective_frontier": {"score": 0.15}}), encoding="utf-8")
    with pytest.raises(ValueError, match="bar has moved"):
        demonstrated_floor_pr130(_DEN, frontier_pointer_path=ptr)


def test_POSITIVE_CONTROL_floor_accepts_the_live_pointer(tmp_path):
    """KNOWN-GOOD companion: at the real pointer value the loader passes."""
    ptr = tmp_path / "pointer.json"
    ptr.write_text(_json.dumps({"effective_frontier": {"score": 0.172}}), encoding="utf-8")
    assert demonstrated_floor_pr130(_DEN, frontier_pointer_path=ptr).total == pytest.approx(
        PR130_FLOOR_PUBLISHED_TOTAL, abs=5e-7
    )


def test_malformed_pointer_raises_rather_than_skipping_the_crosscheck(tmp_path):
    """A missing pointer file is a legitimate absence; a MALFORMED one is not, because
    skipping a check silently is exactly the failure the check exists to prevent."""
    ptr = tmp_path / "pointer.json"
    ptr.write_text(_json.dumps({"nothing": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        demonstrated_floor_pr130(_DEN, frontier_pointer_path=ptr)


# --- CONTROL PAIR 4: scope (empty scope is VACUOUS, never PASS) --------------


def test_POSITIVE_CONTROL_empty_scope_refuses(tmp_path):
    with pytest.raises(ValueError, match="EMPTY SCOPE"):
        live_operating_point([], axis_tag="[advisory]")


def test_POSITIVE_CONTROL_live_operating_point_reports_its_denominator(tmp_path):
    """KNOWN-GOOD: one receipt in scope, and the scope size is REPORTED, so a reader
    can see the population the 'best' was selected from. The PR130 floor is passed with
    the EXPLICIT historical-use skip: this control tests denominator reporting, and the
    superseded-bar refusal (exercised by its own test below) would otherwise mask it."""
    lop = live_operating_point(
        [_write(tmp_path, _CX1_REPORT)],
        axis_tag="[macOS-CPU advisory exact n600]",
        floor=demonstrated_floor_pr130(_DEN, frontier_pointer_path=SKIP_POINTER_CROSSCHECK),
    )
    assert isinstance(lop, LiveOperatingPoint)
    assert lop.receipts_scanned == 1 and lop.receipts_parsed == 1
    assert lop.best_label == "v4d_cx1_pj2ix2"
    summary = lop.summary()
    # The reconciled live figures. Seven variants of this gap were in circulation.
    assert summary["best_S"] == pytest.approx(0.8264972, abs=5e-7)
    assert summary["total_gap"] == pytest.approx(0.6543559, abs=5e-7)
    assert summary["bytes_per_percent_of_gap"] == pytest.approx(9827.2, rel=1e-4)
    assert summary["rank_by_gap"] == ("seg", "pose", "rate")
    assert summary["shares"]["seg"] == pytest.approx(0.6136, abs=1e-4)
    # The stale constants that were in live use must be refuted, not merely absent.
    assert abs(summary["bytes_per_percent_of_gap"] - 10_907) > 900
    assert abs(summary["shares"]["seg"] - 0.553) > 0.05


def test_POSITIVE_CONTROL_cross_axis_legs_are_announced(tmp_path):
    """SENTINEL: our receipts are macOS-CPU advisory; the floor is contest-CUDA. The
    instrument must SAY so on every run. If this canary ever goes quiet without the
    axes actually matching, the instrument is untrusted."""
    lop = live_operating_point(
        [_write(tmp_path, _CX1_REPORT)],
        axis_tag="[macOS-CPU advisory exact n600]",
        floor=demonstrated_floor_pr130(_DEN, frontier_pointer_path=SKIP_POINTER_CROSSCHECK),
    )
    assert "cross_axis_warning" in lop.summary()
    same = _GapDecomposition(ours=_floor(), floor=demonstrated_floor_pr130(_DEN, frontier_pointer_path=SKIP_POINTER_CROSSCHECK))
    assert same.cross_axis_warning() is None


# --- CONTROL PAIR 5: the transformation laws --------------------------------


def test_W_is_exactly_invariant_and_reproduces_the_banked_constant():
    """A live charter assumed W moves as the archive shrinks. It does not: W is a ratio
    of two LINEAR terms and has no operating point."""
    w = seg_rate_exchange_bytes_per_flip(_DEN)
    assert w.value == pytest.approx(1.2731082153320312, abs=1e-12)
    assert w.invariance == EXACT_INVARIANT
    # Same archive-size independence, stated as a test rather than an assertion in prose.
    assert seg_rate_exchange_bytes_per_flip(_DEN).value == w.value


def test_pose_marginal_is_operating_point_dependent_and_seg_rate_are_not():
    """The distinction nothing carried before: two marginals that read alike behave
    completely differently under a move of the operating point."""
    pw1 = MeasuredScoreTriple(
        d_seg=0.00431179, d_pose=0.00764555, archive_bytes=360_323,
        rate_denominator_bytes=_DEN, source_artifact="v4d_pw1 receipt", axis_tag="[advisory]",
    )
    cx1 = MeasuredScoreTriple(
        d_seg=0.00431179, d_pose=0.00255143, archive_bytes=353_808,
        rate_denominator_bytes=_DEN, source_artifact="v4d_cx1 receipt", axis_tag="[advisory]",
    )
    assert marginals(pw1)["seg"].value == marginals(cx1)["seg"].value == 100.0
    assert marginals(pw1)["rate"].value == marginals(cx1)["rate"].value
    assert marginals(pw1)["seg"].invariance == EXACT_INVARIANT
    assert marginals(cx1)["pose"].invariance == RESTATEABLE
    assert marginals(cx1)["pose"].value == pytest.approx(31.3024, rel=1e-5)
    assert marginals(pw1)["pose"].value == pytest.approx(18.0828, rel=1e-5)


def test_banked_pose_deltas_are_UNDER_priced_not_over_priced():
    """The one direction in which re-pricing FINDS value. A d_pose saving banked at the
    pw1 operating point is worth 1.73x more S today."""
    pw1 = MeasuredScoreTriple(
        d_seg=0.00431179, d_pose=0.00764555, archive_bytes=360_323,
        rate_denominator_bytes=_DEN, source_artifact="v4d_pw1 receipt", axis_tag="[advisory]",
    )
    cx1 = MeasuredScoreTriple(
        d_seg=0.00431179, d_pose=0.00255143, archive_bytes=353_808,
        rate_denominator_bytes=_DEN, source_artifact="v4d_cx1 receipt", axis_tag="[advisory]",
    )
    out = restate_pose_delta_at(1e-3, banked_at=pw1, now=cx1)
    assert out["ratio"] == pytest.approx(1.7311, rel=1e-3)
    assert out["delta_s_now"] < out["delta_s_when_banked"] < 0.0


# --- ROUND-2 SELF-REVIEW FIX: the check must not be able to skip itself -------


def test_pointer_crosscheck_default_is_cwd_independent(tmp_path, monkeypatch):
    """The first draft defaulted to the RELATIVE path '.omx/state/...'. From any other
    working directory that file does not exist, so the bar cross-check silently did not
    run -- a check that skips itself under an unstated condition is the vacuity failure
    this module exists to attack. Resolved from __file__ instead."""
    monkeypatch.chdir(tmp_path)
    # The live pointer's effective_frontier (CP135, 0.161955...) has moved BELOW the
    # PR130 floor, so a crosscheck that actually RUNS must REFUSE with the superseded-bar
    # error. The refusal IS the proof-of-run: under the first-draft relative-path bug the
    # pointer would not resolve from this foreign cwd and the call would return the floor
    # quietly. A "DID NOT RUN" warning is equally a failure.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="superseded"):
            demonstrated_floor_pr130(_DEN)
    assert not [w for w in caught if "DID NOT RUN" in str(w.message)], (
        "the cross-check must have actually run from a foreign cwd"
    )


def test_absent_pointer_WARNS_rather_than_passing_quietly(tmp_path):
    """'the check did not run' and 'the check passed' must never look alike."""
    with pytest.warns(UserWarning, match="DID NOT RUN"):
        demonstrated_floor_pr130(_DEN, frontier_pointer_path=tmp_path / "nope.json")


def test_skipping_the_crosscheck_requires_saying_so(tmp_path, monkeypatch):
    """The opt-out is an explicit sentinel, never None -- 'use the default' and
    'deliberately skip' must not share a value, which is how the defect arose."""
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        demonstrated_floor_pr130(_DEN, frontier_pointer_path=SKIP_POINTER_CROSSCHECK)
    assert not caught, "an explicit skip is silent; only an ACCIDENTAL skip warns"


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("Submission file size: 353,808 bytes", "Submission file size: 0 bytes"),
        ("Original uncompressed size: 37,545,489 bytes", "Original uncompressed size: 0 bytes"),
        ("over 600 samples", "over 0 samples"),
    ],
)
def test_nonpositive_receipt_fields_refuse_at_the_parser(tmp_path, field, replacement):
    """ROUND-4 self-review: the regexes accept '0', so total would ZeroDivisionError deep
    inside a caller rather than naming the bad receipt. And a 0-sample receipt is the
    empty-scope failure wearing a report header."""
    broken = _CX1_REPORT.replace(field, replacement)
    with pytest.raises(EvaluatorReportParseError):
        parse_evaluator_report(_write(tmp_path, broken))
