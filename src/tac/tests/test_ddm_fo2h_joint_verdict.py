"""Guard the ddm_fo2h adjudicator against the one-leg-verdict defect.

Generation 1 of this arm emitted `SUPPLIER CONFIRMED-HARDENED` from a branch that read the seg
realization efficiency `eta` and nothing else, while the *same* output dict carried a pose leg
worth +0.001424 S -- 3.9x the seg gain, opposite in sign.  The contest score is
`100*d_seg + sqrt(10*d_pose) + 25*bytes/37_545_489`; a label that inspects one term and says
"SUPPLIER" is a claim about S that S never authorised.

These tests pin the two functions that fix it:

* `pose_lineage_bounds` -- the eta gate decodes GT through PyAV while the contest scores DALI, so
  turning a measured pose RATIO into a Delta-S crosses lineages.  Both defensible transfer
  assumptions must be emitted, never one silently.
* the joint verdict rule -- a SUPPLIER label requires the joint Delta-S to be negative under BOTH
  bounds; straddling zero is INDETERMINATE-ON-LINEAGE, not a pass.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[3] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

fo2h = pytest.importorskip("ddm_fo2h_eta_adjudicate")


def _rows(before: float, after: float, n: int = 4) -> list[dict]:
    """n identical pose rows, so mean(after)/mean(before) is exactly after/before."""
    return [{"d_pose_before": before, "d_pose_after": after, "d_pose_ratio": after / before,
             "flips_before": 10, "flips_after": 5, "n_described_ring0": 10,
             "fixed": 5, "introduced": 0, "pair": i} for i in range(n)]


# --------------------------------------------------------------------------------------------
# pose_lineage_bounds
# --------------------------------------------------------------------------------------------
def test_lineage_bounds_emit_both_transfer_assumptions() -> None:
    out = fo2h.pose_lineage_bounds(_rows(1.0e-4, 1.4e-4))
    assert "delta_S_pose_ratio_transfers" in out
    assert "delta_S_pose_excess_transfers" in out
    assert "PyAV" in out["gt_lineage"]


def test_lineage_bounds_ratio_transfer_matches_closed_form() -> None:
    before, after = 1.0e-4, 1.4e-4
    out = fo2h.pose_lineage_bounds(_rows(before, after))
    base = (10.0 * fo2h.D_POSE_N600) ** 0.5
    want = (10.0 * fo2h.D_POSE_N600 * (after / before)) ** 0.5 - base
    assert out["delta_S_pose_ratio_transfers"] == pytest.approx(want, rel=1e-12)


def test_excess_transfer_is_the_harsher_bound_when_local_base_is_inflated() -> None:
    """The measured case: local base ~15x the contest base, edit worsens pose.

    A fixed absolute excess is a far larger fractional insult on the smaller contest base, so the
    excess bound must exceed the ratio bound.  If this ever inverts, the bracket is inverted and
    the joint verdict would be read off the wrong edge.
    """
    out = fo2h.pose_lineage_bounds(_rows(1.0413e-4, 1.4293e-4))
    assert out["local_base_over_contest_base"] > 10.0
    assert out["delta_S_pose_excess_transfers"] > out["delta_S_pose_ratio_transfers"] > 0.0


def test_pose_improvement_gives_negative_delta_s_on_both_bounds() -> None:
    """pn2's n=12 regime: the projection removed the pose tax (ratio < 1)."""
    out = fo2h.pose_lineage_bounds(_rows(1.0e-4, 0.7935e-4))
    assert out["delta_S_pose_ratio_transfers"] < 0.0
    assert out["delta_S_pose_excess_transfers"] < 0.0


def test_lineage_bounds_empty_rows_return_empty() -> None:
    assert fo2h.pose_lineage_bounds([]) == {}


def test_absolute_excess_is_reported_and_signed() -> None:
    out = fo2h.pose_lineage_bounds(_rows(1.0e-4, 1.4e-4))
    assert out["absolute_excess_local"] == pytest.approx(4.0e-5, rel=1e-9)
    assert out["implied_contest_ratio_excess_transfers"] > 1.0


# --------------------------------------------------------------------------------------------
# the joint verdict rule
# --------------------------------------------------------------------------------------------
def _joint(seg_dS: float, ratio_dS: float, excess_dS: float) -> str:
    """Call the SHIPPED rule, never a copy of it.

    An earlier draft of this file re-implemented the branch here.  That is the class-2 fake: edit
    `main()`'s rule and every one of these tests would still pass.  `joint_verdict` is module
    level precisely so the test and the run take the same code path.
    """
    return fo2h.joint_verdict({"r": seg_dS + ratio_dS, "e": seg_dS + excess_dS})


def test_measured_case_is_net_non_supplier() -> None:
    """The n=48 out-of-sample measurement: seg -0.000336, pose +0.001424 / +0.013075."""
    assert _joint(-0.000336, +0.001424, +0.013075) == "CHANNEL NET NON-SUPPLIER"


def test_seg_gain_alone_never_licenses_supplier() -> None:
    """A seg leg that clears its bar is NOT a supplier if either pose bound outweighs it."""
    assert _joint(-0.000505, +0.001424, +0.013075) != "CHANNEL NET SUPPLIER"


def test_supplier_requires_both_bounds_negative() -> None:
    assert _joint(-0.002, -0.0005, -0.0009) == "CHANNEL NET SUPPLIER"


def test_straddling_bounds_are_indeterminate_not_a_pass() -> None:
    """The lineage question is unresolved locally; straddling zero must refuse a verdict."""
    assert _joint(-0.001, -0.0005, +0.004) == "CHANNEL INDETERMINATE-ON-LINEAGE"


def test_exact_zero_worst_case_is_indeterminate_not_supplier() -> None:
    assert _joint(-0.001, -0.0005, +0.001) == "CHANNEL INDETERMINATE-ON-LINEAGE"


def test_pose_improving_channel_can_supply() -> None:
    """pn2's premise, had it held out-of-sample: pose improves, seg supplies, channel supplies."""
    out = fo2h.pose_lineage_bounds(_rows(1.0e-4, 0.7935e-4))
    v = _joint(-0.000505, out["delta_S_pose_ratio_transfers"],
               out["delta_S_pose_excess_transfers"])
    assert v == "CHANNEL NET SUPPLIER"


# --------------------------------------------------------------------------------------------
# the frozen pins these verdicts are drawn against
# --------------------------------------------------------------------------------------------
def test_frozen_bar_is_fo1_breakeven() -> None:
    assert pytest.approx(0.5196321126365346, rel=1e-15) == fo2h.FO1_BREAKEVEN_ETA


def test_contest_pose_base_is_the_cuda_number_not_a_local_one() -> None:
    """D_POSE_N600 must stay the contest-CUDA term; swapping in a PyAV-lineage value would
    silently inflate the base ~15x and shrink every pose Delta-S toward zero."""
    assert pytest.approx(6.885643e-06, rel=1e-12) == fo2h.D_POSE_N600
    assert math.isclose((10.0 * fo2h.D_POSE_N600) ** 0.5, 0.0082980, abs_tol=1e-6)


def test_joint_verdict_refuses_on_empty_bounds() -> None:
    assert fo2h.joint_verdict({}) == "JOINT UNCOMPUTED-NO-POSE-ROWS"


def test_nan_bound_can_never_produce_a_supplier_label() -> None:
    """A missing/degenerate pose measurement must fail SAFE, never pass."""
    assert "SUPPLIER" not in fo2h.joint_verdict({"r": float("nan"), "e": -0.002}).replace(
        "NON-SUPPLIER", "")


def test_pooled_eta_is_a_ratio_of_sums() -> None:
    """A mean of per-pair etas would weight a 5-flip pair like a 200-flip one."""
    rows = [{"flips_before": 100, "flips_after": 0, "n_described_ring0": 100},
            {"flips_before": 10, "flips_after": 10, "n_described_ring0": 100}]
    assert fo2h.pooled_eta(rows) == pytest.approx(100 / 200)
