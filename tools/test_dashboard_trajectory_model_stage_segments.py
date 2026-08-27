# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""Self-protect tests for ``dashboard_trajectory_model._stage_segments`` (fixed 2026-07-03).

Regression guard for the overlapping-windows bug: the old static CE→tau→l7→Muon boundary
chain produced OVERLAPPING windows when the raw boundaries were not epoch-monotone — the
live #205 run has ``muon_start`` 726 < a DISABLED ``l7_start`` 1000, which rendered tau as
[300,1000) overlapping Muon [726,1000). The fix derives windows from the precedence-correct
``stage_at_epoch`` step function; these tests assert non-overlap on that exact case (and the
general invariant) so it can never silently return.

Run: ``.venv/bin/python -m pytest tools/test_dashboard_trajectory_model_stage_segments.py``
"""
from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dashboard_trajectory_model as dtm


def _assert_non_overlapping_covering(segs, total):
    """Adjacent windows are gap-free, non-overlapping, monotone, and span [0, total)."""
    assert segs, "expected at least one segment"
    assert segs[0][1] == 0, f"first window must start at 0: {segs}"
    assert segs[-1][2] == total, f"last window must end at total_epochs: {segs}"
    for (n0, a0, b0), (n1, a1, b1) in itertools.pairwise(segs):
        assert b0 <= a1, f"OVERLAP between {(n0, a0, b0)} and {(n1, a1, b1)}: {segs}"
        assert b0 == a1, f"GAP between {(n0, a0, b0)} and {(n1, a1, b1)}: {segs}"
        assert n0 != n1, f"adjacent windows share a stage (should have merged): {segs}"
    for _n, a, b in segs:
        assert b > a, f"empty/inverted window {(_n, a, b)}: {segs}"


def test_muon_before_disabled_l7_no_overlap():
    """THE bug case: muon_start(726) < disabled l7_start(1000). Must NOT overlap."""
    sched = {"tau_start": 300, "l7_start": 1000, "muon_start": 726, "epochs": 1000}
    segs = dtm._stage_segments(sched, 1000)
    assert segs == [("CE", 0, 300), ("tau", 300, 726), ("Muon", 726, 1000)], segs
    _assert_non_overlapping_covering(segs, 1000)
    # every window's stage must match the precedence-correct stage_at_epoch at its interior
    for name, a, b in segs:
        assert dtm.stage_at_epoch((a + b) / 2.0, sched) == name, (name, a, b)


def test_in_order_curriculum():
    sched = {"tau_start": 300, "l7_start": 600, "muon_start": 800, "epochs": 1000}
    segs = dtm._stage_segments(sched, 1000)
    assert segs == [("CE", 0, 300), ("tau", 300, 600), ("l7", 600, 800), ("Muon", 800, 1000)], segs
    _assert_non_overlapping_covering(segs, 1000)


def test_no_curriculum_all_ce():
    segs = dtm._stage_segments({"epochs": 500}, 500)
    assert segs == [("CE", 0, 500)], segs


def test_muon_only():
    sched = {"muon_start": 400, "epochs": 1000}
    segs = dtm._stage_segments(sched, 1000)
    assert segs == [("CE", 0, 400), ("Muon", 400, 1000)], segs
    _assert_non_overlapping_covering(segs, 1000)


def test_edge_cases():
    # total_epochs <= 0 / None -> empty, never raises
    assert dtm._stage_segments({"tau_start": 300}, 0) == []
    assert dtm._stage_segments({"tau_start": 300}, None) == []  # type: ignore[arg-type]
    # boundary at exactly total_epochs is a no-op (disabled)
    assert dtm._stage_segments({"tau_start": 1000, "epochs": 1000}, 1000) == [("CE", 0, 1000)]


def test_general_no_overlap_invariant_random_schedules():
    """Fuzz a spread of boundary orderings; the non-overlap invariant must always hold."""
    starts = (None, 100, 300, 500, 700, 900, 1000)
    total = 1000
    for tau in starts:
        for l7 in starts:
            for muon in starts:
                sched = {"epochs": total}
                if tau is not None:
                    sched["tau_start"] = tau
                if l7 is not None:
                    sched["l7_start"] = l7
                if muon is not None:
                    sched["muon_start"] = muon
                segs = dtm._stage_segments(sched, total)
                if not segs:
                    continue
                # non-overlap + covering + stage matches stage_at_epoch at interior
                _assert_non_overlapping_covering(segs, total)
                for name, a, b in segs:
                    assert dtm.stage_at_epoch((a + b) / 2.0, sched) == name, (sched, name, a, b)


def test_dashboard_stage_windows_delegates():
    """The dashboard's ``_stage_windows`` is now a thin alias of the canonical helper."""
    import dashboard_server as ds

    for sched in (
        {"tau_start": 300, "l7_start": 1000, "muon_start": 726, "epochs": 1000},
        {"tau_start": 300, "l7_start": 600, "muon_start": 800, "epochs": 1000},
        {"epochs": 500},
        {"epochs": 0},
    ):
        ep = int(sched.get("epochs") or 0)
        assert ds._stage_windows(sched) == dtm._stage_segments(sched, ep), sched


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"PASS {_name}")
    print("ALL PASS")
