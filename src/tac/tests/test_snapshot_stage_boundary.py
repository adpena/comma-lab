"""Tests for the pre-stage-8 checkpoint watcher's pure decision logic.

Covers the adversarial-review-hardened core of ``tools/snapshot_run_checkpoint_at_stage_boundary.py``:
the per-poll action decision (rolling-copy vs freeze vs noop) and the restart-robust manifest-staleness
exit gate. These are the bug-class guards: the freeze must trigger on ``has_muon`` (not stage_index), and
the exit must NOT trip on a run restart (manifest keeps advancing).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "snapshot_run_checkpoint_at_stage_boundary.py"
_spec = importlib.util.spec_from_file_location("_snap_boundary", _TOOL)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


# ---- decide_action: the freeze trigger is has_muon, not stage_index -------------------------------


def test_freeze_when_has_muon_true_regardless_of_stage_index():
    # has_muon True is the ROBUST stage-8 signal; even an unexpected stage_index must freeze.
    assert mod.decide_action({"has_muon": True, "stage_index": 6, "epoch_in_stage": 10}, None, 5) == mod.FREEZE
    assert mod.decide_action({"has_muon": True, "stage_index": 7, "epoch_in_stage": 0}, (6, 100), 5) == mod.FREEZE


def test_no_copy_before_start_stage_index():
    # stages 0-4 (below start=5) must NOT roll-copy (avoids 8h of needless copies).
    for si in range(5):
        assert mod.decide_action({"has_muon": False, "stage_index": si, "epoch_in_stage": 25}, None, 5) == mod.NOOP


def test_copy_when_in_ending_stage_and_epoch_advanced():
    assert mod.decide_action({"has_muon": False, "stage_index": 5, "epoch_in_stage": 25}, None, 5) == mod.COPY
    assert mod.decide_action({"has_muon": False, "stage_index": 6, "epoch_in_stage": 50}, (6, 25), 5) == mod.COPY


def test_noop_when_epoch_not_advanced():
    # Same (stage_index, epoch) as last_copied → no redundant copy.
    assert mod.decide_action({"has_muon": False, "stage_index": 6, "epoch_in_stage": 50}, (6, 50), 5) == mod.NOOP


def test_has_muon_beats_stage_index_below_start():
    # Defensive: if has_muon is somehow True while stage_index reads below start, FREEZE still wins.
    assert mod.decide_action({"has_muon": True, "stage_index": 2, "epoch_in_stage": 0}, None, 5) == mod.FREEZE


def test_missing_fields_default_to_noop():
    # A partial/transient manifest must never spuriously copy or freeze.
    assert mod.decide_action({}, None, 5) == mod.NOOP
    assert mod.decide_action({"stage_index": 6}, None, 5) == mod.COPY  # epoch -1 != None → one copy, then noop
    assert mod.decide_action({"stage_index": 6}, (6, -1), 5) == mod.NOOP


# ---- is_stale: restart-robust exit gate ----------------------------------------------------------


def test_not_stale_within_threshold():
    # A run actively checkpointing (mtime recent) is NOT stale → watcher keeps going.
    assert mod.is_stale(reference_time=1000.0, now=1000.0 + 60, threshold_s=600) is False


def test_stale_after_threshold():
    # No checkpoint for > threshold → run genuinely stopped → exit.
    assert mod.is_stale(reference_time=1000.0, now=1000.0 + 700, threshold_s=600) is True


def test_restart_gap_under_threshold_is_not_stale():
    # THE F2 FIX: a restart gap (~2 min) is far under the 600s threshold → watcher survives the restart.
    assert mod.is_stale(reference_time=1000.0, now=1000.0 + 120, threshold_s=600) is False


def test_missing_mtime_is_never_stale():
    # No manifest yet (run warming up) must not trigger exit.
    assert mod.is_stale(reference_time=None, now=999999.0, threshold_s=600) is False


# ---- next_change_reference: observe-relative tracking (the R2-1 fix) ------------------------------


def test_change_reference_resets_on_advance():
    # Manifest advanced (mtime changed) → reference clock resets to now.
    new_mtime, new_ref = mod.next_change_reference(prev_mtime=100.0, cur_mtime=200.0, prev_reference=50.0, now=1000.0)
    assert new_mtime == 200.0 and new_ref == 1000.0


def test_change_reference_holds_when_no_advance():
    # Same mtime → reference unchanged (staleness keeps accumulating toward exit).
    new_mtime, new_ref = mod.next_change_reference(prev_mtime=200.0, cur_mtime=200.0, prev_reference=50.0, now=1000.0)
    assert new_mtime == 200.0 and new_ref == 50.0


def test_change_reference_ignores_none_mtime():
    # Transient missing manifest read → keep the prior reference (do not reset, do not crash).
    new_mtime, new_ref = mod.next_change_reference(prev_mtime=200.0, cur_mtime=None, prev_reference=50.0, now=1000.0)
    assert new_mtime == 200.0 and new_ref == 50.0


def test_r2_1_stale_at_startup_does_not_false_exit_until_threshold():
    # THE R2-1 GUARD: a pre-existing OLD mtime at watcher start must NOT immediately exit. The reference is
    # seeded at start; only THRESHOLD seconds with no observed advance triggers exit.
    start = 10_000.0
    old_mtime = 0.0  # manifest is ~10000s old at startup (e.g. watcher relaunched in a quiet window)
    ref = start  # seeded at watcher start
    # First poll, no advance yet, only 5s elapsed → NOT stale (old absolute mtime would have wrongly fired).
    m, ref = mod.next_change_reference(old_mtime, old_mtime, ref, start + 5)
    assert mod.is_stale(ref, start + 5, 600) is False
    # The run then advances the manifest → reference resets, stays alive.
    m, ref = mod.next_change_reference(old_mtime, 12_345.0, ref, start + 50)
    assert mod.is_stale(ref, start + 100, 600) is False
    # Now it genuinely stops: no advance for > threshold → stale.
    m, ref = mod.next_change_reference(12_345.0, 12_345.0, ref, start + 700)
    assert mod.is_stale(ref, start + 700, 600) is True
