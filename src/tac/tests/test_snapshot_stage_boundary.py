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


# ---- IO integration: the real _refresh_snapshot -> _freeze sequence (R3-5 self-protect) -----------


def _write_fake_ckpt(out_dir: Path, *, stage_index: int, has_muon: bool, epoch: int, tag: bytes) -> None:
    """Write a minimal fake checkpoint_state.pt + manifest into ``out_dir`` (the live-run layout)."""
    import json

    import torch

    torch.save(
        {"ema_decoder": {"w": torch.arange(4.0)}, "ema_latents": torch.zeros(3, 2), "_tag": tag.decode()},
        out_dir / "torch_vehicle_checkpoint_state.pt",
    )
    (out_dir / "torch_vehicle_checkpoint_manifest.json").write_text(
        json.dumps({"stage_index": stage_index, "has_muon": has_muon, "epoch_in_stage": epoch})
    )


def test_refresh_then_freeze_integration(tmp_path):
    """Drive the REAL copy/freeze IO over stage6 -> stage7 -> stage8: the frozen snapshot must be the
    last stage-7 copy, with the warm-start pair extracted and a non-skewed FROZEN.marker."""
    import json

    out_dir = tmp_path / "run"
    snap_dir = tmp_path / "snap"
    out_dir.mkdir()
    log = snap_dir / "watcher.log"
    snap_dir.mkdir()
    src_pt = out_dir / "torch_vehicle_checkpoint_state.pt"
    src_man = out_dir / "torch_vehicle_checkpoint_manifest.json"

    # stage6 rolling copy.
    _write_fake_ckpt(out_dir, stage_index=5, has_muon=False, epoch=50, tag=b"STAGE6")
    mod._refresh_snapshot(src_pt, src_man, snap_dir)
    assert (snap_dir / "torch_vehicle_checkpoint_state.pt").exists()
    # stage7 rolling copy (overwrites — the snapshot tracks the latest pre-stage-8 state).
    _write_fake_ckpt(out_dir, stage_index=6, has_muon=False, epoch=100, tag=b"STAGE7-END")
    mod._refresh_snapshot(src_pt, src_man, snap_dir)
    snap_man = json.loads((snap_dir / "torch_vehicle_checkpoint_manifest.json").read_text())
    assert snap_man["stage_index"] == 6 and snap_man["has_muon"] is False

    # stage8 starts in the SOURCE — but the watcher freezes the EXISTING (stage-7) snap_dir copy.
    _write_fake_ckpt(out_dir, stage_index=7, has_muon=True, epoch=0, tag=b"STAGE8-FIRST")
    mod._freeze(snap_dir, (6, 100), out_dir, log)
    assert (snap_dir / "best_ema_decoder.pt").exists() and (snap_dir / "best_ema_latents.pt").exists()
    marker = json.loads((snap_dir / "FROZEN.marker").read_text())
    assert marker["warm_start_pair_emitted"] is True
    assert marker["snapshot_manifest_has_muon"] is False  # the frozen copy is genuinely pre-stage-8
    assert marker["last_copied_stage_epoch"] == [6, 100]


def test_freeze_skew_detected(tmp_path):
    """If the frozen snap_dir manifest is has_muon=True (the ~1e-4 blob-then-manifest skew), the marker
    must record it (detectable, not silent)."""
    import json

    out_dir = tmp_path / "run"
    snap_dir = tmp_path / "snap"
    out_dir.mkdir()
    snap_dir.mkdir()
    # Simulate the skew: the snapshot copy itself caught a has_muon=True state.
    _write_fake_ckpt(snap_dir, stage_index=7, has_muon=True, epoch=0, tag=b"SKEW")
    mod._freeze(snap_dir, (7, 0), out_dir, snap_dir / "watcher.log")
    marker = json.loads((snap_dir / "FROZEN.marker").read_text())
    assert marker["snapshot_manifest_has_muon"] is True  # skew surfaced


def test_emit_warm_start_pair_missing_pt_is_graceful(tmp_path):
    """No .pt in snap_dir → extraction returns False (caught), never crashes the daemon."""
    assert mod._emit_warm_start_pair(tmp_path, tmp_path / "watcher.log") is False
