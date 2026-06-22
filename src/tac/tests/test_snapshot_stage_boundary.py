"""Tests for the per-stage checkpoint watcher's pure logic + IO preservation.

Covers ``tools/snapshot_run_checkpoint_at_stage_boundary.py`` (adversarial rounds 1-4):
* stage-transition detection (preserve the ENDING stage) + the rolling-copy decision,
* restart-robust, observe-relative manifest-staleness exit (a restart must NOT trip it),
* the real _preserve IO over a simulated stage transition (incl. skew detection + graceful missing-.pt).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "snapshot_run_checkpoint_at_stage_boundary.py"
_spec = importlib.util.spec_from_file_location("_snap_boundary", _TOOL)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


# ---- stage_changed: the transition trigger -------------------------------------------------------


def test_stage_changed_true_on_transition():
    assert mod.stage_changed("stage4_v332_qat", "stage5_c1a_l7") is True


def test_stage_changed_false_when_same():
    assert mod.stage_changed("stage5_c1a_l7", "stage5_c1a_l7") is False


def test_stage_changed_false_at_startup_no_prior():
    # First poll has no prior stage → no spurious preserve of a stage we never observed the end of.
    assert mod.stage_changed(None, "stage4_v332_qat") is False


def test_stage_changed_false_on_none_current():
    assert mod.stage_changed("stage4_v332_qat", None) is False


# ---- should_roll_copy: the rolling working-copy decision -----------------------------------------


def test_no_copy_before_start_stage_index():
    for si in range(5):
        assert mod.should_roll_copy({"stage_index": si, "epoch_in_stage": 25}, None, 5) is False


def test_copy_default_start_zero_copies_every_stage():
    # default start=0 → preserve every boundary from the current stage onward.
    assert mod.should_roll_copy({"stage_index": 3, "epoch_in_stage": 100}, None, 0) is True


def test_copy_when_epoch_advanced():
    assert mod.should_roll_copy({"stage_index": 6, "epoch_in_stage": 50}, (6, 25), 0) is True


def test_noop_when_epoch_not_advanced():
    assert mod.should_roll_copy({"stage_index": 6, "epoch_in_stage": 50}, (6, 50), 0) is False


def test_missing_fields_never_copy():
    # A fieldless/transient manifest reads stage_index=-1, which is below ANY start → never copies
    # (defensive: a bad/partial read must not produce a bogus snapshot).
    assert mod.should_roll_copy({}, None, 0) is False
    assert mod.should_roll_copy({}, None, -1) is True  # only an explicit start=-1 would admit it


# ---- is_stale / next_change_reference: restart-robust observe-relative exit -----------------------


def test_not_stale_within_threshold():
    assert mod.is_stale(reference_time=1000.0, now=1000.0 + 60, threshold_s=600) is False


def test_stale_after_threshold():
    assert mod.is_stale(reference_time=1000.0, now=1000.0 + 700, threshold_s=600) is True


def test_restart_gap_under_threshold_is_not_stale():
    assert mod.is_stale(reference_time=1000.0, now=1000.0 + 120, threshold_s=600) is False


def test_missing_reference_is_never_stale():
    assert mod.is_stale(reference_time=None, now=999999.0, threshold_s=600) is False


def test_change_reference_resets_on_advance():
    m, ref = mod.next_change_reference(prev_mtime=100.0, cur_mtime=200.0, prev_reference=50.0, now=1000.0)
    assert m == 200.0 and ref == 1000.0


def test_change_reference_holds_when_no_advance():
    m, ref = mod.next_change_reference(prev_mtime=200.0, cur_mtime=200.0, prev_reference=50.0, now=1000.0)
    assert m == 200.0 and ref == 50.0


def test_change_reference_ignores_none_mtime():
    m, ref = mod.next_change_reference(prev_mtime=200.0, cur_mtime=None, prev_reference=50.0, now=1000.0)
    assert m == 200.0 and ref == 50.0


def test_stale_at_startup_does_not_false_exit_until_threshold():
    start = 10_000.0
    old = 0.0
    ref = start  # seeded at watcher start
    m, ref = mod.next_change_reference(old, old, ref, start + 5)
    assert mod.is_stale(ref, start + 5, 600) is False  # old absolute mtime would have wrongly fired
    m, ref = mod.next_change_reference(old, 12_345.0, ref, start + 50)
    assert mod.is_stale(ref, start + 100, 600) is False
    m, ref = mod.next_change_reference(12_345.0, 12_345.0, ref, start + 700)
    assert mod.is_stale(ref, start + 700, 600) is True


# ---- IO integration: the real _refresh_snapshot -> _preserve sequence -----------------------------


def _write_fake_ckpt(out_dir: Path, *, stage_name: str, has_muon: bool, epoch: int) -> None:
    import json

    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"ema_decoder": {"w": torch.arange(4.0)}, "ema_latents": torch.zeros(3, 2)},
        out_dir / "torch_vehicle_checkpoint_state.pt",
    )
    (out_dir / "torch_vehicle_checkpoint_manifest.json").write_text(
        json.dumps({"stage_name": stage_name, "has_muon": has_muon, "epoch_in_stage": epoch})
    )


def test_roll_then_preserve_stage_end_integration(tmp_path):
    """Roll-copy stage4, then at the stage4->5 transition _preserve must capture the stage-4 end state."""
    import json

    out_dir = tmp_path / "run"
    rolling = out_dir / ".rolling_snapshot"
    log = out_dir / "stage_snapshots" / "watcher.log"

    # stage 4 rolling copy.
    _write_fake_ckpt(out_dir, stage_name="stage4_v332_qat", has_muon=False, epoch=400)
    mod._refresh_snapshot(
        out_dir / "torch_vehicle_checkpoint_state.pt", out_dir / "torch_vehicle_checkpoint_manifest.json", rolling
    )
    # transition 4->5: preserve the rolling (stage-4-end) copy.
    dest = out_dir / "stage_snapshots" / "stage4_v332_qat_end"
    assert mod._preserve(rolling, dest, reason="stage4->stage5", log_path=log) is True
    assert (dest / "best_ema_decoder.pt").exists() and (dest / "best_ema_latents.pt").exists()
    marker = json.loads((dest / "STAGE_END.marker").read_text())
    assert marker["warm_start_pair_emitted"] is True
    assert marker["snapshot_manifest_stage"] == "stage4_v332_qat"
    assert marker["snapshot_manifest_has_muon"] is False


def test_preserve_records_has_muon_for_pre_stage8(tmp_path):
    """The pre-stage-8 (stage7->8) preserve must record has_muon for skew detection / provenance."""
    import json

    out_dir = tmp_path / "run"
    rolling = out_dir / ".rolling_snapshot"
    _write_fake_ckpt(rolling, stage_name="stage7_sigma_sweep", has_muon=False, epoch=3000)
    dest = out_dir / "pre_stage8_snapshot"
    mod._preserve(rolling, dest, reason="pre_stage8 (has_muon)", log_path=out_dir / "w.log")
    marker = json.loads((dest / "STAGE_END.marker").read_text())
    assert marker["snapshot_manifest_has_muon"] is False  # the preserved copy is genuinely pre-stage-8


def test_preserve_graceful_when_no_rolling_pt(tmp_path):
    """No rolling .pt yet (watcher just started) → _preserve returns False, never crashes."""
    assert mod._preserve(tmp_path / "empty", tmp_path / "dest", reason="x", log_path=tmp_path / "w.log") is False


def test_emit_warm_start_pair_missing_pt_is_graceful(tmp_path):
    assert mod._emit_warm_start_pair(tmp_path, tmp_path / "watcher.log") is False


def test_rolling_seed_reads_prior_stage(tmp_path):
    # R4-11: last_stage seeds from the rolling copy's manifest, so a watcher restart across a transition
    # still detects + preserves the prior stage's end. A missing rolling dir seeds None (no spurious seed).
    rolling = tmp_path / ".rolling_snapshot"
    _write_fake_ckpt(rolling, stage_name="stage4_v332_qat", has_muon=False, epoch=400)
    seed = (mod._read_manifest(rolling / "torch_vehicle_checkpoint_manifest.json") or {}).get("stage_name")
    assert seed == "stage4_v332_qat"
    assert (mod._read_manifest(tmp_path / "nope" / "m.json") or {}).get("stage_name") is None
