"""Tests for the fleet watcher (codex_arm_watch) + dispatcher liveness surfaces.

Operator directive 2026-08-04: MAIN receives arm-completion notifications
instead of being polled. Two landings under test here (two-landing rule):
  1. the watcher emitter (event diff → notification lines), and
  2. the dispatcher instrument fix (status reads .done receipts — the
     rc=0-shows-as-DIED mislabel — plus watcher-liveness surfacing).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex_arm_watch as watch_mod  # noqa: E402


def _write(runs: Path, name: str, content: str) -> None:
    (runs / name).write_text(content)


def test_baseline_silence(tmp_path: Path) -> None:
    """Pre-existing receipts must not emit — only NEW terminal events notify."""
    _write(tmp_path, "old.done", "rc=0 elapsed=10 gen=1\n")
    snap = watch_mod._snapshot(tmp_path)
    assert watch_mod.format_events(tmp_path, snap, snap) == []


def test_done_event_emits_receipt_and_final_message(tmp_path: Path) -> None:
    before = watch_mod._snapshot(tmp_path)
    _write(tmp_path, "arm1.done", "launched pid=9 t=0\nrc=0 elapsed=99 gen=2\n")
    _write(tmp_path, "arm1.last.txt", "FINAL: verdict text here\nsecond line")
    lines = watch_mod.format_events(tmp_path, before, watch_mod._snapshot(tmp_path))
    assert len(lines) == 1
    assert lines[0].startswith("ARM arm1 FINISHED rc=0 elapsed=99 gen=2")
    assert "FINAL: verdict text here" in lines[0]


def test_relay_event_emits_only_appended_lines(tmp_path: Path) -> None:
    _write(tmp_path, "arm2.relay", "relay gen 1 -> 2\n")
    before = watch_mod._snapshot(tmp_path)
    _write(tmp_path, "arm2.relay", "relay gen 1 -> 2\nrelay gen 2 -> 3\n")
    lines = watch_mod.format_events(tmp_path, before, watch_mod._snapshot(tmp_path))
    assert lines == ["ARM arm2 RELAYED relay gen 2 -> 3"]


def test_missing_last_txt_is_not_fatal(tmp_path: Path) -> None:
    before = watch_mod._snapshot(tmp_path)
    _write(tmp_path, "arm3.done", "rc=1 elapsed=5 gen=1\n")
    lines = watch_mod.format_events(tmp_path, before, watch_mod._snapshot(tmp_path))
    assert len(lines) == 1 and "(no final message file)" in lines[0]


# --- dispatcher instrument fix (codex_arm_queue) ---------------------------------


def test_queue_done_receipt_and_watcher_line(tmp_path: Path, monkeypatch) -> None:
    import codex_arm_queue as queue_mod

    monkeypatch.setattr(queue_mod, "RUNS", tmp_path)
    # DIED-mislabel fix: an rc=0 .done must resolve to its receipt, not None.
    assert queue_mod._done_receipt("ghost") is None
    _write(tmp_path, "fin.done", "launched pid=1 t=0\nrc=0 elapsed=44 gen=1\n")
    assert queue_mod._done_receipt("fin") == "rc=0 elapsed=44 gen=1"

    # Watcher liveness ladder: absent -> NOT RUNNING, fresh -> ALIVE, old -> STALE.
    assert "NOT RUNNING" in queue_mod._watcher_line()
    hb = tmp_path / "_watcher.alive"
    hb.touch()
    assert "ALIVE" in queue_mod._watcher_line()
    old = time.time() - 300
    os.utime(hb, (old, old))
    assert "STALE" in queue_mod._watcher_line()
