"""Tests for tools/subagent_checkpoint.py — crash-resume checkpoint tool.

Memory ref: feedback_subagent_crash_resume_discipline_landed_20260514.md.

This is Layer 1 of the subagent crash-resume permanent-fix per CLAUDE.md
"Bugs must be permanently fixed AND self-protected against" non-negotiable.
It covers: append-only JSONL writes under fcntl lock (per Catalog #131),
schema validation, latest/read API, and the CLI surface.

The Layer 3 STRICT preflight gate is covered by
``src/tac/tests/test_check_206_subagent_crash_resume_discipline.py``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBAGENT_CHECKPOINT_PATH = REPO_ROOT / "tools" / "subagent_checkpoint.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "subagent_checkpoint", SUBAGENT_CHECKPOINT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def isolated_state_dir(tmp_path, monkeypatch):
    """Point the module's state paths at tmp_path so tests are isolated."""
    module = _load_module()
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True)
    monkeypatch.setattr(module, "STATE_DIR", state_dir)
    monkeypatch.setattr(
        module, "JSONL_PATH", state_dir / "subagent_progress.jsonl"
    )
    monkeypatch.setattr(
        module, "LOCK_PATH", state_dir / ".subagent_progress.lock"
    )
    return module


def test_append_minimal_record(isolated_state_dir):
    mod = isolated_state_dir
    rec = mod.append_checkpoint(
        subagent_id="TEST-A",
        step=1,
        status="in_progress",
        files_touched=["src/tac/foo.py"],
        next_action="continue",
    )
    assert rec["subagent_id"] == "TEST-A"
    assert rec["step"] == 1
    assert rec["status"] == "in_progress"
    assert rec["files_touched"] == ["src/tac/foo.py"]
    assert rec["next_action"] == "continue"
    assert "written_at_utc" in rec
    assert "pid" in rec
    assert "host" in rec
    # JSONL file exists with one row
    rows = mod.JSONL_PATH.read_text().strip().split("\n")
    assert len(rows) == 1


def test_append_with_all_fields(isolated_state_dir):
    mod = isolated_state_dir
    rec = mod.append_checkpoint(
        subagent_id="TEST-B",
        step=5,
        status="blocked",
        files_touched=["a.py", "b.py", "c.py"],
        next_action="awaiting council review of design",
        notes="multi\nline\nnotes ok",
        parent_id_or_session="session-XYZ",
    )
    assert rec["status"] == "blocked"
    assert rec["files_touched"] == ["a.py", "b.py", "c.py"]
    assert rec["notes"] == "multi\nline\nnotes ok"
    assert rec["parent_id_or_session"] == "session-XYZ"


def test_step_complete_literal(isolated_state_dir):
    mod = isolated_state_dir
    rec = mod.append_checkpoint(
        subagent_id="TEST-C",
        step="complete",
        status="complete",
        files_touched=[],
        next_action="",
        notes="",
    )
    assert rec["step"] == "complete"
    assert rec["status"] == "complete"


def test_reject_empty_subagent_id(isolated_state_dir):
    mod = isolated_state_dir
    with pytest.raises(ValueError, match="subagent_id"):
        mod.append_checkpoint(
            subagent_id="",
            step=1,
            status="in_progress",
            files_touched=[],
            next_action="x",
        )


def test_reject_whitespace_only_subagent_id(isolated_state_dir):
    mod = isolated_state_dir
    with pytest.raises(ValueError, match="subagent_id"):
        mod.append_checkpoint(
            subagent_id="   ",
            step=1,
            status="in_progress",
            files_touched=[],
            next_action="x",
        )


def test_reject_newline_in_subagent_id(isolated_state_dir):
    mod = isolated_state_dir
    with pytest.raises(ValueError, match="newlines"):
        mod.append_checkpoint(
            subagent_id="bad\nid",
            step=1,
            status="in_progress",
            files_touched=[],
            next_action="x",
        )


def test_reject_invalid_status(isolated_state_dir):
    mod = isolated_state_dir
    with pytest.raises(ValueError, match="status"):
        mod.append_checkpoint(
            subagent_id="TEST",
            step=1,
            status="bogus",
            files_touched=[],
            next_action="x",
        )


def test_reject_invalid_step_string(isolated_state_dir):
    mod = isolated_state_dir
    with pytest.raises(ValueError, match="step"):
        mod.append_checkpoint(
            subagent_id="TEST",
            step="halfway",
            status="in_progress",
            files_touched=[],
            next_action="x",
        )


def test_reject_invalid_files_touched(isolated_state_dir):
    mod = isolated_state_dir
    with pytest.raises(ValueError, match="files_touched"):
        mod.append_checkpoint(
            subagent_id="TEST",
            step=1,
            status="in_progress",
            files_touched="not-a-list",  # type: ignore[arg-type]
            next_action="x",
        )


def test_read_empty_returns_empty_list(isolated_state_dir):
    mod = isolated_state_dir
    assert mod.read_checkpoints() == []
    assert mod.read_checkpoints("nobody") == []
    assert mod.latest_checkpoint("nobody") is None


def test_read_filters_by_subagent_id(isolated_state_dir):
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="A",
        step=1,
        status="in_progress",
        files_touched=[],
        next_action="x",
    )
    mod.append_checkpoint(
        subagent_id="B",
        step=1,
        status="in_progress",
        files_touched=[],
        next_action="y",
    )
    mod.append_checkpoint(
        subagent_id="A",
        step=2,
        status="in_progress",
        files_touched=[],
        next_action="z",
    )
    a_rows = mod.read_checkpoints("A")
    b_rows = mod.read_checkpoints("B")
    all_rows = mod.read_checkpoints()
    assert len(a_rows) == 2
    assert len(b_rows) == 1
    assert len(all_rows) == 3


def test_latest_returns_most_recent(isolated_state_dir):
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="A",
        step=1,
        status="in_progress",
        files_touched=[],
        next_action="step-1",
    )
    mod.append_checkpoint(
        subagent_id="A",
        step=2,
        status="in_progress",
        files_touched=[],
        next_action="step-2",
    )
    latest = mod.latest_checkpoint("A")
    assert latest is not None
    assert latest["step"] == 2
    assert latest["next_action"] == "step-2"


def test_in_progress_to_complete_transition(isolated_state_dir):
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="X",
        step=1,
        status="in_progress",
        files_touched=["a.py"],
        next_action="continue",
    )
    mod.append_checkpoint(
        subagent_id="X",
        step="complete",
        status="complete",
        files_touched=["a.py", "b.py"],
        next_action="",
    )
    rows = mod.read_checkpoints("X")
    assert len(rows) == 2
    assert rows[-1]["status"] == "complete"
    assert rows[-1]["step"] == "complete"


def test_concurrent_appends_serialize_safely(isolated_state_dir):
    """Sister of Catalog #131: concurrent appenders must NOT corrupt JSONL.

    Spawn N threads, each writing M records. After join, the JSONL file must
    have exactly N*M lines and every line must parse as valid JSON.
    """
    mod = isolated_state_dir
    n_threads = 8
    rows_per_thread = 5

    def worker(tid: int) -> None:
        for j in range(rows_per_thread):
            mod.append_checkpoint(
                subagent_id=f"T-{tid}",
                step=j + 1,
                status="in_progress",
                files_touched=[f"file_{tid}_{j}.py"],
                next_action=f"thread {tid} step {j}",
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = mod.JSONL_PATH.read_text().strip().split("\n")
    assert len(rows) == n_threads * rows_per_thread
    # Every line parses
    parsed = [json.loads(r) for r in rows]
    # Every thread wrote rows_per_thread records
    by_tid: dict[str, int] = {}
    for rec in parsed:
        by_tid[rec["subagent_id"]] = by_tid.get(rec["subagent_id"], 0) + 1
    assert all(count == rows_per_thread for count in by_tid.values())
    assert len(by_tid) == n_threads


def test_cli_write_smoke(tmp_path):
    """End-to-end CLI write: invoke as subprocess against a temp HOME."""
    # Make a tiny throwaway state dir using the env-override pattern: we can't
    # easily monkeypatch a subprocess, so we route via a temp REPO root.
    # The CLI uses module-level paths derived from __file__, so subprocess
    # invocation always writes to the real repo. Instead we drive the module
    # entrypoint directly via a forked process with PYTHONPATH set to override
    # REPO_ROOT. The simplest portable approach: import the module in-process,
    # monkeypatch its globals, and call main() with argv.
    mod = _load_module()
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True)
    mod.STATE_DIR = state_dir
    mod.JSONL_PATH = state_dir / "subagent_progress.jsonl"
    mod.LOCK_PATH = state_dir / ".subagent_progress.lock"

    rc = mod.main([
        "--subagent-id", "CLI-TEST",
        "--step", "3",
        "--status", "in_progress",
        "--files-touched", "a.py,b.py",
        "--next-action", "next thing",
    ])
    assert rc == 0
    rows = mod.JSONL_PATH.read_text().strip().split("\n")
    assert len(rows) == 1
    rec = json.loads(rows[0])
    assert rec["subagent_id"] == "CLI-TEST"
    assert rec["files_touched"] == ["a.py", "b.py"]


def test_cli_read_missing_subagent_returns_2(tmp_path, capsys):
    mod = _load_module()
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True)
    mod.STATE_DIR = state_dir
    mod.JSONL_PATH = state_dir / "subagent_progress.jsonl"
    mod.LOCK_PATH = state_dir / ".subagent_progress.lock"

    rc = mod.main(["read", "--subagent-id", "nobody"])
    assert rc == 2


def test_cli_read_returns_existing_records(tmp_path, capsys):
    mod = _load_module()
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True)
    mod.STATE_DIR = state_dir
    mod.JSONL_PATH = state_dir / "subagent_progress.jsonl"
    mod.LOCK_PATH = state_dir / ".subagent_progress.lock"

    mod.append_checkpoint(
        subagent_id="READ-TEST",
        step=1,
        status="in_progress",
        files_touched=["x"],
        next_action="y",
    )
    capsys.readouterr()  # clear any prior output
    rc = mod.main(["read", "--subagent-id", "READ-TEST"])
    assert rc == 0
    out = capsys.readouterr().out.strip().split("\n")
    assert len(out) == 1
    rec = json.loads(out[0])
    assert rec["subagent_id"] == "READ-TEST"


def test_cli_read_latest_only_emits_one_record(tmp_path, capsys):
    mod = _load_module()
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True)
    mod.STATE_DIR = state_dir
    mod.JSONL_PATH = state_dir / "subagent_progress.jsonl"
    mod.LOCK_PATH = state_dir / ".subagent_progress.lock"

    for step in range(1, 4):
        mod.append_checkpoint(
            subagent_id="MULTI",
            step=step,
            status="in_progress",
            files_touched=[],
            next_action=f"step-{step}",
        )
    capsys.readouterr()
    rc = mod.main(["read", "--subagent-id", "MULTI", "--latest-only"])
    assert rc == 0
    out = capsys.readouterr().out.strip().split("\n")
    assert len(out) == 1
    rec = json.loads(out[0])
    assert rec["step"] == 3


def test_corrupt_row_does_not_break_read(isolated_state_dir):
    """A garbage line in the JSONL file is silently skipped, not fatal."""
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="OK",
        step=1,
        status="in_progress",
        files_touched=[],
        next_action="x",
    )
    # Append a non-JSON garbage line directly (simulating disk corruption)
    with open(mod.JSONL_PATH, "a") as fh:
        fh.write("THIS_IS_NOT_JSON\n")
    mod.append_checkpoint(
        subagent_id="OK",
        step=2,
        status="in_progress",
        files_touched=[],
        next_action="y",
    )
    rows = mod.read_checkpoints("OK")
    assert len(rows) == 2
    assert [r["step"] for r in rows] == [1, 2]


def test_parse_files_touched_handles_empty_and_whitespace():
    mod = _load_module()
    assert mod._parse_files_touched(None) == []
    assert mod._parse_files_touched("") == []
    assert mod._parse_files_touched("   ") == []
    assert mod._parse_files_touched("a.py, b.py , c.py") == ["a.py", "b.py", "c.py"]
    assert mod._parse_files_touched("a.py,,b.py") == ["a.py", "b.py"]


def test_parse_step_complete_string():
    mod = _load_module()
    assert mod._parse_step("complete") == "complete"
    assert mod._parse_step("7") == 7


def test_parse_step_invalid_raises_systemexit():
    mod = _load_module()
    with pytest.raises(SystemExit):
        mod._parse_step("not-a-number")


def test_validate_record_rejects_non_string_next_action(isolated_state_dir):
    mod = isolated_state_dir
    with pytest.raises(ValueError, match="next_action"):
        mod._validate_record({
            "subagent_id": "ok",
            "step": 1,
            "status": "in_progress",
            "files_touched": [],
            "next_action": 12345,
            "notes": "",
        })


def test_validate_record_rejects_non_string_notes(isolated_state_dir):
    mod = isolated_state_dir
    with pytest.raises(ValueError, match="notes"):
        mod._validate_record({
            "subagent_id": "ok",
            "step": 1,
            "status": "in_progress",
            "files_touched": [],
            "next_action": "x",
            "notes": ["list", "is", "wrong"],
        })


def test_lock_path_is_used(isolated_state_dir):
    """The lock file MUST exist after a write."""
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="LK",
        step=1,
        status="in_progress",
        files_touched=[],
        next_action="x",
    )
    assert mod.LOCK_PATH.exists()


def test_jsonl_is_append_only_under_lock(isolated_state_dir):
    """Successive writes accumulate; do not truncate."""
    mod = isolated_state_dir
    for i in range(5):
        mod.append_checkpoint(
            subagent_id=f"AGENT-{i}",
            step=1,
            status="in_progress",
            files_touched=[],
            next_action="x",
        )
    rows = mod.JSONL_PATH.read_text().strip().split("\n")
    assert len(rows) == 5


def test_record_includes_pid_and_host(isolated_state_dir):
    mod = isolated_state_dir
    rec = mod.append_checkpoint(
        subagent_id="META",
        step=1,
        status="in_progress",
        files_touched=[],
        next_action="x",
    )
    assert rec["pid"] == os.getpid()
    assert isinstance(rec["host"], str) and rec["host"]
