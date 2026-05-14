# SPDX-License-Identifier: MIT
"""Tests for the Codex finding #3 predecessor-resume query paths.

Empirical anchor 2026-05-14: when a subagent crashes mid-session (Anthropic
API "Internal server error") the parent typically respawns a SUCCESSOR with
a NEW subagent_id. The original ``read --subagent-id`` flow required an
exact match, so the documented startup query returned no rows even when the
predecessor had checkpointed — defeating the core crash-resume scenario.

This test suite exercises the new query paths:

* ``read --parent-id-or-session <id>``
* ``read --lane-id <lane>``
* ``read --latest-incomplete``

Plus the structured ``lane_id`` schema field on the write side.

Memory: feedback_codex_3_findings_fix_landed_20260514.md.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBAGENT_CHECKPOINT_PATH = REPO_ROOT / "tools" / "subagent_checkpoint.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "subagent_checkpoint_resume_lookup", SUBAGENT_CHECKPOINT_PATH
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


# ─── Structured lane_id field on write ──────────────────────────────────


def test_append_with_lane_id_field(isolated_state_dir):
    mod = isolated_state_dir
    rec = mod.append_checkpoint(
        subagent_id="WAVE-FOO",
        step=1,
        status="in_progress",
        files_touched=["src/tac/foo.py"],
        next_action="continue",
        lane_id="lane_codex_3_findings_fix_20260514",
    )
    assert rec["lane_id"] == "lane_codex_3_findings_fix_20260514"


def test_append_without_lane_id_records_none(isolated_state_dir):
    mod = isolated_state_dir
    rec = mod.append_checkpoint(
        subagent_id="WAVE-NO-LANE",
        step=1,
        status="in_progress",
        files_touched=[],
        next_action="x",
    )
    assert rec["lane_id"] is None


# ─── read_checkpoints_by_parent ─────────────────────────────────────────


def test_read_by_parent_finds_predecessor(isolated_state_dir):
    """Successor with different id can resume via parent_id_or_session."""
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="PREDECESSOR-CRASHED",
        step=3,
        status="in_progress",
        files_touched=["a.py"],
        next_action="resume after rmtree guard wired",
        parent_id_or_session="parent-session-XYZ",
    )
    rows = mod.read_checkpoints_by_parent("parent-session-XYZ")
    assert len(rows) == 1
    assert rows[0]["subagent_id"] == "PREDECESSOR-CRASHED"
    assert rows[0]["step"] == 3


def test_read_by_parent_orders_by_write_order(isolated_state_dir):
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="P1", step=1, status="in_progress",
        files_touched=[], next_action="a",
        parent_id_or_session="shared-parent",
    )
    mod.append_checkpoint(
        subagent_id="P2", step=2, status="in_progress",
        files_touched=[], next_action="b",
        parent_id_or_session="shared-parent",
    )
    mod.append_checkpoint(
        subagent_id="P3", step=1, status="in_progress",
        files_touched=[], next_action="c",
        parent_id_or_session="shared-parent",
    )
    rows = mod.read_checkpoints_by_parent("shared-parent")
    assert [r["subagent_id"] for r in rows] == ["P1", "P2", "P3"]


def test_read_by_parent_empty_for_unknown(isolated_state_dir):
    mod = isolated_state_dir
    rows = mod.read_checkpoints_by_parent("never-existed")
    assert rows == []


def test_read_by_parent_empty_string_raises(isolated_state_dir):
    mod = isolated_state_dir
    with pytest.raises(ValueError):
        mod.read_checkpoints_by_parent("")


def test_read_by_parent_filters_others(isolated_state_dir):
    """A checkpoint with a different parent must NOT match."""
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="X", step=1, status="in_progress",
        files_touched=[], next_action="x",
        parent_id_or_session="parent-A",
    )
    mod.append_checkpoint(
        subagent_id="Y", step=1, status="in_progress",
        files_touched=[], next_action="y",
        parent_id_or_session="parent-B",
    )
    rows = mod.read_checkpoints_by_parent("parent-A")
    assert len(rows) == 1
    assert rows[0]["subagent_id"] == "X"


# ─── read_checkpoints_by_lane ───────────────────────────────────────────


def test_read_by_lane_structured_field(isolated_state_dir):
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="W1", step=1, status="in_progress",
        files_touched=[], next_action="x",
        lane_id="lane_foo_20260514",
    )
    mod.append_checkpoint(
        subagent_id="W2", step=1, status="in_progress",
        files_touched=[], next_action="y",
        lane_id="lane_bar_20260514",
    )
    rows = mod.read_checkpoints_by_lane("lane_foo_20260514")
    assert len(rows) == 1
    assert rows[0]["subagent_id"] == "W1"


def test_read_by_lane_notes_substring_fallback(isolated_state_dir):
    """A checkpoint missing structured lane_id but containing it in notes
    must still match (backward compatibility)."""
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="OLD-STYLE", step=1, status="in_progress",
        files_touched=[], next_action="x",
        notes="lane lane_legacy_20260514 in-flight",
    )
    rows = mod.read_checkpoints_by_lane("lane_legacy_20260514")
    assert len(rows) == 1
    assert rows[0]["subagent_id"] == "OLD-STYLE"


def test_read_by_lane_no_double_count(isolated_state_dir):
    """A record with BOTH structured lane_id AND notes mention must appear once."""
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="BOTH", step=1, status="in_progress",
        files_touched=[], next_action="x",
        lane_id="lane_both_20260514",
        notes="working lane_both_20260514 also in notes",
    )
    rows = mod.read_checkpoints_by_lane("lane_both_20260514")
    assert len(rows) == 1


def test_read_by_lane_empty_string_raises(isolated_state_dir):
    mod = isolated_state_dir
    with pytest.raises(ValueError):
        mod.read_checkpoints_by_lane("")


# ─── latest_incomplete_checkpoint ───────────────────────────────────────


def test_latest_incomplete_returns_in_progress(isolated_state_dir):
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="OLD-DONE", step="complete", status="complete",
        files_touched=[], next_action="done",
    )
    mod.append_checkpoint(
        subagent_id="LATER-INPROG", step=2, status="in_progress",
        files_touched=[], next_action="next",
    )
    rec = mod.latest_incomplete_checkpoint()
    assert rec is not None
    assert rec["subagent_id"] == "LATER-INPROG"


def test_latest_incomplete_skips_complete_records(isolated_state_dir):
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="INPROG-EARLY", step=1, status="in_progress",
        files_touched=[], next_action="x",
    )
    mod.append_checkpoint(
        subagent_id="DONE-LATER", step="complete", status="complete",
        files_touched=[], next_action="done",
    )
    rec = mod.latest_incomplete_checkpoint()
    # Most-recent NON-complete is INPROG-EARLY because DONE-LATER is excluded.
    assert rec is not None
    assert rec["subagent_id"] == "INPROG-EARLY"


def test_latest_incomplete_returns_blocked(isolated_state_dir):
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="STUCK", step=1, status="blocked",
        files_touched=[], next_action="awaiting operator",
    )
    rec = mod.latest_incomplete_checkpoint()
    assert rec is not None
    assert rec["subagent_id"] == "STUCK"
    assert rec["status"] == "blocked"


def test_latest_incomplete_returns_none_when_empty(isolated_state_dir):
    mod = isolated_state_dir
    rec = mod.latest_incomplete_checkpoint()
    assert rec is None


def test_latest_incomplete_for_parent_filter(isolated_state_dir):
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="P-A-INPROG", step=1, status="in_progress",
        files_touched=[], next_action="x",
        parent_id_or_session="parent-A",
    )
    mod.append_checkpoint(
        subagent_id="P-B-COMPLETE", step="complete", status="complete",
        files_touched=[], next_action="done",
        parent_id_or_session="parent-B",
    )
    rec = mod.latest_incomplete_for_parent("parent-A")
    assert rec is not None
    assert rec["subagent_id"] == "P-A-INPROG"
    rec_b = mod.latest_incomplete_for_parent("parent-B")
    assert rec_b is None


def test_latest_incomplete_for_lane_filter(isolated_state_dir):
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="L-FOO-INPROG", step=1, status="in_progress",
        files_touched=[], next_action="x",
        lane_id="lane_foo",
    )
    rec = mod.latest_incomplete_for_lane("lane_foo")
    assert rec is not None
    assert rec["subagent_id"] == "L-FOO-INPROG"


# ─── CLI: backward compat with --subagent-id ────────────────────────────


def test_cli_read_by_subagent_id_still_works(isolated_state_dir, capsys):
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="OLD-CALLER", step=1, status="in_progress",
        files_touched=[], next_action="x",
    )
    rc = mod.main(["read", "--subagent-id", "OLD-CALLER"])
    assert rc == 0
    captured = capsys.readouterr()
    rec = json.loads(captured.out.strip())
    assert rec["subagent_id"] == "OLD-CALLER"


# ─── CLI: --parent-id-or-session ───────────────────────────────────────


def test_cli_read_by_parent_id_or_session(isolated_state_dir, capsys):
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="PRE-CRASHED-X", step=2, status="in_progress",
        files_touched=["x.py"], next_action="resume",
        parent_id_or_session="parent-resume-test",
    )
    rc = mod.main([
        "read",
        "--parent-id-or-session", "parent-resume-test",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    rec = json.loads(captured.out.strip())
    assert rec["subagent_id"] == "PRE-CRASHED-X"


# ─── CLI: --latest-incomplete ──────────────────────────────────────────


def test_cli_read_latest_incomplete(isolated_state_dir, capsys):
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="STILL-WORKING", step=4, status="in_progress",
        files_touched=[], next_action="z",
    )
    rc = mod.main(["read", "--latest-incomplete"])
    assert rc == 0
    captured = capsys.readouterr()
    rec = json.loads(captured.out.strip())
    assert rec["subagent_id"] == "STILL-WORKING"


# ─── CLI: --lane-id ───────────────────────────────────────────────────


def test_cli_read_by_lane_id_structured_field(isolated_state_dir, capsys):
    mod = isolated_state_dir
    mod.append_checkpoint(
        subagent_id="LANE-SUBJECT", step=1, status="in_progress",
        files_touched=[], next_action="x",
        lane_id="lane_xyz",
    )
    rc = mod.main(["read", "--lane-id", "lane_xyz"])
    assert rc == 0
    captured = capsys.readouterr()
    rec = json.loads(captured.out.strip())
    assert rec["subagent_id"] == "LANE-SUBJECT"


# ─── CLI: argument validation ─────────────────────────────────────────


def test_cli_read_no_query_mode_errors(isolated_state_dir):
    mod = isolated_state_dir
    with pytest.raises(SystemExit):
        mod.main(["read"])


def test_cli_read_multiple_query_modes_errors(isolated_state_dir):
    mod = isolated_state_dir
    with pytest.raises(SystemExit):
        mod.main([
            "read",
            "--subagent-id", "A",
            "--parent-id-or-session", "B",
        ])


def test_cli_read_returns_rc2_when_no_records(isolated_state_dir, capsys):
    mod = isolated_state_dir
    rc = mod.main(["read", "--latest-incomplete"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no records" in err


# ─── CLI: --lane-id on write ──────────────────────────────────────────


def test_cli_write_with_lane_id(isolated_state_dir, capsys):
    mod = isolated_state_dir
    rc = mod.main([
        "--subagent-id", "WRITE-W-LANE",
        "--step", "1",
        "--status", "in_progress",
        "--lane-id", "lane_codex_3_findings_fix_20260514",
        "--next-action", "x",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    rec = json.loads(captured.out.strip())
    assert rec["lane_id"] == "lane_codex_3_findings_fix_20260514"
    # Verify via the new query path that the record is locatable by lane.
    rows = mod.read_checkpoints_by_lane(
        "lane_codex_3_findings_fix_20260514"
    )
    assert len(rows) == 1
    assert rows[0]["subagent_id"] == "WRITE-W-LANE"


# ─── End-to-end crash-resume scenario ──────────────────────────────────


def test_successor_with_different_id_recovers_predecessor_state(
    isolated_state_dir, capsys
):
    """Documented crash-resume scenario per CLAUDE.md non-negotiable:

    1. PREDECESSOR subagent writes a checkpoint with parent_id_or_session
       and lane_id.
    2. PREDECESSOR crashes mid-session (API error).
    3. SUCCESSOR is respawned with a DIFFERENT subagent_id but the SAME
       parent_id_or_session.
    4. SUCCESSOR queries via --parent-id-or-session and recovers the
       predecessor's latest incomplete checkpoint.
    """
    mod = isolated_state_dir
    # Step 1+2: predecessor checkpoint then "crash" (just stop writing).
    mod.append_checkpoint(
        subagent_id="PREDECESSOR-uuid-abcde",
        step=5,
        status="in_progress",
        files_touched=["tools/build_pr101_nonlocal_sweep_packets.py"],
        next_action="add Catalog #207 STRICT preflight gate",
        parent_id_or_session="codex-3-findings-fix-session",
        lane_id="lane_codex_3_findings_fix_20260514",
    )
    # Step 3+4: successor's lookup.
    rc = mod.main([
        "read",
        "--parent-id-or-session", "codex-3-findings-fix-session",
        "--latest-only",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    rec = json.loads(out.strip())
    assert rec["subagent_id"] == "PREDECESSOR-uuid-abcde"
    assert rec["step"] == 5
    assert rec["status"] == "in_progress"
    assert rec["next_action"] == "add Catalog #207 STRICT preflight gate"


# ─── End-to-end CLI subprocess invocation ──────────────────────────────


def test_cli_subprocess_read_latest_incomplete(tmp_path, monkeypatch):
    """Smoke-test the CLI as a subprocess (matches operator usage)."""
    # Build an isolated state dir + JSONL with one in-progress record.
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True)
    jsonl_path = state_dir / "subagent_progress.jsonl"
    rec = {
        "subagent_id": "CLI-SUBPROC-TEST",
        "parent_id_or_session": None,
        "lane_id": None,
        "step": 1,
        "status": "in_progress",
        "files_touched": [],
        "next_action": "x",
        "notes": "",
        "written_at_utc": "2026-05-14T10:30:00+00:00",
        "pid": 1,
        "host": "test",
    }
    jsonl_path.write_text(json.dumps(rec) + "\n")

    # Run a child python interpreter against the real tool, but with
    # monkey-patched STATE_DIR via env var. Simpler: import the module and
    # patch it in-process via a small wrapper script.
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(f"""
import sys
import importlib.util
spec = importlib.util.spec_from_file_location(
    'subagent_checkpoint', {str(SUBAGENT_CHECKPOINT_PATH)!r}
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from pathlib import Path
mod.STATE_DIR = Path({str(state_dir)!r})
mod.JSONL_PATH = Path({str(jsonl_path)!r})
mod.LOCK_PATH = Path({str(state_dir / '.subagent_progress.lock')!r})
sys.exit(mod.main(['read', '--latest-incomplete']))
""")
    res = subprocess.run(
        [sys.executable, str(wrapper)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert res.returncode == 0, f"stderr: {res.stderr}"
    out_rec = json.loads(res.stdout.strip())
    assert out_rec["subagent_id"] == "CLI-SUBPROC-TEST"
