"""Tests for tools/claim_lane_dispatch recovered from .pyc cache + spec."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Load the helper as a module
_HELPER_PATH = Path(__file__).resolve().parents[3] / "tools" / "claim_lane_dispatch.py"
spec = importlib.util.spec_from_file_location("claim_lane_dispatch", _HELPER_PATH)
cld = importlib.util.module_from_spec(spec)
sys.modules["claim_lane_dispatch"] = cld
spec.loader.exec_module(cld)


@pytest.fixture
def claims_path(tmp_path):
    return tmp_path / "claims.md"


def test_first_claim_creates_file(claims_path):
    rc = cld.main([
        "claim",
        "--claims-path", str(claims_path),
        "--lane-id", "test_lane",
        "--platform", "lightning",
        "--instance-job-id", "job_001",
        "--agent", "test_agent",
        "--status", "eval",
    ])
    assert rc == 0
    text = claims_path.read_text()
    assert "test_lane" in text
    assert "job_001" in text
    assert "## Claims (newest first)" in text


def test_conflict_refused_by_default(claims_path):
    cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "j1",
        "--agent", "a1", "--status", "eval",
    ])
    rc = cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "j2",
        "--agent", "a2", "--status", "eval",
    ])
    assert rc == 3


def test_terminal_status_does_not_conflict(claims_path):
    cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "j1",
        "--agent", "a1", "--status", "eval",
    ])
    # terminal closing row for the same job is allowed
    rc = cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "j1",
        "--agent", "a1", "--status", "completed_score=0.42",
    ])
    assert rc == 0
    # and a fresh active claim after the terminal closes is allowed
    rc = cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "j2",
        "--agent", "a2", "--status", "eval",
    ])
    assert rc == 0


def test_force_bypasses_conflict(claims_path):
    cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "j1",
        "--agent", "a1", "--status", "eval",
    ])
    rc = cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "j2",
        "--agent", "a2", "--status", "eval",
        "--force",
    ])
    assert rc == 0


def test_allow_parallel_requires_child_of(claims_path):
    cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "parent_job",
        "--agent", "a1", "--status", "training",
    ])
    with pytest.raises(SystemExit) as exc:
        cld.main([
            "claim", "--claims-path", str(claims_path),
            "--lane-id", "X", "--platform", "L", "--instance-job-id", "child_job",
            "--agent", "a2", "--status", "eval",
            "--allow-parallel",
        ])
    assert "child-of" in str(exc.value)


def test_allow_parallel_with_correct_child_of(claims_path):
    cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "parent_job",
        "--agent", "a1", "--status", "training",
    ])
    rc = cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "child_job",
        "--agent", "a2", "--status", "eval",
        "--allow-parallel",
        "--child-of", "parent_job",
        "--parallel-reason", "follow-on T4 promotion of parent's checkpoint",
    ])
    assert rc == 0


def test_allow_parallel_wrong_child_of_refused(claims_path):
    cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "parent_job",
        "--agent", "a1", "--status", "training",
    ])
    rc = cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "child_job",
        "--agent", "a2", "--status", "eval",
        "--allow-parallel",
        "--child-of", "nonexistent_parent",
        "--parallel-reason", "...",
    ])
    assert rc == 3


def test_dry_run_does_not_write(claims_path):
    rc = cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "j1",
        "--agent", "a1", "--status", "eval",
        "--dry-run",
    ])
    assert rc == 0
    assert not claims_path.exists() or claims_path.read_text() == ""


def test_pipe_in_cell_rejected(claims_path):
    with pytest.raises(SystemExit) as exc:
        cld.main([
            "claim", "--claims-path", str(claims_path),
            "--lane-id", "bad|lane", "--platform", "L", "--instance-job-id", "j",
            "--agent", "a", "--status", "eval",
        ])
    assert "VALIDATION_ERROR" in str(exc.value) or "must not contain" in str(exc.value)


def test_shell_expanded_dollar_zero_notes_rejected(claims_path):
    """A double-quoted $0 cost can become /bin/zsh; refuse that signal loss."""
    with pytest.raises(SystemExit) as exc:
        cld.main([
            "claim", "--claims-path", str(claims_path),
            "--lane-id", "cost_lane", "--platform", "gha", "--instance-job-id", "j",
            "--agent", "a", "--status", "eval",
            "--notes", "cost=/bin/zsh.50 GHA after shell expansion",
        ])
    assert "shell-expanded $0" in str(exc.value)


def test_intentional_shell_path_notes_require_explicit_waiver(claims_path):
    rc = cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "path_lane", "--platform", "local", "--instance-job-id", "j",
        "--agent", "a", "--status", "eval",
        "--notes", "shell=/bin/zsh SHELL_ARGV0_OK:documenting local shell",
    ])
    assert rc == 0


def test_whitespace_in_lane_id_rejected(claims_path):
    with pytest.raises(SystemExit):
        cld.main([
            "claim", "--claims-path", str(claims_path),
            "--lane-id", "lane with space", "--platform", "L", "--instance-job-id", "j",
            "--agent", "a", "--status", "eval",
        ])


def test_terminal_prefixes_constants():
    # Spec from .pyc revealed these prefixes — verify they're all present
    expected = {"completed_", "failed_", "preempted", "cancelled",
                "refused_dispatch", "stale_assumed_dead", "stale_superseded", "stopped_",
                "falsified_", "retired_", "config_retired_",
                "measured_implementation_retired_",
                "stop_attempt_timeout_duplicate_after_primary_negative"}
    actual = set(cld.TERMINAL_PREFIXES)
    assert expected == actual


def test_falsified_and_retired_statuses_close_claims(claims_path):
    terminal_statuses = [
        "falsified_score_1.43_pareto_dominated",
        "retired_config_exact_negative",
        "config_retired_pr107_b050",
        "measured_implementation_retired_cuda_negative",
        "stop_attempt_timeout_duplicate_after_primary_negative",
    ]
    for index, status in enumerate(terminal_statuses):
        lane = f"lane_{index}"
        cld.main([
            "claim", "--claims-path", str(claims_path),
            "--lane-id", lane, "--platform", "lightning", "--instance-job-id", "j1",
            "--agent", "a1", "--status", "eval",
        ])
        rc = cld.main([
            "claim", "--claims-path", str(claims_path),
            "--lane-id", lane, "--platform", "lightning", "--instance-job-id", "j1",
            "--agent", "a1", "--status", status,
        ])
        assert rc == 0
        rc = cld.main([
            "claim", "--claims-path", str(claims_path),
            "--lane-id", lane, "--platform", "lightning", "--instance-job-id", "j2",
            "--agent", "a2", "--status", "eval",
        ])
        assert rc == 0


def test_stale_claim_outside_ttl_requires_terminal_stale_closure(claims_path):
    """A TTL-stale active row must be explicitly closed before same-lane refire."""
    cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "j_old",
        "--agent", "a", "--status", "training",
        "--now-utc", "2026-01-01T00:00:00Z",
    ])
    # 48 hours later, the old claim is stale (TTL default 24h), but it still
    # blocks new custody until a terminal stale_* row is written.
    rc = cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "j_new",
        "--agent", "a", "--status", "eval",
        "--now-utc", "2026-01-03T00:00:00Z",
    ])
    assert rc == 3

    rc = cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "j_new",
        "--agent", "a", "--status", "eval",
        "--now-utc", "2026-01-03T00:00:00Z",
        "--dry-run",
    ])
    assert rc == 3

    rc = cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "j_old",
        "--agent", "a", "--status", "stale_superseded_manual_reconcile",
        "--notes", "operator verified provider job is gone",
        "--now-utc", "2026-01-03T00:00:00Z",
    ])
    assert rc == 0

    rc = cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "j_new",
        "--agent", "a", "--status", "eval",
        "--now-utc", "2026-01-03T00:01:00Z",
    ])
    assert rc == 0


def test_completed_status_cannot_silently_close_stale_claim(claims_path):
    cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "j_old",
        "--agent", "a", "--status", "training",
        "--now-utc", "2026-01-01T00:00:00Z",
    ])
    rc = cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "X", "--platform", "L", "--instance-job-id", "j_old",
        "--agent", "a", "--status", "completed_exact_cuda_adjudicated",
        "--now-utc", "2026-01-03T00:00:00Z",
    ])
    assert rc == 3


def test_newest_first_ordering(claims_path):
    cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "L1", "--platform", "lightning", "--instance-job-id", "j1",
        "--agent", "a", "--status", "completed_first",
    ])
    cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "L2", "--platform", "lightning", "--instance-job-id", "j2",
        "--agent", "a", "--status", "completed_second",
    ])
    text = claims_path.read_text()
    # j2 (newer) should come BEFORE j1 in the file
    assert text.find("j2") < text.find("j1")


def test_summary_reports_active_and_stale_claims(claims_path, capsys):
    cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "fresh", "--platform", "lightning", "--instance-job-id", "j_active",
        "--agent", "a", "--status", "training",
        "--now-utc", "2026-05-08T00:00:00Z",
    ])
    cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "old", "--platform", "lightning", "--instance-job-id", "j_stale",
        "--agent", "a", "--status", "training",
        "--now-utc", "2026-05-06T00:00:00Z",
    ])
    cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "done", "--platform", "lightning", "--instance-job-id", "j_done",
        "--agent", "a", "--status", "training",
        "--now-utc", "2026-05-08T00:00:00Z",
    ])
    cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "done", "--platform", "lightning", "--instance-job-id", "j_done",
        "--agent", "a", "--status", "completed_exact_cuda_adjudicated",
        "--now-utc", "2026-05-08T01:00:00Z",
    ])
    capsys.readouterr()
    rc = cld.main([
        "summary", "--claims-path", str(claims_path),
        "--now-utc", "2026-05-08T12:00:00Z",
        "--ttl-hours", "24",
        "--format", "json",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["active_count"] == 1
    assert payload["stale_nonterminal_count"] == 1
    assert payload["terminal_latest_count"] == 1
    assert payload["active"][0]["lane_id"] == "fresh"
    assert payload["stale_nonterminal"][0]["lane_id"] == "old"
    assert payload["terminal_latest"][0]["lane_id"] == "done"


def test_summary_text_lists_active_without_mutating_claims(claims_path, capsys):
    cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", "L", "--platform", "gha", "--instance-job-id", "j1",
        "--agent", "agent", "--status", "active_dispatching",
        "--now-utc", "2026-05-08T10:00:00Z",
    ])
    before = claims_path.read_text()
    capsys.readouterr()
    rc = cld.main([
        "summary", "--claims-path", str(claims_path),
        "--now-utc", "2026-05-08T11:00:00Z",
    ])
    after = claims_path.read_text()
    assert rc == 0
    assert after == before
    out = capsys.readouterr().out
    assert "CLAIM_SUMMARY active=1" in out
    assert "ACTIVE lane_id=L job=j1 platform=gha status=active_dispatching agent=agent" in out
