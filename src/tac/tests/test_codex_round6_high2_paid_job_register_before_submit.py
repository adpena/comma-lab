"""Tests for codex round 6 HIGH 2 fix: create-pending-row-before-submit.

Catalog #143 — refuses ``Job.run(...)`` Lightning paid-submit callers
that don't first call ``register_pending_job_locked(...)``. Pre-fix,
a corrupt active-jobs file → paid job created but tracker write fails
→ invisible orphan paid job.

Bug class: codex round 6 HIGH 2 (2026-05-09). Memory:
feedback_codex_round6_findings_fix_with_self_protection_landed_20260509.md.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tac.deploy.lightning.active_jobs_state import (
    ACTIVE_STATUS_TOKEN,
    PENDING_STATUS_TOKEN,
    ActiveJobsCorruptError,
    PendingJobNotFoundError,
    cancel_pending_job_locked,
    register_pending_job_locked,
    update_pending_to_active_locked,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# Pending-row API tests
# --------------------------------------------------------------------------


def _tmp_active_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Return (active_jobs_path, lock_path) under the tmp dir."""
    p = tmp_path / "lightning_active_jobs.json"
    lock = tmp_path / "lightning_active_jobs.json.lock"
    return p, lock


def test_register_pending_writes_status_pending(tmp_path: Path):
    p, lock = _tmp_active_paths(tmp_path)
    record = {
        "schema_version": "lightning_active_jobs.v1",
        "lane_id": "lane_test",
        "job_name": "test-job-001",
        "machine": "T4",
    }
    rows = register_pending_job_locked(record, path=p, lock_path=lock)
    assert len(rows) == 1
    assert rows[0]["status"] == PENDING_STATUS_TOKEN
    assert rows[0]["submit_result"] == {"status": PENDING_STATUS_TOKEN}
    # Persisted to disk
    on_disk = json.loads(p.read_text())
    assert on_disk[0]["status"] == PENDING_STATUS_TOKEN


def test_register_pending_idempotent(tmp_path: Path):
    p, lock = _tmp_active_paths(tmp_path)
    record = {"job_name": "twice", "lane_id": "x"}
    register_pending_job_locked(record, path=p, lock_path=lock)
    rows = register_pending_job_locked(record, path=p, lock_path=lock)
    # Same job_name pending → no double-append
    assert len([r for r in rows if r["job_name"] == "twice"]) == 1


def test_register_pending_refuses_when_active_row_already_exists(tmp_path: Path):
    p, lock = _tmp_active_paths(tmp_path)
    # Hand-write an active row first
    p.write_text(json.dumps([
        {"job_name": "existing", "status": ACTIVE_STATUS_TOKEN, "lane_id": "x"}
    ]))
    with pytest.raises(ValueError, match="non-pending row"):
        register_pending_job_locked(
            {"job_name": "existing", "lane_id": "x"}, path=p, lock_path=lock,
        )


def test_register_pending_requires_job_name(tmp_path: Path):
    p, lock = _tmp_active_paths(tmp_path)
    with pytest.raises(ValueError, match="job_name"):
        register_pending_job_locked({"lane_id": "x"}, path=p, lock_path=lock)


def test_update_pending_to_active_promotes_status(tmp_path: Path):
    p, lock = _tmp_active_paths(tmp_path)
    register_pending_job_locked(
        {"job_name": "foo", "lane_id": "x"}, path=p, lock_path=lock,
    )
    rows = update_pending_to_active_locked(
        "foo",
        submit_result={"name": "foo", "status_at_submit": "queued"},
        path=p, lock_path=lock,
    )
    assert rows[0]["status"] == ACTIVE_STATUS_TOKEN
    assert rows[0]["submit_result"] == {"name": "foo", "status_at_submit": "queued"}


def test_update_pending_to_active_raises_when_no_pending(tmp_path: Path):
    p, lock = _tmp_active_paths(tmp_path)
    with pytest.raises(PendingJobNotFoundError, match="no pending row"):
        update_pending_to_active_locked(
            "missing", submit_result={}, path=p, lock_path=lock,
        )


def test_cancel_pending_drops_row(tmp_path: Path):
    p, lock = _tmp_active_paths(tmp_path)
    register_pending_job_locked(
        {"job_name": "cancel-me", "lane_id": "x"}, path=p, lock_path=lock,
    )
    rows = cancel_pending_job_locked(
        "cancel-me", failure_reason="ImportError", path=p, lock_path=lock,
    )
    assert all(r["job_name"] != "cancel-me" for r in rows)


def test_cancel_pending_raises_when_no_pending(tmp_path: Path):
    p, lock = _tmp_active_paths(tmp_path)
    with pytest.raises(PendingJobNotFoundError, match="no pending row"):
        cancel_pending_job_locked("missing", path=p, lock_path=lock)


def test_register_pending_refuses_corrupt_tracker(tmp_path: Path):
    """The headline orphan-prevention test: corrupt tracker → refuse submit."""
    p, lock = _tmp_active_paths(tmp_path)
    # Write corrupt JSON
    p.write_text("{not valid json")
    with pytest.raises(ActiveJobsCorruptError):
        register_pending_job_locked(
            {"job_name": "no-orphan", "lane_id": "x"}, path=p, lock_path=lock,
        )
    # The pending row was NOT inserted (no orphan possible)
    # And the corrupt file was quarantined (not silently overwritten)
    assert not p.exists() or "no-orphan" not in p.read_text()


# --------------------------------------------------------------------------
# Preflight check #143 STRICT
# --------------------------------------------------------------------------


def test_preflight_check_143_passes_with_zero_violations():
    """Live count expected: 0 after both Lightning dispatchers refactored."""
    from tac.preflight import check_paid_job_register_before_submit

    violations = check_paid_job_register_before_submit(verbose=False, strict=False)
    assert violations == [], (
        f"Catalog #143 preflight should be at 0; got {len(violations)}:\n  "
        + "\n  ".join(violations[:5])
    )


def test_preflight_check_143_fires_on_simulated_violation(tmp_path: Path):
    """Simulate a Lightning dispatcher that submits without registering."""
    from tac.preflight import check_paid_job_register_before_submit

    fake_repo = tmp_path
    (fake_repo / "experiments").mkdir()
    bad = fake_repo / "experiments" / "fake_lightning_dispatch.py"
    bad.write_text(
        "from lightning_sdk import Job\n"
        "def submit():\n"
        "    job = Job.run(name='x', machine='T4')\n"
        "    return job\n"
    )
    (fake_repo / "tools").mkdir()
    (fake_repo / "scripts").mkdir()
    (fake_repo / "src" / "tac").mkdir(parents=True)

    violations = check_paid_job_register_before_submit(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert len(violations) >= 1
    assert "fake_lightning_dispatch.py" in str(violations)


def test_preflight_check_143_accepts_pending_register_first(tmp_path: Path):
    """Pre-submit pending-register makes the file clean."""
    from tac.preflight import check_paid_job_register_before_submit

    fake_repo = tmp_path
    (fake_repo / "experiments").mkdir()
    good = fake_repo / "experiments" / "good_lightning_dispatch.py"
    good.write_text(
        "from lightning_sdk import Job\n"
        "from tac.deploy.lightning.active_jobs_state import register_pending_job_locked\n"
        "def submit():\n"
        "    register_pending_job_locked({'job_name': 'x'})\n"
        "    job = Job.run(name='x', machine='T4')\n"
        "    return job\n"
    )
    (fake_repo / "tools").mkdir()
    (fake_repo / "scripts").mkdir()
    (fake_repo / "src" / "tac").mkdir(parents=True)

    violations = check_paid_job_register_before_submit(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert violations == []


def test_preflight_check_143_accepts_same_line_waiver(tmp_path: Path):
    """`# JOB_RUN_BEFORE_REGISTER_OK:<reason>` waiver on the Job.run line."""
    from tac.preflight import check_paid_job_register_before_submit

    fake_repo = tmp_path
    (fake_repo / "experiments").mkdir()
    helper = fake_repo / "experiments" / "submit_lightning_helper.py"
    helper.write_text(
        "from lightning_sdk import Job\n"
        "def submit_helper():\n"
        "    return Job.run(name='x')  # JOB_RUN_BEFORE_REGISTER_OK:helper-caller-owns-pending\n"
    )
    (fake_repo / "tools").mkdir()
    (fake_repo / "scripts").mkdir()
    (fake_repo / "src" / "tac").mkdir(parents=True)

    violations = check_paid_job_register_before_submit(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert violations == []


def test_preflight_check_143_strict_mode_raises():
    from tac.preflight import (
        PreflightError,
        check_paid_job_register_before_submit,
    )

    with tempfile.TemporaryDirectory() as td:
        fake_repo = Path(td)
        (fake_repo / "experiments").mkdir()
        bad = fake_repo / "experiments" / "evil_lightning_dispatch.py"
        bad.write_text(
            "from lightning_sdk import Job\n"
            "def submit():\n"
            "    Job.run(name='x')\n"
        )
        (fake_repo / "tools").mkdir()
        (fake_repo / "scripts").mkdir()
        (fake_repo / "src" / "tac").mkdir(parents=True)

        with pytest.raises(PreflightError):
            check_paid_job_register_before_submit(
                repo_root=fake_repo, verbose=False, strict=True,
            )


def test_preflight_check_143_ignores_non_lightning_files(tmp_path: Path):
    """A non-Lightning file with the literal `Job.run` substring should NOT fire."""
    from tac.preflight import check_paid_job_register_before_submit

    fake_repo = tmp_path
    (fake_repo / "experiments").mkdir()
    # File name does NOT contain "lightning"
    other = fake_repo / "experiments" / "modal_dispatch.py"
    other.write_text(
        "# Job.run here is in a comment, not Lightning\n"
        "def f():\n"
        "    pass\n"
    )
    (fake_repo / "tools").mkdir()
    (fake_repo / "scripts").mkdir()
    (fake_repo / "src" / "tac").mkdir(parents=True)

    violations = check_paid_job_register_before_submit(
        repo_root=fake_repo, verbose=False, strict=False,
    )
    assert violations == []
