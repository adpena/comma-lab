"""Tests for the new `mark_pending_failed_unknown_billing_locked` helper.

Bug class: codex round 7+8 HIGH 1 (2026-05-09). The previous Lightning
dispatcher cancelled the pending row on ANY exception escaping
`submit_lightning_job` — including exceptions across the `Job.run` boundary
where billing may have started. This new helper preserves the row and
re-tags it so the harvester / operator can manually reconcile.

Memory: feedback_codex_round78_findings_fix_with_self_protection_landed_20260509.md
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.deploy.lightning.active_jobs_state import (
    ACTIVE_STATUS_TOKEN,
    FAILED_UNKNOWN_BILLING_STATUS_TOKEN,
    PENDING_STATUS_TOKEN,
    PendingJobNotFoundError,
    cancel_pending_job_locked,
    load_active_jobs,
    mark_pending_failed_unknown_billing_locked,
    register_pending_job_locked,
    update_pending_to_active_locked,
)


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    p = tmp_path / "lightning_active_jobs.json"
    lock = tmp_path / "lightning_active_jobs.json.lock"
    return p, lock


def test_mark_failed_unknown_billing_preserves_row_with_new_status(tmp_path: Path) -> None:
    p, lock = _setup(tmp_path)
    register_pending_job_locked(
        {"job_name": "job-x", "lane_id": "lane_test"},
        path=p, lock_path=lock,
    )
    rows = mark_pending_failed_unknown_billing_locked(
        "job-x",
        failure_reason="TimeoutError: SDK timeout mid-Job.run()",
        path=p, lock_path=lock,
    )
    assert len(rows) == 1
    assert rows[0]["job_name"] == "job-x"
    assert rows[0]["status"] == FAILED_UNKNOWN_BILLING_STATUS_TOKEN
    assert "TimeoutError" in rows[0]["failure_reason"]
    assert "submit_status_unknown_at_utc" in rows[0]


def test_mark_failed_unknown_billing_with_partial_result_persists_it(tmp_path: Path) -> None:
    p, lock = _setup(tmp_path)
    register_pending_job_locked(
        {"job_name": "job-y"}, path=p, lock_path=lock,
    )
    partial = {"name": "job-y-was-created", "status_at_create": "running"}
    rows = mark_pending_failed_unknown_billing_locked(
        "job-y",
        failure_reason="property access failed post-create",
        submit_partial_result=partial,
        path=p, lock_path=lock,
    )
    assert rows[0]["submit_partial_result"] == partial


def test_mark_failed_unknown_billing_extra_fields_merged(tmp_path: Path) -> None:
    p, lock = _setup(tmp_path)
    register_pending_job_locked({"job_name": "job-z"}, path=p, lock_path=lock)
    rows = mark_pending_failed_unknown_billing_locked(
        "job-z",
        failure_reason="x",
        extra_fields={"machine": "t4", "studio": "s"},
        path=p, lock_path=lock,
    )
    assert rows[0]["machine"] == "t4"
    assert rows[0]["studio"] == "s"


def test_mark_failed_unknown_billing_raises_when_no_pending(tmp_path: Path) -> None:
    p, lock = _setup(tmp_path)
    with pytest.raises(PendingJobNotFoundError):
        mark_pending_failed_unknown_billing_locked(
            "no-such-job",
            failure_reason="x",
            path=p, lock_path=lock,
        )


def test_mark_failed_unknown_billing_skips_already_active(tmp_path: Path) -> None:
    """If the row was promoted to active before mark, the mark is a no-op (raises)."""
    p, lock = _setup(tmp_path)
    register_pending_job_locked({"job_name": "job-w"}, path=p, lock_path=lock)
    update_pending_to_active_locked(
        "job-w",
        submit_result={"name": "job-w", "status": "running"},
        path=p, lock_path=lock,
    )
    # Now there's an active row but no pending; the mark refuses.
    with pytest.raises(PendingJobNotFoundError):
        mark_pending_failed_unknown_billing_locked(
            "job-w", failure_reason="x", path=p, lock_path=lock,
        )


def test_mark_failed_unknown_billing_requires_failure_reason(tmp_path: Path) -> None:
    p, lock = _setup(tmp_path)
    register_pending_job_locked({"job_name": "job-q"}, path=p, lock_path=lock)
    with pytest.raises(ValueError):
        mark_pending_failed_unknown_billing_locked(
            "job-q", failure_reason="", path=p, lock_path=lock,
        )


def test_failed_unknown_billing_status_is_distinct_from_active_and_pending() -> None:
    assert FAILED_UNKNOWN_BILLING_STATUS_TOKEN != ACTIVE_STATUS_TOKEN
    assert FAILED_UNKNOWN_BILLING_STATUS_TOKEN != PENDING_STATUS_TOKEN
    # The token should be visibly meaningful for harvester filters.
    assert "failed_unknown_billing" in FAILED_UNKNOWN_BILLING_STATUS_TOKEN


def test_cancel_then_mark_recovery_pattern_does_not_corrupt(tmp_path: Path) -> None:
    """Operator-recovery flow: cancel-after-mark should refuse (the row is
    no longer pending)."""
    p, lock = _setup(tmp_path)
    register_pending_job_locked({"job_name": "job-r"}, path=p, lock_path=lock)
    mark_pending_failed_unknown_billing_locked(
        "job-r", failure_reason="x", path=p, lock_path=lock,
    )
    with pytest.raises(PendingJobNotFoundError):
        cancel_pending_job_locked("job-r", path=p, lock_path=lock)


def test_mark_failed_unknown_billing_atomic_persistence(tmp_path: Path) -> None:
    """The marked row should be durably persisted (load_active_jobs sees it)."""
    p, lock = _setup(tmp_path)
    register_pending_job_locked({"job_name": "job-d", "lane_id": "lane_x"}, path=p, lock_path=lock)
    mark_pending_failed_unknown_billing_locked(
        "job-d", failure_reason="x", path=p, lock_path=lock,
    )
    rows = load_active_jobs(p)
    assert len(rows) == 1
    assert rows[0]["status"] == FAILED_UNKNOWN_BILLING_STATUS_TOKEN
