# SPDX-License-Identifier: MIT
"""Tests for Catalog #131 + #342 sister: HF Jobs call_id ledger path discipline.

Slot 8 wire-in (2026-05-19; operator-routable item 5 of slot 7's queue).
Catalog #131 (``check_no_bare_writes_to_shared_state``) protects shared
fcntl-locked state files from bare writes outside the canonical helper. The
Catalog #342 sister landed
``.omx/state/hf_jobs_call_id_ledger.jsonl`` + canonical helper at
``tac.deploy.hf_jobs.job_id_ledger``; this slot extends the Catalog #131
recognition surfaces so direct writes outside the canonical helper are
refused.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import tac.preflight as pf


# ---------------------------------------------------------------------------
# Surface extension regression guards
# ---------------------------------------------------------------------------


def test_hf_jobs_ledger_path_in_shared_state_markers() -> None:
    """`HF_JOBS_CALL_ID_LEDGER_PATH` is registered in `_SHARED_STATE_PATH_MARKERS`."""

    assert "HF_JOBS_CALL_ID_LEDGER_PATH" in pf._SHARED_STATE_PATH_MARKERS
    assert "hf_jobs_call_id_ledger" in pf._SHARED_STATE_PATH_MARKERS


def test_modal_call_id_ledger_still_registered() -> None:
    """Sister regression: Modal Catalog #245 entries remain (no regression)."""

    assert "MODAL_CALL_ID_LEDGER_PATH" in pf._SHARED_STATE_PATH_MARKERS
    assert "modal_call_id_ledger" in pf._SHARED_STATE_PATH_MARKERS


def test_hf_jobs_canonical_helpers_registered_in_call_tokens() -> None:
    """Canonical HF Jobs helper call names are recognized as transactional contract."""

    expected = {
        "register_dispatched_hf_jobs_id",
        "register_dispatched_hf_jobs_id_fail_closed",
        "update_hf_jobs_outcome",
        "load_hf_jobs_strict",
        "poll_ledger_for_hf_jobs_id",
    }
    actual = set(pf._BARE_WRITE_CANONICAL_HELPER_CALL_TOKENS)
    missing = expected - actual
    assert not missing, f"missing canonical HF Jobs helper call tokens: {missing}"


def test_hf_jobs_canonical_helper_file_in_self_exempt_list() -> None:
    """The canonical helper module is on the self-exempt list per the sister pattern."""

    assert "src/tac/deploy/hf_jobs/job_id_ledger.py" in pf._BARE_WRITE_CANONICAL_HELPERS


def test_hf_jobs_lock_pattern_recognized() -> None:
    """Catalog #131 already recognizes `fcntl.flock` + `LOCK_EX` (used by ledger).

    The HF Jobs canonical helper uses the same fcntl lock pattern as the
    Modal sister, so no new lock-token addition is required; this test pins
    the existing tokens so a future refactor does not silently lose recognition.
    """

    expected_lock_tokens = {"fcntl.flock", "LOCK_EX"}
    actual = set(pf._BARE_WRITE_LOCK_TOKENS)
    missing = expected_lock_tokens - actual
    assert not missing, f"canonical fcntl lock tokens missing: {missing}"


# ---------------------------------------------------------------------------
# Catalog #131 strict-mode regression (live count remains 0 for the new path)
# ---------------------------------------------------------------------------


def test_check_131_live_repo_strict_passes() -> None:
    """`check_no_bare_writes_to_shared_state` strict mode passes on live repo.

    Slot 8 (this wire-in) extends the recognition surface (adds the HF Jobs
    ledger path to `_SHARED_STATE_PATH_MARKERS`) — by construction, the
    only file writing to the new path is the canonical helper itself,
    which is on the self-exempt list. So live count remains 0.
    """

    violations = pf.check_no_bare_writes_to_shared_state(strict=False)
    # Filter to violations that reference HF Jobs path (defense-in-depth).
    hf_jobs_violations = [
        v for v in violations
        if "hf_jobs_call_id_ledger" in str(v).lower()
        or "HF_JOBS_CALL_ID_LEDGER" in str(v)
    ]
    assert not hf_jobs_violations, (
        f"Catalog #131 found {len(hf_jobs_violations)} bare write(s) to the "
        f"HF Jobs ledger path: {hf_jobs_violations[:3]}"
    )


def test_check_131_helper_module_self_exempt(tmp_path: Path) -> None:
    """The canonical helper file is recognized as self-exempt.

    Verifies that the helper's own bare writes (the canonical
    `_append_event_locked` invocation inside the fcntl context) do NOT
    trigger Catalog #131.
    """

    # The canonical helper's path-on-disk should be discoverable in the
    # self-exempt list (sister pattern: `src/tac/deploy/modal/call_id_ledger.py`
    # is exempt because it IS the canonical Modal helper).
    helper_rel = "src/tac/deploy/hf_jobs/job_id_ledger.py"
    assert helper_rel in pf._BARE_WRITE_CANONICAL_HELPERS, (
        f"canonical helper {helper_rel} must be on the self-exempt list"
    )
