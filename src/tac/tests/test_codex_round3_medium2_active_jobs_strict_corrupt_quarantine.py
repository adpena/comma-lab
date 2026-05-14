# SPDX-License-Identifier: MIT
"""Tests for codex round-3 MEDIUM 2 fix — active_jobs corrupt-state strict load + quarantine.

Bug class (codex round-3 MEDIUM 2, 2026-05-09): the previous
``load_active_jobs`` returned ``[]`` on corrupt JSON or non-list state.
``update_active_jobs_locked`` then wrote the empty snapshot back,
silently overwriting the malformed file with a fresh one — DROPPING
all active and terminal rows. Harvesters that re-read after the
dispatcher write would see ``[]`` and skip every still-running job.

The fix:

  - new ``load_active_jobs_strict`` raises ``ActiveJobsCorruptError`` on
    corrupt / non-list state
  - ``update_active_jobs_locked`` (and all sister mutators) use the
    strict path inside the lock and quarantine the bad file (rename to
    ``.corrupt.<utc>``) before refusing dispatch
  - the lenient ``load_active_jobs`` survives for read-only callers
    (harvesters that already fail-closed on empty state)

Memory: feedback_codex_round3_findings_fix_landed_20260509.md.
"""
from __future__ import annotations

import json

import pytest

from tac.deploy.lightning import active_jobs_state as ajs
from tac.deploy.lightning.active_jobs_state import ActiveJobsCorruptError


# ── Mutating-path strict semantics ───────────────────────────────────────


def test_update_locked_raises_on_corrupt_json_for_mutating_path(
    tmp_path, monkeypatch
):
    """Mutating writes must REFUSE on corrupt JSON, never overwrite.

    Pre-fix bug: corrupt file would be silently treated as ``[]``,
    overwritten with the empty snapshot, and all active/terminal rows
    silently dropped.
    """
    state_path = tmp_path / "active.json"
    lock_path = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", state_path)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock_path)
    # Seed a corrupt file.
    state_path.write_text("{not valid json at all")

    def _append(rows):
        rows.append({"job_name": "j_new"})
        return rows

    with pytest.raises(ActiveJobsCorruptError, match="quarantined"):
        ajs.update_active_jobs_locked(_append)


def test_update_locked_raises_on_non_list_root_for_mutating_path(
    tmp_path, monkeypatch
):
    """Non-list root (e.g. {} or "string") must REFUSE mutation.

    Same META as corrupt JSON — silently treating it as empty would
    drop the operator's intent. Better to refuse and let the human
    investigate.
    """
    state_path = tmp_path / "active.json"
    lock_path = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", state_path)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock_path)
    # Valid JSON but wrong shape (dict, not list).
    state_path.write_text(json.dumps({"some_key": "some_value"}))

    def _append(rows):
        rows.append({"job_name": "j_new"})
        return rows

    with pytest.raises(ActiveJobsCorruptError):
        ajs.update_active_jobs_locked(_append)


def test_corrupt_state_is_quarantined_not_overwritten(tmp_path, monkeypatch):
    """The corrupt file must be moved to ``<path>.corrupt.<utc>``,
    NOT overwritten with the empty snapshot.

    This preserves forensic evidence so the operator can investigate
    what the file held before the corruption was detected.
    """
    state_path = tmp_path / "active.json"
    lock_path = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", state_path)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock_path)
    bad_payload = "{not valid json"
    state_path.write_text(bad_payload)

    def _noop(rows):
        return rows

    with pytest.raises(ActiveJobsCorruptError):
        ajs.update_active_jobs_locked(_noop)

    # The canonical path no longer holds the corrupt file.
    assert not state_path.exists(), (
        "MEDIUM 2: canonical path must be quarantined (renamed away), "
        "NOT silently overwritten."
    )
    # A quarantine sibling must exist with the original payload preserved.
    quarantine_files = list(tmp_path.glob("active.json.corrupt.*"))
    assert len(quarantine_files) == 1, (
        f"MEDIUM 2: expected exactly one quarantine sibling, "
        f"found {len(quarantine_files)}: {quarantine_files}"
    )
    assert quarantine_files[0].read_text() == bad_payload, (
        "MEDIUM 2: quarantine file must preserve the original "
        "(corrupt) bytes for forensic inspection."
    )


def test_dispatch_refuses_after_corrupt_quarantine(tmp_path, monkeypatch):
    """After a corrupt-state quarantine, the next register_job must NOT
    re-corrupt the canonical path; it should start with an empty list
    cleanly because the corrupt file was renamed away.

    This pins the recovery contract: corrupt → quarantine → next call
    starts fresh.
    """
    state_path = tmp_path / "active.json"
    lock_path = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", state_path)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock_path)
    state_path.write_text("garbage")
    with pytest.raises(ActiveJobsCorruptError):
        ajs.register_job({"job_name": "first_call_fails"})
    # Canonical path is gone (quarantined).
    assert not state_path.exists()
    # Next call starts fresh and succeeds.
    rows = ajs.register_job({"job_name": "second_call_clean"})
    assert rows == [{"job_name": "second_call_clean"}]
    on_disk = json.loads(state_path.read_text())
    assert on_disk == rows


def test_load_active_jobs_for_readonly_caller_still_returns_empty_on_corrupt(
    tmp_path, monkeypatch
):
    """Backward-compat: read-only callers (harvesters) get [] not raise.

    Harvesters already fail-closed when no active jobs are seen; the
    lenient path lets them keep their existing semantics. The strict
    path is only invoked by mutating callers inside the lock.
    """
    state_path = tmp_path / "active.json"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", state_path)
    state_path.write_text("not valid json")
    # Lenient API returns [] (does NOT raise).
    rows = ajs.load_active_jobs(state_path)
    assert rows == []


# ── Strict load explicit semantics ───────────────────────────────────────


def test_load_active_jobs_strict_returns_empty_for_missing(tmp_path):
    """Strict load treats missing file as empty (not corruption).

    The empty-tracker bootstrap case is normal; only EXISTING-but-
    invalid files raise.
    """
    p = tmp_path / "missing.json"
    assert ajs.load_active_jobs_strict(p) == []


def test_load_active_jobs_strict_raises_on_corrupt(tmp_path):
    p = tmp_path / "corrupt.json"
    p.write_text("not valid json")
    with pytest.raises(ActiveJobsCorruptError, match="invalid JSON"):
        ajs.load_active_jobs_strict(p)


def test_load_active_jobs_strict_raises_on_non_list(tmp_path):
    p = tmp_path / "wrong_shape.json"
    p.write_text(json.dumps({"key": "value"}))
    with pytest.raises(ActiveJobsCorruptError, match="non-list"):
        ajs.load_active_jobs_strict(p)


def test_load_active_jobs_strict_returns_list(tmp_path):
    p = tmp_path / "good.json"
    p.write_text(json.dumps([{"job_name": "j1"}, {"job_name": "j2"}]))
    assert ajs.load_active_jobs_strict(p) == [
        {"job_name": "j1"},
        {"job_name": "j2"},
    ]


# ── Quarantine helper ────────────────────────────────────────────────────


def test_quarantine_helper_idempotent_on_missing(tmp_path):
    """If the path doesn't exist, quarantine is a no-op."""
    missing = tmp_path / "missing.json"
    result = ajs._quarantine_corrupt_file(missing)
    assert result == missing
    assert not missing.exists()


def test_quarantine_helper_renames_to_sibling(tmp_path):
    """Quarantine must put the file under .omx/state-style sibling
    paths, NOT under /tmp/* (CLAUDE.md `forbidden_tmp_paths`).
    """
    p = tmp_path / "active.json"
    p.write_text("garbage")
    result = ajs._quarantine_corrupt_file(p)
    assert result.parent == p.parent  # sibling, not /tmp
    assert result.name.startswith("active.json.corrupt.")
    assert "/tmp/" not in str(result)
    assert result.read_text() == "garbage"  # forensic preservation
    assert not p.exists()  # canonical path is freed


def test_quarantine_helper_handles_same_second_collision(tmp_path):
    """If two corruption events happen the same UTC second, the second
    one must use a counter-suffixed name (no collision on rename)."""
    p = tmp_path / "active.json"
    p.write_text("first_corruption")
    first = ajs._quarantine_corrupt_file(p)
    # Now produce a second corruption with the same canonical path.
    p.write_text("second_corruption")
    second = ajs._quarantine_corrupt_file(p)
    assert first != second
    assert first.exists()
    assert second.exists()
    assert first.read_text() == "first_corruption"
    assert second.read_text() == "second_corruption"


def test_quarantine_disabled_raises_without_renaming(tmp_path, monkeypatch):
    """`quarantine_on_corrupt=False` must raise without touching the file."""
    state_path = tmp_path / "active.json"
    lock_path = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", state_path)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock_path)
    state_path.write_text("garbage")
    with pytest.raises(ActiveJobsCorruptError):
        ajs.update_active_jobs_locked(
            lambda rows: rows,
            quarantine_on_corrupt=False,
        )
    # File is untouched.
    assert state_path.exists()
    assert state_path.read_text() == "garbage"
    # No quarantine sibling created.
    assert list(tmp_path.glob("active.json.corrupt.*")) == []


# ── Sister mutators all benefit from strict load ─────────────────────────


def test_register_job_refuses_on_corrupt(tmp_path, monkeypatch):
    state_path = tmp_path / "active.json"
    lock_path = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", state_path)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock_path)
    state_path.write_text("not valid json")
    with pytest.raises(ActiveJobsCorruptError):
        ajs.register_job({"job_name": "j_new"})


def test_upsert_job_refuses_on_corrupt(tmp_path, monkeypatch):
    state_path = tmp_path / "active.json"
    lock_path = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", state_path)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock_path)
    state_path.write_text("not valid json")
    with pytest.raises(ActiveJobsCorruptError):
        ajs.upsert_job({"job_name": "j_new"})


def test_mark_job_terminal_refuses_on_corrupt(tmp_path, monkeypatch):
    state_path = tmp_path / "active.json"
    lock_path = tmp_path / "active.json.lock"
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_PATH", state_path)
    monkeypatch.setattr(ajs, "ACTIVE_JOBS_LOCK", lock_path)
    state_path.write_text("not valid json")
    with pytest.raises(ActiveJobsCorruptError):
        ajs.mark_job_terminal("j_new", terminal_status="completed")
