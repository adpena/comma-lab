"""Runtime quarantine decode gate + M1/C2-family tracked-defaults regression.

Operator 2026-07-21: the guard must apply to main, every worktree, and every
Claude/Codex subagent. That requires (1) the retired-archive family in the TRACKED
``DEFAULT_QUARANTINED`` (ships via ``import tac``, unlike the gitignored manifest),
and (2) a runtime gate at the DECODE, not just the spawn prompt.
"""

from __future__ import annotations

import os

import pytest

from tac.artifact_quarantine import (
    DEFAULT_QUARANTINED,
    QuarantineViolation,
    assert_not_quarantined_archive,
    is_quarantined_archive,
    scan_text,
    sha256_of_file,
)

_ENV = "TAC_ARTIFACT_QUARANTINE_SIGNAL_ONLY"


def _default_ids() -> set[str]:
    return {str(r["id"]) for r in DEFAULT_QUARANTINED}


def test_m1c2_family_is_in_tracked_defaults_not_just_gitignored_manifest():
    # The recurrence-#2 family MUST be in the tracked tuple so worktrees/subagents
    # (which never see the gitignored manifest) are protected by `import tac` alone.
    ids = _default_ids()
    for ident in ("a386a854", "3748485f", "d633e6bf", "e9e42971",
                  "m1_byteclose_20260721", "m1_c2_glue_rebuild_20260719"):
        assert ident in ids, f"{ident} missing from DEFAULT_QUARANTINED"


def test_scan_text_flags_the_m1_sha_prefix():
    hits = scan_text("decode the M1/C2 control archive a386a854e2483f839191")
    assert any(h.identifier == "a386a854" for h in hits)


def test_scan_text_waiver_still_bypasses():
    text = ("archive a386a854 QUARANTINE-WAIVER: HARVEST-SIGNAL-ONLY — "
            "citing only the measured d_seg, never the bytes")
    assert scan_text(text) == []


def test_runtime_gate_refuses_quarantined_path_token(tmp_path):
    # A file under an evidence dir whose name is a quarantined path_token is refused
    # by content-independent path matching (catches decode-by-discovery).
    d = tmp_path / "m1_byteclose_20260721"
    d.mkdir()
    archive = d / "m1_candidate_archive.zip"
    archive.write_bytes(b"PK\x03\x04not-a-real-zip")
    assert is_quarantined_archive(archive)
    with pytest.raises(QuarantineViolation):
        assert_not_quarantined_archive(archive, context="unit-test decode")


def test_runtime_gate_passes_clean_archive(tmp_path):
    archive = tmp_path / "fresh_v10_candidate" / "archive.zip"
    archive.parent.mkdir()
    archive.write_bytes(b"PK\x03\x04fresh-current-vehicle-bytes")
    assert is_quarantined_archive(archive) == []
    assert_not_quarantined_archive(archive, context="clean")  # must not raise


def test_signal_only_env_bypasses_with_rationale(tmp_path, monkeypatch):
    d = tmp_path / "m1_byteclose_20260721"
    d.mkdir()
    archive = d / "archive.zip"
    archive.write_bytes(b"bytes")
    # Too-short rationale does NOT bypass.
    monkeypatch.setenv(_ENV, "short")
    with pytest.raises(QuarantineViolation):
        assert_not_quarantined_archive(archive)
    # A real (>=10 char) rationale bypasses (signal-only, logged).
    monkeypatch.setenv(_ENV, "harvesting only the measured receipt, never the bytes")
    assert_not_quarantined_archive(archive)  # must not raise


def test_sha256_of_file_matches_hashlib(tmp_path):
    import hashlib
    p = tmp_path / "x.bin"
    payload = b"\x00\x01\x02deadbeef" * 4096
    p.write_bytes(payload)
    assert sha256_of_file(p) == hashlib.sha256(payload).hexdigest()
    assert sha256_of_file(tmp_path / "does-not-exist") == ""


def test_gate_is_noop_when_env_absent(tmp_path):
    os.environ.pop(_ENV, None)
    archive = tmp_path / "clean" / "archive.zip"
    archive.parent.mkdir()
    archive.write_bytes(b"clean")
    assert_not_quarantined_archive(archive)
