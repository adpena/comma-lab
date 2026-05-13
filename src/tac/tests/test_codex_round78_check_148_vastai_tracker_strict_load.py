"""Tests for Catalog #148 — `tac.vastai_tracker` mutating writers must use
the strict loader.

Bug class: codex round 7 HIGH 2 (2026-05-09). The previous
`tac.vastai_tracker._load_records` returned `[]` on malformed JSON.
`register_instance` then wrote the new single record over the corrupt file,
dropping every previously-tracked active instance.

Fix: new `tac.vastai_tracker.load_active_instances_strict` raises
`VastaiTrackerCorruptError`; `register_instance` / `remove_instance` use it
inside the fcntl lock and quarantine corrupt files to `<path>.corrupt.<utc>`.

Memory: feedback_codex_round78_findings_fix_with_self_protection_landed_20260509.md
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from tac.preflight import check_vastai_tracker_strict_load
from tac.vastai_tracker import (
    VastaiTrackerCorruptError,
    load_active_instances_strict,
    register_instance,
    remove_instance,
)


# ── Direct API tests for the strict loader ────────────────────────────────


def test_strict_loader_returns_empty_for_missing_file(tmp_path: Path) -> None:
    p = tmp_path / "tracker.json"
    assert load_active_instances_strict(p) == []


def test_strict_loader_returns_empty_for_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "tracker.json"
    p.write_text("")
    assert load_active_instances_strict(p) == []


def test_strict_loader_returns_list_for_canonical_shape(tmp_path: Path) -> None:
    p = tmp_path / "tracker.json"
    p.write_text(json.dumps([{"instance_id": "a"}, {"instance_id": "b"}]))
    out = load_active_instances_strict(p)
    assert out == [{"instance_id": "a"}, {"instance_id": "b"}]


def test_strict_loader_handles_legacy_dict_shape(tmp_path: Path) -> None:
    p = tmp_path / "tracker.json"
    p.write_text(json.dumps({"instances": [{"instance_id": "a"}]}))
    out = load_active_instances_strict(p)
    assert out == [{"instance_id": "a"}]


def test_strict_loader_raises_on_malformed_json(tmp_path: Path) -> None:
    p = tmp_path / "tracker.json"
    p.write_text("not-json")
    with pytest.raises(VastaiTrackerCorruptError) as excinfo:
        load_active_instances_strict(p)
    assert "malformed" in str(excinfo.value).lower()


def test_strict_loader_raises_on_dict_without_instances_key(tmp_path: Path) -> None:
    p = tmp_path / "tracker.json"
    p.write_text(json.dumps({"foo": "bar"}))
    with pytest.raises(VastaiTrackerCorruptError):
        load_active_instances_strict(p)


def test_strict_loader_raises_on_scalar(tmp_path: Path) -> None:
    p = tmp_path / "tracker.json"
    p.write_text(json.dumps(42))
    with pytest.raises(VastaiTrackerCorruptError):
        load_active_instances_strict(p)


def test_strict_loader_raises_on_dict_with_non_list_instances(tmp_path: Path) -> None:
    p = tmp_path / "tracker.json"
    p.write_text(json.dumps({"instances": "not-a-list"}))
    with pytest.raises(VastaiTrackerCorruptError):
        load_active_instances_strict(p)


# ── End-to-end: register_instance + corrupt-tracker quarantine ────────────


def test_register_instance_quarantines_on_corrupt_tracker(tmp_path: Path) -> None:
    """The CRITICAL bug fix: register_instance must NOT silently overwrite
    a corrupt tracker with a single new record (which would drop every
    previously-tracked active instance)."""
    # Set up a fake repo root
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    track_p = tmp_path / ".omx" / "state" / "vastai_active_instances.json"
    track_p.write_text("not-json")  # corrupt!

    with pytest.raises(VastaiTrackerCorruptError):
        register_instance(
            instance_id="666",
            label="should-not-overwrite",
            metadata={},
            repo_root=tmp_path,
        )

    # The corrupt file should be quarantined.
    quarantined = list((tmp_path / ".omx" / "state").glob(
        "vastai_active_instances.json.corrupt.*"
    ))
    assert len(quarantined) == 1, f"expected 1 quarantine; found {quarantined}"
    # The original tracker file should NOT have been silently overwritten
    # with a single new record (the bug).
    if track_p.exists():
        loaded = load_active_instances_strict(track_p)
        # If the file exists post-quarantine it should be empty (rename moved it)
        # OR it has been re-created as missing (rename succeeded).
        assert loaded == [], (
            f"register_instance silently overwrote corrupt tracker with "
            f"{loaded}; expected quarantine + re-raise."
        )


def test_register_instance_succeeds_on_valid_tracker(tmp_path: Path) -> None:
    """Sanity: register_instance still works on the normal happy path."""
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    track_p = tmp_path / ".omx" / "state" / "vastai_active_instances.json"
    track_p.write_text(json.dumps([{"instance_id": "preexisting", "label": "old"}]))

    record = register_instance(
        instance_id="123",
        label="new-instance",
        metadata={"dph": 0.25},
        repo_root=tmp_path,
    )
    assert record["instance_id"] == "123"
    assert record["label"] == "new-instance"

    # Both records should be in the tracker now (no silent drop).
    rows = load_active_instances_strict(track_p)
    assert len(rows) == 2
    assert {r["instance_id"] for r in rows} == {"preexisting", "123"}


def test_register_instance_succeeds_on_missing_tracker(tmp_path: Path) -> None:
    """First-registration-ever case: missing file is allowed to succeed."""
    (tmp_path / ".omx" / "state").mkdir(parents=True)

    record = register_instance(
        instance_id="123",
        label="first-instance",
        metadata={},
        repo_root=tmp_path,
    )
    assert record["instance_id"] == "123"

    track_p = tmp_path / ".omx" / "state" / "vastai_active_instances.json"
    rows = load_active_instances_strict(track_p)
    assert len(rows) == 1
    assert rows[0]["instance_id"] == "123"


def test_remove_instance_quarantines_on_corrupt_tracker(tmp_path: Path) -> None:
    """remove_instance must also fail-closed on corrupt state."""
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    track_p = tmp_path / ".omx" / "state" / "vastai_active_instances.json"
    track_p.write_text("[malformed")  # corrupt

    with pytest.raises(VastaiTrackerCorruptError):
        remove_instance("666", repo_root=tmp_path)

    quarantined = list((tmp_path / ".omx" / "state").glob(
        "vastai_active_instances.json.corrupt.*"
    ))
    assert len(quarantined) == 1


# ── Preflight gate (Catalog #148) static checks ───────────────────────────


def test_check_148_passes_on_real_repo() -> None:
    """Smoke: live count = 0 on the repo as committed."""
    v = check_vastai_tracker_strict_load(strict=False, verbose=False)
    assert v == [], (
        f"Catalog #148 live count must be 0; got {len(v)} violation(s):\n"
        + "\n".join(v[:3])
    )


def test_check_148_strict_mode_does_not_raise_on_real_repo() -> None:
    check_vastai_tracker_strict_load(strict=True, verbose=False)


def test_check_148_detects_writer_with_only_lossy_loader(tmp_path: Path) -> None:
    """Synthetic: a writer that calls _load_records + _write_records but NOT
    load_active_instances_strict should be flagged."""
    target = tmp_path / "src" / "tac" / "vastai_tracker.py"
    target.parent.mkdir(parents=True)
    target.write_text(dedent("""
        def _load_records(path):
            return []

        def _write_records(path, records):
            path.write_text(str(records))

        def register_instance(instance_id, label):
            path = ...
            records = _load_records(path)
            records.append({"id": instance_id})
            _write_records(path, records)
            return records[-1]
    """))
    v = check_vastai_tracker_strict_load(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1, v
    assert "register_instance" in v[0]


def test_check_148_allows_writer_with_strict_loader(tmp_path: Path) -> None:
    """Synthetic: writer that uses load_active_instances_strict is OK."""
    target = tmp_path / "src" / "tac" / "vastai_tracker.py"
    target.parent.mkdir(parents=True)
    target.write_text(dedent("""
        def _load_records(path):
            return []

        def load_active_instances_strict(path):
            raise NotImplementedError

        def _write_records(path, records):
            path.write_text(str(records))

        def register_instance(instance_id, label):
            path = ...
            try:
                records = load_active_instances_strict(path)
            except Exception:
                _load_records(path)  # fallback (also detected — but strict load present)
                raise
            records.append({"id": instance_id})
            _write_records(path, records)
            return records[-1]
    """))
    v = check_vastai_tracker_strict_load(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == [], v


def test_check_148_waiver_on_def_line_is_respected(tmp_path: Path) -> None:
    target = tmp_path / "src" / "tac" / "vastai_tracker.py"
    target.parent.mkdir(parents=True)
    target.write_text(dedent("""
        def _load_records(path):
            return []

        def _write_records(path, records):
            path.write_text(str(records))

        def register_instance(instance_id, label):  # VASTAI_TRACKER_STRICT_LOAD_OK:test-fixture
            path = ...
            records = _load_records(path)
            records.append({"id": instance_id})
            _write_records(path, records)
    """))
    v = check_vastai_tracker_strict_load(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == [], v


def test_check_148_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    from tac.preflight import PreflightError
    target = tmp_path / "src" / "tac" / "vastai_tracker.py"
    target.parent.mkdir(parents=True)
    target.write_text(dedent("""
        def _load_records(path):
            return []

        def _write_records(path, records):
            path.write_text(str(records))

        def register_instance(x):
            r = _load_records(...)
            _write_records(..., r + [x])
    """))
    with pytest.raises(PreflightError):
        check_vastai_tracker_strict_load(
            repo_root=tmp_path, strict=True, verbose=False,
        )


def test_check_148_pure_loader_function_is_not_a_writer(tmp_path: Path) -> None:
    """A function that only loads (no _write_records) is not a writer."""
    target = tmp_path / "src" / "tac" / "vastai_tracker.py"
    target.parent.mkdir(parents=True)
    target.write_text(dedent("""
        def _load_records(path):
            return []

        def list_instances(path):
            return _load_records(path)
    """))
    v = check_vastai_tracker_strict_load(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == [], v


def test_check_148_handles_missing_target(tmp_path: Path) -> None:
    """When src/tac/vastai_tracker.py doesn't exist, the gate is OK."""
    v = check_vastai_tracker_strict_load(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_check_148_handles_syntax_error_target(tmp_path: Path) -> None:
    """A syntactically broken tracker file does not crash the check."""
    target = tmp_path / "src" / "tac" / "vastai_tracker.py"
    target.parent.mkdir(parents=True)
    target.write_text("this is not valid python !!!")
    v = check_vastai_tracker_strict_load(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []
