"""Tests for codex round-4 catalog #133 — Catalog #131 accept-list audit.

Bug class (codex round-4 HIGH 1, 2026-05-09): Catalog #131's
``_BARE_WRITE_CANONICAL_HELPERS`` exempt list named
``src/tac/deploy/azure/azure_dispatch.py`` as "already locked", but the
file did bare ``write_text`` to ``.omx/state/azure_active_vms.json``
with no fcntl context. Strict #131 reported the bare-write META class
extinct while concurrent Azure provisions could still drop VM rows.

The fix gate (#133) iterates every entry in
``_BARE_WRITE_CANONICAL_HELPERS`` and verifies the file actually contains
a canonical lock-pattern token (``fcntl.flock``, ``LOCK_EX``,
``register_active_vm_record``, etc.) OR is in the deferred-rationale
dict ``_CHECK_131_EXEMPT_AUDIT_DEFERRED_RATIONALE`` OR has a file-wide
``# CHECK_131_EXEMPT_AUDIT_OK:<reason>`` waiver.

Memory: feedback_codex_round4_findings_fix_with_self_protection_landed_20260509.md.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tac import preflight as _preflight
from tac.preflight import (
    PreflightError,
    check_no_excluded_writers_in_check_131_accept_list,
)


# ── Live-repo sanity ────────────────────────────────────────────────────


def test_133_live_repo_clean():
    """Live-repo sanity: catalog #133 must land at 0 violations.

    The Azure HIGH 1 fix swapped the exempt list to the new canonical
    helper module. Any future regression (re-introducing a bare-writer
    file in the exempt list) will fail this test loud-and-early.
    """
    v = check_no_excluded_writers_in_check_131_accept_list(
        strict=False, verbose=False
    )
    assert v == [], (
        f"Catalog #133 landed with {len(v)} violations:\n"
        + "\n".join(v[:3])
    )


# ── Catch the Azure-style false-green pattern ──────────────────────────


def test_133_catches_exempt_entry_with_no_lock_pattern(tmp_path, monkeypatch):
    """A file in the exempt list that does bare write_text and contains
    no fcntl/LOCK_EX/canonical-helper token must be flagged.
    """
    root = tmp_path
    # Create a fake exempt file with a bare write
    target = root / "src" / "tac" / "deploy" / "fake" / "fake_dispatch.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        'import json\n'
        'from pathlib import Path\n'
        'P = Path("/tmp/x.json")\n'
        'def _save(rows):\n'
        '    P.write_text(json.dumps(rows))\n'
    )
    # Monkey-patch the helpers list to include only this fake entry
    monkeypatch.setattr(
        _preflight,
        "_BARE_WRITE_CANONICAL_HELPERS",
        ("src/tac/deploy/fake/fake_dispatch.py",),
    )
    monkeypatch.setattr(
        _preflight,
        "_CHECK_131_EXEMPT_AUDIT_DEFERRED_RATIONALE",
        {},
    )
    v = check_no_excluded_writers_in_check_131_accept_list(
        repo_root=root, strict=False, verbose=False
    )
    assert len(v) == 1
    assert "fake_dispatch.py" in v[0]
    assert "false-green" in v[0].lower() or "no canonical lock-pattern" in v[0]


def test_133_accepts_exempt_entry_with_fcntl_flock(tmp_path, monkeypatch):
    """A file with `fcntl.flock` is a real locked writer and must pass."""
    root = tmp_path
    target = root / "src" / "tac" / "deploy" / "fake" / "good.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        'import fcntl\n'
        'import json\n'
        'from pathlib import Path\n'
        'P = Path("/tmp/x.json")\n'
        'def save_locked(rows):\n'
        '    with open("/tmp/lock", "w") as f:\n'
        '        fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n'
        '        P.write_text(json.dumps(rows))\n'
    )
    monkeypatch.setattr(
        _preflight,
        "_BARE_WRITE_CANONICAL_HELPERS",
        ("src/tac/deploy/fake/good.py",),
    )
    monkeypatch.setattr(
        _preflight,
        "_CHECK_131_EXEMPT_AUDIT_DEFERRED_RATIONALE",
        {},
    )
    v = check_no_excluded_writers_in_check_131_accept_list(
        repo_root=root, strict=False, verbose=False
    )
    assert v == []


def test_133_accepts_canonical_helper_delegation(tmp_path, monkeypatch):
    """A file that imports/uses `register_instance` from the canonical
    Vast.ai tracker is a delegation pattern — accept it even though the
    file itself has no fcntl call.
    """
    root = tmp_path
    target = root / "src" / "tac" / "deploy" / "vastai" / "client_like.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        'from tac.vastai_tracker import register_instance\n'
        'def add(handle, label):\n'
        '    register_instance(handle, label)\n'
    )
    monkeypatch.setattr(
        _preflight,
        "_BARE_WRITE_CANONICAL_HELPERS",
        ("src/tac/deploy/vastai/client_like.py",),
    )
    monkeypatch.setattr(
        _preflight,
        "_CHECK_131_EXEMPT_AUDIT_DEFERRED_RATIONALE",
        {},
    )
    v = check_no_excluded_writers_in_check_131_accept_list(
        repo_root=root, strict=False, verbose=False
    )
    assert v == []


def test_133_accepts_threading_lock_for_in_process_module(tmp_path, monkeypatch):
    """In-process cache modules use `threading.Lock` (not fcntl) — accept."""
    root = tmp_path
    target = root / "src" / "tac" / "fake_cache.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        'import threading\n'
        '_PATCH_LOCK = threading.Lock()\n'
        'def cached():\n'
        '    with _PATCH_LOCK:\n'
        '        pass\n'
    )
    monkeypatch.setattr(
        _preflight,
        "_BARE_WRITE_CANONICAL_HELPERS",
        ("src/tac/fake_cache.py",),
    )
    monkeypatch.setattr(
        _preflight,
        "_CHECK_131_EXEMPT_AUDIT_DEFERRED_RATIONALE",
        {},
    )
    v = check_no_excluded_writers_in_check_131_accept_list(
        repo_root=root, strict=False, verbose=False
    )
    assert v == []


# ── Deferred-rationale dict ────────────────────────────────────────────


def test_133_accepts_deferred_rationale_entry(tmp_path, monkeypatch):
    """Files in `_CHECK_131_EXEMPT_AUDIT_DEFERRED_RATIONALE` skip the
    audit. The rationale captures the technical-debt acknowledgement.
    """
    root = tmp_path
    target = root / "tools" / "deferred.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        'import json\n'
        'from pathlib import Path\n'
        'P = Path("/tmp/x.json")\n'
        'def save(rows):\n'
        '    P.write_text(json.dumps(rows))\n'  # bare write, no lock
    )
    monkeypatch.setattr(
        _preflight,
        "_BARE_WRITE_CANONICAL_HELPERS",
        ("tools/deferred.py",),
    )
    monkeypatch.setattr(
        _preflight,
        "_CHECK_131_EXEMPT_AUDIT_DEFERRED_RATIONALE",
        {"tools/deferred.py": "Operator-CLI single-writer; follow-up deferred"},
    )
    v = check_no_excluded_writers_in_check_131_accept_list(
        repo_root=root, strict=False, verbose=False
    )
    assert v == []


# ── File-level waiver ──────────────────────────────────────────────────


def test_133_accepts_file_level_waiver(tmp_path, monkeypatch):
    """A `# CHECK_131_EXEMPT_AUDIT_OK:<reason>` marker file-wide bypasses
    the audit.
    """
    root = tmp_path
    target = root / "tools" / "waived_audit.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        '# CHECK_131_EXEMPT_AUDIT_OK: tested separately via test_concurrent_xyz\n'
        'import json\n'
        'from pathlib import Path\n'
        'P = Path("/tmp/x.json")\n'
        'def save(rows):\n'
        '    P.write_text(json.dumps(rows))\n'
    )
    monkeypatch.setattr(
        _preflight,
        "_BARE_WRITE_CANONICAL_HELPERS",
        ("tools/waived_audit.py",),
    )
    monkeypatch.setattr(
        _preflight,
        "_CHECK_131_EXEMPT_AUDIT_DEFERRED_RATIONALE",
        {},
    )
    v = check_no_excluded_writers_in_check_131_accept_list(
        repo_root=root, strict=False, verbose=False
    )
    assert v == []


# ── Missing-file detection ─────────────────────────────────────────────


def test_133_catches_exempt_entry_for_nonexistent_file(tmp_path, monkeypatch):
    """An exempt-list entry pointing at a file that doesn't exist must be flagged."""
    root = tmp_path
    monkeypatch.setattr(
        _preflight,
        "_BARE_WRITE_CANONICAL_HELPERS",
        ("src/tac/this/does/not/exist.py",),
    )
    monkeypatch.setattr(
        _preflight,
        "_CHECK_131_EXEMPT_AUDIT_DEFERRED_RATIONALE",
        {},
    )
    v = check_no_excluded_writers_in_check_131_accept_list(
        repo_root=root, strict=False, verbose=False
    )
    assert len(v) == 1
    assert "does not exist" in v[0]


# ── Strict-mode round-trip ─────────────────────────────────────────────


def test_133_strict_raises(tmp_path, monkeypatch):
    root = tmp_path
    target = root / "src" / "tac" / "bad.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        'def write_bare():\n'
        '    open("/tmp/x", "w").write("noop")\n'
    )
    monkeypatch.setattr(
        _preflight,
        "_BARE_WRITE_CANONICAL_HELPERS",
        ("src/tac/bad.py",),
    )
    monkeypatch.setattr(
        _preflight,
        "_CHECK_131_EXEMPT_AUDIT_DEFERRED_RATIONALE",
        {},
    )
    with pytest.raises(PreflightError, match="check_no_excluded_writers"):
        check_no_excluded_writers_in_check_131_accept_list(
            repo_root=root, strict=True, verbose=False
        )


# ── Real Azure module integration ──────────────────────────────────────


def test_133_real_azure_active_vms_state_passes():
    """The new canonical Azure helper module must be in the exempt list
    AND pass the audit (contains real fcntl + LOCK_EX patterns).
    """
    azure_helper_rel = "src/tac/deploy/azure/active_vms_state.py"
    assert azure_helper_rel in _preflight._BARE_WRITE_CANONICAL_HELPERS

    repo_root = Path(__file__).resolve().parents[3]
    target = repo_root / azure_helper_rel
    assert target.exists(), f"{azure_helper_rel} should exist after round-4 HIGH 1 fix"
    text = target.read_text()
    assert "fcntl.flock" in text
    assert "LOCK_EX" in text


def test_133_old_azure_dispatch_no_longer_in_exempt_list():
    """The old `azure_dispatch.py` exempt entry should have been replaced
    with the new canonical helper. The dispatcher itself now delegates.
    """
    old_path = "src/tac/deploy/azure/azure_dispatch.py"
    # azure_dispatch.py is no longer the canonical lock-holder; it
    # delegates. It might or might not be in the exempt list, but if it
    # is, it must satisfy the audit (delegation pattern).
    if old_path in _preflight._BARE_WRITE_CANONICAL_HELPERS:
        repo_root = Path(__file__).resolve().parents[3]
        text = (repo_root / old_path).read_text()
        # Must contain a canonical-helper import to satisfy the audit
        assert (
            "register_active_vm_record" in text
            or "active_vms_state" in text
        ), "azure_dispatch.py must delegate to canonical helper if exempt"


# ── Multi-process Azure regression test (the underlying bug) ───────────


def _register_vm_worker(args):
    """Module-level worker for spawn-mode multiprocessing (macOS default)."""
    state_path, lock_path, vm_label = args
    from tac.deploy.azure import active_vms_state
    active_vms_state.register_active_vm_record(
        {"vm_name": vm_label, "label": vm_label, "provisioned_at": 1234},
        path=Path(state_path),
        lock_path=Path(lock_path),
    )


def test_133_concurrent_azure_vm_register_preserves_both_rows(tmp_path):
    """Regression test: two concurrent register_active_vm_record() calls
    with different VMs MUST both end up in the file (the bug class this
    META gate's underlying fix prevents).
    """
    import multiprocessing
    import json

    state_path = tmp_path / "azure_active_vms.json"
    lock_path = tmp_path / "azure_active_vms.json.lock"

    args_list = [
        (str(state_path), str(lock_path), f"vm-{i}") for i in range(8)
    ]
    procs = [
        multiprocessing.Process(target=_register_vm_worker, args=(a,))
        for a in args_list
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=15)

    rows = json.loads(state_path.read_text())
    assert len(rows) == 8, (
        f"Expected 8 rows after concurrent register; got {len(rows)}: {rows}"
    )
    names = {r["vm_name"] for r in rows}
    assert names == {f"vm-{i}" for i in range(8)}
