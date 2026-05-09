"""Tests for codex round-5 catalog #140 — `_save_*` writers must own their lock end-to-end.

Bug class (codex round-5 HIGH 2, 2026-05-09):
``LightningDispatcher._save_state`` documented "MUST be called inside
``_lightning_state_lock``" but did NOT runtime-enforce it. Any caller could
pass a stale snapshot from outside the lock and silently lose concurrent
rows. ``scripts/launch_lane_lightning.py`` violated the contract by doing
``sessions = dispatcher.list_sessions(); ...; dispatcher._save_state(sessions)``
outside any lock.

The fix:

1. ``_lightning_state_lock`` now tracks an in-process depth counter
   (``_lightning_state_lock_depth``).
2. ``_save_state`` calls ``_lightning_state_lock_held()`` at entry and
   raises ``RuntimeError`` if the lock is not held.
3. New canonical ``update_session_locked`` / ``update_sessions_locked``
   APIs own the full lock-load-mutate-save cycle.
4. ``scripts/launch_lane_lightning.py`` uses ``update_session_locked``
   instead of the old ``list_sessions() + _save_state(...)`` pair.
5. Sister fix in ``tac.deploy.lightning.active_jobs_state._save_active_jobs``
   and ``tac.deploy.azure.active_vms_state._save_active_vms_atomic`` adds
   the same lock-held assertion.
6. STRICT preflight gate #140 refuses any future ``_save_*`` writer
   documented as caller-locked without runtime enforcement.

Memory: feedback_codex_round5_findings_fix_with_self_protection_landed_20260509.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_state_writers_own_their_lock_end_to_end,
)


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "tools").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    return tmp_path


# ── Live-repo sanity ────────────────────────────────────────────────────


def test_140_live_repo_clean():
    """Live-repo: catalog #140 lands at 0 violations after fix."""
    v = check_state_writers_own_their_lock_end_to_end(
        strict=False, verbose=False
    )
    assert v == [], (
        f"Catalog #140 landed with {len(v)} violations:\n"
        + "\n".join(v[:3])
    )


# ── Catch the bug class ────────────────────────────────────────────────


def test_140_catches_caller_locked_without_enforcement(tmp_path):
    """A `_save_*` function with the documented "MUST be called inside ..."
    contract but no runtime enforcement MUST be flagged."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_writer.py"
    target.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "P = Path('.fake/state.json')\n"
        "\n"
        "def _save_state(rows):\n"
        "    '''Atomic write — unique tmp + fsync + os.replace.\n"
        "\n"
        "    MUST be called inside _fake_lock. The CALLER is responsible\n"
        "    for the lock; this method enforces only the unique-tmp invariants.\n"
        "    '''\n"
        "    P.write_text(json.dumps(rows))\n"
    )
    v = check_state_writers_own_their_lock_end_to_end(
        repo_root=root, strict=False, verbose=False
    )
    assert any("Check 140" in x and "_save_state" in x for x in v), (
        f"expected Check 140 hit on _save_state; got: {v}"
    )


def test_140_accepts_lock_held_assertion(tmp_path):
    """A `_save_*` function with explicit `_lock_held()` enforcement is OK."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_writer.py"
    target.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "_lock_depth = 0\n"
        "\n"
        "def _lock_held():\n"
        "    return _lock_depth > 0\n"
        "\n"
        "def _save_state(rows):\n"
        "    '''Atomic write.\n"
        "\n"
        "    MUST be called inside the lock.\n"
        "    '''\n"
        "    if not _lock_held():\n"
        "        raise RuntimeError('lock not held')\n"
        "    P = Path('.fake/state.json')\n"
        "    P.write_text(json.dumps(rows))\n"
    )
    v = check_state_writers_own_their_lock_end_to_end(
        repo_root=root, strict=False, verbose=False
    )
    assert v == [], f"expected 0 violations; got: {v}"


def test_140_accepts_fcntl_acquisition(tmp_path):
    """A `_save_*` function that itself does fcntl.flock acquisition is OK
    (it doesn't claim caller-locked; it self-locks)."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_writer.py"
    target.write_text(
        "import fcntl, os, json\n"
        "from pathlib import Path\n"
        "\n"
        "def _save_state(rows):\n"
        "    '''Atomic write.\n"
        "\n"
        "    MUST be called inside _fake_lock context.\n"
        "    '''\n"
        "    P = Path('.fake/state.json')\n"
        "    fd = os.open('lock', os.O_RDWR | os.O_CREAT)\n"
        "    fcntl.flock(fd, fcntl.LOCK_EX)\n"
        "    try:\n"
        "        P.write_text(json.dumps(rows))\n"
        "    finally:\n"
        "        os.close(fd)\n"
    )
    v = check_state_writers_own_their_lock_end_to_end(
        repo_root=root, strict=False, verbose=False
    )
    assert v == [], f"expected 0 violations; got: {v}"


def test_140_respects_same_line_waiver(tmp_path):
    """Same-line `# CALLER_LOCK_ENFORCED_OK:...` waives the gate."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_writer.py"
    target.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "def _save_unittest_state(rows):  # CALLER_LOCK_ENFORCED_OK:single-process-test-scaffold\n"
        "    '''Atomic write.\n"
        "\n"
        "    MUST be called inside the lock (single-process test scaffold).\n"
        "    '''\n"
        "    P = Path('.fake/state.json')\n"
        "    P.write_text(json.dumps(rows))\n"
    )
    v = check_state_writers_own_their_lock_end_to_end(
        repo_root=root, strict=False, verbose=False
    )
    assert v == [], f"expected 0 violations after waiver; got: {v}"


def test_140_strict_mode_raises(tmp_path):
    """strict=True raises PreflightError when violations are present."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_writer.py"
    target.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "def _save_state(rows):\n"
        "    '''MUST be called inside the lock.'''\n"
        "    P = Path('.fake/state.json')\n"
        "    P.write_text(json.dumps(rows))\n"
    )
    with pytest.raises(PreflightError, match="Check 140|own-their-lock"):
        check_state_writers_own_their_lock_end_to_end(
            repo_root=root, strict=True, verbose=False
        )


def test_140_only_save_or_save_state_names(tmp_path):
    """Only `_save_*` / `save_*_state` names are scanned. Other names with
    "MUST be called inside the lock" docstrings are out-of-scope (e.g.
    ``_load_xyz`` doesn't apply)."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_loader.py"
    target.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "def _load_state():\n"
        "    '''MUST be called inside the lock for consistency.'''\n"
        "    return json.loads(Path('.fake/state.json').read_text())\n"
    )
    v = check_state_writers_own_their_lock_end_to_end(
        repo_root=root, strict=False, verbose=False
    )
    assert v == [], f"loader functions should be out of scope; got: {v}"


def test_140_test_files_excluded(tmp_path):
    """Test files (test_*.py / under tests/) must not be scanned."""
    root = _make_repo(tmp_path)
    tests_dir = root / "src" / "tac" / "tests"
    tests_dir.mkdir(parents=True)
    target = tests_dir / "test_fake.py"
    target.write_text(
        "def test_save_state():\n"
        "    def _save_state(rows):\n"
        "        '''MUST be called inside the lock.'''\n"
        "        pass\n"
    )
    v = check_state_writers_own_their_lock_end_to_end(
        repo_root=root, strict=False, verbose=False
    )
    assert v == [], f"test files should be excluded; got: {v}"


# ── Behavioural test of the actual _save_state runtime guard ───────────


def test_lightning_save_state_raises_outside_lock():
    """LightningDispatcher._save_state must raise when called outside the lock."""
    from tac.deploy.lightning.lightning_dispatch import LightningDispatcher

    with pytest.raises(RuntimeError, match="WITHOUT holding|_lightning_state_lock|catalog #140"):
        LightningDispatcher._save_state([])


def test_lightning_save_state_works_inside_lock(tmp_path, monkeypatch):
    """_save_state succeeds when called inside _lightning_state_lock."""
    from tac.deploy.lightning import lightning_dispatch as ld

    state_path = tmp_path / "lightning_state.json"
    monkeypatch.setattr(ld, "LIGHTNING_STATE", state_path)
    monkeypatch.setattr(ld, "LIGHTNING_STATE_LOCK", state_path.with_suffix(".json.lock"))

    with ld._lightning_state_lock():
        ld.LightningDispatcher._save_state([{"session_id": "test"}])

    assert state_path.exists()


def test_update_session_locked_atomic_load_mutate_save(tmp_path, monkeypatch):
    """update_session_locked owns the full lock-load-mutate-save cycle."""
    from tac.deploy.lightning import lightning_dispatch as ld

    state_path = tmp_path / "lightning_state.json"
    monkeypatch.setattr(ld, "LIGHTNING_STATE", state_path)
    monkeypatch.setattr(ld, "LIGHTNING_STATE_LOCK", state_path.with_suffix(".json.lock"))

    state_path.write_text(
        '[{"session_id": "s1", "label": "a"}, {"session_id": "s2", "label": "b"}]'
    )

    def _augment(row):
        row["augmented"] = True
        return row

    updated = ld.LightningDispatcher.update_session_locked("s1", _augment)
    assert updated is True

    import json
    out = json.loads(state_path.read_text())
    s1 = next(r for r in out if r["session_id"] == "s1")
    s2 = next(r for r in out if r["session_id"] == "s2")
    assert s1["augmented"] is True
    assert "augmented" not in s2  # untouched


def test_update_session_locked_returns_false_for_missing(tmp_path, monkeypatch):
    """update_session_locked returns False if the session_id isn't found."""
    from tac.deploy.lightning import lightning_dispatch as ld

    state_path = tmp_path / "lightning_state.json"
    monkeypatch.setattr(ld, "LIGHTNING_STATE", state_path)
    monkeypatch.setattr(ld, "LIGHTNING_STATE_LOCK", state_path.with_suffix(".json.lock"))

    state_path.write_text('[]')

    def _noop(row):
        return row

    updated = ld.LightningDispatcher.update_session_locked("missing", _noop)
    assert updated is False


def test_update_sessions_locked_replaces_full_list(tmp_path, monkeypatch):
    """update_sessions_locked supports full-list replacement under the lock."""
    from tac.deploy.lightning import lightning_dispatch as ld

    state_path = tmp_path / "lightning_state.json"
    monkeypatch.setattr(ld, "LIGHTNING_STATE", state_path)
    monkeypatch.setattr(ld, "LIGHTNING_STATE_LOCK", state_path.with_suffix(".json.lock"))

    state_path.write_text('[{"session_id": "s1"}]')

    def _replace(rows):
        return [{"session_id": "s2"}, {"session_id": "s3"}]

    new = ld.LightningDispatcher.update_sessions_locked(_replace)
    assert {r["session_id"] for r in new} == {"s2", "s3"}


def test_lock_depth_re_entry_is_safe(tmp_path, monkeypatch):
    """Nested `with _lightning_state_lock():` is safe (depth counter handles it)."""
    from tac.deploy.lightning import lightning_dispatch as ld

    state_path = tmp_path / "lightning_state.json"
    monkeypatch.setattr(ld, "LIGHTNING_STATE", state_path)
    monkeypatch.setattr(ld, "LIGHTNING_STATE_LOCK", state_path.with_suffix(".json.lock"))

    with ld._lightning_state_lock():
        assert ld._lightning_state_lock_held()
        with ld._lightning_state_lock():
            assert ld._lightning_state_lock_held()
            ld.LightningDispatcher._save_state([{"session_id": "nested"}])
        assert ld._lightning_state_lock_held()
    assert not ld._lightning_state_lock_held()


def test_active_jobs_save_raises_outside_lock():
    """Sister fix: _save_active_jobs must raise outside _active_jobs_lock."""
    from tac.deploy.lightning.active_jobs_state import _save_active_jobs

    with pytest.raises(RuntimeError, match="WITHOUT holding|_active_jobs_lock|catalog #140"):
        _save_active_jobs([])


def test_active_vms_save_raises_outside_lock():
    """Sister fix: _save_active_vms_atomic must raise outside _active_vms_lock."""
    from tac.deploy.azure.active_vms_state import _save_active_vms_atomic

    with pytest.raises(RuntimeError, match="WITHOUT holding|_active_vms_lock|catalog #140"):
        _save_active_vms_atomic([])
