# SPDX-License-Identifier: MIT
"""Tests for Catalog #138 — state writers must use strict load on mutating path.

Defense-in-depth on codex round-3 MEDIUM 2 (2026-05-09): the round-3 fix patched
ONE writer (`active_jobs_state.py::update_active_jobs_locked` now uses
`load_active_jobs_strict` which raises `ActiveJobsCorruptError` on corrupt JSON).
This META gate refuses ANY future writer from silently resetting corrupt state.

Memory: feedback_production_hardening_polish_defense_in_depth_landed_20260509.md.
Cross-ref Catalog #131 (bare writes), #132 (deletion-merge), #133 (META-meta on
#131 exempt list), #135 (lost-update outside lock).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_state_writers_strict_load_for_mutating_path,
)


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "tools").mkdir(parents=True)
    return tmp_path


# ── Catches the canonical bug pattern ────────────────────────────────────


def test_138_catches_update_locked_with_silent_load(tmp_path):
    root = _make_repo(tmp_path)
    (root / "src" / "tac" / "bad.py").write_text(
        "import json\n"
        "import fcntl\n"
        "from pathlib import Path\n"
        'P = Path("x.json")\n'
        "def update_state_locked(data):\n"
        '    with open("l", "w") as fd:\n'
        "        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)\n"
        '        existing = json.loads(P.read_text() or "{}")\n'
        "        existing.update(data)\n"
        "        P.write_text(json.dumps(existing))\n"
    )
    v = check_state_writers_strict_load_for_mutating_path(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "bad.py" in x]
    assert len(matches) == 1
    assert "update_state_locked" in matches[0]


def test_138_catches_save_with_silent_load(tmp_path):
    root = _make_repo(tmp_path)
    (root / "src" / "tac" / "bad.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        'P = Path("x.json")\n'
        "def _save_state(data):\n"
        '    existing = json.loads(P.read_text() or "{}")\n'
        "        # silent fallback on corrupt\n"
        "    existing.update(data)\n"
        "    P.write_text(json.dumps(existing))\n"
    )
    v = check_state_writers_strict_load_for_mutating_path(
        repo_root=root, strict=False, verbose=False
    )
    assert any("bad.py" in x and "_save_state" in x for x in v)


def test_138_catches_register_with_silent_load(tmp_path):
    root = _make_repo(tmp_path)
    (root / "tools" / "bad.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        'P = Path("x.json")\n'
        "def register_terminal(data):\n"
        '    existing = json.loads(P.read_text() or "{}")\n'
        "    existing.update(data)\n"
        "    P.write_text(json.dumps(existing))\n"
    )
    v = check_state_writers_strict_load_for_mutating_path(
        repo_root=root, strict=False, verbose=False
    )
    assert any("bad.py" in x and "register_terminal" in x for x in v)


# ── Accept clean writers using strict load ───────────────────────────────


def test_138_accepts_strict_load_helper(tmp_path):
    root = _make_repo(tmp_path)
    (root / "src" / "tac" / "ok.py").write_text(
        "import json\n"
        "import fcntl\n"
        "from pathlib import Path\n"
        'P = Path("x.json")\n'
        "def load_active_jobs_strict():\n"
        "    return json.loads(P.read_text())\n"
        "def update_state_locked(data):\n"
        '    with open("l", "w") as fd:\n'
        "        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)\n"
        "        existing = load_active_jobs_strict()\n"
        "        existing.update(data)\n"
        "        P.write_text(json.dumps(existing))\n"
    )
    v = check_state_writers_strict_load_for_mutating_path(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok.py" not in x for x in v)


def test_138_accepts_explicit_corrupt_raise(tmp_path):
    root = _make_repo(tmp_path)
    (root / "src" / "tac" / "ok.py").write_text(
        "import json\n"
        "import fcntl\n"
        "from pathlib import Path\n"
        'P = Path("x.json")\n'
        "class ActiveJobsCorruptError(Exception):\n"
        "    pass\n"
        "def update_state_locked(data):\n"
        '    with open("l", "w") as fd:\n'
        "        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)\n"
        "        try:\n"
        "            existing = json.loads(P.read_text())\n"
        "        except json.JSONDecodeError:\n"
        "            raise ActiveJobsCorruptError\n"
        "        existing.update(data)\n"
        "        P.write_text(json.dumps(existing))\n"
    )
    v = check_state_writers_strict_load_for_mutating_path(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok.py" not in x for x in v)


# ── Out-of-scope patterns ────────────────────────────────────────────────


def test_138_ignores_pure_write_function(tmp_path):
    """A function that only writes (no load) is out-of-scope; #131 covers
    bare unlocked writes."""
    root = _make_repo(tmp_path)
    (root / "tools" / "ok.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        'P = Path("x.json")\n'
        "def update_state_locked(data):\n"
        "    P.write_text(json.dumps(data))\n"
    )
    v = check_state_writers_strict_load_for_mutating_path(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok.py" not in x for x in v)


def test_138_ignores_pure_load_function(tmp_path):
    """A function that only loads (no write) is out-of-scope."""
    root = _make_repo(tmp_path)
    (root / "tools" / "ok.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        'P = Path("x.json")\n'
        "def update_state_locked():\n"
        "    return json.loads(P.read_text())\n"
    )
    v = check_state_writers_strict_load_for_mutating_path(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok.py" not in x for x in v)


def test_138_ignores_non_writer_function_name(tmp_path):
    """Functions whose name doesn't match the writer pattern are out-of-scope."""
    root = _make_repo(tmp_path)
    (root / "tools" / "ok.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        'P = Path("x.json")\n'
        "def merge_state(data):\n"
        '    existing = json.loads(P.read_text() or "{}")\n'
        "    existing.update(data)\n"
        "    P.write_text(json.dumps(existing))\n"
    )
    v = check_state_writers_strict_load_for_mutating_path(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok.py" not in x for x in v)


# ── Waiver works ─────────────────────────────────────────────────────────


def test_138_def_line_waiver_accepts(tmp_path):
    """`# STATE_WRITER_STRICT_LOAD_OK:` waiver on the def line accepts
    a writer that genuinely tolerates corrupt loads (e.g. counter-bumps)."""
    root = _make_repo(tmp_path)
    (root / "tools" / "waived.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        'P = Path("x.json")\n'
        "def update_counter_locked(data):  # STATE_WRITER_STRICT_LOAD_OK: counter-bump tolerates corrupt reset\n"
        '    existing = json.loads(P.read_text() or "{}")\n'
        "    existing.update(data)\n"
        "    P.write_text(json.dumps(existing))\n"
    )
    v = check_state_writers_strict_load_for_mutating_path(
        repo_root=root, strict=False, verbose=False
    )
    assert all("waived.py" not in x for x in v)


# ── Strict-mode round-trip ───────────────────────────────────────────────


def test_138_strict_raises(tmp_path):
    root = _make_repo(tmp_path)
    (root / "src" / "tac" / "bad.py").write_text(
        "import json\n"
        "import fcntl\n"
        "from pathlib import Path\n"
        'P = Path("x.json")\n'
        "def update_state_locked(data):\n"
        '    with open("l", "w") as fd:\n'
        "        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)\n"
        '        existing = json.loads(P.read_text() or "{}")\n'
        "        existing.update(data)\n"
        "        P.write_text(json.dumps(existing))\n"
    )
    with pytest.raises(
        PreflightError, match="check_state_writers_strict_load_for_mutating_path"
    ):
        check_state_writers_strict_load_for_mutating_path(
            repo_root=root, strict=True, verbose=False
        )


def test_138_test_files_excluded(tmp_path):
    """Test files MUST be excluded — they legitimately mock the bug pattern."""
    root = _make_repo(tmp_path)
    (root / "tools" / "test_foo.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        'P = Path("x.json")\n'
        "def update_state_locked(data):\n"
        '    existing = json.loads(P.read_text() or "{}")\n'
        "    existing.update(data)\n"
        "    P.write_text(json.dumps(existing))\n"
    )
    v = check_state_writers_strict_load_for_mutating_path(
        repo_root=root, strict=False, verbose=False
    )
    assert all("test_foo.py" not in x for x in v)


# ── Live-repo sanity ─────────────────────────────────────────────────────


def test_138_live_repo_clean():
    """Live-repo sanity: catalog #138 must land at 0 violations.

    Round-3's fix already added the strict load path to the only known
    instance (`active_jobs_state.py::update_active_jobs_locked`).
    """
    v = check_state_writers_strict_load_for_mutating_path(
        strict=False, verbose=False
    )
    assert v == [], (
        f"Catalog #138 landed with {len(v)} violations:\n"
        + "\n".join(v[:3])
    )
