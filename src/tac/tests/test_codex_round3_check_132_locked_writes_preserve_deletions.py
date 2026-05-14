# SPDX-License-Identifier: MIT
"""Tests for codex round-3 catalog #132 — locked writes must preserve caller deletions.

Bug class (codex round-3 HIGH 1, 2026-05-09): a locked-save helper that
takes the caller's full post-prune map but does ``existing.update(data)``
inside the lock silently re-introduces rows the caller deliberately
pruned. Sister of catalog #131 (bare writes — unlocked META class) and
catalog #128 (continual_learning save_posterior).

The fix gate (#132) refuses any function whose name matches a
"locked save" pattern (``_save_*_first_seen`` / ``_save_*_state`` /
``update_*_locked``) that contains a deletion-merging pattern inside an
fcntl-locked region.

Memory: feedback_codex_round3_findings_fix_landed_20260509.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_locked_writes_preserve_deletions,
)
from tac.source_index import source_index_context


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "tools").mkdir(parents=True)
    (tmp_path / "experiments").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    return tmp_path


# ── Catch the canonical bug pattern ──────────────────────────────────────


def test_132_catches_locked_save_first_seen_existing_update(tmp_path):
    """The canonical HIGH 1 anti-pattern: locked save that does
    `existing.update(data)` instead of transactional replace.
    """
    root = _make_repo(tmp_path)
    (root / "scripts" / "bad_save.py").write_text(
        "import fcntl\n"
        "from pathlib import Path\n"
        "STATE_PATH = Path('/tmp/x.json')\n"
        "import json\n"
        "def _load_setup_first_seen():\n"
        "    return {}\n"
        "def _save_setup_first_seen(data):\n"
        "    with open('/tmp/lock', 'w') as fd:\n"
        "        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)\n"
        "        existing = _load_setup_first_seen()\n"
        "        existing.update(data)  # BUG: deletion-merge\n"
        "        STATE_PATH.write_text(json.dumps(existing))\n"
    )
    v = check_locked_writes_preserve_deletions(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "bad_save.py" in x]
    assert len(matches) == 1, (
        f"#132: expected to catch the canonical bug pattern; got {v}"
    )
    assert "_save_setup_first_seen" in matches[0]
    assert "deletion-MERGE" in matches[0] or "deletion-merge" in matches[0].lower()


def test_132_catches_loaded_update_variant(tmp_path):
    """Variant naming: `loaded.update(data)` is the same anti-pattern."""
    root = _make_repo(tmp_path)
    (root / "tools" / "bad_loaded.py").write_text(
        "import fcntl\n"
        "from pathlib import Path\n"
        "P = Path('/tmp/p.json')\n"
        "import json\n"
        "def _load_state():\n"
        "    return {}\n"
        "def _save_state(data):\n"
        "    with open('/tmp/lock', 'w') as fd:\n"
        "        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)\n"
        "        loaded = _load_state()\n"
        "        loaded.update(data)\n"
        "        P.write_text(json.dumps(loaded))\n"
    )
    v = check_locked_writes_preserve_deletions(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "bad_loaded.py" in x]
    assert len(matches) == 1


# ── Accept clean transactional-replace patterns ──────────────────────────


def test_132_accepts_transactional_replace(tmp_path):
    """Direct write of `data` inside the lock is the safe pattern."""
    root = _make_repo(tmp_path)
    (root / "scripts" / "ok_save.py").write_text(
        "import fcntl\n"
        "from pathlib import Path\n"
        "STATE_PATH = Path('/tmp/x.json')\n"
        "import json\n"
        "def _save_setup_first_seen(data):\n"
        "    with open('/tmp/lock', 'w') as fd:\n"
        "        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)\n"
        "        STATE_PATH.write_text(json.dumps(data))  # transactional replace\n"
    )
    v = check_locked_writes_preserve_deletions(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok_save.py" not in x for x in v)


def test_132_accepts_callback_mutate_pattern(tmp_path):
    """The `update_*_locked(mutate_fn)` pattern (active_jobs_state) is the
    canonical safe form — no `existing.update(data)` inside the body.
    """
    root = _make_repo(tmp_path)
    (root / "tools" / "ok_mutate.py").write_text(
        "import fcntl\n"
        "from pathlib import Path\n"
        "P = Path('/tmp/p.json')\n"
        "import json\n"
        "def update_active_jobs_locked(mutate_fn):\n"
        "    with open('/tmp/lock', 'w') as fd:\n"
        "        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)\n"
        "        rows = json.loads(P.read_text() or '[]')\n"
        "        new_rows = mutate_fn(rows)\n"
        "        P.write_text(json.dumps(new_rows))\n"
    )
    v = check_locked_writes_preserve_deletions(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok_mutate.py" not in x for x in v)


# ── Out-of-scope patterns must NOT be flagged ────────────────────────────


def test_132_ignores_non_locked_save_function(tmp_path):
    """A `merge_options(existing, data)` helper that doesn't match
    `_save_*` / `update_*_locked` and isn't called inside fcntl is
    out-of-scope; the gate must NOT flag it."""
    root = _make_repo(tmp_path)
    (root / "tools" / "ok_misc.py").write_text(
        "def merge_options(existing, data):\n"
        "    existing.update(data)\n"
        "    return existing\n"
    )
    v = check_locked_writes_preserve_deletions(
        repo_root=root, strict=False, verbose=False
    )
    assert all("ok_misc.py" not in x for x in v)


def test_132_ignores_unlocked_save(tmp_path):
    """A function with no fcntl context is not the META class this gate
    targets; #131 covers unlocked writes separately."""
    root = _make_repo(tmp_path)
    (root / "tools" / "unlocked.py").write_text(
        "from pathlib import Path\n"
        "P = Path('/tmp/p.json')\n"
        "import json\n"
        "def _save_setup_first_seen(data):\n"
        "    existing = {}\n"
        "    existing.update(data)\n"
        "    P.write_text(json.dumps(existing))\n"
    )
    v = check_locked_writes_preserve_deletions(
        repo_root=root, strict=False, verbose=False
    )
    assert all("unlocked.py" not in x for x in v)


# ── Waiver works ─────────────────────────────────────────────────────────


def test_132_same_line_waiver_accepts(tmp_path):
    """`# DELETION_MERGE_OK:` waiver lets through genuinely additive
    counter-bumping helpers (rare; must justify per CLAUDE.md)."""
    root = _make_repo(tmp_path)
    (root / "tools" / "waived.py").write_text(
        "import fcntl\n"
        "from pathlib import Path\n"
        "P = Path('/tmp/p.json')\n"
        "import json\n"
        "def _save_state_counter_bump(data):\n"
        "    with open('/tmp/lock', 'w') as fd:\n"
        "        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)\n"
        "        existing = json.loads(P.read_text() or '{}')\n"
        "        existing.update(data)  # DELETION_MERGE_OK: counter-bump only; never replace\n"
        "        P.write_text(json.dumps(existing))\n"
    )
    v = check_locked_writes_preserve_deletions(
        repo_root=root, strict=False, verbose=False
    )
    assert all("waived.py" not in x for x in v)


# ── Strict-mode round-trip ───────────────────────────────────────────────


def test_132_strict_raises(tmp_path):
    root = _make_repo(tmp_path)
    (root / "scripts" / "bad.py").write_text(
        "import fcntl\n"
        "from pathlib import Path\n"
        "P = Path('/tmp/p.json')\n"
        "import json\n"
        "def _save_setup_first_seen(data):\n"
        "    with open('/tmp/lock', 'w') as fd:\n"
        "        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)\n"
        "        existing = {}\n"
        "        existing.update(data)\n"
        "        P.write_text(json.dumps(existing))\n"
    )
    with pytest.raises(PreflightError, match="check_locked_writes_preserve_deletions"):
        check_locked_writes_preserve_deletions(
            repo_root=root, strict=True, verbose=False
        )


def test_132_test_files_excluded(tmp_path):
    """Test files (which legitimately mock the bug pattern) MUST be excluded."""
    root = _make_repo(tmp_path)
    (root / "tools" / "test_foo.py").write_text(
        "import fcntl\n"
        "def _save_state(data):\n"
        "    with open('/tmp/x', 'w') as fd:\n"
        "        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)\n"
        "        existing = {}\n"
        "        existing.update(data)\n"
    )
    v = check_locked_writes_preserve_deletions(
        repo_root=root, strict=False, verbose=False
    )
    assert all("test_foo.py" not in x for x in v)


# ── Live-repo sanity ─────────────────────────────────────────────────────


def test_132_live_repo_clean():
    """Live-repo sanity: catalog #132 must land at 0 violations.

    The verify_vast_instances HIGH 1 fix closed the only known instance.
    Any future regression (re-introducing `existing.update(data)` inside
    a locked _save_*_state helper) will fail this test loud-and-early.
    """
    v = check_locked_writes_preserve_deletions(strict=False, verbose=False)
    assert v == [], (
        f"Catalog #132 landed with {len(v)} violations:\n"
        + "\n".join(v[:3])
    )


def test_132_uses_source_index_substring_prefilter(tmp_path):
    root = _make_repo(tmp_path)
    (root / "tools" / "irrelevant.py").write_text(
        "import fcntl\n"
        "def unrelated(data):\n"
        "    return data\n"
    )
    (root / "scripts" / "bad_save.py").write_text(
        "import fcntl\n"
        "def _save_state(data):\n"
        "    with open('/tmp/lock', 'w') as fd:\n"
        "        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)\n"
        "        existing = {}\n"
        "        existing.update(data)\n"
    )

    with source_index_context(root) as index:
        v = check_locked_writes_preserve_deletions(
            repo_root=root, strict=False, verbose=False
        )
        stats = index.stats()

    assert any("bad_save.py" in item for item in v)
    assert stats["substring_index_entries"] >= 1
