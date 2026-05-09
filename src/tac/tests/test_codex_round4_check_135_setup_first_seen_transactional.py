"""Tests for codex round-4 catalog #135 — SETUP-first-seen lost-update fix.

Bug class (codex round-4 MEDIUM 2, 2026-05-09): the round-3 fix made
``_save_setup_first_seen`` write transactionally INSIDE the lock, but
the ``main()`` flow still loaded the on-disk state OUTSIDE the lock,
ran per-instance verify (~minutes), then saved at the end. Two
overlapping verifier runs both loaded the same stale snapshot, did
per-instance work, then the slower run replaced the file with its
now-stale view — deleting first-seen timestamps the faster run had
created. SETUP age would silently reset for affected instances.

The fix added ``update_setup_first_seen_locked`` (load + merge + prune
+ save inside ONE fcntl-locked window) and refactored ``main()`` to
use it. Catalog #135 refuses any module that has the lost-update
anti-pattern (``_load_*_first_seen`` outside lock + later
``_save_*_first_seen`` under lock).

Memory: feedback_codex_round4_findings_fix_with_self_protection_landed_20260509.md.
"""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_setup_first_seen_uses_transactional_update_inside_lock,
)


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "tools").mkdir(parents=True)
    (tmp_path / "experiments").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    return tmp_path


# ── Live-repo sanity ────────────────────────────────────────────────────


def test_135_live_repo_clean():
    """Live-repo: catalog #135 must land at 0 violations after fix."""
    v = check_setup_first_seen_uses_transactional_update_inside_lock(
        strict=False, verbose=False
    )
    assert v == [], (
        f"Catalog #135 landed with {len(v)} violations:\n"
        + "\n".join(v[:3])
    )


# ── Catch the lost-update anti-pattern ─────────────────────────────────


def test_135_catches_load_outside_lock_save_under_lock(tmp_path):
    """The canonical anti-pattern: _load outside any lock + later _save
    under lock = lost-update race for two overlapping invocations."""
    root = _make_repo(tmp_path)
    (root / "scripts" / "bad_main.py").write_text(
        "import json\n"
        "import fcntl\n"
        "from pathlib import Path\n"
        "P = Path('/tmp/sf.json')\n"
        "def _load_setup_first_seen():\n"
        "    return json.loads(P.read_text())\n"
        "def _save_setup_first_seen(data):\n"
        "    with open('/tmp/lock', 'w') as f:\n"
        "        fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
        "        P.write_text(json.dumps(data))\n"
        "def main():\n"
        "    state = _load_setup_first_seen()  # outside lock!\n"
        "    # ... long work ...\n"
        "    state['x'] = 1\n"
        "    _save_setup_first_seen(state)\n"
    )
    v = check_setup_first_seen_uses_transactional_update_inside_lock(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "bad_main.py" in x]
    assert len(matches) == 1, (
        f"Expected to catch lost-update pattern; got {v}"
    )


# ── Accept clean transactional patterns ────────────────────────────────


def test_135_accepts_canonical_transactional_helper(tmp_path):
    """A function that uses `update_*_first_seen_locked` is the canonical
    safe pattern — must pass even though the module also defines _load and
    _save helpers (the load+save pair is for the helper's own internal use).
    """
    root = _make_repo(tmp_path)
    (root / "scripts" / "good_main.py").write_text(
        "import json\n"
        "import fcntl\n"
        "from pathlib import Path\n"
        "P = Path('/tmp/sf.json')\n"
        "def _load_setup_first_seen():\n"
        "    return json.loads(P.read_text() or '{}')\n"
        "def _save_setup_first_seen(data):\n"
        "    with open('/tmp/lock', 'w') as f:\n"
        "        fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
        "        P.write_text(json.dumps(data))\n"
        "def update_setup_first_seen_locked(observed, tracked, now_ts):\n"
        "    with open('/tmp/lock', 'w') as f:\n"
        "        fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
        "        # full transactional update inside the lock\n"
        "        pass\n"
        "def main():\n"
        "    update_setup_first_seen_locked(set(), set(), 0)\n"
    )
    v = check_setup_first_seen_uses_transactional_update_inside_lock(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "good_main.py" in x]
    assert len(matches) == 0


def test_135_accepts_load_under_lock(tmp_path):
    """A load that's INSIDE a lock is not the lost-update class."""
    root = _make_repo(tmp_path)
    (root / "scripts" / "ok_locked.py").write_text(
        "import json\n"
        "import fcntl\n"
        "from pathlib import Path\n"
        "P = Path('/tmp/sf.json')\n"
        "def _load_setup_first_seen():\n"
        "    return json.loads(P.read_text() or '{}')\n"
        "def _save_setup_first_seen(data):\n"
        "    with open('/tmp/lock', 'w') as f:\n"
        "        fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
        "        P.write_text(json.dumps(data))\n"
        "def main():\n"
        "    with open('/tmp/lock', 'w') as f:\n"
        "        fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
        "        state = _load_setup_first_seen()\n"
        "        state['x'] = 1\n"
        "        _save_setup_first_seen(state)\n"
    )
    v = check_setup_first_seen_uses_transactional_update_inside_lock(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "ok_locked.py" in x]
    assert len(matches) == 0


def test_135_ignores_read_only_function(tmp_path):
    """A function that loads but does NOT save is read-only — out-of-scope."""
    root = _make_repo(tmp_path)
    (root / "tools" / "report.py").write_text(
        "import json\n"
        "import fcntl\n"
        "from pathlib import Path\n"
        "P = Path('/tmp/sf.json')\n"
        "def _load_setup_first_seen():\n"
        "    return json.loads(P.read_text() or '{}')\n"
        "def _save_setup_first_seen(data):\n"
        "    with open('/tmp/lock', 'w') as f:\n"
        "        fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
        "        P.write_text(json.dumps(data))\n"
        "def report():\n"
        "    state = _load_setup_first_seen()  # read-only consumer\n"
        "    print(len(state))\n"
    )
    v = check_setup_first_seen_uses_transactional_update_inside_lock(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "report.py" in x]
    assert len(matches) == 0


# ── Waiver works ───────────────────────────────────────────────────────


def test_135_same_line_waiver_accepts(tmp_path):
    """`# SETUP_FIRST_SEEN_LOST_UPDATE_OK:<reason>` lets through rare
    intentional load-outside-lock paths."""
    root = _make_repo(tmp_path)
    (root / "scripts" / "waived.py").write_text(
        "import json\n"
        "import fcntl\n"
        "from pathlib import Path\n"
        "P = Path('/tmp/sf.json')\n"
        "def _load_setup_first_seen():\n"
        "    return json.loads(P.read_text() or '{}')\n"
        "def _save_setup_first_seen(data):\n"
        "    with open('/tmp/lock', 'w') as f:\n"
        "        fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
        "        P.write_text(json.dumps(data))\n"
        "def main():\n"
        "    state = _load_setup_first_seen()  # SETUP_FIRST_SEEN_LOST_UPDATE_OK: dry-run reporter\n"
        "    state['x'] = 1\n"
        "    _save_setup_first_seen(state)\n"
    )
    v = check_setup_first_seen_uses_transactional_update_inside_lock(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "waived.py" in x]
    assert len(matches) == 0


# ── Strict-mode round-trip ─────────────────────────────────────────────


def test_135_strict_raises(tmp_path):
    root = _make_repo(tmp_path)
    (root / "scripts" / "evil.py").write_text(
        "import json\n"
        "import fcntl\n"
        "from pathlib import Path\n"
        "P = Path('/tmp/sf.json')\n"
        "def _load_setup_first_seen():\n"
        "    return json.loads(P.read_text() or '{}')\n"
        "def _save_setup_first_seen(data):\n"
        "    with open('/tmp/lock', 'w') as f:\n"
        "        fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
        "        P.write_text(json.dumps(data))\n"
        "def main():\n"
        "    state = _load_setup_first_seen()\n"
        "    state['x'] = 1\n"
        "    _save_setup_first_seen(state)\n"
    )
    with pytest.raises(
        PreflightError,
        match="check_setup_first_seen_uses_transactional_update_inside_lock",
    ):
        check_setup_first_seen_uses_transactional_update_inside_lock(
            repo_root=root, strict=True, verbose=False
        )


def test_135_test_files_excluded(tmp_path):
    """Test files (which legitimately mock the bug pattern) MUST be excluded."""
    root = _make_repo(tmp_path)
    (root / "tools" / "test_xyz.py").write_text(
        "import json\n"
        "import fcntl\n"
        "from pathlib import Path\n"
        "def _load_setup_first_seen():\n"
        "    return {}\n"
        "def _save_setup_first_seen(data):\n"
        "    with open('/tmp/lock', 'w') as f:\n"
        "        fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
        "        pass\n"
        "def test_x():\n"
        "    state = _load_setup_first_seen()\n"
        "    _save_setup_first_seen(state)\n"
    )
    v = check_setup_first_seen_uses_transactional_update_inside_lock(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "test_xyz.py" in x]
    assert len(matches) == 0


# ── Real verify_vast_instances regression test (the underlying bug) ────


def _worker_update_setup(args):
    state_path, lock_path, observed, tracked, now_ts = args
    # Simulate the verify_vast_instances main() flow but with explicit
    # paths so we can run in tmp_path
    import sys
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[3] / "scripts")
    )
    # Monkey-patch the canonical paths
    import verify_vast_instances as vvi
    vvi.SETUP_FIRST_SEEN_PATH = state_path
    vvi._setup_first_seen_lock_path = lambda: lock_path
    vvi.update_setup_first_seen_locked(
        observed_setup_ids=set(observed),
        tracked_ids=set(tracked),
        now_ts=now_ts,
    )


def test_135_concurrent_verifier_runs_preserve_first_seen(tmp_path):
    """Regression test for the underlying lost-update bug.

    Spin up two parallel processes that simulate two overlapping verifier
    runs. Each observes the SAME instance i1 in SETUP at slightly
    different timestamps. The ``update_setup_first_seen_locked`` helper's
    KEEP-OLDER merge semantics MUST converge to the EARLIEST timestamp
    regardless of which process commits last.
    """
    state_path = tmp_path / "sf.json"
    lock_path = tmp_path / "sf.json.lock"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    args_list = [
        (state_path, lock_path, ["i1", "i2"], ["i1", "i2", "i3"], 1000.0),
        (state_path, lock_path, ["i1", "i3"], ["i1", "i2", "i3"], 2000.0),
        (state_path, lock_path, ["i2", "i3"], ["i1", "i2", "i3"], 3000.0),
    ]

    procs = [
        multiprocessing.Process(target=_worker_update_setup, args=(a,))
        for a in args_list
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0, f"process failed exitcode={p.exitcode}"

    rows = json.loads(state_path.read_text())
    # Each id should be present (since all 3 are tracked) and timestamp
    # should be the minimum across observations
    assert "i1" in rows
    assert "i2" in rows
    assert "i3" in rows
    # i1 was observed at 1000 and 2000 — KEEP-OLDER => 1000
    assert rows["i1"] == 1000.0, f"i1 first-seen should be 1000, got {rows['i1']}"
    # i2 was observed at 1000 and 3000 — KEEP-OLDER => 1000
    assert rows["i2"] == 1000.0
    # i3 was observed at 2000 and 3000 — KEEP-OLDER => 2000
    assert rows["i3"] == 2000.0


def test_135_remove_setup_first_seen_locked_drops_left_setup_ids(tmp_path):
    """The remove helper drops the named ids transactionally."""
    import sys
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[3] / "scripts")
    )
    import verify_vast_instances as vvi
    state_path = tmp_path / "sf.json"
    lock_path = tmp_path / "sf.json.lock"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    vvi.SETUP_FIRST_SEEN_PATH = state_path
    vvi._setup_first_seen_lock_path = lambda: lock_path

    # Seed the file
    state_path.write_text(
        json.dumps({"i1": 100.0, "i2": 200.0, "i3": 300.0})
    )
    # Remove i2
    result = vvi.remove_setup_first_seen_locked({"i2"})
    assert "i2" not in result
    assert "i1" in result and "i3" in result
    on_disk = json.loads(state_path.read_text())
    assert "i2" not in on_disk
    assert "i1" in on_disk and "i3" in on_disk


def test_135_update_setup_first_seen_locked_prunes_ids_no_longer_tracked(tmp_path):
    """An id present in the on-disk state but NOT in `tracked_ids` is pruned."""
    import sys
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[3] / "scripts")
    )
    import verify_vast_instances as vvi
    state_path = tmp_path / "sf.json"
    lock_path = tmp_path / "sf.json.lock"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    vvi.SETUP_FIRST_SEEN_PATH = state_path
    vvi._setup_first_seen_lock_path = lambda: lock_path

    # Seed with id that's no longer tracked
    state_path.write_text(json.dumps({"i_old": 100.0, "i_keep": 200.0}))
    result = vvi.update_setup_first_seen_locked(
        observed_setup_ids={"i_keep"},
        tracked_ids={"i_keep", "i_new"},
        now_ts=300.0,
    )
    assert "i_old" not in result
    assert "i_keep" in result
    # i_keep had on-disk 200, observed at 300 → KEEP-OLDER = 200
    assert result["i_keep"] == 200.0
