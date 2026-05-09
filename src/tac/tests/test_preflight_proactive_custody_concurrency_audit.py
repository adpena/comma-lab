"""Tests for catalog #130 + #131 — proactive META-class audit preflight gates.

Memory: ``feedback_proactive_custody_concurrency_audit_landed_20260509.md``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_no_bare_writes_to_shared_state,
    check_no_tag_only_custody_validation,
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _make_repo(tmp_path: Path) -> Path:
    """Build a minimal repo skeleton matching the gates' expected layout."""
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "tools").mkdir(parents=True)
    (tmp_path / "experiments").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    return tmp_path


# ──────────────────────────────────────────────────────────────────────────
# Catalog #130 — tag/grade membership without joint validator
# ──────────────────────────────────────────────────────────────────────────


def test_130_clean_repo_passes(tmp_path):
    root = _make_repo(tmp_path)
    v = check_no_tag_only_custody_validation(repo_root=root, strict=False, verbose=False)
    assert v == []


def test_130_evidence_grade_set_membership_caught(tmp_path):
    root = _make_repo(tmp_path)
    # The gate detects the canonical pattern `evidence_grade in {` (bare form).
    (root / "experiments" / "bad.py").write_text(
        "def foo(evidence_grade):\n"
        "    if evidence_grade in {'A', 'A++'}:\n"
        "        return True\n"
        "    return False\n"
    )
    v = check_no_tag_only_custody_validation(repo_root=root, strict=False, verbose=False)
    assert len(v) == 1
    assert "bad.py" in v[0]


def test_130_evidence_grade_lower_set_membership_caught(tmp_path):
    root = _make_repo(tmp_path)
    # The gate detects the canonical literal `evidence_grade.lower() in {`.
    (root / "tools" / "bad.py").write_text(
        "def foo(evidence_grade):\n"
        "    return evidence_grade.lower() in {'a', 'a++'}\n"
    )
    v = check_no_tag_only_custody_validation(repo_root=root, strict=False, verbose=False)
    file_paths = [s for s in v if "bad.py" in s]
    assert len(file_paths) == 1


def test_130_validate_custody_in_file_accepts(tmp_path):
    root = _make_repo(tmp_path)
    (root / "tools" / "ok.py").write_text(
        "def foo(evidence_grade, row):\n"
        "    if evidence_grade in {'A', 'A++'}:\n"
        "        validate_custody(row)\n"
        "    return False\n"
    )
    v = check_no_tag_only_custody_validation(repo_root=root, strict=False, verbose=False)
    assert all("ok.py" not in x for x in v)


def test_130_archive_sha256_blockers_in_file_accepts(tmp_path):
    root = _make_repo(tmp_path)
    (root / "tools" / "ok.py").write_text(
        "def foo(evidence_grade, row):\n"
        "    blockers = []\n"
        "    if row.get('archive_sha256') is None:\n"
        "        blockers.append('missing_sha')\n"
        "    if evidence_grade in {'A', 'A++'}:\n"
        "        return blockers\n"
        "    return None\n"
    )
    v = check_no_tag_only_custody_validation(repo_root=root, strict=False, verbose=False)
    assert all("ok.py" not in x for x in v)


def test_130_same_line_waiver_accepts(tmp_path):
    root = _make_repo(tmp_path)
    (root / "tools" / "ok.py").write_text(
        "def foo(evidence_grade):\n"
        "    if evidence_grade in {'A', 'A++'}:  # CUSTODY_VALIDATOR_OK: read-only filter\n"
        "        return True\n"
    )
    v = check_no_tag_only_custody_validation(repo_root=root, strict=False, verbose=False)
    assert all("ok.py" not in x for x in v)


def test_130_strict_raises(tmp_path):
    root = _make_repo(tmp_path)
    (root / "tools" / "bad.py").write_text(
        "def foo(evidence_grade):\n"
        "    if evidence_grade in {'A', 'A++'}:\n"
        "        return True\n"
    )
    with pytest.raises(PreflightError, match="check_no_tag_only_custody_validation"):
        check_no_tag_only_custody_validation(repo_root=root, strict=True, verbose=False)


def test_130_test_files_excluded(tmp_path):
    root = _make_repo(tmp_path)
    (root / "tools" / "test_foo.py").write_text(
        "def test_foo():\n"
        "    if 'evidence_grade in {' in 'pattern detection':\n"
        "        return True\n"
    )
    v = check_no_tag_only_custody_validation(repo_root=root, strict=False, verbose=False)
    assert all("test_foo.py" not in x for x in v)


def test_130_live_repo_clean():
    """Live-repo sanity: catalog #130 must land at 0 violations."""
    v = check_no_tag_only_custody_validation(strict=False, verbose=False)
    assert v == [], f"Catalog #130 landed with {len(v)} violations:\n" + "\n".join(v[:3])


# ──────────────────────────────────────────────────────────────────────────
# Catalog #131 — bare-writes-on-shared-state
# ──────────────────────────────────────────────────────────────────────────


def test_131_clean_repo_passes(tmp_path):
    root = _make_repo(tmp_path)
    v = check_no_bare_writes_to_shared_state(repo_root=root, strict=False, verbose=False)
    assert v == []


def test_131_bare_write_to_omx_state_caught(tmp_path):
    root = _make_repo(tmp_path)
    (root / "experiments" / "bad.py").write_text(
        "from pathlib import Path\n"
        "STATE_PATH = Path('.omx/state/bad_lane.json')\n"
        "import json\n"
        "def save(rows):\n"
        "    STATE_PATH.write_text(json.dumps(rows, indent=2))\n"
    )
    v = check_no_bare_writes_to_shared_state(repo_root=root, strict=False, verbose=False)
    assert len(v) == 1
    assert "bad.py" in v[0]


def test_131_locked_path_in_file_accepts(tmp_path):
    """Codex round 8 MEDIUM (in-place harden of #131): lock alone is no
    longer sufficient to waive direct write_text on a shared-state path —
    the canonical helper invocation OR explicit transactional pattern
    (write to ``<path>.tmp`` + ``os.replace``) is required. The example
    below now PASSES with the canonical transactional pattern.
    """
    root = _make_repo(tmp_path)
    (root / "experiments" / "ok.py").write_text(
        "import fcntl\n"
        "import os\n"
        "from pathlib import Path\n"
        "STATE_PATH = Path('.omx/state/ok_lane.json')\n"
        "import json\n"
        "def save(rows):\n"
        "    with open(STATE_PATH.with_suffix('.lock'), 'w') as fd:\n"
        "        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)\n"
        "        tmp = STATE_PATH.with_suffix('.tmp')\n"
        "        tmp.write_text(json.dumps(rows, indent=2))\n"
        "        os.replace(tmp, STATE_PATH)\n"
    )
    v = check_no_bare_writes_to_shared_state(repo_root=root, strict=False, verbose=False)
    assert all("ok.py" not in x for x in v), v


def test_131_canonical_helper_accepts(tmp_path):
    root = _make_repo(tmp_path)
    (root / "experiments" / "ok.py").write_text(
        "from pathlib import Path\n"
        "STATE_PATH = Path('.omx/state/ok_lane.json')\n"
        "from tac.deploy.lightning.active_jobs_state import register_job\n"
        "def save(rec):\n"
        "    register_job(rec)\n"
    )
    v = check_no_bare_writes_to_shared_state(repo_root=root, strict=False, verbose=False)
    # The file references a state path AND uses register_job (lock token);
    # but it has no actual write call, so no violation.
    assert all("ok.py" not in x for x in v)


def test_131_register_instance_nearby_does_not_waive_bare_write(tmp_path):
    """A canonical helper call AFTER a bare write does not retroactively
    waive the write — the helper-call lookback only exempts writes that
    happen WITHIN the helper's transactional contract, not writes that
    happen first then call the helper later.

    Codex round 8 MEDIUM (in-place harden): the helper-call exemption now
    requires the helper call to appear in the lookback BEFORE the write.
    This test pins that ordering. Use a non-canonical helper name to
    exercise the bug-class without colliding with the canonical-helper
    accept list.
    """
    root = _make_repo(tmp_path)
    (root / "scripts" / "bad.py").write_text(
        "from pathlib import Path\n"
        "TRACKER_PATH = Path('.omx/state/vastai_active_instances.json')\n"
        "import json\n"
        "def some_unrelated_helper():\n"
        "    pass\n"
        "def write_helper():\n"
        "    TRACKER_PATH.write_text(json.dumps([]))\n"
        "    some_unrelated_helper()\n"
    )
    v = check_no_bare_writes_to_shared_state(repo_root=root, strict=False, verbose=False)
    assert len(v) == 1, v
    assert "bad.py" in v[0]


def test_131_same_line_waiver_accepts(tmp_path):
    root = _make_repo(tmp_path)
    (root / "experiments" / "ok.py").write_text(
        "from pathlib import Path\n"
        "STATE_PATH = Path('.omx/state/ok_lane.json')\n"
        "import json\n"
        "def save(rows):\n"
        "    STATE_PATH.write_text(json.dumps(rows))  # BARE_WRITE_OK: single-writer cron job\n"
    )
    v = check_no_bare_writes_to_shared_state(repo_root=root, strict=False, verbose=False)
    assert all("ok.py" not in x for x in v)


def test_131_strict_raises(tmp_path):
    root = _make_repo(tmp_path)
    (root / "experiments" / "bad.py").write_text(
        "from pathlib import Path\n"
        "STATE_PATH = Path('.omx/state/bad_lane.json')\n"
        "import json\n"
        "def save(rows):\n"
        "    STATE_PATH.write_text(json.dumps(rows))\n"
    )
    with pytest.raises(PreflightError, match="check_no_bare_writes_to_shared_state"):
        check_no_bare_writes_to_shared_state(repo_root=root, strict=True, verbose=False)


def test_131_test_files_excluded(tmp_path):
    root = _make_repo(tmp_path)
    (root / "tools" / "test_foo.py").write_text(
        "from pathlib import Path\n"
        "STATE_PATH = Path('.omx/state/foo.json')\n"
        "import json\n"
        "def test_save():\n"
        "    STATE_PATH.write_text(json.dumps([]))\n"
    )
    v = check_no_bare_writes_to_shared_state(repo_root=root, strict=False, verbose=False)
    assert all("test_foo.py" not in x for x in v)


def test_131_window_lock_token_accepts(tmp_path):
    """Codex round 8 MEDIUM: lock-token alone in the window is no longer
    sufficient to waive direct write_text on a shared-state path. The
    canonical pattern requires lock + atomic-replace (`<path>.tmp` +
    `os.replace`) for partial-read safety.
    """
    root = _make_repo(tmp_path)
    (root / "experiments" / "ok.py").write_text(
        "import fcntl\n"
        "import os\n"
        "from pathlib import Path\n"
        "STATE_PATH = Path('.omx/state/ok_lane.json')\n"
        "import json\n"
        "def save(rows):\n"
        "    with open(STATE_PATH.with_suffix('.lock'), 'w') as fd:\n"
        "        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)\n"
        "        tmp = STATE_PATH.with_suffix('.tmp')\n"
        "        tmp.write_text(json.dumps(rows))\n"
        "        os.replace(tmp, STATE_PATH)\n"
    )
    v = check_no_bare_writes_to_shared_state(repo_root=root, strict=False, verbose=False)
    assert all("ok.py" not in x for x in v), v


def test_131_lock_alone_no_atomic_replace_now_violation(tmp_path):
    """Codex round 8 MEDIUM (NEW behavior): pin the new violation case.

    Lock present but NO atomic-replace pattern → violation. This was a
    silent false-green before the harden landed.
    """
    root = _make_repo(tmp_path)
    (root / "experiments" / "lock_no_replace.py").write_text(
        "import fcntl\n"
        "from pathlib import Path\n"
        "STATE_PATH = Path('.omx/state/lock_only.json')\n"
        "import json\n"
        "def save(rows):\n"
        "    with open(STATE_PATH.with_suffix('.lock'), 'w') as fd:\n"
        "        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)\n"
        "        STATE_PATH.write_text(json.dumps(rows))\n"
    )
    v = check_no_bare_writes_to_shared_state(repo_root=root, strict=False, verbose=False)
    assert len(v) == 1, v
    assert "lock_no_replace.py" in v[0]
    assert "atomic" in v[0].lower()


def test_131_live_repo_clean():
    """Live-repo sanity: catalog #131 must land at 0 violations."""
    v = check_no_bare_writes_to_shared_state(strict=False, verbose=False)
    assert v == [], f"Catalog #131 landed with {len(v)} violations:\n" + "\n".join(v[:3])


# ──────────────────────────────────────────────────────────────────────────
# Sister gates not regressed by this lane
# ──────────────────────────────────────────────────────────────────────────


def test_127_still_clean_after_proactive_audit():
    """Sister gate #127 must still be at 0 after this lane's edits."""
    from tac.preflight import check_authoritative_tag_requires_custody_metadata

    v = check_authoritative_tag_requires_custody_metadata(strict=False, verbose=False)
    assert v == [], (
        f"Sister gate #127 regressed: {len(v)} violations\n" + "\n".join(v[:3])
    )


def test_128_still_clean_after_proactive_audit():
    """Sister gate #128 must still be at 0 after this lane's edits."""
    from tac.preflight import check_continual_learning_writes_use_lock

    v = check_continual_learning_writes_use_lock(strict=False, verbose=False)
    assert v == [], (
        f"Sister gate #128 regressed: {len(v)} violations\n" + "\n".join(v[:3])
    )
