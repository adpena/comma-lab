"""Tests for Catalog #157 — check_commit_serializer_pre_lock_hash_against_head.

2026-05-12 (subagent F, Part 4) — commit-swap class permanent fix.

The 92aba3ca commit-swap incident showed that two subagents that have
ALREADY edited the same file in the working tree BEFORE either took its
pre-lock snapshot can produce a commit-swap. The FIX-1 pre-lock vs
post-lock check would NOT catch that race. The structural fix is to (a)
add a new `--expected-content-sha256` flag that callers declare from
their work-start moment, and (b) refuse any direct `git commit`
invocation outside the canonical serializer file itself.

Memory: feedback_gc_fix_and_commit_swap_class_protect_landed_20260512.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_commit_serializer_pre_lock_hash_against_head,
)


def test_check_157_live_count_zero():
    """The check MUST have 0 live violations at landing (strict-flip atom)."""
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_commit_serializer_pre_lock_hash_against_head(
        repo_root=repo_root, strict=False, verbose=False
    )
    assert violations == [], f"Live violations should be 0, got: {violations}"


def test_check_157_detects_python_subprocess_git_commit(tmp_path):
    (tmp_path / "tools").mkdir()
    bad = tmp_path / "tools" / "bad_subproc.py"
    bad.write_text(
        "import subprocess\n"
        "subprocess.run(['git', 'commit', '-m', 'bypass'])\n"
    )
    violations = check_commit_serializer_pre_lock_hash_against_head(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("bad_subproc.py" in v for v in violations)
    with pytest.raises(PreflightError):
        check_commit_serializer_pre_lock_hash_against_head(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check_157_detects_os_system_git_commit(tmp_path):
    (tmp_path / "tools").mkdir()
    bad = tmp_path / "tools" / "bad_os_system.py"
    bad.write_text(
        "import os\n"
        "os.system('git commit -m \"bypass\"')\n"
    )
    violations = check_commit_serializer_pre_lock_hash_against_head(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("bad_os_system.py" in v for v in violations)


def test_check_157_detects_shell_git_commit(tmp_path):
    (tmp_path / "scripts").mkdir()
    bad = tmp_path / "scripts" / "bad_shell.sh"
    bad.write_text(
        "#!/bin/bash\n"
        "git commit -m 'bypass'\n"
    )
    violations = check_commit_serializer_pre_lock_hash_against_head(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("bad_shell.sh" in v for v in violations)


def test_check_157_accepts_same_line_waiver(tmp_path):
    (tmp_path / "scripts").mkdir()
    ok = tmp_path / "scripts" / "operator_reviewed.sh"
    ok.write_text(
        "#!/bin/bash\n"
        "git commit -m 'fine'  # COMMIT_SERIALIZER_BYPASS_OK:operator-reviewed\n"
    )
    violations = check_commit_serializer_pre_lock_hash_against_head(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_157_accepts_file_level_waiver(tmp_path):
    (tmp_path / "tools").mkdir()
    ok = tmp_path / "tools" / "auto_commit.sh"
    ok.write_text(
        "#!/bin/bash\n"
        "# COMMIT_SERIALIZER_BYPASS_OK_FILE:operator housekeeping\n"
        "git commit -m 'auto'\n"
        "git commit -m 'auto2'\n"
    )
    violations = check_commit_serializer_pre_lock_hash_against_head(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_157_skips_canonical_serializer(tmp_path):
    (tmp_path / "tools").mkdir()
    canonical = tmp_path / "tools" / "subagent_commit_serializer.py"
    canonical.write_text(
        "import subprocess\n"
        "subprocess.run(['git', 'commit', '-m', 'canonical'])\n"
    )
    violations = check_commit_serializer_pre_lock_hash_against_head(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_157_skips_preflight_self(tmp_path):
    (tmp_path / "src" / "tac").mkdir(parents=True)
    preflight = tmp_path / "src" / "tac" / "preflight.py"
    preflight.write_text(
        "# this file defines the patterns and may contain 'git commit'\n"
        "import subprocess\n"
        "subprocess.run(['git', 'commit', '-m', 'test'])\n"
    )
    violations = check_commit_serializer_pre_lock_hash_against_head(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_157_skips_test_files(tmp_path):
    (tmp_path / "src" / "tac" / "tests").mkdir(parents=True)
    t = tmp_path / "src" / "tac" / "tests" / "test_some_commit.py"
    t.write_text(
        "import subprocess\n"
        "subprocess.run(['git', 'commit', '-m', 'fixture'])\n"
    )
    violations = check_commit_serializer_pre_lock_hash_against_head(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_157_skips_intake_clones(tmp_path):
    intake = tmp_path / "experiments" / "results" / "public_pr95_intake_codex"
    intake.mkdir(parents=True)
    bad = intake / "vendored.sh"
    bad.write_text("git commit -m 'vendored'\n")
    violations = check_commit_serializer_pre_lock_hash_against_head(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_157_unrelated_git_commands_pass(tmp_path):
    """Lines containing `git` but not `git commit` are not flagged."""
    (tmp_path / "tools").mkdir()
    ok = tmp_path / "tools" / "unrelated.sh"
    ok.write_text(
        "#!/bin/bash\n"
        "git status\n"
        "git log --oneline\n"
        "git diff HEAD\n"
    )
    violations = check_commit_serializer_pre_lock_hash_against_head(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_157_strict_raises_on_any_violation(tmp_path):
    (tmp_path / "scripts").mkdir()
    bad = tmp_path / "scripts" / "single_bad.sh"
    bad.write_text("git commit -m 'bypass'\n")
    with pytest.raises(PreflightError):
        check_commit_serializer_pre_lock_hash_against_head(
            repo_root=tmp_path, strict=True, verbose=False
        )


# ──────────────────────────────────────────────────────────────────────────
# Companion: serializer's new --expected-content-sha256 flag
# ──────────────────────────────────────────────────────────────────────────


def _load_serializer():
    import importlib.util
    import sys
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "subagent_commit_serializer.py"
    spec = importlib.util.spec_from_file_location(
        "_subagent_commit_serializer_157", path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parse_expected_content_sha256_well_formed():
    mod = _load_serializer()
    out = mod._parse_expected_content_sha256([
        f"foo.py={'a' * 64}",
        f"bar.py={'b' * 64}",
    ])
    assert out == {"foo.py": "a" * 64, "bar.py": "b" * 64}


def test_parse_expected_content_sha256_rejects_no_equals():
    mod = _load_serializer()
    with pytest.raises(ValueError, match="must be"):
        mod._parse_expected_content_sha256(["no_equals_sign"])


def test_parse_expected_content_sha256_rejects_short_sha():
    mod = _load_serializer()
    with pytest.raises(ValueError, match="64 hex"):
        mod._parse_expected_content_sha256(["foo.py=abc"])


def test_parse_expected_content_sha256_rejects_nonhex_sha():
    mod = _load_serializer()
    with pytest.raises(ValueError, match="64 hex"):
        mod._parse_expected_content_sha256([f"foo.py={'Z' * 64}"])


def test_parse_expected_content_sha256_empty_input():
    mod = _load_serializer()
    assert mod._parse_expected_content_sha256([]) == {}
    assert mod._parse_expected_content_sha256(None or []) == {}


def test_expected_content_sha256_check_detects_mismatch(tmp_path):
    mod = _load_serializer()
    old_root = mod.REPO_ROOT
    mod.REPO_ROOT = tmp_path
    try:
        f = tmp_path / "target.py"
        f.write_text("v1")
        actual_sha = hashlib.sha256(b"v1").hexdigest()
        wrong_sha = hashlib.sha256(b"v_old_caller_expected").hexdigest()
        diffs = mod._expected_content_sha256_check({"target.py": wrong_sha})
        assert "target.py" in diffs
        # The actual matches what's on disk; the expected is what the caller wrongly declared.
        assert diffs["target.py"] == (wrong_sha, actual_sha)
    finally:
        mod.REPO_ROOT = old_root


def test_expected_content_sha256_check_passes_on_match(tmp_path):
    mod = _load_serializer()
    old_root = mod.REPO_ROOT
    mod.REPO_ROOT = tmp_path
    try:
        f = tmp_path / "target.py"
        content = b"v1_expected"
        f.write_text(content.decode())
        sha = hashlib.sha256(content).hexdigest()
        diffs = mod._expected_content_sha256_check({"target.py": sha})
        assert diffs == {}
    finally:
        mod.REPO_ROOT = old_root


def test_expected_content_sha256_check_empty_input(tmp_path):
    """Empty expected dict -> empty mismatch dict, no I/O performed."""
    mod = _load_serializer()
    assert mod._expected_content_sha256_check({}) == {}
