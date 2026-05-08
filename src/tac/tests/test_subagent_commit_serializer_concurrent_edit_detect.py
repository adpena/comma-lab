"""Unit tests for FIX-1 concurrent-edit-leak detection (META-META 2026-05-08).

The serializer hashes the working-tree content of the --files BEFORE acquiring
the file lock and AGAIN after acquiring it. If any file's content changed
during the lock-wait window, a sister subagent edited the same file and the
serializer refuses (rc=3) rather than silently package the sister's changes
under our authorship.

Bug class: META-FIX subagent's `src/tac/preflight.py` edits flowed into
sister FIX-5 commit `89d6eba2` because both subagents edited the file in the
working tree concurrently. The temp-index isolates STAGING but `git add`
reads the WORKING TREE, so concurrent working-tree edits still leaked. This
hash check catches that leak.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load_module():
    path = REPO / "tools" / "subagent_commit_serializer.py"
    spec = importlib.util.spec_from_file_location(
        "_subagent_commit_serializer_cedl", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hash_working_tree_files_returns_stable_sha(tmp_path: Path) -> None:
    """Same content yields same hash; different content yields different hash."""
    mod = _load_module()
    # Override REPO_ROOT for this test so paths are tmp_path-relative.
    old_root = mod.REPO_ROOT
    mod.REPO_ROOT = tmp_path
    try:
        f = tmp_path / "a.txt"
        f.write_bytes(b"hello\n")
        h1 = mod._hash_working_tree_files(["a.txt"])
        h2 = mod._hash_working_tree_files(["a.txt"])
        assert h1 == h2
        assert "a.txt" in h1
        assert len(h1["a.txt"]) == 64  # SHA-256 hex digest length

        f.write_bytes(b"goodbye\n")
        h3 = mod._hash_working_tree_files(["a.txt"])
        assert h3["a.txt"] != h1["a.txt"]
    finally:
        mod.REPO_ROOT = old_root


def test_hash_missing_file_returns_sentinel(tmp_path: Path) -> None:
    mod = _load_module()
    old_root = mod.REPO_ROOT
    mod.REPO_ROOT = tmp_path
    try:
        h = mod._hash_working_tree_files(["does_not_exist.txt"])
        assert h["does_not_exist.txt"] == "MISSING"
    finally:
        mod.REPO_ROOT = old_root


def test_hash_multiple_files_independent(tmp_path: Path) -> None:
    mod = _load_module()
    old_root = mod.REPO_ROOT
    mod.REPO_ROOT = tmp_path
    try:
        (tmp_path / "a.txt").write_bytes(b"alpha\n")
        (tmp_path / "b.txt").write_bytes(b"beta\n")
        h = mod._hash_working_tree_files(["a.txt", "b.txt"])
        assert h["a.txt"] != h["b.txt"]
        # Mutating only b.txt must not change a.txt's hash.
        h_a_before = h["a.txt"]
        (tmp_path / "b.txt").write_bytes(b"BETA\n")
        h2 = mod._hash_working_tree_files(["a.txt", "b.txt"])
        assert h2["a.txt"] == h_a_before
        assert h2["b.txt"] != h["b.txt"]
    finally:
        mod.REPO_ROOT = old_root


def _make_throwaway_repo(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "codex@example.invalid"],
        cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Codex"],
        cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=repo, check=True, capture_output=True,
    )
    (repo / "seed.txt").write_text("seed\n")
    subprocess.run(["git", "add", "seed.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "seed"],
        cwd=repo, check=True, capture_output=True,
        env={**os.environ, "GIT_AUTHOR_DATE": "2026-05-08T00:00:00Z"},
    )


def test_serializer_refuses_when_file_changes_between_pre_and_post_lock_hash(
    tmp_path: Path, monkeypatch
) -> None:
    """Simulate the race: pre_lock_hashes != post_lock_hashes -> rc=3 + log."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_throwaway_repo(repo)
    target = repo / "target.txt"
    target.write_text("v1\n")

    mod = _load_module()
    old_root = mod.REPO_ROOT
    old_lock = mod.LOCK_PATH
    old_log = mod.LOG_PATH
    mod.REPO_ROOT = repo
    mod.LOCK_PATH = repo / ".commit-lock"
    mod.LOG_PATH = repo / "commit-serializer.log"
    try:
        # Pre-snapshot at v1.
        pre_hashes = mod._hash_working_tree_files(["target.txt"])
        # Simulate concurrent sister-subagent edit.
        target.write_text("v2_from_sister\n")
        post_hashes = mod._hash_working_tree_files(["target.txt"])
        assert pre_hashes["target.txt"] != post_hashes["target.txt"]
    finally:
        mod.REPO_ROOT = old_root
        mod.LOCK_PATH = old_lock
        mod.LOG_PATH = old_log


def test_serializer_main_returns_3_on_concurrent_edit(tmp_path: Path) -> None:
    """End-to-end: invoke main() in a way that triggers the concurrent-edit refusal.

    We can't easily race two real serializer invocations from one test, so we
    instead patch _hash_working_tree_files to return DIFFERENT values on the
    pre-lock vs post-lock call. This exercises exactly the rc=3 path.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_throwaway_repo(repo)
    (repo / "target.txt").write_text("v1\n")

    mod = _load_module()
    old_root = mod.REPO_ROOT
    old_lock = mod.LOCK_PATH
    old_log = mod.LOG_PATH
    mod.REPO_ROOT = repo
    mod.LOCK_PATH = repo / ".commit-lock"
    mod.LOG_PATH = repo / "commit-serializer.log"

    # Patch _hash_working_tree_files to return different values on each call.
    call_count = {"n": 0}
    real_hash = mod._hash_working_tree_files

    def fake_hash(files: list[str]) -> dict[str, str]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {f: "PRE_HASH" for f in files}
        return {f: "POST_HASH_DIFFERENT" for f in files}

    mod._hash_working_tree_files = fake_hash
    old_argv = sys.argv[:]
    sys.argv = [
        "subagent_commit_serializer.py",
        "--message", "test concurrent-edit refusal",
        "--files", "target.txt",
        "--label", "test_concurrent_edit",
    ]
    try:
        rc = mod.main()
        assert rc == 3, f"expected rc=3 (concurrent edit refusal), got rc={rc}"
        # Lock log must record the concurrent_edit_detected outcome.
        assert mod.LOG_PATH.exists()
        log_lines = [l for l in mod.LOG_PATH.read_text().splitlines() if l.strip()]
        import json
        outcomes = [json.loads(l).get("outcome") for l in log_lines]
        assert "concurrent_edit_detected" in outcomes
    finally:
        sys.argv = old_argv
        mod._hash_working_tree_files = real_hash
        mod.REPO_ROOT = old_root
        mod.LOCK_PATH = old_lock
        mod.LOG_PATH = old_log


def test_no_concurrent_edit_check_flag_disables_refusal(tmp_path: Path) -> None:
    """When --no-concurrent-edit-check is passed, hash mismatch must NOT block."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_throwaway_repo(repo)
    (repo / "target.txt").write_text("v1\n")

    mod = _load_module()
    old_root = mod.REPO_ROOT
    old_lock = mod.LOCK_PATH
    old_log = mod.LOG_PATH
    mod.REPO_ROOT = repo
    mod.LOCK_PATH = repo / ".commit-lock"
    mod.LOG_PATH = repo / "commit-serializer.log"

    # Patch _hash to always claim divergence; with --no-concurrent-edit-check
    # the serializer should bypass the comparison entirely and proceed to commit.
    def divergent_hash(files: list[str]) -> dict[str, str]:
        return {f: f"H_{id(files)}" for f in files}

    real_hash = mod._hash_working_tree_files
    mod._hash_working_tree_files = divergent_hash

    old_argv = sys.argv[:]
    sys.argv = [
        "subagent_commit_serializer.py",
        "--message", "bypass concurrent-edit",
        "--files", "target.txt",
        "--no-concurrent-edit-check",
        "--label", "test_bypass",
    ]
    try:
        rc = mod.main()
        # rc==0 means commit succeeded; the bypass flag worked.
        assert rc == 0, f"expected rc=0 with --no-concurrent-edit-check, got rc={rc}"
    finally:
        sys.argv = old_argv
        mod._hash_working_tree_files = real_hash
        mod.REPO_ROOT = old_root
        mod.LOCK_PATH = old_lock
        mod.LOG_PATH = old_log
