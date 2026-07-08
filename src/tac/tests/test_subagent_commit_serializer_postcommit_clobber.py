# SPDX-License-Identifier: MIT
"""FIX-CLOBBER (2026-07-08 Catalog #405) tests for subagent_commit_serializer.

Two 2026-07-08 incidents motivated this hardening:

  (1) A sibling's file REVERT landed in the working tree BEFORE a builder
      snapshotted --expected-content-sha256, so every PRE-commit working-tree
      check compared against the already-clobbered content and passed; rc=0
      committed the sibling's copy. The gap: no prior check reads HEAD AFTER
      the ref moved. Fix: POST-COMMIT verification (rc=7).

  (2) A whole-file `git add` swept a DIFFERENT sibling's uncommitted hunks
      into the wrong commit body (mis-attribution). Fix: --patch-file
      (intent-manifest) staging + a warn-only --expected-diff-lines heuristic.

ALL tests run against throwaway git repos under tmp_path — NEVER the real
repo (the serializer's REPO_ROOT / LOCK_PATH / LOG_PATH globals are patched
per-test, mirroring test_subagent_commit_serializer_base_sha).

Backward-compatibility guard: existing callers that pass only
--expected-content-sha256 must behave IDENTICALLY on the happy path.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

_SERIALIZER_PATH = Path(
    os.environ.get(
        "SUBAGENT_SERIALIZER_PATH",
        str(REPO / "tools" / "subagent_commit_serializer.py"),
    )
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_subagent_commit_serializer_clobber", _SERIALIZER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False,
    )


def _make_throwaway_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    for cmd in (
        ["init"],
        ["config", "user.email", "test@example.invalid"],
        ["config", "user.name", "ClobberTest"],
        ["config", "commit.gpgsign", "false"],
    ):
        proc = _git(repo, *cmd)
        assert proc.returncode == 0, proc.stderr
    (repo / "shared.txt").write_bytes(b"line1\n")
    assert _git(repo, "add", "shared.txt").returncode == 0
    assert _git(repo, "commit", "-m", "seed").returncode == 0


class _Patched:
    """Context manager: point the serializer module at a throwaway repo."""

    def __init__(self, mod, repo: Path):
        self.mod = mod
        self.repo = repo

    def __enter__(self):
        self.old = (self.mod.REPO_ROOT, self.mod.LOCK_PATH, self.mod.LOG_PATH)
        self.mod.REPO_ROOT = self.repo
        self.mod.LOCK_PATH = self.repo / ".commit-lock"
        self.mod.LOG_PATH = self.repo / "commit-serializer.log"
        return self

    def __exit__(self, *exc):
        self.mod.REPO_ROOT, self.mod.LOCK_PATH, self.mod.LOG_PATH = self.old
        return False


def _run_main(mod, argv: list[str]) -> int:
    old_argv = sys.argv[:]
    sys.argv = ["subagent_commit_serializer.py", *argv]
    try:
        return mod.main()
    finally:
        sys.argv = old_argv


def _log_outcomes(mod) -> list[str]:
    if not mod.LOG_PATH.exists():
        return []
    return [
        json.loads(line).get("outcome")
        for line in mod.LOG_PATH.read_text().splitlines()
        if line.strip()
    ]


def _log_rows(mod) -> list[dict]:
    if not mod.LOG_PATH.exists():
        return []
    return [json.loads(line) for line in mod.LOG_PATH.read_text().splitlines()
            if line.strip()]


def _install_mutating_pre_commit_hook(repo: Path, target: str, new_bytes: bytes) -> None:
    """Install a pre-commit hook that REWRITES + re-stages `target` to
    `new_bytes`, simulating a content mutation (clobber/formatter) that lands
    between the serializer's pre-commit checks and the HEAD-blob landing. The
    hook honours GIT_INDEX_FILE (inherited) so its `git add` targets the temp
    index the commit will snapshot."""
    hooks = repo / ".git" / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    hook = hooks / "pre-commit"
    # Write new_bytes via python for byte-exactness, then re-stage.
    b64 = new_bytes.hex()
    hook.write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        f'python3 -c "import binascii,pathlib; '
        f"pathlib.Path('{target}').write_bytes(binascii.unhexlify('{b64}'))\"\n"
        f'git add -- "{target}"\n'
    )
    hook.chmod(0o755)


# ===========================================================================
# 1. POST-COMMIT verification (rc=7) — the incident-1 clobber gap.
# ===========================================================================

def test_post_commit_check_unit_detects_head_mismatch(tmp_path: Path) -> None:
    """Unit: _post_commit_content_check returns a mismatch when the declared
    sha differs from the file's content at HEAD."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    mod = _load_module()
    with _Patched(mod, repo):
        head_actual = _sha(b"line1\n")
        wrong = _sha(b"line1\nINTENDED\n")
        diffs = mod._post_commit_content_check({"shared.txt": wrong})
        assert "shared.txt" in diffs
        assert diffs["shared.txt"] == (wrong, head_actual)
        # Matching declared -> no mismatch.
        assert mod._post_commit_content_check({"shared.txt": head_actual}) == {}
        # Empty declared -> no-op (backward-compatible).
        assert mod._post_commit_content_check({}) == {}


def test_post_commit_rc7_via_mutating_hook_keeps_commit(tmp_path: Path) -> None:
    """Faithful clobber: the caller declares its INTENDED post-edit sha; the
    working tree matches it (pre-commit checks pass); a mutation lands the
    SIBLING's copy at HEAD. Post-commit verification catches the divergence
    with rc=7 and the commit is KEPT (not auto-reverted)."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    shared = repo / "shared.txt"

    intended = b"line1\nINTENDED\n"
    sibling = b"line1\nSIBLING-CLOBBER\n"
    shared.write_bytes(intended)  # working tree == what the caller declares
    _install_mutating_pre_commit_hook(repo, "shared.txt", sibling)

    mod = _load_module()
    head_before = _git(repo, "rev-parse", "HEAD").stdout.strip()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "builder commit (declared intended, HEAD clobbered)",
            "--files", "shared.txt",
            "--expected-content-sha256", f"shared.txt={_sha(intended)}",
            "--no-sister-checkpoint-check",
            "--label", "builder_clobbered",
        ])
        outcomes = _log_outcomes(mod)
        rows = _log_rows(mod)
    assert rc == 7, f"expected rc=7 post-commit mismatch, got rc={rc}"
    head_after = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert head_after != head_before, "the commit must be KEPT (HEAD moved)"
    blob = _git(repo, "cat-file", "blob", "HEAD:shared.txt").stdout.encode()
    assert b"SIBLING-CLOBBER" in blob, "the clobbered content did land at HEAD"
    assert "post_commit_content_sha_mismatch" in outcomes
    row = next(r for r in rows
               if r["outcome"] == "post_commit_content_sha_mismatch")
    diff = row["post_commit_content_sha_diffs"]["shared.txt"]
    assert diff["declared"] == _sha(intended)
    assert diff["committed_head"] == _sha(sibling)


def test_post_commit_rc7_message_names_file_and_shas(tmp_path: Path, capsys) -> None:
    """The rc=7 refusal message must name the file, declared sha, committed
    sha, the likely cause, and the reconcile guidance (git show / --patch-file
    / git revert) — and must NOT auto-revert."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    shared = repo / "shared.txt"
    intended = b"line1\nINTENDED\n"
    sibling = b"line1\nSIBLING-CLOBBER\n"
    shared.write_bytes(intended)
    _install_mutating_pre_commit_hook(repo, "shared.txt", sibling)

    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "builder commit",
            "--files", "shared.txt",
            "--expected-content-sha256", f"shared.txt={_sha(intended)}",
            "--no-sister-checkpoint-check",
        ])
    assert rc == 7
    err = capsys.readouterr().err
    assert "rc=7" in err
    assert "shared.txt" in err
    assert _sha(intended) in err and _sha(sibling) in err
    assert "git show HEAD:" in err
    assert "--patch-file" in err
    assert "git revert" in err
    assert "NOT auto-reverted" in err


def test_post_commit_happy_path_rc0(tmp_path: Path) -> None:
    """When the committed content matches the declared sha (the normal case),
    post-commit verification passes silently and rc=0 — identical to the
    pre-hardening behavior."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    shared = repo / "shared.txt"
    intended = b"line1\nINTENDED\n"
    shared.write_bytes(intended)

    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "clean commit",
            "--files", "shared.txt",
            "--expected-content-sha256", f"shared.txt={_sha(intended)}",
            "--no-sister-checkpoint-check",
        ])
        outcomes = _log_outcomes(mod)
    assert rc == 0
    assert "committed" in outcomes
    assert "post_commit_content_sha_mismatch" not in outcomes
    blob = _git(repo, "cat-file", "blob", "HEAD:shared.txt").stdout.encode()
    assert blob == intended


def test_post_commit_noop_when_no_expected_sha_backward_compat(tmp_path: Path) -> None:
    """Backward-compat: a caller passing NO --expected-content-sha256 gets
    rc=0 and no post-commit check runs (the check is opt-in via the flag)."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    shared = repo / "shared.txt"
    shared.write_bytes(b"line1\nplain\n")

    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "legacy commit no shas",
            "--files", "shared.txt",
            "--no-sister-checkpoint-check",
        ])
        outcomes = _log_outcomes(mod)
    assert rc == 0
    assert "committed" in outcomes
    assert "post_commit_content_sha_mismatch" not in outcomes


# ===========================================================================
# 2. rc semantics — rc=7 is distinct from the pre-commit refusal codes.
# ===========================================================================

def test_rc4_still_fires_before_commit_distinct_from_rc7(tmp_path: Path) -> None:
    """A pre-lock working-tree mismatch is still rc=4 (no commit), NOT rc=7.
    Guards that the new post-commit path did not disturb rc=4 semantics."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    shared = repo / "shared.txt"
    shared.write_bytes(b"line1\nACTUAL\n")

    other_sha = _sha(b"line1\nOTHER\n")
    mod = _load_module()
    head_before = _git(repo, "rev-parse", "HEAD").stdout.strip()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "declared != working tree",
            "--files", "shared.txt",
            "--expected-content-sha256", f"shared.txt={other_sha}",
            "--no-sister-checkpoint-check",
        ])
    assert rc == 4, f"pre-lock mismatch must be rc=4, got {rc}"
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == head_before


# ===========================================================================
# 3. --patch-file intent-manifest mode — the incident-2 fix.
# ===========================================================================

def _make_patch_of_intended(repo: Path, path: str, intended: bytes) -> str:
    """Write intended content, capture `git diff HEAD` as a patch, return the
    patch text. Leaves the working tree at `intended` (caller may clobber it)."""
    (repo / path).write_bytes(intended)
    diff = _git(repo, "diff", "HEAD", "--", path)
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout.strip(), "patch generation produced empty diff"
    return diff.stdout


def test_patch_file_commits_only_patch_ignoring_clobbered_worktree(tmp_path: Path) -> None:
    """The core incident-2 fix: even when the working tree is clobbered with a
    sibling's hunks, --patch-file commits EXACTLY the caller's patch (applied
    to a temp index seeded from HEAD), never the working-tree content."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    intended = b"line1\nAGENT-HUNK\n"
    patch_text = _make_patch_of_intended(repo, "shared.txt", intended)
    patch_file = repo / "agent.patch"
    patch_file.write_text(patch_text)

    # Now clobber the working tree with a sibling's co-mingled content.
    (repo / "shared.txt").write_bytes(b"line1\nAGENT-HUNK\nSIBLING-SWEPT\n")

    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "agent commit via patch-file",
            "--patch-file", str(patch_file),
            "--no-sister-checkpoint-check",
            "--label", "agent_patch",
        ])
        rows = _log_rows(mod)
    assert rc == 0, f"patch-file commit should succeed, got rc={rc}"
    blob = _git(repo, "cat-file", "blob", "HEAD:shared.txt").stdout.encode()
    assert blob == intended, "committed content must be the patch, not the clobber"
    assert b"SIBLING-SWEPT" not in blob, "sibling's swept hunk must NOT be absorbed"
    assert any(r.get("patch_mode") for r in rows)


def test_patch_file_derives_file_set_from_patch(tmp_path: Path) -> None:
    """--patch-file without --files derives the committed file set from the
    patch headers (for logging + sister-checkpoint)."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    intended = b"line1\nDERIVED\n"
    patch_text = _make_patch_of_intended(repo, "shared.txt", intended)
    patch_file = repo / "d.patch"
    patch_file.write_text(patch_text)

    mod = _load_module()
    # Verify the parser extracts the target.
    assert mod._parse_patch_target_files(patch_text) == ["shared.txt"]
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "derived files",
            "--patch-file", str(patch_file),
            "--no-sister-checkpoint-check",
        ])
        rows = _log_rows(mod)
    assert rc == 0
    row = next(r for r in rows if r["outcome"] == "committed")
    assert row["files"] == ["shared.txt"]


def test_patch_file_not_based_on_head_fails_loudly(tmp_path: Path) -> None:
    """A patch whose context does not match HEAD fails at `git apply --cached`
    (non-zero rc, no commit) instead of silently mis-applying."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    # A patch with wrong context lines (does not match HEAD's 'line1\n').
    bad_patch = (
        "diff --git a/shared.txt b/shared.txt\n"
        "--- a/shared.txt\n"
        "+++ b/shared.txt\n"
        "@@ -1 +1,2 @@\n"
        "-WRONGCONTEXT\n"
        "+WRONGCONTEXT\n"
        "+ADDED\n"
    )
    patch_file = repo / "bad.patch"
    patch_file.write_text(bad_patch)

    mod = _load_module()
    head_before = _git(repo, "rev-parse", "HEAD").stdout.strip()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "bad patch",
            "--patch-file", str(patch_file),
            "--no-sister-checkpoint-check",
        ])
        outcomes = _log_outcomes(mod)
    assert rc != 0, "a patch not based on HEAD must fail loudly"
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == head_before
    assert "git_apply_failed" in outcomes


def test_patch_file_empty_errors(tmp_path: Path) -> None:
    """An empty patch file is a usage error (SystemExit from parser.error)."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    patch_file = repo / "empty.patch"
    patch_file.write_text("   \n")
    mod = _load_module()
    with _Patched(mod, repo), pytest.raises(SystemExit):
        _run_main(mod, [
            "--message", "empty patch",
            "--patch-file", str(patch_file),
            "--no-sister-checkpoint-check",
        ])


def test_patch_file_post_commit_verify_still_runs_rc7(tmp_path: Path) -> None:
    """In patch mode the working-tree sha checks are skipped, but post-commit
    HEAD verification STILL runs when --expected-content-sha256 is passed. A
    deliberately wrong declared sha -> rc=7 (committed patch != declared)."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    intended = b"line1\nAGENT-HUNK\n"
    patch_text = _make_patch_of_intended(repo, "shared.txt", intended)
    patch_file = repo / "p.patch"
    patch_file.write_text(patch_text)
    # reset working tree to HEAD to prove patch mode ignores it entirely
    (repo / "shared.txt").write_bytes(b"line1\n")
    wrong_sha = _sha(b"line1\nWRONG\n")

    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "patch + wrong declared sha",
            "--patch-file", str(patch_file),
            "--expected-content-sha256", f"shared.txt={wrong_sha}",
            "--no-sister-checkpoint-check",
        ])
    assert rc == 7, f"post-commit verify should fire in patch mode, got rc={rc}"
    # The commit is kept; committed content is the patch result.
    blob = _git(repo, "cat-file", "blob", "HEAD:shared.txt").stdout.encode()
    assert blob == intended


# ===========================================================================
# 4. --expected-diff-lines warn-only hunk-attribution heuristic.
# ===========================================================================

def test_expected_diff_lines_overshoot_warns_but_commits(tmp_path: Path, capsys) -> None:
    """A staged diff grossly larger (>2x) than the hint WARNS + logs but does
    NOT refuse — the commit succeeds (rc=0)."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    # Stage a large change while hinting only 1 line.
    big = b"line1\n" + b"".join(f"extra{i}\n".encode() for i in range(20))
    (repo / "shared.txt").write_bytes(big)

    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "big staged diff, small hint",
            "--files", "shared.txt",
            "--expected-diff-lines", "shared.txt=1",
            "--no-sister-checkpoint-check",
        ])
        outcomes = _log_outcomes(mod)
    assert rc == 0, "hunk-attribution heuristic must NOT refuse"
    assert "hunk_attribution_overshoot_warned" in outcomes
    err = capsys.readouterr().err
    assert "WARNING" in err and "--patch-file" in err


def test_expected_diff_lines_within_bounds_no_warn(tmp_path: Path) -> None:
    """A staged diff within 2x of the hint produces no overshoot warning."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    (repo / "shared.txt").write_bytes(b"line1\nadded\n")  # +1 line staged

    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "small staged diff, matching hint",
            "--files", "shared.txt",
            "--expected-diff-lines", "shared.txt=2",
            "--no-sister-checkpoint-check",
        ])
        outcomes = _log_outcomes(mod)
    assert rc == 0
    assert "hunk_attribution_overshoot_warned" not in outcomes
    assert "committed" in outcomes


def test_expected_diff_lines_malformed_rc2(tmp_path: Path) -> None:
    """Malformed --expected-diff-lines -> ValueError -> rc=2 (fatal parse)."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    (repo / "shared.txt").write_bytes(b"line1\nx\n")
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "bad hint",
            "--files", "shared.txt",
            "--expected-diff-lines", "shared.txt=notanint",
            "--no-sister-checkpoint-check",
        ])
    assert rc == 2


def test_parse_expected_diff_lines_unit() -> None:
    """Unit coverage of the parser: valid, empty, negative, missing '='."""
    mod = _load_module()
    assert mod._parse_expected_diff_lines(["a.py=5", "b.py=0"]) == {"a.py": 5, "b.py": 0}
    assert mod._parse_expected_diff_lines([]) == {}
    for bad in (["a.py"], ["a.py=-3"], ["=5"], ["a.py="]):
        with pytest.raises(ValueError):
            mod._parse_expected_diff_lines(bad)


def test_parse_patch_target_files_unit() -> None:
    """Unit coverage of patch file-path extraction (+++ b/ and diff --git)."""
    mod = _load_module()
    patch = (
        "diff --git a/foo/bar.py b/foo/bar.py\n"
        "index 111..222 100644\n"
        "--- a/foo/bar.py\n"
        "+++ b/foo/bar.py\n"
        "@@ -1 +1 @@\n-x\n+y\n"
    )
    assert mod._parse_patch_target_files(patch) == ["foo/bar.py"]
    # /dev/null target (deletion) is ignored on the +++ side.
    assert mod._parse_patch_target_files(
        "diff --git a/g.py b/g.py\n--- a/g.py\n+++ /dev/null\n"
    ) == ["g.py"]


# ===========================================================================
# 5. Backward compatibility — the existing base-sha absorption path is intact.
# ===========================================================================

def test_backward_compat_expected_sha_happy_path_unchanged(tmp_path: Path) -> None:
    """The canonical existing caller (only --expected-content-sha256, matching
    the post-edit working tree) commits at rc=0 exactly as before hardening."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    post = b"line1\nMY-EDIT\n"
    (repo / "shared.txt").write_bytes(post)

    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "canonical existing caller",
            "--files", "shared.txt",
            "--expected-content-sha256", f"shared.txt={_sha(post)}",
            "--no-sister-checkpoint-check",
        ])
        outcomes = _log_outcomes(mod)
    assert rc == 0
    blob = _git(repo, "cat-file", "blob", "HEAD:shared.txt").stdout.encode()
    assert blob == post
    assert "committed" in outcomes
