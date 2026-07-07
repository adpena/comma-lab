# SPDX-License-Identifier: MIT
"""FIX-ABSORPTION (2026-07-07) tests: --base-content-sha256 vs the
serializer_whole_file_staging_absorbs_sibling_hunks class.

Incident (measured, git-history-verified): sibling agents co-edited
experiments/train_levelset_witness_realized_through_R_mlx.py +
src/tac/witness_dsl/curriculum_dsl.py in the shared working tree; the
committing agents' whole-file `git add` staged the sibling's uncommitted
hunks under their commit bodies (1d6704e5b absorbed the --lane-band-dash-comb
trainer wire-in; 049aa0d9f absorbed the DashComb Lever). Every committer
passed --expected-content-sha256 and every check PASSED — the sha was
computed on the already-merged working tree, so Catalog #157 (rc=4) and
#216 (rc=5) are tautological against co-mingled content.

These tests (1) REPRODUCE the absorption in a scratch throwaway repo with
the legacy flag set (proving the old behavior), and (2) prove the new
--base-content-sha256 check refuses it with rc=6.

ALL tests run against throwaway git repos under tmp_path — NEVER the real
repo (the serializer's REPO_ROOT / LOCK_PATH / LOG_PATH globals are patched
per-test, mirroring test_subagent_commit_serializer_concurrent_edit_detect).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

_SERIALIZER_PATH = Path(
    os.environ.get(
        "SUBAGENT_SERIALIZER_PATH",
        str(REPO / "tools" / "subagent_commit_serializer.py"),
    )
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_subagent_commit_serializer_base_sha", _SERIALIZER_PATH,
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
        ["config", "user.name", "AbsorptionTest"],
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


# ---------------------------------------------------------------------------
# 1. REPRODUCTION: legacy behavior absorbs the sibling's hunk (no base flag).
# ---------------------------------------------------------------------------

def test_absorption_reproduced_with_legacy_expected_sha_only(tmp_path: Path) -> None:
    """The measured incident mechanism: sibling hunk in the working tree at
    edit-start; committer declares the (merged) post-edit sha; every legacy
    check passes; the commit ABSORBS the sibling's hunk. This documents the
    old behavior — the class-fix is opt-in, so this path still exists."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    shared = repo / "shared.txt"

    # Sibling C edits first (uncommitted).
    shared.write_bytes(b"line1\nSIBLING-HUNK\n")
    # Agent A edits on top (the shared working tree co-mingles both).
    merged = b"line1\nSIBLING-HUNK\nAGENT-HUNK\n"
    shared.write_bytes(merged)

    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "agent A commit (legacy discipline)",
            "--files", "shared.txt",
            "--expected-content-sha256", f"shared.txt={_sha(merged)}",
            "--no-sister-checkpoint-check",
            "--label", "agent_A_legacy",
        ])
        assert rc == 0, "legacy path must commit (documents the old behavior)"
        head_blob = _git(repo, "cat-file", "blob", "HEAD:shared.txt")
        assert b"SIBLING-HUNK" in head_blob.stdout.encode(), (
            "absorption NOT reproduced — sibling hunk missing from HEAD"
        )
        assert "committed" in _log_outcomes(mod)


# ---------------------------------------------------------------------------
# 2. THE FIX: --base-content-sha256 refuses the same scenario with rc=6.
# ---------------------------------------------------------------------------

def test_base_sha_refuses_absorption_rc6(tmp_path: Path) -> None:
    """Same interleaving as the reproduction, but agent A declares the base
    it actually saw before its own edits (which already contained the
    sibling's hunk). base != HEAD blob -> rc=6, no commit, log row."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    shared = repo / "shared.txt"
    head_before = _git(repo, "rev-parse", "HEAD").stdout.strip()

    sibling_base = b"line1\nSIBLING-HUNK\n"
    shared.write_bytes(sibling_base)          # sibling C's uncommitted edit
    merged = b"line1\nSIBLING-HUNK\nAGENT-HUNK\n"
    shared.write_bytes(merged)                # agent A's edit on top

    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "agent A commit (base-guarded)",
            "--files", "shared.txt",
            "--base-content-sha256", f"shared.txt={_sha(sibling_base)}",
            "--expected-content-sha256", f"shared.txt={_sha(merged)}",
            "--no-sister-checkpoint-check",
            "--label", "agent_A_guarded",
        ])
        assert rc == 6, f"expected rc=6 base-mismatch refusal, got rc={rc}"
        head_after = _git(repo, "rev-parse", "HEAD").stdout.strip()
        assert head_after == head_before, "refusal must not create a commit"
        assert "base_content_sha_mismatch_pre_lock" in _log_outcomes(mod)
        # The log row must carry forensics: declared base + HEAD blob shas.
        rows = [json.loads(line) for line in mod.LOG_PATH.read_text().splitlines()]
        row = next(r for r in rows
                   if r["outcome"] == "base_content_sha_mismatch_pre_lock")
        diffs = row["base_content_sha_diffs"]["shared.txt"]
        assert diffs["declared_base"] == _sha(sibling_base)
        assert diffs["head_blob"] == _sha(b"line1\n")


def test_base_sha_passes_when_base_equals_head(tmp_path: Path) -> None:
    """Positive control: no sibling hunks — base == HEAD blob — commit OK,
    and HEAD contains exactly the agent's edit."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    shared = repo / "shared.txt"
    base = shared.read_bytes()                 # == HEAD content
    post = b"line1\nAGENT-HUNK\n"
    shared.write_bytes(post)

    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "clean base-guarded commit",
            "--files", "shared.txt",
            "--base-content-sha256", f"shared.txt={_sha(base)}",
            "--expected-content-sha256", f"shared.txt={_sha(post)}",
            "--no-sister-checkpoint-check",
            "--label", "agent_clean",
        ])
        assert rc == 0, f"clean base must commit, got rc={rc}"
        blob = _git(repo, "cat-file", "blob", "HEAD:shared.txt")
        assert blob.stdout.encode() == post


def test_base_sha_passes_after_sibling_lands(tmp_path: Path) -> None:
    """WAIT_AND_RETRY resolution: after the sibling COMMITS the hunks that
    were in the caller's base, HEAD matches the base and the retry passes —
    with correct attribution (only the agent's hunk in the new commit)."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    shared = repo / "shared.txt"

    sibling_base = b"line1\nSIBLING-HUNK\n"
    shared.write_bytes(sibling_base)
    merged = b"line1\nSIBLING-HUNK\nAGENT-HUNK\n"
    shared.write_bytes(merged)

    mod = _load_module()
    with _Patched(mod, repo):
        argv = [
            "--message", "agent A retry after sibling lands",
            "--files", "shared.txt",
            "--base-content-sha256", f"shared.txt={_sha(sibling_base)}",
            "--expected-content-sha256", f"shared.txt={_sha(merged)}",
            "--no-sister-checkpoint-check",
            "--label", "agent_A_retry",
        ]
        assert _run_main(mod, argv) == 6      # refused while sibling in-flight

        # Sibling lands ITS hunk (stash agent's, commit sibling's, restore).
        shared.write_bytes(sibling_base)
        assert _git(repo, "add", "shared.txt").returncode == 0
        assert _git(repo, "commit", "-m", "sibling C lands").returncode == 0
        shared.write_bytes(merged)

        rc = _run_main(mod, argv)             # retry: HEAD == base now
        assert rc == 0, f"retry after sibling landed must pass, got rc={rc}"
        show = _git(repo, "show", "HEAD", "--", "shared.txt")
        assert "+AGENT-HUNK" in show.stdout
        assert "+SIBLING-HUNK" not in show.stdout, (
            "sibling hunk mis-attributed to the agent's commit"
        )


def test_base_new_token_for_caller_created_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    (repo / "fresh.txt").write_bytes(b"created by agent\n")

    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "new file",
            "--files", "fresh.txt",
            "--base-content-sha256", "fresh.txt=new",
            "--no-sister-checkpoint-check",
            "--label", "agent_newfile",
        ])
        assert rc == 0, f"'new' base for untracked file must commit, got rc={rc}"


def test_base_new_token_refused_when_tracked_at_head(tmp_path: Path) -> None:
    """Declaring 'new' for a file that exists at HEAD is a stale-base error."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    (repo / "shared.txt").write_bytes(b"overwrite as if new\n")

    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "bogus new claim",
            "--files", "shared.txt",
            "--base-content-sha256", "shared.txt=new",
            "--no-sister-checkpoint-check",
            "--label", "agent_bogus_new",
        ])
        assert rc == 6, f"'new' for HEAD-tracked file must refuse, got rc={rc}"


def test_malformed_base_sha_is_fatal_rc2(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    (repo / "shared.txt").write_bytes(b"x\n")

    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "bad flag",
            "--files", "shared.txt",
            "--base-content-sha256", "shared.txt=nothex",
            "--no-sister-checkpoint-check",
            "--label", "agent_badflag",
        ])
        assert rc == 2


def test_post_lock_base_recheck_refuses_head_movement(tmp_path: Path) -> None:
    """A sibling commit landing DURING the lock-wait (novel content the
    caller never based on) must trip the post-lock re-check. We simulate by
    making the pre-lock check pass and then moving HEAD before the post-lock
    call via a patched _base_content_check that mimics HEAD movement:
    first call (pre-lock) clean, second call (post-lock) mismatched."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    shared = repo / "shared.txt"
    base = shared.read_bytes()
    shared.write_bytes(b"line1\nAGENT-HUNK\n")
    head_before = _git(repo, "rev-parse", "HEAD").stdout.strip()

    mod = _load_module()
    calls = {"n": 0}
    real_check = mod._base_content_check

    def racing_check(declared):
        calls["n"] += 1
        if calls["n"] == 1:
            return {}                          # pre-lock: clean
        return {"shared.txt": (next(iter(declared.values())), "HEAD_MOVED_SHA")}

    with _Patched(mod, repo):
        mod._base_content_check = racing_check
        try:
            rc = _run_main(mod, [
                "--message", "race during lock-wait",
                "--files", "shared.txt",
                "--base-content-sha256", f"shared.txt={_sha(base)}",
                "--no-sister-checkpoint-check",
                "--label", "agent_lockwait_race",
            ])
        finally:
            mod._base_content_check = real_check
        assert rc == 6, f"post-lock base re-check must refuse, got rc={rc}"
        assert calls["n"] == 2, "post-lock re-check did not run"
        assert _git(repo, "rev-parse", "HEAD").stdout.strip() == head_before
        assert "base_content_sha_mismatch_post_lock" in _log_outcomes(mod)


def test_no_base_flag_is_backward_compatible(tmp_path: Path) -> None:
    """Callers that don't pass the new flag see zero behavior change."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    (repo / "shared.txt").write_bytes(b"line1\nplain edit\n")

    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "plain commit, no new flags",
            "--files", "shared.txt",
            "--no-sister-checkpoint-check",
            "--label", "agent_plain",
        ])
        assert rc == 0


def test_hash_head_blob_files_missing_and_present(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    mod = _load_module()
    with _Patched(mod, repo):
        h = mod._hash_head_blob_files(["shared.txt", "ghost.txt"])
        assert h["shared.txt"] == _sha(b"line1\n")
        assert h["ghost.txt"] == "MISSING"
        # HEAD blob is the COMMITTED content, not the working tree.
        (repo / "shared.txt").write_bytes(b"working tree divergence\n")
        h2 = mod._hash_head_blob_files(["shared.txt"])
        assert h2["shared.txt"] == _sha(b"line1\n")
