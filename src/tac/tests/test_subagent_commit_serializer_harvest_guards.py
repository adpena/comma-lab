# SPDX-License-Identifier: MIT
"""PERMANENT FIX (2026-07-18) tests: the two harvest bug classes that BOTH main
and the v10 build-wave subagents hit while committing through the canonical
serializer.

Bug class 1 — gitignored-file silent whole-commit abort (rc=13). A single
gitignored path in --files poisons `git add` and aborts the ENTIRE commit with
only a generic git hint (no named culprit). The 2026-07-18 B power-diagram
harvest said "committing 38 files" but HEAD never moved because ONE gitignored
storage_plan.json failed the `git add`. The fix REFUSES pre-lock, naming every
offending file, so the caller removes it (bulk / rebuildable artifacts belong on
the SSD cold-store, not git).

Bug class 2 — protected append-doc whole-file clobber off a stale base (rc=14).
Multi-writer append-heavy research docs (completeness matrix / sub015_DAG /
DAG-FEED / canonical_equations_registry) never SHRINK in normal operation; a
whole-file overwrite off a STALE base silently drops sibling rows (the
2026-07-18 completeness-matrix incident wiped the compiler arm's factor-1/5
edits). A net line LOSS is the clobber signature; the guard refuses it unless an
INTENTIONAL shrink (consolidation) is declared via --allow-shared-doc-shrink.

This is the ONE canonical commit primitive that main + every subagent + Kama AI
in production route through, so hardening it here fixes the class everywhere.

ALL tests run against throwaway git repos under tmp_path — NEVER the real repo
(REPO_ROOT / LOCK_PATH / LOG_PATH are patched per-test, mirroring
test_subagent_commit_serializer_base_sha).
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
        "_subagent_commit_serializer_harvest_guards", _SERIALIZER_PATH,
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
        ["config", "user.name", "HarvestGuardTest"],
        ["config", "commit.gpgsign", "false"],
    ):
        proc = _git(repo, *cmd)
        assert proc.returncode == 0, proc.stderr
    # .gitignore so we can exercise the gitignored-file path.
    (repo / ".gitignore").write_text("*.gz\nstorage_plan.json\n")
    (repo / "seed.txt").write_bytes(b"seed\n")
    assert _git(repo, "add", ".gitignore", "seed.txt").returncode == 0
    assert _git(repo, "commit", "-m", "seed").returncode == 0


class _Patched:
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


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


# ---------------------------------------------------------------------------
# Bug class 1 — gitignored-file silent whole-commit abort (rc=13).
# ---------------------------------------------------------------------------

def test_gitignored_file_in_files_refused_rc13(tmp_path: Path) -> None:
    """A gitignored path in --files → REFUSED pre-lock with rc=13, HEAD unmoved,
    the culprit named in the log (the 2026-07-18 'committing N files, HEAD
    unmoved' incident)."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    (repo / "storage_plan.json").write_text('{"bulk": true}\n')  # gitignored
    (repo / "real.txt").write_text("real content\n")  # tracked-worthy
    before = _head(repo)
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "harvest with a gitignored bulk file",
            "--files", "real.txt", "storage_plan.json",
            "--no-sister-checkpoint-check",
            "--label", "gitignore_refuse",
        ])
        outcomes = _log_outcomes(mod)  # read INSIDE (patched LOG_PATH)
    assert rc == 13, "gitignored file must be refused with rc=13"
    assert _head(repo) == before, "HEAD must NOT move on refusal"
    assert "gitignored_files_in_commit_refused" in outcomes


def test_normal_files_only_commit_fine(tmp_path: Path) -> None:
    """Negative case: no gitignored files → normal commit succeeds (the guard
    is a scalpel, not a blanket)."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    (repo / "real.txt").write_text("real content\n")
    before = _head(repo)
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "normal harvest",
            "--files", "real.txt",
            "--no-sister-checkpoint-check",
            "--label", "normal_ok",
        ])
    assert rc == 0, "a commit with no gitignored files must succeed"
    assert _head(repo) != before, "HEAD must advance on a real commit"


def test_gz_extension_ignored_file_refused(tmp_path: Path) -> None:
    """The exact 2026-07-18 shape: a `.source.gz` bulk artifact (SSD cold-store
    material) passed to --files is refused, not silently absorbed."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    (repo / "evidence.source.gz").write_bytes(b"\x1f\x8b bulk")  # *.gz ignored
    (repo / "memo.md").write_text("real memo\n")
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "harvest with a .source.gz",
            "--files", "memo.md", "evidence.source.gz",
            "--no-sister-checkpoint-check",
            "--label", "gz_refuse",
        ])
        outcomes = _log_outcomes(mod)
    assert rc == 13
    assert "gitignored_files_in_commit_refused" in outcomes


# ---------------------------------------------------------------------------
# Bug class 2 — protected append-doc whole-file clobber (rc=14).
# ---------------------------------------------------------------------------

def _seed_protected_doc(repo: Path, relpath: str, n_lines: int) -> None:
    p = repo / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f"row {i}\n" for i in range(n_lines)))
    assert _git(repo, "add", relpath).returncode == 0
    assert _git(repo, "commit", "-m", f"seed {relpath}").returncode == 0


def test_protected_doc_shrink_refused_rc14(tmp_path: Path) -> None:
    """A protected append doc that loses >= _PROTECTED_DOC_SHRINK_LINES lines vs
    HEAD (the stale-base whole-file-clobber signature) is REFUSED with rc=14."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    rel = ".omx/research/inverse_solve_completeness_matrix_20260718.md"
    _seed_protected_doc(repo, rel, 40)
    before = _head(repo)
    # Clobber: overwrite with a stale/short version (drops 30 rows >= 8).
    (repo / rel).write_text("".join(f"row {i}\n" for i in range(10)))
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "clobber the matrix off a stale base",
            "--files", rel,
            "--no-sister-checkpoint-check",
            "--label", "matrix_clobber",
        ])
        outcomes = _log_outcomes(mod)
    assert rc == 14, "protected-doc clobber must be refused with rc=14"
    assert _head(repo) == before, "HEAD must NOT move on refusal"
    assert "protected_append_doc_clobber_refused" in outcomes


def test_protected_doc_shrink_allowed_with_rationale(tmp_path: Path) -> None:
    """An INTENTIONAL shrink (consolidation) with a real rationale is allowed
    and logged as such — the escape hatch works."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    rel = "docs/sub015_DAG_topaiml_reopen.md"
    _seed_protected_doc(repo, rel, 40)
    before = _head(repo)
    (repo / rel).write_text("".join(f"row {i}\n" for i in range(10)))
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "deliberate DAG consolidation",
            "--files", rel,
            "--allow-shared-doc-shrink", "quarterly consolidation pass",
            "--no-sister-checkpoint-check",
            "--label", "dag_consolidate",
        ])
        outcomes = _log_outcomes(mod)
    assert rc == 0, "declared intentional shrink must commit"
    assert _head(repo) != before
    assert "protected_append_doc_shrink_allowed" in outcomes


def test_protected_doc_shrink_placeholder_rationale_refused(tmp_path: Path) -> None:
    """A placeholder rationale ('<rationale>') does NOT satisfy the escape hatch
    — the guard's own docstring example cannot self-waive."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    rel = ".omx/research/inverse_solve_completeness_matrix_20260718.md"
    _seed_protected_doc(repo, rel, 40)
    (repo / rel).write_text("".join(f"row {i}\n" for i in range(10)))
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "clobber with a placeholder rationale",
            "--files", rel,
            "--allow-shared-doc-shrink", "<rationale>",
            "--no-sister-checkpoint-check",
            "--label", "placeholder_refuse",
        ])
        outcomes = _log_outcomes(mod)
    assert rc == 14, "placeholder rationale must NOT waive the guard"
    assert "protected_append_doc_clobber_refused" in outcomes


def test_protected_doc_append_grows_commits_fine(tmp_path: Path) -> None:
    """Negative case: a protected doc that GROWS (normal append) commits fine —
    the guard fires only on a net line LOSS, never on the common append."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    rel = ".omx/research/inverse_solve_completeness_matrix_20260718.md"
    _seed_protected_doc(repo, rel, 40)
    before = _head(repo)
    (repo / rel).write_text("".join(f"row {i}\n" for i in range(55)))  # +15
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "append 15 factor rows to the matrix",
            "--files", rel,
            "--no-sister-checkpoint-check",
            "--label", "matrix_append",
        ])
    assert rc == 0, "a normal append to a protected doc must succeed"
    assert _head(repo) != before


def test_non_protected_doc_shrink_commits_fine(tmp_path: Path) -> None:
    """Scope check: a NON-protected file that shrinks by many lines commits fine
    — the clobber guard is scoped to the named multi-writer append docs only."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    rel = "src/tac/some_ordinary_module.py"
    _seed_protected_doc(repo, rel, 40)  # ordinary file, seeded the same way
    before = _head(repo)
    (repo / rel).write_text("".join(f"row {i}\n" for i in range(5)))  # -35
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "legit big deletion in an ordinary module",
            "--files", rel,
            "--no-sister-checkpoint-check",
            "--label", "ordinary_shrink",
        ])
    assert rc == 0, "ordinary files are out of the protected-doc guard's scope"
    assert _head(repo) != before


def test_new_protected_doc_not_on_head_commits_fine(tmp_path: Path) -> None:
    """A brand-new protected doc (not yet on HEAD) cannot be a clobber — the
    guard skips files with no HEAD blob to compare against."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    rel = ".omx/research/inverse_solve_completeness_matrix_new.md"
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text("row 0\n")  # tiny, but brand-new
    before = _head(repo)
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "create a brand-new matrix doc",
            "--files", rel,
            "--no-sister-checkpoint-check",
            "--label", "new_matrix",
        ])
    assert rc == 0, "a brand-new protected doc must commit (nothing to clobber)"
    assert _head(repo) != before


# ---------------------------------------------------------------------------
# Review F1 (2026-07-18): the 4 BYPASSES the fresh-eyes review found — the
# guards only ran on the --files staging path (`not patch_mode`) and skipped
# deletions, and the placeholder set missed 'TODO'. These pin the closures.
# ---------------------------------------------------------------------------

def _make_head_patch(repo: Path, patch_name: str) -> Path:
    """Build a HEAD-based unified-diff patch from the CURRENT working-tree state
    (force-staging even gitignored files), then restore the index to HEAD so the
    patch is the only carrier of the change. Mirrors how a real --patch-file
    caller regenerates a patch against HEAD."""
    assert _git(repo, "add", "-f", "-A").returncode == 0
    diff = _git(repo, "diff", "--cached", "HEAD")
    assert diff.returncode == 0, diff.stderr
    patch = repo / patch_name
    patch.write_text(diff.stdout)
    _git(repo, "reset", "-q")  # unstage; the patch now owns the change
    return patch


def test_patch_mode_gitignored_add_refused_rc13(tmp_path: Path) -> None:
    """BYPASS 1: `git apply --cached` stages regardless of .gitignore, so a patch
    that ADDS a gitignored bulk file slipped past the --files-only rc=13 guard.
    The patch-mode guard now refuses it."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    (repo / "storage_plan.json").write_text('{"bulk": true}\n')  # gitignored
    patch = _make_head_patch(repo, "add_ignored.patch")
    before = _head(repo)
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "patch that adds a gitignored bulk file",
            "--patch-file", str(patch),
            "--no-sister-checkpoint-check",
            "--label", "patch_gitignore_refuse",
        ])
        outcomes = _log_outcomes(mod)
    assert rc == 13, "patch-mode gitignored add must be refused with rc=13"
    assert _head(repo) == before, "HEAD must NOT move on refusal"
    assert "gitignored_files_in_commit_refused" in outcomes


def test_patch_mode_protected_doc_shrink_refused_rc14(tmp_path: Path) -> None:
    """BYPASS 2: a patch that SHRINKS a protected append doc off a stale base
    skipped the --files-only rc=14 guard. The patch-mode guard now refuses it."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    rel = ".omx/research/inverse_solve_completeness_matrix_20260718.md"
    _seed_protected_doc(repo, rel, 40)
    (repo / rel).write_text("".join(f"row {i}\n" for i in range(10)))  # -30
    patch = _make_head_patch(repo, "shrink_matrix.patch")
    before = _head(repo)
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "patch that clobbers the matrix off a stale base",
            "--patch-file", str(patch),
            "--no-sister-checkpoint-check",
            "--label", "patch_matrix_clobber",
        ])
        outcomes = _log_outcomes(mod)
    assert rc == 14, "patch-mode protected-doc shrink must be refused with rc=14"
    assert _head(repo) == before, "HEAD must NOT move on refusal"
    assert "protected_append_doc_clobber_refused" in outcomes


def test_files_deletion_of_protected_doc_refused_rc14(tmp_path: Path) -> None:
    """BYPASS 3: deleting a protected append doc via --files hit the shrink
    check's staged-absent `continue` and was silently allowed. Deletion is the
    MAXIMAL clobber — it now refuses with rc=14 (override still available)."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    rel = "docs/sub015_DAG_topaiml_reopen.md"
    _seed_protected_doc(repo, rel, 40)
    before = _head(repo)
    (repo / rel).unlink()  # delete the protected doc
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "delete the DAG doc",
            "--files", rel,
            "--no-sister-checkpoint-check",
            "--label", "dag_delete",
        ])
        outcomes = _log_outcomes(mod)
    assert rc == 14, "deletion of a protected append doc must be refused with rc=14"
    assert _head(repo) == before, "HEAD must NOT move on refusal"
    assert "protected_append_doc_clobber_refused" in outcomes


def test_files_deletion_of_protected_doc_allowed_with_rationale(tmp_path: Path) -> None:
    """The deletion refusal is overridable: a real --allow-shared-doc-shrink
    rationale lets a deliberate removal through (retirement/consolidation)."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    rel = "docs/sub015_DAG_topaiml_reopen.md"
    _seed_protected_doc(repo, rel, 40)
    before = _head(repo)
    (repo / rel).unlink()
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "retire the superseded DAG doc",
            "--files", rel,
            "--allow-shared-doc-shrink", "retiring the superseded 2026 DAG doc",
            "--no-sister-checkpoint-check",
            "--label", "dag_retire",
        ])
        outcomes = _log_outcomes(mod)
    assert rc == 0, "declared intentional deletion must commit"
    assert _head(repo) != before
    assert "protected_append_doc_shrink_allowed" in outcomes


def test_patch_mode_normal_change_still_commits(tmp_path: Path) -> None:
    """CRITICAL positive case: a legit patch (new normal file, no gitignored add,
    no protected-doc shrink) must STILL commit rc=0 — the new patch-mode guards
    must not false-positive and brick the intent-manifest path."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    (repo / "normal_note.md").write_text("a perfectly ordinary note\nline two\n")
    patch = _make_head_patch(repo, "normal.patch")
    before = _head(repo)
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "patch that adds an ordinary tracked file",
            "--patch-file", str(patch),
            "--no-sister-checkpoint-check",
            "--label", "patch_normal_ok",
        ])
    assert rc == 0, "a legit patch-mode commit must succeed (no false-positive)"
    assert _head(repo) != before, "HEAD must advance on a real patch commit"


def test_patch_mode_protected_doc_append_still_commits(tmp_path: Path) -> None:
    """A patch that GROWS a protected append doc (the common case) must commit —
    the patch-mode shrink guard fires only on a net line LOSS, never an append."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    rel = "x_DAG_FEED_2026.md"
    _seed_protected_doc(repo, rel, 20)
    (repo / rel).write_text("".join(f"row {i}\n" for i in range(35)))  # +15
    patch = _make_head_patch(repo, "append_feed.patch")
    before = _head(repo)
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "patch that appends DAG FEED rows",
            "--patch-file", str(patch),
            "--no-sister-checkpoint-check",
            "--label", "patch_feed_append",
        ])
    assert rc == 0, "a patch-mode append to a protected doc must succeed"
    assert _head(repo) != before


def test_todo_placeholder_rationale_refused(tmp_path: Path) -> None:
    """BYPASS 4: 'TODO' (and fixme/xxx/wip/none/...) are placeholder stubs, not a
    real reason — they must NOT satisfy the --allow-shared-doc-shrink escape."""
    repo = tmp_path / "repo"
    _make_throwaway_repo(repo)
    rel = ".omx/research/inverse_solve_completeness_matrix_20260718.md"
    _seed_protected_doc(repo, rel, 40)
    (repo / rel).write_text("".join(f"row {i}\n" for i in range(10)))
    mod = _load_module()
    with _Patched(mod, repo):
        rc = _run_main(mod, [
            "--message", "clobber with a TODO placeholder rationale",
            "--files", rel,
            "--allow-shared-doc-shrink", "TODO",
            "--no-sister-checkpoint-check",
            "--label", "todo_placeholder_refuse",
        ])
        outcomes = _log_outcomes(mod)
    assert rc == 14, "'TODO' placeholder must NOT waive the guard"
    assert "protected_append_doc_clobber_refused" in outcomes


# ---------------------------------------------------------------------------
# Unit-level: the helper predicates.
# ---------------------------------------------------------------------------

def test_is_protected_append_doc_markers() -> None:
    mod = _load_module()
    assert mod._is_protected_append_doc(
        ".omx/research/inverse_solve_completeness_matrix_20260718.md"
    )
    assert mod._is_protected_append_doc("docs/sub015_DAG_x.md")
    assert mod._is_protected_append_doc("x_DAG_FEED_2026.md")
    assert mod._is_protected_append_doc(
        ".omx/state/canonical_equations_registry.jsonl"
    )
    assert not mod._is_protected_append_doc("src/tac/foo.py")
    assert not mod._is_protected_append_doc("README.md")
