# SPDX-License-Identifier: MIT
"""Tests for CANONICALIZATION UNIT 2 (task #388) triality-legs disposition.

Three surfaces:
  * serializer ``_parse_triality_legs`` — parse/validate (pos + neg).
  * serializer end-to-end — the flag is recorded in the JSONL log row, AND the
    flag being ABSENT produces byte-identical git behavior (rc=0 commit, log row
    with NO triality keys).
  * drift-detector ``triality_disposition_from_rows`` — softening logic (matches a
    committed row's declared legs to a window sha; None otherwise).

All commit tests run against throwaway git repos under tmp_path — NEVER the real
repo (the serializer globals are patched per-test).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SCS = _load("_scs_triality", "tools/subagent_commit_serializer.py")
DDET = _load("_ddet_triality", "tools/triality_drift_detector.py")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )


def _make_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    for cmd in (
        ["init"],
        ["config", "user.email", "test@example.invalid"],
        ["config", "user.name", "TrialityTest"],
        ["config", "commit.gpgsign", "false"],
        ["config", "core.hooksPath", "/dev/null"],  # no pre-commit hook in scratch
    ):
        assert _git(repo, *cmd).returncode == 0
    (repo / "seed.txt").write_bytes(b"seed\n")
    assert _git(repo, "add", "seed.txt").returncode == 0
    assert _git(repo, "commit", "-m", "seed").returncode == 0


class _Patched:
    def __init__(self, repo: Path):
        self.repo = repo

    def __enter__(self):
        self.old = (SCS.REPO_ROOT, SCS.LOCK_PATH, SCS.LOG_PATH)
        SCS.REPO_ROOT = self.repo
        SCS.LOCK_PATH = self.repo / ".commit-lock"
        SCS.LOG_PATH = self.repo / "commit-serializer.log"
        return self

    def __exit__(self, *exc):
        SCS.REPO_ROOT, SCS.LOCK_PATH, SCS.LOG_PATH = self.old
        return False


def _run_main(argv: list[str]) -> int:
    old = sys.argv[:]
    sys.argv = ["subagent_commit_serializer.py", *argv]
    try:
        return SCS.main()
    finally:
        sys.argv = old


def _log_rows(repo: Path) -> list[dict]:
    p = repo / "commit-serializer.log"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# =========================== _parse_triality_legs ==============================
def test_parse_absent_flag_is_noop():
    assert SCS._parse_triality_legs(None, None) == (None, None)
    # a reason without legs is ignored when legs absent (no flag)
    assert SCS._parse_triality_legs(None, "whatever") == (None, None)


def test_parse_legs_subset_ok():
    assert SCS._parse_triality_legs("dag,equations", None) == (["dag", "equations"], None)


def test_parse_legs_dedupe_preserves_order():
    assert SCS._parse_triality_legs("dsl,dag,dsl", None) == (["dsl", "dag"], None)


def test_parse_none_requires_reason():
    import pytest
    with pytest.raises(ValueError):
        SCS._parse_triality_legs("none", None)
    with pytest.raises(ValueError):
        SCS._parse_triality_legs("none", "   ")
    assert SCS._parse_triality_legs("none", "pure apparatus chore") == (["none"], "pure apparatus chore")


def test_parse_none_cannot_combine():
    import pytest
    with pytest.raises(ValueError):
        SCS._parse_triality_legs("dag,none", "r")


def test_parse_rejects_unknown_and_empty():
    import pytest
    with pytest.raises(ValueError):
        SCS._parse_triality_legs("bogus", None)
    with pytest.raises(ValueError):
        SCS._parse_triality_legs(",", None)


# =========================== serializer end-to-end =============================
def test_flag_recorded_in_log_row(tmp_path):
    repo = tmp_path / "r"
    _make_repo(repo)
    (repo / "f.txt").write_bytes(b"x\n")
    with _Patched(repo):
        rc = _run_main([
            "--message", "witness: lever wire-in",
            "--files", "f.txt",
            "--no-sister-checkpoint-check",
            "--triality-legs", "dag,dsl,equations",
        ])
    assert rc == 0
    committed = [r for r in _log_rows(repo) if r.get("outcome") == "committed"]
    assert committed and committed[-1]["triality_legs"] == ["dag", "dsl", "equations"]
    assert committed[-1]["triality_reason"] is None


def test_none_reason_recorded(tmp_path):
    repo = tmp_path / "r"
    _make_repo(repo)
    (repo / "f.txt").write_bytes(b"x\n")
    with _Patched(repo):
        rc = _run_main([
            "--message", "chore: reflow",
            "--files", "f.txt",
            "--no-sister-checkpoint-check",
            "--triality-legs", "none",
            "--triality-reason", "pure formatting chore",
        ])
    assert rc == 0
    row = [r for r in _log_rows(repo) if r.get("outcome") == "committed"][-1]
    assert row["triality_legs"] == ["none"]
    assert row["triality_reason"] == "pure formatting chore"


def test_absent_flag_identical_behavior_and_no_triality_keys(tmp_path):
    """The backward-compat proof: without --triality-legs the commit still lands
    (rc=0) AND the committed log row carries NO triality keys (byte-identical
    row shape for legacy callers)."""
    repo = tmp_path / "r"
    _make_repo(repo)
    (repo / "f.txt").write_bytes(b"x\n")
    with _Patched(repo):
        rc = _run_main([
            "--message", "witness: lever wire-in",
            "--files", "f.txt",
            "--no-sister-checkpoint-check",
        ])
    assert rc == 0
    row = [r for r in _log_rows(repo) if r.get("outcome") == "committed"][-1]
    assert "triality_legs" not in row
    assert "triality_reason" not in row
    # commit really landed
    assert _git(repo, "cat-file", "-e", "HEAD:f.txt").returncode == 0


def test_malformed_flag_refuses_before_any_git_action(tmp_path):
    """'--triality-legs none' without a reason is rc=2 (FATAL parse) and MUST NOT
    create a commit — validation happens before staging."""
    repo = tmp_path / "r"
    _make_repo(repo)
    head_before = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "f.txt").write_bytes(b"x\n")
    with _Patched(repo):
        rc = _run_main([
            "--message", "x",
            "--files", "f.txt",
            "--no-sister-checkpoint-check",
            "--triality-legs", "none",  # missing --triality-reason
        ])
    assert rc == 2
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == head_before


# =========================== drift-detector softening ==========================
def test_disposition_matches_committed_row_in_window():
    rows = [
        {"outcome": "committed", "triality_legs": ["dag"], "triality_reason": None,
         "head_after": "abc1234"},
    ]
    window = {"abc1234def567890"}  # full sha of which the row's head is a prefix
    disp = DDET.triality_disposition_from_rows(rows, window)
    assert disp is not None
    assert disp["legs"] == ["dag"]
    assert disp["head"] == "abc1234"


def test_disposition_none_when_row_outside_window():
    rows = [{"outcome": "committed", "triality_legs": ["dag"], "head_after": "deadbee"}]
    window = {"ffffffffffffffff"}  # unrelated commit
    assert DDET.triality_disposition_from_rows(rows, window) is None


def test_disposition_none_when_no_legs_declared():
    rows = [{"outcome": "committed", "head_after": "abc1234"}]  # legacy row, no legs
    assert DDET.triality_disposition_from_rows(rows, {"abc1234000"}) is None


def test_disposition_ignores_noncommitted_rows():
    rows = [
        {"outcome": "expected_content_sha_mismatch", "triality_legs": ["dag"], "head_after": "abc1234"},
    ]
    assert DDET.triality_disposition_from_rows(rows, {"abc1234000"}) is None


def test_disposition_latest_row_wins():
    rows = [
        {"outcome": "committed", "triality_legs": ["dag"], "head_after": "aaa1111"},
        {"outcome": "committed", "triality_legs": ["none"], "triality_reason": "chore",
         "head_after": "bbb2222"},
    ]
    window = {"aaa1111aaa", "bbb2222bbb"}
    disp = DDET.triality_disposition_from_rows(rows, window)
    assert disp["legs"] == ["none"]
    assert disp["reason"] == "chore"


def test_disposition_empty_window_matches_on_presence():
    # window unresolved (e.g. git error) → a declared row still counts (fail toward
    # softening, per the hook's fail-open philosophy).
    rows = [{"outcome": "committed", "triality_legs": ["equations"], "head_after": "abc"}]
    assert DDET.triality_disposition_from_rows(rows, set())["legs"] == ["equations"]


def test_disposition_robust_to_malformed_rows():
    rows = ["not a dict", {"outcome": "committed"}, 42,
            {"outcome": "committed", "triality_legs": ["dsl"], "head_after": "cc11"}]
    assert DDET.triality_disposition_from_rows(rows, {"cc11aa"})["legs"] == ["dsl"]


def test_read_serializer_rows_missing_file_is_empty(tmp_path):
    assert DDET._read_serializer_rows(str(tmp_path)) == []


# =================== drift-detector main() live softening ======================
_DDET_TOOL = REPO / "tools" / "triality_drift_detector.py"


def _drift_repo(repo: Path) -> tuple[str, str]:
    """Scratch repo with a seed commit + a DRIFTING commit (a 'lever' subject that
    touches no triality leg). Returns (parent_sha, head_sha)."""
    _make_repo(repo)
    parent = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "src").mkdir(exist_ok=True)
    (repo / "src" / "boundary.py").write_bytes(b"x = 1\n")
    assert _git(repo, "add", "src/boundary.py").returncode == 0
    assert _git(
        repo, "commit", "-m", "witness: island-birth lever measured d_seg 0.0031"
    ).returncode == 0
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    return parent, head


def _write_marker(repo: Path, last_head: str) -> None:
    d = repo / ".omx" / "state"
    d.mkdir(parents=True, exist_ok=True)
    (d / "triality_drift_marker.json").write_text(
        json.dumps({"last_head": last_head, "last_block_head": None})
    )


def _run_hook(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_DDET_TOOL)],
        input=json.dumps({"cwd": str(repo)}),
        capture_output=True, text=True, timeout=30, cwd=str(repo),
    )


def test_hook_blocks_drift_without_disposition(tmp_path):
    """Control: a drifting 'lever' commit with NO serializer disposition BLOCKS."""
    repo = tmp_path / "r"
    parent, head = _drift_repo(repo)
    _write_marker(repo, parent)
    proc = _run_hook(repo)
    assert proc.returncode == 0
    assert '"decision": "block"' in proc.stdout


def test_hook_softens_drift_with_serializer_disposition(tmp_path):
    """Treatment: the SAME drift, but a committed serializer-log row for this
    commit declared --triality-legs → the core drift SOFTENS (no block)."""
    repo = tmp_path / "r"
    parent, head = _drift_repo(repo)
    _write_marker(repo, parent)
    short = _git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
    d = repo / ".omx" / "state"
    (d / "commit-serializer.log").write_text(
        json.dumps({
            "outcome": "committed",
            "triality_legs": ["dsl"],
            "triality_reason": None,
            "head_after": short,
        }) + "\n"
    )
    proc = _run_hook(repo)
    assert proc.returncode == 0
    assert '"decision": "block"' not in proc.stdout


def test_hook_softens_with_none_disposition(tmp_path):
    """A structured 'none'+reason disposition also softens (structured [no-triality])."""
    repo = tmp_path / "r"
    parent, head = _drift_repo(repo)
    _write_marker(repo, parent)
    short = _git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
    d = repo / ".omx" / "state"
    (d / "commit-serializer.log").write_text(
        json.dumps({
            "outcome": "committed",
            "triality_legs": ["none"],
            "triality_reason": "pure apparatus chore",
            "head_after": short,
        }) + "\n"
    )
    proc = _run_hook(repo)
    assert proc.returncode == 0
    assert '"decision": "block"' not in proc.stdout
