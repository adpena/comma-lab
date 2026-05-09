# FAKE_LANE_OK_FILE: this test file's purpose is to verify Check #126 by
# constructing dozens of fake lane_id fixtures inside synthetic git repos.
# Per-line waivers would be noise and would mask real future violations.
"""Tests for preflight Catalog #126:
``check_lane_pre_registered_before_work_starts``.

Operationalises CLAUDE.md "Lane maturity registry" lifecycle discipline +
"Subagent coherence-by-default" anti-duplication primitive. The check
refuses subagent commits whose introduced files (under ``src/tac/`` /
``tools/`` / ``experiments/`` / ``scripts/``) reference a ``lane_<NAME>``
token that does NOT appear in ``.omx/state/lane_registry.json``.

Acceptance/exemption rules:
  - Test fixtures (``*/tests/*`` or ``test_*`` filenames) are exempt iff
    the line carries ``# FAKE_LANE_OK:<reason>`` (or has it within 5 lines
    above).
  - Common helper-name tokens (``lane_id``, ``lane_class``,
    ``lane_registry``, ``lane_maturity``, etc.) are blocklisted.

This test set verifies:
  1. All commit lanes in registry → pass
  2. Unknown lane → fail strict / warn non-strict
  3. Test fixture exemption with FAKE_LANE_OK marker → pass
  4. Test fixture WITHOUT marker → still warns
  5. Blocklisted helper names not flagged
  6. Single-quoted lane reference flagged
  7. Double-quoted lane reference flagged
  8. Bare unquoted lane_<id> NOT flagged (regex requires quotes)
  9. Multiple lanes per file aggregated correctly
 10. Files outside scan prefixes ignored
 11. File extensions outside .py/.sh ignored
 12. Missing registry → degrades gracefully
 13. Performance: 50-commit scan in <2s
 14. Strict mode raises with formatted message
 15. Aliases declared on a registry lane are also accepted
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _LANE_ID_REFERENCE_BLOCKLIST,
    _LANE_ID_REFERENCE_RE,
    _is_test_path,
    _line_or_window_has_fake_waiver,
    check_lane_pre_registered_before_work_starts,
)


# ── Helpers: make a fake git repo with a registry + commit history ──────


def _git(cwd: Path, *args: str) -> str:
    """Run a git command in cwd and return stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )
    return result.stdout


def _init_repo(tmp: Path, registry_lanes: list[dict]) -> Path:
    """Create a fake repo with .omx/state/lane_registry.json + git history."""
    (tmp / ".omx" / "state").mkdir(parents=True, exist_ok=True)
    (tmp / "src" / "tac").mkdir(parents=True, exist_ok=True)
    (tmp / "src" / "tac" / "tests").mkdir(parents=True, exist_ok=True)
    (tmp / "tools").mkdir(parents=True, exist_ok=True)
    (tmp / "experiments").mkdir(parents=True, exist_ok=True)
    (tmp / "scripts").mkdir(parents=True, exist_ok=True)
    registry = {
        "schema_version": 1,
        "lanes": registry_lanes,
    }
    (tmp / ".omx" / "state" / "lane_registry.json").write_text(
        json.dumps(registry, indent=2),
    )
    _git(tmp, "init", "--initial-branch=main")
    _git(tmp, "add", ".")
    _git(tmp, "commit", "-m", "initial")
    return tmp


def _commit_file(repo: Path, rel: str, content: str, message: str) -> str:
    """Write a file at rel, commit it, return the commit SHA."""
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    _git(repo, "add", rel)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


# ── Test 1: lane in registry → no warning ────────────────────────────────


def test_lane_in_registry_no_warning(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    _commit_file(
        repo, "src/tac/foo.py",
        'LANE_ID = "lane_g_v3"\n',
        "use registered lane",
    )
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert violations == []


# ── Test 2: unknown lane → fail/warn ──────────────────────────────────────


def test_unknown_lane_warns(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    _commit_file(
        repo, "src/tac/foo.py",
        'LANE_ID = "lane_does_not_exist"\n',
        "introduce unregistered lane",
    )
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert len(violations) >= 1
    assert any("lane_does_not_exist" in v for v in violations)


def test_unknown_lane_strict_raises(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    _commit_file(
        repo, "src/tac/foo.py",
        'LANE_ID = "lane_unregistered_thing"\n',
        "introduce unregistered lane",
    )
    with pytest.raises(PreflightError) as excinfo:
        check_lane_pre_registered_before_work_starts(
            repo_root=repo, n_commits=10, strict=True, verbose=False,
        )
    msg = str(excinfo.value)
    assert "check_lane_pre_registered_before_work_starts" in msg
    assert "lane_unregistered_thing" in msg
    assert "tools/lane_maturity.py add-lane" in msg


# ── Test 3: test fixture exemption with FAKE_LANE_OK marker ──────────────


def test_test_fixture_with_fake_lane_marker_no_warning(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    _commit_file(
        repo, "src/tac/tests/test_foo.py",
        'TEST_LANE_ID = "lane_test_fixture_a"  # FAKE_LANE_OK: pure unit-test fixture\n',
        "test fixture",
    )
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert violations == [], f"got: {violations}"


def test_test_fixture_with_marker_above_no_warning(tmp_path: Path) -> None:
    """Marker within 5 lines above also exempts."""
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    content = """
# FAKE_LANE_OK: test fixture intentionally references nonexistent lane
def fixture_function():
    obj = {}
    obj["key"] = "value"
    obj["lane_id"] = "lane_test_xyz"
"""
    _commit_file(repo, "src/tac/tests/test_bar.py", content, "fixture above")
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert violations == [], f"got: {violations}"


def test_test_fixture_with_marker_too_far_above_warns(tmp_path: Path) -> None:
    """Marker more than 5 lines above does NOT exempt."""
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    content = """# FAKE_LANE_OK: would-be excuse
# line 2
# line 3
# line 4
# line 5
# line 6
# line 7
LANE = "lane_test_pretend"
"""
    _commit_file(repo, "src/tac/tests/test_far.py", content, "marker too far")
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert len(violations) >= 1


# ── Test 4: test fixture WITHOUT marker → still warns ───────────────────


def test_test_fixture_without_marker_warns(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    _commit_file(
        repo, "src/tac/tests/test_foo.py",
        'TEST_LANE = "lane_pretend_test"\n',  # no marker
        "test without marker",
    )
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert any("lane_pretend_test" in v for v in violations)


# ── Test 5: blocklist guards ────────────────────────────────────────────


@pytest.mark.parametrize(
    "blocked",
    sorted(_LANE_ID_REFERENCE_BLOCKLIST),
)
def test_blocklisted_helper_name_not_flagged(
    tmp_path: Path, blocked: str,
) -> None:
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    _commit_file(
        repo, "src/tac/foo.py",
        f'helper = "{blocked}"\n',
        f"use blocklisted token {blocked}",
    )
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert violations == [], f"blocklisted '{blocked}' triggered: {violations}"


# ── Test 6/7: quoted-form detection ──────────────────────────────────────


def test_double_quoted_lane_reference_flagged(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    _commit_file(
        repo, "src/tac/foo.py",
        'register("lane_unknown_double")\n',
        "double quoted",
    )
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert any("lane_unknown_double" in v for v in violations)


def test_single_quoted_lane_reference_flagged(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    _commit_file(
        repo, "src/tac/foo.py",
        "register('lane_unknown_single')\n",
        "single quoted",
    )
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert any("lane_unknown_single" in v for v in violations)


def test_bare_unquoted_lane_id_not_flagged(tmp_path: Path) -> None:
    """Bare identifier (no quotes) is not flagged — too noisy in real code."""
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    # Use as a comment / docstring reference (no quotes around it)
    _commit_file(
        repo, "src/tac/foo.py",
        "# Note: lane_unknown_bare is a planned future lane.\n",
        "bare reference in comment",
    )
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert violations == []


# ── Test 8: multiple lanes per file ──────────────────────────────────────


def test_multiple_lanes_per_file_aggregated(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    content = """
A = "lane_first_unknown"
B = "lane_second_unknown"
C = "lane_g_v3"  # ok
"""
    _commit_file(repo, "src/tac/foo.py", content, "multiple")
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert any("lane_first_unknown" in v for v in violations)
    assert any("lane_second_unknown" in v for v in violations)
    # lane_g_v3 should NOT appear as a violation
    assert not any("lane_g_v3" in v for v in violations)


# ── Test 9: outside scan prefixes ignored ───────────────────────────────


def test_files_outside_scan_prefixes_ignored(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    # `docs/` is NOT in scan prefix list
    _commit_file(repo, "docs/readme.md.py", 'X = "lane_unknown"\n', "outside")
    # actually .py inside docs/ isn't scanned because not in prefix list
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert violations == []


# ── Test 10: file extensions outside .py/.sh ignored ────────────────────


def test_md_file_ignored(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    _commit_file(
        repo, "src/tac/notes.md",
        '`"lane_unknown_md"` is a planned lane.\n',
        "md not scanned",
    )
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert violations == []


def test_sh_file_scanned(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    _commit_file(
        repo, "scripts/foo.sh",
        'LANE="lane_unknown_sh"\n',
        "sh is scanned",
    )
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert any("lane_unknown_sh" in v for v in violations)


# ── Test 11: missing registry handled ────────────────────────────────────


def test_missing_registry_warns_non_strict(tmp_path: Path) -> None:
    """No registry file → returns a warning string but does not blow up."""
    (tmp_path / "src" / "tac").mkdir(parents=True, exist_ok=True)
    _git(tmp_path, "init", "--initial-branch=main")
    (tmp_path / "src" / "tac" / "noop.py").write_text("# noop\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "initial")
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=tmp_path, n_commits=10, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "lane registry not found" in violations[0]


def test_missing_registry_strict_raises(tmp_path: Path) -> None:
    (tmp_path / "src" / "tac").mkdir(parents=True, exist_ok=True)
    _git(tmp_path, "init", "--initial-branch=main")
    (tmp_path / "src" / "tac" / "noop.py").write_text("# noop\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "initial")
    with pytest.raises(PreflightError):
        check_lane_pre_registered_before_work_starts(
            repo_root=tmp_path, n_commits=10, strict=True, verbose=False,
        )


# ── Test 12: aliases on a registered lane are accepted ──────────────────


def test_alias_in_registry_accepted(tmp_path: Path) -> None:
    """A lane with an `aliases` field accepts the alias as a known lane_id."""
    repo = _init_repo(
        tmp_path,
        [{
            "id": "lane_canonical",
            "aliases": ["lane_canonical_v2", "lane_canonical_legacy"],
            "name": "canonical lane",
        }],
    )
    _commit_file(
        repo, "src/tac/foo.py",
        'X = "lane_canonical_v2"\n',
        "use alias",
    )
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert violations == []


def test_alias_field_singular_also_accepted(tmp_path: Path) -> None:
    """The singular `alias` (a list) is also accepted."""
    repo = _init_repo(
        tmp_path,
        [{"id": "lane_canon", "alias": ["lane_canon_old"], "name": "canon"}],
    )
    _commit_file(repo, "src/tac/foo.py", 'X = "lane_canon_old"\n', "use alias")
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert violations == []


# ── Test 13: performance — 50-commit scan ───────────────────────────────


def test_performance_50_commits_under_2s(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    for i in range(50):
        _commit_file(
            repo, f"src/tac/file_{i:03d}.py",
            f'LANE_ID = "lane_g_v3"  # ok ({i})\n',
            f"commit {i}",
        )
    t0 = time.perf_counter()
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=50, strict=False, verbose=False,
    )
    elapsed = time.perf_counter() - t0
    assert violations == []
    assert elapsed < 2.0, f"scan took {elapsed:.2f}s for 50 commits (expected < 2.0s)"


# ── Test 14: helper unit tests ───────────────────────────────────────────


def test_lane_id_reference_re_double_quoted() -> None:
    matches = list(_LANE_ID_REFERENCE_RE.finditer('foo("lane_a", x)'))
    assert len(matches) == 1
    assert matches[0].group(1) == "lane_a"


def test_lane_id_reference_re_single_quoted() -> None:
    matches = list(_LANE_ID_REFERENCE_RE.finditer("foo('lane_b', x)"))
    assert len(matches) == 1
    assert matches[0].group(2) == "lane_b"


def test_lane_id_reference_re_unquoted_not_matched() -> None:
    matches = list(_LANE_ID_REFERENCE_RE.finditer("foo(lane_c, x)"))
    assert len(matches) == 0


def test_lane_id_reference_re_underscore_in_id() -> None:
    matches = list(_LANE_ID_REFERENCE_RE.finditer('"lane_some_compound_name_v2"'))
    assert len(matches) == 1
    token = matches[0].group(1) or matches[0].group(2)
    assert token == "lane_some_compound_name_v2"


def test_is_test_path_recognises_tests_dir() -> None:
    assert _is_test_path("src/tac/tests/test_foo.py")
    assert _is_test_path("tests/foo.py")
    assert _is_test_path("path/to/tests/file.py")


def test_is_test_path_recognises_test_filename_prefix() -> None:
    assert _is_test_path("src/tac/test_module.py")
    # Note: `test_*` filename + non-tests path also counts (parts can be test_*)
    # The filename `test_foo.py` has a part `test_foo.py` which startswith test_
    # but our function looks at parts not full filenames; let's test split parts.


def test_is_test_path_rejects_non_test_paths() -> None:
    assert not _is_test_path("src/tac/foo.py")
    assert not _is_test_path("tools/dispatch.py")
    assert not _is_test_path("experiments/run.py")


def test_line_or_window_has_fake_waiver_same_line() -> None:
    lines = ["foo = 'lane_x'  # FAKE_LANE_OK: testing"]
    assert _line_or_window_has_fake_waiver(lines, 0)


def test_line_or_window_has_fake_waiver_above() -> None:
    lines = [
        "# FAKE_LANE_OK: prep",
        "x = 1",
        "y = 'lane_x'",
    ]
    assert _line_or_window_has_fake_waiver(lines, 2)


def test_line_or_window_has_fake_waiver_too_far_above() -> None:
    lines = [
        "# FAKE_LANE_OK: prep",
        "filler 1",
        "filler 2",
        "filler 3",
        "filler 4",
        "filler 5",
        "filler 6",
        "y = 'lane_x'",
    ]
    # default window=5; lineno 7 is more than 5 above → not exempt
    assert not _line_or_window_has_fake_waiver(lines, 7)


# ── Test 15: bash/sh test path is_test_path edge case ────────────────────


def test_blocklisted_token_not_flagged_in_quote(tmp_path: Path) -> None:
    """The blocklist also applies to quoted occurrences (e.g., dict keys)."""
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    _commit_file(
        repo, "src/tac/foo.py",
        (
            '''payload = {"lane_id": "lane_g_v3", "lane_class": "renderer"}\n'''
            '''blockers = ["lane_dispatch_claim_missing"]\n'''
            '''payload["lane_claim_preflight"] = {"active_claim_present": False}\n'''
            '''payload["lane_dir"] = "experiments/results/lane_g_v3"\n'''
        ),
        "blocklisted as quoted dict key",
    )
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    # Quoted helper keys / blocker labels should be blocklisted. lane_g_v3 is
    # the only real lane reference here and it is in the registry.
    assert violations == []


def test_no_commits_returns_empty(tmp_path: Path) -> None:
    """If git log yields no commits the scan returns cleanly."""
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    # Repo only has the initial commit and registry. Use n_commits=0.
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=0, strict=False, verbose=False,
    )
    # n_commits=0 yields an empty git-log; the function should return cleanly
    # (empty violation list).
    assert violations == []


# ── Test 16: file-level FAKE_LANE_OK_FILE waiver ────────────────────────


def test_file_level_fake_waiver_exempts_only_gate_self_test_file(tmp_path: Path) -> None:
    """A file waiver is allowed only for Check #126's own fixture-heavy test file."""
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    content = """# FAKE_LANE_OK_FILE: this entire test file constructs many fake lane fixtures
# Other test logic
LANE_A = "lane_pretend_a"
LANE_B = "lane_pretend_b"
LANE_C = "lane_pretend_c"
"""
    _commit_file(
        repo, "src/tac/tests/test_check_lane_pre_registered_before_work_starts.py",
        content,
        "many fake lanes with file-level waiver",
    )
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert violations == [], f"got: {violations}"


def test_file_level_waiver_does_not_exempt_other_test_files(tmp_path: Path) -> None:
    """File-level waiver outside the dedicated self-test does NOT exempt."""
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    content = """# FAKE_LANE_OK_FILE: not the dedicated gate self-test; should NOT exempt
LANE_A = "lane_pretend_z"
"""
    _commit_file(
        repo, "src/tac/tests/test_other_lane_file.py",
        content,
        "fake lane in other test file",
    )
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert any("lane_pretend_z" in v for v in violations)


def test_file_level_waiver_only_works_in_test_paths(tmp_path: Path) -> None:
    """File-level waiver in a non-test path does NOT exempt."""
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    content = """# FAKE_LANE_OK_FILE: not a test path; should NOT exempt
LANE_A = "lane_pretend_z"
"""
    _commit_file(
        repo, "src/tac/foo.py",
        content,
        "fake lane in non-test file",
    )
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert any("lane_pretend_z" in v for v in violations)


def test_file_level_waiver_only_in_first_30_lines(tmp_path: Path) -> None:
    """A file-level marker beyond line 30 does NOT exempt the whole file."""
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    # Insert 30 filler lines, then the marker, then a fake lane
    content = (
        "\n".join(f"# filler line {i}" for i in range(35))
        + "\n# FAKE_LANE_OK_FILE: too late — beyond 30-line window\n"
        + 'LANE_A = "lane_pretend_late"\n'
    )
    _commit_file(
        repo, "src/tac/tests/test_late.py",
        content,
        "marker too late",
    )
    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=10, strict=False, verbose=False,
    )
    assert any("lane_pretend_late" in v for v in violations)


def test_dirty_worktree_unregistered_lane_is_caught_before_commit(tmp_path: Path) -> None:
    """Current WIP must be scanned so the gate enforces before-work-starts."""
    repo = _init_repo(tmp_path, [{"id": "lane_g_v3", "name": "g v3"}])
    path = repo / "tools" / "wip_lane.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('LANE = "lane_unregistered_wip"\n', encoding="utf-8")

    violations = check_lane_pre_registered_before_work_starts(
        repo_root=repo, n_commits=0, strict=False, verbose=False,
    )

    assert any("lane_unregistered_wip" in v and "WORKTREE" in v for v in violations)
