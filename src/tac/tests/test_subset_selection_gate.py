# SPDX-License-Identifier: MIT
"""Tests for the subset-selection guard (`ddm_ss1`, landing 2).

The load-bearing test in here is :func:`test_positive_control_a_non_conforming_site_is_refused`.
A guard without an executed failing case is not landed -- it is a decoration that
has never been shown to be capable of saying no.

Equally load-bearing, and the reason the sibling Catalog #184 scan needed five
review rounds: a passing unit test proves the UNIT, never the WIRING. So
:func:`test_hook_calls_the_scan_before_preflight` asserts the hook's own call
order by reading `tools/preflight_hook.py`, because a perfect gate that main()
never calls is exactly the vacuity-equals-pass genus.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from tac.subset_selection_gate import (
    WAIVER_MARKER,
    _violations_for_text,
    added_lines,
    find_sites,
    in_scope_py_files,
    scan_staged,
    staged_py_files,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

NON_CONFORMING = """
def go(pairs, n_pairs):
    subset = pairs[:n_pairs]
    return subset
"""

WAIVED = f"""
def go(pairs, n_pairs):
    subset = pairs[:n_pairs]  # {WAIVER_MARKER}: reproducing the n8 prefix of pose_carrier_arms_measured_20260708
    return subset
"""

PLACEHOLDER_WAIVED = f"""
def go(pairs, n_pairs):
    subset = pairs[:n_pairs]  # {WAIVER_MARKER}: <rationale>
    return subset
"""

CANONICAL = """
from tac.subset_selection import MODE_STRATIFIED, select
def go(pairs, n_pairs, population):
    idx = select(n_pairs, population, mode=MODE_STRATIFIED, seed=1).indices
    also = pairs[:n_pairs]
    return [pairs[i] for i in idx], also
"""

BENIGN_BARE_N = """
def go(bits, n):
    return bits[:n]
"""

BENIGN_TOP_K = """
def go(records, k):
    return records[:k]
"""

ATTRIBUTE_FORM = """
def go(pairs, args):
    return pairs[:args.n_pairs]
"""

STRIDED_IS_NOT_A_PREFIX = """
def go(pairs, n_pairs):
    return pairs[::3]
"""


# --- the controls ----------------------------------------------------------


def test_positive_control_a_non_conforming_site_is_refused() -> None:
    """THE positive control: the guard must be able to say no."""
    violations = _violations_for_text(NON_CONFORMING, "fixture.py")
    assert len(violations) == 1
    assert "no declared selection mode" in violations[0]
    assert "fixture.py:3" in violations[0]


def test_positive_control_raises_in_strict_mode(tmp_path: Path) -> None:
    """And refusing must actually RAISE, not merely return a list."""
    from tac.preflight import PreflightError

    offender = tmp_path / "offender.py"
    offender.write_text(NON_CONFORMING, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "offender.py"], cwd=tmp_path, check=True)
    with pytest.raises(PreflightError, match="no declared selection mode"):
        scan_staged(repo_root=tmp_path, strict=True, verbose=False)


def test_negative_control_a_real_waiver_is_honoured() -> None:
    assert _violations_for_text(WAIVED, "fixture.py") == []


def test_a_placeholder_waiver_does_not_waive() -> None:
    """Catalog #287: the docstring's own example must not self-waive."""
    assert len(_violations_for_text(PLACEHOLDER_WAIVED, "fixture.py")) == 1


def test_using_the_canonical_selector_clears_the_file() -> None:
    assert _violations_for_text(CANONICAL, "fixture.py") == []


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("bare [:n]", BENIGN_BARE_N),
        ("top-k [:k]", BENIGN_TOP_K),
        ("stride, not a prefix", STRIDED_IS_NOT_A_PREFIX),
    ],
)
def test_benign_slices_are_not_flagged(label: str, source: str) -> None:
    """No false positives: a gate that fires on ambiguity gets waived to death."""
    assert _violations_for_text(source, "fixture.py") == [], label


def test_attribute_form_is_caught() -> None:
    """``pairs[:args.n_pairs]`` is the same defect wearing a namespace."""
    assert len(_violations_for_text(ATTRIBUTE_FORM, "fixture.py")) == 1


def test_unparseable_source_is_not_silently_clean() -> None:
    """A SyntaxError yields no sites -- but must not be *claimed* as clean.

    The gate returns [] here (it cannot parse), which is why the caller reports a
    denominator: `examined` counts files it actually read, so this file's silence
    is visible as an unexamined file rather than as a pass.
    """
    assert find_sites("def (:", "broken.py") == []


# --- added-lines scoping ---------------------------------------------------


def test_only_added_lines_are_in_scope(tmp_path: Path) -> None:
    """Editing a file with an OLD site must not fire; adding a NEW one must."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    target = tmp_path / "mod.py"
    target.write_text(NON_CONFORMING, encoding="utf-8")
    subprocess.run(["git", "add", "mod.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "pre-existing site", "--no-verify"],
        cwd=tmp_path,
        check=True,
    )

    # An unrelated edit to the same file: the old site must NOT fire.
    target.write_text(NON_CONFORMING + "\nUNRELATED = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "mod.py"], cwd=tmp_path, check=True)
    assert scan_staged(repo_root=tmp_path, strict=False, verbose=False) == []

    # Now ADD a new site: it must fire.
    target.write_text(
        NON_CONFORMING + "\nUNRELATED = 1\n\ndef more(frames, n_frames):\n    return frames[:n_frames]\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "mod.py"], cwd=tmp_path, check=True)
    violations = scan_staged(repo_root=tmp_path, strict=False, verbose=False)
    assert len(violations) == 1, violations
    assert "frames[:n_frames]" in violations[0]


def test_added_lines_parses_hunk_headers(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.py"], cwd=tmp_path, check=True)
    assert added_lines(tmp_path, "a.py") == {1, 2}


def test_git_failure_is_reported_as_unknown_not_as_no_added_lines(tmp_path: Path) -> None:
    """Review-pass-1 finding: `set()` on git failure made the guard pass SILENTLY.

    `only_lines=set()` filters out every site, so a git failure looked exactly
    like a clean file. `None` is the distinct 'we could not tell' channel, and
    `scan_staged` must then count the file as NOT examined.
    """
    assert added_lines(tmp_path, "nope.py") is None
    # ddm_si1 (task #929): this line previously asserted ``== []``, i.e. the
    # test that NAMES this bug class also pinned the uncured neighbour four
    # lines above the cure. An unanswerable git is not an empty stage.
    assert staged_py_files(tmp_path) is None
    assert in_scope_py_files(tmp_path) is None


def test_scan_staged_refuses_when_git_cannot_enumerate_the_staged_set(
    tmp_path: Path,
) -> None:
    """Executed control: a broken git must REFUSE, not report a vacuous pass.

    Before ddm_si1 this path returned ``[]`` from ``staged_py_files`` and the
    gate printed ``VACUOUS: 0 staged .py files`` -- a false statement (git had
    broken; the commit was not empty) carrying the same symbol as a real pass.
    """
    from tac.preflight import PreflightError

    # tmp_path is not a git repo, so enumeration genuinely fails.
    with pytest.raises(PreflightError, match="UNKNOWN"):
        scan_staged(repo_root=tmp_path, strict=True, verbose=False)

    # And warn-mode must still surface it as a violation rather than silence.
    violations = scan_staged(repo_root=tmp_path, strict=False, verbose=False)
    assert len(violations) == 1, violations
    assert "UNKNOWN" in violations[0]


def test_scan_staged_reports_files_it_could_not_examine(tmp_path: Path, capsys) -> None:
    """An unexaminable file must appear in the denominator, not vanish into OK."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "real.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "real.py"], cwd=tmp_path, check=True)
    # A staged-looking path that does not exist on disk -> unreadable.
    scan_staged(
        repo_root=tmp_path, strict=False, verbose=True, files=["real.py", "ghost.py"]
    )
    out = capsys.readouterr().out
    assert "1 of 2" in out, out
    assert "NOT examined" in out, out
    assert "ghost.py" in out, out


# --- the wiring, which unit tests cannot see -------------------------------


def test_hook_calls_the_scan_before_preflight() -> None:
    """WIRING: main() must call the scan, and call it before run_preflight().

    Order matters for the same reason it mattered for the Catalog #184 sibling:
    run_preflight() early-returns on failure, so a step placed after it is
    skipped exactly when preflight is red.
    """
    hook = (REPO_ROOT / "tools" / "preflight_hook.py").read_text(encoding="utf-8")
    tree = ast.parse(hook)
    main_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    called: list[str] = []
    for node in ast.walk(main_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.append(node.func.id)
    assert "run_subset_selection_scan" in called, called
    assert called.index("run_subset_selection_scan") < called.index("run_preflight"), called


def test_hook_scan_fails_open_on_a_broken_guard() -> None:
    """A guard that crashes must not block every commit in the repo."""
    hook = (REPO_ROOT / "tools" / "preflight_hook.py").read_text(encoding="utf-8")
    body = hook.split("def run_subset_selection_scan")[1].split("\ndef ")[0]
    assert "failing OPEN" in body
    assert body.count("return 0") >= 3  # unavailable / crashed / clean


# --- repo scope ------------------------------------------------------------


def test_in_scope_excludes_vendored_intake() -> None:
    files = in_scope_py_files(REPO_ROOT)
    assert files, "VACUOUS: git ls-files returned nothing"
    assert not [f for f in files if f.startswith("experiments/results/")]


def test_this_landings_own_modules_are_clean() -> None:
    """The landing must not trip its own guard."""
    for rel in (
        "src/tac/subset_selection.py",
        "src/tac/subset_selection_gate.py",
        "src/tac/tests/test_subset_selection.py",
    ):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert _violations_for_text(text, rel) == [], rel
