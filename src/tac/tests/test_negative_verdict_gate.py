# SPDX-License-Identifier: MIT
"""Tests for the negative-verdict scope guard (`ddm_ks1`).

The load-bearing test is :func:`test_positive_control_an_unscoped_negative_is_refused`
together with :func:`test_positive_control_raises_in_strict_mode`. A guard without
an executed failing case is not landed -- it is a decoration that has never been
shown to be capable of saying no.

Equally load-bearing: :func:`test_hook_calls_the_scan_before_preflight`. A passing
unit test proves the UNIT, never the WIRING, and the measured reason this guard
exists at all is that two STRICT gates (Catalog #307/#308) are registered in
``preflight_all()`` and therefore never execute at commit time under the hook's
default ``--no-codebase``.

And :func:`test_the_ladder_is_strictly_growing` is the cost-asymmetry property
itself, asserted rather than asserted-about.
"""

from __future__ import annotations

import itertools
import subprocess
from pathlib import Path

import pytest

from tac.negative_verdict_gate import (
    REQUIRED_BY_LEVEL,
    SCOPE_WINDOW_LINES,
    WAIVER_MARKER,
    _violations_for_text,
    added_lines,
    declared_level,
    find_assertions,
    in_scope_files,
    scan_staged,
    staged_files,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

UNSCOPED = """
# Notes on the lever.
**VERDICT: NO-GO** -- the carrier does not pay for itself.
"""

SCOPED_INSTANCE = """
**VERDICT: NO-GO** -- the carrier does not pay for itself.
verdict_scope: INSTANCE -- the one swept config at n96, seed 0.
"""

SCOPED_FAMILY_NO_EVIDENCE = """
**VERDICT: KILLED**
verdict_scope: FAMILY -- all cheap-carrier formulations.
"""

SCOPED_FAMILY_WITH_EVIDENCE = """
**VERDICT: KILLED**
verdict_scope: FAMILY -- all cheap-carrier formulations.
family_evidence: failures across >= 2 structurally distinct formulations (dxi, warp).
"""

SCOPED_PARADIGM_NO_CAPACITY = """
**VERDICT: FALSIFIED**
verdict_scope: PARADIGM -- post-hoc correction as such.
family_evidence: see arXiv:2401.00000 impossibility bound.
"""

SCOPED_PARADIGM_WITH_CAPACITY = """
**VERDICT: FALSIFIED**
verdict_scope: PARADIGM -- post-hoc correction as such.
family_evidence: see arXiv:2401.00000 impossibility bound.
instrument_capacity: the oracle spans 11 receiver knobs; the denied effect is 6-DOF.
"""

CITATION = """
`na2` reported *"HEADLINE VERDICT: NO-GO"* on n=4 and n=12, which is a prefix.
"""

BARE_MENTION = """
The FALSIFIED lineage is documented elsewhere; this is a NO-GO region of the map.
Per CLAUDE.md, KILL is the last resort and DEAD lanes are archived, not deleted.
"""

WAIVED = f"""
**VERDICT: NO-GO** <!-- {WAIVER_MARKER}: quoting rv1 #110's historical row verbatim -->
"""

PLACEHOLDER_WAIVED = f"""
**VERDICT: NO-GO** <!-- {WAIVER_MARKER}: <rationale> -->
"""

# --- the controls ----------------------------------------------------------


def test_positive_control_an_unscoped_negative_is_refused() -> None:
    """THE positive control: the guard must be able to say no."""
    violations = _violations_for_text(UNSCOPED, "fixture.md")
    assert len(violations) == 1, violations
    assert "INSUFFICIENT_SCOPE" in violations[0]
    assert "fixture.md:3" in violations[0]


def test_positive_control_raises_in_strict_mode(tmp_path: Path) -> None:
    """And refusing must actually RAISE, not merely return a list."""
    from tac.preflight import PreflightError

    (tmp_path / "memo.md").write_text(UNSCOPED, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "memo.md"], cwd=tmp_path, check=True)
    with pytest.raises(PreflightError, match="INSUFFICIENT_SCOPE"):
        scan_staged(repo_root=tmp_path, strict=True, verbose=False)


def test_negative_control_a_scoped_instance_negative_passes() -> None:
    """An INSTANCE kill is deliberately cheap: declare the scope, that is all."""
    assert _violations_for_text(SCOPED_INSTANCE, "fixture.md") == []


def test_a_real_waiver_is_honoured() -> None:
    assert _violations_for_text(WAIVED, "fixture.md") == []


def test_a_placeholder_waiver_does_not_waive() -> None:
    """Catalog #287: the docstring's own example must not self-waive."""
    assert len(_violations_for_text(PLACEHOLDER_WAIVED, "fixture.md")) == 1


# --- cost asymmetry by scope ----------------------------------------------


def test_the_ladder_is_strictly_growing() -> None:
    """A bigger claim must cost STRICTLY more -- the property, not a comment.

    This is the charter's item 2. It is asserted here because the equivalent
    rule in ``VerdictScope.__post_init__`` covers only family_evidence; this
    guard extends the asymmetry to the markdown surface and to capacity.
    """
    order = ["instance", "formulation", "family", "paradigm"]
    for lower, higher in itertools.pairwise(order):
        lo, hi = set(REQUIRED_BY_LEVEL[lower]), set(REQUIRED_BY_LEVEL[higher])
        assert lo <= hi, f"{higher} must require a superset of {lower}"
    assert set(REQUIRED_BY_LEVEL["instance"]) < set(REQUIRED_BY_LEVEL["paradigm"])


def test_family_scope_without_family_evidence_is_refused() -> None:
    v = _violations_for_text(SCOPED_FAMILY_NO_EVIDENCE, "fixture.md")
    assert len(v) == 1 and "INSUFFICIENT_FAMILY_EVIDENCE" in v[0], v


def test_family_scope_with_family_evidence_passes() -> None:
    assert _violations_for_text(SCOPED_FAMILY_WITH_EVIDENCE, "fixture.md") == []


def test_paradigm_scope_without_instrument_capacity_is_refused() -> None:
    """LAW A (`ddm_pu2`): a floor is only a floor if the instrument could resolve it."""
    v = _violations_for_text(SCOPED_PARADIGM_NO_CAPACITY, "fixture.md")
    assert len(v) == 1 and "INSUFFICIENT_INSTRUMENT_CAPACITY" in v[0], v


def test_paradigm_scope_fully_evidenced_passes() -> None:
    assert _violations_for_text(SCOPED_PARADIGM_WITH_CAPACITY, "fixture.md") == []


def test_unreadable_level_is_not_silently_graded_as_instance() -> None:
    """LAW C applied to this guard's own logic -- found in review pass 2.

    An earlier revision read ``REQUIRED_BY_LEVEL.get(level or "instance", ...)``,
    so a scope whose level could not be parsed was graded at the CHEAPEST rung.
    Measured impact at the time: 14 of 23 scoped assertions (60.9%). Widening the
    level regex recovered 10; the remaining 4 must REFUSE, not be demoted.
    """
    text = (
        "**VERDICT: KILLED**\n"
        "verdict_scope: see the discussion above for what this binds to.\n"
    )
    v = _violations_for_text(text, "fixture.md")
    assert len(v) == 1 and "INSUFFICIENT_SCOPE_LEVEL" in v[0], v


def test_prose_level_forms_are_read() -> None:
    """The corpus writes the level in prose; the detector must reach it."""
    lines = [
        "**VERDICT: KILLED**",
        "verdict_scope = `FAMILY` -- all cheap-carrier formulations, see arXiv:1.2",
    ]
    assert declared_level(lines, 0) == "family"


def test_declared_level_reads_the_ladder() -> None:
    lines = SCOPED_FAMILY_WITH_EVIDENCE.splitlines()
    idx = next(i for i, line in enumerate(lines) if "KILLED" in line)
    assert declared_level(lines, idx) == "family"


# --- false-positive shape (measured, not assumed) --------------------------


def test_a_citation_of_another_doc_verdict_is_not_an_assertion() -> None:
    """Measured at 15 of 80 matches (18.8%): reporting a verdict is not asserting one."""
    assert _violations_for_text(CITATION, "fixture.md") == []


def test_bare_mentions_are_not_matched() -> None:
    """`FALSIFIED` alone is on 2,032 lines and `NO-GO` on 1,080 -- nearly all prose.

    Matching those would train the repo to write waivers, which is the failure
    mode `ddm_ss1` named. Only a LABEL bound to a token is the signature.
    """
    assert find_assertions(BARE_MENTION) == []


def test_exempt_doctrine_surfaces_are_not_scanned() -> None:
    """CLAUDE.md quotes kill verdicts as RULES; it is not asserting them."""
    assert _violations_for_text(UNSCOPED, "CLAUDE.md") == []


def test_python_is_residue_not_silently_covered() -> None:
    """`.py` is OUT of scope by measurement, and the scan must not claim it.

    A first cut included `.py` and produced 17 hits, a large minority of which
    were not assertions: a QUERY (`query_by_decision(g, verdict="NO-GO")`), a
    TEST FIXTURE (`_emit(tmp_path, verdict="NO-GO")`), and a CONSTANT NAME
    (`TOMBSTONE_STATUS = "RETIRED_..."`). Python passes, queries and asserts a
    verdict with identical tokens. So the enumerators must not collect `.py` --
    if they ever do again, this test fails and the precision claim gets re-made.
    """
    assert not any(r.endswith(".py") for r in in_scope_files(REPO_ROOT))


def test_scope_beyond_the_window_does_not_launder() -> None:
    """The measured laundering band: a scope bound to some OTHER claim.

    File-wide conformance was 47.5% vs 35.0% at +/-80 -- the 12.5-point jump is
    scope declared once at the top of a long document standing in for every kill
    below it. Past the window, it must not count.
    """
    far = (
        "verdict_scope: INSTANCE -- a different claim entirely.\n"
        + "filler\n" * (SCOPE_WINDOW_LINES + 5)
        + "**VERDICT: NO-GO** -- this one has no scope of its own.\n"
    )
    v = _violations_for_text(far, "fixture.md")
    assert len(v) == 1 and "INSUFFICIENT_SCOPE" in v[0], v


# --- vacuity is not a pass -------------------------------------------------


def test_zero_staged_files_reports_vacuous_not_ok(tmp_path: Path, capsys) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    assert scan_staged(repo_root=tmp_path, strict=True, verbose=True) == []
    assert "VACUOUS" in capsys.readouterr().out


def test_added_lines_returns_none_not_empty_on_git_failure(tmp_path: Path) -> None:
    """`None` != `set()`: an empty set would filter out every line and pass silently."""
    assert added_lines(tmp_path, "nope.md") is None


def test_unreadable_added_range_is_reported_not_passed(tmp_path: Path, capsys) -> None:
    """A file whose diff cannot be read must be named as NOT examined."""
    (tmp_path / "memo.md").write_text(UNSCOPED, encoding="utf-8")
    # No git repo here -> `git diff` fails -> added_lines is None.
    scan_staged(repo_root=tmp_path, strict=False, verbose=True, files=["memo.md"])
    out = capsys.readouterr().out
    assert "NOT examined" in out and "memo.md" in out


# --- added-lines scoping ---------------------------------------------------


def test_only_added_lines_are_in_scope(tmp_path: Path) -> None:
    """Editing a doc with an OLD assertion must not fire; adding a NEW one must."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    target = tmp_path / "memo.md"
    target.write_text(UNSCOPED, encoding="utf-8")
    subprocess.run(["git", "add", "memo.md"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "pre-existing", "--no-verify"],
        cwd=tmp_path,
        check=True,
    )

    target.write_text(UNSCOPED + "\nUnrelated prose.\n", encoding="utf-8")
    subprocess.run(["git", "add", "memo.md"], cwd=tmp_path, check=True)
    assert scan_staged(repo_root=tmp_path, strict=False, verbose=False) == []

    target.write_text(
        UNSCOPED + "\nUnrelated prose.\n\n**VERDICT: FALSIFIED** -- brand new claim.\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "memo.md"], cwd=tmp_path, check=True)
    v = scan_staged(repo_root=tmp_path, strict=False, verbose=False)
    assert len(v) == 1, v


def test_staged_files_picks_up_markdown_only(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "a.md").write_text("x\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    got = set(staged_files(tmp_path))
    assert got == {"a.md"}, got


# --- wiring: a gate main() never calls is vacuity-equals-pass --------------


def test_hook_calls_the_scan_before_preflight() -> None:
    """The unit passing proves the unit. This asserts the WIRING.

    Order matters: ``run_preflight()`` early-returns on failure, so a step placed
    after it is skipped exactly when preflight fails.
    """
    src = (REPO_ROOT / "tools" / "preflight_hook.py").read_text(encoding="utf-8")
    assert "run_negative_verdict_scan" in src, "hook does not call the scan"
    body = src[src.index("def main()") :]
    # Round 1 of this test was itself the bug it guards against: it searched for
    # the bare token `run_preflight()`, whose first occurrence in main() is inside
    # the step-1b COMMENT explaining the ordering, not at the call. It compared a
    # call site against a comment and reported a wiring failure that did not
    # exist. Anchor on the `rc = ` assignment — the executable form.
    i_scan = body.index("rc = run_negative_verdict_scan(")
    i_pre = body.index("rc = run_preflight()")
    assert i_scan < i_pre, "scan must run BEFORE run_preflight (which early-returns)"


def test_hook_collects_staged_markdown() -> None:
    """Negatives live in .md; the hook historically collected only .py."""
    src = (REPO_ROOT / "tools" / "preflight_hook.py").read_text(encoding="utf-8")
    assert "_staged_doc_files" in src, "hook must collect staged .md for this scan"


def test_in_scope_files_excludes_vendored_results() -> None:
    rels = in_scope_files(REPO_ROOT)
    assert rels, "VACUOUS: git ls-files returned nothing"
    assert not any(r.startswith("experiments/results/") for r in rels)
