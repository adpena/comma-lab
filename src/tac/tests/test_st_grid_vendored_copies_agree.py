# SPDX-License-Identifier: MIT
"""The vendored ``ST_GRID`` copies must agree -- and the scan must not be vacuous.

WHY THIS EXISTS (ddm_qd1, 2026-08-03).  ``ST_GRID`` is the s_t translation-scale
codebook.  It is duplicated across the repo ON PURPOSE: ``pfs1_warp_receiver``
documents itself as needing "NO tac dependency" because it is vendored whole
into the shipping decode path, so collapsing the copies into one import would
break the self-containment that rule 118 requires of the receiver.  The copies
are therefore NOT a refactor target.

What was genuinely missing was a DETECTOR.  Nothing asserted the copies still
agree, and the ways that bites are specific and measured:

* ``ddm_cx1`` measured this exact field to be the rule-118 discriminator -- the
  same ``st_grid`` classifies GENERIC on ``dc1_fold`` and VIDEO_DERIVED (fitted)
  on ``pj2``.  A silent drift between copies is therefore a COMPLIANCE question
  (counted bytes, rule 118), not a style question.
* ``tools/ms8_st_codebook_race.py`` already carries ``_assert_incumbent()`` for
  exactly this class, but it anchors to ``experiments/ddm_pfs1_ep_warp_pose_solve``
  while the receiver lives at ``src/tac/optimization/pfs1_warp_receiver``.  Two
  candidate canonicals existed and nothing asserted THEY agreed with each other.
* ``test_ddm_cx1_container_compose`` previously held its own literal copy, so it
  certified itself: the constant could drift and every assertion there would
  still pass.  That test is now anchored, but anchoring alone cannot DETECT
  drift -- it is a round-trip consistency test that uses the value on both
  sides, so it is tautological in the value.  MEASURED: mutating the receiver
  constant left that suite green (14 passed).  Detection needs this file.

THE TWO FAMILIES.  Both are legitimate and the relationship between them is
itself checkable, so neither is an exception:

* the FULL ladder (11 entries, leading ``0.0`` = the identity / no-warp
  codeword);
* the POSITIVE-ONLY ladder (10 entries), used where ``s_t = 0`` is scored
  separately as an explicit null before the sweep -- verified at both sites
  (``pose_frame0_inverse_solve_probe.warp_base_fit`` documents "positive grid +
  s_t=0 null"; ``measure_warp_dpose_through_R`` computes ``null_obj = obj(0.0)``
  and seeds ``best_st = 0.0`` before looping).  It must equal ``canonical[1:]``.

VACUITY IS NOT A PASS.  This scan asserts its own denominator: if the discovery
walk finds fewer sites than we know exist, the test FAILS rather than reporting
a green on an empty scope.  That failure mode -- an empty scan emitting the same
symbol as a clean one -- is the genus this repo has been bitten by repeatedly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]

#: Directories walked for vendored copies.  Kept explicit so the denominator
#: assertion below is meaningful rather than incidentally satisfied.
_SCAN_DIRS = ("src", "tools", "experiments")

#: Names that hold a DELIBERATELY different ladder and must never be forced to
#: equal the canonical.  Each carries the reason it differs.
_INTENTIONALLY_DIFFERENT = {
    # ddm_cx1's fitted / video-derived grid: the whole point is that it is NOT
    # the vendored one (that is what makes it COUNTED under rule 118).
    "FITTED_ST_GRID",
    # ms8's evaluation support: deliberately WIDER than the incumbent at both
    # ends so the instrument can detect clipping instead of assuming the
    # incumbent's support was right.
    "S_EVAL",
    # a scalar cardinality, not a ladder.
    "ST_GRID_SIZE",
}

#: Minimum number of vendored ladders we know exist (MEASURED 2026-08-03: 9
#: discovered, 8 checked after excluding the intentionally-different names --
#: one of the 8 being the copy embedded in the shipped inflate_runner source).
#: If the walk finds fewer, the scan is broken (moved files, renamed constant,
#: a receiver inlined some new way) and must fail loudly rather than report a
#: green over a shrunken scope.
_MIN_EXPECTED_SITES = 8


def _canonical() -> tuple[float, ...]:
    """The receiver's ladder -- the copy that actually ships in the decode path."""
    from tac.optimization.pfs1_warp_receiver import ST_GRID

    return tuple(float(v) for v in ST_GRID)


def _float_seq(node: ast.AST) -> tuple[float, ...] | None:
    """Return a literal list/tuple of numbers, or None if it is not one."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    out: list[float] = []
    for elt in node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float)):
            out.append(float(elt.value))
        elif (
            isinstance(elt, ast.UnaryOp)
            and isinstance(elt.op, ast.USub)
            and isinstance(elt.operand, ast.Constant)
            and isinstance(elt.operand.value, (int, float))
        ):
            out.append(-float(elt.operand.value))
        else:
            return None
    return tuple(out)


def _ladders_in_source(text: str, origin: str) -> list[tuple[str, str, tuple[float, ...]]]:
    """Collect ``*ST_GRID*``-named numeric ladders from one unit of Python source.

    Walks ALL assignments, not just module-level ones: some sites bind the
    ladder inside a function, and a module-level-only walk would silently miss
    them -- the same undercount-that-looks-complete failure this file guards
    against.

    Also recurses into STRING LITERALS that contain embedded Python source.
    That is not a nicety: ``tools/pfs1_recompose_warp_base_and_eval.py`` holds
    the receiver inside an ``INFLATE_RUNNER`` string that it writes out as
    ``inflate_runner.py``, so THE COPY THAT ACTUALLY SHIPS IN THE DECODE PATH
    lives inside a string.  A plain AST walk sees a string, not an assignment,
    and reports "all clean" while being blind to the single most important
    site.  MEASURED: before this recursion the scan found 8 ladders and missed
    the shipping one.
    """
    out: list[tuple[str, str, tuple[float, ...]]] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "ST_GRID" in node.value.upper() and "=" in node.value:
                out.extend(
                    _ladders_in_source(node.value, f"{origin} [embedded source]")
                )
            continue
        targets: list[ast.expr]
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for tgt in targets:
            if not isinstance(tgt, ast.Name):
                continue
            if "ST_GRID" not in tgt.id.upper():
                continue
            vals = _float_seq(value)
            if vals is None or len(vals) < 6:
                continue
            out.append((origin, tgt.id, vals))
    return out


def _discover() -> list[tuple[str, str, tuple[float, ...]]]:
    """Find every ``*ST_GRID*``-named numeric ladder. Returns (path, name, values)."""
    found: list[tuple[str, str, tuple[float, ...]]] = []
    for d in _SCAN_DIRS:
        for path in sorted((_REPO / d).rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            # Cheap text pre-filter before the AST parse. This is a pure
            # speedup and cannot narrow the result set: the walk below only
            # ever accepts targets whose name contains "ST_GRID", so a file
            # without that substring can contribute nothing. Parsing every
            # .py file (preflight.py alone is ~85k lines) made this test take
            # minutes.
            if "ST_GRID" not in text.upper():
                continue
            found.extend(_ladders_in_source(text, str(path.relative_to(_REPO))))
    return found


def test_scan_is_not_vacuous() -> None:
    """An empty or shrunken scope must FAIL, never report a green.

    Without this, every assertion below would pass trivially the moment the
    discovery walk broke -- reporting success over a scope of zero.
    """
    sites = _discover()
    checked = [s for s in sites if s[1] not in _INTENTIONALLY_DIFFERENT]
    assert len(checked) >= _MIN_EXPECTED_SITES, (
        f"ST_GRID scan found only {len(checked)} vendored ladder(s) over "
        f"{_SCAN_DIRS} (expected >= {_MIN_EXPECTED_SITES}). The scan is broken, "
        f"not clean -- a shrunken scope is VACUOUS, not a PASS. Found: {sites!r}"
    )


def test_the_two_candidate_canonicals_agree() -> None:
    """The receiver's ladder and the solver's ladder must be the same object.

    ``tools/ms8_st_codebook_race._assert_incumbent`` guards against the SOLVER
    copy; the shipping decode path uses the RECEIVER copy. Nothing previously
    asserted these two agreed with each other.
    """
    import sys

    sys.path.insert(0, str(_REPO / "experiments"))
    try:
        import ddm_pfs1_ep_warp_pose_solve as solver
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"cannot import the solver module: {exc}")

    assert tuple(float(v) for v in solver.ST_GRID) == _canonical(), (
        "the solver's ST_GRID and the receiver's ST_GRID have drifted apart; "
        "ms8's _assert_incumbent guards only the solver copy, so this drift "
        "would otherwise ship in the receiver unnoticed"
    )


def test_every_vendored_copy_matches_the_canonical_or_its_positive_tail() -> None:
    """Full ladders equal the canonical; positive-only ladders equal ``canonical[1:]``."""
    canonical = _canonical()
    positive_tail = canonical[1:]
    assert canonical[0] == 0.0, (
        "the canonical ladder must lead with the 0.0 identity (no-warp) codeword; "
        "the positive-tail relationship checked below depends on it"
    )

    problems: list[str] = []
    for path, name, vals in _discover():
        if name in _INTENTIONALLY_DIFFERENT:
            continue
        if vals in (canonical, positive_tail):
            continue
        problems.append(
            f"{path}: {name} = {list(vals)!r} matches neither the canonical "
            f"{list(canonical)!r} nor its positive tail {list(positive_tail)!r}"
        )
    assert not problems, (
        "vendored ST_GRID copies have drifted. These copies are duplicated on "
        "purpose (the receiver must stay tac-dependency-free for rule 118), so "
        "the fix is to re-sync the copy, NOT to de-duplicate:\n  "
        + "\n  ".join(problems)
    )
