# SPDX-License-Identifier: MIT
"""Guard: a NEWLY-ASSERTED negative verdict must declare its scope (`ddm_ks1`).

Why this is a GUARD and not another record
------------------------------------------
The typed record already exists and is good. :mod:`tac.verdicts.emit` (landed
2026-07-09) has the ladder ``ScopeLevel`` (INSTANCE < FORMULATION < FAMILY <
PARADIGM), a ``VerdictScope`` whose ``__post_init__`` **already refuses** a
FAMILY/PARADIGM scope without ``family_evidence``, and an ``emit_verdict`` that
**already requires** a reformulation queue when ``is_negative=True``.

So the charter premise this landing started from -- "a FAMILY kill and an
INSTANCE kill cost the same, which is nothing" -- is REFUTED by measurement. The
asymmetry was built a month ago. What is missing is adoption, measured
2026-08-03:

* ``emit_verdict`` production (non-test) call sites: **0**. The only
  ``emit_verdict(`` outside tests is its own ``def``.
* ``is_negative=True``: **5** occurrences in tracked ``.py``, **all 5 in tests**.
* ⚠ **WITHDRAWN 2026-08-03 (`ddm_vw1`, task #936) — the third bullet as landed read
  "0 of 36 tracked ``verdict.v1``-matching JSONs carry the ``level`` key", and that is
  FALSE.** Re-measured twice independently: the ``36`` denominator reproduces exactly,
  but it counts tracked ``.json`` files that merely MENTION the string ``verdict.v1``
  (fixtures, dashboards, the ``frontier_rate_attack_feedback`` corpus) — not
  ``verdict.v1`` DOCUMENTS. **5 of those 36 do carry ``level``**, and under the strict
  matcher ``"schema_version": "verdict.v1"`` the tracked population is **5, of which
  5 carry ``scope.level`` — 100%**. Counting untracked copies the ratio is 12/12.
  The bullet was a numerator reported against the wrong denominator, which is the
  repo's own named poison; it is corrected here rather than silently deleted.

  **The correction inverts the diagnosis, and that matters more than the number.**
  ``emit_verdict`` did not fail at WRITE time — every verdict it ever wrote is
  correctly scoped. It was exercised on ONE day (2026-07-10, commits ``1352982bac``
  + ``f6fd9d81de``), produced 12 conforming documents, and was never called again:
  nothing has emitted a ``verdict.v1`` in the 24 days since. So this is ADOPTION
  DECAY, not a defective record — and the measured reasons are (a) the format has
  **0 readers** outside its own package (exhaustive over 10,728 tracked ``.py``), so
  emitting bought the author nothing, and (b) until #936 the API offered ONLY a
  whole-file write, so the 486 non-test hand-rolled verdict-JSON producers (0 of
  which import ``tac.verdicts``) had no embeddable path. See
  ``tac.verdicts.verdict_payload``.

Meanwhile negatives are written in MARKDOWN, and markdown has no path to that
record and no gate at all: ``tools/preflight_hook.py`` collects only staged
``.py`` (``_staged_py_files``), so a doc-only commit asserting a PARADIGM kill
passes every check in the repo. That is the hole this closes.

The measured generator
----------------------
Measured 2026-08-03 over **7,803 tracked ``.md``**: an explicitly LABELLED
negative verdict assertion (a verdict/status/conclusion label bound by ``:``/``=``
to a negative token) occurs on **80 lines across 53 files**. Bare mentions are
not the signature and are deliberately NOT matched -- ``FALSIFIED`` alone appears
on 2,032 lines and ``NO-GO`` on 1,080, nearly all of them citations, prose, and
CLAUDE.md doctrine. Matching those would train everyone to write waivers, which
is the failure mode `ddm_ss1` warned about.

Why the window is +/-40 lines, and not the file
-----------------------------------------------
Conformance of those same 80 assertions against distance to the nearest
``verdict_scope``, measured (not chosen):

===============  ==========================
window           assertions with scope near
===============  ==========================
same line          6/80  =  7.5%
+/-10             18/80  = 22.5%
+/-20             25/80  = 31.2%
+/-40             27/80  = 33.8%
+/-80             28/80  = 35.0%
FILE-WIDE         38/80  = 47.5%
===============  ==========================

The curve is flat from +/-20 to +/-80 (31.2% -> 35.0%: doubling the window twice
buys 3 lines) and then jumps 12.5 points at FILE-WIDE. That jump IS the
laundering band: a scope declared once at the top of a 49 KB document, bound to
some other claim, standing in for every kill below it. A file-wide rule accepts
exactly those. So the window is +/-40 -- past the knee, short of the laundering.

Cost asymmetry by scope
-----------------------
:data:`REQUIRED_BY_LEVEL` grows strictly with the ladder, so a bigger claim costs
strictly more declaration. An INSTANCE negative is deliberately cheap (declare
the scope, that is all): making the honest narrow case cheap is what stops
authors from reaching for a waiver. A PARADIGM negative must additionally carry
family evidence AND an instrument-capacity statement -- LAW A, the `ddm_pu2`
shape, where a 6-DOF oracle produced a floor denying an effect an 11-knob
receiver could express. Capacity is stated in the claim's own units or it is not
a capacity claim.

Where it runs
-------------
Directly from ``tools/preflight_hook.py``, following the precedent set by the
Catalog #184 heavy-import scan and by `ddm_ss1`'s subset guard, and for the same
measured reason: ``python -m tac.preflight --no-codebase`` -- the hook's DEFAULT
mode -- examines **0 of 27** declared gates, so a gate registered only in
``preflight_all()`` never fires at commit time.

Scope is the staged diff's ADDED LINES over ``.md`` -- the surface no
commit-time check read before, and the one na2 measured the corpus on. That is
what makes STRICT honest: the repo carries **41 non-conforming assertions across
34 files** (measured by :func:`scan_repo`, 2026-08-03), so a whole-repo strict
flip would refuse commits across 34 files; added-lines scope makes the live count
0 on a clean commit while refusing every new one. The debt stays visible as debt.

Reconciling the two counts, because they differ and the difference is the point:
the raw regex matches **80** lines, :func:`scan_repo` reports **41**. 15 of the
80 are citation-form and are excluded by design; of the remaining 65, 24 already
declare a scope inside the window. 65 - 24 = 41. The 80 is a regex count; the 41
is the violation count.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from tac.confound_gates import REPO_ROOT, _finish, _rationale_ok

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

#: Lines either side of an assertion in which a scope declaration counts as bound
#: to it. Derived from the measured conformance curve in the module docstring.
SCOPE_WINDOW_LINES = 40

#: Same-line waiver. Accepted in ``#`` (python) and ``<!-- -->`` (markdown) form.
#: A placeholder rationale does not self-waive (Catalog #287 sister discipline).
WAIVER_MARKER = "NEGATIVE_VERDICT_SCOPE_OK"

#: The label half of the signature: what a verdict assertion calls itself.
_LABEL = r"(?:HEADLINE\s+)?(?:VERDICT|Verdict|verdict|STATUS|Status|CONCLUSION|Conclusion)"

#: The token half: the dispositions that constitute a NEGATIVE verdict.
_NEG_TOKEN = (
    r"(?:KILL|KILLED|FALSIFIED|DEAD|RETIRED|NO-GO|NOGO|REFUTED|NEGATIVE|ABANDONED)"
)

#: A labelled negative verdict assertion. Markdown emphasis/quote characters are
#: allowed between the separator and the token because that is how the corpus
#: actually writes it (``**VERDICT: NO-GO**``, ``Verdict: **KILLED**``).
ASSERTION_RE = re.compile(_LABEL + r"[ \t]*[:=][ \t]*[*_`\"“]*" + _NEG_TOKEN)

#: A CITATION of somebody else's verdict -- the label sits inside a quoted or
#: backticked span. Measured at 15 of 80 (18.8%) of matches, and reporting a
#: historical verdict is not asserting a new one, so these are not violations.
_CITATION_RE = re.compile(r"[\"`“][^\"`”]{0,60}" + _LABEL + r"[ \t]*[:=]")

#: A scope declaration bound to the claim.
_SCOPE_RE = re.compile(r"verdict_scope|VerdictScope|ScopeLevel")

#: Evidence that a FAMILY/PARADIGM claim paid for its breadth: the field name the
#: canonical record uses, or an explicit ">= 2 distinct formulations" statement,
#: or a citation (the same three forms ``VerdictScope.__post_init__`` accepts).
_FAMILY_EVIDENCE_RE = re.compile(
    r"family_evidence"
    r"|(?:>=|≥|at least)\s*2\s+(?:structurally\s+)?distinct\s+formulation"
    r"|arxiv|doi:|theorem|impossibility",
    re.IGNORECASE,
)

#: LAW A (`ddm_pu2`): the instrument's capacity, stated in the claim's own units.
_CAPACITY_RE = re.compile(
    r"instrument_capacity|instrument capacity|resolvable|resolution floor"
    r"|noise_floor|noise floor|dof\b|degrees of freedom",
    re.IGNORECASE,
)

#: Which ladder level a nearby scope declaration claims.
#:
#: The window between the token and the level word is 60 chars of ANY character,
#: not a dozen non-word ones. Measured 2026-08-03: a tighter ``\W{0,12}`` form
#: failed to read the level on **14 of 23** scoped assertions (60.9%), because
#: the corpus writes ``verdict_scope = `INSTANCE` -- the one config`` and
#: ``verdict_scope: the FAMILY of cheap carriers``, where prose sits in between.
_LEVEL_RE = re.compile(
    r"verdict_scope.{0,60}?\b(instance|formulation|family|paradigm)",
    re.IGNORECASE,
)

#: Required declarations per declared ladder level. MONOTONE NON-DECREASING (each
#: level is a superset of the one below) and STRICTLY greater from ``instance`` to
#: ``paradigm`` -- which is the charter's actual requirement: a FAMILY kill must
#: cost strictly more than an INSTANCE one.
#:
#: ``formulation`` deliberately costs the same as ``instance``. Both are narrow
#: claims about a thing that was actually run, and the measured risk of pricing
#: the cheap honest rungs too high is that authors reach for a waiver instead of
#: declaring -- the failure mode `ddm_ss1` named. The cost steps at FAMILY, which
#: is the first rung that generalizes beyond what was measured.
REQUIRED_BY_LEVEL: dict[str, tuple[str, ...]] = {
    "instance": ("scope",),
    "formulation": ("scope",),
    "family": ("scope", "family_evidence"),
    "paradigm": ("scope", "family_evidence", "instrument_capacity"),
}

#: Files that are doctrine/index surfaces rather than places verdicts are
#: asserted. CLAUDE.md and the memory index quote kill verdicts as RULES.
_EXEMPT_PREFIXES = (
    "CLAUDE.md",
    "AGENTS.md",
    "docs/meta_bug_class_catalog.md",
)

_ADDED_LINE_RX = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _is_exempt(rel: str) -> bool:
    return any(rel == p or rel.startswith(p) for p in _EXEMPT_PREFIXES)


def _waiver_on_line(source_line: str) -> bool:
    """True iff the line carries a non-placeholder same-line waiver."""
    rx = re.compile(r"(?:#|<!--)[ \t]*" + re.escape(WAIVER_MARKER) + r":[ \t]*(\S.*)")
    m = rx.search(source_line)
    if not m:
        return False
    rationale = m.group(1)
    # Strip a trailing markdown comment terminator so `<!-- X:why -->` reads as
    # rationale "why" and not "why -->".
    rationale = re.sub(r"-->\s*$", "", rationale)
    return _rationale_ok(rationale)


def _window(lines: Sequence[str], idx: int, radius: int) -> str:
    lo = max(0, idx - radius)
    hi = min(len(lines), idx + radius + 1)
    return "\n".join(lines[lo:hi])


def declared_level(lines: Sequence[str], idx: int, radius: int = SCOPE_WINDOW_LINES) -> str | None:
    """The ladder level a nearby scope declaration claims, if it names one."""
    m = _LEVEL_RE.search(_window(lines, idx, radius))
    return m.group(1).lower() if m else None


def find_assertions(text: str, *, only_lines: set[int] | None = None) -> list[tuple[int, str]]:
    """Labelled negative-verdict assertions. 1-based line numbers.

    ``only_lines`` restricts to those line numbers (the staged added-lines scope).
    Citations of another document's verdict are excluded -- see ``_CITATION_RE``.
    """
    out: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if only_lines is not None and i not in only_lines:
            continue
        if not ASSERTION_RE.search(line):
            continue
        if _CITATION_RE.search(line):
            continue
        out.append((i, line))
    return out


def _violations_for_text(
    text: str, rel: str, *, only_lines: set[int] | None = None
) -> list[str]:
    if _is_exempt(rel):
        return []
    lines = text.splitlines()
    violations: list[str] = []
    for lineno, line in find_assertions(text, only_lines=only_lines):
        if _waiver_on_line(line):
            continue
        idx = lineno - 1
        win = _window(lines, idx, SCOPE_WINDOW_LINES)
        level = declared_level(lines, idx)
        has_scope = bool(_SCOPE_RE.search(win))
        if not has_scope:
            violations.append(
                f"{rel}:{lineno}: INSUFFICIENT_SCOPE - a negative verdict is "
                f"asserted with no verdict_scope within +/-{SCOPE_WINDOW_LINES} "
                f"lines. Declare the narrowest level the evidence supports "
                f"(instance<formulation<family<paradigm), or emit the canonical "
                f"record via tac.verdicts.emit_verdict(is_negative=True). "
                f"| {line.strip()[:110]}"
            )
            continue
        if level is None:
            # LAW C, applied to this guard's OWN logic. An earlier revision read
            # `REQUIRED_BY_LEVEL.get(level or "instance", ...)` -- so a scope
            # whose LEVEL could not be read was graded at the CHEAPEST rung. That
            # is silence resolving to the most permissive default: precisely the
            # defect this module exists to refuse, committed by the module
            # itself. An unreadable level is UNKNOWN, and unknown is not instance.
            violations.append(
                f"{rel}:{lineno}: INSUFFICIENT_SCOPE_LEVEL - a scope is declared "
                f"near this negative, but it does not name a ladder level. Write "
                f"one of instance/formulation/family/paradigm next to "
                f"'verdict_scope' so the claim's COST is determined by its "
                f"breadth rather than by what a reader assumes. "
                f"| {line.strip()[:110]}"
            )
            continue
        required = REQUIRED_BY_LEVEL[level]
        if "family_evidence" in required and not _FAMILY_EVIDENCE_RE.search(win):
            violations.append(
                f"{rel}:{lineno}: INSUFFICIENT_FAMILY_EVIDENCE - scope claims "
                f"'{level}', which costs more than an instance kill: cite a "
                f"theorem/impossibility/arXiv/DOI, or state '>= 2 structurally "
                f"distinct formulations'. This mirrors VerdictScope.__post_init__, "
                f"which already refuses the same claim in code. "
                f"| {line.strip()[:110]}"
            )
            continue
        if "instrument_capacity" in required and not _CAPACITY_RE.search(win):
            violations.append(
                f"{rel}:{lineno}: INSUFFICIENT_INSTRUMENT_CAPACITY - scope claims "
                f"'{level}'. State the instrument's capacity IN THE CLAIM'S OWN "
                f"UNITS and show it is at least the effect being denied (LAW A, "
                f"ddm_pu2: a 6-DOF oracle produced a floor denying an effect an "
                f"11-knob receiver could express). "
                f"| {line.strip()[:110]}"
            )
    return violations


def staged_files(repo_root: Path) -> list[str]:
    """Staged ``.md`` files that exist on disk.

    MARKDOWN ONLY, and that is a measured choice, not an oversight. A first cut
    also scanned ``.py`` and produced 17 hits of which a large minority were not
    assertions at all: ``query_by_decision(g, verdict="NO-GO")`` (a QUERY),
    ``_emit(tmp_path, verdict="NO-GO")`` (a TEST FIXTURE), and
    ``TOMBSTONE_STATUS = "RETIRED_UNSAFE_CLEANUP_CERTIFICATE_FAIL_CLOSED"`` (a
    constant name). In Python a verdict string is passed, queried, and asserted
    with the same tokens, so the signature cannot separate them -- and a guard
    that fires on ambiguity gets waived to death, which is exactly the failure
    mode `ddm_ss1` named. Markdown precision was ~95% on the same sweep.

    ``.py`` verdict strings are therefore RESIDUE THIS GUARD DOES NOT REACH.
    Their proper home is the typed record (``tac.verdicts.emit_verdict``), which
    validates the value instead of pattern-matching it.
    """
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=repo_root,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return []
    return [
        f
        for f in out.splitlines()
        if f.endswith(".md") and (repo_root / f).exists()
    ]


def added_lines(repo_root: Path, rel: str) -> set[int] | None:
    """1-based line numbers ADDED to ``rel`` in the staged diff.

    ``None`` (not an empty set) when the diff cannot be read -- the caller must
    report that as NOT EXAMINED rather than as a pass.
    """
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--unified=0", "--", rel],
            cwd=repo_root,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    result: set[int] = set()
    for line in out.splitlines():
        m = _ADDED_LINE_RX.match(line)
        if m:
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) is not None else 1
            result.update(range(start, start + count))
    return result


def scan_staged(
    *,
    repo_root: str | Path | None = None,
    strict: bool = True,
    verbose: bool = True,
    files: Sequence[str] | None = None,
) -> list[str]:
    """Refuse a NEWLY-ADDED under-scoped negative verdict in the staged diff.

    STRICT by default, which is only honest because the scope is added lines: a
    clean commit has zero, so the live count is 0 by construction rather than by
    a migration claim.
    """
    root = Path(repo_root or REPO_ROOT)
    rels = list(files) if files is not None else staged_files(root)
    violations: list[str] = []
    examined = 0
    unexamined: list[str] = []
    for rel in rels:
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            unexamined.append(f"{rel} unreadable ({exc.__class__.__name__})")
            continue
        scope = added_lines(root, rel)
        if scope is None:
            # Cannot scope the scan -> examine nothing and SAY SO. Scanning the
            # whole file would flag pre-existing assertions the author did not
            # add; scanning nothing silently would be the vacuous pass.
            unexamined.append(f"{rel} added-line range unavailable (git diff failed)")
            continue
        examined += 1
        violations.extend(_violations_for_text(text, rel, only_lines=scope))

    detail = f"{examined} of {len(rels)} staged .md file(s) examined"
    if unexamined:
        detail += f"; NOT examined (not a pass): {'; '.join(unexamined)}"
        if verbose:
            print(f"  [negative-verdict-scope-staged] {len(unexamined)} file(s) NOT examined:")
            for note in unexamined:
                print(f"    - {note}")
    return _finish(
        name="check_negative_verdict_scope_declared_staged",
        tag="negative-verdict-scope-staged",
        violations=violations,
        strict=strict,
        verbose=verbose,
        # The DENOMINATOR, always: `examined 0` is VACUOUS, not clean.
        ok_detail=detail
        if rels
        else "VACUOUS: 0 staged .md files -- nothing examined, not a pass",
    )


def in_scope_files(repo_root: Path) -> list[str]:
    """Git-TRACKED ``.md``, excluding vendored public-PR intake trees.

    ``git ls-files`` rather than a filesystem walk, for the reason
    :func:`tac.subset_selection_gate.in_scope_py_files` measured: ``experiments/
    results/`` holds tens of thousands of untracked files, so the tracked
    universe makes the vendored exclusion nearly free.
    """
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "*.md"], cwd=repo_root, text=True
        )
    except (subprocess.CalledProcessError, OSError):
        return []
    return [
        line
        for line in out.splitlines()
        if line and not line.startswith("experiments/results/")
    ]


def scan_repo(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
    files: Iterable[str] | None = None,
) -> list[str]:
    """Whole-repo sweep -- the standing backlog, warn-only.

    Warn-only and staying that way until the debt is drained: measured
    2026-08-03, **41 non-conforming assertions across 34 files** of **7,804**
    tracked ``.md`` (see the module docstring for how 41 reconciles with the 80
    raw regex matches). Reporting that as debt is honest; flipping it STRICT
    would refuse commits across 34 files, and calling it clean would be the lie
    this landing exists to prevent. The STRICT surface is :func:`scan_staged`.
    """
    root = Path(repo_root or REPO_ROOT)
    rels = list(files) if files is not None else in_scope_files(root)
    violations: list[str] = []
    examined = 0
    for rel in rels:
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        examined += 1
        violations.extend(_violations_for_text(text, rel))
    return _finish(
        name="check_negative_verdict_scope_declared",
        tag="negative-verdict-scope",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=f"{examined} tracked in-scope .md file(s) examined"
        if examined
        else "VACUOUS: 0 files examined -- not a pass",
    )


__all__ = [
    "ASSERTION_RE",
    "REQUIRED_BY_LEVEL",
    "SCOPE_WINDOW_LINES",
    "WAIVER_MARKER",
    "added_lines",
    "declared_level",
    "find_assertions",
    "in_scope_files",
    "scan_repo",
    "scan_staged",
    "staged_files",
]
