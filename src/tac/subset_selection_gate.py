# SPDX-License-Identifier: MIT
"""Guard: a NEW subset-taking site must declare how it selected (`ddm_ss1`, landing 2).

The sister of :mod:`tac.subset_selection`. That module makes a representative
subset *available*; this one makes a silent one *refusable*.

What it refuses
---------------
A newly-added slice of the form ``something[:n_pairs]`` (or ``[:num_frames]``,
``[:args.n_pairs]``, ...) in a file that does not import the canonical selector
and does not carry a waiver. That signature is the measured generator of the
defect: ``ddm_na2`` found **110 tools** slicing ``[:n]`` on pairs/frames with
**zero** offering an alternative, so an omitted selection mode did not mean
"unknown", it meant "prefix" -- on the pose axis a prefix measures **2.54-4.21x
HARDER** than the population, which is precisely the false-negative shape. The
pose ratios were re-derived by ``ddm_mi1`` from receipt sha256
``d2853c92090c28ebe558ece4a21b2847b55e25c9d768bef167bcba9dc67b72e5``.

Where it runs, and why that is the whole design
------------------------------------------------
MEASURED 2026-08-03: ``python -m tac.preflight --no-codebase`` reports
``examined 0 of 27 declared`` -- and ``--no-codebase`` is the pre-commit hook's
DEFAULT mode. A gate wired only into ``preflight_all()`` therefore does not fire
on a normal commit. Catalog #401 (``check_verdict_pairs_default_is_n600``) is
STRICT and still never runs at commit time for exactly this reason; verified by
grepping the hook-mode output for its tag and getting 0 hits.

So this guard is called DIRECTLY from ``tools/preflight_hook.py``, following the
precedent that module already set for the Catalog #184 heavy-import scan
("There is no always-run preflight scope to move it into ... so the hook must
call it directly").

It is scoped to the **staged diff**, and to **added lines only**. That is not a
convenience, it is what makes STRICT honest here:

* Repo-wide there are **207 matching sites across 136 files**, out of **10,699**
  in-scope tracked ``.py`` files (measured 2026-08-03 by :func:`scan_repo`; the
  denominator drifts as sisters land, so it is stated with its date rather than
  as a fact). Refusing those wholesale would be a 136-file migration nobody asked
  for, and would train everyone to write waivers.
* On a clean commit the added-line count is **0**, so STRICT-from-byte-one costs
  nothing and cannot be silently defeated.
* Editing a file that already contains an old site does not fire. You are
  refused only for a site *you are adding*.

The repo-wide sweep is available via :func:`scan_repo` so the 207-site backlog is
visible as debt rather than as a passing green light. It is deliberately NOT
wired into ``preflight_all`` and NOT on the commit path: it reads 10,699 files
and takes **18.4 s** (measured 2026-08-03, down from 41.4 s once the double-parse
was removed). It is an on-demand debt report, run it directly::

    python -c "from tac.subset_selection_gate import scan_repo; scan_repo()"

Waiver
------
Same-line ``# SUBSET_SELECTION_OK:<rationale>`` with a real rationale.
Placeholder rationales (``<rationale>``, ``TBD``, ...) are rejected by the shared
:func:`tac.confound_gates._rationale_ok`, per the Catalog #287 discipline -- so
this gate's own docstring examples cannot self-waive.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from tac.confound_gates import REPO_ROOT, _finish, _rationale_ok

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

WAIVER_MARKER = "SUBSET_SELECTION_OK"

#: Names that, used as a slice UPPER bound, mark the slice as "take a subset of
#: the pair/frame population" rather than an unrelated truncation. Deliberately
#: narrow: a bare ``[:n]`` is ambiguous (it is also how you trim a bit-unpack or
#: take a top-k ranking), and a gate that fires on ambiguity gets waived into
#: uselessness. These names are unambiguous -- they name the population.
PAIR_COUNT_NAMES: frozenset[str] = frozenset(
    {
        "n_pairs",
        "num_pairs",
        "npairs",
        "n_frames",
        "num_frames",
        "nframes",
        "N_PAIRS",
        "NUM_PAIRS",
        "N_FRAMES",
        "sample_pairs",
        "verdict_pairs",
        "n_verdict_pairs",
        "pair_count",
        "frame_count",
    }
)

#: Importing any of these is taken as "this file selects through the canonical
#: surface", which is what the gate is steering toward.
CANONICAL_MODULES: frozenset[str] = frozenset(
    {
        "tac.subset_selection",
        "subset_selection",
    }
)

_ADDED_LINE_RX = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _attribute_tail(node: ast.AST) -> str | None:
    """Return the trailing name of ``x``, ``a.x`` or ``a.b.x``; else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _uses_canonical_selector(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in CANONICAL_MODULES:
            return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in CANONICAL_MODULES:
                    return True
    return False


def _cheap_reject(text: str) -> bool:
    """True when no pair-count name appears at all, so parsing is pointless.

    MEASURED 2026-08-03: the repo sweep spent 41 s parsing 10,699 files, and the
    overwhelming majority contain none of :data:`PAIR_COUNT_NAMES`. A substring
    pre-filter is strictly conservative -- it can only skip files that cannot
    contain a site, since a site's upper bound IS one of those literal names --
    and it removes the parse for them entirely.
    """
    return not any(name in text for name in PAIR_COUNT_NAMES)


def find_sites(text: str, rel: str, *, tree: ast.Module | None = None) -> list[tuple[int, str]]:
    """Return ``(lineno, sliced_expression_hint)`` for every subset-taking slice.

    A site is a ``Subscript`` whose slice is ``[:<pair-count-name>]`` -- no lower
    bound, no step, and an upper bound that names the pair/frame population.
    Waived and canonical-selector files are filtered by the caller so this stays
    a pure syntactic finder (and stays easy to unit-test).

    ``tree`` lets a caller that has already parsed pass the AST in rather than
    paying for a second parse; :func:`_violations_for_text` used to parse every
    file TWICE (once here, once for the canonical-import check), which on a
    3.8 MB module is not a rounding error.
    """
    if tree is None:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        sl = node.slice
        if not isinstance(sl, ast.Slice):
            continue
        if sl.lower is not None or sl.step is not None or sl.upper is None:
            continue
        upper = _attribute_tail(sl.upper)
        if upper is None or upper not in PAIR_COUNT_NAMES:
            continue
        target = _attribute_tail(node.value) or "<expr>"
        out.append((node.lineno, f"{target}[:{upper}]"))
    del rel  # kept in the signature for call-site readability
    return out


def _violations_for_text(
    text: str,
    rel: str,
    *,
    only_lines: set[int] | None = None,
) -> list[str]:
    if _cheap_reject(text):
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    if _uses_canonical_selector(tree):
        return []
    lines = text.splitlines()
    out: list[str] = []
    for lineno, hint in find_sites(text, rel, tree=tree):
        if only_lines is not None and lineno not in only_lines:
            continue
        source_line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
        if _waiver_on_line(source_line):
            continue
        out.append(
            f"{rel}:{lineno}: `{hint}` takes a subset with no declared selection "
            f"mode, so it is a video-order PREFIX (measured: prefix is 2.54-4.21x "
            f"HARDER than the population on the pose axis; ddm_mi1 receipt "
            f"d2853c92090c). Select via "
            f"`tac.subset_selection.select(..., mode=...)` -- which also reports the "
            f"subset/population ratio -- or add "
            f"`# {WAIVER_MARKER}:<why a prefix is correct here>`."
        )
    return out


def _waiver_on_line(source_line: str) -> bool:
    m = re.search(r"#[ \t]*" + re.escape(WAIVER_MARKER) + r":[ \t]*(\S.*)", source_line)
    return bool(m and _rationale_ok(m.group(1)))


def seal_pinned_custody_dirs(repo_root: Path) -> tuple[str, ...]:
    """Repo-relative dirs whose trees are candidate_seal.v1 byte-identity custody.

    A directory carrying a ``CANDIDATE_SEAL_*.json`` is a sha-pinned custody
    copy of shipped runtime bytes (candidate_seal.v1 pins every member file's
    sha256). A scanner that demands an in-file edit there — e.g. a same-line
    waiver comment — would CHANGE the pinned bytes and break the custody it
    exists to protect (Catalog #109 pristine-clone class; the jg5 sub-0.15
    decoder commit incident, 2026-08-20). Skipping these files loses no
    protection: the seal itself is the drift detector for that tree.

    Discovery is INDEX-based (tracked + staged names), never a filesystem walk,
    so the cost is two cheap git calls even on a 60k-file tree. A git failure
    returns () — fail toward SCANNING (worst case a refusal), never toward a
    silent skip.
    """
    dirs: set[str] = set()
    for cmd in (
        ["git", "ls-files"],
        ["git", "diff", "--cached", "--name-only", "--diff-filter=A"],
    ):
        try:
            out = subprocess.check_output(cmd, cwd=repo_root, text=True)
        except (subprocess.CalledProcessError, OSError):
            return ()
        for line in out.splitlines():
            name = line.rsplit("/", 1)[-1]
            if name.startswith("CANDIDATE_SEAL_") and name.endswith(".json"):
                parent = line.rsplit("/", 1)[0] if "/" in line else ""
                if parent:  # a repo-root seal must never blanket the whole repo
                    dirs.add(parent)
    return tuple(sorted(dirs))


def _under_custody_dir(rel: str, custody_dirs: tuple[str, ...]) -> bool:
    return any(rel == d or rel.startswith(d + "/") for d in custody_dirs)


def staged_py_files(repo_root: Path) -> list[str] | None:
    """Staged ``.py`` paths, repo-relative, or ``None`` if git could not tell us.

    ``None`` means the staged set is UNKNOWN, which is NOT the same as "nothing
    was staged" -- ddm_si1 review, task #929. This function previously returned
    ``[]`` on git failure, four lines above :func:`added_lines`, which had
    already been given exactly this ``None`` channel for exactly this reason.
    The empty list flowed into :func:`scan_staged` as a zero denominator and
    printed ``VACUOUS: 0 staged .py files``: a false statement about the world
    (git broke; the commit was not empty) wearing the shape of an honest one.
    """
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            cwd=repo_root,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return [line for line in out.splitlines() if line.endswith(".py")]


def added_lines(repo_root: Path, rel: str) -> set[int] | None:
    """Line numbers ADDED to ``rel`` in the staged diff, or ``None`` if unknown.

    ``None`` means git could not tell us -- which is NOT the same as "no lines
    were added", and the caller must not treat it as one.

    ``-U0`` so the hunk headers bound exactly the added lines and nothing else;
    an unrelated edit to a file that already contains an old site therefore does
    not drag that site into scope.
    """
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "-U0", "--", rel],
            cwd=repo_root,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        # None, NOT set(). Review pass 1 finding: returning an empty set here made
        # a git failure indistinguishable from "this file added no lines", and the
        # caller passes the result as `only_lines` -- so an empty set filtered out
        # EVERY site and the file passed SILENTLY. That is the vacuity-equals-pass
        # genus inside the guard built to close it. None forces the caller to say
        # so out loud.
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
    """Refuse newly-added silent subset sites in the staged diff.

    STRICT by default, which is only honest because the scope is added lines: a
    clean commit has zero, so the live count is 0 by construction rather than by
    a migration claim.
    """
    root = Path(repo_root or REPO_ROOT)
    rels = list(files) if files is not None else staged_py_files(root)
    if rels is None:
        # git could not enumerate the staged set, so the scope is UNKNOWN.
        # Passing here would make a broken git indistinguishable from a clean
        # commit -- the exact genus this gate exists to close -- and reporting
        # it as "0 staged files" would additionally be untrue. Refuse.
        return _finish(
            name="check_subset_selection_mode_declared_staged",
            tag="subset-selection-mode-staged",
            violations=[
                "git could not enumerate staged .py files: the staged scope is "
                "UNKNOWN, which is not the same as empty, so this gate examined "
                "nothing and must not report a pass"
            ],
            strict=strict,
            verbose=verbose,
            ok_detail="",
        )
    violations: list[str] = []
    examined = 0
    unexamined: list[str] = []
    custody_dirs = seal_pinned_custody_dirs(root)
    custody_skipped = 0
    for rel in rels:
        if _under_custody_dir(rel, custody_dirs):
            # Byte-identity custody: a waiver edit here would break the seal
            # this tree ships under. Counted and reported, never silent.
            custody_skipped += 1
            continue
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            unexamined.append(f"{rel} unreadable ({exc.__class__.__name__})")
            continue
        scope = added_lines(root, rel)
        if scope is None:
            # Cannot determine the added lines -> cannot scope the scan. Scanning
            # the WHOLE file would flag pre-existing sites the author did not add;
            # scanning nothing would pass silently. So: examine nothing, and SAY SO.
            unexamined.append(f"{rel} added-line range unavailable (git diff failed)")
            continue
        examined += 1
        violations.extend(_violations_for_text(text, rel, only_lines=scope))

    detail = f"{examined} of {len(rels)} staged .py file(s) examined"
    if custody_skipped:
        detail += (
            f"; {custody_skipped} seal-pinned custody file(s) skipped by design "
            "(candidate_seal.v1 byte-identity, Catalog #109 class)"
        )
    if unexamined:
        detail += f"; NOT examined (not a pass): {'; '.join(unexamined)}"
        # Review pass 2, on pass 1's own fix: `_finish` prints ok_detail ONLY when
        # there are no violations, so the unexamined list -- the thing pass 1 added
        # to stop a silent skip -- disappeared exactly when output matters most.
        # Print it unconditionally instead of relying on the shared epilogue.
        if verbose:
            print(f"  [subset-selection-mode-staged] {len(unexamined)} file(s) NOT examined:")
            for note in unexamined:
                print(f"    - {note}")
    return _finish(
        name="check_subset_selection_mode_declared_staged",
        tag="subset-selection-mode-staged",
        violations=violations,
        strict=strict,
        verbose=verbose,
        # The DENOMINATOR, always. `examined 0` is VACUOUS -- a commit with no
        # staged .py files is not a clean bill of health, and saying "OK" without
        # the count is how vacuity becomes indistinguishable from a pass.
        ok_detail=detail
        if rels
        else "VACUOUS: 0 staged .py files -- nothing examined, not a pass",
    )


def in_scope_py_files(repo_root: Path) -> list[str] | None:
    """Git-TRACKED ``.py`` files, excluding vendored public-PR intake trees.

    Scope note, measured 2026-08-03: on disk ``experiments/results/`` holds 49,131
    of the 63,173 ``.py`` files under src/tools/scripts/experiments (77.8%), but
    git TRACKS only 5 of them. Using ``git ls-files`` therefore makes the vendored
    exclusion nearly free -- 10,696 of 10,701 tracked ``.py`` files are in scope --
    whereas a filesystem walk would need the exclusion to avoid a 78% vendored
    denominator. Both numbers are true about different universes; this function
    picks the git one and says so.

    Returns ``None`` -- never ``[]`` -- when git could not answer, for the same
    reason as :func:`staged_py_files`: an empty in-scope list would make a
    broken git read as "the whole repo is clean" (ddm_si1, task #929).
    """
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "*.py"], cwd=repo_root, text=True
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    custody_dirs = seal_pinned_custody_dirs(repo_root)
    return [
        line
        for line in out.splitlines()
        if line
        and not line.startswith("experiments/results/")
        and not _under_custody_dir(line, custody_dirs)
    ]


def scan_repo(
    *,
    repo_root: str | Path | None = None,
    strict: bool = False,
    verbose: bool = True,
    files: Iterable[str] | None = None,
) -> list[str]:
    """Whole-repo sweep -- the standing backlog, warn-only.

    Warn-only and staying that way until the migration lands: the measured live
    count is **207 sites in 136 files** of 10,699 in-scope tracked ``.py``
    (2026-08-03). Reporting that as debt is honest; flipping it STRICT would
    refuse every commit in the repo, and calling it clean would be the lie this
    landing exists to prevent. The STRICT surface is :func:`scan_staged`, whose
    added-lines scope makes the live count 0 without a migration claim.
    """
    root = Path(repo_root or REPO_ROOT)
    rels = list(files) if files is not None else in_scope_py_files(root)
    if rels is None:
        # Same channel as scan_staged: an unanswerable git is not an empty repo.
        return _finish(
            name="check_subset_selection_mode_declared_repo",
            tag="subset-selection-mode-repo",
            violations=[
                "git could not enumerate tracked .py files: the repo scope is "
                "UNKNOWN, not empty, so this sweep examined nothing"
            ],
            strict=strict,
            verbose=verbose,
            ok_detail="",
        )
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
        name="check_subset_selection_mode_declared",
        tag="subset-selection-mode",
        violations=violations,
        strict=strict,
        verbose=verbose,
        ok_detail=f"{examined} tracked in-scope .py file(s) examined"
        if examined
        else "VACUOUS: 0 files examined -- not a pass",
    )


__all__ = [
    "CANONICAL_MODULES",
    "PAIR_COUNT_NAMES",
    "WAIVER_MARKER",
    "added_lines",
    "find_sites",
    "in_scope_py_files",
    "scan_repo",
    "scan_staged",
    "staged_py_files",
]
