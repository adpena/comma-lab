# SPDX-License-Identifier: MIT
"""ddm_sf1 — REFUSE a solved coefficient that does not record what it was fitted to.

ddm_fs1 landed the LADDER (FRESH / STALE_INTERPOLATED / STALE_EXTRAPOLATED /
STALE_FOREIGN_PARTNER / UNDETERMINED_NO_CONTEXT, absence never FRESH).  It is a
correct instrument with an EMPTY INPUT: MEASURED 2026-08-02, ``fitted_against``
appeared in 0 producers repo-wide, so every live row resolved to
``UNDETERMINED_NO_CONTEXT`` and the ladder could not grade anything.  This module
is the producer-side half: it refuses a solver that emits a solved coefficient
without stamping the partner state that coefficient was solved against.

WHY A GATE AND NOT A CONVENTION.  The defect is SILENT by construction.  A
sequential coordinate descent reports a correctly-negative delta at every stage,
so every receipt reads healthy while an earlier coordinate is left fitted against
a value that no longer exists.  Nothing in the artifact records the partner, so
the staleness is recoverable only by hand-archaeology across arms -- which is how
all three known instances were actually found, one at a time, after they had
already shipped.  A convention that is not enforced degrades to the same silence.

THE SCOPE, AND WHY IT IS THIS ONE (MEASURED 2026-08-02).  Of 90 ``.py`` files that
emit per-item JSONL rows, 25 also perform a fit/solve, and of those exactly 3
emit a solved coefficient beside a partner coordinate in the same row.  This gate
takes the 2 that emit a PHOTOMETRIC pair ``(a, b)`` beside a partner -- the
population where the defect is MEASURED (244 stale pairs, 100 outside the fitted
set; see ``ddm_fs1`` and ``ddm_ft1``).  The vocabulary below is the knob: widening
it widens the gate, and it should be widened as further solved-coefficient
families are measured rather than pre-emptively.

THE VACUITY RULE IS LOAD-BEARING HERE.  An empty scan and a clean scan emit the
same symbol in most gates, which is how a gate rots into a silent pass -- the
recorded ``vacuity_is_indistinguishable_from_pass`` genus.  So this gate REFUSES
on an empty population and always reports its denominator.  If someone renames
the producers out of scope, the gate says VACUOUS, not PASS.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tac.canonical_equations.ddm_fs1_coordinate_fit_staleness_20260802 import (
    FIT_CONTEXT_KEY,
)

#: Directories that can hold a shipping solver.  Tests and vendored public-PR
#: intake clones are excluded: intake clones must stay pristine (CLAUDE.md
#: "Forbidden in-place edits to public PR intake clones").
SCAN_ROOTS: tuple[str, ...] = ("experiments", "tools", "src/tac")
#: Matched against the PATH.  Deliberately does NOT include ``test_``: that must
#: be matched against the FILENAME only (see ``_iter_py``).  As a path substring
#: it silently swallows any ancestor directory containing ``test_`` -- including
#: pytest's own ``tmp_path``, which named the scan 0-of-0 and would have made the
#: gate untestable while still reading green on the live tree.
#: ``experiments/results/`` is RUN OUTPUT (49,131 ``.py`` under it, vs 4,416 in
#: the three roots combined once excluded).  Any solver appearing there is a
#: point-in-time SNAPSHOT of a run, never the live producer, so scanning it
#: costs ~10x the wall clock for zero additional subjects.  If the real
#: producers ever move under it, the VACUITY rule fires rather than passing.
EXCLUDE_MARKERS: tuple[str, ...] = ("_intake_", "/tests/", "/.git/",
                                    "/worktrees/", "/node_modules/",
                                    "experiments/results/")

#: Keys that name a SOLVED coefficient in an emitted row.  ``a``/``b`` are the
#: photometric gain/bias solved by 2-param Gauss-Newton in the v4c/v4d line.
SOLVED_COEFFICIENT_KEYS: tuple[str, ...] = ("a", "b")

#: Keys that name a PARTNER the coefficient was held against -- a co-coordinate
#: (pose ``p``, rolling-shutter ``beta``) or a discrete selector.
PARTNER_KEYS: tuple[str, ...] = ("p", "beta_mag", "beta_idx", "selector")

#: Tokens that indicate the file actually SOLVES rather than merely forwarding
#: a row someone else solved.  A pure re-emitter has no fit context to stamp.
FIT_TOKENS: tuple[str, ...] = ("_refit", "gauss_newton", "_solve", "def _refine",
                               "_select(", "_fit_")

#: Tokens that satisfy the requirement.  Three admissible forms, because there
#: are three legitimate producer shapes:
#:   * ``stamp_fit_context`` -- the solver STAMPS what it solved against;
#:   * ``FIT_CONTEXT_KEY``   -- the imported constant, used by a stage that
#:     CARRIES an upstream context forward.  Carrying is not a weaker form of
#:     stamping, it is the CORRECT form for a stage that moves a partner without
#:     re-solving the coefficient (mq1 refines {p0,p1,p2,beta} and never touches
#:     (a,b)).  A fresh stamp there would assert a freshness never re-established
#:     -- it would convert a true STALE into a false FRESH, which is strictly
#:     worse than no stamp at all;
#:   * the bare ``fitted_against`` literal, for a producer that writes the key
#:     directly.
STAMP_TOKENS: tuple[str, ...] = (FIT_CONTEXT_KEY, "stamp_fit_context",
                                 "FIT_CONTEXT_KEY")

_WAIVER = re.compile(r"#\s*FIT_CONTEXT_STAMP_WAIVED\s*:\s*(?P<why>[^\n]*)")
#: Placeholder rationales are rejected so the gate's own docstring cannot waive it
#: (CLAUDE.md Catalog #287 placeholder-rationale rejection).
_PLACEHOLDERS = frozenset({"", "<rationale>", "<reason>", "rationale", "reason",
                           "tbd", "todo", "placeholder", "n/a", "na", "-"})


def _emits_rows(text: str) -> bool:
    return "json.dumps(rec)" in text or "json.dumps(row)" in text


def _has_key(text: str, key: str) -> bool:
    """Does the source contain a row-dict key literal ``"key":``?"""
    return f'"{key}":' in text or f"'{key}':" in text


def waiver_of(text: str) -> str | None:
    """Return a SUBSTANTIVE waiver rationale, or None.

    A placeholder rationale returns None -- i.e. it does not waive -- so that an
    empty excuse cannot buy the same silence the gate exists to remove.
    """
    for m in _WAIVER.finditer(text):
        why = m.group("why").strip().strip("\"'").rstrip(".").strip()
        if why.lower() not in _PLACEHOLDERS and len(why) >= 8:
            return why
    return None


def classify(path: Path, text: str) -> dict[str, Any]:
    """Is ``path`` in scope, and if so does it stamp?

    Returns a verdict row for every file examined -- including the out-of-scope
    ones, so the denominator is auditable rather than asserted.
    """
    in_scope = bool(
        _emits_rows(text)
        and any(t in text for t in FIT_TOKENS)
        and all(_has_key(text, k) for k in SOLVED_COEFFICIENT_KEYS)
        and any(_has_key(text, k) for k in PARTNER_KEYS)
    )
    if not in_scope:
        return {"path": str(path), "in_scope": False, "stamped": None,
                "waived": None, "verdict": "OUT_OF_SCOPE"}
    stamped = any(t in text for t in STAMP_TOKENS)
    waived = waiver_of(text)
    verdict = ("STAMPED" if stamped else
               "WAIVED" if waived else "UNSTAMPED_REFUSED")
    return {"path": str(path), "in_scope": True, "stamped": stamped,
            "waived": waived, "verdict": verdict}


def _iter_py(repo_root: Path, roots: Sequence[str]) -> Iterable[Path]:
    for r in roots:
        base = repo_root / r
        if not base.exists():
            continue
        # os.walk with in-place dirnames pruning, NOT rglob: rglob would still
        # WALK the 49k-file results tree before the path filter rejected it,
        # which is where the wall clock actually goes.  Pruning skips the
        # subtree outright.
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [
                d for d in sorted(dirnames)
                if not any(m in f"{dirpath}/{d}/" for m in EXCLUDE_MARKERS)
            ]
            for fn in sorted(filenames):
                if not fn.endswith(".py"):
                    continue
                if fn.startswith("test_") or fn.endswith("_test.py"):
                    continue
                yield Path(dirpath) / fn


def scan(repo_root: Path | str = ".", *,
         roots: Sequence[str] = SCAN_ROOTS) -> dict[str, Any]:
    """Scan the producer population and report the census WITH its denominator.

    ``vacuous`` is True when the in-scope population is empty.  That is a
    REFUSAL, not a pass: an empty scope means the gate has lost its subject --
    renamed, moved, or deleted -- and a gate that cannot see its subject must
    say so rather than emit the same green a clean scan emits.
    """
    root = Path(repo_root).resolve()
    rows: list[dict[str, Any]] = []
    scanned = 0
    # A file that emits no row dict cannot be in scope, and the emit marker is a
    # byte-exact literal.  Testing it on RAW BYTES lets the ~99.9% that cannot
    # match skip utf-8 decoding entirely -- the decode, not the read, is the bulk
    # of the cost, and this gate runs on every commit.
    markers = (b"json.dumps(rec)", b"json.dumps(row)")
    for p in _iter_py(root, roots):
        scanned += 1
        try:
            data = p.read_bytes()
        except OSError:
            continue
        if not any(m in data for m in markers):
            continue
        r = classify(p.relative_to(root), data.decode("utf-8", errors="replace"))
        if r["in_scope"]:
            rows.append(r)
    refused = [r for r in rows if r["verdict"] == "UNSTAMPED_REFUSED"]
    vacuous = not rows
    return {
        "scanned_files": scanned,
        "in_scope": len(rows),
        "stamped": sum(r["verdict"] == "STAMPED" for r in rows),
        "waived": sum(r["verdict"] == "WAIVED" for r in rows),
        "refused": len(refused),
        "refused_paths": [r["path"] for r in refused],
        "rows": rows,
        "vacuous": vacuous,
        "ok": (not vacuous) and not refused,
    }


def check_solved_coefficients_stamp_fit_context(
    repo_root: Path | str = ".", *, strict: bool = True,
    roots: Sequence[str] = SCAN_ROOTS,
) -> dict[str, Any]:
    """Preflight entry point.  Raises in strict mode on a refusal OR a vacuum."""
    res = scan(repo_root, roots=roots)
    if res["vacuous"]:
        msg = (f"ddm_sf1 VACUOUS: 0 in-scope producers among {res['scanned_files']} "
               f"scanned .py files. An empty scope is NOT a pass -- the gate has "
               f"lost its subject. Check SOLVED_COEFFICIENT_KEYS/PARTNER_KEYS "
               f"against the live producers.")
    elif res["refused"]:
        msg = (f"ddm_sf1 REFUSED {res['refused']} of {res['in_scope']} in-scope "
               f"producers (scanned {res['scanned_files']}): "
               f"{', '.join(res['refused_paths'])}. Each emits a solved "
               f"coefficient {SOLVED_COEFFICIENT_KEYS} beside a partner "
               f"{PARTNER_KEYS} without recording {FIT_CONTEXT_KEY!r}. Stamp it "
               f"with ddm_fs1.stamp_fit_context() at the emit site, or add "
               f"'# FIT_CONTEXT_STAMP_WAIVED:<substantive rationale>'.")
    else:
        return res
    if strict:
        raise ValueError(msg)
    res["warning"] = msg
    return res
