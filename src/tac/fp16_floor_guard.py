# SPDX-License-Identifier: MIT
"""Canonical detector for the "fp16 cast destroys its own floor" bug class.

THE BUG THIS EXTINCTS (``ddm_fx4``, 2026-08-19, commit ``61c41ab166``)
---------------------------------------------------------------------
A positive floor is applied to a scale, and a NARROWING CAST one line later
silently undoes it::

    ((maxs - mins).float() / 255.0).clamp(min=1e-8).to(torch.float16)

fp16's smallest positive (subnormal) value is ``2**-24 = 5.960464e-08``. Under
round-to-nearest-even every value below half of that rounds to **zero**, so the
``clamp(min=1e-8)`` guard -- written precisely to keep the scale non-zero -- is
destroyed by the very next operation. The stored scale is 0, and the dequantiser
divides by it. Found live at **35 sites across 22 files**.

The class also spans two statements, which a same-statement detector would miss::

    scale = max(max_abs, 1e-8) / 127.0                 # floor here
    scale_fp16 = torch.tensor([scale], torch.float16)  # cast there

CANONICAL FIX
-------------
Re-apply the floor **after** the cast, at :data:`FP16_MIN_POSITIVE`, on the value
actually stored and read back. The fix is byte-neutral where the encoder already
worked: the clamp engages only below ``2**-24``.

WHY THIS LIVES IN A LIBRARY MODULE
----------------------------------
``ddm_fx4`` shipped this predicate inside its own test sweep and left the
preflight wire-in owed, because ``src/tac/preflight.py`` was owned by another arm
that session. Re-typing the predicate in preflight would have produced TWO
detectors for ONE class -- the split-bank disease
(``orphan_sweeps_that_do_not_write_the_store_are_the_disease``): they drift, and
then the class has two different answers. This module is the single
implementation; the fx4 regression sweep and the preflight gate both consume it.

WHAT THE GAUGE READS IF THE CURE IS APPLIED AND NOTHING ELSE CHANGES
--------------------------------------------------------------------
The detector reads the LAST floor in the statement and requires it to sit AFTER
the ``float16`` token and to be fp16-representable. Adding a comment, a marker,
or a docstring moves it by exactly zero; only re-flooring the stored value does.
:func:`scan_repo_for_fp16_destroyed_floors` also returns its ``scanned`` count so
a caller can refuse a VACUOUS pass -- a sweep that matched nothing must never
read as green (the silent-instrument class).

Axis: static source analysis. Never a score.
"""

from __future__ import annotations

import io
import re
import subprocess
import tokenize
from pathlib import Path

__all__ = [
    "EXCLUDED_PATH_PARTS",
    "FP16_MIN_POSITIVE",
    "SAFE_FLOOR_NAMES",
    "SCANNED_ROOTS",
    "WAIVER_TOKEN",
    "analyze_text",
    "floor_is_fp16_safe",
    "scan_repo_for_fp16_destroyed_floors",
    "scan_text_for_fp16_destroyed_floors",
    "waiver_rationale_is_substantive",
]

#: fp16's smallest positive (subnormal) value, ``2**-24``. Any floor at or above
#: this survives the narrowing cast; anything below it can round to zero.
FP16_MIN_POSITIVE = 5.960464477539063e-08

SCANNED_ROOTS: tuple[str, ...] = ("src/tac", "tools", "experiments", "scripts")
#: Frozen custody snapshots and vendored upstream are historical, not live code.
#: Test trees build deliberate breaching fixtures and are excluded by design.
EXCLUDED_PATH_PARTS: tuple[str, ...] = ("experiments/results/", "/tests/", "upstream/")

#: A floor named by the canonical constant is safe without a numeric parse.
SAFE_FLOOR_NAMES: tuple[str, ...] = ("_FP16_MIN_POSITIVE", "FP16_MIN_POSITIVE")

_FLOAT16 = re.compile(r"float16|\.half\s*\(\)")
_FLOOR = re.compile(
    r"(?:clamp\s*\(\s*min\s*=|clamp_min\s*\(|max\s*\(\s*[A-Za-z_][\w.]*\s*,)\s*([\w.+-]+)"
)
_ASSIGN = re.compile(r"^([A-Za-z_]\w*)\s*=(?!=)")
#: How many preceding statements to consider for a cross-statement floor.
_LOOKBACK = 3
#: How many FOLLOWING statements may carry the post-cast cure.
_LOOKFORWARD = 3

#: Same-line waiver for a site that is genuinely not auditable statically.
WAIVER_TOKEN = "FP16_POSTCAST_FLOOR_OK"
_WAIVER_RE = re.compile(rf"{WAIVER_TOKEN}\s*:\s*(.+?)\s*$")
_PLACEHOLDER_RATIONALES = frozenset(
    {
        "<reason>",
        "<rationale>",
        "<reason_here>",
        "<rationale_here>",
        "reason",
        "rationale",
        "todo",
        "tbd",
        "n/a",
        "none",
        "placeholder",
    }
)
#: Characters that, when they occur INSIDE a string literal, corrupt either the
#: comment strip or the bracket-depth statement split.
_STRING_HAZARD_CHARS = frozenset("#()[]{}")

#: A file can only host the defect if it BOTH narrows to fp16 and applies a
#: floor. Both tokens present is a strict SUPERSET of "has a real site", so this
#: pre-filter can skip work but can never hide a violation.
_FLOOR_TOKEN = re.compile(r"clamp\s*\(\s*min\s*=|clamp_min\s*\(|max\s*\(")


def waiver_rationale_is_substantive(rationale: str) -> bool:
    """Is a ``FP16_POSTCAST_FLOOR_OK`` rationale real, per Catalog #287?

    Placeholder literals and sub-4-character stubs are rejected so this gate's
    own documentation cannot self-waive.
    """
    cleaned = rationale.strip().rstrip(".").strip()
    if len(cleaned) < 4:
        return False
    return cleaned.lower() not in _PLACEHOLDER_RATIONALES


def _neutralize_prose(text: str) -> tuple[str, frozenset[int]]:
    """Return ``(code_only_text, waived_line_numbers)``.

    Replaces the previous ``line.find("#")`` comment strip, which had a SILENT
    total-blindness failure: a ``#`` inside a string literal (``"#fff"``) both
    truncated the line and unbalanced its brackets, so
    :func:`_logical_statements` returned NOTHING for the remainder of the file.
    The file then scanned clean AND contributed zero to the denominator, so the
    vacuity guard could not see it either -- an instrument that cannot tell
    "checked clean" from "did not check".

    The cure is to use the real tokenizer instead of a lexical guess (the same
    cure ``ddm_fx3`` prescribed for Catalog #330):

    * COMMENT tokens are blanked -- and captured first, so a same-line
      ``# FP16_POSTCAST_FLOOR_OK:<reason>`` waiver survives the blanking.
    * MULTI-LINE strings (docstrings, the memo-quoting prose this detector must
      never read as code) are blanked entirely.
    * SINGLE-LINE strings keep their text but lose ``#`` and brackets, so
      ``astype("float16")`` is still detected as a real narrowing cast while a
      string can no longer corrupt comment-stripping or bracket depth.

    Line numbers and line counts are always preserved. Falls back to the raw
    text on a tokenize failure -- a file we cannot parse is better over-scanned
    than silently skipped.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError, ValueError):
        return text, frozenset()

    grid = [list(line) for line in text.splitlines()]
    waived: set[int] = set()

    def _edit(srow: int, scol: int, erow: int, ecol: int, *, blank: bool) -> None:
        for row in range(srow, erow + 1):
            if not (1 <= row <= len(grid)):
                continue
            line = grid[row - 1]
            lo = scol if row == srow else 0
            hi = ecol if row == erow else len(line)
            for col in range(max(lo, 0), min(hi, len(line))):
                if blank or line[col] in _STRING_HAZARD_CHARS:
                    line[col] = " "

    string_types = {tokenize.STRING}
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        tok_type = getattr(tokenize, name, None)
        if tok_type is not None:
            string_types.add(tok_type)

    for tok in tokens:
        srow, scol = tok.start
        erow, ecol = tok.end
        if tok.type == tokenize.COMMENT:
            match = _WAIVER_RE.search(tok.string)
            if match and waiver_rationale_is_substantive(match.group(1)):
                waived.add(srow)
            _edit(srow, scol, erow, ecol, blank=True)
        elif tok.type in string_types:
            _edit(srow, scol, erow, ecol, blank=erow > srow)

    return "\n".join("".join(row) for row in grid), frozenset(waived)


def _logical_statements(text: str) -> list[tuple[int, int, str]]:
    """Return ``(start_line, end_line, joined_statement)`` merging continuations."""
    out: list[tuple[int, int, str]] = []
    lines = text.splitlines()
    buf: list[str] = []
    start = 0
    depth = 0
    for i, raw in enumerate(lines, start=1):
        if not buf:
            start = i
        buf.append(raw.strip())
        depth += raw.count("(") + raw.count("[") - raw.count(")") - raw.count("]")
        if depth <= 0:
            out.append((start, i, " ".join(buf)))
            buf, depth = [], 0
    if buf:
        # An unterminated bracket run still has to be reported, not dropped.
        out.append((start, len(lines), " ".join(buf)))
    return out


def _inherited_floors(
    statements: list[tuple[int, int, str]], index: int, stmt: str,
) -> list[re.Match[str]]:
    """Floors applied in a nearby PRECEDING statement to a name this one casts."""
    out: list[re.Match[str]] = []
    for back in range(1, _LOOKBACK + 1):
        j = index - back
        if j < 0:
            break
        prev = statements[j][2]
        assigned = _ASSIGN.match(prev)
        if not assigned:
            continue
        name = assigned.group(1)
        if not re.search(rf"\b{re.escape(name)}\b", stmt):
            continue
        out.extend(_FLOOR.finditer(prev))
    return out


def _cured_in_a_later_statement(
    statements: list[tuple[int, int, str]], index: int, stmt: str,
) -> bool:
    """Is the value cast here re-floored by a nearby FOLLOWING statement?

    The canonical cure -- "re-apply the floor AFTER the cast" -- is naturally
    written across two statements::

        sf = torch.tensor([scale], dtype=torch.float16)
        sf = sf.clamp_min(_FP16_MIN_POSITIVE)          # the cure

    A detector that only looks for a trailing floor on the SAME line reports
    that correctly-cured site as a violation. On a STRICT gate wired into
    ``preflight_all`` that refuses the engineer's commit and tells them to do
    what they have already done, which is how a gate earns a waiver it should
    never have needed.
    """
    assigned = _ASSIGN.match(stmt)
    if not assigned:
        return False
    name = assigned.group(1)
    for forward in range(1, _LOOKFORWARD + 1):
        j = index + forward
        if j >= len(statements):
            break
        nxt = statements[j][2]
        if not re.search(rf"\b{re.escape(name)}\b", nxt):
            continue
        floors = list(_FLOOR.finditer(nxt))
        if floors and all(floor_is_fp16_safe(m.group(1)) for m in floors):
            return True
    return False


def floor_is_fp16_safe(token: str) -> bool:
    """Is ``token`` a floor that survives narrowing to fp16?

    A non-literal, non-canonical floor (a bare variable) is not auditable
    statically, so it is treated as safe: the sweep reports only what it can
    PROVE, which keeps a warn/strict gate free of unfalsifiable rows.
    """
    if token in SAFE_FLOOR_NAMES:
        return True
    try:
        return float(token) >= FP16_MIN_POSITIVE
    except ValueError:
        return True


def _analyze_statements(
    statements: list[tuple[int, int, str]], rel: str, waived: frozenset[int],
) -> tuple[list[str], int]:
    """Core predicate over already-split statements. Returns (violations, sites)."""
    violations: list[str] = []
    scanned = 0
    for index, (lineno, end_lineno, stmt) in enumerate(statements):
        cast = _FLOAT16.search(stmt)
        if not cast:
            continue
        same_line = list(_FLOOR.finditer(stmt))
        floors = same_line + _inherited_floors(statements, index, stmt)
        if not floors:
            continue  # no guard intended at this site
        scanned += 1
        # Only a floor applied AFTER the cast protects the stored value.
        trailing = [m for m in same_line if m.start() > cast.end()]
        if trailing and all(floor_is_fp16_safe(m.group(1)) for m in trailing):
            continue  # re-floored after the cast -- cured
        if all(floor_is_fp16_safe(m.group(1)) for m in floors):
            continue  # every floor is fp16-representable anyway
        if _cured_in_a_later_statement(statements, index, stmt):
            continue  # the cure is written on the NEXT line -- also cured
        if any(line in waived for line in range(lineno, end_lineno + 1)):
            continue  # adjudicated, with a substantive same-line rationale
        violations.append(f"{rel}:{lineno}: {stmt[:150]}")
    return violations, scanned


def analyze_text(text: str, rel: str) -> tuple[list[str], int]:
    """Return ``(violations, guard_and_cast_sites)`` for one source file.

    TWO-STAGE, for wall-clock. Neutralizing prose costs a tokenize pass, and
    ``ddm_sp2`` measured the equivalent parse over every fp16-mentioning file at
    5.8 s of a 30.0 s preflight budget -- the dominant term. Stage 1 therefore
    skips files that cannot host the defect at all.

    The pre-filter is a pure TOKEN-PRESENCE test (an fp16 narrowing AND a floor),
    which is a strict SUPERSET of "has a real site". That matters: ``ddm_sp3``
    measured that the previous stage 1 -- running the whole raw predicate and
    early-returning on ``raw_scanned == 0`` -- inherited the raw text's
    ``#``-in-a-string bracket corruption and could return NOTHING for a file that
    does contain a live violation, while also contributing 0 to the denominator
    that is supposed to catch exactly that. A superset pre-filter can skip work;
    it cannot hide a violation.
    """
    if not _FLOAT16.search(text) or not _FLOOR_TOKEN.search(text):
        return [], 0
    code, waived = _neutralize_prose(text)
    return _analyze_statements(_logical_statements(code), rel, waived)


def scan_text_for_fp16_destroyed_floors(text: str, rel: str) -> list[str]:
    """Return ``rel:lineno: statement`` rows where an fp16 cast destroys a floor."""
    return analyze_text(text, rel)[0]


def count_guard_and_cast_sites(text: str) -> int:
    """Number of guard-and-cast sites in ``text`` -- the sweep's DENOMINATOR."""
    return analyze_text(text, "?")[1]


def _candidate_files(repo_root: Path, roots: tuple[str, ...]) -> list[Path]:
    """Files that can possibly host the defect: they must mention fp16 at all.

    Measured (``ddm_sp2``, 2026-08-19): rglob-and-read over ``src/tac`` + ``tools``
    + ``experiments`` + ``scripts`` cost **9.67 s**, against a
    ``DEFAULT_PREFLIGHT_CLI_TIMEOUT_S`` of 30.0 s -- 32% of the entire preflight
    budget spent reading files that cannot match, inside a STRICT gate. One
    ripgrep pass replaces it. Falls back to the pure-Python walk when ripgrep is
    unavailable, so the RESULT is identical either way and only the cost differs.
    """
    existing = [r for r in roots if (repo_root / r).exists()]
    if not existing:
        return []
    try:
        proc = subprocess.run(
            ["rg", "-l", r"float16|\.half\s*\(", "-g", "*.py", *existing],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=20.0,
        )
        if proc.returncode in (0, 1):
            out: list[Path] = []
            for line in proc.stdout.splitlines():
                if not line:
                    continue
                candidate = repo_root / line
                if candidate.is_file():
                    out.append(candidate)
            return sorted(out)
    except (OSError, subprocess.SubprocessError):
        pass
    fallback: list[Path] = []
    for root in existing:
        fallback.extend(sorted((repo_root / root).rglob("*.py")))
    return fallback


def scan_repo_for_fp16_destroyed_floors(
    repo_root: Path, *, roots: tuple[str, ...] = SCANNED_ROOTS,
) -> tuple[list[str], int]:
    """Scan ``roots`` under ``repo_root``.

    Returns ``(violations, scanned)``. The caller MUST treat ``scanned == 0`` as
    a vacuous run rather than a pass -- an instrument that measured nothing is
    not a green instrument.
    """
    violations: list[str] = []
    scanned = 0
    for path in _candidate_files(repo_root, roots):
        rel = path.relative_to(repo_root).as_posix()
        if any(part in rel for part in EXCLUDED_PATH_PARTS):
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        file_violations, file_sites = analyze_text(text, rel)
        scanned += file_sites
        violations.extend(file_violations)
    return violations, scanned
