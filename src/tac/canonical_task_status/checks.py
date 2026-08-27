# SPDX-License-Identifier: MIT
"""Strict validation checks for the canonical task-status ledger."""

from __future__ import annotations

import re
from pathlib import Path

from .contract import CanonicalTaskStatusRow
from .loader import load_canonical_task_status_strict

# ── SPECIFICATION WITHOUT REGISTRATION (ddm_rg5, task #825, 2026-07-31) ───────────────────────
# THE CLASS, and it is the SAME ROOT as the lever-registry blindness fixed in the same landing:
# the mechanism meant to notice orphans cannot see them. A work item gets SPECIFIED in prose — a
# task number in the live focus doc, a lever named in a memo — and every downstream reader takes
# the prose as evidence that the thing EXISTS. Nothing checks.
#
# RECEIPT (R1-C, 2026-07-31, three instances from ONE day): ``#815``/``ddm_bs1`` was skipped in
# this ledger (812,813,814,**816**,817,818) and ``ddm_bs1`` appeared exactly once repo-wide, in
# ``current_focus.md`` itself; ``#822`` likewise; ``lever_reset_operator`` occurred exactly once
# repo-wide — as a SENTENCE instructing that it be built. MAIN then asserted a BLOCKING RELATION
# between #815 and #820, neither of which existed in the canonical ledger at all.
#
# SCOPE, chosen to be precise rather than loud. Only ``.omx/state/current_focus.md`` is scanned:
# it is the live durable-state doc, the one surface where a ``#NNN`` reads as "this is current
# work". Scanning ``.omx/research`` would fire on hundreds of historical references and become a
# false-positive machine, which is how a gate teaches its readers to ignore it.
#
# Only ids at or above ``_TASK_REF_REGISTRATION_BASELINE`` are required to resolve. Two measured
# reasons: (1) the JSONL holds historical rows while the live TaskList is SoT for #200+, so old
# ids legitimately have no row here; (2) a bare ``#NNN`` regex cannot tell a TASK number from a
# CATALOG number — ``#396`` and ``#316`` in the focus doc are catalog gates, not tasks. MEASURED:
# 35 distinct refs, 18 unresolved naively, but only **8** at/above the baseline — and those 8 are
# exactly the genuine orphans (#809/#815/#819/#820/#821/#822/#824/#825), including all five R1-C
# named. Raising the baseline as the ledger advances keeps the signal-to-noise at 1.
_CURRENT_FOCUS_RELPATH = ".omx/state/current_focus.md"
_TASK_REF_RE = re.compile(r"#(\d{3,4})\b")
_TASK_REF_REGISTRATION_BASELINE = 793


def _registered_numeric_task_ids(root: Path) -> set[int]:
    ids: set[int] = set()
    for row in load_canonical_task_status_strict(root):
        raw = str(row.task_id).lstrip("#")
        if raw.isdigit():
            ids.add(int(raw))
    return ids


def unregistered_task_refs_in_current_focus(
    repo_root: str | Path | None = None,
) -> list[str]:
    """Task ids cited as CURRENT work in ``current_focus.md`` with no canonical ledger row."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    focus = root / _CURRENT_FOCUS_RELPATH
    if not focus.is_file():
        return []
    try:
        text = focus.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    registered = _registered_numeric_task_ids(root)
    refs = sorted({int(m) for m in _TASK_REF_RE.findall(text)})
    return [
        f"#{n}: cited in {_CURRENT_FOCUS_RELPATH} as current work but has NO row in "
        f".omx/state/canonical_task_status.jsonl — SPECIFICATION WITHOUT REGISTRATION "
        f"(ddm_rg5 #825). Prose is not existence: a task number that reads as live to every "
        f"downstream reader while nothing registered it is how a blocking relation got asserted "
        f"between two ids that were not in the ledger at all (R1-C, 2026-07-31). Fix: register "
        f"it via `tac.canonical_task_status.register_task` (or `tools/canonical_task_status.py`) "
        f"before citing it, or cite it as a catalog/PR number in a form this scan does not read "
        f"as a task."
        for n in refs
        if n >= _TASK_REF_REGISTRATION_BASELINE and n not in registered
    ]


def _resolve_source_design_memo(root: Path, memo: str | Path) -> Path | None:
    """Resolve a ``source_design_memo`` citation to an existing file, or ``None``.

    Three citation forms are REAL conventions in this ledger's history and all
    three must resolve (pf2x r86 adjudication, 2026-08-27).  This is scanner
    PRECISION, not weakening: a citation matching none of the forms is still a
    violation, and each form demands an actual file on disk.

    1. Repo-relative path — the common case.
    2. ``~``-prefixed home path — 2026-05-30 rows cite the durable Claude
       memory dir by home-relative path.  ``pathlib`` treats ``~/x`` as a
       RELATIVE path, so the pre-r86 ``root / memo`` join produced
       ``<repo>/~/x`` and misread EXISTING memos as missing.
    3. Bare filename — the memory-dir convention (CLAUDE.md and 2026-05-30
       rows cite ``feedback_*.md`` memos by name alone).  Resolved against the
       Claude project memory dir DERIVED from the repo root (the projects-dir
       naming convention: ``/`` → ``-``), never a hardcoded absolute path —
       this file is on a public surface.
    """
    raw = str(memo)
    if raw.startswith("~"):
        cand = Path(raw).expanduser()
        return cand if cand.is_file() else None
    rel = Path(raw)
    cand = root / rel
    if cand.is_file():
        return cand
    if rel.parent == Path("."):
        memory_dir = (
            Path.home()
            / ".claude"
            / "projects"
            / str(Path(root).resolve()).replace("/", "-")
            / "memory"
        )
        cand = memory_dir / rel
        if cand.is_file():
            return cand
    return None


def canonical_task_status_violations(repo_root: str | Path | None = None) -> list[str]:
    """Return strict ledger violations not already covered by schema loading.

    The memo-custody check reads the LATEST row per task, not every historical
    row (pf2x r86, 2026-08-27).  The ledger is registered append-only with
    ``append_fields: []`` — history is never mutated — so a checker that pins
    EVERY row makes correction structurally impossible: a task whose
    registration mis-cited its memo could never become clean.  The task's
    canonical claim is its latest row (the same latest-row-wins semantics the
    loader applies to status); a correction lands as an append-only note row
    carrying the corrected citation, and the citation must itself resolve
    (``append_note`` refuses unresolvable corrections, so no violation can be
    laundered away).  A task whose LATEST row cites a missing memo still fires.
    """

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    violations: list[str] = []
    latest: dict[str, CanonicalTaskStatusRow] = {}
    for row in load_canonical_task_status_strict(root):
        latest[row.task_id] = row
    for row in latest.values():
        if _resolve_source_design_memo(root, row.source_design_memo) is None:
            violations.append(
                f"{row.task_id}: source_design_memo missing: {row.source_design_memo}"
            )
    violations.extend(unregistered_task_refs_in_current_focus(root))
    return violations


def check_canonical_task_status_no_dangling_transitions(
    *,
    repo_root: str | Path | None = None,
    strict: bool = True,
    verbose: bool = False,
) -> list[str]:
    """Validate canonical task-status schema, transitions, and memo pointers."""

    violations = canonical_task_status_violations(repo_root)
    if violations and strict:
        raise AssertionError(
            "canonical_task_status violations:\n" + "\n".join(f"- {v}" for v in violations)
        )
    if verbose:
        if violations:
            print(
                "  [canonical-task-status] WARN: "
                f"{len(violations)} violation(s)"
            )
        else:
            print("  [canonical-task-status] OK")
    return violations

