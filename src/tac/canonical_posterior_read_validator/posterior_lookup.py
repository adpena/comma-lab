# SPDX-License-Identifier: MIT
"""Canonical posterior lookup for the read-surface validator (Catalog #382).

This module answers ONE question for a given claim token:

    "What is the LATEST canonical-posterior event for this token, and does
     that event mark the token FALSIFIED / KILLED / PHANTOM / INVALIDATED?"

It reads THREE canonical posterior surfaces (latest-event-wins per surface):

  1. canonical_equations_registry  (.omx/state/canonical_equations_registry.jsonl)
  2. canonical_anti_patterns       (.omx/state/canonical_anti_patterns_registry.jsonl)
  3. probe_outcomes_ledger         (.omx/state/probe_outcomes.jsonl)

The validator is READ-ONLY and fail-open on a per-surface basis: a missing or
malformed surface yields NO verdict (the token is treated as not-present in that
surface) rather than raising — so a corrupt ledger cannot block memo authoring.

BLOCKING vs SUPERSEDED (the correct-and-complete split, 2026-05-31 per operator
directive "resolve correctly and completely, don't paper over"):

  * `_BLOCKING_STATUSES` = the cited VALUE/claim is WRONG (falsified / killed /
    phantom / invalidated / refuted). Citing such a token as current truth IS
    the phantom-score bug class Catalog #382 targets. -> is_blocking=True.

  * `_SUPERSEDED_STATUSES` = the token was RENAMED/REFINED (superseded / renamed
    / replaced) but its value-lesson is still valid. Per Catalog #110/#113
    HISTORICAL_PROVENANCE, historical citations of a superseded id remain valid;
    only NEW memos should prefer the successor. A supersession is therefore a
    NON-BLOCKING advisory that surfaces the `superseded_by` successor (no signal
    loss) rather than a phantom-value block. -> is_blocking=False.

This split aligns the implementation to the canonical CLAUDE.md Catalog #382
spec ("FALSIFIED / KILLED / PHANTOM / INVALIDATED") which never listed
`superseded`; the prior implementation conflated a benign rename with a phantom
value, which mis-flagged the legitimate historical citation of
`stand_down_verdict_based_on_stale_canonical_state_currency_v1` (registered
2026-05-29, superseded 2026-05-31 by the Catalog #376/#378 root-cause
anti-pattern, with the supersession's own rationale affirming Catalog #110/#113
historical-citation validity).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The cited VALUE is wrong -> citing it as current truth is a phantom-score bug.
_BLOCKING_STATUSES = frozenset(
    {
        "falsified",
        "killed",
        "phantom",
        "invalidated",
        "refuted",
    }
)

# The token was renamed/refined; the value-lesson is still valid. Historical
# citations remain valid per Catalog #110/#113 -> NON-blocking advisory.
_SUPERSEDED_STATUSES = frozenset(
    {
        "superseded",
        "renamed",
        "replaced",
    }
)

# Backward-compat alias for any external importer that historically referenced
# the blocking set. It now correctly EXCLUDES the superseded/rename statuses.
_FALSIFIED_STATUSES = _BLOCKING_STATUSES

# Fields a row may carry to point at the canonical successor of a superseded id.
_SUCCESSOR_KEYS = ("superseded_by", "superseded_by_id", "replaced_by", "renamed_to")


@dataclass(frozen=True)
class PosteriorVerdict:
    """The latest canonical-posterior verdict for a claim token."""

    token: str
    found: bool
    latest_status: str | None
    is_blocking: bool
    surface: str | None
    rationale: str


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    return rows


def _latest_row_for_token_in_rows(
    token: str,
    rows: list[dict[str, Any]],
    *,
    id_keys: tuple[str, ...],
) -> dict[str, Any] | None:
    """Return the latest (last-in-file) row whose id matches the token, or None."""
    latest: dict[str, Any] | None = None
    for row in rows:
        row_id = None
        for k in id_keys:
            if k in row and isinstance(row[k], str):
                row_id = row[k]
                break
        if row_id == token:
            latest = row
    return latest


def _status_from_row(
    row: dict[str, Any],
    status_keys: tuple[str, ...],
) -> str | None:
    """Return the first-present string status from the row, or None."""
    for sk in status_keys:
        if sk in row and isinstance(row[sk], str):
            return row[sk]
    return None


def _latest_status_for_token_in_rows(
    token: str,
    rows: list[dict[str, Any]],
    *,
    id_keys: tuple[str, ...],
    status_keys: tuple[str, ...],
) -> str | None:
    """Return the latest (last-in-file) status for the token, or None.

    Retained for backward compatibility; delegates to the row-aware helpers.
    """
    row = _latest_row_for_token_in_rows(token, rows, id_keys=id_keys)
    if row is None:
        return None
    return _status_from_row(row, status_keys)


def _successor_from_row(row: dict[str, Any]) -> str | None:
    for k in _SUCCESSOR_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    return None


# Per-surface lookup config: (filename, id_keys, status_keys).
_SURFACES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "canonical_equations_registry.jsonl",
        ("equation_id", "id", "canonical_equation_id"),
        ("event_type", "status", "verdict"),
    ),
    (
        "canonical_anti_patterns_registry.jsonl",
        ("anti_pattern_id", "id", "pattern_id"),
        ("event_type", "status", "verdict"),
    ),
    (
        "probe_outcomes.jsonl",
        ("probe_id", "id", "substrate_id"),
        ("verdict", "status", "event_type"),
    ),
)

_SURFACE_LABEL = {
    "canonical_equations_registry.jsonl": "canonical_equations_registry",
    "canonical_anti_patterns_registry.jsonl": "canonical_anti_patterns_registry",
    "probe_outcomes.jsonl": "probe_outcomes_ledger",
}


def lookup_latest_verdict(token: str, repo_root: Path | str = ".") -> PosteriorVerdict:
    """Return the latest canonical-posterior verdict for a claim token.

    Latest-event-wins WITHIN each surface; the FIRST surface that yields a
    BLOCKING status wins ACROSS surfaces (equations -> anti-patterns -> probes).
    A SUPERSEDED status is recorded as a non-blocking advisory only if NO surface
    yields a blocking status (a phantom-value verdict anywhere dominates a benign
    rename elsewhere).
    """
    root = Path(repo_root)
    state = root / ".omx" / "state"

    superseded_advisory: PosteriorVerdict | None = None

    for fn, id_keys, status_keys in _SURFACES:
        rows = _iter_jsonl(state / fn)
        row = _latest_row_for_token_in_rows(token, rows, id_keys=id_keys)
        if row is None:
            continue
        status = _status_from_row(row, status_keys)
        if status is None:
            continue
        norm = status.lower()
        surface = _SURFACE_LABEL[fn]

        if norm in _BLOCKING_STATUSES:
            # First blocking surface wins immediately.
            return PosteriorVerdict(
                token=token,
                found=True,
                latest_status=status,
                is_blocking=True,
                surface=surface,
                rationale=f"latest {surface} event '{status}' is blocking",
            )

        if norm in _SUPERSEDED_STATUSES and superseded_advisory is None:
            successor = _successor_from_row(row)
            if successor:
                rationale = (
                    f"token superseded by '{successor}'; historical citations "
                    "remain valid per Catalog #110/#113 HISTORICAL_PROVENANCE — "
                    "new memos should cite the successor"
                )
            else:
                rationale = (
                    f"token '{token}' superseded with no declared successor; "
                    "historical citations remain valid per Catalog #110/#113 "
                    "HISTORICAL_PROVENANCE (non-blocking rename-class event)"
                )
            superseded_advisory = PosteriorVerdict(
                token=token,
                found=True,
                latest_status=status,
                is_blocking=False,
                surface=surface,
                rationale=rationale,
            )

    if superseded_advisory is not None:
        return superseded_advisory

    return PosteriorVerdict(
        token=token,
        found=False,
        latest_status=None,
        is_blocking=False,
        surface=None,
        rationale="no blocking canonical-posterior event for token",
    )
