# SPDX-License-Identifier: MIT
"""Source-scope helpers for literature-anchored planning rows.

Literature anchors are hypothesis provenance. They are not score authority
unless a row also states what the source supports, the source claim scope, what
Pact must prove empirically, and decode-complexity evidence.
"""
from __future__ import annotations

from typing import Any

LITERATURE_SOURCE_SCOPE_REQUIRED_FIELDS: tuple[str, ...] = (
    "source_supports",
    "paper_claim_scope",
    "pact_must_prove",
    "decode_complexity_evidence",
)

LITERATURE_SOURCE_SCOPE_EMPTY_MARKERS = frozenset(
    (
        "",
        "none",
        "null",
        "n/a",
        "na",
        "tbd",
        "todo",
        "unknown",
        "<source>",
        "<evidence>",
        "<rationale>",
        "<reason>",
    )
)


def literature_source_scope_row_get(row: object, field: str) -> object:
    """Return ``field`` from either a dict row or an object/dataclass row."""

    if isinstance(row, dict):
        return row.get(field)
    return getattr(row, field, None)


def non_placeholder_literature_source_text(value: object) -> bool:
    """Return true for non-empty, non-placeholder source-scope text."""

    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.lower() not in LITERATURE_SOURCE_SCOPE_EMPTY_MARKERS


def missing_literature_source_scope_fields(row: object) -> list[str]:
    """Return required source-scope fields missing from a literature row.

    Rows without a non-placeholder ``literature_anchor`` are not literature
    rows and therefore do not require the scope contract.
    """

    anchor = literature_source_scope_row_get(row, "literature_anchor")
    if not non_placeholder_literature_source_text(anchor):
        return []
    return [
        field
        for field in LITERATURE_SOURCE_SCOPE_REQUIRED_FIELDS
        if not non_placeholder_literature_source_text(
            literature_source_scope_row_get(row, field)
        )
    ]


def literature_source_scope_blockers(
    row: object,
    *,
    prefix: str = "literature_anchor_source_scope_missing",
) -> list[str]:
    """Return fail-closed blocker tokens for missing source-scope fields."""

    return [
        f"{prefix}:{field}"
        for field in missing_literature_source_scope_fields(row)
    ]


def literature_source_scope_violation_message(
    *,
    surface: str,
    row: object,
    row_id_field: str = "substrate_id",
) -> str | None:
    """Return a human-readable source-scope violation, if any."""

    missing = missing_literature_source_scope_fields(row)
    if not missing:
        return None
    anchor = literature_source_scope_row_get(row, "literature_anchor")
    row_id = literature_source_scope_row_get(row, row_id_field)
    row_id_text = str(row_id).strip() if row_id is not None else "?"
    return (
        f"{surface} row {row_id_text!r} carries literature_anchor={str(anchor)!r} "
        f"but lacks non-placeholder source-scope field(s): {', '.join(missing)}. "
        "Every Cathedral/autopilot literature anchor must state what the source "
        "supports, the paper-claim scope, what Pact must prove empirically, and "
        "decode-complexity evidence before the row can influence planning."
    )


def source_scope_values(row: dict[str, Any]) -> dict[str, str]:
    """Extract literature anchor and source-scope fields as strings."""

    def _text(field: str) -> str:
        value = row.get(field, "")
        return "" if value is None else str(value)

    return {
        "literature_anchor": _text("literature_anchor"),
        **{
            field: _text(field)
            for field in LITERATURE_SOURCE_SCOPE_REQUIRED_FIELDS
        },
    }
