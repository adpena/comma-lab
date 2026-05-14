"""Fail-closed score/dispatch authority helpers.

Planning, smoke, proxy, and diagnostic artifacts are allowed to guide work,
but they must not silently become score claims, promotion evidence, or
rank/kill authority.  This module centralizes that contract so operator
surfaces do not drift into slightly different defaults.
"""

from __future__ import annotations

from typing import Any

FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "rank_or_kill_eligible",
)


def apply_false_authority_contract(
    row: dict[str, Any],
    *,
    preserve_dispatch_ready: bool = True,
    reason: str = "planning_only_no_score_authority",
) -> dict[str, Any]:
    """Mark a row as non-promotable and non-ranking authority.

    ``ready_for_exact_eval_dispatch`` is intentionally configurable because an
    operator planning row can still describe a dispatchable job while remaining
    non-score evidence until the exact-eval result lands.
    """

    for field in FALSE_AUTHORITY_FIELDS:
        row[field] = False
    if preserve_dispatch_ready:
        row["ready_for_exact_eval_dispatch"] = bool(
            row.get("ready_for_exact_eval_dispatch", False)
        )
    else:
        row["ready_for_exact_eval_dispatch"] = False
    row.setdefault("authority_contract", reason)
    return row


def normalize_score_authority_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize authority fields without granting missing rank/kill rights."""

    row["score_claim"] = bool(row.get("score_claim", False))
    row["score_claim_valid"] = bool(row.get("score_claim_valid", False))
    row["promotion_eligible"] = bool(row.get("promotion_eligible", False))
    row["rank_or_kill_eligible"] = bool(row.get("rank_or_kill_eligible", False))
    row["ready_for_exact_eval_dispatch"] = bool(
        row.get("ready_for_exact_eval_dispatch", False)
    )
    return row
