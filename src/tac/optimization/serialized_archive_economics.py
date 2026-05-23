# SPDX-License-Identifier: MIT
"""Canonical source/candidate archive byte economics.

Planner byte values are useful acquisition signal, but they are not evidence
that a serialized submission archive actually got smaller. This module keeps
that boundary explicit for queue rows, materializer manifests, and exact-ready
guards that need to reason about rate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tac.optimization.proxy_candidate_contract import ordered_unique

SERIALIZED_ARCHIVE_DELTA_SCHEMA = "serialized_archive_delta_contract.v1"
MISSING_ARCHIVE_BYTES_BLOCKER = "serialized_archive_delta_bytes_missing"
MODELED_SAVINGS_WITHOUT_REALIZED_BLOCKER = (
    "modeled_savings_without_realized_serialized_savings"
)
SERIALIZED_SAVINGS_NOT_POSITIVE_BLOCKER = "serialized_archive_savings_not_positive"
CANDIDATE_ARCHIVE_LARGER_BLOCKER = "candidate_archive_larger_than_source_archive"

FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}


def archive_record_bytes(record: Any) -> int | None:
    """Return archive bytes from common archive-record shapes."""

    parsed_record = _positive_integral(record)
    if parsed_record is not None:
        return parsed_record
    if not isinstance(record, Mapping):
        return None
    for key in ("bytes", "archive_bytes", "size_bytes", "archive_size_bytes"):
        parsed_value = _positive_integral(record.get(key))
        if parsed_value is not None:
            return parsed_value
    return None


def build_serialized_archive_delta_contract(
    *,
    source_archive: Any = None,
    candidate_archive: Any = None,
    source_archive_bytes: Any = None,
    candidate_archive_bytes: Any = None,
    modeled_saved_bytes: Any = None,
    modeled_cost_bytes: Any = None,
    require_realized_saving: bool = False,
    rate_only_control: bool = False,
) -> dict[str, Any]:
    """Return a fail-closed contract for realized archive-byte movement."""

    source_bytes = archive_record_bytes(source_archive_bytes)
    if source_bytes is None:
        source_bytes = archive_record_bytes(source_archive)
    candidate_bytes = archive_record_bytes(candidate_archive_bytes)
    if candidate_bytes is None:
        candidate_bytes = archive_record_bytes(candidate_archive)
    modeled_saved = _optional_int(modeled_saved_bytes)
    modeled_cost = _optional_int(modeled_cost_bytes)

    blockers: list[str] = []
    archive_delta: int | None = None
    realized_saved: int | None = None
    if source_bytes is None or candidate_bytes is None:
        status = "missing_archive_bytes"
        blockers.append(MISSING_ARCHIVE_BYTES_BLOCKER)
    else:
        archive_delta = candidate_bytes - source_bytes
        realized_saved = source_bytes - candidate_bytes
        if rate_only_control:
            status = "rate_only_control"
        elif realized_saved > 0:
            status = "realized_saving"
        elif realized_saved == 0:
            status = "zero_delta"
            if require_realized_saving:
                blockers.append(SERIALIZED_SAVINGS_NOT_POSITIVE_BLOCKER)
        else:
            status = "realized_cost"
            if require_realized_saving:
                blockers.append(CANDIDATE_ARCHIVE_LARGER_BLOCKER)

    if (
        modeled_saved is not None
        and modeled_saved > 0
        and not rate_only_control
        and (realized_saved is None or realized_saved <= 0)
    ):
        blockers.append(MODELED_SAVINGS_WITHOUT_REALIZED_BLOCKER)

    if require_realized_saving and realized_saved is None:
        blockers.append(SERIALIZED_SAVINGS_NOT_POSITIVE_BLOCKER)

    return {
        "schema": SERIALIZED_ARCHIVE_DELTA_SCHEMA,
        "source_archive_bytes": source_bytes,
        "candidate_archive_bytes": candidate_bytes,
        "archive_delta_bytes": archive_delta,
        "realized_saved_bytes": realized_saved,
        "modeled_saved_bytes": modeled_saved,
        "modeled_cost_bytes": modeled_cost,
        "require_realized_saving": bool(require_realized_saving),
        "rate_only_control": bool(rate_only_control),
        "savings_realized": realized_saved is not None and realized_saved > 0,
        "status": status,
        "blockers": ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def serialized_archive_delta_blockers(contract: Mapping[str, Any]) -> list[str]:
    """Return ordered blockers from a serialized archive delta contract."""

    return ordered_unique(str(item) for item in contract.get("blockers") or [] if str(item))


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_integral(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


__all__ = [
    "CANDIDATE_ARCHIVE_LARGER_BLOCKER",
    "MISSING_ARCHIVE_BYTES_BLOCKER",
    "MODELED_SAVINGS_WITHOUT_REALIZED_BLOCKER",
    "SERIALIZED_ARCHIVE_DELTA_SCHEMA",
    "SERIALIZED_SAVINGS_NOT_POSITIVE_BLOCKER",
    "archive_record_bytes",
    "build_serialized_archive_delta_contract",
    "serialized_archive_delta_blockers",
]
