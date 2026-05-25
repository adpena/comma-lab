# SPDX-License-Identifier: MIT
"""Shared materializer feedback extraction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MATERIALIZER_DELTA_KEYS: tuple[str, ...] = (
    "section_recode",
    "selected_compression",
    "selected_merge",
    "selected_payload",
    "selected_elision",
    "factorization",
)


def optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def selected_materializer_delta(
    payload: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    for key in MATERIALIZER_DELTA_KEYS:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return key, value
    return "", {}


def materializer_archive_delta(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    selected_key, selected = selected_materializer_delta(payload)
    if not selected_key:
        return None
    source_archive = (
        payload.get("source_archive")
        if isinstance(payload.get("source_archive"), Mapping)
        else {}
    )
    candidate_archive = (
        payload.get("candidate_archive")
        if isinstance(payload.get("candidate_archive"), Mapping)
        else {}
    )
    source_bytes = optional_int(
        selected.get("source_archive_bytes") or source_archive.get("bytes")
    )
    candidate_bytes = optional_int(
        selected.get("candidate_archive_bytes") or candidate_archive.get("bytes")
    )
    saved_bytes = optional_int(selected.get("saved_bytes"))
    if saved_bytes is None and source_bytes is not None and candidate_bytes is not None:
        saved_bytes = source_bytes - candidate_bytes
    if saved_bytes is None and source_bytes is None and candidate_bytes is None:
        return None
    if saved_bytes is None:
        saved_bytes = 0
    if saved_bytes > 0:
        status = "realized_saving"
    elif saved_bytes < 0:
        status = "realized_cost"
    else:
        status = "no_realized_delta"
    return {
        "selected_materialization_key": selected_key or None,
        "realized_saved_bytes": saved_bytes,
        "source_archive_bytes": source_bytes,
        "candidate_archive_bytes": candidate_bytes,
        "savings_realized": saved_bytes > 0,
        "status": status,
    }


__all__ = [
    "MATERIALIZER_DELTA_KEYS",
    "materializer_archive_delta",
    "optional_int",
    "selected_materializer_delta",
]
