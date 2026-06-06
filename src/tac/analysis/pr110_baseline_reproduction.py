# SPDX-License-Identifier: MIT
"""Fail-closed PR110 global K=16 baseline reproduction proof.

The PR110 commutator ledger can queue measured composite actions from
ActionEffect rows, but menu/selector escalation is only meaningful after the
current global K=16 baseline has been reproduced byte-for-byte and score-wise
inside an explicit authority surface.  This module is deliberately small: it
validates a typed JSON proof row and emits exact blocker names.  It does not
run a selector, score an archive, or mint any score authority.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PR110_K16_BASELINE_REPRODUCTION_SCHEMA = "tac.pr110_global_k16_baseline_reproduction.v1"
PR110_K16_BASELINE_REPRODUCTION_VALIDATION_SCHEMA = (
    "tac.pr110_global_k16_baseline_reproduction_validation.v1"
)

MENU_ILP_BASELINE_BLOCKER = "menu_ilp_blocked_until_pr110_k16_baseline_reproduces"

BLOCKER_MISSING = "pr110_k16_baseline_reproduction_missing"
BLOCKER_MALFORMED = "pr110_k16_baseline_reproduction_malformed"
BLOCKER_WRONG_SCHEMA = "pr110_k16_baseline_reproduction_wrong_schema"
BLOCKER_NOT_PASSED = "pr110_k16_baseline_reproduction_not_passed"
BLOCKER_GLOBAL_K = "pr110_k16_global_k_not_16"
BLOCKER_AUTHORITY = "pr110_k16_authority_missing"
BLOCKER_SELECTOR_REF = "pr110_k16_selector_ref_missing"
BLOCKER_BYTE_TOLERANCE = "pr110_k16_byte_tolerance_exceeded"
BLOCKER_SCORE_TOLERANCE = "pr110_k16_score_tolerance_exceeded"
BLOCKER_BYTE_ERROR_MISMATCH = "pr110_k16_byte_error_mismatch"
BLOCKER_SCORE_ERROR_MISMATCH = "pr110_k16_score_error_mismatch"

_REQUIRED_NUMERIC_FIELDS = (
    "expected_archive_bytes",
    "actual_archive_bytes",
    "byte_error_abs",
    "byte_tolerance",
    "expected_score",
    "actual_score",
    "score_error_abs",
    "score_tolerance",
)


def validate_pr110_k16_baseline_reproduction(
    payload: Mapping[str, Any] | None,
    *,
    source: str | None = None,
) -> dict[str, Any]:
    """Validate one typed PR110 global K=16 baseline reproduction row.

    Returns a JSON-serializable validation row.  ``passed`` is derived from the
    blocker list, not trusted from the payload.  The payload's own ``passed``
    flag is still required to be ``true`` so stale negative proofs cannot be
    promoted by omission.
    """

    blockers: list[str] = []
    observed: dict[str, Any] = {
        "schema": PR110_K16_BASELINE_REPRODUCTION_VALIDATION_SCHEMA,
        "source": source,
        "baseline_schema": None,
        "global_k": None,
        "authority": None,
        "selector_id": None,
        "selector_family": None,
        "expected_archive_bytes": None,
        "actual_archive_bytes": None,
        "byte_error_abs": None,
        "byte_tolerance": None,
        "expected_score": None,
        "actual_score": None,
        "score_error_abs": None,
        "score_tolerance": None,
    }
    if payload is None:
        blockers.append(BLOCKER_MISSING)
        observed["passed"] = False
        observed["blockers"] = blockers
        return observed

    if not isinstance(payload, Mapping):
        blockers.append(BLOCKER_MALFORMED)
        observed["passed"] = False
        observed["blockers"] = blockers
        return observed

    observed["baseline_schema"] = payload.get("schema")
    observed["global_k"] = payload.get("global_k")
    observed["authority"] = payload.get("authority")
    observed["selector_id"] = payload.get("selector_id")
    observed["selector_family"] = payload.get("selector_family")
    for field in _REQUIRED_NUMERIC_FIELDS:
        observed[field] = payload.get(field)

    if payload.get("schema") != PR110_K16_BASELINE_REPRODUCTION_SCHEMA:
        blockers.append(BLOCKER_WRONG_SCHEMA)
    if payload.get("passed") is not True:
        blockers.append(BLOCKER_NOT_PASSED)
    if _coerce_int(payload.get("global_k")) != 16:
        blockers.append(BLOCKER_GLOBAL_K)
    if not _nonempty_string(payload.get("authority")):
        blockers.append(BLOCKER_AUTHORITY)
    if not (_nonempty_string(payload.get("selector_id")) or _nonempty_string(payload.get("selector_family"))):
        blockers.append(BLOCKER_SELECTOR_REF)

    expected_bytes = _coerce_int(payload.get("expected_archive_bytes"))
    actual_bytes = _coerce_int(payload.get("actual_archive_bytes"))
    byte_error = _coerce_float(payload.get("byte_error_abs"))
    byte_tolerance = _coerce_float(payload.get("byte_tolerance"))
    expected_score = _coerce_float(payload.get("expected_score"))
    actual_score = _coerce_float(payload.get("actual_score"))
    score_error = _coerce_float(payload.get("score_error_abs"))
    score_tolerance = _coerce_float(payload.get("score_tolerance"))

    if None in (expected_bytes, actual_bytes, byte_error, byte_tolerance):
        blockers.append(BLOCKER_MALFORMED)
    else:
        computed_byte_error = abs(int(actual_bytes) - int(expected_bytes))
        if not math.isclose(float(byte_error), float(computed_byte_error), rel_tol=0.0, abs_tol=0.0):
            blockers.append(BLOCKER_BYTE_ERROR_MISMATCH)
        if float(byte_tolerance) < 0.0 or float(byte_error) > float(byte_tolerance):
            blockers.append(BLOCKER_BYTE_TOLERANCE)

    if None in (expected_score, actual_score, score_error, score_tolerance):
        blockers.append(BLOCKER_MALFORMED)
    else:
        computed_score_error = abs(float(actual_score) - float(expected_score))
        if not math.isclose(float(score_error), computed_score_error, rel_tol=0.0, abs_tol=1e-15):
            blockers.append(BLOCKER_SCORE_ERROR_MISMATCH)
        if float(score_tolerance) < 0.0 or float(score_error) > float(score_tolerance):
            blockers.append(BLOCKER_SCORE_TOLERANCE)

    observed["blockers"] = _dedupe(blockers)
    observed["passed"] = not observed["blockers"]
    return observed


def load_and_validate_pr110_k16_baseline_reproduction(path: Path | None) -> dict[str, Any]:
    """Load and validate a PR110 K=16 baseline proof from JSON.

    ``None`` and absent paths return a fail-closed validation row instead of
    raising, so CLI consumers can surface blocker names in their own artifacts.
    """

    if path is None:
        return validate_pr110_k16_baseline_reproduction(None, source=None)
    source = path.as_posix()
    if not path.exists():
        validation = validate_pr110_k16_baseline_reproduction(None, source=source)
        return validation
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        validation = validate_pr110_k16_baseline_reproduction({}, source=source)
        validation["blockers"] = _dedupe([BLOCKER_MALFORMED, *validation["blockers"]])
        validation["passed"] = False
        return validation
    return validate_pr110_k16_baseline_reproduction(payload, source=source)


def baseline_blockers_for_menu_ilp(validation: Mapping[str, Any]) -> list[str]:
    """Return the menu/macro escalation blockers implied by a validation row."""

    if validation.get("passed") is True:
        return []
    return _dedupe([MENU_ILP_BASELINE_BLOCKER, *list(validation.get("blockers", []))])


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


__all__ = [
    "BLOCKER_AUTHORITY",
    "BLOCKER_BYTE_ERROR_MISMATCH",
    "BLOCKER_BYTE_TOLERANCE",
    "BLOCKER_GLOBAL_K",
    "BLOCKER_MALFORMED",
    "BLOCKER_MISSING",
    "BLOCKER_NOT_PASSED",
    "BLOCKER_SCORE_ERROR_MISMATCH",
    "BLOCKER_SCORE_TOLERANCE",
    "BLOCKER_SELECTOR_REF",
    "BLOCKER_WRONG_SCHEMA",
    "MENU_ILP_BASELINE_BLOCKER",
    "PR110_K16_BASELINE_REPRODUCTION_SCHEMA",
    "PR110_K16_BASELINE_REPRODUCTION_VALIDATION_SCHEMA",
    "baseline_blockers_for_menu_ilp",
    "load_and_validate_pr110_k16_baseline_reproduction",
    "validate_pr110_k16_baseline_reproduction",
]
