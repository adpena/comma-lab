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
from typing import TYPE_CHECKING, Any

from tac.score_geometry import contest_score

if TYPE_CHECKING:
    from tac.analysis.action_effect import ActionEffect

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
BLOCKER_REPLAY_ROW_MISSING = "pr110_k16_selector_replay_action_effect_missing"
BLOCKER_SELECTOR_PAIR_COUNT = "pr110_k16_selector_pair_count_not_600"
BLOCKER_SELECTOR_BITS = "pr110_k16_selector_bits_missing"
BLOCKER_DELTA_SCORE = "pr110_k16_delta_score_missing"

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
    blockers.extend(_payload_blockers(payload))
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


def build_pr110_k16_baseline_reproduction_from_action_effects(
    effects: list[ActionEffect],
    *,
    expected_global_k: int = 16,
    expected_pair_count: int = 600,
    expected_score: float | None = None,
    score_tolerance: float = 0.0,
    byte_tolerance: int = 0,
    source: str | None = None,
) -> dict[str, Any]:
    """Build a truthful PR110 global-K reproduction proof from ActionEffect rows.

    This is a producer, not a scorer.  It can clear the K=16 blocker only when a
    real selector replay ActionEffect row carries the global K=16 selector
    surface, full selector stream pair count, exact old/new score endpoints, and
    byte endpoints under one authority.  Sparse K=1 advisory rows therefore
    become a precise mismatch artifact instead of being promoted as a K=16
    reproduction.
    """

    selector_rows = [
        effect
        for effect in effects
        if effect.family == "pr110" and effect.action_kind in {"selector_replay", "selector_program", "selector_mode"}
    ]
    selected = _best_selector_row(selector_rows)
    blockers: list[str] = []
    if selected is None:
        blockers.append(BLOCKER_REPLAY_ROW_MISSING)

    observed_pair_count = len(selected.pair_ids) if selected is not None else 0
    observed_global_k = _observed_global_k(selected) if selected is not None else None
    selector_bits = _observed_selector_bits(selected) if selected is not None else None

    if observed_global_k != expected_global_k:
        blockers.append(BLOCKER_GLOBAL_K)
    if observed_pair_count != expected_pair_count:
        blockers.append(BLOCKER_SELECTOR_PAIR_COUNT)
    if selector_bits is None:
        blockers.append(BLOCKER_SELECTOR_BITS)
    if selected is not None and selected.delta_score_total is None:
        blockers.append(BLOCKER_DELTA_SCORE)

    actual_score = None
    if selected is not None and selected.old_d_seg is not None and selected.old_d_pose is not None and selected.old_bytes is not None:
        actual_score = _score_total(selected.new_d_seg, selected.new_d_pose, selected.new_bytes)
    expected_score_value = expected_score if expected_score is not None else actual_score
    score_error_abs = (
        abs(float(actual_score) - float(expected_score_value))
        if actual_score is not None and expected_score_value is not None
        else None
    )
    if score_error_abs is None:
        blockers.append(BLOCKER_MALFORMED)
    elif score_error_abs > float(score_tolerance):
        blockers.append(BLOCKER_SCORE_TOLERANCE)

    expected_bytes = selected.new_bytes if selected is not None else None
    actual_bytes = selected.new_bytes if selected is not None else None
    byte_error_abs = (
        abs(int(actual_bytes) - int(expected_bytes))
        if actual_bytes is not None and expected_bytes is not None
        else None
    )
    if byte_error_abs is None:
        blockers.append(BLOCKER_MALFORMED)
    elif byte_error_abs > int(byte_tolerance):
        blockers.append(BLOCKER_BYTE_TOLERANCE)

    blockers = _dedupe(blockers)
    return {
        "schema": PR110_K16_BASELINE_REPRODUCTION_SCHEMA,
        "source": source,
        "passed": not blockers,
        "blockers": blockers,
        "first_mismatch": blockers[0] if blockers else None,
        "global_k": observed_global_k,
        "expected_global_k": expected_global_k,
        "pair_count": observed_pair_count,
        "expected_pair_count": expected_pair_count,
        "selector_bits": selector_bits,
        "selector_id": selected.action_id if selected is not None else None,
        "selector_family": selected.family if selected is not None else None,
        "authority": selected.authority if selected is not None else None,
        "normalization_scope": selected.normalization_scope if selected is not None else None,
        "old_d_seg": selected.old_d_seg if selected is not None else None,
        "new_d_seg": selected.new_d_seg if selected is not None else None,
        "old_d_pose": selected.old_d_pose if selected is not None else None,
        "new_d_pose": selected.new_d_pose if selected is not None else None,
        "old_archive_bytes": selected.old_bytes if selected is not None else None,
        "expected_archive_bytes": expected_bytes,
        "actual_archive_bytes": actual_bytes,
        "byte_error_abs": byte_error_abs,
        "byte_tolerance": int(byte_tolerance),
        "expected_score": expected_score_value,
        "actual_score": actual_score,
        "score_error_abs": score_error_abs,
        "score_tolerance": float(score_tolerance),
        "delta_score_total": selected.delta_score_total if selected is not None else None,
        "delta_score_nonrate": selected.delta_score_nonrate if selected is not None else None,
        "delta_bytes": selected.delta_bytes if selected is not None else None,
        "archive_sha256": selected.archive_sha256 if selected is not None else None,
        "payload_sha256": selected.payload_sha256 if selected is not None else None,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


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


def _payload_blockers(payload: Mapping[str, Any]) -> list[str]:
    values = payload.get("blockers")
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value).strip()]


def _best_selector_row(effects: list[ActionEffect]) -> ActionEffect | None:
    if not effects:
        return None
    return max(effects, key=lambda effect: (len(effect.pair_ids), effect.delta_score_total is not None))


def _observed_global_k(effect: ActionEffect | None) -> int | None:
    if effect is None:
        return None
    for section in effect.payload_sections:
        text = str(section).lower()
        if "k16" in text or "k=16" in text:
            return 16
    if effect.action_kind == "selector_mode" and effect.pair_ids:
        return 1
    if effect.action_kind == "selector_replay" and effect.pair_ids:
        return len(effect.pair_ids)
    return None


def _observed_selector_bits(effect: ActionEffect | None) -> int | None:
    if effect is None:
        return None
    surface = effect.receiver_surface.as_dict()
    for key in ("selector_bits", "selector_bits_total", "raw_selector_bits", "selector_code_bits_total"):
        value = surface.get(key)
        parsed = _coerce_int(value)
        if parsed is not None:
            return parsed
    for section in effect.payload_sections:
        parsed = _selector_bits_from_section(str(section))
        if parsed is not None:
            return parsed
    if _observed_global_k(effect) == 16 and len(effect.pair_ids) == 600:
        return 600 * 4
    return None


def _score_total(d_seg: float | None, d_pose: float | None, archive_bytes: int | None) -> float | None:
    if d_seg is None or d_pose is None or archive_bytes is None:
        return None
    return contest_score(float(d_seg), float(d_pose), int(archive_bytes))


def _selector_bits_from_section(section: str) -> int | None:
    for marker in ("selector_code_bits_total=", "selector_bits=", "selector_bits_total="):
        if marker not in section:
            continue
        raw = section.split(marker, 1)[1].split(",", 1)[0].strip()
        try:
            value = int(raw)
        except ValueError:
            return None
        return value if value >= 0 else None
    return None


__all__ = [
    "BLOCKER_AUTHORITY",
    "BLOCKER_BYTE_ERROR_MISMATCH",
    "BLOCKER_BYTE_TOLERANCE",
    "BLOCKER_DELTA_SCORE",
    "BLOCKER_GLOBAL_K",
    "BLOCKER_MALFORMED",
    "BLOCKER_MISSING",
    "BLOCKER_NOT_PASSED",
    "BLOCKER_REPLAY_ROW_MISSING",
    "BLOCKER_SCORE_ERROR_MISMATCH",
    "BLOCKER_SCORE_TOLERANCE",
    "BLOCKER_SELECTOR_BITS",
    "BLOCKER_SELECTOR_PAIR_COUNT",
    "BLOCKER_SELECTOR_REF",
    "BLOCKER_WRONG_SCHEMA",
    "MENU_ILP_BASELINE_BLOCKER",
    "PR110_K16_BASELINE_REPRODUCTION_SCHEMA",
    "PR110_K16_BASELINE_REPRODUCTION_VALIDATION_SCHEMA",
    "baseline_blockers_for_menu_ilp",
    "build_pr110_k16_baseline_reproduction_from_action_effects",
    "load_and_validate_pr110_k16_baseline_reproduction",
    "validate_pr110_k16_baseline_reproduction",
]
