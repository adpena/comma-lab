"""Evidence-contract helpers for score-lowering candidate planners.

The functions in this module are pure schema checks. They do not run evals,
dispatch jobs, or validate files on disk. They exist so planner surfaces can
fail closed before a score-lowering-looking row becomes active ranking or
promotion evidence.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

EXACT_CUDA_POSITIVE_TOKENS = frozenset(
    (
        "contest-cuda",
        "contest_cuda",
        "contest_cuda_auth_eval",
        "contest_cuda_exact_eval",
        "contest_cuda_exact_eval_positive",
        "exact-cuda",
        "exact_cuda",
        "exact_cuda_auth_eval",
        "exact_cuda_auth_eval_complete",
        "exact_cuda_auth_eval_positive",
        "exact_cuda_eval",
        "exact_cuda_eval_positive",
        "exact_cuda_full",
        "exact_cuda_full_600",
    )
)
EXACT_CUDA_POSITIVE_NORMALIZED_TOKENS = frozenset(
    marker.replace("-", "_") for marker in EXACT_CUDA_POSITIVE_TOKENS
)

EXACT_CUDA_MARKER_FIELDS = (
    "evidence_grade",
    "evidence_marker",
    "evidence_semantics",
    "device_axis",
    "hardware",
)

EXACT_CUDA_BOOLEAN_FIELDS = (
    "exact_cuda_auth_eval",
    "contest_cuda_auth_eval",
    "auth_eval_exact_cuda",
    "cuda_auth_eval",
)

STRICT_JSON_BOOLEAN_FIELDS = (
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
    "family_falsified",
    "method_family_retired",
    "score_affecting_payload_changed",
    "charged_bits_changed",
    "cuda_eval_worth_testing",
    "byte_proxy_only",
    "proxy_row",
    *EXACT_CUDA_BOOLEAN_FIELDS,
)

NEGATED_EXACT_CUDA_WORDS = frozenset(
    (
        "missing",
        "need",
        "needs",
        "no",
        "not",
        "pending",
        "require",
        "required",
        "requires",
        "without",
    )
)

PROXY_OR_PLANNING_MARKERS = (
    "mps",
    "cpu-prep",
    "cpu_prep",
    "proxy",
    "prediction",
    "predicted",
    "forensic",
    "research-signal",
    "research_signal",
    "byte_proxy",
    "byte-proxy",
    "planning-only",
)

NON_PROMOTABLE_RESULT_MARKERS = (
    "a-negative",
    "exact-negative",
    "exact_negative",
    "measured_config_retired",
    "measured-config-retired",
    "falsified",
    "deferred",
    "retracted",
    "not_dispatchable",
    "not-deployable",
)


def is_sha256_hex(value: Any) -> bool:
    """Return true when ``value`` is a 64-character hexadecimal SHA-256."""

    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower())


def _ordered_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _text(row: Mapping[str, Any], keys: Iterable[str]) -> str:
    return " ".join(str(row.get(key) or "") for key in keys).strip().lower()


def _tokens(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9_+.-]+", str(value or "").lower())


def _normal_token(value: str) -> str:
    return value.replace("-", "_").strip("._")


def _token_references_exact_cuda(value: str) -> bool:
    token = _normal_token(value)
    return "exact_cuda" in token or "contest_cuda" in token


def _token_negates_exact_cuda(value: str) -> bool:
    token = _normal_token(value)
    if not _token_references_exact_cuda(token):
        return False
    parts = [part for part in re.split(r"[_\-.]+", token) if part]
    if any(part in NEGATED_EXACT_CUDA_WORDS for part in parts):
        return True
    return token.endswith(
        ("_missing", "_needed", "_required", "_not_run", "_pending")
    )


def _token_window_negates(tokens: list[str], index: int) -> bool:
    window = tokens[max(0, index - 3) : min(len(tokens), index + 3)]
    return any(_normal_token(token) in NEGATED_EXACT_CUDA_WORDS for token in window)


def _field_has_positive_exact_cuda_marker(value: Any) -> bool:
    tokens = _tokens(value)
    for index, token in enumerate(tokens):
        normal = _normal_token(token)
        if _token_negates_exact_cuda(token):
            continue
        if (
            token in EXACT_CUDA_POSITIVE_TOKENS
            or normal in EXACT_CUDA_POSITIVE_NORMALIZED_TOKENS
        ):
            if not _token_window_negates(tokens, index):
                return True
    return False


def _evidence_grade_is_exact_cuda(
    value: Any, *, include_negative_grade: bool = False
) -> bool:
    tokens = _tokens(value)
    if not tokens:
        return False
    if any(_token_negates_exact_cuda(token) for token in tokens):
        return False
    if include_negative_grade and any(
        _normal_token(token) == "a_negative" for token in tokens
    ):
        return True
    first = _normal_token(tokens[0])
    return first == "a++" or str(value or "").strip().lower() == "a"


def _schema_blockers(row: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for key in STRICT_JSON_BOOLEAN_FIELDS:
        if key not in row:
            continue
        value = row.get(key)
        if value is None or isinstance(value, bool):
            continue
        blockers.append(f"invalid_evidence_schema_boolean:{key}")
    if blockers:
        return ["invalid_evidence_schema_non_promotable", *blockers]
    return []


def has_positive_exact_cuda_evidence_marker(
    row: Mapping[str, Any],
    *,
    include_negative_grade: bool = False,
) -> bool:
    """Return true only for explicit positive exact-CUDA evidence markers.

    Negated requirement text such as ``requires_exact_cuda_auth_eval`` or
    ``missing exact_cuda_auth_eval`` names the evidence still needed; it must
    never satisfy the exact-CUDA marker requirement.
    """

    if any(row.get(key) is True for key in EXACT_CUDA_BOOLEAN_FIELDS):
        return True
    if _evidence_grade_is_exact_cuda(
        row.get("evidence_grade"),
        include_negative_grade=include_negative_grade,
    ):
        return True
    return any(
        _field_has_positive_exact_cuda_marker(row.get(key))
        for key in EXACT_CUDA_MARKER_FIELDS
    )


def _bool_is_true(row: Mapping[str, Any], key: str) -> bool:
    return row.get(key) is True


def _positive_int(row: Mapping[str, Any], keys: Iterable[str]) -> bool:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return True
    return False


def _finite_numeric_present(row: Mapping[str, Any], keys: Iterable[str]) -> bool:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float) and math.isfinite(float(value)):
            return True
    return False


def _dispatch_blockers(row: Mapping[str, Any]) -> list[str]:
    value = row.get("dispatch_blockers")
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def promotable_exact_cuda_evidence_blockers(row: Mapping[str, Any]) -> list[str]:
    """Return blockers that prevent a row from being active score evidence.

    A row must describe an exact CUDA score on a byte-closed archive/runtime
    pair before planners may use it as promotable empirical evidence. CPU,
    MPS, proxy, byte-only, and negative/retired rows can still seed validation
    queues, but they must not update active score-lowering rankings.
    """

    blockers: list[str] = _schema_blockers(row)
    text = _text(
        row,
        (
            "evidence_grade",
            "evidence_marker",
            "evidence_semantics",
            "source",
            "hardware",
            "device_axis",
            "contest_dispatch_verdict",
            "measured_config_status",
        ),
    )
    grade = str(row.get("evidence_grade") or "").strip().lower()

    if not _bool_is_true(row, "score_claim"):
        blockers.append("score_claim_true_required")
    if not _bool_is_true(row, "promotion_eligible"):
        blockers.append("promotion_eligible_true_required")
    if not _bool_is_true(row, "rank_or_kill_eligible"):
        blockers.append("rank_or_kill_eligible_true_required")
    if not _bool_is_true(row, "ready_for_exact_eval_dispatch"):
        blockers.append("ready_for_exact_eval_dispatch_true_required")

    if not has_positive_exact_cuda_evidence_marker(row):
        blockers.append("exact_cuda_evidence_marker_required")
    if any(marker in text for marker in PROXY_OR_PLANNING_MARKERS):
        blockers.append("proxy_or_planning_marker_not_promotable")
    if any(marker in text for marker in NON_PROMOTABLE_RESULT_MARKERS):
        blockers.append("negative_or_retired_result_not_promotable")

    if not _positive_int(row, ("empirical_archive_bytes", "archive_bytes")):
        blockers.append("positive_archive_bytes_required")
    if not is_sha256_hex(row.get("archive_sha256") or row.get("archive_sha")):
        blockers.append("archive_sha256_required")
    if not is_sha256_hex(
        row.get("runtime_tree_sha256")
        or row.get("inflate_runtime_tree_sha256")
        or row.get("runtime_tree_sha")
    ):
        blockers.append("runtime_tree_sha256_required")
    if not _finite_numeric_present(
        row,
        ("score_contest_cuda", "contest_cuda_score", "canonical_score_contest_cuda"),
    ):
        blockers.append("contest_cuda_score_required")

    if _dispatch_blockers(row):
        blockers.append("source_dispatch_blockers_present")

    return _ordered_unique(blockers)


def is_promotable_exact_cuda_evidence(row: Mapping[str, Any]) -> bool:
    """Return true when a row satisfies the promotable exact-CUDA contract."""

    return not promotable_exact_cuda_evidence_blockers(row)


__all__ = [
    "has_positive_exact_cuda_evidence_marker",
    "is_promotable_exact_cuda_evidence",
    "is_sha256_hex",
    "promotable_exact_cuda_evidence_blockers",
]
