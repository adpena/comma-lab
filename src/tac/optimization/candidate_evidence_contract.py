"""Evidence-contract helpers for score-lowering candidate planners.

The functions in this module are pure schema checks. They do not run evals,
dispatch jobs, or validate files on disk. They exist so planner surfaces can
fail closed before a score-lowering-looking row becomes active ranking or
promotion evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

EXACT_CUDA_MARKERS = (
    "contest-cuda",
    "contest_cuda",
    "exact_cuda_auth_eval",
    "exact-cuda",
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

    blockers: list[str] = []
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

    if not (
        any(marker in text for marker in EXACT_CUDA_MARKERS)
        or grade in {"a", "a++"}
    ):
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
    "is_promotable_exact_cuda_evidence",
    "is_sha256_hex",
    "promotable_exact_cuda_evidence_blockers",
]
