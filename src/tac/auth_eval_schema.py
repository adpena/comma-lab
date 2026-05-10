"""Normalize contest auth-eval JSON schemas without weakening custody.

The canonical evaluator emits ``canonical_score``, component distances, and
archive custody fields. Some older wrappers emitted a nested
``score_components`` object. Dispatch scripts should not silently lose score
signal when one schema is absent, and they must not mark a result claimable
when canonical fields are missing.
"""

from __future__ import annotations

import math
from typing import Any


FULL_CONTEST_SAMPLE_COUNT = 600
CONTEST_CUDA_EVIDENCE_TAG = "[contest-CUDA]"


def numeric_or_none(value: Any) -> float | None:
    """Return ``value`` as a finite float, excluding bool/null."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def int_or_none(value: Any) -> int | None:
    """Return ``value`` as an int when it is an integer-like scalar."""

    parsed = numeric_or_none(value)
    if parsed is None or int(parsed) != parsed:
        return None
    return int(parsed)


def first_numeric(*values: Any) -> float | None:
    """Return the first numeric value from ``values``."""

    for value in values:
        parsed = numeric_or_none(value)
        if parsed is not None:
            return parsed
    return None


def eval_metric_summary(eval_data: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize canonical and legacy auth-eval result keys.

    ``score`` always prefers the recomputed canonical score over rounded
    display scores. ``rate`` is the score contribution; ``rate_unscaled`` is the
    raw ``archive_bytes / contest_denominator`` term.
    """

    if not eval_data:
        return {
            "score": None,
            "pose_avg": None,
            "seg_avg": None,
            "rate": None,
            "rate_unscaled": None,
            "archive_size_bytes": None,
            "n_samples": None,
            "canonical_score_source": None,
        }
    sc = eval_data.get("score_components") or {}
    if not isinstance(sc, dict):
        sc = {}
    return {
        "score": first_numeric(
            eval_data.get("canonical_score"),
            eval_data.get("score_recomputed_from_components"),
            eval_data.get("score"),
            eval_data.get("total_score"),
            eval_data.get("final_score"),
        ),
        "pose_avg": first_numeric(
            eval_data.get("avg_posenet_dist"),
            eval_data.get("pose_avg"),
            sc.get("pose"),
            sc.get("pose_avg"),
            sc.get("posenet"),
        ),
        "seg_avg": first_numeric(
            eval_data.get("avg_segnet_dist"),
            eval_data.get("seg_avg"),
            sc.get("seg"),
            sc.get("seg_avg"),
            sc.get("segnet"),
        ),
        "rate": first_numeric(
            eval_data.get("score_rate_contribution"),
            eval_data.get("rate"),
            sc.get("rate"),
            sc.get("rate_term"),
        ),
        "rate_unscaled": first_numeric(
            eval_data.get("rate_unscaled"),
            sc.get("rate_unscaled"),
        ),
        "archive_size_bytes": int_or_none(
            eval_data.get("archive_size_bytes")
            if eval_data.get("archive_size_bytes") is not None
            else eval_data.get("archive_bytes")
        ),
        "n_samples": int_or_none(eval_data.get("n_samples")),
        "canonical_score_source": eval_data.get("canonical_score_source"),
    }


def required_exact_eval_metric_blockers(
    metrics: dict[str, Any],
    *,
    expected_archive_bytes: int | None = None,
    expected_n_samples: int | None = None,
) -> list[str]:
    """Return blockers that make an eval JSON non-claimable."""

    blockers: list[str] = []
    for key in ("score", "pose_avg", "seg_avg", "rate_unscaled", "archive_size_bytes"):
        if metrics.get(key) is None:
            blockers.append(f"{key}_missing")
    if metrics.get("canonical_score_source") != "score_recomputed_from_components":
        blockers.append("canonical_score_source_not_recomputed_from_components")
    if (
        expected_archive_bytes is not None
        and metrics.get("archive_size_bytes") is not None
        and metrics["archive_size_bytes"] != expected_archive_bytes
    ):
        blockers.append(
            "archive_size_bytes_mismatch:"
            f"manifest={metrics['archive_size_bytes']}:actual={expected_archive_bytes}"
        )
    if (
        expected_n_samples is not None
        and metrics.get("n_samples") is None
    ):
        blockers.append("n_samples_missing")
    elif (
        expected_n_samples is not None
        and metrics["n_samples"] != expected_n_samples
    ):
        blockers.append(f"n_samples_mismatch:manifest={metrics['n_samples']}:expected={expected_n_samples}")
    return blockers


def _truthy(value: Any) -> bool:
    return value is True or (
        isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}
    )


def _provenance(eval_data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(eval_data, dict):
        return {}
    value = eval_data.get("provenance")
    return value if isinstance(value, dict) else {}


def eval_device(eval_data: dict[str, Any] | None) -> str | None:
    """Return the normalized auth-eval device, preferring provenance custody."""

    if not isinstance(eval_data, dict):
        return None
    prov = _provenance(eval_data)
    device = prov.get("device") if prov.get("device") is not None else eval_data.get("device")
    return str(device).strip().lower() if device is not None else None


def contest_cuda_hardware_is_documented(eval_data: dict[str, Any] | None) -> bool:
    """Return True only for T4 or explicitly documented equivalent CUDA hardware."""

    if not isinstance(eval_data, dict):
        return False
    prov = _provenance(eval_data)
    if _truthy(prov.get("gpu_t4_match")) or _truthy(eval_data.get("gpu_t4_match")):
        return True
    equivalent = _truthy(eval_data.get("contest_equivalent_hardware")) or _truthy(
        prov.get("contest_equivalent_hardware")
    )
    note = (
        eval_data.get("contest_equivalent_hardware_note")
        or eval_data.get("hardware_equivalence_note")
        or eval_data.get("contest_equivalent_hardware_source")
        or eval_data.get("hardware_equivalence_source")
        or prov.get("contest_equivalent_hardware_note")
        or prov.get("hardware_equivalence_note")
        or prov.get("contest_equivalent_hardware_source")
        or prov.get("hardware_equivalence_source")
    )
    return equivalent and isinstance(note, str) and bool(note.strip())


def required_contest_cuda_evidence_blockers(
    eval_data: dict[str, Any] | None,
    metrics: dict[str, Any],
    *,
    expected_archive_bytes: int | None = None,
    expected_n_samples: int = FULL_CONTEST_SAMPLE_COUNT,
) -> list[str]:
    """Return blockers for a claimable ``[contest-CUDA]`` exact-eval result."""

    blockers = required_exact_eval_metric_blockers(
        metrics,
        expected_archive_bytes=expected_archive_bytes,
        expected_n_samples=expected_n_samples,
    )
    if eval_device(eval_data) != "cuda":
        blockers.append("device_not_cuda")
    if not contest_cuda_hardware_is_documented(eval_data):
        blockers.append("contest_cuda_hardware_not_t4_or_documented_equivalent")
    if isinstance(eval_data, dict):
        lane_tag = eval_data.get("lane_tag")
        if lane_tag is not None and lane_tag != CONTEST_CUDA_EVIDENCE_TAG:
            blockers.append("evidence_tag_not_contest_cuda")
        score_axis = eval_data.get("score_axis")
        if score_axis is not None and score_axis != "contest_cuda":
            blockers.append("score_axis_not_contest_cuda")
        semantics = eval_data.get("evidence_semantics")
        if semantics is not None and semantics != "contest_cuda_exact_auth_eval":
            blockers.append("evidence_semantics_not_contest_cuda_exact_auth_eval")
        if "score_claim_valid" in eval_data and eval_data.get("score_claim_valid") is not True:
            blockers.append("score_claim_valid_not_true")
    return blockers
