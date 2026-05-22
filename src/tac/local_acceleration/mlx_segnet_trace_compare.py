# SPDX-License-Identifier: MIT
"""Compare local PyTorch-vs-MLX SegNet layer-trace manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "mlx_segnet_layer_trace_comparison.v1"


class MLXSegNetTraceComparisonError(ValueError):
    """Raised when layer-trace manifests cannot be compared."""


def load_trace_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MLXSegNetTraceComparisonError(f"{path}: expected JSON object")
    return payload


def compare_mlx_segnet_layer_traces(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    top_k: int = 12,
) -> dict[str, Any]:
    """Compare two non-authoritative MLX SegNet layer-trace manifests."""

    if top_k <= 0:
        raise MLXSegNetTraceComparisonError("top_k must be positive")
    _require_trace_manifest(baseline, label=baseline_label)
    _require_trace_manifest(candidate, label=candidate_label)
    baseline_rows = _rows_by_name(baseline, label=baseline_label)
    candidate_rows = _rows_by_name(candidate, label=candidate_label)
    common_names = sorted(set(baseline_rows) & set(candidate_rows))

    comparison_rows = [
        _compare_row(
            name=name,
            baseline=baseline_rows[name],
            candidate=candidate_rows[name],
        )
        for name in common_names
    ]
    comparison_rows.sort(key=lambda row: row["name"])
    worsened = sorted(
        comparison_rows,
        key=lambda row: (
            _none_to_neg_inf(row.get("max_abs_delta_change")),
            _none_to_neg_inf(row.get("p99_abs_delta_change")),
            row["name"],
        ),
        reverse=True,
    )[:top_k]
    improved = sorted(
        comparison_rows,
        key=lambda row: (
            _none_to_pos_inf(row.get("max_abs_delta_change")),
            _none_to_pos_inf(row.get("p99_abs_delta_change")),
            row["name"],
        ),
    )[:top_k]

    baseline_argmax = _as_int(baseline.get("segnet_argmax_diff_pixels"))
    candidate_argmax = _as_int(candidate.get("segnet_argmax_diff_pixels"))
    baseline_cliff = baseline.get("drift_cliff") if isinstance(baseline.get("drift_cliff"), dict) else None
    candidate_cliff = candidate.get("drift_cliff") if isinstance(candidate.get("drift_cliff"), dict) else None
    verdict = _comparison_verdict(
        baseline_argmax=baseline_argmax,
        candidate_argmax=candidate_argmax,
        baseline_cliff=baseline_cliff,
        candidate_cliff=candidate_cliff,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": "tac.local_acceleration.mlx_segnet_trace_compare",
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "verdict": verdict,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "baseline_pair_window": baseline.get("pair_window"),
        "candidate_pair_window": candidate.get("pair_window"),
        "baseline_trace_count": len(baseline_rows),
        "candidate_trace_count": len(candidate_rows),
        "common_row_count": len(common_names),
        "missing_in_candidate": sorted(set(baseline_rows) - set(candidate_rows)),
        "missing_in_baseline": sorted(set(candidate_rows) - set(baseline_rows)),
        "baseline_segnet_argmax_diff_pixels": baseline_argmax,
        "candidate_segnet_argmax_diff_pixels": candidate_argmax,
        "segnet_argmax_diff_pixels_change": (
            None
            if baseline_argmax is None or candidate_argmax is None
            else candidate_argmax - baseline_argmax
        ),
        "baseline_drift_cliff": baseline_cliff,
        "candidate_drift_cliff": candidate_cliff,
        "worsened_rows_top": worsened,
        "improved_rows_top": improved,
        "rows": comparison_rows,
    }


def write_trace_comparison(comparison: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_trace_manifest(payload: dict[str, Any], *, label: str) -> None:
    if payload.get("schema_version") != "mlx_segnet_layer_trace.v1":
        raise MLXSegNetTraceComparisonError(f"{label}: schema_version mismatch")
    if payload.get("score_claim") is not False:
        raise MLXSegNetTraceComparisonError(f"{label}: score_claim must be false")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise MLXSegNetTraceComparisonError(f"{label}: rows must be a list")


def _rows_by_name(payload: dict[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(payload.get("rows") or []):
        if not isinstance(row, dict):
            raise MLXSegNetTraceComparisonError(f"{label}: row {index} must be an object")
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise MLXSegNetTraceComparisonError(f"{label}: row {index} missing name")
        if name in out:
            raise MLXSegNetTraceComparisonError(f"{label}: duplicate row name {name}")
        out[name] = row
    return out


def _compare_row(
    *,
    name: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    fields = (
        "max_abs_delta",
        "mean_abs_delta",
        "rms_delta",
        "p95_abs_delta",
        "p99_abs_delta",
    )
    row: dict[str, Any] = {
        "name": name,
        "baseline_shape_match": baseline.get("shape_match"),
        "candidate_shape_match": candidate.get("shape_match"),
        "baseline_exceeds_cliff_threshold": baseline.get("exceeds_cliff_threshold"),
        "candidate_exceeds_cliff_threshold": candidate.get("exceeds_cliff_threshold"),
    }
    for field in fields:
        lhs = _as_float(baseline.get(field))
        rhs = _as_float(candidate.get(field))
        row[f"baseline_{field}"] = lhs
        row[f"candidate_{field}"] = rhs
        row[f"{field}_change"] = None if lhs is None or rhs is None else rhs - lhs
    return row


def _comparison_verdict(
    *,
    baseline_argmax: int | None,
    candidate_argmax: int | None,
    baseline_cliff: dict[str, Any] | None,
    candidate_cliff: dict[str, Any] | None,
) -> str:
    if baseline_argmax is not None and candidate_argmax is not None:
        if candidate_argmax < baseline_argmax:
            return "TRACE_CANDIDATE_IMPROVED_ARGMAX"
        if candidate_argmax > baseline_argmax:
            return "TRACE_CANDIDATE_WORSENED_ARGMAX"
    baseline_cliff_max = _as_float((baseline_cliff or {}).get("max_abs_delta"))
    candidate_cliff_max = _as_float((candidate_cliff or {}).get("max_abs_delta"))
    if baseline_cliff_max is not None and candidate_cliff_max is not None:
        if candidate_cliff_max < baseline_cliff_max:
            return "TRACE_CANDIDATE_IMPROVED_DRIFT"
        if candidate_cliff_max > baseline_cliff_max:
            return "TRACE_CANDIDATE_WORSENED_DRIFT"
    return "TRACE_CANDIDATE_NEUTRAL"


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    return None


def _none_to_neg_inf(value: Any) -> float:
    number = _as_float(value)
    return float("-inf") if number is None else number


def _none_to_pos_inf(value: Any) -> float:
    number = _as_float(value)
    return float("inf") if number is None else number


__all__ = [
    "SCHEMA_VERSION",
    "MLXSegNetTraceComparisonError",
    "compare_mlx_segnet_layer_traces",
    "load_trace_manifest",
    "write_trace_comparison",
]
