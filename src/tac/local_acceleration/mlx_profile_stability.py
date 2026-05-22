# SPDX-License-Identifier: MIT
"""Fail-closed stability audit for local MLX scorer-response profiles."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

SCHEMA_VERSION = "mlx_scorer_response_profile_stability.v1"
PASS_VERDICT = "PASS_MLX_PROFILE_STABILITY"
FAIL_VERDICT = "FAIL_MLX_PROFILE_STABILITY"


@dataclass(frozen=True)
class MLXProfileStabilityThresholds:
    """Thresholds for treating profile rows as the same local scorer signal."""

    max_score_abs_delta: float = 1.0e-5
    max_posenet_avg_abs_delta: float = 1.0e-8
    max_segnet_avg_abs_delta: float = 1.0e-8
    require_component_sha_match: bool = False


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def write_profile_stability_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_profile_stability_manifest(
    profile: dict[str, Any],
    *,
    thresholds: MLXProfileStabilityThresholds | None = None,
    baseline_device: str | None = None,
    baseline_batch_pairs: int | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Compare all profile rows to one baseline row.

    This is a local-MLX stability check only. Passing means the profiled
    batch/device choices agree within local numerical tolerances; it does not
    make the MLX profile an auth-eval score.
    """

    limits = thresholds or MLXProfileStabilityThresholds()
    blockers: list[str] = []
    warnings: list[str] = []

    _append_profile_authority_blockers(blockers, profile)
    rows = profile.get("rows")
    if not isinstance(rows, list) or not rows:
        blockers.append("profile_rows_missing_or_empty")
        rows = []

    normalized_rows = [_normalize_row(index, row, blockers) for index, row in enumerate(rows)]
    normalized_rows = [row for row in normalized_rows if row is not None]
    baseline = _select_baseline_row(
        normalized_rows,
        baseline_device=baseline_device,
        baseline_batch_pairs=baseline_batch_pairs,
        blockers=blockers,
    )
    global_blockers = list(blockers)

    comparisons: list[dict[str, Any]] = []
    if baseline is not None:
        for row in normalized_rows:
            comparison = _compare_row(row, baseline, limits)
            comparisons.append(comparison)
            blockers.extend(comparison["blockers"])
            warnings.extend(comparison["warnings"])
    selection = _build_row_selection(
        rows=normalized_rows,
        comparisons=comparisons,
        global_blockers=global_blockers,
    )

    passed = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "verdict": PASS_VERDICT if passed else FAIL_VERDICT,
        "passed": passed,
        "blockers": blockers,
        "warnings": warnings,
        "thresholds": asdict(limits),
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "baseline": baseline,
        "comparisons": comparisons,
        "selection": selection,
        "profile_summary": {
            "schema_version": profile.get("schema_version"),
            "row_count": len(normalized_rows),
            "reference_cache_dir": profile.get("reference_cache_dir"),
            "candidate_cache_dir": profile.get("candidate_cache_dir"),
            "archive_size_bytes": profile.get("archive_size_bytes"),
            "start_pair": profile.get("start_pair"),
            "max_pairs": profile.get("max_pairs"),
        },
        "authority_status": (
            "Profile stability is local MLX evidence only; paired CPU/CUDA auth eval "
            "remains required for score claims or promotion."
        ),
    }


def _append_profile_authority_blockers(blockers: list[str], profile: dict[str, Any]) -> None:
    if profile.get("score_claim") is True or profile.get("score_claim_valid") is True:
        blockers.append("profile_attempts_score_claim")
    if profile.get("promotion_eligible") is True:
        blockers.append("profile_attempts_promotion_eligibility")
    if profile.get("rank_or_kill_eligible") is True:
        blockers.append("profile_attempts_rank_or_kill_eligibility")
    evidence_grade = profile.get("evidence_grade")
    if evidence_grade is None:
        blockers.append("profile_evidence_grade_missing")
    elif evidence_grade != EVIDENCE_GRADE_MLX:
        blockers.append(f"profile_evidence_grade_not_{EVIDENCE_GRADE_MLX}")


def _normalize_row(index: int, row: Any, blockers: list[str]) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        blockers.append(f"profile_row_not_object:index={index}")
        return None
    required = (
        "device",
        "batch_pairs",
        "n_samples",
        "pair_window",
        "canonical_score",
        "avg_posenet_dist",
        "avg_segnet_dist",
        "posenet_sha256",
        "segnet_sha256",
    )
    missing = [key for key in required if key not in row]
    if missing:
        blockers.append(f"profile_row_missing_fields:index={index}:fields={','.join(missing)}")
        return None
    try:
        normalized = {
            "index": index,
            "device": str(row["device"]),
            "batch_pairs": int(row["batch_pairs"]),
            "n_samples": int(row["n_samples"]),
            "pair_window": list(row["pair_window"]),
            "canonical_score": _finite_float(row["canonical_score"]),
            "avg_posenet_dist": _finite_float(row["avg_posenet_dist"]),
            "avg_segnet_dist": _finite_float(row["avg_segnet_dist"]),
            "posenet_sha256": str(row["posenet_sha256"]),
            "segnet_sha256": str(row["segnet_sha256"]),
            "pairs_per_second": _optional_finite_float(row.get("pairs_per_second")),
            "elapsed_seconds": _optional_finite_float(row.get("elapsed_seconds")),
            "wall_seconds": _optional_finite_float(row.get("wall_seconds")),
            "repeat_index": _optional_int(row.get("repeat_index")),
            "start_pair": _optional_int(row.get("start_pair")),
        }
    except (TypeError, ValueError) as exc:
        blockers.append(f"profile_row_invalid:index={index}:error={exc}")
        return None
    if normalized["batch_pairs"] < 1:
        blockers.append(f"profile_row_batch_pairs_nonpositive:index={index}")
    if normalized["n_samples"] < 1:
        blockers.append(f"profile_row_n_samples_nonpositive:index={index}")
    if len(normalized["pair_window"]) != 2:
        blockers.append(f"profile_row_pair_window_invalid:index={index}")
    return normalized


def _select_baseline_row(
    rows: list[dict[str, Any]],
    *,
    baseline_device: str | None,
    baseline_batch_pairs: int | None,
    blockers: list[str],
) -> dict[str, Any] | None:
    if not rows:
        return None
    if baseline_device is None and baseline_batch_pairs is None:
        return rows[0]
    matches = [
        row
        for row in rows
        if (baseline_device is None or row["device"] == baseline_device)
        and (baseline_batch_pairs is None or row["batch_pairs"] == int(baseline_batch_pairs))
    ]
    if not matches:
        blockers.append(
            "baseline_row_not_found:"
            f"device={baseline_device!r}:batch_pairs={baseline_batch_pairs!r}"
        )
        return rows[0]
    return matches[0]


def _compare_row(
    row: dict[str, Any],
    baseline: dict[str, Any],
    limits: MLXProfileStabilityThresholds,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if row["n_samples"] != baseline["n_samples"]:
        blockers.append(
            "profile_row_n_samples_mismatch:"
            f"index={row['index']}:row={row['n_samples']}:baseline={baseline['n_samples']}"
        )
    if row["pair_window"] != baseline["pair_window"]:
        blockers.append(
            "profile_row_pair_window_mismatch:"
            f"index={row['index']}:row={row['pair_window']}:baseline={baseline['pair_window']}"
        )

    deltas = {
        "score": row["canonical_score"] - baseline["canonical_score"],
        "posenet_avg": row["avg_posenet_dist"] - baseline["avg_posenet_dist"],
        "segnet_avg": row["avg_segnet_dist"] - baseline["avg_segnet_dist"],
    }
    for key, limit in (
        ("score", limits.max_score_abs_delta),
        ("posenet_avg", limits.max_posenet_avg_abs_delta),
        ("segnet_avg", limits.max_segnet_avg_abs_delta),
    ):
        if abs(deltas[key]) > limit:
            blockers.append(
                f"profile_row_{key}_delta_exceeds_threshold:"
                f"index={row['index']}:{abs(deltas[key]):.12g}>{limit:.12g}"
            )

    for key in ("posenet_sha256", "segnet_sha256"):
        if row[key] != baseline[key]:
            message = f"profile_row_{key}_mismatch:index={row['index']}"
            if limits.require_component_sha_match:
                blockers.append(message)
            else:
                warnings.append(message)

    return {
        "index": row["index"],
        "device": row["device"],
        "batch_pairs": row["batch_pairs"],
        "pairs_per_second": row.get("pairs_per_second"),
        "delta": deltas,
        "blockers": blockers,
        "warnings": warnings,
    }


def _build_row_selection(
    *,
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    global_blockers: list[str],
) -> dict[str, Any]:
    comparisons_by_index = {int(item["index"]): item for item in comparisons}
    eligible_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for row in rows:
        comparison = comparisons_by_index.get(int(row["index"]))
        comparison_blockers = [] if comparison is None else list(comparison["blockers"])
        if global_blockers:
            rejected_rows.append(
                {
                    "index": row["index"],
                    "device": row["device"],
                    "batch_pairs": row["batch_pairs"],
                    "reason": "global_profile_blockers",
                    "blockers": global_blockers,
                }
            )
        elif comparison_blockers:
            rejected_rows.append(
                {
                    "index": row["index"],
                    "device": row["device"],
                    "batch_pairs": row["batch_pairs"],
                    "reason": "row_stability_blockers",
                    "blockers": comparison_blockers,
                }
            )
        else:
            eligible_rows.append(row)

    recommended = _select_fastest_row(eligible_rows)
    return {
        "policy": "fastest_row_with_no_stability_blockers",
        "eligible_row_indices": [int(row["index"]) for row in eligible_rows],
        "rejected_rows": rejected_rows,
        "recommended_row": _public_row(recommended) if recommended is not None else None,
        "recommended_reason": (
            "fastest eligible row by pairs_per_second"
            if recommended is not None and recommended.get("pairs_per_second") is not None
            else "first eligible row; throughput missing"
            if recommended is not None
            else "no eligible rows"
        ),
    }


def _select_fastest_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row["pairs_per_second"])
            if row.get("pairs_per_second") is not None
            else float("-inf"),
            -int(row["index"]),
        ),
    )


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": row["index"],
        "device": row["device"],
        "batch_pairs": row["batch_pairs"],
        "n_samples": row["n_samples"],
        "pair_window": row["pair_window"],
        "canonical_score": row["canonical_score"],
        "avg_posenet_dist": row["avg_posenet_dist"],
        "avg_segnet_dist": row["avg_segnet_dist"],
        "pairs_per_second": row.get("pairs_per_second"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "wall_seconds": row.get("wall_seconds"),
        "repeat_index": row.get("repeat_index"),
        "start_pair": row.get("start_pair"),
    }


def _finite_float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"non-finite float: {value!r}")
    return out


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    return _finite_float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"expected int, got bool: {value!r}")
    return int(value)
