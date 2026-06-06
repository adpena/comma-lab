#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Trace where scorer-space signal dies in a NeRV training artifact.

This is a diagnostic harness, not score authority. It consumes an already
materialized metric/training JSON and emits contest-unit rows for the axes that
most often collapse before long training: SegNet target-region support, PoseNet
YUV6 direct-live signal, and archive rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONTEST_ORIGINAL_VIDEO_BYTES = 37_545_489
CONTEST_RATE_SCORE_PER_BYTE = 25.0 / CONTEST_ORIGINAL_VIDEO_BYTES
FALSE_AUTHORITY = "macos_mlx_false_authority_no_score_claim"
TRACE_SCHEMA = "nerv_crux_trace_rows.v1"

SEGNET_TARGET_REGION_DEBT_KEY = (
    "loss_part_segnet_direct_live_target_min_ratio_floor_"
    "score_weighted_total_unsolved_argmax_mass"
)
SEGNET_ARGMAX_DISAGREEMENT_KEY = "loss_part_segnet_direct_live_argmax_disagreement"
SEGNET_OCCUPIED_CLASS_FRACTION_KEY = (
    "loss_part_segnet_direct_live_candidate_occupied_class_fraction"
)
SEGNET_TARGET_CLASS_COVERAGE_KEY = (
    "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction"
)
SEGNET_TARGET_CLASS_MIN_RATIO_KEY = (
    "loss_part_segnet_direct_live_candidate_target_class_min_ratio"
)
POSE_RAW_MSE_KEY = "loss_part_pose_direct_live_raw_mse"
POSE_SCORE_TERM_KEY = "loss_part_pose_direct_live_score_term"
POSE_SCORE_MARGINAL_KEY = "loss_part_pose_direct_live_score_marginal_wrt_raw_mse"
POSE_YUV6_PAIR_STD_KEY = "loss_part_pose_direct_live_yuv6_pair_std"
POSE_YUV6_TEMPORAL_DELTA_STD_KEY = (
    "loss_part_pose_direct_live_yuv6_pair_temporal_delta_std"
)
ARCHIVE_BYTES_KEYS = ("train_time_archive_bytes", "archive_bytes")
ARCHIVE_RATE_SCORE_KEY = "train_time_archive_rate_score"


@dataclass(frozen=True)
class TraceRow:
    schema: str
    stage: str
    axis: str
    metric: str
    value: float | None
    score_units: float | None = None
    authority: str = FALSE_AUTHORITY
    source_path: str | None = None
    source_sha256: str | None = None
    blocker: str | None = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _pose_score_term(raw_mse: float | None) -> float | None:
    if raw_mse is None or raw_mse < 0.0:
        return None
    return math.sqrt(10.0 * raw_mse + 1.0e-12)


def _pose_marginal(raw_mse: float | None) -> float | None:
    if raw_mse is None or raw_mse < 0.0:
        return None
    return 5.0 / math.sqrt(10.0 * raw_mse + 1.0e-12)


def _mapping_at(payload: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else None


def _metric_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the most metric-like mapping from known artifact shapes."""

    for key in (
        "final_loss_components",
        "loss_components",
        "final_metrics",
        "metrics",
        "metric_rows",
    ):
        found = _mapping_at(payload, key)
        if found is not None:
            return found
    return payload


def _metric(metrics: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        found = _finite_or_none(metrics.get(key))
        if found is not None:
            return found
    return None


def _append_metric_row(
    rows: list[TraceRow],
    *,
    source_path: str,
    source_sha256: str,
    stage: str,
    axis: str,
    metric: str,
    value: float | None,
    score_units: float | None = None,
    blocker: str | None = None,
) -> None:
    rows.append(
        TraceRow(
            schema=TRACE_SCHEMA,
            stage=stage,
            axis=axis,
            metric=metric,
            value=value,
            score_units=score_units,
            source_path=source_path,
            source_sha256=source_sha256,
            blocker=blocker,
        )
    )


def build_trace_rows(
    payload: Mapping[str, Any],
    *,
    source_path: str,
    source_sha256: str,
    require_direct_live_posenet: bool = False,
    require_direct_live_segnet: bool = False,
) -> list[TraceRow]:
    metrics = _metric_mapping(payload)
    rows: list[TraceRow] = []

    segnet_target_debt = _metric(metrics, SEGNET_TARGET_REGION_DEBT_KEY)
    _append_metric_row(
        rows,
        source_path=source_path,
        source_sha256=source_sha256,
        stage="loss",
        axis="segnet",
        metric="score_weighted_total_unsolved_argmax_mass",
        value=segnet_target_debt,
        score_units=segnet_target_debt,
        blocker=(
            "missing_segnet_target_region_debt"
            if require_direct_live_segnet and segnet_target_debt is None
            else None
        ),
    )
    for key, metric_name in (
        (SEGNET_ARGMAX_DISAGREEMENT_KEY, "argmax_disagreement"),
        (SEGNET_OCCUPIED_CLASS_FRACTION_KEY, "candidate_occupied_class_fraction"),
        (SEGNET_TARGET_CLASS_COVERAGE_KEY, "candidate_target_class_coverage_fraction"),
        (SEGNET_TARGET_CLASS_MIN_RATIO_KEY, "candidate_target_class_min_ratio"),
    ):
        _append_metric_row(
            rows,
            source_path=source_path,
            source_sha256=source_sha256,
            stage="loss",
            axis="segnet",
            metric=metric_name,
            value=_metric(metrics, key),
        )

    pose_raw_mse = _metric(metrics, POSE_RAW_MSE_KEY)
    observed_pose_score = _metric(metrics, POSE_SCORE_TERM_KEY)
    derived_pose_score = _pose_score_term(pose_raw_mse)
    observed_pose_marginal = _metric(metrics, POSE_SCORE_MARGINAL_KEY)
    derived_pose_marginal = _pose_marginal(pose_raw_mse)
    pose_blocker = None
    if require_direct_live_posenet and pose_raw_mse is None:
        pose_blocker = "missing_direct_live_posenet_raw_mse"
    elif require_direct_live_posenet and observed_pose_marginal is None:
        pose_blocker = "missing_direct_live_posenet_score_marginal"

    _append_metric_row(
        rows,
        source_path=source_path,
        source_sha256=source_sha256,
        stage="loss",
        axis="posenet",
        metric="pose_direct_live_raw_mse",
        value=pose_raw_mse,
        blocker=pose_blocker if pose_raw_mse is None else None,
    )
    _append_metric_row(
        rows,
        source_path=source_path,
        source_sha256=source_sha256,
        stage="loss",
        axis="posenet",
        metric="pose_direct_live_score_term",
        value=observed_pose_score if observed_pose_score is not None else derived_pose_score,
        score_units=observed_pose_score
        if observed_pose_score is not None
        else derived_pose_score,
    )
    _append_metric_row(
        rows,
        source_path=source_path,
        source_sha256=source_sha256,
        stage="loss",
        axis="posenet",
        metric="pose_direct_live_score_marginal_wrt_raw_mse",
        value=observed_pose_marginal
        if observed_pose_marginal is not None
        else derived_pose_marginal,
        blocker=pose_blocker if observed_pose_marginal is None else None,
    )
    for key, metric_name in (
        (POSE_YUV6_PAIR_STD_KEY, "pose_direct_live_yuv6_pair_std"),
        (POSE_YUV6_TEMPORAL_DELTA_STD_KEY, "pose_direct_live_yuv6_pair_temporal_delta_std"),
    ):
        _append_metric_row(
            rows,
            source_path=source_path,
            source_sha256=source_sha256,
            stage="loss",
            axis="posenet",
            metric=metric_name,
            value=_metric(metrics, key),
        )

    archive_bytes = _metric(metrics, *ARCHIVE_BYTES_KEYS)
    archive_rate_score = _metric(metrics, ARCHIVE_RATE_SCORE_KEY)
    if archive_rate_score is None and archive_bytes is not None:
        archive_rate_score = archive_bytes * CONTEST_RATE_SCORE_PER_BYTE
    _append_metric_row(
        rows,
        source_path=source_path,
        source_sha256=source_sha256,
        stage="rate",
        axis="rate",
        metric="archive_bytes",
        value=archive_bytes,
    )
    _append_metric_row(
        rows,
        source_path=source_path,
        source_sha256=source_sha256,
        stage="rate",
        axis="rate",
        metric="archive_rate_score",
        value=archive_rate_score,
        score_units=archive_rate_score,
    )

    return rows


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SystemExit(f"{path} must contain a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-direct-live-posenet", action="store_true")
    parser.add_argument("--require-direct-live-segnet", action="store_true")
    args = parser.parse_args()

    source = args.training_artifact
    digest = _sha256(source)
    rows = build_trace_rows(
        _read_json(source),
        source_path=source.as_posix(),
        source_sha256=digest,
        require_direct_live_posenet=bool(args.require_direct_live_posenet),
        require_direct_live_segnet=bool(args.require_direct_live_segnet),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
