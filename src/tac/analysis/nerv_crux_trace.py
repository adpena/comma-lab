# SPDX-License-Identifier: MIT
"""Contest-unit crux tracing for NeRV-family training artifacts.

The trace is local diagnostic evidence only. It deliberately carries false
authority flags while preserving the exact score units needed to decide whether
a short smoke is moving SegNet, PoseNet, and archive rate in the right space.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONTEST_ORIGINAL_VIDEO_BYTES = 37_545_489
CONTEST_RATE_SCORE_PER_BYTE = 25.0 / CONTEST_ORIGINAL_VIDEO_BYTES
FALSE_AUTHORITY = "macos_mlx_false_authority_no_score_claim"
TRACE_SCHEMA = "nerv_crux_trace_rows.v1"
TRACE_ATTACHMENT_SCHEMA = "nerv_crux_trace_attachment.v1"

SEGNET_TARGET_REGION_DEBT_KEY = (
    "loss_part_segnet_direct_live_target_min_ratio_floor_"
    "score_weighted_total_unsolved_argmax_mass"
)
SEGNET_TARGET_REGION_DEBT_KEYS = (
    SEGNET_TARGET_REGION_DEBT_KEY,
    "score_weighted_total_unsolved_argmax_mass",
)
SEGNET_ARGMAX_DISAGREEMENT_KEY = "loss_part_segnet_direct_live_argmax_disagreement"
SEGNET_ARGMAX_DISAGREEMENT_KEYS = (
    SEGNET_ARGMAX_DISAGREEMENT_KEY,
    "segnet_direct_live_argmax_disagreement",
    "argmax_disagreement",
)
SEGNET_OCCUPIED_CLASS_FRACTION_KEY = (
    "loss_part_segnet_direct_live_candidate_occupied_class_fraction"
)
SEGNET_OCCUPIED_CLASS_FRACTION_KEYS = (
    SEGNET_OCCUPIED_CLASS_FRACTION_KEY,
    "segnet_direct_live_candidate_occupied_class_fraction",
    "candidate_occupied_class_fraction",
)
SEGNET_TARGET_CLASS_COVERAGE_KEY = (
    "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction"
)
SEGNET_TARGET_CLASS_COVERAGE_KEYS = (
    SEGNET_TARGET_CLASS_COVERAGE_KEY,
    "segnet_direct_live_candidate_target_class_coverage_fraction",
    "candidate_target_class_coverage_fraction",
)
SEGNET_TARGET_CLASS_MIN_RATIO_KEY = (
    "loss_part_segnet_direct_live_candidate_target_class_min_ratio"
)
SEGNET_TARGET_CLASS_MIN_RATIO_KEYS = (
    SEGNET_TARGET_CLASS_MIN_RATIO_KEY,
    "segnet_direct_live_candidate_target_class_min_ratio",
    "candidate_target_class_min_ratio",
)
POSE_RAW_MSE_KEY = "loss_part_pose_direct_live_raw_mse"
POSE_RAW_MSE_KEYS = (POSE_RAW_MSE_KEY, "pose_direct_live_raw_mse")
POSE_SCORE_TERM_KEY = "loss_part_pose_direct_live_score_term"
POSE_SCORE_TERM_KEYS = (POSE_SCORE_TERM_KEY, "pose_direct_live_score_term")
POSE_SCORE_MARGINAL_KEY = "loss_part_pose_direct_live_score_marginal_wrt_raw_mse"
POSE_SCORE_MARGINAL_KEYS = (
    POSE_SCORE_MARGINAL_KEY,
    "pose_direct_live_score_marginal_wrt_raw_mse",
)
POSE_YUV6_PAIR_STD_KEY = "loss_part_pose_direct_live_yuv6_pair_std"
POSE_YUV6_PAIR_STD_KEYS = (
    POSE_YUV6_PAIR_STD_KEY,
    "pose_direct_live_yuv6_pair_std",
)
POSE_YUV6_TEMPORAL_DELTA_STD_KEY = (
    "loss_part_pose_direct_live_yuv6_pair_temporal_delta_std"
)
POSE_YUV6_TEMPORAL_DELTA_STD_KEYS = (
    POSE_YUV6_TEMPORAL_DELTA_STD_KEY,
    "pose_direct_live_yuv6_pair_temporal_delta_std",
)
ARCHIVE_BYTES_KEYS = ("train_time_archive_bytes", "archive_bytes")
ARCHIVE_RATE_SCORE_KEY = "train_time_archive_rate_score"

SEGNET_DIRECT_LIVE_EVIDENCE_KEYS = (
    *SEGNET_TARGET_REGION_DEBT_KEYS,
    *SEGNET_ARGMAX_DISAGREEMENT_KEYS,
    *SEGNET_OCCUPIED_CLASS_FRACTION_KEYS,
    *SEGNET_TARGET_CLASS_COVERAGE_KEYS,
    *SEGNET_TARGET_CLASS_MIN_RATIO_KEYS,
)
POSE_DIRECT_LIVE_EVIDENCE_KEYS = (
    *POSE_RAW_MSE_KEYS,
    *POSE_SCORE_TERM_KEYS,
    *POSE_SCORE_MARGINAL_KEYS,
    *POSE_YUV6_PAIR_STD_KEYS,
    *POSE_YUV6_TEMPORAL_DELTA_STD_KEYS,
)


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def trace_rows_as_dicts(rows: Sequence[TraceRow]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def trace_blockers(rows: Sequence[TraceRow]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        if row.blocker and row.blocker not in seen:
            seen.add(row.blocker)
            out.append(row.blocker)
    return out


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


def _append_unique_mapping(
    mappings: list[Mapping[str, Any]],
    seen: set[int],
    candidate: Mapping[str, Any] | None,
) -> None:
    if candidate is None:
        return
    ident = id(candidate)
    if ident in seen:
        return
    seen.add(ident)
    mappings.append(candidate)


def _last_mapping_from_sequence(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    for item in reversed(value):
        if isinstance(item, Mapping):
            return item
    return None


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_last_jsonl_mapping(path: Path) -> Mapping[str, Any] | None:
    last: Mapping[str, Any] | None = None
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            last = payload
    return last


def _resolve_telemetry_path(
    payload: Mapping[str, Any],
    *,
    training_artifact_path: Path | None,
) -> Path | None:
    candidates: list[Path] = []
    raw = payload.get("telemetry_path")
    if isinstance(raw, str) and raw:
        path = Path(raw).expanduser()
        if not path.is_absolute() and training_artifact_path is not None:
            path = training_artifact_path.parent / path
        candidates.append(path.resolve(strict=False))
    if training_artifact_path is not None:
        candidates.append(training_artifact_path.parent / "telemetry.jsonl")
    for path in candidates:
        if path.is_file():
            return path
    return None


def metric_mappings_from_training_payload(
    payload: Mapping[str, Any],
    *,
    telemetry_last_row: Mapping[str, Any] | None = None,
) -> list[Mapping[str, Any]]:
    """Return metric layers from canonical runner artifact shapes.

    HiNeRV/SNeRV artifacts put archive bytes at the top level and live scorer
    losses inside the last `per_epoch_metrics[*].loss_components` row. A trace
    must search all layers instead of selecting one dictionary.
    """

    mappings: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    for key in (
        "final_loss_components",
        "loss_components",
        "final_metrics",
        "metrics",
        "metric_rows",
    ):
        _append_unique_mapping(mappings, seen, _mapping_at(payload, key))

    latest_epoch = _last_mapping_from_sequence(payload.get("per_epoch_metrics"))
    if latest_epoch is not None:
        _append_unique_mapping(mappings, seen, _mapping_at(latest_epoch, "loss_components"))
        _append_unique_mapping(mappings, seen, _mapping_at(latest_epoch, "metrics"))
        _append_unique_mapping(mappings, seen, latest_epoch)

    if telemetry_last_row is not None:
        _append_unique_mapping(
            mappings,
            seen,
            _mapping_at(telemetry_last_row, "loss_components"),
        )
        _append_unique_mapping(mappings, seen, _mapping_at(telemetry_last_row, "metrics"))
        _append_unique_mapping(mappings, seen, telemetry_last_row)

    for gate_key in ("direct_live_segnet_gate", "direct_live_posenet_gate"):
        gate = _mapping_at(payload, gate_key)
        if gate is not None:
            _append_unique_mapping(mappings, seen, _mapping_at(gate, "metrics"))

    _append_unique_mapping(mappings, seen, payload)
    return mappings


def _metric(metrics: Sequence[Mapping[str, Any]], *keys: str) -> float | None:
    for mapping in metrics:
        for key in keys:
            found = _finite_or_none(mapping.get(key))
            if found is not None:
                return found
    return None


def _has_any_metric(metrics: Sequence[Mapping[str, Any]], *keys: str) -> bool:
    return any(any(key in mapping for key in keys) for mapping in metrics)


def _any_key_with_prefix(metrics: Sequence[Mapping[str, Any]], prefix: str) -> bool:
    return any(
        any(str(key).startswith(prefix) for key in mapping) for mapping in metrics
    )


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
    require_direct_live_posenet: bool = True,
    require_direct_live_segnet: bool = True,
    telemetry_last_row: Mapping[str, Any] | None = None,
) -> list[TraceRow]:
    metrics = metric_mappings_from_training_payload(
        payload,
        telemetry_last_row=telemetry_last_row,
    )
    rows: list[TraceRow] = []

    segnet_target_debt = _metric(metrics, *SEGNET_TARGET_REGION_DEBT_KEYS)
    segnet_blocker = None
    if require_direct_live_segnet and not (
        _any_key_with_prefix(metrics, "loss_part_segnet_direct_live_")
        or _has_any_metric(metrics, *SEGNET_DIRECT_LIVE_EVIDENCE_KEYS)
    ):
        segnet_blocker = "missing_direct_live_segnet_path"
    elif require_direct_live_segnet and segnet_target_debt is None:
        segnet_blocker = "missing_direct_live_segnet_target_region_debt"
    _append_metric_row(
        rows,
        source_path=source_path,
        source_sha256=source_sha256,
        stage="loss",
        axis="segnet",
        metric="score_weighted_total_unsolved_argmax_mass",
        value=segnet_target_debt,
        score_units=segnet_target_debt,
        blocker=segnet_blocker,
    )
    for keys, metric_name in (
        (SEGNET_ARGMAX_DISAGREEMENT_KEYS, "argmax_disagreement"),
        (SEGNET_OCCUPIED_CLASS_FRACTION_KEYS, "candidate_occupied_class_fraction"),
        (SEGNET_TARGET_CLASS_COVERAGE_KEYS, "candidate_target_class_coverage_fraction"),
        (SEGNET_TARGET_CLASS_MIN_RATIO_KEYS, "candidate_target_class_min_ratio"),
    ):
        _append_metric_row(
            rows,
            source_path=source_path,
            source_sha256=source_sha256,
            stage="loss",
            axis="segnet",
            metric=metric_name,
            value=_metric(metrics, *keys),
        )

    pose_raw_mse = _metric(metrics, *POSE_RAW_MSE_KEYS)
    observed_pose_score = _metric(metrics, *POSE_SCORE_TERM_KEYS)
    derived_pose_score = _pose_score_term(pose_raw_mse)
    observed_pose_marginal = _metric(metrics, *POSE_SCORE_MARGINAL_KEYS)
    derived_pose_marginal = _pose_marginal(pose_raw_mse)
    pose_blocker = None
    if require_direct_live_posenet and not (
        _any_key_with_prefix(metrics, "loss_part_pose_direct_live_")
        or _has_any_metric(metrics, *POSE_DIRECT_LIVE_EVIDENCE_KEYS)
    ):
        pose_blocker = "missing_direct_live_posenet_path"
    elif require_direct_live_posenet and pose_raw_mse is None:
        pose_blocker = "missing_direct_live_posenet_raw_mse"

    _append_metric_row(
        rows,
        source_path=source_path,
        source_sha256=source_sha256,
        stage="loss",
        axis="posenet",
        metric="pose_direct_live_raw_mse",
        value=pose_raw_mse,
        blocker=pose_blocker,
    )
    _append_metric_row(
        rows,
        source_path=source_path,
        source_sha256=source_sha256,
        stage="loss",
        axis="posenet",
        metric="pose_direct_live_score_term",
        value=observed_pose_score if observed_pose_score is not None else derived_pose_score,
        score_units=(
            observed_pose_score
            if observed_pose_score is not None
            else derived_pose_score
        ),
    )
    _append_metric_row(
        rows,
        source_path=source_path,
        source_sha256=source_sha256,
        stage="loss",
        axis="posenet",
        metric="pose_direct_live_score_marginal_wrt_raw_mse",
        value=(
            observed_pose_marginal
            if observed_pose_marginal is not None
            else derived_pose_marginal
        ),
    )
    for keys, metric_name in (
        (POSE_YUV6_PAIR_STD_KEYS, "pose_direct_live_yuv6_pair_std"),
        (
            POSE_YUV6_TEMPORAL_DELTA_STD_KEYS,
            "pose_direct_live_yuv6_pair_temporal_delta_std",
        ),
    ):
        _append_metric_row(
            rows,
            source_path=source_path,
            source_sha256=source_sha256,
            stage="loss",
            axis="posenet",
            metric=metric_name,
            value=_metric(metrics, *keys),
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


def build_trace_rows_for_training_artifact(
    training_artifact_path: Path,
    *,
    require_direct_live_posenet: bool = True,
    require_direct_live_segnet: bool = True,
) -> list[TraceRow]:
    source = training_artifact_path.expanduser().resolve(strict=False)
    payload = _read_json_mapping(source)
    telemetry_path = _resolve_telemetry_path(payload, training_artifact_path=source)
    telemetry_last_row = (
        _read_last_jsonl_mapping(telemetry_path) if telemetry_path is not None else None
    )
    return build_trace_rows(
        payload,
        source_path=source.as_posix(),
        source_sha256=sha256_file(source),
        require_direct_live_posenet=bool(require_direct_live_posenet),
        require_direct_live_segnet=bool(require_direct_live_segnet),
        telemetry_last_row=telemetry_last_row,
    )


def write_trace_rows_for_training_artifact(
    training_artifact_path: Path,
    *,
    output_path: Path,
    require_direct_live_posenet: bool = True,
    require_direct_live_segnet: bool = True,
) -> dict[str, Any]:
    source = training_artifact_path.expanduser().resolve(strict=False)
    out = output_path.expanduser().resolve(strict=False)
    rows = build_trace_rows_for_training_artifact(
        source,
        require_direct_live_posenet=bool(require_direct_live_posenet),
        require_direct_live_segnet=bool(require_direct_live_segnet),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(trace_rows_as_dicts(rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema": TRACE_ATTACHMENT_SCHEMA,
        "path": out.as_posix(),
        "sha256": sha256_file(out),
        "source_path": source.as_posix(),
        "source_sha256": sha256_file(source),
        "row_count": len(rows),
        "blockers": trace_blockers(rows),
        "authority": FALSE_AUTHORITY,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
