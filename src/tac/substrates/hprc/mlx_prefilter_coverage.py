# SPDX-License-Identifier: MIT
"""Coverage classification for HPRC MLX scorer-response prefilters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.substrates.hprc.resolution_contract import CONTEST_PAIR_COUNT

HPRC_MLX_COMPONENT_PROFILE_SCHEMA = "hprc_mlx_component_neutralization_profile.v1"
MLX_SCORER_RESPONSE_SCHEMA = "mlx_scorer_response.v1"
HPRC_MLX_PREFILTER_COVERAGE_SCHEMA = "hprc_mlx_prefilter_coverage.v1"
DEFAULT_MAX_MLX_SCORE_FOR_LOCAL_REPLAY = 0.5


def mlx_profile_pair_count(profile: dict[str, Any]) -> int | None:
    """Return the largest declared pair/sample count in an MLX profile."""

    counts: list[int] = []
    containers = [
        profile,
        profile.get("mlx_response_summary"),
        profile.get("response_metadata"),
        profile.get("score_context"),
    ]
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in (
            "max_pairs",
            "num_pairs",
            "n_samples",
            "candidate_cache_pairs",
            "reference_cache_pairs",
        ):
            value = _nonnegative_int(container.get(key))
            if value is not None:
                counts.append(value)
    return max(counts) if counts else None


def mlx_profile_batch_pairs(profile: dict[str, Any]) -> int | None:
    """Return the declared scorer batch-pair count when present."""

    containers = [
        profile,
        profile.get("mlx_response_summary"),
        profile.get("response_metadata"),
        profile.get("score_context"),
    ]
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in ("scorer_batch_pairs", "batch_pairs"):
            value = _nonnegative_int(container.get(key))
            if value is not None:
                return value
    return None


def mlx_profile_full_video_scope(profile: dict[str, Any]) -> str:
    """Return the profile's declared full-video scope marker."""

    if profile.get("schema") == MLX_SCORER_RESPONSE_SCHEMA:
        count = mlx_profile_pair_count(profile)
        return (
            "executed"
            if count is not None and int(count) >= int(CONTEST_PAIR_COUNT)
            else "sampled_prefix_requires_full_video_rerun"
        )
    scope = profile.get("scope_status")
    if not isinstance(scope, dict):
        return "missing_scope_status"
    marker = scope.get("full_video")
    if marker == "executed" or marker is True:
        return "executed"
    if isinstance(marker, str) and marker:
        return marker
    return "missing_full_video_scope"


def mlx_profile_has_full_video_coverage(
    profile: dict[str, Any],
    *,
    required_pairs: int = CONTEST_PAIR_COUNT,
) -> bool:
    """Return whether a profile covers the full video.

    Full-video coverage is useful acquisition evidence even when it is batched
    or run on local MLX GPU. Unlocking local CPU replay is stricter and handled
    separately by ``local_replay_prefilter`` below.
    """

    if profile.get("schema") not in {
        HPRC_MLX_COMPONENT_PROFILE_SCHEMA,
        MLX_SCORER_RESPONSE_SCHEMA,
    }:
        return False
    count = mlx_profile_pair_count(profile)
    return (
        mlx_profile_full_video_scope(profile) == "executed"
        and count is not None
        and int(count) >= int(required_pairs)
    )


def mlx_profile_score_estimate(profile: dict[str, Any]) -> float | None:
    """Return the profile's full-video MLX score estimate when present."""

    containers = [
        profile,
        profile.get("score_components"),
        profile.get("mlx_response_summary"),
        profile.get("response_metadata"),
    ]
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in (
            "canonical_score",
            "recomputed_total_score",
            "score_recomputed_from_components",
            "local_score_estimate",
        ):
            value = _finite_float(container.get(key))
            if value is not None:
                return value
    return None


def summarize_mlx_prefilter_coverage(
    profile_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    root: str | Path,
    required_pairs: int = CONTEST_PAIR_COUNT,
    max_mlx_score_for_local_replay: float | None = DEFAULT_MAX_MLX_SCORE_FOR_LOCAL_REPLAY,
) -> dict[str, Any]:
    """Load MLX profiles and report whether any is full-video replay evidence."""

    base = Path(root).expanduser().resolve(strict=False)
    records = [
        _profile_path_record(path, root=base, required_pairs=int(required_pairs))
        for path in profile_paths
    ]
    has_full = any(record.get("full_video_prefilter") is True for record in records)
    score_threshold = _finite_float(max_mlx_score_for_local_replay)
    full_records = [
        record for record in records if record.get("full_video_prefilter") is True
    ]
    replay_records = [
        record for record in records if record.get("local_replay_prefilter") is True
    ]
    scored_full_records = [
        record
        for record in full_records
        if _finite_float(record.get("mlx_score_estimate")) is not None
    ]
    scored_replay_records = [
        record
        for record in replay_records
        if _finite_float(record.get("mlx_score_estimate")) is not None
    ]
    local_replay_passed = bool(replay_records)
    if score_threshold is not None:
        local_replay_passed = any(
            float(record["mlx_score_estimate"]) < score_threshold
            for record in scored_replay_records
        )
    blockers: list[str] = []
    if not records:
        blockers.append("full_video_mlx_scorer_replay_not_attached")
    elif not has_full:
        blockers.append("full_video_mlx_scorer_replay_not_attached")
        if any(
            record.get("full_video_scope") == "sampled_prefix_requires_full_video_rerun"
            for record in records
        ):
            blockers.append("sampled_mlx_prefilter_requires_full_video_rerun")
        blockers.extend(
            str(blocker)
            for record in records
            for blocker in record.get("blockers", [])
        )
    elif not replay_records:
        blockers.extend(
            str(blocker)
            for record in full_records
            for blocker in record.get("blockers", [])
            if blocker == "mlx_profile_batch_pairs_not_singleton"
            or _is_local_replay_quality_blocker(str(blocker))
        )
    elif score_threshold is not None and not local_replay_passed:
        blockers.append("mlx_prefilter_score_not_below_local_replay_threshold")
        if not scored_replay_records:
            blockers.append("mlx_score_missing_or_nonfinite")
        else:
            blockers.append("mlx_score_above_hard_demote_threshold")
    return {
        "schema": HPRC_MLX_PREFILTER_COVERAGE_SCHEMA,
        "required_pairs": int(required_pairs),
        "max_mlx_score_for_local_replay": score_threshold,
        "profile_count": len(records),
        "has_full_video_mlx_prefilter": has_full,
        "local_replay_mlx_prefilter_passed": local_replay_passed,
        "best_full_video_mlx_score": (
            min(float(record["mlx_score_estimate"]) for record in scored_full_records)
            if scored_full_records
            else None
        ),
        "full_video_profile_paths": [
            str(record["path"])
            for record in records
            if record.get("full_video_prefilter") is True
        ],
        "local_replay_profile_paths": [
            str(record["path"])
            for record in records
            if record.get("local_replay_prefilter") is True
        ],
        "profile_records": records,
        "blockers": _dedupe(blockers),
    }


def _profile_path_record(
    path: str | Path,
    *,
    root: Path,
    required_pairs: int,
) -> dict[str, Any]:
    resolved = _resolve(path, base=root)
    record: dict[str, Any] = {
        "path": resolved.as_posix(),
        "exists": resolved.is_file(),
        "required_pairs": int(required_pairs),
        "full_video_prefilter": False,
        "blockers": [],
    }
    if not resolved.is_file():
        record["blockers"].append("mlx_profile_missing_or_unreadable")
        return record
    try:
        payload = _load_json_object(resolved)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        record["blockers"].append(f"mlx_profile_missing_or_unreadable:{exc}")
        return record
    schema = payload.get("schema")
    pair_count = mlx_profile_pair_count(payload)
    batch_pairs = mlx_profile_batch_pairs(payload)
    full_video_scope = mlx_profile_full_video_scope(payload)
    mlx_score = mlx_profile_score_estimate(payload)
    record.update(
        {
            "bytes": resolved.stat().st_size,
            "sha256": _sha256_file(resolved),
            "schema_name": schema,
            "pair_count": pair_count,
            "batch_pairs": batch_pairs,
            "mlx_score_estimate": mlx_score,
            "full_video_scope": full_video_scope,
            "score_claim": bool(payload.get("score_claim")),
            "promotion_eligible": bool(payload.get("promotion_eligible")),
            "ready_for_exact_eval_dispatch": bool(
                payload.get("ready_for_exact_eval_dispatch")
            ),
            "full_video_prefilter": mlx_profile_has_full_video_coverage(
                payload,
                required_pairs=int(required_pairs),
            ),
        }
    )
    quality_blockers = _profile_local_replay_quality_blockers(payload)
    record["local_replay_prefilter"] = bool(
        record["full_video_prefilter"] is True
        and batch_pairs == 1
        and not quality_blockers
    )
    if schema not in {HPRC_MLX_COMPONENT_PROFILE_SCHEMA, MLX_SCORER_RESPONSE_SCHEMA}:
        record["blockers"].append("mlx_profile_schema_unsupported")
    if full_video_scope != "executed":
        record["blockers"].append("mlx_profile_not_full_video_executed")
    if pair_count is None:
        record["blockers"].append("mlx_profile_pair_count_missing")
    elif int(pair_count) < int(required_pairs):
        record["blockers"].append("mlx_profile_pair_count_below_full_video")
    if batch_pairs != 1:
        record["blockers"].append("mlx_profile_batch_pairs_not_singleton")
    record["blockers"].extend(quality_blockers)
    return record


def _profile_local_replay_quality_blockers(profile: Mapping[str, Any]) -> list[str]:
    """Return MLX profile blockers that invalidate local CPU replay triage."""

    blockers = [
        str(blocker)
        for blocker in profile.get("blockers") or ()
        if _is_local_replay_quality_blocker(str(blocker))
    ]
    diagnosis = profile.get("scorer_input_diagnosis")
    if isinstance(diagnosis, Mapping):
        blockers.extend(
            str(blocker)
            for blocker in diagnosis.get("blockers") or ()
            if _is_local_replay_quality_blocker(str(blocker))
        )
        if diagnosis.get("candidate_output_likely_saturated_or_clipped") is True:
            blockers.append(
                "mlx_renderer_prefilter_candidate_output_saturated_or_clipped"
            )
        if diagnosis.get("candidate_output_out_of_distribution") is True:
            blockers.append("mlx_renderer_prefilter_scorer_input_out_of_distribution")
    cache_gate = profile.get("cache_quality_gate")
    if isinstance(cache_gate, Mapping):
        blockers.extend(_cache_quality_gate_blockers(cache_gate))
    nested_hinerv_prefilter = profile.get("hinerv_receiver_raw_cache_prefilter")
    if isinstance(nested_hinerv_prefilter, Mapping):
        nested_gate = nested_hinerv_prefilter.get("cache_quality_gate")
        if isinstance(nested_gate, Mapping):
            blockers.extend(_cache_quality_gate_blockers(nested_gate))
        else:
            blockers.append("hinerv_receiver_raw_cache_quality_gate_missing")
    post_export_quality = profile.get("post_export_receiver_cache_quality")
    if isinstance(post_export_quality, Mapping):
        blockers.extend(_receiver_cache_quality_blockers(post_export_quality))
    metadata = profile.get("substrate_artifact_metadata")
    if isinstance(metadata, Mapping):
        nested_quality = metadata.get("post_export_receiver_cache_quality")
        if isinstance(nested_quality, Mapping):
            blockers.extend(_receiver_cache_quality_blockers(nested_quality))
        score_training = metadata.get("score_aware_training")
        if isinstance(score_training, Mapping):
            nested_quality = score_training.get("post_export_receiver_cache_quality")
            if isinstance(nested_quality, Mapping):
                blockers.extend(_receiver_cache_quality_blockers(nested_quality))
    return _dedupe(blockers)


def _cache_quality_gate_blockers(gate: Mapping[str, Any]) -> list[str]:
    blockers = [
        str(blocker)
        for blocker in gate.get("blockers") or ()
        if _is_local_replay_quality_blocker(str(blocker))
    ]
    if gate.get("fit_gate_passed") is not True:
        blockers.append("mlx_prefilter_cache_quality_gate_not_passed")
    if gate.get("candidate_cache_nondegenerate") is not True:
        blockers.append("mlx_prefilter_cache_quality_gate_degenerate_candidate_cache")
    verdict = gate.get("verdict")
    if isinstance(verdict, str) and verdict and verdict != "CACHE_INPUTS_NONDEGENERATE_LOCAL_ONLY":
        blockers.append(f"mlx_prefilter_cache_quality_verdict:{verdict}")
    return blockers


def _receiver_cache_quality_blockers(report: Mapping[str, Any]) -> list[str]:
    blockers = [
        str(blocker)
        for blocker in report.get("blockers") or ()
        if _is_local_replay_quality_blocker(str(blocker))
    ]
    if report.get("quality_gate_passed") is not True:
        blockers.append("hi_nerv_post_export_receiver_cache_quality_gate_failed")
    gate = report.get("quality_gate")
    if isinstance(gate, Mapping):
        blockers.extend(_cache_quality_gate_blockers(gate))
    verdict = report.get("quality_gate_verdict")
    if isinstance(verdict, str) and verdict and verdict != "CACHE_INPUTS_NONDEGENERATE_LOCAL_ONLY":
        blockers.append(f"mlx_prefilter_cache_quality_verdict:{verdict}")
    return blockers


def _is_local_replay_quality_blocker(blocker: str) -> bool:
    return (
        blocker.startswith("scorer_input_")
        or blocker.startswith("candidate_segnet_")
        or blocker.startswith("candidate_posenet_")
        or blocker.startswith("segnet_cache_")
        or blocker.startswith("posenet_cache_")
        or blocker.startswith("mlx_prefilter_cache_quality_")
        or blocker.startswith("mlx_prefilter_cache_quality_verdict:")
        or blocker
        in {
            "hi_nerv_archive_export_missing_for_receiver_cache_quality",
            "hi_nerv_archive_export_path_missing_for_receiver_cache_quality",
            "hi_nerv_post_export_receiver_cache_quality_gate_failed",
            "hi_nerv_receiver_cache_quality_reference_gate_not_run",
            "hi_nerv_reference_cache_missing_for_receiver_cache_quality",
            "hinerv_receiver_raw_cache_quality_gate_missing",
            "mlx_scorer_response_cache_quality_gate_failed",
            "mlx_scorer_response_candidate_cache_degenerate",
            "mlx_renderer_prefilter_candidate_output_saturated_or_clipped",
            "mlx_renderer_prefilter_scorer_input_out_of_distribution",
        }
    )


def _resolve(path: str | Path, *, base: Path) -> Path:
    raw = Path(path).expanduser()
    return raw if raw.is_absolute() else (base / raw).resolve(strict=False)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def _dedupe(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out
