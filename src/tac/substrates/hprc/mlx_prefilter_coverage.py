# SPDX-License-Identifier: MIT
"""Coverage classification for HPRC MLX scorer-response prefilters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tac.substrates.hprc.resolution_contract import CONTEST_PAIR_COUNT

HPRC_MLX_COMPONENT_PROFILE_SCHEMA = "hprc_mlx_component_neutralization_profile.v1"
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
    """Return whether a profile is eligible as full-video MLX prefilter evidence."""

    if profile.get("schema") != HPRC_MLX_COMPONENT_PROFILE_SCHEMA:
        return False
    count = mlx_profile_pair_count(profile)
    batch_pairs = mlx_profile_batch_pairs(profile)
    return (
        mlx_profile_full_video_scope(profile) == "executed"
        and count is not None
        and int(count) >= int(required_pairs)
        and batch_pairs == 1
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
    scored_full_records = [
        record
        for record in full_records
        if _finite_float(record.get("mlx_score_estimate")) is not None
    ]
    local_replay_passed = bool(full_records)
    if score_threshold is not None:
        local_replay_passed = any(
            float(record["mlx_score_estimate"]) < score_threshold
            for record in scored_full_records
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
    elif score_threshold is not None and not local_replay_passed:
        blockers.append("mlx_prefilter_score_not_below_local_replay_threshold")
        if not scored_full_records:
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
    if schema != HPRC_MLX_COMPONENT_PROFILE_SCHEMA:
        record["blockers"].append("mlx_profile_schema_unsupported")
    if full_video_scope != "executed":
        record["blockers"].append("mlx_profile_not_full_video_executed")
    if pair_count is None:
        record["blockers"].append("mlx_profile_pair_count_missing")
    elif int(pair_count) < int(required_pairs):
        record["blockers"].append("mlx_profile_pair_count_below_full_video")
    if batch_pairs != 1:
        record["blockers"].append("mlx_profile_batch_pairs_not_singleton")
    return record


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
