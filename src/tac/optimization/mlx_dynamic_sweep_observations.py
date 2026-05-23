# SPDX-License-Identifier: MIT
"""Append-only observations for MLX dynamic sweep replanning.

The dynamic sweep planner emits candidate/config/pass work rows. This module
records the follow-up observations without granting score, rank, promotion, or
dispatch authority to local MLX/proxy evidence.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

SCHEMA = "mlx_dynamic_sweep_observations.v1"
ROW_SCHEMA = "mlx_dynamic_sweep_observation.v1"
SUMMARY_SCHEMA = "mlx_dynamic_sweep_observation_summary.v1"
TOOL = "tac.optimization.mlx_dynamic_sweep_observations"

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
}

REQUIRED_IDENTITY_FIELDS = (
    "candidate_id",
    "sweep_config_id",
    "optimization_pass_id",
    "family",
    "observed_axis",
    "evidence_tag",
)

REQUIRED_HASH_FIELDS = (
    "archive_sha256",
    "runtime_sha256",
    "raw_output_or_cache_sha256",
)

REQUIRED_COMPONENT_DELTAS = (
    "segnet_delta",
    "posenet_delta",
    "rate_delta",
)

DUPLICATE_OBSERVATION_KEY_FIELDS = (
    "candidate_id",
    "observed_axis",
    "archive_sha256",
    "raw_output_or_cache_sha256",
    "source_artifact_sha256",
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_CONTEST_AXIS_BY_GRADE = {
    "contest-CPU": "contest_cpu",
    "[contest-CPU]": "contest_cpu",
    "contest_cpu": "contest_cpu",
    "contest-CUDA": "contest_cuda",
    "[contest-CUDA]": "contest_cuda",
    "contest_cuda": "contest_cuda",
}
_CONTEST_EVIDENCE_BY_AXIS = {
    "contest_cpu": ("contest-CPU", "[contest-CPU]"),
    "contest_cuda": ("contest-CUDA", "[contest-CUDA]"),
}


class MLXDynamicSweepObservationError(ValueError):
    """Raised when a dynamic-sweep observation would blur authority boundaries."""


def json_text(payload: Any) -> str:
    """Return deterministic JSON text for CLI and report outputs."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        raise MLXDynamicSweepObservationError(f"{key} is required")
    text = str(value).strip()
    if not text:
        raise MLXDynamicSweepObservationError(f"{key} must be non-empty")
    return text


def _as_finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise MLXDynamicSweepObservationError(f"{label} must be numeric, not boolean")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MLXDynamicSweepObservationError(f"{label} must be numeric") from exc
    if not math.isfinite(out):
        raise MLXDynamicSweepObservationError(f"{label} must be finite")
    return out


def _required_sha256(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        raise MLXDynamicSweepObservationError(f"{key} is required")
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise MLXDynamicSweepObservationError(f"{key} must be a 64-character SHA-256 hex digest")
    return text


def _optional_sha256(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise MLXDynamicSweepObservationError(f"{label} must be a 64-character SHA-256 hex digest")
    return text


def _require_false_authority(row: Mapping[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if key in row and row.get(key) is not False:
            raise MLXDynamicSweepObservationError(f"{label} {key} must be false")


def _normalize_component_deltas(row: Mapping[str, Any]) -> dict[str, float]:
    raw = row.get("component_deltas", row.get("component_axis_deltas"))
    component_row: Mapping[str, Any] = raw if isinstance(raw, Mapping) else row

    deltas: dict[str, float] = {}
    for key in REQUIRED_COMPONENT_DELTAS:
        value = component_row.get(key, row.get(key))
        if value is None:
            raise MLXDynamicSweepObservationError(f"{key} is required")
        normalized = _as_finite_float(value, label=key)
        if key in row and row.get(key) is not None:
            top_level = _as_finite_float(row.get(key), label=key)
            if top_level != normalized:
                raise MLXDynamicSweepObservationError(f"{key} conflicts with component_deltas")
        deltas[key] = normalized

    for key, value in component_row.items():
        text_key = str(key)
        if text_key in deltas:
            continue
        if value is None:
            continue
        deltas[text_key] = _as_finite_float(value, label=f"component_deltas.{text_key}")
    return dict(sorted(deltas.items()))


def _source_artifact_from_payload(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    source_path = row.get("source_artifact_path")
    source_sha = row.get("source_artifact_sha256")
    source_artifact = row.get("source_artifact")
    if isinstance(source_artifact, Mapping):
        source_path = source_path or source_artifact.get("path")
        source_sha = source_sha or source_artifact.get("sha256")

    path_text = None if source_path is None else str(source_path)
    sha_text = _optional_sha256(source_sha, label="source_artifact_sha256")
    if path_text and sha_text is None:
        path = Path(path_text)
        if not path.is_file():
            raise MLXDynamicSweepObservationError(
                "source_artifact_sha256 is required when source_artifact_path is not a local file"
            )
        sha_text = file_sha256(path)
    if sha_text is not None and not path_text:
        raise MLXDynamicSweepObservationError("source_artifact_path is required with source_artifact_sha256")
    return path_text, sha_text


def _contest_axis_from_row(row: Mapping[str, Any]) -> str | None:
    axes = {
        _CONTEST_AXIS_BY_GRADE.get(str(row.get("observed_axis") or "")),
        _CONTEST_AXIS_BY_GRADE.get(str(row.get("evidence_grade") or "")),
        _CONTEST_AXIS_BY_GRADE.get(str(row.get("evidence_tag") or "")),
    }
    axes.discard(None)
    if not axes:
        return None
    if len(axes) != 1:
        raise MLXDynamicSweepObservationError(
            "contest-axis observation labels disagree across observed_axis/evidence_grade/evidence_tag"
        )
    return next(iter(axes))


def _normalize_contest_axis_evidence(normalized: dict[str, Any]) -> None:
    contest_axis = _contest_axis_from_row(normalized)
    if contest_axis is None:
        return
    evidence_grade, evidence_tag = _CONTEST_EVIDENCE_BY_AXIS[contest_axis]
    proxy_defaults = {EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX}
    for key, default_value in (
        ("evidence_grade", EVIDENCE_GRADE_MLX),
        ("evidence_tag", EVIDENCE_TAG_MLX),
    ):
        value = str(normalized.get(key) or "")
        if value == default_value or value in proxy_defaults:
            continue
        axis = _CONTEST_AXIS_BY_GRADE.get(value)
        if axis != contest_axis:
            raise MLXDynamicSweepObservationError(
                f"contest-axis observation {key} must match {contest_axis}"
            )
    normalized["observed_axis"] = contest_axis
    normalized["evidence_grade"] = evidence_grade
    normalized["evidence_tag"] = evidence_tag


def _payload_archive_sha256(payload: Mapping[str, Any]) -> str | None:
    provenance = payload.get("provenance")
    value = payload.get("archive_sha256")
    if value is None and isinstance(provenance, Mapping):
        value = provenance.get("archive_sha256")
    return _optional_sha256(value, label="source auth archive_sha256")


def _payload_inflated_aggregate_sha256(payload: Mapping[str, Any]) -> str | None:
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        manifest = provenance.get("inflated_output_manifest")
        if isinstance(manifest, Mapping):
            manifest_payload = manifest.get("payload")
            if isinstance(manifest_payload, Mapping):
                value = manifest_payload.get("aggregate_sha256")
                if value is not None:
                    return _optional_sha256(
                        value,
                        label="source auth inflated aggregate_sha256",
                    )
    value = payload.get("inflated_outputs_aggregate_sha256")
    return _optional_sha256(value, label="source auth inflated aggregate_sha256")


def _payload_auth_score(payload: Mapping[str, Any]) -> float | None:
    for key in (
        "canonical_score",
        "score_recomputed_from_components",
        "score",
        "final_score",
    ):
        value = payload.get(key)
        if value is not None:
            return _as_finite_float(value, label=f"source auth {key}")
    return None


def _validate_contest_auth_source(normalized: dict[str, Any]) -> None:
    contest_axis = _contest_axis_from_row(normalized)
    if contest_axis is None:
        return
    source_path = normalized.get("source_artifact_path")
    if not source_path:
        raise MLXDynamicSweepObservationError(
            "contest-axis observation requires local source_artifact_path"
        )
    path = Path(str(source_path))
    if not path.is_file():
        raise MLXDynamicSweepObservationError(
            "contest-axis observation source_artifact_path must be locally readable"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MLXDynamicSweepObservationError(
            "contest-axis source artifact must be auth-eval JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise MLXDynamicSweepObservationError(
            "contest-axis source artifact must be a JSON object"
        )
    payload_axis = str(payload.get("score_axis") or "")
    if payload_axis != contest_axis:
        raise MLXDynamicSweepObservationError(
            f"contest-axis source score_axis mismatch: {payload_axis!r} != {contest_axis!r}"
        )
    payload_grade = str(payload.get("evidence_grade") or "")
    if _CONTEST_AXIS_BY_GRADE.get(payload_grade) != contest_axis:
        raise MLXDynamicSweepObservationError(
            "contest-axis source evidence_grade mismatch"
        )
    if payload.get("score_claim_valid") is not True:
        raise MLXDynamicSweepObservationError(
            "contest-axis source auth eval must have score_claim_valid=true"
        )
    archive_sha = _payload_archive_sha256(payload)
    if archive_sha != normalized["archive_sha256"]:
        raise MLXDynamicSweepObservationError(
            "contest-axis source archive_sha256 mismatch"
        )
    aggregate_sha = _payload_inflated_aggregate_sha256(payload)
    if aggregate_sha != normalized["raw_output_or_cache_sha256"]:
        raise MLXDynamicSweepObservationError(
            "contest-axis source inflated aggregate SHA mismatch"
        )
    score = _payload_auth_score(payload)
    if score is not None:
        observed = float(normalized["observed_score_or_delta"])
        if abs(score - observed) > 1.0e-12:
            raise MLXDynamicSweepObservationError(
                "contest-axis source score does not match observed_score_or_delta"
            )


def build_observation_row(
    *,
    candidate_id: str,
    sweep_config_id: str,
    optimization_pass_id: str,
    family: str,
    observed_axis: str,
    evidence_tag: str,
    observed_score_or_delta: float,
    archive_sha256: str,
    runtime_sha256: str,
    raw_output_or_cache_sha256: str,
    component_deltas: Mapping[str, Any],
    source_artifact_path: str | Path | None = None,
    source_artifact_sha256: str | None = None,
    observed_at_utc: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate one observation row."""

    row: dict[str, Any] = {
        "schema": ROW_SCHEMA,
        "candidate_id": candidate_id,
        "sweep_config_id": sweep_config_id,
        "optimization_pass_id": optimization_pass_id,
        "family": family,
        "observed_axis": observed_axis,
        "evidence_tag": evidence_tag,
        "observed_score_or_delta": observed_score_or_delta,
        "archive_sha256": archive_sha256,
        "runtime_sha256": runtime_sha256,
        "raw_output_or_cache_sha256": raw_output_or_cache_sha256,
        "component_deltas": dict(component_deltas),
    }
    if observed_at_utc is not None:
        row["observed_at_utc"] = observed_at_utc
    if source_artifact_path is not None:
        row["source_artifact_path"] = str(source_artifact_path)
    if source_artifact_sha256 is not None:
        row["source_artifact_sha256"] = source_artifact_sha256
    if extra:
        row.update(dict(extra))
    return normalize_observation_row(row)


def normalize_observation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize an observation row.

    Missing authority fields are stamped as explicit ``False``. Any caller that
    tries to pass a truthy or null authority value is rejected.
    """

    if not isinstance(row, Mapping):
        raise MLXDynamicSweepObservationError("observation row must be a JSON object")
    if row.get("schema", ROW_SCHEMA) != ROW_SCHEMA:
        raise MLXDynamicSweepObservationError("observation row schema mismatch")
    _require_false_authority(row, label="observation row")

    normalized: dict[str, Any] = {
        "schema": ROW_SCHEMA,
        "producer": str(row.get("producer") or TOOL),
        "observed_at_utc": str(row.get("observed_at_utc") or _utc_now()),
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "observation_only": True,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "allowed_use": "mlx_dynamic_sweep_observation_replanning_only",
    }
    for key in REQUIRED_IDENTITY_FIELDS:
        normalized[key] = _required_text(row, key)
    normalized["evidence_grade"] = str(
        row.get("evidence_grade") or normalized["evidence_tag"] or EVIDENCE_GRADE_MLX
    )
    _normalize_contest_axis_evidence(normalized)
    normalized["observed_score_or_delta"] = _as_finite_float(
        row.get("observed_score_or_delta"),
        label="observed_score_or_delta",
    )
    for key in REQUIRED_HASH_FIELDS:
        normalized[key] = _required_sha256(row, key)

    component_deltas = _normalize_component_deltas(row)
    normalized["component_deltas"] = component_deltas
    normalized["component_axis_deltas"] = dict(component_deltas)
    for key in REQUIRED_COMPONENT_DELTAS:
        normalized[key] = component_deltas[key]

    source_path, source_sha = _source_artifact_from_payload(row)
    normalized["source_artifact_path"] = source_path
    normalized["source_artifact_sha256"] = source_sha
    normalized["source_artifact"] = (
        None
        if source_path is None
        else {
            "path": source_path,
            "sha256": source_sha,
        }
    )
    _validate_contest_auth_source(normalized)

    passthrough_keys = (
        "notes",
        "run_id",
        "selected_pair_indices",
        "selected_pair_count",
        "selector_kind",
        "acquisition_operation",
        "source_schema",
        "sweep_rank",
        "source_row",
        "planner_artifact_path",
        "planner_artifact_sha256",
        "baseline_candidate_id",
        "baseline_artifact_path",
        "baseline_artifact_sha256",
        "baseline_score",
        "baseline_archive_size_bytes",
        "score_delta_vs_baseline",
        "archive_byte_delta_vs_baseline",
        "component_delta_baseline_policy",
    )
    for key in passthrough_keys:
        if key in row and row.get(key) is not None:
            normalized[key] = row[key]
    return normalized


def append_observation_row(
    row: Mapping[str, Any],
    *,
    output_path: Path,
    allow_duplicate_observation: bool = False,
) -> dict[str, Any]:
    """Append one validated observation row to a JSONL file under an fcntl lock."""

    normalized = normalize_observation_row(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(normalized, sort_keys=True, allow_nan=False) + "\n"
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    with lock_path.open("a", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            if not allow_duplicate_observation:
                duplicate = _find_duplicate_observation(output_path, normalized)
                if duplicate is not None:
                    key = _duplicate_observation_key(normalized)
                    key_text = ", ".join(f"{field}={value!r}" for field, value in key)
                    raise MLXDynamicSweepObservationError(
                        "duplicate MLX dynamic sweep observation refused; "
                        f"{key_text}; use allow_duplicate_observation only for intentional duplicates"
                    )
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    return normalized


def _duplicate_observation_key(row: Mapping[str, Any]) -> tuple[tuple[str, str | None], ...]:
    return tuple(
        (field, None if row.get(field) is None else str(row[field]))
        for field in DUPLICATE_OBSERVATION_KEY_FIELDS
    )


def _find_duplicate_observation(
    path: Path,
    normalized: Mapping[str, Any],
) -> dict[str, Any] | None:
    target_key = _duplicate_observation_key(normalized)
    for existing in load_observation_rows(path):
        if _duplicate_observation_key(existing) == target_key:
            return existing
    return None


def load_observation_rows(path: Path) -> list[dict[str, Any]]:
    """Load and validate observation rows from JSONL."""

    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MLXDynamicSweepObservationError(f"{path}: line {line_number} is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise MLXDynamicSweepObservationError(f"{path}: line {line_number} is not a JSON object")
        rows.append(normalize_observation_row(payload))
    return rows


def _append_unique(target: list[str], values: Iterable[Any]) -> None:
    seen = set(target)
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text and text not in seen:
            target.append(text)
            seen.add(text)


def _empty_group_summary() -> dict[str, Any]:
    return {
        "row_count": 0,
        "observed_axes": [],
        "evidence_tags": [],
        "latest_observed_at_utc": None,
        "observed_score_or_delta_min": None,
        "observed_score_or_delta_max": None,
        "observed_score_or_delta_latest": None,
        "component_delta_sums": dict.fromkeys(REQUIRED_COMPONENT_DELTAS, 0.0),
        **FALSE_AUTHORITY,
    }


def _update_group_summary(summary: dict[str, Any], row: Mapping[str, Any]) -> None:
    score = float(row["observed_score_or_delta"])
    summary["row_count"] += 1
    _append_unique(summary["observed_axes"], [row.get("observed_axis")])
    _append_unique(summary["evidence_tags"], [row.get("evidence_tag")])
    for key in REQUIRED_COMPONENT_DELTAS:
        summary["component_delta_sums"][key] += float(row[key])
    if summary["observed_score_or_delta_min"] is None or score < summary["observed_score_or_delta_min"]:
        summary["observed_score_or_delta_min"] = score
    if summary["observed_score_or_delta_max"] is None or score > summary["observed_score_or_delta_max"]:
        summary["observed_score_or_delta_max"] = score
    observed_at = str(row.get("observed_at_utc") or "")
    if summary["latest_observed_at_utc"] is None or observed_at >= str(summary["latest_observed_at_utc"]):
        summary["latest_observed_at_utc"] = observed_at
        summary["observed_score_or_delta_latest"] = score


def _group_by(
    rows: Sequence[Mapping[str, Any]],
    key_fn: Callable[[Mapping[str, Any]], str],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = key_fn(row)
        groups.setdefault(key, _empty_group_summary())
        _update_group_summary(groups[key], row)
    return dict(sorted(groups.items()))


def summarize_observations(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize observations by candidate, config, pass, family, and tuple key."""

    normalized_rows = [normalize_observation_row(row) for row in rows]
    evidence_grades = sorted({str(row["evidence_grade"]) for row in normalized_rows})
    evidence_tags = sorted({str(row["evidence_tag"]) for row in normalized_rows})
    return {
        "schema": SCHEMA,
        "summary_schema": SUMMARY_SCHEMA,
        "row_schema": ROW_SCHEMA,
        "producer": TOOL,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "observation_only": True,
        "allowed_use": "mlx_dynamic_sweep_replanning_summary_only",
        "evidence_grade": (
            evidence_grades[0]
            if len(evidence_grades) == 1
            else "mixed_dynamic_sweep_observations"
        ),
        "evidence_tag": (
            evidence_tags[0] if len(evidence_tags) == 1 else "[mixed observations]"
        ),
        "evidence_grades": evidence_grades,
        "evidence_tags": evidence_tags,
        "row_count": len(normalized_rows),
        "by_candidate_id": _group_by(normalized_rows, lambda row: str(row["candidate_id"])),
        "by_sweep_config_id": _group_by(normalized_rows, lambda row: str(row["sweep_config_id"])),
        "by_optimization_pass_id": _group_by(
            normalized_rows,
            lambda row: str(row["optimization_pass_id"]),
        ),
        "by_family": _group_by(normalized_rows, lambda row: str(row["family"])),
        "by_candidate_config_pass_family": _group_by(
            normalized_rows,
            lambda row: "|".join(
                (
                    str(row["candidate_id"]),
                    str(row["sweep_config_id"]),
                    str(row["optimization_pass_id"]),
                    str(row["family"]),
                )
            ),
        ),
    }


def summarize_observation_file(path: Path) -> dict[str, Any]:
    return summarize_observations(load_observation_rows(path))


__all__ = [
    "DUPLICATE_OBSERVATION_KEY_FIELDS",
    "EVIDENCE_TAG_MLX",
    "FALSE_AUTHORITY",
    "REQUIRED_COMPONENT_DELTAS",
    "REQUIRED_HASH_FIELDS",
    "REQUIRED_IDENTITY_FIELDS",
    "ROW_SCHEMA",
    "SCHEMA",
    "SUMMARY_SCHEMA",
    "TOOL",
    "MLXDynamicSweepObservationError",
    "append_observation_row",
    "build_observation_row",
    "file_sha256",
    "json_text",
    "load_observation_rows",
    "normalize_observation_row",
    "summarize_observation_file",
    "summarize_observations",
]
