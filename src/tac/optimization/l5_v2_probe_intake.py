# SPDX-License-Identifier: MIT
"""Artifact intake for the L5 v2 C1/Z5/TT5L probe gate.

This module is intentionally conservative. It scans already-materialized exact
eval/review artifacts and translates any plausible C1/Z5/TT5L evidence into the
typed probe-observation format. The resulting gate artifact remains fail-closed
unless all candidates carry paired contest CPU/CUDA custody and the
probe-disambiguator independently approves architecture lock-in.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.exact_eval_custody import (
    extract_archive_sha256,
    extract_runtime_tree_sha256,
    finite_float,
    normalize_sha256,
    positive_int,
)
from tac.optimization.l5_v2_probe_disambiguator import (
    L5V2_CANDIDATES,
    L5V2ProbeObservation,
    build_l5_v2_probe_gate_artifact,
)

L5V2_PROBE_OBSERVATION_INTAKE_SCHEMA = "l5_v2_probe_observation_intake_v1"
L5V2_PROBE_OBSERVATION_INTAKE_TOOL_PATH = (
    "tools/audit_l5_v2_probe_observations.py"
)
DEFAULT_L5V2_PROBE_SOURCE_PATHS: tuple[str, ...] = (
    ".omx/research/time_traveler_recovered_tt5l_25ep_exact_cuda_evidence_row_20260515_codex.json",
    ".omx/research/time_traveler_recovered_tt5l_25ep_exact_cuda_result_review_20260515_codex.json",
    "experiments/results/modal_auth_eval/time_traveler_recovered_tt5l_25ep_exact_cuda_20260514T105300Z/contest_auth_eval.json",
    "experiments/results/c1_probe_v2_realvideo_20260514T174815Z/probe_v2_realvideo.json",
    "experiments/results/c1_probe_v2_realvideo_20260514T174815Z/probe_v2_realvideo_higher_capacity.json",
    "reports/raw/c1_wm_probe_20260514T161531Z.json",
    "reports/raw/c1_fov_probe_20260514T161537Z.json",
)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _first_finite_float(mapping: Mapping[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value: Any = mapping
        for part in key.split("."):
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(part)
        out = finite_float(value)
        if out is not None:
            return out
    return None


def _first_positive_int(mapping: Mapping[str, Any], keys: Iterable[str]) -> int | None:
    for key in keys:
        value: Any = mapping
        for part in key.split("."):
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(part)
        out = positive_int(value)
        if out is not None:
            return out
    return None


def _command_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value or "")


def _candidate_id_for_payload(path: Path, payload: Mapping[str, Any]) -> str | None:
    haystack = " ".join(
        str(item or "").lower()
        for item in (
            path.as_posix(),
            payload.get("candidate_id"),
            payload.get("lane_id"),
            payload.get("technique"),
            payload.get("job_name"),
            payload.get("source"),
            payload.get("tool"),
        )
    )
    if "tt5l" in haystack or "time_traveler" in haystack or "time-traveler" in haystack:
        return "time_traveler_l5_autonomy"
    if "z5" in haystack or "predictive_coding_world_model" in haystack:
        return "z5_predictive_coding_world_model"
    if "c1" in haystack or "world_model_foveation" in haystack or "foveation" in haystack:
        return "c1_world_model_foveation"
    return None


def _axis_for_payload(payload: Mapping[str, Any]) -> str | None:
    axis = str(payload.get("score_axis") or "").strip().lower()
    if axis in {"contest_cpu", "contest_cuda"}:
        return axis
    if payload.get("exact_cuda_evidence") is True:
        return "contest_cuda"
    if payload.get("exact_cpu_evidence") is True:
        return "contest_cpu"
    device = str(_nested(payload, "custody", "device") or "").strip().lower()
    if device == "cuda":
        return "contest_cuda"
    if device == "cpu":
        return "contest_cpu"
    return None


def _archive_sha_for_payload(payload: Mapping[str, Any]) -> str:
    direct = extract_archive_sha256(payload)
    if direct:
        return direct
    for nested_key in ("custody", "score_recomputation", "provenance"):
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping):
            candidate = extract_archive_sha256(nested)
            if candidate:
                return candidate
    return ""


def _runtime_sha_for_payload(payload: Mapping[str, Any]) -> str:
    direct = extract_runtime_tree_sha256(payload)
    if direct:
        return direct
    for nested_key in ("runtime_custody", "custody", "provenance"):
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping):
            candidate = extract_runtime_tree_sha256(nested)
            if candidate:
                return candidate
    return ""


def _axis_evidence_from_payload(
    path: Path,
    payload: Mapping[str, Any],
    *,
    axis: str,
    repo_root: Path,
) -> dict[str, Any]:
    command = _command_text(
        _nested(payload, "custody", "command")
        or _nested(payload, "provenance", "command")
        or _nested(payload, "provenance", "tool")
        or payload.get("auth_eval_command")
    )
    artifact_path = _relative_path(path, repo_root)
    return {
        "axis": axis,
        "archive_sha256": _archive_sha_for_payload(payload),
        "runtime_tree_sha256": _runtime_sha_for_payload(payload),
        "score": _first_finite_float(
            payload,
            (
                "score_recomputation.recomputed_score",
                "canonical_score",
                "empirical_score",
                "score_recomputed_from_components",
                "final_score",
            ),
        ),
        "seg_dist": _first_finite_float(
            payload,
            (
                "score_recomputation.avg_segnet_dist",
                "segnet_distortion",
                "avg_segnet_dist",
            ),
        ),
        "pose_dist": _first_finite_float(
            payload,
            (
                "score_recomputation.avg_posenet_dist",
                "posenet_distortion",
                "avg_posenet_dist",
            ),
        ),
        "archive_bytes": _first_positive_int(
            payload,
            (
                "score_recomputation.archive_bytes",
                "custody.archive_bytes",
                "empirical_archive_bytes",
                "archive_size_bytes",
            ),
        ),
        "n_samples": _first_positive_int(payload, ("custody.n_samples", "n_samples")),
        "hardware": str(
            _nested(payload, "custody", "gpu_model")
            or _nested(payload, "custody", "hardware")
            or payload.get("hardware")
            or ""
        ),
        "inflate_device": str(
            _nested(payload, "custody", "inflate_device")
            or payload.get("inflate_device")
            or ""
        ),
        "eval_device": str(
            _nested(payload, "custody", "device")
            or payload.get("eval_device")
            or ""
        ),
        "auth_eval_command": command,
        "log_path": str(payload.get("log_path") or ""),
        "artifact_path": artifact_path,
        "score_delta": _first_finite_float(
            payload,
            ("score_delta", "component_deltas.score_delta"),
        ),
    }


def _axis_evidence_quality(evidence: Mapping[str, Any]) -> int:
    """Return a small completeness score for choosing one row per axis."""

    fields = (
        "archive_sha256",
        "runtime_tree_sha256",
        "score",
        "seg_dist",
        "pose_dist",
        "archive_bytes",
        "n_samples",
        "hardware",
        "inflate_device",
        "eval_device",
        "auth_eval_command",
        "log_path",
        "artifact_path",
        "score_delta",
    )
    return sum(1 for field in fields if evidence.get(field) not in (None, ""))


def _source_blockers(
    *,
    candidate_id: str | None,
    axis: str | None,
    payload: Mapping[str, Any] | None,
) -> list[str]:
    blockers: list[str] = []
    if payload is None:
        blockers.append("source_json_not_object_or_unreadable")
    if candidate_id is None:
        blockers.append("source_candidate_unrecognized")
    if axis is None:
        blockers.append("source_exact_axis_missing_or_unrecognized")
    return blockers


def _empty_observation(candidate_id: str, notes: str) -> L5V2ProbeObservation:
    return L5V2ProbeObservation(
        candidate_id=candidate_id,
        predicted_or_measured_delta=0.0,
        evidence_grade="contest_cpu_and_cuda_required",
        exact_axes=(),
        predicate_id="l5_v2_probe_observation_intake_v1",
        predicate_passed=False,
        sideinfo_consumed=False,
        byte_closed_archive=False,
        notes=notes,
    )


def build_l5_v2_probe_observation_intake(
    source_paths: Iterable[str | Path] = DEFAULT_L5V2_PROBE_SOURCE_PATHS,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Scan known artifacts and emit a fail-closed L5 v2 probe intake report."""

    root = Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    source_records: list[dict[str, Any]] = []
    grouped_axis_evidence: dict[str, dict[str, dict[str, Any]]] = {
        candidate_id: {} for candidate_id in L5V2_CANDIDATES
    }
    grouped_source_records: dict[str, list[dict[str, Any]]] = {
        candidate_id: [] for candidate_id in L5V2_CANDIDATES
    }

    for raw_path in source_paths:
        path = Path(raw_path)
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file():
            source_records.append(
                {
                    "path": str(raw_path),
                    "exists": False,
                    "accepted_for_observation": False,
                    "blockers": ["source_file_missing"],
                }
            )
            continue
        payload = _read_json_object(resolved)
        candidate_id = _candidate_id_for_payload(resolved, payload or {})
        axis = _axis_for_payload(payload or {})
        blockers = _source_blockers(
            candidate_id=candidate_id,
            axis=axis,
            payload=payload,
        )
        axis_evidence: dict[str, Any] | None = None
        if payload is not None and candidate_id is not None and axis is not None:
            axis_evidence = _axis_evidence_from_payload(
                resolved,
                payload,
                axis=axis,
                repo_root=root,
            )
            existing = grouped_axis_evidence[candidate_id].get(axis)
            if existing is None or _axis_evidence_quality(axis_evidence) > _axis_evidence_quality(existing):
                grouped_axis_evidence[candidate_id][axis] = axis_evidence
        source_record = {
            "path": _relative_path(resolved, root),
            "exists": True,
            "sha256": _sha256_file(resolved),
            "candidate_id": candidate_id,
            "axis": axis,
            "axis_evidence": axis_evidence,
            "accepted_for_observation": not blockers,
            "blockers": blockers,
        }
        source_records.append(source_record)
        if candidate_id in grouped_source_records:
            grouped_source_records[candidate_id].append(source_record)

    observations: list[L5V2ProbeObservation] = []
    for candidate_id in L5V2_CANDIDATES:
        axis_rows = list(grouped_axis_evidence[candidate_id].values())
        if not axis_rows:
            observations.append(
                _empty_observation(
                    candidate_id,
                    "no exact CPU/CUDA source artifact found by L5 v2 intake",
                )
            )
            continue
        axes = tuple(sorted({str(row.get("axis") or "") for row in axis_rows if row.get("axis")}))
        archive_sha = normalize_sha256(axis_rows[0].get("archive_sha256"))
        runtime_by_axis = {
            str(row.get("axis")): str(row.get("runtime_tree_sha256") or "")
            for row in axis_rows
            if row.get("axis")
        }
        first_source = grouped_source_records[candidate_id][0]
        observations.append(
            L5V2ProbeObservation(
                candidate_id=candidate_id,
                predicted_or_measured_delta=0.0,
                evidence_grade="artifact_intake_not_architecture_lock_evidence",
                exact_axes=axes,
                artifact_path=str(first_source.get("path") or ""),
                artifact_sha256=str(first_source.get("sha256") or ""),
                predicate_id="l5_v2_probe_observation_intake_v1",
                predicate_passed=False,
                archive_sha256=archive_sha,
                runtime_tree_sha256_by_axis=runtime_by_axis,
                axis_evidence=tuple(axis_rows),
                sideinfo_consumed=False,
                byte_closed_archive=bool(archive_sha and runtime_by_axis),
                notes=(
                    "auto-intake only; fill missing paired axis, score_delta, "
                    "payload-consumption predicate, and candidate-specific custody "
                    "before architecture lock"
                ),
            )
        )

    gate_artifact = build_l5_v2_probe_gate_artifact(observations, repo_root=root)
    verdict = gate_artifact["probe_disambiguator"]["verdict"]
    return {
        "schema": L5V2_PROBE_OBSERVATION_INTAKE_SCHEMA,
        "tool_path": L5V2_PROBE_OBSERVATION_INTAKE_TOOL_PATH,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "source_records": source_records,
        "observations": [
            {
                **dataclasses.asdict(row),
                "exact_axes": list(row.exact_axes),
                "axis_evidence": [dict(item) for item in row.axis_evidence],
                "runtime_tree_sha256_by_axis": dict(row.runtime_tree_sha256_by_axis),
            }
            for row in observations
        ],
        "probe_gate_artifact": gate_artifact,
        "verdict": verdict,
        "architecture_lock_allowed": verdict.get("architecture_lock_allowed") is True,
        "blockers": list(verdict.get("blockers") or ()),
    }


def render_l5_v2_probe_observation_intake_markdown(intake: Mapping[str, Any]) -> str:
    """Render a compact operator-facing intake report."""

    verdict = intake.get("verdict")
    verdict_map = verdict if isinstance(verdict, Mapping) else {}
    lines = [
        "# L5 v2 Probe Observation Intake",
        "",
        f"- schema: `{intake.get('schema')}`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        f"- architecture_lock_allowed: `{str(intake.get('architecture_lock_allowed')).lower()}`",
        f"- selected_candidate_id: `{verdict_map.get('selected_candidate_id')}`",
        "",
        "## Candidate Status",
    ]
    for row in verdict_map.get("evaluated_observations") or []:
        if not isinstance(row, Mapping):
            continue
        blockers = row.get("blockers")
        blocker_text = ", ".join(str(item) for item in blockers) if isinstance(blockers, list) else ""
        lines.append(
            f"- `{row.get('candidate_id')}`: eligible=`{str(row.get('eligible_for_architecture_lock')).lower()}`, "
            f"axes=`{','.join(str(axis) for axis in row.get('exact_axes', []))}`, blockers=`{blocker_text}`"
        )
    lines.extend(["", "## Source Artifacts"])
    for source in intake.get("source_records") or []:
        if not isinstance(source, Mapping):
            continue
        blockers = source.get("blockers")
        blocker_text = ", ".join(str(item) for item in blockers) if isinstance(blockers, list) else ""
        lines.append(
            f"- `{source.get('path')}`: exists=`{str(source.get('exists')).lower()}`, "
            f"candidate=`{source.get('candidate_id')}`, axis=`{source.get('axis')}`, "
            f"accepted=`{str(source.get('accepted_for_observation')).lower()}`, blockers=`{blocker_text}`"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_L5V2_PROBE_SOURCE_PATHS",
    "L5V2_PROBE_OBSERVATION_INTAKE_SCHEMA",
    "L5V2_PROBE_OBSERVATION_INTAKE_TOOL_PATH",
    "build_l5_v2_probe_observation_intake",
    "render_l5_v2_probe_observation_intake_markdown",
]
