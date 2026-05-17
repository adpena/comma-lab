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
import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.exact_eval_custody import (
    extract_archive_sha256,
    extract_expected_runtime_tree_sha256,
    extract_observed_runtime_tree_sha256,
    finite_float,
    normalize_sha256,
    positive_int,
    validate_exact_eval_evidence,
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
L5V2_TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_PATH = (
    ".omx/research/l5_v2_tt5l_materialized_paired_work_unit_plan_20260516_codex.json"
)
DEFAULT_L5V2_PROBE_SOURCE_PATHS: tuple[str, ...] = (
    "experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/modal_auth_eval_cpu/l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_cpu/contest_auth_eval.json",
    "experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/modal_auth_eval/l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_cuda/contest_auth_eval.json",
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


def _dedupe_paths(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(path for path in paths if str(path).strip()))


def materialized_tt5l_probe_source_paths(
    *,
    repo_root: str | Path | None = None,
    plan_path: str | Path = L5V2_TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_PATH,
) -> tuple[str, ...]:
    """Return TT5L source JSON paths from the materialized paired work-unit plan.

    The materialized TT5L plan may be variant/archive-specific. Reading its
    output directories prevents the probe intake from silently reusing stale
    source-archive eval paths after a new side-info variant is selected.
    """

    root = Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    candidate = Path(plan_path)
    resolved = candidate if candidate.is_absolute() else root / candidate
    payload = _read_json_object(resolved)
    if payload is None:
        return ()
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping):
        return ()
    out: list[str] = []
    for axis in ("contest_cpu", "contest_cuda"):
        output_dir = str(outputs.get(axis) or "").strip()
        if not output_dir:
            continue
        out.append(str(Path(output_dir) / "contest_auth_eval.json"))
    return _dedupe_paths(out)


def default_l5_v2_probe_source_paths(
    *,
    repo_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Return dynamic materialized TT5L paths before legacy probe sources."""

    return _dedupe_paths(
        (
            *materialized_tt5l_probe_source_paths(repo_root=repo_root),
            *DEFAULT_L5V2_PROBE_SOURCE_PATHS,
        )
    )


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


def _command_flag_value(command: str, flag: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    for idx, part in enumerate(parts[:-1]):
        if part == flag:
            return str(parts[idx + 1]).strip()
    prefix = f"{flag}="
    for part in parts:
        if part.startswith(prefix):
            return part[len(prefix) :].strip()
    return ""


def _first_existing_relative_path(paths: Iterable[Path], repo_root: Path) -> str:
    for path in paths:
        if path.is_file():
            return _relative_path(path, repo_root)
    return ""


def _local_auth_eval_log_path(
    path: Path,
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> str:
    """Resolve the durable local auth-eval log next to recovered artifacts.

    Modal recovery stores stdout/stderr logs beside the canonical
    ``contest_auth_eval.json`` result, while higher-level review ledgers point
    back to that JSON through ``source_json_path``. Prefer committed or
    workspace-relative logs; never promote the remote ``/root/...`` report path
    as local custody.
    """

    direct = str(payload.get("log_path") or "").strip()
    if direct:
        return direct

    candidate_dirs: list[Path] = [path.parent]
    source_json_path = str(payload.get("source_json_path") or "").strip()
    if source_json_path:
        source_path = Path(source_json_path)
        resolved_source = (
            source_path if source_path.is_absolute() else repo_root / source_path
        )
        candidate_dirs.append(resolved_source.parent)

    candidates: list[Path] = []
    for directory in candidate_dirs:
        candidates.extend(
            (
                directory / "contest_auth_eval.stdout.log",
                directory / "contest_auth_eval.stderr.log",
            )
        )
    return _first_existing_relative_path(candidates, repo_root)


def _local_inflated_outputs_manifest_path(
    path: Path,
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> str:
    direct = str(
        payload.get("inflated_outputs_manifest_path")
        or payload.get("inflated_output_manifest_path")
        or ""
    ).strip()
    if direct:
        return direct

    candidate_dirs: list[Path] = [path.parent]
    source_json_path = str(payload.get("source_json_path") or "").strip()
    if source_json_path:
        source_path = Path(source_json_path)
        resolved_source = (
            source_path if source_path.is_absolute() else repo_root / source_path
        )
        candidate_dirs.append(resolved_source.parent)

    return _first_existing_relative_path(
        (directory / "inflated_outputs_manifest.json" for directory in candidate_dirs),
        repo_root,
    )


def _sha256_for_relative_path(path_text: str, repo_root: Path) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.is_file():
        return ""
    return _sha256_file(resolved)


def _manifest_sha_for_payload(
    payload: Mapping[str, Any],
    *,
    manifest_path: str,
    repo_root: Path,
) -> str:
    for value in (
        payload.get("inflated_outputs_manifest_sha256"),
        payload.get("inflated_output_manifest_sha256"),
        _nested(payload, "runtime_custody", "inflated_output_manifest_sha256"),
        _nested(payload, "provenance", "inflated_output_manifest", "sha256"),
    ):
        normalized = normalize_sha256(value)
        if normalized:
            return normalized
    return _sha256_for_relative_path(manifest_path, repo_root)


def _raw_output_aggregate_sha_for_payload(payload: Mapping[str, Any]) -> str:
    for value in (
        payload.get("raw_output_aggregate_sha256"),
        payload.get("inflated_output_aggregate_sha256"),
        _nested(payload, "runtime_custody", "inflated_output_aggregate_sha256"),
        _nested(
            payload,
            "provenance",
            "inflated_output_manifest",
            "payload",
            "aggregate_sha256",
        ),
    ):
        normalized = normalize_sha256(value)
        if normalized:
            return normalized
    return ""


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
    direct = extract_observed_runtime_tree_sha256(payload)
    if direct:
        return direct
    for nested_key in ("runtime_custody", "custody", "provenance"):
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping):
            candidate = extract_observed_runtime_tree_sha256(nested)
            if candidate:
                return candidate
    return ""


def _expected_runtime_sha_for_payload(payload: Mapping[str, Any]) -> str:
    direct = extract_expected_runtime_tree_sha256(payload)
    if direct:
        return direct
    for nested_key in ("runtime_custody", "custody", "provenance"):
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping):
            candidate = extract_expected_runtime_tree_sha256(nested)
            if candidate:
                return candidate
    return ""


def _hardware_for_payload(payload: Mapping[str, Any]) -> str:
    def _clean_hardware_text(value: object) -> str:
        text = str(value or "").strip()
        if text.startswith("<error:") or text.startswith("<missing:"):
            return ""
        return text

    direct = str(
        _clean_hardware_text(_nested(payload, "custody", "gpu_model"))
        or _clean_hardware_text(_nested(payload, "custody", "hardware"))
        or _clean_hardware_text(_nested(payload, "provenance", "gpu_model"))
        or _clean_hardware_text(_nested(payload, "provenance", "hardware"))
        or _clean_hardware_text(payload.get("hardware"))
    ).strip()
    direct_is_probe_error = "filenotfounderror" in direct.lower() or direct.startswith(
        "<error:"
    )
    if direct and not direct_is_probe_error:
        return direct

    platform_system = str(
        payload.get("provenance_platform_system")
        or _nested(payload, "provenance", "platform_system")
        or ""
    ).strip()
    platform_machine = str(
        payload.get("provenance_platform_machine")
        or _nested(payload, "provenance", "platform_machine")
        or ""
    ).strip()
    if platform_system and platform_machine:
        return f"{platform_system} {platform_machine}"
    return ""


def _eval_device_for_payload(payload: Mapping[str, Any], command: str) -> str:
    return str(
        _nested(payload, "custody", "device")
        or payload.get("eval_device")
        or payload.get("provenance_device")
        or _nested(payload, "provenance", "device")
        or _command_flag_value(command, "--device")
        or ""
    ).strip()


def _inflate_device_for_payload(
    payload: Mapping[str, Any],
    command: str,
    *,
    axis: str,
    eval_device: str,
) -> str:
    explicit = str(
        _nested(payload, "custody", "inflate_device")
        or payload.get("inflate_device")
        or _command_flag_value(command, "--inflate-device")
        or ""
    ).strip()
    if explicit:
        return explicit

    policy = str(
        payload.get("inflate_device_policy")
        or _nested(payload, "provenance", "inflate_device_policy")
        or ""
    ).strip()
    if axis == "contest_cpu" and eval_device.lower() == "cpu":
        return "cpu"
    return policy


def _axis_evidence_from_payload(
    path: Path,
    payload: Mapping[str, Any],
    *,
    axis: str,
    repo_root: Path,
) -> dict[str, Any]:
    command = _command_text(
        _nested(payload, "custody", "command")
        or _nested(payload, "provenance", "sys_argv")
        or payload.get("sys_argv")
        or _nested(payload, "provenance", "command")
        or _nested(payload, "provenance", "tool")
        or payload.get("auth_eval_command")
    )
    artifact_path = _relative_path(path, repo_root)
    inflated_manifest_path = _local_inflated_outputs_manifest_path(
        path,
        payload,
        repo_root=repo_root,
    )
    eval_device = _eval_device_for_payload(payload, command)
    inflate_device = _inflate_device_for_payload(
        payload,
        command,
        axis=axis,
        eval_device=eval_device,
    )
    return {
        "axis": axis,
        "archive_sha256": _archive_sha_for_payload(payload),
        "runtime_tree_sha256": _runtime_sha_for_payload(payload),
        "expected_runtime_tree_sha256": _expected_runtime_sha_for_payload(payload),
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
        "hardware": _hardware_for_payload(payload),
        "inflate_device": inflate_device,
        "eval_device": eval_device,
        "auth_eval_command": command,
        "log_path": _local_auth_eval_log_path(path, payload, repo_root=repo_root),
        "artifact_path": artifact_path,
        "inflated_outputs_manifest_path": inflated_manifest_path,
        "inflated_outputs_manifest_sha256": _manifest_sha_for_payload(
            payload,
            manifest_path=inflated_manifest_path,
            repo_root=repo_root,
        ),
        "raw_output_aggregate_sha256": _raw_output_aggregate_sha_for_payload(payload),
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
        "inflated_outputs_manifest_path",
        "inflated_outputs_manifest_sha256",
        "raw_output_aggregate_sha256",
        "score_delta",
    )
    return sum(1 for field in fields if evidence.get(field) not in (None, ""))


def _axis_custody_blockers(
    evidence: Mapping[str, Any],
    *,
    axis: str,
    repo_root: Path,
) -> list[str]:
    validation = validate_exact_eval_evidence(
        evidence,
        expected_axis=axis,
        expected_archive_sha256=evidence.get("archive_sha256"),
        expected_runtime_tree_sha256=evidence.get("expected_runtime_tree_sha256"),
        require_artifact_path=True,
        require_hardware=True,
        require_auth_eval_command=True,
        require_log_path=True,
        require_devices=True,
        require_inflated_outputs_manifest=True,
        require_raw_output_aggregate_sha256=True,
        artifact_base_dir=repo_root,
    )
    return list(validation.blockers)


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
    source_paths: Iterable[str | Path] | None = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Scan known artifacts and emit a fail-closed L5 v2 probe intake report."""

    root = Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    resolved_source_paths = (
        tuple(source_paths)
        if source_paths is not None
        else tuple(default_l5_v2_probe_source_paths(repo_root=root))
    )
    source_records: list[dict[str, Any]] = []
    grouped_axis_evidence: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        candidate_id: {} for candidate_id in L5V2_CANDIDATES
    }
    grouped_source_records_by_archive: dict[str, dict[str, list[dict[str, Any]]]] = {
        candidate_id: {} for candidate_id in L5V2_CANDIDATES
    }
    grouped_source_records: dict[str, list[dict[str, Any]]] = {
        candidate_id: [] for candidate_id in L5V2_CANDIDATES
    }

    for raw_path in resolved_source_paths:
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
        axis_archive_key = ""
        if payload is not None and candidate_id is not None and axis is not None:
            axis_evidence = _axis_evidence_from_payload(
                resolved,
                payload,
                axis=axis,
                repo_root=root,
            )
            axis_archive_key = (
                normalize_sha256(axis_evidence.get("archive_sha256"))
                or f"missing_archive_sha:{_relative_path(resolved, root)}"
            )
            archive_group = grouped_axis_evidence[candidate_id].setdefault(
                axis_archive_key,
                {},
            )
            existing = archive_group.get(axis)
            if existing is None or _axis_evidence_quality(axis_evidence) > _axis_evidence_quality(existing):
                archive_group[axis] = axis_evidence
        custody_blockers = (
            _axis_custody_blockers(axis_evidence, axis=axis, repo_root=root)
            if axis_evidence is not None and axis is not None
            else []
        )
        recognized_for_observation = not blockers
        source_record = {
            "path": _relative_path(resolved, root),
            "exists": True,
            "sha256": _sha256_file(resolved),
            "candidate_id": candidate_id,
            "axis": axis,
            "axis_evidence": axis_evidence,
            "recognized_for_observation": recognized_for_observation,
            "custody_valid_for_observation": (
                recognized_for_observation and not custody_blockers
            ),
            "accepted_for_observation": (
                recognized_for_observation and not custody_blockers
            ),
            "custody_blockers": custody_blockers,
            "blockers": blockers,
        }
        source_records.append(source_record)
        if candidate_id in grouped_source_records:
            grouped_source_records[candidate_id].append(source_record)
        if candidate_id in grouped_source_records_by_archive and axis_archive_key:
            grouped_source_records_by_archive[candidate_id].setdefault(
                axis_archive_key,
                [],
            ).append(source_record)

    observations: list[L5V2ProbeObservation] = []
    for candidate_id in L5V2_CANDIDATES:
        archive_groups = grouped_axis_evidence[candidate_id]
        if archive_groups:
            selected_archive_key, selected_axis_map = max(
                archive_groups.items(),
                key=lambda item: (
                    len(item[1]),
                    sum(_axis_evidence_quality(row) for row in item[1].values()),
                    item[0],
                ),
            )
        else:
            selected_archive_key = ""
            selected_axis_map = {}
        axis_rows = list(selected_axis_map.values())
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
        observed_runtime_by_axis = {
            axis: digest
            for axis, digest in (
                (axis, normalize_sha256(value))
                for axis, value in runtime_by_axis.items()
            )
            if digest
        }
        evidence_grade = (
            "contest_axis_artifact_intake_not_architecture_lock_evidence"
            if axes
            else "artifact_intake_not_architecture_lock_evidence"
        )
        selected_sources = grouped_source_records_by_archive[candidate_id].get(
            selected_archive_key,
            grouped_source_records[candidate_id],
        )
        first_source = selected_sources[0]
        observations.append(
            L5V2ProbeObservation(
                candidate_id=candidate_id,
                predicted_or_measured_delta=0.0,
                evidence_grade=evidence_grade,
                exact_axes=axes,
                artifact_path=str(first_source.get("path") or ""),
                artifact_sha256=str(first_source.get("sha256") or ""),
                predicate_id="l5_v2_probe_observation_intake_v1",
                predicate_passed=False,
                archive_sha256=archive_sha,
                runtime_tree_sha256_by_axis=observed_runtime_by_axis,
                axis_evidence=tuple(axis_rows),
                sideinfo_consumed=False,
                byte_closed_archive=bool(
                    archive_sha and len(observed_runtime_by_axis) == len(axes)
                ),
                notes=(
                    "auto-intake only; axes are grouped by exact archive SHA so "
                    "stale and newly materialized TT5L runs cannot be mixed; "
                    "fill missing paired axis, score_delta, payload-consumption "
                    "predicate, and candidate-specific custody before architecture lock"
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
            f"recognized=`{str(source.get('recognized_for_observation')).lower()}`, "
            f"custody_valid=`{str(source.get('custody_valid_for_observation')).lower()}`, "
            f"accepted=`{str(source.get('accepted_for_observation')).lower()}`, "
            f"blockers=`{blocker_text}`"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_L5V2_PROBE_SOURCE_PATHS",
    "L5V2_PROBE_OBSERVATION_INTAKE_SCHEMA",
    "L5V2_PROBE_OBSERVATION_INTAKE_TOOL_PATH",
    "L5V2_TT5L_MATERIALIZED_PAIRED_WORK_UNIT_PLAN_PATH",
    "build_l5_v2_probe_observation_intake",
    "default_l5_v2_probe_source_paths",
    "materialized_tt5l_probe_source_paths",
    "render_l5_v2_probe_observation_intake_markdown",
]
