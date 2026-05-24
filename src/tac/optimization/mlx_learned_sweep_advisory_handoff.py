# SPDX-License-Identifier: MIT
"""Stamp macOS-CPU advisory artifacts onto MLX learned-sweep selections."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.mlx_dynamic_learned_sweep import FALSE_AUTHORITY
from tac.optimization.mlx_dynamic_learned_sweep_local_actuator import (
    MACOS_CPU_ADVISORY_EVIDENCE_SEMANTICS,
    MACOS_CPU_ADVISORY_PAYLOAD_SCORE_AXIS,
)
from tac.optimization.mlx_effective_spend_triage_selection import (
    ROW_SCHEMA as SELECTION_ROW_SCHEMA,
)
from tac.optimization.mlx_effective_spend_triage_selection import (
    SCHEMA as SELECTION_SCHEMA,
)
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields
from tac.repo_io import read_json, sha256_file

SCHEMA = "mlx_learned_sweep_macos_cpu_advisory_selection_handoff.v1"
PATH_MAP_SCHEMA = "mlx_learned_sweep_macos_cpu_advisory_path_map.v1"
TOOL = "tac.optimization.mlx_learned_sweep_advisory_handoff"

_CANDIDATE_PATH_KEYS = (
    "candidate_advisory_path",
    "local_cpu_advisory_source_path",
    "macos_cpu_advisory_source_path",
    "local_cpu_advisory_path",
    "macos_cpu_advisory_path",
    "advisory_eval_path",
)
_BASELINE_PATH_KEYS = (
    "baseline_advisory_path",
    "window_baseline_local_cpu_advisory_source_path",
    "window_baseline_macos_cpu_advisory_source_path",
    "window_baseline_local_cpu_advisory_path",
    "window_baseline_macos_cpu_advisory_path",
    "baseline_local_cpu_advisory_path",
    "baseline_macos_cpu_advisory_path",
    "baseline_advisory_eval_path",
)


class MLXLearnedSweepAdvisoryHandoffError(ValueError):
    """Raised when advisory artifact paths cannot be stamped safely."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for key, expected in FALSE_AUTHORITY.items():
        if payload.get(key) is not expected:
            raise MLXLearnedSweepAdvisoryHandoffError(
                f"{label}: {key} must be explicit {expected!r}"
            )
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        raise MLXLearnedSweepAdvisoryHandoffError(str(exc)) from exc


def _optional_false(payload: Mapping[str, Any], key: str, *, label: str) -> None:
    if key in payload and payload.get(key) is not False:
        raise MLXLearnedSweepAdvisoryHandoffError(f"{label}: {key} must be false")


def _require_advisory_payload(payload: Mapping[str, Any], *, label: str) -> None:
    _require_false_authority(payload, label=label)
    _optional_false(payload, "score_claim_eligible", label=label)
    if payload.get("score_axis") != MACOS_CPU_ADVISORY_PAYLOAD_SCORE_AXIS:
        raise MLXLearnedSweepAdvisoryHandoffError(
            f"{label}: score_axis must be {MACOS_CPU_ADVISORY_PAYLOAD_SCORE_AXIS}"
        )
    if payload.get("evidence_semantics") != MACOS_CPU_ADVISORY_EVIDENCE_SEMANTICS:
        raise MLXLearnedSweepAdvisoryHandoffError(
            f"{label}: evidence_semantics must be {MACOS_CPU_ADVISORY_EVIDENCE_SEMANTICS}"
        )
    _required_sha(
        payload.get("archive_sha256")
        or _optional_mapping(payload.get("provenance")).get("archive_sha256"),
        label=f"{label}: archive_sha256",
    )
    provenance = _required_mapping(payload.get("provenance"), label=f"{label}: provenance")
    runtime_manifest = _required_mapping(
        provenance.get("inflate_runtime_manifest"),
        label=f"{label}: provenance.inflate_runtime_manifest",
    )
    _required_sha(
        runtime_manifest.get("runtime_tree_sha256"),
        label=f"{label}: provenance.inflate_runtime_manifest.runtime_tree_sha256",
    )
    inflated_manifest = _required_mapping(
        provenance.get("inflated_output_manifest"),
        label=f"{label}: provenance.inflated_output_manifest",
    )
    inflated_payload = _optional_mapping(inflated_manifest.get("payload"))
    _required_sha(
        inflated_payload.get("aggregate_sha256")
        or payload.get("inflated_outputs_aggregate_sha256"),
        label=f"{label}: inflated output aggregate_sha256",
    )
    for key in (
        "score_seg_contribution",
        "score_pose_contribution",
        "score_rate_contribution",
        "archive_size_bytes",
    ):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise MLXLearnedSweepAdvisoryHandoffError(f"{label}: {key} must be numeric")
        if key == "archive_size_bytes" and value <= 0:
            raise MLXLearnedSweepAdvisoryHandoffError(
                f"{label}: archive_size_bytes must be positive"
            )


def _optional_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MLXLearnedSweepAdvisoryHandoffError(f"{label} must be an object")
    return value


def _required_sha(value: Any, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise MLXLearnedSweepAdvisoryHandoffError(f"{label} must be a sha256 hex digest")
    return text


def _resolve_existing_path(
    value: Any,
    *,
    source_artifact_root: Path,
    label: str,
) -> Path:
    text = str(value or "").strip()
    if not text:
        raise MLXLearnedSweepAdvisoryHandoffError(f"{label} is required")
    path = Path(text).expanduser()
    resolved = path if path.is_absolute() else source_artifact_root / path
    if not resolved.is_file():
        raise MLXLearnedSweepAdvisoryHandoffError(f"{label} does not exist: {resolved}")
    return resolved


def _first_path_value(row: Mapping[str, Any], keys: Sequence[str], *, label: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    raise MLXLearnedSweepAdvisoryHandoffError(
        f"{label} is required; checked keys: {', '.join(keys)}"
    )


def _normalize_path_map_rows(path_map: Any) -> list[dict[str, Any]]:
    if isinstance(path_map, list):
        rows = path_map
        baseline_default = None
    elif isinstance(path_map, Mapping):
        if path_map.get("schema") == "contest_oracle_batch_summary_v1":
            rows = path_map.get("rows")
            baseline_default = path_map.get("baseline_eval")
        else:
            rows = path_map.get("rows", path_map.get("mappings"))
            baseline_default = path_map.get("baseline_advisory_path")
    else:
        raise MLXLearnedSweepAdvisoryHandoffError("path map must be an object or list")
    if not isinstance(rows, list):
        raise MLXLearnedSweepAdvisoryHandoffError("path map rows must be a list")

    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise MLXLearnedSweepAdvisoryHandoffError(
                f"path map rows[{index}] must be an object"
            )
        normalized = dict(row)
        if baseline_default is not None and not any(
            normalized.get(key) for key in _BASELINE_PATH_KEYS
        ):
            normalized["baseline_advisory_path"] = baseline_default
        out.append(normalized)
    return out


def _mapping_id(row: Mapping[str, Any]) -> str:
    for key in ("candidate_id", "row_id", "queue_candidate_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    raise MLXLearnedSweepAdvisoryHandoffError(
        "path map row must include candidate_id, row_id, or queue_candidate_id"
    )


def _selection_match_keys(row: Mapping[str, Any]) -> list[str]:
    return [
        str(row.get(key) or "").strip()
        for key in ("candidate_id", "row_id", "queue_candidate_id")
        if str(row.get(key) or "").strip()
    ]


def _validate_selection(selection: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if selection.get("schema") != SELECTION_SCHEMA:
        raise MLXLearnedSweepAdvisoryHandoffError(
            f"selection schema must be {SELECTION_SCHEMA}"
        )
    _require_false_authority(selection, label="selection")
    rows = selection.get("selected_rows")
    if not isinstance(rows, list):
        raise MLXLearnedSweepAdvisoryHandoffError("selection selected_rows must be a list")
    out: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise MLXLearnedSweepAdvisoryHandoffError(
                f"selection selected_rows[{index}] must be an object"
            )
        if row.get("schema") != SELECTION_ROW_SCHEMA:
            raise MLXLearnedSweepAdvisoryHandoffError(
                f"selection selected_rows[{index}] schema must be {SELECTION_ROW_SCHEMA}"
            )
        _require_false_authority(row, label=f"selection selected_rows[{index}]")
        out.append(row)
    return out


def _path_record(path: Path, *, source_artifact_root: Path) -> dict[str, Any]:
    try:
        relpath = path.resolve(strict=False).relative_to(
            source_artifact_root.resolve(strict=False)
        ).as_posix()
    except ValueError:
        relpath = str(path)
    return {"path": relpath, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _validated_mapping(
    row: Mapping[str, Any],
    *,
    source_artifact_root: Path,
) -> dict[str, Any]:
    row_id = _mapping_id(row)
    candidate_path = _resolve_existing_path(
        _first_path_value(row, _CANDIDATE_PATH_KEYS, label=f"{row_id} candidate path"),
        source_artifact_root=source_artifact_root,
        label=f"{row_id} candidate path",
    )
    baseline_path = _resolve_existing_path(
        _first_path_value(row, _BASELINE_PATH_KEYS, label=f"{row_id} baseline path"),
        source_artifact_root=source_artifact_root,
        label=f"{row_id} baseline path",
    )
    candidate_payload = read_json(candidate_path)
    baseline_payload = read_json(baseline_path)
    if not isinstance(candidate_payload, Mapping):
        raise MLXLearnedSweepAdvisoryHandoffError(
            f"{candidate_path}: expected JSON object"
        )
    if not isinstance(baseline_payload, Mapping):
        raise MLXLearnedSweepAdvisoryHandoffError(
            f"{baseline_path}: expected JSON object"
        )
    _require_advisory_payload(candidate_payload, label=str(candidate_path))
    _require_advisory_payload(baseline_payload, label=str(baseline_path))
    return {
        "match_id": row_id,
        "candidate_id": str(row.get("candidate_id") or ""),
        "row_id": str(row.get("row_id") or ""),
        "queue_candidate_id": str(row.get("queue_candidate_id") or ""),
        "candidate_path": candidate_path,
        "baseline_path": baseline_path,
        "candidate_record": _path_record(
            candidate_path,
            source_artifact_root=source_artifact_root,
        ),
        "baseline_record": _path_record(
            baseline_path,
            source_artifact_root=source_artifact_root,
        ),
    }


def stamp_macos_cpu_advisory_paths(
    selection: Mapping[str, Any],
    path_map: Any,
    *,
    source_artifact_root: str | Path = ".",
    require_all_selected: bool = False,
    require_non_empty: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``selection`` with validated macOS-CPU advisory paths stamped."""

    source_root = Path(source_artifact_root)
    selected_rows = _validate_selection(selection)
    mappings = [
        _validated_mapping(row, source_artifact_root=source_root)
        for row in _normalize_path_map_rows(path_map)
    ]
    mapping_by_id: dict[str, dict[str, Any]] = {}
    for mapping in mappings:
        for key in ("candidate_id", "row_id", "queue_candidate_id", "match_id"):
            value = str(mapping.get(key) or "").strip()
            if value and value not in mapping_by_id:
                mapping_by_id[value] = mapping

    stamped_rows: list[dict[str, Any]] = []
    stamped_match_ids: list[str] = []
    missing_selection_ids: list[str] = []
    for row in selected_rows:
        row_out = dict(row)
        mapping = next(
            (
                mapping_by_id[key]
                for key in _selection_match_keys(row)
                if key in mapping_by_id
            ),
            None,
        )
        if mapping is None:
            missing_selection_ids.append(str(row.get("candidate_id") or row.get("row_id")))
        else:
            row_out["local_cpu_advisory_source_path"] = mapping["candidate_record"]["path"]
            row_out["local_cpu_advisory_source_sha256"] = mapping["candidate_record"][
                "sha256"
            ]
            row_out["window_baseline_local_cpu_advisory_source_path"] = mapping[
                "baseline_record"
            ]["path"]
            row_out["window_baseline_local_cpu_advisory_source_sha256"] = mapping[
                "baseline_record"
            ]["sha256"]
            stamped_match_ids.append(mapping["match_id"])
        stamped_rows.append(row_out)

    stamped_set = set(stamped_match_ids)
    unused_mapping_ids = [
        mapping["match_id"] for mapping in mappings if mapping["match_id"] not in stamped_set
    ]
    if require_non_empty and not stamped_match_ids:
        raise MLXLearnedSweepAdvisoryHandoffError(
            "no selection rows matched validated macOS-CPU advisory path mappings"
        )
    if require_all_selected and missing_selection_ids:
        raise MLXLearnedSweepAdvisoryHandoffError(
            "missing macOS-CPU advisory path mappings for selected rows: "
            + ", ".join(sorted(missing_selection_ids))
        )

    output_selection = dict(selection)
    output_selection["selected_rows"] = stamped_rows
    report = {
        "schema": SCHEMA,
        "tool": TOOL,
        "generated_at_utc": _utc_now(),
        **FALSE_AUTHORITY,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "selection_schema": selection.get("schema"),
        "source_artifact_root": str(source_root),
        "selected_row_count": len(selected_rows),
        "mapping_row_count": len(mappings),
        "stamped_row_count": len(stamped_match_ids),
        "missing_selection_ids": sorted(missing_selection_ids),
        "unused_mapping_ids": sorted(unused_mapping_ids),
        "require_all_selected": bool(require_all_selected),
        "ready_for_macos_cpu_advisory_queue": bool(stamped_match_ids),
        "authority_boundary": {
            **FALSE_AUTHORITY,
            "dispatch_attempted": False,
            "gpu_launched": False,
            "allowed_use": "stamp_existing_macos_cpu_advisory_artifacts_only",
        },
        "stamped_rows": [
            {
                "match_id": mapping["match_id"],
                "candidate_id": mapping.get("candidate_id"),
                "row_id": mapping.get("row_id"),
                "queue_candidate_id": mapping.get("queue_candidate_id"),
                "candidate_advisory": mapping["candidate_record"],
                "baseline_advisory": mapping["baseline_record"],
            }
            for mapping in mappings
            if mapping["match_id"] in stamped_set
        ],
    }
    output_selection["macos_cpu_advisory_handoff"] = report
    _require_false_authority(output_selection, label="output selection")
    _require_false_authority(report, label="handoff report")
    return output_selection, report


__all__ = [
    "PATH_MAP_SCHEMA",
    "SCHEMA",
    "TOOL",
    "MLXLearnedSweepAdvisoryHandoffError",
    "stamp_macos_cpu_advisory_paths",
]
