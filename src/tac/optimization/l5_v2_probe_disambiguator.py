# SPDX-License-Identifier: MIT
"""Planning-only C1/Z5/TT5L probe-disambiguator for the L5 v2 staircase.

This module does not run training and does not claim score movement. It is the
typed arbitration surface that consumes measured probe observations once they
exist. Until an observation carries paired exact-axis custody and byte-consumed
side-info evidence, the corresponding candidate remains visible but ineligible
for architecture lock-in.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tac.exact_eval_custody import is_sha256_hex, validate_exact_eval_evidence

L5V2_PROBE_SCHEMA = "tac_l5_v2_probe_disambiguator_v1"
L5V2_PROBE_TOOL_PATH = "tools/probe_l5_v2_staircase_disambiguator.py"
L5V2_CANDIDATES: tuple[str, ...] = (
    "c1_world_model_foveation",
    "z5_predictive_coding_world_model",
    "time_traveler_l5_autonomy",
)
ContestAxis = Literal["contest_cpu", "contest_cuda"]
REQUIRED_EXACT_AXES: tuple[ContestAxis, ...] = ("contest_cpu", "contest_cuda")


@dataclass(frozen=True)
class L5V2ProbeObservation:
    """One measured candidate observation for the L5 v2 disambiguator."""

    candidate_id: str
    predicted_or_measured_delta: float
    evidence_grade: str
    exact_axes: tuple[str, ...] = ()
    artifact_path: str = ""
    artifact_sha256: str = ""
    predicate_id: str = ""
    predicate_passed: bool = False
    archive_sha256: str = ""
    runtime_tree_sha256: str = ""
    axis_evidence: tuple[Mapping[str, Any], ...] = ()
    sideinfo_consumed: bool = False
    byte_closed_archive: bool = False
    notes: str = ""


def _as_tuple_str(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Iterable):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item)
        return tuple(out)
    return ()


def _as_tuple_mapping(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _require_literal_json_bool(value: object, *, field_name: str) -> bool:
    if value is True:
        return True
    if value is False:
        return False
    raise ValueError(f"{field_name} must be a literal JSON boolean")


def observation_from_mapping(payload: Mapping[str, Any]) -> L5V2ProbeObservation:
    """Parse a JSON-style observation mapping."""

    delta_obj = payload.get("predicted_or_measured_delta", 0.0)
    delta = float(delta_obj) if isinstance(delta_obj, int | float) else math.nan
    return L5V2ProbeObservation(
        candidate_id=str(payload.get("candidate_id") or ""),
        predicted_or_measured_delta=delta,
        evidence_grade=str(payload.get("evidence_grade") or ""),
        exact_axes=_as_tuple_str(payload.get("exact_axes")),
        artifact_path=str(payload.get("artifact_path") or ""),
        artifact_sha256=str(payload.get("artifact_sha256") or ""),
        predicate_id=str(payload.get("predicate_id") or ""),
        predicate_passed=_require_literal_json_bool(
            payload.get("predicate_passed", False),
            field_name="predicate_passed",
        ),
        archive_sha256=str(payload.get("archive_sha256") or ""),
        runtime_tree_sha256=str(payload.get("runtime_tree_sha256") or ""),
        axis_evidence=_as_tuple_mapping(payload.get("axis_evidence")),
        sideinfo_consumed=_require_literal_json_bool(
            payload.get("sideinfo_consumed", False),
            field_name="sideinfo_consumed",
        ),
        byte_closed_archive=_require_literal_json_bool(
            payload.get("byte_closed_archive", False),
            field_name="byte_closed_archive",
        ),
        notes=str(payload.get("notes") or ""),
    )


def load_observations_json(path: Path) -> tuple[L5V2ProbeObservation, ...]:
    """Load observations from a JSON file.

    Accepted shapes:
    - a list of observation objects
    - an object with ``observations`` as a list
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows_obj = payload.get("observations") if isinstance(payload, Mapping) else payload
    if not isinstance(rows_obj, list):
        raise ValueError("L5 v2 probe input must be a list or {'observations': [...]}")
    return tuple(
        observation_from_mapping(row)
        for row in rows_obj
        if isinstance(row, Mapping)
    )


def _missing_required_axes(observation: L5V2ProbeObservation) -> tuple[str, ...]:
    axes = set(observation.exact_axes)
    return tuple(axis for axis in REQUIRED_EXACT_AXES if axis not in axes)


def _is_transient_artifact_path(path: str) -> bool:
    normalized = path.removeprefix("file:").strip()
    return normalized.startswith(("/tmp/", "/private/tmp/", "/var/tmp/"))


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_artifact_path(path: str, repo_root: Path) -> tuple[Path | None, str | None]:
    normalized = path.removeprefix("file:").strip()
    if not normalized:
        return None, None
    artifact_path = Path(normalized).expanduser()
    resolved = (
        artifact_path.resolve()
        if artifact_path.is_absolute()
        else (repo_root / artifact_path).resolve()
    )
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        return resolved, "outside_repo"
    return resolved, None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _axis_evidence_path_blockers(
    evidence: Mapping[str, Any],
    *,
    axis: str,
    repo_root: Path,
) -> tuple[str, ...]:
    blockers: list[str] = []
    log_path = str(evidence.get("log_path") or "")
    if not log_path.strip():
        return ()
    if _is_transient_artifact_path(log_path):
        blockers.append(f"l5_v2_probe_axis_log_path_transient:{axis}")
        return tuple(blockers)
    _resolved, path_error = _resolve_artifact_path(log_path, repo_root)
    if path_error == "outside_repo":
        blockers.append(f"l5_v2_probe_axis_log_path_outside_repo:{axis}")
    return tuple(blockers)


def _axis_evidence_blockers(
    observation: L5V2ProbeObservation,
    *,
    repo_root: Path,
) -> tuple[str, ...]:
    blockers: list[str] = []
    by_axis: dict[str, Mapping[str, Any]] = {}
    for item in observation.axis_evidence:
        axis = str(item.get("axis") or "")
        if axis.strip():
            by_axis[axis] = item

    for axis in REQUIRED_EXACT_AXES:
        evidence = by_axis.get(axis)
        if evidence is None:
            blockers.append(f"l5_v2_probe_axis_evidence_missing:{axis}")
            continue

        blockers.extend(
            _axis_evidence_path_blockers(
                evidence,
                axis=axis,
                repo_root=repo_root,
            )
        )
        validation = validate_exact_eval_evidence(
            evidence,
            expected_axis=axis,
            expected_archive_sha256=observation.archive_sha256,
            expected_runtime_tree_sha256=observation.runtime_tree_sha256,
            require_hardware=True,
            require_auth_eval_command=True,
            require_log_path=True,
            require_devices=True,
            artifact_base_dir=repo_root,
        )
        blocker_map = {
            "archive_sha_invalid": "l5_v2_probe_axis_archive_sha_invalid",
            "archive_sha_mismatch": "l5_v2_probe_axis_archive_sha_mismatch",
            "runtime_tree_sha_invalid": "l5_v2_probe_axis_runtime_tree_sha_invalid",
            "runtime_tree_sha_mismatch": "l5_v2_probe_axis_runtime_tree_sha_mismatch",
            "n_samples_missing": "l5_v2_probe_axis_n_samples_missing",
            "n_samples_not_contest_exact": "l5_v2_probe_axis_n_samples_not_contest_exact",
            "archive_bytes_missing": "l5_v2_probe_axis_archive_bytes_missing",
            "seg_dist_missing": "l5_v2_probe_axis_seg_dist_missing",
            "pose_dist_missing": "l5_v2_probe_axis_pose_dist_missing",
            "score_missing": "l5_v2_probe_axis_score_missing",
            "hardware_missing": "l5_v2_probe_axis_hardware_missing",
            "hardware_not_cuda": "l5_v2_probe_axis_hardware_not_cuda",
            "inflate_device_missing": "l5_v2_probe_axis_inflate_device_missing",
            "inflate_device_not_cpu": "l5_v2_probe_axis_inflate_device_not_cpu",
            "inflate_device_not_cuda": "l5_v2_probe_axis_inflate_device_not_cuda",
            "eval_device_missing": "l5_v2_probe_axis_eval_device_missing",
            "eval_device_not_cpu": "l5_v2_probe_axis_eval_device_not_cpu",
            "eval_device_not_cuda": "l5_v2_probe_axis_eval_device_not_cuda",
            "auth_eval_command_missing": "l5_v2_probe_axis_auth_eval_command_missing",
            "log_path_missing": "l5_v2_probe_axis_log_path_missing",
            "log_path_file_missing": "l5_v2_probe_axis_log_path_file_missing",
            "artifact_path_file_missing": "l5_v2_probe_axis_artifact_path_file_missing",
            "score_formula_mismatch": "l5_v2_probe_axis_score_formula_mismatch",
        }
        for blocker in validation.blockers:
            public_blocker = blocker_map.get(blocker)
            if public_blocker is not None:
                blockers.append(f"{public_blocker}:{axis}")
    return tuple(blockers)


def _observation_blockers(
    observation: L5V2ProbeObservation,
    *,
    repo_root: Path,
) -> tuple[str, ...]:
    blockers: list[str] = []
    resolved_artifact_path: Path | None = None
    path_error: str | None = None
    if observation.candidate_id not in L5V2_CANDIDATES:
        blockers.append("l5_v2_probe_unknown_candidate")
    if not math.isfinite(observation.predicted_or_measured_delta):
        blockers.append("l5_v2_probe_delta_non_finite")
    if not observation.artifact_path.strip():
        blockers.append("l5_v2_probe_artifact_path_missing")
    elif _is_transient_artifact_path(observation.artifact_path):
        blockers.append("l5_v2_probe_artifact_path_transient")
    else:
        resolved_artifact_path, path_error = _resolve_artifact_path(
            observation.artifact_path,
            repo_root,
        )
        if path_error == "outside_repo":
            blockers.append("l5_v2_probe_artifact_path_outside_repo")
        elif resolved_artifact_path is not None and not resolved_artifact_path.is_file():
            blockers.append("l5_v2_probe_artifact_file_missing")
    if not is_sha256_hex(observation.artifact_sha256):
        blockers.append("l5_v2_probe_artifact_sha_invalid")
    elif (
        resolved_artifact_path is not None
        and path_error is None
        and resolved_artifact_path.is_file()
        and _sha256_file(resolved_artifact_path)
        != observation.artifact_sha256.strip().lower()
    ):
        blockers.append("l5_v2_probe_artifact_sha_mismatch")
    if not observation.predicate_id.strip():
        blockers.append("l5_v2_probe_predicate_id_missing")
    if observation.predicate_passed is not True:
        blockers.append("l5_v2_probe_predicate_failed")
    if _missing_required_axes(observation):
        blockers.append("l5_v2_probe_paired_exact_axes_missing")
    if observation.byte_closed_archive is not True:
        blockers.append("l5_v2_probe_byte_closed_archive_missing")
    if observation.sideinfo_consumed is not True:
        blockers.append("l5_v2_probe_sideinfo_consumption_missing")
    if not is_sha256_hex(observation.archive_sha256):
        blockers.append("l5_v2_probe_archive_sha_invalid")
    if not is_sha256_hex(observation.runtime_tree_sha256):
        blockers.append("l5_v2_probe_runtime_tree_sha_invalid")
    if "contest" not in observation.evidence_grade.lower():
        blockers.append("l5_v2_probe_contest_evidence_grade_missing")
    blockers.extend(_axis_evidence_blockers(observation, repo_root=repo_root))
    return tuple(dict.fromkeys(blockers))


def build_probe_template() -> dict[str, Any]:
    """Return an auditable empty input template for future measured probes."""

    return {
        "schema": L5V2_PROBE_SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "observations": [
            {
                "candidate_id": candidate_id,
                "predicted_or_measured_delta": 0.0,
                "evidence_grade": "contest_cpu_and_cuda_required",
                "exact_axes": list(REQUIRED_EXACT_AXES),
                "artifact_path": "",
                "artifact_sha256": "",
                "predicate_id": "",
                "predicate_passed": False,
                "archive_sha256": "",
                "runtime_tree_sha256": "",
                "axis_evidence": [
                    {
                        "axis": axis,
                        "archive_sha256": "",
                        "runtime_tree_sha256": "",
                        "score": None,
                        "seg_dist": None,
                        "pose_dist": None,
                        "archive_bytes": None,
                        "n_samples": None,
                        "hardware": "",
                        "inflate_device": "",
                        "eval_device": "",
                        "auth_eval_command": "",
                        "log_path": "",
                    }
                    for axis in REQUIRED_EXACT_AXES
                ],
                "sideinfo_consumed": False,
                "byte_closed_archive": False,
                "notes": "fill from paired exact probe artifacts",
            }
            for candidate_id in L5V2_CANDIDATES
        ],
    }


def evaluate_l5_v2_probe(
    observations: Iterable[L5V2ProbeObservation],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate observations and return a fail-closed architecture verdict."""

    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    rows = tuple(observations)
    evaluated: list[dict[str, Any]] = []
    eligible: list[L5V2ProbeObservation] = []
    global_blockers: list[str] = []
    eligible_candidate_ids: set[str] = set()

    if not rows:
        global_blockers.append("l5_v2_probe_observations_missing")

    seen = {row.candidate_id for row in rows}
    missing_candidates = [
        candidate_id for candidate_id in L5V2_CANDIDATES if candidate_id not in seen
    ]
    if missing_candidates:
        global_blockers.append("l5_v2_probe_candidate_coverage_incomplete")

    for row in rows:
        blockers = _observation_blockers(row, repo_root=resolved_repo_root)
        row_dict = dataclasses.asdict(row)
        row_dict["exact_axes"] = list(row.exact_axes)
        row_dict["eligible_for_architecture_lock"] = not blockers
        row_dict["blockers"] = list(blockers)
        row_dict["missing_exact_axes"] = list(_missing_required_axes(row))
        evaluated.append(row_dict)
        if not blockers:
            eligible.append(row)
            eligible_candidate_ids.add(row.candidate_id)

    ineligible_required_candidates = [
        candidate_id
        for candidate_id in L5V2_CANDIDATES
        if candidate_id in seen and candidate_id not in eligible_candidate_ids
    ]
    for candidate_id in ineligible_required_candidates:
        global_blockers.append(f"l5_v2_probe_required_candidate_ineligible:{candidate_id}")

    selected = min(
        eligible,
        key=lambda row: row.predicted_or_measured_delta,
        default=None,
    )
    if selected is None:
        global_blockers.append("l5_v2_probe_no_eligible_candidate")

    return {
        "schema": L5V2_PROBE_SCHEMA,
        "tool": L5V2_PROBE_TOOL_PATH,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "architecture_lock_allowed": selected is not None and not global_blockers,
        "selected_candidate_id": selected.candidate_id if selected is not None else None,
        "selected_delta": (
            selected.predicted_or_measured_delta if selected is not None else None
        ),
        "required_candidates": list(L5V2_CANDIDATES),
        "required_exact_axes": list(REQUIRED_EXACT_AXES),
        "evaluated_observations": evaluated,
        "blockers": list(dict.fromkeys(global_blockers)),
        "evidence_semantics": "planning_only_l5_v2_probe_disambiguator",
    }


__all__ = [
    "L5V2_CANDIDATES",
    "L5V2_PROBE_SCHEMA",
    "L5V2_PROBE_TOOL_PATH",
    "REQUIRED_EXACT_AXES",
    "L5V2ProbeObservation",
    "build_probe_template",
    "evaluate_l5_v2_probe",
    "load_observations_json",
    "observation_from_mapping",
]
