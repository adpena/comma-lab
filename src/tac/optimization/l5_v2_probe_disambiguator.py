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
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
    archive_sha256: str = ""
    runtime_tree_sha256: str = ""
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


def observation_from_mapping(payload: Mapping[str, Any]) -> L5V2ProbeObservation:
    """Parse a JSON-style observation mapping."""

    delta_obj = payload.get("predicted_or_measured_delta", 0.0)
    delta = float(delta_obj) if isinstance(delta_obj, int | float) else math.nan
    return L5V2ProbeObservation(
        candidate_id=str(payload.get("candidate_id") or ""),
        predicted_or_measured_delta=delta,
        evidence_grade=str(payload.get("evidence_grade") or ""),
        exact_axes=_as_tuple_str(payload.get("exact_axes")),
        archive_sha256=str(payload.get("archive_sha256") or ""),
        runtime_tree_sha256=str(payload.get("runtime_tree_sha256") or ""),
        sideinfo_consumed=bool(payload.get("sideinfo_consumed", False)),
        byte_closed_archive=bool(payload.get("byte_closed_archive", False)),
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


def _observation_blockers(observation: L5V2ProbeObservation) -> tuple[str, ...]:
    blockers: list[str] = []
    if observation.candidate_id not in L5V2_CANDIDATES:
        blockers.append("l5_v2_probe_unknown_candidate")
    if not math.isfinite(observation.predicted_or_measured_delta):
        blockers.append("l5_v2_probe_delta_non_finite")
    if _missing_required_axes(observation):
        blockers.append("l5_v2_probe_paired_exact_axes_missing")
    if not observation.byte_closed_archive:
        blockers.append("l5_v2_probe_byte_closed_archive_missing")
    if not observation.sideinfo_consumed:
        blockers.append("l5_v2_probe_sideinfo_consumption_missing")
    if not observation.archive_sha256.strip():
        blockers.append("l5_v2_probe_archive_sha_missing")
    if not observation.runtime_tree_sha256.strip():
        blockers.append("l5_v2_probe_runtime_tree_sha_missing")
    if "contest" not in observation.evidence_grade.lower():
        blockers.append("l5_v2_probe_contest_evidence_grade_missing")
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
                "archive_sha256": "",
                "runtime_tree_sha256": "",
                "sideinfo_consumed": False,
                "byte_closed_archive": False,
                "notes": "fill from paired exact probe artifacts",
            }
            for candidate_id in L5V2_CANDIDATES
        ],
    }


def evaluate_l5_v2_probe(
    observations: Iterable[L5V2ProbeObservation],
) -> dict[str, Any]:
    """Evaluate observations and return a fail-closed architecture verdict."""

    rows = tuple(observations)
    evaluated: list[dict[str, Any]] = []
    eligible: list[L5V2ProbeObservation] = []
    global_blockers: list[str] = []

    if not rows:
        global_blockers.append("l5_v2_probe_observations_missing")

    seen = {row.candidate_id for row in rows}
    missing_candidates = [
        candidate_id for candidate_id in L5V2_CANDIDATES if candidate_id not in seen
    ]
    if missing_candidates:
        global_blockers.append("l5_v2_probe_candidate_coverage_incomplete")

    for row in rows:
        blockers = _observation_blockers(row)
        row_dict = dataclasses.asdict(row)
        row_dict["exact_axes"] = list(row.exact_axes)
        row_dict["eligible_for_architecture_lock"] = not blockers
        row_dict["blockers"] = list(blockers)
        row_dict["missing_exact_axes"] = list(_missing_required_axes(row))
        evaluated.append(row_dict)
        if not blockers:
            eligible.append(row)

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
