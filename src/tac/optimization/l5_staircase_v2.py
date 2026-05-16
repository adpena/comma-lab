# SPDX-License-Identifier: MIT
"""Typed L5 v2 staircase plan and fail-closed custody surface.

This module is deliberately planning-only. It turns the Time-Traveler L5 v2
staircase into machine-readable steps and gates so Cathedral/autopilot can see
the frontier path without treating source-backed theory as score evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tac.optimization.l5_v2_probe_disambiguator import (
    L5V2_CANDIDATES,
    L5V2_PROBE_SCHEMA,
    L5V2_PROBE_TOOL_PATH,
)
from tac.optimization.prediction_band import (
    BandSource,
    BaselineRef,
    EmpiricalAnchorRef,
    PredictionBand,
    SupersessionRef,
    UncertaintyRef,
    prediction_band_to_dict,
    validate_prediction_band,
    verdict_to_dict,
)
from tac.optimization.research_basis import research_basis_ids_for_family

SUBJECT_ID = "time_traveler_l5_autonomy"
LANE_ID = "lane_time_traveler_l5_autonomy_substrate_20260513"
CAMPAIGN_ID = "campaign_time_traveler_l5_v2_staircase_20260516"
PREDICTED_DELTA_BAND = (-0.0500, -0.0200)
PREDICTED_DELTA_AXIS = "mixed"

GateStatus = Literal["required", "satisfied", "blocked"]
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_REQUIRED_EXACT_AXES = ("contest_cpu", "contest_cuda")


@dataclass(frozen=True)
class L5V2Gate:
    """One non-negotiable gate before L5 v2 can dispatch or promote."""

    gate_id: str
    status: GateStatus
    description: str
    blocker: str
    evidence_path: str = ""


@dataclass(frozen=True)
class L5V2GateEvidence:
    """Artifact-backed proof that one L5 v2 gate actually passed."""

    gate_id: str
    artifact_path: str
    artifact_sha256: str
    predicate_id: str
    predicate_passed: bool
    evidence_grade: str = ""


@dataclass(frozen=True)
class L5V2Step:
    """One ordered staircase step for the L5 v2 campaign."""

    step_id: str
    title: str
    objective: str
    deliverable_surface: str
    required_gate_ids: tuple[str, ...]
    research_basis_ids: tuple[str, ...]
    dispatch_allowed: bool = False
    promotion_eligible: bool = False


def l5_v2_research_basis_ids() -> tuple[str, ...]:
    """Return the source stack that anchors L5 v2 planning claims."""

    return tuple(research_basis_ids_for_family("time_traveler_l5_v2"))


def l5_v2_required_gates() -> tuple[L5V2Gate, ...]:
    """Return the fail-closed gate set for the current L5 v2 staircase."""

    return (
        L5V2Gate(
            gate_id="byte_closed_temporal_sideinfo_consumption",
            status="required",
            description=(
                "Every temporal side-info byte must be emitted, parsed, and "
                "proven to change inflate output under byte mutation."
            ),
            blocker="requires_byte_closed_temporal_sideinfo_consumption_proof",
        ),
        L5V2Gate(
            gate_id="c1_z5_tt5l_probe_disambiguator",
            status="required",
            description=(
                "C1, Z5, and TT5L interpretations must be probed under one "
                "callable disambiguator before architecture lock."
            ),
            blocker="requires_c1_z5_tt5l_probe_disambiguator_before_architecture_lock",
        ),
        L5V2Gate(
            gate_id="paired_cpu_cuda_axis_plan",
            status="required",
            description=(
                "CPU and CUDA axes need a paired plan with archive SHA, "
                "runtime tree SHA, inflate device, eval device, and component deltas."
            ),
            blocker="requires_paired_cpu_cuda_axis_plan_before_promotion",
        ),
        L5V2Gate(
            gate_id="exact_anchor_or_diagnostic_pair",
            status="required",
            description=(
                "At least one paired exact or explicitly diagnostic anchor must "
                "exist before predicted bands can influence rank reward."
            ),
            blocker="requires_l5_v2_empirical_anchor",
        ),
    )


def l5_v2_staircase_steps() -> tuple[L5V2Step, ...]:
    """Return the ordered L5 v2 campaign steps."""

    basis = l5_v2_research_basis_ids()
    return (
        L5V2Step(
            step_id="l5v2_00_source_and_alias_custody",
            title="Source and alias custody",
            objective=(
                "Bind L5 v2 paper/theory claims to canonical research basis ids "
                "and legacy aliases before dispatch planning."
            ),
            deliverable_surface="src/tac/optimization/research_basis.py",
            required_gate_ids=(),
            research_basis_ids=basis,
        ),
        L5V2Step(
            step_id="l5v2_01_sideinfo_consumption_proof",
            title="Temporal side-info consumption proof",
            objective=(
                "Add byte-mutation and section-consumption proofs for the TT5L "
                "temporal side-info stream."
            ),
            deliverable_surface=(
                "src/tac/substrates/time_traveler_l5_autonomy/archive.py + tests"
            ),
            required_gate_ids=("byte_closed_temporal_sideinfo_consumption",),
            research_basis_ids=basis,
        ),
        L5V2Step(
            step_id="l5v2_02_probe_disambiguator",
            title="C1/Z5/TT5L probe disambiguator",
            objective=(
                "Ship both defensible interpretations and let paired probes "
                "choose architecture direction before full training."
            ),
            deliverable_surface=L5V2_PROBE_TOOL_PATH,
            required_gate_ids=("c1_z5_tt5l_probe_disambiguator",),
            research_basis_ids=basis,
        ),
        L5V2Step(
            step_id="l5v2_03_paired_axis_anchor",
            title="Paired CPU/CUDA anchor",
            objective=(
                "Measure the exact same byte-closed packet on contest-compliant "
                "CPU and CUDA axes before any promotion or submission discussion."
            ),
            deliverable_surface="experiments/results/time_traveler_l5_v2/",
            required_gate_ids=(
                "paired_cpu_cuda_axis_plan",
                "exact_anchor_or_diagnostic_pair",
            ),
            research_basis_ids=basis,
        ),
        L5V2Step(
            step_id="l5v2_04_stack_of_stacks_candidate",
            title="Stack-of-stacks composition candidate",
            objective=(
                "Compose the first proved L5 v2 packet with the best byte-closed "
                "orthogonal winner only after component anchors exist."
            ),
            deliverable_surface="src/tac/optimization/substrate_composition_matrix.py",
            required_gate_ids=(
                "byte_closed_temporal_sideinfo_consumption",
                "exact_anchor_or_diagnostic_pair",
            ),
            research_basis_ids=basis,
        ),
    )


def l5_v2_prediction_band_payload() -> dict[str, Any]:
    """Return the L5 v2 prediction band payload for inventory/autopilot rows.

    The payload carries real source provenance but intentionally lacks baseline
    custody and empirical anchor status. Validators therefore keep it visible
    for dispatch planning while blocking rank reward and promotion.
    """

    band = PredictionBand(
        band_id="time_traveler_l5_v2_delta_prior_20260516",
        subject_id=SUBJECT_ID,
        band_kind="delta_score",
        low=PREDICTED_DELTA_BAND[0],
        high=PREDICTED_DELTA_BAND[1],
        axis=PREDICTED_DELTA_AXIS,
        baseline=BaselineRef(
            label="axis_matched_frontier_baseline_pending",
            axis=PREDICTED_DELTA_AXIS,
            score=None,
            archive_sha256="",
            runtime_tree_sha256="",
            artifact_path="",
        ),
        band_source=BandSource(
            local_ledger_paths=(
                "file:.omx/research/time_traveler_architecture_reverse_engineered_20260513.md",
                "file:.omx/research/campaign_lane_c2_z7_mature_predictive_receiver_l5_20260514.md",
            ),
            research_basis_ids=l5_v2_research_basis_ids(),
            claim_scope=(
                "planning prior for L5 v2 staircase only; not score evidence "
                "until byte-closed paired anchors exist"
            ),
        ),
        uncertainty=UncertaintyRef(
            method="source-backed-prior-with-no-l5-v2-anchor",
            confidence_tag="pre_anchor",
            n_empirical_anchors=0,
            notes="Band remains rank-blocked until L5 v2 empirical anchors land.",
        ),
        supersession=SupersessionRef(status="active"),
        empirical_anchor=EmpiricalAnchorRef(status="pending", anchors=()),
        planning_only=True,
        score_claim=False,
    )
    return prediction_band_to_dict(band)


def _require_literal_json_bool(value: object, *, field_name: str) -> bool:
    if value is True:
        return True
    if value is False:
        return False
    raise ValueError(f"{field_name} must be a literal JSON boolean")


def l5_v2_prediction_band_verdict() -> dict[str, Any]:
    """Return the current validation verdict for the L5 v2 prediction band."""

    payload = l5_v2_prediction_band_payload()
    band = PredictionBand(
        band_id=str(payload["band_id"]),
        subject_id=str(payload["subject_id"]),
        band_kind="delta_score",
        low=float(payload["low"]),
        high=float(payload["high"]),
        axis=str(payload["axis"]),
        baseline=BaselineRef(**payload["baseline"]),
        band_source=BandSource(
            local_ledger_paths=tuple(payload["band_source"]["local_ledger_paths"]),
            research_basis_ids=tuple(payload["band_source"]["research_basis_ids"]),
            claim_scope=str(payload["band_source"]["claim_scope"]),
        ),
        uncertainty=UncertaintyRef(**payload["uncertainty"]),
        supersession=SupersessionRef(**payload["supersession"]),
        empirical_anchor=EmpiricalAnchorRef(**payload["empirical_anchor"]),
        planning_only=_require_literal_json_bool(
            payload["planning_only"],
            field_name="planning_only",
        ),
        score_claim=_require_literal_json_bool(
            payload["score_claim"],
            field_name="score_claim",
        ),
    )
    return verdict_to_dict(
        validate_prediction_band(
            band,
            expected_subject_id=SUBJECT_ID,
            expected_low=PREDICTED_DELTA_BAND[0],
            expected_high=PREDICTED_DELTA_BAND[1],
        )
    )


def _coerce_gate_evidence(
    gate_evidence: (
        Mapping[str, L5V2GateEvidence | Mapping[str, Any]]
        | Iterable[L5V2GateEvidence | Mapping[str, Any]]
        | None
    ),
) -> dict[str, L5V2GateEvidence]:
    if gate_evidence is None:
        return {}

    values: Iterable[L5V2GateEvidence | Mapping[str, Any]]
    if isinstance(gate_evidence, Mapping):
        values = (
            value if isinstance(value, L5V2GateEvidence) else {"gate_id": key, **value}
            for key, value in gate_evidence.items()
        )
    else:
        values = gate_evidence

    coerced: dict[str, L5V2GateEvidence] = {}
    for value in values:
        if isinstance(value, L5V2GateEvidence):
            evidence = value
        else:
            evidence = L5V2GateEvidence(
                gate_id=str(value.get("gate_id", "")),
                artifact_path=str(value.get("artifact_path", "")),
                artifact_sha256=str(value.get("artifact_sha256", "")),
                predicate_id=str(value.get("predicate_id", "")),
                predicate_passed=_require_literal_json_bool(
                    value.get("predicate_passed", False),
                    field_name="predicate_passed",
                ),
                evidence_grade=str(value.get("evidence_grade", "")),
            )
        if evidence.gate_id:
            coerced[evidence.gate_id] = evidence
    return coerced


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


def _finite_json_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _string_items(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if not isinstance(value, Iterable) or isinstance(value, bytes):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def _mapping_items(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _axis_row_map(value: object) -> dict[str, Mapping[str, Any]]:
    rows = _mapping_items(value)
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        axis = str(row.get("axis") or "").strip()
        if axis in _REQUIRED_EXACT_AXES:
            out[axis] = row
    return out


def _paired_row_identity_blockers(
    *,
    gate_id: str,
    rows: dict[str, Mapping[str, Any]],
    section: str,
    require_anchor_type: bool = False,
) -> list[str]:
    blockers: list[str] = []
    missing_axes = [axis for axis in _REQUIRED_EXACT_AXES if axis not in rows]
    if missing_axes:
        blockers.append(
            f"l5_v2_gate_artifact_semantics_missing:{gate_id}:{section}:"
            + ",".join(missing_axes)
        )
        return blockers

    archive_shas = {
        str(row.get("archive_sha256") or "").strip().lower() for row in rows.values()
    }
    runtime_shas = {
        str(row.get("runtime_tree_sha256") or "").strip().lower()
        for row in rows.values()
    }
    if len(archive_shas) != 1 or not _SHA256_HEX_RE.fullmatch(next(iter(archive_shas))):
        blockers.append(
            f"l5_v2_gate_artifact_semantics_invalid:{gate_id}:{section}:archive_sha256"
        )
    if len(runtime_shas) != 1 or not _SHA256_HEX_RE.fullmatch(next(iter(runtime_shas))):
        blockers.append(
            "l5_v2_gate_artifact_semantics_invalid:"
            f"{gate_id}:{section}:runtime_tree_sha256"
        )

    for axis, row in rows.items():
        inflate_device = str(row.get("inflate_device") or "").lower()
        eval_device = str(row.get("eval_device") or "").lower()
        if axis == "contest_cpu":
            if "cpu" not in inflate_device:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:{section}:contest_cpu_inflate_device"
                )
            if "cpu" not in eval_device:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:{section}:contest_cpu_eval_device"
                )
        else:
            if "cuda" not in inflate_device and "gpu" not in inflate_device:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:{section}:contest_cuda_inflate_device"
                )
            if "cuda" not in eval_device and "gpu" not in eval_device:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:{section}:contest_cuda_eval_device"
                )
        deltas = row.get("component_deltas")
        if not isinstance(deltas, Mapping):
            blockers.append(
                "l5_v2_gate_artifact_semantics_missing:"
                f"{gate_id}:{section}:{axis}:component_deltas"
            )
        else:
            for field in ("seg_dist_delta", "pose_dist_delta", "score_delta"):
                if not _finite_json_number(deltas.get(field)):
                    blockers.append(
                        "l5_v2_gate_artifact_semantics_invalid:"
                        f"{gate_id}:{section}:{axis}:{field}"
                    )
        anchor_type = str(row.get("anchor_type") or "").strip()
        if require_anchor_type and anchor_type not in {"exact", "diagnostic"}:
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:{section}:{axis}:anchor_type"
            )
        elif require_anchor_type:
            if row.get("score_claim") is not False:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:{section}:{axis}:score_claim"
                )
            evidence_grade = str(row.get("evidence_grade") or "").strip().lower()
            if anchor_type == "exact" and "contest" not in evidence_grade:
                blockers.append(
                    "l5_v2_gate_artifact_semantics_invalid:"
                    f"{gate_id}:{section}:{axis}:evidence_grade"
                )
            if anchor_type == "diagnostic":
                if "diagnostic" not in evidence_grade:
                    blockers.append(
                        "l5_v2_gate_artifact_semantics_invalid:"
                        f"{gate_id}:{section}:{axis}:evidence_grade"
                    )
                if not str(row.get("diagnostic_reason") or "").strip():
                    blockers.append(
                        "l5_v2_gate_artifact_semantics_missing:"
                        f"{gate_id}:{section}:{axis}:diagnostic_reason"
                    )
    return blockers


def _gate_semantic_blockers(gate_id: str, artifact_payload: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if gate_id == "byte_closed_temporal_sideinfo_consumption":
        proof = artifact_payload.get("byte_mutation_proof")
        if not isinstance(proof, Mapping):
            return [f"l5_v2_gate_artifact_semantics_missing:{gate_id}:byte_mutation_proof"]
        if proof.get("parser_consumed_bytes") is not True:
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:byte_mutation_proof:parser_consumed_bytes"
            )
        if proof.get("output_changed") is not True:
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:byte_mutation_proof:output_changed"
            )
        if str(proof.get("section") or "").strip() not in {
            "temporal_sideinfo",
            "temporal_side_info",
            "tt5l_temporal_sideinfo",
        }:
            blockers.append(
                f"l5_v2_gate_artifact_semantics_invalid:{gate_id}:byte_mutation_proof:section"
            )
        offsets = proof.get("mutated_byte_offsets")
        if (
            not isinstance(offsets, list)
            or not offsets
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in offsets
            )
        ):
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:byte_mutation_proof:mutated_byte_offsets"
            )
        baseline_sha = str(proof.get("baseline_inflate_sha256") or "").strip().lower()
        mutated_sha = str(proof.get("mutated_inflate_sha256") or "").strip().lower()
        if (
            not _SHA256_HEX_RE.fullmatch(baseline_sha)
            or not _SHA256_HEX_RE.fullmatch(mutated_sha)
            or baseline_sha == mutated_sha
        ):
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:byte_mutation_proof:inflate_sha_pair"
            )
        return blockers

    if gate_id == "c1_z5_tt5l_probe_disambiguator":
        probe = artifact_payload.get("probe_disambiguator")
        if not isinstance(probe, Mapping):
            return [f"l5_v2_gate_artifact_semantics_missing:{gate_id}:probe_disambiguator"]
        if probe.get("schema") != L5V2_PROBE_SCHEMA:
            blockers.append(
                f"l5_v2_gate_artifact_semantics_invalid:{gate_id}:probe_schema"
            )
        if probe.get("tool_path") != L5V2_PROBE_TOOL_PATH:
            blockers.append(
                f"l5_v2_gate_artifact_semantics_invalid:{gate_id}:probe_tool_path"
            )
        candidate_ids = set(_string_items(probe.get("candidate_ids")))
        missing_candidates = [
            candidate_id
            for candidate_id in L5V2_CANDIDATES
            if candidate_id not in candidate_ids
        ]
        if missing_candidates:
            blockers.append(
                f"l5_v2_gate_artifact_semantics_missing:{gate_id}:candidates:"
                + ",".join(missing_candidates)
            )
        if probe.get("paired_exact_axes_required") is not True:
            blockers.append(
                "l5_v2_gate_artifact_semantics_invalid:"
                f"{gate_id}:paired_exact_axes_required"
            )
        return blockers

    if gate_id == "paired_cpu_cuda_axis_plan":
        rows = _axis_row_map(
            artifact_payload.get("paired_axis_plan")
            or artifact_payload.get("axis_plan")
        )
        return _paired_row_identity_blockers(
            gate_id=gate_id,
            rows=rows,
            section="paired_axis_plan",
        )

    if gate_id == "exact_anchor_or_diagnostic_pair":
        rows = _axis_row_map(
            artifact_payload.get("anchor_pair")
            or artifact_payload.get("diagnostic_pair")
        )
        return _paired_row_identity_blockers(
            gate_id=gate_id,
            rows=rows,
            section="anchor_pair",
            require_anchor_type=True,
        )

    return blockers


def _gate_evidence_blockers(
    gate: L5V2Gate,
    evidence: L5V2GateEvidence | None,
    *,
    repo_root: Path,
) -> list[str]:
    if evidence is None:
        return [f"l5_v2_gate_evidence_missing:{gate.gate_id}"]

    blockers: list[str] = []
    resolved_artifact_path: Path | None = None
    path_error: str | None = None
    if not evidence.artifact_path.strip():
        blockers.append(f"l5_v2_gate_artifact_path_missing:{gate.gate_id}")
    elif _is_transient_artifact_path(evidence.artifact_path):
        blockers.append(f"l5_v2_gate_artifact_path_transient:{gate.gate_id}")
    else:
        resolved_artifact_path, path_error = _resolve_artifact_path(
            evidence.artifact_path,
            repo_root,
        )
        if path_error == "outside_repo":
            blockers.append(f"l5_v2_gate_artifact_path_outside_repo:{gate.gate_id}")
        elif resolved_artifact_path is not None and not resolved_artifact_path.is_file():
            blockers.append(f"l5_v2_gate_artifact_file_missing:{gate.gate_id}")
        elif resolved_artifact_path is not None and resolved_artifact_path.is_file():
            try:
                artifact_payload = json.loads(
                    resolved_artifact_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as exc:
                blockers.append(
                    f"l5_v2_gate_artifact_json_invalid:{gate.gate_id}:{exc.msg}"
                )
            else:
                if not isinstance(artifact_payload, Mapping):
                    blockers.append(f"l5_v2_gate_artifact_json_not_object:{gate.gate_id}")
                else:
                    artifact_gate_id = str(artifact_payload.get("gate_id") or "").strip()
                    if not artifact_gate_id:
                        blockers.append(f"l5_v2_gate_artifact_gate_id_missing:{gate.gate_id}")
                    elif artifact_gate_id != gate.gate_id:
                        blockers.append(
                            "l5_v2_gate_artifact_gate_id_mismatch:"
                            f"{gate.gate_id}:{artifact_gate_id}"
                        )
                    artifact_predicate_id = str(
                        artifact_payload.get("predicate_id") or ""
                    ).strip()
                    if (
                        artifact_predicate_id
                        and artifact_predicate_id != evidence.predicate_id.strip()
                    ):
                        blockers.append(
                            "l5_v2_gate_artifact_predicate_id_mismatch:"
                            f"{gate.gate_id}:{artifact_predicate_id}"
                        )
                    if "predicate_passed" in artifact_payload:
                        artifact_passed = artifact_payload["predicate_passed"]
                        artifact_bool_field = "predicate_passed"
                    elif "passed" in artifact_payload:
                        artifact_passed = artifact_payload["passed"]
                        artifact_bool_field = "passed"
                    else:
                        artifact_passed = None
                        artifact_bool_field = ""
                    if artifact_passed is False:
                        blockers.append(
                            f"l5_v2_gate_artifact_predicate_failed:{gate.gate_id}"
                        )
                    elif artifact_passed is not True and artifact_bool_field:
                        blockers.append(
                            "l5_v2_gate_artifact_predicate_non_bool:"
                            f"{gate.gate_id}:{artifact_bool_field}"
                        )
                    elif artifact_passed is not True:
                        blockers.append(
                            f"l5_v2_gate_artifact_predicate_missing:{gate.gate_id}"
                        )
                    blockers.extend(
                        _gate_semantic_blockers(gate.gate_id, artifact_payload)
                    )
    if not _SHA256_HEX_RE.fullmatch(evidence.artifact_sha256.strip()):
        blockers.append(f"l5_v2_gate_artifact_sha256_invalid:{gate.gate_id}")
    elif (
        resolved_artifact_path is not None
        and path_error is None
        and resolved_artifact_path.is_file()
        and _sha256_file(resolved_artifact_path)
        != evidence.artifact_sha256.strip().lower()
    ):
        blockers.append(f"l5_v2_gate_artifact_sha256_mismatch:{gate.gate_id}")
    if not evidence.predicate_id.strip():
        blockers.append(f"l5_v2_gate_predicate_id_missing:{gate.gate_id}")
    if evidence.predicate_passed is not True:
        blockers.append(f"l5_v2_gate_predicate_failed:{gate.gate_id}")
    return blockers


def l5_v2_dispatch_readiness(
    satisfied_gate_ids: Mapping[str, bool] | None = None,
    gate_evidence: (
        Mapping[str, L5V2GateEvidence | Mapping[str, Any]]
        | Iterable[L5V2GateEvidence | Mapping[str, Any]]
        | None
    ) = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a fail-closed dispatch readiness summary for L5 v2.

    Boolean gate claims are preserved as planning notes only. Dispatch
    readiness requires artifact-backed evidence for every gate so prose or
    stale booleans cannot unlock L5 v2 actuation.
    """

    satisfied_gate_ids = satisfied_gate_ids or {}
    resolved_repo_root = (
        Path(repo_root).resolve() if repo_root is not None else _default_repo_root()
    )
    evidence_by_gate = _coerce_gate_evidence(gate_evidence)
    gates = []
    blockers = []
    evidence_blockers = []
    for gate in l5_v2_required_gates():
        evidence = evidence_by_gate.get(gate.gate_id)
        gate_evidence_blockers = _gate_evidence_blockers(
            gate,
            evidence,
            repo_root=resolved_repo_root,
        )
        gate_claim_blockers: list[str] = []
        evidence_valid = not gate_evidence_blockers
        if gate.gate_id in satisfied_gate_ids:
            raw_claimed_satisfied = satisfied_gate_ids[gate.gate_id]
            if raw_claimed_satisfied is True:
                claimed_satisfied = True
            elif raw_claimed_satisfied is False:
                claimed_satisfied = False
            else:
                claimed_satisfied = False
                gate_claim_blockers.append(f"l5_v2_gate_claim_non_bool:{gate.gate_id}")
        else:
            claimed_satisfied = False
        status = "satisfied" if evidence_valid else gate.status
        gate_payload = {
            **gate.__dict__,
            "status": status,
            "claimed_satisfied": claimed_satisfied,
            "claimed_satisfied_without_artifact": (
                claimed_satisfied and not evidence_valid
            ),
            "evidence_valid": evidence_valid,
            "evidence": evidence.__dict__ if evidence is not None else None,
            "evidence_blockers": gate_evidence_blockers,
            "claim_blockers": gate_claim_blockers,
        }
        gates.append(gate_payload)
        if not evidence_valid:
            blockers.append(gate.blocker)
        evidence_blockers.extend(gate_evidence_blockers)
        evidence_blockers.extend(gate_claim_blockers)
    all_gate_claims_satisfied = all(gate["evidence_valid"] for gate in gates)
    all_gate_evidence_valid = all(gate["evidence_valid"] for gate in gates)
    prediction_band_verdict = l5_v2_prediction_band_verdict()
    prediction_band_rank_ready_raw = prediction_band_verdict.get(
        "valid_for_rank_reward"
    )
    prediction_band_rank_ready = prediction_band_rank_ready_raw is True
    prediction_band_blockers = [
        str(blocker)
        for blocker in prediction_band_verdict.get("blockers", [])
        if str(blocker)
    ]
    score_dispatch_blockers: list[str] = []
    if (
        prediction_band_rank_ready_raw is not True
        and prediction_band_rank_ready_raw is not False
    ):
        score_dispatch_blockers.append(
            "prediction_band_valid_for_rank_reward_non_bool"
        )
    if not prediction_band_rank_ready:
        score_dispatch_blockers.append("prediction_band_not_dispatch_ready")
        score_dispatch_blockers.extend(
            f"prediction_band:{blocker}" for blocker in prediction_band_blockers
        )
    ready_for_score_or_rank_dispatch = (
        all_gate_evidence_valid and prediction_band_rank_ready
    )
    return {
        "subject_id": SUBJECT_ID,
        "lane_id": LANE_ID,
        "campaign_id": CAMPAIGN_ID,
        "planning_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "all_gate_claims_satisfied": all_gate_claims_satisfied,
        "all_gate_evidence_valid": all_gate_evidence_valid,
        "ready_for_gate_probe_dispatch": all_gate_evidence_valid,
        "ready_for_score_or_rank_dispatch": ready_for_score_or_rank_dispatch,
        "ready_for_dispatch": ready_for_score_or_rank_dispatch,
        "blockers": blockers + evidence_blockers + score_dispatch_blockers,
        "gates": gates,
        "steps": [step.__dict__ for step in l5_v2_staircase_steps()],
        "prediction_band_verdict": prediction_band_verdict,
        "prediction_band_rank_ready": prediction_band_rank_ready,
    }


__all__ = [
    "CAMPAIGN_ID",
    "LANE_ID",
    "PREDICTED_DELTA_AXIS",
    "PREDICTED_DELTA_BAND",
    "SUBJECT_ID",
    "L5V2Gate",
    "L5V2GateEvidence",
    "L5V2Step",
    "l5_v2_dispatch_readiness",
    "l5_v2_prediction_band_payload",
    "l5_v2_prediction_band_verdict",
    "l5_v2_required_gates",
    "l5_v2_research_basis_ids",
    "l5_v2_staircase_steps",
]
