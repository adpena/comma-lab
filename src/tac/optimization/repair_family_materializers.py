# SPDX-License-Identifier: MIT
"""Family-specific repair materializer manifests for repair campaigns.

These builders do not optimize at receiver time. They bind encoder-side repair
family rows to byte-closed archive/runtime/component evidence when it already
exists, otherwise they emit typed blockers that downstream learning can ingest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.family_agnostic_materializers import (
    verify_runtime_consumption_proof,
)
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.repair_campaign_chain_contract import (
    REPAIR_CAMPAIGN_REQUIRED_OPTIMIZER_SOLVER,
    require_interaction_aware_optimizer_decision,
)
from tac.optimization.repair_campaign_scorer import REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA
from tac.repo_io import sha256_file

REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA = "repair_campaign_family_materializer_manifest.v1"

_MATERIALIZATION_PLAN_SCHEMA = "frontier_rate_attack_repair_budget_materialization_plan.v1"

_ENTROPY_PIPELINE_ORDER: tuple[dict[str, Any], ...] = (
    {
        "order": 10,
        "stage": "before_entropy_coder_distribution_shaping",
        "class": "pre_entropy_distribution_shaping",
        "information_effect": "can_shape_symbols_before_entropy_concentration",
    },
    {
        "order": 20,
        "stage": "scorer_entropy_repair_before_selector_codec",
        "class": "scorer_entropy_before_selector_codec",
        "information_effect": "can_move_distortion_into_low_response_scorer_regions",
    },
    {
        "order": 30,
        "stage": "selector_codec_entropy",
        "class": "selector_codec_entropy",
        "information_effect": "can_code_region_or_operator_choices",
    },
    {
        "order": 40,
        "stage": "at_entropy_coder_integer_codeword_boundary",
        "class": "entropy_coder_boundary",
        "information_effect": "can exploit integer_codeword_and_model_boundary_slack",
    },
    {
        "order": 50,
        "stage": "after_entropy_coder_container_or_zip_grammar",
        "class": "post_entropy_container",
        "information_effect": "can_only_repack_or_reorder_container_bytes",
    },
)

_FRACTAL_LEVELS: tuple[str, ...] = (
    "bit",
    "byte",
    "pixel",
    "boundary",
    "region",
    "frame",
    "pair",
    "batch",
    "full_video",
)

_FAMILY_REQUIRED_VALUE_KEYS: Mapping[str, tuple[str, ...]] = {
    "segnet_class_region_waterfill": ("segnet_class_region_mask_ids",),
    "posenet_null_bottom_decile": ("posenet_null_bottom_decile_pair_ids",),
    "palette_frame_asymmetry_prior": (),
    "frame0_k16_palette_asymmetry": ("palette_dynamics_context",),
    "per_region_selector_codec": ("selector_payload_bits_per_region",),
    "entropy_boundary_probe": ("entropy_boundary_probe_manifest",),
}

_FAMILY_REQUIRED_PATH_KEYS: Mapping[str, tuple[str, ...]] = {
    "segnet_class_region_waterfill": (),
    "posenet_null_bottom_decile": (),
    "palette_frame_asymmetry_prior": ("repair_dynamics_palette_probe_matrix_path",),
    "per_region_selector_codec": (),
}


class RepairFamilyMaterializerError(ValueError):
    """Raised when a family materializer manifest cannot be built."""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    resolved = Path(path)
    repo = Path(repo_root)
    try:
        return str(resolved.resolve(strict=False).relative_to(repo.resolve(strict=False)))
    except ValueError:
        return str(resolved)


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(repo_root) / value


def _is_file(path_text: str, repo_root: str | Path) -> bool:
    return bool(path_text) and _resolve(path_text, repo_root).is_file()


def _find_by_typed_or_candidate(
    rows: Sequence[Any],
    *,
    typed_response_id: str,
    candidate_id: str,
) -> Mapping[str, Any]:
    fallback: Mapping[str, Any] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if typed_response_id and str(row.get("typed_response_id") or "") == typed_response_id:
            return row
        if candidate_id and str(row.get("candidate_id") or "") == candidate_id:
            fallback = row
    return fallback


def _find_child_plan_row(
    materialization_plan: Mapping[str, Any],
    *,
    typed_response_id: str,
    candidate_id: str,
    family_id: str,
) -> Mapping[str, Any]:
    fallback: Mapping[str, Any] = {}
    for row in materialization_plan.get("candidate_chain_rows") or []:
        if not isinstance(row, Mapping):
            continue
        if row.get("candidate_kind") != "spent_budget_repair_child":
            continue
        if typed_response_id and str(row.get("typed_response_id") or "") == typed_response_id:
            return row
        if (candidate_id and str(row.get("allocation_candidate_id") or "") == candidate_id) or (
            family_id and str(row.get("correction_family") or "") == family_id
        ):
            fallback = row
    return fallback


def _entropy_stage(label: str) -> dict[str, Any]:
    for item in _ENTROPY_PIPELINE_ORDER:
        if label == item["stage"]:
            return dict(item)
        if label.startswith("before_entropy_coder") and item["order"] == 10:
            return {**item, "stage": label}
        if label.startswith("at_entropy_coder") and item["order"] == 40:
            return {**item, "stage": label}
        if label.startswith("after_entropy_coder") and item["order"] == 50:
            return {**item, "stage": label}
    return {
        "order": 999,
        "stage": label or "unknown_entropy_pipeline_position",
        "class": "unknown_entropy_pipeline_position",
        "information_effect": "unknown_until_typed_by_materializer",
    }


def _multiscale_scope(
    *,
    child_row: Mapping[str, Any],
    allocation: Mapping[str, Any],
    score_row: Mapping[str, Any],
) -> dict[str, Any]:
    action = _mapping(allocation.get("multiscale_action_row")) or _mapping(score_row.get("multiscale_action_row"))
    dynamics = _mapping(action.get("interaction_dynamics"))
    active = ordered_unique(
        [
            *_string_list(action.get("active_scales")),
            *_string_list(child_row.get("operation_levels")),
            *_string_list(allocation.get("operation_levels")),
            *_string_list(score_row.get("operation_levels")),
        ]
    )
    return {
        "schema": "repair_campaign_family_materializer_fractal_scope.v1",
        "ordered_levels": list(_FRACTAL_LEVELS),
        "active_levels": [level for level in _FRACTAL_LEVELS if level in set(active)],
        "declared_levels": active,
        "interaction_edges": list(dynamics.get("interaction_edges") or []),
        "stackability_policy": (
            "measure_local_atoms_then_remeasure_composed_parent_child_and_sibling_interactions_before_budget_spend"
        ),
        "optimizer_must_preserve_entropy_pipeline_order": True,
        "budget_spend_allowed": False,
        **FALSE_AUTHORITY,
    }


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _safe_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _candidate_archive(
    *,
    child_row: Mapping[str, Any],
    allocation: Mapping[str, Any],
    score_row: Mapping[str, Any],
    repo_root: str | Path,
) -> tuple[dict[str, Any], list[str]]:
    lineage = _mapping(allocation.get("repair_materialization_lineage"))
    receiver_proof = _mapping(score_row.get("receiver_proof_status"))
    lineage_archive = _mapping(lineage.get("candidate_archive"))
    receiver_archive = _mapping(receiver_proof.get("receiver_consumed_candidate_archive"))
    archive_path = _first_text(
        child_row.get("candidate_archive_path"),
        _mapping(child_row.get("candidate_archive")).get("path"),
        lineage_archive.get("path"),
        receiver_archive.get("path"),
    )
    archive_sha = _first_text(
        child_row.get("candidate_archive_sha256"),
        _mapping(child_row.get("candidate_archive")).get("sha256"),
        lineage_archive.get("sha256"),
        receiver_archive.get("sha256"),
    )
    blockers: list[str] = []
    archive_bytes: int | None = None
    if not archive_path:
        blockers.append("repair_family_candidate_archive_path_missing")
    else:
        path = _resolve(archive_path, repo_root)
        if not path.is_file():
            blockers.append("repair_family_candidate_archive_file_missing")
        else:
            actual_sha = sha256_file(path)
            archive_bytes = path.stat().st_size
            if archive_sha and archive_sha != actual_sha:
                blockers.append("repair_family_candidate_archive_sha256_mismatch")
            archive_sha = archive_sha or actual_sha
    if not archive_sha:
        blockers.append("repair_family_candidate_archive_sha256_missing")
    return {
        "path": archive_path or None,
        "sha256": archive_sha or None,
        "bytes": archive_bytes if archive_bytes is not None else child_row.get("candidate_archive_bytes"),
    }, ordered_unique(blockers)


def _runtime_proof_path(
    *,
    child_row: Mapping[str, Any],
    allocation: Mapping[str, Any],
    score_row: Mapping[str, Any],
) -> str:
    lineage = _mapping(allocation.get("repair_materialization_lineage"))
    receiver_proof = _mapping(score_row.get("receiver_proof_status"))
    return _first_text(
        child_row.get("runtime_consumption_proof_path"),
        _mapping(child_row.get("runtime_consumption_proof")).get("path"),
        _mapping(lineage.get("runtime_consumption_proof")).get("path"),
        _mapping(receiver_proof.get("runtime_consumption_proof")).get("path"),
    )


def _gate_path(
    score_row: Mapping[str, Any],
    key: str,
) -> str:
    gate = _mapping(score_row.get("execution_gate"))
    for item in gate.get("local_mlx_custody_paths") or []:
        if isinstance(item, Mapping) and str(item.get("key") or "") == key:
            return str(item.get("path") or "").strip()
    return ""


def _gate_value_nonempty(
    score_row: Mapping[str, Any],
    key: str,
) -> bool:
    gate = _mapping(score_row.get("execution_gate"))
    for item in gate.get("local_mlx_custody_values") or []:
        if isinstance(item, Mapping) and str(item.get("key") or "") == key and item.get("nonempty") is True:
            return True
    return False


def _component_replay(
    *,
    score_report_path: str | Path | None,
    child_row: Mapping[str, Any],
    allocation: Mapping[str, Any],
    score_row: Mapping[str, Any],
    repo_root: str | Path,
) -> tuple[dict[str, Any], list[str]]:
    terms = _mapping(score_row.get("component_response_terms")) or _mapping(allocation.get("component_response_terms"))
    objective_delta = _safe_float(
        score_row.get("objective_delta_score_units")
        or allocation.get("objective_delta_score_units")
        or child_row.get("objective_delta_score_units")
    )
    if objective_delta:
        terms = {
            **dict(terms),
            "objective_delta_score_units": objective_delta,
            "measured_component_delta_score_units": terms.get("measured_component_delta_score_units")
            if terms.get("measured_component_delta_score_units") is not None
            else objective_delta,
        }
    local_mlx = _first_text(
        score_row.get("local_mlx_response_path"),
        child_row.get("local_mlx_response_path"),
        _gate_path(score_row, "local_mlx_response_path"),
        _mapping(child_row.get("local_advisory_custody")).get("local_mlx_response_path"),
    )
    reference_mlx = _first_text(
        score_row.get("reference_local_mlx_response_path"),
        child_row.get("reference_local_mlx_response_path"),
        _gate_path(score_row, "reference_local_mlx_response_path"),
        _mapping(child_row.get("local_advisory_custody")).get("reference_local_mlx_response_path"),
    )
    axis = _first_text(
        allocation.get("component_response_axis"),
        terms.get("response_axis"),
        score_row.get("local_mlx_score_axis"),
    )
    blockers = ["component_response_replay_is_mlx_advisory_only"]
    if not local_mlx:
        blockers.append("local_mlx_response_path_missing")
    elif not _is_file(local_mlx, repo_root):
        blockers.append("local_mlx_response_file_missing")
    if not reference_mlx:
        blockers.append("reference_local_mlx_response_path_missing")
    elif not _is_file(reference_mlx, repo_root):
        blockers.append("reference_local_mlx_response_file_missing")
    if axis != "[macOS-MLX research-signal]":
        blockers.append("local_mlx_component_response_axis_not_research_signal")
    replayed = bool(
        local_mlx
        and reference_mlx
        and axis == "[macOS-MLX research-signal]"
        and _is_file(local_mlx, repo_root)
        and _is_file(reference_mlx, repo_root)
    )
    artifact_path = str(score_report_path or "").strip()
    if replayed and not artifact_path:
        blockers.append("component_response_replay_artifact_path_missing")
        replayed = False
    return {
        "schema": "repair_campaign_family_component_response_replay.v1",
        "replayed": replayed,
        "artifact_path": artifact_path or None,
        "local_mlx_response_path": local_mlx or None,
        "reference_local_mlx_response_path": reference_mlx or None,
        "axis_tag": axis or None,
        "evidence_grade": "local_mlx_component_response_replay_only",
        "component_response_terms": dict(terms),
        "budget_spend_allowed": False,
        **FALSE_AUTHORITY,
    }, ordered_unique(blockers)


def _family_required_blockers(
    *,
    family_id: str,
    child_row: Mapping[str, Any],
    allocation: Mapping[str, Any],
    score_row: Mapping[str, Any],
    repo_root: str | Path,
) -> list[str]:
    blockers: list[str] = []
    sources = (child_row, allocation, score_row)
    for key in _FAMILY_REQUIRED_VALUE_KEYS.get(family_id, ()):
        if not ordered_unique(
            item for source in sources for item in _string_list(source.get(key))
        ) and not _gate_value_nonempty(score_row, key):
            blockers.append(f"{key}_missing")
    for key in _FAMILY_REQUIRED_PATH_KEYS.get(family_id, ()):
        path_text = _first_text(*(source.get(key) for source in sources), _gate_path(score_row, key))
        if not path_text:
            blockers.append(f"{key}_missing")
        elif not _is_file(path_text, repo_root):
            blockers.append(f"{key}_file_missing")
    if family_id == "palette_frame_asymmetry_prior":
        palette_context = _mapping(score_row.get("palette_dynamics_context")) or _mapping(
            allocation.get("palette_dynamics_context")
        )
        if not palette_context:
            blockers.append("palette_dynamics_context_missing")
    if family_id == "frame0_k16_palette_asymmetry":
        palette_context = _mapping(score_row.get("palette_dynamics_context")) or _mapping(
            allocation.get("palette_dynamics_context")
        )
        if not palette_context:
            blockers.append("palette_dynamics_context_missing")
    return ordered_unique(blockers)


def build_repair_campaign_family_materializer_manifest(
    *,
    repo_root: str | Path,
    materialization_plan: Mapping[str, Any],
    score_report: Mapping[str, Any],
    typed_response_id: str,
    candidate_id: str = "",
    materialization_plan_path: str | Path | None = None,
    score_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a family materializer manifest for one repair allocation."""

    if materialization_plan.get("schema") != _MATERIALIZATION_PLAN_SCHEMA:
        raise RepairFamilyMaterializerError("repair family materializer requires repair budget materialization plan")
    if score_report.get("schema") != REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA:
        raise RepairFamilyMaterializerError("repair family materializer requires repair_campaign_score_report.v1")
    require_no_truthy_authority_fields(
        materialization_plan,
        context="repair_family_materializer_materialization_plan",
    )
    require_no_truthy_authority_fields(
        score_report,
        context="repair_family_materializer_score_report",
    )
    decision = _mapping(score_report.get("optimizer_decision"))
    require_interaction_aware_optimizer_decision(
        decision,
        context="repair_campaign_family_materializer_manifest",
    )
    allocation = _find_by_typed_or_candidate(
        decision.get("selected_allocation_rows") or [],
        typed_response_id=typed_response_id,
        candidate_id=candidate_id,
    )
    score_row = _find_by_typed_or_candidate(
        score_report.get("rows") or [],
        typed_response_id=typed_response_id,
        candidate_id=candidate_id,
    )
    family_id = _first_text(
        allocation.get("family_id"),
        score_row.get("family_id"),
        allocation.get("correction_family"),
        score_row.get("correction_family"),
        candidate_id,
        "unclassified_repair_family",
    )
    if family_id == "unclassified_repair_family":
        family_id = _first_text(
            allocation.get("correction_family"),
            score_row.get("correction_family"),
            candidate_id,
            family_id,
        )
    child_row = _find_child_plan_row(
        materialization_plan,
        typed_response_id=typed_response_id,
        candidate_id=candidate_id,
        family_id=family_id,
    )
    candidate_chain_id = str(child_row.get("candidate_chain_id") or "").strip()
    entropy_label = _first_text(
        child_row.get("entropy_position_label"),
        allocation.get("entropy_position_label"),
        score_row.get("entropy_position_label"),
    )
    entropy_stage = _entropy_stage(entropy_label)
    archive, archive_blockers = _candidate_archive(
        child_row=child_row,
        allocation=allocation,
        score_row=score_row,
        repo_root=repo_root,
    )
    proof_path = _runtime_proof_path(
        child_row=child_row,
        allocation=allocation,
        score_row=score_row,
    )
    proof_validation = verify_runtime_consumption_proof(
        runtime_consumption_proof=proof_path or None,
        required_candidate_archive_sha256=str(archive.get("sha256") or "") or None,
        repo_root=repo_root,
    )
    component_replay, component_blockers = _component_replay(
        score_report_path=score_report_path,
        child_row=child_row,
        allocation=allocation,
        score_row=score_row,
        repo_root=repo_root,
    )
    family_blockers = _family_required_blockers(
        family_id=family_id,
        child_row=child_row,
        allocation=allocation,
        score_row=score_row,
        repo_root=repo_root,
    )
    blockers = [
        "exact_auth_eval_required_before_score_or_promotion_claim",
        "family_materializer_is_encoder_side_only",
        *([] if allocation else ["optimizer_selected_allocation_missing"]),
        *([] if score_row else ["source_score_row_missing"]),
        *([] if child_row else ["spent_budget_repair_child_plan_row_missing"]),
        *archive_blockers,
        *_string_list(proof_validation.get("blockers")),
        *component_blockers,
        *family_blockers,
    ]
    byte_closed = (
        bool(candidate_chain_id) and bool(archive.get("path")) and bool(archive.get("sha256")) and not archive_blockers
    )
    receiver_satisfied = byte_closed and proof_validation.get("receiver_contract_satisfied") is True
    if not byte_closed:
        blockers.append(f"{family_id}_byte_closed_candidate_archive_not_materialized")
    if not receiver_satisfied:
        blockers.append(f"{family_id}_archive_bound_runtime_consumption_proof_missing")
    manifest = {
        "schema": REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
        "materializer_id": f"repair_family_materializer:{family_id}",
        "manifest_kind": "repair_campaign_family_materializer",
        "target_kind": family_id,
        "candidate_chain_id": candidate_chain_id or None,
        "candidate_chain_ids": [candidate_chain_id] if candidate_chain_id else [],
        "repair_budget_candidate_chain_id": candidate_chain_id or None,
        "repair_budget_candidate_chain_ids": [candidate_chain_id] if candidate_chain_id else [],
        "candidate_kind": child_row.get("candidate_kind"),
        "chain_id": materialization_plan.get("chain_id"),
        "parent_candidate_chain_id": child_row.get("parent_candidate_chain_id"),
        "typed_response_id": typed_response_id or None,
        "allocation_candidate_id": child_row.get("allocation_candidate_id") or candidate_id or None,
        "family_id": family_id,
        "correction_family": child_row.get("correction_family")
        or allocation.get("correction_family")
        or score_row.get("correction_family"),
        "allocated_repair_bytes": _safe_int(
            allocation.get("allocated_repair_bytes")
            or score_row.get("allocated_repair_bytes")
            or child_row.get("allocated_repair_bytes")
            or allocation.get("requested_repair_bytes")
            or score_row.get("requested_repair_bytes")
        ),
        "objective_delta_score_units": _safe_float(
            allocation.get("objective_delta_score_units")
            or score_row.get("objective_delta_score_units")
            or child_row.get("objective_delta_score_units")
        ),
        "entropy_position_label": entropy_stage["stage"],
        "entropy_pipeline_order": list(_ENTROPY_PIPELINE_ORDER),
        "active_entropy_stage": entropy_stage,
        "fractal_optimization_scope": _multiscale_scope(
            child_row=child_row,
            allocation=allocation,
            score_row=score_row,
        ),
        "source_materialization_plan_path": (
            None if materialization_plan_path is None else str(materialization_plan_path)
        ),
        "source_materialization_plan_schema": materialization_plan.get("schema"),
        "source_score_report_path": None if score_report_path is None else str(score_report_path),
        "source_score_report_schema": score_report.get("schema"),
        "source_optimizer_solver": decision.get("solver"),
        "required_optimizer_solver": REPAIR_CAMPAIGN_REQUIRED_OPTIMIZER_SOLVER,
        "stale_solver_contract_rejected": True,
        "byte_closed_candidate_emitted": byte_closed,
        "candidate_archive": archive,
        "runtime_consumption_proof_path": proof_path or None,
        "receiver_contract_kind": "deterministic_decode_only_repair_family_adapter",
        "receiver_contract_satisfied": receiver_satisfied,
        "runtime_adapter_ready": proof_validation.get("runtime_adapter_ready") is True,
        "receiver_verification": {
            "schema": "repair_campaign_family_receiver_verification.v1",
            "proof_path": proof_path or None,
            "proof_present": proof_validation.get("proof_present") is True,
            "receiver_contract_satisfied": receiver_satisfied,
            "runtime_consumption_proof_passed": receiver_satisfied,
            "proof_validation": proof_validation,
            "blockers": _string_list(proof_validation.get("blockers")),
            "budget_spend_allowed": False,
            **FALSE_AUTHORITY,
        },
        "component_response_replayed": component_replay.get("replayed") is True,
        "component_response_replay": component_replay,
        "readiness_blockers": ordered_unique(blockers),
        "allowed_use": "repair_family_materializer_manifest_for_binding_and_learning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_promotion_or_dispatch_authority",
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        manifest,
        context=(f"repair_campaign_family_materializer_manifest:{typed_response_id or candidate_id or family_id}"),
    )
    return manifest


__all__ = [
    "REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA",
    "RepairFamilyMaterializerError",
    "build_repair_campaign_family_materializer_manifest",
]
