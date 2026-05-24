# SPDX-License-Identifier: MIT
"""Generalized planning surface for shaving excess bytes after training.

The planner answers the recurring contest question: how many units should be
changed, from where, and by which operation family? It consumes grammar-aware
unit signals from X-ray, master-gradient, atoms, local proxy sweeps, or trained
candidate manifests and emits a bounded sweep ladder. It never promotes a score
claim; every row remains planning-only until byte-closed archive/runtime and
exact auth-eval gates pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import permutations
from pathlib import Path
from typing import Any

from tac.archive_byte_profile import CONTEST_ORIGINAL_BYTES
from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.packet_compiler.deterministic_compiler import (
    PACKET_IR_OPERATION_SET_SCHEMA,
    packetir_operation_set_bridge_contract,
)

SIGNAL_SURFACE_SCHEMA = "byte_shaving_signal_surface.v1"
PLAN_SCHEMA = "byte_shaving_campaign_plan.v1"
COUPLED_OPERATION_SET_SCHEMA = "byte_shaving_coupled_operation_set.v1"
TOOL_NAME = "tools/plan_byte_shaving_campaign.py"
INVERSE_ACTION_FUNCTIONAL_SCHEMA = "inverse_steganalysis_discrete_action_functional.v1"
BYTE_SHAVING_OPERATION_SET_PROVENANCE_SCHEMA = (
    "inverse_steganalysis_byte_shaving_operation_set_provenance.v1"
)
MLX_ACQUISITION_BATCH_OPERATION_SET_PROVENANCE_SCHEMA = (
    "inverse_steganalysis_mlx_acquisition_batch_operation_set_provenance.v1"
)
INVERSE_ACTION_WATER_BUCKET_PORTFOLIO_SCHEMA = (
    "inverse_steganalysis_water_bucket_materialization_portfolio.v1"
)
INVERSE_ACTION_MATERIALIZATION_BRIDGE_SCHEMA = (
    "inverse_steganalysis_water_bucket_materialization_bridge.v1"
)
INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY = (
    "compile_inverse_steganalysis_operation_set"
)
INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND = (
    "inverse_steganalysis_high_level_operation_set_v1"
)
INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER = (
    "inverse_steganalysis_operation_set_compiler_required"
)
PACKET_IR_OPERATION_SCHEMA = "packet_ir_operation_v1"
RATE_MULTIPLIER = 25.0
ENGINEERED_CORRECTION_TARGETING_SCHEMA = "master_gradient_consumer_engineered_correction_targeting_v1"
OPERATION_METADATA_KEYS: tuple[str, ...] = (
    "source_family",
    "source_families",
    "source_family_classes",
    "representation_family",
    "representation_family_class",
    "representation_contract",
    "representation_contracts",
    "bolt_on_families",
    "receiver_contract_kind",
    "receiver_contract_kinds",
    "materializer_contract_kind",
    "materializer_contract_kinds",
    "operation_portability",
)

FALSE_AUTHORITY: dict[str, bool] = {
    **PROXY_FALSE_AUTHORITY_FIELDS,
    "dispatch_attempted": False,
    "gpu_launched": False,
}

BASE_BLOCKERS: tuple[str, ...] = (
    "byte_shaving_plan_is_planning_only",
    "requires_byte_closed_materialization_before_dispatch",
    "requires_runtime_consumption_proof_before_exact_eval",
    "requires_same_runtime_locality_or_inflate_parity_check",
    "requires_exact_auth_eval_before_score_claim",
)

UNIT_KINDS: frozenset[str] = frozenset(
    {
        "pair",
        "frame",
        "byte_range",
        "archive_section",
        "tensor",
        "packet_member",
        "scorer_response_row",
        "scorer_inverse_surface_cell",
        "correction_target",
    }
)

DEFAULT_OPERATION_FAMILIES: dict[str, tuple[str, ...]] = {
    "pair": (
        "drop_pair",
        "substitute_pair",
        "lower_pair_precision",
        "proceduralize_pair_residual",
    ),
    "frame": (
        "drop_frame",
        "substitute_frame",
        "temporal_predict_frame",
        "lower_frame_precision",
    ),
    "byte_range": (
        "entropy_recode",
        "null_remove_or_seed",
        "delta_encode",
        "literal_elide",
    ),
    "archive_section": (
        "section_header_elide",
        "section_entropy_recode",
        "section_reorder",
        "section_proceduralize",
    ),
    "tensor": (
        "quantize_tensor",
        "prune_tensor",
        "factorize_tensor",
        "shared_codebook_tensor",
    ),
    "packet_member": (
        "zip_header_elide",
        "member_recompress",
        "member_reorder",
        "member_merge",
    ),
    "scorer_response_row": (
        "materialize_scorer_response_candidate",
        "probe_followup_neighbor",
    ),
    "scorer_inverse_surface_cell": (
        "compile_inverse_steganalysis_operation_set",
        "probe_inverse_scorer_surface_cell",
        "materialize_inverse_scorer_cell_candidate",
    ),
    "correction_target": (
        "apply_engineered_correction",
        "probe_correction_neighbor",
    ),
}

DEFAULT_OPERATION_ORDER_PRIORS: dict[str, int] = {
    "materialize_scorer_response_candidate": 0,
    "compile_inverse_steganalysis_operation_set": 0,
    "probe_inverse_scorer_surface_cell": 0,
    "materialize_inverse_scorer_cell_candidate": 1,
    "probe_followup_neighbor": 1,
    "apply_engineered_correction": 5,
    "probe_correction_neighbor": 6,
    "drop_pair": 10,
    "drop_frame": 10,
    "substitute_pair": 20,
    "substitute_frame": 20,
    "temporal_predict_frame": 20,
    "proceduralize_pair_residual": 20,
    "section_proceduralize": 20,
    "lower_pair_precision": 30,
    "lower_frame_precision": 30,
    "quantize_tensor": 30,
    "prune_tensor": 30,
    "factorize_tensor": 30,
    "shared_codebook_tensor": 30,
    "delta_encode": 40,
    "entropy_recode": 40,
    "section_entropy_recode": 40,
    "member_recompress": 40,
    "literal_elide": 50,
    "null_remove_or_seed": 50,
    "section_header_elide": 50,
    "section_reorder": 50,
    "zip_header_elide": 50,
    "member_reorder": 50,
    "member_merge": 50,
}

PACKET_IR_BYTE_CLOSED_UNIT_KINDS = frozenset(
    {
        "archive_section",
        "byte_range",
        "packet_member",
        "tensor",
    }
)

PACKET_IR_OPERATION_PHASE_BY_FAMILY: dict[str, str] = {
    "apply_engineered_correction": "representation",
    "delta_encode": "arithmetic",
    "entropy_recode": "arithmetic",
    "factorize_tensor": "representation",
    "literal_elide": "pack",
    "member_merge": "pack",
    "member_recompress": "arithmetic",
    "member_reorder": "pack",
    "null_remove_or_seed": "pack",
    "prune_tensor": "representation",
    "quantize_tensor": "quantization",
    "section_entropy_recode": "arithmetic",
    "section_header_elide": "pack",
    "section_proceduralize": "prediction",
    "section_reorder": "pack",
    "shared_codebook_tensor": "quantization",
    "zip_header_elide": "pack",
}


class ByteShavingCampaignError(ValueError):
    """Raised when byte-shaving signal would be malformed or over-authoritative."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_repo_path(path_value: Any, repo_root: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rate_delta_for_saved_bytes(saved_bytes: int) -> float:
    return -RATE_MULTIPLIER * float(saved_bytes) / float(CONTEST_ORIGINAL_BYTES)


def _quality_cost(unit: Mapping[str, Any]) -> float:
    explicit_delta = _finite_float(
        unit.get("predicted_quality_score_delta")
        if unit.get("predicted_quality_score_delta") is not None
        else unit.get("quality_delta_score")
        if unit.get("quality_delta_score") is not None
        else unit.get("predicted_non_rate_delta_score")
    )
    if explicit_delta is not None:
        return explicit_delta
    explicit = _finite_float(unit.get("predicted_quality_score_cost"))
    if explicit is not None:
        return max(0.0, explicit)
    seg_cost = _finite_float(unit.get("predicted_seg_score_cost")) or 0.0
    pose_cost = _finite_float(unit.get("predicted_pose_score_cost")) or 0.0
    master_gradient_cost = _finite_float(unit.get("master_gradient_score_cost")) or 0.0
    return max(0.0, seg_cost + pose_cost + master_gradient_cost)


def _quality_cost_with_fallback(
    operation: Mapping[str, Any],
    unit: Mapping[str, Any],
) -> float:
    explicit_delta = _finite_float(
        operation.get("predicted_quality_score_delta")
        if operation.get("predicted_quality_score_delta") is not None
        else operation.get("quality_delta_score")
        if operation.get("quality_delta_score") is not None
        else operation.get("predicted_non_rate_delta_score")
    )
    if explicit_delta is not None:
        return explicit_delta
    explicit = _finite_float(operation.get("predicted_quality_score_cost"))
    if explicit is not None:
        return max(0.0, explicit)
    seg_cost = _finite_float(operation.get("predicted_seg_score_cost"))
    pose_cost = _finite_float(operation.get("predicted_pose_score_cost"))
    master_gradient_cost = _finite_float(operation.get("master_gradient_score_cost"))
    if seg_cost is not None or pose_cost is not None or master_gradient_cost is not None:
        return max(0.0, (seg_cost or 0.0) + (pose_cost or 0.0) + (master_gradient_cost or 0.0))
    return _quality_cost(unit)


def _confidence(unit: Mapping[str, Any]) -> float:
    raw = _finite_float(unit.get("confidence"))
    if raw is None:
        return 0.5
    return min(1.0, max(0.0, raw))


def _unit_kind(unit: Mapping[str, Any]) -> str:
    kind = str(unit.get("unit_kind") or unit.get("kind") or "").strip()
    return kind if kind in UNIT_KINDS else "byte_range"


def _operation_families(unit: Mapping[str, Any]) -> list[str]:
    explicit = ordered_unique(str(item) for item in _as_list(unit.get("operation_families")))
    if explicit:
        return explicit
    return list(DEFAULT_OPERATION_FAMILIES[_unit_kind(unit)])


def _operation_saved_bytes(operation: Mapping[str, Any], unit_saved_bytes: int) -> int:
    saved = _finite_int(
        operation.get("candidate_saved_bytes")
        if operation.get("candidate_saved_bytes") is not None
        else operation.get("saved_bytes")
        if operation.get("saved_bytes") is not None
        else operation.get("bytes")
    )
    if saved is None or saved <= 0:
        return unit_saved_bytes
    return saved


def _operation_confidence(operation: Mapping[str, Any], unit: Mapping[str, Any]) -> float:
    raw = _finite_float(operation.get("confidence"))
    if raw is None:
        return _confidence(unit)
    return min(1.0, max(0.0, raw))


def _operation_target_kind(operation: Mapping[str, Any]) -> str | None:
    value = str(operation.get("target_kind") or operation.get("materializer_target_kind") or "").strip()
    if value:
        return value
    params = operation.get("params")
    if isinstance(params, Mapping):
        value = str(params.get("target_kind") or params.get("materializer_target_kind") or "").strip()
        if value:
            return value
    return None


def _operation_metadata(operation: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve family/receiver metadata through planning permutations."""

    return {
        key: operation[key]
        for key in OPERATION_METADATA_KEYS
        if key in operation and operation[key] is not None
    }


def _operation_candidates(unit: Mapping[str, Any], unit_saved_bytes: int) -> list[dict[str, Any]]:
    explicit_operations = [item for item in _as_list(unit.get("operations")) if isinstance(item, Mapping)]
    raw_operations: list[Mapping[str, Any]] = explicit_operations or [
        {
            "operation_family": family,
            "materializer": unit.get("materializer"),
            "target_kind": unit.get("target_kind") or unit.get("materializer_target_kind"),
            "params": _mapping(unit.get("operation_params") or unit.get("params")),
        }
        for family in _operation_families(unit)
    ]

    out: list[dict[str, Any]] = []
    for index, operation in enumerate(raw_operations):
        family = str(
            operation.get("operation_family")
            or operation.get("family")
            or operation.get("operation")
            or _operation_families(unit)[0]
        )
        saved_bytes = _operation_saved_bytes(operation, unit_saved_bytes)
        quality_cost = _quality_cost_with_fallback(operation, unit)
        rate_delta = _rate_delta_for_saved_bytes(saved_bytes)
        expected_delta = rate_delta + quality_cost
        gain = -expected_delta
        confidence = _operation_confidence(operation, unit)
        blockers = ordered_unique(str(item) for item in _as_list(operation.get("blockers")))
        out.append(
            {
                "operation_id": str(operation.get("operation_id") or operation.get("id") or f"{family}_{index}"),
                "operation_family": family,
                "candidate_saved_bytes": saved_bytes,
                "rate_delta_score": rate_delta,
                "quality_cost_score": quality_cost,
                "expected_delta_score": expected_delta,
                "expected_score_gain": gain,
                "confidence": confidence,
                "confidence_adjusted_gain": gain * confidence,
                "gain_per_byte": gain / float(saved_bytes) if saved_bytes > 0 else 0.0,
                "operation_priority": int(DEFAULT_OPERATION_ORDER_PRIORS.get(family, 100)),
                "materializer": operation.get("materializer"),
                "target_kind": _operation_target_kind(operation),
                "operation_params": dict(_mapping(operation.get("params"))),
                "operation_metadata": _operation_metadata(operation),
                "blockers": blockers,
            }
        )
    return sorted(
        out,
        key=lambda row: (
            -float(row["confidence_adjusted_gain"]),
            -float(row["expected_score_gain"]),
            float(row["quality_cost_score"]),
            -int(row["candidate_saved_bytes"]),
            int(row["operation_priority"]),
            len(_as_list(row.get("blockers"))),
            str(row["operation_family"]),
            str(row["operation_id"]),
        ),
    )


def validate_signal_surface(payload: Mapping[str, Any]) -> None:
    """Validate the signal surface and preserve false-authority semantics."""

    if payload.get("schema") != SIGNAL_SURFACE_SCHEMA:
        raise ByteShavingCampaignError(f"expected schema {SIGNAL_SURFACE_SCHEMA}")
    try:
        require_no_truthy_authority_fields(payload, context="byte_shaving_signal_surface")
    except ValueError as exc:
        raise ByteShavingCampaignError(str(exc)) from exc
    for key, expected in FALSE_AUTHORITY.items():
        if key in payload and payload.get(key) is not expected:
            raise ByteShavingCampaignError(f"{key} must be explicit false")
    units = payload.get("units")
    if not isinstance(units, list) or not units:
        raise ByteShavingCampaignError("units must be a non-empty list")
    for index, item in enumerate(units):
        if not isinstance(item, Mapping):
            raise ByteShavingCampaignError(f"units[{index}] must be an object")
        if not str(item.get("unit_id") or item.get("id") or "").strip():
            raise ByteShavingCampaignError(f"units[{index}].unit_id is required")
        saved = _finite_int(
            item.get("candidate_saved_bytes")
            if item.get("candidate_saved_bytes") is not None
            else item.get("saved_bytes")
            if item.get("saved_bytes") is not None
            else item.get("bytes")
        )
        allows_zero_saved = _unit_kind(item) in {
            "scorer_response_row",
            "scorer_inverse_surface_cell",
            "correction_target",
        }
        if saved is None or saved < 0 or (saved == 0 and not allows_zero_saved):
            raise ByteShavingCampaignError(f"units[{index}] requires positive candidate_saved_bytes/saved_bytes/bytes")


def normalize_unit_signal(unit: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one pair/frame/byte/section/tensor opportunity."""

    saved_bytes = _finite_int(
        unit.get("candidate_saved_bytes")
        if unit.get("candidate_saved_bytes") is not None
        else unit.get("saved_bytes")
        if unit.get("saved_bytes") is not None
        else unit.get("bytes")
    )
    allows_zero_saved = _unit_kind(unit) in {
        "scorer_response_row",
        "scorer_inverse_surface_cell",
        "correction_target",
    }
    if saved_bytes is None or saved_bytes < 0 or (saved_bytes == 0 and not allows_zero_saved):
        raise ByteShavingCampaignError("unit requires positive saved bytes")
    operation_candidates = _operation_candidates(unit, saved_bytes)
    best_operation = operation_candidates[0]
    quality_cost = float(best_operation["quality_cost_score"])
    rate_delta = float(best_operation["rate_delta_score"])
    expected_delta = float(best_operation["expected_delta_score"])
    gain = float(best_operation["expected_score_gain"])
    confidence = float(best_operation["confidence"])
    density = float(best_operation["gain_per_byte"])
    adjusted_gain = float(best_operation["confidence_adjusted_gain"])
    blockers = ordered_unique(
        [
            *[str(item) for item in _as_list(unit.get("blockers"))],
            *[str(item) for item in _as_list(best_operation.get("blockers"))],
        ]
    )
    row = {
        "unit_id": str(unit.get("unit_id") or unit.get("id")),
        "unit_kind": _unit_kind(unit),
        "source_index": unit.get("source_index"),
        "source_span": unit.get("source_span"),
        "candidate_saved_bytes": int(best_operation["candidate_saved_bytes"]),
        "rate_delta_score": rate_delta,
        "quality_cost_score": quality_cost,
        "expected_delta_score": expected_delta,
        "expected_score_gain": gain,
        "confidence": confidence,
        "confidence_adjusted_gain": adjusted_gain,
        "gain_per_byte": density,
        "recommended_operation_id": best_operation["operation_id"],
        "recommended_operation_family": best_operation["operation_family"],
        "recommended_operation_materializer": best_operation["materializer"],
        "recommended_operation_target_kind": best_operation["target_kind"],
        "recommended_operation_priority": best_operation["operation_priority"],
        "recommended_operation_params": best_operation["operation_params"],
        "recommended_operation_metadata": best_operation["operation_metadata"],
        "operation_families": ordered_unique(str(row["operation_family"]) for row in operation_candidates),
        "operation_candidates": operation_candidates,
        "score_axis": unit.get("score_axis") or unit.get("dominant_axis"),
        "evidence_grade": unit.get("evidence_grade"),
        "evidence_semantics": unit.get("evidence_semantics"),
        "source_paths": _as_list(unit.get("source_paths")),
        "source_candidate_id": unit.get("source_candidate_id"),
        "candidate_archive_sha256": unit.get("candidate_archive_sha256"),
        "candidate_archive_bytes": unit.get("candidate_archive_bytes"),
        "local_axis": unit.get("local_axis"),
        "target_axis": unit.get("target_axis"),
        "local_score": unit.get("local_score"),
        "projected_contest_score": unit.get("projected_contest_score"),
        "conservative_projected_contest_score": unit.get("conservative_projected_contest_score"),
        "xray_signal": unit.get("xray_signal"),
        "master_gradient_signal": unit.get("master_gradient_signal"),
        "engineered_correction_signal": unit.get("engineered_correction_signal"),
        "canonical_equation_provenance": unit.get("canonical_equation_provenance"),
        "atom_ids": ordered_unique(str(item) for item in _as_list(unit.get("atom_ids"))),
        "candidate_trust_region_blockers": ordered_unique(
            str(item) for item in _as_list(unit.get("candidate_trust_region_blockers"))
        ),
        "paired_control_required": bool(unit.get("paired_control_required", True)),
        "blockers": blockers,
    }
    return apply_proxy_evidence_boundary(row, dispatch_blockers=blockers)


def _rank_units(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [normalize_unit_signal(unit) for unit in units]
    return sorted(
        normalized,
        key=lambda row: (
            -float(row["confidence_adjusted_gain"]),
            -float(row["expected_score_gain"]),
            float(row["quality_cost_score"]),
            -int(row["candidate_saved_bytes"]),
            str(row["unit_id"]),
        ),
    )


def _conflict_sets(payload: Mapping[str, Any]) -> list[set[str]]:
    conflicts: list[set[str]] = []
    for item in _as_list(payload.get("conflicts")):
        raw_ids: Any = item.get("unit_ids") or item.get("units") if isinstance(item, Mapping) else item
        ids = {str(value) for value in _as_list(raw_ids) if str(value)}
        if len(ids) >= 2:
            conflicts.append(ids)
    return conflicts


def _violates_conflict(unit_ids: set[str], conflicts: Sequence[set[str]]) -> bool:
    return any(len(unit_ids & conflict) >= 2 for conflict in conflicts)


def _interaction_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(_as_list(payload.get("interactions"))):
        if not isinstance(item, Mapping):
            continue
        unit_ids = ordered_unique(str(value) for value in _as_list(item.get("unit_ids") or item.get("units")))
        if len(unit_ids) < 2:
            continue
        rows.append(
            {
                "interaction_id": str(item.get("interaction_id") or item.get("id") or f"interaction_{index}"),
                "unit_ids": unit_ids,
                "operation_families": ordered_unique(
                    str(value) for value in _as_list(item.get("operation_families") or item.get("operations"))
                ),
                "delta_score": _finite_float(item.get("delta_score")) or 0.0,
                "quality_cost_delta_score": _finite_float(item.get("quality_cost_delta_score")) or 0.0,
                "extra_saved_bytes": _finite_int(item.get("extra_saved_bytes")) or 0,
                "shared_overhead_bytes": _finite_int(item.get("shared_overhead_bytes")) or 0,
                "rationale": item.get("rationale"),
            }
        )
    return rows


def _selection_from_unit(unit: Mapping[str, Any], operation: Mapping[str, Any]) -> dict[str, Any]:
    blockers = ordered_unique(
        [
            *[str(item) for item in _as_list(unit.get("blockers"))],
            *[str(item) for item in _as_list(operation.get("blockers"))],
        ]
    )
    return {
        "unit_id": str(unit["unit_id"]),
        "unit_kind": str(unit["unit_kind"]),
        "operation_id": str(operation["operation_id"]),
        "operation_family": str(operation["operation_family"]),
        "candidate_saved_bytes": int(operation["candidate_saved_bytes"]),
        "quality_cost_score": float(operation["quality_cost_score"]),
        "confidence": float(operation["confidence"]),
        "materializer": operation.get("materializer"),
        "target_kind": operation.get("target_kind"),
        "params": operation.get("operation_params") or {},
        "operation_metadata": dict(_mapping(operation.get("operation_metadata"))),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _active_interactions(
    selections: Sequence[Mapping[str, Any]],
    interactions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected_unit_ids = {str(selection["unit_id"]) for selection in selections}
    selected_families = {str(selection["operation_family"]) for selection in selections}
    active: list[dict[str, Any]] = []
    for interaction in interactions:
        required_units = {str(value) for value in _as_list(interaction.get("unit_ids"))}
        if not required_units or not required_units.issubset(selected_unit_ids):
            continue
        required_families = {str(value) for value in _as_list(interaction.get("operation_families"))}
        if required_families and not required_families.issubset(selected_families):
            continue
        active.append(dict(interaction))
    return active


def _combo_row(
    selections: Sequence[Mapping[str, Any]],
    *,
    interactions: Sequence[Mapping[str, Any]],
    combo_id: str,
) -> dict[str, Any]:
    active = _active_interactions(selections, interactions)
    base_saved = sum(int(selection["candidate_saved_bytes"]) for selection in selections)
    extra_saved = sum(int(interaction.get("extra_saved_bytes") or 0) for interaction in active)
    overhead = sum(int(interaction.get("shared_overhead_bytes") or 0) for interaction in active)
    total_saved = max(0, base_saved + extra_saved - overhead)
    quality_cost = sum(float(selection["quality_cost_score"]) for selection in selections) + sum(
        float(interaction.get("quality_cost_delta_score") or 0.0) for interaction in active
    )
    direct_delta = sum(float(interaction.get("delta_score") or 0.0) for interaction in active)
    rate_delta = _rate_delta_for_saved_bytes(total_saved)
    expected_delta = rate_delta + quality_cost + direct_delta
    confidence_values = [float(selection["confidence"]) for selection in selections]
    confidence = min(confidence_values) if confidence_values else 0.0
    return {
        "combo_id": combo_id,
        "unit_count": len(selections),
        "candidate_saved_bytes": total_saved,
        "base_saved_bytes": base_saved,
        "interaction_extra_saved_bytes": extra_saved,
        "interaction_shared_overhead_bytes": overhead,
        "rate_delta_score": rate_delta,
        "quality_cost_score": quality_cost,
        "interaction_delta_score": direct_delta,
        "expected_delta_score": expected_delta,
        "expected_score_gain": -expected_delta,
        "confidence": confidence,
        "confidence_adjusted_gain": -expected_delta * confidence,
        "selected_unit_ids": [str(selection["unit_id"]) for selection in selections],
        "selected_operations": [
            {
                "unit_id": str(selection["unit_id"]),
                "unit_kind": str(selection.get("unit_kind") or ""),
                "operation_id": str(selection["operation_id"]),
                "operation_family": str(selection["operation_family"]),
                "materializer": selection.get("materializer"),
                "target_kind": selection.get("target_kind"),
                "params": selection.get("params") or {},
                **dict(_mapping(selection.get("operation_metadata"))),
                "blockers": _as_list(selection.get("blockers")),
                **FALSE_AUTHORITY,
            }
            for selection in selections
        ],
        "active_interactions": active,
        "operation_families": ordered_unique(str(selection["operation_family"]) for selection in selections),
        "dispatch_blockers": list(BASE_BLOCKERS),
        **FALSE_AUTHORITY,
    }


def _state_sort_key(row: Mapping[str, Any]) -> tuple[float, float, int, str]:
    return (
        -float(row["confidence_adjusted_gain"]),
        -float(row["expected_score_gain"]),
        -int(row["candidate_saved_bytes"]),
        ",".join(str(value) for value in _as_list(row.get("selected_unit_ids"))),
    )


def _operation_order_priors(payload: Mapping[str, Any]) -> dict[str, int]:
    priors = dict(DEFAULT_OPERATION_ORDER_PRIORS)
    raw = payload.get("operation_order_priors")
    if isinstance(raw, Mapping):
        for family, priority in raw.items():
            parsed = _finite_int(priority)
            if parsed is not None:
                priors[str(family)] = parsed
    else:
        for item in _as_list(raw):
            if not isinstance(item, Mapping):
                continue
            family = str(item.get("operation_family") or item.get("family") or "")
            parsed = _finite_int(item.get("priority"))
            if family and parsed is not None:
                priors[family] = parsed
    return dict(sorted(priors.items()))


def _operation_priority(
    operation: Mapping[str, Any],
    priors: Mapping[str, int],
) -> int:
    return int(priors.get(str(operation.get("operation_family") or ""), 100))


def _permutation_penalty(
    operations: Sequence[Mapping[str, Any]],
    *,
    priors: Mapping[str, int],
) -> int:
    priorities = [_operation_priority(operation, priors) for operation in operations]
    inversions = 0
    for left in range(len(priorities)):
        for right in range(left + 1, len(priorities)):
            if priorities[left] > priorities[right]:
                inversions += 1
    return inversions


def _operation_sequence_record(
    operations: Sequence[Mapping[str, Any]],
    *,
    priors: Mapping[str, int],
) -> list[dict[str, Any]]:
    return [
        {
            "unit_id": str(operation.get("unit_id") or ""),
            "operation_id": str(operation.get("operation_id") or ""),
            "operation_family": str(operation.get("operation_family") or ""),
            "unit_kind": str(operation.get("unit_kind") or ""),
            "order_priority": _operation_priority(operation, priors),
            "materializer": operation.get("materializer"),
            "target_kind": operation.get("target_kind"),
            **_operation_metadata(operation),
        }
        for operation in operations
    ]


def _operation_sequence_key(operation: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(operation.get("unit_id") or ""),
        str(operation.get("operation_id") or ""),
    )


def _operation_sequence_hash(sequence: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "unit_id": str(operation.get("unit_id") or ""),
            "operation_id": str(operation.get("operation_id") or ""),
            "operation_family": str(operation.get("operation_family") or ""),
            "unit_kind": str(operation.get("unit_kind") or ""),
            "materializer": operation.get("materializer"),
            "target_kind": operation.get("target_kind"),
            **_operation_metadata(operation),
        }
        for operation in sequence
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _bounded_operation_permutations(
    operations: Sequence[Mapping[str, Any]],
    *,
    priors: Mapping[str, int],
    max_ops: int,
    max_permutations: int,
) -> list[tuple[int, tuple[Mapping[str, Any], ...]]]:
    prefix = tuple(operations[:max_ops])
    if not prefix:
        return []
    unique: dict[tuple[tuple[str, str], ...], tuple[int, tuple[Mapping[str, Any], ...]]] = {}
    for perm in permutations(prefix):
        key = tuple((str(item.get("unit_id") or ""), str(item.get("operation_id") or "")) for item in perm)
        penalty = _permutation_penalty(perm, priors=priors)
        unique[key] = (penalty, perm)
    return sorted(
        unique.values(),
        key=lambda item: (
            item[0],
            [
                (
                    _operation_priority(operation, priors),
                    str(operation.get("operation_family") or ""),
                    str(operation.get("unit_id") or ""),
                )
                for operation in item[1]
            ],
        ),
    )[:max_permutations]


def _permutation_ladder(
    combo_rows: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    *,
    operation_order_priors: Mapping[str, int],
) -> list[dict[str, Any]]:
    max_combo_rows = max(1, int(payload.get("max_permutation_combo_count") or 8))
    max_ops = max(1, min(8, int(payload.get("max_permutation_ops") or 6)))
    max_permutations = max(1, int(payload.get("max_permutations_per_combo") or 24))
    rows: list[dict[str, Any]] = []
    for combo_rank, combo in enumerate(combo_rows[:max_combo_rows], start=1):
        operations = [item for item in _as_list(combo.get("selected_operations")) if isinstance(item, Mapping)]
        bounded = _bounded_operation_permutations(
            operations,
            priors=operation_order_priors,
            max_ops=min(max_ops, len(operations)),
            max_permutations=max_permutations,
        )
        suffix = operations[min(max_ops, len(operations)) :]
        permutation_rows = []
        for order_rank, (penalty, prefix) in enumerate(bounded, start=1):
            sequence = [*prefix, *suffix]
            permutation_rows.append(
                {
                    "order_rank": order_rank,
                    "operation_sequence": _operation_sequence_record(
                        sequence,
                        priors=operation_order_priors,
                    ),
                    "prior_order_inversion_count": penalty,
                    "permuted_operation_count": len(prefix),
                    "fixed_suffix_count": len(suffix),
                    "planning_notes": [
                        "operation_order_is_prior_not_score_claim",
                        "bounded_permutation_search_avoids_factorial_explosion",
                    ],
                    **FALSE_AUTHORITY,
                }
            )
        rows.append(
            {
                "schema": "byte_shaving_operation_permutation_row.v1",
                "combo_rank": combo_rank,
                "combo_id": combo.get("combo_id"),
                "selected_unit_ids": _as_list(combo.get("selected_unit_ids")),
                "expected_delta_score": combo.get("expected_delta_score"),
                "candidate_saved_bytes": combo.get("candidate_saved_bytes"),
                "operation_count": len(operations),
                "permutation_policy": {
                    "max_ops_permuted": max_ops,
                    "max_permutations_per_combo": max_permutations,
                    "order_prior_source": "payload.operation_order_priors_or_default",
                },
                "permutations": permutation_rows,
                "dispatch_blockers": [
                    *BASE_BLOCKERS,
                    "permutation_order_requires_materialization_and_empirical_probe",
                ],
                **FALSE_AUTHORITY,
            }
        )
    return rows


def _operation_set_ladder(
    combo_rows: Sequence[Mapping[str, Any]],
    permutation_rows: Sequence[Mapping[str, Any]],
    *,
    operation_order_priors: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Promote coupled combo/permutation rows into durable atomic action sets."""

    permutations_by_combo = {
        str(row.get("combo_id") or ""): row
        for row in permutation_rows
        if str(row.get("combo_id") or "")
    }
    rows: list[dict[str, Any]] = []
    for rank, combo in enumerate(combo_rows, start=1):
        combo_id = str(combo.get("combo_id") or f"combo_{rank:04d}")
        operations = [
            item
            for item in _as_list(combo.get("selected_operations"))
            if isinstance(item, Mapping)
        ]
        permutation_row = permutations_by_combo.get(combo_id)
        chosen_sequence: list[dict[str, Any]] = []
        order_inversion_count: int | None = None
        sequence_source = "selected_operations_order"
        if permutation_row is not None:
            permutations_payload = [
                item
                for item in _as_list(permutation_row.get("permutations"))
                if isinstance(item, Mapping)
            ]
            if permutations_payload:
                best_permutation = permutations_payload[0]
                chosen_sequence = [
                    dict(item)
                    for item in _as_list(best_permutation.get("operation_sequence"))
                    if isinstance(item, Mapping)
                ]
                parsed_penalty = _finite_int(
                    best_permutation.get("prior_order_inversion_count")
                )
                order_inversion_count = parsed_penalty if parsed_penalty is not None else None
                sequence_source = "bounded_permutation_ladder_rank_1"
        if not chosen_sequence:
            chosen_sequence = _operation_sequence_record(
                operations,
                priors=operation_order_priors,
            )
        sequence_is_permutation = Counter(
            _operation_sequence_key(operation) for operation in operations
        ) == Counter(_operation_sequence_key(operation) for operation in chosen_sequence)
        dispatch_blockers = ordered_unique(
            [
                *[str(item) for item in _as_list(combo.get("dispatch_blockers"))],
                "operation_set_requires_atomic_materializer_or_explicit_partial_set_split",
                "operation_set_order_requires_materialization_probe",
                *(
                    []
                    if sequence_is_permutation
                    else [
                        (
                            "operation_set_sequence_not_permutation_of_"
                            "selected_operations"
                        )
                    ]
                ),
            ]
        )
        rows.append(
            {
                "schema": COUPLED_OPERATION_SET_SCHEMA,
                "operation_set_id": f"opset_{combo_id}",
                "combo_id": combo_id,
                "operation_set_rank": rank,
                "unit_count": combo.get("unit_count"),
                "selected_unit_ids": _as_list(combo.get("selected_unit_ids")),
                "selected_operations": [dict(operation) for operation in operations],
                "chosen_operation_sequence": chosen_sequence,
                "chosen_operation_sequence_sha256": _operation_sequence_hash(
                    chosen_sequence
                ),
                "chosen_operation_sequence_source": sequence_source,
                "chosen_operation_sequence_is_permutation": sequence_is_permutation,
                "prior_order_inversion_count": order_inversion_count,
                "active_interactions": _as_list(combo.get("active_interactions")),
                "operation_families": _as_list(combo.get("operation_families")),
                "candidate_saved_bytes": combo.get("candidate_saved_bytes"),
                "base_saved_bytes": combo.get("base_saved_bytes"),
                "interaction_extra_saved_bytes": combo.get("interaction_extra_saved_bytes"),
                "interaction_shared_overhead_bytes": combo.get(
                    "interaction_shared_overhead_bytes"
                ),
                "rate_delta_score": combo.get("rate_delta_score"),
                "quality_cost_score": combo.get("quality_cost_score"),
                "interaction_delta_score": combo.get("interaction_delta_score"),
                "expected_delta_score": combo.get("expected_delta_score"),
                "expected_score_gain": combo.get("expected_score_gain"),
                "confidence": combo.get("confidence"),
                "confidence_adjusted_gain": combo.get("confidence_adjusted_gain"),
                "partial_materialization_allowed": False,
                "requires_atomic_materialization": True,
                "dispatch_blockers": dispatch_blockers,
                **FALSE_AUTHORITY,
            }
        )
    return rows


def _packet_ir_phase(operation: Mapping[str, Any]) -> str:
    explicit = str(
        operation.get("packet_ir_phase")
        or operation.get("compiler_phase")
        or ""
    ).strip()
    contract = packetir_operation_set_bridge_contract()
    required_order = {
        str(item) for item in _as_list(contract.get("required_order"))
    }
    if explicit in required_order:
        return explicit
    family = str(operation.get("operation_family") or "").strip()
    return PACKET_IR_OPERATION_PHASE_BY_FAMILY.get(family, "pack")


def _packet_ir_lowered_operation(
    operation: Mapping[str, Any],
    *,
    order_index: int,
) -> dict[str, Any]:
    unit_kind = str(operation.get("unit_kind") or "").strip()
    operation_family = str(operation.get("operation_family") or "").strip()
    target_kind = str(operation.get("target_kind") or "").strip()
    materializer = str(operation.get("materializer") or "").strip()
    blockers = ordered_unique(
        [
            *[str(item) for item in _as_list(operation.get("blockers"))],
            *(
                []
                if unit_kind in PACKET_IR_BYTE_CLOSED_UNIT_KINDS
                else [f"packetir_operation_not_byte_closed:{unit_kind or '<missing>'}"]
            ),
            *(
                []
                if target_kind
                else ["packetir_operation_target_kind_missing"]
            ),
            *(
                []
                if materializer
                else ["packetir_operation_materializer_missing"]
            ),
        ]
    )
    metadata = _operation_metadata(operation)
    return {
        "schema": PACKET_IR_OPERATION_SCHEMA,
        "order_index": order_index,
        "compiler_phase": _packet_ir_phase(operation),
        "unit_id": operation.get("unit_id"),
        "unit_kind": unit_kind,
        "operation_id": operation.get("operation_id"),
        "operation_family": operation_family,
        "target_kind": target_kind or None,
        "materializer": materializer or None,
        "params": dict(_mapping(operation.get("params"))),
        "candidate_saved_bytes": operation.get("candidate_saved_bytes"),
        "predicted_quality_score_delta": operation.get(
            "predicted_quality_score_delta"
        ),
        **metadata,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _packet_ir_operation_set_from_coupled_set(
    operation_set: Mapping[str, Any],
    *,
    source_payload: Mapping[str, Any],
    source_paths: Sequence[str],
) -> dict[str, Any]:
    contract = packetir_operation_set_bridge_contract()
    chosen_sequence = [
        item
        for item in _as_list(operation_set.get("chosen_operation_sequence"))
        if isinstance(item, Mapping)
    ]
    selected_operations = [
        item
        for item in _as_list(operation_set.get("selected_operations"))
        if isinstance(item, Mapping)
    ]
    operation_source = chosen_sequence or selected_operations
    lowered_operations = [
        _packet_ir_lowered_operation(operation, order_index=index)
        for index, operation in enumerate(operation_source)
    ]
    operation_blockers = [
        blocker
        for operation in lowered_operations
        for blocker in _as_list(operation.get("blockers"))
    ]
    source_blockers = [
        str(item)
        for item in _as_list(operation_set.get("dispatch_blockers"))
        if str(item)
    ]
    blockers = ordered_unique(
        [
            *source_blockers,
            *[str(item) for item in operation_blockers],
            "packetir_operation_set_requires_materializer_contexts",
            "packetir_operation_set_requires_runtime_consumption_proof",
            "packetir_operation_set_requires_exact_readiness_handoff",
        ]
    )
    byte_closed_count = sum(
        1
        for operation in lowered_operations
        if not any(
            str(blocker).startswith("packetir_operation_not_byte_closed")
            for blocker in _as_list(operation.get("blockers"))
        )
    )
    return {
        "schema": PACKET_IR_OPERATION_SET_SCHEMA,
        "source_schema": COUPLED_OPERATION_SET_SCHEMA,
        "source_paths": list(source_paths),
        "source_portfolio_schema": _mapping(
            source_payload.get("water_bucket_materialization_portfolio")
        ).get("schema"),
        "candidate_id": source_payload.get("candidate_id"),
        "lane_id": source_payload.get("lane_id"),
        "source_operation_set_id": operation_set.get("operation_set_id"),
        "operation_set_id": f"packetir_{operation_set.get('operation_set_id')}",
        "compiler_contract": contract,
        "source_combo_id": operation_set.get("combo_id"),
        "operation_set_rank": operation_set.get("operation_set_rank"),
        "selected_unit_ids": _as_list(operation_set.get("selected_unit_ids")),
        "selected_operations": [dict(operation) for operation in selected_operations],
        "chosen_operation_sequence": [
            dict(operation) for operation in chosen_sequence
        ],
        "chosen_operation_sequence_sha256": operation_set.get(
            "chosen_operation_sequence_sha256"
        ),
        "chosen_operation_sequence_source": operation_set.get(
            "chosen_operation_sequence_source"
        ),
        "chosen_operation_sequence_is_permutation": operation_set.get(
            "chosen_operation_sequence_is_permutation"
        ),
        "requires_atomic_materialization": True,
        "partial_materialization_allowed": False,
        "operation_count": len(lowered_operations),
        "byte_closed_operation_count": byte_closed_count,
        "operations": lowered_operations,
        "active_interactions": _as_list(operation_set.get("active_interactions")),
        "candidate_saved_bytes": operation_set.get("candidate_saved_bytes"),
        "expected_delta_score": operation_set.get("expected_delta_score"),
        "expected_score_gain": operation_set.get("expected_score_gain"),
        "required_order": list(contract["required_order"]),
        "required_proofs": list(contract["required_proofs"]),
        "required_proofs_status": dict.fromkeys(
            _as_list(contract.get("required_proofs")),
            "missing",
        ),
        "lowering_status": "blocked_until_materializer_contexts_and_proofs",
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _packet_ir_operation_sets(
    operation_set_rows: Sequence[Mapping[str, Any]],
    *,
    source_payload: Mapping[str, Any],
    source_paths: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        _packet_ir_operation_set_from_coupled_set(
            row,
            source_payload=source_payload,
            source_paths=source_paths,
        )
        for row in operation_set_rows
        if row.get("schema") == COUPLED_OPERATION_SET_SCHEMA
    ]


def _combo_ladder(
    ranked_units: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    *,
    max_units_considered: int,
    max_ops_per_unit: int,
    beam_width: int,
    max_combos: int,
) -> list[dict[str, Any]]:
    """Bounded beam search over unit/operation combinations with interactions."""

    conflicts = _conflict_sets(payload)
    interactions = _interaction_rows(payload)
    states: list[list[dict[str, Any]]] = [[]]
    completed: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    for unit in ranked_units[:max_units_considered]:
        next_states = list(states)
        unit_operations = [item for item in _as_list(unit.get("operation_candidates")) if isinstance(item, Mapping)][
            :max_ops_per_unit
        ]
        for state in states:
            for operation in unit_operations:
                selection = _selection_from_unit(unit, operation)
                unit_ids = {str(item["unit_id"]) for item in [*state, selection]}
                if _violates_conflict(unit_ids, conflicts):
                    continue
                candidate_state = [*state, selection]
                if len(candidate_state) >= 2:
                    key = tuple(sorted((str(item["unit_id"]), str(item["operation_id"])) for item in candidate_state))
                    row = _combo_row(
                        candidate_state,
                        interactions=interactions,
                        combo_id=f"combo_{len(completed) + 1:04d}",
                    )
                    prior = completed.get(key)
                    if prior is None or _state_sort_key(row) < _state_sort_key(prior):
                        completed[key] = row
                next_states.append(candidate_state)
        ranked_states = sorted(
            [
                state
                for state in next_states
                if not _violates_conflict(
                    {str(item["unit_id"]) for item in state},
                    conflicts,
                )
            ],
            key=lambda state: (
                _state_sort_key(
                    _combo_row(
                        state,
                        interactions=interactions,
                        combo_id="state",
                    )
                )
                if state
                else (0.0, 0.0, 0, "")
            ),
        )
        states = ranked_states[:beam_width]
    return sorted(completed.values(), key=_state_sort_key)[:max_combos]


def _sweep_ladder_counts(n_units: int, *, max_k: int | None) -> list[int]:
    if n_units <= 0:
        return []
    upper = min(n_units, max_k if max_k is not None else n_units)
    base = [1, 2, 3, 4, 5, 8, 13, 21, 32, 48, 64, 96, 128]
    counts = {k for k in base if 1 <= k <= upper}
    counts.update({max(1, upper // 4), max(1, upper // 2), upper})
    return sorted(counts)


def _conflict_violation_ids(unit_ids: Sequence[str], conflicts: Sequence[set[str]]) -> list[list[str]]:
    selected = set(unit_ids)
    return [sorted(selected & conflict) for conflict in conflicts if len(selected & conflict) >= 2]


def _prefix_rows(
    ranked_units: Sequence[Mapping[str, Any]],
    counts: Iterable[int],
    *,
    conflicts: Sequence[set[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for count in counts:
        selected = list(ranked_units[:count])
        selected_unit_ids = [str(row["unit_id"]) for row in selected]
        conflict_violations = _conflict_violation_ids(selected_unit_ids, conflicts)
        saved_bytes = int(sum(int(row["candidate_saved_bytes"]) for row in selected))
        rate_delta = float(sum(float(row["rate_delta_score"]) for row in selected))
        quality_cost = float(sum(float(row["quality_cost_score"]) for row in selected))
        expected_delta = rate_delta + quality_cost
        blockers = list(BASE_BLOCKERS)
        if conflict_violations:
            blockers.append("prefix_selection_violates_conflict_sets")
        rows.append(
            {
                "sweep_id": f"top_{count:04d}",
                "unit_count": count,
                "candidate_saved_bytes": saved_bytes,
                "rate_delta_score": rate_delta,
                "quality_cost_score": quality_cost,
                "expected_delta_score": expected_delta,
                "expected_score_gain": -expected_delta,
                "operation_families": ordered_unique(
                    str(row.get("recommended_operation_family") or family)
                    for row in selected
                    for family in _as_list(row.get("operation_families"))
                ),
                "selected_operations": [
                    {
                        "unit_id": str(row["unit_id"]),
                        "unit_kind": str(row.get("unit_kind") or ""),
                        "operation_id": str(row["recommended_operation_id"]),
                        "operation_family": str(row["recommended_operation_family"]),
                        "materializer": row.get("recommended_operation_materializer"),
                        "target_kind": row.get("recommended_operation_target_kind"),
                        "params": row.get("recommended_operation_params") or {},
                        **dict(_mapping(row.get("recommended_operation_metadata"))),
                        "blockers": _as_list(row.get("blockers")),
                        **FALSE_AUTHORITY,
                    }
                    for row in selected
                ],
                "selected_unit_ids": selected_unit_ids,
                "conflict_violations": conflict_violations,
                "dispatch_blockers": blockers,
                **FALSE_AUTHORITY,
            }
        )
    return rows


def _best_nonpositive_prefix(prefix_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    viable = [
        row
        for row in prefix_rows
        if float(row.get("expected_delta_score") or 0.0) < 0.0 and not row.get("conflict_violations")
    ]
    if not viable:
        return None
    return max(
        viable,
        key=lambda row: (
            float(row["expected_score_gain"]),
            int(row["candidate_saved_bytes"]),
            -int(row["unit_count"]),
        ),
    )


def build_byte_shaving_campaign_plan(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None = None,
    repo_root: Path | None = None,
    max_k: int | None = None,
) -> dict[str, Any]:
    """Build a ranked, bounded sweep plan from one signal surface."""

    validate_signal_surface(payload)
    repo_root = repo_root or Path.cwd()
    ranked = _rank_units([item for item in _as_list(payload.get("units")) if isinstance(item, Mapping)])
    conflicts = _conflict_sets(payload)
    operation_order_priors = _operation_order_priors(payload)
    prefix_rows = _prefix_rows(
        ranked,
        _sweep_ladder_counts(len(ranked), max_k=max_k),
        conflicts=conflicts,
    )
    combo_rows = _combo_ladder(
        ranked,
        payload,
        max_units_considered=int(payload.get("max_combo_units_considered") or 32),
        max_ops_per_unit=int(payload.get("max_combo_ops_per_unit") or 3),
        beam_width=int(payload.get("combo_beam_width") or 64),
        max_combos=int(payload.get("max_combo_count") or 32),
    )
    permutation_rows = _permutation_ladder(
        combo_rows,
        payload,
        operation_order_priors=operation_order_priors,
    )
    operation_set_rows = _operation_set_ladder(
        combo_rows,
        permutation_rows,
        operation_order_priors=operation_order_priors,
    )
    best = _best_nonpositive_prefix(prefix_rows)
    best_combo = _best_nonpositive_prefix(combo_rows)
    best_operation_set = _best_nonpositive_prefix(operation_set_rows)
    source_paths = [_repo_rel(source_path, repo_root)] if source_path is not None else []
    plan = {
        "schema": PLAN_SCHEMA,
        "tool": TOOL_NAME,
        "generated_at_utc": _utc_now(),
        "campaign_id": str(payload.get("campaign_id") or "byte_shaving_campaign"),
        "candidate_id": payload.get("candidate_id"),
        "lane_id": payload.get("lane_id") or "byte_shaving_campaign",
        "source_paths": source_paths,
        "source_signal_refs": _as_list(payload.get("source_signal_refs")),
        "auth_eval_refs": _as_list(payload.get("auth_eval_refs")),
        "mlx_calibration_refs": _as_list(payload.get("mlx_calibration_refs")),
        "scorer_response_refs": _as_list(payload.get("scorer_response_refs")),
        "inverse_scorer_surface_refs": _as_list(payload.get("inverse_scorer_surface_refs")),
        "inverse_action_materialization_portfolios": (
            _inverse_action_materialization_portfolios(payload)
        ),
        "frontier_axis": payload.get("frontier_axis") or "[planning-only]",
        "ranked_units": ranked,
        "sweep_ladder": prefix_rows,
        "recommended_prefix": dict(best) if best is not None else None,
        "combination_ladder": combo_rows,
        "permutation_ladder": permutation_rows,
        "operation_set_ladder": operation_set_rows,
        "packet_ir_operation_sets": _packet_ir_operation_sets(
            operation_set_rows,
            source_payload=payload,
            source_paths=source_paths,
        ),
        "recommended_combination": dict(best_combo) if best_combo is not None else None,
        "recommended_operation_set": (
            dict(best_operation_set) if best_operation_set is not None else None
        ),
        "operation_menu": DEFAULT_OPERATION_FAMILIES,
        "operation_order_priors": operation_order_priors,
        "combination_policy": {
            "max_units_considered": int(payload.get("max_combo_units_considered") or 32),
            "max_ops_per_unit": int(payload.get("max_combo_ops_per_unit") or 3),
            "beam_width": int(payload.get("combo_beam_width") or 64),
            "max_combo_count": int(payload.get("max_combo_count") or 32),
            "interactions_mode": (
                "direct delta_score plus extra_saved_bytes/shared_overhead_bytes "
                "for scorer-axis and runtime externalities"
            ),
            "conflicts_mode": "unit conflict sets cannot co-occur in one combo",
        },
        "search_space_policy": {
            "unit_layers": sorted(UNIT_KINDS),
            "operation_layers": {
                unit_kind: list(families) for unit_kind, families in sorted(DEFAULT_OPERATION_FAMILIES.items())
            },
            "scorer_interaction_terms": [
                "rate_delta_score",
                "quality_cost_score",
                "predicted_seg_score_cost",
                "predicted_pose_score_cost",
                "master_gradient_score_cost",
                "interaction_delta_score",
            ],
            "combination_search": "bounded_beam_over_units_and_operation_alternatives",
            "permutation_search": "bounded_operation_order_permutations_for_top_combos",
            "operation_set_search": (
                "durable_coupled_operation_sets_preserve_interactions_and_order_for_queueing"
            ),
            "non_bruteforce_principle": (
                "rank by rate-distortion prior, component/scorer marginal costs, "
                "explicit interactions, conflicts, and confidence before any local "
                "or exact scorer spend"
            ),
        },
        "dispatch_blockers": ordered_unique(
            [*BASE_BLOCKERS, *[str(item) for item in _as_list(payload.get("blockers"))]]
        ),
        "evidence_boundary": {
            "planning_only": True,
            "rate_delta_formula": f"-25 * saved_bytes / {CONTEST_ORIGINAL_BYTES}",
            "quality_cost_source": ("predicted_quality_score_cost or seg/pose/master-gradient score costs"),
            "next_gate": (
                "materialize selected operation family, run locality/inflate "
                "controls, then exact auth eval before any score claim"
            ),
        },
        **FALSE_AUTHORITY,
    }
    return apply_proxy_evidence_boundary(plan, dispatch_blockers=plan["dispatch_blockers"])


def _inverse_action_materialization_portfolios(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    single = payload.get("water_bucket_materialization_portfolio")
    if isinstance(single, Mapping):
        out.append(dict(single))
    for item in _as_list(payload.get("inverse_action_materialization_portfolios")):
        if isinstance(item, Mapping):
            out.append(dict(item))
    return out


def _mapping_rows(payload: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    return [dict(item) for item in _as_list(payload.get(key)) if isinstance(item, Mapping)]


def _inverse_action_portfolio_row_links(
    portfolio_rows: Sequence[Mapping[str, Any]],
    packet_ir_operation_sets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for row in portfolio_rows:
        unit_ids = [str(item) for item in _as_list(row.get("unit_ids")) if str(item)]
        unit_id_set = set(unit_ids)
        matched_packet_ir = []
        for operation_set in packet_ir_operation_sets:
            selected_ids = {
                str(item)
                for item in _as_list(operation_set.get("selected_unit_ids"))
                if str(item)
            }
            if unit_id_set and unit_id_set.issubset(selected_ids):
                matched_packet_ir.append(operation_set)
        actuation_mode = str(row.get("actuation_mode") or "")
        blockers = [str(item) for item in _as_list(row.get("blockers")) if str(item)]
        if actuation_mode == "high_level_operation_compiler_required":
            blockers.append("inverse_action_operation_set_compiler_required")
        elif actuation_mode == "source_provenance_operation_set" and not matched_packet_ir:
            blockers.append("source_provenance_packet_ir_operation_set_missing")
        elif actuation_mode == "leaf_cell_candidate_explicit_opt_in":
            blockers.append("leaf_cell_candidate_probe_not_portfolio_actuator")
        links.append(
            {
                "schema": "inverse_steganalysis_water_bucket_materialization_bridge_link.v1",
                "atom_id": row.get("atom_id"),
                "source_index": row.get("source_index"),
                "actuation_mode": actuation_mode,
                "unit_ids": unit_ids,
                "matched_packet_ir_operation_set_ids": [
                    str(item.get("operation_set_id") or "")
                    for item in matched_packet_ir
                    if str(item.get("operation_set_id") or "")
                ],
                "matched_source_operation_set_ids": [
                    str(item.get("source_operation_set_id") or "")
                    for item in matched_packet_ir
                    if str(item.get("source_operation_set_id") or "")
                ],
                "packet_ir_lowering_ready": bool(matched_packet_ir),
                "queue_consumable": (
                    actuation_mode == "source_provenance_operation_set"
                    and bool(matched_packet_ir)
                ),
                "blockers": ordered_unique(blockers),
                **FALSE_AUTHORITY,
            }
        )
    return links


def build_inverse_action_materialization_bridge(
    plan_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract inverse-action water-bucket handoff state from a campaign plan."""

    if plan_payload.get("schema") != PLAN_SCHEMA:
        raise ByteShavingCampaignError(f"expected {PLAN_SCHEMA}")
    try:
        require_no_truthy_authority_fields(
            plan_payload,
            context="inverse_action_materialization_bridge",
        )
    except ValueError as exc:
        raise ByteShavingCampaignError(str(exc)) from exc

    portfolios = _inverse_action_materialization_portfolios(plan_payload)
    portfolio_rows = [
        dict(row)
        for portfolio in portfolios
        for row in _as_list(portfolio.get("rows"))
        if isinstance(row, Mapping)
    ]
    packet_ir_operation_sets = _mapping_rows(plan_payload, "packet_ir_operation_sets")
    portfolio_row_links = _inverse_action_portfolio_row_links(
        portfolio_rows,
        packet_ir_operation_sets,
    )
    queue_consumable_packet_ir_operation_set_ids = ordered_unique(
        str(packet_ir_id)
        for link in portfolio_row_links
        if link.get("queue_consumable") is True
        for packet_ir_id in _as_list(link.get("matched_packet_ir_operation_set_ids"))
    )
    actuation_mode_counts = Counter(
        str(row.get("actuation_mode") or "unknown")
        for row in portfolio_rows
    )
    operation_families = ordered_unique(
        [
            *(
                str(family)
                for row in portfolio_rows
                for family in _as_list(row.get("operation_families"))
            ),
            *(
                str(operation.get("operation_family") or "")
                for operation_set in packet_ir_operation_sets
                for operation in _as_list(operation_set.get("operations"))
                if isinstance(operation, Mapping)
            ),
        ]
    )
    target_kinds = ordered_unique(
        [
            *(
                str(target)
                for row in portfolio_rows
                for target in _as_list(row.get("target_kinds"))
            ),
            *(
                str(operation.get("target_kind") or "")
                for operation_set in packet_ir_operation_sets
                for operation in _as_list(operation_set.get("operations"))
                if isinstance(operation, Mapping)
            ),
        ]
    )
    representation_family_classes = ordered_unique(
        str(operation.get("representation_family_class") or "")
        for operation_set in packet_ir_operation_sets
        for operation in _as_list(operation_set.get("operations"))
        if isinstance(operation, Mapping)
    )
    compiler_required_count = actuation_mode_counts.get(
        "high_level_operation_compiler_required",
        0,
    )
    source_provenance_count = actuation_mode_counts.get(
        "source_provenance_operation_set",
        0,
    )
    leaf_cell_count = actuation_mode_counts.get(
        "leaf_cell_candidate_explicit_opt_in",
        0,
    )
    dispatch_blockers = [
        *BASE_BLOCKERS,
        *[str(item) for item in _as_list(plan_payload.get("dispatch_blockers"))],
    ]
    if not portfolios:
        dispatch_blockers.append("inverse_action_materialization_portfolio_missing")
    if compiler_required_count:
        dispatch_blockers.append(
            "inverse_action_operation_set_compiler_required_for_cells_without_source_provenance"
        )
    if leaf_cell_count:
        dispatch_blockers.append(
            "inverse_action_leaf_cell_candidates_are_probe_only_until_materialized"
        )
    if packet_ir_operation_sets:
        dispatch_blockers.append(
            "packet_ir_operation_sets_require_materializer_contexts_and_runtime_proofs"
        )
        next_gate = "build_byte_shaving_campaign_queue_packet_ir_lowering"
    else:
        dispatch_blockers.append(
            "packet_ir_operation_sets_missing_until_source_provenance_or_compiler"
        )
        next_gate = "inverse_action_operation_set_compiler"

    bridge = {
        "schema": INVERSE_ACTION_MATERIALIZATION_BRIDGE_SCHEMA,
        "source_schema": PLAN_SCHEMA,
        "generated_at_utc": _utc_now(),
        "campaign_id": plan_payload.get("campaign_id"),
        "candidate_id": plan_payload.get("candidate_id"),
        "lane_id": plan_payload.get("lane_id"),
        "source_paths": _as_list(plan_payload.get("source_paths")),
        "portfolio_count": len(portfolios),
        "portfolio_row_count": len(portfolio_rows),
        "water_bucket_materialization_portfolios": portfolios,
        "portfolio_row_bridge_links": portfolio_row_links,
        "queue_consumable_portfolio_row_count": sum(
            1 for link in portfolio_row_links if link["queue_consumable"]
        ),
        "queue_consumable_packet_ir_operation_set_count": len(
            queue_consumable_packet_ir_operation_set_ids
        ),
        "queue_consumable_packet_ir_operation_set_ids": (
            queue_consumable_packet_ir_operation_set_ids
        ),
        "actuation_mode_counts": {
            key: actuation_mode_counts[key] for key in sorted(actuation_mode_counts)
        },
        "high_level_operation_compiler_required_count": compiler_required_count,
        "source_provenance_operation_set_count": source_provenance_count,
        "leaf_cell_candidate_count": leaf_cell_count,
        "packet_ir_operation_set_count": len(packet_ir_operation_sets),
        "packet_ir_byte_closed_operation_count": sum(
            int(operation_set.get("byte_closed_operation_count") or 0)
            for operation_set in packet_ir_operation_sets
        ),
        "packet_ir_operation_sets": packet_ir_operation_sets,
        "operation_families": operation_families,
        "target_kinds": target_kinds,
        "representation_family_classes": representation_family_classes,
        "queue_consumption": {
            "next_gate": next_gate,
            "plan_queue_builder": "tools/build_byte_shaving_campaign_queue.py",
            "packet_ir_lowering_ready": bool(packet_ir_operation_sets),
            "compiler_required": bool(compiler_required_count),
            "requires_plan_path": True,
            "requires_materializer_contexts": bool(packet_ir_operation_sets),
        },
        "evidence_boundary": {
            "planning_only": True,
            "authority": "no_score_no_promotion_no_dispatch",
            "score_axis": "[planning-only inverse-steganalysis action]",
        },
        "dispatch_blockers": ordered_unique(dispatch_blockers),
        **FALSE_AUTHORITY,
    }
    return apply_proxy_evidence_boundary(
        bridge,
        dispatch_blockers=bridge["dispatch_blockers"],
    )


def build_signal_surface_from_candidate_queue(
    queue_payload: Mapping[str, Any],
    *,
    campaign_id: str = "candidate_queue_byte_shaving_surface",
) -> dict[str, Any]:
    """Extract coarse byte-shaving units from existing optimizer queue rows."""

    if queue_payload.get("schema") != "optimizer_candidate_queue_v1":
        raise ByteShavingCampaignError("expected optimizer_candidate_queue_v1")
    try:
        require_no_truthy_authority_fields(
            queue_payload,
            context="optimizer_candidate_queue_byte_shaving_surface",
        )
    except ValueError as exc:
        raise ByteShavingCampaignError(str(exc)) from exc
    units: list[dict[str, Any]] = []

    def queue_row_operations(row: Mapping[str, Any]) -> list[dict[str, Any]]:
        consumer_payload = row.get("consumer_payload")
        if not isinstance(consumer_payload, Mapping):
            consumer_payload = {}
        raw_operations = (
            row.get("operations") or row.get("selected_operations") or consumer_payload.get("selected_operations") or []
        )
        operations = [dict(item) for item in _as_list(raw_operations) if isinstance(item, Mapping)]
        operation_families = ordered_unique(str(item) for item in _as_list(row.get("operation_families")) if str(item))
        row_operation_family = str(
            row.get("operation_family") or row.get("operation") or (operation_families[0] if operation_families else "")
        ).strip()
        row_materializer = row.get("materializer")
        row_target_kind = row.get("target_kind") or row.get("materializer_target_kind")
        row_params = dict(_mapping(row.get("operation_params") or row.get("params") or row.get("op_params")))
        if not operations and (row_operation_family or row_materializer or row_target_kind):
            operations.append(
                {
                    "operation_id": row.get("operation_id") or row_operation_family or "queue_row_operation",
                    "operation_family": row_operation_family,
                }
            )
        for operation in operations:
            if row_materializer is not None and operation.get("materializer") is None:
                operation["materializer"] = row_materializer
            if row_target_kind is not None and operation.get("target_kind") is None:
                operation["target_kind"] = row_target_kind
            if row_params and not isinstance(operation.get("params"), Mapping):
                operation["params"] = row_params
            if not str(operation.get("operation_family") or "").strip() and row_operation_family:
                operation["operation_family"] = row_operation_family
        return operations

    for row in _as_list(queue_payload.get("top_k")):
        if not isinstance(row, Mapping):
            continue
        try:
            require_no_truthy_authority_fields(
                row,
                context="optimizer_candidate_queue_byte_shaving_row",
            )
        except ValueError as exc:
            raise ByteShavingCampaignError(str(exc)) from exc
        bytes_value = _finite_int(
            row.get("candidate_saved_bytes")
            or row.get("saved_bytes")
            or row.get("archive_bytes_saved")
            or row.get("source_saved_bytes")
        )
        if bytes_value is None or bytes_value <= 0:
            continue
        operations = queue_row_operations(row)
        operation_families = (
            row.get("operation_families")
            or ordered_unique(
                str(item.get("operation_family") or "")
                for item in operations
                if str(item.get("operation_family") or "")
            )
            or []
        )
        units.append(
            {
                "unit_id": str(row.get("candidate_id") or f"queue_row_{len(units)}"),
                "unit_kind": row.get("unit_kind") or "archive_section",
                "candidate_saved_bytes": bytes_value,
                "predicted_quality_score_cost": _finite_float(
                    row.get("predicted_quality_score_cost") or row.get("quality_cost_score")
                )
                or 0.0,
                "confidence": _finite_float(row.get("confidence")) or 0.5,
                "operation_families": operation_families,
                "operations": operations,
                "materializer": row.get("materializer"),
                "target_kind": row.get("target_kind") or row.get("materializer_target_kind"),
                "operation_params": row.get("operation_params") or row.get("params"),
                "score_axis": row.get("score_axis") or row.get("dominant_axis"),
                "evidence_grade": row.get("evidence_grade"),
                "evidence_semantics": row.get("evidence_semantics"),
                "source_paths": _as_list(row.get("source_paths")),
                "source_candidate_id": row.get("source_candidate_id"),
                "candidate_archive_sha256": row.get("candidate_archive_sha256") or row.get("archive_sha256"),
                "candidate_archive_bytes": row.get("candidate_archive_bytes") or row.get("archive_bytes"),
                "candidate_trust_region_blockers": _as_list(row.get("candidate_trust_region_blockers")),
                "local_axis": row.get("local_axis"),
                "target_axis": row.get("target_axis"),
                "local_score": row.get("local_score"),
                "projected_contest_score": row.get("projected_contest_score"),
                "conservative_projected_contest_score": row.get("conservative_projected_contest_score"),
                "master_gradient_signal": row.get("master_gradient_signal") or row.get("master_gradient_provenance"),
                "canonical_equation_provenance": row.get("canonical_equation_provenance"),
                "atom_ids": _as_list(row.get("atom_ids")),
                "blockers": row.get("dispatch_blockers") or [],
            }
        )
    if not units:
        raise ByteShavingCampaignError("optimizer queue has no rows with saved-byte estimates")
    return {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": campaign_id,
        "source_signal_refs": _as_list(queue_payload.get("source_signal_refs")),
        "auth_eval_refs": _as_list(queue_payload.get("auth_eval_refs")),
        "mlx_calibration_refs": _as_list(queue_payload.get("mlx_calibration_refs")),
        "scorer_response_refs": _as_list(queue_payload.get("scorer_response_refs")),
        "inverse_scorer_surface_refs": _as_list(queue_payload.get("inverse_scorer_surface_refs")),
        "units": units,
        **FALSE_AUTHORITY,
    }


def build_signal_surface_from_engineered_correction_targeting(
    targeting_payload: Mapping[str, Any],
    *,
    campaign_id: str = "engineered_correction_targeting_surface",
    max_targets: int | None = None,
    default_predicted_quality_score_delta: float = 0.0,
) -> dict[str, Any]:
    """Bridge legacy engineered-correction targeting into the modern planner.

    Engineered-correction targets spend correction bytes to buy quality. They
    are therefore modeled as zero-saved-byte planning units unless an upstream
    empirical row supplies an explicit score delta. This keeps the signal
    discoverable for combinations without pretending it is byte-saving or
    promotion-ready.
    """

    if targeting_payload.get("schema") != ENGINEERED_CORRECTION_TARGETING_SCHEMA:
        raise ByteShavingCampaignError(f"expected {ENGINEERED_CORRECTION_TARGETING_SCHEMA}")
    try:
        require_no_truthy_authority_fields(
            targeting_payload,
            context="engineered_correction_targeting_byte_shaving_surface",
        )
    except ValueError as exc:
        raise ByteShavingCampaignError(str(exc)) from exc
    if max_targets is not None and max_targets < 1:
        raise ByteShavingCampaignError("max_targets must be >= 1")
    default_delta = _finite_float(default_predicted_quality_score_delta)
    if default_delta is None:
        raise ByteShavingCampaignError("default_predicted_quality_score_delta must be finite")

    targets = [item for item in _as_list(targeting_payload.get("top_per_pair_targets")) if isinstance(item, Mapping)]
    if max_targets is not None:
        targets = targets[:max_targets]
    units: list[dict[str, Any]] = []
    archive_sha = targeting_payload.get("archive_sha256")
    confidence = _finite_float(targeting_payload.get("confidence"))
    for index, target in enumerate(targets):
        pair_index = _finite_int(target.get("pair_index"))
        byte_index = _finite_int(target.get("byte_index"))
        if pair_index is None or pair_index < 0 or byte_index is None or byte_index < 0:
            continue
        variance_rank = _finite_int(target.get("per_pair_variance_rank"))
        magnitude = _finite_float(target.get("per_pair_distortion_magnitude")) or 0.0
        unit_id = f"engineered_correction_pair{pair_index:04d}_byte{byte_index:08d}"
        operation = {
            "operation_id": "apply_engineered_correction",
            "operation_family": "apply_engineered_correction",
            "candidate_saved_bytes": 0,
            "predicted_quality_score_delta": default_delta,
            "materializer": "engineered_correction_sidecar_patch",
            "target_kind": "engineered_correction_target_v1",
            "params": {
                "pair_index": pair_index,
                "byte_index": byte_index,
                "per_pair_distortion_magnitude": magnitude,
                "per_pair_variance_rank": variance_rank,
                "archive_sha256": archive_sha,
            },
            "blockers": [
                "engineered_correction_target_requires_correction_synthesis",
                "engineered_correction_target_requires_readiness_audit",
                "engineered_correction_target_requires_runtime_consumption_proof",
                "engineered_correction_target_requires_exact_auth_eval",
            ],
        }
        units.append(
            {
                "unit_id": unit_id,
                "unit_kind": "correction_target",
                "source_index": index,
                "source_span": [byte_index, byte_index + 1],
                "candidate_saved_bytes": 0,
                "predicted_quality_score_delta": default_delta,
                "confidence": confidence if confidence is not None else 0.5,
                "operation_families": ["apply_engineered_correction"],
                "operations": [operation],
                "score_axis": targeting_payload.get("measurement_axis"),
                "evidence_grade": "[predicted]",
                "evidence_semantics": ("legacy_engineered_correction_targeting_subsumed_by_byte_shaving_surface"),
                "source_candidate_id": targeting_payload.get("consumer_id"),
                "master_gradient_signal": {
                    "archive_sha256": archive_sha,
                    "measurement_axis": targeting_payload.get("measurement_axis"),
                    "measurement_hardware": targeting_payload.get("measurement_hardware"),
                    "consumer_id": targeting_payload.get("consumer_id"),
                },
                "engineered_correction_signal": {
                    "schema": ENGINEERED_CORRECTION_TARGETING_SCHEMA,
                    "pair_index": pair_index,
                    "byte_index": byte_index,
                    "per_pair_distortion_magnitude": magnitude,
                    "per_pair_variance_rank": variance_rank,
                    "targets_per_pair": targeting_payload.get("targets_per_pair"),
                    "total_targets": targeting_payload.get("total_targets"),
                    "planning_role": "quality_spend_target_not_byte_savings",
                },
                "blockers": [
                    "engineered_correction_target_is_planning_only",
                    "requires_correction_artifact_before_local_patch",
                    "requires_engineered_correction_readiness_audit",
                    "requires_exact_auth_eval_before_score_claim",
                ],
            }
        )
    if not units:
        raise ByteShavingCampaignError("engineered correction targeting payload has no plannable targets")
    return {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": campaign_id,
        "source_signal_refs": [
            {
                "kind": "engineered_correction_targeting",
                "schema": targeting_payload.get("schema"),
                "consumer_id": targeting_payload.get("consumer_id"),
                "archive_sha256": archive_sha,
                "measurement_axis": targeting_payload.get("measurement_axis"),
                "measurement_hardware": targeting_payload.get("measurement_hardware"),
                "n_bytes": targeting_payload.get("n_bytes"),
                "n_pairs": targeting_payload.get("n_pairs"),
                "targets_per_pair": targeting_payload.get("targets_per_pair"),
                "total_targets": targeting_payload.get("total_targets"),
                "surface_unit_count": len(units),
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "units": units,
        **FALSE_AUTHORITY,
    }


def _contiguous_spans(indices: Sequence[int], *, max_span_bytes: int) -> list[tuple[int, int]]:
    if not indices:
        return []
    spans: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for raw in indices[1:]:
        idx = int(raw)
        if idx == prev + 1 and idx - start < max_span_bytes:
            prev = idx
            continue
        spans.append((start, prev + 1))
        start = prev = idx
    spans.append((start, prev + 1))
    return spans


def build_signal_surface_from_master_gradient_anchor(
    *,
    archive_sha256: str,
    repo_root: Path | str = ".",
    ledger_path: Path | str | None = None,
    axis: str | None = None,
    campaign_id: str = "master_gradient_byte_shaving_surface",
    low_sensitivity_quantile: float = 0.05,
    max_units: int = 32,
    max_span_bytes: int = 4096,
    quality_cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Build byte-range planning units from a canonical master-gradient anchor.

    This bridge intentionally emits planning-only byte-range opportunities, not
    packet mutations. Master-gradient coordinates must still be mapped through
    archive grammar/materializers, then checked with locality controls and exact
    auth eval before any score claim.
    """

    if not archive_sha256:
        raise ByteShavingCampaignError("archive_sha256 is required")
    if not (0.0 < low_sensitivity_quantile <= 1.0):
        raise ByteShavingCampaignError("low_sensitivity_quantile must be in (0, 1]")
    if max_units < 1:
        raise ByteShavingCampaignError("max_units must be >= 1")
    if max_span_bytes < 1:
        raise ByteShavingCampaignError("max_span_bytes must be >= 1")
    if quality_cost_multiplier < 0.0 or not math.isfinite(quality_cost_multiplier):
        raise ByteShavingCampaignError("quality_cost_multiplier must be finite and >= 0")

    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ByteShavingCampaignError("numpy is required for master-gradient surfaces") from exc

    from tac.master_gradient import (  # local import keeps generic planner light
        latest_anchor_for_archive,
    )

    repo = Path(repo_root)
    ledger = repo / ".omx/state/master_gradient_anchors.jsonl" if ledger_path is None else Path(ledger_path)
    if ledger is not None and not ledger.is_absolute():
        ledger = repo / ledger
    anchor = latest_anchor_for_archive(archive_sha256, path=ledger, axis=axis)
    if anchor is None:
        axis_text = f" axis={axis!r}" if axis is not None else ""
        raise ByteShavingCampaignError(
            f"no usable master-gradient planning anchor for archive_sha256={archive_sha256!r}{axis_text}"
        )
    npy_path = _resolve_repo_path(anchor.get("gradient_array_path"), repo)
    if npy_path is None or not npy_path.is_file():
        raise ByteShavingCampaignError(
            f"master-gradient anchor missing readable gradient_array_path: {anchor.get('gradient_array_path')!r}"
        )
    arr = np.load(npy_path)
    if arr.ndim == 2 and arr.shape[1] == 3:
        sensitivity = np.abs(arr).sum(axis=1).astype(np.float64)
        tensor_kind = "aggregate_per_byte_v1"
    elif arr.ndim == 3 and arr.shape[2] == 3:
        sensitivity = np.abs(arr).sum(axis=2).mean(axis=1).astype(np.float64)
        tensor_kind = "per_pair_per_byte_v1_mean_abs"
    else:
        raise ByteShavingCampaignError(f"master-gradient array must have shape (N, 3) or (N, P, 3); got {arr.shape}")
    if sensitivity.size == 0:
        raise ByteShavingCampaignError("master-gradient array is empty")
    declared_n = _finite_int(anchor.get("n_bytes"))
    if declared_n is not None and declared_n != int(sensitivity.size):
        raise ByteShavingCampaignError(
            f"anchor n_bytes={declared_n} but gradient array has {int(sensitivity.size)} bytes"
        )

    threshold = float(np.quantile(sensitivity, low_sensitivity_quantile))
    candidate_indices = np.flatnonzero(sensitivity <= threshold).astype(int).tolist()
    spans = _contiguous_spans(candidate_indices, max_span_bytes=max_span_bytes)
    span_rows: list[dict[str, Any]] = []
    for start, end in spans:
        span_sensitivity = sensitivity[start:end]
        if span_sensitivity.size == 0:
            continue
        sensitivity_sum = float(span_sensitivity.sum())
        length = end - start
        quality_cost = max(0.0, sensitivity_sum * quality_cost_multiplier)
        span_rows.append(
            {
                "start": start,
                "end": end,
                "length": length,
                "mean_sensitivity": float(span_sensitivity.mean()),
                "max_sensitivity": float(span_sensitivity.max()),
                "sensitivity_sum": sensitivity_sum,
                "quality_cost": quality_cost,
            }
        )
    span_rows.sort(
        key=lambda row: (
            float(row["quality_cost"]),
            float(row["mean_sensitivity"]),
            -int(row["length"]),
            int(row["start"]),
        )
    )
    blockers = [
        "master_gradient_byte_ranges_are_planning_coordinates_only",
        "requires_archive_grammar_mapping_before_materialization",
        "master_gradient_quality_cost_is_first_order_proxy",
        "requires_locality_control_before_exact_auth_eval",
        "requires_exact_auth_eval_before_score_claim",
    ]
    units: list[dict[str, Any]] = []
    for row in span_rows[:max_units]:
        start = int(row["start"])
        end = int(row["end"])
        saved_bytes = int(row["length"])
        quality_cost = float(row["quality_cost"])
        mean_sensitivity = float(row["mean_sensitivity"])
        units.append(
            {
                "unit_id": f"mg_byte_span_{start:07d}_{end:07d}",
                "unit_kind": "byte_range",
                "source_index": start,
                "source_span": {"start": start, "end_exclusive": end},
                "candidate_saved_bytes": saved_bytes,
                "predicted_quality_score_cost": quality_cost,
                "confidence": 0.55 if mean_sensitivity == 0.0 else 0.35,
                "operation_families": [
                    "null_remove_or_seed",
                    "entropy_recode",
                    "delta_encode",
                ],
                "operations": [
                    {
                        "operation_id": "null_remove_or_seed",
                        "operation_family": "null_remove_or_seed",
                        "candidate_saved_bytes": saved_bytes,
                        "predicted_quality_score_cost": quality_cost,
                        "blockers": blockers,
                    },
                    {
                        "operation_id": "entropy_recode",
                        "operation_family": "entropy_recode",
                        "candidate_saved_bytes": saved_bytes,
                        "predicted_quality_score_cost": quality_cost,
                        "blockers": blockers,
                    },
                    {
                        "operation_id": "delta_encode",
                        "operation_family": "delta_encode",
                        "candidate_saved_bytes": max(1, saved_bytes // 2),
                        "predicted_quality_score_cost": quality_cost * 0.5,
                        "blockers": blockers,
                    },
                ],
                "master_gradient_score_cost": quality_cost,
                "master_gradient_signal": {
                    "archive_sha256": archive_sha256,
                    "measurement_axis": anchor.get("measurement_axis"),
                    "measurement_hardware": anchor.get("measurement_hardware"),
                    "measurement_call_id": anchor.get("measurement_call_id"),
                    "gradient_tensor_kind": anchor.get("gradient_tensor_kind") or tensor_kind,
                    "gradient_array_path": _repo_rel(npy_path, repo),
                    "low_sensitivity_quantile": low_sensitivity_quantile,
                    "threshold": threshold,
                    "span_sensitivity_sum": float(row["sensitivity_sum"]),
                    "span_mean_sensitivity": mean_sensitivity,
                    "span_max_sensitivity": float(row["max_sensitivity"]),
                    "quality_cost_multiplier": quality_cost_multiplier,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "score_axis": anchor.get("measurement_axis"),
                "blockers": blockers,
            }
        )
    if not units:
        raise ByteShavingCampaignError("master-gradient anchor produced no byte-range units")
    return {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": campaign_id,
        "candidate_id": f"{archive_sha256[:12]}_master_gradient_low_sensitivity",
        "lane_id": "master_gradient_byte_shaving_planning",
        "frontier_axis": anchor.get("measurement_axis") or "[planning-only]",
        "source_signal_refs": [
            {
                "kind": "master_gradient_anchor",
                "archive_sha256": archive_sha256,
                "measurement_axis": anchor.get("measurement_axis"),
                "measurement_hardware": anchor.get("measurement_hardware"),
                "measurement_call_id": anchor.get("measurement_call_id"),
                "gradient_array_path": _repo_rel(npy_path, repo),
                "ledger_path": _repo_rel(ledger, repo),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "surface_parameters": {
            "low_sensitivity_quantile": low_sensitivity_quantile,
            "max_units": max_units,
            "max_span_bytes": max_span_bytes,
            "quality_cost_multiplier": quality_cost_multiplier,
            "sensitivity_threshold": threshold,
            "candidate_span_count_before_cap": len(span_rows),
        },
        "units": units,
        **FALSE_AUTHORITY,
    }


def build_signal_surface_from_inverse_action_functional(
    action_payload: Mapping[str, Any],
    *,
    campaign_id: str = "inverse_steganalysis_action_byte_shaving_surface",
    allow_leaf_cell_candidates: bool = False,
) -> dict[str, Any]:
    """Bridge inverse-action water buckets into the byte-shaving planner."""

    if action_payload.get("schema") != INVERSE_ACTION_FUNCTIONAL_SCHEMA:
        raise ByteShavingCampaignError(f"expected {INVERSE_ACTION_FUNCTIONAL_SCHEMA}")
    try:
        require_no_truthy_authority_fields(
            action_payload,
            context="inverse_action_functional_byte_shaving_surface",
        )
    except ValueError as exc:
        raise ByteShavingCampaignError(str(exc)) from exc
    water_bucket = _mapping(action_payload.get("water_bucket"))
    selected_cells = [item for item in _as_list(water_bucket.get("selected_cells")) if isinstance(item, Mapping)]
    full_cells_by_atom = {
        str(item.get("atom_id") or ""): item
        for item in _as_list(action_payload.get("cells"))
        if isinstance(item, Mapping)
    }
    units: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    for index, selected in enumerate(selected_cells):
        atom_id = str(selected.get("atom_id") or f"inverse_action_cell_{index}")
        full_cell = full_cells_by_atom.get(atom_id, {})
        expected_gain = _finite_float(selected.get("expected_score_gain")) or 0.0
        if expected_gain <= 0.0:
            continue
        provenance_units = _units_from_inverse_action_source_provenance(
            atom_id=atom_id,
            selected=selected,
            full_cell=full_cell,
            source_index=index,
            expected_gain=expected_gain,
        )
        if provenance_units:
            units.extend(provenance_units)
            portfolio_rows.append(
                _inverse_action_portfolio_row(
                    selected=selected,
                    full_cell=full_cell,
                    atom_id=atom_id,
                    source_index=index,
                    expected_gain=expected_gain,
                    actuation_mode="source_provenance_operation_set",
                    unit_ids=[str(unit["unit_id"]) for unit in provenance_units],
                    operation_families=[
                        str(unit.get("operation_families", [""])[0])
                        for unit in provenance_units
                    ],
                    target_kinds=[
                        str(
                            _as_list(unit.get("operations"))[0].get("target_kind")
                        )
                        for unit in provenance_units
                        if _as_list(unit.get("operations"))
                        and isinstance(_as_list(unit.get("operations"))[0], Mapping)
                        and _as_list(unit.get("operations"))[0].get("target_kind")
                    ],
                    blockers=[
                        "requires_materializer_queue_execution",
                        "requires_runtime_consumption_proof_before_exact_eval",
                        "requires_exact_auth_eval_before_score_claim",
                    ],
                )
            )
            continue
        if not allow_leaf_cell_candidates:
            units.append(
                _high_level_operation_compiler_gap_unit(
                    atom_id=atom_id,
                    selected=selected,
                    full_cell=full_cell,
                    source_index=index,
                    expected_gain=expected_gain,
                )
            )
            portfolio_rows.append(
                _inverse_action_portfolio_row(
                    selected=selected,
                    full_cell=full_cell,
                    atom_id=atom_id,
                    source_index=index,
                    expected_gain=expected_gain,
                    actuation_mode="high_level_operation_compiler_required",
                    unit_ids=[f"inverse_action_{atom_id}"],
                    operation_families=[INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY],
                    target_kinds=[INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND],
                    blockers=[
                        "inverse_action_cell_lacks_source_operation_provenance",
                        "requires_candidate_family_operation_compiler",
                        "requires_archive_runtime_materializer_before_candidate",
                        "requires_exact_auth_eval_before_score_claim",
                    ],
                )
            )
            continue
        operation = {
            "operation_id": "materialize_inverse_scorer_cell_candidate",
            "operation_family": "materialize_inverse_scorer_cell_candidate",
            "candidate_saved_bytes": 0,
            "predicted_quality_score_delta": -expected_gain,
            "materializer": "inverse_scorer_cell_candidate_adapter",
            "target_kind": "inverse_scorer_cell_candidate_v1",
            "params": {
                "atom_id": atom_id,
                "candidate_id": selected.get("candidate_id"),
                "scope_axis": selected.get("scope_axis"),
                "component": selected.get("component"),
                "euler_lagrange_residual": selected.get("euler_lagrange_residual"),
                "water_fill_cost_bytes": selected.get("water_fill_cost_bytes"),
                "water_fill_cost_bytes_semantics": (
                    "planner_budget_cost_not_serialized_savings"
                ),
            },
            "blockers": [
                "inverse_action_cell_requires_deterministic_materializer",
                "inverse_action_cell_requires_runtime_consumption_proof",
                "inverse_action_cell_requires_exact_auth_eval_before_score_claim",
            ],
        }
        units.append(
            {
                "unit_id": f"inverse_action_{atom_id}",
                "unit_kind": "scorer_inverse_surface_cell",
                "source_index": index,
                "candidate_saved_bytes": 0,
                "predicted_quality_score_delta": -expected_gain,
                "confidence": 0.5,
                "operation_families": ["materialize_inverse_scorer_cell_candidate"],
                "operations": [operation],
                "score_axis": selected.get("component") or full_cell.get("component"),
                "evidence_grade": "[planning-only inverse-steganalysis action]",
                "evidence_semantics": "discrete_action_water_bucket_selection",
                "source_candidate_id": selected.get("candidate_id"),
                "atom_ids": [atom_id],
                "inverse_action_cell": {
                    "euler_lagrange_residual": selected.get("euler_lagrange_residual"),
                    "water_fill_cost_bytes": selected.get("water_fill_cost_bytes"),
                    "water_fill_cost_bytes_semantics": (
                        "planner_budget_cost_not_serialized_savings"
                    ),
                    "scope_axis": selected.get("scope_axis"),
                    "component": selected.get("component"),
                },
                "blockers": [
                    "inverse_action_unit_is_planning_only",
                    "inverse_action_leaf_cell_candidate_explicitly_enabled",
                    "requires_inverse_scorer_cell_materializer",
                    "requires_byte_closed_archive_before_dispatch",
                    "requires_exact_auth_eval_before_score_claim",
                ],
            }
        )
        portfolio_rows.append(
            _inverse_action_portfolio_row(
                selected=selected,
                full_cell=full_cell,
                atom_id=atom_id,
                source_index=index,
                expected_gain=expected_gain,
                actuation_mode="leaf_cell_candidate_explicit_opt_in",
                unit_ids=[f"inverse_action_{atom_id}"],
                operation_families=["materialize_inverse_scorer_cell_candidate"],
                target_kinds=["inverse_scorer_cell_candidate_v1"],
                blockers=[
                    "leaf_cell_candidate_is_probe_not_portfolio_actuator",
                    "requires_runtime_consumption_proof_before_exact_eval",
                    "requires_exact_auth_eval_before_score_claim",
                ],
            )
        )
    if not units:
        raise ByteShavingCampaignError("inverse action functional produced no selected cells")
    return {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": campaign_id,
        "candidate_id": water_bucket.get("schema") or "inverse_steganalysis_action",
        "lane_id": "inverse_steganalysis_action_byte_shaving",
        "frontier_axis": "[planning-only inverse-steganalysis action]",
        "source_signal_refs": [
            {
                "kind": "inverse_steganalysis_action_functional",
                "schema": action_payload.get("schema"),
                "water_bucket_materialization_portfolio_schema": (
                    INVERSE_ACTION_WATER_BUCKET_PORTFOLIO_SCHEMA
                ),
                "cell_count": _mapping(action_payload.get("integral_totals")).get("cell_count"),
                "selected_count": water_bucket.get("selected_count"),
                "selected_expected_score_gain": water_bucket.get("selected_expected_score_gain"),
                "source_provenance_operation_set_count": sum(
                    1
                    for row in portfolio_rows
                    if row["actuation_mode"] == "source_provenance_operation_set"
                ),
                "high_level_operation_compiler_required_count": sum(
                    1
                    for row in portfolio_rows
                    if row["actuation_mode"]
                    == "high_level_operation_compiler_required"
                ),
                "leaf_cell_candidate_count": sum(
                    1
                    for row in portfolio_rows
                    if row["actuation_mode"] == "leaf_cell_candidate_explicit_opt_in"
                ),
                **FALSE_AUTHORITY,
            }
        ],
        "water_bucket_materialization_portfolio": {
            "schema": INVERSE_ACTION_WATER_BUCKET_PORTFOLIO_SCHEMA,
            "selected_cell_count": len(selected_cells),
            "portfolio_row_count": len(portfolio_rows),
            "actuation_modes": ordered_unique(
                str(row.get("actuation_mode")) for row in portfolio_rows
            ),
            "rows": portfolio_rows,
            **FALSE_AUTHORITY,
        },
        "units": units,
        "blockers": _as_list(action_payload.get("dispatch_blockers")),
        **FALSE_AUTHORITY,
    }


def _inverse_action_portfolio_row(
    *,
    selected: Mapping[str, Any],
    full_cell: Mapping[str, Any],
    atom_id: str,
    source_index: int,
    expected_gain: float,
    actuation_mode: str,
    unit_ids: Sequence[str],
    operation_families: Sequence[str],
    target_kinds: Sequence[str],
    blockers: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": "inverse_steganalysis_water_bucket_materialization_portfolio_row.v1",
        "source_index": source_index,
        "atom_id": atom_id,
        "candidate_id": selected.get("candidate_id"),
        "scope_axis": selected.get("scope_axis") or full_cell.get("scope_axis"),
        "component": selected.get("component") or full_cell.get("component"),
        "expected_score_gain": expected_gain,
        "euler_lagrange_residual": selected.get("euler_lagrange_residual"),
        "water_fill_cost_bytes": selected.get("water_fill_cost_bytes"),
        "water_fill_cost_bytes_semantics": (
            "planner_budget_cost_not_serialized_savings"
        ),
        "actuation_mode": actuation_mode,
        "unit_ids": ordered_unique(str(item) for item in unit_ids),
        "operation_families": ordered_unique(str(item) for item in operation_families),
        "target_kinds": ordered_unique(str(item) for item in target_kinds),
        "blockers": ordered_unique(str(item) for item in blockers),
        **FALSE_AUTHORITY,
    }


def _high_level_operation_compiler_gap_unit(
    *,
    atom_id: str,
    selected: Mapping[str, Any],
    full_cell: Mapping[str, Any],
    source_index: int,
    expected_gain: float,
) -> dict[str, Any]:
    operation = {
        "operation_id": INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY,
        "operation_family": INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY,
        "candidate_saved_bytes": 0,
        "predicted_quality_score_delta": -expected_gain,
        "materializer": INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER,
        "target_kind": INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND,
        "params": {
            "atom_id": atom_id,
            "candidate_id": selected.get("candidate_id"),
            "scope_axis": selected.get("scope_axis") or full_cell.get("scope_axis"),
            "component": selected.get("component") or full_cell.get("component"),
            "euler_lagrange_residual": selected.get("euler_lagrange_residual"),
            "water_fill_cost_bytes": selected.get("water_fill_cost_bytes"),
            "water_fill_cost_bytes_semantics": (
                "planner_budget_cost_not_serialized_savings"
            ),
        },
        "blockers": [
            "inverse_action_cell_lacks_source_operation_provenance",
            "requires_candidate_family_operation_compiler",
            "requires_archive_runtime_materializer_before_candidate",
            "requires_exact_auth_eval_before_score_claim",
        ],
    }
    return {
        "unit_id": f"inverse_action_{atom_id}",
        "unit_kind": "scorer_inverse_surface_cell",
        "source_index": source_index,
        "candidate_saved_bytes": 0,
        "predicted_quality_score_delta": -expected_gain,
        "confidence": 0.5,
        "operation_families": [INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY],
        "operations": [operation],
        "score_axis": selected.get("component") or full_cell.get("component"),
        "evidence_grade": "[planning-only inverse-steganalysis action]",
        "evidence_semantics": "water_bucket_high_level_operation_compiler_gap",
        "source_candidate_id": selected.get("candidate_id"),
        "atom_ids": [atom_id],
        "inverse_action_cell": {
            "euler_lagrange_residual": selected.get("euler_lagrange_residual"),
            "water_fill_cost_bytes": selected.get("water_fill_cost_bytes"),
            "water_fill_cost_bytes_semantics": (
                "planner_budget_cost_not_serialized_savings"
            ),
            "scope_axis": selected.get("scope_axis"),
            "component": selected.get("component"),
            "actuation_mode": "high_level_operation_compiler_required",
        },
        "blockers": [
            "inverse_action_unit_is_planning_only",
            "inverse_action_cell_lacks_source_operation_provenance",
            "requires_candidate_family_operation_compiler",
            "requires_byte_closed_archive_before_dispatch",
            "requires_exact_auth_eval_before_score_claim",
        ],
    }


def _units_from_inverse_action_source_provenance(
    *,
    atom_id: str,
    selected: Mapping[str, Any],
    full_cell: Mapping[str, Any],
    source_index: int,
    expected_gain: float,
) -> list[dict[str, Any]]:
    provenance = _mapping(full_cell.get("source_provenance"))
    if provenance.get("schema") not in {
        BYTE_SHAVING_OPERATION_SET_PROVENANCE_SCHEMA,
        MLX_ACQUISITION_BATCH_OPERATION_SET_PROVENANCE_SCHEMA,
    }:
        return []
    operations = [
        dict(item)
        for item in _as_list(provenance.get("selected_operations"))
        if isinstance(item, Mapping)
    ]
    if not operations:
        return []
    total_saved = _finite_int(provenance.get("candidate_saved_bytes"))
    if total_saved is None:
        total_saved = _finite_int(selected.get("water_fill_cost_bytes")) or 0
    unit_count = len(operations)
    units: list[dict[str, Any]] = []
    for op_index, operation in enumerate(operations):
        units.append(
            _unit_from_inverse_action_operation_provenance(
                operation,
                provenance=provenance,
                atom_id=atom_id,
                selected=selected,
                source_index=source_index,
                op_index=op_index,
                operation_count=unit_count,
                total_saved=max(0, total_saved),
                expected_gain=expected_gain,
            )
        )
    return units


def _unit_kind_from_operation(operation: Mapping[str, Any]) -> str:
    unit_kind = str(operation.get("unit_kind") or "").strip()
    if unit_kind in UNIT_KINDS:
        return unit_kind
    return "scorer_response_row"


def _distributed_saved_bytes(
    operation: Mapping[str, Any],
    *,
    op_index: int,
    operation_count: int,
    total_saved: int,
    unit_kind: str,
) -> int:
    saved = _finite_int(operation.get("candidate_saved_bytes"))
    if saved is not None and saved >= 0:
        return saved
    base = total_saved // max(1, operation_count)
    if op_index < total_saved % max(1, operation_count):
        base += 1
    if base > 0 or unit_kind in {"scorer_response_row", "scorer_inverse_surface_cell"}:
        return base
    return 1


def _operation_quality_delta_from_provenance(
    operation: Mapping[str, Any],
    *,
    expected_gain: float,
    operation_count: int,
) -> float:
    for key in (
        "predicted_quality_score_delta",
        "quality_delta_score",
        "predicted_non_rate_delta_score",
        "quality_cost_score",
        "predicted_quality_score_cost",
    ):
        value = _finite_float(operation.get(key))
        if value is not None:
            return value
    return -expected_gain / float(max(1, operation_count))


def _unit_from_inverse_action_operation_provenance(
    operation: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    atom_id: str,
    selected: Mapping[str, Any],
    source_index: int,
    op_index: int,
    operation_count: int,
    total_saved: int,
    expected_gain: float,
) -> dict[str, Any]:
    unit_kind = _unit_kind_from_operation(operation)
    saved_bytes = _distributed_saved_bytes(
        operation,
        op_index=op_index,
        operation_count=operation_count,
        total_saved=total_saved,
        unit_kind=unit_kind,
    )
    operation_family = str(
        operation.get("operation_family")
        or operation.get("family")
        or "materialize_scorer_response_candidate"
    )
    source_unit_id = str(operation.get("unit_id") or f"op{op_index:04d}")
    operation_id = str(operation.get("operation_id") or f"{operation_family}_{op_index}")
    metadata = {
        **_operation_metadata(provenance),
        **_operation_metadata(operation),
    }
    params = {
        **dict(_mapping(operation.get("params"))),
        "inverse_action_atom_id": atom_id,
        "inverse_action_source_schema": provenance.get("schema"),
        "inverse_action_operation_set_id": provenance.get("operation_set_id"),
        "water_fill_cost_bytes": selected.get("water_fill_cost_bytes"),
        "water_fill_cost_bytes_semantics": (
            "planner_budget_cost_not_serialized_savings"
        ),
    }
    blockers = ordered_unique(
        [
            *[str(item) for item in _as_list(operation.get("blockers"))],
            "inverse_action_rehydrated_operation_is_planning_only",
            "requires_byte_closed_archive_before_dispatch",
            "requires_runtime_consumption_proof_before_exact_eval",
            "requires_exact_auth_eval_before_score_claim",
        ]
    )
    surface_operation = {
        "operation_id": operation_id,
        "operation_family": operation_family,
        "candidate_saved_bytes": saved_bytes,
        "predicted_quality_score_delta": _operation_quality_delta_from_provenance(
            operation,
            expected_gain=expected_gain,
            operation_count=operation_count,
        ),
        "materializer": operation.get("materializer"),
        "target_kind": operation.get("target_kind"),
        "params": params,
        **metadata,
        "blockers": blockers,
    }
    return {
        "unit_id": f"inverse_action_{atom_id}_{source_unit_id}_{op_index:04d}",
        "unit_kind": unit_kind,
        "source_index": source_index,
        "candidate_saved_bytes": saved_bytes,
        "predicted_quality_score_delta": surface_operation[
            "predicted_quality_score_delta"
        ],
        "confidence": 0.5,
        "operation_families": [operation_family],
        "operations": [surface_operation],
        "score_axis": selected.get("component") or "inverse_action",
        "evidence_grade": "[planning-only inverse-steganalysis action]",
        "evidence_semantics": "rehydrated_action_source_provenance",
        "source_candidate_id": selected.get("candidate_id"),
        "atom_ids": [atom_id],
        "source_provenance_schema": provenance.get("schema"),
        **metadata,
        "inverse_action_cell": {
            "euler_lagrange_residual": selected.get("euler_lagrange_residual"),
            "water_fill_cost_bytes": selected.get("water_fill_cost_bytes"),
            "water_fill_cost_bytes_semantics": (
                "planner_budget_cost_not_serialized_savings"
            ),
            "scope_axis": selected.get("scope_axis"),
            "component": selected.get("component"),
        },
        "blockers": [
            "inverse_action_unit_is_planning_only",
            "inverse_action_source_operations_rehydrated_from_provenance",
            "requires_byte_closed_archive_before_dispatch",
            "requires_exact_auth_eval_before_score_claim",
        ],
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "BASE_BLOCKERS",
    "COUPLED_OPERATION_SET_SCHEMA",
    "FALSE_AUTHORITY",
    "INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER",
    "INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY",
    "INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND",
    "INVERSE_ACTION_MATERIALIZATION_BRIDGE_SCHEMA",
    "INVERSE_ACTION_WATER_BUCKET_PORTFOLIO_SCHEMA",
    "PACKET_IR_OPERATION_SCHEMA",
    "PLAN_SCHEMA",
    "SIGNAL_SURFACE_SCHEMA",
    "ByteShavingCampaignError",
    "build_byte_shaving_campaign_plan",
    "build_inverse_action_materialization_bridge",
    "build_signal_surface_from_candidate_queue",
    "build_signal_surface_from_engineered_correction_targeting",
    "build_signal_surface_from_inverse_action_functional",
    "build_signal_surface_from_master_gradient_anchor",
    "normalize_unit_signal",
    "validate_signal_surface",
]
