# SPDX-License-Identifier: MIT
"""Score repair-waterfill typed ledgers for local campaign execution.

The scorer is intentionally false-authority: it ranks encoder-side repair
work for bounded local MLX follow-up, and it names the exact custody artifacts
missing before any budget spend, promotion, or exact dispatch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)

REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA = "repair_campaign_score_report.v1"
REPAIR_CAMPAIGN_SCORE_ROW_SCHEMA = "repair_campaign_score_row.v1"
REPAIR_OPERATOR_FAMILY_PRIORS_SCHEMA = "repair_operator_family_priors.v1"
REPAIR_OPERATOR_FAMILY_PRIOR_ROW_SCHEMA = "repair_operator_family_prior_row.v1"

_TYPED_LEDGER_SCHEMA = "frontier_rate_attack_repair_budget_typed_response_ledger.v1"
_TYPED_ROW_SCHEMA = "frontier_rate_attack_repair_budget_typed_response_row.v1"
_WORK_ORDER_SCHEMA = "frontier_rate_attack_repair_budget_waterfill_work_order.v1"

_ENTROPY_POSITION_WEIGHTS: dict[str, float] = {
    "before_entropy_coder_distribution_shaping": 1.20,
    "scorer_entropy_repair_before_selector_codec": 1.15,
    "at_entropy_coder_integer_codeword_boundary": 1.05,
    "at_entropy_coder": 1.00,
    "selector_codec_entropy": 0.85,
    "after_entropy_coder_container_or_zip_grammar": 0.45,
    "unknown_entropy_pipeline_position": 0.20,
}


class RepairCampaignScorerError(ValueError):
    """Raised when a repair campaign payload cannot be scored."""


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _safe_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _repo_path_exists(path_text: str, repo_root: str | Path | None) -> bool:
    if not path_text:
        return False
    path = Path(path_text)
    if not path.is_absolute() and repo_root is not None:
        path = Path(repo_root) / path
    return path.exists()


def repair_operator_family_priors() -> dict[str, Any]:
    """Return first-class repair families the scorer understands."""

    rows = [
        {
            "schema": REPAIR_OPERATOR_FAMILY_PRIOR_ROW_SCHEMA,
            "family_id": "posenet_null_bottom_decile",
            "aliases": [
                "posenet_null_bottom_decile",
                "posenet-bottom-decile",
                "P19",
            ],
            "targeted_dimensions": ["posenet", "pair", "frame0"],
            "entropy_position_label": "before_entropy_coder_distribution_shaping",
            "campaign_prior_multiplier": 1.18,
            "required_local_artifacts": [
                "local_mlx_response_path",
                "reference_local_mlx_response_path",
                "posenet_null_bottom_decile_pair_ids",
            ],
            "missing_artifact_label": "posenet_null_bottom_decile_mlx_probe_missing",
            "stackability_role": "pose_repair_in_segnet_nullspace",
            **FALSE_AUTHORITY,
        },
        {
            "schema": REPAIR_OPERATOR_FAMILY_PRIOR_ROW_SCHEMA,
            "family_id": "segnet_class_region_waterfill",
            "aliases": [
                "segnet_class_region_waterfill",
                "segnet-class-region",
                "P18",
            ],
            "targeted_dimensions": ["segnet", "region", "frame1"],
            "entropy_position_label": "before_entropy_coder_distribution_shaping",
            "campaign_prior_multiplier": 1.22,
            "required_local_artifacts": [
                "local_mlx_response_path",
                "reference_local_mlx_response_path",
                "segnet_class_region_mask_ids",
            ],
            "missing_artifact_label": "segnet_class_region_mlx_probe_missing",
            "stackability_role": "segnet_margin_waterfill_before_selector_codec",
            **FALSE_AUTHORITY,
        },
        {
            "schema": REPAIR_OPERATOR_FAMILY_PRIOR_ROW_SCHEMA,
            "family_id": "per_region_selector_codec",
            "aliases": ["per_region_selector_codec", "per-region-selector", "P11"],
            "targeted_dimensions": ["selector_stream", "region"],
            "entropy_position_label": "selector_codec_entropy",
            "campaign_prior_multiplier": 0.55,
            "required_local_artifacts": [
                "selector_payload_bits_per_region",
                "receiver_consumed_runtime_replay_proof",
            ],
            "missing_artifact_label": "per_region_selector_codec_replay_missing",
            "stackability_role": "selector_payload_for_measured_region_decisions",
            **FALSE_AUTHORITY,
        },
        {
            "schema": REPAIR_OPERATOR_FAMILY_PRIOR_ROW_SCHEMA,
            "family_id": "palette_frame_asymmetry_prior",
            "aliases": [
                "palette_frame_asymmetry_prior",
                "repair_dynamics_frame0_palette_interaction_waterfill",
                "frame0_palette",
            ],
            "targeted_dimensions": ["palette", "frame0", "posenet"],
            "entropy_position_label": "before_entropy_coder_distribution_shaping",
            "campaign_prior_multiplier": 1.10,
            "required_local_artifacts": [
                "local_mlx_response_path",
                "repair_dynamics_palette_probe_matrix_path",
            ],
            "missing_artifact_label": "palette_frame_asymmetry_probe_missing",
            "stackability_role": "frame0_pose_repair_using_canonical_k16_palette",
            **FALSE_AUTHORITY,
        },
        {
            "schema": REPAIR_OPERATOR_FAMILY_PRIOR_ROW_SCHEMA,
            "family_id": "entropy_position_cascade",
            "aliases": ["entropy_position_cascade", "cascade_c", "Cascade C"],
            "targeted_dimensions": ["segnet", "posenet", "selector_stream"],
            "entropy_position_label": "scorer_entropy_repair_before_selector_codec",
            "campaign_prior_multiplier": 1.12,
            "required_local_artifacts": [
                "posenet_null_bottom_decile_pair_ids",
                "segnet_class_region_mask_ids",
                "selector_payload_bits_per_region",
            ],
            "missing_artifact_label": "entropy_position_cascade_probe_missing",
            "stackability_role": "compose_scorer_entropy_repair_before_codec_replay",
            **FALSE_AUTHORITY,
        },
    ]
    payload = {
        "schema": REPAIR_OPERATOR_FAMILY_PRIORS_SCHEMA,
        "row_schema": REPAIR_OPERATOR_FAMILY_PRIOR_ROW_SCHEMA,
        "row_count": len(rows),
        "rows": rows,
        "allowed_use": "repair_campaign_scoring_prior_only",
        "forbidden_use": "score_claim_or_dispatch_or_budget_spend_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(payload, context="repair_operator_family_priors")
    return payload


def _family_prior(row: Mapping[str, Any]) -> Mapping[str, Any]:
    corpus = " ".join(
        [
            str(row.get("correction_family") or ""),
            str(row.get("candidate_id") or ""),
            " ".join(_string_list(row.get("targeted_dimensions"))),
            " ".join(_string_list(row.get("operation_levels"))),
        ]
    ).lower()
    for prior in repair_operator_family_priors()["rows"]:
        aliases = [alias.lower() for alias in _string_list(prior.get("aliases"))]
        if any(alias and alias in corpus for alias in aliases):
            return prior
    return {}


def _objective_delta(row: Mapping[str, Any]) -> float | None:
    direct = _safe_float(row.get("objective_delta_score_units"))
    if direct is not None:
        return direct
    curves = _mapping(row.get("marginal_response_curves"))
    objective = _mapping(curves.get("objective"))
    from_curve = _safe_float(objective.get("delta_score_units"))
    if from_curve is not None:
        return from_curve
    return _safe_float(row.get("measured_lagrangian_delta_score_units"))


def _requested_bytes(row: Mapping[str, Any]) -> int:
    requested = _safe_int(row.get("requested_repair_bytes"))
    if requested > 0:
        return requested
    curves = _mapping(row.get("marginal_response_curves"))
    return max(0, _safe_int(curves.get("requested_repair_bytes")))


def _entropy_position(row: Mapping[str, Any], prior: Mapping[str, Any]) -> str:
    return str(
        row.get("entropy_position_label")
        or prior.get("entropy_position_label")
        or "unknown_entropy_pipeline_position"
    )


def _interaction_penalty(row: Mapping[str, Any]) -> float:
    scope = _mapping(row.get("interaction_scope"))
    stack_terms = _mapping(row.get("stacking_interaction_terms"))
    penalty = 0.0
    if stack_terms.get("must_remeasure_with_parent_and_sibling_repairs") is True:
        penalty += 0.12
    if scope.get("pair_indices") or scope.get("region_ids") or scope.get("mode_ids"):
        penalty += 0.05
    if _safe_int(scope.get("pair_count")) > 128:
        penalty += 0.04
    if _safe_int(scope.get("region_count")) > 8:
        penalty += 0.04
    return min(penalty, 0.35)


def _path_status(
    row: Mapping[str, Any],
    key: str,
    *,
    repo_root: str | Path | None,
) -> dict[str, Any]:
    text = str(row.get(key) or "").strip()
    return {
        "key": key,
        "path": text or None,
        "present": bool(text),
        "exists": _repo_path_exists(text, repo_root) if text else False,
    }


def _execution_gate(
    row: Mapping[str, Any],
    prior: Mapping[str, Any],
    *,
    repo_root: str | Path | None,
) -> dict[str, Any]:
    local_keys = ordered_unique(
        [
            "local_mlx_response_path",
            "reference_local_mlx_response_path",
            *[
                key
                for key in _string_list(prior.get("required_local_artifacts"))
                if key.endswith("_path")
            ],
        ]
    )
    local_status = [
        _path_status(row, key, repo_root=repo_root)
        for key in local_keys
    ]
    local_required = [
        item["key"]
        for item in local_status
        if item["key"] in {"local_mlx_response_path", "reference_local_mlx_response_path"}
    ]
    local_ready = bool(local_required) and all(
        item["exists"]
        for item in local_status
        if item["key"] in set(local_required)
    )
    missing = [
        f"{item['key']}:missing_or_unverified"
        for item in local_status
        if item["key"] in set(local_required) and not item["exists"]
    ]
    exact_missing = [
        "receiver_consumed_candidate_archive",
        "runtime_consumption_proof_path",
        "component_response_replay_manifest",
        "exact_axis_component_response_artifact",
    ]
    if not local_ready:
        missing.extend(exact_missing)
    blocker_label = str(prior.get("missing_artifact_label") or "")
    if blocker_label and not local_ready:
        missing.append(blocker_label)
    return {
        "schema": "repair_campaign_execution_gate.v1",
        "local_mlx_advisory_custody_ready": local_ready,
        "local_mlx_custody_paths": local_status,
        "recommended_queue_status": (
            "ready_for_local_mlx_advisory_execution"
            if local_ready
            else "blocked_missing_artifact"
        ),
        "missing_artifacts": ordered_unique(missing),
        "exact_missing_artifacts_if_not_local": exact_missing if not local_ready else [],
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "local_repair_campaign_execution_gate_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _typed_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    schema = str(payload.get("schema") or "")
    if schema == _TYPED_LEDGER_SCHEMA:
        rows = payload.get("rows")
    elif schema == _WORK_ORDER_SCHEMA:
        ledger = payload.get("typed_response_ledger")
        if not isinstance(ledger, Mapping):
            raise RepairCampaignScorerError("work order missing typed_response_ledger")
        rows = ledger.get("rows")
    else:
        raise RepairCampaignScorerError(
            f"unsupported repair campaign scorer input schema: {schema or '<missing>'}"
        )
    return [row for row in rows or [] if isinstance(row, Mapping)]


def score_repair_campaign(
    *,
    payload: Mapping[str, Any],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Score repair typed rows for the next queue-owned local campaign slice."""

    require_no_truthy_authority_fields(payload, context="repair_campaign_scorer_input")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(_typed_rows(payload), start=1):
        require_no_truthy_authority_fields(
            row,
            context=f"repair_campaign_scorer_row:{index}",
        )
        prior = _family_prior(row)
        objective_delta = _objective_delta(row)
        requested_bytes = _requested_bytes(row)
        improvement = (
            -objective_delta
            if objective_delta is not None and objective_delta < 0.0
            else 0.0
        )
        entropy_position = _entropy_position(row, prior)
        entropy_weight = _ENTROPY_POSITION_WEIGHTS.get(
            entropy_position,
            _ENTROPY_POSITION_WEIGHTS["unknown_entropy_pipeline_position"],
        )
        family_multiplier = _safe_float(prior.get("campaign_prior_multiplier")) or 1.0
        interaction_penalty = _interaction_penalty(row)
        bytes_denominator = requested_bytes if requested_bytes > 0 else 1
        improvement_per_byte = improvement / bytes_denominator
        campaign_score = (
            improvement_per_byte
            * entropy_weight
            * family_multiplier
            * (1.0 - interaction_penalty)
        )
        gate = _execution_gate(row, prior, repo_root=repo_root)
        scored_row = {
            "schema": REPAIR_CAMPAIGN_SCORE_ROW_SCHEMA,
            "source_row_schema": row.get("schema"),
            "rank_input_order": index,
            "typed_response_id": row.get("typed_response_id"),
            "candidate_id": row.get("candidate_id"),
            "acquisition_id": row.get("acquisition_id"),
            "correction_family": row.get("correction_family"),
            "family_id": prior.get("family_id") or "unclassified_repair_family",
            "family_prior": dict(prior),
            "entropy_position_label": entropy_position,
            "entropy_position_weight": entropy_weight,
            "requested_repair_bytes": requested_bytes,
            "objective_delta_score_units": objective_delta,
            "improvement_score_units": improvement,
            "improvement_per_byte": improvement_per_byte,
            "family_prior_multiplier": family_multiplier,
            "interaction_penalty": interaction_penalty,
            "campaign_score": campaign_score,
            "execution_gate": gate,
            "recommended_next_action": (
                "run_local_mlx_repair_stackability_probe"
                if gate["local_mlx_advisory_custody_ready"]
                else "materialize_missing_repair_campaign_artifacts"
            ),
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            "allowed_use": "repair_campaign_local_acquisition_ranking_only",
            "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
            **FALSE_AUTHORITY,
        }
        require_no_truthy_authority_fields(
            scored_row,
            context=f"repair_campaign_score_row:{index}",
        )
        rows.append(scored_row)
    rows.sort(
        key=lambda item: (
            item["execution_gate"]["recommended_queue_status"]
            != "ready_for_local_mlx_advisory_execution",
            -float(item.get("campaign_score") or 0.0),
            str(item.get("typed_response_id") or item.get("candidate_id") or ""),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["campaign_rank"] = rank
    ready_rows = [
        row
        for row in rows
        if row["execution_gate"]["recommended_queue_status"]
        == "ready_for_local_mlx_advisory_execution"
    ]
    missing_artifacts = ordered_unique(
        artifact
        for row in rows
        for artifact in _string_list(row["execution_gate"].get("missing_artifacts"))
    )
    report = {
        "schema": REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA,
        "row_schema": REPAIR_CAMPAIGN_SCORE_ROW_SCHEMA,
        "default_campaign_scorer": True,
        "input_schema": payload.get("schema"),
        "row_count": len(rows),
        "ready_for_local_mlx_advisory_execution_count": len(ready_rows),
        "blocked_missing_artifact_count": len(rows) - len(ready_rows),
        "operator_family_priors": repair_operator_family_priors(),
        "missing_artifacts": missing_artifacts,
        "rows": rows,
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "default_repair_campaign_scorer_for_queue_planning",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(report, context="repair_campaign_score_report")
    return report


__all__ = [
    "REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA",
    "REPAIR_CAMPAIGN_SCORE_ROW_SCHEMA",
    "REPAIR_OPERATOR_FAMILY_PRIORS_SCHEMA",
    "REPAIR_OPERATOR_FAMILY_PRIOR_ROW_SCHEMA",
    "RepairCampaignScorerError",
    "repair_operator_family_priors",
    "score_repair_campaign",
]
