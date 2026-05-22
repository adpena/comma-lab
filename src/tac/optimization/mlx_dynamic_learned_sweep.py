# SPDX-License-Identifier: MIT
"""Planning-only dynamic sweep ranking for MLX/local scorer candidates."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.optimization.bayesian_experimental_design import expected_improvement_minimize

SCHEMA = "mlx_dynamic_learned_sweep_plan.v1"
ROW_SCHEMA = "mlx_dynamic_learned_sweep_row.v1"
OPTIMIZATION_PASS_SCHEMA = "mlx_dynamic_learned_sweep_optimization_pass.v1"
TOOL = "tac.optimization.mlx_dynamic_learned_sweep"
DEFAULT_SCORE_VARIANCE = 2.5e-10

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
}


class MLXDynamicLearnedSweepError(ValueError):
    """Raised when a dynamic-sweep input would blur authority boundaries."""


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MLXDynamicLearnedSweepError(f"{path}: expected JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(payload), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MLXDynamicLearnedSweepError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise MLXDynamicLearnedSweepError(f"{label} must be finite")
    return result


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if payload.get(key) is not False:
            raise MLXDynamicLearnedSweepError(f"{label} {key} must be explicit false")


def _optional_float_from_keys(
    row: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: float = 0.0,
) -> float:
    for key in keys:
        if row.get(key) is None:
            continue
        value = _as_float(row.get(key), label=key)
        return value
    return float(default)


def _metadata_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def default_execution_configs() -> list[dict[str, Any]]:
    """Return built-in sweep configs, ordered from cheapest local to exact-gated."""

    return [
        {
            "config_id": "mlx_local_response",
            "substrate": EVIDENCE_TAG_MLX,
            "execution_layer": "local_mlx",
            "cost_units": 1.0,
            "signal_quality": 0.45,
            "parallelizable": True,
            "exact_eval_candidate": False,
            "allowed_use": "local_response_surface_learning_only",
        },
        {
            "config_id": "macos_cpu_advisory",
            "substrate": "[macOS-CPU advisory]",
            "execution_layer": "local_cpu",
            "cost_units": 4.0,
            "signal_quality": 0.60,
            "parallelizable": True,
            "exact_eval_candidate": False,
            "allowed_use": "local_advisory_filtering_only",
        },
        {
            "config_id": "contest_cpu_exact_candidate",
            "substrate": "[contest-CPU]",
            "execution_layer": "claimed_exact_cpu_eval",
            "cost_units": 12.0,
            "signal_quality": 0.90,
            "parallelizable": False,
            "exact_eval_candidate": True,
            "allowed_use": "requires_lane_claim_and_exact_auth_eval_before_score_claim",
        },
        {
            "config_id": "contest_cuda_diagnostic",
            "substrate": "[contest-CUDA T4]",
            "execution_layer": "claimed_exact_cuda_eval",
            "cost_units": 18.0,
            "signal_quality": 0.95,
            "parallelizable": False,
            "exact_eval_candidate": True,
            "allowed_use": "requires_lane_claim_and_exact_auth_eval_before_score_claim",
        },
    ]


def default_optimization_passes() -> list[dict[str, Any]]:
    """Return the default recursive sweep scales from smoke to freeze-ready."""

    return [
        {
            "pass_id": "smoke",
            "scale": "smoke",
            "recursive_stage": 0,
            "sample_budget": 1,
            "cost_multiplier": 0.25,
            "expected_improvement_weight": 0.25,
            "exploration_weight": 1.75,
            "freeze_candidate": False,
            "allowed_use": "cheap_liveness_and_sign_probe_before_wider_sweep",
        },
        {
            "pass_id": "micro",
            "scale": "micro",
            "recursive_stage": 1,
            "sample_budget": 8,
            "cost_multiplier": 1.0,
            "expected_improvement_weight": 0.75,
            "exploration_weight": 1.25,
            "freeze_candidate": False,
            "allowed_use": "local_candidate_config_learning",
        },
        {
            "pass_id": "intermediate",
            "scale": "intermediate",
            "recursive_stage": 2,
            "sample_budget": 32,
            "cost_multiplier": 4.0,
            "expected_improvement_weight": 1.25,
            "exploration_weight": 0.75,
            "freeze_candidate": False,
            "allowed_use": "confirm_config_family_and_update_posterior",
        },
        {
            "pass_id": "macro",
            "scale": "macro",
            "recursive_stage": 3,
            "sample_budget": 128,
            "cost_multiplier": 16.0,
            "expected_improvement_weight": 2.0,
            "exploration_weight": 0.25,
            "freeze_candidate": True,
            "allowed_use": "freeze_config_candidate_after_reproducible_passes",
        },
    ]


def _normalize_execution_configs(
    execution_configs: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    raw_configs = list(execution_configs) if execution_configs is not None else default_execution_configs()
    if not raw_configs:
        raise MLXDynamicLearnedSweepError("at least one execution config is required")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_configs):
        config_id = str(raw.get("config_id") or raw.get("id") or "")
        if not config_id:
            raise MLXDynamicLearnedSweepError(f"execution config {index} missing config_id")
        if config_id in seen:
            raise MLXDynamicLearnedSweepError(f"duplicate execution config: {config_id}")
        seen.add(config_id)
        cost_units = _as_float(raw.get("cost_units", 1.0), label=f"{config_id} cost_units")
        signal_quality = _as_float(
            raw.get("signal_quality", 0.5), label=f"{config_id} signal_quality"
        )
        if cost_units <= 0.0:
            raise MLXDynamicLearnedSweepError(f"{config_id} cost_units must be positive")
        if not 0.0 <= signal_quality <= 1.0:
            raise MLXDynamicLearnedSweepError(f"{config_id} signal_quality must be in [0, 1]")
        normalized.append(
            {
                "config_id": config_id,
                "substrate": str(raw.get("substrate") or EVIDENCE_TAG_MLX),
                "execution_layer": str(raw.get("execution_layer") or "local"),
                "cost_units": cost_units,
                "signal_quality": signal_quality,
                "parallelizable": bool(raw.get("parallelizable", True)),
                "exact_eval_candidate": bool(raw.get("exact_eval_candidate", False)),
                "allowed_use": str(raw.get("allowed_use") or "candidate_generation_only"),
            }
        )
    return normalized


def _normalize_optimization_passes(
    optimization_passes: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    raw_passes = (
        list(optimization_passes)
        if optimization_passes is not None
        else default_optimization_passes()
    )
    if not raw_passes:
        raise MLXDynamicLearnedSweepError("at least one optimization pass is required")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_passes):
        pass_id = str(raw.get("pass_id") or raw.get("id") or "")
        if not pass_id:
            raise MLXDynamicLearnedSweepError(f"optimization pass {index} missing pass_id")
        if pass_id in seen:
            raise MLXDynamicLearnedSweepError(f"duplicate optimization pass: {pass_id}")
        seen.add(pass_id)
        cost_multiplier = _as_float(
            raw.get("cost_multiplier", 1.0),
            label=f"{pass_id} cost_multiplier",
        )
        expected_weight = _as_float(
            raw.get("expected_improvement_weight", 1.0),
            label=f"{pass_id} expected_improvement_weight",
        )
        exploration_weight = _as_float(
            raw.get("exploration_weight", 1.0),
            label=f"{pass_id} exploration_weight",
        )
        if cost_multiplier <= 0.0:
            raise MLXDynamicLearnedSweepError(f"{pass_id} cost_multiplier must be positive")
        if expected_weight < 0.0 or exploration_weight < 0.0:
            raise MLXDynamicLearnedSweepError(
                f"{pass_id} pass weights must be non-negative"
            )
        sample_budget = int(raw.get("sample_budget", 1))
        if sample_budget <= 0:
            raise MLXDynamicLearnedSweepError(f"{pass_id} sample_budget must be positive")
        normalized.append(
            {
                "schema": OPTIMIZATION_PASS_SCHEMA,
                "pass_id": pass_id,
                "scale": str(raw.get("scale") or pass_id),
                "recursive_stage": int(raw.get("recursive_stage", index)),
                "sample_budget": sample_budget,
                "cost_multiplier": cost_multiplier,
                "expected_improvement_weight": expected_weight,
                "exploration_weight": exploration_weight,
                "freeze_candidate": bool(raw.get("freeze_candidate", False)),
                "allowed_use": str(raw.get("allowed_use") or "candidate_generation_only"),
            }
        )
    return sorted(
        normalized,
        key=lambda row: (int(row["recursive_stage"]), str(row["pass_id"])),
    )


def _variance_from_candidate(
    row: Mapping[str, Any],
    *,
    default_score_variance: float,
) -> float:
    for key in ("predicted_score_variance", "score_variance"):
        if row.get(key) is not None:
            variance = _as_float(row[key], label=f"{row.get('selector_id')} {key}")
            if variance < 0.0:
                raise MLXDynamicLearnedSweepError(f"{key} must be non-negative")
            return max(variance, 0.0)
    estimate = row.get("exact_cpu_calibrated_estimate")
    if isinstance(estimate, Mapping):
        delta = abs(float(estimate.get("predicted_delta_vs_base", 0.0)))
        return max(default_score_variance, (0.25 * delta) ** 2)
    return default_score_variance


def _selector_pareto_candidates(
    selector_pareto: Mapping[str, Any],
    *,
    default_score_variance: float,
) -> list[dict[str, Any]]:
    if selector_pareto.get("schema") != "decoder_q_selective_selector_pareto.v1":
        raise MLXDynamicLearnedSweepError("selector pareto schema mismatch")
    _require_false_authority(selector_pareto, label="selector pareto")
    rows = selector_pareto.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise MLXDynamicLearnedSweepError("selector pareto candidates[] missing")
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        _require_false_authority(row, label=f"selector candidate {index}")
        estimate = row.get("exact_cpu_calibrated_estimate")
        if isinstance(estimate, Mapping) and estimate.get("predicted_score") is not None:
            predicted_score = _as_float(
                estimate["predicted_score"],
                label=f"{row.get('selector_id')} predicted_score",
            )
            mean_source = "exact_cpu_calibrated_estimate"
        else:
            predicted_score = _as_float(
                row.get("predicted_score_mean"),
                label=f"{row.get('selector_id')} predicted_score_mean",
            )
            mean_source = "predicted_score_mean"
        out.append(
            {
                "candidate_id": str(row.get("selector_id") or f"selector_{index:04d}"),
                "family": "decoder_q_selective_dqs1",
                "source_schema": selector_pareto.get("schema"),
                "selector_kind": row.get("selector_kind"),
                "selected_pair_count": row.get("selected_pair_count"),
                "selected_pair_indices": row.get("selected_pair_indices"),
                "payload_bytes": row.get("payload_bytes"),
                "pair_encoding": row.get("pair_encoding"),
                "predicted_score_mean": predicted_score,
                "predicted_score_variance": _variance_from_candidate(
                    row,
                    default_score_variance=default_score_variance,
                ),
                "prediction_source": mean_source,
                "non_authoritative_mlx_gain_sum": row.get("non_authoritative_mlx_gain_sum"),
                "exact_cpu_calibrated_estimate": estimate if isinstance(estimate, Mapping) else None,
                "component_axis_context": _metadata_value(
                    row,
                    (
                        "component_axis_context",
                        "axis_mass",
                        "axis_score_impact_abs_sum",
                    ),
                ),
                "segnet_context": _metadata_value(
                    row,
                    ("segnet_context", "segnet_features", "segnet_response"),
                ),
                "posenet_context": _metadata_value(
                    row,
                    ("posenet_context", "posenet_features", "posenet_response"),
                ),
                "frame_pair_context": _metadata_value(
                    row,
                    (
                        "frame_pair_context",
                        "source_pair_window",
                        "pair_window",
                        "selected_pair_indices",
                    ),
                ),
                "ll_response_context": _metadata_value(
                    row,
                    (
                        "ll_response_context",
                        "scorer_response_context",
                        "oof_prediction_context",
                        "prediction_fit",
                    ),
                ),
                "waterbucket_context": _metadata_value(
                    row,
                    (
                        "waterbucket_context",
                        "signed_calibration",
                        "response_surface_objective",
                    ),
                ),
                "orthogonality_score": _optional_float_from_keys(
                    row,
                    ("orthogonality_score", "orthogonal_stack_score"),
                ),
                "master_gradient_priority": _optional_float_from_keys(
                    row,
                    ("master_gradient_priority", "per_pair_master_gradient_priority"),
                ),
                "canonical_equation_provenance": _metadata_value(
                    row,
                    (
                        "canonical_equation_provenance",
                        "canonical_equation",
                        "canonical_equations",
                    ),
                ),
                "master_gradient_provenance": _metadata_value(
                    row,
                    (
                        "master_gradient_provenance",
                        "master_gradient_annotation",
                        "master_gradient_anchor",
                        "per_pair_master_gradient_wire_in",
                    ),
                ),
                "orthogonality_contract": _metadata_value(
                    row,
                    ("orthogonality_contract", "orthogonal_stack_contract"),
                ),
                "freezing_provenance": _metadata_value(
                    row,
                    ("freezing_provenance", "freeze_report", "freezing_reports"),
                ),
                "portable_trainer_provenance": _metadata_value(
                    row,
                    (
                        "portable_trainer_provenance",
                        "mlx_portable_trainer",
                        "portable_trainer_contract",
                    ),
                ),
                **FALSE_AUTHORITY,
            }
        )
    return out


def _explicit_candidates(
    payload: Mapping[str, Any],
    *,
    default_score_variance: float,
) -> list[dict[str, Any]]:
    rows = payload.get("candidates")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        _require_false_authority(row, label=f"candidate {index}")
        predicted_score = _as_float(
            row.get("predicted_score_mean", row.get("score_mean")),
            label=f"candidate {index} predicted_score_mean",
        )
        out.append(
            {
                "candidate_id": str(row.get("candidate_id") or row.get("id") or f"candidate_{index:04d}"),
                "family": str(row.get("family") or row.get("family_id") or "unknown"),
                "source_schema": payload.get("schema"),
                "predicted_score_mean": predicted_score,
                "predicted_score_variance": _variance_from_candidate(
                    row,
                    default_score_variance=default_score_variance,
                ),
                "prediction_source": row.get("prediction_source") or "explicit_candidate_row",
                "payload_bytes": row.get("payload_bytes") or row.get("archive_size_bytes"),
                "selected_pair_indices": row.get("selected_pair_indices") or row.get("pair_indices"),
                "component_axis_context": _metadata_value(
                    row,
                    (
                        "component_axis_context",
                        "axis_mass",
                        "axis_score_impact_abs_sum",
                    ),
                ),
                "segnet_context": _metadata_value(
                    row,
                    ("segnet_context", "segnet_features", "segnet_response"),
                ),
                "posenet_context": _metadata_value(
                    row,
                    ("posenet_context", "posenet_features", "posenet_response"),
                ),
                "frame_pair_context": _metadata_value(
                    row,
                    (
                        "frame_pair_context",
                        "source_pair_window",
                        "pair_window",
                        "selected_pair_indices",
                        "pair_indices",
                    ),
                ),
                "ll_response_context": _metadata_value(
                    row,
                    (
                        "ll_response_context",
                        "scorer_response_context",
                        "oof_prediction_context",
                        "prediction_fit",
                    ),
                ),
                "waterbucket_context": _metadata_value(
                    row,
                    (
                        "waterbucket_context",
                        "signed_calibration",
                        "response_surface_objective",
                    ),
                ),
                "orthogonality_score": _optional_float_from_keys(
                    row,
                    ("orthogonality_score", "orthogonal_stack_score"),
                ),
                "master_gradient_priority": _optional_float_from_keys(
                    row,
                    ("master_gradient_priority", "per_pair_master_gradient_priority"),
                ),
                "canonical_equation_provenance": _metadata_value(
                    row,
                    (
                        "canonical_equation_provenance",
                        "canonical_equation",
                        "canonical_equations",
                    ),
                ),
                "master_gradient_provenance": _metadata_value(
                    row,
                    (
                        "master_gradient_provenance",
                        "master_gradient_annotation",
                        "master_gradient_anchor",
                        "per_pair_master_gradient_wire_in",
                    ),
                ),
                "orthogonality_contract": _metadata_value(
                    row,
                    ("orthogonality_contract", "orthogonal_stack_contract"),
                ),
                "freezing_provenance": _metadata_value(
                    row,
                    ("freezing_provenance", "freeze_report", "freezing_reports"),
                ),
                "portable_trainer_provenance": _metadata_value(
                    row,
                    (
                        "portable_trainer_provenance",
                        "mlx_portable_trainer",
                        "portable_trainer_contract",
                    ),
                ),
                **FALSE_AUTHORITY,
            }
        )
    return out


def _normalize_candidates(
    *,
    selector_pareto: Mapping[str, Any] | None,
    candidate_payloads: Sequence[Mapping[str, Any]] | None,
    default_score_variance: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if selector_pareto is not None:
        candidates.extend(
            _selector_pareto_candidates(
                selector_pareto,
                default_score_variance=default_score_variance,
            )
        )
    for payload in candidate_payloads or ():
        candidates.extend(
            _explicit_candidates(
                payload,
                default_score_variance=default_score_variance,
            )
        )
    if not candidates:
        raise MLXDynamicLearnedSweepError("no dynamic sweep candidates supplied")
    return candidates


def _row_for_config(
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
    optimization_pass: Mapping[str, Any],
    *,
    incumbent_score: float,
    lcb_z: float,
    expected_improvement_weight: float,
    exploration_weight: float,
) -> dict[str, Any]:
    mean = float(candidate["predicted_score_mean"])
    variance = float(candidate["predicted_score_variance"])
    sigma = math.sqrt(max(0.0, variance))
    cost = float(config["cost_units"]) * float(optimization_pass["cost_multiplier"])
    expected_improvement = expected_improvement_minimize(mean, variance, incumbent_score)
    lower_confidence_score = mean - float(lcb_z) * sigma
    orthogonality_score = max(0.0, float(candidate.get("orthogonality_score", 0.0)))
    master_gradient_priority = max(
        0.0,
        float(candidate.get("master_gradient_priority", 0.0)),
    )
    geometry_multiplier = 1.0 + min(1.0, orthogonality_score) + min(
        1.0,
        master_gradient_priority,
    )
    learning_value = float(config["signal_quality"]) * sigma * geometry_multiplier / cost
    pass_expected_weight = (
        float(expected_improvement_weight)
        * float(optimization_pass["expected_improvement_weight"])
    )
    pass_exploration_weight = (
        float(exploration_weight) * float(optimization_pass["exploration_weight"])
    )
    acquisition_value = (
        pass_expected_weight * expected_improvement / cost
        + pass_exploration_weight * learning_value
    )
    exact_config = bool(config["exact_eval_candidate"])
    blockers = [
        "score_claim_requires_exact_auth_eval_result",
        "promotion_requires_canonical_auth_axis_payload",
    ]
    if exact_config:
        blockers.extend(
            [
                "lane_claim_required_before_dispatch",
                "locality_or_runtime_controls_required_before_dispatch",
                "dispatch_not_attempted_by_dynamic_sweep_planner",
            ]
        )
    return {
        "schema": ROW_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "sweep_config_id": config["config_id"],
        "optimization_pass_id": optimization_pass["pass_id"],
        "optimization_scale": optimization_pass["scale"],
        "recursive_stage": optimization_pass["recursive_stage"],
        "family": candidate["family"],
        "rank": None,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "allowed_use": (
            "local_dynamic_sweep_candidate_generation"
            if not exact_config
            else "exact_eval_candidate_queue_requires_separate_claim"
        ),
        "substrate": config["substrate"],
        "execution_layer": config["execution_layer"],
        "parallelizable": config["parallelizable"],
        "exact_eval_candidate": exact_config,
        "ready_for_local_sweep": not exact_config,
        "sample_budget": optimization_pass["sample_budget"],
        "freeze_candidate": bool(optimization_pass["freeze_candidate"]),
        "frozen_config_eligible_after_pass": bool(optimization_pass["freeze_candidate"]),
        "frozen_config_contract": {
            "schema": "mlx_dynamic_learned_sweep_frozen_config_contract.v1",
            "eligible_after_pass": bool(optimization_pass["freeze_candidate"]),
            "requires_reproducible_observations": True,
            "requires_byte_closed_materialization": True,
            "requires_exact_auth_axis_before_score_claim": True,
            "may_compile_into_submission_runtime_or_archive": bool(
                optimization_pass["freeze_candidate"]
            ),
            "may_seed_next_baseline": bool(optimization_pass["freeze_candidate"]),
            **FALSE_AUTHORITY,
        },
        "dispatch_blockers": sorted(dict.fromkeys(blockers)),
        "predicted_score_mean": mean,
        "predicted_score_variance": variance,
        "predicted_score_sigma": sigma,
        "lower_confidence_score": lower_confidence_score,
        "incumbent_score": incumbent_score,
        "expected_improvement": expected_improvement,
        "learning_value_per_cost": learning_value,
        "geometry_multiplier": geometry_multiplier,
        "orthogonality_score": orthogonality_score,
        "master_gradient_priority": master_gradient_priority,
        "acquisition_value": acquisition_value,
        "cost_units": cost,
        "base_config_cost_units": config["cost_units"],
        "pass_cost_multiplier": optimization_pass["cost_multiplier"],
        "signal_quality": config["signal_quality"],
        "pass_expected_improvement_weight": pass_expected_weight,
        "pass_exploration_weight": pass_exploration_weight,
        "prediction_source": candidate.get("prediction_source"),
        "source_schema": candidate.get("source_schema"),
        "selector_kind": candidate.get("selector_kind"),
        "selected_pair_count": candidate.get("selected_pair_count"),
        "selected_pair_indices": candidate.get("selected_pair_indices"),
        "payload_bytes": candidate.get("payload_bytes"),
        "pair_encoding": candidate.get("pair_encoding"),
        "non_authoritative_mlx_gain_sum": candidate.get("non_authoritative_mlx_gain_sum"),
        "component_axis_context": candidate.get("component_axis_context"),
        "segnet_context": candidate.get("segnet_context"),
        "posenet_context": candidate.get("posenet_context"),
        "frame_pair_context": candidate.get("frame_pair_context"),
        "ll_response_context": candidate.get("ll_response_context"),
        "waterbucket_context": candidate.get("waterbucket_context"),
        "canonical_equation_provenance": candidate.get("canonical_equation_provenance"),
        "master_gradient_provenance": candidate.get("master_gradient_provenance"),
        "orthogonality_contract": candidate.get("orthogonality_contract"),
        "freezing_provenance": candidate.get("freezing_provenance"),
        "portable_trainer_provenance": candidate.get("portable_trainer_provenance"),
        "next_action": (
            "run_local_mlx_or_cpu_sweep_then_append_observation"
            if not exact_config
            else "materialize_controls_claim_then_exact_eval_in_separate_flow"
        ),
        "recursive_update_contract": {
            "append_observation_fields": [
                "candidate_id",
                "sweep_config_id",
                "optimization_pass_id",
                "family",
                "selected_pair_indices",
                "component_axis_deltas",
                "segnet_delta",
                "posenet_delta",
                "rate_delta",
                "observed_axis",
                "observed_score_or_delta",
                "archive_sha256",
                "raw_output_or_cache_sha256",
                "runtime_sha256",
            ],
            "rerun_planner_after_observation": True,
        },
    }


def _row_sort_key(row: Mapping[str, Any]) -> tuple[float, int, float, float, str, str, str]:
    return (
        -float(row["acquisition_value"]),
        int(row["recursive_stage"]),
        float(row["lower_confidence_score"]),
        float(row["cost_units"]),
        str(row["candidate_id"]),
        str(row["sweep_config_id"]),
        str(row["optimization_pass_id"]),
    )


def _stratified_rows_by_pass(
    rows: Sequence[dict[str, Any]],
    *,
    passes: Sequence[Mapping[str, Any]],
    per_pass_top_k: int,
    top_k: int,
) -> list[dict[str, Any]]:
    min_rows = per_pass_top_k * len(passes)
    if top_k < min_rows:
        raise MLXDynamicLearnedSweepError(
            "top_k must be at least per_pass_top_k * optimization_pass_count"
        )
    selected: list[dict[str, Any]] = []
    for optimization_pass in passes:
        pass_id = str(optimization_pass["pass_id"])
        pass_rows = [
            row for row in rows if str(row["optimization_pass_id"]) == pass_id
        ]
        selected.extend(pass_rows[:per_pass_top_k])
    selected.sort(
        key=lambda row: (
            int(row["recursive_stage"]),
            *_row_sort_key(row),
        )
    )
    return selected[:top_k]


def build_mlx_dynamic_learned_sweep_plan(
    *,
    incumbent_score: float,
    selector_pareto: Mapping[str, Any] | None = None,
    candidate_payloads: Sequence[Mapping[str, Any]] | None = None,
    execution_configs: Sequence[Mapping[str, Any]] | None = None,
    optimization_passes: Sequence[Mapping[str, Any]] | None = None,
    top_k: int = 32,
    per_pass_top_k: int | None = None,
    default_score_variance: float = DEFAULT_SCORE_VARIANCE,
    lcb_z: float = 1.0,
    expected_improvement_weight: float = 1.0,
    exploration_weight: float = 1.0,
    source_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank dynamic local/exact-gated sweep configs without score authority."""

    if top_k <= 0:
        raise MLXDynamicLearnedSweepError("top_k must be positive")
    if per_pass_top_k is not None and per_pass_top_k <= 0:
        raise MLXDynamicLearnedSweepError("per_pass_top_k must be positive")
    incumbent = _as_float(incumbent_score, label="incumbent_score")
    if incumbent <= 0.0:
        raise MLXDynamicLearnedSweepError("incumbent_score must be positive")
    default_variance = _as_float(default_score_variance, label="default_score_variance")
    if default_variance < 0.0:
        raise MLXDynamicLearnedSweepError("default_score_variance must be non-negative")
    configs = _normalize_execution_configs(execution_configs)
    passes = _normalize_optimization_passes(optimization_passes)
    candidates = _normalize_candidates(
        selector_pareto=selector_pareto,
        candidate_payloads=candidate_payloads,
        default_score_variance=default_variance,
    )
    rows = [
        _row_for_config(
            candidate,
            config,
            optimization_pass,
            incumbent_score=incumbent,
            lcb_z=float(lcb_z),
            expected_improvement_weight=float(expected_improvement_weight),
            exploration_weight=float(exploration_weight),
        )
        for candidate in candidates
        for config in configs
        for optimization_pass in passes
    ]
    rows.sort(key=_row_sort_key)
    if per_pass_top_k is None:
        rows = rows[: int(top_k)]
    else:
        rows = _stratified_rows_by_pass(
            rows,
            passes=passes,
            per_pass_top_k=int(per_pass_top_k),
            top_k=int(top_k),
        )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    local_rows = [row for row in rows if row["ready_for_local_sweep"] is True]
    exact_rows = [row for row in rows if row["exact_eval_candidate"] is True]
    freeze_rows = [row for row in rows if row["freeze_candidate"] is True]
    return {
        "schema": SCHEMA,
        "producer": TOOL,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "archive_materialization_required": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "allowed_use": (
            "dynamic_learned_local_sweep_planning_only_no_score_or_dispatch_authority"
        ),
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "incumbent_score": incumbent,
        "selection_policy": {
            "top_k": int(top_k),
            "per_pass_top_k": per_pass_top_k,
            "default_score_variance": default_variance,
            "lcb_z": float(lcb_z),
            "expected_improvement_weight": float(expected_improvement_weight),
            "exploration_weight": float(exploration_weight),
            "sort": [
                "acquisition_value_desc",
                "recursive_stage_asc",
                "lower_confidence_score_asc",
                "cost_units_asc",
            ],
        },
        "execution_configs": configs,
        "optimization_passes": passes,
        "source_artifacts": dict(source_artifacts or {}),
        "recursive_learning_contract": {
            "schema": "mlx_dynamic_learned_sweep_recursive_learning_contract.v1",
            "initial_scale": "smoke",
            "later_scales": ["micro", "intermediate", "macro"],
            "freeze_after": "macro",
            "observation_append_only": True,
            "rerun_planner_after_each_pass": True,
            "frozen_configs_are_baselines_not_scores": True,
            **FALSE_AUTHORITY,
        },
        "summary": {
            "candidate_count": len(candidates),
            "config_count": len(configs),
            "optimization_pass_count": len(passes),
            "ranked_row_count": len(rows),
            "local_ready_row_count": len(local_rows),
            "exact_candidate_row_count": len(exact_rows),
            "freeze_candidate_row_count": len(freeze_rows),
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "ranked_sweep_rows": rows,
    }


def render_mlx_dynamic_learned_sweep_markdown(plan: Mapping[str, Any]) -> str:
    rows = plan.get("ranked_sweep_rows")
    summary = plan.get("summary")
    lines = [
        "# MLX Dynamic Learned Sweep Plan",
        "",
        f"- Score claim: `{plan.get('score_claim')}`",
        f"- Ready for exact eval dispatch: `{plan.get('ready_for_exact_eval_dispatch')}`",
        f"- Local ready rows: `{summary.get('local_ready_row_count') if isinstance(summary, Mapping) else None}`",
        "",
        "| rank | candidate | config | pass | acq | mean | lcb | local ready | next action |",
        "|---:|---|---|---|---:|---:|---:|---|---|",
    ]
    if isinstance(rows, list):
        for row in rows[:24]:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| "
                f"{row.get('rank')} | `{row.get('candidate_id')}` | "
                f"`{row.get('sweep_config_id')}` | "
                f"`{row.get('optimization_pass_id')}` | "
                f"{float(row.get('acquisition_value', 0.0)):.12g} | "
                f"{float(row.get('predicted_score_mean', 0.0)):.12g} | "
                f"{float(row.get('lower_confidence_score', 0.0)):.12g} | "
                f"`{row.get('ready_for_local_sweep')}` | `{row.get('next_action')}` |"
            )
    lines.extend(
        [
            "",
            "Authority: planning only. Local MLX/CPU rows may drive more local sweeps; "
            "exact CPU/CUDA rows still require materialization, controls, lane claim, "
            "canonical auth eval, and harvest before any score claim.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_SCORE_VARIANCE",
    "FALSE_AUTHORITY",
    "OPTIMIZATION_PASS_SCHEMA",
    "SCHEMA",
    "TOOL",
    "MLXDynamicLearnedSweepError",
    "build_mlx_dynamic_learned_sweep_plan",
    "default_execution_configs",
    "default_optimization_passes",
    "dumps_json",
    "file_sha256",
    "load_json_object",
    "render_mlx_dynamic_learned_sweep_markdown",
    "write_json",
]
