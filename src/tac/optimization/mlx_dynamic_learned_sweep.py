# SPDX-License-Identifier: MIT
"""Planning-only dynamic sweep ranking for MLX/local scorer candidates."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.exact_eval_custody import CONTEST_EXACT_SAMPLE_COUNT
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.optimization.bayesian_experimental_design import expected_improvement_minimize
from tac.optimization.mlx_dynamic_sweep_observations import (
    normalize_observation_row,
    summarize_observations,
)
from tac.optimization.optimizer_scheduler_registry import (
    enumerate_optimizer_scheduler_candidates,
)
from tac.optimization.optimizer_training_signal_bridge import (
    build_optimizer_training_signal_wire_in,
)
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

SCHEMA = "mlx_dynamic_learned_sweep_plan.v1"
ROW_SCHEMA = "mlx_dynamic_learned_sweep_row.v1"
OPTIMIZATION_PASS_SCHEMA = "mlx_dynamic_learned_sweep_optimization_pass.v1"
TOOL = "tac.optimization.mlx_dynamic_learned_sweep"
DEFAULT_SCORE_VARIANCE = 2.5e-10
QUALITY_EVIDENCE_SCHEMA = "mlx_dynamic_learned_sweep_quality_evidence.v1"
DISALLOWED_CANDIDATE_PAYLOAD_SCHEMAS: frozenset[str] = frozenset(
    {"optimizer_candidate_queue_v1"}
)
TIMING_OR_PROXY_RANK_FIELD_FRAGMENTS: tuple[str, ...] = (
    "seconds_per_",
    "timing",
    "_cost_signal_not_score",
    "planner_priority_not_score",
    "negative_acquisition_value_proxy_not_score",
    "proxy_objective_not_score",
)
TIMING_ONLY_CONSUMER_PAYLOAD_SCHEMAS: frozenset[str] = frozenset(
    {
        "trainer_runtime_profile_candidate_payload.v1",
        "representation_training_candidate_payload.v1",
        "pr95_muon_local_training_candidate_payload.v1",
    }
)
_CONTEST_OBSERVATION_LABELS = {
    "contest_cpu",
    "contest_cuda",
    "contest-CPU",
    "contest-CUDA",
    "[contest-CPU]",
    "[contest-CUDA]",
}

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
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        raise MLXDynamicLearnedSweepError(str(exc)) from exc


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


def _require_allowed_candidate_payload_schema(payload: Mapping[str, Any]) -> None:
    schema = str(payload.get("schema") or "")
    if schema in DISALLOWED_CANDIDATE_PAYLOAD_SCHEMAS:
        raise MLXDynamicLearnedSweepError(
            f"{schema} is timing/planning signal; use an explicit calibrated "
            "quality adapter before learned-sweep candidate intake"
        )


def _consumer_payload_schema(row: Mapping[str, Any]) -> str:
    payload = row.get("consumer_payload")
    return str(payload.get("schema") or "") if isinstance(payload, Mapping) else ""


def _require_quality_evidence(row: Mapping[str, Any], *, label: str) -> None:
    rank_field = str(row.get("rank_score_field") or "")
    consumer_schema = _consumer_payload_schema(row)
    if any(fragment in rank_field for fragment in TIMING_OR_PROXY_RANK_FIELD_FRAGMENTS):
        raise MLXDynamicLearnedSweepError(
            f"{label} rank_score_field={rank_field!r} is timing/proxy signal; "
            "learned sweep requires explicit calibrated quality evidence"
        )
    if consumer_schema in TIMING_ONLY_CONSUMER_PAYLOAD_SCHEMAS:
        raise MLXDynamicLearnedSweepError(
            f"{label} consumer_payload schema {consumer_schema!r} is timing-only; "
            "use an explicit calibrated quality adapter"
        )

    estimate = row.get("exact_cpu_calibrated_estimate")
    if (
        row.get("exact_cpu_calibrated_estimate_scope") == "candidate_specific"
        and isinstance(estimate, Mapping)
        and estimate.get("predicted_score") is not None
    ):
        _require_false_authority(estimate, label=f"{label} exact_cpu_calibrated_estimate")
        return

    quality = row.get("quality_evidence")
    if not isinstance(quality, Mapping):
        raise MLXDynamicLearnedSweepError(
            f"{label} missing explicit calibrated quality evidence"
        )
    if quality.get("schema") != QUALITY_EVIDENCE_SCHEMA:
        raise MLXDynamicLearnedSweepError(
            f"{label} quality_evidence schema must be {QUALITY_EVIDENCE_SCHEMA}"
        )
    _require_false_authority(quality, label=f"{label} quality_evidence")
    if quality.get("calibrated_score_mean") is None and quality.get(
        "calibrated_gain_mean"
    ) is None:
        raise MLXDynamicLearnedSweepError(
            f"{label} quality_evidence requires calibrated_score_mean or calibrated_gain_mean"
        )
    gate_statuses = quality.get("gate_statuses")
    if not isinstance(gate_statuses, Mapping):
        raise MLXDynamicLearnedSweepError(
            f"{label} quality_evidence.gate_statuses must be an object"
        )
    required_gates = (
        "calibration",
        "parity",
        "production_contract",
        "effective_spend_triage",
    )
    for gate in required_gates:
        if gate_statuses.get(gate) != "strict_pass":
            raise MLXDynamicLearnedSweepError(
                f"{label} quality_evidence.gate_statuses.{gate} must be strict_pass"
            )


def _as_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise MLXDynamicLearnedSweepError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MLXDynamicLearnedSweepError(f"{label} must be an integer") from exc
    if parsed != value and not (isinstance(value, str) and str(parsed) == value):
        raise MLXDynamicLearnedSweepError(f"{label} must be integral")
    return parsed


def _validate_normalized_gain_sum(row: Mapping[str, Any], *, label: str) -> None:
    """Guard aggregate MLX gain aliases before dynamic-sweep admission."""

    normalized_raw = row.get("non_authoritative_normalized_full_video_gain_sum")
    mlx_alias_raw = row.get("non_authoritative_mlx_gain_sum")
    window_raw = row.get("non_authoritative_mlx_window_gain_sum")
    if normalized_raw is None and mlx_alias_raw is None and window_raw is None:
        return
    denominator = _as_int(row.get("full_video_denominator"), label=f"{label} full_video_denominator")
    if denominator != CONTEST_EXACT_SAMPLE_COUNT:
        raise MLXDynamicLearnedSweepError(
            f"{label} full_video_denominator must be {CONTEST_EXACT_SAMPLE_COUNT}"
        )
    normalized = _as_float(
        normalized_raw,
        label=f"{label} non_authoritative_normalized_full_video_gain_sum",
    )
    window = _as_float(
        window_raw,
        label=f"{label} non_authoritative_mlx_window_gain_sum",
    )
    expected = window / float(denominator)
    if not math.isclose(normalized, expected, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise MLXDynamicLearnedSweepError(
            f"{label} non_authoritative_normalized_full_video_gain_sum mismatch"
        )
    if mlx_alias_raw is not None:
        mlx_alias = _as_float(
            mlx_alias_raw,
            label=f"{label} non_authoritative_mlx_gain_sum",
        )
        if not math.isclose(mlx_alias, normalized, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise MLXDynamicLearnedSweepError(
                f"{label} non_authoritative_mlx_gain_sum must equal normalized full-video gain sum"
            )


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
    if isinstance(estimate, Mapping) and _exact_estimate_is_candidate_specific(row):
        delta = abs(float(estimate.get("predicted_delta_vs_base", 0.0)))
        return max(default_score_variance, (0.25 * delta) ** 2)
    return default_score_variance


def _exact_estimate_is_candidate_specific(row: Mapping[str, Any]) -> bool:
    scope = row.get("exact_cpu_calibrated_estimate_scope")
    return str(scope) == "candidate_specific"


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
        row_label = str(row.get("selector_id") or f"selector candidate {index}")
        _validate_normalized_gain_sum(row, label=row_label)
        estimate = row.get("exact_cpu_calibrated_estimate")
        if (
            isinstance(estimate, Mapping)
            and estimate.get("predicted_score") is not None
            and _exact_estimate_is_candidate_specific(row)
        ):
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
                "non_authoritative_normalized_full_video_gain_sum": row.get(
                    "non_authoritative_normalized_full_video_gain_sum"
                ),
                "non_authoritative_mlx_window_gain_sum": row.get(
                    "non_authoritative_mlx_window_gain_sum"
                ),
                "full_video_denominator": row.get("full_video_denominator"),
                "exact_cpu_calibrated_estimate": estimate if isinstance(estimate, Mapping) else None,
                "exact_cpu_calibrated_estimate_scope": row.get(
                    "exact_cpu_calibrated_estimate_scope"
                ),
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
    _require_allowed_candidate_payload_schema(payload)
    rows = payload.get("candidates")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        _require_false_authority(row, label=f"candidate {index}")
        row_label = str(row.get("candidate_id") or row.get("id") or f"candidate {index}")
        _validate_normalized_gain_sum(row, label=row_label)
        _require_quality_evidence(row, label=row_label)
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
                "prediction_source": (
                    row.get("prediction_source")
                    or row.get("predicted_score_source")
                    or "explicit_candidate_row"
                ),
                "prediction_scope": row.get("predicted_score_scope"),
                "exact_cpu_calibrated_estimate_scope": row.get(
                    "exact_cpu_calibrated_estimate_scope"
                ),
                "payload_bytes": row.get("payload_bytes") or row.get("archive_size_bytes"),
                "selected_pair_indices": row.get("selected_pair_indices") or row.get("pair_indices"),
                "selected_pair_count": row.get("selected_pair_count"),
                "non_authoritative_mlx_gain_sum": row.get("non_authoritative_mlx_gain_sum"),
                "non_authoritative_normalized_full_video_gain_sum": row.get(
                    "non_authoritative_normalized_full_video_gain_sum"
                ),
                "non_authoritative_mlx_window_gain_sum": row.get(
                    "non_authoritative_mlx_window_gain_sum"
                ),
                "full_video_denominator": row.get("full_video_denominator"),
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
    blockers = sorted(dict.fromkeys(blockers))
    row_candidate_id = (
        f"{candidate['candidate_id']}::{config['config_id']}::{optimization_pass['pass_id']}"
    )
    solver_candidate_params = {
        "source_candidate_id": candidate["candidate_id"],
        "sweep_config_id": config["config_id"],
        "optimization_pass_id": optimization_pass["pass_id"],
        "optimization_scale": optimization_pass["scale"],
        "recursive_stage": optimization_pass["recursive_stage"],
        "execution_layer": config["execution_layer"],
        "substrate": config["substrate"],
        "sample_budget": optimization_pass["sample_budget"],
        "selected_pair_indices": candidate.get("selected_pair_indices"),
        "selected_pair_count": candidate.get("selected_pair_count"),
        "payload_bytes": candidate.get("payload_bytes"),
        "exact_eval_candidate": exact_config,
    }
    solver_stack_wire_in = build_optimizer_training_signal_wire_in(
        candidate_id=row_candidate_id,
        profile_id="mlx_dynamic_learned_sweep",
        lane_id="mlx_dynamic_learned_sweep_planning",
        lane_class=str(candidate["family"]),
        candidate_family=str(candidate["family"]),
        representation_family=str(candidate.get("representation_family") or candidate["family"]),
        substrate_family=str(candidate.get("substrate_family") or config["substrate"]),
        training_signal_kind="mlx_dynamic_learned_sweep_proxy",
        param_schema="mlx_dynamic_learned_sweep_config_params_v1",
        candidate_params=solver_candidate_params,
        source_anchor=str(candidate.get("source_schema") or "mlx dynamic learned sweep candidate"),
        score_lowering_hypothesis=(
            "Use MLX/local/advisory response-surface observations to rank recursive "
            "sweep configs before byte-closed exact auth replay."
        ),
        dispatch_blockers=blockers,
        variant_axes=[
            "source_candidate",
            "sweep_config",
            "optimization_pass",
            "component_axis_context",
            "execution_substrate",
        ],
        paired_modes=[
            "same_candidate_different_config",
            "same_config_different_candidate",
            "local_axis_then_exact_axis_anchor",
        ],
    )
    return {
        "schema": ROW_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "queue_candidate_id": row_candidate_id,
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
        "dispatch_blockers": blockers,
        "predicted_score_mean": mean,
        "predicted_score_variance": variance,
        "predicted_score_sigma": sigma,
        "lower_confidence_score": lower_confidence_score,
        "incumbent_score": incumbent_score,
        "expected_improvement": expected_improvement,
        "learning_value_per_cost": learning_value,
        "geometry_multiplier": geometry_multiplier,
        "acquisition_terms": {
            "schema": "mlx_dynamic_learned_sweep_acquisition_terms.v1",
            "expected_improvement_per_cost": expected_improvement / cost,
            "weighted_expected_improvement_per_cost": (
                pass_expected_weight * expected_improvement / cost
            ),
            "learning_value_per_cost": learning_value,
            "weighted_learning_value_per_cost": pass_exploration_weight * learning_value,
            "geometry_multiplier": geometry_multiplier,
            "orthogonality_score": orthogonality_score,
            "master_gradient_priority": master_gradient_priority,
            "score_sigma": sigma,
            "cost_units": cost,
            **FALSE_AUTHORITY,
        },
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
        "prediction_scope": candidate.get("prediction_scope"),
        "exact_cpu_calibrated_estimate_scope": candidate.get(
            "exact_cpu_calibrated_estimate_scope"
        ),
        "source_schema": candidate.get("source_schema"),
        "selector_kind": candidate.get("selector_kind"),
        "selected_pair_count": candidate.get("selected_pair_count"),
        "selected_pair_indices": candidate.get("selected_pair_indices"),
        "payload_bytes": candidate.get("payload_bytes"),
        "pair_encoding": candidate.get("pair_encoding"),
        "non_authoritative_mlx_gain_sum": candidate.get("non_authoritative_mlx_gain_sum"),
        "non_authoritative_normalized_full_video_gain_sum": candidate.get(
            "non_authoritative_normalized_full_video_gain_sum"
        ),
        "non_authoritative_mlx_window_gain_sum": candidate.get(
            "non_authoritative_mlx_window_gain_sum"
        ),
        "full_video_denominator": candidate.get("full_video_denominator"),
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
        "solver_stack_wire_in": solver_stack_wire_in,
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


def _observation_tuple_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(row["candidate_id"]),
            str(row["sweep_config_id"]),
            str(row["optimization_pass_id"]),
            str(row["family"]),
        )
    )


def _candidate_config_family_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(row["candidate_id"]),
            str(row["sweep_config_id"]),
            str(row["family"]),
        )
    )


def _is_contest_axis_observation(row: Mapping[str, Any]) -> bool:
    return any(
        str(row.get(key) or "") in _CONTEST_OBSERVATION_LABELS
        for key in ("observed_axis", "evidence_grade", "evidence_tag")
    )


def _latest_observation_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "row_count": 0,
        "observed_axes": [],
        "evidence_tags": [],
        "latest_observed_at_utc": None,
        "observed_score_or_delta_latest": None,
        **FALSE_AUTHORITY,
    }
    for row in rows:
        observed_at = str(row.get("observed_at_utc") or "")
        summary["row_count"] += 1
        for key, target in (
            ("observed_axis", "observed_axes"),
            ("evidence_tag", "evidence_tags"),
        ):
            value = str(row.get(key) or "")
            if value and value not in summary[target]:
                summary[target].append(value)
        if summary["latest_observed_at_utc"] is None or observed_at >= str(
            summary["latest_observed_at_utc"]
        ):
            summary["latest_observed_at_utc"] = observed_at
            summary["observed_score_or_delta_latest"] = row.get(
                "observed_score_or_delta"
            )
    summary["observed_axes"].sort()
    summary["evidence_tags"].sort()
    return summary


def _normalize_observations(
    observations: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    return [normalize_observation_row(row) for row in observations or ()]


def _observation_feedback_indexes(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, list[Mapping[str, Any]]]]:
    by_tuple: dict[str, list[Mapping[str, Any]]] = {}
    exact_by_candidate_config_family: dict[str, list[Mapping[str, Any]]] = {}
    for row in observations:
        by_tuple.setdefault(_observation_tuple_key(row), []).append(row)
        if _is_contest_axis_observation(row):
            exact_by_candidate_config_family.setdefault(
                _candidate_config_family_key(row),
                [],
            ).append(row)
    return by_tuple, exact_by_candidate_config_family


def _apply_observation_feedback(
    rows: Sequence[dict[str, Any]],
    *,
    observations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_tuple, exact_by_candidate_config_family = _observation_feedback_indexes(
        observations
    )
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for row in rows:
        tuple_key = _observation_tuple_key(row)
        candidate_config_family_key = _candidate_config_family_key(row)
        feedback_rows: list[Mapping[str, Any]] = []
        reason: str | None = None
        if tuple_key in by_tuple:
            reason = "already_observed_candidate_config_pass_family"
            feedback_rows = by_tuple[tuple_key]
        elif candidate_config_family_key in exact_by_candidate_config_family:
            reason = "exact_axis_observed_candidate_config_family"
            feedback_rows = exact_by_candidate_config_family[candidate_config_family_key]

        if reason is None:
            kept.append(row)
            continue

        suppressed_row = dict(row)
        suppressed_row["rank"] = None
        suppressed_row["suppressed_by_observation_feedback"] = True
        suppressed_row["observation_feedback"] = {
            "schema": "mlx_dynamic_learned_sweep_observation_feedback.v1",
            "suppression_reason": reason,
            "candidate_config_pass_family_key": tuple_key,
            "candidate_config_family_key": candidate_config_family_key,
            "summary": _latest_observation_summary(feedback_rows),
            **FALSE_AUTHORITY,
        }
        suppressed.append(suppressed_row)
    return kept, suppressed


def _optimizer_scheduler_pairings(
    rows: Sequence[dict[str, Any]],
    optimizer_scheduler_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pairings: list[dict[str, Any]] = []
    for parent in rows:
        if parent.get("ready_for_local_sweep") is not True:
            continue
        for recipe_index, recipe in enumerate(optimizer_scheduler_rows, start=1):
            descriptor_id = str(recipe.get("descriptor_id") or "")
            if not descriptor_id:
                continue
            parent_queue_id = str(parent["queue_candidate_id"])
            queue_candidate_id = f"{parent_queue_id}::optimizer_scheduler::{descriptor_id}"
            blockers = [
                "optimizer_scheduler_pairing_is_planning_only",
                "same_seed_local_ablation_required_before_recipe_posterior_update",
                "training_telemetry_required_before_candidate_selection",
                "archive_export_required_before_exact_eval_readiness",
                "exact_auth_eval_result_required_before_score_claim",
            ]
            solver_stack_wire_in = build_optimizer_training_signal_wire_in(
                candidate_id=queue_candidate_id,
                profile_id="mlx_dynamic_learned_sweep_optimizer_pairing",
                lane_id="mlx_dynamic_learned_sweep_planning",
                lane_class=str(parent["family"]),
                candidate_family=str(parent["family"]),
                representation_family=str(parent.get("family") or "unspecified"),
                substrate_family=str(parent.get("substrate") or "unspecified"),
                training_signal_kind="optimizer_scheduler_pairing_proxy",
                param_schema="mlx_dynamic_optimizer_scheduler_pairing_params_v1",
                candidate_params={
                    "parent_queue_candidate_id": parent_queue_id,
                    "source_candidate_id": parent["candidate_id"],
                    "sweep_config_id": parent["sweep_config_id"],
                    "optimization_pass_id": parent["optimization_pass_id"],
                    "optimizer_scheduler_descriptor_id": descriptor_id,
                    "optimizer_scheduler_config_sha256": recipe.get("config_sha256"),
                    "parameter_group_lr_policy_id": recipe.get(
                        "parameter_group_lr_policy_id"
                    ),
                    "parameter_group_lr_policy_sha256": recipe.get(
                        "parameter_group_lr_policy_sha256"
                    ),
                    "sample_budget": parent.get("sample_budget"),
                },
                source_anchor=str(parent.get("source_schema") or "mlx dynamic learned sweep row"),
                score_lowering_hypothesis=(
                    "Run same-candidate, same-config, same-pass optimizer/scheduler "
                    "ablations so optimizer recipes become measured posterior signal "
                    "instead of disconnected static config names."
                ),
                dispatch_blockers=blockers,
                variant_axes=[
                    "optimizer_scheduler_recipe",
                    "parameter_group_lr_policy",
                    "same_candidate_config_pass",
                    "same_seed_budget",
                    "training_telemetry",
                ],
                paired_modes=[
                    "same_candidate_config_pass_different_optimizer_scheduler",
                    "same_optimizer_scheduler_different_candidate",
                    "same_optimizer_scheduler_different_execution_substrate",
                ],
            )
            acquisition_value = float(parent["acquisition_value"])
            tool_wiring = {
                "schema": "optimizer_scheduler_pairing_tool_wiring.v1",
                "ablation_surfaces": [
                    "src/tac/findings_lagrangian/phase_2_ablation/ablation_framework.py",
                    "tools/dispatch_phase_a_track_1_ablations.py",
                ],
                "xray_surfaces": [
                    "tools/master_gradient_xray.py",
                    "tools/xray_hardpair_hitlist.py",
                    "tools/xray_paired_cpu_cuda_axis_delta.py",
                ],
                "atom_surfaces": [
                    "src/tac/atom/ledger.py",
                    "tools/build_cross_paradigm_atom_ledger.py",
                    "tools/meta_lagrangian_atom_ledger_adapter.py",
                ],
                "materialization_surfaces": [
                    "tools/materialize_decoder_q_selective_runtime_candidate.py",
                    "tools/materialize_codec_op_bitstream.py",
                ],
                "freezing_surfaces": [
                    "src/tac/freezing/frozen_teacher_distillation.py",
                    "src/tac/freezing/swa_checkpoint_averaging.py",
                    "src/tac/freezing/lottery_ticket_extraction.py",
                    "src/tac/freezing/compress_time_scorer_freeze.py",
                ],
                "observation_surfaces": [
                    "tools/append_mlx_dynamic_sweep_observation.py",
                    "src/tac/optimization/mlx_dynamic_sweep_observations.py",
                    "src/tac/optimization/optimizer_scheduler_registry.py",
                ],
                **FALSE_AUTHORITY,
            }
            pairings.append(
                {
                    "schema": "mlx_dynamic_learned_sweep_optimizer_scheduler_pairing.v1",
                    "queue_candidate_id": queue_candidate_id,
                    "candidate_id": parent["candidate_id"],
                    "parent_queue_candidate_id": parent_queue_id,
                    "parent_rank": parent.get("rank"),
                    "sweep_config_id": parent["sweep_config_id"],
                    "optimization_pass_id": parent["optimization_pass_id"],
                    "optimization_scale": parent["optimization_scale"],
                    "family": parent["family"],
                    "substrate": parent["substrate"],
                    "execution_layer": parent["execution_layer"],
                    "sample_budget": parent["sample_budget"],
                    "optimizer_scheduler_descriptor_id": descriptor_id,
                    "optimizer_scheduler_config_sha256": recipe.get("config_sha256"),
                    "optimizer": recipe.get("optimizer"),
                    "scheduler": recipe.get("scheduler"),
                    "parameter_group_lr_policy_id": recipe.get(
                        "parameter_group_lr_policy_id"
                    ),
                    "parameter_group_lr_policy_sha256": recipe.get(
                        "parameter_group_lr_policy_sha256"
                    ),
                    "rank_score": -acquisition_value + recipe_index * 1e-12,
                    "rank_score_field": "parent_negative_acquisition_value_plus_recipe_tiebreak_not_score",
                    "pairing_acquisition_value": acquisition_value,
                    "pairing_recipe_rank": recipe_index,
                    "dispatch_blockers": blockers,
                    "next_action": "run_same_seed_local_optimizer_ablation_then_append_observation",
                    "paired_ablation_contract": {
                        "schema": "optimizer_scheduler_paired_ablation_contract.v1",
                        "control_fields": [
                            "source_candidate_id",
                            "sweep_config_id",
                            "optimization_pass_id",
                            "sample_budget",
                            "seed",
                            "archive_export_contract",
                        ],
                        "treatment_fields": [
                            "optimizer_scheduler_descriptor_id",
                            "optimizer_scheduler_config_sha256",
                            "parameter_group_lr_policy_id",
                            "parameter_group_lr_policy_sha256",
                        ],
                        "required_observation_fields": [
                            "seconds_per_candidate",
                            "best_proxy_loss_or_score",
                            "state_bytes",
                            "archive_ready",
                            "export_ready",
                            "component_axis_deltas",
                        ],
                        "tool_wiring": tool_wiring,
                        "no_orphan_signal_contract": (
                            "Every optimizer pairing observation must append through "
                            "the observation surface, update the queue row, or carry "
                            "an explicit research_only=true rationale."
                        ),
                        **FALSE_AUTHORITY,
                    },
                    "tool_wiring": tool_wiring,
                    "solver_stack_wire_in": solver_stack_wire_in,
                    "candidate_generation_only": True,
                    "dispatch_attempted": False,
                    "gpu_launched": False,
                    **FALSE_AUTHORITY,
                }
            )
    sorted_pairings = sorted(
        pairings,
        key=lambda item: (
            item["parent_rank"] or 10**9,
            item["pairing_recipe_rank"],
            item["queue_candidate_id"],
        ),
    )
    for index, row in enumerate(sorted_pairings, start=1):
        row["pairing_rank"] = index
    return sorted_pairings


def build_mlx_dynamic_learned_sweep_plan(
    *,
    incumbent_score: float,
    selector_pareto: Mapping[str, Any] | None = None,
    candidate_payloads: Sequence[Mapping[str, Any]] | None = None,
    execution_configs: Sequence[Mapping[str, Any]] | None = None,
    optimization_passes: Sequence[Mapping[str, Any]] | None = None,
    observations: Sequence[Mapping[str, Any]] | None = None,
    optimizer_scheduler_candidates: Sequence[Mapping[str, Any]] | None = None,
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
    optimizer_scheduler_rows = [
        dict(row)
        for row in (
            optimizer_scheduler_candidates
            if optimizer_scheduler_candidates is not None
            else enumerate_optimizer_scheduler_candidates()
        )
    ]
    for index, row in enumerate(optimizer_scheduler_rows):
        try:
            require_no_truthy_authority_fields(
                row,
                context=f"optimizer_scheduler_candidate[{index}]",
            )
        except ValueError as exc:
            raise MLXDynamicLearnedSweepError(str(exc)) from exc
    observation_rows = _normalize_observations(observations)
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
    rows, suppressed_rows = _apply_observation_feedback(
        rows,
        observations=observation_rows,
    )
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
    optimizer_scheduler_pairings = _optimizer_scheduler_pairings(
        rows,
        optimizer_scheduler_rows,
    )
    observation_summary = summarize_observations(observation_rows)
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
        "optimizer_scheduler_candidates": optimizer_scheduler_rows,
        "source_artifacts": dict(source_artifacts or {}),
        "observation_feedback": {
            "schema": "mlx_dynamic_learned_sweep_observation_feedback_summary.v1",
            "observation_row_count": len(observation_rows),
            "suppressed_observed_row_count": len(suppressed_rows),
            "suppression_policy": {
                "local_or_advisory_observation": (
                    "suppress_exact_candidate_config_pass_family_tuple_only"
                ),
                "contest_axis_observation": (
                    "suppress_same_candidate_config_family_across_passes"
                ),
            },
            "observation_summary": observation_summary,
            **FALSE_AUTHORITY,
        },
        "recursive_learning_contract": {
            "schema": "mlx_dynamic_learned_sweep_recursive_learning_contract.v1",
            "initial_scale": "smoke",
            "later_scales": ["micro", "intermediate", "macro"],
            "freeze_after": "macro",
            "observation_append_only": True,
            "rerun_planner_after_each_pass": True,
            "frozen_configs_are_baselines_not_scores": True,
            "optimizer_scheduler_registry_surface": (
                "tac.optimization.optimizer_scheduler_registry."
                "enumerate_optimizer_scheduler_candidates"
            ),
            "optimizer_scheduler_telemetry_surface": (
                "tac.optimization.optimizer_scheduler_registry."
                "build_optimizer_scheduler_telemetry_record"
            ),
            "optimizer_scheduler_pairing_surface": (
                "optimizer_scheduler_pairings[].paired_ablation_contract"
            ),
            **FALSE_AUTHORITY,
        },
        "summary": {
            "candidate_count": len(candidates),
            "config_count": len(configs),
            "optimizer_scheduler_candidate_count": len(optimizer_scheduler_rows),
            "optimization_pass_count": len(passes),
            "observation_row_count": len(observation_rows),
            "suppressed_observed_row_count": len(suppressed_rows),
            "ranked_row_count": len(rows),
            "local_ready_row_count": len(local_rows),
            "exact_candidate_row_count": len(exact_rows),
            "freeze_candidate_row_count": len(freeze_rows),
            "optimizer_scheduler_pairing_count": len(optimizer_scheduler_pairings),
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "ranked_sweep_rows": rows,
        "optimizer_scheduler_pairings": optimizer_scheduler_pairings,
        "suppressed_observed_sweep_rows": suppressed_rows,
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
        f"- Observation rows: `{summary.get('observation_row_count') if isinstance(summary, Mapping) else None}`",
        f"- Suppressed observed rows: `{summary.get('suppressed_observed_row_count') if isinstance(summary, Mapping) else None}`",
        f"- Optimizer pairings: `{summary.get('optimizer_scheduler_pairing_count') if isinstance(summary, Mapping) else None}`",
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
