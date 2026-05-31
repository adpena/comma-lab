# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    ordered_unique,
    truthy_authority_field_violations,
)

LOCAL_EXACT_AUTH_GATE_SCHEMA = "local_candidate_exact_auth_gate.v1"
LOCAL_REPLAY_SCHEMA = "local_submission_replay.v1"
MLX_REPLAY_SCHEMA = "z8_full_video_mlx_replay.v1"
MACOS_CPU_AXIS_TAG = "[macOS-CPU advisory]"
MACOS_MLX_AXIS_TAG = "[macOS-MLX research-signal]"

FALSE_AUTHORITY: dict[str, bool] = {
    **PROXY_FALSE_AUTHORITY_FIELDS,
    "score_claim_valid": False,
    "score_claim_eligible": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "exact_cuda_auth_eval": False,
    "contest_cuda_auth_eval": False,
}


@dataclass(frozen=True)
class LocalExactAuthGateConfig:
    exact_auth_axis: str = "[contest-CPU]"
    auth_target_score: float | None = None
    local_baseline_score: float | None = None
    min_local_improvement: float = 0.0
    expected_local_axis_tag: str = MACOS_CPU_AXIS_TAG
    require_mlx_prefilter: bool = False
    mlx_target_action: float | None = None
    expected_mlx_axis_tag: str = MACOS_MLX_AXIS_TAG


@dataclass(frozen=True)
class LocalExactAuthGateReport:
    schema: str
    exact_auth_axis: str
    local_replay_summary_path: str | None
    mlx_prefilter_summary_path: str | None
    local_axis_tag: str | None
    local_score_estimate: float | None
    local_baseline_score: float | None
    auth_target_score: float | None
    min_local_improvement: float
    mlx_axis_tag: str | None
    mlx_action_proxy: float | None
    mlx_target_action: float | None
    exact_auth_dispatch_recommended: bool
    exact_cpu_dispatch_recommended: bool
    exact_cuda_dispatch_recommended: bool
    next_required_action: str
    blockers: list[str]
    warnings: list[str]
    local_replay_checks: dict[str, Any] | None
    mlx_prefilter_checks: dict[str, Any] | None
    score_claim: bool
    score_claim_valid: bool
    score_claim_eligible: bool
    promotion_eligible: bool
    promotable: bool
    rank_or_kill_eligible: bool
    ready_for_exact_eval_dispatch: bool
    field_selection_ready_for_exact_eval_dispatch: bool
    dispatch_attempted: bool
    gpu_launched: bool
    exact_cuda_auth_eval: bool
    contest_cuda_auth_eval: bool
    score_affecting_payload_changed: bool
    charged_bits_changed: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _as_mapping(payload: Mapping[str, Any] | Any, *, label: str) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, Mapping):
        return {}, [f"{label}_not_json_object"]
    return dict(payload), []


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _local_replay_checks(
    payload: Mapping[str, Any],
    *,
    config: LocalExactAuthGateConfig,
) -> tuple[dict[str, Any], list[str], list[str], float | None, str | None]:
    blockers: list[str] = []
    warnings: list[str] = []
    schema = payload.get("schema")
    if schema != LOCAL_REPLAY_SCHEMA:
        blockers.append(f"local_replay_schema_unsupported:{schema}")
    if payload.get("evaluation_passed") is not True:
        blockers.append("local_replay_not_evaluation_passed")
    axis_tag = str(payload.get("axis_tag") or "")
    if axis_tag != config.expected_local_axis_tag:
        blockers.append(f"local_replay_axis_mismatch:{axis_tag or 'missing'}")
    local_score = _finite_float(payload.get("local_score_estimate"))
    if local_score is None:
        blockers.append("local_replay_score_missing_or_nonfinite")

    for violation in truthy_authority_field_violations(payload):
        blockers.append(f"local_replay_forbidden_authority:{violation}")

    if config.auth_target_score is None and config.local_baseline_score is None:
        blockers.append("local_improvement_target_missing")

    required_margin = float(config.min_local_improvement)
    if required_margin < 0.0:
        blockers.append("min_local_improvement_negative")
    if local_score is not None and config.auth_target_score is not None:
        threshold = float(config.auth_target_score) - max(required_margin, 0.0)
        if not local_score < threshold:
            blockers.append("local_score_not_below_auth_target")
    if local_score is not None and config.local_baseline_score is not None:
        threshold = float(config.local_baseline_score) - max(required_margin, 0.0)
        if not local_score < threshold:
            blockers.append("local_score_not_below_local_baseline")

    if axis_tag == config.expected_local_axis_tag:
        warnings.append("local_axis_is_advisory_exact_auth_still_required")

    checks = {
        "schema": schema,
        "evaluation_passed": payload.get("evaluation_passed") is True,
        "expected_axis_tag": config.expected_local_axis_tag,
        "axis_tag": axis_tag or None,
        "local_score_estimate": local_score,
        "auth_target_score": config.auth_target_score,
        "local_baseline_score": config.local_baseline_score,
        "min_local_improvement": config.min_local_improvement,
        "blockers": ordered_unique(blockers),
        "warnings": ordered_unique(warnings),
    }
    return checks, blockers, warnings, local_score, axis_tag or None


def _mlx_prefilter_checks(
    payload: Mapping[str, Any] | None,
    *,
    config: LocalExactAuthGateConfig,
) -> tuple[dict[str, Any] | None, list[str], list[str], float | None, str | None]:
    if payload is None:
        if config.require_mlx_prefilter:
            return None, ["mlx_prefilter_required_but_missing"], [], None, None
        return None, [], [], None, None

    blockers: list[str] = []
    warnings: list[str] = []
    schema = payload.get("schema")
    if schema != MLX_REPLAY_SCHEMA:
        blockers.append(f"mlx_prefilter_schema_unsupported:{schema}")
    if payload.get("full_video_local_replay_executed") is not True:
        blockers.append("mlx_prefilter_not_full_video_replay")
    if payload.get("replay_ok", True) is not True:
        blockers.append("mlx_prefilter_replay_not_ok")
    if payload.get("full_video_local_replay_scope") not in {None, "full_video"}:
        blockers.append(f"mlx_prefilter_scope_mismatch:{payload.get('full_video_local_replay_scope')}")
    axis = str(payload.get("local_axis") or payload.get("axis_tag") or "")
    if axis and axis != config.expected_mlx_axis_tag:
        blockers.append(f"mlx_prefilter_axis_mismatch:{axis}")

    action = _finite_float(payload.get("contest_action_proxy"))
    if action is None:
        blockers.append("mlx_prefilter_action_missing_or_nonfinite")
    if (
        action is not None
        and config.mlx_target_action is not None
        and not action < float(config.mlx_target_action)
    ):
        blockers.append("mlx_prefilter_action_not_below_target")

    for violation in truthy_authority_field_violations(payload):
        blockers.append(f"mlx_prefilter_forbidden_authority:{violation}")
    warnings.append("mlx_axis_is_research_signal_cpu_replay_still_required")

    checks = {
        "schema": schema,
        "full_video_local_replay_executed": payload.get("full_video_local_replay_executed") is True,
        "full_video_local_replay_scope": payload.get("full_video_local_replay_scope"),
        "replay_ok": payload.get("replay_ok", True) is True,
        "expected_axis_tag": config.expected_mlx_axis_tag,
        "axis_tag": axis or None,
        "contest_action_proxy": action,
        "mlx_target_action": config.mlx_target_action,
        "blockers": ordered_unique(blockers),
        "warnings": ordered_unique(warnings),
    }
    return checks, blockers, warnings, action, axis or None


def build_local_exact_auth_gate_report(
    *,
    local_replay_summary: Mapping[str, Any] | None = None,
    config: LocalExactAuthGateConfig | None = None,
    mlx_prefilter_summary: Mapping[str, Any] | None = None,
    local_replay_summary_path: str | Path | None = None,
    mlx_prefilter_summary_path: str | Path | None = None,
) -> LocalExactAuthGateReport:
    cfg = config or LocalExactAuthGateConfig()
    local_shape_blockers: list[str] = []
    local_blockers: list[str] = []
    local_warnings: list[str] = []
    local_checks: dict[str, Any] | None = None
    local_score: float | None = None
    local_axis: str | None = None
    if local_replay_summary is None:
        local_blockers.append("local_replay_required_for_exact_auth")
    else:
        local_payload, local_shape_blockers = _as_mapping(local_replay_summary, label="local_replay")
        local_checks, local_blockers, local_warnings, local_score, local_axis = _local_replay_checks(
            local_payload,
            config=cfg,
        )
    mlx_checks, mlx_blockers, mlx_warnings, mlx_action, mlx_axis = _mlx_prefilter_checks(
        dict(mlx_prefilter_summary) if isinstance(mlx_prefilter_summary, Mapping) else None,
        config=cfg,
    )
    if mlx_prefilter_summary is not None and not isinstance(mlx_prefilter_summary, Mapping):
        mlx_blockers = [*mlx_blockers, "mlx_prefilter_not_json_object"]

    blockers = ordered_unique([*local_shape_blockers, *local_blockers, *mlx_blockers])
    warnings = ordered_unique([*local_warnings, *mlx_warnings])
    exact_axis = str(cfg.exact_auth_axis)
    exact_auth_recommended = not blockers and local_replay_summary is not None
    axis_lc = exact_axis.strip().lower()
    mlx_prefilter_passed = mlx_prefilter_summary is not None and not mlx_blockers
    if exact_auth_recommended and "cpu" in axis_lc:
        next_action = "claim_lane_and_run_exact_cpu_auth_eval"
    elif exact_auth_recommended:
        next_action = "claim_lane_and_run_exact_auth_eval"
    elif local_replay_summary is None and mlx_prefilter_passed:
        next_action = "run_local_cpu_replay"
    elif mlx_prefilter_summary is not None and mlx_blockers:
        next_action = "do_not_run_local_cpu_replay"
    else:
        next_action = "do_not_dispatch_exact_auth"
    return LocalExactAuthGateReport(
        schema=LOCAL_EXACT_AUTH_GATE_SCHEMA,
        exact_auth_axis=exact_axis,
        local_replay_summary_path=str(local_replay_summary_path) if local_replay_summary_path else None,
        mlx_prefilter_summary_path=str(mlx_prefilter_summary_path) if mlx_prefilter_summary_path else None,
        local_axis_tag=local_axis,
        local_score_estimate=local_score,
        local_baseline_score=cfg.local_baseline_score,
        auth_target_score=cfg.auth_target_score,
        min_local_improvement=float(cfg.min_local_improvement),
        mlx_axis_tag=mlx_axis,
        mlx_action_proxy=mlx_action,
        mlx_target_action=cfg.mlx_target_action,
        exact_auth_dispatch_recommended=exact_auth_recommended,
        exact_cpu_dispatch_recommended=bool(exact_auth_recommended and "cpu" in axis_lc),
        exact_cuda_dispatch_recommended=bool(exact_auth_recommended and "cuda" in axis_lc),
        next_required_action=next_action,
        blockers=blockers,
        warnings=warnings,
        local_replay_checks=local_checks,
        mlx_prefilter_checks=mlx_checks,
        **FALSE_AUTHORITY,
    )


def load_json_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {resolved}")
    return payload


__all__ = [
    "FALSE_AUTHORITY",
    "LOCAL_EXACT_AUTH_GATE_SCHEMA",
    "LocalExactAuthGateConfig",
    "LocalExactAuthGateReport",
    "build_local_exact_auth_gate_report",
    "load_json_object",
]
