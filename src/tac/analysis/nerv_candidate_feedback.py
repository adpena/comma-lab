# SPDX-License-Identifier: MIT
"""Harvestable feedback rows for NeRV candidate curriculum runs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.adaptation.hard_pair_indices import (
    HardPairIndicesError,
    pair_indices_from_mapping,
)
from tac.substrates.hprc.mlx_prefilter_coverage import (
    DEFAULT_MAX_MLX_SCORE_FOR_LOCAL_REPLAY,
    summarize_mlx_prefilter_coverage,
)
from tac.substrates.hprc.resolution_contract import CONTEST_PAIR_COUNT
from tac.substrates.snerv_inverse_steg_carrier.mlx_native_adapter_contract import (
    build_snerv_mlx_native_training_export_guard,
)

SCHEMA = "nerv_candidate_feedback_row.v1"
LEDGER_SCHEMA = "nerv_candidate_byte_feedback_ledger.v1"
REFRESH_SCHEMA = "nerv_candidate_feedback_refresh.v1"
TELEMETRY_FEEDBACK_SCHEMA = "nerv_training_telemetry_feedback.v1"
SAMPLE_GENERALIZATION_GATE_SCHEMA = "nerv_sample_generalization_gate.v1"
HINERV_ARCHIVE_LADDER_FEEDBACK_SCHEMA = "hinerv_archive_ladder_candidate_feedback.v1"
FULL_VIDEO_MLX_SCORER_FEEDBACK_SCHEMA = "nerv_full_video_mlx_scorer_feedback.v1"
SNERV_RENDERER_NONDEGENERATE_PROOF_SCHEMA = "snerv_renderer_nondegenerate_proof.v1"
SNERV_RENDERER_NONDEGENERATE_MIN_PAIR_COUNT = 16
SNERV_SCORER_TETHER_SMOKE_HEALTH_SCHEMA = (
    "snerv_scorer_domain_tether_smoke_health.v1"
)
SNERV_SCORER_INPUT_DISTRIBUTION_GUARD_PROOF_SCHEMA = (
    "snerv_scorer_input_distribution_guard_proof.v1"
)

FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

_MLX_PREFILTER_MISSING_BLOCKERS = {
    "full_video_mlx_scorer_replay_not_attached",
    "local_cpu_replay_waiting_for_full_video_mlx_prefilter",
    "hi_nerv_full_video_local_prefilter_missing",
    "snerv_full_video_local_prefilter_missing",
    "mlx_prefilter_not_full_video",
    "sampled_mlx_prefilter_requires_full_video_rerun",
}

_LOCAL_REPLAY_BLOCKED_BY_MLX_SCORE = (
    "local_cpu_replay_blocked_by_mlx_prefilter_score"
)
_POSE_LOSS_INSTABILITY_THRESHOLD = 1_000.0
_POSE_AXIS_INSTABILITY_THRESHOLD = 1_000.0
_POSE_INSTABILITY_WINDOW_EPOCHS = 32
_POSE_INSTABILITY_BAD_FRACTION = 0.5
_POSE_INSTABILITY_LR_MULTIPLIER = 0.3
_MIDRUN_STOP_REASONS = frozenset(
    {
        "midrun_feedback_snapshot_do_not_stop_training",
        "training_running_midrun_feedback_snapshot",
    }
)
_SEG_STAGNATION_MIN_EPOCHS = 128
_SEG_STAGNATION_WINDOW_EPOCHS = 64
_SEG_STAGNATION_MIN_RELATIVE_IMPROVEMENT = 0.05
_SEG_STAGNATION_WEIGHT_MULTIPLIER = 2.0
_SEG_STAGNATION_MAX_DISTILLATION_WEIGHT = 8.0
_FULL_VIDEO_FIT_WEIGHT_MIN_MULTIPLIER = 2.0
_FULL_VIDEO_FIT_WEIGHT_MAX_DISTILLATION_WEIGHT = 8.0
_TRAINING_CONTROL_RECENT_WINDOW_EPOCHS = 64
_TRAINING_CONTROL_MIN_RECENT_RELATIVE_IMPROVEMENT = 0.01
_POSE_TAIL_BURST_MIN_EPOCHS = 64
_POSE_TAIL_BURST_WINDOW_EPOCHS = 64
_POSE_TAIL_BURST_MIN_AXIS = 8.0
_POSE_TAIL_BURST_MEDIAN_MULTIPLIER = 4.0
_POSE_TAIL_BURST_BAD_FRACTION = 0.05
_PR95_FINAL_MUON_STAGE_INDEX = 8
_SNERV_SCORER_TETHER_METRICS = (
    "snerv_posenet_yuv6_pair_distill",
    "snerv_segnet_last_frame_distill",
)
_SNERV_SCORER_TETHER_RECENT_MISSING_FRACTION = 0.9
_SNERV_SCORER_TETHER_LAMBDA_ACTIVE_EPS = 1.0e-12
_HINERV_TRAIN_TIME_CONTROL_LAMBDA_ACTIVE_EPS = 1.0e-12
_FULL_VIDEO_RESPONSE_BAD_SCORE_THRESHOLD = DEFAULT_MAX_MLX_SCORE_FOR_LOCAL_REPLAY
_FULL_VIDEO_RESPONSE_BAD_SEG_THRESHOLD = 0.02
_FULL_VIDEO_RESPONSE_BAD_POSE_THRESHOLD = 1.0
_FULL_VIDEO_RESPONSE_SCORE_AUTHORITY_KEYS = (
    "score_claim",
    "score_claim_valid",
    "frontier_score_claim",
    "rank_or_kill_eligible",
    "promotion_eligible",
    "promotable",
    "ready_for_exact_eval_dispatch",
)
_HINERV_RECEIVER_CACHE_COLLAPSE_BLOCKER_FRAGMENTS = (
    "receiver_argmax_class_collapse",
    "segnet_argmax_class_collapse",
    "receiver_export_segnet_argmax_class_collapse",
)


def _bool_metric_or_none(value: Any) -> bool | None:
    numeric = _float_or_none(value)
    if numeric is not None:
        return bool(numeric >= 0.5)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "on"}:
        return True
    if text in {"false", "no", "off"}:
        return False
    return None


def _canonical_pr95_stage_status(
    *,
    current_epoch: int | None,
    current_stage_index: int | None,
) -> dict[str, Any]:
    """Return canonical PR95 stage context for telemetry interpretation."""

    status: dict[str, Any] = {
        "schema": "nerv_training_pr95_stage_status.v1",
        "current_epoch": current_epoch,
        "observed_current_stage_index": current_stage_index,
        "canonical_total_epoch_budget": None,
        "canonical_final_muon_stage_index": _PR95_FINAL_MUON_STAGE_INDEX,
        "canonical_final_muon_stage_start_epoch": None,
        "canonical_expected_stage_index": None,
        "observed_stage_matches_canonical_epoch": None,
        "blockers": [],
    }
    try:
        from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
            PR95FaithfulCurriculumFactory,
        )

        factory = PR95FaithfulCurriculumFactory()
        status["canonical_total_epoch_budget"] = factory.total_epoch_budget
        for stage_index, start_epoch, _end_epoch in factory.stage_epoch_boundaries:
            if int(stage_index) == _PR95_FINAL_MUON_STAGE_INDEX:
                status["canonical_final_muon_stage_start_epoch"] = int(start_epoch)
                break
        if current_epoch is not None:
            expected_stage = int(factory.current_stage_index(int(current_epoch)))
            status["canonical_expected_stage_index"] = expected_stage
            if current_stage_index is not None:
                status["observed_stage_matches_canonical_epoch"] = (
                    int(current_stage_index) == expected_stage
                )
    except Exception as exc:
        status["blockers"] = [
            "canonical_pr95_stage_status_unavailable",
            f"canonical_pr95_stage_status_error:{type(exc).__name__}",
        ]
    return status


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_present(*sources_and_key: Any) -> Any:
    *sources, key = sources_and_key
    for source in sources:
        if isinstance(source, Mapping) and source.get(key) is not None:
            return source.get(key)
    return None


def _snerv_official_checkpoint_mapping_manifest(
    *sources: Any,
) -> dict[str, Any]:
    """Return the strongest embedded official checkpoint mapping manifest."""

    for source in sources:
        manifest = _snerv_official_checkpoint_mapping_manifest_from_source(source)
        if manifest:
            return manifest
    return {}


def _snerv_official_checkpoint_mapping_manifest_from_source(
    source: Any,
) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        return {}
    direct = source.get("official_trained_checkpoint_mapping_manifest")
    if isinstance(direct, Mapping):
        return dict(direct)
    binding = source.get("official_checkpoint_export_binding")
    if isinstance(binding, Mapping):
        manifest = binding.get("official_trained_checkpoint_mapping_manifest")
        if isinstance(manifest, Mapping):
            return dict(manifest)
    train_export = source.get("official_mfu_hfr_tub_train_export")
    if isinstance(train_export, Mapping):
        manifest = train_export.get("official_trained_checkpoint_mapping_manifest")
        if isinstance(manifest, Mapping):
            return dict(manifest)
    replay = source.get("official_mfu_hfr_tub_source_forward_replay")
    if isinstance(replay, Mapping):
        manifest = replay.get("official_trained_checkpoint_mapping_manifest")
        if isinstance(manifest, Mapping):
            return dict(manifest)
    score_training = source.get("score_aware_long_training")
    if isinstance(score_training, Mapping):
        manifest = _snerv_official_checkpoint_mapping_manifest_from_source(
            score_training
        )
        if manifest:
            return manifest
    native_export = source.get("snerv_mlx_native_export") or source.get(
        "mlx_native_export"
    )
    if isinstance(native_export, Mapping):
        manifest = _snerv_official_checkpoint_mapping_manifest_from_source(
            native_export
        )
        if manifest:
            return manifest
    file_evidence = source.get("snerv_mlx_native_file_backed_export_evidence")
    if isinstance(file_evidence, Mapping):
        manifest = _snerv_official_checkpoint_mapping_manifest_from_source(
            file_evidence
        )
        if manifest:
            return manifest
    return {}


def _snerv_trained_state_exportability(*sources: Any) -> bool | None:
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in (
            "checkpoint_trained_state_exportable",
            "score_aware_long_training_trained_state_exportable",
            "trained_state_exportable",
            "official_trained_state_exportable",
        ):
            value = source.get(key)
            if value is True:
                return True
            if value is False:
                return False
        binding = source.get("official_checkpoint_export_binding")
        if isinstance(binding, Mapping):
            value = _snerv_trained_state_exportability(binding)
            if value is not None:
                return value
        score_training = source.get("score_aware_long_training")
        if isinstance(score_training, Mapping):
            value = _snerv_trained_state_exportability(score_training)
            if value is not None:
                return value
    return None


def _snerv_score_aware_training_telemetry_contract(
    native_export: Mapping[str, Any],
) -> dict[str, Any]:
    return _snerv_score_aware_training_telemetry_contract_from_sources(native_export)


def _snerv_score_aware_training_telemetry_contract_from_sources(
    *sources: Any,
) -> dict[str, Any]:
    for source in sources:
        contract = _snerv_score_aware_training_telemetry_contract_from_source(
            source,
            seen=set(),
        )
        if contract:
            return contract
    return {}


def _snerv_score_aware_training_telemetry_contract_from_source(
    source: Any,
    *,
    seen: set[int],
) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        return {}
    identity = id(source)
    if identity in seen:
        return {}
    seen.add(identity)
    for key in (
        "score_aware_long_training_telemetry_contract",
        "training_telemetry_contract",
        "snerv_score_aware_long_training_telemetry_contract",
    ):
        direct = source.get(key)
        if isinstance(direct, Mapping):
            return dict(direct)
    training = source.get("score_aware_long_training")
    if isinstance(training, Mapping):
        contract = _snerv_score_aware_training_telemetry_contract_from_source(
            training,
            seen=seen,
        )
        if contract:
            return contract
    for key in ("score_aware_training", "snerv_mlx_native_export", "mlx_native_export"):
        nested = source.get(key)
        if isinstance(nested, Mapping):
            contract = _snerv_score_aware_training_telemetry_contract_from_source(
                nested,
                seen=seen,
            )
            if contract:
                return contract
    return {}


def _snerv_scorer_tether_smoke_gate_from_sources(*sources: Any) -> dict[str, Any]:
    for source in sources:
        gate = _snerv_scorer_tether_smoke_gate_from_source(source, seen=set())
        if gate:
            return gate
    return {}


def _snerv_scorer_tether_smoke_gate_from_source(
    source: Any,
    *,
    seen: set[int],
) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        return {}
    identity = id(source)
    if identity in seen:
        return {}
    seen.add(identity)
    for key in ("snerv_scorer_tether_smoke_gate", "scorer_tether_smoke_gate"):
        gate = source.get(key)
        if isinstance(gate, Mapping):
            return dict(gate)
    for key in (
        "score_aware_training",
        "score_aware_long_training",
        "snerv_mlx_native_export",
        "mlx_native_export",
    ):
        nested = source.get(key)
        if isinstance(nested, Mapping):
            gate = _snerv_scorer_tether_smoke_gate_from_source(nested, seen=seen)
            if gate:
                return gate
    return {}


def _snerv_metric_contract_keys(metric: str) -> tuple[str, str]:
    if "segnet" in metric:
        return "segnet_dual_metric_observed", "segnet_dual_lambda_active_observed"
    return "posenet_dual_metric_observed", "posenet_dual_lambda_active_observed"


def _snerv_scorer_tether_smoke_health(
    *,
    scorer_tether_gate: Mapping[str, Any],
    telemetry_contract: Mapping[str, Any],
    evidence_expected: bool,
) -> dict[str, Any]:
    if not evidence_expected and not scorer_tether_gate and not telemetry_contract:
        return {}
    blockers: list[str] = []
    if not scorer_tether_gate:
        blockers.extend(
            [
                "snerv_scorer_tether_smoke_report_missing",
                "snerv_scorer_domain_tether_missing_telemetry",
            ]
        )
    elif scorer_tether_gate.get("passed") is not True:
        blockers.append("snerv_scorer_tether_smoke_failed")
        blockers.extend(
            str(blocker)
            for blocker in scorer_tether_gate.get("blockers") or []
            if str(blocker)
        )
    if telemetry_contract:
        if telemetry_contract.get("passed") is not True:
            blockers.append("snerv_score_aware_long_training_telemetry_contract_failed")
            blockers.extend(
                str(blocker)
                for blocker in telemetry_contract.get("blockers") or []
                if str(blocker)
            )
    elif evidence_expected:
        blockers.append("snerv_score_aware_long_training_telemetry_contract_missing")

    final_metrics: Mapping[str, Any] = {}
    smoke_report = scorer_tether_gate.get("smoke_report")
    if isinstance(smoke_report, Mapping):
        metric_summary = smoke_report.get("metric_summary")
        if isinstance(metric_summary, Mapping):
            final = metric_summary.get("final")
            if isinstance(final, Mapping):
                final_metrics = final

    gate_passed = bool(
        scorer_tether_gate and scorer_tether_gate.get("passed") is True
    )
    contract_passed = bool(
        telemetry_contract and telemetry_contract.get("passed") is True
    )
    metric_health: dict[str, Any] = {}
    missing_metrics: list[str] = []
    lambda_inactive_metrics: list[str] = []
    for metric in _SNERV_SCORER_TETHER_METRICS:
        missing_value = _float_or_none(
            final_metrics.get(f"dual_ascent_missing_metric__{metric}")
        )
        lambda_value = _float_or_none(
            final_metrics.get(f"dual_ascent_lambda__{metric}")
        )
        metric_key, lambda_key = _snerv_metric_contract_keys(metric)
        metric_observed = (
            missing_value is not None and float(missing_value) < 0.5
        ) or bool(telemetry_contract.get(metric_key)) or bool(
            gate_passed and contract_passed
        )
        lambda_active = (
            lambda_value is not None
            and abs(float(lambda_value)) > _SNERV_SCORER_TETHER_LAMBDA_ACTIVE_EPS
        ) or bool(telemetry_contract.get(lambda_key)) or bool(
            gate_passed and contract_passed
        )
        if not metric_observed:
            missing_metrics.append(metric)
        if not lambda_active:
            lambda_inactive_metrics.append(metric)
        metric_health[metric] = {
            "missing_metric_value": missing_value,
            "lambda_value": lambda_value,
            "metric_observed": bool(metric_observed),
            "lambda_active_observed": bool(lambda_active),
        }

    if missing_metrics:
        blockers.append("snerv_scorer_domain_tether_missing_telemetry")
    if "snerv_posenet_yuv6_pair_distill" in missing_metrics:
        blockers.append("snerv_posenet_yuv6_pair_distill_metric_missing_telemetry")
    if "snerv_segnet_last_frame_distill" in missing_metrics:
        blockers.append("snerv_segnet_last_frame_distill_metric_missing_telemetry")
    if lambda_inactive_metrics:
        blockers.append("snerv_scorer_domain_tether_lambda_inactive_telemetry")
    if "snerv_segnet_last_frame_distill" in lambda_inactive_metrics:
        blockers.append(
            "snerv_score_aware_long_training_dual_segnet_lambda_never_active"
        )
    if "snerv_posenet_yuv6_pair_distill" in lambda_inactive_metrics:
        blockers.append(
            "snerv_score_aware_long_training_dual_posenet_lambda_never_active"
        )

    blockers = _dedupe_strings(blockers)
    return {
        "schema": SNERV_SCORER_TETHER_SMOKE_HEALTH_SCHEMA,
        "source": "snerv_score_aware_training_scorer_tether_smoke_gate",
        "evidence_expected": bool(evidence_expected),
        "scorer_tether_gate_attached": bool(scorer_tether_gate),
        "scorer_tether_gate_passed": (
            scorer_tether_gate.get("passed") is True if scorer_tether_gate else None
        ),
        "telemetry_contract_attached": bool(telemetry_contract),
        "telemetry_contract_passed": (
            telemetry_contract.get("passed") is True if telemetry_contract else None
        ),
        "metric_health": metric_health,
        "missing_metrics": missing_metrics,
        "lambda_inactive_metrics": lambda_inactive_metrics,
        "passed": not blockers,
        "degenerate_renderer_risk_detected": bool(blockers),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _snerv_scorer_input_distribution_guard_proof(
    *,
    native_export: Mapping[str, Any],
    score_aware_training: Mapping[str, Any],
    runner_report: Mapping[str, Any],
    telemetry_contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return proof that the train-time scorer-input guard was live telemetry."""

    if not native_export and not score_aware_training and not telemetry_contract:
        return None
    required_control_contract = {}
    for source in (native_export, score_aware_training, runner_report):
        if not isinstance(source, Mapping):
            continue
        direct = source.get("score_aware_long_training_required_control_contract")
        if isinstance(direct, Mapping):
            required_control_contract = dict(direct)
            break
        training = source.get("score_aware_long_training")
        if isinstance(training, Mapping):
            nested = training.get("required_control_contract")
            if isinstance(nested, Mapping):
                required_control_contract = dict(nested)
                break
    control = {}
    controls = required_control_contract.get("controls")
    if isinstance(controls, Mapping):
        maybe_control = controls.get("scorer_input_distribution_guard")
        if isinstance(maybe_control, Mapping):
            control = dict(maybe_control)
    score_training = (
        native_export.get("score_aware_long_training")
        if isinstance(native_export.get("score_aware_long_training"), Mapping)
        else {}
    )
    bound = bool(
        native_export.get("score_aware_long_training_scorer_input_distribution_guard_bound")
        is True
        or score_training.get("scorer_input_distribution_guard_bound") is True
        or control.get("bound") is True
    )
    required = bool(
        telemetry_contract.get("expected_scorer_input_guard_metric") is True
        or control.get("required") is True
        or bound
    )
    metric_observed = bool(
        telemetry_contract.get("scorer_input_guard_metric_observed") is True
        or control.get("telemetry_observed") is True
    )
    dual_metric_observed = bool(
        telemetry_contract.get("scorer_input_guard_dual_metric_observed") is True
        or control.get("telemetry_observed") is True
    )
    telemetry_contract_passed = bool(telemetry_contract.get("passed") is True)
    blockers: list[str] = []
    if not required:
        blockers.append("snerv_scorer_input_distribution_guard_not_required_by_feedback")
    if required and not bound:
        blockers.append("snerv_scorer_input_distribution_guard_not_bound")
    if required and not telemetry_contract:
        blockers.append("snerv_scorer_input_distribution_guard_telemetry_contract_missing")
    if required and not telemetry_contract_passed:
        blockers.append("snerv_scorer_input_distribution_guard_telemetry_contract_failed")
    if required and not metric_observed:
        blockers.append("snerv_scorer_input_distribution_guard_metric_missing")
    if required and not dual_metric_observed:
        blockers.append("snerv_scorer_input_distribution_guard_dual_metric_missing")
    blockers = _dedupe_strings(blockers)
    return {
        "schema": SNERV_SCORER_INPUT_DISTRIBUTION_GUARD_PROOF_SCHEMA,
        "required": required,
        "bound": bound,
        "telemetry_contract_passed": telemetry_contract_passed,
        "metric_observed": metric_observed,
        "dual_metric_observed": dual_metric_observed,
        "required_control_contract_passed": (
            required_control_contract.get("passed")
            if required_control_contract
            else None
        ),
        "passed": bool(required and not blockers),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _snerv_receiver_value_domain_passed(profile: Any) -> bool | None:
    if not isinstance(profile, Mapping):
        return None
    gate = profile.get("receiver_value_domain_gate")
    if isinstance(gate, Mapping):
        return bool(gate.get("passed") is True)
    blockers = profile.get("blockers")
    if isinstance(blockers, Sequence) and not isinstance(blockers, (str, bytes)):
        return not any(str(blocker) for blocker in blockers)
    return None


def _snerv_renderer_nondegenerate_proof(
    native_export: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not native_export:
        return None
    blockers: list[str] = []
    measured_pairs = _int_or_none(native_export.get("num_pairs")) or 0
    scorer_tether_gate = (
        dict(native_export.get("snerv_scorer_tether_smoke_gate"))
        if isinstance(native_export.get("snerv_scorer_tether_smoke_gate"), Mapping)
        else {}
    )
    telemetry_contract = _snerv_score_aware_training_telemetry_contract(native_export)
    reconstruction = (
        dict(native_export.get("receiver_reconstruction"))
        if isinstance(native_export.get("receiver_reconstruction"), Mapping)
        else {}
    )
    target_profile = (
        reconstruction.get("target_profile")
        if isinstance(reconstruction.get("target_profile"), Mapping)
        else native_export.get("receiver_target_reconstruction_profile")
    )
    export_profile = (
        reconstruction.get("export_profile")
        if isinstance(reconstruction.get("export_profile"), Mapping)
        else native_export.get("receiver_export_reconstruction_profile")
    )
    target_value_domain_passed = _snerv_receiver_value_domain_passed(target_profile)
    export_value_domain_passed = _snerv_receiver_value_domain_passed(export_profile)
    official_skip_gate = (
        dict(native_export.get("official_skip_high_value_domain_gate"))
        if isinstance(native_export.get("official_skip_high_value_domain_gate"), Mapping)
        else {}
    )

    if measured_pairs < SNERV_RENDERER_NONDEGENERATE_MIN_PAIR_COUNT:
        blockers.append("snerv_renderer_nondegenerate_smoke_min16_pairs_missing")
    if scorer_tether_gate.get("passed") is not True:
        blockers.append("snerv_renderer_nondegenerate_tether_gate_missing_or_failed")
    if telemetry_contract.get("passed") is not True:
        blockers.append(
            "snerv_renderer_nondegenerate_telemetry_contract_missing_or_failed"
        )
    if reconstruction.get("receiver_reconstruction_verified") is not True:
        blockers.append("snerv_renderer_nondegenerate_receiver_reconstruction_not_verified")
    if target_value_domain_passed is not True:
        blockers.append("snerv_renderer_nondegenerate_target_value_domain_not_passed")
    if export_value_domain_passed is not True:
        blockers.append("snerv_renderer_nondegenerate_export_value_domain_not_passed")
    reconstruction_blockers = [
        str(blocker)
        for blocker in reconstruction.get("blockers") or ()
        if str(blocker)
    ]
    blockers.extend(reconstruction_blockers)
    if official_skip_gate and official_skip_gate.get("passed") is not True:
        blockers.append(
            "snerv_renderer_nondegenerate_compact_skip_high_value_domain_not_passed"
        )
        blockers.extend(
            str(blocker)
            for blocker in official_skip_gate.get("blockers") or ()
            if str(blocker)
        )
    blockers = _dedupe_strings(blockers)
    return {
        "schema": SNERV_RENDERER_NONDEGENERATE_PROOF_SCHEMA,
        "min_pair_count": SNERV_RENDERER_NONDEGENERATE_MIN_PAIR_COUNT,
        "measured_num_pairs": int(measured_pairs),
        "scorer_tether_gate_passed": scorer_tether_gate.get("passed") is True,
        "telemetry_contract_passed": telemetry_contract.get("passed") is True,
        "receiver_reconstruction_verified": (
            reconstruction.get("receiver_reconstruction_verified") is True
        ),
        "target_value_domain_passed": target_value_domain_passed,
        "export_value_domain_passed": export_value_domain_passed,
        "official_skip_high_value_domain_passed": (
            official_skip_gate.get("passed") if official_skip_gate else None
        ),
        "official_skip_high_mode": (
            official_skip_gate.get("official_skip_high_mode")
            if official_skip_gate
            else native_export.get("official_skip_high_mode")
        ),
        "passed": not blockers,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _snerv_official_checkpoint_mapping_verified(
    manifest: Mapping[str, Any],
) -> bool:
    rows = [
        row
        for row in manifest.get("component_rows") or ()
        if isinstance(row, Mapping)
    ]
    return bool(
        manifest.get("official_trained_checkpoint_loaded") is True
        and rows
        and all(
            row.get("trained_checkpoint_weight_mapping_proven") is True
            for row in rows
        )
    )


def _sha256_file(path: Path) -> str | None:
    import hashlib

    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hi_nerv_receiver_cache_feedback_control(
    *,
    runner_report: Mapping[str, Any],
    byte_feedback: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize post-export receiver scorer survival for planner feedback."""

    summary = runner_report.get("post_export_receiver_cache_quality")
    if not isinstance(summary, Mapping):
        score_training = runner_report.get("score_aware_training")
        if isinstance(score_training, Mapping):
            summary = score_training.get("post_export_receiver_cache_quality")
    if not isinstance(summary, Mapping):
        summary = {}
    train_receiver_contract = runner_report.get("train_receiver_class_escape_contract")
    if not isinstance(train_receiver_contract, Mapping):
        train_receiver_contract = {}
    direct_blockers = _dedupe_strings(
        [
            *[str(blocker) for blocker in summary.get("blockers") or []],
            *[str(blocker) for blocker in byte_feedback.get("byte_oracle_blockers") or []],
            *[
                str(blocker)
                for blocker in train_receiver_contract.get("blockers") or []
            ],
        ]
    )
    candidate_occupied_fraction = _float_or_none(
        summary.get("segnet_candidate_occupied_class_fraction")
    )
    reference_occupied_fraction = _float_or_none(
        summary.get("segnet_reference_occupied_class_fraction")
    )
    numeric_collapse_detected = bool(
        candidate_occupied_fraction is not None
        and reference_occupied_fraction is not None
        and reference_occupied_fraction >= 0.400001
        and candidate_occupied_fraction < 0.400001
    )
    blocker_collapse_detected = any(
        any(fragment in blocker for fragment in _HINERV_RECEIVER_CACHE_COLLAPSE_BLOCKER_FRAGMENTS)
        for blocker in direct_blockers
    )
    collapse_detected = bool(blocker_collapse_detected or numeric_collapse_detected)
    if numeric_collapse_detected and not blocker_collapse_detected:
        direct_blockers = _dedupe_strings(
            [
                *direct_blockers,
                "hi_nerv_receiver_cache_segnet_argmax_class_collapse_numeric",
            ]
        )
    gate_failed = bool(summary and summary.get("quality_gate_passed") is not True)
    mlx_response_required = bool(summary.get("mlx_scorer_response_probe_required"))
    mlx_response_passed = summary.get("mlx_scorer_response_probe_passed")
    mlx_response_avg_posenet = _float_or_none(
        summary.get("mlx_scorer_response_avg_posenet_dist")
    )
    mlx_response_avg_segnet = _float_or_none(
        summary.get("mlx_scorer_response_avg_segnet_dist")
    )
    posenet_response_too_high = any(
        "hi_nerv_receiver_cache_posenet_response_too_high" in blocker
        for blocker in direct_blockers
    )
    segnet_response_too_high = any(
        "hi_nerv_receiver_cache_segnet_response_too_high" in blocker
        for blocker in direct_blockers
    )
    mlx_response_failed = bool(
        mlx_response_required
        and (
            mlx_response_passed is not True
            or posenet_response_too_high
            or segnet_response_too_high
        )
    )
    mutations: list[str] = []
    if collapse_detected:
        mutations.extend(
            [
                "increase_hi_nerv_receiver_class_survival_pressure",
                "disable_hi_nerv_byte_feedback_learning_from_receiver_collapsed_export",
            ]
        )
    if posenet_response_too_high:
        mutations.append("increase_hi_nerv_posenet_response_pressure")
    if segnet_response_too_high:
        mutations.append("increase_hi_nerv_segnet_response_pressure")
    if mlx_response_failed:
        mutations.append("rerun_hi_nerv_short_probe_with_mlx_scorer_response_gate")
    elif gate_failed:
        mutations.append("rerun_hi_nerv_short_probe_with_receiver_cache_quality_gate")
    return {
        "schema": "hi_nerv_receiver_cache_feedback_control.v1",
        "quality_gate_passed": (
            summary.get("quality_gate_passed") if summary else None
        ),
        "segnet_candidate_occupied_class_fraction": candidate_occupied_fraction,
        "segnet_candidate_any_occupied_class_fraction": summary.get(
            "segnet_candidate_any_occupied_class_fraction"
        ),
        "segnet_reference_occupied_class_fraction": reference_occupied_fraction,
        "segnet_reference_any_occupied_class_fraction": summary.get(
            "segnet_reference_any_occupied_class_fraction"
        ),
        "segnet_argmax_occupancy_min_class_pixel_count": summary.get(
            "segnet_argmax_occupancy_min_class_pixel_count"
        ),
        "segnet_argmax_disagreement_rate": summary.get(
            "segnet_argmax_disagreement_rate"
        ),
        "mlx_scorer_response_probe_required": mlx_response_required,
        "mlx_scorer_response_probe_passed": mlx_response_passed,
        "mlx_scorer_response_avg_posenet_dist": mlx_response_avg_posenet,
        "mlx_scorer_response_avg_segnet_dist": mlx_response_avg_segnet,
        "mlx_scorer_response_failed": mlx_response_failed,
        "posenet_response_too_high": posenet_response_too_high,
        "segnet_response_too_high": segnet_response_too_high,
        "byte_oracle_feedback_ready": byte_feedback.get("byte_oracle_feedback_ready"),
        "byte_oracle_post_export_receiver_cache_quality_feedback_ready": (
            byte_feedback.get(
                "byte_oracle_post_export_receiver_cache_quality_feedback_ready"
            )
        ),
        "collapse_detected": collapse_detected,
        "numeric_collapse_detected": numeric_collapse_detected,
        "blocker_collapse_detected": blocker_collapse_detected,
        "gate_failed": gate_failed,
        "direct_feedback_blockers": direct_blockers,
        "recommended_launch_mutations": _dedupe_strings(mutations),
        "launch_control_feedback_ready": bool(
            collapse_detected or gate_failed or mlx_response_failed
        ),
        **FALSE_AUTHORITY,
    }


def build_nerv_candidate_feedback_row(
    *,
    runner_report: Mapping[str, Any],
    source_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one false-authority feedback row from a runner report."""

    selection = dict(runner_report.get("modelsize_candidate_selection") or {})
    curriculum = dict(
        runner_report.get("candidate_curriculum_plan")
        or selection.get("candidate_curriculum_plan")
        or {}
    )
    byte_feedback = dict(curriculum.get("byte_oracle_logging") or {})
    pr95_binding = dict(
        curriculum.get("pr95_stack_binding")
        or selection.get("pr95_stack_binding")
        or {}
    )
    prelaunch_gate = dict(
        curriculum.get("long_campaign_prelaunch_gate")
        or selection.get("long_campaign_prelaunch_gate")
        or {}
    )
    candidate = selection.get("candidate")
    candidate_row = dict(candidate) if isinstance(candidate, Mapping) else {}
    source_path = (
        Path(source_report_path).expanduser().resolve(strict=False)
        if source_report_path
        else None
    )
    local_replay = runner_report.get("local_cpu_replay_summary")
    local_replay_gate = dict(runner_report.get("local_cpu_replay_gate") or {})
    mlx_prefilter = dict(runner_report.get("mlx_prefilter_coverage") or {})
    snerv_profile = dict(runner_report.get("snerv_binary_profile") or {})
    score_aware_training = dict(runner_report.get("score_aware_training") or {})
    snerv_native_export = _mapping_or_empty(
        runner_report.get("snerv_mlx_native_export")
        or score_aware_training.get("mlx_native_export")
    )
    snerv_native_evidence = _mapping_or_empty(
        runner_report.get("snerv_mlx_native_file_backed_export_evidence")
        or score_aware_training.get("mlx_native_file_backed_export_evidence")
        or _mapping_or_empty(curriculum.get("training_plan")).get(
            "native_mlx_file_backed_export_evidence"
        )
    )
    snerv_native_packet_metadata = _mapping_or_empty(
        snerv_native_evidence.get("packet_metadata_summary")
    )
    snerv_official_checkpoint_mapping = (
        _snerv_official_checkpoint_mapping_manifest(
            snerv_native_export,
            snerv_native_evidence,
            snerv_native_packet_metadata,
            score_aware_training,
            runner_report,
        )
    )
    snerv_trained_state_exportable = _snerv_trained_state_exportability(
        snerv_native_export,
        snerv_native_evidence,
        snerv_native_packet_metadata,
        score_aware_training,
        runner_report,
    )
    snerv_native_scorer_loop = _mapping_or_empty(
        snerv_native_export.get("scorer_loop_qat")
    )
    snerv_native_training_guard = (
        build_snerv_mlx_native_training_export_guard(snerv_native_export)
        if snerv_native_export
        else {}
    )
    snerv_native_receiver_reconstruction = _mapping_or_empty(
        snerv_native_export.get("receiver_reconstruction")
        or score_aware_training.get("mlx_native_receiver_reconstruction")
    )
    snerv_scorer_tether_gate = _snerv_scorer_tether_smoke_gate_from_sources(
        snerv_native_export,
        score_aware_training,
        runner_report,
    )
    snerv_score_aware_training_telemetry_contract = (
        _snerv_score_aware_training_telemetry_contract_from_sources(
            snerv_native_export,
            score_aware_training,
            runner_report,
        )
    )
    snerv_native_export_for_proof = dict(snerv_native_export)
    if (
        snerv_scorer_tether_gate
        and "snerv_scorer_tether_smoke_gate" not in snerv_native_export_for_proof
    ):
        snerv_native_export_for_proof["snerv_scorer_tether_smoke_gate"] = (
            snerv_scorer_tether_gate
        )
    if (
        snerv_score_aware_training_telemetry_contract
        and "score_aware_long_training_telemetry_contract"
        not in snerv_native_export_for_proof
    ):
        snerv_native_export_for_proof[
            "score_aware_long_training_telemetry_contract"
        ] = snerv_score_aware_training_telemetry_contract
    snerv_family_key = _family_key(str(runner_report.get("execute_family") or ""))
    snerv_tether_evidence_expected = bool(
        snerv_family_key == "snerv" and (snerv_native_export or score_aware_training)
    )
    snerv_scorer_tether_health = _snerv_scorer_tether_smoke_health(
        scorer_tether_gate=snerv_scorer_tether_gate,
        telemetry_contract=snerv_score_aware_training_telemetry_contract,
        evidence_expected=snerv_tether_evidence_expected,
    )
    snerv_scorer_input_distribution_guard_proof = (
        _snerv_scorer_input_distribution_guard_proof(
            native_export=snerv_native_export,
            score_aware_training=score_aware_training,
            runner_report=runner_report,
            telemetry_contract=snerv_score_aware_training_telemetry_contract,
        )
        if snerv_family_key == "snerv"
        else None
    )
    snerv_renderer_nondegenerate_proof = _snerv_renderer_nondegenerate_proof(
        snerv_native_export_for_proof
    )
    native_byte_feedback = _snerv_native_file_backed_byte_feedback(
        family=runner_report.get("execute_family"),
        candidate_row=candidate_row,
        byte_feedback=byte_feedback,
        native_export=snerv_native_export,
        native_evidence=snerv_native_evidence,
    )
    feedback_source = native_byte_feedback.get("byte_feedback_source") or byte_feedback.get(
        "byte_feedback_source"
    )
    candidate_num_pairs = native_byte_feedback.get(
        "candidate_num_pairs", byte_feedback.get("candidate_num_pairs")
    )
    measured_num_pairs = native_byte_feedback.get(
        "measured_num_pairs", byte_feedback.get("measured_num_pairs")
    )
    feedback_scope = native_byte_feedback.get(
        "feedback_scope", byte_feedback.get("feedback_scope")
    )
    scope_matches_candidate = native_byte_feedback.get(
        "scope_matches_candidate", bool(byte_feedback.get("scope_matches_candidate"))
    )
    feedback_ready = native_byte_feedback.get(
        "feedback_ready", bool(byte_feedback.get("feedback_ready"))
    )
    measured_payload_bytes = native_byte_feedback.get(
        "measured_payload_bytes", byte_feedback.get("measured_payload_bytes")
    )
    measured_archive_bytes = native_byte_feedback.get(
        "measured_archive_bytes", byte_feedback.get("measured_archive_bytes")
    )
    measured_minus_nominal_bytes = native_byte_feedback.get(
        "measured_minus_nominal_bytes",
        byte_feedback.get("measured_minus_nominal_bytes"),
    )
    hi_nerv_receiver_cache_control = (
        _hi_nerv_receiver_cache_feedback_control(
            runner_report=runner_report,
            byte_feedback=byte_feedback,
        )
        if _family_key(str(runner_report.get("execute_family") or "")) == "hi_nerv"
        else {}
    )
    sample_generalization_gate = _sample_generalization_gate(
        runner_report=runner_report,
        curriculum=curriculum,
        selection=selection,
        byte_feedback=byte_feedback,
        candidate_num_pairs=candidate_num_pairs,
        measured_num_pairs=measured_num_pairs,
        mlx_prefilter=mlx_prefilter,
        local_replay=local_replay,
    )
    blockers = _dedupe_strings(
        [
            *[str(blocker) for blocker in runner_report.get("blockers") or []],
            *sample_generalization_gate["blockers"],
            *[
                str(blocker)
                for blocker in snerv_native_training_guard.get("blockers") or []
            ],
            *[
                str(blocker)
                for blocker in snerv_native_receiver_reconstruction.get("blockers")
                or []
            ],
            *[
                str(blocker)
                for blocker in snerv_scorer_tether_health.get("blockers") or []
            ],
            *(
                [
                    str(blocker)
                    for blocker in (
                        snerv_scorer_input_distribution_guard_proof.get("blockers")
                        or []
                    )
                    if blocker
                    != "snerv_scorer_input_distribution_guard_not_required_by_feedback"
                ]
                if isinstance(
                    snerv_scorer_input_distribution_guard_proof, Mapping
                )
                else []
            ),
            *[
                str(blocker)
                for blocker in snerv_official_checkpoint_mapping.get("blockers")
                or []
            ],
            *(
                [
                    str(blocker)
                    for blocker in (
                        snerv_renderer_nondegenerate_proof.get("blockers") or []
                    )
                ]
                if isinstance(snerv_renderer_nondegenerate_proof, Mapping)
                else []
            ),
        ]
    )
    direct_feedback_blockers = _dedupe_strings(
        [
            *list(hi_nerv_receiver_cache_control.get("direct_feedback_blockers") or []),
            *list(snerv_scorer_tether_health.get("blockers") or []),
            *(
                [
                    str(blocker)
                    for blocker in (
                        snerv_scorer_input_distribution_guard_proof.get("blockers")
                        or []
                    )
                    if blocker
                    != "snerv_scorer_input_distribution_guard_not_required_by_feedback"
                ]
                if isinstance(
                    snerv_scorer_input_distribution_guard_proof, Mapping
                )
                else []
            ),
        ]
    )
    recommended_launch_mutations = _dedupe_strings(
        hi_nerv_receiver_cache_control.get("recommended_launch_mutations") or []
    )
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_report_path": source_path.as_posix() if source_path else None,
        "source_report_sha256": _sha256_file(source_path) if source_path else None,
        "mode": runner_report.get("mode"),
        "family": runner_report.get("execute_family"),
        "candidate_id": candidate_row.get("candidate_id") or curriculum.get("candidate_id"),
        "candidate_conditioned": bool(curriculum.get("candidate_conditioned")),
        "candidate_num_pairs": candidate_num_pairs,
        "measured_num_pairs": measured_num_pairs,
        "feedback_scope": feedback_scope,
        "scope_matches_candidate": bool(scope_matches_candidate),
        "feedback_ready": bool(feedback_ready),
        "byte_feedback_source": feedback_source,
        "hard_byte_ceiling": byte_feedback.get("hard_byte_ceiling"),
        "nominal_total_payload_bytes": byte_feedback.get("nominal_total_payload_bytes"),
        "measured_payload_bytes": measured_payload_bytes,
        "measured_archive_bytes": measured_archive_bytes,
        "measured_minus_nominal_bytes": measured_minus_nominal_bytes,
        "hi_nerv_receiver_cache_feedback_control": (
            hi_nerv_receiver_cache_control or None
        ),
        "post_export_receiver_cache_quality_gate_passed": (
            hi_nerv_receiver_cache_control.get("quality_gate_passed")
            if hi_nerv_receiver_cache_control
            else None
        ),
        "post_export_receiver_segnet_candidate_occupied_class_fraction": (
            hi_nerv_receiver_cache_control.get(
                "segnet_candidate_occupied_class_fraction"
            )
            if hi_nerv_receiver_cache_control
            else None
        ),
        "post_export_receiver_segnet_candidate_any_occupied_class_fraction": (
            hi_nerv_receiver_cache_control.get(
                "segnet_candidate_any_occupied_class_fraction"
            )
            if hi_nerv_receiver_cache_control
            else None
        ),
        "post_export_receiver_segnet_reference_occupied_class_fraction": (
            hi_nerv_receiver_cache_control.get(
                "segnet_reference_occupied_class_fraction"
            )
            if hi_nerv_receiver_cache_control
            else None
        ),
        "post_export_receiver_segnet_reference_any_occupied_class_fraction": (
            hi_nerv_receiver_cache_control.get(
                "segnet_reference_any_occupied_class_fraction"
            )
            if hi_nerv_receiver_cache_control
            else None
        ),
        "post_export_receiver_segnet_occupancy_min_class_pixel_count": (
            hi_nerv_receiver_cache_control.get(
                "segnet_argmax_occupancy_min_class_pixel_count"
            )
            if hi_nerv_receiver_cache_control
            else None
        ),
        "post_export_receiver_class_collapse_detected": bool(
            hi_nerv_receiver_cache_control.get("collapse_detected")
            if hi_nerv_receiver_cache_control
            else False
        ),
        "post_export_receiver_mlx_scorer_response_probe_required": (
            hi_nerv_receiver_cache_control.get("mlx_scorer_response_probe_required")
            if hi_nerv_receiver_cache_control
            else None
        ),
        "post_export_receiver_mlx_scorer_response_probe_passed": (
            hi_nerv_receiver_cache_control.get("mlx_scorer_response_probe_passed")
            if hi_nerv_receiver_cache_control
            else None
        ),
        "post_export_receiver_mlx_scorer_response_avg_posenet_dist": (
            hi_nerv_receiver_cache_control.get("mlx_scorer_response_avg_posenet_dist")
            if hi_nerv_receiver_cache_control
            else None
        ),
        "post_export_receiver_mlx_scorer_response_avg_segnet_dist": (
            hi_nerv_receiver_cache_control.get("mlx_scorer_response_avg_segnet_dist")
            if hi_nerv_receiver_cache_control
            else None
        ),
        "post_export_receiver_posenet_response_too_high": (
            hi_nerv_receiver_cache_control.get("posenet_response_too_high")
            if hi_nerv_receiver_cache_control
            else None
        ),
        "post_export_receiver_segnet_response_too_high": (
            hi_nerv_receiver_cache_control.get("segnet_response_too_high")
            if hi_nerv_receiver_cache_control
            else None
        ),
        "recommended_launch_mutations": recommended_launch_mutations,
        "launch_control_feedback_ready": bool(
            hi_nerv_receiver_cache_control.get("launch_control_feedback_ready")
            if hi_nerv_receiver_cache_control
            else False
        ),
        "direct_feedback_blockers": direct_feedback_blockers,
        "archive_path": runner_report.get("archive_path"),
        "archive_bytes": runner_report.get("archive_bytes"),
        "archive_sha256": runner_report.get("archive_sha256"),
        "snerv_mlx_native_export_executed": snerv_native_export.get("executed"),
        "snerv_mlx_native_export_artifact_report_path": (
            snerv_native_export.get("artifact_report_path")
            or snerv_native_export.get("report_path")
        ),
        "snerv_mlx_native_export_packet_path": snerv_native_export.get("packet_path"),
        "snerv_mlx_native_export_packet_bytes": snerv_native_export.get("packet_bytes"),
        "snerv_mlx_native_export_packet_sha256": snerv_native_export.get(
            "packet_sha256"
        ),
        "snerv_mlx_native_export_archive_path": snerv_native_export.get("archive_path"),
        "snerv_mlx_native_export_archive_bytes": snerv_native_export.get(
            "archive_bytes"
        ),
        "snerv_mlx_native_export_archive_sha256": snerv_native_export.get(
            "archive_sha256"
        ),
        "snerv_mlx_native_export_receiver_proof_path": snerv_native_export.get(
            "receiver_proof_path"
        ),
        "snerv_mlx_native_export_receiver_proof_passed": snerv_native_export.get(
            "receiver_proof_passed"
        ),
        "snerv_mlx_native_export_receiver_contract_satisfied": (
            snerv_native_export.get("receiver_contract_satisfied")
        ),
        "snerv_official_mfu_hfr_tub_numeric_primitives_requested": _first_present(
            snerv_native_export,
            snerv_native_evidence,
            snerv_native_packet_metadata,
            "snerv_official_mfu_hfr_tub_numeric_primitives_requested",
        ),
        "snerv_official_mfu_hfr_tub_export_bound": _first_present(
            snerv_native_export,
            snerv_native_evidence,
            snerv_native_packet_metadata,
            "snerv_official_mfu_hfr_tub_export_bound",
        ),
        "snerv_official_mfu_hfr_tub_export_bound_semantics": _first_present(
            snerv_native_export,
            snerv_native_evidence,
            snerv_native_packet_metadata,
            "snerv_official_mfu_hfr_tub_export_bound_semantics",
        ),
        "snerv_official_mfu_hfr_tub_receiver_payload_bound": _first_present(
            snerv_native_export,
            snerv_native_evidence,
            snerv_native_packet_metadata,
            "snerv_official_mfu_hfr_tub_receiver_payload_bound",
        ),
        "snerv_official_mfu_hfr_tub_frame_producing_export": _first_present(
            snerv_native_export,
            snerv_native_evidence,
            snerv_native_packet_metadata,
            "snerv_official_mfu_hfr_tub_frame_producing_export",
        ),
        "snerv_official_mfu_hfr_tub_source_forward_replay_bound": _first_present(
            snerv_native_export,
            snerv_native_evidence,
            snerv_native_packet_metadata,
            "snerv_official_mfu_hfr_tub_source_forward_replay_bound",
        ),
        "snerv_official_mfu_hfr_tub_source_forward_replay_authority": _first_present(
            snerv_native_export,
            snerv_native_evidence,
            snerv_native_packet_metadata,
            "snerv_official_mfu_hfr_tub_source_forward_replay_authority",
        ),
        "source_faithful_stack": _first_present(
            snerv_native_export,
            snerv_native_evidence,
            snerv_native_packet_metadata,
            "source_faithful_stack",
        ),
        "official_source_parity_blockers": list(
            snerv_native_export.get("official_source_parity_blockers")
            or snerv_native_export.get("snerv_official_mfu_hfr_tub_export_blockers")
            or snerv_native_evidence.get("official_source_parity_blockers")
            or snerv_native_evidence.get("snerv_official_mfu_hfr_tub_export_blockers")
            or snerv_native_packet_metadata.get("official_source_parity_blockers")
            or []
        ),
        "snerv_official_trained_checkpoint_mapping_manifest": (
            snerv_official_checkpoint_mapping or None
        ),
        "snerv_official_trained_checkpoint_loaded": (
            snerv_official_checkpoint_mapping.get("official_trained_checkpoint_loaded")
            if snerv_official_checkpoint_mapping
            else None
        ),
        "snerv_official_hfr_trained_checkpoint_weight_mapping_proven": (
            snerv_official_checkpoint_mapping.get(
                "official_hfr_trained_checkpoint_weight_mapping_proven"
            )
            if snerv_official_checkpoint_mapping
            else None
        ),
        "snerv_official_mfu_trained_checkpoint_weight_mapping_proven": (
            snerv_official_checkpoint_mapping.get(
                "official_mfu_trained_checkpoint_weight_mapping_proven"
            )
            if snerv_official_checkpoint_mapping
            else None
        ),
        "snerv_official_mfu_hfr_trained_checkpoint_weight_mapping_proven": (
            snerv_official_checkpoint_mapping.get(
                "official_mfu_hfr_trained_checkpoint_weight_mapping_proven"
            )
            if snerv_official_checkpoint_mapping
            else None
        ),
        "snerv_official_mfu_receiver_activation_payload_bound": (
            snerv_official_checkpoint_mapping.get(
                "official_mfu_receiver_activation_payload_bound"
            )
            if snerv_official_checkpoint_mapping
            else None
        ),
        "snerv_official_tub_receiver_activation_payload_bound": (
            snerv_official_checkpoint_mapping.get(
                "official_tub_receiver_activation_payload_bound"
            )
            if snerv_official_checkpoint_mapping
            else None
        ),
        "snerv_official_native_receiver_state_mapping_proven": (
            snerv_official_checkpoint_mapping.get(
                "official_native_receiver_state_mapping_proven"
            )
            if snerv_official_checkpoint_mapping
            else None
        ),
        "snerv_official_tub_temporal_encoder_weight_mapping_proven": (
            snerv_official_checkpoint_mapping.get(
                "official_tub_temporal_encoder_weight_mapping_proven"
            )
            if snerv_official_checkpoint_mapping
            else None
        ),
        "snerv_official_trained_checkpoint_state_dict_mapping_verified": (
            _snerv_official_checkpoint_mapping_verified(
                snerv_official_checkpoint_mapping
            )
            if snerv_official_checkpoint_mapping
            else None
        ),
        "snerv_trained_state_exportable": snerv_trained_state_exportable,
        "snerv_checkpoint_trained_state_exportable": _first_present(
            snerv_native_export,
            snerv_native_evidence,
            snerv_native_packet_metadata,
            "checkpoint_trained_state_exportable",
        ),
        "snerv_score_aware_long_training_trained_state_exportable": _first_present(
            snerv_native_export,
            snerv_native_evidence,
            snerv_native_packet_metadata,
            "score_aware_long_training_trained_state_exportable",
        ),
        "snerv_official_trained_checkpoint_mapping_blockers": list(
            snerv_official_checkpoint_mapping.get("blockers") or []
        ),
        "snerv_mlx_native_export_packet_source": snerv_native_export.get(
            "packet_source"
        ),
        "snerv_mlx_native_training_executed": snerv_native_export.get(
            "native_mlx_training_executed"
        ),
        "snerv_mlx_native_training_kind": snerv_native_export.get(
            "native_mlx_training_kind"
        ),
        "snerv_mlx_native_hf_decoder_training": (
            snerv_native_export.get("native_mlx_hf_decoder_training")
        ),
        "snerv_mlx_native_training_export_guard": (
            snerv_native_training_guard or None
        ),
        "snerv_mlx_native_training_export_guard_passed": (
            snerv_native_training_guard.get("export_guard_passed")
            if snerv_native_training_guard
            else None
        ),
        "snerv_mlx_native_training_export_guard_blockers": list(
            snerv_native_training_guard.get("blockers") or []
        ),
        "snerv_mlx_native_receiver_reconstruction_verified": (
            snerv_native_receiver_reconstruction.get(
                "receiver_reconstruction_verified"
            )
        ),
        "snerv_mlx_native_receiver_target_mse_nchw255": (
            snerv_native_receiver_reconstruction.get("target_mse_nchw255")
        ),
        "snerv_mlx_native_receiver_target_max_abs_nchw255": (
            snerv_native_receiver_reconstruction.get("target_max_abs_nchw255")
        ),
        "snerv_mlx_native_receiver_export_mse_nchw255": (
            snerv_native_receiver_reconstruction.get("export_mse_nchw255")
        ),
        "snerv_mlx_native_receiver_export_max_abs_nchw255": (
            snerv_native_receiver_reconstruction.get("export_max_abs_nchw255")
        ),
        "snerv_mlx_native_receiver_reconstruction_blockers": list(
            snerv_native_receiver_reconstruction.get("blockers") or []
        ),
        "snerv_mlx_native_receiver_reconstruction": (
            snerv_native_receiver_reconstruction or None
        ),
        "snerv_scorer_tether_smoke_gate": snerv_scorer_tether_gate or None,
        "snerv_score_aware_long_training_telemetry_contract": (
            snerv_score_aware_training_telemetry_contract or None
        ),
        "snerv_scorer_domain_tether_health": (
            snerv_scorer_tether_health or None
        ),
        "snerv_scorer_domain_tether_passed": (
            snerv_scorer_tether_health.get("passed")
            if snerv_scorer_tether_health
            else None
        ),
        "snerv_scorer_domain_tether_blockers": list(
            snerv_scorer_tether_health.get("blockers") or []
        ),
        "snerv_scorer_input_distribution_guard_proof": (
            snerv_scorer_input_distribution_guard_proof or None
        ),
        "snerv_scorer_input_distribution_guard_proof_passed": (
            snerv_scorer_input_distribution_guard_proof.get("passed")
            if isinstance(snerv_scorer_input_distribution_guard_proof, Mapping)
            else None
        ),
        "snerv_scorer_input_distribution_guard_blockers": (
            list(snerv_scorer_input_distribution_guard_proof.get("blockers") or [])
            if isinstance(snerv_scorer_input_distribution_guard_proof, Mapping)
            else []
        ),
        "snerv_renderer_nondegenerate_proof": (
            snerv_renderer_nondegenerate_proof or None
        ),
        "snerv_renderer_nondegenerate_proof_passed": (
            snerv_renderer_nondegenerate_proof.get("passed")
            if isinstance(snerv_renderer_nondegenerate_proof, Mapping)
            else None
        ),
        "snerv_renderer_nondegenerate_smoke_min_pair_count": (
            SNERV_RENDERER_NONDEGENERATE_MIN_PAIR_COUNT
        ),
        "snerv_renderer_nondegenerate_blockers": (
            list(snerv_renderer_nondegenerate_proof.get("blockers") or [])
            if isinstance(snerv_renderer_nondegenerate_proof, Mapping)
            else []
        ),
        "snerv_mlx_native_export_blockers": list(
            snerv_native_export.get("blockers") or []
        ),
        "snerv_mlx_native_file_backed_export_proof_passed": (
            snerv_native_evidence.get("file_backed_export_proof_passed")
        ),
        "snerv_mlx_native_required_pair_file_backed_export_proof_passed": (
            snerv_native_evidence.get("required_pair_file_backed_export_proof_passed")
        ),
        "snerv_mlx_native_file_backed_export_blockers": list(
            snerv_native_evidence.get("blockers") or []
        ),
        "snerv_mlx_native_file_backed_export_evidence": snerv_native_evidence or None,
        "snerv_mlx_native_file_backed_byte_feedback": native_byte_feedback or None,
        "snerv_mlx_native_scorer_loop_qat_attached": (
            snerv_native_export.get("scorer_loop_qat_attached")
            if "scorer_loop_qat_attached" in snerv_native_export
            else snerv_native_scorer_loop.get("executed")
        ),
        "snerv_mlx_native_scorer_loop_qat_accepted_improvement": (
            snerv_native_export.get("scorer_loop_qat_accepted_improvement")
            if "scorer_loop_qat_accepted_improvement" in snerv_native_export
            else snerv_native_scorer_loop.get("accepted_improvement")
        ),
        "snerv_mlx_native_scorer_loop_qat_best_materialized": (
            snerv_native_export.get("scorer_loop_qat_best_materialized")
            if "scorer_loop_qat_best_materialized" in snerv_native_export
            else snerv_native_scorer_loop.get(
                "emitted_packet_uses_scorer_loop_best_decoder"
            )
            or snerv_native_scorer_loop.get("best_packet_materialized")
        ),
        "snerv_binary_profile_path": snerv_profile.get("profile_path"),
        "snerv_binary_profile_written": bool(snerv_profile.get("profile_written")),
        "snerv_binary_profile_verdict": snerv_profile.get("verdict"),
        "snerv_binary_profile_charged_archive_bytes": snerv_profile.get(
            "charged_archive_bytes"
        ),
        "snerv_binary_profile_snar1_packet_bytes": snerv_profile.get(
            "snar1_packet_bytes"
        ),
        "snerv_binary_profile_lf_payload_bytes": snerv_profile.get(
            "lf_payload_bytes"
        ),
        "snerv_binary_profile_lf_payload_fraction_of_packet": snerv_profile.get(
            "lf_payload_fraction_of_packet"
        ),
        "snerv_binary_profile_lf_payload_bytes_per_coeff": snerv_profile.get(
            "lf_payload_bytes_per_coeff"
        ),
        "snerv_binary_profile_blockers": list(snerv_profile.get("blockers") or []),
        "pr95_stack_binding_schema": pr95_binding.get("schema"),
        "pr95_stack_binding_satisfied_count": pr95_binding.get("satisfied_count"),
        "pr95_stack_binding_missing_count": pr95_binding.get("missing_count"),
        "pr95_stack_binding_complete": pr95_binding.get("complete"),
        "pr95_stack_binding_blockers": list(pr95_binding.get("blockers") or []),
        "long_campaign_prelaunch_gate_schema": prelaunch_gate.get("schema"),
        "long_campaign_prelaunch_launch_allowed": prelaunch_gate.get(
            "launch_allowed"
        ),
        "long_campaign_prelaunch_blockers": list(prelaunch_gate.get("blockers") or []),
        "receiver_proof_report_paths": list(
            runner_report.get("receiver_proof_report_paths") or []
        ),
        "local_cpu_replay_summary_present": isinstance(local_replay, Mapping),
        "local_cpu_replay_score_estimate": (
            local_replay.get("local_score_estimate")
            if isinstance(local_replay, Mapping)
            else None
        ),
        "local_cpu_replay_gate_requested": local_replay_gate.get("requested"),
        "local_cpu_replay_gate_default_enabled_for_full_coverage": (
            local_replay_gate.get("default_enabled_for_full_coverage")
        ),
        "local_cpu_replay_gate_has_full_video_mlx_prefilter": (
            local_replay_gate.get("has_full_video_mlx_prefilter")
        ),
        "local_cpu_replay_gate_local_replay_mlx_prefilter_passed": (
            local_replay_gate.get("local_replay_mlx_prefilter_passed")
        ),
        "local_cpu_replay_gate_coverage_valid_for_replay": (
            local_replay_gate.get("coverage_valid_for_replay")
        ),
        "local_cpu_replay_gate_executed": local_replay_gate.get("executed"),
        "mlx_prefilter_profile_count": mlx_prefilter.get("profile_count"),
        "mlx_prefilter_has_full_video": mlx_prefilter.get(
            "has_full_video_mlx_prefilter"
        ),
        "mlx_prefilter_local_replay_passed": mlx_prefilter.get(
            "local_replay_mlx_prefilter_passed"
        ),
        "mlx_prefilter_best_full_video_mlx_score": mlx_prefilter.get(
            "best_full_video_mlx_score"
        ),
        "mlx_prefilter_full_video_profile_paths": list(
            mlx_prefilter.get("full_video_profile_paths") or []
        ),
        "mlx_prefilter_local_replay_profile_paths": list(
            mlx_prefilter.get("local_replay_profile_paths") or []
        ),
        "mlx_prefilter_blockers": list(mlx_prefilter.get("blockers") or []),
        "sample_generalization_gate": sample_generalization_gate,
        "sample_generalization_gate_schema": sample_generalization_gate["schema"],
        "sample_generalization_verdict": sample_generalization_gate["verdict"],
        "sample_generalization_representative_distortion_evidence": (
            sample_generalization_gate["representative_distortion_evidence"]
        ),
        "sample_generalization_small_pair_smoke_only": sample_generalization_gate[
            "small_pair_smoke_only"
        ],
        "sample_generalization_blockers": list(
            sample_generalization_gate["blockers"]
        ),
        "sample_generalization_recommended_next_actions": list(
            sample_generalization_gate["recommended_next_actions"]
        ),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def build_hinerv_archive_ladder_feedback_report(
    *,
    archive_ladder_report: Mapping[str, Any],
    source_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convert HiNeRV archive-ladder rows into planner feedback rows.

    The ladder is full-scope, receiver-closed rate evidence. It is explicitly
    not scorer evidence, so the generated rows feed measured archive bytes and
    receiver-proof custody into the long-training planner while preserving
    blockers for MLX prefilter, local replay, and exact auth.
    """

    if archive_ladder_report.get("schema") != "hinerv_archive_size_ladder.v1":
        raise ValueError(
            "archive_ladder_report must have schema 'hinerv_archive_size_ladder.v1'"
        )
    source_path = (
        Path(source_report_path).expanduser().resolve(strict=False)
        if source_report_path
        else None
    )
    source_sha = _sha256_file(source_path) if source_path else None
    num_pairs = _int_or_none(archive_ladder_report.get("num_pairs")) or CONTEST_PAIR_COUNT
    rows = [
        _hinerv_archive_ladder_feedback_row(
            row,
            num_pairs=int(num_pairs),
            source_report_path=source_path.as_posix() if source_path else None,
            source_report_sha256=source_sha,
        )
        for row in archive_ladder_report.get("archive_rows") or ()
        if isinstance(row, Mapping)
    ]
    blockers = _dedupe_strings(
        [
            "hinerv_archive_ladder_feedback_is_rate_only",
            "full_video_mlx_prefilter_missing",
            "local_cpu_replay_gate_missing",
            "contest_cpu_cuda_exact_eval_not_executed",
            *(["hinerv_archive_ladder_feedback_rows_missing"] if not rows else []),
            *[
                str(blocker)
                for row in rows
                for blocker in row.get("direct_feedback_blockers", ())
                if blocker
            ],
        ]
    )
    return {
        "schema": HINERV_ARCHIVE_LADDER_FEEDBACK_SCHEMA,
        "source_schema": archive_ladder_report.get("schema"),
        "source_report_path": source_path.as_posix() if source_path else None,
        "source_report_sha256": source_sha,
        "family": "hi_nerv",
        "feedback_kind": "receiver_closed_archive_ladder_rate_only",
        "row_count": len(rows),
        "num_pairs": int(num_pairs),
        "rows": rows,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def build_nerv_full_video_mlx_scorer_feedback_row(
    *,
    mlx_response: Mapping[str, Any],
    archive_export_report: Mapping[str, Any] | None = None,
    mlx_response_path: str | Path | None = None,
    archive_export_report_path: str | Path | None = None,
    candidate_id: str | None = None,
    family: str | None = None,
    hard_byte_ceiling: int | None = None,
    current_segnet_distillation_weight: float | None = None,
    max_mlx_score_for_local_replay: float | None = (
        DEFAULT_MAX_MLX_SCORE_FOR_LOCAL_REPLAY
    ),
) -> dict[str, Any]:
    """Bind a full-video MLX scorer response to a candidate feedback row.

    The MLX scorer response is local research signal only. Binding it to the
    matching checkpoint/archive export by archive SHA makes it useful for
    training control without granting score or promotion authority.
    """

    if max_mlx_score_for_local_replay is None:
        max_mlx_score_for_local_replay = DEFAULT_MAX_MLX_SCORE_FOR_LOCAL_REPLAY
    response_schema = str(
        mlx_response.get("schema") or mlx_response.get("schema_version") or ""
    )
    if response_schema != "mlx_scorer_response.v1":
        raise ValueError(
            "mlx_response must have schema/schema_version 'mlx_scorer_response.v1'"
        )
    export = _normalize_full_video_archive_export_report(
        archive_export_report or {}
    )
    response_path = (
        Path(mlx_response_path).expanduser().resolve(strict=False)
        if mlx_response_path
        else None
    )
    export_path = (
        Path(archive_export_report_path).expanduser().resolve(strict=False)
        if archive_export_report_path
        else None
    )
    family_key = _family_key(
        str(
            family
            or export.get("family")
            or mlx_response.get("response_family")
            or ""
        )
    )
    resolved_candidate_id = str(
        candidate_id
        or export.get("candidate_id")
        or export.get("modelsize_candidate_id")
        or _path_get(export, ("modelsize_candidate", "candidate_id"))
        or ""
    ).strip()
    response_archive_sha = str(
        mlx_response.get("archive_sha256")
        or _path_get(mlx_response, ("cache_identity", "candidate", "archive_sha256"))
        or ""
    ).strip()
    export_archive_sha = str(export.get("archive_sha256") or "").strip()
    archive_sha = export_archive_sha or response_archive_sha or None
    archive_bytes = (
        _int_or_none(export.get("archive_bytes"))
        or _int_or_none(mlx_response.get("archive_size_bytes"))
        or _int_or_none(mlx_response.get("archive_bytes"))
    )
    response_archive_bytes = _int_or_none(mlx_response.get("archive_size_bytes"))
    candidate_pairs = (
        _int_or_none(_path_get(export, ("modelsize_candidate", "num_pairs")))
        or _int_or_none(export.get("num_pairs"))
        or _int_or_none(mlx_response.get("max_pairs"))
        or _int_or_none(mlx_response.get("n_samples"))
        or CONTEST_PAIR_COUNT
    )
    measured_pairs = _evidence_pair_count(mlx_response)
    avg_seg = _float_or_none(mlx_response.get("avg_segnet_dist"))
    avg_pose = _float_or_none(mlx_response.get("avg_posenet_dist"))
    score = _float_or_none(
        mlx_response.get("score_recomputed_from_components")
        or mlx_response.get("canonical_score")
    )
    rate_term = _float_or_none(mlx_response.get("score_rate_contribution"))
    seg_term = None if avg_seg is None else 100.0 * float(avg_seg)
    pose_term = (
        None
        if avg_pose is None or avg_pose < 0.0
        else math.sqrt(10.0 * float(avg_pose))
    )
    nonrate_score = (
        None
        if seg_term is None or pose_term is None
        else float(seg_term) + float(pose_term)
    )
    if rate_term is None and score is not None and nonrate_score is not None:
        rate_term = float(score) - float(nonrate_score)
    ceiling = (
        _int_or_none(hard_byte_ceiling)
        or _int_or_none(_path_get(export, ("modelsize_candidate", "hard_byte_ceiling")))
        or _min_positive_int(export.get("hard_byte_ceilings"))
    )
    observed_seg_weight, observed_seg_weight_source = (
        _infer_full_video_feedback_segnet_distillation_weight(
            export=export,
            explicit_weight=current_segnet_distillation_weight,
        )
    )
    observed_pose_weight, observed_pose_weight_source = (
        _infer_full_video_feedback_pose_distillation_weight(
            export=export,
            explicit_weight=None,
        )
    )
    full_video = bool(int(measured_pairs) >= max(int(candidate_pairs), CONTEST_PAIR_COUNT))
    response_false_authority_blockers = _mlx_response_authority_blockers(mlx_response)
    direct_blockers: list[str] = []
    if not family_key:
        direct_blockers.append("full_video_mlx_response_family_missing")
    if not resolved_candidate_id:
        direct_blockers.append("full_video_mlx_response_candidate_id_missing")
    if not response_archive_sha:
        direct_blockers.append("full_video_mlx_response_archive_sha256_missing")
    if export and not export_archive_sha:
        direct_blockers.append("archive_export_report_archive_sha256_missing")
    if response_archive_sha and export_archive_sha and response_archive_sha != export_archive_sha:
        direct_blockers.append("full_video_mlx_response_archive_sha256_mismatch")
    if archive_bytes is None:
        direct_blockers.append("full_video_mlx_response_archive_bytes_missing")
    if response_archive_bytes is not None and archive_bytes is not None and response_archive_bytes != archive_bytes:
        direct_blockers.append("full_video_mlx_response_archive_bytes_mismatch")
    if not full_video:
        direct_blockers.append("full_video_mlx_response_not_full600")
    direct_blockers.extend(response_false_authority_blockers)
    archive_under_ceiling = (
        None if ceiling is None or archive_bytes is None else int(archive_bytes) <= int(ceiling)
    )
    if archive_under_ceiling is False:
        direct_blockers.append(f"{family_key or 'nerv'}_full_video_mlx_response_archive_over_hard_byte_ceiling")
    scorer_control = _full_video_mlx_response_training_control(
        family=family_key,
        score=score,
        nonrate_score=nonrate_score,
        avg_seg=avg_seg,
        avg_pose=avg_pose,
        archive_under_ceiling=archive_under_ceiling,
        full_video=full_video,
        current_segnet_distillation_weight=observed_seg_weight,
        current_pose_distillation_weight=observed_pose_weight,
        max_mlx_score_for_local_replay=max_mlx_score_for_local_replay,
    )
    recommended_mutations = list(
        scorer_control.get("recommended_launch_mutations") or []
    )
    receiver_proof_attached = _archive_export_receiver_proof_attached(export)
    sample_blockers = [] if full_video else ["full600_mlx_scorer_response_required"]
    local_replay_passed = bool(
        score is not None
        and max_mlx_score_for_local_replay is not None
        and float(score) <= float(max_mlx_score_for_local_replay)
    )
    response_sha = _sha256_file(response_path) if response_path else None
    export_sha = _sha256_file(export_path) if export_path else None
    return {
        "schema": SCHEMA,
        "feedback_kind": "full_video_mlx_scorer_response",
        "full_video_mlx_feedback_schema": FULL_VIDEO_MLX_SCORER_FEEDBACK_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_report_path": response_path.as_posix() if response_path else None,
        "source_report_sha256": response_sha,
        "archive_export_report_path": export_path.as_posix() if export_path else None,
        "archive_export_report_sha256": export_sha,
        "mode": "full_video_mlx_scorer_response_harvested",
        "family": family_key or None,
        "candidate_id": resolved_candidate_id or None,
        "candidate_conditioned": bool(resolved_candidate_id),
        "candidate_num_pairs": int(candidate_pairs),
        "measured_num_pairs": int(measured_pairs),
        "feedback_scope": (
            "full600_mlx_scorer_response" if full_video else "partial_mlx_scorer_response"
        ),
        "scope_matches_candidate": bool(full_video and resolved_candidate_id),
        "feedback_ready": bool(
            full_video and resolved_candidate_id and not direct_blockers
        ),
        "launch_control_feedback_ready": bool(full_video and resolved_candidate_id),
        "hard_byte_ceiling": ceiling,
        "nominal_total_payload_bytes": _path_get(
            export, ("modelsize_candidate", "nominal_total_payload_bytes")
        ),
        "measured_payload_bytes": _int_or_none(export.get("packet_bytes")),
        "measured_archive_bytes": archive_bytes,
        "measured_minus_nominal_bytes": _minus_or_none(
            archive_bytes,
            _path_get(export, ("modelsize_candidate", "nominal_total_payload_bytes")),
        ),
        "archive_path": export.get("archive_path"),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "receiver_proof_attached": receiver_proof_attached,
        "receiver_proof_path": export.get("receiver_proof_path"),
        "receiver_proof_sha256": export.get("receiver_proof_sha256"),
        "receiver_contract_satisfied": receiver_proof_attached,
        "full_video_local_prefilter_attached": bool(full_video and response_sha),
        "full_video_mlx_response_attached": bool(full_video and response_sha),
        "full_video_mlx_response_path": response_path.as_posix() if response_path else None,
        "full_video_mlx_response_sha256": response_sha,
        "local_cpu_replay_gate_attached": False,
        "local_cpu_replay_gate_has_full_video_mlx_prefilter": bool(full_video),
        "local_cpu_replay_gate_local_replay_mlx_prefilter_passed": local_replay_passed,
        "local_cpu_replay_gate_coverage_valid_for_replay": bool(full_video),
        "local_cpu_replay_gate_executed": False,
        "mlx_prefilter_profile_count": 1 if full_video else 0,
        "mlx_prefilter_has_full_video": bool(full_video),
        "mlx_prefilter_local_replay_passed": local_replay_passed,
        "mlx_prefilter_best_full_video_mlx_score": score,
        "mlx_prefilter_full_video_profile_paths": (
            [response_path.as_posix()] if response_path else []
        ),
        "mlx_prefilter_local_replay_profile_paths": [],
        "mlx_prefilter_blockers": (
            []
            if local_replay_passed
            else ["local_cpu_replay_blocked_by_mlx_prefilter_score"]
        ),
        "full_video_mlx_scorer_response": {
            "schema": FULL_VIDEO_MLX_SCORER_FEEDBACK_SCHEMA,
            "source_schema": response_schema,
            "score_axis": mlx_response.get("score_axis")
            or mlx_response.get("evidence_tag"),
            "hardware_substrate": mlx_response.get("hardware_substrate"),
            "evidence_grade": mlx_response.get("evidence_grade"),
            "candidate_generation_only": bool(
                mlx_response.get("candidate_generation_only")
            ),
            "n_samples": int(measured_pairs),
            "score_recomputed_from_components": score,
            "avg_segnet_dist": avg_seg,
            "avg_posenet_dist": avg_pose,
            "score_rate_contribution": rate_term,
            "seg_score_term": seg_term,
            "pose_score_term": pose_term,
            "nonrate_score_estimate": nonrate_score,
            "archive_size_bytes": archive_bytes,
            "archive_under_hard_byte_ceiling": archive_under_ceiling,
            "response_archive_sha256": response_archive_sha or None,
            "export_archive_sha256": export_archive_sha or None,
            "inflated_outputs_aggregate_sha256": mlx_response.get(
                "inflated_outputs_aggregate_sha256"
            ),
            "cache_identity": mlx_response.get("cache_identity"),
            "components": mlx_response.get("components"),
            "direct_blockers": list(direct_blockers),
            **FALSE_AUTHORITY,
        },
        "sample_generalization_gate": {
            "schema": SAMPLE_GENERALIZATION_GATE_SCHEMA,
            "candidate_num_pairs": int(candidate_pairs),
            "measured_num_pairs": int(measured_pairs),
            "required_pairs": int(max(CONTEST_PAIR_COUNT, int(candidate_pairs))),
            "full_video_mlx_prefilter": bool(full_video),
            "representative_distortion_evidence": bool(full_video),
            "small_pair_smoke_only": False,
            "verdict": (
                "representative_full_video_mlx_scorer_response_present"
                if full_video
                else "partial_mlx_scorer_response_requires_full600"
            ),
            "blockers": sample_blockers,
            **FALSE_AUTHORITY,
        },
        "sample_generalization_blockers": sample_blockers,
        "full_video_mlx_response_control": scorer_control,
        "training_control": scorer_control,
        "training_control_action": scorer_control["action"],
        "training_control_reason": scorer_control["reason"],
        "training_control_should_stop_current_run": scorer_control[
            "should_stop_current_run"
        ],
        "training_control_successor_required": scorer_control[
            "successor_required"
        ],
        "pose_instability_detected": bool(
            scorer_control.get("pose_fit_failure_detected")
        ),
        "pose_tail_burst_detected": bool(
            scorer_control.get("pose_fit_failure_detected")
        ),
        "seg_stagnation_detected": bool(
            scorer_control.get("segnet_fit_failure_detected")
        ),
        "observed_learning_rate": None,
        "recommended_learning_rate": None,
        "observed_segnet_distillation_weight": observed_seg_weight,
        "segnet_distillation_weight_source": observed_seg_weight_source,
        "observed_pose_distillation_weight": observed_pose_weight,
        "pose_distillation_weight_source": observed_pose_weight_source,
        "recommended_segnet_distillation_weight": scorer_control.get(
            "recommended_segnet_distillation_weight"
        ),
        "recommended_segnet_distillation_weight_multiplier": scorer_control.get(
            "recommended_segnet_distillation_weight_multiplier"
        ),
        "recommended_pose_distillation_weight": scorer_control.get(
            "recommended_pose_distillation_weight"
        ),
        "recommended_pose_distillation_weight_multiplier": scorer_control.get(
            "recommended_pose_distillation_weight_multiplier"
        ),
        "recommended_launch_mutations": recommended_mutations,
        "direct_feedback_blockers": _dedupe_strings(direct_blockers),
        "blockers": _dedupe_strings([*direct_blockers, *sample_blockers]),
        **FALSE_AUTHORITY,
    }


def write_nerv_full_video_mlx_scorer_feedback_files(
    *,
    mlx_response: Mapping[str, Any],
    output_dir: str | Path,
    archive_export_report: Mapping[str, Any] | None = None,
    mlx_response_path: str | Path | None = None,
    archive_export_report_path: str | Path | None = None,
    candidate_id: str | None = None,
    family: str | None = None,
    hard_byte_ceiling: int | None = None,
    current_segnet_distillation_weight: float | None = None,
    max_mlx_score_for_local_replay: float | None = (
        DEFAULT_MAX_MLX_SCORE_FOR_LOCAL_REPLAY
    ),
) -> dict[str, Any]:
    """Write a full-video MLX response feedback row plus append-only ledger."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    export_report = _attach_startup_json_to_archive_export_report(
        archive_export_report
    )
    row = build_nerv_full_video_mlx_scorer_feedback_row(
        mlx_response=mlx_response,
        archive_export_report=export_report,
        mlx_response_path=mlx_response_path,
        archive_export_report_path=archive_export_report_path,
        candidate_id=candidate_id,
        family=family,
        hard_byte_ceiling=hard_byte_ceiling,
        current_segnet_distillation_weight=current_segnet_distillation_weight,
        max_mlx_score_for_local_replay=max_mlx_score_for_local_replay,
    )
    row_path = out / "nerv_full_video_mlx_scorer_feedback_row.json"
    ledger_path = out / "nerv_full_video_mlx_scorer_feedback.jsonl"
    row_path.write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema": FULL_VIDEO_MLX_SCORER_FEEDBACK_SCHEMA,
        "row": row,
        "row_path": row_path.as_posix(),
        "ledger_path": ledger_path.as_posix(),
        "append_only": True,
        **FALSE_AUTHORITY,
    }
    manifest_path = out / "nerv_full_video_mlx_scorer_feedback.json"
    manifest.update({"manifest_path": manifest_path.as_posix()})
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _attach_startup_json_to_archive_export_report(
    archive_export_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    export = dict(archive_export_report or {})
    if "runner_startup_json" in export or "startup_json" in export:
        return export
    startup_path_raw = (
        export.get("startup_json_path")
        or export.get("runner_startup_json_path")
        or export.get("compact_renderer_mlx_spine_runner_startup_json_path")
    )
    if not startup_path_raw:
        return export
    startup_path = Path(str(startup_path_raw)).expanduser()
    try:
        startup_payload = json.loads(startup_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        export["startup_json_load_blocker"] = (
            "archive_export_startup_json_path_unreadable"
        )
        return export
    export["runner_startup_json"] = startup_payload
    export["runner_startup_json_path"] = startup_path.as_posix()
    startup_sha = _sha256_file(startup_path)
    if startup_sha is not None:
        export["runner_startup_json_sha256"] = startup_sha
    return export


def _hinerv_archive_ladder_feedback_row(
    row: Mapping[str, Any],
    *,
    num_pairs: int,
    source_report_path: str | None,
    source_report_sha256: str | None,
) -> dict[str, Any]:
    candidate_id = str(row.get("row_id") or row.get("candidate_id") or "").strip()
    archive_path = str(row.get("archive_path") or "")
    receiver_proof_path = str(row.get("receiver_proof_path") or "")
    archive_bytes = _int_or_none(row.get("archive_bytes"))
    receiver_ready = bool(row.get("runtime_consumption_proof_ready") is True)
    proof_path_obj = (
        Path(receiver_proof_path).expanduser().resolve(strict=False)
        if receiver_proof_path
        else None
    )
    receiver_proof_sha = _sha256_file(proof_path_obj) if proof_path_obj else None
    direct_blockers: list[str] = []
    if not candidate_id:
        direct_blockers.append("hinerv_archive_ladder_feedback_candidate_id_missing")
    if archive_bytes is None:
        direct_blockers.append("hinerv_archive_ladder_feedback_archive_bytes_missing")
    if not archive_path:
        direct_blockers.append("hinerv_archive_ladder_feedback_archive_path_missing")
    if not row.get("archive_sha256"):
        direct_blockers.append("hinerv_archive_ladder_feedback_archive_sha256_missing")
    if not receiver_ready:
        direct_blockers.append("hinerv_archive_ladder_feedback_receiver_proof_not_ready")
    if receiver_ready and not receiver_proof_sha:
        direct_blockers.append("hinerv_archive_ladder_feedback_receiver_proof_file_missing")
    sample_blockers = [
        "representative_distortion_evidence_missing",
        "hinerv_archive_ladder_nonrate_score_missing",
        "full600_mlx_scorer_prefilter_required_before_cpu_replay",
    ]
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_report_path": source_report_path,
        "source_report_sha256": source_report_sha256,
        "mode": "hinerv_archive_ladder_feedback",
        "family": "hi_nerv",
        "candidate_id": candidate_id,
        "candidate_conditioned": bool(candidate_id),
        "candidate_num_pairs": int(num_pairs),
        "measured_num_pairs": int(num_pairs),
        "feedback_scope": "full600_receiver_closed_archive_ladder_rate_only",
        "scope_matches_candidate": bool(candidate_id),
        "feedback_ready": bool(receiver_ready and archive_bytes is not None),
        "byte_feedback_source": "hinerv_archive_size_ladder",
        "hard_byte_ceiling": _path_get(row, ("modelsize_candidate", "hard_byte_ceiling")),
        "nominal_total_payload_bytes": row.get("nominal_total_payload_bytes"),
        "measured_payload_bytes": None,
        "measured_archive_bytes": archive_bytes,
        "measured_minus_nominal_bytes": row.get("measured_minus_nominal_bytes"),
        "archive_path": archive_path or None,
        "archive_bytes": archive_bytes,
        "archive_sha256": row.get("archive_sha256"),
        "receiver_proof_attached": bool(receiver_ready and receiver_proof_sha),
        "receiver_proof_path": receiver_proof_path or None,
        "receiver_proof_sha256": receiver_proof_sha,
        "receiver_contract_satisfied": bool(receiver_ready),
        "runtime_consumption_proof_ready": receiver_ready,
        "full_video_local_prefilter_attached": False,
        "local_cpu_replay_gate_attached": False,
        "sample_generalization_blockers": sample_blockers,
        "sample_generalization_gate": {
            "schema": SAMPLE_GENERALIZATION_GATE_SCHEMA,
            "candidate_num_pairs": int(num_pairs),
            "measured_num_pairs": int(num_pairs),
            "required_pairs": int(max(CONTEST_PAIR_COUNT, num_pairs)),
            "byte_feedback_full_scope": True,
            "representative_distortion_evidence": False,
            "small_pair_smoke_only": False,
            "verdict": "rate_only_receiver_closed_ladder_requires_scorer_prefilter",
            "blockers": sample_blockers,
            **FALSE_AUTHORITY,
        },
        "direct_feedback_blockers": _dedupe_strings(direct_blockers),
        "source_archive_ladder_row_blockers": list(row.get("blockers") or ()),
        **FALSE_AUTHORITY,
    }


def _sample_generalization_gate(
    *,
    runner_report: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    selection: Mapping[str, Any],
    byte_feedback: Mapping[str, Any],
    candidate_num_pairs: Any,
    measured_num_pairs: Any,
    mlx_prefilter: Mapping[str, Any],
    local_replay: Any,
) -> dict[str, Any]:
    """Classify whether distortion evidence is representative or smoke-only."""

    candidate_pairs = (
        _int_or_none(candidate_num_pairs)
        or _int_or_none(byte_feedback.get("candidate_num_pairs"))
        or _int_or_none(runner_report.get("num_pairs"))
        or CONTEST_PAIR_COUNT
    )
    measured_pairs = (
        _int_or_none(measured_num_pairs)
        or _int_or_none(byte_feedback.get("measured_num_pairs"))
        or _int_or_none(runner_report.get("measured_num_pairs"))
        or 0
    )
    required_pairs = max(CONTEST_PAIR_COUNT, int(candidate_pairs))
    full_video_mlx_prefilter = bool(mlx_prefilter.get("has_full_video_mlx_prefilter"))
    local_replay_pairs = _evidence_pair_count(local_replay)
    local_replay_full_video = bool(local_replay_pairs >= required_pairs)
    byte_feedback_full_scope = bool(
        measured_pairs >= required_pairs
        and (
            byte_feedback.get("scope_matches_candidate") is True
            or str(byte_feedback.get("feedback_scope") or "").startswith("full600")
        )
    )
    hard_pair_coverage = _hard_pair_coverage_evidence(
        runner_report=runner_report,
        curriculum=curriculum,
        selection=selection,
    )
    hard_pair_distortion_coverage = bool(
        hard_pair_coverage.get("representative_distortion_evidence")
    )
    representative_distortion = bool(
        full_video_mlx_prefilter
        or local_replay_full_video
        or hard_pair_distortion_coverage
    )
    small_pair_smoke_only = bool(
        measured_pairs > 0
        and measured_pairs < required_pairs
        and not representative_distortion
    )
    chunked_micro_rows = _chunked_micro_row_evidence(
        runner_report=runner_report,
        curriculum=curriculum,
        selection=selection,
    )
    chunked_micro_rows_profile_only = bool(
        chunked_micro_rows.get("present")
        and chunked_micro_rows.get("receiver_closed_single_archive") is not True
    )
    blockers: list[str] = []
    recommended_next_actions: list[str] = []
    if small_pair_smoke_only:
        blockers.extend(
            [
                "small_pair_distortion_smoke_only_not_representative",
                "full600_or_hardpair_distortion_replay_required",
            ]
        )
        recommended_next_actions.extend(
            [
                "run_full600_mlx_scorer_prefilter_before_reading_distortion",
                "run_xray_hardpair_hitlist_replay_for_segnet_frame1_and_posenet_pair_axes",
                "keep_partial_pair_rows_as_rate_or_smoke_signal_only",
            ]
        )
        verdict = "small_pair_smoke_only_requires_full600_or_hardpair_distortion_gate"
    elif representative_distortion:
        verdict = "representative_distortion_evidence_present"
    else:
        blockers.append("representative_distortion_evidence_missing")
        recommended_next_actions.extend(
            [
                "attach_full600_mlx_scorer_prefilter",
                "attach_local_cpu_replay_gate_after_full_video_prefilter_passes",
            ]
        )
        verdict = "representative_distortion_evidence_missing"
    if chunked_micro_rows_profile_only:
        blockers.append("four_pair_chunk_rows_profile_only_no_rate_arbitrage")
        recommended_next_actions.extend(
            [
                "use_four_pair_rows_for_hard_pair_curriculum_and_section_value_profiling",
                "merge_chunks_only_when_receiver_closed_single_archive_bytes_are_measured",
                "price_chunked_or_moe_decoders_by_total_archive_zip_bytes_not_per_chunk_smoke_bytes",
            ]
        )
    blockers.extend(str(blocker) for blocker in hard_pair_coverage.get("blockers") or [])
    return {
        "schema": SAMPLE_GENERALIZATION_GATE_SCHEMA,
        "candidate_num_pairs": int(candidate_pairs),
        "measured_num_pairs": int(measured_pairs),
        "required_pairs": int(required_pairs),
        "full_video_mlx_prefilter": full_video_mlx_prefilter,
        "local_replay_full_video": local_replay_full_video,
        "local_replay_num_pairs": int(local_replay_pairs) if local_replay_pairs else None,
        "byte_feedback_full_scope": byte_feedback_full_scope,
        "hard_pair_distortion_coverage": hard_pair_distortion_coverage,
        "hard_pair_coverage": hard_pair_coverage or None,
        "chunked_micro_row_evidence": chunked_micro_rows or None,
        "chunked_micro_rows_profile_only": chunked_micro_rows_profile_only,
        "representative_distortion_evidence": representative_distortion,
        "small_pair_smoke_only": small_pair_smoke_only,
        "verdict": verdict,
        "why_small_pair_can_look_good": [
            "easy_or_contiguous_pair_subset_can_miss_full_video_hard_pair_tail",
            "segnet_scores_only_last_frame_so_frame1_boundary_coverage_matters",
            "posenet_scores_both_frames_so_pair_geometry_tail_can_dominate",
            "partial_pair_fit_can_overfit_local_latents_without_decoder_generalization",
            "local_or_mlx_advisory_evidence_is_false_authority_until_receiver_and_replay_close",
        ],
        "why_four_pair_rows_do_not_trick_the_scorer": [
            "official_eval_scores_one_inflated_full_video_from_one_archive",
            "rate_term_is_total_archive_zip_bytes_not_per_chunk_best_row_bytes",
            "multiple_chunk_decoders_pay_combined_weights_sidecars_and_zip_overhead",
            "chunking_is_useful_only_if_it_learns_shared_receiver_closed_grammar_or_hard_pair_curriculum",
        ],
        "contest_score_geometry": {
            "schema": "nerv_contest_score_geometry.v1",
            "score_lagrangian": "100*d_seg + sqrt(10*d_pose) + 25*archive_zip_bytes/uncompressed_total_bytes",
            "segnet_domain": "last_frame_of_each_pair_only_at_scorer_resize",
            "posenet_domain": "both_frames_of_each_pair_at_scorer_resize",
            "rate_domain": "single_receiver_archive_zip_bytes",
            "optimization_basis": (
                "protect parameters_or_payload_sections whose measured marginal "
                "nonrate score drop exceeds fixed archive byte price"
            ),
            **FALSE_AUTHORITY,
        },
        "pr95_distortion_controls_to_bind": [
            "full600_training_not_tiny_pair_selection",
            "score_axis_distillation_on_segnet_frame1_and_posenet_pair",
            "long_pr95_curriculum_through_late_qat_coder_pressure_and_final_muon",
            "ema_archive_selection_and_parseback_before_score_claim",
            "coder_aware_regularization_kept_rate_pressure_in_loop_while_fitting_distortion",
        ],
        "blockers": _dedupe_strings(blockers),
        "recommended_next_actions": _dedupe_strings(recommended_next_actions),
        **FALSE_AUTHORITY,
    }


def _evidence_pair_count(source: Any) -> int:
    if not isinstance(source, Mapping):
        return 0
    candidates: list[int] = []
    for key in (
        "n_samples",
        "num_pairs",
        "max_pairs",
        "measured_num_pairs",
    ):
        value = _int_or_none(source.get(key))
        if value is not None:
            candidates.append(value)
    for nested_key in ("score_components", "scope_status", "metadata"):
        nested = source.get(nested_key)
        if isinstance(nested, Mapping):
            nested_count = _evidence_pair_count(nested)
            if nested_count:
                candidates.append(nested_count)
    return max(candidates) if candidates else 0


def _hard_pair_coverage_evidence(
    *,
    runner_report: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    for source_key in (
        "hard_pair_coverage",
        "xray_hardpair_coverage",
        "hard_pair_replay_coverage",
        "sample_generalization_gate",
    ):
        for container in (runner_report, curriculum, selection):
            value = container.get(source_key)
            if isinstance(value, Mapping):
                representative = any(
                    value.get(key) is True
                    for key in (
                        "representative_distortion_evidence",
                        "coverage_valid_for_distortion",
                        "distortion_representative",
                        "score_axis_hard_pair_coverage",
                    )
                )
                pair_indices, parse_blockers, parse_error = _hard_pair_indices_payload(
                    value
                )
                payload = {
                    "schema": "nerv_hard_pair_coverage_evidence.v1",
                    "source_key": source_key,
                    "source_schema": value.get("schema"),
                    "representative_distortion_evidence": representative,
                    "hard_pair_count": _int_or_none(value.get("hard_pair_count")),
                    "prioritized_pair_indices": pair_indices,
                    "coverage_verdict": value.get("verdict")
                    or value.get("coverage_verdict"),
                    "blockers": parse_blockers,
                    **FALSE_AUTHORITY,
                }
                if parse_error is not None:
                    payload["hard_pair_indices_parse_error"] = parse_error
                return payload
    return {}


def _hard_pair_indices_payload(
    value: Mapping[str, Any],
) -> tuple[list[int], list[str], str | None]:
    try:
        return list(pair_indices_from_mapping(value)), [], None
    except HardPairIndicesError as exc:
        return [], ["hard_pair_indices_parse_failed"], str(exc)


def _chunked_micro_row_evidence(
    *,
    runner_report: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    for source_key in (
        "four_pair_byte_rows",
        "chunked_pair_byte_rows",
        "pair_chunk_feedback_rows",
        "micro_pair_rows",
    ):
        for container in (runner_report, curriculum, selection):
            value = container.get(source_key)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                continue
            rows = [row for row in value if isinstance(row, Mapping)]
            if not rows:
                continue
            counts = [_chunk_row_pair_count(row) for row in rows]
            measured_pair_sum = sum(count for count in counts if count > 0)
            archive_bytes = [
                _int_or_none(row.get("measured_archive_bytes") or row.get("archive_bytes"))
                for row in rows
            ]
            receiver_closed_single_archive = any(
                row.get("receiver_closed_single_archive") is True
                or row.get("single_archive_receiver_closed") is True
                for row in rows
            )
            return {
                "schema": "nerv_chunked_micro_row_evidence.v1",
                "source_key": source_key,
                "present": True,
                "row_count": len(rows),
                "max_row_pairs": max(counts) if counts else None,
                "measured_pair_sum": int(measured_pair_sum),
                "archive_byte_sum": sum(
                    int(value) for value in archive_bytes if value is not None
                )
                if any(value is not None for value in archive_bytes)
                else None,
                "receiver_closed_single_archive": receiver_closed_single_archive,
                "scorer_contract": (
                    "official_scorer_charges_one_full_archive_not_independent_micro_rows"
                ),
                **FALSE_AUTHORITY,
            }
    return {}


def _chunk_row_pair_count(row: Mapping[str, Any]) -> int:
    for key in ("measured_num_pairs", "n_samples", "num_pairs", "max_pairs"):
        value = _int_or_none(row.get(key))
        if value is not None:
            return max(0, int(value))
    return 0


def _snerv_native_file_backed_byte_feedback(
    *,
    family: Any,
    candidate_row: Mapping[str, Any],
    byte_feedback: Mapping[str, Any],
    native_export: Mapping[str, Any],
    native_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Promote strict SNeRV file-backed export bytes into planner feedback.

    This is byte feedback only: it deliberately leaves score authority false.
    The evidence object is produced by the native receiver/archive validator, so
    require its full-pair proof before letting SNAR bytes unblock byte pricing.
    """

    if _family_key(str(family)) != "snerv":
        return {}
    if not native_export or not native_evidence:
        return {}
    training_guard = build_snerv_mlx_native_training_export_guard(native_export)
    if training_guard.get("export_guard_passed") is not True:
        return {}
    if native_evidence.get("required_pair_file_backed_export_proof_passed") is not True:
        return {}
    packet_bytes = _int_or_none(native_export.get("packet_bytes"))
    archive_bytes = _int_or_none(native_export.get("archive_bytes"))
    if packet_bytes is None or archive_bytes is None:
        return {}
    if packet_bytes <= 0 or archive_bytes <= 0:
        return {}
    native_pairs = (
        _int_or_none(native_export.get("num_pairs"))
        or _int_or_none(native_evidence.get("num_pairs"))
        or _int_or_none(byte_feedback.get("measured_num_pairs"))
    )
    candidate_pairs = (
        _int_or_none(byte_feedback.get("candidate_num_pairs"))
        or _int_or_none(candidate_row.get("num_pairs"))
        or CONTEST_PAIR_COUNT
    )
    if native_pairs is None or native_pairs < max(candidate_pairs, CONTEST_PAIR_COUNT):
        return {}
    candidate_id = str(candidate_row.get("candidate_id") or "").strip()
    if not candidate_id:
        return {}
    native_candidate_id = str(
        native_export.get("candidate_id")
        or native_export.get("modelsize_candidate_id")
        or ""
    ).strip()
    if not native_candidate_id or native_candidate_id != candidate_id:
        return {}
    packet_path = Path(str(native_export.get("packet_path") or "")).expanduser()
    archive_path = Path(str(native_export.get("archive_path") or "")).expanduser()
    packet_sha = str(native_export.get("packet_sha256") or "").strip()
    archive_sha = str(native_export.get("archive_sha256") or "").strip()
    if (
        not _file_matches_declared_bytes_and_sha(
            packet_path,
            declared_bytes=packet_bytes,
            declared_sha256=packet_sha,
        )
        or not _file_matches_declared_bytes_and_sha(
            archive_path,
            declared_bytes=archive_bytes,
            declared_sha256=archive_sha,
        )
    ):
        return {}
    nominal_payload = _int_or_none(byte_feedback.get("nominal_total_payload_bytes"))
    measured_minus_nominal = (
        None if nominal_payload is None else int(packet_bytes) - int(nominal_payload)
    )
    return {
        "schema": "snerv_native_file_backed_byte_feedback.v1",
        "byte_feedback_source": "snerv_mlx_native_file_backed_export",
        "candidate_id": candidate_id or None,
        "candidate_num_pairs": int(candidate_pairs),
        "measured_num_pairs": int(native_pairs),
        "feedback_scope": "full600_native_file_backed_snar1_export",
        "scope_matches_candidate": True,
        "feedback_ready": True,
        "measured_payload_bytes": int(packet_bytes),
        "measured_archive_bytes": int(archive_bytes),
        "measured_minus_nominal_bytes": measured_minus_nominal,
        "packet_path": packet_path.as_posix(),
        "packet_sha256": packet_sha,
        "archive_path": archive_path.as_posix(),
        "archive_sha256": archive_sha,
        **FALSE_AUTHORITY,
    }


def _file_matches_declared_bytes_and_sha(
    path: Path,
    *,
    declared_bytes: int,
    declared_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size != int(declared_bytes):
        return False
    return bool(declared_sha256 and _sha256_file(path) == declared_sha256)


def build_nerv_training_telemetry_feedback_row(
    *,
    telemetry_path: str | Path,
    family: str,
    candidate_id: str,
    candidate_num_pairs: int,
    source_queue_path: str | Path | None = None,
    stop_reason: str | None = None,
    current_segnet_distillation_weight: float | None = None,
    pose_loss_instability_threshold: float = _POSE_LOSS_INSTABILITY_THRESHOLD,
    pose_axis_instability_threshold: float = _POSE_AXIS_INSTABILITY_THRESHOLD,
    instability_window_epochs: int = _POSE_INSTABILITY_WINDOW_EPOCHS,
    instability_bad_fraction: float = _POSE_INSTABILITY_BAD_FRACTION,
    learning_rate_multiplier: float = _POSE_INSTABILITY_LR_MULTIPLIER,
) -> dict[str, Any]:
    """Build a false-authority candidate-feedback row from training telemetry."""

    telemetry = Path(telemetry_path).expanduser().resolve(strict=False)
    if not telemetry.is_file():
        raise FileNotFoundError(f"telemetry file not found: {telemetry}")
    source_queue = (
        Path(source_queue_path).expanduser().resolve(strict=False)
        if source_queue_path
        else None
    )
    rows = _read_telemetry_rows(telemetry)
    if not rows:
        raise ValueError(f"telemetry file has no JSON rows: {telemetry}")
    health = _summarize_training_telemetry_health(
        rows,
        pose_loss_instability_threshold=float(pose_loss_instability_threshold),
        pose_axis_instability_threshold=float(pose_axis_instability_threshold),
        instability_window_epochs=int(instability_window_epochs),
        instability_bad_fraction=float(instability_bad_fraction),
        learning_rate_multiplier=float(learning_rate_multiplier),
    )
    current_seg_weight = _float_or_none(current_segnet_distillation_weight)
    if current_seg_weight is not None and current_seg_weight > 0.0:
        health = {
            **health,
            "observed_segnet_distillation_weight": current_seg_weight,
            "recommended_segnet_distillation_weight": (
                recommend_segnet_distillation_weight_for_stagnation(
                    current_seg_weight
                )
                if bool(health.get("seg_stagnation_detected"))
                else None
            ),
            "segnet_distillation_weight_source": (
                "harvest_current_segnet_distillation_weight"
            ),
        }
    candidate_pairs = int(candidate_num_pairs)
    measured_pairs = _int_or_none(health.get("num_pairs")) or candidate_pairs
    family_key = _family_key(family)
    snerv_scorer_tether_health = (
        _snerv_scorer_tether_health(rows) if family_key == "snerv" else {}
    )
    nerv_train_time_control_health = (
        _nerv_train_time_control_health(rows, family_key=family_key)
        if family_key in {"hi_nerv", "snerv"}
        else {}
    )
    gradient_multiplier_control_health = _gradient_multiplier_control_health(
        rows,
        family_key=family_key,
    )
    recommended_launch_mutations = _training_telemetry_mutations_for_family(
        family_key,
        [
            *list(health.get("recommended_launch_mutations") or []),
            *list(snerv_scorer_tether_health.get("recommended_launch_mutations") or []),
            *list(
                nerv_train_time_control_health.get(
                    "recommended_launch_mutations"
                )
                or []
            ),
            *list(
                gradient_multiplier_control_health.get(
                    "recommended_launch_mutations"
                )
                or []
            ),
        ],
    )
    health_for_control = {
        **health,
        "degenerate_renderer_risk_detected": bool(
            snerv_scorer_tether_health.get("degenerate_renderer_risk_detected")
        ),
        "snerv_scorer_domain_tether_health": (
            snerv_scorer_tether_health if snerv_scorer_tether_health else None
        ),
        "hinerv_train_time_control_health": (
            nerv_train_time_control_health
            if family_key == "hi_nerv" and nerv_train_time_control_health
            else None
        ),
        "snerv_train_time_control_health": (
            nerv_train_time_control_health
            if family_key == "snerv" and nerv_train_time_control_health
            else None
        ),
        "nerv_train_time_control_health": (
            nerv_train_time_control_health
            if nerv_train_time_control_health
            else None
        ),
        "gradient_multiplier_control_health": (
            gradient_multiplier_control_health
            if gradient_multiplier_control_health
            else None
        ),
        "recommended_launch_mutations": recommended_launch_mutations,
    }
    direct_blockers: list[str] = []
    blockers: list[str] = [
        _training_telemetry_blocker(
            family_key,
            "trained_archive_byte_oracle_feedback_missing",
        ),
        _training_telemetry_blocker(
            family_key,
            "byte_closed_archive_export_missing",
        ),
        _training_telemetry_blocker(family_key, "receiver_proof_missing"),
        _training_telemetry_blocker(
            family_key,
            "full_video_local_prefilter_missing",
        ),
        _training_telemetry_blocker(family_key, "local_cpu_replay_gate_missing"),
    ]
    if health["pose_instability_detected"]:
        blockers.append(
            _training_telemetry_blocker(
                family_key,
                "pose_instability_telemetry_feedback",
            )
        )
    if health.get("pose_tail_burst_detected"):
        blockers.append(
            _training_telemetry_blocker(
                family_key,
                "pose_tail_burst_telemetry_feedback",
            )
        )
    if health.get("seg_stagnation_detected"):
        blockers.append(
            _training_telemetry_blocker(
                family_key,
                "segnet_stagnation_telemetry_feedback",
            )
        )
    if health.get("pr95_stage_mismatch_detected"):
        blockers.append(
            _training_telemetry_blocker(
                family_key,
                "pr95_stage_index_mismatch_telemetry",
            )
        )
    if health.get("pr95_final_stage_muon_missing"):
        blockers.append(
            _training_telemetry_blocker(
                family_key,
                "pr95_final_stage_muon_missing_telemetry",
            )
        )
    if bool(snerv_scorer_tether_health.get("degenerate_renderer_risk_detected")):
        scorer_tether_blockers = [
            str(blocker)
            for blocker in snerv_scorer_tether_health.get("blockers") or []
            if blocker
        ]
        blockers.extend(scorer_tether_blockers)
        direct_blockers.extend(scorer_tether_blockers)
    if bool(nerv_train_time_control_health.get("control_inert_risk_detected")):
        control_blockers = [
            str(blocker)
            for blocker in nerv_train_time_control_health.get("blockers") or []
            if blocker
        ]
        blockers.extend(control_blockers)
        direct_blockers.extend(control_blockers)
    if bool(gradient_multiplier_control_health.get("control_inert_risk_detected")):
        gradient_blockers = [
            str(blocker)
            for blocker in gradient_multiplier_control_health.get("blockers") or []
            if blocker
        ]
        blockers.extend(gradient_blockers)
        direct_blockers.extend(gradient_blockers)
    training_stopped = not _is_midrun_feedback_snapshot(stop_reason)
    training_control = _training_control_recommendation(
        health=health_for_control,
        training_stopped=training_stopped,
        measured_pairs=measured_pairs,
        candidate_pairs=candidate_pairs,
    )
    return {
        "schema": SCHEMA,
        "feedback_kind": "training_telemetry",
        "telemetry_feedback_schema": TELEMETRY_FEEDBACK_SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_report_path": telemetry.as_posix(),
        "source_report_sha256": _sha256_file(telemetry),
        "source_queue_path": source_queue.as_posix() if source_queue else None,
        "source_queue_sha256": _sha256_file(source_queue) if source_queue else None,
        "mode": "training_telemetry_harvested",
        "family": family_key,
        "candidate_id": str(candidate_id),
        "candidate_conditioned": True,
        "candidate_num_pairs": candidate_pairs,
        "measured_num_pairs": measured_pairs,
        "feedback_scope": "full600_training_telemetry"
        if measured_pairs >= CONTEST_PAIR_COUNT
        else "partial_training_telemetry",
        "scope_matches_candidate": measured_pairs >= candidate_pairs,
        "feedback_ready": False,
        "direct_feedback_blockers": _dedupe_strings(direct_blockers),
        "hard_byte_ceiling": None,
        "nominal_total_payload_bytes": None,
        "measured_payload_bytes": None,
        "measured_archive_bytes": None,
        "measured_minus_nominal_bytes": None,
        "archive_path": None,
        "archive_bytes": None,
        "archive_sha256": None,
        "training_completed": False,
        "training_stopped": training_stopped,
        "training_stop_reason": stop_reason
        or (
            "pose_instability_telemetry"
            if health["pose_instability_detected"]
            else "telemetry_harvest_without_completion_artifact"
        ),
        "training_telemetry": health_for_control,
        "training_control": training_control,
        "training_control_action": training_control["action"],
        "training_control_reason": training_control["reason"],
        "training_control_should_stop_current_run": training_control[
            "should_stop_current_run"
        ],
        "training_control_successor_required": training_control[
            "successor_required"
        ],
        "training_row_count": health.get("row_count"),
        "training_first_epoch": health.get("first_epoch"),
        "training_last_epoch": health.get("last_epoch"),
        "training_median_pose_axis": health.get("median_pose_axis"),
        "training_median_pose_distill_loss": health.get("median_pose_distill_loss"),
        "training_median_seg_axis": health.get("median_seg_axis"),
        "training_max_pose_axis": health.get("max_pose_axis"),
        "training_max_pose_distill_loss": health.get("max_pose_distill_loss"),
        "degenerate_renderer_risk_detected": bool(
            snerv_scorer_tether_health.get("degenerate_renderer_risk_detected")
        ),
        "snerv_scorer_domain_tether_health": (
            snerv_scorer_tether_health if snerv_scorer_tether_health else None
        ),
        "snerv_scorer_domain_tether_blockers": list(
            snerv_scorer_tether_health.get("blockers") or []
        ),
        "hinerv_train_time_control_health": (
            nerv_train_time_control_health
            if family_key == "hi_nerv" and nerv_train_time_control_health
            else None
        ),
        "snerv_train_time_control_health": (
            nerv_train_time_control_health
            if family_key == "snerv" and nerv_train_time_control_health
            else None
        ),
        "nerv_train_time_control_health": (
            nerv_train_time_control_health
            if nerv_train_time_control_health
            else None
        ),
        "hinerv_train_time_control_blockers": list(
            nerv_train_time_control_health.get("blockers") or []
            if family_key == "hi_nerv"
            else []
        ),
        "snerv_train_time_control_blockers": (
            list(nerv_train_time_control_health.get("blockers") or [])
            if family_key == "snerv"
            else []
        ),
        "gradient_multiplier_control_health": (
            gradient_multiplier_control_health
            if gradient_multiplier_control_health
            else None
        ),
        "gradient_multiplier_control_blockers": list(
            gradient_multiplier_control_health.get("blockers") or []
        ),
        "gradient_multiplier_control_inert_risk_detected": bool(
            gradient_multiplier_control_health.get("control_inert_risk_detected")
        ),
        "pose_instability_detected": bool(health["pose_instability_detected"]),
        "pose_instability_ever_detected": bool(
            health.get("pose_instability_ever_detected")
        ),
        "pose_instability_recovered": bool(
            health.get("pose_instability_recovered")
        ),
        "pose_instability_active_latest_window": bool(
            health.get("pose_instability_active_latest_window")
        ),
        "pose_instability_partial_window_detected": bool(
            health.get("pose_instability_partial_window_detected")
        ),
        "pose_instability_first_epoch": health.get("pose_instability_first_epoch"),
        "pose_instability_last_window_bad_fraction": health.get(
            "pose_instability_last_window_bad_fraction"
        ),
        "pose_tail_burst_detected": bool(health.get("pose_tail_burst_detected")),
        "pose_tail_burst_recent_window_epochs": health.get(
            "pose_tail_burst_recent_window_epochs"
        ),
        "pose_tail_burst_recent_bad_fraction": health.get(
            "pose_tail_burst_recent_bad_fraction"
        ),
        "pose_tail_burst_threshold": health.get("pose_tail_burst_threshold"),
        "pose_tail_burst_recent_p95": health.get("pose_tail_burst_recent_p95"),
        "pose_tail_burst_recent_max": health.get("pose_tail_burst_recent_max"),
        "pose_tail_burst_baseline_median": health.get(
            "pose_tail_burst_baseline_median"
        ),
        "observed_learning_rate": health.get("observed_learning_rate"),
        "recommended_learning_rate": health.get("recommended_learning_rate"),
        "recommended_learning_rate_multiplier": health.get(
            "recommended_learning_rate_multiplier"
        ),
        "seg_stagnation_detected": bool(health.get("seg_stagnation_detected")),
        "seg_stagnation_relative_improvement": health.get(
            "seg_stagnation_relative_improvement"
        ),
        "seg_stagnation_first_window_median": health.get(
            "seg_stagnation_first_window_median"
        ),
        "seg_stagnation_last_window_median": health.get(
            "seg_stagnation_last_window_median"
        ),
        "seg_recent_relative_improvement": health.get(
            "seg_recent_relative_improvement"
        ),
        "seg_recent_window_median": health.get("seg_recent_window_median"),
        "seg_previous_window_median": health.get("seg_previous_window_median"),
        "optimizer_control_observed": bool(
            health.get("optimizer_control_observed")
        ),
        "optimizer_stage_assessment": health.get("optimizer_stage_assessment"),
        "optimizer_muon_observed": bool(health.get("optimizer_muon_observed")),
        "pact_muon_adamw_observed": bool(
            health.get("pact_muon_adamw_observed")
        ),
        "pr95_curriculum_observed": bool(health.get("pr95_curriculum_observed")),
        "pr95_current_stage_index": health.get("pr95_current_stage_index"),
        "pr95_canonical_expected_stage_index": health.get(
            "pr95_canonical_expected_stage_index"
        ),
        "pr95_authoritative_stage_index": health.get(
            "pr95_authoritative_stage_index"
        ),
        "pr95_stage_mismatch_detected": bool(
            health.get("pr95_stage_mismatch_detected")
        ),
        "pr95_stage_uses_muon_current": health.get(
            "pr95_stage_uses_muon_current"
        ),
        "pr95_final_stage_reached": bool(health.get("pr95_final_stage_reached")),
        "pr95_final_stage_muon_expected_currently": bool(
            health.get("pr95_final_stage_muon_expected_currently")
        ),
        "pr95_final_stage_muon_missing": bool(
            health.get("pr95_final_stage_muon_missing")
        ),
        "pr95_stage_status": health.get("pr95_stage_status"),
        "observed_segnet_distillation_weight": health.get(
            "observed_segnet_distillation_weight"
        ),
        "segnet_distillation_weight_source": health.get(
            "segnet_distillation_weight_source"
        ),
        "recommended_segnet_distillation_weight": health.get(
            "recommended_segnet_distillation_weight"
        ),
        "recommended_segnet_distillation_weight_multiplier": health.get(
            "recommended_segnet_distillation_weight_multiplier"
        ),
        "recommended_launch_mutations": recommended_launch_mutations,
        "receiver_proof_report_paths": [],
        "local_cpu_replay_summary_present": False,
        "local_cpu_replay_score_estimate": None,
        "local_cpu_replay_gate_requested": None,
        "local_cpu_replay_gate_default_enabled_for_full_coverage": False,
        "local_cpu_replay_gate_has_full_video_mlx_prefilter": False,
        "local_cpu_replay_gate_local_replay_mlx_prefilter_passed": False,
        "local_cpu_replay_gate_coverage_valid_for_replay": False,
        "local_cpu_replay_gate_executed": False,
        "mlx_prefilter_profile_count": 0,
        "mlx_prefilter_has_full_video": False,
        "mlx_prefilter_local_replay_passed": False,
        "mlx_prefilter_best_full_video_mlx_score": None,
        "mlx_prefilter_full_video_profile_paths": [],
        "mlx_prefilter_local_replay_profile_paths": [],
        "mlx_prefilter_blockers": ["full_video_mlx_scorer_replay_not_attached"],
        "blockers": _dedupe_strings(blockers),
        **FALSE_AUTHORITY,
    }


def refresh_nerv_candidate_feedback_report(
    *,
    runner_report: Mapping[str, Any],
    repo_root: str | Path,
    mlx_profile_paths: tuple[str | Path, ...] = (),
    required_pairs: int = CONTEST_PAIR_COUNT,
    max_mlx_score_for_local_replay: float | None = (
        DEFAULT_MAX_MLX_SCORE_FOR_LOCAL_REPLAY
    ),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a feedback-safe runner report with refreshed MLX gate evidence.

    This is a backfill helper for reports produced before the batched-MLX
    acquisition/replay split existed. It does not rewrite the source report or
    grant authority; it only recomputes typed coverage from file-backed profile
    paths so candidate-feedback ledgers do not lose useful full-video GPU
    acquisition signal.
    """

    root = Path(repo_root).expanduser().resolve(strict=False)
    selected_profiles = _dedupe_strings(
        [
            *(str(path) for path in mlx_profile_paths),
            *_infer_mlx_profile_paths(runner_report),
        ]
    )
    refreshed = json.loads(json.dumps(dict(runner_report), sort_keys=True, default=str))
    old_blockers = list(refreshed.get("blockers") or [])
    coverage = summarize_mlx_prefilter_coverage(
        tuple(selected_profiles),
        root=root,
        required_pairs=int(required_pairs),
        max_mlx_score_for_local_replay=max_mlx_score_for_local_replay,
    )
    refreshed["mlx_profile_paths"] = [
        str(path) for path in coverage.get("full_video_profile_paths") or selected_profiles
    ]
    refreshed["mlx_prefilter_coverage"] = coverage
    has_full = bool(coverage.get("has_full_video_mlx_prefilter"))
    replay_passed = bool(coverage.get("local_replay_mlx_prefilter_passed"))
    num_pairs = _int_or_none(refreshed.get("num_pairs")) or 0
    coverage_valid_for_replay = int(num_pairs) >= int(required_pairs)
    gate = dict(refreshed.get("local_cpu_replay_gate") or {})
    gate.setdefault("schema", "compact_runner_local_cpu_replay_gate.v1")
    gate["has_full_video_mlx_prefilter"] = has_full
    gate["local_replay_mlx_prefilter_passed"] = replay_passed
    gate["coverage_valid_for_replay"] = coverage_valid_for_replay
    gate["default_enabled_for_full_coverage"] = bool(
        coverage_valid_for_replay and replay_passed
    )
    gate.setdefault("requested", None)
    gate.setdefault(
        "executed",
        isinstance(refreshed.get("local_cpu_replay_summary"), Mapping),
    )
    refreshed["local_cpu_replay_gate"] = gate

    blockers = list(old_blockers)
    removed_blockers: list[str] = []
    if has_full:
        kept: list[str] = []
        for blocker in blockers:
            if blocker in _MLX_PREFILTER_MISSING_BLOCKERS:
                removed_blockers.append(str(blocker))
            else:
                kept.append(str(blocker))
        blockers = kept
    blockers.extend(str(blocker) for blocker in coverage.get("blockers") or [])
    if has_full and coverage_valid_for_replay and not replay_passed:
        blockers.append(_LOCAL_REPLAY_BLOCKED_BY_MLX_SCORE)
    refreshed["blockers"] = _dedupe_strings(blockers)
    nested_removed = (
        _refresh_nested_pr95_stack_binding_blockers(refreshed) if has_full else []
    )

    refresh = {
        "schema": REFRESH_SCHEMA,
        "profile_paths": selected_profiles,
        "required_pairs": int(required_pairs),
        "max_mlx_score_for_local_replay": max_mlx_score_for_local_replay,
        "has_full_video_mlx_prefilter": has_full,
        "local_replay_mlx_prefilter_passed": replay_passed,
        "old_blockers": old_blockers,
        "removed_stale_blockers": removed_blockers,
        "removed_nested_pr95_stack_binding_blockers": nested_removed,
        "new_blockers": refreshed["blockers"],
        "mlx_prefilter_coverage": coverage,
        **FALSE_AUTHORITY,
    }
    return refreshed, refresh


def write_nerv_candidate_feedback_files(
    *,
    runner_report: Mapping[str, Any],
    output_dir: str | Path,
    source_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write JSON and JSONL feedback artifacts under ``output_dir``."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    row = build_nerv_candidate_feedback_row(
        runner_report=runner_report,
        source_report_path=source_report_path,
    )
    row_path = out / "nerv_candidate_byte_feedback_row.json"
    ledger_path = out / "nerv_candidate_byte_feedback.jsonl"
    row_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "schema": LEDGER_SCHEMA,
        "row": row,
        "row_path": row_path.as_posix(),
        "ledger_path": ledger_path.as_posix(),
        "append_only": True,
        **FALSE_AUTHORITY,
    }


def write_refreshed_nerv_candidate_feedback_files(
    *,
    runner_report: Mapping[str, Any],
    output_dir: str | Path,
    repo_root: str | Path,
    source_report_path: str | Path | None = None,
    mlx_profile_paths: tuple[str | Path, ...] = (),
    required_pairs: int = CONTEST_PAIR_COUNT,
    max_mlx_score_for_local_replay: float | None = (
        DEFAULT_MAX_MLX_SCORE_FOR_LOCAL_REPLAY
    ),
) -> dict[str, Any]:
    """Refresh MLX coverage and write a feedback row plus refresh manifest."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    refreshed, refresh = refresh_nerv_candidate_feedback_report(
        runner_report=runner_report,
        repo_root=repo_root,
        mlx_profile_paths=mlx_profile_paths,
        required_pairs=int(required_pairs),
        max_mlx_score_for_local_replay=max_mlx_score_for_local_replay,
    )
    refreshed_report_path = out / "refreshed_runner_report_for_feedback.json"
    refreshed_report_path.write_text(
        json.dumps(refreshed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    feedback = write_nerv_candidate_feedback_files(
        runner_report=refreshed,
        output_dir=out,
        source_report_path=source_report_path,
    )
    refresh_path = out / "nerv_candidate_feedback_refresh.json"
    refresh.update(
        {
            "refreshed_runner_report_path": refreshed_report_path.as_posix(),
            "candidate_feedback_row_path": feedback["row_path"],
            "candidate_feedback_ledger_path": feedback["ledger_path"],
        }
    )
    refresh_path.write_text(
        json.dumps(refresh, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema": REFRESH_SCHEMA,
        "refresh": refresh,
        "refresh_path": refresh_path.as_posix(),
        "refreshed_runner_report_path": refreshed_report_path.as_posix(),
        "candidate_feedback": feedback,
        **FALSE_AUTHORITY,
    }


def write_nerv_training_telemetry_feedback_files(
    *,
    telemetry_path: str | Path,
    output_dir: str | Path,
    family: str,
    candidate_id: str,
    candidate_num_pairs: int,
    source_queue_path: str | Path | None = None,
    stop_reason: str | None = None,
    current_segnet_distillation_weight: float | None = None,
) -> dict[str, Any]:
    """Write a telemetry feedback row plus append-only ledger."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    row = build_nerv_training_telemetry_feedback_row(
        telemetry_path=telemetry_path,
        family=family,
        candidate_id=candidate_id,
        candidate_num_pairs=int(candidate_num_pairs),
        source_queue_path=source_queue_path,
        stop_reason=stop_reason,
        current_segnet_distillation_weight=current_segnet_distillation_weight,
    )
    row_path = out / "nerv_candidate_training_telemetry_feedback_row.json"
    ledger_path = out / "nerv_candidate_training_telemetry_feedback.jsonl"
    row_path.write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema": TELEMETRY_FEEDBACK_SCHEMA,
        "row": row,
        "row_path": row_path.as_posix(),
        "ledger_path": ledger_path.as_posix(),
        "append_only": True,
        **FALSE_AUTHORITY,
    }
    manifest_path = out / "nerv_training_telemetry_feedback.json"
    manifest.update({"manifest_path": manifest_path.as_posix()})
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _infer_mlx_profile_paths(runner_report: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("mlx_profile_paths", "local_mlx_prefilter_profile_paths"):
        value = runner_report.get(key)
        if isinstance(value, (list, tuple)):
            paths.extend(str(path) for path in value if path)
    auto_path = runner_report.get("auto_mlx_prefilter_profile_path")
    if auto_path:
        paths.append(str(auto_path))
    coverage = runner_report.get("mlx_prefilter_coverage")
    if isinstance(coverage, Mapping):
        for key in ("full_video_profile_paths", "local_replay_profile_paths"):
            value = coverage.get(key)
            if isinstance(value, (list, tuple)):
                paths.extend(str(path) for path in value if path)
    return _dedupe_strings(paths)


def _read_telemetry_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON telemetry row") from exc
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def _summarize_training_telemetry_health(
    rows: Sequence[Mapping[str, Any]],
    *,
    pose_loss_instability_threshold: float,
    pose_axis_instability_threshold: float,
    instability_window_epochs: int,
    instability_bad_fraction: float,
    learning_rate_multiplier: float,
) -> dict[str, Any]:
    epochs: list[int] = []
    pose_losses: list[float] = []
    pose_axes: list[float] = []
    seg_axes: list[float] = []
    segnet_distillation_weights: list[float] = []
    learning_rates: list[float] = []
    pr95_stage_indices: list[int] = []
    pr95_stage_uses_muon_flags: list[bool] = []
    pact_muon_flags: list[bool] = []
    bad_epochs: list[int] = []
    window_size = max(1, int(instability_window_epochs))
    bad_fraction_threshold = min(max(float(instability_bad_fraction), 0.0), 1.0)
    partial_window_min_epochs = max(
        1,
        math.ceil(float(window_size) * float(bad_fraction_threshold)),
    )
    first_bad_window_epoch: int | None = None
    rolling_flags: list[bool] = []
    for row in rows:
        epoch = _int_or_none(row.get("epoch"))
        if epoch is not None:
            epochs.append(epoch)
        lr = _float_or_none(row.get("learning_rate"))
        if lr is not None:
            learning_rates.append(lr)
        loss_components = row.get("loss_components")
        per_axis = row.get("per_axis_decomposition")
        if isinstance(loss_components, Mapping):
            stage_index = _int_or_none(loss_components.get("pr95_stage_index"))
            if stage_index is not None:
                pr95_stage_indices.append(stage_index)
            stage_uses_muon = _bool_metric_or_none(
                loss_components.get("pr95_stage_uses_muon")
            )
            if stage_uses_muon is not None:
                pr95_stage_uses_muon_flags.append(stage_uses_muon)
            pact_uses_muon = _bool_metric_or_none(
                loss_components.get("pact_optimizer_uses_muon")
            )
            if pact_uses_muon is not None:
                pact_muon_flags.append(pact_uses_muon)
        pose_loss = (
            _float_or_none(loss_components.get("loss_part_pose_distill"))
            if isinstance(loss_components, Mapping)
            else None
        )
        pose_axis = (
            _float_or_none(per_axis.get("pose"))
            if isinstance(per_axis, Mapping)
            else None
        )
        seg_axis = (
            _float_or_none(per_axis.get("seg"))
            if isinstance(per_axis, Mapping)
            else None
        )
        if pose_loss is not None:
            pose_losses.append(pose_loss)
        if pose_axis is not None:
            pose_axes.append(pose_axis)
        if seg_axis is not None:
            seg_axes.append(seg_axis)
        seg_weight = _effective_distillation_weight(loss_components)
        if seg_weight is not None:
            segnet_distillation_weights.append(seg_weight)
        bad = bool(
            (pose_loss is not None and pose_loss >= pose_loss_instability_threshold)
            or (pose_axis is not None and pose_axis >= pose_axis_instability_threshold)
        )
        rolling_flags.append(bad)
        if bad and epoch is not None:
            bad_epochs.append(epoch)
        window = rolling_flags[-window_size:]
        if len(window) == window_size and first_bad_window_epoch is None:
            bad_fraction = sum(1 for flag in window if flag) / float(window_size)
            if bad_fraction >= bad_fraction_threshold:
                first_bad_window_epoch = epoch
    last_window = rolling_flags[-window_size:]
    last_bad_fraction = (
        sum(1 for flag in last_window if flag) / float(len(last_window))
        if last_window
        else 0.0
    )
    partial_window_instability = bool(
        first_bad_window_epoch is None
        and len(rolling_flags) < window_size
        and len(rolling_flags) >= partial_window_min_epochs
        and last_bad_fraction >= bad_fraction_threshold
    )
    if partial_window_instability:
        first_bad_window_epoch = epochs[-1] if epochs else None
    observed_lr = learning_rates[-1] if learning_rates else None
    ever_instability = bool(first_bad_window_epoch is not None or partial_window_instability)
    active_latest_window = bool(last_bad_fraction >= bad_fraction_threshold)
    recovered_instability = bool(
        first_bad_window_epoch is not None
        and not partial_window_instability
        and len(rolling_flags) >= window_size
        and last_bad_fraction == 0.0
    )
    instability = bool(
        partial_window_instability
        or (
            first_bad_window_epoch is not None
            and not recovered_instability
            and active_latest_window
        )
    )
    recommended_lr = (
        max(float(observed_lr) * float(learning_rate_multiplier), 1.0e-6)
        if instability and observed_lr is not None
        else None
    )
    mutations: list[str] = []
    if instability:
        mutations.extend(
            [
                "lower_learning_rate_from_pose_instability_telemetry",
                "preserve_pose_instability_guard_for_relaunch",
                "treat_previous_hi_nerv_run_as_fit_failure_not_rate_negative",
            ]
        )
    tail_window_size = max(1, int(_POSE_TAIL_BURST_WINDOW_EPOCHS))
    tail_recent_axes = pose_axes[-tail_window_size:]
    tail_baseline_axes = (
        pose_axes[:-tail_window_size]
        if len(pose_axes) > tail_window_size
        else pose_axes
    )
    tail_baseline_median = _median(tail_baseline_axes)
    tail_threshold = (
        max(
            float(_POSE_TAIL_BURST_MIN_AXIS),
            float(tail_baseline_median) * float(_POSE_TAIL_BURST_MEDIAN_MULTIPLIER),
        )
        if tail_baseline_median is not None
        else float(_POSE_TAIL_BURST_MIN_AXIS)
    )
    tail_bad_count = sum(1 for value in tail_recent_axes if value >= tail_threshold)
    tail_bad_fraction = (
        tail_bad_count / float(len(tail_recent_axes)) if tail_recent_axes else 0.0
    )
    pose_tail_burst = bool(
        len(pose_axes) >= _POSE_TAIL_BURST_MIN_EPOCHS
        and tail_bad_count > 0
        and tail_bad_fraction >= float(_POSE_TAIL_BURST_BAD_FRACTION)
    )
    if pose_tail_burst:
        mutations.extend(
            [
                "build_xray_hardpair_hitlist_from_full_video_pose_tail",
                "launch_hard_pair_prioritized_sampler_successor",
                "preserve_random_full_video_fill_when_prioritizing_hard_pairs",
                "treat_previous_hi_nerv_run_as_hard_pair_tail_fit_failure_not_rate_negative",
            ]
        )
    seg_first_window_median = _median(seg_axes[: _SEG_STAGNATION_WINDOW_EPOCHS])
    seg_last_window_median = _median(seg_axes[-_SEG_STAGNATION_WINDOW_EPOCHS:])
    seg_previous_window_median = _median(
        seg_axes[
            -2 * _TRAINING_CONTROL_RECENT_WINDOW_EPOCHS : -_TRAINING_CONTROL_RECENT_WINDOW_EPOCHS
        ]
        if len(seg_axes) >= 2 * _TRAINING_CONTROL_RECENT_WINDOW_EPOCHS
        else []
    )
    seg_recent_window_median = _median(
        seg_axes[-_TRAINING_CONTROL_RECENT_WINDOW_EPOCHS:]
    )
    seg_recent_relative_improvement = _relative_improvement(
        seg_previous_window_median,
        seg_recent_window_median,
    )
    seg_relative_improvement = _relative_improvement(
        seg_first_window_median,
        seg_last_window_median,
    )
    seg_stagnation = bool(
        len(seg_axes) >= _SEG_STAGNATION_MIN_EPOCHS
        and seg_relative_improvement is not None
        and seg_relative_improvement < _SEG_STAGNATION_MIN_RELATIVE_IMPROVEMENT
    )
    observed_seg_weight = _median(
        segnet_distillation_weights[-_SEG_STAGNATION_WINDOW_EPOCHS:]
    )
    recommended_seg_weight = (
        recommend_segnet_distillation_weight_for_stagnation(observed_seg_weight)
        if seg_stagnation
        else None
    )
    if seg_stagnation:
        mutations.extend(
            [
                "increase_segnet_distillation_weight_from_stagnation_telemetry",
                "preserve_pose_guard_while_raising_segnet_pressure",
                "treat_previous_hi_nerv_run_as_segnet_fit_failure_not_rate_negative",
            ]
        )
    current_stage_index = pr95_stage_indices[-1] if pr95_stage_indices else None
    current_stage_uses_muon = (
        pr95_stage_uses_muon_flags[-1] if pr95_stage_uses_muon_flags else None
    )
    last_epoch = max(epochs) if epochs else None
    pr95_curriculum_observed = bool(pr95_stage_indices)
    pact_muon_observed = any(pact_muon_flags)
    pr95_muon_observed = any(pr95_stage_uses_muon_flags)
    pr95_stage_status = _canonical_pr95_stage_status(
        current_epoch=last_epoch,
        current_stage_index=current_stage_index,
    )
    canonical_expected_stage = _int_or_none(
        pr95_stage_status.get("canonical_expected_stage_index")
    )
    authoritative_stage_index = (
        canonical_expected_stage
        if canonical_expected_stage is not None
        else current_stage_index
    )
    pr95_stage_mismatch_detected = (
        pr95_stage_status.get("observed_stage_matches_canonical_epoch") is False
    )
    pr95_final_stage_reached = bool(
        authoritative_stage_index is not None
        and authoritative_stage_index >= _PR95_FINAL_MUON_STAGE_INDEX
    )
    pr95_final_stage_muon_expected = pr95_final_stage_reached
    pr95_final_stage_muon_missing = bool(
        pr95_final_stage_muon_expected and current_stage_uses_muon is False
    )
    pr95_pre_final_no_muon_expected = bool(
        authoritative_stage_index is not None
        and authoritative_stage_index < _PR95_FINAL_MUON_STAGE_INDEX
        and current_stage_uses_muon is False
        and not pr95_stage_mismatch_detected
    )
    if pr95_stage_mismatch_detected:
        mutations.extend(
            [
                "fix_pr95_stage_telemetry_or_curriculum_epoch_routing",
                "treat_previous_hi_nerv_run_as_stage_telemetry_failure_not_rate_negative",
            ]
        )
    if pr95_final_stage_muon_missing:
        mutations.extend(
            [
                "fix_pr95_final_stage_muon_optimizer_routing",
                "treat_previous_hi_nerv_run_as_optimizer_wiring_failure_not_rate_negative",
            ]
        )
    if pr95_final_stage_muon_missing:
        optimizer_stage_assessment = "pr95_final_stage_muon_missing"
    elif pr95_pre_final_no_muon_expected:
        optimizer_stage_assessment = "pr95_curriculum_pre_final_muon_not_expected"
    elif pr95_muon_observed:
        optimizer_stage_assessment = "pr95_curriculum_final_muon_observed"
    elif pact_muon_observed:
        optimizer_stage_assessment = "pact_muon_adamw_partition_observed"
    elif pr95_curriculum_observed:
        optimizer_stage_assessment = "pr95_curriculum_stage_observed"
    else:
        optimizer_stage_assessment = "optimizer_control_telemetry_missing"
    return {
        "schema": TELEMETRY_FEEDBACK_SCHEMA,
        "row_count": len(rows),
        "num_pairs": CONTEST_PAIR_COUNT,
        "first_epoch": min(epochs) if epochs else None,
        "last_epoch": last_epoch,
        "observed_learning_rate": observed_lr,
        "pose_loss_instability_threshold": float(pose_loss_instability_threshold),
        "pose_axis_instability_threshold": float(pose_axis_instability_threshold),
        "instability_window_epochs": int(window_size),
        "instability_bad_fraction_threshold": float(bad_fraction_threshold),
        "partial_window_instability_min_epochs": int(partial_window_min_epochs),
        "pose_instability_partial_window_detected": partial_window_instability,
        "pose_instability_ever_detected": ever_instability,
        "pose_instability_recovered": recovered_instability,
        "pose_instability_active_latest_window": active_latest_window,
        "pose_bad_epoch_count": len(bad_epochs),
        "pose_bad_epoch_fraction": (
            len(bad_epochs) / float(len(rows)) if rows else 0.0
        ),
        "pose_instability_detected": instability,
        "pose_instability_first_epoch": first_bad_window_epoch,
        "pose_instability_last_window_bad_fraction": last_bad_fraction,
        "pose_tail_burst_detected": pose_tail_burst,
        "pose_tail_burst_min_epochs": _POSE_TAIL_BURST_MIN_EPOCHS,
        "pose_tail_burst_recent_window_epochs": len(tail_recent_axes),
        "pose_tail_burst_config_window_epochs": _POSE_TAIL_BURST_WINDOW_EPOCHS,
        "pose_tail_burst_min_axis": _POSE_TAIL_BURST_MIN_AXIS,
        "pose_tail_burst_median_multiplier": _POSE_TAIL_BURST_MEDIAN_MULTIPLIER,
        "pose_tail_burst_bad_fraction_threshold": _POSE_TAIL_BURST_BAD_FRACTION,
        "pose_tail_burst_recent_bad_count": tail_bad_count,
        "pose_tail_burst_recent_bad_fraction": tail_bad_fraction,
        "pose_tail_burst_threshold": tail_threshold,
        "pose_tail_burst_recent_p95": _quantile(tail_recent_axes, 0.95),
        "pose_tail_burst_recent_max": max(tail_recent_axes) if tail_recent_axes else None,
        "pose_tail_burst_baseline_median": tail_baseline_median,
        "max_pose_distill_loss": max(pose_losses) if pose_losses else None,
        "max_pose_axis": max(pose_axes) if pose_axes else None,
        "median_pose_distill_loss": _median(pose_losses),
        "median_pose_axis": _median(pose_axes),
        "median_seg_axis": _median(seg_axes),
        "seg_stagnation_min_epochs": _SEG_STAGNATION_MIN_EPOCHS,
        "seg_stagnation_window_epochs": _SEG_STAGNATION_WINDOW_EPOCHS,
        "seg_stagnation_min_relative_improvement": (
            _SEG_STAGNATION_MIN_RELATIVE_IMPROVEMENT
        ),
        "seg_stagnation_first_window_median": seg_first_window_median,
        "seg_stagnation_last_window_median": seg_last_window_median,
        "seg_stagnation_relative_improvement": seg_relative_improvement,
        "seg_recent_window_epochs": _TRAINING_CONTROL_RECENT_WINDOW_EPOCHS,
        "seg_recent_min_relative_improvement": (
            _TRAINING_CONTROL_MIN_RECENT_RELATIVE_IMPROVEMENT
        ),
        "seg_previous_window_median": seg_previous_window_median,
        "seg_recent_window_median": seg_recent_window_median,
        "seg_recent_relative_improvement": seg_recent_relative_improvement,
        "seg_stagnation_detected": seg_stagnation,
        "optimizer_control_observed": bool(
            pr95_curriculum_observed or pact_muon_flags
        ),
        "optimizer_stage_assessment": optimizer_stage_assessment,
        "optimizer_muon_observed": bool(pr95_muon_observed or pact_muon_observed),
        "pact_muon_adamw_observed": bool(pact_muon_observed),
        "pr95_curriculum_observed": pr95_curriculum_observed,
        "pr95_current_stage_index": current_stage_index,
        "pr95_canonical_expected_stage_index": canonical_expected_stage,
        "pr95_authoritative_stage_index": authoritative_stage_index,
        "pr95_stage_mismatch_detected": pr95_stage_mismatch_detected,
        "pr95_stage_uses_muon_current": current_stage_uses_muon,
        "pr95_final_stage_reached": pr95_final_stage_reached,
        "pr95_final_stage_muon_expected_currently": (
            pr95_final_stage_muon_expected
        ),
        "pr95_final_stage_muon_missing": pr95_final_stage_muon_missing,
        "pr95_pre_final_no_muon_expected": pr95_pre_final_no_muon_expected,
        "pr95_stage_status": pr95_stage_status,
        "observed_segnet_distillation_weight": observed_seg_weight,
        "recommended_learning_rate": recommended_lr,
        "recommended_learning_rate_multiplier": (
            float(learning_rate_multiplier) if instability else None
        ),
        "recommended_segnet_distillation_weight": recommended_seg_weight,
        "recommended_segnet_distillation_weight_multiplier": (
            _SEG_STAGNATION_WEIGHT_MULTIPLIER if seg_stagnation else None
        ),
        "recommended_launch_mutations": mutations,
        **FALSE_AUTHORITY,
    }


def _snerv_scorer_tether_health(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metric_health = {
        metric: _snerv_scorer_tether_metric_health(rows, metric=metric)
        for metric in _SNERV_SCORER_TETHER_METRICS
    }
    missing_metrics = [
        metric
        for metric, health in metric_health.items()
        if bool(health.get("metric_missing_detected"))
    ]
    lambda_inactive_metrics = [
        metric
        for metric, health in metric_health.items()
        if bool(health.get("lambda_inactive_detected"))
    ]
    degenerate_risk = bool(missing_metrics)
    blockers: list[str] = []
    if degenerate_risk:
        blockers.append("snerv_scorer_domain_tether_missing_telemetry")
    if "snerv_posenet_yuv6_pair_distill" in missing_metrics:
        blockers.append("snerv_posenet_yuv6_pair_distill_metric_missing_telemetry")
    if "snerv_segnet_last_frame_distill" in missing_metrics:
        blockers.append("snerv_segnet_last_frame_distill_metric_missing_telemetry")
    if lambda_inactive_metrics:
        blockers.append("snerv_scorer_domain_tether_lambda_inactive_telemetry")
    recommended: list[str] = []
    if degenerate_risk:
        recommended.extend(
            [
                "bind_snerv_posenet_yuv6_and_segnet_last_frame_distill_metrics_before_more_long_training",
                "reject_snerv_degenerate_renderer_even_when_archive_bytes_are_frontier",
                "preserve_snerv_snar2_snsa2_byte_layout_while_rebinding_scorer_tethers",
            ]
        )
    return {
        "schema": "snerv_scorer_domain_tether_health.v1",
        "row_count": len(rows),
        "recent_window_epochs": _TRAINING_CONTROL_RECENT_WINDOW_EPOCHS,
        "recent_missing_fraction_threshold": (
            _SNERV_SCORER_TETHER_RECENT_MISSING_FRACTION
        ),
        "metric_health": metric_health,
        "missing_metrics": missing_metrics,
        "lambda_inactive_metrics": lambda_inactive_metrics,
        "degenerate_renderer_risk_detected": degenerate_risk,
        "blockers": _dedupe_strings(blockers),
        "recommended_launch_mutations": recommended,
        **FALSE_AUTHORITY,
    }


def _snerv_scorer_tether_metric_health(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    missing_flags: list[bool] = []
    lambda_active_flags: list[bool] = []
    recent_missing_flags: list[bool] = []
    recent_lambda_active_flags: list[bool] = []
    recent_rows = list(rows)[-_TRAINING_CONTROL_RECENT_WINDOW_EPOCHS:]
    for row in rows:
        loss_components = row.get("loss_components")
        if not isinstance(loss_components, Mapping):
            continue
        missing_value = _float_or_none(
            loss_components.get(f"dual_ascent_missing_metric__{metric}")
        )
        lambda_value = _float_or_none(
            loss_components.get(f"dual_ascent_lambda__{metric}")
        )
        if missing_value is not None:
            missing_flags.append(missing_value >= 0.5)
        if lambda_value is not None:
            lambda_active_flags.append(
                abs(float(lambda_value)) > _SNERV_SCORER_TETHER_LAMBDA_ACTIVE_EPS
            )
    for row in recent_rows:
        loss_components = row.get("loss_components")
        if not isinstance(loss_components, Mapping):
            continue
        missing_value = _float_or_none(
            loss_components.get(f"dual_ascent_missing_metric__{metric}")
        )
        lambda_value = _float_or_none(
            loss_components.get(f"dual_ascent_lambda__{metric}")
        )
        if missing_value is not None:
            recent_missing_flags.append(missing_value >= 0.5)
        if lambda_value is not None:
            recent_lambda_active_flags.append(
                abs(float(lambda_value)) > _SNERV_SCORER_TETHER_LAMBDA_ACTIVE_EPS
            )
    missing_fraction = (
        sum(1 for flag in missing_flags if flag) / float(len(missing_flags))
        if missing_flags
        else None
    )
    recent_missing_fraction = (
        sum(1 for flag in recent_missing_flags if flag)
        / float(len(recent_missing_flags))
        if recent_missing_flags
        else None
    )
    lambda_active_fraction = (
        sum(1 for flag in lambda_active_flags if flag)
        / float(len(lambda_active_flags))
        if lambda_active_flags
        else None
    )
    recent_lambda_active_fraction = (
        sum(1 for flag in recent_lambda_active_flags if flag)
        / float(len(recent_lambda_active_flags))
        if recent_lambda_active_flags
        else None
    )
    metric_missing = bool(
        recent_missing_fraction is not None
        and recent_missing_fraction
        >= _SNERV_SCORER_TETHER_RECENT_MISSING_FRACTION
    )
    lambda_inactive = bool(
        recent_lambda_active_fraction is not None
        and recent_lambda_active_fraction == 0.0
    )
    return {
        "metric": metric,
        "missing_metric_observation_count": len(missing_flags),
        "missing_metric_recent_observation_count": len(recent_missing_flags),
        "missing_metric_fraction": missing_fraction,
        "missing_metric_recent_fraction": recent_missing_fraction,
        "lambda_observation_count": len(lambda_active_flags),
        "lambda_recent_observation_count": len(recent_lambda_active_flags),
        "lambda_active_fraction": lambda_active_fraction,
        "lambda_recent_active_fraction": recent_lambda_active_fraction,
        "metric_missing_detected": metric_missing,
        "lambda_inactive_detected": lambda_inactive,
    }


def _nerv_train_time_control_health(
    rows: Sequence[Mapping[str, Any]],
    *,
    family_key: str,
) -> dict[str, Any]:
    family_key = _family_key(family_key)
    metric_names = (
        f"{family_key}_segnet_last_frame_distill",
        f"{family_key}_posenet_yuv6_pair_distill",
    )
    metric_health = {
        metric: _snerv_scorer_tether_metric_health(rows, metric=metric)
        for metric in metric_names
    }
    section_health = _nerv_section_byte_control_health(
        rows,
        family_key=family_key,
    )
    inactive_metrics = [
        metric
        for metric, health in metric_health.items()
        if bool(health.get("lambda_inactive_detected"))
    ]
    missing_metrics = [
        metric
        for metric, health in metric_health.items()
        if bool(health.get("metric_missing_detected"))
    ]
    blockers: list[str] = []
    if missing_metrics:
        blockers.append(f"{family_key}_train_time_dual_missing_metric_telemetry")
    if inactive_metrics:
        blockers.append(f"{family_key}_train_time_dual_lambda_inactive_telemetry")
    if f"{family_key}_segnet_last_frame_distill" in missing_metrics:
        blockers.append(f"{family_key}_train_time_dual_segnet_metric_missing_telemetry")
    if f"{family_key}_posenet_yuv6_pair_distill" in missing_metrics:
        blockers.append(f"{family_key}_train_time_dual_posenet_metric_missing_telemetry")
    if f"{family_key}_segnet_last_frame_distill" in inactive_metrics:
        blockers.append(f"{family_key}_train_time_dual_segnet_lambda_inactive_telemetry")
    if f"{family_key}_posenet_yuv6_pair_distill" in inactive_metrics:
        blockers.append(f"{family_key}_train_time_dual_posenet_lambda_inactive_telemetry")
    if bool(section_health.get("section_rate_metric_observed")) and not bool(
        section_health.get("section_byte_dual_lambda_active_observed")
    ):
        blockers.append(
            f"{family_key}_train_time_section_byte_dual_lambda_inactive_telemetry"
        )
    if bool(section_health.get("section_rate_metric_observed")) and not bool(
        section_health.get("archive_rate_metric_observed")
    ):
        blockers.append(f"{family_key}_train_time_archive_rate_metric_missing_telemetry")
    if bool(section_health.get("archive_rate_metric_observed")) and not bool(
        section_health.get("archive_byte_dual_lambda_active_observed")
    ):
        blockers.append(
            f"{family_key}_train_time_archive_byte_dual_lambda_inactive_telemetry"
        )
    if (
        bool(section_health.get("section_byte_dual_lambda_active_observed"))
        and not bool(section_health.get("section_byte_dual_weight_applied_observed"))
    ):
        blockers.append(
            f"{family_key}_train_time_section_byte_dual_weight_not_applied_telemetry"
        )
    if (
        bool(section_health.get("archive_byte_dual_lambda_active_observed"))
        and not bool(section_health.get("archive_byte_dual_weight_applied_observed"))
    ):
        blockers.append(
            f"{family_key}_train_time_archive_byte_dual_weight_not_applied_telemetry"
        )
    if bool(section_health.get("section_byte_dual_zero_base_masked_observed")):
        blockers.append(f"{family_key}_train_time_section_byte_dual_zero_base_masked")
    family_slug = "hinerv" if family_key == "hi_nerv" else family_key
    recommended: list[str] = []
    if blockers:
        recommended.extend(
            [
                f"require_{family_slug}_scorer_and_section_dual_lambdas_before_long_training_reuse",
                f"relaunch_{family_slug}_with_live_byte_cap_dual_ascent_actuating_from_step_zero",
            ]
        )
    return {
        "schema": f"{family_key}_train_time_control_health.v1",
        "family": family_key,
        "row_count": len(rows),
        "recent_window_epochs": _TRAINING_CONTROL_RECENT_WINDOW_EPOCHS,
        "metric_health": metric_health,
        "section_byte_control_health": section_health,
        "missing_metrics": missing_metrics,
        "lambda_inactive_metrics": inactive_metrics,
        "control_inert_risk_detected": bool(blockers),
        "blockers": _dedupe_strings(blockers),
        "recommended_launch_mutations": recommended,
        **FALSE_AUTHORITY,
    }


def _nerv_section_byte_control_health(
    rows: Sequence[Mapping[str, Any]],
    *,
    family_key: str,
) -> dict[str, Any]:
    family_key = _family_key(family_key)
    all_rate_flags: list[bool] = []
    all_lambda_flags: list[bool] = []
    all_archive_rate_flags: list[bool] = []
    all_archive_lambda_flags: list[bool] = []
    all_archive_applied_flags: list[bool] = []
    all_section_applied_flags: list[bool] = []
    all_section_zero_base_masked_flags: list[bool] = []
    recent_rate_flags: list[bool] = []
    recent_lambda_flags: list[bool] = []
    recent_archive_rate_flags: list[bool] = []
    recent_archive_lambda_flags: list[bool] = []
    recent_archive_applied_flags: list[bool] = []
    recent_section_applied_flags: list[bool] = []
    recent_section_zero_base_masked_flags: list[bool] = []
    recent_rows = list(rows)[-_TRAINING_CONTROL_RECENT_WINDOW_EPOCHS:]
    for row in rows:
        (
            rate_seen,
            lambda_active,
            archive_rate_seen,
            archive_lambda_active,
            section_weight_applied,
            archive_weight_applied,
            section_zero_base_masked,
        ) = _nerv_section_byte_control_row_flags(row, family_key=family_key)
        all_rate_flags.append(rate_seen)
        all_lambda_flags.append(lambda_active)
        all_archive_rate_flags.append(archive_rate_seen)
        all_archive_lambda_flags.append(archive_lambda_active)
        all_section_applied_flags.append(section_weight_applied)
        all_archive_applied_flags.append(archive_weight_applied)
        all_section_zero_base_masked_flags.append(section_zero_base_masked)
    for row in recent_rows:
        (
            rate_seen,
            lambda_active,
            archive_rate_seen,
            archive_lambda_active,
            section_weight_applied,
            archive_weight_applied,
            section_zero_base_masked,
        ) = _nerv_section_byte_control_row_flags(row, family_key=family_key)
        recent_rate_flags.append(rate_seen)
        recent_lambda_flags.append(lambda_active)
        recent_archive_rate_flags.append(archive_rate_seen)
        recent_archive_lambda_flags.append(archive_lambda_active)
        recent_section_applied_flags.append(section_weight_applied)
        recent_archive_applied_flags.append(archive_weight_applied)
        recent_section_zero_base_masked_flags.append(section_zero_base_masked)
    rate_observed = any(all_rate_flags)
    lambda_active_observed = any(all_lambda_flags)
    archive_rate_observed = any(all_archive_rate_flags)
    archive_lambda_active_observed = any(all_archive_lambda_flags)
    archive_weight_applied_observed = any(all_archive_applied_flags)
    section_weight_applied_observed = any(all_section_applied_flags)
    section_zero_base_masked_observed = any(all_section_zero_base_masked_flags)
    recent_rate_observed = any(recent_rate_flags)
    recent_lambda_active_observed = any(recent_lambda_flags)
    recent_archive_rate_observed = any(recent_archive_rate_flags)
    recent_archive_lambda_active_observed = any(recent_archive_lambda_flags)
    recent_archive_weight_applied_observed = any(recent_archive_applied_flags)
    recent_section_weight_applied_observed = any(recent_section_applied_flags)
    recent_section_zero_base_masked_observed = any(
        recent_section_zero_base_masked_flags
    )
    return {
        "schema": f"{family_key}_section_byte_control_health.v1",
        "family": family_key,
        "archive_rate_metric_observation_count": sum(
            1 for flag in all_archive_rate_flags if flag
        ),
        "archive_rate_metric_recent_observation_count": sum(
            1 for flag in recent_archive_rate_flags if flag
        ),
        "archive_lambda_observation_count": sum(
            1 for flag in all_archive_lambda_flags if flag
        ),
        "archive_lambda_recent_observation_count": sum(
            1 for flag in recent_archive_lambda_flags if flag
        ),
        "rate_metric_observation_count": sum(1 for flag in all_rate_flags if flag),
        "rate_metric_recent_observation_count": sum(
            1 for flag in recent_rate_flags if flag
        ),
        "lambda_observation_count": sum(1 for flag in all_lambda_flags if flag),
        "lambda_recent_observation_count": sum(
            1 for flag in recent_lambda_flags if flag
        ),
        "archive_rate_metric_observed": bool(archive_rate_observed),
        "archive_rate_metric_recent_observed": bool(recent_archive_rate_observed),
        "archive_byte_dual_lambda_active_observed": bool(
            archive_lambda_active_observed
        ),
        "archive_byte_dual_lambda_recent_active_observed": bool(
            recent_archive_lambda_active_observed
        ),
        "archive_byte_dual_weight_applied_observed": bool(
            archive_weight_applied_observed
        ),
        "archive_byte_dual_weight_recent_applied_observed": bool(
            recent_archive_weight_applied_observed
        ),
        "section_rate_metric_observed": bool(rate_observed),
        "section_rate_metric_recent_observed": bool(recent_rate_observed),
        "section_byte_dual_lambda_active_observed": bool(lambda_active_observed),
        "section_byte_dual_lambda_recent_active_observed": bool(
            recent_lambda_active_observed
        ),
        "section_byte_dual_weight_applied_observed": bool(
            section_weight_applied_observed
        ),
        "section_byte_dual_weight_recent_applied_observed": bool(
            recent_section_weight_applied_observed
        ),
        "section_byte_dual_zero_base_masked_observed": bool(
            section_zero_base_masked_observed
        ),
        "section_byte_dual_zero_base_masked_recent_observed": bool(
            recent_section_zero_base_masked_observed
        ),
    }


def _nerv_section_byte_control_row_flags(
    row: Mapping[str, Any],
    *,
    family_key: str,
) -> tuple[bool, bool, bool, bool, bool, bool, bool]:
    family_key = _family_key(family_key)
    loss_components = row.get("loss_components")
    sources = [row]
    if isinstance(loss_components, Mapping):
        sources.append(loss_components)
    archive_rate_seen = any(
        str(key) == "train_time_archive_rate_score"
        and _float_or_none(value) is not None
        for source in sources
        for key, value in source.items()
    )
    archive_lambda_active = any(
        str(key) == f"dual_ascent_lambda__{family_key}_archive_total_bytes"
        and (value_float := _float_or_none(value)) is not None
        and abs(float(value_float)) > _HINERV_TRAIN_TIME_CONTROL_LAMBDA_ACTIVE_EPS
        for source in sources
        for key, value in source.items()
    )
    rate_seen = any(
        str(key).startswith("train_time_section_rate_score__")
        and _float_or_none(value) is not None
        for source in sources
        for key, value in source.items()
    )
    lambda_active = any(
        str(key).startswith(f"dual_ascent_lambda__{family_key}_")
        and str(key).endswith("_section_bytes")
        and (value_float := _float_or_none(value)) is not None
        and abs(float(value_float)) > _HINERV_TRAIN_TIME_CONTROL_LAMBDA_ACTIVE_EPS
        for source in sources
        for key, value in source.items()
    )
    section_weight_applied = any(
        (
            str(key).startswith(f"dual_ascent_weight_applied__{family_key}_")
            or str(key).startswith(
                f"dual_ascent_effective_loss_weight__{family_key}_"
            )
        )
        and str(key).endswith("_section_bytes")
        and (value_float := _float_or_none(value)) is not None
        and abs(float(value_float)) > _HINERV_TRAIN_TIME_CONTROL_LAMBDA_ACTIVE_EPS
        for source in sources
        for key, value in source.items()
    )
    archive_weight_applied = any(
        str(key)
        in {
            f"dual_ascent_weight_applied__{family_key}_archive_total_bytes",
            f"dual_ascent_effective_loss_weight__{family_key}_archive_total_bytes",
        }
        and (value_float := _float_or_none(value)) is not None
        and abs(float(value_float)) > _HINERV_TRAIN_TIME_CONTROL_LAMBDA_ACTIVE_EPS
        for source in sources
        for key, value in source.items()
    )
    section_zero_base_masked = any(
        str(key).startswith(f"dual_ascent_zero_base_masked__{family_key}_")
        and str(key).endswith("_section_bytes")
        and (value_float := _float_or_none(value)) is not None
        and abs(float(value_float)) > _HINERV_TRAIN_TIME_CONTROL_LAMBDA_ACTIVE_EPS
        for source in sources
        for key, value in source.items()
    )
    return (
        rate_seen,
        lambda_active,
        archive_rate_seen,
        archive_lambda_active,
        section_weight_applied,
        archive_weight_applied,
        section_zero_base_masked,
    )


def _gradient_multiplier_control_health(
    rows: Sequence[Mapping[str, Any]],
    *,
    family_key: str,
) -> dict[str, Any]:
    all_requested: list[bool] = []
    all_applied: list[bool] = []
    all_missing: list[bool] = []
    all_noop: list[bool] = []
    recent_requested: list[bool] = []
    recent_applied: list[bool] = []
    recent_missing: list[bool] = []
    recent_noop: list[bool] = []
    recent_rows = list(rows)[-_TRAINING_CONTROL_RECENT_WINDOW_EPOCHS:]
    for row in rows:
        requested, applied, missing, noop = _gradient_multiplier_control_row_flags(row)
        all_requested.append(requested)
        all_applied.append(applied)
        all_missing.append(missing)
        all_noop.append(noop)
    for row in recent_rows:
        requested, applied, missing, noop = _gradient_multiplier_control_row_flags(row)
        recent_requested.append(requested)
        recent_applied.append(applied)
        recent_missing.append(missing)
        recent_noop.append(noop)
    requested_observed = any(all_requested)
    applied_observed = any(all_applied)
    missing_observed = any(all_missing)
    noop_observed = any(all_noop)
    recent_requested_observed = any(recent_requested)
    recent_applied_observed = any(recent_applied)
    recent_missing_observed = any(recent_missing)
    recent_noop_observed = any(recent_noop)
    blockers: list[str] = []
    if requested_observed and not applied_observed:
        blockers.append(
            f"{family_key}_train_time_gradient_multiplier_never_applied_telemetry"
        )
    if requested_observed and missing_observed:
        blockers.append(
            f"{family_key}_train_time_gradient_multiplier_missing_requested_leaf_telemetry"
        )
    if requested_observed and noop_observed:
        blockers.append(
            f"{family_key}_train_time_gradient_multiplier_requested_but_unapplied_telemetry"
        )
    if recent_requested_observed and not recent_applied_observed:
        blockers.append(
            f"{family_key}_train_time_gradient_multiplier_recent_never_applied_telemetry"
        )
    if recent_requested_observed and recent_missing_observed:
        blockers.append(
            f"{family_key}_train_time_gradient_multiplier_recent_missing_requested_leaf_telemetry"
        )
    if recent_requested_observed and recent_noop_observed:
        blockers.append(
            f"{family_key}_train_time_gradient_multiplier_recent_requested_but_unapplied_telemetry"
        )
    family_slug = "hi_nerv" if family_key == "hi_nerv" else family_key
    recommended: list[str] = []
    if blockers:
        recommended.extend(
            [
                f"refresh_{family_slug}_gradient_multiplier_names_from_current_model_leaf_inventory",
                f"relaunch_{family_slug}_with_verified_decoder_weight_waterfill_actuators",
                f"treat_previous_{family_slug}_run_as_optimizer_actuator_wiring_failure_not_rate_negative",
            ]
        )
    return {
        "schema": "nerv_gradient_multiplier_control_health.v1",
        "family": family_key,
        "row_count": len(rows),
        "recent_window_epochs": _TRAINING_CONTROL_RECENT_WINDOW_EPOCHS,
        "requested_observation_count": sum(1 for flag in all_requested if flag),
        "requested_recent_observation_count": sum(
            1 for flag in recent_requested if flag
        ),
        "applied_observation_count": sum(1 for flag in all_applied if flag),
        "applied_recent_observation_count": sum(
            1 for flag in recent_applied if flag
        ),
        "missing_requested_observation_count": sum(
            1 for flag in all_missing if flag
        ),
        "missing_requested_recent_observation_count": sum(
            1 for flag in recent_missing if flag
        ),
        "noop_observation_count": sum(1 for flag in all_noop if flag),
        "noop_recent_observation_count": sum(1 for flag in recent_noop if flag),
        "requested_observed": bool(requested_observed),
        "requested_recent_observed": bool(recent_requested_observed),
        "applied_observed": bool(applied_observed),
        "applied_recent_observed": bool(recent_applied_observed),
        "missing_requested_observed": bool(missing_observed),
        "missing_requested_recent_observed": bool(recent_missing_observed),
        "requested_but_unapplied_observed": bool(noop_observed),
        "requested_but_unapplied_recent_observed": bool(recent_noop_observed),
        "control_inert_risk_detected": bool(blockers),
        "blockers": _dedupe_strings(blockers),
        "recommended_launch_mutations": _dedupe_strings(recommended),
        **FALSE_AUTHORITY,
    }


def _gradient_multiplier_control_row_flags(
    row: Mapping[str, Any],
) -> tuple[bool, bool, bool, bool]:
    loss_components = row.get("loss_components")
    sources = [row]
    if isinstance(loss_components, Mapping):
        sources.append(loss_components)
    requested = any(
        str(key) == "gradient_multiplier_requested_control_count"
        and (value_float := _float_or_none(value)) is not None
        and float(value_float) > 0.0
        for source in sources
        for key, value in source.items()
    )
    applied = any(
        str(key) == "gradient_multiplier_applied_leaf_count"
        and (value_float := _float_or_none(value)) is not None
        and float(value_float) > 0.0
        for source in sources
        for key, value in source.items()
    )
    missing = any(
        str(key) == "gradient_multiplier_missing_requested_count"
        and (value_float := _float_or_none(value)) is not None
        and float(value_float) > 0.0
        for source in sources
        for key, value in source.items()
    )
    noop = any(
        str(key) == "gradient_multiplier_requested_but_unapplied"
        and (value_float := _float_or_none(value)) is not None
        and float(value_float) > 0.0
        for source in sources
        for key, value in source.items()
    )
    return requested, applied, missing, noop


def _is_midrun_feedback_snapshot(stop_reason: str | None) -> bool:
    return str(stop_reason or "").strip() in _MIDRUN_STOP_REASONS


def _training_control_recommendation(
    *,
    health: Mapping[str, Any],
    training_stopped: bool,
    measured_pairs: int,
    candidate_pairs: int,
) -> dict[str, Any]:
    """Return a queue-consumable live-training action without mutating jobs."""

    recommended_mutations = list(health.get("recommended_launch_mutations") or [])
    if training_stopped:
        action = "terminal_feedback_no_live_training_action"
        reason = "training_already_terminal_or_not_marked_running"
        should_stop = False
        successor_required = bool(recommended_mutations)
    elif int(measured_pairs) < min(int(candidate_pairs), CONTEST_PAIR_COUNT):
        action = "continue_running_until_full_video_feedback"
        reason = "partial_training_telemetry_not_enough_for_lane_control"
        should_stop = False
        successor_required = False
    elif bool(health.get("pose_instability_detected")) and health.get(
        "recommended_learning_rate"
    ):
        action = "checkpoint_then_supersede_with_lower_learning_rate"
        reason = "active_pose_instability_requires_lower_lr_successor"
        should_stop = True
        successor_required = True
    elif bool(health.get("degenerate_renderer_risk_detected")):
        action = "checkpoint_then_block_degenerate_renderer_successor"
        reason = "snerv_scorer_domain_tether_missing_blocks_live_training"
        should_stop = True
        successor_required = True
    elif bool(
        (health.get("gradient_multiplier_control_health") or {}).get(
            "control_inert_risk_detected"
        )
    ):
        action = "checkpoint_then_supersede_with_verified_optimizer_actuators"
        reason = "gradient_multiplier_waterfill_actuator_configured_but_not_applied"
        should_stop = True
        successor_required = True
    elif bool(health.get("pose_tail_burst_detected")):
        action = "continue_running_queue_hardpair_prioritized_successor"
        reason = "full_video_pose_tail_burst_requires_hardpair_curriculum"
        should_stop = False
        successor_required = True
    elif bool(health.get("seg_stagnation_detected")) and health.get(
        "recommended_segnet_distillation_weight"
    ):
        recent_improvement = _float_or_none(
            health.get("seg_recent_relative_improvement")
        )
        recent_flat = bool(
            recent_improvement is not None
            and recent_improvement < _TRAINING_CONTROL_MIN_RECENT_RELATIVE_IMPROVEMENT
        )
        if recent_flat:
            action = "checkpoint_then_supersede_with_higher_segnet_weight"
            reason = "full_video_segnet_stagnation_recent_window_flat"
            should_stop = True
            successor_required = True
        else:
            action = "continue_running_recheck_segnet_recent_window"
            reason = "segnet_stagnation_detected_but_recent_window_still_improving"
            should_stop = False
            successor_required = True
    else:
        action = "continue_running"
        reason = "no_live_training_replan_trigger"
        should_stop = False
        successor_required = False
    return {
        "schema": "nerv_training_control_recommendation.v1",
        "action": action,
        "reason": reason,
        "should_stop_current_run": should_stop,
        "successor_required": successor_required,
        "recommended_successor_mutations": recommended_mutations,
        "measured_pairs": int(measured_pairs),
        "candidate_pairs": int(candidate_pairs),
        "seg_recent_relative_improvement": health.get(
            "seg_recent_relative_improvement"
        ),
        "seg_recent_min_relative_improvement": (
            _TRAINING_CONTROL_MIN_RECENT_RELATIVE_IMPROVEMENT
        ),
        **FALSE_AUTHORITY,
    }


def _refresh_nested_pr95_stack_binding_blockers(report: dict[str, Any]) -> list[str]:
    removed: list[str] = []
    candidate_paths = [
        ("candidate_curriculum_plan", "pr95_stack_binding"),
        (
            "modelsize_candidate_selection",
            "candidate_curriculum_plan",
            "pr95_stack_binding",
        ),
        ("modelsize_candidate_selection", "pr95_stack_binding"),
    ]
    for path in candidate_paths:
        container = report
        for key in path:
            next_value = container.get(key) if isinstance(container, dict) else None
            if not isinstance(next_value, dict):
                container = {}
                break
            container = next_value
        if not container:
            continue
        blockers = list(container.get("blockers") or [])
        kept = [
            str(blocker)
            for blocker in blockers
            if blocker not in _MLX_PREFILTER_MISSING_BLOCKERS
        ]
        path_removed = [
            str(blocker)
            for blocker in blockers
            if blocker in _MLX_PREFILTER_MISSING_BLOCKERS
        ]
        if not path_removed:
            continue
        container["blockers"] = _dedupe_strings(kept)
        container["missing_count"] = len(container["blockers"])
        satisfied = _int_or_none(container.get("satisfied_count"))
        if satisfied is not None:
            container["satisfied_count"] = satisfied + len(path_removed)
        container["complete"] = not container["blockers"]
        removed.extend(path_removed)
    return _dedupe_strings(removed)


def _mlx_response_authority_blockers(response: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    for key in _FULL_VIDEO_RESPONSE_SCORE_AUTHORITY_KEYS:
        if response.get(key) is True:
            blockers.append(f"mlx_scorer_response_truthy_authority_field:{key}")
    return blockers


def _archive_export_receiver_proof_attached(export: Mapping[str, Any]) -> bool:
    return bool(
        export.get("receiver_proof_ready") is True
        or export.get("receiver_proof_passed") is True
        or export.get("runtime_consumption_proof_ready") is True
        or export.get("receiver_contract_satisfied") is True
    )


def _normalize_full_video_archive_export_report(
    archive_export_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose archive-bound package custody with checkpoint-export field names."""

    export = dict(archive_export_report or {})
    contract = _archive_bound_best_acquisition_contract(export)
    if not contract:
        return export
    candidate_archive = _mapping_or_empty(contract.get("candidate_archive"))
    file_custody = _mapping_or_empty(contract.get("archive_file_custody"))
    identity = _mapping_or_empty(contract.get("contract_identity"))
    out = dict(export)
    out.setdefault("family", _family_key(str(contract.get("family_id") or "")))
    out.setdefault(
        "candidate_id",
        contract.get("candidate_chain_id") or identity.get("candidate_chain_id"),
    )
    out.setdefault(
        "archive_path",
        candidate_archive.get("path") or file_custody.get("path"),
    )
    out.setdefault(
        "archive_bytes",
        candidate_archive.get("bytes") or file_custody.get("bytes"),
    )
    out.setdefault(
        "archive_sha256",
        candidate_archive.get("sha256") or file_custody.get("sha256"),
    )
    out.setdefault(
        "receiver_proof_path",
        identity.get("runtime_consumption_proof_path")
        or contract.get("runtime_consumption_proof_path"),
    )
    out.setdefault(
        "runtime_consumption_proof_ready",
        bool(contract.get("receiver_contract_satisfied"))
        or bool(file_custody.get("custody_complete")),
    )
    out.setdefault(
        "receiver_contract_satisfied",
        bool(contract.get("receiver_contract_satisfied")),
    )
    out.setdefault("archive_bound_adapter_package_normalized", True)
    return out


def _archive_bound_best_acquisition_contract(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    surfaces = _path_get(
        report,
        (
            "archive_bound_candidate_adapter_package",
            "archive_bound_candidate_contract_surfaces",
        ),
    )
    if not isinstance(surfaces, Sequence) or isinstance(surfaces, (str, bytes)):
        return {}
    for surface in surfaces:
        if not isinstance(surface, Mapping):
            continue
        contract = surface.get("best_acquisition_contract")
        if isinstance(contract, Mapping):
            return dict(contract)
    return {}


def _full_video_mlx_response_training_control(
    *,
    family: str,
    score: float | None,
    nonrate_score: float | None,
    avg_seg: float | None,
    avg_pose: float | None,
    archive_under_ceiling: bool | None,
    full_video: bool,
    current_segnet_distillation_weight: float | None,
    current_pose_distillation_weight: float | None = None,
    max_mlx_score_for_local_replay: float | None = None,
) -> dict[str, Any]:
    family_key = _family_key(str(family or "nerv"))
    bad_score = bool(
        score is not None
        and max_mlx_score_for_local_replay is not None
        and float(score) > float(max_mlx_score_for_local_replay)
    )
    seg_failure = bool(
        avg_seg is not None and float(avg_seg) >= _FULL_VIDEO_RESPONSE_BAD_SEG_THRESHOLD
    )
    pose_failure = bool(
        avg_pose is not None
        and float(avg_pose) >= _FULL_VIDEO_RESPONSE_BAD_POSE_THRESHOLD
    )
    recommended_seg_weight = (
        recommend_distillation_weight_for_full_video_fit_failure(
            current_segnet_distillation_weight,
            observed_component=avg_seg,
            failure_threshold=_FULL_VIDEO_RESPONSE_BAD_SEG_THRESHOLD,
        )
        if seg_failure and family_key in {"hi_nerv", "snerv"}
        else None
    )
    recommended_pose_weight = (
        recommend_distillation_weight_for_full_video_fit_failure(
            current_pose_distillation_weight,
            observed_component=avg_pose,
            failure_threshold=_FULL_VIDEO_RESPONSE_BAD_POSE_THRESHOLD,
        )
        if pose_failure and family_key in {"hi_nerv", "snerv"}
        else None
    )
    mutations: list[str] = []
    if archive_under_ceiling is False:
        mutations.extend(
            [
                f"treat_previous_{family_key}_run_as_rate_failure_not_distortion_negative",
                (
                    "switch_snerv_representation_before_more_same_modelsize_training"
                    if family_key == "snerv"
                    else "select_smaller_or_more_entropy_friendly_hinerv_modelsize_candidate"
                ),
            ]
        )
    if bad_score and archive_under_ceiling is not False:
        mutations.append(
            f"treat_previous_{family_key}_run_as_fit_failure_not_rate_negative"
        )
    if seg_failure and recommended_seg_weight is not None:
        mutations.extend(
            [
                "increase_segnet_distillation_weight_from_full_video_mlx_response",
                "preserve_pose_guard_while_raising_segnet_pressure",
            ]
        )
    if pose_failure:
        mutations.extend(
            [
                "increase_pose_distillation_weight_from_full_video_mlx_response",
                "build_xray_hardpair_hitlist_from_full_video_pose_tail",
                "launch_hard_pair_prioritized_sampler_successor",
                "preserve_random_full_video_fill_when_prioritizing_hard_pairs",
            ]
        )
    if not full_video:
        action = "wait_for_full600_mlx_scorer_response"
        reason = "partial_mlx_scorer_response_not_enough_for_training_control"
        should_stop = False
        successor_required = False
    elif archive_under_ceiling is False:
        action = "checkpoint_then_stop_same_representation_rate_over_cap"
        reason = "full_video_archive_bytes_exceed_hard_byte_ceiling"
        should_stop = True
        successor_required = True
    elif bad_score:
        action = "checkpoint_then_supersede_with_full_video_fit_mutation"
        reason = "full_video_mlx_response_distortion_above_local_replay_gate"
        should_stop = True
        successor_required = True
    else:
        action = "eligible_for_local_cpu_replay_gate"
        reason = "full_video_mlx_response_passes_local_replay_prefilter"
        should_stop = False
        successor_required = False
    return {
        "schema": "nerv_full_video_mlx_response_training_control.v1",
        "family": family_key,
        "action": action,
        "reason": reason,
        "should_stop_current_run": should_stop,
        "successor_required": successor_required,
        "max_mlx_score_for_local_replay": max_mlx_score_for_local_replay,
        "score_recomputed_from_components": score,
        "nonrate_score_estimate": nonrate_score,
        "avg_segnet_dist": avg_seg,
        "avg_posenet_dist": avg_pose,
        "archive_under_hard_byte_ceiling": archive_under_ceiling,
        "segnet_fit_failure_detected": seg_failure,
        "pose_fit_failure_detected": pose_failure,
        "observed_segnet_distillation_weight": current_segnet_distillation_weight,
        "observed_pose_distillation_weight": current_pose_distillation_weight,
        "recommended_segnet_distillation_weight": recommended_seg_weight,
        "recommended_segnet_distillation_weight_multiplier": (
            _fit_failure_weight_multiplier(
                current_segnet_distillation_weight,
                recommended_seg_weight,
            )
            if recommended_seg_weight is not None
            else None
        ),
        "recommended_pose_distillation_weight": recommended_pose_weight,
        "recommended_pose_distillation_weight_multiplier": (
            _fit_failure_weight_multiplier(
                current_pose_distillation_weight,
                recommended_pose_weight,
            )
            if recommended_pose_weight is not None
            else None
        ),
        "recommended_launch_mutations": _dedupe_strings(mutations),
        **FALSE_AUTHORITY,
    }


def _infer_full_video_feedback_segnet_distillation_weight(
    *,
    export: Mapping[str, Any],
    explicit_weight: float | None,
) -> tuple[float | None, str | None]:
    explicit = _float_or_none(explicit_weight)
    if explicit is not None:
        return explicit, "harvest_current_segnet_distillation_weight"
    export_paths = (
        ("command_args", "segnet_distillation_weight"),
        ("campaign_identity", "argv", "segnet_distillation_weight"),
        ("startup_json", "campaign_identity", "argv", "segnet_distillation_weight"),
        ("runner_startup_json", "campaign_identity", "argv", "segnet_distillation_weight"),
        ("score_aware_training", "segnet_distillation_weight"),
        ("training_config", "segnet_distillation_weight"),
        ("config", "segnet_distillation_weight"),
    )
    for path in export_paths:
        value = _float_or_none(_path_get(export, path))
        if value is not None:
            return value, ".".join(path)
    return None, None


def _infer_full_video_feedback_pose_distillation_weight(
    *,
    export: Mapping[str, Any],
    explicit_weight: float | None,
) -> tuple[float | None, str | None]:
    explicit = _float_or_none(explicit_weight)
    if explicit is not None:
        return explicit, "harvest_current_pose_distillation_weight"
    export_paths = (
        ("command_args", "pose_distillation_weight"),
        ("campaign_identity", "argv", "pose_distillation_weight"),
        ("startup_json", "campaign_identity", "argv", "pose_distillation_weight"),
        ("runner_startup_json", "campaign_identity", "argv", "pose_distillation_weight"),
        ("score_aware_training", "pose_distillation_weight"),
        ("training_config", "pose_distillation_weight"),
        ("config", "pose_distillation_weight"),
    )
    for path in export_paths:
        value = _float_or_none(_path_get(export, path))
        if value is not None:
            return value, ".".join(path)
    return None, None


def _min_positive_int(value: Any) -> int | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        candidates = [_int_or_none(item) for item in value]
        positives = [int(item) for item in candidates if item is not None and item > 0]
        return min(positives) if positives else None
    number = _int_or_none(value)
    return number if number is not None and number > 0 else None


def _minus_or_none(left: Any, right: Any) -> int | None:
    left_i = _int_or_none(left)
    right_i = _int_or_none(right)
    if left_i is None or right_i is None:
        return None
    return int(left_i) - int(right_i)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    import math

    return number if math.isfinite(number) else None


def _effective_distillation_weight(loss_components: Any) -> float | None:
    if not isinstance(loss_components, Mapping):
        return None
    unweighted = _float_or_none(loss_components.get("loss_part_distill"))
    weighted = _float_or_none(loss_components.get("loss_part_weighted_distill"))
    if unweighted is None or weighted is None or unweighted <= 1.0e-12:
        return None
    return weighted / unweighted


def recommend_segnet_distillation_weight_for_stagnation(
    observed_weight: float | None,
) -> float | None:
    """Return the next bounded SegNet pressure for a stagnant HiNeRV run."""

    observed = _float_or_none(observed_weight)
    if observed is None or observed <= 0.0:
        return float(_SEG_STAGNATION_WEIGHT_MULTIPLIER)
    if observed >= float(_SEG_STAGNATION_MAX_DISTILLATION_WEIGHT):
        return None
    next_weight = max(
        float(_SEG_STAGNATION_WEIGHT_MULTIPLIER),
        observed * float(_SEG_STAGNATION_WEIGHT_MULTIPLIER),
    )
    bounded = min(next_weight, float(_SEG_STAGNATION_MAX_DISTILLATION_WEIGHT))
    return bounded if bounded > observed else None


def recommend_distillation_weight_for_full_video_fit_failure(
    observed_weight: float | None,
    *,
    observed_component: float | None,
    failure_threshold: float,
) -> float | None:
    """Return a bounded scorer pressure from full-video component severity."""

    observed = _float_or_none(observed_weight)
    component = _float_or_none(observed_component)
    threshold = _float_or_none(failure_threshold)
    if component is None or threshold is None or component < threshold or threshold <= 0.0:
        return None
    base = observed if observed is not None and observed > 0.0 else 1.0
    severity = math.sqrt(max(float(component) / float(threshold), 1.0))
    multiplier = min(
        max(severity, float(_FULL_VIDEO_FIT_WEIGHT_MIN_MULTIPLIER)),
        float(_FULL_VIDEO_FIT_WEIGHT_MAX_DISTILLATION_WEIGHT),
    )
    bounded = min(
        base * multiplier,
        float(_FULL_VIDEO_FIT_WEIGHT_MAX_DISTILLATION_WEIGHT),
    )
    return bounded if bounded > base else None


def _fit_failure_weight_multiplier(
    observed_weight: float | None,
    recommended_weight: float | None,
) -> float | None:
    recommended = _float_or_none(recommended_weight)
    if recommended is None:
        return None
    observed = _float_or_none(observed_weight)
    base = observed if observed is not None and observed > 0.0 else 1.0
    return float(recommended) / float(base)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    clipped = min(max(float(q), 0.0), 1.0)
    if len(ordered) == 1:
        return ordered[0]
    position = clipped * float(len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return lower + (upper - lower) * (position - float(lower_index))


def _relative_improvement(
    first_value: float | None,
    last_value: float | None,
) -> float | None:
    if first_value is None or last_value is None:
        return None
    if not (math.isfinite(first_value) and math.isfinite(last_value)):
        return None
    if first_value <= 0.0:
        return None
    return (first_value - last_value) / first_value


def _path_get(source: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = source
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _family_key(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_")
    if text in {"hinerv", "hi_nerv_mlx"}:
        return "hi_nerv"
    return text


def _training_telemetry_blocker(family_key: str, suffix: str) -> str:
    if family_key == "hi_nerv":
        if suffix == "trained_archive_byte_oracle_feedback_missing":
            return "hinerv_trained_archive_byte_oracle_feedback_missing"
        return f"hi_nerv_{suffix}"
    return f"{family_key}_{suffix}"


def _training_telemetry_mutations_for_family(
    family_key: str,
    mutations: Sequence[Any],
) -> list[str]:
    family_slug = "hi_nerv" if family_key == "hi_nerv" else family_key
    return [
        str(mutation).replace(
            "treat_previous_hi_nerv_run_as",
            f"treat_previous_{family_slug}_run_as",
        )
        for mutation in mutations
    ]


__all__ = [
    "FULL_VIDEO_MLX_SCORER_FEEDBACK_SCHEMA",
    "HINERV_ARCHIVE_LADDER_FEEDBACK_SCHEMA",
    "LEDGER_SCHEMA",
    "REFRESH_SCHEMA",
    "SAMPLE_GENERALIZATION_GATE_SCHEMA",
    "SCHEMA",
    "TELEMETRY_FEEDBACK_SCHEMA",
    "build_hinerv_archive_ladder_feedback_report",
    "build_nerv_candidate_feedback_row",
    "build_nerv_full_video_mlx_scorer_feedback_row",
    "build_nerv_training_telemetry_feedback_row",
    "recommend_distillation_weight_for_full_video_fit_failure",
    "recommend_segnet_distillation_weight_for_stagnation",
    "refresh_nerv_candidate_feedback_report",
    "write_nerv_candidate_feedback_files",
    "write_nerv_full_video_mlx_scorer_feedback_files",
    "write_nerv_training_telemetry_feedback_files",
    "write_refreshed_nerv_candidate_feedback_files",
]
