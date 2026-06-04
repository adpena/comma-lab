# SPDX-License-Identifier: MIT
"""Candidate-conditioned curriculum plans for NeRV-family campaigns.

Model-size candidates choose capacity and nominal byte budgets. This module
binds those candidates to the training pressure required to make the capacity
meaningful: PR95-style stages, real scorer teachers, coder-aware QAT, and
measured archive-byte feedback. It is planner/runner metadata, not score
authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tac.analysis.nerv_modelsize_budget import RATE_SCORE_PER_BYTE
from tac.analysis.pr95_stack_binding_requirements import (
    build_pr95_long_campaign_prelaunch_gate,
    build_pr95_stack_binding_evidence,
    build_pr95_stack_binding_requirements,
)
from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
    CANONICAL_PR95_TOTAL_EPOCHS,
)
from tac.substrates.hprc.resolution_contract import CONTEST_PAIR_COUNT
from tac.substrates.snerv_inverse_steg_carrier.mlx_native_adapter_contract import (
    build_snerv_mlx_native_adapter_contract,
    build_snerv_mlx_native_file_backed_evidence,
)

SCHEMA = "nerv_candidate_curriculum_plan.v1"
BYTE_FEEDBACK_SCHEMA = "nerv_candidate_byte_feedback.v1"
SNERV_OFFICIAL_MFU_HFR_TUB_ADAPTER = "snerv_official_mfu_hfr_tub_numeric_primitives_v1"
NERV_SCORER_INPUT_HEALTH_GATE_SCHEMA = "nerv_local_scorer_input_health_gate.v1"
SNERV_SKIP_HIGH_EXPORT_ADMISSION_GATE_SCHEMA = (
    "snerv_skip_high_export_admission_gate.v1"
)

FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
AUTHORITY_FIELD_NAMES = frozenset(FALSE_AUTHORITY)


def _codec_bits(codec: Any, *, default: int = 8) -> int:
    text = str(codec or "").strip().lower()
    if text.startswith("int2"):
        return 2
    if text.startswith("int4") or "mixed_magnitude" in text:
        return 4
    if text.startswith("int8") or text in {"portfolio_auto", "auto"}:
        return 8
    if text.startswith("fp16"):
        return 16
    if text.startswith("float32") or text.startswith("fp32"):
        return 32
    return int(default)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _artifact_mappings(root: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """Return shallow metadata maps that exporter variants use for custody fields."""

    if not isinstance(root, Mapping):
        return []
    maps: list[Mapping[str, Any]] = [root]
    for key in (
        "packet_metadata_summary",
        "selected_archive_metadata",
        "selected_official_authority",
        "official_primitive_binding",
        "official_receiver_tensor_map_custody",
        "official_mfu_hfr_tub_source_forward_replay",
        "official_checkpoint_export_binding",
        "local_mlx_prefilter_profile",
        "receiver_value_domain_xray",
        "snerv_receiver_value_domain_xray",
    ):
        child = root.get(key)
        if isinstance(child, Mapping):
            maps.append(child)
    score_training = root.get("score_aware_long_training")
    if isinstance(score_training, Mapping):
        official_replay = score_training.get(
            "official_mfu_hfr_tub_source_forward_replay"
        )
        if isinstance(official_replay, Mapping):
            maps.append(official_replay)
    return maps


def _has_prefix(value: str, prefixes: tuple[str, ...]) -> bool:
    return any(value.startswith(prefix) for prefix in prefixes)


def _any_true(maps: list[Mapping[str, Any]], *keys: str) -> bool:
    return any(row.get(key) is True for row in maps for key in keys)


def _string_field(maps: list[Mapping[str, Any]], *keys: str) -> str:
    for row in maps:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def _collect_blockers(
    maps: list[Mapping[str, Any]],
    *keys: str,
    official_only: bool = False,
) -> list[str]:
    values: list[str] = []
    for row in maps:
        for key in keys:
            raw = row.get(key)
            if isinstance(raw, str):
                items = [raw]
            elif isinstance(raw, (list, tuple, set)):
                items = list(raw)
            else:
                continue
            for item in items:
                text = str(item)
                if not text:
                    continue
                if official_only and not (
                    "snerv_official" in text
                    or "source_forward" in text
                    or "source_parity" in text
                    or "tub" in text
                    or "mfu_hfr" in text
                ):
                    continue
                values.append(text.removeprefix("source_parity:"))
    return _dedupe(values)


def _collect_prefixed_blockers(
    maps: list[Mapping[str, Any]],
    *,
    prefixes: tuple[str, ...],
) -> list[str]:
    values: list[str] = []
    for row in maps:
        raw = row.get("blockers")
        if isinstance(raw, str):
            items = [raw]
        elif isinstance(raw, (list, tuple, set)):
            items = list(raw)
        else:
            items = []
        for item in items:
            text = str(item)
            if text and _has_prefix(text, prefixes):
                values.append(text)
    return _dedupe(values)


def _nested_mapping(root: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Any = root
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def _build_nerv_local_scorer_input_health_gate(
    artifact_evidence: Mapping[str, Any] | None,
    *,
    family_token: str,
    require_profile: bool,
) -> dict[str, Any]:
    """Summarize whether local scorer inputs are in-distribution enough to replay."""

    family = str(family_token or "nerv").strip() or "nerv"
    artifact = _as_mapping(artifact_evidence)
    maps = _artifact_mappings(artifact)
    profile = _as_mapping(artifact.get("local_mlx_prefilter_profile"))
    distribution = _as_mapping(profile.get("scorer_input_distribution"))
    if not distribution:
        for row in maps:
            distribution = _as_mapping(row.get("scorer_input_distribution"))
            if distribution:
                break
    if not distribution:
        for row in maps:
            distribution = _as_mapping(row.get("profile_scorer_input_summary"))
            if distribution:
                break
    blockers = _collect_prefixed_blockers(
        maps,
        prefixes=(
            "scorer_input_",
            "snerv_profile_segnet_",
            "snerv_profile_posenet_",
            "skip_high_prefilter_scorer_input_",
        ),
    )
    profile_present = bool(profile or distribution)
    gate_blockers = list(blockers)
    if require_profile and not profile_present:
        gate_blockers.append(f"{family}_local_scorer_input_profile_missing")
    if require_profile and profile_present and not distribution:
        gate_blockers.append(f"{family}_local_scorer_input_distribution_missing")
    if gate_blockers:
        gate_blockers.append(f"{family}_local_scorer_input_health_gate_failed")
    return {
        "schema": NERV_SCORER_INPUT_HEALTH_GATE_SCHEMA,
        "family": family,
        "profile_required": bool(require_profile),
        "profile_present": profile_present,
        "profile_path": artifact.get("local_mlx_prefilter_profile_path")
        or profile.get("profile_path"),
        "scorer_input_distribution_present": bool(distribution),
        "local_replay_admissible": bool(
            profile_present and distribution and not gate_blockers
        ),
        "blockers": _dedupe(gate_blockers),
        **FALSE_AUTHORITY,
    }


def _build_snerv_skip_high_export_admission_gate(
    *,
    candidate: Mapping[str, Any],
    artifact_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fail closed when byte-attractive skip_high storage collapses value domain."""

    artifact = _as_mapping(artifact_evidence)
    maps = _artifact_mappings(artifact)
    mode = str(
        candidate.get("official_skip_high_mode")
        or candidate.get("snerv_official_skip_high_mode")
        or ""
    ).strip().lower()
    storage: Mapping[str, Any] = {}
    for row in maps:
        storage = _nested_mapping(row, "decoder_payload_header", "skip_high_storage")
        if storage:
            break
        storage = _as_mapping(row.get("skip_high_storage"))
        if storage:
            break
    stored_shape = [int(v) for v in storage.get("stored_shape") or []]
    codec = str(storage.get("codec") or "").strip().lower()
    blocker_prefixes = (
        "snerv_official_skip_high_",
        "no_skip_high_",
        "skip_high_",
    )
    blockers = _collect_prefixed_blockers(maps, prefixes=blocker_prefixes)
    scalar_mode = mode == "scalar_mean"
    scalar_storage = bool(
        codec.startswith("scalar_mean") or stored_shape == [1, 1, 1, 1]
    )
    collapse_blocker_present = any(
        "scalar_mean_receiver_expand_collapse_risk" in blocker
        for blocker in blockers
    )
    if scalar_mode:
        blockers.append(
            "snerv_official_skip_high_scalar_mean_requires_value_domain_xray_noncollapse"
        )
    if scalar_storage or collapse_blocker_present:
        blockers.append(
            "snerv_official_skip_high_scalar_mean_receiver_expand_collapse_risk"
        )
    scalar_collapse_risk = bool(
        scalar_mode or scalar_storage or collapse_blocker_present
    )
    return {
        "schema": SNERV_SKIP_HIGH_EXPORT_ADMISSION_GATE_SCHEMA,
        "official_skip_high_mode": mode or None,
        "skip_high_storage_present": bool(storage),
        "skip_high_storage_codec": storage.get("codec"),
        "skip_high_storage_shape": stored_shape,
        "scalar_collapse_risk": scalar_collapse_risk,
        "local_training_allowed": True,
        "exact_eval_admissible": bool(not scalar_collapse_risk and not blockers),
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }


def _build_snerv_official_source_forward_authority_split(
    *,
    candidate: Mapping[str, Any],
    artifact_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep official receiver custody separate from source-forward authority."""

    artifact = _as_mapping(artifact_evidence)
    maps = _artifact_mappings(artifact)
    adapter = str(candidate.get("snerv_model_size_adapter") or "")
    official_requested = bool(
        adapter == SNERV_OFFICIAL_MFU_HFR_TUB_ADAPTER
        or _any_true(
            maps,
            "snerv_official_mfu_hfr_tub_numeric_primitives_requested",
            "official_mfu_hfr_tub_numeric_primitives_requested",
            "official_mfu_hfr_tub_primitives_present",
        )
    )
    if not official_requested:
        return {
            "schema": "snerv_official_source_forward_authority_split.v1",
            "official_adapter_requested": False,
            "full_source_forward_authority_proven": False,
            "blockers": [],
            **FALSE_AUTHORITY,
        }

    export_bound = _any_true(
        maps,
        "snerv_official_mfu_hfr_tub_export_bound",
        "official_mfu_hfr_tub_export_bound",
    )
    receiver_payload_bound = _any_true(
        maps,
        "snerv_official_mfu_hfr_tub_receiver_payload_bound",
        "official_mfu_hfr_tub_receiver_payload_bound",
        "receiver_payload_bound",
        "trained_receiver_payload_export_bound",
    )
    frame_producing_export = _any_true(
        maps,
        "snerv_official_mfu_hfr_tub_frame_producing_export",
        "official_mfu_hfr_tub_frame_producing_export",
    )
    source_forward_bound = _any_true(
        maps,
        "snerv_official_mfu_hfr_tub_source_forward_replay_bound",
        "official_mfu_hfr_tub_source_forward_replay_bound",
        "source_forward_replay_bound",
        "source_forward_replay_bound_by_export",
        "receiver_source_forward_replay_bound",
    )
    source_forward_verified = _any_true(
        maps,
        "source_forward_replay_verified",
        "source_forward_replay_verified_by_export",
        "source_forward_parity_proven",
        "full_tub_source_forward_parity_proven",
        "full_stack_source_forward_replay_proven",
    )
    source_forward_authority = _any_true(
        maps,
        "snerv_official_mfu_hfr_tub_source_forward_replay_authority",
        "official_mfu_hfr_tub_source_forward_replay_authority",
        "source_forward_replay_authority",
    )
    source_faithful_stack = _any_true(maps, "source_faithful_stack")
    export_semantics = _string_field(
        maps,
        "snerv_official_mfu_hfr_tub_export_bound_semantics",
        "official_mfu_hfr_tub_export_bound_semantics",
        "official_parity_status",
    )
    official_blockers = _collect_blockers(
        maps,
        "snerv_official_mfu_hfr_tub_export_blockers",
        "official_source_parity_blockers",
        "source_forward_blockers",
        "blockers",
        "required_blockers",
        "nonblocking_gaps",
        official_only=True,
    )
    blockers: list[str] = []
    if not export_bound:
        blockers.append("snerv_official_mfu_hfr_tub_export_not_bound")
    if not receiver_payload_bound:
        blockers.append("snerv_official_mfu_hfr_tub_receiver_payload_not_bound")
    if not frame_producing_export:
        blockers.append("snerv_official_mfu_hfr_tub_frame_producing_export_missing")
    source_authority_ready = bool(
        source_forward_bound
        and source_forward_verified
        and source_forward_authority
        and source_faithful_stack
    )
    if not source_authority_ready:
        blockers.append(
            "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        )
        if not official_blockers:
            official_blockers.append(
                "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing"
            )
    full_source_forward_authority_proven = bool(
        official_requested
        and export_bound
        and receiver_payload_bound
        and frame_producing_export
        and source_authority_ready
        and not official_blockers
    )
    if full_source_forward_authority_proven:
        launch_semantics = (
            "official_source_forward_parity_available_false_authority_until_score_gate"
        )
    elif receiver_payload_bound:
        launch_semantics = (
            "receiver_bound_training_allowed_but_official_source_authority_false"
        )
    else:
        launch_semantics = "official_training_waits_on_receiver_payload_binding"
    return {
        "schema": "snerv_official_source_forward_authority_split.v1",
        "official_adapter_requested": True,
        "candidate_adapter": adapter,
        "export_bound": export_bound,
        "receiver_payload_bound": receiver_payload_bound,
        "frame_producing_export": frame_producing_export,
        "source_forward_replay_bound": source_forward_bound,
        "source_forward_replay_verified": source_forward_verified,
        "source_forward_replay_authority": source_forward_authority,
        "source_faithful_stack": source_faithful_stack,
        "export_bound_semantics": export_semantics,
        "receiver_payload_bound_is_byte_runtime_custody_only": True,
        "source_forward_authority_required_for_pr95_faithful_claim": True,
        "receiver_bound_training_evidence_usable": bool(receiver_payload_bound),
        "full_source_forward_authority_proven": full_source_forward_authority_proven,
        "official_blockers": _dedupe(official_blockers),
        "launch_semantics": launch_semantics,
        "blockers": _dedupe([*blockers, *official_blockers]),
        **FALSE_AUTHORITY,
    }


def strip_candidate_curriculum_authority_fields(value: Any) -> Any:
    """Return a metadata-safe copy without canonical authority/readiness keys."""

    if isinstance(value, Mapping):
        return {
            str(key): strip_candidate_curriculum_authority_fields(item)
            for key, item in value.items()
            if str(key) not in AUTHORITY_FIELD_NAMES
        }
    if isinstance(value, list):
        return [strip_candidate_curriculum_authority_fields(item) for item in value]
    if isinstance(value, tuple):
        return [strip_candidate_curriculum_authority_fields(item) for item in value]
    return value


def _base_byte_feedback(
    *,
    candidate: Mapping[str, Any],
    measured_num_pairs: int,
    measured_payload_bytes: int | None = None,
    measured_archive_bytes: int | None = None,
    archive_minus_nominal_bytes: int | None = None,
    archive_to_nominal_ratio: float | None = None,
    calibrated_archive_overrun_bytes: int | None = None,
    required_nominal_payload_bytes_max: int | None = None,
    hard_byte_ceiling_measurement_bypass_enabled: bool | None = None,
    hard_byte_ceiling_checked_after_export: bool | None = None,
) -> dict[str, Any]:
    nominal = _int(candidate.get("nominal_total_payload_bytes"))
    measured = measured_archive_bytes if measured_archive_bytes is not None else measured_payload_bytes
    delta = (
        archive_minus_nominal_bytes
        if archive_minus_nominal_bytes is not None
        else (None if measured is None else int(measured) - int(nominal))
    )
    hard_byte_ceiling = _int(candidate.get("hard_byte_ceiling"))
    charged_archive_bytes = (
        None if measured_archive_bytes is None else int(measured_archive_bytes)
    )
    archive_under_hard_byte_ceiling = (
        None
        if charged_archive_bytes is None or hard_byte_ceiling <= 0
        else bool(charged_archive_bytes <= hard_byte_ceiling)
    )
    archive_over_hard_byte_ceiling_bytes = (
        None
        if charged_archive_bytes is None or hard_byte_ceiling <= 0
        else max(0, charged_archive_bytes - hard_byte_ceiling)
    )
    candidate_num_pairs = _int(candidate.get("num_pairs"))
    measured_pairs = int(measured_num_pairs)
    scope_matches = candidate_num_pairs > 0 and measured_pairs == candidate_num_pairs
    return {
        "schema": BYTE_FEEDBACK_SCHEMA,
        "candidate_id": candidate.get("candidate_id"),
        "candidate_num_pairs": candidate_num_pairs,
        "measured_num_pairs": measured_pairs,
        "feedback_scope": (
            "candidate_full_scope" if scope_matches else "partial_pair_advisory"
        ),
        "scope_matches_candidate": scope_matches,
        "hard_byte_ceiling": hard_byte_ceiling,
        "nominal_total_payload_bytes": nominal,
        "measured_payload_bytes": (
            None if measured_payload_bytes is None else int(measured_payload_bytes)
        ),
        "measured_archive_bytes": charged_archive_bytes,
        "measured_minus_nominal_bytes": delta,
        "archive_to_nominal_ratio": (
            float(archive_to_nominal_ratio)
            if archive_to_nominal_ratio is not None
            else (
                None
                if measured is None or nominal <= 0
                else float(measured) / float(nominal)
            )
        ),
        "calibrated_archive_overrun_bytes": (
            None
            if calibrated_archive_overrun_bytes is None
            else int(calibrated_archive_overrun_bytes)
        ),
        "required_nominal_payload_bytes_max": (
            None
            if required_nominal_payload_bytes_max is None
            else int(required_nominal_payload_bytes_max)
        ),
        "hard_byte_ceiling_measurement_bypass_enabled": (
            None
            if hard_byte_ceiling_measurement_bypass_enabled is None
            else bool(hard_byte_ceiling_measurement_bypass_enabled)
        ),
        "hard_byte_ceiling_checked_after_export": (
            None
            if hard_byte_ceiling_checked_after_export is None
            else bool(hard_byte_ceiling_checked_after_export)
        ),
        "measured_minus_nominal_rate_score_delta": (
            None if delta is None else float(delta * RATE_SCORE_PER_BYTE)
        ),
        "archive_under_hard_byte_ceiling": archive_under_hard_byte_ceiling,
        "archive_over_hard_byte_ceiling_bytes": archive_over_hard_byte_ceiling_bytes,
        "rate_axis_feedback_verdict": (
            "archive_bytes_not_measured"
            if charged_archive_bytes is None
            else (
                "hard_byte_ceiling_missing"
                if hard_byte_ceiling <= 0
                else (
                    "receiver_proven_archive_under_hard_byte_ceiling"
                    if archive_under_hard_byte_ceiling
                    else "receiver_proven_archive_over_hard_byte_ceiling"
                )
            )
        ),
        "feedback_ready": measured is not None and scope_matches,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def build_hinerv_candidate_curriculum_plan(
    *,
    candidate: Mapping[str, Any] | None,
    requested_epochs: int,
    num_pairs: int,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
    coder_aware_qat: bool,
    coder_qat_quant_bits: int,
    recon_pixel_weight_attached: bool,
    eval_roundtrip_ste_attached: bool = False,
    scorer_input_distribution_guard_attached: bool = False,
    differentiable_pose_preprocess_attached: bool = False,
    ema_archive_selection_attached: bool = False,
    pr95_staged_curriculum_bound: bool | None = None,
    muon_adamw_partition_bound: bool | None = None,
    receiver_proof_attached: bool = False,
    full_video_local_prefilter_attached: bool = False,
    local_cpu_replay_gate_attached: bool = False,
    measured_archive_bytes: int | None = None,
    measured_num_pairs: int | None = None,
    archive_minus_nominal_bytes: int | None = None,
    archive_to_nominal_ratio: float | None = None,
    calibrated_archive_overrun_bytes: int | None = None,
    required_nominal_payload_bytes_max: int | None = None,
    hard_byte_ceiling_measurement_bypass_enabled: bool | None = None,
    hard_byte_ceiling_checked_after_export: bool | None = None,
    native_mlx_artifact_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a HiNeRV modelsize candidate to its required training pressure."""

    candidate_row = dict(candidate or {})
    candidate_selected = bool(candidate_row)
    codec = candidate_row.get("decoder_codec", "manual_cli")
    codec_bits = _codec_bits(codec, default=int(coder_qat_quant_bits))
    q_bits = min(max(1, int(coder_qat_quant_bits)), codec_bits)
    if candidate_selected:
        q_bits = codec_bits
    effective_coder_regularizer = bool(coder_aware_qat or candidate_selected)
    epochs = max(0, int(requested_epochs))
    full_video = int(num_pairs) >= 600
    pr95_staged_bound = (
        epochs >= 8
        if pr95_staged_curriculum_bound is None
        else bool(pr95_staged_curriculum_bound)
    )
    muon_partition_bound = (
        epochs >= 8
        if muon_adamw_partition_bound is None
        else bool(muon_adamw_partition_bound)
    )
    blockers: list[str] = []
    launch_mutations: list[str] = []
    if candidate_selected and not coder_aware_qat:
        launch_mutations.append(
            "enabled_decoder_coder_regularizer_from_modelsize_candidate"
        )
    if candidate_selected and int(coder_qat_quant_bits) != q_bits:
        launch_mutations.append("aligned_coder_qat_quant_bits_to_candidate_codec")
    if epochs < 8:
        blockers.append("hinerv_candidate_curriculum_requires_min_8_epochs")
    if _num(segnet_distillation_weight) <= 0.0:
        blockers.append("hinerv_candidate_curriculum_requires_real_segnet_teacher")
    if _num(pose_distillation_weight) <= 0.0:
        blockers.append("hinerv_candidate_curriculum_requires_real_posenet_teacher")
    if not recon_pixel_weight_attached:
        blockers.append("hinerv_candidate_curriculum_recon_pixel_weight_missing")
    if not full_video:
        blockers.append("hinerv_candidate_curriculum_full600_required_for_promotion")
    if not candidate_selected:
        blockers.append("hinerv_modelsize_candidate_not_selected_manual_probe")
    serious_full_video_candidate = bool(candidate_selected and full_video and epochs >= 8)
    scorer_input_health_gate = _build_nerv_local_scorer_input_health_gate(
        native_mlx_artifact_evidence,
        family_token="hinerv",
        require_profile=bool(
            serious_full_video_candidate
            and (
                scorer_input_distribution_guard_attached
                or full_video_local_prefilter_attached
            )
        ),
    )
    blockers.extend(scorer_input_health_gate.get("blockers") or [])
    scorer_input_distribution_guard_verified = bool(
        scorer_input_distribution_guard_attached
        and (
            not scorer_input_health_gate["profile_required"]
            or scorer_input_health_gate["local_replay_admissible"]
        )
    )
    byte_feedback = _base_byte_feedback(
        candidate=candidate_row,
        measured_num_pairs=(
            int(num_pairs) if measured_num_pairs is None else int(measured_num_pairs)
        ),
        measured_archive_bytes=measured_archive_bytes,
        archive_minus_nominal_bytes=archive_minus_nominal_bytes,
        archive_to_nominal_ratio=archive_to_nominal_ratio,
        calibrated_archive_overrun_bytes=calibrated_archive_overrun_bytes,
        required_nominal_payload_bytes_max=required_nominal_payload_bytes_max,
        hard_byte_ceiling_measurement_bypass_enabled=(
            hard_byte_ceiling_measurement_bypass_enabled
        ),
        hard_byte_ceiling_checked_after_export=hard_byte_ceiling_checked_after_export,
    ) if candidate_selected else {
        "schema": BYTE_FEEDBACK_SCHEMA,
        "candidate_id": None,
        "candidate_num_pairs": None,
        "measured_num_pairs": int(num_pairs),
        "feedback_scope": "manual_cli_probe",
        "scope_matches_candidate": False,
        "feedback_ready": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if candidate_selected and measured_archive_bytes is not None and not byte_feedback["scope_matches_candidate"]:
        blockers.append("partial_pair_byte_feedback_only")
    elif candidate_selected and not byte_feedback["feedback_ready"]:
        blockers.append("hinerv_trained_archive_byte_oracle_feedback_missing")
    if (
        candidate_selected
        and byte_feedback.get("feedback_ready") is True
        and byte_feedback.get("archive_under_hard_byte_ceiling") is False
    ):
        blockers.append("hinerv_receiver_proven_archive_over_hard_byte_ceiling")
    pr95_binding = build_pr95_stack_binding_requirements(
        family="hi_nerv",
        evidence=build_pr95_stack_binding_evidence(
            modelsize_archive_budget=candidate_selected,
            pr95_staged_curriculum=pr95_staged_bound,
            real_segnet_teacher=_num(segnet_distillation_weight) > 0.0,
            real_posenet_teacher=_num(pose_distillation_weight) > 0.0,
            differentiable_pose_preprocess=bool(
                differentiable_pose_preprocess_attached
            ),
            eval_roundtrip_ste=bool(eval_roundtrip_ste_attached),
            scorer_input_distribution_guard=scorer_input_distribution_guard_verified,
            ema_archive_selection=bool(ema_archive_selection_attached),
            qat_forward=effective_coder_regularizer,
            coder_aware_regularizer=effective_coder_regularizer,
            muon_adamw_partition=muon_partition_bound,
            archive_in_loop_byte_oracle=bool(byte_feedback.get("feedback_ready")),
            byte_closed_archive_export=measured_archive_bytes is not None,
            receiver_proof=bool(receiver_proof_attached),
            full_video_local_prefilter=bool(full_video_local_prefilter_attached),
            local_cpu_replay_gate=bool(local_cpu_replay_gate_attached),
        ),
    )
    long_campaign_prelaunch_gate = build_pr95_long_campaign_prelaunch_gate(
        pr95_binding
    )
    blockers.extend(pr95_binding["blockers"])
    return {
        "schema": SCHEMA,
        "family": "hi_nerv",
        "candidate_conditioned": candidate_selected,
        "candidate_id": candidate_row.get("candidate_id"),
        "num_pairs": int(num_pairs),
        "campaign_scope": "full600" if full_video else "partial_pair_smoke",
        "pr95_stage_plan": {
            "enabled": pr95_staged_bound,
            "requested_epochs": epochs,
            "minimum_candidate_epochs": 8,
            "canonical_full_epochs": CANONICAL_PR95_TOTAL_EPOCHS,
            "stage_count": 8,
            "stage_policy": "scaled_pr95_8_stage_curriculum",
            "evidence_source": (
                "explicit_runner_optimizer_policy"
                if pr95_staged_curriculum_bound is not None
                else "legacy_epoch_floor"
            ),
            "muon_adamw_partition_bound": muon_partition_bound,
        },
        "scorer_pressure": {
            "segnet_distillation_weight": float(segnet_distillation_weight),
            "pose_distillation_weight": float(pose_distillation_weight),
            "joint_p18_p19_weight_attached": bool(recon_pixel_weight_attached),
            "eval_roundtrip_ste_attached": bool(eval_roundtrip_ste_attached),
            "scorer_input_distribution_guard_attached": bool(
                scorer_input_distribution_guard_attached
            ),
            "scorer_input_distribution_guard_verified": (
                scorer_input_distribution_guard_verified
            ),
            "scorer_input_health_gate": scorer_input_health_gate,
            "ema_archive_selection_attached": bool(
                ema_archive_selection_attached
            ),
            "receiver_proof_attached": bool(receiver_proof_attached),
            "differentiable_pose_preprocess_attached": bool(
                differentiable_pose_preprocess_attached
            ),
            "auto_teacher_weights_are_forbidden": True,
        },
        "coder_pressure": {
            "enabled": effective_coder_regularizer,
            "regularizer_enabled": effective_coder_regularizer,
            "fake_quant_forward_enabled": effective_coder_regularizer,
            "quant_bits": int(q_bits),
            "candidate_decoder_codec": str(codec),
            "candidate_decoder_codec_bits": int(codec_bits),
            "source": (
                "modelsize_candidate" if candidate_selected else "manual_cli_knobs"
            ),
            "implementation_status": (
                "decoder_weight_fake_quant_forward_plus_quant_residual_regularizer"
                if effective_coder_regularizer
                else "disabled"
            ),
        },
        "byte_oracle_logging": byte_feedback,
        "scorer_input_health_gate": scorer_input_health_gate,
        "pr95_stack_binding": pr95_binding,
        "long_campaign_prelaunch_gate": long_campaign_prelaunch_gate,
        "launch_mutations": launch_mutations,
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }


def build_snerv_candidate_curriculum_plan(
    *,
    candidate: Mapping[str, Any] | None,
    requested_epochs: int,
    num_pairs: int,
    step_map_coder_mode: str,
    measured_packet_bytes: int | None = None,
    measured_archive_bytes: int | None = None,
    archive_minus_nominal_bytes: int | None = None,
    archive_to_nominal_ratio: float | None = None,
    calibrated_archive_overrun_bytes: int | None = None,
    required_nominal_payload_bytes_max: int | None = None,
    hard_byte_ceiling_measurement_bypass_enabled: bool | None = None,
    hard_byte_ceiling_checked_after_export: bool | None = None,
    scorer_loop_qat_attached: bool = False,
    scorer_loop_qat_receiver_contract_satisfied: bool = False,
    scorer_loop_qat_ready_for_pose_guard_gate: bool = False,
    scorer_loop_qat_accepted_improvement: bool = False,
    receiver_proof_attached: bool = False,
    full_video_local_prefilter_attached: bool = False,
    local_cpu_replay_gate_attached: bool = False,
    native_mlx_train_export_attached: bool = False,
    native_mlx_long_training_bound: bool = False,
    native_mlx_receiver_proof_passed: bool = False,
    native_mlx_full600_campaign_ready: bool = False,
    native_mlx_scorer_loop_qat_attached: bool = False,
    native_mlx_scorer_loop_qat_receiver_contract_satisfied: bool = False,
    native_mlx_scorer_loop_qat_ready_for_pose_guard_gate: bool = False,
    native_mlx_scorer_loop_qat_accepted_improvement: bool = False,
    native_mlx_scorer_loop_qat_best_materialized: bool = False,
    native_mlx_real_segnet_teacher_bound: bool = False,
    native_mlx_real_posenet_teacher_bound: bool = False,
    native_mlx_pr95_curriculum_bound: bool = False,
    native_mlx_eval_roundtrip_ste_bound: bool = False,
    native_mlx_scorer_input_distribution_guard_bound: bool = False,
    native_mlx_differentiable_pose_preprocess_bound: bool = False,
    native_mlx_coder_qat_bound: bool = False,
    native_mlx_muon_adamw_partition_bound: bool = False,
    native_mlx_artifact_evidence: Mapping[str, Any] | None = None,
    measured_num_pairs: int | None = None,
) -> dict[str, Any]:
    """Bind a SNeRV receiver-grammar candidate to byte feedback and blockers."""

    candidate_row = dict(candidate or {})
    native_file_evidence = build_snerv_mlx_native_file_backed_evidence(
        native_mlx_artifact_evidence,
        required_num_pairs=CONTEST_PAIR_COUNT,
    )
    native_file_proof_passed = bool(
        native_file_evidence.get("required_pair_file_backed_export_proof_passed")
    )
    native_contract = build_snerv_mlx_native_adapter_contract(
        extra_evidence={
            "file_backed_export_artifact": native_mlx_artifact_evidence or {},
            "required_num_pairs": CONTEST_PAIR_COUNT,
        }
    )
    native_export_verified = bool(
        native_mlx_train_export_attached
        and native_mlx_receiver_proof_passed
        and native_file_proof_passed
    )
    native_train_export_planned = bool(native_mlx_train_export_attached)
    current_execution_path = (
        "cpu_advisory_plus_mlx_native_export_attachment"
        if native_train_export_planned
        else "cpu_advisory_receiver_bound_packet"
    )
    next_required_adapter = (
        "snerv_learned_scoreaware_mlx_training_loop_bound_to_native_export"
        if bool(native_contract.get("surfaces_ready"))
        else "snerv_mlx_native_train_export_archive"
    )
    standalone_scorer_loop_attached = bool(scorer_loop_qat_attached)
    native_scorer_loop_planned = bool(native_mlx_scorer_loop_qat_attached)
    native_scorer_loop_verified = bool(
        native_scorer_loop_planned
        and native_mlx_scorer_loop_qat_receiver_contract_satisfied
        and native_mlx_scorer_loop_qat_ready_for_pose_guard_gate
        and native_mlx_scorer_loop_qat_accepted_improvement
        and native_mlx_scorer_loop_qat_best_materialized
    )
    native_scorer_loop_file_backed_ready = bool(
        native_export_verified
        and native_mlx_full600_campaign_ready
        and native_scorer_loop_verified
    )
    long_training_or_materialized_scorer_loop_bound = bool(
        native_mlx_long_training_bound or native_scorer_loop_file_backed_ready
    )
    standalone_scorer_loop_verified = False
    effective_scorer_loop_verified = bool(
        standalone_scorer_loop_verified or native_scorer_loop_verified
    )
    effective_scorer_loop_attached = bool(
        standalone_scorer_loop_attached or native_scorer_loop_planned
    )
    effective_scorer_loop_receiver_contract = bool(
        (
            standalone_scorer_loop_attached
            and scorer_loop_qat_receiver_contract_satisfied
        )
        or (
            native_scorer_loop_planned
            and native_mlx_scorer_loop_qat_receiver_contract_satisfied
        )
    )
    effective_scorer_loop_pose_guard = bool(
        (
            standalone_scorer_loop_attached
            and scorer_loop_qat_ready_for_pose_guard_gate
        )
        or (
            native_scorer_loop_planned
            and native_mlx_scorer_loop_qat_ready_for_pose_guard_gate
        )
    )
    effective_scorer_loop_accepted = bool(
        (standalone_scorer_loop_attached and scorer_loop_qat_accepted_improvement)
        or (
            native_scorer_loop_planned
            and native_mlx_scorer_loop_qat_accepted_improvement
        )
    )
    native_real_teachers_bound = bool(
        native_mlx_real_segnet_teacher_bound
        and native_mlx_real_posenet_teacher_bound
    )
    effective_real_segnet_teacher = bool(
        effective_scorer_loop_verified or native_mlx_real_segnet_teacher_bound
    )
    effective_real_posenet_teacher = bool(
        effective_scorer_loop_verified or native_mlx_real_posenet_teacher_bound
    )
    candidate_selected = bool(candidate_row)
    official_source_forward_split = (
        _build_snerv_official_source_forward_authority_split(
            candidate=candidate_row,
            artifact_evidence=native_mlx_artifact_evidence,
        )
    )
    full_video = int(num_pairs) >= CONTEST_PAIR_COUNT
    serious_full_video_candidate = bool(
        candidate_selected and full_video and int(requested_epochs) >= 8
    )
    scorer_input_health_gate = _build_nerv_local_scorer_input_health_gate(
        native_mlx_artifact_evidence,
        family_token="snerv",
        require_profile=bool(
            serious_full_video_candidate
            and (
                native_mlx_scorer_input_distribution_guard_bound
                or full_video_local_prefilter_attached
            )
        ),
    )
    skip_high_export_admission_gate = _build_snerv_skip_high_export_admission_gate(
        candidate=candidate_row,
        artifact_evidence=native_mlx_artifact_evidence,
    )
    levels = _int(candidate_row.get("levels"), 3)
    lf_bits = _num(candidate_row.get("bits_per_coeff"), 2.5)
    step_bits = _num(candidate_row.get("step_map_bits_per_coeff"), 4.0)
    decoder_codec = str(candidate_row.get("decoder_payload_codec", "manual_cli"))
    byte_feedback = _base_byte_feedback(
        candidate=candidate_row,
        measured_num_pairs=(
            int(num_pairs) if measured_num_pairs is None else int(measured_num_pairs)
        ),
        measured_payload_bytes=measured_packet_bytes,
        measured_archive_bytes=measured_archive_bytes,
        archive_minus_nominal_bytes=archive_minus_nominal_bytes,
        archive_to_nominal_ratio=archive_to_nominal_ratio,
        calibrated_archive_overrun_bytes=calibrated_archive_overrun_bytes,
        required_nominal_payload_bytes_max=required_nominal_payload_bytes_max,
        hard_byte_ceiling_measurement_bypass_enabled=(
            hard_byte_ceiling_measurement_bypass_enabled
        ),
        hard_byte_ceiling_checked_after_export=hard_byte_ceiling_checked_after_export,
    ) if candidate_selected else {
        "schema": BYTE_FEEDBACK_SCHEMA,
        "candidate_id": None,
        "candidate_num_pairs": None,
        "measured_num_pairs": int(num_pairs),
        "feedback_scope": "manual_cli_probe",
        "scope_matches_candidate": False,
        "feedback_ready": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    blockers: list[str] = [
        str(blocker)
        for blocker in list(native_contract.get("blockers") or [])
        if not (
            native_export_verified
            and str(blocker)
            == "snerv_mlx_native_adapter_surfaces_present_but_unproven"
        )
    ]
    blockers.extend(official_source_forward_split.get("blockers") or [])
    blockers.extend(scorer_input_health_gate.get("blockers") or [])
    blockers.extend(skip_high_export_admission_gate.get("blockers") or [])
    if not bool(native_contract.get("surfaces_ready")):
        blockers.append("snerv_mlx_native_train_export_adapter_missing")
    elif not native_export_verified:
        blockers.append("snerv_mlx_native_adapter_surfaces_present_but_unproven")
    if native_mlx_train_export_attached and not native_mlx_receiver_proof_passed:
        blockers.append("snerv_mlx_native_receiver_proof_missing_or_failed")
    if (
        native_mlx_train_export_attached
        and native_mlx_receiver_proof_passed
        and not native_file_proof_passed
    ):
        blockers.append("snerv_mlx_native_file_backed_export_proof_missing_or_failed")
        blockers.extend(native_file_evidence.get("blockers") or [])
    if not effective_scorer_loop_attached:
        blockers.append("snerv_scorer_loop_qat_not_attached")
    if not long_training_or_materialized_scorer_loop_bound:
        blockers.append(
            "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only"
        )
    if not native_mlx_full600_campaign_ready:
        blockers.append("snerv_mlx_native_full600_campaign_not_ready")
    if (
        native_scorer_loop_planned
        and not native_mlx_scorer_loop_qat_best_materialized
    ):
        blockers.append("snerv_native_scorer_loop_best_packet_not_materialized")
    if not candidate_selected:
        blockers.append("snerv_modelsize_candidate_not_selected_manual_probe")
    if not full_video:
        blockers.append("snerv_candidate_curriculum_full600_required_for_promotion")
    if str(step_map_coder_mode) != "waterfill":
        blockers.append("snerv_candidate_curriculum_requires_waterfill_step_maps")
    if (
        candidate_selected
        and (measured_packet_bytes is not None or measured_archive_bytes is not None)
        and not byte_feedback["scope_matches_candidate"]
    ):
        blockers.append("partial_pair_byte_feedback_only")
    elif candidate_selected and not byte_feedback["feedback_ready"]:
        blockers.append("snerv_snar1_byte_feedback_missing")
    if (
        candidate_selected
        and byte_feedback.get("feedback_ready") is True
        and byte_feedback.get("archive_under_hard_byte_ceiling") is False
    ):
        blockers.append("snerv_receiver_proven_archive_over_hard_byte_ceiling")
        if (
            str(candidate_row.get("snerv_model_size_adapter") or "")
            != "snerv_official_mfu_hfr_tub_numeric_primitives_v1"
        ):
            blockers.append(
                "snerv_over_ceiling_local_lf_grammar_reroute_to_official_packet_or_lf_recode"
            )
    if effective_scorer_loop_attached:
        if not effective_scorer_loop_receiver_contract:
            blockers.append("snerv_scorer_loop_qat_receiver_contract_failed")
        if not effective_scorer_loop_pose_guard:
            blockers.append("snerv_scorer_loop_qat_pose_guard_not_ready")
        if not effective_scorer_loop_accepted:
            blockers.append("snerv_scorer_loop_qat_no_accepted_improvement")
    scorer_input_distribution_guard_verified = bool(
        native_mlx_scorer_input_distribution_guard_bound
        and (
            not scorer_input_health_gate["profile_required"]
            or scorer_input_health_gate["local_replay_admissible"]
        )
    )
    pr95_binding = build_pr95_stack_binding_requirements(
        family="snerv",
        evidence=build_pr95_stack_binding_evidence(
            modelsize_archive_budget=candidate_selected,
            pr95_staged_curriculum=bool(native_mlx_pr95_curriculum_bound),
            real_segnet_teacher=effective_real_segnet_teacher,
            real_posenet_teacher=effective_real_posenet_teacher,
            differentiable_pose_preprocess=bool(
                native_mlx_differentiable_pose_preprocess_bound
            ),
            eval_roundtrip_ste=bool(native_mlx_eval_roundtrip_ste_bound),
            scorer_input_distribution_guard=scorer_input_distribution_guard_verified,
            qat_forward=bool(
                effective_scorer_loop_verified or native_mlx_coder_qat_bound
            ),
            coder_aware_regularizer=bool(
                effective_scorer_loop_verified or native_mlx_coder_qat_bound
            ),
            muon_adamw_partition=bool(native_mlx_muon_adamw_partition_bound),
            archive_in_loop_byte_oracle=bool(byte_feedback.get("feedback_ready")),
            byte_closed_archive_export=measured_archive_bytes is not None,
            receiver_proof=bool(receiver_proof_attached),
            full_video_local_prefilter=bool(full_video_local_prefilter_attached),
            local_cpu_replay_gate=bool(local_cpu_replay_gate_attached),
        ),
    )
    long_campaign_prelaunch_gate = build_pr95_long_campaign_prelaunch_gate(
        pr95_binding
    )
    blockers.extend(pr95_binding["blockers"])
    return {
        "schema": SCHEMA,
        "family": "snerv",
        "candidate_conditioned": candidate_selected,
        "candidate_id": candidate_row.get("candidate_id"),
        "num_pairs": int(num_pairs),
        "campaign_scope": "full600" if full_video else "partial_pair_smoke",
        "receiver_grammar_controls": {
            "levels": levels,
            "lf_bits_per_coeff": lf_bits,
            "step_map_bits_per_coeff": step_bits,
            "step_map_coder_mode": str(step_map_coder_mode),
            "decoder_payload_codec": decoder_codec,
        },
        "training_plan": {
            "requested_epochs": int(requested_epochs),
            "native_mlx_training_required": True,
            "current_execution_path": current_execution_path,
            "next_required_adapter": next_required_adapter,
            "native_mlx_adapter_contract": native_contract,
            "native_mlx_adapter_surfaces_ready": bool(
                native_contract.get("surfaces_ready")
            ),
            "native_mlx_adapter_full600_campaign_ready": bool(
                native_contract.get("full600_campaign_ready")
            ),
            "attachment_authority_contract": {
                "schema": "snerv_training_attachment_authority_contract.v1",
                "planned_surfaces_are_not_receiver_or_score_authority": True,
                "queue_ready_is_not_receiver_or_exact_authority": True,
                "receiver_authority_requires_file_backed_export_and_replay": True,
                "official_receiver_payload_is_not_source_forward_authority": True,
                "scorer_input_health_required_for_local_replay": True,
                "skip_high_value_domain_xray_required_for_scalar_modes": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "official_source_forward_authority_split": (
                official_source_forward_split
            ),
            "scorer_input_health_gate": scorer_input_health_gate,
            "skip_high_export_admission_gate": skip_high_export_admission_gate,
            "native_mlx_train_export_planned": native_train_export_planned,
            "native_mlx_train_export_attached": bool(native_mlx_train_export_attached),
            "native_mlx_train_export_verified": native_export_verified,
            "native_mlx_export_attachment_is_learned_training": False,
            "learned_scoreaware_mlx_training_required_next": (
                not long_training_or_materialized_scorer_loop_bound
            ),
            "native_mlx_long_training_bound": bool(native_mlx_long_training_bound),
            "native_mlx_scorer_loop_file_backed_long_training_ready": (
                native_scorer_loop_file_backed_ready
            ),
            "native_mlx_receiver_proof_passed": bool(
                native_mlx_receiver_proof_passed
            ),
            "native_mlx_file_backed_export_proof_passed": native_file_proof_passed,
            "native_mlx_file_backed_export_evidence": native_file_evidence,
            "native_mlx_export_verified": native_export_verified,
            "native_mlx_export_full600_campaign_ready": bool(
                native_mlx_full600_campaign_ready
            ),
            "scorer_loop_qat_attached": effective_scorer_loop_attached,
            "standalone_scorer_loop_qat_attached": standalone_scorer_loop_attached,
            "standalone_scorer_loop_qat_verified": standalone_scorer_loop_verified,
            "standalone_scorer_loop_qat_requires_receiver_packet_materialization": (
                standalone_scorer_loop_attached
            ),
            "scorer_loop_qat_receiver_contract_satisfied": bool(
                effective_scorer_loop_receiver_contract
            ),
            "scorer_loop_qat_ready_for_pose_guard_gate": bool(
                effective_scorer_loop_pose_guard
            ),
            "scorer_loop_qat_accepted_improvement": bool(
                effective_scorer_loop_accepted
            ),
            "native_mlx_scorer_loop_qat_attached": native_scorer_loop_planned,
            "native_mlx_scorer_loop_qat_planned": native_scorer_loop_planned,
            "native_mlx_scorer_loop_qat_verified": native_scorer_loop_verified,
            "native_mlx_scorer_loop_qat_receiver_contract_satisfied": bool(
                native_mlx_scorer_loop_qat_receiver_contract_satisfied
            ),
            "native_mlx_scorer_loop_qat_ready_for_pose_guard_gate": bool(
                native_mlx_scorer_loop_qat_ready_for_pose_guard_gate
            ),
            "native_mlx_scorer_loop_qat_accepted_improvement": bool(
                native_mlx_scorer_loop_qat_accepted_improvement
            ),
            "native_mlx_scorer_loop_qat_best_materialized": bool(
                native_mlx_scorer_loop_qat_best_materialized
            ),
            "native_mlx_real_segnet_teacher_bound": bool(
                native_mlx_real_segnet_teacher_bound
            ),
            "native_mlx_real_posenet_teacher_bound": bool(
                native_mlx_real_posenet_teacher_bound
            ),
            "native_mlx_joint_real_teachers_bound": native_real_teachers_bound,
            "effective_real_segnet_teacher": effective_real_segnet_teacher,
            "effective_real_posenet_teacher": effective_real_posenet_teacher,
            "native_mlx_pr95_curriculum_bound": bool(
                native_mlx_pr95_curriculum_bound
            ),
            "native_mlx_eval_roundtrip_ste_bound": bool(
                native_mlx_eval_roundtrip_ste_bound
            ),
            "native_mlx_scorer_input_distribution_guard_bound": bool(
                native_mlx_scorer_input_distribution_guard_bound
            ),
            "native_mlx_scorer_input_distribution_guard_verified": (
                scorer_input_distribution_guard_verified
            ),
            "native_mlx_differentiable_pose_preprocess_bound": bool(
                native_mlx_differentiable_pose_preprocess_bound
            ),
            "native_mlx_coder_qat_bound": bool(native_mlx_coder_qat_bound),
            "native_mlx_muon_adamw_partition_bound": bool(
                native_mlx_muon_adamw_partition_bound
            ),
            "receiver_proof_attached": bool(receiver_proof_attached),
            "full_video_local_prefilter_attached": bool(
                full_video_local_prefilter_attached
            ),
            "local_cpu_replay_gate_attached": bool(local_cpu_replay_gate_attached),
        },
        "official_source_forward_authority_split": official_source_forward_split,
        "scorer_input_health_gate": scorer_input_health_gate,
        "skip_high_export_admission_gate": skip_high_export_admission_gate,
        "byte_oracle_logging": byte_feedback,
        "pr95_stack_binding": pr95_binding,
        "long_campaign_prelaunch_gate": long_campaign_prelaunch_gate,
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }


__all__ = [
    "AUTHORITY_FIELD_NAMES",
    "BYTE_FEEDBACK_SCHEMA",
    "SCHEMA",
    "build_hinerv_candidate_curriculum_plan",
    "build_snerv_candidate_curriculum_plan",
    "strip_candidate_curriculum_authority_fields",
]
