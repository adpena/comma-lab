# SPDX-License-Identifier: MIT
"""Fail-closed planner for compact-carrier score-aware training campaigns.

This module is the evidence boundary between cheap compact-carrier probes
(``hi_nerv``, ``snerv``, ``rnerv``/enhancers) and the shared MLX score-aware
training harness. It encodes the current hard lesson:

* a small projected archive is not a score win when the carrier is unfit;
* near-zero latent JVP leverage routes allocation pressure into decoder-weight
  training, not post-hoc latent edits;
* PR95-style optimizer/QAT/coder-aware pressure is part of the rate axis, not
  a finishing pass.

The output is a proxy/planning row by construction. It can enqueue local MLX
training or replay work, but it cannot promote scores or exact-auth dispatch.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    require_no_truthy_authority_fields,
)
from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    MODEL_SIZE_BUDGET_PLAN_SCHEMA,
    build_modelsize_budget_plan,
)

__all__ = [
    "CANONICAL_SCORE_AWARE_DECODER_TRAINING_STACK",
    "CarrierTrainingPlanError",
    "CarrierTrainingThresholds",
    "build_score_aware_carrier_training_plan",
]


CANONICAL_SCORE_AWARE_DECODER_TRAINING_STACK: tuple[str, ...] = (
    "real_segnet_teacher",
    "real_posenet_teacher",
    "eval_roundtrip",
    "ema",
    "pr95_8_stage_curriculum",
    "adamw_stage_schedule",
    "muon_final_stage",
    "coder_aware_regularization_c1a",
    "sigma_noise_qat",
    "quant_noise_qat",
    "nvrc_learned_quantization",
    "byte_closed_archive_export_smoke",
)

_STACK_READINESS_KEYS: dict[str, str] = {
    "real_segnet_teacher": "real_segnet_teacher_ready",
    "real_posenet_teacher": "real_posenet_teacher_ready",
    "eval_roundtrip": "eval_roundtrip_ready",
    "ema": "ema_ready",
    "pr95_8_stage_curriculum": "pr95_curriculum_ready",
    "adamw_stage_schedule": "adamw_ready",
    "muon_final_stage": "muon_ready",
    "coder_aware_regularization_c1a": "coder_aware_regularization_ready",
    "sigma_noise_qat": "sigma_noise_qat_ready",
    "quant_noise_qat": "quant_noise_qat_ready",
    "nvrc_learned_quantization": "nvrc_learned_quant_ready",
    "byte_closed_archive_export_smoke": "byte_closed_archive_export_ready",
}

_RECEIVER_CLOSED_MODELSIZE_STATUS = "receiver_closed_modelsize_budget_selected"


@dataclass(frozen=True)
class CarrierTrainingThresholds:
    """Decision thresholds for local carrier evidence.

    ``competitive_advisory_score_ceiling`` intentionally defaults to ``0.5``:
    above that, local MLX/advisory rows are not competitive enough to justify
    exact-auth spend without first improving fit. This is a planning threshold,
    not score authority.
    """

    unusable_d_seg_floor: float = 0.20
    plausible_d_seg_ceiling: float = 0.05
    competitive_advisory_score_ceiling: float = 0.50
    low_latent_jvp_norm_ceiling: float = 1.0e-3


class CarrierTrainingPlanError(ValueError):
    """Raised when carrier evidence violates the planner contract."""


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ordered_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _classify_fit(
    evidence: Mapping[str, Any],
    thresholds: CarrierTrainingThresholds,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    d_seg = _float_or_none(evidence.get("d_seg"))
    advisory_score = _float_or_none(evidence.get("advisory_score"))

    if d_seg is not None and d_seg >= thresholds.unusable_d_seg_floor:
        blockers.append("carrier_fit_unusable_d_seg")
        return "unusable", blockers
    if advisory_score is not None and advisory_score > thresholds.competitive_advisory_score_ceiling:
        blockers.append("carrier_fit_unusable_advisory_score")
        return "unusable", blockers
    if d_seg is not None and d_seg <= thresholds.plausible_d_seg_ceiling:
        return "locally_plausible", blockers
    return "pending_full_fit_evidence", blockers


def _classify_latent_leverage(
    evidence: Mapping[str, Any],
    thresholds: CarrierTrainingThresholds,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    status = str(evidence.get("latent_leverage_status") or "").strip().lower()
    norm = _float_or_none(evidence.get("latent_jvp_norm_max"))
    if status in {"near_zero", "low", "demoted_low_leverage"}:
        blockers.append("latent_posthoc_allocator_demoted_low_leverage")
        return "near_zero", blockers
    if norm is not None and norm <= thresholds.low_latent_jvp_norm_ceiling:
        blockers.append("latent_posthoc_allocator_demoted_low_leverage")
        return "near_zero", blockers
    if norm is None and not status:
        blockers.append("latent_leverage_probe_missing")
        return "unknown", blockers
    return "active_or_unresolved", blockers


def _missing_stack_items(evidence: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for item, key in _STACK_READINESS_KEYS.items():
        if not _truthy(evidence.get(key)):
            missing.append(item)
    return missing


def build_score_aware_carrier_training_plan(
    evidence: Mapping[str, Any],
    *,
    carrier_id: str,
    baseline_id: str = "pr95_hnerv",
    thresholds: CarrierTrainingThresholds | None = None,
) -> dict[str, Any]:
    """Route compact-carrier evidence into the next score-aware training action.

    Args:
        evidence: Advisory/local carrier evidence. Must not contain truthy
            score, promotion, dispatch, or exact-auth authority fields.
        carrier_id: Stable carrier id, e.g. ``"hi_nerv"`` or ``"snerv"``.
        baseline_id: Baseline/control arm id, defaulting to PR95/HNeRV.
        thresholds: Optional local-planning thresholds.

    Returns:
        A proxy/planning row with canonical false-authority fields, blockers,
        route, and required stack items.

    Raises:
        CarrierTrainingPlanError: if ``carrier_id`` is empty or authority leaks
            through the evidence payload.
    """

    carrier = str(carrier_id).strip()
    if not carrier:
        raise CarrierTrainingPlanError("carrier_id must be non-empty")
    try:
        require_no_truthy_authority_fields(
            evidence,
            context=f"score_aware_carrier_training_plan[{carrier}]",
        )
    except ValueError as exc:
        raise CarrierTrainingPlanError(str(exc)) from exc

    thresholds = thresholds or CarrierTrainingThresholds()
    fit_status, fit_blockers = _classify_fit(evidence, thresholds)
    latent_status, latent_blockers = _classify_latent_leverage(evidence, thresholds)
    missing_stack = _missing_stack_items(evidence)
    modelsize_status = (
        "structural_rate_knob_present"
        if _truthy(evidence.get("modelsize_knob_present"))
        else "rate_knob_unproven"
    )
    modelsize_budget_rows = evidence.get("modelsize_budget_rows")
    modelsize_budget_plan = _modelsize_budget_plan(
        modelsize_budget_rows,
        carrier_id=carrier,
        baseline_id=baseline_id,
        hard_byte_ceiling=_int_or_none(evidence.get("hard_byte_ceiling")),
    )
    modelsize_budget_ready = _receiver_closed_modelsize_budget_ready(
        modelsize_budget_plan
    )
    g3_status = (
        "verified_exact"
        if _truthy(evidence.get("g3_adjoint_exact"))
        else "requires_adjoint_probe"
    )

    allocator_surface = (
        "decoder_weights"
        if latent_status == "near_zero"
        else "decoder_weights_plus_latents_pending_leverage_probe"
    )
    route = "run_receiver_closed_modelsize_ladder_before_score_aware_training"
    if modelsize_budget_ready:
        route = "run_score_aware_decoder_weight_training_full_main"
    if fit_status == "locally_plausible" and not missing_stack and modelsize_budget_ready:
        route = "run_byte_closed_local_replay_gate_before_exact_auth"

    blockers = _ordered_unique(
        [
            *fit_blockers,
            *latent_blockers,
            *(f"missing_training_stack:{item}" for item in missing_stack),
            *(
                []
                if modelsize_budget_ready
                else ["receiver_closed_modelsize_budget_ladder_missing"]
            ),
            *(
                f"modelsize_budget:{blocker}"
                for blocker in modelsize_budget_plan.get("blockers", [])
            ),
            *([] if g3_status == "verified_exact" else ["g3_adjoint_exactness_unverified"]),
            "no_score_claim_from_advisory_carrier_probe",
            "requires_byte_closed_archive_before_promotion",
            "requires_receiver_proof_before_exact_auth",
            "requires_full_video_local_replay_before_exact_auth",
            "requires_contest_cpu_then_cuda_auth_only_after_local_win",
        ]
    )

    row = {
        "schema": "score_aware_carrier_training_plan.v1",
        "carrier_id": carrier,
        "baseline_id": str(baseline_id),
        "planner_action": route,
        "carrier_fit_status": fit_status,
        "rate_knob_status": modelsize_status,
        "modelsize_budget_plan_status": modelsize_budget_plan["status"],
        "modelsize_budget_receiver_closed_ready": modelsize_budget_ready,
        "modelsize_budget_plan": modelsize_budget_plan,
        "g3_adjoint_status": g3_status,
        "latent_leverage_status": latent_status,
        "allocator_target_surface": allocator_surface,
        "linf_latent_posthoc_status": (
            "demoted"
            if latent_status == "near_zero"
            else "probe_before_use"
        ),
        "required_training_stack": list(CANONICAL_SCORE_AWARE_DECODER_TRAINING_STACK),
        "missing_training_stack": missing_stack,
        "score_aware_training_ready": (
            not missing_stack
            and g3_status == "verified_exact"
            and modelsize_budget_ready
        ),
        "q_at_rate_levers": [
            "coder_aware_regularization_c1a",
            "sigma_noise_qat",
            "quant_noise_qat",
            "nvrc_learned_quantization",
        ],
        "control_arm": {
            "baseline_id": str(baseline_id),
            "role": "train_export_archive_control_arm",
        },
        "evidence_summary": {
            "archive_bytes": evidence.get("archive_bytes"),
            "projected_archive_bytes_600pair": evidence.get("projected_archive_bytes_600pair"),
            "selected_modelsize_archive_bytes": modelsize_budget_plan.get(
                "selected_archive_bytes"
            ),
            "receiver_closed_selected_modelsize_archive_bytes": (
                modelsize_budget_plan.get("receiver_closed_selected_archive_bytes")
            ),
            "d_seg": evidence.get("d_seg"),
            "advisory_score": evidence.get("advisory_score"),
            "latent_jvp_norm_max": evidence.get("latent_jvp_norm_max"),
            "linf_delta_vs_l2": evidence.get("linf_delta_vs_l2"),
        },
    }
    return apply_proxy_evidence_boundary(row, dispatch_blockers=blockers)


def _modelsize_budget_plan(
    rows: Any,
    *,
    carrier_id: str,
    baseline_id: str,
    hard_byte_ceiling: int | None = None,
) -> dict[str, Any]:
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        return build_modelsize_budget_plan(
            rows,
            carrier_id=carrier_id,
            baseline_id=baseline_id,
            hard_byte_ceiling=hard_byte_ceiling,
        )
    return {
        "schema": MODEL_SIZE_BUDGET_PLAN_SCHEMA,
        "carrier_id": carrier_id,
        "baseline_id": baseline_id,
        "hard_byte_ceiling": hard_byte_ceiling,
        "status": "modelsize_budget_ladder_not_measured",
        "decision_basis": "no_modelsize_budget_rows",
        "selected_archive_bytes": None,
        "receiver_closed_selected_archive_bytes": None,
        "points": [],
        "measured_points": [],
        "receiver_closed_points": [],
        "point_count_by_evidence": {},
        "marginal_steps": [],
        "recommended_next_actions": [
            "run_receiver_closed_modelsize_ladder_before_long_budget_spend",
            "sweep_latent_dim_embed_dim_decoder_channel_and_codec_under_byte_ceiling",
        ],
        "blockers": [
            "modelsize_budget_ladder_not_measured",
            "receiver_closed_modelsize_ladder_has_fewer_than_two_points",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _receiver_closed_modelsize_budget_ready(plan: Mapping[str, Any]) -> bool:
    return (
        plan.get("status") == _RECEIVER_CLOSED_MODELSIZE_STATUS
        and plan.get("receiver_closed_selected_archive_bytes") is not None
    )
