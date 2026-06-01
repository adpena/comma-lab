# SPDX-License-Identifier: MIT
"""Fail-closed SNeRV scorer-loop decoder/QAT implementation contract.

The pose-guarded decoder gate closed the scalar HF-weighting family: it can
improve SegNet while destroying PoseNet. This module turns that negative result
into a machine-readable next-build contract, so the next agent implements the
missing scorer-loop decoder-weight trainer instead of rerunning another scalar
sweep or dispatching exact eval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SCHEMA = "snerv_scorer_loop_decoder_qat_contract.v1"
AXIS_TAG = "[macOS-CPU advisory]"
DEFAULT_LANE_ID = "lane_snerv_scorer_loop_decoder_qat_contract_20260601"
EXPECTED_FAILED_GATE_VERDICT = "NO_GO_FOR_PROMOTION_OR_EXACT_EVAL"


class SnervScorerLoopDecoderQatContractError(ValueError):
    """Raised when a pose-gate payload cannot produce a safe contract."""


@dataclass(frozen=True)
class SnervDecoderQatTrainingMode:
    """One allowed implementation mode for the next SNeRV decoder-fit patch."""

    mode_id: str
    objective: str
    protected_axis: str
    decoder_weight_surface: str
    quantization_policy: str
    allocator_home: str
    receiver_export_requirement: str
    acceptance_gate: str
    large_artifact_policy: str
    status: str = "implementation_missing"

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SnervScorerLoopDecoderQatContract:
    """Machine-readable contract for the next SNeRV scorer-loop implementation."""

    schema: str
    lane_id: str
    axis_tag: str
    source_gate_path: str | None
    source_gate_sha256: str | None
    source_gate_verdict: str
    source_gate_next_action: str | None
    baseline_label: str
    baseline_archive_bytes: int
    baseline_d_seg_linf: float
    baseline_d_pose_linf: float
    baseline_score_linf: float
    accepted_rows_in_source_gate: int
    closed_form_scalar_weighting_no_go: bool
    ready_for_scorer_loop_trainer_implementation: bool
    ready_for_local_training_smoke: bool
    ready_for_exact_eval_dispatch: bool
    score_claim: bool
    promotion_eligible: bool
    rank_or_kill_eligible: bool
    dispatch_hold_reason: str | None
    required_preconditions: tuple[str, ...]
    satisfied_preconditions: tuple[str, ...]
    blockers: tuple[str, ...]
    forbidden_actions: tuple[str, ...]
    acceptance_gate: dict[str, Any]
    allowed_training_modes: tuple[SnervDecoderQatTrainingMode, ...]
    next_code_artifacts: tuple[str, ...]
    verifier_tests_to_add: tuple[str, ...]

    def as_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_training_modes"] = [
            mode.as_jsonable() for mode in self.allowed_training_modes
        ]
        return payload


def build_snerv_scorer_loop_decoder_qat_contract(
    pose_gate_payload: dict[str, Any],
    *,
    source_gate_path: str | None = None,
    source_gate_sha256: str | None = None,
    lane_id: str = DEFAULT_LANE_ID,
    dispatch_hold_reason: str | None = None,
) -> SnervScorerLoopDecoderQatContract:
    """Build the next SNeRV implementation contract from a pose-gate artifact."""

    if not isinstance(pose_gate_payload, dict):
        raise SnervScorerLoopDecoderQatContractError("pose_gate_payload must be a dict")
    if pose_gate_payload.get("score_claim") is not False:
        raise SnervScorerLoopDecoderQatContractError("source gate must be false-authority")
    if pose_gate_payload.get("ready_for_exact_eval_dispatch") is not False:
        raise SnervScorerLoopDecoderQatContractError(
            "source gate must not be exact-eval ready"
        )

    verdict = _required_str(pose_gate_payload.get("verdict"), "verdict")
    accepted_rows = pose_gate_payload.get("accepted_rows")
    if not isinstance(accepted_rows, list):
        raise SnervScorerLoopDecoderQatContractError("accepted_rows must be a list")
    accepted_count = len(accepted_rows)
    closed_form_no_go = pose_gate_payload.get("closed_form_scalar_weighting_no_go") is True

    required = (
        "pose_guarded_gate_artifact_consumed",
        "least_squares_waterfill_control_present",
        "closed_form_scalar_component_weighting_no_go",
        "no_pose_guard_accepted_candidate",
        "exact_and_promotion_authority_false",
    )
    satisfied: list[str] = ["pose_guarded_gate_artifact_consumed"]
    blockers: list[str] = []
    if _has_baseline_metrics(pose_gate_payload):
        satisfied.append("least_squares_waterfill_control_present")
    else:
        blockers.append("least_squares_waterfill_control_missing")
    if closed_form_no_go:
        satisfied.append("closed_form_scalar_component_weighting_no_go")
    else:
        blockers.append("closed_form_scalar_component_weighting_not_closed")
    if accepted_count == 0 and verdict == EXPECTED_FAILED_GATE_VERDICT:
        satisfied.append("no_pose_guard_accepted_candidate")
    else:
        blockers.append("pose_gate_has_accepted_candidate_or_nonterminal_verdict")
    satisfied.append("exact_and_promotion_authority_false")

    implementation_ready = len(blockers) == 0
    local_training_blockers = [
        "snerv_scorer_loop_decoder_qat_trainer_cli_missing",
        "segnet_posenet_in_loop_gradient_path_missing",
        "decoder_weight_qat_receiver_export_proof_missing",
    ]
    exact_blockers = [
        "full_600_pair_receiver_proof_missing",
        "paired_contest_cpu_cuda_pass_missing",
    ]
    if dispatch_hold_reason:
        exact_blockers.append(dispatch_hold_reason)

    baseline_archive = _required_int(
        pose_gate_payload.get("baseline_archive_bytes"), "baseline_archive_bytes"
    )
    baseline_seg = _required_float(
        pose_gate_payload.get("baseline_d_seg_linf"), "baseline_d_seg_linf"
    )
    baseline_pose = _required_float(
        pose_gate_payload.get("baseline_d_pose_linf"), "baseline_d_pose_linf"
    )
    baseline_score = _required_float(
        pose_gate_payload.get("baseline_score_linf"), "baseline_score_linf"
    )
    acceptance_gate = {
        "control_label": _required_str(
            pose_gate_payload.get("baseline_label"), "baseline_label"
        ),
        "max_archive_bytes": _required_int(
            pose_gate_payload.get("max_archive_bytes"), "max_archive_bytes"
        ),
        "required_receiver_archive_replay_verified": True,
        "max_d_pose_linf": baseline_pose,
        "max_d_seg_linf": min(
            baseline_seg,
            _required_float(pose_gate_payload.get("seg_ceiling"), "seg_ceiling"),
        ),
        "max_score_linf": baseline_score,
        "authority_after_pass": "GO_LOCAL_CONTINUATION_ONLY",
        "authority_not_granted": (
            "promotion",
            "rank_or_kill",
            "exact_eval_dispatch",
            "score_claim",
        ),
    }

    return SnervScorerLoopDecoderQatContract(
        schema=SCHEMA,
        lane_id=lane_id,
        axis_tag=AXIS_TAG,
        source_gate_path=source_gate_path,
        source_gate_sha256=source_gate_sha256,
        source_gate_verdict=verdict,
        source_gate_next_action=_optional_str(pose_gate_payload.get("next_action")),
        baseline_label=str(acceptance_gate["control_label"]),
        baseline_archive_bytes=baseline_archive,
        baseline_d_seg_linf=baseline_seg,
        baseline_d_pose_linf=baseline_pose,
        baseline_score_linf=baseline_score,
        accepted_rows_in_source_gate=accepted_count,
        closed_form_scalar_weighting_no_go=closed_form_no_go,
        ready_for_scorer_loop_trainer_implementation=implementation_ready,
        ready_for_local_training_smoke=False,
        ready_for_exact_eval_dispatch=False,
        score_claim=False,
        promotion_eligible=False,
        rank_or_kill_eligible=False,
        dispatch_hold_reason=dispatch_hold_reason,
        required_preconditions=required,
        satisfied_preconditions=tuple(satisfied),
        blockers=tuple(dict.fromkeys(blockers + local_training_blockers + exact_blockers)),
        forbidden_actions=(
            "rerun_closed_form_scalar_or_component_hf_weight_sweep_for_promotion",
            "dispatch_full_video_or_exact_eval_from_pose_gate_negative",
            "promote_without_byte_closed_receiver_proof",
            "claim_score_from_macos_cpu_advisory_rows",
        ),
        acceptance_gate=acceptance_gate,
        allowed_training_modes=_allowed_modes(),
        next_code_artifacts=(
            "src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py",
            "tools/run_snerv_scorer_loop_decoder_qat_smoke.py",
            "src/tac/substrates/snerv_inverse_steg_carrier/tests/test_scorer_loop_decoder_qat.py",
            ".omx/research/snerv_scorer_loop_decoder_qat_smoke_<UTC>.json",
        ),
        verifier_tests_to_add=(
            "real_frames_are_loaded_from_upstream_video_not_synthetic_noise",
            "segnet_and_posenet_losses_both_enter_training_objective",
            "pose_guard_is_hard_constraint_before_segnet_gain",
            "quantized_decoder_export_roundtrips_through_snar1_receiver",
            "local_smoke_cannot_set_score_claim_or_exact_eval_ready",
        ),
    )


def _allowed_modes() -> tuple[SnervDecoderQatTrainingMode, ...]:
    return (
        SnervDecoderQatTrainingMode(
            mode_id="decoder_weight_linf_waterfill_qat",
            objective=(
                "train shared HF decoder weights against SegNet/PoseNet advisory "
                "losses, with PoseNet d_pose_linf as a hard guard"
            ),
            protected_axis="pose_first_then_seg_score",
            decoder_weight_surface="HfGenerationDecoder.kernels",
            quantization_policy=(
                "mixed precision by waterfill: protect high-leverage kernels at "
                "int8/fp16, demote low-leverage atoms toward int4/int2/zero only "
                "after replay and pose guard"
            ),
            allocator_home="decoder_weights_not_posthoc_per_pair_latents",
            receiver_export_requirement="SNAR1 receiver packet must consume quantized decoder",
            acceptance_gate="must pass pose-guarded decoder gate versus least-squares waterfill",
            large_artifact_policy="SSD temp/cold-store required before any full-frame cache",
        ),
        SnervDecoderQatTrainingMode(
            mode_id="nonlinear_hf_decoder_qat",
            objective=(
                "replace scalar weighted least-squares with a tiny nonlinear HF "
                "decoder trained in scorer loop"
            ),
            protected_axis="pose_first_then_seg_score",
            decoder_weight_surface="new receiver-portable HF decoder section",
            quantization_policy="QAT during fit, not post-hoc quantization",
            allocator_home="decoder_weight_training",
            receiver_export_requirement="inflate path must be numpy-portable and scorer-free",
            acceptance_gate="must beat least-squares waterfill under same archive/runtime gate",
            large_artifact_policy="local bounded smoke only until storage preflight is wired",
        ),
    )


def _has_baseline_metrics(payload: dict[str, Any]) -> bool:
    return (
        _optional_str(payload.get("baseline_label")) is not None
        and _optional_int(payload.get("baseline_archive_bytes")) is not None
        and _optional_float(payload.get("baseline_d_seg_linf")) is not None
        and _optional_float(payload.get("baseline_d_pose_linf")) is not None
        and _optional_float(payload.get("baseline_score_linf")) is not None
        and _optional_int(payload.get("max_archive_bytes")) is not None
        and _optional_float(payload.get("seg_ceiling")) is not None
    )


def _required_str(value: Any, name: str) -> str:
    out = _optional_str(value)
    if out is None:
        raise SnervScorerLoopDecoderQatContractError(f"missing string: {name}")
    return out


def _required_int(value: Any, name: str) -> int:
    out = _optional_int(value)
    if out is None:
        raise SnervScorerLoopDecoderQatContractError(f"missing integer: {name}")
    return out


def _required_float(value: Any, name: str) -> float:
    out = _optional_float(value)
    if out is None:
        raise SnervScorerLoopDecoderQatContractError(f"missing float: {name}")
    return out


def _optional_str(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
