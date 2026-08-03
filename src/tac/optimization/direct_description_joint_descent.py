# SPDX-License-Identifier: MIT
"""Executable low-dimensional joint-descent consumer for DDM #366.

The receiver archive remains the source of truth.  This module adds three
strictly separated surfaces:

* a lossless G1/lane/template parameter lift and exact stage-00 recompile;
* a local realized-secant MLX module whose only differentiable leaves are the
  counted description coordinates;
* atomic, stage-preserving optimizer checkpoints with EMA and identity custody.

MLX/Metal results are training signal only.  Exact CPU/CUDA evaluation of the
emitted receiver archive remains the contest authority.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.canonical_equations.adam_v_variance_warmup_20260717 import (
    adam_v_variance_warmup_epochs,
)
from tac.optimization.ddm_rg1_receiver_grammar import project_polygon_center
from tac.optimization.ddm_ws1_warm_start import (
    WS1WarmStartArchiveV1,
    parse_ws1_warm_start_archive,
    receive_joint_descent_archive,
)
from tac.optimization.direct_description_carrier_compose import (
    LANE_PROGRAM_MEMBER,
    REALIZATION_STATIC_RULE_MEMBER,
    WORLDSHEET_G1_MEMBER,
    CarrierComposeReceiverV1,
    LanePeriodicProgramV1,
    RowBandScorerTemplateV1,
    compile_carrier_compose_archive,
    parse_carrier_compose_archive,
)
from tac.optimization.direct_description_entropy_priced_member import rfc8785_canonicalize
from tac.optimization.direct_description_g1_worldsheet import (
    G1WorldsheetParameterLiftV1,
    encode_lifted_g1_movable_worldsheet,
    lift_g1_movable_worldsheet,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.witness_control.costate_estimator import slope_with_stderr
from tac.witness_control.resume_registry import (
    RESUME_REGISTRY_MANIFEST_KEY,
    ResumeRegistry,
)
from tac.witness_control.sigma_min_plateau import (
    DEFAULT_EMA_SPAN,
    DEFAULT_FLAT_REL_BAND,
    DEFAULT_HYSTERESIS,
    DEFAULT_SETTLE_WINDOW,
)
from tac.witness_dsl.guarded_constant_registry import (
    MARGIN_FLOOR_INCUMBENT as _MARGIN_FLOOR_DEFAULT,
)
from tac.witness_dsl.guarded_constant_registry import (
    MARGIN_FLOOR_MIGRATION_IS_BYTE_IDENTICAL as _MARGIN_FLOOR_BYTE_IDENTICAL,
)

# ddm_gk1 2026-08-03 migration invariant: converting a hardcoded literal to a
# resolved constant is a BEHAVIOUR CHANGE unless byte-identity is PROVEN.  It is
# asserted here, at the consuming module, so the proof travels with the consumer
# and not only with the declaration.
assert _MARGIN_FLOOR_BYTE_IDENTICAL and _MARGIN_FLOOR_DEFAULT == 0.1, (  # GUARDED_CONSTANT_OK:the migration invariant itself — this literal IS the byte-identity assertion, not a frozen use
    "seg_margin_hinge_floor migration is not byte-identical: adopting it would "
    f"change the shipped margin_floor from 0.1 to {_MARGIN_FLOOR_DEFAULT!r}. That is a "
    "measurable behaviour change and must be measured, not landed silently."
)

TYPED_SCHEMA: Final = "DirectDescriptionJointDescentTypedConfigV1"
TICKET_SCHEMA: Final = "ddm_joint_descent_witness_program_ticket.v1"
MEMORY_RECEIPT_SCHEMA: Final = "ddm_joint_descent_memory_preflight.v1"
CHECKPOINT_SCHEMA: Final = "ddm_joint_descent_stage_checkpoint.v1"
POSE_FINISH_ENGAGE_CONFIG_SCHEMA: Final = "ddm_pose_finish_exact_verdict_plateau.v1"
POSE_FINISH_ENGAGE_STATE_SCHEMA: Final = "ddm_pose_finish_exact_verdict_plateau_state.v1"
WORST_GEOMETRY_MEMORY_SCHEMA: Final = "ddm_joint_descent_worst_geometry_memory_contract.v1"
LEGACY_PROGRAM_SHA256: Final = "68a8aa97b25a6be2f8f08e36fcf4957fe032233e43b1050b75ad13c9d7dad89c"
J3_PROGRAM_SHA256: Final = "df8db01f60d582b0a716ae62af3422997fcc12c014364939ab2935a2c403b824"
J5_PROGRAM_SHA256: Final = "13e194a8a354d53489f0ff68a5042237e69b4b6841a6b7959a15873fffa7b6e8"
J6A_PROGRAM_SHA256: Final = "3ba05e4d8fd2f85475173f0a9e17e668198507350d353a4257aaf196692b98c2"
J7_PROGRAM_SHA256: Final = "bb30eade311ed15e7541bdda4f5d5edbd72b28933a0dd2066be8b967a20aadf2"
J7_W_SEG_PROGRAM_SHA256: Final = "de285d70b7ac1c823e70f4b2c5e2f5f728e5ff9e65e03c4e2c9583c486dda0a1"
J7_W_JOINT_PROGRAM_SHA256: Final = "81ae90f3d1bfec508e23cbebe37e94f965b46e5d82903d2ae9d077eb365d7ce4"
J9_W_JOINT_PROGRAM_SHA256: Final = "96ca852b61168cf86a6e6d9166a27aa73d955a00b5d06ed940210d79f92f34d7"
J10_PROGRAM_ID: Final = "ddm_j10_366_ema_verdict_shadow_cure_n600_seed0"
WS3_W_SEG_PROGRAM_SHA256: Final = "a90004c7d75571a2f97c7f6f87770b25cfda7ea46e76ce2e1e9d230e454ce838"
J7_PROGRAM_SHA256S: Final = frozenset(
    {
        J7_PROGRAM_SHA256,
        J7_W_SEG_PROGRAM_SHA256,
        J7_W_JOINT_PROGRAM_SHA256,
        J9_W_JOINT_PROGRAM_SHA256,
        WS3_W_SEG_PROGRAM_SHA256,
    }
)
# Resealed after editing the semantic ticket; updated by the deterministic hash
# seal step in this landing.
EXPECTED_PROGRAM_SHA256: Final = "9c3575aa58a5264bd0897afaaf22a62807336c037c42a8943e89ee69c84efd5b"
SUPPORTED_PROGRAM_SHA256: Final = frozenset(
    {
        LEGACY_PROGRAM_SHA256,
        J3_PROGRAM_SHA256,
        EXPECTED_PROGRAM_SHA256,
        J5_PROGRAM_SHA256,
        J6A_PROGRAM_SHA256,
        J7_PROGRAM_SHA256,
        J7_W_SEG_PROGRAM_SHA256,
        J7_W_JOINT_PROGRAM_SHA256,
        J9_W_JOINT_PROGRAM_SHA256,
        WS3_W_SEG_PROGRAM_SHA256,
    }
)
EXPECTED_ARCHIVE_SHA256: Final = "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"
EXPECTED_ARCHIVE_BYTES: Final = 133_941
POINTER: Final = "0.1910828242 [contest-CPU]"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
BASELINE_DSEG: Final = 0.027470296224
SCORER_WIDTH: Final = 512
SCORER_HEIGHT: Final = 384


class ProposalGeometryInfeasibleError(DirectDescriptionError):
    """A proposal-only geometry failure that may be rejected without hiding custody faults."""

    def __init__(self, event: Mapping[str, Any]) -> None:
        self.event = dict(event)
        super().__init__(str(self.event["reason"]))


def _canonical_parameter_shadow(value: Any) -> str:
    """Collapse descriptive row labels onto the two load-bearing shadows."""

    shadow = str(value).strip().lower()
    if shadow.startswith("live"):
        return "live"
    if shadow.startswith("ema"):
        return "ema"
    return shadow


def classify_verdict_informativeness(
    *,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    minimum_realized_parameter_count_delta: int,
) -> tuple[bool, str]:
    """Classify whether receiver bytes/counts carry a verdict-scale signal."""

    minimum_delta = int(minimum_realized_parameter_count_delta)
    if minimum_delta < 1:
        return False, "REFUSE_VERDICT_INFORMATIVENESS_THRESHOLD_INVALID"
    reference_sha = reference.get("archive_sha256")
    candidate_sha = candidate.get("archive_sha256")
    if not isinstance(reference_sha, str) or not isinstance(candidate_sha, str):
        return False, "REFUSE_VERDICT_INFORMATIVENESS_ARCHIVE_CUSTODY_MISSING"
    if reference_sha == candidate_sha:
        return False, "ARCHIVE_BYTES_IDENTICAL_TO_SAME_SHADOW_REFERENCE"
    try:
        reference_count = int(reference["realized_parameter_count"])
        candidate_count = int(candidate["realized_parameter_count"])
    except (KeyError, TypeError, ValueError):
        return False, "REFUSE_VERDICT_INFORMATIVENESS_REALIZED_COUNT_MISSING"
    if reference_count < 0 or candidate_count < 0:
        return False, "REFUSE_VERDICT_INFORMATIVENESS_REALIZED_COUNT_INVALID"
    if abs(candidate_count - reference_count) < minimum_delta:
        return False, "REALIZED_PARAMETER_COUNT_DELTA_BELOW_SIGNAL_FLOOR"
    return True, "RECEIVER_BYTES_AND_REALIZED_COUNT_ABOVE_SIGNAL_FLOOR"


def classify_realized_stage_verdict(
    *,
    reference_d_seg: float,
    reference_d_pose: float,
    candidate_d_seg: float,
    candidate_d_pose: float,
    target_d_seg: float,
    target_d_pose: float | None,
) -> str:
    """Classify only an exact realized-through-receiver stage measurement.

    A stage may continue only when neither exact component regresses and at
    least one exact component descends.  This deliberately refuses proxy-loss,
    STE, or first-order predictions as campaign decisions.
    """

    values = (
        reference_d_seg,
        reference_d_pose,
        candidate_d_seg,
        candidate_d_pose,
        target_d_seg,
    )
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
        return "REFUSE_REALIZED_STAGE_VERDICT_NONFINITE_OR_NEGATIVE"
    if target_d_pose is not None and (not math.isfinite(float(target_d_pose)) or float(target_d_pose) < 0.0):
        return "REFUSE_REALIZED_STAGE_TARGET_NONFINITE_OR_NEGATIVE"
    if candidate_d_seg > reference_d_seg:
        return "BLOCKED_REALIZED_DSEG_REGRESSION"
    if candidate_d_pose > reference_d_pose:
        return "BLOCKED_REALIZED_DPOSE_REGRESSION"
    if candidate_d_seg == reference_d_seg:
        if candidate_d_pose < reference_d_pose:
            return "REALIZED_STAGE_SEG_FLAT_POSE_DESCENT_CONTINUE"
        return "BLOCKED_REALIZED_NO_COMPONENT_DESCENT"
    target_met = candidate_d_seg <= target_d_seg and (target_d_pose is None or candidate_d_pose <= target_d_pose)
    return "REALIZED_STAGE_TARGET_MET" if target_met else "REALIZED_STAGE_DESCENT_CONTINUE"


def classify_shadow_consistent_stage_verdict(
    *,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    policy: VerdictShadowPolicyV1,
    consecutive_degenerate_verdicts: int,
    target_d_seg: float,
    target_d_pose: float | None,
) -> tuple[str, dict[str, Any]]:
    """Apply the J10 same-shadow and two-verdict informativeness apparatus."""

    reference_shadow = _canonical_parameter_shadow(reference.get("parameter_shadow"))
    candidate_shadow = _canonical_parameter_shadow(candidate.get("parameter_shadow"))
    if policy.same_shadow_reference_required and (
        reference_shadow != candidate_shadow or candidate_shadow != policy.decision_shadow
    ):
        return (
            "REFUSE_REALIZED_STAGE_VERDICT_SHADOW_MISMATCH",
            {
                "informative": False,
                "informativeness_reason": "PARAMETER_SHADOW_MISMATCH",
                "reference_parameter_shadow": reference_shadow,
                "candidate_parameter_shadow": candidate_shadow,
            },
        )
    informative, reason = classify_verdict_informativeness(
        reference=reference,
        candidate=candidate,
        minimum_realized_parameter_count_delta=policy.minimum_realized_parameter_count_delta,
    )
    apparatus = {
        "informative": informative,
        "informativeness_reason": reason,
        "reference_parameter_shadow": reference_shadow,
        "candidate_parameter_shadow": candidate_shadow,
        "consecutive_degenerate_verdicts_before": int(consecutive_degenerate_verdicts),
    }
    if not informative:
        if reason.startswith("REFUSE_"):
            return reason, apparatus
        if int(consecutive_degenerate_verdicts) < policy.maximum_consecutive_degenerate_verdicts:
            return "VERDICT_NOT_YET_INFORMATIVE", apparatus
        return "BLOCKED_VERDICT_STILL_NOT_INFORMATIVE_NEXT_SCHEDULED_VERDICT", apparatus
    return (
        classify_realized_stage_verdict(
            reference_d_seg=float(reference["d_seg"]),
            reference_d_pose=float(reference["d_pose"]),
            candidate_d_seg=float(candidate["d_seg"]),
            candidate_d_pose=float(candidate["d_pose"]),
            target_d_seg=target_d_seg,
            target_d_pose=target_d_pose,
        ),
        apparatus,
    )


def count_consecutive_scheduled_degenerate_verdicts(history: Sequence[Mapping[str, Any]]) -> int:
    """Count only scheduled same-shadow verdicts for J10's one-verdict grace."""

    count = 0
    for historical in reversed(history):
        if historical.get("scheduled_stage_verdict") is not True:
            continue
        if historical.get("realized_stage_decision") != "VERDICT_NOT_YET_INFORMATIVE":
            break
        count += 1
    return count


def classify_cumulative_fire_gate(
    *,
    baseline_d_seg: float,
    baseline_d_pose: float,
    candidate_d_seg: float,
    candidate_d_pose: float,
    cumulative_residual_delta_errors: int,
    residual_descent_required: bool,
) -> tuple[bool, str]:
    """Fire readiness is cumulative against immutable stage00, never local."""

    values = (baseline_d_seg, baseline_d_pose, candidate_d_seg, candidate_d_pose)
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
        return False, "REFUSE_CUMULATIVE_FIRE_GATE_NONFINITE_OR_NEGATIVE"
    if candidate_d_seg > baseline_d_seg:
        return False, "BLOCKED_CUMULATIVE_DSEG_REGRESSION_VS_STAGE00"
    if candidate_d_pose > baseline_d_pose:
        return False, "BLOCKED_CUMULATIVE_DPOSE_REGRESSION_VS_STAGE00"
    if candidate_d_seg == baseline_d_seg and candidate_d_pose == baseline_d_pose:
        return False, "BLOCKED_CUMULATIVE_NO_COMPONENT_DESCENT_VS_STAGE00"
    if residual_descent_required and int(cumulative_residual_delta_errors) >= 0:
        return False, "BLOCKED_CUMULATIVE_RESIDUAL_NOT_DESCENDING_VS_STAGE00"
    return True, "CUMULATIVE_COMPONENT_AND_RESIDUAL_FIRE_GATE_GREEN_VS_STAGE00"


def classify_governed_stage_exit(
    *,
    target_met: bool,
    stage_limit: bool,
    plateau: bool,
    component_decision: str | None,
) -> str:
    """Only an exact target hit advances; exhausted target-unmet work stops."""

    if target_met:
        return "ADVANCE_EXACT_TARGET_MET"
    if component_decision is not None and component_decision in {
        "BLOCKED_REALIZED_DSEG_REGRESSION",
        "BLOCKED_REALIZED_DPOSE_REGRESSION",
        "BLOCKED_VERDICT_STILL_NOT_INFORMATIVE_NEXT_SCHEDULED_VERDICT",
        "REFUSE_REALIZED_STAGE_VERDICT_SHADOW_MISMATCH",
        "REFUSE_VERDICT_INFORMATIVENESS_ARCHIVE_CUSTODY_MISSING",
        "REFUSE_VERDICT_INFORMATIVENESS_REALIZED_COUNT_MISSING",
        "REFUSE_VERDICT_INFORMATIVENESS_REALIZED_COUNT_INVALID",
        "REFUSE_VERDICT_INFORMATIVENESS_THRESHOLD_INVALID",
        "REFUSE_REALIZED_STAGE_VERDICT_NONFINITE_OR_NEGATIVE",
        "REFUSE_REALIZED_STAGE_TARGET_NONFINITE_OR_NEGATIVE",
    }:
        return component_decision
    if plateau:
        return "STOPPED_BELOW_TARGET_PLATEAU"
    if stage_limit:
        return "STOPPED_BELOW_TARGET_MAXIMUM_STEPS"
    return "CONTINUE_EXACT_TARGET_UNMET"


def exact_final_target_gate(
    *,
    final_verdict: Mapping[str, Any] | None,
    final_stage: FullRunStageV1 | Mapping[str, Any],
) -> tuple[bool, str]:
    if not isinstance(final_verdict, Mapping):
        return False, "REFUSE_SCHEDULE_COMPLETE_WITHOUT_EXACT_FINAL_VERDICT"
    stage_id = final_stage.stage_id if isinstance(final_stage, FullRunStageV1) else str(final_stage["stage_id"])
    target_d_seg = (
        final_stage.target_d_seg if isinstance(final_stage, FullRunStageV1) else float(final_stage["target_d_seg"])
    )
    target_d_pose = (
        final_stage.target_d_pose if isinstance(final_stage, FullRunStageV1) else final_stage.get("target_d_pose")
    )
    if final_verdict.get("stage_id") != stage_id:
        return False, "REFUSE_SCHEDULE_COMPLETE_FINAL_VERDICT_STAGE_MISMATCH"
    try:
        d_seg = float(final_verdict["d_seg"])
        d_pose = float(final_verdict["d_pose"])
    except (KeyError, TypeError, ValueError):
        return False, "REFUSE_SCHEDULE_COMPLETE_FINAL_VERDICT_INVALID"
    if (
        not math.isfinite(d_seg)
        or d_seg < 0.0
        or not math.isfinite(d_pose)
        or d_pose < 0.0
        or not math.isfinite(float(target_d_seg))
        or float(target_d_seg) < 0.0
        or (target_d_pose is not None and (not math.isfinite(float(target_d_pose)) or float(target_d_pose) < 0.0))
    ):
        return False, "REFUSE_SCHEDULE_COMPLETE_FINAL_VERDICT_INVALID"
    if d_seg > float(target_d_seg) or (target_d_pose is not None and d_pose > float(target_d_pose)):
        return False, "REFUSE_SCHEDULE_COMPLETE_FINAL_TARGET_UNMET"
    return True, "FULL_RUN_SCHEDULE_COMPLETE_EXACT_FINAL_TARGETS_MET"


@dataclass(frozen=True, slots=True)
class FullRunStageV1:
    stage_id: str
    active_groups: tuple[str, ...]
    maximum_steps: int
    verdict_interval_steps: int
    target_d_seg: float
    target_d_pose: float | None


@dataclass(frozen=True, slots=True)
class VerdictShadowPolicyV1:
    """Typed L3 apparatus contract for dual-shadow scheduled verdicts."""

    schema: str
    decision_shadow: str
    export_shadow: str
    emit_dual_rows: bool
    same_shadow_reference_required: bool
    minimum_realized_parameter_count_delta: int
    maximum_consecutive_degenerate_verdicts: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> VerdictShadowPolicyV1:
        if not isinstance(payload, Mapping):
            raise DirectDescriptionError("verdict shadow policy must be a mapping")
        result = cls(
            schema=str(payload["schema"]),
            decision_shadow=_canonical_parameter_shadow(payload["decision_shadow"]),
            export_shadow=_canonical_parameter_shadow(payload["export_shadow"]),
            emit_dual_rows=bool(payload["emit_dual_rows"]),
            same_shadow_reference_required=bool(payload["same_shadow_reference_required"]),
            minimum_realized_parameter_count_delta=int(payload["minimum_realized_parameter_count_delta"]),
            maximum_consecutive_degenerate_verdicts=int(payload["maximum_consecutive_degenerate_verdicts"]),
        )
        if (
            result.schema != "ddm_dual_shadow_verdict_policy.v1"
            or result.decision_shadow != "live"
            or result.export_shadow != "ema"
            or not result.emit_dual_rows
            or not result.same_shadow_reference_required
            or result.minimum_realized_parameter_count_delta != 1
            or result.maximum_consecutive_degenerate_verdicts != 1
        ):
            raise DirectDescriptionError("verdict shadow policy differs from the J10 L3 apparatus contract")
        return result

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "decision_shadow": self.decision_shadow,
            "export_shadow": self.export_shadow,
            "emit_dual_rows": self.emit_dual_rows,
            "same_shadow_reference_required": self.same_shadow_reference_required,
            "minimum_realized_parameter_count_delta": self.minimum_realized_parameter_count_delta,
            "maximum_consecutive_degenerate_verdicts": self.maximum_consecutive_degenerate_verdicts,
        }


@dataclass(frozen=True, slots=True)
class PoseFinishEngageConfigV1:
    """Typed #383-style conditioning latch over exact same-vehicle verdicts.

    DDM does not expose the level-set vehicle's ξ→PoseNet Jacobian σ_min
    series.  Reusing that metric would therefore fabricate telemetry.  This
    sibling detector reuses #383's derived EMA/window/hysteresis/flat-band
    constants, but applies them to the run-owned exact n600 d_seg trajectory.
    It is deliberately slower to engage than the former one-bit
    ``first Seg admission`` switch.
    """

    schema: str
    detector: str
    metric: str
    ema_span: int
    settle_window: int
    hysteresis: int
    flat_relative_slope_band: float
    minimum_strict_seg_admissions: int
    fallback_policy: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PoseFinishEngageConfigV1:
        if not isinstance(payload, Mapping):
            raise DirectDescriptionError("pose-finish engage config must be a mapping")
        result = cls(
            schema=str(payload["schema"]),
            detector=str(payload["detector"]),
            metric=str(payload["metric"]),
            ema_span=int(payload["ema_span"]),
            settle_window=int(payload["settle_window"]),
            hysteresis=int(payload["hysteresis"]),
            flat_relative_slope_band=float(payload["flat_relative_slope_band"]),
            minimum_strict_seg_admissions=int(payload["minimum_strict_seg_admissions"]),
            fallback_policy=str(payload["fallback_policy"]),
        )
        if result.schema != POSE_FINISH_ENGAGE_CONFIG_SCHEMA:
            raise DirectDescriptionError("pose-finish engage schema differs")
        if result.detector != "rolling_relative_slope_plateau":
            raise DirectDescriptionError("pose-finish engage detector differs from the governed mode")
        if result.metric != "exact_n600_d_seg_realized_through_R":
            raise DirectDescriptionError("pose-finish engage metric is not exact run-owned d_seg")
        if (
            result.ema_span != DEFAULT_EMA_SPAN
            or result.settle_window != DEFAULT_SETTLE_WINDOW
            or result.hysteresis != DEFAULT_HYSTERESIS
            or result.flat_relative_slope_band != DEFAULT_FLAT_REL_BAND
        ):
            raise DirectDescriptionError("pose-finish engage constants differ from the #383 derived defaults")
        if result.minimum_strict_seg_admissions < 1:
            raise DirectDescriptionError("pose-finish engage requires a strict Seg admission")
        if result.fallback_policy != "emit_banked_r1_comparator_harvest_signal_non_promoting":
            raise DirectDescriptionError("pose-finish engage fallback policy differs")
        return result

    @property
    def minimum_points(self) -> int:
        return self.settle_window + self.hysteresis - 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "detector": self.detector,
            "metric": self.metric,
            "ema_span": self.ema_span,
            "settle_window": self.settle_window,
            "hysteresis": self.hysteresis,
            "flat_relative_slope_band": self.flat_relative_slope_band,
            "minimum_strict_seg_admissions": self.minimum_strict_seg_admissions,
            "fallback_policy": self.fallback_policy,
        }


@dataclass(frozen=True, slots=True)
class PoseFinishEngageStateV1:
    """Checkpoint payload and monotone pose-finish latch for one DDM run."""

    exact_verdict_steps: tuple[int, ...] = ()
    exact_d_seg: tuple[float, ...] = ()
    strict_seg_admission_steps: tuple[int, ...] = ()
    strict_seg_admissions: int = 0
    engaged_global_step: int | None = None
    classification: str = "INSUFFICIENT_EXACT_VERDICTS"
    latest_relative_slope: float | None = None
    latest_relative_slope_stderr: float | None = None

    def observe(
        self,
        *,
        global_step: int,
        d_seg: float,
        strict_seg_admission: bool,
        config: PoseFinishEngageConfigV1,
    ) -> PoseFinishEngageStateV1:
        step = int(global_step)
        value = float(d_seg)
        if step < 0 or not math.isfinite(value) or value < 0.0:
            raise DirectDescriptionError("pose-finish engage observation is invalid")
        if self.exact_verdict_steps and step <= self.exact_verdict_steps[-1]:
            if step == self.exact_verdict_steps[-1] and value == self.exact_d_seg[-1]:
                return self
            raise DirectDescriptionError("pose-finish exact verdict history is non-monotone")
        steps = (*self.exact_verdict_steps, step)
        values = (*self.exact_d_seg, value)
        strict_steps = (
            (*self.strict_seg_admission_steps, step) if strict_seg_admission else self.strict_seg_admission_steps
        )
        strict = len(strict_steps)
        if self.engaged_global_step is not None:
            return replace(
                self,
                exact_verdict_steps=steps,
                exact_d_seg=values,
                strict_seg_admission_steps=strict_steps,
                strict_seg_admissions=strict,
                classification="POSE_FINISH_ENGAGED_LATCHED",
            )
        classification, slope, stderr = _classify_pose_finish_engage(
            steps=steps,
            values=values,
            strict_seg_admissions=strict,
            config=config,
        )
        engaged = step if classification == "POSE_FINISH_ENGAGED_PLATEAU" else None
        return PoseFinishEngageStateV1(
            exact_verdict_steps=steps,
            exact_d_seg=values,
            strict_seg_admission_steps=strict_steps,
            strict_seg_admissions=strict,
            engaged_global_step=engaged,
            classification=classification,
            latest_relative_slope=slope,
            latest_relative_slope_stderr=stderr,
        )

    @property
    def engaged(self) -> bool:
        return self.engaged_global_step is not None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": POSE_FINISH_ENGAGE_STATE_SCHEMA,
            "exact_verdict_steps": list(self.exact_verdict_steps),
            "exact_d_seg": list(self.exact_d_seg),
            "strict_seg_admission_steps": list(self.strict_seg_admission_steps),
            "strict_seg_admissions": self.strict_seg_admissions,
            "engaged_global_step": self.engaged_global_step,
            "classification": self.classification,
            "latest_relative_slope": self.latest_relative_slope,
            "latest_relative_slope_stderr": self.latest_relative_slope_stderr,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        config: PoseFinishEngageConfigV1,
    ) -> PoseFinishEngageStateV1:
        if payload.get("schema") != POSE_FINISH_ENGAGE_STATE_SCHEMA:
            raise DirectDescriptionError("pose-finish engage checkpoint schema differs")
        steps = tuple(int(value) for value in payload.get("exact_verdict_steps", ()))
        values = tuple(float(value) for value in payload.get("exact_d_seg", ()))
        strict_steps = tuple(int(value) for value in payload.get("strict_seg_admission_steps", ()))
        if len(steps) != len(values) or any(after <= before for before, after in pairwise(steps)):
            raise DirectDescriptionError("pose-finish engage checkpoint history differs")
        if (
            any(step not in steps for step in strict_steps)
            or len(set(strict_steps)) != len(strict_steps)
            or len(strict_steps) != int(payload.get("strict_seg_admissions", 0))
        ):
            raise DirectDescriptionError("pose-finish engage checkpoint strict-admission history differs")
        state = cls()
        for step, value in zip(steps, values, strict=True):
            state = state.observe(
                global_step=step,
                d_seg=value,
                strict_seg_admission=step in strict_steps,
                config=config,
            )
        expected_engaged = payload.get("engaged_global_step")
        if state.engaged_global_step != expected_engaged:
            raise DirectDescriptionError("pose-finish engage checkpoint latch differs on re-derivation")
        if state.classification != payload.get("classification"):
            raise DirectDescriptionError("pose-finish engage checkpoint classification differs on re-derivation")
        return state


def _classify_pose_finish_engage(
    *,
    steps: Sequence[int],
    values: Sequence[float],
    strict_seg_admissions: int,
    config: PoseFinishEngageConfigV1,
) -> tuple[str, float | None, float | None]:
    if strict_seg_admissions < config.minimum_strict_seg_admissions:
        return "WAITING_FOR_STRICT_SEG_ADMISSION", None, None
    if len(values) < config.minimum_points:
        return "INSUFFICIENT_EXACT_VERDICTS", None, None
    alpha = 2.0 / (config.ema_span + 1.0)
    smoothed: list[float] = []
    for value in values:
        smoothed.append(float(value) if not smoothed else alpha * float(value) + (1.0 - alpha) * smoothed[-1])
    flat_windows: list[bool] = []
    latest_slope: float | None = None
    latest_stderr: float | None = None
    for offset in range(config.hysteresis):
        end = len(values) - offset
        start = end - config.settle_window
        xs = [float(value) for value in steps[start:end]]
        ys = smoothed[start:end]
        fit = slope_with_stderr(xs, ys)
        level = sum(ys) / len(ys)
        if level <= 0.0:
            return "DEGENERATE_EXACT_DSEG_HISTORY", None, None
        relative_slope = float(fit.slope / level)
        relative_stderr = None if fit.stderr is None else float(fit.stderr / level)
        if offset == 0:
            latest_slope = relative_slope
            latest_stderr = relative_stderr
        flat_windows.append(abs(relative_slope) <= config.flat_relative_slope_band and relative_slope <= 0.0)
    if latest_stderr is None or latest_stderr > config.flat_relative_slope_band:
        return "DEGENERATE_EXACT_DSEG_HISTORY", latest_slope, latest_stderr
    if all(flat_windows):
        return "POSE_FINISH_ENGAGED_PLATEAU", latest_slope, latest_stderr
    return "DSEG_STILL_TRENDING", latest_slope, latest_stderr


@dataclass(frozen=True, slots=True)
class WorstGeometryMemoryContractV1:
    schema: str
    selected_pair_start: int
    train_batch: int
    active_groups: tuple[str, ...]
    expected_island_secants: int
    expected_lane_secants: int
    expected_total_secants: int
    derived_basis_gib: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> WorstGeometryMemoryContractV1:
        if not isinstance(payload, Mapping):
            raise DirectDescriptionError("worst-geometry memory contract must be a mapping")
        result = cls(
            schema=str(payload["schema"]),
            selected_pair_start=int(payload["selected_pair_start"]),
            train_batch=int(payload["train_batch"]),
            active_groups=tuple(str(value) for value in payload["active_groups"]),
            expected_island_secants=int(payload["expected_island_secants"]),
            expected_lane_secants=int(payload["expected_lane_secants"]),
            expected_total_secants=int(payload["expected_total_secants"]),
            derived_basis_gib=float(payload["derived_basis_gib"]),
        )
        if (
            result.schema != WORST_GEOMETRY_MEMORY_SCHEMA
            or result.selected_pair_start not in {498, 499}
            or result.train_batch != 4
            or result.active_groups != ("island_worldsheet", "lane_program", "shared_template_dof")
            or result.expected_island_secants != 28
            or result.expected_lane_secants != 24
            or result.expected_total_secants != 52
            or not math.isclose(result.derived_basis_gib, 4.72976016998291, rel_tol=0.0, abs_tol=1.0e-12)
        ):
            raise DirectDescriptionError("worst-geometry memory contract differs from the J6 re-derivation")
        return result

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "selected_pair_start": self.selected_pair_start,
            "train_batch": self.train_batch,
            "active_groups": list(self.active_groups),
            "expected_island_secants": self.expected_island_secants,
            "expected_lane_secants": self.expected_lane_secants,
            "expected_total_secants": self.expected_total_secants,
            "derived_basis_gib": self.derived_basis_gib,
        }


@dataclass(frozen=True, slots=True)
class WarmStartReformV1:
    """Derived opening geometry and strict receiver-realized admission."""

    adam_beta2: float
    lr_rewarmup_c: float
    lr_rewarmup_steps: int
    lr_rewarmup_floor: float
    lr_rewarmup_shape: str
    maximum_continuous_update_quantum_fraction: float | None
    frozen_groups_until_first_admission: tuple[str, ...]
    group_release_condition: str
    pose_objective_engage_condition: str | None
    first_realized_admission: str
    realized_acceptance_policy: str
    proposal_staging: str
    proposal_q8_denominator: int
    proposal_multipliers: tuple[float, ...]
    proposal_ordering: str
    opening_active_groups: tuple[str, ...]
    opening_candidate_ids: tuple[str, ...]
    opening_candidate_pair_ids: tuple[int, ...]
    residual_bucket_admission_required: bool
    component_fire_gate: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> WarmStartReformV1:
        if not isinstance(payload, Mapping):
            raise DirectDescriptionError("full-run warm-start reform must be a mapping")
        result = cls(
            adam_beta2=float(payload["adam_beta2"]),
            lr_rewarmup_c=float(payload["lr_rewarmup_c"]),
            lr_rewarmup_steps=int(payload["lr_rewarmup_steps"]),
            lr_rewarmup_floor=float(payload["lr_rewarmup_floor"]),
            lr_rewarmup_shape=str(payload["lr_rewarmup_shape"]),
            maximum_continuous_update_quantum_fraction=(
                None
                if payload.get("maximum_continuous_update_quantum_fraction") is None
                else float(payload["maximum_continuous_update_quantum_fraction"])
            ),
            frozen_groups_until_first_admission=tuple(
                str(value) for value in payload["frozen_groups_until_first_admission"]
            ),
            group_release_condition=str(payload["group_release_condition"]),
            pose_objective_engage_condition=(
                None
                if payload.get("pose_objective_engage_condition") is None
                else str(payload["pose_objective_engage_condition"])
            ),
            first_realized_admission=str(payload["first_realized_admission"]),
            realized_acceptance_policy=str(payload.get("realized_acceptance_policy", "component_safe_exact_n600")),
            proposal_staging=str(payload.get("proposal_staging", "continuous_receiver_wire")),
            proposal_q8_denominator=int(payload.get("proposal_q8_denominator", 1)),
            proposal_multipliers=tuple(float(value) for value in payload.get("proposal_multipliers", (1.0, 0.5, 0.25))),
            proposal_ordering=str(payload.get("proposal_ordering", "candidate_then_shrink")),
            opening_active_groups=tuple(str(value) for value in payload.get("opening_active_groups", ())),
            opening_candidate_ids=tuple(str(value) for value in payload.get("opening_candidate_ids", ())),
            opening_candidate_pair_ids=tuple(int(value) for value in payload.get("opening_candidate_pair_ids", ())),
            residual_bucket_admission_required=bool(payload.get("residual_bucket_admission_required", False)),
            component_fire_gate=str(
                payload.get(
                    "component_fire_gate",
                    "dseg_flat_or_down_pose_not_worse_and_any_component_descends",
                )
            ),
        )
        if not 0.0 < result.adam_beta2 < 1.0 or result.lr_rewarmup_c <= 0.0:
            raise DirectDescriptionError("warm-start Adam beta2/rewarmup multiple is invalid")
        derived_steps = adam_v_variance_warmup_epochs(
            result.adam_beta2,
            1,
            c=result.lr_rewarmup_c,
        )
        if result.lr_rewarmup_steps != derived_steps:
            raise DirectDescriptionError("warm-start rewarmup steps differ from adam_v_variance_warmup_length_v1")
        if not 0.0 < result.lr_rewarmup_floor <= 1.0 or result.lr_rewarmup_shape != "linear":
            raise DirectDescriptionError("warm-start LR rewarmup floor/shape is invalid")
        if result.maximum_continuous_update_quantum_fraction is not None and not (
            0.0 < result.maximum_continuous_update_quantum_fraction <= 0.25
        ):
            raise DirectDescriptionError("warm-start update exceeds the quarter-quantum cap")
        allowed_groups = {"island_worldsheet", "lane_program", "shared_template_dof"}
        if set(result.frozen_groups_until_first_admission) - allowed_groups:
            raise DirectDescriptionError("warm-start frozen group policy is invalid")
        if result.group_release_condition not in {
            "first_strict_n600_island_admission",
            "first_component_safe_n600_residual_admission",
        }:
            raise DirectDescriptionError("warm-start group release condition is invalid")
        if result.pose_objective_engage_condition not in {
            None,
            "after_first_strict_n600_seg_admission",
        }:
            raise DirectDescriptionError("warm-start Pose engage condition is invalid")
        if result.first_realized_admission not in {
            "exact_n600_dseg_descent_and_dpose_nonregression_else_abort_rollback",
            "exact_n600_joint_delta_s_lt_zero_else_shrink_and_exact_rollback",
        }:
            raise DirectDescriptionError("warm-start realized admission law is invalid")
        if result.realized_acceptance_policy not in {
            "component_safe_exact_n600",
            "pure_priced_exact_n600",
            "campaign_component_safe_exact_n600",
        }:
            raise DirectDescriptionError("warm-start realized acceptance policy is invalid")
        if result.proposal_staging not in {
            "continuous_receiver_wire",
            "camera_874x1164_q8_pre_final_uint8",
        }:
            raise DirectDescriptionError("warm-start proposal staging is invalid")
        expected_q8 = 256 if result.proposal_staging == "camera_874x1164_q8_pre_final_uint8" else 1
        if result.proposal_q8_denominator != expected_q8:
            raise DirectDescriptionError("warm-start proposal Q8 denominator differs from staging")
        if (
            not result.proposal_multipliers
            or any(not math.isfinite(value) or value <= 0.0 for value in result.proposal_multipliers)
            or tuple(sorted(result.proposal_multipliers, reverse=True)) != result.proposal_multipliers
        ):
            raise DirectDescriptionError("warm-start proposal shrink ladder is invalid")
        if result.proposal_ordering not in {
            "candidate_then_shrink",
            "seg_lexicographic_proxy_then_exact_component_gate",
        }:
            raise DirectDescriptionError("warm-start proposal ordering is invalid")
        if set(result.opening_active_groups) - allowed_groups:
            raise DirectDescriptionError("warm-start opening active groups are invalid")
        known_candidates = {
            "local_exact_gradient",
            "worldsheet_joint_active_x_+1",
            "worldsheet_joint_active_x_-1",
            "worldsheet_joint_active_y_+1",
            "worldsheet_joint_active_y_-1",
        }
        if set(result.opening_candidate_ids) - known_candidates:
            raise DirectDescriptionError("warm-start opening candidate id is invalid")
        if len(set(result.opening_candidate_pair_ids)) != len(result.opening_candidate_pair_ids) or any(
            pair_id < 0 or pair_id >= 600 for pair_id in result.opening_candidate_pair_ids
        ):
            raise DirectDescriptionError("warm-start opening candidate pair ids are invalid")
        if result.realized_acceptance_policy in {
            "pure_priced_exact_n600",
            "campaign_component_safe_exact_n600",
        }:
            if result.maximum_continuous_update_quantum_fraction is not None:
                raise DirectDescriptionError("receiver-realized warm start must drop the continuous cap")
            if (
                result.first_realized_admission != "exact_n600_joint_delta_s_lt_zero_else_shrink_and_exact_rollback"
                or result.group_release_condition != "first_component_safe_n600_residual_admission"
                or not result.opening_active_groups
                or not result.opening_candidate_ids
                or not result.opening_candidate_pair_ids
                or not result.residual_bucket_admission_required
            ):
                raise DirectDescriptionError("receiver-realized warm-start opening contract is incomplete")
        if (
            result.realized_acceptance_policy == "campaign_component_safe_exact_n600"
            and result.proposal_ordering != "seg_lexicographic_proxy_then_exact_component_gate"
        ):
            raise DirectDescriptionError("campaign-safe warm start requires seg-lexicographic proposal ordering")
        if result.component_fire_gate not in {
            "dseg_flat_or_down_pose_not_worse_and_any_component_descends",
            "cumulative_vs_stage00_dseg_flat_or_down_pose_not_worse_any_component_and_residual_descend",
        }:
            raise DirectDescriptionError("warm-start component fire gate is invalid")
        if (
            result.pose_objective_engage_condition is None
            and result.component_fire_gate
            != "cumulative_vs_stage00_dseg_flat_or_down_pose_not_worse_any_component_and_residual_descend"
        ):
            raise DirectDescriptionError("typed pose engage requires the cumulative stage00 fire gate")
        return result


@dataclass(frozen=True, slots=True)
class FullRunScheduleV1:
    """Hash-sealed schedule consumed only by the real n600 full-run path."""

    train_batch: int
    learning_rate_quantum_fraction: float
    checkpoint_interval_steps: int | None
    plateau_verdicts: int | None
    warm_start_pair: int | None
    warm_start_steps: int | None
    measured_seconds_per_step: float
    measured_seconds_per_step_low: float
    measured_seconds_per_step_high: float
    warm_start_reform: WarmStartReformV1 | None
    pose_finish_engage: PoseFinishEngageConfigV1 | None
    verdict_shadow_policy: VerdictShadowPolicyV1 | None
    stages: tuple[FullRunStageV1, ...]
    event_continuation: Any | None = None

    @classmethod
    def from_semantic_program(cls, semantic: Mapping[str, Any]) -> FullRunScheduleV1 | None:
        payload = semantic.get("full_run_schedule")
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise DirectDescriptionError("full-run schedule must be a mapping")
        event_payload = payload.get("event_graph")
        if event_payload is not None:
            from tac.optimization.ddm_event_continuation import (
                DDMEventContinuationError,
                DDMEventContinuationV1,
            )

            if not isinstance(event_payload, Mapping):
                raise DirectDescriptionError("DDM event graph must be a mapping")
            ambiguous_legacy_keys = {
                "stages",
                "checkpoint_interval_steps",
                "plateau_verdicts",
                "warm_start_pair",
                "warm_start_steps",
                "derived_steps_per_n600_exposure",
                "derived_total_steps",
                "stage_transition_rule",
            }
            present_legacy_keys = ambiguous_legacy_keys & set(payload)
            if present_legacy_keys:
                raise DirectDescriptionError(
                    "event continuation contains ambiguous legacy schedule keys: "
                    + ",".join(sorted(present_legacy_keys))
                )
            try:
                continuation = DDMEventContinuationV1.from_payload(event_payload)
            except DDMEventContinuationError as exc:
                raise DirectDescriptionError(f"DDM event graph is invalid: {exc}") from exc
            result = cls(
                train_batch=int(payload["train_batch"]),
                learning_rate_quantum_fraction=float(payload["learning_rate_quantum_fraction"]),
                checkpoint_interval_steps=None,
                plateau_verdicts=None,
                warm_start_pair=None,
                warm_start_steps=None,
                measured_seconds_per_step=float(payload["measured_seconds_per_step"]),
                measured_seconds_per_step_low=float(payload["measured_seconds_per_step_low"]),
                measured_seconds_per_step_high=float(payload["measured_seconds_per_step_high"]),
                warm_start_reform=(
                    None
                    if payload.get("warm_start_reform") is None
                    else WarmStartReformV1.from_payload(payload["warm_start_reform"])
                ),
                pose_finish_engage=(
                    None
                    if payload.get("pose_finish_engage") is None
                    else PoseFinishEngageConfigV1.from_payload(payload["pose_finish_engage"])
                ),
                verdict_shadow_policy=(
                    None
                    if payload.get("verdict_shadow_policy") is None
                    else VerdictShadowPolicyV1.from_payload(payload["verdict_shadow_policy"])
                ),
                stages=(),
                event_continuation=continuation,
            )
            if not 1 <= result.train_batch <= 600:
                raise DirectDescriptionError("event-continuation train batch is outside n600")
            if not 0.0 < result.learning_rate_quantum_fraction <= 0.25:
                raise DirectDescriptionError(
                    "event-continuation learning rate exceeds the realized uint8 quarter-quantum bound"
                )
            timings = (
                result.measured_seconds_per_step_low,
                result.measured_seconds_per_step,
                result.measured_seconds_per_step_high,
            )
            if not all(math.isfinite(value) and value > 0.0 for value in timings) or not (
                timings[0] <= timings[1] <= timings[2]
            ):
                raise DirectDescriptionError("event-continuation measured timing band is invalid")
            return result
        stages_raw = payload.get("stages")
        if not isinstance(stages_raw, list) or not stages_raw:
            raise DirectDescriptionError("full-run schedule requires nonempty stages")
        stages = tuple(
            FullRunStageV1(
                stage_id=str(row["stage_id"]),
                active_groups=tuple(str(value) for value in row["active_groups"]),
                maximum_steps=int(row["maximum_steps"]),
                verdict_interval_steps=int(row["verdict_interval_steps"]),
                target_d_seg=float(row["target_d_seg"]),
                target_d_pose=None if row.get("target_d_pose") is None else float(row["target_d_pose"]),
            )
            for row in stages_raw
        )
        result = cls(
            train_batch=int(payload["train_batch"]),
            learning_rate_quantum_fraction=float(payload["learning_rate_quantum_fraction"]),
            checkpoint_interval_steps=int(payload["checkpoint_interval_steps"]),
            plateau_verdicts=int(payload["plateau_verdicts"]),
            warm_start_pair=int(payload["warm_start_pair"]),
            warm_start_steps=int(payload["warm_start_steps"]),
            measured_seconds_per_step=float(payload["measured_seconds_per_step"]),
            measured_seconds_per_step_low=float(payload["measured_seconds_per_step_low"]),
            measured_seconds_per_step_high=float(payload["measured_seconds_per_step_high"]),
            warm_start_reform=(
                None
                if payload.get("warm_start_reform") is None
                else WarmStartReformV1.from_payload(payload["warm_start_reform"])
            ),
            pose_finish_engage=(
                None
                if payload.get("pose_finish_engage") is None
                else PoseFinishEngageConfigV1.from_payload(payload["pose_finish_engage"])
            ),
            verdict_shadow_policy=(
                None
                if payload.get("verdict_shadow_policy") is None
                else VerdictShadowPolicyV1.from_payload(payload["verdict_shadow_policy"])
            ),
            stages=stages,
            event_continuation=None,
        )
        if not 1 <= result.train_batch <= 600:
            raise DirectDescriptionError("full-run train batch is outside n600")
        if not 0.0 < result.learning_rate_quantum_fraction <= 0.25:
            raise DirectDescriptionError("full-run learning rate exceeds the realized uint8 quarter-quantum bound")
        if result.checkpoint_interval_steps <= 0 or result.plateau_verdicts <= 0:
            raise DirectDescriptionError("full-run checkpoint/plateau schedule is invalid")
        if not 0 <= result.warm_start_pair < 600:
            raise DirectDescriptionError("full-run warm-start pair is outside n600")
        if result.warm_start_steps <= 0:
            raise DirectDescriptionError("full-run warm-start steps are invalid")
        timings = (
            result.measured_seconds_per_step_low,
            result.measured_seconds_per_step,
            result.measured_seconds_per_step_high,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in timings) or not (
            timings[0] <= timings[1] <= timings[2]
        ):
            raise DirectDescriptionError("full-run measured timing band is invalid")
        allowed_groups = {"island_worldsheet", "lane_program", "shared_template_dof"}
        if any(
            stage.maximum_steps <= 0
            or stage.verdict_interval_steps <= 0
            or stage.verdict_interval_steps > stage.maximum_steps
            or stage.maximum_steps % stage.verdict_interval_steps != 0
            or not set(stage.active_groups) <= allowed_groups
            or not 0.0 <= stage.target_d_seg <= BASELINE_DSEG
            or (stage.target_d_pose is not None and stage.target_d_pose < 0.0)
            for stage in result.stages
        ):
            raise DirectDescriptionError("full-run stage schedule is invalid")
        if (
            result.warm_start_reform is not None
            and result.warm_start_reform.pose_objective_engage_condition is None
            and result.pose_finish_engage is None
        ):
            raise DirectDescriptionError("typed pose-finish engage config is required when the legacy string is absent")
        return result


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bound_file(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise DirectDescriptionError(f"bound regular file is unavailable: {path}")
    payload = path.read_bytes()
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise DirectDescriptionError(f"bound file byte count differs for {path}: {len(payload)} != {expected_bytes}")
    actual = _sha256(payload)
    if actual != expected_sha256:
        raise DirectDescriptionError(f"bound file sha256 differs for {path}: {actual} != {expected_sha256}")
    return payload


def _validate_execution_custody(payload: Mapping[str, Any]) -> None:
    """Validate the non-semantic artifact bindings without creating a hash cycle.

    Source bytes and the newly measured memory receipt are sealed at the ticket
    top level after the semantic DSL hash is known.  They intentionally do not
    enter ``typed_config_hash``: the consumer source embeds the semantic hash,
    while the memory receipt embeds the typed hash.
    """

    def binding(row: Any, *, allow_pending_sha: bool = False) -> None:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str) or not row["path"]:
            raise DirectDescriptionError("execution custody binding lacks a path")
        digest = row.get("sha256")
        if digest is None and allow_pending_sha:
            return
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise DirectDescriptionError("execution custody binding lacks a canonical SHA-256")

    sources = payload.get("source_files")
    artifacts = payload.get("j5_producer_artifacts")
    memory = payload.get("worst_geometry_memory_receipt")
    if not isinstance(sources, Mapping) or set(sources) != {"consumer", "launcher"}:
        raise DirectDescriptionError("execution custody source-file set differs")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "baseline_verdict",
        "proposal_verdict",
        "runtime_memory_receipt",
        "checkpoint",
    }:
        raise DirectDescriptionError("execution custody J5 artifact set differs")
    for row in sources.values():
        binding(row)
    for row in artifacts.values():
        binding(row)
    binding(memory, allow_pending_sha=True)
    comparator = payload.get("banked_r1_comparator")
    if not isinstance(comparator, Mapping) or comparator != {
        "d_pose": 0.00161,
        "score_contribution": 0.127,
        "payload_bytes": 7200,
        "authority": "comparator_and_fallback_harvest_signal_only",
        "binding_target": False,
        "promotion_eligible": False,
    }:
        raise DirectDescriptionError("banked R1 comparator custody differs")


def _validate_j10_ema_compile(
    *,
    semantic: Mapping[str, Any],
    resolved_ema_decay: float,
) -> None:
    """Re-resolve the J10 EMA DSL LawRef and its receiver-quantum geometry."""

    compile_payload = semantic.get("joint_objective", {}).get("ema_decay_compile")
    if not isinstance(compile_payload, Mapping):
        raise DirectDescriptionError("J10 EMA decay lacks DSL compile custody")
    geometry = compile_payload.get("run_geometry")
    declaration = compile_payload.get("lawref_declaration")
    manifest = compile_payload.get("constant_manifest")
    if not isinstance(geometry, Mapping) or not isinstance(declaration, Mapping) or not isinstance(manifest, Mapping):
        raise DirectDescriptionError("J10 EMA DSL compile payload is incomplete")
    expected_geometry = {
        "original_total_updates": 450,
        "remaining_updates_after_materialized_step50": 400,
        "stage_updates": [150, 150, 150],
        "remaining_stage_updates": [100, 150, 150],
        "first_scheduled_verdict_updates": 50,
        "strict_equilibration_budget_updates": 49,
        "equilibration_time_constants": 3,
        "learning_rate_quantum_fraction": 0.25,
        "historical_named_parameter_count": 706,
        "receiver_effective_parameter_count": 368,
        "minimum_receiver_quantum_by_first_verdict": 1,
    }
    if dict(geometry) != expected_geometry:
        raise DirectDescriptionError("J10 EMA run geometry differs from the sealed derivation")
    from tac.witness_dsl.lawref import (
        lawref_from_declaration,
        resolve,
    )

    ref = lawref_from_declaration(declaration)
    resolved = resolve(ref, repo_root=Path(__file__).resolve().parents[3])
    if (
        ref.equation_id != "ema_decay_run_geometry_v1"
        or resolved.fallback_used
        or manifest.get("equation_id") != ref.equation_id
        or manifest.get("fallback_used") is not False
        or manifest.get("ladder_class") != "derived_at_config"
        or not math.isclose(float(manifest.get("value", math.nan)), float(resolved.value), rel_tol=0.0, abs_tol=1.0e-15)
        or not math.isclose(float(resolved.value), float(resolved_ema_decay), rel_tol=0.0, abs_tol=1.0e-15)
    ):
        raise DirectDescriptionError("J10 EMA LawRef resolution differs from the typed config")
    horizon = geometry["equilibration_time_constants"] / (1.0 - float(resolved.value))
    blend_at_first_verdict = 1.0 - float(resolved.value) ** geometry["first_scheduled_verdict_updates"]
    if (
        not math.isclose(
            horizon,
            float(geometry["strict_equilibration_budget_updates"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or horizon >= float(geometry["first_scheduled_verdict_updates"])
        or blend_at_first_verdict < 0.5
    ):
        raise DirectDescriptionError("J10 EMA decay does not clear the first-verdict receiver-quantum geometry")


def _validate_j10_materialized_warm_start(warm: Mapping[str, Any]) -> None:
    """Re-derive the materialized archive and measured-baseline custody."""

    try:
        archive_path = Path(str(warm["path"]))
        receipt_path = Path(str(warm["receipt_path"]))
        verdict_path = Path(str(warm["baseline_verdict_path"]))
    except KeyError as exc:
        raise DirectDescriptionError("J10 materialized source custody is incomplete") from exc
    for label, path in (
        ("archive", archive_path),
        ("materialization receipt", receipt_path),
        ("baseline verdict", verdict_path),
    ):
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise DirectDescriptionError(f"J10 materialized {label} is unavailable")

    receipt_bytes = receipt_path.read_bytes()
    verdict_bytes = verdict_path.read_bytes()
    if _sha256(receipt_bytes) != warm.get("receipt_sha256"):
        raise DirectDescriptionError("J10 materialization receipt SHA-256 differs")
    if _sha256(verdict_bytes) != warm.get("baseline_verdict_sha256"):
        raise DirectDescriptionError("J10 materialized baseline verdict SHA-256 differs")
    try:
        receipt = json.loads(receipt_bytes)
        verdict = json.loads(verdict_bytes)
    except (TypeError, ValueError) as exc:
        raise DirectDescriptionError("J10 materialized source receipt is not valid JSON") from exc

    archive_sha256 = str(warm["sha256"])
    archive_bytes = int(warm["bytes"])
    materialized = receipt.get("materialized_archive")
    source_checkpoint = receipt.get("source_checkpoint")
    if (
        receipt.get("schema") != "ddm_j10_step50_live_warm_start_materialization.v1"
        or not isinstance(materialized, Mapping)
        or not isinstance(source_checkpoint, Mapping)
        or materialized.get("path") != str(archive_path)
        or materialized.get("sha256") != archive_sha256
        or int(materialized.get("bytes", -1)) != archive_bytes
        or materialized.get("candidate") != "W_joint"
        or materialized.get("receiver_parseback_identity") is not True
        or materialized.get("fresh_zero_state_reemit_identity") is not True
        or int(materialized.get("live_realized_parameter_count", -1)) != 119
        or receipt.get("fresh_ema_anchored_at_materialized_live_state") is not True
        or receipt.get("optimizer_arrays_preserved_but_not_loaded") is not True
        or int(source_checkpoint.get("global_step", -1)) != 50
        or source_checkpoint.get("sha256") != "043c2a8b3c89688510cc0ff002f37a375a974205a5f8760d93133c47b7cec7c1"
    ):
        raise DirectDescriptionError("J10 materialization receipt binding differs")

    try:
        baseline_d_seg = float(warm["baseline_d_seg"])
        baseline_d_pose = float(warm["baseline_d_pose"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DirectDescriptionError("J10 materialized source lacks measured baseline custody") from exc
    if (
        verdict.get("schema") != "ddm_materialized_warm_start_n600_verdict.v1"
        or verdict.get("archive_path") != str(archive_path)
        or verdict.get("archive_sha256") != archive_sha256
        or int(verdict.get("archive_bytes", -1)) != archive_bytes
        or verdict.get("parameter_shadow") != "live_materialized_step50"
        or verdict.get("decision_authority") != "BASELINE_CUSTODY_ONLY"
        or verdict.get("receiver_parseback_identity") is not True
        or verdict.get("score_claim") is not False
        or int(verdict.get("num_pairs", -1)) != 600
        or int(verdict.get("batch_size", -1)) != 32
        or not math.isclose(float(verdict.get("d_seg", math.nan)), baseline_d_seg, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(float(verdict.get("d_pose", math.nan)), baseline_d_pose, rel_tol=0.0, abs_tol=0.0)
    ):
        raise DirectDescriptionError("J10 materialized baseline verdict binding differs")


@dataclass(frozen=True, slots=True)
class DirectDescriptionJointDescentTypedConfigV1:
    """Executable typed projection of the hash-sealed J1 semantic program."""

    ticket_path: str
    semantic_program: Mapping[str, Any]
    dsl_compile_hash: str
    source_archive_path: str
    source_archive_sha256: str
    source_archive_bytes: int
    source_baseline_d_seg: float | None
    source_baseline_d_pose: float | None
    target_cache_path: str
    target_cache_sha256: str
    target_cache_bytes: int
    upstream_root: str
    num_pairs: int
    seed: int
    verdict_batch: int
    ema_decay: float
    grad_clip: float
    memory_ceiling_gib: float
    custom_grouped_backward_required: bool
    fused_r_required: bool
    full_run_schedule: FullRunScheduleV1 | None
    worst_geometry_memory_contract: WorstGeometryMemoryContractV1 | None
    execution_custody: Mapping[str, Any] | None
    score_claim: bool = False
    research_only: bool = True

    @classmethod
    def from_ticket(cls, ticket_path: Path) -> DirectDescriptionJointDescentTypedConfigV1:
        ticket_payload = ticket_path.read_bytes()
        ticket = json.loads(ticket_payload)
        if ticket.get("schema") != TICKET_SCHEMA:
            raise DirectDescriptionError("joint-descent ticket schema is not canonical")
        semantic = ticket.get("semantic_program")
        if not isinstance(semantic, dict):
            raise DirectDescriptionError("joint-descent ticket lacks a semantic program")
        semantic_hash = _sha256(rfc8785_canonicalize(semantic))
        sealed = ticket.get("compile_custody", {}).get("semantic_program_sha256")
        schedule_payload = semantic.get("full_run_schedule")
        event_payload = schedule_payload.get("event_graph") if isinstance(schedule_payload, Mapping) else None
        event_program = isinstance(event_payload, Mapping) and event_payload.get("schema") == "DDMEventContinuationV1"
        j10_program = (
            semantic.get("program_id") == J10_PROGRAM_ID
            and isinstance(schedule_payload, Mapping)
            and isinstance(schedule_payload.get("verdict_shadow_policy"), Mapping)
        )
        if semantic_hash != sealed or (
            semantic_hash not in SUPPORTED_PROGRAM_SHA256 and not event_program and not j10_program
        ):
            raise DirectDescriptionError(f"joint-descent DSL hash mismatch: computed={semantic_hash} sealed={sealed}")
        if event_program:
            compile_target = ticket.get("compile_custody", {}).get("typed_target")
            if compile_target != "DDMWitnessProgramV1":
                raise DirectDescriptionError("event continuation lacks DDMWitnessProgramV1 compile custody")
            if event_payload.get("execution_allowed") is not False:
                raise DirectDescriptionError("unreviewed DDM event continuation must be execution-disabled")
            witness_program = semantic.get("ddm_witness_program")
            if not isinstance(witness_program, Mapping) or (witness_program.get("schema") != "DDMWitnessProgramV1"):
                raise DirectDescriptionError("event continuation lacks its embedded typed WitnessProgram")
            if witness_program.get("program_id") != semantic.get("program_id") or (
                witness_program.get("event_continuation") != event_payload
            ):
                raise DirectDescriptionError("event continuation and embedded WitnessProgram custody differ")
            if (
                witness_program.get("execution_allowed") is not False
                or witness_program.get("op_gc1_5_execution_enabled") is not False
                or witness_program.get("inference_shadow") != "ema"
                or witness_program.get("ema_decay") != 0.997
                or witness_program.get("beta2") != 0.999
            ):
                raise DirectDescriptionError("event WitnessProgram optimizer or execution contract differs")
            source_bindings = witness_program.get("source_bindings")
            if not isinstance(source_bindings, Mapping) or set(source_bindings) != {
                "launcher",
                "consumer",
                "event_engine",
                "dm4_adapter",
                "dm4_constructor",
            }:
                raise DirectDescriptionError("event WitnessProgram source-binding set differs")
            if any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                for digest in source_bindings.values()
            ):
                raise DirectDescriptionError("event WitnessProgram source binding lacks canonical SHA-256")
        warm = semantic["warm_start"]
        cache = semantic["target_cache"]
        stability = semantic["joint_objective"]["stability"]
        compute = semantic["compute_contract"]["baseline"]
        worst_geometry_payload = semantic.get("worst_geometry_memory_preflight")
        execution_custody = ticket.get("execution_custody")
        if int(semantic["num_pairs"]) != 600 or int(semantic["seed"]) != 0:
            raise DirectDescriptionError("joint-descent ticket must remain n600/seed0")
        if (
            event_program
            or j10_program
            or semantic_hash
            in {
                J7_W_SEG_PROGRAM_SHA256,
                J7_W_JOINT_PROGRAM_SHA256,
                J9_W_JOINT_PROGRAM_SHA256,
                WS3_W_SEG_PROGRAM_SHA256,
            }
        ):
            expected_warm_kind = (
                "receiver_closed_checkpoint_materialization" if j10_program else "receiver_closed_ws1_archive"
            )
            if warm.get("kind") != expected_warm_kind:
                raise DirectDescriptionError("J7 WS1 warm start lacks its receiver-closed kind")
            warm_path = Path(str(warm["path"]))
            if not warm_path.is_absolute() or not warm_path.is_file() or warm_path.is_symlink():
                raise DirectDescriptionError("J7 WS1 warm-start archive is unavailable")
            warm_archive = warm_path.read_bytes()
            if len(warm_archive) != int(warm["bytes"]) or _sha256(warm_archive) != warm["sha256"]:
                raise DirectDescriptionError("J7 WS1 warm-start archive custody differs")
            parsed_warm = parse_ws1_warm_start_archive(warm_archive)
            expected_candidate = (
                "W_seg" if semantic_hash in {J7_W_SEG_PROGRAM_SHA256, WS3_W_SEG_PROGRAM_SHA256} else "W_joint"
            )
            if parsed_warm.candidate != expected_candidate or parsed_warm.exact_reemit() != warm_archive:
                raise DirectDescriptionError("J7 WS1 warm-start receiver identity differs")
        elif warm["sha256"] != EXPECTED_ARCHIVE_SHA256 or int(warm["bytes"]) != EXPECTED_ARCHIVE_BYTES:
            raise DirectDescriptionError("joint-descent warm-start identity drifted")
        env = compute.get("environment", {})
        required_kernels = " ".join(str(value) for value in compute.get("required_kernels", ())).lower()
        verdict_batch = int(semantic.get("telemetry", {}).get("verdict_batch", 16))
        if verdict_batch not in {16, 32}:
            raise DirectDescriptionError("joint-descent verdict batch is not a sealed supported geometry")
        if (semantic_hash in J7_PROGRAM_SHA256S or event_program or j10_program) and verdict_batch != 32:
            raise DirectDescriptionError("J7 exact n600 scorer verdicts require batch32")
        if semantic_hash == J6A_PROGRAM_SHA256 or semantic_hash in J7_PROGRAM_SHA256S or j10_program:
            if worst_geometry_payload is None or not isinstance(execution_custody, Mapping):
                raise DirectDescriptionError("J6A ticket lacks worst-geometry or execution custody")
            _validate_execution_custody(execution_custody)
        ema_decay = float(semantic["joint_objective"]["ema_decay"])
        if j10_program:
            _validate_j10_materialized_warm_start(warm)
            _validate_j10_ema_compile(
                semantic=semantic,
                resolved_ema_decay=ema_decay,
            )
            if schedule_payload.get("derived_total_steps") != 400:
                raise DirectDescriptionError("J10 remaining schedule does not preserve exactly 400 updates")
            if [int(row["maximum_steps"]) for row in schedule_payload["stages"]] != [100, 150, 150]:
                raise DirectDescriptionError("J10 remaining per-stage schedule differs")
            if not all(int(row["verdict_interval_steps"]) == 50 for row in schedule_payload["stages"]):
                raise DirectDescriptionError("J10 verdict cadence differs")
        return cls(
            ticket_path=str(ticket_path),
            semantic_program=semantic,
            dsl_compile_hash=semantic_hash,
            source_archive_path=str(warm["path"]),
            source_archive_sha256=str(warm["sha256"]),
            source_archive_bytes=int(warm["bytes"]),
            source_baseline_d_seg=(None if warm.get("baseline_d_seg") is None else float(warm["baseline_d_seg"])),
            source_baseline_d_pose=(None if warm.get("baseline_d_pose") is None else float(warm["baseline_d_pose"])),
            target_cache_path=str(cache["path"]),
            target_cache_sha256=str(cache["sha256"]),
            target_cache_bytes=int(cache["bytes"]),
            upstream_root=str(Path(ticket["authority"]["delegation_prompt_path"]).parents[3] / "upstream"),
            num_pairs=600,
            seed=0,
            verdict_batch=verdict_batch,
            ema_decay=ema_decay,
            grad_clip=float(stability["grad_clip"]),
            memory_ceiling_gib=116.0,
            custom_grouped_backward_required=env.get("TAC_MLX_CUSTOM_GROUPED_BACKWARD") == "1",
            fused_r_required="fused differentiable-r" in required_kernels,
            full_run_schedule=FullRunScheduleV1.from_semantic_program(semantic),
            worst_geometry_memory_contract=(
                None
                if worst_geometry_payload is None
                else WorstGeometryMemoryContractV1.from_payload(worst_geometry_payload)
            ),
            execution_custody=execution_custody,
        )

    def identity_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": TYPED_SCHEMA,
            "dsl_compile_hash": self.dsl_compile_hash,
            "source_archive_sha256": self.source_archive_sha256,
            "target_cache_sha256": self.target_cache_sha256,
            "num_pairs": self.num_pairs,
            "seed": self.seed,
            "verdict_batch": self.verdict_batch,
            "ema_decay": self.ema_decay,
            "grad_clip": self.grad_clip,
            "memory_ceiling_gib": self.memory_ceiling_gib,
            "custom_grouped_backward_required": self.custom_grouped_backward_required,
            "fused_r_required": self.fused_r_required,
            "score_claim": False,
            "research_only": True,
        }
        if self.source_baseline_d_seg is not None or self.source_baseline_d_pose is not None:
            payload["source_baseline"] = {
                "d_seg": self.source_baseline_d_seg,
                "d_pose": self.source_baseline_d_pose,
            }
        if self.worst_geometry_memory_contract is not None:
            payload["worst_geometry_memory_contract"] = self.worst_geometry_memory_contract.to_payload()
        if self.full_run_schedule is not None:
            schedule_payload: dict[str, Any] = {
                "train_batch": self.full_run_schedule.train_batch,
                "learning_rate_quantum_fraction": self.full_run_schedule.learning_rate_quantum_fraction,
                "measured_seconds_per_step": self.full_run_schedule.measured_seconds_per_step,
                "measured_seconds_per_step_low": self.full_run_schedule.measured_seconds_per_step_low,
                "measured_seconds_per_step_high": self.full_run_schedule.measured_seconds_per_step_high,
            }
            if self.full_run_schedule.event_continuation is not None:
                schedule_payload["event_graph"] = self.full_run_schedule.event_continuation.to_payload()
            else:
                schedule_payload.update(
                    {
                        "checkpoint_interval_steps": self.full_run_schedule.checkpoint_interval_steps,
                        "plateau_verdicts": self.full_run_schedule.plateau_verdicts,
                        "warm_start_pair": self.full_run_schedule.warm_start_pair,
                        "warm_start_steps": self.full_run_schedule.warm_start_steps,
                        "stages": [
                            {
                                "stage_id": stage.stage_id,
                                "active_groups": list(stage.active_groups),
                                "maximum_steps": stage.maximum_steps,
                                "verdict_interval_steps": stage.verdict_interval_steps,
                                "target_d_seg": stage.target_d_seg,
                                "target_d_pose": stage.target_d_pose,
                            }
                            for stage in self.full_run_schedule.stages
                        ],
                    }
                )
            if self.full_run_schedule.warm_start_reform is not None:
                reform = self.full_run_schedule.warm_start_reform
                reform_payload: dict[str, Any] = {
                    "adam_beta2": reform.adam_beta2,
                    "lr_rewarmup_c": reform.lr_rewarmup_c,
                    "lr_rewarmup_steps": reform.lr_rewarmup_steps,
                    "lr_rewarmup_floor": reform.lr_rewarmup_floor,
                    "lr_rewarmup_shape": reform.lr_rewarmup_shape,
                    "maximum_continuous_update_quantum_fraction": (reform.maximum_continuous_update_quantum_fraction),
                    "frozen_groups_until_first_admission": list(reform.frozen_groups_until_first_admission),
                    "group_release_condition": reform.group_release_condition,
                    "first_realized_admission": reform.first_realized_admission,
                }
                if reform.pose_objective_engage_condition is not None:
                    reform_payload["pose_objective_engage_condition"] = reform.pose_objective_engage_condition
                # Preserve the canonical typed hashes of historical J4 tickets.
                # J5 fields are additive and enter identity only for the new
                # pure-priced policy.
                if reform.realized_acceptance_policy in {
                    "pure_priced_exact_n600",
                    "campaign_component_safe_exact_n600",
                }:
                    reform_payload.update(
                        {
                            "realized_acceptance_policy": reform.realized_acceptance_policy,
                            "proposal_staging": reform.proposal_staging,
                            "proposal_q8_denominator": reform.proposal_q8_denominator,
                            "proposal_multipliers": list(reform.proposal_multipliers),
                            "opening_active_groups": list(reform.opening_active_groups),
                            "opening_candidate_ids": list(reform.opening_candidate_ids),
                            "opening_candidate_pair_ids": list(reform.opening_candidate_pair_ids),
                            "residual_bucket_admission_required": reform.residual_bucket_admission_required,
                            "component_fire_gate": reform.component_fire_gate,
                        }
                    )
                    if reform.proposal_ordering != "candidate_then_shrink":
                        reform_payload["proposal_ordering"] = reform.proposal_ordering
                schedule_payload["warm_start_reform"] = reform_payload
            if self.full_run_schedule.pose_finish_engage is not None:
                schedule_payload["pose_finish_engage"] = self.full_run_schedule.pose_finish_engage.to_payload()
            if self.full_run_schedule.verdict_shadow_policy is not None:
                schedule_payload["verdict_shadow_policy"] = self.full_run_schedule.verdict_shadow_policy.to_payload()
            payload["full_run_schedule"] = schedule_payload
        return payload

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.identity_payload()))

    def dm4_j5_proposal_source(
        self,
        *,
        base_archive: bytes,
        enabled: bool,
    ) -> tuple[bytes, tuple[Any, ...], dict[str, Any]]:
        """Load the hash-bound scorer-recursive DM4 source through the J5 consumer."""

        proposal_sources = self.semantic_program.get("proposal_sources")
        if not isinstance(proposal_sources, Mapping):
            raise DirectDescriptionError("J5 consumer lacks typed proposal sources")
        binding = proposal_sources.get("dm4_scorer_recursive")
        if not isinstance(binding, Mapping):
            raise DirectDescriptionError("J5 consumer lacks its DM4 proposal source")
        if binding.get("adapter") != ("tac.optimization.ddm_dm4_j5_adapter.adapt_dm4_proposals"):
            raise DirectDescriptionError("J5 consumer DM4 adapter identity differs")
        if binding.get("application_authority") != ("fail_closed_until_counted_J5_application_operator_exists"):
            raise DirectDescriptionError("J5 consumer DM4 application authority differs")
        path_value = binding.get("path")
        digest = binding.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            raise DirectDescriptionError("J5 consumer DM4 receipt path is absent")
        if not isinstance(digest, str) or len(digest) != 64:
            raise DirectDescriptionError("J5 consumer DM4 receipt SHA-256 is absent")
        receipt_path = Path(path_value)
        if not receipt_path.is_absolute():
            receipt_path = Path(self.ticket_path).resolve().parents[3] / receipt_path
        from tac.optimization.ddm_dm4_j5_adapter import (
            DM4J5AdapterError,
            adapt_dm4_proposals,
        )

        try:
            return adapt_dm4_proposals(
                receipt_path=receipt_path,
                receipt_sha256=digest,
                base_archive=base_archive,
                enabled=enabled,
            )
        except DM4J5AdapterError as exc:
            raise DirectDescriptionError(f"J5 consumer DM4 proposal source refused: {exc}") from exc

    def j11_opening_proposal_decomposition_source(
        self,
        *,
        audit_config_path: Path,
    ) -> dict[str, Any]:
        """Audit the J11 component source without changing J10 admission semantics.

        The returned receipt may contain zero proposals.  In particular, a
        COMPLETE scorer-output metric bundle is not silently promoted into a
        receiver-coordinate null-space projector.
        """

        from tac.optimization.ddm_j11_opening_proposal_decomposition import (
            J11ProposalDecompositionError,
            build_refusal_receipt,
        )

        try:
            return build_refusal_receipt(
                typed_descent=self,
                audit_config_path=audit_config_path,
            )
        except J11ProposalDecompositionError as exc:
            raise DirectDescriptionError(
                f"J11 opening proposal decomposition source refused: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class LaneProgramSeedV1:
    """Counted-on-activation Lane seed recovered from an inherited coherent slot.

    Stage 00 keeps these encode-side records inactive so the V15 archive stays
    byte-identical.  Before a Lane coordinate becomes trainable, the complete
    record is emitted through :class:`LanePeriodicProgramV1`, making its range
    gate, BEV polynomial, width, and dash-comb phase counted and receiver-used.
    """

    line_index: int
    birth_pair: int
    death_pair_exclusive: int
    bev_coefficients: tuple[float, float, float, float]
    width_bias: float
    width_slope: float
    dash_phase_origin: float
    dash_phase_xi_gain: float
    range_gate_forward_max_m: float
    activation_rule: str = "emit_complete_record_before_first_lane_gradient"

    def counted_record(self) -> LanePeriodicProgramV1:
        return LanePeriodicProgramV1(
            line_index=self.line_index,
            birth_pair=self.birth_pair,
            death_pair_exclusive=self.death_pair_exclusive,
            dash_phase_origin_delta_q8=0,
            dash_phase_xi_gain_q8=int(np.clip(np.rint(self.dash_phase_xi_gain * 256.0), -32768, 32767)),
            width_bias_q8=0,
            width_slope_q12=0,
        )


def derive_lane_program_seeds(receiver: CarrierComposeReceiverV1) -> tuple[LaneProgramSeedV1, ...]:
    lane = next((row for row in receiver.layers if row.role == "Lane"), None)
    if lane is None or lane.lane_lines is None:
        raise DirectDescriptionError("joint-descent lift lacks inherited coherent Lane slots")
    start = receiver.predictor.source_pair_start
    stop = start + receiver.z.n_pairs
    maximum = max(len(lane.lane_lines[pair]) for pair in range(start, stop))
    rows: list[LaneProgramSeedV1] = []
    range_max = float((lane.lane_header or {}).get("dash_forward_max_m", 50.0))
    for line_index in range(maximum):
        present = [pair for pair in range(start, stop) if line_index < len(lane.lane_lines[pair])]
        if not present:
            continue
        birth, death = present[0], present[-1] + 1
        vectors = np.stack([lane.lane_lines[pair][line_index] for pair in present]).astype(np.float64)
        representative = np.median(vectors, axis=0)
        local = np.asarray(present, dtype=np.int64) - start
        xi = receiver.pose6_codes[local, 0].astype(np.int16).astype(np.float64)
        xi -= xi[0]
        design = np.stack((np.ones_like(xi), xi), axis=1)
        intercept, gain = np.linalg.lstsq(design, vectors[:, 7], rcond=None)[0]
        rows.append(
            LaneProgramSeedV1(
                line_index=line_index,
                birth_pair=birth,
                death_pair_exclusive=death,
                bev_coefficients=(
                    float(representative[0]),
                    float(representative[1]),
                    float(representative[2]),
                    float(representative[3]),
                ),
                width_bias=float(representative[4]),
                width_slope=float(representative[5]),
                dash_phase_origin=float(intercept),
                dash_phase_xi_gain=float(gain),
                range_gate_forward_max_m=range_max,
            )
        )
    if not rows:
        raise DirectDescriptionError("joint-descent lift derived zero Lane seed records")
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class JointDescriptionParameterLiftV1:
    source_archive: bytes
    source_archive_sha256: str
    g1: G1WorldsheetParameterLiftV1
    lane_seeds: tuple[LaneProgramSeedV1, ...]
    template_rows: tuple[RowBandScorerTemplateV1, ...]
    parameter_names: tuple[str, ...]
    template_parameter_start: int
    ws1_adapter: WS1WarmStartArchiveV1 | None

    @property
    def carrier_archive(self) -> bytes:
        return self.source_archive if self.ws1_adapter is None else self.ws1_adapter.carrier_archive

    def rewrap_carrier(self, archive: bytes) -> bytes:
        return archive if self.ws1_adapter is None else self.ws1_adapter.rewrap_carrier(archive)

    def exact_reemit(self) -> bytes:
        members, _ = parse_carrier_compose_archive(self.carrier_archive)
        receiver = receive_joint_descent_archive(self.carrier_archive)
        payload = encode_lifted_g1_movable_worldsheet(self.g1)
        carrier_archive, _ = compile_carrier_compose_archive(
            members["predictor.zip"],
            worldsheet_g1_payload=payload,
            realization_profile=receiver.realization_profile,
            realization_static_rule_payload=members.get(REALIZATION_STATIC_RULE_MEMBER, b""),
            realization_static_rule_id=receiver.realization_static_rule_id,
            scorer_solved_templates=receiver.scorer_solved_templates,
        )
        archive = self.rewrap_carrier(carrier_archive)
        if archive != self.source_archive:
            raise DirectDescriptionError("joint-descent stage-00 archive recompile is not byte-identical")
        return archive

    def lane_seed_archive(self) -> bytes:
        """Emit every Lane seed atomically before any Lane coordinate trains."""

        members, _ = parse_carrier_compose_archive(self.carrier_archive)
        receiver = receive_joint_descent_archive(self.carrier_archive)
        records = tuple(row.counted_record() for row in self.lane_seeds)
        carrier_archive, _ = compile_carrier_compose_archive(
            members["predictor.zip"],
            worldsheet_g1_payload=encode_lifted_g1_movable_worldsheet(self.g1),
            lane_programs=records,
            realization_profile=receiver.realization_profile,
            realization_static_rule_payload=members.get(REALIZATION_STATIC_RULE_MEMBER, b""),
            realization_static_rule_id=receiver.realization_static_rule_id,
            scorer_solved_templates=receiver.scorer_solved_templates,
        )
        parsed, _ = parse_carrier_compose_archive(carrier_archive)
        if LANE_PROGRAM_MEMBER not in parsed:
            raise DirectDescriptionError("counted Lane seed lacks a receiver-consumed archive home")
        return self.rewrap_carrier(carrier_archive)

    def inventory(self) -> dict[str, Any]:
        lane_archive = self.lane_seed_archive()
        return {
            "schema": "ddm_joint_descent_parameter_lift.v1",
            "source_archive_bytes": len(self.source_archive),
            "source_archive_sha256": self.source_archive_sha256,
            "stage00_reemit_byte_identical": self.exact_reemit() == self.source_archive,
            "g1_payload_bytes": self.g1.source_payload_bytes,
            "g1_payload_sha256": self.g1.source_payload_sha256,
            "worldsheet_track_count": len(self.g1.tracks),
            "worldsheet_knot_count": len(self.g1.knots),
            "worldsheet_template_count": len(self.g1.templates),
            "lane_program_seed_count": len(self.lane_seeds),
            "lane_seed_archive_bytes": len(lane_archive),
            "lane_seed_archive_sha256": _sha256(lane_archive),
            "lane_seed_counted_byte_delta": len(lane_archive) - len(self.source_archive),
            "scorer_solved_template_count": len(self.template_rows),
            "low_dim_parameter_count": len(self.parameter_names),
            "score_claim": False,
            "evidence_axis": EVIDENCE_AXIS,
            "pointer": POINTER,
            "pointer_moved": False,
        }


def lift_v15_archive(archive: bytes) -> JointDescriptionParameterLiftV1:
    ws1_adapter: WS1WarmStartArchiveV1 | None = None
    if len(archive) == EXPECTED_ARCHIVE_BYTES and _sha256(archive) == EXPECTED_ARCHIVE_SHA256:
        carrier_archive = archive
    else:
        try:
            ws1_adapter = parse_ws1_warm_start_archive(archive)
        except DirectDescriptionError as exc:
            raise DirectDescriptionError(
                "joint-descent parameter lift requires the sealed V15 archive or a receiver-closed WS1 archive"
            ) from exc
        carrier_archive = ws1_adapter.carrier_archive
    members, _ = parse_carrier_compose_archive(carrier_archive)
    receiver = receive_joint_descent_archive(carrier_archive)
    g1 = lift_g1_movable_worldsheet(members[WORLDSHEET_G1_MEMBER])
    if receiver.scorer_solved_templates is None:
        raise DirectDescriptionError("joint-descent V15 warm start lacks its counted template bank")
    lanes = derive_lane_program_seeds(receiver)
    names: list[str] = []
    # Only coordinates that survive the current receiver encoder belong in the
    # optimizer surface.  ``aspect_log``/``rotation_radians`` are lift metadata
    # but are not encoded by G1S1; the inherited BEV/range seed values likewise
    # have no LanePeriodicProgramV1 wire fields.  Counting those was the J2
    # 706-name overstatement.  The executable surface is therefore 368 DOFs:
    # 2*163 track translations + 4*6 counted Lane fields + 3*6 template bytes.
    for track in g1.tracks:
        names.extend(f"island.track{track.object_id}.{field}" for field in ("center_x", "center_y"))
    for lane in lanes:
        names.extend(
            f"lane.line{lane.line_index}.{field}"
            for field in ("dash_phase_origin_q8", "dash_phase_xi_gain_q8", "width_bias_q8", "width_slope_q12")
        )
    template_start = len(names)
    template_rows = receiver.scorer_solved_templates.templates
    for index, _ in enumerate(template_rows):
        names.extend(f"template.row{index}.rgb_{channel}" for channel in ("r", "g", "b"))
    result = JointDescriptionParameterLiftV1(
        source_archive=archive,
        source_archive_sha256=_sha256(archive),
        g1=g1,
        lane_seeds=lanes,
        template_rows=template_rows,
        parameter_names=tuple(names),
        template_parameter_start=template_start,
        ws1_adapter=ws1_adapter,
    )
    result.exact_reemit()
    return result


def _compile_lift_variant(
    lift: JointDescriptionParameterLiftV1,
    *,
    g1: G1WorldsheetParameterLiftV1 | None = None,
    lane_programs: Sequence[LanePeriodicProgramV1] = (),
    template_rows: Sequence[RowBandScorerTemplateV1] | None = None,
    verify_member_effects: bool = True,
) -> bytes:
    """Compile one receiver-consumed parameter mutation from the sealed base."""

    members, _ = parse_carrier_compose_archive(lift.carrier_archive)
    receiver = receive_joint_descent_archive(lift.carrier_archive)
    bank = receiver.scorer_solved_templates
    if bank is None:
        raise DirectDescriptionError("joint-descent variant lacks the inherited template bank")
    if template_rows is not None:
        bank = replace(bank, templates=tuple(template_rows))
    archive, _ = compile_carrier_compose_archive(
        members["predictor.zip"],
        worldsheet_g1_payload=encode_lifted_g1_movable_worldsheet(g1 or lift.g1),
        lane_programs=tuple(lane_programs),
        realization_profile=receiver.realization_profile,
        realization_static_rule_payload=members.get(REALIZATION_STATIC_RULE_MEMBER, b""),
        realization_static_rule_id=receiver.realization_static_rule_id,
        scorer_solved_templates=bank,
    )
    receive_joint_descent_archive(
        archive,
        verify_member_effects=verify_member_effects,
    )
    return lift.rewrap_carrier(archive)


def parameter_group_indices(lift: JointDescriptionParameterLiftV1) -> dict[str, tuple[int, ...]]:
    """Return the three receiver-effective parameter groups by exact name."""

    groups = {
        "island_worldsheet": tuple(
            index for index, name in enumerate(lift.parameter_names) if name.startswith("island.")
        ),
        "lane_program": tuple(index for index, name in enumerate(lift.parameter_names) if name.startswith("lane.")),
        "shared_template_dof": tuple(
            index for index, name in enumerate(lift.parameter_names) if name.startswith("template.")
        ),
    }
    if tuple(len(groups[name]) for name in groups) != (
        2 * len(lift.g1.tracks),
        4 * len(lift.lane_seeds),
        3 * len(lift.template_rows),
    ):
        raise DirectDescriptionError("joint-descent receiver-effective parameter grouping differs")
    return groups


def _project_realized_island_geometry(
    lift: JointDescriptionParameterLiftV1,
    realized: np.ndarray,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    """Apply the landed RG1 projection to each shared G1 track translation."""

    projected = np.asarray(realized, dtype=np.float64).copy()
    templates = {row.template_ref: row for row in lift.g1.templates}
    events: list[dict[str, Any]] = []
    for track_index, track in enumerate(lift.g1.tracks):
        x_index = 2 * track_index
        y_index = x_index + 1
        requested = (int(projected[x_index]), int(projected[y_index]))
        relative_x: list[int] = []
        relative_y: list[int] = []
        for knot_index in track.knot_indices:
            knot = lift.g1.knots[knot_index]
            vertices = templates[knot.template_ref].relative_vertices_xy
            relative_x.extend(knot.center_x + int(vertex[0]) for vertex in vertices)
            relative_y.extend(knot.center_y + int(vertex[1]) for vertex in vertices)
        try:
            cured = (
                project_polygon_center(requested[0], relative_x, SCORER_WIDTH),
                project_polygon_center(requested[1], relative_y, SCORER_HEIGHT),
            )
        except DirectDescriptionError as exc:
            event = {
                "schema": "ddm_joint_descent_geometry_projection_event.v1",
                "event": "proposal_infeasible_geometry",
                "status": "rejected",
                "track_index": track_index,
                "track_object_id": track.object_id,
                "parameter_indices": [x_index, y_index],
                "parameter_names": [lift.parameter_names[x_index], lift.parameter_names[y_index]],
                "requested_translation_xy": list(requested),
                "projected_translation_xy": None,
                "scorer_extent_wh": [SCORER_WIDTH, SCORER_HEIGHT],
                "projection": "rg1.project_polygon_center",
                "reason": str(exc),
                "verdict_scope": "INSTANCE proposal geometry only",
                "score_claim": False,
            }
            raise ProposalGeometryInfeasibleError(event) from exc
        if cured != requested:
            projected[x_index], projected[y_index] = cured
            events.append(
                {
                    "schema": "ddm_joint_descent_geometry_projection_event.v1",
                    "event": "proposal_infeasible_geometry",
                    "status": "cured",
                    "track_index": track_index,
                    "track_object_id": track.object_id,
                    "parameter_indices": [x_index, y_index],
                    "parameter_names": [lift.parameter_names[x_index], lift.parameter_names[y_index]],
                    "requested_translation_xy": list(requested),
                    "projected_translation_xy": list(cured),
                    "scorer_extent_wh": [SCORER_WIDTH, SCORER_HEIGHT],
                    "projection": "rg1.project_polygon_center",
                    "reason": "projected_shared_track_translation_into_legal_scorer_plane",
                    "verdict_scope": "INSTANCE proposal geometry only",
                    "score_claim": False,
                }
            )
    return np.ascontiguousarray(projected, dtype=np.float32), tuple(events)


def realize_parameter_theta(
    lift: JointDescriptionParameterLiftV1,
    theta: np.ndarray,
    *,
    geometry_events: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Quantize optimizer coordinates into exact receiver wire quanta."""

    value = np.asarray(theta, dtype=np.float32)
    if value.shape != (len(lift.parameter_names),) or not np.all(np.isfinite(value)):
        raise DirectDescriptionError("joint-descent parameter theta is invalid")
    realized = np.rint(value.astype(np.float64))
    groups = parameter_group_indices(lift)
    lane = np.asarray(groups["lane_program"], dtype=np.int64)
    realized[lane] = np.clip(realized[lane], -32768, 32767)
    projected, events = _project_realized_island_geometry(lift, realized)
    if geometry_events is not None:
        geometry_events.extend(events)
    return projected


def project_adam_state_geometry(
    lift: JointDescriptionParameterLiftV1,
    state: AdamStateV1,
) -> tuple[AdamStateV1, tuple[dict[str, Any], ...]]:
    """Project proposal weights into legal G1 geometry before proxy or exact scoring."""

    events: list[dict[str, Any]] = []
    projected_theta = realize_parameter_theta(lift, state.theta, geometry_events=events)
    projected_ema = realize_parameter_theta(lift, state.ema)
    theta = np.asarray(state.theta, dtype=np.float32).copy()
    ema = np.asarray(state.ema, dtype=np.float32).copy()
    for event in events:
        indices = np.asarray(event["parameter_indices"], dtype=np.int64)
        theta[indices] = projected_theta[indices]
    ema_realized = np.rint(np.asarray(state.ema, dtype=np.float64)).astype(np.float32)
    changed_ema = np.flatnonzero(projected_ema != ema_realized)
    ema[changed_ema] = projected_ema[changed_ema]
    return (
        AdamStateV1(
            step=state.step,
            theta=np.ascontiguousarray(theta),
            ema=np.ascontiguousarray(ema),
            first_moment=state.first_moment,
            second_moment=state.second_moment,
        ),
        tuple(events),
    )


def compile_parameterized_archive(
    lift: JointDescriptionParameterLiftV1,
    theta: np.ndarray,
    *,
    include_lane_programs: bool,
    geometry_events: list[dict[str, Any]] | None = None,
) -> tuple[bytes, np.ndarray]:
    """Compile quantized low-dimensional state into one receiver-closed archive."""

    realized = realize_parameter_theta(lift, theta, geometry_events=geometry_events)
    cursor = 0
    knots = list(lift.g1.knots)
    for track in lift.g1.tracks:
        dx, dy = (int(realized[cursor]), int(realized[cursor + 1]))
        cursor += 2
        for knot_index in track.knot_indices:
            knot = knots[knot_index]
            knots[knot_index] = replace(knot, center_x=knot.center_x + dx, center_y=knot.center_y + dy)
    g1 = replace(lift.g1, knots=tuple(knots))

    lanes: list[LanePeriodicProgramV1] = []
    for seed in lift.lane_seeds:
        base = seed.counted_record()
        deltas = tuple(int(value) for value in realized[cursor : cursor + 4])
        cursor += 4
        lanes.append(
            replace(
                base,
                dash_phase_origin_delta_q8=int(np.clip(base.dash_phase_origin_delta_q8 + deltas[0], -32768, 32767)),
                dash_phase_xi_gain_q8=int(np.clip(base.dash_phase_xi_gain_q8 + deltas[1], -32768, 32767)),
                width_bias_q8=int(np.clip(base.width_bias_q8 + deltas[2], -32768, 32767)),
                width_slope_q12=int(np.clip(base.width_slope_q12 + deltas[3], -32768, 32767)),
            )
        )

    templates: list[RowBandScorerTemplateV1] = []
    for row in lift.template_rows:
        channel_delta = np.asarray(realized[cursor : cursor + 3], dtype=np.int16)
        cursor += 3
        rgb = np.frombuffer(row.rgb_u8, dtype=np.uint8).reshape(-1, 3).astype(np.int16)
        rgb = np.clip(rgb + channel_delta[None, :], 0, 255).astype(np.uint8)
        templates.append(replace(row, rgb_u8=rgb.tobytes()))
    if cursor != len(realized):
        raise DirectDescriptionError("joint-descent parameter compiler left coordinates unconsumed")
    archive = _compile_lift_variant(
        lift,
        g1=g1,
        lane_programs=lanes if include_lane_programs else (),
        template_rows=templates,
        verify_member_effects=False,
    )
    return archive, realized


def realized_training_state(
    lift: JointDescriptionParameterLiftV1,
    theta: np.ndarray,
    *,
    pair_ids: Sequence[int],
    active_groups: Sequence[str],
    include_lane_programs: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], np.ndarray, bytes]:
    """Build sparse exact +1-quantum secants around current parse-back state.

    The returned basis contains only receiver-effective island/Lane coordinates
    that can affect this pair window.  Shared-template coordinates use their
    exact grammar masks in :class:`DirectDescriptionJointDescentMLXModule`.
    """

    indexes = tuple(int(value) for value in pair_ids)
    archive, realized = compile_parameterized_archive(lift, theta, include_lane_programs=include_lane_programs)
    receiver = receive_joint_descent_archive(
        archive,
        verify_member_effects=False,
    )
    camera = receiver.render_camera_pairs(indexes).astype(np.float32)
    template_rows = receiver.scorer_solved_templates
    if template_rows is None:
        raise DirectDescriptionError("parameterized archive lost its template bank")
    masks = np.stack(
        [receiver.template_camera_masks(indexes, row).astype(np.float32) for row in template_rows.templates],
        axis=0,
    )
    if "shared_template_dof" not in active_groups:
        masks.fill(0.0)
    if np.any(masks.sum(axis=0) > 1.0):
        raise DirectDescriptionError("parameterized template masks overlap")

    groups = parameter_group_indices(lift)
    selected: list[int] = []
    if "island_worldsheet" in active_groups:
        pair_set = set(indexes)
        for track_index, track in enumerate(lift.g1.tracks):
            if any(lift.g1.knots[knot_index].pair_index in pair_set for knot_index in track.knot_indices):
                selected.extend((2 * track_index, 2 * track_index + 1))
    if "lane_program" in active_groups:
        selected.extend(groups["lane_program"])
    secants: list[np.ndarray] = []
    for parameter_index in selected:
        secant: np.ndarray | None = None
        errors: list[str] = []
        for direction in (1.0, -1.0):
            probe = realized.copy()
            probe[parameter_index] += np.float32(direction)
            try:
                probe_archive, _ = compile_parameterized_archive(
                    lift, probe, include_lane_programs=include_lane_programs
                )
                probe_camera = receive_joint_descent_archive(
                    probe_archive, verify_member_effects=False
                ).render_camera_pairs(indexes)
            except DirectDescriptionError as exc:
                errors.append(str(exc))
                continue
            secant = (probe_camera.astype(np.float32) - camera) / np.float32(direction)
            break
        if secant is None:
            raise DirectDescriptionError(
                "joint-descent coordinate has no feasible one-quantum secant: "
                f"{lift.parameter_names[parameter_index]}: {'; '.join(errors)}"
            )
        secants.append(secant)
    basis = np.stack(secants, axis=0) if secants else np.empty((0, *camera.shape), dtype=np.float32)
    local_theta = np.asarray(theta, dtype=np.float32) - realized
    return camera, masks, basis, tuple(selected), local_theta, archive


def verify_trainable_group_ownership(lift: JointDescriptionParameterLiftV1) -> dict[str, Any]:
    """Prove each trainable group owns counted bytes and receiver-visible output.

    This is deliberately a bounded one-coordinate proof, not efficacy evidence.
    Every mutation is encoded into a receiver-consumed archive member before its
    camera output is compared with stage 00.
    """

    base_receiver = receive_joint_descent_archive(lift.source_archive)
    rows: dict[str, Any] = {}

    # Island: translate one explicit lifecycle track by one scorer-grid pixel.
    track = next((row for row in lift.g1.tracks if row.knot_indices), None)
    if track is None:
        raise DirectDescriptionError("island ownership probe lacks a nonempty G1 track")
    selected = set(track.knot_indices)
    island_g1 = replace(
        lift.g1,
        knots=tuple(
            replace(knot, center_x=knot.center_x + 1) if index in selected else knot
            for index, knot in enumerate(lift.g1.knots)
        ),
    )
    island_archive = _compile_lift_variant(lift, g1=island_g1)
    island_pair = lift.g1.knots[track.knot_indices[0]].pair_index
    island_delta = int(
        np.count_nonzero(
            base_receiver.render_camera_pairs((island_pair,))
            != receive_joint_descent_archive(island_archive).render_camera_pairs((island_pair,))
        )
    )
    rows["island_worldsheet"] = {
        "coordinate": f"track{track.object_id}.center_x_plus_1",
        "pair_id": island_pair,
        "archive_bytes": len(island_archive),
        "archive_sha256": _sha256(island_archive),
        "archive_changed": island_archive != lift.source_archive,
        "receiver_camera_changed_values": island_delta,
    }

    # Lane: materialize the complete counted seed, then change its phase.
    lane = lift.lane_seeds[0]
    lane_records = [row.counted_record() for row in lift.lane_seeds]
    lane_records[0] = replace(lane_records[0], dash_phase_origin_delta_q8=256)
    lane_archive = _compile_lift_variant(lift, lane_programs=lane_records)
    candidate_pairs = tuple(
        sorted(
            {
                lane.birth_pair,
                (lane.birth_pair + lane.death_pair_exclusive - 1) // 2,
                lane.death_pair_exclusive - 1,
            }
        )
    )
    lane_delta = int(
        np.count_nonzero(
            base_receiver.render_camera_pairs(candidate_pairs)
            != receive_joint_descent_archive(lane_archive).render_camera_pairs(candidate_pairs)
        )
    )
    rows["lane_program"] = {
        "coordinate": f"line{lane.line_index}.dash_phase_origin_plus_1",
        "pair_ids": list(candidate_pairs),
        "archive_bytes": len(lane_archive),
        "archive_sha256": _sha256(lane_archive),
        "archive_changed": lane_archive != lift.source_archive,
        "receiver_camera_changed_values": lane_delta,
        "base_bev_coefficients": list(lane.bev_coefficients),
        "base_range_gate_forward_max_m": lane.range_gate_forward_max_m,
        "seed_is_counted_before_gradient": True,
    }

    # Template: perturb one counted RGB byte while retaining its typed record.
    active_template: tuple[int, RowBandScorerTemplateV1, int] | None = None
    for template_index, candidate in enumerate(lift.template_rows):
        candidate_pair = next(
            (
                pair_id
                for pair_id in range(lift.g1.pair_count)
                if np.any(base_receiver.template_camera_masks((pair_id,), candidate))
            ),
            None,
        )
        if candidate_pair is not None:
            active_template = (template_index, candidate, candidate_pair)
            break
    if active_template is None:
        raise DirectDescriptionError("counted template ownership probe found no receiver-visible site")
    template_index, template, template_pair = active_template
    rgb = bytearray(value + 1 if value < 255 else value - 1 for value in template.rgb_u8)
    templates = list(lift.template_rows)
    templates[template_index] = replace(template, rgb_u8=bytes(rgb))
    template_archive = _compile_lift_variant(lift, template_rows=templates)
    template_delta = int(
        np.count_nonzero(
            base_receiver.render_camera_pairs((template_pair,))
            != receive_joint_descent_archive(template_archive).render_camera_pairs((template_pair,))
        )
    )
    rows["shared_template"] = {
        "coordinate": f"template{template_index}.all_rgb_plus_or_minus_1",
        "pair_id": template_pair,
        "archive_bytes": len(template_archive),
        "archive_sha256": _sha256(template_archive),
        "archive_changed": template_archive != lift.source_archive,
        "receiver_camera_changed_values": template_delta,
    }

    inert = [
        name for name, row in rows.items() if not row["archive_changed"] or row["receiver_camera_changed_values"] <= 0
    ]
    if inert:
        raise DirectDescriptionError(
            "joint-descent trainable groups are counted but receiver-inert: " + ",".join(inert)
        )
    return {
        "schema": "ddm_joint_descent_trainable_group_ownership.v1",
        "groups": rows,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
        "pointer": POINTER,
        "pointer_moved": False,
    }


def template_camera_state(
    lift: JointDescriptionParameterLiftV1,
    pair_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact V15 camera pairs and disjoint template masks for an MLX step."""

    receiver = receive_joint_descent_archive(lift.source_archive)
    indexes = tuple(int(value) for value in pair_ids)
    camera = receiver.render_camera_pairs(indexes).astype(np.float32)
    masks = np.stack(
        [receiver.template_camera_masks(indexes, row).astype(np.float32) for row in lift.template_rows],
        axis=0,
    )
    # Same solved-template paint is applied to both frames.  Overlapping rows
    # would make a linear color lift order-dependent, so refuse rather than
    # silently double-own a pixel.
    if np.any(masks.sum(axis=0) > 1.0):
        raise DirectDescriptionError("V15 solved-template masks overlap in the trainable lift")
    return np.ascontiguousarray(camera), np.ascontiguousarray(masks)


class DirectDescriptionJointDescentMLXModule:
    """MLX params -> exact grammar paint -> uint8 STE -> fused R -> frozen scorers.

    Island/Lane coordinates enter through caller-supplied *realized secant*
    fields produced by exact archive parse-back.  Template coordinates use the
    receiver's exact camera masks directly.  No pixel tensor is trainable.
    """

    def __init__(
        self,
        *,
        lift: JointDescriptionParameterLiftV1,
        scorer_adapter: Any,
        seg_targets: np.ndarray,
        pose_targets: np.ndarray,
        margin_targets: np.ndarray | None = None,
        margin_hinge_weight: float = 0.05,
        # ddm_gk1 2026-08-03: was the bare literal `0.1`.  Now the class-4 custodied
        # value of the GuardedConstant `seg_margin_hinge_floor`, which also declares
        # the LIVE derivation (tac.optimization.lane_guard:derive_margin_floor) this
        # literal froze the output of.  MIGRATION IS BYTE-IDENTICAL — proven, not
        # assumed, at registry import (MARGIN_FLOOR_MIGRATION_IS_BYTE_IDENTICAL) and
        # re-asserted below, so this landing changes no shipped value.  A caller
        # holding the GT-Lane-restricted margin field should instead resolve the
        # constant live:
        #     MARGIN_FLOOR.resolve(consumer_role=ROLE_THRESHOLD, sample=<lane margins>,
        #                          sample_domain_id=LANE_MARGIN_DOMAIN.domain_id)
        margin_floor: float = _MARGIN_FLOOR_DEFAULT,
    ) -> None:
        import mlx.core as mx

        self.mx = mx
        self.lift = lift
        self.scorer = scorer_adapter
        self.seg_targets = mx.array(np.asarray(seg_targets, dtype=np.int32))
        self.pose_targets = mx.array(np.asarray(pose_targets, dtype=np.float32))
        self.margin_targets = None if margin_targets is None else mx.array(np.asarray(margin_targets, dtype=np.float32))
        self.margin_hinge_weight = float(margin_hinge_weight)
        self.margin_floor = float(margin_floor)
        self.parameter_count = len(lift.parameter_names)

    def _validate_step_arrays(
        self,
        theta: np.ndarray,
        pair_ids: Sequence[int],
        base_camera: np.ndarray,
        template_masks: np.ndarray,
        realized_secant_basis: np.ndarray | None,
        realized_secant_indices: Sequence[int] | None,
    ) -> None:
        count = len(tuple(pair_ids))
        if np.asarray(theta).shape != (self.parameter_count,):
            raise DirectDescriptionError("joint-descent theta geometry differs")
        if np.asarray(base_camera).shape != (count, 2, 874, 1164, 3):
            raise DirectDescriptionError("joint-descent camera batch geometry differs")
        if np.asarray(template_masks).shape != (len(self.lift.template_rows), count, 874, 1164):
            raise DirectDescriptionError("joint-descent template-mask geometry differs")
        if realized_secant_basis is not None and np.asarray(realized_secant_basis).shape != (
            len(tuple(realized_secant_indices or ())),
            count,
            2,
            874,
            1164,
            3,
        ):
            raise DirectDescriptionError("joint-descent realized-secant basis geometry differs")
        if realized_secant_basis is not None and any(
            index < 0 or index >= self.parameter_count for index in tuple(realized_secant_indices or ())
        ):
            raise DirectDescriptionError("joint-descent realized-secant index is outside theta")

    def _render_camera(
        self,
        theta: Any,
        base_camera: Any,
        template_masks: Any,
        realized_secant_basis: Any | None,
        realized_secant_indices: Sequence[int] | None,
    ) -> Any:
        mx = self.mx
        template_count = len(self.lift.template_rows)
        start = self.lift.template_parameter_start
        colour_delta = mx.reshape(theta[start : start + template_count * 3], (template_count, 3))
        # masks K,B,H,W; delta -> B,H,W,3 and is shared across the two frames.
        paint_delta = mx.einsum("kbhw,kc->bhwc", template_masks, colour_delta)
        camera = base_camera + paint_delta[:, None, :, :, :]
        if realized_secant_basis is not None:
            # Basis is K,B,2,H,W,3 and is derived from exact archive finite
            # secants.  It is immutable receiver geometry, never a trainable
            # frame table; theta is the sole differentiable state.
            selected = theta[mx.array(np.asarray(realized_secant_indices, dtype=np.int32))]
            camera = camera + mx.tensordot(selected, realized_secant_basis, axes=[[0], [0]])
        clipped = mx.clip(camera, 0.0, 255.0)
        return clipped + mx.stop_gradient(mx.round(clipped) - clipped)

    def _loss(
        self,
        theta: Any,
        *,
        pair_ids: Any,
        base_camera: Any,
        template_masks: Any,
        realized_secant_basis: Any | None,
        realized_secant_indices: Sequence[int] | None,
        pose_objective_weight: float,
    ) -> Any:
        mx = self.mx
        seg, pose_mse, _ = self._components(
            theta,
            pair_ids=pair_ids,
            base_camera=base_camera,
            template_masks=template_masks,
            realized_secant_basis=realized_secant_basis,
            realized_secant_indices=realized_secant_indices,
        )
        # The sqrt term is the exact contest action; epsilon only defines its
        # derivative at zero and is far below the observed warm-start value.
        return 100.0 * seg + float(pose_objective_weight) * mx.sqrt(10.0 * pose_mse + 1.0e-12)

    def _components(
        self,
        theta: Any,
        *,
        pair_ids: Any,
        base_camera: Any,
        template_masks: Any,
        realized_secant_basis: Any | None,
        realized_secant_indices: Sequence[int] | None,
    ) -> tuple[Any, Any, Any]:
        mx = self.mx
        from tac.local_acceleration.metal_fused_r_operator import fused_r_roundtrip
        from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx
        from tac.mlx_pr95_port.mlx_losses import (
            ce_seg_loss_mlx,
            margin_floor_hinge_mlx,
            pose_loss_mlx,
        )

        camera = self._render_camera(
            theta,
            base_camera,
            template_masks,
            realized_secant_basis,
            realized_secant_indices,
        )
        flat = mx.reshape(camera, (-1, 874, 1164, 3))
        scorer_rgb = fused_r_roundtrip(
            flat,
            camera_hw=(874, 1164),
            output_hw=(384, 512),
            ste_round=True,
        )
        pairs = mx.reshape(scorer_rgb, (-1, 2, 384, 512, 3))
        seg_logits = self.scorer.segnet(pairs[:, 1])
        seg_logits_nchw = mx.transpose(seg_logits, (0, 3, 1, 2))
        targets = self.seg_targets[pair_ids]
        seg = ce_seg_loss_mlx(seg_logits_nchw, targets)
        if self.margin_hinge_weight > 0.0:
            seg = seg + self.margin_hinge_weight * margin_floor_hinge_mlx(
                seg_logits_nchw, targets, margin_floor=self.margin_floor
            )
        yuv6 = rgb_to_yuv6_mlx(pairs)
        pose_input = mx.reshape(mx.transpose(yuv6, (0, 2, 3, 1, 4)), (-1, 192, 256, 12))
        pose = self.scorer.posenet(pose_input)["pose"][..., :6]
        pose_mse = pose_loss_mlx(pose, self.pose_targets[pair_ids])
        d_seg = mx.mean(mx.not_equal(mx.argmax(seg_logits, axis=-1), targets).astype(mx.float32))
        return seg, pose_mse, d_seg

    def loss_and_grad(
        self,
        theta: np.ndarray,
        *,
        pair_ids: Sequence[int],
        base_camera: np.ndarray,
        template_masks: np.ndarray,
        realized_secant_basis: np.ndarray | None = None,
        realized_secant_indices: Sequence[int] | None = None,
        pose_objective_weight: float = 1.0,
    ) -> tuple[float, np.ndarray]:
        self._validate_step_arrays(
            theta,
            pair_ids,
            base_camera,
            template_masks,
            realized_secant_basis,
            realized_secant_indices,
        )
        if not math.isfinite(float(pose_objective_weight)) or not 0.0 <= pose_objective_weight <= 1.0:
            raise DirectDescriptionError("joint-descent Pose objective weight is invalid")
        mx = self.mx
        pair_mx = mx.array(np.asarray(pair_ids, dtype=np.int32))
        base_mx = mx.array(np.asarray(base_camera, dtype=np.float32))
        masks_mx = mx.array(np.asarray(template_masks, dtype=np.float32))
        basis_mx = (
            None if realized_secant_basis is None else mx.array(np.asarray(realized_secant_basis, dtype=np.float32))
        )

        def closure(value: Any) -> Any:
            return self._loss(
                value,
                pair_ids=pair_mx,
                base_camera=base_mx,
                template_masks=masks_mx,
                realized_secant_basis=basis_mx,
                realized_secant_indices=realized_secant_indices,
                pose_objective_weight=pose_objective_weight,
            )

        value, gradient = mx.value_and_grad(closure)(mx.array(np.asarray(theta, dtype=np.float32)))
        mx.eval(value, gradient)
        return float(np.asarray(value)), np.asarray(gradient, dtype=np.float32)

    def measure_components(
        self,
        theta: np.ndarray,
        *,
        pair_ids: Sequence[int],
        base_camera: np.ndarray,
        template_masks: np.ndarray,
        realized_secant_basis: np.ndarray | None = None,
        realized_secant_indices: Sequence[int] | None = None,
    ) -> dict[str, float]:
        """Measure the same MLX research-signal components without updating state."""

        self._validate_step_arrays(
            theta,
            pair_ids,
            base_camera,
            template_masks,
            realized_secant_basis,
            realized_secant_indices,
        )
        mx = self.mx
        pair_mx = mx.array(np.asarray(pair_ids, dtype=np.int32))
        basis = None if realized_secant_basis is None else mx.array(np.asarray(realized_secant_basis, dtype=np.float32))
        seg, pose, d_seg = self._components(
            mx.array(np.asarray(theta, dtype=np.float32)),
            pair_ids=pair_mx,
            base_camera=mx.array(np.asarray(base_camera, dtype=np.float32)),
            template_masks=mx.array(np.asarray(template_masks, dtype=np.float32)),
            realized_secant_basis=basis,
            realized_secant_indices=realized_secant_indices,
        )
        objective = 100.0 * seg + mx.sqrt(10.0 * pose + 1.0e-12)
        mx.eval(seg, pose, d_seg, objective)
        return {
            "seg_ce_margin": float(np.asarray(seg)),
            "d_seg": float(np.asarray(d_seg)),
            "d_pose": float(np.asarray(pose)),
            "joint_objective_no_rate": float(np.asarray(objective)),
        }


@dataclass(frozen=True, slots=True)
class AdamStateV1:
    step: int
    theta: np.ndarray
    ema: np.ndarray
    first_moment: np.ndarray
    second_moment: np.ndarray


def _optimizer_state_sha256(state: AdamStateV1) -> str:
    digest = hashlib.sha256(int(state.step).to_bytes(8, "little", signed=False))
    for value in (state.theta, state.ema, state.first_moment, state.second_moment):
        array = np.ascontiguousarray(value, dtype="<f4")
        digest.update(len(array).to_bytes(8, "little", signed=False))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(slots=True)
class JointDescentResumeControllerV1:
    """Canonical resume-registry integrity controller for optimizer state."""

    state: AdamStateV1
    typed_config_hash: str
    event_mode: bool = True

    def state_arrays(self, prefix: str) -> dict[str, Any]:
        return {
            prefix + "step": np.asarray(self.state.step, dtype=np.int64),
            prefix + "optimizer_state_sha256": np.asarray(_optimizer_state_sha256(self.state)),
            prefix + "typed_config_hash": np.asarray(self.typed_config_hash),
        }

    def restore_from_cfg(self, prefix: str, cfg: dict[str, Any]) -> bool:
        required = (prefix + "step", prefix + "optimizer_state_sha256", prefix + "typed_config_hash")
        if any(key not in cfg for key in required):
            return False
        if int(cfg[required[0]]) != self.state.step:
            raise DirectDescriptionError("resume-registry optimizer step differs")
        if str(cfg[required[1]]) != _optimizer_state_sha256(self.state):
            raise DirectDescriptionError("resume-registry optimizer state hash differs")
        if str(cfg[required[2]]) != self.typed_config_hash:
            raise DirectDescriptionError("resume-registry typed config hash differs")
        return True


def _optimizer_resume_registry(
    state: AdamStateV1,
    config: DirectDescriptionJointDescentTypedConfigV1,
) -> ResumeRegistry:
    registry = ResumeRegistry()
    registry.register(
        "ddm_joint_descent_optimizer",
        "__ddmjd_",
        JointDescentResumeControllerV1(state=state, typed_config_hash=config.typed_config_hash()),
    )
    return registry


def initial_adam_state(parameter_count: int) -> AdamStateV1:
    zeros = np.zeros(int(parameter_count), dtype=np.float32)
    return AdamStateV1(0, zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy())


def linear_rewarmup_factor(
    *,
    completed_steps: int,
    rewarmup_steps: int,
    floor: float,
) -> float:
    """Return the #518 linear LR factor at a fresh/resumed Adam boundary."""

    if completed_steps < 0 or rewarmup_steps <= 0 or not 0.0 < floor <= 1.0:
        raise DirectDescriptionError("joint-descent LR rewarmup geometry is invalid")
    progress = min(float(completed_steps) / float(rewarmup_steps), 1.0)
    return float(floor + (1.0 - floor) * progress)


def clipped_adam_step(
    state: AdamStateV1,
    gradient: np.ndarray,
    *,
    learning_rate: float,
    grad_clip: float,
    ema_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    maximum_update: float | None = None,
    theta_lattice_denominator: int | None = None,
) -> AdamStateV1:
    grad = np.asarray(gradient, dtype=np.float32)
    if grad.shape != state.theta.shape or any(
        value.shape != state.theta.shape for value in (state.ema, state.first_moment, state.second_moment)
    ):
        raise DirectDescriptionError("joint-descent Adam state/gradient geometry differs")
    hyperparameters = (learning_rate, grad_clip, ema_decay, beta1, beta2)
    if (
        not all(math.isfinite(float(value)) for value in hyperparameters)
        or learning_rate <= 0.0
        or grad_clip <= 0.0
        or not 0.0 <= ema_decay < 1.0
        or not 0.0 < beta1 < 1.0
        or not 0.0 < beta2 < 1.0
        or (maximum_update is not None and (not math.isfinite(float(maximum_update)) or maximum_update <= 0.0))
        or (
            theta_lattice_denominator is not None
            and (
                isinstance(theta_lattice_denominator, bool)
                or not isinstance(theta_lattice_denominator, int)
                or theta_lattice_denominator <= 0
            )
        )
    ):
        raise DirectDescriptionError("joint-descent Adam hyperparameters are invalid")
    norm = float(np.linalg.norm(grad.astype(np.float64)))
    if not math.isfinite(norm):
        raise DirectDescriptionError("joint-descent gradient is nonfinite")
    if norm > grad_clip:
        grad = grad * np.float32(grad_clip / norm)
    step = state.step + 1
    first = beta1 * state.first_moment + (1.0 - beta1) * grad
    second = beta2 * state.second_moment + (1.0 - beta2) * np.square(grad)
    first_hat = first / (1.0 - beta1**step)
    second_hat = second / (1.0 - beta2**step)
    update = learning_rate * first_hat / (np.sqrt(second_hat) + 1.0e-8)
    if maximum_update is not None:
        update = np.clip(update, -float(maximum_update), float(maximum_update))
    theta = state.theta - update
    if theta_lattice_denominator is not None:
        denominator = float(theta_lattice_denominator)
        theta = np.rint(theta.astype(np.float64) * denominator) / denominator
    ema = ema_decay * state.ema + (1.0 - ema_decay) * theta
    return AdamStateV1(
        step=step,
        theta=np.asarray(theta, dtype=np.float32),
        ema=np.asarray(ema, dtype=np.float32),
        first_moment=np.asarray(first, dtype=np.float32),
        second_moment=np.asarray(second, dtype=np.float32),
    )


def opening_candidate_gradient(
    lift: JointDescriptionParameterLiftV1,
    candidate_id: str,
    local_gradient: np.ndarray,
    *,
    active_pair_ids: Sequence[int] = (),
) -> np.ndarray:
    """Return one typed J5 proposal direction in optimizer-gradient sign.

    The coherent worldsheet rows are the exact grammar-native v19 proposal
    family.  They are only proposal sources: receiver parse-back plus frozen
    n600 scorers and exact archive bytes remain the admission authority.
    """

    gradient = np.asarray(local_gradient, dtype=np.float32)
    if gradient.shape != (len(lift.parameter_names),) or not np.all(np.isfinite(gradient)):
        raise DirectDescriptionError("warm-start local proposal gradient is invalid")
    if candidate_id == "local_exact_gradient":
        return np.ascontiguousarray(gradient)
    axes = {
        "worldsheet_joint_active_x_+1": (".center_x", 0, 1, -1.0),
        "worldsheet_joint_active_x_-1": (".center_x", 0, -1, 1.0),
        "worldsheet_joint_active_y_+1": (".center_y", 1, 1, -1.0),
        "worldsheet_joint_active_y_-1": (".center_y", 1, -1, 1.0),
    }
    try:
        suffix, axis_offset, realized_shift, optimizer_sign = axes[candidate_id]
    except KeyError as exc:
        raise DirectDescriptionError("warm-start opening candidate id is invalid") from exc
    pair_set = {int(value) for value in active_pair_ids}
    if not pair_set or any(value < 0 or value >= 600 for value in pair_set):
        raise DirectDescriptionError("warm-start coherent proposal pair selection is invalid")

    templates = {row.template_ref: row for row in lift.g1.templates}

    def translation_bound(track_index: int) -> tuple[int, int]:
        minimum = 1 << 30
        maximum = -(1 << 30)
        for knot_index in lift.g1.tracks[track_index].knot_indices:
            knot = lift.g1.knots[knot_index]
            vertices = templates[knot.template_ref].relative_vertices_xy
            coordinates = [
                (knot.center_x if axis_offset == 0 else knot.center_y) + int(vertex[axis_offset]) for vertex in vertices
            ]
            minimum = min(minimum, min(coordinates))
            maximum = max(maximum, max(coordinates))
        extent = 511 if axis_offset == 0 else 383
        return -minimum, extent - maximum

    result = np.zeros_like(gradient)
    selected = []
    for track_index, track in enumerate(lift.g1.tracks):
        if not any(lift.g1.knots[index].pair_index in pair_set for index in track.knot_indices):
            continue
        lower, upper = translation_bound(track_index)
        if not lower <= realized_shift <= upper:
            continue
        parameter_index = 2 * track_index + axis_offset
        if not lift.parameter_names[parameter_index].endswith(suffix):
            raise DirectDescriptionError("warm-start coherent proposal parameter ordering differs")
        selected.append(parameter_index)
    if not selected:
        raise DirectDescriptionError("warm-start coherent proposal selected zero receiver DOFs")
    result[np.asarray(selected, dtype=np.int64)] = np.float32(optimizer_sign)
    return np.ascontiguousarray(result)


def save_stage_checkpoint(
    path: Path,
    state: AdamStateV1,
    *,
    stage_id: str,
    config: DirectDescriptionJointDescentTypedConfigV1,
    telemetry: Sequence[Mapping[str, Any]],
    run_cursor: Mapping[str, Any] | None = None,
    realized_archive: Mapping[str, Any] | None = None,
) -> str:
    """Atomically preserve a distinct stage/step checkpoint; never overwrite."""

    if path.exists():
        raise DirectDescriptionError(f"preserved joint-descent checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "stage_id": stage_id,
        "step": state.step,
        "typed_config_hash": config.typed_config_hash(),
        "dsl_compile_hash": config.dsl_compile_hash,
        "source_archive_sha256": config.source_archive_sha256,
        "target_cache_sha256": config.target_cache_sha256,
        "seed": config.seed,
        "rng": {"kind": "deterministic_no_sampling", "state": config.seed},
        "ema_shadow_saved": True,
        "live_weights_saved_for_resume_only": True,
        "optimizer": "adam_fp32",
        "canonical_resume_registry": {
            "helper": "tac.witness_control.resume_registry.ResumeRegistry",
            "controller": "ddm_joint_descent_optimizer",
            "prefix": "__ddmjd_",
            "manifest_key": RESUME_REGISTRY_MANIFEST_KEY,
        },
        "telemetry": list(telemetry),
        "run_cursor": dict(run_cursor or {}),
        "realized_archive": dict(realized_archive or {}),
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
        "pointer": POINTER,
        "pointer_moved": False,
    }
    registry_arrays = _optimizer_resume_registry(state, config).state_arrays()
    if RESUME_REGISTRY_MANIFEST_KEY not in registry_arrays:
        raise DirectDescriptionError("joint-descent checkpoint lacks canonical resume manifest")
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            theta=state.theta,
            ema=state.ema,
            first_moment=state.first_moment,
            second_moment=state.second_moment,
            metadata=np.frombuffer(rfc8785_canonicalize(metadata), dtype=np.uint8),
            **registry_arrays,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return _sha256(path.read_bytes())


def load_stage_checkpoint(
    path: Path,
    *,
    config: DirectDescriptionJointDescentTypedConfigV1,
) -> tuple[AdamStateV1, dict[str, Any]]:
    payload = path.read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(np.asarray(archive["metadata"], dtype=np.uint8).tobytes())
        if metadata.get("schema") != CHECKPOINT_SCHEMA:
            raise DirectDescriptionError("joint-descent checkpoint schema differs")
        if metadata.get("typed_config_hash") != config.typed_config_hash():
            raise DirectDescriptionError("joint-descent checkpoint typed config differs")
        state = AdamStateV1(
            step=int(metadata["step"]),
            theta=np.ascontiguousarray(archive["theta"], dtype=np.float32),
            ema=np.ascontiguousarray(archive["ema"], dtype=np.float32),
            first_moment=np.ascontiguousarray(archive["first_moment"], dtype=np.float32),
            second_moment=np.ascontiguousarray(archive["second_moment"], dtype=np.float32),
        )
        shapes = {
            state.theta.shape,
            state.ema.shape,
            state.first_moment.shape,
            state.second_moment.shape,
        }
        if (
            len(shapes) != 1
            or len(state.theta.shape) != 1
            or not all(
                np.all(np.isfinite(value))
                for value in (state.theta, state.ema, state.first_moment, state.second_moment)
            )
        ):
            raise DirectDescriptionError("joint-descent checkpoint optimizer arrays are invalid")
        cfg = {key: np.asarray(archive[key]).item() for key in archive.files if key.startswith("__")}
        report = _optimizer_resume_registry(state, config).restore(cfg)
        if not report.manifest_present or report.restored != {"ddm_joint_descent_optimizer": True}:
            raise DirectDescriptionError("joint-descent canonical resume-registry restore is incomplete")
    return state, metadata


def classify_memory_preflight(projected_peak_gib: float, *, ceiling_gib: float = 116.0) -> tuple[bool, str]:
    peak = float(projected_peak_gib)
    ceiling = float(ceiling_gib)
    if not math.isfinite(peak) or peak <= 0.0:
        return False, "REFUSE_INVALID_MEASURED_PEAK"
    if peak > ceiling:
        return False, "REFUSE_PROJECTED_PEAK_EXCEEDS_116_GIB_CEILING"
    return True, "SAFE_PROJECTED_PEAK_WITHIN_116_GIB_CEILING"


__all__ = [
    "J5_PROGRAM_SHA256",
    "J6A_PROGRAM_SHA256",
    "J9_W_JOINT_PROGRAM_SHA256",
    "J10_PROGRAM_ID",
    "AdamStateV1",
    "DirectDescriptionJointDescentMLXModule",
    "DirectDescriptionJointDescentTypedConfigV1",
    "FullRunScheduleV1",
    "FullRunStageV1",
    "JointDescentResumeControllerV1",
    "JointDescriptionParameterLiftV1",
    "LaneProgramSeedV1",
    "PoseFinishEngageConfigV1",
    "PoseFinishEngageStateV1",
    "ProposalGeometryInfeasibleError",
    "VerdictShadowPolicyV1",
    "WarmStartReformV1",
    "WorstGeometryMemoryContractV1",
    "classify_cumulative_fire_gate",
    "classify_governed_stage_exit",
    "classify_memory_preflight",
    "classify_realized_stage_verdict",
    "classify_shadow_consistent_stage_verdict",
    "classify_verdict_informativeness",
    "clipped_adam_step",
    "compile_parameterized_archive",
    "count_consecutive_scheduled_degenerate_verdicts",
    "derive_lane_program_seeds",
    "exact_final_target_gate",
    "initial_adam_state",
    "lift_v15_archive",
    "linear_rewarmup_factor",
    "load_stage_checkpoint",
    "opening_candidate_gradient",
    "parameter_group_indices",
    "project_adam_state_geometry",
    "realize_parameter_theta",
    "realized_training_state",
    "save_stage_checkpoint",
    "template_camera_state",
    "verify_trainable_group_ownership",
]
