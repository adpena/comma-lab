# SPDX-License-Identifier: MIT
"""Pair-local evaluator-surface servo for NeRV long-run readiness.

The contest scorer is not a smooth RGB reconstruction metric.  A useful local
training action must survive the same receiver surfaces that ``evaluate.py``
uses: uint8 emission, scorer preprocessing, SegNet argmax / PoseNet movement,
fake-quantization, archive parse-back, and finally the exact nonlinear contest
objective.  This module is the shared torch-free admission kernel for those
actions so HiNeRV and SNeRV do not each grow a subtly different proxy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS
from tac.score_geometry import (
    CONTEST_REFERENCE_BYTES,
    contest_score,
)

PAIR_LOCAL_DISTORTION_SERVO_ADMISSION_SCHEMA = (
    "nerv_pair_local_distortion_servo_admission.v1"
)
PAIR_LOCAL_DISTORTION_SERVO_STATIC_CONTRACT_SCHEMA = (
    "nerv_pair_local_distortion_servo_static_contract.v1"
)
PAIR_LOCAL_DISTORTION_SERVO_RECEIPT_SCHEMA = (
    "nerv_pair_local_distortion_servo_receipt.v1"
)
PAIR_LOCAL_DISTORTION_SERVO_REPORT_SCHEMA = (
    "nerv_pair_local_distortion_servo_report.v1"
)

PR95_SERVO_CURRICULUM_STAGES: tuple[str, ...] = (
    "ce_birth",
    "tau_softplus_margin",
    "smooth_disagreement",
    "round_ste_eval_surface",
    "fakequant_qat",
    "hard_pixel_c1a_entropy",
    "lambda_sigma_trust_region",
    "final_optimizer_polish",
)

PR95_SERVO_AUTHORITY_ORDER: tuple[str, ...] = (
    "live_mlx",
    "round_ste_mlx",
    "fakequant_mlx",
    "parseback_mlx",
    "inflate_torch_cpu",
    "inflate_torch_cuda",
)

PR95_SERVO_PROMOTABLE_AUTHORITIES: frozenset[str] = frozenset(
    {"parseback_mlx", "inflate_torch_cpu", "inflate_torch_cuda"}
)

FrameScope = Literal["frame0_pose_only", "frame1_seg_pose_joint", "both_frames_joint"]


@dataclass(frozen=True)
class PairLocalScoreState:
    """Exact contest objective state at one scorer surface."""

    d_seg: float
    d_pose: float
    archive_bytes: int

    def validate(self) -> None:
        if self.d_seg < 0.0 or self.d_pose < 0.0 or self.archive_bytes < 0:
            raise ValueError("score state values must be non-negative")
        if not (math.isfinite(self.d_seg) and math.isfinite(self.d_pose)):
            raise ValueError("score state distortion values must be finite")

    def score(self, *, reference_bytes: int = CONTEST_REFERENCE_BYTES) -> float:
        self.validate()
        return contest_score(
            self.d_seg,
            self.d_pose,
            self.archive_bytes,
            reference_bytes=reference_bytes,
        )


@dataclass(frozen=True)
class PairLocalSurfaceTrace:
    """Measured receiver-surface movement for one pair-local action."""

    family: Literal["hinerv", "snerv", "shared"]
    frame_scope: FrameScope
    actuator_id: str
    pair_index: int | None = None
    float_rgb_delta_linf: float | None = None
    uint8_changed_pixels: int | None = None
    uint8_delta_abs_max: float | None = None
    segnet_input_delta_linf: float | None = None
    posenet_input_delta_linf: float | None = None
    segnet_margin_delta: float | None = None
    segnet_argmax_flipped_pixels: int | None = None
    pose_output_delta_l2: float | None = None
    fakequant_segnet_margin_delta: float | None = None
    fakequant_argmax_flipped_pixels: int | None = None
    fakequant_pose_output_delta_l2: float | None = None
    parseback_segnet_margin_delta: float | None = None
    parseback_argmax_flipped_pixels: int | None = None
    parseback_pose_output_delta_l2: float | None = None
    inflated_argmax_flipped_pixels: int | None = None
    inflated_pose_output_delta_l2: float | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PairLocalSurfaceTrace:
        """Build a trace from canonical runner/crux-trace key aliases."""

        family = str(payload.get("family") or payload.get("substrate_family") or "shared")
        if family == "hi_nerv":
            family = "hinerv"
        if family not in {"hinerv", "snerv", "shared"}:
            family = "shared"
        frame_scope = str(payload.get("frame_scope") or payload.get("scope") or "")
        if frame_scope not in {
            "frame0_pose_only",
            "frame1_seg_pose_joint",
            "both_frames_joint",
        }:
            frame_scope = "both_frames_joint"
        return cls(
            family=family,  # type: ignore[arg-type]
            frame_scope=frame_scope,  # type: ignore[arg-type]
            actuator_id=str(payload.get("actuator_id") or payload.get("actuator") or "unknown"),
            pair_index=_int_or_none(payload.get("pair_index")),
            float_rgb_delta_linf=_first_float(
                payload,
                "receiver_surface_float_rgb_delta_linf",
                "float_rgb_delta_linf",
            ),
            uint8_changed_pixels=_first_int(
                payload,
                "receiver_surface_uint8_changed_pixels",
                "uint8_changed_pixels",
            ),
            uint8_delta_abs_max=_first_float(
                payload,
                "receiver_surface_uint8_delta_abs_max",
                "uint8_delta_abs_max",
                "max_accepted_frame1_receiver_uint8_delta_abs",
            ),
            segnet_input_delta_linf=_first_float(
                payload,
                "receiver_surface_segnet_input_delta_linf",
                "segnet_input_delta_linf",
            ),
            posenet_input_delta_linf=_first_float(
                payload,
                "receiver_surface_posenet_input_delta_linf",
                "posenet_input_delta_linf",
            ),
            segnet_margin_delta=_first_float(
                payload,
                "receiver_surface_worst_region_margin_p50_delta",
                "worst_region_margin_p50_delta",
                "segnet_margin_delta",
                "margin_delta",
            ),
            segnet_argmax_flipped_pixels=_first_int(
                payload,
                "receiver_surface_argmax_flipped_pixels",
                "argmax_flipped_pixels",
            ),
            pose_output_delta_l2=_first_float(
                payload,
                "receiver_surface_pose_output_delta",
                "pose_output_delta",
                "pose_output_delta_l2",
            ),
            fakequant_segnet_margin_delta=_first_float(
                payload,
                "receiver_surface_fakequant_margin_delta",
                "fakequant_margin_delta",
                "fakequant_segnet_margin_delta",
            ),
            fakequant_argmax_flipped_pixels=_first_int(
                payload,
                "receiver_surface_fakequant_argmax_flipped_pixels",
                "fakequant_argmax_flipped_pixels",
            ),
            fakequant_pose_output_delta_l2=_first_float(
                payload,
                "receiver_surface_fakequant_pose_output_delta",
                "fakequant_pose_output_delta",
                "fakequant_pose_output_delta_l2",
            ),
            parseback_segnet_margin_delta=_first_float(
                payload,
                "receiver_surface_parseback_margin_delta",
                "parseback_margin_delta",
                "parseback_segnet_margin_delta",
            ),
            parseback_argmax_flipped_pixels=_first_int(
                payload,
                "receiver_surface_parseback_argmax_flipped_pixels",
                "parseback_argmax_flipped_pixels",
            ),
            parseback_pose_output_delta_l2=_first_float(
                payload,
                "receiver_surface_parseback_pose_output_delta",
                "parseback_pose_output_delta",
                "parseback_pose_output_delta_l2",
            ),
            inflated_argmax_flipped_pixels=_first_int(
                payload,
                "receiver_surface_inflated_argmax_flipped_pixels",
                "inflated_argmax_flipped_pixels",
            ),
            inflated_pose_output_delta_l2=_first_float(
                payload,
                "receiver_surface_inflated_pose_output_delta",
                "inflated_pose_output_delta",
                "inflated_pose_output_delta_l2",
            ),
        )


@dataclass(frozen=True)
class PairLocalServoAdmission:
    """Exact admission decision for one pair-local action."""

    schema: str
    admitted: bool
    blockers: tuple[str, ...]
    family: str
    frame_scope: str
    actuator_id: str
    pair_index: int | None
    score_before: float
    score_after: float
    exact_score_delta: float
    delta_score_nonrate: float
    delta_d_seg: float
    delta_d_pose: float
    delta_archive_bytes: int
    rate_score_delta: float
    value_per_added_byte: float | None
    surfaces: Mapping[str, bool]
    trace: Mapping[str, Any]
    evaluator_formula: str
    axis_tag: str = "[planning/control]"
    evidence_grade: str = "[local deterministic admission contract]"
    score_claim: bool = False
    promotion_eligible: bool = False
    rank_or_kill_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(PROXY_FALSE_AUTHORITY_FIELDS)
        return payload


@dataclass(frozen=True)
class ScorerDebtTarget:
    """One pair/region debt target in exact score units."""

    target_id: str
    score_units: float
    axis: Literal["seg", "pose", "joint"]
    pair_index: int | None = None
    frame_scope: FrameScope = "both_frames_joint"
    raw: Mapping[str, Any] | None = None


def exact_pair_local_score_delta(
    before: PairLocalScoreState,
    after: PairLocalScoreState,
    *,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> float:
    """Return finite exact ``S(after)-S(before)`` for contest admission."""

    return after.score(reference_bytes=reference_bytes) - before.score(
        reference_bytes=reference_bytes
    )


def admit_pair_local_distortion_action(
    *,
    before: PairLocalScoreState,
    after: PairLocalScoreState,
    trace: PairLocalSurfaceTrace | Mapping[str, Any],
    min_score_improvement: float = 0.0,
    min_margin_delta: float = 1.0e-6,
    min_pose_output_delta: float = 1.0e-9,
    require_fakequant_survival: bool = True,
    require_parseback_survival: bool = True,
    require_inflate_survival: bool = False,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> PairLocalServoAdmission:
    """Admit only receiver-surviving, exact-score-improving pair actions."""

    if min_score_improvement < 0.0:
        raise ValueError("min_score_improvement must be non-negative")
    if min_margin_delta < 0.0 or min_pose_output_delta < 0.0:
        raise ValueError("movement floors must be non-negative")
    trace_obj = (
        trace if isinstance(trace, PairLocalSurfaceTrace) else PairLocalSurfaceTrace.from_mapping(trace)
    )
    before.validate()
    after.validate()
    score_before = before.score(reference_bytes=reference_bytes)
    score_after = after.score(reference_bytes=reference_bytes)
    score_delta = score_after - score_before
    delta_archive_bytes = after.archive_bytes - before.archive_bytes
    rate_score_delta = delta_archive_bytes * (25.0 / reference_bytes)
    delta_score_nonrate = score_delta - rate_score_delta
    value_per_added_byte = (
        (-delta_score_nonrate / float(delta_archive_bytes))
        if delta_archive_bytes > 0
        else None
    )
    surfaces = _surface_flags(
        trace_obj,
        min_margin_delta=min_margin_delta,
        min_pose_output_delta=min_pose_output_delta,
    )
    blockers = _servo_blockers(
        before=before,
        after=after,
        trace=trace_obj,
        surfaces=surfaces,
        score_delta=score_delta,
        min_score_improvement=min_score_improvement,
        require_fakequant_survival=require_fakequant_survival,
        require_parseback_survival=require_parseback_survival,
        require_inflate_survival=require_inflate_survival,
    )
    return PairLocalServoAdmission(
        schema=PAIR_LOCAL_DISTORTION_SERVO_ADMISSION_SCHEMA,
        admitted=not blockers,
        blockers=tuple(blockers),
        family=trace_obj.family,
        frame_scope=trace_obj.frame_scope,
        actuator_id=trace_obj.actuator_id,
        pair_index=trace_obj.pair_index,
        score_before=score_before,
        score_after=score_after,
        exact_score_delta=score_delta,
        delta_score_nonrate=delta_score_nonrate,
        delta_d_seg=after.d_seg - before.d_seg,
        delta_d_pose=after.d_pose - before.d_pose,
        delta_archive_bytes=delta_archive_bytes,
        rate_score_delta=rate_score_delta,
        value_per_added_byte=value_per_added_byte,
        surfaces=surfaces,
        trace=asdict(trace_obj),
        evaluator_formula=(
            "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489"
        ),
    )


def select_worst_scorer_debt_target(
    debts: Sequence[ScorerDebtTarget | Mapping[str, Any]],
) -> ScorerDebtTarget:
    """Return the highest exact-score-unit debt target."""

    if not debts:
        raise ValueError("at least one scorer debt target is required")
    parsed = [
        item if isinstance(item, ScorerDebtTarget) else _debt_from_mapping(item)
        for item in debts
    ]
    for item in parsed:
        if item.score_units < 0.0 or not math.isfinite(item.score_units):
            raise ValueError("score debt targets must have finite non-negative score_units")
    return max(parsed, key=lambda item: item.score_units)


def seg_argmax_pixel_debt_score_units(
    *,
    wrong_pixels: int,
    total_scored_pixels: int,
) -> float:
    """Return exact SegNet score debt for wrong last-frame argmax pixels."""

    if wrong_pixels < 0 or total_scored_pixels <= 0:
        raise ValueError("pixel counts must be wrong>=0 and total>0")
    if wrong_pixels > total_scored_pixels:
        raise ValueError("wrong_pixels cannot exceed total_scored_pixels")
    return 100.0 * (wrong_pixels / total_scored_pixels)


def byte_cost_score_units(
    added_bytes: int,
    *,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> float:
    """Return exact score cost for adding archive bytes."""

    if added_bytes < 0:
        raise ValueError("added_bytes must be non-negative")
    return 25.0 * float(added_bytes) / reference_bytes


def pair_local_servo_static_contract() -> dict[str, Any]:
    """Return the durable implementation contract consumed by readiness DAGs."""

    return {
        "schema": PAIR_LOCAL_DISTORTION_SERVO_STATIC_CONTRACT_SCHEMA,
        "servo": (
            "worst_scorer_debt -> pair_local_actuator -> uint8/preprocess/"
            "logit_argmax_pose_movement -> fakequant/archive_parseback_survival "
            "-> exact_nonlinear_score_admission"
        ),
        "families": ["hinerv", "snerv"],
        "frame_incidence": {
            "frame0_pose_only": {
                "segnet_direct_price": 0.0,
                "required_surfaces": ["uint8", "posenet_preprocess", "pose_output"],
            },
            "frame1_seg_pose_joint": {
                "segnet_direct_price": 100.0,
                "required_surfaces": [
                    "uint8",
                    "segnet_preprocess",
                    "segnet_argmax_or_margin",
                    "pose_guard",
                ],
            },
        },
        "exact_admission_rule": (
            "S_after - S_before < -min_score_improvement, where "
            "S=100*d_seg+sqrt(10*d_pose)+25*archive_bytes/37545489"
        ),
        "survival_gates": [
            "uint8_motion",
            "scorer_preprocess_motion",
            "live_scorer_motion",
            "fakequant_survival",
            "archive_parseback_survival",
            "optional_inflate_survival",
        ],
        "pr95_grade_or_better_required_gates": [
            "worst_scorer_debt_selected",
            "pair_frame_incidence_bound",
            "stage_separated_curriculum",
            "architecture_specific_pair_local_actuator",
            "receiver_surface_projection",
            "live_scorer_movement",
            "fakequant_survival",
            "archive_parseback_selection_authority",
            "value_per_byte_exact_score_units",
            "hardware_margin_declared",
            "exact_nonlinear_score_admission",
        ],
        "curriculum_stage_order": list(PR95_SERVO_CURRICULUM_STAGES),
        "authority_order": list(PR95_SERVO_AUTHORITY_ORDER),
        "acceptance_authorities": sorted(PR95_SERVO_PROMOTABLE_AUTHORITIES),
        "score_claim": False,
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def build_pr95_grade_pair_local_servo_report(
    receipt: Mapping[str, Any],
    *,
    family: str | None = None,
    min_score_improvement: float = 0.0,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> dict[str, Any]:
    """Return the PR95-grade-or-better servo report for a launch receipt.

    This is the long-run admission surface.  It intentionally goes beyond
    generic "the loss moved" evidence: the receipt must show a pair-local
    actuator, frame-incidence aware movement, receiver-surface projection,
    fakequant and parse-back survival, value-per-byte accounting, hardware
    margin, and exact nonlinear score improvement.
    """

    if not isinstance(receipt, Mapping):
        raise TypeError("servo receipt must be a mapping")
    family_key = _normalize_family(family or receipt.get("family") or "shared")
    before = _score_state_from_mapping(receipt, prefix="old")
    after = _score_state_from_mapping(receipt, prefix="new")
    trace = PairLocalSurfaceTrace.from_mapping(
        {
            **receipt,
            "family": family_key,
            "frame_scope": receipt.get("frame_scope") or receipt.get("frame_incidence"),
            "actuator_id": receipt.get("actuator_id") or receipt.get("actuator_kind"),
        }
    )
    admission = admit_pair_local_distortion_action(
        before=before,
        after=after,
        trace=trace,
        min_score_improvement=min_score_improvement,
        require_fakequant_survival=True,
        require_parseback_survival=True,
        reference_bytes=reference_bytes,
    )
    stage_rows = [
        _stage_row(
            "receipt_schema",
            receipt.get("schema") == PAIR_LOCAL_DISTORTION_SERVO_RECEIPT_SCHEMA,
            {
                "schema": receipt.get("schema"),
                "required_schema": PAIR_LOCAL_DISTORTION_SERVO_RECEIPT_SCHEMA,
            },
            "pair_local_servo_receipt_schema_missing",
        ),
        _stage_row(
            "worst_scorer_debt_selected",
            _worst_debt_target_ok(receipt),
            _worst_debt_evidence(receipt),
            "pair_local_servo_worst_scorer_debt_target_missing",
        ),
        _stage_row(
            "pair_frame_incidence_bound",
            _frame_incidence_ok(receipt, trace),
            _frame_incidence_evidence(receipt, trace),
            "pair_local_servo_frame_incidence_missing",
        ),
        _stage_row(
            "stage_separated_curriculum",
            _curriculum_ok(receipt),
            _curriculum_evidence(receipt),
            "pair_local_servo_pr95_curriculum_stage_manifest_missing",
        ),
        _stage_row(
            "architecture_specific_pair_local_actuator",
            _actuator_ok(receipt, family=family_key),
            _actuator_evidence(receipt, family=family_key),
            "pair_local_servo_architecture_specific_actuator_missing",
        ),
        _stage_row(
            "receiver_surface_projection",
            bool(
                admission.surfaces.get("uint8_motion")
                and admission.surfaces.get("scorer_preprocess_motion")
            ),
            {
                "uint8_motion": admission.surfaces.get("uint8_motion"),
                "scorer_preprocess_motion": admission.surfaces.get(
                    "scorer_preprocess_motion"
                ),
                "float_rgb_delta_linf": trace.float_rgb_delta_linf,
                "uint8_changed_pixels": trace.uint8_changed_pixels,
                "segnet_input_delta_linf": trace.segnet_input_delta_linf,
                "posenet_input_delta_linf": trace.posenet_input_delta_linf,
            },
            "pair_local_servo_receiver_surface_projection_missing",
        ),
        _stage_row(
            "live_scorer_movement",
            bool(admission.surfaces.get("live_scorer_motion")),
            {
                "seg_movement": admission.surfaces.get("seg_movement"),
                "pose_movement": admission.surfaces.get("pose_movement"),
                "segnet_argmax_flipped_pixels": trace.segnet_argmax_flipped_pixels,
                "segnet_margin_delta": trace.segnet_margin_delta,
                "pose_output_delta_l2": trace.pose_output_delta_l2,
            },
            "pair_local_servo_live_scorer_movement_missing",
        ),
        _stage_row(
            "fakequant_survival",
            bool(admission.surfaces.get("fakequant_survival")),
            {
                "fakequant_segnet_margin_delta": trace.fakequant_segnet_margin_delta,
                "fakequant_argmax_flipped_pixels": trace.fakequant_argmax_flipped_pixels,
                "fakequant_pose_output_delta_l2": trace.fakequant_pose_output_delta_l2,
            },
            "pair_local_servo_fakequant_survival_missing",
        ),
        _stage_row(
            "archive_parseback_survival",
            _parseback_authority_ok(receipt, admission),
            {
                "authority": receipt.get("authority"),
                "parseback_survival": admission.surfaces.get("parseback_survival"),
                "parseback_segnet_margin_delta": trace.parseback_segnet_margin_delta,
                "parseback_argmax_flipped_pixels": trace.parseback_argmax_flipped_pixels,
                "parseback_pose_output_delta_l2": trace.parseback_pose_output_delta_l2,
            },
            "pair_local_servo_archive_parseback_authority_missing",
        ),
        _stage_row(
            "action_algebra_effect_bound",
            _action_algebra_ok(receipt),
            _action_algebra_evidence(receipt),
            "pair_local_servo_action_algebra_effect_missing",
        ),
        _stage_row(
            "value_per_byte_priced",
            _value_per_byte_ok(receipt, before=before, after=after),
            _value_per_byte_evidence(receipt, before=before, after=after),
            "pair_local_servo_value_per_byte_not_priced",
        ),
        _stage_row(
            "hardware_margin_bound",
            _hardware_margin_ok(receipt),
            _hardware_margin_evidence(receipt),
            "pair_local_servo_hardware_margin_missing",
        ),
        _stage_row(
            "exact_nonlinear_score_admitted",
            admission.admitted,
            {
                "score_before": admission.score_before,
                "score_after": admission.score_after,
                "exact_score_delta": admission.exact_score_delta,
                "delta_d_seg": admission.delta_d_seg,
                "delta_d_pose": admission.delta_d_pose,
                "delta_archive_bytes": admission.delta_archive_bytes,
                "rate_score_delta": admission.rate_score_delta,
            },
            "pair_local_servo_exact_nonlinear_score_not_improved",
        ),
    ]
    blockers = [
        *admission.blockers,
        *[
            str(row["blocker"])
            for row in stage_rows
            if row["green"] is not True and row.get("blocker")
        ],
    ]
    report = {
        "schema": PAIR_LOCAL_DISTORTION_SERVO_REPORT_SCHEMA,
        "receipt_schema": receipt.get("schema"),
        "family": family_key,
        "pair_ids": list(_pair_ids(receipt)),
        "authority": receipt.get("authority"),
        "long_run_admission_ready": not _dedupe(blockers),
        "admitted": not _dedupe(blockers),
        "stage_rows": stage_rows,
        "blockers": _dedupe(blockers),
        "score_before": admission.score_before,
        "score_after": admission.score_after,
        "exact_score_delta": admission.exact_score_delta,
        "delta_score_nonrate": (
            100.0 * admission.delta_d_seg
            + (
                math.sqrt(10.0 * after.d_pose)
                - math.sqrt(10.0 * before.d_pose)
            )
        ),
        "rate_score_delta": admission.rate_score_delta,
        "value_per_byte": _measured_value_per_byte(receipt, before=before, after=after),
        "byte_price": 25.0 / reference_bytes,
        "surfaces": dict(admission.surfaces),
        "static_contract": pair_local_servo_static_contract(),
        "policy": {
            "human_visual_fidelity_is_not_authority": True,
            "pose_uses_exact_sqrt_term_not_linearized_acceptance": True,
            "segnet_frame1_and_posenet_both_frames_incidence_required": True,
            "updates_must_survive_uint8_preprocess_fakequant_parseback": True,
            "selection_authority_must_be_parseback_or_inflate": True,
            "score_value_per_byte_must_exceed_byte_price_when_bytes_increase": True,
            "score_claim": False,
        },
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }
    return report


def pair_local_servo_receipt_ready(
    receipt: Mapping[str, Any],
    *,
    family: str | None = None,
) -> bool:
    """Return True only when the PR95-grade servo report is fully green."""

    return bool(
        build_pr95_grade_pair_local_servo_report(
            receipt,
            family=family,
        ).get("long_run_admission_ready")
    )


def _normalize_family(value: Any) -> str:
    family = str(value or "shared").strip().lower().replace("-", "_")
    if family == "hi_nerv":
        return "hinerv"
    if family in {"hinerv", "snerv"}:
        return family
    return "shared"


def _score_state_from_mapping(
    payload: Mapping[str, Any],
    *,
    prefix: Literal["old", "new"],
) -> PairLocalScoreState:
    nested_keys = (prefix, "before") if prefix == "old" else (prefix, "after")
    for key in nested_keys:
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            return PairLocalScoreState(
                d_seg=_required_float(nested, "d_seg"),
                d_pose=_required_float(nested, "d_pose"),
                archive_bytes=_required_int(nested, "archive_bytes"),
            )
    alt = "before" if prefix == "old" else "after"
    return PairLocalScoreState(
        d_seg=_required_first_float(
            payload,
            f"{prefix}_d_seg",
            f"d_seg_{alt}",
            f"{alt}_d_seg",
        ),
        d_pose=_required_first_float(
            payload,
            f"{prefix}_d_pose",
            f"d_pose_{alt}",
            f"{alt}_d_pose",
        ),
        archive_bytes=_required_first_int(
            payload,
            f"{prefix}_archive_bytes",
            f"archive_bytes_{alt}",
            f"{alt}_archive_bytes",
        ),
    )


def _stage_row(
    name: str,
    green: bool,
    evidence: Mapping[str, Any],
    blocker: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "green": bool(green),
        "evidence": dict(evidence),
        "blocker": None if green else blocker,
    }


def _worst_debt_target_ok(receipt: Mapping[str, Any]) -> bool:
    return bool(_worst_debt_target(receipt))


def _worst_debt_evidence(receipt: Mapping[str, Any]) -> dict[str, Any]:
    target = _worst_debt_target(receipt)
    if target is None:
        return {
            "target_id": receipt.get("target_id"),
            "score_units": receipt.get("score_units"),
        }
    return {
        "target_id": target.target_id,
        "score_units": target.score_units,
        "axis": target.axis,
        "pair_index": target.pair_index,
        "frame_scope": target.frame_scope,
    }


def _worst_debt_target(receipt: Mapping[str, Any]) -> ScorerDebtTarget | None:
    raw_target = receipt.get("worst_scorer_debt_target") or receipt.get(
        "selected_debt_target"
    )
    candidates: list[ScorerDebtTarget | Mapping[str, Any]] = []
    if isinstance(raw_target, Mapping):
        candidates.append(raw_target)
    if not candidates and receipt.get("target_id"):
        candidates.append(
            {
                "target_id": receipt.get("target_id"),
                "score_units": receipt.get("score_units")
                or receipt.get("worst_debt_score_units"),
                "axis": receipt.get("axis") or receipt.get("target_axis"),
                "pair_index": receipt.get("pair_index"),
                "frame_scope": receipt.get("frame_scope")
                or receipt.get("frame_incidence"),
            }
        )
    try:
        return select_worst_scorer_debt_target(candidates) if candidates else None
    except ValueError:
        return None


def _frame_incidence_ok(
    receipt: Mapping[str, Any],
    trace: PairLocalSurfaceTrace,
) -> bool:
    return trace.frame_scope in {
        "frame0_pose_only",
        "frame1_seg_pose_joint",
        "both_frames_joint",
    } and bool(_pair_ids(receipt) or trace.pair_index is not None)


def _frame_incidence_evidence(
    receipt: Mapping[str, Any],
    trace: PairLocalSurfaceTrace,
) -> dict[str, Any]:
    return {
        "frame_scope": trace.frame_scope,
        "pair_ids": list(_pair_ids(receipt)),
        "pair_index": trace.pair_index,
        "segnet_incidence": "none" if trace.frame_scope == "frame0_pose_only" else "last_frame",
        "posenet_incidence": "both_frames",
    }


def _curriculum_ok(receipt: Mapping[str, Any]) -> bool:
    stage = str(receipt.get("stage") or receipt.get("curriculum_stage") or "")
    if stage in PR95_SERVO_CURRICULUM_STAGES:
        return True
    stage_order = receipt.get("stage_order") or receipt.get("curriculum_stage_order")
    if not isinstance(stage_order, Sequence) or isinstance(stage_order, (str, bytes)):
        return False
    observed = {str(item) for item in stage_order}
    return bool(observed.intersection(PR95_SERVO_CURRICULUM_STAGES))


def _curriculum_evidence(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": receipt.get("stage") or receipt.get("curriculum_stage"),
        "stage_order": list(receipt.get("stage_order") or receipt.get("curriculum_stage_order") or []),
        "required_stage_order": list(PR95_SERVO_CURRICULUM_STAGES),
    }


def _actuator_ok(receipt: Mapping[str, Any], *, family: str) -> bool:
    names = _actuator_names(receipt)
    if not names:
        return False
    if family == "hinerv":
        allowed = ("hinerv", "latents_fine", "output_head", "high_grid", "pair_adapter", "birth_basis")
    elif family == "snerv":
        allowed = ("snerv", "mfu", "hfr", "tub", "lf", "hf", "output_2")
    else:
        allowed = ("pair", "servo", "actuator")
    return any(any(token in name for token in allowed) for name in names)


def _actuator_evidence(receipt: Mapping[str, Any], *, family: str) -> dict[str, Any]:
    return {
        "family": family,
        "actuator_id": receipt.get("actuator_id") or receipt.get("actuator_kind"),
        "trained_param_groups": list(receipt.get("trained_param_groups") or []),
    }


def _actuator_names(receipt: Mapping[str, Any]) -> tuple[str, ...]:
    names = [
        str(receipt.get("actuator_id") or "").lower(),
        str(receipt.get("actuator_kind") or "").lower(),
    ]
    groups = receipt.get("trained_param_groups")
    if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)):
        names.extend(str(item).lower() for item in groups)
    return tuple(name for name in names if name)


def _parseback_authority_ok(
    receipt: Mapping[str, Any],
    admission: PairLocalServoAdmission,
) -> bool:
    authority = str(receipt.get("authority") or "").strip()
    archive_identity = bool(
        receipt.get("archive_sha256")
        or receipt.get("archive_path")
        or receipt.get("packet_sha256")
        or receipt.get("packet_path")
    )
    return (
        authority in PR95_SERVO_PROMOTABLE_AUTHORITIES
        and bool(admission.surfaces.get("parseback_survival"))
        and archive_identity
    )


def _action_algebra_ok(receipt: Mapping[str, Any]) -> bool:
    return bool(
        receipt.get("action_id")
        or receipt.get("action_effect")
        or receipt.get("effect_vector")
        or receipt.get("transform_id")
    )


def _action_algebra_evidence(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action_id": receipt.get("action_id"),
        "transform_id": receipt.get("transform_id"),
        "action_effect": receipt.get("action_effect"),
        "effect_vector": receipt.get("effect_vector"),
    }


def _value_per_byte_ok(
    receipt: Mapping[str, Any],
    *,
    before: PairLocalScoreState,
    after: PairLocalScoreState,
) -> bool:
    added_bytes = after.archive_bytes - before.archive_bytes
    if added_bytes <= 0:
        return True
    value = _measured_value_per_byte(receipt, before=before, after=after)
    if value is None:
        return False
    return value > 25.0 / CONTEST_REFERENCE_BYTES


def _value_per_byte_evidence(
    receipt: Mapping[str, Any],
    *,
    before: PairLocalScoreState,
    after: PairLocalScoreState,
) -> dict[str, Any]:
    value = _measured_value_per_byte(receipt, before=before, after=after)
    added_bytes = after.archive_bytes - before.archive_bytes
    return {
        "added_bytes": added_bytes,
        "value_per_byte": value,
        "byte_price": 25.0 / CONTEST_REFERENCE_BYTES,
        "receipt_value_per_byte": receipt.get("value_per_byte")
        or receipt.get("value_per_added_byte"),
    }


def _measured_value_per_byte(
    receipt: Mapping[str, Any],
    *,
    before: PairLocalScoreState,
    after: PairLocalScoreState,
) -> float | None:
    explicit = _first_float(receipt, "value_per_byte", "value_per_added_byte")
    if explicit is not None:
        return explicit
    added_bytes = after.archive_bytes - before.archive_bytes
    if added_bytes <= 0:
        return None
    old_nonrate = 100.0 * before.d_seg + math.sqrt(10.0 * before.d_pose)
    new_nonrate = 100.0 * after.d_seg + math.sqrt(10.0 * after.d_pose)
    return (old_nonrate - new_nonrate) / float(added_bytes)


def _hardware_margin_ok(receipt: Mapping[str, Any]) -> bool:
    if receipt.get("authority") == "inflate_torch_cuda":
        return True
    margin = receipt.get("hardware_margin") or receipt.get("cpu_cuda_margin")
    if isinstance(margin, Mapping):
        return bool(
            margin.get("declared")
            or margin.get("pass")
            or margin.get("safe")
            or margin.get("cpu_cuda_safe")
        )
    return bool(receipt.get("hardware_margin_declared"))


def _hardware_margin_evidence(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "authority": receipt.get("authority"),
        "hardware_margin": receipt.get("hardware_margin"),
        "cpu_cuda_margin": receipt.get("cpu_cuda_margin"),
        "hardware_margin_declared": receipt.get("hardware_margin_declared"),
    }


def _pair_ids(receipt: Mapping[str, Any]) -> tuple[int, ...]:
    raw = receipt.get("pair_ids")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        ids = tuple(item for item in (_int_or_none(value) for value in raw) if item is not None)
        if ids:
            return ids
    for key in ("pair_id", "pair_index"):
        value = _int_or_none(receipt.get(key))
        if value is not None:
            return (value,)
    return ()


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _servo_blockers(
    *,
    before: PairLocalScoreState,
    after: PairLocalScoreState,
    trace: PairLocalSurfaceTrace,
    surfaces: Mapping[str, bool],
    score_delta: float,
    min_score_improvement: float,
    require_fakequant_survival: bool,
    require_parseback_survival: bool,
    require_inflate_survival: bool,
) -> list[str]:
    blockers: list[str] = []
    if trace.actuator_id == "unknown":
        blockers.append("pair_local_servo_actuator_id_missing")
    if score_delta >= -min_score_improvement:
        blockers.append("pair_local_servo_exact_nonlinear_score_not_improved")
    if _positive(trace.float_rgb_delta_linf) and not surfaces["uint8_motion"]:
        blockers.append("pair_local_servo_subquantum_float_update_no_uint8_motion")
    if not surfaces["uint8_motion"]:
        blockers.append("pair_local_servo_receiver_uint8_motion_missing")
    if not surfaces["scorer_preprocess_motion"]:
        blockers.append("pair_local_servo_scorer_preprocess_motion_missing")
    if not surfaces["live_scorer_motion"]:
        blockers.append("pair_local_servo_live_scorer_movement_missing")
    if require_fakequant_survival and not surfaces["fakequant_survival"]:
        blockers.append("pair_local_servo_fakequant_survival_missing")
    if require_parseback_survival and not surfaces["parseback_survival"]:
        blockers.append("pair_local_servo_archive_parseback_survival_missing")
    if require_inflate_survival and not surfaces["inflate_survival"]:
        blockers.append("pair_local_servo_inflate_survival_missing")
    if trace.frame_scope == "frame0_pose_only" and not math.isclose(
        after.d_seg,
        before.d_seg,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        blockers.append("pair_local_servo_frame0_pose_only_changed_segnet_distortion")
    if trace.frame_scope == "frame0_pose_only" and not surfaces["pose_movement"]:
        blockers.append("pair_local_servo_frame0_pose_only_pose_movement_missing")
    if trace.frame_scope == "frame1_seg_pose_joint" and not surfaces["seg_movement"]:
        blockers.append("pair_local_servo_frame1_seg_movement_missing")
    return blockers


def _score_state_from_mapping(
    payload: Mapping[str, Any],
    *,
    prefix: str,
) -> PairLocalScoreState:
    d_seg = _first_float(
        payload,
        f"{prefix}_d_seg",
        f"d_seg_{prefix}",
        f"{prefix}_segnet_distortion",
        f"segnet_distortion_{prefix}",
    )
    d_pose = _first_float(
        payload,
        f"{prefix}_d_pose",
        f"d_pose_{prefix}",
        f"{prefix}_posenet_distortion",
        f"posenet_distortion_{prefix}",
    )
    archive_bytes = _first_int(
        payload,
        f"{prefix}_archive_bytes",
        f"archive_bytes_{prefix}",
    )
    if d_seg is None or d_pose is None or archive_bytes is None:
        raise ValueError(f"servo receipt missing {prefix} score state")
    return PairLocalScoreState(
        d_seg=d_seg,
        d_pose=d_pose,
        archive_bytes=archive_bytes,
    )


def _stage_row(
    stage_id: str,
    green: bool,
    evidence: Mapping[str, Any],
    blocker: str,
) -> dict[str, Any]:
    return {
        "schema": "nerv_pair_local_distortion_servo_stage.v1",
        "stage_id": stage_id,
        "green": bool(green),
        "blocker": None if green else blocker,
        "evidence": dict(evidence),
    }


def _normalize_family(value: Any) -> str:
    family = str(value or "shared").strip().lower().replace("-", "_")
    if family == "hinerv":
        family = "hi_nerv"
    return family or "shared"


def _pair_ids(payload: Mapping[str, Any]) -> tuple[int, ...]:
    value = payload.get("pair_ids")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        out = tuple(
            parsed for item in value if (parsed := _int_or_none(item)) is not None
        )
        if out:
            return out
    one = _int_or_none(payload.get("pair_id"))
    return () if one is None else (one,)


def _worst_debt_target_ok(payload: Mapping[str, Any]) -> bool:
    target = _first_mapping(
        payload,
        "worst_scorer_debt",
        "worst_region_debt",
        "scorer_debt_target",
        "debt_target",
    )
    if not target:
        return False
    before = _first_float(
        target,
        "score_debt_before",
        "old_score_debt",
        "old_worst_region_unsolved_fraction",
        "old_worst_pair_pose_mse",
    )
    after = _first_float(
        target,
        "score_debt_after",
        "new_score_debt",
        "new_worst_region_unsolved_fraction",
        "new_worst_pair_pose_mse",
    )
    target_id = str(
        target.get("target_id")
        or target.get("worst_region_id")
        or target.get("worst_pair_id")
        or ""
    )
    return before is not None and after is not None and after < before and bool(target_id)


def _worst_debt_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    target = _first_mapping(
        payload,
        "worst_scorer_debt",
        "worst_region_debt",
        "scorer_debt_target",
        "debt_target",
    )
    return {
        "target_id": target.get("target_id")
        or target.get("worst_region_id")
        or target.get("worst_pair_id"),
        "score_debt_before": _first_float(
            target,
            "score_debt_before",
            "old_score_debt",
            "old_worst_region_unsolved_fraction",
            "old_worst_pair_pose_mse",
        ),
        "score_debt_after": _first_float(
            target,
            "score_debt_after",
            "new_score_debt",
            "new_worst_region_unsolved_fraction",
            "new_worst_pair_pose_mse",
        ),
    }


def _frame_incidence_ok(
    payload: Mapping[str, Any],
    trace: PairLocalSurfaceTrace,
) -> bool:
    incidence = _first_mapping(payload, "frame_incidence", "pair_frame_incidence")
    frame0_pose = (
        incidence.get("frame0_pose_only") is True
        or incidence.get("frame0_posenet_incidence") is True
        or trace.frame_scope == "frame0_pose_only"
        or trace.frame_scope == "both_frames_joint"
    )
    frame0_seg_zero = (
        incidence.get("frame0_segnet_incidence") in {False, "pose_only", None}
        or trace.frame_scope in {"frame0_pose_only", "both_frames_joint"}
    )
    frame1_seg = (
        incidence.get("frame1_segnet_incidence") is True
        or trace.frame_scope == "frame1_seg_pose_joint"
        or trace.frame_scope == "both_frames_joint"
    )
    frame1_pose = (
        incidence.get("frame1_posenet_incidence") is True
        or trace.frame_scope in {"frame1_seg_pose_joint", "both_frames_joint"}
    )
    split_ok = (
        incidence.get("frame0_frame1_control_split") is True
        or incidence.get("separate_frame_heads") is True
        or incidence.get("pair_local_incidence_bound") is True
    )
    return bool(frame0_pose and frame0_seg_zero and frame1_seg and frame1_pose and split_ok)


def _frame_incidence_evidence(
    payload: Mapping[str, Any],
    trace: PairLocalSurfaceTrace,
) -> dict[str, Any]:
    incidence = _first_mapping(payload, "frame_incidence", "pair_frame_incidence")
    return {
        "frame_scope": trace.frame_scope,
        "frame0_pose_only": incidence.get("frame0_pose_only"),
        "frame0_segnet_incidence": incidence.get("frame0_segnet_incidence"),
        "frame1_segnet_incidence": incidence.get("frame1_segnet_incidence"),
        "frame1_posenet_incidence": incidence.get("frame1_posenet_incidence"),
        "frame0_frame1_control_split": incidence.get("frame0_frame1_control_split"),
        "separate_frame_heads": incidence.get("separate_frame_heads"),
    }


def _curriculum_ok(payload: Mapping[str, Any]) -> bool:
    manifest = _first_mapping(payload, "stage_manifest", "curriculum_trace")
    stages = _string_sequence(
        manifest.get("completed_stage_ids")
        or manifest.get("stage_ids")
        or payload.get("completed_stage_ids")
    )
    stage_index = {stage: idx for idx, stage in enumerate(stages)}
    required_order_ok = all(stage in stage_index for stage in PR95_SERVO_CURRICULUM_STAGES)
    if required_order_ok:
        required_order_ok = [
            stage_index[stage] for stage in PR95_SERVO_CURRICULUM_STAGES
        ] == sorted(stage_index[stage] for stage in PR95_SERVO_CURRICULUM_STAGES)
    gates_ok = (
        manifest.get("stage_order_respected") is True
        and manifest.get("byte_pressure_after_birth") is True
        and manifest.get("qat_after_round_ste") is True
        and manifest.get("final_optimizer_after_survival") is True
    )
    return bool(required_order_ok and gates_ok)


def _curriculum_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _first_mapping(payload, "stage_manifest", "curriculum_trace")
    return {
        "completed_stage_ids": _string_sequence(
            manifest.get("completed_stage_ids")
            or manifest.get("stage_ids")
            or payload.get("completed_stage_ids")
        ),
        "required_stage_ids": list(PR95_SERVO_CURRICULUM_STAGES),
        "stage_order_respected": manifest.get("stage_order_respected"),
        "byte_pressure_after_birth": manifest.get("byte_pressure_after_birth"),
        "qat_after_round_ste": manifest.get("qat_after_round_ste"),
        "final_optimizer_after_survival": manifest.get("final_optimizer_after_survival"),
    }


def _actuator_ok(payload: Mapping[str, Any], *, family: str) -> bool:
    actuator = _first_mapping(payload, "actuation", "actuator", "pair_local_actuator")
    groups = _string_sequence(payload.get("trained_param_groups")) or _string_sequence(
        actuator.get("trained_param_groups")
    )
    grad = _first_mapping(payload, "grad_norm_by_group", "gradient_norm_by_group")
    if not grad:
        grad = _first_mapping(actuator, "grad_norm_by_group", "gradient_norm_by_group")
    updates = _first_mapping(payload, "update_norm_by_group")
    if not updates:
        updates = _first_mapping(actuator, "update_norm_by_group")
    groups_have_signal = bool(groups) and all(
        _positive_float_in_mapping(grad, group)
        and _positive_float_in_mapping(updates, group)
        for group in groups
    )
    pair_local = (
        actuator.get("pair_local") is True
        or actuator.get("pair_locality_verified") is True
        or payload.get("pair_locality_verified") is True
    )
    if family == "hi_nerv":
        family_group = any(
            group.startswith(
                (
                    "head_rgb",
                    "output_head",
                    "high_grid",
                    "fine_grid",
                    "pair_adapter",
                    "birth_basis",
                    "birth_gate",
                    "latents_fine",
                )
            )
            for group in groups
        )
    elif family == "snerv":
        family_group = any(
            group.startswith(("mfu", "hfr", "tub", "output_2", "lf_hf", "lf", "hf"))
            for group in groups
        )
        family_group = family_group and (
            actuator.get("source_forward_replay_bound") is True
            or actuator.get("source_forward_replay_verified") is True
            or actuator.get("mfu_hfr_tub_source_forward_parity_proven") is True
        )
    else:
        family_group = bool(groups)
    return bool(pair_local and groups_have_signal and family_group)


def _actuator_evidence(payload: Mapping[str, Any], *, family: str) -> dict[str, Any]:
    actuator = _first_mapping(payload, "actuation", "actuator", "pair_local_actuator")
    groups = _string_sequence(payload.get("trained_param_groups")) or _string_sequence(
        actuator.get("trained_param_groups")
    )
    return {
        "family": family,
        "actuator_id": payload.get("actuator_id") or actuator.get("actuator_id"),
        "actuator_kind": payload.get("actuator_kind") or actuator.get("kind"),
        "pair_local": actuator.get("pair_local")
        or actuator.get("pair_locality_verified")
        or payload.get("pair_locality_verified"),
        "trained_param_groups": groups,
        "grad_norm_by_group": dict(
            _first_mapping(payload, "grad_norm_by_group", "gradient_norm_by_group")
            or _first_mapping(actuator, "grad_norm_by_group", "gradient_norm_by_group")
        ),
        "update_norm_by_group": dict(
            _first_mapping(payload, "update_norm_by_group")
            or _first_mapping(actuator, "update_norm_by_group")
        ),
    }


def _parseback_authority_ok(
    payload: Mapping[str, Any],
    admission: PairLocalServoAdmission,
) -> bool:
    authority = str(payload.get("authority") or "").strip()
    return bool(
        authority in PR95_SERVO_PROMOTABLE_AUTHORITIES
        and admission.surfaces.get("parseback_survival") is True
    )


def _action_algebra_ok(payload: Mapping[str, Any]) -> bool:
    action = _first_mapping(payload, "action_algebra_trace", "transform_action")
    action_id = str(action.get("selected_action_id") or action.get("action_id") or "")
    frame_scope = str(action.get("frame_scope") or payload.get("frame_scope") or "")
    noncomm = (
        action.get("noncommutative_interactions_checked") is True
        or action.get("composite_mode_interaction_checked") is True
    )
    return bool(
        action_id
        and frame_scope in {"frame0_pose_only", "frame1_seg_pose_joint", "both_frames_joint"}
        and _finite_float_from_mapping(action, "effect_delta_seg") is not None
        and _finite_float_from_mapping(action, "effect_delta_pose") is not None
        and _finite_float_from_mapping(action, "effect_delta_bytes") is not None
        and (
            _non_negative_float_from_mapping(action, "runtime_delta_ms") is not None
            or _non_negative_float_from_mapping(action, "runtime_seconds_delta") is not None
        )
        and (
            _non_negative_float_from_mapping(action, "action_payload_bits") is not None
            or _non_negative_float_from_mapping(action, "selector_bits") is not None
        )
        and noncomm
    )


def _action_algebra_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    action = _first_mapping(payload, "action_algebra_trace", "transform_action")
    return {
        "selected_action_id": action.get("selected_action_id") or action.get("action_id"),
        "frame_scope": action.get("frame_scope") or payload.get("frame_scope"),
        "effect_delta_seg": _finite_float_from_mapping(action, "effect_delta_seg"),
        "effect_delta_pose": _finite_float_from_mapping(action, "effect_delta_pose"),
        "effect_delta_bytes": _finite_float_from_mapping(action, "effect_delta_bytes"),
        "runtime_delta_ms": _finite_float_from_mapping(action, "runtime_delta_ms"),
        "runtime_seconds_delta": _finite_float_from_mapping(
            action,
            "runtime_seconds_delta",
        ),
        "action_payload_bits": _finite_float_from_mapping(action, "action_payload_bits"),
        "selector_bits": _finite_float_from_mapping(action, "selector_bits"),
        "noncommutative_interactions_checked": action.get(
            "noncommutative_interactions_checked"
        ),
        "composite_mode_interaction_checked": action.get(
            "composite_mode_interaction_checked"
        ),
    }


def _value_per_byte_ok(
    payload: Mapping[str, Any],
    *,
    before: PairLocalScoreState,
    after: PairLocalScoreState,
) -> bool:
    delta_bytes = after.archive_bytes - before.archive_bytes
    if delta_bytes <= 0:
        return True
    value = _float_or_none(payload.get("value_per_byte"))
    return value is not None and value > (25.0 / CONTEST_REFERENCE_BYTES)


def _value_per_byte_evidence(
    payload: Mapping[str, Any],
    *,
    before: PairLocalScoreState,
    after: PairLocalScoreState,
) -> dict[str, Any]:
    delta_bytes = after.archive_bytes - before.archive_bytes
    nonrate_delta = 100.0 * (after.d_seg - before.d_seg) + (
        math.sqrt(10.0 * after.d_pose) - math.sqrt(10.0 * before.d_pose)
    )
    return {
        "delta_archive_bytes": delta_bytes,
        "delta_score_nonrate": nonrate_delta,
        "measured_value_per_byte": _measured_value_per_byte(
            payload,
            before=before,
            after=after,
        ),
        "byte_price": 25.0 / CONTEST_REFERENCE_BYTES,
        "reported_value_per_byte": _float_or_none(payload.get("value_per_byte")),
    }


def _measured_value_per_byte(
    payload: Mapping[str, Any],
    *,
    before: PairLocalScoreState,
    after: PairLocalScoreState,
) -> float | None:
    reported = _float_or_none(payload.get("value_per_byte"))
    if reported is not None:
        return reported
    delta_bytes = after.archive_bytes - before.archive_bytes
    if delta_bytes <= 0:
        return None
    return None


def _hardware_margin_ok(payload: Mapping[str, Any]) -> bool:
    margin = _first_mapping(payload, "hardware_margin_trace", "hardware_margin")
    authority = str(
        margin.get("target_authority")
        or payload.get("target_authority")
        or payload.get("authority")
        or ""
    )
    checked = (
        margin.get("cpu_cuda_margin_checked") is True
        or margin.get("target_authority_margin_checked") is True
    )
    drift = str(margin.get("hardware_drift_risk") or "bounded")
    seg_margin = _non_negative_float_from_mapping(
        margin,
        "segnet_margin_min",
        "segnet_margin_safety",
    )
    pose_slack = _non_negative_float_from_mapping(
        margin,
        "pose_margin_radius",
        "pose_error_slack",
    )
    return bool(
        authority in PR95_SERVO_AUTHORITY_ORDER
        or authority in {"cpu", "cuda_t4", "dual_cpu_cuda"}
    ) and checked and drift != "unbounded" and seg_margin is not None and pose_slack is not None


def _hardware_margin_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    margin = _first_mapping(payload, "hardware_margin_trace", "hardware_margin")
    return {
        "target_authority": margin.get("target_authority")
        or payload.get("target_authority")
        or payload.get("authority"),
        "cpu_cuda_margin_checked": margin.get("cpu_cuda_margin_checked"),
        "target_authority_margin_checked": margin.get(
            "target_authority_margin_checked"
        ),
        "hardware_drift_risk": margin.get("hardware_drift_risk"),
        "segnet_margin_min": _finite_float_from_mapping(margin, "segnet_margin_min"),
        "segnet_margin_safety": _finite_float_from_mapping(
            margin,
            "segnet_margin_safety",
        ),
        "pose_margin_radius": _finite_float_from_mapping(margin, "pose_margin_radius"),
        "pose_error_slack": _finite_float_from_mapping(margin, "pose_error_slack"),
    }


def _surface_flags(
    trace: PairLocalSurfaceTrace,
    *,
    min_margin_delta: float,
    min_pose_output_delta: float,
) -> dict[str, bool]:
    uint8_motion = _positive_int(trace.uint8_changed_pixels) or _positive(
        trace.uint8_delta_abs_max
    )
    seg_preprocess = _positive(trace.segnet_input_delta_linf)
    pose_preprocess = _positive(trace.posenet_input_delta_linf)
    seg_live = _positive_int(trace.segnet_argmax_flipped_pixels) or _above_abs(
        trace.segnet_margin_delta,
        min_margin_delta,
    )
    pose_live = _above_abs(trace.pose_output_delta_l2, min_pose_output_delta)
    fakequant_seg = _positive_int(trace.fakequant_argmax_flipped_pixels) or _above_abs(
        trace.fakequant_segnet_margin_delta,
        min_margin_delta,
    )
    fakequant_pose = _above_abs(
        trace.fakequant_pose_output_delta_l2,
        min_pose_output_delta,
    )
    parseback_seg = _positive_int(trace.parseback_argmax_flipped_pixels) or _above_abs(
        trace.parseback_segnet_margin_delta,
        min_margin_delta,
    )
    parseback_pose = _above_abs(
        trace.parseback_pose_output_delta_l2,
        min_pose_output_delta,
    )
    inflated_seg = _positive_int(trace.inflated_argmax_flipped_pixels)
    inflated_pose = _above_abs(trace.inflated_pose_output_delta_l2, min_pose_output_delta)
    if trace.frame_scope == "frame0_pose_only":
        scorer_preprocess = pose_preprocess
        live_scorer = pose_live
        fakequant_survival = fakequant_pose
        parseback_survival = parseback_pose
        inflate_survival = inflated_pose
    elif trace.frame_scope == "frame1_seg_pose_joint":
        scorer_preprocess = seg_preprocess
        live_scorer = seg_live
        fakequant_survival = fakequant_seg
        parseback_survival = parseback_seg
        inflate_survival = inflated_seg
    else:
        scorer_preprocess = seg_preprocess or pose_preprocess
        live_scorer = seg_live or pose_live
        fakequant_survival = fakequant_seg or fakequant_pose
        parseback_survival = parseback_seg or parseback_pose
        inflate_survival = inflated_seg or inflated_pose
    return {
        "uint8_motion": uint8_motion,
        "seg_preprocess_movement": seg_preprocess,
        "pose_preprocess_movement": pose_preprocess,
        "scorer_preprocess_motion": scorer_preprocess,
        "seg_movement": seg_live,
        "pose_movement": pose_live,
        "live_scorer_motion": live_scorer,
        "fakequant_survival": fakequant_survival,
        "parseback_survival": parseback_survival,
        "inflate_survival": inflate_survival,
    }


def _debt_from_mapping(payload: Mapping[str, Any]) -> ScorerDebtTarget:
    score_units = _first_float(payload, "score_units", "score_debt", "debt_score_units")
    if score_units is None:
        raise ValueError("debt mapping missing score_units")
    axis = str(payload.get("axis") or "joint")
    if axis not in {"seg", "pose", "joint"}:
        axis = "joint"
    frame_scope = str(payload.get("frame_scope") or "both_frames_joint")
    if frame_scope not in {
        "frame0_pose_only",
        "frame1_seg_pose_joint",
        "both_frames_joint",
    }:
        frame_scope = "both_frames_joint"
    return ScorerDebtTarget(
        target_id=str(payload.get("target_id") or payload.get("id") or "unknown"),
        score_units=score_units,
        axis=axis,  # type: ignore[arg-type]
        pair_index=_int_or_none(payload.get("pair_index")),
        frame_scope=frame_scope,  # type: ignore[arg-type]
        raw=payload,
    )


def _first_float(payload: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        out = _float_or_none(payload.get(key))
        if out is not None:
            return out
    return None


def _first_mapping(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _first_int(payload: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        out = _int_or_none(payload.get(key))
        if out is not None:
            return out
    return None


def _required_float(payload: Mapping[str, Any], key: str) -> float:
    out = _float_or_none(payload.get(key))
    if out is None:
        raise ValueError(f"servo receipt missing finite {key}")
    return out


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    out = _int_or_none(payload.get(key))
    if out is None:
        raise ValueError(f"servo receipt missing integer {key}")
    return out


def _required_first_float(payload: Mapping[str, Any], *keys: str) -> float:
    out = _first_float(payload, *keys)
    if out is None:
        joined = ", ".join(keys)
        raise ValueError(f"servo receipt missing one finite value from: {joined}")
    return out


def _required_first_int(payload: Mapping[str, Any], *keys: str) -> int:
    out = _first_int(payload, *keys)
    if out is None:
        joined = ", ".join(keys)
        raise ValueError(f"servo receipt missing one integer value from: {joined}")
    return out


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out


def _positive(value: float | None) -> bool:
    return value is not None and value > 0.0


def _positive_int(value: int | None) -> bool:
    return value is not None and value > 0


def _above_abs(value: float | None, floor: float) -> bool:
    return value is not None and abs(value) > floor


def _string_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _positive_float_in_mapping(payload: Mapping[str, Any], key: str) -> bool:
    value = _float_or_none(payload.get(key))
    if value is not None:
        return value > 0.0
    for item_key, item_value in payload.items():
        if str(item_key).startswith(f"{key}.") or str(item_key).startswith(f"{key}/"):
            parsed = _float_or_none(item_value)
            if parsed is not None and parsed > 0.0:
                return True
    return False


def _finite_float_from_mapping(payload: Mapping[str, Any], *keys: str) -> float | None:
    return _first_float(payload, *keys)


def _non_negative_float_from_mapping(
    payload: Mapping[str, Any],
    *keys: str,
) -> float | None:
    value = _first_float(payload, *keys)
    return value if value is not None and value >= 0.0 else None


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


__all__ = [
    "PAIR_LOCAL_DISTORTION_SERVO_ADMISSION_SCHEMA",
    "PAIR_LOCAL_DISTORTION_SERVO_RECEIPT_SCHEMA",
    "PAIR_LOCAL_DISTORTION_SERVO_REPORT_SCHEMA",
    "PAIR_LOCAL_DISTORTION_SERVO_STATIC_CONTRACT_SCHEMA",
    "PR95_SERVO_AUTHORITY_ORDER",
    "PR95_SERVO_CURRICULUM_STAGES",
    "PR95_SERVO_PROMOTABLE_AUTHORITIES",
    "PairLocalScoreState",
    "PairLocalServoAdmission",
    "PairLocalSurfaceTrace",
    "ScorerDebtTarget",
    "admit_pair_local_distortion_action",
    "build_pr95_grade_pair_local_servo_report",
    "byte_cost_score_units",
    "exact_pair_local_score_delta",
    "pair_local_servo_receipt_ready",
    "pair_local_servo_static_contract",
    "seg_argmax_pixel_debt_score_units",
    "select_worst_scorer_debt_target",
]
