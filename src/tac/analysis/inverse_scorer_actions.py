# SPDX-License-Identifier: MIT
"""Thin inverse-scorer ActionEffect generation.

This module does not invert evaluate.py by fiat and does not fabricate scorer
motion.  It consumes measured ActionEffect rows that already carry receiver
surface movement (for example HiNeRV four-arm hard-birth rows or PR110 replay
rows) and re-emits the subset that forms the inverse-evaluator action basis:

* frame0 PoseNet-only actions;
* frame1 SegNet target-region margin actions;
* frame1 birth + frame0 pose-compensation composites.

The output remains ``tac.action_effect.v1`` and non-promotional.  Missing scorer
atoms stay missing; callers get blockers instead of synthetic candidate rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.analysis.action_effect import ActionEffect
from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS

INVERSE_SCORER_GENERATION_SCHEMA = "tac.inverse_scorer_action_generation.v1"
INVERSE_SCORER_QUEUE_ROW_SCHEMA = "tac.inverse_scorer_candidate_queue_row.v1"
SCORE_PROGRAM_WORD_SCHEMA = "tac.score_program_word.v1"
SCORE_PROGRAM_OPERATION_SCHEMA = "tac.score_program_operation.v1"
SCORE_PROGRAM_INTERPRETER = "inflate_action_word_v1"
DIRECT_SEG_WALL_ORACLE_SCHEMA = "tac.direct_seg_wall_oracle_receipt.v1"
TARGET_REGION_WALL_NORMAL_LIFT_SCHEMA = "tac.target_region_wall_normal_lift.v1"

BLOCKER_NO_FRAME0_POSE = "inverse_scorer_frame0_pose_action_missing"
BLOCKER_NO_FRAME1_SEG = "inverse_scorer_frame1_seg_margin_action_missing"
BLOCKER_NO_COMPOSITE = "inverse_scorer_composite_action_missing"
BLOCKER_RECEIVER_SURFACE_MISSING = "inverse_scorer_receiver_surface_motion_missing"
BLOCKER_SCORE_DELTA_MISSING = "inverse_scorer_exact_delta_missing"
BLOCKER_SCORE_PROGRAM_ARCHIVE_HASH_MISSING = "score_program_archive_hash_missing"
BLOCKER_SCORE_PROGRAM_PARSEBACK_MISSING = "score_program_parseback_survival_missing"
BLOCKER_SCORE_PROGRAM_INFLATE_MISSING = "score_program_inflate_survival_missing"
BLOCKER_REGION_SUPPORT_IDENTITY_MISSING = "inverse_scorer_region_support_identity_missing"
BLOCKER_REGION_SUPPORT_RESEARCH_ONLY = "inverse_scorer_region_support_research_only"
BLOCKER_ARCHIVE_CLOSED_BIRTH_REQUIRES_EXECUTABLE_SUPPORT = (
    "archive_closed_birth_requires_executable_support"
)
BLOCKER_DIRECT_SEG_WALL_EMPTY_SUPPORT = "direct_seg_wall_oracle_empty_support"
BLOCKER_DIRECT_SEG_WALL_NO_UINT8_MOTION = "direct_seg_wall_oracle_no_uint8_motion"
BLOCKER_UINT8_MOTION_WITHOUT_SEGNET_WALL_CROSSING = (
    "uint8_motion_without_segnet_wall_crossing"
)
BLOCKER_DIRECT_SEG_WALL_EXACT_SCORE_NOT_IMPROVED = (
    "direct_seg_wall_oracle_exact_score_not_improved"
)
BLOCKER_WALL_NORMAL_DIRECT_TEACHER_MISSING = "target_region_wall_normal_direct_teacher_missing"
BLOCKER_WALL_NORMAL_DIRECT_TEACHER_NOT_CROSSED = "target_region_wall_normal_direct_teacher_not_crossed"
BLOCKER_WALL_NORMAL_DIRECT_TEACHER_EXACT_SCORE_NOT_ACCEPTED = (
    "target_region_wall_normal_direct_teacher_exact_score_not_accepted"
)
BLOCKER_WALL_NORMAL_DIRECT_TEACHER_NOT_TRUE_WALL_NORMAL = (
    "target_region_wall_normal_direct_teacher_not_true_wall_normal"
)
BLOCKER_WALL_NORMAL_BACKEND_FIT_MISSING = "target_region_wall_normal_backend_fit_missing"
BLOCKER_WALL_NORMAL_BACKEND_NOT_REALIZED = "target_region_wall_normal_backend_not_realized"
BLOCKER_WALL_NORMAL_BACKEND_EXACT_SCORE_NOT_ACCEPTED = (
    "target_region_wall_normal_backend_exact_score_not_accepted"
)
BLOCKER_WALL_NORMAL_BACKEND_ACTION_EFFECT_INVALID = (
    "target_region_wall_normal_backend_action_effect_invalid"
)
BLOCKER_WALL_NORMAL_SIDECAR_ARCHIVE_UNCLOSED = "target_region_wall_normal_sidecar_archive_unclosed"

TRUE_SEG_WALL_NORMAL_INVERSE_SOURCES = frozenset(
    {
        "segnet_margin_gradient",
        "segnet_margin_vjp",
        "segnet_target_margin_vjp",
        "target_margin_vjp",
        "support_projected_segnet_margin_vjp",
    }
)

_PROMOTION_ONLY_BLOCKERS = frozenset(
    {
        BLOCKER_REGION_SUPPORT_IDENTITY_MISSING,
        BLOCKER_REGION_SUPPORT_RESEARCH_ONLY,
        BLOCKER_ARCHIVE_CLOSED_BIRTH_REQUIRES_EXECUTABLE_SUPPORT,
    }
)


@dataclass(frozen=True)
class InverseCandidate:
    """A measured ActionEffect row plus planner queue metadata."""

    effect: ActionEffect
    menu_cluster_hint: str
    dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()


def build_direct_seg_wall_oracle_receipt(
    *,
    action_id: str,
    authority: str,
    pair_id: int,
    target_class: int,
    support_mask: Any,
    before_argmax: Any,
    after_argmax: Any,
    old_d_seg: float,
    new_d_seg: float,
    old_d_pose: float | None = None,
    new_d_pose: float | None = None,
    before_uint8: Any | None = None,
    after_uint8: Any | None = None,
    region_id: str | None = None,
    support_source: str = "direct_seg_wall_oracle_support_mask",
    support_encoding: str = "bool_packbits_not_archive_priced",
    support_encoded_bytes: int | None = None,
    archive_executable_support_sha256: str | None = None,
    archive_executable_support_encoding: str | None = None,
    archive_executable_support_cardinality: int | None = None,
    archive_executable_support_encoded_bytes: int | None = None,
    inverse_source: str = "segnet_margin_vjp",
    inverse_basis: str | None = None,
    uses_official_seg_preprocess: bool | None = None,
    uses_target_class_margin: bool | None = None,
    margin_convention: str | None = None,
    frontier_pixel_policy: str | None = None,
    producer: str = "direct_seg_wall_oracle",
    consumer: str | None = "inverse_evaluate_candidate_queue",
) -> dict[str, Any]:
    """Build a measured teacher receipt for direct SegNet wall crossing.

    This helper does not run SegNet and does not fabricate an optimizer result.
    It consumes caller-supplied pre/post argmax and optional uint8 surfaces, then
    emits the exact failure classification needed by HiNeRV/SNeRV fitting code:
    did receiver-visible motion cross the target argmax wall, or did it merely
    move pixels inside the wrong chamber?
    """

    support = np.asarray(support_mask, dtype=bool)
    before = np.asarray(before_argmax)
    after = np.asarray(after_argmax)
    if support.shape != before.shape or support.shape != after.shape:
        raise ValueError(
            "support_mask, before_argmax, and after_argmax must share shape; "
            f"got support={support.shape} before={before.shape} after={after.shape}"
        )
    support_cardinality = int(np.count_nonzero(support))
    before_target = before == int(target_class)
    after_target = after == int(target_class)
    wrong_to_target = int(np.count_nonzero(support & (~before_target) & after_target))
    target_to_wrong = int(np.count_nonzero(support & before_target & (~after_target)))
    wrong_to_wrong = int(
        np.count_nonzero(
            support
            & (~before_target)
            & (~after_target)
            & (np.asarray(before) != np.asarray(after))
        )
    )
    argmax_changed = int(np.count_nonzero(support & (np.asarray(before) != np.asarray(after))))
    uint8_changed: int | None = None
    uint8_delta_linf: float | None = None
    if before_uint8 is not None and after_uint8 is not None:
        before_u8 = np.asarray(before_uint8)
        after_u8 = np.asarray(after_uint8)
        if before_u8.shape != after_u8.shape:
            raise ValueError(
                f"before_uint8/after_uint8 shape mismatch: {before_u8.shape} != {after_u8.shape}"
            )
        if before_u8.ndim == support.ndim + 1:
            delta = np.abs(after_u8.astype(np.int16) - before_u8.astype(np.int16))
            moved = np.any(delta > 0, axis=-1)
            uint8_delta_linf = float(np.max(delta[support])) if np.any(support) else 0.0
        elif before_u8.ndim == support.ndim:
            delta = np.abs(after_u8.astype(np.int16) - before_u8.astype(np.int16))
            moved = delta > 0
            uint8_delta_linf = float(np.max(delta[support])) if np.any(support) else 0.0
        else:
            raise ValueError(
                "uint8 surfaces must be support-shaped or support+channel-shaped; "
                f"got uint8={before_u8.shape} support={support.shape}"
            )
        uint8_changed = int(np.count_nonzero(moved & support))

    normalized_inverse_source = str(inverse_source or "unknown").strip() or "unknown"
    true_wall_normal_source = normalized_inverse_source in TRUE_SEG_WALL_NORMAL_INVERSE_SOURCES
    if uses_official_seg_preprocess is None:
        uses_official_seg_preprocess = true_wall_normal_source
    if uses_target_class_margin is None:
        uses_target_class_margin = true_wall_normal_source
    teacher_is_true_wall_normal = bool(
        true_wall_normal_source
        and uses_official_seg_preprocess
        and uses_target_class_margin
    )
    action_effect_inverse_source = (
        "segnet_margin_gradient" if teacher_is_true_wall_normal else "qrgb_basis"
    )

    blockers: list[str] = []
    if support_cardinality <= 0:
        blockers.append(BLOCKER_DIRECT_SEG_WALL_EMPTY_SUPPORT)
    if uint8_changed == 0:
        blockers.append(BLOCKER_DIRECT_SEG_WALL_NO_UINT8_MOTION)
    if (uint8_changed or 0) > 0 and wrong_to_target <= 0:
        blockers.append(BLOCKER_UINT8_MOTION_WITHOUT_SEGNET_WALL_CROSSING)

    support_hash = _support_mask_sha256(support)
    provisional = ActionEffect.build(
        action_id=str(action_id),
        family="direct_seg_wall_oracle",
        action_kind="direct_seg_wall_teacher",
        inverse_source=action_effect_inverse_source,
        frame_index=1,
        frame_incidence="seg_pose_joint",
        candidate_status="measured" if wrong_to_target > 0 else "rejected",
        authority=str(authority),
        producer=str(producer),
        consumer=consumer,
        pair_ids=[int(pair_id)],
        class_ids=[int(target_class)],
        region_ids=[] if region_id is None else [str(region_id)],
        payload_sections=["receiver_rgb_frame1_teacher_delta"],
        old_d_seg=float(old_d_seg),
        new_d_seg=float(new_d_seg),
        old_d_pose=old_d_pose,
        new_d_pose=new_d_pose,
        receiver_surface={
            "uint8_changed_pixels": uint8_changed,
            "seg_argmax_changed_pixels": argmax_changed,
            "seg_wrong_to_target_count": wrong_to_target,
            "seg_target_to_wrong_count": target_to_wrong,
            "seg_wrong_to_wrong_count": wrong_to_wrong,
            "seg_input_delta_linf": uint8_delta_linf,
        },
        exact_score_decision="not_applicable",
        hard_won_count=wrong_to_target,
        wrong_to_target=wrong_to_target,
        target_to_wrong=target_to_wrong,
        wrong_to_wrong=wrong_to_wrong,
        net_target_support_delta=wrong_to_target - target_to_wrong,
        uint8_changed_count_region=uint8_changed,
        seg_input_delta_linf_region=uint8_delta_linf,
        argmax_changed_count_region=argmax_changed,
        support_source=str(support_source),
        support_cardinality=support_cardinality,
        support_sha256=support_hash,
        support_encoding=str(support_encoding),
        support_encoded_bytes=support_encoded_bytes,
        support_research_only=not (
            support_encoded_bytes is not None and int(support_encoded_bytes) > 0
        ),
        blockers=blockers,
    )
    exact_blockers = list(blockers)
    if provisional.delta_score_nonrate is None or provisional.delta_score_nonrate >= 0.0:
        exact_blockers.append(BLOCKER_DIRECT_SEG_WALL_EXACT_SCORE_NOT_IMPROVED)
    effect_payload = provisional.as_dict()
    effect_payload["exact_score_decision"] = "accept" if not exact_blockers else "reject"
    effect_payload["blockers"] = exact_blockers
    effect = ActionEffect.from_dict(effect_payload)
    archive_support_hash = (
        str(archive_executable_support_sha256)
        if archive_executable_support_sha256 is not None
        else None
    )
    return {
        "schema": DIRECT_SEG_WALL_ORACLE_SCHEMA,
        "action_id": effect.action_id,
        "authority": effect.authority,
        "pair_id": int(pair_id),
        "target_class": int(target_class),
        "support_source": str(support_source),
        "inverse_source": normalized_inverse_source,
        "inverse_basis": (
            str(inverse_basis)
            if inverse_basis is not None
            else normalized_inverse_source
        ),
        "uses_official_seg_preprocess": bool(uses_official_seg_preprocess),
        "uses_target_class_margin": bool(uses_target_class_margin),
        "teacher_is_true_wall_normal": bool(teacher_is_true_wall_normal),
        "margin_convention": margin_convention,
        "frontier_pixel_policy": frontier_pixel_policy,
        "support_cardinality": int(support_cardinality),
        "support_sha256": support_hash,
        "support_hash_domain": "bool_mask_bhw",
        "support_encoding": str(support_encoding),
        "support_encoded_bytes": support_encoded_bytes,
        "archive_executable_support_sha256": archive_support_hash,
        "archive_executable_support_hash_domain": (
            "target_region_action_coordinates_v1" if archive_support_hash else None
        ),
        "archive_executable_support_encoding": (
            None
            if archive_executable_support_encoding is None
            else str(archive_executable_support_encoding)
        ),
        "archive_executable_support_cardinality": archive_executable_support_cardinality,
        "archive_executable_support_encoded_bytes": archive_executable_support_encoded_bytes,
        "archive_executable": bool(
            support_encoded_bytes is not None and int(support_encoded_bytes) > 0
        ),
        "wrong_to_target_count": int(wrong_to_target),
        "target_to_wrong_count": int(target_to_wrong),
        "wrong_to_wrong_count": int(wrong_to_wrong),
        "argmax_changed_count": int(argmax_changed),
        "uint8_changed_pixels": uint8_changed,
        "uint8_delta_linf": uint8_delta_linf,
        "crossed_target_wall": bool(wrong_to_target > 0),
        "blockers": list(effect.blockers),
        "action_effect": effect.as_dict(),
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def build_target_region_wall_normal_lift_receipt(
    *,
    action_id: str,
    pair_id: int,
    target_class: int,
    region_id: str,
    direct_teacher_candidate: Mapping[str, Any] | None,
    backend_birth_receipt: Mapping[str, Any] | None,
    sidecar_candidate: Mapping[str, Any] | None = None,
    authority: str = "batch_local_live_mlx",
) -> dict[str, Any]:
    """Summarize the PR95-grade wall-normal servo DAG for one region.

    The receiver-quantum line search proves that a backend can touch the uint8
    lattice.  This receipt answers the next question without promoting a proxy:
    did a direct scorer-space teacher cross the target SegNet wall, did the
    HiNeRV backend realize that same wall crossing under exact Seg/Pose score,
    and, if not, is the measured byte-priced sidecar/action atom the correct
    fallback to close next?
    """

    direct = dict(direct_teacher_candidate or {})
    backend = dict(backend_birth_receipt or {})
    sidecar = dict(sidecar_candidate or direct)
    direct_wall = _mapping(direct.get("direct_seg_wall_oracle"))
    direct_effect_payload = _mapping(direct_wall.get("action_effect"))
    direct_effect: ActionEffect | None = None
    if direct_effect_payload:
        try:
            direct_effect = ActionEffect.from_dict(direct_effect_payload)
        except Exception:
            direct_effect = None

    backend_effect: ActionEffect | None = None
    if backend:
        try:
            backend_effect = ActionEffect.from_hinerv_birth_receipt(
                backend,
                consumer="target_region_wall_normal_lift",
            )
        except Exception:
            backend_effect = None

    direct_wrong_to_target = _first_int(
        direct_wall,
        "wrong_to_target_count",
        default=_first_int(direct, "wrong_to_target_count", default=0),
    )
    direct_target_to_wrong = _first_int(
        direct_wall,
        "target_to_wrong_count",
        default=_first_int(direct, "target_to_wrong_count", default=0),
    )
    direct_delta_nonrate = _first_float(
        direct,
        "exact_delta_score_nonrate",
        default=(
            direct_effect.delta_score_nonrate
            if direct_effect is not None
            else None
        ),
    )
    direct_crossed = bool(direct_wrong_to_target > 0)
    direct_inverse_source = str(
        direct_wall.get("inverse_source")
        or direct.get("inverse_source")
        or direct.get("oracle_kind")
        or (
            direct_effect.inverse_source
            if direct_effect is not None
            else ""
        )
        or "unknown"
    )
    direct_uses_official_seg_preprocess = bool(
        direct_wall.get("uses_official_seg_preprocess") is True
    )
    direct_uses_target_class_margin = bool(
        direct_wall.get("uses_target_class_margin") is True
    )
    direct_teacher_is_true_wall_normal = bool(
        direct_wall.get("teacher_is_true_wall_normal") is True
        and direct_inverse_source in TRUE_SEG_WALL_NORMAL_INVERSE_SOURCES
        and direct_uses_official_seg_preprocess
        and direct_uses_target_class_margin
    )
    direct_exact_accept = bool(
        direct_wall.get("crossed_target_wall") is True
        and not _contains_any(
            direct_wall.get("blockers"),
            (
                BLOCKER_DIRECT_SEG_WALL_EXACT_SCORE_NOT_IMPROVED,
                BLOCKER_UINT8_MOTION_WITHOUT_SEGNET_WALL_CROSSING,
                BLOCKER_DIRECT_SEG_WALL_NO_UINT8_MOTION,
            ),
        )
    )
    direct_usable_wall_normal = bool(
        direct_crossed and direct_exact_accept and direct_teacher_is_true_wall_normal
    )

    backend_transitions = _mapping(backend.get("argmax_transitions"))
    backend_wrong_to_target = _first_int(
        backend_transitions,
        "wrong_to_target_count",
        default=_first_int(backend, "wrong_to_target_count", default=0),
    )
    backend_target_to_wrong = _first_int(
        backend_transitions,
        "target_to_wrong_count",
        default=_first_int(backend, "target_to_wrong_count", default=0),
    )
    backend_delta_nonrate = (
        backend_effect.delta_score_nonrate
        if backend_effect is not None
        else _first_float(_mapping(backend.get("exact_nonrate")), "delta_score_nonrate")
    )
    backend_exact_decision = (
        backend_effect.exact_score_decision
        if backend_effect is not None
        else str(_mapping(backend.get("exact_nonrate")).get("exact_score_decision") or "")
    )
    backend_accepted = bool(
        _first_int(backend, "accepted_step_count", default=0) > 0
        or backend.get("accepted") is True
    )
    backend_realized_wall = bool(
        backend_accepted
        and backend_wrong_to_target > 0
        and (backend_target_to_wrong <= backend_wrong_to_target)
    )
    backend_exact_accept = bool(backend_exact_decision in {"accept", "accepted"})
    sidecar_payload_bytes = _first_int(
        sidecar,
        "target_region_action_payload_bytes",
        default=0,
    )
    sidecar_exact_delta = _first_float(sidecar, "exact_delta_score_nonrate")
    sidecar_available = bool(sidecar_payload_bytes > 0 and direct_usable_wall_normal)
    sidecar_archive_closed = bool(sidecar.get("archive_closed") is True)

    blockers: list[str] = []
    if not direct:
        blockers.append(BLOCKER_WALL_NORMAL_DIRECT_TEACHER_MISSING)
    if direct and not direct_crossed:
        blockers.append(BLOCKER_WALL_NORMAL_DIRECT_TEACHER_NOT_CROSSED)
    if direct and not direct_teacher_is_true_wall_normal:
        blockers.append(BLOCKER_WALL_NORMAL_DIRECT_TEACHER_NOT_TRUE_WALL_NORMAL)
    if direct_crossed and not direct_exact_accept:
        blockers.append(BLOCKER_WALL_NORMAL_DIRECT_TEACHER_EXACT_SCORE_NOT_ACCEPTED)
    if not backend:
        blockers.append(BLOCKER_WALL_NORMAL_BACKEND_FIT_MISSING)
    if backend and backend_effect is None:
        blockers.append(BLOCKER_WALL_NORMAL_BACKEND_ACTION_EFFECT_INVALID)
    if direct_crossed and backend and not backend_realized_wall:
        blockers.append(BLOCKER_WALL_NORMAL_BACKEND_NOT_REALIZED)
    if backend_realized_wall and not backend_exact_accept:
        blockers.append(BLOCKER_WALL_NORMAL_BACKEND_EXACT_SCORE_NOT_ACCEPTED)
    if sidecar_available and not backend_realized_wall:
        blockers.append(BLOCKER_WALL_NORMAL_SIDECAR_ARCHIVE_UNCLOSED)
    blockers.extend(str(item) for item in direct_wall.get("blockers") or ())
    blockers.extend(str(item) for item in backend.get("blockers") or ())

    if direct_usable_wall_normal and backend_realized_wall and backend_exact_accept:
        selected = "backend_fit_live"
        next_surface = "fakequant_archive_parseback_survival"
    elif direct_usable_wall_normal and sidecar_available:
        selected = "byte_priced_action_fallback"
        next_surface = "archive_materialize_parseback_inflate"
    elif direct_usable_wall_normal:
        selected = "backend_actuator_basis_gap"
        next_surface = "progressive_backend_actuator_ladder"
    else:
        selected = "direct_wall_teacher_gap"
        next_surface = "inverse_scorer_candidate_generation"

    if not direct:
        decision_state = "DIRECT_TEACHER_NO_WALL_CROSS"
        first_failing_surface = "direct_teacher"
    elif not direct_teacher_is_true_wall_normal:
        decision_state = "DIRECT_TEACHER_NO_WALL_CROSS"
        first_failing_surface = "direct_teacher_basis"
    elif not direct_crossed:
        decision_state = "DIRECT_TEACHER_NO_WALL_CROSS"
        first_failing_surface = "segnet_argmax_margin"
    elif not direct_exact_accept:
        decision_state = "DIRECT_TEACHER_EXACT_REJECTED"
        first_failing_surface = "exact_nonlinear_score"
    elif backend_realized_wall and backend_exact_accept:
        decision_state = "BACKEND_REALIZATION_ACCEPTED"
        first_failing_surface = None
    elif backend and not backend_realized_wall:
        decision_state = "BACKEND_REALIZATION_FAILED"
        first_failing_surface = "backend_realization"
    elif sidecar_available:
        decision_state = "SUPPORT_NOT_ARCHIVE_EXECUTABLE"
        first_failing_surface = "archive_materialize_parseback_inflate"
    else:
        decision_state = "BACKEND_REALIZATION_FAILED"
        first_failing_surface = "backend_fit"

    if not direct or not direct_teacher_is_true_wall_normal or not direct_crossed:
        direct_decision_state = "DIRECT_TEACHER_NO_WALL_CROSS"
    elif not direct_exact_accept:
        direct_decision_state = "DIRECT_TEACHER_EXACT_REJECTED"
    else:
        direct_decision_state = "DIRECT_TEACHER_ACCEPTED"

    if not direct_usable_wall_normal:
        backend_decision_state = "SKIPPED_DIRECT_TEACHER_FAILED"
    elif backend_realized_wall and backend_exact_accept:
        backend_decision_state = "BACKEND_REALIZATION_ACCEPTED"
    else:
        backend_decision_state = "BACKEND_REALIZATION_FAILED"

    if not direct_usable_wall_normal:
        sidecar_decision_state = "SKIPPED_DIRECT_TEACHER_FAILED"
    elif sidecar_payload_bytes <= 0:
        sidecar_decision_state = "SIDECAR_FALLBACK_MISSING"
    elif sidecar_archive_closed:
        sidecar_decision_state = "SIDECAR_FALLBACK_ACCEPTED"
    else:
        sidecar_decision_state = "SUPPORT_NOT_ARCHIVE_EXECUTABLE"

    direct_summary = {
        "available": bool(direct),
        "decision_state": direct_decision_state,
        "source": str(direct.get("oracle_kind") or "unknown"),
        "inverse_source": direct_inverse_source,
        "inverse_basis": direct_wall.get("inverse_basis"),
        "uses_official_seg_preprocess": bool(direct_uses_official_seg_preprocess),
        "uses_target_class_margin": bool(direct_uses_target_class_margin),
        "margin_convention": direct_wall.get("margin_convention"),
        "frontier_pixel_policy": direct_wall.get("frontier_pixel_policy"),
        "teacher_is_true_wall_normal": bool(direct_teacher_is_true_wall_normal),
        "crossed_target_wall": bool(direct_crossed),
        "qualified_crossed_target_wall": bool(
            direct_crossed and direct_teacher_is_true_wall_normal
        ),
        "exact_score_decision": "accept" if direct_exact_accept else "reject",
        "wrong_to_target_count": int(direct_wrong_to_target),
        "target_to_wrong_count": int(direct_target_to_wrong),
        "exact_delta_score_nonrate": direct_delta_nonrate,
        "support_source": direct_wall.get("support_source"),
        "support_sha256": direct_wall.get("support_sha256"),
        "support_cardinality": direct_wall.get("support_cardinality"),
        "support_encoding": direct_wall.get("support_encoding"),
        "support_encoded_bytes": direct_wall.get("support_encoded_bytes"),
        "support_research_only": direct_wall.get("archive_executable") is not True,
        "action_effect": (
            direct_effect.as_dict()
            if direct_effect is not None
            else direct_effect_payload
        ),
    }
    backend_summary = {
        "attempted": bool(backend),
        "decision_state": backend_decision_state,
        "realized_target_wall": bool(backend_realized_wall),
        "accepted_step_count": _first_int(backend, "accepted_step_count", default=0),
        "wrong_to_target_count": int(backend_wrong_to_target),
        "target_to_wrong_count": int(backend_target_to_wrong),
        "realization_gap_wrong_to_target_count": int(
            max(0, direct_wrong_to_target - backend_wrong_to_target)
        ),
        "realization_gap_delta_score_nonrate": (
            None
            if direct_delta_nonrate is None or backend_delta_nonrate is None
            else float(backend_delta_nonrate - direct_delta_nonrate)
        ),
        "exact_score_decision": backend_exact_decision or "not_applicable",
        "exact_delta_score_nonrate": backend_delta_nonrate,
        "trained_groups": list(backend.get("trained_groups") or ()),
        "updated_parameter_names": list(backend.get("updated_parameter_names") or ()),
        "action_effect": (
            backend_effect.as_dict()
            if backend_effect is not None
            else None
        ),
    }
    sidecar_summary = {
        "available": bool(sidecar_available),
        "decision_state": sidecar_decision_state,
        "compiled": False,
        "payload_bytes": int(sidecar_payload_bytes),
        "exact_delta_score_nonrate": sidecar_exact_delta,
        "value_per_payload_byte_nonrate": _first_float(
            sidecar,
            "target_region_action_value_per_payload_byte_nonrate",
        ),
        "archive_closed": sidecar_archive_closed,
        "archive_executable": sidecar_archive_closed,
        "support_sha256": _mapping(sidecar.get("target_region_action_section_telemetry")).get(
            "support_sha256"
        ),
        "blockers": list(sidecar.get("charged_byte_sections_missing") or ()),
    }

    return {
        "schema": TARGET_REGION_WALL_NORMAL_LIFT_SCHEMA,
        "operator": "TargetRegionWallNormalLift",
        "action_id": str(action_id),
        "authority": str(authority),
        "pair_id": int(pair_id),
        "target_class": int(target_class),
        "region_id": str(region_id),
        "stage_order": [
            "ReceiverQuantumLineSearch",
            "SegNetWallNormalLift",
            "PoseYUV6TrustProjection",
            "BackendRealization",
            "BytePricedActionFallback",
            "ExactReplayAdmission",
        ],
        "direct_teacher": direct_summary,
        "backend_fit": backend_summary,
        "sidecar_fallback": sidecar_summary,
        "decision_state": decision_state,
        "first_failing_surface": first_failing_surface,
        "selected_next_operator": selected,
        "next_required_surface": next_surface,
        "backend_realization_required_before_long_run": bool(
            selected != "backend_fit_live"
        ),
        "parseback_required_before_promotion": True,
        "promotion_eligible": False,
        "score_claim": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": _dedupe(blockers),
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def build_masked_residual_oracle_action_effect(
    candidate: Mapping[str, Any],
    *,
    action_id: str,
    pair_id: int,
    target_class: int,
    region_id: str,
    authority: str = "batch_local_live_mlx",
    producer: str = "hinerv_target_region_masked_residual_oracle",
    consumer: str | None = "inverse_evaluate_candidate_queue",
) -> ActionEffect:
    """Convert a measured masked-residual/scorer-pixel branch into ActionEffect.

    This is intentionally honest about basis: source-RGB residual copy and
    scorer-causal pixel synthesis are receiver-side action branches, not
    SegNet-margin VJP teachers.  They may be valuable sidecar/fallback actions,
    but they do not clear the true wall-normal backend gate by name.
    """

    if not isinstance(candidate, Mapping):
        raise TypeError("candidate must be a mapping")
    admission = _mapping(candidate.get("admission_decision"))
    transitions = _mapping(candidate.get("region_argmax_transitions"))
    support = _mapping(candidate.get("target_region_action_section_telemetry"))
    inverse_source = str(
        candidate.get("inverse_source")
        or candidate.get("oracle_kind")
        or "masked_residual"
    )
    blockers = [
        str(item)
        for item in (
            list(candidate.get("blockers") or ())
            + list(candidate.get("charged_byte_sections_missing") or ())
        )
        if str(item).strip()
    ]
    if not bool(candidate.get("archive_closed") is True):
        blockers.append(BLOCKER_WALL_NORMAL_SIDECAR_ARCHIVE_UNCLOSED)
    return ActionEffect.build(
        action_id=str(action_id),
        family="hinerv",
        action_kind="target_region_birth_sidecar_candidate",
        inverse_source=inverse_source,
        frame_index=1,
        frame_incidence="seg_pose_joint",
        candidate_status=(
            "measured"
            if _first_int(transitions, "wrong_to_target_count", default=0) > 0
            else "rejected"
        ),
        authority=str(authority),
        producer=str(producer),
        consumer=consumer,
        pair_ids=[int(pair_id)],
        class_ids=[int(target_class)],
        region_ids=[str(region_id)],
        payload_sections=["target_region_action_sidecar"],
        old_d_seg=_first_float(admission, "old_d_seg"),
        new_d_seg=_first_float(admission, "new_d_seg", default=_first_float(candidate, "d_seg_batch")),
        old_d_pose=_first_float(admission, "old_d_pose"),
        new_d_pose=_first_float(admission, "new_d_pose", default=_first_float(candidate, "d_pose_batch")),
        receiver_surface={
            "uint8_changed_pixels": _first_int(candidate, "receiver_uint8_changed_pixels_region"),
            "seg_input_delta_linf": _first_float(candidate, "seg_input_delta_linf_region", default=1.0),
            "posenet_input_delta_linf": _first_float(candidate, "posenet_input_delta_linf_pair", default=1.0),
            "seg_argmax_changed_pixels": _first_int(transitions, "argmax_changed_count_region"),
            "seg_wrong_to_target_count": _first_int(transitions, "wrong_to_target_count"),
            "seg_target_hard_lost_count": _first_int(transitions, "target_to_wrong_count"),
            "seg_wrong_to_wrong_count": _first_int(transitions, "wrong_to_wrong_count"),
            "pose_output_l2_delta": _first_float(candidate, "pose_output_delta_l2"),
        },
        exact_score_decision=_normal_exact_decision(
            admission.get("exact_score_decision")
            or ("accepted" if admission.get("accepted") is True else "rejected")
        ),
        raw_cap_decision=(
            None if admission.get("raw_cap_decision") is None else str(admission["raw_cap_decision"])
        ),
        catastrophic_guard_decision=(
            None
            if admission.get("catastrophic_guard_decision") is None
            else str(admission["catastrophic_guard_decision"])
        ),
        would_accept_exact_score_if_raw_cap_disabled=_bool_or_none(
            admission.get("would_accept_exact_score_if_raw_cap_disabled")
        ),
        would_accept_without_catastrophic_guard=_bool_or_none(
            admission.get("would_accept_without_catastrophic_guard")
        ),
        rejected_by_raw_cap=_bool_or_none(admission.get("rejected_by_raw_pose_cap")),
        rejected_by_exact_score=_bool_or_none(admission.get("rejected_by_exact_delta_score")),
        rejected_by_catastrophic_guard=_bool_or_none(
            admission.get("rejected_by_catastrophic_pose_guard")
        ),
        hard_won_count=_first_int(transitions, "target_hard_won_count", "wrong_to_target_count"),
        wrong_to_target=_first_int(transitions, "wrong_to_target_count"),
        target_to_wrong=_first_int(transitions, "target_to_wrong_count"),
        wrong_to_wrong=_first_int(transitions, "wrong_to_wrong_count"),
        net_target_support_delta=_first_int(transitions, "net_target_support_delta"),
        uint8_changed_count_region=_first_int(candidate, "receiver_uint8_changed_pixels_region"),
        seg_input_delta_linf_region=_first_float(candidate, "seg_input_delta_linf_region", default=1.0),
        posenet_input_delta_linf_pair=_first_float(candidate, "posenet_input_delta_linf_pair", default=1.0),
        argmax_changed_count_region=_first_int(transitions, "argmax_changed_count_region"),
        pose_output_l2_delta=_first_float(candidate, "pose_output_delta_l2"),
        seg_score_delta=_first_float(admission, "seg_score_delta"),
        pose_score_delta=_first_float(admission, "pose_score_delta"),
        segnet_margin_delta=_first_float(
            candidate,
            "segnet_margin_delta",
            "target_region_margin_delta",
            "worst_region_margin_p50_delta",
            "receiver_surface_worst_region_margin_p50_delta",
        ),
        fakequant_segnet_margin_delta=_first_float(
            candidate,
            "fakequant_segnet_margin_delta",
            "fakequant_worst_region_margin_p50_delta",
        ),
        parseback_segnet_margin_delta=_first_float(
            candidate,
            "parseback_segnet_margin_delta",
            "parseback_worst_region_margin_p50_delta",
        ),
        rejection_source=(
            None if admission.get("rejection_source") is None else str(admission["rejection_source"])
        ),
        support_source=(
            str(support.get("support_source"))
            if support.get("support_source") is not None
            else "masked_residual_oracle_support_missing"
        ),
        support_cardinality=_first_int(support, "support_cardinality"),
        support_sha256=(None if support.get("support_sha256") is None else str(support["support_sha256"])),
        support_encoding=(
            None if support.get("support_encoding") is None else str(support["support_encoding"])
        ),
        support_encoded_bytes=_first_int(support, "support_encoded_bytes"),
        support_research_only=not bool(support.get("archive_executable_support") is True),
        blockers=_dedupe(blockers),
    )


def generate_inverse_scorer_candidates(
    measured_effects: Sequence[ActionEffect],
    *,
    include_rejected: bool = True,
) -> dict[str, Any]:
    """Return inverse-scorer ActionEffect candidates from measured effects.

    ``measured_effects`` must be real ActionEffect rows.  The generator only
    annotates rows whose action kind and receiver evidence match one of the
    inverse-evaluator bases; it does not synthesize a candidate from absent
    gradients or missing receiver motion.
    """

    effects = _coerce_effects(measured_effects)
    candidates: list[InverseCandidate] = []
    blockers: list[str] = []

    frame0 = _first_matching(effects, _is_frame0_pose_action)
    frame1 = _first_matching(effects, _is_frame1_seg_action)
    composites = _matching(effects, _is_composite_action)
    wall_normal_branches = _matching(effects, _is_wall_normal_branch_action)

    frame0_candidate: InverseCandidate | None = None
    frame1_candidate: InverseCandidate | None = None
    if frame0 is None:
        blockers.append(BLOCKER_NO_FRAME0_POSE)
    else:
        frame0_candidate = _build_candidate(frame0, "frame0_pose", dependencies=(), conflicts=())
        candidates.append(frame0_candidate)

    if frame1 is None:
        blockers.append(BLOCKER_NO_FRAME1_SEG)
    else:
        frame1_candidate = _build_candidate(frame1, "frame1_seg", dependencies=(), conflicts=())
        candidates.append(frame1_candidate)

    if not composites:
        blockers.append(BLOCKER_NO_COMPOSITE)
    else:
        for composite in composites:
            deps, action_id_override = _composite_dependencies_and_id(
                composite,
                frame0_candidate=frame0_candidate,
                frame1_candidate=frame1_candidate,
            )
            candidates.append(
                _build_candidate(
                    composite,
                    "composite",
                    dependencies=deps,
                    conflicts=(),
                    action_id_override=action_id_override,
                )
            )
    for branch in wall_normal_branches:
        if any(candidate.effect.action_id == branch.action_id for candidate in candidates):
            continue
        candidates.append(
            InverseCandidate(
                effect=branch,
                menu_cluster_hint="frame1_seg_margin",
                dependencies=(),
                conflicts=(),
            )
        )

    if not include_rejected:
        candidates = [
            candidate
            for candidate in candidates
            if candidate.effect.exact_score_decision != "reject"
        ]

    candidate_effects = [candidate.effect for candidate in candidates]
    queue_rows = build_candidate_queue(candidate_effects, candidates)
    score_program_word = build_score_program_word(queue_rows)
    return {
        "schema": INVERSE_SCORER_GENERATION_SCHEMA,
        "input_effect_count": len(effects),
        "candidate_count": len(candidate_effects),
        "blockers": blockers,
        "passed": not blockers,
        "action_effects": candidate_effects,
        "candidate_queue": queue_rows,
        "score_program_word": score_program_word,
        "policy": {
            "measured_effects_only_no_synthetic_scorer_motion": True,
            "promotion_eligible_is_false_for_generated_rows": True,
            "frame0_pose_only_maps_to_pose_incidence": True,
            "frame1_maps_to_seg_pose_joint_incidence": True,
            "score_program_word_is_planning_only": True,
        },
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def build_candidate_queue(
    effects: Sequence[ActionEffect],
    candidates: Sequence[InverseCandidate] | None = None,
) -> list[dict[str, Any]]:
    """Build menu-ILP input rows without optimizing a menu."""

    candidate_list = list(candidates or ())
    by_id = {candidate.effect.action_id: candidate for candidate in candidate_list}
    rows: list[dict[str, Any]] = []
    for index, effect in enumerate(effects):
        candidate = (
            candidate_list[index]
            if index < len(candidate_list) and candidate_list[index].effect == effect
            else by_id.get(effect.action_id)
        )
        blockers = _candidate_blockers(effect)
        cluster = candidate.menu_cluster_hint if candidate is not None else _menu_cluster_hint(effect)
        score_program_operation = _score_program_operation_for_effect(
            effect,
            index=len(rows),
            menu_cluster_hint=cluster,
            row_blockers=blockers,
        )
        rows.append(
            {
                "schema": INVERSE_SCORER_QUEUE_ROW_SCHEMA,
                "action_id": effect.action_id,
                "family": effect.family,
                "authority": effect.authority,
                "normalization_scope": effect.normalization_scope,
                "pair_id": effect.pair_ids[0] if effect.pair_ids else None,
                "pair_ids": list(effect.pair_ids),
                "region_ids": list(effect.region_ids),
                "class_ids": list(effect.class_ids),
                "payload_sections": list(effect.payload_sections),
                "trained_groups": list(effect.trained_groups),
                "frame_index": effect.frame_index,
                "frame_incidence": effect.frame_incidence,
                "support_source": effect.support_source,
                "support_cardinality": effect.support_cardinality,
                "support_sha256": effect.support_sha256,
                "support_encoding": effect.support_encoding,
                "support_encoded_bytes": effect.support_encoded_bytes,
                "support_research_only": effect.support_research_only,
                "menu_cluster_hint": cluster,
                "score_program_opcode": score_program_operation["opcode"],
                "evaluator_action_basis": score_program_operation["basis"],
                "backend": score_program_operation["backend"],
                "receiver_visible": _receiver_visible(effect),
                "fakequant_survived": effect.fakequant_survived,
                "parseback_survived": effect.parseback_survived,
                "inflate_survived": effect.inflate_survived,
                "archive_sha256": effect.archive_sha256,
                "payload_sha256": effect.payload_sha256,
                "base_state_sha256": effect.base_state_sha256,
                "score_ev": _score_ev(effect),
                "byte_cost": _byte_cost(effect),
                "delta_score_nonrate": effect.delta_score_nonrate,
                "delta_score_total": effect.delta_score_total,
                "segnet_margin_delta": effect.segnet_margin_delta,
                "fakequant_segnet_margin_delta": effect.fakequant_segnet_margin_delta,
                "parseback_segnet_margin_delta": effect.parseback_segnet_margin_delta,
                "delta_bytes": effect.delta_bytes,
                "value_per_byte": effect.value_per_byte,
                "dependencies": list(candidate.dependencies if candidate is not None else ()),
                "conflicts": list(candidate.conflicts if candidate is not None else ()),
                "blockers": blockers,
                "promotion_blockers": _score_program_promotion_blockers(effect),
                "score_program_operation": score_program_operation,
                "menu_ilp_allowed": False,
                "menu_ilp_blockers": [
                    "menu_ilp_blocked_until_pr110_k16_baseline_reproduces",
                    "pr110_k16_baseline_reproduction_missing",
                ],
                "promotion_eligible": False,
                "score_claim": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    return rows


def build_score_program_word(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compile candidate rows into a deterministic evaluator-action word.

    The word is a planning artifact for the score-program compiler: it lowers
    measured ``ActionEffect`` rows into opcodes, basis labels, backend targets,
    exact score economics, and survival blockers.  It does not optimize,
    synthesize missing scorer motion, or promote an archive.
    """

    operations: list[dict[str, Any]] = []
    candidate_blockers: list[str] = []
    blockers_by_basis: dict[str, list[str]] = {}
    clean_basis: set[str] = set()
    promotion_blockers: list[str] = []
    total_score_ev = 0.0
    total_byte_cost = 0
    have_score_ev = False
    have_byte_cost = False
    executable_operation_count = 0
    for index, row in enumerate(rows):
        operation = _score_program_operation_from_row(row, index=index)
        operations.append(operation)
        runtime_blockers = _score_program_runtime_blockers(row, operation)
        candidate_blockers.extend(runtime_blockers)
        basis = str(operation.get("basis") or "unknown")
        blockers_by_basis.setdefault(basis, []).extend(runtime_blockers)
        if not runtime_blockers:
            clean_basis.add(basis)
            executable_operation_count += 1
        promotion_blockers.extend(str(value) for value in row.get("promotion_blockers") or [])
        promotion_blockers.extend(str(value) for value in operation.get("promotion_blockers") or [])
        score_ev = _finite_or_none(row.get("score_ev"))
        if score_ev is not None:
            have_score_ev = True
            total_score_ev += score_ev
        byte_cost = _int_or_none(row.get("byte_cost"))
        if byte_cost is not None:
            have_byte_cost = True
            total_byte_cost += abs(byte_cost)

    blockers: list[str] = []
    for basis, basis_blockers in blockers_by_basis.items():
        if basis not in clean_basis:
            blockers.extend(basis_blockers)
    word_blockers = _dedupe(blockers)
    all_candidate_blockers = _dedupe(candidate_blockers)
    word_promotion_blockers = _dedupe(promotion_blockers)
    return {
        "schema": SCORE_PROGRAM_WORD_SCHEMA,
        "interpreter": SCORE_PROGRAM_INTERPRETER,
        "operation_count": len(operations),
        "executable_operation_count": executable_operation_count,
        "operations": operations,
        "blocked": bool(word_blockers),
        "blockers": word_blockers,
        "candidate_blockers": all_candidate_blockers,
        "basis_with_clean_candidate": sorted(clean_basis),
        "promotion_blockers": word_promotion_blockers,
        "total_score_ev": total_score_ev if have_score_ev else None,
        "total_byte_cost": total_byte_cost if have_byte_cost else None,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "policy": {
            "measured_effects_only_no_synthetic_scorer_motion": True,
            "exact_delta_score_required_per_operation": True,
            "receiver_surface_motion_required_per_operation": True,
            "parseback_and_inflate_required_for_promotion": True,
            "menu_ilp_requires_pr110_k16_baseline": True,
        },
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def candidate_queue_jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    """Serialize queue rows to deterministic JSONL text."""

    return "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)


def _build_candidate(
    source: ActionEffect,
    candidate_kind: str,
    *,
    dependencies: Sequence[str],
    conflicts: Sequence[str],
    action_id_override: str | None = None,
) -> InverseCandidate:
    frame_index, frame_incidence, inverse_source, action_kind, cluster = _candidate_metadata(candidate_kind)
    payload = source.as_dict()
    payload.pop("old_archive_bytes", None)
    payload.pop("new_archive_bytes", None)
    payload.pop("restore_state_passed", None)
    support_identity = _support_identity_for_inverse_candidate(source, candidate_kind)
    payload.update(
        {
            "action_id": action_id_override or f"{source.action_id}__inverse_{candidate_kind}",
            "action_kind": action_kind,
            "inverse_source": inverse_source,
            "frame_index": frame_index,
            "frame_incidence": frame_incidence,
            "candidate_status": "measured",
            "producer": "inverse_scorer_actions",
            "consumer": "inverse_evaluate_candidate_queue",
            "promotion_eligible": False,
            **support_identity,
        }
    )
    effect = ActionEffect.from_dict(payload)
    return InverseCandidate(
        effect=effect,
        menu_cluster_hint=cluster,
        dependencies=tuple(str(item) for item in dependencies if item),
        conflicts=tuple(str(item) for item in conflicts if item),
    )


def _support_identity_for_inverse_candidate(
    source: ActionEffect,
    candidate_kind: str,
) -> dict[str, Any]:
    """Return explicit support identity or a research-only marker.

    Older measured HiNeRV ActionEffect rows predate archive-executable support
    custody.  They remain valuable for local planning, but they must not clear
    archive-closed birth or parse-back gates.  Preserve them by explicitly
    marking the support as research-only instead of inventing a support hash.
    """

    if candidate_kind == "frame0_pose":
        return {}
    if source.support_sha256:
        return {
            "support_source": source.support_source,
            "support_cardinality": source.support_cardinality,
            "support_sha256": source.support_sha256,
            "support_encoding": source.support_encoding,
            "support_encoded_bytes": source.support_encoded_bytes,
            "support_research_only": source.support_research_only,
        }
    if not source.region_ids and not source.class_ids:
        return {}
    return {
        "support_source": "action_effect_region_id_only_no_pixel_support",
        "support_cardinality": None,
        "support_sha256": None,
        "support_encoding": "research_only_region_id_not_archive_support",
        "support_encoded_bytes": None,
        "support_research_only": True,
    }


def _candidate_metadata(candidate_kind: str) -> tuple[int | str, str, str, str, str]:
    if candidate_kind == "frame0_pose":
        return (
            0,
            "pose_only",
            "posenet_yuv6_gradient",
            "frame0_pose_inverse_candidate",
            "frame0_pose",
        )
    if candidate_kind == "frame1_seg":
        return (
            1,
            "seg_pose_joint",
            "segnet_margin_vjp",
            "frame1_seg_margin_inverse_candidate",
            "frame1_seg_margin",
        )
    if candidate_kind == "composite":
        return (
            "both",
            "seg_pose_joint",
            "joint_seg_pose_projection",
            "frame1_birth_frame0_pose_composite_candidate",
            "frame1_birth_frame0_pose_comp",
        )
    raise ValueError(f"unknown inverse candidate kind: {candidate_kind}")


def _candidate_blockers(effect: ActionEffect) -> list[str]:
    blockers = list(effect.blockers)
    if not _receiver_visible(effect):
        blockers.append(BLOCKER_RECEIVER_SURFACE_MISSING)
    if effect.delta_score_total is None and effect.delta_score_nonrate is None:
        blockers.append(BLOCKER_SCORE_DELTA_MISSING)
    if _requires_region_support(effect):
        if effect.support_research_only is True:
            blockers.append(BLOCKER_REGION_SUPPORT_RESEARCH_ONLY)
            blockers.append(BLOCKER_ARCHIVE_CLOSED_BIRTH_REQUIRES_EXECUTABLE_SUPPORT)
        if not _has_executable_region_support(effect):
            blockers.append(BLOCKER_REGION_SUPPORT_IDENTITY_MISSING)
    return _dedupe(blockers)


def _requires_region_support(effect: ActionEffect) -> bool:
    return bool(
        effect.frame_index in {1, "both"}
        and (
            effect.region_ids
            or effect.wrong_to_target is not None
            or effect.target_to_wrong is not None
            or effect.wrong_to_wrong is not None
        )
    )


def _has_executable_region_support(effect: ActionEffect) -> bool:
    return bool(
        effect.support_source
        and effect.support_cardinality is not None
        and effect.support_cardinality >= 0
        and effect.support_sha256
        and effect.support_encoding
        and effect.support_encoded_bytes is not None
        and effect.support_encoded_bytes >= 0
        and effect.support_research_only is not True
    )


def _score_program_operation_for_effect(
    effect: ActionEffect,
    *,
    index: int,
    menu_cluster_hint: str,
    row_blockers: Sequence[str],
) -> dict[str, Any]:
    opcode, basis = _score_program_opcode_and_basis(menu_cluster_hint, effect)
    blockers = list(row_blockers)
    promotion_blockers = _score_program_promotion_blockers(effect)
    return {
        "schema": SCORE_PROGRAM_OPERATION_SCHEMA,
        "index": int(index),
        "action_id": effect.action_id,
        "opcode": opcode,
        "basis": basis,
        "backend": _score_program_backend(effect),
        "family": effect.family,
        "authority": effect.authority,
        "normalization_scope": effect.normalization_scope,
        "pair_ids": list(effect.pair_ids),
        "frame_index": effect.frame_index,
        "frame_incidence": effect.frame_incidence,
        "region_ids": list(effect.region_ids),
        "class_ids": list(effect.class_ids),
        "support_source": effect.support_source,
        "support_cardinality": effect.support_cardinality,
        "support_sha256": effect.support_sha256,
        "support_encoding": effect.support_encoding,
        "support_encoded_bytes": effect.support_encoded_bytes,
        "support_research_only": effect.support_research_only,
        "payload_sections": list(effect.payload_sections),
        "trained_groups": list(effect.trained_groups),
        "delta_score_nonrate": effect.delta_score_nonrate,
        "delta_score_total": effect.delta_score_total,
        "segnet_margin_delta": effect.segnet_margin_delta,
        "fakequant_segnet_margin_delta": effect.fakequant_segnet_margin_delta,
        "parseback_segnet_margin_delta": effect.parseback_segnet_margin_delta,
        "delta_bytes": effect.delta_bytes,
        "value_per_byte": effect.value_per_byte,
        "receiver_visible": _receiver_visible(effect),
        "survival": {
            "fakequant": effect.fakequant_survived,
            "parseback": effect.parseback_survived,
            "inflate": effect.inflate_survived,
        },
        "archive_sha256": effect.archive_sha256,
        "payload_sha256": effect.payload_sha256,
        "base_state_sha256": effect.base_state_sha256,
        "blockers": _dedupe(blockers),
        "promotion_blockers": promotion_blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _score_program_operation_from_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    embedded = row.get("score_program_operation")
    if isinstance(embedded, Mapping):
        operation = dict(embedded)
    else:
        opcode, basis = _score_program_opcode_and_basis(
            str(row.get("menu_cluster_hint") or ""),
            None,
        )
        operation = {
            "schema": SCORE_PROGRAM_OPERATION_SCHEMA,
            "action_id": str(row.get("action_id") or ""),
            "opcode": opcode,
            "basis": basis,
            "backend": str(row.get("backend") or "native_score_program"),
            "pair_ids": list(row.get("pair_ids") or []),
            "frame_index": row.get("frame_index"),
            "frame_incidence": row.get("frame_incidence"),
            "region_ids": list(row.get("region_ids") or []),
            "segnet_margin_delta": row.get("segnet_margin_delta"),
            "fakequant_segnet_margin_delta": row.get("fakequant_segnet_margin_delta"),
            "parseback_segnet_margin_delta": row.get("parseback_segnet_margin_delta"),
        }
    operation["index"] = int(index)
    operation.setdefault("schema", SCORE_PROGRAM_OPERATION_SCHEMA)
    operation.setdefault("score_claim", False)
    operation.setdefault("promotion_eligible", False)
    operation.setdefault("ready_for_exact_eval_dispatch", False)
    operation.setdefault("blockers", list(row.get("blockers") or []))
    operation.setdefault("promotion_blockers", list(row.get("promotion_blockers") or []))
    return operation


def _score_program_runtime_blockers(
    row: Mapping[str, Any],
    operation: Mapping[str, Any],
) -> list[str]:
    return _dedupe(
        [
            *[
                blocker
                for value in row.get("blockers") or []
                if (blocker := str(value)) not in _PROMOTION_ONLY_BLOCKERS
            ],
            *[str(value) for value in row.get("menu_ilp_blockers") or []],
            *[
                blocker
                for value in operation.get("blockers") or []
                if (blocker := str(value)) not in _PROMOTION_ONLY_BLOCKERS
            ],
        ]
    )


def _score_program_opcode_and_basis(
    menu_cluster_hint: str,
    effect: ActionEffect | None,
) -> tuple[str, str]:
    cluster = str(menu_cluster_hint or "").lower()
    if cluster == "frame0_pose" or (effect is not None and effect.frame_index == 0):
        return "APPLY_FRAME0_POSE_ACTION", "B0_frame0_pose_only"
    if cluster == "frame1_seg_margin" or (effect is not None and effect.frame_index == 1):
        return "APPLY_FRAME1_SEG_ACTION", "B1_frame1_seg_wall_cross"
    if cluster == "frame1_birth_frame0_pose_comp" or (
        effect is not None and effect.frame_index == "both"
    ):
        return "APPLY_BOTH_FRAME_COMPOSITE", "B3_both_frame_composite"
    if effect is not None and effect.family == "frontier_rate_attack":
        return "APPLY_QRGB_OR_SCORER_EFFECT", "B4_byte_archive_grammar"
    if effect is not None and effect.family == "snerv":
        return "RENDER_BASE_WITNESS_PAIR", "B5_source_forward_representation"
    return "APPLY_QRGB_OR_SCORER_EFFECT", "B2_frame1_joint_seg_pose_trust"


def _score_program_backend(effect: ActionEffect) -> str:
    family = effect.family.lower()
    if family == "hinerv":
        return "hinerv_grid_adapter"
    if family == "snerv":
        return "snerv_lf_hf_mfu_hfr_tub"
    if family == "pr110":
        return "selector_vm"
    if family == "frontier_rate_attack":
        return "byte_archive_rewrite"
    if family == "pact_nerv":
        return "semantic_pose_renderer"
    return "native_score_program"


def _score_program_promotion_blockers(effect: ActionEffect) -> list[str]:
    blockers: list[str] = []
    for blocker in _candidate_blockers(effect):
        if blocker in _PROMOTION_ONLY_BLOCKERS:
            blockers.append(blocker)
    if not effect.archive_sha256:
        blockers.append(BLOCKER_SCORE_PROGRAM_ARCHIVE_HASH_MISSING)
    if effect.parseback_survived is not True:
        blockers.append(BLOCKER_SCORE_PROGRAM_PARSEBACK_MISSING)
    if effect.inflate_survived is not True:
        blockers.append(BLOCKER_SCORE_PROGRAM_INFLATE_MISSING)
    return _dedupe(blockers)


def _receiver_visible(effect: ActionEffect) -> bool:
    surface = effect.receiver_surface
    return any(
        value not in (None, 0, 0.0)
        for value in (
            surface.uint8_changed_pixels,
            surface.seg_input_delta_linf,
            surface.posenet_input_delta_linf,
            surface.seg_argmax_changed_pixels,
            surface.pose_output_l2_delta,
            effect.uint8_changed_count_region,
            effect.seg_input_delta_linf_region,
            effect.posenet_input_delta_linf_pair,
            effect.pose_output_l2_delta,
            effect.wrong_to_target,
            effect.wrong_to_wrong,
        )
    )


def _is_frame0_pose_action(effect: ActionEffect) -> bool:
    text = " ".join([effect.action_kind, effect.arm or "", *effect.trained_groups, *effect.payload_sections]).lower()
    if _composite_text(text):
        return False
    return (
        "frame0_pose" in text
        or "pose_target_only" in text
        or ("head_rgb_0" in text and "head_rgb_1" not in text)
    )


def _is_frame1_seg_action(effect: ActionEffect) -> bool:
    text = " ".join([effect.action_kind, effect.arm or "", *effect.trained_groups, *effect.payload_sections]).lower()
    return (
        ("birth" in text or "seg" in text or "head_rgb_1" in text)
        and "frame0_pose" not in text
        and "composite" not in text
        and "joint" not in text
        and (effect.wrong_to_target or effect.hard_won_count or 0) > 0
    )


def _is_composite_action(effect: ActionEffect) -> bool:
    text = " ".join([effect.action_kind, effect.arm or "", *effect.trained_groups, *effect.payload_sections]).lower()
    return _composite_text(text) or effect.interaction_or_commutator is not None


def _is_wall_normal_branch_action(effect: ActionEffect) -> bool:
    text = " ".join([effect.action_kind, effect.inverse_source or "", *effect.payload_sections]).lower()
    return (
        "sidecar_candidate" in text
        or "wall_normal_branch" in text
        or "target_region_action_sidecar" in text
    )


def _composite_text(text: str) -> bool:
    return (
        "composite" in text
        or "joint_line_search" in text
        or "birth_plus_frame0_pose" in text
    )


def _first_matching(effects: Sequence[ActionEffect], predicate: Any) -> ActionEffect | None:
    matches = _matching(effects, predicate)
    if not matches:
        return None
    return matches[0]


def _matching(effects: Sequence[ActionEffect], predicate: Any) -> list[ActionEffect]:
    matches = [effect for effect in effects if predicate(effect) and _receiver_visible(effect)]
    return sorted(matches, key=_candidate_sort_key, reverse=True)


def _composite_dependencies_and_id(
    composite: ActionEffect,
    *,
    frame0_candidate: InverseCandidate | None,
    frame1_candidate: InverseCandidate | None,
) -> tuple[tuple[str, ...], str | None]:
    ordered = _composite_order(composite)
    by_kind = {
        "frame0_pose": frame0_candidate,
        "frame1_seg": frame1_candidate,
    }
    deps = tuple(
        by_kind[kind].effect.action_id
        for kind in ordered
        if by_kind.get(kind) is not None
    )
    action_id_override = (
        f"{deps[0]}__then__{deps[1]}__inverse_composite"
        if len(deps) == 2
        else None
    )
    return deps, action_id_override


def _composite_order(effect: ActionEffect) -> tuple[str, str]:
    text = " ".join([effect.action_kind, effect.arm or "", *effect.trained_groups, *effect.payload_sections]).lower()
    if (
        "frame0_pose_then_birth" in text
        or "pose_then_birth" in text
        or "pose_then_seg" in text
    ):
        return ("frame0_pose", "frame1_seg")
    return ("frame1_seg", "frame0_pose")


def _candidate_sort_key(effect: ActionEffect) -> tuple[float, int, int, int, str]:
    score_ev = _score_ev(effect)
    return (
        score_ev if score_ev is not None else -math.inf,
        1 if effect.exact_score_decision == "accept" else 0,
        1 if not effect.blockers else 0,
        1 if effect.restore_state_pass is True else 0,
        effect.action_kind,
    )


def _score_ev(effect: ActionEffect) -> float | None:
    delta = effect.delta_score_total if effect.delta_score_total is not None else effect.delta_score_nonrate
    if delta is None:
        return None
    value = -float(delta)
    return value if math.isfinite(value) else None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _normal_exact_decision(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"accept", "accepted"}:
        return "accept"
    if text in {"reject", "rejected"}:
        return "reject"
    return "not_applicable"


def _first_int(
    payload: Mapping[str, Any],
    *keys: str,
    default: int | None = None,
) -> int | None:
    for key in keys:
        if key not in payload:
            continue
        parsed = _int_or_none(payload.get(key))
        if parsed is not None:
            return parsed
    return default


def _first_float(
    payload: Mapping[str, Any],
    *keys: str,
    default: float | None = None,
) -> float | None:
    for key in keys:
        if key not in payload:
            continue
        parsed = _finite_or_none(payload.get(key))
        if parsed is not None:
            return parsed
    return default


def _contains_any(values: Any, needles: Sequence[str]) -> bool:
    if isinstance(values, str):
        value_set = {values}
    elif isinstance(values, Sequence):
        value_set = {str(item) for item in values}
    else:
        value_set = set()
    return any(needle in value_set for needle in needles)


def _support_mask_sha256(mask: np.ndarray) -> str:
    mask_bool = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    h = hashlib.sha256()
    h.update(str(tuple(int(v) for v in mask_bool.shape)).encode("ascii"))
    h.update(b"\0")
    h.update(np.packbits(mask_bool.reshape(-1).astype(np.uint8)).tobytes())
    return h.hexdigest()


def _byte_cost(effect: ActionEffect) -> int | None:
    if effect.delta_bytes is None:
        return None
    return abs(int(effect.delta_bytes))


def _menu_cluster_hint(effect: ActionEffect) -> str:
    if effect.frame_index == 0:
        return "frame0_pose"
    if effect.frame_index == 1:
        return "frame1_seg_margin"
    if effect.frame_index == "both":
        return "frame1_birth_frame0_pose_comp"
    return "unclassified"


def _coerce_effects(effects: Sequence[Any]) -> list[ActionEffect]:
    if isinstance(effects, (str, bytes)) or not isinstance(effects, Sequence):
        raise TypeError("effects must be a sequence of ActionEffect rows")
    out: list[ActionEffect] = []
    for effect in effects:
        if not isinstance(effect, ActionEffect):
            raise TypeError(f"expected ActionEffect; got {type(effect)!r}")
        out.append(effect)
    return out


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _finite_or_none(value: Any) -> float | None:
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


__all__ = [
    "BLOCKER_ARCHIVE_CLOSED_BIRTH_REQUIRES_EXECUTABLE_SUPPORT",
    "BLOCKER_DIRECT_SEG_WALL_EMPTY_SUPPORT",
    "BLOCKER_DIRECT_SEG_WALL_EXACT_SCORE_NOT_IMPROVED",
    "BLOCKER_DIRECT_SEG_WALL_NO_UINT8_MOTION",
    "BLOCKER_NO_COMPOSITE",
    "BLOCKER_NO_FRAME0_POSE",
    "BLOCKER_NO_FRAME1_SEG",
    "BLOCKER_RECEIVER_SURFACE_MISSING",
    "BLOCKER_REGION_SUPPORT_IDENTITY_MISSING",
    "BLOCKER_REGION_SUPPORT_RESEARCH_ONLY",
    "BLOCKER_SCORE_DELTA_MISSING",
    "BLOCKER_SCORE_PROGRAM_ARCHIVE_HASH_MISSING",
    "BLOCKER_SCORE_PROGRAM_INFLATE_MISSING",
    "BLOCKER_SCORE_PROGRAM_PARSEBACK_MISSING",
    "BLOCKER_UINT8_MOTION_WITHOUT_SEGNET_WALL_CROSSING",
    "BLOCKER_WALL_NORMAL_BACKEND_EXACT_SCORE_NOT_ACCEPTED",
    "BLOCKER_WALL_NORMAL_BACKEND_FIT_MISSING",
    "BLOCKER_WALL_NORMAL_BACKEND_NOT_REALIZED",
    "BLOCKER_WALL_NORMAL_DIRECT_TEACHER_EXACT_SCORE_NOT_ACCEPTED",
    "BLOCKER_WALL_NORMAL_DIRECT_TEACHER_MISSING",
    "BLOCKER_WALL_NORMAL_DIRECT_TEACHER_NOT_CROSSED",
    "BLOCKER_WALL_NORMAL_DIRECT_TEACHER_NOT_TRUE_WALL_NORMAL",
    "BLOCKER_WALL_NORMAL_SIDECAR_ARCHIVE_UNCLOSED",
    "DIRECT_SEG_WALL_ORACLE_SCHEMA",
    "INVERSE_SCORER_GENERATION_SCHEMA",
    "INVERSE_SCORER_QUEUE_ROW_SCHEMA",
    "SCORE_PROGRAM_INTERPRETER",
    "SCORE_PROGRAM_OPERATION_SCHEMA",
    "SCORE_PROGRAM_WORD_SCHEMA",
    "TARGET_REGION_WALL_NORMAL_LIFT_SCHEMA",
    "TRUE_SEG_WALL_NORMAL_INVERSE_SOURCES",
    "InverseCandidate",
    "build_candidate_queue",
    "build_direct_seg_wall_oracle_receipt",
    "build_masked_residual_oracle_action_effect",
    "build_score_program_word",
    "build_target_region_wall_normal_lift_receipt",
    "candidate_queue_jsonl",
    "generate_inverse_scorer_candidates",
]
