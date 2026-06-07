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

BLOCKER_NO_FRAME0_POSE = "inverse_scorer_frame0_pose_action_missing"
BLOCKER_NO_FRAME1_SEG = "inverse_scorer_frame1_seg_margin_action_missing"
BLOCKER_NO_COMPOSITE = "inverse_scorer_composite_action_missing"
BLOCKER_RECEIVER_SURFACE_MISSING = "inverse_scorer_receiver_surface_motion_missing"
BLOCKER_SCORE_DELTA_MISSING = "inverse_scorer_exact_delta_missing"
BLOCKER_SCORE_PROGRAM_ARCHIVE_HASH_MISSING = "score_program_archive_hash_missing"
BLOCKER_SCORE_PROGRAM_PARSEBACK_MISSING = "score_program_parseback_survival_missing"
BLOCKER_SCORE_PROGRAM_INFLATE_MISSING = "score_program_inflate_survival_missing"
BLOCKER_DIRECT_SEG_WALL_EMPTY_SUPPORT = "direct_seg_wall_oracle_empty_support"
BLOCKER_DIRECT_SEG_WALL_NO_UINT8_MOTION = "direct_seg_wall_oracle_no_uint8_motion"
BLOCKER_UINT8_MOTION_WITHOUT_SEGNET_WALL_CROSSING = (
    "uint8_motion_without_segnet_wall_crossing"
)
BLOCKER_DIRECT_SEG_WALL_EXACT_SCORE_NOT_IMPROVED = (
    "direct_seg_wall_oracle_exact_score_not_improved"
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

    blockers: list[str] = []
    if support_cardinality <= 0:
        blockers.append(BLOCKER_DIRECT_SEG_WALL_EMPTY_SUPPORT)
    if uint8_changed == 0:
        blockers.append(BLOCKER_DIRECT_SEG_WALL_NO_UINT8_MOTION)
    if (uint8_changed or 0) > 0 and wrong_to_target <= 0:
        blockers.append(BLOCKER_UINT8_MOTION_WITHOUT_SEGNET_WALL_CROSSING)

    provisional = ActionEffect.build(
        action_id=str(action_id),
        family="direct_seg_wall_oracle",
        action_kind="direct_seg_wall_teacher",
        inverse_source="segnet_margin_gradient",
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
        blockers=blockers,
    )
    exact_blockers = list(blockers)
    if provisional.delta_score_nonrate is None or provisional.delta_score_nonrate >= 0.0:
        exact_blockers.append(BLOCKER_DIRECT_SEG_WALL_EXACT_SCORE_NOT_IMPROVED)
    effect_payload = provisional.as_dict()
    effect_payload["exact_score_decision"] = "accept" if not exact_blockers else "reject"
    effect_payload["blockers"] = exact_blockers
    effect = ActionEffect.from_dict(effect_payload)
    support_hash = _support_mask_sha256(support)
    return {
        "schema": DIRECT_SEG_WALL_ORACLE_SCHEMA,
        "action_id": effect.action_id,
        "authority": effect.authority,
        "pair_id": int(pair_id),
        "target_class": int(target_class),
        "support_source": str(support_source),
        "support_cardinality": int(support_cardinality),
        "support_sha256": support_hash,
        "support_encoding": str(support_encoding),
        "support_encoded_bytes": support_encoded_bytes,
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
        }
    )
    effect = ActionEffect.from_dict(payload)
    return InverseCandidate(
        effect=effect,
        menu_cluster_hint=cluster,
        dependencies=tuple(str(item) for item in dependencies if item),
        conflicts=tuple(str(item) for item in conflicts if item),
    )


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
            "segnet_margin_gradient",
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
    return _dedupe(blockers)


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
        "payload_sections": list(effect.payload_sections),
        "trained_groups": list(effect.trained_groups),
        "delta_score_nonrate": effect.delta_score_nonrate,
        "delta_score_total": effect.delta_score_total,
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
            *[str(value) for value in row.get("blockers") or []],
            *[str(value) for value in row.get("menu_ilp_blockers") or []],
            *[str(value) for value in operation.get("blockers") or []],
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
    "BLOCKER_NO_COMPOSITE",
    "BLOCKER_NO_FRAME0_POSE",
    "BLOCKER_NO_FRAME1_SEG",
    "BLOCKER_RECEIVER_SURFACE_MISSING",
    "BLOCKER_SCORE_DELTA_MISSING",
    "BLOCKER_SCORE_PROGRAM_ARCHIVE_HASH_MISSING",
    "BLOCKER_SCORE_PROGRAM_INFLATE_MISSING",
    "BLOCKER_SCORE_PROGRAM_PARSEBACK_MISSING",
    "INVERSE_SCORER_GENERATION_SCHEMA",
    "INVERSE_SCORER_QUEUE_ROW_SCHEMA",
    "SCORE_PROGRAM_INTERPRETER",
    "SCORE_PROGRAM_OPERATION_SCHEMA",
    "SCORE_PROGRAM_WORD_SCHEMA",
    "InverseCandidate",
    "build_candidate_queue",
    "build_score_program_word",
    "candidate_queue_jsonl",
    "generate_inverse_scorer_candidates",
]
