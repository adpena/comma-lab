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

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tac.analysis.action_effect import ActionEffect
from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS

INVERSE_SCORER_GENERATION_SCHEMA = "tac.inverse_scorer_action_generation.v1"
INVERSE_SCORER_QUEUE_ROW_SCHEMA = "tac.inverse_scorer_candidate_queue_row.v1"

BLOCKER_NO_FRAME0_POSE = "inverse_scorer_frame0_pose_action_missing"
BLOCKER_NO_FRAME1_SEG = "inverse_scorer_frame1_seg_margin_action_missing"
BLOCKER_NO_COMPOSITE = "inverse_scorer_composite_action_missing"
BLOCKER_RECEIVER_SURFACE_MISSING = "inverse_scorer_receiver_surface_motion_missing"
BLOCKER_SCORE_DELTA_MISSING = "inverse_scorer_exact_delta_missing"


@dataclass(frozen=True)
class InverseCandidate:
    """A measured ActionEffect row plus planner queue metadata."""

    effect: ActionEffect
    menu_cluster_hint: str
    dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()


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
    composite = _first_matching(effects, _is_composite_action)

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

    if composite is None:
        blockers.append(BLOCKER_NO_COMPOSITE)
    else:
        deps = tuple(
            candidate.effect.action_id
            for candidate in (frame1_candidate, frame0_candidate)
            if candidate is not None
        )
        action_id_override = (
            f"{deps[0]}__then__{deps[1]}__inverse_composite"
            if len(deps) == 2
            else None
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
    return {
        "schema": INVERSE_SCORER_GENERATION_SCHEMA,
        "input_effect_count": len(effects),
        "candidate_count": len(candidate_effects),
        "blockers": blockers,
        "passed": not blockers,
        "action_effects": candidate_effects,
        "candidate_queue": queue_rows,
        "policy": {
            "measured_effects_only_no_synthetic_scorer_motion": True,
            "promotion_eligible_is_false_for_generated_rows": True,
            "frame0_pose_only_maps_to_pose_incidence": True,
            "frame1_maps_to_seg_pose_joint_incidence": True,
        },
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def build_candidate_queue(
    effects: Sequence[ActionEffect],
    candidates: Sequence[InverseCandidate] | None = None,
) -> list[dict[str, Any]]:
    """Build menu-ILP input rows without optimizing a menu."""

    by_id = {candidate.effect.action_id: candidate for candidate in candidates or ()}
    rows: list[dict[str, Any]] = []
    for effect in effects:
        candidate = by_id.get(effect.action_id)
        blockers = _candidate_blockers(effect)
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
                "menu_cluster_hint": (
                    candidate.menu_cluster_hint if candidate is not None else _menu_cluster_hint(effect)
                ),
                "score_ev": _score_ev(effect),
                "byte_cost": _byte_cost(effect),
                "value_per_byte": effect.value_per_byte,
                "dependencies": list(candidate.dependencies if candidate is not None else ()),
                "conflicts": list(candidate.conflicts if candidate is not None else ()),
                "blockers": blockers,
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
    return (
        "composite" in text
        or "joint_line_search" in text
        or "birth_plus_frame0_pose" in text
        or effect.interaction_or_commutator is not None
    )


def _first_matching(effects: Sequence[ActionEffect], predicate: Any) -> ActionEffect | None:
    matches = [effect for effect in effects if predicate(effect) and _receiver_visible(effect)]
    if not matches:
        return None
    return max(matches, key=lambda effect: _score_ev(effect) if _score_ev(effect) is not None else -math.inf)


def _score_ev(effect: ActionEffect) -> float | None:
    delta = effect.delta_score_total if effect.delta_score_total is not None else effect.delta_score_nonrate
    if delta is None:
        return None
    value = -float(delta)
    return value if math.isfinite(value) else None


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


__all__ = [
    "BLOCKER_NO_COMPOSITE",
    "BLOCKER_NO_FRAME0_POSE",
    "BLOCKER_NO_FRAME1_SEG",
    "BLOCKER_RECEIVER_SURFACE_MISSING",
    "BLOCKER_SCORE_DELTA_MISSING",
    "INVERSE_SCORER_GENERATION_SCHEMA",
    "INVERSE_SCORER_QUEUE_ROW_SCHEMA",
    "InverseCandidate",
    "build_candidate_queue",
    "candidate_queue_jsonl",
    "generate_inverse_scorer_candidates",
]
