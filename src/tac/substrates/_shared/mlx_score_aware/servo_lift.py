# SPDX-License-Identifier: MIT
"""Lift continuous MLX scorer proposals into canonical ActionEffect rows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tac.analysis.action_effect import ACTION_EFFECT_SCHEMA, build_action_effect
from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS

PARSEBACK_SERVO_LIFT_SCHEMA = "mlx_score_aware_parseback_servo_lift.v1"


def servo_lift(
    proposal: Mapping[str, Any],
    trace_old: Mapping[str, Any] | None = None,
    *,
    family: str,
    stage: str,
    consumer: str = "nerv_long_training_campaign_admission",
    min_score_improvement: float = 0.0,
    require_inflate_survival: bool = True,
) -> dict[str, Any]:
    """Convert a continuous proposal into a receiver-surface ActionEffect.

    ``proposal`` may come from HiNeRV target-region birth, SNeRV LF/HF/TUB
    movement, or selector replay. The output is intentionally the shared
    ``ActionEffect`` payload so downstream admission, commutator, and menu
    solvers consume one typed currency.
    """

    old = dict(trace_old or proposal.get("trace_old") or proposal.get("old") or {})
    new = dict(proposal.get("trace_new") or proposal.get("new") or {})
    receiver_surface = _receiver_surface(proposal)
    action_effect = build_action_effect(
        {
            "action_id": _text(
                proposal.get("action_id")
                or proposal.get("proposal_id")
                or proposal.get("id")
                or f"{family}_{stage}_servo_lift"
            ),
            "family": proposal.get("family") or family,
            "authority": proposal.get("authority") or "parseback_mlx",
            "producer": proposal.get("producer") or "mlx_score_aware_parseback_servo_lift",
            "consumer": proposal.get("consumer") or consumer,
            "affected_pairs": proposal.get("affected_pairs") or proposal.get("pair_ids"),
            "affected_regions": proposal.get("affected_regions"),
            "payload_sections": proposal.get("payload_sections"),
            "state_custody": proposal.get("state_custody"),
            "archive_sha256": proposal.get("archive_sha256"),
            "candidate_archive_sha256": proposal.get("candidate_archive_sha256"),
            "source_archive_sha256": proposal.get("source_archive_sha256"),
            "payload_sha256": proposal.get("payload_sha256"),
            "runtime_tree_sha256": proposal.get("runtime_tree_sha256"),
            "section_tree_sha256": proposal.get("section_tree_sha256"),
            "old_d_seg": _first(proposal, old, "old_d_seg", "d_seg"),
            "new_d_seg": _first(proposal, new, "new_d_seg", "d_seg"),
            "old_d_pose": _first(proposal, old, "old_d_pose", "d_pose"),
            "new_d_pose": _first(proposal, new, "new_d_pose", "d_pose"),
            "old_bytes": _first(proposal, old, "old_bytes", "archive_bytes"),
            "new_bytes": _first(proposal, new, "new_bytes", "archive_bytes"),
            "receiver_surface": receiver_surface,
            "fakequant_survived": _truthy(proposal, receiver_surface, "fakequant_survived", "fakequant_survival"),
            "parseback_survived": _truthy(proposal, receiver_surface, "parseback_survived", "parseback_survival"),
            "inflate_survived": _optional_truthy(proposal, receiver_surface, "inflate_survived", "inflate_survival"),
            "value_per_byte": proposal.get("value_per_byte"),
        },
        min_score_improvement=min_score_improvement,
    )
    blockers = list(action_effect.get("blockers") or [])
    blockers.extend(
        _servo_surface_blockers(
            receiver_surface,
            require_inflate_survival=bool(require_inflate_survival),
        )
    )
    blockers = _dedupe(blockers)
    accepted = action_effect.get("action_effect_admitted") is True and not blockers
    return {
        "schema": PARSEBACK_SERVO_LIFT_SCHEMA,
        "family": str(family),
        "stage": str(stage),
        "proposal_id": action_effect.get("action_id"),
        "action_effect_schema": ACTION_EFFECT_SCHEMA,
        "action_effect": action_effect,
        "servo_lift_accepted": accepted,
        "receiver_visible": action_effect.get("receiver_visible") is True,
        "score_admissible": action_effect.get("score_admissible") is True,
        "byte_priced": action_effect.get("byte_priced") is True,
        "uint8_receiver_contact": _positive(
            receiver_surface.get("receiver_surface_uint8_changed_pixels"),
            receiver_surface.get("uint8_changed_pixels"),
        ),
        "scorer_surface_motion": _scorer_surface_motion(receiver_surface),
        "inflate_survived": _optional_truthy(
            proposal,
            receiver_surface,
            "inflate_survived",
            "inflate_survival",
        ),
        "blockers": blockers,
        "policy": {
            "continuous_proposal_is_not_authority": True,
            "uint8_receiver_surface_required": True,
            "fakequant_and_parseback_survival_required": True,
            "inflate_survival_required": bool(require_inflate_survival),
            "exact_nonlinear_delta_score_prices_admission": True,
        },
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def _receiver_surface(proposal: Mapping[str, Any]) -> dict[str, Any]:
    surface = proposal.get("receiver_surface")
    if isinstance(surface, Mapping):
        return dict(surface)
    trace = proposal.get("receiver_trace")
    if isinstance(trace, Mapping):
        return dict(trace)
    return {}


def _servo_surface_blockers(
    receiver_surface: Mapping[str, Any],
    *,
    require_inflate_survival: bool,
) -> list[str]:
    blockers: list[str] = []
    if not _positive(
        receiver_surface.get("receiver_surface_uint8_changed_pixels"),
        receiver_surface.get("uint8_changed_pixels"),
    ):
        blockers.append("servo_lift_uint8_receiver_contact_missing")
    if not _scorer_surface_motion(receiver_surface):
        blockers.append("servo_lift_scorer_surface_motion_missing")
    if require_inflate_survival and receiver_surface.get("inflate_survival") is not True:
        blockers.append("servo_lift_inflate_survival_missing")
    return blockers


def _scorer_surface_motion(receiver_surface: Mapping[str, Any]) -> bool:
    return _positive(
        receiver_surface.get("receiver_surface_segnet_input_delta_linf"),
        receiver_surface.get("segnet_input_delta_linf"),
        receiver_surface.get("receiver_surface_posenet_input_delta_linf"),
        receiver_surface.get("posenet_input_delta_linf"),
        receiver_surface.get("receiver_surface_argmax_flipped_pixels"),
        receiver_surface.get("argmax_flipped_pixels"),
        receiver_surface.get("receiver_surface_pose_output_delta"),
        receiver_surface.get("pose_output_delta_l2"),
    )


def _positive(*values: Any) -> bool:
    for value in values:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            continue
        if candidate > 0.0:
            return True
    return False


def _first(
    proposal: Mapping[str, Any],
    trace: Mapping[str, Any],
    proposal_key: str,
    trace_key: str,
) -> Any:
    return proposal.get(proposal_key, trace.get(trace_key))


def _truthy(
    proposal: Mapping[str, Any],
    surface: Mapping[str, Any],
    proposal_key: str,
    surface_key: str,
) -> bool:
    return proposal.get(proposal_key) is True or surface.get(surface_key) is True


def _optional_truthy(
    proposal: Mapping[str, Any],
    surface: Mapping[str, Any],
    proposal_key: str,
    surface_key: str,
) -> bool | None:
    if proposal_key not in proposal and surface_key not in surface:
        return None
    return _truthy(proposal, surface, proposal_key, surface_key)


def _text(value: object) -> str:
    return str(value or "").strip()


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


__all__ = [
    "PARSEBACK_SERVO_LIFT_SCHEMA",
    "servo_lift",
]
