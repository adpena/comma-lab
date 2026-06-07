# SPDX-License-Identifier: MIT
"""Diagnose HiNeRV live/fakequant scorer-effect loss at archive parse-back.

This receipt is deliberately narrow: it binds one accepted hard-birth action
identity across the live/fakequant birth row, selected archive parse-back birth
row, optional target-region sidecar parse-back row, and sampled live-vs-receiver
export parity proof.  It does not grant score authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tac.score_geometry import SEG_COEFFICIENT
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

HI_NERV_LIVE_TO_PARSEBACK_DELTA_AUDIT_SCHEMA = (
    "hi_nerv_live_to_parseback_scorer_effect_delta_audit.v1"
)

FIRST_DIVERGENCES = (
    "tensor_not_exported",
    "archive_selection_swapped_candidate",
    "support_identity_drift",
    "decoded_action_identity_drift",
    "quantization_mismatch",
    "parseback_preprocess_mismatch",
    "margin_safety_too_low",
)


def build_hinerv_live_to_parseback_scorer_effect_delta_audit(
    *,
    fakequant_survival: Mapping[str, Any] | None = None,
    selected_birth_parseback_survival: Mapping[str, Any] | None = None,
    target_region_action_parseback_survival: Mapping[str, Any] | None = None,
    target_region_action_export_selection: Mapping[str, Any] | None = None,
    live_receiver_export_parity: Mapping[str, Any] | None = None,
    exported_state_manifest: Mapping[str, Any] | None = None,
    archive_resolution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact-action live -> parse-back scorer-effect audit row."""

    fakequant = _mapping(fakequant_survival)
    birth = _mapping(selected_birth_parseback_survival)
    action = _mapping(target_region_action_parseback_survival)
    selection = _mapping(target_region_action_export_selection)
    parity = _mapping(live_receiver_export_parity)
    manifest = _mapping(exported_state_manifest)
    archive = _mapping(archive_resolution)

    action_id = _first_text(
        fakequant,
        birth,
        action,
        selection,
        keys=("action_id",),
    )
    support_sha256 = _first_text(
        selection,
        action,
        keys=(
            "target_region_action_support_sha256",
            "expected_support_sha256",
            "support_sha256",
        ),
    )
    decoded_action_sha256 = _first_text(
        action,
        _nested_mapping(selection, "target_region_action_section_telemetry"),
        keys=("decoded_action_sha256", "expected_decoded_action_sha256"),
    )
    decoded_support_sha256 = _first_text(
        action,
        _nested_mapping(selection, "target_region_action_section_telemetry"),
        keys=("decoded_support_sha256", "expected_decoded_support_sha256"),
    )
    selected_archive_sha256 = _first_text(
        birth,
        archive,
        keys=("selected_archive_sha256", "parseback_archive_sha256", "archive_sha256"),
    )
    selected_archive_bytes = _first_int(
        birth,
        archive,
        keys=("selected_archive_bytes", "parseback_archive_bytes", "archive_bytes"),
    )
    action_archive_sha256 = _first_text(action, keys=("archive_sha256",))
    action_archive_bytes = _first_int(action, keys=("archive_bytes",))
    checkpoint_identity = {
        "selected_candidate_kind": _first_text(
            birth,
            archive,
            keys=("selected_candidate_kind", "candidate_kind"),
        ),
        "exported_state_npz_sha256": _first_text(
            manifest,
            keys=("artifact_sha256", "npz_sha256", "state_npz_sha256"),
        ),
        "live_tensor_sha256": _first_text(parity, keys=("live_tensor_sha256",)),
        "receiver_tensor_sha256": _first_text(parity, keys=("receiver_tensor_sha256",)),
    }

    live_wrong = _first_int(
        fakequant,
        birth,
        keys=("live_wrong_to_target_count", "live_wrong_to_target"),
    )
    fake_wrong = _first_int(
        fakequant,
        keys=(
            "fakequant_wrong_to_target_count",
            "wrong_to_target_count",
            "surface_wrong_to_target_count",
        ),
    )
    parse_wrong = _first_int(
        birth,
        keys=(
            "parseback_wrong_to_target_count",
            "wrong_to_target_count",
            "surface_wrong_to_target_count",
        ),
    )
    fake_retention = _ratio(fake_wrong, live_wrong)
    parse_retention = _ratio(parse_wrong, live_wrong)

    archive_identity = {
        "selected_archive_sha256": selected_archive_sha256,
        "selected_archive_bytes": selected_archive_bytes,
        "target_region_action_archive_sha256": action_archive_sha256,
        "target_region_action_archive_bytes": action_archive_bytes,
        "same_selected_and_action_archive": _same_text_or_none(
            selected_archive_sha256,
            action_archive_sha256,
        ),
    }
    action_support_cardinality = _first_int(
        selection,
        action,
        _nested_mapping(selection, "target_region_action_section_telemetry"),
        keys=(
            "target_region_action_support_cardinality",
            "support_cardinality",
            "total_action_pixels",
        ),
    )
    birth_region_pixel_count = _first_int(
        fakequant,
        birth,
        keys=("region_pixel_count",),
    )
    support_identity = {
        "support_sha256": support_sha256,
        "decoded_support_sha256": decoded_support_sha256,
        "target_region_action_support_cardinality": action_support_cardinality,
        "birth_region_pixel_count": birth_region_pixel_count,
        "live_wrong_to_target_count": live_wrong,
        "fakequant_wrong_to_target_count": fake_wrong,
        "parseback_wrong_to_target_count": parse_wrong,
    }
    decoded_action_identity = {
        "target_region_action_program_sha256": _first_text(
            action,
            selection,
            keys=(
                "target_region_action_program_sha256",
                "target_region_action_program_sha256",
            ),
        ),
        "encoded_program_sha256": _first_text(
            action,
            _nested_mapping(selection, "target_region_action_section_telemetry"),
            keys=("encoded_program_sha256",),
        ),
        "decoded_action_sha256": decoded_action_sha256,
        "expected_payload_sha256": _first_text(action, keys=("expected_payload_sha256",)),
        "stored_payload_sha256": _first_text(action, keys=("stored_payload_sha256",)),
    }

    surfaces = [
        _surface_row(
            "live_accepted",
            source=fakequant,
            action_id=action_id,
            support_sha256=support_sha256,
            decoded_action_sha256=decoded_action_sha256,
            archive_sha256=None,
            checkpoint_identity=checkpoint_identity,
            wrong_to_target=live_wrong,
            total_scored_pixels=_first_int(fakequant, birth, keys=("total_scored_pixels",)),
            margin_source=fakequant,
            rgb_uint8_delta_on_support=None,
            seg_input_delta_on_support=None,
            pose_output_delta=_pose_delta(fakequant),
        ),
        _surface_row(
            "fakequant",
            source=fakequant,
            action_id=action_id,
            support_sha256=support_sha256,
            decoded_action_sha256=decoded_action_sha256,
            archive_sha256=None,
            checkpoint_identity=checkpoint_identity,
            wrong_to_target=fake_wrong,
            total_scored_pixels=_first_int(fakequant, keys=("total_scored_pixels",)),
            margin_source=fakequant,
            rgb_uint8_delta_on_support=None,
            seg_input_delta_on_support=None,
            pose_output_delta=_pose_delta(fakequant),
        ),
        _surface_row(
            "archive_serialized_tensors_program",
            source=action,
            action_id=action_id,
            support_sha256=support_sha256,
            decoded_action_sha256=decoded_action_sha256,
            archive_sha256=selected_archive_sha256,
            checkpoint_identity=checkpoint_identity,
            wrong_to_target=None,
            total_scored_pixels=None,
            margin_source={},
            rgb_uint8_delta_on_support={
                "exact_uint8_action_pixels_applied": _first_int(
                    action,
                    keys=("exact_uint8_action_pixels_applied",),
                ),
                "receiver_changed_action_pixels": _first_int(
                    action,
                    keys=("receiver_changed_action_pixels",),
                ),
                "max_abs_action_rgb_error": _first_float(
                    action,
                    keys=("max_abs_action_rgb_error",),
                ),
                "max_abs_receiver_delta_vs_no_action": _first_float(
                    action,
                    keys=("max_abs_receiver_delta_vs_no_action",),
                ),
            },
            seg_input_delta_on_support=None,
            pose_output_delta=None,
        ),
        _surface_row(
            "parseback_loaded_tensors_program",
            source=parity,
            action_id=action_id,
            support_sha256=support_sha256,
            decoded_action_sha256=decoded_action_sha256,
            archive_sha256=selected_archive_sha256,
            checkpoint_identity=checkpoint_identity,
            wrong_to_target=None,
            total_scored_pixels=None,
            margin_source={},
            rgb_uint8_delta_on_support={
                "live_receiver_export_parity_passed": _bool_or_none(parity.get("passed")),
                "receiver_decode_passed": _bool_or_none(parity.get("receiver_decode_passed")),
                "changed_element_count": _first_int(parity, keys=("changed_element_count",)),
                "max_abs_delta": _first_float(parity, keys=("max_abs_delta",)),
                "mean_abs_delta": _first_float(parity, keys=("mean_abs_delta",)),
            },
            seg_input_delta_on_support=None,
            pose_output_delta=None,
        ),
        _surface_row(
            "parseback_segnet_logits_argmax",
            source=birth,
            action_id=action_id,
            support_sha256=support_sha256,
            decoded_action_sha256=decoded_action_sha256,
            archive_sha256=selected_archive_sha256,
            checkpoint_identity=checkpoint_identity,
            wrong_to_target=parse_wrong,
            total_scored_pixels=_first_int(birth, keys=("total_scored_pixels",)),
            margin_source=birth,
            rgb_uint8_delta_on_support=None,
            seg_input_delta_on_support=None,
            pose_output_delta=_pose_delta(birth),
        ),
        _surface_row(
            "parseback_posenet",
            source=birth,
            action_id=action_id,
            support_sha256=support_sha256,
            decoded_action_sha256=decoded_action_sha256,
            archive_sha256=selected_archive_sha256,
            checkpoint_identity=checkpoint_identity,
            wrong_to_target=None,
            total_scored_pixels=None,
            margin_source={},
            rgb_uint8_delta_on_support=None,
            seg_input_delta_on_support=None,
            pose_output_delta=_pose_delta(birth),
        ),
    ]

    first_divergence, divergence_reasons = _first_divergence(
        fakequant=fakequant,
        birth=birth,
        action=action,
        selection=selection,
        parity=parity,
        selected_archive_sha256=selected_archive_sha256,
        action_archive_sha256=action_archive_sha256,
        support_sha256=support_sha256,
        decoded_action_sha256=decoded_action_sha256,
        fake_retention=fake_retention,
        parse_retention=parse_retention,
    )
    blockers = _dedupe(
        [
            *[str(v) for v in fakequant.get("blockers") or []],
            *[str(v) for v in birth.get("blockers") or []],
            *[str(v) for v in action.get("blockers") or []],
            *[str(v) for v in parity.get("blockers") or []],
            *divergence_reasons,
            *(
                []
                if bool(birth.get("parseback_scorer_effect_survived") is True)
                else ["hinerv_birth_parseback_scorer_effect_collapse"]
            ),
        ]
    )
    next_operator = {
        "quantization_mismatch": (
            "bind fakequant/QAT to the exact HIV1 archive quantizer and sampled "
            "live-vs-receiver tensor parity surface"
        ),
        "tensor_not_exported": (
            "hash learned birth tensor groups live/fakequant/archive/parseback "
            "and bind any missing groups into export"
        ),
        "archive_selection_swapped_candidate": (
            "force selected archive/action continuity by action_id, support, "
            "program hash, archive sha, and checkpoint identity"
        ),
        "support_identity_drift": (
            "align birth support, direct-teacher support, sidecar support, and "
            "scorer-region support before another backend fit"
        ),
        "decoded_action_identity_drift": (
            "repair target-region action encoding/decoding identity before "
            "using it as scorer-effect evidence"
        ),
        "parseback_preprocess_mismatch": (
            "materialize official SegNet/PoseNet preprocess tensors for live "
            "and parseback and diff them on support"
        ),
        "margin_safety_too_low": (
            "increase receiver-surface margin floor and require parseback "
            "margin survival before long-run launch"
        ),
    }[first_divergence]

    return {
        "schema": HI_NERV_LIVE_TO_PARSEBACK_DELTA_AUDIT_SCHEMA,
        "family": "hinerv",
        "action_id": action_id,
        "support_sha256": support_sha256,
        "decoded_support_sha256": decoded_support_sha256,
        "decoded_action_sha256": decoded_action_sha256,
        "archive_sha256": selected_archive_sha256,
        "archive_bytes": selected_archive_bytes,
        "archive_identity": archive_identity,
        "checkpoint_identity": checkpoint_identity,
        "support_identity": support_identity,
        "decoded_action_identity": decoded_action_identity,
        "retention": {
            "live_wrong_to_target": live_wrong,
            "fakequant_wrong_to_target": fake_wrong,
            "parseback_wrong_to_target": parse_wrong,
            "fakequant_wrong_to_target_retention_ratio": fake_retention,
            "parseback_wrong_to_target_retention_ratio": parse_retention,
            "retention_floor": _first_float(
                fakequant,
                birth,
                keys=("scorer_effect_retention_floor",),
            ),
        },
        "parseback_payload_survived": _bool_or_none(birth.get("parseback_payload_survived")),
        "parseback_program_survived": _bool_or_none(
            action.get("parseback_program_survived")
            if "parseback_program_survived" in action
            else action.get("parseback_survived")
        ),
        "parseback_scorer_effect_survived": _bool_or_none(
            birth.get("parseback_scorer_effect_survived")
        ),
        "surfaces": surfaces,
        "first_divergence": first_divergence,
        "first_failed_surface": first_divergence,
        "first_divergence_reasons": divergence_reasons,
        "next_operator": next_operator,
        "blockers": blockers,
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def _surface_row(
    surface: str,
    *,
    source: Mapping[str, Any],
    action_id: str | None,
    support_sha256: str | None,
    decoded_action_sha256: str | None,
    archive_sha256: str | None,
    checkpoint_identity: Mapping[str, Any],
    wrong_to_target: int | None,
    total_scored_pixels: int | None,
    margin_source: Mapping[str, Any],
    rgb_uint8_delta_on_support: Mapping[str, Any] | None,
    seg_input_delta_on_support: Mapping[str, Any] | None,
    pose_output_delta: Mapping[str, Any] | None,
) -> dict[str, Any]:
    target_to_wrong = _first_int(source, keys=("target_to_wrong_count",))
    wrong_to_wrong = _first_int(source, keys=("wrong_to_wrong_count",))
    return {
        "surface": surface,
        "action_id": action_id,
        "support_sha256": support_sha256,
        "decoded_action_sha256": decoded_action_sha256,
        "archive_sha256": archive_sha256,
        "checkpoint_identity": dict(checkpoint_identity),
        "rgb_uint8_delta_on_support": (
            None if rgb_uint8_delta_on_support is None else dict(rgb_uint8_delta_on_support)
        ),
        "seg_input_delta_on_support": (
            None if seg_input_delta_on_support is None else dict(seg_input_delta_on_support)
        ),
        "margin_min": _first_float(margin_source, keys=("region_margin_min", "margin_min")),
        "margin_mean": _first_float(margin_source, keys=("region_margin_mean", "margin_mean")),
        "margin_p50": _first_float(
            margin_source,
            keys=("region_margin_p50", "margin_p50", "margin_median"),
        ),
        "wrong_to_target": wrong_to_target,
        "target_to_wrong": target_to_wrong,
        "wrong_to_wrong": wrong_to_wrong,
        "pose_output_delta": pose_output_delta,
        "exact_delta_score_nonrate": _seg_delta_score(
            wrong_to_target=wrong_to_target,
            target_to_wrong=target_to_wrong,
            total_scored_pixels=total_scored_pixels,
            pose_output_delta=pose_output_delta,
        ),
        "exact_delta_score_nonrate_source": (
            "seg_argmax_transitions_pose_delta_zero_or_unneeded"
            if _seg_delta_score(
                wrong_to_target=wrong_to_target,
                target_to_wrong=target_to_wrong,
                total_scored_pixels=total_scored_pixels,
                pose_output_delta=pose_output_delta,
            )
            is not None
            else None
        ),
        "blockers": [str(v) for v in source.get("blockers") or []],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _first_divergence(
    *,
    fakequant: Mapping[str, Any],
    birth: Mapping[str, Any],
    action: Mapping[str, Any],
    selection: Mapping[str, Any],
    parity: Mapping[str, Any],
    selected_archive_sha256: str | None,
    action_archive_sha256: str | None,
    support_sha256: str | None,
    decoded_action_sha256: str | None,
    fake_retention: float | None,
    parse_retention: float | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    action_ids = _nonempty_texts(
        fakequant.get("action_id"),
        birth.get("action_id"),
        action.get("action_id"),
        selection.get("action_id"),
    )
    if len(set(action_ids)) > 1:
        reasons.append("live_fakequant_parseback_action_id_mismatch")
        return "archive_selection_swapped_candidate", reasons

    if (
        selected_archive_sha256
        and action_archive_sha256
        and selected_archive_sha256 != action_archive_sha256
    ):
        reasons.append("selected_birth_archive_and_target_action_archive_mismatch")
        return "archive_selection_swapped_candidate", reasons

    support_candidates = _nonempty_texts(
        support_sha256,
        action.get("support_sha256"),
        action.get("expected_support_sha256"),
        selection.get("target_region_action_support_sha256"),
    )
    if len(set(support_candidates)) > 1:
        reasons.append("target_region_action_support_hash_mismatch")
        return "support_identity_drift", reasons

    telemetry = _nested_mapping(selection, "target_region_action_section_telemetry")
    decoded_action_candidates = _nonempty_texts(
        decoded_action_sha256,
        action.get("decoded_action_sha256"),
        telemetry.get("decoded_action_sha256"),
        telemetry.get("expected_decoded_action_sha256"),
    )
    if len(set(decoded_action_candidates)) > 1:
        reasons.append("target_region_action_decoded_action_hash_mismatch")
        return "decoded_action_identity_drift", reasons

    if action and action.get("parseback_survived") is not True:
        reasons.append("target_region_action_program_not_parseback_surviving")
        return "decoded_action_identity_drift", reasons

    if fake_retention is not None and fake_retention < _retention_floor(fakequant, birth):
        reasons.append("fakequant_scorer_effect_retention_below_floor")
        return "quantization_mismatch", reasons

    parity_passed = parity.get("passed")
    if (
        parity
        and parity_passed is not True
        and (
            parity.get("receiver_decode_passed") is True
            or _first_float(parity, keys=("max_abs_delta",), default=0.0) > 0.0
            or _first_int(parity, keys=("changed_element_count",), default=0) > 0
        )
    ):
        reasons.append("live_receiver_export_parity_failed")
        return "quantization_mismatch", reasons

    if birth.get("parseback_scorer_effect_survived") is not True:
        if _first_float(birth, keys=("region_margin_min",), default=1.0) <= 0.0:
            reasons.append("parseback_region_margin_min_not_positive")
            return "margin_safety_too_low", reasons
        if action.get("parseback_survived") is True:
            reasons.append("parseback_program_survived_but_scorer_effect_collapsed")
            return "parseback_preprocess_mismatch", reasons
        reasons.append("parseback_scorer_effect_collapse_without_tensor_trace")
        return "tensor_not_exported", reasons

    if parse_retention is not None and parse_retention < _retention_floor(fakequant, birth):
        reasons.append("parseback_scorer_effect_retention_below_floor")
        return "margin_safety_too_low", reasons

    reasons.append("parseback_scorer_effect_trace_incomplete")
    return "tensor_not_exported", reasons


def _seg_delta_score(
    *,
    wrong_to_target: int | None,
    target_to_wrong: int | None,
    total_scored_pixels: int | None,
    pose_output_delta: Mapping[str, Any] | None,
) -> float | None:
    if wrong_to_target is None or total_scored_pixels is None or int(total_scored_pixels) <= 0:
        return None
    if pose_output_delta and pose_output_delta.get("score_delta") not in (None, 0.0):
        return None
    lost = int(target_to_wrong or 0)
    return float(SEG_COEFFICIENT * (lost - int(wrong_to_target)) / int(total_scored_pixels))


def _pose_delta(source: Mapping[str, Any]) -> dict[str, Any] | None:
    pose = source.get("pose_compensation_survival")
    if not isinstance(pose, Mapping):
        return None
    return {
        "required": _bool_or_none(pose.get("required")),
        "survived": _bool_or_none(pose.get("survived")),
        "live_composite_d_pose_batch": _first_float(
            pose,
            keys=("live_composite_d_pose_batch",),
        ),
        "surface_d_pose_batch": _first_float(pose, keys=("surface_d_pose_batch",)),
        "score_delta": None,
        "blockers": [str(v) for v in pose.get("blockers") or []],
    }


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested_mapping(source: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Any = source
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def _first_text(*nodes: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        for key in keys:
            value = node.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _first_int(
    *nodes: Mapping[str, Any],
    keys: Sequence[str],
    default: int | None = None,
) -> int | None:
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        for key in keys:
            value = node.get(key)
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                continue
    return default


def _first_float(
    *nodes: Mapping[str, Any],
    keys: Sequence[str],
    default: float | None = None,
) -> float | None:
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        for key in keys:
            value = node.get(key)
            try:
                if value is not None:
                    out = float(value)
                    return out if out == out else default
            except (TypeError, ValueError):
                continue
    return default


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None or int(denominator) <= 0:
        return None
    return float(int(numerator) / int(denominator))


def _same_text_or_none(left: str | None, right: str | None) -> bool | None:
    if left is None or right is None:
        return None
    return left == right


def _nonempty_texts(*values: Any) -> list[str]:
    return [str(value) for value in values if isinstance(value, str) and value]


def _retention_floor(*nodes: Mapping[str, Any]) -> float:
    value = _first_float(*nodes, keys=("scorer_effect_retention_floor",))
    return 0.5 if value is None else float(value)


def _dedupe(values: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


__all__ = [
    "FIRST_DIVERGENCES",
    "HI_NERV_LIVE_TO_PARSEBACK_DELTA_AUDIT_SCHEMA",
    "build_hinerv_live_to_parseback_scorer_effect_delta_audit",
]
