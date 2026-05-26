# SPDX-License-Identifier: MIT
"""Canonical entropy-position classification for rate-attack planning.

The same byte delta means different things depending on where the transform
lands relative to entropy concentration.  These rows are planner features only:
they never grant score, dispatch, or promotion authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ENTROPY_POSITION_CLASSIFICATION_SCHEMA = "entropy_position_classification.v1"

BEFORE_ENTROPY_CODER = "before_entropy_coder_distribution_shaping"
AT_ENTROPY_CODER = "at_entropy_coder_symbol_coding"
AFTER_ENTROPY_CODER = "after_entropy_coder_container_runtime_overhead"
META_ENTROPY_POSITION = "meta_position_aggregate_planning"
UNKNOWN_ENTROPY_POSITION = "unknown_entropy_position_requires_explicit_context"


@dataclass(frozen=True)
class EntropyPosition:
    position_id: str
    position_name: str
    phase: str
    position_class: str
    composition_key: str
    information_effect: str
    composition_rule: str
    downstream_coder_rerun_recommended: bool

    def to_dict(self, *, target_kind: str, operation_family: str | None) -> dict[str, Any]:
        return {
            "schema": ENTROPY_POSITION_CLASSIFICATION_SCHEMA,
            "target_kind": target_kind,
            "operation_family": operation_family,
            "entropy_position_id": self.position_id,
            "entropy_position_name": self.position_name,
            "entropy_phase": self.phase,
            "entropy_position_class": self.position_class,
            "entropy_position_composition_key": self.composition_key,
            "entropy_information_effect": self.information_effect,
            "entropy_position_composition_rule": self.composition_rule,
            "downstream_entropy_coder_rerun_recommended": (
                self.downstream_coder_rerun_recommended
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }


_POSITIONS: dict[str, EntropyPosition] = {
    "P2": EntropyPosition(
        "P2",
        "loss_shape",
        "train",
        BEFORE_ENTROPY_CODER,
        "P2_loss_shape",
        "changes_trained_signal_distribution",
        "upstream_cascade_enables_downstream_codecs",
        True,
    ),
    "P3": EntropyPosition(
        "P3",
        "optimizer",
        "train",
        BEFORE_ENTROPY_CODER,
        "P3_optimizer",
        "changes_trained_weight_and_residual_distribution",
        "upstream_cascade_enables_downstream_codecs",
        True,
    ),
    "P4": EntropyPosition(
        "P4",
        "qat_noise",
        "train",
        BEFORE_ENTROPY_CODER,
        "P4_qat_noise",
        "changes_quantized_symbol_distribution",
        "upstream_cascade_enables_downstream_codecs",
        True,
    ),
    "P5": EntropyPosition(
        "P5",
        "substrate_renderer",
        "archive",
        BEFORE_ENTROPY_CODER,
        "P5_substrate_renderer",
        "changes_receiver_reconstructable_information",
        "position_disjoint_effects_compose_additively_until_empirical_anchor",
        True,
    ),
    "P8": EntropyPosition(
        "P8",
        "per_tensor_weight_archive",
        "archive",
        BEFORE_ENTROPY_CODER,
        "P8_tensor_weight_archive",
        "changes_weight_symbols_before_container_codec",
        "upstream_cascade_enables_downstream_codecs",
        True,
    ),
    "P9": EntropyPosition(
        "P9",
        "codebook_entropy",
        "archive",
        AT_ENTROPY_CODER,
        "P9_codebook_entropy",
        "recodes_codebook_or_index_symbols",
        "same_position_transforms_are_subadditive",
        False,
    ),
    "P11": EntropyPosition(
        "P11",
        "selector_stream",
        "archive",
        AT_ENTROPY_CODER,
        "P11_selector_stream",
        "recodes_selector_or_mode_stream_symbols",
        "same_position_transforms_are_subadditive",
        False,
    ),
    "P13": EntropyPosition(
        "P13",
        "first_order_markov_context",
        "codec",
        AT_ENTROPY_CODER,
        "P13_markov_context",
        "uses_context_model_for_symbol_code_lengths",
        "same_position_transforms_are_subadditive",
        False,
    ),
    "P14": EntropyPosition(
        "P14",
        "higher_order_context",
        "codec",
        AT_ENTROPY_CODER,
        "P14_higher_order_context",
        "uses_higher_order_context_model_for_symbol_code_lengths",
        "same_position_transforms_are_subadditive",
        False,
    ),
    "P15": EntropyPosition(
        "P15",
        "outer_codec_wrap",
        "codec",
        AFTER_ENTROPY_CODER,
        "P15_outer_codec_wrap",
        "changes_container_or_generic_codec_overhead",
        "cannot_extract_entropy_already_concentrated_upstream",
        False,
    ),
    "P16": EntropyPosition(
        "P16",
        "runtime_payload_and_inflate",
        "runtime",
        AFTER_ENTROPY_CODER,
        "P16_runtime_payload",
        "changes_runtime_or_receiver_payload_overhead",
        "can_save_overhead_but_requires_runtime_consumption_proof",
        False,
    ),
    "P17": EntropyPosition(
        "P17",
        "frame_pair_post_decode",
        "runtime",
        BEFORE_ENTROPY_CODER,
        "P17_frame_pair_post_decode",
        "changes_scorer_consumed_frames_after_decode",
        "position_disjoint_effects_compose_additively_until_empirical_anchor",
        True,
    ),
    "P18": EntropyPosition(
        "P18",
        "segnet_entropy",
        "scorer",
        BEFORE_ENTROPY_CODER,
        "P18_segnet_entropy",
        "changes_segnet_sufficient_statistics_or_boundaries",
        "inverse_steganalysis_waterfill_position",
        True,
    ),
    "P19": EntropyPosition(
        "P19",
        "posenet_entropy",
        "scorer",
        BEFORE_ENTROPY_CODER,
        "P19_posenet_entropy",
        "changes_posenet_sufficient_statistics_or_pose_stability",
        "inverse_steganalysis_waterfill_position",
        True,
    ),
    "P20": EntropyPosition(
        "P20",
        "aggregate_delta_s_shaping",
        "meta",
        META_ENTROPY_POSITION,
        "P20_meta_lagrangian",
        "planner_only_aggregate_tradeoff_accounting",
        "not_a_materializer_position",
        False,
    ),
}

_TARGET_KIND_POSITION_OVERRIDES: dict[str, str] = {
    "archive_section_entropy_recode_v1": "P9",
    "archive_section_header_elide_v1": "P16",
    "archive_section_proceduralize_v1": "P5",
    "archive_section_reorder_v1": "P16",
    "byte_range_entropy_recode_v1": "P13",
    "dqs1_pairset_drop_pair": "P17",
    "frame_pair_geometry_transform_v1": "P17",
    "inverse_scorer_cell_candidate_v1": "P18",
    "inverse_steganalysis_high_level_operation_set_v1": "P20",
    "higher_order_markov_selector_recode_v1": "P14",
    "markov_context_selector_recode_v1": "P13",
    "packet_member_merge_v1": "P16",
    "packet_member_recompress_v1": "P15",
    "packet_member_reorder_v1": "P16",
    "packet_member_zip_header_elide_v1": "P16",
    "renderer_payload_dfl1_v1": "P16",
    "selector_stream_context_recode_v1": "P11",
    "tensor_factorize_v1": "P8",
    "tensor_prune_v1": "P8",
    "tensor_quantize_v1": "P8",
    "tensor_shared_codebook_v1": "P8",
}

_OPERATION_FAMILY_POSITION_OVERRIDES: dict[str, str] = {
    "second_order": "P14",
    "higher_order": "P14",
    "markov2": "P14",
    "huffman": "P13",
    "markov": "P13",
    "ans": "P13",
    "range": "P13",
    "arithmetic": "P13",
    "optimizer": "P3",
    "muon": "P3",
    "adamw": "P3",
    "qat": "P4",
    "lsq": "P4",
    "segnet": "P18",
    "posenet": "P19",
    "pose": "P19",
    "uniward": "P18",
    "fridrich": "P18",
}


def classify_entropy_position(
    target_kind: str | None,
    *,
    operation_family: str | None = None,
    payload_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a materializer/search row by entropy position.

    ``payload_context`` may supply an explicit ``entropy_position_id`` or
    ``position_id`` when the same target kind has multiple valid positions.
    Unknown rows fail closed as P20-style planning features.
    """

    normalized_target = str(target_kind or "").strip()
    normalized_family = str(operation_family or "").strip() or None
    context = payload_context if isinstance(payload_context, Mapping) else {}
    explicit_position = _explicit_position_id(context)
    family_position = _position_from_operation_family(normalized_family)
    position_id = (
        explicit_position
        or _position_from_context(context)
        or ("P14" if family_position == "P14" else None)
        or _TARGET_KIND_POSITION_OVERRIDES.get(normalized_target)
        or family_position
    )
    if position_id in _POSITIONS:
        return _POSITIONS[position_id].to_dict(
            target_kind=normalized_target,
            operation_family=normalized_family,
        )
    return {
        "schema": ENTROPY_POSITION_CLASSIFICATION_SCHEMA,
        "target_kind": normalized_target,
        "operation_family": normalized_family,
        "entropy_position_id": "UNKNOWN",
        "entropy_position_name": UNKNOWN_ENTROPY_POSITION,
        "entropy_phase": "unknown",
        "entropy_position_class": UNKNOWN_ENTROPY_POSITION,
        "entropy_position_composition_key": "UNKNOWN_requires_explicit_context",
        "entropy_information_effect": "unknown_requires_empirical_anchor",
        "entropy_position_composition_rule": "do_not_apply_position_bonus",
        "downstream_entropy_coder_rerun_recommended": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def entropy_position_fields_for_row(row: Mapping[str, Any]) -> dict[str, Any]:
    target_kind = str(row.get("target_kind") or "").strip()
    operation_family = row.get("operation_family") or row.get("operation_id")
    return classify_entropy_position(
        target_kind,
        operation_family=str(operation_family) if operation_family else None,
        payload_context=row,
    )


def _explicit_position_id(context: Mapping[str, Any]) -> str | None:
    for key in ("entropy_position_id", "position_id", "pipeline_position_id"):
        value = str(context.get(key) or "").strip().upper()
        if value in _POSITIONS:
            return value
    return None


def _position_from_context(context: Mapping[str, Any]) -> str | None:
    scorer_component = str(context.get("scorer_component") or "").lower()
    if "pose" in scorer_component:
        return "P19"
    if "seg" in scorer_component or "mask" in scorer_component:
        return "P18"
    position_ids: list[str] = []
    for item in context.get("targeted_positions") or []:
        if isinstance(item, Mapping):
            value = str(item.get("position_id") or "").strip().upper()
        else:
            value = str(item or "").strip().upper()
        if value in _POSITIONS:
            position_ids.append(value)
    if "P19" in position_ids:
        return "P19"
    if "P18" in position_ids:
        return "P18"
    return position_ids[0] if position_ids else None


def _position_from_operation_family(operation_family: str | None) -> str | None:
    text = str(operation_family or "").lower()
    for needle, position_id in _OPERATION_FAMILY_POSITION_OVERRIDES.items():
        if needle in text:
            return position_id
    return None


__all__ = [
    "AFTER_ENTROPY_CODER",
    "AT_ENTROPY_CODER",
    "BEFORE_ENTROPY_CODER",
    "ENTROPY_POSITION_CLASSIFICATION_SCHEMA",
    "META_ENTROPY_POSITION",
    "UNKNOWN_ENTROPY_POSITION",
    "classify_entropy_position",
    "entropy_position_fields_for_row",
]
