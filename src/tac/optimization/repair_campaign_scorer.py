# SPDX-License-Identifier: MIT
"""Score repair-waterfill typed ledgers for local campaign execution.

The scorer is intentionally false-authority: it ranks encoder-side repair
work for bounded local MLX follow-up, and it names the exact custody artifacts
missing before any budget spend, promotion, or exact dispatch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

from tac.fec6_selector_operator_space import FEC6_FIXED_K16_MODE_IDS
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)

REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA = "repair_campaign_score_report.v1"
REPAIR_CAMPAIGN_SCORE_ROW_SCHEMA = "repair_campaign_score_row.v1"
REPAIR_CAMPAIGN_OPTIMIZER_DECISION_SCHEMA = "repair_campaign_optimizer_decision.v1"
REPAIR_CAMPAIGN_OPTIMIZER_ALLOCATION_ROW_SCHEMA = (
    "repair_campaign_optimizer_allocation_row.v1"
)
REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA = "repair_campaign_stackability_probe.v1"
REPAIR_OPERATOR_FAMILY_PRIORS_SCHEMA = "repair_operator_family_priors.v1"
REPAIR_OPERATOR_FAMILY_PRIOR_ROW_SCHEMA = "repair_operator_family_prior_row.v1"
REPAIR_CAMPAIGN_POSTERIOR_PRIOR_SUMMARY_SCHEMA = (
    "repair_campaign_posterior_prior_summary.v1"
)
REPAIR_CAMPAIGN_POSTERIOR_FAMILY_PRIOR_SCHEMA = (
    "repair_campaign_posterior_family_prior.v1"
)
REPAIR_CAMPAIGN_MULTISCALE_ACTION_LEDGER_SCHEMA = (
    "repair_campaign_multiscale_action_ledger.v1"
)
REPAIR_CAMPAIGN_MULTISCALE_ACTION_ROW_SCHEMA = (
    "repair_campaign_multiscale_action_row.v1"
)

_TYPED_LEDGER_SCHEMA = "frontier_rate_attack_repair_budget_typed_response_ledger.v1"
_TYPED_ROW_SCHEMA = "frontier_rate_attack_repair_budget_typed_response_row.v1"
_WORK_ORDER_SCHEMA = "frontier_rate_attack_repair_budget_waterfill_work_order.v1"
_REPAIR_CASCADE_OPPORTUNITY_ROW_SCHEMA = (
    "frontier_rate_attack_repair_cascade_opportunity_row.v1"
)
_REPAIR_DYNAMICS_PALETTE_PRIOR_SCHEMA = (
    "frontier_rate_attack_repair_dynamics_palette_prior.v1"
)

_ENTROPY_POSITION_WEIGHTS: dict[str, float] = {
    "before_entropy_coder_distribution_shaping": 1.20,
    "scorer_entropy_repair_before_selector_codec": 1.15,
    "at_entropy_coder_integer_codeword_boundary": 1.05,
    "at_entropy_coder": 1.00,
    "selector_codec_entropy": 0.85,
    "after_entropy_coder_container_or_zip_grammar": 0.45,
    "unknown_entropy_pipeline_position": 0.20,
}

_SCALE_ORDER: tuple[str, ...] = (
    "bit",
    "byte",
    "pixel",
    "boundary",
    "region",
    "frame",
    "pair",
    "batch",
    "full_video",
)
_NON_SIGNAL_SCOPE_KEYS = frozenset(FALSE_AUTHORITY)


class RepairCampaignScorerError(ValueError):
    """Raised when a repair campaign payload cannot be scored."""


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _safe_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _safe_float(value)
        if parsed is not None:
            return parsed
    return None


def _repo_path_exists(path_text: str, repo_root: str | Path | None) -> bool:
    if not path_text:
        return False
    path = Path(path_text)
    if not path.is_absolute() and repo_root is not None:
        path = Path(repo_root) / path
    return path.exists()


def _palette_mode_frame_index(mode: str) -> int | None:
    if not mode.startswith("frame"):
        return None
    suffix = mode[len("frame") :]
    number = suffix.split("_", 1)[0]
    return int(number) if number.isdigit() else None


def _palette_mode_family(mode: str) -> str:
    if mode == "none":
        return "identity"
    if "blue_chroma" in mode:
        return "blue_chroma"
    if "luma_bias" in mode:
        return "luma_bias"
    if "rgb_bias" in mode:
        return "rgb_bias"
    if "roll_" in mode:
        return "geometry_roll"
    return "other"


def _palette_dynamics_from_modes(
    modes: Sequence[str],
    *,
    source: str,
) -> dict[str, Any]:
    palette_modes = ordered_unique(_string_list(modes))
    frame_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for mode in palette_modes:
        frame_index = _palette_mode_frame_index(mode)
        frame_key = f"frame{frame_index}" if frame_index is not None else "no_frame"
        frame_counts[frame_key] = frame_counts.get(frame_key, 0) + 1
        family = _palette_mode_family(mode)
        family_counts[family] = family_counts.get(family, 0) + 1
    total = len(palette_modes)
    identity_count = family_counts.get("identity", 0)
    non_identity_total = max(0, total - identity_count)
    frame0_count = frame_counts.get("frame0", 0)
    frame1_count = frame_counts.get("frame1", 0)
    return {
        "schema": _REPAIR_DYNAMICS_PALETTE_PRIOR_SCHEMA,
        "source": source,
        "palette_modes": palette_modes,
        "mode_count": total,
        "identity_mode_count": identity_count,
        "non_identity_mode_count": non_identity_total,
        "frame_mode_counts": dict(sorted(frame_counts.items())),
        "mode_family_counts": dict(sorted(family_counts.items())),
        "frame0_mode_count": frame0_count,
        "frame1_mode_count": frame1_count,
        "frame0_mode_fraction": frame0_count / total if total else 0.0,
        "frame0_non_identity_fraction": (
            frame0_count / non_identity_total if non_identity_total else 0.0
        ),
        "zero_frame1_modes": frame1_count == 0,
        "dominant_dynamics_interpretation": (
            "frame0_global_color_geometry_calibration_prior"
            if frame0_count and frame1_count == 0
            else "mixed_or_unclassified_palette_prior"
        ),
        "repair_waterfill_hints": ordered_unique(
            [
                *(
                    ["frame0_palette_modes_are_first_class_repair_operators"]
                    if frame0_count
                    else []
                ),
                *(
                    ["empirical_non_identity_palette_is_all_frame0"]
                    if non_identity_total and frame0_count == non_identity_total
                    else []
                ),
                *(
                    ["do_not_assume_frame1_direct_repair_mode_exists"]
                    if frame1_count == 0
                    else []
                ),
            ]
        ),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_campaign_palette_dynamics_prior_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _canonical_k16_palette_dynamics() -> dict[str, Any]:
    return _palette_dynamics_from_modes(
        FEC6_FIXED_K16_MODE_IDS,
        source="live_6bae0201_archive_manifest_fec6_fixed_k16_palette",
    )


def _repair_dynamics_palette_context(
    row: Mapping[str, Any],
    prior: Mapping[str, Any],
) -> dict[str, Any]:
    value = row.get("repair_dynamics_palette_prior")
    if isinstance(value, Mapping) and value:
        require_no_truthy_authority_fields(
            value,
            context="repair_campaign_row_repair_dynamics_palette_prior",
        )
        modes = _string_list(value.get("palette_modes"))
        context = (
            _palette_dynamics_from_modes(
                modes,
                source=str(value.get("source") or "typed_response_palette_prior"),
            )
            if modes
            else dict(value)
        )
        context["source_prior_schema"] = value.get("schema")
        context["explicit_row_prior"] = True
    elif prior.get("family_id") == "palette_frame_asymmetry_prior":
        context = dict(_mapping(prior.get("empirical_canonical_palette")))
        context["explicit_row_prior"] = False
    else:
        return {}
    context.setdefault("schema", _REPAIR_DYNAMICS_PALETTE_PRIOR_SCHEMA)
    context.setdefault("mode_count", _safe_int(context.get("mode_count")))
    context.setdefault("frame0_mode_count", _safe_int(context.get("frame0_mode_count")))
    context.setdefault("frame1_mode_count", _safe_int(context.get("frame1_mode_count")))
    context.setdefault(
        "zero_frame1_modes",
        context.get("frame1_mode_count") == 0,
    )
    context.update(
        {
            "action_functional_role": (
                "global_frame0_color_geometry_interaction_prior"
            ),
            "required_remeasurement": (
                "same_axis_stackability_probe_before_budget_spend"
            ),
            "allowed_use": "repair_campaign_palette_context_for_local_ranking_only",
            "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
            **FALSE_AUTHORITY,
        }
    )
    require_no_truthy_authority_fields(
        context,
        context="repair_campaign_palette_dynamics_context",
    )
    return context


def _palette_frame_asymmetry_multiplier(context: Mapping[str, Any]) -> float:
    if not context:
        return 1.0
    mode_count = _safe_int(context.get("mode_count"))
    frame0_fraction = _safe_float(context.get("frame0_non_identity_fraction"))
    if frame0_fraction is None:
        frame0_fraction = _safe_float(context.get("frame0_mode_fraction")) or 0.0
    multiplier = 1.0
    multiplier += min(0.04, mode_count / 400.0)
    multiplier += 0.04 * max(0.0, min(1.0, frame0_fraction))
    if context.get("zero_frame1_modes") is True:
        multiplier += 0.025
    return max(1.0, min(1.12, multiplier))


def repair_operator_family_priors() -> dict[str, Any]:
    """Return first-class repair families the scorer understands."""

    canonical_palette = _canonical_k16_palette_dynamics()
    rows = [
        {
            "schema": REPAIR_OPERATOR_FAMILY_PRIOR_ROW_SCHEMA,
            "family_id": "posenet_null_bottom_decile",
            "aliases": [
                "posenet_null_bottom_decile",
                "posenet-bottom-decile",
                "P19",
            ],
            "targeted_dimensions": ["posenet", "pair", "frame0"],
            "entropy_position_label": "before_entropy_coder_distribution_shaping",
            "campaign_prior_multiplier": 1.18,
            "required_local_artifacts": [
                "local_mlx_response_path",
                "reference_local_mlx_response_path",
                "posenet_null_bottom_decile_pair_ids",
            ],
            "missing_artifact_label": "posenet_null_bottom_decile_mlx_probe_missing",
            "stackability_role": "pose_repair_in_segnet_nullspace",
            **FALSE_AUTHORITY,
        },
        {
            "schema": REPAIR_OPERATOR_FAMILY_PRIOR_ROW_SCHEMA,
            "family_id": "segnet_class_region_waterfill",
            "aliases": [
                "segnet_class_region_waterfill",
                "segnet-class-region",
                "P18",
            ],
            "targeted_dimensions": ["segnet", "region", "frame1"],
            "entropy_position_label": "before_entropy_coder_distribution_shaping",
            "campaign_prior_multiplier": 1.22,
            "required_local_artifacts": [
                "local_mlx_response_path",
                "reference_local_mlx_response_path",
                "segnet_class_region_mask_ids",
            ],
            "missing_artifact_label": "segnet_class_region_mlx_probe_missing",
            "stackability_role": "segnet_margin_waterfill_before_selector_codec",
            **FALSE_AUTHORITY,
        },
        {
            "schema": REPAIR_OPERATOR_FAMILY_PRIOR_ROW_SCHEMA,
            "family_id": "per_region_selector_codec",
            "aliases": ["per_region_selector_codec", "per-region-selector", "P11"],
            "targeted_dimensions": ["selector_stream", "region"],
            "entropy_position_label": "selector_codec_entropy",
            "campaign_prior_multiplier": 0.55,
            "required_local_artifacts": [
                "selector_payload_bits_per_region",
                "receiver_consumed_runtime_replay_proof",
            ],
            "missing_artifact_label": "per_region_selector_codec_replay_missing",
            "stackability_role": "selector_payload_for_measured_region_decisions",
            **FALSE_AUTHORITY,
        },
        {
            "schema": REPAIR_OPERATOR_FAMILY_PRIOR_ROW_SCHEMA,
            "family_id": "palette_frame_asymmetry_prior",
            "aliases": [
                "palette_frame_asymmetry_prior",
                "repair_dynamics_frame0_palette_interaction_waterfill",
                "frame0_palette",
            ],
            "targeted_dimensions": ["palette", "frame0", "posenet"],
            "entropy_position_label": "before_entropy_coder_distribution_shaping",
            "campaign_prior_multiplier": 1.10,
            "required_local_artifacts": [
                "local_mlx_response_path",
                "reference_local_mlx_response_path",
                "repair_dynamics_palette_probe_matrix_path",
            ],
            "missing_artifact_label": "palette_frame_asymmetry_probe_missing",
            "stackability_role": "frame0_pose_repair_using_canonical_k16_palette",
            "empirical_canonical_palette": canonical_palette,
            "mathematical_prior": {
                "schema": "repair_campaign_palette_asymmetry_mathematical_prior.v1",
                "observation": (
                    "canonical K16 palette has one identity mode, fifteen frame0 "
                    "non-identity modes, and zero frame1 modes"
                ),
                "action_functional_effect": (
                    "treat frame0 palette operators as global interaction terms "
                    "spanning pixel, boundary, region, frame, pair, batch, and "
                    "scorer axes"
                ),
                "hard_constraint": (
                    "frame1 repair modes require counterfactual measurement before "
                    "allocation because no live frame1 palette mode exists"
                ),
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
                **FALSE_AUTHORITY,
            },
            **FALSE_AUTHORITY,
        },
        {
            "schema": REPAIR_OPERATOR_FAMILY_PRIOR_ROW_SCHEMA,
            "family_id": "entropy_position_cascade",
            "aliases": ["entropy_position_cascade", "cascade_c", "Cascade C"],
            "targeted_dimensions": ["segnet", "posenet", "selector_stream"],
            "entropy_position_label": "scorer_entropy_repair_before_selector_codec",
            "campaign_prior_multiplier": 1.12,
            "required_local_artifacts": [
                "posenet_null_bottom_decile_pair_ids",
                "segnet_class_region_mask_ids",
                "selector_payload_bits_per_region",
            ],
            "missing_artifact_label": "entropy_position_cascade_probe_missing",
            "stackability_role": "compose_scorer_entropy_repair_before_codec_replay",
            **FALSE_AUTHORITY,
        },
    ]
    payload = {
        "schema": REPAIR_OPERATOR_FAMILY_PRIORS_SCHEMA,
        "row_schema": REPAIR_OPERATOR_FAMILY_PRIOR_ROW_SCHEMA,
        "row_count": len(rows),
        "rows": rows,
        "allowed_use": "repair_campaign_scoring_prior_only",
        "forbidden_use": "score_claim_or_dispatch_or_budget_spend_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(payload, context="repair_operator_family_priors")
    return payload


def _family_prior(row: Mapping[str, Any]) -> Mapping[str, Any]:
    corpus = " ".join(
        [
            str(row.get("correction_family") or ""),
            str(row.get("candidate_id") or ""),
            " ".join(_string_list(row.get("targeted_dimensions"))),
            " ".join(_string_list(row.get("operation_levels"))),
        ]
    ).lower()
    for prior in repair_operator_family_priors()["rows"]:
        aliases = [alias.lower() for alias in _string_list(prior.get("aliases"))]
        if any(alias and alias in corpus for alias in aliases):
            return prior
    return {}


def _objective_delta(row: Mapping[str, Any]) -> float | None:
    direct = _safe_float(row.get("objective_delta_score_units"))
    if direct is not None:
        return direct
    curves = _mapping(row.get("marginal_response_curves"))
    objective = _mapping(curves.get("objective"))
    from_curve = _safe_float(objective.get("delta_score_units"))
    if from_curve is not None:
        return from_curve
    return _safe_float(row.get("measured_lagrangian_delta_score_units"))


def _requested_bytes(row: Mapping[str, Any]) -> int:
    requested = _safe_int(row.get("requested_repair_bytes"))
    if requested > 0:
        return requested
    curves = _mapping(row.get("marginal_response_curves"))
    return max(0, _safe_int(curves.get("requested_repair_bytes")))


def _entropy_position(row: Mapping[str, Any], prior: Mapping[str, Any]) -> str:
    return str(
        row.get("entropy_position_label")
        or prior.get("entropy_position_label")
        or "unknown_entropy_pipeline_position"
    )


def _interaction_penalty(row: Mapping[str, Any]) -> float:
    scope = _mapping(row.get("interaction_scope"))
    stack_terms = _mapping(row.get("stacking_interaction_terms"))
    penalty = 0.0
    if stack_terms.get("must_remeasure_with_parent_and_sibling_repairs") is True:
        penalty += 0.12
    if scope.get("pair_indices") or scope.get("region_ids") or scope.get("mode_ids"):
        penalty += 0.05
    if _safe_int(scope.get("pair_count")) > 128:
        penalty += 0.04
    if _safe_int(scope.get("region_count")) > 8:
        penalty += 0.04
    return min(penalty, 0.35)


def _allocation_action_term(row: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("allocation_action_term", "operator_action_term", "action_term"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _per_op_bytes_delta(row: Mapping[str, Any]) -> int | None:
    term = _allocation_action_term(row)
    transform = _mapping(term.get("T_i"))
    delta = _first_float(
        row.get("archive_byte_delta_vs_baseline"),
        row.get("receiver_closed_archive_byte_delta_vs_reference"),
        row.get("selector_payload_wire_delta_bytes"),
        transform.get("archive_byte_delta_vs_baseline"),
        transform.get("receiver_closed_archive_byte_delta_vs_reference"),
        transform.get("bytes_delta"),
    )
    if delta is None:
        return None
    return int(delta)


def _component_delta(row: Mapping[str, Any], component: str) -> float | None:
    component_terms = (
        _mapping(row.get("local_mlx_component_terms"))
        or _mapping(row.get("local_cpu_component_terms"))
        or _mapping(row.get("component_terms"))
    )
    curves = _mapping(row.get("marginal_response_curves"))
    component_curve = _mapping(curves.get(component))
    return _first_float(
        row.get(f"{component}_delta_score_units"),
        row.get(f"measured_{component}_delta_score_units"),
        component_terms.get(f"{component}_delta_score_units"),
        component_terms.get("delta_score_units"),
        component_curve.get("delta_score_units"),
    )


def _component_response_terms(row: Mapping[str, Any]) -> dict[str, Any]:
    segnet_delta = _component_delta(row, "segnet")
    posenet_delta = _component_delta(row, "posenet")
    measured_component = _first_float(
        row.get("measured_component_delta_score_units"),
        row.get("component_delta_score_units"),
    )
    return {
        "schema": "repair_campaign_component_response_terms.v1",
        "segnet_delta_score_units": segnet_delta,
        "posenet_delta_score_units": posenet_delta,
        "measured_component_delta_score_units": measured_component,
        "response_axis": (
            "[macOS-MLX research-signal]"
            if row.get("local_mlx_response_path")
            else "unknown_or_unmeasured_component_response_axis"
        ),
        "allowed_use": "repair_campaign_component_response_ranking_feature",
        "forbidden_use": "score_claim_or_budget_spend_authority",
        **FALSE_AUTHORITY,
    }


def _entropy_position_class(entropy_position: str) -> str:
    if entropy_position.startswith("before_entropy_coder"):
        return "pre_entropy_distribution_shaping"
    if entropy_position == "scorer_entropy_repair_before_selector_codec":
        return "scorer_entropy_before_selector_codec"
    if entropy_position.startswith("at_entropy_coder"):
        return "entropy_coder_boundary"
    if entropy_position.startswith("after_entropy_coder"):
        return "post_entropy_container"
    if entropy_position == "selector_codec_entropy":
        return "selector_codec_entropy"
    return "unknown_entropy_pipeline_position"


def _ordered_scales(values: Sequence[str]) -> list[str]:
    seen = ordered_unique(values)
    known = [scale for scale in _SCALE_ORDER if scale in set(seen)]
    extra = [scale for scale in seen if scale not in set(_SCALE_ORDER)]
    return [*known, *extra]


def _has_any_key(mapping: Mapping[str, Any], fragments: Sequence[str]) -> bool:
    lowered = [
        str(key).lower()
        for key in mapping
        if str(key) not in _NON_SIGNAL_SCOPE_KEYS
    ]
    return any(any(fragment in key for fragment in fragments) for key in lowered)


def _multiscale_action_row(
    *,
    source_row: Mapping[str, Any],
    prior: Mapping[str, Any],
    entropy_position: str,
    requested_bytes: int,
    objective_delta: float | None,
    per_op_bytes_delta: int | None,
    component_response_terms: Mapping[str, Any],
    interaction_penalty: float,
    hard_constraints: Sequence[str],
    palette_dynamics_context: Mapping[str, Any],
) -> dict[str, Any]:
    scope = _mapping(source_row.get("interaction_scope"))
    stack_terms = _mapping(source_row.get("stacking_interaction_terms"))
    term = _allocation_action_term(source_row)
    transform = _mapping(term.get("T_i"))
    operation_levels = _string_list(source_row.get("operation_levels"))
    targeted_dimensions = _string_list(source_row.get("targeted_dimensions"))
    corpus = " ".join(
        [
            *operation_levels,
            *targeted_dimensions,
            str(source_row.get("correction_family") or ""),
            str(source_row.get("candidate_id") or ""),
            str(prior.get("family_id") or ""),
        ]
    ).lower()

    scale_reasons: dict[str, list[str]] = {scale: [] for scale in _SCALE_ORDER}

    def add(scale: str, reason: str) -> None:
        scale_reasons.setdefault(scale, []).append(reason)

    bit_delta = _first_float(
        source_row.get("compressed_bit_delta_vs_baseline"),
        source_row.get("selector_payload_bit_delta_vs_baseline"),
        source_row.get("bit_delta_vs_baseline"),
        transform.get("compressed_bit_delta_vs_baseline"),
        transform.get("selector_payload_bit_delta_vs_baseline"),
        transform.get("bit_delta_vs_baseline"),
        transform.get("bits_delta"),
    )
    if bit_delta is not None or "bit" in corpus or _has_any_key(
        scope, ("bit", "bits")
    ):
        add("bit", "bit_delta_or_bit_level_scope")
    if requested_bytes > 0 or per_op_bytes_delta is not None or "byte" in corpus:
        add("byte", "repair_byte_budget_or_archive_byte_delta")
    if "pixel" in corpus or _has_any_key(scope, ("pixel", "mask")):
        add("pixel", "pixel_or_mask_scope")
    if (
        "boundary" in corpus
        or "segnet" in corpus
        or _has_any_key(scope, ("boundary", "edge"))
    ):
        add("boundary", "segnet_boundary_or_explicit_boundary_scope")
    if "region" in corpus or _has_any_key(scope, ("region", "class_region")):
        add("region", "region_scope")
    if "frame" in corpus or _has_any_key(scope, ("frame",)):
        add("frame", "frame_scope")
    if "pair" in corpus or _has_any_key(scope, ("pair",)):
        add("pair", "pair_scope")
    if "batch" in corpus or _has_any_key(scope, ("batch",)):
        add("batch", "batch_scope")
    if (
        "full_video" in corpus
        or "video" in corpus
        or _has_any_key(scope, ("full_video", "video"))
    ):
        add("full_video", "full_video_scope")

    active_scales = _ordered_scales(
        [scale for scale, reasons in scale_reasons.items() if reasons]
    )
    scale_rows = [
        {
            "scale": scale,
            "source_reasons": ordered_unique(scale_reasons.get(scale) or []),
        }
        for scale in active_scales
    ]
    segnet_delta = _safe_float(component_response_terms.get("segnet_delta_score_units"))
    posenet_delta = _safe_float(
        component_response_terms.get("posenet_delta_score_units")
    )
    component_axis_values: list[str] = []
    if segnet_delta is not None:
        component_axis_values.append("segnet")
    if posenet_delta is not None:
        component_axis_values.append("posenet")
    if per_op_bytes_delta is not None or requested_bytes:
        component_axis_values.append("rate_bytes")
    if bit_delta is not None:
        component_axis_values.append("selector_bits")
    component_axes = ordered_unique(component_axis_values)
    interaction_order = len(active_scales)
    remeasure_required = (
        stack_terms.get("must_remeasure_with_parent_and_sibling_repairs") is True
        or interaction_order >= 4
    )
    row = {
        "schema": REPAIR_CAMPAIGN_MULTISCALE_ACTION_ROW_SCHEMA,
        "typed_response_id": source_row.get("typed_response_id"),
        "candidate_id": source_row.get("candidate_id"),
        "family_id": prior.get("family_id") or "unclassified_repair_family",
        "correction_family": source_row.get("correction_family"),
        "active_scales": active_scales,
        "scale_rows": scale_rows,
        "interaction_order": interaction_order,
        "component_axes": component_axes,
        "entropy_position_label": entropy_position,
        "entropy_position_class": _entropy_position_class(entropy_position),
        "action_functional": {
            "schema": "repair_campaign_multiscale_action_functional_terms.v1",
            "objective": "minimize_delta_segnet_plus_delta_posenet_plus_lambda_delta_bytes",
            "delta_segnet_score_units": segnet_delta,
            "delta_posenet_score_units": posenet_delta,
            "objective_delta_score_units": objective_delta,
            "per_op_bytes_delta": per_op_bytes_delta,
            "requested_repair_bytes": requested_bytes,
            "bit_delta_vs_baseline": bit_delta,
            "interaction_penalty": interaction_penalty,
            "entropy_position_weight": _ENTROPY_POSITION_WEIGHTS.get(
                entropy_position,
                _ENTROPY_POSITION_WEIGHTS["unknown_entropy_pipeline_position"],
            ),
            "palette_frame_asymmetry_multiplier": _palette_frame_asymmetry_multiplier(
                palette_dynamics_context
            ),
        },
        "palette_dynamics_context": dict(palette_dynamics_context),
        "interaction_scope": dict(scope),
        "stacking_interaction_terms": dict(stack_terms),
        "remeasure_required_before_budget_spend": remeasure_required,
        "hard_legal_runtime_constraints": list(hard_constraints),
        "mathematical_grounding": {
            "schema": "repair_campaign_multiscale_mathematical_grounding.v1",
            "scale_order": list(_SCALE_ORDER),
            "principle": (
                "repair value depends on where a transformation acts relative "
                "to entropy concentration and which spatial/temporal support it touches"
            ),
            "objective": (
                "DeltaS = DeltaSegNet + DeltaPoseNet + lambda*DeltaBytes + "
                "cross_scale_interaction_terms"
            ),
            "hard_constraint": (
                "allocated repair bytes are bounded by receiver-closed rate credit"
            ),
        },
        "allowed_use": "multiscale_repair_action_ranking_feature_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        row,
        context=(
            "repair_campaign_multiscale_action_row:"
            f"{source_row.get('typed_response_id') or source_row.get('candidate_id')}"
        ),
    )
    return row


def _multiscale_action_ledger(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scale_histogram: dict[str, int] = {}
    entropy_class_histogram: dict[str, int] = {}
    interaction_order_histogram: dict[str, int] = {}
    action_rows: list[dict[str, Any]] = []
    for row in rows:
        action = _mapping(row.get("multiscale_action_row"))
        if not action:
            continue
        action_rows.append(dict(action))
        for scale in _string_list(action.get("active_scales")):
            scale_histogram[scale] = scale_histogram.get(scale, 0) + 1
        entropy_class = str(
            action.get("entropy_position_class")
            or "unknown_entropy_pipeline_position"
        )
        entropy_class_histogram[entropy_class] = (
            entropy_class_histogram.get(entropy_class, 0) + 1
        )
        order = str(_safe_int(action.get("interaction_order")))
        interaction_order_histogram[order] = interaction_order_histogram.get(order, 0) + 1
    ledger = {
        "schema": REPAIR_CAMPAIGN_MULTISCALE_ACTION_LEDGER_SCHEMA,
        "row_schema": REPAIR_CAMPAIGN_MULTISCALE_ACTION_ROW_SCHEMA,
        "row_count": len(action_rows),
        "scale_order": list(_SCALE_ORDER),
        "scale_histogram": dict(sorted(scale_histogram.items())),
        "entropy_position_class_histogram": dict(
            sorted(entropy_class_histogram.items())
        ),
        "interaction_order_histogram": dict(
            sorted(interaction_order_histogram.items())
        ),
        "rows": action_rows,
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_campaign_multiscale_action_ledger_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        ledger,
        context="repair_campaign_multiscale_action_ledger",
    )
    return ledger


def _load_posterior_rows(
    posterior_path: str | Path | None,
) -> list[dict[str, Any]]:
    if posterior_path is None:
        return []
    from tac.optimization.repair_campaign_posterior import (
        load_repair_campaign_stackability_posterior_rows,
    )

    return load_repair_campaign_stackability_posterior_rows(posterior_path)


def _posterior_prior_summary(
    *,
    posterior_path: str | Path | None,
    posterior_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in posterior_rows:
        if not isinstance(row, Mapping):
            continue
        family_id = str(row.get("family_id") or "unclassified_repair_family")
        grouped.setdefault(family_id, []).append(row)

    family_rows: list[dict[str, Any]] = []
    for family_id, rows in sorted(grouped.items()):
        increase_count = 0
        hold_count = 0
        blocked_count = 0
        improvement_per_byte_values: list[float] = []
        expected_improvement_values: list[float] = []
        missing_artifact_total = 0
        blocker_total = 0
        policy_counts: dict[str, int] = {}
        for row in rows:
            policy = str(
                _mapping(row.get("acquisition_policy_delta")).get(
                    "recommended_acquisition_policy"
                )
                or _mapping(row.get("local_planning_update")).get(
                    "recommended_acquisition_policy"
                )
                or ""
            )
            if policy:
                policy_counts[policy] = policy_counts.get(policy, 0) + 1
            direction = str(
                _mapping(row.get("acquisition_policy_delta")).get(
                    "family_priority_direction"
                )
                or ""
            )
            if direction == "increase":
                increase_count += 1
            else:
                hold_count += 1
            if str(row.get("evidence_grade") or "").startswith("blocked_"):
                blocked_count += 1
            feature_vector = _mapping(row.get("planner_feature_vector"))
            local_update = _mapping(row.get("local_planning_update"))
            if not feature_vector:
                feature_vector = _mapping(local_update.get("planner_feature_vector"))
            improvement_per_byte = _safe_float(
                feature_vector.get("improvement_per_allocated_byte")
            )
            if improvement_per_byte is not None and improvement_per_byte > 0.0:
                improvement_per_byte_values.append(improvement_per_byte)
            expected_improvement = _safe_float(
                feature_vector.get("expected_local_improvement_score_units")
            )
            if expected_improvement is not None and expected_improvement > 0.0:
                expected_improvement_values.append(expected_improvement)
            missing_artifact_total += _safe_int(
                feature_vector.get("missing_artifact_count")
            )
            blocker_total += _safe_int(feature_vector.get("blocker_count"))

        observation_count = len(rows)
        blocked_fraction = blocked_count / observation_count if observation_count else 0.0
        increase_fraction = (
            increase_count / observation_count if observation_count else 0.0
        )
        mean_improvement_per_byte = (
            sum(improvement_per_byte_values) / len(improvement_per_byte_values)
            if improvement_per_byte_values
            else 0.0
        )
        mean_expected_improvement = (
            sum(expected_improvement_values) / len(expected_improvement_values)
            if expected_improvement_values
            else 0.0
        )
        multiplier = 1.0 + (0.16 * increase_fraction) - (0.10 * blocked_fraction)
        if mean_improvement_per_byte > 0.0:
            multiplier += min(0.18, mean_improvement_per_byte * 4000.0)
        multiplier = max(0.70, min(1.35, multiplier))
        family_rows.append(
            {
                "schema": REPAIR_CAMPAIGN_POSTERIOR_FAMILY_PRIOR_SCHEMA,
                "family_id": family_id,
                "observation_count": observation_count,
                "increase_count": increase_count,
                "hold_count": hold_count,
                "blocked_count": blocked_count,
                "blocked_fraction": blocked_fraction,
                "increase_fraction": increase_fraction,
                "mean_improvement_per_allocated_byte": mean_improvement_per_byte,
                "mean_expected_local_improvement_score_units": (
                    mean_expected_improvement
                ),
                "missing_artifact_total": missing_artifact_total,
                "blocker_total": blocker_total,
                "policy_counts": dict(sorted(policy_counts.items())),
                "family_priority_multiplier": multiplier,
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
                "allowed_use": "posterior_repair_campaign_family_prior_only",
                "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
                **FALSE_AUTHORITY,
            }
        )

    summary = {
        "schema": REPAIR_CAMPAIGN_POSTERIOR_PRIOR_SUMMARY_SCHEMA,
        "posterior_path": str(posterior_path) if posterior_path is not None else None,
        "posterior_row_count": len(posterior_rows),
        "family_prior_count": len(family_rows),
        "family_priors": family_rows,
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_campaign_posterior_prior_for_local_scoring_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        summary,
        context="repair_campaign_posterior_prior_summary",
    )
    return summary


def _posterior_family_prior(
    summary: Mapping[str, Any],
    *,
    family_id: str,
) -> Mapping[str, Any]:
    for row in summary.get("family_priors") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("family_id") or "") == family_id:
            return row
    return {}


def _receiver_proof_status(
    row: Mapping[str, Any],
    *,
    repo_root: str | Path | None,
) -> dict[str, Any]:
    proof = _path_status(row, "runtime_consumption_proof_path", repo_root=repo_root)
    archive_key = (
        "receiver_consumed_candidate_archive_path"
        if row.get("receiver_consumed_candidate_archive_path")
        else "candidate_archive_path"
    )
    archive = _path_status(row, archive_key, repo_root=repo_root)
    replay = _path_status(
        row,
        "component_response_replay_manifest_path",
        repo_root=repo_root,
    )
    exact = _path_status(
        row,
        "exact_axis_component_response_path",
        repo_root=repo_root,
    )
    statuses = [proof, archive, replay, exact]
    missing = [
        f"{item['key']}:missing_or_unverified"
        for item in statuses
        if item["exists"] is not True
    ]
    return {
        "schema": "repair_campaign_receiver_proof_status.v1",
        "receiver_runtime_custody_ready": not missing,
        "runtime_consumption_proof": proof,
        "receiver_consumed_candidate_archive": archive,
        "component_response_replay_manifest": replay,
        "exact_axis_component_response": exact,
        "missing_artifacts": ordered_unique(missing),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_campaign_receiver_proof_custody_status_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _hard_legal_runtime_constraints(row: Mapping[str, Any]) -> list[str]:
    term = _allocation_action_term(row)
    return ordered_unique(
        [
            "allocated_bytes_must_not_exceed_receiver_closed_rate_credit",
            "local_mlx_response_is_planning_signal_only",
            "parent_rate_only_archive_must_materialize_first",
            "receiver_consumes_materialized_runtime_output",
            "component_response_replayed_before_budget_spend",
            "exact_auth_eval_required_before_score_or_promotion_claim",
            *_string_list(row.get("legal_runtime_constraints")),
            *_string_list(term.get("legal_runtime_constraints")),
        ]
    )


def _path_status(
    row: Mapping[str, Any],
    key: str,
    *,
    repo_root: str | Path | None,
) -> dict[str, Any]:
    text = str(row.get(key) or "").strip()
    return {
        "key": key,
        "path": text or None,
        "present": bool(text),
        "exists": _repo_path_exists(text, repo_root) if text else False,
    }


def _execution_gate(
    row: Mapping[str, Any],
    prior: Mapping[str, Any],
    *,
    repo_root: str | Path | None,
) -> dict[str, Any]:
    local_keys = ordered_unique(
        [
            "local_mlx_response_path",
            "reference_local_mlx_response_path",
            *[
                key
                for key in _string_list(prior.get("required_local_artifacts"))
                if key.endswith("_path")
            ],
        ]
    )
    local_status = [
        _path_status(row, key, repo_root=repo_root)
        for key in local_keys
    ]
    local_required = [item["key"] for item in local_status]
    local_ready = bool(local_required) and all(item["exists"] for item in local_status)
    missing = [
        f"{item['key']}:missing_or_unverified"
        for item in local_status
        if not item["exists"]
    ]
    exact_missing = [
        "receiver_consumed_candidate_archive",
        "runtime_consumption_proof_path",
        "component_response_replay_manifest",
        "exact_axis_component_response_artifact",
    ]
    if not local_ready:
        missing.extend(exact_missing)
    blocker_label = str(prior.get("missing_artifact_label") or "")
    if blocker_label and not local_ready:
        missing.append(blocker_label)
    missing.extend(_string_list(row.get("missing_artifacts")))
    palette_context = _repair_dynamics_palette_context(row, prior)
    if palette_context and not row.get("repair_dynamics_palette_probe_matrix_path"):
        local_ready = False
        missing.append("repair_dynamics_palette_probe_matrix_path:missing_or_unverified")
    return {
        "schema": "repair_campaign_execution_gate.v1",
        "local_mlx_advisory_custody_ready": local_ready,
        "local_mlx_custody_paths": local_status,
        "recommended_queue_status": (
            "ready_for_local_mlx_advisory_execution"
            if local_ready
            else "blocked_missing_artifact"
        ),
        "missing_artifacts": ordered_unique(missing),
        "exact_missing_artifacts_if_not_local": exact_missing if not local_ready else [],
        "palette_dynamics_context": dict(palette_context),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "local_repair_campaign_execution_gate_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


def _typed_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    schema = str(payload.get("schema") or "")
    if schema == _TYPED_LEDGER_SCHEMA:
        rows = payload.get("rows")
    elif schema == _WORK_ORDER_SCHEMA:
        ledger = payload.get("typed_response_ledger")
        if not isinstance(ledger, Mapping):
            if payload.get("repair_cascade_opportunity_rows"):
                return []
            raise RepairCampaignScorerError("work order missing typed_response_ledger")
        rows = ledger.get("rows")
    else:
        raise RepairCampaignScorerError(
            f"unsupported repair campaign scorer input schema: {schema or '<missing>'}"
        )
    return [row for row in rows or [] if isinstance(row, Mapping)]


def _cascade_opportunity_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(payload.get("schema") or "") != _WORK_ORDER_SCHEMA:
        return []
    rows: list[dict[str, Any]] = []
    for index, cascade in enumerate(
        payload.get("repair_cascade_opportunity_rows") or [],
        start=1,
    ):
        if not isinstance(cascade, Mapping):
            continue
        if cascade.get("schema") != _REPAIR_CASCADE_OPPORTUNITY_ROW_SCHEMA:
            continue
        cascade_id = str(cascade.get("cascade_id") or f"cascade_opportunity_{index}")
        targeted_positions = [
            item for item in cascade.get("targeted_positions") or [] if isinstance(item, Mapping)
        ]
        targeted_dimensions = ordered_unique(
            [
                "segnet",
                "posenet",
                "selector_stream",
                *[
                    str(item.get("entropy_surface") or "").strip()
                    for item in targeted_positions
                    if str(item.get("entropy_surface") or "").strip()
                ],
            ]
        )
        required_measurements = _string_list(cascade.get("required_probe_measurements"))
        blockers = ordered_unique(
            [
                *_string_list(cascade.get("blockers")),
                "structural_cascade_typed_component_response_missing",
                "local_mlx_structural_cascade_probe_missing",
            ]
        )
        rows.append(
            {
                "schema": _REPAIR_CASCADE_OPPORTUNITY_ROW_SCHEMA,
                "source_row_kind": "repair_cascade_opportunity",
                "typed_response_id": f"structural_repair_cascade:{cascade_id}",
                "candidate_id": cascade_id,
                "acquisition_id": cascade.get("next_queue_action"),
                "correction_family": "entropy_position_cascade",
                "family_id_hint": "entropy_position_cascade",
                "targeted_dimensions": targeted_dimensions,
                "operation_levels": [
                    "structural_repair_cascade",
                    "mlx_local_component_probe",
                    "selector_codec_stackability",
                ],
                "entropy_position_label": (
                    cascade.get("pipeline_position")
                    or "scorer_entropy_repair_before_selector_codec"
                ),
                "requested_repair_bytes": 0,
                "objective_delta_score_units": None,
                "cascade_id": cascade_id,
                "cascade_label": cascade.get("label"),
                "source_relation": cascade.get("source_relation"),
                "source_hint": cascade.get("source_hint"),
                "required_probe_measurements": required_measurements,
                "missing_artifacts": blockers,
                "blockers": blockers,
                "source_structural_opportunity": dict(cascade),
                "budget_spend_allowed": False,
                "ready_for_budget_spend": False,
                "ready_for_exact_eval_dispatch": False,
                "allowed_use": "structural_repair_cascade_scoring_input_only",
                "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
                **FALSE_AUTHORITY,
            }
        )
    return rows


def _receiver_closed_rate_credit_bytes(payload: Mapping[str, Any]) -> int:
    direct = _safe_int(payload.get("available_receiver_closed_rate_credit_bytes"))
    if direct > 0:
        return direct
    credit = _mapping(payload.get("receiver_closed_rate_credit"))
    from_credit = _safe_int(credit.get("receiver_closed_saved_bytes_total"))
    if from_credit > 0:
        return from_credit
    ledger = _mapping(payload.get("typed_response_ledger"))
    from_ledger = _safe_int(ledger.get("available_receiver_closed_rate_credit_bytes"))
    return max(0, from_ledger)


def _build_optimizer_decision(
    *,
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    posterior_summary: Mapping[str, Any],
) -> dict[str, Any]:
    available_bytes = _receiver_closed_rate_credit_bytes(payload)
    remaining = available_bytes
    selected_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    for row in rows:
        gate = _mapping(row.get("execution_gate"))
        ready = gate.get("recommended_queue_status") == "ready_for_local_mlx_advisory_execution"
        requested_bytes = _safe_int(row.get("requested_repair_bytes"))
        improvement = _safe_float(row.get("improvement_score_units")) or 0.0
        typed_response_id = str(row.get("typed_response_id") or "")
        blockers: list[str] = _string_list(row.get("source_blockers"))
        if not ready:
            blockers.append("local_mlx_advisory_custody_missing")
        if requested_bytes <= 0:
            blockers.append("requested_repair_bytes_missing")
        if improvement <= 0.0:
            blockers.append("non_improving_local_objective_delta")
        if remaining <= 0:
            blockers.append("receiver_closed_rate_credit_exhausted")
        if blockers:
            receiver_proof_status = _mapping(row.get("receiver_proof_status"))
            blocked_rows.append(
                {
                    "typed_response_id": typed_response_id,
                    "candidate_id": row.get("candidate_id"),
                    "acquisition_id": row.get("acquisition_id"),
                    "family_id": row.get("family_id"),
                    "correction_family": row.get("correction_family"),
                    "targeted_dimensions": _string_list(row.get("targeted_dimensions")),
                    "operation_levels": _string_list(row.get("operation_levels")),
                    "campaign_rank": row.get("campaign_rank"),
                    "entropy_position_label": row.get("entropy_position_label"),
                    "requested_repair_bytes": requested_bytes,
                    "objective_delta_score_units": row.get("objective_delta_score_units"),
                    "expected_local_improvement_score_units": improvement,
                    "campaign_score": row.get("campaign_score"),
                    "posterior_prior_multiplier": row.get(
                        "posterior_prior_multiplier"
                    ),
                    "palette_frame_asymmetry_multiplier": row.get(
                        "palette_frame_asymmetry_multiplier"
                    ),
                    "palette_dynamics_context": dict(
                        _mapping(row.get("palette_dynamics_context"))
                    ),
                    "posterior_family_prior": dict(
                        _mapping(row.get("posterior_family_prior"))
                    ),
                    "per_op_bytes_delta": row.get("per_op_bytes_delta"),
                    "component_response_terms": dict(
                        _mapping(row.get("component_response_terms"))
                    ),
                    "multiscale_action_row": dict(
                        _mapping(row.get("multiscale_action_row"))
                    ),
                    "receiver_proof_status": dict(receiver_proof_status),
                    "hard_legal_runtime_constraints": _string_list(
                        row.get("hard_legal_runtime_constraints")
                    ),
                    "execution_gate": dict(gate),
                    "missing_artifacts": ordered_unique(
                        [
                            *_string_list(row.get("source_missing_artifacts")),
                            *_string_list(gate.get("missing_artifacts")),
                            *_string_list(
                                receiver_proof_status.get("missing_artifacts")
                            ),
                        ]
                    ),
                    "interaction_penalty": row.get("interaction_penalty"),
                    "interaction_scope": dict(_mapping(row.get("interaction_scope"))),
                    "stacking_interaction_terms": dict(
                        _mapping(row.get("stacking_interaction_terms"))
                    ),
                    "blockers": ordered_unique(blockers),
                    "budget_spend_allowed": False,
                    "ready_for_exact_eval_dispatch": False,
                    "allowed_use": "blocked_repair_optimizer_signal_only",
                    "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
                    **FALSE_AUTHORITY,
                }
            )
            continue
        allocated = min(remaining, requested_bytes)
        remaining -= allocated
        allocation_fraction = allocated / requested_bytes if requested_bytes > 0 else 0.0
        scaled_improvement = improvement * allocation_fraction
        selected_rows.append(
            {
                "schema": REPAIR_CAMPAIGN_OPTIMIZER_ALLOCATION_ROW_SCHEMA,
                "typed_response_id": typed_response_id,
                "candidate_id": row.get("candidate_id"),
                "acquisition_id": row.get("acquisition_id"),
                "family_id": row.get("family_id"),
                "correction_family": row.get("correction_family"),
                "targeted_dimensions": _string_list(row.get("targeted_dimensions")),
                "operation_levels": _string_list(row.get("operation_levels")),
                "campaign_rank": row.get("campaign_rank"),
                "entropy_position_label": row.get("entropy_position_label"),
                "requested_repair_bytes": requested_bytes,
                "allocated_repair_bytes": allocated,
                "allocation_fraction": allocation_fraction,
                "remaining_receiver_closed_rate_credit_bytes_after": remaining,
                "objective_delta_score_units": row.get("objective_delta_score_units"),
                "expected_local_improvement_score_units": improvement,
                "scaled_expected_local_improvement_score_units": scaled_improvement,
                "campaign_score": row.get("campaign_score"),
                "posterior_prior_multiplier": row.get(
                    "posterior_prior_multiplier"
                ),
                "palette_frame_asymmetry_multiplier": row.get(
                    "palette_frame_asymmetry_multiplier"
                ),
                "palette_dynamics_context": dict(
                    _mapping(row.get("palette_dynamics_context"))
                ),
                "posterior_family_prior": dict(
                    _mapping(row.get("posterior_family_prior"))
                ),
                "per_op_bytes_delta": row.get("per_op_bytes_delta"),
                "component_response_terms": dict(
                    _mapping(row.get("component_response_terms"))
                ),
                "multiscale_action_row": dict(
                    _mapping(row.get("multiscale_action_row"))
                ),
                "receiver_proof_status": dict(
                    _mapping(row.get("receiver_proof_status"))
                ),
                "hard_legal_runtime_constraints": _string_list(
                    row.get("hard_legal_runtime_constraints")
                ),
                "interaction_penalty": row.get("interaction_penalty"),
                "interaction_scope": dict(_mapping(row.get("interaction_scope"))),
                "stacking_interaction_terms": dict(
                    _mapping(row.get("stacking_interaction_terms"))
                ),
                "selection_rationale": (
                    "greedy_campaign_score_waterfill_under_receiver_closed_byte_credit"
                ),
                "component_response_axis": "[macOS-MLX research-signal]",
                "budget_spend_allowed": False,
                "ready_for_budget_spend": False,
                "ready_for_exact_eval_dispatch": False,
                "allowed_use": "local_mlx_repair_optimizer_selection_only",
                "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
                **FALSE_AUTHORITY,
            }
        )
    allocated_total = sum(int(row.get("allocated_repair_bytes") or 0) for row in selected_rows)
    expected_improvement_total = sum(
        float(row.get("scaled_expected_local_improvement_score_units") or 0.0)
        for row in selected_rows
    )
    entropy_histogram: dict[str, int] = {}
    family_histogram: dict[str, int] = {}
    for row in selected_rows:
        entropy = str(row.get("entropy_position_label") or "unknown_entropy_pipeline_position")
        family = str(row.get("family_id") or "unclassified_repair_family")
        entropy_histogram[entropy] = entropy_histogram.get(entropy, 0) + 1
        family_histogram[family] = family_histogram.get(family, 0) + 1
    blockers = [
        "local_mlx_repair_optimizer_is_not_budget_spend_authority",
        "receiver_runtime_materialization_required_before_budget_spend",
        "exact_axis_component_response_required_before_budget_spend",
        "exact_auth_eval_required_before_score_or_promotion_claim",
        "stacking_interactions_must_be_remeasured_after_materialization",
    ]
    if available_bytes <= 0:
        blockers.append("receiver_closed_rate_credit_missing")
    if not selected_rows:
        blockers.append("no_repair_rows_selected_under_current_constraints")
    decision = {
        "schema": REPAIR_CAMPAIGN_OPTIMIZER_DECISION_SCHEMA,
        "input_schema": payload.get("schema"),
        "objective": "minimize_delta_segnet_plus_delta_posenet_plus_lambda_delta_bytes",
        "solver": "greedy_campaign_score_waterfill_v1",
        "receiver_closed_rate_credit_bytes": available_bytes,
        "selected_allocation_count": len(selected_rows),
        "blocked_allocation_count": len(blocked_rows),
        "allocated_repair_bytes_total": allocated_total,
        "unallocated_receiver_closed_rate_credit_bytes": remaining,
        "expected_local_improvement_score_units_total": expected_improvement_total,
        "entropy_position_allocation_histogram": dict(sorted(entropy_histogram.items())),
        "family_allocation_histogram": dict(sorted(family_histogram.items())),
        "selected_allocation_rows": selected_rows,
        "blocked_allocation_rows": blocked_rows,
        "posterior_prior_summary": dict(posterior_summary),
        "hard_constraints": [
            "allocated_bytes_must_not_exceed_receiver_closed_rate_credit",
            "local_mlx_response_is_planning_signal_only",
            "parent_rate_only_archive_must_materialize_first",
            "receiver_consumes_materialized_runtime_output",
            "component_response_replayed_before_budget_spend",
            "exact_auth_eval_required_before_score_or_promotion_claim",
        ],
        "blockers": ordered_unique(blockers),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "recommended_next_action": (
            "emit_local_mlx_repair_stackability_probe_queue"
            if selected_rows
            else "materialize_missing_repair_campaign_artifacts"
        ),
        "allowed_use": "queue_owned_repair_campaign_optimizer_decision_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        decision,
        context="repair_campaign_optimizer_decision",
    )
    return decision


def _find_mapping_by_typed_response_id(
    rows: Sequence[Any],
    *,
    typed_response_id: str,
) -> Mapping[str, Any]:
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("typed_response_id") or "") == typed_response_id:
            return row
    return {}


def build_repair_campaign_stackability_probe(
    *,
    score_report: Mapping[str, Any],
    typed_response_id: str,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build an executable local stackability probe from an optimizer allocation.

    The probe is still advisory. It proves the allocation is backed by local MLX
    response custody before a queue spends additional local work on stacking or
    remeasurement, but it never becomes a score, budget, promotion, or dispatch
    authority.
    """

    typed_id = str(typed_response_id or "").strip()
    if not typed_id:
        raise RepairCampaignScorerError("typed_response_id is required")
    if score_report.get("schema") != REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA:
        raise RepairCampaignScorerError(
            "stackability probe requires repair_campaign_score_report.v1"
        )
    require_no_truthy_authority_fields(
        score_report,
        context="repair_campaign_stackability_probe_input",
    )
    decision = _mapping(score_report.get("optimizer_decision"))
    if decision.get("schema") != REPAIR_CAMPAIGN_OPTIMIZER_DECISION_SCHEMA:
        raise RepairCampaignScorerError(
            "score report missing repair_campaign_optimizer_decision.v1"
        )
    allocation = _find_mapping_by_typed_response_id(
        decision.get("selected_allocation_rows") or [],
        typed_response_id=typed_id,
    )
    score_row = _find_mapping_by_typed_response_id(
        score_report.get("rows") or [],
        typed_response_id=typed_id,
    )
    blockers: list[str] = []
    missing_artifacts: list[str] = []
    if not allocation:
        blockers.append("optimizer_selected_allocation_missing")
    if not score_row:
        blockers.append("source_score_row_missing")

    allocated_bytes = _safe_int(allocation.get("allocated_repair_bytes"))
    if allocated_bytes <= 0:
        blockers.append("allocated_repair_bytes_missing")

    gate = _mapping(score_row.get("execution_gate"))
    custody_paths = [
        item
        for item in gate.get("local_mlx_custody_paths") or []
        if isinstance(item, Mapping)
    ]
    required_keys = {"local_mlx_response_path", "reference_local_mlx_response_path"}
    required_status = [
        item for item in custody_paths if str(item.get("key") or "") in required_keys
    ]
    if not required_status:
        blockers.append("local_mlx_required_custody_paths_missing")
    for item in required_status:
        if item.get("exists") is not True:
            key = str(item.get("key") or "unknown_local_mlx_path")
            missing_artifacts.append(f"{key}:missing_or_unverified")
            path_text = str(item.get("path") or "").strip()
            if path_text and not _repo_path_exists(path_text, repo_root):
                missing_artifacts.append(f"{key}:{path_text}")
    if gate.get("local_mlx_advisory_custody_ready") is not True:
        blockers.append("local_mlx_advisory_custody_missing")
    missing_artifacts.extend(_string_list(gate.get("missing_artifacts")))

    entropy_position = str(
        allocation.get("entropy_position_label")
        or score_row.get("entropy_position_label")
        or "unknown_entropy_pipeline_position"
    )
    stackability_ready = not blockers
    status = (
        "ready_for_local_mlx_stackability_probe"
        if stackability_ready
        else "blocked_missing_artifact"
    )
    probe = {
        "schema": REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA,
        "typed_response_id": typed_id,
        "status": status,
        "source_score_report_schema": score_report.get("schema"),
        "source_optimizer_decision_schema": decision.get("schema"),
        "component_response_axis": "[macOS-MLX research-signal]",
        "probe_execution_mode": (
            "local_manifest_probe_from_existing_mlx_advisory_custody"
        ),
        "stackability_ready": stackability_ready,
        "optimizer_allocation": dict(allocation),
        "source_score_row": dict(score_row),
        "allocated_repair_bytes": allocated_bytes,
        "receiver_closed_rate_credit_bytes": _safe_int(
            decision.get("receiver_closed_rate_credit_bytes")
        ),
        "remaining_receiver_closed_rate_credit_bytes_after": _safe_int(
            allocation.get("remaining_receiver_closed_rate_credit_bytes_after")
        ),
        "expected_local_improvement_score_units": _safe_float(
            allocation.get("scaled_expected_local_improvement_score_units")
        )
        or 0.0,
        "multiscale_action_row": dict(
            _mapping(
                allocation.get("multiscale_action_row")
                if allocation
                else score_row.get("multiscale_action_row")
            )
        ),
        "palette_dynamics_context": dict(
            _mapping(
                allocation.get("palette_dynamics_context")
                if allocation
                else score_row.get("palette_dynamics_context")
            )
        ),
        "palette_frame_asymmetry_multiplier": (
            _safe_float(
                allocation.get("palette_frame_asymmetry_multiplier")
                if allocation
                else score_row.get("palette_frame_asymmetry_multiplier")
            )
            or 1.0
        ),
        "entropy_position_label": entropy_position,
        "entropy_position_class": (
            "pre_entropy_distribution_shaping"
            if entropy_position.startswith("before_entropy_coder")
            else (
                "scorer_entropy_before_selector_codec"
                if entropy_position == "scorer_entropy_repair_before_selector_codec"
                else (
                    "entropy_coder_boundary"
                    if entropy_position.startswith("at_entropy_coder")
                    else (
                        "post_entropy_container"
                        if entropy_position.startswith("after_entropy_coder")
                        else "selector_or_unknown_entropy_position"
                    )
                )
            )
        ),
        "interaction_penalty": _safe_float(
            allocation.get("interaction_penalty")
            if allocation
            else score_row.get("interaction_penalty")
        )
        or 0.0,
        "interaction_scope": dict(
            _mapping(
                allocation.get("interaction_scope")
                if allocation
                else score_row.get("interaction_scope")
            )
        ),
        "stacking_interaction_terms": dict(
            _mapping(
                allocation.get("stacking_interaction_terms")
                if allocation
                else score_row.get("stacking_interaction_terms")
            )
        ),
        "local_mlx_custody_paths": [dict(item) for item in custody_paths],
        "missing_artifacts": ordered_unique(missing_artifacts),
        "blockers": ordered_unique(
            [
                *blockers,
                "local_mlx_probe_is_not_score_authority",
                "exact_axis_component_response_required_before_budget_spend",
                "receiver_runtime_materialization_required_before_exact_dispatch",
                "stacking_interactions_must_be_remeasured_after_materialization",
            ]
        ),
        "hard_constraints": [
            "allocated_bytes_must_not_exceed_receiver_closed_rate_credit",
            "local_mlx_response_is_planning_signal_only",
            "repair_probe_must_not_modify_receiver_runtime",
            "exact_auth_eval_required_before_score_or_promotion_claim",
            "post_materialization_stackability_must_be_remeasured",
        ],
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "local_mlx_repair_stackability_probe_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        probe,
        context=f"repair_campaign_stackability_probe:{typed_id}",
    )
    return probe


def score_repair_campaign(
    *,
    payload: Mapping[str, Any],
    repo_root: str | Path | None = None,
    posterior_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score repair typed rows for the next queue-owned local campaign slice."""

    require_no_truthy_authority_fields(payload, context="repair_campaign_scorer_input")
    posterior_rows = _load_posterior_rows(posterior_path)
    posterior_summary = _posterior_prior_summary(
        posterior_path=posterior_path,
        posterior_rows=posterior_rows,
    )
    campaign_input_rows = [*_typed_rows(payload), *_cascade_opportunity_rows(payload)]
    structural_opportunity_count = sum(
        1
        for row in campaign_input_rows
        if row.get("source_row_kind") == "repair_cascade_opportunity"
    )
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(campaign_input_rows, start=1):
        require_no_truthy_authority_fields(
            row,
            context=f"repair_campaign_scorer_row:{index}",
        )
        prior = _family_prior(row)
        objective_delta = _objective_delta(row)
        requested_bytes = _requested_bytes(row)
        improvement = (
            -objective_delta
            if objective_delta is not None and objective_delta < 0.0
            else 0.0
        )
        entropy_position = _entropy_position(row, prior)
        entropy_weight = _ENTROPY_POSITION_WEIGHTS.get(
            entropy_position,
            _ENTROPY_POSITION_WEIGHTS["unknown_entropy_pipeline_position"],
        )
        family_multiplier = _safe_float(prior.get("campaign_prior_multiplier")) or 1.0
        interaction_penalty = _interaction_penalty(row)
        per_op_bytes_delta = _per_op_bytes_delta(row)
        component_response_terms = _component_response_terms(row)
        receiver_proof = _receiver_proof_status(row, repo_root=repo_root)
        palette_dynamics_context = _repair_dynamics_palette_context(row, prior)
        hard_constraints = _hard_legal_runtime_constraints(row)
        if palette_dynamics_context:
            hard_constraints = ordered_unique(
                [
                    *hard_constraints,
                    "frame0_palette_repairs_are_global_interaction_terms",
                    "palette_frame_asymmetry_requires_same_axis_stackability_remeasure",
                    *(
                        ["frame1_palette_repairs_require_counterfactual_probe"]
                        if palette_dynamics_context.get("zero_frame1_modes") is True
                        else []
                    ),
                ]
            )
        multiscale_action = _multiscale_action_row(
            source_row=row,
            prior=prior,
            entropy_position=entropy_position,
            requested_bytes=requested_bytes,
            objective_delta=objective_delta,
            per_op_bytes_delta=per_op_bytes_delta,
            component_response_terms=component_response_terms,
            interaction_penalty=interaction_penalty,
            hard_constraints=hard_constraints,
            palette_dynamics_context=palette_dynamics_context,
        )
        bytes_denominator = requested_bytes if requested_bytes > 0 else 1
        improvement_per_byte = improvement / bytes_denominator
        posterior_family_prior = _posterior_family_prior(
            posterior_summary,
            family_id=str(prior.get("family_id") or "unclassified_repair_family"),
        )
        posterior_prior_multiplier = (
            _safe_float(posterior_family_prior.get("family_priority_multiplier"))
            if posterior_family_prior
            else 1.0
        ) or 1.0
        palette_multiplier = _palette_frame_asymmetry_multiplier(
            palette_dynamics_context
        )
        campaign_score = (
            improvement_per_byte
            * entropy_weight
            * family_multiplier
            * posterior_prior_multiplier
            * palette_multiplier
            * (1.0 - interaction_penalty)
        )
        gate = _execution_gate(row, prior, repo_root=repo_root)
        scored_row = {
            "schema": REPAIR_CAMPAIGN_SCORE_ROW_SCHEMA,
            "source_row_schema": row.get("schema"),
            "source_row_kind": row.get("source_row_kind") or "typed_response",
            "rank_input_order": index,
            "typed_response_id": row.get("typed_response_id"),
            "candidate_id": row.get("candidate_id"),
            "acquisition_id": row.get("acquisition_id"),
            "correction_family": row.get("correction_family"),
            "family_id": prior.get("family_id") or "unclassified_repair_family",
            "cascade_id": row.get("cascade_id"),
            "source_relation": row.get("source_relation"),
            "source_structural_opportunity": dict(
                _mapping(row.get("source_structural_opportunity"))
            ),
            "source_missing_artifacts": _string_list(row.get("missing_artifacts")),
            "source_blockers": _string_list(row.get("blockers")),
            "family_prior": dict(prior),
            "targeted_dimensions": _string_list(row.get("targeted_dimensions")),
            "operation_levels": _string_list(row.get("operation_levels")),
            "entropy_position_label": entropy_position,
            "entropy_position_weight": entropy_weight,
            "requested_repair_bytes": requested_bytes,
            "objective_delta_score_units": objective_delta,
            "improvement_score_units": improvement,
            "improvement_per_byte": improvement_per_byte,
            "per_op_bytes_delta": per_op_bytes_delta,
            "component_response_terms": component_response_terms,
            "multiscale_action_row": multiscale_action,
            "receiver_proof_status": receiver_proof,
            "hard_legal_runtime_constraints": hard_constraints,
            "family_prior_multiplier": family_multiplier,
            "posterior_prior_multiplier": posterior_prior_multiplier,
            "palette_frame_asymmetry_multiplier": palette_multiplier,
            "palette_dynamics_context": dict(palette_dynamics_context),
            "posterior_family_prior": dict(posterior_family_prior),
            "interaction_penalty": interaction_penalty,
            "campaign_score": campaign_score,
            "marginal_response_curves": dict(
                _mapping(row.get("marginal_response_curves"))
            ),
            "interaction_scope": dict(_mapping(row.get("interaction_scope"))),
            "stacking_interaction_terms": dict(
                _mapping(row.get("stacking_interaction_terms"))
            ),
            "execution_gate": gate,
            "recommended_next_action": (
                "run_local_mlx_repair_stackability_probe"
                if gate["local_mlx_advisory_custody_ready"]
                else "materialize_missing_repair_campaign_artifacts"
            ),
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            "allowed_use": "repair_campaign_local_acquisition_ranking_only",
            "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
            **FALSE_AUTHORITY,
        }
        require_no_truthy_authority_fields(
            scored_row,
            context=f"repair_campaign_score_row:{index}",
        )
        rows.append(scored_row)
    rows.sort(
        key=lambda item: (
            item["execution_gate"]["recommended_queue_status"]
            != "ready_for_local_mlx_advisory_execution",
            -float(item.get("campaign_score") or 0.0),
            str(item.get("typed_response_id") or item.get("candidate_id") or ""),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["campaign_rank"] = rank
        action = row.get("multiscale_action_row")
        if isinstance(action, dict):
            action["campaign_rank"] = rank
    ready_rows = [
        row
        for row in rows
        if row["execution_gate"]["recommended_queue_status"]
        == "ready_for_local_mlx_advisory_execution"
    ]
    missing_artifacts = ordered_unique(
        artifact
        for row in rows
        for artifact in _string_list(row["execution_gate"].get("missing_artifacts"))
    )
    optimizer_decision = _build_optimizer_decision(
        payload=payload,
        rows=rows,
        posterior_summary=posterior_summary,
    )
    multiscale_ledger = _multiscale_action_ledger(rows)
    report = {
        "schema": REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA,
        "row_schema": REPAIR_CAMPAIGN_SCORE_ROW_SCHEMA,
        "optimizer_decision_schema": REPAIR_CAMPAIGN_OPTIMIZER_DECISION_SCHEMA,
        "default_campaign_scorer": True,
        "input_schema": payload.get("schema"),
        "posterior_path": str(posterior_path) if posterior_path is not None else None,
        "posterior_prior_summary": posterior_summary,
        "row_count": len(rows),
        "ready_for_local_mlx_advisory_execution_count": len(ready_rows),
        "blocked_missing_artifact_count": len(rows) - len(ready_rows),
        "operator_family_priors": repair_operator_family_priors(),
        "structural_repair_opportunity_count": structural_opportunity_count,
        "multiscale_action_ledger": multiscale_ledger,
        "missing_artifacts": missing_artifacts,
        "optimizer_decision": optimizer_decision,
        "rows": rows,
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "default_repair_campaign_scorer_for_queue_planning",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(report, context="repair_campaign_score_report")
    return report


__all__ = [
    "REPAIR_CAMPAIGN_MULTISCALE_ACTION_LEDGER_SCHEMA",
    "REPAIR_CAMPAIGN_MULTISCALE_ACTION_ROW_SCHEMA",
    "REPAIR_CAMPAIGN_OPTIMIZER_ALLOCATION_ROW_SCHEMA",
    "REPAIR_CAMPAIGN_OPTIMIZER_DECISION_SCHEMA",
    "REPAIR_CAMPAIGN_POSTERIOR_FAMILY_PRIOR_SCHEMA",
    "REPAIR_CAMPAIGN_POSTERIOR_PRIOR_SUMMARY_SCHEMA",
    "REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA",
    "REPAIR_CAMPAIGN_SCORE_ROW_SCHEMA",
    "REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA",
    "REPAIR_OPERATOR_FAMILY_PRIORS_SCHEMA",
    "REPAIR_OPERATOR_FAMILY_PRIOR_ROW_SCHEMA",
    "RepairCampaignScorerError",
    "build_repair_campaign_stackability_probe",
    "repair_operator_family_priors",
    "score_repair_campaign",
]
