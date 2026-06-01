# SPDX-License-Identifier: MIT
"""Contest-rate bitstream grammar planner for compact HPRC spines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tac.archive_byte_profile import contest_rate_term

HPRC_OPTIMAL_BITSTREAM_GRAMMAR_PLAN_SCHEMA = "hprc_optimal_bitstream_grammar_plan.v1"
HPRC_OPTIMAL_BITSTREAM_GRAMMAR_ROW_SCHEMA = "hprc_optimal_bitstream_grammar_row.v1"
SATURATION_RATIO_STRONG = 1.02
SATURATION_RATIO_WEAK = 1.10

SECTION_ORDER: tuple[str, ...] = (
    "decoder_qw",
    "latents_rc",
    "selectors_rc",
    "codebooks_q",
    "residual_rc",
    "receiver_state",
    "rdo_plan",
    "manifest_json",
)

_SECTION_POLICY: dict[str, dict[str, str]] = {
    "decoder_qw": {
        "role": "charged_program_weights",
        "admission_policy": (
            "protect_until_full_video_neutralization_proves marginal value below "
            "contest rate price"
        ),
        "preferred_coding": "weight_quant_delta_or_fixed_decoder_id_plus_entropy_coded_weights",
    },
    "latents_rc": {
        "role": "charged_temporal_latent_token_stream",
        "admission_policy": "protect_and_entropy_code_against temporal_context_prior",
        "preferred_coding": "range_or_ans_tokens_with_pair_gop_context",
    },
    "selectors_rc": {
        "role": "charged_mode_region_boundary_temporal_selectors",
        "admission_policy": "cut_or_recode_unless full_video_value_per_byte wins",
        "preferred_coding": "bitpacked_context_tokens_or_implicit_rule_if_cheaper",
    },
    "codebooks_q": {
        "role": "charged_vq_or_implicit_atom_tables",
        "admission_policy": "admit_only_when table_bytes amortize over index_stream_gain",
        "preferred_coding": "quantized_codebook_plus_entropy_coded_indices",
    },
    "residual_rc": {
        "role": "charged_scorer_priced_residual_tokens",
        "admission_policy": "admit_only_if measured_delta_nonrate_plus_rate_cost_lt_zero",
        "preferred_coding": "significance_ordered_tokens_with_p18_p19_value_sort",
    },
    "receiver_state": {
        "role": "charged_decode_constants",
        "admission_policy": "minimize_and_inline_defaults; no hidden sidecars",
        "preferred_coding": "implicit_defaults_plus_varint_deltas",
    },
    "rdo_plan": {
        "role": "charged_allocation_hint_no_authority",
        "admission_policy": "strip_from_runtime_if_decoder_does_not_need_it",
        "preferred_coding": "proof_manifest_only_or_compact_flags",
    },
    "manifest_json": {
        "role": "charged_runtime_manifest_when_present",
        "admission_policy": "keep proof metadata outside paid runtime unless consumed",
        "preferred_coding": "canonical_external_manifest_or_minimal_cbor_like_fields",
    },
}


def joint_p18_p19_saliency_contract() -> dict[str, Any]:
    """Return the score-faithful saliency contract used by grammar planning."""

    return {
        "schema": "hprc_joint_p18_p19_saliency_contract.v1",
        "do_not_collapse_channels_before_incidence_projection": True,
        "segnet_frame_channel": {
            "schema": "hprc_segnet_frame_saliency_channel.v1",
            "contest_term": "100*d_seg",
            "data_domain": "single RGB frame -> SegNet five-class logits",
            "measurement": "argmax_flip_indicator_against_gt_frame",
            "local_proxy": "top2_logit_margin_smoothed_step_or_crammer_singer_hinge",
            "spatial_structure": "class_boundary_concentrated; flat_interiors_near_zero",
            "scope": "frame_pixel_class_boundary",
        },
        "posenet_pair_channel": {
            "schema": "hprc_posenet_pair_saliency_channel.v1",
            "contest_term": "sqrt(10*d_pose)",
            "data_domain": "two RGB frames converted to 12-channel YUV6 -> six-DOF pose",
            "measurement": "inverse_variance_mahalanobis_pose_error",
            "local_proxy": "pose_pixel_jacobian_squared_times_5_over_sqrt_10_d_pose",
            "spatial_structure": (
                "camera_geometry_and_texture_gradient_spread; not class-boundary-only"
            ),
            "scope": "pair_pixel_geometry",
        },
        "incidence_projection": {
            "schema": "hprc_pair_latent_frame_pair_incidence.v1",
            "rule": (
                "project frame and pair saliencies to the candidate atom before "
                "adding them; one pair latent can move both decoded frames, and "
                "each decoded frame participates in SegNet plus PoseNet pair terms"
            ),
            "atom_domains": [
                "pixel",
                "class_region",
                "boundary_band",
                "wavelet_subband",
                "latent_dimension",
                "pair",
                "temporal_window",
                "full_video",
            ],
        },
        "combined_saliency": {
            "schema": "hprc_joint_saliency_combine_rule.v1",
            "formula": (
                "s_atom=sum_frame_incidence(100*phi(seg_margin))"
                "+sum_pair_incidence((5/sqrt(10*d_pose))*||J_pose_atom||^2_Sigma_inv)"
            ),
            "allocator": (
                "reverse water-fill over atoms after full-video reduction; "
                "dead-zone below lambda, spend precision where value_per_byte "
                "exceeds contest rate price"
            ),
        },
        "authority": {
            "mlx_or_proxy_rows": "proposal_only",
            "promotion_requires": [
                "byte_closed_archive",
                "receiver_decode_only_proof",
                "full_video_replay_or_exact_axis_blocker",
                "contest_cpu_cuda_exact_eval_for_score_authority",
            ],
        },
    }


def build_optimal_bitstream_grammar_plan(
    *,
    acquisition_rows: Sequence[Mapping[str, Any]],
    hard_byte_ceilings: Sequence[int],
) -> dict[str, Any]:
    """Return a row-level grammar plan tied to contest rate accounting.

    This is intentionally not a score claim.  It converts compact-base rows into
    executable byte grammar choices and section admission rules so acquisition
    can route real work instead of family-specific folklore.
    """

    ceilings = [int(value) for value in hard_byte_ceilings if int(value) > 0]
    rows = [_row_plan(row=row, ceilings=ceilings) for row in acquisition_rows]
    rows.sort(
        key=lambda row: (
            row["fits_smallest_ceiling"] is False,
            int(row["effective_archive_bytes"]),
            row["family"],
        )
    )
    return {
        "schema": HPRC_OPTIMAL_BITSTREAM_GRAMMAR_PLAN_SCHEMA,
        "objective": {
            "schema": "hprc_contest_score_byte_objective.v1",
            "minimize": (
                "100*d_seg(full_video)+sqrt(10*d_pose(full_video))"
                "+25*archive_zip_bytes/original_bytes"
            ),
            "rate_price_per_byte": contest_rate_term(1),
            "section_admission_rule": (
                "keep a charged section only when full-video scorer value exceeds "
                "its exact archive byte price"
            ),
        },
        "semantic_allocation_grammar": joint_p18_p19_saliency_contract(),
        "entropy_saturation_diagnostic": {
            "schema": "hprc_entropy_saturation_diagnostic.v1",
            "saturated_if_coded_over_floor_ratio_lte": SATURATION_RATIO_STRONG,
            "weak_gap_if_coded_over_floor_ratio_lte": SATURATION_RATIO_WEAK,
            "strong_gap_action": (
                "open coder-selection materializer only when coded/floor gap remains "
                "above weak saturation after receiver proof"
            ),
        },
        "hard_byte_ceilings": ceilings,
        "packet_grammar_choices": [
            {
                "grammar": "source_archive_native_single_member_zip",
                "use_for": "public_pr95_or_other_public_archive_replay",
                "promotion_rule": "exact source runtime custody before authority",
            },
            {
                "grammar": "hprc_g1_compact_bitmask_varint",
                "use_for": "new HPRC-native compact candidates",
                "promotion_rule": "parse_roundtrip plus receiver proof plus exact gate",
            },
            {
                "grammar": "section_entropy_portfolio",
                "use_for": "payload sections after semantic value is proven",
                "promotion_rule": "per-section codec chosen by measured bytes at equal replay value",
            },
        ],
        "ordered_section_policy": [
            {"section": section, **_SECTION_POLICY[section]} for section in SECTION_ORDER
        ],
        "rows": rows,
    }


def _row_plan(*, row: Mapping[str, Any], ceilings: Sequence[int]) -> dict[str, Any]:
    effective_bytes = int(row.get("effective_archive_bytes") or 0)
    source_archive = row.get("source_archive")
    source_archive_present = isinstance(source_archive, Mapping) and bool(
        source_archive.get("path") or source_archive.get("archive_path")
    )
    raw_sections = row.get("section_rows")
    sections = raw_sections if isinstance(raw_sections, Sequence) and not isinstance(raw_sections, (str, bytes)) else []
    section_rows = [_section_plan(raw) for raw in sections if isinstance(raw, Mapping)]
    section_rows.sort(key=lambda item: SECTION_ORDER.index(item["section"]))
    smallest = min(ceilings) if ceilings else None
    return {
        "schema": HPRC_OPTIMAL_BITSTREAM_GRAMMAR_ROW_SCHEMA,
        "family": str(row.get("family") or "unknown"),
        "effective_archive_bytes": effective_bytes,
        "effective_rate_term": contest_rate_term(effective_bytes),
        "fits_smallest_ceiling": None
        if smallest is None
        else effective_bytes <= int(smallest),
        "recommended_outer_grammar": (
            "source_archive_native_single_member_zip"
            if source_archive_present
            else "hprc_g1_compact_bitmask_varint"
        ),
        "section_count": len(section_rows),
        "paid_section_bytes": sum(int(item["bytes"]) for item in section_rows),
        "section_rows": section_rows,
        "missing_value_measurement_sections": [
            item["section"]
            for item in section_rows
            if item["requires_full_video_value_measurement"]
        ],
        "next_executable_task": _next_task(
            source_archive_present=source_archive_present,
            section_rows=section_rows,
            effective_bytes=effective_bytes,
            smallest_ceiling=smallest,
        ),
    }


def _section_plan(raw: Mapping[str, Any]) -> dict[str, Any]:
    name = str(raw.get("name") or "")
    if name not in _SECTION_POLICY:
        raise ValueError(f"unknown HPRC section in grammar plan: {name!r}")
    byte_count = int(raw.get("bytes") or 0)
    policy = _SECTION_POLICY[name]
    entropy = _entropy_gap(raw, fallback_coded_bytes=byte_count)
    return {
        "schema": "hprc_optimal_bitstream_section_plan.v1",
        "section": name,
        "bytes": byte_count,
        "rate_cost": contest_rate_term(byte_count),
        "role": policy["role"],
        "admission_policy": policy["admission_policy"],
        "preferred_coding": policy["preferred_coding"],
        "entropy_gap": entropy,
        "requires_full_video_value_measurement": byte_count > 0,
    }


def _entropy_gap(raw: Mapping[str, Any], *, fallback_coded_bytes: int) -> dict[str, Any]:
    profile = raw.get("entropy_profile")
    if not isinstance(profile, Mapping):
        profile = raw.get("entropy_gap")
    if not isinstance(profile, Mapping):
        return {
            "schema": "hprc_section_entropy_gap.v1",
            "status": "measurement_required",
            "coded_bytes": int(fallback_coded_bytes),
            "empirical_floor_bytes": None,
            "coded_over_floor_ratio": None,
            "next_action": "run_per_section_or_per_tensor_entropy_probe",
        }
    coded = _positive_float(profile.get("coded_bytes")) or float(fallback_coded_bytes)
    floor = _positive_float(
        profile.get("empirical_floor_bytes")
        or profile.get("structured_floor_bytes")
        or profile.get("shannon_floor_bytes")
    )
    if floor is None:
        return {
            "schema": "hprc_section_entropy_gap.v1",
            "status": "floor_missing",
            "coded_bytes": int(coded),
            "empirical_floor_bytes": None,
            "coded_over_floor_ratio": None,
            "next_action": "compute_structured_shannon_floor_before_codec_work",
        }
    ratio = coded / floor
    if ratio <= SATURATION_RATIO_STRONG:
        status = "entropy_saturated"
        action = "do_not_spend_on_format_grammar_without_new_substrate_signal"
    elif ratio <= SATURATION_RATIO_WEAK:
        status = "weak_entropy_gap"
        action = "queue_low_priority_per_tensor_solver_after_scoreful_work"
    else:
        status = "unsaturated_entropy_gap"
        action = "open_smallest_bit_exact_coder_selection_materializer_task"
    return {
        "schema": "hprc_section_entropy_gap.v1",
        "status": status,
        "coded_bytes": int(coded),
        "empirical_floor_bytes": float(floor),
        "coded_over_floor_ratio": ratio,
        "next_action": action,
    }


def _next_task(
    *,
    source_archive_present: bool,
    section_rows: Sequence[Mapping[str, Any]],
    effective_bytes: int,
    smallest_ceiling: int | None,
) -> str:
    if not section_rows:
        return "materialize_runtime_packet_sections_before_grammar_optimization"
    if smallest_ceiling is not None and effective_bytes > smallest_ceiling:
        return "shrink_primary_carrier_before_residual_or_selector_bytes"
    residuals = [
        row
        for row in section_rows
        if row["section"] == "residual_rc" and int(row["bytes"]) > 0
    ]
    if residuals:
        return "run_full_video_value_per_byte_then_keep_only_negative_delta_residuals"
    unsaturated = [
        row
        for row in section_rows
        if row.get("entropy_gap", {}).get("status") == "unsaturated_entropy_gap"
    ]
    if unsaturated:
        return "open_entropy_gap_materializer_for_unsaturated_sections"
    if source_archive_present:
        return "receiver_proof_native_archive_then_full_video_value_replay"
    return "pack_hprc_g1_then_receiver_proof_and_exact_gate"


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


__all__ = [
    "HPRC_OPTIMAL_BITSTREAM_GRAMMAR_PLAN_SCHEMA",
    "HPRC_OPTIMAL_BITSTREAM_GRAMMAR_ROW_SCHEMA",
    "SATURATION_RATIO_STRONG",
    "SATURATION_RATIO_WEAK",
    "SECTION_ORDER",
    "build_optimal_bitstream_grammar_plan",
    "joint_p18_p19_saliency_contract",
]
