# SPDX-License-Identifier: MIT
"""Emission contract for receiver-closed SNeRV/HiNeRV modelsize rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SCHEMA = "nerv_ladder_row_emission_contract.v1"
AXIS_TAG = "[planning/control:false-authority]"
FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "production_hardened_claim": False,
    "ready_for_exact_eval_dispatch": False,
}


GENERIC_REQUIRED_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "group_id": "carrier_identity",
        "requirement": "one_of",
        "paths": ("family", "carrier_id"),
        "why": "prevents HNeRV/SNeRV/HiNeRV row laundering",
    },
    {
        "group_id": "source_bound_modelsize_control",
        "requirement": "one_of",
        "paths": (
            "modelsize_mparams",
            "official_controls.--modelsize",
            "solved_budget.modelsize_mparams",
            "fc_dim",
            "derived.fc_dim",
            "solved_budget.derived.fc_dim",
        ),
        "why": "makes --modelsize/fc_dim byte curves receiver-priced instead of projected",
    },
    {
        "group_id": "archive_byte_custody",
        "requirement": "all_of_any_alias",
        "paths": (
            ("archive_bytes", "archive_bytes_total", "measured_archive_bytes", "archive_zip_bytes"),
            ("archive_sha256", "receiver_archive_sha256", "archive_zip_sha256"),
        ),
        "why": "rate evidence must be measured archive bytes plus content hash",
    },
    {
        "group_id": "full600_scope",
        "requirement": "numeric_at_least_600",
        "paths": ("pair_count", "n_pairs", "sample_pair_count", "n_samples", "sample_count"),
        "why": "local 1/2/4-pair smokes are useful but not ladder authority",
    },
    {
        "group_id": "receiver_replay_proof",
        "requirement": "truthy",
        "paths": (
            "receiver_archive_replay_verified",
            "receiver_contract_satisfied",
            "byte_closed_receiver_proof",
        ),
        "why": "MLX prefilter/full-scope coverage is not receiver archive replay",
    },
    {
        "group_id": "nonrate_distortion",
        "requirement": "one_of",
        "paths": (
            "nonrate_score",
            "nonrate_score_value",
            ("d_seg+d_pose", "d_seg_mean_linf+d_pose_mean_linf"),
            ("avg_segnet_dist+avg_posenet_dist",),
        ),
        "why": "waterfill needs distortion movement independent of rate bytes",
    },
)


FAMILY_REQUIRED_GROUPS: dict[str, tuple[dict[str, Any], ...]] = {
    "snerv": (
        {
            "group_id": "snerv_official_source_controls",
            "requirement": "must_emit_all_available",
            "paths": (
                "official_controls.--modelsize",
                "official_controls.fc_dim",
                "official_controls.emb_size",
                "official_controls.wavelet",
                "official_controls.levels",
                "official_controls.mfu_enabled",
                "official_controls.hfr_enabled",
                "official_controls.snerv_t_enabled",
            ),
            "why": "source-faithful SNeRV evidence must preserve MFU/HFR/SNeRV_T and size controls",
        },
        {
            "group_id": "snerv_receiver_pricing_controls",
            "requirement": "must_emit_all_available",
            "paths": (
                "receiver_codec_mode",
                "lf_payload_codec",
                "decoder_precision_mode",
                "step_map_codec",
                "target_bits_per_coeff",
                "qat_bits",
            ),
            "why": "decoder QAT and LF/HF rate work must price the receiver-visible grammar",
        },
    ),
    "hinerv": (
        {
            "group_id": "hinerv_official_source_controls",
            "requirement": "must_emit_all_available",
            "paths": (
                "official_controls.config_name",
                "official_controls.patch_mode",
                "official_controls.hierarchical_grid_shapes",
                "official_controls.decoder_channels",
                "official_controls.prune_config",
                "official_controls.quant_config",
                "official_controls.bitstream_q",
            ),
            "why": "HiNeRV is distinct from HNeRV and must preserve its own config/bitstream controls",
        },
        {
            "group_id": "hinerv_receiver_pricing_controls",
            "requirement": "must_emit_all_available",
            "paths": (
                "receiver_codec_mode",
                "decoder_precision_mode",
                "latent_precision_mode",
                "hierarchical_grid_precision_modes",
                "bitstream_codec",
            ),
            "why": "HiNeRV pruning/quantization must be receiver-priced before ladder authority",
        },
    ),
}


def build_nerv_ladder_row_emission_contract(
    *,
    families: Iterable[str] = ("snerv", "hinerv"),
    source_parity_contract: Mapping[str, Any] | None = None,
    row_harvests: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a fail-closed contract for trainer/export row emission."""

    selected = tuple(dict.fromkeys(_family_key(family) for family in families))
    harvest_summaries = [_harvest_summary(harvest) for harvest in row_harvests]
    family_rows = [
        _family_contract_row(
            family,
            source_parity_contract=source_parity_contract,
            harvest_summaries=harvest_summaries,
        )
        for family in selected
    ]
    blockers = _ordered_unique(
        blocker for row in family_rows for blocker in row["blockers"]
    )
    return {
        "schema": SCHEMA,
        "axis_tag": AXIS_TAG,
        "families": selected,
        "generic_required_field_groups": list(GENERIC_REQUIRED_GROUPS),
        "family_rows": family_rows,
        "row_harvest_summaries": harvest_summaries,
        "ready_for_trained_ladder_row_emission": not blockers,
        "blockers": blockers,
        "next_actions": _next_actions(family_rows),
        **FALSE_AUTHORITY,
    }


def _family_contract_row(
    family: str,
    *,
    source_parity_contract: Mapping[str, Any] | None,
    harvest_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_blockers = _source_parity_blockers(family, source_parity_contract)
    harvest = _merge_harvest_summaries(family, harvest_summaries)
    missing = []
    if harvest["harvested_row_count"] == 0:
        missing.append("no_harvested_rows_observed")
    if harvest["full_scope_row_count"] < 2:
        missing.append("fewer_than_two_full600_rows_observed")
    if harvest["receiver_proof_row_count"] < 2:
        missing.append("fewer_than_two_receiver_proof_rows_observed")
    if harvest["modelsize_present_row_count"] < 2:
        missing.append("fewer_than_two_modelsize_or_fc_dim_rows_observed")
    if harvest["ladder_candidate_row_count"] < 2:
        missing.append("fewer_than_two_ladder_candidate_rows_observed")

    blockers = _ordered_unique(
        [
            *(f"source_parity:{blocker}" for blocker in source_blockers),
            *(f"emission_gap:{family}:{gap}" for gap in missing),
        ]
    )
    return {
        "family": family,
        "required_field_groups": [
            *GENERIC_REQUIRED_GROUPS,
            *FAMILY_REQUIRED_GROUPS.get(family, ()),
        ],
        "source_parity_blockers": source_blockers,
        "observed_harvest_summary": harvest,
        "emission_gap_ids": missing,
        "ready_for_trained_ladder_row_emission": not blockers,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _harvest_summary(harvest: Mapping[str, Any]) -> dict[str, Any]:
    carrier = _family_key(harvest.get("carrier_id") or "unknown")
    return {
        "schema": harvest.get("schema"),
        "carrier_id": carrier,
        "status": harvest.get("status"),
        "source_path": harvest.get("source_artifact_path"),
        "harvested_row_count": _int(harvest.get("harvested_row_count")),
        "full_scope_row_count": _int(harvest.get("full_scope_row_count")),
        "local_receiver_replay_row_count": _int(
            harvest.get("local_receiver_replay_row_count")
        ),
        "receiver_proof_row_count": _int(harvest.get("receiver_proof_row_count")),
        "modelsize_present_row_count": _int(harvest.get("modelsize_present_row_count")),
        "ladder_candidate_row_count": _int(harvest.get("ladder_candidate_row_count")),
        "score_claim": harvest.get("score_claim") is True,
        "promotion_eligible": harvest.get("promotion_eligible") is True,
        "ready_for_exact_eval_dispatch": harvest.get("ready_for_exact_eval_dispatch") is True,
    }


def _merge_harvest_summaries(
    family: str,
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = [summary for summary in summaries if summary.get("carrier_id") == family]
    return {
        "harvest_payload_count": len(selected),
        "source_paths": _ordered_unique(
            str(row["source_path"]) for row in selected if row.get("source_path")
        ),
        "harvested_row_count": sum(_int(row.get("harvested_row_count")) for row in selected),
        "full_scope_row_count": sum(_int(row.get("full_scope_row_count")) for row in selected),
        "local_receiver_replay_row_count": sum(
            _int(row.get("local_receiver_replay_row_count")) for row in selected
        ),
        "receiver_proof_row_count": sum(
            _int(row.get("receiver_proof_row_count")) for row in selected
        ),
        "modelsize_present_row_count": sum(
            _int(row.get("modelsize_present_row_count")) for row in selected
        ),
        "ladder_candidate_row_count": sum(
            _int(row.get("ladder_candidate_row_count")) for row in selected
        ),
    }


def _source_parity_blockers(
    family: str,
    source_parity_contract: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(source_parity_contract, Mapping):
        return ["source_parity_contract_missing"]
    aliases = {family}
    if family == "hinerv":
        aliases.update({"hi_nerv", "hi-nerv"})
    for row in source_parity_contract.get("family_rows", ()):
        if not isinstance(row, Mapping):
            continue
        if _family_key(row.get("family")) in aliases:
            return [str(blocker) for blocker in row.get("blockers", ()) if blocker]
    return ["source_parity_family_row_missing"]


def _next_actions(family_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    actions = []
    for row in family_rows:
        family = row["family"]
        gaps = set(row["emission_gap_ids"])
        if "fewer_than_two_modelsize_or_fc_dim_rows_observed" in gaps:
            actions.append(
                f"{family}: emit modelsize_mparams or fc_dim from trainer/export metadata"
            )
        if "fewer_than_two_full600_rows_observed" in gaps:
            actions.append(f"{family}: run byte-closed full600 receiver replay row producer")
        if "fewer_than_two_receiver_proof_rows_observed" in gaps:
            actions.append(f"{family}: attach receiver archive replay proof before ladder input")
        if row["source_parity_blockers"]:
            actions.append(f"{family}: close source-parity blockers before long training")
    return _ordered_unique(actions)


def _family_key(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text == "hi_nerv":
        return "hinerv"
    return text


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _ordered_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


__all__ = [
    "SCHEMA",
    "build_nerv_ladder_row_emission_contract",
]
