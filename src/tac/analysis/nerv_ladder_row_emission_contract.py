# SPDX-License-Identifier: MIT
"""Emission contract for receiver-closed SNeRV/HiNeRV modelsize rows."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
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
            "byte_closed_receiver_proof",
            "receiver_proof_passed",
            "receiver_closed",
        ),
        "why": "raw replay booleans are local evidence only; ladder authority requires byte-closed receiver proof",
    },
    {
        "group_id": "receiver_proof_identity",
        "requirement": "all_of",
        "paths": (
            "receiver_proof_identity_bound",
            "receiver_proof_path",
            "receiver_proof_sha256",
        ),
        "why": "file-backed proof identity prevents advisory replay flags from masquerading as receiver-closed evidence",
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
    packet_probe_artifacts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a fail-closed contract for trainer/export row emission."""

    selected = tuple(dict.fromkeys(_family_key(family) for family in families))
    harvest_summaries = [_harvest_summary(harvest) for harvest in row_harvests]
    packet_summaries = [
        _packet_probe_summary(artifact) for artifact in packet_probe_artifacts
    ]
    family_rows = [
        _family_contract_row(
            family,
            source_parity_contract=source_parity_contract,
            harvest_summaries=harvest_summaries,
            packet_summaries=packet_summaries,
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
        "receiver_packet_probe_summaries": packet_summaries,
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
    packet_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_blockers = _source_parity_blockers(family, source_parity_contract)
    harvest = _merge_harvest_summaries(family, harvest_summaries, packet_summaries)
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
    if harvest.get("schema") == "hinerv_archive_size_ladder.v1":
        return _hinerv_archive_size_ladder_summary(harvest)
    carrier = _family_key(
        harvest.get("carrier_id") or harvest.get("family") or "unknown"
    )
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


def _hinerv_archive_size_ladder_summary(harvest: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in harvest.get("archive_rows", ())
        if isinstance(row, Mapping)
    ]
    num_pairs = _int(harvest.get("num_pairs"))
    full_scope = num_pairs >= 600
    receiver_proof_rows = [
        row for row in rows if row.get("runtime_consumption_proof_ready") is True
    ]
    modelsize_rows = [
        row
        for row in rows
        if _int(row.get("num_parameters")) > 0 or isinstance(row.get("config"), Mapping)
    ]
    ladder_rows = [
        row
        for row in rows
        if full_scope
        and row.get("runtime_consumption_proof_ready") is True
        and _int(row.get("archive_bytes")) > 0
        and bool(row.get("archive_sha256"))
        and _int(row.get("num_parameters")) > 0
        and _row_has_nonrate_distortion(row)
        and not row.get("backend_claim_blockers")
    ]
    return {
        "schema": harvest.get("schema"),
        "carrier_id": _family_key(harvest.get("family") or "hinerv"),
        "status": "hinerv_archive_size_ladder_observed_false_authority",
        "source_path": harvest.get("source_artifact_path") or harvest.get("report_path"),
        "harvested_row_count": len(rows),
        "full_scope_row_count": len(rows) if full_scope else 0,
        "local_receiver_replay_row_count": len(receiver_proof_rows),
        "receiver_proof_row_count": len(receiver_proof_rows),
        "modelsize_present_row_count": len(modelsize_rows),
        "ladder_candidate_row_count": len(ladder_rows),
        "archive_export_backend_counts": dict(
            harvest.get("archive_export_backend_counts") or {}
        ),
        "row_blockers": _ordered_unique(
            blocker
            for row in rows
            for blocker in row.get("blockers", ())
            if blocker
        ),
        "report_blockers": list(harvest.get("blockers") or ()),
        "score_claim": harvest.get("score_claim") is True,
        "promotion_eligible": harvest.get("promotion_eligible") is True,
        "ready_for_exact_eval_dispatch": harvest.get("ready_for_exact_eval_dispatch") is True,
    }


def _row_has_nonrate_distortion(row: Mapping[str, Any]) -> bool:
    return any(
        key in row and row.get(key) is not None
        for key in (
            "nonrate_score",
            "nonrate_score_value",
            "d_seg_mean_linf",
            "d_pose_mean_linf",
            "avg_segnet_dist",
            "avg_posenet_dist",
        )
    )


def _packet_probe_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    carrier = _packet_probe_carrier(artifact)
    source_path = artifact.get("source_artifact_path") or artifact.get("report_path")
    candidates = [
        candidate
        for candidate in artifact.get("candidates", ())
        if isinstance(candidate, Mapping)
    ]
    packet_rows = [
        _packet_export_row(candidate)
        for candidate in candidates
        if isinstance(candidate.get("receiver_archive_packet_export"), Mapping)
    ]
    verified_rows = [
        row
        for row in packet_rows
        if row["file_exists"]
        and row["sha256_matches_export"]
        and row["bytes_match_export"]
        and row["expected_sha256_matches"]
    ]
    return {
        "schema": "nerv_receiver_packet_probe_summary.v1",
        "source_schema": artifact.get("schema"),
        "carrier_id": carrier,
        "source_path": source_path,
        "axis_tag": artifact.get("axis_tag"),
        "n_pairs": _int(artifact.get("n_pairs")),
        "candidate_count": len(candidates),
        "receiver_packet_export_count": len(packet_rows),
        "receiver_packet_export_verified_count": len(verified_rows),
        "local_receiver_replay_verified_count": sum(
            1
            for candidate in candidates
            if candidate.get("receiver_archive_replay_verified") is True
        ),
        "contest_archive_zip_export_count": sum(
            1 for row in packet_rows if row["contest_archive_zip"]
        ),
        "receiver_packet_export_bytes_total": sum(
            row["bytes_actual"] for row in verified_rows
        ),
        "full600_receiver_packet_export_count": (
            len(verified_rows) if _int(artifact.get("n_pairs")) >= 600 else 0
        ),
        "packet_rows": packet_rows,
        "blockers": _ordered_unique(
            [
                *(
                    ("packet_probe_carrier_missing_or_unknown",)
                    if carrier == "unknown"
                    else ()
                ),
                *(
                    "packet_export:" + blocker
                    for row in packet_rows
                    for blocker in row["blockers"]
                ),
                *(
                    str(blocker)
                    for candidate in candidates
                    for blocker in candidate.get("blockers", ())
                    if blocker
                ),
            ]
        ),
        **FALSE_AUTHORITY,
    }


def _packet_probe_carrier(artifact: Mapping[str, Any]) -> str:
    explicit = artifact.get("family") or artifact.get("carrier_id")
    if explicit:
        return _family_key(explicit)
    schema = str(artifact.get("schema") or "").lower()
    source = str(
        artifact.get("source_artifact_path") or artifact.get("report_path") or ""
    ).lower()
    haystack = f"{schema} {source}"
    if "hinerv" in haystack or "hi_nerv" in haystack or "hi-nerv" in haystack:
        return "hinerv"
    if "snerv" in haystack:
        return "snerv"
    return "unknown"


def _packet_export_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    export = candidate.get("receiver_archive_packet_export")
    if not isinstance(export, Mapping):
        export = {}
    path_text = str(export.get("path") or "")
    path = Path(path_text).expanduser() if path_text else None
    file_exists = bool(path and path.is_file())
    bytes_declared = _int(export.get("bytes"))
    sha_declared = str(export.get("sha256") or "")
    expected_sha = str(export.get("expected_sha256") or "")
    bytes_actual = path.stat().st_size if file_exists and path is not None else 0
    sha_actual = _sha256_file(path) if file_exists and path is not None else ""
    blockers = []
    if not path_text:
        blockers.append("receiver_packet_export_path_missing")
    if not file_exists:
        blockers.append("receiver_packet_export_file_missing")
    if file_exists and not bytes_declared:
        blockers.append("receiver_packet_export_bytes_missing")
    if file_exists and not sha_declared:
        blockers.append("receiver_packet_export_sha256_missing")
    if file_exists and bytes_declared and bytes_actual != bytes_declared:
        blockers.append("receiver_packet_export_bytes_mismatch")
    if file_exists and sha_declared and sha_actual != sha_declared:
        blockers.append("receiver_packet_export_sha256_mismatch")
    if expected_sha and not sha_declared:
        blockers.append("receiver_packet_export_expected_sha256_without_sha256")
    if expected_sha and sha_declared and expected_sha != sha_declared:
        blockers.append("receiver_packet_export_expected_sha256_mismatch")
    if export.get("contest_archive_zip") is True:
        blockers.append("receiver_packet_export_unexpected_contest_archive_zip")
    return {
        "label": candidate.get("label"),
        "path": path_text or None,
        "kind": export.get("kind"),
        "bytes_declared": bytes_declared,
        "bytes_actual": bytes_actual,
        "sha256_declared": sha_declared or None,
        "sha256_actual": sha_actual or None,
        "expected_sha256": expected_sha or None,
        "file_exists": file_exists,
        "bytes_match_export": bool(
            file_exists and bool(bytes_declared) and bytes_actual == bytes_declared
        ),
        "sha256_matches_export": bool(
            file_exists and bool(sha_declared) and sha_actual == sha_declared
        ),
        "expected_sha256_matches": bool(
            not expected_sha or (bool(sha_declared) and expected_sha == sha_declared)
        ),
        "contest_archive_zip": export.get("contest_archive_zip") is True,
        "blockers": blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _merge_harvest_summaries(
    family: str,
    summaries: Sequence[Mapping[str, Any]],
    packet_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = [summary for summary in summaries if summary.get("carrier_id") == family]
    selected_packets = [
        summary for summary in packet_summaries if summary.get("carrier_id") == family
    ]
    packet_verified_count = sum(
        _int(row.get("receiver_packet_export_verified_count"))
        for row in selected_packets
    )
    return {
        "harvest_payload_count": len(selected),
        "source_paths": _ordered_unique(
            [
                *(str(row["source_path"]) for row in selected if row.get("source_path")),
                *(
                    str(row["source_path"])
                    for row in selected_packets
                    if row.get("source_path")
                ),
            ]
        ),
        "receiver_packet_probe_payload_count": len(selected_packets),
        "receiver_packet_export_count": sum(
            _int(row.get("receiver_packet_export_count")) for row in selected_packets
        ),
        "receiver_packet_export_verified_count": packet_verified_count,
        "receiver_packet_export_bytes_total": sum(
            _int(row.get("receiver_packet_export_bytes_total"))
            for row in selected_packets
        ),
        "full600_receiver_packet_export_count": sum(
            _int(row.get("full600_receiver_packet_export_count"))
            for row in selected_packets
        ),
        "harvested_row_count": sum(_int(row.get("harvested_row_count")) for row in selected),
        "full_scope_row_count": sum(_int(row.get("full_scope_row_count")) for row in selected),
        "local_receiver_replay_row_count": sum(
            _int(row.get("local_receiver_replay_row_count")) for row in selected
        )
        + packet_verified_count,
        "receiver_proof_row_count": sum(
            _int(row.get("receiver_proof_row_count")) for row in selected
        ),
        "modelsize_present_row_count": sum(
            _int(row.get("modelsize_present_row_count")) for row in selected
        ),
        "ladder_candidate_row_count": sum(
            _int(row.get("ladder_candidate_row_count")) for row in selected
        ),
        "archive_export_backend_counts": _merge_count_maps(
            row.get("archive_export_backend_counts") for row in selected
        ),
        "row_blockers": _ordered_unique(
            blocker
            for row in selected
            for blocker in row.get("row_blockers", ())
            if blocker
        ),
        "report_blockers": _ordered_unique(
            blocker
            for row in selected
            for blocker in row.get("report_blockers", ())
            if blocker
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge_count_maps(values: Iterable[Any]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for value in values:
        if not isinstance(value, Mapping):
            continue
        for key, count in value.items():
            text = str(key)
            merged[text] = merged.get(text, 0) + _int(count)
    return merged


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
