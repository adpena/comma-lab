#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Xray SNeRV receiver decode value-domain pressure on selected pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.substrates.snerv_inverse_steg_carrier.archive import (  # noqa: E402
    OfficialMfuHfrTubReceiverPayload,
    SnervArchiveError,
    _decode_official_mfu_hfr_tub_payload_tensor_manifest,
    _selected_official_mfu_hfr_tub_tensors,
    decode_snerv_archive_pair_frames,
    inspect_decoder_payload_header,
    unpack_snerv_archive,
)

SCHEMA = "snerv_receiver_value_domain_xray.v1"
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument(
        "--pair-indices",
        default="0",
        help="Comma-separated source pair indices to decode, e.g. 0,17,599.",
    )
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument(
        "--official-scalar-skip-high-scan-values",
        help=(
            "Optional comma-separated scalar skip-high values to test against "
            "the official MFU/HFR receiver path as false-authority diagnostic "
            "evidence."
        ),
    )
    return parser


def build_snerv_receiver_value_domain_xray(
    *,
    packet: bytes,
    pair_indices: Sequence[int],
    packet_path: str | Path | None = None,
    profile: Mapping[str, Any] | None = None,
    official_scalar_skip_high_scan_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    decoded = unpack_snerv_archive(packet)
    clean_indices = tuple(int(idx) for idx in pair_indices)
    decoder_header = inspect_decoder_payload_header(decoded.sections["decoder_payload"])
    profile_diagnosis = _profile_scorer_input_diagnosis(profile)
    profile_input_summary = _profile_scorer_input_summary(profile)
    try:
        unclipped = decode_snerv_archive_pair_frames(
            packet,
            clean_indices,
            clip_to_uint8_range=False,
        )
        clipped = decode_snerv_archive_pair_frames(
            packet,
            clean_indices,
            clip_to_uint8_range=True,
        )
    except SnervArchiveError as exc:
        if "selected-frame decode is not supported for official MFU/HFR/TUB" not in str(
            exc
        ):
            raise
        return _official_payload_header_only_report(
            decoded=decoded,
            decoder_header=decoder_header,
            packet=packet,
            pair_indices=clean_indices,
            packet_path=packet_path,
            profile_diagnosis=profile_diagnosis,
            profile_input_summary=profile_input_summary,
            decode_failure=repr(exc),
        )
    clip_delta = np.asarray(clipped, dtype=np.float32) - np.asarray(
        unclipped,
        dtype=np.float32,
    )
    clipped_stats = _array_stats(clipped)
    unclipped_stats = _array_stats(unclipped, include_outside_uint8=True)
    last_frame_clipped_stats = _array_stats(clipped[:, -1])
    last_frame_unclipped_stats = _array_stats(
        unclipped[:, -1],
        include_outside_uint8=True,
    )
    clip_delta_stats = _abs_array_stats(clip_delta)
    unclipped_channel_stats = _channel_stats(unclipped, include_outside_uint8=True)
    clipped_channel_stats = _channel_stats(clipped)
    clip_delta_channel_stats = _abs_channel_stats(clip_delta)
    skip_high_value_domain = _official_skip_high_value_domain_summary(
        decoder_header,
        unclipped_channel_stats=unclipped_channel_stats,
        clip_delta_channel_stats=clip_delta_channel_stats,
    )
    scalar_skip_high_scan = _official_scalar_skip_high_value_domain_scan(
        decoded=decoded,
        decoder_header=decoder_header,
        pair_indices=clean_indices,
        scan_values=official_scalar_skip_high_scan_values,
    )
    blockers = _value_domain_blockers(
        unclipped_stats=unclipped_stats,
        clipped_stats=clipped_stats,
        last_frame_clipped_stats=last_frame_clipped_stats,
        clip_delta_stats=clip_delta_stats,
        skip_high_value_domain=skip_high_value_domain,
        scalar_skip_high_scan=scalar_skip_high_scan,
    )
    noncollapse_passed = not _domain_is_bad(blockers)
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "packet_path": str(packet_path) if packet_path is not None else None,
        "packet_bytes": len(packet),
        "packet_sha256": _sha256_bytes(packet),
        "packet_metadata": _packet_metadata_summary(decoded.metadata),
        "decoder_payload_header": _decoder_payload_header_summary(decoder_header),
        "packet_section_bytes": {
            str(name): len(blob) for name, blob in decoded.sections.items()
        },
        "pair_indices": list(clean_indices),
        "sample_shape_b2chw": list(np.asarray(clipped).shape),
        "value_domain_sample_status": "selected_pair_decode_completed",
        "receiver_payload_decode_sample_proven": True,
        "value_domain_noncollapse_proof_passed": noncollapse_passed,
        "closed_campaign_blockers": _closed_value_domain_blockers(
            noncollapse_passed
        ),
        "unclipped_receiver_stats": unclipped_stats,
        "clipped_receiver_stats": clipped_stats,
        "unclipped_receiver_channel_stats": unclipped_channel_stats,
        "clipped_receiver_channel_stats": clipped_channel_stats,
        "unclipped_last_frame_stats": last_frame_unclipped_stats,
        "clipped_last_frame_stats": last_frame_clipped_stats,
        "clip_delta_abs_stats": clip_delta_stats,
        "clip_delta_abs_channel_stats": clip_delta_channel_stats,
        "official_skip_high_value_domain": skip_high_value_domain,
        "official_scalar_skip_high_value_domain_scan": scalar_skip_high_scan,
        "profile_scorer_input_diagnosis": (
            dict(profile_diagnosis) if isinstance(profile_diagnosis, Mapping) else None
        ),
        "profile_scorer_input_summary": profile_input_summary,
        "verdict": (
            "RECEIVER_VALUE_DOMAIN_OUT_OF_RANGE"
            if _domain_is_bad(blockers)
            else "receiver_value_domain_sample_within_limits"
        ),
        "recommended_next_actions": _recommended_next_actions(blockers),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _official_payload_header_only_report(
    *,
    decoded: Any,
    decoder_header: Mapping[str, Any],
    packet: bytes,
    pair_indices: Sequence[int],
    packet_path: str | Path | None,
    profile_diagnosis: Mapping[str, Any] | None,
    profile_input_summary: Mapping[str, Any] | None,
    decode_failure: str,
) -> dict[str, Any]:
    skip_high_storage = decoder_header.get("skip_high_storage")
    if not isinstance(skip_high_storage, Mapping):
        skip_high_storage = {}
    profile_blockers = _profile_scorer_input_blockers(profile_input_summary)
    blockers = _ordered_unique(
        [
            "snerv_receiver_value_domain_xray_false_authority",
            "snerv_official_payload_selected_pair_value_xray_unavailable",
            *_official_skip_high_storage_blockers(skip_high_storage),
            *profile_blockers,
        ]
    )
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "packet_path": str(packet_path) if packet_path is not None else None,
        "packet_bytes": len(packet),
        "packet_sha256": _sha256_bytes(packet),
        "packet_metadata": _packet_metadata_summary(decoded.metadata),
        "decoder_payload_header": _decoder_payload_header_summary(decoder_header),
        "packet_section_bytes": {
            str(name): len(blob) for name, blob in decoded.sections.items()
        },
        "pair_indices": list(pair_indices),
        "sample_shape_b2chw": None,
        "value_domain_sample_status": "selected_pair_decode_unavailable_for_official_payload",
        "receiver_payload_decode_sample_proven": False,
        "value_domain_noncollapse_proof_passed": False,
        "closed_campaign_blockers": [],
        "decode_failure": decode_failure,
        "profile_scorer_input_diagnosis": (
            dict(profile_diagnosis) if isinstance(profile_diagnosis, Mapping) else None
        ),
        "profile_scorer_input_summary": profile_input_summary,
        "verdict": (
            "OFFICIAL_SKIP_HIGH_SCALAR_MEAN_COLLAPSE_RISK"
            if "snerv_official_skip_high_scalar_mean_receiver_expand_collapse_risk"
            in blockers
            else "OFFICIAL_PAYLOAD_HEADER_ONLY_VALUE_DOMAIN_BLOCKED"
        ),
        "recommended_next_actions": _recommended_next_actions(blockers),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _value_domain_blockers(
    *,
    unclipped_stats: Mapping[str, Any],
    clipped_stats: Mapping[str, Any],
    last_frame_clipped_stats: Mapping[str, Any],
    clip_delta_stats: Mapping[str, Any],
    skip_high_value_domain: Mapping[str, Any] | None,
    scalar_skip_high_scan: Mapping[str, Any] | None,
) -> list[str]:
    blockers = ["snerv_receiver_value_domain_xray_false_authority"]
    if float(unclipped_stats.get("outside_0_255_fraction") or 0.0) > 0.0:
        blockers.append("snerv_receiver_decode_unclipped_outside_uint8_domain")
    if float(clip_delta_stats.get("mean_abs") or 0.0) > 0.0:
        blockers.append("snerv_receiver_decode_clipping_changes_pixels")
    if float(clipped_stats.get("saturation_fraction") or 0.0) >= 0.5:
        blockers.append("snerv_receiver_decode_clipped_output_saturated")
    if float(last_frame_clipped_stats.get("saturation_fraction") or 0.0) >= 0.5:
        blockers.append("snerv_receiver_decode_last_frame_saturated_for_segnet")
    if float(clipped_stats.get("std") or 0.0) <= 1.0e-6:
        blockers.append("snerv_receiver_decode_clipped_output_near_constant")
    if float(last_frame_clipped_stats.get("std") or 0.0) <= 1.0e-6:
        blockers.append("snerv_receiver_decode_last_frame_near_constant_for_segnet")
    if isinstance(skip_high_value_domain, Mapping):
        blockers.extend(
            str(blocker)
            for blocker in skip_high_value_domain.get("blockers") or ()
            if blocker
        )
    if isinstance(scalar_skip_high_scan, Mapping):
        blockers.extend(
            str(blocker)
            for blocker in scalar_skip_high_scan.get("blockers") or ()
            if blocker
        )
    return _ordered_unique(blockers)


def _domain_is_bad(blockers: Sequence[str]) -> bool:
    return any(
        blocker
        for blocker in blockers
        if blocker != "snerv_receiver_value_domain_xray_false_authority"
    )


def _closed_value_domain_blockers(noncollapse_passed: bool) -> list[str]:
    if not noncollapse_passed:
        return []
    return [
        "snerv_official_skip_high_scalar_mean_requires_value_domain_xray_noncollapse",
        "snerv_renderer_nondegenerate_compact_skip_high_value_domain_not_passed",
        "snerv_renderer_nondegenerate_target_value_domain_not_passed",
    ]


def _recommended_next_actions(blockers: Sequence[str]) -> list[str]:
    actions: list[str] = []
    if "snerv_receiver_decode_unclipped_outside_uint8_domain" in blockers:
        actions.append("inspect_lf_zero_step_decoder_scale_before_more_training")
    if "snerv_official_skip_high_scalar_mean_receiver_range_unfit" in blockers:
        actions.append("rerun_bounded_snerv_smoke_with_non_scalar_skip_high_storage")
    if "snerv_official_skip_high_scalar_mean_clipping_delta_unfit" in blockers:
        actions.append("compare_scalar_mean_against_channel_or_shared_skip_high_clip_delta")
    if "snerv_official_skip_high_scalar_mean_channel_range_skew" in blockers:
        actions.append("choose_skip_high_mode_with_channel_or_spatial_state_before_lf_hf_smoke")
    if "snerv_official_scalar_skip_high_no_range_safe_scalar_found" in blockers:
        actions.append("train_or_encode_non_scalar_skip_high_before_range_repair_claim")
    if "snerv_official_scalar_skip_high_range_safe_values_are_degenerate" in blockers:
        actions.append("reject_range_only_scalar_rescue_without_nondegenerate_renderer_proof")
    if "snerv_official_scalar_skip_high_no_value_domain_safe_scalar_found" in blockers:
        actions.append("treat_scalar_skip_high_as_rate_proof_only_and_train_non_scalar_successor")
    if "snerv_receiver_decode_last_frame_saturated_for_segnet" in blockers:
        actions.append("repair_last_frame_receiver_dynamic_range_before_segnet_spend")
    if "snerv_receiver_decode_clipping_changes_pixels" in blockers:
        actions.append("compare_unclipped_to_clipped_receiver_histograms_by_section")
    if "snerv_receiver_decode_clipped_output_near_constant" in blockers:
        actions.append("repair_receiver_value_noncollapse_before_lf_conditioned_hf_spend")
    if "snerv_receiver_decode_last_frame_near_constant_for_segnet" in blockers:
        actions.append("restore_last_frame_contrast_before_lf_conditioned_hf_spend")
    if (
        "snerv_official_skip_high_scalar_mean_receiver_expand_collapse_risk"
        in blockers
    ):
        actions.extend(
            [
                "rerun_snerv_local_export_with_full_or_shared_skip_high_before_exact_eval",
                "compare_scalar_mean_channel_mean_shared_mean_value_domain_locally",
            ]
        )
    if "snerv_profile_segnet_last_frame_saturated" in blockers:
        actions.append("repair_last_frame_receiver_dynamic_range_before_segnet_spend")
    actions.append("keep_exact_eval_dispatch_blocked_until_value_domain_xray_clears")
    return _ordered_unique(actions)


def _official_skip_high_storage_blockers(storage: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    codec = str(storage.get("codec") or "")
    if storage.get("lossless_relative_to_source_skip_high") is False:
        blockers.append("snerv_official_skip_high_not_lossless_relative_to_source")
    if storage.get("receiver_expands_skip_high") is True:
        blockers.append("snerv_official_skip_high_receiver_expands_compact_state")
    if codec == "scalar_mean_float64" or storage.get("stored_shape") == [1, 1, 1, 1]:
        blockers.append("snerv_official_skip_high_scalar_mean_receiver_expand_collapse_risk")
    return blockers


def _official_skip_high_value_domain_summary(
    decoder_header: Mapping[str, Any],
    *,
    unclipped_channel_stats: Sequence[Mapping[str, Any]],
    clip_delta_channel_stats: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    storage = decoder_header.get("skip_high_storage")
    if not isinstance(storage, Mapping):
        return None
    codec = str(storage.get("codec") or "")
    stored_shape = [int(v) for v in storage.get("stored_shape") or ()]
    source_shape = [int(v) for v in storage.get("source_shape") or ()]
    scalar_mean = codec == "scalar_mean_float64" or stored_shape == [1, 1, 1, 1]
    outside_by_channel = [
        _float_or_zero(row.get("outside_0_255_fraction"))
        for row in unclipped_channel_stats
    ]
    clip_mean_by_channel = [
        _float_or_zero(row.get("mean_abs"))
        for row in clip_delta_channel_stats
    ]
    max_outside = max(outside_by_channel, default=0.0)
    min_outside = min(outside_by_channel, default=0.0)
    max_clip_mean = max(clip_mean_by_channel, default=0.0)
    blockers: list[str] = []
    if scalar_mean and max_outside > 0.0:
        blockers.append("snerv_official_skip_high_scalar_mean_receiver_range_unfit")
    if scalar_mean and max_clip_mean > 0.0:
        blockers.append("snerv_official_skip_high_scalar_mean_clipping_delta_unfit")
    if scalar_mean and max_outside > 0.0 and (max_outside - min_outside) >= 0.05:
        blockers.append("snerv_official_skip_high_scalar_mean_channel_range_skew")
    if storage.get("lossless_relative_to_source_skip_high") is False and max_outside > 0.0:
        blockers.append("snerv_official_skip_high_not_lossless_value_domain_unfit")
    return {
        "schema": "snerv_official_skip_high_value_domain_summary.v1",
        "codec": codec or None,
        "stored_shape": stored_shape,
        "source_shape": source_shape,
        "receiver_expands_skip_high": bool(storage.get("receiver_expands_skip_high")),
        "lossless_relative_to_source_skip_high": storage.get(
            "lossless_relative_to_source_skip_high"
        ),
        "stored_raw_bytes": storage.get("stored_raw_bytes"),
        "source_raw_bytes": storage.get("source_raw_bytes"),
        "raw_byte_savings": storage.get("raw_byte_savings"),
        "scalar_mean_storage": scalar_mean,
        "max_unclipped_outside_0_255_fraction_by_channel": max_outside,
        "min_unclipped_outside_0_255_fraction_by_channel": min_outside,
        "max_clip_delta_mean_abs_by_channel": max_clip_mean,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _official_scalar_skip_high_value_domain_scan(
    *,
    decoded: Any,
    decoder_header: Mapping[str, Any],
    pair_indices: Sequence[int],
    scan_values: Sequence[float] | None,
) -> dict[str, Any] | None:
    values = tuple(float(value) for value in (scan_values or ()))
    if not values:
        return None
    storage = decoder_header.get("skip_high_storage")
    if not isinstance(storage, Mapping):
        return None
    codec = str(storage.get("codec") or "")
    stored_shape = [int(v) for v in storage.get("stored_shape") or ()]
    scalar_mean = codec == "scalar_mean_float64" or stored_shape == [1, 1, 1, 1]
    if not scalar_mean:
        return None

    metadata = decoded.metadata if isinstance(decoded.metadata, Mapping) else {}
    n_pairs = int(metadata.get("n_pairs") or 0)
    frames_per_pair = int(metadata.get("frames_per_pair") or 2)
    expected_frame_count = int(n_pairs * frames_per_pair)
    frame_indices = tuple(
        int(pair_index) * int(frames_per_pair) + frame_index
        for pair_index in pair_indices
        for frame_index in range(frames_per_pair)
    )
    try:
        header, tensors = _decode_official_mfu_hfr_tub_payload_tensor_manifest(
            decoded.sections["decoder_payload"]
        )
        selected_tensors = _selected_official_mfu_hfr_tub_tensors(
            header,
            tensors,
            selected_frame_indices=frame_indices,
            expected_frame_count=expected_frame_count,
        )
        skip_shape = tuple(int(v) for v in selected_tensors["inputs.mfu.skip_high"].shape)
        rows = [
            _scalar_skip_high_scan_row(
                header=header,
                selected_tensors=selected_tensors,
                skip_shape=skip_shape,
                scalar_value=value,
            )
            for value in values
        ]
    except Exception as exc:  # pragma: no cover - diagnostic fail-closed path
        return {
            "schema": "snerv_official_scalar_skip_high_value_domain_scan.v1",
            "scan_executed": False,
            "scan_failure": repr(exc),
            "scalar_values": list(values),
            "blockers": [
                "snerv_official_scalar_skip_high_value_domain_scan_failed",
                f"snerv_official_scalar_skip_high_value_domain_scan_failed_{type(exc).__name__}",
            ],
            **FALSE_AUTHORITY,
        }

    best = min(
        rows,
        key=lambda row: (
            float(row["outside_0_255_fraction"]),
            float(row["out_of_range_magnitude"]),
            -float(row["std"]),
        ),
    )
    range_safe_rows = [row for row in rows if row["range_passed"] is True]
    safe_rows = [row for row in rows if row["value_domain_passed"] is True]
    blockers: list[str] = []
    if not range_safe_rows:
        blockers.append("snerv_official_scalar_skip_high_no_range_safe_scalar_found")
    elif not safe_rows:
        blockers.append("snerv_official_scalar_skip_high_range_safe_values_are_degenerate")
    if not safe_rows:
        blockers.append("snerv_official_scalar_skip_high_no_value_domain_safe_scalar_found")
    if best["outside_0_255_fraction"] > 0.0:
        blockers.append("snerv_official_scalar_skip_high_scan_best_still_outside_uint8")
    return {
        "schema": "snerv_official_scalar_skip_high_value_domain_scan.v1",
        "scan_executed": True,
        "scan_false_authority": True,
        "pair_indices": [int(v) for v in pair_indices],
        "frame_indices": [int(v) for v in frame_indices],
        "scalar_values": list(values),
        "range_safe_scalar_value_count": len(range_safe_rows),
        "range_safe_scalar_values": [
            float(row["scalar_value"]) for row in range_safe_rows
        ],
        "safe_scalar_value_count": len(safe_rows),
        "safe_scalar_values": [float(row["scalar_value"]) for row in safe_rows],
        "best_scalar_row": dict(best),
        "rows": rows,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _scalar_skip_high_scan_row(
    *,
    header: Mapping[str, Any],
    selected_tensors: Mapping[str, np.ndarray],
    skip_shape: Sequence[int],
    scalar_value: float,
) -> dict[str, Any]:
    tensors = dict(selected_tensors)
    tensors["inputs.mfu.skip_high"] = np.full(
        tuple(int(v) for v in skip_shape),
        float(scalar_value),
        dtype=np.float64,
    )
    payload = OfficialMfuHfrTubReceiverPayload(
        header=dict(header),
        tensors=tensors,
        payload_sha256="diagnostic_scalar_skip_high_scan",
        payload_bytes=0,
    )
    frames = payload.decode_frames(clip_to_uint8_range=False)
    stats = _array_stats(frames, include_outside_uint8=True)
    out_of_range = _out_of_range_magnitude(stats)
    std = float(stats.get("std") or 0.0)
    outside = float(stats.get("outside_0_255_fraction") or 0.0)
    range_passed = bool(outside == 0.0 and out_of_range == 0.0)
    nonconstant_passed = bool(std > 1.0e-6)
    passed = bool(range_passed and nonconstant_passed)
    return {
        "scalar_value": float(scalar_value),
        "range_passed": range_passed,
        "nonconstant_passed": nonconstant_passed,
        "value_domain_passed": passed,
        "out_of_range_magnitude": out_of_range,
        **stats,
    }


def _out_of_range_magnitude(stats: Mapping[str, Any]) -> float:
    lo = _float_or_zero(stats.get("min"))
    hi = _float_or_zero(stats.get("max"))
    return float(max(0.0, -lo) + max(0.0, hi - 255.0))


def _profile_scorer_input_diagnosis(
    profile: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(profile, Mapping):
        return None
    diagnosis = profile.get("scorer_input_diagnosis")
    return diagnosis if isinstance(diagnosis, Mapping) else None


def _profile_scorer_input_summary(
    profile: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(profile, Mapping):
        return None
    distribution = profile.get("scorer_input_distribution")
    if not isinstance(distribution, Mapping):
        return None
    keys = (
        "candidate_segnet_last_rgb",
        "reference_segnet_last_rgb",
        "segnet_last_rgb_absdiff",
        "candidate_posenet_yuv6_pair",
        "reference_posenet_yuv6_pair",
        "posenet_yuv6_pair_absdiff",
    )
    return {
        key: dict(distribution[key])
        for key in keys
        if isinstance(distribution.get(key), Mapping)
    }


def _profile_scorer_input_blockers(
    profile_input_summary: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(profile_input_summary, Mapping):
        return []
    blockers: list[str] = []
    seg = profile_input_summary.get("candidate_segnet_last_rgb")
    pose = profile_input_summary.get("candidate_posenet_yuv6_pair")
    seg_abs = profile_input_summary.get("segnet_last_rgb_absdiff")
    pose_abs = profile_input_summary.get("posenet_yuv6_pair_absdiff")
    if isinstance(seg, Mapping) and float(seg.get("saturation_fraction") or 0.0) >= 0.5:
        blockers.append("snerv_profile_segnet_last_frame_saturated")
    if isinstance(pose, Mapping) and float(pose.get("saturation_fraction") or 0.0) >= 0.5:
        blockers.append("snerv_profile_posenet_yuv6_pair_saturated")
    if isinstance(seg_abs, Mapping) and float(seg_abs.get("mean_abs") or 0.0) > 50.0:
        blockers.append("snerv_profile_segnet_last_frame_mean_absdiff_gt_50")
    if isinstance(pose_abs, Mapping) and float(pose_abs.get("mean_abs") or 0.0) > 50.0:
        blockers.append("snerv_profile_posenet_yuv6_pair_mean_absdiff_gt_50")
    return blockers


def _packet_metadata_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "n_pairs",
        "frames_per_pair",
        "channels",
        "height",
        "width",
        "levels",
        "wavelet",
        "lf_payload_codec",
        "decoder_payload_schema",
    )
    return {key: metadata.get(key) for key in keys if key in metadata}


def _decoder_payload_header_summary(header: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "skip_high_storage",
        "tub_input_storage",
        "official_runtime_contract",
    )
    return {key: header.get(key) for key in keys if key in header}


def _array_stats(array: np.ndarray, *, include_outside_uint8: bool = False) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        base: dict[str, Any] = {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "saturation_fraction": None,
        }
        if include_outside_uint8:
            base["outside_0_255_fraction"] = None
        return base
    saturated = np.count_nonzero((arr <= 0.5) | (arr >= 254.5))
    out = {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr, dtype=np.float64)),
        "std": float(np.std(arr, dtype=np.float64)),
        "saturation_fraction": float(saturated) / float(arr.size),
    }
    if include_outside_uint8:
        outside = np.count_nonzero((arr < 0.0) | (arr > 255.0))
        out["outside_0_255_fraction"] = float(outside) / float(arr.size)
    return out


def _channel_stats(
    array: np.ndarray,
    *,
    include_outside_uint8: bool = False,
) -> list[dict[str, Any]]:
    arr = np.asarray(array, dtype=np.float32)
    channel_axis = _receiver_channel_axis(arr)
    if channel_axis is None:
        return []
    moved = np.moveaxis(arr, channel_axis, 0)
    return [
        {
            "channel_index": int(channel_index),
            **_array_stats(channel, include_outside_uint8=include_outside_uint8),
        }
        for channel_index, channel in enumerate(moved)
    ]


def _abs_array_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.abs(np.asarray(array, dtype=np.float32))
    if arr.size == 0:
        return {"count": 0, "mean_abs": None, "max_abs": None}
    return {
        "count": int(arr.size),
        "mean_abs": float(np.mean(arr, dtype=np.float64)),
        "max_abs": float(np.max(arr)),
    }


def _abs_channel_stats(array: np.ndarray) -> list[dict[str, Any]]:
    arr = np.asarray(array, dtype=np.float32)
    channel_axis = _receiver_channel_axis(arr)
    if channel_axis is None:
        return []
    moved = np.moveaxis(arr, channel_axis, 0)
    return [
        {"channel_index": int(channel_index), **_abs_array_stats(channel)}
        for channel_index, channel in enumerate(moved)
    ]


def _receiver_channel_axis(array: np.ndarray) -> int | None:
    if array.ndim >= 5:
        return 2
    if array.ndim >= 3:
        return array.ndim - 3
    return None


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_pair_indices(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("--pair-indices must contain at least one integer")
    return values


def _parse_float_values(raw: str | None) -> tuple[float, ...] | None:
    if raw is None or not str(raw).strip():
        return None
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("scalar scan values must contain at least one float")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("scalar scan values must be finite")
    return values


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _ordered_unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    packet = args.packet.read_bytes()
    report = build_snerv_receiver_value_domain_xray(
        packet=packet,
        pair_indices=_parse_pair_indices(args.pair_indices),
        packet_path=args.packet,
        profile=_load_json(args.profile_json),
        official_scalar_skip_high_scan_values=_parse_float_values(
            args.official_scalar_skip_high_scan_values
        ),
    )
    _write_json(args.output_json, report)
    print(json.dumps({"report_path": args.output_json.as_posix(), **report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
