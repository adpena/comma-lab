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
    SnervArchiveError,
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
    return parser


def build_snerv_receiver_value_domain_xray(
    *,
    packet: bytes,
    pair_indices: Sequence[int],
    packet_path: str | Path | None = None,
    profile: Mapping[str, Any] | None = None,
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
    blockers = _value_domain_blockers(
        unclipped_stats=unclipped_stats,
        clipped_stats=clipped_stats,
        last_frame_clipped_stats=last_frame_clipped_stats,
        clip_delta_stats=clip_delta_stats,
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
        "unclipped_last_frame_stats": last_frame_unclipped_stats,
        "clipped_last_frame_stats": last_frame_clipped_stats,
        "clip_delta_abs_stats": clip_delta_stats,
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
    if "snerv_receiver_decode_last_frame_saturated_for_segnet" in blockers:
        actions.append("repair_last_frame_receiver_dynamic_range_before_segnet_spend")
    if "snerv_receiver_decode_clipping_changes_pixels" in blockers:
        actions.append("compare_unclipped_to_clipped_receiver_histograms_by_section")
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


def _abs_array_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.abs(np.asarray(array, dtype=np.float32))
    if arr.size == 0:
        return {"count": 0, "mean_abs": None, "max_abs": None}
    return {
        "count": int(arr.size),
        "mean_abs": float(np.mean(arr, dtype=np.float64)),
        "max_abs": float(np.max(arr)),
    }


def _parse_pair_indices(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("--pair-indices must contain at least one integer")
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
    )
    _write_json(args.output_json, report)
    print(json.dumps({"report_path": args.output_json.as_posix(), **report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
