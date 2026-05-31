#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Byte profile a Z8HPC1 archive and rank rate-axis opportunities.

This is a read-only, non-promotable forensic tool for the hierarchical
predictive-coding lane. It profiles the byte stack that matters for contest
rate: ZIP wrapper, Z8HPC1 sections, wavelet pair blobs, detail codec choices,
top-LL float payloads, optional full-video entropy-headroom curves, and local
replay rate terms. It makes no score claim; it turns "archive too big" into a
machine-readable compression backlog.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import math
import statistics
import struct
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import brotli  # type: ignore[import-not-found]
import numpy as np

from tac.substrates.z8_hierarchical_predictive_coding.archive import (
    parse_archive,
    parse_z8hpc1_archive_bytes,
)
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    _DETAIL_CODEC_NAMES,
    _PAIR_BLOB_LOSSLESS_PRECONDITIONED_DETAIL_SCHEMA_VERSION,
    _PAIR_BLOB_QUANTIZED_DETAIL_SCHEMA_VERSION,
    _PAIR_BLOB_SCHEMA_VERSION,
)

SCHEMA = "z8_hpc_archive_byte_profile.v1"
TOOL = "tools/profile_z8_hpc_archive_bytes.py"

NON_PROMOTABLE_MARKERS: dict[str, Any] = {
    "axis_tag": "[macOS-CPU advisory]",
    "evidence_grade": "macOS-CPU-advisory",
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _entropy_order0_bytes(data: bytes) -> float:
    if not data:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    probs = counts[counts > 0] / float(arr.size)
    bits = float(-np.sum(probs * np.log2(probs)) * arr.size)
    return bits / 8.0


def _codec_profile(data: bytes, *, brotli_quality: int) -> dict[str, Any]:
    """Small deterministic codec probes for a byte section.

    These are diagnostics, not replacement claims. They reveal whether a section
    is already entropy-like or still has easy generic-compressor structure.
    """

    out: dict[str, Any] = {
        "raw_bytes": len(data),
        "order0_entropy_floor_bytes": round(_entropy_order0_bytes(data), 1),
        "zlib9_bytes": len(zlib.compress(data, level=9)),
        "bz2_9_bytes": len(bz2.compress(data, compresslevel=9)),
        "brotli_bytes": len(brotli.compress(data, quality=brotli_quality)),
        "brotli_quality": int(brotli_quality),
    }
    try:
        import zstandard as zstd  # type: ignore[import-not-found]

        out["zstd19_bytes"] = len(zstd.ZstdCompressor(level=19).compress(data))
    except Exception:
        out["zstd19_bytes"] = None
    return out


def _stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    vals = sorted(int(v) for v in values)
    return {
        "count": len(vals),
        "min": vals[0],
        "p50": vals[len(vals) // 2],
        "p90": vals[min(len(vals) - 1, math.ceil(0.90 * len(vals)) - 1)],
        "p99": vals[min(len(vals) - 1, math.ceil(0.99 * len(vals)) - 1)],
        "max": vals[-1],
        "mean": round(float(statistics.fmean(vals)), 2),
        "total": int(sum(vals)),
    }


def _parse_pair_raw(raw: bytes) -> dict[str, Any]:
    pos = 0
    (version,) = struct.unpack("<B", raw[pos : pos + 1])
    pos += 1
    if version not in {
        _PAIR_BLOB_SCHEMA_VERSION,
        _PAIR_BLOB_QUANTIZED_DETAIL_SCHEMA_VERSION,
        _PAIR_BLOB_LOSSLESS_PRECONDITIONED_DETAIL_SCHEMA_VERSION,
    }:
        raise ValueError(f"unsupported pair blob schema version {version}")

    top_ll_bytes = 0
    top_ll_shapes: list[tuple[int, int, int]] = []
    top_ll_header_bytes = 0
    for _frame_key in ("frame_0_top_ll", "frame_1_top_ll"):
        h, w, c = struct.unpack("<HHH", raw[pos : pos + 6])
        pos += 6
        nbytes = int(h) * int(w) * int(c) * 4
        pos += nbytes
        top_ll_header_bytes += 6
        top_ll_bytes += nbytes
        top_ll_shapes.append((int(h), int(w), int(c)))

    detail_rows: list[dict[str, Any]] = []
    detail_list_header_bytes = 0
    detail_shape_header_bytes = 0
    detail_codec_header_bytes = 0
    for details_key in ("frame_0_details", "frame_1_details"):
        (num_levels,) = struct.unpack("<B", raw[pos : pos + 1])
        pos += 1
        detail_list_header_bytes += 1
        for level_idx in range(int(num_levels)):
            for subband_key in ("lh", "hl", "hh"):
                h, w, c = struct.unpack("<HHH", raw[pos : pos + 6])
                pos += 6
                detail_shape_header_bytes += 6
                count = int(h) * int(w) * int(c)
                q_step: float | None = None
                if version == _PAIR_BLOB_SCHEMA_VERSION:
                    method_name = "float32_raw"
                    payload_len = count * 4
                    pos += payload_len
                elif version == _PAIR_BLOB_QUANTIZED_DETAIL_SCHEMA_VERSION:
                    method, q_step, payload_len = struct.unpack("<BfI", raw[pos : pos + 9])
                    pos += 9
                    detail_codec_header_bytes += 9
                    method_name = _DETAIL_CODEC_NAMES.get(int(method), f"unknown_{int(method)}")
                    pos += int(payload_len)
                else:
                    method, payload_len = struct.unpack("<BI", raw[pos : pos + 5])
                    pos += 5
                    detail_codec_header_bytes += 5
                    method_name = _DETAIL_CODEC_NAMES.get(int(method), f"unknown_{int(method)}")
                    pos += int(payload_len)
                detail_rows.append(
                    {
                        "frame": details_key,
                        "level": int(level_idx),
                        "subband": subband_key,
                        "shape": [int(h), int(w), int(c)],
                        "coefficients": int(count),
                        "method": method_name,
                        "quantization_step": None if q_step is None else float(q_step),
                        "payload_bytes": int(payload_len),
                    }
                )
    if pos != len(raw):
        raise ValueError(f"pair blob trailing bytes (pos={pos} len={len(raw)})")

    return {
        "schema_version": int(version),
        "top_ll_bytes": int(top_ll_bytes),
        "top_ll_shapes": top_ll_shapes,
        "top_ll_header_bytes": int(top_ll_header_bytes),
        "detail_list_header_bytes": int(detail_list_header_bytes),
        "detail_shape_header_bytes": int(detail_shape_header_bytes),
        "detail_codec_header_bytes": int(detail_codec_header_bytes),
        "detail_rows": detail_rows,
        "raw_bytes": len(raw),
    }


def _wavelet_blob_profile(
    wavelet_blob: bytes,
    *,
    measure_solid_pair_brotli: bool,
    solid_brotli_quality: int,
) -> dict[str, Any]:
    pos = 0
    (num_pairs,) = struct.unpack("<I", wavelet_blob[pos : pos + 4])
    pos += 4

    schema_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    method_payload_bytes: Counter[str] = Counter()
    q_step_counts: Counter[str] = Counter()
    by_subband: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "coefficients": 0,
            "payload_bytes": 0,
            "method_counts": Counter(),
            "method_payload_bytes": Counter(),
            "quantization_step_counts": Counter(),
        }
    )
    pair_blob_lengths: list[int] = []
    pair_raw_lengths: list[int] = []
    top_ll_bytes = 0
    top_ll_header_bytes = 0
    detail_payload_bytes = 0
    detail_shape_header_bytes = 0
    detail_codec_header_bytes = 0
    detail_list_header_bytes = 0
    detail_coefficients = 0
    raw_pair_blobs: list[bytes] = []

    for _pair_idx in range(int(num_pairs)):
        (blob_len,) = struct.unpack("<I", wavelet_blob[pos : pos + 4])
        pos += 4
        pair_blob = wavelet_blob[pos : pos + int(blob_len)]
        pos += int(blob_len)
        raw = brotli.decompress(pair_blob)
        parsed = _parse_pair_raw(raw)
        schema_counts[str(parsed["schema_version"])] += 1
        pair_blob_lengths.append(int(blob_len))
        pair_raw_lengths.append(len(raw))
        if measure_solid_pair_brotli:
            raw_pair_blobs.append(raw)

        top_ll_bytes += int(parsed["top_ll_bytes"])
        top_ll_header_bytes += int(parsed["top_ll_header_bytes"])
        detail_list_header_bytes += int(parsed["detail_list_header_bytes"])
        detail_shape_header_bytes += int(parsed["detail_shape_header_bytes"])
        detail_codec_header_bytes += int(parsed["detail_codec_header_bytes"])

        for row in parsed["detail_rows"]:
            method = str(row["method"])
            payload = int(row["payload_bytes"])
            coeffs = int(row["coefficients"])
            detail_payload_bytes += payload
            detail_coefficients += coeffs
            method_counts[method] += 1
            method_payload_bytes[method] += payload
            if row["quantization_step"] is not None:
                q_step_counts[str(row["quantization_step"])] += 1
            sub_key = f"{row['frame']}:L{row['level']}:{row['subband']}"
            sub = by_subband[sub_key]
            sub["count"] += 1
            sub["coefficients"] += coeffs
            sub["payload_bytes"] += payload
            sub["method_counts"][method] += 1
            sub["method_payload_bytes"][method] += payload
            if row["quantization_step"] is not None:
                sub["quantization_step_counts"][str(row["quantization_step"])] += 1

    if pos != len(wavelet_blob):
        raise ValueError(f"wavelet_blob trailing bytes (pos={pos} len={len(wavelet_blob)})")

    by_subband_json = []
    for key in sorted(by_subband):
        row = by_subband[key]
        by_subband_json.append(
            {
                "key": key,
                "count": int(row["count"]),
                "coefficients": int(row["coefficients"]),
                "payload_bytes": int(row["payload_bytes"]),
                "bytes_per_coeff": (
                    round(float(row["payload_bytes"]) / float(row["coefficients"]), 6)
                    if row["coefficients"]
                    else 0.0
                ),
                "method_counts": dict(sorted(row["method_counts"].items())),
                "method_payload_bytes": dict(sorted(row["method_payload_bytes"].items())),
                "quantization_step_counts": dict(sorted(row["quantization_step_counts"].items())),
            }
        )

    length_prefix_bytes = 4 + int(num_pairs) * 4
    profile: dict[str, Any] = {
        "schema": "z8_wavelet_blob_byte_profile.v1",
        "num_pairs": int(num_pairs),
        "wavelet_blob_bytes": len(wavelet_blob),
        "length_prefix_bytes": length_prefix_bytes,
        "pair_blob_compressed_bytes": int(sum(pair_blob_lengths)),
        "pair_blob_compressed_length_stats": _stats(pair_blob_lengths),
        "pair_blob_raw_length_stats": _stats(pair_raw_lengths),
        "pair_blob_schema_counts": dict(sorted(schema_counts.items())),
        "top_ll_raw_payload_bytes": int(top_ll_bytes),
        "top_ll_header_bytes": int(top_ll_header_bytes),
        "detail_payload_bytes": int(detail_payload_bytes),
        "detail_shape_header_bytes": int(detail_shape_header_bytes),
        "detail_codec_header_bytes": int(detail_codec_header_bytes),
        "detail_list_header_bytes": int(detail_list_header_bytes),
        "detail_coefficients": int(detail_coefficients),
        "detail_payload_bytes_per_coeff": (
            round(detail_payload_bytes / detail_coefficients, 6) if detail_coefficients else 0.0
        ),
        "detail_codec_method_counts": dict(sorted(method_counts.items())),
        "detail_codec_payload_bytes": dict(sorted(method_payload_bytes.items())),
        "quantization_step_counts": dict(sorted(q_step_counts.items())),
        "by_frame_level_subband": by_subband_json,
    }
    if measure_solid_pair_brotli:
        solid_raw = b"".join(raw_pair_blobs)
        profile["solid_pair_raw_bytes"] = len(solid_raw)
        profile["solid_pair_raw_brotli_quality"] = int(solid_brotli_quality)
        profile["solid_pair_raw_brotli_bytes"] = len(
            brotli.compress(solid_raw, quality=int(solid_brotli_quality))
        )
        profile["solid_pair_raw_brotli_delta_vs_current_pair_blobs"] = (
            int(profile["solid_pair_raw_brotli_bytes"]) - int(sum(pair_blob_lengths))
        )
    return profile


def _zip_profile(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    rows: list[dict[str, Any]] = []
    total_compressed = 0
    total_uncompressed = 0
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            member_sha256 = _sha256_bytes(zf.read(info.filename))
            rows.append(
                {
                    "filename": info.filename,
                    "compress_type": int(info.compress_type),
                    "compress_size": int(info.compress_size),
                    "file_size": int(info.file_size),
                    "sha256": member_sha256,
                    "crc": f"{int(info.CRC):08x}",
                    "flag_bits": int(info.flag_bits),
                    "header_offset": int(info.header_offset),
                }
            )
            total_compressed += int(info.compress_size)
            total_uncompressed += int(info.file_size)
    archive_bytes = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(archive_bytes),
        "sha256": _sha256_bytes(archive_bytes),
        "member_count": len(rows),
        "member_rows": rows,
        "total_member_compressed_bytes": int(total_compressed),
        "total_member_uncompressed_bytes": int(total_uncompressed),
        "zip_container_overhead_bytes": int(len(archive_bytes) - total_compressed),
    }


def _zip_binding_blockers(zip_prof: dict[str, Any] | None, archive_sha256: str) -> list[str]:
    if zip_prof is None:
        return []
    rows = [row for row in zip_prof["member_rows"] if row["filename"] == "0.bin"]
    blockers: list[str] = []
    if len(rows) != 1:
        blockers.append(f"zip_expected_single_0_bin_member_got_{len(rows)}")
    elif rows[0].get("sha256") != archive_sha256:
        blockers.append(
            f"zip_0_bin_sha256_mismatch:{rows[0].get('sha256')}!={archive_sha256}"
        )
    return blockers


def _headroom_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "path": report.get("archive_path"),
        "archive_total_bytes": report.get("archive_total_bytes"),
        "wavelet_blob_bytes": report.get("wavelet_blob_bytes"),
        "pairs_measured": report.get("pairs_measured"),
        "total_pairs_in_archive": report.get("total_pairs_in_archive"),
        "headline_by_quant_step": report.get("headline_by_quant_step", []),
        "advisory_markers": {
            key: report.get(key)
            for key in (
                "axis_tag",
                "score_claim",
                "promotion_eligible",
                "ready_for_exact_eval_dispatch",
            )
        },
    }


def _local_replay_summary(replay: dict[str, Any] | None) -> dict[str, Any] | None:
    if replay is None:
        return None
    archive_rate_bytes = replay.get("archive_rate_bytes")
    rate = replay.get("rate")
    denom = None
    if isinstance(archive_rate_bytes, (int, float)) and isinstance(rate, (int, float)) and rate:
        denom = float(archive_rate_bytes) / float(rate)
    return {
        "schema": replay.get("schema"),
        "axis_tag": replay.get("axis_tag"),
        "score_claim": replay.get("score_claim"),
        "archive_rate_bytes": archive_rate_bytes,
        "rate": rate,
        "inferred_rate_denominator_bytes": denom,
        "rate_term_25x": (25.0 * float(rate) if isinstance(rate, (int, float)) else None),
        "d_seg": replay.get("d_seg"),
        "d_pose": replay.get("d_pose"),
        "pose_term_sqrt10": (
            math.sqrt(10.0 * float(replay["d_pose"]))
            if isinstance(replay.get("d_pose"), (int, float))
            else None
        ),
    }


def _opportunities(
    *,
    archive_bytes: int,
    archive_zip_bytes: int | None,
    frontier_archive_bytes: int | None,
    sections: list[dict[str, Any]],
    wavelet_profile: dict[str, Any],
    headroom: dict[str, Any] | None,
    replay: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wavelet_bytes = int(wavelet_profile["wavelet_blob_bytes"])
    top_ll = int(wavelet_profile["top_ll_raw_payload_bytes"])
    detail_payload = int(wavelet_profile["detail_payload_bytes"])
    if frontier_archive_bytes:
        target_ratio = (archive_zip_bytes or archive_bytes) / float(frontier_archive_bytes)
    else:
        target_ratio = None

    rows.append(
        {
            "name": "wavelet_blob_dominance",
            "entropy_position": "ARCHIVE/PAYLOAD_BEFORE_OUTER_ZIP",
            "current_bytes": wavelet_bytes,
            "archive_fraction": round(wavelet_bytes / archive_bytes, 6),
            "verdict": "binding_rate_axis",
            "next_action": "continue Z8 work only through wavelet/top-state byte reductions; other sections are not current rate binders",
        }
    )

    if headroom is not None:
        best = min(
            headroom.get("headline_by_quant_step", []),
            key=lambda row: float(row.get("v2_codec_detail_bytes", float("inf"))),
            default=None,
        )
        rows.append(
            {
                "name": "detail_coeff_quantize_entropy_code",
                "entropy_position": "AT_DETAIL_SYMBOL_CODER",
                "current_detail_payload_bytes": detail_payload,
                "best_measured_headroom_row": best,
                "verdict": "real_large_lever_but_distortion_operating_point_must_be_full_replay_gated",
                "next_action": "materialize RD schedules from full-video headroom rows, replay locally, and let exact archive projection accept or reject",
            }
        )
        q_counts = wavelet_profile.get("quantization_step_counts") or {}
        if len(q_counts) == 1:
            actual_q = float(next(iter(q_counts.keys())))
            matched = min(
                headroom.get("headline_by_quant_step", []),
                key=lambda row: abs(float(row.get("quant_step", float("inf"))) - actual_q),
                default=None,
            )
            if matched is not None and abs(float(matched.get("quant_step")) - actual_q) < 1e-9:
                predicted = round(float(matched.get("v2_codec_detail_bytes", 0.0)))
                actual = int(wavelet_profile.get("detail_payload_bytes", 0))
                rows.append(
                    {
                        "name": "materialized_vs_aggregate_rd_curve_gap",
                        "entropy_position": "AT_DETAIL_SYMBOL_CODER / PACKET_GRANULARITY",
                        "quantization_step": actual_q,
                        "aggregate_headroom_predicted_detail_bytes": predicted,
                        "materialized_inner_detail_payload_bytes": actual,
                        "materialized_minus_predicted_bytes": int(actual - predicted),
                        "verdict": (
                            "granularity_gap_large"
                            if predicted and actual > 1.25 * predicted
                            else "granularity_gap_small"
                        ),
                        "next_action": (
                            "make RD curves materialization-aware: measure per-pair table/context overhead or "
                            "move to section-level symbol streams before trusting aggregate subband floors"
                        ),
                    }
                )
    methods = wavelet_profile.get("detail_codec_method_counts") or {}
    if methods.get("qi16_static_range"):
        rows.append(
            {
                "name": "legacy_static_range_detail_codec",
                "entropy_position": "AT_DETAIL_SYMBOL_CODER / RUNTIME_DECODE",
                "method_counts": methods,
                "verdict": "byte_viable_but_runtime_suspect_until_benchmarked",
                "next_action": (
                    "benchmark full inflate decode and transcode to native constriction/RLE/byteplane "
                    "if static-range decode threatens the auth window"
                ),
            }
        )

    rows.append(
        {
            "name": "top_ll_float_payload",
            "entropy_position": "BEFORE_DETAIL_ENTROPY_CODER",
            "current_raw_top_ll_bytes_inside_pair_blobs": top_ll,
            "archive_fraction_if_uncompressed": round(top_ll / archive_bytes, 6),
            "verdict": "next_binding_after_detail_collapse",
            "next_action": "build top-LL RD curves: delta/DC quantization, predictive top-LL from frame0, Wyner-Ziv conditional residual, and entropy-code accepted residuals",
        }
    )

    solid_delta = wavelet_profile.get("solid_pair_raw_brotli_delta_vs_current_pair_blobs")
    if solid_delta is not None:
        quality = int(wavelet_profile.get("solid_pair_raw_brotli_quality", -1))
        rows.append(
            {
                "name": "solid_pair_blob_coding",
                "entropy_position": "PACKET_LAYOUT / OUTER_CONTEXT",
                "solid_pair_raw_brotli_delta_vs_current_pair_blobs": solid_delta,
                "verdict": (
                    "positive_candidate"
                    if int(solid_delta) < 0
                    else (
                        "q11_required_before_demote"
                        if quality != 11
                        else "not_a_generic_win_at_measured_quality"
                    )
                ),
                "next_action": (
                    "replace independent per-pair brotli members with global section coding plus indexed seek table "
                    "when q=11 is positive; otherwise rerun at q=11 before demotion"
                ),
            }
        )

    small_sections = [
        row for row in sections if row["name"] not in {"wavelet_blob", "z8hpc1_header"}
    ]
    rows.append(
        {
            "name": "non_wavelet_sections",
            "entropy_position": "ARCHIVE_CONTROL_AND_STACK_CUSTODY",
            "current_bytes": int(sum(int(row["bytes"]) for row in small_sections)),
            "largest_rows": sorted(small_sections, key=lambda row: int(row["bytes"]), reverse=True)[:5],
            "verdict": "secondary_until_wavelet_rate_axis_moves",
            "next_action": "receiver-proof elision/proceduralization only for sections not consumed by runtime; do not spend primary effort here before top-LL/detail collapse",
        }
    )

    if archive_zip_bytes is not None:
        rows.append(
            {
                "name": "outer_zip_and_repack",
                "entropy_position": "AFTER_PRIMARY_PAYLOAD_ENTROPY",
                "archive_zip_bytes": archive_zip_bytes,
                "zip_vs_0bin_delta_bytes": int(archive_zip_bytes - archive_bytes),
                "verdict": "minor_unless_runtime_members_or_headers_are_large",
                "next_action": "run deterministic min-zip/rebrotli only after payload grammar changes; after-entropy transforms cannot fix float payload entropy",
            }
        )

    if target_ratio is not None:
        rows.append(
            {
                "name": "contest_rate_distance",
                "entropy_position": "OBJECTIVE_RATE_TERM",
                "profiled_archive_bytes": int(archive_zip_bytes or archive_bytes),
                "frontier_archive_bytes": int(frontier_archive_bytes),
                "bytes_ratio_to_frontier": round(float(target_ratio), 3),
                "local_replay_summary": replay,
                "verdict": "must_get_to_same_order_of_magnitude_before_exact_auth",
                "next_action": "gate Z8 exact-auth only after byte-closed local archive is near frontier-byte scale and MLX/CPU distortion remains plausible",
            }
        )
    return rows


def build_profile(
    *,
    archive_bin: Path,
    archive_zip: Path | None,
    headroom_json: Path | None,
    replay_json: Path | None,
    frontier_archive_bytes: int | None,
    brotli_quality: int,
    measure_solid_pair_brotli: bool,
    solid_brotli_quality: int,
) -> dict[str, Any]:
    archive_data = archive_bin.read_bytes()
    archive_sha = _sha256_bytes(archive_data)
    parsed = parse_archive(archive_data)
    offsets = parse_z8hpc1_archive_bytes(archive_data)
    section_rows: list[dict[str, Any]] = []
    for name, (start, length) in offsets.items():
        payload = archive_data[start : start + length]
        row = {
            "name": name,
            "start": int(start),
            "bytes": int(length),
            "archive_fraction": round(float(length) / float(len(archive_data)), 8),
            "sha256": _sha256_bytes(payload),
            "codec_profile": _codec_profile(payload, brotli_quality=brotli_quality),
        }
        section_rows.append(row)

    wavelet_profile = _wavelet_blob_profile(
        parsed.wavelet_coeffs_blob,
        measure_solid_pair_brotli=measure_solid_pair_brotli,
        solid_brotli_quality=solid_brotli_quality,
    )
    zip_prof = _zip_profile(archive_zip)
    raw_headroom = _read_json(headroom_json)
    raw_replay = _read_json(replay_json)
    headroom = _headroom_summary(raw_headroom)
    replay = _local_replay_summary(raw_replay)
    auxiliary_evidence_blockers: list[str] = []
    if int(wavelet_profile["num_pairs"]) != int(parsed.num_pairs):
        auxiliary_evidence_blockers.append(
            f"wavelet_blob_num_pairs_mismatch_outer_archive:{wavelet_profile['num_pairs']}!={parsed.num_pairs}"
        )
    auxiliary_evidence_blockers.extend(_zip_binding_blockers(zip_prof, archive_sha))
    if raw_headroom is not None and raw_headroom.get("archive_sha256") != archive_sha:
        auxiliary_evidence_blockers.append(
            "headroom_archive_sha256_mismatch:"
            f"{raw_headroom.get('archive_sha256')}!={archive_sha}"
        )
    if raw_replay is not None:
        replay_archive_sha = raw_replay.get("archive_sha256") or raw_replay.get("candidate_archive_sha256")
        if replay_archive_sha is not None and replay_archive_sha != archive_sha:
            auxiliary_evidence_blockers.append(
                f"replay_archive_sha256_mismatch:{replay_archive_sha}!={archive_sha}"
            )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        **NON_PROMOTABLE_MARKERS,
        "archive": {
            "path": str(archive_bin),
            "bytes": len(archive_data),
            "sha256": archive_sha,
            "schema_version": int(parsed.schema_version),
            "num_pairs": int(parsed.num_pairs),
            "num_levels": int(parsed.num_levels),
            "num_groups_per_level": list(parsed.num_groups_per_level),
            "num_categories_per_level": list(parsed.num_categories_per_level),
            "decoder_latent_dim": int(parsed.decoder_latent_dim),
            "base_channels": int(parsed.base_channels),
            "wavelet_basis_id": int(parsed.wavelet_basis_id),
        },
        "zip_profile": zip_prof,
        "z8hpc1_sections": section_rows,
        "wavelet_blob_profile": wavelet_profile,
        "headroom_profile": headroom,
        "local_replay_profile": replay,
        "auxiliary_evidence_blockers": auxiliary_evidence_blockers,
        "auxiliary_evidence_bound_to_archive": not auxiliary_evidence_blockers,
    }
    report["opportunities"] = _opportunities(
        archive_bytes=len(archive_data),
        archive_zip_bytes=None if zip_prof is None else int(zip_prof["bytes"]),
        frontier_archive_bytes=frontier_archive_bytes,
        sections=section_rows,
        wavelet_profile=wavelet_profile,
        headroom=headroom,
        replay=replay,
    )
    return report


def render_markdown(report: dict[str, Any]) -> str:
    archive = report["archive"]
    zip_prof = report.get("zip_profile") or {}
    wavelet = report["wavelet_blob_profile"]
    lines = [
        "# Z8/HPC Archive Byte Profile",
        "",
        f"- archive: `{archive['path']}`",
        f"- 0.bin bytes: `{archive['bytes']:,}`",
        f"- archive sha256: `{archive['sha256']}`",
        f"- archive.zip bytes: `{zip_prof.get('bytes', 'not provided')}`",
        f"- pairs / levels: `{archive['num_pairs']}` / `{archive['num_levels']}`",
        f"- evidence: `{report['axis_tag']}`; score_claim=`{report['score_claim']}`",
        "",
        "## Sections",
        "",
        "| section | bytes | archive % | brotli probe | entropy floor |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(report["z8hpc1_sections"], key=lambda r: int(r["bytes"]), reverse=True):
        cp = row["codec_profile"]
        lines.append(
            f"| `{row['name']}` | {int(row['bytes']):,} | "
            f"{100.0 * float(row['archive_fraction']):.3f}% | "
            f"{int(cp['brotli_bytes']):,} | {float(cp['order0_entropy_floor_bytes']):,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Wavelet Blob",
            "",
            f"- wavelet blob bytes: `{wavelet['wavelet_blob_bytes']:,}`",
            f"- pair blob compressed bytes: `{wavelet['pair_blob_compressed_bytes']:,}`",
            f"- top-LL raw payload inside pair blobs: `{wavelet['top_ll_raw_payload_bytes']:,}`",
            f"- detail payload bytes inside pair blobs: `{wavelet['detail_payload_bytes']:,}`",
            f"- detail coefficients: `{wavelet['detail_coefficients']:,}`",
            f"- detail codec methods: `{wavelet['detail_codec_method_counts']}`",
        ]
    )
    if "solid_pair_raw_brotli_bytes" in wavelet:
        lines.append(
            "- solid raw-pair brotli probe: "
            f"`{wavelet['solid_pair_raw_brotli_bytes']:,}` bytes "
            f"(delta `{wavelet['solid_pair_raw_brotli_delta_vs_current_pair_blobs']:,}`)"
        )
    headroom = report.get("headroom_profile")
    if headroom:
        lines.extend(["", "## Detail Headroom", "", "| Δ | v2 bytes | floor bytes | headroom | MSE |", "|---:|---:|---:|---:|---:|"])
        for row in headroom.get("headline_by_quant_step", []):
            lines.append(
                f"| {row['quant_step']} | {int(row['v2_codec_detail_bytes']):,} | "
                f"{int(row['structured_shannon_floor_detail_bytes']):,} | "
                f"{100.0 * float(row['headroom_fraction']):.1f}% | {float(row['mean_distortion_mse']):.3e} |"
            )
    lines.extend(["", "## Ranked Opportunities", ""])
    for idx, row in enumerate(report["opportunities"], start=1):
        lines.append(f"{idx}. `{row['name']}` — `{row['verdict']}`")
        lines.append(f"   - position: `{row['entropy_position']}`")
        lines.append(f"   - next: {row['next_action']}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-bin", required=True, type=Path)
    ap.add_argument("--archive-zip", type=Path, default=None)
    ap.add_argument("--headroom-json", type=Path, default=None)
    ap.add_argument("--replay-json", type=Path, default=None)
    ap.add_argument("--frontier-archive-bytes", type=int, default=None)
    ap.add_argument("--brotli-quality", type=int, default=11)
    ap.add_argument("--measure-solid-pair-brotli", action="store_true")
    ap.add_argument("--solid-brotli-quality", type=int, default=5)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args(argv)

    report = build_profile(
        archive_bin=args.archive_bin.resolve(),
        archive_zip=args.archive_zip.resolve() if args.archive_zip else None,
        headroom_json=args.headroom_json.resolve() if args.headroom_json else None,
        replay_json=args.replay_json.resolve() if args.replay_json else None,
        frontier_archive_bytes=args.frontier_archive_bytes,
        brotli_quality=int(args.brotli_quality),
        measure_solid_pair_brotli=bool(args.measure_solid_pair_brotli),
        solid_brotli_quality=int(args.solid_brotli_quality),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "archive_bytes": report["archive"]["bytes"],
                "zip_bytes": None
                if report["zip_profile"] is None
                else report["zip_profile"]["bytes"],
                "wavelet_blob_bytes": report["wavelet_blob_profile"]["wavelet_blob_bytes"],
                "top_ll_raw_payload_bytes": report["wavelet_blob_profile"]["top_ll_raw_payload_bytes"],
                "detail_payload_bytes": report["wavelet_blob_profile"]["detail_payload_bytes"],
                "opportunity_count": len(report["opportunities"]),
                "out_json": str(args.out_json),
                "out_md": None if args.out_md is None else str(args.out_md),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
