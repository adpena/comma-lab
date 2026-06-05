# SPDX-License-Identifier: MIT
"""Binary attribution profile for SNeRV receiver packets and contest packages.

This is rate/grammar evidence only. It deliberately imports no scorer code and
never grants score, rank, kill, promotion, CPU-auth, or CUDA-auth authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tac.analysis.nerv_modelsize_budget import (
    CONTEST_RATE_DENOM_BYTES,
    RATE_SCORE_PER_BYTE,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    SECTION_ORDER,
    SNERV_ARCHIVE_MAGIC,
    SNERV_ARCHIVE_MAGIC_V2,
    SNERV_ARCHIVE_SCHEMA,
    SNERV_ARCHIVE_SCHEMA_V2,
    SnervArchiveError,
    decode_lf_metadata_payload,
    inspect_decoder_payload_header,
    inspect_lf_quant_payload_header,
    unpack_snerv_archive,
)

SCHEMA = "snerv_binary_profile.v1"
AXIS_TAG = "[local-rate-profile false-authority]"
DEFAULT_FRONTIER_BYTES = 178_493
FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


class SnervBinaryProfileError(ValueError):
    """Raised when a SNeRV binary/profile input is malformed."""


def build_snerv_binary_profile(
    *,
    input_path: str | Path,
    frontier_bytes: int = DEFAULT_FRONTIER_BYTES,
) -> dict[str, Any]:
    """Build a machine-readable SNeRV binary attribution profile.

    ``input_path`` may be a contest-shaped ``archive.zip`` containing ``0.bin``
    or a raw SNAR1/SNAR2 packet. The output is intentionally false-authority.
    """

    path = Path(input_path).expanduser().resolve(strict=False)
    packet, package = _load_snerv_packet(path)
    decoded = unpack_snerv_archive(packet)
    wire_format = _wire_format_for_schema(decoded.schema)
    sections = decoded.sections
    section_rows = _section_rows(sections, total_bytes=len(packet))
    packet_sha256 = _sha256_bytes(packet)
    packet_header_bytes = len(packet) - sum(len(sections[name]) for name in SECTION_ORDER)
    lf_payload_header = inspect_lf_quant_payload_header(sections["lf_payload"])
    lf_stats, lf_blockers = _build_lf_quant_profile(
        decoded=decoded,
        lf_payload_header=lf_payload_header,
        section_bytes=len(sections["lf_payload"]),
        packet_bytes=len(packet),
    )
    zero_points = decode_lf_metadata_payload(
        sections["metadata_payload"],
        expected_count=int(decoded.metadata.get("lf_plane_count"))
        if decoded.metadata.get("lf_plane_count") is not None
        else None,
    )
    step_maps = decoded.decode_step_maps()
    decoder_header = inspect_decoder_payload_header(sections["decoder_payload"])
    charged_bytes = (
        int(package["archive_bytes"]) if package.get("archive_bytes") else len(packet)
    )
    blockers = _profile_blockers(
        charged_bytes=charged_bytes,
        frontier_bytes=int(frontier_bytes),
        lf_payload_fraction=float(lf_stats["section_fraction_of_packet"]),
        package=package,
        lf_blockers=lf_blockers,
    )
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "axis_tag": AXIS_TAG,
        "input_path": path.as_posix(),
        "input_kind": package["input_kind"],
        "input_sha256": _sha256_bytes(path.read_bytes()),
        "frontier_bytes": int(frontier_bytes),
        "contest_rate_denominator_bytes": CONTEST_RATE_DENOM_BYTES,
        "rate_score_per_byte": RATE_SCORE_PER_BYTE,
        "charged_archive_bytes": charged_bytes,
        "charged_rate_score": float(charged_bytes * RATE_SCORE_PER_BYTE),
        "bytes_above_frontier": int(charged_bytes) - int(frontier_bytes),
        "packet_bytes": len(packet),
        "packet_sha256": packet_sha256,
        "packet_wire_format": wire_format,
        "packet_schema": decoded.schema,
        "packet_header_bytes": packet_header_bytes,
        "receiver_packet_bytes": len(packet),
        "receiver_packet_sha256": packet_sha256,
        "receiver_packet_wire_format": wire_format,
        "receiver_packet_schema": decoded.schema,
        "receiver_packet_header_bytes": packet_header_bytes,
        "snar1_packet_bytes": len(packet) if wire_format == "snar1" else None,
        "snar1_packet_sha256": packet_sha256 if wire_format == "snar1" else None,
        "snar1_header_bytes": packet_header_bytes if wire_format == "snar1" else None,
        "snar2_packet_bytes": len(packet) if wire_format == "snar2" else None,
        "snar2_packet_sha256": packet_sha256 if wire_format == "snar2" else None,
        "snar2_header_bytes": packet_header_bytes if wire_format == "snar2" else None,
        "package_profile": package,
        "receiver_packet_metadata": _metadata_summary(decoded.metadata),
        "snar1_metadata": _metadata_summary(decoded.metadata)
        if wire_format == "snar1"
        else None,
        "snar2_metadata": _metadata_summary(decoded.metadata)
        if wire_format == "snar2"
        else None,
        "section_rows": section_rows,
        "section_summary": _section_summary(section_rows),
        "lf_payload_header": lf_payload_header,
        "lf_quant_profile": lf_stats,
        "lf_zero_point_profile": {
            "count": int(zero_points.size),
            "dtype": str(zero_points.dtype),
            "min": _finite_float(np.min(zero_points)) if zero_points.size else None,
            "max": _finite_float(np.max(zero_points)) if zero_points.size else None,
            "mean": _finite_float(np.mean(zero_points)) if zero_points.size else None,
        },
        "step_map_profile": _step_map_profile(
            step_maps,
            section_bytes=len(sections["step_map_packet"]),
            packet_bytes=len(packet),
        ),
        "decoder_payload_header": decoder_header,
        "verdict": _verdict(
            charged_bytes=charged_bytes,
            frontier_bytes=int(frontier_bytes),
            lf_payload_fraction=float(lf_stats["section_fraction_of_packet"]),
        ),
        "next_actions": [
            "native_mlx_snerv_train_export_archive_adapter",
            "full_video_mlx_prefilter_then_local_cpu_replay",
            "learned_or_symbolic_lf_hf_generator_to_reduce_explicit_lf_payload",
            "sr_low_resolution_carrier_with_posenet_protected_pathway",
            "pose_hard_score_primary_decoder_fit_with_explicit_segnet_slack",
            "wavelet_group_p18_p19_saliency_binding",
        ],
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def write_snerv_binary_profile(
    *,
    input_path: str | Path,
    output_path: str | Path,
    frontier_bytes: int = DEFAULT_FRONTIER_BYTES,
) -> dict[str, Any]:
    """Write a SNeRV binary profile and return the payload."""

    profile = build_snerv_binary_profile(
        input_path=input_path,
        frontier_bytes=frontier_bytes,
    )
    out = Path(output_path).expanduser().resolve(strict=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return profile


def _load_snerv_packet(path: Path) -> tuple[bytes, dict[str, Any]]:
    if not path.is_file():
        raise SnervBinaryProfileError(f"SNeRV input does not exist: {path}")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            selected_info, packet, input_kind, wire_format = _select_zip_packet_member(
                zf,
                infos,
            )
            members = [
                {
                    "name": info.filename,
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "compress_type": int(info.compress_type),
                    "crc": f"{info.CRC:08x}",
                }
                for info in infos
            ]
        archive_bytes = path.stat().st_size
        selected_compress_size = int(selected_info.compress_size)
        selected_file_size = int(selected_info.file_size)
        selected_name = selected_info.filename
        is_root_0bin = selected_name == "0.bin"
        return packet, {
            "input_kind": input_kind,
            "packet_wire_format": wire_format,
            "archive_bytes": int(archive_bytes),
            "zip_member_count": len(members),
            "zip_members": members,
            "zip_packet_member_name": selected_name,
            "zip_packet_file_size": selected_file_size,
            "zip_packet_compress_size": selected_compress_size,
            "zip_0bin_file_size": len(packet) if is_root_0bin else None,
            "zip_0bin_compress_size": selected_compress_size if is_root_0bin else None,
            "zip_overhead_bytes": int(archive_bytes)
            - int(sum(row["compress_size"] for row in members)),
        }
    packet = path.read_bytes()
    wire_format = _detect_packet_wire_format(packet)
    return packet, {
        "input_kind": f"raw_{wire_format}_packet",
        "packet_wire_format": wire_format,
        "archive_bytes": None,
        "zip_member_count": None,
        "zip_members": [],
        "zip_packet_member_name": None,
        "zip_packet_file_size": None,
        "zip_packet_compress_size": None,
        "zip_0bin_file_size": None,
        "zip_0bin_compress_size": None,
        "zip_overhead_bytes": None,
    }


def _select_zip_packet_member(
    zf: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
) -> tuple[zipfile.ZipInfo, bytes, str, str]:
    by_name = {info.filename: info for info in infos}
    if "0.bin" in by_name:
        info = by_name["0.bin"]
        packet = zf.read(info.filename)
        return info, packet, "contest_archive_zip", _detect_packet_wire_format(packet)
    if len(infos) == 1:
        info = infos[0]
        packet = zf.read(info.filename)
        wire_format = _detect_packet_wire_format(packet)
        return info, packet, "single_member_snar_archive_zip", wire_format
    raise SnervBinaryProfileError(
        "SNeRV archive.zip is missing 0.bin and does not contain a single SNAR1/SNAR2 member"
    )


def _detect_packet_wire_format(packet: bytes) -> str:
    if packet.startswith(SNERV_ARCHIVE_MAGIC):
        return "snar1"
    if packet.startswith(SNERV_ARCHIVE_MAGIC_V2):
        return "snar2"
    raise SnervBinaryProfileError("SNeRV packet is not SNAR1 or SNAR2")


def _wire_format_for_schema(schema: str) -> str:
    if schema == SNERV_ARCHIVE_SCHEMA:
        return "snar1"
    if schema == SNERV_ARCHIVE_SCHEMA_V2:
        return "snar2"
    raise SnervBinaryProfileError(f"unsupported SNeRV packet schema: {schema!r}")


def _section_rows(sections: Mapping[str, bytes], *, total_bytes: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in SECTION_ORDER:
        blob = sections[name]
        rows.append(
            {
                "section": name,
                "bytes": len(blob),
                "sha256": _sha256_bytes(blob),
                "fraction_of_packet": (
                    float(len(blob) / total_bytes) if total_bytes else 0.0
                ),
            }
        )
    return rows


def _section_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(row["bytes"]) for row in rows)
    largest = max(rows, key=lambda row: int(row["bytes"]))
    return {
        "section_payload_bytes": int(total),
        "largest_section": largest["section"],
        "largest_section_bytes": int(largest["bytes"]),
        "largest_section_fraction_of_payload": (
            float(int(largest["bytes"]) / total) if total else 0.0
        ),
    }


def _lf_quant_stats(
    lf_planes: list[np.ndarray],
    *,
    section_bytes: int,
    packet_bytes: int,
) -> dict[str, Any]:
    if not lf_planes:
        raise SnervBinaryProfileError("SNeRV LF payload decoded to zero planes")
    flat = np.concatenate([np.asarray(a, dtype=np.int64).reshape(-1) for a in lf_planes])
    values, counts = np.unique(flat, return_counts=True)
    entropy_bits = _entropy_bits(counts, total=int(flat.size))
    nonzero = flat[flat != 0]
    nonzero_entropy_bits = (
        _entropy_bits(np.unique(nonzero, return_counts=True)[1], total=int(nonzero.size))
        if nonzero.size
        else 0.0
    )
    p_nonzero = float(nonzero.size / flat.size)
    binary_entropy_bits = _binary_entropy_bits(p_nonzero)
    structured_bits_per_coeff = binary_entropy_bits + p_nonzero * nonzero_entropy_bits
    raw_bytes = int(flat.nbytes)
    entropy_floor_bytes = math.ceil(float(flat.size) * entropy_bits / 8.0)
    structured_floor_bytes = math.ceil(
        float(flat.size) * structured_bits_per_coeff / 8.0
    )
    return {
        "plane_count": len(lf_planes),
        "plane_shapes": [list(map(int, a.shape)) for a in lf_planes[:8]],
        "plane_shapes_truncated": len(lf_planes) > 8,
        "coeff_count": int(flat.size),
        "raw_int64_bytes": raw_bytes,
        "section_bytes": int(section_bytes),
        "section_fraction_of_packet": (
            float(section_bytes / packet_bytes) if packet_bytes else 0.0
        ),
        "bytes_per_coeff": float(section_bytes / flat.size),
        "min": int(np.min(flat)),
        "max": int(np.max(flat)),
        "unique_value_count": len(counts),
        "zero_fraction": float(np.mean(flat == 0)),
        "nonzero_fraction": p_nonzero,
        "order0_entropy_bits_per_coeff": entropy_bits,
        "order0_entropy_floor_bytes": entropy_floor_bytes,
        "structured_zero_nonzero_bits_per_coeff": structured_bits_per_coeff,
        "structured_zero_nonzero_floor_bytes": structured_floor_bytes,
        "section_bytes_over_order0_floor": _safe_ratio(section_bytes, entropy_floor_bytes),
        "section_bytes_over_structured_floor": _safe_ratio(
            section_bytes,
            structured_floor_bytes,
        ),
        "top_values": [
            {"value": int(values[idx]), "count": int(counts[idx])}
            for idx in np.argsort(counts)[::-1][:12]
        ],
    }


def _build_lf_quant_profile(
    *,
    decoded: Any,
    lf_payload_header: Mapping[str, Any],
    section_bytes: int,
    packet_bytes: int,
) -> tuple[dict[str, Any], list[str]]:
    try:
        lf_planes = decoded.decode_lf_quant_planes()
    except SnervArchiveError as exc:
        return (
            _lf_quant_header_only_stats(
                lf_payload_header,
                section_bytes=section_bytes,
                packet_bytes=packet_bytes,
                decode_error=str(exc),
            ),
            ["snerv_lf_payload_decode_failed_for_profile"],
        )
    stats = _lf_quant_stats(
        lf_planes,
        section_bytes=section_bytes,
        packet_bytes=packet_bytes,
    )
    stats["decode_status"] = "decoded"
    return stats, []


def _lf_quant_header_only_stats(
    header: Mapping[str, Any],
    *,
    section_bytes: int,
    packet_bytes: int,
    decode_error: str,
) -> dict[str, Any]:
    plane_count = header.get("shape_count", header.get("plane_count"))
    shared_shape = header.get("shared_shape")
    coeff_count: int | None = None
    if plane_count is not None and isinstance(shared_shape, list):
        coeff_count = int(plane_count) * int(np.prod([int(v) for v in shared_shape]))
    elif header.get("canonical_int64_raw_bytes") is not None:
        coeff_count = int(header["canonical_int64_raw_bytes"]) // np.dtype("<i8").itemsize
    return {
        "decode_status": "header_only_decode_failed",
        "decode_error": decode_error,
        "schema": header.get("schema"),
        "codec": header.get("codec"),
        "plane_count": int(plane_count) if plane_count is not None else None,
        "shared_shape": shared_shape,
        "coeff_count": coeff_count,
        "raw_int64_bytes": header.get("canonical_int64_raw_bytes", header.get("raw_bytes")),
        "section_bytes": int(section_bytes),
        "section_fraction_of_packet": (
            float(section_bytes / packet_bytes) if packet_bytes else 0.0
        ),
        "bytes_per_coeff": (
            float(section_bytes / coeff_count) if coeff_count else None
        ),
        "order0_entropy_bits_per_coeff": None,
        "order0_entropy_floor_bytes": None,
        "structured_zero_nonzero_bits_per_coeff": None,
        "structured_zero_nonzero_floor_bytes": None,
        "section_bytes_over_order0_floor": None,
        "section_bytes_over_structured_floor": None,
        "top_values": [],
    }


def _step_map_profile(
    step_maps: list[np.ndarray],
    *,
    section_bytes: int,
    packet_bytes: int,
) -> dict[str, Any]:
    if not step_maps:
        return {
            "map_count": 0,
            "section_bytes": int(section_bytes),
            "section_fraction_of_packet": (
                float(section_bytes / packet_bytes) if packet_bytes else 0.0
            ),
        }
    sizes = [int(np.asarray(a).size) for a in step_maps]
    return {
        "map_count": len(step_maps),
        "map_shapes": [list(map(int, a.shape)) for a in step_maps[:8]],
        "map_shapes_truncated": len(step_maps) > 8,
        "value_count": int(sum(sizes)),
        "section_bytes": int(section_bytes),
        "section_fraction_of_packet": (
            float(section_bytes / packet_bytes) if packet_bytes else 0.0
        ),
        "bytes_per_value": float(section_bytes / sum(sizes)) if sum(sizes) else None,
        "min": _finite_float(min(float(np.min(a)) for a in step_maps)),
        "max": _finite_float(max(float(np.max(a)) for a in step_maps)),
    }


def _metadata_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "n_pairs",
        "frames_per_pair",
        "channels",
        "height",
        "width",
        "lf_plane_count",
        "levels",
        "wavelet",
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
    }
    return {key: metadata.get(key) for key in sorted(keys) if key in metadata}


def _profile_blockers(
    *,
    charged_bytes: int,
    frontier_bytes: int,
    lf_payload_fraction: float,
    package: Mapping[str, Any],
    lf_blockers: list[str],
) -> list[str]:
    blockers = [
        "snerv_binary_profile_is_rate_only_not_score_authority",
        "full_video_scorer_component_profile_missing",
        "paired_contest_cpu_cuda_auth_eval_missing",
    ]
    blockers.extend(lf_blockers)
    if package.get("input_kind") != "contest_archive_zip":
        blockers.append("not_packaged_as_contest_archive_zip")
    if charged_bytes > frontier_bytes:
        blockers.append("snerv_archive_rate_exceeds_frontier")
    if lf_payload_fraction > 0.5:
        blockers.append("snerv_lf_payload_dominates_packet")
    return blockers


def _verdict(
    *,
    charged_bytes: int,
    frontier_bytes: int,
    lf_payload_fraction: float,
) -> str:
    if charged_bytes > frontier_bytes and lf_payload_fraction > 0.5:
        return "current_snerv_artifact_rate_blocked_by_explicit_lf_payload"
    if charged_bytes > frontier_bytes:
        return "current_snerv_artifact_rate_blocked"
    if lf_payload_fraction > 0.5:
        return "snerv_payload_lf_dominant_but_archive_under_frontier"
    return "snerv_binary_rate_profile_not_lf_dominant"


def _entropy_bits(counts: Any, *, total: int) -> float:
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counts:
        p = float(count) / float(total)
        if p > 0.0:
            out -= p * math.log2(p)
    return out


def _binary_entropy_bits(p: float) -> float:
    p = min(max(float(p), 0.0), 1.0)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _safe_ratio(num: int | float, den: int | float | None) -> float | None:
    if den is None or float(den) == 0.0:
        return None
    return float(num) / float(den)


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _finite_float(value: Any) -> float | None:
    out = float(value)
    return out if math.isfinite(out) else None


__all__ = [
    "AXIS_TAG",
    "DEFAULT_FRONTIER_BYTES",
    "SCHEMA",
    "SnervBinaryProfileError",
    "build_snerv_binary_profile",
    "write_snerv_binary_profile",
]
