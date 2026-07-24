# SPDX-License-Identifier: MIT
"""Standalone deterministic receiver emitted by the DDM E1/E2 exporter.

This file is copied byte-for-byte to ``inflate.py``.  It intentionally imports
only the Python standard library plus Torch and Brotli.  All instance-derived
bytes live in the extracted archive directory and are covered by the manifest.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import os
import shutil
import struct
import sys
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import torch.nn.functional as F

SCHEMA = "ddm_e1_runtime_archive.v1"
E2_SCHEMA = "ddm_e2_runtime_archive.v1"
E3_SCHEMA = "ddm_e3_runtime_archive.v1"
RATE_DOCTRINE_SCHEMA = "ddm_four_clause_rate_doctrine.v1"
BLOB_MAGIC = b"DDE1B"
BLOB_HEADER = struct.Struct(">5sBBBBQQ32s")
LZMA_FILTERS = [
    {
        "id": lzma.FILTER_LZMA1,
        "dict_size": 1 << 20,
        "lc": 3,
        "lp": 0,
        "pb": 2,
    }
]
EXPECTED_MEMBERS = ("manifest.json", "base/chart.ddb", "semantic/composed.dds")
EXPECTED_EXTENSION_SLOTS = [
    {
        "active_member": None,
        "block": "D1",
        "frame": "DDE1B",
        "member_prefix": "amplitude/",
        "rate_custody": "independent_member_bytes_sha256",
        "schema": "ddm_amplitude_field_section.v1",
    },
    {
        "active_member": None,
        "block": "D2",
        "frame": "DDE1B",
        "member_prefix": "tolerance/",
        "rate_custody": "independent_member_bytes_sha256",
        "schema": "ddm_per_element_tolerance_dual_section.v1",
    },
    {
        "active_member": None,
        "block": "D5",
        "frame": "DDE1B",
        "member_prefix": "texture_quotient/",
        "rate_custody": "independent_member_bytes_sha256",
        "schema": "ddm_texture_quotient_residual_stats_section.v1",
    },
    {
        "active_member": None,
        "block": "D4",
        "frame": "DDE1B",
        "member_prefix": "coder/",
        "rate_custody": "independent_member_bytes_sha256",
        "schema": "ddm_coder_probability_parameters_section.v1",
    },
    {
        "active_member": None,
        "block": "D6",
        "frame": "DDE1B",
        "member_prefix": "realization/",
        "rate_custody": "independent_member_bytes_sha256",
        "schema": "ddm_realization_map_metadata_section.v1",
    },
]
EXPECTED_REFINEMENT = {
    "apparatus_validity_stamps_required": True,
    "best_so_far_checkpoint": True,
    "block_order": ["L", "D2", "D1", "D4", "D6", "D5"],
    "block_order_policy": ("dependency_topological_then_coarse_to_fine_then_largest_marginal_first"),
    "curve_authority": "verification_receipt",
    "cycle_index": 0,
    "fixed_budget": True,
    "global_reinvestment": True,
    "schema": "ddm_joint_fixed_budget_refinement.v1",
    "status": "OPEN_SUCCESSOR_CYCLES_OWED",
    "stop_law": "full_joint_cycle_no_net_gain_at_constant_bytes",
    "train_least_order": ["derive", "solve", "fit", "train_last"],
}
LANGUAGE_VERSION = "ddm_composed_language.v1"
PAIR_H = 384
PAIR_W = 512
CAMERA_H = 874
CAMERA_W = 1164
CHANNELS = 3
FRAMES_PER_PAIR = 2
SAFETY_BYTES = 256 * 1024 * 1024
PA1_TRANSFORM_ID = "ddm_pa1_scorer_only_bn_inverse_frame0_receiver_v1"
# Generic scorer constants. These values are derived only from the frozen
# PoseNet first-stem convolution weights and BN running statistics. They are
# video-independent rule-118 code, not learned or instance-derived payload.
PA1_TARGET_RAW_MEAN = (
    80.95886115606237,
    81.32804175880518,
    81.42772803659768,
    80.91749681114962,
    131.59657382020762,
    124.00907917446821,
    81.15453720095364,
    81.20753723118113,
    81.50170914664491,
    80.9142866240545,
    131.39192023195903,
    124.1646389598534,
)
PA1_TARGET_RAW_VARIANCE = (
    0.4064062500000001,
    6777.201663733891,
    2017.1683030183508,
    4499.897476726892,
    178.73376014986843,
    2.3033717730748027,
    0.4064062500000001,
    0.4064062500000001,
    0.4064062500000001,
    0.4064062500000001,
    130.6544297496608,
    0.40640625000014863,
)


class ReceiverError(ValueError):
    """The counted state or runtime lifecycle failed closed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_lower_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _duplicate_refusing_json(payload: bytes, *, label: str) -> Any:
    def object_pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ReceiverError(f"{label} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", "strict"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ReceiverError(f"{label} has non-finite JSON number {token!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiverError(f"{label} is malformed JSON") from exc
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if canonical != payload:
        raise ReceiverError(f"{label} is not canonical JSON")
    return value


def _safe_relative_name(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ReceiverError("video name must be a nonempty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ReceiverError("video name escaped the output root")
    return path


def _read_exact_members(archive_dir: Path) -> dict[str, bytes]:
    observed = tuple(
        sorted(path.relative_to(archive_dir).as_posix() for path in archive_dir.rglob("*") if path.is_file())
    )
    if observed != tuple(sorted(EXPECTED_MEMBERS)):
        raise ReceiverError(f"extracted archive members differ: {observed!r} != {EXPECTED_MEMBERS!r}")
    result: dict[str, bytes] = {}
    for name in EXPECTED_MEMBERS:
        path = archive_dir / name
        if path.is_symlink() or not path.is_file():
            raise ReceiverError(f"archive member is not one regular file: {name}")
        result[name] = path.read_bytes()
    return result


def _parse_blob(payload: bytes, *, expected_kind: int, label: str) -> tuple[bytes, tuple[int, ...]]:
    if len(payload) < BLOB_HEADER.size:
        raise ReceiverError(f"{label} frame is truncated")
    magic, version, codec, kind, rank, raw_bytes, coded_bytes, raw_digest = BLOB_HEADER.unpack_from(payload)
    if magic != BLOB_MAGIC or version != 1 or codec != 2 or kind != expected_kind or rank > 8:
        raise ReceiverError(f"{label} frame identity mismatch")
    dimension_bytes = rank * 4
    coded_offset = BLOB_HEADER.size + dimension_bytes
    if coded_offset > len(payload) or len(payload) != coded_offset + coded_bytes:
        raise ReceiverError(f"{label} frame length is noncanonical")
    dimensions = struct.unpack_from(f">{rank}I", payload, BLOB_HEADER.size) if rank else ()
    product = 1
    for dimension in dimensions:
        if dimension <= 0:
            raise ReceiverError(f"{label} has a zero dimension")
        product *= dimension
    if kind == 1 and product != raw_bytes:
        raise ReceiverError(f"{label} uint8 shape disagrees with raw length")
    try:
        raw = lzma.decompress(
            payload[coded_offset:],
            format=lzma.FORMAT_RAW,
            filters=LZMA_FILTERS,
        )
    except lzma.LZMAError as exc:
        raise ReceiverError(f"{label} raw LZMA1 decode failed") from exc
    if len(raw) != raw_bytes or hashlib.sha256(raw).digest() != raw_digest:
        raise ReceiverError(f"{label} decoded custody mismatch")
    return raw, tuple(int(value) for value in dimensions)


def _validate_block_versions(
    value: Any,
    members: dict[str, bytes],
    *,
    amplitude_enabled: bool,
) -> None:
    order = ["L", "D2", "D1", "D4", "D6", "D5"]
    versions = {
        "L": "ddm_L_composed_semantic.v1",
        "D2": "ddm_D2_inactive.v1",
        "D1": ("ddm_D1_pa1_scorer_stat_affine_free.v1" if amplitude_enabled else "ddm_D1_inactive.v1"),
        "D4": "ddm_D4_lzma1_raw_d1m_measure.v1",
        "D6": "ddm_D6_camera_realization.v1",
        "D5": "ddm_D5_inactive.v1",
    }
    active = {"L", "D4", "D6"} | ({"D1"} if amplitude_enabled else set())
    if (
        not isinstance(value, list)
        or any(not isinstance(row, dict) for row in value)
        or [row.get("block") for row in value] != order
    ):
        raise ReceiverError("block-version order changed")
    for row in value:
        block = row["block"]
        expected_inputs = (
            [
                {
                    "member": name,
                    "sha256": _sha256(members[name]),
                }
                for name in EXPECTED_MEMBERS[1:]
            ]
            if block in active
            else []
        )
        expected = {
            "block": block,
            "consumption_time_policy": (
                "verify_input_member_hashes_or_refuse" if block in active else "refuse_unstamped_activation"
            ),
            "input_members": expected_inputs,
            "status": "active" if block in active else "inactive",
            "validity_horizon": (
                {
                    "kind": "exact_input_member_hash_equality",
                    "reason": "byte_identity_requires_zero_untracked_input_drift",
                }
                if block in active
                else {
                    "kind": "inactive_no_cached_quantity",
                    "reason": "no_section_bytes_or_cached_state_consumed",
                }
            ),
            "version": versions[block],
        }
        if row != expected:
            raise ReceiverError(f"stale or malformed block-version stamp: {block}")


def _validate_manifest(manifest: Any, members: dict[str, bytes]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ReceiverError("manifest must be an object")
    schema = manifest.get("schema")
    common_keys = {
        "archive",
        "block_versions",
        "chart",
        "dependencies",
        "extension_slots",
        "false_authority",
        "geometry",
        "language_version",
        "output",
        "refinement",
        "schema",
        "sections",
        "state",
    }
    e2_keys = common_keys | {
        "pose_contract",
        "rate_doctrine",
        "semantic_frame_policy",
    }
    expected_keys = common_keys if schema == SCHEMA else e2_keys
    if schema not in {SCHEMA, E2_SCHEMA, E3_SCHEMA} or set(manifest) != expected_keys:
        raise ReceiverError("manifest keys differ from the sealed schema")
    archive = manifest["archive"]
    if (
        not isinstance(archive, dict)
        or set(archive)
        != {
            "source_bytes",
            "source_sha256",
            "state_bytes",
            "state_sha256",
        }
        or archive["source_bytes"] != 133_941
        or archive["state_bytes"] != 134_211
        or not _is_lower_sha256(archive["source_sha256"])
        or not _is_lower_sha256(archive["state_sha256"])
    ):
        raise ReceiverError("manifest source/state provenance is malformed")
    if manifest["state"] != {
        "batch_pairs": 16,
        "name": "v15_j2_lane_seed_theta0",
        "receiver_effective_dofs": {
            "island_translation_dofs": 326,
            "lane_program_dofs": 24,
            "shared_template_dofs": 18,
            "total": 368,
        },
    }:
        raise ReceiverError("manifest receiver-effective state changed")
    if manifest["dependencies"] != ["torch"]:
        raise ReceiverError("runtime dependency contract changed")
    if manifest["language_version"] != LANGUAGE_VERSION:
        raise ReceiverError("language-version stamp changed")
    _validate_block_versions(
        manifest["block_versions"],
        members,
        amplitude_enabled=schema == E3_SCHEMA,
    )
    if manifest["extension_slots"] != EXPECTED_EXTENSION_SLOTS:
        raise ReceiverError("typed extension-slot contract changed")
    if any(row["active_member"] is not None for row in EXPECTED_EXTENSION_SLOTS):
        raise ReceiverError("this receiver has no active typed extension section")
    if manifest["refinement"] != EXPECTED_REFINEMENT:
        raise ReceiverError("joint-refinement contract changed")
    geometry = manifest["geometry"]
    if geometry != {
        "camera_hw": [CAMERA_H, CAMERA_W],
        "chart_grid_hw": [12, 16],
        "chart_hw": [32, 32],
        "channels": CHANNELS,
        "frames_per_pair": FRAMES_PER_PAIR,
        "pair_count": 600,
        "scorer_hw": [PAIR_H, PAIR_W],
    }:
        raise ReceiverError("manifest geometry mismatch")
    if manifest["false_authority"] != {
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "research_only": True,
        "score_claim": False,
    }:
        raise ReceiverError("manifest false-authority contract changed")
    if schema in {E2_SCHEMA, E3_SCHEMA}:
        if manifest["semantic_frame_policy"] != "frame1_only_seg_free_frame0":
            raise ReceiverError("E2 semantic frame ownership changed")
        _validate_rate_doctrine(manifest["rate_doctrine"], members)
        _validate_pose_contract(manifest["pose_contract"])
    sections = manifest["sections"]
    if (
        not isinstance(sections, list)
        or any(not isinstance(row, dict) for row in sections)
        or [row.get("member") for row in sections] != list(EXPECTED_MEMBERS[1:])
    ):
        raise ReceiverError("manifest section order mismatch")
    for row in sections:
        if set(row) != {"bytes", "member", "sha256"}:
            raise ReceiverError("manifest section keys differ")
        payload = members[row["member"]]
        if row["bytes"] != len(payload) or row["sha256"] != _sha256(payload):
            raise ReceiverError(f"manifest section custody mismatch: {row['member']}")
    return manifest


def _validate_rate_doctrine(value: Any, members: dict[str, bytes]) -> None:
    """Require one complete audit row per counted description stream."""

    stream_names = list(EXPECTED_MEMBERS[1:])
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "candidate_admissible",
            "correction_policy",
            "ordered_redundancy_matrix",
            "schema",
            "single_owner_facts",
            "streams",
            "verdict_scope",
        }
        or value["schema"] != RATE_DOCTRINE_SCHEMA
        or type(value["candidate_admissible"]) is not bool
        or not isinstance(value["verdict_scope"], str)
        or not value["verdict_scope"]
    ):
        raise ReceiverError("E2 four-clause doctrine header is malformed")
    streams = value["streams"]
    if (
        not isinstance(streams, list)
        or any(not isinstance(row, dict) for row in streams)
        or [row.get("member") for row in streams] != stream_names
    ):
        raise ReceiverError("E2 stream audit rows are missing or reordered")
    audit_keys = {
        "audit_triple",
        "candidate_admissible",
        "first_rung",
        "member",
        "non_redundancy",
        "verdict_scope",
    }
    triple_keys = {
        "scorer_visibility",
        "sensitivity_priced_tolerance",
        "three_layer_decomposition",
    }
    for row in streams:
        if (
            set(row) != audit_keys
            or type(row["candidate_admissible"]) is not bool
            or row["first_rung"] is not True
            or set(row["audit_triple"]) != triple_keys
            or any(
                not isinstance(row["audit_triple"][key], dict) or not row["audit_triple"][key] for key in triple_keys
            )
            or not isinstance(row["non_redundancy"], dict)
            or not row["non_redundancy"]
            or not isinstance(row["verdict_scope"], str)
            or not row["verdict_scope"]
        ):
            raise ReceiverError(f"E2 audit triple is incomplete: {row.get('member')}")
    matrix = value["ordered_redundancy_matrix"]
    expected_pairs = {(left, right) for left in stream_names for right in stream_names if left != right}
    observed_pairs: set[tuple[str, str]] = set()
    if not isinstance(matrix, list):
        raise ReceiverError("E2 redundancy matrix is not a list")
    for row in matrix:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "conditioned_bytes",
                "conditioner",
                "first_rung",
                "redundancy_bytes",
                "standalone_bytes",
                "stream",
            }
            or row["first_rung"] is not True
            or any(
                type(row[key]) is not int
                for key in (
                    "conditioned_bytes",
                    "redundancy_bytes",
                    "standalone_bytes",
                )
            )
        ):
            raise ReceiverError("E2 redundancy matrix row is malformed")
        observed_pairs.add((row["conditioner"], row["stream"]))
    if observed_pairs != expected_pairs or len(matrix) != len(expected_pairs):
        raise ReceiverError("E2 redundancy matrix does not cover every ordered pair")
    if (
        not isinstance(value["single_owner_facts"], list)
        or not value["single_owner_facts"]
        or not isinstance(value["correction_policy"], dict)
        or not value["correction_policy"]
    ):
        raise ReceiverError("E2 non-redundancy ownership contract is incomplete")
    for name in stream_names:
        payload = members[name]
        if not payload:
            raise ReceiverError(f"E2 audited member is empty: {name}")


def _validate_pose_contract(value: Any) -> None:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "classification",
            "compact_inverse_status",
            "exact_lattice_control",
            "nested_pose6",
            "packet_member",
            "receiver_bijection",
            "verdict_scope",
        }
        or value["classification"] != "ABSENT_FROM_COMPOSED_PACKET"
        or value["packet_member"] is not None
        or value["compact_inverse_status"] != "BLOCKED_NOT_PRESENT"
        or not isinstance(value["exact_lattice_control"], dict)
        or not isinstance(value["nested_pose6"], dict)
        or not isinstance(value["receiver_bijection"], dict)
        or not isinstance(value["verdict_scope"], str)
    ):
        raise ReceiverError("E2 pose-contract blocker is malformed")


def _chart_views(raw: bytes, chart: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expected = {
        "byteorder": "little",
        "dtype": "int16",
        "streams": [
            {"name": "anchors", "offset": 0, "shape": [600, 2, 3]},
            {"name": "gradients", "offset": 7200, "shape": [600, 2, 2, 3]},
            {"name": "residuals", "offset": 21600, "shape": [600, 2, 12, 16, 3]},
        ],
    }
    if chart != expected or sys.byteorder != "little":
        raise ReceiverError("chart layout or host byte order mismatch")
    if len(raw) != 1_404_000:
        raise ReceiverError("chart raw byte length mismatch")
    storage = torch.frombuffer(bytearray(raw), dtype=torch.int16)
    anchors = storage[0:3600].reshape(600, 2, 3)
    gradients = storage[3600:10800].reshape(600, 2, 2, 3)
    residuals = storage[10800:].reshape(600, 2, 12, 16, 3)
    return anchors, gradients, residuals


def _round_div_signed(value: torch.Tensor, denominator: int) -> torch.Tensor:
    half = denominator // 2
    return torch.where(
        value >= 0,
        torch.div(value + half, denominator, rounding_mode="floor"),
        -torch.div(-value + half, denominator, rounding_mode="floor"),
    )


def _official_pose_yuv6(camera: torch.Tensor) -> torch.Tensor:
    """Reproduce the pinned PoseNet resize and RGB-to-YUV6 input transform."""

    if (
        camera.ndim != 5
        or tuple(camera.shape[1:])
        != (
            FRAMES_PER_PAIR,
            CAMERA_H,
            CAMERA_W,
            CHANNELS,
        )
        or camera.dtype != torch.uint8
    ):
        raise ReceiverError("PoseNet input must be uint8 [B,2,874,1164,3]")
    batch = int(camera.shape[0])
    flat = (
        camera.permute(0, 1, 4, 2, 3)
        .contiguous()
        .reshape(batch * FRAMES_PER_PAIR, CHANNELS, CAMERA_H, CAMERA_W)
        .float()
    )
    resized = F.interpolate(
        flat,
        size=(PAIR_H, PAIR_W),
        mode="bilinear",
        align_corners=False,
    )
    red = resized[:, 0]
    green = resized[:, 1]
    blue = resized[:, 2]
    y = (red * 0.299 + green * 0.587 + blue * 0.114).clamp(0.0, 255.0)
    u = ((blue - y) / 1.772 + 128.0).clamp(0.0, 255.0)
    v = ((red - y) / 1.402 + 128.0).clamp(0.0, 255.0)
    u_sub = (u[:, 0::2, 0::2] + u[:, 1::2, 0::2] + u[:, 0::2, 1::2] + u[:, 1::2, 1::2]) * 0.25
    v_sub = (v[:, 0::2, 0::2] + v[:, 1::2, 0::2] + v[:, 0::2, 1::2] + v[:, 1::2, 1::2]) * 0.25
    yuv = torch.stack(
        (
            y[:, 0::2, 0::2],
            y[:, 1::2, 0::2],
            y[:, 0::2, 1::2],
            y[:, 1::2, 1::2],
            u_sub,
            v_sub,
        ),
        dim=1,
    )
    return yuv.reshape(batch, FRAMES_PER_PAIR, 6, PAIR_H // 2, PAIR_W // 2)


def _pose_moment_row(camera: torch.Tensor) -> dict[str, Any]:
    """Return mergeable float64 moments of the official 12-channel input."""

    value = _official_pose_yuv6(camera).reshape(int(camera.shape[0]), 12, PAIR_H // 2, PAIR_W // 2)
    return {
        "count": int(value.shape[0] * value.shape[2] * value.shape[3]),
        "sum": value.sum(dim=(0, 2, 3), dtype=torch.float64).tolist(),
        "sum_sq": value.square().sum(dim=(0, 2, 3), dtype=torch.float64).tolist(),
    }


def _merge_pose_moments(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ReceiverError("cannot merge an empty PoseNet moment inventory")
    count = 0
    sums = [0.0] * 12
    sums_sq = [0.0] * 12
    for row in rows:
        if (
            set(row) != {"count", "sum", "sum_sq"}
            or type(row["count"]) is not int
            or row["count"] <= 0
            or not isinstance(row["sum"], list)
            or not isinstance(row["sum_sq"], list)
            or len(row["sum"]) != 12
            or len(row["sum_sq"]) != 12
        ):
            raise ReceiverError("PoseNet moment row is malformed")
        count += int(row["count"])
        for index in range(12):
            sums[index] += float(row["sum"][index])
            sums_sq[index] += float(row["sum_sq"][index])
    return {"count": count, "sum": sums, "sum_sq": sums_sq}


def _derive_pa1_affine(
    moments: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive the FREE PA1 affine from decoded-content moments."""

    merged = _merge_pose_moments([moments])
    count = int(merged["count"])
    source_mean = [float(value) / count for value in merged["sum"]]
    source_variance = [
        max(float(merged["sum_sq"][index]) / count - source_mean[index] ** 2, 0.0) for index in range(12)
    ]
    if any(value <= 1e-12 for value in source_variance):
        raise ReceiverError("decoded PoseNet input variance is degenerate")
    gain64 = [math.sqrt(PA1_TARGET_RAW_VARIANCE[index] / source_variance[index]) for index in range(12)]
    bias64 = [PA1_TARGET_RAW_MEAN[index] - gain64[index] * source_mean[index] for index in range(12)]
    gain = torch.tensor(gain64, dtype=torch.float32)
    bias = torch.tensor(bias64, dtype=torch.float32)
    if not bool(torch.all(torch.isfinite(gain))) or not bool(torch.all(torch.isfinite(bias))):
        raise ReceiverError("derived PA1 affine is non-finite")
    return gain, bias


def _inverse_yuv6(yuv: torch.Tensor) -> torch.Tensor:
    if yuv.ndim != 5 or yuv.shape[2] != 6:
        raise ReceiverError("inverse YUV6 expects [B,2,6,H,W]")
    batch, frames, _channels, height, width = yuv.shape
    y = torch.empty(
        (batch, frames, height * 2, width * 2),
        dtype=yuv.dtype,
        device=yuv.device,
    )
    y[:, :, 0::2, 0::2] = yuv[:, :, 0]
    y[:, :, 1::2, 0::2] = yuv[:, :, 1]
    y[:, :, 0::2, 1::2] = yuv[:, :, 2]
    y[:, :, 1::2, 1::2] = yuv[:, :, 3]
    u = yuv[:, :, 4].repeat_interleave(2, -2).repeat_interleave(2, -1)
    v = yuv[:, :, 5].repeat_interleave(2, -2).repeat_interleave(2, -1)
    red = y + 1.402 * (v - 128.0)
    blue = y + 1.772 * (u - 128.0)
    green = (y - 0.299 * red - 0.114 * blue) / 0.587
    return torch.stack((red, green, blue), dim=2).clamp(0.0, 255.0)


def _apply_pa1_frame0_affine(
    camera: torch.Tensor,
    gain: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Apply PA1's exact frame-0 camera-side residual realization."""

    if gain.shape != (12,) or bias.shape != (12,):
        raise ReceiverError("PA1 affine must have 12 gain and bias values")
    raw_yuv = _official_pose_yuv6(camera)
    gain_view = gain.reshape(1, FRAMES_PER_PAIR, 6, 1, 1)
    bias_view = bias.reshape(1, FRAMES_PER_PAIR, 6, 1, 1)
    corrected_yuv = (raw_yuv * gain_view + bias_view).clamp(0.0, 255.0)
    corrected_low = _inverse_yuv6(corrected_yuv)
    baseline_low = _inverse_yuv6(raw_yuv)
    residual_low = corrected_low - baseline_low
    residual_low[:, 1].zero_()
    residual_camera = F.interpolate(
        residual_low.reshape(-1, CHANNELS, PAIR_H, PAIR_W),
        size=(CAMERA_H, CAMERA_W),
        mode="bilinear",
        align_corners=False,
    ).reshape(
        int(camera.shape[0]),
        FRAMES_PER_PAIR,
        CHANNELS,
        CAMERA_H,
        CAMERA_W,
    )
    source = camera.permute(0, 1, 4, 2, 3).contiguous().float()
    corrected = (source + residual_camera).clamp(0.0, 255.0).round()
    return corrected.to(torch.uint8).permute(0, 1, 3, 4, 2).contiguous()


def _render_batch(
    *,
    start: int,
    stop: int,
    anchors: torch.Tensor,
    gradients: torch.Tensor,
    residuals: torch.Tensor,
    labels: torch.Tensor,
    palette: torch.Tensor,
    camera_rows: torch.Tensor,
    camera_columns: torch.Tensor,
    semantic_frame_policy: str = "both_frames",
) -> torch.Tensor:
    batch_anchors = anchors[start:stop].to(torch.int64)
    batch_gradients = gradients[start:stop].to(torch.int64)
    batch_residuals = residuals[start:stop].to(torch.int64)
    rows = torch.arange(12, dtype=torch.int64).reshape(1, 1, 12, 1, 1)
    columns = torch.arange(16, dtype=torch.int64).reshape(1, 1, 1, 16, 1)
    row_term = _round_div_signed(
        batch_gradients[:, :, 0, :].reshape(stop - start, 2, 1, 1, 3) * (2 * rows - 11),
        22,
    )
    column_term = _round_div_signed(
        batch_gradients[:, :, 1, :].reshape(stop - start, 2, 1, 1, 3) * (2 * columns - 15),
        30,
    )
    charts = batch_anchors.reshape(stop - start, 2, 1, 1, 3) + row_term + column_term + batch_residuals
    if bool(torch.any((charts < 0) | (charts > 255))):
        raise ReceiverError("chart reconstruction escaped uint8")
    grid = charts.to(torch.uint8).repeat_interleave(32, dim=2).repeat_interleave(32, dim=3)
    flat = grid.reshape((stop - start) * 2, PAIR_H, PAIR_W, 3).permute(0, 3, 1, 2).float()
    with torch.inference_mode():
        camera = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
        camera = torch.clamp(torch.round(camera), 0.0, 255.0).to(torch.uint8)
    camera = camera.permute(0, 2, 3, 1).reshape(stop - start, 2, CAMERA_H, CAMERA_W, CHANNELS)
    overlay = labels[start:stop].index_select(1, camera_rows).index_select(2, camera_columns)
    if semantic_frame_policy not in {
        "both_frames",
        "frame1_only_seg_free_frame0",
    }:
        raise ReceiverError("semantic frame policy is unknown")
    for code in range(1, palette.shape[0]):
        mask = (overlay == code).reshape(stop - start, CAMERA_H, CAMERA_W, 1)
        colour = palette[code].reshape(1, 1, 1, CHANNELS)
        if semantic_frame_policy == "both_frames":
            camera = torch.where(
                mask.reshape(stop - start, 1, CAMERA_H, CAMERA_W, 1),
                colour.reshape(1, 1, 1, 1, CHANNELS),
                camera,
            )
        else:
            camera = torch.stack(
                (
                    camera[:, 0],
                    torch.where(mask, colour, camera[:, 1]),
                ),
                dim=1,
            )
    contiguous = camera.contiguous()
    if contiguous.numel() != ((stop - start) * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS):
        raise ReceiverError("rendered batch size mismatch")
    return contiguous


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    if temporary.exists():
        raise ReceiverError(f"stale temporary metadata path exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_preserved_stage(
    *,
    stage_path: Path,
    state_path: Path,
    manifest_sha256: str,
    start: int,
    stop: int,
    expected_bytes: int,
) -> dict[str, Any] | None:
    if not state_path.exists():
        return None
    if not stage_path.is_file():
        raise ReceiverError("stage metadata exists without stage bytes")
    state = _duplicate_refusing_json(state_path.read_bytes(), label=state_path.name)
    if (
        not isinstance(state, dict)
        or set(state)
        != {
            "bytes",
            "manifest_sha256",
            "pair_start",
            "pair_stop",
            "sha256",
        }
        or state["bytes"] != expected_bytes
        or state["manifest_sha256"] != manifest_sha256
        or state["pair_start"] != start
        or state["pair_stop"] != stop
        or _sha256_file(stage_path) != (state["bytes"], state["sha256"])
    ):
        raise ReceiverError("preserved stage custody mismatch")
    return state


def _write_or_adopt_rendered_stage(
    *,
    stage_path: Path,
    state_path: Path,
    rendered: torch.Tensor,
    manifest_sha256: str,
    start: int,
    stop: int,
) -> dict[str, Any]:
    expected_bytes = int(rendered.numel())
    temporary = stage_path.with_name(stage_path.name + f".partial.{os.getpid()}")
    if temporary.exists():
        raise ReceiverError(f"stale stage temporary exists: {temporary}")
    with temporary.open("xb") as handle:
        rendered.untyped_storage()._write_file(handle, False, False, 1)
        handle.flush()
        os.fsync(handle.fileno())
    observed_bytes, digest = _sha256_file(temporary)
    if observed_bytes != expected_bytes:
        raise ReceiverError("rendered stage file length mismatch")
    expected = {
        "bytes": observed_bytes,
        "manifest_sha256": manifest_sha256,
        "pair_start": start,
        "pair_stop": stop,
        "sha256": digest,
    }
    if stage_path.exists():
        if _sha256_file(stage_path) != (observed_bytes, digest):
            raise ReceiverError("orphaned stage bytes differ from deterministic replay")
        temporary.unlink()
    else:
        os.replace(temporary, stage_path)
    _atomic_json(state_path, expected)
    return expected


def _load_stage_tensor(
    *,
    stage_path: Path,
    start: int,
    stop: int,
) -> torch.Tensor:
    expected_bytes = (stop - start) * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS
    if not stage_path.is_file() or stage_path.stat().st_size != expected_bytes:
        raise ReceiverError("preserved stage is absent or has the wrong length")
    return torch.frombuffer(bytearray(stage_path.read_bytes()), dtype=torch.uint8).reshape(
        stop - start,
        FRAMES_PER_PAIR,
        CAMERA_H,
        CAMERA_W,
        CHANNELS,
    )


def _load_or_measure_pose_moments(
    *,
    stage_path: Path,
    state_path: Path,
    stage_row: Mapping[str, Any],
    manifest_sha256: str,
    start: int,
    stop: int,
) -> dict[str, Any]:
    expected_binding = {
        "manifest_sha256": manifest_sha256,
        "pair_start": start,
        "pair_stop": stop,
        "stage_sha256": stage_row["sha256"],
    }
    if state_path.exists():
        value = _duplicate_refusing_json(state_path.read_bytes(), label=state_path.name)
        if (
            not isinstance(value, dict)
            or set(value) != {*expected_binding, "moments"}
            or any(value[key] != expected for key, expected in expected_binding.items())
        ):
            raise ReceiverError("preserved PA1 moment custody mismatch")
        return _merge_pose_moments([value["moments"]])
    moments = _pose_moment_row(_load_stage_tensor(stage_path=stage_path, start=start, stop=stop))
    _atomic_json(state_path, {**expected_binding, "moments": moments})
    return moments


def _load_or_write_pa1_affine(
    *,
    path: Path,
    manifest_sha256: str,
    moments: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    merged = _merge_pose_moments([moments])
    gain, bias = _derive_pa1_affine(merged)
    expected = {
        "bias_f32": bias.tolist(),
        "gain_f32": gain.tolist(),
        "manifest_sha256": manifest_sha256,
        "moments": merged,
        "payload_bytes": 0,
        "rate_class": "FREE",
        "target_derivation": ("frozen_posenet_first_stem_conv_and_bn_only_video_independent"),
        "transform_id": PA1_TRANSFORM_ID,
    }
    if path.exists():
        value = _duplicate_refusing_json(path.read_bytes(), label=path.name)
        if value != expected:
            raise ReceiverError("preserved PA1 affine custody mismatch")
    else:
        _atomic_json(path, expected)
    return gain, bias, expected


def _assemble_final(
    *,
    final_path: Path,
    stage_rows: list[dict[str, Any]],
    stage_paths: list[Path],
    expected_bytes: int,
    expected_sha256: str,
) -> tuple[int, str]:
    if final_path.exists():
        observed = _sha256_file(final_path)
        if observed != (expected_bytes, expected_sha256):
            raise ReceiverError("existing final raw custody mismatch")
        return observed
    temporary = final_path.with_name(final_path.name + f".partial.{os.getpid()}")
    digest = hashlib.sha256()
    total = 0
    with temporary.open("xb") as output:
        for row, stage_path in zip(stage_rows, stage_paths, strict=True):
            stage_bytes, stage_sha256 = _sha256_file(stage_path)
            if (stage_bytes, stage_sha256) != (row["bytes"], row["sha256"]):
                raise ReceiverError("stage changed before final assembly")
            with stage_path.open("rb") as source:
                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    observed = (total, digest.hexdigest())
    if observed != (expected_bytes, expected_sha256):
        raise ReceiverError(f"final raw identity mismatch: {observed!r} != {(expected_bytes, expected_sha256)!r}")
    os.replace(temporary, final_path)
    return observed


def inflate(archive_dir: Path, output_dir: Path, video_names_file: Path) -> dict[str, Any]:
    started = time.monotonic()
    # PA1 was measured with this fixed thread count. PyTorch's bilinear CPU
    # kernel resolves a small number of float32 rounding ties differently at
    # one thread, so the count is part of the deterministic runtime contract.
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)

    names = [row.strip() for row in video_names_file.read_text(encoding="utf-8").splitlines() if row.strip()]
    if len(names) != 1:
        raise ReceiverError("DDM E1 packet describes exactly one video")
    relative = _safe_relative_name(names[0]).with_suffix(".raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir.joinpath(*relative.parts)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    members = _read_exact_members(archive_dir)
    manifest_payload = members["manifest.json"]
    manifest_sha256 = _sha256(manifest_payload)
    manifest = _validate_manifest(_duplicate_refusing_json(manifest_payload, label="manifest.json"), members)
    semantic_frame_policy = str(manifest.get("semantic_frame_policy", "both_frames"))
    chart_raw, chart_shape = _parse_blob(members["base/chart.ddb"], expected_kind=0, label="base/chart.ddb")
    if chart_shape:
        raise ReceiverError("chart blob must be an opaque rank-zero container")
    semantic_raw, semantic_shape = _parse_blob(
        members["semantic/composed.dds"],
        expected_kind=1,
        label="semantic/composed.dds",
    )
    if semantic_shape != (600, PAIR_H, PAIR_W):
        raise ReceiverError("semantic shape mismatch")
    anchors, gradients, residuals = _chart_views(chart_raw, manifest["chart"])
    labels = torch.frombuffer(bytearray(semantic_raw), dtype=torch.uint8).reshape(600, PAIR_H, PAIR_W)
    output = manifest["output"]
    if set(output) != {"bytes", "palette_rgb_u8", "sha256"}:
        raise ReceiverError("output manifest keys differ")
    palette_value = output["palette_rgb_u8"]
    if (
        not isinstance(palette_value, list)
        or not 2 <= len(palette_value) <= 16
        or any(
            not isinstance(row, list)
            or len(row) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255 for value in row)
            for row in palette_value
        )
    ):
        raise ReceiverError("output palette is invalid")
    palette = torch.tensor(palette_value, dtype=torch.uint8)
    if int(labels.max()) >= len(palette):
        raise ReceiverError("semantic code escaped the counted palette")
    expected_bytes = 600 * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS
    if output["bytes"] != expected_bytes:
        raise ReceiverError("output byte count differs from geometry")
    expected_sha256 = output["sha256"]
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ReceiverError("output SHA-256 is invalid")

    checkpoint_root = (
        output_dir.parent
        / ".ddm_runtime_checkpoints"
        / manifest_sha256
        / relative.with_suffix("").as_posix().replace("/", "__")
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(checkpoint_root).free
    amplitude_enabled = manifest["schema"] == E3_SCHEMA
    required_free_bytes = expected_bytes * (3 if amplitude_enabled else 2) + SAFETY_BYTES
    if free_bytes < required_free_bytes:
        raise ReceiverError(f"storage preflight failed: {free_bytes} < {required_free_bytes}")

    batch_pairs = int(manifest["state"]["batch_pairs"])
    if batch_pairs != 16:
        raise ReceiverError("stage checkpoint interval changed")
    camera_rows = torch.div(
        torch.arange(CAMERA_H, dtype=torch.int64) * PAIR_H,
        CAMERA_H,
        rounding_mode="floor",
    )
    camera_columns = torch.div(
        torch.arange(CAMERA_W, dtype=torch.int64) * PAIR_W,
        CAMERA_W,
        rounding_mode="floor",
    )
    base_rows: list[dict[str, Any]] = []
    base_paths: list[Path] = []
    moment_rows: list[dict[str, Any]] = []
    render_seconds = 0.0
    for start in range(0, 600, batch_pairs):
        stop = min(start + batch_pairs, 600)
        stage_prefix = "base_pairs" if amplitude_enabled else "pairs"
        stage_path = checkpoint_root / f"{stage_prefix}_{start:04d}_{stop:04d}.raw"
        state_path = checkpoint_root / f"{stage_prefix}_{start:04d}_{stop:04d}.json"
        expected_stage_bytes = (stop - start) * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS
        preserved = _load_preserved_stage(
            stage_path=stage_path,
            state_path=state_path,
            manifest_sha256=manifest_sha256,
            start=start,
            stop=stop,
            expected_bytes=expected_stage_bytes,
        )
        if preserved is not None:
            row = preserved
        else:
            stage_started = time.monotonic()
            rendered = _render_batch(
                start=start,
                stop=stop,
                anchors=anchors,
                gradients=gradients,
                residuals=residuals,
                labels=labels,
                palette=palette,
                camera_rows=camera_rows,
                camera_columns=camera_columns,
                semantic_frame_policy=semantic_frame_policy,
            )
            render_seconds += time.monotonic() - stage_started
            row = _write_or_adopt_rendered_stage(
                stage_path=stage_path,
                state_path=state_path,
                rendered=rendered,
                manifest_sha256=manifest_sha256,
                start=start,
                stop=stop,
            )
        base_rows.append(row)
        base_paths.append(stage_path)
        if amplitude_enabled:
            moment_rows.append(
                _load_or_measure_pose_moments(
                    stage_path=stage_path,
                    state_path=checkpoint_root / f"base_pairs_{start:04d}_{stop:04d}.moments.json",
                    stage_row=row,
                    manifest_sha256=manifest_sha256,
                    start=start,
                    stop=stop,
                )
            )

    amplitude_receipt: dict[str, Any] | None = None
    stage_rows = base_rows
    stage_paths = base_paths
    if amplitude_enabled:
        gain, bias, amplitude_receipt = _load_or_write_pa1_affine(
            path=checkpoint_root / "pa1_affine.json",
            manifest_sha256=manifest_sha256,
            moments=_merge_pose_moments(moment_rows),
        )
        stage_rows = []
        stage_paths = []
        for start in range(0, 600, batch_pairs):
            stop = min(start + batch_pairs, 600)
            stage_path = checkpoint_root / f"pairs_{start:04d}_{stop:04d}.raw"
            state_path = checkpoint_root / f"pairs_{start:04d}_{stop:04d}.json"
            expected_stage_bytes = (stop - start) * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS
            preserved = _load_preserved_stage(
                stage_path=stage_path,
                state_path=state_path,
                manifest_sha256=manifest_sha256,
                start=start,
                stop=stop,
                expected_bytes=expected_stage_bytes,
            )
            if preserved is not None:
                row = preserved
            else:
                stage_started = time.monotonic()
                rendered = _apply_pa1_frame0_affine(
                    _load_stage_tensor(
                        stage_path=base_paths[start // batch_pairs],
                        start=start,
                        stop=stop,
                    ),
                    gain,
                    bias,
                )
                render_seconds += time.monotonic() - stage_started
                row = _write_or_adopt_rendered_stage(
                    stage_path=stage_path,
                    state_path=state_path,
                    rendered=rendered,
                    manifest_sha256=manifest_sha256,
                    start=start,
                    stop=stop,
                )
            stage_rows.append(row)
            stage_paths.append(stage_path)
    final_bytes, final_sha256 = _assemble_final(
        final_path=final_path,
        stage_rows=stage_rows,
        stage_paths=stage_paths,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )
    receipt = {
        "archive_manifest_sha256": manifest_sha256,
        "block_versions": manifest["block_versions"],
        "amplitude_transform": amplitude_receipt,
        "dependencies": ["torch"],
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "final": {
            "bytes": final_bytes,
            "path": str(final_path),
            "sha256": final_sha256,
        },
        "member_consumption": [
            {
                "bytes": len(members[name]),
                "member": name,
                "sha256": _sha256(members[name]),
            }
            for name in EXPECTED_MEMBERS
        ],
        "pair_count": 600,
        "language_version": manifest["language_version"],
        "render_seconds": format(render_seconds, ".6f"),
        "research_only": True,
        "rate_partition": {
            "COUNTED": {
                "member_payload_bytes": sum(len(members[name]) for name in EXPECTED_MEMBERS),
                "members": list(EXPECTED_MEMBERS),
            },
            "FREE": {
                "bytes": 0,
                "objects": [PA1_TRANSFORM_ID] if amplitude_enabled else [],
            },
            "NULL": {
                "blocks": ["D2", "D5"],
                "bytes": 0,
            },
        },
        "resume": {
            "all_stage_checkpoints_preserved": True,
            "checkpoint_root": str(checkpoint_root),
            "stage_count": len(stage_rows),
        },
        "schema": (
            "ddm_e3_runtime_inflate_receipt.v1"
            if manifest["schema"] == E3_SCHEMA
            else "ddm_e2_runtime_inflate_receipt.v1"
            if manifest["schema"] == E2_SCHEMA
            else "ddm_e1_runtime_inflate_receipt.v1"
        ),
        "score_claim": False,
        "fixed_torch_threads": 4,
        "single_thread_cpu": False,
        "staleness": {
            "consumption_time_input_hashes_verified": True,
            "policy": "fail_closed_on_hash_or_version_mismatch",
        },
        "total_seconds": format(time.monotonic() - started, ".6f"),
    }
    _atomic_json(checkpoint_root / "inflate_receipt.json", receipt)
    return receipt


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        raise SystemExit("Usage: inflate.py <archive_dir> <output_dir> <video_names_file>")
    receipt = inflate(
        Path(argv[1]).resolve(),
        Path(argv[2]).resolve(),
        Path(argv[3]).resolve(),
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
