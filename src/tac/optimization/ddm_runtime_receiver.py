# SPDX-License-Identifier: MIT
"""Standalone deterministic receiver emitted by the DDM E1/E2 exporter.

This file is copied byte-for-byte to ``inflate.py``.  It intentionally imports
only the Python standard library plus Torch and Brotli.  All instance-derived
bytes live in the extracted archive directory and are covered by the manifest.
"""

from __future__ import annotations

import base64
import hashlib
import json
import lzma
import math
import os
import shutil
import struct
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import brotli
except ImportError:  # E4 raw-LZMA1 fallback packets must remain decodable.
    brotli = None  # type: ignore[assignment]

import torch
import torch.nn.functional as F

SCHEMA = "ddm_e1_runtime_archive.v1"
E2_SCHEMA = "ddm_e2_runtime_archive.v1"
E3_SCHEMA = "ddm_e3_runtime_archive.v1"
E4_SCHEMA = "ddm_e4_runtime_archive.v1"
E4_WS1_SCHEMA = "ddm_e4_ws1_runtime_archive.v1"
E5A_SCHEMA = "ddm_e5a_midcampaign_runtime_archive.v1"
IC1_SCHEMA = "ddm_ic1_runtime_archive.v1"
IC2_SCHEMA = "ddm_ic2_runtime_archive.v1"
CB1_SCHEMA = "ddm_cb1_rg4_perclass_runtime_archive.v1"
RECEIVER_GRAMMAR_ADMISSION_SCHEMA = "ddm_receiver_grammar_admission.v1"
RATE_DOCTRINE_SCHEMA = "ddm_four_clause_rate_doctrine.v1"
TYPED_STREAM_SCHEMA = "ddm_typed_stream_tag.v1"
STREAM_TYPES = {"SKELETON", "CONNECTION", "FIBER", "GAUGE", "RESIDUAL"}
LAYER_HOMES = {
    "L1_program",
    "L2_chart",
    "L3_raster",
    "L4_scorer_feature",
    "L5_verdict",
}
BLOB_MAGIC = b"DDE1B"
BLOB_HEADER = struct.Struct(">5sBBBBQQ32s")
E5A_BUNDLE_MAGIC = b"E5AL1"
E5A_BUNDLE_PREFIX = struct.Struct(">5sBBB")
E5A_BUNDLE_ENTRY = struct.Struct(">BBI")
E5A_CODECS = (
    "RAW_EXPLICIT",
    "BROTLI_Q11",
    "RAW_LZMA1",
    "G4_FREE_DECODER_CONTEXT",
    "BELLARD_KT_MIXER",
)
E5A_COMPONENTS = (
    "predictor_archive",
    "worldsheet_g1_payload",
    "realization_profile_payload",
    "scorer_solved_templates_payload",
    "lane_programs_payload",
    "coupled_margin_program",
    "preuint8_q8_program",
    "warm_start_payload",
)
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
EXPECTED_WS1_MEMBERS = ("manifest.json", "state/ws1.ddj5")
EXPECTED_CB1_MEMBERS = ("manifest.json", "state/rg4.ddr4")
EXPECTED_CC3_MEMBERS = ("manifest/pc1.json", "pose/pc1.ddp", "parent/ws1.zip")
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
WS1_GRAMMAR_VERSION = "ddm_ws1_receiver_closed_warm_start.v1"
WS1_SOURCE_BUNDLE_B85 = b""
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


def _read_exact_ws1_members(archive_dir: Path) -> dict[str, bytes]:
    observed = tuple(
        sorted(path.relative_to(archive_dir).as_posix() for path in archive_dir.rglob("*") if path.is_file())
    )
    if observed != tuple(sorted(EXPECTED_WS1_MEMBERS)):
        raise ReceiverError(f"extracted WS1 packet members differ: {observed!r} != {EXPECTED_WS1_MEMBERS!r}")
    result: dict[str, bytes] = {}
    for name in EXPECTED_WS1_MEMBERS:
        path = archive_dir / name
        if path.is_symlink() or not path.is_file():
            raise ReceiverError(f"WS1 packet member is not one regular file: {name}")
        result[name] = path.read_bytes()
    return result


def _read_exact_cb1_members(archive_dir: Path) -> dict[str, bytes]:
    observed = tuple(
        sorted(path.relative_to(archive_dir).as_posix() for path in archive_dir.rglob("*") if path.is_file())
    )
    if observed != tuple(sorted(EXPECTED_CB1_MEMBERS)):
        raise ReceiverError(f"extracted CB1 packet members differ: {observed!r} != {EXPECTED_CB1_MEMBERS!r}")
    result: dict[str, bytes] = {}
    for name in EXPECTED_CB1_MEMBERS:
        path = archive_dir / name
        if path.is_symlink() or not path.is_file():
            raise ReceiverError(f"CB1 packet member is not one regular file: {name}")
        result[name] = path.read_bytes()
    return result


def _read_exact_cc3_members(archive_dir: Path) -> dict[str, bytes]:
    expected_directories = {
        PurePosixPath(name).parent.as_posix()
        for name in EXPECTED_CC3_MEMBERS
        if PurePosixPath(name).parent != PurePosixPath(".")
    }
    observed_files: list[str] = []
    for path in archive_dir.rglob("*"):
        relative = path.relative_to(archive_dir).as_posix()
        if path.is_symlink():
            raise ReceiverError(f"CC3 extracted archive contains a symlink: {relative}")
        if path.is_dir():
            if relative not in expected_directories:
                raise ReceiverError(f"CC3 extracted archive contains an extra directory: {relative}")
            continue
        if not path.is_file():
            raise ReceiverError(f"CC3 extracted archive contains a non-regular entry: {relative}")
        observed_files.append(relative)
    observed = tuple(sorted(observed_files))
    if observed != tuple(sorted(EXPECTED_CC3_MEMBERS)):
        raise ReceiverError(f"CC3 extracted archive members differ: {observed!r} != {EXPECTED_CC3_MEMBERS!r}")
    result: dict[str, bytes] = {}
    for name in EXPECTED_CC3_MEMBERS:
        path = archive_dir / name
        if path.is_symlink() or not path.is_file():
            raise ReceiverError(f"CC3 archive member is not one regular file: {name}")
        result[name] = path.read_bytes()
    return result


def _parse_blob(payload: bytes, *, expected_kind: int, label: str) -> tuple[bytes, tuple[int, ...]]:
    if len(payload) < BLOB_HEADER.size:
        raise ReceiverError(f"{label} frame is truncated")
    magic, version, codec, kind, rank, raw_bytes, coded_bytes, raw_digest = BLOB_HEADER.unpack_from(payload)
    if magic != BLOB_MAGIC or version != 1 or codec not in (1, 2) or kind != expected_kind or rank > 8:
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
    if codec == 1:
        if brotli is None:
            raise ReceiverError(f"{label} requires declared Brotli dependency")
        try:
            raw = brotli.decompress(payload[coded_offset:])
        except brotli.error as exc:
            raise ReceiverError(f"{label} Brotli decode failed") from exc
    else:
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
    coder: str = "lzma1_raw_d1m_lc3_lp0_pb2",
) -> None:
    order = ["L", "D2", "D1", "D4", "D6", "D5"]
    versions = {
        "L": "ddm_L_composed_semantic.v1",
        "D2": "ddm_D2_inactive.v1",
        "D1": ("ddm_D1_pa1_scorer_stat_affine_free.v1" if amplitude_enabled else "ddm_D1_inactive.v1"),
        "D4": ("ddm_D4_brotli_q11_measure.v1" if coder == "brotli_q11" else "ddm_D4_lzma1_raw_d1m_measure.v1"),
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
    if schema not in {SCHEMA, E2_SCHEMA, E3_SCHEMA, E4_SCHEMA} or set(manifest) != expected_keys:
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
    member_codecs = {
        BLOB_HEADER.unpack_from(members[name])[2]
        for name in EXPECTED_MEMBERS[1:]
        if len(members[name]) >= BLOB_HEADER.size
    }
    if member_codecs == {1}:
        coder = "brotli_q11"
        expected_dependencies = ["torch", "brotli"]
    elif member_codecs == {2}:
        coder = "lzma1_raw_d1m_lc3_lp0_pb2"
        expected_dependencies = ["torch"]
    else:
        raise ReceiverError("runtime section coder contract changed")
    if schema != E4_SCHEMA and coder != "lzma1_raw_d1m_lc3_lp0_pb2":
        raise ReceiverError("pre-E4 runtime section coder changed")
    if manifest["dependencies"] != expected_dependencies:
        raise ReceiverError("runtime dependency contract changed")
    if manifest["language_version"] != LANGUAGE_VERSION:
        raise ReceiverError("language-version stamp changed")
    _validate_block_versions(
        manifest["block_versions"],
        members,
        amplitude_enabled=schema in {E3_SCHEMA, E4_SCHEMA},
        coder=coder,
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
    if schema in {E2_SCHEMA, E3_SCHEMA, E4_SCHEMA}:
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
        if set(row) != {"bytes", "member", "sha256", "typed_stream_tag"}:
            raise ReceiverError("manifest section keys differ")
        payload = members[row["member"]]
        if row["bytes"] != len(payload) or row["sha256"] != _sha256(payload):
            raise ReceiverError(f"manifest section custody mismatch: {row['member']}")
        tag = row["typed_stream_tag"]
        if (
            not isinstance(tag, dict)
            or set(tag)
            != {
                "schema",
                "type",
                "layer_home",
                "evaluate_py_recursion_level_cited",
                "counted_bytes",
                "free_receiver_code",
            }
            or tag["schema"] != TYPED_STREAM_SCHEMA
            or tag["type"] not in STREAM_TYPES
            or tag["layer_home"] not in LAYER_HOMES
            or not isinstance(tag["evaluate_py_recursion_level_cited"], str)
            or not tag["evaluate_py_recursion_level_cited"].strip()
            or isinstance(tag["counted_bytes"], bool)
            or not isinstance(tag["counted_bytes"], int)
            or tag["counted_bytes"] < 0
            or type(tag["free_receiver_code"]) is not bool
            or tag["counted_bytes"] != row["bytes"]
            or (tag["type"] == "GAUGE" and tag["counted_bytes"] != 0)
        ):
            raise ReceiverError(f"manifest typed-stream custody mismatch: {row['member']}")
    return manifest


def _validate_ws1_manifest(
    manifest: Any,
    members: dict[str, bytes],
) -> dict[str, Any]:
    """Validate the explicit WS1/J5 grammar admission route."""

    if not isinstance(manifest, dict):
        raise ReceiverError("WS1 manifest is not an object")
    is_ic1 = manifest.get("schema") == IC1_SCHEMA
    is_ic2 = manifest.get("schema") == IC2_SCHEMA
    is_e5a = manifest.get("schema") == E5A_SCHEMA
    amplitude_enabled = is_ic1 or is_ic2
    expected_keys = {
        "dependencies",
        "false_authority",
        "geometry",
        "grammar_admission",
        "output",
        "schema",
        "sections",
        "state",
    }
    if amplitude_enabled:
        expected_keys.add("amplitude_transform")
    if is_e5a:
        expected_keys.add("la1_framing")
    if set(manifest) != expected_keys or manifest.get("schema") not in {
        E4_WS1_SCHEMA,
        E5A_SCHEMA,
        IC1_SCHEMA,
        IC2_SCHEMA,
    }:
        raise ReceiverError("WS1 manifest keys differ from the sealed schema")
    if is_ic1 and manifest["amplitude_transform"] != {
        "application_frame": 0,
        "composition_order": "W_joint_then_PA1_frame0",
        "payload_bytes": 0,
        "rate_class": "FREE",
        "target_derivation": ("frozen_posenet_first_stem_conv_and_bn_only_video_independent"),
        "transform_id": PA1_TRANSFORM_ID,
    }:
        raise ReceiverError("IC1 amplitude composition contract changed")
    if is_ic2 and manifest["amplitude_transform"] != {
        "application_frame": 0,
        "composition_order": "W_seg_then_PA1_frame0",
        "payload_bytes": 0,
        "rate_class": "FREE",
        "target_derivation": ("frozen_posenet_first_stem_conv_and_bn_only_video_independent"),
        "transform_id": PA1_TRANSFORM_ID,
    }:
        raise ReceiverError("IC2 amplitude composition contract changed")
    if manifest["false_authority"] != {
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "research_only": True,
        "score_claim": False,
    }:
        raise ReceiverError("WS1 manifest false-authority contract changed")
    if manifest["geometry"] != {
        "camera_hw": [CAMERA_H, CAMERA_W],
        "channels": CHANNELS,
        "frames_per_pair": FRAMES_PER_PAIR,
        "pair_count": 600,
        "scorer_hw": [PAIR_H, PAIR_W],
    }:
        raise ReceiverError("WS1 manifest geometry mismatch")
    state = manifest["state"]
    if (
        not isinstance(state, dict)
        or set(state)
        != {
            "batch_pairs",
            "name",
            "receiver_effective_dofs",
        }
        or state["batch_pairs"] != (16 if amplitude_enabled else 32)
        or not isinstance(state["name"], str)
        or not state["name"]
        or state["receiver_effective_dofs"] != 368
    ):
        raise ReceiverError("WS1 receiver-effective state changed")
    framed = members["state/ws1.ddj5"]
    if is_e5a:
        if len(framed) < E5A_BUNDLE_PREFIX.size:
            raise ReceiverError("E5A LA1 bundle is truncated")
        magic, version, component_count, reserved = E5A_BUNDLE_PREFIX.unpack_from(framed)
        la1 = manifest["la1_framing"]
        if (
            magic != E5A_BUNDLE_MAGIC
            or version != 1
            or reserved != 0
            or not isinstance(la1, dict)
            or set(la1)
            != {
                "bundle_bytes",
                "bundle_sha256",
                "component_count",
                "prior_prospective_bytes",
                "schema",
                "selection",
            }
            or la1
            != {
                "bundle_bytes": len(framed),
                "bundle_sha256": _sha256(framed),
                "component_count": component_count,
                "prior_prospective_bytes": 128254,
                "schema": "ddm_e5a_la1_semantic_bundle.v1",
                "selection": "minimum_exact_framed_bytes_then_codec_name",
            }
        ):
            raise ReceiverError("E5A LA1 framing contract changed")
        expected_dependencies = ["numpy", "scipy", "torch", "brotli"]
    elif len(framed) < BLOB_HEADER.size:
        raise ReceiverError("WS1 state frame is truncated")
    elif BLOB_HEADER.unpack_from(framed)[2] == 1:
        expected_dependencies = (
            ["numpy", "torch", "brotli", "cv2"]
            if is_ic2
            else ["numpy", "scipy", "torch", "brotli"]
        )
    elif BLOB_HEADER.unpack_from(framed)[2] == 2:
        expected_dependencies = ["numpy", "scipy", "torch"]
    else:
        raise ReceiverError("WS1 packet coder contract changed")
    if manifest["dependencies"] != expected_dependencies:
        raise ReceiverError("WS1 runtime dependency contract changed")
    sections = manifest["sections"]
    if (
        not isinstance(sections, list)
        or len(sections) != 1
        or not isinstance(sections[0], dict)
        or set(sections[0]) != {"bytes", "member", "sha256", "typed_stream_tag"}
        or sections[0]["member"] != "state/ws1.ddj5"
        or sections[0]["bytes"] != len(framed)
        or sections[0]["sha256"] != _sha256(framed)
    ):
        raise ReceiverError("WS1 packet section custody differs")
    tag = sections[0]["typed_stream_tag"]
    if (
        not isinstance(tag, dict)
        or set(tag)
        != {
            "schema",
            "type",
            "layer_home",
            "evaluate_py_recursion_level_cited",
            "counted_bytes",
            "free_receiver_code",
        }
        or tag
        != {
            "schema": TYPED_STREAM_SCHEMA,
            "type": "SKELETON",
            "layer_home": "L1_program",
            "evaluate_py_recursion_level_cited": ("L1_program -> L3_raster -> L4_scorer_feature -> L5_verdict"),
            "counted_bytes": len(framed),
            "free_receiver_code": True,
        }
    ):
        raise ReceiverError("WS1 packet typed-stream custody differs")
    output = manifest["output"]
    if (
        not isinstance(output, dict)
        or set(output) != {"bytes", "sha256"}
        or output["bytes"] != 600 * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS
        or not _is_lower_sha256(output["sha256"])
    ):
        raise ReceiverError("WS1 output identity is malformed")
    return manifest


def _validate_cb1_manifest(
    manifest: Any,
    members: dict[str, bytes],
) -> dict[str, Any]:
    """Validate one opaque RG4 source-local state on E4's framed coder path."""

    if not isinstance(manifest, dict) or set(manifest) != {
        "dependencies",
        "false_authority",
        "geometry",
        "output",
        "schema",
        "sections",
        "state",
    }:
        raise ReceiverError("CB1 manifest keys differ from the sealed schema")
    if manifest.get("schema") != CB1_SCHEMA:
        raise ReceiverError("CB1 manifest schema differs")
    if manifest["false_authority"] != {
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "research_only": True,
        "score_claim": False,
    }:
        raise ReceiverError("CB1 false-authority contract changed")
    if manifest["geometry"] != {
        "camera_hw": [CAMERA_H, CAMERA_W],
        "channels": CHANNELS,
        "frames_per_pair": FRAMES_PER_PAIR,
        "pair_count": 600,
        "scorer_hw": [PAIR_H, PAIR_W],
    }:
        raise ReceiverError("CB1 geometry differs")
    framed = members["state/rg4.ddr4"]
    if len(framed) < BLOB_HEADER.size:
        raise ReceiverError("CB1 state frame is truncated")
    codec = BLOB_HEADER.unpack_from(framed)[2]
    expected_dependencies = (
        ["numpy", "scipy", "torch", "brotli"]
        if codec == 1
        else ["numpy", "scipy", "torch"]
        if codec == 2
        else None
    )
    if expected_dependencies is None or manifest["dependencies"] != expected_dependencies:
        raise ReceiverError("CB1 runtime dependency contract changed")
    sections = manifest["sections"]
    if (
        not isinstance(sections, list)
        or len(sections) != 1
        or not isinstance(sections[0], dict)
        or set(sections[0]) != {"bytes", "member", "sha256", "typed_stream_tag"}
        or sections[0]["member"] != "state/rg4.ddr4"
        or sections[0]["bytes"] != len(framed)
        or sections[0]["sha256"] != _sha256(framed)
    ):
        raise ReceiverError("CB1 packet section custody differs")
    tag = sections[0]["typed_stream_tag"]
    if (
        not isinstance(tag, dict)
        or tag
        != {
            "schema": TYPED_STREAM_SCHEMA,
            "type": "SKELETON",
            "layer_home": "L1_program",
            "evaluate_py_recursion_level_cited": (
                "L1_program -> L3_raster -> L4_scorer_feature -> L5_verdict"
            ),
            "counted_bytes": len(framed),
            "free_receiver_code": True,
        }
    ):
        raise ReceiverError("CB1 packet typed-stream custody differs")
    state = manifest["state"]
    if (
        not isinstance(state, dict)
        or set(state)
        != {
            "batch_pairs",
            "name",
            "parent_archive_sha256",
            "pc1_packet_sha256",
            "source_archive_bytes",
            "source_archive_sha256",
            "source_schema",
        }
        or state["batch_pairs"] != 32
        or not isinstance(state["name"], str)
        or not state["name"]
        or state["source_schema"] != "ddm_rg4_pc1_source_local_composition.v1"
        or isinstance(state["source_archive_bytes"], bool)
        or not isinstance(state["source_archive_bytes"], int)
        or state["source_archive_bytes"] <= 0
        or any(
            not _is_lower_sha256(state[key])
            for key in (
                "parent_archive_sha256",
                "pc1_packet_sha256",
                "source_archive_sha256",
            )
        )
    ):
        raise ReceiverError("CB1 typed source state differs")
    output = manifest["output"]
    if (
        not isinstance(output, dict)
        or set(output) != {"bytes", "sha256"}
        or output["bytes"] != 600 * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS
        or not _is_lower_sha256(output["sha256"])
    ):
        raise ReceiverError("CB1 output identity is malformed")
    return manifest


def _reconstruct_e5a_la1_bundle(
    bundle: bytes,
) -> tuple[bytes, tuple[dict[str, Any], ...]]:
    """Decode LA1 frames and rebuild every generic WS1 wrapper canonically."""

    from tac.optimization.ddm_la1_layer_assignment_context_pricing import (
        decode_la1_frame,
    )
    from tac.optimization.ddm_ws1_warm_start import compile_ws1_warm_start_archive
    from tac.optimization.direct_description_carrier_compose import (
        _decode_lane_programs,
        _decode_realization_profile,
        compile_carrier_compose_archive,
        decode_scorer_solved_template_bank,
    )
    from tac.optimization.direct_description_coupled_margin import (
        compile_coupled_margin_archive,
        decode_coupled_margin_program,
    )
    from tac.optimization.direct_description_preuint8_channel import (
        compile_preuint8_q8_archive,
        decode_preuint8_q8_program,
    )

    value = bytes(bundle)
    if len(value) < E5A_BUNDLE_PREFIX.size:
        raise ReceiverError("E5A LA1 bundle is truncated")
    magic, version, component_count, reserved = E5A_BUNDLE_PREFIX.unpack_from(value)
    if magic != E5A_BUNDLE_MAGIC or version != 1 or reserved != 0:
        raise ReceiverError("E5A LA1 bundle prefix differs")
    cursor = E5A_BUNDLE_PREFIX.size
    components: dict[str, bytes] = {}
    consumed: list[dict[str, Any]] = []
    previous_component = -1
    for _ in range(component_count):
        if cursor + E5A_BUNDLE_ENTRY.size > len(value):
            raise ReceiverError("E5A LA1 bundle entry is truncated")
        component_id, codec_id, frame_bytes = E5A_BUNDLE_ENTRY.unpack_from(value, cursor)
        cursor += E5A_BUNDLE_ENTRY.size
        if (
            component_id <= previous_component
            or component_id >= len(E5A_COMPONENTS)
            or codec_id >= len(E5A_CODECS)
            or frame_bytes <= 0
            or cursor + frame_bytes > len(value)
        ):
            raise ReceiverError("E5A LA1 bundle entry header differs")
        frame = value[cursor : cursor + frame_bytes]
        cursor += frame_bytes
        name = E5A_COMPONENTS[component_id]
        codec = E5A_CODECS[codec_id]
        try:
            raw = decode_la1_frame(frame, codec)
        except ValueError as exc:
            raise ReceiverError(f"E5A LA1 frame decode failed: {name}") from exc
        components[name] = raw
        consumed.append(
            {
                "bytes": len(frame),
                "component": name,
                "codec": codec,
                "frame_sha256": _sha256(frame),
                "raw_bytes": len(raw),
                "raw_sha256": _sha256(raw),
            }
        )
        previous_component = component_id
    if cursor != len(value):
        raise ReceiverError("E5A LA1 bundle has trailing bytes")
    required = set(E5A_COMPONENTS) - {"lane_programs_payload"}
    observed = set(components)
    if observed != required and observed != set(E5A_COMPONENTS):
        raise ReceiverError("E5A LA1 semantic component set differs")

    profile = _decode_realization_profile(components["realization_profile_payload"])
    templates = decode_scorer_solved_template_bank(
        components["scorer_solved_templates_payload"]
    )
    lanes = _decode_lane_programs(components.get("lane_programs_payload", b""))
    carrier, _ = compile_carrier_compose_archive(
        components["predictor_archive"],
        worldsheet_g1_payload=components["worldsheet_g1_payload"],
        lane_programs=lanes,
        realization_profile=profile,
        scorer_solved_templates=templates,
    )
    coupled = compile_coupled_margin_archive(
        carrier,
        decode_coupled_margin_program(components["coupled_margin_program"]),
        verify_base_member_effects=False,
    )
    preuint8 = compile_preuint8_q8_archive(
        coupled,
        decode_preuint8_q8_program(components["preuint8_q8_program"]),
        verify_base_member_effects=False,
    )
    archive = compile_ws1_warm_start_archive(
        preuint8,
        candidate="W_joint",
        payload=components["warm_start_payload"],
    )
    return archive, tuple(consumed)


def _reconstruct_ws1_state(
    manifest: Mapping[str, Any],
    members: dict[str, bytes],
) -> tuple[bytes, tuple[dict[str, Any], ...]]:
    """Decode and reassemble every admitted grammar stream byte-for-byte."""

    if manifest["schema"] == E5A_SCHEMA:
        archive, la1_consumed = _reconstruct_e5a_la1_bundle(
            members["state/ws1.ddj5"]
        )
    else:
        archive, dimensions = _parse_blob(
            members["state/ws1.ddj5"],
            expected_kind=0,
            label="state/ws1.ddj5",
        )
        if dimensions:
            raise ReceiverError("WS1 state blob must be an opaque rank-zero container")
        la1_consumed = ()
    admission = manifest["grammar_admission"]
    expected_keys = {
        "archive_bytes",
        "archive_sha256",
        "grammar_version",
        "receiver_consumption_complete",
        "schema",
        "streams",
    }
    if (
        not isinstance(admission, dict)
        or set(admission) != expected_keys
        or admission["schema"] != RECEIVER_GRAMMAR_ADMISSION_SCHEMA
        or admission["grammar_version"] != WS1_GRAMMAR_VERSION
        or admission["receiver_consumption_complete"] is not True
        or admission["archive_bytes"] != len(archive)
        or admission["archive_sha256"] != _sha256(archive)
    ):
        raise ReceiverError("WS1 grammar admission header differs")
    streams = admission["streams"]
    if not isinstance(streams, list) or len(streams) != 2 or any(not isinstance(row, dict) for row in streams):
        raise ReceiverError("WS1 grammar stream manifest differs")
    expected_names = ["nested_preuint8_archive", "warm_start_payload"]
    candidate = str(manifest["state"]["name"]).split(":", 1)[0]
    if candidate not in {"W_seg", "W_joint"}:
        raise ReceiverError("WS1 state name lacks a typed candidate prefix")
    expected_consumers = {
        "nested_preuint8_archive": "receive_preuint8_q8_archive",
        "warm_start_payload": (
            "apply_temporal_affine+reassert_frame1"
            if candidate == "W_seg"
            else "signed_distance_geometry+apply_local_statistics"
        ),
    }
    cursor = 0
    consumed: list[dict[str, Any]] = []
    for index, row in enumerate(streams):
        if (
            set(row)
            != {
                "bytes",
                "name",
                "offset",
                "receiver_consumer",
                "sha256",
            }
            or row["name"] != expected_names[index]
            or row["offset"] != cursor
            or isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or row["bytes"] <= 0
            or row["receiver_consumer"] != expected_consumers[row["name"]]
            or not _is_lower_sha256(row["sha256"])
        ):
            raise ReceiverError("WS1 grammar stream contract differs")
        stop = cursor + row["bytes"]
        if stop > len(archive) or _sha256(archive[cursor:stop]) != row["sha256"]:
            raise ReceiverError(f"WS1 grammar stream custody differs: {row['name']}")
        consumed.append(dict(row))
        cursor = stop
    if cursor != len(archive):
        raise ReceiverError("WS1 grammar streams do not cover every archive byte")
    return archive, (*la1_consumed, *consumed)


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


def _install_ws1_source_bundle(checkpoint_root: Path) -> dict[str, Any]:
    """Install the exporter-sealed generic receiver code without network I/O."""

    if not WS1_SOURCE_BUNDLE_B85:
        raise ReceiverError("WS1 runtime source bundle is absent")
    try:
        payload = base64.b85decode(WS1_SOURCE_BUNDLE_B85)
    except ValueError as exc:
        raise ReceiverError("WS1 runtime source bundle is malformed") from exc
    bundle_sha256 = _sha256(payload)
    bundle_path = checkpoint_root / f"ws1_receiver_sources_{bundle_sha256}.zip"
    if bundle_path.exists():
        if _sha256_file(bundle_path) != (len(payload), bundle_sha256):
            raise ReceiverError("preserved WS1 runtime source bundle differs")
    else:
        temporary = bundle_path.with_name(bundle_path.name + f".partial.{os.getpid()}")
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, bundle_path)
    if str(bundle_path) not in sys.path:
        sys.path.insert(0, str(bundle_path))
    return {
        "bytes": len(payload),
        "path": str(bundle_path),
        "sha256": bundle_sha256,
    }


def _inflate_cb1(
    *,
    archive_dir: Path,
    output_dir: Path,
    final_path: Path,
    relative: PurePosixPath,
    started: float,
) -> dict[str, Any]:
    """Inflate one RG4 source-local archive through the E4 framed runtime."""

    members = _read_exact_cb1_members(archive_dir)
    manifest_payload = members["manifest.json"]
    manifest_sha256 = _sha256(manifest_payload)
    manifest = _validate_cb1_manifest(
        _duplicate_refusing_json(manifest_payload, label="manifest.json"),
        members,
    )
    source_archive, dimensions = _parse_blob(
        members["state/rg4.ddr4"],
        expected_kind=0,
        label="state/rg4.ddr4",
    )
    if dimensions:
        raise ReceiverError("CB1 state blob must be an opaque rank-zero container")
    state = manifest["state"]
    if (
        len(source_archive) != state["source_archive_bytes"]
        or _sha256(source_archive) != state["source_archive_sha256"]
    ):
        raise ReceiverError("CB1 reconstructed source custody differs")
    checkpoint_root = (
        output_dir.parent
        / ".ddm_runtime_checkpoints"
        / state["source_archive_sha256"]
        / relative.with_suffix("").as_posix().replace("/", "__")
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(manifest["output"]["bytes"])
    required_free_bytes = expected_bytes * 2 + SAFETY_BYTES
    free_bytes = shutil.disk_usage(checkpoint_root).free
    if free_bytes < required_free_bytes:
        raise ReceiverError(f"storage preflight failed: {free_bytes} < {required_free_bytes}")
    source_bundle = _install_ws1_source_bundle(checkpoint_root)
    try:
        import numpy as np

        from tac.optimization.ddm_pc1_pose_stream import serialize_pc1_packet
        from tac.optimization.ddm_rg4_g3_blocks_and_active_tube import (
            parse_source_local_composition_archive,
            receive_source_local_pc1_camera_pairs,
        )
        from tac.optimization.ddm_ws1_warm_start import (
            parse_ws1_warm_start_archive,
            receive_ws1_warm_start_archive,
        )
    except ImportError as exc:
        raise ReceiverError("bundled CB1 receiver import failed") from exc
    parent_archive, packet, source_manifest = parse_source_local_composition_archive(source_archive)
    parsed_parent = parse_ws1_warm_start_archive(parent_archive)
    if parsed_parent.exact_reemit() != parent_archive:
        raise ReceiverError("CB1 parent parse/re-emit changed source bytes")
    if (
        source_manifest["schema"] != state["source_schema"]
        or _sha256(parent_archive) != state["parent_archive_sha256"]
        or _sha256(serialize_pc1_packet(packet)) != state["pc1_packet_sha256"]
    ):
        raise ReceiverError("CB1 nested source identities differ")
    parent_receiver = receive_ws1_warm_start_archive(parent_archive)
    try:
        movable_layer = next(layer for layer in parent_receiver.layers if layer.role == "Movable")
    except StopIteration as exc:
        raise ReceiverError("CB1 parent lacks a Movable layer") from exc

    batch_pairs = int(state["batch_pairs"])
    stage_rows: list[dict[str, Any]] = []
    stage_paths: list[Path] = []
    render_seconds = 0.0
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
        if preserved is None:
            stage_started = time.monotonic()
            pair_ids = tuple(range(start, stop))
            parent_camera = parent_receiver.render_camera_pairs(pair_ids)
            movable_masks = np.stack(
                [
                    parent_receiver._mask_for_layer(
                        movable_layer,
                        pair_id,
                        replace_g1_movable=True,
                    )
                    for pair_id in pair_ids
                ],
                axis=0,
            ).astype(np.bool_)
            camera = receive_source_local_pc1_camera_pairs(
                parent_camera=parent_camera,
                packet=packet,
                pair_ids=pair_ids,
                movable_masks=movable_masks,
            )
            rendered = torch.from_numpy(camera)
            if (
                rendered.dtype != torch.uint8
                or tuple(rendered.shape)
                != (
                    stop - start,
                    FRAMES_PER_PAIR,
                    CAMERA_H,
                    CAMERA_W,
                    CHANNELS,
                )
                or not rendered.is_contiguous()
            ):
                raise ReceiverError("CB1 receiver rendered noncanonical camera bytes")
            render_seconds += time.monotonic() - stage_started
            row = _write_or_adopt_rendered_stage(
                stage_path=stage_path,
                state_path=state_path,
                rendered=rendered,
                manifest_sha256=manifest_sha256,
                start=start,
                stop=stop,
            )
        else:
            row = preserved
        stage_rows.append(row)
        stage_paths.append(stage_path)
    final_bytes, final_sha256 = _assemble_final(
        final_path=final_path,
        stage_rows=stage_rows,
        stage_paths=stage_paths,
        expected_bytes=expected_bytes,
        expected_sha256=str(manifest["output"]["sha256"]),
    )
    receipt = {
        "coder": (
            "brotli_q11"
            if BLOB_HEADER.unpack_from(members["state/rg4.ddr4"])[2] == 1
            else "lzma1_raw_d1m_lc3_lp0_pb2"
        ),
        "dependencies": manifest["dependencies"],
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "final": {
            "bytes": final_bytes,
            "path": str(final_path),
            "sha256": final_sha256,
        },
        "grammar_admission": {
            "parent_parse_reemit_byte_identical": True,
            "source_archive_bytes": len(source_archive),
            "source_archive_sha256": _sha256(source_archive),
            "source_local_parse_reemit_byte_identical": True,
        },
        "member_consumption": [
            {
                "bytes": len(members[name]),
                "member": name,
                "sha256": _sha256(members[name]),
            }
            for name in EXPECTED_CB1_MEMBERS
        ],
        "pair_count": 600,
        "render_seconds": format(render_seconds, ".6f"),
        "research_only": True,
        "resume": {
            "all_stage_checkpoints_preserved": True,
            "checkpoint_root": str(checkpoint_root),
            "stage_count": len(stage_rows),
        },
        "runtime_source_bundle": source_bundle,
        "schema": "ddm_cb1_rg4_perclass_runtime_inflate_receipt.v1",
        "score_claim": False,
        "fixed_torch_threads": 4,
        "total_seconds": format(time.monotonic() - started, ".6f"),
    }
    _atomic_json(checkpoint_root / "inflate_receipt.json", receipt)
    return receipt


def _inflate_ws1(
    *,
    archive_dir: Path,
    output_dir: Path,
    final_path: Path,
    relative: PurePosixPath,
    started: float,
) -> dict[str, Any]:
    members = _read_exact_ws1_members(archive_dir)
    manifest_payload = members["manifest.json"]
    manifest_sha256 = _sha256(manifest_payload)
    manifest = _validate_ws1_manifest(
        _duplicate_refusing_json(manifest_payload, label="manifest.json"),
        members,
    )
    source_bundle: dict[str, Any] | None = None
    if manifest["schema"] == E5A_SCHEMA:
        bootstrap_root = (
            output_dir.parent
            / ".ddm_runtime_checkpoints"
            / f"e5a_manifest_{manifest_sha256}"
        )
        bootstrap_root.mkdir(parents=True, exist_ok=True)
        source_bundle = _install_ws1_source_bundle(bootstrap_root)
    state_archive, consumed_streams = _reconstruct_ws1_state(manifest, members)
    state_sha256 = _sha256(state_archive)
    checkpoint_root = (
        output_dir.parent
        / ".ddm_runtime_checkpoints"
        / state_sha256
        / relative.with_suffix("").as_posix().replace("/", "__")
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(manifest["output"]["bytes"])
    required_free_bytes = (
        expected_bytes * (3 if manifest["schema"] in {IC1_SCHEMA, IC2_SCHEMA} else 2)
        + SAFETY_BYTES
    )
    free_bytes = shutil.disk_usage(checkpoint_root).free
    if free_bytes < required_free_bytes:
        raise ReceiverError(f"storage preflight failed: {free_bytes} < {required_free_bytes}")
    if source_bundle is None:
        source_bundle = _install_ws1_source_bundle(checkpoint_root)
    try:
        from tac.optimization.ddm_ws1_warm_start import (
            parse_ws1_warm_start_archive,
            receive_ws1_warm_start_archive,
        )
    except ImportError as exc:
        raise ReceiverError("bundled WS1 receiver import failed") from exc
    parsed = parse_ws1_warm_start_archive(state_archive)
    if parsed.exact_reemit() != state_archive:
        raise ReceiverError("WS1 grammar parse/re-emit changed source bytes")
    candidate = str(manifest["state"]["name"]).split(":", 1)[0]
    if parsed.candidate != candidate:
        raise ReceiverError("WS1 parsed candidate differs from typed state name")
    source_receiver = receive_ws1_warm_start_archive(state_archive)

    batch_pairs = int(manifest["state"]["batch_pairs"])
    amplitude_enabled = manifest["schema"] in {IC1_SCHEMA, IC2_SCHEMA}
    stage_rows: list[dict[str, Any]] = []
    stage_paths: list[Path] = []
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
            manifest_sha256=state_sha256,
            start=start,
            stop=stop,
            expected_bytes=expected_stage_bytes,
        )
        if preserved is None:
            stage_started = time.monotonic()
            camera = source_receiver.render_camera_pairs(tuple(range(start, stop)))
            rendered = torch.from_numpy(camera)
            if (
                rendered.dtype != torch.uint8
                or tuple(rendered.shape)
                != (
                    stop - start,
                    FRAMES_PER_PAIR,
                    CAMERA_H,
                    CAMERA_W,
                    CHANNELS,
                )
                or not rendered.is_contiguous()
            ):
                raise ReceiverError("WS1 receiver rendered noncanonical camera bytes")
            render_seconds += time.monotonic() - stage_started
            row = _write_or_adopt_rendered_stage(
                stage_path=stage_path,
                state_path=state_path,
                rendered=rendered,
                manifest_sha256=state_sha256,
                start=start,
                stop=stop,
            )
        else:
            row = preserved
        stage_rows.append(row)
        stage_paths.append(stage_path)
        if amplitude_enabled:
            moment_rows.append(
                _load_or_measure_pose_moments(
                    stage_path=stage_path,
                    state_path=(checkpoint_root / f"base_pairs_{start:04d}_{stop:04d}.moments.json"),
                    stage_row=row,
                    manifest_sha256=manifest_sha256,
                    start=start,
                    stop=stop,
                )
            )
    amplitude_receipt: dict[str, Any] | None = None
    if amplitude_enabled:
        gain, bias, amplitude_receipt = _load_or_write_pa1_affine(
            path=checkpoint_root / "pa1_affine.json",
            manifest_sha256=manifest_sha256,
            moments=_merge_pose_moments(moment_rows),
        )
        base_paths = stage_paths
        stage_rows = []
        stage_paths = []
        for index, start in enumerate(range(0, 600, batch_pairs)):
            stop = min(start + batch_pairs, 600)
            stage_path = checkpoint_root / f"pairs_{start:04d}_{stop:04d}.raw"
            state_path = checkpoint_root / f"pairs_{start:04d}_{stop:04d}.json"
            expected_stage_bytes = (stop - start) * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS
            row = _load_preserved_stage(
                stage_path=stage_path,
                state_path=state_path,
                manifest_sha256=manifest_sha256,
                start=start,
                stop=stop,
                expected_bytes=expected_stage_bytes,
            )
            if row is None:
                stage_started = time.monotonic()
                corrected = _apply_pa1_frame0_affine(
                    _load_stage_tensor(
                        stage_path=base_paths[index],
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
                    rendered=corrected,
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
        expected_sha256=str(manifest["output"]["sha256"]),
    )
    ws1_coder, la1_component_coders = _ws1_receipt_coder(
        manifest=manifest,
        framed=members["state/ws1.ddj5"],
        consumed_streams=consumed_streams,
    )
    receipt = {
        "coder": ws1_coder,
        "final_output_binding": "COUNTED_MANIFEST_SHA256",
        "la1_component_coders": la1_component_coders,
        "dependencies": manifest["dependencies"],
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "final": {
            "bytes": final_bytes,
            "path": str(final_path),
            "sha256": final_sha256,
        },
        "grammar_admission": {
            "archive_bytes": len(state_archive),
            "archive_sha256": state_sha256,
            "grammar_version": WS1_GRAMMAR_VERSION,
            "parse_reemit_byte_identical": True,
            "streams_consumed": list(consumed_streams),
        },
        "member_consumption": [
            {
                "bytes": len(members[name]),
                "member": name,
                "sha256": _sha256(members[name]),
            }
            for name in EXPECTED_WS1_MEMBERS
        ],
        "pair_count": 600,
        "render_seconds": format(render_seconds, ".6f"),
        "research_only": True,
        "resume": {
            "all_stage_checkpoints_preserved": True,
            "checkpoint_root": str(checkpoint_root),
            "stage_count": len(stage_rows),
        },
        "runtime_source_bundle": source_bundle,
        "schema": "ddm_e4_ws1_runtime_inflate_receipt.v1",
        "score_claim": False,
        "fixed_torch_threads": 4,
        "staleness": {
            "consumption_time_input_hashes_verified": True,
            "policy": "fail_closed_on_grammar_stream_hash_or_version_mismatch",
        },
        "total_seconds": format(time.monotonic() - started, ".6f"),
    }
    if amplitude_enabled:
        receipt["amplitude_transform"] = amplitude_receipt
        receipt["resume"]["base_stage_count"] = len(moment_rows)
    _atomic_json(checkpoint_root / "inflate_receipt.json", receipt)
    return receipt


def _ws1_receipt_coder(
    *,
    manifest: Mapping[str, Any],
    framed: bytes,
    consumed_streams: Sequence[Mapping[str, Any]],
) -> tuple[str, list[dict[str, Any]] | None]:
    """Report the coder that actually framed this WS1 member.

    An E5A packet's `state/ws1.ddj5` is an LA1 bundle framed by
    `E5A_BUNDLE_PREFIX`, not a `DDE1B` blob: byte 6 there is `component_count`,
    so reading it through `BLOB_HEADER[2]` produced `brotli_q11` iff the bundle
    happened to carry exactly one component and `lzma1_raw_d1m_lc3_lp0_pb2`
    otherwise.  Both were meaningless — E5A's real coders are the per-component
    `E5A_CODECS`, already censused by `_reconstruct_e5a_la1_bundle`.
    """

    if manifest["schema"] == E5A_SCHEMA:
        census = [
            {
                "bytes": int(row["bytes"]),
                "codec": str(row["codec"]),
                "component": str(row["component"]),
                "frame_sha256": str(row["frame_sha256"]),
            }
            for row in consumed_streams
            if "codec" in row and "component" in row
        ]
        if not census:
            raise ReceiverError("E5A LA1 bundle produced no per-component codec census")
        return "e5a_la1_per_component", census
    # Only read the DDE1B layout when the member really is a DDE1B frame.
    if len(framed) < BLOB_HEADER.size or framed[: len(BLOB_MAGIC)] != BLOB_MAGIC:
        raise ReceiverError("WS1 state member is not a DDE1B frame")
    codec = BLOB_HEADER.unpack_from(framed)[2]
    if codec == 1:
        return "brotli_q11", None
    if codec == 2:
        return "lzma1_raw_d1m_lc3_lp0_pb2", None
    raise ReceiverError("WS1 packet coder contract changed")


def _preserved_stage_sequence_identity(
    rows: list[dict[str, Any]],
    paths: list[Path],
) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    for row, path in zip(rows, paths, strict=True):
        if _sha256_file(path) != (row["bytes"], row["sha256"]):
            raise ReceiverError("CC3 preserved stage changed before final assembly")
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
                total += len(chunk)
    return total, digest.hexdigest()


def _inflate_cc3(
    *,
    archive_dir: Path,
    output_dir: Path,
    final_path: Path,
    relative: PurePosixPath,
    started: float,
) -> dict[str, Any]:
    """Restore eight counted DCC3 leaves, then run the real WS1+PC1 receiver."""

    members = _read_exact_cc3_members(archive_dir)
    member_binding = _sha256(
        b"".join(name.encode("ascii") + b"\0" + hashlib.sha256(members[name]).digest() for name in EXPECTED_CC3_MEMBERS)
    )
    checkpoint_root = (
        output_dir.parent
        / ".ddm_runtime_checkpoints"
        / f"cc3_{member_binding}"
        / relative.with_suffix("").as_posix().replace("/", "__")
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    source_bundle = _install_ws1_source_bundle(checkpoint_root)
    try:
        import numpy as np

        from tac.optimization.ddm_cc3_mixed_coder_receiver import restore_extracted_composition
        from tac.optimization.ddm_pc1_pose_stream import (
            parse_counted_composition_archive,
            receive_pc1_camera_pairs,
        )
        from tac.optimization.ddm_ws1_warm_start import (
            parse_ws1_warm_start_archive,
            receive_ws1_warm_start_archive,
        )
    except ImportError as exc:
        raise ReceiverError("bundled CC3/WS1 receiver import failed") from exc
    try:
        source_archive, parent_archive, packet, restoration = restore_extracted_composition(members)
    except ValueError as exc:
        raise ReceiverError("CC3 exact recursive restoration failed") from exc
    if (
        restoration["decoded_frame_count"] != 8
        or restoration["physical_leaf_count"] != 27
        or restoration["codec_counts"]
        != {
            "G4_FREE_DECODER_CONTEXT": 1,
            "BELLARD_CLASS_MIXING": 7,
        }
    ):
        raise ReceiverError("CC3 selected frame census differs")
    parsed = parse_ws1_warm_start_archive(parent_archive)
    if parsed.exact_reemit() != parent_archive or parsed.candidate != "W_joint":
        raise ReceiverError("CC3 parent is not one exact W_joint receiver state")
    source_receiver = receive_ws1_warm_start_archive(parent_archive)
    try:
        movable_layer = next(layer for layer in source_receiver.layers if layer.role == "Movable")
    except StopIteration as exc:
        raise ReceiverError("CC3 W_joint receiver lacks its Movable layer") from exc

    expected_bytes = 600 * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS
    required_free_bytes = expected_bytes * 2 + SAFETY_BYTES
    free_bytes = shutil.disk_usage(checkpoint_root).free
    if free_bytes < required_free_bytes:
        raise ReceiverError(f"storage preflight failed: {free_bytes} < {required_free_bytes}")
    source_sha256 = _sha256(source_archive)
    stage_rows: list[dict[str, Any]] = []
    stage_paths: list[Path] = []
    render_seconds = 0.0
    batch_pairs = 32
    for start in range(0, 600, batch_pairs):
        stop = min(start + batch_pairs, 600)
        stage_path = checkpoint_root / f"pairs_{start:04d}_{stop:04d}.raw"
        state_path = checkpoint_root / f"pairs_{start:04d}_{stop:04d}.json"
        expected_stage_bytes = (stop - start) * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS
        row = _load_preserved_stage(
            stage_path=stage_path,
            state_path=state_path,
            manifest_sha256=source_sha256,
            start=start,
            stop=stop,
            expected_bytes=expected_stage_bytes,
        )
        if row is None:
            stage_started = time.monotonic()
            pair_ids = tuple(range(start, stop))
            parent_camera = source_receiver.render_camera_pairs(pair_ids)
            movable_masks = np.stack(
                [
                    source_receiver._mask_for_layer(
                        movable_layer,
                        pair_id,
                        replace_g1_movable=True,
                    )
                    for pair_id in pair_ids
                ],
                axis=0,
            ).astype(np.bool_)
            camera = receive_pc1_camera_pairs(
                parent_camera=parent_camera,
                packet=packet,
                pair_ids=pair_ids,
                movable_masks=movable_masks,
                torch_module=torch,
            )
            rendered = torch.from_numpy(camera)
            if (
                rendered.dtype != torch.uint8
                or tuple(rendered.shape)
                != (
                    stop - start,
                    FRAMES_PER_PAIR,
                    CAMERA_H,
                    CAMERA_W,
                    CHANNELS,
                )
                or not rendered.is_contiguous()
            ):
                raise ReceiverError("CC3 composition receiver rendered noncanonical camera bytes")
            render_seconds += time.monotonic() - stage_started
            row = _write_or_adopt_rendered_stage(
                stage_path=stage_path,
                state_path=state_path,
                rendered=rendered,
                manifest_sha256=source_sha256,
                start=start,
                stop=stop,
            )
        stage_rows.append(row)
        stage_paths.append(stage_path)
    stage_bytes, stage_sha256 = _preserved_stage_sequence_identity(stage_rows, stage_paths)
    if stage_bytes != expected_bytes:
        raise ReceiverError("CC3 stage byte total differs from output geometry")
    # The stage digest concatenates the SAME files `_assemble_final` concatenates,
    # so passing it back as `expected_sha256` can only ever compare a value to
    # itself on a fresh assembly: the "final raw identity mismatch" guard was dead
    # code there, and the receipt still claimed verified custody.  Every other
    # inflate route binds its final bytes to the COUNTED `manifest["output"]["sha256"]`.
    # CC3's counted PC1 composition manifest has no output digest today, so bind to
    # it when it appears and otherwise say plainly that the binding is absent.
    counted_output_sha256 = parse_counted_composition_archive(source_archive)[2].get("output_sha256")
    if counted_output_sha256 is not None and not _is_lower_sha256(str(counted_output_sha256)):
        raise ReceiverError("CC3 counted output SHA-256 is malformed")
    final_output_binding = (
        "COUNTED_COMPOSITION_MANIFEST_SHA256"
        if counted_output_sha256 is not None
        else "DERIVED_FROM_PRESERVED_STAGES_NO_COUNTED_BINDING"
    )
    final_bytes, final_sha256 = _assemble_final(
        final_path=final_path,
        stage_rows=stage_rows,
        stage_paths=stage_paths,
        expected_bytes=expected_bytes,
        expected_sha256=str(counted_output_sha256) if counted_output_sha256 is not None else stage_sha256,
    )
    receipt = {
        "final_output_binding": final_output_binding,
        "composition": {
            "active": packet.active,
            "parent_archive_bytes": len(parent_archive),
            "parent_archive_sha256": _sha256(parent_archive),
            "source_archive_bytes": len(source_archive),
            "source_archive_sha256": source_sha256,
        },
        "dependencies": ["numpy", "scipy", "torch", "brotli"],
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "final": {
            "bytes": final_bytes,
            "path": str(final_path),
            "sha256": final_sha256,
        },
        "fixed_torch_threads": 4,
        "member_consumption": [
            {
                "bytes": len(members[name]),
                "member": name,
                "sha256": _sha256(members[name]),
            }
            for name in EXPECTED_CC3_MEMBERS
        ],
        "pair_count": 600,
        "render_seconds": format(render_seconds, ".6f"),
        "research_only": True,
        "restoration": restoration,
        "resume": {
            "all_stage_checkpoints_preserved": True,
            "checkpoint_root": str(checkpoint_root),
            "stage_count": len(stage_rows),
        },
        "runtime_source_bundle": source_bundle,
        "schema": "ddm_cc3_mixed_coder_runtime_inflate_receipt.v1",
        "score_claim": False,
        "staleness": {
            "consumption_time_input_hashes_verified": True,
            "policy": "fail_closed_on_frame_hash_final_byte_nested_member_or_source_manifest_mismatch",
        },
        "total_seconds": format(time.monotonic() - started, ".6f"),
    }
    _atomic_json(checkpoint_root / "inflate_receipt.json", receipt)
    return receipt


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

    if (archive_dir / "manifest/pc1.json").is_file():
        return _inflate_cc3(
            archive_dir=archive_dir,
            output_dir=output_dir,
            final_path=final_path,
            relative=relative,
            started=started,
        )

    if (archive_dir / "state/rg4.ddr4").is_file():
        return _inflate_cb1(
            archive_dir=archive_dir,
            output_dir=output_dir,
            final_path=final_path,
            relative=relative,
            started=started,
        )

    if (archive_dir / "state/ws1.ddj5").is_file():
        return _inflate_ws1(
            archive_dir=archive_dir,
            output_dir=output_dir,
            final_path=final_path,
            relative=relative,
            started=started,
        )

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
    amplitude_enabled = manifest["schema"] in {E3_SCHEMA, E4_SCHEMA}
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
        "dependencies": manifest["dependencies"],
        "coder": (
            "brotli_q11" if BLOB_HEADER.unpack_from(members["base/chart.ddb"])[2] == 1 else "lzma1_raw_d1m_lc3_lp0_pb2"
        ),
        "final_output_binding": "COUNTED_MANIFEST_SHA256",
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
            "ddm_e4_runtime_inflate_receipt.v1"
            if manifest["schema"] == E4_SCHEMA
            else "ddm_e3_runtime_inflate_receipt.v1"
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
