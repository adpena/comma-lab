# SPDX-License-Identifier: MIT
"""Receiver-valid SNeRV semantic section neutralization.

SNeRV sections cannot be literally removed without changing the receiver
grammar: the packet requires all four sections. This module therefore performs
real semantic neutralization for optional sections while preserving the input
wire format and a valid receiver packet.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from tac.analysis.snerv_step_map_coder import encode_step_maps
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    SECTION_ORDER,
    SNERV_ARCHIVE_SCHEMA,
    SNERV_ARCHIVE_SCHEMA_V2,
    decode_snerv_archive_frames_from_decoded,
    encode_decoder_payload,
    pack_snerv_archive,
    pack_snerv_archive_snar2,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import HfGenerationDecoder

SNERV_SECTION_VALUE_SCHEMA = "snerv_receiver_section_value_neutralization.v2"
SNERV_SNAR1_SECTION_VALUE_SCHEMA = SNERV_SECTION_VALUE_SCHEMA
NEUTRALIZABLE_SNERV_SECTIONS = ("decoder_payload", "step_map_packet")
RECEIVER_DECODE_ONLY_STATUS = "receiver_decode_only"


class SnervSectionValueError(ValueError):
    """Raised when a SNeRV receiver section cannot be neutralized honestly."""


def neutralize_snerv_section(
    packet: bytes,
    section: str,
    *,
    step_map_bins: int = 16,
    verify_receiver_decode: bool = True,
) -> dict[str, Any]:
    """Return a valid receiver packet with one optional section neutralized.

    Supported sections:
    - ``decoder_payload``: replace the HF generator with a zero decoder using
      the decoded model-size controls.
    - ``step_map_packet``: replace per-coefficient maps with per-map constants.

    ``metadata_payload`` and ``lf_payload`` are not optional for the current
    receiver grammar and fail closed. SNAR2 inputs remain SNAR2 so byte-pressure
    profiles do not accidentally price a debug SNAR1 rebuild.
    """

    requested = str(section)
    if requested not in SECTION_ORDER:
        raise SnervSectionValueError(f"unknown SNeRV section: {requested!r}")
    if requested not in NEUTRALIZABLE_SNERV_SECTIONS:
        raise SnervSectionValueError(
            f"SNeRV section {requested!r} is not neutralizable without a new "
            "receiver grammar"
        )
    decoded = unpack_snerv_archive(bytes(packet))
    wire_format = _wire_format_for_schema(decoded.schema)
    sections = dict(decoded.sections)
    baseline_section = sections[requested]
    baseline_packet_sha = _sha256(bytes(packet))
    if requested == "decoder_payload":
        old_decoder = decoded.decode_decoder()
        replacement = encode_decoder_payload(
            HfGenerationDecoder.zeros(
                levels=int(old_decoder.levels),
                model_size=old_decoder.model_size,
            )
        )
        method = "zero_hf_generation_decoder"
    else:
        step_maps = decoded.decode_step_maps()
        if not step_maps:
            raise SnervSectionValueError("step_map_packet decoded to zero maps")
        neutral_maps = [
            np.full_like(
                np.asarray(step_map, dtype=np.float32),
                fill_value=float(np.median(np.asarray(step_map, dtype=np.float64))),
                dtype=np.float32,
            )
            for step_map in step_maps
        ]
        replacement = encode_step_maps(neutral_maps, bins=int(step_map_bins)).packet
        method = "per_map_constant_step_map"
    sections[requested] = replacement
    packer = pack_snerv_archive_snar2 if wire_format == "snar2" else pack_snerv_archive
    rebuilt = packer(
        metadata_payload=sections["metadata_payload"],
        lf_payload=sections["lf_payload"],
        decoder_payload=sections["decoder_payload"],
        step_map_packet=sections["step_map_packet"],
        metadata=dict(decoded.metadata),
    )
    replay_status = "not_requested"
    replay_frame_shape = None
    if verify_receiver_decode:
        try:
            replay = decode_snerv_archive_frames_from_decoded(
                unpack_snerv_archive(rebuilt.packet)
            )
        except Exception as exc:  # pragma: no cover - exercised by fail path callers.
            raise SnervSectionValueError(
                f"neutralized SNeRV receiver decode failed: {exc}"
            ) from exc
        replay_status = "receiver_decode_succeeded"
        replay_frame_shape = [int(v) for v in replay.shape]
    return {
        "schema": SNERV_SNAR1_SECTION_VALUE_SCHEMA,
        "packet_wire_format": wire_format,
        "baseline_packet_schema": decoded.schema,
        "neutralized_packet_schema": rebuilt.schema,
        "section": requested,
        "neutralization_method": method,
        "baseline_packet_bytes": len(packet),
        "neutralized_packet_bytes": len(rebuilt.packet),
        "baseline_packet_sha256": baseline_packet_sha,
        "neutralized_packet_sha256": _sha256(rebuilt.packet),
        "baseline_section_bytes": len(baseline_section),
        "neutralized_section_bytes": len(replacement),
        "baseline_section_sha256": _sha256(baseline_section),
        "neutralized_section_sha256": _sha256(replacement),
        "section_changed": baseline_section != replacement,
        "receiver_decode_status": replay_status,
        "receiver_frame_shape": replay_frame_shape,
        "packet": rebuilt.packet,
        "section_value_row": _section_value_row(
            section=requested,
            baseline_bytes=len(baseline_section),
            packet_sha256=baseline_packet_sha,
            method=method,
            receiver_status=replay_status,
            packet_wire_format=wire_format,
        ),
        **FALSE_AUTHORITY,
    }


def neutralize_snerv_sections(
    packet: bytes,
    sections: tuple[str, ...] = NEUTRALIZABLE_SNERV_SECTIONS,
    *,
    step_map_bins: int = 16,
    verify_receiver_decode: bool = True,
) -> dict[str, Any]:
    """Build neutralized receiver-packet variants for multiple optional sections."""

    variants = [
        neutralize_snerv_section(
            packet,
            section,
            step_map_bins=step_map_bins,
            verify_receiver_decode=verify_receiver_decode,
        )
        for section in sections
    ]
    return {
        "schema": SNERV_SNAR1_SECTION_VALUE_SCHEMA,
        "packet_wire_format": (
            variants[0].get("packet_wire_format") if variants else None
        ),
        "variant_count": len(variants),
        "neutralizable_sections": list(NEUTRALIZABLE_SNERV_SECTIONS),
        "variants": [
            {key: value for key, value in variant.items() if key != "packet"}
            for variant in variants
        ],
        "section_value_rows": [
            variant["section_value_row"]
            for variant in variants
            if isinstance(variant.get("section_value_row"), Mapping)
        ],
        **FALSE_AUTHORITY,
    }


def _section_value_row(
    *,
    section: str,
    baseline_bytes: int,
    packet_sha256: str,
    method: str,
    receiver_status: str,
    packet_wire_format: str,
) -> dict[str, Any]:
    receiver_decode_passed = receiver_status == "receiver_decode_succeeded"
    blockers = [
        "snerv_snar1_neutralization_false_authority_no_scorer_replay",
        "delta_nonrate_score_missing",
        "full_video_coverage_missing",
        "runtime_consumption_proof_not_executed_for_neutralized_packet",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if not receiver_decode_passed:
        blockers.append("neutralized_packet_receiver_decode_not_proven")
    return {
        "row_id": f"snerv_{packet_wire_format}_neutralize_{section}",
        "section_id": f"snerv_{section}",
        "family": "snerv",
        "scope": "snerv_receiver_semantic_neutralization",
        "row_kind": "existing_section_cut",
        "packet_wire_format": packet_wire_format,
        "neutralization_method": method,
        "byte_delta": -int(baseline_bytes),
        "section_bytes": int(baseline_bytes),
        "delta_nonrate_score": None,
        "axis_tag": "[planning/control:false-authority]",
        "receiver_decode_status": receiver_status,
        "receiver_decode_passed": receiver_decode_passed,
        "receiver_proof_status": RECEIVER_DECODE_ONLY_STATUS,
        "runtime_consumption_proof_passed": False,
        "full_video_coverage": False,
        "archive_sha256": packet_sha256,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def _wire_format_for_schema(schema: str) -> str:
    if schema == SNERV_ARCHIVE_SCHEMA:
        return "snar1"
    if schema == SNERV_ARCHIVE_SCHEMA_V2:
        return "snar2"
    raise SnervSectionValueError(f"unsupported SNeRV packet schema: {schema!r}")


__all__ = [
    "NEUTRALIZABLE_SNERV_SECTIONS",
    "RECEIVER_DECODE_ONLY_STATUS",
    "SNERV_SECTION_VALUE_SCHEMA",
    "SNERV_SNAR1_SECTION_VALUE_SCHEMA",
    "SnervSectionValueError",
    "neutralize_snerv_section",
    "neutralize_snerv_sections",
]
