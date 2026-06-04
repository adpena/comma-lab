# SPDX-License-Identifier: MIT
"""Prune receiver-inert metadata from SNAR1 packets.

This is a compatibility bridge, not the final compact bitstream grammar.  It
keeps the current SNAR1 unpacker contract while moving large provenance fields
out of archive bytes and into a durable manifest.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    HEADER_LEN_FMT,
    SECTION_ORDER,
    SNERV_ARCHIVE_MAGIC,
    SNERV_ARCHIVE_SCHEMA,
    SnervArchiveError,
    decode_snerv_archive_pair_frames_from_decoded,
    unpack_snerv_archive,
)

SCHEMA = "snerv_snar_header_minimization.v1"
AXIS_TAG = "[receiver-proof:false-authority]"
DEFAULT_FRAME_PROOF_MAX_OUTPUT_BYTES = 256 * 1024 * 1024

RECEIVER_METADATA_KEYS = (
    "n_pairs",
    "frames_per_pair",
    "channels",
    "lf_plane_count",
    "levels",
    "wavelet",
)


class SnervSnarHeaderMinimizerError(ValueError):
    """Raised when a SNAR1 packet cannot be minimized safely."""


def build_snerv_snar_header_minimization(
    source_packet: bytes,
    *,
    source_packet_path: str | None = None,
    candidate_id: str | None = None,
    proof_pair_indices: Sequence[int] = (),
    full_video_receiver_proof: bool = False,
    frame_proof_max_output_bytes: int = DEFAULT_FRAME_PROOF_MAX_OUTPUT_BYTES,
    hard_byte_ceilings: Sequence[int] = (),
    generated_utc: str | None = None,
    raw_argv: Sequence[str] = (),
) -> tuple[dict[str, Any], bytes]:
    """Return a minimized SNAR1 packet plus a provenance/proof manifest."""

    source_blob = bytes(source_packet)
    if not source_blob:
        raise SnervSnarHeaderMinimizerError("source_packet must be non-empty")

    source = unpack_snerv_archive(source_blob)
    minimal_metadata = _receiver_metadata(source.metadata)
    candidate_packet = _pack_minimal_snar1(
        sections=source.sections,
        metadata=minimal_metadata,
    )
    candidate = unpack_snerv_archive(candidate_packet)
    removed_metadata = {
        str(key): value
        for key, value in source.metadata.items()
        if key not in minimal_metadata
    }
    section_rows = [
        {
            "name": name,
            "source_bytes": len(source.sections[name]),
            "candidate_bytes": len(candidate.sections[name]),
            "source_sha256": _sha256(source.sections[name]),
            "candidate_sha256": _sha256(candidate.sections[name]),
            "bytes_exact_equal": source.sections[name] == candidate.sections[name],
            "sha256_exact_equal": _sha256(source.sections[name])
            == _sha256(candidate.sections[name]),
        }
        for name in SECTION_ORDER
    ]
    proof = _receiver_pair_frame_proof(
        source,
        candidate,
        requested_pair_indices=proof_pair_indices,
        full_video=bool(full_video_receiver_proof),
        max_output_bytes=int(frame_proof_max_output_bytes),
    )
    section_parity = all(row["bytes_exact_equal"] for row in section_rows)
    receiver_contract = bool(section_parity and proof["status"] == "proven_exact")
    full_video_receiver_contract = bool(
        receiver_contract and proof.get("scope") == "full_video_streaming"
    )
    bound_candidate_id = _nonempty_text(candidate_id)
    source_header_bytes = _outer_header_bytes(source_blob)
    candidate_header_bytes = _outer_header_bytes(candidate_packet)
    report = {
        "schema": SCHEMA,
        "axis_tag": AXIS_TAG,
        "generated_utc": generated_utc or datetime.now(UTC).isoformat(),
        "operation": "snar1_receiver_metadata_prune",
        "candidate_binding": {
            "candidate_id": bound_candidate_id,
            "binding_status": (
                "candidate_id_and_source_packet_sha256"
                if bound_candidate_id
                else "source_packet_sha256_only_false_authority"
            ),
            "source_packet_sha256": _sha256(source_blob),
            "candidate_packet_sha256": _sha256(candidate_packet),
            "candidate_id_required_for_launch_reenable": True,
        },
        "source_packet": {
            "path": source_packet_path,
            "bytes": len(source_blob),
            "sha256": _sha256(source_blob),
            "header_bytes": source_header_bytes,
        },
        "candidate_packet": {
            "bytes": len(candidate_packet),
            "sha256": _sha256(candidate_packet),
            "header_bytes": candidate_header_bytes,
        },
        "packet_byte_delta": int(len(candidate_packet) - len(source_blob)),
        "header_byte_delta": int(candidate_header_bytes - source_header_bytes),
        "section_parity_rows": section_rows,
        "sections_exact_equal": section_parity,
        "source_metadata": {
            "key_count": len(source.metadata),
            "json_bytes": _json_len(source.metadata),
            "sha256": _json_sha(source.metadata),
        },
        "candidate_metadata": {
            "keys": list(minimal_metadata),
            "key_count": len(minimal_metadata),
            "json_bytes": _json_len(minimal_metadata),
            "sha256": _json_sha(minimal_metadata),
        },
        "removed_metadata": {
            "key_count": len(removed_metadata),
            "json_bytes": _json_len(removed_metadata),
            "sha256": _json_sha(removed_metadata),
            "top_level_keys": sorted(removed_metadata),
            "receiver_consumption_reason": (
                "removed keys are not read by current unpack/decode/inflate path; "
                "the section bytes remain unchanged and receiver replay is checked"
            ),
        },
        "receiver_pair_frame_equality_proof": proof,
        "receiver_contract_satisfied": receiver_contract,
        "full_video_receiver_contract_satisfied": full_video_receiver_contract,
        "runtime_consumption_proof_ready": full_video_receiver_contract,
        "hard_byte_ceiling_rows": [
            _hard_byte_ceiling_row(
                ceiling=int(ceiling),
                source_packet_bytes=len(source_blob),
                candidate_packet_bytes=len(candidate_packet),
                source_header_bytes=source_header_bytes,
                candidate_header_bytes=candidate_header_bytes,
            )
            for ceiling in hard_byte_ceilings
        ],
        "contest_compliance_contract": {
            "archive_bytes_changed": True,
            "archive_zip_materialized": False,
            "inflate_py_changed": False,
            "external_runtime_state_required": False,
            "removed_metadata_preserved_outside_archive": True,
            "candidate_id_bound": bool(bound_candidate_id),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "next_actions": _next_actions(
            receiver_contract=receiver_contract,
            full_video_receiver_contract=full_video_receiver_contract,
        ),
        "blockers": _blockers(
            receiver_contract=receiver_contract,
            full_video_receiver_contract=full_video_receiver_contract,
            candidate_id_bound=bool(bound_candidate_id),
        ),
        "raw_argv": list(raw_argv),
        **FALSE_AUTHORITY,
    }
    return report, candidate_packet


def _pack_minimal_snar1(
    *,
    sections: Mapping[str, bytes],
    metadata: Mapping[str, Any],
) -> bytes:
    cursor = 0
    section_headers: list[dict[str, Any]] = []
    payload_parts: list[bytes] = []
    for name in SECTION_ORDER:
        blob = bytes(sections[name])
        section_headers.append(
            {
                "name": name,
                "offset": cursor,
                "bytes": len(blob),
                "sha256": _sha256(blob),
            }
        )
        payload_parts.append(blob)
        cursor += len(blob)
    header = {
        "schema": SNERV_ARCHIVE_SCHEMA,
        "section_order": list(SECTION_ORDER),
        "sections": section_headers,
        "metadata": dict(metadata),
    }
    header_blob = json.dumps(
        header,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return (
        SNERV_ARCHIVE_MAGIC
        + struct.pack(HEADER_LEN_FMT, len(header_blob))
        + header_blob
        + b"".join(payload_parts)
    )


def _receiver_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    out = {
        key: metadata[key]
        for key in RECEIVER_METADATA_KEYS
        if key in metadata
    }
    if "carrier_hw" in metadata:
        out["carrier_hw"] = metadata["carrier_hw"]
    elif "orig_hw" in metadata:
        out["orig_hw"] = metadata["orig_hw"]
    else:
        raise SnervSnarHeaderMinimizerError(
            "source metadata missing carrier_hw/orig_hw required by receiver replay"
        )
    for key in ("n_pairs", "levels", "wavelet"):
        if key not in out:
            raise SnervSnarHeaderMinimizerError(
                f"source metadata missing receiver-required key: {key}"
            )
    return out


def _receiver_pair_frame_proof(
    source: Any,
    candidate: Any,
    *,
    requested_pair_indices: Sequence[int],
    full_video: bool,
    max_output_bytes: int,
) -> dict[str, Any]:
    pair_indices = (
        _full_video_pair_indices(source.metadata)
        if full_video
        else _proof_pair_indices(source.metadata, requested_pair_indices)
    )
    estimate = _pair_frame_output_bytes(source.metadata, pair_indices)
    if full_video:
        return _streaming_receiver_pair_frame_proof(
            source,
            candidate,
            pair_indices=pair_indices,
            estimated_output_bytes=estimate,
        )
    if estimate is not None and estimate > int(max_output_bytes):
        return {
            "status": "skipped_by_output_byte_guard",
            "scope": "sampled_pairs",
            "pair_indices": pair_indices,
            "estimated_output_bytes": estimate,
            "max_output_bytes": int(max_output_bytes),
            "exact_equal": None,
            "blockers": ["snerv_snar_header_minimizer_pair_frame_proof_skipped"],
        }
    try:
        source_frames = decode_snerv_archive_pair_frames_from_decoded(
            source,
            pair_indices,
        )
        candidate_frames = decode_snerv_archive_pair_frames_from_decoded(
            candidate,
            pair_indices,
        )
    except SnervArchiveError as exc:
        return {
            "status": "error",
            "scope": "sampled_pairs",
            "pair_indices": pair_indices,
            "estimated_output_bytes": estimate,
            "error": f"{type(exc).__name__}:{exc}",
            "exact_equal": False,
            "blockers": ["snerv_snar_header_minimizer_receiver_pair_decode_failed"],
        }
    exact = bool(np.array_equal(source_frames, candidate_frames))
    return {
        "status": "proven_exact" if exact else "failed",
        "scope": "sampled_pairs",
        "pair_indices": pair_indices,
        "estimated_output_bytes": estimate,
        "source_shape": list(source_frames.shape),
        "candidate_shape": list(candidate_frames.shape),
        "source_sha256": _sha256(np.ascontiguousarray(source_frames).tobytes()),
        "candidate_sha256": _sha256(np.ascontiguousarray(candidate_frames).tobytes()),
        "exact_equal": exact,
        "blockers": []
        if exact
        else ["snerv_snar_header_minimizer_receiver_pair_frames_changed"],
    }


def _streaming_receiver_pair_frame_proof(
    source: Any,
    candidate: Any,
    *,
    pair_indices: Sequence[int],
    estimated_output_bytes: int | None,
) -> dict[str, Any]:
    source_hash = hashlib.sha256()
    candidate_hash = hashlib.sha256()
    source_shape: list[int] | None = None
    candidate_shape: list[int] | None = None
    first_mismatch: int | None = None
    decoded_pairs = 0
    try:
        for pair_index in pair_indices:
            source_frames = decode_snerv_archive_pair_frames_from_decoded(
                source,
                [int(pair_index)],
            )
            candidate_frames = decode_snerv_archive_pair_frames_from_decoded(
                candidate,
                [int(pair_index)],
            )
            if source_shape is None:
                source_shape = list(source_frames.shape)
            if candidate_shape is None:
                candidate_shape = list(candidate_frames.shape)
            source_bytes = np.ascontiguousarray(source_frames).tobytes()
            candidate_bytes = np.ascontiguousarray(candidate_frames).tobytes()
            source_hash.update(source_bytes)
            candidate_hash.update(candidate_bytes)
            decoded_pairs += 1
            if first_mismatch is None and source_bytes != candidate_bytes:
                first_mismatch = int(pair_index)
    except SnervArchiveError as exc:
        return {
            "status": "error",
            "scope": "full_video_streaming",
            "pair_count": len(pair_indices),
            "decoded_pair_count": decoded_pairs,
            "estimated_output_bytes": estimated_output_bytes,
            "error": f"{type(exc).__name__}:{exc}",
            "exact_equal": False,
            "blockers": ["snerv_snar_header_minimizer_full_video_receiver_decode_failed"],
        }
    exact = first_mismatch is None
    return {
        "status": "proven_exact" if exact else "failed",
        "scope": "full_video_streaming",
        "pair_count": len(pair_indices),
        "decoded_pair_count": decoded_pairs,
        "pair_indices_summary": {
            "first": int(pair_indices[0]) if pair_indices else None,
            "last": int(pair_indices[-1]) if pair_indices else None,
        },
        "estimated_output_bytes": estimated_output_bytes,
        "first_pair_shape": source_shape,
        "first_candidate_shape": candidate_shape,
        "source_stream_sha256": source_hash.hexdigest(),
        "candidate_stream_sha256": candidate_hash.hexdigest(),
        "first_mismatch_pair_index": first_mismatch,
        "exact_equal": exact,
        "blockers": []
        if exact
        else ["snerv_snar_header_minimizer_full_video_receiver_frames_changed"],
    }


def _full_video_pair_indices(metadata: Mapping[str, Any]) -> list[int]:
    n_pairs = _positive_int(metadata.get("n_pairs"))
    if n_pairs is None:
        raise SnervSnarHeaderMinimizerError(
            "source metadata missing n_pairs for full-video receiver proof"
        )
    return list(range(int(n_pairs)))


def _proof_pair_indices(
    metadata: Mapping[str, Any],
    requested_pair_indices: Sequence[int],
) -> list[int]:
    if requested_pair_indices:
        return _dedupe_ints(requested_pair_indices)
    n_pairs = _positive_int(metadata.get("n_pairs"))
    if n_pairs is None or n_pairs <= 1:
        return [0]
    return [0, int(n_pairs) - 1]


def _pair_frame_output_bytes(
    metadata: Mapping[str, Any],
    pair_indices: Sequence[int],
) -> int | None:
    n_pairs = _positive_int(metadata.get("n_pairs"))
    if n_pairs is None:
        return None
    frames_per_pair = _positive_int(metadata.get("frames_per_pair")) or 2
    channels = _positive_int(metadata.get("channels")) or 3
    hw = metadata.get("carrier_hw", metadata.get("orig_hw"))
    if not isinstance(hw, Sequence) or len(hw) != 2:
        return None
    try:
        h, w = int(hw[0]), int(hw[1])
    except (TypeError, ValueError):
        return None
    return int(len(pair_indices) * frames_per_pair * channels * h * w * 4)


def _outer_header_bytes(packet: bytes) -> int:
    blob = bytes(packet)
    if not blob.startswith(SNERV_ARCHIVE_MAGIC):
        raise SnervSnarHeaderMinimizerError("input is not a SNAR1 packet")
    offset = len(SNERV_ARCHIVE_MAGIC)
    size = struct.calcsize(HEADER_LEN_FMT)
    if len(blob) < offset + size:
        raise SnervSnarHeaderMinimizerError("truncated SNAR1 header length")
    (header_len,) = struct.unpack(HEADER_LEN_FMT, blob[offset : offset + size])
    return int(len(SNERV_ARCHIVE_MAGIC) + size + int(header_len))


def _next_actions(
    *,
    receiver_contract: bool,
    full_video_receiver_contract: bool,
) -> list[str]:
    if full_video_receiver_contract:
        return [
            "rerun_snerv_lf_recode_admission_with_minimized_packet",
            "package_minimized_packet_as_archive_zip_with_runtime_custody",
            "run_full_video_local_replay_before_any_long_training_reenable",
            "prototype_snar2_binary_or_cbor_header_after_snar1_prune",
        ]
    if receiver_contract:
        return [
            "run_full_video_streaming_receiver_replay_for_minimized_packet",
            "keep_sampled_pair_proof_as_advisory_until_full_video_replay_passes",
            "prototype_snar2_binary_or_cbor_header_after_snar1_prune",
        ]
    return [
        "repair_minimized_header_receiver_replay",
        "keep_source_packet_as_authoritative_until_replay_passes",
    ]


def _blockers(
    *,
    receiver_contract: bool,
    full_video_receiver_contract: bool,
    candidate_id_bound: bool,
) -> list[str]:
    blockers = [
        "snerv_snar_header_minimization_false_authority",
        "not_packaged_as_contest_archive_zip",
        "full_video_scorer_replay_missing",
        "paired_contest_cpu_cuda_auth_eval_missing",
        "snar2_no_human_readable_label_bitstream_not_implemented",
    ]
    if not candidate_id_bound:
        blockers.append("snerv_snar_header_minimization_candidate_id_binding_missing")
    if not receiver_contract:
        blockers.append("snerv_snar_header_minimization_receiver_proof_failed")
    elif not full_video_receiver_contract:
        blockers.append(
            "snerv_snar_header_minimization_full_video_receiver_proof_missing"
        )
    return blockers


def _hard_byte_ceiling_row(
    *,
    ceiling: int,
    source_packet_bytes: int,
    candidate_packet_bytes: int,
    source_header_bytes: int,
    candidate_header_bytes: int,
) -> dict[str, Any]:
    return {
        "hard_byte_ceiling": int(ceiling),
        "source_packet_bytes": int(source_packet_bytes),
        "candidate_packet_bytes": int(candidate_packet_bytes),
        "source_packet_over_ceiling_bytes": max(
            int(source_packet_bytes) - int(ceiling),
            0,
        ),
        "candidate_packet_over_ceiling_bytes": max(
            int(candidate_packet_bytes) - int(ceiling),
            0,
        ),
        "candidate_packet_under_ceiling": int(candidate_packet_bytes) <= int(ceiling),
        "header_bytes_removed": int(source_header_bytes) - int(candidate_header_bytes),
    }


def _json_len(value: Any) -> int:
    return len(
        json.dumps(
            value,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _json_sha(value: Any) -> str:
    return _sha256(
        json.dumps(
            value,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _nonempty_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _dedupe_ints(values: Sequence[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        parsed = int(value)
        if parsed in seen:
            continue
        seen.add(parsed)
        out.append(parsed)
    return out


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(bytes(blob)).hexdigest()


__all__ = [
    "SCHEMA",
    "SnervSnarHeaderMinimizerError",
    "build_snerv_snar_header_minimization",
]
