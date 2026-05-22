# SPDX-License-Identifier: MIT
"""Byte-closed packet planning for selective FEC6 decoder-q mutations.

The bridge plan identifies MLX-positive windows for a full-video decoder-q
mutation. This module does not claim score authority. It verifies the mutation
against the FEC6 parent bytes and emits the compact archive-local patch contract
that a future selective inflate adapter must implement.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tac.optimization.fec6_byte_targets import parse_fec6_sections
from tac.optimization.fec6_decoder_mutations import (
    DECODER_BLOB_LEN,
    LATENT_BLOB_LEN,
    DecoderQMutation,
    extract_fec6_decoder_blob,
    prepare_decoder_blob,
    probe_q_mutation,
)
from tac.optimization.scorer_response_dataset import render_authority_markdown_block
from tac.pr101_split_brotli_codec import CONV4_STORAGE_PERMS

SCHEMA = "decoder_q_selective_runtime_packet_plan.v1"
TOOL = "tac.optimization.decoder_q_selective_runtime_packet"
BRIDGE_SCHEMA = "decoder_q_selective_window_bridge_plan.v1"
PACKET_MAGIC = b"DQS1"
SPEC_MAGIC = PACKET_MAGIC
SPEC_HEADER = struct.Struct("<4sBBHbH")
CONTEST_RATE_DENOMINATOR_BYTES = 37_545_489
FEC6_PAIR_COUNT = 600

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
}

FRAME_POLICY_CODES = {
    "pair_all_frames": 1,
    "segnet_last_frame_only": 2,
}
FRAME_POLICY_BY_CODE = {value: key for key, value in FRAME_POLICY_CODES.items()}
PAIR_ENCODING_CODES = {
    "raw_u16": 0,
    "sorted_gap_uleb": 1,
}
PAIR_ENCODING_BY_CODE = {value: key for key, value in PAIR_ENCODING_CODES.items()}
RAW_PAIR_ENCODING = "raw_u16"
COMPACT_PAIR_ENCODING = "sorted_gap_uleb"


class DecoderQSelectiveRuntimePacketError(ValueError):
    """Raised when a selective runtime packet plan would lose custody."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DecoderQSelectiveRuntimePacketError(f"{path}: expected JSON object")
    return payload


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _require_false_authority(payload: dict[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if payload.get(key) is not False:
            raise DecoderQSelectiveRuntimePacketError(
                f"{label} {key} must be explicit false"
            )


def _as_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise DecoderQSelectiveRuntimePacketError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise DecoderQSelectiveRuntimePacketError(f"{label} must be an integer") from exc
    if result != value and not (isinstance(value, str) and str(result) == value):
        raise DecoderQSelectiveRuntimePacketError(f"{label} must be integral")
    return result


def _resolve_path(path_value: Any, *, repo_root: Path) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise DecoderQSelectiveRuntimePacketError("artifact path must be a non-empty string")
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _read_single_stored_member(path: Path) -> tuple[str, bytes, dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise DecoderQSelectiveRuntimePacketError(
                f"{path}: expected exactly one ZIP member, found {len(infos)}"
            )
        info = infos[0]
        if info.compress_type != zipfile.ZIP_STORED:
            raise DecoderQSelectiveRuntimePacketError(
                f"{path}: member {info.filename!r} must be ZIP_STORED"
            )
        data = zf.read(info.filename)
    return (
        info.filename,
        data,
        {
            "member_name": info.filename,
            "compress_type": "ZIP_STORED",
            "member_bytes": len(data),
            "member_sha256": sha256_bytes(data),
            "crc32_hex": f"{info.CRC:08x}",
        },
    )


def _diff_ranges(lhs: bytes, rhs: bytes) -> list[dict[str, int]]:
    if len(lhs) != len(rhs):
        raise DecoderQSelectiveRuntimePacketError("diff inputs must have equal length")
    ranges: list[dict[str, int]] = []
    start: int | None = None
    last: int | None = None
    for index, (left, right) in enumerate(zip(lhs, rhs, strict=True)):
        if left == right:
            if start is not None and last is not None:
                ranges.append({"start": start, "end": last + 1, "length": last + 1 - start})
                start = None
                last = None
            continue
        if start is None:
            start = index
        last = index
    if start is not None and last is not None:
        ranges.append({"start": start, "end": last + 1, "length": last + 1 - start})
    return ranges


def _unravel_q_offset(offset: int, shape: Sequence[int]) -> tuple[int, ...]:
    if offset < 0:
        raise DecoderQSelectiveRuntimePacketError("q_offset must be non-negative")
    remaining = int(offset)
    out = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        dim = int(shape[axis])
        if dim <= 0:
            raise DecoderQSelectiveRuntimePacketError("tensor shape dimensions must be positive")
        out[axis] = remaining % dim
        remaining //= dim
    if remaining:
        raise DecoderQSelectiveRuntimePacketError(
            f"q_offset {offset} exceeds tensor storage shape {list(shape)}"
        )
    return tuple(out)


def tensor_index_for_stored_q_offset(
    *,
    tensor_shape: Sequence[int],
    storage_index: int,
    q_offset: int,
) -> dict[str, Any]:
    """Map a flat stored q offset to the decoded state_dict tensor index."""

    shape = tuple(int(value) for value in tensor_shape)
    if int(storage_index) in CONV4_STORAGE_PERMS:
        storage_perm = tuple(int(value) for value in CONV4_STORAGE_PERMS[int(storage_index)])
        stored_shape = tuple(shape[axis] for axis in storage_perm)
        stored_index = _unravel_q_offset(int(q_offset), stored_shape)
        tensor_index = [0] * len(shape)
        for storage_axis, tensor_axis in enumerate(storage_perm):
            tensor_index[tensor_axis] = stored_index[storage_axis]
    else:
        storage_perm = tuple(range(len(shape)))
        stored_shape = shape
        stored_index = _unravel_q_offset(int(q_offset), stored_shape)
        tensor_index = list(stored_index)
    return {
        "storage_shape": list(stored_shape),
        "storage_perm": list(storage_perm),
        "stored_index": list(stored_index),
        "tensor_index": tensor_index,
    }


def _selected_pair_indices(
    bridge_plan: dict[str, Any],
    *,
    max_units: int | None,
    selected_pair_indices: Sequence[int] | None = None,
) -> list[int]:
    units = bridge_plan.get("work_units")
    if not isinstance(units, list) or not units:
        raise DecoderQSelectiveRuntimePacketError("bridge plan work_units[] missing")
    if selected_pair_indices is not None and max_units is not None:
        raise DecoderQSelectiveRuntimePacketError(
            "selected_pair_indices and max_units are mutually exclusive"
        )
    if selected_pair_indices is not None:
        available = set()
        for index, unit in enumerate(units):
            if not isinstance(unit, dict):
                raise DecoderQSelectiveRuntimePacketError(f"work unit {index} must be object")
            _require_false_authority(unit, label=f"work unit {index}")
            window = unit.get("pair_window")
            if not isinstance(window, list) or len(window) != 2:
                raise DecoderQSelectiveRuntimePacketError(f"work unit {index} pair_window invalid")
            start = _as_int(window[0], label=f"work unit {index} pair_window[0]")
            end = _as_int(window[1], label=f"work unit {index} pair_window[1]")
            if end != start + 1:
                raise DecoderQSelectiveRuntimePacketError(
                    "selective packet v1 only supports singleton pair windows; "
                    f"got {window}"
                )
            available.add(start)
        pairs = _canonical_pair_indices(selected_pair_indices)
        missing = sorted(set(pairs) - available)
        if missing:
            raise DecoderQSelectiveRuntimePacketError(
                f"selected pairs missing from bridge work units: {missing}"
            )
        return pairs
    if max_units is not None:
        if max_units <= 0:
            raise DecoderQSelectiveRuntimePacketError("max_units must be positive")
        units = units[:max_units]
    pairs: list[int] = []
    for index, unit in enumerate(units):
        if not isinstance(unit, dict):
            raise DecoderQSelectiveRuntimePacketError(f"work unit {index} must be object")
        _require_false_authority(unit, label=f"work unit {index}")
        window = unit.get("pair_window")
        if not isinstance(window, list) or len(window) != 2:
            raise DecoderQSelectiveRuntimePacketError(f"work unit {index} pair_window invalid")
        start = _as_int(window[0], label=f"work unit {index} pair_window[0]")
        end = _as_int(window[1], label=f"work unit {index} pair_window[1]")
        if end != start + 1:
            raise DecoderQSelectiveRuntimePacketError(
                "selective packet v1 only supports singleton pair windows; "
                f"got {window}"
            )
        pairs.append(start)
    if len(set(pairs)) != len(pairs):
        raise DecoderQSelectiveRuntimePacketError("selected pair indices contain duplicates")
    return _canonical_pair_indices(sorted(pairs))


def affected_frames_for_pairs(pair_indices: Sequence[int], *, frame_policy: str) -> list[int]:
    if frame_policy not in FRAME_POLICY_CODES:
        raise DecoderQSelectiveRuntimePacketError(
            f"unknown frame_policy {frame_policy!r}; expected one of {sorted(FRAME_POLICY_CODES)}"
        )
    frames: list[int] = []
    for pair in pair_indices:
        base = int(pair) * 2
        if frame_policy == "pair_all_frames":
            frames.extend([base, base + 1])
        elif frame_policy == "segnet_last_frame_only":
            frames.append(base + 1)
    return sorted(set(frames))


def _canonical_pair_indices(pair_indices: Sequence[int]) -> list[int]:
    if len(pair_indices) > 65535:
        raise DecoderQSelectiveRuntimePacketError("too many selected pairs")
    pairs = [
        _as_int(pair, label=f"pair_indices[{index}]")
        for index, pair in enumerate(pair_indices)
    ]
    if pairs != sorted(pairs):
        raise DecoderQSelectiveRuntimePacketError("pair indices must be sorted")
    if len(set(pairs)) != len(pairs):
        raise DecoderQSelectiveRuntimePacketError("pair indices contain duplicates")
    for pair in pairs:
        if not 0 <= int(pair) < FEC6_PAIR_COUNT:
            raise DecoderQSelectiveRuntimePacketError(
                f"pair index out of FEC6 range: {pair}"
            )
    return pairs


def _pack_mode_byte(*, frame_policy: str, pair_encoding: str) -> int:
    frame_policy_code = FRAME_POLICY_CODES.get(frame_policy)
    if frame_policy_code is None:
        raise DecoderQSelectiveRuntimePacketError(f"unknown frame_policy {frame_policy!r}")
    encoding_code = PAIR_ENCODING_CODES.get(pair_encoding)
    if encoding_code is None:
        raise DecoderQSelectiveRuntimePacketError(f"unknown pair_encoding {pair_encoding!r}")
    if frame_policy_code > 0x0F or encoding_code > 0x0F:
        raise DecoderQSelectiveRuntimePacketError("DQS1 mode byte codes must fit nibbles")
    return (encoding_code << 4) | frame_policy_code


def _unpack_mode_byte(mode_byte: int) -> tuple[int, str, int, str]:
    frame_policy_code = int(mode_byte) & 0x0F
    pair_encoding_code = int(mode_byte) >> 4
    frame_policy = FRAME_POLICY_BY_CODE.get(frame_policy_code)
    if frame_policy is None:
        raise DecoderQSelectiveRuntimePacketError(
            f"unknown DQS1 frame_policy code: {frame_policy_code}"
        )
    pair_encoding = PAIR_ENCODING_BY_CODE.get(pair_encoding_code)
    if pair_encoding is None:
        raise DecoderQSelectiveRuntimePacketError(
            f"unknown DQS1 pair_encoding code: {pair_encoding_code}"
        )
    return frame_policy_code, frame_policy, pair_encoding_code, pair_encoding


def _pack_uleb128(value: int) -> bytes:
    if value < 0:
        raise DecoderQSelectiveRuntimePacketError("ULEB value must be non-negative")
    out = bytearray()
    remaining = int(value)
    while True:
        byte = remaining & 0x7F
        remaining >>= 7
        if remaining:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _unpack_uleb128(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    start = offset
    while offset < len(payload):
        byte = payload[offset]
        value |= (byte & 0x7F) << shift
        offset += 1
        if byte & 0x80 == 0:
            if payload[start:offset] != _pack_uleb128(value):
                raise DecoderQSelectiveRuntimePacketError("noncanonical DQS1 ULEB gap")
            return value, offset
        shift += 7
        if shift > 28:
            raise DecoderQSelectiveRuntimePacketError("DQS1 ULEB gap is too large")
    raise DecoderQSelectiveRuntimePacketError("truncated DQS1 ULEB gap")


def _pack_pair_index_body(pairs: Sequence[int], *, pair_encoding: str) -> bytes:
    if pair_encoding == RAW_PAIR_ENCODING:
        return b"".join(struct.pack("<H", int(pair)) for pair in pairs)
    if pair_encoding == COMPACT_PAIR_ENCODING:
        out = bytearray()
        previous: int | None = None
        for pair in pairs:
            value = int(pair) if previous is None else int(pair) - previous
            if previous is not None and value <= 0:
                raise DecoderQSelectiveRuntimePacketError(
                    "sorted_gap_uleb pair gaps must be positive"
                )
            out.extend(_pack_uleb128(value))
            previous = int(pair)
        return bytes(out)
    raise DecoderQSelectiveRuntimePacketError(f"unknown pair_encoding {pair_encoding!r}")


def _unpack_pair_index_body(
    payload: bytes,
    *,
    count: int,
    pair_encoding: str,
) -> list[int]:
    if pair_encoding == RAW_PAIR_ENCODING:
        expected = int(count) * 2
        if len(payload) != expected:
            raise DecoderQSelectiveRuntimePacketError(
                f"DQS1 descriptor length mismatch: got {len(payload) + SPEC_HEADER.size}, "
                f"expected {expected + SPEC_HEADER.size}"
            )
        return [
            struct.unpack_from("<H", payload, index * 2)[0]
            for index in range(int(count))
        ]
    if pair_encoding == COMPACT_PAIR_ENCODING:
        pair_indices: list[int] = []
        previous: int | None = None
        offset = 0
        for _ in range(int(count)):
            value, offset = _unpack_uleb128(payload, offset)
            pair = value if previous is None else previous + value
            if previous is not None and value <= 0:
                raise DecoderQSelectiveRuntimePacketError(
                    "DQS1 sorted_gap_uleb gap must be positive"
                )
            if pair > 65535:
                raise DecoderQSelectiveRuntimePacketError("DQS1 pair index must fit u16")
            pair_indices.append(pair)
            previous = pair
        if offset != len(payload):
            raise DecoderQSelectiveRuntimePacketError(
                "DQS1 descriptor has trailing pair-index bytes"
            )
        return pair_indices
    raise DecoderQSelectiveRuntimePacketError(f"unknown pair_encoding {pair_encoding!r}")


def choose_dqs1_pair_encoding(pair_indices: Sequence[int]) -> dict[str, Any]:
    """Return the smallest supported DQS1 pair-index encoding for sorted pairs."""

    pairs = _canonical_pair_indices(pair_indices)
    candidates: list[dict[str, Any]] = []
    for pair_encoding, pair_encoding_code in PAIR_ENCODING_CODES.items():
        body = _pack_pair_index_body(pairs, pair_encoding=pair_encoding)
        candidates.append(
            {
                "pair_encoding": pair_encoding,
                "pair_encoding_code": pair_encoding_code,
                "pair_index_payload_bytes": len(body),
                "descriptor_bytes": SPEC_HEADER.size + len(body),
            }
        )
    best = min(
        candidates,
        key=lambda row: (
            int(row["descriptor_bytes"]),
            0 if row["pair_encoding"] == RAW_PAIR_ENCODING else 1,
            str(row["pair_encoding"]),
        ),
    )
    return {"selected": best, "candidates": candidates}


def pack_dqs1_payload(
    *,
    pair_indices: Sequence[int],
    frame_policy: str,
    storage_index: int,
    q_offset: int,
    delta: int,
    pair_encoding: str = RAW_PAIR_ENCODING,
) -> bytes:
    """Pack the compact archive-local selective mutation descriptor."""

    if frame_policy not in FRAME_POLICY_CODES:
        raise DecoderQSelectiveRuntimePacketError(f"unknown frame_policy {frame_policy!r}")
    if not 0 <= int(storage_index) <= 255:
        raise DecoderQSelectiveRuntimePacketError("storage_index must fit u8")
    if not 0 <= int(q_offset) <= 65535:
        raise DecoderQSelectiveRuntimePacketError("q_offset must fit u16 in DQS1")
    if not -128 <= int(delta) <= 127:
        raise DecoderQSelectiveRuntimePacketError("delta must fit i8")
    pairs = _canonical_pair_indices(pair_indices)
    pair_index_body = _pack_pair_index_body(pairs, pair_encoding=pair_encoding)
    payload = bytearray()
    payload.extend(
        SPEC_HEADER.pack(
            SPEC_MAGIC,
            _pack_mode_byte(frame_policy=frame_policy, pair_encoding=pair_encoding),
            int(storage_index),
            int(q_offset),
            int(delta),
            len(pairs),
        )
    )
    payload.extend(pair_index_body)
    return bytes(payload)


def unpack_dqs1_payload(payload: bytes) -> dict[str, Any]:
    """Unpack and validate a compact DQS1 selective mutation descriptor."""

    if len(payload) < SPEC_HEADER.size:
        raise DecoderQSelectiveRuntimePacketError("DQS1 descriptor truncated")
    magic, mode_byte, storage_index, q_offset, delta, count = SPEC_HEADER.unpack_from(
        payload
    )
    if magic != SPEC_MAGIC:
        raise DecoderQSelectiveRuntimePacketError(
            f"DQS1 descriptor magic mismatch: {magic!r}"
        )
    frame_policy_code, frame_policy, pair_encoding_code, pair_encoding = _unpack_mode_byte(
        int(mode_byte)
    )
    pair_indices = _unpack_pair_index_body(
        payload[SPEC_HEADER.size:],
        count=int(count),
        pair_encoding=pair_encoding,
    )
    pair_indices = _canonical_pair_indices(pair_indices)
    return {
        "frame_policy": frame_policy,
        "frame_policy_code": int(frame_policy_code),
        "mode_byte": int(mode_byte),
        "pair_encoding": pair_encoding,
        "pair_encoding_code": int(pair_encoding_code),
        "storage_index": int(storage_index),
        "q_offset": int(q_offset),
        "delta": int(delta),
        "pair_indices": pair_indices,
    }


def _validate_bridge_plan(bridge_plan: dict[str, Any]) -> None:
    if bridge_plan.get("schema") != BRIDGE_SCHEMA:
        raise DecoderQSelectiveRuntimePacketError("bridge plan schema mismatch")
    _require_false_authority(bridge_plan, label="bridge plan")
    if bridge_plan.get("candidate_generation_only") is not True:
        raise DecoderQSelectiveRuntimePacketError("bridge plan must be candidate_generation_only")
    if bridge_plan.get("evidence_grade") != "macOS-MLX-research-signal":
        raise DecoderQSelectiveRuntimePacketError("bridge plan evidence_grade must remain MLX")


def build_decoder_q_selective_runtime_packet_plan(
    bridge_plan: dict[str, Any],
    *,
    base_archive: Path,
    repo_root: Path,
    frame_policy: str = "pair_all_frames",
    max_units: int | None = None,
    selected_pair_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Verify a bridge plan and emit a compact selective-runtime packet plan."""

    _validate_bridge_plan(bridge_plan)
    pair_indices = _selected_pair_indices(
        bridge_plan,
        max_units=max_units,
        selected_pair_indices=selected_pair_indices,
    )
    affected_frames = affected_frames_for_pairs(pair_indices, frame_policy=frame_policy)

    candidate = bridge_plan.get("materialized_decoder_q_candidate")
    if not isinstance(candidate, dict):
        raise DecoderQSelectiveRuntimePacketError("bridge plan candidate missing")
    candidate_archive = _resolve_path(candidate.get("archive_zip_path"), repo_root=repo_root)
    if not candidate_archive.is_file():
        raise DecoderQSelectiveRuntimePacketError(f"candidate archive missing: {candidate_archive}")
    if not base_archive.is_file():
        raise DecoderQSelectiveRuntimePacketError(f"base archive missing: {base_archive}")

    base_member_name, base_member, base_zip_meta = _read_single_stored_member(base_archive)
    candidate_member_name, candidate_member, candidate_zip_meta = _read_single_stored_member(candidate_archive)
    if base_member_name != candidate_member_name:
        raise DecoderQSelectiveRuntimePacketError(
            f"ZIP member name mismatch: base={base_member_name!r} candidate={candidate_member_name!r}"
        )
    if len(base_member) != len(candidate_member):
        raise DecoderQSelectiveRuntimePacketError("candidate member length changed")

    mutation = candidate.get("mutation")
    if not isinstance(mutation, dict):
        raise DecoderQSelectiveRuntimePacketError("candidate mutation missing")
    decoder_mutation = DecoderQMutation(
        tensor_name=str(mutation.get("tensor_name")),
        q_offset=_as_int(mutation.get("q_offset"), label="mutation q_offset"),
        delta=_as_int(mutation.get("delta"), label="mutation delta"),
    )

    base_decoder = extract_fec6_decoder_blob(base_member)
    candidate_decoder = extract_fec6_decoder_blob(candidate_member)
    base_decoder_sha = sha256_bytes(base_decoder)
    candidate_decoder_sha = sha256_bytes(candidate_decoder)
    if mutation.get("source_decoder_sha256") != base_decoder_sha:
        raise DecoderQSelectiveRuntimePacketError(
            "base archive decoder SHA does not match bridge mutation source_decoder_sha256"
        )
    if mutation.get("mutated_decoder_sha256") != candidate_decoder_sha:
        raise DecoderQSelectiveRuntimePacketError(
            "candidate archive decoder SHA does not match bridge mutation mutated_decoder_sha256"
        )

    prepared = prepare_decoder_blob(base_decoder)
    probe = probe_q_mutation(prepared, decoder_mutation)
    if probe.mutated_decoder_sha256 != candidate_decoder_sha:
        raise DecoderQSelectiveRuntimePacketError(
            "recomputed q mutation does not match materialized candidate decoder"
        )
    if not probe.fixed_length_runtime_compatible:
        raise DecoderQSelectiveRuntimePacketError("mutation is not fixed-length compatible")

    tensor_index = tensor_index_for_stored_q_offset(
        tensor_shape=probe.tensor.shape,
        storage_index=probe.tensor.storage_index,
        q_offset=decoder_mutation.q_offset,
    )
    pair_encoding_choice = choose_dqs1_pair_encoding(pair_indices)
    selected_pair_encoding = str(pair_encoding_choice["selected"]["pair_encoding"])
    packet_payload = pack_dqs1_payload(
        pair_indices=pair_indices,
        frame_policy=frame_policy,
        storage_index=probe.tensor.storage_index,
        q_offset=decoder_mutation.q_offset,
        delta=decoder_mutation.delta,
        pair_encoding=selected_pair_encoding,
    )
    parsed_packet_payload = unpack_dqs1_payload(packet_payload)
    if parsed_packet_payload["pair_indices"] != pair_indices:
        raise DecoderQSelectiveRuntimePacketError("DQS1 packet pair encoding round-trip failed")

    decoder_diff_ranges = _diff_ranges(base_decoder, candidate_decoder)
    member_diff_ranges = _diff_ranges(base_member, candidate_member)
    sections = {
        section.name: section
        for section in parse_fec6_sections(
            base_member,
            decoder_blob_len=DECODER_BLOB_LEN,
            latent_blob_len=LATENT_BLOB_LEN,
        )
    }
    decoder_range = sections["decoder"].byte_range
    outside_decoder_diff_bytes = sum(
        1
        for index, (left, right) in enumerate(zip(base_member, candidate_member, strict=True))
        if left != right and not (decoder_range.start <= index < decoder_range.end)
    )
    if outside_decoder_diff_bytes:
        raise DecoderQSelectiveRuntimePacketError(
            f"candidate differs outside decoder section: {outside_decoder_diff_bytes} bytes"
        )

    descriptor_len = len(packet_payload)
    rate_delta = 25.0 * descriptor_len / CONTEST_RATE_DENOMINATOR_BYTES
    selected_pair_set = set(pair_indices)
    observed_gain_sum = 0.0
    for unit in bridge_plan["work_units"]:
        window = unit.get("pair_window")
        if isinstance(window, list) and window:
            start = int(window[0])
            if start in selected_pair_set:
                observed_gain_sum += float(unit.get("observed_mlx_gain", 0.0))
    return {
        "schema": SCHEMA,
        "producer": TOOL,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "evidence_grade": "macOS-MLX-research-signal",
        "evidence_tag": "[macOS-MLX research-signal]",
        "packet_status": "runtime_packet_l0_byte_plan_ready",
        "allowed_use": "byte_closed_selective_inflate_adapter_implementation_plan",
        "base_archive": {
            "path": str(base_archive),
            "zip_bytes": base_archive.stat().st_size,
            "zip_sha256": file_sha256(base_archive),
            **base_zip_meta,
            "decoder_sha256": base_decoder_sha,
        },
        "materialized_full_video_candidate": {
            "path": str(candidate_archive),
            "zip_bytes": candidate_archive.stat().st_size,
            "zip_sha256": file_sha256(candidate_archive),
            **candidate_zip_meta,
            "decoder_sha256": candidate_decoder_sha,
        },
        "mutation": {
            **probe.mutation.as_dict(),
            "mutation_id": probe.mutation_id,
            "q_before": probe.q_before,
            "q_after": probe.q_after,
            "tensor": probe.tensor.as_dict(),
            "tensor_index_mapping": tensor_index,
            "source_decoder_sha256": probe.source_decoder_sha256,
            "mutated_decoder_sha256": probe.mutated_decoder_sha256,
        },
        "selective_packet": {
            "wire_format": (
                "member = legacy_FP11_FEC6_member || DQS1_trailer; "
                "DQS1_trailer = DQS1,u8 mode,u8 storage_index,u16 q_offset,i8 delta,"
                "u16 count,pair_index_payload; mode low nibble = frame_policy, "
                "mode high nibble = pair_encoding"
            ),
            "frame_policy": frame_policy,
            "frame_policy_code": FRAME_POLICY_CODES[frame_policy],
            "mode_byte": parsed_packet_payload["mode_byte"],
            "pair_encoding": parsed_packet_payload["pair_encoding"],
            "pair_encoding_code": parsed_packet_payload["pair_encoding_code"],
            "pair_index_payload_bytes": pair_encoding_choice["selected"][
                "pair_index_payload_bytes"
            ],
            "pair_encoding_candidates": pair_encoding_choice["candidates"],
            "descriptor_bytes": descriptor_len,
            "wrapper_header_bytes": 0,
            "wrapped_member_overhead_bytes": descriptor_len,
            "payload_bytes": descriptor_len,
            "payload_sha256": sha256_bytes(packet_payload),
            "selected_pair_indices": pair_indices,
            "selected_pair_count": len(pair_indices),
            "affected_frame_indices": affected_frames,
            "affected_frame_count": len(affected_frames),
            "estimated_archive_byte_delta_if_appended_to_member": descriptor_len,
            "estimated_archive_byte_delta_if_wrapped_member": descriptor_len,
            "estimated_rate_score_delta": rate_delta,
            "non_authoritative_mlx_gain_sum": observed_gain_sum,
            "net_gain_after_rate_if_mlx_additive_non_authoritative": observed_gain_sum - rate_delta,
        },
        "diff_profile": {
            "decoder_blob_diff_bytes_full_recompressed_candidate": sum(
                value["length"] for value in decoder_diff_ranges
            ),
            "decoder_blob_diff_range_count": len(decoder_diff_ranges),
            "member_diff_bytes": sum(value["length"] for value in member_diff_ranges),
            "member_diff_range_count": len(member_diff_ranges),
            "outside_decoder_diff_bytes": outside_decoder_diff_bytes,
            "first_decoder_diff_ranges": decoder_diff_ranges[:16],
            "note": (
                "Full recompressed decoder differs widely; selective runtime must "
                "apply the q-domain patch before decoder state construction, not "
                "store a byte diff of the recompressed Brotli blob."
            ),
        },
        "runtime_adapter_contract": {
            "proposed_wrapper": (
                "archive-member tail extension: legacy FP11 source + selector "
                "plus charged DQS1 payload after selector"
            ),
            "legacy_fp11_parser_reusable_unmodified": False,
            "must_keep_patch_bytes_inside_archive_zip_member": True,
            "must_derive_mutated_decoder_state_from_base_decoder_blob_and_dqs1_patch": True,
            "must_not_import_scorer_or_use_network": True,
            "must_decode_dqs1_mode_byte_pair_encoding": True,
            "batch_decode_strategy": (
                "decode base batch; when a local batch contains selected pairs, "
                "decode the full same local batch with the mutated decoder and "
                "splice only the selected rows according to frame_policy before "
                "selector transforms"
            ),
            "selector_order": "apply existing FEC6 selector after selective decoder-frame stitching",
        },
        "dispatch_blockers": [
            "packet plan must be materialized with selective runtime adapter before use",
            "official inflate.sh raw-output locality controls not run",
            "local advisory scorer not run on selective packet",
            "exact contest auth eval not run",
        ],
    }


def render_decoder_q_selective_runtime_packet_markdown(plan: dict[str, Any]) -> str:
    packet = plan.get("selective_packet")
    if not isinstance(packet, dict):
        packet = {}
    mutation = plan.get("mutation")
    if not isinstance(mutation, dict):
        mutation = {}
    lines = [
        "# Decoder-Q Selective Runtime Packet Plan",
        "",
        f"- Packet status: `{plan.get('packet_status')}`",
        f"- Frame policy: `{packet.get('frame_policy')}`",
        f"- Pair encoding: `{packet.get('pair_encoding')}`",
        f"- Selected pairs: `{packet.get('selected_pair_count')}`",
        f"- Affected frames: `{packet.get('affected_frame_count')}`",
        f"- DQS1 payload bytes: `{packet.get('payload_bytes')}`",
        f"- Pair-index payload bytes: `{packet.get('pair_index_payload_bytes')}`",
        f"- Estimated rate delta: `{packet.get('estimated_rate_score_delta')}`",
        f"- Non-authoritative MLX gain sum: `{packet.get('non_authoritative_mlx_gain_sum')}`",
        f"- Mutation: `{mutation.get('tensor_name')}` q_offset=`{mutation.get('q_offset')}` delta=`{mutation.get('delta')}`",
        "",
    ]
    lines.extend(render_authority_markdown_block(plan))
    lines.extend(["## Runtime Contract", ""])
    contract = plan.get("runtime_adapter_contract")
    if isinstance(contract, dict):
        for key, value in contract.items():
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Dispatch Blockers", ""])
    for blocker in plan.get("dispatch_blockers", []):
        lines.append(f"- {blocker}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "BRIDGE_SCHEMA",
    "FALSE_AUTHORITY",
    "FRAME_POLICY_BY_CODE",
    "FRAME_POLICY_CODES",
    "PACKET_MAGIC",
    "PAIR_ENCODING_BY_CODE",
    "PAIR_ENCODING_CODES",
    "SCHEMA",
    "SPEC_MAGIC",
    "DecoderQSelectiveRuntimePacketError",
    "affected_frames_for_pairs",
    "build_decoder_q_selective_runtime_packet_plan",
    "choose_dqs1_pair_encoding",
    "dumps_json",
    "load_json_object",
    "pack_dqs1_payload",
    "render_decoder_q_selective_runtime_packet_markdown",
    "tensor_index_for_stored_q_offset",
    "unpack_dqs1_payload",
]
