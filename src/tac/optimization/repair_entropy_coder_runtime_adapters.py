# SPDX-License-Identifier: MIT
"""Decode adapters for repair entropy-coder prototype packets.

These functions are intentionally stdlib-only and decode-only on the receiver
side. They are reusable by archive materializers and by future contest inflate
adapters; they do not inspect scorer state or grant score authority.
"""

from __future__ import annotations

import lzma
import struct
from collections.abc import Mapping
from typing import Any

from tac.repo_io import sha256_bytes

RANGE_CODER_PROTOTYPE_MAGIC = b"TACRNG1\0"
ANS_CODER_PROTOTYPE_MAGIC = b"TACANS1\0"
ANS_SCALE_BITS = 12
ANS_TOTAL_FREQ = 1 << ANS_SCALE_BITS
ANS_BYTE_L = 1 << 23

REPAIR_ENTROPY_CODER_RUNTIME_ADAPTER_MANIFEST_SCHEMA = (
    "repair_entropy_coder_runtime_adapter_manifest.v1"
)


class RepairEntropyCoderRuntimeAdapterError(ValueError):
    """Raised when an entropy-coder prototype packet cannot be decoded."""


def range_lzma_prototype_encode(payload: bytes) -> bytes:
    """Encode a prototype range-coder packet with a stdlib LZMA codestream."""

    encoded = lzma.compress(
        payload,
        format=lzma.FORMAT_ALONE,
        preset=9 | lzma.PRESET_EXTREME,
    )
    return (
        RANGE_CODER_PROTOTYPE_MAGIC
        + struct.pack("<BQ", 1, len(payload))
        + bytes.fromhex(sha256_bytes(payload))
        + struct.pack("<I", len(encoded))
        + encoded
    )


def range_lzma_prototype_decode(packet: bytes) -> bytes:
    """Decode a prototype range-coder packet and verify its embedded hash."""

    header_size = len(RANGE_CODER_PROTOTYPE_MAGIC) + struct.calcsize("<BQ") + 32 + 4
    if len(packet) < header_size or not packet.startswith(RANGE_CODER_PROTOTYPE_MAGIC):
        raise RepairEntropyCoderRuntimeAdapterError("range prototype packet has invalid magic")
    offset = len(RANGE_CODER_PROTOTYPE_MAGIC)
    version, original_len = struct.unpack_from("<BQ", packet, offset)
    offset += struct.calcsize("<BQ")
    if version != 1:
        raise RepairEntropyCoderRuntimeAdapterError("range prototype packet version unsupported")
    expected_sha = packet[offset : offset + 32].hex()
    offset += 32
    (encoded_len,) = struct.unpack_from("<I", packet, offset)
    offset += 4
    encoded = packet[offset : offset + encoded_len]
    if len(encoded) != encoded_len or offset + encoded_len != len(packet):
        raise RepairEntropyCoderRuntimeAdapterError("range prototype packet length mismatch")
    decoded = lzma.decompress(encoded, format=lzma.FORMAT_ALONE)
    if len(decoded) != original_len or sha256_bytes(decoded) != expected_sha:
        raise RepairEntropyCoderRuntimeAdapterError("range prototype packet decode proof failed")
    return decoded


def _normalise_ans_frequencies(payload: bytes) -> dict[int, int]:
    if not payload:
        return {0: ANS_TOTAL_FREQ}
    counts: dict[int, int] = {}
    for byte in payload:
        counts[byte] = counts.get(byte, 0) + 1
    raw: dict[int, float] = {
        symbol: count * ANS_TOTAL_FREQ / len(payload)
        for symbol, count in counts.items()
    }
    freqs = {symbol: max(1, int(value)) for symbol, value in raw.items()}
    while sum(freqs.values()) < ANS_TOTAL_FREQ:
        symbol = max(
            freqs,
            key=lambda item: (raw[item] - freqs[item], counts[item], -item),
        )
        freqs[symbol] += 1
    while sum(freqs.values()) > ANS_TOTAL_FREQ:
        candidates = [symbol for symbol, freq in freqs.items() if freq > 1]
        if not candidates:
            break
        symbol = min(
            candidates,
            key=lambda item: (raw[item] - freqs[item], counts[item], item),
        )
        freqs[symbol] -= 1
    return dict(sorted(freqs.items()))


def _ans_model_tables(freqs: Mapping[int, int]) -> tuple[dict[int, int], list[int]]:
    starts: dict[int, int] = {}
    decode_table = [-1] * ANS_TOTAL_FREQ
    cursor = 0
    for symbol, freq in sorted(freqs.items()):
        starts[int(symbol)] = cursor
        for slot in range(cursor, cursor + int(freq)):
            decode_table[slot] = int(symbol)
        cursor += int(freq)
    if cursor != ANS_TOTAL_FREQ or any(symbol < 0 for symbol in decode_table):
        raise RepairEntropyCoderRuntimeAdapterError("ANS prototype frequency table invalid")
    return starts, decode_table


def _ans_rans_encode(payload: bytes, freqs: Mapping[int, int]) -> bytes:
    starts, _decode_table = _ans_model_tables(freqs)
    state = ANS_BYTE_L
    stream = bytearray()
    for symbol in reversed(payload):
        freq = int(freqs[symbol])
        start = starts[symbol]
        x_max = ((ANS_BYTE_L >> ANS_SCALE_BITS) << 8) * freq
        while state >= x_max:
            stream.append(state & 0xFF)
            state >>= 8
        state = ((state // freq) << ANS_SCALE_BITS) + (state % freq) + start
    return struct.pack("<I", state) + bytes(reversed(stream))


def _ans_rans_decode(encoded: bytes, *, output_len: int, freqs: Mapping[int, int]) -> bytes:
    if len(encoded) < 4:
        raise RepairEntropyCoderRuntimeAdapterError("ANS prototype stream missing state")
    starts, decode_table = _ans_model_tables(freqs)
    state = struct.unpack_from("<I", encoded, 0)[0]
    cursor = 4
    output = bytearray()
    mask = ANS_TOTAL_FREQ - 1
    for _index in range(output_len):
        slot = state & mask
        symbol = decode_table[slot]
        if symbol < 0:
            raise RepairEntropyCoderRuntimeAdapterError("ANS prototype decode slot invalid")
        freq = int(freqs[symbol])
        state = freq * (state >> ANS_SCALE_BITS) + (slot - starts[symbol])
        while state < ANS_BYTE_L and cursor < len(encoded):
            state = (state << 8) | encoded[cursor]
            cursor += 1
        output.append(symbol)
    if cursor != len(encoded):
        raise RepairEntropyCoderRuntimeAdapterError("ANS prototype stream has trailing bytes")
    return bytes(output)


def ans_rans_prototype_encode(payload: bytes) -> bytes:
    """Encode a deterministic rANS prototype packet."""

    freqs = _normalise_ans_frequencies(payload)
    encoded = _ans_rans_encode(payload, freqs)
    entries = b"".join(
        struct.pack("<BH", symbol, freq)
        for symbol, freq in sorted(freqs.items())
    )
    return (
        ANS_CODER_PROTOTYPE_MAGIC
        + struct.pack("<BQH", 1, len(payload), len(freqs))
        + entries
        + bytes.fromhex(sha256_bytes(payload))
        + struct.pack("<I", len(encoded))
        + encoded
    )


def ans_rans_prototype_decode(packet: bytes) -> bytes:
    """Decode a deterministic rANS prototype packet."""

    fixed_header = len(ANS_CODER_PROTOTYPE_MAGIC) + struct.calcsize("<BQH")
    if len(packet) < fixed_header or not packet.startswith(ANS_CODER_PROTOTYPE_MAGIC):
        raise RepairEntropyCoderRuntimeAdapterError("ANS prototype packet has invalid magic")
    offset = len(ANS_CODER_PROTOTYPE_MAGIC)
    version, original_len, entry_count = struct.unpack_from("<BQH", packet, offset)
    offset += struct.calcsize("<BQH")
    if version != 1:
        raise RepairEntropyCoderRuntimeAdapterError("ANS prototype packet version unsupported")
    freqs: dict[int, int] = {}
    for _index in range(entry_count):
        if offset + 3 > len(packet):
            raise RepairEntropyCoderRuntimeAdapterError("ANS prototype frequency table truncated")
        symbol, freq = struct.unpack_from("<BH", packet, offset)
        offset += 3
        freqs[int(symbol)] = int(freq)
    if sum(freqs.values()) != ANS_TOTAL_FREQ:
        raise RepairEntropyCoderRuntimeAdapterError("ANS prototype frequencies do not sum to scale")
    if offset + 36 > len(packet):
        raise RepairEntropyCoderRuntimeAdapterError("ANS prototype packet footer truncated")
    expected_sha = packet[offset : offset + 32].hex()
    offset += 32
    (encoded_len,) = struct.unpack_from("<I", packet, offset)
    offset += 4
    encoded = packet[offset : offset + encoded_len]
    if len(encoded) != encoded_len or offset + encoded_len != len(packet):
        raise RepairEntropyCoderRuntimeAdapterError("ANS prototype encoded length mismatch")
    decoded = _ans_rans_decode(encoded, output_len=original_len, freqs=freqs)
    if sha256_bytes(decoded) != expected_sha:
        raise RepairEntropyCoderRuntimeAdapterError("ANS prototype packet decode proof failed")
    return decoded


def entropy_coder_runtime_adapter_manifest(coder_family: str) -> dict[str, Any]:
    """Return the decode-only adapter contract for a prototype coder family."""

    if coder_family == "range":
        transform_kind = "range_coder_lzma_prototype"
        decode_function = "range_lzma_prototype_decode"
        encode_function = "range_lzma_prototype_encode"
        packet_magic = RANGE_CODER_PROTOTYPE_MAGIC.hex()
    elif coder_family == "ans":
        transform_kind = "ans_coder_rans_prototype"
        decode_function = "ans_rans_prototype_decode"
        encode_function = "ans_rans_prototype_encode"
        packet_magic = ANS_CODER_PROTOTYPE_MAGIC.hex()
    else:
        raise RepairEntropyCoderRuntimeAdapterError(
            f"unsupported entropy coder runtime adapter family: {coder_family}"
        )
    return {
        "schema": REPAIR_ENTROPY_CODER_RUNTIME_ADAPTER_MANIFEST_SCHEMA,
        "coder_family": coder_family,
        "transform_kind": transform_kind,
        "module": "tac.optimization.repair_entropy_coder_runtime_adapters",
        "encode_function": encode_function,
        "decode_function": decode_function,
        "packet_magic_hex": packet_magic,
        "stdlib_only": True,
        "decode_only_receiver_contract": True,
        "scorer_state_access": False,
        "network_access": False,
        "sidecar_fetch": False,
        "runtime_adapter_ready": True,
        "contest_runtime_decoder_adapter_ready": True,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }


def decode_entropy_coder_prototype_member(
    *,
    coder_family: str,
    packet: bytes,
) -> bytes:
    """Decode a prototype member packet through the public adapter entrypoint."""

    if coder_family == "range":
        return range_lzma_prototype_decode(packet)
    if coder_family == "ans":
        return ans_rans_prototype_decode(packet)
    raise RepairEntropyCoderRuntimeAdapterError(
        f"unsupported entropy coder runtime adapter family: {coder_family}"
    )


__all__ = [
    "ANS_BYTE_L",
    "ANS_CODER_PROTOTYPE_MAGIC",
    "ANS_SCALE_BITS",
    "ANS_TOTAL_FREQ",
    "RANGE_CODER_PROTOTYPE_MAGIC",
    "REPAIR_ENTROPY_CODER_RUNTIME_ADAPTER_MANIFEST_SCHEMA",
    "RepairEntropyCoderRuntimeAdapterError",
    "ans_rans_prototype_decode",
    "ans_rans_prototype_encode",
    "decode_entropy_coder_prototype_member",
    "entropy_coder_runtime_adapter_manifest",
    "range_lzma_prototype_decode",
    "range_lzma_prototype_encode",
]

