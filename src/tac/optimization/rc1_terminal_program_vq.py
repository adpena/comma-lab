"""Counted terminal temporal-program vector quantization for RC1.

This module is deliberately scorer-free.  It learns a dictionary of complete
categorical time trajectories and a spatial assignment lattice, serializes both
with real lossless coders, and reconstructs the resulting token tensor through
an independent receiver.  The learned codebook and assignment map are
video-derived and therefore counted bytes.

The representation is lossy with respect to the source token tensor unless the
codebook contains every distinct trajectory.  Token Hamming error is a
diagnostic only; it is not a SegNet or contest-score proxy.
"""

from __future__ import annotations

import binascii
import hashlib
import lzma
import struct
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import brotli
import numpy as np

NUM_CLASSES = 5
PAYLOAD_MAGIC = b"RC1V"
PAYLOAD_VERSION = 1
PAYLOAD_HEADER = struct.Struct("<4sBBBBHHHHIIII32s")

SHADOW_MAGIC = b"RC1A"
SHADOW_VERSION = 1
SHADOW_HEADER = struct.Struct("<4sBBHHHIIII32s")

RX1_MODEL_HEADER = struct.Struct("<4sBBBBHHH")
RX1_MAGIC = b"RX1M"
DX2_ARCHIVE_SHA256 = "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
DX2_TOKEN_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"  # gitleaks:allow -- public content digest
DX2_TOKEN_BYTES = 117_964_800
DX2_TOKEN_STREAM_BYTES = 113_777
DX2_HPAC_STREAM_BYTES = 13_515
DX2_SEMANTIC_STREAM_BYTES = 30_856
DX2_CARRIER_STREAM_BYTES = 22_010
DX2_RESIDUAL_STREAM_BYTES = 96
DX2_ARCHIVE_BYTES = 180_368
STRICT_SUB012_ARCHIVE_BYTES = 137_986

LZMA_FILTERS = [
    {
        "id": lzma.FILTER_LZMA1,
        "dict_size": 1 << 16,
        "lc": 3,
        "lp": 0,
        "pb": 0,
        "mode": lzma.MODE_NORMAL,
        "nice_len": 273,
        "mf": lzma.MF_BT4,
        "depth": 0,
    }
]


class RC1FormatError(ValueError):
    """Raised when an RC1 research payload is malformed or non-canonical."""


@dataclass(frozen=True)
class EncodedVariant:
    """One retained real-coder stream."""

    method_id: int
    name: str
    payload: bytes
    raw_bytes: int


@dataclass(frozen=True)
class TokenVQModel:
    """A categorical temporal codebook and its spatial assignment map."""

    assignments: np.ndarray
    codebook: np.ndarray

    def validate(self) -> None:
        assignments = np.asarray(self.assignments)
        codebook = np.asarray(self.codebook)
        if assignments.ndim != 2 or assignments.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
            raise RC1FormatError("assignments must be a two-dimensional uint8/uint16 array")
        if codebook.ndim != 2 or codebook.dtype != np.uint8:
            raise RC1FormatError("codebook must be a two-dimensional uint8 array")
        if not 1 <= codebook.shape[0] <= 65_535:
            raise RC1FormatError("codebook cardinality must be in [1, 65535]")
        if np.any(codebook >= NUM_CLASSES):
            raise RC1FormatError("codebook contains a class outside the five-class alphabet")
        if assignments.size and int(assignments.max()) >= codebook.shape[0]:
            raise RC1FormatError("assignment references a missing codeword")


@dataclass(frozen=True)
class ShadowSections:
    """Physical non-token streams copied from the exact DX2 archive."""

    semantic: bytes
    carrier: bytes
    residual: bytes
    source_archive_sha256: str


def sha256_bytes(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _uleb_encode(value: int) -> bytes:
    if value < 0:
        raise ValueError("ULEB128 only accepts nonnegative integers")
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(output)


def _uleb_decode(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(payload) or shift > 63:
            raise RC1FormatError("truncated or overlong ULEB128 field")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if _uleb_encode(value) != payload[offset - ((shift // 7) + 1) : offset]:
                raise RC1FormatError("non-canonical ULEB128 field")
            return value, offset
        shift += 7


def canonicalize_model(model: TokenVQModel) -> TokenVQModel:
    """Sort and deduplicate codewords, remapping assignments canonically."""
    model.validate()
    codebook = np.ascontiguousarray(model.codebook)
    keys = [bytes(row) for row in codebook]
    unique_keys = sorted(set(keys))
    new_index = {key: index for index, key in enumerate(unique_keys)}
    assignment_dtype = np.uint8 if len(unique_keys) <= 256 else np.uint16
    remap = np.asarray([new_index[key] for key in keys], dtype=assignment_dtype)
    assignments = remap[np.asarray(model.assignments, dtype=np.uint16)]
    canonical = TokenVQModel(
        assignments=np.ascontiguousarray(assignments, dtype=assignment_dtype),
        codebook=np.frombuffer(b"".join(unique_keys), dtype=np.uint8)
        .reshape(len(unique_keys), codebook.shape[1])
        .copy(),
    )
    canonical.validate()
    return canonical


def assign_programs(
    programs: np.ndarray,
    codebook: np.ndarray,
    *,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each unique program to the nearest codeword in Hamming distance."""
    programs = np.asarray(programs, dtype=np.uint8)
    codebook = np.asarray(codebook, dtype=np.uint8)
    if programs.ndim != 2 or codebook.ndim != 2 or programs.shape[1] != codebook.shape[1]:
        raise ValueError("program and codebook shapes differ")
    assignments = np.empty(programs.shape[0], dtype=np.uint16)
    distances = np.empty(programs.shape[0], dtype=np.int32)
    for start in range(0, len(programs), chunk_size):
        stop = min(start + chunk_size, len(programs))
        batch = programs[start:stop]
        costs = np.count_nonzero(batch[:, None, :] != codebook[None, :, :], axis=2)
        selected = np.argmin(costs, axis=1)
        assignments[start:stop] = selected
        distances[start:stop] = costs[np.arange(stop - start), selected]
    return assignments, distances


def _initial_codebook(programs: np.ndarray, counts: np.ndarray, k: int) -> np.ndarray:
    if not 1 <= k <= min(65_535, len(programs)):
        raise ValueError("invalid codebook size")
    time_steps = programs.shape[1]
    selected: list[int] = []
    lookup = {bytes(row): index for index, row in enumerate(programs)}
    for class_id in range(NUM_CLASSES):
        index = lookup.get(bytes([class_id]) * time_steps)
        if index is not None and index not in selected and len(selected) < k:
            selected.append(index)
    ranked = sorted(range(len(programs)), key=lambda index: (-int(counts[index]), bytes(programs[index])))
    for index in ranked:
        if index not in selected:
            selected.append(index)
        if len(selected) == k:
            break
    return np.ascontiguousarray(programs[np.asarray(selected)], dtype=np.uint8)


def fit_weighted_k_modes(
    programs: np.ndarray,
    counts: np.ndarray,
    k: int,
    *,
    iterations: int = 3,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit deterministic weighted categorical k-modes on unique trajectories."""
    programs = np.ascontiguousarray(programs, dtype=np.uint8)
    counts = np.asarray(counts, dtype=np.int64)
    if programs.ndim != 2 or counts.shape != (len(programs),):
        raise ValueError("unique-program inputs have inconsistent shapes")
    if np.any(programs >= NUM_CLASSES) or np.any(counts <= 0):
        raise ValueError("unique-program inputs violate their domains")
    codebook = _initial_codebook(programs, counts, k)
    trajectory: list[dict[str, int]] = []
    assignments = np.zeros(len(programs), dtype=np.uint16)
    distances = np.zeros(len(programs), dtype=np.int32)
    for iteration in range(iterations):
        assignments, distances = assign_programs(programs, codebook, chunk_size=chunk_size)
        weighted_error = int(np.dot(distances.astype(np.int64), counts))
        trajectory.append(
            {
                "iteration": iteration,
                "weighted_token_mismatches": weighted_error,
                "active_clusters": int(np.unique(assignments).size),
            }
        )
        updated = codebook.copy()
        for cluster in range(k):
            members = np.flatnonzero(assignments == cluster)
            if not len(members):
                continue
            member_programs = programs[members]
            member_counts = counts[members]
            scores = np.empty((NUM_CLASSES, programs.shape[1]), dtype=np.int64)
            for class_id in range(NUM_CLASSES):
                scores[class_id] = np.sum(
                    (member_programs == class_id) * member_counts[:, None],
                    axis=0,
                    dtype=np.int64,
                )
            updated[cluster] = np.argmax(scores, axis=0).astype(np.uint8)
        if np.array_equal(updated, codebook):
            break
        codebook = updated
    assignments, distances = assign_programs(programs, codebook, chunk_size=chunk_size)
    trajectory.append(
        {
            "iteration": len(trajectory),
            "weighted_token_mismatches": int(np.dot(distances.astype(np.int64), counts)),
            "active_clusters": int(np.unique(assignments).size),
        }
    )
    return codebook, assignments, {"trajectory": trajectory}


def fit_nested_debt_k_modes(
    programs: np.ndarray,
    counts: np.ndarray,
    k: int,
    *,
    iterations: int = 3,
    base_k: int = 256,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit k-modes, then spend larger codebooks on weighted residual debt.

    For ``k <= base_k`` this is ordinary weighted k-modes. Larger codebooks
    retain that fitted basis and add exact temporal programs in descending
    ``population * remaining Hamming distance`` order. The construction is
    nested and deterministic; it does not claim a global large-K optimum.
    """
    programs = np.ascontiguousarray(programs, dtype=np.uint8)
    counts = np.asarray(counts, dtype=np.int64)
    if not 1 <= k <= min(65_535, len(programs)):
        raise ValueError("invalid nested codebook size")
    fitted_k = min(k, base_k)
    base_codebook, assignments, fit = fit_weighted_k_modes(
        programs,
        counts,
        fitted_k,
        iterations=iterations,
        chunk_size=chunk_size,
    )
    if k == fitted_k:
        fit["large_k_policy"] = "ordinary_weighted_k_modes"
        return base_codebook, assignments, fit
    _, distances = assign_programs(programs, base_codebook, chunk_size=chunk_size)
    existing = {bytes(row) for row in base_codebook}
    candidates = [index for index, row in enumerate(programs) if bytes(row) not in existing]
    ranked = sorted(
        candidates,
        key=lambda index: (
            -int(distances[index]) * int(counts[index]),
            -int(counts[index]),
            bytes(programs[index]),
        ),
    )
    selected = np.asarray(ranked[: k - fitted_k], dtype=np.int64)
    codebook = np.concatenate((base_codebook, programs[selected]), axis=0)
    expanded_assignments = assignments.astype(np.uint16)
    expanded_assignments[selected] = np.arange(fitted_k, k, dtype=np.uint16)
    before = int(np.dot(distances.astype(np.int64), counts))
    after_distances = distances.copy()
    after_distances[selected] = 0
    after = int(np.dot(after_distances.astype(np.int64), counts))
    fit.update(
        {
            "large_k_policy": "nested_weighted_residual_debt_exact_programs",
            "base_k": fitted_k,
            "added_exact_programs": len(selected),
            "weighted_token_mismatches_before_expansion": before,
            "weighted_token_mismatches_after_expansion": after,
            "weighted_token_mismatches_removed": before - after,
            "global_reassignment_to_added_programs": False,
        }
    )
    return codebook, expanded_assignments, fit


def _assignment_raw_forms(assignments: np.ndarray, k: int) -> dict[str, bytes]:
    assignments = np.asarray(assignments)
    if assignments.ndim != 2 or np.any(assignments >= k):
        raise ValueError("assignment lattice violates its declared codebook")
    storage_dtype = np.dtype(np.uint8) if k <= 256 else np.dtype("<u2")
    assignments = np.asarray(assignments, dtype=storage_dtype)
    height, width = assignments.shape
    serpentine = assignments.copy()
    serpentine[1::2] = serpentine[1::2, ::-1]
    row_delta = assignments.astype(np.int32)
    row_delta[:, 1:] = (
        assignments[:, 1:].astype(np.int32) - assignments[:, :-1].astype(np.int32)
    ) % k
    serp_flat = serpentine.reshape(-1).astype(np.int32)
    serp_delta = serp_flat.copy()
    serp_delta[1:] = (serp_delta[1:] - serp_flat[:-1]) % k
    rle = bytearray()
    for row in assignments:
        changes = np.flatnonzero(row[1:] != row[:-1]) + 1
        starts = np.concatenate(([0], changes))
        stops = np.concatenate((changes, [width]))
        rle.extend(_uleb_encode(len(starts)))
        for start, stop in zip(starts, stops, strict=True):
            rle.extend(_uleb_encode(int(stop - start)))
            rle.extend(_uleb_encode(int(row[start])))
    return {
        "row": assignments.astype(storage_dtype).tobytes(),
        "serpentine": serpentine.astype(storage_dtype).tobytes(),
        "row_delta": row_delta.astype(storage_dtype).tobytes(),
        "serpentine_delta": serp_delta.astype(storage_dtype).tobytes(),
        "row_rle": bytes(rle),
    }


def _decode_assignment_raw(name: str, raw: bytes, height: int, width: int, k: int) -> np.ndarray:
    count = height * width
    storage_dtype = np.dtype(np.uint8) if k <= 256 else np.dtype("<u2")
    output_dtype = np.uint8 if k <= 256 else np.uint16
    if name in {"row", "serpentine", "row_delta", "serpentine_delta"}:
        if len(raw) != count * storage_dtype.itemsize:
            raise RC1FormatError("assignment raw length differs")
        values = np.frombuffer(raw, dtype=storage_dtype).copy()
        if name == "row":
            output = values.reshape(height, width)
        elif name == "serpentine":
            output = values.reshape(height, width)
            output[1::2] = output[1::2, ::-1]
        elif name == "row_delta":
            output = values.reshape(height, width).astype(np.uint16)
            output = np.mod(np.cumsum(output, axis=1, dtype=np.uint64), k).astype(output_dtype)
        else:
            output = np.mod(np.cumsum(values.astype(np.uint64)), k).astype(output_dtype).reshape(height, width)
            output[1::2] = output[1::2, ::-1]
    elif name == "row_rle":
        output = np.empty((height, width), dtype=output_dtype)
        offset = 0
        for y in range(height):
            runs, offset = _uleb_decode(raw, offset)
            x = 0
            for _ in range(runs):
                length, offset = _uleb_decode(raw, offset)
                if length <= 0:
                    raise RC1FormatError("invalid assignment RLE run")
                value, offset = _uleb_decode(raw, offset)
                if value >= k or x + length > width:
                    raise RC1FormatError("assignment RLE run exceeds its domain")
                output[y, x : x + length] = value
                x += length
            if x != width:
                raise RC1FormatError("assignment RLE row has wrong width")
        if offset != len(raw):
            raise RC1FormatError("assignment RLE has trailing bytes")
    else:
        raise RC1FormatError(f"unknown assignment transform {name}")
    if output.size and int(output.max()) >= k:
        raise RC1FormatError("decoded assignment references a missing codeword")
    return np.ascontiguousarray(output, dtype=output_dtype)


def _codebook_raw_forms(codebook: np.ndarray) -> dict[str, bytes]:
    codebook = np.asarray(codebook, dtype=np.uint8)
    k, time_steps = codebook.shape
    delta = codebook.copy()
    delta[:, 1:] = (codebook[:, 1:].astype(np.int16) - codebook[:, :-1].astype(np.int16)) % NUM_CLASSES
    events = bytearray()
    for row in codebook:
        changes = np.flatnonzero(row[1:] != row[:-1]) + 1
        events.append(int(row[0]))
        events.extend(_uleb_encode(len(changes)))
        previous = 0
        for position in changes:
            events.extend(_uleb_encode(int(position - previous)))
            events.append(int(row[position]))
            previous = int(position)
    return {
        "row": codebook.tobytes(),
        "time_major": np.ascontiguousarray(codebook.T).tobytes(),
        "temporal_delta": delta.tobytes(),
        "events": bytes(events),
    }


def _decode_codebook_raw(name: str, raw: bytes, k: int, time_steps: int) -> np.ndarray:
    count = k * time_steps
    if name in {"row", "time_major", "temporal_delta"}:
        if len(raw) != count:
            raise RC1FormatError("codebook raw length differs")
        values = np.frombuffer(raw, dtype=np.uint8).copy()
        if name == "row":
            output = values.reshape(k, time_steps)
        elif name == "time_major":
            output = np.ascontiguousarray(values.reshape(time_steps, k).T)
        else:
            delta = values.reshape(k, time_steps).astype(np.uint16)
            output = np.mod(np.cumsum(delta, axis=1, dtype=np.uint32), NUM_CLASSES).astype(np.uint8)
    elif name == "events":
        output = np.empty((k, time_steps), dtype=np.uint8)
        offset = 0
        for index in range(k):
            if offset >= len(raw) or raw[offset] >= NUM_CLASSES:
                raise RC1FormatError("invalid codebook event initial class")
            value = raw[offset]
            offset += 1
            transitions, offset = _uleb_decode(raw, offset)
            cursor = 0
            output[index].fill(value)
            for _ in range(transitions):
                delta, offset = _uleb_decode(raw, offset)
                cursor += delta
                if delta <= 0 or cursor >= time_steps or offset >= len(raw):
                    raise RC1FormatError("invalid codebook transition")
                value = raw[offset]
                offset += 1
                if value >= NUM_CLASSES:
                    raise RC1FormatError("codebook transition class is invalid")
                output[index, cursor:] = value
        if offset != len(raw):
            raise RC1FormatError("codebook events have trailing bytes")
    else:
        raise RC1FormatError(f"unknown codebook transform {name}")
    if np.any(output >= NUM_CLASSES):
        raise RC1FormatError("decoded codebook contains an invalid class")
    return np.ascontiguousarray(output, dtype=np.uint8)


def _compress_variants(raw_forms: dict[str, bytes], base_method: int) -> list[EncodedVariant]:
    variants: list[EncodedVariant] = []
    method = base_method
    for transform, raw in raw_forms.items():
        for coder in ("brotli_q11", "lzma1_raw", "zlib9"):
            if coder == "brotli_q11":
                payload = brotli.compress(raw, quality=11)
            elif coder == "lzma1_raw":
                payload = lzma.compress(raw, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
            else:
                payload = zlib.compress(raw, level=9)
            variants.append(
                EncodedVariant(
                    method_id=method,
                    name=f"{transform}__{coder}",
                    payload=payload,
                    raw_bytes=len(raw),
                )
            )
            method += 1
    return variants


ASSIGNMENT_METHOD_NAMES = {
    variant.method_id: variant.name
    for variant in _compress_variants(
        dict.fromkeys(("row", "serpentine", "row_delta", "serpentine_delta", "row_rle"), b""),
        1,
    )
}
CODEBOOK_METHOD_NAMES = {
    variant.method_id: variant.name
    for variant in _compress_variants(
        dict.fromkeys(("row", "time_major", "temporal_delta", "events"), b""),
        101,
    )
}


def encode_assignment_variants(assignments: np.ndarray, k: int) -> list[EncodedVariant]:
    return _compress_variants(_assignment_raw_forms(assignments, k), 1)


def encode_codebook_variants(codebook: np.ndarray) -> list[EncodedVariant]:
    return _compress_variants(_codebook_raw_forms(codebook), 101)


def _decompress_method(name: str, payload: bytes) -> bytes:
    if name.endswith("__brotli_q11"):
        return brotli.decompress(payload)
    if name.endswith("__lzma1_raw"):
        try:
            return lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
        except lzma.LZMAError as error:
            raise RC1FormatError("invalid raw LZMA1 stream") from error
    if name.endswith("__zlib9"):
        try:
            return zlib.decompress(payload)
        except zlib.error as error:
            raise RC1FormatError("invalid zlib stream") from error
    raise RC1FormatError(f"unknown coder in method {name}")


def build_payload(
    model: TokenVQModel,
    assignment_variant: EncodedVariant,
    codebook_variant: EncodedVariant,
    decoded_token_sha256: str,
) -> bytes:
    """Build a canonical counted RC1 token-description payload."""
    model = canonicalize_model(model)
    height, width = model.assignments.shape
    k, time_steps = model.codebook.shape
    if assignment_variant.method_id not in ASSIGNMENT_METHOD_NAMES:
        raise ValueError("unknown assignment method")
    if codebook_variant.method_id not in CODEBOOK_METHOD_NAMES:
        raise ValueError("unknown codebook method")
    decoded_digest = bytes.fromhex(decoded_token_sha256)
    if len(decoded_digest) != 32:
        raise ValueError("decoded token SHA-256 must be 32 bytes")
    assignment_crc = binascii.crc32(assignment_variant.payload) & 0xFFFFFFFF
    codebook_crc = binascii.crc32(codebook_variant.payload) & 0xFFFFFFFF
    flags = 1 if k > 256 else 0
    header = PAYLOAD_HEADER.pack(
        PAYLOAD_MAGIC,
        PAYLOAD_VERSION,
        assignment_variant.method_id,
        codebook_variant.method_id,
        flags,
        height,
        width,
        time_steps,
        k,
        len(assignment_variant.payload),
        len(codebook_variant.payload),
        assignment_crc,
        codebook_crc,
        decoded_digest,
    )
    return header + assignment_variant.payload + codebook_variant.payload


def parse_payload(payload: bytes) -> tuple[TokenVQModel, str]:
    """Parse an RC1 token-description payload and reconstruct its learned state."""
    if len(payload) < PAYLOAD_HEADER.size:
        raise RC1FormatError("RC1 payload is truncated")
    (
        magic,
        version,
        assignment_method,
        codebook_method,
        flags,
        height,
        width,
        time_steps,
        k,
        assignment_bytes,
        codebook_bytes,
        assignment_crc,
        codebook_crc,
        decoded_digest,
    ) = PAYLOAD_HEADER.unpack_from(payload)
    if magic != PAYLOAD_MAGIC or version != PAYLOAD_VERSION or flags not in (0, 1):
        raise RC1FormatError("unsupported RC1 payload header")
    if min(height, width, time_steps, k, assignment_bytes, codebook_bytes) <= 0 or k > 65_535:
        raise RC1FormatError("RC1 payload header violates its domain")
    if flags != (1 if k > 256 else 0):
        raise RC1FormatError("RC1 assignment-width flag differs from codebook size")
    expected = PAYLOAD_HEADER.size + assignment_bytes + codebook_bytes
    if len(payload) != expected:
        raise RC1FormatError("RC1 payload length accounting differs")
    assignment_stream = payload[PAYLOAD_HEADER.size : PAYLOAD_HEADER.size + assignment_bytes]
    codebook_stream = payload[PAYLOAD_HEADER.size + assignment_bytes :]
    if binascii.crc32(assignment_stream) & 0xFFFFFFFF != assignment_crc:
        raise RC1FormatError("RC1 assignment stream CRC differs")
    if binascii.crc32(codebook_stream) & 0xFFFFFFFF != codebook_crc:
        raise RC1FormatError("RC1 codebook stream CRC differs")
    assignment_name = ASSIGNMENT_METHOD_NAMES.get(assignment_method)
    codebook_name = CODEBOOK_METHOD_NAMES.get(codebook_method)
    if assignment_name is None or codebook_name is None:
        raise RC1FormatError("RC1 stream method is unknown")
    assignment_raw = _decompress_method(assignment_name, assignment_stream)
    codebook_raw = _decompress_method(codebook_name, codebook_stream)
    assignments = _decode_assignment_raw(assignment_name.split("__", 1)[0], assignment_raw, height, width, k)
    codebook = _decode_codebook_raw(codebook_name.split("__", 1)[0], codebook_raw, k, time_steps)
    model = TokenVQModel(assignments=assignments, codebook=codebook)
    canonical_assignment = next(
        variant
        for variant in encode_assignment_variants(model.assignments, k)
        if variant.method_id == assignment_method
    )
    canonical_codebook = next(
        variant
        for variant in encode_codebook_variants(model.codebook)
        if variant.method_id == codebook_method
    )
    if build_payload(
        model,
        canonical_assignment,
        canonical_codebook,
        decoded_digest.hex(),
    ) != payload:
        raise RC1FormatError("RC1 payload is not canonical")
    return model, decoded_digest.hex()


def iter_decoded_frames(model: TokenVQModel) -> Iterable[np.ndarray]:
    """Yield receiver-expanded token frames without materializing the full tensor."""
    model.validate()
    flat = model.assignments.reshape(-1)
    for time_index in range(model.codebook.shape[1]):
        yield model.codebook[flat, time_index].reshape(model.assignments.shape)


def decoded_sha256(model: TokenVQModel) -> str:
    digest = hashlib.sha256()
    for frame in iter_decoded_frames(model):
        digest.update(memoryview(np.ascontiguousarray(frame)))
    return digest.hexdigest()


def extract_dx2_shadow_sections(archive_path: Path) -> ShadowSections:
    """Read the exact physical semantic/carrier/residual streams from DX2."""
    archive_bytes = archive_path.read_bytes()
    digest = sha256_bytes(archive_bytes)
    if len(archive_bytes) != DX2_ARCHIVE_BYTES or digest != DX2_ARCHIVE_SHA256:
        raise RC1FormatError("DX2 archive custody pin differs")
    import zipfile

    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["p"]:
            raise RC1FormatError("DX2 archive must contain exactly member p")
        outer = archive.read("p")
    if len(outer) < RX1_MODEL_HEADER.size:
        raise RC1FormatError("DX2 member is truncated")
    magic, version, _codec, _table_mode, _reserved, hpac_bytes, semantic_bytes, carrier_bytes = RX1_MODEL_HEADER.unpack_from(outer)
    if magic != RX1_MAGIC or version != 1:
        raise RC1FormatError("DX2 member is not RX1 v1")
    offset = RX1_MODEL_HEADER.size
    hpac = outer[offset : offset + hpac_bytes]
    offset += hpac_bytes
    semantic = outer[offset : offset + semantic_bytes]
    offset += semantic_bytes
    carrier = outer[offset : offset + carrier_bytes]
    offset += carrier_bytes
    residual = outer[offset : offset + DX2_RESIDUAL_STREAM_BYTES]
    token = outer[offset + DX2_RESIDUAL_STREAM_BYTES :]
    if (
        len(hpac) != DX2_HPAC_STREAM_BYTES
        or len(semantic) != DX2_SEMANTIC_STREAM_BYTES
        or len(carrier) != DX2_CARRIER_STREAM_BYTES
        or len(residual) != DX2_RESIDUAL_STREAM_BYTES
        or len(token) != DX2_TOKEN_STREAM_BYTES
        or offset + DX2_RESIDUAL_STREAM_BYTES + len(token) != len(outer)
    ):
        raise RC1FormatError("DX2 physical stream anatomy differs")
    return ShadowSections(semantic, carrier, residual, digest)


def build_shadow_outer(sections: ShadowSections, rc1_payload: bytes) -> bytes:
    """Build a research-only complete member with HPAC+RC64 replaced by RC1."""
    source_digest = bytes.fromhex(sections.source_archive_sha256)
    header = SHADOW_HEADER.pack(
        SHADOW_MAGIC,
        SHADOW_VERSION,
        0,
        len(sections.semantic),
        len(sections.carrier),
        len(sections.residual),
        len(rc1_payload),
        binascii.crc32(sections.semantic) & 0xFFFFFFFF,
        binascii.crc32(sections.carrier) & 0xFFFFFFFF,
        binascii.crc32(sections.residual) & 0xFFFFFFFF,
        source_digest,
    )
    return header + sections.semantic + sections.carrier + sections.residual + rc1_payload


def parse_shadow_outer(outer: bytes) -> tuple[ShadowSections, TokenVQModel, str]:
    """Strictly parse the research RC1 shadow member and its active token payload."""
    if len(outer) < SHADOW_HEADER.size:
        raise RC1FormatError("RC1 shadow member is truncated")
    (
        magic,
        version,
        flags,
        semantic_bytes,
        carrier_bytes,
        residual_bytes,
        rc1_bytes,
        semantic_crc,
        carrier_crc,
        residual_crc,
        source_digest,
    ) = SHADOW_HEADER.unpack_from(outer)
    if magic != SHADOW_MAGIC or version != SHADOW_VERSION or flags != 0:
        raise RC1FormatError("unsupported RC1 shadow header")
    if (semantic_bytes, carrier_bytes, residual_bytes) != (
        DX2_SEMANTIC_STREAM_BYTES,
        DX2_CARRIER_STREAM_BYTES,
        DX2_RESIDUAL_STREAM_BYTES,
    ):
        raise RC1FormatError("RC1 shadow copied-stream lengths differ")
    if len(outer) != SHADOW_HEADER.size + semantic_bytes + carrier_bytes + residual_bytes + rc1_bytes:
        raise RC1FormatError("RC1 shadow length accounting differs")
    offset = SHADOW_HEADER.size
    semantic = outer[offset : offset + semantic_bytes]
    offset += semantic_bytes
    carrier = outer[offset : offset + carrier_bytes]
    offset += carrier_bytes
    residual = outer[offset : offset + residual_bytes]
    offset += residual_bytes
    if binascii.crc32(semantic) & 0xFFFFFFFF != semantic_crc:
        raise RC1FormatError("RC1 shadow semantic CRC differs")
    if binascii.crc32(carrier) & 0xFFFFFFFF != carrier_crc:
        raise RC1FormatError("RC1 shadow carrier CRC differs")
    if binascii.crc32(residual) & 0xFFFFFFFF != residual_crc:
        raise RC1FormatError("RC1 shadow residual CRC differs")
    model, decoded_digest = parse_payload(outer[offset:])
    return ShadowSections(semantic, carrier, residual, source_digest.hex()), model, decoded_digest
