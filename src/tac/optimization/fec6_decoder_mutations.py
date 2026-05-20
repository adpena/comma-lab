# SPDX-License-Identifier: MIT
"""Legal q-symbol mutation helpers for PR101/FEC6 decoder blobs."""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli
import numpy as np

from tac.optimization.fec6_byte_targets import ByteRange, parse_fec6_sections
from tac.pr101_split_brotli_codec import (
    DECODER_BLOB_LEN,
    DECODER_BYTE_MAPS,
    DECODER_STORAGE_ORDER,
    DECODER_STREAM_ENDS,
    FIXED_STATE_SCHEMA,
    LATENT_BLOB_LEN,
    _encode_mapped_u8,
    decode_mapped_u8,
    pack_brotli_stream,
)


class Fec6DecoderMutationError(ValueError):
    """Raised when a decoder mutation cannot be represented safely."""


@dataclass(frozen=True)
class BrotliStreamSpan:
    """Compressed and decompressed spans for one split-Brotli stream."""

    stream_index: int
    compressed_range: ByteRange
    raw_range: ByteRange

    def as_dict(self) -> dict[str, Any]:
        return {
            "stream_index": self.stream_index,
            "compressed_range": self.compressed_range.as_dict(),
            "raw_range": self.raw_range.as_dict(),
        }


@dataclass(frozen=True)
class DecoderQTensorSpan:
    """Raw q-symbol span for one decoder tensor inside split-Brotli raw bytes."""

    name: str
    storage_index: int
    storage_position: int
    shape: tuple[int, ...]
    numel: int
    byte_map: str
    stream_index: int
    raw_q_range: ByteRange
    raw_scale_range: ByteRange

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "storage_index": self.storage_index,
            "storage_position": self.storage_position,
            "shape": list(self.shape),
            "numel": self.numel,
            "byte_map": self.byte_map,
            "stream_index": self.stream_index,
            "raw_q_range": self.raw_q_range.as_dict(),
            "raw_scale_range": self.raw_scale_range.as_dict(),
        }


@dataclass(frozen=True)
class DecoderQMutation:
    """A single legal q-domain mutation before Brotli recompression."""

    tensor_name: str
    q_offset: int
    delta: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "tensor_name": self.tensor_name,
            "q_offset": self.q_offset,
            "delta": self.delta,
        }


@dataclass(frozen=True)
class DecoderMutationResult:
    """Result row for a recompressed decoder mutation."""

    mutation: DecoderQMutation
    tensor: DecoderQTensorSpan
    q_before: int
    q_after: int
    source_decoder_len: int
    mutated_decoder_len: int
    source_decoder_sha256: str
    mutated_decoder_sha256: str
    fixed_length_runtime_compatible: bool

    @property
    def length_delta(self) -> int:
        return self.mutated_decoder_len - self.source_decoder_len

    @property
    def mutation_id(self) -> str:
        seed = (
            f"{self.mutation.tensor_name}:{self.mutation.q_offset}:"
            f"{self.mutation.delta}:{self.q_before}:{self.q_after}:"
            f"{self.mutated_decoder_sha256}"
        )
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "mutation": self.mutation.as_dict(),
            "tensor": self.tensor.as_dict(),
            "q_before": self.q_before,
            "q_after": self.q_after,
            "source_decoder_len": self.source_decoder_len,
            "mutated_decoder_len": self.mutated_decoder_len,
            "length_delta": self.length_delta,
            "source_decoder_sha256": self.source_decoder_sha256,
            "mutated_decoder_sha256": self.mutated_decoder_sha256,
            "fixed_length_runtime_compatible": self.fixed_length_runtime_compatible,
        }


@dataclass(frozen=True)
class PreparedDecoderBlob:
    """Decoded split-Brotli state needed for many q-symbol probes."""

    decoder_blob: bytes
    raw: bytes
    stream_spans: tuple[BrotliStreamSpan, ...]
    tensor_spans: tuple[DecoderQTensorSpan, ...]

    @property
    def decoder_sha256(self) -> str:
        return hashlib.sha256(self.decoder_blob).hexdigest()

    @property
    def raw_sha256(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()

    def tensor_by_name(self) -> dict[str, DecoderQTensorSpan]:
        return {span.name: span for span in self.tensor_spans}

    def as_dict(self) -> dict[str, Any]:
        return {
            "decoder_len": len(self.decoder_blob),
            "decoder_sha256": self.decoder_sha256,
            "raw_len": len(self.raw),
            "raw_sha256": self.raw_sha256,
            "stream_spans": [span.as_dict() for span in self.stream_spans],
            "tensor_spans": [span.as_dict() for span in self.tensor_spans],
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_brotli_streams(data: bytes, n_streams: int) -> tuple[tuple[bytes, ...], tuple[ByteRange, ...]]:
    """Decompress concatenated Brotli streams and preserve compressed spans."""

    outputs: list[bytes] = []
    compressed_ranges: list[ByteRange] = []
    pos = 0
    for _stream_index in range(n_streams):
        compressed_start = pos
        dec = brotli.Decompressor()
        chunks: list[bytes] = []
        while pos < len(data) and not dec.is_finished():
            chunks.append(dec.process(data[pos : pos + 1]))
            pos += 1
        if not dec.is_finished():
            raise Fec6DecoderMutationError("truncated split-Brotli decoder payload")
        outputs.append(b"".join(chunks))
        compressed_ranges.append(ByteRange(compressed_start, pos))
    if pos != len(data):
        raise Fec6DecoderMutationError("trailing split-Brotli decoder payload")
    return tuple(outputs), tuple(compressed_ranges)


def decoder_q_tensor_spans(stream_raw_lengths: Sequence[int]) -> tuple[DecoderQTensorSpan, ...]:
    """Return raw q-symbol spans for decoder tensors in global raw coordinates."""

    stream_bounds: list[ByteRange] = []
    raw_pos = 0
    for length in stream_raw_lengths:
        stream_bounds.append(ByteRange(raw_pos, raw_pos + int(length)))
        raw_pos += int(length)

    spans: list[DecoderQTensorSpan] = []
    stream_index = 0
    next_stream_end = int(DECODER_STREAM_ENDS[stream_index])
    raw_pos = 0
    for storage_position, storage_index in enumerate(DECODER_STORAGE_ORDER):
        if storage_position >= next_stream_end:
            if raw_pos != stream_bounds[stream_index].end:
                raise Fec6DecoderMutationError("raw tensor spans do not align to stream boundary")
            stream_index += 1
            next_stream_end = int(DECODER_STREAM_ENDS[stream_index])
        name, shape = FIXED_STATE_SCHEMA[int(storage_index)]
        numel = int(np.prod(shape))
        q_start = raw_pos
        q_end = q_start + numel
        scale_start = q_end
        scale_end = scale_start + 2
        if scale_end > stream_bounds[stream_index].end:
            raise Fec6DecoderMutationError("tensor span crosses split-Brotli stream boundary")
        spans.append(
            DecoderQTensorSpan(
                name=str(name),
                storage_index=int(storage_index),
                storage_position=int(storage_position),
                shape=tuple(int(v) for v in shape),
                numel=numel,
                byte_map=str(DECODER_BYTE_MAPS.get(int(storage_index), "zig")),
                stream_index=int(stream_index),
                raw_q_range=ByteRange(q_start, q_end),
                raw_scale_range=ByteRange(scale_start, scale_end),
            )
        )
        raw_pos = scale_end
    if raw_pos != stream_bounds[-1].end:
        raise Fec6DecoderMutationError("raw tensor spans did not consume decoder raw bytes")
    return tuple(spans)


def prepare_decoder_blob(decoder_blob: bytes) -> PreparedDecoderBlob:
    streams, compressed_ranges = split_brotli_streams(decoder_blob, len(DECODER_STREAM_ENDS))
    raw_parts: list[bytes] = []
    stream_spans: list[BrotliStreamSpan] = []
    raw_pos = 0
    for stream_index, (stream_raw, compressed_range) in enumerate(zip(streams, compressed_ranges, strict=True)):
        raw_parts.append(stream_raw)
        raw_range = ByteRange(raw_pos, raw_pos + len(stream_raw))
        stream_spans.append(
            BrotliStreamSpan(
                stream_index=stream_index,
                compressed_range=compressed_range,
                raw_range=raw_range,
            )
        )
        raw_pos += len(stream_raw)
    raw = b"".join(raw_parts)
    return PreparedDecoderBlob(
        decoder_blob=decoder_blob,
        raw=raw,
        stream_spans=tuple(stream_spans),
        tensor_spans=decoder_q_tensor_spans([len(part) for part in streams]),
    )


def recompress_prepared_decoder(
    prepared: PreparedDecoderBlob,
    raw: bytes,
    *,
    brotli_quality: int = 11,
    brotli_lgwin: int | None = None,
    brotli_lgblock: int | None = None,
) -> bytes:
    """Recompress mutated global raw bytes using the original split windows."""

    if len(raw) != len(prepared.raw):
        raise Fec6DecoderMutationError("mutated decoder raw length changed")
    streams: list[bytes] = []
    for span in prepared.stream_spans:
        window = raw[span.raw_range.start : span.raw_range.end]
        streams.append(
            pack_brotli_stream(
                window,
                quality=brotli_quality,
                lgwin=brotli_lgwin,
                lgblock=brotli_lgblock,
            )
        )
    return b"".join(streams)


def apply_q_mutation(prepared: PreparedDecoderBlob, mutation: DecoderQMutation) -> tuple[bytes, DecoderQTensorSpan, int, int]:
    """Apply one q-domain mutation to prepared raw bytes and return raw + q values."""

    tensors = prepared.tensor_by_name()
    if mutation.tensor_name not in tensors:
        raise Fec6DecoderMutationError(f"unknown decoder tensor: {mutation.tensor_name}")
    tensor = tensors[mutation.tensor_name]
    if mutation.q_offset < 0 or mutation.q_offset >= tensor.numel:
        raise Fec6DecoderMutationError(
            f"q_offset {mutation.q_offset} out of range for {tensor.name} numel={tensor.numel}"
        )
    q_bytes = np.frombuffer(
        prepared.raw[tensor.raw_q_range.start : tensor.raw_q_range.end],
        dtype=np.uint8,
    ).copy()
    q_values = decode_mapped_u8(q_bytes, tensor.byte_map)
    q_before = int(q_values[int(mutation.q_offset)])
    q_after = int(np.clip(q_before + int(mutation.delta), -127, 127))
    if q_after == q_before:
        raise Fec6DecoderMutationError("mutation is clipped/no-op in q-domain")
    q_values[int(mutation.q_offset)] = np.int8(q_after)
    remapped = _encode_mapped_u8(q_values, tensor.byte_map)
    raw = bytearray(prepared.raw)
    raw[tensor.raw_q_range.start : tensor.raw_q_range.end] = remapped.tobytes()
    return bytes(raw), tensor, q_before, q_after


def apply_q_mutations(
    prepared: PreparedDecoderBlob,
    mutations: Sequence[DecoderQMutation],
) -> tuple[bytes, list[dict[str, Any]]]:
    """Apply multiple q-domain mutations cumulatively to prepared raw bytes.

    This is the primitive needed for waterbucket/combinatorial candidates:
    fixed-length Brotli compatibility must be checked after the whole edit set,
    not inferred from individually valid edits.
    """

    if not mutations:
        raise Fec6DecoderMutationError("at least one q mutation is required")
    tensors = prepared.tensor_by_name()
    grouped: dict[str, list[DecoderQMutation]] = {}
    for mutation in mutations:
        grouped.setdefault(mutation.tensor_name, []).append(mutation)

    raw = bytearray(prepared.raw)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for tensor_name, tensor_mutations in grouped.items():
        if tensor_name not in tensors:
            raise Fec6DecoderMutationError(f"unknown decoder tensor: {tensor_name}")
        tensor = tensors[tensor_name]
        q_bytes = np.frombuffer(
            raw[tensor.raw_q_range.start : tensor.raw_q_range.end],
            dtype=np.uint8,
        ).copy()
        q_values = decode_mapped_u8(q_bytes, tensor.byte_map)
        for mutation in tensor_mutations:
            key = (mutation.tensor_name, int(mutation.q_offset))
            if key in seen:
                raise Fec6DecoderMutationError(
                    f"duplicate q mutation target in one candidate: {mutation.tensor_name}[{mutation.q_offset}]"
                )
            seen.add(key)
            if mutation.q_offset < 0 or mutation.q_offset >= tensor.numel:
                raise Fec6DecoderMutationError(
                    f"q_offset {mutation.q_offset} out of range for {tensor.name} numel={tensor.numel}"
                )
            q_before = int(q_values[int(mutation.q_offset)])
            q_after = int(np.clip(q_before + int(mutation.delta), -127, 127))
            if q_after == q_before:
                raise Fec6DecoderMutationError("mutation is clipped/no-op in q-domain")
            q_values[int(mutation.q_offset)] = np.int8(q_after)
            records.append(
                {
                    "mutation": mutation.as_dict(),
                    "tensor": tensor.as_dict(),
                    "q_before": q_before,
                    "q_after": q_after,
                }
            )
        remapped = _encode_mapped_u8(q_values, tensor.byte_map)
        raw[tensor.raw_q_range.start : tensor.raw_q_range.end] = remapped.tobytes()
    return bytes(raw), records


def probe_q_mutation(
    prepared: PreparedDecoderBlob,
    mutation: DecoderQMutation,
    *,
    brotli_quality: int = 11,
) -> DecoderMutationResult:
    """Apply and recompress one q-domain mutation."""

    raw, tensor, q_before, q_after = apply_q_mutation(prepared, mutation)
    mutated = recompress_prepared_decoder(prepared, raw, brotli_quality=brotli_quality)
    return DecoderMutationResult(
        mutation=mutation,
        tensor=tensor,
        q_before=q_before,
        q_after=q_after,
        source_decoder_len=len(prepared.decoder_blob),
        mutated_decoder_len=len(mutated),
        source_decoder_sha256=prepared.decoder_sha256,
        mutated_decoder_sha256=sha256_bytes(mutated),
        fixed_length_runtime_compatible=(len(mutated) == len(prepared.decoder_blob)),
    )


def extract_fec6_decoder_blob(member_bytes: bytes) -> bytes:
    sections = {
        section.name: section
        for section in parse_fec6_sections(
            member_bytes,
            decoder_blob_len=DECODER_BLOB_LEN,
            latent_blob_len=LATENT_BLOB_LEN,
        )
    }
    decoder_range = sections["decoder"].byte_range
    return member_bytes[decoder_range.start : decoder_range.end]


def replace_fec6_decoder_blob(member_bytes: bytes, decoder_blob: bytes) -> bytes:
    """Replace the fixed-length FEC6 decoder blob inside an FP11 member."""

    sections = {
        section.name: section
        for section in parse_fec6_sections(
            member_bytes,
            decoder_blob_len=DECODER_BLOB_LEN,
            latent_blob_len=LATENT_BLOB_LEN,
        )
    }
    decoder_range = sections["decoder"].byte_range
    if len(decoder_blob) != decoder_range.length:
        raise Fec6DecoderMutationError(
            f"replacement decoder len {len(decoder_blob)} != fixed section len {decoder_range.length}"
        )
    out = bytearray(member_bytes)
    out[decoder_range.start : decoder_range.end] = decoder_blob
    return bytes(out)


def write_stored_archive(archive_path: Path, member_bytes: bytes, *, member_name: str = "x") -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo(member_name)
        info.compress_type = zipfile.ZIP_STORED
        info.date_time = (1980, 1, 1, 0, 0, 0)
        zf.writestr(info, member_bytes)


def build_mutation_grid(
    tensors: Mapping[str, DecoderQTensorSpan],
    tensor_names: Sequence[str],
    *,
    deltas: Iterable[int] = (-1, 1),
    max_offsets_per_tensor: int = 512,
) -> tuple[DecoderQMutation, ...]:
    """Build a deterministic q-offset mutation grid for selected tensors."""

    rows: list[DecoderQMutation] = []
    for tensor_name in tensor_names:
        tensor = tensors[tensor_name]
        limit = min(int(max_offsets_per_tensor), int(tensor.numel))
        if limit == tensor.numel:
            offsets = list(range(tensor.numel))
        else:
            offsets = sorted({int(round(v)) for v in np.linspace(0, tensor.numel - 1, num=limit)})
        for q_offset in offsets:
            for delta in deltas:
                if int(delta) == 0:
                    continue
                rows.append(
                    DecoderQMutation(
                        tensor_name=tensor_name,
                        q_offset=int(q_offset),
                        delta=int(delta),
                    )
                )
    return tuple(rows)
