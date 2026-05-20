# SPDX-License-Identifier: MIT
"""Map score-impact byte targets onto the FEC6/PR101 payload layout."""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class Fec6ByteTargetError(ValueError):
    """Raised when FEC6 target mapping cannot be performed safely."""


@dataclass(frozen=True)
class ByteRange:
    """Half-open byte range `[start, end)`."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlap_len(self, other: "ByteRange") -> int:
        return max(0, min(self.end, other.end) - max(self.start, other.start))

    def as_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end, "length": self.length}


@dataclass(frozen=True)
class Fec6Section:
    """Named byte section in the FEC6 member payload."""

    name: str
    byte_range: ByteRange
    codec: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "range": self.byte_range.as_dict(),
            "codec": self.codec,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class DecoderTensorRange:
    """Approximate compressed-byte span for one decoder tensor."""

    name: str
    storage_index: int
    shape: tuple[int, ...]
    numel: int
    byte_map: str
    fp16_scale: float
    decoded_mantissa_range: ByteRange
    decoded_scale_range: ByteRange
    compressed_range: ByteRange

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "storage_index": self.storage_index,
            "shape": list(self.shape),
            "numel": self.numel,
            "byte_map": self.byte_map,
            "fp16_scale": self.fp16_scale,
            "decoded_mantissa_range": self.decoded_mantissa_range.as_dict(),
            "decoded_scale_range": self.decoded_scale_range.as_dict(),
            "approx_compressed_range": self.compressed_range.as_dict(),
            "mapping_authority": (
                "approximate_uniform_decompressed_to_compressed_tensor_range; "
                "do not treat as exact Brotli symbol localization"
            ),
        }


def parse_fec6_sections(member_bytes: bytes, *, decoder_blob_len: int, latent_blob_len: int) -> tuple[Fec6Section, ...]:
    """Parse the outer FP11 wrapper and inner PR101-family sections."""

    if len(member_bytes) < 10 or member_bytes[:4] != b"FP11":
        raise Fec6ByteTargetError("FEC6 member must begin with FP11 wrapper")
    source_len = struct.unpack_from("<I", member_bytes, 4)[0]
    source_offset = 8
    source_end = source_offset + source_len
    if source_len < decoder_blob_len + latent_blob_len:
        raise Fec6ByteTargetError("source payload too short for decoder+latent blobs")
    if source_end + 2 > len(member_bytes):
        raise Fec6ByteTargetError("source_len exceeds member length")
    selector_len = struct.unpack_from("<H", member_bytes, source_end)[0]
    selector_offset = source_end + 2
    selector_end = selector_offset + selector_len
    if selector_end != len(member_bytes):
        raise Fec6ByteTargetError("selector length does not consume member tail")
    if member_bytes[selector_offset : selector_offset + 4] != b"FEC6":
        raise Fec6ByteTargetError("selector payload missing FEC6 magic")
    decoder_start = source_offset
    decoder_end = decoder_start + decoder_blob_len
    latent_start = decoder_end
    latent_end = latent_start + latent_blob_len
    sidecar_start = latent_end
    sidecar_end = source_end
    return (
        Fec6Section("fp11_magic", ByteRange(0, 4), "raw_magic", "Outer wrapper marker."),
        Fec6Section("source_len_le_u32", ByteRange(4, 8), "raw_uint32_le", "Inner source payload length."),
        Fec6Section("source_payload", ByteRange(source_offset, source_end), "pr101_source_payload", "Inner PR101-family payload."),
        Fec6Section("decoder", ByteRange(decoder_start, decoder_end), "brotli_streams_int8", "Compressed decoder tensor streams."),
        Fec6Section("latent", ByteRange(latent_start, latent_end), "lzma_temporal_delta", "LZMA latent min/scale/delta stream."),
        Fec6Section("sidecar", ByteRange(sidecar_start, sidecar_end), "brotli_per_pair_corrections", "Inner per-pair correction sidecar."),
        Fec6Section("selector_len_le_u16", ByteRange(source_end, selector_offset), "raw_uint16_le", "FEC6 selector payload length."),
        Fec6Section("selector_payload", ByteRange(selector_offset, selector_end), "fec6_huffman_selector", "Discrete post-round selector payload."),
    )


def decoder_tensor_ranges(codec_module: Any, member_bytes: bytes) -> tuple[DecoderTensorRange, ...]:
    """Return approximate compressed-byte ranges for FEC6 decoder tensors."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy required for FEC6 byte-target mapping") from exc

    sections = {section.name: section for section in parse_fec6_sections(
        member_bytes,
        decoder_blob_len=int(codec_module.DECODER_BLOB_LEN),
        latent_blob_len=int(codec_module.LATENT_BLOB_LEN),
    )}
    decoder_range = sections["decoder"].byte_range
    decoder_blob = member_bytes[decoder_range.start : decoder_range.end]
    raw = codec_module.decompress_brotli_streams(
        decoder_blob,
        len(codec_module.DECODER_STREAM_ENDS),
    )
    probe = codec_module.HNeRVDecoder(
        latent_dim=codec_module.LATENT_DIM,
        base_channels=codec_module.BASE_CHANNELS,
        eval_size=codec_module.EVAL_SIZE,
    )
    items = list(probe.state_dict().items())
    ratio = int(codec_module.DECODER_BLOB_LEN) / max(len(raw), 1)
    out: list[DecoderTensorRange] = []
    pos = 0
    for storage_index in codec_module.DECODER_STORAGE_ORDER:
        name, tensor = items[int(storage_index)]
        shape = tuple(int(v) for v in tuple(tensor.shape))
        numel = int(tensor.numel())
        mantissa_start = pos
        mantissa_end = mantissa_start + numel
        scale_start = mantissa_end
        scale_end = scale_start + 2
        if scale_end > len(raw):
            raise Fec6ByteTargetError("decoder raw stream truncated while mapping tensors")
        fp16_scale = float(np.frombuffer(raw, dtype=np.float16, count=1, offset=scale_start)[0])
        compressed_start = decoder_range.start + round(mantissa_start * ratio)
        compressed_end = decoder_range.start + round(mantissa_end * ratio)
        compressed_end = max(compressed_end, compressed_start + 1)
        compressed_end = min(compressed_end, decoder_range.end)
        out.append(
            DecoderTensorRange(
                name=str(name),
                storage_index=int(storage_index),
                shape=shape,
                numel=numel,
                byte_map=str(codec_module.DECODER_BYTE_MAPS.get(int(storage_index), "zig")),
                fp16_scale=fp16_scale,
                decoded_mantissa_range=ByteRange(mantissa_start, mantissa_end),
                decoded_scale_range=ByteRange(scale_start, scale_end),
                compressed_range=ByteRange(compressed_start, compressed_end),
            )
        )
        pos = scale_end
    if pos != len(raw):
        raise Fec6ByteTargetError("decoder tensor map did not consume decompressed decoder bytes")
    return tuple(out)


def section_overlaps(byte_range: ByteRange, sections: Sequence[Fec6Section]) -> list[dict[str, Any]]:
    """Return named section overlaps for a half-open byte range."""

    rows = []
    for section in sections:
        overlap = byte_range.overlap_len(section.byte_range)
        if overlap <= 0:
            continue
        rows.append(
            {
                "section": section.name,
                "codec": section.codec,
                "overlap_bytes": overlap,
                "section_range": section.byte_range.as_dict(),
            }
        )
    return rows


def tensor_overlaps(byte_range: ByteRange, tensors: Sequence[DecoderTensorRange]) -> list[dict[str, Any]]:
    """Return decoder tensor overlaps for a half-open byte range."""

    rows = []
    for tensor in tensors:
        overlap = byte_range.overlap_len(tensor.compressed_range)
        if overlap <= 0:
            continue
        row = tensor.as_dict()
        row["overlap_bytes"] = overlap
        rows.append(row)
    rows.sort(key=lambda row: (-int(row["overlap_bytes"]), str(row["name"])))
    return rows


def summarize_tensor_shortlist(
    top_records: Sequence[Mapping[str, Any]],
    tensors: Sequence[DecoderTensorRange],
) -> list[dict[str, Any]]:
    """Aggregate top-byte score-impact records by overlapping decoder tensor."""

    accum: dict[str, dict[str, Any]] = {}
    for record in top_records:
        byte_index = int(record["byte_index"])
        byte_range = ByteRange(byte_index, byte_index + 1)
        axis_score = record.get("axis_score_impact")
        if not isinstance(axis_score, Mapping):
            axis_score = {}
        score_sum = float(record.get("score_impact_abs_sum", 0.0))
        for tensor in tensors:
            if tensor.compressed_range.overlap_len(byte_range) <= 0:
                continue
            row = accum.setdefault(
                tensor.name,
                {
                    "tensor_name": tensor.name,
                    "storage_index": tensor.storage_index,
                    "shape": list(tensor.shape),
                    "byte_map": tensor.byte_map,
                    "approx_compressed_range": tensor.compressed_range.as_dict(),
                    "top_byte_count": 0,
                    "top_byte_indices": [],
                    "score_impact_abs_sum": 0.0,
                    "axis_score_impact_abs_sum": {"seg": 0.0, "pose": 0.0, "rate": 0.0},
                    "mutation_policy": "tensor_domain_reencode_required",
                },
            )
            row["top_byte_count"] += 1
            row["top_byte_indices"].append(byte_index)
            row["score_impact_abs_sum"] += score_sum
            for axis in ("seg", "pose", "rate"):
                row["axis_score_impact_abs_sum"][axis] += float(axis_score.get(axis, 0.0))
    rows = list(accum.values())
    rows.sort(key=lambda row: (-float(row["score_impact_abs_sum"]), str(row["tensor_name"])))
    return rows


__all__ = [
    "ByteRange",
    "DecoderTensorRange",
    "Fec6ByteTargetError",
    "Fec6Section",
    "decoder_tensor_ranges",
    "parse_fec6_sections",
    "section_overlaps",
    "summarize_tensor_shortlist",
    "tensor_overlaps",
]
