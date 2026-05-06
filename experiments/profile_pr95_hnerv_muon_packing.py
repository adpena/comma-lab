#!/usr/bin/env python3
"""Profile and safely repack PR95 HNeRV Muon single-payload archives.

This is a forensic/deconstruction tool. It parses the public PR95 archive wire
format into charged sections and low-level decoder/latent records, then emits a
deterministic byte profile. It does not dequantize/requantize tensors or claim a
score; exact CUDA auth eval remains the only score truth.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import io
import json
import math
import struct
from pathlib import Path
from typing import Iterable, Sequence

import brotli

from tac.pr95_hnerv import (
    LatentPayload,
    encode_top_blob,
    parse_latents_raw,
    parse_top_blob,
    read_single_member_zip,
    sha256_bytes,
)


@dataclasses.dataclass(frozen=True)
class BrotliChoice:
    label: str
    raw_bytes: int
    compressed_bytes: int
    quality: int
    lgwin: int | None
    sha256: str


@dataclasses.dataclass(frozen=True)
class DecoderRecord:
    name: str
    shape: tuple[int, ...]
    scale: float
    quantized_zigzag: bytes

    @property
    def n_values(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def to_bytes(self) -> bytes:
        if len(self.quantized_zigzag) != self.n_values:
            raise ValueError(
                f"{self.name}: q bytes {len(self.quantized_zigzag)} != shape product {self.n_values}"
            )
        out = io.BytesIO()
        name_b = self.name.encode("utf-8")
        out.write(struct.pack("<I", len(name_b)))
        out.write(name_b)
        out.write(struct.pack("<I", len(self.shape)))
        for dim in self.shape:
            out.write(struct.pack("<I", int(dim)))
        out.write(struct.pack("<f", float(self.scale)))
        out.write(struct.pack("<I", len(self.quantized_zigzag)))
        out.write(self.quantized_zigzag)
        return out.getvalue()


@dataclasses.dataclass(frozen=True)
class SectionProfile:
    name: str
    raw_bytes: int
    compressed_bytes: int | None
    sha256: str
    entropy_bits_per_byte: float


def shannon_entropy_bits_per_byte(data: bytes) -> float:
    if not data:
        return 0.0
    n = len(data)
    counts = collections.Counter(data)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def _read_exact(buf: io.BytesIO, n: int, label: str) -> bytes:
    data = buf.read(n)
    if len(data) != n:
        raise ValueError(f"truncated {label}: wanted {n}, got {len(data)}")
    return data


def compact_meta_raw(meta_raw: bytes) -> bytes:
    payload = json.loads(meta_raw)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def brotli_search(
    label: str,
    raw: bytes,
    *,
    qualities: Iterable[int] = (9, 10, 11),
    lgwins: Iterable[int | None] = (None,),
) -> BrotliChoice:
    best: BrotliChoice | None = None
    for quality in qualities:
        for lgwin in lgwins:
            kwargs = {"quality": int(quality)}
            if lgwin is not None:
                kwargs["lgwin"] = int(lgwin)
            compressed = brotli.compress(raw, **kwargs)
            choice = BrotliChoice(
                label=label,
                raw_bytes=len(raw),
                compressed_bytes=len(compressed),
                quality=int(quality),
                lgwin=None if lgwin is None else int(lgwin),
                sha256=sha256_bytes(compressed),
            )
            if best is None or choice.compressed_bytes < best.compressed_bytes:
                best = choice
    assert best is not None
    return best


def parse_decoder_records_structured(decoder_raw: bytes) -> list[DecoderRecord]:
    buf = io.BytesIO(decoder_raw)
    (count,) = struct.unpack("<I", _read_exact(buf, 4, "decoder_record_count"))
    records: list[DecoderRecord] = []
    for record_index in range(count):
        (name_len,) = struct.unpack("<I", _read_exact(buf, 4, f"record {record_index} name_len"))
        name = _read_exact(buf, name_len, f"record {record_index} name").decode("utf-8")
        (shape_len,) = struct.unpack("<I", _read_exact(buf, 4, f"{name} shape_len"))
        shape = tuple(
            struct.unpack("<I", _read_exact(buf, 4, f"{name} shape[{i}]"))[0]
            for i in range(shape_len)
        )
        (scale,) = struct.unpack("<f", _read_exact(buf, 4, f"{name} scale"))
        (q_size,) = struct.unpack("<I", _read_exact(buf, 4, f"{name} q_size"))
        quantized = _read_exact(buf, q_size, f"{name} quantized_zigzag")
        record = DecoderRecord(name=name, shape=shape, scale=float(scale), quantized_zigzag=quantized)
        if record.n_values != q_size:
            raise ValueError(f"{name}: shape product {record.n_values} != q_size {q_size}")
        records.append(record)
    rest = buf.read()
    if rest:
        raise ValueError(f"trailing bytes after decoder records: {len(rest)}")
    return records


def parse_decoder_records(decoder_raw: bytes) -> list[bytes]:
    return [record.to_bytes() for record in parse_decoder_records_structured(decoder_raw)]


def decoder_record_name(record: bytes) -> str:
    buf = io.BytesIO(record)
    (name_len,) = struct.unpack("<I", _read_exact(buf, 4, "decoder_record_name_len"))
    return _read_exact(buf, name_len, "decoder_record_name").decode("utf-8")


def rebuild_decoder_raw(records: Sequence[bytes]) -> bytes:
    out = io.BytesIO()
    out.write(struct.pack("<I", len(records)))
    for record in records:
        out.write(record)
    return out.getvalue()


def rebuild_structured_decoder_raw(records: Sequence[DecoderRecord]) -> bytes:
    return rebuild_decoder_raw([record.to_bytes() for record in records])


def decoder_raw_variants(decoder_raw: bytes) -> dict[str, bytes]:
    records = parse_decoder_records(decoder_raw)
    return {
        "original": decoder_raw,
        "name_asc": rebuild_decoder_raw(sorted(records, key=decoder_record_name)),
        "name_desc": rebuild_decoder_raw(sorted(records, key=decoder_record_name, reverse=True)),
        "size_desc": rebuild_decoder_raw(sorted(records, key=len, reverse=True)),
        "size_asc": rebuild_decoder_raw(sorted(records, key=len)),
    }


def validate_permutation(permutation: Sequence[int], width: int) -> None:
    if sorted(permutation) != list(range(width)):
        raise ValueError(f"not a width-{width} permutation: {list(permutation)}")


def reorder_latents_raw(latents_raw: bytes, permutation: Sequence[int]) -> bytes:
    payload = parse_latents_raw(latents_raw)
    validate_permutation(permutation, payload.latent_dim)
    mins = b"".join(payload.mins_f16[i * 2 : i * 2 + 2] for i in permutation)
    scales = b"".join(payload.scales_f16[i * 2 : i * 2 + 2] for i in permutation)
    rows = tuple(tuple(row[i] for i in permutation) for row in payload.quantized)
    return LatentPayload(
        n_pairs=payload.n_pairs,
        latent_dim=payload.latent_dim,
        mins_f16=mins,
        scales_f16=scales,
        quantized=rows,
    ).to_bytes()


def reorder_stem_weight_record(record: DecoderRecord, permutation: Sequence[int]) -> DecoderRecord:
    if record.name != "stem.weight":
        return record
    if len(record.shape) < 2:
        raise ValueError("stem.weight must have at least two dimensions")
    latent_dim = record.shape[1]
    validate_permutation(permutation, latent_dim)
    outer = record.shape[0]
    inner_block = 1
    for dim in record.shape[2:]:
        inner_block *= dim
    q = record.quantized_zigzag
    reordered = bytearray(record.n_values)
    for out_index in range(outer):
        out_base = out_index * latent_dim * inner_block
        for out_col, in_col in enumerate(permutation):
            dst = out_base + out_col * inner_block
            src = out_base + in_col * inner_block
            reordered[dst : dst + inner_block] = q[src : src + inner_block]
    return dataclasses.replace(record, quantized_zigzag=bytes(reordered))


def decoder_raw_with_latent_permutation(decoder_raw: bytes, permutation: Sequence[int]) -> bytes:
    records = [
        reorder_stem_weight_record(record, permutation)
        for record in parse_decoder_records_structured(decoder_raw)
    ]
    return rebuild_structured_decoder_raw(records)


def _section_profile(name: str, raw: bytes, compressed: bytes | None = None) -> SectionProfile:
    data = raw if compressed is None else compressed
    return SectionProfile(
        name=name,
        raw_bytes=len(raw),
        compressed_bytes=None if compressed is None else len(compressed),
        sha256=sha256_bytes(data),
        entropy_bits_per_byte=round(shannon_entropy_bits_per_byte(data), 6),
    )


def build_profile(archive: Path) -> dict[str, object]:
    member_name, payload, zip_info = read_single_member_zip(archive)
    top = parse_top_blob(payload)
    decoder_records = parse_decoder_records_structured(top["decoder_raw"])
    latents = parse_latents_raw(top["latents_raw"])
    if latents.to_bytes() != top["latents_raw"]:
        raise ValueError("latent raw stream did not round-trip")
    if rebuild_structured_decoder_raw(decoder_records) != top["decoder_raw"]:
        raise ValueError("decoder raw stream did not round-trip")
    return {
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_bytes(archive.read_bytes()),
        "member_name": member_name,
        "member_bytes": len(payload),
        "member_sha256": sha256_bytes(payload),
        "zip_info": zip_info,
        "top_blob_lengths": {
            "meta_brotli": len(top["meta_brotli"]),
            "decoder_brotli": len(top["decoder_brotli"]),
            "latents_brotli": len(top["latents_brotli"]),
            "framing_bytes": 12,
        },
        "meta": json.loads(top["meta_raw"]),
        "decoder": {
            "record_count": len(decoder_records),
            "raw_bytes": len(top["decoder_raw"]),
            "brotli_bytes": len(top["decoder_brotli"]),
            "largest_records": [
                {
                    "name": record.name,
                    "shape": list(record.shape),
                    "q_bytes": len(record.quantized_zigzag),
                    "scale": record.scale,
                    "sha256": sha256_bytes(record.quantized_zigzag),
                }
                for record in sorted(decoder_records, key=lambda item: len(item.quantized_zigzag), reverse=True)[:16]
            ],
        },
        "latents": {
            "n_pairs": latents.n_pairs,
            "latent_dim": latents.latent_dim,
            "raw_bytes": len(top["latents_raw"]),
            "brotli_bytes": len(top["latents_brotli"]),
            "mins_f16_sha256": sha256_bytes(latents.mins_f16),
            "scales_f16_sha256": sha256_bytes(latents.scales_f16),
        },
        "sections": [
            dataclasses.asdict(_section_profile("meta_brotli", top["meta_raw"], top["meta_brotli"])),
            dataclasses.asdict(_section_profile("decoder_brotli", top["decoder_raw"], top["decoder_brotli"])),
            dataclasses.asdict(_section_profile("latents_brotli", top["latents_raw"], top["latents_brotli"])),
        ],
        "roundtrip_top_blob_sha256": sha256_bytes(
            encode_top_blob(top["meta_brotli"], top["decoder_brotli"], top["latents_brotli"])
        ),
        "evidence_grade": "forensic_byte_profile",
        "score_claim": False,
    }


def run(archive: Path, output_json: Path | None = None, output_md: Path | None = None) -> dict[str, object]:
    record = build_profile(archive)
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(record))
    return record


def render_markdown(record: dict[str, object]) -> str:
    lines = [
        f"# PR95 HNeRV Muon Packing Profile: {record['archive']}",
        "",
        f"- archive_bytes: `{record['archive_bytes']}`",
        f"- archive_sha256: `{record['archive_sha256']}`",
        f"- member_name: `{record['member_name']}`",
        f"- member_bytes: `{record['member_bytes']}`",
        f"- member_sha256: `{record['member_sha256']}`",
        f"- score_claim: `{record['score_claim']}`",
        f"- evidence_grade: `{record['evidence_grade']}`",
        "",
        "## Top Blob",
        "",
        "| section | raw bytes | compressed bytes | entropy b/B | sha256 |",
        "|---|---:|---:|---:|---|",
    ]
    for section in record["sections"]:
        lines.append(
            "| {name} | {raw_bytes} | {compressed_bytes} | {entropy_bits_per_byte:.6f} | `{sha256}` |".format(
                **section
            )
        )
    lines.extend(["", "## Decoder Largest Records", ""])
    lines.append("| name | shape | q bytes | scale | q sha256 |")
    lines.append("|---|---|---:|---:|---|")
    for item in record["decoder"]["largest_records"]:
        lines.append(
            "| {name} | `{shape}` | {q_bytes} | {scale:.9g} | `{sha256}` |".format(**item)
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args(argv)

    record = run(args.archive, args.json_out, args.md_out)
    if not args.json_out and not args.md_out:
        print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
