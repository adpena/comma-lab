#!/usr/bin/env python3
"""Profile and safely repack PR95 HNeRV Muon archives.

This tool only performs byte-preserving transformations: brotli parameter
search, compact equivalent JSON metadata, and raw decoder-record reordering.
It does not dequantize/requantize tensors or alter latent values. Any emitted
candidate therefore has the same decoded model/latent contract as the source
blob, but it still requires exact CUDA eval before making a score claim.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import io
import json
import math
import struct
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import brotli


@dataclasses.dataclass(frozen=True)
class BrotliChoice:
    label: str
    raw_len: int
    compressed_len: int
    quality: int
    lgwin: int
    mode: int
    sha256: str
    payload: bytes = dataclasses.field(repr=False)

    def asdict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "raw_len": self.raw_len,
            "compressed_len": self.compressed_len,
            "quality": self.quality,
            "lgwin": self.lgwin,
            "mode": self.mode,
            "sha256": self.sha256,
        }


@dataclasses.dataclass(frozen=True)
class DecoderRecord:
    name: str
    shape: tuple[int, ...]
    scale: float
    q_zz: bytes

    @property
    def numel(self) -> int:
        out = 1
        for dim in self.shape:
            out *= dim
        return out

    def to_bytes(self) -> bytes:
        name_b = self.name.encode("utf-8")
        out = io.BytesIO()
        out.write(struct.pack("<I", len(name_b)))
        out.write(name_b)
        out.write(struct.pack("<I", len(self.shape)))
        for size in self.shape:
            out.write(struct.pack("<I", size))
        out.write(struct.pack("<f", self.scale))
        out.write(struct.pack("<I", len(self.q_zz)))
        out.write(self.q_zz)
        return out.getvalue()


@dataclasses.dataclass(frozen=True)
class LatentPayload:
    n_pairs: int
    latent_dim: int
    mins_f16: bytes
    scales_f16: bytes
    quantized: tuple[tuple[int, ...], ...]

    def to_bytes(self) -> bytes:
        total = self.n_pairs * self.latent_dim
        out = io.BytesIO()
        out.write(struct.pack("<II", self.n_pairs, self.latent_dim))
        out.write(self.mins_f16)
        out.write(self.scales_f16)

        lo = bytearray(total)
        hi = bytearray(total)
        prev = [0] * self.latent_dim
        offset = 0
        for row_index, row in enumerate(self.quantized):
            for dim_index, value in enumerate(row):
                delta = value if row_index == 0 else value - prev[dim_index]
                zz = 2 * delta if delta >= 0 else -2 * delta - 1
                lo[offset] = zz & 0xFF
                hi[offset] = (zz >> 8) & 0xFF
                offset += 1
            prev = list(row)
        out.write(lo)
        out.write(hi)
        return out.getvalue()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shannon_entropy_bits(data: bytes) -> float:
    if not data:
        return 0.0
    counts = collections.Counter(data)
    n = len(data)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def read_single_member_zip(path: Path) -> tuple[str, bytes, dict[str, object]]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise ValueError(f"expected exactly one archive member, got {len(infos)}")
        info = infos[0]
        if info.filename != "0.bin":
            raise ValueError(f"expected member 0.bin, got {info.filename!r}")
        data = zf.read(info)
        return info.filename, data, {
            "compress_type": int(info.compress_type),
            "file_size": int(info.file_size),
            "compress_size": int(info.compress_size),
            "crc": int(info.CRC),
            "date_time": list(info.date_time),
        }


def write_stored_zip(path: Path, member_name: str, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as zf:
        zf.writestr(info, payload)


def parse_top_blob(blob: bytes) -> dict[str, object]:
    buf = io.BytesIO(blob)
    meta_len = struct.unpack("<I", buf.read(4))[0]
    meta_brotli = buf.read(meta_len)
    dec_len = struct.unpack("<I", buf.read(4))[0]
    decoder_brotli = buf.read(dec_len)
    lat_len = struct.unpack("<I", buf.read(4))[0]
    latents_brotli = buf.read(lat_len)
    rest = buf.read()
    if rest:
        raise ValueError(f"trailing bytes after PR95 blob: {len(rest)}")
    return {
        "meta_brotli": meta_brotli,
        "meta_raw": brotli.decompress(meta_brotli),
        "decoder_brotli": decoder_brotli,
        "decoder_raw": brotli.decompress(decoder_brotli),
        "latents_brotli": latents_brotli,
        "latents_raw": brotli.decompress(latents_brotli),
    }


def encode_top_blob(meta_brotli: bytes, decoder_brotli: bytes, latents_brotli: bytes) -> bytes:
    out = io.BytesIO()
    out.write(struct.pack("<I", len(meta_brotli)))
    out.write(meta_brotli)
    out.write(struct.pack("<I", len(decoder_brotli)))
    out.write(decoder_brotli)
    out.write(struct.pack("<I", len(latents_brotli)))
    out.write(latents_brotli)
    return out.getvalue()


def brotli_search(label: str, raw: bytes, *, qualities: Iterable[int], lgwins: Iterable[int]) -> BrotliChoice:
    best: BrotliChoice | None = None
    for quality in qualities:
        for lgwin in lgwins:
            for mode in (brotli.MODE_GENERIC, brotli.MODE_TEXT):
                try:
                    payload = brotli.compress(raw, quality=quality, lgwin=lgwin, mode=mode)
                except brotli.error:
                    continue
                if brotli.decompress(payload) != raw:
                    raise AssertionError(f"brotli roundtrip mismatch for {label}")
                choice = BrotliChoice(
                    label=label,
                    raw_len=len(raw),
                    compressed_len=len(payload),
                    quality=quality,
                    lgwin=lgwin,
                    mode=mode,
                    sha256=sha256_bytes(payload),
                    payload=payload,
                )
                if best is None or choice.compressed_len < best.compressed_len:
                    best = choice
    if best is None:
        raise RuntimeError(f"no brotli candidate produced for {label}")
    return best


def compact_meta_raw(meta_raw: bytes) -> bytes:
    payload = json.loads(meta_raw)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def parse_decoder_records(decoder_raw: bytes) -> list[bytes]:
    return [record.to_bytes() for record in parse_decoder_records_structured(decoder_raw)]


def parse_decoder_records_structured(decoder_raw: bytes) -> list[DecoderRecord]:
    buf = io.BytesIO(decoder_raw)
    count = struct.unpack("<I", buf.read(4))[0]
    records: list[DecoderRecord] = []
    for _ in range(count):
        name_len = struct.unpack("<I", buf.read(4))[0]
        name = buf.read(name_len).decode("utf-8")
        ndims = struct.unpack("<I", buf.read(4))[0]
        shape = tuple(struct.unpack("<I", buf.read(4))[0] for _ in range(ndims))
        scale = struct.unpack("<f", buf.read(4))[0]
        q_size = struct.unpack("<I", buf.read(4))[0]
        q_zz = buf.read(q_size)
        record = DecoderRecord(name=name, shape=shape, scale=scale, q_zz=q_zz)
        if record.numel != q_size:
            raise ValueError(f"decoder record {name!r} shape {shape} has {record.numel} values, stored {q_size}")
        records.append(record)
    rest = buf.read()
    if rest:
        raise ValueError(f"decoder raw has trailing bytes: {len(rest)}")
    return records


def decoder_record_name(record: bytes) -> str:
    buf = io.BytesIO(record)
    name_len = struct.unpack("<I", buf.read(4))[0]
    return buf.read(name_len).decode("utf-8")


def rebuild_decoder_raw(records: list[bytes]) -> bytes:
    out = io.BytesIO()
    out.write(struct.pack("<I", len(records)))
    for record in records:
        out.write(record)
    return out.getvalue()


def rebuild_structured_decoder_raw(records: Sequence[DecoderRecord]) -> bytes:
    return rebuild_decoder_raw([record.to_bytes() for record in records])


def decoder_raw_variants(decoder_raw: bytes) -> dict[str, bytes]:
    records = parse_decoder_records(decoder_raw)
    variants = {
        "original": decoder_raw,
        "name_asc": rebuild_decoder_raw(sorted(records, key=decoder_record_name)),
        "name_desc": rebuild_decoder_raw(sorted(records, key=decoder_record_name, reverse=True)),
        "size_desc": rebuild_decoder_raw(sorted(records, key=len, reverse=True)),
        "size_asc": rebuild_decoder_raw(sorted(records, key=len)),
    }
    return variants


def parse_latents_raw(latents_raw: bytes) -> LatentPayload:
    buf = io.BytesIO(latents_raw)
    n_pairs, latent_dim = struct.unpack("<II", buf.read(8))
    mins_f16 = buf.read(latent_dim * 2)
    scales_f16 = buf.read(latent_dim * 2)
    total = n_pairs * latent_dim
    lo = buf.read(total)
    hi = buf.read(total)
    rest = buf.read()
    if len(mins_f16) != latent_dim * 2 or len(scales_f16) != latent_dim * 2:
        raise ValueError("latent raw header is truncated")
    if len(lo) != total or len(hi) != total:
        raise ValueError("latent delta streams are truncated")
    if rest:
        raise ValueError(f"latent raw has trailing bytes: {len(rest)}")

    rows: list[tuple[int, ...]] = []
    prev = [0] * latent_dim
    for pair_index in range(n_pairs):
        row: list[int] = []
        for dim_index in range(latent_dim):
            offset = pair_index * latent_dim + dim_index
            zz = lo[offset] | (hi[offset] << 8)
            delta = zz // 2 if zz % 2 == 0 else -(zz // 2) - 1
            value = delta if pair_index == 0 else prev[dim_index] + delta
            if not 0 <= value <= 255:
                raise ValueError(f"latent quantized value out of uint8 range at pair {pair_index}, dim {dim_index}: {value}")
            row.append(value)
        rows.append(tuple(row))
        prev = row
    return LatentPayload(
        n_pairs=n_pairs,
        latent_dim=latent_dim,
        mins_f16=mins_f16,
        scales_f16=scales_f16,
        quantized=tuple(rows),
    )


def reorder_latents_raw(latents_raw: bytes, permutation: Sequence[int]) -> bytes:
    payload = parse_latents_raw(latents_raw)
    validate_permutation(permutation, payload.latent_dim)

    def reorder_f16_pairs(raw: bytes) -> bytes:
        return b"".join(raw[2 * dim : 2 * dim + 2] for dim in permutation)

    reordered_rows = tuple(tuple(row[dim] for dim in permutation) for row in payload.quantized)
    return LatentPayload(
        n_pairs=payload.n_pairs,
        latent_dim=payload.latent_dim,
        mins_f16=reorder_f16_pairs(payload.mins_f16),
        scales_f16=reorder_f16_pairs(payload.scales_f16),
        quantized=reordered_rows,
    ).to_bytes()


def validate_permutation(permutation: Sequence[int], width: int) -> None:
    if sorted(permutation) != list(range(width)):
        raise ValueError(f"not a width-{width} permutation: {list(permutation)}")


def reorder_stem_weight_record(record: DecoderRecord, permutation: Sequence[int]) -> DecoderRecord:
    if record.name != "stem.weight" or len(record.shape) != 2:
        raise ValueError("latent-dimension permutation requires the stem.weight matrix record")
    out_features, in_features = record.shape
    validate_permutation(permutation, in_features)
    rows = []
    for row_index in range(out_features):
        base = row_index * in_features
        rows.append(bytes(record.q_zz[base + dim] for dim in permutation))
    return dataclasses.replace(record, q_zz=b"".join(rows))


def decoder_raw_with_latent_permutation(decoder_raw: bytes, permutation: Sequence[int]) -> bytes:
    records = parse_decoder_records_structured(decoder_raw)
    out_records = [
        reorder_stem_weight_record(record, permutation) if record.name == "stem.weight" else record
        for record in records
    ]
    return rebuild_structured_decoder_raw(out_records)


def latent_dimension_profiles(decoder_raw: bytes, latents_raw: bytes) -> list[dict[str, object]]:
    records = parse_decoder_records_structured(decoder_raw)
    stem = next((record for record in records if record.name == "stem.weight"), None)
    if stem is None:
        raise ValueError("decoder does not contain stem.weight")
    _, latent_dim = stem.shape
    payload = parse_latents_raw(latents_raw)
    if payload.latent_dim != latent_dim:
        raise ValueError(f"latent dim mismatch: decoder {latent_dim}, latent stream {payload.latent_dim}")

    profiles = []
    for dim in range(latent_dim):
        latent_values = bytes(row[dim] for row in payload.quantized)
        stem_column = bytes(stem.q_zz[row * latent_dim + dim] for row in range(stem.shape[0]))
        profiles.append(
            {
                "dim": dim,
                "latent_unique": len(set(latent_values)),
                "latent_entropy_bits": shannon_entropy_bits(latent_values),
                "latent_brotli_q11_len": len(brotli.compress(latent_values, quality=11)),
                "stem_unique": len(set(stem_column)),
                "stem_entropy_bits": shannon_entropy_bits(stem_column),
                "stem_brotli_q11_len": len(brotli.compress(stem_column, quality=11)),
            }
        )
    return profiles


def permutation_variants(decoder_raw: bytes, latents_raw: bytes) -> dict[str, list[int]]:
    profiles = latent_dimension_profiles(decoder_raw, latents_raw)
    latent_dim = len(profiles)
    variants = {
        "original": list(range(latent_dim)),
        "reverse": list(reversed(range(latent_dim))),
        "latent_brotli_asc": [p["dim"] for p in sorted(profiles, key=lambda p: (p["latent_brotli_q11_len"], p["dim"]))],
        "latent_brotli_desc": [p["dim"] for p in sorted(profiles, key=lambda p: (-p["latent_brotli_q11_len"], p["dim"]))],
        "stem_brotli_asc": [p["dim"] for p in sorted(profiles, key=lambda p: (p["stem_brotli_q11_len"], p["dim"]))],
        "stem_brotli_desc": [p["dim"] for p in sorted(profiles, key=lambda p: (-p["stem_brotli_q11_len"], p["dim"]))],
        "combined_brotli_asc": [
            p["dim"]
            for p in sorted(
                profiles,
                key=lambda p: (p["latent_brotli_q11_len"] + p["stem_brotli_q11_len"], p["dim"]),
            )
        ],
        "combined_brotli_desc": [
            p["dim"]
            for p in sorted(
                profiles,
                key=lambda p: (-(p["latent_brotli_q11_len"] + p["stem_brotli_q11_len"]), p["dim"]),
            )
        ],
    }
    deduped: dict[str, list[int]] = {}
    seen: set[tuple[int, ...]] = set()
    for label, perm in variants.items():
        key = tuple(perm)
        if key in seen:
            continue
        seen.add(key)
        deduped[label] = perm
    return deduped


def decoder_record_accounting(decoder_raw: bytes) -> list[dict[str, object]]:
    records = parse_decoder_records_structured(decoder_raw)
    out = []
    for index, record in enumerate(records):
        record_bytes = record.to_bytes()
        out.append(
            {
                "index": index,
                "name": record.name,
                "shape": list(record.shape),
                "numel": record.numel,
                "raw_record_bytes": len(record_bytes),
                "q_bytes": len(record.q_zz),
                "q_unique": len(set(record.q_zz)),
                "q_zero_zigzag_fraction": record.q_zz.count(0) / len(record.q_zz) if record.q_zz else 0.0,
                "q_entropy_bits": shannon_entropy_bits(record.q_zz),
                "standalone_brotli_q11_bytes": len(brotli.compress(record_bytes, quality=11)),
                "sha256": sha256_bytes(record_bytes),
            }
        )
    return out


@dataclasses.dataclass(frozen=True)
class CoupledCandidate:
    permutation_label: str
    permutation: list[int]
    decoder_choice: BrotliChoice
    latents_choice: BrotliChoice

    @property
    def compressed_len(self) -> int:
        return self.decoder_choice.compressed_len + self.latents_choice.compressed_len

    def asdict(self) -> dict[str, object]:
        return {
            "permutation_label": self.permutation_label,
            "permutation": self.permutation,
            "compressed_len": self.compressed_len,
            "decoder": self.decoder_choice.asdict(),
            "latents": self.latents_choice.asdict(),
        }


def run(args: argparse.Namespace) -> int:
    source_zip = Path(args.archive)
    output_dir = Path(args.output_dir)
    member_name, blob, zip_meta = read_single_member_zip(source_zip)
    parts = parse_top_blob(blob)

    qualities = range(args.min_quality, args.max_quality + 1)
    lgwins = range(args.min_lgwin, args.max_lgwin + 1)

    meta_raws = {
        "original": parts["meta_raw"],
        "compact_sorted": compact_meta_raw(parts["meta_raw"]),
    }
    meta_choices = [
        brotli_search(f"meta:{name}", raw, qualities=qualities, lgwins=lgwins)
        for name, raw in meta_raws.items()
    ]
    best_meta = min(meta_choices, key=lambda c: c.compressed_len)

    coupled_candidates: list[CoupledCandidate] = []
    for permutation_label, permutation in permutation_variants(parts["decoder_raw"], parts["latents_raw"]).items():
        permuted_decoder_raw = decoder_raw_with_latent_permutation(parts["decoder_raw"], permutation)
        permuted_latents_raw = reorder_latents_raw(parts["latents_raw"], permutation)
        decoder_choices = [
            brotli_search(
                f"decoder:{permutation_label}:{record_variant_label}",
                raw,
                qualities=qualities,
                lgwins=lgwins,
            )
            for record_variant_label, raw in decoder_raw_variants(permuted_decoder_raw).items()
        ]
        latents_choice = brotli_search(
            f"latents:{permutation_label}",
            permuted_latents_raw,
            qualities=qualities,
            lgwins=lgwins,
        )
        coupled_candidates.append(
            CoupledCandidate(
                permutation_label=permutation_label,
                permutation=permutation,
                decoder_choice=min(decoder_choices, key=lambda c: c.compressed_len),
                latents_choice=latents_choice,
            )
        )
    best_coupled = min(coupled_candidates, key=lambda c: c.compressed_len)
    conservative_coupled = next(c for c in coupled_candidates if c.permutation_label == "original")
    best_decoder = best_coupled.decoder_choice
    best_latents = best_coupled.latents_choice

    candidate_blob = encode_top_blob(best_meta.payload, best_decoder.payload, best_latents.payload)
    conservative_blob = encode_top_blob(
        best_meta.payload,
        conservative_coupled.decoder_choice.payload,
        conservative_coupled.latents_choice.payload,
    )
    conservative_zip = output_dir / "archive.pr95_repacked.zip"
    write_stored_zip(conservative_zip, member_name, conservative_blob)

    candidate_zip = (
        conservative_zip
        if best_coupled.permutation_label == "original"
        else output_dir / "archive.pr95_repacked_stemperm.zip"
    )
    write_stored_zip(candidate_zip, member_name, candidate_blob)

    manifest = {
        "schema_version": 1,
        "tool": "profile_pr95_hnerv_muon_packing",
        "source_archive": str(source_zip),
        "source_archive_bytes": source_zip.stat().st_size,
        "source_archive_sha256": sha256_bytes(source_zip.read_bytes()),
        "source_member": member_name,
        "source_member_bytes": len(blob),
        "source_member_sha256": sha256_bytes(blob),
        "source_zip_member_meta": zip_meta,
        "candidate_archive": str(candidate_zip),
        "candidate_archive_bytes": candidate_zip.stat().st_size,
        "candidate_archive_sha256": sha256_bytes(candidate_zip.read_bytes()),
        "candidate_member_bytes": len(candidate_blob),
        "candidate_member_sha256": sha256_bytes(candidate_blob),
        "archive_byte_delta": candidate_zip.stat().st_size - source_zip.stat().st_size,
        "member_byte_delta": len(candidate_blob) - len(blob),
        "conservative_archive": str(conservative_zip),
        "conservative_archive_bytes": conservative_zip.stat().st_size,
        "conservative_archive_sha256": sha256_bytes(conservative_zip.read_bytes()),
        "conservative_member_bytes": len(conservative_blob),
        "conservative_member_sha256": sha256_bytes(conservative_blob),
        "conservative_archive_byte_delta": conservative_zip.stat().st_size - source_zip.stat().st_size,
        "conservative_choice": conservative_coupled.asdict(),
        "choices": {
            "meta": best_meta.asdict(),
            "decoder": best_decoder.asdict(),
            "latents": best_latents.asdict(),
            "coupled_variant": best_coupled.asdict(),
        },
        "searched": {
            "qualities": [args.min_quality, args.max_quality],
            "lgwins": [args.min_lgwin, args.max_lgwin],
            "meta_choices": [c.asdict() for c in sorted(meta_choices, key=lambda c: c.compressed_len)[:8]],
            "coupled_candidates": [
                c.asdict() for c in sorted(coupled_candidates, key=lambda c: c.compressed_len)[:12]
            ],
        },
        "accounting": {
            "logical_streams": {
                "meta": {
                    "source_compressed_bytes": len(parts["meta_brotli"]),
                    "source_raw_bytes": len(parts["meta_raw"]),
                    "source_raw_sha256": sha256_bytes(parts["meta_raw"]),
                    "candidate_compressed_bytes": best_meta.compressed_len,
                    "candidate_raw_bytes": best_meta.raw_len,
                    "candidate_compressed_sha256": best_meta.sha256,
                },
                "decoder": {
                    "source_compressed_bytes": len(parts["decoder_brotli"]),
                    "source_raw_bytes": len(parts["decoder_raw"]),
                    "source_raw_sha256": sha256_bytes(parts["decoder_raw"]),
                    "candidate_compressed_bytes": best_decoder.compressed_len,
                    "candidate_raw_bytes": best_decoder.raw_len,
                    "candidate_compressed_sha256": best_decoder.sha256,
                },
                "latents": {
                    "source_compressed_bytes": len(parts["latents_brotli"]),
                    "source_raw_bytes": len(parts["latents_raw"]),
                    "source_raw_sha256": sha256_bytes(parts["latents_raw"]),
                    "candidate_compressed_bytes": best_latents.compressed_len,
                    "candidate_raw_bytes": best_latents.raw_len,
                    "candidate_compressed_sha256": best_latents.sha256,
                },
            },
            "decoder_records": decoder_record_accounting(parts["decoder_raw"]),
            "latent_dimension_profiles": latent_dimension_profiles(parts["decoder_raw"], parts["latents_raw"]),
        },
        "no_op_detection": {
            "archive_sha_changed": sha256_bytes(candidate_zip.read_bytes()) != sha256_bytes(source_zip.read_bytes()),
            "member_sha_changed": sha256_bytes(candidate_blob) != sha256_bytes(blob),
            "archive_bytes_changed": candidate_zip.stat().st_size != source_zip.stat().st_size,
            "candidate_smaller": candidate_zip.stat().st_size < source_zip.stat().st_size,
            "source_archive_reused": False,
        },
        "evidence_grade": "byte_exact_candidate_until_exact_cuda_eval",
        "score_claim": False,
        "safety": {
            "transformation": "brotli_recompression_decoder_record_reordering_and_lossless_latent_stem_permutation",
            "decoded_stream_roundtrip_checked": True,
            "no_tensor_requantization": True,
            "latent_dimension_permutation_compensated_by_stem_weight": True,
            "uses_existing_pr95_runtime_contract": True,
            "score_preserving_claim": best_coupled.permutation_label == "original",
            "floating_accumulation_order_can_change": best_coupled.permutation_label != "original",
            "requires_exact_cuda_eval": True,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "profile_pr95_hnerv_muon_packing.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        default="experiments/results/public_pr95_intake_20260504_codex/archive.zip",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex",
    )
    parser.add_argument("--min-quality", type=int, default=8)
    parser.add_argument("--max-quality", type=int, default=11)
    parser.add_argument("--min-lgwin", type=int, default=18)
    parser.add_argument("--max-lgwin", type=int, default=24)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
