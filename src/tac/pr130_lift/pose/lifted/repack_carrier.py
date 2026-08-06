#!/usr/bin/env python3
# borrowed_substrate_accounting:
#   source_repo: /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo
#   source_head: 2f94596bb0136d342254022a5c9584756eae0468
#   source_path: code/repack_carrier.py
#   source_sha256: df6bdaa23d0bd1f717af588931ebdb5ed3777517af8b046e287835aa16a7a72e
#   theirs: PR130 pose-leg mechanism, tensor shapes, search/codec semantics.
#   ours: vendored custody copy only; mx2 behavior changes live in sibling wrappers.
#   score_claim: false
#
"""Repack a legacy carrier with the lossless CPR1 carrier codec."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import struct
import zipfile
from pathlib import Path

import numpy as np

from carrier_codec import (
    HEADER,
    MAGIC,
    decode_compact_carrier,
    encode_compact_carrier,
)


N = 600
CARRIER_DIM = 12
CARRIER_H, CARRIER_W = 24, 32
BASIS_BITS = 5
BASIS_COUNT = CARRIER_DIM * 3 * CARRIER_H * CARRIER_W
COEFFICIENT_COUNT = N * CARRIER_DIM
SOURCE_ARCHIVE_SHA256 = (
    "f4457de09a6e69c8cd29e886a84705462a8c77dc6978020b11dff52e661a1451"
)
EXPECTED_ARCHIVE_BYTES = 191_052
EXPECTED_ARCHIVE_SHA256 = (
    "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
)
REFERENCE_BYTES = 37_545_489
LZMA_FILTERS = [{
    "id": lzma.FILTER_LZMA2,
    "dict_size": 1 << 16,
    "lc": 0,
    "lp": 1,
    "pb": 0,
    "mode": lzma.MODE_NORMAL,
    "nice_len": 273,
    "mf": lzma.MF_BT4,
    "depth": 0,
}]


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def unpack_signed_bits(
    blob: bytes | memoryview,
    count: int,
    bits: int,
) -> tuple[np.ndarray, memoryview]:
    byte_count = (count * bits + 7) // 8
    blob = memoryview(blob)
    if len(blob) < byte_count:
        raise ValueError("truncated packed basis codes")
    packed = np.frombuffer(blob[:byte_count], dtype=np.uint8)
    bitstream = np.unpackbits(
        packed,
        bitorder="little",
    )[:count * bits].reshape(count, bits)
    shifts = (1 << np.arange(bits, dtype=np.int32))[None]
    unsigned = (bitstream * shifts).sum(axis=1, dtype=np.int32)
    sign = 1 << (bits - 1)
    signed = np.where(unsigned >= sign, unsigned - (1 << bits), unsigned)
    return signed.astype(np.int32), blob[byte_count:]


def unpack_signed_int12(
    blob: bytes | memoryview,
    count: int,
) -> tuple[np.ndarray, memoryview]:
    byte_count = ((count + 1) // 2) * 3
    blob = memoryview(blob)
    if len(blob) < byte_count:
        raise ValueError("truncated packed coefficient codes")
    packed = np.frombuffer(
        blob[:byte_count],
        dtype=np.uint8,
    ).astype(np.uint16)
    values = np.empty((byte_count // 3) * 2, dtype=np.int32)
    values[0::2] = packed[0::3] | ((packed[1::3] & 0xF) << 8)
    values[1::2] = (packed[1::3] >> 4) | (packed[2::3] << 4)
    values[values >= 0x800] -= 0x1000
    return values[:count], blob[byte_count:]


def read_payload(path: Path, require_canonical_source: bool = True) -> bytes:
    archive_bytes = path.read_bytes()
    if (
        require_canonical_source
        and sha256(archive_bytes) != SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("source archive does not match the frozen canonical SHA-256")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != "p":
            raise ValueError("source archive must contain exactly one member named p")
        if infos[0].compress_type != zipfile.ZIP_STORED:
            raise ValueError("source archive member p must be stored")
        return archive.read("p")


def split_payload(payload: bytes) -> tuple[bytes, bytes, bytes]:
    if len(payload) < 4:
        raise ValueError("truncated submission payload")
    compressed_model_bytes = struct.unpack_from("<I", payload)[0]
    model_end = 4 + compressed_model_bytes
    if model_end > len(payload):
        raise ValueError("compressed model length exceeds submission payload")
    compressed_models = payload[4:model_end]
    tokens = payload[model_end:]
    models = lzma.decompress(compressed_models)
    recompressed = lzma.compress(
        models,
        format=lzma.FORMAT_XZ,
        filters=LZMA_FILTERS,
    )
    if recompressed != compressed_models:
        raise ValueError("recovered LZMA settings do not reproduce the source model")
    return compressed_models, models, tokens


def split_models(
    models: bytes,
) -> tuple[bytes, bytes, bytes, int, int]:
    if len(models) < 8:
        raise ValueError("truncated model bundle")
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models)
    semantic_start = 8
    carrier_start = semantic_start + semantic_bytes
    hpac_start = carrier_start + carrier_bytes
    if hpac_start > len(models):
        raise ValueError("semantic/carrier lengths exceed the model bundle")
    return (
        models[semantic_start:carrier_start],
        models[carrier_start:hpac_start],
        models[hpac_start:],
        semantic_bytes,
        carrier_bytes,
    )


def decode_legacy_carrier(
    carrier: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale_bytes = CARRIER_DIM * 4
    basis_bytes = BASIS_COUNT * BASIS_BITS // 8
    coefficient_bytes = ((COEFFICIENT_COUNT + 1) // 2) * 3
    expected_bytes = 2 * scale_bytes + basis_bytes + coefficient_bytes
    if len(carrier) != expected_bytes:
        raise ValueError(
            f"legacy carrier length mismatch: {len(carrier)} != {expected_bytes}"
        )
    remaining = memoryview(carrier)
    basis_scales = np.frombuffer(
        remaining[:scale_bytes],
        dtype="<f4",
    ).copy()
    remaining = remaining[scale_bytes:]
    basis_codes, remaining = unpack_signed_bits(
        remaining,
        BASIS_COUNT,
        BASIS_BITS,
    )
    coefficient_scales = np.frombuffer(
        remaining[:scale_bytes],
        dtype="<f4",
    ).copy()
    remaining = remaining[scale_bytes:]
    coefficient_codes, remaining = unpack_signed_int12(
        remaining,
        COEFFICIENT_COUNT,
    )
    if remaining:
        raise ValueError("legacy carrier has trailing bytes")
    return (
        basis_scales,
        basis_codes,
        coefficient_scales,
        (coefficient_codes.reshape(N, CARRIER_DIM) & 0xFFF),
    )


def write_deterministic_zip(output: Path, payload: bytes) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, payload)


def projected_score(
    archive_bytes: int,
    segmentation: float,
    pose_mse: float,
) -> float:
    return (
        100.0 * segmentation
        + math.sqrt(10.0 * pose_mse)
        + 25.0 * archive_bytes / REFERENCE_BYTES
    )


def repack(
    source_archive: Path,
    output_archive: Path,
    require_canonical_source: bool = True,
) -> dict:
    source_archive_bytes = source_archive.read_bytes()
    payload = read_payload(source_archive, require_canonical_source)
    source_compressed, models, tokens = split_payload(payload)
    semantic, carrier, hpac, semantic_bytes, carrier_bytes = split_models(models)
    (
        basis_scales,
        basis_codes,
        coefficient_scales,
        encoded_coefficients,
    ) = decode_legacy_carrier(carrier)
    compact_carrier = encode_compact_carrier(
        basis_scales,
        basis_codes,
        coefficient_scales,
        encoded_coefficients,
    )
    if compact_carrier[:HEADER.size][:4] != MAGIC:
        raise AssertionError("compact carrier encoder did not emit CPR1")
    decoded = decode_compact_carrier(
        compact_carrier,
        BASIS_COUNT,
        N,
        CARRIER_DIM,
    )
    expected = (
        basis_scales,
        basis_codes,
        coefficient_scales,
        encoded_coefficients,
    )
    for actual, wanted in zip(decoded, expected, strict=True):
        if not np.array_equal(actual, wanted):
            raise AssertionError("compact carrier round-trip changed a symbol")

    rebuilt_models = b"".join([
        struct.pack("<II", semantic_bytes, len(compact_carrier)),
        semantic,
        compact_carrier,
        hpac,
    ])
    compressed_models = lzma.compress(
        rebuilt_models,
        format=lzma.FORMAT_XZ,
        filters=LZMA_FILTERS,
    )
    rebuilt_payload = b"".join([
        struct.pack("<I", len(compressed_models)),
        compressed_models,
        tokens,
    ])
    write_deterministic_zip(output_archive, rebuilt_payload)
    output_archive_bytes = output_archive.read_bytes()
    output_archive_sha256 = sha256(output_archive_bytes)
    if require_canonical_source:
        if len(output_archive_bytes) != EXPECTED_ARCHIVE_BYTES:
            raise RuntimeError(
                f"CPR1 archive length mismatch: {len(output_archive_bytes)} "
                f"!= {EXPECTED_ARCHIVE_BYTES}"
            )
        if output_archive_sha256 != EXPECTED_ARCHIVE_SHA256:
            raise RuntimeError(
                f"CPR1 archive SHA-256 mismatch: {output_archive_sha256} "
                f"!= {EXPECTED_ARCHIVE_SHA256}"
            )
    semantic_pose = b"".join([
        struct.pack("<II", semantic_bytes, len(compact_carrier)),
        semantic,
        compact_carrier,
    ])
    result = {
        "schema_version": 1,
        "format": "CPR1",
        "canonical_source_required": require_canonical_source,
        "source": {
            "archive_bytes": len(source_archive_bytes),
            "archive_sha256": sha256(source_archive_bytes),
            "compressed_models_bytes": len(source_compressed),
            "raw_models_bytes": len(models),
            "carrier_bytes": carrier_bytes,
        },
        "archive": {
            "bytes": len(output_archive_bytes),
            "sha256": output_archive_sha256,
            "member": "p",
            "member_bytes": len(rebuilt_payload),
            "member_sha256": sha256(rebuilt_payload),
        },
        "payload": {
            "compressed_models_bytes": len(compressed_models),
            "compressed_models_sha256": sha256(compressed_models),
            "raw_models_bytes": len(rebuilt_models),
            "raw_models_sha256": sha256(rebuilt_models),
            "semantic_bytes": len(semantic),
            "carrier_bytes": len(compact_carrier),
            "carrier_sha256": sha256(compact_carrier),
            "semantic_pose_bytes": len(semantic_pose),
            "semantic_pose_sha256": sha256(semantic_pose),
            "hpac_raw_bytes": len(hpac),
            "hpac_raw_sha256": sha256(hpac),
            "token_bytes": len(tokens),
            "token_sha256": sha256(tokens),
            "basis_symbols": BASIS_COUNT,
            "coefficient_symbols": COEFFICIENT_COUNT,
            "lossless_carrier_roundtrip": True,
        },
    }
    if require_canonical_source:
        result["projection_from_displayed_metrics"] = {
            "ada": projected_score(
                len(output_archive_bytes),
                segmentation=0.00028609,
                pose_mse=0.00001967,
            ),
            "a4500": projected_score(
                len(output_archive_bytes),
                segmentation=0.00029607,
                pose_mse=0.00001981,
            ),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_archive", type=Path)
    parser.add_argument("output_archive", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--allow-noncanonical-source",
        action="store_true",
        help=(
            "accept a freshly trained legacy archive after validating its "
            "structure and lossless CPR1 symbol round-trip"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = repack(
        args.source_archive,
        args.output_archive,
        require_canonical_source=not args.allow_noncanonical_source,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
