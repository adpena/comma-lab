#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Non-promotable custom mask codec probe.

This is a bounded, dependency-free diagnostic scaffold for custom overfit mask
payloads. It round-trips deterministic synthetic class-id masks through a small
bitpacked RLE container and writes empirical byte/custody metadata. It does not
build a contest archive, does not call scorer networks, and does not make score
claims.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAGIC = b"CMCPRLE1"
VERSION = 1
SCHEMA = "custom_mask_codec_probe_v1"
PAYLOAD_SCHEMA = "cmcp_rle1"
EVIDENCE_GRADE = "empirical"
REPORT_NAME = "custom_mask_codec_probe_manifest.json"
PAYLOAD_NAME = "synthetic_masks.cmcp_rle1"
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
SCORE_CLAIM_WARNING = (
    "No score claim is made by this probe. Synthetic byte counts and round-trip "
    "checks are empirical design evidence only; a deterministic archive and "
    "exact CUDA auth eval are required before any score claim."
)

_HEADER = struct.Struct(">8sBBBBIIII")
_CRC = struct.Struct(">I")


@dataclass(frozen=True)
class MaskShape:
    frames: int
    height: int
    width: int
    classes: int = 5

    @property
    def pixels(self) -> int:
        return self.frames * self.height * self.width

    @property
    def bits_per_symbol(self) -> int:
        return max(1, (self.classes - 1).bit_length())

    def as_list(self) -> list[int]:
        return [self.frames, self.height, self.width]


@dataclass(frozen=True)
class ProbeConfig:
    frames: int = 6
    height: int = 24
    width: int = 32
    classes: int = 5

    def shape(self) -> MaskShape:
        return MaskShape(
            frames=self.frames,
            height=self.height,
            width=self.width,
            classes=self.classes,
        )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_shape(shape: MaskShape) -> None:
    if shape.frames <= 0:
        raise ValueError(f"frames must be positive, got {shape.frames}")
    if shape.height <= 0:
        raise ValueError(f"height must be positive, got {shape.height}")
    if shape.width <= 0:
        raise ValueError(f"width must be positive, got {shape.width}")
    if not (1 <= shape.classes <= 256):
        raise ValueError(f"classes must be in [1, 256], got {shape.classes}")


def _validate_symbols(symbols: list[int], shape: MaskShape) -> None:
    _validate_shape(shape)
    if len(symbols) != shape.pixels:
        raise ValueError(f"symbol count mismatch: got {len(symbols)}, expected {shape.pixels}")
    for index, value in enumerate(symbols):
        if not (0 <= int(value) < shape.classes):
            raise ValueError(f"symbol at index {index} is outside [0, {shape.classes})")


def _encode_varuint(value: int) -> bytes:
    if value <= 0:
        raise ValueError(f"run length must be positive, got {value}")
    out = bytearray()
    cursor = int(value)
    while cursor >= 0x80:
        out.append((cursor & 0x7F) | 0x80)
        cursor >>= 7
    out.append(cursor)
    return bytes(out)


def _decode_varuint(data: bytes, offset: int) -> tuple[int, int]:
    shift = 0
    value = 0
    cursor = offset
    while True:
        if cursor >= len(data):
            raise ValueError("truncated varuint")
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            if value <= 0:
                raise ValueError("decoded zero run length")
            if cursor - offset > 1 and data[cursor - 1] == 0:
                raise ValueError("non-canonical varuint encoding")
            return value, cursor
        shift += 7
        if shift > 63:
            raise ValueError("varuint is too large")


def _rle_runs(symbols: list[int]) -> list[tuple[int, int]]:
    if not symbols:
        return []
    runs: list[tuple[int, int]] = []
    current = int(symbols[0])
    length = 1
    for raw in symbols[1:]:
        value = int(raw)
        if value == current:
            length += 1
        else:
            runs.append((length, current))
            current = value
            length = 1
    runs.append((length, current))
    return runs


def pack_fixed_width(values: list[int], *, bits_per_symbol: int) -> bytes:
    """Pack unsigned values MSB-first into a byte stream."""
    if not (1 <= bits_per_symbol <= 8):
        raise ValueError(f"bits_per_symbol must be in [1, 8], got {bits_per_symbol}")
    max_value = (1 << bits_per_symbol) - 1
    out = bytearray()
    current = 0
    bits = 0
    for index, raw in enumerate(values):
        value = int(raw)
        if not (0 <= value <= max_value):
            raise ValueError(f"value at index {index} does not fit in {bits_per_symbol} bits")
        current = (current << bits_per_symbol) | value
        bits += bits_per_symbol
        while bits >= 8:
            bits -= 8
            out.append((current >> bits) & 0xFF)
            current &= (1 << bits) - 1
    if bits:
        out.append((current << (8 - bits)) & 0xFF)
    return bytes(out)


def unpack_fixed_width(data: bytes, *, count: int, bits_per_symbol: int) -> list[int]:
    """Unpack MSB-first fixed-width unsigned values."""
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if not (1 <= bits_per_symbol <= 8):
        raise ValueError(f"bits_per_symbol must be in [1, 8], got {bits_per_symbol}")
    needed_bytes = (count * bits_per_symbol + 7) // 8
    if len(data) != needed_bytes:
        raise ValueError(f"bitpacked value byte count mismatch: got {len(data)}, expected {needed_bytes}")
    values: list[int] = []
    accumulator = 0
    bits = 0
    mask = (1 << bits_per_symbol) - 1
    for byte in data:
        accumulator = (accumulator << 8) | byte
        bits += 8
        while bits >= bits_per_symbol and len(values) < count:
            bits -= bits_per_symbol
            values.append((accumulator >> bits) & mask)
            accumulator &= (1 << bits) - 1
    if len(values) != count:
        raise ValueError(f"decoded {len(values)} values, expected {count}")
    if bits and accumulator != 0:
        raise ValueError("non-zero padding bits in bitpacked value stream")
    return values


def encode_mask_rle_bitpacked(symbols: list[int], shape: MaskShape) -> bytes:
    """Encode flat class-id masks as CMCP_RLE1 bytes."""
    _validate_symbols(symbols, shape)
    runs = _rle_runs(symbols)
    if len(runs) > 0xFFFFFFFF:
        raise ValueError("too many runs for CMCP_RLE1 header")

    lengths = bytearray()
    values: list[int] = []
    for length, value in runs:
        lengths.extend(_encode_varuint(length))
        values.append(value)

    value_bits = pack_fixed_width(values, bits_per_symbol=shape.bits_per_symbol)
    header = _HEADER.pack(
        MAGIC,
        VERSION,
        0,
        shape.classes - 1,
        shape.bits_per_symbol,
        shape.frames,
        shape.height,
        shape.width,
        len(runs),
    )
    body = header + bytes(lengths) + value_bits
    return body + _CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def decode_mask_rle_bitpacked(blob: bytes) -> tuple[MaskShape, list[int]]:
    """Decode CMCP_RLE1 bytes and return shape plus flat class-id symbols."""
    data = bytes(blob)
    if len(data) < _HEADER.size + _CRC.size:
        raise ValueError("truncated CMCP_RLE1 payload")
    body = data[:-_CRC.size]
    expected_crc = _CRC.unpack_from(data, len(body))[0]
    actual_crc = zlib.crc32(body) & 0xFFFFFFFF
    if expected_crc != actual_crc:
        raise ValueError(f"CRC mismatch: expected 0x{expected_crc:08x}, got 0x{actual_crc:08x}")

    (
        magic,
        version,
        flags,
        classes_minus_one,
        bits_per_symbol,
        frames,
        height,
        width,
        run_count,
    ) = _HEADER.unpack_from(body, 0)
    if magic != MAGIC:
        raise ValueError("bad CMCP_RLE1 magic")
    if version != VERSION:
        raise ValueError(f"unsupported CMCP_RLE1 version: {version}")
    if flags != 0:
        raise ValueError(f"unsupported CMCP_RLE1 flags: {flags}")
    shape = MaskShape(frames=frames, height=height, width=width, classes=classes_minus_one + 1)
    _validate_shape(shape)
    if bits_per_symbol != shape.bits_per_symbol:
        raise ValueError("bits_per_symbol does not match class count")
    if run_count > shape.pixels:
        raise ValueError("run count exceeds declared pixel count")

    cursor = _HEADER.size
    lengths: list[int] = []
    total = 0
    for _ in range(run_count):
        length, cursor = _decode_varuint(body, cursor)
        lengths.append(length)
        total += length
        if total > shape.pixels:
            raise ValueError("run lengths exceed declared pixel count")
    if total != shape.pixels:
        raise ValueError(f"run lengths total {total}, expected {shape.pixels}")

    values = unpack_fixed_width(body[cursor:], count=run_count, bits_per_symbol=bits_per_symbol)
    out: list[int] = []
    for length, value in zip(lengths, values, strict=True):
        if value >= shape.classes:
            raise ValueError(f"decoded class value {value} outside [0, {shape.classes})")
        out.extend([value] * length)
    return shape, out


def make_synthetic_masks(shape: MaskShape) -> list[int]:
    """Build deterministic road-like synthetic masks for local byte probes."""
    _validate_shape(shape)
    symbols: list[int] = []
    for frame in range(shape.frames):
        drift = frame % max(1, shape.width // 8)
        box_left = max(0, shape.width // 3 - drift)
        box_right = min(shape.width, box_left + max(2, shape.width // 6))
        box_top = shape.height // 3
        box_bottom = min(shape.height, box_top + max(2, shape.height // 5))
        for y in range(shape.height):
            for x in range(shape.width):
                if box_top <= y < box_bottom and box_left <= x < box_right and shape.classes > 1:
                    value = 1
                elif y >= (shape.height * 2) // 3 and shape.classes > 2:
                    value = 2
                elif x < max(1, shape.width // 8) and shape.classes > 3:
                    value = 3
                elif x >= shape.width - max(1, shape.width // 8) and shape.classes > 4:
                    value = 4
                else:
                    value = 0
                symbols.append(value)
    return symbols


def raw_bitpacked_bytes(shape: MaskShape) -> int:
    _validate_shape(shape)
    return math.ceil(shape.pixels * shape.bits_per_symbol / 8)


def build_probe_report(*, config: ProbeConfig, command: list[str]) -> tuple[dict[str, Any], bytes]:
    shape = config.shape()
    symbols = make_synthetic_masks(shape)
    source_bytes = bytes(symbols)
    payload = encode_mask_rle_bitpacked(symbols, shape)
    decoded_shape, decoded = decode_mask_rle_bitpacked(payload)
    roundtrip_ok = decoded_shape == shape and decoded == symbols
    if not roundtrip_ok:
        raise RuntimeError("CMCP_RLE1 round-trip failed")

    runs = _rle_runs(symbols)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "local_probe_only": True,
        "scorer_network_loaded": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "score_claim_warning": SCORE_CLAIM_WARNING,
        "command": list(command),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "source": {
            "kind": "deterministic_synthetic_masks",
            "shape": shape.as_list(),
            "classes": shape.classes,
            "pixels": shape.pixels,
            "class_id_u8_size_bytes": len(source_bytes),
            "class_id_u8_sha256": _sha256_bytes(source_bytes),
        },
        "codec": {
            "schema": PAYLOAD_SCHEMA,
            "magic_hex": MAGIC.hex(),
            "version": VERSION,
            "header_bytes": _HEADER.size,
            "crc_trailer_bytes": _CRC.size,
            "bits_per_symbol": shape.bits_per_symbol,
            "run_count": len(runs),
            "payload_size_bytes": len(payload),
            "payload_sha256": _sha256_bytes(payload),
            "raw_bitpacked_size_bytes": raw_bitpacked_bytes(shape),
            "raw_u8_size_bytes": len(source_bytes),
            "roundtrip_bit_exact": True,
            "score_claim": False,
            "promotion_eligible": False,
        },
    }
    report["byte_observations"] = {
        "payload_minus_raw_bitpacked_bytes": len(payload) - report["codec"]["raw_bitpacked_size_bytes"],
        "payload_minus_raw_u8_bytes": len(payload) - len(source_bytes),
        "empirical_only": True,
    }
    return report, payload


def run_probe(
    *,
    output_dir: Path,
    config: ProbeConfig,
    command: list[str],
    force: bool = False,
) -> dict[str, Any]:
    report, payload = build_probe_report(config=config, command=command)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is non-empty; pass --force to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    payload_path = output_dir / PAYLOAD_NAME
    report_path = output_dir / REPORT_NAME
    payload_path.write_bytes(payload)
    report["artifacts"] = {
        "payload": {
            "path": str(payload_path),
            "size_bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        },
        "manifest": {
            "path": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=_positive_int, default=ProbeConfig.frames)
    parser.add_argument("--height", type=_positive_int, default=ProbeConfig.height)
    parser.add_argument("--width", type=_positive_int, default=ProbeConfig.width)
    parser.add_argument("--classes", type=_positive_int, default=ProbeConfig.classes)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ProbeConfig(
        frames=args.frames,
        height=args.height,
        width=args.width,
        classes=args.classes,
    )
    report = run_probe(
        output_dir=args.output_dir,
        config=config,
        command=[Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])],
        force=args.force,
    )
    print(json.dumps(dataclasses.asdict(config), sort_keys=True))
    print(f"wrote {report['artifacts']['manifest']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
