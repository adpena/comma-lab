#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Empirical CMG2 mask-stream byte probe.

This tool is deliberately not a scorer and not a submission builder.  It takes
an already-decoded mask tensor, applies deterministic lossless transforms, and
measures compressed byte sizes against a charged baseline mask stream.  Any
winning variant still needs a real archive member, inflate runtime decoder, and
exact CUDA auth eval before it can support a score claim.
"""
from __future__ import annotations

import argparse
import bz2
import dataclasses
import hashlib
import json
import lzma
import platform
import sys
import time
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "cmg2_mask_codec_probe_v1"
EVIDENCE_GRADE = "empirical_byte_probe_only"
REPORT_NAME = "cmg2_mask_codec_probe_manifest.json"
BEST_PAYLOAD_NAME = "best_payload.cmg2probe"
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
SCORE_CLAIM_WARNING = (
    "No score claim is made by this probe. Byte wins here are lossless codec "
    "design evidence only; a deterministic archive, runtime decoder, and exact "
    "CUDA auth eval on the exact archive bytes are required before ranking or "
    "promotion."
)


@dataclass(frozen=True)
class ProbeConfig:
    decoded_mask_array: Path
    output_dir: Path
    baseline_mask_stream: Path | None = None
    baseline_bytes: int | None = None
    compressors: tuple[str, ...] = ("zlib9", "lzma9", "bz2_9")
    transforms: tuple[str, ...] = (
        "raw_u8",
        "symbol_packed3",
        "bitplane_packed3",
        "frame_xor_symbol_packed3",
        "pair_xor_symbol_packed3",
        "class_planes_packed5",
    )
    force: bool = False
    write_best_payload: bool = False


@dataclass(frozen=True)
class VariantBytes:
    transform: str
    uncompressed_bytes: bytes


@dataclass(frozen=True)
class CompressedVariant:
    variant_id: str
    transform: str
    compressor: str
    uncompressed_size_bytes: int
    compressed_size_bytes: int
    compressed_sha256: str
    seconds: float


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _load_masks(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"decoded mask array must have rank 3, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"decoded mask array must be uint8, got {arr.dtype}")
    if arr.size == 0:
        raise ValueError("decoded mask array is empty")
    max_value = int(arr.max())
    min_value = int(arr.min())
    if min_value < 0 or max_value > 7:
        raise ValueError(
            "CMG2 lossless probe expects class ids fitting 3 bits; "
            f"observed range [{min_value}, {max_value}]"
        )
    return np.ascontiguousarray(arr)


def _pack_symbol_bits(arr: np.ndarray, *, bits: int = 3) -> bytes:
    flat = arr.reshape(-1)
    planes = ((flat[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).astype(np.uint8, copy=False)
    return np.packbits(planes.reshape(-1), bitorder="little").tobytes()


def _pack_bitplanes(arr: np.ndarray, *, bits: int = 3) -> bytes:
    flat = arr.reshape(-1)
    return b"".join(
        np.packbits(((flat >> np.uint8(bit)) & np.uint8(1)).astype(np.uint8, copy=False), bitorder="little").tobytes()
        for bit in range(bits)
    )


def _frame_xor(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr)
    out[0] = arr[0]
    if arr.shape[0] > 1:
        np.bitwise_xor(arr[1:], arr[:-1], out=out[1:])
    return out


def _pair_xor(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    if arr.shape[0] > 1:
        np.bitwise_xor(arr[1::2], arr[0::2], out=out[1::2])
    return out


def _class_planes(arr: np.ndarray, *, classes: int) -> bytes:
    flat = arr.reshape(-1)
    return b"".join(
        np.packbits((flat == np.uint8(cls)).astype(np.uint8, copy=False), bitorder="little").tobytes()
        for cls in range(classes)
    )


def _build_transform(arr: np.ndarray, transform: str) -> VariantBytes:
    if transform == "raw_u8":
        return VariantBytes(transform=transform, uncompressed_bytes=arr.tobytes(order="C"))
    if transform == "symbol_packed3":
        return VariantBytes(transform=transform, uncompressed_bytes=_pack_symbol_bits(arr, bits=3))
    if transform == "bitplane_packed3":
        return VariantBytes(transform=transform, uncompressed_bytes=_pack_bitplanes(arr, bits=3))
    if transform == "frame_xor_symbol_packed3":
        return VariantBytes(transform=transform, uncompressed_bytes=_pack_symbol_bits(_frame_xor(arr), bits=3))
    if transform == "pair_xor_symbol_packed3":
        return VariantBytes(transform=transform, uncompressed_bytes=_pack_symbol_bits(_pair_xor(arr), bits=3))
    if transform == "class_planes_packed5":
        return VariantBytes(transform=transform, uncompressed_bytes=_class_planes(arr, classes=max(5, int(arr.max()) + 1)))
    raise ValueError(f"unknown transform: {transform}")


def _compressor(name: str) -> Callable[[bytes], bytes]:
    if name == "none":
        return bytes
    if name == "zlib6":
        return lambda data: zlib.compress(data, level=6)
    if name == "zlib9":
        return lambda data: zlib.compress(data, level=9)
    if name == "bz2_9":
        return lambda data: bz2.compress(data, compresslevel=9)
    if name == "lzma6":
        return lambda data: lzma.compress(data, preset=6 | lzma.PRESET_EXTREME)
    if name == "lzma9":
        return lambda data: lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
    if name == "brotli11":
        try:
            import brotli  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional env
            raise ValueError("compressor brotli11 requested but brotli is unavailable") from exc

        return lambda data: brotli.compress(data, quality=11, mode=brotli.MODE_GENERIC)
    raise ValueError(f"unknown compressor: {name}")


def _available_compressors(names: Iterable[str]) -> tuple[str, ...]:
    available: list[str] = []
    for name in names:
        _compressor(name)
        available.append(name)
    return tuple(available)


def _baseline_record(config: ProbeConfig) -> dict[str, Any] | None:
    if config.baseline_mask_stream is None and config.baseline_bytes is None:
        return None
    record: dict[str, Any] = {
        "role": "charged_current_mask_stream",
        "bytes": int(config.baseline_bytes) if config.baseline_bytes is not None else None,
    }
    if config.baseline_mask_stream is not None:
        path = config.baseline_mask_stream.resolve()
        if not path.exists():
            raise FileNotFoundError(f"baseline mask stream does not exist: {path}")
        size = path.stat().st_size
        if config.baseline_bytes is not None and int(config.baseline_bytes) != size:
            raise ValueError(
                f"baseline byte mismatch: --baseline-bytes={config.baseline_bytes} "
                f"but {path} is {size} bytes"
            )
        record.update(
            {
                "path": str(path),
                "bytes": size,
                "sha256": _sha256_file(path),
            }
        )
    return record


def run_probe(config: ProbeConfig, *, command: list[str]) -> dict[str, Any]:
    output_dir = config.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not config.force:
        raise FileExistsError(f"output directory is non-empty; pass --force to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    compressors = _available_compressors(config.compressors)
    arr = _load_masks(config.decoded_mask_array.resolve())
    source_sha = _sha256_bytes(arr.tobytes(order="C"))
    npy_sha = _sha256_file(config.decoded_mask_array.resolve())
    baseline = _baseline_record(config)
    baseline_bytes = baseline.get("bytes") if baseline else None

    variants: list[CompressedVariant] = []
    best_payload: bytes | None = None
    best_variant_id: str | None = None
    for transform in config.transforms:
        transformed = _build_transform(arr, transform)
        for compressor_name in compressors:
            compressor = _compressor(compressor_name)
            t0 = time.monotonic()
            compressed = compressor(transformed.uncompressed_bytes)
            elapsed = time.monotonic() - t0
            variant_id = f"{transform}_{compressor_name}"
            variant = CompressedVariant(
                variant_id=variant_id,
                transform=transform,
                compressor=compressor_name,
                uncompressed_size_bytes=len(transformed.uncompressed_bytes),
                compressed_size_bytes=len(compressed),
                compressed_sha256=_sha256_bytes(compressed),
                seconds=elapsed,
            )
            variants.append(variant)
            if best_payload is None or len(compressed) < len(best_payload):
                best_payload = compressed
                best_variant_id = variant_id

    variants_sorted = sorted(variants, key=lambda v: (v.compressed_size_bytes, v.variant_id))
    payload_record: dict[str, Any] | None = None
    if config.write_best_payload and best_payload is not None and best_variant_id is not None:
        payload_path = output_dir / BEST_PAYLOAD_NAME
        payload_path.write_bytes(best_payload)
        payload_record = {
            "path": str(payload_path),
            "variant_id": best_variant_id,
            "bytes": len(best_payload),
            "sha256": _sha256_bytes(best_payload),
            "runtime_ready": False,
            "reason": "probe payload lacks a reviewed CMG2 archive/runtime envelope",
        }

    def _variant_json(v: CompressedVariant) -> dict[str, Any]:
        entry: dict[str, Any] = dataclasses.asdict(v)
        if baseline_bytes is not None:
            entry["baseline_delta_bytes"] = int(v.compressed_size_bytes) - int(baseline_bytes)
            entry["beats_baseline"] = int(v.compressed_size_bytes) < int(baseline_bytes)
        return entry

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": "experiments/probe_cmg2_mask_codecs.py",
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
            "numpy": np.__version__,
        },
        "source": {
            "decoded_mask_array_path": str(config.decoded_mask_array.resolve()),
            "decoded_mask_array_npy_sha256": npy_sha,
            "decoded_mask_tensor_sha256": source_sha,
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
            "class_min": int(arr.min()),
            "class_max": int(arr.max()),
            "raw_u8_bytes": int(arr.nbytes),
            "pixel_count": int(arr.size),
        },
        "baseline": baseline,
        "probe_config": {
            "transforms": list(config.transforms),
            "compressors": list(compressors),
            "write_best_payload": bool(config.write_best_payload),
        },
        "variants": [_variant_json(v) for v in variants_sorted],
        "best_variant": _variant_json(variants_sorted[0]) if variants_sorted else None,
        "artifacts": {
            "manifest": {
                "path": str(output_dir / REPORT_NAME),
            },
            "best_payload": payload_record,
        },
        "required_next_steps_for_score_claim": [
            "define a reviewed CMG2 archive member envelope and deterministic decoder",
            "add archive validator and local-smoke allowlist parity for the new suffix",
            "build an exact archive that carries every score-affecting byte",
            "run exact CUDA auth eval on the exact archive bytes",
        ],
    }
    (output_dir / REPORT_NAME).write_text(_canonical_json(report))
    return report


def _csv(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one comma-separated value")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-mask-array", type=Path, required=True)
    parser.add_argument("--baseline-mask-stream", type=Path)
    parser.add_argument("--baseline-bytes", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compressors", type=_csv, default=ProbeConfig.compressors)
    parser.add_argument("--transforms", type=_csv, default=ProbeConfig.transforms)
    parser.add_argument("--write-best-payload", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ProbeConfig(
        decoded_mask_array=args.decoded_mask_array,
        baseline_mask_stream=args.baseline_mask_stream,
        baseline_bytes=args.baseline_bytes,
        output_dir=args.output_dir,
        compressors=tuple(args.compressors),
        transforms=tuple(args.transforms),
        write_best_payload=bool(args.write_best_payload),
        force=bool(args.force),
    )
    report = run_probe(
        config,
        command=[Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])],
    )
    best = report["best_variant"]
    print(
        json.dumps(
            {
                "manifest": report["artifacts"]["manifest"]["path"],
                "best_variant_id": None if best is None else best["variant_id"],
                "best_bytes": None if best is None else best["compressed_size_bytes"],
                "baseline_delta_bytes": None if best is None else best.get("baseline_delta_bytes"),
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
