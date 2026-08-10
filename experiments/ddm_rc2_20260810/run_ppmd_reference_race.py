#!/usr/bin/env python3
"""Retained PPMd adaptive-arithmetic race on the four PR130 wire sections.

This is a byte race, not an entropy projection.  Every candidate packet and
every decoded output is written atomically before its byte count enters the
manifest.  The packet header counts the decoder parameters selected by the
race; the generic PPMd algorithm remains rule-118-free receiver code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import platform
import struct
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyppmd  # PYPPMD_LGPL_OK:bounded-research-race-retains-wire-payloads-no-package-runtime

AXIS = "[macOS-CPU advisory, scorer-free]"
SCORE_CLAIM = False
SCHEMA = "ddm_rc2_ppmd_reference_race.v1"
PACKET_MAGIC = b"RC2P"
BASE_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip")
EXPECTED_BASE_SHA256 = "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
DEFAULT_OUTPUT = Path("/Volumes/APDataStore/pact/ddm_rc2_20260810/ppmd_reference")
MIN_FREE_BYTES = 1 << 30
SEED = 0  # PPMd is deterministic; retained for the campaign-wide seed contract.


@dataclass(frozen=True)
class Section:
    name: str
    source: bytes
    incumbent_bytes: int
    incumbent: str
    memoryless_bound_bytes: int


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_bytes(path, encoded)


def storage_preflight(output: Path) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    stats = os.statvfs(output)
    free = stats.f_bavail * stats.f_frsize
    total = stats.f_blocks * stats.f_frsize
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"storage preflight failed: {free} free bytes < {MIN_FREE_BYTES} required")
    return {"free_bytes": free, "total_bytes": total, "required_free_bytes": MIN_FREE_BYTES}


def extract_sections(archive: Path) -> list[Section]:
    if sha256_file(archive) != EXPECTED_BASE_SHA256:
        raise RuntimeError("PR130 archive SHA-256 does not match the charter pin")
    with zipfile.ZipFile(archive) as bundle:
        member = bundle.read("p")
    model_bytes = struct.unpack_from("<I", member, 0)[0]
    compressed_models = member[4 : 4 + model_bytes]
    tokens_wire = member[4 + model_bytes :]
    models_raw = lzma.decompress(compressed_models)
    semantic_len, pose_len = struct.unpack_from("<II", models_raw, 0)
    semantic_start = 8
    pose_start = semantic_start + semantic_len
    hpac_start = pose_start + pose_len
    semantic = models_raw[semantic_start:pose_start]
    pose = models_raw[pose_start:hpac_start]
    hpac = models_raw[hpac_start:]
    expected = {
        "tokens_wire": 116_980,
        "semantic_raw": 40_252,
        "pose_raw": 23_054,
        "hpac_raw": 20_179,
    }
    actual = {
        "tokens_wire": len(tokens_wire),
        "semantic_raw": len(semantic),
        "pose_raw": len(pose),
        "hpac_raw": len(hpac),
    }
    if actual != expected:
        raise RuntimeError(f"PR130 section lengths drifted: {actual!r}")
    return [
        Section("tokens_wire", tokens_wire, 114_860, "retained ANS under existing HPAC model", 114_852),
        Section("semantic_raw", semantic, 35_033, "Brotli q11", 36_805),
        Section("pose_raw", pose, 23_054, "shipped canonical Huffman representation", 22_989),
        Section("hpac_raw", hpac, 14_962, "Brotli q11", 16_567),
    ]


def packetize(payload: bytes, *, variant: str, order: int, mem_size: int, source_bytes: int) -> bytes:
    if variant not in {"H", "I"}:
        raise ValueError(f"unsupported PPMd variant {variant!r}")
    if mem_size <= 0 or mem_size & (mem_size - 1):
        raise ValueError("mem_size must be a positive power of two")
    log2_mem = mem_size.bit_length() - 1
    if not 0 <= order <= 255 or not 0 <= log2_mem <= 255:
        raise ValueError("packet parameter does not fit the counted header")
    return (
        PACKET_MAGIC + variant.encode("ascii") + bytes((order, log2_mem, 0)) + struct.pack("<I", source_bytes) + payload
    )


def depacketize(packet: bytes) -> tuple[bytes, str, int, int, int]:
    if len(packet) < 12 or packet[:4] != PACKET_MAGIC:
        raise ValueError("invalid RC2 PPMd packet")
    variant = packet[4:5].decode("ascii")
    order = packet[5]
    mem_size = 1 << packet[6]
    source_bytes = struct.unpack_from("<I", packet, 8)[0]
    return packet[12:], variant, order, mem_size, source_bytes


def decompress_packet(packet: bytes) -> bytes:
    body, variant, order, memory, source_bytes = depacketize(packet)
    if variant == "H":
        decoder = pyppmd.Ppmd7Decoder(max_order=order, mem_size=memory)
        decoded = decoder.decode(body, source_bytes)
    else:
        decoded = pyppmd.decompress(
            body,
            max_order=order,
            mem_size=memory,
            variant=variant,
        )
    if not isinstance(decoded, bytes):
        raise RuntimeError("PPMd returned text for a byte input")
    return decoded


def candidate_id(variant: str, order: int, mem_size: int) -> str:
    return f"variant_{variant}_order_{order:02d}_mem_{mem_size >> 20:02d}mib"


def completed_candidate(row: dict[str, Any], source_sha: str) -> bool:
    try:
        packet = Path(row["packet_path"])
        if not (
            row.get("complete") is True
            and row.get("source_sha256") == source_sha
            and packet.is_file()
            and packet.stat().st_size == row["packet_bytes"]
            and sha256_file(packet) == row["packet_sha256"]
        ):
            return False
        if "decoded_path" in row:
            decoded = Path(row["decoded_path"])
            if not (
                decoded.is_file()
                and decoded.stat().st_size == row["decoded_bytes"]
                and sha256_file(decoded) == row["decoded_sha256"]
            ):
                return False
            return not row.get("exact_decode") or row["decoded_sha256"] == source_sha
        error_path = Path(row["decode_error_path"])
        return error_path.is_file() and sha256_file(error_path) == row["decode_error_sha256"]
    except (KeyError, OSError, TypeError):
        return False


def initial_manifest(output: Path, storage: dict[str, int]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "complete": False,
        "seed": SEED,
        "base_archive": {
            "path": str(BASE_ARCHIVE),
            "bytes": BASE_ARCHIVE.stat().st_size,
            "sha256": sha256_file(BASE_ARCHIVE),
        },
        "storage_preflight": storage,
        "host": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "pyppmd": getattr(pyppmd, "__version__", "1.3.1"),
        },
        "rule_118": {
            "free_receiver_code": "generic PPMd algorithm and packet parser",
            "counted_payload": "PPMd stream plus 12-byte per-candidate parameter/length header",
        },
        "sections": {},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    storage = storage_preflight(args.output)
    manifest_path = args.output / "ppmd_reference_manifest.json"
    manifest = initial_manifest(args.output, storage)
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("schema") == SCHEMA:
            manifest["sections"] = prior.get("sections", {})
    sections = extract_sections(BASE_ARCHIVE)
    for section in sections:
        section_dir = args.output / section.name
        source_path = section_dir / "source.bin"
        source_sha = sha256_bytes(section.source)
        if not source_path.exists() or sha256_file(source_path) != source_sha:
            atomic_bytes(source_path, section.source)
        section_row = manifest["sections"].setdefault(
            section.name,
            {
                "complete": False,
                "source_path": str(source_path),
                "source_bytes": len(section.source),
                "source_sha256": source_sha,
                "incumbent": section.incumbent,
                "incumbent_bytes": section.incumbent_bytes,
                "memoryless_bound_bytes": section.memoryless_bound_bytes,
                "candidates": {},
            },
        )
        section_row.update(
            {
                "source_path": str(source_path),
                "source_bytes": len(section.source),
                "source_sha256": source_sha,
                "incumbent": section.incumbent,
                "incumbent_bytes": section.incumbent_bytes,
                "memoryless_bound_bytes": section.memoryless_bound_bytes,
            }
        )
        for variant in args.variants:
            for order in args.orders:
                for mem_mib in args.memory_mib:
                    mem_size = mem_mib << 20
                    name = candidate_id(variant, order, mem_size)
                    old = section_row["candidates"].get(name, {})
                    if completed_candidate(old, source_sha):
                        continue
                    candidate_dir = section_dir / name
                    packet_path = candidate_dir / "stream.rc2p"
                    decoded_path = candidate_dir / "decoded.bin"
                    started = time.perf_counter()
                    compressed = pyppmd.compress(
                        section.source,
                        max_order=order,
                        mem_size=mem_size,
                        variant=variant,
                    )
                    packet = packetize(
                        compressed,
                        variant=variant,
                        order=order,
                        mem_size=mem_size,
                        source_bytes=len(section.source),
                    )
                    atomic_bytes(packet_path, packet)
                    encode_s = time.perf_counter() - started
                    decode_started = time.perf_counter()
                    try:
                        decoded = decompress_packet(packet_path.read_bytes())
                    except Exception as error:  # Candidate-invalid result, not a race-wide blocker.
                        decode_s = time.perf_counter() - decode_started
                        error_path = candidate_dir / "decode_error.json"
                        atomic_json(
                            error_path,
                            {
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "section": section.name,
                                "candidate": name,
                            },
                        )
                        section_row["candidates"][name] = {
                            "complete": True,
                            "variant": variant,
                            "max_order": order,
                            "mem_size_bytes": mem_size,
                            "source_sha256": source_sha,
                            "packet_path": str(packet_path),
                            "packet_bytes": packet_path.stat().st_size,
                            "packet_sha256": sha256_file(packet_path),
                            "decode_error_path": str(error_path),
                            "decode_error_sha256": sha256_file(error_path),
                            "exact_decode": False,
                            "encode_s": encode_s,
                            "decode_s": decode_s,
                            "delta_vs_incumbent_bytes": packet_path.stat().st_size - section.incumbent_bytes,
                        }
                        atomic_json(manifest_path, manifest)
                        continue
                    decode_s = time.perf_counter() - decode_started
                    atomic_bytes(decoded_path, decoded)
                    decoded_sha = sha256_file(decoded_path)
                    exact_decode = decoded_sha == source_sha and decoded == section.source
                    first_mismatch = None
                    if not exact_decode:
                        common = min(len(decoded), len(section.source))
                        first_mismatch = next(
                            (index for index in range(common) if decoded[index] != section.source[index]),
                            common,
                        )
                    row = {
                        "complete": True,
                        "variant": variant,
                        "max_order": order,
                        "mem_size_bytes": mem_size,
                        "source_sha256": source_sha,
                        "packet_path": str(packet_path),
                        "packet_bytes": packet_path.stat().st_size,
                        "packet_sha256": sha256_file(packet_path),
                        "decoded_path": str(decoded_path),
                        "decoded_bytes": decoded_path.stat().st_size,
                        "decoded_sha256": decoded_sha,
                        "exact_decode": exact_decode,
                        "first_mismatch": first_mismatch,
                        "encode_s": encode_s,
                        "decode_s": decode_s,
                        "delta_vs_incumbent_bytes": packet_path.stat().st_size - section.incumbent_bytes,
                    }
                    section_row["candidates"][name] = row
                    atomic_json(manifest_path, manifest)
        valid_candidates = [item for item in section_row["candidates"].items() if item[1]["exact_decode"]]
        if not valid_candidates:
            raise RuntimeError(f"no exact-decode PPMd candidate survived for {section.name}")
        best_name, best = min(valid_candidates, key=lambda item: item[1]["packet_bytes"])
        section_row["best_candidate"] = best_name
        section_row["best_packet_bytes"] = best["packet_bytes"]
        section_row["best_delta_vs_incumbent_bytes"] = best["delta_vs_incumbent_bytes"]
        section_row["complete"] = True
        atomic_json(manifest_path, manifest)
    manifest["complete"] = True
    manifest["completed_at_unix_s"] = time.time()
    atomic_json(manifest_path, manifest)
    return manifest


def self_test(output: Path) -> None:
    storage_preflight(output)
    source = (b"abracadabra:" * 53) + bytes(range(64))
    atomic_bytes(output / "self_test_source.bin", source)
    for variant in ("H", "I"):
        payload = pyppmd.compress(source, max_order=4, mem_size=1 << 20, variant=variant)
        packet = packetize(
            payload,
            variant=variant,
            order=4,
            mem_size=1 << 20,
            source_bytes=len(source),
        )
        packet_path = output / f"self_test_{variant}.rc2p"
        atomic_bytes(packet_path, packet)
        decoded = decompress_packet(packet_path.read_bytes())
        if decoded != source:
            raise RuntimeError(f"self-test decode failed for variant {variant}")
        atomic_bytes(output / f"self_test_{variant}.decoded.bin", decoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 4, 6, 8, 12, 16])
    parser.add_argument("--memory-mib", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--variants", nargs="+", choices=("H", "I"), default=["H", "I"])
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(args.output)
        print(json.dumps({"self_test": "passed", "output": str(args.output)}, sort_keys=True))
        return
    result = run(args)
    summary = {
        name: {
            "best_candidate": row["best_candidate"],
            "best_packet_bytes": row["best_packet_bytes"],
            "best_delta_vs_incumbent_bytes": row["best_delta_vs_incumbent_bytes"],
        }
        for name, row in result["sections"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
