#!/usr/bin/env python3
"""Receiver-closed ET1 implicit generator-tree byte gate on the exact DX2 object.

This apparatus prices a concrete, narrow V8/V9 formulation without materializing
boundaries or a dense cell payload.  A deterministic binary space-time partition
is traversed from the root.  Homogeneous leaves are class-owned *generators*:
the receiver fills their implicit evaluator cells with one of five class labels.
Internal topology is shared, so every derived inter-class edge has one home.

The source is the retained categorical field decoded from the exact DX2 archive.
The apparatus races deterministic split/traversal policies and real lossless
coders, retains every raw and coded payload (plus deterministic repeats), and
parse-backs each coded stream.  The best complete archive also retains the exact
decoded field and byte-identical inherited DX2 renderer/carrier/residual sections.

This is intentionally scorer-free and GPU/Metal-free.  It does not claim to be
the entire heterogeneous analytic V8 family; its verdict scope is the exact
implicit BSP generator-tree formulation implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import struct
import sys
import time
import zipfile
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import brotli
import numpy as np

DX2_ARCHIVE_SHA256 = "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
DX2_ARCHIVE_BYTES = 180_368
DX2_TOKEN_FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
TOKEN_SHAPE = (600, 384, 512)
TOKEN_BYTES = int(np.prod(TOKEN_SHAPE))
NUM_CLASSES = 5

RX1_HEADER = struct.Struct("<4sBBBBHHH")
RX1_MAGIC = b"RX1M"
EXPECTED_HPAC_BYTES = 13_515
EXPECTED_SEMANTIC_BYTES = 30_856
EXPECTED_CARRIER_BYTES = 22_010
EXPECTED_RESIDUAL_BYTES = 96
EXPECTED_TOKEN_STREAM_BYTES = 113_777
EXPECTED_SECTION_SHA256 = {
    "hpac_model": "602115b323b0e403d08287af9b273a2d4fb23e026d83c1f6e4609ed77ef98f98",
    "semantic_renderer": "39d1be52ba62933498395c48ce4d9482f37db097d504da76c2a321efe3e4a76f",
    "pose_carrier": "932b979f5181b331a9099162c6f392f558860b7998c62a36f38c2c99629c9b12",
    "compact_residual": "8ab2fe748ab7d69d2102ba2292289e22bd7ea503f8ae29938e0854ec46ca3da1",
    "token_stream": "e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5",
}

TREE_MAGIC = b"ET1G"
TREE_VERSION = 1
TREE_HEADER = struct.Struct("<4s8BHHHQII")
COMPLETE_MAGIC = b"ET1C"
COMPLETE_VERSION = 1
COMPLETE_HEADER = struct.Struct("<4sBIIII")

POLICY_IDS = {"time_first": 1, "space_first": 2, "balanced_tyx": 3}
POLICY_NAMES = {value: key for key, value in POLICY_IDS.items()}
CODER_IDS = {"brotli_q11": 1, "zlib_9": 2, "lzma2_extreme": 3}
CODER_NAMES = {value: key for key, value in CODER_IDS.items()}


class ET1Error(RuntimeError):
    """Raised when a source, container, or parse-back invariant fails."""


def sha256_path(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_sources(source_archive: Path, source_tokens: Path) -> None:
    if not source_archive.is_file() or not source_tokens.is_file():
        raise FileNotFoundError("the retained DX2 archive and token field are both required")
    archive_fact = file_fact(source_archive)
    if archive_fact["bytes"] != DX2_ARCHIVE_BYTES or archive_fact["sha256"] != DX2_ARCHIVE_SHA256:
        raise ET1Error(f"DX2 archive identity mismatch: {archive_fact}")
    token_fact = file_fact(source_tokens)
    if token_fact["bytes"] != TOKEN_BYTES or token_fact["sha256"] != DX2_TOKEN_FIELD_SHA256:
        raise ET1Error(f"DX2 token-field identity mismatch: {token_fact}")


def extract_dx2_sections(source_archive: Path, retained_dir: Path) -> dict[str, Path]:
    """Retain and return every physical RX1 section from the exact DX2 member."""

    retained_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_archive) as archive:
        if archive.namelist() != ["p"]:
            raise ET1Error("DX2 archive must contain exactly the stored member 'p'")
        raw = archive.read("p")
    if len(raw) != DX2_ARCHIVE_BYTES - 100:
        raise ET1Error(f"unexpected DX2 member bytes: {len(raw)}")
    if len(raw) < RX1_HEADER.size:
        raise ET1Error("truncated RX1 member")
    magic, version, _codec, _mode, _reserved, hpac_n, semantic_n, carrier_n = RX1_HEADER.unpack_from(raw)
    if magic != RX1_MAGIC or version != 1:
        raise ET1Error("source member is not canonical RX1M v1")
    if (hpac_n, semantic_n, carrier_n) != (
        EXPECTED_HPAC_BYTES,
        EXPECTED_SEMANTIC_BYTES,
        EXPECTED_CARRIER_BYTES,
    ):
        raise ET1Error("RX1 section lengths disagree with the AR1B decomposition")
    cursor = RX1_HEADER.size
    sections: dict[str, bytes] = {}
    for name, length in (
        ("hpac_model", hpac_n),
        ("semantic_renderer", semantic_n),
        ("pose_carrier", carrier_n),
        ("compact_residual", EXPECTED_RESIDUAL_BYTES),
    ):
        sections[name] = raw[cursor : cursor + length]
        cursor += length
    sections["token_stream"] = raw[cursor:]
    if len(sections["token_stream"]) != EXPECTED_TOKEN_STREAM_BYTES:
        raise ET1Error("RX1 token-stream length disagrees with the AR1B decomposition")

    paths: dict[str, Path] = {}
    for name, payload in sections.items():
        digest = hashlib.sha256(payload).hexdigest()
        if digest != EXPECTED_SECTION_SHA256[name]:
            raise ET1Error(f"{name} SHA-256 mismatch: {digest}")
        path = retained_dir / f"source_{name}.bin"
        atomic_bytes(path, payload)
        paths[name] = path
    return paths


class ThreeBitWriter:
    """Streaming little-endian writer for the six-symbol tree event alphabet."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.word = 0
        self.bits = 0
        self.events = 0

    def write(self, symbol: int) -> None:
        if symbol < 0 or symbol > NUM_CLASSES:
            raise ET1Error(f"tree symbol outside [0,{NUM_CLASSES}]: {symbol}")
        self.word |= int(symbol) << self.bits
        self.bits += 3
        self.events += 1
        while self.bits >= 8:
            self.stream.write(bytes((self.word & 0xFF,)))
            self.word >>= 8
            self.bits -= 8

    def finish(self) -> None:
        if self.bits:
            self.stream.write(bytes((self.word & 0xFF,)))
        self.word = 0
        self.bits = 0


class ThreeBitReader:
    """Streaming inverse of :class:`ThreeBitWriter`."""

    def __init__(self, payload: bytes, event_count: int) -> None:
        self.payload = payload
        self.event_count = int(event_count)
        self.offset = 0
        self.word = 0
        self.bits = 0
        self.events = 0

    def read(self) -> int:
        if self.events >= self.event_count:
            raise ET1Error("tree event stream exhausted")
        while self.bits < 3:
            if self.offset >= len(self.payload):
                raise ET1Error("truncated three-bit tree stream")
            self.word |= self.payload[self.offset] << self.bits
            self.offset += 1
            self.bits += 8
        value = self.word & 7
        self.word >>= 3
        self.bits -= 3
        self.events += 1
        if value > NUM_CLASSES:
            raise ET1Error(f"reserved tree event symbol {value}")
        return value

    def finish(self) -> None:
        if self.events != self.event_count:
            raise ET1Error("tree decoder left unread events")
        expected = (self.event_count * 3 + 7) // 8
        if self.offset != expected:
            raise ET1Error("tree decoder left unread payload bytes")
        padding = (8 - (self.event_count * 3) % 8) % 8
        if padding and self.payload[-1] >> (8 - padding):
            raise ET1Error("tree stream has nonzero padding")


Box = tuple[int, int, int, int, int, int]


def choose_axis(box: Box, policy: str) -> int:
    t0, t1, y0, y1, x0, x1 = box
    lengths = (t1 - t0, y1 - y0, x1 - x0)
    available = [axis for axis, length in enumerate(lengths) if length > 1]
    if not available:
        raise ET1Error("non-homogeneous unit cell has no split axis")
    if policy == "time_first":
        if lengths[0] > 1:
            return 0
        return max((1, 2), key=lambda axis: (lengths[axis], -axis))
    if policy == "space_first":
        spatial = [axis for axis in (1, 2) if lengths[axis] > 1]
        if spatial:
            return max(spatial, key=lambda axis: (lengths[axis], -axis))
        return 0
    if policy == "balanced_tyx":
        full = TOKEN_SHAPE
        return max(available, key=lambda axis: (lengths[axis] / full[axis], -axis))
    raise ET1Error(f"unknown split policy: {policy}")


def split_box(box: Box, axis: int) -> tuple[Box, Box]:
    bounds = list(box)
    lo_index = axis * 2
    hi_index = lo_index + 1
    midpoint = (bounds[lo_index] + bounds[hi_index]) // 2
    lower = bounds.copy()
    upper = bounds.copy()
    lower[hi_index] = midpoint
    upper[lo_index] = midpoint
    return tuple(lower), tuple(upper)  # type: ignore[return-value]


def block_view(tokens: np.ndarray, box: Box) -> np.ndarray:
    t0, t1, y0, y1, x0, x1 = box
    return tokens[t0:t1, y0:y1, x0:x1]


def encode_tree(tokens: np.ndarray, policy: str, raw_path: Path) -> dict[str, object]:
    """Encode exact class generators; boundaries never enter the payload."""

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_name(f".{raw_path.name}.{os.getpid()}.tmp")
    leaf_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    internal_count = 0
    started = time.monotonic()
    with temporary.open("wb") as stream:
        writer = ThreeBitWriter(stream)
        stack: list[Box] = [(0, TOKEN_SHAPE[0], 0, TOKEN_SHAPE[1], 0, TOKEN_SHAPE[2])]
        while stack:
            box = stack.pop()
            block = block_view(tokens, box)
            first = int(block.flat[0])
            if np.all(block == first):
                writer.write(first + 1)
                leaf_counts[first] += 1
                continue
            writer.write(0)
            internal_count += 1
            lower, upper = split_box(box, choose_axis(box, policy))
            stack.append(upper)
            stack.append(lower)
        writer.finish()
        event_count = writer.events
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, raw_path)
    expected_bytes = (event_count * 3 + 7) // 8
    if raw_path.stat().st_size != expected_bytes:
        raise ET1Error("three-bit encoder length invariant failed")
    return {
        "policy": policy,
        "event_count": event_count,
        "internal_count": internal_count,
        "leaf_count": int(leaf_counts.sum()),
        "leaf_counts_by_class": leaf_counts.tolist(),
        "seconds": time.monotonic() - started,
        "raw_payload": file_fact(raw_path),
    }


def compress_payload(raw: bytes, coder: str) -> bytes:
    if coder == "brotli_q11":
        return brotli.compress(raw, mode=brotli.MODE_GENERIC, quality=11, lgwin=24)
    if coder == "zlib_9":
        return zlib.compress(raw, level=9)
    if coder == "lzma2_extreme":
        return lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    raise ET1Error(f"unknown coder: {coder}")


def decompress_payload(payload: bytes, coder: str) -> bytes:
    if coder == "brotli_q11":
        return brotli.decompress(payload)
    if coder == "zlib_9":
        return zlib.decompress(payload)
    if coder == "lzma2_extreme":
        return lzma.decompress(payload, format=lzma.FORMAT_XZ)
    raise ET1Error(f"unknown coder: {coder}")


def tree_packet(policy: str, coder: str, event_count: int, raw_bytes: int, coded: bytes) -> bytes:
    # Identity class map is explicit and makes later alphabet races receiver-closed.
    class_map = bytes(range(NUM_CLASSES))
    header = TREE_HEADER.pack(
        TREE_MAGIC,
        TREE_VERSION,
        POLICY_IDS[policy],
        CODER_IDS[coder],
        *class_map,
        *TOKEN_SHAPE,
        int(event_count),
        int(raw_bytes),
        len(coded),
    )
    return header + coded


@dataclass(frozen=True)
class ParsedTreePacket:
    policy: str
    coder: str
    class_map: tuple[int, ...]
    shape: tuple[int, int, int]
    event_count: int
    raw_bytes: int
    coded: bytes


def parse_tree_packet(packet: bytes) -> ParsedTreePacket:
    if len(packet) < TREE_HEADER.size:
        raise ET1Error("truncated ET1 generator-tree packet")
    unpacked = TREE_HEADER.unpack_from(packet)
    magic, version, policy_id, coder_id = unpacked[:4]
    class_map = tuple(int(value) for value in unpacked[4:9])
    shape = tuple(int(value) for value in unpacked[9:12])
    event_count, raw_bytes, coded_bytes = (int(value) for value in unpacked[12:15])
    if magic != TREE_MAGIC or version != TREE_VERSION:
        raise ET1Error("unsupported ET1 generator-tree packet")
    if policy_id not in POLICY_NAMES or coder_id not in CODER_NAMES:
        raise ET1Error("ET1 packet names an unknown policy or coder")
    if sorted(class_map) != list(range(NUM_CLASSES)) or shape != TOKEN_SHAPE:
        raise ET1Error("ET1 packet class map or field shape is non-canonical")
    coded = packet[TREE_HEADER.size :]
    if len(coded) != coded_bytes:
        raise ET1Error("ET1 coded payload length mismatch")
    if raw_bytes != (event_count * 3 + 7) // 8:
        raise ET1Error("ET1 raw tree length is inconsistent with event count")
    return ParsedTreePacket(
        policy=POLICY_NAMES[policy_id],
        coder=CODER_NAMES[coder_id],
        class_map=class_map,
        shape=shape,
        event_count=event_count,
        raw_bytes=raw_bytes,
        coded=coded,
    )


def decode_tree_packet_to_file(packet: bytes, output_path: Path) -> dict[str, object]:
    parsed = parse_tree_packet(packet)
    raw = decompress_payload(parsed.coded, parsed.coder)
    if len(raw) != parsed.raw_bytes:
        raise ET1Error("ET1 coder parse-back produced the wrong raw length")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    output = np.memmap(temporary, mode="w+", dtype=np.uint8, shape=TOKEN_SHAPE)
    reader = ThreeBitReader(raw, parsed.event_count)
    stack: list[Box] = [(0, TOKEN_SHAPE[0], 0, TOKEN_SHAPE[1], 0, TOKEN_SHAPE[2])]
    while stack:
        box = stack.pop()
        symbol = reader.read()
        if symbol:
            label = parsed.class_map[symbol - 1]
            t0, t1, y0, y1, x0, x1 = box
            output[t0:t1, y0:y1, x0:x1] = label
            continue
        lower, upper = split_box(box, choose_axis(box, parsed.policy))
        stack.append(upper)
        stack.append(lower)
    reader.finish()
    output.flush()
    del output
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    return file_fact(output_path)


def build_complete_archive(
    output_path: Path,
    topology_packet: bytes,
    semantic: bytes,
    carrier: bytes,
    residual: bytes,
) -> None:
    member = COMPLETE_HEADER.pack(
        COMPLETE_MAGIC,
        COMPLETE_VERSION,
        len(semantic),
        len(carrier),
        len(residual),
        len(topology_packet),
    ) + semantic + carrier + residual + topology_packet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(temporary, mode="w") as archive:
        archive.writestr(info, member)
    os.replace(temporary, output_path)


def parse_complete_archive(archive_path: Path) -> tuple[dict[str, bytes], bytes]:
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["p"]:
            raise ET1Error("ET1 complete archive must contain exactly member 'p'")
        member = archive.read("p")
    if len(member) < COMPLETE_HEADER.size:
        raise ET1Error("truncated ET1 complete archive")
    magic, version, semantic_n, carrier_n, residual_n, topology_n = COMPLETE_HEADER.unpack_from(member)
    if magic != COMPLETE_MAGIC or version != COMPLETE_VERSION:
        raise ET1Error("unsupported ET1 complete archive")
    if len(member) != COMPLETE_HEADER.size + semantic_n + carrier_n + residual_n + topology_n:
        raise ET1Error("ET1 complete archive section lengths do not close")
    cursor = COMPLETE_HEADER.size
    sections = {}
    for name, length in (
        ("semantic_renderer", semantic_n),
        ("pose_carrier", carrier_n),
        ("compact_residual", residual_n),
    ):
        sections[name] = member[cursor : cursor + length]
        cursor += length
    return sections, member[cursor:]


def retain_complete_archive_framing(archive_path: Path, output_path: Path) -> dict[str, object]:
    """Retain the exact ZIP plus ET1C framing bytes as one accounting payload."""

    archive_bytes = archive_path.read_bytes()
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo("p")
        member = archive.read("p")
    local_offset = int(info.header_offset)
    if archive_bytes[local_offset : local_offset + 4] != b"PK\x03\x04":
        raise ET1Error("ET1 archive local ZIP header is malformed")
    filename_bytes = int.from_bytes(archive_bytes[local_offset + 26 : local_offset + 28], "little")
    extra_bytes = int.from_bytes(archive_bytes[local_offset + 28 : local_offset + 30], "little")
    member_start = local_offset + 30 + filename_bytes + extra_bytes
    member_end = member_start + int(info.compress_size)
    framing = (
        archive_bytes[:member_start]
        + member[: COMPLETE_HEADER.size]
        + archive_bytes[member_end:]
    )
    atomic_bytes(output_path, framing)
    return file_fact(output_path)


def run(args: argparse.Namespace) -> dict[str, object]:
    source_archive = args.source_archive.resolve()
    source_tokens = args.source_tokens.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    validate_sources(source_archive, source_tokens)
    section_paths = extract_dx2_sections(source_archive, output_root / "retained" / "source_sections")

    if args.resume_from is not None:
        resume_path = args.resume_from.resolve()
        if resume_path != manifest_path:
            raise ET1Error("--resume-from must name this run's manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "schema": "ddm_et1_edge_topology_container_gate_v1",
            "source_archive": file_fact(source_archive),
            "source_tokens": file_fact(source_tokens),
            "shape": list(TOKEN_SHAPE),
            "classes": NUM_CLASSES,
            "mechanism": {
                "family": "shared-topology per-class implicit BSP generators",
                "boundaries_stored": False,
                "dense_cells_stored": False,
                "evaluator_cells": "implicit from deterministic split policy",
                "verdict_scope": "current-object exact implicit BSP generator-tree formulation",
            },
            "policies": {},
        }
        atomic_json(manifest_path, manifest)

    tokens = np.memmap(source_tokens, mode="r", dtype=np.uint8, shape=TOKEN_SHAPE)
    policies = manifest["policies"]
    if not isinstance(policies, dict):
        raise ET1Error("manifest policies field is malformed")
    for policy in args.policies:
        policy_dir = output_root / "retained" / policy
        raw_path = policy_dir / "tree_events.3bit"
        policy_result = policies.get(policy)
        if not isinstance(policy_result, dict) or not raw_path.is_file():
            policy_result = encode_tree(tokens, policy, raw_path)
            policy_result["coders"] = {}
            policies[policy] = policy_result
            atomic_json(manifest_path, manifest)
        coders = policy_result.setdefault("coders", {})
        if not isinstance(coders, dict):
            raise ET1Error(f"manifest coder field is malformed for {policy}")
        raw = raw_path.read_bytes()
        event_count = int(policy_result["event_count"])
        for coder in args.coders:
            coder_dir = policy_dir / coder
            coded_path = coder_dir / "topology.coded"
            repeat_path = coder_dir / "topology.repeat.coded"
            packet_path = coder_dir / "topology.et1g"
            if coder not in coders or not all(path.is_file() for path in (coded_path, repeat_path, packet_path)):
                started = time.monotonic()
                coded = compress_payload(raw, coder)
                repeated = compress_payload(raw, coder)
                atomic_bytes(coded_path, coded)
                atomic_bytes(repeat_path, repeated)
                if coded != repeated:
                    raise ET1Error(f"{coder} is not deterministic on {policy}")
                restored = decompress_payload(coded, coder)
                if restored != raw:
                    raise ET1Error(f"{coder} failed exact coder parse-back on {policy}")
                packet = tree_packet(policy, coder, event_count, len(raw), coded)
                atomic_bytes(packet_path, packet)
                coders[coder] = {
                    "seconds": time.monotonic() - started,
                    "coded_payload": file_fact(coded_path),
                    "repeat_payload": file_fact(repeat_path),
                    "deterministic_repeat_equal": True,
                    "raw_coder_parseback_equal": True,
                    "receiver_packet": file_fact(packet_path),
                }
                atomic_json(manifest_path, manifest)
        del raw
    del tokens

    candidates: list[tuple[int, str, str, Path]] = []
    for policy, policy_result in policies.items():
        if not isinstance(policy_result, dict):
            continue
        coders = policy_result.get("coders", {})
        if not isinstance(coders, dict):
            continue
        for coder, coder_result in coders.items():
            if not isinstance(coder_result, dict):
                continue
            packet_fact = coder_result.get("receiver_packet", {})
            if isinstance(packet_fact, dict) and isinstance(packet_fact.get("path"), str):
                packet_path = Path(packet_fact["path"])
                candidates.append((packet_path.stat().st_size, policy, coder, packet_path))
    if not candidates:
        raise ET1Error("no complete coder candidate was produced")
    _, best_policy, best_coder, best_packet_path = min(candidates)
    best_dir = output_root / "retained" / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_packet = best_packet_path.read_bytes()
    decoded_path = best_dir / "decoded_tokens.u8"
    decoded_fact = decode_tree_packet_to_file(best_packet, decoded_path)
    if decoded_fact["sha256"] != DX2_TOKEN_FIELD_SHA256:
        raise ET1Error(f"best ET1 receiver did not reproduce the exact DX2 field: {decoded_fact}")

    semantic = section_paths["semantic_renderer"].read_bytes()
    carrier = section_paths["pose_carrier"].read_bytes()
    residual = section_paths["compact_residual"].read_bytes()
    archive_path = best_dir / "candidate_et1_generator_tree.zip"
    repeat_archive_path = best_dir / "candidate_et1_generator_tree.repeat.zip"
    build_complete_archive(archive_path, best_packet, semantic, carrier, residual)
    build_complete_archive(repeat_archive_path, best_packet, semantic, carrier, residual)
    if archive_path.read_bytes() != repeat_archive_path.read_bytes():
        raise ET1Error("complete ET1 archive build is not deterministic")
    parsed_sections, parsed_packet = parse_complete_archive(archive_path)
    for name, payload in parsed_sections.items():
        if hashlib.sha256(payload).hexdigest() != EXPECTED_SECTION_SHA256[name]:
            raise ET1Error(f"complete ET1 archive changed inherited {name}")
    if parsed_packet != best_packet:
        raise ET1Error("complete ET1 archive changed the selected topology packet")
    archive_decoded_path = best_dir / "archive_parseback_tokens.u8"
    archive_decoded_fact = decode_tree_packet_to_file(parsed_packet, archive_decoded_path)
    if archive_decoded_fact["sha256"] != DX2_TOKEN_FIELD_SHA256:
        raise ET1Error("complete ET1 archive parse-back did not reproduce the exact DX2 field")

    topology_packet_bytes = len(best_packet)
    total_bytes = archive_path.stat().st_size
    zip_and_et1_framing = total_bytes - (
        len(semantic) + len(carrier) + len(residual) + topology_packet_bytes
    )
    framing_fact = retain_complete_archive_framing(
        archive_path,
        best_dir / "container_framing.bin",
    )
    if framing_fact["bytes"] != zip_and_et1_framing:
        raise ET1Error("retained ET1 framing does not close the archive accounting")
    final = {
        "best_policy": best_policy,
        "best_coder": best_coder,
        "best_topology_packet": file_fact(best_packet_path),
        "decoded_tokens": decoded_fact,
        "complete_archive": file_fact(archive_path),
        "complete_archive_repeat": file_fact(repeat_archive_path),
        "complete_archive_repeat_equal": True,
        "complete_archive_parseback_tokens": archive_decoded_fact,
        "container_framing": framing_fact,
        "exact_token_field_equal": True,
        "typed_residue_rows": [
            {
                "residue": "semantic_renderer",
                "disposition": "INHERITED",
                "source_bytes": len(semantic),
                "container_bytes": len(semantic),
                "coder": "RX1 inherited byte-identically",
                "payload_sha256": EXPECTED_SECTION_SHA256["semantic_renderer"],
            },
            {
                "residue": "pose_carrier",
                "disposition": "INHERITED",
                "source_bytes": len(carrier),
                "container_bytes": len(carrier),
                "coder": "RX1 inherited byte-identically",
                "payload_sha256": EXPECTED_SECTION_SHA256["pose_carrier"],
            },
            {
                "residue": "compact_residual",
                "disposition": "INHERITED",
                "source_bytes": len(residual),
                "container_bytes": len(residual),
                "coder": "RX1 inherited byte-identically",
                "payload_sha256": EXPECTED_SECTION_SHA256["compact_residual"],
            },
            {
                "residue": "hpac_model",
                "disposition": "REPLACED",
                "source_bytes": EXPECTED_HPAC_BYTES,
                "container_bytes": 0,
                "coder": "RX1 model stream removed with its coupled token representation",
                "payload_sha256": EXPECTED_SECTION_SHA256["hpac_model"],
            },
            {
                "residue": "token_stream",
                "disposition": "REPLACED",
                "source_bytes": EXPECTED_TOKEN_STREAM_BYTES,
                "container_bytes": 0,
                "coder": "RX1 RC64 stream replaced by the generator topology",
                "payload_sha256": EXPECTED_SECTION_SHA256["token_stream"],
            },
            {
                "residue": "implicit_generator_topology",
                "disposition": "NEW",
                "source_bytes": 0,
                "container_bytes": topology_packet_bytes,
                "coder": best_coder,
                "payload_sha256": hashlib.sha256(best_packet).hexdigest(),
            },
            {
                "residue": "container_framing",
                "disposition": "REPLACED",
                "source_bytes": 114,
                "container_bytes": zip_and_et1_framing,
                "coder": "deterministic ZIP_STORED plus ET1C header",
                "payload_sha256": framing_fact["sha256"],
            },
        ],
        "archive_bytes": total_bytes,
        "fixed_distortion_cap_bytes": 137_986,
        "bytes_over_fixed_distortion_cap": total_bytes - 137_986,
        "zero_distortion_cap_bytes": DX2_ARCHIVE_BYTES - 150,
        "zero_distortion_required_shed_bytes": 150,
        "bytes_over_zero_distortion_cap": total_bytes - (DX2_ARCHIVE_BYTES - 150),
        "bytes_shed_vs_dx2": DX2_ARCHIVE_BYTES - total_bytes,
        "prediction": "CONFIRMED" if total_bytes < 137_986 else "REFUTED",
        "prediction_scope": "current-object exact implicit BSP generator-tree formulation",
    }
    manifest["final"] = final
    atomic_json(manifest_path, manifest)
    return manifest


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-tokens", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=tuple(POLICY_IDS),
        default=list(POLICY_IDS),
    )
    parser.add_argument(
        "--coders",
        nargs="+",
        choices=tuple(CODER_IDS),
        default=list(CODER_IDS),
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest = run(args)
    print(json.dumps(manifest["final"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
