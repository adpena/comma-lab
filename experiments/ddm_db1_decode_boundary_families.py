#!/usr/bin/env python3
"""Price dc1s metadata-boundary variants on its retained full-n600 object.

This experiment is scorer-free.  It does not implement or rename DC1 Family B
(constraint-shipped task-cell solve) or Family C (REC on the same quotient).
Instead, it tests the two concrete metadata hypotheses inherited from dc1s:

* dense-width addressing removes the explicit sparse-position field by making
  every fixed-grid slot receiver-addressable; zero width means the MAP answer;
* group-uniform questions remove every per-block width by deriving one width
  from the counted group header.

The variants consume real question bytes and decode every retained non-MAP
block without receiving its target.  Every raw packet, coder output, repeat,
and decoded answer transcript is retained under the caller-owned APDataStore
directory.  No payload is measured and discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import lzma
import math
import os
import struct
import sys
import time
import traceback
import zlib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import brotli
import ddm_dc1s_sparse_grid_sweep as dc1s
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_STORE = Path(
    "/Volumes/APDataStore/pact/ddm_dc1_decode_time_compute/full_frame_sparse_sweep"
)
DEFAULT_STORE = Path("/Volumes/APDataStore/pact/ddm_db1_decode_boundary_families/retained")
SOURCE_SCRIPT_SHA256 = "2e4ca2576abac8259d11b6bfaf3deb92940ee5434feff3a2a82e23b7ba94813c"
SOURCE_PACKET_SHA256 = "9ca6e59e789abdd0c02c70c3d5d52d2b0da917518f03f792b3bbcc31c30fa839"
SOURCE_RESULT_SHA256 = "4dfff0bbf699187c810b42f65facd60715081d5b4d84e4c1fdd34cf4bf22f6ff"
RB1_SHA256 = "fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09"
TL1_SHA256 = "d307c971f7cdb41806f39135acbc5ff68549283700699ae7a8b1bd77d60ecf15"
FRAME_COUNT = 600
GROUP_COUNT = 190
TOKEN_MEMBER_BYTES = 113_777
RATE_DEMAND_BYTES = 42_382
TOKEN_REPLACEMENT_CEILING_BYTES = TOKEN_MEMBER_BYTES - RATE_DEMAND_BYTES
STORE_ESTIMATE_BYTES = 1 << 30
STORE_RESERVE_BYTES = 8 << 30
AXIS = "[macOS-CPU advisory / scorer-free retained-fx5 n600 rate measurement]"

CODED_MAGIC = b"DB1C"
CODED_VERSION = 1
CODED_HEADER = struct.Struct("<4sBBII32s")
DENSE_MAGIC = b"DB1P"
DENSE_VERSION = 1
DENSE_HEADER = struct.Struct("<4sBBHHIIQ32s32s")


class Db1Error(RuntimeError):
    """Fail-closed DB1 source, packet, coder, or decode error."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        partial.write_bytes(payload)
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def atomic_json(path: Path, payload: object) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode())


def pack_fixed_width(values: Iterable[int], width: int) -> bytes:
    bits: list[int] = []
    for value in values:
        bits.extend(dc1s.integer_bits(int(value), width))
    return dc1s.pack_bits(bits)


def unpack_fixed_width(payload: bytes, count: int, width: int) -> list[int]:
    bits = dc1s.unpack_bits(payload, count * width)
    return [
        dc1s.bits_integer(bits[index * width : (index + 1) * width])
        for index in range(count)
    ]


def chosen_group_records(bundle: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(bundle, allow_pickle=False) as payload:
        metadata = json.loads(bytes(payload["chosen_metadata_json"]).decode())
        block_size = int(metadata["block_size"])
        record_keys = (
            "block_size",
            "block_index",
            "symbols",
            "hash_bits",
            "target_digest",
            "coding_rows",
            "target",
            "target_rank",
        )
        mask = payload["block_size"] == block_size
        order = np.argsort(payload["block_index"][mask], kind="stable")
        records = {key: payload[key][mask][order].copy() for key in record_keys}
    if len(records["block_index"]) != int(metadata["non_map_blocks"]):
        raise Db1Error(f"chosen record count disagrees with metadata: {bundle}")
    if len(records["block_index"]) and np.any(
        records["block_index"][1:] <= records["block_index"][:-1]
    ):
        raise Db1Error(f"chosen record indices are not strictly increasing: {bundle}")
    return records, metadata


def source_pins(source_store: Path) -> dict[str, object]:
    pins = {
        REPO / "experiments/ddm_dc1s_sparse_grid_sweep.py": SOURCE_SCRIPT_SHA256,
        source_store / "retained/sparse_grid_packet.bin": SOURCE_PACKET_SHA256,
        source_store / "retained/result.json": SOURCE_RESULT_SHA256,
        REPO / ".omx/research/ddm_rb1_rate_bound_decomposition_20260822.md": RB1_SHA256,
        REPO / ".omx/research/ddm_tl1_teacher_ledger_20260822.md": TL1_SHA256,
    }
    facts: dict[str, object] = {}
    for path, expected in pins.items():
        if not path.is_file():
            raise Db1Error(f"pinned source is absent: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise Db1Error(f"pinned source changed: {path}: {actual} != {expected}")
        facts[str(path)] = file_fact(path)
    source_result = json.loads((source_store / "retained/result.json").read_text())
    groups = source_result.get("groups", [])
    if source_result.get("frames") != FRAME_COUNT or len(groups) != GROUP_COUNT:
        raise Db1Error("pinned dc1s result has the wrong full-population dimensions")
    for group, row in enumerate(groups):
        path = source_store / f"retained/group_{group:03d}.npz"
        expected_fact = row.get("bundle", {})
        if Path(str(expected_fact.get("path", ""))).resolve() != path.resolve():
            raise Db1Error(f"dc1s group {group} receipt names a different bundle")
        actual_fact = file_fact(path)
        if actual_fact["bytes"] != expected_fact.get("bytes") or actual_fact[
            "sha256"
        ] != expected_fact.get("sha256"):
            raise Db1Error(f"dc1s group {group} bundle differs from its pinned result receipt")
        facts[str(path)] = actual_fact
    return facts


def storage_preflight(store: Path, pins: dict[str, object], argv: list[str]) -> dict[str, object]:
    store.mkdir(parents=True, exist_ok=True)
    stats = os.statvfs(store)
    free = stats.f_bavail * stats.f_frsize
    required = STORE_ESTIMATE_BYTES + STORE_RESERVE_BYTES
    if free < required:
        raise Db1Error(f"APDataStore preflight failed: {free} free bytes < {required} required")
    receipt = {
        "schema": "ddm_db1_storage_preflight.v1",
        "axis": AXIS,
        "store": str(store),
        "free_bytes_before": free,
        "estimated_output_bytes": STORE_ESTIMATE_BYTES,
        "reserve_bytes": STORE_RESERVE_BYTES,
        "passed": True,
        "implementation_sha256": sha256_file(Path(__file__)),
        "argv": argv,
        "pins": pins,
    }
    atomic_json(store / "preflight.json", receipt)
    return receipt


def load_manifest(path: Path, store: Path, source_store: Path) -> dict[str, Any]:
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != "ddm_db1_decode_boundary_families_manifest.v1":
            raise Db1Error("resume manifest has the wrong schema")
        if manifest.get("store") != str(store) or manifest.get("source_store") != str(source_store):
            raise Db1Error("resume manifest belongs to different stores")
        current_sha = sha256_file(Path(__file__))
        previous_sha = str(manifest.get("implementation_sha256", ""))
        if previous_sha and previous_sha != current_sha:
            for stage in manifest.get("stages", {}).values():
                if isinstance(stage, dict):
                    stage.setdefault("implementation_sha256", previous_sha)
            manifest.setdefault("implementation_history", []).append(previous_sha)
            manifest["implementation_sha256"] = current_sha
            atomic_json(path, manifest)
        return manifest
    manifest = {
        "schema": "ddm_db1_decode_boundary_families_manifest.v1",
        "axis": AXIS,
        "store": str(store),
        "source_store": str(source_store),
        "implementation_sha256": sha256_file(Path(__file__)),
        "stages": {},
    }
    atomic_json(path, manifest)
    return manifest


def stage_valid(stage: object, *, require_current_implementation: bool = False) -> bool:
    if not isinstance(stage, dict) or not stage.get("complete"):
        return False
    if require_current_implementation and stage.get("implementation_sha256") != sha256_file(
        Path(__file__)
    ):
        return False
    for fact in stage.get("artifacts", []):
        path = Path(str(fact.get("path", "")))
        if not path.is_file() or path.stat().st_size != int(fact.get("bytes", -1)):
            return False
        if sha256_file(path) != fact.get("sha256"):
            return False
    return True


def save_stage(manifest_path: Path, manifest: dict[str, Any], name: str, artifacts: list[Path]) -> None:
    manifest["stages"][name] = {
        "complete": True,
        "implementation_sha256": sha256_file(Path(__file__)),
        "artifacts": [file_fact(path) for path in artifacts],
    }
    atomic_json(manifest_path, manifest)


def build_dense_width_packet(
    source_store: Path,
    output: Path,
) -> tuple[dict[str, object], list[dict[str, Any]]]:
    group_codes: list[int] = []
    dense_parts: list[bytes] = []
    question_bits: list[int] = []
    specifications: list[dict[str, Any]] = []
    total_non_map = 0
    for group in range(GROUP_COUNT):
        records, metadata = chosen_group_records(source_store / f"retained/group_{group:03d}.npz")
        block_size = int(metadata["block_size"])
        total_blocks = int(metadata["total_blocks"])
        group_codes.append(dc1s.BLOCK_SIZE_CODES[block_size])
        widths = np.zeros(total_blocks, dtype=np.uint8)
        indices = records["block_index"].astype(np.int64)
        widths[indices] = records["hash_bits"]
        if np.count_nonzero(widths) != len(indices) or np.any(widths[indices] == 0):
            raise Db1Error(f"dense width support is not bijective for group {group}")
        dense_parts.append(widths.tobytes())
        for digest, width in zip(records["target_digest"], records["hash_bits"], strict=True):
            question_bits.extend(dc1s.digest_prefix_bits(digest, int(width)))
        specifications.append(
            {
                "group": group,
                "block_size": block_size,
                "total_blocks": total_blocks,
                "non_map_blocks": len(indices),
                "width_bytes_offset": sum(len(part) for part in dense_parts[:-1]),
                "width_bytes": len(widths),
                "question_bits": int(records["hash_bits"].astype(np.uint64).sum()),
            }
        )
        total_non_map += len(indices)
    code_bytes = pack_fixed_width(group_codes, 2)
    dense_widths = b"".join(dense_parts)
    question_payload = dc1s.pack_bits(question_bits)
    header = DENSE_HEADER.pack(
        DENSE_MAGIC,
        DENSE_VERSION,
        0,
        FRAME_COUNT,
        GROUP_COUNT,
        len(code_bytes),
        len(dense_widths),
        len(question_bits),
        hashlib.sha256(dense_widths).digest(),
        hashlib.sha256(question_payload).digest(),
    )
    raw = header + code_bytes + dense_widths + question_payload
    atomic_bytes(output, raw)
    return (
        {
            "schema": "ddm_db1_dense_width_packet.v1",
            "variant": "dense_width_addressing",
            "position_derivation": (
                "fixed-grid slot order; nonzero decoded width selects a question, zero selects MAP"
            ),
            "position_payload_semantics": (
                "no separate position list; support information remains counted inside the dense width field"
            ),
            "frames": FRAME_COUNT,
            "groups": GROUP_COUNT,
            "non_map_blocks": total_non_map,
            "group_code_bytes": len(code_bytes),
            "dense_width_bytes": len(dense_widths),
            "question_bits": len(question_bits),
            "question_bytes": len(question_payload),
            "raw_packet": file_fact(output),
        },
        specifications,
    )


def parse_dense_width_packet(
    payload: bytes,
    specifications: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[list[list[int]]]]:
    if len(payload) < DENSE_HEADER.size:
        raise Db1Error("dense-width packet is truncated")
    (
        magic,
        version,
        reserved,
        frames,
        groups,
        code_bytes_count,
        dense_width_count,
        question_bit_count,
        widths_sha,
        questions_sha,
    ) = DENSE_HEADER.unpack_from(payload)
    if (magic, version, reserved, frames, groups) != (
        DENSE_MAGIC,
        DENSE_VERSION,
        0,
        FRAME_COUNT,
        GROUP_COUNT,
    ):
        raise Db1Error("dense-width packet header differs from the sealed schema")
    cursor = DENSE_HEADER.size
    code_bytes = payload[cursor : cursor + code_bytes_count]
    cursor += code_bytes_count
    dense_widths = payload[cursor : cursor + dense_width_count]
    cursor += dense_width_count
    question_payload = payload[cursor:]
    if len(code_bytes) != code_bytes_count or len(dense_widths) != dense_width_count:
        raise Db1Error("dense-width packet field is truncated")
    if len(question_payload) != math.ceil(question_bit_count / 8):
        raise Db1Error("dense-width packet question length or trailing bytes differ")
    if hashlib.sha256(dense_widths).digest() != widths_sha:
        raise Db1Error("dense-width field SHA-256 differs")
    if hashlib.sha256(question_payload).digest() != questions_sha:
        raise Db1Error("dense-width question SHA-256 differs")
    group_codes = unpack_fixed_width(code_bytes, GROUP_COUNT, 2)
    all_question_bits = dc1s.unpack_bits(question_payload, question_bit_count)
    width_arrays: list[np.ndarray] = []
    prefix_groups: list[list[list[int]]] = []
    width_cursor = 0
    question_cursor = 0
    for specification in specifications:
        group = int(specification["group"])
        expected_block_size = int(specification["block_size"])
        if dc1s.BLOCK_CODE_SIZES[group_codes[group]] != expected_block_size:
            raise Db1Error(f"dense-width packet block size changed for group {group}")
        count = int(specification["width_bytes"])
        widths = np.frombuffer(dense_widths[width_cursor : width_cursor + count], dtype=np.uint8).copy()
        width_cursor += count
        prefixes: list[list[int]] = []
        for width in widths[widths > 0]:
            value = int(width)
            prefixes.append(all_question_bits[question_cursor : question_cursor + value])
            question_cursor += value
        width_arrays.append(widths)
        prefix_groups.append(prefixes)
    if width_cursor != len(dense_widths) or question_cursor != question_bit_count:
        raise Db1Error("dense-width packet did not consume every meaningful byte and bit")
    return width_arrays, prefix_groups


def build_group_uniform_packet(source_store: Path, output: Path) -> dict[str, object]:
    headers = bytearray()
    body_bits: list[int] = []
    position_bits = 0
    question_bits = 0
    total_non_map = 0
    for group in range(GROUP_COUNT):
        records, metadata = chosen_group_records(source_store / f"retained/group_{group:03d}.npz")
        block_size = int(metadata["block_size"])
        total_blocks = int(metadata["total_blocks"])
        indices = records["block_index"].astype(np.int64)
        elias_l, positions = dc1s.elias_fano_encode(indices, total_blocks)
        uniform_width = int(records["hash_bits"].max()) if len(indices) else 0
        if uniform_width > 31:
            raise Db1Error(f"uniform width exceeds source header for group {group}")
        flags = dc1s.BLOCK_SIZE_CODES[block_size] | (uniform_width << 3)
        headers.extend(struct.pack("<IBB", len(indices), flags, elias_l))
        body_bits.extend(positions)
        for digest in records["target_digest"]:
            body_bits.extend(dc1s.digest_prefix_bits(digest, uniform_width))
        position_bits += len(positions)
        question_bits += len(indices) * uniform_width
        total_non_map += len(indices)
    top = struct.pack(
        "<4sBBHH",
        dc1s.PACKET_MAGIC,
        dc1s.PACKET_VERSION,
        0,
        FRAME_COUNT,
        GROUP_COUNT,
    )
    raw = top + bytes(headers) + dc1s.pack_bits(body_bits)
    atomic_bytes(output, raw)
    return {
        "schema": "ddm_db1_group_uniform_packet.v1",
        "variant": "group_uniform_width",
        "width_derivation": "one counted five-bit group-header width applies to every sparse question",
        "width_payload_semantics": "no per-block width table; sparse positions remain counted",
        "frames": FRAME_COUNT,
        "groups": GROUP_COUNT,
        "non_map_blocks": total_non_map,
        "top_header_bits": len(top) * 8,
        "group_header_bits": len(headers) * 8,
        "position_bits": position_bits,
        "per_block_width_bits": 0,
        "question_bits": question_bits,
        "raw_packet": file_fact(output),
    }


def parse_sparse_packet(
    payload: bytes,
    source_store: Path,
) -> list[tuple[np.ndarray, np.ndarray, list[list[int]]]]:
    top_size = struct.calcsize("<4sBBHH")
    if len(payload) < top_size:
        raise Db1Error("sparse packet is truncated")
    magic, version, reserved, frames, groups = struct.unpack_from("<4sBBHH", payload)
    if (magic, version, reserved, frames, groups) != (
        dc1s.PACKET_MAGIC,
        dc1s.PACKET_VERSION,
        0,
        FRAME_COUNT,
        GROUP_COUNT,
    ):
        raise Db1Error("sparse packet header differs from the sealed source schema")
    group_header_size = struct.calcsize("<IBB")
    headers_end = top_size + GROUP_COUNT * group_header_size
    body_bits = dc1s.unpack_bits(payload[headers_end:], 8 * len(payload[headers_end:]))
    cursor = 0
    decoded: list[tuple[np.ndarray, np.ndarray, list[list[int]]]] = []
    for group in range(GROUP_COUNT):
        _, metadata = chosen_group_records(source_store / f"retained/group_{group:03d}.npz")
        header = payload[
            top_size + group * group_header_size : top_size + (group + 1) * group_header_size
        ]
        before = cursor
        cursor, block_size, indices, widths, prefixes = dc1s.parse_group_body(
            body_bits,
            cursor,
            int(metadata["total_blocks"]),
            header,
        )
        if block_size != int(metadata["block_size"]):
            raise Db1Error(f"sparse packet block size changed for group {group}")
        if cursor <= before and len(indices):
            raise Db1Error(f"sparse packet did not consume group {group}")
        decoded.append((indices, widths, prefixes))
    meaningful_padding = body_bits[cursor:]
    if len(meaningful_padding) > 7 or any(meaningful_padding):
        raise Db1Error("sparse packet has nonzero or excessive trailing bits")
    return decoded


def candidate_from_state(rows: np.ndarray, choices: np.ndarray, state: tuple[int, ...]) -> np.ndarray:
    return np.fromiter(
        (choices[column, state[column]] for column in range(len(state))),
        dtype=np.uint8,
        count=len(state),
    )


def digest_prefix_from_bytes(digest: bytes, count: int) -> list[int]:
    """Read a SHA prefix without relying on dc1s's unexercised bytes annotation."""
    raw = np.frombuffer(digest, dtype=np.uint8)
    return dc1s.digest_prefix_bits(raw, count)


def decode_question(rows: np.ndarray, prefix: list[int], group: int) -> np.ndarray:
    """Return the first real-law candidate matching a transmitted SHA prefix."""
    rows = np.asarray(rows, dtype=np.float64)
    symbols = len(rows)
    if rows.shape != (symbols, dc1s.NUM_CLASSES) or not prefix:
        raise Db1Error("question decoder requires real rows and a nonempty prefix")
    lexical = np.arange(dc1s.NUM_CLASSES, dtype=np.int64)
    choices = np.empty((symbols, dc1s.NUM_CLASSES), dtype=np.uint8)
    for column in range(symbols):
        choices[column] = np.lexsort((lexical, -rows[column])).astype(np.uint8)
    logs = np.log2(np.maximum(rows, 1e-300))

    def log_probability(state: tuple[int, ...]) -> float:
        # Match dc1s's deliberately sequential float64 accumulation exactly.
        # Python 3.12+ built-in sum() uses compensated summation and can change
        # the heap order at 1e-15-class product-law ties.
        total = 0.0
        for column in range(symbols):
            total += float(logs[column, int(choices[column, state[column]])])
        return total

    initial = (0,) * symbols
    first = candidate_from_state(rows, choices, initial)
    heap: list[tuple[float, int, tuple[int, ...]]] = [
        (-log_probability(initial), dc1s.candidate_code(first), initial)
    ]
    seen = {initial}
    while heap:
        _, _, state = heapq.heappop(heap)
        candidate = candidate_from_state(rows, choices, state)
        digest = dc1s.candidate_digest(group, symbols, candidate)
        if digest_prefix_from_bytes(digest, len(prefix)) == prefix:
            return candidate
        for column in range(symbols):
            if state[column] + 1 >= dc1s.NUM_CLASSES:
                continue
            child = list(state)
            child[column] += 1
            child_state = tuple(child)
            if child_state in seen:
                continue
            seen.add(child_state)
            child_candidate = candidate_from_state(rows, choices, child_state)
            heapq.heappush(
                heap,
                (-log_probability(child_state), dc1s.candidate_code(child_candidate), child_state),
            )
    raise Db1Error("question decoder exhausted the real product law")


def answer_record(group: int, block_index: int, answer: np.ndarray) -> bytes:
    raw = np.asarray(answer, dtype=np.uint8).tobytes()
    return struct.pack("<HIB", group, block_index, len(raw)) + raw


def verify_sparse_questions(
    name: str,
    parsed: list[tuple[np.ndarray, np.ndarray, list[list[int]]]],
    source_store: Path,
    output: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    transcript = bytearray()
    attempt_dir = output.parent / f"{output.stem}.groups.attempt_{time.time_ns()}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    group_files: list[dict[str, object]] = []
    decoded_blocks = 0
    candidates_expected = 0
    attempt_error: dict[str, str] | None = None
    attempt_receipt_path = attempt_dir / "attempt_receipt.json"
    try:
        for group, (indices, widths, prefixes) in enumerate(parsed):
            records, _ = chosen_group_records(source_store / f"retained/group_{group:03d}.npz")
            expected_indices = records["block_index"].astype(np.int64)
            if not np.array_equal(indices, expected_indices):
                raise Db1Error(f"{name} sparse support changed for group {group}")
            if len(widths) != len(indices) or len(prefixes) != len(indices):
                raise Db1Error(f"{name} question arrays disagree for group {group}")
            group_path = attempt_dir / f"group_{group:03d}.answers.bin"
            with group_path.open("xb") as group_handle:
                for row_index, (block_index, width, prefix) in enumerate(
                    zip(indices, widths, prefixes, strict=True)
                ):
                    if len(prefix) != int(width):
                        raise Db1Error(f"{name} prefix width changed for group {group}")
                    symbols = int(records["symbols"][row_index])
                    decoded = decode_question(
                        records["coding_rows"][row_index, :symbols], prefix, group
                    )
                    expected = records["target"][row_index, :symbols]
                    if not np.array_equal(decoded, expected):
                        raise Db1Error(
                            f"{name} decoded the wrong answer at group {group}, block {block_index}"
                        )
                    record = answer_record(group, int(block_index), decoded)
                    group_handle.write(record)
                    transcript.extend(record)
                    decoded_blocks += 1
                    candidates_expected += int(records["target_rank"][row_index]) + 1
                    if row_index % 256 == 255:
                        group_handle.flush()
                        os.fsync(group_handle.fileno())
                group_handle.flush()
                os.fsync(group_handle.fileno())
            group_files.append(file_fact(group_path))
        atomic_bytes(output, bytes(transcript))
    except BaseException as exc:
        attempt_error = {"error_type": type(exc).__name__, "error": str(exc)}
        raise
    finally:
        retained_group_files = [
            file_fact(path) for path in sorted(attempt_dir.glob("group_*.answers.bin"))
        ]
        atomic_json(
            attempt_receipt_path,
            {
                "schema": "ddm_db1_decode_attempt.v1",
                "variant": name,
                "complete": attempt_error is None,
                "error": attempt_error,
                "decoded_non_map_blocks": decoded_blocks,
                "expected_candidates_examined": candidates_expected,
                "group_answer_files": retained_group_files,
                "payload_policy": (
                    "every completed or partial group transcript is retained and content-addressed"
                ),
            },
        )
    return {
        "variant": name,
        "decoded_non_map_blocks": decoded_blocks,
        "implicit_map_blocks": "all fixed-grid slots omitted from the retained non-MAP census",
        "expected_candidates_examined": candidates_expected,
        "decode_seconds_local": time.perf_counter() - started,
        "answer_transcript": file_fact(output),
        "group_answer_files": group_files,
        "attempt_receipt": file_fact(attempt_receipt_path),
        "target_not_passed_to_question_decoder": True,
    }


def verify_dense_questions(
    payload: bytes,
    specifications: list[dict[str, Any]],
    source_store: Path,
    output: Path,
) -> dict[str, object]:
    widths_by_group, prefixes_by_group = parse_dense_width_packet(payload, specifications)
    parsed: list[tuple[np.ndarray, np.ndarray, list[list[int]]]] = []
    for widths, prefixes in zip(widths_by_group, prefixes_by_group, strict=True):
        indices = np.flatnonzero(widths > 0).astype(np.int64)
        parsed.append((indices, widths[indices], prefixes))
    receipt = verify_sparse_questions("dense_width_addressing", parsed, source_store, output)
    receipt["dense_slots"] = sum(len(widths) for widths in widths_by_group)
    receipt["position_field_consumed"] = False
    receipt["support_recovered_from_nonzero_width_slots"] = True
    return receipt


def first_answer_records(payload: bytes, count: int) -> bytes:
    cursor = 0
    for _ in range(count):
        if cursor + 7 > len(payload):
            raise Db1Error("answer transcript is truncated before the requested recovery prefix")
        _, _, symbols = struct.unpack_from("<HIB", payload, cursor)
        cursor += 7 + symbols
        if cursor > len(payload):
            raise Db1Error("answer transcript record is truncated")
    return payload[:cursor]


def recover_failed_decode_prefix(family_a: dict[str, object], store: Path) -> dict[str, object]:
    """Recover the exact transcript prefix discarded by the pre-checkpointing attempt."""
    group_files = list(family_a["group_answer_files"])
    if len(group_files) != GROUP_COUNT:
        raise Db1Error("successful Family-A decode lacks the complete group checkpoint set")
    pieces = [Path(str(fact["path"])).read_bytes() for fact in group_files[:129]]
    group_129 = Path(str(group_files[129]["path"])).read_bytes()
    pieces.append(first_answer_records(group_129, 5))
    recovered = b"".join(pieces)
    path = store / "recovered_incidents/family_a_pre_group129_block94_failure_prefix.answers.bin"
    atomic_bytes(path, recovered)
    return {
        "classification": "P0_PAYLOAD_RETENTION_INCIDENT_RECOVERED",
        "original_failure": "family_a_source decoded the wrong answer at group 129, block 94",
        "root_cause": (
            "the first decoder implementation used Python 3.12+ compensated sum(), changing "
            "1e-15-class heap ties from dc1s's sequential float64 accumulation"
        ),
        "discarded_at_failure": True,
        "recovery_method": (
            "successful sequential-law decode proved the same answers; concatenated complete groups "
            "0..128 plus the five answers completed before group 129 block 94"
        ),
        "recovered_payload": file_fact(path),
        "per_group_checkpointing_now_active": True,
    }


def codec_table() -> dict[str, tuple[int, Callable[[bytes], bytes], Callable[[bytes], bytes]]]:
    filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "dict_size": 1 << 23}]
    return {
        "raw": (0, lambda value: value, lambda value: value),
        "brotli_q1": (
            1,
            lambda value: brotli.compress(value, quality=1, lgwin=22, mode=brotli.MODE_GENERIC),
            brotli.decompress,
        ),
        "brotli_q6": (
            2,
            lambda value: brotli.compress(value, quality=6, lgwin=22, mode=brotli.MODE_GENERIC),
            brotli.decompress,
        ),
        "brotli_q9": (
            3,
            lambda value: brotli.compress(value, quality=9, lgwin=22, mode=brotli.MODE_GENERIC),
            brotli.decompress,
        ),
        "brotli_q11": (
            4,
            lambda value: brotli.compress(value, quality=11, lgwin=22, mode=brotli.MODE_GENERIC),
            brotli.decompress,
        ),
        "zlib9": (5, lambda value: zlib.compress(value, level=9), zlib.decompress),
        "xz_extreme": (
            6,
            lambda value: lzma.compress(value, format=lzma.FORMAT_RAW, filters=filters),
            lambda value: lzma.decompress(value, format=lzma.FORMAT_RAW, filters=filters),
        ),
    }


def wrap_coded(raw: bytes, codec_id: int, coded: bytes) -> bytes:
    return CODED_HEADER.pack(
        CODED_MAGIC,
        CODED_VERSION,
        codec_id,
        len(raw),
        len(coded),
        hashlib.sha256(raw).digest(),
    ) + coded


def unwrap_coded(payload: bytes) -> tuple[int, bytes]:
    if len(payload) < CODED_HEADER.size:
        raise Db1Error("coded wrapper is truncated")
    magic, version, codec_id, raw_len, coded_len, raw_sha = CODED_HEADER.unpack_from(payload)
    coded = payload[CODED_HEADER.size :]
    if magic != CODED_MAGIC or version != CODED_VERSION or len(coded) != coded_len:
        raise Db1Error("coded wrapper header or exact length differs")
    decoders = {value[0]: value[2] for value in codec_table().values()}
    if codec_id not in decoders:
        raise Db1Error("coded wrapper names an unknown coder")
    raw = decoders[codec_id](coded)
    if len(raw) != raw_len or hashlib.sha256(raw).digest() != raw_sha:
        raise Db1Error("coded wrapper decode length or SHA-256 differs")
    return codec_id, raw


def coder_race(raw_path: Path, output_dir: Path) -> dict[str, object]:
    raw = raw_path.read_bytes()
    rows: list[dict[str, object]] = []
    outputs: list[tuple[int, str, bytes, Path]] = []
    for name, (codec_id, encoder, decoder) in codec_table().items():
        coded = encoder(raw)
        if decoder(coded) != raw:
            raise Db1Error(f"coder {name} did not round-trip {raw_path.name}")
        complete = raw if name == "raw" else wrap_coded(raw, codec_id, coded)
        path = output_dir / f"{raw_path.stem}.{name}.bin"
        atomic_bytes(path, complete)
        if name != "raw":
            observed_id, observed_raw = unwrap_coded(path.read_bytes())
            if observed_id != codec_id or observed_raw != raw:
                raise Db1Error(f"coded wrapper did not parse back for {name}")
        rows.append(
            {
                "coder": name,
                "raw_bytes": len(raw),
                "coded_body_bytes": len(coded),
                "complete_payload": file_fact(path),
            }
        )
        outputs.append((len(complete), name, complete, path))
    _, winner_name, _, winner_path = min(outputs, key=lambda row: (row[0], row[1]))
    winner_codec_id, winner_encoder, winner_decoder = codec_table()[winner_name]
    repeat_coded = winner_encoder(raw)
    if winner_decoder(repeat_coded) != raw:
        raise Db1Error(f"winner repeat did not round-trip {raw_path.name}")
    repeat_payload = (
        raw if winner_name == "raw" else wrap_coded(raw, winner_codec_id, repeat_coded)
    )
    repeat = output_dir / f"{raw_path.stem}.{winner_name}.repeat.bin"
    atomic_bytes(repeat, repeat_payload)
    if sha256_file(winner_path) != sha256_file(repeat):
        raise Db1Error(f"winner repeat changed for {raw_path.name}")
    return {
        "source_raw": file_fact(raw_path),
        "coders": rows,
        "winner": winner_name,
        "winner_payload": file_fact(winner_path),
        "winner_repeat": file_fact(repeat),
        "winner_repeat_identical": True,
    }


def decoded_raw(candidate_fact: dict[str, object]) -> bytes:
    path = Path(str(candidate_fact["path"]))
    payload = path.read_bytes()
    if payload.startswith(CODED_MAGIC):
        return unwrap_coded(payload)[1]
    return payload


def mutate_and_refuse_dense(raw: bytes, store: Path) -> dict[str, object]:
    if len(raw) <= DENSE_HEADER.size + 48:
        raise Db1Error("dense packet is unexpectedly small")
    mutated = bytearray(raw)
    mutated[DENSE_HEADER.size + 48] ^= 1
    path = store / "negative_controls/dense_width_one_bit_mutation.bin"
    atomic_bytes(path, bytes(mutated))
    refused = False
    reason = ""
    try:
        header = DENSE_HEADER.unpack_from(mutated)
        fake_specs = [
            {
                "group": group,
                "block_size": 8,
                "width_bytes": 0,
            }
            for group in range(GROUP_COUNT)
        ]
        parse_dense_width_packet(bytes(mutated), fake_specs)
        reason = f"unexpected parse success: header={header[:5]}"
    except Db1Error as exc:
        refused = True
        reason = str(exc)
    if not refused:
        raise Db1Error("dense-width one-bit mutation was not refused")
    return {"payload": file_fact(path), "refused": True, "reason": reason}


def build_inventory(source_store: Path, store: Path) -> dict[str, object]:
    packets = store / "packets"
    packets.mkdir(parents=True, exist_ok=True)
    source_copy = packets / "family_a_source_packet.bin"
    atomic_bytes(source_copy, (source_store / "retained/sparse_grid_packet.bin").read_bytes())
    dense_path = packets / "dense_width_addressing.raw.bin"
    dense_receipt, specifications = build_dense_width_packet(source_store, dense_path)
    uniform_path = packets / "group_uniform_width.raw.bin"
    uniform_receipt = build_group_uniform_packet(source_store, uniform_path)
    inventory = {
        "schema": "ddm_db1_boundary_inventory.v1",
        "axis": AXIS,
        "selection_mode": "full_population_n600_all_190_groups_dc1s_chosen_block_sizes",
        "source_family_a": {
            "definition": (
                "dc1s sparse-grid packet with explicit Elias-Fano non-MAP positions, "
                "individual-or-uniform widths, and SHA-prefix questions"
            ),
            "packet": file_fact(source_copy),
        },
        "position_boundary_test": dense_receipt,
        "width_boundary_test": uniform_receipt,
        "dense_specifications": specifications,
        "rule_118": (
            "all support, width, block-selection, and hash bytes are counted; only generic parsing, "
            "HPAC evaluation, grid traversal, hashing, and search are receiver code"
        ),
    }
    atomic_json(store / "stage_1_inventory.json", inventory)
    return inventory


def run_coder_stage(store: Path, inventory: dict[str, object]) -> dict[str, object]:
    races_dir = store / "coder_races"
    race_paths = {
        "family_a_source": Path(str(inventory["source_family_a"]["packet"]["path"])),
        "dense_width_addressing": Path(
            str(inventory["position_boundary_test"]["raw_packet"]["path"])
        ),
        "group_uniform_width": Path(str(inventory["width_boundary_test"]["raw_packet"]["path"])),
    }
    races = {name: coder_race(path, races_dir / name) for name, path in race_paths.items()}
    result = {"schema": "ddm_db1_real_coder_races.v1", "axis": AXIS, "races": races}
    atomic_json(store / "stage_2_coder_races.json", result)
    return result


def run_decode_stage(
    source_store: Path,
    store: Path,
    inventory: dict[str, object],
    coder_result: dict[str, object],
) -> dict[str, object]:
    decoded_dir = store / "decoded"
    races = coder_result["races"]
    family_a_raw = decoded_raw(races["family_a_source"]["winner_payload"])
    dense_raw = decoded_raw(races["dense_width_addressing"]["winner_payload"])
    uniform_raw = decoded_raw(races["group_uniform_width"]["winner_payload"])
    family_a = verify_sparse_questions(
        "family_a_source",
        parse_sparse_packet(family_a_raw, source_store),
        source_store,
        decoded_dir / "family_a_source.answers.bin",
    )
    dense = verify_dense_questions(
        dense_raw,
        list(inventory["dense_specifications"]),
        source_store,
        decoded_dir / "dense_width_addressing.answers.bin",
    )
    uniform = verify_sparse_questions(
        "group_uniform_width",
        parse_sparse_packet(uniform_raw, source_store),
        source_store,
        decoded_dir / "group_uniform_width.answers.bin",
    )
    transcript_hashes = {
        row["variant"]: row["answer_transcript"]["sha256"] for row in (family_a, dense, uniform)
    }
    if len(set(transcript_hashes.values())) != 1:
        raise Db1Error(f"boundary variants decoded different non-MAP answers: {transcript_hashes}")
    mutation = mutate_and_refuse_dense(dense_raw, store)
    incident = recover_failed_decode_prefix(family_a, store)
    result = {
        "schema": "ddm_db1_decode_boundary_verification.v1",
        "axis": AXIS,
        "full_population": True,
        "frames": FRAME_COUNT,
        "groups": GROUP_COUNT,
        "variants": [family_a, dense, uniform],
        "answer_transcripts_byte_identical": True,
        "answer_transcript_sha256": next(iter(transcript_hashes.values())),
        "negative_control": mutation,
        "retention_incident": incident,
    }
    atomic_json(store / "stage_3_decode_verification.json", result)
    return result


def final_result(
    store: Path,
    preflight: dict[str, object],
    inventory: dict[str, object],
    coder_result: dict[str, object],
    decode_result: dict[str, object],
) -> dict[str, object]:
    rows = []
    for name, race in coder_result["races"].items():
        payload_bytes = int(race["winner_payload"]["bytes"])
        savings = TOKEN_MEMBER_BYTES - payload_bytes
        rows.append(
            {
                "variant": name,
                "winner_coder": race["winner"],
                "complete_payload_bytes": payload_bytes,
                "savings_vs_113777_token_member_bytes": savings,
                "distance_to_42382_byte_demand": savings - RATE_DEMAND_BYTES,
                "replacement_ceiling_bytes": TOKEN_REPLACEMENT_CEILING_BYTES,
                "clears_same_distortion_sub012_replacement_ceiling": (
                    payload_bytes <= TOKEN_REPLACEMENT_CEILING_BYTES
                ),
                "winner_payload": race["winner_payload"],
            }
        )
    boundary_rows = [row for row in rows if row["variant"] != "family_a_source"]
    prior_delta = max(
        abs(int(row["complete_payload_bytes"]) - 388_326) for row in boundary_rows
    )
    result = {
        "schema": "ddm_db1_decode_boundary_families.v1",
        "axis": AXIS,
        "score_claim": False,
        "archive_candidate_built": False,
        "scorer_used": False,
        "modal_used": False,
        "frames": FRAME_COUNT,
        "groups": GROUP_COUNT,
        "selection_mode": "full_population_n600",
        "implementation_sha256": sha256_file(Path(__file__)),
        "preflight": preflight,
        "inventory": file_fact(store / "stage_1_inventory.json"),
        "coder_races": file_fact(store / "stage_2_coder_races.json"),
        "decode_verification": file_fact(store / "stage_3_decode_verification.json"),
        "pricing": rows,
        "prior_law": {
            "prediction": (
                "metadata-boundary variants should change packet arithmetic by more than 1434 bytes"
            ),
            "largest_absolute_change_vs_388326_family_a_bytes": prior_delta,
            "falsifier_every_variant_within_1434_bytes": prior_delta <= 1_434,
        },
        "decode_identity": {
            "non_map_answer_transcripts_byte_identical": decode_result[
                "answer_transcripts_byte_identical"
            ],
            "sha256": decode_result["answer_transcript_sha256"],
            "map_rule": "zero-width or absent sparse slot deterministically emits HPAC MAP",
            "adaptive_state_boundary": (
                "coding rows are the retained receiver-known rows from dc1s's exact n600 production walk; "
                "this arm did not rerun or alter adaptive HPAC state"
            ),
        },
        "family_b_c_boundary": {
            "family_b": (
                "DC1 constraint-shipping: count a scorer-equivalence-cell constraint plus solve seed/index "
                "and deterministically decode any cell member"
            ),
            "family_c": (
                "receiver-runnable REC/bits-back on Q=P(.|C); ideal price KL(Q||P)=-log2 P(C), "
                "therefore folded into Family B until the same quotient exists"
            ),
            "implemented_here": False,
            "reason": (
                "no receiver-checkable task-cell constraint or runnable quotient posterior exists; "
                "the NR1 fixture is specification-only and explicitly forbids use as an executable shortcut"
            ),
        },
    }
    atomic_json(store / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-store", type=Path, default=DEFAULT_SOURCE_STORE)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--resume-from", type=Path, required=True)
    return parser.parse_args()


def execute(args: argparse.Namespace) -> int:
    source_store = args.source_store.resolve()
    store = args.store.resolve()
    resume = args.resume_from.resolve()
    if resume.parent != store:
        raise Db1Error("--resume-from must be the manifest inside --store")
    pins = source_pins(source_store)
    preflight = storage_preflight(store, pins, sys.argv)
    manifest = load_manifest(resume, store, source_store)

    inventory_path = store / "stage_1_inventory.json"
    if stage_valid(
        manifest["stages"].get("inventory"), require_current_implementation=True
    ):
        inventory = json.loads(inventory_path.read_text())
    else:
        inventory = build_inventory(source_store, store)
        inventory_artifacts = [inventory_path]
        inventory_artifacts.extend(
            Path(str(inventory[key]["raw_packet"]["path"]))
            for key in ("position_boundary_test", "width_boundary_test")
        )
        inventory_artifacts.append(Path(str(inventory["source_family_a"]["packet"]["path"])))
        save_stage(resume, manifest, "inventory", inventory_artifacts)

    coder_path = store / "stage_2_coder_races.json"
    if stage_valid(
        manifest["stages"].get("coder_races"), require_current_implementation=True
    ):
        coder_result = json.loads(coder_path.read_text())
    else:
        coder_result = run_coder_stage(store, inventory)
        coder_artifacts = [coder_path]
        for race in coder_result["races"].values():
            coder_artifacts.extend(Path(str(row["complete_payload"]["path"])) for row in race["coders"])
            coder_artifacts.append(Path(str(race["winner_repeat"]["path"])))
        save_stage(resume, manifest, "coder_races", coder_artifacts)

    decode_path = store / "stage_3_decode_verification.json"
    if stage_valid(
        manifest["stages"].get("decode"), require_current_implementation=True
    ):
        decode_result = json.loads(decode_path.read_text())
    else:
        decode_result = run_decode_stage(source_store, store, inventory, coder_result)
        decode_artifacts = [decode_path]
        decode_artifacts.extend(
            Path(str(row["answer_transcript"]["path"])) for row in decode_result["variants"]
        )
        for row in decode_result["variants"]:
            decode_artifacts.extend(Path(str(fact["path"])) for fact in row["group_answer_files"])
            decode_artifacts.append(Path(str(row["attempt_receipt"]["path"])))
        decode_artifacts.append(Path(str(decode_result["negative_control"]["payload"]["path"])))
        decode_artifacts.append(
            Path(str(decode_result["retention_incident"]["recovered_payload"]["path"]))
        )
        save_stage(resume, manifest, "decode", decode_artifacts)

    result = final_result(store, preflight, inventory, coder_result, decode_result)
    save_stage(resume, manifest, "final", [store / "result.json"])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = parse_args()
    store = args.store.resolve()
    try:
        return execute(args)
    except Exception as exc:
        failure = {
            "schema": "ddm_db1_decode_boundary_failure.v1",
            "axis": AXIS,
            "implementation_sha256": sha256_file(Path(__file__)),
            "argv": sys.argv,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "payload_policy": (
                "all completed candidate outputs and per-group decode checkpoints remain retained"
            ),
        }
        atomic_json(store / f"failures/failure_{time.time_ns()}.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
