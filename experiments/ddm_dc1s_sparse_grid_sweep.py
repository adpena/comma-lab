#!/usr/bin/env python3
"""Run dc1's full-frame sparse-grid hash-question sweep on the real fx5 body.

This is scorer-free rate research.  It walks the exact retained fx5 token field
through the real HPAC conditional law, searches every non-MAP block with a
deterministic best-first product enumerator, retains crash-resumable frame-chunk
checkpoints, and emits one final NPZ bundle per HPAC group plus an actual parsed
question packet.  It does not edit a receiver, build an archive, or claim a
score.

The generic search states are streamed one at a time; no exhaustive candidate
table is materialized.  Each search retains its source probability rows, target,
rank, chosen hash length, target digest, candidate count, heap peak, and a digest
chain over the complete ordered search trace.  Every video-derived question bit
is persisted in the final packet and in the corresponding group bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.util
import json
import math
import os
import struct
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DC1_SOURCE = REPO / "experiments/ddm_dc1_decode_time_compute.py"
DEFAULT_STORE = Path("/Volumes/APDataStore/pact/ddm_dc1_decode_time_compute/full_frame_sparse_sweep")
DEFAULT_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_fx5/candidate_runtime_fx5")
DEFAULT_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_fx5/decode_r1/inflated/"
    ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
DEFAULT_LEDGER = Path("/Volumes/APDataStore/pact/ddm_fx5/work/bits_per_frame_e1_19member.npy")
DEFAULT_CONTROL = Path("/Volumes/APDataStore/pact/ddm_fx5/retained/S1_control_600.json")
DC1_PACKET = Path(
    "/Volumes/APDataStore/pact/ddm_dc1_decode_time_compute/"
    "run_c_fixed_grid_g0/retained/fixed_grid_sparse_group_000.bin"
)
DC1_DECODED = Path(
    "/Volumes/APDataStore/pact/ddm_dc1_decode_time_compute/"
    "run_c_fixed_grid_g0/retained/fixed_grid_sparse_group_000_decoded.u8"
)

PINNED_DC1_COMMIT = "badc6e2e9b"
EXPECTED_PROTOTYPE_SHA256 = "51d537ba0b1ac4db835e4c162ec624932498a2ea5c176a260fe12b8a7b18e8bd"
EXPECTED_ARCHIVE_SHA256 = "4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841"
EXPECTED_TOKEN_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
EXPECTED_LEDGER_SHA256 = "0585b0d98ba2958be3e20021641dd0a74bc61714d1434cab16efb005320418df"
EXPECTED_CONTROL_SHA256 = "b5a2668f499bc7060f5c09fa36b8435fd98bae80e62d4d5a0fc2ddc3713c2685"
EXPECTED_DC1_PACKET_SHA256 = "3e688f11533311a307760d45176a88d99dd4b573848713ed552ddf0ac398ca77"
EXPECTED_DC1_DECODED_SHA256 = "cbb99eb650b335230c4f21ca21323d85eb919d79df140f0337da2c19682e52a7"

QUESTION_DOMAIN = b"ddm_dc1_hash_region_v1\x00"
PACKET_MAGIC = b"DC1F"
PACKET_VERSION = 1
NUM_CLASSES = 5
SEED = 20260821
FRAME_COUNT = 600
MAX_SYMBOLS = 8
STORE_ESTIMATE_BYTES = 8 << 30
STORE_RESERVE_BYTES = 8 << 30
WARM_NATIVE_HEADROOM_SECONDS = 691.470886
COLD_NATIVE_HEADROOM_SECONDS = 211.470886
DC1_REFERENCE_CANDIDATES_PER_SECOND = 363_456.964
BLOCK_SIZE_CODES = {1: 0, 2: 1, 4: 2, 8: 3}
BLOCK_CODE_SIZES = {value: key for key, value in BLOCK_SIZE_CODES.items()}


class Dc1sError(RuntimeError):
    """Fail-closed sparse-grid sweep error."""


@dataclass(frozen=True)
class SearchResult:
    """Exact retained facts for one non-MAP block search."""

    target_rank: int
    rank_bits: int
    hash_bits: int
    target_digest: bytes
    candidates_examined: int
    heap_peak: int
    search_seconds: float
    trace_sha256: bytes


@dataclass(frozen=True)
class GroupEncoding:
    """One real body encoding for one group and block-size option."""

    group: int
    block_size: int
    total_blocks: int
    non_map_indices: np.ndarray
    hash_lengths: np.ndarray
    target_digests: np.ndarray
    elias_l: int
    hash_mode: str
    hash_param: int
    position_bits: int
    hash_length_bits: int
    hash_prefix_bits: int
    body_bits: list[int]


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


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with partial.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Dc1sError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def integer_bits(value: int, width: int) -> list[int]:
    if width < 0 or value < 0 or value >= (1 << width if width else 1):
        raise Dc1sError("integer does not fit its declared width")
    return [(value >> (width - 1 - index)) & 1 for index in range(width)]


def bits_integer(bits: Iterable[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def pack_bits(bits: list[int]) -> bytes:
    payload = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        if bit:
            payload[index // 8] |= 1 << (7 - index % 8)
    return bytes(payload)


def unpack_bits(payload: bytes, count: int) -> list[int]:
    if count < 0 or count > 8 * len(payload):
        raise Dc1sError("meaningful bit count exceeds retained payload")
    return [(payload[index // 8] >> (7 - index % 8)) & 1 for index in range(count)]


def common_prefix_bits(left: bytes, right: bytes) -> int:
    for index, (lbyte, rbyte) in enumerate(zip(left, right, strict=True)):
        xor = lbyte ^ rbyte
        if xor:
            return index * 8 + (8 - xor.bit_length())
    return len(left) * 8


def candidate_digest(group: int, symbols: int, candidate: np.ndarray) -> bytes:
    header = QUESTION_DOMAIN + struct.pack("<HB", group, symbols)
    return hashlib.sha256(header + np.asarray(candidate, dtype=np.uint8).tobytes()).digest()


def candidate_code(candidate: np.ndarray) -> int:
    code = 0
    for symbol in np.asarray(candidate, dtype=np.uint8):
        code = code * NUM_CLASSES + int(symbol)
    return code


def best_first_search(rows: np.ndarray, target: np.ndarray, group: int) -> SearchResult:
    """Enumerate the real product law in descending probability, then lexical order."""
    started = time.perf_counter()
    rows = np.asarray(rows, dtype=np.float64)
    target = np.asarray(target, dtype=np.uint8)
    symbols = len(target)
    if rows.shape != (symbols, NUM_CLASSES) or symbols < 1 or symbols > MAX_SYMBOLS:
        raise Dc1sError("search block has an invalid shape")

    lexical = np.arange(NUM_CLASSES, dtype=np.int64)
    choices = np.empty((symbols, NUM_CLASSES), dtype=np.uint8)
    for column in range(symbols):
        choices[column] = np.lexsort((lexical, -rows[column])).astype(np.uint8)
    logs = np.log2(np.maximum(rows, 1e-300))
    target_digest = candidate_digest(group, symbols, target)
    target_tuple = tuple(int(value) for value in target)

    def state_candidate(state: tuple[int, ...]) -> np.ndarray:
        return np.fromiter(
            (choices[column, state[column]] for column in range(symbols)),
            dtype=np.uint8,
            count=symbols,
        )

    def state_log_probability(state: tuple[int, ...]) -> float:
        total = 0.0
        for column in range(symbols):
            total += float(logs[column, int(choices[column, state[column]])])
        return total

    initial = (0,) * symbols
    first = state_candidate(initial)
    heap: list[tuple[float, int, tuple[int, ...]]] = [
        (-state_log_probability(initial), candidate_code(first), initial)
    ]
    seen = {initial}
    heap_peak = 1
    rank = 0
    max_earlier_prefix = -1
    trace = hashlib.sha256()

    while heap:
        negative_logp, code, state = heapq.heappop(heap)
        candidate = state_candidate(state)
        logp = -negative_logp
        digest = candidate_digest(group, symbols, candidate)
        trace.update(struct.pack("<Id", code, logp))
        trace.update(digest)
        if tuple(int(value) for value in candidate) == target_tuple:
            hash_bits = 0 if rank == 0 else max_earlier_prefix + 1
            if hash_bits > 256:
                raise Dc1sError("SHA-256 did not distinguish the target")
            return SearchResult(
                target_rank=rank,
                rank_bits=rank.bit_length(),
                hash_bits=hash_bits,
                target_digest=target_digest,
                candidates_examined=rank + 1,
                heap_peak=heap_peak,
                search_seconds=time.perf_counter() - started,
                trace_sha256=trace.digest(),
            )
        max_earlier_prefix = max(max_earlier_prefix, common_prefix_bits(digest, target_digest))
        rank += 1
        for column in range(symbols):
            if state[column] + 1 >= NUM_CLASSES:
                continue
            child = list(state)
            child[column] += 1
            child_state = tuple(child)
            if child_state in seen:
                continue
            seen.add(child_state)
            child_candidate = state_candidate(child_state)
            heapq.heappush(
                heap,
                (
                    -state_log_probability(child_state),
                    candidate_code(child_candidate),
                    child_state,
                ),
            )
        heap_peak = max(heap_peak, len(heap))
    raise Dc1sError("best-first search exhausted support before finding the target")


def exhaustive_search_reference(rows: np.ndarray, target: np.ndarray, group: int) -> SearchResult:
    """Small-support test oracle for the heap search; never used by the sweep."""
    started = time.perf_counter()
    rows = np.asarray(rows, dtype=np.float64)
    target = np.asarray(target, dtype=np.uint8)
    symbols = len(target)
    support = NUM_CLASSES**symbols
    powers = NUM_CLASSES ** np.arange(symbols - 1, -1, -1, dtype=np.int64)
    codes = np.arange(support, dtype=np.int64)
    candidates = ((codes[:, None] // powers[None, :]) % NUM_CLASSES).astype(np.uint8)
    logp = np.zeros(support, dtype=np.float64)
    for column in range(symbols):
        logp += np.log2(np.maximum(rows[column, candidates[:, column]], 1e-300))
    order = np.lexsort((codes, -logp))
    target_code = candidate_code(target)
    rank = int(np.flatnonzero(order == target_code)[0])
    target_digest = candidate_digest(group, symbols, target)
    max_prefix = -1
    trace = hashlib.sha256()
    for code_value in order[: rank + 1]:
        index = int(code_value)
        digest = candidate_digest(group, symbols, candidates[index])
        trace.update(struct.pack("<Id", index, float(logp[index])))
        trace.update(digest)
        if index != target_code:
            max_prefix = max(max_prefix, common_prefix_bits(digest, target_digest))
    return SearchResult(
        target_rank=rank,
        rank_bits=rank.bit_length(),
        hash_bits=0 if rank == 0 else max_prefix + 1,
        target_digest=target_digest,
        candidates_examined=rank + 1,
        heap_peak=0,
        search_seconds=time.perf_counter() - started,
        trace_sha256=trace.digest(),
    )


def elias_fano_encode(values: np.ndarray, population: int) -> tuple[int, list[int]]:
    """Encode a sorted subset exactly with Elias-Fano low bits plus unary highs."""
    values = np.asarray(values, dtype=np.int64)
    count = len(values)
    if population < 0 or count > population:
        raise Dc1sError("invalid Elias-Fano population")
    if count == 0:
        return 0, []
    if np.any(values < 0) or np.any(values >= population) or np.any(values[1:] <= values[:-1]):
        raise Dc1sError("Elias-Fano values are not a sorted subset")
    ratio = population // count
    low_width = max(0, ratio.bit_length() - 1)
    low_mask = (1 << low_width) - 1
    low_bits: list[int] = []
    for value in values:
        low_bits.extend(integer_bits(int(value) & low_mask, low_width))
    high_span = ((population - 1) >> low_width) + 1 if population else 0
    high_bits = [0] * (high_span + count)
    for index, value in enumerate(values):
        high_bits[(int(value) >> low_width) + index] = 1
    return low_width, low_bits + high_bits


def elias_fano_decode(bits: list[int], population: int, count: int, low_width: int) -> np.ndarray:
    if count == 0:
        if bits:
            raise Dc1sError("empty Elias-Fano set carried bits")
        return np.empty(0, dtype=np.int64)
    low_count = count * low_width
    high_span = ((population - 1) >> low_width) + 1
    expected = low_count + high_span + count
    if len(bits) != expected:
        raise Dc1sError("Elias-Fano bit count does not match its header")
    lows = [bits_integer(bits[index * low_width : (index + 1) * low_width]) for index in range(count)]
    ones = [index for index, bit in enumerate(bits[low_count:]) if bit]
    if len(ones) != count:
        raise Dc1sError("Elias-Fano unary high field has the wrong cardinality")
    values = np.asarray(
        [((ones[index] - index) << low_width) | lows[index] for index in range(count)],
        dtype=np.int64,
    )
    if np.any(values >= population) or np.any(values[1:] <= values[:-1]):
        raise Dc1sError("Elias-Fano decoded an invalid subset")
    return values


def digest_prefix_bits(digest: bytes | np.ndarray, count: int) -> list[int]:
    raw = bytes(np.asarray(digest, dtype=np.uint8))
    return [(raw[index // 8] >> (7 - index % 8)) & 1 for index in range(count)]


def build_group_encoding(
    group: int,
    block_size: int,
    total_blocks: int,
    indices: np.ndarray,
    hash_lengths: np.ndarray,
    digests: np.ndarray,
) -> GroupEncoding:
    indices = np.asarray(indices, dtype=np.int64)
    hash_lengths = np.asarray(hash_lengths, dtype=np.uint8)
    digests = np.asarray(digests, dtype=np.uint8)
    if len(indices) != len(hash_lengths) or digests.shape != (len(indices), 32):
        raise Dc1sError("group search arrays disagree")
    elias_l, position = elias_fano_encode(indices, total_blocks)
    max_hash = int(hash_lengths.max()) if len(hash_lengths) else 0
    uniform_bits = len(hash_lengths) * max_hash
    length_width = max(1, max_hash.bit_length()) if len(hash_lengths) else 0
    individual_bits = len(hash_lengths) * length_width + int(hash_lengths.sum())
    if len(hash_lengths) and (individual_bits < uniform_bits or max_hash > 31):
        mode = "individual"
        param = length_width
        length_bits: list[int] = []
        for length in hash_lengths:
            length_bits.extend(integer_bits(int(length), length_width))
        prefix_bits: list[int] = []
        for digest, length in zip(digests, hash_lengths, strict=True):
            prefix_bits.extend(digest_prefix_bits(digest, int(length)))
    else:
        mode = "uniform"
        param = max_hash
        length_bits = []
        prefix_bits = []
        for digest in digests:
            prefix_bits.extend(digest_prefix_bits(digest, max_hash))
    if param > 31:
        raise Dc1sError("group hash parameter does not fit the packet header")
    body = position + length_bits + prefix_bits
    return GroupEncoding(
        group=group,
        block_size=block_size,
        total_blocks=total_blocks,
        non_map_indices=indices,
        hash_lengths=hash_lengths,
        target_digests=digests,
        elias_l=elias_l,
        hash_mode=mode,
        hash_param=param,
        position_bits=len(position),
        hash_length_bits=len(length_bits),
        hash_prefix_bits=len(prefix_bits),
        body_bits=body,
    )


def group_header(encoding: GroupEncoding) -> bytes:
    flags = BLOCK_SIZE_CODES[encoding.block_size]
    if encoding.hash_mode == "individual":
        flags |= 1 << 2
    flags |= encoding.hash_param << 3
    return struct.pack("<IBB", len(encoding.non_map_indices), flags, encoding.elias_l)


def parse_group_body(
    bits: list[int],
    cursor: int,
    population: int,
    header: bytes,
) -> tuple[int, int, np.ndarray, np.ndarray, list[list[int]]]:
    count, flags, elias_l = struct.unpack("<IBB", header)
    block_size = BLOCK_CODE_SIZES[flags & 0x03]
    individual = bool(flags & (1 << 2))
    param = flags >> 3
    position_count = 0 if count == 0 else count * elias_l + (((population - 1) >> elias_l) + 1) + count
    position_bits = bits[cursor : cursor + position_count]
    cursor += position_count
    indices = elias_fano_decode(position_bits, population, count, elias_l)
    if individual:
        length_count = count * param
        raw_lengths = bits[cursor : cursor + length_count]
        cursor += length_count
        lengths = np.asarray(
            [bits_integer(raw_lengths[index * param : (index + 1) * param]) for index in range(count)],
            dtype=np.uint8,
        )
    else:
        lengths = np.full(count, param, dtype=np.uint8)
    prefixes: list[list[int]] = []
    for length in lengths:
        width = int(length)
        prefixes.append(bits[cursor : cursor + width])
        cursor += width
    return cursor, block_size, indices, lengths, prefixes


def verify_pins(args: argparse.Namespace) -> dict[str, object]:
    archive = args.runtime_root / "archive.zip"
    expected = {
        DC1_SOURCE: EXPECTED_PROTOTYPE_SHA256,
        archive: EXPECTED_ARCHIVE_SHA256,
        args.tokens: EXPECTED_TOKEN_SHA256,
        args.ledger: EXPECTED_LEDGER_SHA256,
        args.control: EXPECTED_CONTROL_SHA256,
        DC1_PACKET: EXPECTED_DC1_PACKET_SHA256,
        DC1_DECODED: EXPECTED_DC1_DECODED_SHA256,
    }
    facts: dict[str, object] = {}
    for path, digest in expected.items():
        if not path.is_file():
            raise Dc1sError(f"pinned source is absent: {path}")
        actual = sha256_file(path)
        if actual != digest:
            raise Dc1sError(f"pinned source changed: {path}: {actual} != {digest}")
        facts[path.name] = file_fact(path)
    control = json.loads(args.control.read_text())
    if not control.get("byte_identical") or int(control.get("frames", 0)) != FRAME_COUNT:
        raise Dc1sError("the retained real-coder control is not a byte-identical n600 control")
    pinned_blob = subprocess.run(
        ["git", "show", f"{PINNED_DC1_COMMIT}:experiments/ddm_dc1_decode_time_compute.py"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(pinned_blob).hexdigest() != EXPECTED_PROTOTYPE_SHA256:
        raise Dc1sError("dc1's pinned commit does not contain the expected prototype")
    return {
        "dc1_commit": PINNED_DC1_COMMIT,
        "facts": facts,
        "packet_prefix_sha256": EXPECTED_DC1_PACKET_SHA256[:8],
        "decoded_prefix_sha256": EXPECTED_DC1_DECODED_SHA256[:8],
    }


def storage_preflight(store: Path, pins: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    store.mkdir(parents=True, exist_ok=True)
    stats = os.statvfs(store)
    free = stats.f_bavail * stats.f_frsize
    required = STORE_ESTIMATE_BYTES + STORE_RESERVE_BYTES
    if free < required:
        raise Dc1sError(f"APDataStore preflight failed: {free} free bytes < {required} required")
    receipt = {
        "schema": "ddm_dc1s_storage_preflight.v1",
        "axis": "[macOS-CPU advisory / scorer-free]",
        "store": str(store),
        "free_bytes_before": free,
        "estimated_sweep_bytes": STORE_ESTIMATE_BYTES,
        "reserve_bytes": STORE_RESERVE_BYTES,
        "passed": True,
        "implementation_sha256": sha256_file(Path(__file__)),
        "argv": sys.argv,
        "seed": SEED,
        "block_sizes": args.block_sizes,
        "frame_start": args.frame_start,
        "frame_end": args.frame_end,
        "chunk_frames": args.chunk_frames,
        "pins": pins,
        "rebuildability": (
            "Frame-chunk checkpoints are deterministic from the pinned fx5 archive, token field, "
            "ledger, source commit, seed, and argv; incomplete .partial files are scratch."
        ),
    }
    atomic_json(store / "preflight.json", receipt)
    return receipt


def empty_record_lists() -> dict[str, list[Any]]:
    return {
        "group": [],
        "block_size": [],
        "block_index": [],
        "symbols": [],
        "direct_bits": [],
        "target_rank": [],
        "rank_bits": [],
        "hash_bits": [],
        "target_digest": [],
        "candidates_examined": [],
        "heap_peak": [],
        "search_seconds": [],
        "trace_sha256": [],
        "coding_rows": [],
        "target": [],
    }


def append_record(
    records: dict[str, list[Any]],
    group: int,
    block_size: int,
    block_index: int,
    rows: np.ndarray,
    target: np.ndarray,
    direct_bits: float,
    result: SearchResult,
) -> None:
    padded_rows = np.zeros((MAX_SYMBOLS, NUM_CLASSES), dtype=np.float64)
    padded_target = np.full(MAX_SYMBOLS, 255, dtype=np.uint8)
    padded_rows[: len(target)] = rows
    padded_target[: len(target)] = target
    values = {
        "group": group,
        "block_size": block_size,
        "block_index": block_index,
        "symbols": len(target),
        "direct_bits": direct_bits,
        "target_rank": result.target_rank,
        "rank_bits": result.rank_bits,
        "hash_bits": result.hash_bits,
        "target_digest": np.frombuffer(result.target_digest, dtype=np.uint8).copy(),
        "candidates_examined": result.candidates_examined,
        "heap_peak": result.heap_peak,
        "search_seconds": result.search_seconds,
        "trace_sha256": np.frombuffer(result.trace_sha256, dtype=np.uint8).copy(),
        "coding_rows": padded_rows,
        "target": padded_target,
    }
    for key, value in values.items():
        records[key].append(value)


def record_arrays(records: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    count = len(records["group"])
    shapes = {
        "target_digest": (count, 32),
        "trace_sha256": (count, 32),
        "coding_rows": (count, MAX_SYMBOLS, NUM_CLASSES),
        "target": (count, MAX_SYMBOLS),
    }
    dtypes = {
        "group": np.uint16,
        "block_size": np.uint8,
        "block_index": np.uint32,
        "symbols": np.uint8,
        "direct_bits": np.float64,
        "target_rank": np.uint32,
        "rank_bits": np.uint8,
        "hash_bits": np.uint8,
        "target_digest": np.uint8,
        "candidates_examined": np.uint32,
        "heap_peak": np.uint32,
        "search_seconds": np.float64,
        "trace_sha256": np.uint8,
        "coding_rows": np.float64,
        "target": np.uint8,
    }
    output: dict[str, np.ndarray] = {}
    for key, dtype in dtypes.items():
        if records[key]:
            output[key] = np.asarray(records[key], dtype=dtype)
        else:
            output[key] = np.empty(shapes.get(key, (0,)), dtype=dtype)
    return output


def analyze_group(
    records: dict[str, list[Any]],
    frame: int,
    group: int,
    coding: np.ndarray,
    target: np.ndarray,
    block_sizes: tuple[int, ...],
) -> float:
    coding = np.asarray(coding, dtype=np.float64)
    target = np.asarray(target, dtype=np.uint8)
    selected = coding[np.arange(len(target)), target.astype(np.int64)]
    symbol_bits = -np.log2(np.maximum(selected, 1e-300))
    map_symbols = coding.argmax(axis=1).astype(np.uint8)
    mismatch = map_symbols != target
    for block_size in block_sizes:
        starts = np.arange(0, len(target), block_size, dtype=np.int64)
        mismatch_counts = np.add.reduceat(mismatch.astype(np.int64), starts)
        surprising_blocks = np.flatnonzero(mismatch_counts > 0)
        blocks_per_frame = len(starts)
        for local_block in surprising_blocks:
            start = int(starts[int(local_block)])
            stop = min(len(target), start + block_size)
            block_rows = coding[start:stop]
            block_target = target[start:stop]
            result = best_first_search(block_rows, block_target, group)
            if result.target_rank == 0:
                raise Dc1sError("a block marked non-MAP had rank zero")
            append_record(
                records,
                group,
                block_size,
                frame * blocks_per_frame + int(local_block),
                block_rows,
                block_target,
                float(symbol_bits[start:stop].sum()),
                result,
            )
    return float(symbol_bits.sum())


def write_checkpoint(
    store: Path,
    frame_start: int,
    frame_end: int,
    records: dict[str, list[Any]],
    frame_group_bits: np.ndarray,
    group_sizes: np.ndarray,
    corrector_schema: str,
    corrector_state: dict[str, np.ndarray],
) -> dict[str, object]:
    arrays = record_arrays(records)
    metadata = {
        "schema": "ddm_dc1s_frame_chunk.v1",
        "implementation_sha256": sha256_file(Path(__file__)),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "seed": SEED,
        "record_count": len(arrays["group"]),
        "corrector_schema": corrector_schema,
        "corrector_state_keys": sorted(corrector_state),
    }
    path = store / "checkpoints" / f"frames_{frame_start:04d}_{frame_end - 1:04d}.npz"
    atomic_npz(
        path,
        **arrays,
        frame_group_bits=np.asarray(frame_group_bits, dtype=np.float64),
        group_sizes=np.asarray(group_sizes, dtype=np.int64),
        metadata_json=np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8),
        **{f"corrector__{key}": value for key, value in corrector_state.items()},
    )
    return file_fact(path)


def validate_checkpoint(path: Path, expected_sha: str) -> dict[str, object]:
    if sha256_file(path) != expected_sha:
        raise Dc1sError(f"checkpoint hash changed: {path}")
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(bytes(payload["metadata_json"]).decode())
        if metadata["implementation_sha256"] != sha256_file(Path(__file__)):
            raise Dc1sError("checkpoint was produced by a different implementation")
        retained_keys = sorted(key.removeprefix("corrector__") for key in payload.files if key.startswith("corrector__"))
        if retained_keys != metadata.get("corrector_state_keys"):
            raise Dc1sError("checkpoint corrector state keys disagree with metadata")
    return metadata


def load_runtime(args: argparse.Namespace):
    import torch

    dc1 = load_module(DC1_SOURCE, "ddm_dc1s_source")
    jg2 = dc1.load_jg2()
    residual, renderer, renderer_dir = jg2.load_runtime(args.runtime_root)
    parts = residual.read_residual_archive(args.runtime_root / "archive.zip")
    device = torch.device("cpu")
    base_hpac = residual.materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    masks = renderer.group_masks(device)
    sparse = residual._sparse_class(renderer_dir)(model, renderer.EVAL_H, renderer.EVAL_W)
    from runtime.free_corrector import FreeCorrector
    from runtime.hpac_inference import optimize_sparse_evaluator

    optimize_sparse_evaluator(sparse)
    positions = []
    plans = []
    for mask in masks:
        flat = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1)).astype(np.int64)
        positions.append(flat)
        plans.append((torch.from_numpy(flat).to(device), flat))
    token_shape = (renderer.N, renderer.EVAL_H, renderer.EVAL_W)
    tokens = np.memmap(args.tokens, dtype=np.uint8, mode="r", shape=token_shape)
    return {
        "jg2": jg2,
        "residual": residual,
        "renderer": renderer,
        "parts": parts,
        "model": model,
        "sparse": sparse,
        "plans": plans,
        "positions": positions,
        "tokens": tokens,
        "device": device,
        "corrector": FreeCorrector(renderer.EVAL_H * renderer.EVAL_W),
        "cold_corrector": FreeCorrector(renderer.EVAL_H * renderer.EVAL_W),
    }


def restore_corrector_for_chunk(runtime: dict[str, Any], manifest: dict[str, Any], frame_start: int) -> None:
    """Restore the full adaptive corrector state at a resumable chunk boundary."""
    if frame_start == 0:
        return
    predecessor: dict[str, object] | None = None
    for fact in manifest.get("checkpoints", []):
        path = Path(str(fact["path"]))
        metadata = validate_checkpoint(path, str(fact["sha256"]))
        if int(metadata["frame_end"]) == frame_start:
            predecessor = fact
            break
    if predecessor is None:
        raise Dc1sError(f"no retained corrector checkpoint ends at frame {frame_start}")
    with np.load(Path(str(predecessor["path"])), allow_pickle=False) as payload:
        state = {
            key.removeprefix("corrector__"): payload[key].copy()
            for key in payload.files
            if key.startswith("corrector__")
        }
    runtime["jg2"].load_corrector_state(runtime["corrector"], state)


def process_chunk(
    args: argparse.Namespace,
    frame_start: int,
    frame_end: int,
    runtime: dict[str, Any],
    ledger: np.ndarray,
) -> tuple[dict[str, list[Any]], np.ndarray, np.ndarray]:
    import torch

    residual = runtime["residual"]
    renderer = runtime["renderer"]
    parts = runtime["parts"]
    model = runtime["model"]
    sparse = runtime["sparse"]
    plans = runtime["plans"]
    positions = runtime["positions"]
    tokens = runtime["tokens"]
    device = runtime["device"]
    corrector = runtime["corrector"]
    records = empty_record_lists()
    group_sizes = np.asarray([len(flat) for flat in positions], dtype=np.int64)
    frame_group_bits = np.zeros((frame_end - frame_start, len(plans)), dtype=np.float64)
    block_sizes = tuple(args.block_sizes)

    with torch.inference_mode():
        for frame_offset, frame in enumerate(range(frame_start, frame_end)):
            target = np.asarray(tokens[frame], dtype=np.uint8).reshape(-1)
            if frame == 0:
                previous_np = np.zeros((renderer.EVAL_H, renderer.EVAL_W), dtype=np.uint8)
            else:
                previous_np = np.asarray(tokens[frame - 1], dtype=np.uint8)
            previous = torch.from_numpy(previous_np.copy()).to(device=device, dtype=torch.long).unsqueeze(0)
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(torch.tensor([frame], dtype=torch.long), previous)
            if frame:
                boundary = residual._boundary_buckets(previous_np).reshape(-1)
            else:
                boundary = np.full(renderer.EVAL_H * renderer.EVAL_W, 4, dtype=np.uint8)
            corrector.begin_frame(boundary)
            for group, (device_positions, flat_positions) in enumerate(plans):
                base_logits = sparse.selected_logits(current, context, group).cpu().numpy()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = boundary[flat_positions].astype(np.int64) * NUM_CLASSES + predicted
                corrected = base_logits + parts.table.values[feature]
                probability = residual._probability_table(corrected, renderer.HPAC_LOGIT_PRECISION)
                state = corrector.group_state(probability, predicted, flat_positions)
                coding = np.asarray(corrector.coding_row(state), dtype=np.float64)
                symbols = target[flat_positions].astype(np.uint8)
                frame_group_bits[frame_offset, group] = analyze_group(
                    records,
                    frame,
                    group,
                    coding,
                    symbols,
                    block_sizes,
                )
                corrector.observe(state, symbols.astype(np.int64))
                current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(device=device, dtype=torch.long)
            reconstructed = current[0].to(device="cpu", dtype=torch.uint8).numpy().reshape(-1)
            if not np.array_equal(reconstructed, target):
                raise Dc1sError(f"HPAC walk diverged from the retained token field at frame {frame}")
            corrector.end_frame(reconstructed)
            measured = float(frame_group_bits[frame_offset].sum())
            if not math.isclose(measured, float(ledger[frame]), rel_tol=0.0, abs_tol=1e-9):
                raise Dc1sError(f"frame {frame} ideal bits disagree with the retained fx5 ledger")
            print(
                json.dumps(
                    {
                        "frame": frame,
                        "ideal_bits": measured,
                        "non_map_records": len(records["group"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return records, frame_group_bits, group_sizes


def checkpoint_paths_from_manifest(store: Path, manifest: dict[str, Any]) -> list[Path]:
    paths = []
    for fact in manifest.get("checkpoints", []):
        path = Path(str(fact["path"]))
        validate_checkpoint(path, str(fact["sha256"]))
        paths.append(path)
    return paths


def load_all_records(paths: list[Path]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load each retained chunk once for final group bundling and accounting."""
    empty = record_arrays(empty_record_lists())
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in empty}
    frame_group_pieces: list[np.ndarray] = []
    group_sizes: np.ndarray | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            for key in pieces:
                pieces[key].append(payload[key])
            frame_group_pieces.append(payload["frame_group_bits"])
            observed_sizes = payload["group_sizes"].astype(np.int64)
            if group_sizes is None:
                group_sizes = observed_sizes
            elif not np.array_equal(group_sizes, observed_sizes):
                raise Dc1sError("frame chunks disagree on HPAC group sizes")
    records = {
        key: np.concatenate(arrays, axis=0) if arrays else empty[key]
        for key, arrays in pieces.items()
    }
    if group_sizes is None:
        raise Dc1sError("no retained group-size table")
    frame_group_bits = np.concatenate(frame_group_pieces, axis=0)
    return records, frame_group_bits, group_sizes


def repeat_records(group_records: dict[str, np.ndarray], group: int, count: int) -> dict[str, np.ndarray]:
    priorities = []
    for index in range(len(group_records["group"])):
        payload = struct.pack(
            "<IHBII",
            SEED,
            group,
            int(group_records["block_size"][index]),
            int(group_records["block_index"][index]),
            int(group_records["target_rank"][index]),
        )
        priorities.append((hashlib.sha256(payload).digest(), index))
    selected = [index for _, index in sorted(priorities)[:count]]
    repeat_rank = []
    repeat_hash = []
    repeat_trace = []
    for index in selected:
        symbols = int(group_records["symbols"][index])
        result = best_first_search(
            group_records["coding_rows"][index, :symbols],
            group_records["target"][index, :symbols],
            group,
        )
        repeat_rank.append(result.target_rank)
        repeat_hash.append(result.hash_bits)
        repeat_trace.append(np.frombuffer(result.trace_sha256, dtype=np.uint8).copy())
        if result.target_rank != int(group_records["target_rank"][index]):
            raise Dc1sError("repeat search changed target rank")
        if result.hash_bits != int(group_records["hash_bits"][index]):
            raise Dc1sError("repeat search changed hash length")
        if result.trace_sha256 != bytes(group_records["trace_sha256"][index]):
            raise Dc1sError("repeat search changed ordered trace digest")
    return {
        "repeat_record_indices": np.asarray(selected, dtype=np.uint32),
        "repeat_target_rank": np.asarray(repeat_rank, dtype=np.uint32),
        "repeat_hash_bits": np.asarray(repeat_hash, dtype=np.uint8),
        "repeat_trace_sha256": np.asarray(repeat_trace, dtype=np.uint8).reshape(-1, 32),
    }


def option_encoding(
    group: int,
    block_size: int,
    group_size: int,
    frame_count: int,
    records: dict[str, np.ndarray],
) -> GroupEncoding:
    mask = records["block_size"] == block_size
    indices = records["block_index"][mask].astype(np.int64)
    total_blocks = frame_count * math.ceil(group_size / block_size)
    return build_group_encoding(
        group,
        block_size,
        total_blocks,
        indices,
        records["hash_bits"][mask],
        records["target_digest"][mask],
    )


def encoding_metadata(encoding: GroupEncoding) -> dict[str, object]:
    return {
        "group": encoding.group,
        "block_size": encoding.block_size,
        "total_blocks": encoding.total_blocks,
        "non_map_blocks": len(encoding.non_map_indices),
        "elias_l": encoding.elias_l,
        "hash_mode": encoding.hash_mode,
        "hash_param": encoding.hash_param,
        "position_bits": encoding.position_bits,
        "hash_length_bits": encoding.hash_length_bits,
        "hash_prefix_bits": encoding.hash_prefix_bits,
        "body_bits": len(encoding.body_bits),
    }


def wire_hash_lengths(encoding: GroupEncoding) -> np.ndarray:
    """Widths the packet actually transmits after the group's mode choice."""
    if encoding.hash_mode == "individual":
        return encoding.hash_lengths
    return np.full(len(encoding.hash_lengths), encoding.hash_param, dtype=np.uint8)


def finalize(
    args: argparse.Namespace,
    store: Path,
    manifest: dict[str, Any],
    ledger: np.ndarray,
) -> dict[str, object]:
    paths = checkpoint_paths_from_manifest(store, manifest)
    if not paths:
        raise Dc1sError("no frame checkpoints were retained")
    all_records, frame_group_bits, group_sizes = load_all_records(paths)
    frame_count = args.frame_end - args.frame_start
    if args.frame_start != 0:
        raise Dc1sError("final packet requires a sweep beginning at frame zero")

    group_encodings: list[GroupEncoding] = []
    group_receipts: list[dict[str, object]] = []
    total_candidates = 0
    total_search_seconds = 0.0
    total_non_map = 0
    group_header_bytes = bytearray()
    all_body_bits: list[int] = []
    retained_dir = store / "retained"
    retained_dir.mkdir(parents=True, exist_ok=True)

    for group, group_size in enumerate(group_sizes):
        mask = all_records["group"] == group
        records = {key: value[mask] for key, value in all_records.items()}
        order = np.lexsort((records["block_index"], records["block_size"]))
        records = {key: value[order] for key, value in records.items()}
        direct_group_bits = float(frame_group_bits[:, group].sum())
        options = [
            option_encoding(group, block_size, int(group_size), frame_count, records)
            for block_size in args.block_sizes
        ]
        chosen = min(options, key=lambda item: (len(item.body_bits), -item.block_size))
        repeats = repeat_records(records, group, args.repeat_per_group)
        option_receipts = [
            {
                **encoding_metadata(option),
                "direct_ideal_bits": direct_group_bits,
                "credit_bits_after_group_tax": direct_group_bits - len(option.body_bits) - 48,
            }
            for option in options
        ]
        chosen_receipt = {
            **encoding_metadata(chosen),
            "direct_ideal_bits": direct_group_bits,
            "credit_bits_after_group_tax": direct_group_bits - len(chosen.body_bits) - 48,
            "rank_table_format": "sparse non-MAP rows keyed by block_index; omitted MAP rows have rank_bits=0",
        }
        option_json = json.dumps(option_receipts, sort_keys=True).encode()
        chosen_json = json.dumps(chosen_receipt, sort_keys=True).encode()
        body_offsets = [0]
        body_bits_flat: list[int] = []
        for option in options:
            body_bits_flat.extend(option.body_bits)
            body_offsets.append(len(body_bits_flat))
        bundle = retained_dir / f"group_{group:03d}.npz"
        atomic_npz(
            bundle,
            **records,
            **repeats,
            option_metadata_json=np.frombuffer(option_json, dtype=np.uint8),
            chosen_metadata_json=np.frombuffer(chosen_json, dtype=np.uint8),
            option_body_offsets=np.asarray(body_offsets, dtype=np.uint64),
            option_body_bits=np.asarray(body_bits_flat, dtype=np.uint8),
        )
        group_encodings.append(chosen)
        group_header_bytes.extend(group_header(chosen))
        all_body_bits.extend(chosen.body_bits)
        chosen_mask = records["block_size"] == chosen.block_size
        chosen_candidates = int(records["candidates_examined"][chosen_mask].astype(np.uint64).sum())
        chosen_seconds = float(records["search_seconds"][chosen_mask].sum())
        total_candidates += chosen_candidates
        total_search_seconds += chosen_seconds
        total_non_map += int(chosen_mask.sum())
        group_receipts.append(
            {
                "chosen": chosen_receipt,
                "options": option_receipts,
                "chosen_candidates_examined": chosen_candidates,
                "chosen_search_seconds": chosen_seconds,
                "repeat_count": len(repeats["repeat_record_indices"]),
                "repeat_identical": True,
                "bundle": file_fact(bundle),
            }
        )

    top_header = struct.pack("<4sBBHH", PACKET_MAGIC, PACKET_VERSION, 0, frame_count, len(group_sizes))
    packet = top_header + bytes(group_header_bytes) + pack_bits(all_body_bits)
    packet_path = retained_dir / "sparse_grid_packet.bin"
    atomic_bytes(packet_path, packet)
    repeat_path = retained_dir / "sparse_grid_packet.repeat.bin"
    atomic_bytes(repeat_path, top_header + bytes(group_header_bytes) + pack_bits(all_body_bits))
    if sha256_file(packet_path) != sha256_file(repeat_path):
        raise Dc1sError("packet construction was not byte-identical on repeat")

    received = packet_path.read_bytes()
    magic, version, reserved, wire_frames, wire_groups = struct.unpack_from("<4sBBHH", received)
    if (magic, version, reserved, wire_frames, wire_groups) != (
        PACKET_MAGIC,
        PACKET_VERSION,
        0,
        frame_count,
        len(group_sizes),
    ):
        raise Dc1sError("retained packet top header did not parse back")
    header_offset = struct.calcsize("<4sBBHH")
    group_header_width = struct.calcsize("<IBB")
    headers_end = header_offset + wire_groups * group_header_width
    body = unpack_bits(received[headers_end:], len(all_body_bits))
    cursor = 0
    for group, (encoding, group_size) in enumerate(zip(group_encodings, group_sizes, strict=True)):
        header = received[
            header_offset + group * group_header_width : header_offset + (group + 1) * group_header_width
        ]
        cursor, block_size, indices, lengths, prefixes = parse_group_body(
            body,
            cursor,
            frame_count * math.ceil(int(group_size) / encoding.block_size),
            header,
        )
        if block_size != encoding.block_size or not np.array_equal(indices, encoding.non_map_indices):
            raise Dc1sError("retained packet sparse positions did not parse back")
        transmitted_lengths = wire_hash_lengths(encoding)
        if not np.array_equal(lengths, transmitted_lengths):
            raise Dc1sError("retained packet hash lengths did not parse back")
        expected_prefixes = [
            digest_prefix_bits(digest, int(length))
            for digest, length in zip(encoding.target_digests, transmitted_lengths, strict=True)
        ]
        if prefixes != expected_prefixes:
            raise Dc1sError("retained packet hash prefixes did not parse back")
    if cursor != len(body):
        raise Dc1sError("retained packet left meaningful bits unconsumed")

    baseline_ideal_bits = float(ledger[args.frame_start : args.frame_end].sum())
    fixed_header_bits = len(top_header) * 8
    group_tax_bits = len(group_header_bytes) * 8
    meaningful_body_bits = len(all_body_bits)
    packet_bits = len(packet) * 8
    final_padding_bits = packet_bits - fixed_header_bits - group_tax_bits - meaningful_body_bits
    honest_credit_bits = baseline_ideal_bits - packet_bits
    measured_throughput = total_candidates / total_search_seconds if total_search_seconds else math.inf
    conservative_throughput = min(measured_throughput, DC1_REFERENCE_CANDIDATES_PER_SECOND)
    added_wall_seconds = total_candidates / conservative_throughput if total_candidates else 0.0
    result = {
        "schema": "ddm_dc1s_sparse_grid_sweep.v1",
        "axis": "[macOS-CPU advisory / scorer-free real-fx5 full-frame rate measurement]",
        "score_claim": False,
        "archive_candidate_built": False,
        "implementation_sha256": sha256_file(Path(__file__)),
        "selection_mode": "full_population_n600" if frame_count == FRAME_COUNT else "declared_frame_scope",
        "frames": frame_count,
        "population_frames": FRAME_COUNT,
        "seed": SEED,
        "block_sizes_swept": args.block_sizes,
        "groups": group_receipts,
        "accounting": {
            "baseline_hpac_ideal_bits": baseline_ideal_bits,
            "fixed_header_bits": fixed_header_bits,
            "group_tax_bits": group_tax_bits,
            "meaningful_body_bits": meaningful_body_bits,
            "final_padding_bits": final_padding_bits,
            "actual_packet_bits": packet_bits,
            "actual_packet_bytes": len(packet),
            "honest_credit_bits": honest_credit_bits,
            "honest_credit_bytes": honest_credit_bits / 8.0,
            "member_credit_vs_fx5_token_stream_113777_bytes": (
                113_777 - len(packet) if frame_count == FRAME_COUNT else None
            ),
            "stop_rule_credit_at_least_3kb": honest_credit_bits >= 3 * 1024 * 8,
        },
        "search": {
            "chosen_non_map_blocks": total_non_map,
            "chosen_candidates_examined": total_candidates,
            "local_search_seconds": total_search_seconds,
            "local_candidates_per_second": measured_throughput,
            "dc1_reference_candidates_per_second": DC1_REFERENCE_CANDIDATES_PER_SECOND,
            "wall_projection_throughput": conservative_throughput,
            "projected_added_decode_seconds": added_wall_seconds,
            "cold_native_headroom_seconds": COLD_NATIVE_HEADROOM_SECONDS,
            "warm_native_headroom_seconds": WARM_NATIVE_HEADROOM_SECONDS,
            "cold_wall_pass": added_wall_seconds <= COLD_NATIVE_HEADROOM_SECONDS,
            "warm_wall_pass": added_wall_seconds <= WARM_NATIVE_HEADROOM_SECONDS,
            "projection_not_t4_measurement": True,
        },
        "retention": {
            "group_bundle_count": len(group_receipts),
            "packet": file_fact(packet_path),
            "packet_repeat": file_fact(repeat_path),
            "packet_repeat_identical": True,
            "checkpoints": manifest["checkpoints"],
        },
        "pins": json.loads((store / "preflight.json").read_text())["pins"],
    }
    result_path = retained_dir / "result.json"
    atomic_json(result_path, result)
    return result


def run_self_tests() -> None:
    rng = np.random.default_rng(SEED)
    for symbols in (1, 2, 3, 4):
        rows = rng.random((symbols, NUM_CLASSES), dtype=np.float64)
        rows /= rows.sum(axis=1, keepdims=True)
        target = rng.integers(0, NUM_CLASSES, size=symbols, dtype=np.uint8)
        actual = best_first_search(rows, target, 7)
        expected = exhaustive_search_reference(rows, target, 7)
        if (
            actual.target_rank,
            actual.hash_bits,
            actual.target_digest,
            actual.trace_sha256,
        ) != (
            expected.target_rank,
            expected.hash_bits,
            expected.target_digest,
            expected.trace_sha256,
        ):
            raise Dc1sError("heap search differs from exhaustive reference")
    for population, values in (
        (10, np.asarray([], dtype=np.int64)),
        (10, np.asarray([0], dtype=np.int64)),
        (10, np.asarray([1, 3, 9], dtype=np.int64)),
        (1000, np.asarray([2, 17, 99, 512, 999], dtype=np.int64)),
    ):
        low, bits = elias_fano_encode(values, population)
        decoded = elias_fano_decode(bits, population, len(values), low)
        if not np.array_equal(decoded, values):
            raise Dc1sError("Elias-Fano self-test failed")
    digests = np.asarray(
        [np.frombuffer(hashlib.sha256(bytes([index])).digest(), dtype=np.uint8) for index in range(3)],
        dtype=np.uint8,
    )
    for lengths in (np.asarray([3, 3, 3], dtype=np.uint8), np.asarray([2, 9, 3], dtype=np.uint8)):
        encoding = build_group_encoding(
            7,
            4,
            100,
            np.asarray([1, 17, 99], dtype=np.int64),
            lengths,
            digests,
        )
        header = group_header(encoding)
        cursor, block_size, indices, parsed_lengths, prefixes = parse_group_body(
            encoding.body_bits,
            0,
            encoding.total_blocks,
            header,
        )
        transmitted = wire_hash_lengths(encoding)
        expected_prefixes = [
            digest_prefix_bits(digest, int(length))
            for digest, length in zip(encoding.target_digests, transmitted, strict=True)
        ]
        if cursor != len(encoding.body_bits) or block_size != encoding.block_size:
            raise Dc1sError("group packet self-test left bits or changed block size")
        if not np.array_equal(indices, encoding.non_map_indices):
            raise Dc1sError("group packet self-test changed sparse positions")
        if not np.array_equal(parsed_lengths, transmitted) or prefixes != expected_prefixes:
            raise Dc1sError("group packet self-test changed hash questions")
    print(json.dumps({"self_tests": "passed", "seed": SEED}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--block-sizes", default="1,2,4,8")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=FRAME_COUNT)
    parser.add_argument("--chunk-frames", type=int, default=20)
    parser.add_argument("--repeat-per-group", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.store = args.store.resolve()
    args.runtime_root = args.runtime_root.resolve()
    args.tokens = args.tokens.resolve()
    args.ledger = args.ledger.resolve()
    args.control = args.control.resolve()
    if args.self_test:
        return args
    if args.resume_from is None:
        raise Dc1sError("--resume-from is mandatory and must name <store>/manifest.json")
    args.resume_from = args.resume_from.resolve()
    if args.resume_from != args.store / "manifest.json":
        raise Dc1sError("--resume-from must name <store>/manifest.json")
    block_sizes = tuple(sorted({int(item) for item in args.block_sizes.split(",") if item.strip()}))
    if not block_sizes or any(size not in BLOCK_SIZE_CODES for size in block_sizes):
        raise Dc1sError("--block-sizes must be a subset of 1,2,4,8")
    args.block_sizes = block_sizes
    if not (0 <= args.frame_start < args.frame_end <= FRAME_COUNT):
        raise Dc1sError("frame scope must satisfy 0 <= start < end <= 600")
    if args.chunk_frames < 1 or args.chunk_frames > 120:
        raise Dc1sError("--chunk-frames must be in 1..120")
    if args.repeat_per_group < 1 or args.repeat_per_group > 32:
        raise Dc1sError("--repeat-per-group must be in 1..32")
    return args


def main(argv: list[str] | None = None) -> int:
    args = normalize_args(build_parser().parse_args(argv))
    if args.self_test:
        run_self_tests()
        return 0
    pins = verify_pins(args)
    preflight = storage_preflight(args.store, pins, args)
    ledger = np.load(args.ledger, allow_pickle=False)
    if ledger.shape != (FRAME_COUNT,):
        raise Dc1sError("fx5 ideal-bit ledger is not n600")

    if args.resume_from.is_file():
        manifest = json.loads(args.resume_from.read_text())
        if manifest.get("implementation_sha256") != sha256_file(Path(__file__)):
            raise Dc1sError("resume manifest belongs to a different implementation")
        if tuple(manifest.get("block_sizes", ())) != args.block_sizes:
            raise Dc1sError("resume manifest block sizes differ")
        if (int(manifest.get("frame_start", -1)), int(manifest.get("frame_end", -1))) != (
            args.frame_start,
            args.frame_end,
        ):
            raise Dc1sError("resume manifest frame scope differs")
    else:
        manifest = {
            "schema": "ddm_dc1s_manifest.v1",
            "implementation_sha256": sha256_file(Path(__file__)),
            "frame_start": args.frame_start,
            "frame_end": args.frame_end,
            "block_sizes": args.block_sizes,
            "seed": SEED,
            "checkpoints": [],
            "preflight": file_fact(args.store / "preflight.json"),
        }
        atomic_json(args.resume_from, manifest)

    completed = set()
    for fact in manifest.get("checkpoints", []):
        metadata = validate_checkpoint(Path(str(fact["path"])), str(fact["sha256"]))
        completed.add((int(metadata["frame_start"]), int(metadata["frame_end"])))
    runtime = None
    for start in range(args.frame_start, args.frame_end, args.chunk_frames):
        end = min(args.frame_end, start + args.chunk_frames)
        if (start, end) in completed:
            continue
        stats = os.statvfs(args.store)
        if stats.f_bavail * stats.f_frsize < STORE_RESERVE_BYTES:
            raise Dc1sError("APDataStore fell below the retained reserve during the sweep")
        if runtime is None:
            runtime = load_runtime(args)
        restore_corrector_for_chunk(runtime, manifest, start)
        records, frame_group_bits, group_sizes = process_chunk(args, start, end, runtime, ledger)
        corrector_state = runtime["jg2"].corrector_state(runtime["corrector"])
        lost = runtime["jg2"].uncaptured_divergent_state(
            runtime["corrector"],
            runtime["cold_corrector"],
            set(corrector_state),
        )
        if lost:
            raise Dc1sError(f"corrector checkpoint would lose adaptive state: {lost[:8]}")
        fact = write_checkpoint(
            args.store,
            start,
            end,
            records,
            frame_group_bits,
            group_sizes,
            runtime["jg2"].CHECKPOINT_SCHEMA,
            corrector_state,
        )
        manifest["checkpoints"].append(fact)
        manifest["completed_through_frame"] = end
        atomic_json(args.resume_from, manifest)
        print(json.dumps({"checkpoint": fact, "completed_through_frame": end}, sort_keys=True), flush=True)

    expected_chunks = math.ceil((args.frame_end - args.frame_start) / args.chunk_frames)
    if len(manifest["checkpoints"]) != expected_chunks:
        raise Dc1sError("resume manifest does not cover every requested frame chunk")
    result = finalize(args, args.store, manifest, ledger)
    manifest["complete"] = True
    manifest["result"] = file_fact(args.store / "retained/result.json")
    manifest["storage_preflight"] = preflight
    atomic_json(args.resume_from, manifest)
    print(json.dumps({"accounting": result["accounting"], "search": result["search"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
