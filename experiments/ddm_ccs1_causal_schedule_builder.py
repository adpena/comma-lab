#!/usr/bin/env python3
"""Build and byte-close the CCS1 receiver-causal schedule on the AFR1 field.

CCS1 is a lossless replacement probability model.  Its nonlinear leaf table is
indexed only by values available at the exact decode step: two classes from the
fully decoded previous frame, two classes from earlier groups of the current
frame, a boundary state computed from that decoded prefix, and deterministic
position cells.  A small learned backoff table covers contexts without a leaf.

The fit uses LM1's 20 contiguous blocks of 30 frames, with blocks 1, 5, 9, 13,
and 17 held out.  The seed controls the sampled training positions.  Every
materialized model/coder payload is retained; fitting and full encoding have
immutable stage checkpoints and can resume from disk.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import lzma
import os
import shutil
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIELD = Path(
    "/Volumes/VertigoDataTier/pact/ddm_bl1_per_position_bit_allocation/"
    "measurement_v1/retained/fields/decoded_tokens_instrumented.u8"
)
BASE_RUNTIME = Path("/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/runtime_candidate_native")
BASE_ARCHIVE = BASE_RUNTIME / "archive.zip"
RC64_BASE = Path("/Volumes/APDataStore/pact/ddm_rr2_encoder_build/work/rc64_backend_checkpoint.c")
ROUTE_B = REPO / "experiments/ddm_rc64p_native_cpu_decode/route_b_rc64.py"

FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
BASE_ARCHIVE_SHA256 = "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"
BASE_RAW_SHA256 = "7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7"
RC64_BASE_SHA256 = "5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6"

N, H, W, K = 600, 384, 512, 5
PLANE = H * W
FIELD_BYTES = N * PLANE
PATCH, DELTA, GROUPS = 64, 2, 190
UNK = 5
ALPHABET = 6
BOUNDARY_STATES = 25
GROUP_BINS = 8
TILES = 48
BASE_ROWS = ALPHABET * GROUP_BINS
FULL_CONTEXTS = ALPHABET**4 * BOUNDARY_STATES * TILES * GROUP_BINS
CDF_TOTAL_16 = 1 << 16

BLOCK = 30
TEST_BLOCKS = (1, 5, 9, 13, 17)
DEFAULT_SEED = 20260901
DEFAULT_SAMPLES_PER_FRAME = 8192
DEFAULT_LEAVES = 512

SCHEDULE_SCHEMA = "ddm_ccs1.receiver_causal_gm_schedule.v1"
MODEL_MAGIC = b"CCM1"
MODEL_HEADER = struct.Struct("<4sBBBBHHII32s")
CANDIDATE_MAGIC = b"CX1M"
CANDIDATE_HEADER = struct.Struct("<4sBBBBIIIII")
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
ZIP_EXTERNAL_ATTR = 0x81A40000


class Ccs1Error(RuntimeError):
    """CCS1 refused because a byte-close invariant failed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def immutable_bytes(path: Path, payload: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != payload:
            raise Ccs1Error(f"immutable payload changed: {path}")
        return
    atomic_bytes(path, payload)


def atomic_json(path: Path, payload: object) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def atomic_npz(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_field(path: Path) -> np.memmap:
    if path.stat().st_size != FIELD_BYTES:
        raise Ccs1Error(f"field bytes {path.stat().st_size} != {FIELD_BYTES}")
    if sha256_file(path) != FIELD_SHA256:
        raise Ccs1Error("unchanged AFR1 field fails its pinned SHA-256")
    field = np.memmap(path, mode="r", dtype=np.uint8, shape=(N, H, W))
    if int(field.max()) >= K:
        raise Ccs1Error("AFR1 field contains a class outside 0..4")
    return field


def split_frames() -> tuple[np.ndarray, np.ndarray]:
    is_test = np.zeros(N, dtype=bool)
    for block in TEST_BLOCKS:
        is_test[block * BLOCK : (block + 1) * BLOCK] = True
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


@dataclass(frozen=True)
class Schedule:
    groups: np.ndarray
    group_positions: tuple[np.ndarray, ...]
    order: np.ndarray
    tile: np.ndarray
    group_bin: np.ndarray
    blob: bytes


def build_schedule() -> Schedule:
    yy, xx = np.mgrid[0:H, 0:W]
    groups = ((xx % PATCH) + DELTA * (yy % PATCH)).astype(np.int16)
    if int(groups.max()) + 1 != GROUPS:
        raise Ccs1Error("G/M group construction did not produce 190 groups")
    flat_groups = groups.reshape(-1)
    positions = tuple(np.flatnonzero(flat_groups == group).astype(np.int32) for group in range(GROUPS))
    order = np.concatenate(positions)
    if order.size != PLANE or np.unique(order).size != PLANE:
        raise Ccs1Error("schedule is not a permutation of the plane")
    tile = ((yy // PATCH) * (W // PATCH) + (xx // PATCH)).astype(np.uint8).reshape(-1)
    group_bin = (flat_groups.astype(np.int32) * GROUP_BINS // GROUPS).astype(np.uint8)
    spec = {
        "schema": SCHEDULE_SCHEMA,
        "dimensions": [N, H, W],
        "alphabet": K,
        "reset": "previous=UNK(5), current=UNK(5), boundary recomputed per coded group",
        "parser_order": "frame 0..599; group g=(x%64)+2*(y%64) 0..189; row-major sites",
        "previous_decoded_classes": ["previous[y,x]", "previous[y-1,x] or UNK"],
        "decoded_prefix_classes": ["current[y,x-1]", "current[y-1,x]"],
        "boundary_state": "known_count + 5*disagree_with_previous_center across left/up/up-left/up-right; unavailable excluded",
        "position_cells": ["tile64 in 0..47", "floor(group*8/190) in 0..7"],
        "cdf": "five positive uint16 frequencies summing 65536, lifted exactly by *32768 to RC64 total 2^31",
        "leaf": "nonlinear exact joint-context override; missing key backs off to previous-center x group-bin row",
    }
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return Schedule(groups, positions, order, tile, group_bin, blob)


def _values(
    frame: np.ndarray,
    positions: np.ndarray,
    dy: int,
    dx: int,
    *,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    y = positions // W
    x = positions - y * W
    ny, nx = y + dy, x + dx
    valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
    if groups is not None:
        own = groups.reshape(-1)[positions]
        neighbour_group = np.full(positions.size, GROUPS, dtype=np.int16)
        good = np.flatnonzero(valid)
        neighbour_group[good] = groups[ny[good], nx[good]]
        valid &= neighbour_group < own
    result = np.full(positions.size, UNK, dtype=np.uint8)
    good = np.flatnonzero(valid)
    result[good] = frame[ny[good], nx[good]]
    return result


def context_keys(
    previous: np.ndarray,
    current: np.ndarray,
    positions: np.ndarray,
    schedule: Schedule,
) -> tuple[np.ndarray, np.ndarray]:
    prev0 = _values(previous, positions, 0, 0)
    prev_up = _values(previous, positions, -1, 0)
    left = _values(current, positions, 0, -1, groups=schedule.groups)
    up = _values(current, positions, -1, 0, groups=schedule.groups)
    prefix = [
        left,
        up,
        _values(current, positions, -1, -1, groups=schedule.groups),
        _values(current, positions, -1, 1, groups=schedule.groups),
    ]
    known = np.zeros(positions.size, dtype=np.uint8)
    disagree = np.zeros(positions.size, dtype=np.uint8)
    for values in prefix:
        is_known = values != UNK
        known += is_known
        disagree += is_known & (values != prev0)
    boundary = known + 5 * disagree
    if int(boundary.max(initial=0)) >= BOUNDARY_STATES:
        raise Ccs1Error("boundary state left its serialized domain")
    pos_tile = schedule.tile[positions]
    pos_group_bin = schedule.group_bin[positions]
    key = prev0.astype(np.uint32)
    for values, radix in (
        (prev_up, ALPHABET),
        (left, ALPHABET),
        (up, ALPHABET),
        (boundary, BOUNDARY_STATES),
        (pos_tile, TILES),
        (pos_group_bin, GROUP_BINS),
    ):
        key = key * radix + values.astype(np.uint32)
    base = prev0.astype(np.uint16) * GROUP_BINS + pos_group_bin.astype(np.uint16)
    return key, base


def quantize_rows(counts: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    values = counts.astype(np.float64) + alpha
    probabilities = values / values.sum(axis=1, keepdims=True)
    frequencies = np.floor(probabilities * CDF_TOTAL_16).astype(np.int64)
    np.maximum(frequencies, 1, out=frequencies)
    winners = probabilities.argmax(axis=1)
    balance = CDF_TOTAL_16 - frequencies.sum(axis=1)
    frequencies[np.arange(frequencies.shape[0]), winners] += balance
    if np.any(frequencies <= 0) or np.any(frequencies >= CDF_TOTAL_16):
        raise Ccs1Error("quantized CDF contains a non-positive or overflowing cell")
    if np.any(frequencies.sum(axis=1) != CDF_TOTAL_16):
        raise Ccs1Error("quantized CDF row does not sum to 65536")
    return frequencies.astype(np.uint16)


@dataclass(frozen=True)
class Model:
    base_freq: np.ndarray
    leaf_keys: np.ndarray
    leaf_freq: np.ndarray
    raw: bytes

    def frequencies(self, keys: np.ndarray, base: np.ndarray) -> np.ndarray:
        rows = self.base_freq[base].copy()
        indices = np.searchsorted(self.leaf_keys, keys)
        valid = indices < self.leaf_keys.size
        hits = np.zeros(keys.size, dtype=bool)
        hits[valid] = self.leaf_keys[indices[valid]] == keys[valid]
        rows[hits] = self.leaf_freq[indices[hits]]
        return rows


def serialize_model(base_freq: np.ndarray, leaf_keys: np.ndarray, leaf_freq: np.ndarray) -> bytes:
    body = base_freq.astype("<u2").tobytes()
    body += leaf_keys.astype("<u4").tobytes()
    body += leaf_freq.astype("<u2").tobytes()
    header = MODEL_HEADER.pack(
        MODEL_MAGIC,
        1,
        K,
        BOUNDARY_STATES,
        0,
        base_freq.shape[0],
        leaf_keys.size,
        len(body),
        zlib.crc32(body),
        bytes.fromhex(FIELD_SHA256),
    )
    return header + body


def parse_model(raw: bytes) -> Model:
    if len(raw) < MODEL_HEADER.size:
        raise Ccs1Error("model is truncated")
    magic, version, alphabet, boundary, flags, n_base, n_leaf, body_bytes, crc, field_sha = MODEL_HEADER.unpack_from(
        raw
    )
    body = raw[MODEL_HEADER.size :]
    expected = n_base * K * 2 + n_leaf * 4 + n_leaf * K * 2
    if (
        magic != MODEL_MAGIC
        or version != 1
        or alphabet != K
        or boundary != BOUNDARY_STATES
        or flags != 0
        or n_base != BASE_ROWS
        or body_bytes != len(body)
        or len(body) != expected
        or zlib.crc32(body) != crc
        or field_sha.hex() != FIELD_SHA256
    ):
        raise Ccs1Error("model header/body contract differs")
    offset = 0
    base = np.frombuffer(body, dtype="<u2", count=n_base * K, offset=offset).reshape(n_base, K).copy()
    offset += n_base * K * 2
    keys = np.frombuffer(body, dtype="<u4", count=n_leaf, offset=offset).copy()
    offset += n_leaf * 4
    leaves = np.frombuffer(body, dtype="<u2", count=n_leaf * K, offset=offset).reshape(n_leaf, K).copy()
    if not np.all(keys[1:] > keys[:-1]):
        raise Ccs1Error("leaf keys are not strictly sorted")
    if np.any(base.astype(np.uint32).sum(axis=1) != CDF_TOTAL_16) or np.any(
        leaves.astype(np.uint32).sum(axis=1) != CDF_TOTAL_16
    ):
        raise Ccs1Error("parsed model CDF rows have the wrong sum")
    return Model(base, keys, leaves, raw)


def fit_model(
    field: np.ndarray,
    schedule: Schedule,
    root: Path,
    *,
    seed: int,
    samples_per_frame: int,
    leaves: int,
) -> tuple[Model, dict[str, Any]]:
    checkpoints = root / "checkpoints" / "fit"
    checkpoints.mkdir(parents=True, exist_ok=True)
    train_frames, test_frames = split_frames()
    sample_path = checkpoints / "stage_00_sample_positions.npz"
    if sample_path.is_file():
        with np.load(sample_path, allow_pickle=False) as blob:
            sampled = blob["positions"]
            stored_seed = int(blob["seed"][0])
        if stored_seed != seed or sampled.shape != (train_frames.size, samples_per_frame):
            raise Ccs1Error("fit sample checkpoint differs from requested config")
    else:
        rng = np.random.default_rng(seed)
        sampled = np.empty((train_frames.size, samples_per_frame), dtype=np.uint32)
        for row in range(train_frames.size):
            sampled[row] = rng.choice(PLANE, size=samples_per_frame, replace=False)
        atomic_npz(
            sample_path,
            positions=sampled,
            seed=np.array([seed], dtype=np.int64),
            train_frames=train_frames,
            test_frames=test_frames,
        )

    counts_path = checkpoints / "stage_01_joint_counts.npz"
    if counts_path.is_file():
        with np.load(counts_path, allow_pickle=False) as blob:
            base_counts = blob["base_counts"]
            full_counts = blob["full_counts"]
    else:
        base_counts = np.zeros((BASE_ROWS, K), dtype=np.uint32)
        full_counts = np.zeros((FULL_CONTEXTS, K), dtype=np.uint32)
        blank = np.full((H, W), UNK, dtype=np.uint8)
        for row, frame_index in enumerate(train_frames):
            frame = np.asarray(field[frame_index])
            previous = np.asarray(field[frame_index - 1]) if frame_index else blank
            positions = sampled[row].astype(np.int32)
            keys, base = context_keys(previous, frame, positions, schedule)
            symbols = frame.reshape(-1)[positions]
            np.add.at(base_counts, (base, symbols), 1)
            np.add.at(full_counts, (keys, symbols), 1)
        atomic_npz(counts_path, base_counts=base_counts, full_counts=full_counts)

    base_freq = quantize_rows(base_counts)
    totals = full_counts.sum(axis=1)
    active = np.flatnonzero(totals >= 32)
    if active.size < leaves:
        raise Ccs1Error(f"only {active.size} eligible nonlinear contexts for {leaves} leaves")
    active_counts = full_counts[active]
    leaf_prob = (active_counts + 0.5) / (active_counts.sum(axis=1, keepdims=True) + 0.5 * K)
    key = active.astype(np.uint32)
    temp = key // GROUP_BINS
    group_bin = key % GROUP_BINS
    temp //= TILES
    temp //= BOUNDARY_STATES
    temp //= ALPHABET
    temp //= ALPHABET
    temp //= ALPHABET
    prev0 = temp
    base_id = prev0 * GROUP_BINS + group_bin
    base_prob = base_freq[base_id].astype(np.float64) / CDF_TOTAL_16
    gain = np.sum(
        active_counts * (np.log2(np.maximum(leaf_prob, 1e-300)) - np.log2(base_prob)),
        axis=1,
    )
    selected = np.argpartition(gain, -leaves)[-leaves:]
    leaf_keys = active[selected].astype(np.uint32)
    leaf_freq = quantize_rows(active_counts[selected])
    order = np.argsort(leaf_keys)
    leaf_keys, leaf_freq = leaf_keys[order], leaf_freq[order]
    raw = serialize_model(base_freq, leaf_keys, leaf_freq)
    model = parse_model(raw)
    model_path = checkpoints / "stage_02_fitted_model.raw"
    immutable_bytes(model_path, raw)
    receipt = {
        "seed": seed,
        "split": {
            "protocol": "20 contiguous blocks of 30; test blocks 1,5,9,13,17",
            "train_frames": int(train_frames.size),
            "heldout_frames": int(test_frames.size),
            "heldout_positions": int(test_frames.size * PLANE),
        },
        "training_sample_positions": int(sampled.size),
        "samples_per_train_frame": samples_per_frame,
        "eligible_joint_contexts_min_count_32": int(active.size),
        "selected_nonlinear_leaves": leaves,
        "selected_train_gain_bits": float(gain[selected].sum()),
        "checkpoints": [file_fact(sample_path), file_fact(counts_path), file_fact(model_path)],
    }
    return model, receipt


def load_route_b() -> Any:
    spec = importlib.util.spec_from_file_location("ddm_ccs1_route_b", ROUTE_B)
    if spec is None or spec.loader is None:
        raise Ccs1Error("cannot import the RC64 binding")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_rc64(root: Path, route_b: Any) -> tuple[Path, dict[str, Any]]:
    build = root / "retained" / "rc64_build"
    build.mkdir(parents=True, exist_ok=True)
    generated = RC64_BASE.read_bytes()
    extension = ("\n" + route_b.RC64_CHECKPOINT_EXTENSION).encode()
    if not generated.endswith(extension):
        raise Ccs1Error("generated RC64 custody source lacks its pinned checkpoint extension")
    base = generated[: -len(extension)]
    if sha256_bytes(base) != RC64_BASE_SHA256:
        raise Ccs1Error("recovered RC64 encoder base fails its pinned SHA-256")
    source = build / "ccs1_rc64.c"
    library = build / "libccs1_rc64.dylib"
    payload = base + extension
    immutable_bytes(source, payload)
    command = [
        "/usr/bin/cc",
        "-O3",
        "-std=c11",
        "-shared",
        "-fPIC",
        "-ffp-contract=off",
        "-fno-fast-math",
        str(source),
        "-o",
        str(library),
    ]
    if not library.is_file():
        subprocess.run(command, check=True)
    return library, {"argv": command, "source": file_fact(source), "library": file_fact(library)}


def probabilities(model: Model, keys: np.ndarray, base: np.ndarray) -> np.ndarray:
    # uint16 / 2^16 is exactly representable in float32.  Route-B's conversion
    # therefore recovers the serialized row * 2^15 exactly at total 2^31.
    return model.frequencies(keys, base).astype(np.float32) / np.float32(CDF_TOTAL_16)


def ideal_bits(freq: np.ndarray, symbols: np.ndarray) -> float:
    chosen = freq[np.arange(symbols.size), symbols].astype(np.float64)
    return float(np.sum(16.0 - np.log2(chosen)))


def encode_frame(
    encoder: Any,
    model: Model,
    schedule: Schedule,
    previous: np.ndarray,
    target: np.ndarray,
) -> float:
    keys, base = context_keys(previous, target, schedule.order, schedule)
    freq = model.frequencies(keys, base)
    symbols = target.reshape(-1)[schedule.order].astype(np.int32)
    encoder.encode(symbols, freq.astype(np.float32) / np.float32(CDF_TOTAL_16))
    return ideal_bits(freq, symbols)


def decode_frame(
    decoder: Any,
    model: Model,
    schedule: Schedule,
    previous: np.ndarray,
) -> np.ndarray:
    current = np.full((H, W), UNK, dtype=np.uint8)
    flat = current.reshape(-1)
    for positions in schedule.group_positions:
        keys, base = context_keys(previous, current, positions, schedule)
        decoded = decoder.decode(None, probabilities(model, keys, base))
        if np.any((decoded < 0) | (decoded >= K)):
            raise Ccs1Error("RC64 decoded a class outside 0..4")
        flat[positions] = decoded.astype(np.uint8)
    return current


def heldout_coding(
    field: np.ndarray,
    model: Model,
    schedule: Schedule,
    route_b: Any,
    library: Path,
    root: Path,
) -> dict[str, Any]:
    retained = root / "retained" / "heldout"
    retained.mkdir(parents=True, exist_ok=True)
    decoded_path = retained / "heldout_decoded.u8"
    decoded = np.memmap(decoded_path, mode="w+", dtype=np.uint8, shape=(len(TEST_BLOCKS) * BLOCK, H, W))
    rows = []
    write_index = 0
    blank = np.full((H, W), UNK, dtype=np.uint8)
    for block in TEST_BLOCKS:
        encoder = route_b.NativeRc64Encoder(library)
        previous = blank
        bits = 0.0
        start = block * BLOCK
        for frame_index in range(start, start + BLOCK):
            target = np.asarray(field[frame_index])
            bits += encode_frame(encoder, model, schedule, previous, target)
            previous = target
        payload = encoder.finish()
        payload_path = retained / f"block_{block:02d}.rc64"
        immutable_bytes(payload_path, payload)
        decoder = route_b.NativeRc64Decoder(library, payload)
        previous = blank
        for frame_index in range(start, start + BLOCK):
            current = decode_frame(decoder, model, schedule, previous)
            if not np.array_equal(current, field[frame_index]):
                raise Ccs1Error(f"heldout block {block} decode differs at frame {frame_index}")
            decoded[write_index] = current
            write_index += 1
            previous = current
        rows.append(
            {
                "block": block,
                "frames": [start, start + BLOCK - 1],
                "symbols": BLOCK * PLANE,
                "ideal_bits": bits,
                "payload": file_fact(payload_path),
            }
        )
    decoded.flush()
    del decoded
    return {
        "selection": "LM1 heldout contiguous blocks; each block coder/reset is independent",
        "blocks": rows,
        "frames": len(TEST_BLOCKS) * BLOCK,
        "symbols": len(TEST_BLOCKS) * BLOCK * PLANE,
        "physical_payload_bytes": int(sum(row["payload"]["bytes"] for row in rows)),
        "decoded": file_fact(decoded_path),
        "byte_identical": True,
    }


def encode_full(
    field: np.ndarray,
    model: Model,
    schedule: Schedule,
    route_b: Any,
    library: Path,
    root: Path,
    checkpoint_every: int,
) -> tuple[Path, dict[str, Any]]:
    retained = root / "retained"
    checkpoints = root / "checkpoints" / "encode"
    checkpoints.mkdir(parents=True, exist_ok=True)
    candidates = sorted(checkpoints.glob("frame_*.npz"))
    start = 0
    per_frame = np.zeros(N, dtype=np.float64)
    blank = np.full((H, W), UNK, dtype=np.uint8)
    previous = blank
    encoder: Any
    resumed = None
    if candidates:
        state_path = candidates[-1]
        frame = int(state_path.stem.split("_")[1])
        encoder_path = state_path.with_suffix(".encoder")
        with np.load(state_path, allow_pickle=False) as state:
            if int(state["frame"][0]) != frame:
                raise Ccs1Error("encode checkpoint frame disagrees with its filename")
            previous = state["previous"].copy()
            per_frame = state["per_frame_bits"].copy()
        encoder = route_b.NativeRc64Encoder(library, encoder_path.read_bytes())
        start = frame
        resumed = {"state": file_fact(state_path), "encoder": file_fact(encoder_path)}
    else:
        encoder = route_b.NativeRc64Encoder(library)
    checkpoint_facts = []
    for frame_index in range(start, N):
        target = np.asarray(field[frame_index])
        per_frame[frame_index] = encode_frame(encoder, model, schedule, previous, target)
        previous = target
        boundary = frame_index + 1
        if boundary % checkpoint_every == 0 and boundary < N:
            state_path = checkpoints / f"frame_{boundary:04d}.npz"
            encoder_path = state_path.with_suffix(".encoder")
            snapshot = encoder.snapshot()
            if state_path.is_file() or encoder_path.is_file():
                if not (state_path.is_file() and encoder_path.is_file()):
                    raise Ccs1Error("partial immutable encode checkpoint exists")
            else:
                atomic_npz(
                    state_path,
                    frame=np.array([boundary], dtype=np.int64),
                    previous=previous,
                    per_frame_bits=per_frame,
                )
                immutable_bytes(encoder_path, snapshot)
            checkpoint_facts.append(
                {"frame": boundary, "state": file_fact(state_path), "encoder": file_fact(encoder_path)}
            )
    payload = encoder.finish()
    stream_path = retained / "ccs1_n600.rc64"
    immutable_bytes(stream_path, payload)
    ledger_path = retained / "ccs1_per_frame_ideal_bits.f64le"
    immutable_bytes(ledger_path, per_frame.astype("<f8").tobytes())
    terminal_path = checkpoints / "frame_0600_terminal.json"
    terminal = {
        "frame": 600,
        "stream": file_fact(stream_path),
        "per_frame_ideal_bits": file_fact(ledger_path),
        "ideal_bits_total": float(per_frame.sum()),
    }
    atomic_json(terminal_path, terminal)
    return stream_path, {
        "resumed_from": resumed,
        "checkpoint_every_frames": checkpoint_every,
        "checkpoints_written_or_verified": checkpoint_facts,
        "terminal": file_fact(terminal_path),
        **terminal,
    }


def compress_model(raw: bytes, retained: Path) -> tuple[int, Path, list[dict[str, Any]]]:
    variants = {
        1: lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME),
        2: brotli.compress(raw, quality=11),
        3: zlib.compress(raw, 9),
    }
    names = {1: "xz", 2: "brotli_q11", 3: "zlib9"}
    facts = []
    paths: dict[int, Path] = {}
    for codec, payload in variants.items():
        path = retained / f"model.{names[codec]}.bin"
        immutable_bytes(path, payload)
        paths[codec] = path
        facts.append({"codec": codec, "name": names[codec], **file_fact(path)})
    selected = min(variants, key=lambda codec: (len(variants[codec]), codec))
    return selected, paths[selected], facts


def base_sections() -> dict[str, bytes | int]:
    if sha256_file(BASE_ARCHIVE) != BASE_ARCHIVE_SHA256:
        raise Ccs1Error("AFR1 archive fails its pinned SHA-256")
    with zipfile.ZipFile(BASE_ARCHIVE) as archive:
        member = archive.read("p")
    rx1 = struct.Struct("<4sBBBBHHH")
    magic, version, codec, table_mode, reserved, hpac_bytes, semantic_bytes, carrier_bytes = rx1.unpack_from(member)
    if magic != b"RX1M" or version != 1 or codec != 2:
        raise Ccs1Error("AFR1 outer representation differs")
    offset = rx1.size + hpac_bytes
    semantic = member[offset : offset + semantic_bytes]
    offset += semantic_bytes
    carrier = member[offset : offset + carrier_bytes]
    offset += carrier_bytes
    tail = member[offset:]
    if len(tail) <= 96:
        raise Ccs1Error("AFR1 residual/token tail is truncated")
    return {
        "reserved": reserved,
        "semantic": semantic,
        "carrier": carrier,
        "residual": tail[:96],
        "shipped_stream": tail[96:],
    }


def pack_archive(member: bytes, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        info = zipfile.ZipInfo("p", date_time=ZIP_DATE)
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = ZIP_EXTERNAL_ATTR
        info.create_system = 3
        archive.writestr(info, member)
    os.replace(temporary, path)


def build_candidate(
    schedule: Schedule,
    model: Model,
    stream_path: Path,
    root: Path,
) -> tuple[Path, dict[str, Any]]:
    retained = root / "retained"
    raw_model_path = retained / "model.raw"
    immutable_bytes(raw_model_path, model.raw)
    schedule_path = retained / "schedule_v1.json"
    immutable_bytes(schedule_path, schedule.blob)
    codec, compressed_path, variants = compress_model(model.raw, retained)
    sections = base_sections()
    compressed = compressed_path.read_bytes()
    stream = stream_path.read_bytes()
    header = CANDIDATE_HEADER.pack(
        CANDIDATE_MAGIC,
        1,
        codec,
        1,
        int(sections["reserved"]),
        len(schedule.blob),
        len(compressed),
        len(sections["semantic"]),
        len(sections["carrier"]),
        len(sections["residual"]),
    )
    member = (
        header + schedule.blob + compressed + sections["semantic"] + sections["carrier"] + sections["residual"] + stream
    )
    archive_path = retained / "archive.zip"
    repeat_path = retained / "archive.repeat.zip"
    pack_archive(member, archive_path)
    pack_archive(member, repeat_path)
    if archive_path.read_bytes() != repeat_path.read_bytes():
        raise Ccs1Error("deterministic archive repeat differs")
    report = {
        "schedule": file_fact(schedule_path),
        "model_raw": file_fact(raw_model_path),
        "model_variants": variants,
        "selected_model_codec": codec,
        "selected_model": file_fact(compressed_path),
        "semantic_compressed_bytes": len(sections["semantic"]),
        "carrier_compressed_bytes": len(sections["carrier"]),
        "residual_compact_bytes": len(sections["residual"]),
        "stream": file_fact(stream_path),
        "candidate_header_bytes": len(header),
        "zip_container_bytes": archive_path.stat().st_size - len(member),
        "archive": file_fact(archive_path),
        "repeat": file_fact(repeat_path),
        "repeat_byte_identical": True,
    }
    return archive_path, report


def parse_candidate(path: Path, expected_schedule: bytes) -> tuple[Model, bytes, dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["p"]:
            raise Ccs1Error("candidate archive must contain exactly p")
        member = archive.read("p")
    if len(member) < CANDIDATE_HEADER.size:
        raise Ccs1Error("candidate member is truncated")
    fields = CANDIDATE_HEADER.unpack_from(member)
    (
        magic,
        version,
        codec,
        schedule_version,
        reserved,
        schedule_bytes,
        model_bytes,
        semantic_bytes,
        carrier_bytes,
        residual_bytes,
    ) = fields
    if magic != CANDIDATE_MAGIC or version != 1 or schedule_version != 1 or codec not in (1, 2, 3):
        raise Ccs1Error("candidate header differs")
    offset = CANDIDATE_HEADER.size
    schedule_blob = member[offset : offset + schedule_bytes]
    offset += schedule_bytes
    compressed = member[offset : offset + model_bytes]
    offset += model_bytes
    semantic = member[offset : offset + semantic_bytes]
    offset += semantic_bytes
    carrier = member[offset : offset + carrier_bytes]
    offset += carrier_bytes
    residual = member[offset : offset + residual_bytes]
    offset += residual_bytes
    stream = member[offset:]
    raw = (
        lzma.decompress(compressed, format=lzma.FORMAT_XZ)
        if codec == 1
        else brotli.decompress(compressed)
        if codec == 2
        else zlib.decompress(compressed)
    )
    base = base_sections()
    if (
        schedule_blob != expected_schedule
        or reserved != base["reserved"]
        or semantic != base["semantic"]
        or carrier != base["carrier"]
        or residual != base["residual"]
        or not stream
    ):
        raise Ccs1Error("candidate parse-back changed a schedule or fixed AFR1 section")
    model = parse_model(raw)
    return (
        model,
        stream,
        {
            "member_bytes": len(member),
            "schedule_sha256": sha256_bytes(schedule_blob),
            "model_raw_sha256": sha256_bytes(raw),
            "semantic_sha256": sha256_bytes(semantic),
            "carrier_sha256": sha256_bytes(carrier),
            "residual_sha256": sha256_bytes(residual),
            "stream_sha256": sha256_bytes(stream),
            "strict_parseback": True,
        },
    )


def decode_full(
    archive_path: Path,
    schedule: Schedule,
    route_b: Any,
    library: Path,
    destination: Path,
) -> dict[str, Any]:
    model, stream, parse = parse_candidate(archive_path, schedule.blob)
    if destination.is_file() and sha256_file(destination) == FIELD_SHA256:
        return {
            "decoded": file_fact(destination),
            "field_byte_identical": True,
            "seconds": 0.0,
            "parseback": parse,
            "status": "reused_verified_complete_decode",
        }
    decoder = route_b.NativeRc64Decoder(library, stream)
    output = np.memmap(destination, mode="w+", dtype=np.uint8, shape=(N, H, W))
    previous = np.full((H, W), UNK, dtype=np.uint8)
    started = time.perf_counter()
    for frame_index in range(N):
        current = decode_frame(decoder, model, schedule, previous)
        output[frame_index] = current
        previous = current
    output.flush()
    del output
    if not decoder.is_empty():
        raise Ccs1Error("RC64 decoder did not consume exactly n600 symbols")
    sha = sha256_file(destination)
    if sha != FIELD_SHA256:
        raise Ccs1Error("decoded n600 field differs from AFR1")
    return {
        "decoded": file_fact(destination),
        "field_byte_identical": True,
        "seconds": time.perf_counter() - started,
        "parseback": parse,
    }


def _load_runtime() -> tuple[Any, Any, Any]:
    if str(BASE_RUNTIME) not in sys.path:
        sys.path.insert(0, str(BASE_RUNTIME))
    from runtime import f26_inflate
    from runtime.residual_archive import read_residual_archive

    renderer = f26_inflate._load_renderer(BASE_RUNTIME / "cpr1")
    parts = read_residual_archive(BASE_ARCHIVE)
    return f26_inflate, renderer, parts


def render_field(token_path: Path, destination: Path) -> dict[str, Any]:
    import torch

    if destination.is_file() and sha256_file(destination) == BASE_RAW_SHA256:
        return {
            "raw": file_fact(destination),
            "seconds": 0.0,
            "selector": {"status": "reused_verified_complete_render"},
            "renderer_source": str((BASE_RUNTIME / "cpr1" / "inflate.py").resolve()),
        }
    f26, renderer, parts = _load_runtime()
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    device = torch.device("cpu")
    carrier_blob, selector_blob = f26.split_frame0_selector_carrier(parts.carrier_blob)
    canonical_carrier = f26.materialize_cpr1(carrier_blob, renderer)
    semantic_width_marker = bytes(40_252)
    semantic_pose = (
        struct.pack("<II", len(semantic_width_marker), len(canonical_carrier))
        + semantic_width_marker
        + canonical_carrier
    )
    _, basis, coefficients = renderer.unpack_semantic_pose(semantic_pose)
    semantic = renderer.SemanticTokenRenderer(96)
    tagged = renderer.unpack_variant_semantic_or_none(parts.semantic_blob, semantic.state_dict())
    if tagged is None:
        from runtime.entropy.renderer_weight_codec import decode_wans1

        tagged = {
            record.schema.name: torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32))
            for record in decode_wans1(parts.semantic_blob)
        }
    semantic.load_state_dict(tagged, strict=True)
    mm = np.memmap(token_path, mode="r", dtype=np.uint8, shape=(N, H, W))
    tokens = torch.from_numpy(np.asarray(mm).copy())
    partial = destination.with_name(f".{destination.name}.render.partial")
    started = time.perf_counter()
    renderer.render_video(semantic, basis, coefficients, tokens, partial, device)
    selector = None
    if selector_blob is not None:
        modes, indices = f26.decode_selector(selector_blob)
        output = np.memmap(
            partial,
            mode="r+",
            dtype=np.uint8,
            shape=(N * 2, renderer.CAMERA_H, renderer.CAMERA_W, 3),
        )
        for mode_index, mode in enumerate(modes):
            frame_ids = np.flatnonzero(indices == mode_index)
            for start in range(0, frame_ids.size, 16):
                chosen = frame_ids[start : start + 16]
                output[2 * chosen] = f26.apply_pixel_mode(np.asarray(output[2 * chosen]).copy(), mode)
        output.flush()
        del output
        selector = {
            "payload_bytes": len(selector_blob),
            "payload_sha256": sha256_bytes(selector_blob),
            "mode_count": len(modes),
            "frame_count": int(indices.size),
            "io_chunk_frames": 16,
        }
    os.replace(partial, destination)
    report = {
        "raw": file_fact(destination),
        "seconds": time.perf_counter() - started,
        "selector": selector,
        "renderer_source": str((BASE_RUNTIME / "cpr1" / "inflate.py").resolve()),
    }
    if report["raw"]["sha256"] != BASE_RAW_SHA256:
        raise Ccs1Error("rendered candidate field differs from the pinned AFR1 output")
    return report


def manifest_tree(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted((root / "retained").rglob("*")):
        if path.is_file():
            records.append(file_fact(path))
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {"files": records, "tree_sha256": sha256_bytes(payload), "file_count": len(records)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.store.resolve()
    retained = root / "retained"
    retained.mkdir(parents=True, exist_ok=True)
    start_free = shutil.disk_usage(root).free
    if start_free < (1 << 30):
        raise Ccs1Error("APDataStore has less than the mandatory 1 GiB reserve")
    command_path = retained / "command.json"
    atomic_json(
        command_path,
        {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "seed": args.seed,
            "samples_per_frame": args.samples_per_frame,
            "leaves": args.leaves,
            "checkpoint_every": args.checkpoint_every,
        },
    )
    started = time.time()
    field = load_field(args.field)
    schedule = build_schedule()
    model, fit = fit_model(
        field,
        schedule,
        root,
        seed=args.seed,
        samples_per_frame=args.samples_per_frame,
        leaves=args.leaves,
    )
    route_b = load_route_b()
    library, build = compile_rc64(root, route_b)
    heldout = heldout_coding(field, model, schedule, route_b, library, root)
    stream_path, encoding = encode_full(field, model, schedule, route_b, library, root, args.checkpoint_every)
    archive_path, archive = build_candidate(schedule, model, stream_path, root)
    decoded1 = retained / "decoded_tokens_pass1.u8"
    decoded2 = retained / "decoded_tokens_pass2.u8"
    decode_receipts = [
        decode_full(archive_path, schedule, route_b, library, decoded1),
        decode_full(archive_path, schedule, route_b, library, decoded2),
    ]
    if decoded1.read_bytes() != decoded2.read_bytes():
        raise Ccs1Error("the two complete candidate decodes differ")
    raw1 = retained / "rendered_pass1.raw"
    raw2 = retained / "rendered_pass2.raw"
    render_receipts = [render_field(decoded1, raw1), render_field(decoded2, raw2)]
    if sha256_file(raw1) != sha256_file(raw2):
        raise Ccs1Error("the two complete candidate renders differ")
    complete = archive_path.stat().st_size
    fixed = (
        archive["semantic_compressed_bytes"]
        + archive["carrier_compressed_bytes"]
        + archive["residual_compact_bytes"]
        + archive["candidate_header_bytes"]
        + archive["zip_container_bytes"]
    )
    result = {
        "schema": "ddm_ccs1.causal_schedule_builder.result.v1",
        "axis": "[macOS-CPU advisory / scorer-free exact lossless coding measurement]",
        "score_claim": False,
        "authority_replay_run": False,
        "start_free_bytes": start_free,
        "preserved_ap_reserve_bytes": 1 << 30,
        "field": file_fact(args.field),
        "base_archive": file_fact(BASE_ARCHIVE),
        "schedule_law": {
            "equation_id": "decoder_causal_condition_transport_v1",
            "law": "H(E_i(C_i) | D_<i,p_i)=0 at every coded site",
            "satisfied_by_construction": True,
            "schema": json.loads(schedule.blob),
        },
        "fit": fit,
        "rc64_build": build,
        "heldout_coding": heldout,
        "full_encoding": encoding,
        "archive": archive,
        "counted_pool": {
            "model_selected_bytes": archive["selected_model"]["bytes"],
            "schedule_bytes": archive["schedule"]["bytes"],
            "stream_bytes": archive["stream"]["bytes"],
            "unchanged_semantic_bytes": archive["semantic_compressed_bytes"],
            "unchanged_carrier_bytes": archive["carrier_compressed_bytes"],
            "residual_and_container_headers_bytes": fixed
            - archive["semantic_compressed_bytes"]
            - archive["carrier_compressed_bytes"],
            "fixed_archive_bytes_excluding_model_schedule_stream": fixed,
            "complete_archive_bytes": complete,
            "model_plus_stream_gate_bytes": 84_910,
            "complete_archive_gate_bytes": 137_986,
            "shipped_token_pool_bytes": 126_926,
            "allowance_bytes": 87_403.86,
        },
        "double_decode": {
            "passes": decode_receipts,
            "pass1_pass2_byte_identical": True,
            "each_equals_afr1_field": True,
        },
        "double_render": {
            "passes": render_receipts,
            "pass1_pass2_byte_identical": True,
            "each_equals_afr1_render": True,
        },
        "verdict": "FIRE-ORDER-TO-MAIN" if complete <= 137_986 else "CLOSED-AT-GATE",
        "verdict_scope": "INSTANCE: CCS1 v1, seed 20260901, 512-leaf nonlinear receiver-causal schedule on unchanged AFR1 field",
        "elapsed_seconds": time.time() - started,
        "command": file_fact(command_path),
    }
    result_path = root / "RESULT.json"
    atomic_json(result_path, result)
    manifest = manifest_tree(root)
    manifest_path = root / "MANIFEST.json"
    atomic_json(manifest_path, manifest)
    done = {
        "status": "complete",
        "result": file_fact(result_path),
        "manifest": file_fact(manifest_path),
        "verdict": result["verdict"],
        "complete_archive_bytes": complete,
    }
    atomic_json(root / "DONE.json", done)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    value.add_argument("--store", type=Path, required=True)
    value.add_argument("--field", type=Path, default=FIELD)
    value.add_argument("--seed", type=int, default=DEFAULT_SEED)
    value.add_argument("--samples-per-frame", type=int, default=DEFAULT_SAMPLES_PER_FRAME)
    value.add_argument("--leaves", type=int, default=DEFAULT_LEAVES)
    value.add_argument("--checkpoint-every", type=int, default=25)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = run(args)
    print(json.dumps({"verdict": result["verdict"], "counted_pool": result["counted_pool"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
