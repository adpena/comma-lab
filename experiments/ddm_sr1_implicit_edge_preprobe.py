#!/usr/bin/env python3
"""Scorer-free n600 pre-probes for implicit edge conditioning on CP135.

The live F26 probability lattice already includes its shipped
previous-boundary x predicted-class calibration.  This probe starts from that
exact lattice and asks whether two *additional*, decoder-known contexts pay:

* current-frame causal edge topology from groups already decoded; and
* sign/delta-sign of an already-carried pose coefficient.

The analysis is frame-holdout first.  Any candidate sent to RC64 is then fit
on all 600 frames, serialized as a charged ``IEC1`` table, encoded, decoded,
and byte-compared with the retained source.  Every generated payload is kept
under ``--output``.  Analysis, encode, and decode checkpoint every 24 frames.
No scorer, renderer, network, or paid service is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import itertools
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

FRAME_COUNT = 600
HEIGHT = 384
WIDTH = 512
CLASSES = 5
EVENTS_PER_FRAME = HEIGHT * WIDTH
TOTAL_EVENTS = FRAME_COUNT * EVENTS_PER_FRAME
BASE_TOKEN_BYTES = 114_706
BASE_TOKEN_SHA256 = "e77c075f4f7ee93a2d0c40343df263429b58e4c4d4f14c794529b53b466b9c73"
EVENT_SHA256 = "8eb51ab7a2884c9d7b6e73ee60f78ded38c691d6b82e639b75dddec6e0ac1366"
SPATIAL_SHA256 = "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"
ARCHIVE_SHA256 = "12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004"
LIBRARY_SHA256 = "4f4b72a8afb7c419f7bdcf91352f575906a5423031b384102d866308a9137a7e"
AXIS = "[macOS-CPU advisory, scorer-free n600 token entropy]"
DEFAULT_OUTPUT = Path("/Volumes/APDataStore/pact/ddm_sr1_implicit_edge_20260811")
DEFAULT_RUNTIME = Path("/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime")
DEFAULT_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/archive.zip")
DEFAULT_DT1 = Path("/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/chunk_manifest.json")
DEFAULT_CODES = Path("/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/retained/probabilities/control")
DEFAULT_LIBRARY = Path("/Volumes/VertigoDataTier/pact/ddm_rc64p_20260810/build/rc64_a/liblc2_rc64.dylib")
CHECKPOINT_FRAMES = 24
IEC_HEADER = struct.Struct("<4sBBBBH")
COMPOSITE_HEADER = struct.Struct("<4sI")
Feature = Literal["causal_edge_predicted", "pose_sign_predicted", "pose_delta_sign_predicted"]


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **values)
    atomic_bytes(path, buffer.getvalue())


def cleanup_stale_atomic_scratch(output: Path) -> dict[str, Any]:
    removed = []
    protected_younger_than_s = 24 * 60 * 60
    now = time.time()
    if output.exists():
        for path in output.rglob(".*.tmp"):
            if not path.is_file() or now - path.stat().st_mtime < protected_younger_than_s:
                continue
            removed.append(
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "reason": "incomplete atomic-write scratch; complete stage checkpoints are preserved",
                }
            )
            path.unlink()
    result = {
        "schema": "ddm_sr1_atomic_scratch_cleanup.v1",
        "removed": removed,
        "removed_count": len(removed),
        "recoverable": False,
        "protected_younger_than_s": protected_younger_than_s,
    }
    atomic_json(output / "ATOMIC_SCRATCH_CLEANUP.json", result)
    return result


def require_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise RuntimeError(f"custody mismatch: {path}")


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class SourceChunk:
    start: int
    end: int
    symbols: Path
    sha256: str


class SourceSymbols:
    def __init__(self, manifest: Path) -> None:
        payload = json.loads(manifest.read_text())
        rows = tuple(
            SourceChunk(
                int(row["start_frame"]),
                int(row["end_frame"]),
                Path(row["symbols_path"]),
                str(row["symbols_sha256"]),
            )
            for row in payload["chunks"]
        )
        if not rows or rows[0].start != 0 or rows[-1].end != FRAME_COUNT:
            raise RuntimeError("DT1 source does not cover n600")
        if any(left.end != right.start for left, right in itertools.pairwise(rows)):
            raise RuntimeError("DT1 source chunks are not contiguous")
        for row in rows:
            require_file(row.symbols, row.sha256)
        self.rows = rows
        self.loaded_path: Path | None = None
        self.loaded: np.ndarray | None = None

    def frame(self, frame: int) -> np.ndarray:
        row = next(item for item in self.rows if item.start <= frame < item.end)
        if self.loaded_path != row.symbols:
            self.loaded = np.load(row.symbols, mmap_mode="r", allow_pickle=False)
            self.loaded_path = row.symbols
        assert self.loaded is not None
        start = (frame - row.start) * EVENTS_PER_FRAME
        value = np.asarray(self.loaded[start : start + EVENTS_PER_FRAME], dtype=np.uint8)
        if value.shape != (EVENTS_PER_FRAME,) or np.any(value >= CLASSES):
            raise RuntimeError(f"invalid source symbols at frame {frame}")
        return value


@dataclass(frozen=True)
class Table:
    feature: Feature
    dimension: int
    bits: int
    codes: np.ndarray
    scale: float
    values: np.ndarray


def pack_signed(values: np.ndarray, bits: int) -> bytes:
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    mask = (1 << bits) - 1
    unsigned = flat & mask
    accumulator = 0
    count = 0
    output = bytearray()
    for value in unsigned.tolist():
        accumulator |= int(value) << count
        count += bits
        while count >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            count -= 8
    if count:
        output.append(accumulator & 0xFF)
    return bytes(output)


def unpack_signed(payload: bytes, count: int, bits: int) -> np.ndarray:
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    values = np.empty(count, dtype=np.int8)
    accumulator = 0
    available = 0
    offset = 0
    for index in range(count):
        while available < bits:
            if offset >= len(payload):
                raise RuntimeError("truncated IEC1 table")
            accumulator |= payload[offset] << available
            available += 8
            offset += 1
        value = accumulator & mask
        accumulator >>= bits
        available -= bits
        values[index] = value - (1 << bits) if value & sign else value
    if offset != len(payload) or accumulator:
        raise RuntimeError("IEC1 table has trailing nonzero data")
    return values


def quantize_table(feature: Feature, dimension: int, values: np.ndarray, bits: int) -> Table:
    if bits not in (4, 6, 8):
        raise ValueError(bits)
    limit = (1 << (bits - 1)) - 1
    maximum = float(np.max(np.abs(values)))
    scale = maximum / limit if maximum else 1.0
    scale = float(np.asarray([scale], dtype="<f2")[0])
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    codes = np.clip(np.rint(values / scale), -limit, limit).astype(np.int8)
    deployed = codes.astype(np.float32) * scale
    return Table(feature, dimension, bits, codes, scale, deployed)


def serialize_table(table: Table) -> bytes:
    feature_ids = {"causal_edge_predicted": 1, "pose_sign_predicted": 2, "pose_delta_sign_predicted": 3}
    states = table.codes.shape[0]
    if table.codes.shape != (states, CLASSES) or not 0 <= table.dimension <= 255:
        raise RuntimeError("invalid IEC1 table geometry")
    header = IEC_HEADER.pack(b"IEC1", 1, feature_ids[table.feature], table.bits, table.dimension, states)
    return header + np.asarray([table.scale], dtype="<f2").tobytes() + pack_signed(table.codes, table.bits)


def deserialize_table(payload: bytes) -> Table:
    if len(payload) < IEC_HEADER.size + 2:
        raise RuntimeError("truncated IEC1 payload")
    magic, version, feature_id, bits, dimension, states = IEC_HEADER.unpack_from(payload)
    names = {1: "causal_edge_predicted", 2: "pose_sign_predicted", 3: "pose_delta_sign_predicted"}
    if magic != b"IEC1" or version != 1 or feature_id not in names or bits not in (4, 6, 8) or not states:
        raise RuntimeError("invalid IEC1 header")
    scale = float(np.frombuffer(payload[IEC_HEADER.size : IEC_HEADER.size + 2], dtype="<f2")[0])
    count = states * CLASSES
    packed_bytes = (count * bits + 7) // 8
    if len(payload) != IEC_HEADER.size + 2 + packed_bytes or not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("invalid IEC1 payload length or scale")
    codes = unpack_signed(payload[-packed_bytes:], count, bits).reshape(states, CLASSES)
    feature = names[feature_id]
    return Table(feature, dimension, bits, codes, scale, codes.astype(np.float32) * scale)  # type: ignore[arg-type]


def probabilities_from_codes(codes: np.ndarray) -> np.ndarray:
    logits = np.asarray(codes, dtype=np.int16).astype(np.float64) / 8.0
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def corrected_probabilities(codes: np.ndarray, context: np.ndarray, table: Table) -> np.ndarray:
    logits = np.asarray(codes, dtype=np.int16).astype(np.float32) / 8.0
    logits = logits + table.values[np.asarray(context, dtype=np.int64)]
    corrected = np.clip(np.rint(logits * 8.0), -32768, 32767).astype(np.int16)
    return probabilities_from_codes(corrected)


def nll_bits(symbols: np.ndarray, probabilities: np.ndarray) -> float:
    chosen = probabilities[np.arange(len(symbols)), np.asarray(symbols, dtype=np.int64)]
    return float((-np.log2(chosen.astype(np.float64))).sum())


def group_positions(runtime_root: Path) -> list[np.ndarray]:
    import torch

    sys.path.insert(0, str(runtime_root))
    try:
        import runtime.f26_inflate as runtime

        renderer = runtime._load_renderer(runtime_root / "cpr1")
        return [
            np.flatnonzero(mask.detach().cpu().numpy().reshape(-1))
            for mask in renderer.group_masks(torch.device("cpu"))
        ]
    finally:
        sys.path.pop(0)


def load_pose_coefficients(runtime_root: Path, archive: Path) -> np.ndarray:
    import torch

    sys.path.insert(0, str(runtime_root))
    try:
        import runtime.f26_inflate as inflate
        from runtime.carrier_repack import materialize_cpr1, split_frame0_selector_carrier

        parts = inflate.read_residual_archive(archive)
        renderer = inflate._load_renderer(runtime_root / "cpr1")
        carrier, _ = split_frame0_selector_carrier(parts.carrier_blob)
        canonical = materialize_cpr1(carrier, renderer)
        semantic_pose = struct.pack("<II", 40_252, len(canonical)) + bytes(40_252) + canonical
        _, _, coefficients = renderer.unpack_semantic_pose(semantic_pose)
        values = coefficients.detach().cpu().numpy().astype(np.float32, copy=False)
    finally:
        sys.path.pop(0)
    if values.shape != (FRAME_COUNT, 12) or not np.all(np.isfinite(values)):
        raise RuntimeError("invalid decoded pose coefficients")
    if not isinstance(coefficients, torch.Tensor):
        raise RuntimeError("pose coefficients did not parse as a tensor")
    return np.asarray(values)


def causal_group_state(
    current: np.ndarray,
    known: np.ndarray,
    predicted: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    rows, cols = divmod(positions, WIDTH)
    neighbors = np.stack(
        [
            np.where(rows > 0, positions - WIDTH, -1),
            np.where(rows + 1 < HEIGHT, positions + WIDTH, -1),
            np.where(cols > 0, positions - 1, -1),
            np.where(cols + 1 < WIDTH, positions + 1, -1),
        ],
        axis=1,
    )
    valid = neighbors >= 0
    safe = np.maximum(neighbors, 0)
    seen = valid & known[safe]
    labels = current[safe]
    minimum = np.where(seen, labels, CLASSES).min(axis=1)
    maximum = np.where(seen, labels, -1).max(axis=1)
    has = seen.any(axis=1)
    mixed = has & (minimum != maximum)
    state = np.zeros(len(positions), dtype=np.uint8)
    state[has & ~mixed & (minimum == predicted)] = 1
    state[has & ~mixed & (minimum != predicted)] = 2
    state[mixed] = 3
    return state


def causal_states(events: np.ndarray, predicted: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    current = np.zeros(EVENTS_PER_FRAME, dtype=np.uint8)
    known = np.zeros(EVENTS_PER_FRAME, dtype=bool)
    result = np.empty(EVENTS_PER_FRAME, dtype=np.uint8)
    offset = 0
    for positions in groups:
        end = offset + len(positions)
        result[offset:end] = causal_group_state(current, known, predicted[offset:end], positions)
        current[positions] = events[offset:end]
        known[positions] = True
        offset = end
    if offset != EVENTS_PER_FRAME:
        raise RuntimeError("group positions do not cover a frame")
    return result


def spatial_frame(events: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    flat = np.empty(EVENTS_PER_FRAME, dtype=np.uint8)
    offset = 0
    for positions in groups:
        end = offset + len(positions)
        flat[positions] = events[offset:end]
        offset = end
    if offset != EVENTS_PER_FRAME:
        raise RuntimeError("group positions do not cover a frame")
    return flat.reshape(HEIGHT, WIDTH)


def update_statistics(target: np.ndarray, expected: np.ndarray, context: np.ndarray, symbols: np.ndarray, probabilities: np.ndarray) -> None:
    states = target.shape[0]
    target += np.bincount(
        np.asarray(context, dtype=np.int64) * CLASSES + np.asarray(symbols, dtype=np.int64),
        minlength=states * CLASSES,
    ).reshape(states, CLASSES)
    for klass in range(CLASSES):
        expected[:, klass] += np.bincount(context, weights=probabilities[:, klass], minlength=states)


def update_pose_statistics(
    target: np.ndarray,
    expected: np.ndarray,
    states: np.ndarray,
    predicted: np.ndarray,
    symbols: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    target_by_prediction = np.bincount(
        predicted * CLASSES + symbols.astype(np.int64),
        minlength=CLASSES * CLASSES,
    ).reshape(CLASSES, CLASSES)
    expected_by_prediction = np.empty((CLASSES, CLASSES), dtype=np.float64)
    for klass in range(CLASSES):
        expected_by_prediction[:, klass] = np.bincount(
            predicted, weights=probabilities[:, klass], minlength=CLASSES
        )
    for dimension, state in enumerate(np.asarray(states, dtype=np.int64)):
        start = int(state) * CLASSES
        target[dimension, start : start + CLASSES] += target_by_prediction
        expected[dimension, start : start + CLASSES] += expected_by_prediction


def fitted_values(target: np.ndarray, expected: np.ndarray) -> np.ndarray:
    values = np.log((target + 0.5) / (expected + 0.5))
    return values - values.mean(axis=1, keepdims=True)


def surrogate_gain_bits(target: np.ndarray, expected: np.ndarray, values: np.ndarray) -> float:
    gain_nats = float((target * values - expected * np.expm1(values)).sum())
    return gain_nats / np.log(2.0)


def code_record(code_root: Path, frame: int) -> dict[str, Any]:
    receipt = json.loads((code_root / f"codes_{frame:04d}.json").read_text())
    return receipt["codes"]


def load_codes(code_root: Path, frame: int) -> np.ndarray:
    path = code_root / f"codes_{frame:04d}.npy"
    record = code_record(code_root, frame)
    if file_record(path) != record:
        raise RuntimeError(f"F26 code custody failed at frame {frame}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.dtype != np.int16 or values.shape != (EVENTS_PER_FRAME, CLASSES):
        raise RuntimeError(f"invalid F26 codes at frame {frame}")
    return np.asarray(values)


def contexts_for_frame(
    feature: Feature,
    dimension: int,
    frame: int,
    predicted: np.ndarray,
    events: np.ndarray | None,
    groups: list[np.ndarray],
    coefficients: np.ndarray,
) -> np.ndarray:
    if feature == "causal_edge_predicted":
        if events is None:
            raise RuntimeError("causal edge context requires decoded current-frame events")
        state = causal_states(events, predicted, groups).astype(np.int64)
    elif feature == "pose_sign_predicted":
        state = np.full(len(predicted), int(coefficients[frame, dimension] >= 0.0), dtype=np.int64)
    else:
        delta = 0.0 if frame == 0 else float(coefficients[frame, dimension] - coefficients[frame - 1, dimension])
        state = np.full(len(predicted), int(delta >= 0.0), dtype=np.int64)
    return state * CLASSES + predicted.astype(np.int64)


def analysis_checkpoint(output: Path, frame: int, arrays: dict[str, np.ndarray]) -> None:
    path = output / "checkpoints" / "analysis_collect" / f"through_frame_{frame:04d}.npz"
    atomic_npz(path, next_frame=np.asarray([frame]), **arrays)
    atomic_json(
        output / "checkpoints" / "analysis_collect" / "LATEST.json",
        {"schema": "ddm_sr1_analysis_checkpoint.v1", "next_frame": frame, "checkpoint": file_record(path)},
    )


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    require_file(args.archive, ARCHIVE_SHA256)
    require_file(args.library, LIBRARY_SHA256)
    source = SourceSymbols(args.dt1_manifest)
    groups = group_positions(args.runtime)
    coefficients = load_pose_coefficients(args.runtime, args.archive)
    arrays = {
        "edge_target_train": np.zeros((20, CLASSES), dtype=np.float64),
        "edge_expected_train": np.zeros((20, CLASSES), dtype=np.float64),
        "edge_target_full": np.zeros((20, CLASSES), dtype=np.float64),
        "edge_expected_full": np.zeros((20, CLASSES), dtype=np.float64),
        "pose_sign_target_train": np.zeros((12, 10, CLASSES), dtype=np.float64),
        "pose_sign_expected_train": np.zeros((12, 10, CLASSES), dtype=np.float64),
        "pose_sign_target_full": np.zeros((12, 10, CLASSES), dtype=np.float64),
        "pose_sign_expected_full": np.zeros((12, 10, CLASSES), dtype=np.float64),
        "pose_delta_target_train": np.zeros((12, 10, CLASSES), dtype=np.float64),
        "pose_delta_expected_train": np.zeros((12, 10, CLASSES), dtype=np.float64),
        "pose_delta_target_full": np.zeros((12, 10, CLASSES), dtype=np.float64),
        "pose_delta_expected_full": np.zeros((12, 10, CLASSES), dtype=np.float64),
        "base_nll": np.zeros(3, dtype=np.float64),
    }
    latest = args.output / "checkpoints" / "analysis_collect" / "LATEST.json"
    start = 0
    if latest.exists():
        progress = json.loads(latest.read_text())
        checkpoint = Path(progress["checkpoint"]["path"])
        require_file(checkpoint, progress["checkpoint"]["sha256"])
        loaded = np.load(checkpoint, allow_pickle=False)
        start = int(loaded["next_frame"][0])
        for name in arrays:
            arrays[name] = np.asarray(loaded[name]).copy()
    started = time.time()
    for frame in range(start, FRAME_COUNT):
        symbols = source.frame(frame)
        codes = load_codes(args.codes, frame)
        probabilities = probabilities_from_codes(codes)
        predicted = codes.argmax(axis=1).astype(np.int64)
        edge_context = contexts_for_frame(
            "causal_edge_predicted", 0, frame, predicted, symbols, groups, coefficients
        )
        split = 0 if frame % 2 == 0 else 1
        arrays["base_nll"][split] += nll_bits(symbols, probabilities)
        arrays["base_nll"][2] += nll_bits(symbols, probabilities)
        update_statistics(arrays["edge_target_full"], arrays["edge_expected_full"], edge_context, symbols, probabilities)
        if split == 0:
            update_statistics(arrays["edge_target_train"], arrays["edge_expected_train"], edge_context, symbols, probabilities)
        sign_states = (coefficients[frame] >= 0.0).astype(np.int64)
        delta = np.zeros(12, dtype=np.float32) if frame == 0 else coefficients[frame] - coefficients[frame - 1]
        delta_states = (delta >= 0.0).astype(np.int64)
        for prefix, states in (("pose_sign", sign_states), ("pose_delta", delta_states)):
            update_pose_statistics(
                arrays[f"{prefix}_target_full"],
                arrays[f"{prefix}_expected_full"],
                states,
                predicted,
                symbols,
                probabilities,
            )
            if split == 0:
                update_pose_statistics(
                    arrays[f"{prefix}_target_train"],
                    arrays[f"{prefix}_expected_train"],
                    states,
                    predicted,
                    symbols,
                    probabilities,
                )
        if (frame + 1) % CHECKPOINT_FRAMES == 0 or frame + 1 == FRAME_COUNT:
            analysis_checkpoint(args.output, frame + 1, arrays)
            print(json.dumps({"stage": "collect", "frames": frame + 1, "elapsed_s": round(time.time() - started, 3)}), flush=True)

    pose_ranking = []
    for dimension in range(12):
        for prefix, feature in (("pose_sign", "pose_sign_predicted"), ("pose_delta", "pose_delta_sign_predicted")):
            train_values = fitted_values(arrays[f"{prefix}_target_train"][dimension], arrays[f"{prefix}_expected_train"][dimension])
            holdout_target = arrays[f"{prefix}_target_full"][dimension] - arrays[f"{prefix}_target_train"][dimension]
            holdout_expected = arrays[f"{prefix}_expected_full"][dimension] - arrays[f"{prefix}_expected_train"][dimension]
            pose_ranking.append(
                {
                    "feature": feature,
                    "dimension": dimension,
                    "holdout_surrogate_gain_bits": surrogate_gain_bits(holdout_target, holdout_expected, train_values),
                }
            )
    pose_ranking.sort(key=lambda row: row["holdout_surrogate_gain_bits"], reverse=True)
    selected = [
        {"name": "causal_edge", "feature": "causal_edge_predicted", "dimension": 0},
        {"name": "pose_cross_stream", **{key: pose_ranking[0][key] for key in ("feature", "dimension")}},
    ]
    evaluation_nll = {candidate["name"]: {str(bits): np.zeros(3, dtype=np.float64) for bits in (4, 6, 8)} for candidate in selected}
    tables: dict[str, dict[int, tuple[Table, Table]]] = {}
    for candidate in selected:
        if candidate["feature"] == "causal_edge_predicted":
            train_target, train_expected = arrays["edge_target_train"], arrays["edge_expected_train"]
            full_target, full_expected = arrays["edge_target_full"], arrays["edge_expected_full"]
        else:
            prefix = "pose_sign" if candidate["feature"] == "pose_sign_predicted" else "pose_delta"
            dimension = int(candidate["dimension"])
            train_target, train_expected = arrays[f"{prefix}_target_train"][dimension], arrays[f"{prefix}_expected_train"][dimension]
            full_target, full_expected = arrays[f"{prefix}_target_full"][dimension], arrays[f"{prefix}_expected_full"][dimension]
        tables[candidate["name"]] = {
            bits: (
                quantize_table(candidate["feature"], int(candidate["dimension"]), fitted_values(train_target, train_expected), bits),
                quantize_table(candidate["feature"], int(candidate["dimension"]), fitted_values(full_target, full_expected), bits),
            )
            for bits in (4, 6, 8)
        }

    evaluation_latest = args.output / "checkpoints" / "analysis_evaluate" / "LATEST.json"
    evaluation_start = 0
    if evaluation_latest.exists():
        progress = json.loads(evaluation_latest.read_text())
        evaluation_start = int(progress["next_frame"])
        for name, rows in progress["evaluation_nll"].items():
            for bits, value in rows.items():
                evaluation_nll[name][bits] = np.asarray(value, dtype=np.float64)
    for frame in range(evaluation_start, FRAME_COUNT):
        symbols = source.frame(frame)
        codes = load_codes(args.codes, frame)
        predicted = codes.argmax(axis=1).astype(np.int64)
        for candidate in selected:
            context = contexts_for_frame(
                candidate["feature"], int(candidate["dimension"]), frame, predicted,
                symbols if candidate["feature"] == "causal_edge_predicted" else None,
                groups, coefficients,
            )
            for bits, (train_table, full_table) in tables[candidate["name"]].items():
                evaluation_nll[candidate["name"]][str(bits)][2] += nll_bits(
                    symbols, corrected_probabilities(codes, context, full_table)
                )
                if frame % 2:
                    evaluation_nll[candidate["name"]][str(bits)][1] += nll_bits(
                        symbols, corrected_probabilities(codes, context, train_table)
                    )
        if (frame + 1) % CHECKPOINT_FRAMES == 0 or frame + 1 == FRAME_COUNT:
            checkpoint = args.output / "checkpoints" / "analysis_evaluate" / f"through_frame_{frame + 1:04d}.json"
            atomic_json(
                checkpoint,
                {
                    "schema": "ddm_sr1_evaluation_checkpoint.v1",
                    "next_frame": frame + 1,
                    "evaluation_nll": {
                        name: {bits: value.tolist() for bits, value in rows.items()}
                        for name, rows in evaluation_nll.items()
                    },
                },
            )
            atomic_json(
                evaluation_latest,
                {
                    "next_frame": frame + 1,
                    "checkpoint": file_record(checkpoint),
                    "evaluation_nll": {
                        name: {bits: value.tolist() for bits, value in rows.items()}
                        for name, rows in evaluation_nll.items()
                    },
                },
            )
            print(json.dumps({"stage": "evaluate", "frames": frame + 1}), flush=True)

    rows = []
    retained_tables = args.output / "retained" / "tables"
    for candidate in selected:
        for bits, (_, full_table) in tables[candidate["name"]].items():
            payload = serialize_table(full_table)
            table_path = retained_tables / f"{candidate['name']}_int{bits}.iec1"
            atomic_bytes(table_path, payload)
            holdout_base = arrays["base_nll"][1]
            holdout_nll = evaluation_nll[candidate["name"]][str(bits)][1]
            full_nll = evaluation_nll[candidate["name"]][str(bits)][2]
            rows.append(
                {
                    **candidate,
                    "bits": bits,
                    "table_payload": file_record(table_path),
                    "holdout_base_bits": float(holdout_base),
                    "holdout_corrected_bits": float(holdout_nll),
                    "holdout_gain_bits": float(holdout_base - holdout_nll),
                    "holdout_net_bits_after_table": float(holdout_base - holdout_nll - len(payload) * 8),
                    "full_base_bits": float(arrays["base_nll"][2]),
                    "full_corrected_bits": float(full_nll),
                    "full_fit_gain_bits": float(arrays["base_nll"][2] - full_nll),
                }
            )
    best_by_name = {}
    for name in ("causal_edge", "pose_cross_stream"):
        best_by_name[name] = max(
            (row for row in rows if row["name"] == name),
            key=lambda row: row["holdout_net_bits_after_table"],
        )
    result = {
        "schema": "ddm_sr1_implicit_edge_analysis.v1",
        "axis": AXIS,
        "score_claim": False,
        "events": TOTAL_EVENTS,
        "selection_mode": "even-frame fit, odd-frame holdout; full-n600 refit only after selection",
        "baseline": {
            "token_bytes": BASE_TOKEN_BYTES,
            "token_sha256": BASE_TOKEN_SHA256,
            "nll_bits_full": float(arrays["base_nll"][2]),
            "achieved_bits_per_symbol": BASE_TOKEN_BYTES * 8 / TOTAL_EVENTS,
        },
        "pose_ranking": pose_ranking,
        "candidates": rows,
        "selected_for_real_coder": best_by_name,
        "falsifier": "rate route closes when charged real-coder gain is <=1% of 114706 B",
        "inputs": {
            "probe_script": file_record(Path(__file__)),
            "archive": file_record(args.archive),
            "library": file_record(args.library),
            "dt1_manifest": file_record(args.dt1_manifest),
            "probability_export": file_record(args.codes / "EXPORT_RESULT.json"),
        },
        "resumable_from_disk": True,
        "checkpoint_frames": CHECKPOINT_FRAMES,
    }
    atomic_json(args.output / "ANALYSIS_RESULT.json", result)
    return result


def selected_table(args: argparse.Namespace, candidate: str) -> tuple[Table, dict[str, Any]]:
    analysis = json.loads((args.output / "ANALYSIS_RESULT.json").read_text())
    row = analysis["selected_for_real_coder"][candidate]
    path = Path(row["table_payload"]["path"])
    require_file(path, row["table_payload"]["sha256"])
    return deserialize_table(path.read_bytes()), row


def rc64_module() -> Any:
    path = Path(__file__).resolve().parent / "ddm_rc64p_native_cpu_decode" / "route_b_rc64.py"
    return import_from_path("_sr1_route_b_rc64", path)


def run_encode(args: argparse.Namespace) -> dict[str, Any]:
    table, analysis_row = selected_table(args, args.candidate)
    require_file(args.library, LIBRARY_SHA256)
    source = SourceSymbols(args.dt1_manifest)
    groups = group_positions(args.runtime)
    coefficients = load_pose_coefficients(args.runtime, args.archive)
    module = rc64_module()
    retained = args.output / "retained" / "candidates" / args.candidate
    checkpoint_root = retained / "encode_checkpoints"
    latest = checkpoint_root / "LATEST.json"
    start = 0
    if latest.exists():
        progress = json.loads(latest.read_text())
        checkpoint = Path(progress["checkpoint"]["path"])
        require_file(checkpoint, progress["checkpoint"]["sha256"])
        encoder = module.NativeRc64Encoder(args.library, checkpoint.read_bytes())
        start = int(progress["next_frame"])
    else:
        encoder = module.NativeRc64Encoder(args.library)
    started = time.time()
    for frame in range(start, FRAME_COUNT):
        symbols = source.frame(frame)
        codes = load_codes(args.codes, frame)
        predicted = codes.argmax(axis=1).astype(np.int64)
        context = contexts_for_frame(
            table.feature, table.dimension, frame, predicted,
            symbols if table.feature == "causal_edge_predicted" else None,
            groups, coefficients,
        )
        encoder.encode(symbols.astype(np.int32), corrected_probabilities(codes, context, table))
        if (frame + 1) % CHECKPOINT_FRAMES == 0 or frame + 1 == FRAME_COUNT:
            checkpoint_path = checkpoint_root / f"through_frame_{frame + 1:04d}.rc64.state"
            atomic_bytes(checkpoint_path, encoder.snapshot())
            atomic_json(
                checkpoint_root / f"through_frame_{frame + 1:04d}.json",
                {"schema": "ddm_sr1_rc64_encode_checkpoint.v1", "next_frame": frame + 1, "checkpoint": file_record(checkpoint_path)},
            )
            atomic_json(latest, {"next_frame": frame + 1, "checkpoint": file_record(checkpoint_path)})
            print(json.dumps({"stage": "encode", "candidate": args.candidate, "frames": frame + 1, "elapsed_s": round(time.time() - started, 3)}), flush=True)
    token_payload = encoder.finish()
    encoder.close()
    token_path = retained / "tokens.rc64"
    atomic_bytes(token_path, token_payload)
    model_path = Path(analysis_row["table_payload"]["path"])
    model_payload = model_path.read_bytes()
    composite = COMPOSITE_HEADER.pack(b"IEP1", len(model_payload)) + model_payload + token_payload
    composite_path = retained / "candidate.iep1_rc64"
    atomic_bytes(composite_path, composite)
    result = {
        "schema": "ddm_sr1_implicit_edge_encode.v1",
        "axis": AXIS,
        "score_claim": False,
        "candidate": args.candidate,
        "model_payload": file_record(model_path),
        "token_payload": file_record(token_path),
        "composite_payload": file_record(composite_path),
        "baseline_token_bytes": BASE_TOKEN_BYTES,
        "charged_delta_bytes": len(composite) - BASE_TOKEN_BYTES,
        "charged_gain_fraction": (BASE_TOKEN_BYTES - len(composite)) / BASE_TOKEN_BYTES,
        "resumable_from_disk": True,
        "checkpoint_count": len(list(checkpoint_root.glob("through_frame_*.rc64.state"))),
        "wall_s": time.time() - started,
    }
    atomic_json(retained / "ENCODE_RESULT.json", result)
    return result


def parse_composite(path: Path) -> tuple[Table, bytes]:
    payload = path.read_bytes()
    if len(payload) < COMPOSITE_HEADER.size:
        raise RuntimeError("truncated IEP1 payload")
    magic, model_bytes = COMPOSITE_HEADER.unpack_from(payload)
    if magic != b"IEP1" or not model_bytes or COMPOSITE_HEADER.size + model_bytes >= len(payload):
        raise RuntimeError("invalid IEP1 framing")
    model = payload[COMPOSITE_HEADER.size : COMPOSITE_HEADER.size + model_bytes]
    return deserialize_table(model), payload[COMPOSITE_HEADER.size + model_bytes :]


def run_decode(args: argparse.Namespace) -> dict[str, Any]:
    retained = args.output / "retained" / "candidates" / args.candidate
    encode_result = json.loads((retained / "ENCODE_RESULT.json").read_text())
    composite_path = Path(encode_result["composite_payload"]["path"])
    require_file(composite_path, encode_result["composite_payload"]["sha256"])
    require_file(args.library, LIBRARY_SHA256)
    table, tokens = parse_composite(composite_path)
    source = SourceSymbols(args.dt1_manifest)
    groups = group_positions(args.runtime)
    coefficients = load_pose_coefficients(args.runtime, args.archive)
    module = rc64_module()
    latest = retained / "decode_chunks" / "LATEST.json"
    if latest.exists():
        progress = json.loads(latest.read_text())
        checkpoint_path = Path(progress["decoder_checkpoint"]["path"])
        require_file(checkpoint_path, progress["decoder_checkpoint"]["sha256"])
        decoder = module.NativeRc64Decoder(args.library, checkpoint_path.read_bytes())
        decode_start = int(progress["next_frame"])
    else:
        decoder = module.NativeRc64Decoder(args.library, tokens)
        decode_start = 0
    chunk_root = retained / "decode_chunks"
    started = time.time()
    event_digest = hashlib.sha256()
    spatial_digest = hashlib.sha256()
    for stage_start in range(0, decode_start, CHECKPOINT_FRAMES):
        stage_end = min(stage_start + CHECKPOINT_FRAMES, FRAME_COUNT)
        for prefix, digest in (("events", event_digest), ("spatial", spatial_digest)):
            path = chunk_root / f"{prefix}_{stage_start:04d}_{stage_end:04d}.bin"
            if not path.is_file():
                raise RuntimeError(f"missing retained decode chunk on resume: {path}")
            with path.open("rb") as stream:
                while block := stream.read(1 << 20):
                    digest.update(block)
    for stage_start in range(decode_start, FRAME_COUNT, CHECKPOINT_FRAMES):
        stage_end = min(stage_start + CHECKPOINT_FRAMES, FRAME_COUNT)
        event_path = chunk_root / f"events_{stage_start:04d}_{stage_end:04d}.bin"
        spatial_path = chunk_root / f"spatial_{stage_start:04d}_{stage_end:04d}.bin"
        if event_path.exists() or spatial_path.exists():
            raise RuntimeError("decode is fail-closed rather than silently reusing partial chunks")
        event_buffer = bytearray()
        spatial_buffer = bytearray()
        for frame in range(stage_start, stage_end):
            codes = load_codes(args.codes, frame)
            expected = source.frame(frame)
            current = np.zeros(EVENTS_PER_FRAME, dtype=np.uint8)
            known = np.zeros(EVENTS_PER_FRAME, dtype=bool)
            known_events = np.empty(EVENTS_PER_FRAME, dtype=np.uint8)
            offset = 0
            for positions in groups:
                end = offset + len(positions)
                predicted = codes[offset:end].argmax(axis=1).astype(np.int64)
                if table.feature == "causal_edge_predicted":
                    state = causal_group_state(current, known, predicted, positions)
                    context = state.astype(np.int64) * CLASSES + predicted
                else:
                    context = contexts_for_frame(
                        table.feature, table.dimension, frame, predicted, None, groups, coefficients
                    )
                probabilities = corrected_probabilities(codes[offset:end], context, table)
                decoded = decoder.decode(None, probabilities).astype(np.uint8)
                current[positions] = decoded
                known[positions] = True
                known_events[offset:end] = decoded
                offset = end
            if offset != EVENTS_PER_FRAME or not np.array_equal(known_events, expected):
                raise RuntimeError(f"candidate decoded symbols differ at frame {frame}")
            event_buffer.extend(known_events.tobytes())
            spatial_buffer.extend(current.tobytes())
        atomic_bytes(event_path, bytes(event_buffer))
        atomic_bytes(spatial_path, bytes(spatial_buffer))
        event_digest.update(event_buffer)
        spatial_digest.update(spatial_buffer)
        checkpoint = decoder.get_compressed().astype("<u4", copy=False).tobytes()
        checkpoint_path = chunk_root / f"decoder_through_frame_{stage_end:04d}.state"
        atomic_bytes(checkpoint_path, checkpoint)
        atomic_json(
            chunk_root / f"stage_{stage_start:04d}_{stage_end:04d}.json",
            {
                "schema": "ddm_sr1_rc64_decode_stage.v1",
                "frames": [stage_start, stage_end],
                "events": file_record(event_path),
                "spatial": file_record(spatial_path),
                "decoder_checkpoint": file_record(checkpoint_path),
            },
        )
        atomic_json(
            latest,
            {"next_frame": stage_end, "decoder_checkpoint": file_record(checkpoint_path)},
        )
        print(json.dumps({"stage": "decode", "candidate": args.candidate, "frames": stage_end, "elapsed_s": round(time.time() - started, 3)}), flush=True)
    if not decoder.is_empty():
        raise RuntimeError("RC64 decoder terminal state is not empty")
    decoder.close()
    event_path = retained / "decoded_symbols.rc64.bin"
    spatial_path = retained / "decoded_spatial_tokens.rc64.bin"
    for destination, prefix in ((event_path, "events"), (spatial_path, "spatial")):
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as output:
            for stage_start in range(0, FRAME_COUNT, CHECKPOINT_FRAMES):
                stage_end = min(stage_start + CHECKPOINT_FRAMES, FRAME_COUNT)
                with (chunk_root / f"{prefix}_{stage_start:04d}_{stage_end:04d}.bin").open("rb") as source_stream:
                    while block := source_stream.read(1 << 20):
                        output.write(block)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    if event_digest.hexdigest() != EVENT_SHA256 or spatial_digest.hexdigest() != SPATIAL_SHA256:
        raise RuntimeError("decoded n600 digests differ from the canonical retained corpus")
    result = {
        "schema": "ddm_sr1_implicit_edge_decode.v1",
        "axis": AXIS,
        "score_claim": False,
        "candidate": args.candidate,
        "composite_payload": file_record(composite_path),
        "decoded_symbols": file_record(event_path),
        "decoded_spatial_tokens": file_record(spatial_path),
        "symbol_identity": True,
        "terminal_state_empty": True,
        "events": TOTAL_EVENTS,
        "stage_checkpoints": len(list(chunk_root.glob("decoder_through_frame_*.state"))),
        "wall_s": time.time() - started,
    }
    atomic_json(retained / "DECODE_RESULT.json", result)
    return result


def storage_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(output)
    free = stat.f_bavail * stat.f_frsize
    required = 2 * TOTAL_EVENTS + (1 << 30)
    if free < required:
        raise RuntimeError(f"storage preflight failed: {free} < {required}")
    result = {
        "schema": "ddm_sr1_storage_preflight.v1",
        "output": str(output.resolve()),
        "free_bytes": free,
        "required_bytes": required,
        "routing": "APDataStore selected because VertigoDataTier had only 26 GiB free and was 99% full",
        "cleanup": "certify-or-block; retained evidence is not auto-deleted",
    }
    atomic_json(output / "STORAGE_PREFLIGHT.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("stage", choices=("analyze", "encode", "decode"))
    value.add_argument("--candidate", choices=("causal_edge", "pose_cross_stream"))
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    value.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    value.add_argument("--dt1-manifest", type=Path, default=DEFAULT_DT1)
    value.add_argument("--codes", type=Path, default=DEFAULT_CODES)
    value.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.stage in ("encode", "decode") and not args.candidate:
        raise SystemExit("--candidate is required for encode/decode")
    args.output.mkdir(parents=True, exist_ok=True)
    cleanup_stale_atomic_scratch(args.output)
    storage_preflight(args.output)
    if args.stage == "analyze":
        result = run_analysis(args)
    elif args.stage == "encode":
        result = run_encode(args)
    else:
        result = run_decode(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
