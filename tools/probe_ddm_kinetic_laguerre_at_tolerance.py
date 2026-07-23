#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Fit and measure the bounded DDM kinetic Laguerre at-tolerance ladder.

The output is research-only local evidence.  n64 is compute/integrity evidence;
only the complete n600 cached-label ladder can decide Stage A.  Frozen scorer
work is gated behind a complete Stage-A winner and remains macOS-CPU advisory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import os
import shutil
import struct
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import brotli
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.spatial import ConvexHull, QhullError, cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.boundary_math.power_diagram_witness import (  # noqa: E402
    open_stored_npy_memmap,
    sha256_file,
)
from tac.optimization.direct_description_minimizer import (  # noqa: E402
    rfc8785_canonicalize,
)

SCHEMA = "ddm_kinetic_laguerre_at_tolerance_probe_receipt.v1"
PROGRAM_MAGIC = b"KLP1"
ENVELOPE_MAGIC = b"KLC1"
RICE_MAGIC = b"KLR1"
PROGRAM_HEADER = struct.Struct("<4sI")
ENVELOPE_HEADER = struct.Struct("<4sB3xII")
RICE_HEADER = struct.Struct("<4sI")
CODEC_IDS = {
    "brotli_q11": 1,
    "lzma_xz_preset9_extreme": 2,
    "split_metadata_plus_rice_golomb": 3,
}
CODEC_NAMES = {value: key for key, value in CODEC_IDS.items()}
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
N_CLASSES = len(CLASS_NAMES)
AXIS = "[macOS-CPU frozen-scorer advisory]"
LANE_ID = "lane_ddm_m2_kinetic_laguerre_probe_20260723"
COORD_Q = 8.0
WEIGHT_Q = 4.0
XI_Q = 1024.0
POSE_RIDGE_LAMBDA = 0.05
METRIC_Q = 1_000_000.0
TARGET_SOURCE_BYTES = 37_545_489
V19B_MATCH_ERRORS = 3_137_206
PAIR_BATCH = 16


class ProbeError(RuntimeError):
    """Fail-closed probe error."""


class RunnerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Literal["tools/probe_ddm_kinetic_laguerre_at_tolerance.py"]
    present_in_delegated_snapshot: bool
    status: str
    semantic_argv: tuple[str, ...]


class InputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_cache_path: str
    target_cache_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    modules_path: str
    modules_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v19b_receipt_path: str
    v19b_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v19b_n600_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sealed_v8_control: dict[str, Any]


class PopulationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pair_ladder: tuple[Literal[64, 600], Literal[64, 600]]
    seg_sites_n600: Literal[117_964_800]
    maximum_seg_errors: Literal[136_839]
    maximum_d_seg: Literal[0.001159998576]
    maximum_archive_bytes: Literal[200_000]
    current_predictor_home_bytes: Literal[100_099]

    @model_validator(mode="after")
    def _ladder(self) -> PopulationSpec:
        if tuple(self.pair_ladder) != (64, 600):
            raise ValueError("pair ladder must be n64 then n600")
        return self


class RepresentationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["kinetic_anisotropic_laguerre_regular_triangulation_event_code"]
    decode_object: Literal["image_plane_partition_then_scorer_free_rgb_pullback"]
    spatial_charts: tuple[str, ...]
    site_counts: tuple[Literal[64, 128, 256, 512], ...]
    trajectory_degrees: tuple[Literal[1, 2, 3], ...]
    metric_modes: tuple[
        Literal[
            "isotropic_power_control",
            "shared_chart_anisotropic_spd",
            "projective_depth_stratified",
        ],
        ...,
    ]
    temporal_modes: tuple[
        Literal[
            "independent_frame_control",
            "spline_sites_weights_plus_sparse_regular_triangulation_flips",
        ],
        ...,
    ]
    shared_edge_accounting: str
    event_types: tuple[str, ...]
    real_coder_race: tuple[str, ...]
    forbidden_archive_payloads: tuple[str, ...]


class ExecutionAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    authority_file: str
    sha256: Literal["cb5bc9d90cc9285407cd830dc9c4f310aabb706ce437cfa4effab3c5499248a8"]
    delegation_checkpoint_key: Literal["codex_delegate:ddm_m2_kinetic_laguerre_probe:20260723T064512Z"]


class StorageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    waterfall: tuple[str, ...]
    local_bulk_allowed: Literal[False]
    preflight_required: Literal[True]
    success_only_scratch_cleanup: Literal[True]
    certify_or_block_cleanup: Literal[True]


class DDMKineticLaguerreAtToleranceProbeV1(BaseModel):
    """Typed live transition of the M1 design-only contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMKineticLaguerreAtToleranceProbeV1"] = Field(alias="schema", serialization_alias="schema")
    run_id: str = Field(min_length=8)
    seed: Literal[1234]
    research_only: Literal[True]
    execution_allowed: Literal[True]
    score_claim: Literal[False]
    promotion_eligible: Literal[False]
    evidence_axis: Literal["[macOS-CPU frozen-scorer advisory]"]
    execution_authority: ExecutionAuthority
    runner: RunnerSpec
    inputs: InputSpec
    population: PopulationSpec
    representation: RepresentationSpec
    stages: tuple[dict[str, Any], ...]
    controls: tuple[str, ...]
    acceptance: dict[str, Any]
    storage: StorageSpec
    main_landing_review_required: Literal[True]

    @model_validator(mode="after")
    def _sealed_ladder(self) -> DDMKineticLaguerreAtToleranceProbeV1:
        if self.representation.site_counts != (64, 128, 256, 512):
            raise ValueError("site ladder differs from preregistration")
        if self.representation.trajectory_degrees != (1, 2, 3):
            raise ValueError("degree ladder differs from preregistration")
        if set(self.representation.real_coder_race) != set(CODEC_IDS):
            raise ValueError("real coder race differs from preregistration")
        if len(self.stages) != 2:
            raise ValueError("exactly Stage A and Stage B are required")
        return self

    def typed_config_hash(self) -> str:
        return hashlib.sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True))).hexdigest()


@dataclass(frozen=True, slots=True)
class MetricSpec:
    mode: str
    row_scale: float = 1.0
    col_scale: float = 1.0
    horizon_row: float = 0.0
    depth_alpha: float = 4.0

    def as_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "row_scale": self.row_scale,
            "col_scale": self.col_scale,
            "horizon_row": self.horizon_row,
            "depth_alpha": self.depth_alpha,
        }


@dataclass(frozen=True, slots=True)
class ProgramState:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class DecodedProgram:
    metadata: dict[str, Any]
    site_classes: np.ndarray
    sites: np.ndarray
    class_weights: np.ndarray
    palette_rgb: np.ndarray
    event_rows: np.ndarray


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ProbeError(f"expected regular file: {path}")
    return path.read_bytes()


def _publish_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_regular(path) != payload:
            raise ProbeError(f"immutable artifact differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _publish_immutable(path, rfc8785_canonicalize(value))


def _array_spec(array: np.ndarray, offset: int) -> dict[str, Any]:
    arr = np.ascontiguousarray(array)
    return {
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
        "offset": offset,
        "nbytes": arr.nbytes,
    }


def encode_raw_program(program: ProgramState) -> bytes:
    """Canonical metadata plus fixed-order little-endian integer arrays."""

    ordered = []
    offset = 0
    specs: dict[str, Any] = {}
    for name in sorted(program.arrays):
        arr = np.asarray(program.arrays[name])
        if arr.dtype.kind not in "iu":
            raise ProbeError(f"program array must be integer: {name}")
        arr = np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder("<"), copy=False))
        specs[name] = _array_spec(arr, offset)
        ordered.append(arr.tobytes(order="C"))
        offset += arr.nbytes
    metadata = {**program.metadata, "arrays": specs}
    meta_bytes = rfc8785_canonicalize(metadata)
    return PROGRAM_HEADER.pack(PROGRAM_MAGIC, len(meta_bytes)) + meta_bytes + b"".join(ordered)


def parse_raw_program(payload: bytes) -> ProgramState:
    if len(payload) < PROGRAM_HEADER.size:
        raise ProbeError("truncated KLP1 program")
    magic, meta_size = PROGRAM_HEADER.unpack_from(payload)
    if magic != PROGRAM_MAGIC or meta_size > len(payload) - PROGRAM_HEADER.size:
        raise ProbeError("invalid KLP1 header")
    start = PROGRAM_HEADER.size
    try:
        metadata = json.loads(payload[start : start + meta_size])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError("invalid KLP1 metadata") from exc
    specs = metadata.pop("arrays", None)
    if not isinstance(specs, dict):
        raise ProbeError("KLP1 array table missing")
    data = memoryview(payload)[start + meta_size :]
    arrays: dict[str, np.ndarray] = {}
    covered = 0
    for name in sorted(specs):
        spec = specs[name]
        offset, nbytes = int(spec["offset"]), int(spec["nbytes"])
        dtype = np.dtype(spec["dtype"])
        shape = tuple(int(value) for value in spec["shape"])
        if offset != covered or math.prod(shape) * dtype.itemsize != nbytes:
            raise ProbeError("KLP1 array extent is noncanonical")
        if offset + nbytes > len(data):
            raise ProbeError("KLP1 array exceeds payload")
        arrays[name] = np.frombuffer(data[offset : offset + nbytes], dtype=dtype).reshape(shape).copy()
        covered += nbytes
    if covered != len(data):
        raise ProbeError("KLP1 has trailing bytes")
    return ProgramState(metadata, arrays)


class _BitWriter:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.byte = 0
        self.nbits = 0

    def bit(self, value: int) -> None:
        self.byte = (self.byte << 1) | (value & 1)
        self.nbits += 1
        if self.nbits == 8:
            self.buf.append(self.byte)
            self.byte = 0
            self.nbits = 0

    def bits(self, value: int, count: int) -> None:
        for shift in range(count - 1, -1, -1):
            self.bit(value >> shift)

    def finish(self) -> bytes:
        if self.nbits:
            self.buf.append(self.byte << (8 - self.nbits))
        return bytes(self.buf)


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def bit(self) -> int:
        if self.offset >= 8 * len(self.payload):
            raise ProbeError("truncated Rice bitstream")
        value = (self.payload[self.offset // 8] >> (7 - self.offset % 8)) & 1
        self.offset += 1
        return value

    def bits(self, count: int) -> int:
        value = 0
        for _ in range(count):
            value = (value << 1) | self.bit()
        return value


def _zigzag(value: int) -> int:
    return 2 * value if value >= 0 else -2 * value - 1


def _unzigzag(value: int) -> int:
    return value // 2 if value % 2 == 0 else -(value // 2) - 1


def _rice_encode(values: np.ndarray) -> tuple[int, bytes]:
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    if flat.size == 0:
        return 0, b""
    delta = np.diff(np.concatenate((np.zeros(1, np.int64), flat)))
    zig = np.fromiter((_zigzag(int(value)) for value in delta), dtype=np.uint64)
    bit_cost = [int(np.sum(zig >> k, dtype=np.uint64)) + flat.size * (1 + k) for k in range(16)]
    k = int(np.argmin(bit_cost))
    writer = _BitWriter()
    mask = (1 << k) - 1
    for raw in zig:
        value = int(raw)
        for _ in range(value >> k):
            writer.bit(0)
        writer.bit(1)
        if k:
            writer.bits(value & mask, k)
    return k, writer.finish()


def _rice_decode(payload: bytes, *, count: int, k: int) -> np.ndarray:
    reader = _BitReader(payload)
    out = np.empty(count, dtype=np.int64)
    previous = 0
    for index in range(count):
        quotient = 0
        while reader.bit() == 0:
            quotient += 1
            if quotient > (1 << 31):
                raise ProbeError("invalid Rice quotient")
        remainder = reader.bits(k) if k else 0
        previous += _unzigzag((quotient << k) | remainder)
        out[index] = previous
    return out


def encode_rice_program(raw_program: bytes) -> bytes:
    program = parse_raw_program(raw_program)
    table = []
    streams = []
    offset = 0
    for name in sorted(program.arrays):
        array = program.arrays[name]
        k, stream = _rice_encode(array)
        table.append(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "count": array.size,
                "k": k,
                "offset": offset,
                "nbytes": len(stream),
            }
        )
        streams.append(stream)
        offset += len(stream)
    metadata = {
        "program_metadata": program.metadata,
        "array_table": table,
        "raw_sha256": _sha256(raw_program),
        "raw_bytes": len(raw_program),
    }
    meta_bytes = rfc8785_canonicalize(metadata)
    return RICE_HEADER.pack(RICE_MAGIC, len(meta_bytes)) + meta_bytes + b"".join(streams)


def decode_rice_program(payload: bytes) -> bytes:
    if len(payload) < RICE_HEADER.size:
        raise ProbeError("truncated KLR1 program")
    magic, size = RICE_HEADER.unpack_from(payload)
    if magic != RICE_MAGIC or size > len(payload) - RICE_HEADER.size:
        raise ProbeError("invalid KLR1 header")
    start = RICE_HEADER.size
    metadata = json.loads(payload[start : start + size])
    data = payload[start + size :]
    arrays: dict[str, np.ndarray] = {}
    covered = 0
    for row in metadata["array_table"]:
        offset, nbytes = int(row["offset"]), int(row["nbytes"])
        if offset != covered or offset + nbytes > len(data):
            raise ProbeError("noncanonical KLR1 stream extent")
        decoded = _rice_decode(
            data[offset : offset + nbytes],
            count=int(row["count"]),
            k=int(row["k"]),
        )
        dtype = np.dtype(row["dtype"])
        info = np.iinfo(dtype)
        if decoded.size and (decoded.min() < info.min or decoded.max() > info.max):
            raise ProbeError("Rice value exceeds declared dtype")
        arrays[row["name"]] = decoded.astype(dtype).reshape(row["shape"])
        covered += nbytes
    if covered != len(data):
        raise ProbeError("KLR1 has trailing bytes")
    raw = encode_raw_program(ProgramState(metadata["program_metadata"], arrays))
    if len(raw) != metadata["raw_bytes"] or _sha256(raw) != metadata["raw_sha256"]:
        raise ProbeError("KLR1 parse-back identity failed")
    return raw


def pack_program(raw_program: bytes, codec: str) -> bytes:
    if codec == "brotli_q11":
        body = brotli.compress(raw_program, quality=11)
    elif codec == "lzma_xz_preset9_extreme":
        body = lzma.compress(
            raw_program,
            format=lzma.FORMAT_XZ,
            preset=9 | lzma.PRESET_EXTREME,
        )
    elif codec == "split_metadata_plus_rice_golomb":
        body = encode_rice_program(raw_program)
    else:
        raise ProbeError(f"unknown coder: {codec}")
    return ENVELOPE_HEADER.pack(ENVELOPE_MAGIC, CODEC_IDS[codec], len(raw_program), len(body)) + body


def unpack_program(payload: bytes) -> bytes:
    if len(payload) < ENVELOPE_HEADER.size:
        raise ProbeError("truncated KLC1 envelope")
    magic, codec_id, raw_size, body_size = ENVELOPE_HEADER.unpack_from(payload)
    if magic != ENVELOPE_MAGIC or body_size != len(payload) - ENVELOPE_HEADER.size:
        raise ProbeError("invalid KLC1 envelope")
    body = payload[ENVELOPE_HEADER.size :]
    codec = CODEC_NAMES.get(codec_id)
    if codec == "brotli_q11":
        raw = brotli.decompress(body)
    elif codec == "lzma_xz_preset9_extreme":
        raw = lzma.decompress(body, format=lzma.FORMAT_XZ)
    elif codec == "split_metadata_plus_rice_golomb":
        raw = decode_rice_program(body)
    else:
        raise ProbeError("unknown KLC1 codec id")
    if len(raw) != raw_size:
        raise ProbeError("KLC1 raw length differs")
    parse_raw_program(raw)
    return raw


def coder_race(raw_program: bytes, coders: Sequence[str]) -> tuple[bytes, dict[str, Any]]:
    rows = []
    winners = []
    for coder in coders:
        packed = pack_program(raw_program, coder)
        first = unpack_program(packed)
        second = unpack_program(packed)
        if first != raw_program or second != raw_program:
            raise ProbeError(f"{coder} parse-back differs")
        row = {
            "coder": coder,
            "bytes": len(packed),
            "sha256": _sha256(packed),
            "raw_bytes": len(raw_program),
            "raw_sha256": _sha256(raw_program),
            "double_parseback_identity": True,
        }
        rows.append(row)
        winners.append((len(packed), coder, packed))
    _, coder, payload = min(winners, key=lambda row: (row[0], row[1]))
    return payload, {"rows": rows, "selected_coder": coder, "selected_bytes": len(payload)}


def _metric_from_json(value: Mapping[str, Any]) -> MetricSpec:
    return MetricSpec(
        str(value["mode"]),
        float(value["row_scale"]),
        float(value["col_scale"]),
        float(value["horizon_row"]),
        float(value["depth_alpha"]),
    )


def transform_points(points: np.ndarray, metric: MetricSpec, shape: tuple[int, int]) -> np.ndarray:
    raw = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    out = raw.copy()
    if metric.mode == "isotropic_power_control":
        return out
    if metric.mode == "shared_chart_anisotropic_spd":
        out[:, 0] *= np.float32(metric.row_scale)
        out[:, 1] *= np.float32(metric.col_scale)
        return out
    if metric.mode != "projective_depth_stratified":
        raise ProbeError(f"unknown metric mode: {metric.mode}")
    h, _w = shape
    horizon = np.float32(metric.horizon_row)
    span = np.maximum(horizon, np.float32(h - 1) - horizon)
    normalized = (out[:, 0] - horizon) / max(float(span), 1.0)
    mapped = (
        np.sign(normalized)
        * np.log1p(np.float32(metric.depth_alpha) * np.abs(normalized))
        / np.log1p(np.float32(metric.depth_alpha))
    )
    out[:, 0] = horizon + mapped * span
    return out


def power_assign_numpy_fp32(
    sites: np.ndarray,
    class_weights: np.ndarray,
    site_classes: np.ndarray,
    pixels: np.ndarray,
    metric: MetricSpec,
    shape: tuple[int, int],
) -> np.ndarray:
    """Bounded brute-force authority kernel."""

    transformed_sites = transform_points(sites, metric, shape).astype(np.float32)
    transformed_pixels = transform_points(pixels, metric, shape).astype(np.float32)
    best = np.full(transformed_pixels.shape[0], np.float32(np.inf), np.float32)
    classes = np.zeros(transformed_pixels.shape[0], np.uint8)
    for class_id in range(N_CLASSES):
        selected = transformed_sites[np.asarray(site_classes) == class_id]
        if not len(selected):
            continue
        delta = transformed_pixels[:, None, :] - selected[None, :, :]
        distance = np.min(np.sum(delta * delta, axis=2, dtype=np.float32), axis=1)
        power = distance - np.float32(class_weights[class_id])
        update = power < best
        best[update] = power[update]
        classes[update] = class_id
    return classes


def power_assign_ckdtree(
    sites: np.ndarray,
    class_weights: np.ndarray,
    site_classes: np.ndarray,
    pixels: np.ndarray,
    metric: MetricSpec,
    shape: tuple[int, int],
) -> np.ndarray:
    """Chunk-scalable kernel, bit-identical to the fp32 authority at strict non-ties."""

    transformed_sites = transform_points(sites, metric, shape)
    transformed_pixels = transform_points(pixels, metric, shape)
    best = np.full(transformed_pixels.shape[0], np.inf, np.float64)
    classes = np.zeros(transformed_pixels.shape[0], np.uint8)
    for class_id in range(N_CLASSES):
        selected = transformed_sites[np.asarray(site_classes) == class_id]
        if not len(selected):
            continue
        distance, _ = cKDTree(selected).query(transformed_pixels, k=1, workers=1)
        power = distance * distance - float(class_weights[class_id])
        update = power < best
        best[update] = power[update]
        classes[update] = class_id
    return classes


def _pixel_grid(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = np.indices(shape, dtype=np.float32)
    return np.column_stack((rows.reshape(-1), cols.reshape(-1)))


def _radical_inverse_base2(index: int) -> float:
    value = 0.0
    denominator = 1.0
    while index:
        denominator *= 2.0
        value += (index & 1) / denominator
        index >>= 1
    return value


def _spread_indices(count: int, requested: int) -> np.ndarray:
    if count <= 0 or requested <= 0:
        return np.zeros(0, np.int64)
    fractions = np.fromiter(
        (_radical_inverse_base2(index + 1) for index in range(requested)),
        dtype=np.float64,
    )
    return np.minimum((fractions * count).astype(np.int64), count - 1)


def _part1by1(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.uint32) & np.uint32(0x0000FFFF)
    value = (value | (value << 8)) & np.uint32(0x00FF00FF)
    value = (value | (value << 4)) & np.uint32(0x0F0F0F0F)
    value = (value | (value << 2)) & np.uint32(0x33333333)
    return (value | (value << 1)) & np.uint32(0x55555555)


def _morton_sorted_pixels(mask: np.ndarray) -> np.ndarray:
    flat = np.flatnonzero(mask)
    if not flat.size:
        return flat
    width = mask.shape[1]
    rows = (flat // width).astype(np.uint32)
    cols = (flat % width).astype(np.uint32)
    morton = _part1by1(cols) | (_part1by1(rows) << np.uint32(1))
    return flat[np.argsort(morton, kind="stable")]


def derive_site_schedule(labels: np.ndarray, max_k: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Boundary/area-derived weighted fair class schedule."""

    area = np.zeros(N_CLASSES, np.float64)
    boundary = np.zeros(N_CLASSES, np.float64)
    for frame in labels:
        area += np.bincount(frame.reshape(-1), minlength=N_CLASSES)
        edges = np.zeros(frame.shape, bool)
        edges[:, 1:] |= frame[:, 1:] != frame[:, :-1]
        edges[:, :-1] |= frame[:, 1:] != frame[:, :-1]
        edges[1:, :] |= frame[1:, :] != frame[:-1, :]
        edges[:-1, :] |= frame[1:, :] != frame[:-1, :]
        boundary += np.bincount(frame[edges], minlength=N_CLASSES)
    score = np.sqrt(area / max(1.0, area.sum())) + np.sqrt(boundary / max(1.0, boundary.sum()))
    proportions = score / score.sum()
    schedule = list(range(N_CLASSES))
    counts = np.ones(N_CLASSES, np.int64)
    while len(schedule) < max_k:
        index = len(schedule) + 1
        class_id = int(np.argmax(proportions * index - counts))
        schedule.append(class_id)
        counts[class_id] += 1
    return np.asarray(schedule, np.uint8), {
        "area_counts": area.astype(np.int64).tolist(),
        "boundary_counts": boundary.astype(np.int64).tolist(),
        "site_proportions": proportions.tolist(),
        "max_k_class_counts": counts.tolist(),
    }


def extract_site_states(labels: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    """Nested per-class Morton/van-der-Corput site states."""

    n, _h, width = labels.shape
    max_counts = np.bincount(schedule, minlength=N_CLASSES)
    states = np.empty((n, len(schedule), 2), np.float32)
    prior = np.zeros((N_CLASSES, int(max_counts.max()), 2), np.float32)
    for frame_index, frame in enumerate(labels):
        selected: dict[int, np.ndarray] = {}
        for class_id in range(N_CLASSES):
            count = int(max_counts[class_id])
            ordered = _morton_sorted_pixels(frame == class_id)
            if ordered.size:
                flat = ordered[_spread_indices(len(ordered), count)]
                points = np.column_stack((flat // width, flat % width)).astype(np.float32)
                prior[class_id, :count] = points
            elif frame_index == 0:
                prior[class_id, :count] = np.array([frame.shape[0] / 2.0, frame.shape[1] / 2.0], np.float32)
            selected[class_id] = prior[class_id, :count].copy()
        offsets = np.zeros(N_CLASSES, np.int64)
        for site_index, class_id_raw in enumerate(schedule):
            class_id = int(class_id_raw)
            states[frame_index, site_index] = selected[class_id][offsets[class_id]]
            offsets[class_id] += 1
    return states


def derive_metrics(labels: np.ndarray) -> tuple[MetricSpec, ...]:
    rows = []
    cols = []
    horizons = []
    for frame in labels[:: max(1, len(labels) // 16)]:
        edges = np.zeros(frame.shape, bool)
        edges[:, 1:] |= frame[:, 1:] != frame[:, :-1]
        edges[:, :-1] |= frame[:, 1:] != frame[:, :-1]
        edges[1:, :] |= frame[1:, :] != frame[:-1, :]
        edges[:-1, :] |= frame[1:, :] != frame[:-1, :]
        rr, cc = np.nonzero(edges)
        if rr.size:
            rows.append(rr[:: _max_stride(len(rr), 8192)])
            cols.append(cc[:: _max_stride(len(cc), 8192)])
        road = np.argwhere((frame == 0) | (frame == 2))
        if road.size:
            horizons.append(float(np.quantile(road[:, 0], 0.05)))
    row = np.concatenate(rows).astype(np.float64)
    col = np.concatenate(cols).astype(np.float64)
    row_std = max(float(np.std(row)), 1.0)
    col_std = max(float(np.std(col)), 1.0)
    ratio = math.sqrt(col_std / row_std)
    row_scale = float(np.clip(ratio, 0.5, 2.0))
    col_scale = 1.0 / row_scale
    horizon = float(np.median(horizons)) if horizons else labels.shape[1] / 2
    return (
        MetricSpec("isotropic_power_control"),
        MetricSpec("shared_chart_anisotropic_spd", row_scale, col_scale),
        MetricSpec("projective_depth_stratified", horizon_row=horizon),
    )


def _max_stride(count: int, maximum: int) -> int:
    return max(1, math.ceil(count / maximum))


def derive_class_weights(
    states: np.ndarray,
    site_classes: np.ndarray,
    metric: MetricSpec,
    shape: tuple[int, int],
) -> np.ndarray:
    weights = np.zeros((len(states), N_CLASSES), np.float32)
    for frame_index, sites in enumerate(states):
        transformed = transform_points(sites, metric, shape)
        for class_id in range(N_CLASSES):
            selected = transformed[site_classes == class_id]
            if len(selected) >= 2:
                distance, _ = cKDTree(selected).query(selected, k=2, workers=1)
                weights[frame_index, class_id] = np.float32(0.25 * np.median(distance[:, 1] ** 2))
    return weights


def derive_palette(labels: np.ndarray, gt_f1: np.ndarray) -> np.ndarray:
    """Counted scorer-free class palette from deterministic sparse source samples."""

    palette = np.zeros((N_CLASSES, 3), np.uint8)
    h, w = labels.shape[1:]
    camera_h, camera_w = gt_f1.shape[1:3]
    sample_frames = np.linspace(0, len(labels) - 1, min(16, len(labels)), dtype=int)
    for class_id in range(N_CLASSES):
        samples = []
        for frame_index in sample_frames:
            rr, cc = np.nonzero(labels[frame_index] == class_id)
            if not rr.size:
                continue
            stride = _max_stride(len(rr), 2048)
            source_r = np.minimum(
                np.rint(rr[::stride] * (camera_h - 1) / max(1, h - 1)).astype(int),
                camera_h - 1,
            )
            source_c = np.minimum(
                np.rint(cc[::stride] * (camera_w - 1) / max(1, w - 1)).astype(int),
                camera_w - 1,
            )
            samples.append(np.asarray(gt_f1[frame_index, source_r, source_c]))
        if not samples:
            raise ProbeError(f"palette class absent: {class_id}")
        palette[class_id] = np.median(np.concatenate(samples), axis=0).astype(np.uint8)
    return palette


def _segment_bounds(n_frames: int, segment_count: int) -> np.ndarray:
    bounds = np.rint(np.linspace(0, n_frames, segment_count + 1)).astype(np.int64)
    bounds[0], bounds[-1] = 0, n_frames
    if np.any(np.diff(bounds) <= 0):
        raise ProbeError("temporal segments are not strictly increasing")
    return bounds


def _poly_design(count: int, degree: int) -> np.ndarray:
    u = np.zeros(count, np.float64) if count <= 1 else np.linspace(-1.0, 1.0, count)
    return np.column_stack([u**power for power in range(degree + 1)])


def _normalize_xi(xi: np.ndarray) -> tuple[np.ndarray, list[float], list[float]]:
    value = np.asarray(xi, np.float64)
    mean = value.mean(axis=0)
    scale = value.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (value - mean) / scale, mean.tolist(), scale.tolist()


def _quantize_checked(values: np.ndarray, scale: float, dtype: str, name: str) -> np.ndarray:
    scaled = np.rint(np.asarray(values, np.float64) * scale)
    if not np.all(np.isfinite(scaled)):
        raise ProbeError(f"{name} quantization received non-finite values")
    info = np.iinfo(np.dtype(dtype))
    observed_min = float(scaled.min(initial=0.0))
    observed_max = float(scaled.max(initial=0.0))
    if observed_min < info.min or observed_max > info.max:
        raise ProbeError(
            f"{name} quantization exceeds {dtype}: [{observed_min}, {observed_max}] "
            f"outside [{info.min}, {info.max}]"
        )
    return scaled.astype(dtype)


def _stable_pose_advection(xi_segment: np.ndarray, mean_target: np.ndarray) -> np.ndarray:
    """Centered, ridge-stabilized pose pullback for short temporal segments."""

    x = np.asarray(xi_segment, np.float64)
    y = np.asarray(mean_target, np.float64)
    if len(x) < 2:
        return np.zeros((x.shape[1], y.shape[1]), np.float64)
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    gram = x_centered.T @ x_centered
    gram.flat[:: gram.shape[0] + 1] += POSE_RIDGE_LAMBDA
    advection = np.linalg.solve(gram, x_centered.T @ y_centered)
    if not np.all(np.isfinite(advection)):
        raise ProbeError("pose advection regression produced non-finite coefficients")
    return advection


def _regular_triangulation_edges(
    sites: np.ndarray,
    class_weights: np.ndarray,
    site_classes: np.ndarray,
    metric: MetricSpec,
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    transformed = transform_points(sites, metric, shape).astype(np.float64)
    weights = np.asarray(class_weights, np.float64)[site_classes]
    lifted = np.column_stack((transformed, np.sum(transformed * transformed, axis=1) - weights))
    try:
        hull = ConvexHull(lifted)
    except QhullError:
        # Qhull's QJ option adds process-external random jitter, which is
        # incompatible with the bit-identical program contract.  Break exact
        # lifted-point degeneracies with a deterministic, index-keyed
        # perturbation instead.
        perturbed = lifted.copy()
        perturbed[:, 2] += np.arange(len(perturbed), dtype=np.float64) * 1e-10
        try:
            hull = ConvexHull(perturbed)
        except QhullError:
            return set()
    edges: set[tuple[int, int]] = set()
    for simplex, equation in zip(hull.simplices, hull.equations, strict=True):
        if equation[2] >= -1e-10:
            continue
        for i in range(3):
            a, b = sorted((int(simplex[i]), int(simplex[(i + 1) % 3])))
            edges.add((a, b))
    return edges


def _event_rows(
    sites: np.ndarray,
    weights: np.ndarray,
    site_classes: np.ndarray,
    metric: MetricSpec,
    shape: tuple[int, int],
    sample_frames: Sequence[int],
) -> np.ndarray:
    rows = []
    prior: set[tuple[int, int]] | None = None
    for frame_index in sample_frames:
        current = _regular_triangulation_edges(
            sites[frame_index],
            weights[frame_index],
            site_classes,
            metric,
            shape,
        )
        if prior is not None:
            for a, b in sorted(prior - current):
                rows.append((frame_index, 0, a, b))
            for a, b in sorted(current - prior):
                rows.append((frame_index, 1, a, b))
        prior = current
    return np.asarray(rows, dtype="<i2").reshape(-1, 4)


def _fit_independent_program(
    sites: np.ndarray,
    weights: np.ndarray,
    site_classes: np.ndarray,
    metric: MetricSpec,
    palette: np.ndarray,
    *,
    degree: int,
    pair_ids: Sequence[int],
) -> ProgramState:
    arrays = {
        "class_weights_q": _quantize_checked(weights, WEIGHT_Q, "<i4", "class_weights"),
        "event_rows": np.zeros((0, 4), "<i2"),
        "palette_rgb": np.asarray(palette, "<u1"),
        "pair_ids": np.asarray(pair_ids, "<u2"),
        "site_classes": np.asarray(site_classes, "<u1"),
        "site_states_q": _quantize_checked(sites, COORD_Q, "<i2", "site_states"),
    }
    return ProgramState(
        {
            "schema": "kinetic_laguerre_program.v1",
            "temporal_mode": "independent_frame_control",
            "degree_cell_label": degree,
            "metric": metric.as_json(),
            "coord_q": COORD_Q,
            "weight_q": WEIGHT_Q,
            "shared_edge_accounting": "once",
            "xi_driven": False,
            "event_semantics": "per-frame states; topology derived, no event sidecar",
        },
        arrays,
    )


def _fit_kinetic_program(
    sites: np.ndarray,
    weights: np.ndarray,
    xi: np.ndarray,
    site_classes: np.ndarray,
    metric: MetricSpec,
    palette: np.ndarray,
    *,
    degree: int,
    segment_count: int,
    pair_ids: Sequence[int],
    image_shape: tuple[int, int],
    include_events: bool,
) -> ProgramState:
    n_frames, k, _ = sites.shape
    bounds = _segment_bounds(n_frames, segment_count)
    xi_norm, xi_mean, xi_scale = _normalize_xi(xi)
    site_coef = np.zeros((segment_count, k, degree + 1, 2), np.float64)
    weight_coef = np.zeros((segment_count, N_CLASSES, degree + 1), np.float64)
    xi_advection = np.zeros((segment_count, N_CLASSES, 6, 2), np.float64)
    for segment in range(segment_count):
        start, stop = int(bounds[segment]), int(bounds[segment + 1])
        design = _poly_design(stop - start, degree)
        xi_segment = xi_norm[start:stop]
        for class_id in range(N_CLASSES):
            indices = np.flatnonzero(site_classes == class_id)
            target = sites[start:stop, indices]
            mean_target = target.mean(axis=1)
            if len(xi_segment) >= 2:
                advection = _stable_pose_advection(xi_segment, mean_target)
                xi_advection[segment, class_id] = advection
                target = target - (xi_segment @ advection)[:, None, :]
            flattened = target.reshape(stop - start, -1)
            coefficients = np.linalg.lstsq(design, flattened, rcond=None)[0]
            site_coef[segment, indices] = coefficients.reshape(degree + 1, len(indices), 2).transpose(1, 0, 2)
        weight_coef[segment] = np.linalg.lstsq(design, weights[start:stop], rcond=None)[0].T
    arrays = {
        "class_weights_coef_q": _quantize_checked(
            weight_coef,
            WEIGHT_Q,
            "<i4",
            "class_weights_coefficients",
        ),
        "event_rows": np.zeros((0, 4), "<i2"),
        "palette_rgb": np.asarray(palette, "<u1"),
        "pair_ids": np.asarray(pair_ids, "<u2"),
        "segment_bounds": bounds.astype("<u2"),
        "site_classes": np.asarray(site_classes, "<u1"),
        "site_coefficients_q": _quantize_checked(
            site_coef,
            COORD_Q,
            "<i2",
            "site_coefficients",
        ),
        "xi_advection_q": _quantize_checked(
            xi_advection,
            COORD_Q,
            "<i2",
            "xi_advection",
        ),
        "xi_q": _quantize_checked(xi_norm, XI_Q, "<i2", "xi"),
    }
    metadata = {
        "schema": "kinetic_laguerre_program.v1",
        "temporal_mode": "spline_sites_weights_plus_sparse_regular_triangulation_flips",
        "degree": degree,
        "segment_count": segment_count,
        "metric": metric.as_json(),
        "coord_q": COORD_Q,
        "weight_q": WEIGHT_Q,
        "xi_q": XI_Q,
        "xi_mean": xi_mean,
        "xi_scale": xi_scale,
        "pose_advection_regression": {
            "solver": "centered_tikhonov",
            "lambda": POSE_RIDGE_LAMBDA,
        },
        "shared_edge_accounting": "once",
        "xi_driven": True,
        "event_semantics": "regular-triangulation edge deletion/addition at segment starts",
    }
    program = ProgramState(metadata, arrays)
    if include_events:
        decoded = decode_program(program)
        event_rows = _event_rows(
            decoded.sites,
            decoded.class_weights,
            decoded.site_classes,
            metric,
            image_shape,
            bounds[:-1],
        )
        arrays = {**arrays, "event_rows": event_rows}
        metadata = {
            **metadata,
            "regular_triangulation_event_count": len(event_rows),
        }
        program = ProgramState(metadata, arrays)
    return program


def decode_program(program: ProgramState) -> DecodedProgram:
    metadata, arrays = program.metadata, program.arrays
    site_classes = np.asarray(arrays["site_classes"], np.uint8)
    palette = np.asarray(arrays["palette_rgb"], np.uint8)
    if metadata["temporal_mode"] == "independent_frame_control":
        sites = np.asarray(arrays["site_states_q"], np.float32) / COORD_Q
        weights = np.asarray(arrays["class_weights_q"], np.float32) / WEIGHT_Q
    else:
        bounds = np.asarray(arrays["segment_bounds"], np.int64)
        coefficient = np.asarray(arrays["site_coefficients_q"], np.float32) / COORD_Q
        weight_coef = np.asarray(arrays["class_weights_coef_q"], np.float32) / WEIGHT_Q
        xi = np.asarray(arrays["xi_q"], np.float32) / XI_Q
        advection = np.asarray(arrays["xi_advection_q"], np.float32) / COORD_Q
        n_frames = len(xi)
        sites = np.zeros((n_frames, len(site_classes), 2), np.float32)
        weights = np.zeros((n_frames, N_CLASSES), np.float32)
        degree = int(metadata["degree"])
        for segment in range(len(bounds) - 1):
            start, stop = int(bounds[segment]), int(bounds[segment + 1])
            design = _poly_design(stop - start, degree).astype(np.float32)
            sites[start:stop] = np.einsum("fd,kdc->fkc", design, coefficient[segment], optimize=True)
            weights[start:stop] = np.einsum("fd,cd->fc", design, weight_coef[segment], optimize=True)
            for class_id in range(N_CLASSES):
                sites[start:stop, site_classes == class_id] += (xi[start:stop] @ advection[segment, class_id])[
                    :, None, :
                ]
    return DecodedProgram(
        metadata,
        site_classes,
        sites,
        weights,
        palette,
        np.asarray(arrays["event_rows"], np.int16),
    )


def decode_envelope(payload: bytes) -> DecodedProgram:
    return decode_program(parse_raw_program(unpack_program(payload)))


def _prediction_state_sha256(decoded: DecodedProgram) -> str:
    digest = hashlib.sha256()
    for array in (
        decoded.site_classes,
        decoded.sites,
        decoded.class_weights,
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    digest.update(rfc8785_canonicalize(decoded.metadata["metric"]))
    return digest.hexdigest()


def _program_parseback_receipt(payload: bytes) -> tuple[DecodedProgram, dict[str, Any]]:
    first_raw = unpack_program(payload)
    second_raw = unpack_program(payload)
    if first_raw != second_raw:
        raise ProbeError("program double parse-back raw bytes differ")
    first = decode_program(parse_raw_program(first_raw))
    second = decode_program(parse_raw_program(second_raw))
    for name in ("site_classes", "sites", "class_weights", "palette_rgb", "event_rows"):
        if not np.array_equal(getattr(first, name), getattr(second, name)):
            raise ProbeError(f"program double decode differs: {name}")
    return first, {
        "program_bytes": len(payload),
        "program_sha256": _sha256(payload),
        "raw_bytes": len(first_raw),
        "raw_sha256": _sha256(first_raw),
        "prediction_state_sha256": _prediction_state_sha256(first),
        "double_decode_identity": True,
    }


def _program_kernel_parity(decoded: DecodedProgram, image_shape: tuple[int, int]) -> dict[str, Any]:
    pixels = _pixel_grid(image_shape)
    selected_pixels = pixels[np.linspace(0, len(pixels) - 1, min(512, len(pixels)), dtype=np.int64)]
    frame_ids = sorted({0, len(decoded.sites) // 2, len(decoded.sites) - 1})
    metric = _metric_from_json(decoded.metadata["metric"])
    rows = []
    for frame_index in frame_ids:
        authority = power_assign_numpy_fp32(
            decoded.sites[frame_index],
            decoded.class_weights[frame_index],
            decoded.site_classes,
            selected_pixels,
            metric,
            image_shape,
        )
        scalable = power_assign_ckdtree(
            decoded.sites[frame_index],
            decoded.class_weights[frame_index],
            decoded.site_classes,
            selected_pixels,
            metric,
            image_shape,
        )
        mismatches = int(np.count_nonzero(authority != scalable))
        if mismatches:
            raise ProbeError(
                f"real program kernel parity differs at frame {frame_index}: {mismatches}/{len(selected_pixels)}"
            )
        rows.append(
            {
                "frame_index": frame_index,
                "sample_sites": len(selected_pixels),
                "mismatch_count": 0,
                "cells_sha256": _sha256(authority.tobytes()),
            }
        )
    return {
        "authority": "numpy_fp32_bruteforce",
        "scalable_kernel": "scipy_ckdtree",
        "sampled_real_program_rows": rows,
        "bit_identical": True,
    }


def _fit_pack_kinetic_waterfill(
    *,
    sites: np.ndarray,
    weights: np.ndarray,
    xi: np.ndarray,
    site_classes: np.ndarray,
    metric: MetricSpec,
    palette: np.ndarray,
    degree: int,
    pair_ids: Sequence[int],
    image_shape: tuple[int, int],
    home_bytes: int,
    coders: Sequence[str],
) -> tuple[bytes, dict[str, Any]]:
    """Largest deterministic temporal segment count admitted by real bytes."""

    n_frames = len(sites)
    minimum_frames_per_segment = max(8, degree + 2)
    maximum_segment_count = max(1, n_frames // minimum_frames_per_segment)
    cache: dict[int, tuple[bytes, dict[str, Any]]] = {}

    def compile_count(segment_count: int, *, events: bool) -> tuple[bytes, dict[str, Any]]:
        key = segment_count if not events else -segment_count
        if key in cache:
            return cache[key]
        program = _fit_kinetic_program(
            sites,
            weights,
            xi,
            site_classes,
            metric,
            palette,
            degree=degree,
            segment_count=segment_count,
            pair_ids=pair_ids,
            image_shape=image_shape,
            include_events=events,
        )
        raw = encode_raw_program(program)
        payload, race = coder_race(raw, coders)
        cache[key] = (
            payload,
            {
                "segment_count": segment_count,
                "events_included": events,
                "coder_race": race,
            },
        )
        return cache[key]

    trace = []
    low = 1
    payload, row = compile_count(low, events=False)
    trace.append({**row, "selected_bytes": len(payload)})
    if len(payload) > home_bytes:
        selected_count = 1
    else:
        high = 2
        while high <= maximum_segment_count:
            payload, row = compile_count(high, events=False)
            trace.append({**row, "selected_bytes": len(payload)})
            if len(payload) > home_bytes:
                break
            low = high
            high *= 2
        high = min(high, maximum_segment_count)
        if high == maximum_segment_count and low < high:
            payload, row = compile_count(high, events=False)
            trace.append({**row, "selected_bytes": len(payload)})
            if len(payload) <= home_bytes:
                low = high
        if low < high:
            while high - low > 1:
                middle = (low + high) // 2
                payload, row = compile_count(middle, events=False)
                trace.append({**row, "selected_bytes": len(payload)})
                if len(payload) <= home_bytes:
                    low = middle
                else:
                    high = middle
        selected_count = low

    # Events are a charged certificate.  If they cross the home, reduce the
    # temporal segmentation until the complete stream is admitted.
    while True:
        payload, complete = compile_count(selected_count, events=True)
        trace.append({**complete, "selected_bytes": len(payload)})
        if len(payload) <= home_bytes or selected_count == 1:
            break
        selected_count = max(1, math.floor(selected_count * 0.8))
    return payload, {
        "waterfill_trace": sorted(
            trace,
            key=lambda item: (
                int(item["segment_count"]),
                bool(item["events_included"]),
            ),
        ),
        "selected_segment_count": selected_count,
        "maximum_segment_count": maximum_segment_count,
        "minimum_frames_per_segment": minimum_frames_per_segment,
        "pose_advection_regression": {
            "solver": "centered_tikhonov",
            "lambda": POSE_RIDGE_LAMBDA,
        },
        "complete_program_bytes": len(payload),
        "within_predictor_home": len(payload) <= home_bytes,
    }


def _kernel_contract() -> dict[str, Any]:
    rng = np.random.default_rng(1234)
    shape = (13, 17)
    sites = np.column_stack((rng.uniform(0, shape[0], 12), rng.uniform(0, shape[1], 12))).astype(np.float32)
    classes = np.tile(np.arange(N_CLASSES, dtype=np.uint8), 3)[: len(sites)]
    weights = np.linspace(0.25, 3.25, N_CLASSES, dtype=np.float32)
    pixels = _pixel_grid(shape)
    rows = []
    for metric in (
        MetricSpec("isotropic_power_control"),
        MetricSpec("shared_chart_anisotropic_spd", 1.25, 0.8),
        MetricSpec("projective_depth_stratified", horizon_row=4.0),
    ):
        authority = power_assign_numpy_fp32(sites, weights, classes, pixels, metric, shape)
        scalable = power_assign_ckdtree(sites, weights, classes, pixels, metric, shape)
        if not np.array_equal(authority, scalable):
            raise ProbeError(f"kernel contract differs for {metric.mode}")
        rows.append(
            {
                "metric": metric.mode,
                "cells_sha256": _sha256(authority.tobytes()),
                "bit_identical": True,
            }
        )
    return {
        "authority": "numpy_fp32_bruteforce",
        "scalable_kernel": "scipy_ckdtree_float64_distance_strict_non_tie",
        "rows": rows,
        "all_bit_identical": True,
    }


def measure_label_program(
    decoded: DecodedProgram,
    labels: np.ndarray,
    *,
    stop_after_errors: int | None,
) -> dict[str, Any]:
    """Measure real cached-label errors, with an exact early lower bound."""

    if len(decoded.sites) != len(labels):
        raise ProbeError("program/label frame count differs")
    shape = tuple(int(value) for value in labels.shape[1:])
    pixels = _pixel_grid(shape)
    metric = _metric_from_json(decoded.metadata["metric"])
    total_errors = 0
    total_sites = 0
    class_errors = np.zeros(N_CLASSES, np.int64)
    class_sites = np.zeros(N_CLASSES, np.int64)
    digest = hashlib.sha256()
    started = time.monotonic()
    completed = True
    for frame_index in range(len(labels)):
        pred = power_assign_ckdtree(
            decoded.sites[frame_index],
            decoded.class_weights[frame_index],
            decoded.site_classes,
            pixels,
            metric,
            shape,
        ).reshape(shape)
        truth = np.asarray(labels[frame_index], np.uint8)
        errors = pred != truth
        digest.update(pred.tobytes())
        total_errors += int(np.count_nonzero(errors))
        total_sites += errors.size
        for class_id in range(N_CLASSES):
            mask = truth == class_id
            class_errors[class_id] += int(np.count_nonzero(errors & mask))
            class_sites[class_id] += int(np.count_nonzero(mask))
        if stop_after_errors is not None and total_errors > stop_after_errors:
            completed = False
            break
    evaluated_frames = frame_index + 1
    return {
        "status": "COMPLETE" if completed else "EXACT_LOWER_BOUND_EXCEEDS_MATCH_GATE",
        "evaluated_frames": evaluated_frames,
        "total_frames": len(labels),
        "complete": completed,
        "errors": total_errors if completed else None,
        "errors_lower_bound": total_errors,
        "sites": total_sites if completed else None,
        "sites_evaluated": total_sites,
        "d_seg": (f"{total_errors / total_sites:.12f}" if completed else None),
        "observed_prefix_error_fraction": f"{total_errors / total_sites:.12f}",
        "per_class": {
            CLASS_NAMES[class_id]: {
                "errors": int(class_errors[class_id]),
                "sites": int(class_sites[class_id]),
                "observed_error_fraction": (f"{class_errors[class_id] / max(1, class_sites[class_id]):.12f}"),
            }
            for class_id in range(N_CLASSES)
        },
        "cells_digest_sha256": digest.hexdigest(),
        "seconds": round(time.monotonic() - started, 3),
        "evidence": "MEASURED cached n600 argmax label-space" if len(labels) == 600 else "n64 compute-integrity only",
        "score_claim": False,
    }


def _cell_id(site_count: int, degree: int, metric: str, temporal: str) -> str:
    return f"k{site_count}_d{degree}_{metric}_{temporal}"


def _load_cell(path: Path, config_hash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(_read_regular(path))
    if value.get("typed_config_sha256") != config_hash:
        raise ProbeError(f"resumed cell config differs: {path}")
    return value


def _stage_a_rung(
    *,
    rung: Literal["n64", "n600"],
    config: DDMKineticLaguerreAtToleranceProbeV1,
    output_root: Path,
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    palette: np.ndarray,
) -> dict[str, Any]:
    aggregate_path = output_root / "stage_checkpoints" / f"stage_a_{rung}.json"
    resumed = _load_cell(aggregate_path, config.typed_config_hash())
    if resumed is not None:
        return resumed
    pair_ids = np.arange(448, 512, dtype=np.int64) if rung == "n64" else np.arange(600, dtype=np.int64)
    labels = np.asarray(labels_all[pair_ids], np.uint8)
    poses = np.asarray(poses_all[pair_ids], np.float64)
    schedule, schedule_receipt = derive_site_schedule(
        labels[:: max(1, len(labels) // 32)],
        max(config.representation.site_counts),
    )
    site_states = extract_site_states(labels, schedule)
    metrics = {metric.mode: metric for metric in derive_metrics(labels)}
    rows = []
    measurement_cache: dict[str, dict[str, Any]] = {}
    stop_after = math.ceil(V19B_MATCH_ERRORS * len(labels) / 600)
    for site_count in config.representation.site_counts:
        sites = site_states[:, :site_count]
        classes = schedule[:site_count]
        for metric_name in config.representation.metric_modes:
            metric = metrics[metric_name]
            weights = derive_class_weights(sites, classes, metric, labels.shape[1:])
            for degree in config.representation.trajectory_degrees:
                for temporal in config.representation.temporal_modes:
                    cell_id = _cell_id(site_count, degree, metric_name, temporal)
                    checkpoint = output_root / "stage_checkpoints" / rung / f"{cell_id}.json"
                    existing = _load_cell(checkpoint, config.typed_config_hash())
                    if existing is not None:
                        rows.append(existing)
                        continue
                    if temporal == "independent_frame_control":
                        program = _fit_independent_program(
                            sites,
                            weights,
                            classes,
                            metric,
                            palette,
                            degree=degree,
                            pair_ids=pair_ids,
                        )
                        raw = encode_raw_program(program)
                        payload, race = coder_race(raw, config.representation.real_coder_race)
                        waterfill = {
                            "waterfill_trace": [],
                            "selected_segment_count": len(labels),
                            "complete_program_bytes": len(payload),
                            "within_predictor_home": (len(payload) <= config.population.current_predictor_home_bytes),
                            "coder_race": race,
                        }
                    else:
                        payload, waterfill = _fit_pack_kinetic_waterfill(
                            sites=sites,
                            weights=weights,
                            xi=poses,
                            site_classes=classes,
                            metric=metric,
                            palette=palette,
                            degree=degree,
                            pair_ids=pair_ids,
                            image_shape=labels.shape[1:],
                            home_bytes=config.population.current_predictor_home_bytes,
                            coders=config.representation.real_coder_race,
                        )
                    decoded, parseback = _program_parseback_receipt(payload)
                    kernel_parity = _program_kernel_parity(decoded, labels.shape[1:])
                    state_key = parseback["prediction_state_sha256"]
                    measurement = measurement_cache.get(state_key)
                    reused = measurement is not None
                    if measurement is None:
                        measurement = measure_label_program(
                            decoded,
                            labels,
                            stop_after_errors=stop_after,
                        )
                        measurement_cache[state_key] = measurement
                    program_path = output_root / "programs" / rung / f"{cell_id}.klp"
                    _publish_immutable(program_path, payload)
                    errors = measurement["errors"]
                    within_error = bool(
                        measurement["complete"]
                        and errors is not None
                        and errors <= math.ceil(config.population.maximum_seg_errors * len(labels) / 600)
                    )
                    row = {
                        "schema": "ddm_kinetic_laguerre_stage_a_cell.v1",
                        "typed_config_sha256": config.typed_config_hash(),
                        "rung": rung,
                        "cell_id": cell_id,
                        "site_count": site_count,
                        "trajectory_degree": degree,
                        "metric_mode": metric_name,
                        "temporal_mode": temporal,
                        "metric": metric.as_json(),
                        "program": {
                            **parseback,
                            "path": _portable(program_path),
                            "selected_coder": CODEC_NAMES[ENVELOPE_HEADER.unpack_from(payload)[1]],
                        },
                        "kernel_parity": kernel_parity,
                        "waterfill": waterfill,
                        "measurement": measurement,
                        "measurement_reused_for_identical_decoded_state": reused,
                        "regular_triangulation_event_count": len(decoded.event_rows),
                        "within_error_gate": within_error,
                        "within_predictor_home": (len(payload) <= config.population.current_predictor_home_bytes),
                        "stage_a_admitted": bool(
                            within_error and len(payload) <= config.population.current_predictor_home_bytes
                        ),
                        "authority": ("N64_COMPUTE_INTEGRITY_ONLY" if rung == "n64" else "N600_LABEL_SPACE_MEASURED"),
                        "score_claim": False,
                    }
                    _write_json(checkpoint, row)
                    rows.append(row)
                    print(
                        f"[{rung}] {cell_id}: {len(payload)} B "
                        f"errors={measurement['errors'] if measurement['errors'] is not None else '>' + str(measurement['errors_lower_bound'])} "
                        f"frames={measurement['evaluated_frames']}/{measurement['total_frames']}",
                        flush=True,
                    )

    expected_cells = (
        len(config.representation.site_counts)
        * len(config.representation.trajectory_degrees)
        * len(config.representation.metric_modes)
        * len(config.representation.temporal_modes)
    )
    if len(rows) != expected_cells or len({row["cell_id"] for row in rows}) != expected_cells:
        raise ProbeError(f"{rung} Stage-A cell coverage differs")
    admitted = [row for row in rows if row["stage_a_admitted"]]
    matched = [
        row
        for row in rows
        if row["measurement"]["complete"]
        and int(row["measurement"]["errors"]) <= math.ceil(V19B_MATCH_ERRORS * len(labels) / 600)
    ]
    best_complete = min(
        (row for row in rows if row["measurement"]["complete"]),
        key=lambda row: (
            int(row["measurement"]["errors"]),
            int(row["program"]["program_bytes"]),
            row["cell_id"],
        ),
        default=None,
    )
    result = {
        "schema": "ddm_kinetic_laguerre_stage_a_rung.v1",
        "typed_config_sha256": config.typed_config_hash(),
        "rung": rung,
        "pair_ids": pair_ids.tolist(),
        "cell_count": len(rows),
        "expected_cell_count": expected_cells,
        "all_cells_dispositioned": True,
        "site_schedule": schedule_receipt,
        "metric_specs": [metric.as_json() for metric in metrics.values()],
        "rows": rows,
        "stage_a_winner_ids": [row["cell_id"] for row in admitted],
        "stage_a_winner_count": len(admitted),
        "matched_v19b_rows": [
            {
                "cell_id": row["cell_id"],
                "program_bytes": row["program"]["program_bytes"],
                "errors": row["measurement"]["errors"],
                "d_seg": row["measurement"]["d_seg"],
            }
            for row in sorted(
                matched,
                key=lambda row: (
                    row["program"]["program_bytes"],
                    row["measurement"]["errors"],
                ),
            )
        ],
        "best_complete_row": (
            {
                "cell_id": best_complete["cell_id"],
                "program_bytes": best_complete["program"]["program_bytes"],
                "errors": best_complete["measurement"]["errors"],
                "d_seg": best_complete["measurement"]["d_seg"],
            }
            if best_complete
            else None
        ),
        "verdict": (
            "N64_INTEGRITY_COMPLETE_NO_DECISION"
            if rung == "n64"
            else "STAGE_A_WINNER_EXISTS"
            if admitted
            else "NO_REGISTERED_CELL_REACHED_JOINT_ERROR_RATE_GATE"
        ),
        "score_claim": False,
        "evidence_axis": ("n64 compute-integrity only" if rung == "n64" else "[macOS-CPU cached-label advisory]"),
    }
    _write_json(aggregate_path, result)
    return result


def _receiver_container(base_archive: bytes, program: bytes, mode: str) -> bytes:
    metadata = rfc8785_canonicalize(
        {
            "schema": "ddm_kinetic_laguerre_probe_container.v1",
            "mode": mode,
            "base_archive_bytes": len(base_archive),
            "base_archive_sha256": _sha256(base_archive),
            "program_bytes": len(program),
            "program_sha256": _sha256(program),
            "not_a_candidate": True,
            "score_claim": False,
        }
    )
    return (
        b"KLA1"
        + struct.pack("<III", len(metadata), len(base_archive), len(program))
        + metadata
        + base_archive
        + program
    )


def _render_partition_frame(
    decoded: DecodedProgram,
    frame_index: int,
    output_shape: tuple[int, int],
) -> np.ndarray:
    shape = (384, 512)
    cells = power_assign_ckdtree(
        decoded.sites[frame_index],
        decoded.class_weights[frame_index],
        decoded.site_classes,
        _pixel_grid(shape),
        _metric_from_json(decoded.metadata["metric"]),
        shape,
    ).reshape(shape)
    source_r = np.minimum(
        np.rint(np.arange(output_shape[0]) * (shape[0] - 1) / max(1, output_shape[0] - 1)).astype(int),
        shape[0] - 1,
    )
    source_c = np.minimum(
        np.rint(np.arange(output_shape[1]) * (shape[1] - 1) / max(1, output_shape[1] - 1)).astype(int),
        shape[1] - 1,
    )
    return decoded.palette_rgb[cells[source_r[:, None], source_c[None, :]]]


def _receiver_measurement(
    *,
    decoded: DecodedProgram,
    program: bytes,
    base_archive: bytes,
    base_receiver: Any,
    correction_receiver: Any | None,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
    mode: str,
    output_root: Path,
) -> dict[str, Any]:
    from tools.measure_ddm_v14_realization_fidelity import _forward

    container = _receiver_container(base_archive, program, mode)
    container_path = output_root / "receiver_archives" / f"{mode}.kla.receipt-bytes"
    _publish_immutable(container_path, container)
    errors = 0
    pose_sse = 0.0
    sites = 0
    pose_coordinates = 0
    first_digest = hashlib.sha256()
    second_digest = hashlib.sha256()
    for start in range(0, 600, PAIR_BATCH):
        stop = min(start + PAIR_BATCH, 600)
        pair_ids = tuple(range(start, stop))
        base_camera = base_receiver.render_camera_pairs(pair_ids)
        camera = base_camera.copy()
        for local, frame_index in enumerate(range(start, stop)):
            camera[local, 1] = _render_partition_frame(decoded, frame_index, tuple(camera.shape[2:4]))
        if correction_receiver is not None:
            correction = correction_receiver.render_camera_pairs(pair_ids).astype(np.int16) - base_camera.astype(
                np.int16
            )
            camera = np.clip(camera.astype(np.int16) + correction, 0, 255).astype(np.uint8)
        cells, pose6 = _forward(segnet, posenet, camera)
        replay_cells, replay_pose6 = _forward(segnet, posenet, camera)
        if not np.array_equal(cells, replay_cells) or not np.array_equal(pose6, replay_pose6):
            raise ProbeError("receiver double scorer replay differs")
        first_digest.update(cells.tobytes())
        first_digest.update(pose6.tobytes())
        second_digest.update(replay_cells.tobytes())
        second_digest.update(replay_pose6.tobytes())
        errors += int(np.count_nonzero(cells != labels[start:stop]))
        sites += cells.size
        pose_sse += float(np.square(pose6 - poses[start:stop]).sum(dtype=np.float64))
        pose_coordinates += pose6.size
    d_seg = errors / sites
    d_pose = pose_sse / pose_coordinates
    return {
        "status": "MEASURED",
        "mode": mode,
        "archive": {
            "path": _portable(container_path),
            "bytes": len(container),
            "sha256": _sha256(container),
        },
        "errors": errors,
        "sites": sites,
        "d_seg": f"{d_seg:.12f}",
        "d_pose": f"{d_pose:.12f}",
        "advisory_score_formula_value": f"{100 * d_seg + math.sqrt(10 * d_pose) + 25 * len(container) / TARGET_SOURCE_BYTES:.12f}",
        "double_scorer_replay_identity": (first_digest.digest() == second_digest.digest()),
        "scorer_digest_sha256": first_digest.hexdigest(),
        "within_archive_gate": len(container) <= 200_000,
        "within_d_seg_gate": d_seg <= 0.001159998576,
        "within_d_pose_gate": d_pose <= 0.00161,
        "vehicle_admitted": bool(len(container) <= 200_000 and d_seg <= 0.001159998576 and d_pose <= 0.00161),
        "authority": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "not_a_candidate": True,
    }


def _stage_b_receiver(
    *,
    config: DDMKineticLaguerreAtToleranceProbeV1,
    output_root: Path,
    stage_a_n600: Mapping[str, Any],
) -> dict[str, Any]:
    path = output_root / "stage_checkpoints" / "stage_b_receiver.json"
    resumed = _load_cell(path, config.typed_config_hash())
    if resumed is not None:
        return resumed
    winners = [row for row in stage_a_n600["rows"] if row.get("stage_a_admitted") is True]
    if not winners:
        result = {
            "schema": "ddm_kinetic_laguerre_stage_b.v1",
            "typed_config_sha256": config.typed_config_hash(),
            "status": "NOT_RUN_STAGE_A_GATE_CLOSED",
            "kinetic_only": None,
            "composed_with_v19b_corrections": None,
            "secondary_falsifier_reached": False,
            "score_claim": False,
            "evidence_axis": AXIS,
        }
        _write_json(path, result)
        return result

    winner = min(
        winners,
        key=lambda row: (
            row["program"]["program_bytes"],
            row["measurement"]["errors"],
            row["cell_id"],
        ),
    )
    program = _read_regular(REPO_ROOT / winner["program"]["path"])
    decoded, _parseback = _program_parseback_receipt(program)

    # Reuse the exact v19/v19b model and receiver custody; this is a
    # conservative real container upper bound because the existing predictor
    # remains charged.  No replacement-byte saving is inferred from this row.
    from tac.optimization.direct_description_carrier_compose import (
        receive_carrier_compose_archive,
    )
    from tac.optimization.direct_description_preuint8_channel import (
        receive_preuint8_q8_archive,
    )
    from tools.measure_ddm_v19b_joint_remeasure_stack import (
        DDMV19BJointRemeasureStackConfigV1,
        _load_sources,
    )

    v19b_receipt = json.loads(_read_regular(REPO_ROOT / config.inputs.v19b_receipt_path))
    v19b_config = DDMV19BJointRemeasureStackConfigV1.model_validate(v19b_receipt["typed_config"])
    _v19_config, _v19_receipt, context = _load_sources(v19b_config)
    v19b_archive = _read_regular(REPO_ROOT / v19b_receipt["n600"]["archive"]["path"])
    v15_archive = context["n600_archive"]
    base_receiver = receive_carrier_compose_archive(v15_archive)
    correction_receiver = receive_preuint8_q8_archive(v19b_archive)
    labels = np.asarray(context["labels_all"][:600], np.uint8)
    poses = np.asarray(context["poses_all"][:600], np.float64)
    kinetic = _receiver_measurement(
        decoded=decoded,
        program=program,
        base_archive=v15_archive,
        base_receiver=base_receiver,
        correction_receiver=None,
        labels=labels,
        poses=poses,
        segnet=context["segnet"],
        posenet=context["posenet"],
        mode="kinetic_plus_v15_pose_base",
        output_root=output_root,
    )
    composed = _receiver_measurement(
        decoded=decoded,
        program=program,
        base_archive=v19b_archive,
        base_receiver=base_receiver,
        correction_receiver=correction_receiver,
        labels=labels,
        poses=poses,
        segnet=context["segnet"],
        posenet=context["posenet"],
        mode="kinetic_plus_v19b_measured_correction_delta",
        output_root=output_root,
    )
    result = {
        "schema": "ddm_kinetic_laguerre_stage_b.v1",
        "typed_config_sha256": config.typed_config_hash(),
        "status": "MEASURED",
        "stage_a_winner_id": winner["cell_id"],
        "kinetic_only": kinetic,
        "composed_with_v19b_corrections": composed,
        "secondary_falsifier_reached": not bool(kinetic["vehicle_admitted"] or composed["vehicle_admitted"]),
        "accounting_scope": (
            "EXACT_CONSERVATIVE_CONTAINER_UPPER_BOUND_EXISTING_V15_OR_V19B_"
            "PREDICTOR_REMAINS_CHARGED; no replacement byte saving inferred"
        ),
        "score_claim": False,
        "evidence_axis": AXIS,
    }
    _write_json(path, result)
    return result


def _storage_preflight(config: DDMKineticLaguerreAtToleranceProbeV1) -> dict[str, Any]:
    rows = []
    selected = None
    required = 128 * 1024 * 1024
    for raw in config.storage.waterfall:
        path = Path(raw)
        if not path.exists():
            rows.append({"path": raw, "status": "ABSENT"})
            continue
        usage = shutil.disk_usage(path)
        row = {
            "path": raw,
            "status": "PASS" if usage.free >= required else "INSUFFICIENT",
            "required_free_bytes": required,
            "observed_free_bytes": usage.free,
        }
        rows.append(row)
        if selected is None and row["status"] == "PASS":
            selected = raw
    if selected is None:
        raise ProbeError("no SSD tier passes the storage preflight")
    return {
        "status": "PASS",
        "waterfall_rows": rows,
        "selected_bulk_tier": selected,
        "local_output_scope": "small programs, JSON checkpoints, and receipt-byte containers only",
        "local_bulk_allowed": False,
        "success_only_scratch_cleanup": True,
    }


def _preflight(
    config: DDMKineticLaguerreAtToleranceProbeV1,
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    authority = Path(config.execution_authority.authority_file)
    actual_authority = sha256_file(authority)
    if actual_authority != config.execution_authority.sha256:
        raise ProbeError("delegated execution authority SHA differs")
    inputs = [
        (
            Path(config.inputs.target_cache_path),
            config.inputs.target_cache_sha256,
            "target_cache",
        ),
        (Path(config.inputs.modules_path), config.inputs.modules_sha256, "modules"),
        (
            REPO_ROOT / config.inputs.v19b_receipt_path,
            config.inputs.v19b_receipt_sha256,
            "v19b_receipt",
        ),
    ]
    rows = []
    for path, expected, name in inputs:
        actual = sha256_file(path)
        if actual != expected:
            raise ProbeError(f"{name} SHA differs: {actual} != {expected}")
        rows.append(
            {
                "name": name,
                "path": _portable(path),
                "bytes": path.stat().st_size,
                "sha256": actual,
            }
        )
    v19b = json.loads(_read_regular(REPO_ROOT / config.inputs.v19b_receipt_path))
    archive_path = REPO_ROOT / v19b["n600"]["archive"]["path"]
    archive_sha = sha256_file(archive_path)
    if archive_sha != config.inputs.v19b_n600_archive_sha256:
        raise ProbeError("v19b n600 archive SHA differs")
    if v19b.get("score_claim") is not False or v19b.get("pointer_moved") is not False:
        raise ProbeError("v19b false-authority labels drifted")
    resolved_output = output_root.resolve()
    research_root = (REPO_ROOT / ".omx" / "research").resolve()
    if research_root not in resolved_output.parents:
        raise ProbeError("output directory must be a child of .omx/research")
    return {
        "status": "PASS",
        "config_path": _portable(config_path),
        "typed_config_sha256": config.typed_config_hash(),
        "authority": {
            "path": str(authority),
            "sha256": actual_authority,
            "delegation_checkpoint_key": config.execution_authority.delegation_checkpoint_key,
        },
        "inputs": rows,
        "v19b_archive": {
            "path": _portable(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": archive_sha,
        },
        "storage": _storage_preflight(config),
        "kernel_contract": _kernel_contract(),
        "memory_ceiling_gib": 116,
        "no_paid_dispatch": True,
        "score_claim": False,
    }


def _matched_race(
    config: DDMKineticLaguerreAtToleranceProbeV1,
    stage_a: Mapping[str, Any],
) -> dict[str, Any]:
    rows = stage_a["matched_v19b_rows"]
    if not rows:
        return {
            "status": "KINETIC_DID_NOT_REACH_V19B_MATCH_FIDELITY",
            "kinetic": None,
            "describe_line": {
                "predictor_home_bytes": config.population.current_predictor_home_bytes,
                "errors": V19B_MATCH_ERRORS,
                "d_seg": "0.026594424778",
            },
            "kinetic_strictly_fewer_bytes_at_equal_or_better_d_seg": False,
        }
    kinetic = min(rows, key=lambda row: (row["program_bytes"], row["errors"]))
    line_bytes = config.population.current_predictor_home_bytes
    return {
        "status": (
            "KINETIC_FEWER_BYTES_AT_EQUAL_OR_BETTER_DSEG"
            if kinetic["program_bytes"] < line_bytes
            else "DESCRIBE_LINE_NO_MORE_BYTES_AT_MATCHED_DSEG"
        ),
        "kinetic": kinetic,
        "describe_line": {
            "predictor_home_bytes": line_bytes,
            "errors": V19B_MATCH_ERRORS,
            "d_seg": "0.026594424778",
        },
        "byte_delta_kinetic_minus_describe_line": kinetic["program_bytes"] - line_bytes,
        "kinetic_strictly_fewer_bytes_at_equal_or_better_d_seg": (kinetic["program_bytes"] < line_bytes),
    }


def run_probe(
    config: DDMKineticLaguerreAtToleranceProbeV1,
    *,
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = _preflight(config, config_path, output_root)
    labels_all = open_stored_npy_memmap(config.inputs.target_cache_path, "lstars")
    poses_all = open_stored_npy_memmap(config.inputs.target_cache_path, "gt_poses")
    gt_f1 = open_stored_npy_memmap(config.inputs.target_cache_path, "gt_f1")
    if labels_all.shape != (600, 384, 512):
        raise ProbeError(f"target label shape differs: {labels_all.shape}")
    palette = derive_palette(labels_all, gt_f1)
    n64 = _stage_a_rung(
        rung="n64",
        config=config,
        output_root=output_root,
        labels_all=labels_all,
        poses_all=poses_all,
        palette=palette,
    )
    n600 = _stage_a_rung(
        rung="n600",
        config=config,
        output_root=output_root,
        labels_all=labels_all,
        poses_all=poses_all,
        palette=palette,
    )
    stage_b = _stage_b_receiver(config=config, output_root=output_root, stage_a_n600=n600)
    matched = _matched_race(config, n600)
    if n600["stage_a_winner_count"]:
        vehicle = [
            row
            for row in (
                stage_b.get("kinetic_only"),
                stage_b.get("composed_with_v19b_corrections"),
            )
            if isinstance(row, dict) and row.get("vehicle_admitted") is True
        ]
        verdict = (
            "KINETIC_LAGUERRE_VEHICLE_REPOINT_TRIGGER"
            if vehicle
            else "STAGE_A_CONFIRMED_STAGE_B_RECEIVER_FALSIFIED_AT_REGISTERED_PULLBACK"
        )
        scope = (
            "INSTANCE:KINETIC_LAGUERRE_STAGE_A_WINNER_PLUS_REGISTERED_PALETTE_"
            "PULLBACK_AND_CONSERVATIVE_V15_V19B_CONTAINER"
        )
    else:
        verdict = "KINETIC_LAGUERRE_REGISTERED_LADDER_FORMULATION_FALSIFIED_STAGE_A"
        scope = (
            "FORMULATION:KINETIC_ANISOTROPIC_LAGUERRE_REGISTERED_LADDER; "
            "literal global Euclidean few-generator form remains sealed-dead; "
            "broader generator and operations-grammar families remain open"
        )
    receipt = {
        "schema": SCHEMA,
        "run_id": config.run_id,
        "lane_id": LANE_ID,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "preflight": preflight,
        "palette_rgb_counted": palette.tolist(),
        "stage_a_n64": n64,
        "stage_a_n600": n600,
        "stage_b": stage_b,
        "matched_d_seg_race": matched,
        "sealed_controls": config.inputs.sealed_v8_control,
        "verdict": verdict,
        "verdict_scope": scope,
        "vehicle_repoint": verdict == "KINETIC_LAGUERRE_VEHICLE_REPOINT_TRIGGER",
        "composed_row": stage_b.get("composed_with_v19b_corrections"),
        "triality": {
            "lane": LANE_ID,
            "dag_feed": ".omx/research/ddm_m2_kinetic_laguerre_probe_DAG_FEED_20260723.md",
            "equation": "N/A unless a stable non-vacuous n600 rate law is measured",
            "dsl": "typed DDMKineticLaguerreAtToleranceProbeV1 config; no trainer lever",
        },
        "six_hook_disposition": {
            "sensitivity_map": "receipt handoff only; no live v9 mutation",
            "pareto_constraint": "hard Stage-A and Stage-B boxes",
            "bit_allocator": "real-codec temporal segment waterfill",
            "cathedral_autopilot": "disabled research_only probe",
            "continual_learning": "DAG FEED and measured receipt; MAIN ingest required",
            "probe_disambiguator": "generator bytes versus describe-line predictor home at equal-or-better d_seg",
        },
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
        "not_a_candidate": True,
        "research_only": True,
        "execution_allowed": True,
        "evidence_axis": AXIS,
        "main_landing_review_required": True,
    }
    _write_json(output_root / "ddm_m2_kinetic_laguerre_probe_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    payload = _read_regular(config_path)
    try:
        config = DDMKineticLaguerreAtToleranceProbeV1.model_validate_json(payload)
    except Exception as exc:
        raise ProbeError(f"invalid typed config: {exc}") from exc
    receipt = run_probe(
        config,
        config_path=config_path,
        output_root=Path(args.output_directory),
    )
    best = receipt["stage_a_n600"]["best_complete_row"]
    composed = receipt["composed_row"]
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "verdict_scope": receipt["verdict_scope"],
                "best_kinetic_label_row": best,
                "matched_d_seg_race": receipt["matched_d_seg_race"],
                "composed_row": composed,
                "score_claim": False,
                "pointer_moved": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
