#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure V10's paid frame-0/free-predictor residual floor on real cache pairs.

``measure`` is deliberately capped at twelve pairs.  It forms both scorer
planes with the exact integer resize operator, exercises every production
predictor payload through parse-back, and reports actual Brotli-Q11 and
zstd-19 bytes.  Four canonical chunks compose to the n48 advisory surface.

``rung-e`` is an encode-side integration hook.  It builds and inflates the
additive production archive, verifies both factor-2 planes, and only then loads
the frozen CPU scorers.  Neither the predictor codec nor archive decoder loads
SegNet, PoseNet, Torch, source frames, labels, or margins.

``banked-ab`` consumes one immutable prepared V10 chunk and runs a matched
receiver-closed control versus scorer-plane predictor-residual precision drop.
It reports actual archive bytes and local hard-oracle distortion before any
explicitly labeled n600 projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _path in (REPO, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.codec.v10_predictor_residual import (  # noqa: E402
    AFFINE6_Q12_ID,
    MODE_BY_ID,
    PREVIOUS_PLANE_COPY_ID,
    SPATIAL_SMOOTH_121_ID,
    decode_predictor_residual,
    encode_predictor_residual,
    fit_affine6_q12_descriptor,
    predict_plane,
)
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator  # noqa: E402
from tools.measure_uint8_lattice_feasibility import stored_npy_memmap  # noqa: E402

SCHEMA_CHUNK = "v10_free_predictor_floor_chunk.v1"
SCHEMA_STATE = "v10_free_predictor_floor_state.v1"
SCHEMA_STAGE = "v10_free_predictor_floor_pair_stage.v1"
SCHEMA_COMPOSED = "v10_free_predictor_floor_n48.v1"
SCHEMA_RUNG_E = "v10_free_predictor_floor_rung_e.v1"
SCHEMA_BANKED_AB = "v10_free_predictor_floor_banked_ab.v1"
SCHEMA_BANKED_AB_STAGE = "v10_free_predictor_floor_banked_ab_stage.v1"
PREPARED_CHUNK_SCHEMA = "v10_two_plane_receiver_prepare_chunk.v1"
SCHEMA_ATTRIBUTION = "v10_predictor_attribution_stream.v1"
POINTER = "0.1910828242 [contest-CPU Linux x86_64] UNMOVED"
POINTER_VALUE = 0.1910828242
CONTEST_ARCHIVE_DENOMINATOR = 37_545_489
CONTEST_PAIR_COUNT = 600
AXIS = f"[{platform.system()}-{platform.machine()} CPU advisory real-cache subset] NON-PROMOTABLE"
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
N_CLASSES = 5
MAX_CHUNK = 12
N48_PAIRS = tuple(range(48))
CANONICAL_CHUNKS = tuple(tuple(range(start, start + MAX_CHUNK)) for start in range(0, 48, MAX_CHUNK))
MODES = (PREVIOUS_PLANE_COPY_ID, AFFINE6_Q12_ID, SPATIAL_SMOOTH_121_ID)
MARGIN_EDGES = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, float("inf"))
MARGIN_NAMES = ("[0,.1)", "[.1,.25)", "[.25,.5)", "[.5,1)", "[1,2)", "[2,inf)")
DEFAULT_CACHE = Path("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
DEFAULT_UPSTREAM = Path("/Users/adpena/Projects/pact/upstream")
DEFAULT_SACRED = Path("/Users/adpena/Projects/pact/experiments/results/levelset_n600_witness_20260717T113932Z")
EXPECTED_CACHE_SHA256 = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
SSD_ROOTS = (Path("/Volumes/VertigoDataTier/pact"), Path("/Volumes/APDataStore/pact"))

_ATTR_MAGIC = b"V10ATTR1"
_ATTR_PREFIX = struct.Struct("<8sHBBIHHH")
_ATTR_PAIR = struct.Struct("<IIIII32s32s")
_ATTR_KIND = {"class": 1, "margin": 2}
_ATTR_KIND_BY_ID = {value: key for key, value in _ATTR_KIND.items()}
_EPHEMERAL_MARKER_NAME = ".pact-rung-e-ephemeral-owner"
_EPHEMERAL_MARKER_BYTES = b"pact.v10.rung-e.ephemeral.v1\n"


class PredictorFloorError(RuntimeError):
    """Fail-closed cache, predictor, compressor, resume, or custody error."""


@dataclass(frozen=True)
class RungEInputs:
    pair_ids: tuple[int, ...]
    mode: str
    frame0_y_planes: np.ndarray
    frame1_y_planes: np.ndarray
    descriptors: tuple[bytes, ...]
    cache_sha256: str
    predictor_payload_sha256: str


@dataclass(frozen=True)
class PreparedChunkCustody:
    """One immutable prepared V10 chunk plus its bound source artifacts."""

    inputs: RungEInputs
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    y0_path: Path
    y1_path: Path
    predictor_path: Path


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(value).tobytes())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def _write_once_or_equal(path: Path, value: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != value:
            raise PredictorFloorError(f"preserved write-once artifact drifted: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.write-once-{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if not path.is_file() or path.read_bytes() != value:
                raise PredictorFloorError(f"preserved write-once artifact drifted: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _durable_path(path: Path, field: str, *, require_ssd: bool = False, allow_local: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    for root in (Path("/tmp"), Path("/private/tmp"), Path("/var/tmp")):
        if _is_relative_to(resolved, root):
            raise PredictorFloorError(f"{field} must be durable, not under {root}")
    if _is_relative_to(resolved, DEFAULT_SACRED.resolve()):
        raise PredictorFloorError(f"{field} may not mutate the sacred result tree")
    if (
        require_ssd
        and not allow_local
        and not any(root.exists() and _is_relative_to(resolved, root.resolve()) for root in SSD_ROOTS)
    ):
        raise PredictorFloorError(f"{field} must use the SSD waterfall (or explicit local opt-in)")
    return resolved


def _ephemeral_output_root(path: Path) -> Path:
    """Admit only one narrowly named temporary rung-E tree for certified cleanup."""

    resolved = path.expanduser().resolve()
    temp_roots = {
        Path(tempfile.gettempdir()).resolve(),
        Path("/tmp").resolve(),
        Path("/private/tmp").resolve(),
        Path("/var/tmp").resolve(),
    }
    if not any(resolved != root and _is_relative_to(resolved, root) for root in temp_roots):
        raise PredictorFloorError("ephemeral rung-E output must be a child of the system temporary root")
    if not resolved.name.startswith("pact-rung-e-"):
        raise PredictorFloorError("ephemeral rung-E output basename must start with 'pact-rung-e-'")
    if resolved.is_symlink():
        raise PredictorFloorError("ephemeral rung-E output may not be a symlink")
    return resolved


def _ephemeral_storage_preflight(path: Path, pair_count: int) -> dict[str, Any]:
    existing = path.parent
    while not existing.exists():
        if existing.parent == existing:
            raise PredictorFloorError("cannot resolve ephemeral output filesystem")
        existing = existing.parent
    frame_bytes = CAMERA_HW[0] * CAMERA_HW[1] * 3
    required = pair_count * 2 * frame_bytes * 2 + (256 << 20)
    free = shutil.disk_usage(existing).free
    if free < required:
        raise PredictorFloorError(f"ephemeral rung-E storage preflight refused: {free} < {required} bytes")
    return {"required_bytes": required, "free_bytes": free, "passed": True}


def _cleanup_ephemeral_output(path: Path) -> None:
    validated = _ephemeral_output_root(path)
    if validated.exists():
        if not validated.is_dir() or validated.is_symlink():
            raise PredictorFloorError("ephemeral cleanup target is not one exact real directory")
        marker = validated / _EPHEMERAL_MARKER_NAME
        if not marker.is_file() or marker.read_bytes() != _EPHEMERAL_MARKER_BYTES:
            raise PredictorFloorError("ephemeral cleanup target lacks the exact ownership marker")
        shutil.rmtree(validated)
    if validated.exists():
        raise PredictorFloorError("ephemeral rung-E cleanup did not remove its exact output tree")


def _tree_snapshot(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    entries = 0
    if not root.is_dir():
        return {"exists": False, "entries": 0, "metadata_sha256": None}
    for current, directories, files in os.walk(root, followlinks=False):
        directories.sort()
        files.sort()
        base = Path(current)
        for name in (*directories, *files):
            path = base / name
            stat = path.lstat()
            digest.update(
                f"{path.relative_to(root).as_posix()}\0{stat.st_mode}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
            )
            entries += 1
    return {"exists": True, "entries": entries, "metadata_sha256": digest.hexdigest()}


def _load_cache(path: Path, *, require_canonical_hash: bool = True) -> tuple[dict[str, np.memmap], str]:
    if not path.is_file():
        raise PredictorFloorError(f"real cache is absent: {path}")
    cache_sha = _sha256_file(path)
    if require_canonical_hash and cache_sha != EXPECTED_CACHE_SHA256:
        raise PredictorFloorError("real n600 cache SHA-256 differs from the frozen Task #541 input")
    keys = ("n_pairs", "gt_f0", "gt_f1", "lstars", "margins", "gt_poses")
    try:
        fields = {key: stored_npy_memmap(path, key) for key in keys}
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise PredictorFloorError("real cache must expose ZIP_STORED mmap-compatible NPY members") from exc
    if int(np.asarray(fields["n_pairs"]).reshape(())) != 600:
        raise PredictorFloorError("only the real n600 cache is admissible")
    expected = {
        "gt_f0": (600, *CAMERA_HW, 3),
        "gt_f1": (600, *CAMERA_HW, 3),
        "lstars": (600, *SCORER_HW),
        "margins": (600, *SCORER_HW),
        "gt_poses": (600, 6),
    }
    for key, shape in expected.items():
        if fields[key].shape != shape:
            raise PredictorFloorError(f"real cache {key} geometry drift: {fields[key].shape} != {shape}")
    if fields["gt_f0"].dtype != np.uint8 or fields["gt_f1"].dtype != np.uint8:
        raise PredictorFloorError("real cache source frames must remain uint8")
    return fields, cache_sha


def exact_operator_round_u8(operator: DisjointResizeOperator, frame: np.ndarray) -> np.ndarray:
    """Apply exact integer A and round half-up to the production uint8 plane."""

    numerators, denominator = operator.apply_numerators(frame)
    if denominator <= 0 or np.any(numerators < 0):
        raise PredictorFloorError("exact resize numerator/denominator left the uint8 domain")
    rounded = (numerators.astype(np.int64) + denominator // 2) // denominator
    if np.any(rounded > 255):
        raise PredictorFloorError("rounded scorer plane exceeds uint8")
    return np.ascontiguousarray(rounded.astype(np.uint8))


def _mode_descriptor(y0: np.ndarray, y1: np.ndarray, mode: str) -> bytes:
    if mode not in MODE_BY_ID:
        raise PredictorFloorError(f"unknown production predictor mode: {mode}")
    return fit_affine6_q12_descriptor(y0, y1) if mode == AFFINE6_Q12_ID else b""


def _mode_residual(y0: np.ndarray, y1: np.ndarray, mode: str, descriptor: bytes) -> np.ndarray:
    predictor = predict_plane(y0, mode, descriptor)
    residual = y1.astype(np.int16) - predictor.astype(np.int16)
    return np.ascontiguousarray(residual.astype("<i2", copy=False))


def _brotli_roundtrip(payload: bytes) -> dict[str, Any]:
    compressed = bytes(brotli.compress(payload, quality=11, mode=brotli.MODE_GENERIC))
    if bytes(brotli.decompress(compressed)) != payload:
        raise PredictorFloorError("Brotli-Q11 decompression differs from input")
    return {"codec": "brotli", "level": 11, "bytes": len(compressed), "sha256": _sha256_bytes(compressed)}


def _zstd_roundtrip(payload: bytes, binary: Path | str) -> dict[str, Any]:
    executable = str(binary)
    try:
        compressed_run = subprocess.run(
            [executable, "-q", "-19", "-T1", "--stdout"],
            input=payload,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise PredictorFloorError(f"zstd executable is unavailable: {executable}") from exc
    if compressed_run.returncode != 0:
        raise PredictorFloorError(f"zstd-19 compression failed: {compressed_run.stderr.decode(errors='replace')}")
    compressed = compressed_run.stdout
    decompressed_run = subprocess.run(
        [executable, "-q", "-d", "--stdout"],
        input=compressed,
        capture_output=True,
        check=False,
    )
    if decompressed_run.returncode != 0 or decompressed_run.stdout != payload:
        raise PredictorFloorError("zstd-19 decompression differs from input")
    return {"codec": "zstd", "level": 19, "bytes": len(compressed), "sha256": _sha256_bytes(compressed)}


def compress_roundtrip(payload: bytes, *, zstd_binary: Path | str = "zstd") -> dict[str, Any]:
    """Measure both required outer coders and verify their exact inverse."""

    if not isinstance(payload, bytes):
        raise PredictorFloorError("compression input must be immutable bytes")
    return {
        "raw_bytes": len(payload),
        "raw_sha256": _sha256_bytes(payload),
        "brotli_q11": _brotli_roundtrip(payload),
        "zstd_19": _zstd_roundtrip(payload, zstd_binary),
        "decompression_verified": True,
    }


def pack_attribution_stream(
    *,
    kind: str,
    bucket_id: int,
    pair_ids: Sequence[int],
    masks: Sequence[np.ndarray],
    residuals: Sequence[np.ndarray],
) -> bytes:
    """Pack counted membership masks plus their selected little-endian RGB residuals."""

    if kind not in _ATTR_KIND or not 0 <= int(bucket_id) <= 255:
        raise PredictorFloorError("attribution kind/bucket is outside the closed registry")
    if not pair_ids or len(pair_ids) != len(masks) or len(pair_ids) != len(residuals):
        raise PredictorFloorError("attribution pair/mask/residual counts must agree")
    first = np.asarray(residuals[0])
    if first.ndim != 3 or first.shape[-1] != 3:
        raise PredictorFloorError("attribution residuals must be HxWx3")
    height, width, channels = map(int, first.shape)
    payload = bytearray(
        _ATTR_PREFIX.pack(_ATTR_MAGIC, 1, _ATTR_KIND[kind], int(bucket_id), len(pair_ids), height, width, channels)
    )
    for pair_id, raw_mask, raw_residual in zip(pair_ids, masks, residuals, strict=True):
        mask = np.asarray(raw_mask)
        residual = np.asarray(raw_residual)
        if mask.shape != (height, width) or mask.dtype.kind != "b":
            raise PredictorFloorError("attribution mask must be exact boolean HxW")
        if residual.shape != (height, width, channels) or residual.dtype != np.dtype("<i2"):
            raise PredictorFloorError("attribution residual must be exact little-endian int16 HxWx3")
        packed_mask = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
        selected = np.ascontiguousarray(residual[mask].astype("<i2", copy=False)).tobytes()
        selected_count = int(np.count_nonzero(mask))
        payload.extend(
            _ATTR_PAIR.pack(
                int(pair_id),
                height * width,
                selected_count,
                len(packed_mask),
                len(selected),
                bytes.fromhex(_sha256_bytes(packed_mask)),
                bytes.fromhex(_sha256_bytes(selected)),
            )
        )
        payload.extend(packed_mask)
        payload.extend(selected)
    result = bytes(payload)
    parse_attribution_stream(result)
    return result


def parse_attribution_stream(payload: bytes) -> dict[str, Any]:
    """Strictly parse an attribution stream, including unused packed-mask bits."""

    if not isinstance(payload, bytes) or len(payload) < _ATTR_PREFIX.size:
        raise PredictorFloorError("attribution stream is truncated")
    magic, version, kind_id, bucket_id, pair_count, height, width, channels = _ATTR_PREFIX.unpack_from(payload)
    if magic != _ATTR_MAGIC or version != 1 or kind_id not in _ATTR_KIND_BY_ID or channels != 3:
        raise PredictorFloorError("attribution stream header is invalid")
    if not pair_count or not height or not width:
        raise PredictorFloorError("attribution stream geometry must be nonempty")
    cursor = _ATTR_PREFIX.size
    pair_ids: list[int] = []
    selected_counts: list[int] = []
    expected_mask_bytes = (height * width + 7) // 8
    for _ in range(pair_count):
        if cursor + _ATTR_PAIR.size > len(payload):
            raise PredictorFloorError("attribution pair header is truncated")
        pair_id, pixels, selected_count, mask_len, values_len, mask_sha, values_sha = _ATTR_PAIR.unpack_from(
            payload, cursor
        )
        cursor += _ATTR_PAIR.size
        if pixels != height * width or mask_len != expected_mask_bytes or values_len != selected_count * channels * 2:
            raise PredictorFloorError("attribution pair length/geometry drift")
        end = cursor + mask_len + values_len
        if end > len(payload):
            raise PredictorFloorError("attribution pair body is truncated")
        packed_mask = payload[cursor : cursor + mask_len]
        cursor += mask_len
        values = payload[cursor : cursor + values_len]
        cursor += values_len
        if hashlib.sha256(packed_mask).digest() != mask_sha or hashlib.sha256(values).digest() != values_sha:
            raise PredictorFloorError("attribution pair hash custody failure")
        unpacked = np.unpackbits(np.frombuffer(packed_mask, dtype=np.uint8), bitorder="little")
        if np.any(unpacked[height * width :]) or int(np.count_nonzero(unpacked[: height * width])) != selected_count:
            raise PredictorFloorError("attribution packed membership mask is noncanonical")
        pair_ids.append(int(pair_id))
        selected_counts.append(int(selected_count))
    if cursor != len(payload) or len(set(pair_ids)) != len(pair_ids):
        raise PredictorFloorError("attribution stream has trailing data or duplicate pair ids")
    return {
        "schema": SCHEMA_ATTRIBUTION,
        "kind": _ATTR_KIND_BY_ID[kind_id],
        "bucket_id": int(bucket_id),
        "pair_ids": pair_ids,
        "selected_counts": selected_counts,
        "geometry": [height, width, channels],
    }


def _pair_planes(
    fields: Mapping[str, np.ndarray], pair_id: int, operator: DisjointResizeOperator
) -> tuple[np.ndarray, np.ndarray]:
    source0 = np.asarray(fields["gt_f0"][pair_id], dtype=np.uint8)
    source1 = np.asarray(fields["gt_f1"][pair_id], dtype=np.uint8)
    return exact_operator_round_u8(operator, source0), exact_operator_round_u8(operator, source1)


def _pair_stage(fields: Mapping[str, np.ndarray], pair_id: int, operator: DisjointResizeOperator) -> dict[str, Any]:
    y0, y1 = _pair_planes(fields, pair_id, operator)
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        descriptor = _mode_descriptor(y0, y1, mode)
        residual = _mode_residual(y0, y1, mode, descriptor)
        payload = encode_predictor_residual(
            y0[None], y1[None], modes=mode, descriptors=(descriptor,), pair_ids=(pair_id,)
        )
        decoded = decode_predictor_residual(payload)
        if (
            decoded.pair_ids != (pair_id,)
            or not np.array_equal(decoded.frame0[0], y0)
            or not np.array_equal(decoded.frame1[0], y1)
        ):
            raise PredictorFloorError("production predictor parse-back differs from exact scorer planes")
        rows.append(
            {
                "mode": mode,
                "descriptor_bytes": len(descriptor),
                "descriptor_sha256": _sha256_bytes(descriptor),
                "residual_bytes": int(residual.nbytes),
                "residual_sha256": _sha256_array(residual),
                "residual_nonzero_values": int(np.count_nonzero(residual)),
                "residual_abs_sum": int(np.abs(residual.astype(np.int32)).sum(dtype=np.int64)),
                "payload_sha256": _sha256_bytes(payload),
                "parseback_exact": True,
            }
        )
    return {
        "schema": SCHEMA_STAGE,
        "pair_id": int(pair_id),
        "frame0_y_sha256": _sha256_array(y0),
        "frame1_y_sha256": _sha256_array(y1),
        "labels_sha256": _sha256_array(np.asarray(fields["lstars"][pair_id])),
        "margins_sha256": _sha256_array(np.asarray(fields["margins"][pair_id])),
        "modes": rows,
    }


def _attribution_rows(
    *,
    pair_ids: Sequence[int],
    labels: Sequence[np.ndarray],
    margins: Sequence[np.ndarray],
    residuals: Sequence[np.ndarray],
    zstd_binary: Path | str,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"class": [], "margin": []}
    for class_id in range(N_CLASSES):
        masks = [np.asarray(value) == class_id for value in labels]
        stream = pack_attribution_stream(
            kind="class", bucket_id=class_id, pair_ids=pair_ids, masks=masks, residuals=residuals
        )
        result["class"].append(
            {
                "class_id": class_id,
                "selected_pixels": int(sum(np.count_nonzero(mask) for mask in masks)),
                "counted_stream": compress_roundtrip(stream, zstd_binary=zstd_binary),
            }
        )
    for bucket_id, (lower, upper, name) in enumerate(
        zip(MARGIN_EDGES[:-1], MARGIN_EDGES[1:], MARGIN_NAMES, strict=True)
    ):
        masks = [(np.asarray(value) >= lower) & (np.asarray(value) < upper) for value in margins]
        stream = pack_attribution_stream(
            kind="margin", bucket_id=bucket_id, pair_ids=pair_ids, masks=masks, residuals=residuals
        )
        result["margin"].append(
            {
                "margin_bin": name,
                "lower_inclusive": lower,
                "upper_exclusive": None if np.isinf(upper) else upper,
                "selected_pixels": int(sum(np.count_nonzero(mask) for mask in masks)),
                "counted_stream": compress_roundtrip(stream, zstd_binary=zstd_binary),
            }
        )
    return result


def measure_planes(
    *,
    pair_ids: Sequence[int],
    frame0_y_planes: np.ndarray,
    frame1_y_planes: np.ndarray,
    labels: Sequence[np.ndarray],
    margins: Sequence[np.ndarray],
    zstd_binary: Path | str = "zstd",
) -> list[dict[str, Any]]:
    """Pure measurement core used by chunk execution and focused tests."""

    y0s = np.asarray(frame0_y_planes)
    y1s = np.asarray(frame1_y_planes)
    if y0s.dtype != np.uint8 or y1s.dtype != np.uint8 or y0s.shape != y1s.shape:
        raise PredictorFloorError("measurement planes must be equal-shape uint8 arrays")
    if y0s.ndim != 4 or y0s.shape[0] != len(pair_ids) or y0s.shape[-1] != 3:
        raise PredictorFloorError("measurement planes must be [pairs,H,W,3]")
    selected_pair_ids = tuple(int(value) for value in pair_ids)
    if (
        any(right <= left for left, right in pairwise(selected_pair_ids))
        or len(labels) != len(pair_ids)
        or len(margins) != len(pair_ids)
    ):
        raise PredictorFloorError("measurement pair ids/classes/margins must be unique and aligned")
    expected_map_shape = y0s.shape[1:3]
    for pair_index, (raw_labels, raw_margins) in enumerate(zip(labels, margins, strict=True)):
        pair_labels = np.asarray(raw_labels)
        pair_margins = np.asarray(raw_margins)
        if pair_labels.shape != expected_map_shape or pair_labels.dtype.kind not in ("i", "u"):
            raise PredictorFloorError(f"pair {pair_index} class map geometry/dtype drift")
        if np.any(pair_labels < 0) or np.any(pair_labels >= N_CLASSES):
            raise PredictorFloorError(f"pair {pair_index} class map leaves canonical classes 0..4")
        if pair_margins.shape != expected_map_shape or pair_margins.dtype.kind not in ("i", "u", "f"):
            raise PredictorFloorError(f"pair {pair_index} margin map geometry/dtype drift")
        if not np.all(np.isfinite(pair_margins)) or np.any(pair_margins < 0):
            raise PredictorFloorError(f"pair {pair_index} margins must be finite and nonnegative")
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        descriptors = tuple(_mode_descriptor(y0, y1, mode) for y0, y1 in zip(y0s, y1s, strict=True))
        residuals = tuple(
            _mode_residual(y0, y1, mode, descriptor) for y0, y1, descriptor in zip(y0s, y1s, descriptors, strict=True)
        )
        payload = encode_predictor_residual(
            y0s,
            y1s,
            modes=mode,
            descriptors=descriptors,
            pair_ids=selected_pair_ids,
        )
        decoded = decode_predictor_residual(payload)
        if (
            decoded.pair_ids != selected_pair_ids
            or decoded.modes != (mode,) * len(pair_ids)
            or decoded.descriptors != descriptors
            or not np.array_equal(decoded.frame0, y0s)
            or not np.array_equal(decoded.frame1, y1s)
        ):
            raise PredictorFloorError("production codec descriptor/two-plane parse-back differs")
        decoded_conditional = b"".join(
            descriptor + residual.tobytes(order="C")
            for descriptor, residual in zip(descriptors, residuals, strict=True)
        )
        accounting = decoded.accounting
        if accounting.descriptor_bytes + accounting.decoded_residual_bytes != len(decoded_conditional):
            raise PredictorFloorError("decoded conditional descriptor+residual accounting drift")
        rows.append(
            {
                "mode": mode,
                "pair_count": len(pair_ids),
                "full_representation": {
                    "definition": "complete production predictor payload with inner Brotli-Q11 frame0/residual content pricing",
                    "accounting": {
                        "framing_bytes": accounting.framing_bytes,
                        "bootstrap_brotli_q11_bytes": accounting.bootstrap_bytes,
                        "descriptor_bytes": accounting.descriptor_bytes,
                        "residual_brotli_q11_bytes": accounting.residual_bytes,
                        "decoded_bootstrap_bytes": accounting.decoded_bootstrap_bytes,
                        "decoded_residual_bytes": accounting.decoded_residual_bytes,
                    },
                    "production_counted_stream": {
                        "codec": "predictor-residual-u8.v1 / brotli-q11.v1",
                        "bytes": len(payload),
                        "sha256": _sha256_bytes(payload),
                        "parseback_decompression_verified": True,
                    },
                    "secondary_double_compression_diagnostic": compress_roundtrip(payload, zstd_binary=zstd_binary),
                },
                "conditional_representation": {
                    "definition": "descriptor + inner-Brotli-Q11 residual bytes conditioned on decoded frame0, mode, and geometry",
                    "production_brotli_q11_bytes": accounting.conditional_bytes,
                    "decoded_descriptor_plus_residual_bytes": len(decoded_conditional),
                    "direct_global_coder_ab": compress_roundtrip(decoded_conditional, zstd_binary=zstd_binary),
                },
                "prediction": {
                    "residual_values": int(sum(residual.size for residual in residuals)),
                    "residual_nonzero_values": int(sum(np.count_nonzero(residual) for residual in residuals)),
                    "residual_abs_sum": int(
                        sum(np.abs(residual.astype(np.int32)).sum(dtype=np.int64) for residual in residuals)
                    ),
                },
                "attribution": _attribution_rows(
                    pair_ids=pair_ids,
                    labels=labels,
                    margins=margins,
                    residuals=residuals,
                    zstd_binary=zstd_binary,
                ),
                "production_parseback_exact": True,
            }
        )
    return rows


def _pair_ids(explicit: Sequence[int] | None, chunk_index: int | None) -> tuple[int, ...]:
    if explicit is not None and chunk_index is not None:
        raise PredictorFloorError("choose either --pairs or --chunk-index, not both")
    if chunk_index is not None:
        if not 0 <= int(chunk_index) < len(CANONICAL_CHUNKS):
            raise PredictorFloorError("chunk index must be one of 0,1,2,3")
        result = CANONICAL_CHUNKS[int(chunk_index)]
    elif explicit is not None:
        result = tuple(int(value) for value in explicit)
    else:
        raise PredictorFloorError("measure requires --pairs or --chunk-index")
    if (
        not 1 <= len(result) <= MAX_CHUNK
        or any(value < 0 or value >= 600 for value in result)
        or any(right <= left for left, right in pairwise(result))
    ):
        raise PredictorFloorError(f"measure requires 1..{MAX_CHUNK} unique real pair ids in [0,600)")
    return result


def _zstd_version(binary: Path | str) -> str:
    try:
        run = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=False)
    except OSError as exc:
        raise PredictorFloorError(f"zstd executable is unavailable: {binary}") from exc
    if run.returncode != 0 or not run.stdout.strip():
        raise PredictorFloorError("zstd --version failed")
    return run.stdout.strip()


def measure_chunk(args: argparse.Namespace) -> dict[str, Any]:
    """Measure one bounded real-cache chunk with re-derived crash resume."""

    output = _durable_path(args.output, "output")
    state_path = _durable_path(args.state, "state")
    stage_dir = _durable_path(args.stage_dir, "stage-dir")
    if (
        len({output, state_path, stage_dir}) != 3
        or _is_relative_to(output, stage_dir)
        or _is_relative_to(state_path, stage_dir)
    ):
        raise PredictorFloorError("output, state, and stage-dir must be distinct non-overlapping paths")
    if stage_dir.exists() and not stage_dir.is_dir():
        raise PredictorFloorError("stage-dir exists but is not a directory")
    if output.exists():
        raise PredictorFloorError(f"write-once receipt already exists: {output}")
    pair_ids = _pair_ids(args.pairs, args.chunk_index)
    cache_path = args.cache.expanduser().resolve()
    sacred = args.sacred.expanduser().resolve()
    sacred_before = _tree_snapshot(sacred)
    fields, cache_sha = _load_cache(cache_path, require_canonical_hash=not args.allow_noncanonical_cache)
    zstd_version = _zstd_version(args.zstd_binary)
    codec_path = SRC / "tac/codec/v10_predictor_residual.py"
    config = {
        "schema": SCHEMA_STATE,
        "pair_ids": list(pair_ids),
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha,
        "codec_path": str(codec_path),
        "codec_sha256": _sha256_file(codec_path),
        "tool_sha256": _sha256_file(Path(__file__).resolve()),
        "camera_hw": list(CAMERA_HW),
        "scorer_hw": list(SCORER_HW),
        "predictor_modes": list(MODES),
        "rounding": "exact integer numerator, nonnegative round-half-up to uint8",
        "brotli": {"version": getattr(brotli, "__version__", "unknown"), "quality": 11},
        "zstd": {"binary": str(args.zstd_binary), "version": zstd_version, "level": 19, "threads": 1},
    }
    config_sha = _sha256_bytes(_canonical(config))
    completed = 0
    if args.resume:
        if not state_path.is_file():
            raise PredictorFloorError("--resume requires an existing state file")
        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise PredictorFloorError("resume state is unreadable") from exc
        if (
            state.get("schema") != SCHEMA_STATE
            or state.get("config_sha256") != config_sha
            or state.get("config") != config
        ):
            raise PredictorFloorError("resume state config/input/tool custody mismatch")
        completed_ids = state.get("completed_pair_ids")
        if not isinstance(completed_ids, list) or len(completed_ids) > len(pair_ids):
            raise PredictorFloorError("resume completed_pair_ids must be a bounded list")
        if completed_ids != list(pair_ids[: len(completed_ids)]):
            raise PredictorFloorError("resume completed pairs are not one canonical prefix")
        completed = len(completed_ids)
    elif state_path.exists() or (stage_dir.exists() and any(stage_dir.iterdir())):
        raise PredictorFloorError("preserved state/stages exist; use --resume or new paths")
    else:
        _atomic_json(
            state_path,
            {"schema": SCHEMA_STATE, "config_sha256": config_sha, "config": config, "completed_pair_ids": []},
        )

    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HW[0], camera_w=CAMERA_HW[1], scorer_h=SCORER_HW[0], scorer_w=SCORER_HW[1]
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    # Re-derive all completed stages from frozen inputs.  Stored scientific rows
    # never flow into the final receipt merely because state says "complete".
    for index, pair_id in enumerate(pair_ids):
        stage = _pair_stage(fields, pair_id, operator)
        stage["config_sha256"] = config_sha
        stage_bytes = json.dumps(stage, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
        stage_path = stage_dir / f"pair-{pair_id:04d}.json"
        if index < completed:
            if not stage_path.is_file() or stage_path.read_bytes() != stage_bytes:
                raise PredictorFloorError(f"resume stage re-derivation/custody mismatch for pair {pair_id}")
        else:
            _write_once_or_equal(stage_path, stage_bytes)
            _atomic_json(
                state_path,
                {
                    "schema": SCHEMA_STATE,
                    "config_sha256": config_sha,
                    "config": config,
                    "completed_pair_ids": list(pair_ids[: index + 1]),
                },
            )

    frame0_rows: list[np.ndarray] = []
    frame1_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    for pair_id in pair_ids:
        y0, y1 = _pair_planes(fields, pair_id, operator)
        frame0_rows.append(y0)
        frame1_rows.append(y1)
        labels.append(np.asarray(fields["lstars"][pair_id], dtype=np.uint8).copy())
        margins.append(np.asarray(fields["margins"][pair_id], dtype=np.float32).copy())
    measured = measure_planes(
        pair_ids=pair_ids,
        frame0_y_planes=np.stack(frame0_rows),
        frame1_y_planes=np.stack(frame1_rows),
        labels=labels,
        margins=margins,
        zstd_binary=args.zstd_binary,
    )
    if _tree_snapshot(sacred) != sacred_before:
        raise PredictorFloorError("sacred result tree changed during read-only measurement")
    receipt = {
        "schema": SCHEMA_CHUNK,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "axis": AXIS,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": POINTER,
            "pointer_moved": False,
            "verdict_scope": "selected real n600-cache pairs and production predictor codec; no contest score or family kill",
        },
        "labels": {
            "MEASURED": [
                "exact cache-derived uint8 scorer planes",
                "actual Brotli quality-11 bytes with decompression",
                "actual zstd level-19 bytes with decompression",
                "packed-mask class and margin attribution streams",
            ],
            "DERIVED": ["conditional descriptor+residual rate", "round-half-up exact-operator y0/y1"],
            "INFERRED": [],
        },
        "config": config,
        "config_sha256": config_sha,
        "pair_ids": list(pair_ids),
        "pair_count": len(pair_ids),
        "modes": measured,
        "cache_custody": {"path": str(cache_path), "bytes": cache_path.stat().st_size, "sha256": cache_sha},
        "resumability": {
            "state": str(state_path),
            "stage_dir": str(stage_dir),
            "per_pair_stages_preserved": True,
            "resume_rederives_frozen_inputs": True,
        },
        "sacred_tree": {"path": str(sacred), "unchanged": True, "snapshot": sacred_before},
        "research_only": True,
    }
    _atomic_json(output, receipt)
    return receipt


def _source_without_pairs(doc: Mapping[str, Any]) -> bytes:
    config = dict(doc["config"])
    config.pop("pair_ids", None)
    return _canonical(config)


def _sum_stream(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "stream_count": len(rows),
        "raw_bytes": int(sum(row["raw_bytes"] for row in rows)),
        "brotli_q11_bytes": int(sum(row["brotli_q11"]["bytes"] for row in rows)),
        "zstd_19_bytes": int(sum(row["zstd_19"]["bytes"] for row in rows)),
        "all_decompression_verified": all(row["decompression_verified"] for row in rows),
        "composition": "sum of four independently parseable, actually compressed canonical chunk streams",
    }


def compose_chunks(receipts: Sequence[Path], output: Path) -> dict[str, Any]:
    """Compose exactly the four disjoint canonical n12 chunks to n48."""

    target = _durable_path(output, "output")
    if target.exists():
        raise PredictorFloorError(f"write-once composed receipt already exists: {target}")
    paths = [path.expanduser().resolve() for path in receipts]
    if len(paths) != 4 or len(set(paths)) != 4:
        raise PredictorFloorError("n48 composition requires exactly four distinct chunk receipts")
    try:
        docs = [json.loads(path.read_text()) for path in paths]
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictorFloorError("a source chunk receipt is unreadable") from exc
    if any(doc.get("schema") != SCHEMA_CHUNK for doc in docs):
        raise PredictorFloorError("all composition inputs must be predictor-floor chunk v1 receipts")
    chunks = [tuple(int(value) for value in doc["pair_ids"]) for doc in docs]
    if set(chunks) != set(CANONICAL_CHUNKS):
        raise PredictorFloorError("composition requires exactly chunks 0..11, 12..23, 24..35, 36..47")
    if len({_source_without_pairs(doc) for doc in docs}) != 1:
        raise PredictorFloorError("chunk codec/cache/compressor/tool custody differs")
    by_mode = [{row["mode"]: row for row in doc["modes"]} for doc in docs]
    if any(set(rows) != set(MODES) for rows in by_mode):
        raise PredictorFloorError("chunk predictor mode registry differs")
    mode_rows: list[dict[str, Any]] = []
    for mode in MODES:
        group = [rows[mode] for rows in by_mode]
        attribution: dict[str, list[dict[str, Any]]] = {"class": [], "margin": []}
        for kind, count in (("class", N_CLASSES), ("margin", len(MARGIN_NAMES))):
            for bucket_id in range(count):
                bucket_group = [row["attribution"][kind][bucket_id] for row in group]
                identity_key = "class_id" if kind == "class" else "margin_bin"
                if len({row[identity_key] for row in bucket_group}) != 1:
                    raise PredictorFloorError("attribution bucket ordering differs across chunks")
                attribution[kind].append(
                    {
                        identity_key: bucket_group[0][identity_key],
                        "selected_pixels": int(sum(row["selected_pixels"] for row in bucket_group)),
                        "counted_streams": _sum_stream([row["counted_stream"] for row in bucket_group]),
                    }
                )
        mode_rows.append(
            {
                "mode": mode,
                "pair_count": 48,
                "full_representation": {
                    "production_brotli_q11_archive_section_bytes": int(
                        sum(row["full_representation"]["production_counted_stream"]["bytes"] for row in group)
                    ),
                    "production_stream_count": 4,
                    "all_parseback_decompression_verified": all(
                        row["full_representation"]["production_counted_stream"]["parseback_decompression_verified"]
                        for row in group
                    ),
                    "secondary_double_compression_diagnostic": _sum_stream(
                        [row["full_representation"]["secondary_double_compression_diagnostic"] for row in group]
                    ),
                },
                "conditional_representation": {
                    "production_brotli_q11_bytes": int(
                        sum(row["conditional_representation"]["production_brotli_q11_bytes"] for row in group)
                    ),
                    "decoded_descriptor_plus_residual_bytes": int(
                        sum(
                            row["conditional_representation"]["decoded_descriptor_plus_residual_bytes"] for row in group
                        )
                    ),
                    "direct_global_coder_ab": _sum_stream(
                        [row["conditional_representation"]["direct_global_coder_ab"] for row in group]
                    ),
                },
                "prediction": {
                    key: int(sum(row["prediction"][key] for row in group))
                    for key in ("residual_values", "residual_nonzero_values", "residual_abs_sum")
                },
                "attribution": attribution,
            }
        )
    best = min(
        mode_rows,
        key=lambda row: (
            row["full_representation"]["production_brotli_q11_archive_section_bytes"],
            row["conditional_representation"]["direct_global_coder_ab"]["zstd_19_bytes"],
            MODES.index(row["mode"]),
        ),
    )
    result = {
        "schema": SCHEMA_COMPOSED,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "axis": AXIS.replace("subset", "n48"),
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": POINTER,
            "pointer_moved": False,
            "verdict_scope": "48 real pairs and four independently compressed n12 streams only",
        },
        "labels": {
            "MEASURED": ["four actual n12 codec/compressor receipts"],
            "DERIVED": ["n48 sums and best-mode ordering"],
            "INFERRED": [],
        },
        "pair_ids": list(N48_PAIRS),
        "pair_count": 48,
        "exact_canonical_chunks": [list(chunk) for chunk in CANONICAL_CHUNKS],
        "modes": mode_rows,
        "best_predictor": {
            "mode": best["mode"],
            "selection_rule": "minimum production inner-Brotli-Q11 archive-section bytes; direct conditional zstd-19 then registry order break ties",
            "production_brotli_q11_archive_section_bytes": best["full_representation"][
                "production_brotli_q11_archive_section_bytes"
            ],
            "conditional_zstd_19_bytes": best["conditional_representation"]["direct_global_coder_ab"]["zstd_19_bytes"],
        },
        "source_receipts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)} for path in paths
        ],
        "cache_custody": docs[0]["cache_custody"],
        "research_only": True,
    }
    _atomic_json(target, result)
    return result


def build_rung_e_inputs(
    cache_path: Path,
    pair_ids: Sequence[int],
    mode: str,
    *,
    require_canonical_hash: bool = True,
) -> RungEInputs:
    """Rebuild the exact two planes and fitted descriptors for archive rung E."""

    ids = tuple(int(value) for value in pair_ids)
    if (
        not ids
        or len(ids) > 48
        or any(value < 0 or value >= 600 for value in ids)
        or any(right <= left for left, right in pairwise(ids))
    ):
        raise PredictorFloorError("rung-E pair ids must be 1..48 unique ids in [0,600)")
    if mode not in MODE_BY_ID:
        raise PredictorFloorError("rung-E mode is outside the production registry")
    fields, cache_sha = _load_cache(cache_path.expanduser().resolve(), require_canonical_hash=require_canonical_hash)
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HW[0], camera_w=CAMERA_HW[1], scorer_h=SCORER_HW[0], scorer_w=SCORER_HW[1]
    )
    pairs = [_pair_planes(fields, pair_id, operator) for pair_id in ids]
    y0s = np.stack([pair[0] for pair in pairs])
    y1s = np.stack([pair[1] for pair in pairs])
    descriptors = tuple(_mode_descriptor(y0, y1, mode) for y0, y1 in pairs)
    payload = encode_predictor_residual(y0s, y1s, modes=mode, descriptors=descriptors, pair_ids=ids)
    decoded = decode_predictor_residual(payload)
    if decoded.pair_ids != ids or not np.array_equal(decoded.frame0, y0s) or not np.array_equal(decoded.frame1, y1s):
        raise PredictorFloorError("rung-E production predictor parse-back differs")
    return RungEInputs(
        pair_ids=ids,
        mode=mode,
        frame0_y_planes=y0s,
        frame1_y_planes=y1s,
        descriptors=descriptors,
        cache_sha256=cache_sha,
        predictor_payload_sha256=_sha256_bytes(payload),
    )


def load_prepared_chunk(manifest_path: Path) -> PreparedChunkCustody:
    """Load one settled n<=12 V10 prepare chunk without re-deriving it."""

    target = manifest_path.expanduser().resolve()
    if not target.is_file() or not target.name.endswith(".manifest.json"):
        raise PredictorFloorError("prepared chunk manifest is absent or misnamed")
    try:
        manifest = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictorFloorError("prepared chunk manifest is unreadable") from exc
    pair_ids = manifest.get("pair_ids")
    pair_count = manifest.get("pair_count")
    if (
        manifest.get("schema") != PREPARED_CHUNK_SCHEMA
        or manifest.get("complete") is not True
        or type(pair_count) is not int
        or not isinstance(pair_ids, list)
        or pair_count != len(pair_ids)
        or not 1 <= pair_count <= MAX_CHUNK
        or any(type(value) is not int or value < 0 or value >= 600 for value in pair_ids)
        or any(right <= left for left, right in pairwise(pair_ids))
        or manifest.get("camera_hw") != list(CAMERA_HW)
        or manifest.get("scorer_hw") != list(SCORER_HW)
        or manifest.get("source_cache_sha256") != EXPECTED_CACHE_SHA256
        or manifest.get("y_codec_id") != "predictor-residual-u8.v1"
        or manifest.get("predictor_mode_id") not in MODE_BY_ID
    ):
        raise PredictorFloorError("prepared chunk manifest violates the settled V10 contract")
    prefix = target.name[: -len(".manifest.json")]
    y0_path = target.with_name(f"{prefix}.y0.bin")
    y1_path = target.with_name(f"{prefix}.y1.bin")
    predictor_path = target.with_name(f"{prefix}.predictor.bin")
    for path, byte_field, sha_field in (
        (y0_path, "y0_bytes", "y0_sha256"),
        (y1_path, "y1_bytes", "y1_sha256"),
        (predictor_path, "predictor_bytes", "predictor_sha256"),
    ):
        if (
            not path.is_file()
            or type(manifest.get(byte_field)) is not int
            or path.stat().st_size != manifest[byte_field]
            or _sha256_file(path) != manifest.get(sha_field)
        ):
            raise PredictorFloorError(f"prepared chunk {path.name} custody drifted")
    shape = (pair_count, *SCORER_HW, 3)
    expected_plane_bytes = int(np.prod(shape, dtype=np.int64))
    if manifest["y0_bytes"] != expected_plane_bytes or manifest["y1_bytes"] != expected_plane_bytes:
        raise PredictorFloorError("prepared chunk plane byte geometry drifted")
    y0 = np.frombuffer(y0_path.read_bytes(), dtype=np.uint8).reshape(shape).copy()
    y1 = np.frombuffer(y1_path.read_bytes(), dtype=np.uint8).reshape(shape).copy()
    predictor_payload = predictor_path.read_bytes()
    try:
        decoded = decode_predictor_residual(predictor_payload)
    except Exception as exc:
        raise PredictorFloorError("prepared predictor payload parse-back refused") from exc
    mode = str(manifest["predictor_mode_id"])
    if (
        decoded.pair_ids != tuple(pair_ids)
        or decoded.modes != (mode,) * pair_count
        or not np.array_equal(decoded.frame0, y0)
        or not np.array_equal(decoded.frame1, y1)
    ):
        raise PredictorFloorError("prepared predictor payload differs from its bound planes")
    inputs = RungEInputs(
        pair_ids=tuple(pair_ids),
        mode=mode,
        frame0_y_planes=y0,
        frame1_y_planes=y1,
        descriptors=decoded.descriptors,
        cache_sha256=str(manifest["source_cache_sha256"]),
        predictor_payload_sha256=str(manifest["predictor_sha256"]),
    )
    return PreparedChunkCustody(
        inputs=inputs,
        manifest_path=target,
        manifest_sha256=_sha256_file(target),
        manifest=manifest,
        y0_path=y0_path,
        y1_path=y1_path,
        predictor_path=predictor_path,
    )


def truncate_scorer_plane_predictor_residual(
    inputs: RungEInputs,
    *,
    drop_low_bits: int,
) -> tuple[RungEInputs, dict[str, Any]]:
    """Coarsen exact uint8 scorer-plane residuals toward their V10 predictor."""

    if isinstance(drop_low_bits, bool) or int(drop_low_bits) != drop_low_bits:
        raise PredictorFloorError("banked A/B drop-low-bits must be an exact integer")
    bits = int(drop_low_bits)
    if not 1 <= bits <= 7:
        raise PredictorFloorError("banked A/B drop-low-bits must be in [1,7]")
    rows: list[np.ndarray] = []
    changed_values = 0
    residual_abs_before = 0
    residual_abs_after = 0
    for y0, y1, descriptor in zip(
        inputs.frame0_y_planes,
        inputs.frame1_y_planes,
        inputs.descriptors,
        strict=True,
    ):
        predictor = predict_plane(y0, inputs.mode, descriptor)
        residual = y1.astype(np.int16) - predictor.astype(np.int16)
        magnitude = np.abs(residual).astype(np.int16)
        truncated_magnitude = (magnitude >> bits) << bits
        truncated = np.where(residual < 0, -truncated_magnitude, truncated_magnitude)
        reconstructed = predictor.astype(np.int16) + truncated
        if np.any(reconstructed < 0) or np.any(reconstructed > 255):
            raise PredictorFloorError("banked A/B precision treatment left the uint8 lattice")
        treated = reconstructed.astype(np.uint8)
        changed_values += int(np.count_nonzero(treated != y1))
        residual_abs_before += int(np.abs(residual.astype(np.int32)).sum(dtype=np.int64))
        residual_abs_after += int(np.abs(truncated.astype(np.int32)).sum(dtype=np.int64))
        rows.append(treated)
    frame1 = np.stack(rows)
    payload = encode_predictor_residual(
        inputs.frame0_y_planes,
        frame1,
        modes=inputs.mode,
        descriptors=inputs.descriptors,
        pair_ids=inputs.pair_ids,
    )
    decoded = decode_predictor_residual(payload)
    if not np.array_equal(decoded.frame0, inputs.frame0_y_planes) or not np.array_equal(decoded.frame1, frame1):
        raise PredictorFloorError("banked A/B treatment predictor parse-back differs")
    result = RungEInputs(
        pair_ids=inputs.pair_ids,
        mode=inputs.mode,
        frame0_y_planes=inputs.frame0_y_planes,
        frame1_y_planes=frame1,
        descriptors=inputs.descriptors,
        cache_sha256=inputs.cache_sha256,
        predictor_payload_sha256=_sha256_bytes(payload),
    )
    return result, {
        "operation": "truncate scorer-plane signed predictor residual magnitudes toward zero",
        "drop_low_bits": bits,
        "changed_values": changed_values,
        "changed_fraction": changed_values / int(frame1.size),
        "residual_abs_sum_before": residual_abs_before,
        "residual_abs_sum_after": residual_abs_after,
        "exact_uint8_reachable_plane": True,
        "camera_preimage_secant_equivalence_claim": False,
    }


def projected_action(*, archive_bytes: int, pair_count: int, d_seg: float, d_pose: float) -> dict[str, Any]:
    """Project a subset archive linearly to n600 and apply the exact objective."""

    if type(archive_bytes) is not int or archive_bytes <= 0 or type(pair_count) is not int or pair_count <= 0:
        raise PredictorFloorError("projected action requires positive exact byte/pair counts")
    if not all(math.isfinite(value) and value >= 0.0 for value in (float(d_seg), float(d_pose))):
        raise PredictorFloorError("projected action requires finite nonnegative distortion")
    projected_bytes = (archive_bytes * CONTEST_PAIR_COUNT + pair_count - 1) // pair_count
    distortion_term = 100.0 * float(d_seg) + math.sqrt(10.0 * float(d_pose))
    action = distortion_term + 25.0 * projected_bytes / CONTEST_ARCHIVE_DENOMINATOR
    slack = POINTER_VALUE - distortion_term
    byte_cap = math.floor(slack * CONTEST_ARCHIVE_DENOMINATOR / 25.0) if slack > 0.0 else -1
    return {
        "projection": "ceil(measured subset archive bytes * 600 / measured pair count)",
        "projected_n600_archive_bytes": projected_bytes,
        "measured_subset_d_seg": float(d_seg),
        "measured_subset_d_pose": float(d_pose),
        "distortion_term": distortion_term,
        "projected_exact_objective": action,
        "strict_pointer_byte_cap_at_measured_distortion": byte_cap,
        "projected_beats_pointer": action < POINTER_VALUE,
    }


def _load_scorers(upstream: Path, cpu_threads: int) -> tuple[Any, Any, Any]:
    if not (upstream / "modules.py").is_file():
        raise PredictorFloorError(f"frozen upstream scorer source is absent: {upstream}")
    if not isinstance(cpu_threads, int) or isinstance(cpu_threads, bool) or cpu_threads < 1:
        raise PredictorFloorError("cpu_threads must be a positive exact integer")
    expected_modules = (upstream / "modules.py").resolve()
    loaded = sys.modules.get("modules")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() != expected_modules:
        raise PredictorFloorError("a different modules.py is already imported; scorer custody is ambiguous")
    retained = []
    for entry in sys.path:
        try:
            if Path(entry or ".").resolve() == upstream.resolve():
                continue
        except OSError:
            pass
        retained.append(entry)
    sys.path[:] = [str(upstream.resolve()), *retained]
    try:
        import modules
        import torch
    except (ImportError, OSError) as exc:
        raise PredictorFloorError("frozen CPU scorer import failed") from exc
    if Path(getattr(modules, "__file__", "")).resolve() != expected_modules:
        raise PredictorFloorError("frozen scorer module resolved from the wrong source path")
    DistortionNet = modules.DistortionNet
    posenet_sd_path = modules.posenet_sd_path
    segnet_sd_path = modules.segnet_sd_path
    torch.set_num_threads(int(cpu_threads))
    torch.manual_seed(20260719)
    torch.use_deterministic_algorithms(True)
    distortion = DistortionNet().eval().to("cpu")
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    for parameter in distortion.parameters():
        parameter.requires_grad_(False)
    return distortion.segnet, distortion.posenet, torch


def _score_one_pair(
    segnet: Any,
    posenet: Any,
    torch: Any,
    frame0: np.ndarray,
    frame1: np.ndarray,
    labels: np.ndarray,
    target_pose: np.ndarray,
) -> dict[str, Any]:
    try:
        import einops
    except ImportError as exc:
        raise PredictorFloorError("einops is required by the frozen scorer path") from exc
    pair = torch.from_numpy(np.stack((frame0, frame1), axis=0)[None]).float()
    x = einops.rearrange(pair, "b t h w c -> b t c h w")
    with torch.inference_mode():
        logits = segnet(segnet.preprocess_input(x))[0]
        argmax = logits.argmax(dim=0).cpu().numpy()
        pose_output = posenet(posenet.preprocess_input(x))
        pose = pose_output["pose"] if isinstance(pose_output, dict) else pose_output
        pose6 = pose[0, :6].cpu().numpy().astype(np.float64)
    mismatch = np.asarray(argmax) != np.asarray(labels)
    return {
        "d_seg": float(np.mean(mismatch)),
        "seg_mismatched_pixels": int(np.count_nonzero(mismatch)),
        "d_pose": float(np.mean((pose6 - np.asarray(target_pose, dtype=np.float64)) ** 2)),
        "pose6": pose6.tolist(),
    }


def score_inflated_raw(
    raw_path: Path,
    *,
    pair_ids: Sequence[int],
    cache_path: Path = DEFAULT_CACHE,
    upstream: Path = DEFAULT_UPSTREAM,
    cpu_threads: int = 4,
    require_canonical_hash: bool = True,
    scorer_bundle: tuple[Any, Any, Any] | None = None,
) -> dict[str, Any]:
    """Run the encode-side native-f32 hard oracle on already-inflated raw bytes."""

    ids = tuple(int(value) for value in pair_ids)
    if not ids or len(set(ids)) != len(ids):
        raise PredictorFloorError("hard-oracle pair ids must be nonempty and unique")
    frame_bytes = CAMERA_HW[0] * CAMERA_HW[1] * 3
    expected_bytes = len(ids) * 2 * frame_bytes
    if not raw_path.is_file() or raw_path.stat().st_size != expected_bytes:
        raise PredictorFloorError("inflated raw byte count disagrees with pair/camera geometry")
    fields, cache_sha = _load_cache(cache_path.expanduser().resolve(), require_canonical_hash=require_canonical_hash)
    segnet, posenet, torch = scorer_bundle or _load_scorers(upstream.expanduser().resolve(), cpu_threads)
    rows: list[dict[str, Any]] = []
    with raw_path.open("rb") as handle:
        for pair_id in ids:
            frame0_raw = handle.read(frame_bytes)
            frame1_raw = handle.read(frame_bytes)
            if len(frame0_raw) != frame_bytes or len(frame1_raw) != frame_bytes:
                raise PredictorFloorError("inflated raw truncated within a pair")
            frame0 = np.frombuffer(frame0_raw, dtype=np.uint8).reshape(*CAMERA_HW, 3)
            frame1 = np.frombuffer(frame1_raw, dtype=np.uint8).reshape(*CAMERA_HW, 3)
            row = _score_one_pair(
                segnet,
                posenet,
                torch,
                frame0,
                frame1,
                np.asarray(fields["lstars"][pair_id]),
                np.asarray(fields["gt_poses"][pair_id]),
            )
            row["pair_id"] = pair_id
            rows.append(row)
        if handle.read(1):
            raise PredictorFloorError("inflated raw has trailing data")
    return {
        "receiver_arithmetic": "native_float32_cpu_torch",
        "tie_policy": "first maximum in frozen SegNet class index order",
        "cache_sha256": cache_sha,
        "raw_sha256": _sha256_file(raw_path),
        "pairs": rows,
        "mean_d_seg": float(np.mean([row["d_seg"] for row in rows])),
        "mean_d_pose": float(np.mean([row["d_pose"] for row in rows])),
    }


def _verify_inflated_planes(raw_path: Path, inputs: RungEInputs) -> dict[str, Any]:
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HW[0], camera_w=CAMERA_HW[1], scorer_h=SCORER_HW[0], scorer_w=SCORER_HW[1]
    )
    frame_bytes = CAMERA_HW[0] * CAMERA_HW[1] * 3
    verified = 0
    with raw_path.open("rb") as handle:
        for index in range(len(inputs.pair_ids)):
            for expected in (inputs.frame0_y_planes[index], inputs.frame1_y_planes[index]):
                raw = handle.read(frame_bytes)
                if len(raw) != frame_bytes:
                    raise PredictorFloorError("rung-E raw truncated before plane verification")
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(*CAMERA_HW, 3)
                numerators, denominator = operator.apply_numerators(frame)
                if not np.array_equal(numerators, expected.astype(numerators.dtype) * denominator):
                    raise PredictorFloorError("rung-E factor-2 exact numerator equality failed")
                verified += int(expected.size)
        if handle.read(1):
            raise PredictorFloorError("rung-E raw has trailing data")
    return {"numerator_equal_values": verified, "both_planes_exact": True}


def _open_existing_archive(archive_path: Path) -> bytes:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if len(infos) != 1 or infos[0].filename != "0.bin":
                raise PredictorFloorError("existing rung-E archive member set differs")
            return archive.read(infos[0])
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise PredictorFloorError("existing rung-E archive cannot be parsed") from exc


def _completed_inflate_raw_path(inflate_result: Any) -> Path:
    """Close the typed production-receiver completion seam before scoring."""

    try:
        completed = inflate_result.completed
        raw_path = inflate_result.raw_path
    except AttributeError as exc:
        raise PredictorFloorError("rung-E inflate result schema drift") from exc
    if type(completed) is not bool or not completed or not isinstance(raw_path, Path):
        raise PredictorFloorError("rung-E inflate did not complete")
    return raw_path


def _run_banked_ab_arm(
    *,
    arm_id: str,
    inputs: RungEInputs,
    output_root: Path,
    cache_path: Path,
    upstream: Path,
    cpu_threads: int,
    resume: bool,
    require_canonical_hash: bool,
    scorer_bundle: tuple[Any, Any, Any],
) -> dict[str, Any]:
    """Build, inflate, verify, and hard-score one matched banked A/B arm."""

    from tac.witness_dsl.v10_production_receiver import (
        PREDICTOR_RESIDUAL_Y_CODEC_ID,
        build_packet,
        build_production_archive,
        decode_y_plane_pair,
        inflate_archive,
        parse_packet,
    )

    if arm_id not in {"control", "precision_drop"}:
        raise PredictorFloorError("banked A/B arm id is outside the closed registry")
    arm_root = output_root / arm_id
    arm_root.mkdir(parents=True, exist_ok=True)
    packet = build_packet(
        inputs.frame1_y_planes,
        camera_height=CAMERA_HW[0],
        camera_width=CAMERA_HW[1],
        y_codec_id=PREDICTOR_RESIDUAL_Y_CODEC_ID,
        frame0_y_planes=inputs.frame0_y_planes,
        predictor_modes=inputs.mode,
        predictor_descriptors=inputs.descriptors,
        predictor_pair_ids=inputs.pair_ids,
    )
    parsed = parse_packet(packet)
    decoded = decode_y_plane_pair(parsed)
    if not np.array_equal(decoded.frame0, inputs.frame0_y_planes) or not np.array_equal(
        decoded.frame1, inputs.frame1_y_planes
    ):
        raise PredictorFloorError(f"banked A/B {arm_id} packet parse-back differs")
    archive_path = arm_root / "archive.zip"
    manifest_path = arm_root / "archive.zip.manifest.json"
    if archive_path.exists() or manifest_path.exists():
        if not resume or not archive_path.is_file() or not manifest_path.is_file():
            raise PredictorFloorError(f"banked A/B {arm_id} archive state requires --resume")
        if _open_existing_archive(archive_path) != packet:
            raise PredictorFloorError(f"banked A/B {arm_id} preserved packet differs")
        try:
            archive_manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise PredictorFloorError(f"banked A/B {arm_id} archive manifest is unreadable") from exc
        archive_bytes = archive_path.stat().st_size
        archive_sha = _sha256_file(archive_path)
        if (
            archive_manifest.get("archive_bytes") != archive_bytes
            or archive_manifest.get("archive_sha256") != archive_sha
        ):
            raise PredictorFloorError(f"banked A/B {arm_id} archive custody drifted")
    else:
        built = build_production_archive(
            inputs.frame1_y_planes,
            archive_path=archive_path,
            camera_height=CAMERA_HW[0],
            camera_width=CAMERA_HW[1],
            y_codec_id=PREDICTOR_RESIDUAL_Y_CODEC_ID,
            frame0_y_planes=inputs.frame0_y_planes,
            predictor_modes=inputs.mode,
            predictor_descriptors=inputs.descriptors,
            predictor_pair_ids=inputs.pair_ids,
            manifest_path=manifest_path,
        )
        archive_bytes, archive_sha = built.archive_bytes, built.archive_sha256
    names_path = arm_root / "video_names.txt"
    _write_once_or_equal(names_path, f"v10-banked-ab-{arm_id}.mp4\n".encode())
    inflate_result = inflate_archive(arm_root, arm_root / "inflated", names_path)
    raw_path = _completed_inflate_raw_path(inflate_result)
    numerator_proof = _verify_inflated_planes(raw_path, inputs)
    hard_oracle = score_inflated_raw(
        raw_path,
        pair_ids=inputs.pair_ids,
        cache_path=cache_path,
        upstream=upstream,
        cpu_threads=cpu_threads,
        require_canonical_hash=require_canonical_hash,
        scorer_bundle=scorer_bundle,
    )
    stage = {
        "schema": SCHEMA_BANKED_AB_STAGE,
        "arm_id": arm_id,
        "pair_ids": list(inputs.pair_ids),
        "archive": {"bytes": archive_bytes, "sha256": archive_sha},
        "packet": {"bytes": len(packet), "sha256": parsed.packet_sha256},
        "predictor_payload_sha256": inputs.predictor_payload_sha256,
        "frame0_y_sha256": _sha256_array(inputs.frame0_y_planes),
        "frame1_y_sha256": _sha256_array(inputs.frame1_y_planes),
        "inflated": {
            "bytes": inflate_result.raw_bytes,
            "sha256": inflate_result.raw_sha256,
        },
        "integer_numerator_proof": numerator_proof,
        "hard_oracle": hard_oracle,
        "stage_complete": True,
    }
    _write_once_or_equal(
        arm_root / "stage-receipt.json",
        json.dumps(stage, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n",
    )
    return stage


def run_banked_ab(args: argparse.Namespace) -> dict[str, Any]:
    """Run a same-harness receiver-closed control/precision A/B on one banked chunk."""

    receipt_path = _durable_path(args.receipt, "banked A/B receipt")
    if receipt_path.exists():
        raise PredictorFloorError(f"write-once banked A/B receipt already exists: {receipt_path}")
    custody = load_prepared_chunk(args.prepared_manifest)
    if args.cache.expanduser().resolve() != DEFAULT_CACHE.resolve() and not args.allow_noncanonical_cache:
        raise PredictorFloorError("banked A/B cache path differs from canonical custody")
    output_root = (
        _ephemeral_output_root(args.output_root)
        if args.ephemeral_output
        else _durable_path(
            args.output_root,
            "banked A/B output-root",
            require_ssd=True,
            allow_local=args.allow_local_output,
        )
    )
    preflight = (
        _ephemeral_storage_preflight(output_root, 2 * len(custody.inputs.pair_ids)) if args.ephemeral_output else None
    )
    marker_path = output_root / _EPHEMERAL_MARKER_NAME
    if args.ephemeral_output:
        if args.resume:
            if not marker_path.is_file() or marker_path.read_bytes() != _EPHEMERAL_MARKER_BYTES:
                raise PredictorFloorError("ephemeral banked A/B resume lacks its ownership marker")
        elif output_root.exists():
            raise PredictorFloorError("new ephemeral banked A/B output root must not already exist")
    elif output_root.exists() and not args.resume:
        raise PredictorFloorError("banked A/B output exists; use --resume or a new output root")
    output_root.mkdir(parents=True, exist_ok=True)
    if args.ephemeral_output:
        _write_once_or_equal(marker_path, _EPHEMERAL_MARKER_BYTES)

    treatment, treatment_operation = truncate_scorer_plane_predictor_residual(
        custody.inputs,
        drop_low_bits=args.drop_low_bits,
    )
    scorer_bundle = _load_scorers(args.upstream.expanduser().resolve(), args.cpu_threads)
    arm_arguments = {
        "output_root": output_root,
        "cache_path": args.cache.expanduser().resolve(),
        "upstream": args.upstream.expanduser().resolve(),
        "cpu_threads": args.cpu_threads,
        "resume": args.resume,
        "require_canonical_hash": not args.allow_noncanonical_cache,
        "scorer_bundle": scorer_bundle,
    }
    control = _run_banked_ab_arm(arm_id="control", inputs=custody.inputs, **arm_arguments)
    treated = _run_banked_ab_arm(arm_id="precision_drop", inputs=treatment, **arm_arguments)
    control_projection = projected_action(
        archive_bytes=control["archive"]["bytes"],
        pair_count=len(custody.inputs.pair_ids),
        d_seg=control["hard_oracle"]["mean_d_seg"],
        d_pose=control["hard_oracle"]["mean_d_pose"],
    )
    treatment_projection = projected_action(
        archive_bytes=treated["archive"]["bytes"],
        pair_count=len(custody.inputs.pair_ids),
        d_seg=treated["hard_oracle"]["mean_d_seg"],
        d_pose=treated["hard_oracle"]["mean_d_pose"],
    )
    receipt = {
        "schema": SCHEMA_BANKED_AB,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "axis": AXIS.replace("subset", f"banked n{len(custody.inputs.pair_ids)} hard-oracle A/B"),
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": POINTER,
            "pointer_moved": False,
            "verdict_scope": "one settled banked chunk; local native-f32 CPU hard oracle; linear n600 byte projection",
        },
        "labels": {
            "MEASURED": [
                "actual production archive bytes",
                "production packet parse-back and receiver inflation",
                "factor-2 exact integer numerator equality",
                "native-f32 frozen CPU-Torch d_seg/d_pose",
            ],
            "DERIVED": [
                "ceil-linear n600 archive-byte projection",
                "exact contest objective formula evaluated on projected bytes and measured subset distortion",
            ],
            "INFERRED": [],
        },
        "source_prepared_chunk": {
            "manifest": str(custody.manifest_path),
            "manifest_sha256": custody.manifest_sha256,
            "y0": {"path": str(custody.y0_path), "sha256": custody.manifest["y0_sha256"]},
            "y1": {"path": str(custody.y1_path), "sha256": custody.manifest["y1_sha256"]},
            "predictor": {
                "path": str(custody.predictor_path),
                "sha256": custody.manifest["predictor_sha256"],
            },
            "source_cache_sha256": custody.inputs.cache_sha256,
            "pair_ids": list(custody.inputs.pair_ids),
        },
        "treatment_operation": treatment_operation,
        "arms": {
            "control": {**control, "projection": control_projection},
            "precision_drop": {**treated, "projection": treatment_projection},
        },
        "treatment_minus_control": {
            "archive_bytes": treated["archive"]["bytes"] - control["archive"]["bytes"],
            "mean_d_seg": treated["hard_oracle"]["mean_d_seg"] - control["hard_oracle"]["mean_d_seg"],
            "mean_d_pose": treated["hard_oracle"]["mean_d_pose"] - control["hard_oracle"]["mean_d_pose"],
            "projected_exact_objective": (
                treatment_projection["projected_exact_objective"] - control_projection["projected_exact_objective"]
            ),
        },
        "frontier_verdict": {
            "control_projected_beats_pointer": control_projection["projected_beats_pointer"],
            "treatment_projected_beats_pointer": treatment_projection["projected_beats_pointer"],
            "missing_compact_predictor_description": True,
            "no_n600_launch_authorized": True,
        },
        "decode_import_boundary": "production inflate imports no scorer/Torch/source bank; hard oracle ran afterward encode-side",
        "artifact_lifecycle": {
            "ephemeral_output": bool(args.ephemeral_output),
            "temporary_paths_redacted": bool(args.ephemeral_output),
            "retained": not args.ephemeral_output,
            "cleanup_completed": False if args.ephemeral_output else None,
            "storage_preflight": preflight,
            "per_arm_stage_receipts_preserved_until_success": True,
            "resume_reopens_exact_archive_and_receiver_pair_stages": True,
            "rebuildable_from": {
                "prepared_manifest_sha256": custody.manifest_sha256,
                "tool_sha256": _sha256_file(Path(__file__).resolve()),
                "codec_sha256": _sha256_file(SRC / "tac/codec/v10_predictor_residual.py"),
                "receiver_sha256": _sha256_file(SRC / "tac/witness_dsl/v10_production_receiver.py"),
            },
            "cleanup_rule": "durable redacted receipt precedes deletion of only the marker-owned pact-rung-e-* temp tree",
        },
        "research_only": True,
    }
    _atomic_json(receipt_path, receipt)
    if args.ephemeral_output:
        _cleanup_ephemeral_output(output_root)
        receipt["artifact_lifecycle"]["cleanup_completed"] = True
        _atomic_json(receipt_path, receipt)
    return receipt


def run_rung_e(args: argparse.Namespace) -> dict[str, Any]:
    """Build, inflate, exact-verify, and hard-score one n48 production archive."""

    from tac.witness_dsl.v10_production_receiver import (
        PREDICTOR_RESIDUAL_Y_CODEC_ID,
        build_packet,
        build_production_archive,
        decode_y_plane_pair,
        inflate_archive,
        parse_packet,
    )

    ephemeral = bool(args.ephemeral_output)
    output_root = (
        _ephemeral_output_root(args.output_root)
        if ephemeral
        else _durable_path(
            args.output_root, "rung-E output-root", require_ssd=True, allow_local=args.allow_local_output
        )
    )
    receipt_path = _durable_path(args.receipt, "rung-E receipt")
    composed_path = _durable_path(args.composed_receipt, "rung-E composed receipt")
    if receipt_path.exists():
        raise PredictorFloorError(f"write-once rung-E receipt already exists: {receipt_path}")
    try:
        composed = json.loads(composed_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictorFloorError("composed predictor receipt is unreadable") from exc
    if composed.get("schema") != SCHEMA_COMPOSED or composed.get("pair_ids") != list(N48_PAIRS):
        raise PredictorFloorError("rung-E requires the exact canonical n48 composed receipt")
    mode = composed.get("best_predictor", {}).get("mode")
    if mode not in MODE_BY_ID:
        raise PredictorFloorError("composed receipt lacks a production-registry best predictor")
    composed_modes = composed.get("modes")
    if not isinstance(composed_modes, list) or {
        row.get("mode") for row in composed_modes if isinstance(row, dict)
    } != set(MODES):
        raise PredictorFloorError("composed receipt predictor table is malformed")
    recomputed_best = min(
        composed_modes,
        key=lambda row: (
            row["full_representation"]["production_brotli_q11_archive_section_bytes"],
            row["conditional_representation"]["direct_global_coder_ab"]["zstd_19_bytes"],
            MODES.index(row["mode"]),
        ),
    )["mode"]
    if mode != recomputed_best:
        raise PredictorFloorError("composed receipt best predictor disagrees with its measured table")
    ephemeral_preflight = _ephemeral_storage_preflight(output_root, len(N48_PAIRS)) if ephemeral else None
    marker_path = output_root / _EPHEMERAL_MARKER_NAME
    if ephemeral:
        if args.resume:
            if not marker_path.is_file() or marker_path.read_bytes() != _EPHEMERAL_MARKER_BYTES:
                raise PredictorFloorError("ephemeral rung-E resume lacks its exact ownership marker")
        elif output_root.exists():
            raise PredictorFloorError("new ephemeral rung-E output root must not already exist")
    output_root.mkdir(parents=True, exist_ok=True)
    if ephemeral:
        _write_once_or_equal(marker_path, _EPHEMERAL_MARKER_BYTES)
    inputs = build_rung_e_inputs(
        args.cache,
        N48_PAIRS,
        mode,
        require_canonical_hash=not args.allow_noncanonical_cache,
    )
    packet = build_packet(
        inputs.frame1_y_planes,
        camera_height=CAMERA_HW[0],
        camera_width=CAMERA_HW[1],
        y_codec_id=PREDICTOR_RESIDUAL_Y_CODEC_ID,
        frame0_y_planes=inputs.frame0_y_planes,
        predictor_modes=mode,
        predictor_descriptors=inputs.descriptors,
        predictor_pair_ids=inputs.pair_ids,
    )
    parsed = parse_packet(packet)
    decoded = decode_y_plane_pair(parsed)
    if not np.array_equal(decoded.frame0, inputs.frame0_y_planes) or not np.array_equal(
        decoded.frame1, inputs.frame1_y_planes
    ):
        raise PredictorFloorError("production archive packet parse-back differs before write")
    archive_path = output_root / "archive.zip"
    manifest_path = output_root / "archive.zip.manifest.json"
    if args.resume:
        if not archive_path.is_file() or not manifest_path.is_file():
            raise PredictorFloorError("rung-E --resume requires preserved archive and manifest")
        if _open_existing_archive(archive_path) != packet:
            raise PredictorFloorError("rung-E preserved archive differs from deterministic rebuild")
        archive_bytes = archive_path.stat().st_size
        archive_sha = _sha256_file(archive_path)
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise PredictorFloorError("rung-E preserved archive manifest is unreadable") from exc
        if manifest.get("archive_bytes") != archive_bytes or manifest.get("archive_sha256") != archive_sha:
            raise PredictorFloorError("rung-E preserved archive manifest custody drift")
    else:
        if archive_path.exists() or manifest_path.exists():
            raise PredictorFloorError("rung-E output exists; use --resume or a new output root")
        built = build_production_archive(
            inputs.frame1_y_planes,
            archive_path=archive_path,
            camera_height=CAMERA_HW[0],
            camera_width=CAMERA_HW[1],
            y_codec_id=PREDICTOR_RESIDUAL_Y_CODEC_ID,
            frame0_y_planes=inputs.frame0_y_planes,
            predictor_modes=mode,
            predictor_descriptors=inputs.descriptors,
            predictor_pair_ids=inputs.pair_ids,
            manifest_path=manifest_path,
        )
        archive_bytes, archive_sha = built.archive_bytes, built.archive_sha256
    names_path = output_root / "video_names.txt"
    _write_once_or_equal(names_path, b"v10-rung-e.mp4\n")
    inflated_root = output_root / "inflated"
    inflate_result = inflate_archive(output_root, inflated_root, names_path)
    inflated_raw_path = _completed_inflate_raw_path(inflate_result)
    numerator_proof = _verify_inflated_planes(inflated_raw_path, inputs)
    hard_oracle = score_inflated_raw(
        inflated_raw_path,
        pair_ids=N48_PAIRS,
        cache_path=args.cache,
        upstream=args.upstream,
        cpu_threads=args.cpu_threads,
        require_canonical_hash=not args.allow_noncanonical_cache,
    )
    receipt = {
        "schema": SCHEMA_RUNG_E,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "axis": AXIS.replace("subset", "n48 hard-oracle"),
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": POINTER,
            "pointer_moved": False,
            "verdict_scope": "rung-E real n48 local CPU hard-oracle archive; not contest CPU/CUDA",
        },
        "labels": {
            "MEASURED": ["actual archive bytes", "native-f32 CPU hard-oracle d_seg/d_pose", "inflated raw hash"],
            "DERIVED": ["best predictor selected by composed n48 receipt", "exact factor-2 numerator equality"],
            "INFERRED": [],
        },
        "pair_ids": list(N48_PAIRS),
        "mode": mode,
        "archive": {
            "path": None if ephemeral else str(archive_path),
            "bytes": archive_bytes,
            "sha256": archive_sha,
            "retained": not ephemeral,
        },
        "predictor_payload_sha256": inputs.predictor_payload_sha256,
        "inflated": {
            "raw_path": None if ephemeral else str(inflate_result.raw_path),
            "raw_bytes": inflate_result.raw_bytes,
            "raw_sha256": inflate_result.raw_sha256,
            "retained": not ephemeral,
        },
        "integer_numerator_proof": numerator_proof,
        "hard_oracle": hard_oracle,
        "source_composed_receipt": {
            "path": str(composed_path),
            "sha256": _sha256_file(composed_path),
        },
        "cache_custody": {"path": str(args.cache.resolve()), "sha256": inputs.cache_sha256},
        "decode_import_boundary": "production inflate imports no scorer/Torch/source cache; hard oracle ran afterward encode-side",
        "artifact_lifecycle": {
            "ephemeral_output": ephemeral,
            "temporary_paths_redacted": ephemeral,
            "retained": not ephemeral,
            "cleanup_completed": False if ephemeral else None,
            "storage_preflight": ephemeral_preflight,
            "rebuildable_from": {
                "cache_sha256": inputs.cache_sha256,
                "composed_receipt_sha256": _sha256_file(composed_path),
                "tool_sha256": _sha256_file(Path(__file__).resolve()),
                "codec_sha256": _sha256_file(SRC / "tac/codec/v10_predictor_residual.py"),
            },
            "cleanup_rule": "durable redacted receipt is written before deleting only the validated pact-rung-e-* temp tree",
        },
        "research_only": True,
    }
    _atomic_json(receipt_path, receipt)
    if ephemeral:
        _cleanup_ephemeral_output(output_root)
        receipt["artifact_lifecycle"]["cleanup_completed"] = True
        _atomic_json(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser("measure", help="measure one at-most-12-pair chunk")
    measure.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    measure.add_argument("--sacred", type=Path, default=DEFAULT_SACRED)
    selection = measure.add_mutually_exclusive_group(required=True)
    selection.add_argument("--pairs", type=int, nargs="+")
    selection.add_argument("--chunk-index", type=int, choices=range(4))
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--state", type=Path, required=True)
    measure.add_argument("--stage-dir", type=Path, required=True)
    measure.add_argument("--zstd-binary", type=Path, default=Path(shutil.which("zstd") or "zstd"))
    measure.add_argument("--resume", action="store_true")
    measure.add_argument("--allow-noncanonical-cache", action="store_true", help=argparse.SUPPRESS)

    compose = sub.add_parser("compose", help="compose exactly four canonical n12 chunks")
    compose.add_argument("--receipts", type=Path, nargs="+", required=True)
    compose.add_argument("--output", type=Path, required=True)

    rung = sub.add_parser("rung-e", help="build, inflate, and native-f32 score the n48 best predictor")
    rung.add_argument("--composed-receipt", type=Path, required=True)
    rung.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    rung.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    rung.add_argument("--output-root", type=Path, required=True)
    rung.add_argument("--receipt", type=Path, required=True)
    rung.add_argument("--cpu-threads", type=int, default=4)
    rung.add_argument("--resume", action="store_true")
    output_policy = rung.add_mutually_exclusive_group()
    output_policy.add_argument("--allow-local-output", action="store_true")
    output_policy.add_argument("--ephemeral-output", action="store_true")
    rung.add_argument("--allow-noncanonical-cache", action="store_true", help=argparse.SUPPRESS)

    banked = sub.add_parser("banked-ab", help="run one matched banked n<=12 production-receiver A/B")
    banked.add_argument("--prepared-manifest", type=Path, required=True)
    banked.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    banked.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    banked.add_argument("--output-root", type=Path, required=True)
    banked.add_argument("--receipt", type=Path, required=True)
    banked.add_argument("--drop-low-bits", type=int, default=1, choices=range(1, 8))
    banked.add_argument("--cpu-threads", type=int, default=4)
    banked.add_argument("--resume", action="store_true")
    banked_output = banked.add_mutually_exclusive_group()
    banked_output.add_argument("--allow-local-output", action="store_true")
    banked_output.add_argument("--ephemeral-output", action="store_true")
    banked.add_argument("--allow-noncanonical-cache", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.monotonic()
    if args.command == "measure":
        result = measure_chunk(args)
        target = args.output
    elif args.command == "compose":
        result = compose_chunks(args.receipts, args.output)
        target = args.output
    elif args.command == "rung-e":
        result = run_rung_e(args)
        target = args.receipt
    else:
        result = run_banked_ab(args)
        target = args.receipt
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "output": str(target.expanduser().resolve()),
                "elapsed_seconds": time.monotonic() - started,
                "pointer_moved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
