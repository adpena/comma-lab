#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only M1 checkpoint observer with receiver-closed facet accounting.

The observer snapshots the trainer's complete, stable base raw into certified
observer-owned scratch, streams one bootstrap checkpoint through the real
emitter, parsed R1b4 receiver, and frozen scorers, then freezes a recurring
top-32-plus-seeded-16 cohort.  Only the parsed receiver is scored.  Quantization
parity against the unquantized NumPy emitter is measured, never assumed.

Authority is always ``[macOS-CPU advisory]``.  Rows emitted here are advisory
telemetry and are never score or promotion claims.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.boundary_math.c2_r1b4_curvelet_binding import (  # noqa: E402
    BINDING_BASIS_ID,
    C2R1B4CurveletBinding,
)
from tac.boundary_math.integer_plane_banded_trainer import (  # noqa: E402
    LOGICAL_PAIR_COUNT,
    BandArtifact,
    canonical_json,
)
from tac.boundary_math.integer_plane_emitter import (  # noqa: E402
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    PLANE_COUNT,
    RGB_CHANNELS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    QuotientResidualState,
    factor2_operator,
    numpy_uint8,
)
from tac.optimization.boundary_coordinate_joint_solve import (  # noqa: E402
    apply_boundary_packet,
    decode_boundary_packet,
    encode_boundary_packet,
)
from tac.optimization.uint8_lattice_feasibility import (  # noqa: E402
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)
from tac.witness_dsl.integer_plane_emitter_policy import (  # noqa: E402
    IntegerPlaneEmitterStageCheckpoint,
    PolicyMode,
)

AXIS: Final = "[macOS-CPU advisory]"
ROW_SCHEMA: Final = "m1_banded_checkpoint_facet_observation.v2"
RANK_ROW_SCHEMA: Final = "m1_banded_checkpoint_perpair_rank.v1"
ERROR_SCHEMA: Final = "m1_banded_checkpoint_observer_error.v1"
BASE_SNAPSHOT_SCHEMA: Final = "m1_banded_checkpoint_base_snapshot.v2"
COHORT_SCHEMA: Final = "m1_banded_checkpoint_recurring_cohort.v1"
PREFLIGHT_SCHEMA: Final = "m1_banded_checkpoint_bootstrap_preflight.v1"
CLEANUP_SCHEMA: Final = "m1_banded_checkpoint_scratch_cleanup.v1"
PANEL_PLAN_SCHEMA: Final = "m1_banded_checkpoint_panel_plan.v2"
BREAK_EVEN_EQUATION_ID: Final = "realization_breakeven_bytes_v1"
PANEL_FIX_PAYLOAD_BYTES: Final = 150.0  # operator-specified comparator
EXPECTED_BAND_MANIFEST_SHA256: Final = "2fd10841dc0cb344454e4af55bd8d27e5e1d819a97df3fc03307604dfffcc367"
EXPECTED_GT_CACHE_SHA256: Final = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
PAIR_SAMPLE_SEED: Final = 1234
FULL_BOOTSTRAP_SIZE: Final = LOGICAL_PAIR_COUNT
FALLBACK_BOOTSTRAP_SIZE: Final = 128
HARD_PAIR_COUNT: Final = 32
BACKGROUND_PAIR_COUNT: Final = 16
PAIR_SAMPLE_SIZE: Final = HARD_PAIR_COUNT + BACKGROUND_PAIR_COUNT
SCORER_BATCH_SIZE: Final = 16
SEALED_GT_SEGNET_BATCH_SIZE: Final = 32
DEFAULT_POLL_SECONDS: Final = 120.0
CLASS_NAMES: Final = ("Road", "Lane", "Undriv", "Movable", "MyCar")
SSD_ROOTS: Final = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)
CHECKPOINT_GLOB: Final = "*__ipe_stage*_ep*_step*.json"
CHECKPOINT_NAME_RE: Final = re.compile(
    r"^(?P<run>[A-Za-z0-9_.-]+)__ipe_stage(?P<stage>[0-9]{3})_"
    r"(?P<name>[A-Za-z0-9_.-]+)_ep(?P<epoch>[0-9]{6})_"
    r"step(?P<step>[0-9]{12})\.json$"
)
FULL_RAW_SHAPE: Final = (
    LOGICAL_PAIR_COUNT,
    PLANE_COUNT,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    RGB_CHANNELS,
)
FULL_RAW_BYTES: Final = int(np.prod(FULL_RAW_SHAPE))
MAX_OBSERVER_FOOTPRINT_BYTES: Final = 6 << 30
SNAPSHOT_HEADROOM_BYTES: Final = 64 << 20
FULL_BASE_NAME: Final = "base_camera_pairs_n600_scratch.npy"
FULL_BASE_MANIFEST_NAME: Final = "base_camera_pairs_n600_scratch.json"
FALLBACK_BASE_NAME: Final = "base_camera_pairs_n128_scratch.npy"
FALLBACK_BASE_MANIFEST_NAME: Final = "base_camera_pairs_n128_scratch.json"
COHORT_BASE_NAME: Final = "base_camera_pairs_recurring_n48.npy"
COHORT_BASE_MANIFEST_NAME: Final = "base_camera_pairs_recurring_n48.json"
PANEL_BASE_NAME: Final = "base_camera_pairs_panel_derived.npy"
PANEL_BASE_MANIFEST_NAME: Final = "base_camera_pairs_panel_derived.json"
ROWS_NAME: Final = "facets.jsonl"
RANK_ROWS_NAME: Final = "facets_perpair_rank.jsonl"
COHORT_RECEIPT_NAME: Final = "recurring_cohort.json"
PREFLIGHT_RECEIPT_NAME: Final = "bootstrap_preflight.json"
CLEANUP_RECEIPT_NAME: Final = "scratch_cleanup_receipt.jsonl"
PANELS_DIR_NAME: Final = "panels"
ERRORS_NAME: Final = "observer_errors.jsonl"


class ObserverError(RuntimeError):
    """Fail-closed observer input, custody, or equality violation."""


class IncompleteCheckpointError(ObserverError):
    """A checkpoint changed while read or is plausibly still being written."""


@dataclass(frozen=True, slots=True)
class ParsedCheckpoint:
    path: Path
    name: str
    sha256: str
    payload_bytes: int
    run_id: str
    checkpoint: IntegerPlaneEmitterStageCheckpoint

    @property
    def sort_key(self) -> tuple[int, int, int, str]:
        value = self.checkpoint
        return value.stage_index, value.epoch, value.global_step, self.name


@dataclass(frozen=True, slots=True)
class StratumMasks:
    realizable: np.ndarray
    dead_candidate: np.ndarray
    outside_candidate: np.ndarray

    def __post_init__(self) -> None:
        arrays = (self.realizable, self.dead_candidate, self.outside_candidate)
        expected_tail = (SCORER_HEIGHT, SCORER_WIDTH)
        if not arrays or arrays[0].ndim != 3 or arrays[0].shape[1:] != expected_tail:
            raise ObserverError(f"stratum masks must have shape [N,{expected_tail}]")
        expected = arrays[0].shape
        if any(value.dtype != np.bool_ or value.shape != expected for value in arrays):
            raise ObserverError(f"stratum masks must be bool arrays with shape {expected}")
        cover = self.realizable.astype(np.uint8)
        cover += self.dead_candidate.astype(np.uint8)
        cover += self.outside_candidate.astype(np.uint8)
        if not np.all(cover == 1):
            raise ObserverError("stratum masks must be disjoint and exhaustive")


@dataclass(frozen=True, slots=True)
class ObserverConfig:
    checkpoint_dir: Path
    output_dir: Path
    band_manifest: Path
    carrier_binding: Path
    gt_cache: Path
    upstream_dir: Path
    live_base_raw: Path | None
    live_base_scorer_npy: Path | None = None
    poll_seconds: float = DEFAULT_POLL_SECONDS
    once: bool = False


@dataclass(frozen=True, slots=True)
class BatchObservation:
    pair_ids: tuple[int, ...]
    receiver_planes: np.ndarray
    unquantized_emitter_planes: np.ndarray
    changed_pixels: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    frame0_predictions: np.ndarray
    per_pair_d_seg: np.ndarray
    per_pair_d_pose: np.ndarray
    parity_rows: tuple[dict[str, Any], ...]
    factor2_proof_count: int


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    if ".partial" in path.name:
        raise ObserverError(f"refusing to hash partial input: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def pair_sample(
    *,
    seed: int = PAIR_SAMPLE_SEED,
    population: int = LOGICAL_PAIR_COUNT,
    size: int = FALLBACK_BOOTSTRAP_SIZE,
) -> tuple[int, ...]:
    """Return a sealed PCG64 sample in generator order, without sorting."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ObserverError("sample seed must be an integer in [0,2**64)")
    if isinstance(population, bool) or not isinstance(population, int) or population < 1:
        raise ObserverError("sample population must be positive")
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= population:
        raise ObserverError("sample size must be in [1,population]")
    values = np.random.Generator(np.random.PCG64(seed)).choice(population, size=size, replace=False)
    return tuple(int(value) for value in values)


def freeze_recurring_cohort(
    per_pair_d_seg: Mapping[int, float],
    *,
    population_ids: Sequence[int],
    hard_count: int = HARD_PAIR_COUNT,
    background_count: int = BACKGROUND_PAIR_COUNT,
    seed: int = PAIR_SAMPLE_SEED,
) -> tuple[int, ...]:
    """Freeze top-dseg pairs plus seeded complement pairs in deterministic order."""

    population = tuple(int(value) for value in population_ids)
    if len(population) != len(set(population)) or not population:
        raise ObserverError("bootstrap population IDs must be unique and nonempty")
    if set(per_pair_d_seg) != set(population):
        raise ObserverError("per-pair d_seg rows must exactly cover the bootstrap population")
    if hard_count < 1 or background_count < 1 or hard_count + background_count > len(population):
        raise ObserverError("recurring cohort counts do not fit the bootstrap population")
    for pair_id, value in per_pair_d_seg.items():
        if not isinstance(pair_id, int) or isinstance(pair_id, bool):
            raise ObserverError("per-pair d_seg keys must be integer pair IDs")
        if not np.isfinite(value) or value < 0.0:
            raise ObserverError("per-pair d_seg values must be finite and nonnegative")
    hardest = tuple(
        pair_id
        for pair_id, _value in sorted(per_pair_d_seg.items(), key=lambda item: (-float(item[1]), int(item[0])))[
            :hard_count
        ]
    )
    complement = tuple(sorted(set(population) - set(hardest)))
    chosen_positions = np.random.Generator(np.random.PCG64(seed)).choice(
        len(complement), size=background_count, replace=False
    )
    background = tuple(complement[int(position)] for position in chosen_positions)
    result = hardest + background
    if len(result) != hard_count + background_count or len(set(result)) != len(result):
        raise ObserverError("recurring cohort construction produced duplicates")
    return result


def pair_sample_sha256(pair_ids: Sequence[int]) -> str:
    """Bind the ordered sample as canonical JSON, independent of host endian."""

    values = list(pair_ids)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ObserverError("pair IDs must be integers")
    return sha256_bytes(canonical_json(values))


def output_names() -> dict[str, str]:
    """Return the binding observer output names as a pure restart contract."""

    return {
        "facets": ROWS_NAME,
        "per_pair_rank": RANK_ROWS_NAME,
        "cohort": COHORT_RECEIPT_NAME,
        "preflight": PREFLIGHT_RECEIPT_NAME,
        "cleanup": CLEANUP_RECEIPT_NAME,
        "panels": PANELS_DIR_NAME,
        "errors": ERRORS_NAME,
    }


def scoring_geometry_receipt() -> dict[str, Any]:
    """Expose the sealed-cache/candidate batch split without claiming parity."""

    return {
        "observer_decode_batch_size": SCORER_BATCH_SIZE,
        "candidate_segnet_batch_size": SCORER_BATCH_SIZE,
        "candidate_posenet_batch_size": SCORER_BATCH_SIZE,
        "sealed_gt_segnet_batch_size": SEALED_GT_SEGNET_BATCH_SIZE,
        "segnet_batch_geometry_parity": False,
        "d_seg_definition": "sealed_cached_lstars_vs_candidate_batch16_frozen_cpu_segnet_argmax",
        "d_pose_definition": "sealed_cached_gt_poses_vs_candidate_batch16_frozen_cpu_posenet",
        "disposition": "explicit_macOS_advisory_only_no_contest_or_promotion_authority",
    }


def panel_name(checkpoint_sha256: str, pair_id: int) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
        raise ObserverError("panel checkpoint SHA-256 must be lowercase hexadecimal")
    if isinstance(pair_id, bool) or not isinstance(pair_id, int) or not 0 <= pair_id < LOGICAL_PAIR_COUNT:
        raise ObserverError("panel pair ID is out of range")
    return f"m1_{checkpoint_sha256}_pair{pair_id:04d}.png"


def stored_npy_memmap(npz_path: Path, key: str) -> np.memmap:
    """Map one unencrypted ZIP_STORED NPY member without loading the NPZ."""

    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(npz_path) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError as exc:
            raise ObserverError(f"GT cache lacks {member}") from exc
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
            raise ObserverError(f"GT cache member must be unencrypted ZIP_STORED: {member}")
        offset = int(info.header_offset)
    with npz_path.open("rb") as handle:
        handle.seek(offset)
        header = handle.read(30)
        if len(header) != 30:
            raise ObserverError(f"truncated ZIP header: {member}")
        fields = struct.unpack("<IHHHHHIIIHH", header)
        if fields[0] != 0x04034B50:
            raise ObserverError(f"invalid ZIP local header: {member}")
        handle.seek(offset + 30 + int(fields[-2]) + int(fields[-1]))
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ObserverError(f"unsupported NPY version for {member}: {version}")
        data_offset = handle.tell()
    result = np.memmap(
        npz_path,
        mode="r",
        dtype=dtype,
        shape=shape,
        offset=data_offset,
        order="F" if fortran else "C",
    )
    if result.flags.writeable:
        raise ObserverError(f"GT cache member unexpectedly opened writeable: {member}")
    return result


def parse_checkpoint_bytes(path: Path, payload: bytes) -> ParsedCheckpoint:
    """Strictly parse a checkpoint envelope and bind all filename counters."""

    match = CHECKPOINT_NAME_RE.fullmatch(path.name)
    if match is None:
        raise ObserverError(f"checkpoint filename is noncanonical: {path.name}")
    try:
        checkpoint = IntegerPlaneEmitterStageCheckpoint.from_bytes(payload)
    except Exception as exc:
        raise ObserverError(f"strict checkpoint envelope refusal: {exc}") from exc
    expected_name = checkpoint.filename(match.group("run"))
    if path.name != expected_name:
        raise ObserverError("checkpoint filename counters or stage name differ from the sealed envelope")
    if (
        int(match.group("stage")) != checkpoint.stage_index
        or int(match.group("epoch")) != checkpoint.epoch
        or int(match.group("step")) != checkpoint.global_step
    ):
        raise ObserverError("checkpoint filename counters differ from the sealed envelope")
    return ParsedCheckpoint(
        path=path,
        name=path.name,
        sha256=sha256_bytes(payload),
        payload_bytes=len(payload),
        run_id=match.group("run"),
        checkpoint=checkpoint,
    )


def read_stable_checkpoint(path: Path) -> ParsedCheckpoint:
    """Read a checkpoint only when size, mtime, and inode remain stable."""

    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(payload) != after.st_size:
        raise IncompleteCheckpointError(f"checkpoint changed while read: {path.name}")
    if not payload:
        raise IncompleteCheckpointError(f"checkpoint is empty: {path.name}")
    return parse_checkpoint_bytes(path, payload)


def checkpoint_sort_key(path: Path) -> tuple[int, int, int, str]:
    match = CHECKPOINT_NAME_RE.fullmatch(path.name)
    if match is None:
        return sys.maxsize, sys.maxsize, sys.maxsize, path.name
    return (
        int(match.group("stage")),
        int(match.group("epoch")),
        int(match.group("step")),
        path.name,
    )


def _require_canonical_jsonl_record(value: Any, *, schema: str, line_number: int) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ObserverError(f"JSONL line {line_number} has wrong schema")
    return value


def load_processed_checkpoint_sha256s(path: Path) -> set[str]:
    """Load restart state from valid observation rows; corruption refuses."""

    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.endswith(b"\n"):
                raise ObserverError(f"observation JSONL line {line_number} is incomplete")
            encoded = raw[:-1]
            try:
                value = json.loads(encoded.decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ObserverError(f"observation JSONL line {line_number} is invalid") from exc
            row = _require_canonical_jsonl_record(value, schema=ROW_SCHEMA, line_number=line_number)
            if canonical_json(row) != encoded:
                raise ObserverError(f"observation JSONL line {line_number} is noncanonical")
            checkpoint = row.get("checkpoint")
            digest = checkpoint.get("sha256") if isinstance(checkpoint, dict) else None
            if not isinstance(digest, str) or len(digest) != 64:
                raise ObserverError(f"observation JSONL line {line_number} lacks checkpoint SHA")
            if digest in seen:
                raise ObserverError(f"observation JSONL repeats checkpoint SHA at line {line_number}")
            seen.add(digest)
    return seen


def load_per_pair_rank_rows(path: Path) -> tuple[str | None, dict[int, dict[str, Any]]]:
    """Load the resumable bootstrap rank table without accepting duplicates."""

    if not path.exists():
        return None, {}
    checkpoint_sha: str | None = None
    rows: dict[int, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.endswith(b"\n"):
                raise ObserverError(f"rank JSONL line {line_number} is incomplete")
            encoded = raw[:-1]
            try:
                value = json.loads(encoded.decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ObserverError(f"rank JSONL line {line_number} is invalid") from exc
            row = _require_canonical_jsonl_record(value, schema=RANK_ROW_SCHEMA, line_number=line_number)
            if canonical_json(row) != encoded:
                raise ObserverError(f"rank JSONL line {line_number} is noncanonical")
            checkpoint = row.get("checkpoint")
            digest = checkpoint.get("sha256") if isinstance(checkpoint, dict) else None
            pair_id = row.get("pair_id")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ObserverError(f"rank JSONL line {line_number} lacks checkpoint SHA")
            if checkpoint_sha is None:
                checkpoint_sha = digest
            elif checkpoint_sha != digest:
                raise ObserverError("rank JSONL mixes bootstrap checkpoint SHAs")
            if isinstance(pair_id, bool) or not isinstance(pair_id, int) or not 0 <= pair_id < LOGICAL_PAIR_COUNT:
                raise ObserverError(f"rank JSONL line {line_number} has invalid pair ID")
            if pair_id in rows:
                raise ObserverError(f"rank JSONL repeats pair ID at line {line_number}")
            rows[pair_id] = row
    return checkpoint_sha, rows


def load_error_checkpoint_sha256s(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.endswith(b"\n"):
                raise ObserverError(f"error JSONL line {line_number} is incomplete")
            encoded = raw[:-1]
            value = json.loads(encoded.decode("ascii"))
            row = _require_canonical_jsonl_record(value, schema=ERROR_SCHEMA, line_number=line_number)
            if canonical_json(row) != encoded:
                raise ObserverError(f"error JSONL line {line_number} is noncanonical")
            digest = row.get("checkpoint_sha256")
            if isinstance(digest, str) and len(digest) == 64:
                seen.add(digest)
    return seen


def append_canonical_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json(dict(value)) + b"\n"
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically create a canonical receipt, or verify an identical resume."""

    payload = canonical_json(dict(value))
    if path.exists():
        if path.read_bytes() != payload:
            raise ObserverError(f"existing receipt differs from deterministic resume: {path.name}")
        return
    partial = path.parent / f".{path.name}.partial.{os.getpid()}"
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def load_cohort_receipt(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObserverError("recurring cohort receipt is invalid") from exc
    if not isinstance(value, dict) or value.get("schema") != COHORT_SCHEMA or canonical_json(value) != raw:
        raise ObserverError("recurring cohort receipt is noncanonical or has wrong schema")
    pair_ids = value.get("pair_ids")
    if (
        not isinstance(pair_ids, list)
        or len(pair_ids) != PAIR_SAMPLE_SIZE
        or len(set(pair_ids)) != PAIR_SAMPLE_SIZE
        or pair_sample_sha256(pair_ids) != value.get("pair_ids_sha256")
    ):
        raise ObserverError("recurring cohort receipt has invalid pair custody")
    if (
        value.get("hardest_count") != HARD_PAIR_COUNT
        or value.get("background_count") != BACKGROUND_PAIR_COUNT
        or value.get("background_seed") != PAIR_SAMPLE_SEED
        or value.get("hardest_pair_ids") != pair_ids[:HARD_PAIR_COUNT]
        or value.get("background_pair_ids") != pair_ids[HARD_PAIR_COUNT:]
        or value.get("frozen") is not True
    ):
        raise ObserverError("recurring cohort receipt selection contract drift")
    return value


def estimate_pair_plane_code_bytes(codes: np.ndarray) -> dict[str, Any]:
    """Return the labelled zlib-9 estimate for live float32 pair codes."""

    value = np.asarray(codes)
    if value.dtype != np.float32 or value.ndim != 3 or value.shape[1] != PLANE_COUNT:
        raise ObserverError("live pair_plane_codes must be float32 [N,2,K]")
    if not np.isfinite(value).all():
        raise ObserverError("live pair_plane_codes contain nonfinite values")
    payload = np.ascontiguousarray(value, dtype="<f4").tobytes(order="C")
    return {
        "label": "ESTIMATED_ZLIB_LEVEL_9_ON_LIVE_FLOAT32_PAIR_PLANE_CODES",
        "authority": "byte_estimate_not_archive_bytes",
        "source_state": "live_not_ema",
        "dtype": "little_endian_float32",
        "shape": list(value.shape),
        "raw_bytes": len(payload),
        "zlib_level": 9,
        "zlib_level9_bytes": len(zlib.compress(payload, level=9)),
        "raw_sha256": sha256_bytes(payload),
    }


def _tensor(payload: Any, name: str) -> np.ndarray:
    if not isinstance(payload, dict) or set(payload) != {"dtype", "shape", "data"}:
        raise ObserverError(f"{name} tensor payload fields mismatch")
    if payload["dtype"] != "float32":
        raise ObserverError(f"{name} must have dtype float32")
    try:
        shape = tuple(int(value) for value in payload["shape"])
        value = np.asarray(payload["data"], dtype=np.float32).reshape(shape)
    except (TypeError, ValueError) as exc:
        raise ObserverError(f"{name} tensor shape mismatch") from exc
    if not np.isfinite(value).all():
        raise ObserverError(f"{name} contains nonfinite values")
    return value


def checkpoint_residuals(
    checkpoint: IntegerPlaneEmitterStageCheckpoint,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if checkpoint.policy_contract.get("mode") != PolicyMode.BANDED_TRAINING.value:
        raise ObserverError("checkpoint is not an active banded-training state")
    if checkpoint.basis_id != BINDING_BASIS_ID:
        raise ObserverError("checkpoint basis is not the R1b4 curvelet binding")
    live_codes = _tensor(
        checkpoint.live_residual_parameters.get("pair_plane_codes"),
        "live.pair_plane_codes",
    )
    ema_codes = _tensor(checkpoint.ema_shadow.get("pair_plane_codes"), "ema.pair_plane_codes")
    ema_head = _tensor(checkpoint.ema_shadow.get("shared_rgb_head"), "ema.shared_rgb_head")
    expected_codes = (LOGICAL_PAIR_COUNT, PLANE_COUNT, 4)
    if live_codes.shape != expected_codes or ema_codes.shape != expected_codes:
        raise ObserverError(f"checkpoint codes must have shape {expected_codes}")
    if ema_head.shape != (4, RGB_CHANNELS):
        raise ObserverError("checkpoint EMA shared head must have shape [4,3]")
    if np.count_nonzero(ema_codes[:, 0]) != 0:
        raise ObserverError("checkpoint EMA has unconsumed nonzero frame-0 curvelet codes")
    return live_codes, ema_codes, ema_head


def _ensure_beneath(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ObserverError(f"{label} escapes its custody root: {resolved}") from exc
    return resolved


def load_stratum_masks(band: BandArtifact, pair_ids: Sequence[int]) -> tuple[StratumMasks, dict[str, Any]]:
    """Load hash-bound candidate/realizability sidecars from band custody."""

    band_root = band.manifest_path.parent.resolve(strict=True)
    try:
        resize_record = band.custody["ev_selection"]["artifact_records"]["resize"]
        resize_path = _ensure_beneath(band_root / resize_record["path"], band_root, label="resize record")
    except (KeyError, TypeError) as exc:
        raise ObserverError("band lacks resize-stratum custody") from exc
    if sha256_file(resize_path) != resize_record.get("sha256"):
        raise ObserverError("resize-stratum record SHA drift")
    resize_doc = json.loads(resize_path.read_text(encoding="ascii"))
    if resize_doc.get("schema") != "m1_band_resize_realizability.v1":
        raise ObserverError("resize-stratum record schema mismatch")

    def sidecar(name: str) -> tuple[np.memmap, dict[str, Any]]:
        try:
            record = resize_doc["candidate_sidecars"][name]
            path = _ensure_beneath(resize_path.parent / record["path"], band_root, label=name)
        except (KeyError, TypeError) as exc:
            raise ObserverError(f"resize record lacks {name} custody") from exc
        if sha256_file(path) != record.get("sha256"):
            raise ObserverError(f"{name} SHA drift")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(value, np.memmap) or value.flags.writeable:
            raise ObserverError(f"{name} must be a read-only NPY memmap")
        return value, {"path": str(path), "sha256": record["sha256"], "bytes": path.stat().st_size}

    cells, cells_record = sidecar("cells")
    realizable_flags, flags_record = sidecar("pixels")
    if cells.dtype != np.int32 or cells.ndim != 2 or cells.shape[1] != 4:
        raise ObserverError("candidate cells must be int32 [N,4]")
    if realizable_flags.dtype != np.bool_ or realizable_flags.shape != (cells.shape[0],):
        raise ObserverError("pixel realizability must be bool [N]")
    if cells.shape[0] != 38_077:
        raise ObserverError("candidate population differs from sealed 38,077 cells")
    if np.any(cells[:, 1] != 1):
        raise ObserverError("all candidate cells must reside on semantic frame 1")
    sample_position = {pair_id: position for position, pair_id in enumerate(pair_ids)}
    candidate = np.zeros((len(pair_ids), SCORER_HEIGHT, SCORER_WIDTH), dtype=np.bool_)
    realizable = np.zeros_like(candidate)
    seen: set[tuple[int, int, int]] = set()
    for index in range(cells.shape[0]):
        pair_id, _plane, y, x = (int(value) for value in cells[index])
        if not (0 <= pair_id < LOGICAL_PAIR_COUNT and 0 <= y < SCORER_HEIGHT and 0 <= x < SCORER_WIDTH):
            raise ObserverError("candidate cell coordinates are out of range")
        key = (pair_id, y, x)
        if key in seen:
            raise ObserverError("candidate cells contain duplicates")
        seen.add(key)
        position = sample_position.get(pair_id)
        if position is not None:
            candidate[position, y, x] = True
            realizable[position, y, x] = bool(realizable_flags[index])
    dead = candidate & ~realizable
    outside = ~candidate
    masks = StratumMasks(realizable=realizable, dead_candidate=dead, outside_candidate=outside)
    return masks, {
        "resize_record": {
            "path": str(resize_path),
            "bytes": resize_path.stat().st_size,
            "sha256": resize_record["sha256"],
        },
        "candidate_cells": cells_record,
        "pixel_realizable": flags_record,
        "sealed_candidate_count": int(cells.shape[0]),
        "sample_candidate_count": int(np.count_nonzero(candidate)),
        "sample_realizable_count": int(np.count_nonzero(realizable)),
        "sample_dead_candidate_count": int(np.count_nonzero(dead)),
    }


def facet_accounting(
    labels: np.ndarray,
    predictions: np.ndarray,
    changed_pixels: np.ndarray,
    masks: StratumMasks,
) -> dict[str, Any]:
    """Decompose d_seg and changed-pixel residency without double counting."""

    expected = masks.realizable.shape
    if labels.shape != expected or predictions.shape != expected:
        raise ObserverError(f"labels and predictions must have shape {expected}")
    if labels.dtype.kind not in ("i", "u") or predictions.dtype.kind not in ("i", "u"):
        raise ObserverError("labels and predictions must be integer arrays")
    if np.any(labels < 0) or np.any(labels >= len(CLASS_NAMES)):
        raise ObserverError("GT labels contain an unknown class")
    if np.any(predictions < 0) or np.any(predictions >= len(CLASS_NAMES)):
        raise ObserverError("predictions contain an unknown class")
    if changed_pixels.dtype != np.bool_ or changed_pixels.shape != expected:
        raise ObserverError(f"changed_pixels must be bool with shape {expected}")
    mismatch = labels != predictions
    total = int(mismatch.size)

    def error_row(mask: np.ndarray) -> dict[str, Any]:
        count = int(np.count_nonzero(mask))
        errors = int(np.count_nonzero(mismatch & mask))
        return {
            "pixel_count": count,
            "mismatch_count": errors,
            "conditional_error": float(errors / count) if count else 0.0,
            "total_d_seg_contribution": float(errors / total),
        }

    per_class = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        row = error_row(labels == class_id)
        per_class.append({"class_id": class_id, "class_name": class_name, **row})
    strata = {
        "realizable_band": error_row(masks.realizable),
        "structurally_dead_candidate": error_row(masks.dead_candidate),
        "outside_candidate": error_row(masks.outside_candidate),
    }
    if sum(row["mismatch_count"] for row in strata.values()) != int(np.count_nonzero(mismatch)):
        raise ObserverError("stratum mismatch accounting is not exhaustive")
    changed_total = int(np.count_nonzero(changed_pixels))
    changed_inside = int(np.count_nonzero(changed_pixels & masks.realizable))
    changed_outside = changed_total - changed_inside
    return {
        "overall_d_seg": float(np.count_nonzero(mismatch) / total),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "pixel_count": total,
        "per_class": per_class,
        "d_seg_by_candidate_stratum": strata,
        "changed_pixel_residency": {
            "changed_pixel_count": changed_total,
            "inside_realizable_band": changed_inside,
            "outside_realizable_band": changed_outside,
            "inside_fraction_of_changed": float(changed_inside / changed_total) if changed_total else 0.0,
            "outside_fraction_of_changed": float(changed_outside / changed_total) if changed_total else 0.0,
        },
    }


def out_of_band_excursion(
    labels: np.ndarray,
    predictions: np.ndarray,
    receiver_planes: np.ndarray,
    source_planes: np.ndarray,
    radii: np.ndarray,
    realizable: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Measure source/radius violations by canonical GT-to-pred class pair."""

    expected_labels = labels.shape
    expected_planes = (*expected_labels, RGB_CHANNELS)
    if predictions.shape != expected_labels or realizable.shape != expected_labels:
        raise ObserverError("excursion label/mask geometry mismatch")
    if any(value.shape != expected_planes for value in (receiver_planes, source_planes, radii)):
        raise ObserverError("excursion scorer-plane geometry mismatch")
    if receiver_planes.dtype != np.uint8 or source_planes.dtype != np.uint8:
        raise ObserverError("excursion receiver/source planes must be uint8")
    if radii.dtype.kind != "f" or not np.isfinite(radii).all() or np.any(radii < 0.0):
        raise ObserverError("excursion radii must be finite nonnegative floats")
    delta = np.abs(receiver_planes.astype(np.int16) - source_planes.astype(np.int16))
    excursion = realizable & np.any(delta.astype(np.float64) > radii.astype(np.float64), axis=-1)
    total_pixels = int(np.count_nonzero(realizable))
    total_excursions = int(np.count_nonzero(excursion))
    rows = []
    for gt_class, gt_name in enumerate(CLASS_NAMES):
        for emitted_class, emitted_name in enumerate(CLASS_NAMES):
            stratum = realizable & (labels == gt_class) & (predictions == emitted_class)
            pixel_count = int(np.count_nonzero(stratum))
            excursion_count = int(np.count_nonzero(excursion & stratum))
            rows.append(
                {
                    "gt_class_id": gt_class,
                    "gt_class_name": gt_name,
                    "emitted_class_id": emitted_class,
                    "emitted_class_name": emitted_name,
                    "pixel_count": pixel_count,
                    "excursion_count": excursion_count,
                    "excursion_fraction": (float(excursion_count / pixel_count) if pixel_count else 0.0),
                }
            )
    if sum(row["pixel_count"] for row in rows) != total_pixels:
        raise ObserverError("GT-to-pred excursion strata are not exhaustive")
    if sum(row["excursion_count"] for row in rows) != total_excursions:
        raise ObserverError("GT-to-pred excursion counts are not exhaustive")
    return {
        "definition": (
            "realizable pixel excursion iff any receiver RGB value lies outside "
            "BandArtifact.source_planes +/- BandArtifact.radii"
        ),
        "realizable_pixel_count": total_pixels,
        "excursion_pixel_count": total_excursions,
        "excursion_fraction": float(total_excursions / total_pixels) if total_pixels else 0.0,
        "gt_to_emitted_class_rows": rows,
    }, excursion


def top_pair_rows(pair_ids: Sequence[int], values: Sequence[float], *, limit: int = 8) -> list[dict[str, Any]]:
    if len(pair_ids) != len(values):
        raise ObserverError("pair tail IDs and values have different lengths")
    rows = [{"pair_id": int(pair_id), "value": float(value)} for pair_id, value in zip(pair_ids, values, strict=True)]
    if any(not np.isfinite(row["value"]) or row["value"] < 0.0 for row in rows):
        raise ObserverError("pair tail values must be finite and nonnegative")
    return sorted(rows, key=lambda row: (-row["value"], row["pair_id"]))[:limit]


def temporal_argmax_instability(
    pair_ids: Sequence[int], frame0_predictions: np.ndarray, frame1_predictions: np.ndarray
) -> dict[str, Any]:
    expected = (len(pair_ids), SCORER_HEIGHT, SCORER_WIDTH)
    if frame0_predictions.shape != expected or frame1_predictions.shape != expected:
        raise ObserverError("temporal argmax predictions have wrong geometry")
    per_pair = np.mean(frame0_predictions != frame1_predictions, axis=(1, 2), dtype=np.float64)
    return {
        "definition": (
            "pair_internal_consecutive_frame_argmax_instability: fraction of scorer pixels "
            "whose candidate SegNet argmax differs between frame0 and frame1 of the same pair"
        ),
        "aggregate_fraction": float(np.mean(per_pair, dtype=np.float64)),
        "per_pair": [
            {"pair_id": int(pair_id), "instability_fraction": float(value)}
            for pair_id, value in zip(pair_ids, per_pair, strict=True)
        ],
        "worst_pairs": [
            {"pair_id": row["pair_id"], "instability_fraction": row["value"]}
            for row in top_pair_rows(pair_ids, per_pair)
        ],
    }


def mechanism_signature(
    labels: np.ndarray,
    predictions: np.ndarray,
    frame0_predictions: np.ndarray,
) -> dict[str, Any]:
    """Return the operator-bound flip/boundary/flicker mechanism signature."""

    expected = (SCORER_HEIGHT, SCORER_WIDTH)
    if any(value.shape != expected for value in (labels, predictions, frame0_predictions)):
        raise ObserverError("mechanism signature label geometry mismatch")
    flip = labels != predictions
    counts = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(counts, (labels[flip], predictions[flip]), 1)
    flip_count = int(np.count_nonzero(flip))
    composition = counts.astype(np.float64).ravel()
    if flip_count:
        composition /= float(flip_count)
    boundary = np.zeros(expected, dtype=np.bool_)
    boundary[1:] |= labels[1:] != labels[:-1]
    boundary[:-1] |= labels[:-1] != labels[1:]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary_flips = int(np.count_nonzero(flip & boundary))
    boundary_fraction = float(boundary_flips / flip_count) if flip_count else 0.0
    flicker_count = int(np.count_nonzero(frame0_predictions != predictions))
    flicker_flag = int(flicker_count > 0)
    vector = [float(value) for value in composition]
    vector.extend((boundary_fraction, float(flicker_flag)))
    return {
        "definition": "gt_to_emitted_flip_composition_25_plus_boundary_flip_fraction_plus_any_pair_flicker",
        "flip_count": flip_count,
        "class_flip_counts": counts.ravel().tolist(),
        "class_flip_composition": composition.tolist(),
        "boundary_flip_count": boundary_flips,
        "boundary_flip_fraction": boundary_fraction,
        "temporal_flicker_count": flicker_count,
        "temporal_flicker_flag": bool(flicker_flag),
        "vector": vector,
    }


def consume_panel_break_even_equation() -> dict[str, Any]:
    """Consume the registered rate law by ID and invert it for 150 bytes."""

    from tac.canonical_equations.registry import get_equation_by_id

    equation = get_equation_by_id(BREAK_EVEN_EQUATION_ID)
    if equation is None:
        raise ObserverError(f"canonical equation {BREAK_EVEN_EQUATION_ID} is absent")
    payload = equation.to_dict()
    module_name, callable_name = str(payload["python_callable_module_path"]).split(":", 1)
    callable_fn = getattr(__import__(module_name, fromlist=[callable_name]), callable_name)
    low, high = 0.0, 1.0
    while float(callable_fn(high)) < PANEL_FIX_PAYLOAD_BYTES:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if float(callable_fn(middle)) < PANEL_FIX_PAYLOAD_BYTES:
            low = middle
        else:
            high = middle
    score_floor = high
    anchors = payload.get("empirical_anchors") or []
    return {
        "equation_id": BREAK_EVEN_EQUATION_ID,
        "python_callable_module_path": payload["python_callable_module_path"],
        "operator_fix_payload_comparator_bytes": PANEL_FIX_PAYLOAD_BYTES,
        "derived_nonrate_score_floor": score_floor,
        "callable_roundtrip_bytes": float(callable_fn(score_floor)),
        "anchor_ids": [row.get("anchor_id") for row in anchors],
    }


def derive_panel_plan(rank_rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Derive visual tiers from the exhaustive census without sample estimators."""

    ordered_ids = tuple(sorted(int(pair_id) for pair_id in rank_rows))
    if not ordered_ids or len(ordered_ids) != len(rank_rows):
        raise ObserverError("panel census must contain unique integer pair IDs")
    dseg_by_id = {pair_id: float(rank_rows[pair_id]["d_seg"]) for pair_id in ordered_ids}
    if any(not np.isfinite(value) or value < 0.0 for value in dseg_by_id.values()):
        raise ObserverError("panel census d_seg must be finite and nonnegative")
    ranked_ids = tuple(sorted(ordered_ids, key=lambda pair_id: (-dseg_by_id[pair_id], pair_id)))
    dseg = np.asarray([dseg_by_id[pair_id] for pair_id in ranked_ids], dtype=np.float64)
    total_mass = float(np.sum(dseg, dtype=np.float64))
    cumulative = np.cumsum(dseg, dtype=np.float64)

    def coverage_prefix(fraction: float) -> list[int]:
        if total_mass == 0.0:
            return []
        count = int(np.searchsorted(cumulative, fraction * total_mass, side="left")) + 1
        return list(ranked_ids[:count])

    full_mass_ids = coverage_prefix(0.5)
    contact_mass_ids = coverage_prefix(0.9)
    concentration_curve = [
        {
            "rank": rank,
            "pair_id": pair_id,
            "d_seg": dseg_by_id[pair_id],
            "cumulative_dseg_mass_fraction": (float(cumulative[rank - 1] / total_mass) if total_mass else 0.0),
        }
        for rank, pair_id in enumerate(ranked_ids, start=1)
    ]
    ascending = np.sort(dseg)
    if total_mass:
        n = len(ascending)
        gini = float(np.sum((2.0 * np.arange(1, n + 1, dtype=np.float64) - n - 1.0) * ascending) / (n * total_mass))
    else:
        gini = 0.0

    equation = consume_panel_break_even_equation()
    score_floor = float(equation["derived_nonrate_score_floor"])

    mechanism_strata = []
    mechanism_exemplars = []
    class_pair_count = len(CLASS_NAMES) ** 2
    for stratum_index in range(class_pair_count):
        contributors = []
        aggregate_flip_count = 0
        stratum_dseg_mass = 0.0
        for pair_id in ranked_ids:
            signature = rank_rows[pair_id]["mechanism_signature"]
            counts = signature["class_flip_counts"]
            composition = signature["class_flip_composition"]
            if len(counts) != class_pair_count or len(composition) != class_pair_count:
                raise ObserverError("mechanism signature class-pair geometry drift")
            count = int(counts[stratum_index])
            share = float(composition[stratum_index])
            if count > 0:
                contribution = dseg_by_id[pair_id] * share
                contributors.append((pair_id, contribution, count))
                aggregate_flip_count += count
                stratum_dseg_mass += contribution
        if not contributors:
            continue
        exemplar = min(contributors, key=lambda row: (-row[1], -row[2], row[0]))[0]
        mechanism_exemplars.append(exemplar)
        gt_class, emitted_class = divmod(stratum_index, len(CLASS_NAMES))
        score_contribution = 100.0 * stratum_dseg_mass / len(ordered_ids)
        mechanism_strata.append(
            {
                "stratum_id": f"{gt_class}_to_{emitted_class}",
                "gt_class_id": gt_class,
                "gt_class_name": CLASS_NAMES[gt_class],
                "emitted_class_id": emitted_class,
                "emitted_class_name": CLASS_NAMES[emitted_class],
                "participating_pair_count": len(contributors),
                "aggregate_flip_count": aggregate_flip_count,
                "partitioned_d_seg_mass": stratum_dseg_mass,
                "partitioned_score_contribution": score_contribution,
                "fix_ev_above_150_byte_floor": score_contribution >= score_floor,
                "exemplar_pair_id": exemplar,
            }
        )

    full_derived = list(dict.fromkeys([*full_mass_ids, *mechanism_exemplars]))
    contact_derived = list(contact_mass_ids)
    snapshot_derived = list(dict.fromkeys([*full_derived, *contact_derived]))
    bytes_per_pair = FULL_RAW_BYTES // LOGICAL_PAIR_COUNT
    recurring_bytes = PAIR_SAMPLE_SIZE * bytes_per_pair
    envelope_cap = max(
        1,
        int(
            (MAX_OBSERVER_FOOTPRINT_BYTES - FULL_RAW_BYTES - recurring_bytes - SNAPSHOT_HEADROOM_BYTES)
            // bytes_per_pair
        ),
    )
    snapshot_ids = snapshot_derived[:envelope_cap]
    snapshot_set = set(snapshot_ids)
    full_ids = [pair_id for pair_id in full_derived if pair_id in snapshot_set]
    contact_ids = [pair_id for pair_id in contact_derived if pair_id in snapshot_set]
    return {
        "schema": PANEL_PLAN_SCHEMA,
        "selection_method": "exhaustive_census_mass_concentration_plus_direct_class_flip_strata",
        "population_size": len(ordered_ids),
        "total_d_seg_mass": total_mass,
        "concentration": {
            "pairs_for_50pct_mass": len(full_mass_ids),
            "pair_ids_for_50pct_mass": full_mass_ids,
            "pairs_for_90pct_mass": len(contact_mass_ids),
            "pair_ids_for_90pct_mass": contact_mass_ids,
            "gini_concentration_index": gini,
            "curve": concentration_curve,
        },
        "mechanism_strata": mechanism_strata,
        "mechanism_exemplar_pair_ids": list(dict.fromkeys(mechanism_exemplars)),
        "break_even": equation,
        "full_panel_pair_ids_derived": full_derived,
        "contact_sheet_pair_ids_derived": contact_derived,
        "snapshot_pair_ids_derived": snapshot_derived,
        "envelope_cap_pairs": envelope_cap,
        "envelope_capped": len(snapshot_derived) > envelope_cap,
        "full_panel_pair_ids": full_ids,
        "contact_sheet_pair_ids": contact_ids,
        "snapshot_pair_ids": snapshot_ids,
        "pair_ids": snapshot_ids,
        "pair_ids_sha256": pair_sample_sha256(snapshot_ids),
    }


def _project_camera_pairs(camera_pairs: np.ndarray) -> np.ndarray:
    expected_tail = (PLANE_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, RGB_CHANNELS)
    if camera_pairs.dtype != np.uint8 or camera_pairs.ndim != 5 or camera_pairs.shape[1:] != expected_tail:
        raise ObserverError(f"base camera sample must be uint8 [N,{expected_tail}]")
    operator = factor2_operator()
    result = np.empty(
        (camera_pairs.shape[0], PLANE_COUNT, SCORER_HEIGHT, SCORER_WIDTH, RGB_CHANNELS),
        dtype=np.uint8,
    )
    for pair_position in range(camera_pairs.shape[0]):
        for plane in range(PLANE_COUNT):
            numerator, denominator = operator.apply_numerators(camera_pairs[pair_position, plane])
            result[pair_position, plane] = np.clip(
                np.rint(numerator.astype(np.float64) / denominator), 0.0, 255.0
            ).astype(np.uint8)
    return result


def _snapshot_names(kind: str) -> tuple[str, str]:
    names = {
        "full_n600_scratch": (FULL_BASE_NAME, FULL_BASE_MANIFEST_NAME),
        "fallback_n128_scratch": (FALLBACK_BASE_NAME, FALLBACK_BASE_MANIFEST_NAME),
        "recurring_n48": (COHORT_BASE_NAME, COHORT_BASE_MANIFEST_NAME),
        "panel_derived": (PANEL_BASE_NAME, PANEL_BASE_MANIFEST_NAME),
    }
    try:
        return names[kind]
    except KeyError as exc:
        raise ObserverError(f"unknown base snapshot kind: {kind}") from exc


def _snapshot_paths(output_dir: Path, kind: str) -> tuple[Path, Path]:
    array_name, manifest_name = _snapshot_names(kind)
    return output_dir / array_name, output_dir / manifest_name


def _load_snapshot_manifest_only(output_dir: Path, kind: str) -> dict[str, Any] | None:
    _snapshot_path, manifest_path = _snapshot_paths(output_dir, kind)
    if not manifest_path.exists():
        return None
    raw = manifest_path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObserverError(f"base {kind} manifest is invalid") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != BASE_SNAPSHOT_SCHEMA
        or value.get("kind") != kind
        or canonical_json(value) != raw
    ):
        raise ObserverError(f"base {kind} manifest is noncanonical or has wrong schema")
    return value


def load_base_snapshot(
    output_dir: Path, pair_ids: Sequence[int], *, kind: str
) -> tuple[np.memmap, dict[str, Any]] | None:
    snapshot_path, manifest_path = _snapshot_paths(output_dir, kind)
    if not snapshot_path.exists() and not manifest_path.exists():
        return None
    if snapshot_path.exists() != manifest_path.exists():
        raise ObserverError(f"base {kind} snapshot is incomplete")
    manifest = _load_snapshot_manifest_only(output_dir, kind)
    if manifest is None:
        raise ObserverError(f"base {kind} manifest disappeared")
    expected_shape = (
        len(pair_ids),
        PLANE_COUNT,
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        RGB_CHANNELS,
    )
    expected = {
        "kind": kind,
        "pair_ids": list(pair_ids),
        "pair_ids_sha256": pair_sample_sha256(pair_ids),
        "snapshot_shape": list(expected_shape),
        "snapshot_dtype": "uint8",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ObserverError(f"base {kind} snapshot custody drift")
    if snapshot_path.stat().st_size != manifest.get("snapshot_npy_bytes"):
        raise ObserverError(f"base {kind} snapshot byte count drift")
    if sha256_file(snapshot_path) != manifest.get("snapshot_npy_sha256"):
        raise ObserverError(f"base {kind} snapshot SHA drift")
    value = np.load(snapshot_path, mmap_mode="r", allow_pickle=False)
    if (
        not isinstance(value, np.memmap)
        or value.dtype != np.uint8
        or value.shape != expected_shape
        or value.flags.writeable
    ):
        raise ObserverError(f"base {kind} snapshot array geometry/access drift")
    return value, manifest


def _bootstrap_preflight(output_dir: Path) -> tuple[str, tuple[int, ...], dict[str, Any]]:
    full_ids = tuple(range(LOGICAL_PAIR_COUNT))
    projected = FULL_RAW_BYTES + (PAIR_SAMPLE_SIZE * FULL_RAW_BYTES // LOGICAL_PAIR_COUNT)
    projected += SNAPSHOT_HEADROOM_BYTES
    usage = shutil.disk_usage(output_dir)
    accepted = projected <= MAX_OBSERVER_FOOTPRINT_BYTES and usage.free >= projected
    if accepted:
        kind = "full_n600_scratch"
        pair_ids = full_ids
        disposition = "ACCEPT_FULL_N600"
        reason = "projected observer-owned scratch and products fit below 6GiB with free space"
    else:
        kind = "fallback_n128_scratch"
        pair_ids = pair_sample(size=FALLBACK_BOOTSTRAP_SIZE)
        disposition = "REFUSE_FULL_N600_USE_EXPLICIT_N128_FALLBACK"
        reason = (
            "full n600 preservation refused by explicit footprint/free-space preflight; "
            "seeded n128 fallback authorized by binding amendment"
        )
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "created_at_utc": utc_now(),
        "disposition": disposition,
        "reason": reason,
        "max_observer_footprint_bytes": MAX_OBSERVER_FOOTPRINT_BYTES,
        "projected_full_n600_bytes": projected,
        "disk_free_bytes": usage.free,
        "bootstrap_kind": kind,
        "bootstrap_population_size": len(pair_ids),
        "bootstrap_pair_ids": list(pair_ids),
        "bootstrap_pair_ids_sha256": pair_sample_sha256(pair_ids),
        "fallback_seed": PAIR_SAMPLE_SEED if not accepted else None,
        "fallback_generator": "numpy.random.PCG64" if not accepted else None,
        "score_claim": False,
    }
    return kind, pair_ids, receipt


def _load_preflight(output_dir: Path) -> tuple[str, tuple[int, ...], dict[str, Any]] | None:
    path = output_dir / PREFLIGHT_RECEIPT_NAME
    if not path.exists():
        return None
    raw = path.read_bytes()
    try:
        receipt = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObserverError("bootstrap preflight receipt is invalid") from exc
    if canonical_json(receipt) != raw or receipt.get("schema") != PREFLIGHT_SCHEMA:
        raise ObserverError("bootstrap preflight receipt is noncanonical or has wrong schema")
    kind = receipt.get("bootstrap_kind")
    pair_ids = receipt.get("bootstrap_pair_ids")
    if kind not in {"full_n600_scratch", "fallback_n128_scratch"} or not isinstance(pair_ids, list):
        raise ObserverError("bootstrap preflight receipt fields mismatch")
    if pair_sample_sha256(pair_ids) != receipt.get("bootstrap_pair_ids_sha256"):
        raise ObserverError("bootstrap preflight pair custody drift")
    return kind, tuple(pair_ids), receipt


def snapshot_base_population(
    output_dir: Path,
    source_path: Path | None,
    pair_ids: Sequence[int],
    *,
    kind: str,
) -> tuple[np.memmap, dict[str, Any]] | None:
    """Snapshot a stable final raw in <=16-pair batches; disappearance retries."""

    existing = load_base_snapshot(output_dir, pair_ids, kind=kind)
    if existing is not None:
        return existing
    if source_path is None or ".partial" in source_path.name:
        return None
    try:
        source = source_path.expanduser().resolve(strict=True)
        if ".partial" in source.name:
            return None
        before = source.stat()
    except FileNotFoundError:
        return None
    if before.st_size != FULL_RAW_BYTES:
        return None
    try:
        source_sha = sha256_file(source)
        middle = source.stat()
    except FileNotFoundError:
        return None
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (middle.st_dev, middle.st_ino, middle.st_size, middle.st_mtime_ns):
        return None
    full = np.memmap(source, mode="r", dtype=np.uint8, shape=FULL_RAW_SHAPE)
    snapshot_path, manifest_path = _snapshot_paths(output_dir, kind)
    partial = output_dir / f".{snapshot_path.name}.partial.{os.getpid()}"
    if partial.exists():
        partial.unlink()
    completed = False
    try:
        snapshot = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype=np.uint8,
            shape=(len(pair_ids), PLANE_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, RGB_CHANNELS),
        )
        for start in range(0, len(pair_ids), SCORER_BATCH_SIZE):
            stop = min(start + SCORER_BATCH_SIZE, len(pair_ids))
            ids = np.asarray(pair_ids[start:stop], dtype=np.int64)
            snapshot[start:stop] = np.asarray(full[ids])
        snapshot.flush()
        del snapshot, full
        try:
            after = source.stat()
        except FileNotFoundError:
            return None
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            return None
        os.replace(partial, snapshot_path)
        completed = True
        manifest = {
            "schema": BASE_SNAPSHOT_SCHEMA,
            "created_at_utc": utc_now(),
            "kind": kind,
            "pair_ids": list(pair_ids),
            "pair_ids_sha256": pair_sample_sha256(pair_ids),
            "source_path": str(source),
            "source_bytes": before.st_size,
            "source_sha256": source_sha,
            "source_stat": {
                "device": before.st_dev,
                "inode": before.st_ino,
                "mtime_ns": before.st_mtime_ns,
            },
            "snapshot_shape": [
                len(pair_ids),
                PLANE_COUNT,
                CAMERA_HEIGHT,
                CAMERA_WIDTH,
                RGB_CHANNELS,
            ],
            "snapshot_dtype": "uint8",
            "snapshot_path": str(snapshot_path),
            "snapshot_npy_bytes": snapshot_path.stat().st_size,
            "snapshot_npy_sha256": sha256_file(snapshot_path),
            "copy_batch_size_max": SCORER_BATCH_SIZE,
            "read_only_source": True,
        }
        write_canonical_json(manifest_path, manifest)
    finally:
        if partial.exists():
            partial.unlink()
        if not completed and snapshot_path.exists() and not manifest_path.exists():
            snapshot_path.unlink()
    return load_base_snapshot(output_dir, pair_ids, kind=kind)


def snapshot_base_population_from_scorer_npy(
    output_dir: Path,
    source_path: Path | None,
    pair_ids: Sequence[int],
    *,
    kind: str,
) -> tuple[np.memmap, dict[str, Any]] | None:
    """Factor-2 realize a stable materializer scorer-plane NPY, read-only."""

    existing = load_base_snapshot(output_dir, pair_ids, kind=kind)
    if existing is not None:
        return existing
    if source_path is None or ".partial" in source_path.name:
        return None
    try:
        source = source_path.expanduser().resolve(strict=True)
        before = source.stat()
    except FileNotFoundError:
        return None
    source_sha = sha256_file(source)
    try:
        scorer = np.load(source, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ObserverError("base scorer-plane NPY is invalid") from exc
    if (
        not isinstance(scorer, np.memmap)
        or scorer.flags.writeable
        or scorer.dtype != np.uint8
        or scorer.shape != (LOGICAL_PAIR_COUNT, PLANE_COUNT, SCORER_HEIGHT, SCORER_WIDTH, RGB_CHANNELS)
    ):
        raise ObserverError("base scorer-plane NPY geometry/access drift")
    receipt_path = source.with_name("base_scorer_planes.materialization.json")
    try:
        receipt_raw = receipt_path.read_bytes()
        receipt = json.loads(receipt_raw.decode("ascii"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObserverError("base scorer-plane materialization receipt is absent or invalid") from exc
    if (
        receipt.get("schema") != "c2_base_plane_materialization.v1"
        or Path(receipt.get("scorer_planes", "")).resolve() != source
        or receipt.get("scorer_planes_bytes") != before.st_size
        or receipt.get("scorer_planes_sha256") != source_sha
    ):
        raise ObserverError("base scorer-plane materialization custody drift")
    snapshot_path, manifest_path = _snapshot_paths(output_dir, kind)
    partial = output_dir / f".{snapshot_path.name}.partial.{os.getpid()}"
    completed = False
    exact_proofs = 0
    try:
        snapshot = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype=np.uint8,
            shape=(len(pair_ids), PLANE_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, RGB_CHANNELS),
        )
        operator = factor2_operator()
        for start in range(0, len(pair_ids), SCORER_BATCH_SIZE):
            stop = min(start + SCORER_BATCH_SIZE, len(pair_ids))
            for local, pair_id in enumerate(pair_ids[start:stop], start=start):
                for plane in range(PLANE_COUNT):
                    target = np.asarray(scorer[int(pair_id), plane])
                    frame = realize_factor2_uint8_scorer_plane(operator, target)
                    proof = verify_factor2_uint8_scorer_plane(operator, frame, target)
                    if not proof.numerator_exact or not proof.certified_exact:
                        raise ObserverError("base scorer-plane factor-2 realization proof failed")
                    snapshot[local, plane] = frame
                    exact_proofs += 1
        snapshot.flush()
        del snapshot, scorer
        after = source.stat()
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after or sha256_file(source) != source_sha:
            return None
        os.replace(partial, snapshot_path)
        completed = True
        manifest = {
            "schema": BASE_SNAPSHOT_SCHEMA,
            "created_at_utc": utc_now(),
            "kind": kind,
            "pair_ids": list(pair_ids),
            "pair_ids_sha256": pair_sample_sha256(pair_ids),
            "source_kind": "stable_materializer_scorer_planes_npy_factor2_realized",
            "source_path": str(source),
            "source_bytes": before.st_size,
            "source_sha256": source_sha,
            "source_stat": {
                "device": before.st_dev,
                "inode": before.st_ino,
                "mtime_ns": before.st_mtime_ns,
            },
            "source_receipt": {
                "path": str(receipt_path),
                "bytes": len(receipt_raw),
                "sha256": sha256_bytes(receipt_raw),
                "schema": receipt["schema"],
            },
            "factor2_realization": {
                "operator": "factor2_align_corners_false_disjoint_2x2",
                "proof_count": exact_proofs,
                "all_exact": True,
            },
            "snapshot_shape": [
                len(pair_ids),
                PLANE_COUNT,
                CAMERA_HEIGHT,
                CAMERA_WIDTH,
                RGB_CHANNELS,
            ],
            "snapshot_dtype": "uint8",
            "snapshot_path": str(snapshot_path),
            "snapshot_npy_bytes": snapshot_path.stat().st_size,
            "snapshot_npy_sha256": sha256_file(snapshot_path),
            "copy_batch_size_max": SCORER_BATCH_SIZE,
            "read_only_source": True,
        }
        write_canonical_json(manifest_path, manifest)
    finally:
        if partial.exists():
            partial.unlink()
        if not completed and snapshot_path.exists() and not manifest_path.exists():
            snapshot_path.unlink()
    return load_base_snapshot(output_dir, pair_ids, kind=kind)


def snapshot_recurring_from_bootstrap(
    output_dir: Path,
    bootstrap: np.memmap,
    bootstrap_manifest: Mapping[str, Any],
    cohort_ids: Sequence[int],
) -> tuple[np.memmap, dict[str, Any]]:
    existing = load_base_snapshot(output_dir, cohort_ids, kind="recurring_n48")
    if existing is not None:
        return existing
    bootstrap_ids = list(bootstrap_manifest["pair_ids"])
    positions = {pair_id: position for position, pair_id in enumerate(bootstrap_ids)}
    if not set(cohort_ids).issubset(positions):
        raise ObserverError("recurring cohort is not covered by bootstrap base snapshot")
    snapshot_path, manifest_path = _snapshot_paths(output_dir, "recurring_n48")
    partial = output_dir / f".{snapshot_path.name}.partial.{os.getpid()}"
    try:
        snapshot = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype=np.uint8,
            shape=(PAIR_SAMPLE_SIZE, PLANE_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, RGB_CHANNELS),
        )
        for start in range(0, PAIR_SAMPLE_SIZE, SCORER_BATCH_SIZE):
            stop = min(start + SCORER_BATCH_SIZE, PAIR_SAMPLE_SIZE)
            selected = [positions[pair_id] for pair_id in cohort_ids[start:stop]]
            snapshot[start:stop] = np.asarray(bootstrap[np.asarray(selected, dtype=np.int64)])
        snapshot.flush()
        del snapshot
        os.replace(partial, snapshot_path)
        manifest = {
            "schema": BASE_SNAPSHOT_SCHEMA,
            "created_at_utc": utc_now(),
            "kind": "recurring_n48",
            "pair_ids": list(cohort_ids),
            "pair_ids_sha256": pair_sample_sha256(cohort_ids),
            "source_path": bootstrap_manifest["source_path"],
            "source_bytes": bootstrap_manifest["source_bytes"],
            "source_sha256": bootstrap_manifest["source_sha256"],
            "source_stat": bootstrap_manifest["source_stat"],
            "derived_from_snapshot": {
                "path": bootstrap_manifest["snapshot_path"],
                "sha256": bootstrap_manifest["snapshot_npy_sha256"],
                "kind": bootstrap_manifest["kind"],
            },
            "snapshot_shape": [
                PAIR_SAMPLE_SIZE,
                PLANE_COUNT,
                CAMERA_HEIGHT,
                CAMERA_WIDTH,
                RGB_CHANNELS,
            ],
            "snapshot_dtype": "uint8",
            "snapshot_path": str(snapshot_path),
            "snapshot_npy_bytes": snapshot_path.stat().st_size,
            "snapshot_npy_sha256": sha256_file(snapshot_path),
            "copy_batch_size_max": SCORER_BATCH_SIZE,
            "read_only_source": True,
        }
        write_canonical_json(manifest_path, manifest)
    finally:
        if partial.exists():
            partial.unlink()
    loaded = load_base_snapshot(output_dir, cohort_ids, kind="recurring_n48")
    if loaded is None:
        raise ObserverError("recurring base snapshot did not become durable")
    return loaded


def snapshot_panel_from_bootstrap(
    output_dir: Path,
    bootstrap: np.memmap,
    bootstrap_manifest: Mapping[str, Any],
    panel_ids: Sequence[int],
) -> tuple[np.memmap, dict[str, Any]]:
    """Preserve the data-derived visual cohort before full scratch cleanup."""

    existing = load_base_snapshot(output_dir, panel_ids, kind="panel_derived")
    if existing is not None:
        return existing
    positions = {pair_id: position for position, pair_id in enumerate(bootstrap_manifest["pair_ids"])}
    if not panel_ids or not set(panel_ids).issubset(positions):
        raise ObserverError("derived panel cohort is empty or escapes bootstrap coverage")
    snapshot_path, manifest_path = _snapshot_paths(output_dir, "panel_derived")
    partial = output_dir / f".{snapshot_path.name}.partial.{os.getpid()}"
    try:
        snapshot = np.lib.format.open_memmap(
            partial,
            mode="w+",
            dtype=np.uint8,
            shape=(len(panel_ids), PLANE_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, RGB_CHANNELS),
        )
        for start in range(0, len(panel_ids), SCORER_BATCH_SIZE):
            stop = min(start + SCORER_BATCH_SIZE, len(panel_ids))
            selected = [positions[pair_id] for pair_id in panel_ids[start:stop]]
            snapshot[start:stop] = np.asarray(bootstrap[np.asarray(selected, dtype=np.int64)])
        snapshot.flush()
        del snapshot
        os.replace(partial, snapshot_path)
        manifest = {
            "schema": BASE_SNAPSHOT_SCHEMA,
            "created_at_utc": utc_now(),
            "kind": "panel_derived",
            "pair_ids": list(panel_ids),
            "pair_ids_sha256": pair_sample_sha256(panel_ids),
            "source_path": bootstrap_manifest["source_path"],
            "source_bytes": bootstrap_manifest["source_bytes"],
            "source_sha256": bootstrap_manifest["source_sha256"],
            "source_stat": bootstrap_manifest["source_stat"],
            "derived_from_snapshot": {
                "path": bootstrap_manifest["snapshot_path"],
                "sha256": bootstrap_manifest["snapshot_npy_sha256"],
                "kind": bootstrap_manifest["kind"],
            },
            "snapshot_shape": [
                len(panel_ids),
                PLANE_COUNT,
                CAMERA_HEIGHT,
                CAMERA_WIDTH,
                RGB_CHANNELS,
            ],
            "snapshot_dtype": "uint8",
            "snapshot_path": str(snapshot_path),
            "snapshot_npy_bytes": snapshot_path.stat().st_size,
            "snapshot_npy_sha256": sha256_file(snapshot_path),
            "copy_batch_size_max": SCORER_BATCH_SIZE,
            "read_only_source": True,
        }
        write_canonical_json(manifest_path, manifest)
    finally:
        if partial.exists():
            partial.unlink()
    loaded = load_base_snapshot(output_dir, panel_ids, kind="panel_derived")
    if loaded is None:
        raise ObserverError("derived panel base snapshot did not become durable")
    return loaded


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else "UNAVAILABLE"


def _load_distortion_net(upstream: Path) -> tuple[Any, dict[str, Any]]:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    upstream = upstream.expanduser().resolve(strict=True)
    modules_path = upstream / "modules.py"
    pose_path = upstream / "models" / "posenet.safetensors"
    seg_path = upstream / "models" / "segnet.safetensors"
    for path in (modules_path, pose_path, seg_path):
        path.resolve(strict=True)
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from modules import DistortionNet

    model = DistortionNet().eval().to(device="cpu")
    model.load_state_dicts(str(pose_path), str(seg_path), torch.device("cpu"))
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, {
        "class": "upstream.modules.DistortionNet",
        "device": "cpu",
        "scoring_geometry": scoring_geometry_receipt(),
        "torch_threads": 1,
        "modules_py": {"path": str(modules_path), "sha256": sha256_file(modules_path)},
        "posenet": {"path": str(pose_path), "sha256": sha256_file(pose_path)},
        "segnet": {"path": str(seg_path), "sha256": sha256_file(seg_path)},
    }


def _validate_static_custody(
    config: ObserverConfig,
) -> tuple[
    BandArtifact,
    C2R1B4CurveletBinding,
    dict[str, np.memmap],
    dict[str, Any],
]:
    band_path = config.band_manifest.expanduser().resolve(strict=True)
    if sha256_file(band_path) != EXPECTED_BAND_MANIFEST_SHA256:
        raise ObserverError("band manifest differs from the sealed M1 SHA-256")
    band = BandArtifact.load(band_path)
    if band.manifest_sha256 != EXPECTED_BAND_MANIFEST_SHA256:
        raise ObserverError("BandArtifact returned unexpected manifest SHA")
    binding = C2R1B4CurveletBinding.load(config.carrier_binding)
    if binding.band_manifest_sha256 != EXPECTED_BAND_MANIFEST_SHA256:
        raise ObserverError("R1b4 binding is not bound to the sealed M1 band")
    gt_path = config.gt_cache.expanduser().resolve(strict=True)
    gt_sha = sha256_file(gt_path)
    if gt_sha != EXPECTED_GT_CACHE_SHA256:
        raise ObserverError("GT cache differs from the sealed n600 SHA-256")
    gt = {key: stored_npy_memmap(gt_path, key) for key in ("n_pairs", "gt_f0", "gt_f1", "lstars", "gt_poses")}
    if int(np.asarray(gt["n_pairs"]).reshape(())) != LOGICAL_PAIR_COUNT:
        raise ObserverError("GT cache n_pairs differs from 600")
    if gt["gt_f0"].shape != (LOGICAL_PAIR_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, RGB_CHANNELS):
        raise ObserverError("GT frame-0 geometry mismatch")
    if gt["gt_f1"].shape != gt["gt_f0"].shape or gt["gt_f1"].dtype != np.uint8:
        raise ObserverError("GT frame-1 geometry mismatch")
    if gt["lstars"].shape != (LOGICAL_PAIR_COUNT, SCORER_HEIGHT, SCORER_WIDTH):
        raise ObserverError("GT label geometry mismatch")
    if gt["gt_poses"].shape != (LOGICAL_PAIR_COUNT, 6):
        raise ObserverError("GT pose geometry mismatch")
    gt_custody = {
        "path": str(gt_path),
        "bytes": gt_path.stat().st_size,
        "sha256": gt_sha,
        "access": "ZIP_STORED_member_memmap_no_dense_n600_load",
        "members": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in gt.items()},
    }
    return band, binding, gt, gt_custody


def _checkpoint_custody(
    parsed: ParsedCheckpoint,
    band: BandArtifact,
    binding: C2R1B4CurveletBinding,
) -> None:
    checkpoint = parsed.checkpoint
    custody = checkpoint.rng_state.get("run_custody")
    if not isinstance(custody, dict):
        raise ObserverError("checkpoint lacks run custody")
    if custody.get("band_sha256") != EXPECTED_BAND_MANIFEST_SHA256:
        raise ObserverError("checkpoint band SHA differs from sealed M1 authority")
    if custody.get("source_sha256") != band.source_sha256:
        raise ObserverError("checkpoint band source-plane SHA drift")
    if checkpoint.policy_contract.get("basis") != BINDING_BASIS_ID:
        raise ObserverError("checkpoint policy does not compile to R1b4 curvelet")
    topology_sha = sha256_bytes(binding.coordinate_basis().tobytes(order="C"))
    if checkpoint.topology_state_sha256 != topology_sha:
        raise ObserverError("checkpoint topology differs from the loaded R1b4 binding")


def _prepare_receiver_packet(
    parsed: ParsedCheckpoint, binding: C2R1B4CurveletBinding
) -> tuple[np.ndarray, np.ndarray, Any, bytes, dict[str, Any]]:
    _live_codes, ema_codes, ema_head = checkpoint_residuals(parsed.checkpoint)
    packet_payload, export_receipt = binding.export_packet(ema_codes, ema_head)
    packet = decode_boundary_packet(packet_payload)
    if encode_boundary_packet(packet) != packet_payload:
        raise ObserverError("exported R1b4 packet is not parse/re-encode identical")
    return ema_codes, ema_head, packet, packet_payload, export_receipt


def _score_receiver_batch(
    distortion_net: Any,
    gt: Mapping[str, np.memmap],
    pair_ids: Sequence[int],
    candidate_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score only realized parsed-receiver bytes and return exact per-pair rows."""

    import torch

    ids = np.asarray(pair_ids, dtype=np.int64)
    candidate_tensor = torch.from_numpy(candidate_camera)

    with torch.inference_mode():
        candidate_seg_input = distortion_net.segnet.preprocess_input(
            candidate_tensor[:, -1].permute(0, 3, 1, 2).float()[:, None]
        )
        candidate_seg_out = distortion_net.segnet(candidate_seg_input)
        del candidate_seg_input
        candidate_pose_input = distortion_net.posenet.preprocess_input(candidate_tensor.permute(0, 1, 4, 2, 3).float())
        candidate_pose_out = distortion_net.posenet(candidate_pose_input)
        del candidate_pose_input
        frame0_input = distortion_net.segnet.preprocess_input(
            candidate_tensor[:, 0].permute(0, 3, 1, 2).float()[:, None]
        )
        frame0_out = distortion_net.segnet(frame0_input)
    cached_labels = np.asarray(gt["lstars"][ids], dtype=np.int64)
    cached_pose = np.asarray(gt["gt_poses"][ids], dtype=np.float64)
    candidate_pose = candidate_pose_out["pose"][..., :6].detach().cpu().numpy().astype(np.float64, copy=False)
    predictions = candidate_seg_out.argmax(dim=1).cpu().numpy().astype(np.int64, copy=False)
    frame0_predictions = frame0_out.argmax(dim=1).cpu().numpy().astype(np.int64, copy=False)
    per_d_seg = np.mean(cached_labels != predictions, axis=(1, 2), dtype=np.float64)
    per_d_pose = np.mean((cached_pose - candidate_pose) ** 2, axis=1, dtype=np.float64)
    return (
        cached_labels,
        predictions,
        frame0_predictions,
        per_d_seg,
        per_d_pose,
    )


def _segnet_argmax_scorer_planes(distortion_net: Any, planes: np.ndarray) -> np.ndarray:
    """Run production SegNet on already-realized scorer-plane RGB bytes."""

    import torch

    value = np.asarray(planes)
    expected = (value.shape[0], SCORER_HEIGHT, SCORER_WIDTH, RGB_CHANNELS)
    if value.dtype != np.uint8 or value.shape != expected:
        raise ObserverError(f"direct SegNet scorer planes must be uint8 with shape {expected}")
    pair_chw = torch.from_numpy(value).permute(0, 3, 1, 2).float()[:, None]
    with torch.inference_mode():
        scorer_input = distortion_net.segnet.preprocess_input(pair_chw)
        output = distortion_net.segnet(scorer_input)
    return output.argmax(dim=1).cpu().numpy().astype(np.int64, copy=False)


def _evaluate_batch(
    *,
    parsed: ParsedCheckpoint,
    binding: C2R1B4CurveletBinding,
    packet: Any,
    ema_codes: np.ndarray,
    ema_head: np.ndarray,
    base_camera: np.ndarray,
    pair_ids: Sequence[int],
    distortion_net: Any,
    gt: Mapping[str, np.memmap],
) -> BatchObservation:
    if not 1 <= len(pair_ids) <= SCORER_BATCH_SIZE:
        raise ObserverError("observer scorer/decode batch must contain 1..16 pairs")
    ids = np.asarray(pair_ids, dtype=np.int64)
    camera_batch = np.asarray(base_camera)
    base_scorer = _project_camera_pairs(camera_batch)
    structured = binding.structured_state(base_scorer.astype(np.float32))
    residual = QuotientResidualState(
        pair_plane_codes=np.asarray(ema_codes[ids], dtype=np.float32),
        shared_rgb_head=np.asarray(ema_head, dtype=np.float32),
        seed=int(parsed.checkpoint.rng_state["seed"]),
    )
    unquantized_emitter = numpy_uint8(structured, residual, require_distinct_planes=False)
    receiver = base_scorer.copy()
    for local, pair_id in enumerate(ids.tolist()):
        receiver[local, 1] = apply_boundary_packet(base_scorer[local, 1], packet, pair_id)
    candidate_camera = np.empty_like(camera_batch)
    candidate_camera[:, 0] = camera_batch[:, 0]
    changed = np.empty((len(pair_ids), SCORER_HEIGHT, SCORER_WIDTH), dtype=np.bool_)
    parity_rows = []
    operator = factor2_operator()
    for local, pair_id in enumerate(ids.tolist()):
        target = receiver[local, 1]
        frame = realize_factor2_uint8_scorer_plane(operator, target)
        proof = verify_factor2_uint8_scorer_plane(operator, frame, target)
        if not proof.numerator_exact or not proof.certified_exact:
            raise ObserverError("exact factor-2 parsed-receiver proof failed")
        candidate_camera[local, 1] = frame
        changed[local] = np.any(target != base_scorer[local, 1], axis=-1)
        difference = np.abs(unquantized_emitter[local].astype(np.int16) - receiver[local].astype(np.int16))
        parity_rows.append(
            {
                "pair_id": pair_id,
                "exact": bool(np.count_nonzero(difference) == 0),
                "differing_uint8_value_count": int(np.count_nonzero(difference)),
                "max_abs_uint8_difference": int(np.max(difference, initial=0)),
            }
        )
    labels, predictions, frame0_predictions, per_d_seg, per_d_pose = _score_receiver_batch(
        distortion_net, gt, pair_ids, candidate_camera
    )
    return BatchObservation(
        pair_ids=tuple(int(value) for value in pair_ids),
        receiver_planes=receiver[:, 1].copy(),
        unquantized_emitter_planes=unquantized_emitter[:, 1].copy(),
        changed_pixels=changed,
        labels=labels,
        predictions=predictions,
        frame0_predictions=frame0_predictions,
        per_pair_d_seg=per_d_seg,
        per_pair_d_pose=per_d_pose,
        parity_rows=tuple(parity_rows),
        factor2_proof_count=len(pair_ids),
    )


def _render_panel_bytes(
    *,
    pair_id: int,
    checkpoint_sha256: str,
    source_plane: np.ndarray,
    receiver_plane: np.ndarray,
    source_labels: np.ndarray,
    receiver_labels: np.ndarray,
    realizable: np.ndarray,
    dead_candidate: np.ndarray,
    excursion: np.ndarray,
) -> bytes:
    from PIL import Image, ImageDraw

    disagreement = source_labels != receiver_labels
    overlay = (source_plane.astype(np.uint16) // 2).astype(np.uint8)
    overlay[disagreement] = np.asarray([255, 32, 32], dtype=np.uint8)
    residency = np.zeros_like(source_plane)
    residency[realizable & ~excursion] = np.asarray([32, 200, 64], dtype=np.uint8)
    residency[realizable & excursion] = np.asarray([255, 144, 0], dtype=np.uint8)
    residency[dead_candidate] = np.asarray([210, 32, 210], dtype=np.uint8)
    delta = receiver_plane.astype(np.int16) - source_plane.astype(np.int16)
    channel = np.argmax(np.abs(delta), axis=-1)
    signed = np.take_along_axis(delta, channel[..., None], axis=-1)[..., 0]
    heat = np.zeros_like(source_plane)
    heat[..., 0] = np.clip(signed, 0, 255).astype(np.uint8)
    heat[..., 2] = np.clip(-signed, 0, 255).astype(np.uint8)
    header = 22
    canvas = Image.new("RGB", (SCORER_WIDTH * 3, SCORER_HEIGHT + header), "black")
    canvas.paste(Image.fromarray(overlay, mode="RGB"), (0, header))
    canvas.paste(Image.fromarray(residency, mode="RGB"), (SCORER_WIDTH, header))
    canvas.paste(Image.fromarray(heat, mode="RGB"), (SCORER_WIDTH * 2, header))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (4, 4),
        (
            f"pair {pair_id} ckpt {checkpoint_sha256[:12]} | argmax disagreement | "
            "in-band green / out-band orange / dead magenta | signed receiver-source delta"
        ),
        fill="white",
    )
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _contact_thumbnail(source_plane: np.ndarray, source_labels: np.ndarray, receiver_labels: np.ndarray) -> Any:
    """Return an exact /2 nearest-neighbor disagreement thumbnail."""

    from PIL import Image

    disagreement = source_labels != receiver_labels
    overlay = (source_plane.astype(np.uint16) // 2).astype(np.uint8)
    overlay[disagreement] = np.asarray([255, 32, 32], dtype=np.uint8)
    return Image.fromarray(overlay, mode="RGB").resize(
        (SCORER_WIDTH // 2, SCORER_HEIGHT // 2),
        resample=Image.Resampling.NEAREST,
    )


def _write_panel(
    output_dir: Path,
    *,
    pair_id: int,
    checkpoint_sha256: str,
    payload: bytes,
    receiver_argmax: np.ndarray,
    signed_delta: np.ndarray,
) -> dict[str, Any]:
    from PIL import Image

    panels_dir = output_dir / PANELS_DIR_NAME
    panels_dir.mkdir(parents=True, exist_ok=True)
    path = panels_dir / panel_name(checkpoint_sha256, pair_id)
    if path.exists():
        if path.read_bytes() != payload:
            raise ObserverError(f"existing panel differs from deterministic resume: {path.name}")
    else:
        partial = panels_dir / f".{path.name}.partial.{os.getpid()}"
        try:
            with partial.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, path)
        finally:
            if partial.exists():
                partial.unlink()
    stem = path.with_suffix("")
    argmax_path = Path(f"{stem}.receiver_argmax.png")
    delta_path = Path(f"{stem}.signed_delta.npy")
    if receiver_argmax.shape != (SCORER_HEIGHT, SCORER_WIDTH):
        raise ObserverError("panel argmax sidecar geometry mismatch")
    indexed = Image.fromarray(receiver_argmax.astype(np.uint8), mode="P")
    palette = [0, 0, 0, 255, 255, 0, 255, 0, 255, 0, 255, 255, 255, 0, 0]
    indexed.putpalette(palette + [0] * (768 - len(palette)))
    if not argmax_path.exists():
        temporary = panels_dir / f".{argmax_path.name}.partial.{os.getpid()}"
        indexed.save(temporary, format="PNG", optimize=False, compress_level=9)
        os.replace(temporary, argmax_path)
    with Image.open(argmax_path) as loaded:
        roundtrip = np.asarray(loaded, dtype=np.uint8)
    if not np.array_equal(roundtrip, receiver_argmax.astype(np.uint8)):
        raise ObserverError("persisted indexed argmax PNG is not pixel-exact")
    delta = np.asarray(signed_delta, dtype=np.int16)
    if delta.shape != (SCORER_HEIGHT, SCORER_WIDTH, RGB_CHANNELS):
        raise ObserverError("panel signed-delta sidecar geometry mismatch")
    if not delta_path.exists():
        temporary = panels_dir / f".{delta_path.name}.partial.{os.getpid()}"
        with temporary.open("xb") as handle:
            np.save(handle, delta, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, delta_path)
    persisted_delta = np.load(delta_path, mmap_mode="r", allow_pickle=False)
    if persisted_delta.dtype != np.int16 or not np.array_equal(persisted_delta, delta):
        raise ObserverError("persisted signed-delta NPY is not exact")
    return {
        "pair_id": pair_id,
        "path": str(path),
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "checkpoint_sha256": checkpoint_sha256,
        "native_scorer_geometry": [SCORER_HEIGHT, SCORER_WIDTH],
        "lossless": True,
        "resampling": "none_native_1_to_1",
        "receiver_argmax_indexed_png": {
            "path": str(argmax_path),
            "bytes": argmax_path.stat().st_size,
            "sha256": sha256_file(argmax_path),
            "roundtrip_pixel_exact": True,
        },
        "signed_delta_npy": {
            "path": str(delta_path),
            "bytes": delta_path.stat().st_size,
            "sha256": sha256_file(delta_path),
            "dtype": "int16",
            "shape": [SCORER_HEIGHT, SCORER_WIDTH, RGB_CHANNELS],
        },
    }


def observe_checkpoint(
    *,
    parsed: ParsedCheckpoint,
    band: BandArtifact,
    binding: C2R1B4CurveletBinding,
    masks: StratumMasks,
    stratum_custody: Mapping[str, Any],
    gt: Mapping[str, np.memmap],
    gt_custody: Mapping[str, Any],
    base_camera: np.memmap,
    base_manifest: Mapping[str, Any],
    distortion_net: Any,
    scorer_custody: Mapping[str, Any],
    pair_ids: Sequence[int],
    output_dir: Path,
    cohort_receipt: Mapping[str, Any],
    panel_camera: np.memmap,
    panel_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _checkpoint_custody(parsed, band, binding)
    live_codes, _ema_codes, _ema_head = checkpoint_residuals(parsed.checkpoint)
    byte_estimate = estimate_pair_plane_code_bytes(live_codes)
    ema_codes, ema_head, packet, packet_payload, export_receipt = _prepare_receiver_packet(parsed, binding)
    count = len(pair_ids)
    labels = np.empty((count, SCORER_HEIGHT, SCORER_WIDTH), dtype=np.int8)
    predictions = np.empty_like(labels)
    frame0_predictions = np.empty_like(labels)
    changed = np.empty((count, SCORER_HEIGHT, SCORER_WIDTH), dtype=np.bool_)
    receiver_planes = np.empty((count, SCORER_HEIGHT, SCORER_WIDTH, RGB_CHANNELS), dtype=np.uint8)
    per_d_seg = np.empty((count,), dtype=np.float64)
    per_d_pose = np.empty((count,), dtype=np.float64)
    parity_rows: list[dict[str, Any]] = []
    factor2_count = 0
    for start in range(0, count, SCORER_BATCH_SIZE):
        stop = min(start + SCORER_BATCH_SIZE, count)
        batch = _evaluate_batch(
            parsed=parsed,
            binding=binding,
            packet=packet,
            ema_codes=ema_codes,
            ema_head=ema_head,
            base_camera=np.asarray(base_camera[start:stop]),
            pair_ids=pair_ids[start:stop],
            distortion_net=distortion_net,
            gt=gt,
        )
        labels[start:stop] = batch.labels
        predictions[start:stop] = batch.predictions
        frame0_predictions[start:stop] = batch.frame0_predictions
        changed[start:stop] = batch.changed_pixels
        receiver_planes[start:stop] = batch.receiver_planes
        per_d_seg[start:stop] = batch.per_pair_d_seg
        per_d_pose[start:stop] = batch.per_pair_d_pose
        parity_rows.extend(batch.parity_rows)
        factor2_count += batch.factor2_proof_count
        del batch
        gc.collect()
    d_seg = float(np.mean(per_d_seg, dtype=np.float64))
    d_pose = float(np.mean(per_d_pose, dtype=np.float64))
    facets = facet_accounting(labels, predictions, changed, masks)
    if not np.isclose(d_seg, facets["overall_d_seg"], rtol=0.0, atol=1e-12):
        raise ObserverError("overall d_seg differs from facet accounting")
    ids = np.asarray(pair_ids, dtype=np.int64)
    source_planes = np.asarray(band.source_planes[ids, 1])
    radii = np.zeros_like(source_planes, dtype=np.float32) if band.radii is None else np.asarray(band.radii[ids, 1])
    excursion_row, excursion_mask = out_of_band_excursion(
        labels,
        predictions,
        receiver_planes,
        source_planes,
        radii,
        masks.realizable,
    )
    source_argmax = np.empty_like(labels)
    receiver_argmax = np.empty_like(labels)
    for start in range(0, count, SCORER_BATCH_SIZE):
        stop = min(start + SCORER_BATCH_SIZE, count)
        source_argmax[start:stop] = _segnet_argmax_scorer_planes(distortion_net, source_planes[start:stop])
        receiver_argmax[start:stop] = _segnet_argmax_scorer_planes(distortion_net, receiver_planes[start:stop])
    if not np.array_equal(receiver_argmax, predictions):
        raise ObserverError("direct parsed-receiver scorer-plane SegNet argmax differs from factor2 camera score")
    instability = temporal_argmax_instability(pair_ids, frame0_predictions, predictions)
    per_pair = [
        {
            "pair_id": int(pair_id),
            "pair_idx": int(pair_id),
            "d_seg": float(pair_d_seg),
            "d_pose": float(pair_d_pose),
            "temporal_argmax_instability": instability["per_pair"][position]["instability_fraction"],
        }
        for position, (pair_id, pair_d_seg, pair_d_pose) in enumerate(zip(pair_ids, per_d_seg, per_d_pose, strict=True))
    ]
    pair_position = {int(pair_id): position for position, pair_id in enumerate(pair_ids)}

    def component_tail(values: np.ndarray) -> list[dict[str, Any]]:
        result = []
        for tail in top_pair_rows(pair_ids, values):
            position = pair_position[tail["pair_id"]]
            result.append(
                {
                    "pair_id": tail["pair_id"],
                    "pair_idx": tail["pair_id"],
                    "d_seg": float(per_d_seg[position]),
                    "d_pose": float(per_d_pose[position]),
                }
            )
        return result

    top8_d_seg = component_tail(per_d_seg)
    top8_d_pose = component_tail(per_d_pose)
    parity = {
        "exact": all(row["exact"] for row in parity_rows),
        "comparison": (
            "unquantized_fp32_integer_plane_emitter.numpy_uint8_vs_decode_boundary_packet_plus_apply_boundary_packet"
        ),
        "differing_uint8_value_count": sum(row["differing_uint8_value_count"] for row in parity_rows),
        "max_abs_uint8_difference": max((row["max_abs_uint8_difference"] for row in parity_rows), default=0),
        "per_pair": parity_rows,
        "quantization_mismatch_disposition": "measured_telemetry_receiver_valid_row_retained",
        "scored_surface": "parsed_r1b4_receiver_factor2_realized_camera_bytes",
        "packet_bytes": len(packet_payload),
        "packet_sha256": sha256_bytes(packet_payload),
        "packet_parse_reencode_identical": True,
        "export_receipt": export_receipt,
    }
    factor2 = {
        "exact": True,
        "proof_count": factor2_count,
        "semantic_frame": 1,
        "frame0_policy": "preserve_snapshotted_base_camera_frame",
        "operator": "factor2_align_corners_false_disjoint_2x2",
    }
    panel_records: list[dict[str, Any]] = []
    contact_sheet_record: dict[str, Any] | None = None
    if bool(parsed.checkpoint.rng_state["stage_complete"]):
        panel_plan = cohort_receipt["panel_plan"]
        snapshot_ids = tuple(int(value) for value in panel_plan["snapshot_pair_ids"])
        full_panel_ids = {int(value) for value in panel_plan["full_panel_pair_ids"]}
        contact_ids = tuple(int(value) for value in panel_plan["contact_sheet_pair_ids"])
        if tuple(panel_manifest.get("pair_ids", ())) != snapshot_ids:
            raise ObserverError("derived panel base snapshot custody drift")
        panel_masks, _panel_stratum_custody = load_stratum_masks(band, snapshot_ids)
        contact_thumbnails: dict[int, Any] = {}
        for start in range(0, len(snapshot_ids), SCORER_BATCH_SIZE):
            stop = min(start + SCORER_BATCH_SIZE, len(snapshot_ids))
            batch_ids = snapshot_ids[start:stop]
            batch = _evaluate_batch(
                parsed=parsed,
                binding=binding,
                packet=packet,
                ema_codes=ema_codes,
                ema_head=ema_head,
                base_camera=np.asarray(panel_camera[start:stop]),
                pair_ids=batch_ids,
                distortion_net=distortion_net,
                gt=gt,
            )
            batch_source = np.asarray(band.source_planes[np.asarray(batch_ids, dtype=np.int64), 1])
            batch_radii = (
                np.zeros_like(batch_source, dtype=np.float32)
                if band.radii is None
                else np.asarray(band.radii[np.asarray(batch_ids, dtype=np.int64), 1])
            )
            local_realizable = panel_masks.realizable[start:stop]
            _excursion_row, local_excursion = out_of_band_excursion(
                batch.labels,
                batch.predictions,
                batch.receiver_planes,
                batch_source,
                batch_radii,
                local_realizable,
            )
            batch_source_argmax = _segnet_argmax_scorer_planes(distortion_net, batch_source)
            batch_receiver_argmax = _segnet_argmax_scorer_planes(distortion_net, batch.receiver_planes)
            for local, pair_id in enumerate(batch_ids):
                if pair_id in contact_ids:
                    contact_thumbnails[pair_id] = _contact_thumbnail(
                        batch_source[local],
                        batch_source_argmax[local],
                        batch_receiver_argmax[local],
                    )
                if pair_id not in full_panel_ids:
                    continue
                payload = _render_panel_bytes(
                    pair_id=pair_id,
                    checkpoint_sha256=parsed.sha256,
                    source_plane=batch_source[local],
                    receiver_plane=batch.receiver_planes[local],
                    source_labels=batch_source_argmax[local],
                    receiver_labels=batch_receiver_argmax[local],
                    realizable=local_realizable[local],
                    dead_candidate=panel_masks.dead_candidate[start + local],
                    excursion=local_excursion[local],
                )
                panel_records.append(
                    _write_panel(
                        output_dir,
                        pair_id=pair_id,
                        checkpoint_sha256=parsed.sha256,
                        payload=payload,
                        receiver_argmax=batch_receiver_argmax[local],
                        signed_delta=(
                            batch.receiver_planes[local].astype(np.int16) - batch_source[local].astype(np.int16)
                        ),
                    )
                )
            del batch, batch_source, batch_radii, batch_source_argmax, batch_receiver_argmax
            gc.collect()
        from PIL import Image

        if set(contact_thumbnails) != set(contact_ids):
            raise ObserverError("contact-sheet cohort was not evaluated exhaustively")
        columns = 8
        rows = (len(contact_ids) + columns - 1) // columns
        width = min(columns, len(contact_ids)) * (SCORER_WIDTH // 2)
        height = rows * (SCORER_HEIGHT // 2)
        sheet = Image.new("RGB", (width, height), "black")
        for position, pair_id in enumerate(contact_ids):
            row_index, column = divmod(position, columns)
            sheet.paste(
                contact_thumbnails[pair_id],
                (column * (SCORER_WIDTH // 2), row_index * (SCORER_HEIGHT // 2)),
            )
        sheet_path = output_dir / PANELS_DIR_NAME / f"m1_{parsed.sha256}_concentration_contact_sheet.png"
        if not sheet_path.exists():
            sheet.save(sheet_path, format="PNG", optimize=False, compress_level=9)
        contact_sheet_record = {
            "path": str(sheet_path),
            "bytes": sheet_path.stat().st_size,
            "sha256": sha256_file(sheet_path),
            "lossless": True,
            "thumbnail_transform": "exact_integer_divide_by_2_nearest_neighbor",
            "columns": columns,
            "rows": rows,
            "pair_ids": list(contact_ids),
            "selection": "minimal_exhaustive_census_prefix_covering_90pct_d_seg_mass",
        }
    checkpoint = parsed.checkpoint
    custody = checkpoint.rng_state["run_custody"]
    row = {
        "schema": ROW_SCHEMA,
        "observed_at_utc": utc_now(),
        "checkpoint": {
            "path": str(parsed.path),
            "name": parsed.name,
            "bytes": parsed.payload_bytes,
            "sha256": parsed.sha256,
            "run_id": parsed.run_id,
            "stage_name": checkpoint.stage_name,
            "stage_index": checkpoint.stage_index,
            "epoch": checkpoint.epoch,
            "global_step": checkpoint.global_step,
            "stage_complete": bool(checkpoint.rng_state["stage_complete"]),
            "authority_state": "ema",
            "trainer_config_sha256": checkpoint.config_sha256,
            "policy_sha256": checkpoint.policy_contract["policy_sha256"],
        },
        "sample": {
            "seed": PAIR_SAMPLE_SEED,
            "generator": "numpy.random.PCG64",
            "selection": "bootstrap_top32_dseg_plus_16_pcg64_complement",
            "pair_ids": list(pair_ids),
            "pair_ids_sha256": pair_sample_sha256(pair_ids),
            "pair_ids_sha256_encoding": "canonical_json_ordered_integer_array",
            "sample_size": len(pair_ids),
            "cohort_receipt": dict(cohort_receipt),
        },
        "provenance": {
            "git_head": _git_head(),
            "scorer": dict(scorer_custody),
            "band": {
                "path": str(band.manifest_path),
                "sha256": band.manifest_sha256,
                "source_planes_sha256": band.source_sha256,
                "checkpoint_band_sha256": custody["band_sha256"],
                "strata": dict(stratum_custody),
            },
            "binding": {
                "path": str(binding.manifest_path),
                "sha256": binding.manifest_sha256,
                "topology_sha256": binding.topology_sha256,
                "basis_id": BINDING_BASIS_ID,
            },
            "base": dict(base_manifest),
            "gt": dict(gt_custody),
        },
        "authority": {
            "axis": AXIS,
            "score_claim": False,
            "subsample_advisory": True,
            "promotion_eligible": False,
            "pointer_mutation": False,
        },
        "scoring": {
            "model": "frozen upstream DistortionNet",
            "device": "cpu",
            "batch_size": SCORER_BATCH_SIZE,
            "sample_size": len(pair_ids),
            "geometry": scoring_geometry_receipt(),
            "d_seg": d_seg,
            "d_pose": d_pose,
            "pose_policy": "snapshotted base frame0 plus factor2-realized EMA frame1",
            "per_pair": per_pair,
            "top8_d_seg": top8_d_seg,
            "top8_d_pose": top8_d_pose,
        },
        "facets": facets,
        "out_of_band_excursion": excursion_row,
        "temporal_argmax_instability": instability,
        "emitter_r1b4_equality": parity,
        "factor2_receiver_proof": factor2,
        "direct_scorer_plane_argmax_proof": {
            "receiver_matches_factor2_realized_camera_score": True,
            "source_surface": "BandArtifact.source_planes semantic frame1",
            "receiver_surface": "parsed R1b4 receiver semantic frame1",
            "production_api": "DistortionNet.segnet.preprocess_input_plus_SegNet.forward",
        },
        "stage_panels": {
            "required": bool(checkpoint.rng_state["stage_complete"]),
            "definition": "census_concentration_50pct_full_90pct_contact_plus_direct_flip_strata_exemplars",
            "panel_plan": cohort_receipt["panel_plan"],
            "records": panel_records,
            "contact_sheet": contact_sheet_record,
        },
        "live_pair_plane_codes_compressed_byte_estimate": byte_estimate,
        "score_claim": False,
        "subsample_advisory": True,
        "verdict_scope": (
            "one deterministic frozen n48 [macOS-CPU advisory] checkpoint observation; "
            "not an n600, contest-CPU, contest-CUDA, archive-score, promotion, or pointer verdict"
        ),
    }
    del (
        changed,
        labels,
        predictions,
        frame0_predictions,
        receiver_planes,
        source_planes,
        source_argmax,
        receiver_argmax,
        radii,
    )
    gc.collect()
    return row


def _validate_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not any(resolved == root or root in resolved.parents for root in SSD_ROOTS):
        raise ObserverError(
            "observer output must remain under /Volumes/VertigoDataTier/pact or /Volumes/APDataStore/pact"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _rank_row_template(
    *,
    parsed: ParsedCheckpoint,
    pair_id: int,
    observation: BatchObservation,
    position: int,
    bootstrap_kind: str,
    bootstrap_ids: Sequence[int],
) -> dict[str, Any]:
    parity = observation.parity_rows[position]
    signature = mechanism_signature(
        observation.labels[position],
        observation.predictions[position],
        observation.frame0_predictions[position],
    )
    return {
        "schema": RANK_ROW_SCHEMA,
        "observed_at_utc": utc_now(),
        "checkpoint": {
            "path": str(parsed.path),
            "name": parsed.name,
            "sha256": parsed.sha256,
            "stage_name": parsed.checkpoint.stage_name,
            "stage_index": parsed.checkpoint.stage_index,
            "epoch": parsed.checkpoint.epoch,
            "global_step": parsed.checkpoint.global_step,
            "authority_state": "ema",
        },
        "bootstrap": {
            "kind": bootstrap_kind,
            "population_size": len(bootstrap_ids),
            "pair_ids_sha256": pair_sample_sha256(bootstrap_ids),
            "fallback": bootstrap_kind == "fallback_n128_scratch",
            "fallback_seed": (PAIR_SAMPLE_SEED if bootstrap_kind == "fallback_n128_scratch" else None),
            "batch_size_max": SCORER_BATCH_SIZE,
        },
        "pair_id": pair_id,
        "pair_idx": pair_id,
        "d_seg": float(observation.per_pair_d_seg[position]),
        "d_pose": float(observation.per_pair_d_pose[position]),
        "scoring_geometry": scoring_geometry_receipt(),
        "mechanism_signature": signature,
        "receiver_emitter_parity": dict(parity),
        "receiver_scored": True,
        "packet_parse_reencode_identical": True,
        "factor2_receiver_proof_exact": True,
        "axis": AXIS,
        "score_claim": False,
        "subsample_advisory": True,
        "verdict_scope": (
            "one bootstrap pair on [macOS-CPU advisory]; ranking signal only, "
            "not an n600 contest score or promotion verdict"
        ),
    }


def _complete_bootstrap_rank(
    *,
    output_dir: Path,
    parsed: ParsedCheckpoint,
    band: BandArtifact,
    binding: C2R1B4CurveletBinding,
    gt: Mapping[str, np.memmap],
    base_camera: np.memmap,
    bootstrap_kind: str,
    bootstrap_ids: Sequence[int],
    distortion_net: Any,
) -> dict[int, dict[str, Any]]:
    _checkpoint_custody(parsed, band, binding)
    rank_path = output_dir / RANK_ROWS_NAME
    existing_sha, existing = load_per_pair_rank_rows(rank_path)
    if existing_sha is not None and existing_sha != parsed.sha256:
        raise ObserverError("bootstrap rank resume checkpoint differs from selected checkpoint")
    if not set(existing).issubset(bootstrap_ids):
        raise ObserverError("bootstrap rank rows escape the preflight population")
    expected_bootstrap = {
        "kind": bootstrap_kind,
        "population_size": len(bootstrap_ids),
        "pair_ids_sha256": pair_sample_sha256(bootstrap_ids),
        "fallback": bootstrap_kind == "fallback_n128_scratch",
        "fallback_seed": (PAIR_SAMPLE_SEED if bootstrap_kind == "fallback_n128_scratch" else None),
        "batch_size_max": SCORER_BATCH_SIZE,
    }
    if any(row.get("bootstrap") != expected_bootstrap for row in existing.values()):
        raise ObserverError("bootstrap rank restart population custody drift")
    ema_codes, ema_head, packet, _packet_payload, _export = _prepare_receiver_packet(parsed, binding)
    rows = dict(existing)
    missing = [pair_id for pair_id in bootstrap_ids if pair_id not in rows]
    positions = {pair_id: position for position, pair_id in enumerate(bootstrap_ids)}
    pending: dict[int, dict[str, Any]] = {}
    for start in range(0, len(missing), SCORER_BATCH_SIZE):
        batch_ids = missing[start : start + SCORER_BATCH_SIZE]
        base_positions = [positions[pair_id] for pair_id in batch_ids]
        observation = _evaluate_batch(
            parsed=parsed,
            binding=binding,
            packet=packet,
            ema_codes=ema_codes,
            ema_head=ema_head,
            base_camera=np.asarray(base_camera[np.asarray(base_positions, dtype=np.int64)]),
            pair_ids=batch_ids,
            distortion_net=distortion_net,
            gt=gt,
        )
        for position, pair_id in enumerate(batch_ids):
            pending[pair_id] = _rank_row_template(
                parsed=parsed,
                pair_id=pair_id,
                observation=observation,
                position=position,
                bootstrap_kind=bootstrap_kind,
                bootstrap_ids=bootstrap_ids,
            )
        del observation
        gc.collect()
    combined = {**rows, **pending}
    if set(combined) != set(bootstrap_ids):
        raise ObserverError("bootstrap rank measurement is incomplete")
    ranked_ids = sorted(bootstrap_ids, key=lambda pair_id: (-float(combined[pair_id]["d_seg"]), pair_id))
    rank_by_id = {pair_id: rank for rank, pair_id in enumerate(ranked_ids, start=1)}
    for pair_id, row in existing.items():
        if row.get("dseg_rank") != rank_by_id[pair_id]:
            raise ObserverError("existing bootstrap rank order differs from deterministic replay")
    for pair_id in ranked_ids:
        if pair_id in existing:
            continue
        row = {**pending[pair_id], "dseg_rank": rank_by_id[pair_id]}
        append_canonical_jsonl(rank_path, row)
        rows[pair_id] = row
    _digest, durable = load_per_pair_rank_rows(rank_path)
    if set(durable) != set(bootstrap_ids):
        raise ObserverError("durable bootstrap rank table is incomplete")
    return durable


def _freeze_cohort(
    *,
    output_dir: Path,
    rank_rows: Mapping[int, Mapping[str, Any]],
    bootstrap_kind: str,
    bootstrap_ids: Sequence[int],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    path = output_dir / COHORT_RECEIPT_NAME
    existing = load_cohort_receipt(path)
    if existing is not None:
        return existing
    dseg = {pair_id: float(row["d_seg"]) for pair_id, row in rank_rows.items()}
    cohort_ids = freeze_recurring_cohort(dseg, population_ids=bootstrap_ids)
    hardest = list(cohort_ids[:HARD_PAIR_COUNT])
    background = list(cohort_ids[HARD_PAIR_COUNT:])
    receipt = {
        "schema": COHORT_SCHEMA,
        "created_at_utc": utc_now(),
        "bootstrap_checkpoint_sha256": checkpoint_sha256,
        "bootstrap_kind": bootstrap_kind,
        "bootstrap_population_size": len(bootstrap_ids),
        "bootstrap_pair_ids_sha256": pair_sample_sha256(bootstrap_ids),
        "ranking": "d_seg_descending_then_pair_id_ascending",
        "hardest_count": HARD_PAIR_COUNT,
        "hardest_pair_ids": hardest,
        "background_count": BACKGROUND_PAIR_COUNT,
        "background_seed": PAIR_SAMPLE_SEED,
        "background_generator": "numpy.random.PCG64",
        "background_selection": "choice_without_replacement_from_sorted_complement",
        "background_pair_ids": background,
        "pair_ids": list(cohort_ids),
        "pair_ids_sha256": pair_sample_sha256(cohort_ids),
        "panel_plan": derive_panel_plan(rank_rows),
        "frozen": True,
        "score_claim": False,
        "subsample_advisory": True,
    }
    return receipt


def _cleanup_full_bootstrap_scratch(
    *,
    output_dir: Path,
    bootstrap_kind: str,
    bootstrap_manifest: Mapping[str, Any],
    cohort_manifest: Mapping[str, Any],
    rank_path: Path,
) -> None:
    if bootstrap_kind != "full_n600_scratch":
        return
    snapshot_path, _manifest_path = _snapshot_paths(output_dir, bootstrap_kind)
    cleanup_path = output_dir / CLEANUP_RECEIPT_NAME
    events: dict[str, dict[str, Any]] = {}
    if cleanup_path.exists():
        with cleanup_path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.endswith(b"\n"):
                    raise ObserverError(f"cleanup JSONL line {line_number} is incomplete")
                encoded = raw[:-1]
                try:
                    row = json.loads(encoded.decode("ascii"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ObserverError(f"cleanup JSONL line {line_number} is invalid") from exc
                if not isinstance(row, dict) or row.get("schema") != CLEANUP_SCHEMA or canonical_json(row) != encoded:
                    raise ObserverError(f"cleanup JSONL line {line_number} is noncanonical")
                event = row.get("event")
                if event not in {"CERTIFIED_BEFORE_UNLINK", "UNLINK_COMPLETE"}:
                    raise ObserverError(f"cleanup JSONL line {line_number} has unknown event")
                if event in events:
                    raise ObserverError(f"cleanup JSONL repeats {event}")
                events[event] = row
    expected = (output_dir / FULL_BASE_NAME).resolve()
    if snapshot_path.resolve() != expected or output_dir.resolve() not in expected.parents:
        raise ObserverError("full scratch cleanup target escaped the observer output directory")
    if not rank_path.exists() or not (output_dir / COHORT_RECEIPT_NAME).exists():
        raise ObserverError("full scratch cleanup requires durable rank and cohort receipts")
    if "UNLINK_COMPLETE" in events:
        if snapshot_path.exists():
            raise ObserverError("cleanup receipt says complete but full scratch still exists")
        return
    if not snapshot_path.exists():
        certificate = events.get("CERTIFIED_BEFORE_UNLINK")
        if certificate is None:
            raise ObserverError("full scratch disappeared without cleanup certification")
        append_canonical_jsonl(
            cleanup_path,
            {
                "schema": CLEANUP_SCHEMA,
                "recorded_at_utc": utc_now(),
                "event": "UNLINK_COMPLETE",
                "original_path": str(snapshot_path),
                "bytes": certificate["bytes"],
                "sha256": certificate["sha256"],
                "certification_event": "CERTIFIED_BEFORE_UNLINK",
                "observer_owned": True,
                "recovered_after_restart": True,
                "score_claim": False,
            },
        )
        return
    if sha256_file(snapshot_path) != bootstrap_manifest["snapshot_npy_sha256"]:
        raise ObserverError("full scratch cleanup SHA custody drift")
    certificate = {
        "schema": CLEANUP_SCHEMA,
        "recorded_at_utc": utc_now(),
        "event": "CERTIFIED_BEFORE_UNLINK",
        "original_path": str(snapshot_path),
        "bytes": snapshot_path.stat().st_size,
        "sha256": bootstrap_manifest["snapshot_npy_sha256"],
        "source_custody": {
            "path": bootstrap_manifest["source_path"],
            "bytes": bootstrap_manifest["source_bytes"],
            "sha256": bootstrap_manifest["source_sha256"],
            "stat": bootstrap_manifest["source_stat"],
        },
        "derived_artifacts": {
            "rank_table": {
                "path": str(rank_path),
                "bytes": rank_path.stat().st_size,
                "sha256": sha256_file(rank_path),
            },
            "recurring_snapshot": {
                "path": cohort_manifest["snapshot_path"],
                "bytes": cohort_manifest["snapshot_npy_bytes"],
                "sha256": cohort_manifest["snapshot_npy_sha256"],
            },
            "cohort_receipt": str(output_dir / COHORT_RECEIPT_NAME),
        },
        "command": list(sys.argv),
        "config": {
            "bootstrap_population_size": FULL_BOOTSTRAP_SIZE,
            "batch_size_max": SCORER_BATCH_SIZE,
            "recurring_size": PAIR_SAMPLE_SIZE,
        },
        "reason": (
            "observer-owned full n600 scratch is deterministically rebuildable from the "
            "hash-bound stable source and is superseded by complete rank plus recurring snapshot"
        ),
        "success_only": True,
        "score_claim": False,
    }
    prior_certificate = events.get("CERTIFIED_BEFORE_UNLINK")
    if prior_certificate is None:
        append_canonical_jsonl(cleanup_path, certificate)
    elif (
        prior_certificate.get("original_path") != certificate["original_path"]
        or prior_certificate.get("bytes") != certificate["bytes"]
        or prior_certificate.get("sha256") != certificate["sha256"]
    ):
        raise ObserverError("existing cleanup certificate custody differs")
    snapshot_path.unlink()
    append_canonical_jsonl(
        cleanup_path,
        {
            "schema": CLEANUP_SCHEMA,
            "recorded_at_utc": utc_now(),
            "event": "UNLINK_COMPLETE",
            "original_path": str(snapshot_path),
            "bytes": certificate["bytes"],
            "sha256": certificate["sha256"],
            "certification_event": "CERTIFIED_BEFORE_UNLINK",
            "observer_owned": True,
            "score_claim": False,
        },
    )


def _record_checkpoint_error(errors_path: Path, parsed: ParsedCheckpoint, exc: Exception) -> None:
    append_canonical_jsonl(
        errors_path,
        {
            "schema": ERROR_SCHEMA,
            "observed_at_utc": utc_now(),
            "checkpoint_path": str(parsed.path),
            "checkpoint_name": parsed.name,
            "checkpoint_bytes": parsed.payload_bytes,
            "checkpoint_sha256": parsed.sha256,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "disposition": "fail_closed_no_facet_row",
            "score_claim": False,
        },
    )


def run(config: ObserverConfig) -> int:
    output_dir = _validate_output_dir(config.output_dir)
    checkpoint_dir = config.checkpoint_dir.expanduser().resolve(strict=True)
    if not checkpoint_dir.is_dir():
        raise ObserverError("checkpoint path is not a directory")
    if not np.isfinite(config.poll_seconds) or config.poll_seconds <= 0:
        raise ObserverError("poll interval must be finite and positive")
    band, binding, gt, gt_custody = _validate_static_custody(config)
    rows_path = output_dir / ROWS_NAME
    errors_path = output_dir / ERRORS_NAME
    lock_path = output_dir / ".observer.lock"
    incomplete_attempts: dict[tuple[str, int, int], int] = {}
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ObserverError("another observer owns this output directory") from exc
        processed = load_processed_checkpoint_sha256s(rows_path)
        failed = load_error_checkpoint_sha256s(errors_path)
        distortion_net: Any | None = None
        scorer_custody: dict[str, Any] | None = None
        while True:
            parsed_checkpoints: list[ParsedCheckpoint] = []
            for path in sorted(checkpoint_dir.glob(CHECKPOINT_GLOB), key=checkpoint_sort_key):
                try:
                    parsed_checkpoints.append(read_stable_checkpoint(path))
                except IncompleteCheckpointError:
                    continue
                except Exception as exc:
                    try:
                        stat = path.stat()
                    except FileNotFoundError:
                        continue
                    key = (str(path), stat.st_size, stat.st_mtime_ns)
                    incomplete_attempts[key] = incomplete_attempts.get(key, 0) + 1
                    if incomplete_attempts[key] < 3:
                        continue
                    digest = sha256_file(path)
                    if digest not in failed:
                        append_canonical_jsonl(
                            errors_path,
                            {
                                "schema": ERROR_SCHEMA,
                                "observed_at_utc": utc_now(),
                                "checkpoint_path": str(path),
                                "checkpoint_name": path.name,
                                "checkpoint_bytes": stat.st_size,
                                "checkpoint_sha256": digest,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "disposition": "fail_closed_no_facet_row",
                                "score_claim": False,
                            },
                        )
                        failed.add(digest)
            cohort_receipt = load_cohort_receipt(output_dir / COHORT_RECEIPT_NAME)
            if cohort_receipt is None:
                preflight = _load_preflight(output_dir)
                if preflight is None:
                    bootstrap_kind, bootstrap_ids, preflight_receipt = _bootstrap_preflight(output_dir)
                    write_canonical_json(output_dir / PREFLIGHT_RECEIPT_NAME, preflight_receipt)
                else:
                    bootstrap_kind, bootstrap_ids, _preflight_receipt = preflight
                bootstrap = snapshot_base_population(
                    output_dir,
                    config.live_base_raw,
                    bootstrap_ids,
                    kind=bootstrap_kind,
                )
                if bootstrap is None:
                    bootstrap = snapshot_base_population_from_scorer_npy(
                        output_dir,
                        config.live_base_scorer_npy,
                        bootstrap_ids,
                        kind=bootstrap_kind,
                    )
                if bootstrap is not None and parsed_checkpoints:
                    if distortion_net is None:
                        distortion_net, scorer_custody = _load_distortion_net(config.upstream_dir)
                    bootstrap_camera, bootstrap_manifest = bootstrap
                    existing_rank_sha, _existing_rank = load_per_pair_rank_rows(output_dir / RANK_ROWS_NAME)
                    if existing_rank_sha is None:
                        bootstrap_parsed = next(
                            (item for item in parsed_checkpoints if item.sha256 not in failed), None
                        )
                    else:
                        bootstrap_parsed = next(
                            (item for item in parsed_checkpoints if item.sha256 == existing_rank_sha),
                            None,
                        )
                    if bootstrap_parsed is not None:
                        assert distortion_net is not None
                        try:
                            rank_rows = _complete_bootstrap_rank(
                                output_dir=output_dir,
                                parsed=bootstrap_parsed,
                                band=band,
                                binding=binding,
                                gt=gt,
                                base_camera=bootstrap_camera,
                                bootstrap_kind=bootstrap_kind,
                                bootstrap_ids=bootstrap_ids,
                                distortion_net=distortion_net,
                            )
                        except Exception as exc:
                            if bootstrap_parsed.sha256 not in failed:
                                _record_checkpoint_error(errors_path, bootstrap_parsed, exc)
                                failed.add(bootstrap_parsed.sha256)
                        else:
                            cohort_receipt = _freeze_cohort(
                                output_dir=output_dir,
                                rank_rows=rank_rows,
                                bootstrap_kind=bootstrap_kind,
                                bootstrap_ids=bootstrap_ids,
                                checkpoint_sha256=bootstrap_parsed.sha256,
                            )
                            cohort_ids = tuple(cohort_receipt["pair_ids"])
                            _cohort_camera, cohort_manifest = snapshot_recurring_from_bootstrap(
                                output_dir,
                                bootstrap_camera,
                                bootstrap_manifest,
                                cohort_ids,
                            )
                            panel_ids = tuple(cohort_receipt["panel_plan"]["snapshot_pair_ids"])
                            _panel_camera, panel_manifest = snapshot_panel_from_bootstrap(
                                output_dir,
                                bootstrap_camera,
                                bootstrap_manifest,
                                panel_ids,
                            )
                            write_canonical_json(output_dir / COHORT_RECEIPT_NAME, cohort_receipt)
                            durable_cohort = load_cohort_receipt(output_dir / COHORT_RECEIPT_NAME)
                            if durable_cohort is None:
                                raise ObserverError("recurring cohort receipt did not become durable")
                            cohort_receipt = durable_cohort
                            del bootstrap_camera, bootstrap
                            gc.collect()
                            _cleanup_full_bootstrap_scratch(
                                output_dir=output_dir,
                                bootstrap_kind=bootstrap_kind,
                                bootstrap_manifest=bootstrap_manifest,
                                cohort_manifest={
                                    **cohort_manifest,
                                    "panel_snapshot": panel_manifest,
                                },
                                rank_path=output_dir / RANK_ROWS_NAME,
                            )
            if cohort_receipt is not None:
                pair_ids = tuple(cohort_receipt["pair_ids"])
                base = load_base_snapshot(output_dir, pair_ids, kind="recurring_n48")
                if base is None:
                    raise ObserverError("cohort receipt exists without recurring base snapshot")
                base_camera, base_manifest = base
                panel_ids = tuple(cohort_receipt["panel_plan"]["snapshot_pair_ids"])
                panel_base = load_base_snapshot(output_dir, panel_ids, kind="panel_derived")
                if panel_base is None:
                    raise ObserverError("cohort receipt exists without derived panel base snapshot")
                panel_camera, panel_manifest = panel_base
                resume_preflight = _load_preflight(output_dir)
                if resume_preflight is not None and resume_preflight[0] == "full_n600_scratch":
                    full_manifest = _load_snapshot_manifest_only(output_dir, "full_n600_scratch")
                    if full_manifest is None:
                        raise ObserverError("full bootstrap cleanup lost its custody manifest")
                    _cleanup_full_bootstrap_scratch(
                        output_dir=output_dir,
                        bootstrap_kind="full_n600_scratch",
                        bootstrap_manifest=full_manifest,
                        cohort_manifest=base_manifest,
                        rank_path=output_dir / RANK_ROWS_NAME,
                    )
                masks, stratum_custody = load_stratum_masks(band, pair_ids)
                if distortion_net is None:
                    distortion_net, scorer_custody = _load_distortion_net(config.upstream_dir)
                for parsed in parsed_checkpoints:
                    if parsed.sha256 in processed or parsed.sha256 in failed:
                        continue
                    try:
                        assert distortion_net is not None and scorer_custody is not None
                        row = observe_checkpoint(
                            parsed=parsed,
                            band=band,
                            binding=binding,
                            masks=masks,
                            stratum_custody=stratum_custody,
                            gt=gt,
                            gt_custody=gt_custody,
                            base_camera=base_camera,
                            base_manifest=base_manifest,
                            distortion_net=distortion_net,
                            scorer_custody=scorer_custody,
                            pair_ids=pair_ids,
                            output_dir=output_dir,
                            cohort_receipt=cohort_receipt,
                            panel_camera=panel_camera,
                            panel_manifest=panel_manifest,
                        )
                    except Exception as exc:
                        _record_checkpoint_error(errors_path, parsed, exc)
                        failed.add(parsed.sha256)
                        continue
                    append_canonical_jsonl(rows_path, row)
                    processed.add(parsed.sha256)
            if config.once:
                return 0
            time.sleep(config.poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--band-manifest", type=Path, required=True)
    parser.add_argument("--carrier-binding", type=Path, required=True)
    parser.add_argument("--gt-cache", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument(
        "--live-base-raw",
        type=Path,
        default=None,
        help=(
            "Complete final trainer base_camera_frames.raw; read-only and optional after the "
            "recurring n48 snapshot exists; .partial paths are never read or hashed"
        ),
    )
    parser.add_argument(
        "--live-base-scorer-npy",
        type=Path,
        default=None,
        help=(
            "Stable final materializer base_scorer_planes.npy; read-only factor-2 fallback "
            "when the success-cleaned camera raw is absent"
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--once", action="store_true", help="Process the current stable set and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ObserverConfig(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        band_manifest=args.band_manifest,
        carrier_binding=args.carrier_binding,
        gt_cache=args.gt_cache,
        upstream_dir=args.upstream_dir,
        live_base_raw=args.live_base_raw,
        live_base_scorer_npy=args.live_base_scorer_npy,
        poll_seconds=args.poll_seconds,
        once=args.once,
    )
    try:
        return run(config)
    except (ObserverError, OSError, ValueError) as exc:
        print(f"M1 observer refusal: {exc}", file=sys.stderr)
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
