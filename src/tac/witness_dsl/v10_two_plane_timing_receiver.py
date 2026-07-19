# SPDX-License-Identifier: MIT
"""Timed two-independent-plane receiver for the V10 C1 control.

This module is deliberately decode-only.  It accepts the closed production
predictor packet, realizes each described plane independently, checkpoints one
pair at a time, and records local timing without making a contest-runtime or
score claim.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from tac.codec.v10_jxl_plane_codec import (
    CODEC_ID as JXL_PLANE_Y_CODEC_ID,
)
from tac.codec.v10_jxl_plane_codec import (
    decode_payload as decode_jxl_plane_payload,
)
from tac.codec.v10_predictor_residual import decode_predictor_residual
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Uint8LatticeError,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)
from tac.witness_dsl.v10_production_receiver import (
    DESCRIPTION_FRAME0_POLICY_ID,
    MEMBER_NAME,
    PREDICTOR_RESIDUAL_Y_CODEC_ID,
    PREFIX,
    SECTION_LENGTH,
    ParsedProductionPacket,
    ProductionReceiverError,
    parse_packet,
)

TIMING_RECEIPT_SCHEMA = "v10_two_plane_timing_receiver_receipt.v1"
PAIR_MANIFEST_SCHEMA = "v10_two_plane_timing_pair_manifest.v1"
PLANE0_MANIFEST_SCHEMA = "v10_two_plane_timing_plane0_manifest.v1"
CHUNK_MANIFEST_SCHEMA = "v10_two_plane_timing_chunk_manifest.v1"
MLX_PARITY_SCHEMA = "v10_two_plane_mlx_parity.v1"
STATE_DIRECTORY = ".v10-two-plane-timing-receiver"
CHUNK_PAIRS = 12
FULL_PAIR_COUNT = 600
FULL_CAMERA_HW = (874, 1164)
FULL_SCORER_HW = (384, 512)
FULL_NUMERATOR_VALUES = 707_788_800
FULL_RAW_BYTES = 3_662_409_600
FULL_Y0_SHA256 = "5e86e419cdd5bd41c9482cabc78cf27cec22281098b64c715d91f1f067d11566"
FULL_Y1_SHA256 = "6a731946e3d9de82089c90de9784c5a5bc72c607c963fb6f79dac16f00ac89bc"
CPU_TIMING_AXIS = "[macOS-CPU local timing] NON-PROMOTABLE"
MLX_TIMING_AXIS = "[macOS-MLX research-signal] NON-PROMOTABLE"


class TwoPlaneTimingReceiverError(ValueError):
    """Fail-closed C1 archive, stage, resume, timing, or parity error."""


@dataclass(frozen=True)
class TwoPlaneTimedInflateResult:
    """Result of one receiver invocation."""

    completed: bool
    raw_path: Path | None
    raw_sha256: str | None
    raw_bytes: int
    pair_stages_preserved: int
    resumed_pairs: int
    numerator_values_verified: int
    stage_tree_sha256: str
    chunk_tree_sha256: str
    output_tree_sha256: str | None
    timing_receipt_path: Path
    timing_receipt_sha256: str
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class _SolvedPlane:
    pair_index: int
    payload: bytes
    solve_seconds: float


@dataclass(frozen=True)
class _VerifiedPlane:
    pair_index: int
    numerator_equal_values: int
    frame_sha256: str


@dataclass(frozen=True)
class MlxFactor2Plan:
    """One reusable device index plan for all planes in a parity run."""

    camera_hw: tuple[int, int]
    scorer_hw: tuple[int, int]
    device_indices: Any
    device_valid: Any


_PROCESS_OPERATOR: DisjointResizeOperator | None = None


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TwoPlaneTimingReceiverError("value is not canonical JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _positive_elapsed(start: float) -> float:
    elapsed = time.monotonic() - start
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise TwoPlaneTimingReceiverError("monotonic component timing was not positive and finite")
    return elapsed


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TwoPlaneTimingReceiverError(f"{label} must be an exact integer >= {minimum}")
    return value


def _require_manifest_scalar_types(
    manifest: Any,
    *,
    label: str,
    integer_fields: Sequence[str],
) -> Mapping[str, Any]:
    """Reject JSON numeric/bool coercions before semantic equality checks."""

    if not isinstance(manifest, dict):
        raise TwoPlaneTimingReceiverError(f"{label} manifest must be a JSON object")
    for field in integer_fields:
        if type(manifest.get(field)) is not int:
            raise TwoPlaneTimingReceiverError(f"{label} manifest field {field} must be an exact integer")
    for field in ("score_claim", "promotion_eligible"):
        if type(manifest.get(field)) is not bool:
            raise TwoPlaneTimingReceiverError(f"{label} manifest field {field} must be boolean")
    return manifest


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _atomic_write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise TwoPlaneTimingReceiverError(f"write-once path exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_write_once_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise TwoPlaneTimingReceiverError(f"preserved write-once bytes drifted: {path}")
        return
    try:
        _atomic_write_once(path, payload)
    except TwoPlaneTimingReceiverError:
        if not path.is_file() or path.read_bytes() != payload:
            raise


def _atomic_replace_from_stages(
    target: Path,
    stage_paths: Sequence[Path],
    *,
    expected_bytes: int,
) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    total = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            for stage_path in stage_paths:
                with stage_path.open("rb") as stage:
                    while block := stage.read(8 << 20):
                        output.write(block)
                        digest.update(block)
                        total += len(block)
            output.flush()
            os.fsync(output.fileno())
        if total != expected_bytes or temporary.stat().st_size != expected_bytes:
            raise TwoPlaneTimingReceiverError("assembled raw byte count drifted")
        expected_sha = digest.hexdigest()
        if target.exists():
            if not target.is_file() or target.stat().st_size != expected_bytes or _sha256_file(target) != expected_sha:
                raise TwoPlaneTimingReceiverError("existing final raw bytes drifted")
            temporary.unlink()
        else:
            os.replace(temporary, target)
        return expected_sha, total
    finally:
        temporary.unlink(missing_ok=True)


def _tree_sha256(root: Path, *, suffixes: tuple[str, ...] | None = None) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if suffixes is not None and path.suffix not in suffixes:
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">Q", size))
        with path.open("rb") as handle:
            while block := handle.read(8 << 20):
                digest.update(block)
    return digest.hexdigest()


def _safe_video_name(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TwoPlaneTimingReceiverError("video names file is unreadable") from exc
    if len(lines) != 1 or not lines[0] or lines[0] != lines[0].strip():
        raise TwoPlaneTimingReceiverError("exactly one canonical video name is required")
    candidate = PurePosixPath(lines[0])
    if candidate.is_absolute() or ".." in candidate.parts or any(part in ("", ".") for part in candidate.parts):
        raise TwoPlaneTimingReceiverError("video name escapes the output root")
    return candidate.as_posix()


def _safe_raw_output_path(output_root: Path, video_name: str) -> Path:
    """Match official ``${video_name%.*}.raw`` output naming safely."""

    pure = PurePosixPath(video_name)
    try:
        relative = Path(*pure.parts).with_suffix(".raw")
    except ValueError as exc:
        raise TwoPlaneTimingReceiverError("video name has no valid output stem") from exc
    candidate = output_root / relative
    root_resolved = output_root.resolve(strict=False)
    parent_resolved = candidate.parent.resolve(strict=False)
    if parent_resolved != root_resolved and root_resolved not in parent_resolved.parents:
        raise TwoPlaneTimingReceiverError("resolved video output escapes the output root")
    return candidate


def _canonical_zip(packet_bytes: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", allowZip64=True) as archive:
        info = zipfile.ZipInfo(MEMBER_NAME, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, packet_bytes)
    return buffer.getvalue()


def _read_exact_archive_root(archive_root: Path) -> tuple[bytes, bytes, str]:
    """Read either builder custody or the official extracted ``0.bin`` ABI."""

    archive_path = archive_root / "archive.zip"
    packet_path = archive_root / MEMBER_NAME
    has_archive = archive_path.is_file()
    has_packet = packet_path.is_file()
    if not has_archive and not has_packet:
        raise TwoPlaneTimingReceiverError("archive directory contains neither archive.zip nor extracted 0.bin")
    archive_bytes: bytes | None = None
    packet_from_archive: bytes | None = None
    if has_archive:
        try:
            archive_bytes = archive_path.read_bytes()
        except OSError as exc:
            raise TwoPlaneTimingReceiverError("archive.zip cannot be read exactly") from exc
        try:
            with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
                infos = archive.infolist()
                if len(infos) != 1 or infos[0].filename != MEMBER_NAME or infos[0].is_dir():
                    raise TwoPlaneTimingReceiverError("archive must contain exactly stored 0.bin")
                info = infos[0]
                if info.compress_type != zipfile.ZIP_STORED or info.compress_size != info.file_size:
                    raise TwoPlaneTimingReceiverError("0.bin must be ZIP_STORED")
                packet_from_archive = archive.read(info)
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            raise TwoPlaneTimingReceiverError("archive cannot be reopened exactly") from exc
        if _canonical_zip(packet_from_archive) != archive_bytes:
            raise TwoPlaneTimingReceiverError("archive ZIP bytes are not canonical")
    packet_from_directory: bytes | None = None
    if has_packet:
        try:
            packet_from_directory = packet_path.read_bytes()
        except OSError as exc:
            raise TwoPlaneTimingReceiverError("extracted 0.bin cannot be read exactly") from exc
    if packet_from_archive is not None and packet_from_directory is not None:
        if packet_from_archive != packet_from_directory:
            raise TwoPlaneTimingReceiverError("archive.zip and extracted 0.bin disagree")
        return archive_bytes or _canonical_zip(packet_from_archive), packet_from_archive, "archive_zip_and_0_bin"
    if packet_from_archive is not None:
        return archive_bytes or _canonical_zip(packet_from_archive), packet_from_archive, "archive_zip"
    assert packet_from_directory is not None
    return _canonical_zip(packet_from_directory), packet_from_directory, "extracted_0_bin"


def reserialize_parsed_packet(packet: ParsedProductionPacket) -> bytes:
    """Serialize every parsed production field and section canonically."""

    if not isinstance(packet, ParsedProductionPacket):
        raise TwoPlaneTimingReceiverError("packet reserialization requires parsed production bytes")
    header_bytes = _canonical_json(dict(packet.header))
    rebuilt = bytearray(
        PREFIX.pack(PREFIX.unpack_from(packet.packet_bytes)[0], packet.header["version"], len(header_bytes))
    )
    rebuilt.extend(header_bytes)
    for section in packet.sections:
        rebuilt.extend(SECTION_LENGTH.pack(len(section.payload)))
        rebuilt.extend(section.payload)
    return bytes(rebuilt)


def _validate_packet(packet: ParsedProductionPacket) -> None:
    header = packet.header
    if header.get("y_codec_id") not in (PREDICTOR_RESIDUAL_Y_CODEC_ID, JXL_PLANE_Y_CODEC_ID):
        raise TwoPlaneTimingReceiverError("C1 requires predictor-residual-u8.v1 or jxl-lossless-plane.v1")
    if header.get("frame0_policy_id") != DESCRIPTION_FRAME0_POLICY_ID:
        raise TwoPlaneTimingReceiverError("C1 requires description-frame0.v1")
    if header.get("residual_codec_id") is not None or len(packet.sections) != 2:
        raise TwoPlaneTimingReceiverError("C1 refuses quotient residual sections")
    if reserialize_parsed_packet(packet) != packet.packet_bytes:
        raise TwoPlaneTimingReceiverError("strict packet parse/re-encode differs")


def _validate_targets(
    frame0: np.ndarray,
    frame1: np.ndarray,
    pair_ids: Sequence[int],
    *,
    expected_pair_count: int,
    expected_scorer_hw: tuple[int, int],
    expected_y0_sha256: str | None,
    expected_y1_sha256: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = (expected_pair_count, *expected_scorer_hw, 3)
    if frame0.dtype != np.uint8 or frame1.dtype != np.uint8:
        raise TwoPlaneTimingReceiverError("expanded planes must be exact uint8")
    if frame0.shape != expected_shape or frame1.shape != expected_shape:
        raise TwoPlaneTimingReceiverError(f"expanded plane geometry differs from {expected_shape}")
    if tuple(pair_ids) != tuple(range(expected_pair_count)):
        raise TwoPlaneTimingReceiverError("predictor pair ids must be exactly 0..N-1")
    if np.shares_memory(frame0, frame1):
        raise TwoPlaneTimingReceiverError("expanded planes alias memory")
    y0 = np.ascontiguousarray(frame0)
    y1 = np.ascontiguousarray(frame1)
    if np.shares_memory(y0, y1):
        raise TwoPlaneTimingReceiverError("owned expanded planes alias memory")
    for pair_index in range(expected_pair_count):
        if np.array_equal(y0[pair_index], y1[pair_index]):
            raise TwoPlaneTimingReceiverError(f"pair {pair_index} planes are byte-equal")
    y0_sha = _sha256(y0.tobytes(order="C"))
    y1_sha = _sha256(y1.tobytes(order="C"))
    if expected_y0_sha256 is not None and y0_sha != expected_y0_sha256:
        raise TwoPlaneTimingReceiverError("aggregate Y0 digest differs from frozen custody")
    if expected_y1_sha256 is not None and y1_sha != expected_y1_sha256:
        raise TwoPlaneTimingReceiverError("aggregate Y1 digest differs from frozen custody")
    return y0, y1


def _process_initializer(operator: DisjointResizeOperator) -> None:
    global _PROCESS_OPERATOR
    _PROCESS_OPERATOR = operator


def _process_solve(pair_index: int, target: np.ndarray) -> _SolvedPlane:
    operator = _PROCESS_OPERATOR
    if operator is None:
        raise TwoPlaneTimingReceiverError("process worker lacks its bound resize operator")
    start = time.monotonic()
    try:
        frame = realize_factor2_uint8_scorer_plane(operator, target)
    except Uint8LatticeError as exc:
        raise TwoPlaneTimingReceiverError("factor-2 plane realization refused") from exc
    elapsed = _positive_elapsed(start)
    return _SolvedPlane(pair_index, np.ascontiguousarray(frame).tobytes(order="C"), elapsed)


def _process_verify(pair_index: int, payload: bytes, target: np.ndarray) -> _VerifiedPlane:
    operator = _PROCESS_OPERATOR
    if operator is None:
        raise TwoPlaneTimingReceiverError("process worker lacks its bound resize operator")
    frame = _frame_from_payload(payload, (operator.camera_h, operator.camera_w))
    count = _verify_one(operator, frame, target)
    return _VerifiedPlane(pair_index, count, _sha256(payload))


def _local_solve(operator: DisjointResizeOperator, pair_index: int, target: np.ndarray) -> _SolvedPlane:
    start = time.monotonic()
    try:
        frame = realize_factor2_uint8_scorer_plane(operator, target)
    except Uint8LatticeError as exc:
        raise TwoPlaneTimingReceiverError("factor-2 plane realization refused") from exc
    return _SolvedPlane(
        pair_index,
        np.ascontiguousarray(frame).tobytes(order="C"),
        _positive_elapsed(start),
    )


def _solve_phase(
    operator: DisjointResizeOperator,
    executor: ProcessPoolExecutor | None,
    indices: Sequence[int],
    targets: np.ndarray,
) -> tuple[dict[int, _SolvedPlane], float]:
    start = time.monotonic()
    if executor is None:
        rows = [_local_solve(operator, pair_index, targets[pair_index]) for pair_index in indices]
    else:
        futures = [executor.submit(_process_solve, pair_index, targets[pair_index]) for pair_index in indices]
        rows = [future.result() for future in futures]
    elapsed = _positive_elapsed(start)
    by_index = {row.pair_index: row for row in rows}
    if tuple(sorted(by_index)) != tuple(indices):
        raise TwoPlaneTimingReceiverError("solve workers returned duplicate or missing pair indices")
    return by_index, elapsed


def _verify_phase(
    operator: DisjointResizeOperator,
    executor: ProcessPoolExecutor | None,
    rows: Mapping[int, bytes],
    targets: np.ndarray,
) -> tuple[dict[int, _VerifiedPlane], float]:
    indices = tuple(sorted(rows))
    start = time.monotonic()
    if executor is None:
        verified = [
            _VerifiedPlane(
                pair_index,
                _verify_one(
                    operator,
                    _frame_from_payload(rows[pair_index], (operator.camera_h, operator.camera_w)),
                    targets[pair_index],
                ),
                _sha256(rows[pair_index]),
            )
            for pair_index in indices
        ]
    else:
        futures = [
            executor.submit(_process_verify, pair_index, rows[pair_index], targets[pair_index])
            for pair_index in indices
        ]
        verified = [future.result() for future in futures]
    elapsed = _positive_elapsed(start)
    by_index = {row.pair_index: row for row in verified}
    if tuple(sorted(by_index)) != indices:
        raise TwoPlaneTimingReceiverError("verification workers returned duplicate or missing pair indices")
    return by_index, elapsed


def _frame_from_payload(payload: bytes, camera_hw: tuple[int, int]) -> np.ndarray:
    expected = camera_hw[0] * camera_hw[1] * 3
    if len(payload) != expected:
        raise TwoPlaneTimingReceiverError("solved frame byte count drifted")
    return np.frombuffer(payload, dtype=np.uint8).reshape(*camera_hw, 3)


def _verify_one(
    operator: DisjointResizeOperator,
    frame: np.ndarray,
    target: np.ndarray,
) -> int:
    try:
        proof = verify_factor2_uint8_scorer_plane(operator, frame, target)
    except Uint8LatticeError as exc:
        raise TwoPlaneTimingReceiverError("exact integer numerator verification refused") from exc
    if not proof.numerator_exact or not proof.certified_exact:
        raise TwoPlaneTimingReceiverError("exact integer numerator verification failed")
    return int(proof.numerator_equal_values)


def _pair_paths(stage_root: Path, pair_index: int) -> tuple[Path, Path]:
    return (
        stage_root / "pairs" / f"pair-{pair_index:06d}.bin",
        stage_root / "pair_manifests" / f"pair-{pair_index:06d}.json",
    )


def _plane0_paths(stage_root: Path, pair_index: int) -> tuple[Path, Path]:
    return (
        stage_root / "plane0" / f"pair-{pair_index:06d}.bin",
        stage_root / "plane0_manifests" / f"pair-{pair_index:06d}.json",
    )


def _plane0_manifest(
    *,
    archive_sha256: str,
    packet_sha256: str,
    pair_index: int,
    y0: np.ndarray,
    frame0_payload: bytes,
    numerator0: int,
) -> dict[str, Any]:
    return {
        "schema": PLANE0_MANIFEST_SCHEMA,
        "archive_sha256": archive_sha256,
        "packet_sha256": packet_sha256,
        "pair_index": pair_index,
        "pair_id": pair_index,
        "y0_sha256": _sha256(y0.tobytes(order="C")),
        "x0_sha256": _sha256(frame0_payload),
        "x0_bytes": len(frame0_payload),
        "numerator0_equal_values": numerator0,
        "frame0_policy_id": DESCRIPTION_FRAME0_POLICY_ID,
        "y_codec_id": PREDICTOR_RESIDUAL_Y_CODEC_ID,
        "score_claim": False,
        "promotion_eligible": False,
    }


def _persist_plane0(
    stage_root: Path,
    pair_index: int,
    payload: bytes,
    manifest: Mapping[str, Any],
) -> None:
    stage_path, manifest_path = _plane0_paths(stage_root, pair_index)
    _ensure_write_once_exact(stage_path, payload)
    manifest_payload = _canonical_json(manifest)
    _ensure_write_once_exact(manifest_path, manifest_payload)
    if stage_path.read_bytes() != payload or manifest_path.read_bytes() != manifest_payload:
        raise TwoPlaneTimingReceiverError("plane-0 checkpoint immediate reopen differed")


def _reopen_plane0_bytes(
    stage_root: Path,
    pair_index: int,
    *,
    archive_sha256: str,
    packet_sha256: str,
    y0: np.ndarray,
    recover_stage_only: bool = False,
) -> tuple[bytes, Mapping[str, Any] | None]:
    stage_path, manifest_path = _plane0_paths(stage_root, pair_index)
    if manifest_path.exists() and not stage_path.is_file():
        raise TwoPlaneTimingReceiverError("plane-0 manifest exists without its stage bytes")
    if not stage_path.is_file():
        raise TwoPlaneTimingReceiverError("plane-0 checkpoint is incomplete")
    payload = stage_path.read_bytes()
    if not manifest_path.is_file():
        if recover_stage_only:
            return payload, None
        raise TwoPlaneTimingReceiverError("plane-0 stage lacks its deterministic manifest")
    manifest_payload = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TwoPlaneTimingReceiverError("plane-0 manifest is not valid JSON") from exc
    manifest = _require_manifest_scalar_types(
        manifest,
        label="plane-0",
        integer_fields=("pair_index", "pair_id", "x0_bytes", "numerator0_equal_values"),
    )
    expected_static = {
        "schema": PLANE0_MANIFEST_SCHEMA,
        "archive_sha256": archive_sha256,
        "packet_sha256": packet_sha256,
        "pair_index": pair_index,
        "pair_id": pair_index,
        "y0_sha256": _sha256(y0.tobytes(order="C")),
        "x0_sha256": _sha256(payload),
        "x0_bytes": len(payload),
        "numerator0_equal_values": manifest["numerator0_equal_values"],
        "frame0_policy_id": DESCRIPTION_FRAME0_POLICY_ID,
        "y_codec_id": PREDICTOR_RESIDUAL_Y_CODEC_ID,
        "score_claim": False,
        "promotion_eligible": False,
    }
    if _canonical_json(manifest) != manifest_payload or manifest != expected_static:
        raise TwoPlaneTimingReceiverError("preserved plane-0 manifest or bytes drifted")
    return payload, manifest


def _pair_manifest(
    *,
    archive_sha256: str,
    packet_sha256: str,
    pair_index: int,
    pair_id: int,
    y0: np.ndarray,
    y1: np.ndarray,
    frame0_payload: bytes,
    frame1_payload: bytes,
    numerator0: int,
    numerator1: int,
) -> dict[str, Any]:
    stage_payload = frame0_payload + frame1_payload
    return {
        "schema": PAIR_MANIFEST_SCHEMA,
        "archive_sha256": archive_sha256,
        "packet_sha256": packet_sha256,
        "pair_index": pair_index,
        "pair_id": pair_id,
        "y0_sha256": _sha256(y0.tobytes(order="C")),
        "y1_sha256": _sha256(y1.tobytes(order="C")),
        "x0_sha256": _sha256(frame0_payload),
        "x1_sha256": _sha256(frame1_payload),
        "stage_bytes": len(stage_payload),
        "stage_sha256": _sha256(stage_payload),
        "frame0_policy_id": DESCRIPTION_FRAME0_POLICY_ID,
        "y_codec_id": PREDICTOR_RESIDUAL_Y_CODEC_ID,
        "numerator0_equal_values": numerator0,
        "numerator1_equal_values": numerator1,
        "score_claim": False,
        "promotion_eligible": False,
    }


def _persist_pair(
    stage_root: Path,
    pair_index: int,
    stage_payload: bytes,
    manifest: Mapping[str, Any],
) -> None:
    stage_path, manifest_path = _pair_paths(stage_root, pair_index)
    _ensure_write_once_exact(stage_path, stage_payload)
    manifest_payload = _canonical_json(manifest)
    _ensure_write_once_exact(manifest_path, manifest_payload)
    if stage_path.read_bytes() != stage_payload or manifest_path.read_bytes() != manifest_payload:
        raise TwoPlaneTimingReceiverError("pair checkpoint immediate reopen differed")


def _reopen_pair(
    stage_root: Path,
    pair_index: int,
    *,
    archive_sha256: str,
    packet_sha256: str,
    y0: np.ndarray,
    y1: np.ndarray,
    operator: DisjointResizeOperator,
    camera_hw: tuple[int, int],
    recover_stage_only: bool = False,
) -> tuple[bytes, Mapping[str, Any], int, int]:
    stage_path, manifest_path = _pair_paths(stage_root, pair_index)
    if manifest_path.exists() and not stage_path.is_file():
        raise TwoPlaneTimingReceiverError("pair manifest exists without its stage bytes")
    if not stage_path.is_file():
        raise TwoPlaneTimingReceiverError("resume/final verification found an incomplete pair checkpoint")
    stage_payload = stage_path.read_bytes()
    manifest_payload = manifest_path.read_bytes() if manifest_path.is_file() else None
    if manifest_payload is None:
        if not recover_stage_only:
            raise TwoPlaneTimingReceiverError("pair stage lacks its deterministic manifest")
        manifest: Mapping[str, Any] | None = None
    else:
        try:
            manifest = json.loads(manifest_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TwoPlaneTimingReceiverError("pair manifest is not valid JSON") from exc
        manifest = _require_manifest_scalar_types(
            manifest,
            label="pair",
            integer_fields=(
                "pair_index",
                "pair_id",
                "stage_bytes",
                "numerator0_equal_values",
                "numerator1_equal_values",
            ),
        )
    frame_bytes = camera_hw[0] * camera_hw[1] * 3
    if len(stage_payload) != frame_bytes * 2:
        raise TwoPlaneTimingReceiverError("preserved pair stage byte count drifted")
    frame0_payload = stage_payload[:frame_bytes]
    frame1_payload = stage_payload[frame_bytes:]
    numerator0 = _verify_one(operator, _frame_from_payload(frame0_payload, camera_hw), y0)
    numerator1 = _verify_one(operator, _frame_from_payload(frame1_payload, camera_hw), y1)
    expected = _pair_manifest(
        archive_sha256=archive_sha256,
        packet_sha256=packet_sha256,
        pair_index=pair_index,
        pair_id=pair_index,
        y0=y0,
        y1=y1,
        frame0_payload=frame0_payload,
        frame1_payload=frame1_payload,
        numerator0=numerator0,
        numerator1=numerator1,
    )
    if manifest is None:
        _persist_pair(stage_root, pair_index, stage_payload, expected)
        manifest = expected
    elif _canonical_json(manifest) != manifest_payload or manifest != expected:
        raise TwoPlaneTimingReceiverError("preserved pair manifest or stage custody drifted")
    return stage_payload, manifest, numerator0, numerator1


def _read_pair_checkpoint(
    stage_root: Path,
    pair_index: int,
    *,
    camera_hw: tuple[int, int],
) -> tuple[bytes, Mapping[str, Any]]:
    stage_path, manifest_path = _pair_paths(stage_root, pair_index)
    if not stage_path.is_file() or not manifest_path.is_file():
        raise TwoPlaneTimingReceiverError("final verification found an incomplete pair checkpoint")
    stage_payload = stage_path.read_bytes()
    manifest_payload = manifest_path.read_bytes()
    if len(stage_payload) != camera_hw[0] * camera_hw[1] * 3 * 2:
        raise TwoPlaneTimingReceiverError("final pair checkpoint byte count drifted")
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TwoPlaneTimingReceiverError("final pair manifest is not valid JSON") from exc
    manifest = _require_manifest_scalar_types(
        manifest,
        label="final pair",
        integer_fields=(
            "pair_index",
            "pair_id",
            "stage_bytes",
            "numerator0_equal_values",
            "numerator1_equal_values",
        ),
    )
    if _canonical_json(manifest) != manifest_payload:
        raise TwoPlaneTimingReceiverError("final pair manifest is not canonical JSON")
    return stage_payload, manifest


def _chunk_manifest(stage_root: Path, chunk_index: int, pair_indices: Sequence[int]) -> Mapping[str, Any]:
    pair_rows = []
    for pair_index in pair_indices:
        _stage_path, manifest_path = _pair_paths(stage_root, pair_index)
        payload = manifest_path.read_bytes()
        pair_rows.append(
            {
                "pair_index": pair_index,
                "pair_manifest_bytes": len(payload),
                "pair_manifest_sha256": _sha256(payload),
            }
        )
    return {
        "schema": CHUNK_MANIFEST_SCHEMA,
        "chunk_index": chunk_index,
        "pair_start": pair_indices[0],
        "pair_stop_exclusive": pair_indices[-1] + 1,
        "pair_manifests": pair_rows,
        "score_claim": False,
        "promotion_eligible": False,
    }


def _persist_complete_chunks(stage_root: Path, completed_pairs: int, pair_count: int) -> int:
    chunks = 0
    for start in range(0, pair_count, CHUNK_PAIRS):
        stop = min(start + CHUNK_PAIRS, pair_count)
        if stop > completed_pairs:
            break
        indices = tuple(range(start, stop))
        payload = _canonical_json(_chunk_manifest(stage_root, start // CHUNK_PAIRS, indices))
        path = stage_root / "chunk_manifests" / f"chunk-{start // CHUNK_PAIRS:04d}.json"
        _ensure_write_once_exact(path, payload)
        chunks += 1
    return chunks


def _verify_chunks(stage_root: Path, pair_count: int) -> None:
    for start in range(0, pair_count, CHUNK_PAIRS):
        stop = min(start + CHUNK_PAIRS, pair_count)
        path = stage_root / "chunk_manifests" / f"chunk-{start // CHUNK_PAIRS:04d}.json"
        expected = _canonical_json(_chunk_manifest(stage_root, start // CHUNK_PAIRS, tuple(range(start, stop))))
        if not path.is_file() or path.read_bytes() != expected:
            raise TwoPlaneTimingReceiverError("chunk manifest custody drifted")


def _source_hashes() -> Mapping[str, str]:
    receiver = Path(__file__).resolve()
    production = receiver.with_name("v10_production_receiver.py")
    solver = receiver.parents[1] / "optimization" / "uint8_lattice_feasibility.py"
    return {
        "timed_receiver_sha256": _sha256_file(receiver),
        "production_receiver_sha256": _sha256_file(production),
        "integer_solver_sha256": _sha256_file(solver),
    }


def _thread_environment() -> Mapping[str, str | None]:
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {name: os.environ.get(name) for name in names}


def _storage_preflight(output_root: Path, required_bytes: int) -> Mapping[str, Any]:
    required = _exact_int(required_bytes, "required storage bytes")
    probe = output_root.resolve()
    while not probe.exists():
        if probe.parent == probe:
            raise TwoPlaneTimingReceiverError("no existing parent for storage preflight")
        probe = probe.parent
    stats = os.statvfs(probe)
    free = int(stats.f_bavail * stats.f_frsize)
    passed = free >= required
    row = {
        "schema": "v10_two_plane_receiver_storage_preflight.v1",
        "probe_path": str(probe),
        "required_bytes": required,
        "free_bytes": free,
        "passed": passed,
    }
    if not passed:
        raise TwoPlaneTimingReceiverError(f"receiver storage preflight refused: free={free}, required={required}")
    return row


def _write_timing_receipt(
    path: Path,
    receipt: Mapping[str, Any],
) -> tuple[bytes, str]:
    payload = _canonical_json(receipt)
    _atomic_write_once(path, payload)
    return payload, _sha256(payload)


def _final_verify_complete(
    *,
    state_root: Path,
    raw_path: Path,
    raw_sha256: str,
    packet_bytes: bytes,
    archive_sha256: str,
    packet_sha256: str,
    y0: np.ndarray,
    y1: np.ndarray,
    operator: DisjointResizeOperator,
    camera_hw: tuple[int, int],
    workers: int,
    raw_expected: int,
    numerator_expected: int,
) -> tuple[int, float]:
    """Reopen and re-prove every deterministic output under guaranteed pool cleanup."""

    final_executor: ProcessPoolExecutor | None = None
    if workers >= 4:
        try:
            final_executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_process_initializer,
                initargs=(operator,),
            )
        except (OSError, RuntimeError) as exc:
            raise TwoPlaneTimingReceiverError(
                f"final verification process pool host-custody refusal: {type(exc).__name__}: {exc}"
            ) from exc
    try:
        start = time.monotonic()
        if reserialize_parsed_packet(parse_packet(packet_bytes)) != packet_bytes:
            raise TwoPlaneTimingReceiverError("final strict packet re-encode differs")
        final_verified = 0
        raw_digest = hashlib.sha256()
        raw_reopened_bytes = 0
        frame_bytes = camera_hw[0] * camera_hw[1] * 3
        with raw_path.open("rb") as raw_handle:
            for batch_start in range(0, len(y0), workers):
                indices = tuple(range(batch_start, min(batch_start + workers, len(y0))))
                checkpoints = {
                    pair_index: _read_pair_checkpoint(
                        state_root,
                        pair_index,
                        camera_hw=camera_hw,
                    )
                    for pair_index in indices
                }
                payload0 = {index: checkpoints[index][0][:frame_bytes] for index in indices}
                payload1 = {index: checkpoints[index][0][frame_bytes:] for index in indices}
                proofs0, _verify0 = _verify_phase(operator, final_executor, payload0, y0)
                proofs1, _verify1 = _verify_phase(operator, final_executor, payload1, y1)
                for pair_index in indices:
                    stage_payload, manifest = checkpoints[pair_index]
                    expected_manifest = _pair_manifest(
                        archive_sha256=archive_sha256,
                        packet_sha256=packet_sha256,
                        pair_index=pair_index,
                        pair_id=pair_index,
                        y0=y0[pair_index],
                        y1=y1[pair_index],
                        frame0_payload=payload0[pair_index],
                        frame1_payload=payload1[pair_index],
                        numerator0=proofs0[pair_index].numerator_equal_values,
                        numerator1=proofs1[pair_index].numerator_equal_values,
                    )
                    if manifest != expected_manifest:
                        raise TwoPlaneTimingReceiverError("final pair manifest custody drifted")
                    raw_payload = raw_handle.read(len(stage_payload))
                    if raw_payload != stage_payload:
                        raise TwoPlaneTimingReceiverError("assembled raw differs from reopened pair stage")
                    raw_digest.update(raw_payload)
                    raw_reopened_bytes += len(raw_payload)
                    final_verified += (
                        proofs0[pair_index].numerator_equal_values + proofs1[pair_index].numerator_equal_values
                    )
            if raw_handle.read(1):
                raise TwoPlaneTimingReceiverError("assembled raw has trailing bytes")
        _verify_chunks(state_root, len(y0))
        if (
            raw_reopened_bytes != raw_expected
            or raw_digest.hexdigest() != raw_sha256
            or final_verified != numerator_expected
        ):
            raise TwoPlaneTimingReceiverError("final raw or numerator proof totals drifted")
        return final_verified, _positive_elapsed(start)
    finally:
        if final_executor is not None:
            final_executor.shutdown(wait=True, cancel_futures=True)


def timed_inflate_two_plane_archive(
    archive_dir: Path | str,
    output_dir: Path | str,
    video_names_file: Path | str,
    *,
    timing_receipt_path: Path | str,
    resume: bool = False,
    stop_after_pairs: int | None = None,
    stop_after_plane0_pairs: int | None = None,
    workers: int = 1,
    expected_pair_count: int = FULL_PAIR_COUNT,
    expected_camera_hw: tuple[int, int] = FULL_CAMERA_HW,
    expected_scorer_hw: tuple[int, int] = FULL_SCORER_HW,
    expected_y0_sha256: str | None = None,
    expected_y1_sha256: str | None = None,
) -> TwoPlaneTimedInflateResult:
    """Run one timed scorer-free C1 inflate invocation.

    ``workers=1`` is the serial attribution baseline.  Any parallel invocation
    requires at least four worker processes, matching the declared contest CPU
    class.  The output bytes and deterministic manifests are independent of
    worker count.
    """

    total_start = time.monotonic()
    pair_count_expected = _exact_int(expected_pair_count, "expected_pair_count", minimum=1)
    workers = _exact_int(workers, "workers", minimum=1)
    if workers not in (1,) and workers < 4:
        raise TwoPlaneTimingReceiverError("parallel C1 execution requires at least four workers")
    if type(resume) is not bool:
        raise TwoPlaneTimingReceiverError("resume must be boolean")
    timing_receipt = Path(timing_receipt_path)
    archive_root = Path(archive_dir)
    output_root = Path(output_dir)
    if _path_is_within(timing_receipt, output_root):
        raise TwoPlaneTimingReceiverError("timing receipt must remain outside the hashed output root")
    if timing_receipt.exists():
        raise TwoPlaneTimingReceiverError("timing receipt is write-once")
    video_name = _safe_video_name(Path(video_names_file))
    raw_path = _safe_raw_output_path(output_root, video_name)
    state_root = output_root / STATE_DIRECTORY / raw_path.relative_to(output_root).with_suffix("")
    if not resume and output_root.exists() and any(output_root.iterdir()):
        raise TwoPlaneTimingReceiverError("fresh timing invocation requires a fresh output directory")

    parse_start = time.monotonic()
    archive_bytes, packet_bytes, archive_input_kind = _read_exact_archive_root(archive_root)
    try:
        packet = parse_packet(packet_bytes)
    except ProductionReceiverError as exc:
        raise TwoPlaneTimingReceiverError("production packet parsing refused") from exc
    _validate_packet(packet)
    parse_seconds = _positive_elapsed(parse_start)

    expansion_start = time.monotonic()
    y_codec_id = packet.header.get("y_codec_id")
    try:
        if y_codec_id == JXL_PLANE_Y_CODEC_ID:
            expanded = decode_jxl_plane_payload(packet.section("y_description").payload)
        else:
            expanded = decode_predictor_residual(packet.section("y_description").payload)
    except Exception as exc:
        raise TwoPlaneTimingReceiverError("y description expansion refused") from exc
    y0, y1 = _validate_targets(
        expanded.frame0,
        expanded.frame1,
        expanded.pair_ids,
        expected_pair_count=pair_count_expected,
        expected_scorer_hw=expected_scorer_hw,
        expected_y0_sha256=(
            FULL_Y0_SHA256
            if pair_count_expected == FULL_PAIR_COUNT and expected_y0_sha256 is None
            else expected_y0_sha256
        ),
        expected_y1_sha256=(
            FULL_Y1_SHA256
            if pair_count_expected == FULL_PAIR_COUNT and expected_y1_sha256 is None
            else expected_y1_sha256
        ),
    )
    expansion_seconds = _positive_elapsed(expansion_start)

    geometry = packet.header.get("geometry")
    if not isinstance(geometry, Mapping):
        raise TwoPlaneTimingReceiverError("production geometry is missing")
    camera_hw = (int(geometry["camera_height"]), int(geometry["camera_width"]))
    scorer_hw = (int(geometry["scorer_height"]), int(geometry["scorer_width"]))
    if camera_hw != expected_camera_hw or scorer_hw != expected_scorer_hw:
        raise TwoPlaneTimingReceiverError("packet geometry differs from the declared C1 geometry")
    if int(packet.header["pair_count"]) != pair_count_expected:
        raise TwoPlaneTimingReceiverError("packet pair count differs from the declared C1 count")
    operator = DisjointResizeOperator.build(
        camera_h=camera_hw[0],
        camera_w=camera_hw[1],
        scorer_h=scorer_hw[0],
        scorer_w=scorer_hw[1],
    )
    archive_sha = _sha256(archive_bytes)
    packet_sha = _sha256(packet_bytes)
    frame_bytes = camera_hw[0] * camera_hw[1] * 3
    pair_stage_bytes = frame_bytes * 2
    raw_expected = pair_count_expected * pair_stage_bytes
    numerator_expected = pair_count_expected * 2 * scorer_hw[0] * scorer_hw[1] * 3
    if pair_count_expected == FULL_PAIR_COUNT and (
        raw_expected != FULL_RAW_BYTES or numerator_expected != FULL_NUMERATOR_VALUES
    ):
        raise TwoPlaneTimingReceiverError("full-n600 exact count constants drifted")
    if stop_after_pairs is None:
        limit = pair_count_expected
    else:
        limit = min(_exact_int(stop_after_pairs, "stop_after_pairs"), pair_count_expected)
    plane0_stop = (
        None
        if stop_after_plane0_pairs is None
        else min(
            _exact_int(stop_after_plane0_pairs, "stop_after_plane0_pairs", minimum=1),
            pair_count_expected,
        )
    )

    missing_plane0_bytes = sum(
        frame_bytes
        for pair_index in range(pair_count_expected)
        if not _plane0_paths(state_root, pair_index)[0].is_file()
    )
    missing_pair_bytes = sum(
        pair_stage_bytes
        for pair_index in range(pair_count_expected)
        if not _pair_paths(state_root, pair_index)[0].is_file()
    )
    required_storage = (
        missing_plane0_bytes + missing_pair_bytes + (0 if raw_path.is_file() else raw_expected) + (64 << 20)
    )
    storage_preflight = _storage_preflight(output_root, required_storage)

    resumed_pairs = 0
    verified_values = 0
    verification_seconds = 0.0
    solve0_seconds = 0.0
    solve1_seconds = 0.0
    assembly_io_seconds = 0.0
    per_pair: list[dict[str, Any]] = []
    next_index = 0
    if not resume and state_root.exists():
        raise TwoPlaneTimingReceiverError("fresh invocation refuses preserved receiver state")

    executor: ProcessPoolExecutor | None = None
    if workers >= 4:
        try:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_process_initializer,
                initargs=(operator,),
            )
        except (OSError, RuntimeError) as exc:
            raise TwoPlaneTimingReceiverError(
                f"solve process pool host-custody refusal: {type(exc).__name__}: {exc}"
            ) from exc
    try:
        if resume:
            resume_start = time.monotonic()
            while next_index < limit:
                stage_path, manifest_path = _pair_paths(state_root, next_index)
                if not stage_path.exists() and not manifest_path.exists():
                    break
                _stage, _manifest, count0, count1 = _reopen_pair(
                    state_root,
                    next_index,
                    archive_sha256=archive_sha,
                    packet_sha256=packet_sha,
                    y0=y0[next_index],
                    y1=y1[next_index],
                    operator=operator,
                    camera_hw=camera_hw,
                    recover_stage_only=True,
                )
                x0_payload, x0_manifest = _reopen_plane0_bytes(
                    state_root,
                    next_index,
                    archive_sha256=archive_sha,
                    packet_sha256=packet_sha,
                    y0=y0[next_index],
                    recover_stage_only=True,
                )
                x0_count = _verify_one(
                    operator,
                    _frame_from_payload(x0_payload, camera_hw),
                    y0[next_index],
                )
                expected_x0_manifest = _plane0_manifest(
                    archive_sha256=archive_sha,
                    packet_sha256=packet_sha,
                    pair_index=next_index,
                    y0=y0[next_index],
                    frame0_payload=x0_payload,
                    numerator0=x0_count,
                )
                if x0_manifest is None:
                    _persist_plane0(state_root, next_index, x0_payload, expected_x0_manifest)
                elif x0_manifest != expected_x0_manifest:
                    raise TwoPlaneTimingReceiverError("resumed plane-0 checkpoint custody drifted")
                verified_values += count0 + count1
                resumed_pairs += 1
                per_pair.append(
                    {
                        "pair_index": next_index,
                        "solve0_seconds": 0.0,
                        "solve1_seconds": 0.0,
                        "resumed": True,
                        "resumed_plane0": True,
                    }
                )
                next_index += 1
            verification_seconds += _positive_elapsed(resume_start)

        stopped_after_plane0 = False
        while next_index < limit:
            indices = tuple(range(next_index, min(next_index + workers, limit)))
            frame0_payloads: dict[int, bytes] = {}
            existing0_manifests: dict[int, Mapping[str, Any] | None] = {}
            solve0_rows: dict[int, _SolvedPlane] = {}
            missing0: list[int] = []
            for pair_index in indices:
                x0_path, x0_manifest_path = _plane0_paths(state_root, pair_index)
                if resume and (x0_path.exists() or x0_manifest_path.exists()):
                    payload, existing_manifest = _reopen_plane0_bytes(
                        state_root,
                        pair_index,
                        archive_sha256=archive_sha,
                        packet_sha256=packet_sha,
                        y0=y0[pair_index],
                        recover_stage_only=True,
                    )
                    frame0_payloads[pair_index] = payload
                    existing0_manifests[pair_index] = existing_manifest
                else:
                    missing0.append(pair_index)
            if missing0:
                solved0, elapsed0 = _solve_phase(operator, executor, tuple(missing0), y0)
                solve0_seconds += elapsed0
                solve0_rows.update(solved0)
                frame0_payloads.update({index: row.payload for index, row in solved0.items()})
            verified0, elapsed_verify0 = _verify_phase(operator, executor, frame0_payloads, y0)
            verification_seconds += elapsed_verify0
            write0_start = time.monotonic()
            for pair_index in indices:
                proof0 = verified0[pair_index]
                expected0 = _plane0_manifest(
                    archive_sha256=archive_sha,
                    packet_sha256=packet_sha,
                    pair_index=pair_index,
                    y0=y0[pair_index],
                    frame0_payload=frame0_payloads[pair_index],
                    numerator0=proof0.numerator_equal_values,
                )
                existing0 = existing0_manifests.get(pair_index)
                if pair_index in missing0 or existing0 is None:
                    _persist_plane0(state_root, pair_index, frame0_payloads[pair_index], expected0)
                else:
                    if existing0 != expected0:
                        raise TwoPlaneTimingReceiverError("resumed plane-0 numerator custody drifted")
            assembly_io_seconds += _positive_elapsed(write0_start)
            if plane0_stop is not None and indices[-1] + 1 >= plane0_stop:
                verified_values += sum(row.numerator_equal_values for row in verified0.values())
                for pair_index in indices:
                    per_pair.append(
                        {
                            "pair_index": pair_index,
                            "solve0_seconds": (
                                solve0_rows[pair_index].solve_seconds if pair_index in solve0_rows else 0.0
                            ),
                            "solve1_seconds": 0.0,
                            "resumed": False,
                            "resumed_plane0": pair_index not in missing0,
                            "plane0_only": True,
                        }
                    )
                stopped_after_plane0 = True
                break

            solved1, elapsed1 = _solve_phase(operator, executor, indices, y1)
            solve1_seconds += elapsed1
            verified1, elapsed_verify1 = _verify_phase(
                operator,
                executor,
                {index: row.payload for index, row in solved1.items()},
                y1,
            )
            verification_seconds += elapsed_verify1
            rows: list[tuple[int, bytes, Mapping[str, Any], int, int]] = []
            for pair_index in indices:
                row1 = solved1[pair_index]
                count0 = verified0[pair_index].numerator_equal_values
                count1 = verified1[pair_index].numerator_equal_values
                frame0_payload = frame0_payloads[pair_index]
                stage_payload = frame0_payload + row1.payload
                manifest = _pair_manifest(
                    archive_sha256=archive_sha,
                    packet_sha256=packet_sha,
                    pair_index=pair_index,
                    pair_id=expanded.pair_ids[pair_index],
                    y0=y0[pair_index],
                    y1=y1[pair_index],
                    frame0_payload=frame0_payload,
                    frame1_payload=row1.payload,
                    numerator0=count0,
                    numerator1=count1,
                )
                rows.append((pair_index, stage_payload, manifest, count0, count1))
                per_pair.append(
                    {
                        "pair_index": pair_index,
                        "solve0_seconds": solve0_rows[pair_index].solve_seconds if pair_index in solve0_rows else 0.0,
                        "solve1_seconds": row1.solve_seconds,
                        "resumed": False,
                        "resumed_plane0": pair_index not in missing0,
                    }
                )
            write_start = time.monotonic()
            for pair_index, stage_payload, manifest, count0, count1 in rows:
                _persist_pair(state_root, pair_index, stage_payload, manifest)
                verified_values += count0 + count1
            assembly_io_seconds += _positive_elapsed(write_start)
            next_index = indices[-1] + 1
            chunk_start = time.monotonic()
            _persist_complete_chunks(state_root, next_index, pair_count_expected)
            assembly_io_seconds += _positive_elapsed(chunk_start)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            executor = None

    completed = limit == pair_count_expected and not stopped_after_plane0
    raw_sha: str | None = None
    output_tree_sha: str | None = None
    if completed:
        assembly_start = time.monotonic()
        _persist_complete_chunks(state_root, pair_count_expected, pair_count_expected)
        stage_paths = [_pair_paths(state_root, pair_index)[0] for pair_index in range(pair_count_expected)]
        raw_sha, raw_bytes = _atomic_replace_from_stages(
            raw_path,
            stage_paths,
            expected_bytes=raw_expected,
        )
        assembly_io_seconds += _positive_elapsed(assembly_start)

        final_verified, final_verify_seconds = _final_verify_complete(
            state_root=state_root,
            raw_path=raw_path,
            raw_sha256=raw_sha,
            packet_bytes=packet_bytes,
            archive_sha256=archive_sha,
            packet_sha256=packet_sha,
            y0=y0,
            y1=y1,
            operator=operator,
            camera_hw=camera_hw,
            workers=workers,
            raw_expected=raw_expected,
            numerator_expected=numerator_expected,
        )
        verified_values = final_verified
        verification_seconds += final_verify_seconds
        output_tree_sha = _tree_sha256(output_root)
    else:
        raw_bytes = 0

    component = {
        "parse_seconds": parse_seconds,
        "expansion_seconds": expansion_seconds,
        "solve0_seconds": solve0_seconds,
        "solve1_seconds": solve1_seconds,
        "assembly_io_seconds": assembly_io_seconds,
        "verification_seconds": verification_seconds,
    }
    if completed and not resume and any(not math.isfinite(value) or value <= 0 for value in component.values()):
        raise TwoPlaneTimingReceiverError("complete timing receipt requires six positive finite components")
    if any(not math.isfinite(value) or value < 0 for value in component.values()):
        raise TwoPlaneTimingReceiverError("timing components must be finite and nonnegative")
    component_sum = sum(component.values())
    stage_tree_sha = _tree_sha256(state_root / "pairs")
    plane0_tree_sha = _tree_sha256(state_root / "plane0")
    pair_manifest_tree_sha = _tree_sha256(state_root / "pair_manifests")
    plane0_manifest_tree_sha = _tree_sha256(state_root / "plane0_manifests")
    chunk_tree_sha = _tree_sha256(state_root / "chunk_manifests")
    preserved_stage_bytes = sum(
        path.stat().st_size
        for root in (state_root / "plane0", state_root / "pairs")
        if root.exists()
        for path in root.glob("*.bin")
    )
    y0_chunk_digests = [
        _sha256(y0[start : min(start + CHUNK_PAIRS, pair_count_expected)].tobytes(order="C"))
        for start in range(0, pair_count_expected, CHUNK_PAIRS)
    ]
    y1_chunk_digests = [
        _sha256(y1[start : min(start + CHUNK_PAIRS, pair_count_expected)].tobytes(order="C"))
        for start in range(0, pair_count_expected, CHUNK_PAIRS)
    ]
    source_hashes = _source_hashes()
    host_custody = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "pid": os.getpid(),
    }
    thread_environment = _thread_environment()
    invocation_argv = list(sys.argv)
    # Close only after every decode/custody computation that precedes receipt
    # serialization. The measurement CLI records the true outer call wall,
    # including receipt persistence, on a separate non-self-referential row.
    total_seconds = _positive_elapsed(total_start)
    unclassified = max(0.0, total_seconds - component_sum)
    receipt: dict[str, Any] = {
        "schema": TIMING_RECEIPT_SCHEMA,
        "axis": CPU_TIMING_AXIS,
        "completed": completed,
        "fresh": not resume,
        "resume_requested": resume,
        "resumed_pairs": resumed_pairs,
        "execution": {
            "mode": "serial" if workers == 1 else "process_pool",
            "workers": workers,
            "fixed_pair_assembly_order": True,
            "timed_solver_operator_builds": 1,
        },
        "timing": {
            **component,
            "component_sum_seconds": component_sum,
            "total_seconds": total_seconds,
            "total_boundary": "entry_through_pre_receipt_evidence_collection",
            "receipt_serialization_and_persistence_included": False,
            "unclassified_overhead_seconds": unclassified,
            "per_pair": sorted(per_pair, key=lambda row: int(row["pair_index"])),
        },
        "pair_count": pair_count_expected,
        "chunk_count": (pair_count_expected + CHUNK_PAIRS - 1) // CHUNK_PAIRS
        if completed
        else next_index // CHUNK_PAIRS,
        "pair_stages_preserved": next_index,
        "plane0_stages_preserved": sum(
            1 for pair_index in range(pair_count_expected) if _plane0_paths(state_root, pair_index)[0].is_file()
        ),
        "preserved_stage_bytes": preserved_stage_bytes,
        "archive_sha256": archive_sha,
        "archive_bytes": len(archive_bytes),
        "archive_input_kind": archive_input_kind,
        "canonical_archive_reconstructed": archive_input_kind == "extracted_0_bin",
        "packet_sha256": packet_sha,
        "packet_bytes": len(packet_bytes),
        "strict_packet_reencode_identical": True,
        "canonical_zip_identical": True,
        "frame0_policy_id": DESCRIPTION_FRAME0_POLICY_ID,
        "y_codec_id": PREDICTOR_RESIDUAL_Y_CODEC_ID,
        "y0_sha256": _sha256(y0.tobytes(order="C")),
        "y1_sha256": _sha256(y1.tobytes(order="C")),
        "y0_chunk_sha256": y0_chunk_digests,
        "y1_chunk_sha256": y1_chunk_digests,
        "raw_sha256": raw_sha,
        "raw_bytes": raw_bytes,
        "raw_relative_path": raw_path.relative_to(output_root).as_posix() if completed else None,
        "output_root": str(output_root.resolve()),
        "state_root": str(state_root.resolve()),
        "stage_tree_sha256": stage_tree_sha,
        "plane0_tree_sha256": plane0_tree_sha,
        "pair_manifest_tree_sha256": pair_manifest_tree_sha,
        "plane0_manifest_tree_sha256": plane0_manifest_tree_sha,
        "chunk_tree_sha256": chunk_tree_sha,
        "output_tree_sha256": output_tree_sha,
        "numerator_values_verified": verified_values,
        "numerator_values_expected": numerator_expected,
        "both_planes_exact": completed and verified_values == numerator_expected,
        "storage_preflight": storage_preflight,
        "host": host_custody,
        "thread_environment": thread_environment,
        "argv": invocation_argv,
        "source_hashes": source_hashes,
        "contest_budget_verdict": None,
        "contest_budget_authority": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    receipt_payload, receipt_sha = _write_timing_receipt(timing_receipt, receipt)
    if timing_receipt.read_bytes() != receipt_payload:
        raise TwoPlaneTimingReceiverError("timing receipt immediate reopen differed")
    return TwoPlaneTimedInflateResult(
        completed=completed,
        raw_path=raw_path if completed else None,
        raw_sha256=raw_sha,
        raw_bytes=raw_bytes,
        pair_stages_preserved=next_index,
        resumed_pairs=resumed_pairs,
        numerator_values_verified=verified_values,
        stage_tree_sha256=stage_tree_sha,
        chunk_tree_sha256=chunk_tree_sha,
        output_tree_sha256=output_tree_sha,
        timing_receipt_path=timing_receipt,
        timing_receipt_sha256=receipt_sha,
        receipt=receipt,
    )


def _mlx_index_map(operator: DisjointResizeOperator) -> tuple[np.ndarray, np.ndarray]:
    indices = np.zeros(operator.camera_h * operator.camera_w, dtype=np.int32)
    valid = np.zeros(operator.camera_h * operator.camera_w, dtype=bool)
    for scorer_row, row_support in enumerate(operator.row_supports):
        for scorer_col, col_support in enumerate(operator.col_supports):
            source_index = scorer_row * operator.scorer_w + scorer_col
            for camera_row in row_support.indices:
                for camera_col in col_support.indices:
                    flat = int(camera_row) * operator.camera_w + int(camera_col)
                    if valid[flat]:
                        raise TwoPlaneTimingReceiverError("MLX map found overlapping camera ownership")
                    valid[flat] = True
                    indices[flat] = source_index
    return indices, valid


def build_mlx_factor2_plan(
    operator: DisjointResizeOperator,
    *,
    mlx_module: Any,
) -> MlxFactor2Plan:
    """Build and transfer the static support map exactly once."""

    indices, valid = _mlx_index_map(operator)
    return MlxFactor2Plan(
        camera_hw=(operator.camera_h, operator.camera_w),
        scorer_hw=(operator.scorer_h, operator.scorer_w),
        device_indices=mlx_module.array(indices),
        device_valid=mlx_module.array(valid)[:, None],
    )


def realize_factor2_uint8_scorer_plane_mlx(
    operator: DisjointResizeOperator,
    target: np.ndarray,
    *,
    mlx_module: Any | None = None,
    plan: MlxFactor2Plan | None = None,
) -> np.ndarray:
    """Metal-backed integer support fill returning host ``uint8`` bytes.

    The result has no authority until compared byte-for-byte with the NumPy
    integer reference by :func:`parity_check_mlx_two_plane`.
    """

    if not isinstance(operator, DisjointResizeOperator):
        raise TwoPlaneTimingReceiverError("MLX twin requires the certified resize operator")
    y = np.asarray(target)
    if y.dtype != np.uint8 or y.shape != (operator.scorer_h, operator.scorer_w, 3):
        raise TwoPlaneTimingReceiverError("MLX twin target must be uint8 scorer-plane RGB")
    if mlx_module is None:
        try:
            mlx_module = importlib.import_module("mlx.core")
        except ImportError as exc:
            raise TwoPlaneTimingReceiverError("MLX is unavailable on this host") from exc
    selected_plan = plan or build_mlx_factor2_plan(operator, mlx_module=mlx_module)
    if selected_plan.camera_hw != (operator.camera_h, operator.camera_w) or selected_plan.scorer_hw != (
        operator.scorer_h,
        operator.scorer_w,
    ):
        raise TwoPlaneTimingReceiverError("MLX support plan geometry differs from the operator")
    source = mlx_module.array(np.ascontiguousarray(y).reshape(-1, 3))
    gathered = mlx_module.take(source, selected_plan.device_indices, axis=0)
    zeros = mlx_module.zeros_like(gathered)
    output = mlx_module.where(selected_plan.device_valid, gathered, zeros)
    mlx_module.eval(output)
    host_native = np.asarray(output)
    if host_native.dtype != np.uint8:
        raise TwoPlaneTimingReceiverError(
            f"MLX twin produced native dtype {host_native.dtype}; coercive uint8 conversion is forbidden"
        )
    host = host_native.reshape(operator.camera_h, operator.camera_w, 3)
    return np.ascontiguousarray(host)


def mlx_runtime_status(*, mlx_module: Any | None = None) -> Mapping[str, Any]:
    """Prove the real ``mlx.core`` runtime and an evaluated Metal integer op.

    Injected modules remain useful for algorithmic unit tests, but they can
    never establish host/device custody.  The real probe runs in a disposable
    process so failed Metal initialization and atexit handlers stay contained.
    """

    if mlx_module is not None:
        return {
            "runtime_installed": None,
            "backend_identity_verified": False,
            "metal_usable": False,
            "backend_module": getattr(mlx_module, "__name__", type(mlx_module).__name__),
            "host_custody_refusal": "injected MLX backend cannot establish mlx.core/Metal custody",
        }
    probe_source = r"""
import json
import os
import sys

import numpy as np

row = {
    "runtime_installed": False,
    "backend_identity_verified": False,
    "metal_usable": False,
}
try:
    import mlx
    import mlx.core as mx
    row["runtime_installed"] = True
    row["backend_module"] = mx.__name__
    row["backend_file"] = str(getattr(mx, "__file__", None))
    row["runtime_version"] = str(getattr(mlx, "__version__", None))
    if mx.__name__ != "mlx.core":
        raise RuntimeError("import did not resolve to mlx.core")
    row["backend_identity_verified"] = True
    row["reported_metal_available"] = bool(mx.metal.is_available())
    if not row["reported_metal_available"]:
        raise RuntimeError("mx.metal.is_available() returned false")
    previous = mx.default_device()
    try:
        mx.set_default_device(mx.gpu)
        row["default_device"] = str(mx.default_device())
        if "gpu" not in row["default_device"].lower():
            raise RuntimeError("MLX default device is not GPU")
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        row["device_info"] = str(info)
        source = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.uint8)
        indices = mx.array([1, 0], dtype=mx.int32)
        gathered = mx.take(source, indices, axis=0)
        output = mx.where(mx.array([[True], [True]]), gathered, mx.zeros_like(gathered))
        mx.eval(output)
        host = np.asarray(output)
        row["canary_dtype"] = str(host.dtype)
        row["canary_values"] = host.tolist()
        if host.dtype != np.uint8 or row["canary_values"] != [[4, 5, 6], [1, 2, 3]]:
            raise RuntimeError("evaluated Metal integer canary drifted")
    finally:
        mx.set_default_device(previous)
    row["metal_usable"] = True
except BaseException as exc:
    row["host_custody_refusal"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(row, sort_keys=True))
    sys.stdout.flush()
    os._exit(7)
print(json.dumps(row, sort_keys=True))
sys.stdout.flush()
os._exit(0)
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe_source],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise TwoPlaneTimingReceiverError(f"MLX Metal custody probe emitted no JSON: {completed.stderr[-1000:]!r}")
        row = json.loads(lines[-1])
        if not isinstance(row, dict):
            raise TwoPlaneTimingReceiverError("MLX Metal custody probe emitted a non-object")
        row["probe_returncode"] = completed.returncode
        if completed.returncode != 0 or row.get("metal_usable") is not True:
            row["metal_usable"] = False
            row.setdefault(
                "host_custody_refusal",
                completed.stderr[-1000:] or "MLX Metal custody canary failed",
            )
        return row
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, TwoPlaneTimingReceiverError) as exc:
        return {
            "runtime_installed": None,
            "backend_identity_verified": False,
            "metal_usable": False,
            "host_custody_refusal": f"MLX Metal custody probe failed closed: {exc}",
        }


def parity_check_mlx_two_plane(
    operator: DisjointResizeOperator,
    frame0_planes: np.ndarray,
    frame1_planes: np.ndarray,
    *,
    pair_ids: Sequence[int],
    mlx_module: Any | None = None,
    test_only_allow_unverified_backend: bool = False,
) -> Mapping[str, Any]:
    """Parity-gate at least six independent two-plane pairs against NumPy."""

    y0 = np.asarray(frame0_planes)
    y1 = np.asarray(frame1_planes)
    if type(test_only_allow_unverified_backend) is not bool:
        raise TwoPlaneTimingReceiverError("test-only backend allowance must be boolean")
    ids = tuple(_exact_int(value, f"pair_ids[{index}]") for index, value in enumerate(pair_ids))
    if len(ids) < 6 or len(ids) != len(set(ids)):
        raise TwoPlaneTimingReceiverError("MLX parity requires at least six unique real pair ids")
    if y0.dtype != np.uint8 or y1.dtype != np.uint8 or y0.shape != y1.shape:
        raise TwoPlaneTimingReceiverError("MLX parity planes must be equal-shape uint8 arrays")
    if y0.shape != (len(ids), operator.scorer_h, operator.scorer_w, 3):
        raise TwoPlaneTimingReceiverError("MLX parity plane geometry or pair count drifted")
    runtime = mlx_runtime_status(mlx_module=mlx_module)
    using_test_backend = (
        mlx_module is not None and runtime["metal_usable"] is not True and test_only_allow_unverified_backend
    )
    if runtime["metal_usable"] is not True and not using_test_backend:
        raise TwoPlaneTimingReceiverError(str(runtime["host_custody_refusal"]))
    if runtime["metal_usable"] is True and runtime.get("backend_identity_verified") is not True:
        raise TwoPlaneTimingReceiverError("MLX Metal custody lacks verified mlx.core identity")
    if mlx_module is None:
        mlx_module = importlib.import_module("mlx.core")
    rows: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    total_start = time.monotonic()
    previous_device: Any | None = None
    real_device_pinned = False
    try:
        if not using_test_backend:
            previous_device = mlx_module.default_device()
            mlx_module.set_default_device(mlx_module.gpu)
            real_device_pinned = True
        plan = build_mlx_factor2_plan(operator, mlx_module=mlx_module)
        for index, pair_id in enumerate(ids):
            row: dict[str, Any] = {
                "axis": MLX_TIMING_AXIS,
                "pair_id": pair_id,
                "planes": [],
                "score_claim": False,
                "promotion_eligible": False,
            }
            for plane_index, target in enumerate((y0[index], y1[index])):
                reference = realize_factor2_uint8_scorer_plane(operator, target)
                start = time.monotonic()
                candidate = realize_factor2_uint8_scorer_plane_mlx(
                    operator,
                    target,
                    mlx_module=mlx_module,
                    plan=plan,
                )
                seconds = _positive_elapsed(start)
                mismatch = np.flatnonzero(candidate.reshape(-1) != reference.reshape(-1))
                reference_count = _verify_one(operator, reference, target)
                try:
                    candidate_proof = verify_factor2_uint8_scorer_plane(operator, candidate, target)
                except Uint8LatticeError as exc:
                    raise TwoPlaneTimingReceiverError("MLX candidate numerator verification refused") from exc
                candidate_count = int(candidate_proof.numerator_equal_values)
                first = int(mismatch[0]) if mismatch.size else None
                failures = []
                if mismatch.size:
                    failures.append("byte_mismatch")
                if not candidate_proof.numerator_exact:
                    failures.append("numerator_mismatch")
                if not candidate_proof.certified_exact:
                    failures.append("canonical_preimage_mismatch")
                row["planes"].append(
                    {
                        "axis": MLX_TIMING_AXIS,
                        "plane_index": plane_index,
                        "seconds": seconds,
                        "byte_identical": not bool(mismatch.size),
                        "mismatched_values": int(mismatch.size),
                        "first_mismatch_flat_index": first,
                        "reference_numerator_equal_values": reference_count,
                        "mlx_numerator_equal_values": candidate_count,
                        "numerator_exact": bool(candidate_proof.numerator_exact),
                        "certified_exact": bool(candidate_proof.certified_exact),
                        "canonical_equal_values": int(candidate_proof.canonical_equal_values),
                        "failure_kinds": failures,
                        "reference_sha256": _sha256(reference.tobytes(order="C")),
                        "mlx_sha256": _sha256(candidate.tobytes(order="C")),
                        "score_claim": False,
                        "promotion_eligible": False,
                    }
                )
                aggregate.update(candidate.tobytes(order="C"))
            rows.append(row)
    finally:
        if real_device_pinned:
            mlx_module.set_default_device(previous_device)
    algorithmic_parity = all(
        plane["byte_identical"] and plane["numerator_exact"] and plane["certified_exact"]
        for row in rows
        for plane in row["planes"]
    )
    parity = algorithmic_parity and runtime["metal_usable"] is True
    divergences = [
        {"pair_id": row["pair_id"], **plane} for row in rows for plane in row["planes"] if plane["failure_kinds"]
    ]
    return {
        "schema": MLX_PARITY_SCHEMA,
        "axis": MLX_TIMING_AXIS,
        "pair_count": len(ids),
        "pair_ids": list(ids),
        "parity_passed": parity,
        "algorithmic_parity_passed": algorithmic_parity,
        "backend_kind": "explicit_test_double" if using_test_backend else "mlx.core/Metal",
        "rows": rows,
        "divergences": divergences,
        "output_sha256": aggregate.hexdigest(),
        "numerator_values_verified": sum(
            int(plane["mlx_numerator_equal_values"]) for row in rows for plane in row["planes"]
        ),
        "runtime_custody": runtime,
        "total_seconds": _positive_elapsed(total_start),
        "contest_timing_verdict_eligible": False,
        "score_claim": False,
        "promotion_eligible": False,
    }


__all__ = [
    "CHUNK_MANIFEST_SCHEMA",
    "CHUNK_PAIRS",
    "CPU_TIMING_AXIS",
    "FULL_CAMERA_HW",
    "FULL_NUMERATOR_VALUES",
    "FULL_PAIR_COUNT",
    "FULL_RAW_BYTES",
    "FULL_SCORER_HW",
    "FULL_Y0_SHA256",
    "FULL_Y1_SHA256",
    "MLX_PARITY_SCHEMA",
    "MLX_TIMING_AXIS",
    "PAIR_MANIFEST_SCHEMA",
    "PLANE0_MANIFEST_SCHEMA",
    "STATE_DIRECTORY",
    "TIMING_RECEIPT_SCHEMA",
    "MlxFactor2Plan",
    "TwoPlaneTimedInflateResult",
    "TwoPlaneTimingReceiverError",
    "build_mlx_factor2_plan",
    "mlx_runtime_status",
    "parity_check_mlx_two_plane",
    "realize_factor2_uint8_scorer_plane_mlx",
    "reserialize_parsed_packet",
    "timed_inflate_two_plane_archive",
]
