# SPDX-License-Identifier: MIT
"""Counted semantic-program base plus an exact two-plane quotient.

This is the first honest seam container between an original semantic program
and the production two-plane uint8 receiver.  The archive stores the semantic
program packet itself and LZMA-compressed bytewise-XOR quotient chunks.  It
does *not* store materialized base planes and it does not pretend to contain an
E1 renderer.  Decode therefore requires the exact generic renderer identity
declared by the manifest; that renderer must regenerate the base planes from
the counted program packet before the quotient can be applied.

The XOR coordinate is chosen only because it is reversible and exposes the
seam exactly.  It is not claimed to be entropy-optimal.  In particular, a
dense C1 quotient is a non-promotable scientific/seam baseline, never a
frontier candidate or score result.
"""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np

from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Uint8LatticeError,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)
from tac.witness_dsl.v10_production_receiver import (
    RECEIVER_CONTRACT_ID as V10_PRODUCTION_RECEIVER_CONTRACT_ID,
)

ARCHIVE_SCHEMA = "tac.c0b_semantic_quotient_archive.v1"
MANIFEST_SCHEMA = "tac.c0b_semantic_quotient_manifest.v1"
BUILD_RECEIPT_SCHEMA = "tac.c0b_semantic_quotient_build_receipt.v1"
CHUNK_STAGE_SCHEMA = "tac.c0b_semantic_quotient_chunk_stage.v1"
DOUBLE_DECODE_SCHEMA = "tac.c0b_semantic_quotient_double_decode.v1"
STORAGE_PREFLIGHT_SCHEMA = "tac.c0b_semantic_quotient_storage_preflight.v1"
TARGET_TEACHER_CUSTODY_SCHEMA = "tac.c0b_target_teacher_custody.v1"
RENDERER_CONTRACT_ID = "counted-semantic-packet-to-independent-u8-scorer-planes.v1"
SEMANTIC_BASE_TYPE_ID = "original-v15-e1-counted-semantic-program.v1"
QUOTIENT_CODEC_ID = "xor-u8-lzma1-raw-d8m-lc3-lp0-pb2.v1"
FACTOR2_CONTRACT_ID = V10_PRODUCTION_RECEIVER_CONTRACT_ID
SCIENTIFIC_LABEL = "NONPROMOTABLE_DENSE_C1_QUOTIENT_SCIENTIFIC_SEAM_BASELINE"
SEMANTIC_MEMBER = "semantic/base.packet"
MANIFEST_MEMBER = "manifest.json"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_REGULAR_MODE = 0o100644
MAX_MANIFEST_BYTES = 16 << 20
MAX_SEMANTIC_PACKET_BYTES = 1 << 30
MAX_QUOTIENT_MEMBER_BYTES = 1 << 30
MAX_PAIR_COUNT = 10_000
MAX_DIMENSION = 4096
MAX_CHANNELS = 16
PREFERRED_ARTIFACT_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)
LZMA_FILTERS: tuple[dict[str, int], ...] = (
    {
        "id": lzma.FILTER_LZMA1,
        "dict_size": 8 << 20,
        "lc": 3,
        "lp": 0,
        "pb": 2,
        "mode": lzma.MODE_NORMAL,
        "nice_len": 64,
        "mf": lzma.MF_BT4,
        "depth": 0,
    },
)


class SemanticQuotientError(ValueError):
    """Fail-closed malformed input, archive, renderer, or quotient error."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while chunk := handle.read(chunk_bytes):
                digest.update(chunk)
    except OSError as exc:
        raise SemanticQuotientError(f"cannot hash file: {path}") from exc
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SemanticQuotientError("value is not canonical-JSON encodable") from exc


def _read_canonical_json(payload: bytes, *, label: str) -> Any:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticQuotientError(f"{label} is not valid UTF-8 JSON") from exc
    if canonical_json(value) != payload:
        raise SemanticQuotientError(f"{label} is not canonical JSON")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SemanticQuotientError(f"{label} must be an exact integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise SemanticQuotientError(f"{label} is outside its admitted bounds")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not (
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
    ):
        raise SemanticQuotientError(f"{label} must be a lowercase SHA-256")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SemanticQuotientError(f"{label} must be a non-empty trimmed string")
    return value


def _immutable_uint8(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 4:
        raise SemanticQuotientError(f"{label} must have shape (pairs,H,W,C) and dtype uint8")
    if any(dimension <= 0 for dimension in array.shape):
        raise SemanticQuotientError(f"{label} cannot have an empty dimension")
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RendererIdentity:
    """Content identity of the generic semantic-packet renderer."""

    renderer_id: str
    renderer_source_sha256: str
    semantic_packet_schema: str
    expected_semantic_packet_sha256: str
    expected_camera_raw_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "renderer_id", _require_nonempty_string(self.renderer_id, "renderer_id"))
        object.__setattr__(
            self,
            "renderer_source_sha256",
            _require_sha256(self.renderer_source_sha256, "renderer_source_sha256"),
        )
        object.__setattr__(
            self,
            "semantic_packet_schema",
            _require_nonempty_string(self.semantic_packet_schema, "semantic_packet_schema"),
        )
        object.__setattr__(
            self,
            "expected_semantic_packet_sha256",
            _require_sha256(self.expected_semantic_packet_sha256, "expected_semantic_packet_sha256"),
        )
        if self.expected_camera_raw_sha256 is not None:
            object.__setattr__(
                self,
                "expected_camera_raw_sha256",
                _require_sha256(self.expected_camera_raw_sha256, "expected_camera_raw_sha256"),
            )

    def as_manifest(self) -> dict[str, Any]:
        return {
            "contract_id": RENDERER_CONTRACT_ID,
            "renderer_id": self.renderer_id,
            "renderer_source_sha256": self.renderer_source_sha256,
            "semantic_packet_schema": self.semantic_packet_schema,
            "expected_semantic_packet_sha256": self.expected_semantic_packet_sha256,
            "expected_camera_raw_sha256": self.expected_camera_raw_sha256,
            "renderer_required_at_decode": True,
            "renderer_embedded_in_archive": False,
        }


@dataclass(frozen=True)
class PlaneChunk:
    """One ordered pair range of independent scorer-space uint8 planes."""

    chunk_index: int
    pair_ids: tuple[int, ...]
    y0: np.ndarray
    y1: np.ndarray

    def __post_init__(self) -> None:
        chunk_index = _require_int(self.chunk_index, "chunk_index")
        if not isinstance(self.pair_ids, tuple) or not self.pair_ids:
            raise SemanticQuotientError("pair_ids must be a non-empty tuple")
        pair_ids = tuple(_require_int(value, "pair_id", maximum=MAX_PAIR_COUNT - 1) for value in self.pair_ids)
        if pair_ids != tuple(range(pair_ids[0], pair_ids[0] + len(pair_ids))):
            raise SemanticQuotientError("pair_ids must be one consecutive ordered range")
        y0 = _immutable_uint8(self.y0, "y0")
        y1 = _immutable_uint8(self.y1, "y1")
        if y0.shape != y1.shape or y0.shape[0] != len(pair_ids):
            raise SemanticQuotientError("y0/y1 geometry must match pair_ids exactly")
        if np.shares_memory(y0, y1):
            raise SemanticQuotientError("y0 and y1 must be independently owned")
        object.__setattr__(self, "chunk_index", chunk_index)
        object.__setattr__(self, "pair_ids", pair_ids)
        object.__setattr__(self, "y0", y0)
        object.__setattr__(self, "y1", y1)


class SemanticPlaneRenderer(Protocol):
    """External generic renderer required by this container at build/decode."""

    @property
    def identity(self) -> RendererIdentity: ...

    def render_chunks(
        self,
        semantic_packet: bytes,
        *,
        work_root: Path,
        chunk_pairs: int,
        resume: bool,
    ) -> Iterable[PlaneChunk]: ...


@dataclass(frozen=True)
class ParsedSemanticQuotientArchive:
    archive_path: Path
    archive_sha256: str
    archive_bytes: int
    manifest: Mapping[str, Any]
    semantic_packet: bytes
    archive_payload: bytes


@dataclass(frozen=True)
class DecodeReceipt:
    y0_sha256: str
    y1_sha256: str
    camera0_chunk_hashes_sha256: str | None
    camera1_chunk_hashes_sha256: str | None
    pair_count: int
    chunk_count: int
    factor2_verified_values: int

    def as_manifest(self) -> dict[str, Any]:
        return {
            "y0_sha256": self.y0_sha256,
            "y1_sha256": self.y1_sha256,
            "camera0_chunk_hashes_sha256": self.camera0_chunk_hashes_sha256,
            "camera1_chunk_hashes_sha256": self.camera1_chunk_hashes_sha256,
            "pair_count": self.pair_count,
            "chunk_count": self.chunk_count,
            "factor2_verified_values": self.factor2_verified_values,
        }


@dataclass(frozen=True)
class SemanticQuotientBuildResult:
    archive_path: Path
    receipt_path: Path
    archive_sha256: str
    archive_bytes: int
    manifest: Mapping[str, Any]
    double_decode: Mapping[str, Any]


def _compress_quotient(payload: bytes) -> bytes:
    try:
        return lzma.compress(payload, format=lzma.FORMAT_RAW, filters=list(LZMA_FILTERS))
    except lzma.LZMAError as exc:
        raise SemanticQuotientError("LZMA quotient compression failed") from exc


def _decompress_quotient(payload: bytes, *, expected_bytes: int) -> bytes:
    expected = _require_int(expected_bytes, "expected quotient bytes", maximum=MAX_QUOTIENT_MEMBER_BYTES)
    try:
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=list(LZMA_FILTERS))
        decoded = decoder.decompress(payload, max_length=expected + 1)
    except lzma.LZMAError as exc:
        raise SemanticQuotientError("LZMA quotient decompression failed") from exc
    if len(decoded) != expected or not decoder.eof or decoder.unused_data:
        raise SemanticQuotientError("decoded quotient length, terminator, or trailing bytes differ")
    return decoded


def _quotient_member(chunk_index: int, plane: int) -> str:
    if plane not in (0, 1):
        raise SemanticQuotientError("plane must be 0 or 1")
    return f"quotient/chunk-{chunk_index:04d}.y{plane}.xor.lzma"


def _checkpoint_member(work_root: Path, chunk_index: int, plane: int) -> Path:
    return work_root / "quotient_chunks" / _quotient_member(chunk_index, plane).split("/")[-1]


def _atomic_write(path: Path, payload: bytes, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise SemanticQuotientError(f"write-once path already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _read_regular_file_once(path: Path, *, label: str) -> bytes:
    """Read one final pathname through a no-follow descriptor exactly once."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SemanticQuotientError(f"cannot open {label} as a no-follow regular file: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SemanticQuotientError(f"{label} is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        if len(payload) != metadata.st_size:
            raise SemanticQuotientError(f"{label} changed length while it was read: {path}")
        return payload
    except OSError as exc:
        raise SemanticQuotientError(f"cannot read {label}: {path}") from exc
    finally:
        os.close(descriptor)


def write_once_or_equal(path: Path, payload: bytes) -> None:
    """Publish one immutable stage, or adopt byte-identical resume state."""

    try:
        _atomic_write(path, payload, replace=False)
        return
    except SemanticQuotientError as exc:
        if "write-once path already exists" not in str(exc):
            raise
    current = _read_regular_file_once(path, label="write-once path")
    if current != payload:
        raise SemanticQuotientError(f"preserved write-once bytes drifted: {path}")


def _is_within(path: Path, root: Path) -> bool:
    resolved = path.resolve(strict=False)
    parent = root.resolve(strict=False)
    return resolved == parent or parent in resolved.parents


def storage_preflight(
    work_root: Path | str,
    *,
    required_bytes: int,
    test_only_small_fixture: bool = False,
    allow_local_storage: bool = False,
) -> Mapping[str, Any]:
    """Apply the SSD-first storage contract without creating bulky artifacts."""

    required = _require_int(required_bytes, "required_bytes")
    target = Path(work_root).resolve(strict=False)
    selected_tier: str | None = None
    for index, root in enumerate(PREFERRED_ARTIFACT_ROOTS):
        if _is_within(target, root):
            selected_tier = ("vertigo", "apdatastore")[index]
            break
    if selected_tier is None:
        if not (test_only_small_fixture or allow_local_storage):
            raise SemanticQuotientError("full semantic quotient work must use a governed SSD tier")
        selected_tier = "test-local" if test_only_small_fixture else "explicit-local-opt-in"
    existing = target
    while not existing.exists():
        if existing.parent == existing:
            raise SemanticQuotientError("cannot resolve storage filesystem")
        existing = existing.parent
    free = int(shutil.disk_usage(existing).free)
    if not os.access(existing, os.W_OK):
        raise SemanticQuotientError(f"storage preflight refused: nearest existing parent is not writable: {existing}")
    if free < required:
        raise SemanticQuotientError(f"storage preflight refused: need {required} bytes, only {free} free")
    return {
        "schema": STORAGE_PREFLIGHT_SCHEMA,
        "selected_tier": selected_tier,
        "work_root": str(target),
        "required_bytes": required,
        "free_bytes_at_check": free,
        "passed": True,
        "test_only_small_fixture": bool(test_only_small_fixture),
        "allow_local_storage": bool(allow_local_storage),
    }


def _stable_storage_receipt(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Remove observation-time free-space fields from resumable identities."""

    return {
        "schema": value["schema"],
        "selected_tier": value["selected_tier"],
        "required_bytes": value["required_bytes"],
        "passed": value["passed"],
        "test_only_small_fixture": value["test_only_small_fixture"],
        "allow_local_storage": value["allow_local_storage"],
    }


def select_storage_root(
    workload_relative_path: str,
    *,
    required_bytes: int,
    allow_local_storage: bool = False,
    local_root: Path | str | None = None,
) -> tuple[Path, Mapping[str, Any]]:
    """Select the first writable tier with enough space in canonical order."""

    pure = PurePosixPath(_require_nonempty_string(workload_relative_path, "workload_relative_path"))
    if pure.is_absolute() or ".." in pure.parts:
        raise SemanticQuotientError("workload_relative_path must be safe and relative")
    candidates = list(PREFERRED_ARTIFACT_ROOTS)
    if allow_local_storage:
        candidates.append(Path(local_root or Path.cwd()))
    blockers: list[str] = []
    for root in candidates:
        if not root.is_dir():
            blockers.append(f"missing:{root}")
            continue
        target = root.joinpath(*pure.parts)
        try:
            row = storage_preflight(
                target,
                required_bytes=required_bytes,
                allow_local_storage=allow_local_storage,
            )
        except SemanticQuotientError as exc:
            blockers.append(f"{root}:{exc}")
            continue
        if not os.access(root, os.W_OK):
            blockers.append(f"not-writable:{root}")
            continue
        return target, row
    raise SemanticQuotientError("storage waterfall found no eligible tier: " + "; ".join(blockers))


def exact_resize_round_u8(operator: DisjointResizeOperator, frame: np.ndarray) -> np.ndarray:
    """Apply the exact integer resize and round nonnegative ties upward."""

    try:
        numerators, denominator = operator.apply_numerators(frame)
    except Uint8LatticeError as exc:
        raise SemanticQuotientError("exact resize refused the camera frame") from exc
    if denominator <= 0 or np.any(numerators < 0):
        raise SemanticQuotientError("exact resize escaped the nonnegative uint8 domain")
    rounded = (numerators.astype(np.int64) + denominator // 2) // denominator
    if np.any(rounded > 255):
        raise SemanticQuotientError("exact resize result exceeds uint8")
    return np.ascontiguousarray(rounded.astype(np.uint8))


def _validate_chunk_geometry(
    chunk: PlaneChunk,
    *,
    expected_chunk_index: int,
    expected_pair_start: int,
    scorer_hw: tuple[int, int],
    channels: int,
) -> None:
    if chunk.chunk_index != expected_chunk_index:
        raise SemanticQuotientError("renderer/teacher chunk index sequence drifted")
    if chunk.pair_ids[0] != expected_pair_start:
        raise SemanticQuotientError("renderer/teacher pair sequence drifted")
    expected_tail = (*scorer_hw, channels)
    if chunk.y0.shape[1:] != expected_tail or chunk.y1.shape[1:] != expected_tail:
        raise SemanticQuotientError("renderer/teacher scorer geometry drifted")


def _factor2_chunk_receipt(
    chunk: PlaneChunk,
    *,
    operator: DisjointResizeOperator,
) -> Mapping[str, Any]:
    camera0 = hashlib.sha256()
    camera1 = hashlib.sha256()
    verified_values = 0
    for y0, y1 in zip(chunk.y0, chunk.y1, strict=True):
        try:
            frame0 = realize_factor2_uint8_scorer_plane(operator, y0)
            frame1 = realize_factor2_uint8_scorer_plane(operator, y1)
            proof0 = verify_factor2_uint8_scorer_plane(operator, frame0, y0)
            proof1 = verify_factor2_uint8_scorer_plane(operator, frame1, y1)
        except Uint8LatticeError as exc:
            raise SemanticQuotientError("production factor-2 realization refused target planes") from exc
        if not proof0.certified_exact or not proof1.certified_exact:
            raise SemanticQuotientError("production factor-2 realization was not certified exact")
        camera0.update(frame0.tobytes(order="C"))
        camera1.update(frame1.tobytes(order="C"))
        verified_values += proof0.numerator_equal_values + proof1.numerator_equal_values
    return {
        "contract_id": FACTOR2_CONTRACT_ID,
        "camera0_sha256": camera0.hexdigest(),
        "camera1_sha256": camera1.hexdigest(),
        "verified_values": verified_values,
        "certified_exact": True,
    }


def _renderer_chunks(
    renderer: SemanticPlaneRenderer,
    semantic_packet: bytes,
    *,
    work_root: Path,
    chunk_pairs: int,
    resume: bool,
) -> Iterator[PlaneChunk]:
    if not hasattr(renderer, "identity") or not hasattr(renderer, "render_chunks"):
        raise SemanticQuotientError("renderer does not implement the semantic-plane protocol")
    try:
        rows = renderer.render_chunks(
            semantic_packet,
            work_root=work_root,
            chunk_pairs=chunk_pairs,
            resume=resume,
        )
        for row in rows:
            if not isinstance(row, PlaneChunk):
                raise SemanticQuotientError("renderer emitted a non-PlaneChunk value")
            yield row
    except SemanticQuotientError:
        raise
    except Exception as exc:
        raise SemanticQuotientError("semantic base renderer failed") from exc


def _validate_renderer_identity(renderer: SemanticPlaneRenderer, declared: Mapping[str, Any]) -> RendererIdentity:
    identity = renderer.identity
    if not isinstance(identity, RendererIdentity):
        raise SemanticQuotientError("renderer identity is not RendererIdentity")
    if identity.as_manifest() != declared:
        raise SemanticQuotientError("decode renderer identity differs from the counted manifest")
    return identity


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = ZIP_REGULAR_MODE << 16
    return info


def _archive_member_order(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    rows = manifest.get("chunks")
    if not isinstance(rows, list):
        raise SemanticQuotientError("manifest chunks must be a list")
    names = [MANIFEST_MEMBER, SEMANTIC_MEMBER]
    for row in rows:
        if not isinstance(row, dict):
            raise SemanticQuotientError("manifest chunk row must be an object")
        quotient = row.get("quotient")
        if not isinstance(quotient, dict):
            raise SemanticQuotientError("manifest chunk quotient must be an object")
        for key in ("y0", "y1"):
            leg = quotient.get(key)
            if not isinstance(leg, dict):
                raise SemanticQuotientError("manifest quotient leg must be an object")
            name = leg.get("member")
            if not isinstance(name, str) or name != _quotient_member(row.get("chunk_index"), int(key[-1])):
                raise SemanticQuotientError("manifest quotient member name drifted")
            names.append(name)
    if len(names) != len(set(names)):
        raise SemanticQuotientError("archive member names are not unique")
    return tuple(names)


def _validate_target_teacher_custody(
    value: Any,
    *,
    pair_count: int,
    chunk_pairs: int,
    scorer_hw: tuple[int, int],
    channels: int,
    observed_chunks: list[dict[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Close teacher provenance over the exact target bytes consumed by this build."""

    required = {
        "schema",
        "teacher_id",
        "pair_count",
        "chunk_count",
        "chunk_pairs",
        "scorer_hw",
        "channels",
        "y0_sha256",
        "y1_sha256",
        "consumed_chunk_target_hashes",
        "consumed_chunk_target_hashes_sha256",
        "provenance",
        "all_video_derived_metadata_counted",
        "score_claim",
        "promotion_eligible",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise SemanticQuotientError("target teacher custody fields differ from the closed schema")
    if value["schema"] != TARGET_TEACHER_CUSTODY_SCHEMA:
        raise SemanticQuotientError("target teacher custody schema differs")
    _require_nonempty_string(value["teacher_id"], "target teacher id")
    expected_chunk_count = (pair_count + chunk_pairs - 1) // chunk_pairs
    if (
        value["pair_count"] != pair_count
        or value["chunk_count"] != expected_chunk_count
        or value["chunk_pairs"] != chunk_pairs
        or value["scorer_hw"] != list(scorer_hw)
        or value["channels"] != channels
    ):
        raise SemanticQuotientError("target teacher custody geometry differs from the build")
    _require_sha256(value["y0_sha256"], "target teacher aggregate y0 sha256")
    _require_sha256(value["y1_sha256"], "target teacher aggregate y1 sha256")
    if (
        value["all_video_derived_metadata_counted"] is not True
        or value["score_claim"] is not False
        or value["promotion_eligible"] is not False
    ):
        raise SemanticQuotientError("target teacher custody authority flags differ")
    if not isinstance(value["provenance"], dict):
        raise SemanticQuotientError("target teacher provenance must be an object")
    canonical_json(value["provenance"])
    declared = value["consumed_chunk_target_hashes"]
    if not isinstance(declared, list) or len(declared) != expected_chunk_count:
        raise SemanticQuotientError("target teacher consumed chunk hash list differs")
    expected_pair = 0
    normalized: list[dict[str, Any]] = []
    for chunk_index, row in enumerate(declared):
        if not isinstance(row, dict) or set(row) != {"chunk_index", "pair_ids", "y0_sha256", "y1_sha256"}:
            raise SemanticQuotientError("target teacher chunk hash row fields differ")
        expected_pairs = min(chunk_pairs, pair_count - expected_pair)
        pair_ids = list(range(expected_pair, expected_pair + expected_pairs))
        if row["chunk_index"] != chunk_index or row["pair_ids"] != pair_ids:
            raise SemanticQuotientError("target teacher chunk hash geometry differs")
        _require_sha256(row["y0_sha256"], "target teacher chunk y0 sha256")
        _require_sha256(row["y1_sha256"], "target teacher chunk y1 sha256")
        normalized.append(dict(row))
        expected_pair += expected_pairs
    if expected_pair != pair_count:
        raise SemanticQuotientError("target teacher chunk hash pair coverage differs")
    declared_digest = _require_sha256(
        value["consumed_chunk_target_hashes_sha256"],
        "target teacher consumed chunk hash list sha256",
    )
    if declared_digest != _sha256(canonical_json(normalized)):
        raise SemanticQuotientError("target teacher consumed chunk hash list digest differs")
    if observed_chunks is not None:
        observed = [
            {
                "chunk_index": row["chunk_index"],
                "pair_ids": row["pair_ids"],
                "y0_sha256": row["target"]["y0_sha256"],
                "y1_sha256": row["target"]["y1_sha256"],
            }
            for row in observed_chunks
        ]
        if observed != normalized:
            raise SemanticQuotientError("observed target chunks differ from teacher custody")
    return value


def _validate_manifest(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SemanticQuotientError("manifest must be an object")
    required = {
        "schema",
        "scientific_label",
        "pair_count",
        "chunk_count",
        "chunk_pairs",
        "geometry",
        "semantic_base",
        "target_teacher_custody",
        "quotient_codec",
        "chunks",
        "decoded_targets",
        "rate_accounting",
        "limitations",
        "research_only",
        "launch_ready",
        "score_claim",
        "promotion_eligible",
        "pointer_moved",
    }
    if set(value) != required:
        raise SemanticQuotientError("manifest fields differ from the closed schema")
    if value["schema"] != MANIFEST_SCHEMA or value["scientific_label"] != SCIENTIFIC_LABEL:
        raise SemanticQuotientError("manifest schema/scientific label differs")
    if any(value[key] is not False for key in ("launch_ready", "score_claim", "promotion_eligible", "pointer_moved")):
        raise SemanticQuotientError("semantic quotient archive cannot carry authority claims")
    if value["research_only"] is not True:
        raise SemanticQuotientError("semantic quotient archive must remain research-only")
    pair_count = _require_int(value["pair_count"], "pair_count", minimum=1, maximum=MAX_PAIR_COUNT)
    chunk_count = _require_int(value["chunk_count"], "chunk_count", minimum=1, maximum=MAX_PAIR_COUNT)
    chunk_pairs = _require_int(value["chunk_pairs"], "chunk_pairs", minimum=1, maximum=MAX_PAIR_COUNT)
    if chunk_count != (pair_count + chunk_pairs - 1) // chunk_pairs:
        raise SemanticQuotientError("manifest chunk count arithmetic differs")
    geometry = value["geometry"]
    if not isinstance(geometry, dict) or set(geometry) != {"camera_hw", "scorer_hw", "channels"}:
        raise SemanticQuotientError("manifest geometry differs")
    for key in ("camera_hw", "scorer_hw"):
        dims = geometry[key]
        if not isinstance(dims, list) or len(dims) != 2:
            raise SemanticQuotientError(f"manifest {key} must contain two dimensions")
        for dimension in dims:
            _require_int(dimension, f"{key} dimension", minimum=1, maximum=MAX_DIMENSION)
    channels = _require_int(geometry["channels"], "channels", minimum=1, maximum=MAX_CHANNELS)
    scorer_hw = tuple(geometry["scorer_hw"])
    target_plane_bytes = pair_count * scorer_hw[0] * scorer_hw[1] * channels
    semantic = value["semantic_base"]
    if not isinstance(semantic, dict):
        raise SemanticQuotientError("semantic_base must be an object")
    semantic_required = {
        "base_type",
        "member",
        "bytes",
        "sha256",
        "renderer",
        "derived_y0_sha256",
        "derived_y1_sha256",
        "video_derived",
    }
    if (
        set(semantic) != semantic_required
        or semantic["base_type"] != SEMANTIC_BASE_TYPE_ID
        or semantic["member"] != SEMANTIC_MEMBER
        or semantic["video_derived"] is not True
    ):
        raise SemanticQuotientError("semantic_base fields differ")
    _require_int(semantic["bytes"], "semantic packet bytes", minimum=1, maximum=MAX_SEMANTIC_PACKET_BYTES)
    _require_sha256(semantic["sha256"], "semantic packet sha256")
    _require_sha256(semantic["derived_y0_sha256"], "base y0 sha256")
    _require_sha256(semantic["derived_y1_sha256"], "base y1 sha256")
    renderer = semantic["renderer"]
    if not isinstance(renderer, dict) or renderer.get("contract_id") != RENDERER_CONTRACT_ID:
        raise SemanticQuotientError("semantic renderer contract differs")
    if (
        renderer.get("renderer_required_at_decode") is not True
        or renderer.get("renderer_embedded_in_archive") is not False
    ):
        raise SemanticQuotientError("semantic renderer limitation flags differ")
    for key in ("renderer_source_sha256", "expected_semantic_packet_sha256"):
        _require_sha256(renderer.get(key), f"renderer {key}")
    _require_nonempty_string(renderer.get("renderer_id"), "renderer_id")
    _require_nonempty_string(renderer.get("semantic_packet_schema"), "semantic_packet_schema")
    if renderer.get("expected_camera_raw_sha256") is not None:
        _require_sha256(renderer["expected_camera_raw_sha256"], "expected camera raw sha256")
    if renderer["expected_semantic_packet_sha256"] != semantic["sha256"]:
        raise SemanticQuotientError("renderer and semantic packet identities differ")
    quotient_codec = value["quotient_codec"]
    if not isinstance(quotient_codec, dict) or quotient_codec != {
        "codec_id": QUOTIENT_CODEC_ID,
        "coordinate": "bytewise_xor_against_semantic_base_plane",
        "exact_reversible": True,
        "entropy_optimal_claim": False,
    }:
        raise SemanticQuotientError("quotient codec declaration differs")
    rows = value["chunks"]
    if not isinstance(rows, list) or len(rows) != chunk_count:
        raise SemanticQuotientError("manifest chunk count differs")
    expected_pair = 0
    raw_totals = {"y0": 0, "y1": 0}
    coded_total = 0
    for chunk_index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("chunk_index") != chunk_index:
            raise SemanticQuotientError("manifest chunk sequence differs")
        pair_ids = row.get("pair_ids")
        if not isinstance(pair_ids, list) or not pair_ids:
            raise SemanticQuotientError("manifest pair_ids must be nonempty")
        if pair_ids != list(range(expected_pair, expected_pair + len(pair_ids))):
            raise SemanticQuotientError("manifest pair sequence differs")
        expected_chunk_pairs = min(chunk_pairs, pair_count - expected_pair)
        if len(pair_ids) != expected_chunk_pairs:
            raise SemanticQuotientError("manifest chunk size arithmetic differs")
        expected_pair += len(pair_ids)
        expected_chunk_plane_bytes = len(pair_ids) * scorer_hw[0] * scorer_hw[1] * channels
        for group in ("base", "target"):
            legs = row.get(group)
            if not isinstance(legs, dict) or set(legs) != {"y0_sha256", "y1_sha256"}:
                raise SemanticQuotientError(f"manifest {group} hashes differ")
            _require_sha256(legs["y0_sha256"], f"{group} y0 sha256")
            _require_sha256(legs["y1_sha256"], f"{group} y1 sha256")
        quotient = row.get("quotient")
        if not isinstance(quotient, dict) or set(quotient) != {"y0", "y1"}:
            raise SemanticQuotientError("manifest quotient legs differ")
        for plane, key in enumerate(("y0", "y1")):
            leg = quotient[key]
            if not isinstance(leg, dict) or set(leg) != {
                "member",
                "raw_bytes",
                "raw_sha256",
                "coded_bytes",
                "coded_sha256",
                "video_derived",
            }:
                raise SemanticQuotientError("manifest quotient leg fields differ")
            if leg["member"] != _quotient_member(chunk_index, plane) or leg["video_derived"] is not True:
                raise SemanticQuotientError("manifest quotient ownership differs")
            raw_bytes = _require_int(leg["raw_bytes"], "quotient raw bytes", minimum=1)
            if raw_bytes != expected_chunk_plane_bytes:
                raise SemanticQuotientError("manifest quotient raw chunk bytes differ")
            raw_totals[key] += raw_bytes
            coded_total += _require_int(leg["coded_bytes"], "quotient coded bytes", minimum=1)
            _require_sha256(leg["raw_sha256"], "quotient raw sha256")
            _require_sha256(leg["coded_sha256"], "quotient coded sha256")
        factor2 = row.get("factor2")
        if not isinstance(factor2, dict) or factor2.get("contract_id") != FACTOR2_CONTRACT_ID:
            raise SemanticQuotientError("manifest factor2 receipt differs")
        if factor2.get("certified_exact") is not True:
            raise SemanticQuotientError("manifest factor2 receipt is not exact")
        _require_sha256(factor2.get("camera0_sha256"), "factor2 camera0 sha256")
        _require_sha256(factor2.get("camera1_sha256"), "factor2 camera1 sha256")
        verified_values = _require_int(factor2.get("verified_values"), "factor2 verified values", minimum=1)
        if verified_values != 2 * expected_chunk_plane_bytes:
            raise SemanticQuotientError("manifest factor2 verified-value arithmetic differs")
    if expected_pair != pair_count or raw_totals != {"y0": target_plane_bytes, "y1": target_plane_bytes}:
        raise SemanticQuotientError("manifest pair coverage differs")
    decoded = value["decoded_targets"]
    if not isinstance(decoded, dict) or set(decoded) != {"y0_bytes", "y1_bytes", "y0_sha256", "y1_sha256"}:
        raise SemanticQuotientError("decoded target aggregate differs")
    if (
        _require_int(decoded["y0_bytes"], "decoded y0 bytes", minimum=1) != target_plane_bytes
        or _require_int(decoded["y1_bytes"], "decoded y1 bytes", minimum=1) != target_plane_bytes
    ):
        raise SemanticQuotientError("decoded target byte arithmetic differs")
    _require_sha256(decoded["y0_sha256"], "decoded y0 sha256")
    _require_sha256(decoded["y1_sha256"], "decoded y1 sha256")
    rate = value["rate_accounting"]
    if not isinstance(rate, dict) or set(rate) != {
        "semantic_packet_bytes",
        "quotient_raw_bytes",
        "quotient_coded_bytes",
        "all_video_derived_metadata_counted",
    }:
        raise SemanticQuotientError("rate accounting fields differ")
    if (
        rate["semantic_packet_bytes"] != semantic["bytes"]
        or rate["quotient_raw_bytes"] != 2 * target_plane_bytes
        or rate["quotient_coded_bytes"] != coded_total
        or rate["all_video_derived_metadata_counted"] is not True
    ):
        raise SemanticQuotientError("rate accounting totals differ")
    limitations = value["limitations"]
    if (
        not isinstance(limitations, list)
        or len(limitations) != 4
        or set(limitations)
        != {
            "dense C1 quotient is a scientific seam baseline, not a frontier candidate",
            "XOR is exact and reversible but is not presumed entropy-optimal",
            "the counted semantic packet requires the separately hash-bound generic renderer at decode",
            "the container imports no scorer and produces no score authority",
        }
    ):
        raise SemanticQuotientError("manifest limitations differ")
    custody = _validate_target_teacher_custody(
        value["target_teacher_custody"],
        pair_count=pair_count,
        chunk_pairs=chunk_pairs,
        scorer_hw=scorer_hw,
        channels=channels,
        observed_chunks=rows,
    )
    if custody["y0_sha256"] != decoded["y0_sha256"] or custody["y1_sha256"] != decoded["y1_sha256"]:
        raise SemanticQuotientError("target teacher aggregate hashes differ from decoded targets")
    _archive_member_order(value)
    return value


def _build_manifest(
    *,
    semantic_packet: bytes,
    renderer_identity: RendererIdentity,
    pair_count: int,
    chunk_pairs: int,
    camera_hw: tuple[int, int],
    scorer_hw: tuple[int, int],
    channels: int,
    chunk_rows: list[dict[str, Any]],
    target_teacher_custody: Mapping[str, Any],
    base_y0_sha256: str,
    base_y1_sha256: str,
    target_y0_sha256: str,
    target_y1_sha256: str,
    target_plane_bytes: int,
) -> Mapping[str, Any]:
    raw_total = sum(row["quotient"][key]["raw_bytes"] for row in chunk_rows for key in ("y0", "y1"))
    coded_total = sum(row["quotient"][key]["coded_bytes"] for row in chunk_rows for key in ("y0", "y1"))
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "scientific_label": SCIENTIFIC_LABEL,
        "pair_count": pair_count,
        "chunk_count": len(chunk_rows),
        "chunk_pairs": chunk_pairs,
        "geometry": {"camera_hw": list(camera_hw), "scorer_hw": list(scorer_hw), "channels": channels},
        "semantic_base": {
            "base_type": SEMANTIC_BASE_TYPE_ID,
            "member": SEMANTIC_MEMBER,
            "bytes": len(semantic_packet),
            "sha256": _sha256(semantic_packet),
            "renderer": renderer_identity.as_manifest(),
            "derived_y0_sha256": base_y0_sha256,
            "derived_y1_sha256": base_y1_sha256,
            "video_derived": True,
        },
        "target_teacher_custody": dict(target_teacher_custody),
        "quotient_codec": {
            "codec_id": QUOTIENT_CODEC_ID,
            "coordinate": "bytewise_xor_against_semantic_base_plane",
            "exact_reversible": True,
            "entropy_optimal_claim": False,
        },
        "chunks": chunk_rows,
        "decoded_targets": {
            "y0_bytes": target_plane_bytes,
            "y1_bytes": target_plane_bytes,
            "y0_sha256": target_y0_sha256,
            "y1_sha256": target_y1_sha256,
        },
        "rate_accounting": {
            "semantic_packet_bytes": len(semantic_packet),
            "quotient_raw_bytes": raw_total,
            "quotient_coded_bytes": coded_total,
            "all_video_derived_metadata_counted": True,
        },
        "limitations": [
            "dense C1 quotient is a scientific seam baseline, not a frontier candidate",
            "XOR is exact and reversible but is not presumed entropy-optimal",
            "the counted semantic packet requires the separately hash-bound generic renderer at decode",
            "the container imports no scorer and produces no score authority",
        ],
        "research_only": True,
        "launch_ready": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    return _validate_manifest(manifest)


def _build_archive_file(
    *,
    archive_path: Path,
    manifest: Mapping[str, Any],
    semantic_packet: bytes,
    work_root: Path,
) -> bytes:
    payloads: dict[str, bytes] = {}
    for row in manifest["chunks"]:
        chunk_index = row["chunk_index"]
        for plane in (0, 1):
            member = _quotient_member(chunk_index, plane)
            payloads[member] = _read_regular_file_once(
                _checkpoint_member(work_root, chunk_index, plane),
                label="quotient checkpoint",
            )
    canonical = _canonical_archive_bytes(manifest, semantic_packet, payloads)
    write_once_or_equal(archive_path, canonical)
    return canonical


def _canonical_archive_bytes(
    manifest: Mapping[str, Any],
    semantic_packet: bytes,
    quotient_payloads: Mapping[str, bytes],
) -> bytes:
    """Construct the sole admitted ZIP byte encoding for one manifest state."""

    expected_quotients = set(_archive_member_order(manifest)[2:])
    if set(quotient_payloads) != expected_quotients:
        raise SemanticQuotientError("canonical archive quotient member payload set differs")
    output = io.BytesIO()
    try:
        with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
            archive.comment = b""
            archive.writestr(_zip_info(MANIFEST_MEMBER), canonical_json(manifest))
            archive.writestr(_zip_info(SEMANTIC_MEMBER), semantic_packet)
            for member in _archive_member_order(manifest)[2:]:
                archive.writestr(_zip_info(member), quotient_payloads[member])
    except (OSError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        raise SemanticQuotientError("canonical archive construction failed") from exc
    return output.getvalue()


def parse_semantic_quotient_archive(path: Path | str) -> ParsedSemanticQuotientArchive:
    """Strictly parse and hash every counted member without rendering a base."""

    archive_path = Path(path)
    archive_payload = _read_regular_file_once(archive_path, label="semantic quotient archive")
    archive_bytes = len(archive_payload)
    archive_sha = _sha256(archive_payload)
    try:
        with zipfile.ZipFile(io.BytesIO(archive_payload), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if not infos or names[0] != MANIFEST_MEMBER:
                raise SemanticQuotientError("manifest.json must be the first archive member")
            if len(names) != len(set(names)):
                raise SemanticQuotientError("archive contains duplicate member names")
            if archive.comment != b"" or infos[0].file_size > MAX_MANIFEST_BYTES:
                raise SemanticQuotientError("archive comment or manifest size is noncanonical")
            for info in infos:
                if (
                    info.is_dir()
                    or PurePosixPath(info.filename).is_absolute()
                    or ".." in PurePosixPath(info.filename).parts
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.compress_size != info.file_size
                    or info.flag_bits & 0x1
                    or info.date_time != ZIP_TIMESTAMP
                    or info.create_system != 3
                    or info.external_attr != ZIP_REGULAR_MODE << 16
                    or info.extra != b""
                    or info.comment != b""
                ):
                    raise SemanticQuotientError("archive member framing is unsafe or noncanonical")
            manifest_payload = archive.read(MANIFEST_MEMBER)
            if len(manifest_payload) > MAX_MANIFEST_BYTES:
                raise SemanticQuotientError("manifest exceeds its byte cap")
            manifest = _validate_manifest(_read_canonical_json(manifest_payload, label=MANIFEST_MEMBER))
            expected_names = list(_archive_member_order(manifest))
            if names != expected_names:
                raise SemanticQuotientError("archive member order/set differs from the manifest")
            info_by_name = {info.filename: info for info in infos}
            semantic_info = info_by_name[SEMANTIC_MEMBER]
            if semantic_info.file_size != manifest["semantic_base"]["bytes"]:
                raise SemanticQuotientError("semantic packet declared ZIP size differs")
            semantic = archive.read(SEMANTIC_MEMBER)
            semantic_row = manifest["semantic_base"]
            if len(semantic) != semantic_row["bytes"] or _sha256(semantic) != semantic_row["sha256"]:
                raise SemanticQuotientError("semantic packet length/hash differs")
            quotient_payloads: dict[str, bytes] = {}
            for row in manifest["chunks"]:
                for plane, key in enumerate(("y0", "y1")):
                    leg = row["quotient"][key]
                    member_info = info_by_name[_quotient_member(row["chunk_index"], plane)]
                    if member_info.file_size != leg["coded_bytes"] or member_info.file_size > MAX_QUOTIENT_MEMBER_BYTES:
                        raise SemanticQuotientError("quotient member declared ZIP size differs")
                    payload = archive.read(_quotient_member(row["chunk_index"], plane))
                    if len(payload) != leg["coded_bytes"] or _sha256(payload) != leg["coded_sha256"]:
                        raise SemanticQuotientError("quotient member length/hash differs")
                    quotient_payloads[leg["member"]] = payload
            if _canonical_archive_bytes(manifest, semantic, quotient_payloads) != archive_payload:
                raise SemanticQuotientError("archive bytes differ from the canonical ZIP encoding")
    except SemanticQuotientError:
        raise
    except (OSError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        raise SemanticQuotientError("semantic quotient archive parse failed") from exc
    return ParsedSemanticQuotientArchive(
        archive_path,
        archive_sha,
        archive_bytes,
        manifest,
        semantic,
        archive_payload,
    )


def _decode_once(
    parsed: ParsedSemanticQuotientArchive,
    renderer: SemanticPlaneRenderer,
    *,
    work_root: Path,
    resume: bool,
    verify_factor2: bool,
) -> DecodeReceipt:
    manifest = parsed.manifest
    identity = _validate_renderer_identity(renderer, manifest["semantic_base"]["renderer"])
    if _sha256(parsed.semantic_packet) != identity.expected_semantic_packet_sha256:
        raise SemanticQuotientError("semantic packet differs from renderer identity")
    scorer_hw = tuple(manifest["geometry"]["scorer_hw"])
    camera_hw = tuple(manifest["geometry"]["camera_hw"])
    channels = manifest["geometry"]["channels"]
    operator = None
    if verify_factor2:
        try:
            operator = DisjointResizeOperator.build(
                camera_h=camera_hw[0],
                camera_w=camera_hw[1],
                scorer_h=scorer_hw[0],
                scorer_w=scorer_hw[1],
            )
        except Uint8LatticeError as exc:
            raise SemanticQuotientError("production factor-2 geometry refused") from exc
    base_y0_digest = hashlib.sha256()
    base_y1_digest = hashlib.sha256()
    target_y0_digest = hashlib.sha256()
    target_y1_digest = hashlib.sha256()
    camera0_digest = hashlib.sha256()
    camera1_digest = hashlib.sha256()
    verified_values = 0
    expected_pair = 0
    expected_chunk = 0
    with zipfile.ZipFile(io.BytesIO(parsed.archive_payload), "r") as archive:
        rows = manifest["chunks"]
        renderer_rows = _renderer_chunks(
            renderer,
            parsed.semantic_packet,
            work_root=work_root,
            chunk_pairs=manifest["chunk_pairs"],
            resume=resume,
        )
        for base_chunk, row in zip(renderer_rows, rows, strict=True):
            _validate_chunk_geometry(
                base_chunk,
                expected_chunk_index=expected_chunk,
                expected_pair_start=expected_pair,
                scorer_hw=scorer_hw,
                channels=channels,
            )
            if list(base_chunk.pair_ids) != row["pair_ids"]:
                raise SemanticQuotientError("renderer pair IDs differ from manifest")
            if _sha256(base_chunk.y0.tobytes(order="C")) != row["base"]["y0_sha256"]:
                raise SemanticQuotientError("renderer base Y0 differs from build custody")
            if _sha256(base_chunk.y1.tobytes(order="C")) != row["base"]["y1_sha256"]:
                raise SemanticQuotientError("renderer base Y1 differs from build custody")
            decoded_legs: list[np.ndarray] = []
            for key, base in (("y0", base_chunk.y0), ("y1", base_chunk.y1)):
                leg = row["quotient"][key]
                coded = archive.read(leg["member"])
                residual = _decompress_quotient(coded, expected_bytes=leg["raw_bytes"])
                if _sha256(residual) != leg["raw_sha256"] or len(residual) != base.nbytes:
                    raise SemanticQuotientError("decoded quotient bytes/hash differ")
                residual_array = np.frombuffer(residual, dtype=np.uint8).reshape(base.shape)
                target = np.bitwise_xor(base, residual_array)
                if _sha256(target.tobytes(order="C")) != row["target"][f"{key}_sha256"]:
                    raise SemanticQuotientError("recovered target plane differs from build custody")
                decoded_legs.append(np.ascontiguousarray(target))
            y0, y1 = decoded_legs
            base_y0_digest.update(base_chunk.y0.tobytes(order="C"))
            base_y1_digest.update(base_chunk.y1.tobytes(order="C"))
            target_y0_digest.update(y0.tobytes(order="C"))
            target_y1_digest.update(y1.tobytes(order="C"))
            if operator is not None:
                factor2 = _factor2_chunk_receipt(
                    PlaneChunk(base_chunk.chunk_index, base_chunk.pair_ids, y0, y1),
                    operator=operator,
                )
                if factor2 != row["factor2"]:
                    raise SemanticQuotientError("decode factor-2 receipt differs from build custody")
                camera0_digest.update(bytes.fromhex(factor2["camera0_sha256"]))
                camera1_digest.update(bytes.fromhex(factor2["camera1_sha256"]))
                verified_values += factor2["verified_values"]
            expected_pair += len(base_chunk.pair_ids)
            expected_chunk += 1
    if expected_pair != manifest["pair_count"] or expected_chunk != manifest["chunk_count"]:
        raise SemanticQuotientError("renderer did not cover the complete manifest")
    if base_y0_digest.hexdigest() != manifest["semantic_base"]["derived_y0_sha256"]:
        raise SemanticQuotientError("aggregate semantic base Y0 digest differs")
    if base_y1_digest.hexdigest() != manifest["semantic_base"]["derived_y1_sha256"]:
        raise SemanticQuotientError("aggregate semantic base Y1 digest differs")
    if target_y0_digest.hexdigest() != manifest["decoded_targets"]["y0_sha256"]:
        raise SemanticQuotientError("aggregate decoded target Y0 digest differs")
    if target_y1_digest.hexdigest() != manifest["decoded_targets"]["y1_sha256"]:
        raise SemanticQuotientError("aggregate decoded target Y1 digest differs")
    return DecodeReceipt(
        y0_sha256=target_y0_digest.hexdigest(),
        y1_sha256=target_y1_digest.hexdigest(),
        camera0_chunk_hashes_sha256=camera0_digest.hexdigest() if verify_factor2 else None,
        camera1_chunk_hashes_sha256=camera1_digest.hexdigest() if verify_factor2 else None,
        pair_count=expected_pair,
        chunk_count=expected_chunk,
        factor2_verified_values=verified_values,
    )


def double_decode_archive(
    archive_path: Path | str,
    renderer: SemanticPlaneRenderer,
    *,
    work_root: Path | str,
    resume: bool = True,
    verify_factor2: bool = True,
) -> Mapping[str, Any]:
    """Decode twice through the renderer and require identical exact receipts."""

    parsed = parse_semantic_quotient_archive(archive_path)
    root = Path(work_root)
    first = _decode_once(
        parsed, renderer, work_root=root / "decode-pass-1", resume=resume, verify_factor2=verify_factor2
    )
    second = _decode_once(
        parsed,
        renderer,
        work_root=root / "decode-pass-2",
        resume=resume,
        verify_factor2=verify_factor2,
    )
    if first != second:
        raise SemanticQuotientError("double decode receipts differ")
    receipt = {
        "schema": DOUBLE_DECODE_SCHEMA,
        "archive_sha256": parsed.archive_sha256,
        "first": first.as_manifest(),
        "second": second.as_manifest(),
        "byte_identical": True,
        "factor2_verified": bool(verify_factor2),
        "score_claim": False,
        "promotion_eligible": False,
    }
    write_once_or_equal(root / "stage_checkpoints" / "03_double_decode.json", canonical_json(receipt))
    return receipt


def build_semantic_quotient_archive(
    semantic_packet: bytes,
    renderer: SemanticPlaneRenderer,
    target_chunks: Iterable[PlaneChunk],
    *,
    archive_path: Path | str,
    work_root: Path | str,
    target_teacher_custody: Mapping[str, Any],
    camera_hw: tuple[int, int],
    scorer_hw: tuple[int, int],
    channels: int,
    pair_count: int,
    chunk_pairs: int,
    resume: bool,
    test_only_small_fixture: bool = False,
    allow_local_storage: bool = False,
) -> SemanticQuotientBuildResult:
    """Build, parse back, and double-decode the exact quotient archive."""

    if not isinstance(semantic_packet, bytes) or not semantic_packet:
        raise SemanticQuotientError("semantic_packet must be non-empty bytes")
    if len(semantic_packet) > MAX_SEMANTIC_PACKET_BYTES:
        raise SemanticQuotientError("semantic packet exceeds its byte cap")
    identity = renderer.identity
    if not isinstance(identity, RendererIdentity):
        raise SemanticQuotientError("renderer identity is not RendererIdentity")
    if _sha256(semantic_packet) != identity.expected_semantic_packet_sha256:
        raise SemanticQuotientError("semantic packet differs from renderer identity")
    if not test_only_small_fixture and identity.expected_camera_raw_sha256 is None:
        raise SemanticQuotientError("full build requires a provenance-bound semantic renderer output")
    count = _require_int(pair_count, "pair_count", minimum=1, maximum=MAX_PAIR_COUNT)
    chunk_size = _require_int(chunk_pairs, "chunk_pairs", minimum=1, maximum=count)
    channel_count = _require_int(channels, "channels", minimum=1, maximum=MAX_CHANNELS)
    for label, dims in (("camera_hw", camera_hw), ("scorer_hw", scorer_hw)):
        if not isinstance(dims, tuple) or len(dims) != 2:
            raise SemanticQuotientError(f"{label} must be a two-integer tuple")
        for dimension in dims:
            _require_int(dimension, f"{label} dimension", minimum=1, maximum=MAX_DIMENSION)
    custody = _validate_target_teacher_custody(
        target_teacher_custody,
        pair_count=count,
        chunk_pairs=chunk_size,
        scorer_hw=scorer_hw,
        channels=channel_count,
    )
    if not isinstance(resume, bool):
        raise SemanticQuotientError("resume must be boolean")
    root = Path(work_root)
    target = Path(archive_path)
    root_resolved = root.resolve(strict=False)
    target_resolved = target.resolve(strict=False)
    if root_resolved not in target_resolved.parents:
        raise SemanticQuotientError("archive_path must be a strict descendant of work_root")
    root = root_resolved
    target = target_resolved
    plane_bytes = count * scorer_hw[0] * scorer_hw[1] * channel_count
    preflight = storage_preflight(
        root,
        required_bytes=len(semantic_packet) + 4 * plane_bytes + (1 << 20),
        test_only_small_fixture=test_only_small_fixture,
        allow_local_storage=allow_local_storage,
    )
    if root.exists() and not resume and any(root.iterdir()):
        raise SemanticQuotientError("fresh build refuses a non-empty work root")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SemanticQuotientError("cannot create semantic quotient work root") from exc
    inputs = {
        "schema": ARCHIVE_SCHEMA,
        "semantic_packet_bytes": len(semantic_packet),
        "semantic_packet_sha256": _sha256(semantic_packet),
        "renderer": identity.as_manifest(),
        "target_teacher_custody": dict(custody),
        "geometry": {"camera_hw": list(camera_hw), "scorer_hw": list(scorer_hw), "channels": channel_count},
        "pair_count": count,
        "chunk_pairs": chunk_size,
        "storage_preflight": _stable_storage_receipt(preflight),
        "test_only_small_fixture": bool(test_only_small_fixture),
        "score_claim": False,
        "promotion_eligible": False,
    }
    canonical_json(inputs)
    write_once_or_equal(root / "stage_checkpoints" / "00_inputs.json", canonical_json(inputs))
    write_once_or_equal(root / "semantic" / "base.packet", semantic_packet)

    try:
        operator = DisjointResizeOperator.build(
            camera_h=camera_hw[0],
            camera_w=camera_hw[1],
            scorer_h=scorer_hw[0],
            scorer_w=scorer_hw[1],
        )
    except Uint8LatticeError as exc:
        raise SemanticQuotientError("production factor-2 geometry refused") from exc
    base_y0_digest = hashlib.sha256()
    base_y1_digest = hashlib.sha256()
    target_y0_digest = hashlib.sha256()
    target_y1_digest = hashlib.sha256()
    expected_pair = 0
    expected_chunk = 0
    chunk_rows: list[dict[str, Any]] = []
    renderer_rows = _renderer_chunks(
        renderer,
        semantic_packet,
        work_root=root / "semantic_renderer",
        chunk_pairs=chunk_size,
        resume=resume,
    )
    for base, teacher in zip(renderer_rows, target_chunks, strict=True):
        if not isinstance(teacher, PlaneChunk):
            raise SemanticQuotientError("target teacher emitted a non-PlaneChunk value")
        _validate_chunk_geometry(
            base,
            expected_chunk_index=expected_chunk,
            expected_pair_start=expected_pair,
            scorer_hw=scorer_hw,
            channels=channel_count,
        )
        _validate_chunk_geometry(
            teacher,
            expected_chunk_index=expected_chunk,
            expected_pair_start=expected_pair,
            scorer_hw=scorer_hw,
            channels=channel_count,
        )
        if base.pair_ids != teacher.pair_ids or base.y0.shape != teacher.y0.shape:
            raise SemanticQuotientError("semantic base and target teacher chunks do not align")
        expected_chunk_pairs = min(chunk_size, count - expected_pair)
        if len(base.pair_ids) != expected_chunk_pairs:
            raise SemanticQuotientError("renderer/teacher chunk size arithmetic differs")
        quotient_rows: dict[str, Any] = {}
        for plane, (key, base_plane, target_plane) in enumerate(
            (("y0", base.y0, teacher.y0), ("y1", base.y1, teacher.y1))
        ):
            raw = np.bitwise_xor(base_plane, target_plane).tobytes(order="C")
            coded = _compress_quotient(raw)
            member = _quotient_member(expected_chunk, plane)
            write_once_or_equal(_checkpoint_member(root, expected_chunk, plane), coded)
            quotient_rows[key] = {
                "member": member,
                "raw_bytes": len(raw),
                "raw_sha256": _sha256(raw),
                "coded_bytes": len(coded),
                "coded_sha256": _sha256(coded),
                "video_derived": True,
            }
        factor2 = _factor2_chunk_receipt(teacher, operator=operator)
        row = {
            "chunk_index": expected_chunk,
            "pair_ids": list(base.pair_ids),
            "base": {
                "y0_sha256": _sha256(base.y0.tobytes(order="C")),
                "y1_sha256": _sha256(base.y1.tobytes(order="C")),
            },
            "target": {
                "y0_sha256": _sha256(teacher.y0.tobytes(order="C")),
                "y1_sha256": _sha256(teacher.y1.tobytes(order="C")),
            },
            "quotient": quotient_rows,
            "factor2": factor2,
        }
        write_once_or_equal(
            root / "stage_checkpoints" / f"01_chunk_{expected_chunk:04d}.json",
            canonical_json({"schema": CHUNK_STAGE_SCHEMA, **row}),
        )
        chunk_rows.append(row)
        base_y0_digest.update(base.y0.tobytes(order="C"))
        base_y1_digest.update(base.y1.tobytes(order="C"))
        target_y0_digest.update(teacher.y0.tobytes(order="C"))
        target_y1_digest.update(teacher.y1.tobytes(order="C"))
        expected_pair += len(base.pair_ids)
        expected_chunk += 1
    if expected_pair != count or expected_chunk != (count + chunk_size - 1) // chunk_size:
        raise SemanticQuotientError("renderer/teacher did not cover the declared pair set")
    _validate_target_teacher_custody(
        custody,
        pair_count=count,
        chunk_pairs=chunk_size,
        scorer_hw=scorer_hw,
        channels=channel_count,
        observed_chunks=chunk_rows,
    )
    if custody["y0_sha256"] != target_y0_digest.hexdigest() or custody["y1_sha256"] != target_y1_digest.hexdigest():
        raise SemanticQuotientError("observed target aggregates differ from teacher custody")
    manifest = _build_manifest(
        semantic_packet=semantic_packet,
        renderer_identity=identity,
        pair_count=count,
        chunk_pairs=chunk_size,
        camera_hw=camera_hw,
        scorer_hw=scorer_hw,
        channels=channel_count,
        chunk_rows=chunk_rows,
        target_teacher_custody=custody,
        base_y0_sha256=base_y0_digest.hexdigest(),
        base_y1_sha256=base_y1_digest.hexdigest(),
        target_y0_sha256=target_y0_digest.hexdigest(),
        target_y1_sha256=target_y1_digest.hexdigest(),
        target_plane_bytes=plane_bytes,
    )
    manifest_payload = canonical_json(manifest)
    write_once_or_equal(root / "stage_checkpoints" / "02_manifest.json", manifest_payload)
    expected_archive = _build_archive_file(
        archive_path=target,
        manifest=manifest,
        semantic_packet=semantic_packet,
        work_root=root,
    )
    parsed = parse_semantic_quotient_archive(target)
    if (
        parsed.manifest != manifest
        or parsed.semantic_packet != semantic_packet
        or parsed.archive_bytes != len(expected_archive)
        or parsed.archive_sha256 != _sha256(expected_archive)
    ):
        raise SemanticQuotientError("published archive differs from rebuilt canonical scientific state")
    archive_checkpoint = {
        "schema": "tac.c0b_semantic_quotient_archive_stage.v1",
        "archive_path": str(target),
        "archive_bytes": parsed.archive_bytes,
        "archive_sha256": parsed.archive_sha256,
        "manifest_sha256": _sha256(manifest_payload),
        "score_claim": False,
        "promotion_eligible": False,
    }
    write_once_or_equal(root / "stage_checkpoints" / "02_archive.json", canonical_json(archive_checkpoint))
    double_decode = double_decode_archive(
        target,
        renderer,
        work_root=root,
        resume=True,
        verify_factor2=True,
    )
    receipt = {
        "schema": BUILD_RECEIPT_SCHEMA,
        "scientific_label": SCIENTIFIC_LABEL,
        "archive": {
            "path": str(target),
            "bytes": parsed.archive_bytes,
            "sha256": parsed.archive_sha256,
        },
        "manifest_sha256": _sha256(manifest_payload),
        "semantic_packet_sha256": _sha256(semantic_packet),
        "double_decode": double_decode,
        "storage_preflight": _stable_storage_receipt(preflight),
        "resumable": True,
        "per_stage_checkpoints": True,
        "dense_nonpromotable_baseline": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    receipt_path = root / "build_receipt.json"
    write_once_or_equal(receipt_path, canonical_json(receipt))
    return SemanticQuotientBuildResult(
        target,
        receipt_path,
        parsed.archive_sha256,
        parsed.archive_bytes,
        manifest,
        double_decode,
    )


__all__ = [
    "ARCHIVE_SCHEMA",
    "BUILD_RECEIPT_SCHEMA",
    "DOUBLE_DECODE_SCHEMA",
    "FACTOR2_CONTRACT_ID",
    "MANIFEST_SCHEMA",
    "QUOTIENT_CODEC_ID",
    "RENDERER_CONTRACT_ID",
    "SCIENTIFIC_LABEL",
    "SEMANTIC_BASE_TYPE_ID",
    "ParsedSemanticQuotientArchive",
    "PlaneChunk",
    "RendererIdentity",
    "SemanticPlaneRenderer",
    "SemanticQuotientBuildResult",
    "SemanticQuotientError",
    "build_semantic_quotient_archive",
    "canonical_json",
    "double_decode_archive",
    "exact_resize_round_u8",
    "parse_semantic_quotient_archive",
    "select_storage_root",
    "sha256_file",
    "storage_preflight",
    "write_once_or_equal",
]
