# SPDX-License-Identifier: MIT
"""Counted G/A[/T] fragment receiver for the finite task-space program.

The ZIP contains one canonical counted member.  That member is a strict binary
container whose first section is the literal ``G`` packet and whose optional
remaining sections are the exact ``A`` packet and terminal quotient.  This
module owns only ``G`` semantics: it binds the other counted sections as bytes
but deliberately leaves their typed semantic parse to their owning receivers.

The receiver strict-parses and applies ``G`` to a freshly supplied predictor
semantic state, paints the counted realization profile, and realizes the
resulting frame-1 scorer plane into exact uint8 camera bytes.  It never imports
or invokes a scorer.  Production-size output is streamed with per-pair durable
checkpoints instead of being retained in memory.

The causal proof is behavioral.  It replays the same baseline fragment twice,
then decodes a valid fragment differing only in its counted ``G`` section.  It
requires the decoded semantic state, scorer-grid RGB, and realized raw bytes to
all change while runtime, predictor state, and every non-G section remain
fixed.  Caller-authored presence flags, output hashes, or receipt sidecars are
not accepted as proof.

This is research-only structural custody.  The ZIP and its raw output are only
a G/A[/T] fragment: G is semantically consumed here, A/T are byte-bound for
their owning receivers, and the emitted bytes are realized Y1 rather than a
complete chronological Y0/Y1 contest stream.  A counted P receiver, the A
receiver, and root composition must still derive the predictor state, produce
Y0 conditioned on exact Y1, and emit the complete pair before any candidate,
score, originality, or promotion claim.
The closure is specifically against ``PredictorSemanticStateV1``.  Its Pose6
codes are consumed by transported finite primitives; this module neither
synthesizes Pose6 from level-set latents nor proves compatibility with an ep725
frame1 predictor that has no typed Pose6 coordinate.  That vehicle requires a
separate counted temporal-coordinate adapter.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import shutil
import stat
import struct
import tempfile
import zipfile
import zlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, BinaryIO, Final, Literal

import numpy as np

import tac.optimization.direct_description_carrier_compose as _direct_description_module
import tac.optimization.uint8_lattice_feasibility as _uint8_lattice_module
import tac.witness_dsl.factorized_v9_predictor as _factorized_predictor_module
import tac.witness_dsl.generative_taskspace_correction as _generative_correction_module
from tac.optimization.direct_description_carrier_compose import REALIZATION_PAINT_ORDER, ROLE_CLASS_IDS
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Uint8LatticeError,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)
from tac.witness_dsl.generative_taskspace_correction import (
    DecodedGenerativeCorrectionV1,
    GenerativeTaskspaceCorrectionError,
    PredictorSemanticStateV1,
    apply_generative_taskspace_correction,
)

ARCHIVE_MEMBER_NAME: Final = "0.bin"
FRAGMENT_ARCHIVE_SCHEMA: Final = "tac.generative_taskspace_receiver_fragment_archive.v1"
COUNTED_PAYLOAD_SCHEMA: Final = "tac.taskspace_counted_payload.v1"
DECODE_RECEIPT_SCHEMA: Final = "tac.generative_taskspace_receiver_decode_receipt.v1"
CONSUMPTION_PROOF_SCHEMA: Final = "tac.generative_taskspace_receiver_consumption_proof.v1"
CONSUMPTION_PROOF_ENVELOPE_SCHEMA: Final = "tac.generative_taskspace_receiver_consumption_proof.envelope.v1"
RECEIVER_CONTRACT_ID: Final = "strict-G-parse-apply-factor2-Y1-raw.v1"

_COUNTED_MAGIC: Final = b"TACG2R\x00\x00"
_COUNTED_VERSION: Final = 1
_COUNTED_PREFIX: Final = struct.Struct(">8sBBH")
_SECTION_DESCRIPTOR: Final = struct.Struct(">BQI32s")
_MAX_SECTION_BYTES: Final = 1 << 31
_MAX_COUNTED_PAYLOAD_BYTES: Final = 1 << 32
_MAX_PAIR_COUNT: Final = 600
_ZIP_MODE: Final = 0o100644 << 16
_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
_SCORER_HEIGHT: Final = 384
_SCORER_WIDTH: Final = 512
_CHANNELS: Final = 3
_LITERAL_SCAN_MIN_SECTION_BYTES: Final = 32


class CountedTaskspaceRole(StrEnum):
    """Canonical counted-section order for this G/A/optional-T fragment."""

    GENERATIVE_CORRECTION = "generative_correction"
    COUPLED_PREIMAGE = "coupled_preimage"
    TERMINAL_QUOTIENT = "terminal_quotient"


_ROLE_CODE: Final = {
    CountedTaskspaceRole.GENERATIVE_CORRECTION: 1,
    CountedTaskspaceRole.COUPLED_PREIMAGE: 2,
    CountedTaskspaceRole.TERMINAL_QUOTIENT: 3,
}
_ROLE_BY_CODE: Final = {code: role for role, code in _ROLE_CODE.items()}
_ADMITTED_ROLE_ORDERS: Final = (
    (CountedTaskspaceRole.GENERATIVE_CORRECTION,),
    (
        CountedTaskspaceRole.GENERATIVE_CORRECTION,
        CountedTaskspaceRole.COUPLED_PREIMAGE,
    ),
    (
        CountedTaskspaceRole.GENERATIVE_CORRECTION,
        CountedTaskspaceRole.COUPLED_PREIMAGE,
        CountedTaskspaceRole.TERMINAL_QUOTIENT,
    ),
)


class GenerativeTaskspaceReceiverError(ValueError):
    """Malformed counted bytes, failed G causality, or raw-output custody."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with _open_regular_binary(path, flags=os.O_RDONLY, mode="rb", label="hash input") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file_prefix(path: Path, byte_count: int, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    remaining = byte_count
    with _open_regular_binary(path, flags=os.O_RDONLY, mode="rb", label="partial raw") as handle:
        while remaining:
            chunk = handle.read(min(remaining, chunk_bytes))
            if not chunk:
                raise GenerativeTaskspaceReceiverError("partial raw is shorter than its durable checkpoint")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise GenerativeTaskspaceReceiverError("receipt value is not canonical finite ASCII JSON") from exc


def _decode_canonical_json(payload: bytes, *, label: str) -> Any:
    def unique_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise GenerativeTaskspaceReceiverError(f"{label} contains duplicate JSON keys")
            value[key] = item
        return value

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GenerativeTaskspaceReceiverError(f"{label} is not ASCII JSON") from exc
    if _canonical_json(value) != payload:
        raise GenerativeTaskspaceReceiverError(f"{label} is not canonical JSON")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _stable_file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _lstat_optional(path: Path, *, label: str) -> os.stat_result | None:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise GenerativeTaskspaceReceiverError(f"{label} path cannot be inspected safely") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise GenerativeTaskspaceReceiverError(f"{label} path must be a regular non-symlink file")
    return value


def _open_regular_binary(
    path: Path,
    *,
    flags: int,
    mode: str,
    label: str,
    create: bool = False,
    exclusive_create: bool = False,
) -> BinaryIO:
    """Open a regular path without following symlinks and bind its inode."""

    _lstat_optional(path, label=label)
    open_flags = flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if create:
        open_flags |= os.O_CREAT
    if exclusive_create:
        if not create:
            raise GenerativeTaskspaceReceiverError("exclusive regular-file creation requires create=True")
        open_flags |= os.O_EXCL
    try:
        descriptor = os.open(path, open_flags, 0o600)
    except OSError as exc:
        raise GenerativeTaskspaceReceiverError(f"{label} path cannot be opened as a regular non-symlink file") from exc
    try:
        opened = os.fstat(descriptor)
        current = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or _file_identity(opened) != _file_identity(current)
        ):
            raise GenerativeTaskspaceReceiverError(f"{label} path changed identity during safe open")
        return os.fdopen(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        raise


def _require_regular_identity(path: Path, expected: tuple[int, int], *, label: str) -> os.stat_result:
    current = _lstat_optional(path, label=label)
    if current is None or _file_identity(current) != expected:
        raise GenerativeTaskspaceReceiverError(f"{label} path changed identity during receiver execution")
    return current


def _read_stable_regular_bytes(path: Path, *, label: str) -> bytes:
    """Read one regular file through a bound descriptor and reject mutation/replacement."""

    try:
        with _open_regular_binary(path, flags=os.O_RDONLY, mode="rb", label=label) as handle:
            before = os.fstat(handle.fileno())
            payload = handle.read()
            after = os.fstat(handle.fileno())
            current = _require_regular_identity(path, _file_identity(before), label=label)
    except GenerativeTaskspaceReceiverError:
        raise
    except OSError as exc:
        raise GenerativeTaskspaceReceiverError(f"{label} cannot be read stably") from exc
    if (
        _stable_file_identity(before) != _stable_file_identity(after)
        or _stable_file_identity(after) != _stable_file_identity(current)
        or len(payload) != after.st_size
    ):
        raise GenerativeTaskspaceReceiverError(f"{label} mutated during descriptor-bound read")
    return payload


def _cleanup_owned_temporary(path: Path, expected: tuple[int, int]) -> None:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise GenerativeTaskspaceReceiverError("atomic-write temporary cannot be inspected for cleanup") from exc
    if stat.S_ISREG(current.st_mode) and _file_identity(current) == expected:
        path.unlink()
        _fsync_directory(path.parent)
        return
    raise GenerativeTaskspaceReceiverError("atomic-write temporary changed identity before cleanup")


def _atomic_write(path: Path, payload: bytes) -> None:
    """Collision-safe same-directory atomic write with deterministic cleanup."""

    _lstat_optional(path, label="atomic-write destination")
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
    except OSError as exc:
        raise GenerativeTaskspaceReceiverError("atomic-write temporary cannot be created") from exc
    temporary = Path(temporary_name)
    identity = _file_identity(os.fstat(descriptor))
    installed = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _require_regular_identity(temporary, identity, label="atomic-write temporary")
        _lstat_optional(path, label="atomic-write destination")
        os.replace(temporary, path)
        installed = True
        _fsync_directory(path.parent)
    finally:
        if not installed:
            _cleanup_owned_temporary(temporary, identity)


def _exact_positive_int(value: Any, label: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise GenerativeTaskspaceReceiverError(f"{label} is not an admitted exact positive integer")
    return value


@dataclass(frozen=True, slots=True)
class CountedTaskspaceSectionV1:
    """One exact typed section inside the sole counted ZIP member."""

    role: CountedTaskspaceRole
    offset: int
    payload: bytes
    crc32: int

    def __post_init__(self) -> None:
        if not isinstance(self.role, CountedTaskspaceRole):
            raise GenerativeTaskspaceReceiverError("counted section role is invalid")
        if type(self.offset) is not int or self.offset < 0:
            raise GenerativeTaskspaceReceiverError("counted section offset is invalid")
        if not isinstance(self.payload, bytes) or not self.payload or len(self.payload) > _MAX_SECTION_BYTES:
            raise GenerativeTaskspaceReceiverError("counted section payload is empty, non-bytes, or too large")
        if type(self.crc32) is not int or self.crc32 != zlib.crc32(self.payload) & 0xFFFFFFFF:
            raise GenerativeTaskspaceReceiverError("counted section CRC differs from exact payload bytes")

    @property
    def byte_length(self) -> int:
        return len(self.payload)

    @property
    def payload_sha256(self) -> str:
        return _sha256(self.payload)

    def custody_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "offset": self.offset,
            "byte_length": self.byte_length,
            "payload_sha256": self.payload_sha256,
            "crc32": self.crc32,
        }


@dataclass(frozen=True, slots=True)
class ParsedCountedTaskspacePayloadV1:
    payload: bytes
    sections: tuple[CountedTaskspaceSectionV1, ...]

    @property
    def payload_sha256(self) -> str:
        return _sha256(self.payload)

    def section(self, role: CountedTaskspaceRole) -> CountedTaskspaceSectionV1:
        rows = tuple(row for row in self.sections if row.role is role)
        if len(rows) != 1:
            raise GenerativeTaskspaceReceiverError(f"counted payload has no unique {role.value} section")
        return rows[0]

    @property
    def non_g_sections(self) -> tuple[CountedTaskspaceSectionV1, ...]:
        return tuple(row for row in self.sections if row.role is not CountedTaskspaceRole.GENERATIVE_CORRECTION)

    @property
    def non_g_manifest_sha256(self) -> str:
        return _sha256(_canonical_json([row.custody_dict() for row in self.non_g_sections]))


@dataclass(frozen=True, slots=True)
class ParsedGenerativeTaskspaceFragmentArchiveV1:
    """One counted G/A[/T] ZIP fragment; no counted P or complete pair."""

    archive: bytes
    counted: ParsedCountedTaskspacePayloadV1
    member_crc32: int

    @property
    def archive_sha256(self) -> str:
        return _sha256(self.archive)


@dataclass(frozen=True, slots=True)
class ReceiverRuntimeFileV1:
    module: str
    source_bytes: int
    source_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "source_bytes": self.source_bytes,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class GReceiverDecodeReceiptV1:
    schema: Literal["tac.generative_taskspace_receiver_decode_receipt.v1"]
    archive_bytes: int
    archive_sha256: str
    archive_member_name: str
    archive_member_bytes: int
    archive_member_sha256: str
    archive_member_crc32: int
    rate_charged_archive_bytes: int
    generic_receiver_code_counted_bytes: int
    g_section_offset: int
    g_section_bytes: int
    g_section_sha256: str
    g_section_crc32: int
    non_g_sections: tuple[Mapping[str, Any], ...]
    non_g_manifest_sha256: str
    predictor_binding_sha256: str
    predictor_program_sha256: str
    predictor_renderer_sha256: str
    predictor_labels_sha256: str
    predictor_pose6_sha256: str
    predictor_state_contract: str
    pose6_serialized_in_g: bool
    levelset_latents_relabelled_as_pose6: bool
    ep725_frame1_predictor_compatible: bool
    counted_temporal_coordinate_adapter_owed: bool
    source_pair_ids: tuple[int, ...]
    decoded_labels_sha256: str
    changed_semantic_cells: int
    scorer_y1_bytes: int
    scorer_y1_sha256: str
    realized_y1_raw_bytes: int
    realized_y1_raw_sha256: str
    scorer_height: int
    scorer_width: int
    camera_height: int
    camera_width: int
    channels: int
    factor2_scorer_values_verified: int
    factor2_numerator_values_verified: int
    runtime_files: tuple[ReceiverRuntimeFileV1, ...]
    runtime_manifest_sha256: str
    receiver_contract_id: str
    strict_g_parse_back: bool
    g_applied_to_predictor_state: bool
    realized_through_factor2_uint8: bool
    raw_output_materialized: bool
    raw_output_role: str
    complete_chronological_pair_output: bool
    non_g_sections_consumed_by_g_receiver: bool
    repository_decoder_dependency: bool
    standalone_runtime_closure_bound: bool
    literal_counted_sections_ge_32_absent_from_runtime_sources: bool
    literal_scan_min_section_bytes: int
    complete_data_in_code_lineage_audit: bool
    research_only: bool
    score_claim: bool
    originality_claim: bool
    promotion_eligible: bool

    def as_dict(self) -> dict[str, Any]:
        value = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field not in {"runtime_files", "non_g_sections", "source_pair_ids"}
        }
        value["runtime_files"] = [row.as_dict() for row in self.runtime_files]
        value["non_g_sections"] = [dict(row) for row in self.non_g_sections]
        value["source_pair_ids"] = list(self.source_pair_ids)
        return value


@dataclass(frozen=True, slots=True)
class GReceiverDecodeResultV1:
    decoded: DecodedGenerativeCorrectionV1
    receipt: GReceiverDecodeReceiptV1


@dataclass(frozen=True, slots=True)
class GReceiverConsumptionProofV1:
    """Untrusted claim object until fresh behavioral replay verifies its exact bytes."""

    baseline: GReceiverDecodeReceiptV1
    counterfactual: GReceiverDecodeReceiptV1
    baseline_replay_identical: bool
    same_runtime_manifest: bool
    same_predictor_binding: bool
    target_g_section_only_changed: bool
    unchanged_non_g_sections: bool
    decoded_state_changed: bool
    scorer_y1_changed: bool
    realized_raw_output_changed: bool
    archive_member_to_strict_g_parse_apply_to_raw_causal: bool
    candidate_archive_eligible: bool = False
    n600_complete_pair_archive_present: bool = False
    research_only: bool = True
    score_claim: bool = False
    originality_claim: bool = False
    promotion_eligible: bool = False
    remaining_blocker: str = (
        "compose source-custodied A-derived Y0 conditioned on exact Y1, emit a standalone chronological n600 "
        "Y0/Y1 archive, close runtime lineage, and exact-evaluate the same archive"
    )

    def __post_init__(self) -> None:
        required_true = (
            self.baseline_replay_identical,
            self.same_runtime_manifest,
            self.same_predictor_binding,
            self.target_g_section_only_changed,
            self.unchanged_non_g_sections,
            self.decoded_state_changed,
            self.scorer_y1_changed,
            self.realized_raw_output_changed,
            self.archive_member_to_strict_g_parse_apply_to_raw_causal,
        )
        if not all(value is True for value in required_true):
            raise GenerativeTaskspaceReceiverError("G consumption proof lacks a required behavioral custody leg")
        if (
            any(
                value is not False
                for value in (
                    self.candidate_archive_eligible,
                    self.n600_complete_pair_archive_present,
                    self.score_claim,
                    self.originality_claim,
                    self.promotion_eligible,
                )
            )
            or self.research_only is not True
        ):
            raise GenerativeTaskspaceReceiverError("structural G proof cannot claim candidate authority")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": CONSUMPTION_PROOF_SCHEMA,
            "baseline": self.baseline.as_dict(),
            "counterfactual": self.counterfactual.as_dict(),
            "baseline_replay_identical": self.baseline_replay_identical,
            "same_runtime_manifest": self.same_runtime_manifest,
            "same_predictor_binding": self.same_predictor_binding,
            "target_g_section_only_changed": self.target_g_section_only_changed,
            "unchanged_non_g_sections": self.unchanged_non_g_sections,
            "decoded_state_changed": self.decoded_state_changed,
            "scorer_y1_changed": self.scorer_y1_changed,
            "realized_raw_output_changed": self.realized_raw_output_changed,
            "archive_member_to_strict_g_parse_apply_to_raw_causal": (
                self.archive_member_to_strict_g_parse_apply_to_raw_causal
            ),
            "candidate_archive_eligible": self.candidate_archive_eligible,
            "n600_complete_pair_archive_present": self.n600_complete_pair_archive_present,
            "research_only": self.research_only,
            "score_claim": self.score_claim,
            "originality_claim": self.originality_claim,
            "promotion_eligible": self.promotion_eligible,
            "remaining_blocker": self.remaining_blocker,
        }

    def to_bytes(self) -> bytes:
        """Serialize an untrusted claim; this alone grants no durable proof authority."""

        body = self.as_dict()
        return _canonical_json(
            {
                "schema": CONSUMPTION_PROOF_ENVELOPE_SCHEMA,
                "body": body,
                "body_sha256": _sha256(_canonical_json(body)),
            }
        )


def encode_counted_taskspace_payload(
    g_packet: bytes,
    *,
    coupled_preimage_packet: bytes | None = None,
    terminal_quotient_packet: bytes | None = None,
) -> bytes:
    """Encode the canonical G -> A -> optional T counted payload."""

    rows: list[tuple[CountedTaskspaceRole, bytes]] = [(CountedTaskspaceRole.GENERATIVE_CORRECTION, g_packet)]
    if coupled_preimage_packet is not None:
        rows.append((CountedTaskspaceRole.COUPLED_PREIMAGE, coupled_preimage_packet))
    if terminal_quotient_packet is not None:
        if coupled_preimage_packet is None:
            raise GenerativeTaskspaceReceiverError("terminal quotient requires the preceding counted A section")
        rows.append((CountedTaskspaceRole.TERMINAL_QUOTIENT, terminal_quotient_packet))
    roles = tuple(role for role, _payload in rows)
    if roles not in _ADMITTED_ROLE_ORDERS:
        raise GenerativeTaskspaceReceiverError("counted payload roles are not canonical G -> A -> optional T")
    for role, payload in rows:
        if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_SECTION_BYTES:
            raise GenerativeTaskspaceReceiverError(f"counted {role.value} section is empty, non-bytes, or too large")
    prefix = _COUNTED_PREFIX.pack(_COUNTED_MAGIC, _COUNTED_VERSION, len(rows), 0)
    descriptors = b"".join(
        _SECTION_DESCRIPTOR.pack(
            _ROLE_CODE[role],
            len(payload),
            zlib.crc32(payload) & 0xFFFFFFFF,
            bytes.fromhex(_sha256(payload)),
        )
        for role, payload in rows
    )
    encoded = prefix + descriptors + b"".join(payload for _role, payload in rows)
    if len(encoded) > _MAX_COUNTED_PAYLOAD_BYTES:
        raise GenerativeTaskspaceReceiverError("counted payload exceeds its uint32-scale bound")
    parsed = parse_counted_taskspace_payload(encoded)
    if tuple(section.payload for section in parsed.sections) != tuple(payload for _role, payload in rows):
        raise GenerativeTaskspaceReceiverError("counted payload changed on strict parse-back")
    return encoded


def parse_counted_taskspace_payload(payload: bytes) -> ParsedCountedTaskspacePayloadV1:
    """Strictly parse every counted section and reject all unconsumed bytes."""

    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_COUNTED_PAYLOAD_BYTES:
        raise GenerativeTaskspaceReceiverError("counted payload is empty, non-bytes, or too large")
    if len(payload) < _COUNTED_PREFIX.size:
        raise GenerativeTaskspaceReceiverError("counted payload prefix is truncated")
    magic, version, section_count, flags = _COUNTED_PREFIX.unpack_from(payload)
    if magic != _COUNTED_MAGIC or version != _COUNTED_VERSION or flags != 0:
        raise GenerativeTaskspaceReceiverError("counted payload magic/version/flags are invalid")
    if section_count not in {1, 2, 3}:
        raise GenerativeTaskspaceReceiverError("counted payload section count is invalid")
    directory_stop = _COUNTED_PREFIX.size + section_count * _SECTION_DESCRIPTOR.size
    if directory_stop > len(payload):
        raise GenerativeTaskspaceReceiverError("counted payload section directory is truncated")
    descriptors: list[tuple[CountedTaskspaceRole, int, int, str]] = []
    for index in range(section_count):
        descriptor_offset = _COUNTED_PREFIX.size + index * _SECTION_DESCRIPTOR.size
        role_code, length, checksum, digest = _SECTION_DESCRIPTOR.unpack_from(payload, descriptor_offset)
        role = _ROLE_BY_CODE.get(role_code)
        if role is None:
            raise GenerativeTaskspaceReceiverError("counted payload contains an unknown section role")
        if not 0 < length <= _MAX_SECTION_BYTES:
            raise GenerativeTaskspaceReceiverError("counted section length is invalid")
        descriptors.append((role, length, checksum, digest.hex()))
    roles = tuple(row[0] for row in descriptors)
    if roles not in _ADMITTED_ROLE_ORDERS:
        raise GenerativeTaskspaceReceiverError("counted payload section order is not G -> A -> optional T")
    cursor = directory_stop
    sections: list[CountedTaskspaceSectionV1] = []
    for role, length, checksum, digest in descriptors:
        stop = cursor + length
        if stop > len(payload):
            raise GenerativeTaskspaceReceiverError("counted section is truncated")
        section_payload = payload[cursor:stop]
        if _sha256(section_payload) != digest or zlib.crc32(section_payload) & 0xFFFFFFFF != checksum:
            raise GenerativeTaskspaceReceiverError("counted section hash or CRC mismatch")
        sections.append(
            CountedTaskspaceSectionV1(
                role=role,
                offset=cursor,
                payload=section_payload,
                crc32=checksum,
            )
        )
        cursor = stop
    if cursor != len(payload):
        raise GenerativeTaskspaceReceiverError("counted payload has trailing or unconsumed bytes")
    # Fixed-width descriptors, canonical role order, exact hashes/CRCs, and
    # total stream consumption leave no alternate valid encoding.
    return ParsedCountedTaskspacePayloadV1(payload=payload, sections=tuple(sections))


def _canonical_zip(counted_payload: bytes) -> bytes:
    output = io.BytesIO()
    info = zipfile.ZipInfo(ARCHIVE_MEMBER_NAME, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = _ZIP_MODE
    info.flag_bits = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as archive:
        archive.writestr(info, counted_payload)
    return output.getvalue()


def parse_generative_taskspace_receiver_fragment_archive(
    archive: bytes,
) -> ParsedGenerativeTaskspaceFragmentArchiveV1:
    """Parse one G/A[/T] fragment, not a closed P/G/A candidate archive."""

    if not isinstance(archive, bytes) or not archive:
        raise GenerativeTaskspaceReceiverError("receiver fragment archive must be nonempty exact bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
            infos = reader.infolist()
            if len(infos) != 1 or infos[0].filename != ARCHIVE_MEMBER_NAME or reader.comment:
                raise GenerativeTaskspaceReceiverError("receiver fragment archive must contain exactly canonical 0.bin")
            info = infos[0]
            if (
                info.compress_type != zipfile.ZIP_STORED
                or info.date_time != _ZIP_TIMESTAMP
                or info.create_system != 3
                or info.external_attr != _ZIP_MODE
                or info.extra
                or info.comment
                or info.file_size > _MAX_COUNTED_PAYLOAD_BYTES
            ):
                raise GenerativeTaskspaceReceiverError("receiver fragment member metadata is not canonical")
            counted_payload = reader.read(info)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise GenerativeTaskspaceReceiverError("receiver fragment archive is not a valid counted ZIP") from exc
    if _canonical_zip(counted_payload) != archive:
        raise GenerativeTaskspaceReceiverError("receiver fragment archive is not canonical deterministic ZIP")
    return ParsedGenerativeTaskspaceFragmentArchiveV1(
        archive=archive,
        counted=parse_counted_taskspace_payload(counted_payload),
        member_crc32=info.CRC,
    )


def build_generative_taskspace_receiver_fragment_archive(
    g_packet: bytes,
    *,
    predictor_state: PredictorSemanticStateV1,
    coupled_preimage_packet: bytes | None = None,
    terminal_quotient_packet: bytes | None = None,
) -> bytes:
    """Build a G/A[/T] ZIP fragment only after strict G parse/apply succeeds."""

    try:
        apply_generative_taskspace_correction(g_packet, predictor_state=predictor_state)
    except GenerativeTaskspaceCorrectionError as exc:
        raise GenerativeTaskspaceReceiverError("strict G receiver refused the counted packet") from exc
    counted = encode_counted_taskspace_payload(
        g_packet,
        coupled_preimage_packet=coupled_preimage_packet,
        terminal_quotient_packet=terminal_quotient_packet,
    )
    archive = _canonical_zip(counted)
    parsed = parse_generative_taskspace_receiver_fragment_archive(archive)
    if parsed.counted.section(CountedTaskspaceRole.GENERATIVE_CORRECTION).payload != g_packet:
        raise GenerativeTaskspaceReceiverError("built fragment changed literal G bytes on parse-back")
    return archive


def build_g_only_counterfactual_fragment_archive(
    baseline_archive: bytes,
    counterfactual_g_packet: bytes,
    *,
    predictor_state: PredictorSemanticStateV1,
) -> bytes:
    """Replace only G in a fragment while retaining every non-G section byte-for-byte."""

    baseline = parse_generative_taskspace_receiver_fragment_archive(baseline_archive)
    roles = tuple(row.role for row in baseline.counted.sections)
    return build_generative_taskspace_receiver_fragment_archive(
        counterfactual_g_packet,
        predictor_state=predictor_state,
        coupled_preimage_packet=(
            baseline.counted.section(CountedTaskspaceRole.COUPLED_PREIMAGE).payload
            if CountedTaskspaceRole.COUPLED_PREIMAGE in roles
            else None
        ),
        terminal_quotient_packet=(
            baseline.counted.section(CountedTaskspaceRole.TERMINAL_QUOTIENT).payload
            if CountedTaskspaceRole.TERMINAL_QUOTIENT in roles
            else None
        ),
    )


def _runtime_files() -> tuple[ReceiverRuntimeFileV1, ...]:
    modules = (
        ("tac.witness_dsl.generative_taskspace_receiver", Path(__file__)),
        ("tac.witness_dsl.generative_taskspace_correction", Path(_generative_correction_module.__file__ or "")),
        ("tac.witness_dsl.factorized_v9_predictor", Path(_factorized_predictor_module.__file__ or "")),
        (
            "tac.optimization.direct_description_carrier_compose",
            Path(_direct_description_module.__file__ or ""),
        ),
        ("tac.optimization.uint8_lattice_feasibility", Path(_uint8_lattice_module.__file__ or "")),
    )
    rows: list[ReceiverRuntimeFileV1] = []
    for module, source_path in modules:
        payload = _read_stable_regular_bytes(source_path, label="receiver runtime source")
        rows.append(ReceiverRuntimeFileV1(module=module, source_bytes=len(payload), source_sha256=_sha256(payload)))
    return tuple(rows)


def _runtime_manifest_sha256(rows: Sequence[ReceiverRuntimeFileV1]) -> str:
    return _sha256(_canonical_json([row.as_dict() for row in rows]))


def _paint_palette(decoded: DecodedGenerativeCorrectionV1) -> np.ndarray:
    if decoded.realization_profile is None:
        raise GenerativeTaskspaceReceiverError("counted G has no realization profile for evaluator-facing RGB")
    palette = np.zeros((5, 3), dtype=np.uint8)
    populated: set[int] = set()
    for role in REALIZATION_PAINT_ORDER:
        class_id = int(ROLE_CLASS_IDS[role])
        palette[class_id] = decoded.realization_profile.colour_for(role)
        populated.add(class_id)
    if populated != set(range(5)):
        raise GenerativeTaskspaceReceiverError("realization profile did not cover the five-class semantic universe")
    return palette


def _iter_realized_y1(
    decoded: DecodedGenerativeCorrectionV1,
    operator: DisjointResizeOperator,
) -> Iterator[tuple[bytes, bytes, int, int]]:
    palette = _paint_palette(decoded)
    for local_pair in range(int(decoded.labels.shape[0])):
        scorer_y1 = np.ascontiguousarray(palette[decoded.labels[local_pair]])
        frame = realize_factor2_uint8_scorer_plane(operator, scorer_y1)
        proof = verify_factor2_uint8_scorer_plane(operator, frame, scorer_y1)
        if not proof.certified_exact or not proof.numerator_exact:
            raise GenerativeTaskspaceReceiverError("Y1 raw realization failed exact factor-2 parse-back")
        yield (
            frame.tobytes(order="C"),
            scorer_y1.tobytes(order="C"),
            proof.scorer_values,
            proof.numerator_equal_values,
        )


@contextmanager
def _exclusive_output_lock(output_path: Path) -> Iterator[None]:
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    with _open_regular_binary(
        lock_path,
        flags=os.O_RDWR,
        mode="r+b",
        label="receiver lock",
        create=True,
    ) as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GenerativeTaskspaceReceiverError("another G receiver owns this raw output") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _resume_state(
    *,
    archive_sha256: str,
    predictor_binding_sha256: str,
    completed_pairs: int,
    pair_count: int,
    frame_bytes: int,
    raw_prefix_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": FRAGMENT_ARCHIVE_SCHEMA,
        "archive_sha256": archive_sha256,
        "predictor_binding_sha256": predictor_binding_sha256,
        "completed_pairs": completed_pairs,
        "pair_count": pair_count,
        "frame_bytes": frame_bytes,
        "partial_bytes": completed_pairs * frame_bytes,
        "raw_prefix_sha256": raw_prefix_sha256,
        "output_role": "realized_frame1_Y1_raw_fragment",
        "research_only": True,
        "score_claim": False,
    }


def _load_resume_state(
    state_path: Path,
    partial_path: Path,
    *,
    archive_sha256: str,
    predictor_binding_sha256: str,
    pair_count: int,
    frame_bytes: int,
) -> int:
    partial_entry = _lstat_optional(partial_path, label="partial raw")
    state_entry = _lstat_optional(state_path, label="resume state")
    if partial_entry is None:
        if state_entry is not None:
            _require_regular_identity(
                state_path,
                _file_identity(state_entry),
                label="resume state",
            )
            state_path.unlink()
            _fsync_directory(state_path.parent)
        return 0
    if state_entry is None:
        with _open_regular_binary(
            partial_path,
            flags=os.O_RDWR,
            mode="r+b",
            label="partial raw",
        ) as handle:
            handle.truncate(0)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(partial_path.parent)
        return 0
    with _open_regular_binary(
        state_path,
        flags=os.O_RDONLY,
        mode="rb",
        label="resume state",
    ) as handle:
        state_raw = handle.read()
    _require_regular_identity(
        state_path,
        _file_identity(state_entry),
        label="resume state",
    )
    state = _decode_canonical_json(state_raw, label="G receiver resume checkpoint")
    if not isinstance(state, dict):
        raise GenerativeTaskspaceReceiverError("G receiver resume checkpoint is not an object")
    completed = state.get("completed_pairs")
    partial_bytes = state.get("partial_bytes")
    if (
        state.get("schema") != FRAGMENT_ARCHIVE_SCHEMA
        or state.get("archive_sha256") != archive_sha256
        or state.get("predictor_binding_sha256") != predictor_binding_sha256
        or state.get("pair_count") != pair_count
        or state.get("frame_bytes") != frame_bytes
        or state.get("output_role") != "realized_frame1_Y1_raw_fragment"
        or state.get("research_only") is not True
        or state.get("score_claim") is not False
        or type(completed) is not int
        or not 0 <= completed <= pair_count
        or type(partial_bytes) is not int
        or partial_bytes != completed * frame_bytes
        or partial_bytes > partial_entry.st_size
    ):
        raise GenerativeTaskspaceReceiverError("G receiver resume checkpoint differs from raw custody")
    if _sha256_file_prefix(partial_path, partial_bytes) != state.get("raw_prefix_sha256"):
        raise GenerativeTaskspaceReceiverError("G receiver resume hash differs from partial raw prefix")
    current_partial = _require_regular_identity(
        partial_path,
        _file_identity(partial_entry),
        label="partial raw",
    )
    if current_partial.st_size != partial_bytes:
        with _open_regular_binary(
            partial_path,
            flags=os.O_RDWR,
            mode="r+b",
            label="partial raw",
        ) as handle:
            handle.truncate(partial_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(partial_path.parent)
    return completed


def _decode_receipt(
    parsed: ParsedGenerativeTaskspaceFragmentArchiveV1,
    predictor_state: PredictorSemanticStateV1,
    decoded: DecodedGenerativeCorrectionV1,
    *,
    scorer_y1_sha256: str,
    raw_sha256: str,
    raw_bytes: int,
    camera_height: int,
    camera_width: int,
    scorer_values: int,
    numerator_values: int,
    runtime_files: tuple[ReceiverRuntimeFileV1, ...],
) -> GReceiverDecodeReceiptV1:
    g_section = parsed.counted.section(CountedTaskspaceRole.GENERATIVE_CORRECTION)
    changed_cells = int(np.count_nonzero(decoded.labels != predictor_state.labels))
    runtime_payloads = (
        *(
            _read_stable_regular_bytes(
                Path(module.__file__ or ""),
                label="receiver runtime source",
            )
            for module in (
                _generative_correction_module,
                _factorized_predictor_module,
                _direct_description_module,
                _uint8_lattice_module,
            )
        ),
        _read_stable_regular_bytes(Path(__file__), label="receiver runtime source"),
    )
    literal_sections_ge_32_absent = all(
        section.payload not in source
        for section in parsed.counted.sections
        if len(section.payload) >= _LITERAL_SCAN_MIN_SECTION_BYTES
        for source in runtime_payloads
    )
    if not literal_sections_ge_32_absent:
        raise GenerativeTaskspaceReceiverError(
            "a counted section selected by the >=32-byte heuristic appears literally in generic runtime source"
        )
    return GReceiverDecodeReceiptV1(
        schema=DECODE_RECEIPT_SCHEMA,
        archive_bytes=len(parsed.archive),
        archive_sha256=parsed.archive_sha256,
        archive_member_name=ARCHIVE_MEMBER_NAME,
        archive_member_bytes=len(parsed.counted.payload),
        archive_member_sha256=parsed.counted.payload_sha256,
        archive_member_crc32=parsed.member_crc32,
        rate_charged_archive_bytes=len(parsed.archive),
        generic_receiver_code_counted_bytes=0,
        g_section_offset=g_section.offset,
        g_section_bytes=g_section.byte_length,
        g_section_sha256=g_section.payload_sha256,
        g_section_crc32=g_section.crc32,
        non_g_sections=tuple(row.custody_dict() for row in parsed.counted.non_g_sections),
        non_g_manifest_sha256=parsed.counted.non_g_manifest_sha256,
        predictor_binding_sha256=predictor_state.binding_sha256,
        predictor_program_sha256=predictor_state.predictor_program_sha256,
        predictor_renderer_sha256=predictor_state.predictor_renderer_sha256,
        predictor_labels_sha256=predictor_state.labels_sha256,
        predictor_pose6_sha256=predictor_state.pose6_sha256,
        predictor_state_contract="PredictorSemanticStateV1_with_explicit_Pose6_transport.v1",
        pose6_serialized_in_g=False,
        levelset_latents_relabelled_as_pose6=False,
        ep725_frame1_predictor_compatible=False,
        counted_temporal_coordinate_adapter_owed=True,
        source_pair_ids=predictor_state.source_pair_ids,
        decoded_labels_sha256=_sha256(memoryview(decoded.labels).cast("B")),
        changed_semantic_cells=changed_cells,
        scorer_y1_bytes=predictor_state.pair_count * _SCORER_HEIGHT * _SCORER_WIDTH * _CHANNELS,
        scorer_y1_sha256=scorer_y1_sha256,
        realized_y1_raw_bytes=raw_bytes,
        realized_y1_raw_sha256=raw_sha256,
        scorer_height=_SCORER_HEIGHT,
        scorer_width=_SCORER_WIDTH,
        camera_height=camera_height,
        camera_width=camera_width,
        channels=_CHANNELS,
        factor2_scorer_values_verified=scorer_values,
        factor2_numerator_values_verified=numerator_values,
        runtime_files=runtime_files,
        runtime_manifest_sha256=_runtime_manifest_sha256(runtime_files),
        receiver_contract_id=RECEIVER_CONTRACT_ID,
        strict_g_parse_back=True,
        g_applied_to_predictor_state=True,
        realized_through_factor2_uint8=True,
        raw_output_materialized=True,
        raw_output_role="realized_frame1_Y1_raw_fragment",
        complete_chronological_pair_output=False,
        non_g_sections_consumed_by_g_receiver=False,
        repository_decoder_dependency=True,
        standalone_runtime_closure_bound=False,
        literal_counted_sections_ge_32_absent_from_runtime_sources=True,
        literal_scan_min_section_bytes=_LITERAL_SCAN_MIN_SECTION_BYTES,
        complete_data_in_code_lineage_audit=False,
        research_only=True,
        score_claim=False,
        originality_claim=False,
        promotion_eligible=False,
    )


def decode_generative_taskspace_fragment_to_y1_raw(
    archive: bytes,
    *,
    predictor_state: PredictorSemanticStateV1,
    output_path: Path,
    camera_height: int = 874,
    camera_width: int = 1164,
) -> GReceiverDecodeResultV1:
    """Consume G from a G/A[/T] fragment and atomically stream realized Y1 raw."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(output.parent)
    _lstat_optional(output, label="Y1 fragment output")
    _lstat_optional(output.with_name(f"{output.name}.partial"), label="partial raw")
    _lstat_optional(output.with_name(f"{output.name}.resume.json"), label="resume state")
    _lstat_optional(output.with_name(f"{output.name}.lock"), label="receiver lock")
    with _exclusive_output_lock(output):
        return _decode_generative_taskspace_fragment_to_y1_raw_locked(
            archive,
            predictor_state=predictor_state,
            output_path=output,
            camera_height=camera_height,
            camera_width=camera_width,
        )


def _decode_generative_taskspace_fragment_to_y1_raw_locked(
    archive: bytes,
    *,
    predictor_state: PredictorSemanticStateV1,
    output_path: Path,
    camera_height: int,
    camera_width: int,
) -> GReceiverDecodeResultV1:
    parsed = parse_generative_taskspace_receiver_fragment_archive(archive)
    runtime_before = _runtime_files()
    g_packet = parsed.counted.section(CountedTaskspaceRole.GENERATIVE_CORRECTION).payload
    try:
        decoded = apply_generative_taskspace_correction(g_packet, predictor_state=predictor_state)
    except GenerativeTaskspaceCorrectionError as exc:
        raise GenerativeTaskspaceReceiverError("literal counted G section failed strict parse/apply") from exc
    if decoded.correction_packet_sha256 != _sha256(g_packet):
        raise GenerativeTaskspaceReceiverError("decoded G state is not bound to literal counted section bytes")
    _exact_positive_int(predictor_state.pair_count, "predictor pair_count", maximum=_MAX_PAIR_COUNT)
    try:
        operator = DisjointResizeOperator.build(
            camera_h=_exact_positive_int(camera_height, "camera_height", maximum=4096),
            camera_w=_exact_positive_int(camera_width, "camera_width", maximum=4096),
            scorer_h=_SCORER_HEIGHT,
            scorer_w=_SCORER_WIDTH,
        )
    except Uint8LatticeError as exc:
        raise GenerativeTaskspaceReceiverError("receiver geometry is not a certified factor-2 lattice") from exc
    frame_bytes = camera_height * camera_width * _CHANNELS
    expected_bytes = predictor_state.pair_count * frame_bytes
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    state_path = output_path.with_name(f"{output_path.name}.resume.json")

    output_entry = _lstat_optional(output_path, label="Y1 fragment output")
    _lstat_optional(partial_path, label="partial raw")
    _lstat_optional(state_path, label="resume state")
    if output_entry is not None:
        if output_entry.st_size != expected_bytes:
            raise GenerativeTaskspaceReceiverError("existing Y1 raw output has incompatible size or type")
        completed_pairs = predictor_state.pair_count
        mode = "verify"
    else:
        completed_pairs = _load_resume_state(
            state_path,
            partial_path,
            archive_sha256=parsed.archive_sha256,
            predictor_binding_sha256=predictor_state.binding_sha256,
            pair_count=predictor_state.pair_count,
            frame_bytes=frame_bytes,
        )
        partial_entry = _lstat_optional(partial_path, label="partial raw")
        missing_bytes = expected_bytes - partial_entry.st_size if partial_entry is not None else expected_bytes
        if shutil.disk_usage(output_path.parent).free < missing_bytes + (1 << 20):
            raise GenerativeTaskspaceReceiverError("storage preflight refused insufficient raw-output tier space")
        mode = "resume"

    raw_digest = hashlib.sha256()
    scorer_digest = hashlib.sha256()
    scorer_values = 0
    numerator_values = 0
    if mode == "verify":
        handle = _open_regular_binary(
            output_path,
            flags=os.O_RDONLY,
            mode="rb",
            label="Y1 fragment output",
        )
    else:
        partial_entry = _lstat_optional(partial_path, label="partial raw")
        handle = _open_regular_binary(
            partial_path,
            flags=os.O_RDWR,
            mode="r+b",
            label="partial raw",
            create=partial_entry is None,
            exclusive_create=partial_entry is None,
        )
    opened_identity = _file_identity(os.fstat(handle.fileno()))
    try:
        handle.seek(0)
        for pair_index, (pair_raw, scorer_raw, pair_scorer_values, pair_numerator_values) in enumerate(
            _iter_realized_y1(decoded, operator)
        ):
            if len(pair_raw) != frame_bytes:
                raise GenerativeTaskspaceReceiverError("realized Y1 pair byte count drift")
            raw_digest.update(pair_raw)
            scorer_digest.update(scorer_raw)
            scorer_values += pair_scorer_values
            numerator_values += pair_numerator_values
            if mode == "verify" or pair_index < completed_pairs:
                if handle.read(frame_bytes) != pair_raw:
                    raise GenerativeTaskspaceReceiverError("materialized Y1 raw differs from deterministic replay")
            else:
                handle.seek(0, os.SEEK_END)
                handle.write(pair_raw)
                handle.flush()
                os.fsync(handle.fileno())
                _atomic_write(
                    state_path,
                    _canonical_json(
                        _resume_state(
                            archive_sha256=parsed.archive_sha256,
                            predictor_binding_sha256=predictor_state.binding_sha256,
                            completed_pairs=pair_index + 1,
                            pair_count=predictor_state.pair_count,
                            frame_bytes=frame_bytes,
                            raw_prefix_sha256=raw_digest.hexdigest(),
                        )
                    ),
                )
        if handle.read(1):
            raise GenerativeTaskspaceReceiverError("materialized Y1 raw has trailing bytes")
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    if mode != "verify":
        completed_partial = _require_regular_identity(
            partial_path,
            opened_identity,
            label="partial raw",
        )
        if completed_partial.st_size != expected_bytes:
            raise GenerativeTaskspaceReceiverError("completed Y1 raw byte count drift")
        if _lstat_optional(output_path, label="Y1 fragment output") is not None:
            raise GenerativeTaskspaceReceiverError("Y1 fragment output appeared before atomic installation")
        os.replace(partial_path, output_path)
        _fsync_directory(output_path.parent)
    installed_output = _require_regular_identity(
        output_path,
        opened_identity,
        label="Y1 fragment output",
    )
    if installed_output.st_size != expected_bytes or _sha256_file(output_path) != raw_digest.hexdigest():
        raise GenerativeTaskspaceReceiverError("installed Y1 raw differs from deterministic receiver output")
    runtime_after = _runtime_files()
    if runtime_after != runtime_before:
        raise GenerativeTaskspaceReceiverError("receiver runtime source changed during decode")
    receipt = _decode_receipt(
        parsed,
        predictor_state,
        decoded,
        scorer_y1_sha256=scorer_digest.hexdigest(),
        raw_sha256=raw_digest.hexdigest(),
        raw_bytes=expected_bytes,
        camera_height=camera_height,
        camera_width=camera_width,
        scorer_values=scorer_values,
        numerator_values=numerator_values,
        runtime_files=runtime_before,
    )
    return GReceiverDecodeResultV1(decoded=decoded, receipt=receipt)


def prove_generative_taskspace_receiver_consumption(
    baseline_archive: bytes,
    counterfactual_archive: bytes,
    *,
    predictor_state: PredictorSemanticStateV1,
    baseline_output_path: Path,
    counterfactual_output_path: Path,
) -> GReceiverConsumptionProofV1:
    """Derive a matched G-only causal proof from actual receiver executions."""

    baseline_parsed = parse_generative_taskspace_receiver_fragment_archive(baseline_archive)
    counterfactual_parsed = parse_generative_taskspace_receiver_fragment_archive(counterfactual_archive)
    baseline_roles = tuple(row.role for row in baseline_parsed.counted.sections)
    counterfactual_roles = tuple(row.role for row in counterfactual_parsed.counted.sections)
    if baseline_roles != counterfactual_roles:
        raise GenerativeTaskspaceReceiverError("G counterfactual changed counted section roles")
    baseline_g = baseline_parsed.counted.section(CountedTaskspaceRole.GENERATIVE_CORRECTION)
    counterfactual_g = counterfactual_parsed.counted.section(CountedTaskspaceRole.GENERATIVE_CORRECTION)
    if baseline_g.payload == counterfactual_g.payload:
        raise GenerativeTaskspaceReceiverError("G counterfactual did not change literal counted G bytes")
    if tuple(row.payload for row in baseline_parsed.counted.non_g_sections) != tuple(
        row.payload for row in counterfactual_parsed.counted.non_g_sections
    ):
        raise GenerativeTaskspaceReceiverError("G counterfactual changed a non-G counted section")

    baseline = decode_generative_taskspace_fragment_to_y1_raw(
        baseline_archive,
        predictor_state=predictor_state,
        output_path=baseline_output_path,
    )
    baseline_replay = decode_generative_taskspace_fragment_to_y1_raw(
        baseline_archive,
        predictor_state=predictor_state,
        output_path=baseline_output_path,
    )
    counterfactual = decode_generative_taskspace_fragment_to_y1_raw(
        counterfactual_archive,
        predictor_state=predictor_state,
        output_path=counterfactual_output_path,
    )
    if baseline.receipt != baseline_replay.receipt or not np.array_equal(
        baseline.decoded.labels, baseline_replay.decoded.labels
    ):
        raise GenerativeTaskspaceReceiverError("identical fragment double-decode was nondeterministic")

    same_runtime = baseline.receipt.runtime_manifest_sha256 == counterfactual.receipt.runtime_manifest_sha256
    same_predictor = baseline.receipt.predictor_binding_sha256 == counterfactual.receipt.predictor_binding_sha256
    unchanged_non_g = baseline.receipt.non_g_manifest_sha256 == counterfactual.receipt.non_g_manifest_sha256
    decoded_changed = baseline.receipt.decoded_labels_sha256 != counterfactual.receipt.decoded_labels_sha256
    scorer_changed = baseline.receipt.scorer_y1_sha256 != counterfactual.receipt.scorer_y1_sha256
    raw_changed = baseline.receipt.realized_y1_raw_sha256 != counterfactual.receipt.realized_y1_raw_sha256
    if not same_runtime:
        raise GenerativeTaskspaceReceiverError("G counterfactual did not retain the same receiver runtime")
    if not same_predictor:
        raise GenerativeTaskspaceReceiverError("G counterfactual did not retain the same predictor state")
    if not unchanged_non_g:
        raise GenerativeTaskspaceReceiverError("G counterfactual did not retain non-G section custody")
    if not decoded_changed:
        raise GenerativeTaskspaceReceiverError("G counterfactual did not change decoded semantic state")
    if not scorer_changed:
        raise GenerativeTaskspaceReceiverError("G counterfactual did not change scorer-grid Y1 bytes")
    if not raw_changed:
        raise GenerativeTaskspaceReceiverError("G counterfactual did not change realized raw output")
    return GReceiverConsumptionProofV1(
        baseline=baseline.receipt,
        counterfactual=counterfactual.receipt,
        baseline_replay_identical=True,
        same_runtime_manifest=True,
        same_predictor_binding=True,
        target_g_section_only_changed=True,
        unchanged_non_g_sections=True,
        decoded_state_changed=True,
        scorer_y1_changed=True,
        realized_raw_output_changed=True,
        archive_member_to_strict_g_parse_apply_to_raw_causal=True,
    )


def write_generative_taskspace_consumption_proof(
    proof: GReceiverConsumptionProofV1,
    output_path: Path,
    *,
    baseline_archive: bytes,
    counterfactual_archive: bytes,
    predictor_state: PredictorSemanticStateV1,
    scratch_root: Path,
) -> GReceiverConsumptionProofV1:
    """Fresh-replay a claim, then atomically persist only the verified proof bytes."""

    if not isinstance(proof, GReceiverConsumptionProofV1):
        raise GenerativeTaskspaceReceiverError("consumption proof claim must be the typed object")
    verified = verify_generative_taskspace_consumption_proof(
        proof.to_bytes(),
        baseline_archive,
        counterfactual_archive,
        predictor_state=predictor_state,
        scratch_root=scratch_root,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(path.parent)
    _atomic_write(path, verified.to_bytes())
    return verified


def verify_generative_taskspace_consumption_proof(
    proof_payload: bytes,
    baseline_archive: bytes,
    counterfactual_archive: bytes,
    *,
    predictor_state: PredictorSemanticStateV1,
    scratch_root: Path,
) -> GReceiverConsumptionProofV1:
    """Re-execute both archives; never trust hashes or booleans from the receipt."""

    value = _decode_canonical_json(proof_payload, label="G receiver consumption proof")
    if not isinstance(value, dict) or set(value) != {"schema", "body", "body_sha256"}:
        raise GenerativeTaskspaceReceiverError("G receiver consumption proof envelope has field drift")
    body = value["body"]
    if (
        value["schema"] != CONSUMPTION_PROOF_ENVELOPE_SCHEMA
        or not isinstance(body, dict)
        or body.get("schema") != CONSUMPTION_PROOF_SCHEMA
        or value["body_sha256"] != _sha256(_canonical_json(body))
    ):
        raise GenerativeTaskspaceReceiverError("G receiver consumption proof body binding is invalid")
    scratch = Path(scratch_root)
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="g_receiver_verify_", dir=scratch) as directory:
        root = Path(directory)
        recomputed = prove_generative_taskspace_receiver_consumption(
            baseline_archive,
            counterfactual_archive,
            predictor_state=predictor_state,
            baseline_output_path=root / "baseline_y1.raw",
            counterfactual_output_path=root / "counterfactual_y1.raw",
        )
    if recomputed.to_bytes() != proof_payload:
        raise GenerativeTaskspaceReceiverError("G receiver consumption proof differs from fresh behavioral replay")
    return recomputed


__all__ = [
    "ARCHIVE_MEMBER_NAME",
    "CONSUMPTION_PROOF_ENVELOPE_SCHEMA",
    "CONSUMPTION_PROOF_SCHEMA",
    "COUNTED_PAYLOAD_SCHEMA",
    "DECODE_RECEIPT_SCHEMA",
    "FRAGMENT_ARCHIVE_SCHEMA",
    "RECEIVER_CONTRACT_ID",
    "CountedTaskspaceRole",
    "CountedTaskspaceSectionV1",
    "GReceiverConsumptionProofV1",
    "GReceiverDecodeReceiptV1",
    "GReceiverDecodeResultV1",
    "GenerativeTaskspaceReceiverError",
    "ParsedCountedTaskspacePayloadV1",
    "ParsedGenerativeTaskspaceFragmentArchiveV1",
    "ReceiverRuntimeFileV1",
    "build_g_only_counterfactual_fragment_archive",
    "build_generative_taskspace_receiver_fragment_archive",
    "decode_generative_taskspace_fragment_to_y1_raw",
    "encode_counted_taskspace_payload",
    "parse_counted_taskspace_payload",
    "parse_generative_taskspace_receiver_fragment_archive",
    "prove_generative_taskspace_receiver_consumption",
    "verify_generative_taskspace_consumption_proof",
    "write_generative_taskspace_consumption_proof",
]
