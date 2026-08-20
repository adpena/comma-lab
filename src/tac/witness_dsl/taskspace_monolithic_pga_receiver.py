# SPDX-License-Identifier: MIT
"""Strict G3 ``P/G/A[/T]`` member grammar and bounded ep725 receiver audit.

The counted object is one monolithic member inside the canonical outer archive
owned by :mod:`taskspace_outer_archive_codec`.  This module owns the inner
section directory only.  Its exact order is ``PREDICTOR``,
``GENERATIVE_CORRECTION``, ``COUPLED_PREIMAGE`` and, optionally,
``TERMINAL_QUOTIENT``.  Every offset, length, SHA-256 and CRC-32 comes from that
directory; no caller manifest or decoded state is accepted as slicing
authority.

The ep725 adapter exposes exact ephemeral decoder frames and G exposes an exact
semantic ownership mask.  The predictor-preserving overlay therefore closes
P-to-G at the camera surface while proving byte/numerator identity everywhere
outside G-owned supports.  The counted predictor-preserving A3 grammar consumes
that exact P0/corrected-Y1 surface and emits chronological ``[Y0,Y1]`` bytes;
this outer receiver emits only their hashes and never persists dense frames.

The inherited A2 grammar remains accepted only for an explicit diagnostic: it
conditions on ``decoded_g.paint_rgb()``, a full flat-palette repaint rather
than predictor-preserving Y1, and therefore raises the machine-typed
``A_SIGNAL_LOSS_FULL_REPAINT`` blocker.  Neither path invokes a scorer or makes
a score, candidate, originality, promotion, full-A3, or target-closure claim.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Final, Literal

import numpy as np

import tac.witness_dsl.coupled_preimage_program as _a
from tac.witness_dsl.coupled_preimage_program import (
    CoupledPreimageMode,
    CoupledPreimageProgramError,
    CoupledPreimageProgramV1,
    parse_coupled_preimage_program,
)
from tac.witness_dsl.ep725_levelset_predictor_adapter import (
    EP725_MEMBER_BYTES,
    EP725_MEMBER_SHA256,
    Ep725BoundedDecodeV2,
    Ep725CountedMemberCausalDecodeReceiptV3,
    Ep725CountedMemberCausalSurfaceV3,
    Ep725EphemeralRuntimeSurfaceV2,
    Ep725LevelsetPredictorAdapterError,
    decode_ep725_counted_member_ephemeral_surface,
    parse_ep725_counted_member_causal_decode_receipt,
)
from tac.witness_dsl.generative_taskspace_correction import (
    DecodedGenerativeCorrectionV1,
    GenerativeTaskspaceCorrectionError,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    PACKET_MAGIC as A3_PACKET_MAGIC,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    DecodedPredictorPreservingA3V1,
    PredictorCameraPairSurfaceV1,
    PredictorPreservingA3Error,
    PredictorPreservingA3ReceiptV1,
    decode_predictor_preserving_a3_packet,
    parse_predictor_preserving_a3_receipt,
)
from tac.witness_dsl.predictor_preserving_taskspace_overlay import (
    PredictorPreservingOverlayError,
    PredictorPreservingOverlayReceiptV1,
    PredictorPreservingOverlayResultV1,
    overlay_g_on_predictor_camera_y1,
    parse_predictor_preserving_overlay_receipt,
)
from tac.witness_dsl.taskspace_outer_archive_codec import (
    DEFAULT_MAX_MEMBER_BYTES,
    OuterArchiveEncoding,
    ParsedTaskspaceOuterArchive,
    TaskspaceOuterArchiveBuild,
    TaskspaceOuterArchiveError,
    build_taskspace_outer_archive,
    parse_taskspace_outer_archive,
)
from tac.witness_dsl.taskspace_pass_conditional_a import (
    PACKET_MAGIC as PASS_CONDITIONAL_A_PACKET_MAGIC,
)
from tac.witness_dsl.taskspace_pass_conditional_a import (
    DecodedPassConditionalAV1,
    PassConditionalAError,
    PassConditionalAReceiptV1,
    decode_pass_conditional_a_packet,
    parse_pass_conditional_a_receipt,
)
from tac.witness_dsl.taskspace_pass_semantic_g import (
    ENVELOPE_MAGIC as PASS_SEMANTIC_G_ENVELOPE_MAGIC,
)
from tac.witness_dsl.taskspace_pass_semantic_g import (
    DecodedPassSemanticGEnvelopeV1,
    PassSemanticGEnvelopeReceiptV1,
    PassSemanticGError,
    decode_pass_semantic_g_envelope,
    parse_pass_semantic_g_envelope_receipt,
)
from tac.witness_dsl.taskspace_post_g8_conditional_a import (
    PACKET_MAGIC as POST_G8_A_PACKET_MAGIC,
)
from tac.witness_dsl.taskspace_post_g8_conditional_a import (
    DecodedPostG8ConditionalAV1,
    PostG8ConditionalAError,
    PostG8ConditionalAReceiptV1,
    decode_post_g8_conditional_a_packet,
    parse_post_g8_conditional_a_receipt,
)
from tac.witness_dsl.taskspace_predictor_v2_consumer_seam import (
    GCorrectionOwnershipV2,
    TaskspacePredictorV2ConsumerSeamError,
    apply_generative_taskspace_correction_v2,
    derive_g_correction_ownership_v2,
    parse_generative_taskspace_correction_v2,
)
from tac.witness_dsl.taskspace_same_class_realization_repair import (
    COMPOSITE_G_SECTION_MAGIC,
    DecodedSameClassRealizationRepairPGAIntegrationV1,
    SameClassRealizationRepairError,
    SameClassRealizationRepairReceiptV1,
    decode_same_class_realization_repair_from_pga_sections,
    parse_same_class_realization_repair_receipt,
)

MEMBER_SCHEMA: Final = "tac.taskspace_monolithic_pga_member.v1"
BLOCKER_RECEIPT_SCHEMA: Final = "tac.taskspace_monolithic_pga_receiver_blocker_receipt.v1"
SUCCESS_RECEIPT_SCHEMA: Final = "tac.taskspace_monolithic_pga_receiver_receipt.v1"
RECEIVER_CONTRACT_ID: Final = "tac.ep725_p_g_owned_overlay_then_a_factor2_chronological.v1"
G8_SUCCESS_RECEIPT_SCHEMA: Final = "tac.taskspace_monolithic_pga_g8_receiver_receipt.v1"
G8_RECEIVER_CONTRACT_ID: Final = "tac.ep725_p_semantic_g_g8_then_post_g8_a_factor2.v1"
PASS_SUCCESS_RECEIPT_SCHEMA: Final = "tac.taskspace_monolithic_pga_pass_receiver_receipt.v1"
PASS_RECEIVER_CONTRACT_ID: Final = "tac.ep725_p_pass_semantic_g_optional_g8_then_conditional_a_factor2.v1"
MEMBER_MAGIC: Final = b"TACPGA3\x00"
MEMBER_VERSION: Final = 1
MAX_BOUNDED_PAIRS: Final = 2
SCORER_HEIGHT: Final = 384
SCORER_WIDTH: Final = 512
CHANNELS: Final = 3

_MEMBER_PREFIX = struct.Struct(">8sBBH")
_SECTION_DESCRIPTOR = struct.Struct(">B3xQQI32s")
_MAX_SECTION_BYTES: Final = 1 << 31
_MAX_MEMBER_BYTES: Final = DEFAULT_MAX_MEMBER_BYTES
_G_PACKET_MAGIC: Final = b"TACG1C\x00\x00"
_P_PACKET_MAGIC: Final = b"LVLS1\x00"
_SIGNAL_LOSS_DETAIL: Final = (
    "P-to-G is predictor-preserving at the exact camera/R surface, but counted A is bound to inherited "
    "full-frame paint_rgb Y1 rather than the predictor-preserving corrected Y1"
)


class TaskspaceMonolithicPGARole(StrEnum):
    """Closed role universe for the one counted inner member."""

    PREDICTOR = "predictor"
    GENERATIVE_CORRECTION = "generative_correction"
    COUPLED_PREIMAGE = "coupled_preimage"
    TERMINAL_QUOTIENT = "terminal_quotient"


_ROLE_CODE: Final = {
    # Preserve the already-landed G2 role codes and add P without aliases.
    TaskspaceMonolithicPGARole.GENERATIVE_CORRECTION: 1,
    TaskspaceMonolithicPGARole.COUPLED_PREIMAGE: 2,
    TaskspaceMonolithicPGARole.TERMINAL_QUOTIENT: 3,
    TaskspaceMonolithicPGARole.PREDICTOR: 4,
}
_ROLE_BY_CODE: Final = {value: key for key, value in _ROLE_CODE.items()}
REQUIRED_ROLE_ORDER: Final = (
    TaskspaceMonolithicPGARole.PREDICTOR,
    TaskspaceMonolithicPGARole.GENERATIVE_CORRECTION,
    TaskspaceMonolithicPGARole.COUPLED_PREIMAGE,
)
OPTIONAL_ROLE_ORDER: Final = (*REQUIRED_ROLE_ORDER, TaskspaceMonolithicPGARole.TERMINAL_QUOTIENT)
_ADMITTED_ROLE_ORDERS: Final = (REQUIRED_ROLE_ORDER, OPTIONAL_ROLE_ORDER)


class TaskspaceMonolithicPGAError(ValueError):
    """Malformed member/archive bytes or failed exact P/G/A binding."""


class TaskspaceMonolithicPGABlockerCode(StrEnum):
    """Machine-typed reasons why bounded semantic output is not emitted."""

    A_SIGNAL_LOSS_FULL_REPAINT = "A_SIGNAL_LOSS_FULL_REPAINT"
    TERMINAL_QUOTIENT_CONSUMER_OWED = "TERMINAL_QUOTIENT_CONSUMER_OWED"


class TaskspaceMonolithicPGAReceiverBlocker(TaskspaceMonolithicPGAError):
    """A structurally valid object reached a known, typed semantic blocker."""

    def __init__(
        self,
        code: TaskspaceMonolithicPGABlockerCode,
        detail: str,
        *,
        receipt: TaskspaceMonolithicPGABlockerReceiptV1 | None = None,
    ) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail
        self.receipt = receipt


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TaskspaceMonolithicPGAError(f"{label} must be canonical lowercase SHA-256 hex")
    return value


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
        raise TaskspaceMonolithicPGAError("receipt is not finite canonical ASCII JSON") from exc


def _decode_canonical_json(payload: bytes, *, label: str) -> Any:
    def unique_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TaskspaceMonolithicPGAError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    if type(payload) is not bytes or not payload:
        raise TaskspaceMonolithicPGAError(f"{label} must be nonempty exact bytes")
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                TaskspaceMonolithicPGAError(f"{label} contains non-finite constant {token}")
            ),
        )
    except TaskspaceMonolithicPGAError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TaskspaceMonolithicPGAError(f"{label} is not strict ASCII JSON") from exc
    if _canonical_json(value) != payload:
        raise TaskspaceMonolithicPGAError(f"{label} is not byte-canonical JSON")
    return value


@dataclass(frozen=True, slots=True)
class TaskspaceMonolithicPGASectionV1:
    """One directory-owned exact byte range in the counted member."""

    role: TaskspaceMonolithicPGARole
    offset: int
    payload: bytes
    crc32: int

    def __post_init__(self) -> None:
        if type(self.role) is not TaskspaceMonolithicPGARole:
            raise TaskspaceMonolithicPGAError("section role is outside the closed P/G/A[/T] universe")
        if type(self.offset) is not int or self.offset < 0:
            raise TaskspaceMonolithicPGAError("section offset must be an exact nonnegative integer")
        if type(self.payload) is not bytes or not 0 < len(self.payload) <= _MAX_SECTION_BYTES:
            raise TaskspaceMonolithicPGAError("section payload must be bounded nonempty immutable bytes")
        expected_crc = zlib.crc32(self.payload) & 0xFFFFFFFF
        if type(self.crc32) is not int or self.crc32 != expected_crc:
            raise TaskspaceMonolithicPGAError("section CRC-32 differs from exact payload bytes")
        if zipfile.is_zipfile(io.BytesIO(self.payload)):
            raise TaskspaceMonolithicPGAError("nested ZIP sections are forbidden")

    @property
    def byte_length(self) -> int:
        return len(self.payload)

    @property
    def stop(self) -> int:
        return self.offset + self.byte_length

    @property
    def payload_sha256(self) -> str:
        return _sha256(self.payload)

    def custody_dict(self) -> dict[str, Any]:
        return {
            "byte_length": self.byte_length,
            "crc32": self.crc32,
            "offset": self.offset,
            "payload_sha256": self.payload_sha256,
            "role": self.role.value,
        }


@dataclass(frozen=True, slots=True)
class ParsedTaskspaceMonolithicPGAMemberV1:
    """Strictly parsed member whose directory is the sole slicing authority."""

    member_bytes: bytes
    sections: tuple[TaskspaceMonolithicPGASectionV1, ...]

    def __post_init__(self) -> None:
        if type(self.member_bytes) is not bytes or not self.member_bytes:
            raise TaskspaceMonolithicPGAError("parsed member must retain exact nonempty bytes")
        roles = tuple(row.role for row in self.sections)
        if roles not in _ADMITTED_ROLE_ORDERS:
            raise TaskspaceMonolithicPGAError("parsed roles are not exact P -> G -> A -> optional T")

    @property
    def member_sha256(self) -> str:
        return _sha256(self.member_bytes)

    @property
    def descriptor_manifest_sha256(self) -> str:
        return _sha256(_canonical_json([row.custody_dict() for row in self.sections]))

    def section(self, role: TaskspaceMonolithicPGARole) -> TaskspaceMonolithicPGASectionV1:
        matches = tuple(row for row in self.sections if row.role is role)
        if len(matches) != 1:
            raise TaskspaceMonolithicPGAError(f"member has no unique {role.value} section")
        return matches[0]


def encode_taskspace_monolithic_pga_member(
    predictor_packet: bytes,
    generative_correction_packet: bytes,
    coupled_preimage_packet: bytes,
    *,
    terminal_quotient_packet: bytes | None = None,
) -> bytes:
    """Encode one canonical P/G/A[/T] member and strict-parse it back."""

    rows: list[tuple[TaskspaceMonolithicPGARole, bytes]] = [
        (TaskspaceMonolithicPGARole.PREDICTOR, predictor_packet),
        (TaskspaceMonolithicPGARole.GENERATIVE_CORRECTION, generative_correction_packet),
        (TaskspaceMonolithicPGARole.COUPLED_PREIMAGE, coupled_preimage_packet),
    ]
    if terminal_quotient_packet is not None:
        rows.append((TaskspaceMonolithicPGARole.TERMINAL_QUOTIENT, terminal_quotient_packet))
    for role, payload in rows:
        if type(payload) is not bytes or not 0 < len(payload) <= _MAX_SECTION_BYTES:
            raise TaskspaceMonolithicPGAError(f"{role.value} must be bounded nonempty immutable bytes")
        if zipfile.is_zipfile(io.BytesIO(payload)):
            raise TaskspaceMonolithicPGAError(f"{role.value} may not contain a nested ZIP")

    prefix = _MEMBER_PREFIX.pack(MEMBER_MAGIC, MEMBER_VERSION, len(rows), 0)
    directory_bytes = len(rows) * _SECTION_DESCRIPTOR.size
    cursor = len(prefix) + directory_bytes
    descriptors: list[bytes] = []
    for role, payload in rows:
        descriptors.append(
            _SECTION_DESCRIPTOR.pack(
                _ROLE_CODE[role],
                cursor,
                len(payload),
                zlib.crc32(payload) & 0xFFFFFFFF,
                bytes.fromhex(_sha256(payload)),
            )
        )
        cursor += len(payload)
    member = prefix + b"".join(descriptors) + b"".join(payload for _role, payload in rows)
    if len(member) > _MAX_MEMBER_BYTES:
        raise TaskspaceMonolithicPGAError("monolithic member exceeds the configured receiver bound")
    parsed = parse_taskspace_monolithic_pga_member(member)
    if tuple(row.payload for row in parsed.sections) != tuple(payload for _role, payload in rows):
        raise TaskspaceMonolithicPGAError("member changed on strict parse-back")
    return member


def parse_taskspace_monolithic_pga_member(
    member: bytes,
    *,
    max_member_bytes: int = _MAX_MEMBER_BYTES,
) -> ParsedTaskspaceMonolithicPGAMemberV1:
    """Parse exact offsets/hashes/CRCs and require exact stream EOF."""

    if type(member) is not bytes or not member:
        raise TaskspaceMonolithicPGAError("monolithic member must be nonempty immutable bytes")
    if type(max_member_bytes) is not int or isinstance(max_member_bytes, bool) or max_member_bytes < 1:
        raise TaskspaceMonolithicPGAError("max_member_bytes must be an exact positive integer")
    if len(member) > max_member_bytes:
        raise TaskspaceMonolithicPGAError("monolithic member exceeds the configured receiver bound")
    if len(member) < _MEMBER_PREFIX.size:
        raise TaskspaceMonolithicPGAError("monolithic member prefix is truncated")
    magic, version, section_count, flags = _MEMBER_PREFIX.unpack_from(member)
    if magic != MEMBER_MAGIC or version != MEMBER_VERSION or flags != 0:
        raise TaskspaceMonolithicPGAError("monolithic member magic/version/flags are invalid")
    if section_count not in {3, 4}:
        raise TaskspaceMonolithicPGAError("monolithic member must contain exact P/G/A and optional T")
    directory_stop = _MEMBER_PREFIX.size + section_count * _SECTION_DESCRIPTOR.size
    if directory_stop > len(member):
        raise TaskspaceMonolithicPGAError("monolithic section directory is truncated")

    descriptors: list[tuple[TaskspaceMonolithicPGARole, int, int, int, str]] = []
    for index in range(section_count):
        descriptor_offset = _MEMBER_PREFIX.size + index * _SECTION_DESCRIPTOR.size
        if member[descriptor_offset + 1 : descriptor_offset + 4] != b"\x00\x00\x00":
            raise TaskspaceMonolithicPGAError("monolithic directory descriptor reserved bytes are nonzero")
        role_code, offset, length, checksum, digest = _SECTION_DESCRIPTOR.unpack_from(member, descriptor_offset)
        role = _ROLE_BY_CODE.get(role_code)
        if role is None:
            raise TaskspaceMonolithicPGAError("monolithic directory contains an unknown section role")
        if not 0 < length <= _MAX_SECTION_BYTES:
            raise TaskspaceMonolithicPGAError("monolithic directory contains an invalid section length")
        descriptors.append((role, offset, length, checksum, digest.hex()))
    roles = tuple(row[0] for row in descriptors)
    if roles not in _ADMITTED_ROLE_ORDERS:
        raise TaskspaceMonolithicPGAError("monolithic section order is not P -> G -> A -> optional T")

    cursor = directory_stop
    sections: list[TaskspaceMonolithicPGASectionV1] = []
    for role, offset, length, checksum, digest in descriptors:
        if offset != cursor:
            raise TaskspaceMonolithicPGAError("directory offsets are not exact contiguous slicing authority")
        stop = offset + length
        if stop > len(member):
            raise TaskspaceMonolithicPGAError("directory-owned section range is truncated")
        payload = member[offset:stop]
        if _sha256(payload) != digest:
            raise TaskspaceMonolithicPGAError("directory-owned section SHA-256 mismatch")
        if zlib.crc32(payload) & 0xFFFFFFFF != checksum:
            raise TaskspaceMonolithicPGAError("directory-owned section CRC-32 mismatch")
        sections.append(TaskspaceMonolithicPGASectionV1(role, offset, payload, checksum))
        cursor = stop
    if cursor != len(member):
        raise TaskspaceMonolithicPGAError("monolithic member has trailing or unconsumed bytes")
    return ParsedTaskspaceMonolithicPGAMemberV1(member_bytes=member, sections=tuple(sections))


def build_taskspace_monolithic_pga_archive(
    predictor_packet: bytes,
    generative_correction_packet: bytes,
    coupled_preimage_packet: bytes,
    *,
    terminal_quotient_packet: bytes | None = None,
    max_member_bytes: int = _MAX_MEMBER_BYTES,
) -> TaskspaceOuterArchiveBuild:
    """Build the inner member, then delegate all ZIP pricing to the outer codec."""

    member = encode_taskspace_monolithic_pga_member(
        predictor_packet,
        generative_correction_packet,
        coupled_preimage_packet,
        terminal_quotient_packet=terminal_quotient_packet,
    )
    try:
        return build_taskspace_outer_archive(member, max_member_bytes=max_member_bytes)
    except TaskspaceOuterArchiveError as exc:
        raise TaskspaceMonolithicPGAError("outer archive codec refused the exact monolithic member") from exc


@dataclass(frozen=True, slots=True)
class TaskspaceMonolithicPGASectionReceiptV1:
    role: TaskspaceMonolithicPGARole
    offset: int
    byte_length: int
    payload_sha256: str
    crc32: int

    def __post_init__(self) -> None:
        if type(self.role) is not TaskspaceMonolithicPGARole:
            raise TaskspaceMonolithicPGAError("receipt section role is invalid")
        if type(self.offset) is not int or self.offset < 0:
            raise TaskspaceMonolithicPGAError("receipt section offset is invalid")
        if type(self.byte_length) is not int or self.byte_length < 1:
            raise TaskspaceMonolithicPGAError("receipt section byte length is invalid")
        _require_sha256(self.payload_sha256, label="receipt section payload_sha256")
        if type(self.crc32) is not int or not 0 <= self.crc32 <= 0xFFFFFFFF:
            raise TaskspaceMonolithicPGAError("receipt section CRC-32 is invalid")

    @property
    def stop(self) -> int:
        return self.offset + self.byte_length

    def as_dict(self) -> dict[str, Any]:
        return {
            "byte_length": self.byte_length,
            "crc32": self.crc32,
            "offset": self.offset,
            "payload_sha256": self.payload_sha256,
            "role": self.role.value,
        }


@dataclass(frozen=True, slots=True)
class TaskspaceMonolithicPGABlockerReceiptV1:
    """Canonical receipt for all work completed before the signal-loss wall."""

    schema: Literal["tac.taskspace_monolithic_pga_receiver_blocker_receipt.v1"]
    receiver_contract_id: Literal["tac.ep725_p_g_owned_overlay_then_a_factor2_chronological.v1"]
    blocker_code: TaskspaceMonolithicPGABlockerCode
    blocker_detail: str
    outer_encoding: OuterArchiveEncoding
    archive_bytes: int
    archive_sha256: str
    compressed_member_bytes: int
    member_bytes: int
    member_sha256: str
    member_crc32: int
    sections: tuple[
        TaskspaceMonolithicPGASectionReceiptV1,
        TaskspaceMonolithicPGASectionReceiptV1,
        TaskspaceMonolithicPGASectionReceiptV1,
    ]
    section_descriptor_manifest_sha256: str
    source_pair_ids: tuple[int, ...]
    predictor_state_binding_sha256: str
    predictor_semantic_binding_sha256: str
    predictor_source_archive_sha256: str
    predictor_source_runtime_sha256: str
    predictor_bounded_receipt_sha256: str
    predictor_causal_decode_receipt: Ep725CountedMemberCausalDecodeReceiptV3
    predictor_causal_decode_receipt_sha256: str
    predictor_chronological_camera_frame_sha256: tuple[str, ...]
    predictor_frame1_camera_sha256: tuple[str, ...]
    decoded_g_labels_sha256: str
    g_owned_mask_sha256: str
    g_owned_cells: int
    g_transport_kind: Literal["NONE"]
    a_mode: CoupledPreimageMode
    a_source_binding_sha256: str
    a_bound_exact_y1_sha256: str
    flat_repaint_exact_y1_sha256: str
    a_exact_y1_binding_matches_full_repaint: bool
    p_g_overlay_receipt_sha256: str
    p_g_base_camera_y1_sha256: str
    p_g_corrected_camera_y1_sha256: str
    p_g_corrected_camera_frame_sha256: tuple[str, ...]
    p_g_base_scorer_numerators_sha256: str
    p_g_corrected_scorer_numerators_sha256: str
    p_g_owned_camera_values: int
    p_g_actually_changed_camera_values: int
    p_ephemeral_y1_surface_available: Literal[True] = True
    p_ephemeral_surface_persisted: Literal[False] = False
    p_ephemeral_surface_additional_counted_bytes: Literal[0] = 0
    g_ownership_derived_from_exact_p_g: Literal[True] = True
    p_g_ownership_scope: Literal["decoded_g_labels_not_equal_predictor_internal_labels"] = (
        "decoded_g_labels_not_equal_predictor_internal_labels"
    )
    p_g_label_inequality_overlay_r_abi_bound: Literal[True] = True
    g_exact_owned_overlay_applied: Literal[True] = True
    unchanged_p_y1_camera_values_byte_identical_verified: Literal[True] = True
    unchanged_p_y1_scorer_numerators_identical_verified: Literal[True] = True
    p_g_complete_bounded_camera_y1_bytes: Literal[True] = True
    same_class_realization_repair_coordinate_present: Literal[False] = False
    through_r_target_realization_debt_closed: Literal[False] = False
    target_labels_consumed: Literal[False] = False
    scorer_or_evaluator_closure: Literal[False] = False
    a_conditioned_on_predictor_preserving_y1: Literal[False] = False
    exact_counted_p_reopened_and_matched: Literal[True] = True
    exact_counted_g_reapplied: Literal[True] = True
    exact_counted_a_strictly_parsed_and_source_bound: Literal[True] = True
    exact_counted_a_materialized: Literal[False] = False
    all_counted_p_g_a_semantically_consumed: Literal[False] = False
    section_directory_is_sole_slicing_authority: Literal[True] = True
    external_predictor_state_accepted: Literal[False] = False
    caller_decoded_g_accepted: Literal[False] = False
    optional_terminal_quotient_admitted: Literal[False] = False
    predictor_double_decode_exact: Literal[True] = True
    counted_p_is_actual_decode_input: Literal[True] = True
    explicit_runtime_is_actual_decode_input: Literal[True] = True
    source_archive_read_for_decode: Literal[False] = False
    deterministic_preblock_double_replay: Literal[True] = True
    scorer_y0_sha256: None = None
    scorer_y1_sha256: None = None
    factor2_camera_frame_sha256_chronological: tuple[()] = ()
    complete_bounded_chronological_output: Literal[False] = False
    dense_frames_persisted: Literal[False] = False
    scorer_invoked: Literal[False] = False
    n600_evidence_claim: Literal[False] = False
    exact_score_claim: Literal[False] = False
    candidate_archive_eligible: Literal[False] = False
    originality_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    standalone_runtime_closure: Literal[False] = False
    complete_data_in_code_lineage_audit: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != BLOCKER_RECEIPT_SCHEMA or self.receiver_contract_id != RECEIVER_CONTRACT_ID:
            raise TaskspaceMonolithicPGAError("blocker receipt schema or receiver contract changed")
        if self.blocker_code not in {
            TaskspaceMonolithicPGABlockerCode.A_SIGNAL_LOSS_FULL_REPAINT,
        }:
            raise TaskspaceMonolithicPGAError("blocker receipt uses a non-auditable terminal code")
        if type(self.blocker_detail) is not str or not self.blocker_detail:
            raise TaskspaceMonolithicPGAError("blocker receipt detail must be nonempty exact text")
        if type(self.outer_encoding) is not OuterArchiveEncoding:
            raise TaskspaceMonolithicPGAError("blocker receipt outer encoding is outside the closed codec enum")
        for field_name in (
            "archive_sha256",
            "member_sha256",
            "section_descriptor_manifest_sha256",
            "predictor_state_binding_sha256",
            "predictor_semantic_binding_sha256",
            "predictor_source_archive_sha256",
            "predictor_source_runtime_sha256",
            "predictor_bounded_receipt_sha256",
            "predictor_causal_decode_receipt_sha256",
            "decoded_g_labels_sha256",
            "g_owned_mask_sha256",
            "a_source_binding_sha256",
            "a_bound_exact_y1_sha256",
            "flat_repaint_exact_y1_sha256",
            "p_g_overlay_receipt_sha256",
            "p_g_base_camera_y1_sha256",
            "p_g_corrected_camera_y1_sha256",
            "p_g_base_scorer_numerators_sha256",
            "p_g_corrected_scorer_numerators_sha256",
        ):
            _require_sha256(getattr(self, field_name), label=field_name)
        integer_fields = (
            "archive_bytes",
            "compressed_member_bytes",
            "member_bytes",
            "member_crc32",
            "g_owned_cells",
            "p_g_owned_camera_values",
            "p_g_actually_changed_camera_values",
            "p_ephemeral_surface_additional_counted_bytes",
        )
        if any(type(getattr(self, field_name)) is not int for field_name in integer_fields):
            raise TaskspaceMonolithicPGAError("blocker receipt counts must be exact integers")
        if self.archive_bytes < 1 or self.compressed_member_bytes < 1 or self.member_bytes < 1:
            raise TaskspaceMonolithicPGAError("blocker receipt byte counts must be positive")
        if (
            not 0 <= self.member_crc32 <= 0xFFFFFFFF
            or self.g_owned_cells < 1
            or not 1 <= self.p_g_actually_changed_camera_values <= self.p_g_owned_camera_values
        ):
            raise TaskspaceMonolithicPGAError("blocker receipt CRC or G ownership count is invalid")
        if (
            type(self.sections) is not tuple
            or len(self.sections) != 3
            or any(type(row) is not TaskspaceMonolithicPGASectionReceiptV1 for row in self.sections)
        ):
            raise TaskspaceMonolithicPGAError("blocker receipt requires exact P/G/A section rows")
        if tuple(row.role for row in self.sections) != REQUIRED_ROLE_ORDER:
            raise TaskspaceMonolithicPGAError("blocker receipt role order changed")
        directory_stop = _MEMBER_PREFIX.size + len(self.sections) * _SECTION_DESCRIPTOR.size
        if self.sections[0].offset != directory_stop:
            raise TaskspaceMonolithicPGAError("blocker receipt first offset differs from directory EOF")
        if any(left.stop != right.offset for left, right in zip(self.sections[:-1], self.sections[1:], strict=True)):
            raise TaskspaceMonolithicPGAError("blocker receipt section ranges are not contiguous")
        if self.sections[-1].stop != self.member_bytes:
            raise TaskspaceMonolithicPGAError("blocker receipt section ranges do not consume exact member EOF")
        expected_manifest = _sha256(_canonical_json([row.as_dict() for row in self.sections]))
        if self.section_descriptor_manifest_sha256 != expected_manifest:
            raise TaskspaceMonolithicPGAError("blocker receipt descriptor manifest does not recompose")
        if self.sections[0].byte_length != EP725_MEMBER_BYTES:
            raise TaskspaceMonolithicPGAError("blocker receipt counted P length differs from frozen ep725")
        if self.sections[0].payload_sha256 != EP725_MEMBER_SHA256:
            raise TaskspaceMonolithicPGAError("blocker receipt counted P hash differs from frozen ep725")
        if type(self.predictor_causal_decode_receipt) is not Ep725CountedMemberCausalDecodeReceiptV3:
            raise TaskspaceMonolithicPGAError("blocker receipt requires exact causal P receipt type")
        causal = self.predictor_causal_decode_receipt
        if (
            causal.receipt_sha256 != self.predictor_causal_decode_receipt_sha256
            or causal.counted_member_sha256 != self.sections[0].payload_sha256
            or causal.counted_member_bytes != self.sections[0].byte_length
            or causal.explicit_runtime_sha256 != self.predictor_source_runtime_sha256
            or causal.legacy_v2_bounded_receipt_sha256 != self.predictor_bounded_receipt_sha256
            or causal.historical_source_archive_lineage_sha256 != self.predictor_source_archive_sha256
            or causal.source_pair_ids != self.source_pair_ids
            or causal.chronological_camera_frame_sha256 != self.predictor_chronological_camera_frame_sha256
        ):
            raise TaskspaceMonolithicPGAError("blocker receipt causal P custody differs from exact section/runtime")
        if (
            type(self.source_pair_ids) is not tuple
            or any(type(value) is not int for value in self.source_pair_ids)
            or self.source_pair_ids not in {(0,), (0, 1)}
        ):
            raise TaskspaceMonolithicPGAError("blocker receipt pair IDs escaped bounded ep725 prefix n1/n2")
        if type(self.predictor_chronological_camera_frame_sha256) is not tuple or len(
            self.predictor_chronological_camera_frame_sha256
        ) != 2 * len(self.source_pair_ids):
            raise TaskspaceMonolithicPGAError("blocker receipt P camera hash count changed factor2 chronology")
        for index, digest in enumerate(self.predictor_chronological_camera_frame_sha256):
            _require_sha256(digest, label=f"predictor_chronological_camera_frame_sha256[{index}]")
        if (
            type(self.predictor_frame1_camera_sha256) is not tuple
            or self.predictor_frame1_camera_sha256 != self.predictor_chronological_camera_frame_sha256[1::2]
        ):
            raise TaskspaceMonolithicPGAError("blocker receipt P frame1 hashes changed chronological ownership")
        if type(self.p_g_corrected_camera_frame_sha256) is not tuple or len(
            self.p_g_corrected_camera_frame_sha256
        ) != len(self.source_pair_ids):
            raise TaskspaceMonolithicPGAError("blocker receipt corrected P/G Y1 frame-hash count changed")
        for index, digest in enumerate(self.p_g_corrected_camera_frame_sha256):
            _require_sha256(digest, label=f"p_g_corrected_camera_frame_sha256[{index}]")
        if self.g_transport_kind != "NONE" or type(self.a_mode) is not CoupledPreimageMode:
            raise TaskspaceMonolithicPGAError("blocker receipt changed G transport or A mode")
        if self.a_exact_y1_binding_matches_full_repaint is not True:
            raise TaskspaceMonolithicPGAError("blocker code disagrees with the measured full-repaint binding")
        fixed_truth = (
            self.p_ephemeral_y1_surface_available is True
            and self.p_ephemeral_surface_persisted is False
            and self.p_ephemeral_surface_additional_counted_bytes == 0
            and self.g_ownership_derived_from_exact_p_g is True
            and self.p_g_ownership_scope == "decoded_g_labels_not_equal_predictor_internal_labels"
            and self.p_g_label_inequality_overlay_r_abi_bound is True
            and self.g_exact_owned_overlay_applied is True
            and self.unchanged_p_y1_camera_values_byte_identical_verified is True
            and self.unchanged_p_y1_scorer_numerators_identical_verified is True
            and self.p_g_complete_bounded_camera_y1_bytes is True
            and self.same_class_realization_repair_coordinate_present is False
            and self.through_r_target_realization_debt_closed is False
            and self.target_labels_consumed is False
            and self.scorer_or_evaluator_closure is False
            and self.a_conditioned_on_predictor_preserving_y1 is False
            and self.exact_counted_p_reopened_and_matched is True
            and self.exact_counted_g_reapplied is True
            and self.exact_counted_a_strictly_parsed_and_source_bound is True
            and self.exact_counted_a_materialized is False
            and self.all_counted_p_g_a_semantically_consumed is False
            and self.section_directory_is_sole_slicing_authority is True
            and self.external_predictor_state_accepted is False
            and self.caller_decoded_g_accepted is False
            and self.optional_terminal_quotient_admitted is False
            and self.predictor_double_decode_exact is True
            and self.counted_p_is_actual_decode_input is True
            and self.explicit_runtime_is_actual_decode_input is True
            and self.source_archive_read_for_decode is False
            and self.deterministic_preblock_double_replay is True
            and self.scorer_y0_sha256 is None
            and self.scorer_y1_sha256 is None
            and self.factor2_camera_frame_sha256_chronological == ()
            and self.complete_bounded_chronological_output is False
            and self.dense_frames_persisted is False
            and self.scorer_invoked is False
            and self.n600_evidence_claim is False
            and self.exact_score_claim is False
            and self.candidate_archive_eligible is False
            and self.originality_claim is False
            and self.promotion_eligible is False
            and self.standalone_runtime_closure is False
            and self.complete_data_in_code_lineage_audit is False
            and self.research_only is True
        )
        if not fixed_truth:
            raise TaskspaceMonolithicPGAError("blocker receipt truth labels became permissive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["blocker_code"] = self.blocker_code.value
        value["outer_encoding"] = self.outer_encoding.value
        value["a_mode"] = self.a_mode.value
        value["sections"] = [row.as_dict() for row in self.sections]
        value["predictor_causal_decode_receipt"] = self.predictor_causal_decode_receipt.as_dict()
        value["source_pair_ids"] = list(self.source_pair_ids)
        value["predictor_chronological_camera_frame_sha256"] = list(self.predictor_chronological_camera_frame_sha256)
        value["predictor_frame1_camera_sha256"] = list(self.predictor_frame1_camera_sha256)
        value["p_g_corrected_camera_frame_sha256"] = list(self.p_g_corrected_camera_frame_sha256)
        value["factor2_camera_frame_sha256_chronological"] = []
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


@dataclass(frozen=True, slots=True)
class TaskspaceMonolithicPGAReceiverReceiptV1:
    """Canonical receipt for a complete bounded P/G/A3 chronological decode."""

    schema: Literal["tac.taskspace_monolithic_pga_receiver_receipt.v1"]
    receiver_contract_id: Literal["tac.ep725_p_g_owned_overlay_then_a_factor2_chronological.v1"]
    outer_encoding: OuterArchiveEncoding
    archive_bytes: int
    archive_sha256: str
    compressed_member_bytes: int
    member_bytes: int
    member_sha256: str
    member_crc32: int
    sections: tuple[
        TaskspaceMonolithicPGASectionReceiptV1,
        TaskspaceMonolithicPGASectionReceiptV1,
        TaskspaceMonolithicPGASectionReceiptV1,
    ]
    section_descriptor_manifest_sha256: str
    source_pair_ids: tuple[int, ...]
    predictor_state_binding_sha256: str
    predictor_semantic_binding_sha256: str
    predictor_source_archive_sha256: str
    predictor_source_runtime_sha256: str
    predictor_bounded_receipt_sha256: str
    predictor_causal_decode_receipt: Ep725CountedMemberCausalDecodeReceiptV3
    predictor_causal_decode_receipt_sha256: str
    predictor_chronological_camera_frame_sha256: tuple[str, ...]
    predictor_frame0_camera_sha256: tuple[str, ...]
    predictor_frame1_camera_sha256: tuple[str, ...]
    decoded_g_labels_sha256: str
    g_owned_mask_sha256: str
    g_owned_cells: int
    g_transport_kind: Literal["NONE"]
    p_g_overlay_receipt_sha256: str
    p_g_base_camera_y1_sha256: str
    p_g_corrected_camera_y1_sha256: str
    p_g_corrected_camera_frame_sha256: tuple[str, ...]
    p_g_base_scorer_numerators_sha256: str
    p_g_corrected_scorer_numerators_sha256: str
    p_g_owned_camera_values: int
    p_g_actually_changed_camera_values: int
    a_receipt: PredictorPreservingA3ReceiptV1
    a_receipt_sha256: str
    camera_y0_frame_sha256: tuple[str, ...]
    camera_y1_frame_sha256: tuple[str, ...]
    factor2_camera_frame_sha256_chronological: tuple[str, ...]
    exact_counted_p_reopened_and_matched: Literal[True] = True
    exact_counted_g_reapplied: Literal[True] = True
    exact_counted_a_strictly_parsed_and_source_bound: Literal[True] = True
    exact_counted_a_materialized: Literal[True] = True
    all_counted_p_g_a_semantically_consumed: Literal[True] = True
    section_directory_is_sole_slicing_authority: Literal[True] = True
    section_descriptors_sha_crc_and_exact_eof_verified: Literal[True] = True
    external_predictor_state_accepted: Literal[False] = False
    caller_decoded_g_accepted: Literal[False] = False
    optional_terminal_quotient_admitted: Literal[False] = False
    predictor_double_decode_exact: Literal[True] = True
    counted_p_is_actual_decode_input: Literal[True] = True
    explicit_runtime_is_actual_decode_input: Literal[True] = True
    source_archive_read_for_decode: Literal[False] = False
    deterministic_receiver_double_replay: Literal[True] = True
    a_conditioned_on_predictor_preserving_y1: Literal[True] = True
    corrected_y1_unchanged_by_a: Literal[True] = True
    outside_a_owned_p0_camera_values_preserved: Literal[True] = True
    outside_a_owned_p0_scorer_numerators_preserved: Literal[True] = True
    chronological_pair_order_y0_then_y1: Literal[True] = True
    complete_bounded_chronological_output: Literal[True] = True
    dense_frames_persisted: Literal[False] = False
    a3_camera_seam_control_only: Literal[True] = True
    full_a3_se3_xi_basis_universe_implemented: Literal[False] = False
    se3_xi_inverse_solve_implemented: Literal[False] = False
    encoder_side_a_row_selection_solved: Literal[False] = False
    same_class_realization_repair_coordinate_present: Literal[False] = False
    through_r_target_realization_debt_closed: Literal[False] = False
    target_labels_consumed: Literal[False] = False
    scorer_or_evaluator_closure: Literal[False] = False
    scorer_invoked: Literal[False] = False
    n600_evidence_claim: Literal[False] = False
    exact_score_claim: Literal[False] = False
    candidate_archive_eligible: Literal[False] = False
    originality_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    standalone_runtime_closure: Literal[False] = False
    complete_data_in_code_lineage_audit: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != SUCCESS_RECEIPT_SCHEMA or self.receiver_contract_id != RECEIVER_CONTRACT_ID:
            raise TaskspaceMonolithicPGAError("receiver receipt schema or contract changed")
        if type(self.outer_encoding) is not OuterArchiveEncoding:
            raise TaskspaceMonolithicPGAError("receiver receipt outer encoding escaped the closed codec enum")
        for field_name in (
            "archive_sha256",
            "member_sha256",
            "section_descriptor_manifest_sha256",
            "predictor_state_binding_sha256",
            "predictor_semantic_binding_sha256",
            "predictor_source_archive_sha256",
            "predictor_source_runtime_sha256",
            "predictor_bounded_receipt_sha256",
            "predictor_causal_decode_receipt_sha256",
            "decoded_g_labels_sha256",
            "g_owned_mask_sha256",
            "p_g_overlay_receipt_sha256",
            "p_g_base_camera_y1_sha256",
            "p_g_corrected_camera_y1_sha256",
            "p_g_base_scorer_numerators_sha256",
            "p_g_corrected_scorer_numerators_sha256",
            "a_receipt_sha256",
        ):
            _require_sha256(getattr(self, field_name), label=field_name)
        integer_fields = (
            "archive_bytes",
            "compressed_member_bytes",
            "member_bytes",
            "member_crc32",
            "g_owned_cells",
            "p_g_owned_camera_values",
            "p_g_actually_changed_camera_values",
        )
        if any(type(getattr(self, field_name)) is not int for field_name in integer_fields):
            raise TaskspaceMonolithicPGAError("receiver receipt counts must be exact integers")
        if self.archive_bytes < 1 or self.compressed_member_bytes < 1 or self.member_bytes < 1:
            raise TaskspaceMonolithicPGAError("receiver receipt byte counts must be positive")
        if (
            not 0 <= self.member_crc32 <= 0xFFFFFFFF
            or self.g_owned_cells < 1
            or not 1 <= self.p_g_actually_changed_camera_values <= self.p_g_owned_camera_values
        ):
            raise TaskspaceMonolithicPGAError("receiver receipt CRC or G ownership count is invalid")
        if (
            type(self.sections) is not tuple
            or len(self.sections) != 3
            or any(type(row) is not TaskspaceMonolithicPGASectionReceiptV1 for row in self.sections)
            or tuple(row.role for row in self.sections) != REQUIRED_ROLE_ORDER
        ):
            raise TaskspaceMonolithicPGAError("receiver receipt requires exact P/G/A section rows")
        directory_stop = _MEMBER_PREFIX.size + len(self.sections) * _SECTION_DESCRIPTOR.size
        if self.sections[0].offset != directory_stop:
            raise TaskspaceMonolithicPGAError("receiver receipt first section does not start at directory EOF")
        if any(left.stop != right.offset for left, right in zip(self.sections[:-1], self.sections[1:], strict=True)):
            raise TaskspaceMonolithicPGAError("receiver receipt section ranges are not contiguous")
        if self.sections[-1].stop != self.member_bytes:
            raise TaskspaceMonolithicPGAError("receiver receipt sections do not consume exact member EOF")
        manifest = _sha256(_canonical_json([row.as_dict() for row in self.sections]))
        if manifest != self.section_descriptor_manifest_sha256:
            raise TaskspaceMonolithicPGAError("receiver receipt descriptor manifest does not recompose")
        if self.sections[0].byte_length != EP725_MEMBER_BYTES or self.sections[0].payload_sha256 != EP725_MEMBER_SHA256:
            raise TaskspaceMonolithicPGAError("receiver receipt P section differs from frozen ep725")
        if type(self.predictor_causal_decode_receipt) is not Ep725CountedMemberCausalDecodeReceiptV3:
            raise TaskspaceMonolithicPGAError("receiver receipt requires exact causal P receipt type")
        causal = self.predictor_causal_decode_receipt
        if (
            causal.receipt_sha256 != self.predictor_causal_decode_receipt_sha256
            or causal.counted_member_sha256 != self.sections[0].payload_sha256
            or causal.counted_member_bytes != self.sections[0].byte_length
            or causal.explicit_runtime_sha256 != self.predictor_source_runtime_sha256
            or causal.legacy_v2_bounded_receipt_sha256 != self.predictor_bounded_receipt_sha256
            or causal.historical_source_archive_lineage_sha256 != self.predictor_source_archive_sha256
            or causal.source_pair_ids != self.source_pair_ids
            or causal.chronological_camera_frame_sha256 != self.predictor_chronological_camera_frame_sha256
        ):
            raise TaskspaceMonolithicPGAError("receiver receipt causal P custody differs from exact section/runtime")
        if (
            type(self.source_pair_ids) is not tuple
            or any(type(value) is not int for value in self.source_pair_ids)
            or self.source_pair_ids not in {(0,), (0, 1)}
        ):
            raise TaskspaceMonolithicPGAError("receiver receipt pair IDs escaped bounded ep725 prefix n1/n2")
        pair_count = len(self.source_pair_ids)
        hash_tuple_fields = {
            "predictor_chronological_camera_frame_sha256": 2 * pair_count,
            "predictor_frame0_camera_sha256": pair_count,
            "predictor_frame1_camera_sha256": pair_count,
            "p_g_corrected_camera_frame_sha256": pair_count,
            "camera_y0_frame_sha256": pair_count,
            "camera_y1_frame_sha256": pair_count,
            "factor2_camera_frame_sha256_chronological": 2 * pair_count,
        }
        for field_name, expected_count in hash_tuple_fields.items():
            values = getattr(self, field_name)
            if type(values) is not tuple or len(values) != expected_count:
                raise TaskspaceMonolithicPGAError(f"receiver receipt {field_name} count changed")
            for index, digest in enumerate(values):
                _require_sha256(digest, label=f"{field_name}[{index}]")
        if self.predictor_frame0_camera_sha256 != self.predictor_chronological_camera_frame_sha256[0::2]:
            raise TaskspaceMonolithicPGAError("receiver receipt P0 frame hashes changed chronology")
        if self.predictor_frame1_camera_sha256 != self.predictor_chronological_camera_frame_sha256[1::2]:
            raise TaskspaceMonolithicPGAError("receiver receipt P1 frame hashes changed chronology")
        if self.camera_y1_frame_sha256 != self.p_g_corrected_camera_frame_sha256:
            raise TaskspaceMonolithicPGAError("receiver receipt A changed corrected Y1 frame hashes")
        if self.factor2_camera_frame_sha256_chronological[0::2] != self.camera_y0_frame_sha256:
            raise TaskspaceMonolithicPGAError("receiver receipt chronological even frames are not Y0")
        if self.factor2_camera_frame_sha256_chronological[1::2] != self.camera_y1_frame_sha256:
            raise TaskspaceMonolithicPGAError("receiver receipt chronological odd frames are not Y1")
        if self.g_transport_kind != "NONE" or type(self.a_receipt) is not PredictorPreservingA3ReceiptV1:
            raise TaskspaceMonolithicPGAError("receiver receipt changed G transport or A3 receipt type")
        if self.a_receipt.receipt_sha256 != self.a_receipt_sha256:
            raise TaskspaceMonolithicPGAError("receiver receipt A3 receipt hash does not close")
        if (
            self.a_receipt.source_pair_ids != self.source_pair_ids
            or self.a_receipt.packet_sha256 != self.sections[2].payload_sha256
            or self.a_receipt.packet_bytes != self.sections[2].byte_length
            or self.a_receipt.predictor_state_binding_sha256 != self.predictor_state_binding_sha256
            or self.a_receipt.predictor_semantic_binding_sha256 != self.predictor_semantic_binding_sha256
            or self.a_receipt.correction_packet_sha256 != self.sections[1].payload_sha256
            or self.a_receipt.correction_labels_sha256 != self.decoded_g_labels_sha256
            or self.a_receipt.overlay_receipt_sha256 != self.p_g_overlay_receipt_sha256
            or self.a_receipt.corrected_camera_y1_sha256 != self.p_g_corrected_camera_y1_sha256
            or self.a_receipt.output_camera_y1_sha256 != self.p_g_corrected_camera_y1_sha256
            or self.a_receipt.upstream_decode_receipt_sha256 != self.predictor_bounded_receipt_sha256
        ):
            raise TaskspaceMonolithicPGAError("receiver receipt A3 foreign keys differ from exact P/G custody")
        if self.a_receipt.corrected_y1_unchanged is not True:
            raise TaskspaceMonolithicPGAError("receiver receipt A3 changed corrected Y1")
        fixed_truth = (
            self.exact_counted_p_reopened_and_matched is True
            and self.exact_counted_g_reapplied is True
            and self.exact_counted_a_strictly_parsed_and_source_bound is True
            and self.exact_counted_a_materialized is True
            and self.all_counted_p_g_a_semantically_consumed is True
            and self.section_directory_is_sole_slicing_authority is True
            and self.section_descriptors_sha_crc_and_exact_eof_verified is True
            and self.external_predictor_state_accepted is False
            and self.caller_decoded_g_accepted is False
            and self.optional_terminal_quotient_admitted is False
            and self.predictor_double_decode_exact is True
            and self.counted_p_is_actual_decode_input is True
            and self.explicit_runtime_is_actual_decode_input is True
            and self.source_archive_read_for_decode is False
            and self.deterministic_receiver_double_replay is True
            and self.a_conditioned_on_predictor_preserving_y1 is True
            and self.corrected_y1_unchanged_by_a is True
            and self.outside_a_owned_p0_camera_values_preserved is True
            and self.outside_a_owned_p0_scorer_numerators_preserved is True
            and self.chronological_pair_order_y0_then_y1 is True
            and self.complete_bounded_chronological_output is True
            and self.dense_frames_persisted is False
            and self.a3_camera_seam_control_only is True
            and self.full_a3_se3_xi_basis_universe_implemented is False
            and self.se3_xi_inverse_solve_implemented is False
            and self.encoder_side_a_row_selection_solved is False
            and self.same_class_realization_repair_coordinate_present is False
            and self.through_r_target_realization_debt_closed is False
            and self.target_labels_consumed is False
            and self.scorer_or_evaluator_closure is False
            and self.scorer_invoked is False
            and self.n600_evidence_claim is False
            and self.exact_score_claim is False
            and self.candidate_archive_eligible is False
            and self.originality_claim is False
            and self.promotion_eligible is False
            and self.standalone_runtime_closure is False
            and self.complete_data_in_code_lineage_audit is False
            and self.research_only is True
        )
        if not fixed_truth:
            raise TaskspaceMonolithicPGAError("receiver receipt truth labels became permissive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["outer_encoding"] = self.outer_encoding.value
        value["sections"] = [row.as_dict() for row in self.sections]
        value["predictor_causal_decode_receipt"] = self.predictor_causal_decode_receipt.as_dict()
        value["source_pair_ids"] = list(self.source_pair_ids)
        value["predictor_chronological_camera_frame_sha256"] = list(self.predictor_chronological_camera_frame_sha256)
        value["predictor_frame0_camera_sha256"] = list(self.predictor_frame0_camera_sha256)
        value["predictor_frame1_camera_sha256"] = list(self.predictor_frame1_camera_sha256)
        value["p_g_corrected_camera_frame_sha256"] = list(self.p_g_corrected_camera_frame_sha256)
        value["a_receipt"] = self.a_receipt.as_dict()
        value["camera_y0_frame_sha256"] = list(self.camera_y0_frame_sha256)
        value["camera_y1_frame_sha256"] = list(self.camera_y1_frame_sha256)
        value["factor2_camera_frame_sha256_chronological"] = list(self.factor2_camera_frame_sha256_chronological)
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


@dataclass(frozen=True, slots=True)
class TaskspaceMonolithicPGAG8ReceiverReceiptV1:
    """Canonical receipt for composite G8 then post-G8 conditional A."""

    schema: Literal["tac.taskspace_monolithic_pga_g8_receiver_receipt.v1"]
    receiver_contract_id: Literal["tac.ep725_p_semantic_g_g8_then_post_g8_a_factor2.v1"]
    outer_encoding: OuterArchiveEncoding
    archive_bytes: int
    archive_sha256: str
    compressed_member_bytes: int
    member_bytes: int
    member_sha256: str
    member_crc32: int
    sections: tuple[
        TaskspaceMonolithicPGASectionReceiptV1,
        TaskspaceMonolithicPGASectionReceiptV1,
        TaskspaceMonolithicPGASectionReceiptV1,
    ]
    section_descriptor_manifest_sha256: str
    source_pair_ids: tuple[int, ...]
    predictor_causal_decode_receipt: Ep725CountedMemberCausalDecodeReceiptV3
    predictor_causal_decode_receipt_sha256: str
    inner_semantic_g_packet_sha256: str
    inner_g8_repair_packet_sha256: str
    semantic_g_labels_sha256: str
    semantic_overlay_receipt: PredictorPreservingOverlayReceiptV1
    semantic_overlay_receipt_sha256: str
    pre_g8_corrected_y1_sha256: str
    g8_receipt: SameClassRealizationRepairReceiptV1
    g8_receipt_sha256: str
    post_g8_corrected_y1_sha256: str
    post_g8_a_receipt: PostG8ConditionalAReceiptV1
    post_g8_a_receipt_sha256: str
    camera_y0_frame_sha256: tuple[str, ...]
    camera_y1_frame_sha256: tuple[str, ...]
    factor2_camera_frame_sha256_chronological: tuple[str, ...]
    exact_counted_p_reopened_and_matched: Literal[True] = True
    exact_composite_g_strictly_parsed: Literal[True] = True
    inner_semantic_g_reapplied: Literal[True] = True
    semantic_overlay_applied_before_g8: Literal[True] = True
    exact_g8_repair_applied_after_semantic_g: Literal[True] = True
    semantic_g_labels_unchanged_by_g8: Literal[True] = True
    exact_post_g8_a_strictly_parsed_and_source_bound: Literal[True] = True
    exact_post_g8_a_materialized: Literal[True] = True
    post_g8_y1_unchanged_by_a: Literal[True] = True
    all_counted_p_g_a_semantically_consumed: Literal[True] = True
    section_directory_is_sole_slicing_authority: Literal[True] = True
    section_descriptors_sha_crc_and_exact_eof_verified: Literal[True] = True
    external_predictor_state_accepted: Literal[False] = False
    caller_decoded_g_accepted: Literal[False] = False
    optional_terminal_quotient_admitted: Literal[False] = False
    cached_p_surface_requires_per_archive_identity_match: Literal[True] = True
    deterministic_receiver_double_replay: Literal[True] = True
    chronological_pair_order_y0_then_y1: Literal[True] = True
    complete_bounded_chronological_output: Literal[True] = True
    dense_frames_persisted: Literal[False] = False
    same_class_realization_repair_coordinate_present: Literal[True] = True
    through_r_target_realization_debt_closed: Literal[False] = False
    target_labels_consumed_by_receiver: Literal[False] = False
    scorer_or_evaluator_closure: Literal[False] = False
    scorer_invoked: Literal[False] = False
    n600_evidence_claim: Literal[False] = False
    exact_score_claim: Literal[False] = False
    candidate_archive_eligible: Literal[False] = False
    originality_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    standalone_runtime_closure: Literal[False] = False
    complete_data_in_code_lineage_audit: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != G8_SUCCESS_RECEIPT_SCHEMA or self.receiver_contract_id != G8_RECEIVER_CONTRACT_ID:
            raise TaskspaceMonolithicPGAError("G8 receiver receipt schema or contract changed")
        if type(self.outer_encoding) is not OuterArchiveEncoding:
            raise TaskspaceMonolithicPGAError("G8 receiver outer encoding escaped the closed enum")
        for field in (
            "archive_sha256",
            "member_sha256",
            "section_descriptor_manifest_sha256",
            "predictor_causal_decode_receipt_sha256",
            "inner_semantic_g_packet_sha256",
            "inner_g8_repair_packet_sha256",
            "semantic_g_labels_sha256",
            "semantic_overlay_receipt_sha256",
            "pre_g8_corrected_y1_sha256",
            "g8_receipt_sha256",
            "post_g8_corrected_y1_sha256",
            "post_g8_a_receipt_sha256",
        ):
            _require_sha256(getattr(self, field), label=field)
        for field in (
            "archive_bytes",
            "compressed_member_bytes",
            "member_bytes",
            "member_crc32",
        ):
            if type(getattr(self, field)) is not int:
                raise TaskspaceMonolithicPGAError("G8 receiver receipt counts must be exact integers")
        if (
            self.archive_bytes < 1
            or self.compressed_member_bytes < 1
            or self.member_bytes < 1
            or not 0 <= self.member_crc32 <= 0xFFFFFFFF
        ):
            raise TaskspaceMonolithicPGAError("G8 receiver receipt byte counts or CRC are invalid")
        if (
            type(self.sections) is not tuple
            or len(self.sections) != 3
            or any(type(row) is not TaskspaceMonolithicPGASectionReceiptV1 for row in self.sections)
            or tuple(row.role for row in self.sections) != REQUIRED_ROLE_ORDER
        ):
            raise TaskspaceMonolithicPGAError("G8 receiver receipt requires exact P/G/A section rows")
        directory_stop = _MEMBER_PREFIX.size + len(self.sections) * _SECTION_DESCRIPTOR.size
        if self.sections[0].offset != directory_stop:
            raise TaskspaceMonolithicPGAError("G8 receiver first section does not start at directory EOF")
        if (
            any(left.stop != right.offset for left, right in zip(self.sections[:-1], self.sections[1:], strict=True))
            or self.sections[-1].stop != self.member_bytes
        ):
            raise TaskspaceMonolithicPGAError("G8 receiver section ranges do not consume exact member EOF")
        manifest = _sha256(_canonical_json([row.as_dict() for row in self.sections]))
        if manifest != self.section_descriptor_manifest_sha256:
            raise TaskspaceMonolithicPGAError("G8 receiver descriptor manifest does not recompose")
        if self.sections[0].byte_length != EP725_MEMBER_BYTES or self.sections[0].payload_sha256 != EP725_MEMBER_SHA256:
            raise TaskspaceMonolithicPGAError("G8 receiver P section differs from frozen ep725")
        if (
            type(self.source_pair_ids) is not tuple
            or any(type(value) is not int for value in self.source_pair_ids)
            or self.source_pair_ids not in {(0,), (0, 1)}
        ):
            raise TaskspaceMonolithicPGAError("G8 receiver pair IDs escaped bounded ep725 n1/n2")
        pair_count = len(self.source_pair_ids)
        for field, expected_count in {
            "camera_y0_frame_sha256": pair_count,
            "camera_y1_frame_sha256": pair_count,
            "factor2_camera_frame_sha256_chronological": 2 * pair_count,
        }.items():
            values = getattr(self, field)
            if type(values) is not tuple or len(values) != expected_count:
                raise TaskspaceMonolithicPGAError(f"G8 receiver {field} count changed")
            for index, digest in enumerate(values):
                _require_sha256(digest, label=f"{field}[{index}]")
        if (
            self.factor2_camera_frame_sha256_chronological[0::2] != self.camera_y0_frame_sha256
            or self.factor2_camera_frame_sha256_chronological[1::2] != self.camera_y1_frame_sha256
        ):
            raise TaskspaceMonolithicPGAError("G8 receiver chronology is not exact [Y0,Y1]")
        if type(self.predictor_causal_decode_receipt) is not Ep725CountedMemberCausalDecodeReceiptV3:
            raise TaskspaceMonolithicPGAError("G8 receiver requires exact causal P receipt")
        causal = self.predictor_causal_decode_receipt
        if (
            causal.receipt_sha256 != self.predictor_causal_decode_receipt_sha256
            or causal.counted_member_sha256 != self.sections[0].payload_sha256
            or causal.counted_member_bytes != self.sections[0].byte_length
            or causal.source_pair_ids != self.source_pair_ids
        ):
            raise TaskspaceMonolithicPGAError("G8 receiver causal P custody differs from exact section")
        if type(self.semantic_overlay_receipt) is not PredictorPreservingOverlayReceiptV1:
            raise TaskspaceMonolithicPGAError("G8 receiver requires exact semantic overlay receipt")
        overlay = self.semantic_overlay_receipt
        if (
            overlay.receipt_sha256 != self.semantic_overlay_receipt_sha256
            or overlay.pair_count != pair_count
            or overlay.correction_packet_sha256 != self.inner_semantic_g_packet_sha256
            or overlay.corrected_labels_sha256 != self.semantic_g_labels_sha256
            or overlay.corrected_camera_y1_sha256 != self.pre_g8_corrected_y1_sha256
        ):
            raise TaskspaceMonolithicPGAError("G8 receiver semantic overlay custody does not close")
        if type(self.g8_receipt) is not SameClassRealizationRepairReceiptV1:
            raise TaskspaceMonolithicPGAError("G8 receiver requires exact G8R1 receipt")
        g8 = self.g8_receipt
        if (
            g8.receipt_sha256 != self.g8_receipt_sha256
            or g8.source_pair_ids != self.source_pair_ids
            or g8.corrected_y1_camera_sha256 != self.pre_g8_corrected_y1_sha256
            or g8.packet_sha256 != self.inner_g8_repair_packet_sha256
            or g8.output_g_labels_sha256 != self.semantic_g_labels_sha256
            or g8.output_camera_y1_sha256 != self.post_g8_corrected_y1_sha256
        ):
            raise TaskspaceMonolithicPGAError("G8 receiver repair custody does not close")
        if type(self.post_g8_a_receipt) is not PostG8ConditionalAReceiptV1:
            raise TaskspaceMonolithicPGAError("G8 receiver requires exact post-G8 A receipt")
        post_a = self.post_g8_a_receipt
        if (
            post_a.receipt_sha256 != self.post_g8_a_receipt_sha256
            or post_a.source_pair_ids != self.source_pair_ids
            or post_a.packet_sha256 != self.sections[2].payload_sha256
            or post_a.packet_bytes != self.sections[2].byte_length
            or post_a.composite_g_section_sha256 != self.sections[1].payload_sha256
            or post_a.semantic_g_packet_sha256 != self.inner_semantic_g_packet_sha256
            or post_a.semantic_g_labels_sha256 != self.semantic_g_labels_sha256
            or post_a.repair_packet_sha256 != self.inner_g8_repair_packet_sha256
            or post_a.repair_receipt_sha256 != self.g8_receipt_sha256
            or post_a.pre_g8_corrected_y1_sha256 != self.pre_g8_corrected_y1_sha256
            or post_a.post_g8_corrected_y1_sha256 != self.post_g8_corrected_y1_sha256
            or post_a.output_camera_y1_sha256 != self.post_g8_corrected_y1_sha256
        ):
            raise TaskspaceMonolithicPGAError("G8 receiver post-G8 A foreign keys do not close")
        truth = {
            "exact_counted_p_reopened_and_matched": True,
            "exact_composite_g_strictly_parsed": True,
            "inner_semantic_g_reapplied": True,
            "semantic_overlay_applied_before_g8": True,
            "exact_g8_repair_applied_after_semantic_g": True,
            "semantic_g_labels_unchanged_by_g8": True,
            "exact_post_g8_a_strictly_parsed_and_source_bound": True,
            "exact_post_g8_a_materialized": True,
            "post_g8_y1_unchanged_by_a": True,
            "all_counted_p_g_a_semantically_consumed": True,
            "section_directory_is_sole_slicing_authority": True,
            "section_descriptors_sha_crc_and_exact_eof_verified": True,
            "external_predictor_state_accepted": False,
            "caller_decoded_g_accepted": False,
            "optional_terminal_quotient_admitted": False,
            "cached_p_surface_requires_per_archive_identity_match": True,
            "deterministic_receiver_double_replay": True,
            "chronological_pair_order_y0_then_y1": True,
            "complete_bounded_chronological_output": True,
            "dense_frames_persisted": False,
            "same_class_realization_repair_coordinate_present": True,
            "through_r_target_realization_debt_closed": False,
            "target_labels_consumed_by_receiver": False,
            "scorer_or_evaluator_closure": False,
            "scorer_invoked": False,
            "n600_evidence_claim": False,
            "exact_score_claim": False,
            "candidate_archive_eligible": False,
            "originality_claim": False,
            "promotion_eligible": False,
            "standalone_runtime_closure": False,
            "complete_data_in_code_lineage_audit": False,
            "research_only": True,
        }
        if any(getattr(self, field) is not expected for field, expected in truth.items()):
            raise TaskspaceMonolithicPGAError("G8 receiver receipt truth labels became permissive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["outer_encoding"] = self.outer_encoding.value
        value["sections"] = [row.as_dict() for row in self.sections]
        value["source_pair_ids"] = list(self.source_pair_ids)
        value["predictor_causal_decode_receipt"] = self.predictor_causal_decode_receipt.as_dict()
        value["semantic_overlay_receipt"] = self.semantic_overlay_receipt.as_dict()
        value["g8_receipt"] = self.g8_receipt.as_dict()
        value["post_g8_a_receipt"] = self.post_g8_a_receipt.as_dict()
        value["camera_y0_frame_sha256"] = list(self.camera_y0_frame_sha256)
        value["camera_y1_frame_sha256"] = list(self.camera_y1_frame_sha256)
        value["factor2_camera_frame_sha256_chronological"] = list(self.factor2_camera_frame_sha256_chronological)
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


@dataclass(frozen=True, slots=True)
class TaskspaceMonolithicPGAPassReceiverReceiptV1:
    """Canonical production receipt for PASS-G, optional G8, then A."""

    schema: Literal["tac.taskspace_monolithic_pga_pass_receiver_receipt.v1"]
    receiver_contract_id: Literal["tac.ep725_p_pass_semantic_g_optional_g8_then_conditional_a_factor2.v1"]
    outer_encoding: OuterArchiveEncoding
    archive_bytes: int
    archive_sha256: str
    compressed_member_bytes: int
    member_bytes: int
    member_sha256: str
    member_crc32: int
    sections: tuple[
        TaskspaceMonolithicPGASectionReceiptV1,
        TaskspaceMonolithicPGASectionReceiptV1,
        TaskspaceMonolithicPGASectionReceiptV1,
    ]
    section_descriptor_manifest_sha256: str
    source_pair_ids: tuple[int, ...]
    predictor_causal_decode_receipt: Ep725CountedMemberCausalDecodeReceiptV3
    predictor_causal_decode_receipt_sha256: str
    pass_g_receipt: PassSemanticGEnvelopeReceiptV1
    pass_g_receipt_sha256: str
    conditional_a_receipt: PassConditionalAReceiptV1
    conditional_a_receipt_sha256: str
    camera_y0_frame_sha256: tuple[str, ...]
    camera_y1_frame_sha256: tuple[str, ...]
    factor2_camera_frame_sha256_chronological: tuple[str, ...]
    optional_g8_repair_present: bool
    exact_counted_p_reopened_and_matched: Literal[True] = True
    exact_nonempty_pass_semantic_g_strictly_parsed: Literal[True] = True
    pass_semantic_g_source_bound_and_materialized: Literal[True] = True
    optional_g8_mode_explicit: Literal[True] = True
    exact_conditional_a_strictly_parsed_and_source_bound: Literal[True] = True
    exact_conditional_a_materialized: Literal[True] = True
    conditional_y1_unchanged_by_a: Literal[True] = True
    all_counted_p_g_a_semantically_consumed: Literal[True] = True
    section_directory_is_sole_slicing_authority: Literal[True] = True
    section_descriptors_sha_crc_and_exact_eof_verified: Literal[True] = True
    external_predictor_state_accepted: Literal[False] = False
    caller_decoded_g_accepted: Literal[False] = False
    optional_terminal_quotient_admitted: Literal[False] = False
    cached_p_surface_requires_per_archive_identity_match: Literal[True] = True
    deterministic_receiver_double_replay: Literal[True] = True
    chronological_pair_order_y0_then_y1: Literal[True] = True
    complete_bounded_chronological_output: Literal[True] = True
    dense_frames_persisted: Literal[False] = False
    through_r_target_realization_debt_closed: Literal[False] = False
    target_labels_consumed_by_receiver: Literal[False] = False
    scorer_or_evaluator_closure: Literal[False] = False
    scorer_invoked: Literal[False] = False
    n600_evidence_claim: Literal[False] = False
    exact_score_claim: Literal[False] = False
    candidate_archive_eligible: Literal[False] = False
    originality_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    standalone_runtime_closure: Literal[False] = False
    complete_data_in_code_lineage_audit: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != PASS_SUCCESS_RECEIPT_SCHEMA or self.receiver_contract_id != PASS_RECEIVER_CONTRACT_ID:
            raise TaskspaceMonolithicPGAError("PASS receiver receipt schema or contract changed")
        if type(self.outer_encoding) is not OuterArchiveEncoding:
            raise TaskspaceMonolithicPGAError("PASS receiver outer encoding escaped the closed enum")
        for field in (
            "archive_sha256",
            "member_sha256",
            "section_descriptor_manifest_sha256",
            "predictor_causal_decode_receipt_sha256",
            "pass_g_receipt_sha256",
            "conditional_a_receipt_sha256",
        ):
            _require_sha256(getattr(self, field), label=field)
        for field in ("archive_bytes", "compressed_member_bytes", "member_bytes", "member_crc32"):
            if type(getattr(self, field)) is not int:
                raise TaskspaceMonolithicPGAError("PASS receiver receipt counts must be exact integers")
        if (
            self.archive_bytes < 1
            or self.compressed_member_bytes < 1
            or self.member_bytes < 1
            or not 0 <= self.member_crc32 <= 0xFFFFFFFF
        ):
            raise TaskspaceMonolithicPGAError("PASS receiver receipt byte counts or CRC are invalid")
        if (
            type(self.sections) is not tuple
            or len(self.sections) != 3
            or any(type(row) is not TaskspaceMonolithicPGASectionReceiptV1 for row in self.sections)
            or tuple(row.role for row in self.sections) != REQUIRED_ROLE_ORDER
        ):
            raise TaskspaceMonolithicPGAError("PASS receiver requires exact P/G/A section rows")
        directory_stop = _MEMBER_PREFIX.size + len(self.sections) * _SECTION_DESCRIPTOR.size
        if self.sections[0].offset != directory_stop:
            raise TaskspaceMonolithicPGAError("PASS receiver first section does not start at directory EOF")
        if (
            any(left.stop != right.offset for left, right in zip(self.sections[:-1], self.sections[1:], strict=True))
            or self.sections[-1].stop != self.member_bytes
        ):
            raise TaskspaceMonolithicPGAError("PASS receiver section ranges do not consume exact member EOF")
        manifest = _sha256(_canonical_json([row.as_dict() for row in self.sections]))
        if manifest != self.section_descriptor_manifest_sha256:
            raise TaskspaceMonolithicPGAError("PASS receiver descriptor manifest does not recompose")
        if self.sections[0].byte_length != EP725_MEMBER_BYTES or self.sections[0].payload_sha256 != EP725_MEMBER_SHA256:
            raise TaskspaceMonolithicPGAError("PASS receiver P section differs from frozen ep725")
        if (
            type(self.source_pair_ids) is not tuple
            or any(type(value) is not int for value in self.source_pair_ids)
            or self.source_pair_ids not in {(0,), (0, 1)}
        ):
            raise TaskspaceMonolithicPGAError("PASS receiver pair IDs escaped bounded ep725 n1/n2")
        pair_count = len(self.source_pair_ids)
        for field, expected_count in {
            "camera_y0_frame_sha256": pair_count,
            "camera_y1_frame_sha256": pair_count,
            "factor2_camera_frame_sha256_chronological": 2 * pair_count,
        }.items():
            values = getattr(self, field)
            if type(values) is not tuple or len(values) != expected_count:
                raise TaskspaceMonolithicPGAError(f"PASS receiver {field} count changed")
            for index, digest in enumerate(values):
                _require_sha256(digest, label=f"{field}[{index}]")
        if (
            self.factor2_camera_frame_sha256_chronological[0::2] != self.camera_y0_frame_sha256
            or self.factor2_camera_frame_sha256_chronological[1::2] != self.camera_y1_frame_sha256
        ):
            raise TaskspaceMonolithicPGAError("PASS receiver chronology is not exact [Y0,Y1]")
        if type(self.predictor_causal_decode_receipt) is not Ep725CountedMemberCausalDecodeReceiptV3:
            raise TaskspaceMonolithicPGAError("PASS receiver requires exact causal P receipt")
        causal = self.predictor_causal_decode_receipt
        if (
            causal.receipt_sha256 != self.predictor_causal_decode_receipt_sha256
            or causal.counted_member_sha256 != self.sections[0].payload_sha256
            or causal.counted_member_bytes != self.sections[0].byte_length
            or causal.source_pair_ids != self.source_pair_ids
        ):
            raise TaskspaceMonolithicPGAError("PASS receiver causal P custody differs from exact section")
        if type(self.pass_g_receipt) is not PassSemanticGEnvelopeReceiptV1:
            raise TaskspaceMonolithicPGAError("PASS receiver requires exact PASS-G receipt")
        pass_g = self.pass_g_receipt
        if (
            pass_g.receipt_sha256 != self.pass_g_receipt_sha256
            or pass_g.source_pair_ids != self.source_pair_ids
            or pass_g.envelope_sha256 != self.sections[1].payload_sha256
            or pass_g.envelope_bytes != self.sections[1].byte_length
            or pass_g.pass_receipt.predictor_program_sha256 != self.sections[0].payload_sha256
        ):
            raise TaskspaceMonolithicPGAError("PASS receiver PASS-G custody does not close")
        expected_repair = pass_g.mode == "PASS_THEN_G8_V1"
        if type(self.optional_g8_repair_present) is not bool or self.optional_g8_repair_present is not expected_repair:
            raise TaskspaceMonolithicPGAError("PASS receiver optional-G8 truth differs from exact envelope mode")
        if type(self.conditional_a_receipt) is not PassConditionalAReceiptV1:
            raise TaskspaceMonolithicPGAError("PASS receiver requires exact conditional-A receipt")
        conditional_a = self.conditional_a_receipt
        source = conditional_a.source
        if (
            conditional_a.receipt_sha256 != self.conditional_a_receipt_sha256
            or source.source_pair_ids != self.source_pair_ids
            or conditional_a.packet_sha256 != self.sections[2].payload_sha256
            or conditional_a.packet_bytes != self.sections[2].byte_length
            or source.pass_g_mode != pass_g.mode
            or source.pass_g_envelope_sha256 != self.sections[1].payload_sha256
            or source.pass_g_receipt_sha256 != self.pass_g_receipt_sha256
            or source.pass_semantic_packet_sha256 != pass_g.pass_packet_sha256
            or source.pass_semantic_receipt_sha256 != pass_g.pass_receipt_sha256
            or source.semantic_labels_sha256 != pass_g.semantic_labels_sha256
            or source.pre_repair_camera_y1_sha256 != pass_g.pre_repair_camera_y1_sha256
            or source.repair_packet_sha256 != pass_g.repair_packet_sha256
            or source.repair_receipt_sha256 != pass_g.repair_receipt_sha256
            or source.conditional_camera_y1_sha256 != pass_g.output_camera_y1_sha256
            or conditional_a.output_camera_y1_sha256 != pass_g.output_camera_y1_sha256
        ):
            raise TaskspaceMonolithicPGAError("PASS receiver conditional-A foreign keys do not close")
        truth = {
            "exact_counted_p_reopened_and_matched": True,
            "exact_nonempty_pass_semantic_g_strictly_parsed": True,
            "pass_semantic_g_source_bound_and_materialized": True,
            "optional_g8_mode_explicit": True,
            "exact_conditional_a_strictly_parsed_and_source_bound": True,
            "exact_conditional_a_materialized": True,
            "conditional_y1_unchanged_by_a": True,
            "all_counted_p_g_a_semantically_consumed": True,
            "section_directory_is_sole_slicing_authority": True,
            "section_descriptors_sha_crc_and_exact_eof_verified": True,
            "external_predictor_state_accepted": False,
            "caller_decoded_g_accepted": False,
            "optional_terminal_quotient_admitted": False,
            "cached_p_surface_requires_per_archive_identity_match": True,
            "deterministic_receiver_double_replay": True,
            "chronological_pair_order_y0_then_y1": True,
            "complete_bounded_chronological_output": True,
            "dense_frames_persisted": False,
            "through_r_target_realization_debt_closed": False,
            "target_labels_consumed_by_receiver": False,
            "scorer_or_evaluator_closure": False,
            "scorer_invoked": False,
            "n600_evidence_claim": False,
            "exact_score_claim": False,
            "candidate_archive_eligible": False,
            "originality_claim": False,
            "promotion_eligible": False,
            "standalone_runtime_closure": False,
            "complete_data_in_code_lineage_audit": False,
            "research_only": True,
        }
        if any(
            type(getattr(self, field)) is not bool or getattr(self, field) != expected
            for field, expected in truth.items()
        ):
            raise TaskspaceMonolithicPGAError("PASS receiver receipt truth labels became permissive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["outer_encoding"] = self.outer_encoding.value
        value["sections"] = [row.as_dict() for row in self.sections]
        value["source_pair_ids"] = list(self.source_pair_ids)
        value["predictor_causal_decode_receipt"] = self.predictor_causal_decode_receipt.as_dict()
        value["pass_g_receipt"] = self.pass_g_receipt.as_dict()
        value["conditional_a_receipt"] = self.conditional_a_receipt.as_dict()
        value["camera_y0_frame_sha256"] = list(self.camera_y0_frame_sha256)
        value["camera_y1_frame_sha256"] = list(self.camera_y1_frame_sha256)
        value["factor2_camera_frame_sha256_chronological"] = list(self.factor2_camera_frame_sha256_chronological)
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


@dataclass(frozen=True, slots=True)
class DecodedTaskspaceMonolithicPGAV1:
    """Ephemeral exact P/G/A3 output plus its counted-object receipt.

    This object deliberately has no serialization method.  A runtime caller can
    write ``chronological_camera_frames`` to its output sink, while the counted
    member remains the sole persisted source.
    """

    decoded_a3: DecodedPredictorPreservingA3V1
    receipt: TaskspaceMonolithicPGAReceiverReceiptV1

    def __post_init__(self) -> None:
        if type(self.decoded_a3) is not DecodedPredictorPreservingA3V1:
            raise TaskspaceMonolithicPGAError("decoded receiver output requires exact A3 decoded type")
        if type(self.receipt) is not TaskspaceMonolithicPGAReceiverReceiptV1:
            raise TaskspaceMonolithicPGAError("decoded receiver output requires exact outer receipt type")
        if self.decoded_a3.receipt != self.receipt.a_receipt:
            raise TaskspaceMonolithicPGAError("decoded receiver output A3 receipt differs from outer receipt")
        y0_hashes = tuple(_sha256(memoryview(frame).cast("B")) for frame in self.decoded_a3.camera_y0)
        y1_hashes = tuple(_sha256(memoryview(frame).cast("B")) for frame in self.decoded_a3.camera_y1)
        chronological_hashes = tuple(
            _sha256(memoryview(frame).cast("B"))
            for pair_frames in self.decoded_a3.chronological_camera_frames
            for frame in pair_frames
        )
        if (
            y0_hashes != self.receipt.camera_y0_frame_sha256
            or y1_hashes != self.receipt.camera_y1_frame_sha256
            or chronological_hashes != self.receipt.factor2_camera_frame_sha256_chronological
        ):
            raise TaskspaceMonolithicPGAError("decoded receiver arrays differ from outer per-frame hashes")
        if not np.array_equal(
            self.decoded_a3.chronological_camera_frames[:, 0],
            self.decoded_a3.camera_y0,
        ) or not np.array_equal(
            self.decoded_a3.chronological_camera_frames[:, 1],
            self.decoded_a3.camera_y1,
        ):
            raise TaskspaceMonolithicPGAError("decoded receiver chronology is not exact [Y0,Y1]")

    @property
    def camera_y0(self) -> np.ndarray:
        return self.decoded_a3.camera_y0

    @property
    def camera_y1(self) -> np.ndarray:
        return self.decoded_a3.camera_y1

    @property
    def chronological_camera_frames(self) -> np.ndarray:
        return self.decoded_a3.chronological_camera_frames


@dataclass(frozen=True, slots=True)
class DecodedTaskspaceMonolithicPGAG8V1:
    """Ephemeral exact composite-G8/post-G8-A output and nested custody."""

    decoded_post_g8_a: DecodedPostG8ConditionalAV1
    receipt: TaskspaceMonolithicPGAG8ReceiverReceiptV1

    def __post_init__(self) -> None:
        if type(self.decoded_post_g8_a) is not DecodedPostG8ConditionalAV1:
            raise TaskspaceMonolithicPGAError("decoded G8 receiver output requires exact post-G8 A type")
        if type(self.receipt) is not TaskspaceMonolithicPGAG8ReceiverReceiptV1:
            raise TaskspaceMonolithicPGAError("decoded G8 receiver output requires exact outer receipt type")
        decoded = self.decoded_post_g8_a
        if decoded.receipt != self.receipt.post_g8_a_receipt:
            raise TaskspaceMonolithicPGAError("decoded post-G8 A receipt differs from outer receipt")
        y0_hashes = tuple(_sha256(memoryview(frame).cast("B")) for frame in decoded.camera_y0)
        y1_hashes = tuple(_sha256(memoryview(frame).cast("B")) for frame in decoded.camera_y1)
        chronological_hashes = tuple(
            _sha256(memoryview(frame).cast("B"))
            for pair_frames in decoded.chronological_camera_frames
            for frame in pair_frames
        )
        if (
            y0_hashes != self.receipt.camera_y0_frame_sha256
            or y1_hashes != self.receipt.camera_y1_frame_sha256
            or chronological_hashes != self.receipt.factor2_camera_frame_sha256_chronological
        ):
            raise TaskspaceMonolithicPGAError("decoded G8 receiver arrays differ from per-frame receipt hashes")
        if not np.array_equal(decoded.chronological_camera_frames[:, 0], decoded.camera_y0) or not np.array_equal(
            decoded.chronological_camera_frames[:, 1], decoded.camera_y1
        ):
            raise TaskspaceMonolithicPGAError("decoded G8 receiver chronology is not exact [Y0,Y1]")

    @property
    def camera_y0(self) -> np.ndarray:
        return self.decoded_post_g8_a.camera_y0

    @property
    def camera_y1(self) -> np.ndarray:
        return self.decoded_post_g8_a.camera_y1

    @property
    def chronological_camera_frames(self) -> np.ndarray:
        return self.decoded_post_g8_a.chronological_camera_frames


@dataclass(frozen=True, slots=True)
class DecodedTaskspaceMonolithicPGAPassV1:
    """Production PASS-G/optional-G8/conditional-A output and custody."""

    decoded_conditional_a: DecodedPassConditionalAV1
    receipt: TaskspaceMonolithicPGAPassReceiverReceiptV1

    def __post_init__(self) -> None:
        if type(self.decoded_conditional_a) is not DecodedPassConditionalAV1:
            raise TaskspaceMonolithicPGAError("decoded PASS receiver requires exact conditional-A output")
        if type(self.receipt) is not TaskspaceMonolithicPGAPassReceiverReceiptV1:
            raise TaskspaceMonolithicPGAError("decoded PASS receiver requires exact outer receipt")
        decoded = self.decoded_conditional_a
        if decoded.receipt != self.receipt.conditional_a_receipt:
            raise TaskspaceMonolithicPGAError("decoded conditional-A receipt differs from outer PASS receipt")
        y0_hashes = tuple(_sha256(memoryview(frame).cast("B")) for frame in decoded.camera_y0)
        y1_hashes = tuple(_sha256(memoryview(frame).cast("B")) for frame in decoded.camera_y1)
        chronological_hashes = tuple(
            _sha256(memoryview(frame).cast("B"))
            for pair_frames in decoded.chronological_camera_frames
            for frame in pair_frames
        )
        if (
            y0_hashes != self.receipt.camera_y0_frame_sha256
            or y1_hashes != self.receipt.camera_y1_frame_sha256
            or chronological_hashes != self.receipt.factor2_camera_frame_sha256_chronological
        ):
            raise TaskspaceMonolithicPGAError("decoded PASS receiver arrays differ from per-frame hashes")
        if not np.array_equal(decoded.chronological_camera_frames[:, 0], decoded.camera_y0) or not np.array_equal(
            decoded.chronological_camera_frames[:, 1], decoded.camera_y1
        ):
            raise TaskspaceMonolithicPGAError("decoded PASS receiver chronology is not exact [Y0,Y1]")

    @property
    def camera_y0(self) -> np.ndarray:
        return self.decoded_conditional_a.camera_y0

    @property
    def camera_y1(self) -> np.ndarray:
        return self.decoded_conditional_a.camera_y1

    @property
    def chronological_camera_frames(self) -> np.ndarray:
        return self.decoded_conditional_a.chronological_camera_frames


@dataclass(frozen=True, slots=True)
class _DecodedPGV1:
    outer: ParsedTaskspaceOuterArchive
    member: ParsedTaskspaceMonolithicPGAMemberV1
    decoded_g: DecodedGenerativeCorrectionV1
    overlay: PredictorPreservingOverlayResultV1
    predictor_frame1_camera_sha256: tuple[str, ...]
    g_owned_mask_sha256: str
    g_owned_cells: int

    @property
    def deterministic_fingerprint(self) -> tuple[object, ...]:
        return (
            self.outer.encoding,
            self.outer.archive_sha256,
            self.outer.member_sha256,
            self.member.descriptor_manifest_sha256,
            self.predictor_frame1_camera_sha256,
            self.overlay.receipt,
            tuple(_sha256(memoryview(frame).cast("B")) for frame in self.overlay.camera_y1),
            _sha256(memoryview(self.decoded_g.labels).cast("B")),
            self.g_owned_mask_sha256,
            self.g_owned_cells,
        )


@dataclass(frozen=True, slots=True)
class _PreblockAuditV1:
    pg: _DecodedPGV1
    a_program: CoupledPreimageProgramV1
    flat_repaint_exact_y1_sha256: str
    blocker_code: TaskspaceMonolithicPGABlockerCode

    @property
    def outer(self) -> ParsedTaskspaceOuterArchive:
        return self.pg.outer

    @property
    def member(self) -> ParsedTaskspaceMonolithicPGAMemberV1:
        return self.pg.member

    @property
    def decoded_g(self) -> DecodedGenerativeCorrectionV1:
        return self.pg.decoded_g

    @property
    def overlay(self) -> PredictorPreservingOverlayResultV1:
        return self.pg.overlay

    @property
    def predictor_frame1_camera_sha256(self) -> tuple[str, ...]:
        return self.pg.predictor_frame1_camera_sha256

    @property
    def g_owned_mask_sha256(self) -> str:
        return self.pg.g_owned_mask_sha256

    @property
    def g_owned_cells(self) -> int:
        return self.pg.g_owned_cells

    @property
    def deterministic_fingerprint(self) -> tuple[object, ...]:
        return (
            self.pg.deterministic_fingerprint,
            self.a_program.to_packet(),
            self.flat_repaint_exact_y1_sha256,
            self.blocker_code,
        )


@dataclass(frozen=True, slots=True)
class _DecodedA3OnceV1:
    pg: _DecodedPGV1
    decoded_a3: DecodedPredictorPreservingA3V1
    camera_y0_frame_sha256: tuple[str, ...]
    camera_y1_frame_sha256: tuple[str, ...]
    factor2_camera_frame_sha256_chronological: tuple[str, ...]

    @property
    def deterministic_fingerprint(self) -> tuple[object, ...]:
        return (
            self.pg.deterministic_fingerprint,
            self.decoded_a3.receipt,
            self.camera_y0_frame_sha256,
            self.camera_y1_frame_sha256,
            self.factor2_camera_frame_sha256_chronological,
        )


@dataclass(frozen=True, slots=True)
class _DecodedG8PGV1:
    outer: ParsedTaskspaceOuterArchive
    member: ParsedTaskspaceMonolithicPGAMemberV1
    decoded_g: DecodedGenerativeCorrectionV1
    overlay: PredictorPreservingOverlayResultV1
    g8_integration: DecodedSameClassRealizationRepairPGAIntegrationV1

    @property
    def deterministic_fingerprint(self) -> tuple[object, ...]:
        repair = self.g8_integration.decoded_repair
        return (
            self.outer.encoding,
            self.outer.archive_sha256,
            self.outer.member_sha256,
            self.member.descriptor_manifest_sha256,
            self.overlay.receipt,
            self.g8_integration.parsed_composite_g_section,
            repair.receipt,
            tuple(_sha256(memoryview(frame).cast("B")) for frame in repair.camera_y1),
            _sha256(memoryview(repair.semantic_labels).cast("B")),
            _sha256(memoryview(repair.owned_scorer_mask).cast("B")),
            _sha256(memoryview(repair.owned_camera_mask).cast("B")),
        )


@dataclass(frozen=True, slots=True)
class _DecodedPostG8AOnceV1:
    pg: _DecodedG8PGV1
    decoded_post_g8_a: DecodedPostG8ConditionalAV1
    camera_y0_frame_sha256: tuple[str, ...]
    camera_y1_frame_sha256: tuple[str, ...]
    factor2_camera_frame_sha256_chronological: tuple[str, ...]

    @property
    def deterministic_fingerprint(self) -> tuple[object, ...]:
        return (
            self.pg.deterministic_fingerprint,
            self.decoded_post_g8_a.receipt,
            self.camera_y0_frame_sha256,
            self.camera_y1_frame_sha256,
            self.factor2_camera_frame_sha256_chronological,
        )


@dataclass(frozen=True, slots=True)
class _DecodedPassPGV1:
    outer: ParsedTaskspaceOuterArchive
    member: ParsedTaskspaceMonolithicPGAMemberV1
    decoded_pass_g: DecodedPassSemanticGEnvelopeV1

    @property
    def deterministic_fingerprint(self) -> tuple[object, ...]:
        decoded = self.decoded_pass_g
        return (
            self.outer.encoding,
            self.outer.archive_sha256,
            self.outer.member_sha256,
            self.member.descriptor_manifest_sha256,
            decoded.receipt,
            _sha256(memoryview(decoded.semantic_labels).cast("B")),
            tuple(_sha256(memoryview(frame).cast("B")) for frame in decoded.conditional_camera_y1),
        )


@dataclass(frozen=True, slots=True)
class _DecodedPassAOnceV1:
    pg: _DecodedPassPGV1
    decoded_conditional_a: DecodedPassConditionalAV1
    camera_y0_frame_sha256: tuple[str, ...]
    camera_y1_frame_sha256: tuple[str, ...]
    factor2_camera_frame_sha256_chronological: tuple[str, ...]

    @property
    def deterministic_fingerprint(self) -> tuple[object, ...]:
        return (
            self.pg.deterministic_fingerprint,
            self.decoded_conditional_a.receipt,
            self.camera_y0_frame_sha256,
            self.camera_y1_frame_sha256,
            self.factor2_camera_frame_sha256_chronological,
        )


def _section_receipts(
    member: ParsedTaskspaceMonolithicPGAMemberV1,
) -> tuple[
    TaskspaceMonolithicPGASectionReceiptV1,
    TaskspaceMonolithicPGASectionReceiptV1,
    TaskspaceMonolithicPGASectionReceiptV1,
]:
    rows = tuple(
        TaskspaceMonolithicPGASectionReceiptV1(
            role=section.role,
            offset=section.offset,
            byte_length=section.byte_length,
            payload_sha256=section.payload_sha256,
            crc32=section.crc32,
        )
        for section in member.sections
    )
    if len(rows) != 3:
        raise TaskspaceMonolithicPGAError("semantic ep725 audit refuses optional T before any output mutation")
    return rows  # type: ignore[return-value]


def _validate_a_source_binding(
    program: CoupledPreimageProgramV1,
    bounded: Ep725BoundedDecodeV2,
    decoded_g: DecodedGenerativeCorrectionV1,
) -> None:
    state = bounded.predictor_state
    if decoded_g.realization_profile is None:
        raise TaskspaceMonolithicPGAError("counted G has no realization profile required by counted A")
    source = program.source_binding
    expected = {
        "predictor_program_sha256": state.predictor_program_sha256,
        "predictor_renderer_sha256": state.predictor_renderer_sha256,
        "source_pair_ids": state.source_pair_ids,
        "predictor_labels_sha256": state.labels_sha256,
        "correction_packet_sha256": decoded_g.correction_packet_sha256,
        "correction_labels_sha256": _a._array_content_sha256(
            decoded_g.labels,
            role="decoded_g_semantic_labels",
        ),
        "realization_profile_sha256": _a._realization_profile_sha256(decoded_g.realization_profile),
    }
    mismatches = sorted(field_name for field_name, value in expected.items() if getattr(source, field_name) != value)
    if mismatches:
        raise TaskspaceMonolithicPGAError(f"counted A live P/G source binding mismatch: {mismatches}")


def _parse_outer_and_member(
    archive: bytes,
    *,
    expected_encoding: OuterArchiveEncoding | str | None,
    expected_archive_sha256: str | None,
    expected_member_sha256: str | None,
    max_member_bytes: int,
) -> tuple[ParsedTaskspaceOuterArchive, ParsedTaskspaceMonolithicPGAMemberV1]:
    try:
        outer = parse_taskspace_outer_archive(
            archive,
            expected_encoding=expected_encoding,
            expected_archive_sha256=expected_archive_sha256,
            expected_member_sha256=expected_member_sha256,
            max_member_bytes=max_member_bytes,
        )
    except TaskspaceOuterArchiveError as exc:
        raise TaskspaceMonolithicPGAError("strict outer archive parse failed") from exc
    member = parse_taskspace_monolithic_pga_member(outer.member_bytes, max_member_bytes=max_member_bytes)
    return outer, member


def _decode_pg_once(
    archive: bytes,
    surface: Ep725EphemeralRuntimeSurfaceV2,
    *,
    expected_encoding: OuterArchiveEncoding | str | None,
    expected_archive_sha256: str | None,
    expected_member_sha256: str | None,
    max_member_bytes: int,
) -> _DecodedPGV1:
    bounded = surface.bounded_decode
    outer, member = _parse_outer_and_member(
        archive,
        expected_encoding=expected_encoding,
        expected_archive_sha256=expected_archive_sha256,
        expected_member_sha256=expected_member_sha256,
        max_member_bytes=max_member_bytes,
    )
    if len(member.sections) == 4:
        raise TaskspaceMonolithicPGAReceiverBlocker(
            TaskspaceMonolithicPGABlockerCode.TERMINAL_QUOTIENT_CONSUMER_OWED,
            "optional T is structurally bound but has no semantic consumer",
        )

    p_section = member.section(TaskspaceMonolithicPGARole.PREDICTOR)
    g_section = member.section(TaskspaceMonolithicPGARole.GENERATIVE_CORRECTION)
    state = bounded.predictor_state
    if (
        p_section.payload != state.predictor_program
        or p_section.byte_length != EP725_MEMBER_BYTES
        or p_section.payload_sha256 != EP725_MEMBER_SHA256
    ):
        raise TaskspaceMonolithicPGAError("counted P section differs from exact reopened frozen ep725 LVLS1 bytes")
    if not p_section.payload.startswith(_P_PACKET_MAGIC):
        raise TaskspaceMonolithicPGAError("counted P section is not the exact LVLS1 grammar")
    if not g_section.payload.startswith(_G_PACKET_MAGIC):
        raise TaskspaceMonolithicPGAError("counted G section is not the strict finite correction grammar")

    try:
        parse_generative_taskspace_correction_v2(g_section.payload, predictor_state=state)
        decoded_g = apply_generative_taskspace_correction_v2(g_section.payload, predictor_state=state)
    except (GenerativeTaskspaceCorrectionError, TaskspacePredictorV2ConsumerSeamError, ValueError) as exc:
        raise TaskspaceMonolithicPGAError("counted G failed strict V2 parse/apply against exact counted P") from exc
    if decoded_g.correction_packet_sha256 != g_section.payload_sha256:
        raise TaskspaceMonolithicPGAError("decoded G is not hash-bound to its directory-owned exact bytes")
    try:
        ownership: GCorrectionOwnershipV2 = derive_g_correction_ownership_v2(state, decoded_g)
    except TaskspacePredictorV2ConsumerSeamError as exc:
        raise TaskspaceMonolithicPGAError("exact G correction ownership could not be derived from P/G") from exc
    owned_cells = ownership.changed_cells
    if owned_cells < 1:
        raise TaskspaceMonolithicPGAError("counted G does not own or change any bounded semantic cell")
    owned_sha = ownership.ownership_mask_sha256

    try:
        overlay = overlay_g_on_predictor_camera_y1(
            surface.frame1_camera,
            state.labels,
            decoded_g,
        )
    except PredictorPreservingOverlayError as exc:
        raise TaskspaceMonolithicPGAError("exact predictor-preserving P/G camera overlay failed") from exc
    if not np.array_equal(overlay.changed_label_mask, ownership.owned_mask):
        raise TaskspaceMonolithicPGAError("P/G overlay ownership differs from the exact consumer-seam mask")
    if overlay.receipt.changed_label_mask_sha256 != ownership.ownership_mask_sha256:
        raise TaskspaceMonolithicPGAError("P/G overlay ownership hash differs from the exact consumer-seam hash")
    if not np.array_equal(
        overlay.camera_y1[~overlay.owned_camera_mask],
        surface.frame1_camera[~overlay.owned_camera_mask],
    ):
        raise TaskspaceMonolithicPGAError("P/G overlay changed decoder-owned camera values outside G support")

    frame1_hashes = tuple(_sha256(memoryview(frame).cast("B")) for frame in surface.frame1_camera)
    if frame1_hashes != bounded.chronological_camera_frame_sha256[1::2]:
        raise TaskspaceMonolithicPGAError("ephemeral P frame1 bytes differ from bounded chronological custody")
    return _DecodedPGV1(
        outer=outer,
        member=member,
        decoded_g=decoded_g,
        overlay=overlay,
        predictor_frame1_camera_sha256=frame1_hashes,
        g_owned_mask_sha256=owned_sha,
        g_owned_cells=owned_cells,
    )


def _audit_once(
    archive: bytes,
    surface: Ep725EphemeralRuntimeSurfaceV2,
    *,
    expected_encoding: OuterArchiveEncoding | str | None,
    expected_archive_sha256: str | None,
    expected_member_sha256: str | None,
    max_member_bytes: int,
) -> _PreblockAuditV1:
    pg = _decode_pg_once(
        archive,
        surface,
        expected_encoding=expected_encoding,
        expected_archive_sha256=expected_archive_sha256,
        expected_member_sha256=expected_member_sha256,
        max_member_bytes=max_member_bytes,
    )
    bounded = surface.bounded_decode
    a_section = pg.member.section(TaskspaceMonolithicPGARole.COUPLED_PREIMAGE)
    if a_section.payload.startswith(A3_PACKET_MAGIC):
        raise TaskspaceMonolithicPGAError("legacy blocker audit refuses predictor-preserving A3; use the full receiver")

    try:
        a_program = parse_coupled_preimage_program(a_section.payload)
        _validate_a_source_binding(a_program, bounded, pg.decoded_g)
    except (CoupledPreimageProgramError, TaskspaceMonolithicPGAError, ValueError) as exc:
        if isinstance(exc, TaskspaceMonolithicPGAError):
            raise
        raise TaskspaceMonolithicPGAError("counted A failed strict parse or exact live P/G source binding") from exc
    if a_program.to_packet() != a_section.payload:
        raise TaskspaceMonolithicPGAError("counted A changed on strict canonical parse-back")

    try:
        full_repaint = np.ascontiguousarray(pg.decoded_g.paint_rgb(), dtype=np.uint8)
    except GenerativeTaskspaceCorrectionError as exc:
        raise TaskspaceMonolithicPGAError("current G cannot produce even its diagnostic flat repaint") from exc
    flat_hash = _a._array_content_sha256(full_repaint, role="exact_realized_uint8_y1")
    bound_to_flat = a_program.source_binding.exact_y1_content_sha256 == flat_hash
    if not bound_to_flat:
        raise TaskspaceMonolithicPGAError("counted A is not bound to the current seam's exact full-repaint Y1")

    return _PreblockAuditV1(
        pg=pg,
        a_program=a_program,
        flat_repaint_exact_y1_sha256=flat_hash,
        blocker_code=TaskspaceMonolithicPGABlockerCode.A_SIGNAL_LOSS_FULL_REPAINT,
    )


def _decode_a3_once(
    archive: bytes,
    surface: Ep725EphemeralRuntimeSurfaceV2,
    *,
    expected_encoding: OuterArchiveEncoding | str | None,
    expected_archive_sha256: str | None,
    expected_member_sha256: str | None,
    max_member_bytes: int,
) -> _DecodedA3OnceV1:
    pg = _decode_pg_once(
        archive,
        surface,
        expected_encoding=expected_encoding,
        expected_archive_sha256=expected_archive_sha256,
        expected_member_sha256=expected_member_sha256,
        max_member_bytes=max_member_bytes,
    )
    a_section = pg.member.section(TaskspaceMonolithicPGARole.COUPLED_PREIMAGE)
    if not a_section.payload.startswith(A3_PACKET_MAGIC):
        raise TaskspaceMonolithicPGAError("full receiver requires the predictor-preserving A3 grammar")
    try:
        predictor_surface = PredictorCameraPairSurfaceV1.from_ep725(surface)
        decoded_a3 = decode_predictor_preserving_a3_packet(
            a_section.payload,
            predictor_surface=predictor_surface,
            decoded_g=pg.decoded_g,
            corrected_y1_overlay=pg.overlay,
        )
        if parse_predictor_preserving_a3_receipt(decoded_a3.receipt.to_receipt_bytes()) != decoded_a3.receipt:
            raise PredictorPreservingA3Error("A3 receipt failed strict receiver parse-back")
    except PredictorPreservingA3Error as exc:
        raise TaskspaceMonolithicPGAError(
            "counted A3 failed exact decode against ephemeral P0 and predictor-preserving G/Y1"
        ) from exc
    if decoded_a3.receipt.packet_sha256 != a_section.payload_sha256:
        raise TaskspaceMonolithicPGAError("decoded A3 is not hash-bound to its directory-owned exact bytes")
    if not np.array_equal(decoded_a3.camera_y1, pg.overlay.camera_y1):
        raise TaskspaceMonolithicPGAError("counted A3 changed exact predictor-preserving corrected Y1")

    y0_hashes = tuple(_sha256(memoryview(frame).cast("B")) for frame in decoded_a3.camera_y0)
    y1_hashes = tuple(_sha256(memoryview(frame).cast("B")) for frame in decoded_a3.camera_y1)
    chronological_hashes = tuple(
        _sha256(memoryview(frame).cast("B"))
        for pair_frames in decoded_a3.chronological_camera_frames
        for frame in pair_frames
    )
    if y1_hashes != tuple(_sha256(memoryview(frame).cast("B")) for frame in pg.overlay.camera_y1):
        raise TaskspaceMonolithicPGAError("A3 Y1 frame hashes differ from exact P/G overlay")
    if chronological_hashes[0::2] != y0_hashes or chronological_hashes[1::2] != y1_hashes:
        raise TaskspaceMonolithicPGAError("A3 chronological output is not exact factor-2 [Y0,Y1] order")
    return _DecodedA3OnceV1(
        pg=pg,
        decoded_a3=decoded_a3,
        camera_y0_frame_sha256=y0_hashes,
        camera_y1_frame_sha256=y1_hashes,
        factor2_camera_frame_sha256_chronological=chronological_hashes,
    )


def _decode_g8_pg_once(
    archive: bytes,
    surface: Ep725EphemeralRuntimeSurfaceV2,
    *,
    expected_encoding: OuterArchiveEncoding | str | None,
    expected_archive_sha256: str | None,
    expected_member_sha256: str | None,
    max_member_bytes: int,
) -> _DecodedG8PGV1:
    """Strictly apply inner semantic G/overlay, then exact G8 repair."""

    bounded = surface.bounded_decode
    outer, member = _parse_outer_and_member(
        archive,
        expected_encoding=expected_encoding,
        expected_archive_sha256=expected_archive_sha256,
        expected_member_sha256=expected_member_sha256,
        max_member_bytes=max_member_bytes,
    )
    if len(member.sections) == 4:
        raise TaskspaceMonolithicPGAReceiverBlocker(
            TaskspaceMonolithicPGABlockerCode.TERMINAL_QUOTIENT_CONSUMER_OWED,
            "optional T is structurally bound but has no semantic consumer",
        )
    p_section = member.section(TaskspaceMonolithicPGARole.PREDICTOR)
    g_section = member.section(TaskspaceMonolithicPGARole.GENERATIVE_CORRECTION)
    state = bounded.predictor_state
    if (
        p_section.payload != state.predictor_program
        or p_section.byte_length != EP725_MEMBER_BYTES
        or p_section.payload_sha256 != EP725_MEMBER_SHA256
        or not p_section.payload.startswith(_P_PACKET_MAGIC)
    ):
        raise TaskspaceMonolithicPGAError("composite-G receiver P differs from exact reopened ep725 bytes")
    if not g_section.payload.startswith(COMPOSITE_G_SECTION_MAGIC):
        raise TaskspaceMonolithicPGAError("composite-G receiver requires exact TACG8S1 section bytes")
    predictor_surface = PredictorCameraPairSurfaceV1.from_ep725(surface)
    try:
        integration = decode_same_class_realization_repair_from_pga_sections(
            g_section.payload,
            predictor_section_payload=p_section.payload,
            predictor_surface=predictor_surface,
            predictor_codec_id="LVLS1.v1",
            semantic_g_codec_id="TACG1C.v2",
        )
    except SameClassRealizationRepairError as exc:
        raise TaskspaceMonolithicPGAError(
            "counted composite G failed strict inner semantic-G then exact G8 decode"
        ) from exc
    parsed = integration.parsed_composite_g_section
    if parsed.section_sha256 != g_section.payload_sha256 or parsed.section_bytes != g_section.payload:
        raise TaskspaceMonolithicPGAError("decoded composite G is not bound to directory-owned exact bytes")
    try:
        parse_generative_taskspace_correction_v2(parsed.semantic_g_packet, predictor_state=state)
        decoded_g = apply_generative_taskspace_correction_v2(parsed.semantic_g_packet, predictor_state=state)
        ownership = derive_g_correction_ownership_v2(state, decoded_g)
        overlay = overlay_g_on_predictor_camera_y1(
            surface.frame1_camera,
            state.labels,
            decoded_g,
        )
    except (
        GenerativeTaskspaceCorrectionError,
        PredictorPreservingOverlayError,
        TaskspacePredictorV2ConsumerSeamError,
        ValueError,
    ) as exc:
        raise TaskspaceMonolithicPGAError("composite G inner semantic-G/overlay replay failed") from exc
    if ownership.changed_cells < 1:
        raise TaskspaceMonolithicPGAError("composite G inner semantic G owns no bounded cell")
    if (
        decoded_g.correction_packet_sha256 != parsed.semantic_g_packet_sha256
        or not np.array_equal(decoded_g.labels, integration.unchanged_g_semantic_labels)
        or not np.array_equal(overlay.camera_y1, integration.surface.corrected_y1_camera)
        or overlay.receipt.corrected_labels_sha256 != integration.decoded_repair.receipt.output_g_labels_sha256
        or integration.decoded_repair.receipt.packet_sha256 != parsed.repair_packet_sha256
    ):
        raise TaskspaceMonolithicPGAError("composite G semantic overlay and G8 custody diverged")
    return _DecodedG8PGV1(
        outer=outer,
        member=member,
        decoded_g=decoded_g,
        overlay=overlay,
        g8_integration=integration,
    )


def _decode_post_g8_a_once(
    archive: bytes,
    surface: Ep725EphemeralRuntimeSurfaceV2,
    *,
    expected_encoding: OuterArchiveEncoding | str | None,
    expected_archive_sha256: str | None,
    expected_member_sha256: str | None,
    max_member_bytes: int,
) -> _DecodedPostG8AOnceV1:
    pg = _decode_g8_pg_once(
        archive,
        surface,
        expected_encoding=expected_encoding,
        expected_archive_sha256=expected_archive_sha256,
        expected_member_sha256=expected_member_sha256,
        max_member_bytes=max_member_bytes,
    )
    a_section = pg.member.section(TaskspaceMonolithicPGARole.COUPLED_PREIMAGE)
    if not a_section.payload.startswith(POST_G8_A_PACKET_MAGIC):
        raise TaskspaceMonolithicPGAError("composite G8 receiver requires exact post-G8 A packet domain")
    predictor_surface = PredictorCameraPairSurfaceV1.from_ep725(surface)
    try:
        decoded = decode_post_g8_conditional_a_packet(
            a_section.payload,
            predictor_surface=predictor_surface,
            g8_integration=pg.g8_integration,
        )
        if parse_post_g8_conditional_a_receipt(decoded.receipt.to_receipt_bytes()) != decoded.receipt:
            raise PostG8ConditionalAError("post-G8 A receipt failed strict receiver parse-back")
    except PostG8ConditionalAError as exc:
        raise TaskspaceMonolithicPGAError(
            "counted post-G8 A failed exact decode against P0 and post-G8 corrected Y1"
        ) from exc
    if decoded.receipt.packet_sha256 != a_section.payload_sha256:
        raise TaskspaceMonolithicPGAError("decoded post-G8 A is not bound to directory-owned exact bytes")
    if not np.array_equal(decoded.camera_y1, pg.g8_integration.post_g8_corrected_y1):
        raise TaskspaceMonolithicPGAError("counted post-G8 A changed exact G8 corrected Y1")
    y0_hashes = tuple(_sha256(memoryview(frame).cast("B")) for frame in decoded.camera_y0)
    y1_hashes = tuple(_sha256(memoryview(frame).cast("B")) for frame in decoded.camera_y1)
    chronological_hashes = tuple(
        _sha256(memoryview(frame).cast("B"))
        for pair_frames in decoded.chronological_camera_frames
        for frame in pair_frames
    )
    if chronological_hashes[0::2] != y0_hashes or chronological_hashes[1::2] != y1_hashes:
        raise TaskspaceMonolithicPGAError("post-G8 A chronology is not exact factor-2 [Y0,Y1]")
    return _DecodedPostG8AOnceV1(
        pg=pg,
        decoded_post_g8_a=decoded,
        camera_y0_frame_sha256=y0_hashes,
        camera_y1_frame_sha256=y1_hashes,
        factor2_camera_frame_sha256_chronological=chronological_hashes,
    )


def _decode_pass_pg_once(
    archive: bytes,
    surface: Ep725EphemeralRuntimeSurfaceV2,
    *,
    expected_encoding: OuterArchiveEncoding | str | None,
    expected_archive_sha256: str | None,
    expected_member_sha256: str | None,
    max_member_bytes: int,
) -> _DecodedPassPGV1:
    """Strictly materialize nonempty PASS semantic G and its optional real G8."""

    outer, member = _parse_outer_and_member(
        archive,
        expected_encoding=expected_encoding,
        expected_archive_sha256=expected_archive_sha256,
        expected_member_sha256=expected_member_sha256,
        max_member_bytes=max_member_bytes,
    )
    if len(member.sections) == 4:
        raise TaskspaceMonolithicPGAReceiverBlocker(
            TaskspaceMonolithicPGABlockerCode.TERMINAL_QUOTIENT_CONSUMER_OWED,
            "optional T is structurally bound but has no semantic consumer",
        )
    p_section = member.section(TaskspaceMonolithicPGARole.PREDICTOR)
    g_section = member.section(TaskspaceMonolithicPGARole.GENERATIVE_CORRECTION)
    state = surface.bounded_decode.predictor_state
    if (
        p_section.payload != state.predictor_program
        or p_section.byte_length != EP725_MEMBER_BYTES
        or p_section.payload_sha256 != EP725_MEMBER_SHA256
        or not p_section.payload.startswith(_P_PACKET_MAGIC)
    ):
        raise TaskspaceMonolithicPGAError("PASS receiver P differs from exact reopened ep725 bytes")
    if not g_section.payload.startswith(PASS_SEMANTIC_G_ENVELOPE_MAGIC):
        raise TaskspaceMonolithicPGAError("PASS receiver requires exact TACPG81 counted G envelope")
    predictor_surface = PredictorCameraPairSurfaceV1.from_ep725(surface)
    try:
        decoded = decode_pass_semantic_g_envelope(
            g_section.payload,
            predictor_section_payload=p_section.payload,
            predictor_surface=predictor_surface,
            predictor_codec_id="LVLS1.v1",
        )
        if parse_pass_semantic_g_envelope_receipt(decoded.receipt.to_receipt_bytes()) != decoded.receipt:
            raise PassSemanticGError("PASS-G receipt failed strict receiver parse-back")
    except PassSemanticGError as exc:
        raise TaskspaceMonolithicPGAError("counted PASS-G failed exact source-bound decode") from exc
    if (
        decoded.receipt.envelope_sha256 != g_section.payload_sha256
        or decoded.receipt.envelope_bytes != g_section.byte_length
        or decoded.parsed_envelope.envelope_bytes != g_section.payload
    ):
        raise TaskspaceMonolithicPGAError("decoded PASS-G is not bound to directory-owned exact bytes")
    return _DecodedPassPGV1(outer=outer, member=member, decoded_pass_g=decoded)


def _decode_pass_a_once(
    archive: bytes,
    surface: Ep725EphemeralRuntimeSurfaceV2,
    *,
    expected_encoding: OuterArchiveEncoding | str | None,
    expected_archive_sha256: str | None,
    expected_member_sha256: str | None,
    max_member_bytes: int,
) -> _DecodedPassAOnceV1:
    pg = _decode_pass_pg_once(
        archive,
        surface,
        expected_encoding=expected_encoding,
        expected_archive_sha256=expected_archive_sha256,
        expected_member_sha256=expected_member_sha256,
        max_member_bytes=max_member_bytes,
    )
    a_section = pg.member.section(TaskspaceMonolithicPGARole.COUPLED_PREIMAGE)
    if not a_section.payload.startswith(PASS_CONDITIONAL_A_PACKET_MAGIC):
        raise TaskspaceMonolithicPGAError("PASS receiver requires exact TACAPG1 conditional-A packet")
    predictor_surface = PredictorCameraPairSurfaceV1.from_ep725(surface)
    try:
        decoded = decode_pass_conditional_a_packet(
            a_section.payload,
            predictor_surface=predictor_surface,
            pass_g=pg.decoded_pass_g,
        )
        if parse_pass_conditional_a_receipt(decoded.receipt.to_receipt_bytes()) != decoded.receipt:
            raise PassConditionalAError("conditional-A receipt failed strict receiver parse-back")
    except PassConditionalAError as exc:
        raise TaskspaceMonolithicPGAError(
            "counted conditional A failed exact decode against P0 and PASS-G resulting Y1"
        ) from exc
    if decoded.receipt.packet_sha256 != a_section.payload_sha256 or decoded.receipt.packet_bytes != (
        a_section.byte_length
    ):
        raise TaskspaceMonolithicPGAError("decoded conditional A is not bound to directory-owned exact bytes")
    if not np.array_equal(decoded.camera_y1, pg.decoded_pass_g.conditional_camera_y1):
        raise TaskspaceMonolithicPGAError("counted conditional A changed exact PASS-G resulting Y1")
    y0_hashes = tuple(_sha256(memoryview(frame).cast("B")) for frame in decoded.camera_y0)
    y1_hashes = tuple(_sha256(memoryview(frame).cast("B")) for frame in decoded.camera_y1)
    chronological_hashes = tuple(
        _sha256(memoryview(frame).cast("B"))
        for pair_frames in decoded.chronological_camera_frames
        for frame in pair_frames
    )
    if chronological_hashes[0::2] != y0_hashes or chronological_hashes[1::2] != y1_hashes:
        raise TaskspaceMonolithicPGAError("PASS receiver chronology is not exact factor-2 [Y0,Y1]")
    return _DecodedPassAOnceV1(
        pg=pg,
        decoded_conditional_a=decoded,
        camera_y0_frame_sha256=y0_hashes,
        camera_y1_frame_sha256=y1_hashes,
        factor2_camera_frame_sha256_chronological=chronological_hashes,
    )


def _decode_ep725_surface(
    predictor_member: bytes,
    predictor_runtime: bytes,
    *,
    pair_count: int,
    timeout_seconds: float,
) -> Ep725CountedMemberCausalSurfaceV3:
    if type(pair_count) is not int or pair_count not in {1, 2}:
        raise TaskspaceMonolithicPGAError("pair_count must be the bounded exact ep725 prefix n1 or n2")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int | float)
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0.0
    ):
        raise TaskspaceMonolithicPGAError("timeout_seconds must be finite and positive")
    try:
        causal_surface = decode_ep725_counted_member_ephemeral_surface(
            predictor_member,
            shipped_runtime=predictor_runtime,
            pair_count=pair_count,
            timeout_seconds=float(timeout_seconds),
        )
    except Ep725LevelsetPredictorAdapterError as exc:
        raise TaskspaceMonolithicPGAError("exact counted ep725 member/runtime decode failed") from exc
    if type(causal_surface) is not Ep725CountedMemberCausalSurfaceV3:
        raise TaskspaceMonolithicPGAError("ep725 adapter returned a noncanonical causal surface type")
    surface = causal_surface.ephemeral_surface
    if type(surface) is not Ep725EphemeralRuntimeSurfaceV2:
        raise TaskspaceMonolithicPGAError("causal ep725 adapter returned a noncanonical ephemeral surface type")
    if type(surface.bounded_decode) is not Ep725BoundedDecodeV2:
        raise TaskspaceMonolithicPGAError("ep725 surface contains a noncanonical bounded result type")
    if surface.bounded_decode.source_pair_ids != tuple(range(pair_count)):
        raise TaskspaceMonolithicPGAError("ep725 adapter returned a non-prefix bounded pair population")
    if causal_surface.causal_receipt.source_pair_ids != tuple(range(pair_count)):
        raise TaskspaceMonolithicPGAError("ep725 causal receipt returned a non-prefix bounded pair population")
    return causal_surface


def _validated_cached_causal_surface(
    member: ParsedTaskspaceMonolithicPGAMemberV1,
    causal_surface: Ep725CountedMemberCausalSurfaceV3,
) -> Ep725EphemeralRuntimeSurfaceV2:
    """Recheck directory-owned P before lawfully reusing one causal P decode."""

    if type(causal_surface) is not Ep725CountedMemberCausalSurfaceV3:
        raise TaskspaceMonolithicPGAError("cached-P receiver requires exact causal surface type")
    try:
        strict_causal = parse_ep725_counted_member_causal_decode_receipt(
            causal_surface.causal_receipt.to_receipt_bytes()
        )
    except Ep725LevelsetPredictorAdapterError as exc:
        raise TaskspaceMonolithicPGAError("cached-P causal receipt failed strict parse closure") from exc
    if strict_causal != causal_surface.causal_receipt:
        raise TaskspaceMonolithicPGAError("cached-P causal receipt changed on strict parse/re-emit")
    surface = causal_surface.ephemeral_surface
    if type(surface) is not Ep725EphemeralRuntimeSurfaceV2 or type(surface.bounded_decode) is not Ep725BoundedDecodeV2:
        raise TaskspaceMonolithicPGAError("cached-P causal surface contains a noncanonical ephemeral output")
    p_section = member.section(TaskspaceMonolithicPGARole.PREDICTOR)
    state = surface.predictor_state
    causal = causal_surface.causal_receipt
    if (
        p_section.payload != state.predictor_program
        or p_section.payload_sha256 != state.predictor_program_sha256
        or p_section.payload_sha256 != causal.counted_member_sha256
        or p_section.byte_length != causal.counted_member_bytes
        or p_section.byte_length != len(state.predictor_program)
        or p_section.payload_sha256 != EP725_MEMBER_SHA256
        or p_section.byte_length != EP725_MEMBER_BYTES
        or not p_section.payload.startswith(_P_PACKET_MAGIC)
    ):
        raise TaskspaceMonolithicPGAError("directory-owned P differs from cached causal P bytes/hash/length")
    bounded = surface.bounded_decode
    if (
        causal.source_pair_ids != bounded.source_pair_ids
        or causal.legacy_v2_bounded_receipt_sha256 != bounded.receipt_sha256
        or causal.legacy_v2_predictor_state_binding_sha256 != state.binding_sha256
        or causal.legacy_v2_predictor_semantic_binding_sha256 != state.semantic_binding_sha256
        or causal.chronological_camera_frame_sha256 != bounded.chronological_camera_frame_sha256
        or causal.labels_sha256 != state.labels_sha256
    ):
        raise TaskspaceMonolithicPGAError("cached causal P receipt differs from typed state/frame custody")
    return surface


def audit_ep725_taskspace_monolithic_pga_archive(
    archive: bytes,
    *,
    predictor_runtime: bytes,
    pair_count: int = MAX_BOUNDED_PAIRS,
    timeout_seconds: float = 180.0,
    expected_encoding: OuterArchiveEncoding | str | None = None,
    expected_archive_sha256: str | None = None,
    expected_member_sha256: str | None = None,
    max_member_bytes: int = _MAX_MEMBER_BYTES,
) -> TaskspaceMonolithicPGABlockerReceiptV1:
    """Reopen exact ep725 P and deterministically audit P/G/A to the blocker.

    This is intentionally an audit entrypoint, not a decode result.  It returns
    a canonical receipt with no Y0/Y1 or factor-2 output hashes.
    """

    _outer, structural_member = _parse_outer_and_member(
        archive,
        expected_encoding=expected_encoding,
        expected_archive_sha256=expected_archive_sha256,
        expected_member_sha256=expected_member_sha256,
        max_member_bytes=max_member_bytes,
    )
    predictor_member = structural_member.section(TaskspaceMonolithicPGARole.PREDICTOR).payload
    causal_surface = _decode_ep725_surface(
        predictor_member,
        predictor_runtime,
        pair_count=pair_count,
        timeout_seconds=timeout_seconds,
    )
    surface = causal_surface.ephemeral_surface
    bounded = surface.bounded_decode

    first = _audit_once(
        archive,
        surface,
        expected_encoding=expected_encoding,
        expected_archive_sha256=expected_archive_sha256,
        expected_member_sha256=expected_member_sha256,
        max_member_bytes=max_member_bytes,
    )
    second = _audit_once(
        archive,
        surface,
        expected_encoding=expected_encoding,
        expected_archive_sha256=expected_archive_sha256,
        expected_member_sha256=expected_member_sha256,
        max_member_bytes=max_member_bytes,
    )
    if first.deterministic_fingerprint != second.deterministic_fingerprint:
        raise TaskspaceMonolithicPGAError("P/G/A pre-blocker replay is not deterministic")
    source = first.a_program.source_binding
    sections = _section_receipts(first.member)
    detail = _SIGNAL_LOSS_DETAIL
    overlay_receipt: PredictorPreservingOverlayReceiptV1 = first.overlay.receipt
    corrected_frame_hashes = tuple(_sha256(memoryview(frame).cast("B")) for frame in first.overlay.camera_y1)
    return TaskspaceMonolithicPGABlockerReceiptV1(
        schema=BLOCKER_RECEIPT_SCHEMA,
        receiver_contract_id=RECEIVER_CONTRACT_ID,
        blocker_code=first.blocker_code,
        blocker_detail=detail,
        outer_encoding=first.outer.encoding,
        archive_bytes=first.outer.archive_nbytes,
        archive_sha256=first.outer.archive_sha256,
        compressed_member_bytes=first.outer.compressed_member_nbytes,
        member_bytes=first.outer.member_nbytes,
        member_sha256=first.outer.member_sha256,
        member_crc32=first.outer.member_crc32,
        sections=sections,
        section_descriptor_manifest_sha256=first.member.descriptor_manifest_sha256,
        source_pair_ids=bounded.source_pair_ids,
        predictor_state_binding_sha256=bounded.predictor_state.binding_sha256,
        predictor_semantic_binding_sha256=bounded.predictor_state.semantic_binding_sha256,
        predictor_source_archive_sha256=bounded.source_archive_sha256,
        predictor_source_runtime_sha256=bounded.source_runtime_sha256,
        predictor_bounded_receipt_sha256=bounded.as_receipt().receipt_sha256,
        predictor_causal_decode_receipt=causal_surface.causal_receipt,
        predictor_causal_decode_receipt_sha256=causal_surface.causal_receipt.receipt_sha256,
        predictor_chronological_camera_frame_sha256=bounded.chronological_camera_frame_sha256,
        predictor_frame1_camera_sha256=first.predictor_frame1_camera_sha256,
        decoded_g_labels_sha256=_sha256(memoryview(first.decoded_g.labels).cast("B")),
        g_owned_mask_sha256=first.g_owned_mask_sha256,
        g_owned_cells=first.g_owned_cells,
        g_transport_kind="NONE",
        a_mode=first.a_program.mode,
        a_source_binding_sha256=source.binding_sha256,
        a_bound_exact_y1_sha256=source.exact_y1_content_sha256,
        flat_repaint_exact_y1_sha256=first.flat_repaint_exact_y1_sha256,
        a_exact_y1_binding_matches_full_repaint=(source.exact_y1_content_sha256 == first.flat_repaint_exact_y1_sha256),
        p_g_overlay_receipt_sha256=overlay_receipt.receipt_sha256,
        p_g_base_camera_y1_sha256=overlay_receipt.base_camera_y1_sha256,
        p_g_corrected_camera_y1_sha256=overlay_receipt.corrected_camera_y1_sha256,
        p_g_corrected_camera_frame_sha256=corrected_frame_hashes,
        p_g_base_scorer_numerators_sha256=overlay_receipt.base_scorer_numerators_sha256,
        p_g_corrected_scorer_numerators_sha256=overlay_receipt.corrected_scorer_numerators_sha256,
        p_g_owned_camera_values=overlay_receipt.owned_camera_values,
        p_g_actually_changed_camera_values=overlay_receipt.actually_changed_camera_values,
    )


def _success_receipt(
    decoded: _DecodedA3OnceV1,
    causal_surface: Ep725CountedMemberCausalSurfaceV3,
) -> TaskspaceMonolithicPGAReceiverReceiptV1:
    pg = decoded.pg
    surface = causal_surface.ephemeral_surface
    bounded = surface.bounded_decode
    overlay_receipt = pg.overlay.receipt
    a_receipt = decoded.decoded_a3.receipt
    corrected_frame_hashes = tuple(_sha256(memoryview(frame).cast("B")) for frame in pg.overlay.camera_y1)
    return TaskspaceMonolithicPGAReceiverReceiptV1(
        schema=SUCCESS_RECEIPT_SCHEMA,
        receiver_contract_id=RECEIVER_CONTRACT_ID,
        outer_encoding=pg.outer.encoding,
        archive_bytes=pg.outer.archive_nbytes,
        archive_sha256=pg.outer.archive_sha256,
        compressed_member_bytes=pg.outer.compressed_member_nbytes,
        member_bytes=pg.outer.member_nbytes,
        member_sha256=pg.outer.member_sha256,
        member_crc32=pg.outer.member_crc32,
        sections=_section_receipts(pg.member),
        section_descriptor_manifest_sha256=pg.member.descriptor_manifest_sha256,
        source_pair_ids=bounded.source_pair_ids,
        predictor_state_binding_sha256=bounded.predictor_state.binding_sha256,
        predictor_semantic_binding_sha256=bounded.predictor_state.semantic_binding_sha256,
        predictor_source_archive_sha256=bounded.source_archive_sha256,
        predictor_source_runtime_sha256=bounded.source_runtime_sha256,
        predictor_bounded_receipt_sha256=bounded.receipt_sha256,
        predictor_causal_decode_receipt=causal_surface.causal_receipt,
        predictor_causal_decode_receipt_sha256=causal_surface.causal_receipt.receipt_sha256,
        predictor_chronological_camera_frame_sha256=bounded.chronological_camera_frame_sha256,
        predictor_frame0_camera_sha256=bounded.chronological_camera_frame_sha256[0::2],
        predictor_frame1_camera_sha256=pg.predictor_frame1_camera_sha256,
        decoded_g_labels_sha256=_sha256(memoryview(pg.decoded_g.labels).cast("B")),
        g_owned_mask_sha256=pg.g_owned_mask_sha256,
        g_owned_cells=pg.g_owned_cells,
        g_transport_kind="NONE",
        p_g_overlay_receipt_sha256=overlay_receipt.receipt_sha256,
        p_g_base_camera_y1_sha256=overlay_receipt.base_camera_y1_sha256,
        p_g_corrected_camera_y1_sha256=overlay_receipt.corrected_camera_y1_sha256,
        p_g_corrected_camera_frame_sha256=corrected_frame_hashes,
        p_g_base_scorer_numerators_sha256=overlay_receipt.base_scorer_numerators_sha256,
        p_g_corrected_scorer_numerators_sha256=overlay_receipt.corrected_scorer_numerators_sha256,
        p_g_owned_camera_values=overlay_receipt.owned_camera_values,
        p_g_actually_changed_camera_values=overlay_receipt.actually_changed_camera_values,
        a_receipt=a_receipt,
        a_receipt_sha256=a_receipt.receipt_sha256,
        camera_y0_frame_sha256=decoded.camera_y0_frame_sha256,
        camera_y1_frame_sha256=decoded.camera_y1_frame_sha256,
        factor2_camera_frame_sha256_chronological=decoded.factor2_camera_frame_sha256_chronological,
    )


def _g8_success_receipt(
    decoded: _DecodedPostG8AOnceV1,
    causal_surface: Ep725CountedMemberCausalSurfaceV3,
) -> TaskspaceMonolithicPGAG8ReceiverReceiptV1:
    pg = decoded.pg
    integration = pg.g8_integration
    parsed = integration.parsed_composite_g_section
    overlay_receipt = pg.overlay.receipt
    g8_receipt = integration.decoded_repair.receipt
    post_a_receipt = decoded.decoded_post_g8_a.receipt
    return TaskspaceMonolithicPGAG8ReceiverReceiptV1(
        schema=G8_SUCCESS_RECEIPT_SCHEMA,
        receiver_contract_id=G8_RECEIVER_CONTRACT_ID,
        outer_encoding=pg.outer.encoding,
        archive_bytes=pg.outer.archive_nbytes,
        archive_sha256=pg.outer.archive_sha256,
        compressed_member_bytes=pg.outer.compressed_member_nbytes,
        member_bytes=pg.outer.member_nbytes,
        member_sha256=pg.outer.member_sha256,
        member_crc32=pg.outer.member_crc32,
        sections=_section_receipts(pg.member),
        section_descriptor_manifest_sha256=pg.member.descriptor_manifest_sha256,
        source_pair_ids=causal_surface.causal_receipt.source_pair_ids,
        predictor_causal_decode_receipt=causal_surface.causal_receipt,
        predictor_causal_decode_receipt_sha256=causal_surface.causal_receipt.receipt_sha256,
        inner_semantic_g_packet_sha256=parsed.semantic_g_packet_sha256,
        inner_g8_repair_packet_sha256=parsed.repair_packet_sha256,
        semantic_g_labels_sha256=_sha256(memoryview(pg.decoded_g.labels).cast("B")),
        semantic_overlay_receipt=overlay_receipt,
        semantic_overlay_receipt_sha256=overlay_receipt.receipt_sha256,
        pre_g8_corrected_y1_sha256=overlay_receipt.corrected_camera_y1_sha256,
        g8_receipt=g8_receipt,
        g8_receipt_sha256=g8_receipt.receipt_sha256,
        post_g8_corrected_y1_sha256=g8_receipt.output_camera_y1_sha256,
        post_g8_a_receipt=post_a_receipt,
        post_g8_a_receipt_sha256=post_a_receipt.receipt_sha256,
        camera_y0_frame_sha256=decoded.camera_y0_frame_sha256,
        camera_y1_frame_sha256=decoded.camera_y1_frame_sha256,
        factor2_camera_frame_sha256_chronological=decoded.factor2_camera_frame_sha256_chronological,
    )


def _pass_success_receipt(
    decoded: _DecodedPassAOnceV1,
    causal_surface: Ep725CountedMemberCausalSurfaceV3,
) -> TaskspaceMonolithicPGAPassReceiverReceiptV1:
    pg = decoded.pg
    pass_g_receipt = pg.decoded_pass_g.receipt
    a_receipt = decoded.decoded_conditional_a.receipt
    return TaskspaceMonolithicPGAPassReceiverReceiptV1(
        schema=PASS_SUCCESS_RECEIPT_SCHEMA,
        receiver_contract_id=PASS_RECEIVER_CONTRACT_ID,
        outer_encoding=pg.outer.encoding,
        archive_bytes=pg.outer.archive_nbytes,
        archive_sha256=pg.outer.archive_sha256,
        compressed_member_bytes=pg.outer.compressed_member_nbytes,
        member_bytes=pg.outer.member_nbytes,
        member_sha256=pg.outer.member_sha256,
        member_crc32=pg.outer.member_crc32,
        sections=_section_receipts(pg.member),
        section_descriptor_manifest_sha256=pg.member.descriptor_manifest_sha256,
        source_pair_ids=causal_surface.causal_receipt.source_pair_ids,
        predictor_causal_decode_receipt=causal_surface.causal_receipt,
        predictor_causal_decode_receipt_sha256=causal_surface.causal_receipt.receipt_sha256,
        pass_g_receipt=pass_g_receipt,
        pass_g_receipt_sha256=pass_g_receipt.receipt_sha256,
        conditional_a_receipt=a_receipt,
        conditional_a_receipt_sha256=a_receipt.receipt_sha256,
        camera_y0_frame_sha256=decoded.camera_y0_frame_sha256,
        camera_y1_frame_sha256=decoded.camera_y1_frame_sha256,
        factor2_camera_frame_sha256_chronological=decoded.factor2_camera_frame_sha256_chronological,
        optional_g8_repair_present=pass_g_receipt.mode == "PASS_THEN_G8_V1",
    )


def _receive_legacy_a3_from_causal_surface(
    archive: bytes,
    causal_surface: Ep725CountedMemberCausalSurfaceV3,
    *,
    expected_encoding: OuterArchiveEncoding | str | None,
    expected_archive_sha256: str | None,
    expected_member_sha256: str | None,
    max_member_bytes: int,
) -> DecodedTaskspaceMonolithicPGAV1:
    surface = causal_surface.ephemeral_surface
    decode_kwargs = {
        "expected_encoding": expected_encoding,
        "expected_archive_sha256": expected_archive_sha256,
        "expected_member_sha256": expected_member_sha256,
        "max_member_bytes": max_member_bytes,
    }
    first = _decode_a3_once(archive, surface, **decode_kwargs)
    second = _decode_a3_once(archive, surface, **decode_kwargs)
    if first.deterministic_fingerprint != second.deterministic_fingerprint:
        raise TaskspaceMonolithicPGAError("full P/G/A3 receiver replay changed exact receipts or hashes")
    for field in (
        "camera_y0",
        "camera_y1",
        "chronological_camera_frames",
        "owned_scorer_mask",
        "owned_camera_mask",
    ):
        if not np.array_equal(getattr(first.decoded_a3, field), getattr(second.decoded_a3, field)):
            raise TaskspaceMonolithicPGAError(f"full P/G/A3 receiver replay changed {field}")
    receipt = _success_receipt(first, causal_surface)
    if parse_taskspace_monolithic_pga_receiver_receipt(receipt.to_receipt_bytes()) != receipt:
        raise TaskspaceMonolithicPGAError("full P/G/A3 receipt failed strict parse/re-emit closure")
    return DecodedTaskspaceMonolithicPGAV1(decoded_a3=first.decoded_a3, receipt=receipt)


def _receive_g8_from_causal_surface(
    archive: bytes,
    causal_surface: Ep725CountedMemberCausalSurfaceV3,
    *,
    expected_encoding: OuterArchiveEncoding | str | None,
    expected_archive_sha256: str | None,
    expected_member_sha256: str | None,
    max_member_bytes: int,
) -> DecodedTaskspaceMonolithicPGAG8V1:
    surface = causal_surface.ephemeral_surface
    decode_kwargs = {
        "expected_encoding": expected_encoding,
        "expected_archive_sha256": expected_archive_sha256,
        "expected_member_sha256": expected_member_sha256,
        "max_member_bytes": max_member_bytes,
    }
    first = _decode_post_g8_a_once(archive, surface, **decode_kwargs)
    second = _decode_post_g8_a_once(archive, surface, **decode_kwargs)
    if first.deterministic_fingerprint != second.deterministic_fingerprint:
        raise TaskspaceMonolithicPGAError("composite G8/post-G8 A replay changed receipts or hashes")
    for field in (
        "camera_y0",
        "camera_y1",
        "chronological_camera_frames",
        "owned_scorer_mask",
        "owned_camera_mask",
    ):
        if not np.array_equal(
            getattr(first.decoded_post_g8_a, field),
            getattr(second.decoded_post_g8_a, field),
        ):
            raise TaskspaceMonolithicPGAError(f"composite G8/post-G8 A replay changed {field}")
    receipt = _g8_success_receipt(first, causal_surface)
    if parse_taskspace_monolithic_pga_g8_receiver_receipt(receipt.to_receipt_bytes()) != receipt:
        raise TaskspaceMonolithicPGAError("composite G8/post-G8 A receipt failed strict parse closure")
    return DecodedTaskspaceMonolithicPGAG8V1(
        decoded_post_g8_a=first.decoded_post_g8_a,
        receipt=receipt,
    )


def _receive_pass_from_causal_surface(
    archive: bytes,
    causal_surface: Ep725CountedMemberCausalSurfaceV3,
    *,
    expected_encoding: OuterArchiveEncoding | str | None,
    expected_archive_sha256: str | None,
    expected_member_sha256: str | None,
    max_member_bytes: int,
) -> DecodedTaskspaceMonolithicPGAPassV1:
    surface = causal_surface.ephemeral_surface
    decode_kwargs = {
        "expected_encoding": expected_encoding,
        "expected_archive_sha256": expected_archive_sha256,
        "expected_member_sha256": expected_member_sha256,
        "max_member_bytes": max_member_bytes,
    }
    first = _decode_pass_a_once(archive, surface, **decode_kwargs)
    second = _decode_pass_a_once(archive, surface, **decode_kwargs)
    if first.deterministic_fingerprint != second.deterministic_fingerprint:
        raise TaskspaceMonolithicPGAError("PASS-G/conditional-A replay changed receipts or hashes")
    for field in (
        "camera_y0",
        "camera_y1",
        "chronological_camera_frames",
        "owned_scorer_mask",
        "owned_camera_mask",
    ):
        if not np.array_equal(
            getattr(first.decoded_conditional_a, field),
            getattr(second.decoded_conditional_a, field),
        ):
            raise TaskspaceMonolithicPGAError(f"PASS-G/conditional-A replay changed {field}")
    receipt = _pass_success_receipt(first, causal_surface)
    if parse_taskspace_monolithic_pga_pass_receiver_receipt(receipt.to_receipt_bytes()) != receipt:
        raise TaskspaceMonolithicPGAError("PASS-G/conditional-A receipt failed strict parse closure")
    return DecodedTaskspaceMonolithicPGAPassV1(
        decoded_conditional_a=first.decoded_conditional_a,
        receipt=receipt,
    )


def receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
    archive: bytes,
    *,
    causal_surface: Ep725CountedMemberCausalSurfaceV3,
    expected_encoding: OuterArchiveEncoding | str | None = None,
    expected_archive_sha256: str | None = None,
    expected_member_sha256: str | None = None,
    max_member_bytes: int = _MAX_MEMBER_BYTES,
) -> DecodedTaskspaceMonolithicPGAV1 | DecodedTaskspaceMonolithicPGAG8V1 | DecodedTaskspaceMonolithicPGAPassV1:
    """Reuse typed causal P while rechecking archive-owned P on every call."""

    _outer, member = _parse_outer_and_member(
        archive,
        expected_encoding=expected_encoding,
        expected_archive_sha256=expected_archive_sha256,
        expected_member_sha256=expected_member_sha256,
        max_member_bytes=max_member_bytes,
    )
    _validated_cached_causal_surface(member, causal_surface)
    if len(member.sections) == 4:
        raise TaskspaceMonolithicPGAReceiverBlocker(
            TaskspaceMonolithicPGABlockerCode.TERMINAL_QUOTIENT_CONSUMER_OWED,
            "optional T is structurally bound but has no semantic consumer",
        )
    g_payload = member.section(TaskspaceMonolithicPGARole.GENERATIVE_CORRECTION).payload
    a_payload = member.section(TaskspaceMonolithicPGARole.COUPLED_PREIMAGE).payload
    common = {
        "expected_encoding": expected_encoding,
        "expected_archive_sha256": expected_archive_sha256,
        "expected_member_sha256": expected_member_sha256,
        "max_member_bytes": max_member_bytes,
    }
    if g_payload.startswith(_G_PACKET_MAGIC) and a_payload.startswith(A3_PACKET_MAGIC):
        return _receive_legacy_a3_from_causal_surface(archive, causal_surface, **common)
    if g_payload.startswith(COMPOSITE_G_SECTION_MAGIC) and a_payload.startswith(POST_G8_A_PACKET_MAGIC):
        return _receive_g8_from_causal_surface(archive, causal_surface, **common)
    if g_payload.startswith(PASS_SEMANTIC_G_ENVELOPE_MAGIC) and a_payload.startswith(PASS_CONDITIONAL_A_PACKET_MAGIC):
        return _receive_pass_from_causal_surface(archive, causal_surface, **common)
    if g_payload.startswith(COMPOSITE_G_SECTION_MAGIC) and a_payload.startswith(A3_PACKET_MAGIC):
        raise TaskspaceMonolithicPGAError(
            "legacy A3 is bound before G8 and cannot consume a composite post-G8 Y1 surface"
        )
    if g_payload.startswith(_G_PACKET_MAGIC) and a_payload.startswith(POST_G8_A_PACKET_MAGIC):
        raise TaskspaceMonolithicPGAError("post-G8 A requires a counted composite G8 section")
    if a_payload.startswith(PASS_CONDITIONAL_A_PACKET_MAGIC):
        raise TaskspaceMonolithicPGAError("production conditional A requires counted PASS semantic-G envelope")
    if g_payload.startswith(PASS_SEMANTIC_G_ENVELOPE_MAGIC):
        raise TaskspaceMonolithicPGAError("counted PASS semantic-G envelope requires production conditional A")
    raise TaskspaceMonolithicPGAError("counted G/A packet domains are unsupported or crossed")


def receive_ep725_taskspace_monolithic_pga_archive(
    archive: bytes,
    *,
    predictor_runtime: bytes,
    pair_count: int = MAX_BOUNDED_PAIRS,
    timeout_seconds: float = 180.0,
    expected_encoding: OuterArchiveEncoding | str | None = None,
    expected_archive_sha256: str | None = None,
    expected_member_sha256: str | None = None,
    max_member_bytes: int = _MAX_MEMBER_BYTES,
) -> DecodedTaskspaceMonolithicPGAV1 | DecodedTaskspaceMonolithicPGAG8V1 | DecodedTaskspaceMonolithicPGAPassV1:
    """Decode one admitted exact P/G/A domain and expose factor-2 bytes.

    Bare ``TACG1C + TACA3P1`` retains its legacy receipt/result contract.
    ``TACG8S1 + TACA8P1`` is the exact-semantic-G diagnostic control.
    Production ``TACPG81 + TACAPG1`` follows nonempty PASS semantic G,
    optional real G8, then source-bound conditional A. Crossed domains fail
    closed. The inherited A2 packet retains its A-scoped blocker. Dense frames
    are immutable and never persisted here.
    """

    _outer, structural_member = _parse_outer_and_member(
        archive,
        expected_encoding=expected_encoding,
        expected_archive_sha256=expected_archive_sha256,
        expected_member_sha256=expected_member_sha256,
        max_member_bytes=max_member_bytes,
    )
    predictor_member = structural_member.section(TaskspaceMonolithicPGARole.PREDICTOR).payload
    g_payload = structural_member.section(TaskspaceMonolithicPGARole.GENERATIVE_CORRECTION).payload
    a_payload = structural_member.section(TaskspaceMonolithicPGARole.COUPLED_PREIMAGE).payload
    if (
        not a_payload.startswith(A3_PACKET_MAGIC)
        and not a_payload.startswith(POST_G8_A_PACKET_MAGIC)
        and not a_payload.startswith(PASS_CONDITIONAL_A_PACKET_MAGIC)
        and g_payload.startswith(_G_PACKET_MAGIC)
    ):
        receipt = audit_ep725_taskspace_monolithic_pga_archive(
            archive,
            predictor_runtime=predictor_runtime,
            pair_count=pair_count,
            timeout_seconds=timeout_seconds,
            expected_encoding=expected_encoding,
            expected_archive_sha256=expected_archive_sha256,
            expected_member_sha256=expected_member_sha256,
            max_member_bytes=max_member_bytes,
        )
        raise TaskspaceMonolithicPGAReceiverBlocker(
            receipt.blocker_code,
            receipt.blocker_detail,
            receipt=receipt,
        )
    if g_payload.startswith(COMPOSITE_G_SECTION_MAGIC) and a_payload.startswith(A3_PACKET_MAGIC):
        raise TaskspaceMonolithicPGAError(
            "legacy A3 is bound before G8 and cannot consume a composite post-G8 Y1 surface"
        )
    if g_payload.startswith(_G_PACKET_MAGIC) and a_payload.startswith(POST_G8_A_PACKET_MAGIC):
        raise TaskspaceMonolithicPGAError("post-G8 A requires a counted composite G8 section")
    if a_payload.startswith(PASS_CONDITIONAL_A_PACKET_MAGIC) and not g_payload.startswith(
        PASS_SEMANTIC_G_ENVELOPE_MAGIC
    ):
        raise TaskspaceMonolithicPGAError("production conditional A requires counted PASS semantic-G envelope")
    if g_payload.startswith(PASS_SEMANTIC_G_ENVELOPE_MAGIC) and not a_payload.startswith(
        PASS_CONDITIONAL_A_PACKET_MAGIC
    ):
        raise TaskspaceMonolithicPGAError("counted PASS semantic-G envelope requires production conditional A")
    admitted = (
        (g_payload.startswith(_G_PACKET_MAGIC) and a_payload.startswith(A3_PACKET_MAGIC))
        or (g_payload.startswith(COMPOSITE_G_SECTION_MAGIC) and a_payload.startswith(POST_G8_A_PACKET_MAGIC))
        or (
            g_payload.startswith(PASS_SEMANTIC_G_ENVELOPE_MAGIC)
            and a_payload.startswith(PASS_CONDITIONAL_A_PACKET_MAGIC)
        )
    )
    if not admitted:
        raise TaskspaceMonolithicPGAError("counted G/A packet domains are unsupported or crossed")

    causal_surface = _decode_ep725_surface(
        predictor_member,
        predictor_runtime,
        pair_count=pair_count,
        timeout_seconds=timeout_seconds,
    )
    return receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
        archive,
        causal_surface=causal_surface,
        expected_encoding=expected_encoding,
        expected_archive_sha256=expected_archive_sha256,
        expected_member_sha256=expected_member_sha256,
        max_member_bytes=max_member_bytes,
    )


def parse_taskspace_monolithic_pga_blocker_receipt(
    payload: bytes,
) -> TaskspaceMonolithicPGABlockerReceiptV1:
    """Strictly parse, reconstruct, and byte-reemit a canonical blocker receipt."""

    value = _decode_canonical_json(payload, label="monolithic P/G/A blocker receipt")
    if not isinstance(value, Mapping):
        raise TaskspaceMonolithicPGAError("blocker receipt root must be one JSON object")
    if set(value) != set(TaskspaceMonolithicPGABlockerReceiptV1.__dataclass_fields__):
        raise TaskspaceMonolithicPGAError("blocker receipt fields differ from the closed V1 schema")
    raw_sections = value["sections"]
    if not isinstance(raw_sections, list) or len(raw_sections) != 3:
        raise TaskspaceMonolithicPGAError("blocker receipt sections must be an exact three-row JSON array")
    section_fields = set(TaskspaceMonolithicPGASectionReceiptV1.__dataclass_fields__)
    sections: list[TaskspaceMonolithicPGASectionReceiptV1] = []
    for index, row in enumerate(raw_sections):
        if not isinstance(row, Mapping) or set(row) != section_fields:
            raise TaskspaceMonolithicPGAError(f"blocker receipt section[{index}] fields changed")
        role_value = row.get("role")
        offset = row.get("offset")
        byte_length = row.get("byte_length")
        payload_sha256 = row.get("payload_sha256")
        crc32 = row.get("crc32")
        if (
            type(role_value) is not str
            or type(offset) is not int
            or type(byte_length) is not int
            or type(payload_sha256) is not str
            or type(crc32) is not int
        ):
            raise TaskspaceMonolithicPGAError(f"blocker receipt section[{index}] values are not exact types")
        try:
            sections.append(
                TaskspaceMonolithicPGASectionReceiptV1(
                    role=TaskspaceMonolithicPGARole(role_value),
                    offset=offset,
                    byte_length=byte_length,
                    payload_sha256=payload_sha256,
                    crc32=crc32,
                )
            )
        except (TypeError, ValueError) as exc:
            raise TaskspaceMonolithicPGAError(f"blocker receipt section[{index}] is not typed") from exc
    raw_causal_receipt = value["predictor_causal_decode_receipt"]
    if type(raw_causal_receipt) is not dict:
        raise TaskspaceMonolithicPGAError("blocker receipt nested causal P receipt must be one JSON object")
    try:
        causal_receipt = parse_ep725_counted_member_causal_decode_receipt(_canonical_json(raw_causal_receipt))
    except Ep725LevelsetPredictorAdapterError as exc:
        raise TaskspaceMonolithicPGAError("blocker receipt nested causal P receipt is invalid") from exc
    arguments = dict(value)
    try:
        arguments["blocker_code"] = TaskspaceMonolithicPGABlockerCode(value["blocker_code"])
        arguments["outer_encoding"] = OuterArchiveEncoding(value["outer_encoding"])
        arguments["a_mode"] = CoupledPreimageMode(value["a_mode"])
        arguments["sections"] = tuple(sections)
        arguments["predictor_causal_decode_receipt"] = causal_receipt
        arguments["source_pair_ids"] = tuple(value["source_pair_ids"])
        arguments["predictor_chronological_camera_frame_sha256"] = tuple(
            value["predictor_chronological_camera_frame_sha256"]
        )
        arguments["predictor_frame1_camera_sha256"] = tuple(value["predictor_frame1_camera_sha256"])
        arguments["p_g_corrected_camera_frame_sha256"] = tuple(value["p_g_corrected_camera_frame_sha256"])
        arguments["factor2_camera_frame_sha256_chronological"] = tuple(
            value["factor2_camera_frame_sha256_chronological"]
        )
        receipt = TaskspaceMonolithicPGABlockerReceiptV1(**arguments)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, TaskspaceMonolithicPGAError):
            raise
        raise TaskspaceMonolithicPGAError("blocker receipt contains invalid typed values") from exc
    if receipt.to_receipt_bytes() != payload:
        raise TaskspaceMonolithicPGAError("blocker receipt changed on strict typed parse-back")
    return receipt


def parse_taskspace_monolithic_pga_receiver_receipt(
    payload: bytes,
) -> TaskspaceMonolithicPGAReceiverReceiptV1:
    """Strictly reconstruct and byte-reemit one successful receiver receipt."""

    value = _decode_canonical_json(payload, label="monolithic P/G/A3 receiver receipt")
    if not isinstance(value, Mapping):
        raise TaskspaceMonolithicPGAError("receiver receipt root must be one JSON object")
    if set(value) != set(TaskspaceMonolithicPGAReceiverReceiptV1.__dataclass_fields__):
        raise TaskspaceMonolithicPGAError("receiver receipt fields differ from the closed V1 schema")

    raw_sections = value["sections"]
    if type(raw_sections) is not list or len(raw_sections) != 3:
        raise TaskspaceMonolithicPGAError("receiver receipt sections must be an exact three-row JSON array")
    section_fields = set(TaskspaceMonolithicPGASectionReceiptV1.__dataclass_fields__)
    sections: list[TaskspaceMonolithicPGASectionReceiptV1] = []
    for index, row in enumerate(raw_sections):
        if type(row) is not dict or set(row) != section_fields:
            raise TaskspaceMonolithicPGAError(f"receiver receipt section[{index}] fields changed")
        role_value = row.get("role")
        offset = row.get("offset")
        byte_length = row.get("byte_length")
        payload_sha256 = row.get("payload_sha256")
        crc32 = row.get("crc32")
        if (
            type(role_value) is not str
            or type(offset) is not int
            or type(byte_length) is not int
            or type(payload_sha256) is not str
            or type(crc32) is not int
        ):
            raise TaskspaceMonolithicPGAError(f"receiver receipt section[{index}] values changed exact types")
        try:
            sections.append(
                TaskspaceMonolithicPGASectionReceiptV1(
                    role=TaskspaceMonolithicPGARole(role_value),
                    offset=offset,
                    byte_length=byte_length,
                    payload_sha256=payload_sha256,
                    crc32=crc32,
                )
            )
        except (TypeError, ValueError) as exc:
            raise TaskspaceMonolithicPGAError(f"receiver receipt section[{index}] is not typed") from exc

    raw_a_receipt = value["a_receipt"]
    if type(raw_a_receipt) is not dict:
        raise TaskspaceMonolithicPGAError("receiver receipt nested A3 receipt must be one JSON object")
    try:
        a_receipt = parse_predictor_preserving_a3_receipt(_canonical_json(raw_a_receipt))
    except PredictorPreservingA3Error as exc:
        raise TaskspaceMonolithicPGAError("receiver receipt nested A3 receipt is invalid") from exc
    raw_causal_receipt = value["predictor_causal_decode_receipt"]
    if type(raw_causal_receipt) is not dict:
        raise TaskspaceMonolithicPGAError("receiver receipt nested causal P receipt must be one JSON object")
    try:
        causal_receipt = parse_ep725_counted_member_causal_decode_receipt(_canonical_json(raw_causal_receipt))
    except Ep725LevelsetPredictorAdapterError as exc:
        raise TaskspaceMonolithicPGAError("receiver receipt nested causal P receipt is invalid") from exc

    tuple_fields = (
        "predictor_chronological_camera_frame_sha256",
        "predictor_frame0_camera_sha256",
        "predictor_frame1_camera_sha256",
        "p_g_corrected_camera_frame_sha256",
        "camera_y0_frame_sha256",
        "camera_y1_frame_sha256",
        "factor2_camera_frame_sha256_chronological",
    )
    if type(value["source_pair_ids"]) is not list or any(
        type(pair_id) is not int for pair_id in value["source_pair_ids"]
    ):
        raise TaskspaceMonolithicPGAError("receiver receipt source_pair_ids must be exact integer JSON array")
    for field_name in tuple_fields:
        if type(value[field_name]) is not list or any(type(digest) is not str for digest in value[field_name]):
            raise TaskspaceMonolithicPGAError(f"receiver receipt {field_name} must be an exact hash JSON array")

    arguments = dict(value)
    try:
        arguments["outer_encoding"] = OuterArchiveEncoding(value["outer_encoding"])
        arguments["sections"] = tuple(sections)
        arguments["source_pair_ids"] = tuple(value["source_pair_ids"])
        arguments["predictor_causal_decode_receipt"] = causal_receipt
        arguments["a_receipt"] = a_receipt
        for field_name in tuple_fields:
            arguments[field_name] = tuple(value[field_name])
        receipt = TaskspaceMonolithicPGAReceiverReceiptV1(**arguments)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, TaskspaceMonolithicPGAError):
            raise
        raise TaskspaceMonolithicPGAError("receiver receipt contains invalid typed values") from exc
    if receipt.to_receipt_bytes() != payload:
        raise TaskspaceMonolithicPGAError("receiver receipt changed on strict typed parse-back")
    return receipt


def parse_taskspace_monolithic_pga_g8_receiver_receipt(
    payload: bytes,
) -> TaskspaceMonolithicPGAG8ReceiverReceiptV1:
    """Strictly reconstruct nested P/overlay/G8/post-A custody and re-emit."""

    value = _decode_canonical_json(payload, label="monolithic composite-G8 receiver receipt")
    if type(value) is not dict:
        raise TaskspaceMonolithicPGAError("G8 receiver receipt root must be one JSON object")
    if set(value) != set(TaskspaceMonolithicPGAG8ReceiverReceiptV1.__dataclass_fields__):
        raise TaskspaceMonolithicPGAError("G8 receiver receipt fields differ from the closed schema")
    raw_sections = value["sections"]
    if type(raw_sections) is not list or len(raw_sections) != 3:
        raise TaskspaceMonolithicPGAError("G8 receiver sections must be an exact three-row array")
    section_fields = set(TaskspaceMonolithicPGASectionReceiptV1.__dataclass_fields__)
    sections: list[TaskspaceMonolithicPGASectionReceiptV1] = []
    for index, row in enumerate(raw_sections):
        if type(row) is not dict or set(row) != section_fields:
            raise TaskspaceMonolithicPGAError(f"G8 receiver section[{index}] fields changed")
        if (
            type(row["role"]) is not str
            or type(row["offset"]) is not int
            or type(row["byte_length"]) is not int
            or type(row["payload_sha256"]) is not str
            or type(row["crc32"]) is not int
        ):
            raise TaskspaceMonolithicPGAError(f"G8 receiver section[{index}] values changed types")
        try:
            sections.append(
                TaskspaceMonolithicPGASectionReceiptV1(
                    role=TaskspaceMonolithicPGARole(row["role"]),
                    offset=row["offset"],
                    byte_length=row["byte_length"],
                    payload_sha256=row["payload_sha256"],
                    crc32=row["crc32"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise TaskspaceMonolithicPGAError(f"G8 receiver section[{index}] is not typed") from exc
    nested_specs = (
        (
            "predictor_causal_decode_receipt",
            parse_ep725_counted_member_causal_decode_receipt,
            Ep725LevelsetPredictorAdapterError,
        ),
        (
            "semantic_overlay_receipt",
            parse_predictor_preserving_overlay_receipt,
            PredictorPreservingOverlayError,
        ),
        (
            "g8_receipt",
            parse_same_class_realization_repair_receipt,
            SameClassRealizationRepairError,
        ),
        (
            "post_g8_a_receipt",
            parse_post_g8_conditional_a_receipt,
            PostG8ConditionalAError,
        ),
    )
    nested: dict[str, object] = {}
    for field, parser, error_type in nested_specs:
        raw = value[field]
        if type(raw) is not dict:
            raise TaskspaceMonolithicPGAError(f"G8 receiver nested {field} must be one object")
        try:
            nested[field] = parser(_canonical_json(raw))
        except error_type as exc:
            raise TaskspaceMonolithicPGAError(f"G8 receiver nested {field} is invalid") from exc
    if type(value["source_pair_ids"]) is not list or any(
        type(pair_id) is not int for pair_id in value["source_pair_ids"]
    ):
        raise TaskspaceMonolithicPGAError("G8 receiver source_pair_ids must be exact integers")
    tuple_fields = (
        "camera_y0_frame_sha256",
        "camera_y1_frame_sha256",
        "factor2_camera_frame_sha256_chronological",
    )
    for field in tuple_fields:
        if type(value[field]) is not list or any(type(digest) is not str for digest in value[field]):
            raise TaskspaceMonolithicPGAError(f"G8 receiver {field} must be an exact hash array")
    arguments = dict(value)
    try:
        arguments["outer_encoding"] = OuterArchiveEncoding(value["outer_encoding"])
        arguments["sections"] = tuple(sections)
        arguments["source_pair_ids"] = tuple(value["source_pair_ids"])
        arguments.update(nested)
        for field in tuple_fields:
            arguments[field] = tuple(value[field])
        receipt = TaskspaceMonolithicPGAG8ReceiverReceiptV1(**arguments)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, TaskspaceMonolithicPGAError):
            raise
        raise TaskspaceMonolithicPGAError("G8 receiver receipt contains invalid typed values") from exc
    if receipt.to_receipt_bytes() != payload:
        raise TaskspaceMonolithicPGAError("G8 receiver receipt changed on strict typed parse-back")
    return receipt


def parse_taskspace_monolithic_pga_pass_receiver_receipt(
    payload: bytes,
) -> TaskspaceMonolithicPGAPassReceiverReceiptV1:
    """Strictly reconstruct nested P/PASS-G/conditional-A custody."""

    value = _decode_canonical_json(payload, label="monolithic PASS receiver receipt")
    if type(value) is not dict:
        raise TaskspaceMonolithicPGAError("PASS receiver receipt root must be one JSON object")
    if set(value) != set(TaskspaceMonolithicPGAPassReceiverReceiptV1.__dataclass_fields__):
        raise TaskspaceMonolithicPGAError("PASS receiver receipt fields differ from the closed schema")
    raw_sections = value["sections"]
    if type(raw_sections) is not list or len(raw_sections) != 3:
        raise TaskspaceMonolithicPGAError("PASS receiver sections must be an exact three-row array")
    section_fields = set(TaskspaceMonolithicPGASectionReceiptV1.__dataclass_fields__)
    sections: list[TaskspaceMonolithicPGASectionReceiptV1] = []
    for index, row in enumerate(raw_sections):
        if type(row) is not dict or set(row) != section_fields:
            raise TaskspaceMonolithicPGAError(f"PASS receiver section[{index}] fields changed")
        if (
            type(row["role"]) is not str
            or type(row["offset"]) is not int
            or type(row["byte_length"]) is not int
            or type(row["payload_sha256"]) is not str
            or type(row["crc32"]) is not int
        ):
            raise TaskspaceMonolithicPGAError(f"PASS receiver section[{index}] values changed types")
        try:
            sections.append(
                TaskspaceMonolithicPGASectionReceiptV1(
                    role=TaskspaceMonolithicPGARole(row["role"]),
                    offset=row["offset"],
                    byte_length=row["byte_length"],
                    payload_sha256=row["payload_sha256"],
                    crc32=row["crc32"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise TaskspaceMonolithicPGAError(f"PASS receiver section[{index}] is not typed") from exc
    nested_specs = (
        (
            "predictor_causal_decode_receipt",
            parse_ep725_counted_member_causal_decode_receipt,
            Ep725LevelsetPredictorAdapterError,
        ),
        (
            "pass_g_receipt",
            parse_pass_semantic_g_envelope_receipt,
            PassSemanticGError,
        ),
        (
            "conditional_a_receipt",
            parse_pass_conditional_a_receipt,
            PassConditionalAError,
        ),
    )
    nested: dict[str, object] = {}
    for field, parser, error_type in nested_specs:
        raw = value[field]
        if type(raw) is not dict:
            raise TaskspaceMonolithicPGAError(f"PASS receiver nested {field} must be one object")
        try:
            nested[field] = parser(_canonical_json(raw))
        except error_type as exc:
            raise TaskspaceMonolithicPGAError(f"PASS receiver nested {field} is invalid") from exc
    if type(value["source_pair_ids"]) is not list or any(
        type(pair_id) is not int for pair_id in value["source_pair_ids"]
    ):
        raise TaskspaceMonolithicPGAError("PASS receiver source_pair_ids must be exact integers")
    tuple_fields = (
        "camera_y0_frame_sha256",
        "camera_y1_frame_sha256",
        "factor2_camera_frame_sha256_chronological",
    )
    for field in tuple_fields:
        if type(value[field]) is not list or any(type(digest) is not str for digest in value[field]):
            raise TaskspaceMonolithicPGAError(f"PASS receiver {field} must be an exact hash array")
    arguments = dict(value)
    try:
        arguments["outer_encoding"] = OuterArchiveEncoding(value["outer_encoding"])
        arguments["sections"] = tuple(sections)
        arguments["source_pair_ids"] = tuple(value["source_pair_ids"])
        arguments.update(nested)
        for field in tuple_fields:
            arguments[field] = tuple(value[field])
        receipt = TaskspaceMonolithicPGAPassReceiverReceiptV1(**arguments)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, TaskspaceMonolithicPGAError):
            raise
        raise TaskspaceMonolithicPGAError("PASS receiver receipt contains invalid typed values") from exc
    if receipt.to_receipt_bytes() != payload:
        raise TaskspaceMonolithicPGAError("PASS receiver receipt changed on strict typed parse-back")
    return receipt


__all__ = [
    "BLOCKER_RECEIPT_SCHEMA",
    "G8_RECEIVER_CONTRACT_ID",
    "G8_SUCCESS_RECEIPT_SCHEMA",
    "MAX_BOUNDED_PAIRS",
    "MEMBER_MAGIC",
    "MEMBER_SCHEMA",
    "OPTIONAL_ROLE_ORDER",
    "PASS_RECEIVER_CONTRACT_ID",
    "PASS_SUCCESS_RECEIPT_SCHEMA",
    "RECEIVER_CONTRACT_ID",
    "REQUIRED_ROLE_ORDER",
    "SUCCESS_RECEIPT_SCHEMA",
    "DecodedTaskspaceMonolithicPGAG8V1",
    "DecodedTaskspaceMonolithicPGAPassV1",
    "DecodedTaskspaceMonolithicPGAV1",
    "ParsedTaskspaceMonolithicPGAMemberV1",
    "TaskspaceMonolithicPGABlockerCode",
    "TaskspaceMonolithicPGABlockerReceiptV1",
    "TaskspaceMonolithicPGAError",
    "TaskspaceMonolithicPGAG8ReceiverReceiptV1",
    "TaskspaceMonolithicPGAPassReceiverReceiptV1",
    "TaskspaceMonolithicPGAReceiverBlocker",
    "TaskspaceMonolithicPGAReceiverReceiptV1",
    "TaskspaceMonolithicPGARole",
    "TaskspaceMonolithicPGASectionReceiptV1",
    "TaskspaceMonolithicPGASectionV1",
    "audit_ep725_taskspace_monolithic_pga_archive",
    "build_taskspace_monolithic_pga_archive",
    "encode_taskspace_monolithic_pga_member",
    "parse_taskspace_monolithic_pga_blocker_receipt",
    "parse_taskspace_monolithic_pga_g8_receiver_receipt",
    "parse_taskspace_monolithic_pga_member",
    "parse_taskspace_monolithic_pga_pass_receiver_receipt",
    "parse_taskspace_monolithic_pga_receiver_receipt",
    "receive_ep725_taskspace_monolithic_pga_archive",
    "receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface",
]
