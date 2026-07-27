# SPDX-License-Identifier: MIT
"""Compact P-V15 selected-actuator container and exact outer archive.

The rich G17/TSPPV2 objects are useful compiler and provenance IRs, but their
JSON identities are not decoder operands.  This module is the corresponding
compact wire layer:

* the five top-level V15 semantic member payloads are stored once;
* their original ZIP names and metadata are decoder-owned codec constants;
* counted actuators are length-delimited, typed binary operands; and
* the complete compact member is raced through the existing exact one-member
  STORE/DEFLATE outer archive codec.

The receiver reconstructs the original canonical semantic-P ZIP byte for byte
with a manual ZIP32 STORE writer, then dispatches the typed actuator.  No
semantic-P ZIP string, verbose target-custody manifest, source hash, Torch
version, or compile receipt is embedded in the compact member.  Those rich
identities remain in external build/measurement receipts bound to the exact
final archive SHA.

V1 admits a semantic-only baseline or one already-closed ``G74RA1`` role-aware
prepaint operand.  A conditional Y0|Y1 actuator needs an additive type/version
after its receiver transition is implemented; unknown types fail closed today.
"""

from __future__ import annotations

import hashlib
import io
import stat
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final, Literal

import numpy as np

from tac.optimization.direct_description_carrier_compose import DirectDescriptionError
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    RoleAwareBoundaryShearletOperandV1,
    V15RoleAwareOverlayDecoderV1,
    V15RoleAwareOverlayError,
    V15RoleAwareOverlayResultV1,
    parse_role_aware_boundary_shearlet_operand,
)
from tac.witness_dsl.taskspace_outer_archive_codec import (
    OuterArchiveEncoding,
    ParsedTaskspaceOuterArchive,
    TaskspaceOuterArchiveBuild,
    TaskspaceOuterArchiveError,
    build_taskspace_outer_archive,
    parse_taskspace_outer_archive,
)

MAGIC: Final = b"PVSA1\x00\x00\x00"
VERSION: Final = 1
SEMANTIC_MEMBER_NAMES: Final = (
    "manifest.json",
    "predictor.zip",
    "predict/movable_polygon_worldsheet.g1s",
    "render/receiver_realization.ddrp",
    "render/scorer_solved_templates.ddst",
)
SEMANTIC_MEMBER_COUNT: Final = len(SEMANTIC_MEMBER_NAMES)
MAX_ACTUATORS: Final = 16
MAX_SECTION_BYTES: Final = (1 << 32) - 1
MAX_STREAM_BATCH_PAIRS: Final = 16

_HEADER: Final = struct.Struct("<8s5IB")
_ACTUATOR_DESCRIPTOR: Final = struct.Struct("<BI")

_LOCAL_FILE_HEADER: Final = struct.Struct("<IHHHHHIIIHH")
_CENTRAL_DIRECTORY_HEADER: Final = struct.Struct("<IHHHHHHIIIHHHHHII")
_END_OF_CENTRAL_DIRECTORY: Final = struct.Struct("<IHHHHIIH")
_LOCAL_FILE_SIGNATURE: Final = 0x04034B50
_CENTRAL_DIRECTORY_SIGNATURE: Final = 0x02014B50
_END_OF_CENTRAL_DIRECTORY_SIGNATURE: Final = 0x06054B50
_ZIP_VERSION: Final = 20
_ZIP_VERSION_MADE_BY: Final = (3 << 8) | _ZIP_VERSION
_ZIP_DOS_TIME: Final = 0
_ZIP_DOS_DATE: Final = 0x0021
_ZIP_EXTERNAL_ATTR: Final = (stat.S_IFREG | 0o644) << 16

COMPACT_RECEIVER_ID: Final = "tac.pvsa.compact_v15_selected_actuator_receiver.v1"
COMPACT_WIRE_POLICY_ID: Final = "FIVE_FIXED_SEMANTIC_PAYLOADS_PLUS_ORDERED_TYPED_BINARY_ACTUATORS"
CONDITIONAL_Y0_ACTUATOR_BLOCKER: Final = "PVSA_CONDITIONAL_Y0_GIVEN_Y1_ACTUATOR_TYPE_AND_TRANSITION_OWED"
PUBLIC_INFLATE_BLOCKER: Final = "PVSA_PUBLIC_INFLATE_SH_RUNTIME_INTEGRATION_OWED"
OPEN_PRODUCT_BLOCKERS: Final = (
    CONDITIONAL_Y0_ACTUATOR_BLOCKER,
    PUBLIC_INFLATE_BLOCKER,
    "PVSA_TSPPV2_RICH_IR_TO_COMPACT_WIRE_LOWERING_RECEIPT_OWED",
    "PVSA_FULL_N600_DOUBLE_DECODE_AND_UPSTREAM_EVAL_OWED",
)


class CompactPVSAError(ValueError):
    """The compact member, semantic reconstruction, or actuator failed."""


class CompactActuatorTypeV1(IntEnum):
    """Closed V1 wire registry in exact execution order."""

    G74_ROLE_AWARE_PREPAINT = 1


V1_ACTUATOR_TRANSITION_PREFIX: Final = (CompactActuatorTypeV1.G74_ROLE_AWARE_PREPAINT,)
ACTUATOR_TRANSITION_DAG_ID: Final = "SEMANTIC_P_BASE_THEN_OPTIONAL_G74_Y1_THEN_FUTURE_CONDITIONAL_Y0_GIVEN_Y1"


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _exact_positive_int(value: object, *, label: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise CompactPVSAError(f"{label} must be an exact integer in [1,{maximum}]")
    return value


def _safe_ascii_name(name: str) -> bytes:
    try:
        encoded = name.encode("ascii")
    except UnicodeEncodeError as exc:  # pragma: no cover - constants are ASCII
        raise CompactPVSAError("semantic member codec name is not ASCII") from exc
    if not encoded or b"\\" in encoded or name.startswith("/") or ".." in name.split("/"):
        raise CompactPVSAError("semantic member codec name is unsafe")
    return encoded


def _validate_actuator_transition_order(
    actuator_types: tuple[CompactActuatorTypeV1, ...],
) -> None:
    """Require an exact prefix of the normative decoder transition DAG."""

    if type(actuator_types) is not tuple or any(type(value) is not CompactActuatorTypeV1 for value in actuator_types):
        raise CompactPVSAError("compact actuator transition types changed exact enum tuple")
    if len(set(actuator_types)) != len(actuator_types):
        raise CompactPVSAError("compact actuator transition is duplicated")
    if actuator_types != V1_ACTUATOR_TRANSITION_PREFIX[: len(actuator_types)]:
        raise CompactPVSAError("compact actuator transition is outside the normative state-machine prefix")


def _canonical_semantic_zip(payloads: tuple[bytes, ...]) -> bytes:
    """Construct the canonical multi-member STORE ZIP without ``zipfile``."""

    if (
        type(payloads) is not tuple
        or len(payloads) != SEMANTIC_MEMBER_COUNT
        or any(type(payload) is not bytes or not payload for payload in payloads)
    ):
        raise CompactPVSAError("semantic payload tuple is incomplete or non-byte")

    local_parts: list[bytes] = []
    central_parts: list[bytes] = []
    offset = 0
    for name, payload in zip(SEMANTIC_MEMBER_NAMES, payloads, strict=True):
        if len(payload) > MAX_SECTION_BYTES:
            raise CompactPVSAError("semantic payload exceeds ZIP32")
        encoded_name = _safe_ascii_name(name)
        crc32 = zlib.crc32(payload) & 0xFFFFFFFF
        local_header = _LOCAL_FILE_HEADER.pack(
            _LOCAL_FILE_SIGNATURE,
            _ZIP_VERSION,
            0,
            0,
            _ZIP_DOS_TIME,
            _ZIP_DOS_DATE,
            crc32,
            len(payload),
            len(payload),
            len(encoded_name),
            0,
        )
        local_parts.extend((local_header, encoded_name, payload))
        central_header = _CENTRAL_DIRECTORY_HEADER.pack(
            _CENTRAL_DIRECTORY_SIGNATURE,
            _ZIP_VERSION_MADE_BY,
            _ZIP_VERSION,
            0,
            0,
            _ZIP_DOS_TIME,
            _ZIP_DOS_DATE,
            crc32,
            len(payload),
            len(payload),
            len(encoded_name),
            0,
            0,
            0,
            0,
            _ZIP_EXTERNAL_ATTR,
            offset,
        )
        central_parts.extend((central_header, encoded_name))
        offset += len(local_header) + len(encoded_name) + len(payload)

    central = b"".join(central_parts)
    if offset > MAX_SECTION_BYTES or len(central) > MAX_SECTION_BYTES:
        raise CompactPVSAError("semantic ZIP directory exceeds ZIP32")
    end = _END_OF_CENTRAL_DIRECTORY.pack(
        _END_OF_CENTRAL_DIRECTORY_SIGNATURE,
        0,
        0,
        SEMANTIC_MEMBER_COUNT,
        SEMANTIC_MEMBER_COUNT,
        len(central),
        offset,
        0,
    )
    return b"".join((*local_parts, central, end))


def _read_canonical_semantic_zip(
    semantic_archive: bytes,
    *,
    maximum_semantic_archive_bytes: int,
    maximum_section_bytes: int,
) -> tuple[bytes, ...]:
    """Read the exact canonical P archive and return only member payloads."""

    import zipfile

    if type(semantic_archive) is not bytes or not semantic_archive:
        raise CompactPVSAError("semantic P must be nonempty exact bytes")
    maximum_archive = _exact_positive_int(
        maximum_semantic_archive_bytes,
        label="maximum_semantic_archive_bytes",
        maximum=MAX_SECTION_BYTES,
    )
    maximum_section = _exact_positive_int(
        maximum_section_bytes,
        label="maximum_section_bytes",
        maximum=MAX_SECTION_BYTES,
    )
    if len(semantic_archive) > maximum_archive:
        raise CompactPVSAError("semantic P exceeds caller archive ceiling")
    try:
        with zipfile.ZipFile(io.BytesIO(semantic_archive), "r") as archive:
            if archive.comment:
                raise CompactPVSAError("semantic P ZIP comment is forbidden")
            infos = archive.infolist()
            if tuple(info.filename for info in infos) != SEMANTIC_MEMBER_NAMES:
                raise CompactPVSAError("semantic P member names/order differ from PVSA V1")
            payloads: list[bytes] = []
            for info in infos:
                if (
                    info.is_dir()
                    or info.flag_bits != 0
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.create_system != 3
                    or info.create_version != _ZIP_VERSION
                    or info.extract_version != _ZIP_VERSION
                    or info.external_attr != _ZIP_EXTERNAL_ATTR
                    or info.extra
                    or info.comment
                    or info.file_size < 1
                    or info.file_size > maximum_section
                    or info.compress_size != info.file_size
                ):
                    raise CompactPVSAError("semantic P ZIP metadata differs from PVSA V1 constants")
                payloads.append(archive.read(info))
            if archive.testzip() is not None:
                raise CompactPVSAError("semantic P ZIP CRC verification failed")
    except CompactPVSAError:
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise CompactPVSAError("strict semantic P ZIP read failed") from exc
    result = tuple(payloads)
    if _canonical_semantic_zip(result) != semantic_archive:
        raise CompactPVSAError("manual canonical semantic P reconstruction differs")
    return result


@dataclass(frozen=True, slots=True)
class CompactActuatorV1:
    """One exact typed counted actuator operand."""

    actuator_type: CompactActuatorTypeV1
    payload: bytes = field(repr=False)
    operand: RoleAwareBoundaryShearletOperandV1 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.actuator_type) is not CompactActuatorTypeV1:
            raise CompactPVSAError("compact actuator type changed exact enum")
        if type(self.payload) is not bytes or not self.payload:
            raise CompactPVSAError("compact actuator payload must be nonempty bytes")
        if len(self.payload) > MAX_SECTION_BYTES:
            raise CompactPVSAError("compact actuator exceeds uint32")
        if self.actuator_type is not CompactActuatorTypeV1.G74_ROLE_AWARE_PREPAINT:
            raise CompactPVSAError("compact actuator type lacks a V1 parser")
        try:
            operand = parse_role_aware_boundary_shearlet_operand(
                self.payload,
                expected_sha256=_sha256(self.payload),
                maximum_operand_bytes=len(self.payload),
            )
        except V15RoleAwareOverlayError as exc:
            raise CompactPVSAError("G74 compact actuator strict parse failed") from exc
        object.__setattr__(self, "operand", operand)

    @property
    def sha256(self) -> str:
        return _sha256(self.payload)


@dataclass(frozen=True, slots=True)
class ParsedCompactPVSAMemberV1:
    """Exact compact member, reconstructed P, and typed actuator sequence."""

    member_bytes: bytes = field(repr=False)
    semantic_payloads: tuple[bytes, ...] = field(repr=False)
    semantic_p_archive: bytes = field(repr=False)
    actuators: tuple[CompactActuatorV1, ...]
    receiver_id: Literal["tac.pvsa.compact_v15_selected_actuator_receiver.v1"] = COMPACT_RECEIVER_ID
    wire_policy_id: Literal["FIVE_FIXED_SEMANTIC_PAYLOADS_PLUS_ORDERED_TYPED_BINARY_ACTUATORS"] = COMPACT_WIRE_POLICY_ID
    rich_compiler_ir_embedded: Literal[False] = False
    target_custody_embedded: Literal[False] = False
    decoder_source_hash_embedded: Literal[False] = False
    semantic_complete_zip_embedded: Literal[False] = False
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            type(self.member_bytes) is not bytes
            or not self.member_bytes
            or type(self.semantic_payloads) is not tuple
            or len(self.semantic_payloads) != SEMANTIC_MEMBER_COUNT
            or type(self.semantic_p_archive) is not bytes
            or not self.semantic_p_archive
            or type(self.actuators) is not tuple
            or any(type(row) is not CompactActuatorV1 for row in self.actuators)
        ):
            raise CompactPVSAError("parsed compact member lost exact typed custody")
        _validate_actuator_transition_order(tuple(row.actuator_type for row in self.actuators))
        if _canonical_semantic_zip(self.semantic_payloads) != self.semantic_p_archive:
            raise CompactPVSAError("parsed semantic payloads do not reconstruct exact P")
        if self.semantic_p_archive in self.member_bytes:
            raise CompactPVSAError("compact member physically nests the complete semantic P ZIP")
        if (
            self.receiver_id != COMPACT_RECEIVER_ID
            or self.wire_policy_id != COMPACT_WIRE_POLICY_ID
            or self.rich_compiler_ir_embedded is not False
            or self.target_custody_embedded is not False
            or self.decoder_source_hash_embedded is not False
            or self.semantic_complete_zip_embedded is not False
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
        ):
            raise CompactPVSAError("compact wire truth labels became permissive")

    @property
    def member_sha256(self) -> str:
        return _sha256(self.member_bytes)

    @property
    def semantic_p_sha256(self) -> str:
        return _sha256(self.semantic_p_archive)

    @property
    def open_product_blockers(self) -> tuple[str, ...]:
        return OPEN_PRODUCT_BLOCKERS

    def open_receiver(
        self,
        *,
        verify_member_effects: bool = True,
    ) -> CompactPVSAReceiverV1:
        """Open semantic P once for bounded streaming decode."""

        return CompactPVSAReceiverV1.open(
            self,
            verify_member_effects=verify_member_effects,
        )

    def decode_g74_pair(
        self,
        pair_index: int,
        *,
        verify_member_effects: bool = True,
    ) -> V15RoleAwareOverlayResultV1:
        if type(pair_index) is not int or not 0 <= pair_index < 600:
            raise CompactPVSAError("pair_index must be an exact n600 local index")
        if len(self.actuators) != 1 or (
            self.actuators[0].actuator_type is not CompactActuatorTypeV1.G74_ROLE_AWARE_PREPAINT
        ):
            raise CompactPVSAError("PVSA V1 decode requires exactly one G74 actuator")
        try:
            decoder = V15RoleAwareOverlayDecoderV1.open(
                self.semantic_p_archive,
                expected_archive_bytes=len(self.semantic_p_archive),
                expected_archive_sha256=self.semantic_p_sha256,
                verify_member_effects=verify_member_effects,
            )
            actuator = self.actuators[0]
            return decoder.decode(
                actuator.payload,
                expected_operand_sha256=actuator.sha256,
                maximum_operand_bytes=len(actuator.payload),
                local_pair_ids=(pair_index,),
            )
        except V15RoleAwareOverlayError as exc:
            raise CompactPVSAError("compact G74 pair decode failed") from exc

    def decode_base_pair(
        self,
        pair_index: int,
        *,
        verify_member_effects: bool = True,
    ) -> np.ndarray:
        """Render one deterministic semantic-P-only pair from a zero-actuator wire."""

        if type(pair_index) is not int or not 0 <= pair_index < 600:
            raise CompactPVSAError("pair_index must be an exact n600 local index")
        if self.actuators:
            raise CompactPVSAError("semantic baseline decode requires zero actuators")
        return self.open_receiver(
            verify_member_effects=verify_member_effects,
        ).decode_pair(pair_index)


@dataclass(frozen=True, slots=True, init=False)
class CompactPVSAReceiverV1:
    """Cached semantic receiver with bounded chronological output batches."""

    parsed: ParsedCompactPVSAMemberV1
    overlay_decoder: V15RoleAwareOverlayDecoderV1
    _parsed_identity: int
    _decoder_identity: int

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("CompactPVSAReceiverV1 must be constructed through .open()")

    @classmethod
    def open(
        cls,
        parsed: ParsedCompactPVSAMemberV1,
        *,
        verify_member_effects: bool = True,
    ) -> CompactPVSAReceiverV1:
        if type(parsed) is not ParsedCompactPVSAMemberV1:
            raise CompactPVSAError("compact receiver requires exact parsed PVSA member")
        try:
            decoder = V15RoleAwareOverlayDecoderV1.open(
                parsed.semantic_p_archive,
                expected_archive_bytes=len(parsed.semantic_p_archive),
                expected_archive_sha256=parsed.semantic_p_sha256,
                verify_member_effects=verify_member_effects,
            )
        except V15RoleAwareOverlayError as exc:
            raise CompactPVSAError("compact cached receiver failed exact semantic P open") from exc
        instance = object.__new__(cls)
        object.__setattr__(instance, "parsed", parsed)
        object.__setattr__(instance, "overlay_decoder", decoder)
        object.__setattr__(instance, "_parsed_identity", id(parsed))
        object.__setattr__(instance, "_decoder_identity", id(decoder))
        instance._validate_custody()
        return instance

    def _validate_custody(self) -> None:
        if (
            type(self.parsed) is not ParsedCompactPVSAMemberV1
            or type(self.overlay_decoder) is not V15RoleAwareOverlayDecoderV1
            or id(self.parsed) != self._parsed_identity
            or id(self.overlay_decoder) != self._decoder_identity
            or self.overlay_decoder.semantic_archive is not self.parsed.semantic_p_archive
            or self.overlay_decoder.semantic_archive_sha256 != self.parsed.semantic_p_sha256
        ):
            raise CompactPVSAError("compact cached receiver custody drifted")

    def render_camera_pair_batch(
        self,
        local_pair_ids: tuple[int, ...],
    ) -> np.ndarray:
        """Render one bounded exact batch without reopening semantic P."""

        self._validate_custody()
        if (
            type(local_pair_ids) is not tuple
            or not 1 <= len(local_pair_ids) <= MAX_STREAM_BATCH_PAIRS
            or any(type(value) is not int or not 0 <= value < 600 for value in local_pair_ids)
            or local_pair_ids != tuple(range(local_pair_ids[0], local_pair_ids[0] + len(local_pair_ids)))
        ):
            raise CompactPVSAError("compact stream batch must be 1..16 contiguous exact n600 pair IDs")
        try:
            if not self.parsed.actuators:
                first = self.overlay_decoder.receiver.render_camera_pairs(local_pair_ids)
                second = self.overlay_decoder.receiver.render_camera_pairs(local_pair_ids)
                if not np.array_equal(first, second):
                    raise CompactPVSAError("semantic baseline batch double decode differs")
                camera_pairs = first
            else:
                if (
                    len(self.parsed.actuators) != 1
                    or self.parsed.actuators[0].actuator_type is not CompactActuatorTypeV1.G74_ROLE_AWARE_PREPAINT
                ):
                    raise CompactPVSAError("compact stream transition is unsupported")
                actuator = self.parsed.actuators[0]
                decoded = self.overlay_decoder.decode(
                    actuator.payload,
                    expected_operand_sha256=actuator.sha256,
                    maximum_operand_bytes=len(actuator.payload),
                    local_pair_ids=local_pair_ids,
                )
                camera_pairs = decoded.camera_pairs
        except (DirectDescriptionError, V15RoleAwareOverlayError) as exc:
            raise CompactPVSAError("compact cached receiver batch decode failed") from exc
        expected_shape = (
            len(local_pair_ids),
            2,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            3,
        )
        if camera_pairs.dtype != np.uint8 or camera_pairs.shape != expected_shape:
            raise CompactPVSAError("compact cached receiver changed exact camera ABI")
        result = np.ascontiguousarray(camera_pairs)
        result.setflags(write=False)
        return result

    def decode_pair(self, pair_index: int) -> np.ndarray:
        if type(pair_index) is not int:
            raise CompactPVSAError("pair_index must be an exact integer")
        return self.render_camera_pair_batch((pair_index,))[0]

    def iter_camera_pair_batches(
        self,
        *,
        batch_pairs: int = 1,
    ) -> Iterator[np.ndarray]:
        """Yield all n600 pairs chronologically with bounded live memory."""

        if type(batch_pairs) is not int or not 1 <= batch_pairs <= MAX_STREAM_BATCH_PAIRS:
            raise CompactPVSAError("batch_pairs must be an exact integer in [1,16]")
        for start in range(0, 600, batch_pairs):
            stop = min(start + batch_pairs, 600)
            yield self.render_camera_pair_batch(tuple(range(start, stop)))


def encode_compact_pvsa_member(
    *,
    semantic_payloads: tuple[bytes, ...],
    actuators: tuple[CompactActuatorV1, ...],
) -> bytes:
    """Encode the minimal fixed-semantic plus typed-actuator member."""

    if (
        type(semantic_payloads) is not tuple
        or len(semantic_payloads) != SEMANTIC_MEMBER_COUNT
        or any(type(payload) is not bytes or not payload for payload in semantic_payloads)
    ):
        raise CompactPVSAError("compact encode requires five exact semantic payloads")
    if (
        type(actuators) is not tuple
        or not 0 <= len(actuators) <= MAX_ACTUATORS
        or any(type(row) is not CompactActuatorV1 for row in actuators)
    ):
        raise CompactPVSAError("compact encode requires a bounded exact actuator tuple")
    _validate_actuator_transition_order(tuple(row.actuator_type for row in actuators))
    lengths = tuple(
        _exact_positive_int(
            len(payload),
            label="semantic payload bytes",
            maximum=MAX_SECTION_BYTES,
        )
        for payload in semantic_payloads
    )
    descriptors = b"".join(
        _ACTUATOR_DESCRIPTOR.pack(
            int(row.actuator_type),
            _exact_positive_int(
                len(row.payload),
                label="actuator payload bytes",
                maximum=MAX_SECTION_BYTES,
            ),
        )
        for row in actuators
    )
    return b"".join(
        (
            _HEADER.pack(MAGIC, *lengths, len(actuators)),
            descriptors,
            *semantic_payloads,
            *(row.payload for row in actuators),
        )
    )


def parse_compact_pvsa_member(
    member_bytes: bytes,
    *,
    maximum_member_bytes: int,
    maximum_section_bytes: int,
) -> ParsedCompactPVSAMemberV1:
    """Parse to exact payload homes, reconstruct P, and re-encode."""

    if type(member_bytes) is not bytes or not member_bytes:
        raise CompactPVSAError("compact member must be nonempty exact bytes")
    member_limit = _exact_positive_int(
        maximum_member_bytes,
        label="maximum_member_bytes",
        maximum=MAX_SECTION_BYTES,
    )
    section_limit = _exact_positive_int(
        maximum_section_bytes,
        label="maximum_section_bytes",
        maximum=MAX_SECTION_BYTES,
    )
    if len(member_bytes) > member_limit or len(member_bytes) < _HEADER.size:
        raise CompactPVSAError("compact member exceeds caller ceiling or is truncated")
    magic, *values = _HEADER.unpack_from(member_bytes)
    semantic_lengths = tuple(values[:SEMANTIC_MEMBER_COUNT])
    actuator_count = values[-1]
    if magic != MAGIC or not 0 <= actuator_count <= MAX_ACTUATORS:
        raise CompactPVSAError("compact member magic or actuator count differs")
    if any(length < 1 or length > section_limit for length in semantic_lengths):
        raise CompactPVSAError("compact semantic length is empty or exceeds ceiling")
    descriptor_stop = _HEADER.size + actuator_count * _ACTUATOR_DESCRIPTOR.size
    if descriptor_stop > len(member_bytes):
        raise CompactPVSAError("compact actuator descriptor table is truncated")
    descriptors: list[tuple[CompactActuatorTypeV1, int]] = []
    for index in range(actuator_count):
        wire, length = _ACTUATOR_DESCRIPTOR.unpack_from(
            member_bytes,
            _HEADER.size + index * _ACTUATOR_DESCRIPTOR.size,
        )
        try:
            actuator_type = CompactActuatorTypeV1(wire)
        except ValueError as exc:
            raise CompactPVSAError("compact actuator type is unknown") from exc
        if length < 1 or length > section_limit:
            raise CompactPVSAError("compact actuator length is empty or exceeds ceiling")
        descriptors.append((actuator_type, length))
    _validate_actuator_transition_order(tuple(row[0] for row in descriptors))

    cursor = descriptor_stop
    semantic_payloads: list[bytes] = []
    for length in semantic_lengths:
        stop = cursor + length
        if stop > len(member_bytes):
            raise CompactPVSAError("compact semantic payload escapes member EOF")
        semantic_payloads.append(member_bytes[cursor:stop])
        cursor = stop
    actuators: list[CompactActuatorV1] = []
    for actuator_type, length in descriptors:
        stop = cursor + length
        if stop > len(member_bytes):
            raise CompactPVSAError("compact actuator payload escapes member EOF")
        actuators.append(
            CompactActuatorV1(
                actuator_type=actuator_type,
                payload=member_bytes[cursor:stop],
            )
        )
        cursor = stop
    if cursor != len(member_bytes):
        raise CompactPVSAError("compact member has trailing or unowned bytes")
    semantic_tuple = tuple(semantic_payloads)
    actuator_tuple = tuple(actuators)
    if (
        encode_compact_pvsa_member(
            semantic_payloads=semantic_tuple,
            actuators=actuator_tuple,
        )
        != member_bytes
    ):
        raise CompactPVSAError("compact member changed on strict parse/re-encode")
    return ParsedCompactPVSAMemberV1(
        member_bytes=member_bytes,
        semantic_payloads=semantic_tuple,
        semantic_p_archive=_canonical_semantic_zip(semantic_tuple),
        actuators=actuator_tuple,
    )


@dataclass(frozen=True, slots=True)
class CompactPVSAArchiveBuildV1:
    """Both exact outer prices and their identical compact decode."""

    outer_build: TaskspaceOuterArchiveBuild
    stored: ParsedCompactPVSAMemberV1
    deflated: ParsedCompactPVSAMemberV1
    selected: ParsedCompactPVSAMemberV1
    compact_member_bytes: int
    semantic_p_archive_bytes: int
    rich_ir_bytes_avoided: int | None = None
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        if type(self.outer_build) is not TaskspaceOuterArchiveBuild:
            raise CompactPVSAError("compact archive build lost exact outer coder race")
        expected = self.stored if self.outer_build.selected.encoding is OuterArchiveEncoding.STORED else self.deflated
        if (
            self.selected != expected
            or self.stored != self.deflated
            or self.compact_member_bytes != self.outer_build.selected.member_nbytes
            or self.semantic_p_archive_bytes != len(self.selected.semantic_p_archive)
            or (
                self.rich_ir_bytes_avoided is not None
                and (type(self.rich_ir_bytes_avoided) is not int or self.rich_ir_bytes_avoided < 0)
            )
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
        ):
            raise CompactPVSAError("compact STORE/DEFLATE build custody or truth differs")


def _parse_outer_compact(
    exact: ParsedTaskspaceOuterArchive,
    *,
    maximum_member_bytes: int,
    maximum_section_bytes: int,
) -> ParsedCompactPVSAMemberV1:
    try:
        reopened = parse_taskspace_outer_archive(
            exact.archive_bytes,
            expected_encoding=exact.encoding,
            expected_archive_sha256=exact.archive_sha256,
            expected_member_sha256=exact.member_sha256,
            max_member_bytes=maximum_member_bytes,
        )
    except TaskspaceOuterArchiveError as exc:
        raise CompactPVSAError("compact outer archive strict reopen failed") from exc
    return parse_compact_pvsa_member(
        reopened.member_bytes,
        maximum_member_bytes=maximum_member_bytes,
        maximum_section_bytes=maximum_section_bytes,
    )


def build_compact_pvsa_archive(
    *,
    semantic_p_archive: bytes,
    actuator_payloads: tuple[bytes, ...],
    maximum_semantic_archive_bytes: int,
    maximum_member_bytes: int,
    maximum_section_bytes: int,
    rich_compiler_ir_bytes: int | None = None,
) -> CompactPVSAArchiveBuildV1:
    """Lower exact P plus typed operands to one member and race outer coding."""

    semantic_payloads = _read_canonical_semantic_zip(
        semantic_p_archive,
        maximum_semantic_archive_bytes=maximum_semantic_archive_bytes,
        maximum_section_bytes=maximum_section_bytes,
    )
    if type(actuator_payloads) is not tuple or len(actuator_payloads) not in (0, 1):
        raise CompactPVSAError("PVSA V1 build requires zero or one G74 actuator payload")
    actuators = tuple(
        CompactActuatorV1(
            actuator_type=CompactActuatorTypeV1.G74_ROLE_AWARE_PREPAINT,
            payload=payload,
        )
        for payload in actuator_payloads
    )
    member = encode_compact_pvsa_member(
        semantic_payloads=semantic_payloads,
        actuators=actuators,
    )
    try:
        outer = build_taskspace_outer_archive(
            member,
            max_member_bytes=maximum_member_bytes,
        )
    except TaskspaceOuterArchiveError as exc:
        raise CompactPVSAError("exact compact outer archive build failed") from exc
    stored = _parse_outer_compact(
        outer.stored,
        maximum_member_bytes=maximum_member_bytes,
        maximum_section_bytes=maximum_section_bytes,
    )
    deflated = _parse_outer_compact(
        outer.deflated,
        maximum_member_bytes=maximum_member_bytes,
        maximum_section_bytes=maximum_section_bytes,
    )
    selected = stored if outer.selected.encoding is OuterArchiveEncoding.STORED else deflated
    if (
        stored.semantic_p_archive != semantic_p_archive
        or deflated.semantic_p_archive != semantic_p_archive
        or stored.member_bytes != member
        or deflated.member_bytes != member
    ):
        raise CompactPVSAError("compact outer coding changed semantic P or member bytes")
    avoided: int | None = None
    if rich_compiler_ir_bytes is not None:
        rich = _exact_positive_int(
            rich_compiler_ir_bytes,
            label="rich_compiler_ir_bytes",
            maximum=MAX_SECTION_BYTES,
        )
        avoided = max(0, rich - sum(len(row.payload) for row in actuators))
    return CompactPVSAArchiveBuildV1(
        outer_build=outer,
        stored=stored,
        deflated=deflated,
        selected=selected,
        compact_member_bytes=len(member),
        semantic_p_archive_bytes=len(semantic_p_archive),
        rich_ir_bytes_avoided=avoided,
    )


__all__ = [
    "COMPACT_RECEIVER_ID",
    "COMPACT_WIRE_POLICY_ID",
    "CONDITIONAL_Y0_ACTUATOR_BLOCKER",
    "MAGIC",
    "OPEN_PRODUCT_BLOCKERS",
    "PUBLIC_INFLATE_BLOCKER",
    "SEMANTIC_MEMBER_NAMES",
    "CompactActuatorTypeV1",
    "CompactActuatorV1",
    "CompactPVSAArchiveBuildV1",
    "CompactPVSAError",
    "ParsedCompactPVSAMemberV1",
    "build_compact_pvsa_archive",
    "encode_compact_pvsa_member",
    "parse_compact_pvsa_member",
]
