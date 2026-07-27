# SPDX-License-Identifier: MIT
"""Additive role-aware selected-preimage program for the V15 receiver.

``TSPPV1`` is frozen.  Its analytic factor deliberately strips the donor
``BoundaryShearletAtomV1.role`` and applies a scorer-grid RGB post-paint.  This
module defines a distinct packet family instead of reinterpreting those bytes.

``TSPPV2`` contains exactly one population-global counted ``G74RA1`` operand.
That operand is already the canonical donor atom bank plus one chronological
frame selector.  The decoder passes the complete bank to G74 once, so every
role-aware atom is merged into immutable semantic ``P`` before one native
``CarrierComposeReceiverV1.render_camera_pairs`` execution.  Legacy factors
cannot be mixed into this packet and sequential factor overlays are
unrepresentable.

The packet carries no semantic archive bytes.  A bound decoder must separately
reopen exact V15 ``P`` bytes and match their SHA/length to the packet identity.
This module is a bounded receiver closure, not a public inflate integration,
candidate, score, Pose, or n600 result.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Literal

import numpy as np

import tac.witness_dsl.taskspace_selected_preimage_program_v1 as _v1
from tac.optimization.direct_description_carrier_compose import (
    CarrierComposeReceiverV1,
)
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    DECODER_CONTRACT_ID,
    DONOR_TAP_COPY_POLICY_ID,
    OPERAND_MAGIC,
    RoleAwareBoundaryShearletOperandV1,
    V15RoleAwareOverlayDecoderV1,
    V15RoleAwareOverlayError,
    V15RoleAwareOverlayResultV1,
    decoder_source_sha256,
    parse_role_aware_boundary_shearlet_operand,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    PAIR_COUNT_N600,
    ScorerTargetCustodyIdentityV1,
    SelectedPreimageFrameSelectorV1,
    V15SemanticProgramIdentityV1,
)

PROGRAM_SCHEMA: Final = "tac.taskspace_selected_preimage_program.v2"
MAGIC: Final = b"TSPPV2\x00\x00"
_HEADER: Final = struct.Struct("<8sI")
_U32_MAX: Final = (1 << 32) - 1

PREPAINT_AGGREGATION_CONTRACT: Final = "ONE_GLOBAL_G74RA1_BANK_MERGED_WITH_P_BEFORE_ONE_NATIVE_V15_RENDER"
PUBLIC_INFLATE_INTEGRATION_BLOCKER: Final = "PUBLIC_INFLATE_FLAT_V15_TSPPV2_RUNTIME_INTEGRATION_OWED"
CROSS_HOST_RUNTIME_BLOCKER: Final = "CROSS_HOST_TORCH_FLOAT32_DETERMINISM_OR_FIXED_CAMERA_BYTES_OWED"
COMPACT_RECEIVER_PACKET_BLOCKER: Final = "COMPACT_BINARY_TSPPV2_RECEIVER_PACKET_OWED"
OPEN_PRODUCT_BLOCKERS: Final = (
    PUBLIC_INFLATE_INTEGRATION_BLOCKER,
    CROSS_HOST_RUNTIME_BLOCKER,
    COMPACT_RECEIVER_PACKET_BLOCKER,
)


class TaskspaceSelectedPreimageProgramV2Error(ValueError):
    """A TSPPV2 packet, custody binding, or V15 decode failed closed."""


class SelectedPreimageFactorRoleV2(StrEnum):
    """The only counted logical factor home admitted by TSPPV2."""

    ANALYTIC_ROLE_AWARE_PREPAINT = "ANALYTIC_ROLE_AWARE_PREPAINT"


class SelectedPreimageFactorModeV2(StrEnum):
    """Closed V2 mode; no legacy V1 factor is accepted."""

    V15_ROLE_AWARE_PREPAINT_G74RA1 = "V15_ROLE_AWARE_PREPAINT_G74RA1"


class SelectedPreimageByteHomeV2(StrEnum):
    COUNTED_PACKET_FRAMING = "COUNTED_PACKET_FRAMING"
    COUNTED_ROLE_AWARE_OPERAND = "COUNTED_ROLE_AWARE_OPERAND"
    GENERIC_DECODER_CODE_FREE = "GENERIC_DECODER_CODE_FREE"
    ENCODER_ONLY_IDENTITY_NO_PAYLOAD = "ENCODER_ONLY_IDENTITY_NO_PAYLOAD"


class SelectedPreimageLineageClassV2(StrEnum):
    FRESH_ORIGINAL_SEMANTIC_COMPILE_IDENTITY = "FRESH_ORIGINAL_SEMANTIC_COMPILE_IDENTITY"
    VIDEO_DERIVED_ROLE_AWARE_ANALYTIC_FACTOR = "VIDEO_DERIVED_ROLE_AWARE_ANALYTIC_FACTOR"
    GENERIC_NON_VIDEO_DECODER = "GENERIC_NON_VIDEO_DECODER"
    ENCODER_TARGET_CUSTODY_IDENTITY_ONLY = "ENCODER_TARGET_CUSTODY_IDENTITY_ONLY"


def _sha256(payload: bytes | memoryview | np.ndarray) -> str:
    digest = hashlib.sha256()
    if type(payload) is np.ndarray:
        digest.update(memoryview(np.ascontiguousarray(payload)).cast("B"))
    else:
        digest.update(payload)
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
        raise TaskspaceSelectedPreimageProgramV2Error("TSPPV2 value is not finite canonical ASCII JSON") from exc


def _decode_canonical_object(payload: bytes, *, label: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload:
        raise TaskspaceSelectedPreimageProgramV2Error(f"{label} must be nonempty exact bytes")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TaskspaceSelectedPreimageProgramV2Error(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except TaskspaceSelectedPreimageProgramV2Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TaskspaceSelectedPreimageProgramV2Error(f"{label} is not strict ASCII JSON") from exc
    if type(value) is not dict or _canonical_json(value) != payload:
        raise TaskspaceSelectedPreimageProgramV2Error(f"{label} is not a canonical JSON object")
    return value


def _exact_keys(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise TaskspaceSelectedPreimageProgramV2Error(f"{label} fields differ from the closed schema")
    return value


def _require_int(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise TaskspaceSelectedPreimageProgramV2Error(f"{label} must be an exact integer in [{minimum},{maximum}]")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    try:
        return _v1._require_sha256(value, label=label)
    except _v1.TaskspaceSelectedPreimageProgramError as exc:
        raise TaskspaceSelectedPreimageProgramV2Error(str(exc)) from exc


def _torch_version() -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - contest runtime has Torch
        raise TaskspaceSelectedPreimageProgramV2Error("CPU Torch is required by the G74 receiver") from exc
    version = str(torch.__version__)
    if not version:
        raise TaskspaceSelectedPreimageProgramV2Error("Torch runtime did not expose a version")
    return version


def program_decoder_source_sha256() -> str:
    """Return exact generic TSPPV2 parser/dispatch source custody."""

    try:
        return _v1._source_sha256(decode_selected_preimage_pair_v2)
    except _v1.TaskspaceSelectedPreimageProgramError as exc:
        raise TaskspaceSelectedPreimageProgramV2Error(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class SelectedPreimageCompileConfigV2:
    source_pair_start: int
    pair_count: int
    maximum_packet_bytes: int = field(compare=False, repr=False)
    score_budget_receipt_sha256: str = field(compare=False, repr=False)
    budget_rule_id: str = field(compare=False, repr=False)
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        start = _require_int(
            self.source_pair_start,
            label="config.source_pair_start",
            minimum=0,
            maximum=PAIR_COUNT_N600 - 1,
        )
        count = _require_int(
            self.pair_count,
            label="config.pair_count",
            minimum=1,
            maximum=PAIR_COUNT_N600,
        )
        if start + count > PAIR_COUNT_N600:
            raise TaskspaceSelectedPreimageProgramV2Error("compile pair window escapes n600")
        _require_int(
            self.maximum_packet_bytes,
            label="config.maximum_packet_bytes",
            minimum=1,
            maximum=_U32_MAX,
        )
        _require_sha256(
            self.score_budget_receipt_sha256,
            label="config.score_budget_receipt_sha256",
        )
        try:
            _v1._require_id(self.budget_rule_id, label="config.budget_rule_id")
        except _v1.TaskspaceSelectedPreimageProgramError as exc:
            raise TaskspaceSelectedPreimageProgramV2Error(str(exc)) from exc
        if self.research_only is not True or self.score_claim is not False:
            raise TaskspaceSelectedPreimageProgramV2Error(
                "TSPPV2 compile config is research-only and cannot claim a score"
            )


@dataclass(frozen=True, slots=True)
class V15RoleAwareDecoderIdentityV2:
    """Exact generic source/runtime identity for the G74 execution path."""

    program_decoder_source_sha256: str
    g74_decoder_source_sha256: str
    torch_version: str
    decoder_contract_id: str = DECODER_CONTRACT_ID
    replacement_policy_id: str = DONOR_TAP_COPY_POLICY_ID
    prepaint_aggregation_contract: str = PREPAINT_AGGREGATION_CONTRACT
    scorer_dependency: Literal[False] = False
    video_derived_constants_in_decoder: Literal[False] = False
    cross_host_torch_parity_proven: Literal[False] = False

    def __post_init__(self) -> None:
        _require_sha256(
            self.program_decoder_source_sha256,
            label="decoder.program_decoder_source_sha256",
        )
        _require_sha256(
            self.g74_decoder_source_sha256,
            label="decoder.g74_decoder_source_sha256",
        )
        if type(self.torch_version) is not str or not self.torch_version:
            raise TaskspaceSelectedPreimageProgramV2Error("decoder Torch version must be nonempty")
        if (
            self.decoder_contract_id != DECODER_CONTRACT_ID
            or self.replacement_policy_id != DONOR_TAP_COPY_POLICY_ID
            or self.prepaint_aggregation_contract != PREPAINT_AGGREGATION_CONTRACT
            or self.scorer_dependency is not False
            or self.video_derived_constants_in_decoder is not False
            or self.cross_host_torch_parity_proven is not False
        ):
            raise TaskspaceSelectedPreimageProgramV2Error("V15 role-aware decoder identity changed closed semantics")

    @classmethod
    def current(cls) -> V15RoleAwareDecoderIdentityV2:
        return cls(
            program_decoder_source_sha256=program_decoder_source_sha256(),
            g74_decoder_source_sha256=decoder_source_sha256(),
            torch_version=_torch_version(),
        )


@dataclass(frozen=True, slots=True)
class TaskspaceSelectedPreimageFactorV2:
    """One global counted G74RA1 role-aware prepaint factor."""

    section_id: str
    source_pair_start: int
    pair_count: int
    operand_payload: bytes
    source_receipt_sha256: str
    role: SelectedPreimageFactorRoleV2 = SelectedPreimageFactorRoleV2.ANALYTIC_ROLE_AWARE_PREPAINT
    mode: SelectedPreimageFactorModeV2 = SelectedPreimageFactorModeV2.V15_ROLE_AWARE_PREPAINT_G74RA1
    byte_home: SelectedPreimageByteHomeV2 = SelectedPreimageByteHomeV2.COUNTED_ROLE_AWARE_OPERAND
    lineage_class: SelectedPreimageLineageClassV2 = (
        SelectedPreimageLineageClassV2.VIDEO_DERIVED_ROLE_AWARE_ANALYTIC_FACTOR
    )
    operand: RoleAwareBoundaryShearletOperandV1 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            _v1._require_id(self.section_id, label="factor.section_id")
        except _v1.TaskspaceSelectedPreimageProgramError as exc:
            raise TaskspaceSelectedPreimageProgramV2Error(str(exc)) from exc
        start = _require_int(
            self.source_pair_start,
            label="factor.source_pair_start",
            minimum=0,
            maximum=PAIR_COUNT_N600 - 1,
        )
        count = _require_int(
            self.pair_count,
            label="factor.pair_count",
            minimum=1,
            maximum=PAIR_COUNT_N600,
        )
        if start + count > PAIR_COUNT_N600:
            raise TaskspaceSelectedPreimageProgramV2Error("factor pair window escapes n600")
        if type(self.operand_payload) is not bytes or not self.operand_payload:
            raise TaskspaceSelectedPreimageProgramV2Error("factor operand must be nonempty exact bytes")
        _require_sha256(
            self.source_receipt_sha256,
            label="factor.source_receipt_sha256",
        )
        if (
            type(self.role) is not SelectedPreimageFactorRoleV2
            or self.role is not SelectedPreimageFactorRoleV2.ANALYTIC_ROLE_AWARE_PREPAINT
            or type(self.mode) is not SelectedPreimageFactorModeV2
            or self.mode is not SelectedPreimageFactorModeV2.V15_ROLE_AWARE_PREPAINT_G74RA1
            or type(self.byte_home) is not SelectedPreimageByteHomeV2
            or self.byte_home is not SelectedPreimageByteHomeV2.COUNTED_ROLE_AWARE_OPERAND
            or type(self.lineage_class) is not SelectedPreimageLineageClassV2
            or self.lineage_class is not SelectedPreimageLineageClassV2.VIDEO_DERIVED_ROLE_AWARE_ANALYTIC_FACTOR
        ):
            raise TaskspaceSelectedPreimageProgramV2Error(
                "V2 factor role/mode/byte-home/lineage combination is invalid"
            )
        try:
            operand = parse_role_aware_boundary_shearlet_operand(
                self.operand_payload,
                expected_sha256=_sha256(self.operand_payload),
                maximum_operand_bytes=len(self.operand_payload),
            )
        except V15RoleAwareOverlayError as exc:
            raise TaskspaceSelectedPreimageProgramV2Error("factor did not reopen as exact G74RA1") from exc
        stop = start + count
        if any(atom.pair_index < start or atom.pair_index >= stop for atom in operand.atoms):
            raise TaskspaceSelectedPreimageProgramV2Error("role-aware atom escaped the global factor pair window")
        object.__setattr__(self, "operand", operand)

    @property
    def operand_sha256(self) -> str:
        return _sha256(self.operand_payload)

    @property
    def frame_selector(self) -> SelectedPreimageFrameSelectorV1:
        return self.operand.frame_selector


@dataclass(frozen=True, slots=True)
class SelectedPreimageByteHomeRecordV2:
    section_id: str
    offset: int
    byte_length: int
    payload_sha256: str
    byte_home: SelectedPreimageByteHomeV2
    lineage_class: SelectedPreimageLineageClassV2


@dataclass(frozen=True, slots=True)
class TaskspaceSelectedPreimageProgramV2:
    """Strict TSPPV2 packet with one global role-aware native-prepaint bank."""

    semantic_program_identity: V15SemanticProgramIdentityV1
    target_custody_identity: ScorerTargetCustodyIdentityV1
    decoder_identity: V15RoleAwareDecoderIdentityV2
    compile_config: SelectedPreimageCompileConfigV2
    factor: TaskspaceSelectedPreimageFactorV2
    semantic_p_embedded: Literal[False] = False
    legacy_v1_factor_semantics_reinterpreted: Literal[False] = False
    mixed_legacy_and_role_aware_factors: Literal[False] = False
    standalone_public_inflate_closed: Literal[False] = False
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        if type(self.semantic_program_identity) is not V15SemanticProgramIdentityV1:
            raise TaskspaceSelectedPreimageProgramV2Error(
                "semantic identity must use the exact frozen V1 identity type"
            )
        if type(self.target_custody_identity) is not ScorerTargetCustodyIdentityV1:
            raise TaskspaceSelectedPreimageProgramV2Error("target custody must use the exact frozen V1 identity type")
        if type(self.decoder_identity) is not V15RoleAwareDecoderIdentityV2:
            raise TaskspaceSelectedPreimageProgramV2Error("decoder identity changed exact V2 type")
        if type(self.compile_config) is not SelectedPreimageCompileConfigV2:
            raise TaskspaceSelectedPreimageProgramV2Error("compile config changed exact V2 type")
        if type(self.factor) is not TaskspaceSelectedPreimageFactorV2:
            raise TaskspaceSelectedPreimageProgramV2Error("program requires exactly one global V2 factor")
        population = (
            self.compile_config.source_pair_start,
            self.compile_config.pair_count,
        )
        if (
            self.semantic_program_identity.source_pair_start,
            self.semantic_program_identity.pair_count,
        ) != population or (self.factor.source_pair_start, self.factor.pair_count) != population:
            raise TaskspaceSelectedPreimageProgramV2Error("semantic P, packet, and global factor pair windows differ")
        if (
            self.semantic_p_embedded is not False
            or self.legacy_v1_factor_semantics_reinterpreted is not False
            or self.mixed_legacy_and_role_aware_factors is not False
            or self.standalone_public_inflate_closed is not False
            or self.research_only is not True
            or self.score_claim is not False
        ):
            raise TaskspaceSelectedPreimageProgramV2Error("TSPPV2 truth labels became permissive")

    @property
    def packet_bytes(self) -> bytes:
        return encode_selected_preimage_program_v2(self)

    @property
    def packet_sha256(self) -> str:
        return _sha256(self.packet_bytes)

    @property
    def open_product_blockers(self) -> tuple[str, ...]:
        return OPEN_PRODUCT_BLOCKERS

    def byte_homes(self) -> tuple[SelectedPreimageByteHomeRecordV2, ...]:
        packet, header = _encode_components(self)
        body_offset = _HEADER.size + len(header)
        rows = (
            SelectedPreimageByteHomeRecordV2(
                section_id="PROGRAM_FRAMING_AND_MANIFEST",
                offset=0,
                byte_length=body_offset,
                payload_sha256=_sha256(packet[:body_offset]),
                byte_home=SelectedPreimageByteHomeV2.COUNTED_PACKET_FRAMING,
                lineage_class=SelectedPreimageLineageClassV2.FRESH_ORIGINAL_SEMANTIC_COMPILE_IDENTITY,
            ),
            SelectedPreimageByteHomeRecordV2(
                section_id=self.factor.section_id,
                offset=body_offset,
                byte_length=len(self.factor.operand_payload),
                payload_sha256=self.factor.operand_sha256,
                byte_home=self.factor.byte_home,
                lineage_class=self.factor.lineage_class,
            ),
        )
        if sum(row.byte_length for row in rows) != len(packet):
            raise TaskspaceSelectedPreimageProgramV2Error("TSPPV2 byte homes do not partition the packet")
        return rows


def _decoder_identity_row(value: V15RoleAwareDecoderIdentityV2) -> dict[str, Any]:
    return {
        "cross_host_torch_parity_proven": value.cross_host_torch_parity_proven,
        "decoder_contract_id": value.decoder_contract_id,
        "g74_decoder_source_sha256": value.g74_decoder_source_sha256,
        "prepaint_aggregation_contract": value.prepaint_aggregation_contract,
        "program_decoder_source_sha256": value.program_decoder_source_sha256,
        "replacement_policy_id": value.replacement_policy_id,
        "scorer_dependency": value.scorer_dependency,
        "torch_version": value.torch_version,
        "video_derived_constants_in_decoder": value.video_derived_constants_in_decoder,
    }


def _compile_config_row(value: SelectedPreimageCompileConfigV2) -> dict[str, Any]:
    return {
        "pair_count": value.pair_count,
        "research_only": value.research_only,
        "score_claim": value.score_claim,
        "source_pair_start": value.source_pair_start,
    }


def _factor_row(value: TaskspaceSelectedPreimageFactorV2) -> dict[str, Any]:
    return {
        "byte_home": value.byte_home.value,
        "byte_length": len(value.operand_payload),
        "frame_selector": value.frame_selector.value,
        "lineage_class": value.lineage_class.value,
        "mode": value.mode.value,
        "payload_offset": 0,
        "payload_sha256": value.operand_sha256,
        "role": value.role.value,
        "section_id": value.section_id,
        "source_pair_count": value.pair_count,
        "source_pair_start": value.source_pair_start,
        "source_receipt_sha256": value.source_receipt_sha256,
    }


def _encode_components(
    program: TaskspaceSelectedPreimageProgramV2,
) -> tuple[bytes, bytes]:
    header = _canonical_json(
        {
            "compile_config": _compile_config_row(program.compile_config),
            "decoder_identity": _decoder_identity_row(program.decoder_identity),
            "factor_section": _factor_row(program.factor),
            "legacy_v1_factor_semantics_reinterpreted": (program.legacy_v1_factor_semantics_reinterpreted),
            "mixed_legacy_and_role_aware_factors": (program.mixed_legacy_and_role_aware_factors),
            "research_only": program.research_only,
            "schema": PROGRAM_SCHEMA,
            "score_claim": program.score_claim,
            "semantic_p_embedded": program.semantic_p_embedded,
            "semantic_program_identity": _v1._semantic_identity_row(program.semantic_program_identity),
            "standalone_public_inflate_closed": (program.standalone_public_inflate_closed),
            "target_custody_identity": _v1._target_identity_row(program.target_custody_identity),
        }
    )
    if len(header) > _U32_MAX:
        raise TaskspaceSelectedPreimageProgramV2Error("TSPPV2 header exceeds uint32")
    packet = _HEADER.pack(MAGIC, len(header)) + header + program.factor.operand_payload
    return packet, header


def encode_selected_preimage_program_v2(
    program: TaskspaceSelectedPreimageProgramV2,
) -> bytes:
    if type(program) is not TaskspaceSelectedPreimageProgramV2:
        raise TaskspaceSelectedPreimageProgramV2Error("encode requires exact TaskspaceSelectedPreimageProgramV2")
    first, _ = _encode_components(program)
    second, _ = _encode_components(program)
    if first != second:
        raise TaskspaceSelectedPreimageProgramV2Error("TSPPV2 encoding is nondeterministic")
    if len(first) > program.compile_config.maximum_packet_bytes:
        raise TaskspaceSelectedPreimageProgramV2Error("TSPPV2 exceeds caller score-derived packet ceiling")
    return first


_HEADER_FIELDS: Final = {
    "compile_config",
    "decoder_identity",
    "factor_section",
    "legacy_v1_factor_semantics_reinterpreted",
    "mixed_legacy_and_role_aware_factors",
    "research_only",
    "schema",
    "score_claim",
    "semantic_p_embedded",
    "semantic_program_identity",
    "standalone_public_inflate_closed",
    "target_custody_identity",
}
_DECODER_FIELDS: Final = {
    "cross_host_torch_parity_proven",
    "decoder_contract_id",
    "g74_decoder_source_sha256",
    "prepaint_aggregation_contract",
    "program_decoder_source_sha256",
    "replacement_policy_id",
    "scorer_dependency",
    "torch_version",
    "video_derived_constants_in_decoder",
}
_CONFIG_FIELDS: Final = {
    "pair_count",
    "research_only",
    "score_claim",
    "source_pair_start",
}
_FACTOR_FIELDS: Final = {
    "byte_home",
    "byte_length",
    "frame_selector",
    "lineage_class",
    "mode",
    "payload_offset",
    "payload_sha256",
    "role",
    "section_id",
    "source_pair_count",
    "source_pair_start",
    "source_receipt_sha256",
}


def parse_selected_preimage_program_v2(
    packet: bytes,
    *,
    maximum_packet_bytes: int,
) -> TaskspaceSelectedPreimageProgramV2:
    """Strictly parse/re-emit exact TSPPV2 bytes and the nested G74RA1 bank."""

    if type(packet) is not bytes or not packet:
        raise TaskspaceSelectedPreimageProgramV2Error("TSPPV2 packet must be nonempty exact bytes")
    limit = _require_int(
        maximum_packet_bytes,
        label="maximum_packet_bytes",
        minimum=1,
        maximum=_U32_MAX,
    )
    if len(packet) > limit or len(packet) < _HEADER.size:
        raise TaskspaceSelectedPreimageProgramV2Error("TSPPV2 exceeds caller bound or is truncated")
    magic, header_bytes = _HEADER.unpack_from(packet)
    if magic != MAGIC or header_bytes < 1:
        raise TaskspaceSelectedPreimageProgramV2Error("TSPPV2 magic or header length is invalid")
    header_stop = _HEADER.size + header_bytes
    if header_stop > len(packet):
        raise TaskspaceSelectedPreimageProgramV2Error("TSPPV2 header escapes packet bytes")
    header = _decode_canonical_object(
        packet[_HEADER.size : header_stop],
        label="TSPPV2 header",
    )
    _exact_keys(header, _HEADER_FIELDS, label="TSPPV2 header")
    if header["schema"] != PROGRAM_SCHEMA:
        raise TaskspaceSelectedPreimageProgramV2Error("TSPPV2 schema mismatch")

    config_row = _exact_keys(
        header["compile_config"],
        _CONFIG_FIELDS,
        label="TSPPV2 compile config",
    )
    decoder_row = _exact_keys(
        header["decoder_identity"],
        _DECODER_FIELDS,
        label="TSPPV2 decoder identity",
    )
    factor_row = _exact_keys(
        header["factor_section"],
        _FACTOR_FIELDS,
        label="TSPPV2 factor section",
    )
    if factor_row["payload_offset"] != 0:
        raise TaskspaceSelectedPreimageProgramV2Error("global G74RA1 operand must begin at body offset zero")
    byte_length = _require_int(
        factor_row["byte_length"],
        label="factor.byte_length",
        minimum=1,
        maximum=_U32_MAX,
    )
    payload = packet[header_stop:]
    if len(payload) != byte_length:
        raise TaskspaceSelectedPreimageProgramV2Error("global G74RA1 length/EOF mismatch")
    if not payload.startswith(OPERAND_MAGIC):
        raise TaskspaceSelectedPreimageProgramV2Error("TSPPV2 body is not one G74RA1 operand")
    if _sha256(payload) != _require_sha256(
        factor_row["payload_sha256"],
        label="factor.payload_sha256",
    ):
        raise TaskspaceSelectedPreimageProgramV2Error("global G74RA1 SHA-256 mismatch")
    try:
        factor = TaskspaceSelectedPreimageFactorV2(
            section_id=factor_row["section_id"],
            source_pair_start=factor_row["source_pair_start"],
            pair_count=factor_row["source_pair_count"],
            operand_payload=payload,
            source_receipt_sha256=factor_row["source_receipt_sha256"],
            role=SelectedPreimageFactorRoleV2(factor_row["role"]),
            mode=SelectedPreimageFactorModeV2(factor_row["mode"]),
            byte_home=SelectedPreimageByteHomeV2(factor_row["byte_home"]),
            lineage_class=SelectedPreimageLineageClassV2(factor_row["lineage_class"]),
        )
        if factor.frame_selector.value != factor_row["frame_selector"]:
            raise TaskspaceSelectedPreimageProgramV2Error("factor frame selector differs from exact G74RA1")
        decoder = V15RoleAwareDecoderIdentityV2(**decoder_row)
        config = SelectedPreimageCompileConfigV2(
            **config_row,
            maximum_packet_bytes=limit,
            score_budget_receipt_sha256=_sha256(b"external TSPPV2 compile proof control is not counted"),
            budget_rule_id="external_compile_control_not_counted_v2",
        )
        semantic = _v1._parse_semantic_identity(header["semantic_program_identity"])
        target = _v1._parse_target_identity(header["target_custody_identity"])
    except TaskspaceSelectedPreimageProgramV2Error:
        raise
    except (
        _v1.TaskspaceSelectedPreimageProgramError,
        TypeError,
        ValueError,
    ) as exc:
        raise TaskspaceSelectedPreimageProgramV2Error("TSPPV2 identity, enum, or factor value is invalid") from exc
    program = TaskspaceSelectedPreimageProgramV2(
        semantic_program_identity=semantic,
        target_custody_identity=target,
        decoder_identity=decoder,
        compile_config=config,
        factor=factor,
        semantic_p_embedded=header["semantic_p_embedded"],
        legacy_v1_factor_semantics_reinterpreted=(header["legacy_v1_factor_semantics_reinterpreted"]),
        mixed_legacy_and_role_aware_factors=(header["mixed_legacy_and_role_aware_factors"]),
        standalone_public_inflate_closed=(header["standalone_public_inflate_closed"]),
        research_only=header["research_only"],
        score_claim=header["score_claim"],
    )
    if encode_selected_preimage_program_v2(program) != packet:
        raise TaskspaceSelectedPreimageProgramV2Error("TSPPV2 changed on strict parse/re-encode")
    if sum(row.byte_length for row in program.byte_homes()) != len(packet):
        raise TaskspaceSelectedPreimageProgramV2Error("TSPPV2 packet bytes lack a unique byte home")
    return program


@dataclass(frozen=True, slots=True)
class BoundV15RoleAwareSelectedPreimageDecoderV2:
    """Exact semantic-P, target, source, and runtime custody for TSPPV2."""

    semantic_identity: V15SemanticProgramIdentityV1
    target_custody_identity: ScorerTargetCustodyIdentityV1
    decoder_identity: V15RoleAwareDecoderIdentityV2
    overlay_decoder: V15RoleAwareOverlayDecoderV1

    def __post_init__(self) -> None:
        if type(self.semantic_identity) is not V15SemanticProgramIdentityV1:
            raise TaskspaceSelectedPreimageProgramV2Error("bound decoder semantic identity changed exact type")
        if type(self.target_custody_identity) is not ScorerTargetCustodyIdentityV1:
            raise TaskspaceSelectedPreimageProgramV2Error("bound decoder target identity changed exact type")
        if type(self.decoder_identity) is not V15RoleAwareDecoderIdentityV2:
            raise TaskspaceSelectedPreimageProgramV2Error("bound decoder runtime identity changed exact type")
        if type(self.overlay_decoder) is not V15RoleAwareOverlayDecoderV1:
            raise TaskspaceSelectedPreimageProgramV2Error("bound decoder requires exact G74 overlay decoder")
        if (
            self.overlay_decoder.semantic_archive_sha256 != self.semantic_identity.compiled_semantic_archive_sha256
            or len(self.overlay_decoder.semantic_archive) != self.semantic_identity.compiled_semantic_archive_bytes
        ):
            raise TaskspaceSelectedPreimageProgramV2Error("bound G74 semantic P differs from exact program identity")
        try:
            current_receiver_source = _v1._source_sha256(CarrierComposeReceiverV1)
        except _v1.TaskspaceSelectedPreimageProgramError as exc:
            raise TaskspaceSelectedPreimageProgramV2Error(str(exc)) from exc
        if self.semantic_identity.receiver_source_sha256 != current_receiver_source:
            raise TaskspaceSelectedPreimageProgramV2Error(
                "semantic P receiver source differs from exact compile identity"
            )
        current = V15RoleAwareDecoderIdentityV2.current()
        if self.decoder_identity != current:
            raise TaskspaceSelectedPreimageProgramV2Error("bound G74 source/Torch runtime differs from packet identity")


@dataclass(frozen=True, slots=True)
class SelectedPreimageDecodedPairV2:
    pair_index: int
    source_pair_id: int
    program_packet_sha256: str
    semantic_p_sha256: str
    role_aware_operand_sha256: str
    frame_selector: SelectedPreimageFrameSelectorV1
    result: V15RoleAwareOverlayResultV1
    deterministic_outer_double_decode: Literal[True] = True

    def __post_init__(self) -> None:
        _require_int(
            self.pair_index,
            label="decoded.pair_index",
            minimum=0,
            maximum=PAIR_COUNT_N600 - 1,
        )
        _require_int(
            self.source_pair_id,
            label="decoded.source_pair_id",
            minimum=0,
            maximum=PAIR_COUNT_N600 - 1,
        )
        for label in (
            "program_packet_sha256",
            "semantic_p_sha256",
            "role_aware_operand_sha256",
        ):
            _require_sha256(getattr(self, label), label=f"decoded.{label}")
        if type(self.frame_selector) is not SelectedPreimageFrameSelectorV1:
            raise TaskspaceSelectedPreimageProgramV2Error("decoded frame selector changed exact type")
        if type(self.result) is not V15RoleAwareOverlayResultV1:
            raise TaskspaceSelectedPreimageProgramV2Error("decoded result changed exact G74 type")
        receipt = self.result.receipt
        if (
            receipt.source_pair_ids != (self.source_pair_id,)
            or receipt.operand_sha256 != self.role_aware_operand_sha256
            or receipt.base_archive_sha256 != self.semantic_p_sha256
            or receipt.frame_selector != self.frame_selector.value
            or receipt.legacy_render_pairs_used
            or not receipt.realization_profile_consumed
            or not receipt.deterministic_double_decode
            or self.deterministic_outer_double_decode is not True
        ):
            raise TaskspaceSelectedPreimageProgramV2Error("decoded G74 receipt differs from TSPPV2 custody/semantics")


def _decode_once(
    program: TaskspaceSelectedPreimageProgramV2,
    pair_index: int,
    decoder: BoundV15RoleAwareSelectedPreimageDecoderV2,
) -> SelectedPreimageDecodedPairV2:
    if type(program) is not TaskspaceSelectedPreimageProgramV2:
        raise TaskspaceSelectedPreimageProgramV2Error("decode requires exact TaskspaceSelectedPreimageProgramV2")
    if type(decoder) is not BoundV15RoleAwareSelectedPreimageDecoderV2:
        raise TaskspaceSelectedPreimageProgramV2Error("decode requires exact bound V15 role-aware decoder")
    if (
        program.semantic_program_identity != decoder.semantic_identity
        or program.target_custody_identity != decoder.target_custody_identity
        or program.decoder_identity != decoder.decoder_identity
    ):
        raise TaskspaceSelectedPreimageProgramV2Error("program and bound source/runtime custody differ")
    local = _require_int(
        pair_index,
        label="pair_index",
        minimum=0,
        maximum=program.compile_config.pair_count - 1,
    )
    source_pair_id = program.compile_config.source_pair_start + local
    try:
        result = decoder.overlay_decoder.decode(
            program.factor.operand_payload,
            expected_operand_sha256=program.factor.operand_sha256,
            maximum_operand_bytes=len(program.factor.operand_payload),
            local_pair_ids=(local,),
        )
    except V15RoleAwareOverlayError as exc:
        raise TaskspaceSelectedPreimageProgramV2Error("G74 native prepaint/support decode failed") from exc
    if (
        result.receipt.operand_atom_count != len(program.factor.operand.atoms)
        or result.receipt.operand_roles != tuple(sorted({atom.role for atom in program.factor.operand.atoms}))
        or result.receipt.torch_version != program.decoder_identity.torch_version
        or result.receipt.decoder_source_sha256 != program.decoder_identity.g74_decoder_source_sha256
    ):
        raise TaskspaceSelectedPreimageProgramV2Error("G74 result lost operand role/source/runtime custody")
    return SelectedPreimageDecodedPairV2(
        pair_index=local,
        source_pair_id=source_pair_id,
        program_packet_sha256=program.packet_sha256,
        semantic_p_sha256=program.semantic_program_identity.compiled_semantic_archive_sha256,
        role_aware_operand_sha256=program.factor.operand_sha256,
        frame_selector=program.factor.frame_selector,
        result=result,
    )


def decode_selected_preimage_pair_v2(
    program: TaskspaceSelectedPreimageProgramV2,
    pair_index: int,
    decoder: BoundV15RoleAwareSelectedPreimageDecoderV2,
) -> SelectedPreimageDecodedPairV2:
    """Decode twice through G74; no legacy factor or sequential overlay exists."""

    first = _decode_once(program, pair_index, decoder)
    second = _decode_once(program, pair_index, decoder)
    if (
        first.pair_index != second.pair_index
        or first.source_pair_id != second.source_pair_id
        or first.program_packet_sha256 != second.program_packet_sha256
        or first.semantic_p_sha256 != second.semantic_p_sha256
        or first.role_aware_operand_sha256 != second.role_aware_operand_sha256
        or first.frame_selector is not second.frame_selector
        or first.result.receipt != second.result.receipt
        or first.deterministic_outer_double_decode is not second.deterministic_outer_double_decode
        or not np.array_equal(first.result.camera_pairs, second.result.camera_pairs)
        or not np.array_equal(
            first.result.changed_support_values,
            second.result.changed_support_values,
        )
        or not np.array_equal(
            first.result.owned_camera_values,
            second.result.owned_camera_values,
        )
    ):
        raise TaskspaceSelectedPreimageProgramV2Error("TSPPV2 outer double decode drifted")
    return first


__all__ = [
    "COMPACT_RECEIVER_PACKET_BLOCKER",
    "CROSS_HOST_RUNTIME_BLOCKER",
    "MAGIC",
    "OPEN_PRODUCT_BLOCKERS",
    "PREPAINT_AGGREGATION_CONTRACT",
    "PROGRAM_SCHEMA",
    "PUBLIC_INFLATE_INTEGRATION_BLOCKER",
    "BoundV15RoleAwareSelectedPreimageDecoderV2",
    "SelectedPreimageByteHomeRecordV2",
    "SelectedPreimageByteHomeV2",
    "SelectedPreimageCompileConfigV2",
    "SelectedPreimageDecodedPairV2",
    "SelectedPreimageFactorModeV2",
    "SelectedPreimageFactorRoleV2",
    "SelectedPreimageLineageClassV2",
    "TaskspaceSelectedPreimageFactorV2",
    "TaskspaceSelectedPreimageProgramV2",
    "TaskspaceSelectedPreimageProgramV2Error",
    "V15RoleAwareDecoderIdentityV2",
    "decode_selected_preimage_pair_v2",
    "encode_selected_preimage_program_v2",
    "parse_selected_preimage_program_v2",
    "program_decoder_source_sha256",
]
