# SPDX-License-Identifier: MIT
"""Externally-custodied lowering from rich TSPPV2 IR to compact PVSA1.

TSPPV2 is the compiler/provenance object.  PVSA1 is the counted receiver wire.
This adapter verifies the rich semantic, target, and decoder identities
externally, preserves the exact G74RA1 operand bytes, and emits an external
lowering receipt bound to both rich and compact archive bytes.

The compact member receives no TSPPV2 JSON or custody strings.  The adapter
also builds the same-container zero-actuator baseline so the actuator marginal
is not confused with the semantic container recode.

This is bounded pair-0, research-only receiver evidence.  It does not invoke a
scorer/evaluator, claim full-n600 decode, close public inflate, or produce a
candidate score.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

import tac.witness_dsl.taskspace_outer_archive_codec as _outer_codec
import tac.witness_dsl.taskspace_pvsa_compact_container_v1 as _pvsa
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    V15RoleAwareOverlayDecoderV1,
    V15RoleAwareOverlayError,
)
from tac.witness_dsl.taskspace_pvsa_compact_container_v1 import (
    CONDITIONAL_Y0_ACTUATOR_BLOCKER,
    PUBLIC_INFLATE_BLOCKER,
    CompactPVSAArchiveBuildV1,
    CompactPVSAError,
    build_compact_pvsa_archive,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    ScorerTargetCustodyIdentityV1,
    V15SemanticProgramIdentityV1,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v2 import (
    BoundV15RoleAwareSelectedPreimageDecoderV2,
    TaskspaceSelectedPreimageProgramV2,
    TaskspaceSelectedPreimageProgramV2Error,
    V15RoleAwareDecoderIdentityV2,
    decode_selected_preimage_pair_v2,
    encode_selected_preimage_program_v2,
    parse_selected_preimage_program_v2,
)

LOWERING_RECEIPT_SCHEMA: Final = "tac.tsppv2_to_pvsa1_external_lowering_receipt.v1"
LOWERING_CONTRACT_ID: Final = "EXACT_TSPPV2_G74RA1_TO_PVSA1_OPERAND_PRESERVING_LOWERING_V1"
FULL_N600_BLOCKER: Final = "PVSA_FULL_N600_DOUBLE_DECODE_AND_UPSTREAM_EVAL_OWED"
OPEN_LOWERING_BLOCKERS: Final = (
    CONDITIONAL_Y0_ACTUATOR_BLOCKER,
    PUBLIC_INFLATE_BLOCKER,
    FULL_N600_BLOCKER,
)

_TSPPV2_HEADER = struct.Struct("<8sI")
_SHA256_HEX_CHARS: Final = frozenset("0123456789abcdef")


class TSPPV2PVSA1LoweringError(ValueError):
    """Rich identity, compact lowering, or equivalence proof failed closed."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in _SHA256_HEX_CHARS for character in value):
        raise TSPPV2PVSA1LoweringError(f"{label} must be canonical lowercase SHA-256")
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
        raise TSPPV2PVSA1LoweringError("lowering receipt value is not canonical ASCII JSON") from exc


def _identity_sha256(value: object) -> str:
    return _sha256(_canonical_json(asdict(value)))


def _module_source_sha256(module: object) -> str:
    path_text = inspect.getsourcefile(module)
    if path_text is None:
        raise TSPPV2PVSA1LoweringError("lowering dependency lacks source-file custody")
    path = Path(path_text)
    try:
        return _sha256(path.read_bytes())
    except OSError as exc:
        raise TSPPV2PVSA1LoweringError("lowering dependency source could not be read") from exc


def lowering_source_sha256() -> str:
    return _module_source_sha256(inspect.getmodule(lowering_source_sha256))


@dataclass(frozen=True, slots=True)
class TSPPV2ToPVSA1ExternalLoweringReceiptV1:
    """Durable external proof that rich custody lowered to exact compact bytes."""

    rich_program_bytes: int
    rich_program_sha256: str
    rich_framing_bytes: int
    semantic_p_bytes: int
    semantic_p_sha256: str
    semantic_identity_sha256: str
    target_identity_sha256: str
    target_custody_receipt_sha256: str
    target_bank_sha256: str
    decoder_identity_sha256: str
    program_decoder_source_sha256: str
    g74_decoder_source_sha256: str
    torch_version: str
    operand_bytes: int
    operand_sha256: str
    frame_selector: str
    source_pair_start: int
    pair_count: int
    semantic_baseline_member_bytes: int
    semantic_baseline_member_sha256: str
    semantic_baseline_archive_bytes: int
    semantic_baseline_archive_sha256: str
    semantic_container_recode_delta_bytes: int
    compact_member_bytes: int
    compact_member_sha256: str
    compact_outer_stored_bytes: int
    compact_outer_stored_sha256: str
    compact_outer_deflated_bytes: int
    compact_outer_deflated_sha256: str
    compact_archive_bytes: int
    compact_archive_sha256: str
    actuator_same_container_marginal_bytes: int
    rich_ir_bytes_avoided: int
    rich_pair_zero_camera_sha256: str
    compact_pair_zero_camera_sha256: str
    pair_zero_execution_receipt_sha256: str
    lowering_source_sha256: str
    pvsa_source_sha256: str
    outer_archive_source_sha256: str
    schema: Literal["tac.tsppv2_to_pvsa1_external_lowering_receipt.v1"] = LOWERING_RECEIPT_SCHEMA
    lowering_contract_id: Literal["EXACT_TSPPV2_G74RA1_TO_PVSA1_OPERAND_PRESERVING_LOWERING_V1"] = LOWERING_CONTRACT_ID
    rich_parse_reemit_exact: Literal[True] = True
    operand_bytes_preserved_exactly: Literal[True] = True
    compact_store_parse_back_exact: Literal[True] = True
    compact_deflate_parse_back_exact: Literal[True] = True
    rich_identity_json_embedded_in_compact: Literal[False] = False
    target_custody_embedded_in_compact: Literal[False] = False
    decoder_identity_embedded_in_compact: Literal[False] = False
    pair_zero_camera_equality_proven: Literal[True] = True
    scorer_invoked: Literal[False] = False
    evaluator_invoked: Literal[False] = False
    public_inflate_closed: Literal[False] = False
    full_n600_decode_evidence: Literal[False] = False
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        positive_fields = (
            "rich_program_bytes",
            "rich_framing_bytes",
            "semantic_p_bytes",
            "operand_bytes",
            "pair_count",
            "semantic_baseline_member_bytes",
            "semantic_baseline_archive_bytes",
            "compact_member_bytes",
            "compact_outer_stored_bytes",
            "compact_outer_deflated_bytes",
            "compact_archive_bytes",
            "rich_ir_bytes_avoided",
        )
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 1 for name in positive_fields):
            raise TSPPV2PVSA1LoweringError("lowering receipt byte/count field is not positive")
        for name in (
            "rich_program_sha256",
            "semantic_p_sha256",
            "semantic_identity_sha256",
            "target_identity_sha256",
            "target_custody_receipt_sha256",
            "target_bank_sha256",
            "decoder_identity_sha256",
            "program_decoder_source_sha256",
            "g74_decoder_source_sha256",
            "operand_sha256",
            "semantic_baseline_member_sha256",
            "semantic_baseline_archive_sha256",
            "compact_member_sha256",
            "compact_outer_stored_sha256",
            "compact_outer_deflated_sha256",
            "compact_archive_sha256",
            "rich_pair_zero_camera_sha256",
            "compact_pair_zero_camera_sha256",
            "pair_zero_execution_receipt_sha256",
            "lowering_source_sha256",
            "pvsa_source_sha256",
            "outer_archive_source_sha256",
        ):
            _require_sha256(getattr(self, name), label=f"receipt.{name}")
        if (
            type(self.semantic_container_recode_delta_bytes) is not int
            or self.semantic_container_recode_delta_bytes
            != self.semantic_baseline_archive_bytes - self.semantic_p_bytes
            or type(self.actuator_same_container_marginal_bytes) is not int
            or self.actuator_same_container_marginal_bytes
            != self.compact_archive_bytes - self.semantic_baseline_archive_bytes
            or self.rich_ir_bytes_avoided != self.rich_program_bytes - self.operand_bytes
            or self.compact_archive_bytes
            != min(
                self.compact_outer_stored_bytes,
                self.compact_outer_deflated_bytes,
            )
            or self.rich_pair_zero_camera_sha256 != self.compact_pair_zero_camera_sha256
            or self.source_pair_start != 0
            or self.pair_count != 600
            or type(self.torch_version) is not str
            or not self.torch_version
            or type(self.frame_selector) is not str
            or not self.frame_selector
        ):
            raise TSPPV2PVSA1LoweringError("lowering receipt rate/window/runtime invariants differ")
        truth = (
            self.schema == LOWERING_RECEIPT_SCHEMA
            and self.lowering_contract_id == LOWERING_CONTRACT_ID
            and self.rich_parse_reemit_exact is True
            and self.operand_bytes_preserved_exactly is True
            and self.compact_store_parse_back_exact is True
            and self.compact_deflate_parse_back_exact is True
            and self.rich_identity_json_embedded_in_compact is False
            and self.target_custody_embedded_in_compact is False
            and self.decoder_identity_embedded_in_compact is False
            and self.pair_zero_camera_equality_proven is True
            and self.scorer_invoked is False
            and self.evaluator_invoked is False
            and self.public_inflate_closed is False
            and self.full_n600_decode_evidence is False
            and self.research_only is True
            and self.candidate_claim is False
            and self.score_claim is False
        )
        if not truth:
            raise TSPPV2PVSA1LoweringError("lowering receipt truth labels became permissive")

    def to_bytes(self) -> bytes:
        return _canonical_json(asdict(self))

    @property
    def sha256(self) -> str:
        return _sha256(self.to_bytes())


@dataclass(frozen=True, slots=True)
class TSPPV2ToPVSA1LoweringV1:
    """Exact rich/compiler and compact/wire objects plus external receipt."""

    rich_program_packet: bytes = field(repr=False)
    rich_program: TaskspaceSelectedPreimageProgramV2
    semantic_p_archive: bytes = field(repr=False)
    semantic_baseline: CompactPVSAArchiveBuildV1
    compact_actuated: CompactPVSAArchiveBuildV1
    receipt: TSPPV2ToPVSA1ExternalLoweringReceiptV1
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            type(self.rich_program_packet) is not bytes
            or not self.rich_program_packet
            or type(self.rich_program) is not TaskspaceSelectedPreimageProgramV2
            or encode_selected_preimage_program_v2(self.rich_program) != self.rich_program_packet
            or type(self.semantic_p_archive) is not bytes
            or not self.semantic_p_archive
            or type(self.semantic_baseline) is not CompactPVSAArchiveBuildV1
            or type(self.compact_actuated) is not CompactPVSAArchiveBuildV1
            or type(self.receipt) is not TSPPV2ToPVSA1ExternalLoweringReceiptV1
        ):
            raise TSPPV2PVSA1LoweringError("lowering result lost exact rich/compact custody")
        actuator = self.compact_actuated.selected.actuators
        if (
            self.semantic_baseline.selected.actuators
            or len(actuator) != 1
            or actuator[0].payload != self.rich_program.factor.operand_payload
            or actuator[0].sha256 != self.rich_program.factor.operand_sha256
            or self.receipt.rich_program_sha256 != _sha256(self.rich_program_packet)
            or self.receipt.compact_archive_sha256 != self.compact_actuated.outer_build.selected.archive_sha256
            or self.receipt.semantic_baseline_archive_sha256
            != self.semantic_baseline.outer_build.selected.archive_sha256
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
        ):
            raise TSPPV2PVSA1LoweringError("lowering result operand/rate/truth custody differs")

    @property
    def open_lowering_blockers(self) -> tuple[str, ...]:
        return OPEN_LOWERING_BLOCKERS


def _rich_header_bytes(packet: bytes) -> bytes:
    if len(packet) < _TSPPV2_HEADER.size:
        raise TSPPV2PVSA1LoweringError("rich TSPPV2 packet is truncated")
    _, header_length = _TSPPV2_HEADER.unpack_from(packet)
    stop = _TSPPV2_HEADER.size + header_length
    if header_length < 1 or stop > len(packet):
        raise TSPPV2PVSA1LoweringError("rich TSPPV2 header length escapes packet")
    return packet[_TSPPV2_HEADER.size : stop]


def _require_external_identities(
    program: TaskspaceSelectedPreimageProgramV2,
    *,
    expected_semantic_identity: V15SemanticProgramIdentityV1,
    expected_target_identity: ScorerTargetCustodyIdentityV1,
    expected_decoder_identity: V15RoleAwareDecoderIdentityV2,
) -> None:
    if (
        type(expected_semantic_identity) is not V15SemanticProgramIdentityV1
        or type(expected_target_identity) is not ScorerTargetCustodyIdentityV1
        or type(expected_decoder_identity) is not V15RoleAwareDecoderIdentityV2
        or program.semantic_program_identity != expected_semantic_identity
        or program.target_custody_identity != expected_target_identity
        or program.decoder_identity != expected_decoder_identity
    ):
        raise TSPPV2PVSA1LoweringError("rich TSPPV2 differs from external semantic/target/decoder custody")
    if program.compile_config.source_pair_start != 0 or program.compile_config.pair_count != 600:
        raise TSPPV2PVSA1LoweringError("PVSA1 pair indexing requires the exact global n600 window")


def _require_no_rich_json_in_compact(
    compact_member: bytes,
    *,
    rich_packet: bytes,
    rich_header: bytes,
    target_identity: ScorerTargetCustodyIdentityV1,
    decoder_identity: V15RoleAwareDecoderIdentityV2,
) -> None:
    forbidden = (
        rich_packet,
        rich_header,
        b"TSPPV2\x00\x00",
        target_identity.target_custody_receipt_sha256.encode("ascii"),
        target_identity.target_bank_sha256.encode("ascii"),
        decoder_identity.program_decoder_source_sha256.encode("ascii"),
        decoder_identity.g74_decoder_source_sha256.encode("ascii"),
        decoder_identity.torch_version.encode("ascii"),
    )
    if any(value and value in compact_member for value in forbidden):
        raise TSPPV2PVSA1LoweringError("rich identity/custody JSON leaked into compact PVSA member")


def lower_tsppv2_to_compact_pvsa1(
    *,
    rich_program_packet: bytes,
    semantic_p_archive: bytes,
    expected_semantic_identity: V15SemanticProgramIdentityV1,
    expected_target_identity: ScorerTargetCustodyIdentityV1,
    expected_decoder_identity: V15RoleAwareDecoderIdentityV2,
    maximum_program_bytes: int,
    maximum_semantic_archive_bytes: int,
    maximum_member_bytes: int,
    maximum_section_bytes: int,
) -> TSPPV2ToPVSA1LoweringV1:
    """Verify rich custody, lower one exact operand, and prove pair-0 equality."""

    if type(rich_program_packet) is not bytes or not rich_program_packet:
        raise TSPPV2PVSA1LoweringError("lowering requires nonempty exact rich TSPPV2 bytes")
    if type(semantic_p_archive) is not bytes or not semantic_p_archive:
        raise TSPPV2PVSA1LoweringError("lowering requires nonempty exact semantic P bytes")
    try:
        program = parse_selected_preimage_program_v2(
            rich_program_packet,
            maximum_packet_bytes=maximum_program_bytes,
        )
    except TaskspaceSelectedPreimageProgramV2Error as exc:
        raise TSPPV2PVSA1LoweringError("rich TSPPV2 strict parse/re-emit failed") from exc
    if encode_selected_preimage_program_v2(program) != rich_program_packet:
        raise TSPPV2PVSA1LoweringError("rich TSPPV2 changed on parse/re-emit")
    _require_external_identities(
        program,
        expected_semantic_identity=expected_semantic_identity,
        expected_target_identity=expected_target_identity,
        expected_decoder_identity=expected_decoder_identity,
    )
    semantic_identity = program.semantic_program_identity
    if (
        len(semantic_p_archive) != semantic_identity.compiled_semantic_archive_bytes
        or _sha256(semantic_p_archive) != semantic_identity.compiled_semantic_archive_sha256
    ):
        raise TSPPV2PVSA1LoweringError("semantic P bytes differ from exact rich TSPPV2 identity")

    try:
        baseline = build_compact_pvsa_archive(
            semantic_p_archive=semantic_p_archive,
            actuator_payloads=(),
            maximum_semantic_archive_bytes=maximum_semantic_archive_bytes,
            maximum_member_bytes=maximum_member_bytes,
            maximum_section_bytes=maximum_section_bytes,
            rich_compiler_ir_bytes=None,
        )
        compact = build_compact_pvsa_archive(
            semantic_p_archive=semantic_p_archive,
            actuator_payloads=(program.factor.operand_payload,),
            maximum_semantic_archive_bytes=maximum_semantic_archive_bytes,
            maximum_member_bytes=maximum_member_bytes,
            maximum_section_bytes=maximum_section_bytes,
            rich_compiler_ir_bytes=len(rich_program_packet),
        )
    except CompactPVSAError as exc:
        raise TSPPV2PVSA1LoweringError("PVSA1 baseline/actuated strict build failed") from exc
    compact_actuators = compact.selected.actuators
    if (
        baseline.selected.actuators
        or len(compact_actuators) != 1
        or compact_actuators[0].payload != program.factor.operand_payload
        or compact_actuators[0].sha256 != program.factor.operand_sha256
        or compact.stored.member_bytes != compact.deflated.member_bytes
        or compact.stored.member_bytes != compact.selected.member_bytes
        or baseline.stored.member_bytes != baseline.deflated.member_bytes
        or baseline.stored.member_bytes != baseline.selected.member_bytes
    ):
        raise TSPPV2PVSA1LoweringError("PVSA1 parse-back changed P, operand, or outer encoding")
    rich_header = _rich_header_bytes(rich_program_packet)
    _require_no_rich_json_in_compact(
        compact.selected.member_bytes,
        rich_packet=rich_program_packet,
        rich_header=rich_header,
        target_identity=expected_target_identity,
        decoder_identity=expected_decoder_identity,
    )

    try:
        overlay = V15RoleAwareOverlayDecoderV1.open(
            semantic_p_archive,
            expected_archive_bytes=len(semantic_p_archive),
            expected_archive_sha256=_sha256(semantic_p_archive),
            verify_member_effects=True,
        )
        bound = BoundV15RoleAwareSelectedPreimageDecoderV2(
            semantic_identity=expected_semantic_identity,
            target_custody_identity=expected_target_identity,
            decoder_identity=expected_decoder_identity,
            overlay_decoder=overlay,
        )
        rich_pair = decode_selected_preimage_pair_v2(program, 0, bound)
        compact_pair = compact.selected.decode_g74_pair(
            0,
            verify_member_effects=True,
        )
    except (
        CompactPVSAError,
        TaskspaceSelectedPreimageProgramV2Error,
        V15RoleAwareOverlayError,
    ) as exc:
        raise TSPPV2PVSA1LoweringError("rich/compact pair-0 native decode failed") from exc
    if (
        not np.array_equal(
            rich_pair.result.camera_pairs,
            compact_pair.camera_pairs,
        )
        or not np.array_equal(
            rich_pair.result.changed_support_values,
            compact_pair.changed_support_values,
        )
        or not np.array_equal(
            rich_pair.result.owned_camera_values,
            compact_pair.owned_camera_values,
        )
        or rich_pair.result.receipt.to_bytes() != compact_pair.receipt.to_bytes()
    ):
        raise TSPPV2PVSA1LoweringError("rich TSPPV2 and compact PVSA pair-0 decode differ")
    rich_pair_camera_sha256 = _sha256(memoryview(np.ascontiguousarray(rich_pair.result.camera_pairs)).cast("B"))
    compact_pair_camera_sha256 = _sha256(memoryview(np.ascontiguousarray(compact_pair.camera_pairs)).cast("B"))

    rich_homes = program.byte_homes()
    selected_outer = compact.outer_build.selected
    baseline_outer = baseline.outer_build.selected
    receipt = TSPPV2ToPVSA1ExternalLoweringReceiptV1(
        rich_program_bytes=len(rich_program_packet),
        rich_program_sha256=_sha256(rich_program_packet),
        rich_framing_bytes=rich_homes[0].byte_length,
        semantic_p_bytes=len(semantic_p_archive),
        semantic_p_sha256=_sha256(semantic_p_archive),
        semantic_identity_sha256=_identity_sha256(expected_semantic_identity),
        target_identity_sha256=_identity_sha256(expected_target_identity),
        target_custody_receipt_sha256=(expected_target_identity.target_custody_receipt_sha256),
        target_bank_sha256=expected_target_identity.target_bank_sha256,
        decoder_identity_sha256=_identity_sha256(expected_decoder_identity),
        program_decoder_source_sha256=(expected_decoder_identity.program_decoder_source_sha256),
        g74_decoder_source_sha256=(expected_decoder_identity.g74_decoder_source_sha256),
        torch_version=expected_decoder_identity.torch_version,
        operand_bytes=len(program.factor.operand_payload),
        operand_sha256=program.factor.operand_sha256,
        frame_selector=program.factor.frame_selector.value,
        source_pair_start=program.compile_config.source_pair_start,
        pair_count=program.compile_config.pair_count,
        semantic_baseline_member_bytes=baseline.compact_member_bytes,
        semantic_baseline_member_sha256=baseline.selected.member_sha256,
        semantic_baseline_archive_bytes=baseline_outer.archive_nbytes,
        semantic_baseline_archive_sha256=baseline_outer.archive_sha256,
        semantic_container_recode_delta_bytes=(baseline_outer.archive_nbytes - len(semantic_p_archive)),
        compact_member_bytes=compact.compact_member_bytes,
        compact_member_sha256=compact.selected.member_sha256,
        compact_outer_stored_bytes=compact.outer_build.stored.archive_nbytes,
        compact_outer_stored_sha256=(compact.outer_build.stored.archive_sha256),
        compact_outer_deflated_bytes=(compact.outer_build.deflated.archive_nbytes),
        compact_outer_deflated_sha256=(compact.outer_build.deflated.archive_sha256),
        compact_archive_bytes=selected_outer.archive_nbytes,
        compact_archive_sha256=selected_outer.archive_sha256,
        actuator_same_container_marginal_bytes=(selected_outer.archive_nbytes - baseline_outer.archive_nbytes),
        rich_ir_bytes_avoided=(len(rich_program_packet) - len(program.factor.operand_payload)),
        rich_pair_zero_camera_sha256=rich_pair_camera_sha256,
        compact_pair_zero_camera_sha256=compact_pair_camera_sha256,
        pair_zero_execution_receipt_sha256=_sha256(compact_pair.receipt.to_bytes()),
        lowering_source_sha256=lowering_source_sha256(),
        pvsa_source_sha256=_module_source_sha256(_pvsa),
        outer_archive_source_sha256=_module_source_sha256(_outer_codec),
    )
    return TSPPV2ToPVSA1LoweringV1(
        rich_program_packet=rich_program_packet,
        rich_program=program,
        semantic_p_archive=semantic_p_archive,
        semantic_baseline=baseline,
        compact_actuated=compact,
        receipt=receipt,
    )


__all__ = [
    "FULL_N600_BLOCKER",
    "LOWERING_CONTRACT_ID",
    "LOWERING_RECEIPT_SCHEMA",
    "OPEN_LOWERING_BLOCKERS",
    "TSPPV2PVSA1LoweringError",
    "TSPPV2ToPVSA1ExternalLoweringReceiptV1",
    "TSPPV2ToPVSA1LoweringV1",
    "lower_tsppv2_to_compact_pvsa1",
    "lowering_source_sha256",
]
