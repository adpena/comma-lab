# SPDX-License-Identifier: MIT
"""Bounded, archive-authoritative receiver for one counted ``G + A`` fragment.

The sole ZIP member and its G2 descriptors are the only section-slicing
authority.  The receiver reapplies exact counted G to external predictor state,
then gives that internally derived G/Y1 to the strict reverse-causal A decoder.
It finally realizes chronological ``(Y0, Y1)`` camera bytes through the existing
certified factor-2 operator.

This closes a real semantic receiver seam, but not a contest candidate: P is
still external, the decoder depends on repository runtime, the fragment is
bounded to four pairs, and complete payload-lineage/evaluator custody is absent.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

import numpy as np

from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Uint8LatticeError,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)
from tac.witness_dsl.coupled_preimage_program import (
    CoupledPreimageMode,
    CoupledPreimageProgramError,
    CoupledPreimageProgramV1,
    DecodedCoupledPreimageV1,
    decode_coupled_preimage_program,
    parse_coupled_preimage_program,
)
from tac.witness_dsl.generative_taskspace_correction import (
    DecodedGenerativeCorrectionV1,
    GenerativeTaskspaceCorrectionError,
    PredictorSemanticStateV1,
    apply_generative_taskspace_correction,
)
from tac.witness_dsl.generative_taskspace_receiver import (
    ARCHIVE_MEMBER_NAME,
    CountedTaskspaceRole,
    GenerativeTaskspaceReceiverError,
    ParsedGenerativeTaskspaceFragmentArchiveV1,
    build_generative_taskspace_receiver_fragment_archive,
    parse_generative_taskspace_receiver_fragment_archive,
)

RECEIPT_SCHEMA: Final = "tac.taskspace_pair_fragment_receiver_receipt.v1"
RECEIVER_CONTRACT_ID: Final = "tac.g2_inband_g_then_a_to_factor2_chronological_pair_fragment.v1"
MAX_IN_MEMORY_PAIRS: Final = 4
CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
SCORER_HEIGHT: Final = 384
SCORER_WIDTH: Final = 512
CHANNELS: Final = 3
REQUIRED_ROLES: Final = (
    CountedTaskspaceRole.GENERATIVE_CORRECTION,
    CountedTaskspaceRole.COUPLED_PREIMAGE,
)
SEMANTIC_ROLE_ORDER: Final = ("GENERATOR_PARAMETERS", "FRAME0_FROM_EXACT_Y1")


class TaskspacePairFragmentReceiverError(ValueError):
    """Malformed G2 bytes, failed G/A causality, or non-exact realization."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> None:
    if type(value) is not str or len(value) != 64:
        raise TaskspacePairFragmentReceiverError(f"{label} is not a SHA-256 hex digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise TaskspacePairFragmentReceiverError(f"{label} is not a SHA-256 hex digest") from exc
    if len(decoded) != 32 or value != value.lower():
        raise TaskspacePairFragmentReceiverError(f"{label} is not canonical lowercase SHA-256")


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
        raise TaskspacePairFragmentReceiverError("receipt value is not canonical finite ASCII JSON") from exc


def _decode_canonical_json(payload: bytes, *, label: str) -> Any:
    def unique_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise TaskspacePairFragmentReceiverError(f"{label} contains duplicate JSON keys")
            value[key] = item
        return value

    if not isinstance(payload, bytes) or not payload:
        raise TaskspacePairFragmentReceiverError(f"{label} must be nonempty exact bytes")
    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TaskspacePairFragmentReceiverError(f"{label} is not canonical ASCII JSON") from exc
    if _canonical_json(value) != payload:
        raise TaskspacePairFragmentReceiverError(f"{label} is not canonical JSON")
    return value


def _immutable_u8(value: np.ndarray) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=np.uint8).copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, slots=True)
class TaskspacePairSectionReceiptV1:
    """Exact G2 descriptor custody retained by the semantic receiver."""

    role: CountedTaskspaceRole
    offset: int
    byte_length: int
    payload_sha256: str
    crc32: int

    def __post_init__(self) -> None:
        if self.role not in REQUIRED_ROLES:
            raise TaskspacePairFragmentReceiverError("pair-fragment receipt contains an unconsumed role")
        if type(self.offset) is not int or self.offset < 0:
            raise TaskspacePairFragmentReceiverError("section receipt offset is invalid")
        if type(self.byte_length) is not int or self.byte_length < 1:
            raise TaskspacePairFragmentReceiverError("section receipt length is invalid")
        _require_sha256(self.payload_sha256, "section payload_sha256")
        if type(self.crc32) is not int or not 0 <= self.crc32 <= 0xFFFFFFFF:
            raise TaskspacePairFragmentReceiverError("section receipt CRC32 is invalid")

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
class TaskspacePairFragmentReceiptV1:
    """Truth-labelled receipt for one bounded chronological pair fragment."""

    schema: Literal["tac.taskspace_pair_fragment_receiver_receipt.v1"]
    receiver_contract_id: Literal["tac.g2_inband_g_then_a_to_factor2_chronological_pair_fragment.v1"]
    archive_bytes: int
    archive_sha256: str
    archive_member_name: Literal["0.bin"]
    archive_member_bytes: int
    archive_member_sha256: str
    archive_member_crc32: int
    sections: tuple[TaskspacePairSectionReceiptV1, TaskspacePairSectionReceiptV1]
    section_descriptor_manifest_sha256: str
    predictor_binding_sha256: str
    source_pair_ids: tuple[int, ...]
    a_mode: CoupledPreimageMode
    decoded_g_labels_sha256: str
    scorer_y0_sha256: str
    scorer_y1_sha256: str
    scorer_chronological_sha256: str
    camera_height: int
    camera_width: int
    scorer_height: int
    scorer_width: int
    channels: int
    camera_raw_bytes: int
    camera_raw_sha256: str
    camera_frame_sha256_chronological: tuple[str, ...]
    factor2_scorer_values_verified: int
    factor2_numerator_values_verified: int
    max_in_memory_pairs: Literal[4] = MAX_IN_MEMORY_PAIRS
    semantic_role_order: tuple[str, str] = SEMANTIC_ROLE_ORDER
    counted_section_descriptors_are_sole_slice_authority: Literal[True] = True
    external_pair_manifest_accepted: Literal[False] = False
    caller_decoded_g_accepted: Literal[False] = False
    exact_counted_g_reapplied: Literal[True] = True
    exact_counted_a_strictly_parsed_and_decoded: Literal[True] = True
    exact_y1_derived_from_counted_g_before_y0: Literal[True] = True
    optional_terminal_quotient_admitted: Literal[False] = False
    all_counted_sections_semantically_consumed: Literal[True] = True
    factor2_camera_parse_back_exact: Literal[True] = True
    chronological_camera_order_y0_then_y1: Literal[True] = True
    deterministic_double_decode: Literal[True] = True
    counted_predictor_section_present: Literal[False] = False
    external_predictor_state_required: Literal[True] = True
    repository_runtime_dependency: Literal[True] = True
    complete_bounded_chronological_pair_output: Literal[True] = True
    complete_data_in_code_lineage_audit: Literal[False] = False
    n600_evidence_claim: Literal[False] = False
    exact_score_claim: Literal[False] = False
    candidate_archive_eligible: Literal[False] = False
    originality_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA or self.receiver_contract_id != RECEIVER_CONTRACT_ID:
            raise TaskspacePairFragmentReceiverError("pair-fragment receipt schema or contract changed")
        if self.archive_member_name != ARCHIVE_MEMBER_NAME:
            raise TaskspacePairFragmentReceiverError("pair-fragment receipt member name changed")
        if type(self.archive_bytes) is not int or self.archive_bytes < 1:
            raise TaskspacePairFragmentReceiverError("receipt archive byte count is invalid")
        if type(self.archive_member_bytes) is not int or self.archive_member_bytes < 1:
            raise TaskspacePairFragmentReceiverError("receipt member byte count is invalid")
        if type(self.archive_member_crc32) is not int or not 0 <= self.archive_member_crc32 <= 0xFFFFFFFF:
            raise TaskspacePairFragmentReceiverError("receipt member CRC32 is invalid")
        for field in (
            "archive_sha256",
            "archive_member_sha256",
            "section_descriptor_manifest_sha256",
            "predictor_binding_sha256",
            "decoded_g_labels_sha256",
            "scorer_y0_sha256",
            "scorer_y1_sha256",
            "scorer_chronological_sha256",
            "camera_raw_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if (
            type(self.sections) is not tuple
            or len(self.sections) != 2
            or any(type(row) is not TaskspacePairSectionReceiptV1 for row in self.sections)
        ):
            raise TaskspacePairFragmentReceiverError("receipt sections must be one exact typed tuple")
        if tuple(row.role for row in self.sections) != REQUIRED_ROLES:
            raise TaskspacePairFragmentReceiverError("receipt roles are not exact in-band G then A")
        if self.sections[0].stop != self.sections[1].offset:
            raise TaskspacePairFragmentReceiverError("receipt sections are not contiguous in descriptor order")
        if self.sections[-1].stop != self.archive_member_bytes:
            raise TaskspacePairFragmentReceiverError("receipt section/member byte accounting is not exact")
        expected_descriptor_manifest = _sha256(_canonical_json([row.as_dict() for row in self.sections]))
        if self.section_descriptor_manifest_sha256 != expected_descriptor_manifest:
            raise TaskspacePairFragmentReceiverError("receipt descriptor manifest hash differs from exact sections")
        if (
            type(self.source_pair_ids) is not tuple
            or not 1 <= len(self.source_pair_ids) <= MAX_IN_MEMORY_PAIRS
            or any(type(pair_id) is not int for pair_id in self.source_pair_ids)
        ):
            raise TaskspacePairFragmentReceiverError("receipt pair population escaped the bounded receiver")
        expected_pair_ids = tuple(range(self.source_pair_ids[0], self.source_pair_ids[0] + len(self.source_pair_ids)))
        if (
            self.source_pair_ids != expected_pair_ids
            or not 0 <= expected_pair_ids[0] < expected_pair_ids[-1] + 1 <= 600
        ):
            raise TaskspacePairFragmentReceiverError("receipt source-pair IDs are not contiguous inside [0,600)")
        if type(self.a_mode) is not CoupledPreimageMode:
            raise TaskspacePairFragmentReceiverError("receipt A mode changed the closed mode universe")
        exact_integer_fields = (
            "camera_height",
            "camera_width",
            "scorer_height",
            "scorer_width",
            "channels",
            "camera_raw_bytes",
            "factor2_scorer_values_verified",
            "factor2_numerator_values_verified",
            "max_in_memory_pairs",
        )
        if any(type(getattr(self, field)) is not int for field in exact_integer_fields):
            raise TaskspacePairFragmentReceiverError("receipt geometry/count fields must be exact integers")
        if (self.camera_height, self.camera_width, self.scorer_height, self.scorer_width, self.channels) != (
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            SCORER_HEIGHT,
            SCORER_WIDTH,
            CHANNELS,
        ):
            raise TaskspacePairFragmentReceiverError("receipt geometry changed the frozen factor-2 ABI")
        expected_raw_bytes = len(self.source_pair_ids) * 2 * CAMERA_HEIGHT * CAMERA_WIDTH * CHANNELS
        if self.camera_raw_bytes != expected_raw_bytes:
            raise TaskspacePairFragmentReceiverError("receipt camera byte count is not a complete bounded pair")
        if type(self.camera_frame_sha256_chronological) is not tuple or len(
            self.camera_frame_sha256_chronological
        ) != 2 * len(self.source_pair_ids):
            raise TaskspacePairFragmentReceiverError("receipt camera-frame hash count changed chronological ABI")
        for index, digest in enumerate(self.camera_frame_sha256_chronological):
            _require_sha256(digest, f"camera_frame_sha256_chronological[{index}]")
        expected_values = len(self.source_pair_ids) * 2 * SCORER_HEIGHT * SCORER_WIDTH * CHANNELS
        if (
            self.factor2_scorer_values_verified != expected_values
            or self.factor2_numerator_values_verified != expected_values
        ):
            raise TaskspacePairFragmentReceiverError("receipt factor-2 verified-value count is incomplete")
        truth_labels = (
            self.max_in_memory_pairs == MAX_IN_MEMORY_PAIRS
            and type(self.semantic_role_order) is tuple
            and self.semantic_role_order == SEMANTIC_ROLE_ORDER
            and self.counted_section_descriptors_are_sole_slice_authority is True
            and self.external_pair_manifest_accepted is False
            and self.caller_decoded_g_accepted is False
            and self.exact_counted_g_reapplied is True
            and self.exact_counted_a_strictly_parsed_and_decoded is True
            and self.exact_y1_derived_from_counted_g_before_y0 is True
            and self.optional_terminal_quotient_admitted is False
            and self.all_counted_sections_semantically_consumed is True
            and self.factor2_camera_parse_back_exact is True
            and self.chronological_camera_order_y0_then_y1 is True
            and self.deterministic_double_decode is True
            and self.counted_predictor_section_present is False
            and self.external_predictor_state_required is True
            and self.repository_runtime_dependency is True
            and self.complete_bounded_chronological_pair_output is True
            and self.complete_data_in_code_lineage_audit is False
            and self.n600_evidence_claim is False
            and self.exact_score_claim is False
            and self.candidate_archive_eligible is False
            and self.originality_claim is False
            and self.promotion_eligible is False
            and self.research_only is True
        )
        if not truth_labels:
            raise TaskspacePairFragmentReceiverError("pair-fragment truth labels became dishonest")

    @property
    def pair_count(self) -> int:
        return len(self.source_pair_ids)

    def as_dict(self) -> dict[str, Any]:
        """Return the complete canonical machine receipt body."""

        value = asdict(self)
        value["a_mode"] = self.a_mode.value
        value["sections"] = [row.as_dict() for row in self.sections]
        value["source_pair_ids"] = list(self.source_pair_ids)
        value["semantic_role_order"] = list(self.semantic_role_order)
        value["camera_frame_sha256_chronological"] = list(self.camera_frame_sha256_chronological)
        return value

    def to_receipt_bytes(self) -> bytes:
        """Serialize every claim as deterministic, finite canonical JSON."""

        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


def parse_taskspace_pair_fragment_receipt(payload: bytes) -> TaskspacePairFragmentReceiptV1:
    """Strictly parse a canonical receipt and re-run every receipt invariant."""

    value = _decode_canonical_json(payload, label="taskspace pair-fragment receipt")
    if not isinstance(value, Mapping):
        raise TaskspacePairFragmentReceiverError("pair-fragment receipt body must be one JSON object")
    expected_fields = set(TaskspacePairFragmentReceiptV1.__dataclass_fields__)
    if set(value) != expected_fields:
        raise TaskspacePairFragmentReceiverError("pair-fragment receipt fields differ from the closed v1 schema")

    section_values = value["sections"]
    if not isinstance(section_values, list) or len(section_values) != 2:
        raise TaskspacePairFragmentReceiverError("pair-fragment receipt requires exactly two section rows")
    sections: list[TaskspacePairSectionReceiptV1] = []
    section_fields = set(TaskspacePairSectionReceiptV1.__dataclass_fields__)
    for index, row in enumerate(section_values):
        if not isinstance(row, Mapping) or set(row) != section_fields:
            raise TaskspacePairFragmentReceiverError(f"receipt section[{index}] fields differ from closed v1 schema")
        try:
            role = CountedTaskspaceRole(row["role"])
            section = TaskspacePairSectionReceiptV1(
                role=role,
                offset=row["offset"],
                byte_length=row["byte_length"],
                payload_sha256=row["payload_sha256"],
                crc32=row["crc32"],
            )
        except (TypeError, ValueError) as exc:
            raise TaskspacePairFragmentReceiverError(f"receipt section[{index}] is not typed canonical data") from exc
        sections.append(section)

    source_pair_ids = value["source_pair_ids"]
    semantic_role_order = value["semantic_role_order"]
    camera_frame_hashes = value["camera_frame_sha256_chronological"]
    if not isinstance(source_pair_ids, list):
        raise TaskspacePairFragmentReceiverError("receipt source_pair_ids must be one canonical JSON list")
    if not isinstance(semantic_role_order, list):
        raise TaskspacePairFragmentReceiverError("receipt semantic_role_order must be one canonical JSON list")
    if not isinstance(camera_frame_hashes, list):
        raise TaskspacePairFragmentReceiverError(
            "receipt camera_frame_sha256_chronological must be one canonical JSON list"
        )

    arguments = dict(value)
    arguments["sections"] = tuple(sections)
    arguments["source_pair_ids"] = tuple(source_pair_ids)
    arguments["semantic_role_order"] = tuple(semantic_role_order)
    arguments["camera_frame_sha256_chronological"] = tuple(camera_frame_hashes)
    try:
        arguments["a_mode"] = CoupledPreimageMode(value["a_mode"])
        receipt = TaskspacePairFragmentReceiptV1(**arguments)
    except (TypeError, ValueError) as exc:
        raise TaskspacePairFragmentReceiverError("pair-fragment receipt contains invalid typed values") from exc
    if receipt.to_receipt_bytes() != payload:
        raise TaskspacePairFragmentReceiverError("pair-fragment receipt changed on strict typed parse-back")
    return receipt


@dataclass(frozen=True, slots=True)
class TaskspacePairFragmentDecodeV1:
    """Complete bounded scorer-grid and camera-byte pair-fragment decode."""

    decoded_g: DecodedGenerativeCorrectionV1
    a_program: CoupledPreimageProgramV1
    decoded_a: DecodedCoupledPreimageV1
    scorer_chronological_frames: np.ndarray
    camera_raw: bytes
    receipt: TaskspacePairFragmentReceiptV1

    def __post_init__(self) -> None:
        if type(self.decoded_g) is not DecodedGenerativeCorrectionV1:
            raise TaskspacePairFragmentReceiverError("decode result G changed typed ABI")
        if type(self.a_program) is not CoupledPreimageProgramV1 or type(self.decoded_a) is not DecodedCoupledPreimageV1:
            raise TaskspacePairFragmentReceiverError("decode result A changed typed ABI")
        scorer = np.asarray(self.scorer_chronological_frames)
        expected_shape = (self.receipt.pair_count, 2, SCORER_HEIGHT, SCORER_WIDTH, CHANNELS)
        if scorer.dtype != np.uint8 or scorer.shape != expected_shape:
            raise TaskspacePairFragmentReceiverError("decode result scorer frames changed chronological ABI")
        if not np.array_equal(scorer, self.decoded_a.chronological_frames):
            raise TaskspacePairFragmentReceiverError("decode result scorer frames differ from strict A output")
        if not isinstance(self.camera_raw, bytes) or len(self.camera_raw) != self.receipt.camera_raw_bytes:
            raise TaskspacePairFragmentReceiverError("decode result camera bytes are incomplete")
        if _sha256(self.camera_raw) != self.receipt.camera_raw_sha256:
            raise TaskspacePairFragmentReceiverError("decode result camera bytes differ from receipt")
        if self.decoded_g.correction_packet_sha256 != self.receipt.sections[0].payload_sha256:
            raise TaskspacePairFragmentReceiverError("decode result G differs from receipt section hash")
        if _sha256(self.a_program.to_packet()) != self.receipt.sections[1].payload_sha256:
            raise TaskspacePairFragmentReceiverError("decode result A differs from receipt section hash")
        if self.a_program.mode is not self.receipt.a_mode:
            raise TaskspacePairFragmentReceiverError("decode result A mode differs from receipt")
        if self.a_program.source_binding.source_pair_ids != self.receipt.source_pair_ids:
            raise TaskspacePairFragmentReceiverError("decode result pair order differs from receipt")
        if _sha256(memoryview(self.decoded_g.labels).cast("B")) != self.receipt.decoded_g_labels_sha256:
            raise TaskspacePairFragmentReceiverError("decode result G labels differ from receipt")
        if _sha256(memoryview(self.decoded_a.y0).cast("B")) != self.receipt.scorer_y0_sha256:
            raise TaskspacePairFragmentReceiverError("decode result scorer Y0 differs from receipt")
        if _sha256(memoryview(self.decoded_a.y1).cast("B")) != self.receipt.scorer_y1_sha256:
            raise TaskspacePairFragmentReceiverError("decode result scorer Y1 differs from receipt")
        if _sha256(memoryview(scorer).cast("B")) != self.receipt.scorer_chronological_sha256:
            raise TaskspacePairFragmentReceiverError("decode result chronological scorer frames differ from receipt")
        frame_bytes = CAMERA_HEIGHT * CAMERA_WIDTH * CHANNELS
        actual_frame_hashes = tuple(
            _sha256(self.camera_raw[offset : offset + frame_bytes])
            for offset in range(0, len(self.camera_raw), frame_bytes)
        )
        if actual_frame_hashes != self.receipt.camera_frame_sha256_chronological:
            raise TaskspacePairFragmentReceiverError("decode result chronological camera frames differ from receipt")
        object.__setattr__(self, "scorer_chronological_frames", _immutable_u8(scorer))


@dataclass(frozen=True, slots=True)
class _DecodedOnceV1:
    parsed_archive: ParsedGenerativeTaskspaceFragmentArchiveV1
    sections: tuple[TaskspacePairSectionReceiptV1, TaskspacePairSectionReceiptV1]
    decoded_g: DecodedGenerativeCorrectionV1
    a_program: CoupledPreimageProgramV1
    decoded_a: DecodedCoupledPreimageV1
    camera_raw: bytes
    camera_frame_sha256_chronological: tuple[str, ...]
    factor2_scorer_values_verified: int
    factor2_numerator_values_verified: int


def _factor2_operator() -> DisjointResizeOperator:
    try:
        return DisjointResizeOperator.build(
            camera_h=CAMERA_HEIGHT,
            camera_w=CAMERA_WIDTH,
            scorer_h=SCORER_HEIGHT,
            scorer_w=SCORER_WIDTH,
        )
    except Uint8LatticeError as exc:
        raise TaskspacePairFragmentReceiverError("frozen camera/scorer geometry is not factor-2 realizable") from exc


def _descriptor_receipts(
    parsed: ParsedGenerativeTaskspaceFragmentArchiveV1,
) -> tuple[TaskspacePairSectionReceiptV1, TaskspacePairSectionReceiptV1]:
    roles = tuple(section.role for section in parsed.counted.sections)
    if roles != REQUIRED_ROLES:
        if CountedTaskspaceRole.TERMINAL_QUOTIENT in roles:
            raise TaskspacePairFragmentReceiverError(
                "optional T is refused until a real in-band semantic T consumer is implemented"
            )
        raise TaskspacePairFragmentReceiverError("pair receiver requires exact in-band G then A sections")
    receipts: list[TaskspacePairSectionReceiptV1] = []
    for section in parsed.counted.sections:
        exact_slice = parsed.counted.payload[section.offset : section.offset + section.byte_length]
        if exact_slice != section.payload:
            raise TaskspacePairFragmentReceiverError("G2 descriptor slice differs from retained section payload")
        if _sha256(exact_slice) != section.payload_sha256 or zlib.crc32(exact_slice) & 0xFFFFFFFF != section.crc32:
            raise TaskspacePairFragmentReceiverError("G2 descriptor hash or CRC differs at semantic consumption")
        receipts.append(
            TaskspacePairSectionReceiptV1(
                role=section.role,
                offset=section.offset,
                byte_length=section.byte_length,
                payload_sha256=section.payload_sha256,
                crc32=section.crc32,
            )
        )
    return receipts[0], receipts[1]


def _realize_chronological_camera_bytes(
    decoded_a: DecodedCoupledPreimageV1,
) -> tuple[bytes, tuple[str, ...], int, int]:
    operator = _factor2_operator()
    raw = bytearray()
    frame_hashes: list[str] = []
    scorer_values = 0
    numerator_values = 0
    try:
        for pair_index in range(decoded_a.chronological_frames.shape[0]):
            for chronological_index in range(2):
                scorer_frame = decoded_a.chronological_frames[pair_index, chronological_index]
                camera_frame = realize_factor2_uint8_scorer_plane(operator, scorer_frame)
                proof = verify_factor2_uint8_scorer_plane(operator, camera_frame, scorer_frame)
                if not proof.certified_exact or not proof.numerator_exact:
                    raise TaskspacePairFragmentReceiverError(
                        "chronological camera frame failed exact factor-2 parse-back"
                    )
                frame_bytes = camera_frame.tobytes(order="C")
                raw.extend(frame_bytes)
                frame_hashes.append(_sha256(frame_bytes))
                scorer_values += proof.scorer_values
                numerator_values += proof.numerator_equal_values
    except Uint8LatticeError as exc:
        raise TaskspacePairFragmentReceiverError("factor-2 chronological camera realization refused") from exc
    return bytes(raw), tuple(frame_hashes), scorer_values, numerator_values


def _decode_once(
    archive: bytes,
    *,
    predictor_state: PredictorSemanticStateV1,
) -> _DecodedOnceV1:
    if type(predictor_state) is not PredictorSemanticStateV1:
        raise TaskspacePairFragmentReceiverError("pair receiver requires exact PredictorSemanticStateV1")
    if predictor_state.pair_count > MAX_IN_MEMORY_PAIRS:
        raise TaskspacePairFragmentReceiverError(
            f"pair receiver hard-caps in-memory fragments at {MAX_IN_MEMORY_PAIRS} pairs"
        )
    try:
        parsed = parse_generative_taskspace_receiver_fragment_archive(archive)
        sections = _descriptor_receipts(parsed)
        g_packet = parsed.counted.sections[0].payload
        a_packet = parsed.counted.sections[1].payload
        decoded_g = apply_generative_taskspace_correction(g_packet, predictor_state=predictor_state)
        if decoded_g.correction_packet_sha256 != sections[0].payload_sha256:
            raise TaskspacePairFragmentReceiverError("strict G reapply differs from descriptor-bound exact G")
        a_program = parse_coupled_preimage_program(a_packet)
        if a_program.source_binding.source_pair_ids != predictor_state.source_pair_ids:
            raise TaskspacePairFragmentReceiverError("counted A source-pair order differs from external P state")
        decoded_a = decode_coupled_preimage_program(
            a_packet,
            predictor_state=predictor_state,
            decoded_g=decoded_g,
        )
    except TaskspacePairFragmentReceiverError:
        raise
    except GenerativeTaskspaceReceiverError as exc:
        raise TaskspacePairFragmentReceiverError("strict G2 archive parse refused exact counted bytes") from exc
    except GenerativeTaskspaceCorrectionError as exc:
        raise TaskspacePairFragmentReceiverError("descriptor-bound counted G failed strict reapply") from exc
    except CoupledPreimageProgramError as exc:
        raise TaskspacePairFragmentReceiverError(
            "descriptor-bound counted A failed strict reverse-causal decode from exact counted G/Y1"
        ) from exc

    exact_y1 = decoded_g.paint_rgb()
    if not np.array_equal(decoded_a.y1, exact_y1):
        raise TaskspacePairFragmentReceiverError("strict A output Y1 differs from internally derived exact counted G")
    camera_raw, frame_hashes, scorer_values, numerator_values = _realize_chronological_camera_bytes(decoded_a)
    return _DecodedOnceV1(
        parsed_archive=parsed,
        sections=sections,
        decoded_g=decoded_g,
        a_program=a_program,
        decoded_a=decoded_a,
        camera_raw=camera_raw,
        camera_frame_sha256_chronological=frame_hashes,
        factor2_scorer_values_verified=scorer_values,
        factor2_numerator_values_verified=numerator_values,
    )


def _same_decode(first: _DecodedOnceV1, second: _DecodedOnceV1) -> bool:
    return (
        first.parsed_archive.archive == second.parsed_archive.archive
        and first.sections == second.sections
        and first.decoded_g.correction_packet_sha256 == second.decoded_g.correction_packet_sha256
        and first.decoded_g.realization_profile == second.decoded_g.realization_profile
        and np.array_equal(first.decoded_g.labels, second.decoded_g.labels)
        and first.a_program == second.a_program
        and np.array_equal(first.decoded_a.chronological_frames, second.decoded_a.chronological_frames)
        and first.camera_raw == second.camera_raw
        and first.camera_frame_sha256_chronological == second.camera_frame_sha256_chronological
        and first.factor2_scorer_values_verified == second.factor2_scorer_values_verified
        and first.factor2_numerator_values_verified == second.factor2_numerator_values_verified
    )


def receive_taskspace_pair_fragment(
    archive: bytes,
    *,
    predictor_state: PredictorSemanticStateV1,
) -> TaskspacePairFragmentDecodeV1:
    """Strictly consume exact in-archive G+A and emit bounded camera-byte pairs.

    There is intentionally no ``decoded_g`` or external section-manifest
    argument.  Both decodes begin again from the archive member descriptors.
    """

    first = _decode_once(archive, predictor_state=predictor_state)
    second = _decode_once(archive, predictor_state=predictor_state)
    if not _same_decode(first, second):
        raise TaskspacePairFragmentReceiverError("identical archive/P inputs did not double-decode deterministically")

    parsed = first.parsed_archive
    section_manifest_sha256 = _sha256(_canonical_json([row.as_dict() for row in first.sections]))
    scorer = first.decoded_a.chronological_frames
    receipt = TaskspacePairFragmentReceiptV1(
        schema=RECEIPT_SCHEMA,
        receiver_contract_id=RECEIVER_CONTRACT_ID,
        archive_bytes=len(parsed.archive),
        archive_sha256=parsed.archive_sha256,
        archive_member_name=ARCHIVE_MEMBER_NAME,
        archive_member_bytes=len(parsed.counted.payload),
        archive_member_sha256=parsed.counted.payload_sha256,
        archive_member_crc32=parsed.member_crc32,
        sections=first.sections,
        section_descriptor_manifest_sha256=section_manifest_sha256,
        predictor_binding_sha256=predictor_state.binding_sha256,
        source_pair_ids=predictor_state.source_pair_ids,
        a_mode=first.a_program.mode,
        decoded_g_labels_sha256=_sha256(memoryview(first.decoded_g.labels).cast("B")),
        scorer_y0_sha256=_sha256(memoryview(first.decoded_a.y0).cast("B")),
        scorer_y1_sha256=_sha256(memoryview(first.decoded_a.y1).cast("B")),
        scorer_chronological_sha256=_sha256(memoryview(scorer).cast("B")),
        camera_height=CAMERA_HEIGHT,
        camera_width=CAMERA_WIDTH,
        scorer_height=SCORER_HEIGHT,
        scorer_width=SCORER_WIDTH,
        channels=CHANNELS,
        camera_raw_bytes=len(first.camera_raw),
        camera_raw_sha256=_sha256(first.camera_raw),
        camera_frame_sha256_chronological=first.camera_frame_sha256_chronological,
        factor2_scorer_values_verified=first.factor2_scorer_values_verified,
        factor2_numerator_values_verified=first.factor2_numerator_values_verified,
    )
    return TaskspacePairFragmentDecodeV1(
        decoded_g=first.decoded_g,
        a_program=first.a_program,
        decoded_a=first.decoded_a,
        scorer_chronological_frames=scorer,
        camera_raw=first.camera_raw,
        receipt=receipt,
    )


def build_taskspace_pair_fragment_archive(
    g_packet: bytes,
    a_packet: bytes,
    *,
    predictor_state: PredictorSemanticStateV1,
) -> bytes:
    """Build canonical G2 bytes only after cross-section semantic receive succeeds."""

    archive = build_generative_taskspace_receiver_fragment_archive(
        g_packet,
        predictor_state=predictor_state,
        coupled_preimage_packet=a_packet,
    )
    received = receive_taskspace_pair_fragment(archive, predictor_state=predictor_state)
    if received.receipt.sections[0].payload_sha256 != _sha256(g_packet):
        raise TaskspacePairFragmentReceiverError("builder cross-receive changed exact G bytes")
    if received.receipt.sections[1].payload_sha256 != _sha256(a_packet):
        raise TaskspacePairFragmentReceiverError("builder cross-receive changed exact A bytes")
    return archive


def build_a_only_counterfactual_fragment_archive(
    baseline_archive: bytes,
    counterfactual_a_packet: bytes,
    *,
    predictor_state: PredictorSemanticStateV1,
) -> bytes:
    """Replace only A and prove realized Y0 changes while exact G/Y1 stays fixed."""

    baseline = receive_taskspace_pair_fragment(baseline_archive, predictor_state=predictor_state)
    parsed = parse_generative_taskspace_receiver_fragment_archive(baseline_archive)
    exact_g = parsed.counted.section(CountedTaskspaceRole.GENERATIVE_CORRECTION).payload
    if parsed.counted.section(CountedTaskspaceRole.COUPLED_PREIMAGE).payload == counterfactual_a_packet:
        raise TaskspacePairFragmentReceiverError("A-only counterfactual must change exact counted A bytes")
    archive = build_taskspace_pair_fragment_archive(
        exact_g,
        counterfactual_a_packet,
        predictor_state=predictor_state,
    )
    counterfactual = receive_taskspace_pair_fragment(archive, predictor_state=predictor_state)
    if baseline.a_program.mode is not counterfactual.a_program.mode:
        raise TaskspacePairFragmentReceiverError("matched A-only counterfactual cannot change A mode")
    if baseline.receipt.sections[0].payload_sha256 != counterfactual.receipt.sections[0].payload_sha256:
        raise TaskspacePairFragmentReceiverError("A-only counterfactual changed exact counted G")
    if baseline.receipt.scorer_y1_sha256 != counterfactual.receipt.scorer_y1_sha256:
        raise TaskspacePairFragmentReceiverError("A-only counterfactual changed exact scorer-grid Y1")
    if (
        baseline.receipt.camera_frame_sha256_chronological[1::2]
        != (counterfactual.receipt.camera_frame_sha256_chronological[1::2])
    ):
        raise TaskspacePairFragmentReceiverError("A-only counterfactual changed realized camera Y1")
    if baseline.receipt.scorer_y0_sha256 == counterfactual.receipt.scorer_y0_sha256:
        raise TaskspacePairFragmentReceiverError("A-only counterfactual did not change scorer-grid Y0")
    if (
        baseline.receipt.camera_frame_sha256_chronological[0::2]
        == (counterfactual.receipt.camera_frame_sha256_chronological[0::2])
    ):
        raise TaskspacePairFragmentReceiverError("A-only counterfactual did not change realized camera Y0")
    return archive
