# SPDX-License-Identifier: MIT
"""Encoder-only acquisition custody for exact G17 population descriptors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from typing import Any, Final, Protocol

from tac.witness_dsl.bounded_target_g_encoder import parse_bounded_target_g_encoder_receipt
from tac.witness_dsl.taskspace_selective_topology_acquisition import (
    parse_selective_topology_acquisition_receipt,
    parse_selective_topology_program_proposal_receipt,
)

SCHEMA: Final = "tac.taskspace_g17_g_descriptor_acquisition_custody.v1"


class G17GDescriptorCustodyError(ValueError):
    """Descriptor acquisition evidence was missing, relabelled, or stale."""


class G17GAcquisitionClassV1(StrEnum):
    PASS_G8_V1 = "PASS_G8_V1"
    SELECTIVE_G15_ROW3_V1 = "SELECTIVE_G15_ROW3_V1"
    EXACT_TARGET_DIAGNOSTIC_V1 = "EXACT_TARGET_DIAGNOSTIC_V1"


class _ReceiptLike(Protocol):
    schema: str

    def to_receipt_bytes(self) -> bytes: ...


StrictReceiptParser = Callable[[bytes], _ReceiptLike]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
        raise G17GDescriptorCustodyError("custody receipt is not finite canonical ASCII JSON") from exc


def _strict_json(payload: bytes, *, expected_fields: set[str]) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise G17GDescriptorCustodyError(f"custody repeats JSON key {key!r}")
            result[key] = value
        return result

    if type(payload) is not bytes or not payload:
        raise G17GDescriptorCustodyError("custody receipt must be nonempty exact bytes")
    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G17GDescriptorCustodyError("custody receipt is not strict ASCII JSON") from exc
    if type(value) is not dict or set(value) != expected_fields or value.get("schema") != SCHEMA:
        raise G17GDescriptorCustodyError("custody receipt field set or schema changed")
    if _canonical_json(value) != payload:
        raise G17GDescriptorCustodyError("custody receipt changed on canonical parse/re-emit")
    return value


def _reopen(payload: bytes, parser: StrictReceiptParser, *, expected_schema: str | None) -> _ReceiptLike:
    if type(payload) is not bytes or not payload:
        raise G17GDescriptorCustodyError("referenced acquisition receipt must be retained exact bytes")
    if not callable(parser):
        raise G17GDescriptorCustodyError("referenced receipt requires a frozen public strict parser")
    try:
        parsed = parser(payload)
    except Exception as exc:
        raise G17GDescriptorCustodyError("frozen strict parser refused referenced acquisition receipt") from exc
    if not hasattr(parsed, "to_receipt_bytes") or parsed.to_receipt_bytes() != payload:
        raise G17GDescriptorCustodyError("referenced receipt failed parse/re-emit identity")
    schema = getattr(parsed, "schema", None)
    if type(schema) is not str or (expected_schema is not None and schema != expected_schema):
        raise G17GDescriptorCustodyError("referenced receipt schema differs from its custody row")
    return parsed


@dataclass(frozen=True, slots=True)
class G17StrictReceiptEvidenceV1:
    """Retained encoder evidence and the exact public parser that reopens it."""

    receipt_bytes: bytes = field(repr=False)
    strict_parser: StrictReceiptParser = field(repr=False, compare=False)
    expected_schema: str | None = None

    def __post_init__(self) -> None:
        parsed = _reopen(self.receipt_bytes, self.strict_parser, expected_schema=self.expected_schema)
        if self.expected_schema is None:
            object.__setattr__(self, "expected_schema", parsed.schema)

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.receipt_bytes)

    @property
    def schema(self) -> str:
        parsed = _reopen(self.receipt_bytes, self.strict_parser, expected_schema=self.expected_schema)
        return parsed.schema

    def reopen(self) -> _ReceiptLike:
        return _reopen(self.receipt_bytes, self.strict_parser, expected_schema=self.expected_schema)


@dataclass(frozen=True, slots=True)
class G17GDescriptorAcquisitionEvidenceV1:
    descriptor_index: int
    acquisition_class: G17GAcquisitionClassV1
    semantic_acquisition_receipt: G17StrictReceiptEvidenceV1 | None
    semantic_program_receipt: G17StrictReceiptEvidenceV1 | None
    fresh_g8_acquisition_receipt: G17StrictReceiptEvidenceV1 | None

    def __post_init__(self) -> None:
        if type(self.descriptor_index) is not int or self.descriptor_index < 0:
            raise G17GDescriptorCustodyError("descriptor evidence index must be nonnegative")
        if type(self.acquisition_class) is not G17GAcquisitionClassV1:
            raise G17GDescriptorCustodyError("descriptor evidence acquisition class is invalid")
        if self.acquisition_class is G17GAcquisitionClassV1.SELECTIVE_G15_ROW3_V1:
            if self.semantic_acquisition_receipt is None or self.semantic_program_receipt is None:
                raise G17GDescriptorCustodyError("selective acquisition requires both frozen G15 receipts")
            if (
                self.semantic_acquisition_receipt.strict_parser is not parse_selective_topology_acquisition_receipt
                or self.semantic_program_receipt.strict_parser is not parse_selective_topology_program_proposal_receipt
            ):
                raise G17GDescriptorCustodyError("selective acquisition must use the exact frozen public G15 parsers")
        elif self.acquisition_class is G17GAcquisitionClassV1.EXACT_TARGET_DIAGNOSTIC_V1:
            if self.semantic_acquisition_receipt is None or self.semantic_program_receipt is not None:
                raise G17GDescriptorCustodyError("exact diagnostic requires only its bounded-target receipt")
            if self.semantic_acquisition_receipt.strict_parser is not parse_bounded_target_g_encoder_receipt:
                raise G17GDescriptorCustodyError("exact diagnostic must use the frozen bounded-target strict parser")
        elif self.semantic_acquisition_receipt is not None or self.semantic_program_receipt is not None:
            raise G17GDescriptorCustodyError("PASS-G8 cannot carry semantic acquisition evidence")


@dataclass(frozen=True, slots=True)
class G17GDescriptorAcquisitionCustodyRowV1:
    descriptor_index: int
    pair_start: int
    pair_count: int
    descriptor_sha256: str
    descriptor_family: str
    descriptor_mode: str
    payload_sha256: str
    semantic_packet_sha256: str | None
    g8_packet_sha256: str | None
    acquisition_class: str
    semantic_acquisition_receipt_schema: str | None
    semantic_acquisition_receipt_sha256: str | None
    semantic_program_receipt_schema: str | None
    semantic_program_receipt_sha256: str | None
    fresh_g8_acquisition_receipt_schema: str | None
    fresh_g8_acquisition_receipt_sha256: str | None
    row_not_exact_control: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class G17GDescriptorAcquisitionCustodyReceiptV1:
    schema: str
    p_section_sha256: str
    g_section_sha256: str
    g_descriptor_window_root_sha256: str
    g_order_root_sha256: str
    population_pair_order_sha256: str
    empty_pass_descriptor_indices: tuple[int, ...]
    rows: tuple[G17GDescriptorAcquisitionCustodyRowV1, ...]
    whole_object_contains_no_exact_control: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "p_section_sha256": self.p_section_sha256,
            "g_section_sha256": self.g_section_sha256,
            "g_descriptor_window_root_sha256": self.g_descriptor_window_root_sha256,
            "g_order_root_sha256": self.g_order_root_sha256,
            "population_pair_order_sha256": self.population_pair_order_sha256,
            "empty_pass_descriptor_indices": list(self.empty_pass_descriptor_indices),
            "rows": [row.as_dict() for row in self.rows],
            "whole_object_contains_no_exact_control": self.whole_object_contains_no_exact_control,
        }

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


@dataclass(frozen=True, slots=True)
class G17GDescriptorAcquisitionCustodyV1:
    receipt: G17GDescriptorAcquisitionCustodyReceiptV1
    g_section_bytes: bytes = field(repr=False)
    evidence: tuple[G17GDescriptorAcquisitionEvidenceV1, ...] = field(repr=False)

    @property
    def receipt_bytes(self) -> bytes:
        return self.receipt.to_receipt_bytes()

    @property
    def custody_sha256(self) -> str:
        return self.receipt.receipt_sha256


def _evidence_fields(
    evidence: G17GDescriptorAcquisitionEvidenceV1,
) -> tuple[str | None, str | None, str | None, str | None, str | None, str | None]:
    output: list[str | None] = []
    for item in (
        evidence.semantic_acquisition_receipt,
        evidence.semantic_program_receipt,
        evidence.fresh_g8_acquisition_receipt,
    ):
        if item is None:
            output.extend((None, None))
        else:
            item.reopen()
            output.extend((item.schema, item.receipt_sha256))
    return tuple(output)  # type: ignore[return-value]


def _attribute(parsed: object, *names: str) -> Any:
    for name in names:
        if hasattr(parsed, name):
            return getattr(parsed, name)
    return None


def _descriptor_packets(descriptor: object) -> tuple[bytes | None, bytes | None]:
    """Recover exact semantic and outer-G8 bytes from retained strict parse."""

    payload = _attribute(descriptor, "payload")
    family = _attribute(descriptor, "family")
    mode = _attribute(descriptor, "mode")
    active = _attribute(descriptor, "active")
    if type(payload) is not bytes or active is None:
        raise G17GDescriptorCustodyError("nonempty descriptor lost its typed strict nested parse")
    parsed = _attribute(active, "parsed_object")
    family_name = _attribute(family, "name")
    mode_name = _attribute(mode, "name")
    if family_name == "PASS_PREDICTOR":
        return None, payload
    if mode_name == "SEMANTIC_ONLY":
        return payload, None
    semantic = _attribute(parsed, "semantic_g_packet")
    if type(semantic) is not bytes or not semantic:
        raise G17GDescriptorCustodyError("fresh-G8 composite lost exact inner semantic packet bytes")
    return semantic, payload


def _validate_evidence_packet_bindings(
    *,
    descriptor: object,
    evidence: G17GDescriptorAcquisitionEvidenceV1,
    semantic_packet: bytes | None,
    g8_packet: bytes | None,
) -> None:
    pair_start = _attribute(descriptor, "pair_start")
    pair_count = _attribute(descriptor, "pair_count")
    source_ids = tuple(range(pair_start, pair_start + pair_count))
    for label, receipt_evidence in (
        ("semantic acquisition", evidence.semantic_acquisition_receipt),
        ("semantic program", evidence.semantic_program_receipt),
        ("fresh G8 acquisition", evidence.fresh_g8_acquisition_receipt),
    ):
        if receipt_evidence is None:
            continue
        reopened = receipt_evidence.reopen()
        receipt_ids = _attribute(reopened, "source_pair_ids")
        if receipt_ids is not None and receipt_ids != source_ids:
            raise G17GDescriptorCustodyError(f"{label} receipt belongs to another source window")
    if semantic_packet is not None:
        semantic_hash = _sha256(semantic_packet)
        for receipt_evidence in (
            evidence.semantic_acquisition_receipt,
            evidence.semantic_program_receipt,
            evidence.fresh_g8_acquisition_receipt,
        ):
            if receipt_evidence is None:
                continue
            reopened = receipt_evidence.reopen()
            bound_hash = _attribute(
                reopened,
                "packet_sha256",
                "selective_g_packet_sha256",
            )
            if bound_hash is not None and bound_hash != semantic_hash:
                raise G17GDescriptorCustodyError("retained evidence binds different semantic packet bytes")
    if g8_packet is not None and evidence.fresh_g8_acquisition_receipt is not None:
        reopened = evidence.fresh_g8_acquisition_receipt.reopen()
        outer_hash = _attribute(reopened, "envelope_sha256", "g8_packet_sha256", "composite_g_sha256")
        if outer_hash is not None and outer_hash != _sha256(g8_packet):
            raise G17GDescriptorCustodyError("fresh-G8 receipt binds different outer packet bytes")


def build_g17_g_descriptor_acquisition_custody(
    *,
    p_section_bytes: bytes,
    g_section_bytes: bytes,
    evidence: tuple[G17GDescriptorAcquisitionEvidenceV1, ...],
    g_active_parser: object | None = None,
) -> G17GDescriptorAcquisitionCustodyV1:
    """Build custody from exact G bytes and retained frozen-parser evidence."""

    if type(p_section_bytes) is not bytes or not p_section_bytes:
        raise G17GDescriptorCustodyError("P section must be retained nonempty exact bytes")
    if type(g_section_bytes) is not bytes or not g_section_bytes:
        raise G17GDescriptorCustodyError("G section must be retained nonempty exact bytes")
    from tac.witness_dsl.taskspace_g17_production_envelope import (  # local import avoids a module cycle
        G17GFamilyV1,
        G17GModeV1,
        parse_g17_g_section,
    )

    if g_active_parser is not None and not callable(g_active_parser):
        raise G17GDescriptorCustodyError("G active parser must be callable when provided")
    parsed = parse_g17_g_section(
        g_section_bytes,
        expected_p_section=p_section_bytes,
        active_parser=g_active_parser,
    )
    evidence_by_index: dict[int, G17GDescriptorAcquisitionEvidenceV1] = {}
    for item in evidence:
        if type(item) is not G17GDescriptorAcquisitionEvidenceV1 or item.descriptor_index in evidence_by_index:
            raise G17GDescriptorCustodyError("descriptor evidence must be unique exact typed rows")
        evidence_by_index[item.descriptor_index] = item

    empty_indices: list[int] = []
    rows: list[G17GDescriptorAcquisitionCustodyRowV1] = []
    for index, descriptor in enumerate(parsed.descriptors):
        if not descriptor.payload:
            if descriptor.family is not G17GFamilyV1.PASS_PREDICTOR or descriptor.mode is not G17GModeV1.SEMANTIC_ONLY:
                raise G17GDescriptorCustodyError("only canonical PASS/no-G8 may omit a G payload")
            empty_indices.append(index)
            if index in evidence_by_index:
                raise G17GDescriptorCustodyError("empty PASS descriptor must not have an evidence row")
            continue
        row_evidence = evidence_by_index.pop(index, None)
        if row_evidence is None:
            raise G17GDescriptorCustodyError("every nonempty G descriptor requires exactly one evidence row")
        if descriptor.family is G17GFamilyV1.PASS_PREDICTOR:
            expected_class = G17GAcquisitionClassV1.PASS_G8_V1
        elif descriptor.family is G17GFamilyV1.SELECTIVE_ROW3:
            expected_class = G17GAcquisitionClassV1.SELECTIVE_G15_ROW3_V1
        else:
            expected_class = G17GAcquisitionClassV1.EXACT_TARGET_DIAGNOSTIC_V1
        if row_evidence.acquisition_class is not expected_class:
            raise G17GDescriptorCustodyError("descriptor family was relabelled as another acquisition class")
        if descriptor.mode is G17GModeV1.SEMANTIC_THEN_FRESH_G8:
            if row_evidence.fresh_g8_acquisition_receipt is None:
                raise G17GDescriptorCustodyError("fresh-G8 descriptor omitted its acquisition receipt")
        elif row_evidence.fresh_g8_acquisition_receipt is not None:
            raise G17GDescriptorCustodyError("semantic-only descriptor carried stale fresh-G8 evidence")

        semantic_packet, g8_packet = _descriptor_packets(descriptor)
        sem_acq = row_evidence.semantic_acquisition_receipt
        sem_program = row_evidence.semantic_program_receipt
        _validate_evidence_packet_bindings(
            descriptor=descriptor,
            evidence=row_evidence,
            semantic_packet=semantic_packet,
            g8_packet=g8_packet,
        )
        if sem_program is not None:
            proposal = sem_program.reopen()
            if _attribute(proposal, "source_pair_ids") != tuple(range(descriptor.pair_start, descriptor.stop)):
                raise G17GDescriptorCustodyError("semantic program receipt belongs to another source window")
            if semantic_packet is None or _attribute(proposal, "packet_sha256") != _sha256(semantic_packet):
                raise G17GDescriptorCustodyError("semantic program receipt binds different packet bytes")
        if sem_acq is not None:
            acquisition = sem_acq.reopen()
            source_ids = _attribute(acquisition, "source_pair_ids")
            if source_ids is not None and source_ids != tuple(range(descriptor.pair_start, descriptor.stop)):
                raise G17GDescriptorCustodyError("semantic acquisition receipt belongs to another window")
            packet_hash = _attribute(acquisition, "packet_sha256")
            if packet_hash is not None and (semantic_packet is None or packet_hash != _sha256(semantic_packet)):
                raise G17GDescriptorCustodyError("semantic acquisition receipt binds different packet bytes")
        schemas_and_hashes = _evidence_fields(row_evidence)
        encoded = parsed.encoded_descriptors[index]
        rows.append(
            G17GDescriptorAcquisitionCustodyRowV1(
                descriptor_index=index,
                pair_start=descriptor.pair_start,
                pair_count=descriptor.pair_count,
                descriptor_sha256=_sha256(encoded.descriptor_bytes),
                descriptor_family=descriptor.family.name,
                descriptor_mode=descriptor.mode.name,
                payload_sha256=_sha256(descriptor.payload),
                semantic_packet_sha256=None if semantic_packet is None else _sha256(semantic_packet),
                g8_packet_sha256=None if g8_packet is None else _sha256(g8_packet),
                acquisition_class=row_evidence.acquisition_class.value,
                semantic_acquisition_receipt_schema=schemas_and_hashes[0],
                semantic_acquisition_receipt_sha256=schemas_and_hashes[1],
                semantic_program_receipt_schema=schemas_and_hashes[2],
                semantic_program_receipt_sha256=schemas_and_hashes[3],
                fresh_g8_acquisition_receipt_schema=schemas_and_hashes[4],
                fresh_g8_acquisition_receipt_sha256=schemas_and_hashes[5],
                row_not_exact_control=expected_class is not G17GAcquisitionClassV1.EXACT_TARGET_DIAGNOSTIC_V1,
            )
        )
    if evidence_by_index:
        raise G17GDescriptorCustodyError("evidence contains an omitted, duplicate, or foreign descriptor index")
    exact_free = all(row.row_not_exact_control for row in rows)
    receipt = G17GDescriptorAcquisitionCustodyReceiptV1(
        schema=SCHEMA,
        p_section_sha256=_sha256(p_section_bytes),
        g_section_sha256=_sha256(g_section_bytes),
        g_descriptor_window_root_sha256=parsed.descriptor_window_root_sha256,
        g_order_root_sha256=parsed.order_root_sha256,
        population_pair_order_sha256=parsed.population_pair_order_sha256,
        empty_pass_descriptor_indices=tuple(empty_indices),
        rows=tuple(rows),
        whole_object_contains_no_exact_control=exact_free,
    )
    return parse_g17_g_descriptor_acquisition_custody(
        receipt.to_receipt_bytes(),
        p_section_bytes=p_section_bytes,
        g_section_bytes=g_section_bytes,
        evidence=evidence,
        g_active_parser=g_active_parser,
    )


def parse_g17_g_descriptor_acquisition_custody(
    payload: bytes,
    *,
    p_section_bytes: bytes,
    g_section_bytes: bytes,
    evidence: tuple[G17GDescriptorAcquisitionEvidenceV1, ...],
    g_active_parser: object | None = None,
) -> G17GDescriptorAcquisitionCustodyV1:
    """Reopen the map, every exact G byte, and every retained evidence receipt."""

    expected_fields = {item.name for item in fields(G17GDescriptorAcquisitionCustodyReceiptV1)}
    value = _strict_json(payload, expected_fields=expected_fields)
    if type(value["empty_pass_descriptor_indices"]) is not list or type(value["rows"]) is not list:
        raise G17GDescriptorCustodyError("custody index/row collections must be JSON lists")
    row_fields = {item.name for item in fields(G17GDescriptorAcquisitionCustodyRowV1)}
    if any(type(row) is not dict or set(row) != row_fields for row in value["rows"]):
        raise G17GDescriptorCustodyError("custody row field set changed")
    try:
        receipt = G17GDescriptorAcquisitionCustodyReceiptV1(
            **{
                **value,
                "empty_pass_descriptor_indices": tuple(value["empty_pass_descriptor_indices"]),
                "rows": tuple(G17GDescriptorAcquisitionCustodyRowV1(**row) for row in value["rows"]),
            }
        )
    except (TypeError, ValueError) as exc:
        raise G17GDescriptorCustodyError("custody contains invalid typed fields") from exc
    from tac.witness_dsl.taskspace_g17_production_envelope import parse_g17_g_section

    if g_active_parser is not None and not callable(g_active_parser):
        raise G17GDescriptorCustodyError("G active parser must be callable when provided")
    parsed = parse_g17_g_section(
        g_section_bytes,
        expected_p_section=p_section_bytes,
        active_parser=g_active_parser,
    )
    if (
        receipt.p_section_sha256 != _sha256(p_section_bytes)
        or receipt.g_section_sha256 != _sha256(g_section_bytes)
        or receipt.g_descriptor_window_root_sha256 != parsed.descriptor_window_root_sha256
        or receipt.g_order_root_sha256 != parsed.order_root_sha256
        or receipt.population_pair_order_sha256 != parsed.population_pair_order_sha256
    ):
        raise G17GDescriptorCustodyError("custody roots differ from reopened exact P/G bytes")
    rebuilt = _build_without_recursive_parse(
        p_section_bytes=p_section_bytes,
        g_section_bytes=g_section_bytes,
        evidence=evidence,
        g_active_parser=g_active_parser,
    )
    if rebuilt.to_receipt_bytes() != payload:
        raise G17GDescriptorCustodyError("custody rows/eligibility drift from retained evidence")
    return G17GDescriptorAcquisitionCustodyV1(
        receipt=receipt,
        g_section_bytes=g_section_bytes,
        evidence=evidence,
    )


def _build_without_recursive_parse(
    *,
    p_section_bytes: bytes,
    g_section_bytes: bytes,
    evidence: tuple[G17GDescriptorAcquisitionEvidenceV1, ...],
    g_active_parser: object | None,
) -> G17GDescriptorAcquisitionCustodyReceiptV1:
    """Internal single-pass builder used by strict parse verification."""

    # Build once through the public entry, but intercept the recursive parse by
    # reproducing its deterministic row construction in a private sentinel.
    # The sentinel is installed only for this call and never exposed as ABI.
    from tac.witness_dsl.taskspace_g17_production_envelope import G17GFamilyV1, G17GModeV1, parse_g17_g_section

    parsed = parse_g17_g_section(
        g_section_bytes,
        expected_p_section=p_section_bytes,
        active_parser=g_active_parser,
    )
    evidence_by_index = {row.descriptor_index: row for row in evidence}
    if len(evidence_by_index) != len(evidence):
        raise G17GDescriptorCustodyError("descriptor evidence contains duplicate indices")
    empty: list[int] = []
    rows: list[G17GDescriptorAcquisitionCustodyRowV1] = []
    for index, descriptor in enumerate(parsed.descriptors):
        if not descriptor.payload:
            empty.append(index)
            if index in evidence_by_index:
                raise G17GDescriptorCustodyError("empty PASS descriptor has evidence")
            continue
        item = evidence_by_index.pop(index, None)
        if item is None:
            raise G17GDescriptorCustodyError("nonempty descriptor omitted evidence")
        expected_class = (
            G17GAcquisitionClassV1.PASS_G8_V1
            if descriptor.family is G17GFamilyV1.PASS_PREDICTOR
            else G17GAcquisitionClassV1.SELECTIVE_G15_ROW3_V1
            if descriptor.family is G17GFamilyV1.SELECTIVE_ROW3
            else G17GAcquisitionClassV1.EXACT_TARGET_DIAGNOSTIC_V1
        )
        if item.acquisition_class is not expected_class:
            raise G17GDescriptorCustodyError("acquisition-class relabel detected")
        if (descriptor.mode is G17GModeV1.SEMANTIC_THEN_FRESH_G8) != (item.fresh_g8_acquisition_receipt is not None):
            raise G17GDescriptorCustodyError("fresh-G8 receipt presence drift")
        semantic_packet, g8_packet = _descriptor_packets(descriptor)
        _validate_evidence_packet_bindings(
            descriptor=descriptor,
            evidence=item,
            semantic_packet=semantic_packet,
            g8_packet=g8_packet,
        )
        sem_program = item.semantic_program_receipt
        if sem_program is not None:
            reopened = sem_program.reopen()
            if _attribute(reopened, "source_pair_ids") != tuple(range(descriptor.pair_start, descriptor.stop)):
                raise G17GDescriptorCustodyError("program evidence window drift")
            if semantic_packet is None or _attribute(reopened, "packet_sha256") != _sha256(semantic_packet):
                raise G17GDescriptorCustodyError("program evidence packet drift")
        details = _evidence_fields(item)
        encoded = parsed.encoded_descriptors[index]
        rows.append(
            G17GDescriptorAcquisitionCustodyRowV1(
                descriptor_index=index,
                pair_start=descriptor.pair_start,
                pair_count=descriptor.pair_count,
                descriptor_sha256=_sha256(encoded.descriptor_bytes),
                descriptor_family=descriptor.family.name,
                descriptor_mode=descriptor.mode.name,
                payload_sha256=_sha256(descriptor.payload),
                semantic_packet_sha256=None if semantic_packet is None else _sha256(semantic_packet),
                g8_packet_sha256=None if g8_packet is None else _sha256(g8_packet),
                acquisition_class=item.acquisition_class.value,
                semantic_acquisition_receipt_schema=details[0],
                semantic_acquisition_receipt_sha256=details[1],
                semantic_program_receipt_schema=details[2],
                semantic_program_receipt_sha256=details[3],
                fresh_g8_acquisition_receipt_schema=details[4],
                fresh_g8_acquisition_receipt_sha256=details[5],
                row_not_exact_control=expected_class is not G17GAcquisitionClassV1.EXACT_TARGET_DIAGNOSTIC_V1,
            )
        )
    if evidence_by_index:
        raise G17GDescriptorCustodyError("custody contains foreign evidence rows")
    return G17GDescriptorAcquisitionCustodyReceiptV1(
        schema=SCHEMA,
        p_section_sha256=_sha256(p_section_bytes),
        g_section_sha256=_sha256(g_section_bytes),
        g_descriptor_window_root_sha256=parsed.descriptor_window_root_sha256,
        g_order_root_sha256=parsed.order_root_sha256,
        population_pair_order_sha256=parsed.population_pair_order_sha256,
        empty_pass_descriptor_indices=tuple(empty),
        rows=tuple(rows),
        whole_object_contains_no_exact_control=all(row.row_not_exact_control for row in rows),
    )


__all__ = [
    "SCHEMA",
    "G17GAcquisitionClassV1",
    "G17GDescriptorAcquisitionCustodyReceiptV1",
    "G17GDescriptorAcquisitionCustodyRowV1",
    "G17GDescriptorAcquisitionCustodyV1",
    "G17GDescriptorAcquisitionEvidenceV1",
    "G17GDescriptorCustodyError",
    "G17StrictReceiptEvidenceV1",
    "build_g17_g_descriptor_acquisition_custody",
    "parse_g17_g_descriptor_acquisition_custody",
]
