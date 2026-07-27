# SPDX-License-Identifier: MIT
"""Exact G17 population envelope and P-once production receiver seams.

This module owns the G17 ``TACG17G1``, ``TACG17A1`` and ``TACG17E1`` ABIs.
Active nested programs are accepted only through typed validation results made
by caller-supplied adapters around the frozen public strict parsers.  Empty
PASS programs need no adapter.  The receiver decodes P exactly once, exposes
read-only bounded shard views, and emits each source pair exactly once.

All truth exported here is ``research_only``.  Nothing in this module invokes a
scorer, an evaluator, a launch, a promotion, or a pointer update.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from enum import IntEnum, StrEnum
from typing import Any, Final, Literal, TypeAlias

import numpy as np

from tac.witness_dsl.taskspace_monolithic_pga_receiver import (
    ParsedTaskspaceMonolithicPGAMemberV1,
    TaskspaceMonolithicPGARole,
    build_taskspace_monolithic_pga_archive,
    parse_taskspace_monolithic_pga_member,
)
from tac.witness_dsl.taskspace_outer_archive_codec import (
    OuterArchiveEncoding,
    ParsedTaskspaceOuterArchive,
    TaskspaceOuterArchiveBuild,
    parse_taskspace_outer_archive,
)

G_PACKET_MAGIC: Final = b"TACG17G\x00"
A_PACKET_MAGIC: Final = b"TACG17A\x00"
E_PACKET_MAGIC: Final = b"TACG17E\x00"
PACKET_VERSION: Final = 1
GA_HEADER: Final = struct.Struct(">8sBBHHHII32s32s")
GA_DESCRIPTOR: Final = struct.Struct(">HHBBII32sI")
GA_FOOTER: Final = struct.Struct(">I")
E_PACKET: Final = struct.Struct(">8sBBBBBBBHH32s32s32s32sI")
GA_HEADER_BYTES: Final = 88
GA_DESCRIPTOR_BYTES: Final = 50
GA_FOOTER_BYTES: Final = 4
E_PACKET_BYTES: Final = 151
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_POPULATION_COUNTS: Final = frozenset({2, 24, 600})
_MAX_NESTED_PAYLOAD_BYTES: Final = 1 << 31


class G17ProductionEnvelopeError(ValueError):
    """Exact G/A/E bytes, source closure, or receiver custody failed closed."""


class G17ProductionEnvelopeBlockerCode(StrEnum):
    ACTIVE_NESTED_STRICT_PARSER_OWED = "G17_ACTIVE_NESTED_STRICT_PARSER_OWED"
    ACTIVE_G_RECEIVER_OWED = "G17_ACTIVE_G_RECEIVER_OWED"
    ACTIVE_A_RECEIVER_OWED = "G17_ACTIVE_A_RECEIVER_OWED"
    G17_STANDALONE_RUNTIME_N600_CUSTODY_OWED = "G17_STANDALONE_RUNTIME_N600_CUSTODY_OWED"
    G17_REAL_N2_EP725_RUNTIME_R_SCORER_CUSTODY_OWED = "G17_REAL_N2_EP725_RUNTIME_R_SCORER_CUSTODY_OWED"


class G17ProductionEnvelopeBlocker(G17ProductionEnvelopeError):
    def __init__(self, code: G17ProductionEnvelopeBlockerCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


class G17PopulationLayout(IntEnum):
    SHARDED = 0
    GLOBAL = 1


class G17GFamily(IntEnum):
    PASS_PREDICTOR = 1
    SELECTIVE_ROW3 = 2
    EXACT_SEMANTIC_DIAGNOSTIC = 3


class G17GMode(IntEnum):
    SEMANTIC_ONLY = 0
    SEMANTIC_THEN_FRESH_G8 = 1


class G17AFamily(IntEnum):
    CANONICAL_PASS = 0
    NATIVE_PASS_CONDITIONAL = 1
    NATIVE_SELECTIVE_NO_G8 = 2
    NATIVE_SELECTIVE_POST_G8 = 3
    G13_PASS_SOURCE_XIP2 = 4
    G17_GENERAL_CONDITIONAL_XIP2 = 5


class G17AMode(IntEnum):
    PASS_P0 = 0
    SPARSE_CONSTANT_RGB = 1
    COPY_FINAL_Y1_SUPPORT = 2
    GLOBAL_COPY_FINAL_Y1 = 3
    QUANTIZED_XIP2 = 4


class G17TerminalSemanticMode(IntEnum):
    PASS = 0
    SELECTIVE = 1
    EXACT_DIAGNOSTIC = 2


class G17TerminalG8Mode(IntEnum):
    NONE = 0
    FRESH = 1
    MIXED = 2


class G17RealizationExtension(IntEnum):
    NONE = 0


class G17TerminalAFamily(IntEnum):
    PASS = 0
    NATIVE = 1
    G13 = 2
    G17_GENERAL = 3
    MIXED = 4


class G17TerminalAMode(IntEnum):
    PASS = 0
    SPARSE_CONSTANT = 1
    COPY_SUPPORT = 2
    GLOBAL_COPY = 3
    QUANTIZED_XIP2 = 4
    MIXED = 5


class G17SemanticSummaryV1(StrEnum):
    """Closed receiver summary derived only from exact G descriptors."""

    PASS = "PASS"
    SELECTIVE = "SELECTIVE"
    EXACT_DIAGNOSTIC = "EXACT_DIAGNOSTIC"

    @property
    def generalized_source_value(self) -> str:
        return {
            G17SemanticSummaryV1.PASS: "PASS_PREDICTOR_V1",
            G17SemanticSummaryV1.SELECTIVE: "SELECTIVE_ROW3_TACG1C_V1",
            G17SemanticSummaryV1.EXACT_DIAGNOSTIC: "EXACT_SEMANTIC_DIAGNOSTIC_V1",
        }[self]


class G17G8SummaryV1(StrEnum):
    """Closed receiver summary derived only from exact G descriptor modes."""

    NONE = "NONE_V1"
    FRESH = "FRESH_POST_TOPOLOGY_G8_V1"
    MIXED = "MIXED_SHARDED_V1"

    @property
    def generalized_source_value(self) -> str:
        return self.value


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _require_bytes(value: object, *, name: str, nonempty: bool = True) -> bytes:
    if type(value) is not bytes or (nonempty and not value):
        qualifier = "nonempty " if nonempty else ""
        raise G17ProductionEnvelopeError(f"{name} must be {qualifier}immutable bytes")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise G17ProductionEnvelopeError(f"{name} must be canonical lowercase SHA-256")
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
        raise G17ProductionEnvelopeError("receipt must be finite canonical ASCII JSON") from exc


def _strict_canonical_json(payload: bytes, *, expected_fields: set[str], schema: str) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise G17ProductionEnvelopeError(f"receipt repeats JSON key {key!r}")
            result[key] = value
        return result

    _require_bytes(payload, name="receipt bytes")
    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G17ProductionEnvelopeError("receipt is not strict ASCII JSON") from exc
    if type(value) is not dict or set(value) != expected_fields or value.get("schema") != schema:
        raise G17ProductionEnvelopeError("receipt schema or exact field set changed")
    if _canonical_json(value) != payload:
        raise G17ProductionEnvelopeError("receipt changed on canonical parse/re-emit")
    return value


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


def _immutable_population_array(
    value: np.ndarray,
    *,
    name: str,
    pair_count: int,
    expected_tail: tuple[int, ...],
) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != np.uint8 or value.shape != (pair_count, *expected_tail):
        raise G17ProductionEnvelopeError(f"{name} must be exact uint8 with shape {(pair_count, *expected_tail)}")
    copied = np.ascontiguousarray(value).copy()
    copied.setflags(write=False)
    return copied


def canonical_g17_shard_windows(pair_start: int, pair_count: int) -> tuple[tuple[int, int], ...]:
    """Return the sole width-four G17 partition for n2/n24/n600."""

    if type(pair_start) is not int or pair_start < 0:
        raise G17ProductionEnvelopeError("population pair_start must be an exact nonnegative integer")
    if type(pair_count) is not int or pair_count not in _POPULATION_COUNTS:
        raise G17ProductionEnvelopeError("population pair_count must be exactly one of 2, 24, or 600")
    if pair_start + pair_count > 600:
        raise G17ProductionEnvelopeError("population window escapes [0,600)")
    return tuple(
        (start, min(4, pair_start + pair_count - start)) for start in range(pair_start, pair_start + pair_count, 4)
    )


def g17_population_pair_order_sha256(pair_start: int, pair_count: int) -> str:
    canonical_g17_shard_windows(pair_start, pair_count)
    packed = struct.pack(">" + "H" * pair_count, *range(pair_start, pair_start + pair_count))
    return _sha256(b"G17-PAIR-ORDER-V1\0" + packed)


def _descriptor_window_root(domain: Literal["G", "A"], windows: Sequence[tuple[int, int]]) -> str:
    rows = [[start, count] for start, count in windows]
    return _sha256(f"G17-{domain}-DESCRIPTOR-WINDOWS-V1\0".encode("ascii") + _canonical_json(rows))


@dataclass(frozen=True, slots=True)
class G17GActiveNestedV1:
    """Typed proof that a frozen public G parser reopened exact active bytes."""

    payload: bytes
    reencoded_payload: bytes
    pair_start: int
    pair_count: int
    family: G17GFamily
    mode: G17GMode
    strict_parser_id: str
    parsed_object: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_bytes(self.payload, name="active G payload")
        if self.reencoded_payload != self.payload:
            raise G17ProductionEnvelopeError("active G payload changed on frozen parse/re-encode")
        _validate_window(self.pair_start, self.pair_count)
        if type(self.family) is not G17GFamily or type(self.mode) is not G17GMode:
            raise G17ProductionEnvelopeError("active G family/mode type is not exact")
        if not self.strict_parser_id or not self.strict_parser_id.isascii():
            raise G17ProductionEnvelopeError("active G parser identity must be nonempty ASCII")
        if self.parsed_object is None:
            raise G17ProductionEnvelopeError("active G validation must retain its frozen parsed object")
        _validate_g_family_mode(self.family, self.mode, self.payload)


@dataclass(frozen=True, slots=True)
class G17AActiveNestedV1:
    """Typed proof that a frozen public A parser reopened exact active bytes."""

    payload: bytes
    reencoded_payload: bytes
    pair_start: int
    pair_count: int
    family: G17AFamily
    mode: G17AMode
    strict_parser_id: str
    parsed_object: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_bytes(self.payload, name="active A payload")
        if self.reencoded_payload != self.payload:
            raise G17ProductionEnvelopeError("active A payload changed on frozen parse/re-encode")
        _validate_window(self.pair_start, self.pair_count)
        if type(self.family) is not G17AFamily or type(self.mode) is not G17AMode:
            raise G17ProductionEnvelopeError("active A family/mode type is not exact")
        if not self.strict_parser_id or not self.strict_parser_id.isascii():
            raise G17ProductionEnvelopeError("active A parser identity must be nonempty ASCII")
        if self.parsed_object is None:
            raise G17ProductionEnvelopeError("active A validation must retain its frozen parsed object")
        _validate_a_family_mode(self.family, self.mode, self.payload)


G17GStrictNestedParser: TypeAlias = Callable[[bytes, int, int, G17GFamily, G17GMode], G17GActiveNestedV1]
G17AStrictNestedParser: TypeAlias = Callable[[bytes, int, int, G17AFamily, G17AMode], G17AActiveNestedV1]


def _validate_window(pair_start: object, pair_count: object) -> tuple[int, int]:
    if type(pair_start) is not int or type(pair_count) is not int or pair_start < 0 or not 0 < pair_count <= 600:
        raise G17ProductionEnvelopeError("descriptor window must be exact positive integers inside [0,600)")
    if pair_start + pair_count > 600:
        raise G17ProductionEnvelopeError("descriptor window escapes [0,600)")
    return pair_start, pair_count


def _validate_g_family_mode(family: G17GFamily, mode: G17GMode, payload: bytes) -> None:
    if family is G17GFamily.PASS_PREDICTOR and mode is G17GMode.SEMANTIC_ONLY:
        if payload:
            raise G17ProductionEnvelopeError("canonical PASS/no-G8 G descriptor must have empty payload")
        return
    if not payload:
        raise G17ProductionEnvelopeError("active G descriptor must retain nonempty nested payload bytes")
    expected_magic = (
        b"TACPG81\x00"
        if family is G17GFamily.PASS_PREDICTOR
        else b"TACG1C\x00\x00"
        if mode is G17GMode.SEMANTIC_ONLY
        else b"TACG8S1\x00"
    )
    if not payload.startswith(expected_magic):
        raise G17ProductionEnvelopeError("active G nested magic disagrees with descriptor family/mode")


def _validate_a_family_mode(family: G17AFamily, mode: G17AMode, payload: bytes) -> None:
    if family is G17AFamily.CANONICAL_PASS and mode is G17AMode.PASS_P0:
        if payload:
            raise G17ProductionEnvelopeError("canonical PASS A descriptor must have empty payload")
        return
    if not payload:
        raise G17ProductionEnvelopeError("active A descriptor must retain nonempty nested payload bytes")
    matrix: dict[G17AFamily, tuple[set[G17AMode], bytes]] = {
        G17AFamily.NATIVE_PASS_CONDITIONAL: (
            {G17AMode.SPARSE_CONSTANT_RGB, G17AMode.COPY_FINAL_Y1_SUPPORT},
            b"TACAPG1\x00",
        ),
        G17AFamily.NATIVE_SELECTIVE_NO_G8: (
            {G17AMode.SPARSE_CONSTANT_RGB, G17AMode.COPY_FINAL_Y1_SUPPORT},
            b"TACA3P1\x00",
        ),
        G17AFamily.NATIVE_SELECTIVE_POST_G8: (
            {G17AMode.SPARSE_CONSTANT_RGB, G17AMode.COPY_FINAL_Y1_SUPPORT},
            b"TACA8P1\x00",
        ),
        G17AFamily.G13_PASS_SOURCE_XIP2: (
            {G17AMode.GLOBAL_COPY_FINAL_Y1, G17AMode.QUANTIZED_XIP2},
            b"TACX2A3\x00",
        ),
        G17AFamily.G17_GENERAL_CONDITIONAL_XIP2: (
            {G17AMode.GLOBAL_COPY_FINAL_Y1, G17AMode.QUANTIZED_XIP2},
            b"TACX2A4\x00",
        ),
    }
    allowed = matrix.get(family)
    if allowed is None or mode not in allowed[0] or not payload.startswith(allowed[1]):
        raise G17ProductionEnvelopeError("active A nested magic or mode disagrees with descriptor family")


@dataclass(frozen=True, slots=True)
class G17GDescriptorV1:
    pair_start: int
    pair_count: int
    family: G17GFamily
    mode: G17GMode
    active: G17GActiveNestedV1 | None = None

    def __post_init__(self) -> None:
        _validate_window(self.pair_start, self.pair_count)
        if type(self.family) is not G17GFamily or type(self.mode) is not G17GMode:
            raise G17ProductionEnvelopeError("G descriptor family/mode must use exact G enum types")
        payload = b"" if self.active is None else self.active.payload
        _validate_g_family_mode(self.family, self.mode, payload)
        if self.active is not None and (
            type(self.active) is not G17GActiveNestedV1
            or (self.active.pair_start, self.active.pair_count) != (self.pair_start, self.pair_count)
            or self.active.family is not self.family
            or self.active.mode is not self.mode
        ):
            raise G17ProductionEnvelopeError("active G validation differs from descriptor window/family/mode")

    @property
    def payload(self) -> bytes:
        return b"" if self.active is None else self.active.payload

    @property
    def stop(self) -> int:
        return self.pair_start + self.pair_count

    @classmethod
    def canonical_pass(cls, pair_start: int, pair_count: int) -> G17GDescriptorV1:
        return cls(pair_start, pair_count, G17GFamily.PASS_PREDICTOR, G17GMode.SEMANTIC_ONLY)


@dataclass(frozen=True, slots=True)
class G17ADescriptorV1:
    pair_start: int
    pair_count: int
    family: G17AFamily
    mode: G17AMode
    active: G17AActiveNestedV1 | None = None

    def __post_init__(self) -> None:
        _validate_window(self.pair_start, self.pair_count)
        if type(self.family) is not G17AFamily or type(self.mode) is not G17AMode:
            raise G17ProductionEnvelopeError("A descriptor family/mode must use exact A enum types")
        payload = b"" if self.active is None else self.active.payload
        _validate_a_family_mode(self.family, self.mode, payload)
        if self.active is not None and (
            type(self.active) is not G17AActiveNestedV1
            or (self.active.pair_start, self.active.pair_count) != (self.pair_start, self.pair_count)
            or self.active.family is not self.family
            or self.active.mode is not self.mode
        ):
            raise G17ProductionEnvelopeError("active A validation differs from descriptor window/family/mode")

    @property
    def payload(self) -> bytes:
        return b"" if self.active is None else self.active.payload

    @property
    def stop(self) -> int:
        return self.pair_start + self.pair_count

    @classmethod
    def canonical_pass(cls, pair_start: int, pair_count: int) -> G17ADescriptorV1:
        return cls(pair_start, pair_count, G17AFamily.CANONICAL_PASS, G17AMode.PASS_P0)


@dataclass(frozen=True, slots=True)
class G17EncodedDescriptorV1:
    descriptor_bytes: bytes
    pair_start: int
    pair_count: int
    family_wire: int
    mode_wire: int
    payload_offset: int
    payload: bytes
    payload_crc32: int

    @property
    def payload_sha256(self) -> str:
        return _sha256(self.payload)

    @property
    def ordered_leaf_sha256(self) -> str:
        return _sha256(b"G17-SHARD-LEAF-V1\0" + self.descriptor_bytes + self.payload)


@dataclass(frozen=True, slots=True)
class ParsedG17GPacketV1:
    packet: bytes
    layout: Literal[G17PopulationLayout.SHARDED]
    pair_start: int
    pair_count: int
    parent_binding_sha256: str
    descriptors: tuple[G17GDescriptorV1, ...]
    encoded_descriptors: tuple[G17EncodedDescriptorV1, ...]
    crc32: int

    @property
    def packet_sha256(self) -> str:
        return _sha256(self.packet)

    @property
    def descriptor_windows(self) -> tuple[tuple[int, int], ...]:
        return tuple((row.pair_start, row.pair_count) for row in self.descriptors)

    @property
    def descriptor_window_root_sha256(self) -> str:
        return _descriptor_window_root("G", self.descriptor_windows)

    @property
    def order_root_sha256(self) -> str:
        leaves = b"".join(bytes.fromhex(row.ordered_leaf_sha256) for row in self.encoded_descriptors)
        return _sha256(b"G17-G-ORDER-ROOT-V1\0" + leaves)

    @property
    def population_pair_order_sha256(self) -> str:
        return g17_population_pair_order_sha256(self.pair_start, self.pair_count)

    @property
    def semantic_summary(self) -> G17SemanticSummaryV1:
        families = {row.family for row in self.descriptors}
        if G17GFamily.EXACT_SEMANTIC_DIAGNOSTIC in families:
            if G17GFamily.SELECTIVE_ROW3 in families:
                raise G17ProductionEnvelopeError("selective and exact-diagnostic G shards cannot mix")
            return G17SemanticSummaryV1.EXACT_DIAGNOSTIC
        if G17GFamily.SELECTIVE_ROW3 in families:
            return G17SemanticSummaryV1.SELECTIVE
        return G17SemanticSummaryV1.PASS

    @property
    def g8_summary(self) -> G17G8SummaryV1:
        fresh_count = sum(row.mode is G17GMode.SEMANTIC_THEN_FRESH_G8 for row in self.descriptors)
        if fresh_count == 0:
            return G17G8SummaryV1.NONE
        if fresh_count == len(self.descriptors):
            return G17G8SummaryV1.FRESH
        return G17G8SummaryV1.MIXED


@dataclass(frozen=True, slots=True)
class ParsedG17APacketV1:
    packet: bytes
    layout: G17PopulationLayout
    pair_start: int
    pair_count: int
    parent_binding_sha256: str
    descriptors: tuple[G17ADescriptorV1, ...]
    encoded_descriptors: tuple[G17EncodedDescriptorV1, ...]
    crc32: int

    @property
    def packet_sha256(self) -> str:
        return _sha256(self.packet)

    @property
    def descriptor_windows(self) -> tuple[tuple[int, int], ...]:
        return tuple((row.pair_start, row.pair_count) for row in self.descriptors)

    @property
    def descriptor_window_root_sha256(self) -> str:
        return _descriptor_window_root("A", self.descriptor_windows)

    @property
    def order_root_sha256(self) -> str:
        leaves = b"".join(bytes.fromhex(row.ordered_leaf_sha256) for row in self.encoded_descriptors)
        return _sha256(b"G17-A-ORDER-ROOT-V1\0" + leaves)

    @property
    def population_pair_order_sha256(self) -> str:
        return g17_population_pair_order_sha256(self.pair_start, self.pair_count)


def derive_g17_a_parent_binding(p_section: bytes, g_section: bytes) -> str:
    return _sha256(b"G17-A-PARENT-V1\0" + bytes.fromhex(_sha256(p_section)) + bytes.fromhex(_sha256(g_section)))


def _encode_population_packet(
    *,
    magic: bytes,
    layout: G17PopulationLayout,
    pair_start: int,
    pair_count: int,
    parent_binding_sha256: str,
    rows: Sequence[G17GDescriptorV1 | G17ADescriptorV1],
) -> bytes:
    canonical_windows = canonical_g17_shard_windows(pair_start, pair_count)
    if not rows:
        raise G17ProductionEnvelopeError("population packet cannot have a zero-entry directory")
    if magic == G_PACKET_MAGIC:
        if layout is not G17PopulationLayout.SHARDED or any(type(row) is not G17GDescriptorV1 for row in rows):
            raise G17ProductionEnvelopeError("G packet requires sharded exact G descriptor types")
        required_windows = canonical_windows
    else:
        if any(type(row) is not G17ADescriptorV1 for row in rows):
            raise G17ProductionEnvelopeError("A packet requires exact A descriptor types")
        required_windows = canonical_windows if layout is G17PopulationLayout.SHARDED else ((pair_start, pair_count),)
        if layout is G17PopulationLayout.GLOBAL:
            only = rows[0] if len(rows) == 1 else None
            if (
                only is None
                or type(only) is not G17ADescriptorV1
                or only.family is not G17AFamily.G17_GENERAL_CONDITIONAL_XIP2
                or only.mode not in {G17AMode.GLOBAL_COPY_FINAL_Y1, G17AMode.QUANTIZED_XIP2}
            ):
                raise G17ProductionEnvelopeError("global A requires one full-population TACX2A4 descriptor")
    observed_windows = tuple((row.pair_start, row.pair_count) for row in rows)
    if observed_windows != required_windows:
        raise G17ProductionEnvelopeError("descriptor windows are not the unique canonical partition/layout")
    parent = _require_sha256(parent_binding_sha256, name="population parent binding")
    payload_body = b"".join(row.payload for row in rows)
    if len(payload_body) > _MAX_NESTED_PAYLOAD_BYTES:
        raise G17ProductionEnvelopeError("population nested payload body exceeds V1 bound")
    header = GA_HEADER.pack(
        magic,
        PACKET_VERSION,
        int(layout),
        pair_start,
        pair_count,
        len(rows),
        GA_DESCRIPTOR.size * len(rows),
        len(payload_body),
        bytes.fromhex(parent),
        bytes.fromhex(_sha256(payload_body)),
    )
    offset = 0
    directory: list[bytes] = []
    for row in rows:
        payload = row.payload
        directory.append(
            GA_DESCRIPTOR.pack(
                row.pair_start,
                row.pair_count,
                int(row.family),
                int(row.mode),
                offset,
                len(payload),
                bytes.fromhex(_sha256(payload)),
                zlib.crc32(payload) & 0xFFFFFFFF,
            )
        )
        offset += len(payload)
    without_footer = header + b"".join(directory) + payload_body
    return without_footer + GA_FOOTER.pack(zlib.crc32(without_footer) & 0xFFFFFFFF)


def build_g17_g_packet(
    *,
    p_section: bytes,
    pair_start: int,
    pair_count: int,
    descriptors: Sequence[G17GDescriptorV1] | None = None,
) -> bytes:
    _require_bytes(p_section, name="P section")
    rows = (
        tuple(descriptors)
        if descriptors is not None
        else tuple(
            G17GDescriptorV1.canonical_pass(start, count)
            for start, count in canonical_g17_shard_windows(pair_start, pair_count)
        )
    )
    packet = _encode_population_packet(
        magic=G_PACKET_MAGIC,
        layout=G17PopulationLayout.SHARDED,
        pair_start=pair_start,
        pair_count=pair_count,
        parent_binding_sha256=_sha256(p_section),
        rows=rows,
    )
    parsed = parse_g17_g_packet(packet, expected_p_section=p_section, active_parser=_typed_g_reopener(rows))
    if parsed.packet != packet:
        raise G17ProductionEnvelopeError("G packet changed on strict build parse-back")
    return packet


def build_g17_a_packet(
    *,
    p_section: bytes,
    g_section: bytes,
    pair_start: int,
    pair_count: int,
    layout: G17PopulationLayout = G17PopulationLayout.SHARDED,
    descriptors: Sequence[G17ADescriptorV1] | None = None,
) -> bytes:
    _require_bytes(p_section, name="P section")
    _require_bytes(g_section, name="G section")
    if type(layout) is not G17PopulationLayout:
        raise G17ProductionEnvelopeError("A layout must use the exact G17PopulationLayout type")
    if descriptors is None:
        if layout is not G17PopulationLayout.SHARDED:
            raise G17ProductionEnvelopeError("global A has no implicit PASS encoding")
        rows: tuple[G17ADescriptorV1, ...] = tuple(
            G17ADescriptorV1.canonical_pass(start, count)
            for start, count in canonical_g17_shard_windows(pair_start, pair_count)
        )
    else:
        rows = tuple(descriptors)
    packet = _encode_population_packet(
        magic=A_PACKET_MAGIC,
        layout=layout,
        pair_start=pair_start,
        pair_count=pair_count,
        parent_binding_sha256=derive_g17_a_parent_binding(p_section, g_section),
        rows=rows,
    )
    parsed = parse_g17_a_packet(
        packet,
        expected_p_section=p_section,
        expected_g_section=g_section,
        active_parser=_typed_a_reopener(rows),
    )
    if parsed.packet != packet:
        raise G17ProductionEnvelopeError("A packet changed on strict build parse-back")
    return packet


def _typed_g_reopener(rows: Sequence[G17GDescriptorV1]) -> G17GStrictNestedParser | None:
    active = {
        (row.pair_start, row.pair_count, row.family, row.mode): row.active for row in rows if row.active is not None
    }
    if not active:
        return None

    def reopen(payload: bytes, start: int, count: int, family: G17GFamily, mode: G17GMode) -> G17GActiveNestedV1:
        result = active.get((start, count, family, mode))
        if result is None or result.payload != payload:
            raise G17ProductionEnvelopeError("active G bytes differ from typed builder validation")
        return result

    return reopen


def _typed_a_reopener(rows: Sequence[G17ADescriptorV1]) -> G17AStrictNestedParser | None:
    active = {
        (row.pair_start, row.pair_count, row.family, row.mode): row.active for row in rows if row.active is not None
    }
    if not active:
        return None

    def reopen(payload: bytes, start: int, count: int, family: G17AFamily, mode: G17AMode) -> G17AActiveNestedV1:
        result = active.get((start, count, family, mode))
        if result is None or result.payload != payload:
            raise G17ProductionEnvelopeError("active A bytes differ from typed builder validation")
        return result

    return reopen


def _parse_population_fixed(packet: bytes, *, expected_magic: bytes) -> tuple[Any, ...]:
    _require_bytes(packet, name="population packet")
    minimum = GA_HEADER.size + GA_DESCRIPTOR.size + GA_FOOTER.size
    if len(packet) < minimum:
        raise G17ProductionEnvelopeError("population packet is truncated")
    try:
        header = GA_HEADER.unpack_from(packet)
    except struct.error as exc:
        raise G17ProductionEnvelopeError("population header is malformed") from exc
    (
        magic,
        version,
        layout_wire,
        pair_start,
        pair_count,
        entry_count,
        directory_bytes,
        payload_bytes,
        parent,
        body_hash,
    ) = header
    if magic != expected_magic or version != PACKET_VERSION:
        raise G17ProductionEnvelopeError("population magic/version changed")
    try:
        layout = G17PopulationLayout(layout_wire)
    except ValueError as exc:
        raise G17ProductionEnvelopeError("population layout is outside the closed enum") from exc
    canonical_g17_shard_windows(pair_start, pair_count)
    if entry_count < 1 or directory_bytes != entry_count * GA_DESCRIPTOR.size:
        raise G17ProductionEnvelopeError("population directory size/count is noncanonical")
    expected_length = GA_HEADER.size + directory_bytes + payload_bytes + GA_FOOTER.size
    if len(packet) != expected_length:
        raise G17ProductionEnvelopeError("population packet length is not exact EOF")
    without_footer = packet[: -GA_FOOTER.size]
    (footer_crc,) = GA_FOOTER.unpack_from(packet, len(packet) - GA_FOOTER.size)
    if zlib.crc32(without_footer) & 0xFFFFFFFF != footer_crc:
        raise G17ProductionEnvelopeError("population packet CRC mismatch")
    body_start = GA_HEADER.size + directory_bytes
    body = packet[body_start : body_start + payload_bytes]
    if _sha256(body) != body_hash.hex():
        raise G17ProductionEnvelopeError("population payload-body SHA-256 mismatch")
    return layout, pair_start, pair_count, entry_count, parent.hex(), body_start, body, footer_crc


def parse_g17_g_packet(
    packet: bytes,
    *,
    expected_p_section: bytes,
    active_parser: G17GStrictNestedParser | None = None,
) -> ParsedG17GPacketV1:
    _require_bytes(expected_p_section, name="expected P section")
    layout, pair_start, pair_count, entry_count, parent, body_start, body, footer_crc = _parse_population_fixed(
        packet,
        expected_magic=G_PACKET_MAGIC,
    )
    if layout is not G17PopulationLayout.SHARDED:
        raise G17ProductionEnvelopeError("G packet layout must be SHARDED")
    if parent != _sha256(expected_p_section):
        raise G17ProductionEnvelopeError("G parent binding differs from exact P bytes")
    descriptors: list[G17GDescriptorV1] = []
    encoded: list[G17EncodedDescriptorV1] = []
    offset = 0
    for index in range(entry_count):
        raw = packet[GA_HEADER.size + index * GA_DESCRIPTOR.size : GA_HEADER.size + (index + 1) * GA_DESCRIPTOR.size]
        start, count, family_wire, mode_wire, payload_offset, payload_bytes, digest, crc = GA_DESCRIPTOR.unpack(raw)
        if payload_offset != offset or payload_offset + payload_bytes > len(body):
            raise G17ProductionEnvelopeError("G descriptor payload offsets are not exact contiguous slices")
        payload = body[payload_offset : payload_offset + payload_bytes]
        if _sha256(payload) != digest.hex() or zlib.crc32(payload) & 0xFFFFFFFF != crc:
            raise G17ProductionEnvelopeError("G descriptor payload hash/CRC mismatch")
        try:
            family = G17GFamily(family_wire)
            mode = G17GMode(mode_wire)
        except ValueError as exc:
            raise G17ProductionEnvelopeError("G descriptor contains an unknown family/mode") from exc
        _validate_g_family_mode(family, mode, payload)
        active = None
        if payload:
            if active_parser is None:
                raise G17ProductionEnvelopeBlocker(
                    G17ProductionEnvelopeBlockerCode.ACTIVE_NESTED_STRICT_PARSER_OWED,
                    "active G descriptor requires a frozen-public strict parser adapter",
                )
            active = active_parser(payload, start, count, family, mode)
            if type(active) is not G17GActiveNestedV1 or active.payload != payload:
                raise G17ProductionEnvelopeError("active G parser returned a foreign typed validation")
        descriptors.append(G17GDescriptorV1(start, count, family, mode, active))
        encoded.append(G17EncodedDescriptorV1(raw, start, count, family_wire, mode_wire, payload_offset, payload, crc))
        offset += payload_bytes
    if offset != len(body):
        raise G17ProductionEnvelopeError("G payload body has trailing or unconsumed bytes")
    if tuple((row.pair_start, row.pair_count) for row in descriptors) != canonical_g17_shard_windows(
        pair_start, pair_count
    ):
        raise G17ProductionEnvelopeError("G descriptors do not equal the canonical shard partition")
    parsed = ParsedG17GPacketV1(
        packet,
        G17PopulationLayout.SHARDED,
        pair_start,
        pair_count,
        parent,
        tuple(descriptors),
        tuple(encoded),
        footer_crc,
    )
    rebuilt = _encode_population_packet(
        magic=G_PACKET_MAGIC,
        layout=G17PopulationLayout.SHARDED,
        pair_start=pair_start,
        pair_count=pair_count,
        parent_binding_sha256=parent,
        rows=parsed.descriptors,
    )
    if rebuilt != packet:
        raise G17ProductionEnvelopeError("G packet changed on strict parse/re-encode")
    return parsed


def parse_g17_a_packet(
    packet: bytes,
    *,
    expected_p_section: bytes,
    expected_g_section: bytes,
    active_parser: G17AStrictNestedParser | None = None,
) -> ParsedG17APacketV1:
    _require_bytes(expected_p_section, name="expected P section")
    _require_bytes(expected_g_section, name="expected G section")
    layout, pair_start, pair_count, entry_count, parent, body_start, body, footer_crc = _parse_population_fixed(
        packet,
        expected_magic=A_PACKET_MAGIC,
    )
    if parent != derive_g17_a_parent_binding(expected_p_section, expected_g_section):
        raise G17ProductionEnvelopeError("A parent binding differs from exact P/G bytes")
    descriptors: list[G17ADescriptorV1] = []
    encoded: list[G17EncodedDescriptorV1] = []
    offset = 0
    for index in range(entry_count):
        raw = packet[GA_HEADER.size + index * GA_DESCRIPTOR.size : GA_HEADER.size + (index + 1) * GA_DESCRIPTOR.size]
        start, count, family_wire, mode_wire, payload_offset, payload_bytes, digest, crc = GA_DESCRIPTOR.unpack(raw)
        if payload_offset != offset or payload_offset + payload_bytes > len(body):
            raise G17ProductionEnvelopeError("A descriptor payload offsets are not exact contiguous slices")
        payload = body[payload_offset : payload_offset + payload_bytes]
        if _sha256(payload) != digest.hex() or zlib.crc32(payload) & 0xFFFFFFFF != crc:
            raise G17ProductionEnvelopeError("A descriptor payload hash/CRC mismatch")
        try:
            family = G17AFamily(family_wire)
            mode = G17AMode(mode_wire)
        except ValueError as exc:
            raise G17ProductionEnvelopeError("A descriptor contains an unknown family/mode") from exc
        _validate_a_family_mode(family, mode, payload)
        active = None
        if payload:
            if active_parser is None:
                raise G17ProductionEnvelopeBlocker(
                    G17ProductionEnvelopeBlockerCode.ACTIVE_NESTED_STRICT_PARSER_OWED,
                    "active A descriptor requires a frozen-public strict parser adapter",
                )
            active = active_parser(payload, start, count, family, mode)
            if type(active) is not G17AActiveNestedV1 or active.payload != payload:
                raise G17ProductionEnvelopeError("active A parser returned a foreign typed validation")
        descriptors.append(G17ADescriptorV1(start, count, family, mode, active))
        encoded.append(G17EncodedDescriptorV1(raw, start, count, family_wire, mode_wire, payload_offset, payload, crc))
        offset += payload_bytes
    if offset != len(body):
        raise G17ProductionEnvelopeError("A payload body has trailing or unconsumed bytes")
    windows = tuple((row.pair_start, row.pair_count) for row in descriptors)
    required = (
        canonical_g17_shard_windows(pair_start, pair_count)
        if layout is G17PopulationLayout.SHARDED
        else ((pair_start, pair_count),)
    )
    if windows != required:
        raise G17ProductionEnvelopeError("A descriptors do not equal the canonical selected layout")
    if layout is G17PopulationLayout.GLOBAL:
        only = descriptors[0] if len(descriptors) == 1 else None
        if (
            only is None
            or only.family is not G17AFamily.G17_GENERAL_CONDITIONAL_XIP2
            or only.mode not in {G17AMode.GLOBAL_COPY_FINAL_Y1, G17AMode.QUANTIZED_XIP2}
        ):
            raise G17ProductionEnvelopeError("global A is not one exact full-population TACX2A4")
    parsed = ParsedG17APacketV1(
        packet, layout, pair_start, pair_count, parent, tuple(descriptors), tuple(encoded), footer_crc
    )
    rebuilt = _encode_population_packet(
        magic=A_PACKET_MAGIC,
        layout=layout,
        pair_start=pair_start,
        pair_count=pair_count,
        parent_binding_sha256=parent,
        rows=parsed.descriptors,
    )
    if rebuilt != packet:
        raise G17ProductionEnvelopeError("A packet changed on strict parse/re-encode")
    return parsed


def _derive_terminal_summaries(
    g_packet: ParsedG17GPacketV1,
    a_packet: ParsedG17APacketV1,
) -> tuple[G17TerminalSemanticMode, G17TerminalG8Mode, G17TerminalAFamily, G17TerminalAMode]:
    g_families = {row.family for row in g_packet.descriptors}
    if G17GFamily.EXACT_SEMANTIC_DIAGNOSTIC in g_families:
        if G17GFamily.SELECTIVE_ROW3 in g_families:
            raise G17ProductionEnvelopeError("selective and exact-diagnostic G shards cannot mix")
        semantic = G17TerminalSemanticMode.EXACT_DIAGNOSTIC
    elif G17GFamily.SELECTIVE_ROW3 in g_families:
        semantic = G17TerminalSemanticMode.SELECTIVE
    else:
        semantic = G17TerminalSemanticMode.PASS
    fresh_count = sum(row.mode is G17GMode.SEMANTIC_THEN_FRESH_G8 for row in g_packet.descriptors)
    g8 = (
        G17TerminalG8Mode.NONE
        if fresh_count == 0
        else G17TerminalG8Mode.FRESH
        if fresh_count == len(g_packet.descriptors)
        else G17TerminalG8Mode.MIXED
    )
    active_pairs: set[tuple[G17TerminalAFamily, G17TerminalAMode]] = set()
    family_map = {
        G17AFamily.NATIVE_PASS_CONDITIONAL: G17TerminalAFamily.NATIVE,
        G17AFamily.NATIVE_SELECTIVE_NO_G8: G17TerminalAFamily.NATIVE,
        G17AFamily.NATIVE_SELECTIVE_POST_G8: G17TerminalAFamily.NATIVE,
        G17AFamily.G13_PASS_SOURCE_XIP2: G17TerminalAFamily.G13,
        G17AFamily.G17_GENERAL_CONDITIONAL_XIP2: G17TerminalAFamily.G17_GENERAL,
    }
    mode_map = {
        G17AMode.SPARSE_CONSTANT_RGB: G17TerminalAMode.SPARSE_CONSTANT,
        G17AMode.COPY_FINAL_Y1_SUPPORT: G17TerminalAMode.COPY_SUPPORT,
        G17AMode.GLOBAL_COPY_FINAL_Y1: G17TerminalAMode.GLOBAL_COPY,
        G17AMode.QUANTIZED_XIP2: G17TerminalAMode.QUANTIZED_XIP2,
    }
    for row in a_packet.descriptors:
        if row.family is not G17AFamily.CANONICAL_PASS:
            active_pairs.add((family_map[row.family], mode_map[row.mode]))
    if not active_pairs:
        return semantic, g8, G17TerminalAFamily.PASS, G17TerminalAMode.PASS
    if len(active_pairs) > 1:
        return semantic, g8, G17TerminalAFamily.MIXED, G17TerminalAMode.MIXED
    a_family, a_mode = next(iter(active_pairs))
    return semantic, g8, a_family, a_mode


def derive_g17_population_binding(
    *, pair_start: int, pair_count: int, p_section: bytes, g_section: bytes, a_section: bytes
) -> str:
    canonical_g17_shard_windows(pair_start, pair_count)
    return _sha256(
        b"G17-POPULATION-V1\0"
        + struct.pack(">HH", pair_start, pair_count)
        + bytes.fromhex(_sha256(p_section))
        + bytes.fromhex(_sha256(g_section))
        + bytes.fromhex(_sha256(a_section))
    )


@dataclass(frozen=True, slots=True)
class ParsedG17TerminalEnvelopeV1:
    packet: bytes
    semantic_mode: G17TerminalSemanticMode
    g8_mode: G17TerminalG8Mode
    realization_extension: Literal[G17RealizationExtension.NONE]
    a_family: G17TerminalAFamily
    a_mode: G17TerminalAMode
    pair_start: int
    pair_count: int
    p_section_sha256: str
    g_section_sha256: str
    a_section_sha256: str
    population_binding_sha256: str
    crc32: int

    @property
    def packet_sha256(self) -> str:
        return _sha256(self.packet)


def build_g17_terminal_envelope(
    *,
    p_section: bytes,
    g_section: bytes,
    a_section: bytes,
    g_active_parser: G17GStrictNestedParser | None = None,
    a_active_parser: G17AStrictNestedParser | None = None,
) -> bytes:
    g_packet = parse_g17_g_packet(
        g_section,
        expected_p_section=p_section,
        active_parser=g_active_parser,
    )
    a_packet = parse_g17_a_packet(
        a_section,
        expected_p_section=p_section,
        expected_g_section=g_section,
        active_parser=a_active_parser,
    )
    if (g_packet.pair_start, g_packet.pair_count) != (a_packet.pair_start, a_packet.pair_count):
        raise G17ProductionEnvelopeError("G/A population windows differ")
    semantic, g8, a_family, a_mode = _derive_terminal_summaries(g_packet, a_packet)
    binding = derive_g17_population_binding(
        pair_start=g_packet.pair_start,
        pair_count=g_packet.pair_count,
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
    )
    values = (
        E_PACKET_MAGIC,
        PACKET_VERSION,
        int(semantic),
        int(g8),
        int(G17RealizationExtension.NONE),
        int(a_family),
        int(a_mode),
        0,
        g_packet.pair_start,
        g_packet.pair_count,
        bytes.fromhex(_sha256(p_section)),
        bytes.fromhex(_sha256(g_section)),
        bytes.fromhex(_sha256(a_section)),
        bytes.fromhex(binding),
        0,
    )
    zero_crc_packet = E_PACKET.pack(*values)
    packet = E_PACKET.pack(*values[:-1], zlib.crc32(zero_crc_packet) & 0xFFFFFFFF)
    if (
        parse_g17_terminal_envelope(
            packet,
            p_section=p_section,
            g_section=g_section,
            a_section=a_section,
            g_active_parser=g_active_parser,
            a_active_parser=a_active_parser,
        ).packet
        != packet
    ):
        raise G17ProductionEnvelopeError("terminal envelope failed strict build parse-back")
    return packet


def parse_g17_terminal_envelope(
    packet: bytes,
    *,
    p_section: bytes,
    g_section: bytes,
    a_section: bytes,
    g_active_parser: G17GStrictNestedParser | None = None,
    a_active_parser: G17AStrictNestedParser | None = None,
) -> ParsedG17TerminalEnvelopeV1:
    if type(packet) is not bytes or len(packet) != E_PACKET.size:
        raise G17ProductionEnvelopeError("terminal envelope must be exactly 151 bytes")
    try:
        values = E_PACKET.unpack(packet)
    except struct.error as exc:
        raise G17ProductionEnvelopeError("terminal envelope fixed struct is malformed") from exc
    (
        magic,
        version,
        semantic_wire,
        g8_wire,
        extension_wire,
        a_family_wire,
        a_mode_wire,
        flags,
        pair_start,
        pair_count,
        p_hash,
        g_hash,
        a_hash,
        binding,
        crc,
    ) = values
    if magic != E_PACKET_MAGIC or version != PACKET_VERSION or flags != 0:
        raise G17ProductionEnvelopeError("terminal magic/version/flags changed")
    zeroed = E_PACKET.pack(*values[:-1], 0)
    if zlib.crc32(zeroed) & 0xFFFFFFFF != crc:
        raise G17ProductionEnvelopeError("terminal envelope CRC mismatch")
    try:
        semantic = G17TerminalSemanticMode(semantic_wire)
        g8 = G17TerminalG8Mode(g8_wire)
        extension = G17RealizationExtension(extension_wire)
        a_family = G17TerminalAFamily(a_family_wire)
        a_mode = G17TerminalAMode(a_mode_wire)
    except ValueError as exc:
        raise G17ProductionEnvelopeError("terminal envelope contains an unknown closed discriminator") from exc
    if extension is not G17RealizationExtension.NONE:
        raise G17ProductionEnvelopeError("V1 realization extension must be NONE")
    g_packet = parse_g17_g_packet(g_section, expected_p_section=p_section, active_parser=g_active_parser)
    a_packet = parse_g17_a_packet(
        a_section,
        expected_p_section=p_section,
        expected_g_section=g_section,
        active_parser=a_active_parser,
    )
    if (pair_start, pair_count) != (g_packet.pair_start, g_packet.pair_count) or (pair_start, pair_count) != (
        a_packet.pair_start,
        a_packet.pair_count,
    ):
        raise G17ProductionEnvelopeError("terminal population window differs from exact G/A bytes")
    if (p_hash.hex(), g_hash.hex(), a_hash.hex()) != (_sha256(p_section), _sha256(g_section), _sha256(a_section)):
        raise G17ProductionEnvelopeError("terminal section hashes differ from exact P/G/A bytes")
    expected_binding = derive_g17_population_binding(
        pair_start=pair_start,
        pair_count=pair_count,
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
    )
    if binding.hex() != expected_binding:
        raise G17ProductionEnvelopeError("terminal population binding differs from exact P/G/A bytes")
    expected_summaries = _derive_terminal_summaries(g_packet, a_packet)
    if (semantic, g8, a_family, a_mode) != expected_summaries:
        raise G17ProductionEnvelopeError("terminal summaries are not the exact function of G/A descriptors")
    return ParsedG17TerminalEnvelopeV1(
        packet,
        semantic,
        g8,
        G17RealizationExtension.NONE,
        a_family,
        a_mode,
        pair_start,
        pair_count,
        p_hash.hex(),
        g_hash.hex(),
        a_hash.hex(),
        binding.hex(),
        crc,
    )


POST_TOPOLOGY_RECEIPT_SCHEMA: Final = "tac.taskspace_g17_post_topology_population_receipt.v1"
POST_G8_RECEIPT_SCHEMA: Final = "tac.taskspace_g17_post_g8_population_receipt.v1"
_SEMANTIC_LABEL_SHAPE: Final = (384, 512)
_CAMERA_Y1_SHAPE: Final = (874, 1164, 3)


def _source_pair_ids(pair_start: int, pair_count: int) -> tuple[int, ...]:
    canonical_g17_shard_windows(pair_start, pair_count)
    return tuple(range(pair_start, pair_start + pair_count))


def _require_receipt_source_ids(
    value: object,
    *,
    pair_start: int,
    pair_count: int,
) -> tuple[int, ...]:
    expected = _source_pair_ids(pair_start, pair_count)
    if type(value) is not tuple or value != expected or any(type(item) is not int for item in value):
        raise G17ProductionEnvelopeError("population receipt source IDs differ from canonical pair order")
    return value


def _pair_output_root(
    *,
    domain: bytes,
    source_pair_ids: tuple[int, ...],
    semantic_labels: np.ndarray,
    camera_y1: np.ndarray,
) -> str:
    rows = [
        [pair_id, _array_sha256(semantic_labels[index]), _array_sha256(camera_y1[index])]
        for index, pair_id in enumerate(source_pair_ids)
    ]
    return _sha256(domain + _canonical_json(rows))


@dataclass(frozen=True, slots=True)
class G17PostTopologyPopulationReceiptV1:
    schema: Literal["tac.taskspace_g17_post_topology_population_receipt.v1"]
    population_pair_start: int
    population_pair_count: int
    source_pair_ids: tuple[int, ...]
    p_section_sha256: str
    g_section_sha256: str
    g_descriptor_window_root_sha256: str
    g_order_root_sha256: str
    population_pair_order_sha256: str
    causal_p_receipt_sha256: str
    predictor_state_binding_sha256: str
    semantic_mode: str
    semantic_labels_sha256: str
    post_topology_camera_y1_sha256: str
    post_topology_pair_output_root_sha256: str

    def __post_init__(self) -> None:
        if self.schema != POST_TOPOLOGY_RECEIPT_SCHEMA:
            raise G17ProductionEnvelopeError("post-topology receipt schema changed")
        _require_receipt_source_ids(
            self.source_pair_ids,
            pair_start=self.population_pair_start,
            pair_count=self.population_pair_count,
        )
        for item in fields(self):
            if item.name.endswith("_sha256"):
                _require_sha256(getattr(self, item.name), name=item.name)
        if self.semantic_mode not in {item.value for item in G17SemanticSummaryV1}:
            raise G17ProductionEnvelopeError("post-topology semantic summary escaped its closed set")

    def as_dict(self) -> dict[str, Any]:
        return {
            item.name: list(value) if item.name == "source_pair_ids" else value
            for item in fields(self)
            if (value := getattr(self, item.name)) is not None
        }

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


def _derive_g17_post_topology_population_receipt(
    *,
    p_section_bytes: bytes,
    g_section_bytes: bytes,
    causal_p_receipt_bytes: bytes,
    predictor_state_binding_sha256: str,
    semantic_labels: np.ndarray,
    post_topology_camera_y1: np.ndarray,
    g_active_parser: G17GStrictNestedParser | None,
) -> G17PostTopologyPopulationReceiptV1:
    _require_bytes(p_section_bytes, name="P section")
    _require_bytes(g_section_bytes, name="G section")
    _require_bytes(causal_p_receipt_bytes, name="causal P receipt")
    predictor_binding = _require_sha256(
        predictor_state_binding_sha256,
        name="predictor state binding",
    )
    parsed_g = parse_g17_g_packet(
        g_section_bytes,
        expected_p_section=p_section_bytes,
        active_parser=g_active_parser,
    )
    labels = _immutable_population_array(
        semantic_labels,
        name="semantic_labels",
        pair_count=parsed_g.pair_count,
        expected_tail=_SEMANTIC_LABEL_SHAPE,
    )
    camera = _immutable_population_array(
        post_topology_camera_y1,
        name="post_topology_camera_y1",
        pair_count=parsed_g.pair_count,
        expected_tail=_CAMERA_Y1_SHAPE,
    )
    pair_ids = _source_pair_ids(parsed_g.pair_start, parsed_g.pair_count)
    return G17PostTopologyPopulationReceiptV1(
        schema=POST_TOPOLOGY_RECEIPT_SCHEMA,
        population_pair_start=parsed_g.pair_start,
        population_pair_count=parsed_g.pair_count,
        source_pair_ids=pair_ids,
        p_section_sha256=_sha256(p_section_bytes),
        g_section_sha256=_sha256(g_section_bytes),
        g_descriptor_window_root_sha256=parsed_g.descriptor_window_root_sha256,
        g_order_root_sha256=parsed_g.order_root_sha256,
        population_pair_order_sha256=parsed_g.population_pair_order_sha256,
        causal_p_receipt_sha256=_sha256(causal_p_receipt_bytes),
        predictor_state_binding_sha256=predictor_binding,
        semantic_mode=parsed_g.semantic_summary.value,
        semantic_labels_sha256=_array_sha256(labels),
        post_topology_camera_y1_sha256=_array_sha256(camera),
        post_topology_pair_output_root_sha256=_pair_output_root(
            domain=b"G17-POST-TOPOLOGY-PAIR-OUTPUTS-V1\0",
            source_pair_ids=pair_ids,
            semantic_labels=labels,
            camera_y1=camera,
        ),
    )


def build_g17_post_topology_population_receipt(
    *,
    p_section_bytes: bytes,
    g_section_bytes: bytes,
    causal_p_receipt_bytes: bytes,
    predictor_state_binding_sha256: str,
    semantic_labels: np.ndarray,
    post_topology_camera_y1: np.ndarray,
    g_active_parser: G17GStrictNestedParser | None = None,
) -> G17PostTopologyPopulationReceiptV1:
    receipt = _derive_g17_post_topology_population_receipt(
        p_section_bytes=p_section_bytes,
        g_section_bytes=g_section_bytes,
        causal_p_receipt_bytes=causal_p_receipt_bytes,
        predictor_state_binding_sha256=predictor_state_binding_sha256,
        semantic_labels=semantic_labels,
        post_topology_camera_y1=post_topology_camera_y1,
        g_active_parser=g_active_parser,
    )
    return parse_g17_post_topology_population_receipt(
        receipt.to_receipt_bytes(),
        p_section_bytes=p_section_bytes,
        g_section_bytes=g_section_bytes,
        causal_p_receipt_bytes=causal_p_receipt_bytes,
        predictor_state_binding_sha256=predictor_state_binding_sha256,
        semantic_labels=semantic_labels,
        post_topology_camera_y1=post_topology_camera_y1,
        g_active_parser=g_active_parser,
    )


def parse_g17_post_topology_population_receipt(
    payload: bytes,
    *,
    p_section_bytes: bytes,
    g_section_bytes: bytes,
    causal_p_receipt_bytes: bytes,
    predictor_state_binding_sha256: str,
    semantic_labels: np.ndarray,
    post_topology_camera_y1: np.ndarray,
    g_active_parser: G17GStrictNestedParser | None = None,
) -> G17PostTopologyPopulationReceiptV1:
    expected = {item.name for item in fields(G17PostTopologyPopulationReceiptV1)}
    value = _strict_canonical_json(
        payload,
        expected_fields=expected,
        schema=POST_TOPOLOGY_RECEIPT_SCHEMA,
    )
    if type(value["source_pair_ids"]) is not list:
        raise G17ProductionEnvelopeError("post-topology source IDs must be a JSON list")
    value["source_pair_ids"] = tuple(value["source_pair_ids"])
    try:
        observed = G17PostTopologyPopulationReceiptV1(**value)
    except (TypeError, ValueError) as exc:
        raise G17ProductionEnvelopeError("post-topology receipt has invalid typed fields") from exc
    expected_receipt = _derive_g17_post_topology_population_receipt(
        p_section_bytes=p_section_bytes,
        g_section_bytes=g_section_bytes,
        causal_p_receipt_bytes=causal_p_receipt_bytes,
        predictor_state_binding_sha256=predictor_state_binding_sha256,
        semantic_labels=semantic_labels,
        post_topology_camera_y1=post_topology_camera_y1,
        g_active_parser=g_active_parser,
    )
    if observed != expected_receipt or observed.to_receipt_bytes() != payload:
        raise G17ProductionEnvelopeError("post-topology receipt differs from exact P/G/array custody")
    return observed


@dataclass(frozen=True, slots=True)
class G17PostG8PopulationReceiptV1:
    schema: Literal["tac.taskspace_g17_post_g8_population_receipt.v1"]
    post_topology_population_receipt_sha256: str
    population_pair_start: int
    population_pair_count: int
    source_pair_ids: tuple[int, ...]
    p_section_sha256: str
    g_section_sha256: str
    g_order_root_sha256: str
    population_pair_order_sha256: str
    g8_mode: str
    semantic_labels_sha256: str
    post_g8_camera_y1_sha256: str
    post_g8_pair_output_root_sha256: str

    def __post_init__(self) -> None:
        if self.schema != POST_G8_RECEIPT_SCHEMA:
            raise G17ProductionEnvelopeError("post-G8 receipt schema changed")
        _require_receipt_source_ids(
            self.source_pair_ids,
            pair_start=self.population_pair_start,
            pair_count=self.population_pair_count,
        )
        for item in fields(self):
            if item.name.endswith("_sha256"):
                _require_sha256(getattr(self, item.name), name=item.name)
        if self.g8_mode not in {G17G8SummaryV1.FRESH.value, G17G8SummaryV1.MIXED.value}:
            raise G17ProductionEnvelopeError("post-G8 receipt cannot represent a NONE G8 chronology")

    def as_dict(self) -> dict[str, Any]:
        return {
            item.name: list(value) if item.name == "source_pair_ids" else value
            for item in fields(self)
            if (value := getattr(self, item.name)) is not None
        }

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


def _derive_g17_post_g8_population_receipt(
    *,
    post_topology_receipt: G17PostTopologyPopulationReceiptV1,
    p_section_bytes: bytes,
    g_section_bytes: bytes,
    semantic_labels: np.ndarray,
    post_g8_camera_y1: np.ndarray,
    g_active_parser: G17GStrictNestedParser | None,
) -> G17PostG8PopulationReceiptV1:
    if type(post_topology_receipt) is not G17PostTopologyPopulationReceiptV1:
        raise G17ProductionEnvelopeError("post-G8 chronology requires exact post-topology receipt type")
    parsed_g = parse_g17_g_packet(
        g_section_bytes,
        expected_p_section=p_section_bytes,
        active_parser=g_active_parser,
    )
    if parsed_g.g8_summary is G17G8SummaryV1.NONE:
        raise G17ProductionEnvelopeError("NONE G8 chronology must not emit a post-G8 receipt")
    if (
        post_topology_receipt.p_section_sha256 != _sha256(p_section_bytes)
        or post_topology_receipt.g_section_sha256 != _sha256(g_section_bytes)
        or post_topology_receipt.source_pair_ids != _source_pair_ids(parsed_g.pair_start, parsed_g.pair_count)
    ):
        raise G17ProductionEnvelopeError("post-G8 chronology received a foreign post-topology receipt")
    labels = _immutable_population_array(
        semantic_labels,
        name="semantic_labels",
        pair_count=parsed_g.pair_count,
        expected_tail=_SEMANTIC_LABEL_SHAPE,
    )
    camera = _immutable_population_array(
        post_g8_camera_y1,
        name="post_g8_camera_y1",
        pair_count=parsed_g.pair_count,
        expected_tail=_CAMERA_Y1_SHAPE,
    )
    if _array_sha256(labels) != post_topology_receipt.semantic_labels_sha256:
        raise G17ProductionEnvelopeError("post-G8 semantic labels differ from post-topology chronology")
    pair_ids = post_topology_receipt.source_pair_ids
    return G17PostG8PopulationReceiptV1(
        schema=POST_G8_RECEIPT_SCHEMA,
        post_topology_population_receipt_sha256=post_topology_receipt.receipt_sha256,
        population_pair_start=parsed_g.pair_start,
        population_pair_count=parsed_g.pair_count,
        source_pair_ids=pair_ids,
        p_section_sha256=_sha256(p_section_bytes),
        g_section_sha256=_sha256(g_section_bytes),
        g_order_root_sha256=parsed_g.order_root_sha256,
        population_pair_order_sha256=parsed_g.population_pair_order_sha256,
        g8_mode=parsed_g.g8_summary.value,
        semantic_labels_sha256=_array_sha256(labels),
        post_g8_camera_y1_sha256=_array_sha256(camera),
        post_g8_pair_output_root_sha256=_pair_output_root(
            domain=b"G17-POST-G8-PAIR-OUTPUTS-V1\0",
            source_pair_ids=pair_ids,
            semantic_labels=labels,
            camera_y1=camera,
        ),
    )


def build_g17_post_g8_population_receipt(
    *,
    post_topology_receipt: G17PostTopologyPopulationReceiptV1,
    p_section_bytes: bytes,
    g_section_bytes: bytes,
    semantic_labels: np.ndarray,
    post_g8_camera_y1: np.ndarray,
    g_active_parser: G17GStrictNestedParser | None = None,
) -> G17PostG8PopulationReceiptV1:
    receipt = _derive_g17_post_g8_population_receipt(
        post_topology_receipt=post_topology_receipt,
        p_section_bytes=p_section_bytes,
        g_section_bytes=g_section_bytes,
        semantic_labels=semantic_labels,
        post_g8_camera_y1=post_g8_camera_y1,
        g_active_parser=g_active_parser,
    )
    return parse_g17_post_g8_population_receipt(
        receipt.to_receipt_bytes(),
        post_topology_receipt=post_topology_receipt,
        p_section_bytes=p_section_bytes,
        g_section_bytes=g_section_bytes,
        semantic_labels=semantic_labels,
        post_g8_camera_y1=post_g8_camera_y1,
        g_active_parser=g_active_parser,
    )


def parse_g17_post_g8_population_receipt(
    payload: bytes,
    *,
    post_topology_receipt: G17PostTopologyPopulationReceiptV1,
    p_section_bytes: bytes,
    g_section_bytes: bytes,
    semantic_labels: np.ndarray,
    post_g8_camera_y1: np.ndarray,
    g_active_parser: G17GStrictNestedParser | None = None,
) -> G17PostG8PopulationReceiptV1:
    expected = {item.name for item in fields(G17PostG8PopulationReceiptV1)}
    value = _strict_canonical_json(
        payload,
        expected_fields=expected,
        schema=POST_G8_RECEIPT_SCHEMA,
    )
    if type(value["source_pair_ids"]) is not list:
        raise G17ProductionEnvelopeError("post-G8 source IDs must be a JSON list")
    value["source_pair_ids"] = tuple(value["source_pair_ids"])
    try:
        observed = G17PostG8PopulationReceiptV1(**value)
    except (TypeError, ValueError) as exc:
        raise G17ProductionEnvelopeError("post-G8 receipt has invalid typed fields") from exc
    expected_receipt = _derive_g17_post_g8_population_receipt(
        post_topology_receipt=post_topology_receipt,
        p_section_bytes=p_section_bytes,
        g_section_bytes=g_section_bytes,
        semantic_labels=semantic_labels,
        post_g8_camera_y1=post_g8_camera_y1,
        g_active_parser=g_active_parser,
    )
    if observed != expected_receipt or observed.to_receipt_bytes() != payload:
        raise G17ProductionEnvelopeError("post-G8 receipt differs from exact P/G/array custody")
    return observed


@dataclass(frozen=True, slots=True)
class ParsedG17ProductionArchiveV1:
    """Strict structural reopen of one exact four-section G17 archive."""

    outer: ParsedTaskspaceOuterArchive
    member: ParsedTaskspaceMonolithicPGAMemberV1
    g_packet: ParsedG17GPacketV1
    a_packet: ParsedG17APacketV1
    terminal: ParsedG17TerminalEnvelopeV1
    p_section_occurrences: Literal[1] = 1
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False
    pointer_moved: Literal[False] = False

    def __post_init__(self) -> None:
        if type(self.outer) is not ParsedTaskspaceOuterArchive:
            raise G17ProductionEnvelopeError("production archive lost exact outer parse type")
        if type(self.member) is not ParsedTaskspaceMonolithicPGAMemberV1:
            raise G17ProductionEnvelopeError("production archive lost exact member parse type")
        if self.outer.member_bytes != self.member.member_bytes:
            raise G17ProductionEnvelopeError("outer and member parsers reopened different exact bytes")
        if self.p_section_occurrences != 1:
            raise G17ProductionEnvelopeError("G17 production archive must contain P exactly once")
        if (
            self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
            or self.pointer_moved is not False
        ):
            raise G17ProductionEnvelopeError("structural archive truth labels became permissive")

    @property
    def p_section(self) -> bytes:
        return self.member.section(TaskspaceMonolithicPGARole.PREDICTOR).payload

    @property
    def g_section(self) -> bytes:
        return self.member.section(TaskspaceMonolithicPGARole.GENERATIVE_CORRECTION).payload

    @property
    def a_section(self) -> bytes:
        return self.member.section(TaskspaceMonolithicPGARole.COUPLED_PREIMAGE).payload

    @property
    def terminal_section(self) -> bytes:
        return self.member.section(TaskspaceMonolithicPGARole.TERMINAL_QUOTIENT).payload


@dataclass(frozen=True, slots=True)
class G17ProductionArchiveBuildV1:
    outer_build: TaskspaceOuterArchiveBuild
    stored: ParsedG17ProductionArchiveV1
    deflated: ParsedG17ProductionArchiveV1
    selected: ParsedG17ProductionArchiveV1

    def __post_init__(self) -> None:
        if type(self.outer_build) is not TaskspaceOuterArchiveBuild:
            raise G17ProductionEnvelopeError("production build lost exact outer coder race")
        if self.stored.outer != self.outer_build.stored or self.deflated.outer != self.outer_build.deflated:
            raise G17ProductionEnvelopeError("production build wrappers differ from exact STORE/DEFLATE objects")
        expected = self.stored if self.outer_build.selected.encoding is OuterArchiveEncoding.STORED else self.deflated
        if self.selected != expected or self.selected.outer != self.outer_build.selected:
            raise G17ProductionEnvelopeError("production selected object differs from frozen outer-codec rule")


def parse_g17_production_archive(
    archive_bytes: bytes,
    *,
    g_active_parser: G17GStrictNestedParser | None = None,
    a_active_parser: G17AStrictNestedParser | None = None,
    max_member_bytes: int = 64 * 1024 * 1024,
) -> ParsedG17ProductionArchiveV1:
    """Reopen exact ZIP/member/P/G/A/E bytes without making a semantic claim."""

    try:
        outer = parse_taskspace_outer_archive(
            archive_bytes,
            max_member_bytes=max_member_bytes,
        )
        member = parse_taskspace_monolithic_pga_member(
            outer.member_bytes,
            max_member_bytes=max_member_bytes,
        )
    except Exception as exc:
        raise G17ProductionEnvelopeError("strict outer/member parser refused production archive") from exc
    roles = tuple(section.role for section in member.sections)
    required = (
        TaskspaceMonolithicPGARole.PREDICTOR,
        TaskspaceMonolithicPGARole.GENERATIVE_CORRECTION,
        TaskspaceMonolithicPGARole.COUPLED_PREIMAGE,
        TaskspaceMonolithicPGARole.TERMINAL_QUOTIENT,
    )
    if roles != required:
        raise G17ProductionEnvelopeError("G17 production archive requires exact P -> G -> A -> E roles")
    p_section = member.section(TaskspaceMonolithicPGARole.PREDICTOR).payload
    g_section = member.section(TaskspaceMonolithicPGARole.GENERATIVE_CORRECTION).payload
    a_section = member.section(TaskspaceMonolithicPGARole.COUPLED_PREIMAGE).payload
    terminal_section = member.section(TaskspaceMonolithicPGARole.TERMINAL_QUOTIENT).payload
    g_packet = parse_g17_g_packet(
        g_section,
        expected_p_section=p_section,
        active_parser=g_active_parser,
    )
    a_packet = parse_g17_a_packet(
        a_section,
        expected_p_section=p_section,
        expected_g_section=g_section,
        active_parser=a_active_parser,
    )
    terminal = parse_g17_terminal_envelope(
        terminal_section,
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
        g_active_parser=g_active_parser,
        a_active_parser=a_active_parser,
    )
    return ParsedG17ProductionArchiveV1(
        outer=outer,
        member=member,
        g_packet=g_packet,
        a_packet=a_packet,
        terminal=terminal,
    )


def build_g17_production_archive(
    *,
    p_section: bytes,
    g_section: bytes,
    a_section: bytes,
    terminal_section: bytes,
    g_active_parser: G17GStrictNestedParser | None = None,
    a_active_parser: G17AStrictNestedParser | None = None,
    max_member_bytes: int = 64 * 1024 * 1024,
) -> G17ProductionArchiveBuildV1:
    """Build/reopen exact STORE and DEFLATE objects; select only by ZIP bytes."""

    parse_g17_terminal_envelope(
        terminal_section,
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
        g_active_parser=g_active_parser,
        a_active_parser=a_active_parser,
    )
    try:
        outer_build = build_taskspace_monolithic_pga_archive(
            p_section,
            g_section,
            a_section,
            terminal_quotient_packet=terminal_section,
            max_member_bytes=max_member_bytes,
        )
    except Exception as exc:
        raise G17ProductionEnvelopeError("frozen outer builders refused exact G17 sections") from exc
    stored = parse_g17_production_archive(
        outer_build.stored.archive_bytes,
        g_active_parser=g_active_parser,
        a_active_parser=a_active_parser,
        max_member_bytes=max_member_bytes,
    )
    deflated = parse_g17_production_archive(
        outer_build.deflated.archive_bytes,
        g_active_parser=g_active_parser,
        a_active_parser=a_active_parser,
        max_member_bytes=max_member_bytes,
    )
    selected = stored if outer_build.selected.encoding is OuterArchiveEncoding.STORED else deflated
    return G17ProductionArchiveBuildV1(
        outer_build=outer_build,
        stored=stored,
        deflated=deflated,
        selected=selected,
    )


# Compatibility names are identities, not parallel definitions.  G17 has one
# canonical family/mode type and one canonical strict G parser.
G17GFamilyV1 = G17GFamily
G17GModeV1 = G17GMode
parse_g17_g_section = parse_g17_g_packet
