# SPDX-License-Identifier: MIT
"""Typed score-quotient functional contract for DDM family (d).

This module is an additive, research-only contract.  It deliberately stops at
the executable interface boundary: it defines the counted bytes, deterministic
receiver, real-coder rate term, and future-fit request, but it does not train,
dispatch, score, or promote a candidate.

The receiver represents only scorer-visible state:

* two 384x512 RGB scorer planes per pair;
* six pose-target statistics per pair;
* externally priced 25-row demand placements; and
* sparse pixels restricted to the Fisher/argmax at-risk flip annulus.

The 25-row values and their coder prices are owned by DM1.  DC1 accepts those
already-coded records as opaque typed content and never silently reprices them.
"""

from __future__ import annotations

import hashlib
import lzma
import math
import re
import struct
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

import numpy as np

from tac.contest_score import compute_contest_score
from tac.optimization.ddm_min_description_contract import (
    LayerHome,
    StreamType,
    TypedStreamTag,
)
from tac.through_r.resolution_chain import CAMERA_H, CAMERA_W, contest_faithful_R_numpy

SCHEMA: Final = "ddm_score_quotient_functional_contract.v1"
PACKET_MAGIC: Final = b"DDMSQF1\x00"
PACKET_VERSION: Final = 1
FIT_SCHEMA: Final = "DDMEventContinuationV1"
AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
SCORE_CLAIM: Final = False
FRONTIER_POINTER: Final = "0.1910828242 [contest-CPU] UNMOVED"
SCORER_H: Final = 384
SCORER_W: Final = 512
PAIR_COUNT: Final = 600
DEMAND_ROW_COUNT: Final = 25
POSE_STAT_DIM: Final = 6
SEG_CLASS_COUNT: Final = 5
SEG_HEAD_RANK: Final = SEG_CLASS_COUNT - 1
CPU_THREADS: Final = 4
V14_BASELINE_D_SEG: Final = 0.027470296224
V14_BASELINE_ARCHIVE_BYTES: Final = 133_247
REVERSE_WATERFILL_RATE_THRESHOLD: Final = 25.0 / 37_545_489.0
AT_RISK_SCOPE: Final = "AT_RISK_FLIP_ANNULUS"

_OUTER_HEADER: Final = struct.Struct(">8sBII")
_BASE_HEADER: Final = struct.Struct(">BQI")
_SECTION_HEADER: Final = struct.Struct(">BBIII")
_LATENT_HEADER: Final = struct.Struct(">5sH")
_LATENT_ROW: Final = struct.Struct(">H12h")
_PLACEMENT_HEADER: Final = struct.Struct(">5sB")
_PLACEMENT_ROW: Final = struct.Struct(">BHHB32sI")
_EXCEPTION_HEADER: Final = struct.Struct(">5sI")
_EXCEPTION_ROW: Final = struct.Struct(">HBBHHBB")
_NAME_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,126}$")


class ScoreQuotientContractError(ValueError):
    """Malformed packet, address, receipt, or receiver declaration."""


class SectionKind(StrEnum):
    PARAMETERS = "PARAMETERS"
    TEMPORAL_LATENTS = "TEMPORAL_LATENTS"
    DEMAND_PLACEMENTS = "DEMAND_PLACEMENTS"
    EXCEPTIONS = "EXCEPTIONS"


class Coder(StrEnum):
    PASSTHROUGH = "PASSTHROUGH"
    ZLIB_9 = "ZLIB_9"
    LZMA1_1M = "LZMA1_1M"


_SECTION_TO_WIRE: Final = {
    SectionKind.PARAMETERS: 1,
    SectionKind.TEMPORAL_LATENTS: 2,
    SectionKind.DEMAND_PLACEMENTS: 3,
    SectionKind.EXCEPTIONS: 4,
}
_WIRE_TO_SECTION: Final = {value: key for key, value in _SECTION_TO_WIRE.items()}
_CODER_TO_WIRE: Final = {
    Coder.PASSTHROUGH: 0,
    Coder.ZLIB_9: 1,
    Coder.LZMA1_1M: 2,
}
_WIRE_TO_CODER: Final = {value: key for key, value in _CODER_TO_WIRE.items()}
_TYPING: Final = {
    SectionKind.PARAMETERS: (
        StreamType.SKELETON,
        LayerHome.L1_PROGRAM,
        "evaluate.py::inflate archive program/parameters before frame reconstruction",
    ),
    SectionKind.TEMPORAL_LATENTS: (
        StreamType.CONNECTION,
        LayerHome.L2_CHART,
        "evaluate.py::pair-local frame_0/frame_1 reconstruction connection",
    ),
    SectionKind.DEMAND_PLACEMENTS: (
        StreamType.FIBER,
        LayerHome.L3_RASTER,
        "evaluate.py::video-derived placement survives reconstruction and resize",
    ),
    SectionKind.EXCEPTIONS: (
        StreamType.RESIDUAL,
        LayerHome.L4_SCORER_FEATURE,
        "evaluate.py::SegNet/PoseNet scorer-visible at-risk correction",
    ),
}


def _require_int(value: int, field: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ScoreQuotientContractError(f"{field} must be an integer in [{lower},{upper}]")
    return value


def _require_sha256(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ScoreQuotientContractError(f"{field} must be a lowercase SHA-256")
    return value


def _array(value: np.ndarray, shape: tuple[int, ...], dtype: np.dtype, field: str) -> np.ndarray:
    out = np.asarray(value)
    if out.shape != shape or out.dtype != dtype:
        raise ScoreQuotientContractError(
            f"{field} must have shape={shape} dtype={np.dtype(dtype)}; got {out.shape} {out.dtype}"
        )
    out = np.ascontiguousarray(out)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class FunctionalParametersV1:
    """Separable rank-one plane basis and per-frame/channel DC values."""

    base_rgb_u8: np.ndarray
    row_basis_i8: np.ndarray
    col_basis_i8: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "base_rgb_u8", _array(self.base_rgb_u8, (2, 3), np.dtype("uint8"), "base_rgb_u8")
        )
        object.__setattr__(
            self,
            "row_basis_i8",
            _array(self.row_basis_i8, (2, 3, SCORER_H), np.dtype("int8"), "row_basis_i8"),
        )
        object.__setattr__(
            self,
            "col_basis_i8",
            _array(self.col_basis_i8, (2, 3, SCORER_W), np.dtype("int8"), "col_basis_i8"),
        )


@dataclass(frozen=True, order=True, slots=True)
class TemporalLatentV1:
    """One canonical pair address with six plane coefficients and six pose stats."""

    pair_index: int
    coefficients_q8: tuple[int, int, int, int, int, int]
    xi_q12: tuple[int, int, int, int, int, int]

    def __post_init__(self) -> None:
        _require_int(self.pair_index, "pair_index", 0, PAIR_COUNT - 1)
        if len(self.coefficients_q8) != 6 or len(self.xi_q12) != POSE_STAT_DIM:
            raise ScoreQuotientContractError("latent coefficients and xi must each have length 6")
        for field, values in (
            ("coefficients_q8", self.coefficients_q8),
            ("xi_q12", self.xi_q12),
        ):
            for value in values:
                _require_int(value, field, -32768, 32767)


@dataclass(frozen=True, order=True, slots=True)
class ExternallyPricedDemandPlacementV1:
    """One DM1-owned value and coder-price record, passed through without repricing."""

    pair_index: int
    bucket_index: int
    slot_index: int
    coder_id: str
    decoded_sha256: str
    coded_record: bytes

    def __post_init__(self) -> None:
        _require_int(self.pair_index, "placement pair_index", 0, PAIR_COUNT - 1)
        _require_int(self.bucket_index, "placement bucket_index", 0, 65535)
        _require_int(self.slot_index, "placement slot_index", 0, DEMAND_ROW_COUNT - 1)
        if not isinstance(self.coder_id, str) or not _NAME_RE.fullmatch(self.coder_id):
            raise ScoreQuotientContractError("placement coder_id is outside the canonical vocabulary")
        _require_sha256(self.decoded_sha256, "placement decoded_sha256")
        if not isinstance(self.coded_record, bytes) or not self.coded_record:
            raise ScoreQuotientContractError("placement coded_record must be nonempty bytes")

    @property
    def typed_tag(self) -> TypedStreamTag:
        return TypedStreamTag(
            type=StreamType.FIBER,
            layer_home=LayerHome.L3_RASTER,
            evaluate_py_recursion_level_cited=_TYPING[SectionKind.DEMAND_PLACEMENTS][2],
            counted_bytes=len(self.coded_record),
            free_receiver_code=True,
        )


@dataclass(frozen=True, slots=True)
class DecodedDemandPlacementV1:
    """DM1 decoder result whose bytes must match the record's decoded SHA."""

    decoded_payload: bytes
    scorer_planes_u8: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.decoded_payload, bytes) or not self.decoded_payload:
            raise ScoreQuotientContractError(
                "DM1 decoded placement payload must be nonempty bytes"
            )


@dataclass(frozen=True, order=True, slots=True)
class PixelExceptionV1:
    """One canonical scorer-plane correction, only in the at-risk flip annulus."""

    pair_index: int
    frame_index: int
    y: int
    x: int
    channel: int
    value_u8: int
    scope: Literal["AT_RISK_FLIP_ANNULUS"] = AT_RISK_SCOPE

    def __post_init__(self) -> None:
        _require_int(self.pair_index, "exception pair_index", 0, PAIR_COUNT - 1)
        _require_int(self.frame_index, "exception frame_index", 0, 1)
        _require_int(self.y, "exception y", 0, SCORER_H - 1)
        _require_int(self.x, "exception x", 0, SCORER_W - 1)
        _require_int(self.channel, "exception channel", 0, 2)
        _require_int(self.value_u8, "exception value_u8", 0, 255)
        if self.scope != AT_RISK_SCOPE:
            raise ScoreQuotientContractError(
                "exceptions are admissible only in AT_RISK_FLIP_ANNULUS"
            )


@dataclass(frozen=True, slots=True)
class SectionReceipt:
    kind: SectionKind
    coder: Coder
    raw_bytes: int
    coded_bytes: int
    packet_bytes: int
    raw_sha256: str
    typed_tag: TypedStreamTag


@dataclass(frozen=True, slots=True)
class FunctionalPacketReceipt:
    schema: str
    named_base: str
    base_sha256: str
    base_counted_bytes: int
    packet_sha256: str
    packet_bytes: int
    total_counted_bytes: int
    inactive_base_identity: bool
    sections: tuple[SectionReceipt, ...]
    axis: str = AXIS
    score_claim: bool = SCORE_CLAIM
    frontier_pointer: str = FRONTIER_POINTER


@dataclass(frozen=True, slots=True)
class CompiledFunctionalPacketV1:
    payload: bytes
    receipt: FunctionalPacketReceipt


@dataclass(frozen=True, slots=True)
class ParsedFunctionalPacketV1:
    named_base: str
    base_sha256: str
    base_counted_bytes: int
    parameters: FunctionalParametersV1 | None
    temporal_latents: tuple[TemporalLatentV1, ...]
    demand_placements: tuple[ExternallyPricedDemandPlacementV1, ...]
    exceptions: tuple[PixelExceptionV1, ...]


def _real_code(raw: bytes) -> tuple[Coder, bytes]:
    """Choose the shortest deterministic real coder with stable tie breaking."""

    choices = (
        (Coder.PASSTHROUGH, raw),
        (Coder.ZLIB_9, zlib.compress(raw, level=9)),
        (
            Coder.LZMA1_1M,
            lzma.compress(
                raw,
                format=lzma.FORMAT_RAW,
                filters=[
                    {
                        "id": lzma.FILTER_LZMA1,
                        "dict_size": 1 << 20,
                        "lc": 3,
                        "lp": 0,
                        "pb": 2,
                        "mode": lzma.MODE_NORMAL,
                        "nice_len": 64,
                        "mf": lzma.MF_BT4,
                    }
                ],
            ),
        ),
    )
    return min(choices, key=lambda item: (len(item[1]), _CODER_TO_WIRE[item[0]]))


def _decode(coder: Coder, coded: bytes) -> bytes:
    if coder is Coder.PASSTHROUGH:
        return coded
    if coder is Coder.ZLIB_9:
        return zlib.decompress(coded)
    if coder is Coder.LZMA1_1M:
        return lzma.decompress(
            coded,
            format=lzma.FORMAT_RAW,
            filters=[
                {
                    "id": lzma.FILTER_LZMA1,
                    "dict_size": 1 << 20,
                    "lc": 3,
                    "lp": 0,
                    "pb": 2,
                }
            ],
        )
    raise ScoreQuotientContractError(f"unsupported coder {coder}")


def _encode_parameters(value: FunctionalParametersV1) -> bytes:
    return (
        b"SQFP1"
        + value.base_rgb_u8.tobytes(order="C")
        + value.row_basis_i8.tobytes(order="C")
        + value.col_basis_i8.tobytes(order="C")
    )


def _decode_parameters(raw: bytes) -> FunctionalParametersV1:
    expected = 5 + 6 + (2 * 3 * SCORER_H) + (2 * 3 * SCORER_W)
    if len(raw) != expected or raw[:5] != b"SQFP1":
        raise ScoreQuotientContractError("parameter section has invalid magic or exact length")
    cursor = 5
    base = np.frombuffer(raw[cursor : cursor + 6], dtype=np.uint8).reshape(2, 3).copy()
    cursor += 6
    row_n = 2 * 3 * SCORER_H
    row = np.frombuffer(raw[cursor : cursor + row_n], dtype=np.int8).reshape(2, 3, SCORER_H).copy()
    cursor += row_n
    col_n = 2 * 3 * SCORER_W
    col = np.frombuffer(raw[cursor : cursor + col_n], dtype=np.int8).reshape(2, 3, SCORER_W).copy()
    return FunctionalParametersV1(base_rgb_u8=base, row_basis_i8=row, col_basis_i8=col)


def _canonical_latents(rows: Sequence[TemporalLatentV1]) -> tuple[TemporalLatentV1, ...]:
    out = tuple(rows)
    if out != tuple(sorted(out, key=lambda row: row.pair_index)):
        raise ScoreQuotientContractError("temporal latents must be in canonical pair order")
    if len({row.pair_index for row in out}) != len(out):
        raise ScoreQuotientContractError("temporal latent pair addresses must be unique")
    return out


def _encode_latents(rows: Sequence[TemporalLatentV1]) -> bytes:
    values = _canonical_latents(rows)
    body = bytearray(_LATENT_HEADER.pack(b"SQFL1", len(values)))
    for row in values:
        body.extend(_LATENT_ROW.pack(row.pair_index, *row.coefficients_q8, *row.xi_q12))
    return bytes(body)


def _decode_latents(raw: bytes) -> tuple[TemporalLatentV1, ...]:
    if len(raw) < _LATENT_HEADER.size:
        raise ScoreQuotientContractError("latent section is truncated")
    magic, count = _LATENT_HEADER.unpack_from(raw)
    if magic != b"SQFL1" or len(raw) != _LATENT_HEADER.size + count * _LATENT_ROW.size:
        raise ScoreQuotientContractError("latent section has invalid magic or exact length")
    rows = []
    cursor = _LATENT_HEADER.size
    for _ in range(count):
        unpacked = _LATENT_ROW.unpack_from(raw, cursor)
        cursor += _LATENT_ROW.size
        rows.append(TemporalLatentV1(unpacked[0], tuple(unpacked[1:7]), tuple(unpacked[7:13])))
    return _canonical_latents(rows)


def _canonical_placements(
    rows: Sequence[ExternallyPricedDemandPlacementV1],
) -> tuple[ExternallyPricedDemandPlacementV1, ...]:
    out = tuple(rows)
    canonical = tuple(sorted(out, key=lambda row: (row.pair_index, row.bucket_index, row.slot_index)))
    if out != canonical:
        raise ScoreQuotientContractError(
            "demand placements must be ordered by canonical (pair,bucket,slot) address"
        )
    if out and {row.slot_index for row in out} != set(range(DEMAND_ROW_COUNT)):
        raise ScoreQuotientContractError("nonempty demand placements must contain slots 0..24 exactly")
    if len({(row.pair_index, row.bucket_index) for row in out}) != len(out):
        raise ScoreQuotientContractError("demand placement pair/bucket addresses must be unique")
    return out


def _encode_placements(rows: Sequence[ExternallyPricedDemandPlacementV1]) -> bytes:
    values = _canonical_placements(rows)
    body = bytearray(_PLACEMENT_HEADER.pack(b"SQFD1", len(values)))
    for row in values:
        coder = row.coder_id.encode("ascii")
        body.extend(
            _PLACEMENT_ROW.pack(
                row.slot_index,
                row.pair_index,
                row.bucket_index,
                len(coder),
                bytes.fromhex(row.decoded_sha256),
                len(row.coded_record),
            )
        )
        body.extend(coder)
        body.extend(row.coded_record)
    return bytes(body)


def _decode_placements(raw: bytes) -> tuple[ExternallyPricedDemandPlacementV1, ...]:
    if len(raw) < _PLACEMENT_HEADER.size:
        raise ScoreQuotientContractError("placement section is truncated")
    magic, count = _PLACEMENT_HEADER.unpack_from(raw)
    if magic != b"SQFD1" or count not in (0, DEMAND_ROW_COUNT):
        raise ScoreQuotientContractError("placement section magic/count differs from schema")
    rows = []
    cursor = _PLACEMENT_HEADER.size
    for _ in range(count):
        if cursor + _PLACEMENT_ROW.size > len(raw):
            raise ScoreQuotientContractError("placement row is truncated")
        slot, pair, bucket, coder_len, decoded_sha, coded_len = _PLACEMENT_ROW.unpack_from(raw, cursor)
        cursor += _PLACEMENT_ROW.size
        end = cursor + coder_len + coded_len
        if end > len(raw):
            raise ScoreQuotientContractError("placement coder or payload is truncated")
        try:
            coder_id = raw[cursor : cursor + coder_len].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ScoreQuotientContractError("placement coder_id must be ASCII") from exc
        cursor += coder_len
        coded = raw[cursor : cursor + coded_len]
        cursor += coded_len
        rows.append(
            ExternallyPricedDemandPlacementV1(
                pair_index=pair,
                bucket_index=bucket,
                slot_index=slot,
                coder_id=coder_id,
                decoded_sha256=decoded_sha.hex(),
                coded_record=coded,
            )
        )
    if cursor != len(raw):
        raise ScoreQuotientContractError("placement section has trailing bytes")
    return _canonical_placements(rows)


def _canonical_exceptions(rows: Sequence[PixelExceptionV1]) -> tuple[PixelExceptionV1, ...]:
    out = tuple(rows)

    def key(row: PixelExceptionV1) -> tuple[int, int, int, int, int]:
        return (row.pair_index, row.frame_index, row.y, row.x, row.channel)

    if out != tuple(sorted(out, key=key)):
        raise ScoreQuotientContractError("exceptions must be in canonical scorer-plane address order")
    if len({key(row) for row in out}) != len(out):
        raise ScoreQuotientContractError("exception addresses must be unique")
    return out


def _encode_exceptions(rows: Sequence[PixelExceptionV1]) -> bytes:
    values = _canonical_exceptions(rows)
    body = bytearray(_EXCEPTION_HEADER.pack(b"SQFE1", len(values)))
    for row in values:
        body.extend(
            _EXCEPTION_ROW.pack(
                row.pair_index,
                row.frame_index,
                row.channel,
                row.y,
                row.x,
                row.value_u8,
                0,
            )
        )
    return bytes(body)


def _decode_exceptions(raw: bytes) -> tuple[PixelExceptionV1, ...]:
    if len(raw) < _EXCEPTION_HEADER.size:
        raise ScoreQuotientContractError("exception section is truncated")
    magic, count = _EXCEPTION_HEADER.unpack_from(raw)
    if magic != b"SQFE1" or len(raw) != _EXCEPTION_HEADER.size + count * _EXCEPTION_ROW.size:
        raise ScoreQuotientContractError("exception section has invalid magic or exact length")
    rows = []
    cursor = _EXCEPTION_HEADER.size
    for _ in range(count):
        pair, frame, channel, y, x, value, scope_wire = _EXCEPTION_ROW.unpack_from(raw, cursor)
        cursor += _EXCEPTION_ROW.size
        if scope_wire != 0:
            raise ScoreQuotientContractError("exception scope wire value is unknown")
        rows.append(PixelExceptionV1(pair, frame, y, x, channel, value))
    return _canonical_exceptions(rows)


def _section_tag(kind: SectionKind, counted_bytes: int) -> TypedStreamTag:
    stream_type, layer, citation = _TYPING[kind]
    return TypedStreamTag(
        type=stream_type,
        layer_home=layer,
        evaluate_py_recursion_level_cited=citation,
        counted_bytes=counted_bytes,
        free_receiver_code=True,
    )


def compile_score_quotient_packet(
    *,
    named_base: str,
    named_base_bytes: bytes,
    parameters: FunctionalParametersV1 | None = None,
    temporal_latents: Sequence[TemporalLatentV1] = (),
    demand_placements: Sequence[ExternallyPricedDemandPlacementV1] = (),
    exceptions: Sequence[PixelExceptionV1] = (),
) -> CompiledFunctionalPacketV1:
    """Compile a canonical packet; inactive streams return base bytes identically."""

    if not isinstance(named_base, str) or not _NAME_RE.fullmatch(named_base):
        raise ScoreQuotientContractError("named_base is outside the canonical vocabulary")
    if not isinstance(named_base_bytes, bytes) or not named_base_bytes:
        raise ScoreQuotientContractError("named_base_bytes must be nonempty bytes")
    latents = _canonical_latents(temporal_latents)
    placements = _canonical_placements(demand_placements)
    exception_rows = _canonical_exceptions(exceptions)
    base_sha = hashlib.sha256(named_base_bytes).hexdigest()
    if parameters is None and not latents and not placements and not exception_rows:
        receipt = FunctionalPacketReceipt(
            schema=SCHEMA,
            named_base=named_base,
            base_sha256=base_sha,
            base_counted_bytes=len(named_base_bytes),
            packet_sha256=base_sha,
            packet_bytes=len(named_base_bytes),
            total_counted_bytes=len(named_base_bytes),
            inactive_base_identity=True,
            sections=(),
        )
        return CompiledFunctionalPacketV1(named_base_bytes, receipt)

    raw_sections: list[tuple[SectionKind, bytes]] = []
    if parameters is not None:
        raw_sections.append((SectionKind.PARAMETERS, _encode_parameters(parameters)))
    if latents:
        raw_sections.append((SectionKind.TEMPORAL_LATENTS, _encode_latents(latents)))
    if placements:
        raw_sections.append((SectionKind.DEMAND_PLACEMENTS, _encode_placements(placements)))
    if exception_rows:
        raw_sections.append((SectionKind.EXCEPTIONS, _encode_exceptions(exception_rows)))

    base_name = named_base.encode("ascii")
    body = bytearray(_BASE_HEADER.pack(len(base_name), len(named_base_bytes), len(raw_sections)))
    body.extend(base_name)
    body.extend(bytes.fromhex(base_sha))
    section_receipts = []
    prefix_bytes = _OUTER_HEADER.size + _BASE_HEADER.size + len(base_name) + 32
    for section_index, (kind, raw) in enumerate(raw_sections):
        if kind is SectionKind.DEMAND_PLACEMENTS:
            coder, coded = Coder.PASSTHROUGH, raw
        else:
            coder, coded = _real_code(raw)
        body.extend(
            _SECTION_HEADER.pack(
                _SECTION_TO_WIRE[kind],
                _CODER_TO_WIRE[coder],
                len(raw),
                len(coded),
                zlib.crc32(raw) & 0xFFFFFFFF,
            )
        )
        body.extend(coded)
        allocated_packet_bytes = _SECTION_HEADER.size + len(coded)
        if section_index == 0:
            allocated_packet_bytes += prefix_bytes
        section_receipts.append(
            SectionReceipt(
                kind=kind,
                coder=coder,
                raw_bytes=len(raw),
                coded_bytes=len(coded),
                packet_bytes=allocated_packet_bytes,
                raw_sha256=hashlib.sha256(raw).hexdigest(),
                typed_tag=_section_tag(kind, allocated_packet_bytes),
            )
        )
    payload = _OUTER_HEADER.pack(
        PACKET_MAGIC,
        PACKET_VERSION,
        len(body),
        zlib.crc32(body) & 0xFFFFFFFF,
    ) + bytes(body)
    receipt = FunctionalPacketReceipt(
        schema=SCHEMA,
        named_base=named_base,
        base_sha256=base_sha,
        base_counted_bytes=len(named_base_bytes),
        packet_sha256=hashlib.sha256(payload).hexdigest(),
        packet_bytes=len(payload),
        total_counted_bytes=len(named_base_bytes) + len(payload),
        inactive_base_identity=False,
        sections=tuple(section_receipts),
    )
    return CompiledFunctionalPacketV1(payload, receipt)


def parse_score_quotient_packet(
    packet: bytes,
    *,
    named_bases: Mapping[str, bytes],
) -> ParsedFunctionalPacketV1:
    """Strictly parse, CRC-check, base-check, and canonical-parseback a packet."""

    if not isinstance(packet, bytes) or len(packet) < _OUTER_HEADER.size:
        raise ScoreQuotientContractError("packet is not a complete byte string")
    magic, version, body_len, body_crc = _OUTER_HEADER.unpack_from(packet)
    if magic != PACKET_MAGIC or version != PACKET_VERSION:
        raise ScoreQuotientContractError("packet magic/version differs from schema")
    body = packet[_OUTER_HEADER.size :]
    if len(body) != body_len or zlib.crc32(body) & 0xFFFFFFFF != body_crc:
        raise ScoreQuotientContractError("packet body length or CRC differs")
    if len(body) < _BASE_HEADER.size + 32:
        raise ScoreQuotientContractError("packet base header is truncated")
    name_len, base_counted_bytes, section_count = _BASE_HEADER.unpack_from(body)
    cursor = _BASE_HEADER.size
    if cursor + name_len + 32 > len(body):
        raise ScoreQuotientContractError("packet base reference is truncated")
    try:
        named_base = body[cursor : cursor + name_len].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ScoreQuotientContractError("packet named base must be ASCII") from exc
    cursor += name_len
    base_sha = body[cursor : cursor + 32].hex()
    cursor += 32
    if named_base not in named_bases:
        raise ScoreQuotientContractError(f"named base {named_base!r} was not supplied")
    base = named_bases[named_base]
    if len(base) != base_counted_bytes or hashlib.sha256(base).hexdigest() != base_sha:
        raise ScoreQuotientContractError("named base bytes do not match counted length/SHA-256")

    decoded: dict[SectionKind, bytes] = {}
    last_wire = 0
    for _ in range(section_count):
        if cursor + _SECTION_HEADER.size > len(body):
            raise ScoreQuotientContractError("packet section header is truncated")
        kind_wire, coder_wire, raw_len, coded_len, raw_crc = _SECTION_HEADER.unpack_from(body, cursor)
        cursor += _SECTION_HEADER.size
        if kind_wire not in _WIRE_TO_SECTION or coder_wire not in _WIRE_TO_CODER:
            raise ScoreQuotientContractError("packet section kind/coder is unknown")
        if kind_wire <= last_wire:
            raise ScoreQuotientContractError("packet sections must be sorted and unique")
        last_wire = kind_wire
        end = cursor + coded_len
        if end > len(body):
            raise ScoreQuotientContractError("packet section payload is truncated")
        coded = body[cursor:end]
        cursor = end
        kind = _WIRE_TO_SECTION[kind_wire]
        coder = _WIRE_TO_CODER[coder_wire]
        if kind is SectionKind.DEMAND_PLACEMENTS and coder is not Coder.PASSTHROUGH:
            raise ScoreQuotientContractError("DM1 placement records must remain pass-through priced")
        try:
            raw = _decode(coder, coded)
        except (lzma.LZMAError, zlib.error) as exc:
            raise ScoreQuotientContractError("real-coder decode failed") from exc
        if len(raw) != raw_len or zlib.crc32(raw) & 0xFFFFFFFF != raw_crc:
            raise ScoreQuotientContractError("decoded section length or CRC differs")
        decoded[kind] = raw
    if cursor != len(body):
        raise ScoreQuotientContractError("packet has trailing bytes")

    parsed = ParsedFunctionalPacketV1(
        named_base=named_base,
        base_sha256=base_sha,
        base_counted_bytes=base_counted_bytes,
        parameters=(
            _decode_parameters(decoded[SectionKind.PARAMETERS])
            if SectionKind.PARAMETERS in decoded
            else None
        ),
        temporal_latents=(
            _decode_latents(decoded[SectionKind.TEMPORAL_LATENTS])
            if SectionKind.TEMPORAL_LATENTS in decoded
            else ()
        ),
        demand_placements=(
            _decode_placements(decoded[SectionKind.DEMAND_PLACEMENTS])
            if SectionKind.DEMAND_PLACEMENTS in decoded
            else ()
        ),
        exceptions=(
            _decode_exceptions(decoded[SectionKind.EXCEPTIONS])
            if SectionKind.EXCEPTIONS in decoded
            else ()
        ),
    )
    rebuilt = compile_score_quotient_packet(
        named_base=parsed.named_base,
        named_base_bytes=base,
        parameters=parsed.parameters,
        temporal_latents=parsed.temporal_latents,
        demand_placements=parsed.demand_placements,
        exceptions=parsed.exceptions,
    )
    if rebuilt.payload != packet:
        raise ScoreQuotientContractError("packet is decodable but not canonical parse-back exact")
    return parsed


def render_scorer_planes(
    parameters: FunctionalParametersV1,
    latent: TemporalLatentV1,
) -> np.ndarray:
    """Deterministically expand one latent to uint8 ``(2,384,512,3)``."""

    out = np.empty((2, SCORER_H, SCORER_W, 3), dtype=np.uint8)
    for frame in range(2):
        for channel in range(3):
            coefficient = latent.coefficients_q8[frame * 3 + channel]
            product = (
                parameters.row_basis_i8[frame, channel].astype(np.int64)[:, None]
                * parameters.col_basis_i8[frame, channel].astype(np.int64)[None, :]
                * int(coefficient)
            )
            delta = np.rint(product.astype(np.float64) / float(1 << 20)).astype(np.int64)
            plane = int(parameters.base_rgb_u8[frame, channel]) + delta
            out[frame, :, :, channel] = np.clip(plane, 0, 255).astype(np.uint8)
    return out


PlacementApplier = Callable[
    [np.ndarray, ExternallyPricedDemandPlacementV1], DecodedDemandPlacementV1
]
PlaneRealizer = Callable[[int, int, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class ReceivedPairV1:
    pair_index: int
    scorer_planes_u8: np.ndarray
    pose_stats: tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class ReceiverProofV1:
    schema: str
    pair_count: int
    hard_tail_prefix: tuple[int, ...]
    exact_parseback: bool
    exact_through_r: bool
    cpu_threads: int
    pairs: tuple[ReceivedPairV1, ...]
    axis: str = AXIS
    score_claim: bool = SCORE_CLAIM


def _pin_torch_cpu_threads() -> int:
    """Pin the frozen-scorer Torch CPU pool to the delegated value."""

    try:
        import torch
    except ImportError as exc:
        raise ScoreQuotientContractError(
            "receiver proof requires Torch so CPU threads can be pinned to 4"
        ) from exc
    torch.set_num_threads(CPU_THREADS)
    actual = int(torch.get_num_threads())
    if actual != CPU_THREADS:
        raise ScoreQuotientContractError(
            f"Torch CPU thread pin failed: expected {CPU_THREADS}, got {actual}"
        )
    return actual


def receive_score_quotient_packet(
    packet: bytes,
    *,
    named_bases: Mapping[str, bytes],
    pair_indices: Sequence[int],
    hard_tail_order: Sequence[int],
    realize_plane_to_camera: PlaneRealizer,
    placement_applier: PlacementApplier | None = None,
) -> ReceiverProofV1:
    """Receive a bounded hard-tail-first subset through uint8 and the real R oracle."""

    pairs = tuple(pair_indices)
    hard_tail = tuple(hard_tail_order)
    if len(pairs) < 24:
        raise ScoreQuotientContractError("receiver proof requires a bounded subset with n>=24")
    if len(set(pairs)) != len(pairs) or any(
        isinstance(pair, bool) or not isinstance(pair, int) or not 0 <= pair < PAIR_COUNT
        for pair in pairs
    ):
        raise ScoreQuotientContractError("receiver pair indices must be unique canonical pair ids")
    prefix_n = min(24, len(hard_tail))
    if prefix_n < 24 or pairs[:prefix_n] != hard_tail[:prefix_n]:
        raise ScoreQuotientContractError("receiver subset must be hard-tail-first for its first 24 pairs")
    if not callable(realize_plane_to_camera):
        raise ScoreQuotientContractError("realize_plane_to_camera must be callable")
    torch_cpu_threads = _pin_torch_cpu_threads()

    parsed = parse_score_quotient_packet(packet, named_bases=named_bases)
    if parsed.parameters is None:
        raise ScoreQuotientContractError("active functional packet is missing PARAMETERS")
    latent_by_pair = {row.pair_index: row for row in parsed.temporal_latents}
    missing = [pair for pair in pairs if pair not in latent_by_pair]
    if missing:
        raise ScoreQuotientContractError(f"TEMPORAL_LATENTS missing requested pairs {missing[:8]}")
    placement_by_pair: dict[int, list[ExternallyPricedDemandPlacementV1]] = {}
    for row in parsed.demand_placements:
        placement_by_pair.setdefault(row.pair_index, []).append(row)
    if parsed.demand_placements and placement_applier is None:
        raise ScoreQuotientContractError(
            "DEMAND_PLACEMENTS present but the external DM1 decoder/applier was not supplied"
        )
    exceptions_by_pair: dict[int, list[PixelExceptionV1]] = {}
    for row in parsed.exceptions:
        exceptions_by_pair.setdefault(row.pair_index, []).append(row)

    received = []
    through_r_cache: dict[str, np.ndarray] = {}
    for pair in pairs:
        latent = latent_by_pair[pair]
        expected = render_scorer_planes(parsed.parameters, latent)
        for placement in placement_by_pair.get(pair, ()):
            assert placement_applier is not None
            applied = placement_applier(expected.copy(), placement)
            if not isinstance(applied, DecodedDemandPlacementV1):
                raise ScoreQuotientContractError(
                    "DM1 placement applier must return DecodedDemandPlacementV1"
                )
            if (
                hashlib.sha256(applied.decoded_payload).hexdigest()
                != placement.decoded_sha256
            ):
                raise ScoreQuotientContractError(
                    "DM1 decoded placement bytes differ from decoded_sha256"
                )
            applied_array = np.asarray(applied.scorer_planes_u8)
            if applied_array.shape != expected.shape or applied_array.dtype != np.uint8:
                raise ScoreQuotientContractError(
                    "DM1 placement applier must return uint8 (2,384,512,3)"
                )
            expected = np.ascontiguousarray(applied_array)
        for row in exceptions_by_pair.get(pair, ()):
            expected[row.frame_index, row.y, row.x, row.channel] = row.value_u8

        realized = np.empty_like(expected)
        for frame in range(2):
            camera = np.asarray(realize_plane_to_camera(pair, frame, expected[frame]))
            if camera.shape != (CAMERA_H, CAMERA_W, 3) or camera.dtype != np.uint8:
                raise ScoreQuotientContractError(
                    f"realizer must return uint8 {(CAMERA_H, CAMERA_W, 3)}"
                )
            camera_key = hashlib.sha256(camera.tobytes(order="C")).hexdigest()
            if camera_key not in through_r_cache:
                through_r = contest_faithful_R_numpy(camera[None], ste_round=False)[0]
                through_r_cache[camera_key] = np.clip(
                    np.rint(through_r), 0, 255
                ).astype(np.uint8)
            realized[frame] = through_r_cache[camera_key]
        if not np.array_equal(realized, expected):
            raise ScoreQuotientContractError(
                f"receiver R/uint8 parse-back differs for pair {pair}"
            )
        pose = tuple(float(value) / 4096.0 for value in latent.xi_q12)
        received.append(ReceivedPairV1(pair, realized, pose))
    return ReceiverProofV1(
        schema="ddm_score_quotient_receiver_proof.v1",
        pair_count=len(received),
        hard_tail_prefix=hard_tail[:24],
        exact_parseback=True,
        exact_through_r=True,
        cpu_threads=torch_cpu_threads,
        pairs=tuple(received),
    )


@dataclass(frozen=True, slots=True)
class ScoreQuotientObjectiveV1:
    d_seg: float
    d_pose: float
    archive_bytes: int
    score: float
    exact_real_coder_bytes: bool
    axis: str = AXIS
    score_claim: bool = SCORE_CLAIM
    frontier_pointer: str = FRONTIER_POINTER


def score_quotient_functional_objective(
    d_seg: float,
    d_pose: float,
    receipt: FunctionalPacketReceipt,
) -> ScoreQuotientObjectiveV1:
    """Canonical S functional using the compiled receipt's real coder bytes."""

    seg = float(d_seg)
    pose = float(d_pose)
    if (
        not math.isfinite(seg)
        or not 0 <= seg <= 1
        or not math.isfinite(pose)
        or pose < 0
    ):
        raise ScoreQuotientContractError(
            "d_seg must be finite in [0,1] and d_pose finite nonnegative"
        )
    if not isinstance(receipt, FunctionalPacketReceipt):
        raise ScoreQuotientContractError("objective requires a FunctionalPacketReceipt")
    archive_bytes = receipt.total_counted_bytes
    return ScoreQuotientObjectiveV1(
        d_seg=seg,
        d_pose=pose,
        archive_bytes=archive_bytes,
        score=compute_contest_score(seg, pose, archive_bytes),
        exact_real_coder_bytes=True,
    )


@dataclass(frozen=True, slots=True)
class CapacityDimensionV1:
    name: str
    exact_value: int | None
    approximate_hint: int | None
    status: str
    derivation: str


@dataclass(frozen=True, slots=True)
class CapacityDerivationV1:
    seg_head: CapacityDimensionV1
    lane_orbit: CapacityDimensionV1
    pose_xi: CapacityDimensionV1
    demand_rows: CapacityDimensionV1
    exact_total: int | None
    status: str


@dataclass(frozen=True, slots=True)
class LaneOrbitRankCertificateV1:
    """Custodied realized-through-R rank evidence; a bare integer is inadmissible."""

    exact_rank: int
    source_artifact: str
    source_sha256: str
    measurement_axis: str
    realized_through_r: bool

    def __post_init__(self) -> None:
        _require_int(self.exact_rank, "lane orbit exact_rank", 0, 64)
        if not isinstance(self.source_artifact, str) or not self.source_artifact.strip():
            raise ScoreQuotientContractError(
                "lane orbit certificate requires a source artifact"
            )
        _require_sha256(self.source_sha256, "lane orbit source_sha256")
        if not isinstance(self.measurement_axis, str) or not self.measurement_axis.strip():
            raise ScoreQuotientContractError(
                "lane orbit certificate requires a measurement axis"
            )
        if self.realized_through_r is not True:
            raise ScoreQuotientContractError(
                "lane orbit rank certificate must be realized through R"
            )


def derive_score_quotient_capacity(
    *, lane_orbit_rank_certificate: LaneOrbitRankCertificateV1 | None = None
) -> CapacityDerivationV1:
    """Derive exact dimensions; retain the ~8D orbit as NULL until certified."""

    if lane_orbit_rank_certificate is not None and not isinstance(
        lane_orbit_rank_certificate, LaneOrbitRankCertificateV1
    ):
        raise ScoreQuotientContractError(
            "lane_orbit_rank_certificate must be LaneOrbitRankCertificateV1 or null"
        )
    lane_rank = (
        lane_orbit_rank_certificate.exact_rank
        if lane_orbit_rank_certificate is not None
        else None
    )
    lane = CapacityDimensionV1(
        name="lane_homography_orbit",
        exact_value=lane_rank,
        approximate_hint=8,
        status=("DERIVED_FROM_RANK_CERTIFICATE" if lane_orbit_rank_certificate is not None else "NULL_DERIVATION_OWED"),
        derivation=(
            "rank of a supplied realized-through-R lane-orbit Jacobian certificate"
            if lane_orbit_rank_certificate is not None
            else "~8 is a prior research-sidecar hint, not an exact sealed capacity"
        ),
    )
    exact_total = (
        SEG_HEAD_RANK + lane.exact_value + POSE_STAT_DIM + DEMAND_ROW_COUNT
        if lane.exact_value is not None
        else None
    )
    return CapacityDerivationV1(
        seg_head=CapacityDimensionV1(
            "seg_argmax_head",
            SEG_HEAD_RANK,
            None,
            "DERIVED",
            "K-1 centered logit-difference rank for K=5; measured rank-4 head anchor",
        ),
        lane_orbit=lane,
        pose_xi=CapacityDimensionV1(
            "pose_xi",
            POSE_STAT_DIM,
            None,
            "DERIVED",
            "upstream PoseNet verdict consumes the first 6 output coordinates",
        ),
        demand_rows=CapacityDimensionV1(
            "externally_priced_demand_rows",
            DEMAND_ROW_COUNT,
            None,
            "DERIVED",
            "Directive 5/DM1 owns exactly 25 heterogeneous demand records",
        ),
        exact_total=exact_total,
        status=("COMPLETE" if exact_total is not None else "NULL_DERIVATION_OWED"),
    )


@dataclass(frozen=True, slots=True)
class V14FalsifierVerdictV1:
    verdict: str
    missing_stream: str | None
    baseline_d_seg: float
    baseline_archive_bytes: int
    candidate_d_seg: float | None
    candidate_archive_bytes: int | None
    receiver_closed: bool


def v14_baseline_falsifier(
    *,
    candidate_d_seg: float | None,
    candidate_archive_bytes: int | None,
    receiver_closed: bool,
    missing_stream: str | None = None,
) -> V14FalsifierVerdictV1:
    """Fail closed unless family (d) expresses v14 or better at no more bytes."""

    passed = (
        candidate_d_seg is not None
        and candidate_archive_bytes is not None
        and math.isfinite(float(candidate_d_seg))
        and 0 <= float(candidate_d_seg) <= 1
        and float(candidate_d_seg) <= V14_BASELINE_D_SEG
        and isinstance(candidate_archive_bytes, int)
        and not isinstance(candidate_archive_bytes, bool)
        and 0 <= candidate_archive_bytes <= V14_BASELINE_ARCHIVE_BYTES
        and receiver_closed is True
    )
    if passed:
        missing = None
        verdict = "EXPRESSIBLE_V14_OR_BETTER"
    else:
        missing = missing_stream or "FIT_RESULT_RECEIVER_CLOSED_V14_OR_BETTER"
        if not isinstance(missing, str) or not missing.strip():
            raise ScoreQuotientContractError("INCOMPLETE verdict requires a named missing stream")
        verdict = "INCOMPLETE"
    return V14FalsifierVerdictV1(
        verdict=verdict,
        missing_stream=missing,
        baseline_d_seg=V14_BASELINE_D_SEG,
        baseline_archive_bytes=V14_BASELINE_ARCHIVE_BYTES,
        candidate_d_seg=(None if candidate_d_seg is None else float(candidate_d_seg)),
        candidate_archive_bytes=candidate_archive_bytes,
        receiver_closed=receiver_closed,
    )


@dataclass(frozen=True, slots=True)
class DDMEventContinuationV1FitRequest:
    schema: str
    status: str
    packet_sha256: str
    packet_total_counted_bytes: int
    objective_callable: str
    receiver_callable: str
    required_pair_subset_min: int
    hard_tail_first: bool
    real_coder_in_loss: bool
    rate_stop_threshold_score_units_per_byte: float
    execution_allowed: bool
    score_claim: bool
    frontier_pointer: str
    support_gaps: tuple[str, ...]


def build_ddm_event_continuation_v1_fit_request(
    receipt: FunctionalPacketReceipt,
) -> DDMEventContinuationV1FitRequest:
    """Build the typed future-fit adapter, not a training or schedule engine."""

    if not isinstance(receipt, FunctionalPacketReceipt):
        raise ScoreQuotientContractError("fit request requires a FunctionalPacketReceipt")
    return DDMEventContinuationV1FitRequest(
        schema=FIT_SCHEMA,
        status="INTERFACE_ONLY_NOT_EXECUTABLE",
        packet_sha256=receipt.packet_sha256,
        packet_total_counted_bytes=receipt.total_counted_bytes,
        objective_callable=(
            "tac.optimization.ddm_score_quotient_functional_contract:"
            "score_quotient_functional_objective"
        ),
        receiver_callable=(
            "tac.optimization.ddm_score_quotient_functional_contract:"
            "receive_score_quotient_packet"
        ),
        required_pair_subset_min=24,
        hard_tail_first=True,
        real_coder_in_loss=True,
        rate_stop_threshold_score_units_per_byte=REVERSE_WATERFILL_RATE_THRESHOLD,
        execution_allowed=False,
        score_claim=False,
        frontier_pointer=FRONTIER_POINTER,
        support_gaps=(
            "DDMEventContinuationV1 executable schedule/engine is not present",
            "DM1 25-row decoded-value schema and real coder-price records are external and not yet supplied",
            "receiver-closed frozen PoseNet/SegNet n600 fit is not measured",
            "exact lane-orbit rank certificate is not supplied",
        ),
    )


__all__ = [
    "AT_RISK_SCOPE",
    "AXIS",
    "DEMAND_ROW_COUNT",
    "FRONTIER_POINTER",
    "REVERSE_WATERFILL_RATE_THRESHOLD",
    "V14_BASELINE_ARCHIVE_BYTES",
    "V14_BASELINE_D_SEG",
    "CapacityDerivationV1",
    "CompiledFunctionalPacketV1",
    "DDMEventContinuationV1FitRequest",
    "DecodedDemandPlacementV1",
    "ExternallyPricedDemandPlacementV1",
    "FunctionalPacketReceipt",
    "FunctionalParametersV1",
    "LaneOrbitRankCertificateV1",
    "PixelExceptionV1",
    "ReceiverProofV1",
    "ScoreQuotientContractError",
    "ScoreQuotientObjectiveV1",
    "TemporalLatentV1",
    "build_ddm_event_continuation_v1_fit_request",
    "compile_score_quotient_packet",
    "derive_score_quotient_capacity",
    "parse_score_quotient_packet",
    "receive_score_quotient_packet",
    "render_scorer_planes",
    "score_quotient_functional_objective",
    "v14_baseline_falsifier",
]
