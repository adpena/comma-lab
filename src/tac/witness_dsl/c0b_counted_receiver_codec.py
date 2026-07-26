# SPDX-License-Identifier: MIT
"""Counted C0B receiver for a factorized V9 semantic program plus PBR1.

The archive contains every video-derived byte consumed by this structural
receiver: the exact V9 predictor program, its predictor-bound partition
residual, and a typed RGB preimage program.  The generic decoder is free code;
it does not consult a source video, scorer, target table, or dense scorer-plane
table at decode time.

This module is deliberately research-only.  Its receipts prove byte custody,
semantic recovery, RGB preimage consumption, and exact factor-2 parse-back.
They are not contest-score evidence and do not make the archive promotable.
Because this ABI paints one recovered semantic partition through two palettes,
it is an explicit causality ablation rather than the distinct-Y0/Y1 capstone.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import shutil
import struct
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Uint8LatticeError,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)
from tac.witness_dsl.factorized_v9_predictor import (
    PREDICTOR_CONTRACT_ID,
    FactorizedV9PredictorError,
    FactorizedV9SemanticReceiver,
    receive_factorized_v9_predictor,
)
from tac.witness_dsl.predictor_bound_residual import (
    PredictorBoundResidualError,
    apply_predictor_bound_partition_residual,
    decode_predictor_bound_partition_residual,
)

PACKET_MAGIC: Final = b"TACC0B\x00\x00"
PACKET_VERSION: Final = 1
PACKET_SCHEMA: Final = "tac.c0b_counted_receiver_packet.v1"
ARCHIVE_SCHEMA: Final = "tac.c0b_counted_receiver_archive.v1"
PREIMAGE_SCHEMA: Final = "tac.c0b_rgb_preimage.v1"
RECEIVER_CONTRACT_ID: Final = "factor2-disjoint-half-pixel-uint8.v1"
CODEC_ROLE: Final = "abi_causality_ablation"
MEMBER_NAME: Final = "0.bin"
SECTION_ORDER: Final = ("factorized_v9_program", "pbr1", "rgb_preimage")
SECTION_CODECS: Final = (
    "factorized-v9-counted-program.v1",
    "predictor-bound-partition-residual.v1",
    "pair-plane-class-palette-sparse-rgb.v1",
)

_PACKET_PREFIX: Final = struct.Struct(">8sHI")
_SECTION_LENGTH: Final = struct.Struct(">Q")
_PREIMAGE_PREFIX: Final = struct.Struct("<4sHII")
_OVERRIDE_RECORD: Final = struct.Struct("<IBHHBBB")
_PREIMAGE_MAGIC: Final = b"C0BP"
_PREIMAGE_VERSION: Final = 1
_MAX_HEADER_BYTES: Final = 1 << 20
_MAX_SECTION_BYTES: Final = 1 << 30
_MAX_PAIRS: Final = 10_000
_MAX_DIMENSION: Final = 4096

_HEADER_FIELDS: Final = frozenset(
    {
        "schema",
        "version",
        "pair_count",
        "pair_ids",
        "pair_population_sha256",
        "geometry",
        "sections",
        "section_count",
        "section_payload_bytes",
        "section_framing_bytes",
        "packet_bytes",
        "receiver_contract_id",
        "research_only",
        "launch_ready",
        "score_claim",
        "promotion_eligible",
        "codec_role",
        "capstone_eligible",
        "shared_semantic_partition_across_planes",
        "separate_dense_target_table_section_bytes",
        "dense_y_table_bytes",
        "decode_scorer_dependency",
        "pbr1_is_target_derived",
        "pbr1_target_derived_section_bytes",
        "pbr1_nested_semantic_residual_bytes",
        "pbr1_event_count",
        "pbr1_event_density_numerator",
        "pbr1_event_density_denominator",
        "target_derived_residual_promotion_admitted",
        "repository_decoder_dependency",
        "standalone_inflate_source_custody",
        "standalone_inflate_owed",
        "authority_receipt_owed",
        "evaluator_obligation_ir_bound",
        "coupled_witness_state_bound",
        "hard_oracle_admission_bound",
        "independent_frame0_preimage",
    }
)
_SECTION_FIELDS: Final = frozenset({"section_id", "codec_id", "bytes", "sha256"})


class CountedReceiverCodecError(ValueError):
    """Fail-closed archive, binding, preimage, or realization error."""


@dataclass(frozen=True, order=True, slots=True)
class SparseRGBOverride:
    """One absolute RGB scorer-cell replacement in local pair coordinates."""

    pair_index: int
    plane: int
    row: int
    col: int
    red: int
    green: int
    blue: int


@dataclass(frozen=True, slots=True)
class ParsedCountedPacket:
    packet: bytes
    header: Mapping[str, Any]
    sections: tuple[bytes, bytes, bytes]

    def section(self, section_id: str) -> bytes:
        try:
            return self.sections[SECTION_ORDER.index(section_id)]
        except ValueError as exc:
            raise CountedReceiverCodecError(f"unknown counted section {section_id!r}") from exc


@dataclass(frozen=True, slots=True)
class CountedDecodeReceipt:
    archive_bytes: int
    archive_sha256: str
    packet_bytes: int
    packet_sha256: str
    archive_container_bytes: int
    packet_framing_and_header_bytes: int
    section_bytes: Mapping[str, int]
    section_sha256: Mapping[str, str]
    pbr1_counted_bytes: int
    pbr1_target_derived_section_bytes: int
    pbr1_nested_semantic_residual_bytes: int
    pbr1_event_count: int
    pbr1_event_density_numerator: int
    pbr1_event_density_denominator: int
    separate_dense_target_table_section_bytes: int
    target_derived_residual_promotion_admitted: bool
    exact_target_semantic_reconstruction: bool
    candidate_payload_allowed: bool
    candidate_archive_blocker: str
    pair_ids: tuple[int, ...]
    pair_population_sha256: str
    predictor_semantic_sha256: str
    target_semantic_sha256: str
    scorer_rgb_sha256: str
    preimage_palette_bytes: int
    preimage_override_records: int
    raw_bytes: int
    raw_sha256: str
    factor2_scorer_values_verified: int
    factor2_numerator_values_verified: int
    factor2_certified_exact: bool
    exact_archive_parse_back: bool
    codec_role: str
    capstone_eligible: bool
    shared_semantic_partition_across_planes: bool
    repository_decoder_dependency: bool
    standalone_inflate_source_custody: bool
    standalone_inflate_owed: bool
    authority_receipt_owed: bool
    evaluator_obligation_ir_bound: bool
    coupled_witness_state_bound: bool
    hard_oracle_admission_bound: bool
    independent_frame0_preimage: bool
    research_only: bool
    score_claim: bool
    promotion_eligible: bool


@dataclass(frozen=True, slots=True)
class CountedDecodeResult:
    raw: bytes
    receipt: CountedDecodeReceipt


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _sha256_file(handle: Any, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(chunk_bytes):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file_prefix(handle: Any, size: int, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = handle.read(min(chunk_bytes, remaining))
        if not chunk:
            raise CountedReceiverCodecError("partial raw is shorter than its durable checkpoint")
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
        raise CountedReceiverCodecError("value is not finite canonical ASCII JSON") from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_resume_state(path: Path, state: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(_canonical_json(state))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


@contextmanager
def _exclusive_output_lock(output: Path) -> Iterator[None]:
    lock_path = output.with_name(f"{output.name}.lock")
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CountedReceiverCodecError("another counted receiver writer owns this output") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _exact_int(value: Any, label: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise CountedReceiverCodecError(f"{label} is not an admitted exact integer")
    return value


def _pair_ids_bytes(pair_ids: Sequence[int]) -> bytes:
    return b"".join(struct.pack("<I", _exact_int(value, "pair_id", maximum=0xFFFFFFFF)) for value in pair_ids)


def pair_population_sha256(pair_ids: Sequence[int]) -> str:
    """Hash the ordered source-pair population, not merely its cardinality."""

    values = tuple(pair_ids)
    if not values or len(set(values)) != len(values):
        raise CountedReceiverCodecError("pair_ids must be a non-empty unique ordered population")
    return _sha256(_pair_ids_bytes(values))


def _canonical_source_pair_ids(receiver: FactorizedV9SemanticReceiver) -> tuple[int, ...]:
    exposed = getattr(receiver, "source_pair_ids", None)
    if exposed is not None:
        values = tuple(int(value) for value in exposed)
    else:
        start = int(receiver.receiver.predictor.source_pair_start)
        values = tuple(range(start, start + receiver.pair_count))
    if len(values) != receiver.pair_count:
        raise CountedReceiverCodecError("predictor source-pair population cardinality drift")
    return values


def _validate_palette(palette: np.ndarray, *, pair_count: int) -> np.ndarray:
    value = np.asarray(palette)
    expected = (pair_count, 2, 5, 3)
    if value.dtype != np.uint8 or value.shape != expected:
        raise CountedReceiverCodecError(f"palette must be exact uint8 with shape {expected}")
    return np.ascontiguousarray(value)


def _validate_overrides(
    overrides: Sequence[SparseRGBOverride],
    *,
    pair_count: int,
    height: int,
    width: int,
) -> tuple[SparseRGBOverride, ...]:
    rows = tuple(overrides)
    if any(not isinstance(row, SparseRGBOverride) for row in rows):
        raise CountedReceiverCodecError("overrides must contain typed SparseRGBOverride rows")
    for row in rows:
        _exact_int(row.pair_index, "override pair_index", maximum=pair_count - 1)
        _exact_int(row.plane, "override plane", maximum=1)
        _exact_int(row.row, "override row", maximum=height - 1)
        _exact_int(row.col, "override col", maximum=width - 1)
        _exact_int(row.red, "override red", maximum=255)
        _exact_int(row.green, "override green", maximum=255)
        _exact_int(row.blue, "override blue", maximum=255)
    if tuple(sorted(rows)) != rows:
        raise CountedReceiverCodecError("override rows must be canonically sorted")
    coordinates = {(row.pair_index, row.plane, row.row, row.col) for row in rows}
    if len(coordinates) != len(rows):
        raise CountedReceiverCodecError("override rows duplicate scorer-cell ownership")
    return rows


def encode_rgb_preimage(
    palette: np.ndarray,
    overrides: Sequence[SparseRGBOverride] = (),
) -> bytes:
    """Encode the compact pair/plane/class palette and sparse absolute RGB rows."""

    raw = np.asarray(palette)
    if raw.ndim != 4:
        raise CountedReceiverCodecError("palette must have pair x plane x class x RGB geometry")
    pair_count = int(raw.shape[0])
    value = _validate_palette(raw, pair_count=pair_count)
    rows = _validate_overrides(
        overrides,
        pair_count=pair_count,
        height=0x10000,
        width=0x10000,
    )
    body = bytearray(_PREIMAGE_PREFIX.pack(_PREIMAGE_MAGIC, _PREIMAGE_VERSION, pair_count, len(rows)))
    body.extend(value.tobytes(order="C"))
    for row in rows:
        body.extend(
            _OVERRIDE_RECORD.pack(
                row.pair_index,
                row.plane,
                row.row,
                row.col,
                row.red,
                row.green,
                row.blue,
            )
        )
    return bytes(body)


def decode_rgb_preimage(
    payload: bytes,
    *,
    pair_count: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, tuple[SparseRGBOverride, ...]]:
    """Strictly decode and re-encode the typed RGB preimage section."""

    if not isinstance(payload, bytes) or len(payload) < _PREIMAGE_PREFIX.size:
        raise CountedReceiverCodecError("RGB preimage is truncated or not bytes")
    magic, version, encoded_pairs, override_count = _PREIMAGE_PREFIX.unpack_from(payload)
    if magic != _PREIMAGE_MAGIC or version != _PREIMAGE_VERSION or encoded_pairs != pair_count:
        raise CountedReceiverCodecError("RGB preimage magic/version/pair count mismatch")
    palette_bytes = pair_count * 2 * 5 * 3
    expected = _PREIMAGE_PREFIX.size + palette_bytes + override_count * _OVERRIDE_RECORD.size
    if len(payload) != expected:
        raise CountedReceiverCodecError("RGB preimage length mismatch or trailing bytes")
    start = _PREIMAGE_PREFIX.size
    palette = np.frombuffer(payload[start : start + palette_bytes], dtype=np.uint8).reshape(pair_count, 2, 5, 3).copy()
    rows = tuple(
        SparseRGBOverride(*_OVERRIDE_RECORD.unpack_from(payload, offset))
        for offset in range(start + palette_bytes, len(payload), _OVERRIDE_RECORD.size)
    )
    rows = _validate_overrides(rows, pair_count=pair_count, height=height, width=width)
    if encode_rgb_preimage(palette, rows) != payload:
        raise CountedReceiverCodecError("RGB preimage is not canonical")
    return palette, rows


def _geometry(camera_height: int, camera_width: int) -> dict[str, int]:
    camera_h = _exact_int(camera_height, "camera_height", minimum=1, maximum=_MAX_DIMENSION)
    camera_w = _exact_int(camera_width, "camera_width", minimum=1, maximum=_MAX_DIMENSION)
    return {
        "camera_height": camera_h,
        "camera_width": camera_w,
        "scorer_height": 384,
        "scorer_width": 512,
        "channels": 3,
    }


def build_counted_receiver_packet(
    *,
    predictor_program: bytes,
    pbr1: bytes,
    pair_ids: Sequence[int],
    palette: np.ndarray,
    overrides: Sequence[SparseRGBOverride] = (),
    camera_height: int = 874,
    camera_width: int = 1164,
    repository_root: Path | None = None,
) -> bytes:
    """Build, fresh-receive, and cross-validate one counted C0B packet."""

    if not isinstance(predictor_program, bytes) or not predictor_program:
        raise CountedReceiverCodecError("predictor_program must be non-empty exact bytes")
    if not isinstance(pbr1, bytes) or not pbr1:
        raise CountedReceiverCodecError("pbr1 must be non-empty exact bytes")
    ids = tuple(pair_ids)
    pair_count = len(ids)
    _exact_int(pair_count, "pair_count", minimum=1, maximum=_MAX_PAIRS)
    population_sha = pair_population_sha256(ids)
    geometry = _geometry(camera_height, camera_width)
    try:
        DisjointResizeOperator.build(
            camera_h=geometry["camera_height"],
            camera_w=geometry["camera_width"],
            scorer_h=geometry["scorer_height"],
            scorer_w=geometry["scorer_width"],
        )
    except Uint8LatticeError as exc:
        raise CountedReceiverCodecError("geometry is not a certified disjoint factor-2 lattice") from exc
    preimage = encode_rgb_preimage(_validate_palette(palette, pair_count=pair_count), overrides)
    try:
        pbr1_envelope = decode_predictor_bound_partition_residual(pbr1)
    except PredictorBoundResidualError as exc:
        raise CountedReceiverCodecError("pbr1 is not a strict predictor-bound residual") from exc
    event_count = len(pbr1_envelope.seed.events)
    event_population = pbr1_envelope.seed.n_pairs * pbr1_envelope.seed.height * pbr1_envelope.seed.width
    sections = (predictor_program, pbr1, preimage)
    if any(len(section) > _MAX_SECTION_BYTES for section in sections):
        raise CountedReceiverCodecError("counted section exceeds its byte cap")
    rows = [
        {"section_id": name, "codec_id": codec, "bytes": len(section), "sha256": _sha256(section)}
        for name, codec, section in zip(SECTION_ORDER, SECTION_CODECS, sections, strict=True)
    ]
    payload_bytes = sum(map(len, sections))
    header: dict[str, Any] = {
        "schema": PACKET_SCHEMA,
        "version": PACKET_VERSION,
        "pair_count": pair_count,
        "pair_ids": list(ids),
        "pair_population_sha256": population_sha,
        "geometry": geometry,
        "sections": rows,
        "section_count": len(sections),
        "section_payload_bytes": payload_bytes,
        "section_framing_bytes": len(sections) * _SECTION_LENGTH.size,
        "packet_bytes": 0,
        "receiver_contract_id": RECEIVER_CONTRACT_ID,
        "research_only": True,
        "launch_ready": False,
        "score_claim": False,
        "promotion_eligible": False,
        "codec_role": CODEC_ROLE,
        "capstone_eligible": False,
        "shared_semantic_partition_across_planes": True,
        "separate_dense_target_table_section_bytes": 0,
        "dense_y_table_bytes": 0,
        "decode_scorer_dependency": False,
        "pbr1_is_target_derived": True,
        "pbr1_target_derived_section_bytes": len(pbr1),
        "pbr1_nested_semantic_residual_bytes": len(pbr1_envelope.residual_payload),
        "pbr1_event_count": event_count,
        "pbr1_event_density_numerator": event_count,
        "pbr1_event_density_denominator": event_population,
        "target_derived_residual_promotion_admitted": False,
        "repository_decoder_dependency": True,
        "standalone_inflate_source_custody": False,
        "standalone_inflate_owed": True,
        "authority_receipt_owed": True,
        "evaluator_obligation_ir_bound": False,
        "coupled_witness_state_bound": False,
        "hard_oracle_admission_bound": False,
        "independent_frame0_preimage": False,
    }
    for _ in range(8):
        header_bytes = _canonical_json(header)
        packet_bytes = _PACKET_PREFIX.size + len(header_bytes) + header["section_framing_bytes"] + payload_bytes
        if header["packet_bytes"] == packet_bytes:
            break
        header["packet_bytes"] = packet_bytes
    else:  # pragma: no cover
        raise CountedReceiverCodecError("packet byte-count fixed point did not converge")
    header_bytes = _canonical_json(header)
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise CountedReceiverCodecError("packet header exceeds its byte cap")
    packet = bytearray(_PACKET_PREFIX.pack(PACKET_MAGIC, PACKET_VERSION, len(header_bytes)))
    packet.extend(header_bytes)
    for section in sections:
        packet.extend(_SECTION_LENGTH.pack(len(section)))
        packet.extend(section)
    encoded = bytes(packet)
    parsed = parse_counted_receiver_packet(encoded)
    _receive_semantics(parsed, repository_root=repository_root)
    return encoded


def parse_counted_receiver_packet(packet: bytes) -> ParsedCountedPacket:
    """Strictly parse all counted sections with exact stream consumption."""

    if not isinstance(packet, bytes) or len(packet) < _PACKET_PREFIX.size:
        raise CountedReceiverCodecError("counted packet is truncated or not bytes")
    magic, version, header_size = _PACKET_PREFIX.unpack_from(packet)
    if magic != PACKET_MAGIC or version != PACKET_VERSION or not 0 < header_size <= _MAX_HEADER_BYTES:
        raise CountedReceiverCodecError("counted packet magic/version/header length mismatch")
    header_end = _PACKET_PREFIX.size + header_size
    if header_end > len(packet):
        raise CountedReceiverCodecError("counted packet header is truncated")
    raw_header = packet[_PACKET_PREFIX.size : header_end]
    try:
        header = json.loads(raw_header.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CountedReceiverCodecError("counted packet header is not valid ASCII JSON") from exc
    if not isinstance(header, dict) or _canonical_json(header) != raw_header or set(header) != _HEADER_FIELDS:
        raise CountedReceiverCodecError("counted packet header is noncanonical or has field drift")
    if header["schema"] != PACKET_SCHEMA or header["version"] != PACKET_VERSION:
        raise CountedReceiverCodecError("counted packet header schema/version mismatch")
    for field in ("launch_ready", "score_claim", "promotion_eligible", "capstone_eligible"):
        if header[field] is not False:
            raise CountedReceiverCodecError(f"counted packet cannot authorize {field}")
    if (
        header["research_only"] is not True
        or header["codec_role"] != CODEC_ROLE
        or header["shared_semantic_partition_across_planes"] is not True
        or header["separate_dense_target_table_section_bytes"] != 0
        or header["dense_y_table_bytes"] != 0
        or header["decode_scorer_dependency"] is not False
        or header["pbr1_is_target_derived"] is not True
        or header["target_derived_residual_promotion_admitted"] is not False
        or header["repository_decoder_dependency"] is not True
        or header["standalone_inflate_source_custody"] is not False
        or header["standalone_inflate_owed"] is not True
        or header["authority_receipt_owed"] is not True
        or header["evaluator_obligation_ir_bound"] is not False
        or header["coupled_witness_state_bound"] is not False
        or header["hard_oracle_admission_bound"] is not False
        or header["independent_frame0_preimage"] is not False
        or header["receiver_contract_id"] != RECEIVER_CONTRACT_ID
    ):
        raise CountedReceiverCodecError("counted packet research/custody declarations drift")
    pair_count = _exact_int(header["pair_count"], "pair_count", minimum=1, maximum=_MAX_PAIRS)
    ids = header["pair_ids"]
    if not isinstance(ids, list) or len(ids) != pair_count:
        raise CountedReceiverCodecError("pair population shape differs from pair_count")
    if pair_population_sha256(ids) != header["pair_population_sha256"]:
        raise CountedReceiverCodecError("pair population hash differs from its ordered IDs")
    geometry = header["geometry"]
    if not isinstance(geometry, dict) or geometry != _geometry(
        geometry.get("camera_height"), geometry.get("camera_width")
    ):
        raise CountedReceiverCodecError("counted packet geometry has field or constant drift")
    try:
        DisjointResizeOperator.build(
            camera_h=geometry["camera_height"],
            camera_w=geometry["camera_width"],
            scorer_h=geometry["scorer_height"],
            scorer_w=geometry["scorer_width"],
        )
    except Uint8LatticeError as exc:
        raise CountedReceiverCodecError("counted geometry is not factor-2 realizable") from exc
    rows = header["sections"]
    if not isinstance(rows, list) or len(rows) != len(SECTION_ORDER) or header["section_count"] != len(rows):
        raise CountedReceiverCodecError("counted section cardinality drift")
    cursor = header_end
    sections: list[bytes] = []
    for index, (name, codec) in enumerate(zip(SECTION_ORDER, SECTION_CODECS, strict=True)):
        row = rows[index]
        if not isinstance(row, dict) or set(row) != _SECTION_FIELDS:
            raise CountedReceiverCodecError("counted section row has field drift")
        if row["section_id"] != name or row["codec_id"] != codec:
            raise CountedReceiverCodecError("counted section order/codec drift")
        if cursor + _SECTION_LENGTH.size > len(packet):
            raise CountedReceiverCodecError("counted section framing is truncated")
        (size,) = _SECTION_LENGTH.unpack_from(packet, cursor)
        cursor += _SECTION_LENGTH.size
        if size > _MAX_SECTION_BYTES or size != row["bytes"] or cursor + size > len(packet):
            raise CountedReceiverCodecError("counted section length differs from its custody row")
        section = packet[cursor : cursor + size]
        cursor += size
        if _sha256(section) != row["sha256"]:
            raise CountedReceiverCodecError("counted section hash differs from its custody row")
        sections.append(section)
    if cursor != len(packet) or header["packet_bytes"] != len(packet):
        raise CountedReceiverCodecError("counted packet has trailing bytes or byte-accounting drift")
    if (
        header["section_payload_bytes"] != sum(map(len, sections))
        or header["section_framing_bytes"] != len(sections) * _SECTION_LENGTH.size
    ):
        raise CountedReceiverCodecError("counted section aggregate accounting drift")
    try:
        pbr1_envelope = decode_predictor_bound_partition_residual(sections[1])
    except PredictorBoundResidualError as exc:
        raise CountedReceiverCodecError("counted PBR1 section is invalid") from exc
    pbr1_accounting_fields = (
        "pbr1_target_derived_section_bytes",
        "pbr1_nested_semantic_residual_bytes",
        "pbr1_event_count",
        "pbr1_event_density_numerator",
        "pbr1_event_density_denominator",
    )
    if any(type(header[field]) is not int or header[field] < 0 for field in pbr1_accounting_fields):
        raise CountedReceiverCodecError("counted PBR1 accounting fields are not exact nonnegative integers")
    if header["pbr1_nested_semantic_residual_bytes"] != len(pbr1_envelope.residual_payload):
        raise CountedReceiverCodecError("counted PBR1 nested semantic-residual accounting drift")
    event_count = len(pbr1_envelope.seed.events)
    event_population = pbr1_envelope.seed.n_pairs * pbr1_envelope.seed.height * pbr1_envelope.seed.width
    if (
        header["pbr1_target_derived_section_bytes"] != len(sections[1])
        or header["pbr1_event_count"] != event_count
        or header["pbr1_event_density_numerator"] != event_count
        or header["pbr1_event_density_denominator"] != event_population
    ):
        raise CountedReceiverCodecError("counted PBR1 target-derived density accounting drift")
    return ParsedCountedPacket(packet=packet, header=header, sections=tuple(sections))  # type: ignore[arg-type]


def _zip_bytes(packet: bytes) -> bytes:
    output = io.BytesIO()
    info = zipfile.ZipInfo(MEMBER_NAME, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as archive:
        archive.writestr(info, packet)
    return output.getvalue()


def build_counted_receiver_archive(**packet_kwargs: Any) -> bytes:
    """Build deterministic bytes only after full cross-section receive closure."""

    archive = _zip_bytes(build_counted_receiver_packet(**packet_kwargs))
    parse_counted_receiver_archive(archive)
    return archive


def parse_counted_receiver_archive(archive: bytes) -> ParsedCountedPacket:
    """Require one canonical stored member and return its strict packet parse."""

    if not isinstance(archive, bytes) or not archive:
        raise CountedReceiverCodecError("archive must be non-empty immutable bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
            infos = reader.infolist()
            if len(infos) != 1 or infos[0].filename != MEMBER_NAME or reader.comment:
                raise CountedReceiverCodecError("archive must contain exactly canonical 0.bin")
            info = infos[0]
            if (
                info.compress_type != zipfile.ZIP_STORED
                or info.date_time != (1980, 1, 1, 0, 0, 0)
                or info.create_system != 3
                or info.external_attr != 0o100644 << 16
                or info.extra
                or info.comment
            ):
                raise CountedReceiverCodecError("archive member metadata is not canonical")
            packet = reader.read(info)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise CountedReceiverCodecError("archive is not a valid counted ZIP") from exc
    if _zip_bytes(packet) != archive:
        raise CountedReceiverCodecError("archive bytes are not the canonical deterministic encoding")
    return parse_counted_receiver_packet(packet)


def _receive_semantics(
    parsed: ParsedCountedPacket,
    *,
    repository_root: Path | None,
) -> tuple[FactorizedV9SemanticReceiver, np.ndarray, np.ndarray, np.ndarray, tuple[SparseRGBOverride, ...]]:
    program, residual, preimage = parsed.sections
    try:
        receiver = receive_factorized_v9_predictor(program, repository_root=repository_root)
        predictor = receiver.decode_all_semantics()
        target = apply_predictor_bound_partition_residual(
            residual,
            predictor_program=program,
            predictor_contract_id=PREDICTOR_CONTRACT_ID,
            predictor_renderer_sha256=receiver.source_manifest_sha256,
            predictor_labels=predictor,
        )
        residual_envelope = decode_predictor_bound_partition_residual(residual)
    except (FactorizedV9PredictorError, PredictorBoundResidualError) as exc:
        raise CountedReceiverCodecError("predictor/PBR1 receiver refused counted semantics") from exc
    expected_ids = _canonical_source_pair_ids(receiver)
    packet_ids = tuple(parsed.header["pair_ids"])
    if packet_ids != expected_ids:
        raise CountedReceiverCodecError("ordered pair population differs from predictor source coordinates")
    if residual_envelope.seed.n_pairs != receiver.pair_count or target.shape != predictor.shape:
        raise CountedReceiverCodecError("PBR1 pair population or semantic geometry differs from predictor")
    palette, overrides = decode_rgb_preimage(
        preimage,
        pair_count=receiver.pair_count,
        height=int(target.shape[1]),
        width=int(target.shape[2]),
    )
    return receiver, predictor, target, palette, overrides


def _iter_realized_pairs(
    target: np.ndarray,
    palette: np.ndarray,
    overrides: Sequence[SparseRGBOverride],
    operator: DisjointResizeOperator,
) -> Iterator[tuple[bytes, bytes, int, int]]:
    overrides_by_pair: list[list[SparseRGBOverride]] = [[] for _ in range(int(target.shape[0]))]
    for row in overrides:
        overrides_by_pair[row.pair_index].append(row)
    for pair_index in range(int(target.shape[0])):
        labels = target[pair_index]
        scorer_pair = np.empty((2, int(target.shape[1]), int(target.shape[2]), 3), dtype=np.uint8)
        for plane in range(2):
            scorer_pair[plane] = palette[pair_index, plane][labels]
        for row in overrides_by_pair[pair_index]:
            scorer_pair[row.plane, row.row, row.col] = (row.red, row.green, row.blue)
        scorer_pair = np.ascontiguousarray(scorer_pair)
        frames: list[np.ndarray] = []
        scorer_values = 0
        numerator_values = 0
        for plane in range(2):
            scorer_plane = scorer_pair[plane]
            frame = realize_factor2_uint8_scorer_plane(operator, scorer_plane)
            proof = verify_factor2_uint8_scorer_plane(operator, frame, scorer_plane)
            if not proof.certified_exact or not proof.numerator_exact:
                raise CountedReceiverCodecError("factor-2 realization failed exact numerator parse-back")
            frames.append(frame)
            scorer_values += proof.scorer_values
            numerator_values += proof.numerator_equal_values
        yield (
            frames[0].tobytes(order="C") + frames[1].tobytes(order="C"),
            scorer_pair.tobytes(order="C"),
            scorer_values,
            numerator_values,
        )


def decode_counted_receiver_archive(
    archive: bytes,
    *,
    repository_root: Path | None = None,
    max_raw_bytes: int = 64 << 20,
) -> CountedDecodeResult:
    """Decode a bounded structural archive fully in memory with exact receipts.

    Production-size callers should use :func:`write_counted_receiver_raw`,
    which streams and checkpoints instead of retaining all raw frames.
    """

    parsed = parse_counted_receiver_archive(archive)
    receiver, predictor, target, palette, overrides = _receive_semantics(parsed, repository_root=repository_root)
    geometry = parsed.header["geometry"]
    operator = DisjointResizeOperator.build(
        camera_h=geometry["camera_height"],
        camera_w=geometry["camera_width"],
        scorer_h=geometry["scorer_height"],
        scorer_w=geometry["scorer_width"],
    )
    expected_raw = receiver.pair_count * 2 * geometry["camera_height"] * geometry["camera_width"] * 3
    _exact_int(max_raw_bytes, "max_raw_bytes", minimum=1)
    if expected_raw > max_raw_bytes:
        raise CountedReceiverCodecError("raw output exceeds the in-memory decode cap; use streaming writer")
    pieces: list[bytes] = []
    scorer_digest = hashlib.sha256()
    scorer_values = 0
    numerator_values = 0
    for pair_raw, pair_scorer_rgb, pair_scorer_values, pair_numerator_values in _iter_realized_pairs(
        target, palette, overrides, operator
    ):
        pieces.append(pair_raw)
        scorer_digest.update(pair_scorer_rgb)
        scorer_values += pair_scorer_values
        numerator_values += pair_numerator_values
    raw = b"".join(pieces)
    if len(raw) != expected_raw:
        raise CountedReceiverCodecError("realized raw byte count drift")
    receipt = _receipt(
        archive,
        parsed,
        predictor,
        target,
        scorer_rgb_sha256=scorer_digest.hexdigest(),
        raw_sha256=_sha256(raw),
        raw_bytes=len(raw),
        scorer_values=scorer_values,
        numerator_values=numerator_values,
    )
    return CountedDecodeResult(raw=raw, receipt=receipt)


def _receipt(
    archive: bytes,
    parsed: ParsedCountedPacket,
    predictor: np.ndarray,
    target: np.ndarray,
    scorer_rgb_sha256: str,
    *,
    raw_sha256: str,
    raw_bytes: int,
    scorer_values: int,
    numerator_values: int,
) -> CountedDecodeReceipt:
    section_bytes = {name: len(section) for name, section in zip(SECTION_ORDER, parsed.sections, strict=True)}
    section_sha = {name: _sha256(section) for name, section in zip(SECTION_ORDER, parsed.sections, strict=True)}
    palette, overrides = decode_rgb_preimage(
        parsed.sections[2],
        pair_count=int(parsed.header["pair_count"]),
        height=int(parsed.header["geometry"]["scorer_height"]),
        width=int(parsed.header["geometry"]["scorer_width"]),
    )
    pbr1_envelope = decode_predictor_bound_partition_residual(parsed.sections[1])
    return CountedDecodeReceipt(
        archive_bytes=len(archive),
        archive_sha256=_sha256(archive),
        packet_bytes=len(parsed.packet),
        packet_sha256=_sha256(parsed.packet),
        archive_container_bytes=len(archive) - len(parsed.packet),
        packet_framing_and_header_bytes=len(parsed.packet) - sum(section_bytes.values()),
        section_bytes=section_bytes,
        section_sha256=section_sha,
        pbr1_counted_bytes=len(parsed.sections[1]),
        pbr1_target_derived_section_bytes=len(parsed.sections[1]),
        pbr1_nested_semantic_residual_bytes=len(pbr1_envelope.residual_payload),
        pbr1_event_count=len(pbr1_envelope.seed.events),
        pbr1_event_density_numerator=len(pbr1_envelope.seed.events),
        pbr1_event_density_denominator=(
            pbr1_envelope.seed.n_pairs * pbr1_envelope.seed.height * pbr1_envelope.seed.width
        ),
        separate_dense_target_table_section_bytes=0,
        target_derived_residual_promotion_admitted=False,
        exact_target_semantic_reconstruction=True,
        candidate_payload_allowed=False,
        candidate_archive_blocker=("contains lossless predictor-conditional target-semantic-table residual"),
        pair_ids=tuple(parsed.header["pair_ids"]),
        pair_population_sha256=str(parsed.header["pair_population_sha256"]),
        predictor_semantic_sha256=_sha256(memoryview(np.ascontiguousarray(predictor)).cast("B")),
        target_semantic_sha256=_sha256(memoryview(np.ascontiguousarray(target)).cast("B")),
        scorer_rgb_sha256=scorer_rgb_sha256,
        preimage_palette_bytes=int(palette.size),
        preimage_override_records=len(overrides),
        raw_bytes=raw_bytes,
        raw_sha256=raw_sha256,
        factor2_scorer_values_verified=scorer_values,
        factor2_numerator_values_verified=numerator_values,
        factor2_certified_exact=True,
        exact_archive_parse_back=True,
        codec_role=CODEC_ROLE,
        capstone_eligible=False,
        shared_semantic_partition_across_planes=True,
        repository_decoder_dependency=True,
        standalone_inflate_source_custody=False,
        standalone_inflate_owed=True,
        authority_receipt_owed=True,
        evaluator_obligation_ir_bound=False,
        coupled_witness_state_bound=False,
        hard_oracle_admission_bound=False,
        independent_frame0_preimage=False,
        research_only=True,
        score_claim=False,
        promotion_eligible=False,
    )


def write_counted_receiver_raw(
    archive: bytes,
    output_path: Path,
    *,
    repository_root: Path | None = None,
) -> CountedDecodeReceipt:
    """Stream raw frames atomically with a resumable per-pair checkpoint.

    ``<output>.partial`` and ``<output>.resume.json`` are preserved after a
    crash. A durable state owns the completed prefix; any state-lagging suffix
    is deterministically truncated and replayed. The success path atomically
    installs the final raw file and keeps the small receipt checkpoint.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(output.parent)
    with _exclusive_output_lock(output):
        return _write_counted_receiver_raw_locked(
            archive,
            output,
            repository_root=repository_root,
        )


def _write_counted_receiver_raw_locked(
    archive: bytes,
    output: Path,
    *,
    repository_root: Path | None,
) -> CountedDecodeReceipt:
    parsed = parse_counted_receiver_archive(archive)
    _receiver, predictor, target, palette, overrides = _receive_semantics(parsed, repository_root=repository_root)
    geometry = parsed.header["geometry"]
    operator = DisjointResizeOperator.build(
        camera_h=geometry["camera_height"],
        camera_w=geometry["camera_width"],
        scorer_h=geometry["scorer_height"],
        scorer_w=geometry["scorer_width"],
    )
    partial = output.with_name(f"{output.name}.partial")
    state_path = output.with_name(f"{output.name}.resume.json")
    pair_bytes = 2 * geometry["camera_height"] * geometry["camera_width"] * 3
    expected_bytes = int(parsed.header["pair_count"]) * pair_bytes
    free_bytes = shutil.disk_usage(output.parent).free
    existing_bytes = partial.stat().st_size if partial.is_file() else 0
    if existing_bytes > expected_bytes:
        raise CountedReceiverCodecError("partial raw exceeds the deterministic output size")
    if free_bytes < expected_bytes - min(existing_bytes, expected_bytes) + (1 << 20):
        raise CountedReceiverCodecError("storage preflight refused insufficient output-tier space")
    archive_sha = _sha256(archive)
    completed_pairs = 0
    if state_path.is_file() and existing_bytes:
        raw_state = state_path.read_bytes()
        try:
            state = json.loads(raw_state.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CountedReceiverCodecError("resume checkpoint is not valid ASCII JSON") from exc
        if not isinstance(state, dict) or _canonical_json(state) != raw_state:
            raise CountedReceiverCodecError("resume checkpoint is not canonical JSON")
        state_completed_pairs = state.get("completed_pairs")
        state_partial_bytes = state.get("partial_bytes")
        if (
            state.get("schema") != ARCHIVE_SCHEMA
            or state.get("archive_sha256") != archive_sha
            or state.get("pair_count") != int(parsed.header["pair_count"])
            or state.get("pair_bytes") != pair_bytes
            or state.get("research_only") is not True
            or state.get("score_claim") is not False
            or type(state_completed_pairs) is not int
            or not 0 <= state_completed_pairs <= int(parsed.header["pair_count"])
            or type(state_partial_bytes) is not int
            or state_partial_bytes != state_completed_pairs * pair_bytes
            or state_partial_bytes > existing_bytes
        ):
            raise CountedReceiverCodecError("resume checkpoint custody differs from partial raw")
        with partial.open("rb") as handle:
            partial_sha = _sha256_file_prefix(handle, state_partial_bytes)
        if state.get("partial_prefix_sha256") != partial_sha:
            raise CountedReceiverCodecError("resume checkpoint hash differs from partial raw prefix")
        completed_pairs = state_completed_pairs
        if existing_bytes != state_partial_bytes:
            with partial.open("r+b") as handle:
                handle.truncate(state_partial_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(partial.parent)
            existing_bytes = state_partial_bytes
    elif existing_bytes:
        # Raw bytes written after the last durable state are one uncommitted
        # interval, not authority. Roll them back and deterministically replay.
        with partial.open("r+b") as handle:
            handle.truncate(0)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(partial.parent)
        existing_bytes = 0
    digest = hashlib.sha256()
    scorer_digest = hashlib.sha256()
    scorer_values = 0
    numerator_values = 0
    mode = "r+b" if partial.exists() else "w+b"
    with partial.open(mode) as handle:
        handle.seek(0)
        for pair_index, (pair_raw, pair_scorer_rgb, pair_scorer, pair_numerator) in enumerate(
            _iter_realized_pairs(target, palette, overrides, operator)
        ):
            scorer_values += pair_scorer
            numerator_values += pair_numerator
            scorer_digest.update(pair_scorer_rgb)
            if pair_index < completed_pairs:
                if handle.read(pair_bytes) != pair_raw:
                    raise CountedReceiverCodecError("partial raw prefix differs from deterministic replay")
            else:
                handle.seek(0, os.SEEK_END)
                handle.write(pair_raw)
                handle.flush()
                os.fsync(handle.fileno())
            digest.update(pair_raw)
            if pair_index >= completed_pairs:
                state = {
                    "schema": ARCHIVE_SCHEMA,
                    "archive_sha256": archive_sha,
                    "completed_pairs": pair_index + 1,
                    "pair_count": int(parsed.header["pair_count"]),
                    "pair_bytes": pair_bytes,
                    "partial_bytes": (pair_index + 1) * pair_bytes,
                    "partial_prefix_sha256": digest.hexdigest(),
                    "research_only": True,
                    "score_claim": False,
                }
                _atomic_write_resume_state(state_path, state)
        handle.flush()
        os.fsync(handle.fileno())
    if partial.stat().st_size != expected_bytes:
        raise CountedReceiverCodecError("completed partial raw byte count drift")
    raw_sha = digest.hexdigest()
    if output.exists():
        if not output.is_file() or output.stat().st_size != expected_bytes:
            raise CountedReceiverCodecError("existing final raw path conflicts with deterministic output")
        with output.open("rb") as handle:
            if _sha256_file(handle) != raw_sha:
                raise CountedReceiverCodecError("existing final raw bytes differ from deterministic output")
        partial.unlink()
        _fsync_directory(output.parent)
    else:
        os.replace(partial, output)
        _fsync_directory(output.parent)
    final_state = {
        "schema": ARCHIVE_SCHEMA,
        "archive_sha256": archive_sha,
        "completed_pairs": int(parsed.header["pair_count"]),
        "pair_count": int(parsed.header["pair_count"]),
        "pair_bytes": pair_bytes,
        "partial_bytes": expected_bytes,
        "partial_prefix_sha256": raw_sha,
        "final_raw_sha256": raw_sha,
        "final_raw_bytes": expected_bytes,
        "completed": True,
        "research_only": True,
        "score_claim": False,
    }
    _atomic_write_resume_state(state_path, final_state)
    return _receipt(
        archive,
        parsed,
        predictor,
        target,
        scorer_rgb_sha256=scorer_digest.hexdigest(),
        raw_sha256=raw_sha,
        raw_bytes=expected_bytes,
        scorer_values=scorer_values,
        numerator_values=numerator_values,
    )


__all__ = [
    "ARCHIVE_SCHEMA",
    "MEMBER_NAME",
    "PACKET_SCHEMA",
    "PREIMAGE_SCHEMA",
    "CountedDecodeReceipt",
    "CountedDecodeResult",
    "CountedReceiverCodecError",
    "ParsedCountedPacket",
    "SparseRGBOverride",
    "build_counted_receiver_archive",
    "build_counted_receiver_packet",
    "decode_counted_receiver_archive",
    "decode_rgb_preimage",
    "encode_rgb_preimage",
    "pair_population_sha256",
    "parse_counted_receiver_archive",
    "parse_counted_receiver_packet",
    "write_counted_receiver_raw",
]
