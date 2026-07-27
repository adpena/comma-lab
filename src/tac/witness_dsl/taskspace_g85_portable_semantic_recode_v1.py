# SPDX-License-Identifier: MIT
"""Encoder-side Brotli-to-zlib recode for the G82 compact PVSA semantic P.

The frozen upstream environment does not provide Brotli or OpenCV.  This
module closes the entropy-codec half of that public-runtime gap without
changing decoded semantic bytes:

* five Brotli-coded chart transforms become version-2 zlib frames;
* three Brotli-coded structured component streams become coder-id-3 zlib;
* two untagged Brotli static masks become explicitly framed ``PZSM1`` zlib;
* the two Brotli G1 productions become the already-registered G1 zlib codec.

Every changed section is decoded before and after recoding and bound by exact
decoded bytes/SHA-256.  The result is a new counted member/archive lineage,
never a mutation or identity claim about G82's original bytes.

This does not close the public receiver.  G1 polygon rasterization still uses
``cv2.fillPoly`` in the repository receiver, and Pillow is measured non-parity
on the complete n600 G1 corpus.  A bit-identical generic rasterizer and a
tree-shaken upstream-only receiver remain mandatory.
"""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import re
import struct
import zipfile
import zlib
from dataclasses import asdict, dataclass
from typing import Any, Final

import brotli

from tac.optimization.direct_description_carrier_compose import rfc8785_canonicalize
from tac.optimization.direct_description_entropy_priced_member import (
    _zip_stored,
)
from tac.optimization.direct_description_entropy_streams import (
    _ENTROPY_FRAME,
    CODER_BROTLI_Q11,
    CODER_SPLIT_RICE,
    MEMBER_BY_STREAM,
    STREAM_BY_MEMBER,
    STREAM_ORDER,
    _decode_coder,
    _decode_transform,
    _deterministic_zip,
    parse_entropy_chart_archive,
    parse_entropy_stream,
)
from tac.optimization.direct_description_g1_worldsheet import (
    CODEC_NAMES,
    PRODUCTION_NAMES,
    _decode_envelope,
)
from tac.optimization.direct_description_g1_worldsheet import (
    MAGIC as G1_MAGIC,
)
from tac.witness_dsl.taskspace_pvsa_compact_container_v1 import (
    CompactPVSAArchiveBuildV1,
    CompactPVSAError,
    _canonical_semantic_zip,
    build_compact_pvsa_archive,
    parse_compact_pvsa_member,
)

PORTABLE_CHART_VERSION: Final = 2
PORTABLE_CHART_ZLIB_CODER_ID: Final = 6
PORTABLE_SITE_ZLIB_CODER_ID: Final = 3
PORTABLE_STATIC_MAGIC: Final = b"PZSM1"
PORTABLE_STATIC_VERSION: Final = 1
PORTABLE_STATIC_ZLIB_CODER_ID: Final = 1
G1_ZLIB_CODEC_ID: Final = 3
_SITE_HEADER: Final = struct.Struct(">BII32s")
_STATIC_V1_HEADER: Final = struct.Struct(">II32s")
_PORTABLE_STATIC_HEADER: Final = struct.Struct(">5sBBII32s")
_G1_SECTION_HEADER: Final = struct.Struct("<BBII")

RECEIPT_SCHEMA: Final = "tac.g85_portable_semantic_recode_receipt.v1"
RASTERIZER_BLOCKER: Final = "G1_OPENCV_FILLPOLY_BIT_EXACT_GENERIC_RASTERIZER_OWED"
TREE_SHAKE_BLOCKER: Final = "PVSA_DECODER_ONLY_UPSTREAM_IMPORT_CLOSURE_OWED"


class PortableSemanticRecodeError(ValueError):
    """The source wire, decoded semantics, or portable recode drifted."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_store_zip(payload: bytes, *, expected_names: tuple[str, ...]) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as reader:
            infos = reader.infolist()
            if tuple(row.filename for row in infos) != expected_names:
                raise PortableSemanticRecodeError("portable recode ZIP member order differs")
            if any(
                row.is_dir()
                or row.compress_type != zipfile.ZIP_STORED
                or row.date_time != (1980, 1, 1, 0, 0, 0)
                or row.filename.startswith("/")
                or ".." in row.filename.split("/")
                for row in infos
            ):
                raise PortableSemanticRecodeError("portable recode ZIP metadata is noncanonical")
            return {row.filename: reader.read(row) for row in infos}
    except PortableSemanticRecodeError:
        raise
    except (KeyError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise PortableSemanticRecodeError("portable recode ZIP parse failed") from exc


def _section_row(
    *,
    name: str,
    source: bytes,
    portable: bytes,
    decoded: bytes,
    source_codec: str,
    portable_codec: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "source_bytes": len(source),
        "source_sha256": _sha256(source),
        "portable_bytes": len(portable),
        "portable_sha256": _sha256(portable),
        "decoded_bytes": len(decoded),
        "decoded_sha256": _sha256(decoded),
        "decoded_bytes_identical": True,
        "source_codec": source_codec,
        "portable_codec": portable_codec,
    }


def _portable_chart_semantic(stream_name: str, frame: bytes) -> bytes:
    if len(frame) < _ENTROPY_FRAME.size:
        raise PortableSemanticRecodeError("portable chart frame is truncated")
    (
        _magic,
        version,
        observed_pairs,
        transform_id,
        coder_id,
        reserved,
        _records,
        semantic_bytes,
        canonical_bytes,
        coded_bytes,
        digest,
    ) = _ENTROPY_FRAME.unpack_from(frame)
    if reserved != 0 or len(frame) != _ENTROPY_FRAME.size + coded_bytes:
        raise PortableSemanticRecodeError("portable chart frame length/reserved field differs")
    if version == 1 and coder_id == CODER_SPLIT_RICE:
        pairs, semantic = parse_entropy_stream(stream_name, frame)
        if pairs != observed_pairs:
            raise PortableSemanticRecodeError("portable unchanged split-Rice pair count differs")
        return semantic
    if version != PORTABLE_CHART_VERSION or coder_id != PORTABLE_CHART_ZLIB_CODER_ID:
        raise PortableSemanticRecodeError("portable chart version/coder is unknown")
    try:
        canonical = zlib.decompress(frame[_ENTROPY_FRAME.size :])
    except zlib.error as exc:
        raise PortableSemanticRecodeError("portable chart zlib decode failed") from exc
    if len(canonical) != canonical_bytes:
        raise PortableSemanticRecodeError("portable chart canonical byte count differs")
    semantic = _decode_transform(stream_name, observed_pairs, transform_id, canonical)
    if len(semantic) != semantic_bytes or hashlib.sha256(semantic).digest() != digest:
        raise PortableSemanticRecodeError("portable chart semantic length/hash differs")
    return semantic


def _transcode_chart(chart: bytes) -> tuple[bytes, list[dict[str, Any]]]:
    parse_entropy_chart_archive(chart)
    names = tuple(MEMBER_BY_STREAM[name] for name in STREAM_ORDER)
    source_members = _read_store_zip(chart, expected_names=names)
    portable_members: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    for member_name in names:
        stream_name = STREAM_BY_MEMBER[member_name]
        frame = source_members[member_name]
        values = list(_ENTROPY_FRAME.unpack_from(frame))
        version = values[1]
        coder_id = values[4]
        coded_bytes = values[9]
        if version != 1 or len(frame) != _ENTROPY_FRAME.size + coded_bytes:
            raise PortableSemanticRecodeError("source chart frame version/length differs")
        source_semantic = parse_entropy_stream(stream_name, frame)[1]
        if coder_id == CODER_BROTLI_Q11:
            canonical = _decode_coder(coder_id, frame[_ENTROPY_FRAME.size :], values[8])
            coded = zlib.compress(canonical, level=9)
            values[1] = PORTABLE_CHART_VERSION
            values[4] = PORTABLE_CHART_ZLIB_CODER_ID
            values[9] = len(coded)
            portable = _ENTROPY_FRAME.pack(*values) + coded
            rows.append(
                _section_row(
                    name=f"predictor.zip::chart.zip::{member_name}",
                    source=frame,
                    portable=portable,
                    decoded=source_semantic,
                    source_codec="brotli_q11",
                    portable_codec="zlib9",
                )
            )
        elif coder_id == CODER_SPLIT_RICE and stream_name == STREAM_ORDER[-1]:
            portable = frame
        else:
            raise PortableSemanticRecodeError("source chart has an unregistered recode case")
        if _portable_chart_semantic(stream_name, portable) != source_semantic:
            raise PortableSemanticRecodeError("portable chart decoded semantic bytes differ")
        portable_members[member_name] = portable
    portable_chart = _deterministic_zip(portable_members)
    reopened = _read_store_zip(portable_chart, expected_names=names)
    if reopened != portable_members:
        raise PortableSemanticRecodeError("portable chart ZIP parse-back differs")
    return portable_chart, rows


def _transcode_site_record(name: str, payload: bytes) -> tuple[bytes, dict[str, Any]]:
    if len(payload) < _SITE_HEADER.size:
        raise PortableSemanticRecodeError("source site record is truncated")
    coder_id, raw_bytes, coded_bytes, digest = _SITE_HEADER.unpack_from(payload)
    if coder_id != 2 or len(payload) != _SITE_HEADER.size + coded_bytes:
        raise PortableSemanticRecodeError("source site record is not exact Brotli V1")
    try:
        raw = brotli.decompress(payload[_SITE_HEADER.size :])
    except brotli.error as exc:
        raise PortableSemanticRecodeError("source site record Brotli decode failed") from exc
    if len(raw) != raw_bytes or hashlib.sha256(raw).digest() != digest:
        raise PortableSemanticRecodeError("source site record decoded custody differs")
    coded = zlib.compress(raw, level=9)
    portable = _SITE_HEADER.pack(PORTABLE_SITE_ZLIB_CODER_ID, len(raw), len(coded), digest) + coded
    if _decode_portable_site_record(portable) != raw:
        raise PortableSemanticRecodeError("portable site record decoded bytes differ")
    return portable, _section_row(
        name=name,
        source=payload,
        portable=portable,
        decoded=raw,
        source_codec="brotli_q11",
        portable_codec="zlib9",
    )


def _decode_portable_site_record(payload: bytes) -> bytes:
    if len(payload) < _SITE_HEADER.size:
        raise PortableSemanticRecodeError("portable site record is truncated")
    coder_id, raw_bytes, coded_bytes, digest = _SITE_HEADER.unpack_from(payload)
    if coder_id != PORTABLE_SITE_ZLIB_CODER_ID or len(payload) != _SITE_HEADER.size + coded_bytes:
        raise PortableSemanticRecodeError("portable site record version/length differs")
    try:
        raw = zlib.decompress(payload[_SITE_HEADER.size :])
    except zlib.error as exc:
        raise PortableSemanticRecodeError("portable site record zlib decode failed") from exc
    if len(raw) != raw_bytes or hashlib.sha256(raw).digest() != digest:
        raise PortableSemanticRecodeError("portable site record decoded custody differs")
    return raw


def _transcode_static_mask(name: str, payload: bytes) -> tuple[bytes, dict[str, Any]]:
    if len(payload) < _STATIC_V1_HEADER.size:
        raise PortableSemanticRecodeError("source static mask is truncated")
    sites, raw_bytes, digest = _STATIC_V1_HEADER.unpack_from(payload)
    try:
        raw = brotli.decompress(payload[_STATIC_V1_HEADER.size :])
    except brotli.error as exc:
        raise PortableSemanticRecodeError("source static mask Brotli decode failed") from exc
    if len(raw) != raw_bytes or hashlib.sha256(raw).digest() != digest:
        raise PortableSemanticRecodeError("source static mask decoded custody differs")
    coded = zlib.compress(raw, level=9)
    portable = (
        _PORTABLE_STATIC_HEADER.pack(
            PORTABLE_STATIC_MAGIC,
            PORTABLE_STATIC_VERSION,
            PORTABLE_STATIC_ZLIB_CODER_ID,
            sites,
            len(raw),
            digest,
        )
        + coded
    )
    if _decode_portable_static_mask(portable) != raw:
        raise PortableSemanticRecodeError("portable static mask decoded bytes differ")
    return portable, _section_row(
        name=name,
        source=payload,
        portable=portable,
        decoded=raw,
        source_codec="brotli_q11_untagged_v1",
        portable_codec="pzsm1_zlib9",
    )


def _decode_portable_static_mask(payload: bytes) -> bytes:
    if len(payload) < _PORTABLE_STATIC_HEADER.size:
        raise PortableSemanticRecodeError("portable static mask is truncated")
    magic, version, coder_id, sites, raw_bytes, digest = _PORTABLE_STATIC_HEADER.unpack_from(payload)
    if (
        magic != PORTABLE_STATIC_MAGIC
        or version != PORTABLE_STATIC_VERSION
        or coder_id != PORTABLE_STATIC_ZLIB_CODER_ID
        or sites != 384 * 512
    ):
        raise PortableSemanticRecodeError("portable static mask identity differs")
    try:
        raw = zlib.decompress(payload[_PORTABLE_STATIC_HEADER.size :])
    except zlib.error as exc:
        raise PortableSemanticRecodeError("portable static mask zlib decode failed") from exc
    if len(raw) != raw_bytes or hashlib.sha256(raw).digest() != digest:
        raise PortableSemanticRecodeError("portable static mask decoded custody differs")
    return raw


def _read_g1_sections(payload: bytes) -> list[tuple[int, int, bytes, bytes]]:
    _decode_envelope(payload)
    if len(payload) < 5 or payload[:4] != G1_MAGIC or payload[4] != 3:
        raise PortableSemanticRecodeError("source G1 envelope identity differs")
    offset = 5
    rows: list[tuple[int, int, bytes, bytes]] = []
    for _ in range(payload[4]):
        if offset + _G1_SECTION_HEADER.size > len(payload):
            raise PortableSemanticRecodeError("source G1 section header is truncated")
        production_id, codec_id, raw_size, coded_size = _G1_SECTION_HEADER.unpack_from(payload, offset)
        offset += _G1_SECTION_HEADER.size
        coded = payload[offset : offset + coded_size]
        offset += coded_size
        try:
            if codec_id == 1:
                raw = brotli.decompress(coded)
            elif codec_id == 2:
                raw = lzma.decompress(
                    coded,
                    format=lzma.FORMAT_RAW,
                    filters=[{"id": lzma.FILTER_LZMA1, "preset": 1, "dict_size": 1 << 20}],
                )
            elif codec_id == G1_ZLIB_CODEC_ID:
                raw = zlib.decompress(coded)
            else:
                raise PortableSemanticRecodeError("source G1 codec is unknown")
        except (brotli.error, lzma.LZMAError, zlib.error) as exc:
            raise PortableSemanticRecodeError("source G1 section decode failed") from exc
        if len(raw) != raw_size:
            raise PortableSemanticRecodeError("source G1 raw byte count differs")
        rows.append((production_id, codec_id, coded, raw))
    if offset != len(payload):
        raise PortableSemanticRecodeError("source G1 envelope has trailing bytes")
    return rows


def _transcode_g1(payload: bytes) -> tuple[bytes, list[dict[str, Any]]]:
    sections = _read_g1_sections(payload)
    output = bytearray(G1_MAGIC)
    output.append(len(sections))
    receipt_rows: list[dict[str, Any]] = []
    for production_id, codec_id, source_coded, raw in sections:
        if production_id not in PRODUCTION_NAMES or codec_id not in CODEC_NAMES:
            raise PortableSemanticRecodeError("source G1 production/codec registry differs")
        if codec_id == 1:
            portable_coded = zlib.compress(raw, level=9)
            portable_codec_id = G1_ZLIB_CODEC_ID
        elif codec_id == 2:
            portable_coded = source_coded
            portable_codec_id = codec_id
        else:
            raise PortableSemanticRecodeError("source G1 has an unregistered recode case")
        output.extend(
            _G1_SECTION_HEADER.pack(
                production_id,
                portable_codec_id,
                len(raw),
                len(portable_coded),
            )
        )
        output.extend(portable_coded)
        if codec_id == 1:
            receipt_rows.append(
                _section_row(
                    name=f"semantic_P::predict/movable_polygon_worldsheet.g1s::{PRODUCTION_NAMES[production_id]}",
                    source=source_coded,
                    portable=portable_coded,
                    decoded=raw,
                    source_codec="brotli_q11",
                    portable_codec="zlib9",
                )
            )
    portable = bytes(output)
    portable_sections = _read_portable_g1_sections(portable)
    if [(row[0], row[3]) for row in portable_sections] != [(row[0], row[3]) for row in sections]:
        raise PortableSemanticRecodeError("portable G1 decoded production bytes differ")
    return portable, receipt_rows


def _read_portable_g1_sections(payload: bytes) -> list[tuple[int, int, bytes, bytes]]:
    if len(payload) < 5 or payload[:4] != G1_MAGIC or payload[4] != 3:
        raise PortableSemanticRecodeError("portable G1 envelope identity differs")
    offset = 5
    rows: list[tuple[int, int, bytes, bytes]] = []
    for _ in range(payload[4]):
        if offset + _G1_SECTION_HEADER.size > len(payload):
            raise PortableSemanticRecodeError("portable G1 section header is truncated")
        production_id, codec_id, raw_size, coded_size = _G1_SECTION_HEADER.unpack_from(payload, offset)
        offset += _G1_SECTION_HEADER.size
        if coded_size > len(payload) - offset:
            raise PortableSemanticRecodeError("portable G1 coded payload is truncated")
        coded = payload[offset : offset + coded_size]
        offset += coded_size
        try:
            if codec_id == 2:
                raw = lzma.decompress(
                    coded,
                    format=lzma.FORMAT_RAW,
                    filters=[{"id": lzma.FILTER_LZMA1, "preset": 1, "dict_size": 1 << 20}],
                )
            elif codec_id == G1_ZLIB_CODEC_ID:
                raw = zlib.decompress(coded)
            else:
                raise PortableSemanticRecodeError("portable G1 codec is not upstream-stdlib")
        except (lzma.LZMAError, zlib.error) as exc:
            raise PortableSemanticRecodeError("portable G1 section decode failed") from exc
        if production_id not in PRODUCTION_NAMES or len(raw) != raw_size:
            raise PortableSemanticRecodeError("portable G1 production/raw bytes differ")
        rows.append((production_id, codec_id, coded, raw))
    if offset != len(payload):
        raise PortableSemanticRecodeError("portable G1 envelope has trailing bytes")
    return rows


@dataclass(frozen=True, slots=True)
class PortableSemanticRecodeReceiptV1:
    source_semantic_p_bytes: int
    source_semantic_p_sha256: str
    portable_semantic_p_bytes: int
    portable_semantic_p_sha256: str
    source_compact_member_bytes: int | None
    source_compact_member_sha256: str | None
    portable_compact_member_bytes: int | None
    portable_compact_member_sha256: str | None
    portable_archive_bytes: int | None
    portable_archive_sha256: str | None
    section_rows: tuple[dict[str, Any], ...]
    schema: str = RECEIPT_SCHEMA
    changed_brotli_section_count: int = 12
    all_changed_sections_decoded_byte_identical: bool = True
    output_video_equality_proven: bool = False
    opencv_rasterizer_replaced: bool = False
    tree_shaken_public_receiver_closed: bool = False
    upstream_default_double_decode_proven: bool = False
    evaluator_invoked: bool = False
    score_claim: bool = False
    candidate_claim: bool = False
    research_only: bool = True
    open_blockers: tuple[str, ...] = (RASTERIZER_BLOCKER, TREE_SHAKE_BLOCKER)

    def __post_init__(self) -> None:
        hashes = (
            self.source_semantic_p_sha256,
            self.portable_semantic_p_sha256,
            *(
                value
                for value in (
                    self.source_compact_member_sha256,
                    self.portable_compact_member_sha256,
                    self.portable_archive_sha256,
                )
                if value is not None
            ),
        )
        if (
            self.schema != RECEIPT_SCHEMA
            or any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes)
            or self.changed_brotli_section_count != len(self.section_rows)
            or not all(row.get("decoded_bytes_identical") is True for row in self.section_rows)
            or self.all_changed_sections_decoded_byte_identical is not True
            or self.output_video_equality_proven is not False
            or self.opencv_rasterizer_replaced is not False
            or self.tree_shaken_public_receiver_closed is not False
            or self.upstream_default_double_decode_proven is not False
            or self.evaluator_invoked is not False
            or self.score_claim is not False
            or self.candidate_claim is not False
            or self.research_only is not True
            or self.open_blockers != (RASTERIZER_BLOCKER, TREE_SHAKE_BLOCKER)
        ):
            raise PortableSemanticRecodeError("portable recode receipt truth differs")

    def to_bytes(self) -> bytes:
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")


@dataclass(frozen=True, slots=True)
class PortableSemanticRecodeV1:
    semantic_p_archive: bytes
    receipt: PortableSemanticRecodeReceiptV1
    compact_build: CompactPVSAArchiveBuildV1 | None = None


def transcode_portable_semantic_p(semantic_p_archive: bytes) -> PortableSemanticRecodeV1:
    """Recode every retained Brotli section while preserving decoded semantics."""

    try:
        with zipfile.ZipFile(io.BytesIO(semantic_p_archive), "r") as reader:
            names = tuple(row.filename for row in reader.infolist())
            if names != (
                "manifest.json",
                "predictor.zip",
                "predict/movable_polygon_worldsheet.g1s",
                "render/receiver_realization.ddrp",
                "render/scorer_solved_templates.ddst",
            ):
                raise PortableSemanticRecodeError("semantic P fixed member order differs")
            semantic_payloads = tuple(reader.read(name) for name in names)
    except PortableSemanticRecodeError:
        raise
    except (KeyError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise PortableSemanticRecodeError("semantic P parse failed") from exc
    if _canonical_semantic_zip(semantic_payloads) != semantic_p_archive:
        raise PortableSemanticRecodeError("source semantic P is not canonical")

    semantic_manifest = json.loads(semantic_payloads[0])
    predictor = semantic_payloads[1]
    with zipfile.ZipFile(io.BytesIO(predictor), "r") as reader:
        predictor_names = tuple(row.filename for row in reader.infolist())
        predictor_members = {name: reader.read(name) for name in predictor_names}
    predictor_manifest = json.loads(predictor_members["manifest.json"])

    portable_chart, rows = _transcode_chart(predictor_members["chart.zip"])
    predictor_members["chart.zip"] = portable_chart
    component_names = (
        "structure/undrivable_components.br",
        "structure/road_components.br",
        "structure/lane_components.br",
    )
    static_names = (
        "structure/road_pxq1_mask.br",
        "structure/mycar_static_hood.br",
    )
    for name in component_names:
        predictor_members[name], row = _transcode_site_record(f"predictor.zip::{name}", predictor_members[name])
        rows.append(row)
    for name in static_names:
        predictor_members[name], row = _transcode_static_mask(f"predictor.zip::{name}", predictor_members[name])
        rows.append(row)
    predictor_manifest["baseline_chart"] = {
        "bytes": len(portable_chart),
        "sha256": _sha256(portable_chart),
    }
    predictor_manifest["portable_codec_policy"] = "chart_v2_zlib6;site_record_zlib3;static_mask_pzsm1_zlib1"
    for declaration in predictor_manifest["structured_payloads"]:
        payload = predictor_members[declaration["name"]]
        declaration["bytes"] = len(payload)
        declaration["sha256"] = _sha256(payload)
    predictor_members["manifest.json"] = rfc8785_canonicalize(predictor_manifest)
    portable_predictor = _zip_stored(predictor_members)
    if tuple(_read_store_zip(portable_predictor, expected_names=predictor_names)) != predictor_names:
        raise PortableSemanticRecodeError("portable predictor ZIP parse-back differs")

    portable_g1, g1_rows = _transcode_g1(semantic_payloads[2])
    rows.extend(g1_rows)
    if len(rows) != 12:
        raise PortableSemanticRecodeError("portable recode did not close all twelve Brotli sections")

    semantic_manifest["predictor"] = {
        "bytes": len(portable_predictor),
        "sha256": _sha256(portable_predictor),
    }
    g1_row = semantic_manifest["grammar"]["movable"]["g1_polygon_worldsheet"]
    g1_row["bytes"] = len(portable_g1)
    g1_row["sha256"] = _sha256(portable_g1)
    g1_row["g1_reference_payload_sha256"] = _sha256(portable_g1)
    semantic_manifest["portable_codec_policy"] = "all_retained_brotli_sections_reframed_as_stdlib_zlib_v1"
    portable_payloads = (
        rfc8785_canonicalize(semantic_manifest),
        portable_predictor,
        portable_g1,
        semantic_payloads[3],
        semantic_payloads[4],
    )
    portable_semantic = _canonical_semantic_zip(portable_payloads)
    receipt = PortableSemanticRecodeReceiptV1(
        source_semantic_p_bytes=len(semantic_p_archive),
        source_semantic_p_sha256=_sha256(semantic_p_archive),
        portable_semantic_p_bytes=len(portable_semantic),
        portable_semantic_p_sha256=_sha256(portable_semantic),
        source_compact_member_bytes=None,
        source_compact_member_sha256=None,
        portable_compact_member_bytes=None,
        portable_compact_member_sha256=None,
        portable_archive_bytes=None,
        portable_archive_sha256=None,
        section_rows=tuple(rows),
    )
    return PortableSemanticRecodeV1(
        semantic_p_archive=portable_semantic,
        receipt=receipt,
    )


def transcode_portable_pvsa_member(member: bytes) -> PortableSemanticRecodeV1:
    """Recode exact compact bytes into a new portable counted lineage."""

    try:
        parsed = parse_compact_pvsa_member(
            member,
            maximum_member_bytes=len(member),
            maximum_section_bytes=len(member),
        )
        semantic = transcode_portable_semantic_p(parsed.semantic_p_archive)
        build = build_compact_pvsa_archive(
            semantic_p_archive=semantic.semantic_p_archive,
            actuator_payloads=tuple(row.payload for row in parsed.actuators),
            maximum_semantic_archive_bytes=max(len(semantic.semantic_p_archive), 1),
            maximum_member_bytes=max(len(member) * 4, len(semantic.semantic_p_archive) * 2),
            maximum_section_bytes=max(len(member) * 4, len(semantic.semantic_p_archive) * 2),
        )
    except CompactPVSAError as exc:
        raise PortableSemanticRecodeError("portable compact/archive rebuild failed") from exc
    receipt = PortableSemanticRecodeReceiptV1(
        source_semantic_p_bytes=semantic.receipt.source_semantic_p_bytes,
        source_semantic_p_sha256=semantic.receipt.source_semantic_p_sha256,
        portable_semantic_p_bytes=semantic.receipt.portable_semantic_p_bytes,
        portable_semantic_p_sha256=semantic.receipt.portable_semantic_p_sha256,
        source_compact_member_bytes=len(member),
        source_compact_member_sha256=_sha256(member),
        portable_compact_member_bytes=len(build.selected.member_bytes),
        portable_compact_member_sha256=build.selected.member_sha256,
        portable_archive_bytes=build.outer_build.selected.archive_nbytes,
        portable_archive_sha256=build.outer_build.selected.archive_sha256,
        section_rows=semantic.receipt.section_rows,
    )
    return PortableSemanticRecodeV1(
        semantic_p_archive=semantic.semantic_p_archive,
        receipt=receipt,
        compact_build=build,
    )


__all__ = [
    "RASTERIZER_BLOCKER",
    "TREE_SHAKE_BLOCKER",
    "PortableSemanticRecodeError",
    "PortableSemanticRecodeReceiptV1",
    "PortableSemanticRecodeV1",
    "transcode_portable_pvsa_member",
    "transcode_portable_semantic_p",
]
