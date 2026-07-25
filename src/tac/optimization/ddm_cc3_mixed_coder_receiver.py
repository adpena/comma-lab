# SPDX-License-Identifier: MIT
"""Receiver-closed recursive mixed-coder archives for DDM CC3.

CC2 priced every physical leaf in one exact counted PC1 composition but did
not replace any bytes in the receiver path.  This module closes that boundary:

* only CC2 rows with a negative exact framed-byte delta are replaced;
* the selected ``DCC3`` frame carries its codec switch, raw length, and raw
  SHA-256 inside counted bytes;
* every other physical leaf remains byte-identical;
* every nested stored-ZIP member name, order, and metadata row is preserved;
* restoration re-encodes every context frame and every ZIP layer, then
  requires the canonical PC1 composition parser to accept the exact source.

The arithmetic models are generic receiver code.  They derive all state from
the already-decoded prefix and carry no video-derived parameter table.
"""

from __future__ import annotations

import copy
import hashlib
import io
import stat
import struct
import zipfile
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any, Final

from tac.optimization.arith_selfcomp_rate_coders import (
    decode_bellard_class_mixing,
    decode_g4_decoder_context,
    encode_bellard_class_mixing,
    encode_g4_decoder_context,
)
from tac.optimization.ddm_pc1_pose_stream import (
    MANIFEST_MEMBER,
    PACKET_MEMBER,
    PARENT_MEMBER,
    parse_counted_composition_archive,
)

SCHEMA: Final = "ddm_cc3_mixed_coder_receiver.v1"
PRICE_TABLE_SCHEMA: Final = "ddm_cc2_c1_costate_stream_price_table.v1"
COMPOSITION_OWNER: Final = "composition.zip"
EXPECTED_COMPOSITION_MEMBERS: Final = (
    MANIFEST_MEMBER,
    PACKET_MEMBER,
    PARENT_MEMBER,
)
DCC3_MAGIC: Final = b"DCC3"
G4_CODEC_ID: Final = 1
BELLARD_CODEC_ID: Final = 3
CODEC_ENCODERS: Final = {
    "G4_FREE_DECODER_CONTEXT": encode_g4_decoder_context,
    "BELLARD_CLASS_MIXING": encode_bellard_class_mixing,
}


class MixedCoderReceiverError(ValueError):
    """A selected leaf, frame, nested ZIP, or composition contract differed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_member_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or name.startswith("/") or "\\" in name or any(part in ("", ".", "..") for part in path.parts):
        raise MixedCoderReceiverError(f"unsafe recursive ZIP member: {name!r}")
    return path.as_posix()


def _zip_payload_end(payload: bytes) -> int | None:
    """Return the real ZIP end before an arbitrary receiver-owned suffix."""

    search_stop = len(payload)
    while search_stop:
        candidate = payload.rfind(b"PK\x05\x06", 0, search_stop)
        if candidate < 0:
            return None
        search_stop = candidate
        if candidate + 22 > len(payload):
            continue
        (
            _signature,
            disk_number,
            central_disk,
            disk_entries,
            total_entries,
            central_bytes,
            central_offset,
            comment_bytes,
        ) = struct.unpack_from("<4s4H2LH", payload, candidate)
        payload_end = candidate + 22 + comment_bytes
        if (
            disk_number == 0
            and central_disk == 0
            and disk_entries == total_entries
            and central_offset + central_bytes == candidate
            and payload_end <= len(payload)
            and zipfile.is_zipfile(io.BytesIO(payload[:payload_end]))
        ):
            return payload_end
    return None


def _read_stored_zip(
    payload: bytes,
    *,
    owner: str,
) -> tuple[tuple[tuple[zipfile.ZipInfo, bytes], ...], bytes]:
    payload_end = _zip_payload_end(payload)
    if payload_end is None:
        raise MixedCoderReceiverError(f"expected a stored ZIP at {owner}")
    stream = io.BytesIO(payload[:payload_end])
    try:
        with zipfile.ZipFile(stream, "r") as archive:
            rows: list[tuple[zipfile.ZipInfo, bytes]] = []
            names: set[str] = set()
            for info in archive.infolist():
                name = _safe_member_name(info.filename)
                if name in names:
                    raise MixedCoderReceiverError(f"duplicate recursive ZIP member: {owner}!/{name}")
                names.add(name)
                mode = info.external_attr >> 16
                if info.is_dir() or stat.S_ISLNK(mode):
                    raise MixedCoderReceiverError(f"directory/symlink is not admitted: {owner}!/{name}")
                if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1:
                    raise MixedCoderReceiverError(f"member is compressed or encrypted: {owner}!/{name}")
                rows.append((copy.copy(info), archive.read(info)))
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise MixedCoderReceiverError(f"invalid recursive ZIP at {owner}") from exc
    return tuple(rows), payload[payload_end:]


def _write_stored_zip(
    rows: tuple[tuple[zipfile.ZipInfo, bytes], ...],
    *,
    suffix: bytes = b"",
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_STORED,
        strict_timestamps=True,
    ) as archive:
        archive.comment = b""
        for source_info, member in rows:
            info = copy.copy(source_info)
            info.compress_type = zipfile.ZIP_STORED
            info.CRC = 0
            info.compress_size = 0
            info.file_size = 0
            archive.writestr(info, member, compress_type=zipfile.ZIP_STORED)
    return output.getvalue() + bytes(suffix)


def _canonical_composition_archive(members: Mapping[str, bytes]) -> bytes:
    if tuple(members) != EXPECTED_COMPOSITION_MEMBERS:
        raise MixedCoderReceiverError("composition members are incomplete or reordered")
    rows: list[tuple[zipfile.ZipInfo, bytes]] = []
    for name in EXPECTED_COMPOSITION_MEMBERS:
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o100644 << 16
        info.create_system = 3
        info.extra = b""
        info.comment = b""
        rows.append((info, bytes(members[name])))
    return _write_stored_zip(tuple(rows))


def _decode_context_frame(frame: bytes) -> tuple[bytes, str]:
    if len(frame) < 6 or not frame.startswith(DCC3_MAGIC):
        raise MixedCoderReceiverError("selected context frame lacks the DCC3 header")
    codec = frame[5]
    try:
        if codec == G4_CODEC_ID:
            return decode_g4_decoder_context(frame), "G4_FREE_DECODER_CONTEXT"
        if codec == BELLARD_CODEC_ID:
            return decode_bellard_class_mixing(frame), "BELLARD_CLASS_MIXING"
    except ValueError as exc:
        raise MixedCoderReceiverError("DCC3 frame failed exact decode/re-encode") from exc
    raise MixedCoderReceiverError(f"unsupported DCC3 codec id: {codec}")


def _selected_rows(price_table: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if price_table.get("price_table_schema") != PRICE_TABLE_SCHEMA:
        raise MixedCoderReceiverError("CC2 price-table schema differs")
    rows = price_table.get("rows")
    if not isinstance(rows, list) or len(rows) != 27:
        raise MixedCoderReceiverError("CC2 price table must contain exactly 27 physical leaves")
    selected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("stream_id"), str):
            raise MixedCoderReceiverError("CC2 price-table row is malformed")
        stream_id = row["stream_id"]
        if stream_id in selected:
            raise MixedCoderReceiverError(f"duplicate CC2 selected stream: {stream_id}")
        if int(row.get("delta_bytes", 0)) < 0:
            if row.get("selected_codec") not in CODEC_ENCODERS:
                raise MixedCoderReceiverError(f"negative row selected an unsupported codec: {stream_id}")
            selected[stream_id] = row
    if len(selected) != 8:
        raise MixedCoderReceiverError(f"CC2 selected leaf count differs: {len(selected)}")
    counts = {codec: sum(row["selected_codec"] == codec for row in selected.values()) for codec in CODEC_ENCODERS}
    if counts != {
        "G4_FREE_DECODER_CONTEXT": 1,
        "BELLARD_CLASS_MIXING": 7,
    }:
        raise MixedCoderReceiverError(f"CC2 selected codec census differs: {counts}")
    return selected


def build_mixed_archive(
    source_archive: bytes,
    price_table: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    """Replace exactly the eight negative CC2 leaves and prove exact restoration."""

    source = bytes(source_archive)
    if price_table.get("composition_archive_bytes") != len(source) or price_table.get(
        "composition_archive_sha256"
    ) != _sha256(source):
        raise MixedCoderReceiverError("CC2 price table is not bound to the source archive")
    parse_counted_composition_archive(source)
    selected = _selected_rows(price_table)
    all_rows = {str(row["stream_id"]): row for row in price_table["rows"]}
    visited: set[str] = set()
    replacement_rows: list[dict[str, Any]] = []

    def rewrite(blob: bytes, owner: str) -> bytes:
        if _zip_payload_end(blob) is not None:
            children, suffix = _read_stored_zip(blob, owner=owner)
            rewritten = tuple(
                (
                    info,
                    rewrite(member, f"{owner}!/{_safe_member_name(info.filename)}"),
                )
                for info, member in children
            )
            return _write_stored_zip(rewritten, suffix=suffix)
        row = all_rows.get(owner)
        if row is None:
            raise MixedCoderReceiverError(f"physical leaf is absent from CC2 price table: {owner}")
        if owner in visited:
            raise MixedCoderReceiverError(f"physical leaf was visited twice: {owner}")
        visited.add(owner)
        if row.get("current_bytes") != len(blob) or row.get("current_sha256") != _sha256(blob):
            raise MixedCoderReceiverError(f"CC2 physical leaf custody differs: {owner}")
        if owner not in selected:
            return blob
        codec = str(row["selected_codec"])
        frame = CODEC_ENCODERS[codec](blob)
        arm = next(
            (candidate for candidate in row.get("arms", []) if candidate.get("codec") == codec),
            None,
        )
        if (
            not isinstance(arm, dict)
            or arm.get("frame_sha256") != _sha256(frame)
            or row.get("selected_framed_bytes") != len(frame)
            or int(row["delta_bytes"]) != len(frame) - len(blob)
        ):
            raise MixedCoderReceiverError(f"selected CC2 frame receipt differs: {owner}")
        decoded, decoded_codec = _decode_context_frame(frame)
        if decoded != blob or decoded_codec != codec:
            raise MixedCoderReceiverError(f"selected CC2 frame parse-back differs: {owner}")
        replacement_rows.append(
            {
                "codec": codec,
                "delta_bytes": len(frame) - len(blob),
                "frame_bytes": len(frame),
                "frame_sha256": _sha256(frame),
                "raw_bytes": len(blob),
                "raw_sha256": _sha256(blob),
                "stream_id": owner,
            }
        )
        return frame

    mixed = rewrite(source, COMPOSITION_OWNER)
    if visited != set(all_rows):
        raise MixedCoderReceiverError("CC2 price table and recursive physical leaves are not bijective")
    restored, restoration = restore_mixed_archive(mixed)
    if restored != source:
        raise MixedCoderReceiverError("mixed archive did not restore the exact source archive")
    if int(price_table.get("selected_total_archive_estimate_bytes", -1)) != len(mixed):
        raise MixedCoderReceiverError("CC2 selected total estimate differs from exact mixed archive")
    if int(price_table.get("selected_total_delta_bytes", 0)) != len(mixed) - len(source):
        raise MixedCoderReceiverError("CC2 selected total delta differs from exact mixed archive")
    replacement_rows.sort(key=lambda row: str(row["stream_id"]))
    return mixed, {
        "schema": SCHEMA,
        "source_archive": {
            "bytes": len(source),
            "sha256": _sha256(source),
        },
        "mixed_archive": {
            "bytes": len(mixed),
            "sha256": _sha256(mixed),
        },
        "delta_bytes": len(mixed) - len(source),
        "selected_leaf_count": len(replacement_rows),
        "raw_leaf_count": len(all_rows) - len(replacement_rows),
        "replacement_rows": replacement_rows,
        "restoration": restoration,
        "member_identity_preserved": True,
        "section_order_preserved": True,
        "score_claim": False,
    }


def restore_mixed_archive(mixed_archive: bytes) -> tuple[bytes, dict[str, Any]]:
    """Restore every DCC3 frame and require one exact canonical PC1 archive."""

    mixed = bytes(mixed_archive)
    decoded_rows: list[dict[str, Any]] = []
    leaf_count = 0

    def restore(blob: bytes, owner: str) -> bytes:
        nonlocal leaf_count
        if _zip_payload_end(blob) is not None:
            children, suffix = _read_stored_zip(blob, owner=owner)
            restored = tuple(
                (
                    info,
                    restore(member, f"{owner}!/{_safe_member_name(info.filename)}"),
                )
                for info, member in children
            )
            return _write_stored_zip(restored, suffix=suffix)
        leaf_count += 1
        if not blob.startswith(DCC3_MAGIC):
            return blob
        raw, codec = _decode_context_frame(blob)
        decoded_rows.append(
            {
                "codec": codec,
                "frame_bytes": len(blob),
                "frame_sha256": _sha256(blob),
                "raw_bytes": len(raw),
                "raw_sha256": _sha256(raw),
                "stream_id": owner,
            }
        )
        return raw

    source = restore(mixed, COMPOSITION_OWNER)
    parent, packet, manifest = parse_counted_composition_archive(source)
    codec_counts = {codec: sum(row["codec"] == codec for row in decoded_rows) for codec in CODEC_ENCODERS}
    decoded_rows.sort(key=lambda row: str(row["stream_id"]))
    return source, {
        "schema": "ddm_cc3_mixed_coder_restoration.v1",
        "mixed_archive": {
            "bytes": len(mixed),
            "sha256": _sha256(mixed),
        },
        "source_archive": {
            "bytes": len(source),
            "sha256": _sha256(source),
        },
        "parent_archive": {
            "bytes": len(parent),
            "sha256": _sha256(parent),
        },
        "packet": {
            "active": packet.active,
            "bytes": len(_composition_members(source)[PACKET_MEMBER]),
            "sha256": manifest["packet_sha256"],
        },
        "physical_leaf_count": leaf_count,
        "decoded_frame_count": len(decoded_rows),
        "codec_counts": codec_counts,
        "decoded_rows": decoded_rows,
        "exact_pc1_parse_reemit": True,
        "all_frame_hashes_and_final_bytes_consumed": True,
    }


def _composition_members(archive: bytes) -> dict[str, bytes]:
    rows, suffix = _read_stored_zip(archive, owner=COMPOSITION_OWNER)
    if suffix:
        raise MixedCoderReceiverError("composition archive has an unowned trailing suffix")
    if tuple(info.filename for info, _ in rows) != EXPECTED_COMPOSITION_MEMBERS:
        raise MixedCoderReceiverError("composition member identity or order differs")
    return {info.filename: member for info, member in rows}


def restore_extracted_composition(
    members: Mapping[str, bytes],
) -> tuple[bytes, bytes, Any, dict[str, Any]]:
    """Runtime bridge from contest-extracted members to exact source state."""

    mixed = _canonical_composition_archive(members)
    source, receipt = restore_mixed_archive(mixed)
    if _composition_members(mixed) != dict(members):
        raise MixedCoderReceiverError("extracted mixed members do not re-emit canonically")
    parent, packet, _manifest = parse_counted_composition_archive(source)
    return source, parent, packet, receipt


__all__ = [
    "COMPOSITION_OWNER",
    "DCC3_MAGIC",
    "EXPECTED_COMPOSITION_MEMBERS",
    "SCHEMA",
    "MixedCoderReceiverError",
    "build_mixed_archive",
    "restore_extracted_composition",
    "restore_mixed_archive",
]
