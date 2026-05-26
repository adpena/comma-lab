#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generated packet-member merge receiver runtime."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

JSON_MAGIC = b"TAC_PACKET_MEMBER_MERGE_V1\0"
BINARY_MAGIC = b"TAC_PACKET_MEMBER_MERGE_BIN1\0"
DEFLATE_SEQUENCE_MAGIC = b"TAC_PACKET_MEMBER_MERGE_DFL1\0"


def _parse(payload: bytes) -> tuple[dict, dict[str, bytes]]:
    if payload.startswith(DEFLATE_SEQUENCE_MAGIC):
        return _parse_deflate_sequence(payload)
    if payload.startswith(BINARY_MAGIC):
        return _parse_binary(payload)
    if payload.startswith(JSON_MAGIC):
        return _parse_json(payload)
    raise RuntimeError("merged member payload has bad magic")


def _parse_json(payload: bytes) -> tuple[dict, dict[str, bytes]]:
    if not payload.startswith(JSON_MAGIC):
        raise RuntimeError("merged member payload has bad magic")
    if len(payload) < len(JSON_MAGIC) + 8:
        raise RuntimeError("merged member payload is truncated")
    table_len = struct.unpack_from("<Q", payload, len(JSON_MAGIC))[0]
    table_start = len(JSON_MAGIC) + 8
    table_end = table_start + int(table_len)
    table = json.loads(payload[table_start:table_end].decode("utf-8"))
    concatenated = payload[table_end:]
    members = {}
    codec = str(table.get("payload_codec") or "raw_member_payload_v1")
    for row in table.get("members") or []:
        name = str(row["name"])
        offset = int(row["offset"])
        length = int(row["length"])
        encoded = concatenated[offset: offset + length]
        if codec == "raw_member_payload_v1":
            members[name] = encoded
        elif codec in {
            "source_zip_compressed_stream_v1",
            "source_zip_compressed_stream_binary_table_v1",
        }:
            members[name] = _decompress_zip_member(
                encoded,
                int(row["zip_compress_type"]),
                name,
            )
        else:
            raise RuntimeError(f"unsupported packet member merge payload codec: {codec}")
    return table, members


def _parse_deflate_sequence(payload: bytes) -> tuple[dict, dict[str, bytes]]:
    cursor = len(DEFLATE_SEQUENCE_MAGIC)
    count, cursor = _decode_uvarint(payload, cursor, "member count")
    names = []
    for _ in range(count):
        name_len, cursor = _decode_uvarint(payload, cursor, "name length")
        name_end = cursor + int(name_len)
        if name_end > len(payload):
            raise RuntimeError("deflate sequence merge table name extends past payload")
        names.append(payload[cursor:name_end].decode("utf-8"))
        cursor = name_end
    remaining = payload[cursor:]
    offset = 0
    members = {}
    table_rows = []
    for index, name in enumerate(names):
        decoded, consumed = _decompress_next_zip_deflate_stream(remaining, name)
        members[name] = decoded
        table_rows.append(
            {
                "name": name,
                "offset": offset,
                "length": consumed,
                "zip_compress_type": zipfile.ZIP_DEFLATED,
                "uncompressed_length": len(decoded),
            }
        )
        offset += consumed
        remaining = remaining[consumed:]
        if index == len(names) - 1 and remaining:
            raise RuntimeError("deflate sequence merge payload has trailing bytes")
    return (
        {
            "schema": "packet_member_merge_table.v1",
            "payload_codec": "fixed_order_raw_deflate_sequence_v1",
            "table_format": "uleb_name_raw_deflate_sequence_v1",
            "member_count": len(table_rows),
            "members": table_rows,
        },
        members,
    )


def _parse_binary(payload: bytes) -> tuple[dict, dict[str, bytes]]:
    cursor = len(BINARY_MAGIC)
    count, cursor = _decode_uvarint(payload, cursor, "member count")
    rows = []
    for _ in range(count):
        name_len, cursor = _decode_uvarint(payload, cursor, "name length")
        name_end = cursor + int(name_len)
        if name_end > len(payload):
            raise RuntimeError("binary merge table name extends past payload")
        name = payload[cursor:name_end].decode("utf-8")
        cursor = name_end
        compress_type, cursor = _decode_uvarint(payload, cursor, "compress type")
        compressed_length, cursor = _decode_uvarint(payload, cursor, "compressed length")
        uncompressed_length, cursor = _decode_uvarint(payload, cursor, "uncompressed length")
        rows.append(
            {
                "name": name,
                "zip_compress_type": int(compress_type),
                "length": int(compressed_length),
                "uncompressed_length": int(uncompressed_length),
            }
        )
    concatenated = payload[cursor:]
    offset = 0
    members = {}
    table_rows = []
    for row in rows:
        name = row["name"]
        length = int(row["length"])
        encoded = concatenated[offset: offset + length]
        if len(encoded) != length:
            raise RuntimeError(f"binary merge payload truncated for {name}")
        decoded = _decompress_zip_member(encoded, int(row["zip_compress_type"]), name)
        if len(decoded) != int(row["uncompressed_length"]):
            raise RuntimeError(f"binary merge payload length mismatch for {name}")
        members[name] = decoded
        table_rows.append(
            {
                "name": name,
                "offset": offset,
                "length": length,
                "zip_compress_type": int(row["zip_compress_type"]),
                "uncompressed_length": len(decoded),
            }
        )
        offset += length
    if offset != len(concatenated):
        raise RuntimeError("binary merge payload has trailing bytes")
    return (
        {
            "schema": "packet_member_merge_table.v1",
            "payload_codec": "source_zip_compressed_stream_binary_table_v1",
            "table_format": "uleb_name_compressed_stream_table_v1",
            "member_count": len(table_rows),
            "members": table_rows,
        },
        members,
    )


def _decompress_zip_member(payload: bytes, compress_type: int, name: str) -> bytes:
    if compress_type == zipfile.ZIP_STORED:
        return payload
    if compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(payload, -zlib.MAX_WBITS)
    raise RuntimeError(f"unsupported ZIP compression method for {name}: {compress_type}")


def _decompress_next_zip_deflate_stream(payload: bytes, name: str) -> tuple[bytes, int]:
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    decoded = decompressor.decompress(payload)
    decoded += decompressor.flush()
    if not decompressor.eof:
        raise RuntimeError(f"deflate stream did not terminate for {name}")
    consumed = len(payload) - len(decompressor.unused_data)
    if consumed <= 0:
        raise RuntimeError(f"deflate stream consumed no bytes for {name}")
    return decoded, consumed


def _write_member_file(root: Path, name: str, payload: bytes) -> None:
    destination = (root / name).resolve()
    root_resolved = root.resolve()
    if destination != root_resolved and root_resolved not in destination.parents:
        raise RuntimeError(f"refusing unsafe reconstructed member path: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def _write_zip_member(
    target: zipfile.ZipFile,
    name: str,
    payload: bytes,
) -> None:
    out = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    out.compress_type = zipfile.ZIP_STORED
    target.writestr(out, payload)


def _write_reconstructed_payload(
    target: zipfile.ZipFile,
    member_root: Path,
    payload: bytes,
    *,
    fallback_name: str,
) -> None:
    if (
        payload.startswith(JSON_MAGIC)
        or payload.startswith(BINARY_MAGIC)
        or payload.startswith(DEFLATE_SEQUENCE_MAGIC)
    ):
        table, members = _parse(payload)
        for row in table.get("members") or []:
            name = str(row["name"])
            member_payload = members[name]
            _write_zip_member(target, name, member_payload)
            _write_member_file(member_root, name, member_payload)
        return
    _write_zip_member(target, fallback_name, payload)
    _write_member_file(member_root, fallback_name, payload)


def _decode_uvarint(data: bytes, offset: int, label: str) -> tuple[int, int]:
    value = 0
    shift = 0
    cursor = offset
    while cursor < len(data):
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, cursor
        shift += 7
        if shift > 63:
            raise RuntimeError(f"{label} uvarint too wide")
    raise RuntimeError(f"{label} uvarint truncated")


def _expand_archive(source_archive: Path, output_archive: Path) -> None:
    member_root = output_archive.parent
    with zipfile.ZipFile(source_archive, "r") as source, zipfile.ZipFile(output_archive, "w") as target:
        for info in source.infolist():
            if info.is_dir():
                target.mkdir(info)
                (member_root / info.filename).mkdir(parents=True, exist_ok=True)
                continue
            payload = source.read(info.filename)
            _write_reconstructed_payload(
                target,
                member_root,
                payload,
                fallback_name=info.filename,
            )


def _expand_archive_dir(source_dir: Path, output_archive: Path) -> None:
    member_root = output_archive.parent
    with zipfile.ZipFile(output_archive, "w") as target:
        for path in sorted(source_dir.rglob("*"), key=lambda item: item.relative_to(source_dir).as_posix()):
            if path.is_dir():
                continue
            rel = path.relative_to(source_dir).as_posix()
            _write_reconstructed_payload(
                target,
                member_root,
                path.read_bytes(),
                fallback_name=rel,
            )


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: inflate.py <archive_dir> <output_dir> <file_list>")
    archive_dir, output_dir, file_list = map(Path, argv)
    source_archive = archive_dir / "archive.zip"
    here = Path(__file__).resolve().parent
    source_runtime = here / "source_runtime"
    with tempfile.TemporaryDirectory(prefix="packet-member-merge-receiver-") as tmp:
        shadow_dir = Path(tmp) / "archive"
        shadow_dir.mkdir()
        if source_archive.is_file():
            _expand_archive(source_archive, shadow_dir / "archive.zip")
        else:
            _expand_archive_dir(archive_dir, shadow_dir / "archive.zip")
        inflate_sh = source_runtime / "inflate.sh"
        inflate_py = source_runtime / "inflate.py"
        if inflate_sh.is_file():
            cmd = [str(inflate_sh), str(shadow_dir), str(output_dir), str(file_list)]
        elif inflate_py.is_file():
            cmd = [sys.executable, str(inflate_py), str(shadow_dir), str(output_dir), str(file_list)]
        else:
            raise SystemExit("source runtime has no inflate.sh or inflate.py")
        proc = subprocess.run(cmd, check=False)
        return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
