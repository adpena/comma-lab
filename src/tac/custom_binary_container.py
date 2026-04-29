"""PACT_BC_v1 custom binary container for archive byte analysis.

Byte budget analysis
--------------------
Lane A's reference archive is 694,045 bytes. Its three member payloads
occupy 693,717 compressed bytes, leaving only 328 bytes of ZIP metadata
and central-directory structure. That means outer-container rewrites
cannot plausibly save 50 KB on this archive: the non-payload budget is
only 0.05% of the file.

The realistic paths are payload changes or better member compression.
Subagent L's archive_diet_pack lane already captured the meaningful
headroom by Brotli-compressing renderer.bin and deterministic ZIP
metadata, saving 14.7 KB on Lane A. This module is intentionally a
lossless container experiment. Expected savings over that diet-pack
baseline are 0-500 bytes, not 50 KB. Carmack's custom-container idea is
directionally useful, but the quoted 50 KB number is unachievable on the
current Lane A starting state without changing payload bytes or changing
the contest's unzip-based outer wrapper.

Format
------
PACT_BC_v1 bytes are:

* 16-byte magic: ``b"PACT_BC_v1\\0\\0\\0\\0\\0\\0"``
* zero or more entries:
  * uint32 big-endian UTF-8 name length
  * uint32 big-endian payload length
  * name bytes
  * raw payload bytes
* uint32 big-endian CRC32 trailer over the magic plus all entries

The container has no compression. Compression belongs in archive-diet
lanes so byte accounting remains explainable.
"""

from __future__ import annotations

import io
import struct
import zipfile
import zlib
from pathlib import PurePosixPath

PACT_MAGIC = b"PACT_BC_v1" + (b"\0" * 6)
_U32 = struct.Struct(">I")
_ENTRY_HEADER = struct.Struct(">II")
_FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ZIP_STORED_NAMES = {"masks.mkv"}
_ZIP_DEFLATED_NAMES = {"renderer.bin", "optimized_poses.pt"}


def pack_archive(file_dict: dict[str, bytes]) -> bytes:
    """Pack a mapping of archive member names to raw bytes into PACT_BC_v1.

    Entries are emitted in lexicographic filename order so repeated calls
    with the same logical member set produce identical bytes. Names are
    validated as safe relative POSIX paths because these bytes may later
    be unpacked into a filesystem-facing archive directory.
    """
    body = bytearray(PACT_MAGIC)
    for name, payload in sorted(file_dict.items()):
        _validate_member_name(name)
        payload_bytes = _coerce_payload(payload, name)
        name_bytes = name.encode("utf-8")
        _check_u32_len(len(name_bytes), f"name too long: {name!r}")
        _check_u32_len(len(payload_bytes), f"payload too large: {name!r}")
        body.extend(_ENTRY_HEADER.pack(len(name_bytes), len(payload_bytes)))
        body.extend(name_bytes)
        body.extend(payload_bytes)

    crc = zlib.crc32(body) & 0xFFFFFFFF
    body.extend(_U32.pack(crc))
    return bytes(body)


def unpack_archive(data: bytes) -> dict[str, bytes]:
    """Unpack PACT_BC_v1 bytes and return a byte-exact member dict.

    Raises:
        ValueError: if the data has bad magic, truncation, invalid UTF-8,
            duplicate names, unsafe names, or a CRC32 mismatch.
    """
    data = bytes(data)
    if len(data) < len(PACT_MAGIC) + _U32.size:
        raise ValueError("truncated header")

    body = data[:-_U32.size]
    if not body.startswith(PACT_MAGIC):
        raise ValueError("bad magic")

    expected_crc = _U32.unpack_from(data, len(body))[0]
    actual_crc = zlib.crc32(body) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError(f"CRC mismatch: expected 0x{expected_crc:08x}, got 0x{actual_crc:08x}")

    members: dict[str, bytes] = {}
    cursor = len(PACT_MAGIC)
    while cursor < len(body):
        remaining = len(body) - cursor
        if remaining < _ENTRY_HEADER.size:
            raise ValueError("truncated entry header")

        name_len, payload_len = _ENTRY_HEADER.unpack_from(body, cursor)
        cursor += _ENTRY_HEADER.size
        name_end = cursor + name_len
        payload_end = name_end + payload_len
        if name_end > len(body) or payload_end > len(body):
            raise ValueError("truncated entry payload")

        name_bytes = body[cursor:name_end]
        try:
            name = name_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-8 member name") from exc
        _validate_member_name(name)
        if name in members:
            raise ValueError(f"duplicate member name: {name!r}")

        members[name] = bytes(body[name_end:payload_end])
        cursor = payload_end

    return members


def pack_archive_to_zip_compatible(file_dict: dict[str, bytes]) -> bytes:
    """Wrap PACT_BC_v1 bytes as ``archive.pact`` in a valid deterministic ZIP.

    Known limitation: this is ZIP-compatible but not contest-compatible
    with the current evaluator. ``upstream/evaluate.sh`` runs
    ``unzip -o "$ARCHIVE_ZIP" -d "$ARCHIVE_DIR"`` and then calls the
    submission's ``inflate.sh``. Today's inflate scripts expect flat
    files such as ``renderer.bin``, ``masks.mkv``, and
    ``optimized_poses.pt`` at the top level. A single ``archive.pact``
    member would require a future inflate.sh evolution or a non-contest
    experiment harness that knows how to unpack it.
    """
    pact_bytes = pack_archive(file_dict)
    return _write_zip_bytes([("archive.pact", pact_bytes, zipfile.ZIP_STORED)])


def pack_archive_minimal_zip(file_dict: dict[str, bytes]) -> bytes:
    """Emit a minimal deterministic ZIP suitable for current contest unzip.

    This keeps the current flat member layout. It stores already-compressed
    ``masks.mkv`` bytes and DEFLATE-compresses likely-compressible
    ``renderer.bin`` and ``optimized_poses.pt`` with compresslevel 9.
    Unknown members default to DEFLATE level 9.

    Honest expectation: compared with current zipfile defaults and
    Subagent L's diet-pack, this usually saves less than 1 KB and can be
    larger if the supposedly incompressible ``masks.mkv`` still benefits
    from DEFLATE. It is the only helper here that can produce a
    contest-shaped ``archive.zip`` without changing inflate.sh.
    """
    members: list[tuple[str, bytes, int]] = []
    for name, payload in sorted(file_dict.items()):
        _validate_member_name(name)
        payload_bytes = _coerce_payload(payload, name)
        compress_type = zipfile.ZIP_STORED if name in _ZIP_STORED_NAMES else zipfile.ZIP_DEFLATED
        if name in _ZIP_DEFLATED_NAMES:
            compress_type = zipfile.ZIP_DEFLATED
        members.append((name, payload_bytes, compress_type))
    return _write_zip_bytes(members)


def _write_zip_bytes(members: list[tuple[str, bytes, int]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for name, payload, compress_type in members:
            info = zipfile.ZipInfo(filename=name, date_time=_FIXED_ZIP_TIMESTAMP)
            info.compress_type = compress_type
            info.create_system = 3
            info.external_attr = 0o600 << 16
            info.extra = b""
            info.comment = b""
            if compress_type == zipfile.ZIP_DEFLATED:
                zf.writestr(info, payload, compresslevel=9)
            else:
                zf.writestr(info, payload)
    return buf.getvalue()


def _coerce_payload(payload: bytes, name: str) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"payload for {name!r} must be bytes-like")
    return bytes(payload)


def _check_u32_len(value: int, message: str) -> None:
    if value > 0xFFFFFFFF:
        raise ValueError(message)


def _validate_member_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("member name must be str")
    if not name or name.endswith("/"):
        raise ValueError(f"invalid archive member path: {name!r}")
    if name.startswith("/") or "\\" in name or "\0" in name:
        raise ValueError(f"unsafe archive member path: {name!r}")
    parts = PurePosixPath(name).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe archive member path: {name!r}")


__all__ = [
    "PACT_MAGIC",
    "pack_archive",
    "pack_archive_minimal_zip",
    "pack_archive_to_zip_compatible",
    "unpack_archive",
]
