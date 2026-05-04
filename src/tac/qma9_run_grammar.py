"""Local QRG1 row-run grammar for PR85 QMA9 mask-token byte screens.

QRG1 is a planning-only format. It encodes the decoded PR85/QMA9 uint8 token
tensor as deterministic row commands, then compresses that command stream with
standard deterministic compressors. The helpers here are intentionally
score-agnostic and runtime-light: they prove local decode parity, but they do
not claim current contest inflate support.
"""
from __future__ import annotations

import bz2
import hashlib
import lzma
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES

QRG1_MAGIC = b"QRG1"
QRG1_HEADER_BYTES = 24
QRG1_CLASS_SYMBOLS = 5

ROW_OP_COPY_UP = 0
ROW_OP_COPY_PREV_FRAME = 1
ROW_OP_RUNS = 2


class QMA9RunGrammarError(ValueError):
    """Raised when a QRG1 payload or local screen contract is invalid."""


@dataclass(frozen=True)
class QRG1Mode:
    """One screened row-run grammar and body compressor."""

    mode_id: int
    name: str
    grammar: str
    compressor: str


@dataclass(frozen=True)
class QRG1Header:
    """Parsed QRG1 payload header."""

    magic: str
    frame_count: int
    width: int
    height: int
    mode_id: int
    mode_name: str
    body_bytes: int
    payload_bytes: int
    decoded_mask_bytes: int
    body_sha256: str
    payload_sha256: str


@dataclass(frozen=True)
class QRG1Encoded:
    """A locally encoded QRG1 payload plus row-run statistics."""

    mode: QRG1Mode
    payload: bytes
    command_bytes: int
    compressed_body_bytes: int
    stats: dict[str, int | float]


@dataclass(frozen=True)
class QRG1Decoded:
    """A locally decoded QRG1 token tensor."""

    header: QRG1Header
    data: bytes
    sha256: str
    stats: dict[str, int]


QRG1_MODES: tuple[QRG1Mode, ...] = (
    QRG1Mode(1, "row_rle_zlib9", "row_rle", "zlib9"),
    QRG1Mode(2, "row_rle_bz2_9", "row_rle", "bz2_9"),
    QRG1Mode(3, "row_rle_lzma6", "row_rle", "lzma6"),
    QRG1Mode(4, "row_copy_up_rle_zlib9", "row_copy_up_rle", "zlib9"),
    QRG1Mode(5, "row_copy_up_rle_bz2_9", "row_copy_up_rle", "bz2_9"),
    QRG1Mode(6, "row_copy_up_prev_rle_bz2_9", "row_copy_up_prev_rle", "bz2_9"),
    QRG1Mode(7, "row_copy_up_prev_rle_lzma6", "row_copy_up_prev_rle", "lzma6"),
)
DEFAULT_QRG1_MODE_NAMES: tuple[str, ...] = tuple(mode.name for mode in QRG1_MODES)

_MODES_BY_NAME = {mode.name: mode for mode in QRG1_MODES}
_MODES_BY_ID = {mode.mode_id: mode for mode in QRG1_MODES}


def sha256_bytes(data: bytes) -> str:
    """Return SHA-256 for in-memory bytes."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return SHA-256 for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_modes(value: str | Iterable[str]) -> tuple[QRG1Mode, ...]:
    """Parse mode names into a stable tuple of QRG1 modes."""

    if isinstance(value, str):
        names = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        names = tuple(str(part).strip() for part in value if str(part).strip())
    if not names:
        raise QMA9RunGrammarError("at least one QRG1 mode is required")
    missing = [name for name in names if name not in _MODES_BY_NAME]
    if missing:
        raise QMA9RunGrammarError(f"unknown QRG1 mode(s): {missing}")
    return tuple(_MODES_BY_NAME[name] for name in names)


def _validate_raw_tokens(
    raw_tokens: bytes | bytearray | memoryview,
    *,
    frame_count: int,
    width: int,
    height: int,
) -> bytes:
    raw = bytes(raw_tokens)
    frame_count = int(frame_count)
    width = int(width)
    height = int(height)
    if min(frame_count, width, height) <= 0:
        raise QMA9RunGrammarError("QRG1 dimensions must be positive")
    expected = frame_count * width * height
    if len(raw) != expected:
        raise QMA9RunGrammarError(f"raw token stream has {len(raw)} bytes, expected {expected}")
    bad = next((value for value in raw if value >= QRG1_CLASS_SYMBOLS), None)
    if bad is not None:
        raise QMA9RunGrammarError(f"raw token class out of range for QRG1: {bad}")
    return raw


def _put_varint(out: bytearray, value: int) -> None:
    if value < 0:
        raise QMA9RunGrammarError("QRG1 varint cannot encode negative values")
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def _read_varint(data: bytes, cursor: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while cursor < len(data):
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, cursor
        shift += 7
        if shift > 63:
            break
    raise QMA9RunGrammarError("truncated or overlong QRG1 varint")


def _compress(data: bytes, compressor: str) -> bytes:
    if compressor == "zlib9":
        return zlib.compress(data, 9)
    if compressor == "bz2_9":
        return bz2.compress(data, compresslevel=9)
    if compressor == "lzma6":
        return lzma.compress(data, preset=6)
    raise QMA9RunGrammarError(f"unknown QRG1 compressor: {compressor!r}")


def _decompress(data: bytes, compressor: str) -> bytes:
    if compressor == "zlib9":
        return zlib.decompress(data)
    if compressor == "bz2_9":
        return bz2.decompress(data)
    if compressor == "lzma6":
        return lzma.decompress(data)
    raise QMA9RunGrammarError(f"unknown QRG1 compressor: {compressor!r}")


def _emit_row_runs(out: bytearray, row: bytes) -> int:
    runs = 0
    x = 0
    width = len(row)
    while x < width:
        cls = row[x]
        end = x + 1
        while end < width and row[end] == cls:
            end += 1
        _put_varint(out, ((end - x - 1) << 3) | int(cls))
        runs += 1
        x = end
    return runs


def _encode_command_stream(
    raw: bytes,
    *,
    frame_count: int,
    width: int,
    height: int,
    mode: QRG1Mode,
) -> tuple[bytes, dict[str, int | float]]:
    frame_size = width * height
    out = bytearray()
    rows = 0
    rle_rows = 0
    copy_up_rows = 0
    copy_prev_frame_rows = 0
    one_run_rows = 0
    runs = 0
    run_pixels = 0
    for t in range(frame_count):
        frame_base = t * frame_size
        prev_frame_base = (t - 1) * frame_size
        for y in range(width):
            row_start = frame_base + y * height
            row = raw[row_start:row_start + height]
            rows += 1
            if mode.grammar in {"row_copy_up_rle", "row_copy_up_prev_rle"} and y > 0:
                up = raw[row_start - height:row_start]
                if row == up:
                    out.append(ROW_OP_COPY_UP)
                    copy_up_rows += 1
                    continue
            if mode.grammar == "row_copy_up_prev_rle" and t > 0:
                prev = raw[prev_frame_base + y * height:prev_frame_base + (y + 1) * height]
                if row == prev:
                    out.append(ROW_OP_COPY_PREV_FRAME)
                    copy_prev_frame_rows += 1
                    continue
            out.append(ROW_OP_RUNS)
            before_runs = runs
            emitted = _emit_row_runs(out, row)
            runs += emitted
            run_pixels += height
            rle_rows += 1
            if runs - before_runs == 1:
                one_run_rows += 1
    stats: dict[str, int | float] = {
        "rows": rows,
        "rle_rows": rle_rows,
        "copy_up_rows": copy_up_rows,
        "copy_prev_frame_rows": copy_prev_frame_rows,
        "runs": runs,
        "one_run_rows": one_run_rows,
        "run_pixels": run_pixels,
        "copy_row_fraction": (copy_up_rows + copy_prev_frame_rows) / max(1, rows),
        "average_runs_per_rle_row": runs / max(1, rle_rows),
    }
    return bytes(out), stats


def encode_qrg1_run_grammar(
    raw_tokens: bytes | bytearray | memoryview,
    *,
    frame_count: int,
    width: int,
    height: int,
    mode: str | QRG1Mode,
) -> QRG1Encoded:
    """Encode raw QMA9 tokens as a local-only QRG1 row-run payload."""

    raw = _validate_raw_tokens(
        raw_tokens,
        frame_count=frame_count,
        width=width,
        height=height,
    )
    qrg_mode = _MODES_BY_NAME[mode] if isinstance(mode, str) else mode
    commands, stats = _encode_command_stream(
        raw,
        frame_count=int(frame_count),
        width=int(width),
        height=int(height),
        mode=qrg_mode,
    )
    body = _compress(commands, qrg_mode.compressor)
    header = struct.pack(
        "<4sIIIII",
        QRG1_MAGIC,
        int(frame_count),
        int(width),
        int(height),
        int(qrg_mode.mode_id),
        len(body),
    )
    payload = header + body
    return QRG1Encoded(
        mode=qrg_mode,
        payload=payload,
        command_bytes=len(commands),
        compressed_body_bytes=len(body),
        stats={
            **stats,
            "command_bytes": len(commands),
            "compressed_body_bytes": len(body),
            "payload_bytes": len(payload),
        },
    )


def parse_qrg1_header(payload: bytes) -> QRG1Header:
    """Parse and validate a QRG1 header."""

    if len(payload) < QRG1_HEADER_BYTES:
        raise QMA9RunGrammarError("QRG1 payload is shorter than its 24-byte header")
    magic, frame_count, width, height, mode_id, body_bytes = struct.unpack_from("<4sIIIII", payload, 0)
    if magic != QRG1_MAGIC:
        raise QMA9RunGrammarError(f"expected QRG1 magic, got {magic!r}")
    mode = _MODES_BY_ID.get(int(mode_id))
    if mode is None:
        raise QMA9RunGrammarError(f"unknown QRG1 mode id: {mode_id}")
    if min(int(frame_count), int(width), int(height)) <= 0:
        raise QMA9RunGrammarError("QRG1 dimensions must be positive")
    packed_bytes = QRG1_HEADER_BYTES + int(body_bytes)
    if packed_bytes != len(payload):
        raise QMA9RunGrammarError(
            f"QRG1 body length mismatch: declared={packed_bytes} actual={len(payload)}"
        )
    body = payload[QRG1_HEADER_BYTES:packed_bytes]
    return QRG1Header(
        magic=magic.decode("ascii"),
        frame_count=int(frame_count),
        width=int(width),
        height=int(height),
        mode_id=int(mode_id),
        mode_name=mode.name,
        body_bytes=int(body_bytes),
        payload_bytes=packed_bytes,
        decoded_mask_bytes=int(frame_count) * int(width) * int(height),
        body_sha256=sha256_bytes(body),
        payload_sha256=sha256_bytes(payload[:packed_bytes]),
    )


def _decode_row_runs(commands: bytes, cursor: int, out: bytearray, row_start: int, height: int) -> int:
    filled = 0
    while filled < height:
        packed, cursor = _read_varint(commands, cursor)
        cls = packed & 0x7
        run_len = (packed >> 3) + 1
        if cls >= QRG1_CLASS_SYMBOLS:
            raise QMA9RunGrammarError(f"QRG1 run class out of range: {cls}")
        if filled + run_len > height:
            raise QMA9RunGrammarError("QRG1 row run overruns row width")
        out[row_start + filled:row_start + filled + run_len] = bytes([cls]) * run_len
        filled += run_len
    return cursor


def decode_qrg1_run_grammar(payload: bytes) -> QRG1Decoded:
    """Decode a local-only QRG1 row-run payload back to raw mask tokens."""

    header = parse_qrg1_header(payload)
    mode = _MODES_BY_ID[header.mode_id]
    commands = _decompress(payload[QRG1_HEADER_BYTES:header.payload_bytes], mode.compressor)
    frame_size = header.width * header.height
    out = bytearray(header.decoded_mask_bytes)
    cursor = 0
    stats = {
        "rows": 0,
        "rle_rows": 0,
        "copy_up_rows": 0,
        "copy_prev_frame_rows": 0,
    }
    for t in range(header.frame_count):
        frame_base = t * frame_size
        prev_frame_base = (t - 1) * frame_size
        for y in range(header.width):
            if cursor >= len(commands):
                raise QMA9RunGrammarError("QRG1 command stream ended before all rows decoded")
            op = commands[cursor]
            cursor += 1
            row_start = frame_base + y * header.height
            if op == ROW_OP_COPY_UP:
                if y == 0:
                    raise QMA9RunGrammarError("QRG1 COPY_UP appears on first row")
                out[row_start:row_start + header.height] = out[row_start - header.height:row_start]
                stats["copy_up_rows"] += 1
            elif op == ROW_OP_COPY_PREV_FRAME:
                if t == 0:
                    raise QMA9RunGrammarError("QRG1 COPY_PREV_FRAME appears in first frame")
                prev_start = prev_frame_base + y * header.height
                out[row_start:row_start + header.height] = out[prev_start:prev_start + header.height]
                stats["copy_prev_frame_rows"] += 1
            elif op == ROW_OP_RUNS:
                cursor = _decode_row_runs(commands, cursor, out, row_start, header.height)
                stats["rle_rows"] += 1
            else:
                raise QMA9RunGrammarError(f"unknown QRG1 row op: {op}")
            stats["rows"] += 1
    if cursor != len(commands):
        raise QMA9RunGrammarError(
            f"QRG1 command stream has trailing bytes: consumed={cursor} total={len(commands)}"
        )
    data = bytes(out)
    return QRG1Decoded(
        header=header,
        data=data,
        sha256=sha256_bytes(data),
        stats=stats,
    )


def qrg1_runtime_custody_contract() -> dict[str, Any]:
    """Return the fail-closed runtime contract for QRG1 candidates."""

    return {
        "live_runtime_supported": False,
        "dispatch_unlocked": False,
        "safe_for_remote_dispatch": False,
        "current_live_runtime_contract": {
            "accepted_mask_magic": "QMA9",
            "accepted_mode": "adaptive9bin range_mask_codec.cpp path",
            "qrg1_supported": False,
        },
        "required_runtime_changes_before_dispatch": [
            "add a reviewed QRG1 row-run decoder to the robust_current inflate path",
            "extend mask payload detection to admit QRG1 without changing PR85 fixed-slice custody",
            "preserve PR85 QMA9 token storage order, 600x512x384 source shape, transpose behavior, and _half_frame_only metadata",
            "add raw-token SHA parity tests against the exact PR85 token source before any exact eval",
            "add runtime output parity against the baseline PR85 inflate masks tensor before any exact eval",
            "record the updated inflate runtime tree SHA in contest_auth_eval provenance",
            "open a fresh Level-2 dispatch claim before any later CUDA exact-eval run",
        ],
    }
