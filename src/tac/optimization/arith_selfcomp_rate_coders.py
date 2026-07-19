# SPDX-License-Identifier: MIT
"""Deterministic, fully framed coders for local rate measurements.

The repository arithmetic streams use :mod:`tac.lossless.range_coder` and
strictly decode without optional packages.  The optional constriction stream is
a different wire format: it is encoded and decoded only by constriction and is
never presented as parseable by the repository ``RangeDecoder``.

``spatial`` has one precise meaning in this module.  An input must have shape
``[..., H, W, C]`` (rank at least three), and each scalar is conditioned on the
already decoded left and upper scalar in the *same channel*.  Contexts include
the sign class and magnitude bucket of both neighbours.  Flat/sequential input
is represented by the separately named IID baseline, never by a spatial label.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import lzma
import shutil
import struct
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np

from tac.lossless.range_coder import RangeDecoder, RangeEncoder


class RateCoderError(ValueError):
    """Raised when a coder is unavailable or a frame is malformed."""


ARRAY_MAGIC: Final = b"ARC2"
ARRAY_VERSION: Final = 2
_ARRAY_PREFIX: Final = struct.Struct("<4sBBBBQ32s")
_U64: Final = struct.Struct("<Q")
_FRAME_LENGTHS: Final = struct.Struct("<QQ")
_MODEL_PREFIX: Final = struct.Struct("<HBB")
_CONTEXT_PREFIX: Final = struct.Struct("<H")
_GROUP_PREFIX: Final = struct.Struct("<HQI")
_DTYPE_CODES: dict[np.dtype[Any], int] = {
    np.dtype("int8"): 1,
    np.dtype("int16"): 2,
    np.dtype("int32"): 3,
    np.dtype("int64"): 4,
}
_CODE_DTYPES = {value: key for key, value in _DTYPE_CODES.items()}
_RAW_CODEC: Final = 0
_IID_CODEC: Final = 1
_SPATIAL_REPO_CODEC: Final = 2
_RLE_CODEC: Final = 3
_SPATIAL_CONSTRICTION_CODEC: Final = 4
_MODE_IID: Final = 0
_MODE_SPATIAL: Final = 1
_MAX_RANK: Final = 8
_SPATIAL_CONTEXTS: Final = 3 * 3 * 4 * 4


@dataclass(frozen=True)
class CodecMeasurement:
    """Measured bytes and exact parse-back status for one complete frame."""

    codec: str
    framed_bytes: int
    sha256: str
    parseback_exact: bool
    implementation_identity: str
    dependency_identity: str
    encoder_seconds: float
    decoder_seconds: float
    available: bool = True
    unavailable_reason: str | None = None
    framing_bytes: int | None = None
    model_table_bytes: int | None = None
    payload_bytes: int | None = None
    decoder_dependency_required: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ModelTable:
    mode: int
    width: int
    # context -> (sign frequencies [3], magnitude-bit frequencies [bits,2])
    rows: dict[int, tuple[np.ndarray, np.ndarray]]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _as_bytes(data: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise RateCoderError("codec input must be bytes-like")
    return bytes(data)


def _signed_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype not in _DTYPE_CODES:
        raise RateCoderError("signed array codec requires native int8/int16/int32/int64 dtype")
    if array.ndim > _MAX_RANK:
        raise RateCoderError(f"array rank must be at most {_MAX_RANK}")
    return np.ascontiguousarray(array)


def _spatial_array(value: np.ndarray) -> np.ndarray:
    array = _signed_array(value)
    if array.ndim < 3:
        raise RateCoderError("spatial context requires shape [..., H, W, C] with rank >= 3")
    return array


def _element_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dimension in shape:
        count *= int(dimension)
    return count


def _canonical_array_bytes(value: np.ndarray) -> bytes:
    array = _signed_array(value)
    little = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return little.tobytes(order="C")


def serialize_signed_array(value: np.ndarray) -> bytes:
    """Return strict endian-stable raw framing for a signed array."""

    array = _signed_array(value)
    raw = _canonical_array_bytes(array)
    prefix = _ARRAY_PREFIX.pack(
        ARRAY_MAGIC,
        ARRAY_VERSION,
        _RAW_CODEC,
        _DTYPE_CODES[array.dtype],
        array.ndim,
        array.size,
        hashlib.sha256(raw).digest(),
    )
    dimensions = b"".join(_U64.pack(int(dimension)) for dimension in array.shape)
    return prefix + dimensions + raw


def deserialize_signed_array(data: bytes | bytearray | memoryview) -> np.ndarray:
    """Strictly decode raw array framing, rejecting truncation and trailers."""

    frame = _as_bytes(data)
    dtype, shape, count, digest, offset = _parse_prefix(frame, _RAW_CODEC)
    expected = count * dtype.itemsize
    if len(frame) != offset + expected:
        relation = "truncated" if len(frame) < offset + expected else "trailing"
        raise RateCoderError(f"{relation} signed-array payload")
    payload = frame[offset:]
    if hashlib.sha256(payload).digest() != digest:
        raise RateCoderError("signed-array payload hash mismatch")
    return np.frombuffer(payload, dtype=dtype.newbyteorder("<")).reshape(shape).copy()


def encode_lzma(data: bytes | bytearray | memoryview) -> bytes:
    """Encode deterministic XZ/LZMA2 preset-9-extreme bytes."""

    return lzma.compress(_as_bytes(data), format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)


def decode_lzma(data: bytes | bytearray | memoryview) -> bytes:
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    try:
        decoded = decoder.decompress(_as_bytes(data))
    except lzma.LZMAError as exc:
        raise RateCoderError("invalid LZMA/XZ stream") from exc
    if not decoder.eof or decoder.unused_data:
        raise RateCoderError("truncated or trailing LZMA/XZ stream")
    return decoded


def _brotli_module() -> Any | None:
    try:
        import brotli
    except ImportError:
        return None
    return brotli


def encode_brotli_q11(data: bytes | bytearray | memoryview) -> bytes:
    brotli = _brotli_module()
    if brotli is None:
        raise RateCoderError("brotli dependency is unavailable")
    return bytes(brotli.compress(_as_bytes(data), quality=11))


def decode_brotli_q11(data: bytes | bytearray | memoryview) -> bytes:
    brotli = _brotli_module()
    if brotli is None:
        raise RateCoderError("brotli dependency is unavailable")
    decoder = brotli.Decompressor()
    try:
        decoded = bytes(decoder.process(_as_bytes(data)))
    except brotli.error as exc:
        raise RateCoderError("invalid Brotli-Q11 stream") from exc
    if not decoder.is_finished():
        raise RateCoderError("truncated Brotli-Q11 stream")
    return decoded


def _zstd_module() -> Any | None:
    try:
        import zstandard as zstd  # type: ignore[import-not-found]
    except ImportError:
        return None
    return zstd


def _zstd_dependency_identity() -> str:
    if _zstd_module() is not None:
        try:
            return f"optional:zstandard-{importlib.metadata.version('zstandard')}-level-19"
        except importlib.metadata.PackageNotFoundError:
            return "optional:zstandard-module-level-19"
    executable = shutil.which("zstd")
    return f"external:zstd-cli-level-19:{executable}" if executable else "unavailable:zstandard"


def encode_zstd_19(data: bytes | bytearray | memoryview) -> bytes:
    zstd = _zstd_module()
    if zstd is not None:
        return bytes(zstd.ZstdCompressor(level=19).compress(_as_bytes(data)))
    executable = shutil.which("zstd")
    if executable is None:
        raise RateCoderError("zstandard module and zstd CLI are unavailable")
    completed = subprocess.run(
        [executable, "-19", "--stdout", "--no-progress"],
        input=_as_bytes(data),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RateCoderError(f"zstd CLI encode failed with exit {completed.returncode}")
    return completed.stdout


def decode_zstd_19(data: bytes | bytearray | memoryview) -> bytes:
    zstd = _zstd_module()
    if zstd is not None:
        try:
            # allow_extra_data=False gives strict trailer rejection.
            return bytes(zstd.ZstdDecompressor().decompress(_as_bytes(data), allow_extra_data=False))
        except zstd.ZstdError as exc:
            raise RateCoderError("invalid, truncated, or trailing zstd stream") from exc
    executable = shutil.which("zstd")
    if executable is None:
        raise RateCoderError("zstandard module and zstd CLI are unavailable")
    completed = subprocess.run(
        [executable, "-d", "--stdout", "--no-progress"],
        input=_as_bytes(data),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RateCoderError("invalid, truncated, or trailing zstd CLI stream")
    return completed.stdout


def _sign_classes(values: np.ndarray) -> np.ndarray:
    return np.where(values == 0, 0, np.where(values > 0, 1, 2)).astype(np.uint8)


def _magnitude_buckets(values: np.ndarray) -> np.ndarray:
    # Avoid abs(INT_MIN) overflow by converting through object-free uint64 math.
    signed = values.astype(np.int64, copy=False)
    magnitude = np.where(signed < 0, -(signed + 1) + 1, signed).astype(np.uint64)
    return np.where(magnitude == 0, 0, np.where(magnitude == 1, 1, np.where(magnitude <= 3, 2, 3))).astype(np.uint8)


def _spatial_context_ids(array: np.ndarray) -> np.ndarray:
    """Vectorize left/up same-channel contexts in canonical raster order."""

    spatial = _spatial_array(array)
    batch = _element_count(tuple(int(value) for value in spatial.shape[:-3]))
    height, width, channels = (int(value) for value in spatial.shape[-3:])
    view = spatial.reshape(batch, height, width, channels)
    left = np.zeros_like(view)
    upper = np.zeros_like(view)
    if width > 1:
        left[:, :, 1:, :] = view[:, :, :-1, :]
    if height > 1:
        upper[:, 1:, :, :] = view[:, :-1, :, :]
    left_sign = _sign_classes(left)
    upper_sign = _sign_classes(upper)
    left_mag = _magnitude_buckets(left)
    upper_mag = _magnitude_buckets(upper)
    contexts = (((left_sign * 3 + upper_sign) * 4 + left_mag) * 4 + upper_mag).astype(np.uint16)
    return contexts.reshape(spatial.shape)


def _magnitude_uint64(values: np.ndarray) -> np.ndarray:
    signed = values.astype(np.int64, copy=False)
    return np.where(signed < 0, -(signed + 1) + 1, signed).astype(np.uint64)


def _build_model(array: np.ndarray, contexts: np.ndarray, *, mode: int) -> _ModelTable:
    flat_values = array.reshape(-1)
    flat_contexts = contexts.reshape(-1).astype(np.int64, copy=False)
    bits = array.dtype.itemsize * 8
    signs = _sign_classes(flat_values).astype(np.int64, copy=False)
    magnitudes = _magnitude_uint64(flat_values)
    rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    order = np.argsort(flat_contexts, kind="stable")
    sorted_contexts = flat_contexts[order]
    unique, starts, counts = np.unique(sorted_contexts, return_index=True, return_counts=True)
    for context, start, count in zip(unique.tolist(), starts.tolist(), counts.tolist(), strict=True):
        indices = order[start : start + count]
        sign_counts = np.bincount(signs[indices], minlength=3).astype(np.uint64) + 1
        bit_counts = np.ones((bits, 2), dtype=np.uint64)
        selected = magnitudes[indices]
        for bit_index in range(bits):
            symbols = ((selected >> (bits - 1 - bit_index)) & 1).astype(np.int64)
            bit_counts[bit_index] += np.bincount(symbols, minlength=2).astype(np.uint64)
        if np.any(sign_counts > np.iinfo(np.uint32).max) or np.any(bit_counts > np.iinfo(np.uint32).max):
            raise RateCoderError("model frequency exceeds uint32 framing")
        rows[int(context)] = (sign_counts.astype(np.uint32), bit_counts.astype(np.uint32))
    return _ModelTable(mode=mode, width=array.dtype.itemsize, rows=rows)


def _serialize_model(model: _ModelTable) -> bytes:
    chunks = [_MODEL_PREFIX.pack(len(model.rows), model.width, model.mode)]
    for context in sorted(model.rows):
        sign_counts, bit_counts = model.rows[context]
        chunks.append(_CONTEXT_PREFIX.pack(context))
        chunks.append(np.asarray(sign_counts, dtype="<u4").tobytes())
        chunks.append(np.asarray(bit_counts, dtype="<u4").tobytes())
    return b"".join(chunks)


def _parse_model(payload: bytes, *, expected_mode: int, expected_width: int) -> _ModelTable:
    if len(payload) < _MODEL_PREFIX.size:
        raise RateCoderError("truncated arithmetic model table")
    row_count, width, mode = _MODEL_PREFIX.unpack_from(payload)
    if width != expected_width or mode != expected_mode:
        raise RateCoderError("arithmetic model mode or width mismatch")
    row_bytes = _CONTEXT_PREFIX.size + 3 * 4 + width * 8 * 2 * 4
    if len(payload) != _MODEL_PREFIX.size + row_count * row_bytes:
        raise RateCoderError("truncated or trailing arithmetic model table")
    rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    offset = _MODEL_PREFIX.size
    for _ in range(row_count):
        (context,) = _CONTEXT_PREFIX.unpack_from(payload, offset)
        offset += _CONTEXT_PREFIX.size
        if context in rows or context >= _SPATIAL_CONTEXTS:
            raise RateCoderError("duplicate or invalid arithmetic context id")
        signs = np.frombuffer(payload, dtype="<u4", count=3, offset=offset).copy()
        offset += 3 * 4
        bit_counts = (
            np.frombuffer(payload, dtype="<u4", count=width * 8 * 2, offset=offset).reshape(width * 8, 2).copy()
        )
        offset += width * 8 * 2 * 4
        if np.any(signs == 0) or np.any(bit_counts == 0):
            raise RateCoderError("arithmetic frequencies must be positive")
        rows[int(context)] = (signs, bit_counts)
    if mode == _MODE_IID and set(rows) not in ({0}, set()):
        raise RateCoderError("IID arithmetic model must contain only context zero")
    return _ModelTable(mode=mode, width=width, rows=rows)


def _parse_prefix(frame: bytes, expected_codec: int) -> tuple[np.dtype[Any], tuple[int, ...], int, bytes, int]:
    if len(frame) < _ARRAY_PREFIX.size:
        raise RateCoderError("truncated arithmetic-array header")
    magic, version, codec, dtype_code, rank, count, digest = _ARRAY_PREFIX.unpack_from(frame)
    if magic != ARRAY_MAGIC or version != ARRAY_VERSION or codec != expected_codec:
        raise RateCoderError("invalid arithmetic-array header")
    dtype = _CODE_DTYPES.get(dtype_code)
    if dtype is None or rank > _MAX_RANK:
        raise RateCoderError("invalid signed-array dtype or rank")
    offset = _ARRAY_PREFIX.size
    if len(frame) < offset + rank * _U64.size:
        raise RateCoderError("truncated arithmetic-array dimensions")
    shape = tuple(int(_U64.unpack_from(frame, offset + index * _U64.size)[0]) for index in range(rank))
    offset += rank * _U64.size
    if _element_count(shape) != count:
        raise RateCoderError("arithmetic-array element count does not match shape")
    return dtype, shape, int(count), bytes(digest), offset


def _array_frame(array: np.ndarray, codec: int, model: bytes, stream: bytes) -> bytes:
    raw = _canonical_array_bytes(array)
    prefix = _ARRAY_PREFIX.pack(
        ARRAY_MAGIC,
        ARRAY_VERSION,
        codec,
        _DTYPE_CODES[array.dtype],
        array.ndim,
        array.size,
        hashlib.sha256(raw).digest(),
    )
    dims = b"".join(_U64.pack(int(dimension)) for dimension in array.shape)
    return prefix + dims + _FRAME_LENGTHS.pack(len(model), len(stream)) + model + stream


def _parse_array_frame(
    data: bytes | bytearray | memoryview, expected_codec: int
) -> tuple[np.dtype[Any], tuple[int, ...], int, bytes, bytes, bytes, int]:
    frame = _as_bytes(data)
    dtype, shape, count, digest, offset = _parse_prefix(frame, expected_codec)
    if len(frame) < offset + _FRAME_LENGTHS.size:
        raise RateCoderError("truncated arithmetic-array lengths")
    model_size, stream_size = _FRAME_LENGTHS.unpack_from(frame, offset)
    offset += _FRAME_LENGTHS.size
    expected = offset + int(model_size) + int(stream_size)
    if len(frame) != expected:
        relation = "truncated" if len(frame) < expected else "trailing"
        raise RateCoderError(f"{relation} arithmetic-array stream")
    model = frame[offset : offset + model_size]
    stream = frame[offset + model_size :]
    return dtype, shape, count, digest, model, stream, offset


def _cumulative(frequencies: np.ndarray) -> list[int]:
    return np.concatenate(([0], np.cumsum(frequencies, dtype=np.uint64))).astype(np.uint64).tolist()


def _decode_symbol(decoder: RangeDecoder, cumulative: list[int]) -> int:
    total = int(cumulative[-1])
    target = decoder.target(total)
    symbol = int(np.searchsorted(cumulative, target, side="right") - 1)
    if symbol < 0 or symbol >= len(cumulative) - 1:
        raise RateCoderError("range symbol falls outside model")
    decoder.update(low_count=int(cumulative[symbol]), high_count=int(cumulative[symbol + 1]), total=total)
    return symbol


def _encode_repository(array: np.ndarray, contexts: np.ndarray, *, codec: int, mode: int) -> bytes:
    model = _build_model(array, contexts, mode=mode)
    cumulative_rows = {
        context: (_cumulative(signs), tuple(_cumulative(row) for row in bits))
        for context, (signs, bits) in model.rows.items()
    }
    encoder = RangeEncoder()
    bits = array.dtype.itemsize * 8
    values = array.reshape(-1)
    context_values = contexts.reshape(-1)
    signs = _sign_classes(values)
    magnitudes = _magnitude_uint64(values)
    for value_index in range(values.size):
        sign_cumulative, bit_cumulative = cumulative_rows[int(context_values[value_index])]
        sign = int(signs[value_index])
        encoder.encode(symbol=sign, cumulative=sign_cumulative, total=sign_cumulative[-1])
        magnitude = int(magnitudes[value_index])
        for bit_index in range(bits):
            symbol = (magnitude >> (bits - 1 - bit_index)) & 1
            cumulative = bit_cumulative[bit_index]
            encoder.encode(symbol=symbol, cumulative=cumulative, total=cumulative[-1])
    return _array_frame(array, codec, _serialize_model(model), encoder.finish())


def encode_iid_arithmetic(value: np.ndarray) -> bytes:
    """Encode an array with one IID sign/magnitude model and repository coder."""

    array = _signed_array(value)
    contexts = np.zeros(array.shape, dtype=np.uint16)
    return _encode_repository(array, contexts, codec=_IID_CODEC, mode=_MODE_IID)


def encode_spatial_context_arithmetic(value: np.ndarray) -> bytes:
    """Encode true left/up same-channel sign/magnitude spatial contexts."""

    array = _spatial_array(value)
    return _encode_repository(array, _spatial_context_ids(array), codec=_SPATIAL_REPO_CODEC, mode=_MODE_SPATIAL)


# Compatibility names retained, but their semantics are now genuinely spatial.
encode_context_arithmetic = encode_spatial_context_arithmetic


def _context_from_restored(view: np.ndarray, batch: int, row: int, column: int, channel: int) -> int:
    left = int(view[batch, row, column - 1, channel]) if column else 0
    upper = int(view[batch, row - 1, column, channel]) if row else 0

    def sign(value: int) -> int:
        return 0 if value == 0 else 1 if value > 0 else 2

    def bucket(value: int) -> int:
        magnitude = abs(value)
        return 0 if magnitude == 0 else 1 if magnitude == 1 else 2 if magnitude <= 3 else 3

    return ((sign(left) * 3 + sign(upper)) * 4 + bucket(left)) * 4 + bucket(upper)


def _decode_repository(data: bytes | bytearray | memoryview, *, codec: int, mode: int) -> np.ndarray:
    dtype, shape, count, digest, model_payload, stream, _ = _parse_array_frame(data, codec)
    if mode == _MODE_SPATIAL and len(shape) < 3:
        raise RateCoderError("spatial context frame must have shape [..., H, W, C]")
    model = _parse_model(model_payload, expected_mode=mode, expected_width=dtype.itemsize)
    if count and not stream:
        raise RateCoderError("nonempty arithmetic frame has no range stream")
    decoder = RangeDecoder(stream)
    restored = np.zeros(shape, dtype=dtype)
    batch_count = _element_count(shape[:-3]) if mode == _MODE_SPATIAL else 1
    if mode == _MODE_SPATIAL:
        height, width, channels = shape[-3:]
        view = restored.reshape(batch_count, height, width, channels)
        positions = (
            (b, y, x, c)
            for b in range(batch_count)
            for y in range(height)
            for x in range(width)
            for c in range(channels)
        )
    else:
        view = restored.reshape(1, 1, count, 1)
        positions = ((0, 0, index, 0) for index in range(count))
    info = np.iinfo(dtype)
    try:
        for batch, row, column, channel in positions:
            context = _context_from_restored(view, batch, row, column, channel) if mode == _MODE_SPATIAL else 0
            if context not in model.rows:
                raise RateCoderError("decoded spatial context is absent from model table")
            sign_frequencies, bit_frequencies = model.rows[context]
            sign = _decode_symbol(decoder, _cumulative(sign_frequencies))
            magnitude = 0
            for bit_row in bit_frequencies:
                magnitude = (magnitude << 1) | _decode_symbol(decoder, _cumulative(bit_row))
            scalar = 0 if sign == 0 else magnitude if sign == 1 else -magnitude
            if (
                scalar < info.min
                or scalar > info.max
                or (sign == 0 and magnitude != 0)
                or (sign != 0 and magnitude == 0)
            ):
                raise RateCoderError("decoded arithmetic value is noncanonical or outside dtype")
            view[batch, row, column, channel] = scalar
    except (IndexError, ValueError) as exc:
        raise RateCoderError("invalid repository arithmetic stream") from exc
    if hashlib.sha256(_canonical_array_bytes(restored)).digest() != digest:
        raise RateCoderError("arithmetic payload hash mismatch")
    # The repository RangeDecoder has zero-fill semantics after EOF.  Exact
    # canonical re-encoding is therefore the strict termination/trailer check.
    canonical = (
        encode_spatial_context_arithmetic(restored) if mode == _MODE_SPATIAL else encode_iid_arithmetic(restored)
    )
    if canonical != _as_bytes(data):
        raise RateCoderError("noncanonical, truncated, or trailing repository arithmetic stream")
    return restored


def decode_iid_arithmetic(data: bytes | bytearray | memoryview) -> np.ndarray:
    return _decode_repository(data, codec=_IID_CODEC, mode=_MODE_IID)


def decode_spatial_context_arithmetic(data: bytes | bytearray | memoryview) -> np.ndarray:
    return _decode_repository(data, codec=_SPATIAL_REPO_CODEC, mode=_MODE_SPATIAL)


decode_context_arithmetic = decode_spatial_context_arithmetic


def _constriction_module() -> Any | None:
    try:
        import constriction  # type: ignore[import-not-found]
    except ImportError:
        return None
    return constriction


def _categorical(constriction: Any, frequencies: np.ndarray) -> Any:
    probabilities = frequencies.astype(np.float64)
    probabilities /= probabilities.sum()
    return constriction.stream.model.Categorical(probabilities=probabilities, perfect=False)


def _encode_constriction_groups(array: np.ndarray, contexts: np.ndarray, model: _ModelTable) -> bytes:
    constriction = _constriction_module()
    if constriction is None:
        raise RateCoderError("constriction dependency is unavailable")
    flat_contexts = contexts.reshape(-1)
    flat_values = array.reshape(-1)
    signs = _sign_classes(flat_values).astype(np.int32)
    magnitudes = _magnitude_uint64(flat_values)
    bits = array.dtype.itemsize * 8
    chunks: list[bytes] = []
    order = np.argsort(flat_contexts, kind="stable")
    sorted_contexts = flat_contexts[order]
    unique, starts, counts = np.unique(sorted_contexts, return_index=True, return_counts=True)
    grouped = {
        int(context): order[int(start) : int(start + count)]
        for context, start, count in zip(unique, starts, counts, strict=True)
    }
    for context in sorted(model.rows):
        indices = grouped[context]
        count = int(indices.size)
        sign_frequencies, bit_frequencies = model.rows[context]
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(signs[indices], _categorical(constriction, sign_frequencies))
        selected = magnitudes[indices]
        for bit_index in range(bits):
            symbols = ((selected >> (bits - 1 - bit_index)) & 1).astype(np.int32)
            encoder.encode(symbols, _categorical(constriction, bit_frequencies[bit_index]))
        stream = np.asarray(encoder.get_compressed(), dtype="<u4").tobytes()
        chunks.append(_GROUP_PREFIX.pack(context, count, len(stream)) + stream)
    return b"".join(chunks)


def encode_spatial_context_constriction(value: np.ndarray) -> bytes:
    """Batch-encode one constriction stream per spatial context group.

    This format requires constriction to decode.  It is intentionally separate
    from, and wire-incompatible with, the repository RangeDecoder format.
    """

    array = _spatial_array(value)
    contexts = _spatial_context_ids(array)
    model = _build_model(array, contexts, mode=_MODE_SPATIAL)
    stream = _encode_constriction_groups(array, contexts, model)
    return _array_frame(array, _SPATIAL_CONSTRICTION_CODEC, _serialize_model(model), stream)


def decode_spatial_context_constriction(data: bytes | bytearray | memoryview) -> np.ndarray:
    constriction = _constriction_module()
    if constriction is None:
        raise RateCoderError("constriction dependency is unavailable for decoder")
    dtype, shape, count, digest, model_payload, stream, _ = _parse_array_frame(data, _SPATIAL_CONSTRICTION_CODEC)
    if len(shape) < 3:
        raise RateCoderError("constriction spatial frame must have shape [..., H, W, C]")
    model = _parse_model(model_payload, expected_mode=_MODE_SPATIAL, expected_width=dtype.itemsize)
    offset = 0
    decoded_groups: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    bits = dtype.itemsize * 8
    for expected_context in sorted(model.rows):
        if len(stream) < offset + _GROUP_PREFIX.size:
            raise RateCoderError("truncated constriction group header")
        context, group_count, stream_size = _GROUP_PREFIX.unpack_from(stream, offset)
        offset += _GROUP_PREFIX.size
        if context != expected_context or len(stream) < offset + stream_size or stream_size % 4:
            raise RateCoderError("invalid constriction group framing")
        words = np.frombuffer(stream[offset : offset + stream_size], dtype="<u4").copy()
        offset += stream_size
        sign_frequencies, bit_frequencies = model.rows[context]
        try:
            decoder = constriction.stream.queue.RangeDecoder(words)
            signs = np.asarray(
                decoder.decode(_categorical(constriction, sign_frequencies), int(group_count)), dtype=np.int64
            )
            bit_values = np.empty((int(group_count), bits), dtype=np.uint8)
            for bit_index in range(bits):
                bit_values[:, bit_index] = decoder.decode(
                    _categorical(constriction, bit_frequencies[bit_index]), int(group_count)
                )
        except (ValueError, RuntimeError) as exc:
            raise RateCoderError("invalid constriction range stream") from exc
        decoded_groups[int(context)] = (signs, bit_values)
    if offset != len(stream):
        raise RateCoderError("trailing constriction group bytes")
    restored = np.zeros(shape, dtype=dtype)
    height, width, channels = shape[-3:]
    batch_count = _element_count(shape[:-3])
    view = restored.reshape(batch_count, height, width, channels)
    consumed = dict.fromkeys(decoded_groups, 0)
    info = np.iinfo(dtype)
    for batch in range(batch_count):
        for row in range(height):
            for column in range(width):
                for channel in range(channels):
                    context = _context_from_restored(view, batch, row, column, channel)
                    if context not in decoded_groups:
                        raise RateCoderError("decoded constriction context is absent")
                    signs, bit_values = decoded_groups[context]
                    index = consumed[context]
                    if index >= signs.size:
                        raise RateCoderError("constriction context symbol count exhausted")
                    magnitude = 0
                    for symbol in bit_values[index]:
                        magnitude = (magnitude << 1) | int(symbol)
                    sign = int(signs[index])
                    scalar = 0 if sign == 0 else magnitude if sign == 1 else -magnitude
                    if (
                        scalar < info.min
                        or scalar > info.max
                        or (sign == 0 and magnitude != 0)
                        or (sign != 0 and magnitude == 0)
                    ):
                        raise RateCoderError("decoded constriction value is noncanonical or outside dtype")
                    view[batch, row, column, channel] = scalar
                    consumed[context] += 1
    if any(consumed[context] != values[0].size for context, values in decoded_groups.items()):
        raise RateCoderError("unused constriction context symbols")
    if hashlib.sha256(_canonical_array_bytes(restored)).digest() != digest:
        raise RateCoderError("constriction arithmetic payload hash mismatch")
    # Exact re-encoding rejects hidden trailers and states the true decoder
    # dependency.  Repository RangeDecoder is never involved in this path.
    if encode_spatial_context_constriction(restored) != _as_bytes(data):
        raise RateCoderError("noncanonical, truncated, or trailing constriction stream")
    if restored.size != count:
        raise RateCoderError("constriction decoded count mismatch")
    return restored


def _zigzag(value: int) -> int:
    return value << 1 if value >= 0 else (-value << 1) - 1


def _unzigzag(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def _varint(value: int) -> bytes:
    if value < 0:
        raise RateCoderError("varint cannot encode a negative value")
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(output)


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data) or shift > 70:
            raise RateCoderError("truncated or oversized RLE varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def _rle_tokens(array: np.ndarray) -> bytes:
    values = [int(value) for value in array.reshape(-1)]
    tokens = bytearray()
    index = 0
    while index < len(values):
        if values[index] == 0:
            end = index + 1
            while end < len(values) and values[end] == 0:
                end += 1
            tokens.append(0)
            tokens.extend(_varint(end - index))
            index = end
        else:
            tokens.append(1)
            tokens.extend(_varint(_zigzag(values[index])))
            index += 1
    return bytes(tokens)


def _rle_restore(tokens: bytes, count: int, dtype: np.dtype[Any]) -> np.ndarray:
    values: list[int] = []
    offset = 0
    info = np.iinfo(dtype)
    while offset < len(tokens):
        marker = tokens[offset]
        offset += 1
        item, offset = _read_varint(tokens, offset)
        if marker == 0:
            if item <= 0 or len(values) + item > count:
                raise RateCoderError("invalid zero-run in RLE stream")
            values.extend([0] * item)
        elif marker == 1:
            value = _unzigzag(item)
            if value == 0 or value < info.min or value > info.max or len(values) >= count:
                raise RateCoderError("invalid nonzero RLE value")
            values.append(value)
        else:
            raise RateCoderError("invalid RLE marker")
    if len(values) != count:
        raise RateCoderError("RLE stream does not restore declared element count")
    return np.asarray(values, dtype=dtype)


def encode_zigzag_rle_arithmetic(value: np.ndarray) -> bytes:
    """Dependency-free zigzag/zero-RLE floor check under a binary range model."""

    array = _signed_array(value)
    tokens = _rle_tokens(array)
    # Encode every token byte as eight IID bits.  This is O(N*8), not O(N*256).
    bit_values = (
        np.unpackbits(np.frombuffer(tokens, dtype=np.uint8), bitorder="big") if tokens else np.empty(0, np.uint8)
    )
    counts = np.bincount(bit_values, minlength=2).astype(np.uint32) + 1
    cumulative = _cumulative(counts)
    encoder = RangeEncoder()
    for symbol in bit_values:
        encoder.encode(symbol=int(symbol), cumulative=cumulative, total=cumulative[-1])
    model = struct.pack("<QII", len(tokens), int(counts[0]), int(counts[1]))
    return _array_frame(array, _RLE_CODEC, model, encoder.finish())


def decode_zigzag_rle_arithmetic(data: bytes | bytearray | memoryview) -> np.ndarray:
    dtype, shape, count, digest, model, stream, _ = _parse_array_frame(data, _RLE_CODEC)
    if len(model) != struct.calcsize("<QII"):
        raise RateCoderError("invalid RLE model table")
    token_count, zero_count, one_count = struct.unpack("<QII", model)
    if zero_count <= 0 or one_count <= 0:
        raise RateCoderError("invalid RLE bit frequencies")
    cumulative = _cumulative(np.array([zero_count, one_count], dtype=np.uint32))
    decoder = RangeDecoder(stream)
    bits = np.empty(int(token_count) * 8, dtype=np.uint8)
    try:
        for index in range(bits.size):
            bits[index] = _decode_symbol(decoder, cumulative)
    except (IndexError, ValueError) as exc:
        raise RateCoderError("invalid RLE arithmetic stream") from exc
    tokens = np.packbits(bits, bitorder="big").tobytes()
    restored = _rle_restore(tokens, count, dtype).reshape(shape)
    if hashlib.sha256(_canonical_array_bytes(restored)).digest() != digest:
        raise RateCoderError("RLE arithmetic payload hash mismatch")
    if encode_zigzag_rle_arithmetic(restored) != _as_bytes(data):
        raise RateCoderError("noncanonical, truncated, or trailing RLE arithmetic stream")
    return restored


def _frame_accounting(frame: bytes, codec: int) -> tuple[int, int, int]:
    _, _, _, _, model, stream, framing = _parse_array_frame(frame, codec)
    return framing, len(model), len(stream)


def _measure(
    name: str,
    payload: Any,
    encoder: Callable[[Any], bytes],
    decoder: Callable[[bytes], Any],
    expected: Any,
    dependency: str,
    *,
    codec_id: int | None = None,
    decoder_dependency_required: bool = False,
) -> CodecMeasurement:
    started = time.perf_counter()
    encoded = encoder(payload)
    encoder_seconds = time.perf_counter() - started
    started = time.perf_counter()
    decoded = decoder(encoded)
    decoder_seconds = time.perf_counter() - started
    if isinstance(expected, np.ndarray):
        exact = (
            isinstance(decoded, np.ndarray) and decoded.dtype == expected.dtype and np.array_equal(decoded, expected)
        )
    else:
        exact = decoded == expected
    if not exact:
        raise RateCoderError(f"{name} parse-back differs from input")
    framing_bytes = model_bytes = payload_bytes = None
    if codec_id is not None:
        framing_bytes, model_bytes, payload_bytes = _frame_accounting(encoded, codec_id)
    return CodecMeasurement(
        codec=name,
        framed_bytes=len(encoded),
        sha256=_sha256(encoded),
        parseback_exact=True,
        implementation_identity=f"{__name__}:v2",
        dependency_identity=dependency,
        encoder_seconds=encoder_seconds,
        decoder_seconds=decoder_seconds,
        framing_bytes=framing_bytes,
        model_table_bytes=model_bytes,
        payload_bytes=payload_bytes,
        decoder_dependency_required=decoder_dependency_required,
    )


def _unavailable(
    name: str, dependency: str, reason: str, *, decoder_dependency_required: bool = False
) -> dict[str, Any]:
    return CodecMeasurement(
        codec=name,
        framed_bytes=0,
        sha256="",
        parseback_exact=False,
        implementation_identity=f"{__name__}:v2",
        dependency_identity=dependency,
        encoder_seconds=0.0,
        decoder_seconds=0.0,
        available=False,
        unavailable_reason=reason,
        decoder_dependency_required=decoder_dependency_required,
    ).as_dict()


def measure_byte_ladder(data: bytes | bytearray | memoryview) -> dict[str, dict[str, Any]]:
    """Measure raw, LZMA, optional Brotli-Q11, and optional zstd-19 bytes."""

    raw = _as_bytes(data)
    rows = {
        "raw": CodecMeasurement("raw", len(raw), _sha256(raw), True, f"{__name__}:v2", "stdlib", 0.0, 0.0).as_dict(),
        "lzma_xz_preset9": _measure("lzma_xz_preset9", raw, encode_lzma, decode_lzma, raw, "stdlib:lzma").as_dict(),
    }
    for name, encoder, decoder, dependency in (
        ("brotli_q11", encode_brotli_q11, decode_brotli_q11, "optional:brotli-quality-11"),
        ("zstd_19", encode_zstd_19, decode_zstd_19, _zstd_dependency_identity()),
    ):
        try:
            rows[name] = _measure(name, raw, encoder, decoder, raw, dependency).as_dict()
        except RateCoderError as exc:
            rows[name] = _unavailable(name, dependency, str(exc))
    return rows


def measure_signed_array_ladder(value: np.ndarray) -> dict[str, dict[str, Any]]:
    """Measure complete exact frames for a signed spatial array."""

    array = _spatial_array(value)
    rows = measure_byte_ladder(serialize_signed_array(array))
    rows["repository_iid_arithmetic"] = _measure(
        "repository_iid_arithmetic",
        array,
        encode_iid_arithmetic,
        decode_iid_arithmetic,
        array,
        "repository:tac.lossless.range_coder",
        codec_id=_IID_CODEC,
    ).as_dict()
    rows["repository_spatial_context_arithmetic"] = _measure(
        "repository_spatial_context_arithmetic",
        array,
        encode_spatial_context_arithmetic,
        decode_spatial_context_arithmetic,
        array,
        "repository:tac.lossless.range_coder;context=left/up-same-channel-sign+magnitude",
        codec_id=_SPATIAL_REPO_CODEC,
    ).as_dict()
    rows["zigzag_rle_arithmetic"] = _measure(
        "zigzag_rle_arithmetic",
        array,
        encode_zigzag_rle_arithmetic,
        decode_zigzag_rle_arithmetic,
        array,
        "repository:tac.lossless.range_coder",
        codec_id=_RLE_CODEC,
    ).as_dict()
    try:
        version = importlib.metadata.version("constriction")
        rows["constriction_spatial_context_arithmetic"] = _measure(
            "constriction_spatial_context_arithmetic",
            array,
            encode_spatial_context_constriction,
            decode_spatial_context_constriction,
            array,
            f"optional:constriction-{version};decoder=constriction-not-repository-RangeDecoder",
            codec_id=_SPATIAL_CONSTRICTION_CODEC,
            decoder_dependency_required=True,
        ).as_dict()
    except (importlib.metadata.PackageNotFoundError, RateCoderError) as exc:
        rows["constriction_spatial_context_arithmetic"] = _unavailable(
            "constriction_spatial_context_arithmetic",
            "optional:constriction;decoder=constriction-not-repository-RangeDecoder",
            str(exc),
            decoder_dependency_required=True,
        )
    return rows


def measure_iid_signed_array_ladder(value: np.ndarray) -> dict[str, dict[str, Any]]:
    """Measure byte comparators plus repository IID coding for any signed rank."""

    array = _signed_array(value)
    rows = measure_byte_ladder(serialize_signed_array(array))
    rows["repository_iid_arithmetic"] = _measure(
        "repository_iid_arithmetic",
        array,
        encode_iid_arithmetic,
        decode_iid_arithmetic,
        array,
        "repository:tac.lossless.range_coder",
        codec_id=_IID_CODEC,
    ).as_dict()
    return rows


def measure_int8_coder_comparison(value: np.ndarray) -> dict[str, Any]:
    """Return the explicit fully framed IID-vs-spatial comparison for int8."""

    array = _spatial_array(value)
    if array.dtype != np.dtype("int8"):
        raise RateCoderError("IID-vs-context comparison requires an int8 tensor")
    ladder = measure_signed_array_ladder(array)
    iid = ladder["repository_iid_arithmetic"]
    spatial = ladder["repository_spatial_context_arithmetic"]
    return {
        "authority_label": "MEASURED_LOCAL_EXACT_INT8_FRAMES",
        "shape_contract": "[...,H,W,C]; left/up same-channel sign+magnitude",
        "iid": iid,
        "spatial_context": spatial,
        "context_over_iid_ratio": float(spatial["framed_bytes"]) / float(iid["framed_bytes"]),
        "constriction_spatial_context": ladder["constriction_spatial_context_arithmetic"],
    }


def measure_block_fp(value: np.ndarray, *, block_size: int = 16, clip_threshold: float = 0.5) -> dict[str, Any]:
    """Measure the repository ternary shared-exponent block-FP quantizer."""

    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        raise RateCoderError("block-FP measurement requires a non-scalar tensor")
    try:
        import torch

        from tac.block_fp_codec import BlockFPHeader, pack_block_fp, unpack_block_fp
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RateCoderError("block-FP measurement requires torch") from exc
    original = torch.from_numpy(np.ascontiguousarray(array))
    started = time.perf_counter()
    packed = pack_block_fp(original, block_size=block_size, clip_threshold=clip_threshold)
    encoder_seconds = time.perf_counter() - started
    header, header_bytes = BlockFPHeader.decode(packed)
    started = time.perf_counter()
    restored = unpack_block_fp(packed).cpu().numpy()
    decoder_seconds = time.perf_counter() - started
    qint_start = header_bytes
    qint_end = qint_start + header.qint_nbytes
    exponent_end = qint_end + header.exponents_nbytes
    if exponent_end != len(packed):
        raise RateCoderError("block-FP qint/exponent/header accounting does not sum to framed bytes")
    error = restored - array
    packed_byte_coders = measure_byte_ladder(packed)
    return {
        "authority_label": "MEASURED_LOSSY_BLOCK_FP_QUANTIZER_ONLY",
        "codec": "tac.block_fp_codec ternary shared exponent",
        "framed_bytes": len(packed),
        "sha256": _sha256(packed),
        "parseback_exact": bool(np.array_equal(restored, array)),
        "byte_accounting": {
            "qint_bytes": int(header.qint_nbytes),
            "exponent_bytes": int(header.exponents_nbytes),
            "header_bytes": int(header_bytes),
            "sum_matches_framed_bytes": int(header.qint_nbytes + header.exponents_nbytes + header_bytes) == len(packed),
        },
        "reconstruction_metadata": {
            "shape": list(array.shape),
            "dtype": "float32",
            "block_size": int(header.block_size),
            "clip_threshold": float(header.clip_threshold),
            "qint_sha256": _sha256(packed[qint_start:qint_end]),
            "exponent_sha256": _sha256(packed[qint_end:exponent_end]),
        },
        "distortion": {
            "mse": float(np.mean(np.square(error, dtype=np.float64))) if error.size else 0.0,
            "max_abs": float(np.max(np.abs(error))) if error.size else 0.0,
        },
        "packed_byte_coders": packed_byte_coders,
        "best_packed_byte_coder": min(
            (
                (name, int(row["framed_bytes"]))
                for name, row in packed_byte_coders.items()
                if row.get("available") and isinstance(row.get("framed_bytes"), int)
            ),
            key=lambda item: item[1],
        )[0],
        "implementation_identity": "tac.block_fp_codec",
        "dependency_identity": f"torch-{torch.__version__}",
        "encoder_seconds": encoder_seconds,
        "decoder_seconds": decoder_seconds,
        "sensitivity_allocator_composition": "UNMEASURED_NO_ALLOCATION_APPLIED",
        "matched_realized_dseg": "OWED_N_GE_24_UNLESS_ACTUAL_RESULTS_SUPPLIED",
    }


def authority_labels() -> dict[str, str]:
    return {
        "codec_bytes": "MEASURED_LOCAL_PARSEBACK_BYTES",
        "pdw1": "MEASURED_REDERIVED_PDW1_EXACTLY_338_BYTES",
        "pdw2": "DERIVED_ONLY_NO_STRICT_ENCODER",
        "block_fp": "MEASURED_LOSSY_BLOCK_FP_QUANTIZER_ONLY",
        "score": "UNMEASURED_NO_SCORER_OR_CONTEST_AXIS",
    }


__all__ = [
    "CodecMeasurement",
    "RateCoderError",
    "authority_labels",
    "decode_brotli_q11",
    "decode_context_arithmetic",
    "decode_iid_arithmetic",
    "decode_lzma",
    "decode_spatial_context_arithmetic",
    "decode_spatial_context_constriction",
    "decode_zigzag_rle_arithmetic",
    "decode_zstd_19",
    "deserialize_signed_array",
    "encode_brotli_q11",
    "encode_context_arithmetic",
    "encode_iid_arithmetic",
    "encode_lzma",
    "encode_spatial_context_arithmetic",
    "encode_spatial_context_constriction",
    "encode_zigzag_rle_arithmetic",
    "encode_zstd_19",
    "measure_block_fp",
    "measure_byte_ladder",
    "measure_iid_signed_array_ladder",
    "measure_int8_coder_comparison",
    "measure_signed_array_ladder",
    "serialize_signed_array",
]
