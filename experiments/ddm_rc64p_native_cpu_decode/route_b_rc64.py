"""Checkpointable PR135 RC64 bindings for the lc2 Route-B race.

The arithmetic-coder recurrence is borrowed unchanged from the granted PR135
``rc64_backend.c``.  The source extension below only exposes deterministic
snapshot/resume state for the encoder and decoder so long runs remain
crash-resumable.
"""

from __future__ import annotations

import ctypes
import struct
from pathlib import Path

import numpy as np


ALPHABET = 5
TOTAL = 1 << 31
EXPECTED_SYMBOLS = 600 * 384 * 512
TOKEN_MAGIC = b"R6D1"
CHECKPOINT_MAGIC = b"R6C1"
ENCODER_MAGIC = b"RCE1"
DECODER_HEADER = struct.Struct("<4sQQQQQQ")
ENCODER_HEADER = struct.Struct("<4sQQQQBB")


RC64_CHECKPOINT_EXTENSION = r'''

int rc64_encoder_snapshot(
    const void *opaque,
    uint64_t *low,
    uint64_t *high,
    uint64_t *pending,
    uint8_t *partial,
    uint8_t *partial_bits,
    uint8_t *data,
    size_t capacity,
    size_t *size
) {
    const rc64_encoder *encoder = (const rc64_encoder *)opaque;
    if (!encoder || encoder->error || encoder->finished || !low || !high ||
        !pending || !partial || !partial_bits || !size) return -1;
    *low = encoder->low;
    *high = encoder->high;
    *pending = encoder->pending;
    *partial = encoder->partial;
    *partial_bits = encoder->partial_bits;
    *size = encoder->size;
    if (!data) return 0;
    if (capacity < encoder->size) return -2;
    if (encoder->size) memcpy(data, encoder->data, encoder->size);
    return 0;
}

void *rc64_encoder_resume(
    uint64_t low,
    uint64_t high,
    uint64_t pending,
    uint8_t partial,
    uint8_t partial_bits,
    const uint8_t *data,
    size_t size
) {
    rc64_encoder *encoder = (rc64_encoder *)rc64_encoder_create();
    if (!encoder || low > high || high > RC64_TOP || partial_bits > 7u ||
        (size && !data)) {
        rc64_encoder_destroy(encoder);
        return NULL;
    }
    if (size && !rc64_reserve(encoder, size)) {
        rc64_encoder_destroy(encoder);
        return NULL;
    }
    if (size) memcpy(encoder->data, data, size);
    encoder->size = size;
    encoder->low = low;
    encoder->high = high;
    encoder->pending = pending;
    encoder->partial = partial;
    encoder->partial_bits = partial_bits;
    return encoder;
}

int rc64_decoder_snapshot(
    const void *opaque,
    uint64_t *low,
    uint64_t *high,
    uint64_t *code,
    size_t *bit_position
) {
    const rc64_decoder *decoder = (const rc64_decoder *)opaque;
    if (!decoder || decoder->error || !low || !high || !code || !bit_position)
        return -1;
    *low = decoder->low;
    *high = decoder->high;
    *code = decoder->code;
    *bit_position = decoder->bit_position;
    return 0;
}

void *rc64_decoder_resume(
    const uint8_t *data,
    size_t size,
    uint64_t low,
    uint64_t high,
    uint64_t code,
    size_t bit_position
) {
    rc64_decoder *decoder;
    if (!data || !size || low > high || high > RC64_TOP || code < low ||
        code > high || bit_position > size * 8u + 63u) return NULL;
    decoder = (rc64_decoder *)calloc(1u, sizeof(rc64_decoder));
    if (!decoder) return NULL;
    decoder->data = data;
    decoder->size = size;
    decoder->low = low;
    decoder->high = high;
    decoder->code = code;
    decoder->bit_position = bit_position;
    return decoder;
}
'''


class Rc64Error(ValueError):
    """The RC64 payload, probability lattice, or checkpoint is invalid."""


def quantize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != ALPHABET or not values.size:
        raise Rc64Error("RC64 probabilities must have shape [N, 5]")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise Rc64Error("RC64 probabilities must be finite and positive")
    row_sums = values.astype(np.float64).sum(axis=1)
    if np.any(np.abs(row_sums - 1.0) > 2e-5):
        raise Rc64Error("RC64 probability rows must sum to one")
    frequencies = np.floor(values.astype(np.float64) * TOTAL).astype(np.int64)
    np.maximum(frequencies, 1, out=frequencies)
    winners = values.argmax(axis=1)
    balance = TOTAL - frequencies.sum(axis=1)
    frequencies[np.arange(len(values)), winners] += balance
    if np.any(frequencies <= 0) or np.any(frequencies.sum(axis=1) != TOTAL):
        raise Rc64Error("RC64 frequency normalization failed")
    return np.ascontiguousarray(frequencies, dtype=np.uint32)


def _library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path.resolve()))
    u8p = ctypes.POINTER(ctypes.c_uint8)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    i32p = ctypes.POINTER(ctypes.c_int32)
    f32p = ctypes.POINTER(ctypes.c_float)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    sizep = ctypes.POINTER(ctypes.c_size_t)
    library.rc64_encoder_create.restype = ctypes.c_void_p
    library.rc64_encoder_destroy.argtypes = [ctypes.c_void_p]
    library.rc64_encoder_encode.argtypes = [
        ctypes.c_void_p, i32p, u32p, ctypes.c_size_t
    ]
    library.rc64_encoder_encode.restype = ctypes.c_int
    library.rc64_encoder_finish.argtypes = [ctypes.c_void_p]
    library.rc64_encoder_finish.restype = ctypes.c_int
    library.rc64_encoder_data.argtypes = [ctypes.c_void_p]
    library.rc64_encoder_data.restype = u8p
    library.rc64_encoder_size.argtypes = [ctypes.c_void_p]
    library.rc64_encoder_size.restype = ctypes.c_size_t
    library.rc64_encoder_snapshot.argtypes = [
        ctypes.c_void_p, u64p, u64p, u64p,
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
        u8p, ctypes.c_size_t, sizep,
    ]
    library.rc64_encoder_snapshot.restype = ctypes.c_int
    library.rc64_encoder_resume.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_uint8, ctypes.c_uint8, u8p, ctypes.c_size_t,
    ]
    library.rc64_encoder_resume.restype = ctypes.c_void_p
    library.rc64_decoder_create.argtypes = [u8p, ctypes.c_size_t]
    library.rc64_decoder_create.restype = ctypes.c_void_p
    library.rc64_decoder_destroy.argtypes = [ctypes.c_void_p]
    library.rc64_decoder_decode_probabilities.argtypes = [
        ctypes.c_void_p, f32p, ctypes.c_size_t, i32p
    ]
    library.rc64_decoder_decode_probabilities.restype = ctypes.c_int
    library.rc64_decoder_snapshot.argtypes = [
        ctypes.c_void_p, u64p, u64p, u64p, sizep
    ]
    library.rc64_decoder_snapshot.restype = ctypes.c_int
    library.rc64_decoder_resume.argtypes = [
        u8p, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_uint64, ctypes.c_size_t,
    ]
    library.rc64_decoder_resume.restype = ctypes.c_void_p
    library.rc64_total_frequency.restype = ctypes.c_uint64
    if library.rc64_total_frequency() != TOTAL:
        raise Rc64Error("RC64 backend uses the wrong total frequency")
    return library


class NativeRc64Encoder:
    def __init__(self, library_path: Path, checkpoint: bytes | None = None) -> None:
        self.library = _library(library_path)
        self.context: int | None = None
        self.finished = False
        if checkpoint is None:
            self.context = self.library.rc64_encoder_create()
        else:
            if len(checkpoint) < ENCODER_HEADER.size:
                raise Rc64Error("RC64 encoder checkpoint is truncated")
            magic, low, high, pending, size, partial, partial_bits = (
                ENCODER_HEADER.unpack_from(checkpoint)
            )
            data = checkpoint[ENCODER_HEADER.size:]
            if magic != ENCODER_MAGIC or len(data) != size:
                raise Rc64Error("RC64 encoder checkpoint framing changed")
            buffer = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
            self.context = self.library.rc64_encoder_resume(
                low, high, pending, partial, partial_bits, buffer, len(data)
            )
        if not self.context:
            raise Rc64Error("RC64 encoder allocation/resume failed")

    def close(self) -> None:
        if self.context:
            self.library.rc64_encoder_destroy(self.context)
            self.context = None

    def __del__(self) -> None:  # pragma: no cover
        self.close()

    def encode(self, symbols: np.ndarray, probabilities: np.ndarray) -> None:
        source = np.ascontiguousarray(symbols, dtype=np.int32).reshape(-1)
        frequencies = quantize_probabilities(probabilities)
        status = self.library.rc64_encoder_encode(
            self.context,
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            frequencies.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            len(source),
        )
        if status:
            raise Rc64Error(f"RC64 encode failed with status {status}")

    def snapshot(self) -> bytes:
        low = ctypes.c_uint64()
        high = ctypes.c_uint64()
        pending = ctypes.c_uint64()
        partial = ctypes.c_uint8()
        partial_bits = ctypes.c_uint8()
        size = ctypes.c_size_t()
        status = self.library.rc64_encoder_snapshot(
            self.context, ctypes.byref(low), ctypes.byref(high),
            ctypes.byref(pending), ctypes.byref(partial),
            ctypes.byref(partial_bits), None, 0, ctypes.byref(size)
        )
        if status:
            raise Rc64Error(f"RC64 encoder snapshot size failed: {status}")
        data = (ctypes.c_uint8 * size.value)()
        status = self.library.rc64_encoder_snapshot(
            self.context, ctypes.byref(low), ctypes.byref(high),
            ctypes.byref(pending), ctypes.byref(partial),
            ctypes.byref(partial_bits), data, size.value, ctypes.byref(size)
        )
        if status:
            raise Rc64Error(f"RC64 encoder snapshot failed: {status}")
        return ENCODER_HEADER.pack(
            ENCODER_MAGIC, low.value, high.value, pending.value, size.value,
            partial.value, partial_bits.value
        ) + bytes(data)

    def finish(self) -> bytes:
        if not self.finished:
            status = self.library.rc64_encoder_finish(self.context)
            if status:
                raise Rc64Error(f"RC64 finish failed with status {status}")
            self.finished = True
        size = int(self.library.rc64_encoder_size(self.context))
        pointer = self.library.rc64_encoder_data(self.context)
        if not size or not pointer:
            raise Rc64Error("RC64 encoder produced no payload")
        payload = TOKEN_MAGIC + ctypes.string_at(pointer, size)
        return payload + b"\0" * ((-len(payload)) % 4)


class NativeRc64Decoder:
    """Receiver-compatible decoder with a u32 checkpoint envelope."""

    def __init__(self, library_path: Path, payload: bytes) -> None:
        self.library = _library(library_path)
        self.context: int | None = None
        if payload.startswith(CHECKPOINT_MAGIC):
            if len(payload) < DECODER_HEADER.size:
                raise Rc64Error("RC64 decoder checkpoint is truncated")
            magic, size, low, high, code, bit_position, decoded = (
                DECODER_HEADER.unpack_from(payload)
            )
            raw = payload[DECODER_HEADER.size:DECODER_HEADER.size + size]
            if magic != CHECKPOINT_MAGIC or len(raw) != size:
                raise Rc64Error("RC64 decoder checkpoint framing changed")
            self.decoded_symbols = int(decoded)
            self.payload = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
            self.context = self.library.rc64_decoder_resume(
                self.payload, len(raw), low, high, code, bit_position
            )
        elif payload.startswith(TOKEN_MAGIC):
            raw = payload[4:]
            if not raw:
                raise Rc64Error("RC64 token payload is empty")
            self.decoded_symbols = 0
            self.payload = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
            self.context = self.library.rc64_decoder_create(self.payload, len(raw))
        else:
            raise Rc64Error("RC64 token payload lacks its data/checkpoint magic")
        if not self.context:
            raise Rc64Error("RC64 decoder allocation/resume failed")

    def close(self) -> None:
        if self.context:
            self.library.rc64_decoder_destroy(self.context)
            self.context = None

    def __del__(self) -> None:  # pragma: no cover
        self.close()

    def decode(self, model: object, probabilities: np.ndarray) -> np.ndarray:
        del model
        values = np.ascontiguousarray(probabilities, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != ALPHABET or not len(values):
            raise Rc64Error("RC64 probabilities must have shape [N, 5]")
        output = np.empty(len(values), dtype=np.int32)
        status = self.library.rc64_decoder_decode_probabilities(
            self.context,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(values),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        if status:
            raise Rc64Error(f"RC64 decode failed with status {status}")
        self.decoded_symbols += len(output)
        if self.decoded_symbols > EXPECTED_SYMBOLS:
            raise Rc64Error("RC64 decoder consumed more than n600 symbols")
        return output

    def is_empty(self) -> bool:
        return self.decoded_symbols == EXPECTED_SYMBOLS

    def get_compressed(self) -> np.ndarray:
        low = ctypes.c_uint64()
        high = ctypes.c_uint64()
        code = ctypes.c_uint64()
        bit_position = ctypes.c_size_t()
        status = self.library.rc64_decoder_snapshot(
            self.context, ctypes.byref(low), ctypes.byref(high),
            ctypes.byref(code), ctypes.byref(bit_position)
        )
        if status:
            raise Rc64Error(f"RC64 decoder snapshot failed with status {status}")
        raw = bytes(self.payload)
        checkpoint = DECODER_HEADER.pack(
            CHECKPOINT_MAGIC, len(raw), low.value, high.value,
            code.value, bit_position.value, self.decoded_symbols
        ) + raw
        padding = (-len(checkpoint)) % 4
        return np.frombuffer(checkpoint + b"\0" * padding, dtype="<u4").copy()
