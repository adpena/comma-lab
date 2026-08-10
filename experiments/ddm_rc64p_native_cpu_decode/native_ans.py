"""ctypes binding for the byte-compatible lc2 native ANS decoder."""

from __future__ import annotations

import ctypes
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

ALPHABET = 5
PRECISION = 24
_SIZE_MAX = ctypes.c_size_t(-1).value
_LOCK = threading.Lock()
_TELEMETRY: dict[str, Any] = {
    "decoder_instances": 0,
    "decode_calls": 0,
    "decoded_symbols": 0,
    "native_entropy_seconds": 0.0,
    "snapshot_calls": 0,
}


class NativeAnsError(ValueError):
    """The native ANS library or its exact probability stream is invalid."""


def _load_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path.resolve()))
    u8_pointer = ctypes.POINTER(ctypes.c_uint8)
    u32_pointer = ctypes.POINTER(ctypes.c_uint32)
    i32_pointer = ctypes.POINTER(ctypes.c_int32)
    f32_pointer = ctypes.POINTER(ctypes.c_float)
    library.lc2_ans_decoder_create.argtypes = [u8_pointer, ctypes.c_size_t]
    library.lc2_ans_decoder_create.restype = ctypes.c_void_p
    library.lc2_ans_decoder_destroy.argtypes = [ctypes.c_void_p]
    library.lc2_ans_decoder_decode_probabilities.argtypes = [
        ctypes.c_void_p,
        f32_pointer,
        ctypes.c_size_t,
        i32_pointer,
    ]
    library.lc2_ans_decoder_decode_probabilities.restype = ctypes.c_int
    library.lc2_ans_decoder_is_empty.argtypes = [ctypes.c_void_p]
    library.lc2_ans_decoder_is_empty.restype = ctypes.c_int
    library.lc2_ans_decoder_words_remaining.argtypes = [ctypes.c_void_p]
    library.lc2_ans_decoder_words_remaining.restype = ctypes.c_size_t
    library.lc2_ans_decoder_state.argtypes = [ctypes.c_void_p]
    library.lc2_ans_decoder_state.restype = ctypes.c_uint64
    library.lc2_ans_decoder_snapshot_words.argtypes = [
        ctypes.c_void_p,
        u32_pointer,
        ctypes.c_size_t,
    ]
    library.lc2_ans_decoder_snapshot_words.restype = ctypes.c_size_t
    library.lc2_ans_precision.restype = ctypes.c_uint32
    library.lc2_ans_alphabet.restype = ctypes.c_uint32
    if library.lc2_ans_precision() != PRECISION:
        raise NativeAnsError("native ANS precision differs from constriction's 24-bit model")
    if library.lc2_ans_alphabet() != ALPHABET:
        raise NativeAnsError("native ANS alphabet differs from lc2's five classes")
    return library


def reset_telemetry() -> None:
    with _LOCK:
        _TELEMETRY.update(
            decoder_instances=0,
            decode_calls=0,
            decoded_symbols=0,
            native_entropy_seconds=0.0,
            snapshot_calls=0,
        )


def telemetry_snapshot() -> dict[str, Any]:
    with _LOCK:
        return dict(_TELEMETRY)


class NativeAnsDecoder:
    """Streaming native decoder with constriction-compatible receiver methods."""

    def __init__(self, library_path: Path, payload: bytes) -> None:
        self.context: int | None = None
        if not payload or len(payload) % 4:
            raise NativeAnsError("ANS payload must be a nonempty multiple of four bytes")
        self.library_path = library_path.resolve()
        self.library = _load_library(self.library_path)
        self.payload = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        self.context = self.library.lc2_ans_decoder_create(self.payload, len(payload))
        if not self.context:
            raise NativeAnsError("native ANS decoder rejected its initial compressed state")
        with _LOCK:
            _TELEMETRY["decoder_instances"] += 1

    def close(self) -> None:
        if self.context:
            self.library.lc2_ans_decoder_destroy(self.context)
            self.context = None

    def __del__(self) -> None:  # pragma: no cover - shutdown ordering is interpreter-specific.
        self.close()

    def decode(self, model: object, probabilities: np.ndarray) -> np.ndarray:
        del model  # The fixed categorical grammar is implemented by the native library.
        values = np.ascontiguousarray(probabilities, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != ALPHABET or not len(values):
            raise NativeAnsError("native ANS probabilities must have shape [N, 5]")
        output = np.empty(values.shape[0], dtype=np.int32)
        started = time.perf_counter()
        status = self.library.lc2_ans_decoder_decode_probabilities(
            self.context,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(values),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        elapsed = time.perf_counter() - started
        with _LOCK:
            _TELEMETRY["decode_calls"] += 1
            _TELEMETRY["decoded_symbols"] += len(values)
            _TELEMETRY["native_entropy_seconds"] += elapsed
        if status:
            raise NativeAnsError(f"native ANS decode failed with status {status}")
        return output

    def is_empty(self) -> bool:
        return bool(self.library.lc2_ans_decoder_is_empty(self.context))

    def get_compressed(self) -> np.ndarray:
        required = int(
            self.library.lc2_ans_decoder_snapshot_words(self.context, None, 0)
        )
        if required == _SIZE_MAX:
            raise NativeAnsError("native ANS snapshot refused an invalid decoder state")
        output = np.empty(required, dtype=np.uint32)
        pointer = output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        written = int(
            self.library.lc2_ans_decoder_snapshot_words(
                self.context,
                pointer,
                required,
            )
        )
        if written != required:
            raise NativeAnsError("native ANS snapshot length changed during export")
        with _LOCK:
            _TELEMETRY["snapshot_calls"] += 1
        return output

    @property
    def words_remaining(self) -> int:
        return int(self.library.lc2_ans_decoder_words_remaining(self.context))

    @property
    def state(self) -> int:
        return int(self.library.lc2_ans_decoder_state(self.context))
