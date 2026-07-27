# SPDX-License-Identifier: MIT
"""Exact population-global recode for selected V10 solution planes.

V1 pays an independent bootstrap and residual stream for every pair. This V2
wire predicts each pair's frame-0 and frame-1 planes from the preceding pair,
with explicit periodic resets. Differences are stored modulo 256 and
Brotli-compressed. The transform is exact for every uint8 value and does not
consult a teacher, target, scorer, or historical candidate at decode time.

Encoding and decoding retain at most the previous and current pair. The file is
written atomically and immutably so a failed materialization cannot be mistaken
for a complete candidate surface.
"""

from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

MAGIC: Final = b"TACV10P2"
VERSION: Final = 2
CONTENT_CODEC_TAG: Final = 1
CODEC_ID: Final = "population-predictor-residual-u8.v2"
CONTENT_CODEC_ID: Final = "brotli-q11.v1"
DECODE_RECEIPT_SCHEMA: Final = "v10_population_predictor_residual_v2_decode.v1"

# magic, version, content codec, pair count, height, width, channels,
# reset interval, pair-major decoded root.
PREFIX: Final = struct.Struct("<8sHBxIIIHI32s")
# pair id, reset flag, frame0 payload length, frame1 payload length,
# payload hashes and decoded frame hashes.
PAIR_PREFIX: Final = struct.Struct("<IB3xII32s32s32s32s")

MAX_PAIRS: Final = 10_000
MAX_HEIGHT: Final = 2048
MAX_WIDTH: Final = 2048
MAX_PAYLOAD_BYTES: Final = 1 << 30
READ_CHUNK_BYTES: Final = 1 << 20


class PopulationPredictorResidualV2Error(ValueError):
    """A V2 geometry, custody, ordering, or reconstruction invariant failed."""


@dataclass(frozen=True, slots=True)
class PopulationPair:
    """One exact selected-solution pair supplied to the encoder."""

    pair_id: int
    frame0: np.ndarray
    frame1: np.ndarray


@dataclass(frozen=True, slots=True)
class PopulationV2Receipt:
    schema: str
    codec_id: str
    path: str
    payload_bytes: int
    payload_sha256: str
    decoded_pair_major_sha256: str
    pair_count: int
    height: int
    width: int
    channels: int
    reset_interval: int
    reset_count: int
    peak_retained_pairs: int

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "codec_id": self.codec_id,
            "path": self.path,
            "payload_bytes": self.payload_bytes,
            "payload_sha256": self.payload_sha256,
            "decoded_pair_major_sha256": self.decoded_pair_major_sha256,
            "pair_count": self.pair_count,
            "geometry": [self.pair_count, self.height, self.width, self.channels],
            "reset_interval": self.reset_interval,
            "reset_count": self.reset_count,
            "peak_retained_pairs": self.peak_retained_pairs,
        }


def _brotli() -> Any:
    try:
        import brotli  # type: ignore[import-not-found]
    except ImportError as exc:
        raise PopulationPredictorResidualV2Error("brotli-q11 runtime is unavailable") from exc
    return brotli


def _compress(payload: bytes) -> bytes:
    try:
        return bytes(_brotli().compress(payload, quality=11))
    except Exception as exc:
        raise PopulationPredictorResidualV2Error("brotli-q11 compression failed") from exc


def _decompress(payload: bytes, *, expected_bytes: int, label: str) -> bytes:
    try:
        decoded = bytes(_brotli().decompress(payload))
    except Exception as exc:
        raise PopulationPredictorResidualV2Error(f"{label} Brotli decode failed") from exc
    if len(decoded) != expected_bytes:
        raise PopulationPredictorResidualV2Error(f"{label} decoded geometry drift")
    return decoded


def _sha(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_int(value: object, label: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise PopulationPredictorResidualV2Error(f"{label} must be an exact integer in [{minimum}, {maximum}]")
    return value


def _plane(value: np.ndarray, label: str) -> np.ndarray:
    plane = np.asarray(value)
    if plane.dtype != np.uint8 or plane.ndim != 3 or plane.shape[-1] != 3:
        raise PopulationPredictorResidualV2Error(f"{label} must be exact uint8 [H,W,3]")
    height, width, channels = map(int, plane.shape)
    if not 1 <= height <= MAX_HEIGHT or not 1 <= width <= MAX_WIDTH or channels != 3:
        raise PopulationPredictorResidualV2Error(f"{label} geometry is outside admitted bounds")
    return np.ascontiguousarray(plane)


def _modulo_delta(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    delta = (current.astype(np.int16) - previous.astype(np.int16)) & 0xFF
    return np.ascontiguousarray(delta.astype(np.uint8))


def _modulo_add(previous: np.ndarray, delta: np.ndarray) -> np.ndarray:
    reconstructed = (previous.astype(np.uint16) + delta.astype(np.uint16)) & 0xFF
    return np.ascontiguousarray(reconstructed.astype(np.uint8))


def _atomic_temp_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")


def encode_population_predictor_residual_v2(
    rows: Iterable[PopulationPair],
    *,
    output_path: Path,
    pair_count: int,
    height: int,
    width: int,
    reset_interval: int,
) -> PopulationV2Receipt:
    """Atomically encode exact pair rows with bounded population state."""

    expected_pairs = _exact_int(pair_count, "pair_count", minimum=1, maximum=MAX_PAIRS)
    expected_height = _exact_int(height, "height", minimum=1, maximum=MAX_HEIGHT)
    expected_width = _exact_int(width, "width", minimum=1, maximum=MAX_WIDTH)
    reset_every = _exact_int(
        reset_interval,
        "reset_interval",
        minimum=1,
        maximum=expected_pairs,
    )
    destination = Path(output_path)
    if not destination.is_absolute():
        raise PopulationPredictorResidualV2Error("output_path must be absolute")
    if destination.exists() or destination.is_symlink():
        raise PopulationPredictorResidualV2Error("immutable output_path already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_temp_path(destination)
    if temporary.exists() or temporary.is_symlink():
        raise PopulationPredictorResidualV2Error("atomic temporary output already exists")

    decoded_hasher = hashlib.sha256()
    previous0: np.ndarray | None = None
    previous1: np.ndarray | None = None
    reset_count = 0
    try:
        with temporary.open("w+b") as handle:
            handle.write(
                PREFIX.pack(
                    MAGIC,
                    VERSION,
                    CONTENT_CODEC_TAG,
                    expected_pairs,
                    expected_height,
                    expected_width,
                    3,
                    reset_every,
                    bytes(32),
                )
            )
            observed = 0
            for expected_pair_id, row in enumerate(rows):
                if not isinstance(row, PopulationPair) or row.pair_id != expected_pair_id:
                    raise PopulationPredictorResidualV2Error(
                        "population pair rows must be typed and ordered exactly 0..N-1"
                    )
                frame0 = _plane(row.frame0, f"pair {row.pair_id} frame0")
                frame1 = _plane(row.frame1, f"pair {row.pair_id} frame1")
                expected_shape = (expected_height, expected_width, 3)
                if frame0.shape != expected_shape or frame1.shape != expected_shape:
                    raise PopulationPredictorResidualV2Error(f"pair {row.pair_id} geometry drift")
                reset = row.pair_id % reset_every == 0
                if reset and row.pair_id != 0:
                    previous0 = None
                    previous1 = None
                raw0 = frame0 if reset else _modulo_delta(frame0, previous0)
                raw1 = frame1 if reset else _modulo_delta(frame1, previous1)
                payload0 = _compress(raw0.tobytes(order="C"))
                payload1 = _compress(raw1.tobytes(order="C"))
                frame0_bytes = frame0.tobytes(order="C")
                frame1_bytes = frame1.tobytes(order="C")
                handle.write(
                    PAIR_PREFIX.pack(
                        row.pair_id,
                        int(reset),
                        len(payload0),
                        len(payload1),
                        _sha(payload0),
                        _sha(payload1),
                        _sha(frame0_bytes),
                        _sha(frame1_bytes),
                    )
                )
                handle.write(payload0)
                handle.write(payload1)
                decoded_hasher.update(frame0_bytes)
                decoded_hasher.update(frame1_bytes)
                previous0, previous1 = frame0, frame1
                reset_count += int(reset)
                observed += 1
                if handle.tell() > MAX_PAYLOAD_BYTES:
                    raise PopulationPredictorResidualV2Error("V2 payload exceeds byte cap")
            if observed != expected_pairs:
                raise PopulationPredictorResidualV2Error(f"expected {expected_pairs} pairs, observed {observed}")
            decoded_root = decoded_hasher.digest()
            handle.seek(0)
            handle.write(
                PREFIX.pack(
                    MAGIC,
                    VERSION,
                    CONTENT_CODEC_TAG,
                    expected_pairs,
                    expected_height,
                    expected_width,
                    3,
                    reset_every,
                    decoded_root,
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return PopulationV2Receipt(
        schema=DECODE_RECEIPT_SCHEMA,
        codec_id=CODEC_ID,
        path=str(destination),
        payload_bytes=destination.stat().st_size,
        payload_sha256=_sha_file(destination),
        decoded_pair_major_sha256=decoded_hasher.hexdigest(),
        pair_count=expected_pairs,
        height=expected_height,
        width=expected_width,
        channels=3,
        reset_interval=reset_every,
        reset_count=reset_count,
        peak_retained_pairs=2,
    )


def decode_population_predictor_residual_v2(
    payload_path: Path,
    *,
    on_pair: Callable[[int, np.ndarray, np.ndarray], None],
) -> PopulationV2Receipt:
    """Strictly decode one pair at a time and invoke the receiver callback."""

    path = Path(payload_path)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PopulationPredictorResidualV2Error("payload_path must be one absolute regular non-symlink file")
    payload_bytes = path.stat().st_size
    if not PREFIX.size <= payload_bytes <= MAX_PAYLOAD_BYTES:
        raise PopulationPredictorResidualV2Error("V2 payload size is outside admitted bounds")
    decoded_hasher = hashlib.sha256()
    reset_count = 0
    previous0: np.ndarray | None = None
    previous1: np.ndarray | None = None
    with path.open("rb") as handle:
        raw_prefix = handle.read(PREFIX.size)
        if len(raw_prefix) != PREFIX.size:
            raise PopulationPredictorResidualV2Error("V2 prefix is truncated")
        (
            magic,
            version,
            content_codec,
            pair_count,
            height,
            width,
            channels,
            reset_interval,
            declared_root,
        ) = PREFIX.unpack(raw_prefix)
        if (
            magic != MAGIC
            or version != VERSION
            or content_codec != CONTENT_CODEC_TAG
            or not 1 <= pair_count <= MAX_PAIRS
            or not 1 <= height <= MAX_HEIGHT
            or not 1 <= width <= MAX_WIDTH
            or channels != 3
            or not 1 <= reset_interval <= pair_count
        ):
            raise PopulationPredictorResidualV2Error("V2 prefix contract drift")
        plane_bytes = height * width * channels
        for pair_id in range(pair_count):
            raw_header = handle.read(PAIR_PREFIX.size)
            if len(raw_header) != PAIR_PREFIX.size:
                raise PopulationPredictorResidualV2Error(f"pair {pair_id} header is truncated")
            (
                recorded_pair_id,
                reset_flag,
                length0,
                length1,
                payload0_sha,
                payload1_sha,
                frame0_sha,
                frame1_sha,
            ) = PAIR_PREFIX.unpack(raw_header)
            expected_reset = pair_id % reset_interval == 0
            if recorded_pair_id != pair_id or reset_flag not in (0, 1):
                raise PopulationPredictorResidualV2Error("V2 pair order/reset flag drift")
            if bool(reset_flag) != expected_reset or length0 < 1 or length1 < 1:
                raise PopulationPredictorResidualV2Error("V2 reset schedule/length drift")
            remaining_bytes = payload_bytes - handle.tell()
            if length0 + length1 > remaining_bytes:
                raise PopulationPredictorResidualV2Error("V2 pair body lengths exceed the bounded payload")
            payload0 = handle.read(length0)
            payload1 = handle.read(length1)
            if len(payload0) != length0 or len(payload1) != length1:
                raise PopulationPredictorResidualV2Error(f"pair {pair_id} body is truncated")
            if _sha(payload0) != payload0_sha or _sha(payload1) != payload1_sha:
                raise PopulationPredictorResidualV2Error(f"pair {pair_id} component hash custody failure")
            decoded0 = np.frombuffer(
                _decompress(payload0, expected_bytes=plane_bytes, label=f"pair {pair_id} frame0"),
                dtype=np.uint8,
            ).reshape(height, width, channels)
            decoded1 = np.frombuffer(
                _decompress(payload1, expected_bytes=plane_bytes, label=f"pair {pair_id} frame1"),
                dtype=np.uint8,
            ).reshape(height, width, channels)
            if expected_reset:
                frame0 = np.ascontiguousarray(decoded0)
                frame1 = np.ascontiguousarray(decoded1)
            else:
                if previous0 is None or previous1 is None:
                    raise PopulationPredictorResidualV2Error("V2 prediction state is absent")
                frame0 = _modulo_add(previous0, decoded0)
                frame1 = _modulo_add(previous1, decoded1)
            frame0_bytes = frame0.tobytes(order="C")
            frame1_bytes = frame1.tobytes(order="C")
            if _sha(frame0_bytes) != frame0_sha or _sha(frame1_bytes) != frame1_sha:
                raise PopulationPredictorResidualV2Error(f"pair {pair_id} reconstructed frame hash custody failure")
            decoded_hasher.update(frame0_bytes)
            decoded_hasher.update(frame1_bytes)
            on_pair(pair_id, frame0, frame1)
            previous0, previous1 = frame0, frame1
            reset_count += int(expected_reset)
        if handle.read(1):
            raise PopulationPredictorResidualV2Error("V2 payload has trailing bytes")
    if decoded_hasher.digest() != declared_root:
        raise PopulationPredictorResidualV2Error("V2 decoded population root drift")
    return PopulationV2Receipt(
        schema=DECODE_RECEIPT_SCHEMA,
        codec_id=CODEC_ID,
        path=str(path),
        payload_bytes=payload_bytes,
        payload_sha256=_sha_file(path),
        decoded_pair_major_sha256=decoded_hasher.hexdigest(),
        pair_count=pair_count,
        height=height,
        width=width,
        channels=channels,
        reset_interval=reset_interval,
        reset_count=reset_count,
        peak_retained_pairs=2,
    )


def verify_population_identity(
    payload_path: Path,
    *,
    expected_frame_sha256: Mapping[tuple[int, int], str],
) -> PopulationV2Receipt:
    """Decode and prove every pair/slot against caller-owned exact hashes."""

    observed = 0

    def check(pair_id: int, frame0: np.ndarray, frame1: np.ndarray) -> None:
        nonlocal observed
        for slot, frame in enumerate((frame0, frame1)):
            key = (pair_id, slot)
            expected = expected_frame_sha256.get(key)
            if (
                type(expected) is not str
                or len(expected) != 64
                or hashlib.sha256(frame.tobytes(order="C")).hexdigest() != expected
            ):
                raise PopulationPredictorResidualV2Error(f"pair {pair_id} frame {slot} identity callback failed")
        observed += 1

    receipt = decode_population_predictor_residual_v2(payload_path, on_pair=check)
    if observed != receipt.pair_count or len(expected_frame_sha256) != receipt.pair_count * 2:
        raise PopulationPredictorResidualV2Error("identity hash population is incomplete")
    return receipt


__all__ = [
    "CODEC_ID",
    "CONTENT_CODEC_ID",
    "DECODE_RECEIPT_SCHEMA",
    "MAGIC",
    "PAIR_PREFIX",
    "PREFIX",
    "VERSION",
    "PopulationPair",
    "PopulationPredictorResidualV2Error",
    "PopulationV2Receipt",
    "decode_population_predictor_residual_v2",
    "encode_population_predictor_residual_v2",
    "verify_population_identity",
]
