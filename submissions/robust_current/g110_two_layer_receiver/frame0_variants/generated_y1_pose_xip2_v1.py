# SPDX-License-Identifier: MIT
"""Public generated-Y1 conditional pose decoder for the G110 product.

The counted packet stores one exact G105 semantic-Y1 packet and one XIP2
quantized SE(3) trajectory.  The source of the frame-0 warp is the *actual*
camera Y1 emitted by the public semantic receiver and its V10 factor-2
realizer.  Frame 0 is the native-camera homography warp of that source,
rounded once at the stored-video uint8 boundary.  There is no scorer-grid
rounding and no second V10 realization after the warp.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from bisect import bisect_right
from typing import Final

import brotli
import numpy as np

VARIANT_ID: Final = "tac.semantic_root_y0.generated_y1_pose_xip2.v1"
MAGIC: Final = b"G110PC01"
VERSION: Final = 1
PAIR_COUNT: Final = 600
SCORER_H: Final = 384
SCORER_W: Final = 512
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
CHANNELS: Final = 3
MAX_PACKET_BYTES: Final = 2_100_000
MAX_SEMANTIC_PACKET_BYTES: Final = 2_000_000
MAX_XIP2_BYTES: Final = 100_000
SEMANTIC_MAGIC: Final = b"SV9Y1V1\0"
XIP2_MAGIC: Final = b"XIP2"
FINAL_Y1_DOMAIN: Final = b"G110_FINAL_Y1_N600_V1\x00"

_HEADER = struct.Struct(">8sBBHIId32sI")
_ST_BITS = 32
_FULL = 1 << _ST_BITS
_HALF = _FULL >> 1
_QTR = _HALF >> 1
_TQTR = _QTR * 3
_XI_FX = 910.0
_XI_CX = 582.0
_XI_CY = 437.0
_XI_D = 1.22
_XI_EPS = 1.0e-6


class GeneratedY1PoseVariantError(ValueError):
    """The typed pose packet or exact public render failed closed."""


def accepts_packet(packet: bytes) -> bool:
    return type(packet) is bytes and packet.startswith(MAGIC)


def _ar_decode(encoded: bytes, count: int, frequencies: list[int]) -> list[int]:
    if count <= 0:
        return []
    if (
        not frequencies
        or any(type(value) is not int or value <= 0 for value in frequencies)
    ):
        raise GeneratedY1PoseVariantError("XIP2 arithmetic model is invalid")
    cumulative = [0]
    for frequency in frequencies:
        cumulative.append(cumulative[-1] + frequency)
    total = cumulative[-1]
    if total <= 0 or total >= (1 << 31):
        raise GeneratedY1PoseVariantError("XIP2 arithmetic total is outside bound")
    byte_index = 0
    bit_index = 0

    def bit() -> int:
        nonlocal byte_index, bit_index
        if byte_index >= len(encoded):
            return 0
        value = (encoded[byte_index] >> (7 - bit_index)) & 1
        bit_index += 1
        if bit_index == 8:
            bit_index = 0
            byte_index += 1
        return value

    low = 0
    high = _FULL - 1
    code = 0
    for _ in range(_ST_BITS):
        code = (code << 1) | bit()
    output: list[int] = []
    for _ in range(count):
        width = high - low + 1
        scaled = ((code - low + 1) * total - 1) // width
        symbol = bisect_right(cumulative, scaled) - 1
        if not 0 <= symbol < len(frequencies):
            raise GeneratedY1PoseVariantError(
                "XIP2 arithmetic stream selected an invalid symbol"
            )
        output.append(symbol)
        high = low + (width * cumulative[symbol + 1] // total) - 1
        low = low + (width * cumulative[symbol] // total)
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                code -= _HALF
            elif low >= _QTR and high < _TQTR:
                low -= _QTR
                high -= _QTR
                code -= _QTR
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            code = (code << 1) | bit()
    return output


def _take(blob: bytes, offset: int, length: int, *, label: str) -> tuple[bytes, int]:
    if length < 0 or offset < 0 or offset + length > len(blob):
        raise GeneratedY1PoseVariantError(f"{label} is truncated")
    return blob[offset : offset + length], offset + length


def _parse_xip2(blob: bytes) -> np.ndarray:
    if (
        type(blob) is not bytes
        or not 8 + 6 * 4 <= len(blob) <= MAX_XIP2_BYTES
        or not blob.startswith(XIP2_MAGIC)
    ):
        raise GeneratedY1PoseVariantError("pose payload is not bounded XIP2")
    offset = 4
    try:
        coder, pairs, dimensions = struct.unpack_from("<BHB", blob, offset)
    except struct.error as exc:
        raise GeneratedY1PoseVariantError("XIP2 header is truncated") from exc
    offset += 4
    if (pairs, dimensions) != (PAIR_COUNT, 6) or coder not in {0, 1}:
        raise GeneratedY1PoseVariantError(
            "XIP2 changes n600/6D geometry or uses an unsupported coder"
        )
    scales_raw, offset = _take(blob, offset, dimensions * 4, label="XIP2 scales")
    scales = np.frombuffer(scales_raw, dtype="<f4").copy()
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise GeneratedY1PoseVariantError("XIP2 scales must be finite positive float32")
    if coder == 0:
        q_raw, offset = _take(
            blob,
            offset,
            pairs * dimensions * 2,
            label="XIP2 raw trajectory",
        )
        q = np.frombuffer(q_raw, dtype="<i2").reshape(pairs, dimensions).copy()
    else:
        columns: list[np.ndarray] = []
        for column_id in range(dimensions):
            fixed_raw, offset = _take(
                blob,
                offset,
                16,
                label=f"XIP2 channel {column_id} header",
            )
            seed, low_delta, high_delta, model_length = struct.unpack(
                "<iiiI", fixed_raw
            )
            if (
                high_delta < low_delta
                or high_delta - low_delta > 65_535
                or model_length > MAX_XIP2_BYTES
            ):
                raise GeneratedY1PoseVariantError(
                    f"XIP2 channel {column_id} model bounds are invalid"
                )
            model, offset = _take(
                blob,
                offset,
                model_length,
                label=f"XIP2 channel {column_id} model",
            )
            stream_length_raw, offset = _take(
                blob,
                offset,
                4,
                label=f"XIP2 channel {column_id} stream length",
            )
            (stream_length,) = struct.unpack("<I", stream_length_raw)
            if stream_length > MAX_XIP2_BYTES:
                raise GeneratedY1PoseVariantError(
                    f"XIP2 channel {column_id} stream exceeds bound"
                )
            stream, offset = _take(
                blob,
                offset,
                stream_length,
                label=f"XIP2 channel {column_id} stream",
            )
            if high_delta > low_delta:
                if not model or not stream:
                    raise GeneratedY1PoseVariantError(
                        f"XIP2 channel {column_id} nonconstant stream is empty"
                    )
                try:
                    decoded_model = brotli.decompress(model)
                except brotli.error as exc:
                    raise GeneratedY1PoseVariantError(
                        f"XIP2 channel {column_id} model is not brotli"
                    ) from exc
                alphabet = high_delta - low_delta + 1
                if len(decoded_model) != alphabet * 4:
                    raise GeneratedY1PoseVariantError(
                        f"XIP2 channel {column_id} model length differs"
                    )
                frequencies = (
                    np.frombuffer(decoded_model, dtype="<u4")
                    .astype(np.int64)
                    .tolist()
                )
                deltas = (
                    np.asarray(
                        _ar_decode(stream, pairs - 1, frequencies),
                        dtype=np.int64,
                    )
                    + low_delta
                )
            else:
                if model or stream:
                    raise GeneratedY1PoseVariantError(
                        f"XIP2 channel {column_id} constant stream is noncanonical"
                    )
                deltas = np.full(pairs - 1, low_delta, dtype=np.int64)
            column = np.empty(pairs, dtype=np.int64)
            column[0] = seed
            column[1:] = seed + np.cumsum(deltas, dtype=np.int64)
            if np.any(column < -32_768) or np.any(column > 32_767):
                raise GeneratedY1PoseVariantError(
                    f"XIP2 channel {column_id} leaves int16 range"
                )
            columns.append(column)
        q = np.stack(columns, axis=1).astype("<i2")
    if offset != len(blob):
        raise GeneratedY1PoseVariantError("XIP2 has unconsumed trailing bytes")
    return q.astype(np.float64) * scales.astype(np.float64)


def _homographies_from_xi(xi: np.ndarray, pitch: float) -> np.ndarray:
    rho = xi[:, :3]
    omega = xi[:, 3:]
    a, b, c = omega[:, 0], omega[:, 1], omega[:, 2]
    zeros = np.zeros_like(a)
    skew = np.stack(
        [
            np.stack([zeros, -c, b], axis=-1),
            np.stack([c, zeros, -a], axis=-1),
            np.stack([-b, a, zeros], axis=-1),
        ],
        axis=-2,
    )
    skew2 = skew @ skew
    theta2 = np.sum(omega * omega, axis=-1)
    theta = np.sqrt(np.maximum(theta2, 0.0))
    small = theta < _XI_EPS
    theta_safe = np.maximum(theta, _XI_EPS)
    theta2_safe = np.maximum(theta2, _XI_EPS * _XI_EPS)
    theta3_safe = np.maximum(theta**3, _XI_EPS**3)
    factor_a = np.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        np.sin(theta) / theta_safe,
    )
    factor_b = np.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - np.cos(theta)) / theta2_safe,
    )
    factor_c = np.where(
        small,
        1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0,
        (theta - np.sin(theta)) / theta3_safe,
    )
    eye = np.broadcast_to(np.eye(3, dtype=np.float64), skew.shape).copy()
    rotation = (
        eye
        + factor_a[..., None, None] * skew
        + factor_b[..., None, None] * skew2
    )
    jacobian = (
        eye
        + factor_b[..., None, None] * skew
        + factor_c[..., None, None] * skew2
    )
    translation = (jacobian @ rho[..., None])[..., 0]
    intrinsics = np.array(
        [
            [_XI_FX, 0.0, _XI_CX],
            [0.0, _XI_FX, _XI_CY],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    inverse_intrinsics = np.linalg.inv(intrinsics)
    normal = np.array(
        [0.0, -math.cos(pitch), -math.sin(pitch)],
        dtype=np.float64,
    )
    plane_map = rotation - (
        translation[..., :, None] * normal[None, None, :]
    ) / _XI_D
    homographies = (
        intrinsics[None] @ plane_map @ inverse_intrinsics[None]
    )
    if not np.all(np.isfinite(homographies)):
        raise GeneratedY1PoseVariantError("derived homography population is non-finite")
    return homographies


def parse_packet(packet: bytes) -> dict[str, object]:
    if (
        type(packet) is not bytes
        or not _HEADER.size < len(packet) <= MAX_PACKET_BYTES
    ):
        raise GeneratedY1PoseVariantError("pose packet must be bounded exact bytes")
    try:
        (
            magic,
            version,
            flags,
            pairs,
            semantic_length,
            xip2_length,
            pitch,
            final_y1_binding,
            expected_crc,
        ) = _HEADER.unpack_from(packet)
    except struct.error as exc:
        raise GeneratedY1PoseVariantError("pose packet header is truncated") from exc
    if (
        magic != MAGIC
        or version != VERSION
        or flags != 0
        or pairs != PAIR_COUNT
        or not 0 < semantic_length <= MAX_SEMANTIC_PACKET_BYTES
        or not 0 < xip2_length <= MAX_XIP2_BYTES
        or not math.isfinite(pitch)
        or abs(pitch) > math.pi / 2.0
        or _HEADER.size + semantic_length + xip2_length != len(packet)
    ):
        raise GeneratedY1PoseVariantError("pose header changes the closed n600 ABI")
    body = packet[_HEADER.size :]
    if zlib.crc32(body) & 0xFFFFFFFF != expected_crc:
        raise GeneratedY1PoseVariantError("pose packet body CRC32 mismatch")
    semantic = body[:semantic_length]
    xip2 = body[semantic_length:]
    if not semantic.startswith(SEMANTIC_MAGIC):
        raise GeneratedY1PoseVariantError(
            "generated-Y1 pose packet requires the exact G105 semantic variant"
        )
    xi = _parse_xip2(xip2)
    homographies = _homographies_from_xi(xi, float(pitch))
    return {
        "packet": packet,
        "semantic": semantic,
        "xip2": xip2,
        "pitch": float(pitch),
        "xi": xi,
        "homographies": homographies,
        "final_y1_binding": bytes(final_y1_binding),
    }


def semantic_packet(state: dict[str, object]) -> bytes:
    semantic = state.get("semantic")
    if type(semantic) is not bytes:
        raise GeneratedY1PoseVariantError("parsed pose state lost semantic packet")
    return semantic


def _warp_camera(source: np.ndarray, homography: np.ndarray) -> np.ndarray:
    src = np.asarray(source, dtype=np.float64)
    if src.shape != (CAMERA_H, CAMERA_W, CHANNELS):
        raise GeneratedY1PoseVariantError("pose source is not camera RGB")
    flat = src.reshape(-1, CHANNELS)
    columns, rows = np.meshgrid(
        np.arange(CAMERA_W),
        np.arange(CAMERA_H),
    )
    grid = np.stack(
        [columns.ravel(), rows.ravel(), np.ones(CAMERA_H * CAMERA_W)],
        axis=0,
    ).astype(np.float64)
    try:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            source_h = np.linalg.inv(
                np.asarray(homography, dtype=np.float64)
            ) @ grid
            depth = source_h[2]
            source_x = source_h[0] / depth
            source_y = source_h[1] / depth
    except np.linalg.LinAlgError as exc:
        raise GeneratedY1PoseVariantError("pose homography is singular") from exc
    valid = (
        np.isfinite(source_x)
        & np.isfinite(source_y)
        & (depth > 0.0)
        & (source_x >= 0.0)
        & (source_x <= CAMERA_W - 1)
        & (source_y >= 0.0)
        & (source_y <= CAMERA_H - 1)
    )
    clipped_x = np.clip(source_x, 0.0, CAMERA_W - 1)
    clipped_y = np.clip(source_y, 0.0, CAMERA_H - 1)
    x0 = np.floor(clipped_x).astype(np.int64)
    y0 = np.floor(clipped_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, CAMERA_W - 1)
    y1 = np.minimum(y0 + 1, CAMERA_H - 1)
    wx = (clipped_x - x0)[:, None]
    wy = (clipped_y - y0)[:, None]
    top = (
        flat[y0 * CAMERA_W + x0] * (1.0 - wx)
        + flat[y0 * CAMERA_W + x1] * wx
    )
    bottom = (
        flat[y1 * CAMERA_W + x0] * (1.0 - wx)
        + flat[y1 * CAMERA_W + x1] * wx
    )
    sampled = top * (1.0 - wy) + bottom * wy
    warped = np.where(valid[:, None], sampled, flat).reshape(
        CAMERA_H,
        CAMERA_W,
        CHANNELS,
    )
    return np.ascontiguousarray(
        np.clip(np.rint(warped), 0.0, 255.0).astype(np.uint8)
    )


def render_camera_y0(
    state: dict[str, object],
    pair_id: int,
    scorer_y1: np.ndarray,
    camera_y1: np.ndarray,
) -> np.ndarray:
    """Warp the actual final public camera Y1; trained precompile Y1 is never read."""

    if type(pair_id) is not int or not 0 <= pair_id < PAIR_COUNT:
        raise GeneratedY1PoseVariantError("pair_id is outside exact n600")
    scorer = np.asarray(scorer_y1)
    camera = np.asarray(camera_y1)
    if scorer.dtype != np.uint8 or scorer.shape != (
        SCORER_H,
        SCORER_W,
        CHANNELS,
    ):
        raise GeneratedY1PoseVariantError("pose conditional lost final scorer Y1 custody")
    if camera.dtype != np.uint8 or camera.shape != (
        CAMERA_H,
        CAMERA_W,
        CHANNELS,
    ):
        raise GeneratedY1PoseVariantError("pose conditional lost final camera Y1 custody")
    homographies = state.get("homographies")
    if type(homographies) is not np.ndarray or homographies.shape != (
        PAIR_COUNT,
        3,
        3,
    ):
        raise GeneratedY1PoseVariantError("parsed pose state lost homographies")
    return _warp_camera(np.ascontiguousarray(camera), homographies[pair_id])


def verify_final_y1_population(
    state: dict[str, object],
    observed_digest: bytes,
) -> None:
    expected = state.get("final_y1_binding")
    semantic = state.get("semantic")
    observed = (
        hashlib.sha256(
            FINAL_Y1_DOMAIN
            + hashlib.sha256(semantic).digest()
            + observed_digest
        ).digest()
        if type(semantic) is bytes and type(observed_digest) is bytes
        else None
    )
    if (
        type(expected) is not bytes
        or len(expected) != 32
        or type(observed_digest) is not bytes
        or len(observed_digest) != 32
        or observed != expected
    ):
        raise GeneratedY1PoseVariantError(
            "final G105 Y1 population binding differs from post-G105 refit custody"
        )
