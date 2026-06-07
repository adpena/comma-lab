# SPDX-License-Identifier: MIT
"""Receiver-consumed HiNeRV target-region action sidecar grammar.

The payload is charged because it lives inside the HIV1 meta blob.  It is not a
training proxy: parsed receivers apply these uint8 pixel actions after model
render and before scorer/raw-output surfaces.
"""

from __future__ import annotations

import base64
import hashlib
import struct
import zlib
from dataclasses import dataclass
from typing import Any

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

TARGET_REGION_ACTION_META_KEY = "_target_region_actions_v1_b64"
TARGET_REGION_ACTION_MAGIC = b"HTRA1"
TARGET_REGION_ACTION_COMPRESSED_MAGIC = b"HTRZ1"
TARGET_REGION_ACTION_BROTLI_MAGIC = b"HTRB1"
TARGET_REGION_ACTION_SPLIT_BROTLI_MAGIC = b"HTRS1"
TARGET_REGION_ACTION_TILE_BROTLI_MAGIC = b"HTRT1"
TARGET_REGION_ACTION_SCHEMA = "hi_nerv_target_region_archive_actions.v1"
_HEADER_FMT = "<5sH"
_COMPRESSED_HEADER_FMT = "<5sI"
_ACTION_HEADER_FMT = "<HBBHHI"
_SPLIT_ACTION_HEADER_FMT = "<HBBHHIII"
_TILE_ACTION_HEADER_FMT = "<HBBHHIII"
_TILE_RECORD_FMT = "<HH"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_COMPRESSED_HEADER_SIZE = struct.calcsize(_COMPRESSED_HEADER_FMT)
_ACTION_HEADER_SIZE = struct.calcsize(_ACTION_HEADER_FMT)
_SPLIT_ACTION_HEADER_SIZE = struct.calcsize(_SPLIT_ACTION_HEADER_FMT)
_TILE_ACTION_HEADER_SIZE = struct.calcsize(_TILE_ACTION_HEADER_FMT)
_TILE_RECORD_SIZE = struct.calcsize(_TILE_RECORD_FMT)
_TILE_BROTLI_SIZE = 16


@dataclass(frozen=True)
class TargetRegionPixelAction:
    """One receiver-visible sparse uint8 paint action."""

    pair_index: int
    frame_index: int
    height: int
    width: int
    yx: np.ndarray
    rgb_u8: np.ndarray

    def __post_init__(self) -> None:
        yx = np.asarray(self.yx, dtype=np.uint16)
        rgb = np.asarray(self.rgb_u8, dtype=np.uint8)
        if int(self.frame_index) not in (0, 1):
            raise ValueError(f"frame_index must be 0 or 1; got {self.frame_index}")
        if int(self.pair_index) < 0:
            raise ValueError(f"pair_index must be non-negative; got {self.pair_index}")
        if int(self.height) <= 0 or int(self.width) <= 0:
            raise ValueError(f"height/width must be positive; got {self.height}x{self.width}")
        if yx.ndim != 2 or yx.shape[1] != 2:
            raise ValueError(f"yx must have shape (N,2); got {yx.shape}")
        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError(f"rgb_u8 must have shape (N,3); got {rgb.shape}")
        if yx.shape[0] != rgb.shape[0]:
            raise ValueError(f"yx/rgb row mismatch: {yx.shape[0]} != {rgb.shape[0]}")
        if yx.shape[0] <= 0:
            raise ValueError("target-region action must contain at least one pixel")
        if int(np.max(yx[:, 0])) >= int(self.height) or int(np.max(yx[:, 1])) >= int(self.width):
            raise ValueError("target-region action coordinate exceeds declared geometry")
        if np.unique(yx, axis=0).shape[0] != yx.shape[0]:
            raise ValueError("target-region action support must be duplicate-free")
        object.__setattr__(self, "yx", np.ascontiguousarray(yx))
        object.__setattr__(self, "rgb_u8", np.ascontiguousarray(rgb))

    @property
    def pixel_count(self) -> int:
        return int(self.yx.shape[0])


def encode_target_region_actions(actions: list[TargetRegionPixelAction]) -> bytes:
    """Encode actions to the deterministic receiver binary grammar."""

    if len(actions) > 65535:
        raise ValueError(f"too many target-region actions: {len(actions)}")
    chunks = [struct.pack(_HEADER_FMT, TARGET_REGION_ACTION_MAGIC, len(actions))]
    for action in actions:
        if action.pair_index > 65535:
            raise ValueError(f"pair_index exceeds u16 grammar: {action.pair_index}")
        if action.height > 65535 or action.width > 65535:
            raise ValueError(f"geometry exceeds u16 grammar: {action.height}x{action.width}")
        chunks.append(
            struct.pack(
                _ACTION_HEADER_FMT,
                int(action.pair_index),
                int(action.frame_index),
                0,
                int(action.height),
                int(action.width),
                int(action.pixel_count),
            )
        )
        chunks.append(np.asarray(action.yx, dtype="<u2").tobytes(order="C"))
        chunks.append(np.asarray(action.rgb_u8, dtype=np.uint8).tobytes(order="C"))
    return b"".join(chunks)


def encode_target_region_actions_payload(actions: list[TargetRegionPixelAction]) -> bytes:
    raw = encode_target_region_actions(actions)
    zlib_compressed = zlib.compress(raw, level=9)
    brotli_compressed = brotli.compress(raw, quality=11)
    split_brotli = _encode_split_brotli_target_region_actions(actions)
    tile_brotli = _try_encode_tile_brotli_target_region_actions(actions)
    candidates = [
        (
            len(raw),
            raw,
        ),
        (
            len(zlib_compressed) + _COMPRESSED_HEADER_SIZE,
            struct.pack(
                _COMPRESSED_HEADER_FMT,
                TARGET_REGION_ACTION_COMPRESSED_MAGIC,
                len(raw),
            )
            + zlib_compressed,
        ),
        (
            len(brotli_compressed) + _COMPRESSED_HEADER_SIZE,
            struct.pack(
                _COMPRESSED_HEADER_FMT,
                TARGET_REGION_ACTION_BROTLI_MAGIC,
                len(raw),
            )
            + brotli_compressed,
        ),
        (
            len(split_brotli),
            split_brotli,
        ),
    ]
    if tile_brotli is not None:
        candidates.append((len(tile_brotli), tile_brotli))
    _size, payload = min(candidates, key=lambda item: item[0])
    if payload is raw:
        return raw
    return payload


def _try_encode_tile_brotli_target_region_actions(
    actions: list[TargetRegionPixelAction],
) -> bytes | None:
    try:
        return _encode_tile_brotli_target_region_actions(actions)
    except ValueError:
        return None


def _encode_tile_brotli_target_region_actions(
    actions: list[TargetRegionPixelAction],
) -> bytes:
    if len(actions) > 65535:
        raise ValueError(f"too many target-region actions: {len(actions)}")
    chunks = [struct.pack(_HEADER_FMT, TARGET_REGION_ACTION_TILE_BROTLI_MAGIC, len(actions))]
    for action in actions:
        if action.pair_index > 65535:
            raise ValueError(f"pair_index exceeds u16 grammar: {action.pair_index}")
        if action.height > 65535 or action.width > 65535:
            raise ValueError(f"geometry exceeds u16 grammar: {action.height}x{action.width}")
        mask = _action_support_mask(action)
        yx = _canonical_mask_yx(mask)
        if yx.shape != action.yx.shape or not np.array_equal(yx, action.yx):
            raise ValueError("tile-brotli grammar requires canonical row-major action order")
        tile_payload = _encode_tile_support_payload(mask, tile_size=_TILE_BROTLI_SIZE)
        rgb_payload = np.asarray(action.rgb_u8, dtype=np.uint8).tobytes(order="C")
        tile_compressed = brotli.compress(tile_payload, quality=11)
        rgb_compressed = brotli.compress(rgb_payload, quality=11)
        chunks.append(
            struct.pack(
                _TILE_ACTION_HEADER_FMT,
                int(action.pair_index),
                int(action.frame_index),
                _TILE_BROTLI_SIZE,
                int(action.height),
                int(action.width),
                int(action.pixel_count),
                len(tile_compressed),
                len(rgb_compressed),
            )
        )
        chunks.append(tile_compressed)
        chunks.append(rgb_compressed)
    return b"".join(chunks)


def _action_support_mask(action: TargetRegionPixelAction) -> np.ndarray:
    mask = np.zeros((int(action.height), int(action.width)), dtype=bool)
    y = action.yx[:, 0].astype(np.int64, copy=False)
    x = action.yx[:, 1].astype(np.int64, copy=False)
    if len({(int(yy), int(xx)) for yy, xx in zip(y, x, strict=True)}) != action.pixel_count:
        raise ValueError("tile-brotli grammar requires duplicate-free support")
    mask[y, x] = True
    return mask


def _canonical_mask_yx(mask: np.ndarray) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(src)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.uint16)
    return np.ascontiguousarray(np.stack([ys, xs], axis=1).astype(np.uint16))


def _encode_tile_support_payload(mask: np.ndarray, *, tile_size: int) -> bytes:
    records: list[bytes] = []
    src = np.asarray(mask, dtype=bool)
    for y0 in range(0, src.shape[0], tile_size):
        for x0 in range(0, src.shape[1], tile_size):
            block = src[y0 : y0 + tile_size, x0 : x0 + tile_size]
            if not np.any(block):
                continue
            packed = np.packbits(block.astype(np.uint8).reshape(-1), bitorder="little")
            records.append(struct.pack(_TILE_RECORD_FMT, int(y0), int(x0)) + packed.tobytes(order="C"))
    if len(records) > 65535:
        raise ValueError(f"too many tile support records: {len(records)}")
    return struct.pack("<H", len(records)) + b"".join(records)


def _encode_split_brotli_target_region_actions(
    actions: list[TargetRegionPixelAction],
) -> bytes:
    if len(actions) > 65535:
        raise ValueError(f"too many target-region actions: {len(actions)}")
    chunks = [struct.pack(_HEADER_FMT, TARGET_REGION_ACTION_SPLIT_BROTLI_MAGIC, len(actions))]
    for action in actions:
        if action.pair_index > 65535:
            raise ValueError(f"pair_index exceeds u16 grammar: {action.pair_index}")
        if action.height > 65535 or action.width > 65535:
            raise ValueError(f"geometry exceeds u16 grammar: {action.height}x{action.width}")
        coord_payload = np.asarray(action.yx, dtype="<u2").tobytes(order="C")
        rgb_payload = np.asarray(action.rgb_u8, dtype=np.uint8).tobytes(order="C")
        coord_compressed = brotli.compress(coord_payload, quality=11)
        rgb_compressed = brotli.compress(rgb_payload, quality=11)
        chunks.append(
            struct.pack(
                _SPLIT_ACTION_HEADER_FMT,
                int(action.pair_index),
                int(action.frame_index),
                0,
                int(action.height),
                int(action.width),
                int(action.pixel_count),
                len(coord_compressed),
                len(rgb_compressed),
            )
        )
        chunks.append(coord_compressed)
        chunks.append(rgb_compressed)
    return b"".join(chunks)


def _decode_raw_target_region_actions(blob: bytes) -> list[TargetRegionPixelAction]:
    if len(blob) < _HEADER_SIZE:
        raise ValueError("target-region action payload too short")
    magic, action_count = struct.unpack(_HEADER_FMT, blob[:_HEADER_SIZE])
    if magic != TARGET_REGION_ACTION_MAGIC:
        raise ValueError(f"bad target-region action magic: {magic!r}")
    offset = _HEADER_SIZE
    actions: list[TargetRegionPixelAction] = []
    for _ in range(int(action_count)):
        if offset + _ACTION_HEADER_SIZE > len(blob):
            raise ValueError("truncated target-region action header")
        pair_index, frame_index, reserved, height, width, pixel_count = struct.unpack(
            _ACTION_HEADER_FMT,
            blob[offset : offset + _ACTION_HEADER_SIZE],
        )
        offset += _ACTION_HEADER_SIZE
        if reserved != 0:
            raise ValueError(f"target-region action reserved byte must be 0; got {reserved}")
        coord_bytes = int(pixel_count) * 4
        rgb_bytes = int(pixel_count) * 3
        if offset + coord_bytes + rgb_bytes > len(blob):
            raise ValueError("truncated target-region action pixel payload")
        yx = np.frombuffer(blob[offset : offset + coord_bytes], dtype="<u2").reshape(
            int(pixel_count),
            2,
        )
        offset += coord_bytes
        rgb = np.frombuffer(blob[offset : offset + rgb_bytes], dtype=np.uint8).reshape(
            int(pixel_count),
            3,
        )
        offset += rgb_bytes
        actions.append(
            TargetRegionPixelAction(
                pair_index=int(pair_index),
                frame_index=int(frame_index),
                height=int(height),
                width=int(width),
                yx=np.array(yx, copy=True),
                rgb_u8=np.array(rgb, copy=True),
            )
        )
    if offset != len(blob):
        raise ValueError("target-region action payload has trailing bytes")
    return actions


def _decode_split_brotli_target_region_actions(
    blob: bytes,
) -> list[TargetRegionPixelAction]:
    if len(blob) < _HEADER_SIZE:
        raise ValueError("split-brotli target-region action payload too short")
    magic, action_count = struct.unpack(_HEADER_FMT, blob[:_HEADER_SIZE])
    if magic != TARGET_REGION_ACTION_SPLIT_BROTLI_MAGIC:
        raise ValueError(f"bad split-brotli target-region action magic: {magic!r}")
    offset = _HEADER_SIZE
    actions: list[TargetRegionPixelAction] = []
    for _ in range(int(action_count)):
        if offset + _SPLIT_ACTION_HEADER_SIZE > len(blob):
            raise ValueError("truncated split-brotli target-region action header")
        (
            pair_index,
            frame_index,
            reserved,
            height,
            width,
            pixel_count,
            coord_compressed_bytes,
            rgb_compressed_bytes,
        ) = struct.unpack(
            _SPLIT_ACTION_HEADER_FMT,
            blob[offset : offset + _SPLIT_ACTION_HEADER_SIZE],
        )
        offset += _SPLIT_ACTION_HEADER_SIZE
        if reserved != 0:
            raise ValueError(
                f"split-brotli target-region action reserved byte must be 0; got {reserved}"
            )
        coord_end = offset + int(coord_compressed_bytes)
        rgb_end = coord_end + int(rgb_compressed_bytes)
        if rgb_end > len(blob):
            raise ValueError("truncated split-brotli target-region action payload")
        try:
            coord_payload = brotli.decompress(blob[offset:coord_end])
            rgb_payload = brotli.decompress(blob[coord_end:rgb_end])
        except brotli.error as exc:
            raise ValueError(f"bad split-brotli target-region action payload: {exc}") from exc
        offset = rgb_end
        expected_coord_bytes = int(pixel_count) * 4
        expected_rgb_bytes = int(pixel_count) * 3
        if len(coord_payload) != expected_coord_bytes:
            raise ValueError(
                "split-brotli target-region coordinate size mismatch: "
                f"{len(coord_payload)} != {expected_coord_bytes}"
            )
        if len(rgb_payload) != expected_rgb_bytes:
            raise ValueError(
                "split-brotli target-region rgb size mismatch: "
                f"{len(rgb_payload)} != {expected_rgb_bytes}"
            )
        yx = np.frombuffer(coord_payload, dtype="<u2").reshape(int(pixel_count), 2)
        rgb = np.frombuffer(rgb_payload, dtype=np.uint8).reshape(int(pixel_count), 3)
        actions.append(
            TargetRegionPixelAction(
                pair_index=int(pair_index),
                frame_index=int(frame_index),
                height=int(height),
                width=int(width),
                yx=np.array(yx, copy=True),
                rgb_u8=np.array(rgb, copy=True),
            )
        )
    if offset != len(blob):
        raise ValueError("split-brotli target-region action payload has trailing bytes")
    return actions


def _decode_tile_brotli_target_region_actions(
    blob: bytes,
) -> list[TargetRegionPixelAction]:
    if len(blob) < _HEADER_SIZE:
        raise ValueError("tile-brotli target-region action payload too short")
    magic, action_count = struct.unpack(_HEADER_FMT, blob[:_HEADER_SIZE])
    if magic != TARGET_REGION_ACTION_TILE_BROTLI_MAGIC:
        raise ValueError(f"bad tile-brotli target-region action magic: {magic!r}")
    offset = _HEADER_SIZE
    actions: list[TargetRegionPixelAction] = []
    for _ in range(int(action_count)):
        if offset + _TILE_ACTION_HEADER_SIZE > len(blob):
            raise ValueError("truncated tile-brotli target-region action header")
        (
            pair_index,
            frame_index,
            tile_size,
            height,
            width,
            pixel_count,
            tile_compressed_bytes,
            rgb_compressed_bytes,
        ) = struct.unpack(
            _TILE_ACTION_HEADER_FMT,
            blob[offset : offset + _TILE_ACTION_HEADER_SIZE],
        )
        offset += _TILE_ACTION_HEADER_SIZE
        if tile_size <= 0:
            raise ValueError(f"bad tile-brotli tile size: {tile_size}")
        tile_end = offset + int(tile_compressed_bytes)
        rgb_end = tile_end + int(rgb_compressed_bytes)
        if rgb_end > len(blob):
            raise ValueError("truncated tile-brotli target-region action payload")
        try:
            tile_payload = brotli.decompress(blob[offset:tile_end])
            rgb_payload = brotli.decompress(blob[tile_end:rgb_end])
        except brotli.error as exc:
            raise ValueError(f"bad tile-brotli target-region action payload: {exc}") from exc
        offset = rgb_end
        yx = _decode_tile_support_payload(
            tile_payload,
            height=int(height),
            width=int(width),
            tile_size=int(tile_size),
        )
        if yx.shape[0] != int(pixel_count):
            raise ValueError(
                "tile-brotli target-region support count mismatch: "
                f"{yx.shape[0]} != {int(pixel_count)}"
            )
        expected_rgb_bytes = int(pixel_count) * 3
        if len(rgb_payload) != expected_rgb_bytes:
            raise ValueError(
                "tile-brotli target-region rgb size mismatch: "
                f"{len(rgb_payload)} != {expected_rgb_bytes}"
            )
        rgb = np.frombuffer(rgb_payload, dtype=np.uint8).reshape(int(pixel_count), 3)
        actions.append(
            TargetRegionPixelAction(
                pair_index=int(pair_index),
                frame_index=int(frame_index),
                height=int(height),
                width=int(width),
                yx=yx,
                rgb_u8=np.array(rgb, copy=True),
            )
        )
    if offset != len(blob):
        raise ValueError("tile-brotli target-region action payload has trailing bytes")
    return actions


def _decode_tile_support_payload(
    payload: bytes,
    *,
    height: int,
    width: int,
    tile_size: int,
) -> np.ndarray:
    if len(payload) < 2:
        raise ValueError("tile-brotli support payload too short")
    tile_count = struct.unpack("<H", payload[:2])[0]
    offset = 2
    mask = np.zeros((int(height), int(width)), dtype=bool)
    for _ in range(int(tile_count)):
        if offset + _TILE_RECORD_SIZE > len(payload):
            raise ValueError("truncated tile-brotli support record")
        y0, x0 = struct.unpack(_TILE_RECORD_FMT, payload[offset : offset + _TILE_RECORD_SIZE])
        offset += _TILE_RECORD_SIZE
        if y0 >= height or x0 >= width:
            raise ValueError(f"tile-brotli support tile origin out of range: {(y0, x0)}")
        block_h = min(tile_size, height - int(y0))
        block_w = min(tile_size, width - int(x0))
        packed_len = (block_h * block_w + 7) // 8
        if offset + packed_len > len(payload):
            raise ValueError("truncated tile-brotli support bitmap")
        packed = np.frombuffer(payload[offset : offset + packed_len], dtype=np.uint8)
        offset += packed_len
        bits = np.unpackbits(packed, bitorder="little")[: block_h * block_w]
        block = bits.reshape(block_h, block_w).astype(bool)
        mask[int(y0) : int(y0) + block_h, int(x0) : int(x0) + block_w] |= block
    if offset != len(payload):
        raise ValueError("tile-brotli support payload has trailing bytes")
    return _canonical_mask_yx(mask)


def decode_target_region_actions(blob: bytes) -> list[TargetRegionPixelAction]:
    """Decode the charged receiver binary grammar."""

    if len(blob) >= _HEADER_SIZE:
        magic, _action_count = struct.unpack(_HEADER_FMT, blob[:_HEADER_SIZE])
        if magic == TARGET_REGION_ACTION_TILE_BROTLI_MAGIC:
            return _decode_tile_brotli_target_region_actions(blob)
        if magic == TARGET_REGION_ACTION_SPLIT_BROTLI_MAGIC:
            return _decode_split_brotli_target_region_actions(blob)
    if len(blob) >= _COMPRESSED_HEADER_SIZE:
        magic, raw_size = struct.unpack(
            _COMPRESSED_HEADER_FMT,
            blob[:_COMPRESSED_HEADER_SIZE],
        )
        if magic == TARGET_REGION_ACTION_COMPRESSED_MAGIC:
            try:
                raw = zlib.decompress(blob[_COMPRESSED_HEADER_SIZE:])
            except zlib.error as exc:
                raise ValueError(f"bad compressed target-region action payload: {exc}") from exc
            if len(raw) != int(raw_size):
                raise ValueError(
                    "target-region action decompressed size mismatch: "
                    f"{len(raw)} != {int(raw_size)}"
                )
            return _decode_raw_target_region_actions(raw)
        if magic == TARGET_REGION_ACTION_BROTLI_MAGIC:
            try:
                raw = brotli.decompress(blob[_COMPRESSED_HEADER_SIZE:])
            except brotli.error as exc:
                raise ValueError(f"bad brotli target-region action payload: {exc}") from exc
            if len(raw) != int(raw_size):
                raise ValueError(
                    "target-region action decompressed size mismatch: "
                    f"{len(raw)} != {int(raw_size)}"
                )
            return _decode_raw_target_region_actions(raw)
    return _decode_raw_target_region_actions(blob)


def target_region_action_payload_codec(payload: bytes) -> str:
    if payload.startswith(TARGET_REGION_ACTION_TILE_BROTLI_MAGIC):
        return "tile_brotli_v1"
    if payload.startswith(TARGET_REGION_ACTION_SPLIT_BROTLI_MAGIC):
        return "split_brotli_v1"
    if payload.startswith(TARGET_REGION_ACTION_BROTLI_MAGIC):
        return "brotli_wrapped_v1"
    if payload.startswith(TARGET_REGION_ACTION_COMPRESSED_MAGIC):
        return "zlib_wrapped_v1"
    return "raw_v1"


def target_region_action_support_sha256(actions: list[TargetRegionPixelAction]) -> str:
    h = hashlib.sha256()
    for action in actions:
        h.update(
            struct.pack(
                "<IHHHII",
                int(action.pair_index),
                int(action.frame_index),
                int(action.height),
                int(action.width),
                int(action.pixel_count),
                0,
            )
        )
        h.update(np.asarray(action.yx, dtype="<u2").tobytes(order="C"))
    return h.hexdigest()


def target_region_action_decoded_support_sha256(
    actions: list[TargetRegionPixelAction],
) -> str:
    """Hash the decoded support semantics, independent of coordinate order."""

    h = hashlib.sha256()
    h.update(b"HTRA_DECODED_SUPPORT_CANON_V1")
    support_keys: set[tuple[int, int, int, int, int, int]] = set()
    for action in actions:
        for y, x in np.asarray(action.yx, dtype=np.uint16):
            support_keys.add(
                (
                    int(action.pair_index),
                    int(action.frame_index),
                    int(action.height),
                    int(action.width),
                    int(y),
                    int(x),
                )
            )
    for key in sorted(support_keys):
        h.update(struct.pack("<IBHHHH", *key))
    return h.hexdigest()


def target_region_action_decoded_action_sha256(
    actions: list[TargetRegionPixelAction],
) -> str:
    """Hash final decoded paint semantics, independent of coordinate order."""

    h = hashlib.sha256()
    h.update(b"HTRA_DECODED_ACTION_CANON_V1")
    final_values: dict[tuple[int, int, int, int, int, int], tuple[int, int, int]] = {}
    for action in actions:
        yx = np.asarray(action.yx, dtype=np.uint16)
        rgb = np.asarray(action.rgb_u8, dtype=np.uint8)
        for (y, x), (r, g, b) in zip(yx, rgb, strict=True):
            final_values[
                (
                    int(action.pair_index),
                    int(action.frame_index),
                    int(action.height),
                    int(action.width),
                    int(y),
                    int(x),
                )
            ] = (int(r), int(g), int(b))
    for key, rgb in sorted(final_values.items()):
        h.update(struct.pack("<IBHHHHBBB", *key, *rgb))
    return h.hexdigest()


def _split_brotli_support_payload_bytes(payload: bytes) -> int:
    magic, action_count = struct.unpack(_HEADER_FMT, payload[:_HEADER_SIZE])
    if magic != TARGET_REGION_ACTION_SPLIT_BROTLI_MAGIC:
        raise ValueError(f"bad split-brotli target-region action magic: {magic!r}")
    offset = _HEADER_SIZE
    total = 0
    for _ in range(int(action_count)):
        if offset + _SPLIT_ACTION_HEADER_SIZE > len(payload):
            raise ValueError("truncated split-brotli target-region action header")
        (
            _pair_index,
            _frame_index,
            _reserved,
            _height,
            _width,
            _pixel_count,
            coord_compressed_bytes,
            rgb_compressed_bytes,
        ) = struct.unpack(
            _SPLIT_ACTION_HEADER_FMT,
            payload[offset : offset + _SPLIT_ACTION_HEADER_SIZE],
        )
        offset += _SPLIT_ACTION_HEADER_SIZE
        total += int(coord_compressed_bytes)
        offset += int(coord_compressed_bytes) + int(rgb_compressed_bytes)
        if offset > len(payload):
            raise ValueError("truncated split-brotli target-region action payload")
    if offset != len(payload):
        raise ValueError("split-brotli target-region action payload has trailing bytes")
    return total


def _tile_brotli_support_payload_bytes(payload: bytes) -> int:
    magic, action_count = struct.unpack(_HEADER_FMT, payload[:_HEADER_SIZE])
    if magic != TARGET_REGION_ACTION_TILE_BROTLI_MAGIC:
        raise ValueError(f"bad tile-brotli target-region action magic: {magic!r}")
    offset = _HEADER_SIZE
    total = 0
    for _ in range(int(action_count)):
        if offset + _TILE_ACTION_HEADER_SIZE > len(payload):
            raise ValueError("truncated tile-brotli target-region action header")
        (
            _pair_index,
            _frame_index,
            _tile_size,
            _height,
            _width,
            _pixel_count,
            tile_compressed_bytes,
            rgb_compressed_bytes,
        ) = struct.unpack(
            _TILE_ACTION_HEADER_FMT,
            payload[offset : offset + _TILE_ACTION_HEADER_SIZE],
        )
        offset += _TILE_ACTION_HEADER_SIZE
        total += int(tile_compressed_bytes)
        offset += int(tile_compressed_bytes) + int(rgb_compressed_bytes)
        if offset > len(payload):
            raise ValueError("truncated tile-brotli target-region action payload")
    if offset != len(payload):
        raise ValueError("tile-brotli target-region action payload has trailing bytes")
    return total


def _target_region_action_support_telemetry(
    payload: bytes,
    actions: list[TargetRegionPixelAction],
) -> dict[str, Any]:
    codec = target_region_action_payload_codec(payload)
    logical_yx_bytes = int(sum(action.yx.nbytes for action in actions))
    if codec == "tile_brotli_v1":
        return {
            "support_source": "tile_bitmap_payload_coordinates",
            "support_encoding": "brotli_tile_bitmap_little_endian",
            "support_encoded_bytes": _tile_brotli_support_payload_bytes(payload),
            "support_logical_yx_bytes": logical_yx_bytes,
        }
    if codec == "split_brotli_v1":
        return {
            "support_source": "split_delta_payload_coordinates",
            "support_encoding": "brotli_split_yx_delta_streams",
            "support_encoded_bytes": _split_brotli_support_payload_bytes(payload),
            "support_logical_yx_bytes": logical_yx_bytes,
        }
    if codec == "brotli_wrapped_v1":
        return {
            "support_source": "wrapped_raw_payload_coordinates",
            "support_encoding": "brotli_wrapped_raw_yx_u16_coordinates",
            "support_encoded_bytes": len(payload) - _COMPRESSED_HEADER_SIZE,
            "support_logical_yx_bytes": logical_yx_bytes,
        }
    if codec == "zlib_wrapped_v1":
        return {
            "support_source": "wrapped_raw_payload_coordinates",
            "support_encoding": "zlib_wrapped_raw_yx_u16_coordinates",
            "support_encoded_bytes": len(payload) - _COMPRESSED_HEADER_SIZE,
            "support_logical_yx_bytes": logical_yx_bytes,
        }
    return {
        "support_source": "explicit_payload_coordinates",
        "support_encoding": "explicit_yx_u16_coordinates",
        "support_encoded_bytes": logical_yx_bytes,
        "support_logical_yx_bytes": logical_yx_bytes,
    }


def encode_target_region_actions_meta(actions: list[TargetRegionPixelAction]) -> str:
    return base64.b64encode(encode_target_region_actions_payload(actions)).decode("ascii")


def decode_target_region_actions_from_meta(meta: dict[str, Any]) -> list[TargetRegionPixelAction]:
    raw = meta.get(TARGET_REGION_ACTION_META_KEY)
    if raw in (None, ""):
        return []
    if not isinstance(raw, str):
        raise ValueError("target-region action meta field must be base64 text")
    return decode_target_region_actions(base64.b64decode(raw.encode("ascii"), validate=True))


def target_region_action_section_telemetry(actions: list[TargetRegionPixelAction]) -> dict[str, Any]:
    return target_region_action_section_telemetry_for_payload(
        actions,
        encode_target_region_actions_payload(actions),
    )


def target_region_action_section_telemetry_for_payload(
    actions: list[TargetRegionPixelAction],
    payload: bytes,
) -> dict[str, Any]:
    raw_payload = encode_target_region_actions(actions)
    stored_payload = bytes(payload)
    program_base64 = base64.b64encode(stored_payload).decode("ascii")
    encoded_program_sha256 = hashlib.sha256(stored_payload).hexdigest()
    support_telemetry = _target_region_action_support_telemetry(stored_payload, actions)
    return {
        "schema": TARGET_REGION_ACTION_SCHEMA,
        "meta_key": TARGET_REGION_ACTION_META_KEY,
        "action_count": len(actions),
        "pixel_count": int(sum(action.pixel_count for action in actions)),
        "payload_bytes": len(stored_payload),
        "payload_sha256": encoded_program_sha256,
        "encoded_program_sha256": encoded_program_sha256,
        "program_base64_sha256": hashlib.sha256(program_base64.encode("ascii")).hexdigest(),
        "raw_payload_bytes": len(raw_payload),
        "payload_codec": target_region_action_payload_codec(stored_payload),
        **support_telemetry,
        "support_cardinality": int(sum(action.pixel_count for action in actions)),
        "support_sha256": target_region_action_support_sha256(actions),
        "decoded_support_sha256": target_region_action_decoded_support_sha256(actions),
        "decoded_action_sha256": target_region_action_decoded_action_sha256(actions),
        "archive_executable_support": True,
        "charged_as_hiv1_meta_blob": True,
        "receiver_consumed": True,
    }


class TargetRegionActionReceiver(torch.nn.Module):
    """Wrap a parsed HiNeRV model and apply charged target-region actions."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        actions: list[TargetRegionPixelAction],
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.cfg = getattr(base_model, "cfg", None)
        self.actions = list(actions)
        self.actions_by_pair: dict[int, list[TargetRegionPixelAction]] = {}
        for action in self.actions:
            self.actions_by_pair.setdefault(int(action.pair_index), []).append(action)

    def forward(self, pair_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb0, rgb1 = self.base_model(pair_indices)
        if not self.actions_by_pair:
            return rgb0, rgb1
        out0 = rgb0.clone()
        out1 = rgb1.clone()
        indices = pair_indices.detach().to(device="cpu", dtype=torch.long).tolist()
        for batch_index, pair_index in enumerate(indices):
            for action in self.actions_by_pair.get(int(pair_index), ()):
                target = out0 if int(action.frame_index) == 0 else out1
                if tuple(target.shape[-2:]) != (int(action.height), int(action.width)):
                    raise ValueError(
                        "target-region action geometry mismatch: "
                        f"action={action.height}x{action.width} rendered={tuple(target.shape[-2:])}"
                    )
                y = torch.as_tensor(action.yx[:, 0].astype(np.int64), device=target.device)
                x = torch.as_tensor(action.yx[:, 1].astype(np.int64), device=target.device)
                values = torch.as_tensor(
                    action.rgb_u8.astype(np.float32) / 255.0,
                    device=target.device,
                    dtype=target.dtype,
                )
                target[batch_index, :, y, x] = values.transpose(0, 1)
        return out0, out1


def wrap_model_with_target_region_actions(
    model: torch.nn.Module,
    meta: dict[str, Any],
) -> torch.nn.Module:
    actions = decode_target_region_actions_from_meta(meta)
    if not actions:
        return model
    return TargetRegionActionReceiver(model, actions).eval()
