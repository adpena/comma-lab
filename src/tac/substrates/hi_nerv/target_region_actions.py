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
TARGET_REGION_ACTION_SCHEMA = "hi_nerv_target_region_archive_actions.v1"
_HEADER_FMT = "<5sH"
_COMPRESSED_HEADER_FMT = "<5sI"
_ACTION_HEADER_FMT = "<HBBHHI"
_SPLIT_ACTION_HEADER_FMT = "<HBBHHIII"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_COMPRESSED_HEADER_SIZE = struct.calcsize(_COMPRESSED_HEADER_FMT)
_ACTION_HEADER_SIZE = struct.calcsize(_ACTION_HEADER_FMT)
_SPLIT_ACTION_HEADER_SIZE = struct.calcsize(_SPLIT_ACTION_HEADER_FMT)


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
    _size, payload = min(candidates, key=lambda item: item[0])
    if payload is raw:
        return raw
    return payload


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


def decode_target_region_actions(blob: bytes) -> list[TargetRegionPixelAction]:
    """Decode the charged receiver binary grammar."""

    if len(blob) >= _HEADER_SIZE:
        magic, _action_count = struct.unpack(_HEADER_FMT, blob[:_HEADER_SIZE])
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
    raw_payload = encode_target_region_actions(actions)
    payload = encode_target_region_actions_payload(actions)
    return {
        "schema": TARGET_REGION_ACTION_SCHEMA,
        "meta_key": TARGET_REGION_ACTION_META_KEY,
        "action_count": len(actions),
        "pixel_count": int(sum(action.pixel_count for action in actions)),
        "payload_bytes": len(payload),
        "raw_payload_bytes": len(raw_payload),
        "payload_codec": target_region_action_payload_codec(payload),
        "support_source": "explicit_payload_coordinates",
        "support_encoding": "explicit_yx_u16_coordinates",
        "support_cardinality": int(sum(action.pixel_count for action in actions)),
        "support_encoded_bytes": int(sum(action.yx.nbytes for action in actions)),
        "support_sha256": target_region_action_support_sha256(actions),
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
