# SPDX-License-Identifier: MIT
"""Receiver-consumed HiNeRV target-region action sidecar grammar.

The payload is charged because it lives inside the HIV1 meta blob.  It is not a
training proxy: parsed receivers apply these uint8 pixel actions after model
render and before scorer/raw-output surfaces.
"""

from __future__ import annotations

import base64
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

TARGET_REGION_ACTION_META_KEY = "_target_region_actions_v1_b64"
TARGET_REGION_ACTION_MAGIC = b"HTRA1"
TARGET_REGION_ACTION_SCHEMA = "hi_nerv_target_region_archive_actions.v1"
_HEADER_FMT = "<5sH"
_ACTION_HEADER_FMT = "<HBBHHI"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_ACTION_HEADER_SIZE = struct.calcsize(_ACTION_HEADER_FMT)


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


def decode_target_region_actions(blob: bytes) -> list[TargetRegionPixelAction]:
    """Decode the charged receiver binary grammar."""

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


def encode_target_region_actions_meta(actions: list[TargetRegionPixelAction]) -> str:
    return base64.b64encode(encode_target_region_actions(actions)).decode("ascii")


def decode_target_region_actions_from_meta(meta: dict[str, Any]) -> list[TargetRegionPixelAction]:
    raw = meta.get(TARGET_REGION_ACTION_META_KEY)
    if raw in (None, ""):
        return []
    if not isinstance(raw, str):
        raise ValueError("target-region action meta field must be base64 text")
    return decode_target_region_actions(base64.b64decode(raw.encode("ascii"), validate=True))


def target_region_action_section_telemetry(actions: list[TargetRegionPixelAction]) -> dict[str, Any]:
    payload = encode_target_region_actions(actions)
    return {
        "schema": TARGET_REGION_ACTION_SCHEMA,
        "meta_key": TARGET_REGION_ACTION_META_KEY,
        "action_count": len(actions),
        "pixel_count": int(sum(action.pixel_count for action in actions)),
        "payload_bytes": len(payload),
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
