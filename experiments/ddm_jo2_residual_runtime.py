#!/usr/bin/env python3
"""Counted JO2 residual payload and its generic receiver implementation.

The archive stores a quantized ``HybridOutputResidual`` state after the exact
base semantic body.  The runtime wrapper consumes that state immediately after
the final semantic TokenBlock and before the renderer's resize/roundtrip.  No
video-derived tensor is embedded in this source file.
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from collections.abc import Mapping
from typing import Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional

SEMANTIC_MAGIC: Final = b"J2R1"
SEMANTIC_VERSION: Final = 1
SEMANTIC_HEADER: Final = struct.Struct("<4sBII32s")
STATE_MAGIC: Final = b"J2S1"
STATE_VERSION: Final = 1
STATE_HEADER: Final = struct.Struct("<4sBBf")
NUM_CLASSES: Final = 5


class JO2ResidualError(ValueError):
    """The counted JO2 residual or its receiver geometry is invalid."""


def _state_shapes(hidden_channels: int) -> OrderedDict[str, tuple[int, ...]]:
    if not 1 <= hidden_channels <= 64:
        raise JO2ResidualError("hidden channel count is outside [1,64]")
    return OrderedDict(
        (
            ("context.weight", (hidden_channels, 15, 1, 1)),
            ("context.bias", (hidden_channels,)),
            ("oriented.weight", (hidden_channels, 1, 3, 3)),
            ("oriented.bias", (hidden_channels,)),
            ("head.weight", (3, hidden_channels, 1, 1)),
            ("head.bias", (3,)),
        )
    )


def encode_residual_state(
    state: Mapping[str, torch.Tensor],
    *,
    hidden_channels: int,
    max_rgb_delta: float,
) -> bytes:
    """Quantize one candidate state to the exact float16 receiver object."""
    if not np.isfinite(max_rgb_delta) or max_rgb_delta <= 0.0:
        raise JO2ResidualError("max_rgb_delta must be positive and finite")
    shapes = _state_shapes(hidden_channels)
    if tuple(state) != tuple(shapes):
        raise JO2ResidualError("residual state names or order differ")
    body = bytearray()
    for name, shape in shapes.items():
        value = state[name].detach().cpu().float().numpy()
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise JO2ResidualError(f"residual state tensor differs: {name}")
        body.extend(np.asarray(value, dtype="<f2").tobytes())
    return STATE_HEADER.pack(
        STATE_MAGIC,
        STATE_VERSION,
        hidden_channels,
        float(max_rgb_delta),
    ) + bytes(body)


def decode_residual_state(
    payload: bytes,
) -> tuple[OrderedDict[str, torch.Tensor], int, float]:
    """Parse the counted state with no trailing-byte ambiguity."""
    if len(payload) < STATE_HEADER.size:
        raise JO2ResidualError("residual state is truncated")
    magic, version, hidden_channels, max_rgb_delta = STATE_HEADER.unpack_from(payload)
    if magic != STATE_MAGIC or version != STATE_VERSION:
        raise JO2ResidualError("residual state magic or version differs")
    if not np.isfinite(max_rgb_delta) or max_rgb_delta <= 0.0:
        raise JO2ResidualError("residual state max_rgb_delta differs")
    shapes = _state_shapes(hidden_channels)
    cursor = STATE_HEADER.size
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, shape in shapes.items():
        count = int(np.prod(shape))
        byte_count = count * np.dtype("<f2").itemsize
        end = cursor + byte_count
        if end > len(payload):
            raise JO2ResidualError(f"residual state tensor is truncated: {name}")
        values = np.frombuffer(payload[cursor:end], dtype="<f2").copy()
        result[name] = torch.from_numpy(values.reshape(shape)).float()
        cursor = end
    if cursor != len(payload):
        raise JO2ResidualError("residual state has trailing bytes")
    return result, int(hidden_channels), float(max_rgb_delta)


def pack_semantic_blob(base_semantic: bytes, residual_payload: bytes) -> bytes:
    """Bind a base semantic body and one exact residual state by SHA-256."""
    if not base_semantic or base_semantic.startswith(SEMANTIC_MAGIC):
        raise JO2ResidualError("base semantic body is empty or already JO2-tagged")
    decode_residual_state(residual_payload)
    return (
        SEMANTIC_HEADER.pack(
            SEMANTIC_MAGIC,
            SEMANTIC_VERSION,
            len(base_semantic),
            len(residual_payload),
            hashlib.sha256(residual_payload).digest(),
        )
        + base_semantic
        + residual_payload
    )


def split_semantic_blob(value: bytes) -> tuple[bytes, bytes | None]:
    """Return the base body and optional JO2 state, rejecting malformed tags."""
    if not value.startswith(SEMANTIC_MAGIC):
        return value, None
    if len(value) < SEMANTIC_HEADER.size:
        raise JO2ResidualError("JO2 semantic body is truncated")
    magic, version, base_bytes, residual_bytes, expected_digest = (
        SEMANTIC_HEADER.unpack_from(value)
    )
    if magic != SEMANTIC_MAGIC or version != SEMANTIC_VERSION:
        raise JO2ResidualError("JO2 semantic magic or version differs")
    expected_bytes = SEMANTIC_HEADER.size + base_bytes + residual_bytes
    if min(base_bytes, residual_bytes) <= 0 or len(value) != expected_bytes:
        raise JO2ResidualError("JO2 semantic section lengths differ")
    base = value[SEMANTIC_HEADER.size : SEMANTIC_HEADER.size + base_bytes]
    residual = value[SEMANTIC_HEADER.size + base_bytes :]
    if base.startswith(SEMANTIC_MAGIC):
        raise JO2ResidualError("nested JO2 semantic bodies are forbidden")
    if hashlib.sha256(residual).digest() != expected_digest:
        raise JO2ResidualError("JO2 residual state digest differs")
    decode_residual_state(residual)
    return base, residual


class OutputResidual(nn.Module):
    """The shipped post-TokenBlock RGB residual actuator."""

    def __init__(self, hidden_channels: int, max_rgb_delta: float) -> None:
        super().__init__()
        _state_shapes(hidden_channels)
        if not np.isfinite(max_rgb_delta) or max_rgb_delta <= 0.0:
            raise JO2ResidualError("max_rgb_delta must be positive and finite")
        self.hidden_channels = int(hidden_channels)
        self.max_rgb_delta = float(max_rgb_delta)
        self.context = nn.Conv2d(15, hidden_channels, 1)
        self.oriented = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            3,
            padding=1,
            groups=hidden_channels,
        )
        self.head = nn.Conv2d(hidden_channels, 3, 1)

    @staticmethod
    def oriented_context(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise JO2ResidualError("token batch must have shape (B,H,W)")
        if torch.any(tokens < 0) or torch.any(tokens >= NUM_CLASSES):
            raise JO2ResidualError("token value is outside the five-class domain")
        one_hot = functional.one_hot(
            tokens.long(), num_classes=NUM_CLASSES
        ).permute(0, 3, 1, 2).float()
        horizontal = functional.pad(one_hot, (1, 1, 0, 0), mode="replicate")
        horizontal = 0.5 * (horizontal[:, :, :, 2:] - horizontal[:, :, :, :-2])
        vertical = functional.pad(one_hot, (0, 0, 1, 1), mode="replicate")
        vertical = 0.5 * (vertical[:, :, 2:, :] - vertical[:, :, :-2, :])
        return torch.cat((one_hot, horizontal, vertical), dim=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        value = self.oriented_context(tokens)
        hidden = functional.gelu(self.context(value))
        hidden = hidden + functional.gelu(self.oriented(hidden))
        return torch.tanh(self.head(hidden)) * self.max_rgb_delta


def residual_from_payload(payload: bytes) -> OutputResidual:
    state, hidden_channels, max_rgb_delta = decode_residual_state(payload)
    model = OutputResidual(hidden_channels, max_rgb_delta)
    model.load_state_dict(state, strict=True)
    return model


class ResidualWrappedRenderer(nn.Module):
    """Apply the counted residual to a base semantic renderer's exact output."""

    def __init__(self, base: nn.Module, residual_payload: bytes) -> None:
        super().__init__()
        self.base = base
        self.residual = residual_from_payload(residual_payload)

    def forward(self, tokens: torch.Tensor, pair_indices: torch.Tensor) -> torch.Tensor:
        base_rgb = self.base(tokens, pair_indices)
        correction = self.residual(tokens)
        if base_rgb.shape != correction.shape:
            raise JO2ResidualError("base and residual renderer geometry differs")
        return (base_rgb + correction).clamp(0.0, 255.0)
