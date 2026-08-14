"""Counted receiver-native multi-token representative for the CP135 renderer.

The receiver derives oriented semantic context from the decoded CP135 token
field.  A counted quantized module turns that context into a simplex-valued
probability state over CP135's five existing semantic embeddings.  It does not
receive or reconstruct an explicit changed-site list.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import brotli
import numpy as np
import torch
from torch import nn
from torch.nn import functional

MAGIC = b"MT1M\x01"
SCHEMA = "ddm_mt1_multitoken_representative.v1"
NUM_CLASSES = 5
CONTEXT_CHANNELS = 25
SEMANTIC_WIDTH = 96


class MT1RuntimeError(RuntimeError):
    """The counted module or receiver-computable context is invalid."""


def _shift(tokens: torch.Tensor, direction: str) -> torch.Tensor:
    if direction == "left":
        return functional.pad(tokens[:, :, :-1], (1, 0, 0, 0), mode="replicate")
    if direction == "right":
        return functional.pad(tokens[:, :, 1:], (0, 1, 0, 0), mode="replicate")
    if direction == "up":
        return functional.pad(tokens[:, :-1, :], (0, 0, 1, 0), mode="replicate")
    if direction == "down":
        return functional.pad(tokens[:, 1:, :], (0, 0, 0, 1), mode="replicate")
    raise ValueError(f"unknown direction: {direction}")


def oriented_context(tokens: torch.Tensor) -> torch.Tensor:
    """Return center plus four directed neighbor one-hots from decoded tokens."""
    if tokens.ndim != 3:
        raise MT1RuntimeError(f"token geometry differs: {tuple(tokens.shape)}")
    if torch.any(tokens < 0) or torch.any(tokens >= NUM_CLASSES):
        raise MT1RuntimeError("token class is outside the CP135 alphabet")
    center = functional.one_hot(tokens.long(), num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
    neighbors = []
    for direction in ("left", "right", "up", "down"):
        neighbor = _shift(tokens, direction)
        edge = (neighbor != tokens)[:, None]
        neighbors.append(
            functional.one_hot(neighbor.long(), num_classes=NUM_CLASSES)
            .permute(0, 3, 1, 2)
            .float()
            * edge
        )
    value = torch.cat((center, *neighbors), dim=1)
    if value.shape[1] != CONTEXT_CHANNELS:
        raise MT1RuntimeError(f"multi-token context width differs: {tuple(value.shape)}")
    return value


class MultiTokenRepresentative(nn.Module):
    """Learn a local simplex state and mix CP135's own semantic embeddings."""

    def __init__(self, hidden: int, max_support_mass: float) -> None:
        super().__init__()
        if hidden <= 0 or not 0.0 < max_support_mass <= 1.0:
            raise ValueError("invalid multi-token representative configuration")
        self.hidden = hidden
        self.max_support_mass = max_support_mass
        self.context = nn.Conv2d(CONTEXT_CHANNELS, hidden, 3, padding=1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.mass_head = nn.Conv2d(hidden, 1, 1)
        self.support_head = nn.Conv2d(hidden, NUM_CLASSES, 1)

    def probability_state(self, tokens: torch.Tensor) -> torch.Tensor:
        context = oriented_context(tokens)
        value = functional.gelu(_quantized_conv(self.context, context, padding=1))
        value = functional.gelu(
            _quantized_conv(
                self.depthwise,
                value,
                padding=1,
                groups=self.hidden,
            )
        )
        support_mass = (
            torch.sigmoid(_quantized_conv(self.mass_head, value))
            * self.max_support_mass
        )
        center = (
            functional.one_hot(tokens.long(), num_classes=NUM_CLASSES)
            .permute(0, 3, 1, 2)
            .float()
        )
        support_logits = _quantized_conv(self.support_head, value).masked_fill(
            center.bool(), -1.0e4
        )
        support = torch.softmax(support_logits, dim=1)
        probability = (1.0 - support_mass) * center + support_mass * support
        return probability

    def representative(self, tokens: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.shape != (NUM_CLASSES, SEMANTIC_WIDTH):
            raise MT1RuntimeError(f"semantic embedding geometry differs: {tuple(embeddings.shape)}")
        probability = self.probability_state(tokens)
        return torch.einsum("bkhw,kd->bdhw", probability, embeddings)


def _int8_weight_ste(value: torch.Tensor) -> torch.Tensor:
    maximum = float(value.detach().abs().amax().cpu())
    scale = torch.tensor(
        max(maximum / 127.0, 1.0e-8),
        dtype=value.dtype,
        device=value.device,
    )
    decoded = torch.clamp(torch.round(value / scale), -127, 127) * scale
    return value + (decoded - value).detach()


def _float16_ste(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    decoded = value.to(torch.float16).to(value.dtype)
    return value + (decoded - value).detach()


def _quantized_conv(
    layer: nn.Conv2d,
    value: torch.Tensor,
    *,
    padding: int = 0,
    groups: int = 1,
) -> torch.Tensor:
    """Use exactly the int8-weight/fp16-bias storage quantizers in-loop."""
    return functional.conv2d(
        value,
        _int8_weight_ste(layer.weight),
        _float16_ste(layer.bias),
        padding=padding,
        groups=groups,
    )


def serialize_model(model: MultiTokenRepresentative) -> tuple[bytes, dict[str, Any]]:
    """Serialize the counted module and return its explicit quantization receipt."""
    tensors: list[dict[str, Any]] = []
    payloads: list[bytes] = []
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if name.endswith("weight"):
            scale = max(float(np.max(np.abs(value))) / 127.0, 1.0e-8)
            quantized = np.clip(np.rint(value / scale), -127, 127).astype(np.int8)
            payload = quantized.tobytes(order="C")
            row = {
                "name": name,
                "dtype": "int8",
                "shape": list(value.shape),
                "bytes": quantized.nbytes,
                "scale": scale,
            }
        else:
            quantized = value.astype("<f2")
            payload = quantized.tobytes(order="C")
            row = {
                "name": name,
                "dtype": "float16",
                "shape": list(value.shape),
                "bytes": quantized.nbytes,
            }
        tensors.append(row)
        payloads.append(payload)
    header: dict[str, Any] = {
        "schema": SCHEMA,
        "hidden": model.hidden,
        "max_support_mass": model.max_support_mass,
        "tensors": tensors,
    }
    header_payload = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    raw = MAGIC + len(header_payload).to_bytes(4, "little") + header_payload
    raw += b"".join(payloads)
    coded = brotli.compress(raw, quality=11)
    return coded, {
        "schema": SCHEMA,
        "raw_bytes": len(raw),
        "tensors": tensors,
    }


def parse_module(coded: bytes) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    raw = brotli.decompress(coded)
    if not raw.startswith(MAGIC) or len(raw) < len(MAGIC) + 4:
        raise MT1RuntimeError("invalid MT1 module payload")
    offset = len(MAGIC)
    header_bytes = int.from_bytes(raw[offset : offset + 4], "little")
    offset += 4
    header = json.loads(raw[offset : offset + header_bytes])
    offset += header_bytes
    if header.get("schema") != SCHEMA:
        raise MT1RuntimeError("unsupported MT1 module schema")
    output: dict[str, np.ndarray] = {}
    for row in header["tensors"]:
        end = offset + int(row["bytes"])
        if end > len(raw):
            raise MT1RuntimeError("truncated MT1 tensor payload")
        if row["dtype"] == "float16":
            value = np.frombuffer(raw[offset:end], dtype="<f2").copy().reshape(row["shape"])
            value = value.astype(np.float32)
        elif row["dtype"] == "int8":
            value = np.frombuffer(raw[offset:end], dtype=np.int8).copy().reshape(row["shape"])
            value = value.astype(np.float32) * float(row["scale"])
        else:
            raise MT1RuntimeError(f"unsupported MT1 tensor dtype: {row['dtype']!r}")
        output[str(row["name"])] = value
        offset = end
    if offset != len(raw):
        raise MT1RuntimeError("MT1 module payload has trailing bytes")
    return header, output


@lru_cache(maxsize=4)
def _cpu_payload(coded: bytes) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    return parse_module(coded)


def load_module(coded: bytes, device: torch.device) -> MultiTokenRepresentative:
    header, arrays = _cpu_payload(coded)
    model = MultiTokenRepresentative(
        hidden=int(header["hidden"]),
        max_support_mass=float(header["max_support_mass"]),
    )
    model.load_state_dict(
        {name: torch.from_numpy(value.copy()) for name, value in arrays.items()},
        strict=True,
    )
    return model.eval().to(device)


def conditioned_semantic_forward(
    semantic: nn.Module,
    tokens: torch.Tensor,
    pair_indices: torch.Tensor,
    coded: bytes,
) -> torch.Tensor:
    """Run the CP135 semantic receiver with the parsed simplex representative."""
    model = load_module(coded, tokens.device)
    value = model.representative(tokens, semantic.token_embed.weight)
    value = semantic.coord_mix(
        torch.cat(
            [
                value,
                semantic.coordinates(value.shape[0], value.device, value.dtype),
            ],
            dim=1,
        )
    )
    frame = semantic.frame_embed(pair_indices)
    for block in semantic.blocks:
        value = block(value, frame)
    return torch.sigmoid(semantic.head(functional.gelu(value))) * 255.0
