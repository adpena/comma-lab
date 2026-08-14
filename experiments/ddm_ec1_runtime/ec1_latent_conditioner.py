"""Counted latent edge conditioner for the CP135 semantic renderer.

The receiver derives every conditioning feature from the decoded semantic
tokens.  The only video-derived values consumed here are the quantized model
parameters carried in ``ec1_latent.br`` inside the counted archive.
"""

from __future__ import annotations

import json
from functools import lru_cache

import brotli
import numpy as np
import torch
from torch import nn
from torch.nn import functional

MAGIC = b"EC1L\x01"
SCHEMA = "ddm_ec1_latent_conditioner.v1"
NUM_CLASSES = 5
CONTEXT_CHANNELS = 25
SEMANTIC_WIDTH = 96
FAMILIES = ("class_only", "undirected", "oriented")


class EC1RuntimeError(RuntimeError):
    """The counted adapter or receiver-computable context is invalid."""


def parse_module(coded: bytes) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    raw = brotli.decompress(coded)
    if not raw.startswith(MAGIC) or len(raw) < len(MAGIC) + 4:
        raise EC1RuntimeError("invalid EC1 latent adapter payload")
    offset = len(MAGIC)
    header_bytes = int.from_bytes(raw[offset : offset + 4], "little")
    offset += 4
    header = json.loads(raw[offset : offset + header_bytes])
    offset += header_bytes
    if header.get("schema") != SCHEMA or header.get("family") not in FAMILIES:
        raise EC1RuntimeError("unsupported EC1 latent adapter schema or family")
    output: dict[str, np.ndarray] = {}
    for row in header["tensors"]:
        end = offset + int(row["bytes"])
        if end > len(raw):
            raise EC1RuntimeError("truncated EC1 tensor payload")
        if row["dtype"] == "float16":
            value = np.frombuffer(raw[offset:end], dtype="<f2").copy().reshape(row["shape"])
            value = value.astype(np.float32)
        elif row["dtype"] == "int8":
            value = np.frombuffer(raw[offset:end], dtype=np.int8).copy().reshape(row["shape"])
            value = value.astype(np.float32) * float(row["scale"])
        else:
            raise EC1RuntimeError(f"unsupported tensor dtype: {row['dtype']!r}")
        output[str(row["name"])] = value
        offset = end
    if offset != len(raw):
        raise EC1RuntimeError("EC1 latent adapter payload has trailing bytes")
    return header, output


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


def edge_context(tokens: torch.Tensor, family: str) -> torch.Tensor:
    """Return a fixed-width context using decoded tokens and nothing else."""
    if family not in FAMILIES:
        raise EC1RuntimeError(f"unknown EC1 context family: {family}")
    if tokens.ndim != 3:
        raise EC1RuntimeError(f"token geometry differs: {tuple(tokens.shape)}")
    center = functional.one_hot(tokens.long(), num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
    zeros = torch.zeros_like(center)
    if family == "class_only":
        value = torch.cat((center, zeros, zeros, zeros, zeros), dim=1)
    else:
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
        if family == "undirected":
            pooled = torch.stack(neighbors).sum(dim=0) / 4.0
            value = torch.cat((center, pooled, zeros, zeros, zeros), dim=1)
        else:
            value = torch.cat((center, *neighbors), dim=1)
    if value.shape[1] != CONTEXT_CHANNELS:
        raise EC1RuntimeError(f"EC1 context width differs: {tuple(value.shape)}")
    return value


class LatentEdgeConditioner(nn.Module):
    """Small counted adapter injected before CP135's four nonlinear blocks."""

    def __init__(self, hidden: int, max_delta: float, family: str) -> None:
        super().__init__()
        if hidden <= 0 or family not in FAMILIES or max_delta <= 0.0:
            raise ValueError("invalid EC1 latent adapter configuration")
        self.hidden = hidden
        self.max_delta = max_delta
        self.family = family
        self.context = nn.Conv2d(CONTEXT_CHANNELS, hidden, 3, padding=1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.head = nn.Conv2d(hidden, SEMANTIC_WIDTH, 1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        value = functional.gelu(self.context(context))
        value = functional.gelu(self.depthwise(value))
        return torch.tanh(self.head(value)) * self.max_delta


@lru_cache(maxsize=4)
def _cpu_payload(coded: bytes) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    return parse_module(coded)


def load_conditioner(coded: bytes, device: torch.device) -> LatentEdgeConditioner:
    header, arrays = _cpu_payload(coded)
    model = LatentEdgeConditioner(
        int(header["hidden"]),
        float(header["max_delta"]),
        str(header["family"]),
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
    """Run the real CP135 semantic path with latent, not post-render, injection."""
    conditioner = load_conditioner(coded, tokens.device)
    value = semantic.token_embed(tokens).permute(0, 3, 1, 2)
    value = semantic.coord_mix(
        torch.cat(
            [
                value,
                semantic.coordinates(value.shape[0], value.device, value.dtype),
            ],
            dim=1,
        )
    )
    value = value + conditioner(edge_context(tokens, conditioner.family))
    frame = semantic.frame_embed(pair_indices)
    for block in semantic.blocks:
        value = block(value, frame)
    return torch.sigmoid(semantic.head(functional.gelu(value))) * 255.0
