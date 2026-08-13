"""Decode and apply the counted SA1 implicit-conditioning module.

This file is copied into the candidate's adapted receiver.  The algorithm is
generic receiver code; all learned values arrive in the counted
``sa1_conditioner.br`` archive member.
"""

from __future__ import annotations

import json
from functools import lru_cache

import brotli
import numpy as np
import torch
from torch import nn
from torch.nn import functional

MODULE_MAGIC = b"JS3C\x01"
NUM_CLASSES = 5
EVAL_H = 384
EVAL_W = 512
CHANNELS = 14


class ConditionerError(RuntimeError):
    """The counted conditioner payload or receiver context is invalid."""


def parse_module(coded: bytes) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    raw = brotli.decompress(coded)
    if not raw.startswith(MODULE_MAGIC) or len(raw) < len(MODULE_MAGIC) + 4:
        raise ConditionerError("invalid SA1 conditioner payload")
    offset = len(MODULE_MAGIC)
    header_bytes = int.from_bytes(raw[offset : offset + 4], "little")
    offset += 4
    header = json.loads(raw[offset : offset + header_bytes])
    offset += header_bytes
    if header.get("schema") != "ddm_js3_conditioner_payload.v1":
        raise ConditionerError("unsupported SA1 conditioner schema")
    output: dict[str, np.ndarray] = {}
    for row in header["tensors"]:
        end = offset + int(row["bytes"])
        dtype = np.dtype("<f2") if row["dtype"] == "float16" else np.dtype(np.int8)
        value = np.frombuffer(raw[offset:end], dtype=dtype).copy().reshape(row["shape"])
        if row["dtype"] == "float16":
            value = value.astype(np.float32)
        elif row["dtype"] == "int8":
            value = value.astype(np.float32) * float(row["scale"])
        else:
            raise ConditionerError(f"unsupported tensor dtype: {row['dtype']!r}")
        output[str(row["name"])] = value
        offset = end
    if offset != len(raw):
        raise ConditionerError("SA1 conditioner payload has trailing bytes")
    return header, output


class EdgeConditioner(nn.Module):
    """The exact tiny context-convolution family trained by JS3/SA1."""

    def __init__(self, hidden: int, max_delta: float) -> None:
        super().__init__()
        self.hidden = hidden
        self.max_delta = max_delta
        self.context = nn.Conv2d(CHANNELS, hidden, 3, padding=1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.head = nn.Conv2d(hidden, 3, 1)

    @staticmethod
    def _fake_quant(weight: torch.Tensor) -> torch.Tensor:
        scale = weight.detach().abs().amax().clamp_min(1e-8) / 127.0
        return torch.fake_quantize_per_tensor_affine(
            weight, float(scale), 0, -127, 127
        )

    def _conv(self, value: torch.Tensor, layer: nn.Conv2d, *, groups: int = 1) -> torch.Tensor:
        return functional.conv2d(
            value,
            self._fake_quant(layer.weight),
            layer.bias,
            padding=layer.padding,
            groups=groups,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = functional.gelu(self._conv(value, self.context))
        value = functional.gelu(
            self._conv(value, self.depthwise, groups=self.hidden)
        )
        return torch.tanh(self._conv(value, self.head)) * self.max_delta


def fixed_context(tokens: torch.Tensor, pre_r: torch.Tensor) -> torch.Tensor:
    one_hot = functional.one_hot(tokens.long(), num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
    left = functional.pad(tokens[:, :, 1:] != tokens[:, :, :-1], (1, 0, 0, 0)).float()
    right = functional.pad(tokens[:, :, 1:] != tokens[:, :, :-1], (0, 1, 0, 0)).float()
    up = functional.pad(tokens[:, 1:, :] != tokens[:, :-1, :], (0, 0, 1, 0)).float()
    down = functional.pad(tokens[:, 1:, :] != tokens[:, :-1, :], (0, 0, 0, 1)).float()
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, EVAL_H, dtype=pre_r.dtype, device=pre_r.device),
        torch.linspace(-1.0, 1.0, EVAL_W, dtype=pre_r.dtype, device=pre_r.device),
        indexing="ij",
    )
    coordinates = torch.stack((xx, yy))[None].expand(tokens.shape[0], -1, -1, -1)
    value = torch.cat(
        (
            one_hot,
            left[:, None],
            right[:, None],
            up[:, None],
            down[:, None],
            coordinates,
            pre_r / 127.5 - 1.0,
        ),
        dim=1,
    )
    if value.shape[1:] != (CHANNELS, EVAL_H, EVAL_W):
        raise ConditionerError(f"SA1 context geometry differs: {tuple(value.shape)}")
    return value


@lru_cache(maxsize=2)
def _cpu_payload(coded: bytes) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    return parse_module(coded)


def load_conditioner(coded: bytes, device: torch.device) -> EdgeConditioner:
    header, arrays = _cpu_payload(coded)
    model = EdgeConditioner(int(header["hidden"]), float(header["max_delta"]))
    model.load_state_dict(
        {name: torch.from_numpy(value.copy()) for name, value in arrays.items()},
        strict=True,
    )
    return model.eval().to(device)


def apply_conditioner(
    coded: bytes,
    tokens: torch.Tensor,
    pre_r: torch.Tensor,
) -> torch.Tensor:
    model = load_conditioner(coded, pre_r.device)
    correction = model(fixed_context(tokens, pre_r))
    return pre_r + correction

