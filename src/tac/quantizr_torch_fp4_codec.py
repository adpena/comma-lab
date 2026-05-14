# SPDX-License-Identifier: MIT
"""PR #63-style Torch FP4 payload codec for JointFrameGenerator.

The public qpose14 submission stores a ``torch.save`` dictionary with
block-FP4 Conv/Embedding weights plus FP16 dense tensors. This module keeps
that format available as a conservative, current-floor-compatible alternative
to the smaller QZS3 packer. It is a build/runtime codec only; any archive
using it remains non-promotable until exact CUDA auth eval lands.

Example::

    >>> from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
    >>> payload = encode_torch_fp4_state_dict(build_quantizr_faithful_renderer())
    >>> model = load_torch_fp4_bytes(payload)

REHYDRATED 2026-05-05 from .recovery_spec.json (preserved at
.recovery_quarantine_20260505T004735Z/src/tac/quantizr_torch_fp4_codec.recovery_spec.json).
Spec source: bytecode disassembly of compiled .pyc; whitespace + inline comments lost.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import torch
from torch import nn

from tac.quantizr_faithful_renderer import (
    JointFrameGenerator,
    build_quantizr_faithful_renderer,
)
from tac.quantizr_qzs3_codec import (
    DEFAULT_BLOCK_SIZE,
    FP4_POS_LEVELS,
    _pack_nibbles,
    _unpack_nibbles,
)

TORCH_FP4_FORMAT = "fp4_standalone"
PROTECTED_MODULES = frozenset(
    {
        "frame1_head.head",
        "frame2_head.head",
        "shared_trunk.embedding",
    }
)


def is_torch_fp4_payload(payload: Any) -> bool:
    """Return true for the public qpose14 Torch-FP4 payload shape."""
    if not isinstance(payload, dict):
        return False
    fmt = payload.get("__format__")
    if not isinstance(fmt, str):
        return False
    if not fmt.startswith(TORCH_FP4_FORMAT):
        return False
    if not isinstance(payload.get("quantized"), dict):
        return False
    return isinstance(payload.get("dense_fp16"), dict)


def _module_type(module: nn.Module) -> str | None:
    if isinstance(module, nn.Conv2d):
        return "conv2d"
    if isinstance(module, nn.Embedding):
        return "embedding"
    return None


def _quantize_fp4_tensor(
    tensor: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = tensor.detach().cpu().float().numpy().reshape(-1)
    pad = (-flat.size) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    blocks = flat.reshape(-1, block_size)
    scales = np.maximum(
        np.max(np.abs(blocks), axis=1) / float(FP4_POS_LEVELS[-1]), 1e-08
    )
    scaled_abs = np.abs(blocks) / scales[:, None]
    idx = np.abs(
        scaled_abs[..., None] - FP4_POS_LEVELS[None, None, :]
    ).argmin(axis=2)
    signs = (blocks < 0).astype(np.uint8) << 3
    nibbles = (signs | idx.astype(np.uint8)).reshape(-1)
    packed = np.frombuffer(_pack_nibbles(nibbles), dtype=np.uint8).copy()
    return (
        torch.from_numpy(packed),
        torch.from_numpy(scales.astype(np.float16, copy=False)),
    )


def _dequantize_fp4_tensor(
    packed_weight: torch.Tensor,
    scales_fp16: torch.Tensor,
    weight_shape: tuple[int, ...],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    shape = tuple(int(s) for s in weight_shape)
    count = int(np.prod(shape))
    packed = (
        packed_weight.detach().cpu().numpy().astype(np.uint8, copy=False).tobytes()
    )
    scales = scales_fp16.detach().cpu().numpy().astype(np.float16, copy=False)
    if scales.size == 0:
        return torch.empty(shape, dtype=torch.float32, device=device)
    block_size = (len(packed) * 2) // int(scales.size)
    nibbles = _unpack_nibbles(packed, int(scales.size) * block_size).reshape(
        scales.size, block_size
    )
    signs = (nibbles >> 3).astype(bool)
    mag_idx = (nibbles & 7).astype(np.int64)
    values = FP4_POS_LEVELS[mag_idx] * scales.astype(np.float32)[:, None]
    values = np.where(signs, -values, values).reshape(-1)[:count]
    return torch.from_numpy(values.reshape(shape).astype(np.float32, copy=False)).to(
        device
    )


def encode_torch_fp4_payload(
    model_or_state: JointFrameGenerator | dict[str, Any],
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, Any]:
    """Encode a JointFrameGenerator state as a PR #63-style payload dict."""
    if block_size <= 0 or block_size > 4096:
        raise ValueError(f"invalid Torch-FP4 block size: {block_size}")
    if isinstance(model_or_state, JointFrameGenerator):
        model = model_or_state
        state = model.state_dict()
    else:
        model = build_quantizr_faithful_renderer()
        state = model_or_state
        model.load_state_dict(state, strict=True)
    template_state = build_quantizr_faithful_renderer().state_dict()
    quantized: dict[str, Any] = {}
    dense_fp16: dict[str, Any] = {}
    for name, module in model.named_modules():
        kind = _module_type(module)
        if kind is None:
            continue
        if name in PROTECTED_MODULES:
            continue
        weight_key = f"{name}.weight"
        if weight_key not in state:
            continue
        weight = state[weight_key]
        if not isinstance(weight, torch.Tensor):
            continue
        packed, scales = _quantize_fp4_tensor(weight, block_size)
        quantized[weight_key] = {
            "kind": kind,
            "packed": packed,
            "scales": scales,
            "shape": tuple(int(s) for s in weight.shape),
            "block_size": int(block_size),
        }
    quantized_keys = set(quantized.keys())
    for key, value in state.items():
        if key in quantized_keys:
            continue
        if isinstance(value, torch.Tensor):
            dense_fp16[key] = value.detach().cpu().to(torch.float16)
        else:
            dense_fp16[key] = value
    return {
        "__format__": TORCH_FP4_FORMAT,
        "block_size": int(block_size),
        "quantized": quantized,
        "dense_fp16": dense_fp16,
    }


def encode_torch_fp4_state_dict(
    model_or_state: JointFrameGenerator | dict[str, Any],
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> bytes:
    """Serialize a PR #63-style Torch-FP4 payload deterministically.

    The public PR #63 file uses Torch's legacy pickle serialization, which
    embeds process-local storage ids. The zip serializer is deterministic in
    this environment and Brotli-compresses within a few KB of the legacy form,
    so use it for contest custody.
    """
    payload = encode_torch_fp4_payload(model_or_state, block_size=block_size)
    out = io.BytesIO()
    torch.save(payload, out)
    return out.getvalue()


def decode_torch_fp4_payload(
    payload: dict[str, Any], *, device: torch.device | str
) -> dict[str, torch.Tensor]:
    """Decode a PR #63-style payload dict into a JointFrameGenerator state dict."""
    if not is_torch_fp4_payload(payload):
        raise ValueError(
            "not a PR63-style Torch-FP4 JointFrameGenerator payload"
        )
    state: dict[str, torch.Tensor] = {}
    quantized = payload["quantized"]
    dense_fp16 = payload["dense_fp16"]
    for key, entry in quantized.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Torch-FP4 quantized entry {key!r} is not a dict")
        if "packed" in entry:
            weight_key = key
            weight = _dequantize_fp4_tensor(
                entry["packed"],
                entry["scales"],
                tuple(entry["shape"]),
                device=device,
            )
            state[weight_key] = weight
            continue
        weight_key = f"{key}.weight"
        if "packed_weight" in entry:
            weight = _dequantize_fp4_tensor(
                entry["packed_weight"],
                entry["scales_fp16"],
                tuple(entry["weight_shape"]),
                device=device,
            )
            state[weight_key] = weight
        elif "weight_fp16" in entry:
            value = entry["weight_fp16"]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Torch-FP4 entry {key!r} has non-tensor weight_fp16")
            state[weight_key] = value.detach().to(device).to(torch.float32)
        else:
            raise ValueError(f"Torch-FP4 entry {key!r} has no recognized weight payload")
        bias = entry.get("bias_fp16")
        if isinstance(bias, torch.Tensor):
            state[f"{key}.bias"] = bias.detach().to(device).to(torch.float32)
    for key, value in dense_fp16.items():
        if isinstance(value, torch.Tensor):
            state[key] = value.detach().to(device).to(torch.float32)
        else:
            state[key] = value
    return state


def load_torch_fp4_payload(
    payload: dict[str, Any], *, device: torch.device | str
) -> JointFrameGenerator:
    """Load a PR #63-style payload dict into JointFrameGenerator."""
    state = decode_torch_fp4_payload(payload, device=device)
    model = build_quantizr_faithful_renderer()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def load_torch_fp4_bytes(
    data: bytes, *, device: torch.device | str
) -> JointFrameGenerator:
    """Load raw ``torch.save`` bytes into JointFrameGenerator."""
    payload = torch.load(
        io.BytesIO(data), map_location=device, weights_only=False
    )
    return load_torch_fp4_payload(payload, device=device)
