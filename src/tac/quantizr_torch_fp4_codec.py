"""PR #63-style Torch FP4 payload codec for JointFrameGenerator.

The public qpose14 submission stores a ``torch.save`` dictionary with
block-FP4 Conv/Embedding weights plus FP16 dense tensors.  This module keeps
that format available as a conservative, current-floor-compatible alternative
to the smaller QZS3 packer.  It is a build/runtime codec only; any archive
using it remains non-promotable until exact CUDA auth eval lands.

Example:
    >>> from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
    >>> payload = encode_torch_fp4_state_dict(build_quantizr_faithful_renderer())
    >>> model = load_torch_fp4_bytes(payload)
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import torch
import torch.nn as nn

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
PROTECTED_MODULES = {
    "shared_trunk.embedding",
    "frame1_head.head",
    "frame2_head.head",
}


def is_torch_fp4_payload(payload: object) -> bool:
    """Return true for the public qpose14 Torch-FP4 payload shape."""

    if not isinstance(payload, dict):
        return False
    fmt = payload.get("__format__")
    return (
        isinstance(fmt, str)
        and fmt.startswith(TORCH_FP4_FORMAT)
        and isinstance(payload.get("quantized"), dict)
        and isinstance(payload.get("dense_fp16"), dict)
    )


def _module_type(module: nn.Module) -> str | None:
    if isinstance(module, nn.Conv2d):
        return "conv2d"
    if isinstance(module, nn.Embedding):
        return "embedding"
    return None


def _quantize_fp4_tensor(tensor: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    flat = tensor.detach().cpu().float().numpy().reshape(-1)
    pad = (-flat.size) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    blocks = flat.reshape(-1, block_size)
    scales = np.maximum(np.max(np.abs(blocks), axis=1) / float(FP4_POS_LEVELS[-1]), 1e-8)
    scaled_abs = np.abs(blocks) / scales[:, None]
    idx = np.abs(scaled_abs[..., None] - FP4_POS_LEVELS[None, None, :]).argmin(axis=2)
    signs = (blocks < 0).astype(np.uint8) << 3
    nibbles = (signs | idx.astype(np.uint8)).reshape(-1)
    packed = np.frombuffer(_pack_nibbles(nibbles), dtype=np.uint8).copy()
    return torch.from_numpy(packed), torch.from_numpy(scales.astype(np.float16, copy=False))


def _dequantize_fp4_tensor(
    packed_weight: torch.Tensor,
    scales_fp16: torch.Tensor,
    weight_shape: object,
    *,
    device: str | torch.device,
) -> torch.Tensor:
    shape = tuple(int(dim) for dim in weight_shape)  # type: ignore[arg-type]
    count = int(np.prod(shape))
    packed = packed_weight.detach().cpu().numpy().astype(np.uint8, copy=False).tobytes()
    scales = scales_fp16.detach().cpu().numpy().astype(np.float16, copy=False)
    if scales.size == 0:
        return torch.empty(shape, dtype=torch.float32, device=device)
    block_size = (len(packed) * 2) // int(scales.size)
    nibbles = _unpack_nibbles(packed, int(scales.size) * block_size).reshape(scales.size, block_size)
    signs = (nibbles >> 3).astype(bool)
    mag_idx = (nibbles & 0x7).astype(np.int64)
    values = FP4_POS_LEVELS[mag_idx] * scales.astype(np.float32)[:, None]
    values = np.where(signs, -values, values).reshape(-1)[:count]
    return torch.from_numpy(values.reshape(shape).astype(np.float32, copy=False)).to(device)


def encode_torch_fp4_payload(
    model_or_state: JointFrameGenerator | dict[str, torch.Tensor],
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
    if list(state.keys()) != list(template_state.keys()):
        missing = [k for k in template_state if k not in state]
        extra = [k for k in state if k not in template_state]
        raise ValueError(
            "state dict is not JointFrameGenerator-compatible: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    export: dict[str, Any] = {
        "__format__": TORCH_FP4_FORMAT,
        "__block_size__": int(block_size),
        "__codebook__": torch.from_numpy(FP4_POS_LEVELS.copy()),
        "quantized": {},
        "dense_fp16": {},
    }
    covered_keys: set[str] = set()
    for name, module in model.named_modules():
        kind = _module_type(module)
        if kind is None:
            continue
        weight = module.weight.detach().float().cpu()
        rec: dict[str, Any] = {
            "type": kind,
            "weight_shape": list(weight.shape),
        }
        covered_keys.add(f"{name}.weight")
        if isinstance(module, nn.Conv2d):
            rec.update(
                {
                    "stride": list(module.stride),
                    "padding": list(module.padding),
                    "dilation": list(module.dilation),
                    "groups": int(module.groups),
                    "bias_fp16": module.bias.detach().half().cpu()
                    if module.bias is not None
                    else None,
                }
            )
            if module.bias is not None:
                covered_keys.add(f"{name}.bias")
        if name in PROTECTED_MODULES:
            rec.update({"weight_kind": "fp16", "weight_fp16": weight.half().cpu()})
        else:
            packed, scales = _quantize_fp4_tensor(weight, block_size)
            rec.update(
                {
                    "weight_kind": "fp4_packed",
                    "weight_numel": int(weight.numel()),
                    "packed_weight": packed.cpu(),
                    "scales_fp16": scales.cpu(),
                }
            )
        export["quantized"][name] = rec

    for key, value in state.items():
        if key in covered_keys:
            continue
        tensor = value.detach().cpu()
        export["dense_fp16"][key] = tensor.half() if torch.is_floating_point(tensor) else tensor
    return export


def encode_torch_fp4_state_dict(
    model_or_state: JointFrameGenerator | dict[str, torch.Tensor],
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> bytes:
    """Serialize a PR #63-style Torch-FP4 payload deterministically.

    The public PR #63 file uses Torch's legacy pickle serialization, which
    embeds process-local storage ids.  The zip serializer is deterministic in
    this environment and Brotli-compresses within a few KB of the legacy form,
    so use it for contest custody.
    """

    payload = encode_torch_fp4_payload(model_or_state, block_size=block_size)
    out = io.BytesIO()
    torch.save(payload, out)
    return out.getvalue()


def decode_torch_fp4_payload(
    payload: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Decode a PR #63-style payload dict into a JointFrameGenerator state dict."""

    if not is_torch_fp4_payload(payload):
        raise ValueError("not a PR63-style Torch-FP4 JointFrameGenerator payload")
    state: dict[str, torch.Tensor] = {}
    for name, rec in payload["quantized"].items():
        if not isinstance(rec, dict):
            raise ValueError(f"invalid quantized record for {name!r}")
        if rec.get("weight_kind") == "fp4_packed":
            weight = _dequantize_fp4_tensor(
                rec["packed_weight"],
                rec["scales_fp16"],
                rec["weight_shape"],
                device=device,
            )
        elif rec.get("weight_fp16") is not None:
            weight = rec["weight_fp16"].to(device).float()
        else:
            raise ValueError(f"unsupported Torch-FP4 weight record for {name!r}")
        state[f"{name}.weight"] = weight.float()
        bias = rec.get("bias_fp16")
        if bias is not None:
            state[f"{name}.bias"] = bias.to(device).float()
    for name, tensor in payload["dense_fp16"].items():
        state[name] = tensor.to(device).float() if torch.is_floating_point(tensor) else tensor.to(device)
    return state


def load_torch_fp4_payload(
    payload: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> JointFrameGenerator:
    """Load a PR #63-style payload dict into JointFrameGenerator."""

    state = decode_torch_fp4_payload(payload, device=device)
    model = build_quantizr_faithful_renderer()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def load_torch_fp4_bytes(
    data: bytes,
    *,
    device: str | torch.device = "cpu",
) -> JointFrameGenerator:
    """Load raw ``torch.save`` bytes into JointFrameGenerator."""

    payload = torch.load(io.BytesIO(data), map_location=device, weights_only=False)
    return load_torch_fp4_payload(payload, device=device)
