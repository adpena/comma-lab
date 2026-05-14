# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("torch is required for tiny frame self-compression") from exc
    return torch, nn


@dataclass(frozen=True)
class TinyFrameSelfCompressionArtifact:
    data: bytes
    summary: dict[str, Any]


def select_tiny_frame_linear_layers(
    model,
    *,
    layer_names: tuple[str, ...] | list[str] | None = ("output_projection",),
    largest_layers: int | None = None,
):
    _, nn = _require_torch()

    if layer_names is not None and largest_layers is not None:
        raise ValueError("specify either layer_names or largest_layers, not both")

    linear_layers = tuple(
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    )
    if not linear_layers:
        raise ValueError("model does not expose any torch.nn.Linear layers")

    linear_by_name = {name: module for name, module in linear_layers}
    if layer_names is not None:
        selected = []
        for name in layer_names:
            if name not in linear_by_name:
                raise ValueError(f"linear layer not found: {name}")
            selected.append((name, linear_by_name[name]))
        return tuple(selected)

    if largest_layers is None or int(largest_layers) <= 0:
        raise ValueError("largest_layers must be a positive integer when layer_names is omitted")

    ordered = sorted(
        linear_layers,
        key=lambda item: (-int(item[1].weight.numel()), item[0]),
    )
    return tuple(ordered[: int(largest_layers)])


def _pack_unsigned_values(values: list[int], bits: int) -> bytes:
    if bits == 8:
        return bytes(value & 0xFF for value in values)

    packed = bytearray()
    bit_buffer = 0
    bits_in_buffer = 0
    mask = (1 << bits) - 1
    for value in values:
        bit_buffer |= (value & mask) << bits_in_buffer
        bits_in_buffer += bits
        while bits_in_buffer >= 8:
            packed.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bits_in_buffer -= 8
    if bits_in_buffer:
        packed.append(bit_buffer & 0xFF)
    return bytes(packed)


def _compress_linear_layer(layer, *, name: str, weight_bits: int) -> tuple[bytes, bytes, dict[str, Any], dict[str, Any]]:
    torch, _ = _require_torch()

    weight = layer.weight.detach().to(device="cpu", dtype=torch.float32)
    rows, cols = (int(weight.shape[0]), int(weight.shape[1]))
    levels = 1 << int(weight_bits)
    half = levels // 2
    signed_max = max(half - 1, 1)

    row_scales: list[float] = []
    unsigned_values: list[int] = []
    for row in weight:
        scale = max(float(row.abs().max().item()), 1e-8)
        row_scales.append(scale)
        quantized = (row / scale * signed_max).round().clamp(-signed_max, signed_max).to(torch.int64)
        unsigned = (quantized + half).clamp(0, levels - 1)
        unsigned_values.extend(int(value) for value in unsigned.tolist())

    row_scale_blob = bytearray()
    for scale in row_scales:
        row_scale_blob.extend(struct.pack("<e", scale))
    packed_weight_blob = _pack_unsigned_values(unsigned_values, int(weight_bits))
    weight_blob = bytes(row_scale_blob) + packed_weight_blob

    bias_blob = b""
    bias_values = 0
    if layer.bias is not None:
        bias_tensor = layer.bias.detach().to(device="cpu", dtype=torch.float32)
        bias_values = int(bias_tensor.numel())
        bias_blob = struct.pack(f"<{bias_values}e", *[float(value) for value in bias_tensor.tolist()])

    packed_weight_bytes = int(math.ceil(rows * cols * int(weight_bits) / 8.0))
    row_scale_bytes = rows * 2
    weight_payload_bytes = row_scale_bytes + packed_weight_bytes
    bias_payload_bytes = len(bias_blob)
    total_payload_bytes = weight_payload_bytes + bias_payload_bytes
    baseline_bytes = (rows * cols + bias_values) * 4

    header_layer = {
        "bias_payload_bytes": bias_payload_bytes,
        "cols": cols,
        "has_bias": layer.bias is not None,
        "name": name,
        "rows": rows,
        "weight_bits": int(weight_bits),
        "weight_payload_bytes": weight_payload_bytes,
    }
    summary_layer = {
        "name": name,
        "rows": rows,
        "cols": cols,
        "weight_bits": int(weight_bits),
        "weight_values": rows * cols,
        "row_scale_bytes": row_scale_bytes,
        "packed_weight_bytes": packed_weight_bytes,
        "weight_payload_bytes": weight_payload_bytes,
        "bias_values": bias_values,
        "bias_payload_bytes": bias_payload_bytes,
        "total_payload_bytes": total_payload_bytes,
        "baseline_bytes": baseline_bytes,
        "compression_ratio": round(baseline_bytes / max(total_payload_bytes, 1), 4),
    }
    return weight_blob, bias_blob, header_layer, summary_layer


def export_tiny_frame_self_compression(
    model,
    *,
    layer_names: tuple[str, ...] | list[str] | None = ("output_projection",),
    largest_layers: int | None = None,
    weight_bits: int = 4,
) -> TinyFrameSelfCompressionArtifact:
    if int(weight_bits) < 2 or int(weight_bits) > 8:
        raise ValueError("weight_bits must be between 2 and 8")

    selected_layers = select_tiny_frame_linear_layers(
        model,
        layer_names=layer_names,
        largest_layers=largest_layers,
    )

    header_layers: list[dict[str, Any]] = []
    summary_layers: list[dict[str, Any]] = []
    payload = bytearray()
    for name, layer in selected_layers:
        weight_blob, bias_blob, header_layer, summary_layer = _compress_linear_layer(
            layer,
            name=name,
            weight_bits=int(weight_bits),
        )
        header_layers.append(header_layer)
        summary_layers.append(summary_layer)
        payload.extend(weight_blob)
        payload.extend(bias_blob)

    header = {
        "format": "tiny_frame_linear_self_compress",
        "layers": header_layers,
        "version": 1,
        "weight_bits": int(weight_bits),
    }
    header_json = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")

    data = bytearray()
    data.extend(struct.pack("<I", len(header_json)))
    data.extend(header_json)
    data.extend(payload)

    payload_bytes = sum(int(layer["total_payload_bytes"]) for layer in summary_layers)
    summary = {
        "command": "lossless_tiny_frame_self_compress_summary",
        "format": header["format"],
        "version": header["version"],
        "weight_bits": int(weight_bits),
        "target_layer_names": [name for name, _ in selected_layers],
        "layer_count": len(summary_layers),
        "header_bytes": 4 + len(header_json),
        "payload_bytes": payload_bytes,
        "total_bytes": len(data),
        "layers": summary_layers,
    }
    return TinyFrameSelfCompressionArtifact(data=bytes(data), summary=summary)


def tiny_frame_self_compression_byte_count(
    model=None,
    *,
    artifact_path: str | Path | None = None,
    layer_names: tuple[str, ...] | list[str] | None = ("output_projection",),
    largest_layers: int | None = None,
    weight_bits: int = 4,
) -> int:
    if artifact_path is not None:
        return int(Path(artifact_path).stat().st_size)
    if model is None:
        raise ValueError("model is required when artifact_path is not provided")
    artifact = export_tiny_frame_self_compression(
        model,
        layer_names=layer_names,
        largest_layers=largest_layers,
        weight_bits=weight_bits,
    )
    return int(artifact.summary["total_bytes"])


__all__ = [
    "TinyFrameSelfCompressionArtifact",
    "export_tiny_frame_self_compression",
    "select_tiny_frame_linear_layers",
    "tiny_frame_self_compression_byte_count",
]
