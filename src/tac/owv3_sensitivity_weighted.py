"""Lane Ω-W-V3 — sensitivity-weighted renderer archive.

Ω-W-V3 is the β fix for the Ω-W-V2 failure mode: uniform renderer-weight
compression saved bytes but over-perturbed PoseNet-sensitive channels. V3
sends low-sensitivity output channels through the existing OWV2 water-fill +
arithmetic codec and, by default, keeps high-sensitivity/fallback channels in
the compact ASYM-style representation instead of silently inflating them to
FP16. The legacy FP16 protection path is still available for explicit
smoke/debug artifacts only.

Sensitivity computation is compress-time only. The decode/inflate path reads
only the bytes in this archive and never imports or runs contest scorers.
"""
from __future__ import annotations

import json
import struct
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn

from tac.sensitivity_map import (
    SensitivityMapError,
    resolve_layer_sensitivity,
    validate_sensitivity_map_for_model,
)
from tac.water_filling_codec_v2 import (
    BlockFPIneligible,
    GateRegression,
    decode_omega_w_v2,
    encode_omega_w_v2,
)


OWV3_ARCHIVE_MAGIC: bytes = b"OWV3"
OWV3_ARCHIVE_VERSION: int = 1
OWV3_DEFAULT_ASYM_BITS: int = 8
OWV3_FALLBACK_ACTION_KEEP_ASYM: str = "keep_asym"
OWV3_FALLBACK_ACTION_ERROR: str = "error"
OWV3_FALLBACK_ACTION_DIAGNOSTIC_FP16: str = "diagnostic_fp16"
OWV3_FALLBACK_ACTIONS: frozenset[str] = frozenset({
    OWV3_FALLBACK_ACTION_KEEP_ASYM,
    OWV3_FALLBACK_ACTION_ERROR,
    OWV3_FALLBACK_ACTION_DIAGNOSTIC_FP16,
})


class OWV3ArchiveError(ValueError):
    """Raised for malformed or non-compliant OWV3 artifacts."""


def _validate_thresholds(protect_threshold: float, aggressive_threshold: float) -> None:
    if not np.isfinite(protect_threshold) or not np.isfinite(aggressive_threshold):
        raise OWV3ArchiveError("OWV3 thresholds must be finite")
    if protect_threshold <= 0.0:
        raise OWV3ArchiveError("protect_threshold must be > 0")
    if aggressive_threshold < 0.0:
        raise OWV3ArchiveError("aggressive_threshold must be >= 0")
    if aggressive_threshold >= protect_threshold:
        raise OWV3ArchiveError(
            "aggressive_threshold must be < protect_threshold"
        )


def _validate_fallback_action(fallback_action: str) -> str:
    action = str(fallback_action)
    if action not in OWV3_FALLBACK_ACTIONS:
        raise OWV3ArchiveError(
            "fallback_action must be one of "
            f"{sorted(OWV3_FALLBACK_ACTIONS)}, got {fallback_action!r}"
        )
    return action


def _eligible_for_owv3(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Conv2d)
        and module.weight.dim() == 4
        and int(module.weight.shape[0]) >= 2
    )


def _v1_raw_byte_estimate(weight: torch.Tensor) -> int:
    o, i, kh, kw = weight.shape
    return int(o * i * kh * kw) + int(o * 4) + 32


def _derive_total_bits(weight: torch.Tensor, ratio: float) -> int:
    if ratio <= 0.0 or ratio >= 1.0:
        raise OWV3ArchiveError(
            f"bit_budget_ratio={ratio} must be in (0, 1)"
        )
    return int(_v1_raw_byte_estimate(weight) * ratio * 8)


def _fp16_blob(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().float().numpy().astype("float16").tobytes()


def _read_fp16_to_tensor(blob: bytes, shape: list[int]) -> torch.Tensor:
    expected = int(np.prod(shape)) * 2
    if len(blob) != expected:
        raise OWV3ArchiveError(
            f"fp16 blob length {len(blob)} does not match shape {shape} "
            f"({expected} bytes expected)"
        )
    arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32).copy()
    return torch.from_numpy(arr).reshape(shape)


def _pack_values(buf: bytearray, values: list[int], bits: int) -> None:
    if bits == 8:
        buf.extend(bytes(v & 0xFF for v in values))
        return
    bit_buffer = 0
    bits_in_buffer = 0
    for v in values:
        bit_buffer |= (v & ((1 << bits) - 1)) << bits_in_buffer
        bits_in_buffer += bits
        while bits_in_buffer >= 8:
            buf.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bits_in_buffer -= 8
    if bits_in_buffer > 0:
        buf.append(bit_buffer & 0xFF)


def _unpack_values(
    data: bytes | bytearray | memoryview,
    offset: int,
    count: int,
    bits: int,
) -> tuple[list[int], int]:
    if bits == 8:
        return [int(data[offset + i]) for i in range(count)], offset + count
    total_bytes = (count * bits + 7) // 8
    raw = data[offset:offset + total_bytes]
    bit_buffer = int.from_bytes(bytes(raw), byteorder="little")
    mask = (1 << bits) - 1
    values = []
    for _ in range(count):
        values.append(bit_buffer & mask)
        bit_buffer >>= bits
    return values, offset + total_bytes


def _quantize_tensor_uniform(tensor: torch.Tensor, bits: int) -> tuple[float, list[int]]:
    bits = max(int(bits), 2)
    flat = tensor.detach().cpu().reshape(-1).float()
    abs_max = flat.abs().max().clamp(min=6.2e-5).item()
    n_levels = 2 ** bits
    half = n_levels // 2
    quantized = (
        (flat / abs_max * (half - 1))
        .round()
        .clamp(-(half - 1), half - 1)
        .long()
    )
    unsigned = (quantized + half).clamp(0, n_levels - 1).tolist()
    return abs_max, [int(v) for v in unsigned]


def _dequantize_values(values: list[int], bits: int, scale: float) -> torch.Tensor:
    bits = max(int(bits), 2)
    n_levels = 2 ** bits
    half = n_levels // 2
    return torch.tensor(
        [(int(v) - half) / max(half - 1, 1) * float(scale) for v in values],
        dtype=torch.float32,
    )


def _asym_channel_geometry(
    shape: list[int],
    *,
    transposed: bool = False,
) -> tuple[int, int, list[int]]:
    if len(shape) == 2:
        return int(shape[0]), int(shape[1]), [int(shape[1])]
    if len(shape) != 4:
        raise OWV3ArchiveError(f"ASYM channel packing expects rank 2 or 4, got {shape}")
    if transposed:
        return int(shape[1]), int(shape[0] * shape[2] * shape[3]), [
            int(shape[0]),
            int(shape[2]),
            int(shape[3]),
        ]
    return int(shape[0]), int(shape[1] * shape[2] * shape[3]), [
        int(shape[1]),
        int(shape[2]),
        int(shape[3]),
    ]


def _asym_pack_channels(
    weight: torch.Tensor,
    indices: list[int],
    *,
    transposed: bool = False,
    bits: int = OWV3_DEFAULT_ASYM_BITS,
) -> bytes:
    shape = [int(v) for v in weight.shape]
    c_out, _fan_in, _ch_shape = _asym_channel_geometry(shape, transposed=transposed)
    packed = bytearray()
    for ch_idx in indices:
        if ch_idx < 0 or ch_idx >= c_out:
            raise OWV3ArchiveError(
                f"channel index {ch_idx} out of range for shape {shape}"
            )
        ch_weight = (
            weight[:, ch_idx] if transposed and len(shape) == 4 else weight[ch_idx]
        ).reshape(-1)
        scale, unsigned = _quantize_tensor_uniform(ch_weight, bits)
        packed.extend(struct.pack("<e", scale))
        _pack_values(packed, unsigned, bits)
    return bytes(packed)


def _asym_read_channels(
    blob: bytes,
    shape: list[int],
    indices: list[int],
    *,
    transposed: bool = False,
    bits: int = OWV3_DEFAULT_ASYM_BITS,
) -> torch.Tensor:
    c_out, fan_in, ch_shape = _asym_channel_geometry(shape, transposed=transposed)
    out = torch.zeros(shape, dtype=torch.float32)
    offset = 0
    for ch_idx in indices:
        if ch_idx < 0 or ch_idx >= c_out:
            raise OWV3ArchiveError(
                f"channel index {ch_idx} out of range for shape {shape}"
            )
        if offset + 2 > len(blob):
            raise OWV3ArchiveError("truncated ASYM channel scale")
        scale = struct.unpack("<e", blob[offset:offset + 2])[0]
        offset += 2
        values, offset = _unpack_values(blob, offset, fan_in, bits)
        dequant = _dequantize_values(values, bits, scale).reshape(ch_shape)
        if transposed and len(shape) == 4:
            out[:, ch_idx] = dequant
        else:
            out[ch_idx] = dequant
    if offset != len(blob):
        raise OWV3ArchiveError(
            f"ASYM channel blob had {len(blob) - offset} trailing byte(s)"
        )
    return out


def _asym_pack_embedding(
    weight: torch.Tensor,
    *,
    bits: int = OWV3_DEFAULT_ASYM_BITS,
) -> bytes:
    packed = bytearray()
    scale, unsigned = _quantize_tensor_uniform(weight.reshape(-1), bits)
    packed.extend(struct.pack("<e", scale))
    _pack_values(packed, unsigned, bits)
    return bytes(packed)


def _asym_read_embedding(
    blob: bytes,
    shape: list[int],
    *,
    bits: int = OWV3_DEFAULT_ASYM_BITS,
) -> torch.Tensor:
    count = int(np.prod(shape))
    if len(blob) < 2:
        raise OWV3ArchiveError("truncated ASYM embedding scale")
    scale = struct.unpack("<e", blob[:2])[0]
    values, offset = _unpack_values(blob, 2, count, bits)
    if offset != len(blob):
        raise OWV3ArchiveError(
            f"ASYM embedding blob had {len(blob) - offset} trailing byte(s)"
        )
    return _dequantize_values(values, bits, scale).reshape(shape)


def _asym_pack_bias(
    bias: torch.Tensor | None,
    *,
    c_out: int,
    bits: int = OWV3_DEFAULT_ASYM_BITS,
) -> bytes:
    if bias is None:
        return b""
    packed = bytearray()
    n_levels = 2 ** max(int(bits), 2)
    half = n_levels // 2
    b = bias.detach().cpu().float()
    for ch_idx in range(c_out):
        b_val = float(b[ch_idx].item())
        abs_max_b = max(abs(b_val), 6.2e-5)
        q = int(round(b_val / abs_max_b * (half - 1)))
        q = max(-(half - 1), min(half - 1, q))
        packed.extend(struct.pack("<e", abs_max_b))
        packed.extend(struct.pack("<H", q + half))
    return bytes(packed)


def _asym_read_bias(
    blob: bytes,
    *,
    c_out: int,
    bits: int = OWV3_DEFAULT_ASYM_BITS,
) -> torch.Tensor:
    expected = int(c_out) * 4
    if len(blob) != expected:
        raise OWV3ArchiveError(
            f"ASYM bias blob length {len(blob)} != expected {expected}"
        )
    n_levels = 2 ** max(int(bits), 2)
    half = n_levels // 2
    out = torch.zeros(c_out, dtype=torch.float32)
    offset = 0
    for ch_idx in range(c_out):
        scale_b = struct.unpack("<e", blob[offset:offset + 2])[0]
        offset += 2
        u_val = struct.unpack("<H", blob[offset:offset + 2])[0]
        offset += 2
        out[ch_idx] = (int(u_val) - half) / max(half - 1, 1) * float(scale_b)
    return out


def _classify_channels(
    sensitivity: torch.Tensor,
    *,
    protect_threshold: float,
) -> tuple[list[int], list[int]]:
    """Return `(quant_indices, protected_indices)`."""
    protected = (sensitivity > protect_threshold).nonzero(as_tuple=False).flatten()
    quant = (sensitivity <= protect_threshold).nonzero(as_tuple=False).flatten()
    return [int(i) for i in quant.tolist()], [int(i) for i in protected.tolist()]


def _emit_fp16_conv_layer(
    *,
    name: str,
    module: nn.Module,
    body_chunks: list[bytes],
    layers_meta: list[dict],
    kind: str = "fp16_conv",
    fallback_reason: str | None = None,
) -> None:
    w_blob = _fp16_blob(module.weight)
    b_blob = b""
    if getattr(module, "bias", None) is not None:
        b_blob = _fp16_blob(module.bias)
    body_chunks.append(w_blob + b_blob)
    meta = {
        "name": name,
        "kind": kind,
        "shape": list(module.weight.shape),
        "stride": module.stride[0] if isinstance(module.stride, tuple) else module.stride,
        "padding": module.padding[0] if isinstance(module.padding, tuple) else module.padding,
        "dilation": module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation,
        "groups": module.groups,
        "padding_mode": module.padding_mode,
        "out_channels": int(getattr(module, "out_channels", module.weight.shape[0])),
        "has_bias": module.bias is not None,
        "weight_blob_len": len(w_blob),
        "bias_blob_len": len(b_blob),
    }
    if fallback_reason:
        meta["fallback_reason"] = fallback_reason
    layers_meta.append(meta)


def _emit_fp16_linear_layer(
    *,
    name: str,
    module: nn.Linear,
    body_chunks: list[bytes],
    layers_meta: list[dict],
    fallback_reason: str | None = None,
) -> None:
    w_blob = _fp16_blob(module.weight)
    b_blob = b""
    if module.bias is not None:
        b_blob = _fp16_blob(module.bias)
    body_chunks.append(w_blob + b_blob)
    meta = {
        "name": name,
        "kind": "fp16_linear",
        "shape": list(module.weight.shape),
        "has_bias": module.bias is not None,
        "weight_blob_len": len(w_blob),
        "bias_blob_len": len(b_blob),
    }
    if fallback_reason:
        meta["fallback_reason"] = fallback_reason
    layers_meta.append(meta)


def _emit_asym_embedding_layer(
    *,
    name: str,
    module: nn.Embedding,
    body_chunks: list[bytes],
    layers_meta: list[dict],
    bits: int = OWV3_DEFAULT_ASYM_BITS,
) -> None:
    blob = _asym_pack_embedding(module.weight.detach().cpu().float(), bits=bits)
    body_chunks.append(blob)
    layers_meta.append({
        "name": name,
        "kind": "asym_emb",
        "shape": list(module.weight.shape),
        "bits": bits,
        "blob_len": len(blob),
        "codec_action": OWV3_FALLBACK_ACTION_KEEP_ASYM,
        "charged_bytes": {"keep_asym": len(blob)},
    })


def _emit_asym_affine_layer(
    *,
    name: str,
    module: nn.Module,
    body_chunks: list[bytes],
    layers_meta: list[dict],
    kind: str,
    fallback_reason: str | None = None,
    bits: int = OWV3_DEFAULT_ASYM_BITS,
) -> None:
    w = module.weight.detach().cpu().float()
    transposed = isinstance(module, nn.ConvTranspose2d)
    c_out, _fan_in, _ch_shape = _asym_channel_geometry(
        list(w.shape),
        transposed=transposed,
    )
    w_blob = _asym_pack_channels(
        w,
        list(range(c_out)),
        transposed=transposed,
        bits=bits,
    )
    b_blob = _asym_pack_bias(
        module.bias.detach().cpu().float() if getattr(module, "bias", None) is not None else None,
        c_out=c_out,
        bits=bits,
    )
    body_chunks.append(w_blob + b_blob)
    meta = {
        "name": name,
        "kind": kind,
        "shape": list(w.shape),
        "bits": bits,
        "has_bias": getattr(module, "bias", None) is not None,
        "weight_blob_len": len(w_blob),
        "bias_blob_len": len(b_blob),
        "bias_codec": "asym" if getattr(module, "bias", None) is not None else "none",
        "codec_action": OWV3_FALLBACK_ACTION_KEEP_ASYM,
        "charged_bytes": {
            OWV3_FALLBACK_ACTION_KEEP_ASYM: len(w_blob),
            "bias": len(b_blob),
        },
    }
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        meta.update({
            "stride": module.stride[0] if isinstance(module.stride, tuple) else module.stride,
            "padding": module.padding[0] if isinstance(module.padding, tuple) else module.padding,
            "dilation": module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation,
            "groups": module.groups,
            "padding_mode": module.padding_mode,
            "out_channels": c_out,
        })
    if fallback_reason:
        meta["fallback_reason"] = fallback_reason
    layers_meta.append(meta)


def _emit_fallback_affine_layer(
    *,
    name: str,
    module: nn.Module,
    body_chunks: list[bytes],
    layers_meta: list[dict],
    fallback_action: str,
    fallback_reason: str,
    asym_kind: str,
    fp16_kind: str,
) -> None:
    if fallback_action == OWV3_FALLBACK_ACTION_ERROR:
        raise OWV3ArchiveError(
            f"{name}: OWV3 promotion fallback blocked ({fallback_reason})"
        )
    if fallback_action == OWV3_FALLBACK_ACTION_DIAGNOSTIC_FP16:
        if isinstance(module, nn.Linear):
            _emit_fp16_linear_layer(
                name=name,
                module=module,
                body_chunks=body_chunks,
                layers_meta=layers_meta,
                fallback_reason=fallback_reason,
            )
        else:
            _emit_fp16_conv_layer(
                name=name,
                module=module,
                body_chunks=body_chunks,
                layers_meta=layers_meta,
                kind=fp16_kind,
                fallback_reason=fallback_reason,
            )
        layers_meta[-1]["codec_action"] = OWV3_FALLBACK_ACTION_DIAGNOSTIC_FP16
        layers_meta[-1]["promotion_eligible"] = False
        layers_meta[-1]["charged_bytes"] = {
            OWV3_FALLBACK_ACTION_DIAGNOSTIC_FP16: (
                layers_meta[-1]["weight_blob_len"]
                + layers_meta[-1].get("bias_blob_len", 0)
            )
        }
        return
    _emit_asym_affine_layer(
        name=name,
        module=module,
        body_chunks=body_chunks,
        layers_meta=layers_meta,
        kind=asym_kind,
        fallback_reason=fallback_reason,
    )


def _summarize_action_counts(layers_meta: list[dict]) -> dict[str, int]:
    counts = {
        "owv2_low_bit_layers": 0,
        "owv2_low_bit_channels": 0,
        "keep_asym_layers": 0,
        "keep_asym_channels": 0,
        "diagnostic_fp16_layers": 0,
        "fp16_protect_channels": 0,
    }
    for meta in layers_meta:
        action = meta.get("codec_action")
        if action == OWV3_FALLBACK_ACTION_KEEP_ASYM:
            counts["keep_asym_layers"] += 1
            counts["keep_asym_channels"] += int(meta.get("out_channels", 0) or 0)
        if action == OWV3_FALLBACK_ACTION_DIAGNOSTIC_FP16:
            counts["diagnostic_fp16_layers"] += 1

        if meta.get("kind") == "owv3_conv":
            quant_n = len(meta.get("quant_indices") or [])
            protected_n = len(meta.get("protected_indices") or [])
            if quant_n:
                counts["owv2_low_bit_layers"] += 1
                counts["owv2_low_bit_channels"] += quant_n
            if meta.get("protected_codec") == "asym":
                counts["keep_asym_channels"] += protected_n
            if meta.get("protected_codec") == "fp16":
                counts["fp16_protect_channels"] += protected_n
    return counts


def _summarize_charged_bytes(layers_meta: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for meta in layers_meta:
        charged = meta.get("charged_bytes") or {}
        if not isinstance(charged, Mapping):
            continue
        for action, value in charged.items():
            bytes_value = int(value or 0)
            if bytes_value == 0:
                continue
            out[str(action)] = out.get(str(action), 0) + bytes_value
    return dict(sorted(out.items()))


def _summarize_fallback_accounting(layers_meta: list[dict]) -> dict[str, dict[str, int]]:
    reason_counts: dict[str, int] = {}
    reason_bytes: dict[str, int] = {}
    for meta in layers_meta:
        reason = meta.get("fallback_reason")
        if not reason:
            continue
        key = str(reason)
        reason_counts[key] = reason_counts.get(key, 0) + 1
        charged = meta.get("charged_bytes") or {}
        charged_bytes = (
            sum(int(value or 0) for value in charged.values())
            if isinstance(charged, Mapping)
            else 0
        )
        reason_bytes[key] = reason_bytes.get(key, 0) + charged_bytes
    return {
        "fallback_reason_counts": dict(sorted(reason_counts.items())),
        "fallback_reason_bytes": dict(sorted(reason_bytes.items())),
    }


def _build_byte_plan(
    *,
    layers_meta: list[dict],
    fallback_action: str,
    body_len: int,
) -> dict[str, object]:
    fallback_accounting = _summarize_fallback_accounting(layers_meta)
    return {
        "fallback_action": fallback_action,
        "asym_bits": OWV3_DEFAULT_ASYM_BITS,
        "promotion_eligible": (
            fallback_action != OWV3_FALLBACK_ACTION_DIAGNOSTIC_FP16
        ),
        "charged_body_bytes": int(body_len),
        "action_counts": _summarize_action_counts(layers_meta),
        "action_bytes": _summarize_charged_bytes(layers_meta),
        **fallback_accounting,
    }


def encode_owv3_archive(
    model: nn.Module | None = None,
    *,
    sensitivities: Mapping[str, torch.Tensor] | None = None,
    bit_budget_ratio: float | None = None,
    protect_threshold: float = 1e-3,
    aggressive_threshold: float = 1e-5,
    require_all_conv_sensitivity: bool = False,
    fallback_action: str = OWV3_FALLBACK_ACTION_KEEP_ASYM,
    arch_extra: dict | None = None,
) -> bytes:
    """Encode a renderer into an OWV3 sensitivity-weighted archive."""
    if model is None:
        raise OWV3ArchiveError("encode_owv3_archive: model is required")
    if sensitivities is None:
        raise OWV3ArchiveError(
            "encode_owv3_archive: sensitivities are required"
        )
    _validate_thresholds(protect_threshold, aggressive_threshold)
    fallback_action = _validate_fallback_action(fallback_action)
    try:
        validate_sensitivity_map_for_model(
            sensitivities,
            model,
            require_all_conv=require_all_conv_sensitivity,
        )
    except SensitivityMapError as exc:
        raise OWV3ArchiveError(str(exc)) from exc

    ratio = 0.7 if bit_budget_ratio is None else float(bit_budget_ratio)
    from tac.renderer_export import _infer_asymmetric_config

    model.eval()
    arch = _infer_asymmetric_config(model)
    if arch_extra:
        arch.update(arch_extra)

    layers_meta: list[dict] = []
    body_chunks: list[bytes] = []
    seen_emb_ids: set[int] = set()

    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            if id(module) in seen_emb_ids:
                continue
            seen_emb_ids.add(id(module))
            if fallback_action == OWV3_FALLBACK_ACTION_ERROR:
                raise OWV3ArchiveError(
                    f"{name}: OWV3 fallback blocked for embedding keep_asym"
                )
            if fallback_action == OWV3_FALLBACK_ACTION_DIAGNOSTIC_FP16:
                blob = _fp16_blob(module.weight)
                body_chunks.append(blob)
                layers_meta.append({
                    "name": name,
                    "kind": "fp16_emb",
                    "shape": list(module.weight.shape),
                    "blob_len": len(blob),
                    "codec_action": OWV3_FALLBACK_ACTION_DIAGNOSTIC_FP16,
                    "promotion_eligible": False,
                    "charged_bytes": {
                        OWV3_FALLBACK_ACTION_DIAGNOSTIC_FP16: len(blob),
                    },
                })
            else:
                _emit_asym_embedding_layer(
                    name=name,
                    module=module,
                    body_chunks=body_chunks,
                    layers_meta=layers_meta,
                )
            continue

        if isinstance(module, nn.ConvTranspose2d):
            _emit_fallback_affine_layer(
                name=name,
                module=module,
                body_chunks=body_chunks,
                layers_meta=layers_meta,
                fallback_action=fallback_action,
                fallback_reason="convt_keep_asym_fallback",
                asym_kind="asym_convt",
                fp16_kind="fp16_convt",
            )
            continue

        if isinstance(module, nn.Conv2d):
            w = module.weight.detach().cpu().float()
            if not _eligible_for_owv3(module):
                _emit_fallback_affine_layer(
                    name=name,
                    module=module,
                    body_chunks=body_chunks,
                    layers_meta=layers_meta,
                    fallback_action=fallback_action,
                    fallback_reason="ineligible_for_owv3",
                    asym_kind="asym_conv",
                    fp16_kind="fp16_conv",
                )
                continue

            try:
                sens = resolve_layer_sensitivity(
                    sensitivities,
                    module_name=name,
                    weight=w,
                    required=True,
                )
            except SensitivityMapError as exc:
                raise OWV3ArchiveError(str(exc)) from exc
            assert sens is not None

            quant_idx, protected_idx = _classify_channels(
                sens,
                protect_threshold=protect_threshold,
            )

            if not quant_idx:
                _emit_fallback_affine_layer(
                    name=name,
                    module=module,
                    body_chunks=body_chunks,
                    layers_meta=layers_meta,
                    fallback_action=fallback_action,
                    fallback_reason="all_channels_protected",
                    asym_kind="asym_conv",
                    fp16_kind="fp16_conv",
                )
                continue

            w_quant = w[torch.tensor(quant_idx, dtype=torch.long)]
            hess = sens[torch.tensor(quant_idx, dtype=torch.long)].clamp_min(1e-12)
            total_bits = _derive_total_bits(w_quant, ratio)
            try:
                owv2_payload = encode_omega_w_v2(
                    weights_block_fp=w_quant,
                    hessian=hess,
                    total_bits=total_bits,
                )
            except (BlockFPIneligible, GateRegression) as exc:
                _emit_fallback_affine_layer(
                    name=name,
                    module=module,
                    body_chunks=body_chunks,
                    layers_meta=layers_meta,
                    fallback_action=fallback_action,
                    fallback_reason=f"owv2_gate:{type(exc).__name__}",
                    asym_kind="asym_conv",
                    fp16_kind="fp16_conv",
                )
                continue

            protected_blob = b""
            protected_codec = "none"
            if protected_idx:
                if fallback_action == OWV3_FALLBACK_ACTION_ERROR:
                    raise OWV3ArchiveError(
                        f"{name}: protected channels require fallback action "
                        f"({len(protected_idx)} protected)"
                    )
                if fallback_action == OWV3_FALLBACK_ACTION_DIAGNOSTIC_FP16:
                    protected_blob = _fp16_blob(
                        w[torch.tensor(protected_idx, dtype=torch.long)]
                    )
                    protected_codec = "fp16"
                else:
                    protected_blob = _asym_pack_channels(w, protected_idx)
                    protected_codec = "asym"
            b_blob = b""
            if module.bias is not None:
                b_blob = _fp16_blob(module.bias)

            body_chunks.append(owv2_payload + protected_blob + b_blob)
            layers_meta.append({
                "name": name,
                "kind": "owv3_conv",
                "shape": list(w.shape),
                "stride": module.stride[0] if isinstance(module.stride, tuple) else module.stride,
                "padding": module.padding[0] if isinstance(module.padding, tuple) else module.padding,
                "dilation": module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation,
                "groups": module.groups,
                "padding_mode": module.padding_mode,
                "has_bias": module.bias is not None,
                "quant_indices": quant_idx,
                "protected_indices": protected_idx,
                "weight_blob_len": len(owv2_payload),
                "protected_blob_len": len(protected_blob),
                "protected_codec": protected_codec,
                "bias_blob_len": len(b_blob),
                "bias_codec": "fp16" if module.bias is not None else "none",
                "bit_budget_ratio": ratio,
                "protect_threshold": protect_threshold,
                "aggressive_threshold": aggressive_threshold,
                "sensitivity_min": float(sens.min().item()),
                "sensitivity_max": float(sens.max().item()),
                "codec_action": "owv2_low_bit",
                "promotion_eligible": (
                    fallback_action != OWV3_FALLBACK_ACTION_DIAGNOSTIC_FP16
                ),
                "charged_bytes": {
                    "owv2_low_bit": len(owv2_payload),
                    protected_codec: len(protected_blob),
                    "bias": len(b_blob),
                },
            })
            continue

        if isinstance(module, nn.Linear):
            _emit_fallback_affine_layer(
                name=name,
                module=module,
                body_chunks=body_chunks,
                layers_meta=layers_meta,
                fallback_action=fallback_action,
                fallback_reason="linear_keep_asym_fallback",
                asym_kind="asym_linear",
                fp16_kind="fp16_linear",
            )
            continue

    captured: set[int] = set()
    for _n, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding)):
            for p in mod.parameters(recurse=False):
                captured.add(id(p))
    scalar_params: dict[str, float] = {}
    for pname, param in model.named_parameters():
        if id(param) in captured:
            continue
        if param.numel() == 1:
            scalar_params[pname] = float(param.item())

    body = b"".join(body_chunks)
    header = {
        "version": OWV3_ARCHIVE_VERSION,
        "format": "owv3_sensitivity_weighted_renderer_archive_v1",
        "arch": arch,
        "layers": layers_meta,
        "scalar_params": scalar_params,
        "body_len": len(body),
        "byte_plan": _build_byte_plan(
            layers_meta=layers_meta,
            fallback_action=fallback_action,
            body_len=len(body),
        ),
        "thresholds": {
            "protect_threshold": protect_threshold,
            "aggressive_threshold": aggressive_threshold,
        },
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    buf = bytearray()
    buf.extend(OWV3_ARCHIVE_MAGIC)
    buf.extend(struct.pack("<I", len(header_json)))
    buf.extend(header_json)
    buf.extend(struct.pack("<I", len(body)))
    buf.extend(body)
    return bytes(buf)


def _build_empty_renderer(arch: dict) -> nn.Module:
    from tac.renderer import AsymmetricPairGenerator, build_renderer

    pair_mode = arch.get("pair_mode", "asymmetric")
    if pair_mode == "asymmetric":
        return AsymmetricPairGenerator(
            num_classes=arch.get("num_classes", 5),
            embed_dim=arch.get("embed_dim", 6),
            base_ch=arch.get("base_ch", 36),
            mid_ch=arch.get("mid_ch", 60),
            motion_hidden=arch.get("motion_hidden", 32),
            depth=arch.get("depth", 1),
            max_flow_px=arch.get("max_flow_px", 20.0),
            max_residual=arch.get("max_residual", 20.0),
            flow_only=arch.get("flow_only", False),
            pose_dim=arch.get("pose_dim", 0),
            use_dsconv=arch.get("use_dsconv", False),
            use_ghost=arch.get("use_ghost", False),
            use_zoom_flow=bool(arch.get("use_zoom_flow") or False),
            padding_mode=arch.get("padding_mode", "zeros"),
            use_dilation=arch.get("use_dilation", False),
        )
    return build_renderer(
        num_classes=arch.get("num_classes", 5),
        embed_dim=arch.get("embed_dim", 6),
        base_ch=arch.get("base_ch", 36),
        mid_ch=arch.get("mid_ch", 60),
        motion_hidden=arch.get("motion_hidden", 32),
        depth=arch.get("depth", 1),
        pose_dim=arch.get("pose_dim", 0),
        use_dsconv=arch.get("use_dsconv", False),
        use_ghost=arch.get("use_ghost", False),
        padding_mode=arch.get("padding_mode", "zeros"),
        use_dilation=arch.get("use_dilation", False),
        use_zoom_flow=bool(arch.get("use_zoom_flow") or False),
        blend_mode=arch.get("blend_mode", "scalar"),
        noise_mode=arch.get("noise_mode", "deterministic"),
        motion_type=arch.get("motion_type", "learned_cnn"),
    )


def decode_owv3_archive(
    data: bytes | None = None,
    device: str | None = None,
) -> nn.Module:
    """Decode OWV3 bytes into an eval-mode renderer."""
    if data is None:
        raise OWV3ArchiveError("decode_owv3_archive: data is required")
    if device is None:
        raise OWV3ArchiveError("decode_owv3_archive: device is required")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise OWV3ArchiveError(
            f"decode_owv3_archive: data must be bytes-like, got {type(data).__name__}"
        )
    blob = bytes(data)
    if len(blob) < 12 or blob[:4] != OWV3_ARCHIVE_MAGIC:
        raise OWV3ArchiveError(
            f"decode_owv3_archive: bad/missing magic {blob[:4]!r}"
        )
    offset = 4
    (header_len,) = struct.unpack("<I", blob[offset:offset + 4])
    offset += 4
    if header_len <= 0 or offset + header_len + 4 > len(blob):
        raise OWV3ArchiveError(
            f"decode_owv3_archive: invalid header_len={header_len}"
        )
    header = json.loads(blob[offset:offset + header_len].decode("utf-8"))
    offset += header_len
    if not isinstance(header, dict):
        raise OWV3ArchiveError("decode_owv3_archive: header is not a JSON object")
    if header.get("version") != OWV3_ARCHIVE_VERSION:
        raise OWV3ArchiveError(
            f"decode_owv3_archive: unsupported version {header.get('version')!r}"
        )
    if offset + 4 > len(blob):
        raise OWV3ArchiveError("decode_owv3_archive: missing body length")
    (body_len,) = struct.unpack("<I", blob[offset:offset + 4])
    offset += 4
    if offset + body_len != len(blob):
        raise OWV3ArchiveError(
            f"decode_owv3_archive: declared body_len={body_len} but "
            f"{len(blob) - offset} byte(s) remain"
        )
    body = blob[offset:offset + body_len]

    model = _build_empty_renderer(header["arch"])
    name_to_module = dict(model.named_modules())
    body_offset = 0
    seen_emb_ids: set[int] = set()

    for layer_meta in header["layers"]:
        name = layer_meta["name"]
        kind = layer_meta["kind"]
        module = name_to_module.get(name)
        if module is None:
            raise OWV3ArchiveError(
                f"decode_owv3_archive: layer {name!r} missing in rebuilt model"
            )

        if kind == "fp16_emb":
            blob_len = int(layer_meta["blob_len"])
            chunk = body[body_offset:body_offset + blob_len]
            body_offset += blob_len
            if id(module) in seen_emb_ids:
                continue
            seen_emb_ids.add(id(module))
            with torch.no_grad():
                module.weight.copy_(_read_fp16_to_tensor(chunk, layer_meta["shape"]))
            continue

        if kind == "asym_emb":
            blob_len = int(layer_meta["blob_len"])
            chunk = body[body_offset:body_offset + blob_len]
            body_offset += blob_len
            if id(module) in seen_emb_ids:
                continue
            seen_emb_ids.add(id(module))
            with torch.no_grad():
                module.weight.copy_(
                    _asym_read_embedding(
                        chunk,
                        layer_meta["shape"],
                        bits=int(layer_meta.get("bits", OWV3_DEFAULT_ASYM_BITS)),
                    )
                )
            continue

        if kind in ("fp16_conv", "fp16_convt", "fp16_linear"):
            w_len = int(layer_meta["weight_blob_len"])
            b_len = int(layer_meta["bias_blob_len"])
            w_chunk = body[body_offset:body_offset + w_len]
            body_offset += w_len
            b_chunk = body[body_offset:body_offset + b_len]
            body_offset += b_len
            w_t = _read_fp16_to_tensor(w_chunk, layer_meta["shape"])
            with torch.no_grad():
                module.weight.copy_(w_t)
                if layer_meta.get("has_bias") and module.bias is not None:
                    bias_len = (
                        w_t.shape[0]
                        if kind != "fp16_convt"
                        else int(layer_meta.get("out_channels", w_t.shape[1]))
                    )
                    module.bias.copy_(_read_fp16_to_tensor(b_chunk, [bias_len]))
            continue

        if kind in ("asym_conv", "asym_convt", "asym_linear"):
            w_len = int(layer_meta["weight_blob_len"])
            b_len = int(layer_meta["bias_blob_len"])
            w_chunk = body[body_offset:body_offset + w_len]
            body_offset += w_len
            b_chunk = body[body_offset:body_offset + b_len]
            body_offset += b_len
            bits = int(layer_meta.get("bits", OWV3_DEFAULT_ASYM_BITS))
            transposed = kind == "asym_convt"
            shape = [int(v) for v in layer_meta["shape"]]
            c_out, _fan_in, _ch_shape = _asym_channel_geometry(
                shape,
                transposed=transposed,
            )
            w_t = _asym_read_channels(
                w_chunk,
                shape,
                list(range(c_out)),
                transposed=transposed,
                bits=bits,
            )
            with torch.no_grad():
                module.weight.copy_(w_t)
                if layer_meta.get("has_bias") and module.bias is not None:
                    module.bias.copy_(_asym_read_bias(b_chunk, c_out=c_out, bits=bits))
            continue

        if kind == "owv3_conv":
            q_len = int(layer_meta["weight_blob_len"])
            p_len = int(layer_meta["protected_blob_len"])
            b_len = int(layer_meta["bias_blob_len"])
            q_chunk = body[body_offset:body_offset + q_len]
            body_offset += q_len
            p_chunk = body[body_offset:body_offset + p_len]
            body_offset += p_len
            b_chunk = body[body_offset:body_offset + b_len]
            body_offset += b_len

            full = torch.zeros(layer_meta["shape"], dtype=torch.float32)
            quant_indices = [int(i) for i in layer_meta["quant_indices"]]
            protected_indices = [int(i) for i in layer_meta["protected_indices"]]
            if quant_indices:
                q_t = decode_omega_w_v2(blob=q_chunk).reshape(
                    [len(quant_indices)] + layer_meta["shape"][1:]
                )
                full[torch.tensor(quant_indices, dtype=torch.long)] = q_t
            if protected_indices:
                protected_codec = layer_meta.get("protected_codec", "fp16")
                if protected_codec == "fp16":
                    p_t = _read_fp16_to_tensor(
                        p_chunk,
                        [len(protected_indices)] + layer_meta["shape"][1:],
                    )
                    full[torch.tensor(protected_indices, dtype=torch.long)] = p_t
                elif protected_codec == "asym":
                    p_full = _asym_read_channels(
                        p_chunk,
                        [int(v) for v in layer_meta["shape"]],
                        protected_indices,
                        bits=int(layer_meta.get("bits", OWV3_DEFAULT_ASYM_BITS)),
                    )
                    idx = torch.tensor(protected_indices, dtype=torch.long)
                    full[idx] = p_full[idx]
                else:
                    raise OWV3ArchiveError(
                        f"decode_owv3_archive: unknown protected_codec {protected_codec!r}"
                    )
            with torch.no_grad():
                module.weight.copy_(full)
                if layer_meta.get("has_bias") and module.bias is not None:
                    module.bias.copy_(_read_fp16_to_tensor(b_chunk, [full.shape[0]]))
            continue

        raise OWV3ArchiveError(
            f"decode_owv3_archive: unknown layer kind {kind!r}"
        )

    scalar_params = header.get("scalar_params", {}) or {}
    if scalar_params:
        param_dict = dict(model.named_parameters())
        with torch.no_grad():
            for pname, pval in scalar_params.items():
                if pname in param_dict:
                    param_dict[pname].fill_(float(pval))

    if body_offset != len(body):
        raise OWV3ArchiveError(
            f"decode_owv3_archive: body had {len(body) - body_offset} trailing byte(s)"
        )

    model = model.to(device)
    model.eval()
    return model


def is_owv3_archive(blob: bytes) -> bool:
    return (
        isinstance(blob, (bytes, bytearray, memoryview))
        and len(blob) >= 4
        and bytes(blob[:4]) == OWV3_ARCHIVE_MAGIC
    )


def inspect_owv3_archive(blob: bytes) -> dict:
    """Return header/body byte provenance for an OWV3 payload."""
    if not is_owv3_archive(blob):
        raise OWV3ArchiveError("inspect_owv3_archive: bad/missing magic")
    data = bytes(blob)
    if len(data) < 12:
        raise OWV3ArchiveError("inspect_owv3_archive: truncated archive")
    offset = 4
    (header_len,) = struct.unpack("<I", data[offset:offset + 4])
    offset += 4
    if header_len <= 0 or offset + header_len + 4 > len(data):
        raise OWV3ArchiveError(
            f"inspect_owv3_archive: invalid header_len={header_len}"
        )
    header = json.loads(data[offset:offset + header_len].decode("utf-8"))
    offset += header_len
    (body_len,) = struct.unpack("<I", data[offset:offset + 4])
    offset += 4
    if offset + body_len != len(data):
        raise OWV3ArchiveError(
            f"inspect_owv3_archive: declared body_len={body_len} but "
            f"{len(data) - offset} byte(s) remain"
        )
    return {
        "magic": OWV3_ARCHIVE_MAGIC.decode("ascii"),
        "version": header.get("version"),
        "total_bytes": len(data),
        "header_len": header_len,
        "body_len": body_len,
        "header": header,
        "byte_plan": header.get("byte_plan", {}),
    }


def enforce_owv3_byte_budget(
    *,
    candidate_bytes: int,
    comparator_bytes: int,
    candidate_label: str = "OWV3 candidate",
    comparator_label: str = "frontier comparator",
    allow_size_regression: bool = False,
    distortion_justification: Mapping[str, object] | None = None,
) -> dict:
    """Fail closed when an OWV3 candidate spends more bytes without evidence."""
    candidate_bytes = int(candidate_bytes)
    comparator_bytes = int(comparator_bytes)
    if candidate_bytes <= 0 or comparator_bytes <= 0:
        raise OWV3ArchiveError("byte budget inputs must be positive")
    delta = candidate_bytes - comparator_bytes
    accepted = delta <= 0 or bool(allow_size_regression) or bool(distortion_justification)
    report = {
        "candidate_label": candidate_label,
        "comparator_label": comparator_label,
        "candidate_bytes": candidate_bytes,
        "comparator_bytes": comparator_bytes,
        "delta_bytes": delta,
        "allow_size_regression": bool(allow_size_regression),
        "has_distortion_justification": bool(distortion_justification),
        "accepted": accepted,
    }
    if not accepted:
        raise OWV3ArchiveError(
            f"{candidate_label} byte plan exceeds {comparator_label}: "
            f"{candidate_bytes} > {comparator_bytes} ({delta:+d} bytes) "
            "without exact distortion justification or explicit smoke/debug override"
        )
    if distortion_justification:
        report["distortion_justification"] = dict(distortion_justification)
    return report


__all__ = [
    "OWV3_ARCHIVE_MAGIC",
    "OWV3_ARCHIVE_VERSION",
    "OWV3ArchiveError",
    "encode_owv3_archive",
    "decode_owv3_archive",
    "enforce_owv3_byte_budget",
    "inspect_owv3_archive",
    "is_owv3_archive",
]
