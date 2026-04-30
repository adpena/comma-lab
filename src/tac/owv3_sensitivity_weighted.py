"""Lane Ω-W-V3 — sensitivity-weighted renderer archive.

Ω-W-V3 is the β fix for the Ω-W-V2 failure mode: uniform renderer-weight
compression saved bytes but over-perturbed PoseNet-sensitive channels. V3
keeps high-sensitivity output channels in FP16 and sends the remaining
channels through the existing OWV2 water-fill + arithmetic codec.

Sensitivity computation is compress-time only. The decode/inflate path reads
only the bytes in this archive and never imports or runs contest scorers.
"""
from __future__ import annotations

import json
import struct
from typing import Mapping

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


def encode_owv3_archive(
    model: nn.Module | None = None,
    *,
    sensitivities: Mapping[str, torch.Tensor] | None = None,
    bit_budget_ratio: float | None = None,
    protect_threshold: float = 1e-3,
    aggressive_threshold: float = 1e-5,
    require_all_conv_sensitivity: bool = False,
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
            blob = _fp16_blob(module.weight)
            body_chunks.append(blob)
            layers_meta.append({
                "name": name,
                "kind": "fp16_emb",
                "shape": list(module.weight.shape),
                "blob_len": len(blob),
            })
            continue

        if isinstance(module, nn.ConvTranspose2d):
            _emit_fp16_conv_layer(
                name=name,
                module=module,
                body_chunks=body_chunks,
                layers_meta=layers_meta,
                kind="fp16_convt",
            )
            continue

        if isinstance(module, nn.Conv2d):
            w = module.weight.detach().cpu().float()
            if not _eligible_for_owv3(module):
                _emit_fp16_conv_layer(
                    name=name,
                    module=module,
                    body_chunks=body_chunks,
                    layers_meta=layers_meta,
                    fallback_reason="ineligible_for_owv3",
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
                _emit_fp16_conv_layer(
                    name=name,
                    module=module,
                    body_chunks=body_chunks,
                    layers_meta=layers_meta,
                    fallback_reason="all_channels_protected",
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
                _emit_fp16_conv_layer(
                    name=name,
                    module=module,
                    body_chunks=body_chunks,
                    layers_meta=layers_meta,
                    fallback_reason=f"owv2_gate:{type(exc).__name__}",
                )
                continue

            protected_blob = b""
            if protected_idx:
                protected_blob = _fp16_blob(
                    w[torch.tensor(protected_idx, dtype=torch.long)]
                )
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
                "bias_blob_len": len(b_blob),
                "bit_budget_ratio": ratio,
                "protect_threshold": protect_threshold,
                "aggressive_threshold": aggressive_threshold,
                "sensitivity_min": float(sens.min().item()),
                "sensitivity_max": float(sens.max().item()),
            })
            continue

        if isinstance(module, nn.Linear):
            w_blob = _fp16_blob(module.weight)
            b_blob = b""
            if module.bias is not None:
                b_blob = _fp16_blob(module.bias)
            body_chunks.append(w_blob + b_blob)
            layers_meta.append({
                "name": name,
                "kind": "fp16_linear",
                "shape": list(module.weight.shape),
                "has_bias": module.bias is not None,
                "weight_blob_len": len(w_blob),
                "bias_blob_len": len(b_blob),
            })
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
                p_t = _read_fp16_to_tensor(
                    p_chunk,
                    [len(protected_indices)] + layer_meta["shape"][1:],
                )
                full[torch.tensor(protected_indices, dtype=torch.long)] = p_t
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


__all__ = [
    "OWV3_ARCHIVE_MAGIC",
    "OWV3_ARCHIVE_VERSION",
    "OWV3ArchiveError",
    "encode_owv3_archive",
    "decode_owv3_archive",
    "is_owv3_archive",
]
