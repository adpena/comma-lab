"""Generic serialization/deserialization for DP-SIMS renderers.

Exports a trained DPSIMSRenderer (~500K params) to a compact binary format
for inclusion in archive.zip. The rate = archive.zip / 37,545,489, so
smaller archive = better score.

Format:
    [4 bytes] header_length (little-endian uint32)
    [header_length bytes] JSON header with architecture config + per-layer metadata
    [per-layer blobs] length-prefixed (4-byte LE uint32) packed weight data

Per-layer packing:
    - Layers with `.bit_depth` (LearnableBitDepth): pack at round(learned_bits)
      per channel using per-channel float16 scale + packed integer values.
    - Layers without `.bit_depth`: pack at `default_bits` (8 = int8).
    - ConvTranspose2d: marked as transposed in header; per-channel dim is 1
      (C_out at index 1 in the (C_in, C_out, kH, kW) PyTorch weight layout).
    - Bias: float16 scale + uint16 value per channel (same as self_compress.py).

Usage::

    from tac.renderer_export import export_renderer_checkpoint, load_renderer_checkpoint

    nbytes = export_renderer_checkpoint(renderer, Path("renderer.bin"))
    restored = load_renderer_checkpoint(Path("renderer.bin"))
    # restored produces identical output to renderer for the same input

Dependencies: torch, struct, json only (no numpy).
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

__all__ = [
    "export_renderer_checkpoint",
    "load_renderer_checkpoint",
    "export_asymmetric_checkpoint",
    "load_asymmetric_checkpoint",
    "export_asymmetric_checkpoint_fp4",
    "load_asymmetric_checkpoint_fp4",
    # Lane F-V5 (hardware FP8 e4m3fn, rescues Lane F regression, 2026-04-28)
    "export_hardware_fp8_checkpoint",
    "load_hardware_fp8_checkpoint",
    # Lane I (Cool-Chic / C3 residual neural-mask renderers, 2026-04-27)
    "export_coolchic_renderer",
    "load_coolchic_renderer",
    "export_c3_residual_renderer",
    "load_c3_residual_renderer",
    # Lane S (Self-Compression renderer, 2026-04-27)
    "export_self_compressed_renderer",
    "load_self_compressed_renderer",
    # Lane J-NWC (Neural Weight Compression renderer.bin, arXiv 2510.11234, 2026-04-28)
    "export_neural_compressed_checkpoint",
    "load_neural_compressed_checkpoint",
    "load_nwcs_sensitivity_compressed_checkpoint",
]


# ── Bit-packing primitives (same logic as self_compress.py) ────────────


def _pack_values(buf: bytearray, values: list[int], bits: int) -> None:
    """Pack a list of unsigned integer values at `bits` per value into buf."""
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


def _unpack_values(data: bytes | bytearray | memoryview, offset: int, count: int, bits: int) -> tuple[list[int], int]:
    """Unpack `count` values at `bits` per value from data starting at offset."""
    if bits == 8:
        values = [data[offset + i] for i in range(count)]
        return values, offset + count
    total_bits = count * bits
    total_bytes = (total_bits + 7) // 8
    if count > 10_000_000:
        raise ValueError(f"Implausible value count={count:,} — possible malformed .bin")
    raw = data[offset:offset + total_bytes]
    # int.from_bytes is O(n) vs O(n²) for the bit-shift loop
    bit_buffer = int.from_bytes(bytes(raw), byteorder="little")
    mask = (1 << bits) - 1
    values = []
    for _ in range(count):
        values.append(bit_buffer & mask)
        bit_buffer >>= bits
    return values, offset + total_bytes


# ── Layer discovery ────────────────────────────────────────────────────


def _collect_conv_layers(renderer: nn.Module) -> list[dict]:
    """Walk the renderer and collect all Conv2d / ConvTranspose2d layers with metadata.

    Returns a list of dicts with keys:
        name: dotted parameter path (e.g. "spade_blocks.0.spade1.shared.0")
        module: the nn.Conv2d or nn.ConvTranspose2d module
        transposed: bool
        bit_depth: LearnableBitDepth or None
    """
    from tac.self_compress import LearnableBitDepth

    layers = []
    module_map = dict(renderer.named_modules())
    for name, module in renderer.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            # Check if the parent has a .bit_depth attribute
            bit_depth = None
            # Walk up to find a parent SelfCompressingConv2d-like wrapper
            # The bit_depth could be a sibling in the parent module
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, child_name = parts
                parent = module_map.get(parent_name)
                if parent is not None and hasattr(parent, "bit_depth"):
                    bd = parent.bit_depth
                    if isinstance(bd, LearnableBitDepth):
                        bit_depth = bd
            # Also check directly on the module
            if bit_depth is None and hasattr(module, "bit_depth"):
                bd = module.bit_depth
                if isinstance(bd, LearnableBitDepth):
                    bit_depth = bd

            layers.append({
                "name": name,
                "module": module,
                "transposed": isinstance(module, nn.ConvTranspose2d),
                "bit_depth": bit_depth,
            })
    return layers


def _infer_renderer_config(renderer: nn.Module) -> dict:
    """Infer architecture config from a DPSIMSRenderer instance.

    Reads instance attributes and module structure to reconstruct the
    constructor arguments needed to rebuild the renderer.
    """
    # These are always stored as instance attributes
    num_classes = getattr(renderer, "num_classes", 5)
    init_h = getattr(renderer, "init_h", 24)
    init_w = getattr(renderer, "init_w", 32)
    use_noise = getattr(renderer, "use_noise", True)

    # Reconstruct channels tuple from spade_blocks output channels
    channels = []
    spade_blocks = getattr(renderer, "spade_blocks", None)
    if spade_blocks is not None:
        for block in spade_blocks:
            # SPADEResBlock.conv2 output channels = block output channels
            channels.append(block.conv2.out_channels)
    else:
        # Fallback: read from const shape
        const = getattr(renderer, "const", None)
        if const is not None:
            channels = [const.shape[1]]

    # Infer spade_hidden from first SPADE block's shared conv output channels
    spade_hidden = 64  # default
    if spade_blocks is not None and len(spade_blocks) > 0:
        shared_conv = spade_blocks[0].spade1.shared[0]  # nn.Conv2d
        spade_hidden = shared_conv.out_channels

    # Infer noise_dim from first noise injector
    noise_dim = 16  # default
    noise_injectors = getattr(renderer, "noise_injectors", None)
    if noise_injectors is not None and len(noise_injectors) > 0:
        noise_dim = noise_injectors[0].noise_dim

    return {
        "num_classes": num_classes,
        "channels": channels,
        "init_h": init_h,
        "init_w": init_w,
        "spade_hidden": spade_hidden,
        "noise_dim": noise_dim,
        "use_noise": use_noise,
    }


# ── Quantization helpers ───────────────────────────────────────────────


def _quantize_tensor_uniform(tensor: torch.Tensor, bits: int) -> tuple[float, list[int]]:
    """Quantize a flat tensor to unsigned integers at given bit-depth.

    Returns (scale, unsigned_values) where:
        scale: float, the abs-max used for normalization
        unsigned_values: list of ints in [0, 2^bits - 1]
    """
    bits = max(bits, 2)  # minimum exportable precision (matches self_compress.py)
    flat = tensor.detach().cpu().reshape(-1).float()
    abs_max = flat.abs().max().clamp(min=6.2e-5).item()  # float16 min normal
    n_levels = 2 ** bits
    half = n_levels // 2
    # Map to signed integer levels
    quantized = (flat / abs_max * (half - 1)).round().clamp(-(half - 1), half - 1).long()
    # Shift to unsigned
    unsigned = (quantized + half).clamp(0, n_levels - 1).tolist()
    return abs_max, unsigned


def _dequantize_values(values: list[int], bits: int, scale: float) -> torch.Tensor:
    """Dequantize unsigned integer values back to float tensor."""
    bits = max(bits, 2)
    n_levels = 2 ** bits
    half = n_levels // 2
    return torch.tensor(
        [(v - half) / max(half - 1, 1) * scale for v in values],
        dtype=torch.float32,
    )


# ── Export ─────────────────────────────────────────────────────────────


def export_renderer_checkpoint(
    renderer: nn.Module,
    output_path: Path,
    default_bits: int = 8,
) -> int:
    """Serialize renderer to compact .bin file.

    Args:
        renderer: DPSIMSRenderer (or compatible nn.Module) to export.
        output_path: path to write the .bin file.
        default_bits: bit-depth for layers without learned bit-depth (default 8).

    Returns:
        Number of bytes written.
    """
    renderer.eval()
    config = _infer_renderer_config(renderer)
    conv_layers = _collect_conv_layers(renderer)

    # Also export the learned constant
    const = getattr(renderer, "const", None)

    # Build layer metadata and weight blobs
    layers_meta: list[dict] = []
    weight_blobs: list[bytes] = []

    # First: export the learned constant as a special layer
    if const is not None:
        const_data = const.detach().cpu().float()
        # Pack constant per-channel to preserve inter-channel dynamic range.
        # Shape is (1, C, H, W) — single global scale would destroy channels
        # at lower magnitude (e.g. 1% of peak gets only ~3 effective bits).
        C = const_data.shape[1]
        packed = bytearray()
        for c in range(C):
            ch_flat = const_data[0, c].reshape(-1)
            scale, unsigned = _quantize_tensor_uniform(ch_flat, default_bits)
            packed.extend(struct.pack("<e", scale))
            _pack_values(packed, unsigned, default_bits)
        weight_blobs.append(bytes(packed))
        layers_meta.append({
            "name": "__const__",
            "shape": list(const_data.shape),
            "bits": default_bits,
            "has_bias": False,
            "transposed": False,
            "is_const": True,
            "per_channel_const": True,
        })

    # Then: export all conv layers
    for layer_info in conv_layers:
        name = layer_info["name"]
        module = layer_info["module"]
        transposed = layer_info["transposed"]
        bit_depth = layer_info["bit_depth"]

        weight = module.weight.detach().cpu().float()
        bias = module.bias.detach().cpu().float() if module.bias is not None else None
        has_bias = bias is not None

        # For Conv2d: weight is (C_out, C_in, kH, kW), per-channel dim = 0
        # For ConvTranspose2d: weight is (C_in, C_out, kH, kW), per-channel dim = 1
        if transposed:
            C_out = weight.shape[1]
            fan_in = weight.shape[0] * weight.shape[2] * weight.shape[3]
        else:
            C_out = weight.shape[0]
            fan_in = weight[0].numel()

        if bit_depth is not None:
            # Per-channel learned bit-depth
            bits_per_channel = bit_depth.bits.detach().cpu()
            active_mask = bits_per_channel >= 0.5
            active_indices = torch.where(active_mask)[0].tolist()
            channel_bits = bits_per_channel[active_mask].round().clamp(1, 8).long().tolist()
            # Promote 1-bit to 2-bit (matches self_compress.py)
            channel_bits = [max(b, 2) for b in channel_bits]

            packed = bytearray()
            for i, ch_idx in enumerate(active_indices):
                ch_bits = channel_bits[i]
                ch_weight = (weight[:, ch_idx] if transposed else weight[ch_idx]).reshape(-1)
                scale, unsigned = _quantize_tensor_uniform(ch_weight, ch_bits)
                packed.extend(struct.pack("<e", scale))
                _pack_values(packed, unsigned, ch_bits)

            # Pack bias for active channels
            bias_packed = bytearray()
            if has_bias:
                for i, ch_idx in enumerate(active_indices):
                    ch_bits = channel_bits[i]
                    b_val = bias[ch_idx].item()
                    abs_max_b = max(abs(b_val), 6.2e-5)
                    n_levels = 2 ** ch_bits
                    half = n_levels // 2
                    q = int(round(b_val / abs_max_b * (half - 1)))
                    q = max(-(half - 1), min(half - 1, q))
                    u = q + half
                    bias_packed.extend(struct.pack("<e", abs_max_b))
                    bias_packed.extend(struct.pack("<H", u))

            weight_blobs.append(bytes(packed))
            weight_blobs.append(bytes(bias_packed))

            layers_meta.append({
                "name": name,
                "shape": list(weight.shape),
                "bits": "per_channel",
                "active_indices": active_indices,
                "channel_bits": channel_bits,
                "has_bias": has_bias,
                "bias_blob_len": len(bias_packed),
                "transposed": transposed,
            })
        else:
            # Uniform bit-depth for all channels
            packed = bytearray()
            for ch_idx in range(C_out):
                ch_weight = (weight[:, ch_idx] if transposed else weight[ch_idx]).reshape(-1)
                scale, unsigned = _quantize_tensor_uniform(ch_weight, default_bits)
                packed.extend(struct.pack("<e", scale))
                _pack_values(packed, unsigned, default_bits)

            bias_packed = bytearray()
            if has_bias:
                for ch_idx in range(C_out):
                    b_val = bias[ch_idx].item()
                    abs_max_b = max(abs(b_val), 6.2e-5)
                    n_levels = 2 ** default_bits
                    half = n_levels // 2
                    q = int(round(b_val / abs_max_b * (half - 1)))
                    q = max(-(half - 1), min(half - 1, q))
                    u = q + half
                    bias_packed.extend(struct.pack("<e", abs_max_b))
                    bias_packed.extend(struct.pack("<H", u))

            weight_blobs.append(bytes(packed))
            weight_blobs.append(bytes(bias_packed))

            layers_meta.append({
                "name": name,
                "shape": list(weight.shape),
                "bits": default_bits,
                "has_bias": has_bias,
                "bias_blob_len": len(bias_packed),
                "transposed": transposed,
            })

    # Build header
    header = {
        "version": 1,
        **config,
        "layers": layers_meta,
    }
    # Collect scalar parameters (e.g. noise gate) not inside Conv2d layers
    scalar_params: dict[str, float] = {}
    for pname, param in renderer.named_parameters():
        if param.numel() == 1 and not any(pname.startswith(cl["name"]) for cl in conv_layers):
            # Skip const (already handled) and conv weights/biases
            if "const" not in pname:
                scalar_params[pname] = param.item()
    header["scalar_params"] = scalar_params

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    # Pack: [magic (4B)] [header_len (4B)] [header JSON] [blob_len (4B)] [blob] ...
    _MAGIC = b"DPSM"  # DP-SIMS renderer binary format
    buf = bytearray()
    buf.extend(_MAGIC)
    buf.extend(struct.pack("<I", len(header_json)))
    buf.extend(header_json)
    for blob in weight_blobs:
        buf.extend(struct.pack("<I", len(blob)))
        buf.extend(blob)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(buf))
    return len(buf)


# ── Load ───────────────────────────────────────────────────────────────


def load_renderer_checkpoint(
    data_or_path: Union[bytes, Path],
    device: str = "cpu",
) -> nn.Module:
    """Deserialize renderer from .bin file.

    Args:
        data_or_path: raw bytes or path to .bin file.
        device: device to place the model on.

    Returns:
        DPSIMSRenderer in eval mode with restored weights.
    """
    from tac.dp_sims_renderer import DPSIMSRenderer

    if isinstance(data_or_path, (str, Path)):
        data = Path(data_or_path).read_bytes()
    else:
        data = data_or_path

    offset = 0

    # Read and verify magic
    _MAGIC = b"DPSM"
    if data[offset:offset + 4] == b"ASYM":
        # Redirect to asymmetric loader
        return load_asymmetric_checkpoint(data_or_path, device=device)
    if data[offset:offset + 4] != _MAGIC:
        raise ValueError(
            f"Not a DPSM renderer binary (expected magic {_MAGIC!r}, "
            f"got {data[offset:offset+4]!r})"
        )
    offset += 4

    # Read header
    header_len = struct.unpack("<I", data[offset:offset + 4])[0]
    offset += 4
    header = json.loads(data[offset:offset + header_len].decode("utf-8"))
    offset += header_len

    version = header.get("version", 0)
    if version != 1:
        raise ValueError(f"Unsupported renderer_export version {version} (expected 1)")

    # Build fresh renderer
    channels = tuple(header["channels"])
    renderer = DPSIMSRenderer(
        num_classes=header.get("num_classes", 5),
        channels=channels,
        init_h=header.get("init_h", 24),
        init_w=header.get("init_w", 32),
        spade_hidden=header.get("spade_hidden", 64),
        noise_dim=header.get("noise_dim", 16),
        use_noise=header.get("use_noise", True),
    )

    # Build name -> module lookup for conv layers
    module_lookup: dict[str, nn.Module] = {}
    for name, module in renderer.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            module_lookup[name] = module

    # Iterate layers in header order and restore weights
    for layer_meta in header["layers"]:
        name = layer_meta["name"]
        is_const = layer_meta.get("is_const", False)

        # Read weight blob
        blob_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        weight_data = data[offset:offset + blob_len]
        offset += blob_len

        if is_const:
            # Restore learned constant — per-channel quantized
            shape = layer_meta["shape"]
            bits = layer_meta["bits"]
            per_channel = layer_meta.get("per_channel_const", False)
            w_offset = 0

            if per_channel and len(shape) >= 2:
                C = shape[1]
                spatial = 1
                for s in shape[2:]:
                    spatial *= s
                channels_list = []
                for _ in range(C):
                    scale = struct.unpack("<e", weight_data[w_offset:w_offset + 2])[0]
                    w_offset += 2
                    values, w_offset = _unpack_values(weight_data, w_offset, spatial, bits)
                    channels_list.append(_dequantize_values(values, bits, scale))
                const_tensor = torch.stack(channels_list).reshape(shape)
            else:
                # Legacy single-scale fallback
                scale = struct.unpack("<e", weight_data[w_offset:w_offset + 2])[0]
                w_offset += 2
                count = 1
                for s in shape:
                    count *= s
                values, w_offset = _unpack_values(weight_data, w_offset, count, bits)
                const_tensor = _dequantize_values(values, bits, scale).reshape(shape)

            with torch.no_grad():
                renderer.const.copy_(const_tensor)
            continue

        # Read bias blob
        has_bias = layer_meta["has_bias"]
        if has_bias:
            bias_blob_len = struct.unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            bias_data = data[offset:offset + bias_blob_len]
            offset += bias_blob_len
        else:
            # Still read the bias blob length (it will be 0)
            bias_blob_len = struct.unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            bias_data = data[offset:offset + bias_blob_len]
            offset += bias_blob_len

        module = module_lookup[name]
        shape = layer_meta["shape"]
        transposed = layer_meta.get("transposed", False)

        # ConvTranspose2d: weight is (C_in, C_out, kH, kW), per-channel dim = 1
        # Conv2d: weight is (C_out, C_in, kH, kW), per-channel dim = 0
        if transposed:
            C_out = shape[1]
            fan_in = shape[0] * shape[2] * shape[3]  # C_in * kH * kW
            ch_shape = [shape[0]] + shape[2:]  # (C_in, kH, kW)
        else:
            C_out = shape[0]
            fan_in = 1
            for s in shape[1:]:
                fan_in *= s
            ch_shape = shape[1:]

        bits_mode = layer_meta["bits"]

        with torch.no_grad():
            module.weight.zero_()
            if module.bias is not None:
                module.bias.zero_()

            if bits_mode == "per_channel":
                # Per-channel variable bit-depth
                active_indices = layer_meta["active_indices"]
                channel_bits = layer_meta["channel_bits"]

                w_offset = 0
                for i, ch_idx in enumerate(active_indices):
                    ch_bits = channel_bits[i]
                    scale = struct.unpack("<e", weight_data[w_offset:w_offset + 2])[0]
                    w_offset += 2
                    values, w_offset = _unpack_values(weight_data, w_offset, fan_in, ch_bits)
                    dequant = _dequantize_values(values, ch_bits, scale)
                    if transposed:
                        module.weight[:, ch_idx] = dequant.reshape(ch_shape)
                    else:
                        module.weight[ch_idx] = dequant.reshape(ch_shape)

                # Restore bias
                if has_bias and bias_data:
                    b_offset = 0
                    for i, ch_idx in enumerate(active_indices):
                        ch_bits = channel_bits[i]
                        scale_b = struct.unpack("<e", bias_data[b_offset:b_offset + 2])[0]
                        b_offset += 2
                        u_val = struct.unpack("<H", bias_data[b_offset:b_offset + 2])[0]
                        b_offset += 2
                        n_levels = 2 ** ch_bits
                        half = n_levels // 2
                        q = u_val - half
                        module.bias[ch_idx] = q / max(half - 1, 1) * scale_b
            else:
                # Uniform bit-depth
                bits = bits_mode
                w_offset = 0
                for ch_idx in range(C_out):
                    scale = struct.unpack("<e", weight_data[w_offset:w_offset + 2])[0]
                    w_offset += 2
                    values, w_offset = _unpack_values(weight_data, w_offset, fan_in, bits)
                    dequant = _dequantize_values(values, bits, scale)
                    if transposed:
                        module.weight[:, ch_idx] = dequant.reshape(ch_shape)
                    else:
                        module.weight[ch_idx] = dequant.reshape(ch_shape)

                # Restore bias
                if has_bias and bias_data:
                    b_offset = 0
                    for ch_idx in range(C_out):
                        scale_b = struct.unpack("<e", bias_data[b_offset:b_offset + 2])[0]
                        b_offset += 2
                        u_val = struct.unpack("<H", bias_data[b_offset:b_offset + 2])[0]
                        b_offset += 2
                        n_levels = 2 ** bits
                        half = n_levels // 2
                        q = u_val - half
                        module.bias[ch_idx] = q / max(half - 1, 1) * scale_b

    # Restore scalar parameters (e.g. noise gate) stored in header
    scalar_params = header.get("scalar_params", {})
    if scalar_params:
        param_dict = dict(renderer.named_parameters())
        with torch.no_grad():
            for pname, pval in scalar_params.items():
                if pname in param_dict:
                    param_dict[pname].fill_(pval)

    # Verify all data consumed (no trailing garbage)
    if offset != len(data):
        raise ValueError(f"Trailing data: {len(data) - offset} bytes unread (expected 0)")

    renderer = renderer.to(device)
    renderer.eval()
    return renderer


# ── Asymmetric PairGenerator export/load ─────────────────────────────


def _infer_asymmetric_config(model: nn.Module) -> dict:
    """Infer architecture config from an AsymmetricPairGenerator (or PairGenerator).

    Returns a dict suitable for splatting into the ASYM/FP4A header. Includes
    `pair_mode` ("asymmetric" for AsymmetricPairGenerator, "pair_generator" for
    legacy PairGenerator) so loaders can dispatch to the right constructor —
    AsymmetricPairGenerator(use_zoom_flow=False) has motion.head shape [6,...]
    while PairGenerator default MotionPredictor has [2,...]. Without the
    pair_mode key the loader cannot tell them apart and silently builds the
    wrong arch (fixed 2026-04-27, Bug 2 in qat_finetune chained arch-drift).
    """
    from tac.renderer import AsymmetricPairGenerator
    renderer = model.renderer
    motion = model.motion

    num_classes = getattr(renderer, "num_classes", 5)
    embed_dim = getattr(renderer, "embed_dim", 6)

    # Infer base_ch: stem_conv may be nn.Sequential (DSConv) or plain Conv2d
    base_ch = 36  # default
    if hasattr(renderer, "stem_conv"):
        stem = renderer.stem_conv
        if hasattr(stem, "out_channels"):
            base_ch = stem.out_channels
        elif isinstance(stem, nn.Sequential):
            # DSConv: Sequential(depthwise, pointwise) — pointwise is [-1]
            base_ch = stem[-1].out_channels

    # Infer mid_ch: down_conv may be nn.Sequential (DSConv) or plain Conv2d
    mid_ch = 60  # default
    if hasattr(renderer, "down_conv"):
        down = renderer.down_conv
        if hasattr(down, "out_channels"):
            mid_ch = down.out_channels
        elif isinstance(down, nn.Sequential):
            mid_ch = down[-1].out_channels

    depth = getattr(renderer, "depth", 1)

    # Motion predictor config — U-Net architecture stores hidden in stem[0]
    motion_hidden = 32  # default
    if hasattr(motion, "stem") and isinstance(motion.stem, nn.Sequential) and len(motion.stem) > 0:
        first_conv = motion.stem[0]
        if isinstance(first_conv, nn.Conv2d):
            motion_hidden = first_conv.out_channels
    elif hasattr(motion, "net") and len(motion.net) > 0:
        # Legacy flat-conv architecture fallback
        first_conv = motion.net[0]
        if isinstance(first_conv, nn.Conv2d):
            motion_hidden = first_conv.out_channels
    output_channels = getattr(motion, "output_channels", 6)
    use_coord_grid = getattr(motion, "use_coord_grid", True)
    use_diff_features = getattr(motion, "use_diff_features", True)

    # Asymmetric-specific motion params (Bug #1: serialize for faithful restore)
    max_flow_px = getattr(motion, "max_flow_px", 20.0)
    max_residual = getattr(motion, "max_residual", 20.0)
    flow_only = getattr(motion, "flow_only", False)

    # FiLM / DSConv architecture flags (required for faithful inline deserialize)
    # 2026-04-26: PairGenerator stores use_dsconv on its `renderer` sub-module
    # (MaskRenderer), not on itself. AsymmetricPairGenerator stores it directly.
    # Fall back through the renderer attr so PairGenerator is handled too —
    # without this, DEN's exported .bin had use_dsconv=False (wrong) and the
    # inflate side built a fatter model whose conv lookup didn't match.
    pose_dim = getattr(model, "pose_dim", getattr(renderer, "pose_dim", 0))
    use_dsconv = getattr(model, "use_dsconv", getattr(renderer, "use_dsconv", False))
    # Lane GH: GhostConv2d arch flag — same fall-through pattern as use_dsconv
    # so PairGenerator (stores it on .renderer) and AsymmetricPairGenerator
    # (stores it on self) both round-trip cleanly.
    use_ghost = getattr(model, "use_ghost", getattr(renderer, "use_ghost", False))
    use_zoom_flow = getattr(model, "use_zoom_flow", False)

    # Conv behavior flags — MUST be serialized to prevent silent corruption
    # (C1 from round 4 review: omitting these caused wrong padding at inflate time)
    padding_mode = getattr(renderer, "padding_mode", "zeros")
    use_dilation = getattr(renderer, "use_dilation", False)

    # pair_mode lets the loader pick the correct constructor.
    # AsymmetricPairGenerator(use_zoom_flow=False) → motion.head [6,...]
    # PairGenerator + default MotionPredictor → motion.head [2,...]
    # Same use_zoom_flow value, different shapes. Dispatch on class identity.
    pair_mode = "asymmetric" if isinstance(model, AsymmetricPairGenerator) else "pair_generator"

    # Codex round-7 fix #3: serialize blend_mode/noise_mode/motion_type so
    # the loader can reconstruct non-default profiles correctly. Profiles
    # set these to "spatial" / "deterministic" / "depth_aware" — without
    # serialization the loader rebuilt with build_renderer's defaults
    # ("scalar" / "deterministic" / "learned_cnn"), causing silent arch
    # drift OR shape-mismatch crash on load_state_dict.
    blend_mode = getattr(model, "blend_mode",
                         getattr(renderer, "blend_mode", "scalar"))
    noise_mode = getattr(model, "noise_mode",
                         getattr(renderer, "noise_mode", "deterministic"))
    motion_type = getattr(model, "motion_type",
                          getattr(motion, "motion_type", "learned_cnn"))

    return {
        "num_classes": num_classes,
        "embed_dim": embed_dim,
        "base_ch": base_ch,
        "mid_ch": mid_ch,
        "depth": depth,
        "motion_hidden": motion_hidden,
        "output_channels": output_channels,
        "use_coord_grid": use_coord_grid,
        "use_diff_features": use_diff_features,
        "max_flow_px": max_flow_px,
        "max_residual": max_residual,
        "flow_only": flow_only,
        "pose_dim": pose_dim,
        "use_dsconv": use_dsconv,
        "use_ghost": use_ghost,
        "use_zoom_flow": use_zoom_flow,
        "padding_mode": padding_mode,
        "use_dilation": use_dilation,
        "pair_mode": pair_mode,
        "blend_mode": blend_mode,
        "noise_mode": noise_mode,
        "motion_type": motion_type,
    }


def _collect_all_conv_layers(model: nn.Module) -> list[dict]:
    """Walk an arbitrary model and collect all Conv2d, ConvTranspose2d, and Linear layers.

    Unlike _collect_conv_layers (which looks for LearnableBitDepth), this is
    a simpler variant used for AsymmetricPairGenerator where all layers use
    uniform quantization.  nn.Linear layers (from FiLM conditioning) are treated
    as Conv2d(in, out, 1) for serialization purposes.
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            layers.append({
                "name": name,
                "module": module,
                "transposed": isinstance(module, nn.ConvTranspose2d),
                "is_linear": False,
                "bit_depth": None,
            })
        elif isinstance(module, nn.Linear):
            layers.append({
                "name": name,
                "module": module,
                "transposed": False,
                "is_linear": True,
                "bit_depth": None,
            })
    return layers


def export_asymmetric_checkpoint(
    model: nn.Module,
    output_path: Path,
    default_bits: int = 8,
) -> int:
    """Serialize an AsymmetricPairGenerator to compact .bin file.

    Format: [ASYM magic (4B)] [header_len (4B)] [JSON header] [blob_len (4B)] [blob] ...

    The header includes pair_mode="asymmetric" and configs for both the
    renderer and motion predictor sub-modules. All nn.Embedding, nn.Conv2d,
    and ConvTranspose2d layers are quantized at default_bits.

    Args:
        model: AsymmetricPairGenerator instance.
        output_path: path to write the .bin file.
        default_bits: bit-depth for weight quantization (default 8).

    Returns:
        Number of bytes written.
    """
    model.eval()
    config = _infer_asymmetric_config(model)

    # Collect all conv layers from the full model (renderer.* and motion.*)
    conv_layers = _collect_all_conv_layers(model)

    # Build layer metadata and weight blobs
    layers_meta: list[dict] = []
    weight_blobs: list[bytes] = []

    # Export all nn.Embedding layers (class embeddings, CLADE gamma/beta)
    # Deduplicate by module id: shared embeddings (renderer + motion share one
    # nn.Embedding) must be exported once, not twice with different quantization noise.
    embedding_layers: list[tuple[str, nn.Embedding]] = []
    seen_emb_ids: set[int] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            if id(module) not in seen_emb_ids:
                embedding_layers.append((name, module))
                seen_emb_ids.add(id(module))

    for emb_name, emb_module in embedding_layers:
        weight = emb_module.weight.detach().cpu().float()
        packed = bytearray()
        # Flatten and quantize the whole embedding table
        flat = weight.reshape(-1)
        scale, unsigned = _quantize_tensor_uniform(flat, default_bits)
        packed.extend(struct.pack("<e", scale))
        _pack_values(packed, unsigned, default_bits)
        weight_blobs.append(bytes(packed))
        layers_meta.append({
            "name": emb_name,
            "shape": list(weight.shape),
            "bits": default_bits,
            "has_bias": False,
            "transposed": False,
            "is_embedding": True,
        })

    # Export all conv/linear layers with uniform bit-depth
    for layer_info in conv_layers:
        name = layer_info["name"]
        module = layer_info["module"]
        transposed = layer_info["transposed"]
        is_linear = layer_info.get("is_linear", False)

        weight = module.weight.detach().cpu().float()
        bias = module.bias.detach().cpu().float() if module.bias is not None else None
        has_bias = bias is not None

        if transposed:
            C_out = weight.shape[1]
        else:
            C_out = weight.shape[0]

        packed = bytearray()
        for ch_idx in range(C_out):
            ch_weight = (weight[:, ch_idx] if transposed else weight[ch_idx]).reshape(-1)
            scale, unsigned = _quantize_tensor_uniform(ch_weight, default_bits)
            packed.extend(struct.pack("<e", scale))
            _pack_values(packed, unsigned, default_bits)

        bias_packed = bytearray()
        if has_bias:
            for ch_idx in range(C_out):
                b_val = bias[ch_idx].item()
                abs_max_b = max(abs(b_val), 6.2e-5)
                n_levels = 2 ** default_bits
                half = n_levels // 2
                q = int(round(b_val / abs_max_b * (half - 1)))
                q = max(-(half - 1), min(half - 1, q))
                u = q + half
                bias_packed.extend(struct.pack("<e", abs_max_b))
                bias_packed.extend(struct.pack("<H", u))

        weight_blobs.append(bytes(packed))
        weight_blobs.append(bytes(bias_packed))

        layers_meta.append({
            "name": name,
            "shape": list(weight.shape),
            "bits": default_bits,
            "has_bias": has_bias,
            "bias_blob_len": len(bias_packed),
            "transposed": transposed,
            "is_linear": is_linear,
        })

    # Build header. pair_mode comes from _infer_asymmetric_config() (it inspects
    # isinstance(model, AsymmetricPairGenerator)). Don't hardcode "asymmetric"
    # here — that was the latent bug: a PairGenerator exported as ASYM would
    # claim pair_mode="asymmetric" but have motion.head shape [2,...] which
    # the loader (which constructs AsymmetricPairGenerator with [6,...]) cannot
    # accept. Trust _infer_asymmetric_config.
    header = {
        "version": 2,
        **config,
        "layers": layers_meta,
    }

    # Collect scalar parameters (e.g. learned blend weights)
    scalar_params: dict[str, float] = {}
    conv_names = {cl["name"] for cl in conv_layers}
    emb_names = {el[0] for el in embedding_layers}
    for pname, param in model.named_parameters():
        if param.numel() == 1:
            # Skip if this is part of a conv or embedding layer
            if not any(pname.startswith(cn) for cn in conv_names | emb_names):
                scalar_params[pname] = param.item()
    header["scalar_params"] = scalar_params

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    # Pack: [magic (4B)] [header_len (4B)] [header JSON] [blob_len (4B)] [blob] ...
    _MAGIC = b"ASYM"
    buf = bytearray()
    buf.extend(_MAGIC)
    buf.extend(struct.pack("<I", len(header_json)))
    buf.extend(header_json)
    for blob in weight_blobs:
        buf.extend(struct.pack("<I", len(blob)))
        buf.extend(blob)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(buf))
    return len(buf)


def load_asymmetric_checkpoint(
    data_or_path: Union[bytes, Path],
    device: str = "cpu",
) -> nn.Module:
    """Deserialize AsymmetricPairGenerator from .bin file.

    Args:
        data_or_path: raw bytes or path to .bin file.
        device: device to place the model on.

    Returns:
        AsymmetricPairGenerator in eval mode with restored weights.
    """
    from tac.renderer import AsymmetricPairGenerator

    if isinstance(data_or_path, (str, Path)):
        data = Path(data_or_path).read_bytes()
    else:
        data = data_or_path

    offset = 0

    # Read and verify magic
    _MAGIC = b"ASYM"
    if data[offset:offset + 4] != _MAGIC:
        raise ValueError(
            f"Not an ASYM renderer binary (expected magic {_MAGIC!r}, "
            f"got {data[offset:offset+4]!r})"
        )
    offset += 4

    # Read header
    header_len = struct.unpack("<I", data[offset:offset + 4])[0]
    offset += 4
    header = json.loads(data[offset:offset + header_len].decode("utf-8"))
    offset += header_len

    version = header.get("version", 0)
    if version != 2:
        raise ValueError(f"Unsupported asymmetric export version {version} (expected 2)")

    # 2026-04-27 arch-drift fix Bug 2 (mirror of FP4A loader fix above):
    # dispatch on pair_mode. Older binaries (pre-fix) DON'T carry pair_mode;
    # default to "asymmetric" since that's what every existing ASYM binary
    # actually was (the exporter hardcoded it). New binaries from the fixed
    # _infer_asymmetric_config() carry the true pair_mode.
    pair_mode = header.get("pair_mode", "asymmetric")
    if pair_mode == "asymmetric":
        # Build fresh AsymmetricPairGenerator (the common case — every
        # AsymmetricPairGenerator-trained renderer including dilated-h64
        # baseline lands here).
        model = AsymmetricPairGenerator(
            num_classes=header.get("num_classes", 5),
            embed_dim=header.get("embed_dim", 6),
            base_ch=header.get("base_ch", 36),
            mid_ch=header.get("mid_ch", 60),
            motion_hidden=header.get("motion_hidden", 32),
            depth=header.get("depth", 1),
            max_flow_px=header.get("max_flow_px", 20.0),
            max_residual=header.get("max_residual", 20.0),
            flow_only=header.get("flow_only", False),
            pose_dim=header.get("pose_dim", 0),
            use_dsconv=header.get("use_dsconv", False),
            use_ghost=header.get("use_ghost", False),
            use_zoom_flow=bool(header.get("use_zoom_flow") or False),
            padding_mode=header.get("padding_mode", "zeros"),
            use_dilation=header.get("use_dilation", False),
        )
    else:
        # Legacy PairGenerator path. build_renderer routes correctly when
        # use_zoom_flow=False is passed (returns PairGenerator wrapping
        # MaskRenderer + MotionPredictor with default output_channels=2).
        from tac.renderer import build_renderer
        # Codex round-7 fix #3: thread blend_mode/noise_mode/motion_type
        # from the binary header. Profiles can set non-defaults (e.g.
        # blend_mode="spatial", motion_type="depth_aware"); without
        # threading them, the loader rebuilt a default-args
        # PairGenerator whose state_dict didn't match the saved model.
        model = build_renderer(
            num_classes=header.get("num_classes", 5),
            embed_dim=header.get("embed_dim", 6),
            base_ch=header.get("base_ch", 36),
            mid_ch=header.get("mid_ch", 60),
            motion_hidden=header.get("motion_hidden", 32),
            depth=header.get("depth", 1),
            pose_dim=header.get("pose_dim", 0),
            use_dsconv=header.get("use_dsconv", False),
            use_ghost=header.get("use_ghost", False),
            padding_mode=header.get("padding_mode", "zeros"),
            use_dilation=header.get("use_dilation", False),
            use_zoom_flow=bool(header.get("use_zoom_flow") or False),
            blend_mode=header.get("blend_mode", "scalar"),
            noise_mode=header.get("noise_mode", "deterministic"),
            motion_type=header.get("motion_type", "learned_cnn"),
        )

    # Build name -> module lookups
    embedding_lookup: dict[str, nn.Embedding] = {}
    conv_lookup: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            embedding_lookup[name] = module
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            conv_lookup[name] = module

    # Iterate layers in header order and restore weights
    for layer_meta in header["layers"]:
        name = layer_meta["name"]
        is_embedding = layer_meta.get("is_embedding", False)

        # Read blob
        blob_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        weight_data = data[offset:offset + blob_len]
        offset += blob_len

        if is_embedding:
            # Restore embedding table
            shape = layer_meta["shape"]
            bits = layer_meta["bits"]
            count = 1
            for s in shape:
                count *= s
            w_offset = 0
            scale = struct.unpack("<e", weight_data[w_offset:w_offset + 2])[0]
            w_offset += 2
            values, w_offset = _unpack_values(weight_data, w_offset, count, bits)
            emb_tensor = _dequantize_values(values, bits, scale).reshape(shape)
            with torch.no_grad():
                embedding_lookup[name].weight.copy_(emb_tensor)
            continue

        # Read bias blob
        has_bias = layer_meta["has_bias"]
        bias_blob_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        bias_data = data[offset:offset + bias_blob_len]
        offset += bias_blob_len

        module = conv_lookup[name]
        shape = layer_meta["shape"]
        transposed = layer_meta.get("transposed", False)
        bits = layer_meta["bits"]

        if transposed:
            C_out = shape[1]
            fan_in = shape[0] * shape[2] * shape[3]
            ch_shape = [shape[0]] + shape[2:]
        else:
            C_out = shape[0]
            fan_in = 1
            for s in shape[1:]:
                fan_in *= s
            ch_shape = shape[1:]

        with torch.no_grad():
            module.weight.zero_()
            if module.bias is not None:
                module.bias.zero_()

            w_offset = 0
            for ch_idx in range(C_out):
                scale = struct.unpack("<e", weight_data[w_offset:w_offset + 2])[0]
                w_offset += 2
                values, w_offset = _unpack_values(weight_data, w_offset, fan_in, bits)
                dequant = _dequantize_values(values, bits, scale)
                if transposed:
                    module.weight[:, ch_idx] = dequant.reshape(ch_shape)
                else:
                    module.weight[ch_idx] = dequant.reshape(ch_shape)

            if has_bias and bias_data:
                b_offset = 0
                for ch_idx in range(C_out):
                    scale_b = struct.unpack("<e", bias_data[b_offset:b_offset + 2])[0]
                    b_offset += 2
                    u_val = struct.unpack("<H", bias_data[b_offset:b_offset + 2])[0]
                    b_offset += 2
                    n_levels = 2 ** bits
                    half = n_levels // 2
                    q = u_val - half
                    module.bias[ch_idx] = q / max(half - 1, 1) * scale_b

    # Restore scalar parameters
    scalar_params = header.get("scalar_params", {})
    if scalar_params:
        param_dict = dict(model.named_parameters())
        with torch.no_grad():
            for pname, pval in scalar_params.items():
                if pname in param_dict:
                    param_dict[pname].fill_(pval)

    # Verify all data consumed
    if offset != len(data):
        raise ValueError(f"Trailing data: {len(data) - offset} bytes unread (expected 0)")

    # Verify shared embedding invariant: export deduplicates by id(), so load
    # relies on renderer.embedding and motion.embedding being the same object.
    if hasattr(model, "renderer") and hasattr(model, "motion"):
        r_emb = getattr(model.renderer, "embedding", None)
        m_emb = getattr(model.motion, "embedding", None)
        if r_emb is not None and m_emb is not None:
            assert r_emb is m_emb, (
                "Shared embedding invariant violated after load — "
                "renderer.embedding and motion.embedding must be the same object"
            )

    model = model.to(device)
    model.eval()
    return model


def export_asymmetric_checkpoint_fp4(
    model: nn.Module,
    output_path: Path,
    block_size: int = 32,
    codebook_name: str = "default",
    robust_scale: bool = False,
) -> int:
    """Serialize an AsymmetricPairGenerator to compact .bin file using FP4 quantization.

    Uses the FP4 codebook scheme (3-bit index + 1-bit sign = 4 bits/weight) with
    per-block scaling (block_size=32). This achieves ~2x compression over int8,
    reducing a 296KB FP8 export to ~148KB.

    Format: [FP4A magic (4B)] [header_len (4B)] [JSON header] [blob_len (4B)] [blob] ...

    Each weight blob is packed as:
        [n_blocks x 2B float16 scale] [packed 4-bit nibbles]

    Biases remain in float16 (negligible size contribution).

    R-fp4-export-fix 2026-04-25: codebook_name and robust_scale args MUST be
    threaded through from the QAT training config (saved in the .pt's __meta__
    dict). Pipeline.step_export reads those flags and passes them here. Without
    this, training with --fp4-codebook=residual would silently ship a default-
    codebook archive — same SHIRAZ-class silent arch drift bug.

    Args:
        model: AsymmetricPairGenerator instance.
        output_path: path to write the .bin file.
        block_size: weights per FP4 scale group (default 32).
        codebook_name: "default" (legacy uniform) or "residual" (denser-near-zero
            for residual heads). MUST match what QAT trained against.
        robust_scale: per-block scale via p99.5 quantile (vs max). MUST match QAT.

    Returns:
        Number of bytes written.
    """
    from tac.fp4_quantize import (
        DEFAULT_CODEBOOK,
        RESIDUAL_CODEBOOK,
        _quantize_block,
        _pack_indices_signs,
    )

    model.eval()
    config = _infer_asymmetric_config(model)

    # Collect all conv layers from the full model (renderer.* and motion.*)
    conv_layers = _collect_all_conv_layers(model)

    # Build layer metadata and weight blobs
    layers_meta: list[dict] = []
    weight_blobs: list[bytes] = []

    if codebook_name == "residual":
        codebook = RESIDUAL_CODEBOOK.clone()
    else:
        codebook = DEFAULT_CODEBOOK.clone()

    # Export all nn.Embedding layers (deduplicated by id)
    embedding_layers: list[tuple[str, nn.Embedding]] = []
    seen_emb_ids: set[int] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            if id(module) not in seen_emb_ids:
                embedding_layers.append((name, module))
                seen_emb_ids.add(id(module))

    for emb_name, emb_module in embedding_layers:
        weight = emb_module.weight.detach().cpu().float().reshape(-1)
        n = weight.shape[0]

        # Pad to block boundary
        pad_len = (block_size - n % block_size) % block_size
        if pad_len > 0:
            weight = torch.cat([weight, torch.zeros(pad_len)])

        # Quantize blocks
        all_packed = []
        all_scales = []
        for start in range(0, weight.shape[0], block_size):
            block = weight[start:start + block_size]
            indices, signs, scale = _quantize_block(block, codebook, robust_scale=robust_scale)
            packed = _pack_indices_signs(indices, signs)
            all_packed.append(packed)
            all_scales.append(float(scale) if isinstance(scale, torch.Tensor) else scale)

        # Build blob: [scales (float16)] [packed nibbles]
        blob = bytearray()
        for s in all_scales:
            blob.extend(struct.pack("<e", s))
        for p in all_packed:
            blob.extend(p.numpy().tobytes())

        weight_blobs.append(bytes(blob))
        layers_meta.append({
            "name": emb_name,
            "shape": list(emb_module.weight.shape),
            "bits": 4,
            "has_bias": False,
            "transposed": False,
            "is_embedding": True,
            "numel": int(emb_module.weight.numel()),
            "block_size": block_size,
        })

    # Export all conv/linear layers with FP4
    for layer_info in conv_layers:
        name = layer_info["name"]
        module = layer_info["module"]
        transposed = layer_info["transposed"]
        is_linear = layer_info.get("is_linear", False)

        weight = module.weight.detach().cpu().float()
        bias = module.bias.detach().cpu().float() if module.bias is not None else None
        has_bias = bias is not None

        # Flatten all weights and quantize with FP4
        flat = weight.reshape(-1)
        n = flat.shape[0]
        pad_len = (block_size - n % block_size) % block_size
        if pad_len > 0:
            flat = torch.cat([flat, torch.zeros(pad_len)])

        all_packed = []
        all_scales = []
        for start in range(0, flat.shape[0], block_size):
            block = flat[start:start + block_size]
            indices, signs, scale = _quantize_block(block, codebook, robust_scale=robust_scale)
            packed = _pack_indices_signs(indices, signs)
            all_packed.append(packed)
            all_scales.append(float(scale) if isinstance(scale, torch.Tensor) else scale)

        # Build weight blob: [scales] [packed data]
        weight_blob = bytearray()
        for s in all_scales:
            weight_blob.extend(struct.pack("<e", s))
        for p in all_packed:
            weight_blob.extend(p.numpy().tobytes())

        # Build bias blob: float16 per channel (no quantization, negligible size)
        bias_blob = bytearray()
        if has_bias:
            for ch_idx in range(bias.shape[0]):
                bias_blob.extend(struct.pack("<e", bias[ch_idx].item()))

        weight_blobs.append(bytes(weight_blob))
        weight_blobs.append(bytes(bias_blob))

        if transposed:
            C_out = weight.shape[1]
        else:
            C_out = weight.shape[0]

        layers_meta.append({
            "name": name,
            "shape": list(weight.shape),
            "bits": 4,
            "has_bias": has_bias,
            "bias_blob_len": len(bias_blob),
            "transposed": transposed,
            "is_linear": is_linear,
            "numel": n,
            "block_size": block_size,
        })

    # Build header. pair_mode comes from _infer_asymmetric_config() (don't
    # hardcode — see export_asymmetric_checkpoint comment for the latent bug).
    header = {
        "version": 3,
        "quantization": "fp4",
        "block_size": block_size,
        "codebook": codebook.tolist(),
        **config,
        "layers": layers_meta,
    }

    # Collect scalar parameters
    scalar_params: dict[str, float] = {}
    conv_names = {cl["name"] for cl in conv_layers}
    emb_names = {el[0] for el in embedding_layers}
    for pname, param in model.named_parameters():
        if param.numel() == 1:
            if not any(pname.startswith(cn) for cn in conv_names | emb_names):
                scalar_params[pname] = param.item()
    header["scalar_params"] = scalar_params

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    # Pack: [magic (4B)] [header_len (4B)] [header JSON] [blob_len (4B)] [blob] ...
    _MAGIC = b"FP4A"
    buf = bytearray()
    buf.extend(_MAGIC)
    buf.extend(struct.pack("<I", len(header_json)))
    buf.extend(header_json)
    for blob in weight_blobs:
        buf.extend(struct.pack("<I", len(blob)))
        buf.extend(blob)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(buf))

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[fp4-export] {param_count:,} params → {len(buf):,} bytes "
          f"({len(buf) / param_count * 8:.2f} bits/param)")
    return len(buf)


def load_asymmetric_checkpoint_fp4(
    data_or_path: Union[bytes, Path],
    device: str = "cpu",
) -> nn.Module:
    """Deserialize AsymmetricPairGenerator from FP4-packed .bin file.

    Args:
        data_or_path: raw bytes or path to .bin file.
        device: device to place the model on.

    Returns:
        AsymmetricPairGenerator in eval mode with restored weights.
    """
    from tac.fp4_quantize import (
        _unpack_indices_signs,
        _dequantize_block,
    )
    from tac.renderer import AsymmetricPairGenerator

    if isinstance(data_or_path, (str, Path)):
        data = Path(data_or_path).read_bytes()
    else:
        data = data_or_path

    offset = 0

    # Read and verify magic
    _MAGIC = b"FP4A"
    if data[offset:offset + 4] != _MAGIC:
        raise ValueError(
            f"Not an FP4A renderer binary (expected magic {_MAGIC!r}, "
            f"got {data[offset:offset+4]!r})"
        )
    offset += 4

    # Read header
    header_len = struct.unpack("<I", data[offset:offset + 4])[0]
    offset += 4
    header = json.loads(data[offset:offset + header_len].decode("utf-8"))
    offset += header_len

    version = header.get("version", 0)
    if version != 3:
        raise ValueError(f"Unsupported FP4 export version {version} (expected 3)")

    block_size = header["block_size"]
    codebook = torch.tensor(header["codebook"], dtype=torch.float32)

    # 2026-04-27 arch-drift fix Bug 2 (memory feedback_qat_finetune_chained_arch_bugs):
    # The dilated-h64 baseline ships in ASYM/FP4A format with pair_mode="asymmetric",
    # use_zoom_flow=False (or absent), and motion.head shape [6, 32, 3, 3] because
    # AsymmetricPairGenerator(use_zoom_flow=False) sets motion_output_channels=6.
    # Routing through build_renderer(use_zoom_flow=False) constructs the LEGACY
    # PairGenerator → MotionPredictor(default output_channels=2) → motion.head
    # shape [2, 32, 3, 3] → shape mismatch when loading the FP4 blob. The header
    # ALREADY carries pair_mode and output_channels via _infer_asymmetric_config,
    # so dispatch on pair_mode: asymmetric → AsymmetricPairGenerator (mirrors the
    # non-FP4 load_asymmetric_checkpoint at line 888); else → build_renderer for
    # the legacy step_export PairGenerator path which DOES want use_zoom_flow=False
    # routing. NEVER default-construct fields that ARE in the header — that's the
    # exact arch-drift trap Bug 1 fixed in load_float_checkpoint.
    pair_mode = header.get("pair_mode", "asymmetric")
    if pair_mode == "asymmetric":
        from tac.renderer import AsymmetricPairGenerator
        model = AsymmetricPairGenerator(
            num_classes=header.get("num_classes", 5),
            embed_dim=header.get("embed_dim", 6),
            base_ch=header.get("base_ch", 36),
            mid_ch=header.get("mid_ch", 60),
            motion_hidden=header.get("motion_hidden", 32),
            depth=header.get("depth", 1),
            max_flow_px=header.get("max_flow_px", 20.0),
            max_residual=header.get("max_residual", 20.0),
            flow_only=header.get("flow_only", False),
            pose_dim=header.get("pose_dim", 0),
            use_dsconv=header.get("use_dsconv", False),
            # use_zoom_flow may be missing (None) in older binaries — coerce to bool.
            # AsymmetricPairGenerator decides motion_output_channels from this:
            # 4 if True else 6. The dilated-h64 baseline has output_channels=6 →
            # use_zoom_flow=False (correct).
            use_zoom_flow=bool(header.get("use_zoom_flow") or False),
            padding_mode=header.get("padding_mode", "zeros"),
            use_dilation=header.get("use_dilation", False),
        )
    else:
        # Legacy PairGenerator path (step_export with use_zoom_flow=False).
        # build_renderer routes to PairGenerator + MotionPredictor here.
        # Note: this path defaults motion.head to output_channels=2 (legacy
        # flow-only). If a future caller exports a PairGenerator with a
        # non-default output_channels, build_renderer will need to thread
        # that through. Today no caller does — every step_export produces
        # pair_mode="asymmetric".
        from tac.renderer import build_renderer
        # Codex round-7 fix #3: thread blend_mode/noise_mode/motion_type
        # from the binary header. Profiles can set non-defaults (e.g.
        # blend_mode="spatial", motion_type="depth_aware"); without
        # threading them, the loader rebuilt a default-args
        # PairGenerator whose state_dict didn't match the saved model.
        model = build_renderer(
            num_classes=header.get("num_classes", 5),
            embed_dim=header.get("embed_dim", 6),
            base_ch=header.get("base_ch", 36),
            mid_ch=header.get("mid_ch", 60),
            motion_hidden=header.get("motion_hidden", 32),
            depth=header.get("depth", 1),
            pose_dim=header.get("pose_dim", 0),
            use_dsconv=header.get("use_dsconv", False),
            use_ghost=header.get("use_ghost", False),
            padding_mode=header.get("padding_mode", "zeros"),
            use_dilation=header.get("use_dilation", False),
            use_zoom_flow=bool(header.get("use_zoom_flow") or False),
            blend_mode=header.get("blend_mode", "scalar"),
            noise_mode=header.get("noise_mode", "deterministic"),
            motion_type=header.get("motion_type", "learned_cnn"),
        )

    # Build name -> module lookups
    embedding_lookup: dict[str, nn.Embedding] = {}
    conv_lookup: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            embedding_lookup[name] = module
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            conv_lookup[name] = module

    def _dequantize_fp4_blob(blob_data: bytes, numel: int, blk_size: int) -> torch.Tensor:
        """Dequantize an FP4 blob back to a flat float tensor."""
        padded_numel = numel + (blk_size - numel % blk_size) % blk_size
        n_blocks = padded_numel // blk_size

        # Read scales (n_blocks x 2 bytes float16)
        scales_bytes = n_blocks * 2
        scales = []
        for i in range(n_blocks):
            s = struct.unpack("<e", blob_data[i * 2:(i + 1) * 2])[0]
            scales.append(s)

        # Read packed nibbles
        packed_offset = scales_bytes
        # Each block: block_size weights -> block_size/2 bytes of packed data
        bytes_per_block = blk_size // 2
        total_packed_bytes = n_blocks * bytes_per_block
        packed_data = torch.tensor(
            list(blob_data[packed_offset:packed_offset + total_packed_bytes]),
            dtype=torch.uint8,
        )

        # Dequantize block by block
        all_values = []
        for i in range(n_blocks):
            block_packed = packed_data[i * bytes_per_block:(i + 1) * bytes_per_block]
            indices, signs = _unpack_indices_signs(block_packed, blk_size)
            scale_t = torch.tensor(scales[i], dtype=torch.float32)
            block_values = _dequantize_block(indices, signs, scale_t, codebook)
            all_values.append(block_values)

        flat = torch.cat(all_values)[:numel]
        return flat

    # Iterate layers in header order and restore weights
    for layer_meta in header["layers"]:
        name = layer_meta["name"]
        is_embedding = layer_meta.get("is_embedding", False)
        numel = layer_meta["numel"]
        blk_size = layer_meta.get("block_size", block_size)

        # Read blob
        blob_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        blob_data = data[offset:offset + blob_len]
        offset += blob_len

        if is_embedding:
            shape = layer_meta["shape"]
            flat = _dequantize_fp4_blob(blob_data, numel, blk_size)
            with torch.no_grad():
                embedding_lookup[name].weight.copy_(flat.reshape(shape))
            continue

        # Read bias blob
        bias_blob_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        bias_data = data[offset:offset + bias_blob_len]
        offset += bias_blob_len

        module = conv_lookup[name]
        shape = layer_meta["shape"]

        # Dequantize weights
        flat = _dequantize_fp4_blob(blob_data, numel, blk_size)
        with torch.no_grad():
            module.weight.copy_(flat.reshape(shape))

            # Restore bias from float16
            if layer_meta["has_bias"] and bias_data:
                transposed = layer_meta.get("transposed", False)
                C_out = shape[1] if transposed else shape[0]
                for ch_idx in range(C_out):
                    b_val = struct.unpack("<e", bias_data[ch_idx * 2:(ch_idx + 1) * 2])[0]
                    module.bias[ch_idx] = b_val

    # Restore scalar parameters
    scalar_params = header.get("scalar_params", {})
    if scalar_params:
        param_dict = dict(model.named_parameters())
        with torch.no_grad():
            for pname, pval in scalar_params.items():
                if pname in param_dict:
                    param_dict[pname].fill_(pval)

    # Verify shared embedding invariant
    if hasattr(model, "renderer") and hasattr(model, "motion"):
        r_emb = getattr(model.renderer, "embedding", None)
        m_emb = getattr(model.motion, "embedding", None)
        if r_emb is not None and m_emb is not None:
            assert r_emb is m_emb, (
                "Shared embedding invariant violated after load"
            )

    model = model.to(device)
    model.eval()
    return model


# ── Lane F-V5: Hardware FP8 (e4m3fn) export / load ────────────────────────
#
# Lane F (FakeQuantFP4 codebook) regressed +0.44 vs baseline because FP4 is
# not hardware-supported on RTX 4090 (CC 8.9 < Blackwell CC 10.0). Lane F-V5
# stores weights at hardware-native FP8 (float8_e4m3fn): 2× the bytes of FP4
# but ~10× faster on tensor cores and ~5–10× lower numerical penalty on the
# YUV6 / FastViT-T12 PoseNet path that wiped Lane F out.
#
# Format:
#   ``[FP8H magic 4B][header_len 4B][JSON header]
#     {[blob_len 4B][scale float32 4B][raw e4m3fn bytes]}*``

_FP8H_MAGIC = b"FP8H"
_FP8H_VERSION = 1


def _e4m3fn_max_bytes() -> float:
    """e4m3fn dynamic range; mirrors ``quantization_fp8._E4M3FN_MAX``."""
    if hasattr(torch, "float8_e4m3fn"):
        try:
            return float(torch.finfo(torch.float8_e4m3fn).max)
        except Exception:
            pass
    return 448.0


def _quant_tensor_e4m3fn(t: torch.Tensor) -> tuple[float, bytes]:
    """Per-tensor quantize to float8_e4m3fn; return ``(scale, raw_bytes)``."""

    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "torch.float8_e4m3fn unavailable; FP8H export requires PyTorch "
            ">= 2.1."
        )
    max_val = _e4m3fn_max_bytes()
    flat = t.detach().cpu().float().reshape(-1).contiguous()
    absmax = float(flat.abs().amax().clamp_min(1e-8))
    scale = absmax / max_val
    if scale < 1e-12:
        scale = 1.0
    scaled = (flat / scale).clamp(-max_val, max_val)
    fp8 = scaled.to(torch.float8_e4m3fn).contiguous()
    raw = fp8.view(torch.uint8).cpu().numpy().tobytes()
    return float(scale), raw


def _dequant_tensor_e4m3fn(scale: float, raw: bytes, numel: int) -> torch.Tensor:
    """Inverse of ``_quant_tensor_e4m3fn``; returns a float32 1-D tensor."""

    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "torch.float8_e4m3fn unavailable; FP8H load requires PyTorch "
            ">= 2.1."
        )
    if len(raw) != numel:
        raise ValueError(
            f"FP8H blob length mismatch: expected {numel} bytes, got {len(raw)}."
        )
    arr = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    fp8 = arr.view(torch.float8_e4m3fn)
    return fp8.to(torch.float32) * scale


def _fp8h_collect_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Walk a model's state_dict and return floating-point tensors in order."""

    out: list[tuple[str, torch.Tensor]] = []
    seen: set[int] = set()
    for name, tensor in model.state_dict().items():
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        if not torch.is_floating_point(tensor):
            continue
        out.append((name, tensor))
    return out


def _fp8h_pack_blob(scale: float, raw: bytes) -> bytes:
    return struct.pack("<f", float(scale)) + raw


def _fp8h_parse_header(data: bytes) -> tuple[dict, int]:
    if data[:4] != _FP8H_MAGIC:
        raise ValueError(
            f"Not an FP8H renderer binary (expected magic {_FP8H_MAGIC!r}, "
            f"got {data[:4]!r})"
        )
    header_len = struct.unpack("<I", data[4:8])[0]
    header = json.loads(data[8:8 + header_len].decode("utf-8"))
    version = header.get("version", 0)
    if version != _FP8H_VERSION:
        raise ValueError(
            f"Unsupported FP8H export version {version} (expected "
            f"{_FP8H_VERSION})"
        )
    return header, 8 + header_len


def _fp8h_iter_blobs(data: bytes, header: dict, offset: int):
    for tensor_meta in header.get("tensors", []):
        blob_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        blob = data[offset:offset + blob_len]
        offset += blob_len
        scale = struct.unpack("<f", blob[:4])[0]
        body = blob[4:]
        yield tensor_meta, scale, body


def _fp8h_build_model_from_header(header: dict, device: str) -> nn.Module:
    """Reconstruct the renderer described by an FP8H header.

    Branches on ``pair_mode`` so AsymmetricPairGenerator and the legacy
    PairGenerator both round-trip cleanly.  ``tensor_only`` stubs (used by
    unit tests on bare nn.Linear / nn.Conv2d) return a placeholder
    ``nn.Module()``; callers compare state_dicts directly via
    ``model._fp8h_state_dict``.
    """

    config = header.get("config", {})
    if config.get("tensor_only"):
        return nn.Module()

    pair_mode = config.get("pair_mode", "asymmetric")
    if pair_mode == "asymmetric":
        from tac.renderer import AsymmetricPairGenerator
        return AsymmetricPairGenerator(
            num_classes=config.get("num_classes", 5),
            embed_dim=config.get("embed_dim", 6),
            base_ch=config.get("base_ch", 36),
            mid_ch=config.get("mid_ch", 60),
            motion_hidden=config.get("motion_hidden", 32),
            depth=config.get("depth", 1),
            max_flow_px=config.get("max_flow_px", 20.0),
            max_residual=config.get("max_residual", 20.0),
            flow_only=config.get("flow_only", False),
            pose_dim=config.get("pose_dim", 0),
            use_dsconv=config.get("use_dsconv", False),
            use_zoom_flow=bool(config.get("use_zoom_flow") or False),
            padding_mode=config.get("padding_mode", "zeros"),
            use_dilation=config.get("use_dilation", False),
        )
    from tac.renderer import build_renderer
    return build_renderer(
        num_classes=config.get("num_classes", 5),
        embed_dim=config.get("embed_dim", 6),
        base_ch=config.get("base_ch", 36),
        mid_ch=config.get("mid_ch", 60),
        motion_hidden=config.get("motion_hidden", 32),
        depth=config.get("depth", 1),
        pose_dim=config.get("pose_dim", 0),
        use_dsconv=config.get("use_dsconv", False),
        use_ghost=config.get("use_ghost", False),
        padding_mode=config.get("padding_mode", "zeros"),
        use_dilation=config.get("use_dilation", False),
        use_zoom_flow=bool(config.get("use_zoom_flow") or False),
        blend_mode=config.get("blend_mode", "scalar"),
        noise_mode=config.get("noise_mode", "deterministic"),
        motion_type=config.get("motion_type", "learned_cnn"),
    )


def export_hardware_fp8_checkpoint(
    model: nn.Module,
    output_path: Path,
    *,
    config_overrides: dict | None = None,
) -> int:
    """Serialize a renderer to a hardware FP8 (e4m3fn) ``.bin`` file.

    Stores arch config in a JSON header (same shape as FP4A) plus a per-tensor
    ``[scale (float32 4B)][raw e4m3fn bytes]`` blob. The state_dict round-trip
    is precision-bound by e4m3fn (atol≈5e-2, rtol≈2.5e-1 for unit-test
    tolerance).

    Returns the number of bytes written (matches ``output_path.stat().st_size``).
    """

    model.eval()
    try:
        config = _infer_asymmetric_config(model)
    except Exception:
        config = {"tensor_only": True}
    if config_overrides:
        config = {**config, **config_overrides}

    tensors = _fp8h_collect_tensors(model)
    blob_meta: list[dict] = []
    blobs: list[bytes] = []
    for name, tensor in tensors:
        scale, raw = _quant_tensor_e4m3fn(tensor)
        blob_meta.append({
            "name": name,
            "shape": list(tensor.shape),
            "numel": int(tensor.numel()),
            "dtype_in": str(tensor.dtype).replace("torch.", ""),
        })
        blobs.append(_fp8h_pack_blob(scale, raw))

    header = {
        "magic": "FP8H",
        "version": _FP8H_VERSION,
        "format": "hardware_fp8",
        "dtype": "float8_e4m3fn",
        "min_capability": [8, 9],
        "config": config,
        "tensors": blob_meta,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    buf = bytearray()
    buf.extend(_FP8H_MAGIC)
    buf.extend(struct.pack("<I", len(header_json)))
    buf.extend(header_json)
    for raw in blobs:
        buf.extend(struct.pack("<I", len(raw)))
        buf.extend(raw)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(buf))

    param_count = sum(p.numel() for p in model.parameters())
    if param_count > 0:
        print(
            f"[fp8h-export] {param_count:,} params → {len(buf):,} bytes "
            f"({len(buf) / param_count * 8:.2f} bits/param)"
        )
    return len(buf)


def load_hardware_fp8_checkpoint(
    data_or_path: Union[bytes, Path],
    device: str = "cpu",
) -> nn.Module:
    """Restore a renderer from an FP8H ``.bin`` (Lane F-V5 hardware FP8)."""

    if isinstance(data_or_path, (str, Path)):
        data = Path(data_or_path).read_bytes()
    else:
        data = data_or_path

    header, offset = _fp8h_parse_header(data)
    model = _fp8h_build_model_from_header(header, device=device)

    new_state: dict[str, torch.Tensor] = {}
    for tensor_meta, scale, body in _fp8h_iter_blobs(data, header, offset):
        flat = _dequant_tensor_e4m3fn(scale, body, tensor_meta["numel"])
        shape = tensor_meta["shape"]
        new_state[tensor_meta["name"]] = flat.reshape(shape)

    if header.get("config", {}).get("tensor_only"):
        model._fp8h_state_dict = new_state  # type: ignore[attr-defined]
        return model

    full_state = dict(model.state_dict())
    for k, v in new_state.items():
        if k in full_state:
            full_state[k] = v.to(full_state[k].dtype)
        else:
            full_state[k] = v
    model.load_state_dict(full_state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def detect_checkpoint_type(data_or_path: Union[bytes, Path]) -> str:
    """Detect the type of a renderer checkpoint.

    Returns:
        "dpsm" for DPSIMSRenderer, "asymmetric" for AsymmetricPairGenerator,
        "asymmetric_fp4" for FP4-quantized AsymmetricPairGenerator,
        "hardware_fp8" for FP8H (Lane F-V5 hardware FP8 e4m3fn),
        "int4_lzma2" for int4 per-tensor + LZMA2 compressed,
        "coolchic" for CoolChic PairGenerator (CCh1 magic),
        "c3_residual" for C3 residual PairGenerator (C3R1 magic),
        "self_compress_v1" for SC per-channel learnable bit-depth (SCv1 magic),
        "omega_v1" for Lane Ω per-weight Hessian-aware bit-depth (OMG1 magic),
        "neural_weight_compression_v1" for Lane J-NWC neural weight codec (NWC1 magic),
        "neural_weight_compression_sensitivity_v1" for Lane J-NWCS (NWCS1 magic),
        "pytorch" for raw PyTorch checkpoints.
    """
    if isinstance(data_or_path, (str, Path)):
        data = Path(data_or_path).read_bytes()[:8]
    else:
        data = data_or_path[:8]

    if data[:4] == b"DPSM":
        return "dpsm"
    elif data[:4] == b"FP4A":
        return "asymmetric_fp4"
    elif data[:4] == b"FP8H":
        return "hardware_fp8"
    elif data[:4] == b"ASYM":
        return "asymmetric"
    elif data[:4] == b"I4LZ":
        return "int4_lzma2"
    elif data[:4] == b"SCv1":
        return "self_compress_v1"
    elif data[:4] == b"OMG1":
        return "omega_v1"
    elif data[:8] == b"NWCS1\0\0\0":
        return "neural_weight_compression_sensitivity_v1"
    elif data[:4] == b"NWC1":
        return "neural_weight_compression_v1"
    elif data[:4] == _COOLCHIC_MAGIC:
        return "coolchic"
    elif data[:4] == _C3_RESIDUAL_MAGIC:
        return "c3_residual"
    else:
        return "pytorch"


def load_any_renderer_checkpoint(
    data_or_path: Union[bytes, Path],
    device: str = "cpu",
) -> nn.Module:
    """Load any renderer checkpoint, auto-detecting the format.

    Supports: DPSM (.bin), ASYM (.bin), FP4A (.bin), INT4_LZMA2 (.bin),
    and raw PyTorch (.pt) checkpoints.

    Returns:
        The loaded model in eval mode on the specified device.
    """
    fmt = detect_checkpoint_type(data_or_path)
    if fmt == "dpsm":
        return load_renderer_checkpoint(data_or_path, device=device)
    elif fmt == "asymmetric_fp4":
        return load_asymmetric_checkpoint_fp4(data_or_path, device=device)
    elif fmt == "asymmetric":
        return load_asymmetric_checkpoint(data_or_path, device=device)
    elif fmt == "coolchic":
        return load_coolchic_renderer(data_or_path, device=device)
    elif fmt == "c3_residual":
        return load_c3_residual_renderer(data_or_path, device=device)
    elif fmt == "self_compress_v1":
        return load_self_compressed_renderer(data_or_path, device=device)
    elif fmt == "omega_v1":
        return load_omega_renderer(data_or_path, device=device)
    elif fmt == "neural_weight_compression_v1":
        return load_neural_compressed_checkpoint(data_or_path, device=device)
    elif fmt == "neural_weight_compression_sensitivity_v1":
        return load_nwcs_sensitivity_compressed_checkpoint(data_or_path, device=device)
    elif fmt == "int4_lzma2":
        from tac.mixed_precision_export import load_int4_lzma2
        from tac.renderer import AsymmetricPairGenerator

        # Load state dict and infer architecture from it
        state_dict = load_int4_lzma2(data_or_path, device=device)

        # Infer architecture params from state dict keys/shapes
        # The embedding weight shape tells us embed_dim and num_classes
        emb_key = next((k for k in state_dict if "embedding.weight" in k), None)
        if emb_key is not None:
            num_classes, embed_dim = state_dict[emb_key].shape
        else:
            num_classes, embed_dim = 5, 6

        # Infer architecture from state dict to prevent silent mismatches
        def _get_ch(prefix, default):
            for suffix in [f"{prefix}.weight", f"{prefix}.1.weight"]:
                if suffix in state_dict:
                    return state_dict[suffix].shape[0]
            return default
        base_ch = _get_ch("renderer.stem_conv", 36)
        mid_ch = _get_ch("renderer.down_conv", 60)
        use_dsconv = "renderer.stem_conv.0.weight" in state_dict
        pose_dim = 6 if any("film_bottleneck" in k for k in state_dict) else 0

        model = AsymmetricPairGenerator(
            num_classes=num_classes,
            embed_dim=embed_dim,
            base_ch=base_ch,
            mid_ch=mid_ch,
            use_dsconv=use_dsconv,
            pose_dim=pose_dim,
        )
        model.load_state_dict(state_dict, strict=True)
        return model.eval().to(device)
    else:
        raise ValueError(
            "Raw PyTorch checkpoint detected — use _load_renderer() in "
            "inflate_renderer.py for .pt format support."
        )


# ── Lane I: Cool-Chic / C3 residual exports ─────────────────────────────
#
# 2026-04-27: Cool-Chic (`CoolChicLatentRenderer`) and C3 residual
# (`C3ResidualRenderer`) wrap a small synthesis decoder + multi-resolution
# learned latent grids. The latent grids are `nn.ParameterList` entries (one
# per resolution), NOT `nn.Conv2d` weights, so the existing FP4A/ASYM
# layer-walk (`_collect_all_conv_layers`) does not pick them up. We add a
# parallel format that:
#
#   * Uses the same FP4 block-quantization primitives (block_size=32, default
#     codebook) as `export_asymmetric_checkpoint_fp4` for everything that
#     IS a conv/linear/embedding layer.
#   * Adds explicit `latent_tensors` blob entries that quantize each
#     `latents.<i>` parameter as a single flat tensor with its shape
#     recorded in the JSON header.
#   * Supports per-tensor MIXED PRECISION via `residual_quant_bits` (Phase
#     3 of the Lane I work). The hypothesis from
#     `reports/local_trend_coolchic_c3_20260425.md` is that FP4 destroys
#     C3's residual-net float-path SegNet gain because the residual head is
#     zero-init and learns small corrections that quantize to 0 under FP4's
#     per-block scaling. Allowing residual_net + class_embed (the
#     small-magnitude tail of the network) to ship at int8 per-channel —
#     same scheme as `export_asymmetric_checkpoint` — preserves the gain at
#     a tiny rate cost (~2 bits/param × ~3K params ≈ 0.75KB).
#
# Magic bytes: `CCh1` for `CoolChicLatentRenderer`-only PairGenerator,
# `C3R1` for `C3ResidualRenderer` PairGenerator. The downstream inflate
# pipeline produces `(B, 2, H, W, 3)` HWC pairs from
# `pair_gen(mask_t, mask_t1)` exactly as for AsymmetricPairGenerator —
# `_load_renderer_and_masks` does not need to special-case the variant.

_COOLCHIC_MAGIC = b"CCh1"
_C3_RESIDUAL_MAGIC = b"C3R1"
_COOLCHIC_FORMAT_VERSION = 1
_C3_RESIDUAL_FORMAT_VERSION = 1


def _is_coolchic_renderer(model: nn.Module) -> bool:
    """True if `model.renderer` is a CoolChicLatentRenderer (no residual head)."""
    from tac.contrib.coolchic_renderer import CoolChicLatentRenderer, C3ResidualRenderer
    inner = getattr(model, "renderer", None)
    return isinstance(inner, CoolChicLatentRenderer) and not isinstance(inner, C3ResidualRenderer)


def _is_c3_residual_renderer(model: nn.Module) -> bool:
    """True if `model.renderer` is a C3ResidualRenderer."""
    from tac.contrib.coolchic_renderer import C3ResidualRenderer
    return isinstance(getattr(model, "renderer", None), C3ResidualRenderer)


def _quantize_block_fp4(
    flat: torch.Tensor,
    codebook: torch.Tensor,
    block_size: int,
    robust_scale: bool = False,
) -> tuple[bytes, int]:
    """FP4 block-quantize a flat tensor; return (blob, original_numel).

    Blob layout matches `export_asymmetric_checkpoint_fp4`:
        [n_blocks × float16 scale] [packed nibbles, 2 weights/byte]
    Caller is responsible for storing `numel` and `block_size` in the header
    so the inverse operation can recover the flat tensor.
    """
    from tac.fp4_quantize import _quantize_block, _pack_indices_signs

    numel = int(flat.numel())
    flat = flat.detach().cpu().float().reshape(-1)
    pad_len = (block_size - numel % block_size) % block_size
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len)])

    all_packed: list[torch.Tensor] = []
    all_scales: list[float] = []
    for start in range(0, flat.shape[0], block_size):
        block = flat[start:start + block_size]
        indices, signs, scale = _quantize_block(block, codebook, robust_scale=robust_scale)
        packed = _pack_indices_signs(indices, signs)
        all_packed.append(packed)
        all_scales.append(float(scale) if isinstance(scale, torch.Tensor) else scale)

    blob = bytearray()
    for s in all_scales:
        blob.extend(struct.pack("<e", s))
    for p in all_packed:
        blob.extend(p.numpy().tobytes())
    return bytes(blob), numel


def _dequantize_block_fp4(
    blob: bytes,
    numel: int,
    block_size: int,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Inverse of `_quantize_block_fp4`; returns a flat float32 tensor of `numel`."""
    from tac.fp4_quantize import _unpack_indices_signs, _dequantize_block

    padded_numel = numel + (block_size - numel % block_size) % block_size
    n_blocks = padded_numel // block_size

    scales_bytes = n_blocks * 2
    scales = []
    for i in range(n_blocks):
        s = struct.unpack("<e", blob[i * 2:(i + 1) * 2])[0]
        scales.append(s)

    bytes_per_block = block_size // 2
    total_packed = n_blocks * bytes_per_block
    packed_raw = bytes(blob[scales_bytes:scales_bytes + total_packed])

    all_values = []
    for i in range(n_blocks):
        block_packed = torch.tensor(
            list(packed_raw[i * bytes_per_block:(i + 1) * bytes_per_block]),
            dtype=torch.uint8,
        )
        indices, signs = _unpack_indices_signs(block_packed, block_size)
        scale_t = torch.tensor(scales[i], dtype=torch.float32)
        block_values = _dequantize_block(indices, signs, scale_t, codebook)
        all_values.append(block_values)

    return torch.cat(all_values)[:numel]


def _quantize_int_per_channel(
    weight: torch.Tensor,
    bits: int,
    transposed: bool,
) -> bytes:
    """Per-channel uniform int quantization (matches `export_asymmetric_checkpoint`).

    Blob layout per channel:
        [float16 scale] [packed values at `bits` per value]
    """
    if transposed:
        C_out = weight.shape[1]
    else:
        C_out = weight.shape[0]
    packed = bytearray()
    for ch_idx in range(C_out):
        ch_weight = (weight[:, ch_idx] if transposed else weight[ch_idx]).reshape(-1)
        scale, unsigned = _quantize_tensor_uniform(ch_weight, bits)
        packed.extend(struct.pack("<e", scale))
        _pack_values(packed, unsigned, bits)
    return bytes(packed)


def _dequantize_int_per_channel(
    blob: bytes,
    shape: list[int],
    bits: int,
    transposed: bool,
) -> torch.Tensor:
    """Inverse of `_quantize_int_per_channel`."""
    if transposed:
        C_out = shape[1]
        ch_shape = [shape[0]] + list(shape[2:])
    else:
        C_out = shape[0]
        ch_shape = list(shape[1:])
    fan_in = 1
    for s in ch_shape:
        fan_in *= s

    out = torch.zeros(shape, dtype=torch.float32)
    w_offset = 0
    for ch_idx in range(C_out):
        scale = struct.unpack("<e", blob[w_offset:w_offset + 2])[0]
        w_offset += 2
        values, w_offset = _unpack_values(blob, w_offset, fan_in, bits)
        dequant = _dequantize_values(values, bits, scale)
        if transposed:
            out[:, ch_idx] = dequant.reshape(ch_shape)
        else:
            out[ch_idx] = dequant.reshape(ch_shape)
    return out


def _infer_coolchic_config(model: nn.Module) -> dict:
    """Infer constructor args for a CoolChic PairGenerator (CoolChicLatentRenderer + MotionPredictor)."""
    from tac.contrib.coolchic_renderer import CoolChicLatentRenderer

    renderer = model.renderer
    motion = model.motion
    assert isinstance(renderer, CoolChicLatentRenderer), \
        f"_infer_coolchic_config expects CoolChicLatentRenderer, got {type(renderer).__name__}"

    # Motion-predictor hidden width (mirrors _infer_asymmetric_config)
    motion_hidden = 32
    if hasattr(motion, "stem") and isinstance(motion.stem, nn.Sequential) and len(motion.stem) > 0:
        first = motion.stem[0]
        if isinstance(first, nn.Conv2d):
            motion_hidden = first.out_channels

    return {
        "num_classes": int(renderer.num_classes),
        "embed_dim": int(renderer.class_embed_dim),
        "latent_ch": int(renderer.latent_ch),
        "hidden": int(renderer.hidden),
        "motion_hidden": int(motion_hidden),
        "latent_shapes": [list(s) for s in renderer.latent_shapes],
        "blend_mode": getattr(model, "blend_mode", "scalar"),
        "noise_mode": getattr(model, "noise_mode", "deterministic"),
    }


def _infer_c3_residual_config(model: nn.Module) -> dict:
    """Infer constructor args for a C3 residual PairGenerator."""
    from tac.contrib.coolchic_renderer import C3ResidualRenderer

    renderer = model.renderer
    motion = model.motion
    assert isinstance(renderer, C3ResidualRenderer), \
        f"_infer_c3_residual_config expects C3ResidualRenderer, got {type(renderer).__name__}"
    base = renderer.base_renderer

    motion_hidden = 32
    if hasattr(motion, "stem") and isinstance(motion.stem, nn.Sequential) and len(motion.stem) > 0:
        first = motion.stem[0]
        if isinstance(first, nn.Conv2d):
            motion_hidden = first.out_channels

    return {
        "num_classes": int(renderer.num_classes),
        # `embed_dim` here refers to the BASE renderer's class_embed_dim, which
        # build_c3_residual_renderer uses as the canonical embed_dim arg.
        "embed_dim": int(base.class_embed_dim),
        "latent_ch": int(base.latent_ch),
        "hidden": int(base.hidden),
        "motion_hidden": int(motion_hidden),
        "residual_hidden": int(renderer.residual_hidden),
        "residual_layers": int(renderer.residual_layers),
        "residual_scale": float(renderer.residual_scale),
        "num_bands": int(renderer.num_bands),
        "latent_shapes": [list(s) for s in base.latent_shapes],
        "blend_mode": getattr(model, "blend_mode", "scalar"),
        "noise_mode": getattr(model, "noise_mode", "deterministic"),
    }


# Names (under `model.renderer.`) that should ship at higher precision when
# `residual_quant_bits` is non-None. These are the C3 residual-net params and
# its dedicated class embedding — the float→FP4 collapse hypothesized in
# `reports/local_trend_coolchic_c3_20260425.md`.
def _is_c3_residual_param_name(full_name: str) -> bool:
    """True for parameters that belong to the C3 residual head (NOT the base CoolChic)."""
    return (
        full_name.startswith("renderer.residual_net.")
        or full_name == "renderer.class_embed.weight"
    )


def _export_coolchic_or_c3(
    model: nn.Module,
    output_path: Path,
    *,
    is_c3: bool,
    block_size: int = 32,
    codebook_name: str = "default",
    robust_scale: bool = False,
    residual_quant_bits: int | None = None,
) -> int:
    """Shared serializer for CoolChic / C3 residual PairGenerators.

    Args:
        model: PairGenerator wrapping CoolChicLatentRenderer (CCh1) or
            C3ResidualRenderer (C3R1).
        output_path: destination .bin path.
        block_size: FP4 block size (default 32, matches FP4A).
        codebook_name: "default" or "residual" (FP4 codebook selection).
        robust_scale: per-block p99.5 scale instead of max.
        residual_quant_bits: if not None, all params matching
            `_is_c3_residual_param_name(...)` are stored at INT-`residual_quant_bits`
            per-channel uniform quantization instead of FP4. Only meaningful
            for C3R1 (no-op for CCh1 since the predicate matches nothing in
            CoolChic state). Common values: 8 (preserves float-path gain) or
            None (legacy, all FP4 — proven to destroy float-path gain).
    """
    from tac.fp4_quantize import DEFAULT_CODEBOOK, RESIDUAL_CODEBOOK

    model.eval()
    if is_c3:
        config = _infer_c3_residual_config(model)
        magic = _C3_RESIDUAL_MAGIC
        version = _C3_RESIDUAL_FORMAT_VERSION
    else:
        config = _infer_coolchic_config(model)
        magic = _COOLCHIC_MAGIC
        version = _COOLCHIC_FORMAT_VERSION
        # Mixed-precision residual quant is C3-only — silently noop on CCh.
        residual_quant_bits = None

    codebook = (RESIDUAL_CODEBOOK if codebook_name == "residual" else DEFAULT_CODEBOOK).clone()

    # Walk the model: split params into three buckets by routing rule.
    #   1. Latent grids (nn.Parameter inside `latents` ParameterList): FP4
    #      block-quantized as flat tensors with explicit shape preservation.
    #   2. Embeddings (nn.Embedding): FP4 block-quantized (or int8/int{N}
    #      per-channel if residual_quant_bits set AND name matches).
    #   3. Conv/Linear weights+biases: FP4 block (default) OR int per-channel
    #      (when residual_quant_bits set AND name matches).
    layers_meta: list[dict] = []
    weight_blobs: list[bytes] = []

    # Find the latent ParameterList path. CoolChic: model.renderer.latents.{i}.
    # C3: model.renderer.base_renderer.latents.{i}.
    if is_c3:
        latent_owner = model.renderer.base_renderer
        latent_owner_path = "renderer.base_renderer"
    else:
        latent_owner = model.renderer
        latent_owner_path = "renderer"

    # Ordered iteration of the ParameterList preserves the resolution order
    # the constructor will reproduce.
    for i, latent in enumerate(latent_owner.latents):
        full_name = f"{latent_owner_path}.latents.{i}"
        blob, numel = _quantize_block_fp4(
            latent.detach().cpu(), codebook, block_size, robust_scale=robust_scale
        )
        weight_blobs.append(blob)
        layers_meta.append({
            "name": full_name,
            "kind": "latent",
            "shape": list(latent.shape),
            "numel": numel,
            "block_size": block_size,
            "bits": 4,
        })

    # Embeddings — dedupe by id() (mirrors FP4A pattern).
    seen_emb_ids: set[int] = set()
    embedding_layers: list[tuple[str, nn.Embedding]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            if id(module) not in seen_emb_ids:
                embedding_layers.append((name, module))
                seen_emb_ids.add(id(module))

    for emb_name, emb_module in embedding_layers:
        full_name = f"{emb_name}.weight"
        weight = emb_module.weight.detach().cpu().float()
        # Mixed-precision check (C3-only): name lives at "renderer.class_embed.weight"
        # → ships at int{bits} when residual_quant_bits is set.
        if residual_quant_bits is not None and _is_c3_residual_param_name(full_name):
            # 2D embedding tables: per-channel along dim 0 (the vocab dim).
            blob = _quantize_int_per_channel(weight, residual_quant_bits, transposed=False)
            weight_blobs.append(blob)
            layers_meta.append({
                "name": emb_name,
                "kind": "embedding_int",
                "shape": list(weight.shape),
                "bits": int(residual_quant_bits),
                "transposed": False,
            })
        else:
            flat = weight.reshape(-1)
            blob, numel = _quantize_block_fp4(
                flat, codebook, block_size, robust_scale=robust_scale
            )
            weight_blobs.append(blob)
            layers_meta.append({
                "name": emb_name,
                "kind": "embedding_fp4",
                "shape": list(weight.shape),
                "numel": numel,
                "block_size": block_size,
                "bits": 4,
            })

    # Conv2d / ConvTranspose2d / Linear — same walk as FP4A.
    conv_layers = _collect_all_conv_layers(model)
    for layer_info in conv_layers:
        name = layer_info["name"]
        module = layer_info["module"]
        transposed = layer_info["transposed"]
        is_linear = layer_info.get("is_linear", False)

        weight = module.weight.detach().cpu().float()
        bias = module.bias.detach().cpu().float() if module.bias is not None else None
        has_bias = bias is not None

        # Mixed-precision: residual_net.* layers ship at int{bits} per-channel.
        weight_full_name = f"{name}.weight"
        use_int = (residual_quant_bits is not None
                   and _is_c3_residual_param_name(weight_full_name))

        if use_int:
            weight_blob = _quantize_int_per_channel(weight, residual_quant_bits, transposed)
            # Bias: same int{bits} per-channel (matches export_asymmetric_checkpoint).
            bias_blob = bytearray()
            if has_bias:
                C_out = weight.shape[1] if transposed else weight.shape[0]
                n_levels = 2 ** residual_quant_bits
                half = n_levels // 2
                for ch_idx in range(C_out):
                    b_val = bias[ch_idx].item()
                    abs_max_b = max(abs(b_val), 6.2e-5)
                    q = int(round(b_val / abs_max_b * (half - 1)))
                    q = max(-(half - 1), min(half - 1, q))
                    u = q + half
                    bias_blob.extend(struct.pack("<e", abs_max_b))
                    bias_blob.extend(struct.pack("<H", u))
            weight_blobs.append(weight_blob)
            weight_blobs.append(bytes(bias_blob))
            layers_meta.append({
                "name": name,
                "kind": "conv_int",
                "shape": list(weight.shape),
                "bits": int(residual_quant_bits),
                "transposed": transposed,
                "is_linear": is_linear,
                "has_bias": has_bias,
                "bias_blob_len": len(bias_blob),
            })
        else:
            flat = weight.reshape(-1)
            weight_blob, numel = _quantize_block_fp4(
                flat, codebook, block_size, robust_scale=robust_scale
            )
            # Bias: float16 per channel (matches FP4A).
            bias_blob = bytearray()
            if has_bias:
                for ch_idx in range(bias.shape[0]):
                    bias_blob.extend(struct.pack("<e", bias[ch_idx].item()))
            weight_blobs.append(weight_blob)
            weight_blobs.append(bytes(bias_blob))
            layers_meta.append({
                "name": name,
                "kind": "conv_fp4",
                "shape": list(weight.shape),
                "numel": numel,
                "block_size": block_size,
                "bits": 4,
                "transposed": transposed,
                "is_linear": is_linear,
                "has_bias": has_bias,
                "bias_blob_len": len(bias_blob),
            })

    # Scalar params (e.g. PairGenerator.blend_logit) — same convention as FP4A.
    scalar_params: dict[str, float] = {}
    layer_names = {meta["name"] for meta in layers_meta}
    for pname, param in model.named_parameters():
        if param.numel() == 1:
            # Skip params already covered by a layer (e.g. someone added a
            # 1-elem weight inside a Conv that we walked above).
            owns = False
            for ln in layer_names:
                if pname.startswith(ln):
                    owns = True
                    break
            if not owns:
                scalar_params[pname] = float(param.item())

    header = {
        "version": version,
        "format": "coolchic" if not is_c3 else "c3_residual",
        "quantization": "fp4_mixed" if residual_quant_bits else "fp4",
        "block_size": block_size,
        "codebook": codebook.tolist(),
        "codebook_name": codebook_name,
        "robust_scale": robust_scale,
        "residual_quant_bits": residual_quant_bits,
        **config,
        "layers": layers_meta,
        "scalar_params": scalar_params,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    buf = bytearray()
    buf.extend(magic)
    buf.extend(struct.pack("<I", len(header_json)))
    buf.extend(header_json)
    for blob in weight_blobs:
        buf.extend(struct.pack("<I", len(blob)))
        buf.extend(blob)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(buf))

    n_params = sum(p.numel() for p in model.parameters())
    fmt_label = ("c3_residual" if is_c3 else "coolchic")
    mp_label = (f" residual_int{residual_quant_bits}" if residual_quant_bits else "")
    print(
        f"[{fmt_label}-export]{mp_label} {n_params:,} params → {len(buf):,} bytes "
        f"({len(buf) / max(n_params, 1) * 8:.2f} bits/param)"
    )
    return len(buf)


def export_coolchic_renderer(
    model: nn.Module,
    output_path: Path,
    *,
    block_size: int = 32,
    codebook_name: str = "default",
    robust_scale: bool = False,
) -> int:
    """Serialize a Cool-Chic PairGenerator (CoolChicLatentRenderer + MotionPredictor) to .bin.

    Format magic: `b"CCh1"`. See `_export_coolchic_or_c3` for the on-disk
    layout. The latent ParameterList is captured as explicit `latent` blobs
    (not pulled in by the conv-layer walk).
    """
    if not _is_coolchic_renderer(model):
        raise TypeError(
            f"export_coolchic_renderer expects a PairGenerator wrapping "
            f"CoolChicLatentRenderer (and NOT C3ResidualRenderer). "
            f"Got renderer={type(getattr(model, 'renderer', None)).__name__}"
        )
    return _export_coolchic_or_c3(
        model, output_path,
        is_c3=False,
        block_size=block_size,
        codebook_name=codebook_name,
        robust_scale=robust_scale,
        residual_quant_bits=None,
    )


def export_c3_residual_renderer(
    model: nn.Module,
    output_path: Path,
    *,
    block_size: int = 32,
    codebook_name: str = "default",
    robust_scale: bool = False,
    residual_quant_bits: int | None = None,
) -> int:
    """Serialize a C3 residual PairGenerator (C3ResidualRenderer + MotionPredictor) to .bin.

    Format magic: `b"C3R1"`. When `residual_quant_bits` is set (typical: 8),
    the residual head's conv weights/biases and dedicated class embedding
    ship at int{N} per-channel uniform quantization while the underlying
    Cool-Chic base + motion predictor stay in FP4. This is Phase 3 of the
    Lane I work — it is the SCIENCE blocker the trend report identified
    (FP4 destroys C3's residual-net float-path SegNet gain).

    Args mirror `export_asymmetric_checkpoint_fp4` plus `residual_quant_bits`.
    """
    if not _is_c3_residual_renderer(model):
        raise TypeError(
            f"export_c3_residual_renderer expects a PairGenerator wrapping "
            f"C3ResidualRenderer. Got renderer={type(getattr(model, 'renderer', None)).__name__}"
        )
    return _export_coolchic_or_c3(
        model, output_path,
        is_c3=True,
        block_size=block_size,
        codebook_name=codebook_name,
        robust_scale=robust_scale,
        residual_quant_bits=residual_quant_bits,
    )


def _load_coolchic_or_c3(
    data_or_path: Union[bytes, Path],
    *,
    is_c3: bool,
    device: str = "cpu",
) -> nn.Module:
    """Shared deserializer for CCh1 / C3R1. Returns a PairGenerator in eval mode."""
    if isinstance(data_or_path, (str, Path)):
        data = Path(data_or_path).read_bytes()
    else:
        data = bytes(data_or_path)

    expected_magic = _C3_RESIDUAL_MAGIC if is_c3 else _COOLCHIC_MAGIC
    expected_version = _C3_RESIDUAL_FORMAT_VERSION if is_c3 else _COOLCHIC_FORMAT_VERSION
    fmt_label = "c3_residual" if is_c3 else "coolchic"

    offset = 0
    if data[offset:offset + 4] != expected_magic:
        raise ValueError(
            f"Not a {fmt_label} ({expected_magic!r}) binary; "
            f"got magic {data[offset:offset+4]!r}"
        )
    offset += 4

    header_len = struct.unpack("<I", data[offset:offset + 4])[0]
    offset += 4
    header = json.loads(data[offset:offset + header_len].decode("utf-8"))
    offset += header_len

    version = header.get("version", 0)
    if version != expected_version:
        raise ValueError(
            f"Unsupported {fmt_label} export version {version} "
            f"(expected {expected_version})"
        )

    block_size = int(header["block_size"])
    codebook = torch.tensor(header["codebook"], dtype=torch.float32)

    # Construct the PairGenerator from the recorded config. This must mirror
    # `build_coolchic_renderer` / `build_c3_residual_renderer` exactly.
    if is_c3:
        from tac.contrib.coolchic_renderer import build_c3_residual_renderer
        latent_shapes = tuple(tuple(s) for s in header["latent_shapes"])
        model = build_c3_residual_renderer(
            num_classes=int(header.get("num_classes", 5)),
            embed_dim=int(header.get("embed_dim", 6)),
            latent_ch=int(header.get("latent_ch", 6)),
            hidden=int(header.get("hidden", 24)),
            motion_hidden=int(header.get("motion_hidden", 32)),
            residual_hidden=int(header.get("residual_hidden", 32)),
            residual_layers=int(header.get("residual_layers", 2)),
            residual_scale=float(header.get("residual_scale", 16.0)),
            latent_shapes=latent_shapes,
            blend_mode=str(header.get("blend_mode", "scalar")),
            noise_mode=str(header.get("noise_mode", "deterministic")),
        )
    else:
        from tac.contrib.coolchic_renderer import build_coolchic_renderer
        latent_shapes = tuple(tuple(s) for s in header["latent_shapes"])
        model = build_coolchic_renderer(
            num_classes=int(header.get("num_classes", 5)),
            embed_dim=int(header.get("embed_dim", 6)),
            latent_ch=int(header.get("latent_ch", 8)),
            hidden=int(header.get("hidden", 32)),
            motion_hidden=int(header.get("motion_hidden", 32)),
            latent_shapes=latent_shapes,
            blend_mode=str(header.get("blend_mode", "scalar")),
            noise_mode=str(header.get("noise_mode", "deterministic")),
        )

    # Build name → module/parameter lookups.
    embedding_lookup: dict[str, nn.Embedding] = {}
    conv_lookup: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            embedding_lookup[name] = module
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            conv_lookup[name] = module

    # Iterate layers in header order; restore weights.
    for layer_meta in header["layers"]:
        name = layer_meta["name"]
        kind = layer_meta["kind"]

        blob_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        blob = data[offset:offset + blob_len]
        offset += blob_len

        if kind == "latent":
            shape = layer_meta["shape"]
            numel = layer_meta["numel"]
            blk = layer_meta.get("block_size", block_size)
            flat = _dequantize_block_fp4(blob, numel, blk, codebook)
            tensor = flat.reshape(shape)
            # Walk attribute path: the latent owner is `renderer` (CCh) or
            # `renderer.base_renderer` (C3); the suffix is `latents.<i>`.
            # Use direct attribute walk so we update the actual nn.Parameter
            # storage rather than registering a new one.
            obj = model
            parts = name.split(".")
            for p in parts[:-2]:
                obj = getattr(obj, p)
            # obj is now the renderer; obj.latents is ParameterList.
            idx = int(parts[-1])
            with torch.no_grad():
                obj.latents[idx].data.copy_(tensor)
            continue

        if kind == "embedding_fp4":
            shape = layer_meta["shape"]
            numel = layer_meta["numel"]
            blk = layer_meta.get("block_size", block_size)
            flat = _dequantize_block_fp4(blob, numel, blk, codebook)
            with torch.no_grad():
                embedding_lookup[name].weight.copy_(flat.reshape(shape))
            continue

        if kind == "embedding_int":
            shape = layer_meta["shape"]
            bits = int(layer_meta["bits"])
            transposed = bool(layer_meta.get("transposed", False))
            tensor = _dequantize_int_per_channel(blob, shape, bits, transposed)
            with torch.no_grad():
                embedding_lookup[name].weight.copy_(tensor)
            continue

        # Conv-like layers: read bias blob next.
        bias_blob_len = struct.unpack("<I", data[offset:offset + 4])[0]
        offset += 4
        bias_blob = data[offset:offset + bias_blob_len]
        offset += bias_blob_len

        module = conv_lookup[name]
        shape = layer_meta["shape"]
        transposed = bool(layer_meta.get("transposed", False))
        has_bias = bool(layer_meta.get("has_bias", False))

        if kind == "conv_fp4":
            numel = layer_meta["numel"]
            blk = layer_meta.get("block_size", block_size)
            flat = _dequantize_block_fp4(blob, numel, blk, codebook)
            with torch.no_grad():
                module.weight.copy_(flat.reshape(shape))
                if has_bias and bias_blob:
                    C_out = shape[1] if transposed else shape[0]
                    for ch_idx in range(C_out):
                        b_val = struct.unpack("<e", bias_blob[ch_idx * 2:(ch_idx + 1) * 2])[0]
                        module.bias[ch_idx] = b_val
            continue

        if kind == "conv_int":
            bits = int(layer_meta["bits"])
            tensor = _dequantize_int_per_channel(blob, shape, bits, transposed)
            with torch.no_grad():
                module.weight.copy_(tensor)
                if has_bias and bias_blob:
                    C_out = shape[1] if transposed else shape[0]
                    n_levels = 2 ** bits
                    half = n_levels // 2
                    b_offset = 0
                    for ch_idx in range(C_out):
                        scale_b = struct.unpack("<e", bias_blob[b_offset:b_offset + 2])[0]
                        b_offset += 2
                        u_val = struct.unpack("<H", bias_blob[b_offset:b_offset + 2])[0]
                        b_offset += 2
                        q = u_val - half
                        module.bias[ch_idx] = q / max(half - 1, 1) * scale_b
            continue

        raise ValueError(f"Unknown layer kind {kind!r} in {fmt_label} binary at offset {offset}")

    # Restore scalar params (e.g. blend_logit).
    scalar_params = header.get("scalar_params", {})
    if scalar_params:
        param_dict = dict(model.named_parameters())
        with torch.no_grad():
            for pname, pval in scalar_params.items():
                if pname in param_dict:
                    param_dict[pname].fill_(pval)

    # Defense: every byte must be consumed (mirrors FP4A loader contract).
    if offset != len(data):
        raise ValueError(
            f"Trailing data: {len(data) - offset} bytes unread (expected 0) in {fmt_label} binary"
        )

    model = model.to(device)
    model.eval()
    return model


def load_coolchic_renderer(
    data_or_path: Union[bytes, Path],
    device: str = "cpu",
) -> nn.Module:
    """Deserialize a Cool-Chic PairGenerator from a `b"CCh1"` .bin."""
    return _load_coolchic_or_c3(data_or_path, is_c3=False, device=device)


def load_c3_residual_renderer(
    data_or_path: Union[bytes, Path],
    device: str = "cpu",
) -> nn.Module:
    """Deserialize a C3 residual PairGenerator from a `b"C3R1"` .bin."""
    return _load_coolchic_or_c3(data_or_path, is_c3=True, device=device)


# ── Smoke test ─────────────────────────────────────────────────────────


def _smoke_test() -> None:
    """Build a small renderer, round-trip export/load, verify output match."""
    import tempfile

    from tac.dp_sims_renderer import DPSIMSRenderer

    print("renderer_export: running smoke tests...")

    # Small renderer for fast testing
    channels = (32, 16)
    renderer = DPSIMSRenderer(
        num_classes=5,
        channels=channels,
        init_h=6,
        init_w=8,
        spade_hidden=32,
        noise_dim=8,
        use_noise=False,  # deterministic for testing
    )
    # Randomize weights so quantization error is exercised
    with torch.no_grad():
        for p in renderer.parameters():
            p.normal_(0, 0.1)
    renderer.eval()

    print(f"  param count: {renderer.param_count():,}")

    # Generate test input
    B = 2
    H, W = 6 * (2 ** len(channels)), 8 * (2 ** len(channels))  # match upsample stages
    masks = torch.randint(0, 5, (B, H, W))

    # Forward pass
    with torch.no_grad():
        out_orig = renderer(masks)
    print(f"  original output shape: {out_orig.shape}, range: [{out_orig.min():.1f}, {out_orig.max():.1f}]")

    # Export
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp_path = Path(f.name)
    nbytes = export_renderer_checkpoint(renderer, tmp_path, default_bits=8)
    print(f"  exported: {nbytes:,} bytes to {tmp_path}")

    # Load
    restored = load_renderer_checkpoint(tmp_path)
    print(f"  loaded: param count {restored.param_count():,}")

    # Verify output
    with torch.no_grad():
        out_restored = restored(masks)

    max_diff = (out_orig - out_restored).abs().max().item()
    mean_diff = (out_orig - out_restored).abs().mean().item()
    print(f"  round-trip max diff: {max_diff:.4f}, mean diff: {mean_diff:.4f}")

    # Quantization introduces error, but it should be bounded
    # At 8 bits with 256 levels, max error per weight ~ scale/127
    # Through the network this compounds but should stay reasonable
    assert max_diff < 20.0, f"Round-trip error too large: {max_diff}"
    print("  round-trip accuracy: PASS")

    # Test with lower bit-depth
    nbytes_4bit = export_renderer_checkpoint(renderer, tmp_path, default_bits=4)
    restored_4bit = load_renderer_checkpoint(tmp_path)
    with torch.no_grad():
        out_4bit = restored_4bit(masks)
    max_diff_4bit = (out_orig - out_4bit).abs().max().item()
    print(f"  4-bit export: {nbytes_4bit:,} bytes, max diff: {max_diff_4bit:.4f}")
    assert nbytes_4bit < nbytes, "4-bit should be smaller than 8-bit"
    print("  4-bit smaller than 8-bit: PASS")

    # Test loading from bytes
    raw_bytes = tmp_path.read_bytes()
    restored_from_bytes = load_renderer_checkpoint(raw_bytes)
    with torch.no_grad():
        out_from_bytes = restored_from_bytes(masks)
    assert (out_4bit - out_from_bytes).abs().max().item() == 0.0, "Load from bytes must match load from path"
    print("  load from bytes: PASS")

    # Clean up
    tmp_path.unlink()

    # --- Test 2: Full-size renderer with noise injectors ---
    print("\n  --- full-size renderer with noise (use_noise=True) ---")
    channels_full = (64, 32, 16)
    renderer_full = DPSIMSRenderer(
        num_classes=5,
        channels=channels_full,
        init_h=6,
        init_w=8,
        spade_hidden=32,
        noise_dim=8,
        use_noise=True,
    )
    with torch.no_grad():
        for p in renderer_full.parameters():
            p.normal_(0, 0.05)
    renderer_full.eval()

    H_full = 6 * (2 ** len(channels_full))
    W_full = 8 * (2 ** len(channels_full))
    masks_full = torch.randint(0, 5, (1, H_full, W_full))
    # Deterministic noise = zeros for reproducible output
    with torch.no_grad():
        out_full_orig = renderer_full(masks_full, noise=torch.zeros(1, 8, 6, 8))

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp_path = Path(f.name)
    nbytes_full = export_renderer_checkpoint(renderer_full, tmp_path, default_bits=8)
    restored_full = load_renderer_checkpoint(tmp_path)
    with torch.no_grad():
        out_full_restored = restored_full(masks_full, noise=torch.zeros(1, 8, 6, 8))

    max_diff_full = (out_full_orig - out_full_restored).abs().max().item()
    print(f"  full renderer: {renderer_full.param_count():,} params, {nbytes_full:,} bytes, max diff: {max_diff_full:.4f}")
    assert max_diff_full < 20.0, f"Full renderer round-trip error too large: {max_diff_full}"
    print("  full renderer round-trip: PASS")

    # Verify config was correctly inferred and restored
    config_orig = _infer_renderer_config(renderer_full)
    config_restored = _infer_renderer_config(restored_full)
    for key in ["num_classes", "channels", "init_h", "init_w", "spade_hidden", "noise_dim", "use_noise"]:
        assert config_orig[key] == config_restored[key], f"Config mismatch on {key}: {config_orig[key]} vs {config_restored[key]}"
    print("  config round-trip: PASS")

    tmp_path.unlink()

    # --- Test 3: AsymmetricPairGenerator round-trip ---
    print("\n  --- asymmetric pair generator round-trip ---")
    from tac.renderer import AsymmetricPairGenerator

    asym = AsymmetricPairGenerator(
        num_classes=5, embed_dim=4, base_ch=8, mid_ch=16,
        motion_hidden=8, depth=1, max_flow_px=15.0, max_residual=25.0,
    )
    with torch.no_grad():
        for p in asym.parameters():
            p.normal_(0, 0.05)
    asym.eval()

    m1 = torch.randint(0, 5, (2, 24, 32))
    m2 = torch.randint(0, 5, (2, 24, 32))
    with torch.no_grad():
        out_asym_orig = asym(m1, m2)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp_path = Path(f.name)
    nbytes_asym = export_asymmetric_checkpoint(asym, tmp_path, default_bits=8)
    restored_asym = load_asymmetric_checkpoint(tmp_path)
    with torch.no_grad():
        out_asym_restored = restored_asym(m1, m2)

    max_diff_asym = (out_asym_orig - out_asym_restored).abs().max().item()
    print(f"  asymmetric: {asym.param_count():,} params, {nbytes_asym:,} bytes, max diff: {max_diff_asym:.4f}")
    assert max_diff_asym < 20.0, f"Asymmetric round-trip error: {max_diff_asym}"
    print("  asymmetric round-trip: PASS")

    # Verify config round-trip includes max_flow_px, max_residual, flow_only
    config_asym = _infer_asymmetric_config(asym)
    assert config_asym["max_flow_px"] == 15.0, f"max_flow_px mismatch: {config_asym['max_flow_px']}"
    assert config_asym["max_residual"] == 25.0, f"max_residual mismatch: {config_asym['max_residual']}"
    assert config_asym["flow_only"] is False, f"flow_only mismatch: {config_asym['flow_only']}"
    assert restored_asym.motion.max_flow_px == 15.0, "Restored max_flow_px mismatch"
    assert restored_asym.motion.max_residual == 25.0, "Restored max_residual mismatch"
    print("  asymmetric config round-trip (max_flow_px, max_residual, flow_only): PASS")

    tmp_path.unlink()

    print("\nrenderer_export: all smoke tests passed")


# ── Lane S: Self-Compression export / load (SCv1 magic) ────────────────
#
# Format design (2026-04-27, Lane S):
#
#   Magic (4B): b"SCv1"
#   Header length (4B little-endian uint32)
#   Header JSON (UTF-8)
#   Body length (4B little-endian uint32)
#   Body bytes (LZMA-compressed concatenation of layer blobs)
#
# Header schema:
#   {
#     "version": 1,
#     "arch": { ... AsymmetricPairGenerator constructor kwargs ... },
#     "layers": [
#       {
#         "name": "renderer.stem_conv",
#         "kind": "self_compress" | "fp16_conv" | "fp16_convt" | "fp16_linear" | "fp16_emb",
#         "shape": [c_out, c_in_per_group, kH, kW],
#         "bias": bool,
#         # SC-only:
#         "bits_per_channel": [int, ...],   # length C_out, each in [0, 8]
#         "scale_per_channel": [float16, ...] (stored as raw bytes in body),
#         "bias_scale_per_channel": [float16, ...] (stored in body),
#       },
#       ...
#     ],
#     "scalar_params": {name: float, ...}  # nn.Parameter(scalar) restoration
#   }
#
# Body layout (concat then LZMA-compress):
#   For each layer (in header order):
#     SC layer:
#       [scales_w  : C_out × float16]
#       [packed_weights : sum_over_active_channels(ceil(fan_in × bits / 8)) bytes]
#       [scales_b  : C_out × float16]              (only if has_bias)
#       [packed_biases : sum_over_active_channels(ceil(bits / 8)) bytes]   (only if has_bias)
#     FP16 layer (conv / convt / linear / embedding):
#       [weights : numel × float16]
#       [bias    : C_out × float16]                (only if has_bias)
#
# Channels with bits == 0 are PRUNED — no per-channel scale or packed data
# stored. The header's bits_per_channel encodes the prune mask. At load
# time pruned channels are reconstructed as zeros.
#
# Why LZMA: the per-channel scale field has high entropy (float16) but the
# packed weight bytes have moderate entropy (often biased toward 0 because
# the bit_depth tensor concentrates near 2-3 bits → many low-magnitude
# nibbles). LZMA gets ~10-15% on top of the raw bit-packing; tested on
# self-compressing postfilters in src/tac/self_compress.py.


def _sc_pack_per_channel_weights(
    weight: torch.Tensor,  # (C_out, C_in_per_group, kH, kW) float32
    bits_per_channel: list[int],
) -> tuple[bytes, list[float]]:
    """Pack per-channel quantized weights at variable bit-depth.

    Returns (packed_bytes, scales) where scales is len-C_out list of float scales
    (one per channel; pruned channels get scale=0.0). The packed bytes
    concatenate all active channels' bit-packed weights.
    """
    flat_weight = weight.reshape(weight.shape[0], -1).cpu().float()
    fan_in = flat_weight.shape[1]
    scales: list[float] = []
    packed = bytearray()

    for ch_idx, bits in enumerate(bits_per_channel):
        if bits <= 0:
            scales.append(0.0)
            continue
        # 1-bit channels are promoted to 2-bit in the export (training STE
        # supports 1-bit at half_levels=0.5 clamped, but 1-bit packing has
        # the same divide-by-zero edge case; use 2-bit as the floor — same
        # rule as self_compress.export_compressed_checkpoint).
        bits_export = max(int(bits), 2)
        ch_w = flat_weight[ch_idx]
        abs_max = float(ch_w.abs().max().clamp(min=1e-10).item())
        scales.append(abs_max)
        n_levels = 2 ** bits_export
        half = n_levels // 2
        # Quantize to signed [-(half-1), half-1], shift to unsigned [0, n_levels-1]
        quantized = (ch_w / abs_max * (half - 1)).round().clamp(-(half - 1), half - 1).long()
        unsigned = (quantized + half).clamp(0, n_levels - 1).tolist()
        # Bit-pack into the body
        bit_buffer = 0
        bits_in_buffer = 0
        for v in unsigned:
            bit_buffer |= (v & ((1 << bits_export) - 1)) << bits_in_buffer
            bits_in_buffer += bits_export
            while bits_in_buffer >= 8:
                packed.append(bit_buffer & 0xFF)
                bit_buffer >>= 8
                bits_in_buffer -= 8
        if bits_in_buffer > 0:
            packed.append(bit_buffer & 0xFF)
    return bytes(packed), scales


def _sc_unpack_per_channel_weights(
    packed: bytes,
    bits_per_channel: list[int],
    fan_in: int,
    scales: list[float],
) -> torch.Tensor:
    """Reverse of _sc_pack_per_channel_weights.

    Returns a (C_out, fan_in) float32 tensor (caller reshapes to weight shape).
    """
    n_out = len(bits_per_channel)
    out = torch.zeros((n_out, fan_in), dtype=torch.float32)
    offset_bytes = 0
    for ch_idx, bits in enumerate(bits_per_channel):
        if bits <= 0:
            continue
        bits_export = max(int(bits), 2)
        n_levels = 2 ** bits_export
        half = n_levels // 2
        total_bits = fan_in * bits_export
        total_bytes = (total_bits + 7) // 8
        raw = packed[offset_bytes:offset_bytes + total_bytes]
        offset_bytes += total_bytes

        bit_buffer = 0
        for i, b in enumerate(raw):
            bit_buffer |= b << (i * 8)
        mask = (1 << bits_export) - 1
        values: list[int] = []
        for _ in range(fan_in):
            values.append(bit_buffer & mask)
            bit_buffer >>= bits_export
        scale = scales[ch_idx]
        # Dequantize
        ch_t = torch.tensor(
            [(v - half) / max(half - 1, 1) * scale for v in values],
            dtype=torch.float32,
        )
        out[ch_idx] = ch_t
    return out


def _sc_pack_per_channel_biases(
    bias: torch.Tensor,  # (C_out,)
    bits_per_channel: list[int],
) -> tuple[bytes, list[float]]:
    """Pack per-channel biases at the same bit-depths as weights."""
    bias_cpu = bias.detach().cpu().float()
    scales: list[float] = []
    packed = bytearray()
    for ch_idx, bits in enumerate(bits_per_channel):
        if bits <= 0:
            scales.append(0.0)
            continue
        bits_export = max(int(bits), 2)
        b_val = float(bias_cpu[ch_idx].item())
        abs_max = max(abs(b_val), 1e-10)
        scales.append(abs_max)
        n_levels = 2 ** bits_export
        half = n_levels // 2
        q = int(round(b_val / abs_max * (half - 1)))
        q = max(-(half - 1), min(half - 1, q))
        u = q + half
        # One bias value per channel; pack as 1-2 bytes depending on bits.
        if bits_export <= 8:
            packed.append(u & 0xFF)
        else:
            packed.append(u & 0xFF)
            packed.append((u >> 8) & 0xFF)
    return bytes(packed), scales


def _sc_unpack_per_channel_biases(
    packed: bytes,
    bits_per_channel: list[int],
    scales: list[float],
) -> torch.Tensor:
    """Reverse of _sc_pack_per_channel_biases."""
    n_out = len(bits_per_channel)
    out = torch.zeros((n_out,), dtype=torch.float32)
    offset = 0
    for ch_idx, bits in enumerate(bits_per_channel):
        if bits <= 0:
            continue
        bits_export = max(int(bits), 2)
        n_levels = 2 ** bits_export
        half = n_levels // 2
        if bits_export <= 8:
            u = packed[offset]
            offset += 1
        else:
            u = packed[offset] | (packed[offset + 1] << 8)
            offset += 2
        q = u - half
        scale = scales[ch_idx]
        out[ch_idx] = q / max(half - 1, 1) * scale
    return out


def export_self_compressed_renderer(
    model: nn.Module,
    output_path: Path,
    *,
    use_lzma: bool = True,
    arch_extra: dict | None = None,
) -> int:
    """Serialize a self-compressed renderer to a compact .bin file (SCv1 magic).

    The renderer must have been built with ``use_self_compress_codec=True``
    (i.e., ``swap_renderer_convs_with_self_compress`` was applied). All
    SelfCompressingConv2d layers are packed at their LEARNED per-channel
    bit-depth; protected layers (FP32 nn.Conv2d, nn.ConvTranspose2d,
    nn.Linear, nn.Embedding) are stored as float16.

    Args:
        model: AsymmetricPairGenerator (or PairGenerator) with SC layers swapped in.
        output_path: where to write the .bin file.
        use_lzma: if True, LZMA-compress the body. Disable only for debugging.
        arch_extra: optional extra metadata to include in the arch header
            (e.g. variant name). Caller-provided overrides win.

    Returns:
        Number of bytes written.
    """
    import json
    import lzma
    import struct

    from tac.self_compress import SelfCompressingConv2d

    model.eval()
    arch = _infer_asymmetric_config(model)
    if arch_extra:
        arch.update(arch_extra)

    layers_meta: list[dict] = []
    body_chunks: list[bytes] = []

    # Iterate ALL parameter-bearing modules. Order is fixed by named_modules
    # walk so the loader can reconstruct deterministically. We must SKIP
    # the inner nn.Conv2d that lives inside each SelfCompressingConv2d —
    # otherwise the SC weight gets stored TWICE (once compressed via the
    # SC branch, once as fp16 via the Conv2d branch). This was the
    # 574KB-vs-67KB byte-explosion bug caught during smoke testing.
    sc_inner_module_ids: set[int] = set()
    for _name, _module in model.named_modules():
        if isinstance(_module, SelfCompressingConv2d):
            sc_inner_module_ids.add(id(_module.conv))

    seen_emb_ids: set[int] = set()
    for name, module in model.named_modules():
        # Skip inner Conv2d wrapped by a SelfCompressingConv2d
        if id(module) in sc_inner_module_ids:
            continue
        if isinstance(module, SelfCompressingConv2d):
            bits = module.bit_depth.bits.detach().cpu().clamp(0.0, 8.0)
            bits_per_channel = [int(b.round().item()) for b in bits]
            # Promote 1-bit to 2-bit at export (matches packer).
            bits_per_channel = [max(b, 2) if b > 0 else 0 for b in bits_per_channel]

            weight = module.conv.weight.detach()
            packed_w, scales_w = _sc_pack_per_channel_weights(weight, bits_per_channel)

            # Pack scales as float16 prefix in the body
            scales_w_bytes = b"".join(struct.pack("<e", s) for s in scales_w)

            bias_packed = b""
            scales_b_bytes = b""
            has_bias = module.conv.bias is not None
            if has_bias:
                packed_b, scales_b = _sc_pack_per_channel_biases(
                    module.conv.bias.detach(), bits_per_channel,
                )
                bias_packed = packed_b
                scales_b_bytes = b"".join(struct.pack("<e", s) for s in scales_b)

            layer_blob = scales_w_bytes + packed_w + scales_b_bytes + bias_packed
            body_chunks.append(layer_blob)
            layers_meta.append({
                "name": name,
                "kind": "self_compress",
                "shape": list(module.conv.weight.shape),
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding,
                "dilation": module.dilation,
                "groups": module.groups,
                "padding_mode": module.padding_mode,
                "has_bias": has_bias,
                "bits_per_channel": bits_per_channel,
                "scales_w_len": len(scales_w_bytes),
                "packed_w_len": len(packed_w),
                "scales_b_len": len(scales_b_bytes),
                "packed_b_len": len(bias_packed),
            })
            continue

        if isinstance(module, nn.Embedding):
            if id(module) in seen_emb_ids:
                continue
            seen_emb_ids.add(id(module))
            arr = module.weight.detach().cpu().float().numpy()
            blob = arr.astype("float16").tobytes()
            body_chunks.append(blob)
            layers_meta.append({
                "name": name,
                "kind": "fp16_emb",
                "shape": list(module.weight.shape),
                "blob_len": len(blob),
            })
            continue

        if isinstance(module, nn.ConvTranspose2d):
            arr = module.weight.detach().cpu().float().numpy()
            blob = arr.astype("float16").tobytes()
            bias_blob = b""
            if module.bias is not None:
                bias_blob = module.bias.detach().cpu().float().numpy().astype("float16").tobytes()
            body_chunks.append(blob + bias_blob)
            layers_meta.append({
                "name": name,
                "kind": "fp16_convt",
                "shape": list(module.weight.shape),
                "stride": module.stride[0] if isinstance(module.stride, tuple) else module.stride,
                "padding": module.padding[0] if isinstance(module.padding, tuple) else module.padding,
                "has_bias": module.bias is not None,
                "weight_blob_len": len(blob),
                "bias_blob_len": len(bias_blob),
            })
            continue

        if isinstance(module, nn.Conv2d):
            # Plain (non-SC) Conv2d → fp16
            arr = module.weight.detach().cpu().float().numpy()
            blob = arr.astype("float16").tobytes()
            bias_blob = b""
            if module.bias is not None:
                bias_blob = module.bias.detach().cpu().float().numpy().astype("float16").tobytes()
            body_chunks.append(blob + bias_blob)
            layers_meta.append({
                "name": name,
                "kind": "fp16_conv",
                "shape": list(module.weight.shape),
                "stride": module.stride[0] if isinstance(module.stride, tuple) else module.stride,
                "padding": module.padding[0] if isinstance(module.padding, tuple) else module.padding,
                "dilation": module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation,
                "groups": module.groups,
                "padding_mode": module.padding_mode,
                "has_bias": module.bias is not None,
                "weight_blob_len": len(blob),
                "bias_blob_len": len(bias_blob),
            })
            continue

        if isinstance(module, nn.Linear):
            arr = module.weight.detach().cpu().float().numpy()
            blob = arr.astype("float16").tobytes()
            bias_blob = b""
            if module.bias is not None:
                bias_blob = module.bias.detach().cpu().float().numpy().astype("float16").tobytes()
            body_chunks.append(blob + bias_blob)
            layers_meta.append({
                "name": name,
                "kind": "fp16_linear",
                "shape": list(module.weight.shape),
                "has_bias": module.bias is not None,
                "weight_blob_len": len(blob),
                "bias_blob_len": len(bias_blob),
            })
            continue

    # Collect scalar nn.Parameters (e.g. blend_logit, noise_scale) that are
    # NOT inside any of the above modules. These are tiny but load-bearing.
    scalar_params: dict[str, float] = {}
    captured_module_param_ids: set[int] = set()
    for _name, module in model.named_modules():
        if isinstance(module, SelfCompressingConv2d):
            # SC submodule captures: inner Conv2d weight/bias + bit_depth.bits
            captured_module_param_ids.add(id(module.conv.weight))
            if module.conv.bias is not None:
                captured_module_param_ids.add(id(module.conv.bias))
            captured_module_param_ids.add(id(module.bit_depth.bits))
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding)):
            for p in module.parameters(recurse=False):
                captured_module_param_ids.add(id(p))
    for pname, param in model.named_parameters():
        if id(param) in captured_module_param_ids:
            continue
        if param.numel() == 1:
            scalar_params[pname] = float(param.item())

    body = b"".join(body_chunks)
    if use_lzma:
        body_compressed = lzma.compress(body, preset=9 | lzma.PRESET_EXTREME)
    else:
        body_compressed = body

    header = {
        "version": 1,
        "format": "self_compress_renderer_v1",
        "use_lzma": bool(use_lzma),
        "arch": arch,
        "layers": layers_meta,
        "scalar_params": scalar_params,
        "body_uncompressed_len": len(body),
        "body_compressed_len": len(body_compressed),
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    buf = bytearray()
    buf.extend(b"SCv1")
    buf.extend(struct.pack("<I", len(header_json)))
    buf.extend(header_json)
    buf.extend(struct.pack("<I", len(body_compressed)))
    buf.extend(body_compressed)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(buf))

    n_total_params = sum(p.numel() for p in model.parameters())
    bits_per_param = (len(buf) * 8) / max(n_total_params, 1)
    print(
        f"[scv1-export] {n_total_params:,} params → {len(buf):,} bytes "
        f"({bits_per_param:.2f} bits/param) | "
        f"body uncompressed={len(body):,} compressed={len(body_compressed):,}"
    )
    return len(buf)


def load_self_compressed_renderer(
    data_or_path,
    device: str = "cpu",
):
    """Reverse of export_self_compressed_renderer.

    Returns: AsymmetricPairGenerator (or PairGenerator) in eval mode with
    weights restored to their per-channel-quantized values.
    """
    import json
    import lzma
    import struct

    from tac.renderer import AsymmetricPairGenerator, build_renderer
    from tac.self_compress import (
        SelfCompressingConv2d,
        swap_renderer_convs_with_self_compress,
    )

    if isinstance(data_or_path, (str, Path)):
        data = Path(data_or_path).read_bytes()
    else:
        data = bytes(data_or_path)

    if data[:4] != b"SCv1":
        raise ValueError(
            f"Not a SCv1 self-compressed renderer (expected magic b'SCv1', "
            f"got {data[:4]!r})"
        )

    offset = 4
    header_len = struct.unpack("<I", data[offset:offset + 4])[0]
    offset += 4
    header = json.loads(data[offset:offset + header_len].decode("utf-8"))
    offset += header_len

    if header.get("version") != 1:
        raise ValueError(
            f"Unsupported SCv1 version {header.get('version')} (expected 1)"
        )

    body_len = struct.unpack("<I", data[offset:offset + 4])[0]
    offset += 4
    body_compressed = data[offset:offset + body_len]
    if header.get("use_lzma", True):
        body = lzma.decompress(body_compressed)
    else:
        body = body_compressed

    arch = header["arch"]
    pair_mode = arch.get("pair_mode", "asymmetric")

    # Reconstruct the model skeleton with the SAME constructor kwargs the
    # exporter saw, then apply the SC swap so all SelfCompressingConv2d
    # layers exist in the right places. The swap uses the SAME name-pattern
    # protection list so the (swapped vs FP32) split is byte-exact.
    # 2026-04-27 codex finding fix: thread use_ghost through both branches.
    # GhostConv layers serialize as nested Conv2d names (*.primary, *.ghost);
    # without this, Lane GH + Lane S/Ω composition fails at load time.
    if pair_mode == "asymmetric":
        model = AsymmetricPairGenerator(
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
    else:
        model = build_renderer(
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

    swap_renderer_convs_with_self_compress(model, init_bits=8.0)

    # Build module lookup for restoration
    name_to_module: dict[str, nn.Module] = dict(model.named_modules())
    body_offset = 0

    seen_emb_ids: set[int] = set()
    for layer_meta in header["layers"]:
        name = layer_meta["name"]
        kind = layer_meta["kind"]
        module = name_to_module.get(name)
        if module is None:
            raise RuntimeError(
                f"SCv1 load: layer {name!r} not found in reconstructed model. "
                f"Header arch was {arch}; the model construction path must "
                f"match the export-time path exactly."
            )

        if kind == "self_compress":
            assert isinstance(module, SelfCompressingConv2d), (
                f"Header says {name} is SC but loaded module is "
                f"{type(module).__name__}"
            )
            bits_per_channel = layer_meta["bits_per_channel"]
            scales_w_len = layer_meta["scales_w_len"]
            packed_w_len = layer_meta["packed_w_len"]
            scales_b_len = layer_meta["scales_b_len"]
            packed_b_len = layer_meta["packed_b_len"]

            scales_w_bytes = body[body_offset:body_offset + scales_w_len]
            body_offset += scales_w_len
            packed_w = body[body_offset:body_offset + packed_w_len]
            body_offset += packed_w_len
            scales_b_bytes = body[body_offset:body_offset + scales_b_len]
            body_offset += scales_b_len
            packed_b = body[body_offset:body_offset + packed_b_len]
            body_offset += packed_b_len

            n_out = len(bits_per_channel)
            scales_w = [
                struct.unpack("<e", scales_w_bytes[i * 2:(i + 1) * 2])[0]
                for i in range(n_out)
            ]
            shape = layer_meta["shape"]
            fan_in = shape[1] * shape[2] * shape[3]  # (C_out, C_in_pg, kH, kW)
            flat = _sc_unpack_per_channel_weights(
                packed_w, bits_per_channel, fan_in, scales_w,
            )
            with torch.no_grad():
                module.conv.weight.copy_(flat.reshape(shape))
                # Restore bit_depth tensor with the loaded values
                module.bit_depth.bits.copy_(
                    torch.tensor([float(b) for b in bits_per_channel])
                )
                if layer_meta["has_bias"] and module.conv.bias is not None:
                    scales_b = [
                        struct.unpack("<e", scales_b_bytes[i * 2:(i + 1) * 2])[0]
                        for i in range(n_out)
                    ]
                    bias_t = _sc_unpack_per_channel_biases(
                        packed_b, bits_per_channel, scales_b,
                    )
                    module.conv.bias.copy_(bias_t)
        elif kind == "fp16_emb":
            if id(module) in seen_emb_ids:
                # Shared embedding — we already restored it via the renderer's
                # own copy. Still advance the offset.
                blob_len = layer_meta["blob_len"]
                body_offset += blob_len
                continue
            seen_emb_ids.add(id(module))
            blob_len = layer_meta["blob_len"]
            blob = body[body_offset:body_offset + blob_len]
            body_offset += blob_len
            import numpy as np
            arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32).copy()
            with torch.no_grad():
                module.weight.copy_(torch.from_numpy(arr).reshape(layer_meta["shape"]))
        elif kind in ("fp16_conv", "fp16_convt", "fp16_linear"):
            w_len = layer_meta["weight_blob_len"]
            b_len = layer_meta["bias_blob_len"]
            w_blob = body[body_offset:body_offset + w_len]
            body_offset += w_len
            b_blob = body[body_offset:body_offset + b_len]
            body_offset += b_len
            import numpy as np
            w_arr = np.frombuffer(w_blob, dtype=np.float16).astype(np.float32).copy()
            with torch.no_grad():
                module.weight.copy_(torch.from_numpy(w_arr).reshape(layer_meta["shape"]))
                if layer_meta["has_bias"] and module.bias is not None:
                    b_arr = np.frombuffer(b_blob, dtype=np.float16).astype(np.float32).copy()
                    module.bias.copy_(torch.from_numpy(b_arr))
        else:
            raise ValueError(f"SCv1 load: unknown layer kind {kind!r}")

    # Restore scalar parameters
    scalar_params = header.get("scalar_params", {})
    if scalar_params:
        param_dict = dict(model.named_parameters())
        with torch.no_grad():
            for pname, pval in scalar_params.items():
                if pname in param_dict:
                    param_dict[pname].fill_(pval)

    # Verify shared embedding invariant survived
    if hasattr(model, "renderer") and hasattr(model, "motion"):
        r_emb = getattr(model.renderer, "embedding", None)
        m_emb = getattr(model.motion, "embedding", None)
        if r_emb is not None and m_emb is not None:
            assert r_emb is m_emb, "SCv1 load: shared embedding invariant violated"

    model = model.to(device)
    model.eval()
    return model


# ─── Lane Ω: per-weight Hessian-aware quantization (OMG1 magic) ──────────
#
# The OMG1 binary format encodes a renderer where each Conv2d / Linear weight
# in the eligible-layer set has its OWN bit-depth (uint8, 1..8) chosen by the
# Phase 2 water-fill allocator. Unlike SCv1 (per-channel), this is
# per-element. Protected layers (renderer.head, motion.head, FiLM linears,
# fuse_conv) stay FP16 — same protection list as SCv1.
#
# Layout:
#   magic (4B = b"OMG1")
#   header_len (4B little-endian uint32)
#   header_json (UTF-8, JSON)
#   body_len (4B little-endian uint32)
#   body_compressed (LZMA-compressed body bytes)
#
# Body for an "omega" layer:
#   scales_per_channel (C_out × float16)
#   bits_lzma (LZMA-compressed uint8 buffer of len = numel(weight))
#   packed_values (per-element symmetric int values bit-packed, 1..8 bits each)
#   bias_blob (optional, float16)
#
# Per-element bit-packing strategy:
#   We bit-pack the values in flat row-major order. For each element, with
#   bit-depth b, we encode the signed integer code in [-(2^(b-1)-1),
#   2^(b-1)-1] using b bits (sign-magnitude with the convention:
#     sign bit (1=negative), then (b-1) magnitude bits, OR for b=1:
#     0 = +scale, 1 = -scale).
#   We use a hand-rolled bit writer (LSB-first within byte) — simple and
#   deterministic, no external deps. The writer is also LZMA-compressed
#   in the body, so coding-overhead efficiency matters less than
#   correctness.


_OMEGA_MAGIC = b"OMG1"
_OMEGA_VERSION = 1


def _bitpack_values_with_bits(
    values: "list[int] | torch.Tensor",
    bits: "list[int] | torch.Tensor",
) -> bytes:
    """LSB-first variable-width bit-packed encoding of signed integer values.

    For each (value, b) pair:
        sign_bit = 1 if value < 0 else 0     (b ≥ 2)
        magnitude = abs(value), 0..(2^(b-1) - 1)
        encoded = (magnitude << 1) | sign_bit  (b ≥ 2, so total = b bits)
        encoded = (1 if value < 0 else 0)      (b == 1)

    Bytes are written LSB-first (lowest bit of the first value lands in
    bit 0 of byte 0).

    Returns: bytes with ceil(sum(bits)/8) bytes.
    """
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    if isinstance(bits, torch.Tensor):
        bits = bits.tolist()
    if len(values) != len(bits):
        raise ValueError(
            f"values and bits length mismatch: {len(values)} vs {len(bits)}"
        )

    total_bits = sum(int(b) for b in bits)
    out = bytearray((total_bits + 7) // 8)
    bit_offset = 0
    for v, b in zip(values, bits):
        b = int(b)
        v = int(v)
        if b == 1:
            code = 1 if v < 0 else 0
        else:
            mag = abs(v)
            max_mag = (1 << (b - 1)) - 1
            if mag > max_mag:
                mag = max_mag
            sign = 1 if v < 0 else 0
            code = (mag << 1) | sign
        # write `b` bits LSB-first starting at bit_offset
        for i in range(b):
            bit = (code >> i) & 1
            byte_idx = (bit_offset + i) >> 3
            bit_in_byte = (bit_offset + i) & 7
            out[byte_idx] |= bit << bit_in_byte
        bit_offset += b
    return bytes(out)


def _bitunpack_values_with_bits(
    blob: bytes,
    bits: "list[int] | torch.Tensor",
) -> list[int]:
    """Inverse of _bitpack_values_with_bits — recover signed integer codes."""
    if isinstance(bits, torch.Tensor):
        bits = bits.tolist()
    out: list[int] = []
    bit_offset = 0
    for b in bits:
        b = int(b)
        code = 0
        for i in range(b):
            byte_idx = (bit_offset + i) >> 3
            bit_in_byte = (bit_offset + i) & 7
            bit = (blob[byte_idx] >> bit_in_byte) & 1
            code |= bit << i
        if b == 1:
            # 1-bit code 0 = +scale (value index +1), code 1 = -scale (value -1).
            # We never produce value=0 from a 1-bit weight (matches packer +
            # FrozenBitFakeQuant 1-bit sign-only path).
            v = -1 if code == 1 else 1
        else:
            sign = code & 1
            mag = code >> 1
            v = -mag if sign else mag
        out.append(v)
        bit_offset += b
    return out


def _omega_quantize_layer(
    weight: torch.Tensor,
    bits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Quantize a weight tensor under per-element bits → (scales, bits_uint8, codes).

    Returns:
        scales: (C_out,) float16 per-channel scales.
        bits_uint8: same shape as weight, uint8.
        codes: list[int] of integer codes (signed), one per weight element,
            in flat row-major order. The packer turns these into bytes.
    """
    if bits.shape != weight.shape:
        raise ValueError(
            f"bits shape {tuple(bits.shape)} must match weight shape "
            f"{tuple(weight.shape)}"
        )
    out_dim = weight.shape[0]
    scales = weight.detach().reshape(out_dim, -1).abs().max(dim=1).values
    scales = scales.clamp(min=1e-8).to(torch.float32)

    bits_f = bits.to(torch.float32)
    levels = (2.0 ** (bits_f - 1.0) - 1.0).clamp(min=1.0)
    scale_b = scales.view(-1, *([1] * (weight.dim() - 1)))
    step = scale_b / levels
    # integer code per element
    codes_f = torch.round(weight.detach() / step)
    # clip per-element to [-levels, +levels]
    codes_clipped = torch.where(
        codes_f > levels, levels, torch.where(codes_f < -levels, -levels, codes_f),
    )
    codes_int = codes_clipped.to(torch.int64).reshape(-1).tolist()
    # For 1-bit weights, the encoding represents the SIGN only:
    #   value index +1 = +scale (encoded as code 0)
    #   value index -1 = -scale (encoded as code 1)
    # If the rounded code lands at 0, snap to +1 (sign of weight, default +).
    bits_flat = bits.reshape(-1).tolist()
    weight_flat = weight.detach().reshape(-1).tolist()
    for i, b in enumerate(bits_flat):
        if int(b) == 1:
            codes_int[i] = -1 if weight_flat[i] < 0 else 1
    return scales.to(torch.float16), bits.to(torch.uint8), codes_int


def _omega_dequantize_layer(
    codes: list[int],
    bits: torch.Tensor,
    scales: torch.Tensor,
    weight_shape: torch.Size,
) -> torch.Tensor:
    """Reverse of _omega_quantize_layer — recover float weight."""
    bits_f = bits.to(torch.float32).reshape(-1)
    levels = (2.0 ** (bits_f - 1.0) - 1.0).clamp(min=1.0)
    # Per-element scale (broadcast from per-channel)
    out_dim = weight_shape[0]
    scales_f = scales.to(torch.float32).reshape(out_dim, *([1] * (len(weight_shape) - 1)))
    # Per-element step: scale_per_channel / level_per_element
    # Build a same-shape scale tensor first.
    full_scale = scales_f.expand(weight_shape).contiguous().reshape(-1)
    step = full_scale / levels
    codes_t = torch.tensor(codes, dtype=torch.float32)
    # For 1-bit (level=1), codes are -1 or +1 → value = ±scale.
    # For b≥2, value = code * step.
    one_bit_mask = (bits_f == 1.0)
    values_general = codes_t * step
    values_one = codes_t * full_scale  # ±1 * scale
    values = torch.where(one_bit_mask, values_one, values_general)
    return values.reshape(weight_shape)


def _maybe_unwrap_learnable_bit_model(
    model: nn.Module,
    bits_per_weight: dict | None,
) -> tuple[nn.Module, dict]:
    """Unwrap a LearnableBitConv2d-wrapped model into a plain Conv2d model
    + auto-derived bits dict.

    Lane Ω-V2 wraps eligible Conv2d layers in ``LearnableBitConv2d``. The
    OMG1 export walker (below) expects to find ``nn.Conv2d`` modules with
    a matching entry in ``bits_per_weight`` keyed by ``<full_name>.weight``.
    This helper:

      1. Detects LearnableBitConv2d modules.
      2. Replaces each with its underlying ``nn.Conv2d`` (carrying the
         same float weights — the QAT loop has already fine-tuned them).
      3. Adds the rounded ``bits_rounded()`` tensor for that layer to the
         returned bits dict (under the wrapper's qualified name).
      4. Leaves caller-supplied ``bits_per_weight`` entries untouched
         (caller bits override auto-derived ones).

    The unwrap is done on a SHALLOW COPY of the model's module tree so we
    don't mutate the caller's model. (We rebuild the module hierarchy by
    re-binding `setattr` on shallow copies; the underlying weight Tensors
    are shared — this is fine because the export only reads them.)

    Returns:
        (plain_model, merged_bits_per_weight)
    """
    # Lazy import to avoid circular dependency (learnable_bit_quant imports
    # from self_compress which is in the same package as renderer_export).
    try:
        from tac.learnable_bit_quant import LearnableBitConv2d  # noqa: WPS433
    except ImportError:
        # learnable_bit_quant not present (older checkpouts) — pass through.
        return model, dict(bits_per_weight or {})

    has_wrapper = any(
        isinstance(m, LearnableBitConv2d) for m in model.modules()
    )
    if not has_wrapper:
        return model, dict(bits_per_weight or {})

    auto_bits: dict[str, torch.Tensor] = {}

    parents: dict[str, nn.Module] = {"": model}
    for name, mod in model.named_modules():
        parents[name] = mod

    # Collect (parent, child_name, full_name, wrapper) tuples first.
    to_replace: list[tuple[nn.Module, str, str, "LearnableBitConv2d"]] = []
    for full_name, mod in model.named_modules():
        if isinstance(mod, LearnableBitConv2d):
            if "." in full_name:
                parent_name, child_name = full_name.rsplit(".", 1)
                parent = parents[parent_name]
            else:
                parent = model
                child_name = full_name
            to_replace.append((parent, child_name, full_name, mod))

    for parent, child_name, full_name, wrapper in to_replace:
        bits_int = wrapper.bit_depth.bits_rounded().detach().cpu()
        auto_bits[f"{full_name}.weight"] = bits_int
        # Replace the wrapper with its underlying Conv2d. The Conv2d
        # carries the QAT-fine-tuned float weight; the export quantizer
        # will re-round per the bits dict.
        setattr(parent, child_name, wrapper.conv)

    # Caller-supplied bits override auto-derived ones (rare; mostly for
    # tests where the caller wants to inject specific bits).
    merged = {**auto_bits, **(bits_per_weight or {})}
    return model, merged


def export_omega_renderer(
    model: nn.Module,
    bits_per_weight: dict | None,
    output_path: Path,
    *,
    use_lzma: bool = True,
    arch_extra: dict | None = None,
) -> int:
    """Serialize a renderer with per-weight Hessian-aware bit-depths to OMG1.

    Args:
        model: AsymmetricPairGenerator (or build_renderer output) — fp32 weights.
            The Lane Ω QAT loop has already fine-tuned these weights given
            the per-element bit-depths.

            **Lane Ω-V2 path:** if the model contains
            ``LearnableBitConv2d`` modules (Lane Ω-V2's
            learnable-bit wrappers), this function auto-unwraps them and
            extracts the per-weight bits from each wrapper's
            ``bit_depth.bits_rounded()``. The caller may pass ``None``
            for ``bits_per_weight`` in that case.
        bits_per_weight: dict mapping `<full_name>.weight` → uint8 tensor of
            same shape as that parameter, values in [1, 8]. Layers NOT in
            this dict are stored as float16 (= protected / non-quantized).
            May be ``None`` if the model has LearnableBitConv2d modules.
        output_path: where to write the .bin file.
        use_lzma: LZMA-compress the body and the bits buffer.
        arch_extra: optional extra metadata for the arch header.

    Returns:
        Number of bytes written.
    """
    import json
    import lzma
    import struct

    model, bits_per_weight = _maybe_unwrap_learnable_bit_model(
        model, bits_per_weight
    )
    if bits_per_weight is None:
        bits_per_weight = {}

    model.eval()
    arch = _infer_asymmetric_config(model)
    if arch_extra:
        arch.update(arch_extra)

    layers_meta: list[dict] = []
    body_chunks: list[bytes] = []

    seen_emb_ids: set[int] = set()
    eligible_param_ids: set[int] = set()
    # Pre-scan: which weight parameter ids are "omega-quantized"?
    name_to_param: dict[str, nn.Parameter] = {
        n: p for n, p in model.named_parameters()
    }
    for layer_name in bits_per_weight:
        p = name_to_param.get(layer_name)
        if p is not None:
            eligible_param_ids.add(id(p))

    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            if id(module) in seen_emb_ids:
                continue
            seen_emb_ids.add(id(module))
            arr = module.weight.detach().cpu().float().numpy()
            blob = arr.astype("float16").tobytes()
            body_chunks.append(blob)
            layers_meta.append({
                "name": name,
                "kind": "fp16_emb",
                "shape": list(module.weight.shape),
                "blob_len": len(blob),
            })
            continue

        if isinstance(module, nn.ConvTranspose2d):
            arr = module.weight.detach().cpu().float().numpy()
            blob = arr.astype("float16").tobytes()
            bias_blob = b""
            if module.bias is not None:
                bias_blob = module.bias.detach().cpu().float().numpy().astype("float16").tobytes()
            body_chunks.append(blob + bias_blob)
            layers_meta.append({
                "name": name,
                "kind": "fp16_convt",
                "shape": list(module.weight.shape),
                "stride": module.stride[0] if isinstance(module.stride, tuple) else module.stride,
                "padding": module.padding[0] if isinstance(module.padding, tuple) else module.padding,
                "has_bias": module.bias is not None,
                "weight_blob_len": len(blob),
                "bias_blob_len": len(bias_blob),
            })
            continue

        if isinstance(module, nn.Conv2d):
            weight_name = f"{name}.weight"
            if weight_name in bits_per_weight:
                bits_tensor = bits_per_weight[weight_name].to(torch.uint8)
                if bits_tensor.shape != module.weight.shape:
                    raise ValueError(
                        f"OMG1 export: bits shape {tuple(bits_tensor.shape)} "
                        f"does not match {weight_name} {tuple(module.weight.shape)}"
                    )
                scales_f16, bits_u8, codes = _omega_quantize_layer(module.weight, bits_tensor)
                bits_blob = bits_u8.contiguous().reshape(-1).numpy().tobytes()
                bits_blob_compressed = lzma.compress(bits_blob, preset=9) if use_lzma else bits_blob
                packed_values = _bitpack_values_with_bits(codes, bits_u8.reshape(-1).tolist())
                scales_blob = scales_f16.contiguous().numpy().tobytes()
                bias_blob = b""
                if module.bias is not None:
                    bias_blob = module.bias.detach().cpu().float().numpy().astype("float16").tobytes()

                layer_blob = (
                    scales_blob
                    + struct.pack("<I", len(bits_blob_compressed))
                    + bits_blob_compressed
                    + struct.pack("<I", len(packed_values))
                    + packed_values
                    + bias_blob
                )
                body_chunks.append(layer_blob)
                layers_meta.append({
                    "name": name,
                    "kind": "omega",
                    "shape": list(module.weight.shape),
                    "stride": module.stride[0] if isinstance(module.stride, tuple) else module.stride,
                    "padding": module.padding[0] if isinstance(module.padding, tuple) else module.padding,
                    "dilation": module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation,
                    "groups": module.groups,
                    "padding_mode": module.padding_mode,
                    "has_bias": module.bias is not None,
                    "scales_f16_len": len(scales_blob),
                    "bits_blob_compressed": bool(use_lzma),
                    "bits_blob_uncompressed_len": len(bits_blob),
                    "bias_blob_len": len(bias_blob),
                    "total_bits": int(bits_u8.to(torch.int64).sum().item()),
                    "n_weights": int(bits_u8.numel()),
                })
                continue
            # Fallthrough: this is an UNEligible Conv2d → store FP16
            arr = module.weight.detach().cpu().float().numpy()
            blob = arr.astype("float16").tobytes()
            bias_blob = b""
            if module.bias is not None:
                bias_blob = module.bias.detach().cpu().float().numpy().astype("float16").tobytes()
            body_chunks.append(blob + bias_blob)
            layers_meta.append({
                "name": name,
                "kind": "fp16_conv",
                "shape": list(module.weight.shape),
                "stride": module.stride[0] if isinstance(module.stride, tuple) else module.stride,
                "padding": module.padding[0] if isinstance(module.padding, tuple) else module.padding,
                "dilation": module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation,
                "groups": module.groups,
                "padding_mode": module.padding_mode,
                "has_bias": module.bias is not None,
                "weight_blob_len": len(blob),
                "bias_blob_len": len(bias_blob),
            })
            continue

        if isinstance(module, nn.Linear):
            weight_name = f"{name}.weight"
            if weight_name in bits_per_weight:
                bits_tensor = bits_per_weight[weight_name].to(torch.uint8)
                scales_f16, bits_u8, codes = _omega_quantize_layer(module.weight, bits_tensor)
                bits_blob = bits_u8.contiguous().reshape(-1).numpy().tobytes()
                bits_blob_compressed = lzma.compress(bits_blob, preset=9) if use_lzma else bits_blob
                packed_values = _bitpack_values_with_bits(codes, bits_u8.reshape(-1).tolist())
                scales_blob = scales_f16.contiguous().numpy().tobytes()
                bias_blob = b""
                if module.bias is not None:
                    bias_blob = module.bias.detach().cpu().float().numpy().astype("float16").tobytes()
                layer_blob = (
                    scales_blob
                    + struct.pack("<I", len(bits_blob_compressed))
                    + bits_blob_compressed
                    + struct.pack("<I", len(packed_values))
                    + packed_values
                    + bias_blob
                )
                body_chunks.append(layer_blob)
                layers_meta.append({
                    "name": name,
                    "kind": "omega_linear",
                    "shape": list(module.weight.shape),
                    "has_bias": module.bias is not None,
                    "scales_f16_len": len(scales_blob),
                    "bits_blob_compressed": bool(use_lzma),
                    "bits_blob_uncompressed_len": len(bits_blob),
                    "bias_blob_len": len(bias_blob),
                    "total_bits": int(bits_u8.to(torch.int64).sum().item()),
                    "n_weights": int(bits_u8.numel()),
                })
                continue
            arr = module.weight.detach().cpu().float().numpy()
            blob = arr.astype("float16").tobytes()
            bias_blob = b""
            if module.bias is not None:
                bias_blob = module.bias.detach().cpu().float().numpy().astype("float16").tobytes()
            body_chunks.append(blob + bias_blob)
            layers_meta.append({
                "name": name,
                "kind": "fp16_linear",
                "shape": list(module.weight.shape),
                "has_bias": module.bias is not None,
                "weight_blob_len": len(blob),
                "bias_blob_len": len(bias_blob),
            })
            continue

    # Scalar params (mirrors SCv1)
    captured_module_param_ids: set[int] = set()
    for _name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding)):
            for p in mod.parameters(recurse=False):
                captured_module_param_ids.add(id(p))
    scalar_params: dict[str, float] = {}
    for pname, param in model.named_parameters():
        if id(param) in captured_module_param_ids:
            continue
        if param.numel() == 1:
            scalar_params[pname] = float(param.item())

    body = b"".join(body_chunks)
    body_compressed = lzma.compress(body, preset=9 | lzma.PRESET_EXTREME) if use_lzma else body

    header = {
        "version": _OMEGA_VERSION,
        "format": "omega_renderer_v1",
        "use_lzma": bool(use_lzma),
        "arch": arch,
        "layers": layers_meta,
        "scalar_params": scalar_params,
        "body_uncompressed_len": len(body),
        "body_compressed_len": len(body_compressed),
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    buf = bytearray()
    buf.extend(_OMEGA_MAGIC)
    buf.extend(struct.pack("<I", len(header_json)))
    buf.extend(header_json)
    buf.extend(struct.pack("<I", len(body_compressed)))
    buf.extend(body_compressed)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(buf))

    n_total_params = sum(p.numel() for p in model.parameters())
    bits_per_param = (len(buf) * 8) / max(n_total_params, 1)
    print(
        f"[omg1-export] {n_total_params:,} params → {len(buf):,} bytes "
        f"({bits_per_param:.2f} bits/param) | "
        f"body uncompressed={len(body):,} compressed={len(body_compressed):,}"
    )
    return len(buf)


def load_omega_renderer(
    data_or_path,
    device: str = "cpu",
):
    """Reverse of export_omega_renderer."""
    import json
    import lzma
    import struct

    from tac.renderer import AsymmetricPairGenerator, build_renderer

    if isinstance(data_or_path, (str, Path)):
        data = Path(data_or_path).read_bytes()
    else:
        data = bytes(data_or_path)

    if data[:4] != _OMEGA_MAGIC:
        raise ValueError(
            f"Not an OMG1 omega renderer (expected magic b'OMG1', "
            f"got {data[:4]!r})"
        )
    offset = 4
    header_len = struct.unpack("<I", data[offset:offset + 4])[0]
    offset += 4
    header = json.loads(data[offset:offset + header_len].decode("utf-8"))
    offset += header_len

    if header.get("version") != _OMEGA_VERSION:
        raise ValueError(
            f"Unsupported OMG1 version {header.get('version')} (expected {_OMEGA_VERSION})"
        )

    body_len = struct.unpack("<I", data[offset:offset + 4])[0]
    offset += 4
    body_compressed = data[offset:offset + body_len]
    body = lzma.decompress(body_compressed) if header.get("use_lzma", True) else body_compressed

    arch = header["arch"]
    pair_mode = arch.get("pair_mode", "asymmetric")
    # 2026-04-27 codex finding fix: thread use_ghost through both branches.
    # GhostConv layers serialize as nested Conv2d names (*.primary, *.ghost);
    # without this, Lane GH + Lane S/Ω composition fails at load time.
    if pair_mode == "asymmetric":
        model = AsymmetricPairGenerator(
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
    else:
        model = build_renderer(
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

    name_to_module: dict[str, nn.Module] = dict(model.named_modules())
    body_offset = 0
    seen_emb_ids: set[int] = set()
    import numpy as np

    for layer_meta in header["layers"]:
        name = layer_meta["name"]
        kind = layer_meta["kind"]
        module = name_to_module.get(name)
        if module is None:
            raise RuntimeError(
                f"OMG1 load: layer {name!r} not found in reconstructed model"
            )

        if kind == "fp16_emb":
            blob_len = layer_meta["blob_len"]
            if id(module) in seen_emb_ids:
                body_offset += blob_len
                continue
            seen_emb_ids.add(id(module))
            blob = body[body_offset:body_offset + blob_len]
            body_offset += blob_len
            arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32).copy()
            with torch.no_grad():
                module.weight.copy_(torch.from_numpy(arr).reshape(layer_meta["shape"]))

        elif kind in ("fp16_conv", "fp16_convt", "fp16_linear"):
            w_len = layer_meta["weight_blob_len"]
            b_len = layer_meta["bias_blob_len"]
            w_blob = body[body_offset:body_offset + w_len]
            body_offset += w_len
            b_blob = body[body_offset:body_offset + b_len]
            body_offset += b_len
            w_arr = np.frombuffer(w_blob, dtype=np.float16).astype(np.float32).copy()
            with torch.no_grad():
                module.weight.copy_(torch.from_numpy(w_arr).reshape(layer_meta["shape"]))
                if layer_meta["has_bias"] and module.bias is not None:
                    b_arr = np.frombuffer(b_blob, dtype=np.float16).astype(np.float32).copy()
                    module.bias.copy_(torch.from_numpy(b_arr))

        elif kind in ("omega", "omega_linear"):
            scales_len = layer_meta["scales_f16_len"]
            scales_blob = body[body_offset:body_offset + scales_len]
            body_offset += scales_len
            bits_compressed_len = struct.unpack("<I", body[body_offset:body_offset + 4])[0]
            body_offset += 4
            bits_blob_in = body[body_offset:body_offset + bits_compressed_len]
            body_offset += bits_compressed_len
            packed_len = struct.unpack("<I", body[body_offset:body_offset + 4])[0]
            body_offset += 4
            packed_blob = body[body_offset:body_offset + packed_len]
            body_offset += packed_len
            bias_blob = b""
            if layer_meta["has_bias"]:
                bias_len = layer_meta["bias_blob_len"]
                bias_blob = body[body_offset:body_offset + bias_len]
                body_offset += bias_len

            shape = torch.Size(layer_meta["shape"])
            n_weights = int(layer_meta["n_weights"])
            scales_arr = np.frombuffer(scales_blob, dtype=np.float16).astype(np.float32).copy()
            scales_t = torch.from_numpy(scales_arr)
            if scales_t.numel() != shape[0]:
                raise RuntimeError(
                    f"OMG1 load: layer {name} scales has {scales_t.numel()} entries, "
                    f"expected {shape[0]}"
                )
            bits_uncompressed = (
                lzma.decompress(bits_blob_in)
                if layer_meta.get("bits_blob_compressed", True)
                else bits_blob_in
            )
            if len(bits_uncompressed) != n_weights:
                raise RuntimeError(
                    f"OMG1 load: layer {name} bits decompressed to "
                    f"{len(bits_uncompressed)} bytes, expected {n_weights}"
                )
            bits_t = torch.frombuffer(bytearray(bits_uncompressed), dtype=torch.uint8).clone()
            bits_t = bits_t.reshape(shape)

            codes = _bitunpack_values_with_bits(
                packed_blob, bits_t.reshape(-1).tolist(),
            )
            weight_t = _omega_dequantize_layer(codes, bits_t, scales_t, shape)

            with torch.no_grad():
                module.weight.copy_(weight_t.to(module.weight.dtype))
                if layer_meta["has_bias"] and module.bias is not None and bias_blob:
                    b_arr = np.frombuffer(bias_blob, dtype=np.float16).astype(np.float32).copy()
                    module.bias.copy_(torch.from_numpy(b_arr))
        else:
            raise ValueError(f"OMG1 load: unknown layer kind {kind!r}")

    scalar_params = header.get("scalar_params", {})
    if scalar_params:
        param_dict = dict(model.named_parameters())
        with torch.no_grad():
            for pname, pval in scalar_params.items():
                if pname in param_dict:
                    param_dict[pname].fill_(pval)

    if hasattr(model, "renderer") and hasattr(model, "motion"):
        r_emb = getattr(model.renderer, "embedding", None)
        m_emb = getattr(model.motion, "embedding", None)
        if r_emb is not None and m_emb is not None:
            assert r_emb is m_emb, "OMG1 load: shared embedding invariant violated"

    model = model.to(device)
    model.eval()
    return model


# ─── Lane J-NWC: Neural Weight Compression (NWC1 magic) ──────────────────
#
# arXiv 2510.11234 — "Neural Weight Compression for Language Models" (late 2025).
# Trains a tiny VQ-VAE-style codec on a corpus of small renderers we have
# already trained (hundreds saved under experiments/results/). At deploy time,
# the codec is loaded, every floating param tensor in `model.state_dict()` is
# encoded to (codebook_indices + per-block float16 scales), and the codec
# weights themselves are bundled into the binary so that
# `load_neural_compressed_checkpoint` can decode without external state.
#
# NWC1 layout:
#   magic            (4B = b"NWC1")
#   header_len       (4B little-endian uint32)
#   header_json      (UTF-8, JSON)
#       header.config        — nn arch config (from _infer_asymmetric_config)
#       header.codec_config  — block_size, codebook_size, latent_dim, hidden
#       header.tensors       — per-tensor metadata: name, shape, blob_len
#   codec_state_blob_len     (4B uint32) — torch.save bytes (.pt of codec weights)
#   codec_state_blob         (codec_state_blob_len bytes)
#   per_tensor_blobs         — sequence of (4B blob_len, blob_bytes) where the
#                              blob is exactly what `WeightCodec.encode()`
#                              produced. The order matches header.tensors.
#
# The header.tensors metadata lets the loader iterate without scanning blob
# contents (length-prefixed framing is also defensive against truncation).

_NWC_MAGIC = b"NWC1"
_NWC_VERSION = 1


def _nwc_collect_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Walk a renderer and collect every floating-point parameter tensor
    deterministically (sorted by named_parameters order).

    Skips non-floating-point and zero-element tensors. Uses
    `state_dict()` keys so buffers (e.g., running stats) are also captured —
    this matches the FP8H / OMG1 round-trip expectation.
    """
    out: list[tuple[str, torch.Tensor]] = []
    for name, val in model.state_dict().items():
        if not isinstance(val, torch.Tensor):
            continue
        if not torch.is_floating_point(val):
            continue
        if val.numel() == 0:
            continue
        out.append((name, val.detach().cpu()))
    return out


def export_neural_compressed_checkpoint(
    model: nn.Module,
    codec_path: str | Path,
    output_path: str | Path,
    *,
    arch_extra: dict | None = None,
) -> int:
    """Compress a renderer with a pre-trained NWC codec → NWC1 binary.

    Args:
        model: a fp32 renderer (typically AsymmetricPairGenerator) whose
            state_dict will be encoded tensor-by-tensor through the codec.
        codec_path: path to the trained `WeightCodec` checkpoint, as written
            by `experiments/train_neural_weight_codec.py` (a torch.save dict
            with keys ``codec_state_dict`` and ``codec_config``).
        output_path: where to write the .bin.
        arch_extra: optional extra metadata for the arch header.

    Returns: number of bytes written.
    """
    from tac.neural_weight_codec import WeightCodec, WeightCodecConfig

    model.eval()
    try:
        config = _infer_asymmetric_config(model)
    except Exception:
        config = {"tensor_only": True}
    if arch_extra:
        config = {**config, **arch_extra}

    codec_blob = Path(codec_path).read_bytes()
    codec_state = torch.load(io.BytesIO(codec_blob), map_location="cpu", weights_only=False)
    codec_cfg_dict = codec_state.get("codec_config")
    if codec_cfg_dict is None:
        raise ValueError(
            f"Codec checkpoint at {codec_path} missing 'codec_config' — "
            "rebuild via experiments/train_neural_weight_codec.py"
        )
    codec_config = WeightCodecConfig(**codec_cfg_dict)
    codec = WeightCodec(codec_config)
    codec.load_state_dict(codec_state["codec_state_dict"])
    codec.eval()

    tensors = _nwc_collect_tensors(model)
    blob_meta: list[dict] = []
    blobs: list[bytes] = []
    for name, t in tensors:
        blob = codec.encode(t)
        blob_meta.append({
            "name": name,
            "shape": list(t.shape),
            "numel": int(t.numel()),
            "dtype_in": str(t.dtype).replace("torch.", ""),
            "blob_len": len(blob),
        })
        blobs.append(blob)

    header = {
        "magic": "NWC1",
        "version": _NWC_VERSION,
        "format": "neural_weight_compression",
        "config": config,
        "codec_config": codec_cfg_dict,
        "tensors": blob_meta,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    buf = bytearray()
    buf.extend(_NWC_MAGIC)
    buf.extend(struct.pack("<I", len(header_json)))
    buf.extend(header_json)
    buf.extend(struct.pack("<I", len(codec_blob)))
    buf.extend(codec_blob)
    for raw in blobs:
        buf.extend(struct.pack("<I", len(raw)))
        buf.extend(raw)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(buf))

    param_count = sum(p.numel() for p in model.parameters())
    if param_count > 0:
        print(
            f"[nwc-export] {param_count:,} params → {len(buf):,} bytes "
            f"({len(buf) / param_count * 8:.2f} bits/param incl. codec)"
        )
    return len(buf)


def load_neural_compressed_checkpoint(
    data_or_path: Union[bytes, Path, str],
    device: str = "cpu",
) -> nn.Module:
    """Restore a renderer from an NWC1 ``.bin``.

    Reads the embedded codec, decodes every tensor blob, and loads them into a
    freshly built model whose architecture is inferred from the JSON header.
    """
    from tac.neural_weight_codec import WeightCodec, WeightCodecConfig

    if isinstance(data_or_path, (str, Path)):
        data = Path(data_or_path).read_bytes()
    else:
        data = data_or_path

    if data[:4] != _NWC_MAGIC:
        raise ValueError(
            f"Not an NWC1 checkpoint (expected magic b'NWC1', got {data[:4]!r})"
        )
    offset = 4
    header_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    header_json = data[offset : offset + header_len].decode("utf-8")
    header = json.loads(header_json)
    offset += header_len
    if header.get("version") != _NWC_VERSION:
        raise ValueError(
            f"Unsupported NWC1 version {header.get('version')} (expected {_NWC_VERSION})"
        )

    codec_blob_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    codec_blob = data[offset : offset + codec_blob_len]
    offset += codec_blob_len

    codec_state = torch.load(io.BytesIO(codec_blob), map_location="cpu", weights_only=False)
    codec_config = WeightCodecConfig(**codec_state["codec_config"])
    codec = WeightCodec(codec_config)
    codec.load_state_dict(codec_state["codec_state_dict"])
    codec.eval()

    new_state: dict[str, torch.Tensor] = {}
    for tensor_meta in header["tensors"]:
        blob_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        blob = data[offset : offset + blob_len]
        offset += blob_len
        decoded = codec.decode(blob)
        # decoder returns float32 — caller may cast to whatever the model expects
        new_state[tensor_meta["name"]] = decoded.reshape(tensor_meta["shape"])

    # Build the model. Reuse the FP8H builder which handles either an inferred
    # asymmetric config or the tensor_only fallback.
    config = header.get("config", {})
    if config.get("tensor_only"):
        # Tensor-only mode: caller supplies their own model and copies state.
        # Mirror the FP8H tensor_only protocol.
        from tac.renderer import AsymmetricPairGenerator

        model = AsymmetricPairGenerator()
        model._nwc_state_dict = new_state  # type: ignore[attr-defined]
        return model.to(device).eval()

    model = _fp8h_build_model_from_header({"config": config, "tensors": []}, device=device)

    full_state = dict(model.state_dict())
    for k, v in new_state.items():
        if k in full_state:
            full_state[k] = v.to(full_state[k].dtype)
        else:
            full_state[k] = v
    model.load_state_dict(full_state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_nwcs_sensitivity_compressed_checkpoint(
    data_or_path: Union[bytes, Path, str],
    device: str = "cpu",
) -> nn.Module:
    """Restore a renderer from an NWCS1 sensitivity-aware NWC container."""
    from tac.neural_weight_codec_sensitivity import (
        SensitivityAwareCodecConfig,
        SensitivityAwareWeightCodec,
        decode_with_per_block_codebook,
        load_nwcs_renderer_container,
    )

    container = load_nwcs_renderer_container(data_or_path)
    codec_state = torch.load(
        io.BytesIO(container.codec_checkpoint_blob),
        map_location="cpu",
        weights_only=False,
    )
    cfg_dict = codec_state.get("config") or codec_state.get("codec_config")
    if cfg_dict is None:
        raise ValueError("NWCS1 codec checkpoint missing config/codec_config")
    codec = SensitivityAwareWeightCodec(SensitivityAwareCodecConfig(**cfg_dict))
    codec.load_state_dict(codec_state["codec_state_dict"])
    codec.eval()

    new_state: dict[str, torch.Tensor] = {}
    for entry in container.tensors:
        decoded = decode_with_per_block_codebook(codec, entry.blob)
        new_state[entry.name] = decoded.reshape(entry.shape)

    metadata = container.header.get("metadata", {})
    config = metadata.get("config") or metadata.get("arch_config") or {"tensor_only": True}
    if config.get("tensor_only"):
        from tac.renderer import AsymmetricPairGenerator

        model = AsymmetricPairGenerator()
        model._nwcs_state_dict = new_state  # type: ignore[attr-defined]
        return model.to(device).eval()

    model = _fp8h_build_model_from_header({"config": config, "tensors": []}, device=device)
    full_state = dict(model.state_dict())
    for key, value in new_state.items():
        if key in full_state:
            full_state[key] = value.to(full_state[key].dtype)
        else:
            full_state[key] = value
    model.load_state_dict(full_state, strict=False)
    return model.to(device).eval()


if __name__ == "__main__":
    _smoke_test()
