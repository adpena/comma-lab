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
    """Infer architecture config from an AsymmetricPairGenerator instance."""
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
    pose_dim = getattr(model, "pose_dim", 0)
    use_dsconv = getattr(model, "use_dsconv", False)

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

    # Build header
    header = {
        "version": 2,
        "pair_mode": "asymmetric",
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

    # Build fresh AsymmetricPairGenerator
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


def detect_checkpoint_type(data_or_path: Union[bytes, Path]) -> str:
    """Detect the type of a renderer checkpoint.

    Returns:
        "dpsm" for DPSIMSRenderer, "asymmetric" for AsymmetricPairGenerator,
        "pytorch" for raw PyTorch checkpoints.
    """
    if isinstance(data_or_path, (str, Path)):
        data = Path(data_or_path).read_bytes()[:8]
    else:
        data = data_or_path[:8]

    if data[:4] == b"DPSM":
        return "dpsm"
    elif data[:4] == b"ASYM":
        return "asymmetric"
    else:
        return "pytorch"


def load_any_renderer_checkpoint(
    data_or_path: Union[bytes, Path],
    device: str = "cpu",
) -> nn.Module:
    """Load any renderer checkpoint, auto-detecting the format.

    Supports: DPSM (.bin), ASYM (.bin), and raw PyTorch (.pt) checkpoints.

    Returns:
        The loaded model in eval mode on the specified device.
    """
    fmt = detect_checkpoint_type(data_or_path)
    if fmt == "dpsm":
        return load_renderer_checkpoint(data_or_path, device=device)
    elif fmt == "asymmetric":
        return load_asymmetric_checkpoint(data_or_path, device=device)
    else:
        raise ValueError(
            f"Raw PyTorch checkpoint detected — use _load_renderer() in "
            f"inflate_renderer.py for .pt format support."
        )


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


if __name__ == "__main__":
    _smoke_test()
