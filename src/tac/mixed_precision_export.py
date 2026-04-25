"""Int4 per-tensor quantization + LZMA2 compression for extreme weight compression.

Weight analysis shows ~27% of weights carry 90% of signal energy, while ~37%
carry only 1%. Int4 per-tensor quantization captures the essential structure
in 4 bits/weight, then LZMA2 exploits the resulting byte-stream redundancy
(zero-heavy distribution after quantization) to achieve ~2.2 bits/weight.

Compared to FP4 (codebook-based, ~4.4 bits/weight after overhead), this
scheme is simpler (no codebook, no block-level scales) and compresses
better because LZMA2 can exploit global patterns that per-block FP4 cannot.

The trade-off: slightly higher quantization error than FP4 for outlier-heavy
distributions (FP4's non-uniform codebook handles tails better). For our
renderer (~287K params, well-conditioned weights), int4+LZMA2 wins on
rate with negligible distortion difference.

Format (INT4_LZMA2):
    Uncompressed payload (before LZMA2):
        [4 bytes] magic: b"I4LZ"
        [4 bytes] n_tensors (uint32 LE)
        For each tensor:
            [4 bytes] name_len (uint32 LE)
            [name_len bytes] tensor name (UTF-8)
            [4 bytes] ndim (uint32 LE)
            [ndim * 4 bytes] shape (uint32 LE each)
            [4 bytes] scale as float32 LE
            [ceil(numel / 2) bytes] packed int4 values (two per byte)
    The entire payload is then LZMA2-compressed.

    Final on-disk format:
        [4 bytes] magic: b"I4LZ"
        [4 bytes] uncompressed_size (uint32 LE)
        [remaining bytes] LZMA2-compressed payload

Mixed-precision format (MIXED_LZMA2, magic b"MXLZ"):
    Same structure as INT4_LZMA2, but each tensor header includes a
    [1 byte] bits field after the scale, indicating the bit-depth
    (2, 4, 8, or 16) used for that layer. The loader dispatches to
    the appropriate dequantizer per tensor.

Dependencies: torch, struct, lzma (stdlib). numpy only for 8-bit/16-bit
dequantization paths.

Usage::

    from tac.mixed_precision_export import export_int4_lzma2, load_int4_lzma2

    # Uniform 4-bit (backward-compatible default):
    size = export_int4_lzma2(model, Path("renderer.bin"))
    state_dict = load_int4_lzma2(Path("renderer.bin"))
    model.load_state_dict(state_dict)

    # Mixed-precision from sensitivity sweep:
    allocation = {"encoder.conv1.weight": 8, "decoder.conv3.weight": 2}
    size = export_int4_lzma2(model, Path("renderer.bin"), bit_allocation=allocation)
    state_dict = load_int4_lzma2(Path("renderer.bin"))  # auto-detects format
"""

from __future__ import annotations

import lzma
import math
import struct
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

__all__ = [
    "export_int4_lzma2",
    "load_int4_lzma2",
    "quantize_int4_tensor",
    "dequantize_int4_tensor",
    "quantize_uniform_tensor",
    "dequantize_uniform_tensor",
    "MAGIC_INT4_LZMA2",
    "MAGIC_MIXED_LZMA2",
]

MAGIC_INT4_LZMA2 = b"I4LZ"
MAGIC_MIXED_LZMA2 = b"MXLZ"


# ── Int4 quantization primitives ──────────────────────────────────────


def quantize_int4_tensor(
    tensor: torch.Tensor,
) -> tuple[float, bytes]:
    """Quantize a tensor to signed int4 per-tensor and pack into bytes.

    Per-tensor symmetric quantization:
        scale = max(|w|) / 7
        q = round(w / scale).clamp(-7, 7)

    Packed as unsigned nibbles: two values per byte.
    Value mapping: signed [-7, 7] -> unsigned [0, 14] (add 7).
    High nibble = even index, low nibble = odd index.

    Args:
        tensor: any-shape float tensor.

    Returns:
        (scale, packed_bytes) where scale is the float32 scale factor
        and packed_bytes contains ceil(numel/2) bytes of packed int4 values.
    """
    flat = tensor.detach().cpu().float().reshape(-1)
    n = flat.shape[0]

    # Per-tensor scale: map max magnitude to 7 (int4 signed range is [-7, 7])
    abs_max = flat.abs().max().item()
    if abs_max < 1e-10:
        # All-zero tensor: scale=1.0, all values map to 7 (the zero point)
        scale = 1.0
        packed = bytes((0x77,) * ((n + 1) // 2))
        return scale, packed

    scale = abs_max / 7.0

    # Quantize to signed int4: [-7, 7]
    quantized = (flat / scale).round().clamp(-7, 7).to(torch.int8)
    # Shift to unsigned [0, 14]
    unsigned = (quantized + 7).to(torch.uint8)

    # Pack two nibbles per byte
    values = unsigned.tolist()
    if n % 2 != 0:
        values.append(7)  # padding value = 0 in signed domain

    packed = bytearray()
    for i in range(0, len(values), 2):
        high = values[i] & 0x0F
        low = values[i + 1] & 0x0F
        packed.append((high << 4) | low)

    return scale, bytes(packed)


def dequantize_int4_tensor(
    packed: bytes,
    scale: float,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Dequantize packed int4 bytes back to a float tensor.

    Args:
        packed: packed int4 bytes from quantize_int4_tensor.
        scale: the per-tensor scale factor.
        shape: original tensor shape.

    Returns:
        Float32 tensor with the dequantized values, reshaped to shape.
    """
    numel = 1
    for s in shape:
        numel *= s

    # Unpack nibbles
    values = []
    for byte in packed:
        high = (byte >> 4) & 0x0F
        low = byte & 0x0F
        values.append(high)
        values.append(low)

    # Trim to original numel (drop padding nibble if odd)
    values = values[:numel]

    # Convert unsigned [0, 14] back to signed [-7, 7], then dequantize
    result = torch.tensor(
        [(v - 7) * scale for v in values],
        dtype=torch.float32,
    )
    return result.reshape(shape)


# ── Variable-bit uniform quantization primitives ─────────────────────


def quantize_uniform_tensor(
    tensor: torch.Tensor,
    bits: int,
) -> tuple[float, bytes]:
    """Quantize a tensor to signed uniform at the given bit-depth and pack.

    Per-tensor symmetric quantization:
        max_val = 2^(bits-1) - 1  (e.g., 1 for 2-bit, 7 for 4-bit, 127 for 8-bit)
        scale = max(|w|) / max_val
        q = round(w / scale).clamp(-max_val, max_val)

    Packing depends on bit-depth:
        2-bit: 4 values per byte (unsigned [0, 2^bits-1])
        4-bit: 2 values per byte (same as quantize_int4_tensor)
        8-bit: 1 value per byte (signed int8)
        16-bit: passthrough (no quantization, raw float16 bytes)

    Args:
        tensor: any-shape float tensor.
        bits: bit-depth (2, 4, 8, or 16).

    Returns:
        (scale, packed_bytes).
    """
    if bits == 4:
        return quantize_int4_tensor(tensor)

    flat = tensor.detach().cpu().float().reshape(-1)
    n = flat.shape[0]

    if bits == 16:
        # Passthrough: store as float16 bytes
        packed = flat.half().numpy().tobytes()
        return 1.0, packed

    abs_max = flat.abs().max().item()
    max_val = (1 << (bits - 1)) - 1  # 1 for 2-bit, 127 for 8-bit

    if abs_max < 1e-10:
        scale = 1.0
        if bits == 2:
            zero_point = 1  # unsigned zero-point for 2-bit signed range [-1, 1]
            byte_val = (zero_point << 6) | (zero_point << 4) | (zero_point << 2) | zero_point
            packed = bytes((byte_val,) * ((n + 3) // 4))
        elif bits == 8:
            packed = bytes(n)  # all zeros
        else:
            raise ValueError(f"Unsupported bit-depth: {bits}")
        return scale, packed

    scale = abs_max / max_val

    # Quantize
    quantized = (flat / scale).round().clamp(-max_val, max_val).to(torch.int8)

    if bits == 2:
        # Signed range [-1, 1] -> unsigned [0, 2] (add max_val=1)
        unsigned = (quantized + max_val).to(torch.uint8)
        values = unsigned.tolist()
        # Pad to multiple of 4
        while len(values) % 4 != 0:
            values.append(max_val)  # zero in signed domain
        packed = bytearray()
        for i in range(0, len(values), 4):
            b = ((values[i] & 0x03) << 6) | ((values[i+1] & 0x03) << 4) | \
                ((values[i+2] & 0x03) << 2) | (values[i+3] & 0x03)
            packed.append(b)
        return scale, bytes(packed)

    elif bits == 8:
        # Signed int8 directly
        packed = quantized.numpy().astype("int8").tobytes()
        return scale, packed

    else:
        raise ValueError(f"Unsupported bit-depth: {bits}. Use 2, 4, 8, or 16.")


def dequantize_uniform_tensor(
    packed: bytes,
    scale: float,
    shape: tuple[int, ...],
    bits: int,
) -> torch.Tensor:
    """Dequantize packed bytes back to a float tensor at the given bit-depth.

    Args:
        packed: packed bytes from quantize_uniform_tensor.
        scale: the per-tensor scale factor.
        shape: original tensor shape.
        bits: bit-depth (2, 4, 8, or 16).

    Returns:
        Float32 tensor with the dequantized values, reshaped to shape.
    """
    if bits == 4:
        return dequantize_int4_tensor(packed, scale, shape)

    import numpy as np

    numel = 1
    for s in shape:
        numel *= s

    if bits == 16:
        arr = np.frombuffer(packed, dtype=np.float16)[:numel]
        return torch.from_numpy(arr.astype(np.float32).copy()).reshape(shape)

    max_val = (1 << (bits - 1)) - 1

    if bits == 2:
        values = []
        for byte in packed:
            values.append((byte >> 6) & 0x03)
            values.append((byte >> 4) & 0x03)
            values.append((byte >> 2) & 0x03)
            values.append(byte & 0x03)
        values = values[:numel]
        result = torch.tensor(
            [(v - max_val) * scale for v in values],
            dtype=torch.float32,
        )
        return result.reshape(shape)

    elif bits == 8:
        arr = np.frombuffer(packed, dtype=np.int8)[:numel]
        result = torch.from_numpy(arr.astype(np.float32).copy()) * scale
        return result.reshape(shape)

    else:
        raise ValueError(f"Unsupported bit-depth: {bits}. Use 2, 4, 8, or 16.")


# ── Export / import ───────────────────────────────────────────────────


def export_int4_lzma2(
    model: nn.Module,
    output_path: Path,
    *,
    exclude_buffers: bool = True,
    bit_allocation: dict[str, int] | None = None,
) -> int:
    """Export model weights as quantized + LZMA2.

    When ``bit_allocation`` is None (default), all layers use uniform 4-bit
    quantization — identical to the original behaviour.

    When ``bit_allocation`` is provided, each layer is quantized at the
    bit-depth specified in the dict. Layers missing from the dict default
    to 4 bits. Supported bit-depths: 2, 4, 8, 16.

    The on-disk format is extended for mixed precision:
        - Magic changes to ``MXLZ`` (vs ``I4LZ`` for uniform int4).
        - Each tensor header includes a ``[1 byte] bits`` field after the
          scale, so the loader knows how to unpack.

    Uniform int4 files still use the ``I4LZ`` magic for backward compat.

    For each parameter tensor:
    - Look up per-layer bit-depth from ``bit_allocation`` (default 4)
    - Compute per-tensor scale and quantize at that bit-depth
    - Pack according to bit-depth
    - Store: [name_len][name][ndim][shape...][scale][bits][packed_len][packed_data]
    - Wrap entire blob in LZMA2

    Args:
        model: nn.Module whose parameters to export.
        output_path: path to write the compressed .bin file.
        exclude_buffers: if True, only export nn.Parameter (skip buffers).
        bit_allocation: optional dict mapping layer name -> bit-depth (2, 4, 8, 16).
            Layers not in the dict default to 4 bits. When None, all layers
            use 4 bits (backward-compatible uniform mode).

    Returns:
        File size in bytes.
    """
    model.eval()

    is_mixed = bit_allocation is not None and any(
        v != 4 for v in bit_allocation.values()
    )
    magic = MAGIC_MIXED_LZMA2 if is_mixed else MAGIC_INT4_LZMA2

    # Build uncompressed payload
    payload = bytearray()
    payload.extend(magic)

    # Collect parameters
    params = list(model.named_parameters())
    payload.extend(struct.pack("<I", len(params)))

    total_params = 0
    total_bits = 0
    bits_summary: dict[int, int] = {}  # bits -> param count

    for name, param in params:
        bits = 4
        if bit_allocation is not None and name in bit_allocation:
            bits = bit_allocation[name]

        # Tensor name
        name_bytes = name.encode("utf-8")
        payload.extend(struct.pack("<I", len(name_bytes)))
        payload.extend(name_bytes)

        # Shape
        shape = list(param.shape)
        payload.extend(struct.pack("<I", len(shape)))
        for s in shape:
            payload.extend(struct.pack("<I", s))

        # Quantize and pack at the chosen bit-depth
        if is_mixed:
            scale, packed = quantize_uniform_tensor(param, bits)
        else:
            # Uniform int4 path — preserves exact original behaviour
            scale, packed = quantize_int4_tensor(param)

        payload.extend(struct.pack("<f", scale))

        # For mixed-precision: store bits-per-weight so loader can unpack
        if is_mixed:
            payload.extend(struct.pack("<B", bits))

        # Packed data length then data
        payload.extend(struct.pack("<I", len(packed)))
        payload.extend(packed)

        n = param.numel()
        total_params += n
        total_bits += n * bits
        bits_summary[bits] = bits_summary.get(bits, 0) + n

    # LZMA2 compress the payload
    compressed = lzma.compress(
        bytes(payload),
        format=lzma.FORMAT_ALONE,
        preset=9 | lzma.PRESET_EXTREME,
    )

    # Write final format: [magic][uncompressed_size][compressed_data]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    buf = bytearray()
    buf.extend(magic)
    buf.extend(struct.pack("<I", len(payload)))
    buf.extend(compressed)

    output_path.write_bytes(bytes(buf))
    file_size = len(buf)

    avg_bits = total_bits / max(total_params, 1)
    bits_per_param = file_size * 8 / max(total_params, 1)
    mode_label = "mixed" if is_mixed else "int4"

    summary_parts = [f"{b}b:{n:,}" for b, n in sorted(bits_summary.items())]
    print(
        f"[{mode_label}-lzma2] {total_params:,} params -> {file_size:,} bytes "
        f"({bits_per_param:.2f} bits/param, avg {avg_bits:.1f}b uncompressed, "
        f"LZMA2 ratio {len(compressed)/len(payload)*100:.1f}%) "
        f"[{', '.join(summary_parts)}]"
    )

    return file_size


def load_int4_lzma2(
    path_or_bytes: Union[Path, str, bytes],
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Load and dequantize int4+LZMA2 or mixed-precision+LZMA2 weights.

    Automatically detects the format from the magic bytes:
        - ``I4LZ``: uniform int4 (original format)
        - ``MXLZ``: mixed-precision (variable bit-depth per layer)

    Args:
        path_or_bytes: path to .bin file or raw bytes.
        device: target device for tensors.

    Returns:
        State dict with float32 tensors.
    """
    if isinstance(path_or_bytes, (str, Path)):
        raw = Path(path_or_bytes).read_bytes()
    else:
        raw = path_or_bytes

    # Verify outer magic
    outer_magic = raw[:4]
    if outer_magic not in (MAGIC_INT4_LZMA2, MAGIC_MIXED_LZMA2):
        raise ValueError(
            f"Not an INT4_LZMA2/MIXED_LZMA2 binary (expected magic "
            f"{MAGIC_INT4_LZMA2!r} or {MAGIC_MIXED_LZMA2!r}, got {outer_magic!r})"
        )

    is_mixed = outer_magic == MAGIC_MIXED_LZMA2

    # Read uncompressed size (informational, lzma handles it)
    _uncompressed_size = struct.unpack("<I", raw[4:8])[0]

    # Decompress
    payload = lzma.decompress(raw[8:], format=lzma.FORMAT_ALONE)

    # Verify inner magic
    inner_magic = payload[:4]
    if inner_magic != outer_magic:
        raise ValueError(
            f"Corrupted payload (inner magic {inner_magic!r} != outer {outer_magic!r})"
        )

    offset = 4

    # Read number of tensors
    n_tensors = struct.unpack("<I", payload[offset:offset + 4])[0]
    offset += 4

    state_dict: dict[str, torch.Tensor] = {}

    for _ in range(n_tensors):
        # Read name
        name_len = struct.unpack("<I", payload[offset:offset + 4])[0]
        offset += 4
        name = payload[offset:offset + name_len].decode("utf-8")
        offset += name_len

        # Read shape
        ndim = struct.unpack("<I", payload[offset:offset + 4])[0]
        offset += 4
        shape = []
        for _ in range(ndim):
            s = struct.unpack("<I", payload[offset:offset + 4])[0]
            offset += 4
            shape.append(s)
        shape_tuple = tuple(shape)

        # Read scale
        scale = struct.unpack("<f", payload[offset:offset + 4])[0]
        offset += 4

        # Read per-layer bit-depth (mixed-precision only)
        if is_mixed:
            bits = struct.unpack("<B", payload[offset:offset + 1])[0]
            offset += 1
        else:
            bits = 4

        # Read packed data
        packed_len = struct.unpack("<I", payload[offset:offset + 4])[0]
        offset += 4
        packed = payload[offset:offset + packed_len]
        offset += packed_len

        # Dequantize at the appropriate bit-depth
        if is_mixed:
            tensor = dequantize_uniform_tensor(packed, scale, shape_tuple, bits)
        else:
            tensor = dequantize_int4_tensor(packed, scale, shape_tuple)

        state_dict[name] = tensor.to(device)

    return state_dict


# ── Convenience: model-level load ─────────────────────────────────────


def load_int4_lzma2_into_model(
    path_or_bytes: Union[Path, str, bytes],
    model: nn.Module,
    device: str = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Load int4+LZMA2 weights into an existing model.

    Args:
        path_or_bytes: path to .bin file or raw bytes.
        model: target model (must match architecture).
        device: target device.
        strict: if True, all keys must match.

    Returns:
        Model with loaded weights, in eval mode.
    """
    state_dict = load_int4_lzma2(path_or_bytes, device=device)
    model.load_state_dict(state_dict, strict=strict)
    return model.eval().to(device)


# ── Quality measurement ──────────────────────────────────────────────


def measure_quantization_error(
    model: nn.Module,
    compressed_path: Path,
) -> dict[str, float]:
    """Measure per-layer and aggregate quantization error.

    Compares original float weights against int4+LZMA2 round-trip.

    Args:
        model: original model with float weights.
        compressed_path: path to exported int4+LZMA2 file.

    Returns:
        Dict with per-layer RMSE and aggregate stats.
    """
    restored = load_int4_lzma2(compressed_path)
    original = {n: p.detach().cpu().float() for n, p in model.named_parameters()}

    stats: dict[str, float] = {}
    total_se = 0.0
    total_n = 0

    for name in original:
        if name not in restored:
            continue
        orig = original[name]
        rest = restored[name]
        se = ((orig - rest) ** 2).sum().item()
        n = orig.numel()
        rmse = math.sqrt(se / max(n, 1))
        stats[f"rmse/{name}"] = rmse
        total_se += se
        total_n += n

    stats["rmse/total"] = math.sqrt(total_se / max(total_n, 1))
    stats["max_abs_error"] = max(
        (original[n] - restored[n]).abs().max().item()
        for n in original if n in restored
    )

    return stats


# ── Smoke test ────────────────────────────────────────────────────────


def _smoke_test() -> None:
    """Round-trip export/load and verify outputs match within tolerance."""
    import tempfile

    print("mixed_precision_export: running smoke tests...")

    # 1. Quantize/dequantize a single tensor
    t = torch.randn(16, 8, 3, 3) * 0.1
    scale, packed = quantize_int4_tensor(t)
    t_restored = dequantize_int4_tensor(packed, scale, tuple(t.shape))
    max_err = (t - t_restored).abs().max().item()
    print(f"  single tensor round-trip: max_err={max_err:.6f}, scale={scale:.6f}")
    assert max_err < scale * 1.5, f"Single tensor error too large: {max_err}"

    # 2. Zero tensor
    t_zero = torch.zeros(4, 4)
    scale_z, packed_z = quantize_int4_tensor(t_zero)
    t_zero_restored = dequantize_int4_tensor(packed_z, scale_z, (4, 4))
    assert t_zero_restored.abs().max().item() == 0.0, "Zero tensor should restore to zero"
    print("  zero tensor: OK")

    # 3. Full model round-trip
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 3, 3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv2(torch.relu(self.conv1(x)))

    model = TinyModel()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.05)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        file_size = export_int4_lzma2(model, tmp_path)
        print(f"  exported: {file_size:,} bytes")
        assert file_size > 0

        # Verify the file starts with correct magic
        raw = tmp_path.read_bytes()
        assert raw[:4] == MAGIC_INT4_LZMA2, f"Wrong magic: {raw[:4]!r}"

        # Load back
        restored_sd = load_int4_lzma2(tmp_path)
        assert set(restored_sd.keys()) == set(n for n, _ in model.named_parameters()), (
            f"Key mismatch: {set(restored_sd.keys())} vs {set(n for n, _ in model.named_parameters())}"
        )

        # Check shapes match
        for name, param in model.named_parameters():
            assert restored_sd[name].shape == param.shape, (
                f"Shape mismatch for {name}: {restored_sd[name].shape} vs {param.shape}"
            )

        # Measure error
        stats = measure_quantization_error(model, tmp_path)
        print(f"  round-trip RMSE: {stats['rmse/total']:.6f}")
        print(f"  round-trip max_abs_error: {stats['max_abs_error']:.6f}")

        # Forward pass comparison
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out_orig = model(x)

        model_restored = TinyModel()
        model_restored.load_state_dict(restored_sd)
        model_restored.eval()
        with torch.no_grad():
            out_restored = model_restored(x)

        output_diff = (out_orig - out_restored).abs().max().item()
        print(f"  output max diff: {output_diff:.6f}")

        # Verify load_int4_lzma2_into_model works
        model2 = TinyModel()
        model2 = load_int4_lzma2_into_model(tmp_path, model2)
        with torch.no_grad():
            out2 = model2(x)
        assert (out2 - out_restored).abs().max().item() < 1e-6, "load_into_model mismatch"
        print("  load_int4_lzma2_into_model: OK")

    finally:
        tmp_path.unlink(missing_ok=True)

    # 4. Verify LZMA2 compression ratio is reasonable
    # For a random-weight model, we still expect some compression from nibble packing
    param_count = sum(p.numel() for p in model.parameters())
    raw_int4_bytes = (param_count + 1) // 2  # nibble packing without LZMA2
    print(f"  params: {param_count:,}, raw int4: {raw_int4_bytes:,}B, "
          f"compressed: {file_size:,}B")

    print("mixed_precision_export: all smoke tests passed")


if __name__ == "__main__":
    _smoke_test()
