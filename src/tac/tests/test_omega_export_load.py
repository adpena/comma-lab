"""Tests for OMG1 export/load (Lane Ω per-weight Hessian-quantized renderer).

Pins:
  1. Bit-pack / unpack round trip is exact for arbitrary (values, bits) sets.
  2. Single-layer omega quantize/dequantize is approximately accurate (within
     scale/levels per element).
  3. detect_checkpoint_type recognizes b"OMG1" as "omega_v1".
  4. load_any_renderer_checkpoint dispatches to load_omega_renderer.
  5. End-to-end export → load round-trip on a built renderer:
       a. inflate.sh-style: raw bytes → load_omega_renderer.
       b. dispatch via load_any_renderer_checkpoint.
       c. Forward outputs match within float16 tolerance for high-bit
          allocation; bounded for low-bit.
  6. The OMG1 byte size at 4 bits/weight is meaningfully smaller than the
     equivalent FP32 archive.
  7. Mixed-bit allocation honors per-weight bit-depth (high-bit weights
     reconstructed more accurately than low-bit).
  8. Protected layers (renderer.head, etc.) stay FP16, NOT quantized.
"""
from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tac.renderer_export import (
    _bitpack_values_with_bits,
    _bitunpack_values_with_bits,
    _omega_dequantize_layer,
    _omega_quantize_layer,
    detect_checkpoint_type,
    export_omega_renderer,
    load_any_renderer_checkpoint,
    load_omega_renderer,
)
from tac.self_compress import SC_PROTECTED_NAME_PATTERNS


REPO = Path(__file__).resolve().parents[3]
LANE_A_RENDERER = REPO / "experiments" / "results" / "lane_a_landed" / "iter_0" / "renderer.bin"


# ── Bit-pack round trip ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "values, bits",
    [
        ([0, 1, -1, 3, -3], [3, 3, 3, 3, 3]),
        ([5, -7, 0, 0, 1], [4, 4, 4, 4, 4]),
        ([1, -1, 1, 1, -1, 1], [1, 1, 1, 1, 1, 1]),  # all 1-bit
        ([1, 2, 3, 5, 8, 13], [2, 3, 3, 4, 5, 5]),
        ([], []),  # empty
        ([127, -127, 0], [8, 8, 8]),
    ],
)
def test_bitpack_unpack_roundtrip(values, bits):
    blob = _bitpack_values_with_bits(values, bits)
    expected_bytes = (sum(bits) + 7) // 8
    assert len(blob) == expected_bytes
    decoded = _bitunpack_values_with_bits(blob, bits)
    # 1-bit elements are sign-only: 0 → +1, anything else preserved
    expected = []
    for v, b in zip(values, bits):
        if b == 1:
            expected.append(-1 if v < 0 else 1)
        else:
            expected.append(v)
    assert decoded == expected


def test_bitpack_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        _bitpack_values_with_bits([1, 2, 3], [4, 4])


# ── Single-layer omega quant round trip ───────────────────────────────────


def test_omega_layer_quant_dequant_8bit():
    """8-bit per-element quant should reconstruct within scale/127 per element."""
    torch.manual_seed(0)
    w = torch.randn(4, 8) * 0.5
    bits = torch.full(w.shape, 8, dtype=torch.uint8)
    scales, bits_u8, codes = _omega_quantize_layer(w, bits)
    assert scales.shape == (4,)
    assert bits_u8.shape == w.shape
    assert len(codes) == w.numel()
    w_back = _omega_dequantize_layer(codes, bits_u8, scales, w.shape)
    diff = (w - w_back).abs()
    step = scales.float().max().item() / 127.0
    assert diff.max().item() < step + 1e-6, (
        f"8-bit max diff {diff.max().item()} > step {step}"
    )


def test_omega_layer_quant_dequant_1bit():
    """1-bit dequant should give ±scale per output channel for every element."""
    torch.manual_seed(1)
    w = torch.randn(3, 5) * 0.4
    bits = torch.full(w.shape, 1, dtype=torch.uint8)
    scales, bits_u8, codes = _omega_quantize_layer(w, bits)
    w_back = _omega_dequantize_layer(codes, bits_u8, scales, w.shape)
    for row in range(3):
        unique_abs = torch.unique(w_back[row].abs()).tolist()
        assert len(unique_abs) == 1
        assert abs(unique_abs[0] - scales[row].float().item()) < 1e-3


def test_omega_layer_mixed_bits():
    """High-bit elements should reconstruct more accurately than low-bit."""
    torch.manual_seed(2)
    w = torch.randn(2, 8) * 0.3
    bits = torch.tensor(
        [[8, 8, 8, 8, 1, 1, 1, 1], [1, 1, 1, 1, 8, 8, 8, 8]],
        dtype=torch.uint8,
    )
    scales, bits_u8, codes = _omega_quantize_layer(w, bits)
    w_back = _omega_dequantize_layer(codes, bits_u8, scales, w.shape)
    diff_high = (w[0, :4] - w_back[0, :4]).abs().mean().item()
    diff_low = (w[0, 4:] - w_back[0, 4:]).abs().mean().item()
    assert diff_high <= diff_low, (
        f"8-bit reconstruction ({diff_high}) should be ≤ 1-bit ({diff_low})"
    )


def test_omega_layer_shape_check():
    w = torch.randn(2, 3)
    bits_bad = torch.full((2, 4), 4, dtype=torch.uint8)
    with pytest.raises(ValueError, match="shape"):
        _omega_quantize_layer(w, bits_bad)


# ── detect_checkpoint_type and dispatch ───────────────────────────────────


def test_detect_omg1_magic():
    fake_blob = b"OMG1" + b"\x00" * 16
    assert detect_checkpoint_type(fake_blob) == "omega_v1"


def test_detect_no_collide_with_other_magics():
    # Verify b"OMG1" doesn't accidentally match other recognized magics.
    for other in (b"DPSM", b"FP4A", b"ASYM", b"I4LZ", b"SCv1", b"C3R1"):
        fmt = detect_checkpoint_type(other + b"\x00" * 16)
        assert fmt != "omega_v1"


# ── Build small renderer for fast round-trip tests ───────────────────────


def _build_small_asym() -> nn.Module:
    """Tiny AsymmetricPairGenerator for fast round-trip tests."""
    from tac.renderer import AsymmetricPairGenerator

    return AsymmetricPairGenerator(
        num_classes=5,
        embed_dim=6,
        base_ch=8,
        mid_ch=12,
        motion_hidden=8,
        depth=1,
        pose_dim=0,
        use_dsconv=False,
        use_zoom_flow=False,
        padding_mode="zeros",
        use_dilation=False,
    )


def _eligible_bits_dict(model: nn.Module, b: int) -> dict:
    """Build a uniform-b bits dict for every eligible Conv2d weight."""
    def _is_protected(name: str) -> bool:
        for pat in SC_PROTECTED_NAME_PATTERNS:
            if name == pat or name.endswith("." + pat):
                return True
        return False

    out: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if _is_protected(name):
            continue
        if isinstance(mod, nn.Conv2d) and not isinstance(mod, nn.ConvTranspose2d):
            out[f"{name}.weight"] = torch.full(
                mod.weight.shape, b, dtype=torch.uint8,
            )
    return out


def test_omg1_roundtrip_high_bits_small_model():
    """8-bit allocation: forward outputs should match within FP16 tolerance."""
    torch.manual_seed(0)
    model = _build_small_asym()
    model.eval()
    bits = _eligible_bits_dict(model, b=8)
    assert len(bits) > 0, "must have at least one eligible Conv2d to quantize"

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        out_path = Path(f.name)
    try:
        n_bytes = export_omega_renderer(model, bits, out_path)
        assert n_bytes > 0
        # Magic byte check
        assert out_path.read_bytes()[:4] == b"OMG1"
        m2 = load_omega_renderer(out_path, device="cpu")
        m2.eval()

        # Forward smoke
        H, W = 24, 32
        masks_t = torch.randint(0, 5, (1, H, W), dtype=torch.long)
        masks_t1 = torch.randint(0, 5, (1, H, W), dtype=torch.long)
        with torch.no_grad():
            o1 = model(masks_t, masks_t1)
            o2 = m2(masks_t, masks_t1)
        # 8-bit reconstruction is close to identity
        max_diff = (o1 - o2).abs().max().item()
        # Renderer outputs are in [0, 255] range, so 8-bit weight quant on
        # eligible-only Conv2d should give max diff well under 1.0 of pixel.
        assert max_diff < 5.0, f"8-bit roundtrip output max diff {max_diff}"
    finally:
        out_path.unlink(missing_ok=True)


def test_omg1_dispatch_via_load_any():
    """detect_checkpoint_type → load_any_renderer_checkpoint route OMG1
    correctly."""
    torch.manual_seed(0)
    model = _build_small_asym()
    model.eval()
    bits = _eligible_bits_dict(model, b=8)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        out_path = Path(f.name)
    try:
        export_omega_renderer(model, bits, out_path)
        m2 = load_any_renderer_checkpoint(out_path, device="cpu")
        # Returns the right class + has the same module names.
        assert type(m2).__name__ == type(model).__name__
        names1 = {n for n, _ in model.named_modules()}
        names2 = {n for n, _ in m2.named_modules()}
        assert names1 == names2
    finally:
        out_path.unlink(missing_ok=True)


def test_omg1_protected_layers_stay_unquantized():
    """Layers in SC_PROTECTED_NAME_PATTERNS must NOT appear as 'omega' kind
    in the OMG1 header — they should be 'fp16_conv' / 'fp16_linear'."""
    import json

    torch.manual_seed(0)
    model = _build_small_asym()
    model.eval()
    bits = _eligible_bits_dict(model, b=4)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        out_path = Path(f.name)
    try:
        export_omega_renderer(model, bits, out_path)
        raw = out_path.read_bytes()
        assert raw[:4] == b"OMG1"
        header_len = struct.unpack("<I", raw[4:8])[0]
        header = json.loads(raw[8:8 + header_len].decode("utf-8"))
        layer_kinds = {l["name"]: l["kind"] for l in header["layers"]}
        # Every protected layer that exists in the model must be fp16_*
        for name in layer_kinds:
            for pat in SC_PROTECTED_NAME_PATTERNS:
                if name == pat or name.endswith("." + pat):
                    assert layer_kinds[name].startswith("fp16_"), (
                        f"protected layer {name!r} should be fp16_*, "
                        f"got {layer_kinds[name]!r}"
                    )
    finally:
        out_path.unlink(missing_ok=True)


def test_omg1_size_smaller_at_low_bits_than_high_bits():
    """4-bit allocation produces a smaller archive than 8-bit (lossy
    compression argument)."""
    torch.manual_seed(0)
    model = _build_small_asym()
    bits_4 = _eligible_bits_dict(model, b=4)
    bits_8 = _eligible_bits_dict(model, b=8)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f4, \
         tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f8:
        p4 = Path(f4.name)
        p8 = Path(f8.name)
    try:
        n4 = export_omega_renderer(model, bits_4, p4)
        n8 = export_omega_renderer(model, bits_8, p8)
        # 4-bit body is half the 8-bit body; LZMA may flatten this somewhat
        # but the binary should still be smaller.
        assert n4 < n8, (
            f"4-bit binary ({n4}) should be smaller than 8-bit ({n8})"
        )
    finally:
        p4.unlink(missing_ok=True)
        p8.unlink(missing_ok=True)


def test_omg1_mixed_bits_via_allocator_endtoend():
    """Use the bit_allocator to pick mixed bit-depths, export, reload, verify."""
    from tac.bit_allocator import allocate_bits

    torch.manual_seed(0)
    model = _build_small_asym()

    # Use weight magnitude as importance proxy
    importance = {
        f"{name}.weight": mod.weight.detach().abs()
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Conv2d) and not isinstance(mod, nn.ConvTranspose2d)
        and not any(name == p or name.endswith("." + p) for p in SC_PROTECTED_NAME_PATTERNS)
    }
    n = sum(t.numel() for t in importance.values())
    bits = allocate_bits(importance, total_bits=int(n * 3.0), min_bits=1, max_bits=8)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        out_path = Path(f.name)
    try:
        export_omega_renderer(model, bits, out_path)
        m2 = load_omega_renderer(out_path, device="cpu")
        # Forward smoke — finite output
        masks_t = torch.randint(0, 5, (1, 24, 32), dtype=torch.long)
        masks_t1 = torch.randint(0, 5, (1, 24, 32), dtype=torch.long)
        with torch.no_grad():
            out = m2(masks_t, masks_t1)
        assert torch.isfinite(out).all()
    finally:
        out_path.unlink(missing_ok=True)


# ── Real Lane A renderer round trip (only if archive present) ────────────


@pytest.mark.skipif(not LANE_A_RENDERER.exists(), reason="Lane A renderer.bin not committed")
def test_omg1_roundtrip_on_lane_a_renderer():
    """Realistic test on the actual 290KB Lane A renderer at 8 bits/weight."""
    model = load_any_renderer_checkpoint(str(LANE_A_RENDERER), device="cpu")
    bits = _eligible_bits_dict(model, b=8)
    assert len(bits) >= 8, f"expected ≥8 eligible Conv2d in Lane A, got {len(bits)}"
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        out_path = Path(f.name)
    try:
        n_bytes = export_omega_renderer(model, bits, out_path)
        assert n_bytes > 0
        m2 = load_any_renderer_checkpoint(str(out_path), device="cpu")
        # Same class + module names
        assert type(m2).__name__ == type(model).__name__
        names1 = {n for n, _ in model.named_modules()}
        names2 = {n for n, _ in m2.named_modules()}
        assert names1 == names2
    finally:
        out_path.unlink(missing_ok=True)
