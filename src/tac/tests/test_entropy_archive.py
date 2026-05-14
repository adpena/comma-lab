# SPDX-License-Identifier: MIT
"""Roundtrip tests for ``tac.entropy_archive``.

Check 46 contract: every public quantizer/encoder must demonstrate
``decode(encode(x)) ≈ x``. ``entropy_archive.py`` exposes:

- ``compress_byte_stream`` / ``decompress_byte_stream`` — lossless
  arithmetic coder (atol=0).
- ``quantize_pose_targets`` / ``dequantize_pose_targets`` — uniform 256-symbol
  quantizer (atol = (max-min) / 255).
- ``quantize_weights`` / ``dequantize_weights`` — symmetric per-tensor
  256-symbol quantizer (atol = abs_max / 127.5 per tensor).

Each pair gets an explicit-tolerance assertion so a future regression
(e.g., off-by-one in the symbol→float mapping) fails loud.
"""

from __future__ import annotations

import torch

from tac.entropy_archive import (
    compress_byte_stream,
    decompress_byte_stream,
    dequantize_pose_targets,
    dequantize_weights,
    quantize_pose_targets,
    quantize_weights,
)


# ── compress_byte_stream / decompress_byte_stream (LOSSLESS) ──────────────


def test_compress_byte_stream_lossless_roundtrip() -> None:
    """Arithmetic coder must be bit-exact (atol=0)."""
    data = bytes(range(256)) * 4  # 1024 bytes covering full byte range
    compressed = compress_byte_stream(data)
    restored = decompress_byte_stream(compressed)
    assert restored == data, (
        "compress_byte_stream/decompress_byte_stream must be lossless "
        "(arithmetic coding is information-preserving)."
    )


def test_compress_byte_stream_empty_roundtrip() -> None:
    """Edge: empty input."""
    assert decompress_byte_stream(compress_byte_stream(b"")) == b""


def test_compress_byte_stream_skewed_distribution_roundtrip() -> None:
    """Edge: heavily skewed distribution (many zeros) — common for FP4 weight blobs."""
    data = b"\x00" * 1000 + b"\x7f" * 24 + b"\xff" * 8
    restored = decompress_byte_stream(compress_byte_stream(data))
    assert restored == data


# ── quantize_pose_targets / dequantize_pose_targets (UNIFORM 8-bit) ───────


def test_quantize_pose_targets_roundtrip_within_step_tolerance() -> None:
    """8-bit uniform quantizer: atol ≤ (max - min) / 255 / 2."""
    torch.manual_seed(0)
    x = torch.randn(64, 6) * 0.5  # pose targets, typically small range

    symbols, min_val, max_val = quantize_pose_targets(x, num_symbols=256)
    restored = dequantize_pose_targets(symbols, x.shape, min_val, max_val, num_symbols=256)

    assert restored.shape == x.shape
    expected_step = (max_val - min_val) / 255.0
    err = (restored - x).abs().max().item()
    # Reconstruction is at the nearest grid point: max error ≤ 1 step.
    assert err <= expected_step + 1e-6, (
        f"pose-target roundtrip err {err:.6f} > step {expected_step:.6f}"
    )


def test_quantize_pose_targets_constant_input_roundtrip() -> None:
    """Edge: all-zero input (min == max)."""
    x = torch.zeros(8, 6)
    symbols, min_val, max_val = quantize_pose_targets(x, num_symbols=256)
    restored = dequantize_pose_targets(symbols, x.shape, min_val, max_val)
    assert torch.allclose(restored, x, atol=1e-6)


# ── quantize_weights / dequantize_weights (SYMMETRIC PER-TENSOR) ──────────


def test_quantize_weights_roundtrip_within_per_tensor_step() -> None:
    """Per-tensor symmetric 8-bit quantizer: atol per tensor ≤ abs_max/127.5."""
    torch.manual_seed(1)
    state = {
        "layer.weight": torch.randn(8, 4) * 0.3,
        "layer.bias": torch.randn(4) * 0.1,
    }

    symbols, meta = quantize_weights(state, num_symbols=256)
    restored = dequantize_weights(symbols, meta)

    assert set(restored.keys()) == set(state.keys())
    for key, orig in state.items():
        back = restored[key]
        assert back.shape == orig.shape
        # Symmetric quantization: full range is [-abs_max, abs_max], so
        # step = 2 * abs_max / 255. Max reconstruction error ≤ 1 step.
        abs_max = orig.abs().max().item()
        step = 2.0 * abs_max / 255.0
        err = (back - orig).abs().max().item()
        assert err <= step + 1e-6, (
            f"weight-roundtrip err {err:.6f} > step {step:.6f} for {key}"
        )


def test_quantize_weights_zero_tensor_roundtrip() -> None:
    """Edge: all-zero tensor (abs_max underflow path)."""
    state = {"a": torch.zeros(4, 4)}
    symbols, meta = quantize_weights(state)
    restored = dequantize_weights(symbols, meta)
    assert torch.allclose(restored["a"], state["a"], atol=1e-6)
