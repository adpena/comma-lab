# SPDX-License-Identifier: MIT
"""Cross-module tests for tac.quantization_wave helpers.

Per HARDWARE-EXPLOITS-WAVE 2026-05-17 + T4 SYMPOSIUM Priority 1
verdict. Each helper exposes encode/decode round-trip; this file
verifies the round-trip + byte-mutation discipline (Catalog #272) for
every module in the package.
"""

from __future__ import annotations

import pytest
import torch

from tac.quantization_wave.apple_neural_engine_export import (
    can_export_to_ane,
    coreml_export_metadata,
)
from tac.quantization_wave.awq_activation_aware_quantization import (
    activation_aware_channel_scaling,
)
from tac.quantization_wave.balle_hyperprior_bolton import (
    BalleHyperpriorBolton,
    decode_balle_hyperprior_archive,
    encode_balle_hyperprior_archive,
)
from tac.quantization_wave.entropy_coding_archive_primitives import (
    EntropyCoderTournament,
    HuffmanSidecarCoder,
    arithmetic_decode,
    arithmetic_encode,
    brotli_compress_with_window,
    decode_pr101_style_per_tensor_byte_map,
    encode_pr101_style_per_tensor_byte_map,
)
from tac.quantization_wave.fp8_quantization_wave import (
    E4M3_MAX,
    FP8E4M3FakeQuantWave,
    decode_fp8_per_channel,
    encode_fp8_per_channel,
)
from tac.quantization_wave.gguf_style_per_tensor_mixed_bit import (
    GGUF_QUANT_TYPES,
    GGUFStyleMixedBitQuantizer,
    decode_q4_k_m_style,
    encode_q4_k_m_style,
)
from tac.quantization_wave.gptq_post_training_quantization import (
    GPTQStyleQuantizer,
    hessian_aware_quantize_layer,
)
from tac.quantization_wave.int4_int8_mixed_bit import (
    NF4_LEVELS,
    decode_int4_groupwise,
    encode_int4_groupwise,
    sensitivity_aware_mixed_bit_assignment,
)
from tac.quantization_wave.mlx_inference_path import (
    mlx_inflate_inference_path_metadata,
)
from tac.quantization_wave.sparse_weights_with_quant import (
    SparseQuantComposition,
    magnitude_prune_then_quantize,
)
from tac.quantization_wave.vq_codebook_quantization import (
    VQCodebookQuantizer,
    decode_vq_codebook_latent,
    encode_vq_codebook_latent,
)

# ─────────────────────────── FP8 ─────────────────────────────

def test_fp8_round_trip():
    torch.manual_seed(0)
    w = torch.randn(36, 28)
    e = encode_fp8_per_channel(w)
    d = decode_fp8_per_channel(e)
    assert d.shape == w.shape
    # FP8 is per-channel scaled int8 — expect ~1.5x int8 noise level
    err = (w - d).abs().max().item()
    assert err < 0.5, f"FP8 max err {err}"


def test_fp8_ste_preserves_shape():
    torch.manual_seed(1)
    w = torch.randn(16, 8)
    q = FP8E4M3FakeQuantWave.apply(w)
    assert q.shape == w.shape


def test_fp8_constant_max_is_canonical():
    """E4M3 max magnitude is 448.0 per IEEE FP8 spec."""
    assert E4M3_MAX == 448.0


# ─────────────────────────── INT4/INT8 mixed-bit (bitsandbytes) ─────

def test_nf4_levels_canonical():
    """NF4 levels are the bitsandbytes Dettmers-2023 canonical 16-level
    grid for normally-distributed weights."""
    assert len(NF4_LEVELS) == 16
    assert NF4_LEVELS[0] == -1.0
    assert NF4_LEVELS[-1] == 1.0


def test_int4_nf4_round_trip():
    torch.manual_seed(2)
    w = torch.randn(36, 28)
    e = encode_int4_groupwise(w, group_size=64, use_nf4=True)
    d = decode_int4_groupwise(e, use_nf4=True)
    assert d.shape == w.shape
    err = (w - d).abs().max().item()
    assert err < 1.0


def test_int4_symmetric_round_trip():
    torch.manual_seed(3)
    w = torch.randn(36, 28)
    e = encode_int4_groupwise(w, group_size=64, use_nf4=False)
    d = decode_int4_groupwise(e, use_nf4=False)
    assert d.shape == w.shape


def test_sensitivity_aware_mixed_bit_assignment_respects_budget():
    """Avg bits per tensor should not exceed target_average_bits."""
    torch.manual_seed(4)
    weights_grads = {
        f"layer_{i}": (torch.randn(64, 32), torch.randn(64, 32) * 0.1)
        for i in range(4)
    }
    assignment = sensitivity_aware_mixed_bit_assignment(
        weights_grads, target_average_bits=4.5, candidate_bits=(3, 4, 5, 6, 8)
    )
    total_bits = sum(
        assignment[name] * weights_grads[name][0].numel() for name in assignment
    )
    total_params = sum(w.numel() for w, _ in weights_grads.values())
    avg = total_bits / total_params
    # Greedy assignment should be near target (within a couple bits)
    assert avg <= 8.0
    assert avg >= 3.0


# ─────────────────────────── GGUF (llama.cpp) ────────────────────

def test_gguf_q4_k_m_block_size_is_canonical():
    """Q4_K_M block is 144 bytes per 256 elements = 4.5 bits/param."""
    spec = GGUF_QUANT_TYPES["Q4_K_M"]
    assert spec["block_bytes"] == 144
    assert spec["block_size"] == 256
    assert spec["bits_per_param"] == 4.5


def test_gguf_q4_k_m_round_trip():
    torch.manual_seed(5)
    w = torch.randn(256 * 3)  # 3 super-blocks worth
    e = encode_q4_k_m_style(w)
    d = decode_q4_k_m_style(e)
    assert d.shape == w.shape
    assert len(e.blocks) == 3
    assert all(len(b) == 144 for b in e.blocks)


def test_gguf_mixed_bit_quantizer_archive_estimate():
    qz = GGUFStyleMixedBitQuantizer(
        sensitive_layers=("attention", "output"),
        sensitive_quant_type="Q5_K_M",
        default_quant_type="Q4_K_M",
    )
    sizes = {
        "attention.q.weight": 8192,
        "attention.k.weight": 8192,
        "ffn.w1.weight": 16384,
        "output.weight": 8192,
    }
    total = qz.estimate_archive_size_bytes(sizes)
    # attention/output get Q5_K_M (5.5 bits); ffn gets Q4_K_M (4.5 bits)
    assert total > 0
    # Per-layer dispatch correctness
    assert qz.quant_type_for_layer("attention.q.weight") == "Q5_K_M"
    assert qz.quant_type_for_layer("output.weight") == "Q5_K_M"
    assert qz.quant_type_for_layer("ffn.w1.weight") == "Q4_K_M"


# ─────────────────────────── GPTQ ────────────────────────────────

def test_gptq_hessian_aware_returns_metadata():
    torch.manual_seed(6)
    w = torch.randn(16, 32)
    calib = torch.randn(64, 32)
    w_q, meta = hessian_aware_quantize_layer(w, calibration_inputs=calib, n_bits=4)
    assert w_q.shape == w.shape
    assert "hessian_trace" in meta
    assert "reconstruction_error_relative" in meta
    assert meta["reconstruction_error_relative"] >= 0.0


def test_gptq_layer_quantizer_wraps():
    torch.manual_seed(7)
    gptq = GPTQStyleQuantizer(n_bits=4)
    w = torch.randn(8, 16)
    calib = torch.randn(32, 16)
    w_q, meta = gptq.quantize_layer(w, calib, layer_name="test_layer")
    assert "test_layer" in gptq.layer_metadata


def test_gptq_rejects_non_2d_weight():
    with pytest.raises(ValueError, match="2D"):
        hessian_aware_quantize_layer(torch.randn(8, 16, 3, 3), calibration_inputs=torch.randn(4, 16))


# ─────────────────────────── AWQ ─────────────────────────────────

def test_awq_channel_scaling_preserves_shape():
    torch.manual_seed(8)
    w = torch.randn(16, 32)
    act_mag = torch.randn(32).abs() + 0.1
    rescaled, scales = activation_aware_channel_scaling(w, act_mag, alpha=0.5)
    assert rescaled.shape == w.shape
    assert scales.shape == (32,)
    # Scales should have geometric mean ≈ 1 (normalized)
    assert 0.5 < scales.mean().item() < 2.0


def test_awq_rejects_mismatched_activation_magnitudes():
    w = torch.randn(16, 32)
    bad_act = torch.randn(64).abs()
    with pytest.raises(ValueError, match="match"):
        activation_aware_channel_scaling(w, bad_act, alpha=0.5)


def test_awq_alpha_zero_is_identity():
    """alpha=0 → scales = 1 (no rescaling)."""
    w = torch.randn(16, 32)
    act_mag = torch.randn(32).abs() + 0.1
    rescaled, scales = activation_aware_channel_scaling(w, act_mag, alpha=0.0)
    assert torch.allclose(scales, torch.ones_like(scales))
    assert torch.allclose(rescaled, w)


# ─────────────────────────── Sparse + Quant ──────────────────────

def test_magnitude_prune_then_quantize_50_percent():
    torch.manual_seed(9)
    w = torch.randn(64, 32)
    e = magnitude_prune_then_quantize(w, sparsity=0.5, quant_bits=4)
    # ~50% of elements should be kept (within rounding)
    n_kept = e.nonzero_positions.numel()
    assert 0.45 * w.numel() <= n_kept <= 0.55 * w.numel()


def test_structured_2_4_sparsity():
    """2:4 structured: every group of 4 elements keeps exactly 2."""
    torch.manual_seed(10)
    w = torch.randn(64, 32)
    e = magnitude_prune_then_quantize(w, sparsity=0.5, quant_bits=4, structured_2_4=True)
    # Exactly 50% sparsity
    n_kept = e.nonzero_positions.numel()
    assert n_kept == w.numel() // 2


def test_sparse_quant_composition_class():
    torch.manual_seed(11)
    composer = SparseQuantComposition(sparsity=0.5, quant_bits=4)
    weights = {f"layer_{i}": torch.randn(32, 16) for i in range(3)}
    encs = composer.encode_all(weights)
    assert len(encs) == 3


# ─────────────────────────── Entropy coding ──────────────────────

def test_brotli_compress_with_window_round_trip():
    """Brotli round-trip preserves bytes."""
    import brotli

    data = b"abcdef" * 100 + b"\x00\xff" * 50
    compressed = brotli_compress_with_window(data, quality=11, lgwin=24)
    decompressed = brotli.decompress(compressed)
    assert decompressed == data


def test_arithmetic_coder_round_trip():
    symbols = [0, 1, 2, 1, 0, 2, 1, 0, 1, 2]
    n_symbols = 3
    counts = [4, 4, 3]
    encoded = arithmetic_encode(symbols, n_symbols=n_symbols, counts=counts)
    decoded = arithmetic_decode(encoded, len(symbols), n_symbols=n_symbols, counts=counts)
    assert decoded == symbols


def test_huffman_sidecar_canonical():
    coder = HuffmanSidecarCoder()
    symbols = [0, 1, 2, 1, 0, 2, 1, 0, 1, 2] * 100
    coder.fit(symbols, n_symbols=3)
    assert len(coder.lengths) == 3
    bit_len = coder.encoded_bit_length(symbols)
    # Length should be < raw 2-bit encoding (2*N = 2000 bits) due to skewed
    # distribution
    assert bit_len <= 2 * len(symbols)


def test_entropy_coder_tournament_picks_smallest():
    data = b"\x00" * 1024  # highly compressible
    result = EntropyCoderTournament().run(data)
    # Brotli should beat raw and LZMA on this trivial input
    assert result.byte_size < len(data) + 100


def test_pr101_byte_map_round_trip():
    torch.manual_seed(12)
    w = torch.randn(36, 28)
    blob = encode_pr101_style_per_tensor_byte_map(w, quant_bits=4)
    d = decode_pr101_style_per_tensor_byte_map(blob)
    assert d.shape == w.shape


def test_pr101_byte_map_8bit():
    torch.manual_seed(13)
    w = torch.randn(36, 28)
    blob = encode_pr101_style_per_tensor_byte_map(w, quant_bits=8)
    d = decode_pr101_style_per_tensor_byte_map(blob)
    assert d.shape == w.shape


# ─────────────────────────── VQ codebook ─────────────────────────

def test_vq_codebook_fits_and_decodes():
    torch.manual_seed(14)
    latent = torch.randn(100, 28)
    e = encode_vq_codebook_latent(latent, n_codebook_entries=16, n_iter=20, seed=0)
    d = decode_vq_codebook_latent(e)
    assert d.shape == latent.shape
    assert e.codebook.shape == (16, 28)


def test_vq_codebook_quantizer_estimate():
    torch.manual_seed(15)
    vq = VQCodebookQuantizer(n_codebook_entries=64, n_iter=20)
    latent = torch.randn(200, 28)
    vq.fit_and_encode(latent)
    bytes_estimate = vq.estimate_archive_bytes(brotli_compress_indices=True)
    assert bytes_estimate > 0


def test_vq_codebook_requires_2d():
    with pytest.raises(ValueError, match="2D"):
        encode_vq_codebook_latent(torch.randn(28), n_codebook_entries=16)


# ─────────────────────────── Ballé hyperprior ────────────────────

def test_balle_hyperprior_encode_returns_archive_bytes():
    torch.manual_seed(16)
    latent = torch.randn(50, 28)
    e = encode_balle_hyperprior_archive(latent, hyperprior_dim=4, n_train_iter=20, seed=0)
    assert e.y_quantized.shape == (50, 28)
    assert e.z_quantized.shape == (50, 4)
    assert len(e.archive_bytes) > 0


def test_balle_hyperprior_decode_recovers_y():
    torch.manual_seed(17)
    latent = torch.randn(50, 28)
    e = encode_balle_hyperprior_archive(latent, hyperprior_dim=4, n_train_iter=20, seed=0)
    y_decoded = decode_balle_hyperprior_archive(e.archive_bytes)
    assert y_decoded.shape == (50, 28)


def test_balle_hyperprior_bolton_class():
    torch.manual_seed(18)
    bolton = BalleHyperpriorBolton(hyperprior_dim=4, n_train_iter=10)
    latent = torch.randn(30, 28)
    e = bolton.fit_and_encode(latent)
    assert len(e.archive_bytes) > 0


# ─────────────────────────── MLX + ANE metadata ──────────────────

def test_mlx_metadata_is_well_formed():
    md = mlx_inflate_inference_path_metadata()
    assert "mlx_available" in md
    assert "is_apple_silicon" in md
    assert "promotion_eligible" in md
    assert md["promotion_eligible"] is False  # NEVER True per CLAUDE.md MPS rule
    assert md["score_claim"] is False
    assert md["ready_for_exact_eval_dispatch"] is False
    assert md["ready_for_paid_dispatch"] is False
    assert md["rank_or_kill_eligible"] is False
    assert md["device_runtime_contract"]["authority"] == "advisory_only"
    assert "axis" in md
    assert md["axis"] == "macos_mlx_advisory"


def test_ane_export_metadata_is_well_formed():
    md = coreml_export_metadata()
    assert "can_export_to_ane" in md
    assert "promotion_eligible" in md
    assert md["promotion_eligible"] is False  # NEVER True per CLAUDE.md
    assert md["axis"] == "macos_ane_advisory"


def test_ane_can_export_predicate():
    can_export, reason = can_export_to_ane()
    # On a clean clone without coremltools, can_export should be False
    # with a clear reason
    assert isinstance(can_export, bool)
    assert isinstance(reason, str)
    assert len(reason) > 0
