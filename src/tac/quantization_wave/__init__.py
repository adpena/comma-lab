# SPDX-License-Identifier: MIT
"""HARDWARE-EXPLOITS-WAVE quantization package.

Per operator directive 2026-05-17 + T4 SYMPOSIUM verdict
(commit ``d53ad33ed``) Priority 1 BOLT-ON-on-A1 wave: this package
channels the hardware/quantization community lessons (Apple/MLX +
Hugging Face bitsandbytes/GPTQ/AWQ + ggerganov GGUF + NVIDIA tensor
cores + Quantizr's 0.33 [contest-CUDA] FP4 winning pattern) into the
substrate corpus as a CANONICAL bolt-on layer.

The package is the **canonical home** for hardware/quantization helpers
that:

1. Take a verified PR95-paradigm substrate at the frontier (currently
   A1 at 0.19284758 [contest-CPU] + 0.22635 [contest-CUDA]) as Stage 1.
2. Add a Stage 2 bolt-on that exploits hardware-specific quantization
   primitives (FP4 / FP8 / INT4 / mixed-bit / GGUF-style / VQ codebook /
   Ballé hyperprior / entropy stack) to push the score below A1.
3. Honor the contest's "all bytes inside ``archive.zip``" non-negotiable
   per CLAUDE.md FORBIDDEN_PATTERNS (no scorers at inflate time).
4. Tag every score by axis per the apples-to-apples evidence discipline.

Per the T4 SYMPOSIUM verdict the Priority 1 BOLT-ONs are:

* **BOLT-ON #1**: Ballé-2018 hyperprior on A1 per-pair latent
  (per Ballé voice; predicted band [0.188, 0.192] frontier_breaking)
* **BOLT-ON #2**: PR101-style per-tensor byte-map + Brotli/LZMA +
  Huffman sidecar on A1 weights
  (per Selfcomp / Quantizr / MacKay voice; predicted band
  [0.188, 0.192] frontier_breaking)
* **BOLT-ON #3**: VQ-codebook on A1 per-pair latent
  (per van den Oord voice; predicted band [0.188, 0.192]
  frontier_breaking)

Each helper in this package:

* declares **target_modes** (which substrate slots it composes with)
* declares **archive_grammar_contribution** (what wire-format bytes it
  emits) per CLAUDE.md HNeRV parity discipline lesson 3
* declares **inflate_runtime_loc_budget** per HNeRV parity lesson 4
* exposes a typed **encode()/decode()** round-trip so the no-op detector
  (Catalog #105 / #139 / #220 / #272) can verify the bytes actually
  affect the score
* provides a **byte_mutation_smoke()** helper that mutates a single
  byte of the encoded blob and asserts the decoded output changes

Lane: ``lane_hardware_exploits_wave_20260517``.
Memory: ``feedback_hardware_exploits_wave_landed_20260517.md`` +
``.omx/research/hardware_exploits_design_and_implementation_landed_20260517.md``.

[verified-against:Quantizr 0.33 [contest-CUDA] anchor + PR101 GOLD
0.193 [contest-CUDA] anchor + leaderboard cluster 0.193-0.197
[contest-CPU]]
"""

from __future__ import annotations

from tac.quantization_wave.fp4_quantization_wave import (
    DEFAULT_FP4_LEVELS,
    FP4_NEG_LEVELS,
    FakeQuantFP4,
    QuantizrFP4Quantizer,
    encode_fp4_per_channel,
    fake_quant_fp4,
    decode_fp4_per_channel,
    QUANTIZR_FP4_LEVELS_E2M1,
)
from tac.quantization_wave.fp8_quantization_wave import (
    FP8E4M3FakeQuantWave,
    encode_fp8_per_channel,
    decode_fp8_per_channel,
)
from tac.quantization_wave.int4_int8_mixed_bit import (
    BitsAndBytesStyleQuantizer,
    encode_int4_groupwise,
    decode_int4_groupwise,
    sensitivity_aware_mixed_bit_assignment,
)
from tac.quantization_wave.gguf_style_per_tensor_mixed_bit import (
    GGUF_QUANT_TYPES,
    GGUFStyleMixedBitQuantizer,
    encode_q4_k_m_style,
    decode_q4_k_m_style,
)
from tac.quantization_wave.gptq_post_training_quantization import (
    GPTQStyleQuantizer,
    hessian_aware_quantize_layer,
)
from tac.quantization_wave.awq_activation_aware_quantization import (
    AWQStyleQuantizer,
    activation_aware_channel_scaling,
)
from tac.quantization_wave.sparse_weights_with_quant import (
    SparseQuantComposition,
    magnitude_prune_then_quantize,
)
from tac.quantization_wave.entropy_coding_archive_primitives import (
    EntropyCoderTournament,
    HuffmanSidecarCoder,
    arithmetic_encode,
    arithmetic_decode,
    brotli_compress_with_window,
    encode_pr101_style_per_tensor_byte_map,
    decode_pr101_style_per_tensor_byte_map,
)
from tac.quantization_wave.vq_codebook_quantization import (
    VQCodebookQuantizer,
    encode_vq_codebook_latent,
    decode_vq_codebook_latent,
)
from tac.quantization_wave.balle_hyperprior_bolton import (
    BalleHyperpriorBolton,
    encode_balle_hyperprior_archive,
    decode_balle_hyperprior_archive,
)
from tac.quantization_wave.mlx_inference_path import (
    MLX_AVAILABLE,
    convert_pytorch_to_mlx_4bit,
    mlx_inflate_inference_path_metadata,
)
from tac.quantization_wave.apple_neural_engine_export import (
    coreml_export_metadata,
    can_export_to_ane,
)

IMPLEMENTED_MODULES: tuple[str, ...] = (
    "apple_neural_engine_export",
    "awq_activation_aware_quantization",
    "balle_hyperprior_bolton",
    "entropy_coding_archive_primitives",
    "fp4_quantization_wave",
    "fp8_quantization_wave",
    "gguf_style_per_tensor_mixed_bit",
    "gptq_post_training_quantization",
    "int4_int8_mixed_bit",
    "mlx_inference_path",
    "sparse_weights_with_quant",
    "vq_codebook_quantization",
)

DEFERRED_MODULES: tuple[str, ...] = ()

DEFERRED_RATIONALE = (
    "No module-level deferrals remain in the 2026-05-17 quantization wave "
    "package. Score claims remain disabled until concrete archive candidates "
    "produce matched scorer-response and exact-eval evidence."
)

__all__ = [
    # FP4 (Quantizr canonical)
    "DEFAULT_FP4_LEVELS",
    "FP4_NEG_LEVELS",
    "FakeQuantFP4",
    "QuantizrFP4Quantizer",
    "encode_fp4_per_channel",
    "fake_quant_fp4",
    "decode_fp4_per_channel",
    "QUANTIZR_FP4_LEVELS_E2M1",
    # FP8 (transformer engine)
    "FP8E4M3FakeQuantWave",
    "encode_fp8_per_channel",
    "decode_fp8_per_channel",
    # INT4/INT8 mixed-bit (bitsandbytes)
    "BitsAndBytesStyleQuantizer",
    "encode_int4_groupwise",
    "decode_int4_groupwise",
    "sensitivity_aware_mixed_bit_assignment",
    # GGUF (llama.cpp)
    "GGUF_QUANT_TYPES",
    "GGUFStyleMixedBitQuantizer",
    "encode_q4_k_m_style",
    "decode_q4_k_m_style",
    # GPTQ
    "GPTQStyleQuantizer",
    "hessian_aware_quantize_layer",
    # AWQ
    "AWQStyleQuantizer",
    "activation_aware_channel_scaling",
    # Sparse + quant
    "SparseQuantComposition",
    "magnitude_prune_then_quantize",
    # Entropy coding (PR101 + Ballé primitives)
    "EntropyCoderTournament",
    "HuffmanSidecarCoder",
    "arithmetic_encode",
    "arithmetic_decode",
    "brotli_compress_with_window",
    "encode_pr101_style_per_tensor_byte_map",
    "decode_pr101_style_per_tensor_byte_map",
    # VQ codebook (van den Oord)
    "VQCodebookQuantizer",
    "encode_vq_codebook_latent",
    "decode_vq_codebook_latent",
    # Ballé 2018 hyperprior
    "BalleHyperpriorBolton",
    "encode_balle_hyperprior_archive",
    "decode_balle_hyperprior_archive",
    # MLX (Prince Canuma)
    "MLX_AVAILABLE",
    "convert_pytorch_to_mlx_4bit",
    "mlx_inflate_inference_path_metadata",
    # Apple Neural Engine
    "coreml_export_metadata",
    "can_export_to_ane",
    # package state
    "IMPLEMENTED_MODULES",
    "DEFERRED_MODULES",
    "DEFERRED_RATIONALE",
]


# Catalog #270 hooks (operator-visible) — every helper here is exempt
# from the canonical scorer-loss + auth-eval gates because it is a
# CODEC/ARCHIVE primitive, not a substrate trainer. Substrate trainers
# that USE these primitives MUST go through the canonical helpers per
# Catalog #164 (scorer-preprocess), Catalog #226 (auth-eval), Catalog
# #205 (inflate device), Catalog #218 (mini-batch reconstruct).
# [empirical:.omx/research/hardware_exploits_design_and_implementation_landed_20260517.md]
