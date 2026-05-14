# SPDX-License-Identifier: MIT
"""Z3 Ballé hyperprior bolt-on substrate package.

Across-class staircase Step 1 per the zen-floor council
(`feedback_zen_floor_band_v2_post_z1_ablation_20260514.md`) + long-term
campaign roadmap (`feedback_long_term_multi_year_campaigns_landed_20260514.md`):
the cheapest $2 validation that Ballé-2018 scale hyperprior side-info
reduces bytes on the frozen A1 base.

Re-exports the public API of the architecture + archive + inflate modules
so callers can ``from tac.substrates.z3_balle_hyperprior_bolton import ...``.

NO score claim. NO promotion. NO exact-eval dispatch from this module.
Tagged ``research_only=true`` until empirical smoke + full-run anchors land.
"""
from __future__ import annotations

from tac.substrates.z3_balle_hyperprior_bolton.architecture import (
    A1_BASE_CHANNELS,
    A1_CAMERA_H,
    A1_CAMERA_W,
    A1_EVAL_H,
    A1_EVAL_W,
    A1_LATENT_DIM,
    A1_N_PAIRS,
    Z3HyperpriorConfig,
    Z3HyperpriorMLP,
    conditional_gaussian_rate_bits,
    factorized_uniform_rate_bits,
    total_balle_rate_bits,
)
from tac.substrates.z3_balle_hyperprior_bolton.archive import (
    Z3_APPEND_ONLY_CONTRACT_BLOCKER,
    Z3_BYTE_IDENTICAL_CONTRACT_BLOCKER,
    Z3CompositionArchiveContract,
    Z3HP1_HEADER_STRUCT,
    Z3HP1_MAGIC,
    Z3HP1_VERSION,
    Z3HP1SidecarMeta,
    build_composition_archive_contract,
    decode_z3hp1_sidecar,
    dequantize_int8_with_scale,
    encode_z3hp1_sidecar,
    pack_composition_archive,
    quantize_int8_with_scale,
    split_composition_archive,
)

__all__ = [
    "A1_BASE_CHANNELS",
    "A1_CAMERA_H",
    "A1_CAMERA_W",
    "A1_EVAL_H",
    "A1_EVAL_W",
    "A1_LATENT_DIM",
    "A1_N_PAIRS",
    "Z3_APPEND_ONLY_CONTRACT_BLOCKER",
    "Z3_BYTE_IDENTICAL_CONTRACT_BLOCKER",
    "Z3CompositionArchiveContract",
    "Z3HP1_HEADER_STRUCT",
    "Z3HP1_MAGIC",
    "Z3HP1_VERSION",
    "Z3HP1SidecarMeta",
    "Z3HyperpriorConfig",
    "Z3HyperpriorMLP",
    "build_composition_archive_contract",
    "conditional_gaussian_rate_bits",
    "decode_z3hp1_sidecar",
    "dequantize_int8_with_scale",
    "encode_z3hp1_sidecar",
    "factorized_uniform_rate_bits",
    "pack_composition_archive",
    "quantize_int8_with_scale",
    "split_composition_archive",
    "total_balle_rate_bits",
]
