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

Versioning + supersession (SELFCOMP-1, R2 MEDIUM, 2026-05-15):

* **v1 (``archive`` / ``inflate`` / ``score_aware_loss``)** is the LEGACY
  append-only Z3HP1 sidecar grammar. Council omnibus Decision 3 (commit
  ``7872c9f4b``, 2026-05-14) resolved that v1 is retired for production
  training/dispatch. The module remains importable only for deterministic
  forensic tests and historical packet parsing.
* **v2 (``archive_v2`` / ``inflate_v2`` / ``score_aware_loss_v2``)** is the
  Z3HV2 latent-replacement archive grammar that REPLACES A1's
  ``latent_blob`` in-place. The trainer uses v2 by default.
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
from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
    A1_DECODER_BLOB_LEN,
    A1_DECODER_SECTION_TOTAL,
    A1_LATENT_BLOB_LEN,
    A1_SECTION_TOTAL_PREFIX_LEN,
    Z3HV2_HEADER_STRUCT,
    Z3HV2_MAGIC,
    Z3HV2_PER_DIM_AFFINE_LEN,
    Z3HV2_VERSION,
    Z3HV2SectionMeta,
    Z3V2CompositionArchiveContract,
    build_z3v2_composition_archive_contract,
    build_z3v2_payload_bytes,
    decode_z3hv2_section,
    encode_z3hv2_section,
    split_z3v2_payload_bytes,
)
from tac.substrates.z3_balle_hyperprior_bolton.quant import (
    dequantize_int8_with_scale,
    quantize_int8_with_scale,
)

__all__ = [
    "A1_BASE_CHANNELS",
    "A1_CAMERA_H",
    "A1_CAMERA_W",
    "A1_DECODER_BLOB_LEN",
    "A1_DECODER_SECTION_TOTAL",
    "A1_EVAL_H",
    "A1_EVAL_W",
    "A1_LATENT_BLOB_LEN",
    "A1_LATENT_DIM",
    "A1_N_PAIRS",
    "A1_SECTION_TOTAL_PREFIX_LEN",
    "Z3HV2_HEADER_STRUCT",
    "Z3HV2_MAGIC",
    "Z3HV2_PER_DIM_AFFINE_LEN",
    "Z3HV2_VERSION",
    "Z3HV2SectionMeta",
    "Z3HyperpriorConfig",
    "Z3HyperpriorMLP",
    "Z3V2CompositionArchiveContract",
    "build_z3v2_composition_archive_contract",
    "build_z3v2_payload_bytes",
    "conditional_gaussian_rate_bits",
    "decode_z3hv2_section",
    "dequantize_int8_with_scale",
    "encode_z3hv2_section",
    "factorized_uniform_rate_bits",
    "quantize_int8_with_scale",
    "split_z3v2_payload_bytes",
    "total_balle_rate_bits",
]
