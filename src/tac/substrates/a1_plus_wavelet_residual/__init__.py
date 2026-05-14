# SPDX-License-Identifier: MIT
"""A1 + wavelet residual sidecar substrate.

Composition substrate that appends a Mallat wavelet residual sidecar to the
frozen A1 base archive.  The residual targets the pose-axis at PR106-level
operating points (2.71x pose marginal vs SegNet) by carrying high-frequency
RGB corrections at selected hard-pair indices, encoded as int8-quantized
Daubechies-4 detail bands (LH/HL/HH) under brotli compression.

Lane: ``lane_a1_plus_wavelet_residual_retarget_20260513``
META-COUNCIL audit: ``.omx/research/meta_council_decision_attribution_audit_20260513.md``
Sister substrate (A1+LAPose, D4.B magic-byte trailer): ``tac.substrates.a1_plus_lapose``

NO score claim. NO promotion. NO exact-eval dispatch from this module.
Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" 13
inviolable lessons + Catalog #100/#124 byte-closure discipline.
"""

from tac.substrates.a1_plus_wavelet_residual.architecture import (
    A1_BASE_CHANNELS,
    A1_CAMERA_H,
    A1_CAMERA_W,
    A1_EVAL_H,
    A1_EVAL_W,
    A1_LATENT_DIM,
    A1_N_PAIRS,
    A1PlusWaveletResidualConfig,
    PerPairWaveletResidualHead,
    parse_wavelet_residual_pair_indices,
)
from tac.substrates.a1_plus_wavelet_residual.archive import (
    WAVELET_SIDECAR_MAGIC,
    WAVELET_SIDECAR_VERSION,
    decode_wavelet_sidecar,
    encode_wavelet_sidecar,
    pack_composition_archive,
    split_composition_archive,
)
from tac.substrates.a1_plus_wavelet_residual.score_aware_loss import (
    A1PlusWaveletResidualLossWeights,
    A1PlusWaveletResidualScoreAwareLoss,
)

__all__ = [
    "A1_BASE_CHANNELS",
    "A1_CAMERA_H",
    "A1_CAMERA_W",
    "A1_EVAL_H",
    "A1_EVAL_W",
    "A1_LATENT_DIM",
    "A1_N_PAIRS",
    "A1PlusWaveletResidualConfig",
    "A1PlusWaveletResidualLossWeights",
    "A1PlusWaveletResidualScoreAwareLoss",
    "PerPairWaveletResidualHead",
    "WAVELET_SIDECAR_MAGIC",
    "WAVELET_SIDECAR_VERSION",
    "decode_wavelet_sidecar",
    "encode_wavelet_sidecar",
    "pack_composition_archive",
    "parse_wavelet_residual_pair_indices",
    "split_composition_archive",
]
