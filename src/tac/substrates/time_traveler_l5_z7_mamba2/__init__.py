# SPDX-License-Identifier: MIT
"""Z7-Mamba-2 recurrent predictive-coding substrate package.

This package wires the Mamba-2 selective state-space predictor primitive
(``tac.optimization.mamba2_predictor.Mamba2Predictor``) into a full
substrate-distinguishing architecture (Mamba-2 recurrent autoregression
+ Z6-compatible PixelShuffle decoder), exports the Z7MCM2 archive
grammar (sister to Z7-LSTM/GRU Z7PCWM1), and provides the score-aware
training loss + scorer-free inflate runtime.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode": the
Mamba-2 predictor is the UNIQUE FORK from sister Z7-LSTM/GRU; decoder,
context conditioner, archive grammar skeleton, and loss formulation are
CANONICAL ADOPT for paired-comparison cleanliness.

Per the Z7-Mamba-2 design memo
(``.omx/research/z7_mamba2_substrate_design_memo_20260518.md``):
this is the TOP-5 #2 FULL substrate per the deep-research wave
(``.omx/research/comprehensive_research_wave_20260518.md`` §0 + §2.2 + §3.6).
"""

from __future__ import annotations

from tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.architecture import (
    CONTEXT_CONDITIONING_MODES,
    EVAL_HW,
    NUM_PAIRS,
    LatentAffineContextConditioner,
    normalize_context_conditioning_mode,
)
from tac.substrates.time_traveler_l5_z7_mamba2.architecture import (
    MAMBA_SSM_AVAILABLE,
    Z7Mamba2PredictiveCodingConfig,
    Z7Mamba2PredictiveCodingSubstrate,
)
from tac.substrates.time_traveler_l5_z7_mamba2.archive import (
    Z7MCM2_HEADER_FMT,
    Z7MCM2_HEADER_SIZE,
    Z7MCM2_MAGIC,
    Z7MCM2_SCHEMA_VERSION,
    Z7MCM2_SECTION_ROLES,
    Z7Mamba2PredictiveCodingArchive,
    pack_archive,
    parse_archive,
    parse_z7mcm2_archive_bytes,
    replay_latent_sequence,
    replay_latent_sequence_with_context,
)
from tac.substrates.time_traveler_l5_z7_mamba2.score_aware_loss import (
    Z7Mamba2PredictiveCodingLossWeights,
    Z7Mamba2PredictiveCodingScoreAwareLoss,
)

IMPLEMENTATION_STATUS = "full_substrate_predictor_archive_inflate_score_aware_loss_recipe_dispatch_disabled_pending_wave_n_plus_1_council"
RESEARCH_ONLY = True

PLANNED_PUBLIC_API = (
    "Z7Mamba2PredictiveCodingConfig",
    "Z7Mamba2PredictiveCodingSubstrate",
    "Z7Mamba2PredictiveCodingArchive",
    "Z7Mamba2PredictiveCodingScoreAwareLoss",
    "Z7Mamba2PredictiveCodingLossWeights",
    "pack_archive",
    "parse_archive",
    "parse_z7mcm2_archive_bytes",
    "replay_latent_sequence",
    "replay_latent_sequence_with_context",
)

__all__ = (
    "EVAL_HW",
    "NUM_PAIRS",
    "CONTEXT_CONDITIONING_MODES",
    "IMPLEMENTATION_STATUS",
    "MAMBA_SSM_AVAILABLE",
    "LatentAffineContextConditioner",
    "PLANNED_PUBLIC_API",
    "RESEARCH_ONLY",
    "Z7MCM2_HEADER_FMT",
    "Z7MCM2_HEADER_SIZE",
    "Z7MCM2_MAGIC",
    "Z7MCM2_SCHEMA_VERSION",
    "Z7MCM2_SECTION_ROLES",
    "Z7Mamba2PredictiveCodingArchive",
    "Z7Mamba2PredictiveCodingConfig",
    "Z7Mamba2PredictiveCodingLossWeights",
    "Z7Mamba2PredictiveCodingScoreAwareLoss",
    "Z7Mamba2PredictiveCodingSubstrate",
    "normalize_context_conditioning_mode",
    "pack_archive",
    "parse_archive",
    "parse_z7mcm2_archive_bytes",
    "replay_latent_sequence",
    "replay_latent_sequence_with_context",
)
