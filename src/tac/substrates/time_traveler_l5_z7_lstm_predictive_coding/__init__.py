# SPDX-License-Identifier: MIT
"""Z7 GRU recurrent predictive-coding prebuild scaffold.

This package is a research-only scaffold for
``time_traveler_l5_z7_lstm_predictive_coding``. It exports the GRU recurrent
predictor primitive, config, substrate, and the Z7PCWM1 archive grammar
scaffold. An opt-in score-aware training loss exists for compress time only;
score authority remains intentionally blocked until the Wave N+1 council and
paired exact-eval disambiguator gates land.
"""

from __future__ import annotations

from tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.architecture import (
    EVAL_HW,
    NUM_PAIRS,
    GruRecurrentPredictor,
    Z7GruPredictiveCodingConfig,
    Z7GruPredictiveCodingSubstrate,
)
from tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.archive import (
    Z7PCWM1_HEADER_FMT,
    Z7PCWM1_HEADER_SIZE,
    Z7PCWM1_MAGIC,
    Z7PCWM1_SCHEMA_VERSION,
    Z7PCWM1_SECTION_ROLES,
    Z7PredictiveCodingArchive,
    pack_archive,
    parse_archive,
    parse_z7pcwm1_archive_bytes,
    replay_latent_sequence,
)

IMPLEMENTATION_STATUS = "prebuild_research_only_gru_substrate_archive_runtime_export_and_score_aware_training_path"
RESEARCH_ONLY = True

PLANNED_PUBLIC_API = (
    "GruRecurrentPredictor",
    "Z7PredictiveCodingArchive",
    "Z7GruPredictiveCodingConfig",
    "Z7GruPredictiveCodingSubstrate",
    "pack_archive",
    "parse_archive",
    "parse_z7pcwm1_archive_bytes",
    "replay_latent_sequence",
)

__all__ = (
    "EVAL_HW",
    "NUM_PAIRS",
    "GruRecurrentPredictor",
    "IMPLEMENTATION_STATUS",
    "PLANNED_PUBLIC_API",
    "RESEARCH_ONLY",
    "Z7PCWM1_HEADER_FMT",
    "Z7PCWM1_HEADER_SIZE",
    "Z7PCWM1_MAGIC",
    "Z7PCWM1_SCHEMA_VERSION",
    "Z7PCWM1_SECTION_ROLES",
    "Z7PredictiveCodingArchive",
    "Z7GruPredictiveCodingConfig",
    "Z7GruPredictiveCodingSubstrate",
    "pack_archive",
    "parse_archive",
    "parse_z7pcwm1_archive_bytes",
    "replay_latent_sequence",
)
