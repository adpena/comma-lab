# SPDX-License-Identifier: MIT
"""Canonical paired-comparison posterior store re-export for substrate_fit_diagnostic_consumer.

The actual fcntl-locked APPEND-ONLY JSONL store implementation lives in
:mod:`tac.cathedral_consumers.information_theoretic_floor_consumer._posterior_store`
because both Tier B promoted consumers share the SAME canonical store path
(``.omx/state/consumer_tier_b_promotion_posterior.jsonl``) — the per-row
``consumer_name`` field disambiguates which consumer emitted the row.

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 6 Step 6.5 + Catalog #131 (no bare
writes to shared state) + Catalog #110/#113 (HISTORICAL_PROVENANCE
APPEND-ONLY) + CLAUDE.md "consolidate everything into META layer" standing
directive.
"""
from __future__ import annotations

from tac.cathedral_consumers.information_theoretic_floor_consumer._posterior_store import (
    CONSUMER_TIER_B_PROMOTION_POSTERIOR_LOCK_PATH,
    CONSUMER_TIER_B_PROMOTION_POSTERIOR_PATH,
    SCHEMA_VERSION,
    PairedComparisonPosteriorCorruptError,
    append_paired_comparison_row,
    load_paired_comparison_rows_lenient,
    load_paired_comparison_rows_strict,
)

__all__ = [
    "SCHEMA_VERSION",
    "CONSUMER_TIER_B_PROMOTION_POSTERIOR_PATH",
    "CONSUMER_TIER_B_PROMOTION_POSTERIOR_LOCK_PATH",
    "PairedComparisonPosteriorCorruptError",
    "append_paired_comparison_row",
    "load_paired_comparison_rows_lenient",
    "load_paired_comparison_rows_strict",
]
