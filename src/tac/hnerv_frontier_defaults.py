# SPDX-License-Identifier: MIT
"""Current HNeRV frontier-routing constants.

These constants are routing defaults only. Promotion still requires the
individual exact-eval, custody, and compliance gates for each candidate.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_RESULTS = REPO_ROOT / "experiments" / "results"

HNERV_ACTIVE_SCORECARD = (
    EXPERIMENT_RESULTS
    / "hnerv_frontier_scorecard_refresh_20260514_hdm8_codex"
    / "scorecard.json"
)
HNERV_ACTIVE_ENTROPY_RANKING = (
    EXPERIMENT_RESULTS
    / "hnerv_frontier_entropy_gap_ranking_20260514_hdm8_codex"
    / "frontier_entropy_gap_ranking.json"
)

ACTIVE_FLOOR_ARCHIVE_BYTES = 185_578
ACTIVE_RATE_ONLY_FLOOR_SCORE = 0.2089810755823297

ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_SCORE = 0.2063556722940441
ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_LABEL = (
    "pr106_hdm8_fixed_meta_rank_elided_exact_cuda_20260515T002100Z"
)

ACTIVE_SCORE_FRONTIER_SCORE = 0.2063556722940441
ACTIVE_SCORE_FRONTIER_LABEL = "PR106-R2-HDM8-HLM2-XMEMBER"
