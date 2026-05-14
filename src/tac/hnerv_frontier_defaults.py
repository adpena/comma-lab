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
    / "hnerv_frontier_scorecard_refresh_20260514_hlm2_codex"
    / "scorecard.json"
)
HNERV_ACTIVE_ENTROPY_RANKING = (
    EXPERIMENT_RESULTS
    / "hnerv_frontier_entropy_gap_ranking_20260514_hlm2_codex"
    / "frontier_entropy_gap_ranking.json"
)

ACTIVE_FLOOR_ARCHIVE_BYTES = 185_578
ACTIVE_RATE_ONLY_FLOOR_SCORE = 0.2089810755823297

ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_SCORE = 0.20637231876787215
ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_LABEL = "hnerv_hlm2_xmember_modal_t4_20260514T065903Z"

ACTIVE_SCORE_FRONTIER_SCORE = 0.20642625334307507
ACTIVE_SCORE_FRONTIER_LABEL = "pr106_r2_lowlevel_hdm4_candidate_pr101_runtime_cuda_20260513_codex"
