# SPDX-License-Identifier: MIT
"""Typed default-off policy for the cached control-interpolator tail audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any


@dataclass(frozen=True)
class ControlTailReliabilityPolicy:
    mode: str = "quant_tail_reliability_20260713"
    lambda_grid: tuple[float, ...] = (
        0.0,
        1.0e-4,
        3.0e-4,
        1.0e-3,
        3.0e-3,
        1.0e-2,
        3.0e-2,
        1.0e-1,
        3.0e-1,
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
        300.0,
        1000.0,
    )
    fit_resample_seeds: tuple[int, ...] = (455, 456, 457)
    cvar_alpha: float = 0.95
    retained_area_fraction: float = 0.047
    pre_se_mean_reference_lambda: float = 0.0
    organ_mean_reference_lambda: float = 1.0e-2
    pre_se_cache: str = "experiments/results/pre_se_locus_20260713"
    organ_cache: str = (
        "experiments/results/costate_organ_backtests/"
        "aniso_perclass_lambda_backtest_20260711T144317Z.json"
    )
    organ_run_dir: str = "experiments/results/levelset_v752_baseline_20260710T185913Z"
    live_training_enabled: bool = False
    scorer_calls_enabled: bool = False
    paid_or_remote_dispatch_enabled: bool = False
    live_run_mutation_enabled: bool = False
    archive_or_pointer_mutation_enabled: bool = False
    research_only: bool = True
    score_claim: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if self.mode != "quant_tail_reliability_20260713":
            raise ValueError("tail reliability mode drifted")
        if self.lambda_grid[0] != 0.0 or any(
            left >= right for left, right in pairwise(self.lambda_grid)
        ):
            raise ValueError("lambda grid must start at zero and increase strictly")
        if self.fit_resample_seeds != (455, 456, 457):
            raise ValueError("fit seeds must remain inherited from the sealed Round-5 contract")
        if self.cvar_alpha != 0.95 or self.retained_area_fraction != 0.047:
            raise ValueError("tail level and retained area require new provenance")
        if any(
            (
                self.live_training_enabled,
                self.scorer_calls_enabled,
                self.paid_or_remote_dispatch_enabled,
                self.live_run_mutation_enabled,
                self.archive_or_pointer_mutation_enabled,
            )
        ):
            raise ValueError("this policy has cached-read-only authority")
        if not self.research_only or self.score_claim or self.promotion_eligible:
            raise ValueError("tail audit is MEANS-only non-promotion evidence")

    def compile_measurement_contract(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["constant_provenance"] = {
            "lambda_grid_1e-4_to_1": (
                "ASSUMED-DIAGNOSTIC preregistered in heavy_tail_interp_fold_20260713"
            ),
            "lambda_grid_3_to_1000": (
                "DERIVED bracket expansion required to observe a turn or boundary optimum"
            ),
            "lambda_zero": "SOURCE operator: MP/ridgeless diagnostic must be included",
            "fit_resample_seeds": "MEASURED-INHERITED Round-5 seeds 455/456/457",
            "cvar_alpha": "SOURCE operator p95 tail request; DIAGNOSTIC, not adoption budget",
            "retained_area_fraction": "MEASURED-INHERITED sealed Round-5 requested area",
            "mean_gate": (
                "DERIVED no-mean-regression versus MP for PRE-SE and versus current 1e-2 for organ"
            ),
        }
        payload["pre_se_measurement_scope"] = {
            "core_states": 420,
            "train_only_dev_states": 60,
            "official_untouched_heldout_states": 120,
            "curve_surface": (
                "three row-stratified bootstrap RankRLS refits on cached 420-core arrays, "
                "evaluated on cached real exact-mass 60-state train-only dev"
            ),
            "official_heldout_limitation": (
                "n120 JSON receipts preserve MP selections and costate hashes but not raw "
                "features/costate mass; new-lambda masks cannot be reconstructed cache-only"
            ),
        }
        payload["numerical_authority"] = (
            "NumPy-fp32 score/rank/decision path; float64 eigensolve is optimization evidence only"
        )
        return payload


__all__ = ["ControlTailReliabilityPolicy"]
