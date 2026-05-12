"""Phase 1 trainer T13/T19/T20/T22 cost estimation refinement.

Per OO landing 2026-05-11 (`feedback_github_release_tag_license_audit_
phase1_wiring_lane_sweep_landed_20260511.md`), T20 (Hinton KL pose distill
at T=2.0) and T22 (Horn-Schunck identity-warp temporal consistency) are
now wired into ``experiments/train_paradigm_delta_epsilon_zeta_track1_
balle_endtoend.py`` with default-OFF flags. T13 (Fridrich √n latent
budget) and T19 (Boyd adaptive ρ ADMM) had been wired earlier per
`feedback_t13_t19_phase1_trainer_integration_landed_20260509.md`.

This module produces machine-readable cost estimates for the 16 distinct
flag combinations so the cathedral autopilot's CandidateRow construction
can consume them directly without re-reading the cost-refinement
research ledger.

Per CLAUDE.md "Forbidden score claims": every cost estimate is a
``[predicted; ...]`` band. Promotion-eligible cost actuals require
empirical per-dispatch wall-clock measurement.

Per CLAUDE.md "operator-gate non-negotiable at every dispatch": this
module produces NO dispatch authorization. It produces cost-band metadata
the operator-gated autopilot consumes.

Cross-references
----------------
- ``.omx/research/phase1_t13_t19_t20_t22_cost_refinement_20260511.md``
- :mod:`tac.optimization.autopilot_dispatch_ranking`
- :mod:`tools.cathedral_autopilot_autonomous_loop`
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterator

SCHEMA_VERSION = "tac_phase1_t13_t19_t20_t22_cost_refinement_v1"


# ── Per-flag overhead bands (verified 2026-05-11) ────────────────────────


@dataclass(frozen=True)
class FlagOverheadBand:
    """Per-flag wall-clock and GPU-memory overhead band."""

    flag: str
    wall_clock_pct_overhead: float
    gpu_memory_mb_overhead: int
    notes: str


CANONICAL_FLAG_OVERHEADS: tuple[FlagOverheadBand, ...] = (
    FlagOverheadBand(
        flag="--enable-t13-sqrt-n-budget",
        wall_clock_pct_overhead=0.0,
        gpu_memory_mb_overhead=0,
        notes=(
            "Per-pair latent-budget reallocation n -> sqrt(n); affects rate "
            "target only; no extra forward pass."
        ),
    ),
    FlagOverheadBand(
        flag="--enable-t19-adaptive-rho",
        wall_clock_pct_overhead=1.0,
        gpu_memory_mb_overhead=0,
        notes=(
            "Boyd adaptive rho ADMM update; reads primal/dual residuals "
            "every step; scalar work."
        ),
    ),
    FlagOverheadBand(
        flag="--enable-t20-kl-pose-distill",
        wall_clock_pct_overhead=12.0,
        gpu_memory_mb_overhead=600,
        notes=(
            "Hinton KL on PoseNet 12-dim head logits at T=2.0; one extra "
            "PoseNet forward (the teacher) per step."
        ),
    ),
    FlagOverheadBand(
        flag="--enable-t22-temporal-consistency",
        wall_clock_pct_overhead=6.0,
        gpu_memory_mb_overhead=60,
        notes=(
            "Horn-Schunck identity-warp on (B, P=2, C, H, W); one grid_sample "
            "+ L2 between rendered frames."
        ),
    ),
)


# ── Reference baseline (no flags enabled; Modal T4) ──────────────────────


@dataclass(frozen=True)
class ReferenceBaseline:
    """Reference baseline cost for the Phase 1 trainer.

    Source: ``scripts/remote_lane_t1_balle_endtoend.sh`` defaults
    (epochs=3000, batch-size=16, rate-target-bytes=80000).
    """

    provider: str
    hardware: str
    wall_clock_minutes: float
    rate_per_hour_usd: float

    @property
    def wall_clock_hours(self) -> float:
        return self.wall_clock_minutes / 60.0

    @property
    def baseline_cost_usd(self) -> float:
        return self.wall_clock_hours * self.rate_per_hour_usd


REFERENCE_MODAL_T4 = ReferenceBaseline(
    provider="modal",
    hardware="t4",
    wall_clock_minutes=50.0,
    rate_per_hour_usd=0.59,
)

REFERENCE_VAST_4090 = ReferenceBaseline(
    provider="vastai",
    hardware="rtx_4090",
    wall_clock_minutes=12.0,  # ~4.2x faster than T4 per CLAUDE.md GPU table
    rate_per_hour_usd=0.25,
)


# ── Combination cost estimator ───────────────────────────────────────────


@dataclass(frozen=True)
class FlagCombinationCostEstimate:
    """Per-flag-combination cost estimate."""

    flags_enabled: tuple[str, ...]
    provider: str
    hardware: str
    wall_clock_minutes: float
    cost_usd: float
    cost_band_usd: float  # rounded up to nearest $0.05 for autopilot ranking
    gpu_memory_mb_overhead: int
    fits_per_dispatch_cap_5usd: bool
    fits_cumulative_cap_20usd: bool
    notes: str


def _round_up_to_band(cost_usd: float, granularity_usd: float = 0.05) -> float:
    if granularity_usd <= 0:
        raise ValueError(f"granularity_usd must be > 0; got {granularity_usd}")
    import math

    return math.ceil(cost_usd / granularity_usd) * granularity_usd


def estimate_combination_cost(
    flags_enabled: tuple[str, ...],
    *,
    baseline: ReferenceBaseline = REFERENCE_MODAL_T4,
    per_dispatch_cap_usd: float = 5.0,
    cumulative_cap_usd: float = 20.0,
) -> FlagCombinationCostEstimate:
    """Estimate the wall-clock + cost for a particular flag combination.

    Per CLAUDE.md "Forbidden score claims": this returns a
    ``[predicted; ...]`` band; never a measurement.
    """
    overheads_by_flag = {fb.flag: fb for fb in CANONICAL_FLAG_OVERHEADS}
    unknown = [f for f in flags_enabled if f not in overheads_by_flag]
    if unknown:
        raise ValueError(
            f"unknown Phase 1 flag(s): {unknown!r}; valid flags are "
            f"{sorted(overheads_by_flag.keys())}"
        )
    pct_overhead = sum(
        overheads_by_flag[f].wall_clock_pct_overhead for f in flags_enabled
    )
    mem_overhead = sum(
        overheads_by_flag[f].gpu_memory_mb_overhead for f in flags_enabled
    )
    wall_clock_minutes = baseline.wall_clock_minutes * (1.0 + pct_overhead / 100.0)
    cost_usd = (wall_clock_minutes / 60.0) * baseline.rate_per_hour_usd
    cost_band_usd = _round_up_to_band(cost_usd)
    return FlagCombinationCostEstimate(
        flags_enabled=tuple(flags_enabled),
        provider=baseline.provider,
        hardware=baseline.hardware,
        wall_clock_minutes=wall_clock_minutes,
        cost_usd=cost_usd,
        cost_band_usd=cost_band_usd,
        gpu_memory_mb_overhead=mem_overhead,
        fits_per_dispatch_cap_5usd=cost_band_usd <= per_dispatch_cap_usd,
        fits_cumulative_cap_20usd=cost_band_usd <= cumulative_cap_usd,
        notes=(
            f"[predicted; Phase 1 T13+T19+T20+T22 cost refinement] "
            f"{baseline.provider} {baseline.hardware}; "
            f"wall_clock={wall_clock_minutes:.1f} min; cost=${cost_usd:.2f}; "
            f"flags={list(flags_enabled)!r}"
        ),
    )


# ── All 16 combinations (recommended default + subsetting order) ─────────


def iter_all_flag_combinations() -> Iterator[tuple[str, ...]]:
    """Yield all 2^4 = 16 flag combinations in canonical order."""
    canonical_flags = tuple(fb.flag for fb in CANONICAL_FLAG_OVERHEADS)
    n = len(canonical_flags)
    for mask in range(1 << n):
        yield tuple(
            canonical_flags[i] for i in range(n) if (mask >> i) & 1
        )


def all_combination_estimates(
    *,
    baseline: ReferenceBaseline = REFERENCE_MODAL_T4,
    per_dispatch_cap_usd: float = 5.0,
    cumulative_cap_usd: float = 20.0,
) -> list[FlagCombinationCostEstimate]:
    """Return cost estimates for all 16 flag combinations."""
    return [
        estimate_combination_cost(
            combo,
            baseline=baseline,
            per_dispatch_cap_usd=per_dispatch_cap_usd,
            cumulative_cap_usd=cumulative_cap_usd,
        )
        for combo in iter_all_flag_combinations()
    ]


# ── Recommended default + subsetting ─────────────────────────────────────


RECOMMENDED_ALL_ON_FLAGS: tuple[str, ...] = (
    "--enable-t13-sqrt-n-budget",
    "--enable-t19-adaptive-rho",
    "--enable-t20-kl-pose-distill",
    "--enable-t22-temporal-consistency",
)


def recommended_default_combination() -> tuple[str, ...]:
    """Return the recommended Phase 1 dispatch flag combination.

    Per the cost refinement research ledger: T13+T19+T20+T22 all-on is
    recommended because (a) it fits well inside the autopilot's
    le-$5/individual cap, (b) all four flags are independently small
    bolt-ons, and (c) the predicted score deltas are additive on
    orthogonal axes per the substrate composition matrix.
    """
    return RECOMMENDED_ALL_ON_FLAGS


def recommended_subsetting_order() -> list[tuple[str, ...]]:
    """Return the recommended subsetting order for partial dispatches.

    When the autopilot's cumulative envelope is partially exhausted, the
    operator can pick a smaller subset by reading down this list.
    """
    return [
        ("--enable-t13-sqrt-n-budget", "--enable-t19-adaptive-rho"),
        (
            "--enable-t13-sqrt-n-budget",
            "--enable-t19-adaptive-rho",
            "--enable-t22-temporal-consistency",
        ),
        (
            "--enable-t13-sqrt-n-budget",
            "--enable-t19-adaptive-rho",
            "--enable-t20-kl-pose-distill",
        ),
        RECOMMENDED_ALL_ON_FLAGS,
    ]


# ── Serialization (for autopilot CandidateRow construction) ──────────────


def serialize_estimate(estimate: FlagCombinationCostEstimate) -> dict:
    d = dataclasses.asdict(estimate)
    d["flags_enabled"] = list(estimate.flags_enabled)
    return d


def serialize_full_table(
    *,
    baseline: ReferenceBaseline = REFERENCE_MODAL_T4,
    per_dispatch_cap_usd: float = 5.0,
    cumulative_cap_usd: float = 20.0,
) -> dict:
    """Return a JSON-safe dict containing the full 16-combination table."""
    estimates = all_combination_estimates(
        baseline=baseline,
        per_dispatch_cap_usd=per_dispatch_cap_usd,
        cumulative_cap_usd=cumulative_cap_usd,
    )
    return {
        "schema": SCHEMA_VERSION,
        "evidence_grade": "[predicted; Phase 1 T13+T19+T20+T22 cost refinement]",
        "claude_md_compliance_tags": [
            "predicted_band_only_no_score_claim",
            "operator_gate_non_negotiable_at_every_dispatch",
            "halt_and_ask_default_on",
            "no_tmp_paths",
            "cost_estimate_dispatch_dollar_envelope",
            "phase1_trainer_t13_t19_t20_t22_aware",
        ],
        "baseline": {
            "provider": baseline.provider,
            "hardware": baseline.hardware,
            "wall_clock_minutes": baseline.wall_clock_minutes,
            "rate_per_hour_usd": baseline.rate_per_hour_usd,
            "baseline_cost_usd": baseline.baseline_cost_usd,
        },
        "per_dispatch_cap_usd": per_dispatch_cap_usd,
        "cumulative_cap_usd": cumulative_cap_usd,
        "n_combinations": len(estimates),
        "recommended_default_flags": list(RECOMMENDED_ALL_ON_FLAGS),
        "estimates": [serialize_estimate(e) for e in estimates],
    }


__all__ = [
    "SCHEMA_VERSION",
    "FlagOverheadBand",
    "CANONICAL_FLAG_OVERHEADS",
    "ReferenceBaseline",
    "REFERENCE_MODAL_T4",
    "REFERENCE_VAST_4090",
    "FlagCombinationCostEstimate",
    "estimate_combination_cost",
    "iter_all_flag_combinations",
    "all_combination_estimates",
    "RECOMMENDED_ALL_ON_FLAGS",
    "recommended_default_combination",
    "recommended_subsetting_order",
    "serialize_estimate",
    "serialize_full_table",
]
