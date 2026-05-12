"""Tests for tac.optimization.phase1_t13_t19_t20_t22_cost_refinement.

Per CLAUDE.md "Forbidden score claims": these tests verify that every
estimate is a ``[predicted; ...]`` band, never a measurement; and that
the cost-band metadata correctly informs the autopilot's
``fits_per_dispatch_cap`` / ``fits_cumulative_envelope`` flags.
"""
from __future__ import annotations

import pytest

from tac.optimization.phase1_t13_t19_t20_t22_cost_refinement import (
    CANONICAL_FLAG_OVERHEADS,
    FlagCombinationCostEstimate,
    FlagOverheadBand,
    RECOMMENDED_ALL_ON_FLAGS,
    REFERENCE_MODAL_T4,
    REFERENCE_VAST_4090,
    SCHEMA_VERSION,
    all_combination_estimates,
    estimate_combination_cost,
    iter_all_flag_combinations,
    recommended_default_combination,
    recommended_subsetting_order,
    serialize_estimate,
    serialize_full_table,
)


# ── Schema constants stable ──────────────────────────────────────────────


def test_schema_version_constant_stable():
    assert SCHEMA_VERSION == "tac_phase1_t13_t19_t20_t22_cost_refinement_v1"


def test_canonical_flag_overheads_count():
    assert len(CANONICAL_FLAG_OVERHEADS) == 4
    flags = {fb.flag for fb in CANONICAL_FLAG_OVERHEADS}
    assert flags == {
        "--enable-t13-sqrt-n-budget",
        "--enable-t19-adaptive-rho",
        "--enable-t20-kl-pose-distill",
        "--enable-t22-temporal-consistency",
    }


def test_canonical_flag_overheads_band_invariants():
    for fb in CANONICAL_FLAG_OVERHEADS:
        assert fb.wall_clock_pct_overhead >= 0.0
        assert fb.gpu_memory_mb_overhead >= 0
        assert fb.notes  # non-empty


def test_t13_overhead_is_zero():
    """T13 (Fridrich √n latent budget) does not add training-loop work."""
    by_flag = {fb.flag: fb for fb in CANONICAL_FLAG_OVERHEADS}
    assert by_flag["--enable-t13-sqrt-n-budget"].wall_clock_pct_overhead == 0.0
    assert by_flag["--enable-t13-sqrt-n-budget"].gpu_memory_mb_overhead == 0


def test_t20_overhead_dominates_others():
    """T20 (Hinton KL pose distill) is the dominant overhead — extra
    PoseNet forward per step."""
    by_flag = {fb.flag: fb for fb in CANONICAL_FLAG_OVERHEADS}
    t20 = by_flag["--enable-t20-kl-pose-distill"]
    others = [
        by_flag["--enable-t13-sqrt-n-budget"],
        by_flag["--enable-t19-adaptive-rho"],
        by_flag["--enable-t22-temporal-consistency"],
    ]
    for o in others:
        assert t20.wall_clock_pct_overhead > o.wall_clock_pct_overhead
        assert t20.gpu_memory_mb_overhead >= o.gpu_memory_mb_overhead


# ── Reference baselines ──────────────────────────────────────────────────


def test_modal_t4_baseline_cost_band():
    baseline = REFERENCE_MODAL_T4
    # 50 min / 60 = 5/6 hr; 5/6 * $0.59 = $0.4917
    assert baseline.baseline_cost_usd == pytest.approx(0.4917, abs=0.01)


def test_vast_4090_baseline_cheaper_than_modal_t4():
    """Per CLAUDE.md GPU budget table: Vast.ai 4090 is the optimal
    price/performance for our workload."""
    assert REFERENCE_VAST_4090.baseline_cost_usd < REFERENCE_MODAL_T4.baseline_cost_usd
    # Modal T4 is roughly 4-5x slower; Vast.ai costs roughly half per hour.
    # Combined: Vast.ai should be ~10x cheaper.
    ratio = REFERENCE_MODAL_T4.baseline_cost_usd / REFERENCE_VAST_4090.baseline_cost_usd
    assert ratio > 5.0


# ── estimate_combination_cost ────────────────────────────────────────────


def test_estimate_baseline_no_flags():
    e = estimate_combination_cost(())
    assert e.flags_enabled == ()
    assert e.wall_clock_minutes == pytest.approx(50.0)
    assert e.cost_usd == pytest.approx(0.4917, abs=0.01)
    assert e.fits_per_dispatch_cap_5usd is True
    assert e.fits_cumulative_cap_20usd is True


def test_estimate_t13_only_zero_overhead():
    """T13 has 0% overhead — same wall clock as baseline."""
    e = estimate_combination_cost(("--enable-t13-sqrt-n-budget",))
    assert e.wall_clock_minutes == pytest.approx(50.0)


def test_estimate_t19_only_one_pct_overhead():
    e = estimate_combination_cost(("--enable-t19-adaptive-rho",))
    # 50 * 1.01 = 50.5
    assert e.wall_clock_minutes == pytest.approx(50.5)


def test_estimate_t20_only_twelve_pct_overhead():
    e = estimate_combination_cost(("--enable-t20-kl-pose-distill",))
    # 50 * 1.12 = 56.0
    assert e.wall_clock_minutes == pytest.approx(56.0)


def test_estimate_all_on_combination():
    e = estimate_combination_cost(RECOMMENDED_ALL_ON_FLAGS)
    # 0 + 1 + 12 + 6 = 19% overhead -> 50 * 1.19 = 59.5 min
    assert e.wall_clock_minutes == pytest.approx(59.5)
    # 59.5 / 60 * 0.59 = $0.585
    assert e.cost_usd == pytest.approx(0.585, abs=0.01)
    # Cost band rounds up to nearest $0.05 → $0.60
    assert e.cost_band_usd == pytest.approx(0.60)
    assert e.fits_per_dispatch_cap_5usd is True
    assert e.fits_cumulative_cap_20usd is True


def test_estimate_all_on_combination_fits_5_dollar_cap_with_huge_headroom():
    """The all-on combination must fit well inside the autopilot's
    ≤$5/individual cap (CLAUDE.md cathedral autopilot mode)."""
    e = estimate_combination_cost(RECOMMENDED_ALL_ON_FLAGS)
    # Headroom: $5 - $0.60 = $4.40, more than 6x the cost itself
    assert e.cost_band_usd <= 1.0
    assert (5.0 - e.cost_band_usd) > 4.0


def test_estimate_all_on_vast_4090_even_cheaper():
    e = estimate_combination_cost(
        RECOMMENDED_ALL_ON_FLAGS, baseline=REFERENCE_VAST_4090
    )
    # 12 * 1.19 = 14.28 min; 14.28/60 * 0.25 = $0.0595
    assert e.wall_clock_minutes == pytest.approx(14.28)
    assert e.cost_usd < 0.10


def test_estimate_unknown_flag_raises():
    with pytest.raises(ValueError, match="unknown Phase 1 flag"):
        estimate_combination_cost(("--enable-totally-fake-flag",))


def test_estimate_gpu_memory_overhead_sums():
    """Combined memory overhead should sum across flags."""
    e_all = estimate_combination_cost(RECOMMENDED_ALL_ON_FLAGS)
    # 0 + 0 + 600 + 60 = 660 MB
    assert e_all.gpu_memory_mb_overhead == 660


def test_estimate_returns_predicted_tag_in_notes():
    e = estimate_combination_cost(RECOMMENDED_ALL_ON_FLAGS)
    assert "[predicted; Phase 1 T13+T19+T20+T22 cost refinement]" in e.notes


def test_estimate_carries_provider_hardware():
    e_modal = estimate_combination_cost((), baseline=REFERENCE_MODAL_T4)
    assert e_modal.provider == "modal"
    assert e_modal.hardware == "t4"
    e_vast = estimate_combination_cost((), baseline=REFERENCE_VAST_4090)
    assert e_vast.provider == "vastai"
    assert e_vast.hardware == "rtx_4090"


# ── Combinations ─────────────────────────────────────────────────────────


def test_iter_all_flag_combinations_yields_16():
    combos = list(iter_all_flag_combinations())
    assert len(combos) == 16


def test_iter_all_flag_combinations_includes_empty():
    combos = list(iter_all_flag_combinations())
    assert () in combos


def test_iter_all_flag_combinations_includes_all_on():
    combos = list(iter_all_flag_combinations())
    assert tuple(sorted(RECOMMENDED_ALL_ON_FLAGS)) in [
        tuple(sorted(c)) for c in combos
    ]


def test_all_combination_estimates_returns_16():
    estimates = all_combination_estimates()
    assert len(estimates) == 16
    for e in estimates:
        assert isinstance(e, FlagCombinationCostEstimate)


def test_all_combination_estimates_all_fit_5_dollar_cap():
    estimates = all_combination_estimates()
    for e in estimates:
        assert e.fits_per_dispatch_cap_5usd is True, (
            f"combination {e.flags_enabled!r} cost ${e.cost_band_usd:.2f} "
            "exceeds $5 cap"
        )


# ── Recommended default + subsetting ─────────────────────────────────────


def test_recommended_default_is_all_on():
    assert recommended_default_combination() == RECOMMENDED_ALL_ON_FLAGS


def test_recommended_subsetting_order_increasing_cost():
    order = recommended_subsetting_order()
    estimates = [estimate_combination_cost(combo) for combo in order]
    for i in range(len(estimates) - 1):
        assert estimates[i].cost_usd <= estimates[i + 1].cost_usd


def test_recommended_subsetting_starts_with_t13_t19():
    """The cheapest recommended subset is T13+T19 (no GPU overhead)."""
    order = recommended_subsetting_order()
    assert order[0] == ("--enable-t13-sqrt-n-budget", "--enable-t19-adaptive-rho")


def test_recommended_subsetting_ends_with_all_on():
    order = recommended_subsetting_order()
    assert order[-1] == RECOMMENDED_ALL_ON_FLAGS


# ── Serialization ────────────────────────────────────────────────────────


def test_serialize_estimate_round_trips_keys():
    e = estimate_combination_cost(RECOMMENDED_ALL_ON_FLAGS)
    d = serialize_estimate(e)
    expected_keys = {
        "flags_enabled",
        "provider",
        "hardware",
        "wall_clock_minutes",
        "cost_usd",
        "cost_band_usd",
        "gpu_memory_mb_overhead",
        "fits_per_dispatch_cap_5usd",
        "fits_cumulative_cap_20usd",
        "notes",
    }
    assert expected_keys.issubset(set(d.keys()))


def test_serialize_estimate_flags_enabled_is_list_not_tuple():
    e = estimate_combination_cost(RECOMMENDED_ALL_ON_FLAGS)
    d = serialize_estimate(e)
    assert isinstance(d["flags_enabled"], list)


def test_serialize_full_table_returns_canonical_payload():
    payload = serialize_full_table()
    assert payload["schema"] == SCHEMA_VERSION
    assert payload["n_combinations"] == 16
    assert payload["recommended_default_flags"] == list(RECOMMENDED_ALL_ON_FLAGS)
    assert payload["per_dispatch_cap_usd"] == 5.0
    assert payload["cumulative_cap_usd"] == 20.0
    assert "predicted_band_only_no_score_claim" in payload["claude_md_compliance_tags"]
    assert "operator_gate_non_negotiable_at_every_dispatch" in payload["claude_md_compliance_tags"]
    assert len(payload["estimates"]) == 16


def test_serialize_full_table_baseline_keys():
    payload = serialize_full_table()
    baseline = payload["baseline"]
    assert baseline["provider"] == "modal"
    assert baseline["hardware"] == "t4"
    assert baseline["wall_clock_minutes"] == 50.0


def test_serialize_full_table_with_vast_baseline():
    payload = serialize_full_table(baseline=REFERENCE_VAST_4090)
    assert payload["baseline"]["provider"] == "vastai"
    assert payload["baseline"]["hardware"] == "rtx_4090"


# ── No score claim ───────────────────────────────────────────────────────


def test_no_estimate_carries_score_claim():
    """Per CLAUDE.md 'Forbidden score claims': cost estimates carry NO score."""
    for e in all_combination_estimates():
        d = serialize_estimate(e)
        assert "score_claim" not in d
        assert "predicted_score_delta" not in d
        assert "actual_score" not in d


def test_full_table_carries_no_score_claim():
    payload = serialize_full_table()
    assert "score_claim" not in payload
    for e in payload["estimates"]:
        assert "score_claim" not in e
        assert "predicted_score_delta" not in e


# ── Cost band rounding ───────────────────────────────────────────────────


def test_cost_band_rounds_up_to_nearest_5_cents():
    """Cost bands round UP to the nearest $0.05 for autopilot ranking."""
    e = estimate_combination_cost(())
    # Baseline is $0.4917, rounds up to $0.50
    assert e.cost_band_usd == pytest.approx(0.50)


def test_cost_band_rounds_up_for_all_on():
    e = estimate_combination_cost(RECOMMENDED_ALL_ON_FLAGS)
    # All-on is $0.585, rounds up to $0.60
    assert e.cost_band_usd == pytest.approx(0.60)


def test_cost_band_never_below_cost_usd():
    """Cost band must be ≥ cost_usd (rounding up)."""
    for e in all_combination_estimates():
        assert e.cost_band_usd >= e.cost_usd
