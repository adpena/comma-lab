"""Tests for ``ddm_fs1_composition_law_headroom``.

These pin BEHAVIOUR, not constants: each test constructs a synthetic per-pair
fixture whose correct answer is derivable by hand, so replacing a function body
with a canonical-looking return value fails.  The two tests that touch the real
retained store are skipped when the SSD is not mounted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.ddm_fs1_composition_law_headroom import (
    JG5_FINAL,
    JS6B_STORE,
    N_PAIRS,
    S_PER_BYTE,
    Jg5PerPair,
    compensation_distribution,
    js6b_compensated_rescreen,
    load_jg5_per_pair,
    marginal_dS_per_pair_d_pose,
    pose_actuator_break_even,
    pose_leg,
    unbanked_population_validity,
)


def _fixture(
    base: list[float],
    candidate: list[float],
    refined: list[float],
    kept: tuple[int, ...],
) -> Jg5PerPair:
    return Jg5PerPair(
        base=np.asarray(base, dtype=np.float64),
        candidate=np.asarray(candidate, dtype=np.float64),
        refined=np.asarray(refined, dtype=np.float64),
        kept_pairs=kept,
    )


# --- the score arithmetic ------------------------------------------------------


def test_pose_leg_is_the_contest_formula() -> None:
    assert pose_leg(1e-5) == pytest.approx(np.sqrt(1e-4))


def test_marginal_derivative_matches_a_numeric_difference() -> None:
    """The analytic dS/dd_i must agree with a finite difference on the real formula."""
    mean = 6.365684192e-06
    analytic = marginal_dS_per_pair_d_pose(mean)
    eps = 1e-12
    # Perturb ONE pair by eps -> the mean moves by eps / N_PAIRS.
    numeric = (pose_leg(mean + eps / N_PAIRS) - pose_leg(mean)) / eps
    assert analytic == pytest.approx(numeric, rel=1e-6)


def test_marginal_derivative_refuses_non_positive_mean() -> None:
    with pytest.raises(ValueError):
        marginal_dS_per_pair_d_pose(0.0)


# --- edited / shipped / dropped bookkeeping ------------------------------------


def test_shipped_takes_refined_on_kept_and_base_on_dropped() -> None:
    jg5 = _fixture(
        base=[1.0, 2.0, 3.0, 4.0],
        candidate=[1.0, 9.0, 9.0, 9.0],  # pair 0 unedited
        refined=[1.0, 0.5, 8.0, 7.0],
        kept=(1,),
    )
    assert jg5.shipped.tolist() == [1.0, 0.5, 3.0, 4.0]
    assert jg5.dropped_pairs == (2, 3)
    assert jg5.unedited_pairs == (0,)


# --- compensation distribution -------------------------------------------------


def test_compensation_factor_is_the_damage_ratio() -> None:
    """One edited pair: damage 10 -> 2 above base must read as exactly 5x."""
    jg5 = _fixture(base=[1.0], candidate=[11.0], refined=[3.0], kept=(0,))
    comp = compensation_distribution(jg5)
    assert comp["pairs_with_residual_damage"] == 1
    assert comp["residual_damage_factor_median"] == pytest.approx(5.0)
    assert comp["aggregate_factor_sum_over_sum"] == pytest.approx(5.0)


def test_pairs_landing_below_base_are_counted_not_folded_into_percentiles() -> None:
    """A below-base pair has no finite factor; it must not pollute the percentiles."""
    jg5 = _fixture(
        base=[1.0, 1.0],
        candidate=[11.0, 11.0],
        refined=[3.0, 0.25],  # second lands BELOW base
        kept=(0, 1),
    )
    comp = compensation_distribution(jg5)
    assert comp["pairs_landing_at_or_below_base"] == 1
    assert comp["fraction_at_or_below_base"] == pytest.approx(0.5)
    assert comp["pairs_with_residual_damage"] == 1
    assert comp["residual_damage_factor_median"] == pytest.approx(5.0)


# --- break-even byte budget ----------------------------------------------------


def test_break_even_bytes_invert_the_score_arithmetic() -> None:
    """A credit worth k*slope score units must break even at k*slope/S_PER_BYTE bytes."""
    jg5 = _fixture(
        base=[1e-5] * 4,
        candidate=[2e-5] * 4,
        refined=[1e-5 - 1e-6, 1e-5 - 1e-6, 1e-5 - 1e-6, 1e-5 - 1e-6],
        kept=(0, 1, 2, 3),
    )
    out = pose_actuator_break_even(jg5)
    slope = out["dS_per_pair_d_pose"]
    median_credit = out["break_even_budgets"]["median"]["per_pair_pose_credit"]
    expected_bytes = median_credit * slope / S_PER_BYTE
    assert out["break_even_budgets"]["median"][
        "break_even_bytes_per_pair"
    ] == pytest.approx(expected_bytes)
    assert median_credit == pytest.approx(1e-6)


def test_more_expensive_encoding_qualifies_fewer_pairs() -> None:
    """Monotonicity: raising the byte price can only shrink the paying set."""
    rng = np.random.default_rng(20260820)
    credits = rng.lognormal(mean=-14.0, sigma=1.5, size=200)
    base = np.full(200, 1e-5)
    jg5 = _fixture(
        base=base.tolist(),
        candidate=(base * 3).tolist(),
        refined=(base - credits).tolist(),
        kept=tuple(range(200)),
    )
    out = pose_actuator_break_even(jg5)
    ordered = sorted(
        out["measured_encodings"].values(), key=lambda r: r["bytes_per_pair"]
    )
    fractions = [r["qualifying_fraction"] for r in ordered]
    assert fractions == sorted(fractions, reverse=True)


# --- the population-validity refusal -------------------------------------------


def test_unbanked_population_validity_refuses_the_prior_transfer() -> None:
    jg5 = _fixture(
        base=[1.0, 1.0, 1.0],
        candidate=[1.0, 5.0, 5.0],  # pair 0 unedited
        refined=[1.0, 4.0, 0.5],  # pair 1 dropped (cost), pair 2 kept (credit)
        kept=(2,),
    )
    v = unbanked_population_validity(jg5)
    assert v["kept_pair_prior_is_admissible_for_unbanked"] is False
    assert v["all_dropped_pairs_measure_as_pose_cost"] is True
    assert v["unedited_have_no_edit_measurement"] is True
    assert v["unbanked_total"] == 2
    assert "HYPOTHESIS" in v["verdict"]


def test_validity_detects_a_dropped_pair_that_is_actually_a_credit() -> None:
    """If a credit pair were ever dropped, the all-costs claim must go False."""
    jg5 = _fixture(
        base=[1.0, 1.0],
        candidate=[5.0, 5.0],
        refined=[0.5, 0.25],  # both are credits, but pair 0 is dropped
        kept=(1,),
    )
    v = unbanked_population_validity(jg5)
    assert v["all_dropped_pairs_measure_as_pose_cost"] is False


# --- the js6b re-screen --------------------------------------------------------


def _write_rows(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "POSE_SCREEN.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def _row(pid: str, pair: int, seg: float, lo: float, hi: float) -> dict:
    return {
        "proposal_id": pid,
        "pair": pair,
        "semantic_cell_count": 1,
        "screen": {
            "optimistic_seg_value_s": seg,
            "measured_pose_risk_lower_s": lo,
            "measured_pose_risk_upper_s": hi,
        },
    }


def test_break_even_compensation_solves_the_screen_equation(tmp_path: Path) -> None:
    """At c = break_even the net is exactly zero; the row admits just above it."""
    rate_bytes = 1.0
    rate_s = rate_bytes * S_PER_BYTE
    seg = rate_s + 1e-5  # headroom is exactly 1e-5
    rows = [_row("a", 0, seg, lo=2e-5, hi=4e-5)]
    out = js6b_compensated_rescreen(_write_rows(tmp_path, rows), rate_bytes, (1.0,))
    entry = out["per_row"][0]
    assert entry["break_even_compensation_lower"] == pytest.approx(2.0)
    assert entry["break_even_compensation_upper"] == pytest.approx(4.0)


def test_row_with_no_seg_headroom_after_rate_reports_no_break_even(
    tmp_path: Path,
) -> None:
    """Rate alone can exceed the seg credit; no compensation rescues that row."""
    rate_bytes = 100.0
    rows = [_row("a", 0, 1e-9, lo=1e-5, hi=2e-5)]
    out = js6b_compensated_rescreen(_write_rows(tmp_path, rows), rate_bytes, (1e9,))
    entry = out["per_row"][0]
    assert entry["seg_headroom_after_rate_s"] < 0
    assert entry["break_even_compensation_lower"] is None
    assert out["compensation_sweep"]["c=1e+09"]["lower"]["admitted_rows"] == 0


def test_rate_is_charged_per_distinct_pair_in_the_ceiling(tmp_path: Path) -> None:
    """Two proposals on ONE pair must be charged one pair's bytes, not two."""
    rows = [_row("a", 7, 1e-5, 1e-9, 1e-9), _row("b", 7, 1e-5, 1e-9, 1e-9)]
    out = js6b_compensated_rescreen(_write_rows(tmp_path, rows), 5.667, (1.0,))
    assert out["bank_ceiling"]["distinct_pairs"] == 1
    assert out["bank_ceiling"]["rate_cost_all_pairs_s"] == pytest.approx(
        5.667 * S_PER_BYTE
    )


def test_higher_compensation_admits_a_superset(tmp_path: Path) -> None:
    rows = [_row(f"p{i}", i, 1e-5, lo=2e-5 * (i + 1), hi=4e-5) for i in range(5)]
    out = js6b_compensated_rescreen(_write_rows(tmp_path, rows), 1.0, (2.0, 20.0))
    low = out["compensation_sweep"]["c=2"]["lower"]["admitted_rows"]
    high = out["compensation_sweep"]["c=20"]["lower"]["admitted_rows"]
    assert high >= low


# --- live-store regression guards ----------------------------------------------


@pytest.mark.skipif(not JG5_FINAL.exists(), reason="jg5 retained store not mounted")
def test_live_jg5_store_reproduces_the_published_shipped_d_pose() -> None:
    """The shipped mixture must reproduce jg5 §6.2's published d_pose."""
    jg5 = load_jg5_per_pair()
    assert len(jg5.base) == N_PAIRS
    assert len(jg5.kept_pairs) == 455
    assert len(jg5.dropped_pairs) == 118
    assert len(jg5.unedited_pairs) == 27
    assert float(jg5.shipped.mean()) == pytest.approx(6.365684e-06, rel=1e-6)


@pytest.mark.skipif(
    not (JS6B_STORE / "POSE_SCREEN.jsonl").exists(),
    reason="js6b sealed store not mounted",
)
def test_uncompensated_rescreen_reproduces_js6b_zero_survivors() -> None:
    """POSITIVE CONTROL: at c=1 with zero rate the re-screen must hold all 200.

    This is the load-bearing control. A re-screen that admitted rows at c=1 would
    be measuring a different object than js6b did, and every compensated number
    downstream would be uninterpretable.
    """
    out = js6b_compensated_rescreen(
        JS6B_STORE / "POSE_SCREEN.jsonl", bytes_per_pair=0.0, compensation_factors=(1.0,)
    )
    assert out["input"]["row_count"] == 200
    assert out["compensation_sweep"]["c=1"]["lower"]["admitted_rows"] == 0
    assert out["compensation_sweep"]["c=1"]["upper"]["admitted_rows"] == 0


def test_empty_credit_population_refuses_rather_than_emitting_nan() -> None:
    """A NaN byte budget reads like a measurement; the function must refuse."""
    jg5 = _fixture(base=[1.0], candidate=[5.0], refined=[4.0], kept=(0,))
    with pytest.raises(ValueError, match="undefined"):
        pose_actuator_break_even(jg5)


def test_duplicate_compensation_factors_do_not_silently_collide(
    tmp_path: Path,
) -> None:
    rows = [_row("a", 0, 1e-5, 1e-6, 2e-6)]
    out = js6b_compensated_rescreen(_write_rows(tmp_path, rows), 1.0, (2.0, 2.0, 3.0))
    assert sorted(out["compensation_sweep"]) == ["c=2", "c=3"]
