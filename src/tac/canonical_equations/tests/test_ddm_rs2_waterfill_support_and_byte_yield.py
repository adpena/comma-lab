"""Tests for the ddm_rs2 / ddm_br1 token-lattice drop-surface equations.

These test BEHAVIOUR, not constants: every evaluator is exercised on inputs that would
make a marker-returning stub fail (wrong ordering, wrong sign, refusal paths, and the
"would this test still pass if the body were `return CANONICAL_MARKERS`" mutation).
"""

from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_rs2_waterfill_support_and_byte_yield_20260803 import (
    BYTE_YIELD_EQUATION_ID,
    ENTROPY_SPLIT_EQUATION_ID,
    MEASURED_RF_NONZERO_PX,
    SUPPORT_EQUATION_ID,
    WR1_TILE_PX,
    build_lattice_cell_drop_pricing_support_v1,
    build_live_vs_dead_symbol_entropy_decomposition_v1,
    build_token_lattice_byte_marginal_flat_uncorrelated_v1,
    byte_side_is_rankable,
    coder_advantage_split,
    drop_support_px,
    support_mispricing,
)

BUILDERS = (
    build_lattice_cell_drop_pricing_support_v1,
    build_token_lattice_byte_marginal_flat_uncorrelated_v1,
    build_live_vs_dead_symbol_entropy_decomposition_v1,
)


# ---------------------------------------------------------------- L1 support


def test_drop_support_px_matches_the_measured_bbox() -> None:
    assert drop_support_px() == 84 * 82


def test_drop_support_px_is_computed_from_its_arguments_not_hardcoded() -> None:
    assert drop_support_px((0, 9), (0, 4)) == 50
    assert drop_support_px((100, 101), (7, 7)) == 2


def test_drop_support_px_refuses_an_inverted_bbox() -> None:
    with pytest.raises(ValueError):
        drop_support_px((10, 3), (0, 4))


def test_support_mispricing_flags_the_wr1_tile_as_unsound() -> None:
    out = support_mispricing()
    assert out["key_support_is_sound"] is False
    assert out["risk_understated"] is True
    assert out["support_ratio_measured_over_key"] == pytest.approx(
        MEASURED_RF_NONZERO_PX / WR1_TILE_PX, rel=1e-12
    )
    assert out["fraction_of_real_footprint_seen_by_key"] == pytest.approx(0.041343, abs=1e-5)


def test_support_mispricing_admits_a_key_whose_support_covers_the_footprint() -> None:
    out = support_mispricing(key_support_px=8000, measured_support_px=6192)
    assert out["key_support_is_sound"] is True
    assert out["risk_understated"] is False
    assert out["fraction_of_real_footprint_seen_by_key"] == 1.0


def test_support_mispricing_scales_with_the_key_support() -> None:
    small = support_mispricing(key_support_px=100, measured_support_px=6192)
    large = support_mispricing(key_support_px=1000, measured_support_px=6192)
    assert small["support_ratio_measured_over_key"] > large["support_ratio_measured_over_key"]
    assert small["support_ratio_measured_over_key"] == pytest.approx(61.92)


@pytest.mark.parametrize("bad", [(0, 10), (10, 0), (-5, 10)])
def test_support_mispricing_refuses_non_positive_supports(bad: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        support_mispricing(key_support_px=bad[0], measured_support_px=bad[1])


# ------------------------------------------------------------ L2 byte side


def test_byte_side_is_not_rankable_when_flat_and_uncorrelated() -> None:
    marg = [210.0, 211.0, 209.0, 212.0, 210.5, 211.5]
    dmg = [5.0, 900.0, 40.0, 3.0, 700.0, 60.0]
    out = byte_side_is_rankable(marg, dmg)
    assert out["byte_side_is_rankable"] is False


def test_byte_side_is_rankable_when_spread_and_correlated() -> None:
    marg = [10.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
    out = byte_side_is_rankable(marg, marg)
    assert out["byte_side_is_rankable"] is True
    assert out["spearman_bytes_vs_damage"] == pytest.approx(1.0)


def test_byte_side_reports_negative_marginals_and_breaks_monotonicity() -> None:
    out = byte_side_is_rankable([-58.0, 196.0, 211.0, 472.0], [1.0, 2.0, 3.0, 4.0])
    assert out["n_negative_marginals"] == 1
    assert out["greedy_monotonicity_holds"] is False


def test_byte_side_monotonicity_holds_when_all_positive() -> None:
    out = byte_side_is_rankable([248.0, 861.0, 1234.0], [1.0, 2.0, 3.0])
    assert out["n_negative_marginals"] == 0
    assert out["greedy_monotonicity_holds"] is True


def test_spearman_detects_a_perfectly_inverted_ranking() -> None:
    out = byte_side_is_rankable([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    assert out["spearman_bytes_vs_damage"] == pytest.approx(-1.0)


def test_byte_side_refuses_empty_and_mismatched_input() -> None:
    with pytest.raises(ValueError):
        byte_side_is_rankable([], [])
    with pytest.raises(ValueError):
        byte_side_is_rankable([1.0, 2.0], [1.0])


# ------------------------------------------------------- L3 entropy split


def test_coder_advantage_split_reproduces_the_measured_ratios() -> None:
    out = coder_advantage_split()
    assert out["apparent_advantage_over_order0"] == pytest.approx(1.4776, abs=5e-4)
    assert out["live_symbol_advantage_over_order0"] == pytest.approx(1.1093, abs=5e-4)
    assert out["free_riding_inflation"] > 1.0


def test_coder_advantage_split_is_computed_from_its_arguments() -> None:
    out = coder_advantage_split(
        all_shipped_b=100, all_order0_b=400, live_shipped_b=100, live_order0_b=200
    )
    assert out["apparent_advantage_over_order0"] == pytest.approx(4.0)
    assert out["live_symbol_advantage_over_order0"] == pytest.approx(2.0)
    assert out["free_riding_inflation"] == pytest.approx(2.0)
    assert out["dead_payload_bytes"] == 0


def test_coder_advantage_split_has_no_inflation_without_dead_symbols() -> None:
    out = coder_advantage_split(
        all_shipped_b=1000, all_order0_b=1500, live_shipped_b=1000, live_order0_b=1500
    )
    assert out["free_riding_inflation"] == pytest.approx(1.0)


def test_coder_advantage_split_refuses_non_positive_byte_counts() -> None:
    with pytest.raises(ValueError):
        coder_advantage_split(all_shipped_b=0)


# -------------------------------------------------------------- equations


@pytest.mark.parametrize("builder", BUILDERS)
def test_every_equation_builds_and_validates(builder) -> None:
    builder()


def test_equation_ids_are_the_declared_ones() -> None:
    got = {b().equation_id for b in BUILDERS}
    assert got == {SUPPORT_EQUATION_ID, BYTE_YIELD_EQUATION_ID, ENTROPY_SPLIT_EQUATION_ID}


@pytest.mark.parametrize("builder", BUILDERS)
def test_every_equation_carries_a_measured_anchor_and_makes_no_score_claim(builder) -> None:
    eq = builder()
    assert eq.empirical_anchors
    for anchor in eq.empirical_anchors:
        assert anchor.source_artifact.startswith(".omx/research/")
        assert anchor.measurement_method
    excluded = " ".join(eq.domain_of_validity["excluded"]).lower()
    assert "score" in excluded or "pointer" in excluded


@pytest.mark.parametrize("builder", BUILDERS)
def test_every_equation_points_at_a_real_importable_callable(builder) -> None:
    import importlib

    module_path, _, attr = builder().python_callable_module_path.partition(":")
    assert callable(getattr(importlib.import_module(module_path), attr))


def test_byte_yield_equation_carries_both_grains() -> None:
    eq = build_token_lattice_byte_marginal_flat_uncorrelated_v1()
    ids = {a.anchor_id for a in eq.empirical_anchors}
    assert any("unit" in i for i in ids)
    assert any("cell" in i for i in ids)
    cell = next(a for a in eq.empirical_anchors if "cell" in a.anchor_id)
    assert cell.empirical_output["n_negative_marginals"] == 0


def test_support_equation_excludes_gradient_keys_which_are_support_correct() -> None:
    eq = build_lattice_cell_drop_pricing_support_v1()
    assert any("gr1" in row or "backprop" in row for row in eq.domain_of_validity["excluded"])
