# SPDX-License-Identifier: MIT
"""Tests for ddm_gd1 (#817) gate-estimator arithmetic.

Behaviour-verifying, not constant-verifying (NO-FAKE class 2): every assertion below
would FAIL if the estimator bodies were replaced by canonical markers or by the
unweighted mean they exist to replace.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.ddm_gd1_gate_estimator import (
    GateDesign,
    anchored_mean,
    bias_decomposition,
    horvitz_thompson_mean,
    neyman_stratified_gate,
    srs_standard_error,
)


def _live_like_design(seed: int = 0) -> GateDesign:
    """Mirror of the live geometry: block 447-450 + 32 SRS from the other 596."""
    block = (447, 448, 449, 450)
    rng = np.random.default_rng(seed)
    off = [p for p in range(600) if p not in set(block)]
    srs = tuple(int(x) for x in rng.choice(off, size=32, replace=False))
    return GateDesign(n_population=600, block_ids=block, srs_ids=srs)


def test_design_matches_live_trainer_geometry():
    """The module's design mirrors resolve_gate_ids(600) exactly (drift guard)."""
    from experiments.train_tr1_partition_renderer_mlx import (
        GATE_BLOCK_PAIRS,
        resolve_gate_ids,
    )

    ids = resolve_gate_ids(600)
    d = _live_like_design()
    assert tuple(GATE_BLOCK_PAIRS) == d.block_ids
    assert set(ids) == set(d.block_ids) | set(d.srs_ids)
    assert len(ids) == 36


def test_block_is_overweighted_by_the_exact_factor():
    d = _live_like_design()
    assert d.n_gate == 36
    assert d.n_offblock == 596
    # 4/36 in the estimator vs 4/600 in the population.
    assert (d.n_block / d.n_gate) / (d.n_block / d.n_population) == pytest.approx(600 / 36)
    assert d.block_overweight_coefficient == pytest.approx(4 / 36 - 4 / 600)


def test_weights_sum_to_one():
    d = _live_like_design()
    wb, ws = d.weights()
    assert wb.sum() + ws.sum() == pytest.approx(1.0)
    assert wb[0] == pytest.approx(1 / 600)
    assert ws[0] == pytest.approx((596 / 600) / 32)


def test_ht_is_exactly_unbiased_when_srs_mean_equals_offblock_mean():
    """If the SRS draw happens to hit the off-block mean, HT is EXACT; the unweighted
    mean is still wrong by the block over-weight term."""
    d = _live_like_design()
    x = np.zeros(600, dtype=np.float64)
    off = [i for i in range(600) if i not in set(d.block_ids)]
    x[off] = 1.0  # every off-block pair identical => SRS mean == off-block mean exactly
    x[list(d.block_ids)] = 5.0
    dec = bias_decomposition(d, x)
    assert dec.horvitz_thompson == pytest.approx(dec.pop_mean, abs=1e-12)
    assert dec.srs_sampling_term == pytest.approx(0.0, abs=1e-12)
    assert dec.unweighted != pytest.approx(dec.pop_mean, abs=1e-6)
    assert dec.total_error == pytest.approx(dec.block_overweight_term, abs=1e-12)


def test_decomposition_terms_sum_to_total_error_on_random_data():
    rng = np.random.default_rng(7)
    d = _live_like_design()
    for _ in range(20):
        x = rng.gamma(2.0, 0.002, size=600)
        dec = bias_decomposition(d, x)
        assert dec.block_overweight_term + dec.srs_sampling_term == pytest.approx(
            dec.total_error, abs=1e-12
        )


def test_ht_mean_from_gate_values_matches_decomposition():
    rng = np.random.default_rng(11)
    d = _live_like_design()
    x = rng.random(600)
    gv = {i: float(x[i]) for i in (*d.block_ids, *d.srs_ids)}
    assert horvitz_thompson_mean(d, gv) == pytest.approx(bias_decomposition(d, x).horvitz_thompson)


def test_ht_is_unbiased_in_expectation_over_the_srs_draw():
    """Averaging HT over many SRS draws converges to the population mean; averaging the
    UNWEIGHTED mean does not (it keeps the block over-weight term)."""
    rng = np.random.default_rng(3)
    x = rng.gamma(2.0, 1.0, size=600)
    x[447:451] += 4.0  # a hard block, like the live one
    pop = float(x.mean())
    hts, unws = [], []
    for s in range(400):
        d = _live_like_design(seed=s)
        gv = {i: float(x[i]) for i in (*d.block_ids, *d.srs_ids)}
        hts.append(horvitz_thompson_mean(d, gv))
        unws.append(float(np.mean([x[i] for i in (*d.block_ids, *d.srs_ids)])))
    ht_bias = abs(float(np.mean(hts)) - pop)
    unw_bias = abs(float(np.mean(unws)) - pop)
    assert ht_bias < 0.15 * unw_bias, (ht_bias, unw_bias)


def test_anchored_mean_removes_a_fixed_offset():
    assert anchored_mean(ht_now=0.0041, ht_at_anchor=0.0043, n600_at_anchor=0.0042) == pytest.approx(
        0.0040
    )


def test_srs_standard_error_shrinks_with_budget():
    rng = np.random.default_rng(5)
    x = rng.gamma(2.0, 0.002, size=600)
    small = GateDesign(600, (447, 448, 449, 450), tuple(range(8)))
    big = GateDesign(600, (447, 448, 449, 450), tuple(range(128)))
    assert srs_standard_error(big, x) < srs_standard_error(small, x)


def test_neyman_allocation_respects_budget_and_beats_srs_on_skewed_data():
    d = _live_like_design()
    rng = np.random.default_rng(13)
    sens = rng.lognormal(mean=0.0, sigma=1.2, size=600)  # skewed, like flip-prone mass
    out = neyman_stratified_gate(d, sens, n_strata=4)
    assert sum(out["neyman_allocation"]) == d.n_srs
    assert all(a >= 1 for a in out["neyman_allocation"])
    assert len(out["candidate_srs_replacement_ids"]) == d.n_srs
    assert set(out["candidate_srs_replacement_ids"]).isdisjoint(set(d.block_ids))
    assert out["predicted_var_ratio_stratified_over_srs"] < 1.0
    # heavy tail => the top stratum gets more than its equal share
    assert out["neyman_allocation"][-1] > d.n_srs / 4


def test_design_rejects_malformed_inputs():
    with pytest.raises(ValueError):
        GateDesign(600, (1, 2), (2, 3))  # overlapping
    with pytest.raises(ValueError):
        GateDesign(600, (1,), ())  # empty srs
    with pytest.raises(ValueError):
        GateDesign(600, (1,), (999,))  # out of range
    d = _live_like_design()
    with pytest.raises(ValueError):
        bias_decomposition(d, np.zeros(599))
    with pytest.raises(KeyError):
        horvitz_thompson_mean(d, {})
