from __future__ import annotations

import numpy as np

from experiments import ddm_js2b_edge_conditioning_relative_gauge as js2b


def test_stratified_sample_is_deterministic_and_covers_population() -> None:
    sample, weights = js2b.stratified_sample()
    assert sample.tolist() == [
        7,
        18,
        53,
        73,
        76,
        96,
        122,
        146,
        157,
        175,
        193,
        206,
        233,
        253,
        277,
        286,
        300,
        327,
        347,
        357,
        387,
        410,
        426,
        443,
        464,
        484,
        494,
        515,
        534,
        548,
        579,
        599,
    ]
    assert int(weights.sum()) == 600
    assert len(np.unique(sample)) == js2b.SAMPLE_N


def test_projected_sum_uses_stratum_denominator() -> None:
    _, weights = js2b.stratified_sample()
    assert js2b.projected_sum(np.ones(js2b.SAMPLE_N, dtype=np.int64), weights) == 600
    assert js2b.projected_sum(-np.ones(js2b.SAMPLE_N, dtype=np.int64), weights) == -600
    assert js2b.stratified_mean(np.ones(js2b.SAMPLE_N), weights) == 1.0


def test_fire_gate_requires_all_three_conditions() -> None:
    assert js2b.fire_gate(-2_000, 1_000, js2b.POSE_GUARD - 1e-12)
    assert not js2b.fire_gate(-1_999, 1_000, 0.0)
    assert not js2b.fire_gate(-2_000, 1_001, 0.0)
    assert not js2b.fire_gate(-2_000, 1_000, js2b.POSE_GUARD)


def test_margin_stats_separates_fragile_and_robust_flips() -> None:
    base = np.zeros((1, 1, 3), dtype=np.uint8)
    candidate = np.asarray([[[1, 1, 0]]], dtype=np.uint8)
    gt = np.asarray([[[1, 1, 0]]], dtype=np.uint8)
    logits = np.zeros((1, 5, 1, 3), dtype=np.float32)
    logits[0, 1, 0, 0] = 0.2
    logits[0, 1, 0, 1] = 2.0
    result = js2b.margin_stats(base, candidate, gt, logits, 1.0, np.asarray([600]))
    assert result["beneficial_flips"] == 2
    assert result["robust_beneficial_flips"] == 1
    assert result["tie_fragile_beneficial_flips"] == 1
    assert result["projected_n600_delta_flips"] == -1_200
    assert result["projected_n600_robust_delta_flips"] == -600
