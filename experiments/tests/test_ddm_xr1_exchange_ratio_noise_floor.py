from __future__ import annotations

import math

import numpy as np

from experiments import ddm_xr1_exchange_ratio_noise_floor as xr1


def test_exact_total_calibrated_bootstrap_preserves_physical_point() -> None:
    base = np.full(xr1.PAIR_COUNT, 16.0, dtype=np.float64)
    candidate = base.copy()
    candidate[:100] -= 8.0
    draws = np.tile(np.arange(xr1.PAIR_COUNT, dtype=np.uint16), (4, 1))
    pair_delta, samples, fixed = xr1.exact_total_calibrated_bootstrap(
        base,
        candidate,
        draws,
        exact_delta_bytes=-101,
    )
    assert pair_delta.sum() == -100.0
    assert fixed == -1.0
    assert samples.tolist() == [-101.0] * 4


def test_scorer_pair_vectors_requires_complete_nonoverlapping_population() -> None:
    receipt = {
        "pair_count": xr1.PAIR_COUNT,
        "batch_stages": [
            {
                "pair_start": 0,
                "pair_stop_exclusive": xr1.PAIR_COUNT,
                "d_seg_per_pair": [0.1] * xr1.PAIR_COUNT,
                "d_pose_per_pair": [0.2] * xr1.PAIR_COUNT,
            }
        ],
    }
    seg, pose = xr1.scorer_pair_vectors(receipt)
    assert seg.shape == (xr1.PAIR_COUNT,)
    assert pose.shape == (xr1.PAIR_COUNT,)
    assert np.all(seg == 0.1)
    assert np.all(pose == 0.2)


def test_exact_mean_calibration_preserves_retained_aggregate() -> None:
    values = np.linspace(0.0, 1.0, xr1.PAIR_COUNT, dtype=np.float64)
    draws = np.tile(np.arange(xr1.PAIR_COUNT, dtype=np.uint16), (3, 1))
    samples, fixed = xr1.exact_mean_calibrated_bootstrap(
        values,
        draws,
        exact_mean=0.500_001,
    )
    assert math.isclose(fixed, 0.000_001, abs_tol=1e-15)
    assert np.allclose(samples, 0.500_001, rtol=0.0, atol=1e-15)


def test_fcd3_point_score_rederives_published_delta() -> None:
    point = xr1.score_delta(
        base_d_seg=0.0003474002587608993,
        candidate_d_seg=0.0003874630492646247,
        base_d_pose=0.0001470109127694741,
        candidate_d_pose=0.00014620431466028094,
        delta_bytes=-2_940,
    )
    assert math.isclose(point["delta_s"], 0.0019433243907622244, abs_tol=1e-15)
    assert point["delta_s"] > 0.0
    assert point["delta_s_rate"] < 0.0
    assert point["delta_s_distortion"] > 0.0


def test_percentile_interval_reports_full_and_half_width() -> None:
    interval = xr1.percentile_interval(np.arange(200, dtype=np.float64))
    assert interval["low"] < interval["high"]
    assert interval["width"] == 2.0 * interval["half_width"]
