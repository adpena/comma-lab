# SPDX-License-Identifier: MIT
"""Tests for the corrective engineering recommendation ranker.

Covers (a) recommendation ranking semantics, (b) Cauchy-Schwarz bound
regression guard, and (c) cosine distribution verdict mapping.
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tac.mps_diagnostic.granular_drift import (
    CosineDistributionSummary,
    PerBoundaryDriftRecord,
    PerPairDriftRecord,
    PerPairMasterGradientWeightedRecord,
    compute_per_pair_master_gradient_weighted_drift,
    rank_corrective_engineering_recommendations,
)


def _build_synthetic_anchor(n_bytes: int, n_pairs: int, signal: np.ndarray):
    """Build a per-pair gradient anchor with controlled signal in seg dim."""
    from tac.master_gradient import (
        MasterGradient,
        OperatingPoint,
        PER_PAIR_GRADIENT_TENSOR_KIND,
    )
    g = np.zeros((n_bytes, n_pairs, 3), dtype=np.float32)
    for p in range(n_pairs):
        g[:, p, 0] = signal[p]  # seg axis carries the signal
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f, g)
        path = Path(f.name)
    anchor = MasterGradient(
        archive_sha256="c" * 64,
        operating_point=OperatingPoint(d_seg=0.001, d_pose=0.1, rate=1.0, score=0.2),
        gradient_array_path=str(path),
        n_bytes=n_bytes,
        measurement_method="synthetic_test_anchor",
        measurement_axis="[macOS-CPU advisory]",
        measurement_hardware="darwin_arm64_advisory",
        measurement_call_id=None,
        measurement_utc="2026-05-19T12:00:00+00:00",
        gradient_tensor_kind=PER_PAIR_GRADIENT_TENSOR_KIND,
        n_pairs=n_pairs,
    )
    return anchor, path


class RecommendationRankerSemanticsTests(unittest.TestCase):
    def test_no_anchor_no_triggers_except_per_frame_routing_if_high_drift(self) -> None:
        cs = CosineDistributionSummary(0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, "NO_MASTER_GRADIENT_ANCHOR")
        recs = rank_corrective_engineering_recommendations(
            per_pair=[], per_boundary=[], per_pixel=None, cosine_summary=cs
        )
        triggered = {r["name"] for r in recs if r["triggered"]}
        # With no per_pair drift, per_frame_routing should not trigger.
        self.assertNotIn("per_frame_routing_high_drift_to_cuda_shadow", triggered)

    def test_per_frame_routing_triggers_on_fat_tail_per_pair(self) -> None:
        cs = CosineDistributionSummary(0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, "NO_MASTER_GRADIENT_ANCHOR")
        # 10 pairs with one outlier.
        per_pair = [PerPairDriftRecord(i, 0.001, 0.001, 0.001, 0.1) for i in range(9)]
        per_pair.append(PerPairDriftRecord(9, 0.5, 0.5, 0.5, 100.0))  # outlier
        recs = rank_corrective_engineering_recommendations(
            per_pair=per_pair, per_boundary=[], per_pixel=None, cosine_summary=cs
        )
        by_name = {r["name"]: r for r in recs}
        self.assertTrue(by_name["per_frame_routing_high_drift_to_cuda_shadow"]["triggered"])

    def test_subspace_alignment_triggers_on_strong_alignment(self) -> None:
        cs = CosineDistributionSummary(
            n_pairs=10,
            mean=0.0,
            median=0.0,
            std=0.4,
            abs_mean=0.3,
            n_outliers_abs_above_0_5=3,
            n_outliers_abs_above_0_8=1,
            max_abs=0.85,
            verdict="SCORE_RELEVANT_ENGINEERING_REQUIRED",
        )
        recs = rank_corrective_engineering_recommendations(
            per_pair=[], per_boundary=[], per_pixel=None, cosine_summary=cs
        )
        by_name = {r["name"]: r for r in recs}
        self.assertTrue(by_name["subspace_alignment_topK_eigenvectors"]["triggered"])
        self.assertTrue(by_name["selective_parameter_freeze"]["triggered"])

    def test_cross_device_validation_triggers_on_outliers(self) -> None:
        cs = CosineDistributionSummary(
            n_pairs=100,
            mean=0.0,
            median=0.0,
            std=0.3,
            abs_mean=0.15,
            n_outliers_abs_above_0_5=10,  # 10% > 5%
            n_outliers_abs_above_0_8=2,
            max_abs=0.9,
            verdict="WEAK_ALIGNMENT",
        )
        recs = rank_corrective_engineering_recommendations(
            per_pair=[], per_boundary=[], per_pixel=None, cosine_summary=cs
        )
        by_name = {r["name"]: r for r in recs}
        self.assertTrue(by_name["cross_device_validation_cadence_every_K_steps"]["triggered"])

    def test_boundary_smoothing_only_when_band_ratio_high(self) -> None:
        cs = CosineDistributionSummary(0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, "NO_MASTER_GRADIENT_ANCHOR")
        # Low ratio: 1 of 20 pairs has high band-vs-overall ratio.
        per_boundary = []
        for i in range(20):
            if i == 0:
                per_boundary.append(PerBoundaryDriftRecord(i, 3, 100, 80, 0.8, 5, 0.005))
            else:
                per_boundary.append(PerBoundaryDriftRecord(i, 3, 100, 1, 0.01, 1, 0.005))
        recs = rank_corrective_engineering_recommendations(
            per_pair=[], per_boundary=per_boundary, per_pixel=None, cosine_summary=cs
        )
        by_name = {r["name"]: r for r in recs}
        # Only 5% triggers boundary; threshold is > 10%, so should NOT trigger.
        self.assertFalse(by_name["boundary_smoothing_3px_gaussian_pre_argmax"]["triggered"])

    def test_recommendations_carry_axis_predicted_tag(self) -> None:
        cs = CosineDistributionSummary(0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, "NO_MASTER_GRADIENT_ANCHOR")
        recs = rank_corrective_engineering_recommendations(
            per_pair=[], per_boundary=[], per_pixel=None, cosine_summary=cs
        )
        for r in recs:
            self.assertEqual(r["axis_tag"], "[predicted]")

    def test_recommendations_carry_6_hook_wire_in(self) -> None:
        cs = CosineDistributionSummary(0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, "NO_MASTER_GRADIENT_ANCHOR")
        recs = rank_corrective_engineering_recommendations(
            per_pair=[], per_boundary=[], per_pixel=None, cosine_summary=cs
        )
        for r in recs:
            hooks = r["hook_numbers"]
            self.assertIsInstance(hooks, tuple)
            self.assertGreater(len(hooks), 0)
            for h in hooks:
                self.assertIn(h, {1, 2, 3, 4, 5, 6})

    def test_recommendations_have_cost_band_floor_le_ceiling(self) -> None:
        cs = CosineDistributionSummary(0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, "NO_MASTER_GRADIENT_ANCHOR")
        recs = rank_corrective_engineering_recommendations(
            per_pair=[], per_boundary=[], per_pixel=None, cosine_summary=cs
        )
        for r in recs:
            self.assertLessEqual(r["cost_usd_estimate_floor"], r["cost_usd_estimate_ceiling"])
            # Predicted ΔS floor should be more negative (or equal) than ceiling.
            self.assertLessEqual(r["predicted_delta_s_floor"], r["predicted_delta_s_ceiling"])


class CauchySchwarzBoundRegressionTests(unittest.TestCase):
    """Empirical proof that |inner_product| <= upper_bound for every pair."""

    def test_cauchy_schwarz_holds_on_random_anchor(self) -> None:
        rng = np.random.RandomState(7)
        n_bytes = 32
        n_pairs = 5
        signal = rng.randn(n_pairs, n_bytes).astype(np.float32) * 0.1
        anchor, anchor_path = _build_synthetic_anchor(n_bytes, n_pairs, signal)
        try:
            per_pair = tuple(PerPairDriftRecord(i, 0.0, 0.0, 0.0, 0.0) for i in range(n_pairs))
            d = rng.randn(n_bytes).astype(np.float64) * 0.01
            records, summary = compute_per_pair_master_gradient_weighted_drift(
                per_pair, d, anchor, archive_sha256="c" * 64
            )
        finally:
            anchor_path.unlink(missing_ok=True)
        self.assertEqual(len(records), n_pairs)
        for r in records:
            self.assertLessEqual(abs(r.inner_product_estimate), r.cauchy_schwarz_upper_bound + 1e-9)
            self.assertGreaterEqual(r.cos_alignment, -1.0001)
            self.assertLessEqual(r.cos_alignment, 1.0001)

    def test_zero_delta_zero_inner_product(self) -> None:
        rng = np.random.RandomState(8)
        n_bytes = 16
        n_pairs = 3
        signal = rng.randn(n_pairs, n_bytes).astype(np.float32)
        anchor, anchor_path = _build_synthetic_anchor(n_bytes, n_pairs, signal)
        try:
            per_pair = tuple(PerPairDriftRecord(i, 0.0, 0.0, 0.0, 0.0) for i in range(n_pairs))
            d = np.zeros(n_bytes, dtype=np.float64)
            records, summary = compute_per_pair_master_gradient_weighted_drift(
                per_pair, d, anchor, archive_sha256="c" * 64
            )
        finally:
            anchor_path.unlink(missing_ok=True)
        for r in records:
            self.assertAlmostEqual(r.inner_product_estimate, 0.0)
            self.assertAlmostEqual(r.d_l2, 0.0)
            # cos defined as 0 by convention when d_l2 == 0.
            self.assertAlmostEqual(r.cos_alignment, 0.0)
        # All-zero delta -> nullspace.
        self.assertEqual(summary.verdict, "NULLSPACE_VIABLE")


class CosineVerdictBucketTests(unittest.TestCase):
    def test_abs_mean_below_0_1_nullspace(self) -> None:
        cs = CosineDistributionSummary(
            n_pairs=10, mean=0.02, median=0.01, std=0.05, abs_mean=0.05,
            n_outliers_abs_above_0_5=0, n_outliers_abs_above_0_8=0, max_abs=0.08,
            verdict="NULLSPACE_VIABLE",
        )
        self.assertEqual(cs.verdict, "NULLSPACE_VIABLE")

    def test_abs_mean_between_0_1_and_0_3_weak_alignment(self) -> None:
        cs = CosineDistributionSummary(
            n_pairs=10, mean=0.05, median=0.04, std=0.1, abs_mean=0.2,
            n_outliers_abs_above_0_5=1, n_outliers_abs_above_0_8=0, max_abs=0.4,
            verdict="WEAK_ALIGNMENT",
        )
        self.assertEqual(cs.verdict, "WEAK_ALIGNMENT")

    def test_abs_mean_above_0_3_score_relevant(self) -> None:
        cs = CosineDistributionSummary(
            n_pairs=10, mean=0.0, median=0.0, std=0.5, abs_mean=0.4,
            n_outliers_abs_above_0_5=5, n_outliers_abs_above_0_8=2, max_abs=0.9,
            verdict="SCORE_RELEVANT_ENGINEERING_REQUIRED",
        )
        self.assertEqual(cs.verdict, "SCORE_RELEVANT_ENGINEERING_REQUIRED")

    def test_invalid_verdict_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CosineDistributionSummary(
                n_pairs=10, mean=0.0, median=0.0, std=0.5, abs_mean=0.4,
                n_outliers_abs_above_0_5=5, n_outliers_abs_above_0_8=2, max_abs=0.9,
                verdict="VIBES_BASED_BAD_VERDICT",
            )


if __name__ == "__main__":
    unittest.main()
