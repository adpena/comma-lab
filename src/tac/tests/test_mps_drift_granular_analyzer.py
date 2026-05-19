# SPDX-License-Identifier: MIT
"""Tests for tac.mps_diagnostic.granular_drift canonical analyzer.

Covers each of the 6 decomposition functions + the report builder + JSON
schema invariants. Each test uses synthetic inputs so the suite runs in
seconds and is reproducible without GPU access.
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

import torch  # noqa: F401  (verifies torch import availability)

from tac.mps_diagnostic.granular_drift import (
    DECOMPOSITION_KEYS,
    GRANULAR_DRIFT_AXIS_TAG,
    GRANULAR_DRIFT_EVIDENCE_GRADE,
    GRANULAR_DRIFT_REPORT_SCHEMA,
    CosineDistributionSummary,
    GranularDriftReport,
    PerBoundaryDriftRecord,
    PerByteDriftRecord,
    PerFrameDriftRecord,
    PerPairDriftRecord,
    PerPairMasterGradientWeightedRecord,
    PerPixelDriftRecord,
    build_granular_drift_report,
    compute_per_boundary_drift,
    compute_per_byte_drift,
    compute_per_frame_drift,
    compute_per_pair_drift,
    compute_per_pair_master_gradient_weighted_drift,
    compute_per_pixel_drift,
    rank_corrective_engineering_recommendations,
    report_to_json_dict,
    write_granular_drift_report,
)


class PerFrameDriftTests(unittest.TestCase):
    def test_zero_drift_recon_zero_aggregate(self) -> None:
        np.random.seed(0)
        recon = np.random.rand(3, 2, 3, 16, 24).astype(np.float32)
        records = compute_per_frame_drift(recon, recon.copy())
        self.assertEqual(len(records), 6)
        for r in records:
            self.assertAlmostEqual(r.pixel_l1, 0.0, places=6)
            self.assertAlmostEqual(r.aggregate, 0.0, places=6)

    def test_nonzero_drift_aggregate_positive(self) -> None:
        np.random.seed(1)
        mps = np.random.rand(2, 2, 3, 8, 10).astype(np.float32)
        cuda = mps + 0.01
        records = compute_per_frame_drift(mps, cuda)
        for r in records:
            self.assertGreater(r.pixel_l1, 0.0)
            self.assertGreater(r.aggregate, 0.0)

    def test_explicit_segnet_pose_overrides(self) -> None:
        recon = np.zeros((1, 2, 3, 4, 4), dtype=np.float32)
        seg = [0.1, 0.2]
        pose = [0.05, 0.07]
        records = compute_per_frame_drift(recon, recon, segnet_logit_l_inf_per_frame=seg, posenet_pose_l2_per_frame=pose)
        self.assertEqual(records[0].segnet_logit_l_inf, 0.1)
        self.assertEqual(records[1].segnet_logit_l_inf, 0.2)
        self.assertEqual(records[0].posenet_pose_l2, 0.05)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_per_frame_drift(
                np.zeros((1, 2, 3, 4, 4), dtype=np.float32),
                np.zeros((1, 2, 3, 4, 5), dtype=np.float32),
            )

    def test_wrong_ndim_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_per_frame_drift(
                np.zeros((1, 3, 4, 4), dtype=np.float32),
                np.zeros((1, 3, 4, 4), dtype=np.float32),
            )


class PerPixelDriftTests(unittest.TestCase):
    def test_zero_drift_zero_norms(self) -> None:
        recon = np.zeros((2, 2, 3, 4, 4), dtype=np.float32)
        rec = compute_per_pixel_drift(recon, recon)
        self.assertEqual(rec.l_inf, 0.0)
        self.assertEqual(rec.l_2, 0.0)
        self.assertEqual(rec.fraction_above_1e_3, 0.0)
        self.assertEqual(rec.backend_pair, ("mps", "cuda"))

    def test_constant_offset_drift(self) -> None:
        recon = np.zeros((2, 2, 3, 4, 4), dtype=np.float32)
        diffed = recon + 0.01
        rec = compute_per_pixel_drift(recon, diffed)
        self.assertAlmostEqual(rec.l_inf, 0.01, places=5)
        self.assertAlmostEqual(rec.mean_abs, 0.01, places=5)
        # All positions > threshold 1e-3.
        self.assertAlmostEqual(rec.fraction_above_1e_3, 1.0, places=5)

    def test_threshold_zero_drift(self) -> None:
        recon = np.zeros((1, 2, 3, 4, 4), dtype=np.float32)
        diffed = recon + 1e-6
        rec = compute_per_pixel_drift(recon, diffed)
        self.assertEqual(rec.fraction_above_1e_3, 0.0)


class PerBoundaryDriftTests(unittest.TestCase):
    def test_no_argmax_flips_zero_rates(self) -> None:
        logits = np.zeros((2, 5, 8, 8), dtype=np.float32)
        logits[..., 0, :, :] = 1.0  # class 0 wins everywhere
        records = compute_per_boundary_drift(logits, logits.copy(), boundary_band_px=2)
        self.assertEqual(len(records), 2)
        for r in records:
            self.assertEqual(r.n_argmax_flips_overall, 0)
            self.assertEqual(r.flip_rate_in_band, 0.0)

    def test_boundary_detection_finds_class_edge(self) -> None:
        # Construct CUDA-baseline logits where left half = class 0, right half = class 1.
        # The 2nd arg to compute_per_boundary_drift is the CUDA baseline, which is
        # where the boundary band is detected.
        cuda_logits = np.full((1, 5, 4, 4), -1.0, dtype=np.float32)
        cuda_logits[0, 0, :, :2] = 1.0  # class 0 wins on left
        cuda_logits[0, 1, :, 2:] = 1.0  # class 1 wins on right
        # MPS flips the right half back to class 0 -> drift exactly in the boundary band.
        mps_logits = cuda_logits.copy()
        mps_logits[0, 0, :, 2:] = 2.0  # flip right half to class 0 too
        records = compute_per_boundary_drift(mps_logits, cuda_logits, boundary_band_px=1)
        self.assertEqual(len(records), 1)
        r = records[0]
        # CUDA-baseline boundary exists (left-vs-right).
        self.assertGreater(r.n_boundary_pixels, 0)
        # MPS flips right-half class 1 -> class 0, producing many argmax flips.
        self.assertGreater(r.n_argmax_flips_overall, 0)
        # Some of the flips land in the boundary band (the column-2 pixels).
        self.assertGreater(r.n_argmax_flips_in_band, 0)

    def test_band_width_scaling(self) -> None:
        logits = np.full((1, 3, 6, 6), -1.0, dtype=np.float32)
        logits[0, 0, :, :3] = 1.0
        logits[0, 1, :, 3:] = 1.0
        records_1 = compute_per_boundary_drift(logits, logits.copy(), boundary_band_px=1)
        records_3 = compute_per_boundary_drift(logits, logits.copy(), boundary_band_px=3)
        self.assertGreater(records_3[0].n_boundary_pixels, records_1[0].n_boundary_pixels)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_per_boundary_drift(
                np.zeros((1, 3, 4, 4), dtype=np.float32),
                np.zeros((1, 3, 4, 5), dtype=np.float32),
            )

    def test_zero_band_width_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_per_boundary_drift(
                np.zeros((1, 3, 4, 4), dtype=np.float32),
                np.zeros((1, 3, 4, 4), dtype=np.float32),
                boundary_band_px=0,
            )


class PerByteDriftTests(unittest.TestCase):
    def test_lazy_mode_returns_empty(self, ) -> None:
        # Need an existing file but no offsets -> empty tuple.
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello world")
            path = Path(f.name)
        try:
            records = compute_per_byte_drift(path, lambda p: 0.0, lambda p: 0.0)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(records, ())

    def test_byte_mutation_record_built(self) -> None:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"\x00" * 16)
            path = Path(f.name)
        try:
            mps_calls = {"count": 0}
            cuda_calls = {"count": 0}
            def mps_fn(p):
                mps_calls["count"] += 1
                return 1.0 if "tmp_off" in p.name else 0.0
            def cuda_fn(p):
                cuda_calls["count"] += 1
                return 1.0 if "tmp_off" in p.name else 0.5
            records = compute_per_byte_drift(path, mps_fn, cuda_fn, byte_offsets=[0, 8])
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(len(records), 2)
        # baseline_mps=0, mutated_mps=1 -> delta=1; baseline_cuda=0.5, mutated_cuda=1 -> delta=0.5; drift=|1-0.5|=0.5
        self.assertAlmostEqual(records[0].mutation_delta_score_mps, 1.0)
        self.assertAlmostEqual(records[0].mutation_delta_score_cuda, 0.5)
        self.assertAlmostEqual(records[0].drift_at_byte, 0.5)

    def test_out_of_range_offset_raises(self) -> None:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"abc")
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                compute_per_byte_drift(path, lambda p: 0.0, lambda p: 0.0, byte_offsets=[10])
        finally:
            path.unlink(missing_ok=True)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            compute_per_byte_drift(Path("/nonexistent/file.zip"), lambda p: 0.0, lambda p: 0.0, byte_offsets=[0])


class PerPairDriftTests(unittest.TestCase):
    def test_zero_drift(self) -> None:
        scores = np.zeros((5, 3), dtype=np.float32)
        records = compute_per_pair_drift(scores, scores)
        self.assertEqual(len(records), 5)
        for r in records:
            self.assertEqual(r.aggregate_drift, 0.0)

    def test_nonzero_drift_aggregates(self) -> None:
        mps = np.zeros((3, 3), dtype=np.float32)
        cuda = np.array([[0.001, 0.002, 0.003], [0.002, 0.001, 0.0001], [0.005, 0.005, 0.005]], dtype=np.float32)
        records = compute_per_pair_drift(mps, cuda)
        # Aggregate per upstream/evaluate.py linearization: 100*seg + 10*pose + pixel
        self.assertAlmostEqual(records[0].aggregate_drift, 100 * 0.002 + 10 * 0.003 + 0.001, places=5)

    def test_wrong_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_per_pair_drift(np.zeros((3, 2), dtype=np.float32), np.zeros((3, 2), dtype=np.float32))


class PerPairMasterGradientTests(unittest.TestCase):
    def test_no_anchor_returns_fallback_verdict(self) -> None:
        per_pair = (PerPairDriftRecord(0, 0.0, 0.0, 0.0, 0.0),)
        records, summary = compute_per_pair_master_gradient_weighted_drift(per_pair, None, None, archive_sha256="0" * 64)
        self.assertEqual(records, ())
        self.assertEqual(summary.verdict, "NO_MASTER_GRADIENT_ANCHOR")

    def test_with_synthetic_anchor_inner_product(self) -> None:
        # Build a synthetic master-gradient anchor with controllable cos alignment.
        import tempfile
        from tac.master_gradient import (
            MasterGradient,
            OperatingPoint,
            PER_PAIR_GRADIENT_TENSOR_KIND,
        )
        n_bytes = 6
        n_pairs = 3
        rng = np.random.RandomState(42)
        # gradient shape (n_bytes, n_pairs, 3); we put a known signal in dim 0 (seg).
        g = np.zeros((n_bytes, n_pairs, 3), dtype=np.float32)
        # Pair 0: g_p in dim 0 aligned with d (high cos)
        g[:, 0, 0] = np.array([1, 2, 3, 4, 5, 6])
        # Pair 1: g_p in dim 0 anti-aligned (high cos magnitude, negative sign)
        g[:, 1, 0] = -g[:, 0, 0]
        # Pair 2: random small
        g[:, 2, 0] = rng.randn(n_bytes) * 0.01
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f, g)
            path = Path(f.name)
        try:
            anchor = MasterGradient(
                archive_sha256="a" * 64,
                operating_point=OperatingPoint(d_seg=0.001, d_pose=0.1, rate=1.0, score=0.2),
                gradient_array_path=str(path),
                n_bytes=n_bytes,
                measurement_method="synthetic_unit_test",
                measurement_axis="[macOS-CPU advisory]",
                measurement_hardware="darwin_arm64_advisory",
                measurement_call_id=None,
                measurement_utc="2026-05-19T12:00:00+00:00",
                gradient_tensor_kind=PER_PAIR_GRADIENT_TENSOR_KIND,
                n_pairs=n_pairs,
            )
            per_pair = (
                PerPairDriftRecord(0, 0.0, 0.0, 0.0, 0.0),
                PerPairDriftRecord(1, 0.0, 0.0, 0.0, 0.0),
                PerPairDriftRecord(2, 0.0, 0.0, 0.0, 0.0),
            )
            d = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
            records, summary = compute_per_pair_master_gradient_weighted_drift(
                per_pair, d, anchor, archive_sha256="a" * 64
            )
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(len(records), 3)
        # Pair 0: g_p collapsed via seg coefficient (100) is in d direction; cos = 1.
        self.assertAlmostEqual(records[0].cos_alignment, 1.0, places=4)
        self.assertAlmostEqual(records[1].cos_alignment, -1.0, places=4)
        # Cauchy-Schwarz: |inner| <= upper for all pairs.
        for r in records:
            self.assertLessEqual(abs(r.inner_product_estimate), r.cauchy_schwarz_upper_bound + 1e-9)
        # The summary's abs_mean should be high (pairs 0+1 saturate at 1.0 magnitude).
        self.assertGreater(summary.abs_mean, 0.5)
        self.assertEqual(summary.n_outliers_abs_above_0_5, 2)
        self.assertEqual(summary.verdict, "SCORE_RELEVANT_ENGINEERING_REQUIRED")

    def test_anchor_pair_mismatch_raises(self) -> None:
        import tempfile
        from tac.master_gradient import MasterGradient, OperatingPoint, PER_PAIR_GRADIENT_TENSOR_KIND
        g = np.zeros((4, 2, 3), dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f, g)
            path = Path(f.name)
        try:
            anchor = MasterGradient(
                archive_sha256="b" * 64,
                operating_point=OperatingPoint(d_seg=0.001, d_pose=0.1, rate=1.0, score=0.2),
                gradient_array_path=str(path),
                n_bytes=4,
                measurement_method="synthetic",
                measurement_axis="[macOS-CPU advisory]",
                measurement_hardware="darwin_arm64",
                measurement_call_id=None,
                measurement_utc="2026-05-19T12:00:00+00:00",
                gradient_tensor_kind=PER_PAIR_GRADIENT_TENSOR_KIND,
                n_pairs=2,
            )
            # per_pair list has 5 entries but anchor has 2.
            per_pair = tuple(PerPairDriftRecord(i, 0.0, 0.0, 0.0, 0.0) for i in range(5))
            d = np.zeros(4, dtype=np.float64)
            with self.assertRaises(ValueError):
                compute_per_pair_master_gradient_weighted_drift(per_pair, d, anchor, archive_sha256="b" * 64)
        finally:
            path.unlink(missing_ok=True)


class RecommendationRankerTests(unittest.TestCase):
    def test_all_recommendations_returned(self) -> None:
        cs = CosineDistributionSummary(
            n_pairs=10,
            mean=0.0,
            median=0.0,
            std=0.0,
            abs_mean=0.0,
            n_outliers_abs_above_0_5=0,
            n_outliers_abs_above_0_8=0,
            max_abs=0.0,
            verdict="NO_MASTER_GRADIENT_ANCHOR",
        )
        recs = rank_corrective_engineering_recommendations(
            per_pair=[], per_boundary=[], per_pixel=None, cosine_summary=cs
        )
        self.assertEqual(len(recs), 5)
        names = {r["name"] for r in recs}
        self.assertEqual(
            names,
            {
                "selective_parameter_freeze",
                "subspace_alignment_topK_eigenvectors",
                "per_frame_routing_high_drift_to_cuda_shadow",
                "cross_device_validation_cadence_every_K_steps",
                "boundary_smoothing_3px_gaussian_pre_argmax",
            },
        )

    def test_selective_freeze_triggers_on_score_relevant(self) -> None:
        cs = CosineDistributionSummary(
            n_pairs=10,
            mean=0.0,
            median=0.0,
            std=0.0,
            abs_mean=0.5,
            n_outliers_abs_above_0_5=8,
            n_outliers_abs_above_0_8=4,
            max_abs=0.9,
            verdict="SCORE_RELEVANT_ENGINEERING_REQUIRED",
        )
        recs = rank_corrective_engineering_recommendations(
            per_pair=[], per_boundary=[], per_pixel=None, cosine_summary=cs
        )
        by_name = {r["name"]: r for r in recs}
        self.assertTrue(by_name["selective_parameter_freeze"]["triggered"])
        self.assertTrue(by_name["subspace_alignment_topK_eigenvectors"]["triggered"])

    def test_boundary_smoothing_triggers_on_high_band_ratio(self) -> None:
        cs = CosineDistributionSummary(0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, "NO_MASTER_GRADIENT_ANCHOR")
        # 4 of 10 pairs have band_rate > 2x overall_rate.
        per_boundary = []
        for i in range(10):
            if i < 4:
                per_boundary.append(PerBoundaryDriftRecord(i, 3, 100, 30, 0.3, 5, 0.01))
            else:
                per_boundary.append(PerBoundaryDriftRecord(i, 3, 100, 1, 0.01, 1, 0.005))
        recs = rank_corrective_engineering_recommendations(
            per_pair=[], per_boundary=per_boundary, per_pixel=None, cosine_summary=cs
        )
        by_name = {r["name"]: r for r in recs}
        self.assertTrue(by_name["boundary_smoothing_3px_gaussian_pre_argmax"]["triggered"])


class GranularReportBuilderTests(unittest.TestCase):
    def test_end_to_end_canonical_report(self) -> None:
        np.random.seed(0)
        mps = np.random.rand(4, 2, 3, 16, 24).astype(np.float32)
        cuda = mps + 0.001
        report = build_granular_drift_report(
            mps_artifact_path="/tmp/mps.pt",
            cuda_artifact_path="/tmp/cuda.pt",
            mps_recon=mps,
            cuda_recon=cuda,
        )
        self.assertEqual(report.schema_version, GRANULAR_DRIFT_REPORT_SCHEMA)
        self.assertEqual(report.evidence_grade, GRANULAR_DRIFT_EVIDENCE_GRADE)
        self.assertEqual(report.axis_tag, GRANULAR_DRIFT_AXIS_TAG)
        self.assertFalse(report.score_claim)
        self.assertFalse(report.promotion_eligible)
        self.assertFalse(report.ready_for_exact_eval_dispatch)
        self.assertEqual(len(report.per_frame), 4 * 2)
        self.assertEqual(len(report.per_pixel), 1)
        self.assertEqual(len(report.per_pair), 4)
        self.assertEqual(report.cosine_distribution_summary.verdict, "NO_MASTER_GRADIENT_ANCHOR")
        self.assertEqual(len(report.summary_corrective_engineering_recommendations), 5)

    def test_report_to_json_dict_round_trip(self) -> None:
        np.random.seed(0)
        mps = np.zeros((2, 2, 3, 8, 8), dtype=np.float32)
        cuda = mps.copy()
        report = build_granular_drift_report(
            mps_artifact_path="/tmp/mps.pt",
            cuda_artifact_path="/tmp/cuda.pt",
            mps_recon=mps,
            cuda_recon=cuda,
        )
        payload = report_to_json_dict(report)
        # JSON-serializable.
        text = json.dumps(payload, sort_keys=True)
        roundtripped = json.loads(text)
        self.assertEqual(roundtripped["schema_version"], GRANULAR_DRIFT_REPORT_SCHEMA)
        self.assertEqual(roundtripped["evidence_grade"], GRANULAR_DRIFT_EVIDENCE_GRADE)
        self.assertFalse(roundtripped["score_claim"])
        for key in DECOMPOSITION_KEYS:
            self.assertIn(key, roundtripped)

    def test_write_granular_drift_report_atomic(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            mps = np.zeros((1, 2, 3, 4, 4), dtype=np.float32)
            cuda = mps.copy()
            report = build_granular_drift_report(
                mps_artifact_path="/tmp/mps.pt",
                cuda_artifact_path="/tmp/cuda.pt",
                mps_recon=mps,
                cuda_recon=cuda,
            )
            out = Path(tmpdir) / "subdir" / "report.json"
            written = write_granular_drift_report(report, out)
            self.assertTrue(written.exists())
            payload = json.loads(written.read_text())
            self.assertEqual(payload["schema_version"], GRANULAR_DRIFT_REPORT_SCHEMA)


class DataclassInvariantTests(unittest.TestCase):
    def test_per_frame_invariants(self) -> None:
        with self.assertRaises(ValueError):
            PerFrameDriftRecord(-1, 0, 0.0, 0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            PerFrameDriftRecord(0, 5, 0.0, 0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            PerFrameDriftRecord(0, 0, float("nan"), 0.0, 0.0, 0.0)

    def test_per_pixel_invariants(self) -> None:
        with self.assertRaises(ValueError):
            PerPixelDriftRecord("", ("mps", "cuda"), 0.0, 0.0, 0.0, (1, 2, 3), 0.0)
        with self.assertRaises(ValueError):
            PerPixelDriftRecord("layer", ("mps",), 0.0, 0.0, 0.0, (1, 2, 3), 0.0)
        with self.assertRaises(ValueError):
            PerPixelDriftRecord("layer", ("mps", "cuda"), 0.0, 0.0, 0.0, (1, 2, 3), 1.5)

    def test_per_boundary_invariants(self) -> None:
        with self.assertRaises(ValueError):
            PerBoundaryDriftRecord(0, 0, 1, 0, 0.0, 0, 0.0)
        with self.assertRaises(ValueError):
            PerBoundaryDriftRecord(0, 1, -1, 0, 0.0, 0, 0.0)

    def test_per_byte_invariants(self) -> None:
        with self.assertRaises(ValueError):
            PerByteDriftRecord(-1, 0.0, 0.0, 0.0, "section")
        with self.assertRaises(ValueError):
            PerByteDriftRecord(0, 0.0, 0.0, -0.1, "section")

    def test_cosine_summary_verdict_invariant(self) -> None:
        with self.assertRaises(ValueError):
            CosineDistributionSummary(0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, "BOGUS_VERDICT")

    def test_report_schema_version_invariant(self) -> None:
        with self.assertRaises(ValueError):
            GranularDriftReport(
                schema_version="bad_version",
                evidence_grade=GRANULAR_DRIFT_EVIDENCE_GRADE,
                axis_tag=GRANULAR_DRIFT_AXIS_TAG,
                score_claim=False,
                promotion_eligible=False,
                ready_for_exact_eval_dispatch=False,
                mps_artifact_path="/tmp/a",
                cuda_artifact_path="/tmp/b",
                n_pairs=0,
                per_frame=(),
                per_pixel=(),
                per_boundary=(),
                per_byte=(),
                per_pair=(),
                per_pair_master_gradient=(),
                cosine_distribution_summary=CosineDistributionSummary(
                    0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, "NO_MASTER_GRADIENT_ANCHOR"
                ),
                summary_aggregate_relative_drift=0.0,
                summary_aggregate_drift_concentrated_in_pairs_above_p95=False,
                summary_drift_cliff_layer=None,
                summary_drift_concentrated_in_boundaries=False,
                summary_corrective_engineering_recommendations=(),
            )

    def test_report_score_claim_must_be_false(self) -> None:
        cs = CosineDistributionSummary(0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, "NO_MASTER_GRADIENT_ANCHOR")
        with self.assertRaises(ValueError):
            GranularDriftReport(
                schema_version=GRANULAR_DRIFT_REPORT_SCHEMA,
                evidence_grade=GRANULAR_DRIFT_EVIDENCE_GRADE,
                axis_tag=GRANULAR_DRIFT_AXIS_TAG,
                score_claim=True,  # forbidden
                promotion_eligible=False,
                ready_for_exact_eval_dispatch=False,
                mps_artifact_path="/tmp/a",
                cuda_artifact_path="/tmp/b",
                n_pairs=0,
                per_frame=(),
                per_pixel=(),
                per_boundary=(),
                per_byte=(),
                per_pair=(),
                per_pair_master_gradient=(),
                cosine_distribution_summary=cs,
                summary_aggregate_relative_drift=0.0,
                summary_aggregate_drift_concentrated_in_pairs_above_p95=False,
                summary_drift_cliff_layer=None,
                summary_drift_concentrated_in_boundaries=False,
                summary_corrective_engineering_recommendations=(),
            )

    def test_report_evidence_grade_must_be_research_signal(self) -> None:
        cs = CosineDistributionSummary(0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, "NO_MASTER_GRADIENT_ANCHOR")
        with self.assertRaises(ValueError):
            GranularDriftReport(
                schema_version=GRANULAR_DRIFT_REPORT_SCHEMA,
                evidence_grade="[contest-CUDA]",  # FORBIDDEN per non-promotability
                axis_tag=GRANULAR_DRIFT_AXIS_TAG,
                score_claim=False,
                promotion_eligible=False,
                ready_for_exact_eval_dispatch=False,
                mps_artifact_path="/tmp/a",
                cuda_artifact_path="/tmp/b",
                n_pairs=0,
                per_frame=(),
                per_pixel=(),
                per_boundary=(),
                per_byte=(),
                per_pair=(),
                per_pair_master_gradient=(),
                cosine_distribution_summary=cs,
                summary_aggregate_relative_drift=0.0,
                summary_aggregate_drift_concentrated_in_pairs_above_p95=False,
                summary_drift_cliff_layer=None,
                summary_drift_concentrated_in_boundaries=False,
                summary_corrective_engineering_recommendations=(),
            )


class CLISmokeTests(unittest.TestCase):
    def test_cli_help_smoke(self) -> None:
        import subprocess
        result = subprocess.run(
            [sys.executable, "tools/analyze_mps_drift_granular.py", "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("analyze_mps_drift_granular", result.stdout)
        self.assertIn("--mps-forward-outputs", result.stdout)


if __name__ == "__main__":
    unittest.main()
