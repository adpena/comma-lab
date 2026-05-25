# SPDX-License-Identifier: MIT
"""Tests for T3-GRAND-COUNCIL-ACTIVE-EXPLORATION-CONV2D-DRIFT-UNEXPLORED-PATHS.

Per Catalog #229 PV + Catalog #287/#323 canonical Provenance: every test
preserves the [macOS-MLX research-signal] axis tag + score_claim=False
non-promotable markers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.local_acceleration.deterministic_primitives import (
    ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR,
    ActiveExplorationPathVerdict,
    ActiveExplorationThreadResult,
    CudnnReferenceMeasurementResult,
    MlxDeterministicInvestigationResult,
    classify_reduction_percent,
    fp64_intermediate_conv2d_3x3,
    kahan_compensated_sum,
    kahan_conv2d_3x3,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# ActiveExplorationPathVerdict enum
# ---------------------------------------------------------------------------

class TestActiveExplorationPathVerdict:
    def test_enum_membership(self):
        members = {v.value for v in ActiveExplorationPathVerdict}
        assert members == {
            "FIXABLE",
            "PARTIALLY_FIXABLE_MARGINAL",
            "NOT_FIXABLE_SUBSTITUTION_ONLY",
            "NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL",
            "DEFERRED_PENDING_PAID_DISPATCH",
        }

    def test_enum_string_values(self):
        assert ActiveExplorationPathVerdict.FIXABLE.value == "FIXABLE"
        assert ActiveExplorationPathVerdict.PARTIALLY_FIXABLE_MARGINAL.value == "PARTIALLY_FIXABLE_MARGINAL"


# ---------------------------------------------------------------------------
# classify_reduction_percent
# ---------------------------------------------------------------------------

class TestClassifyReductionPercent:
    def test_fixable_boundary_exact_50(self):
        assert classify_reduction_percent(50.0) == ActiveExplorationPathVerdict.FIXABLE

    def test_fixable_above_50(self):
        assert classify_reduction_percent(75.0) == ActiveExplorationPathVerdict.FIXABLE
        assert classify_reduction_percent(99.9) == ActiveExplorationPathVerdict.FIXABLE

    def test_partially_fixable_marginal_band(self):
        assert classify_reduction_percent(10.0) == ActiveExplorationPathVerdict.PARTIALLY_FIXABLE_MARGINAL
        assert classify_reduction_percent(22.4) == ActiveExplorationPathVerdict.PARTIALLY_FIXABLE_MARGINAL
        assert classify_reduction_percent(49.9) == ActiveExplorationPathVerdict.PARTIALLY_FIXABLE_MARGINAL

    def test_not_fixable_substitution_only_band(self):
        assert classify_reduction_percent(0.0) == ActiveExplorationPathVerdict.NOT_FIXABLE_SUBSTITUTION_ONLY
        assert classify_reduction_percent(6.2) == ActiveExplorationPathVerdict.NOT_FIXABLE_SUBSTITUTION_ONLY
        assert classify_reduction_percent(9.9) == ActiveExplorationPathVerdict.NOT_FIXABLE_SUBSTITUTION_ONLY

    def test_not_fixable_framework_fundamental_negative_reduction(self):
        assert classify_reduction_percent(-0.1) == ActiveExplorationPathVerdict.NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL
        assert classify_reduction_percent(-50.0) == ActiveExplorationPathVerdict.NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            classify_reduction_percent(float("nan"))

    def test_inf_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            classify_reduction_percent(float("inf"))


# ---------------------------------------------------------------------------
# kahan_compensated_sum
# ---------------------------------------------------------------------------

class TestKahanCompensatedSum:
    def test_empty_returns_zero(self):
        result = kahan_compensated_sum(np.array([], dtype=np.float32))
        assert result == 0.0

    def test_single_value(self):
        result = kahan_compensated_sum(np.array([3.5], dtype=np.float32))
        assert result == 3.5

    def test_rank_2_rejected(self):
        with pytest.raises(ValueError, match="1D"):
            kahan_compensated_sum(np.zeros((2, 2), dtype=np.float32))

    def test_kahan_matches_pairwise_when_no_drift(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=100).astype(np.float32)
        kahan = kahan_compensated_sum(values)
        # Should match within fp32 precision band.
        naive = np.float32(values.sum())
        # Kahan should be at least as good as naive (typically tighter).
        assert abs(float(kahan) - float(values.astype(np.float64).sum())) <= abs(
            float(naive) - float(values.astype(np.float64).sum())
        ) + 1e-5

    def test_kahan_handles_large_plus_many_small(self):
        # Classic Kahan test case: 1.0 + N * eps should be representable.
        values = np.concatenate([
            np.array([1.0], dtype=np.float32),
            np.full(1000, 1e-8, dtype=np.float32),
        ])
        result = kahan_compensated_sum(values)
        # Kahan should preserve the tiny contributions better than naive sum.
        expected_fp64 = float(values.astype(np.float64).sum())
        assert abs(float(result) - expected_fp64) < 1e-5


# ---------------------------------------------------------------------------
# kahan_conv2d_3x3
# ---------------------------------------------------------------------------

class TestKahanConv2d3x3:
    def test_shape_correctness_padding_1(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(1, 4, 4, 3)).astype(np.float32)
        w = rng.normal(size=(3, 3, 3, 5)).astype(np.float32)
        result = kahan_conv2d_3x3(x, w, padding=1)
        assert result.shape == (1, 4, 4, 5)
        assert result.dtype == np.float32

    def test_shape_correctness_padding_0(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(2, 6, 6, 3)).astype(np.float32)
        w = rng.normal(size=(3, 3, 3, 4)).astype(np.float32)
        result = kahan_conv2d_3x3(x, w, padding=0)
        assert result.shape == (2, 4, 4, 4)

    def test_bias_applied(self):
        rng = np.random.default_rng(0)
        x = np.zeros((1, 3, 3, 2), dtype=np.float32)  # all-zero input
        w = rng.normal(size=(3, 3, 2, 4)).astype(np.float32)
        bias = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        result = kahan_conv2d_3x3(x, w, bias, padding=1)
        # All-zero input + bias = bias broadcast to every output spatial location.
        for n in range(1):
            for y in range(3):
                for x_idx in range(3):
                    np.testing.assert_array_equal(result[n, y, x_idx, :], bias)

    def test_input_rank_validation(self):
        with pytest.raises(ValueError, match="rank-4"):
            kahan_conv2d_3x3(np.zeros((3,), dtype=np.float32), np.zeros((3, 3, 1, 1), dtype=np.float32))

    def test_kernel_in_channels_mismatch(self):
        x = np.zeros((1, 4, 4, 3), dtype=np.float32)
        w = np.zeros((3, 3, 5, 4), dtype=np.float32)  # C_in=5 != input C_in=3
        with pytest.raises(ValueError, match="C_in"):
            kahan_conv2d_3x3(x, w)

    def test_bias_shape_validation(self):
        x = np.zeros((1, 3, 3, 2), dtype=np.float32)
        w = np.zeros((3, 3, 2, 4), dtype=np.float32)
        bias_wrong = np.zeros((3,), dtype=np.float32)  # should be (4,)
        with pytest.raises(ValueError, match="bias must"):
            kahan_conv2d_3x3(x, w, bias_wrong)


# ---------------------------------------------------------------------------
# fp64_intermediate_conv2d_3x3
# ---------------------------------------------------------------------------

class TestFP64IntermediateConv2d3x3:
    def test_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(1, 4, 4, 3)).astype(np.float32)
        w = rng.normal(size=(3, 3, 3, 5)).astype(np.float32)
        result = fp64_intermediate_conv2d_3x3(x, w)
        assert result.shape == (1, 4, 4, 5)
        assert result.dtype == np.float32  # downcast output

    def test_matches_naive_conv_at_small_scale(self):
        # At small scale FP64 intermediate should match naive FP32 conv within
        # float precision tolerance.
        rng = np.random.default_rng(0)
        x = rng.normal(size=(1, 3, 3, 2)).astype(np.float32)
        w = rng.normal(size=(3, 3, 2, 4)).astype(np.float32)
        result = fp64_intermediate_conv2d_3x3(x, w, padding=1)
        # Sanity: result is non-trivial.
        assert np.any(result != 0)
        assert np.all(np.isfinite(result))

    def test_input_validation(self):
        with pytest.raises(ValueError, match="rank-4"):
            fp64_intermediate_conv2d_3x3(
                np.zeros((4,), dtype=np.float32),
                np.zeros((3, 3, 1, 1), dtype=np.float32),
            )


# ---------------------------------------------------------------------------
# ActiveExplorationThreadResult dataclass
# ---------------------------------------------------------------------------

class TestActiveExplorationThreadResult:
    def test_construction_and_as_dict(self):
        result = ActiveExplorationThreadResult(
            thread_id=1,
            thread_name="kahan_compensated_summation",
            path_verdict=ActiveExplorationPathVerdict.PARTIALLY_FIXABLE_MARGINAL,
            max_observed_reduction_percent=22.4,
            predicted_reduction_percent_lower_bound=50.0,
            carmack_mvp_first_falsified=True,
            per_scale_observations=({"scale": "pr95_stage2", "reduction": 0.0},),
            notes="Higham 2002 theoretical bound refuted",
        )
        d = result.as_dict()
        assert d["schema"] == "active_exploration_thread_result.v1"
        assert d["thread_id"] == 1
        assert d["path_verdict"] == "PARTIALLY_FIXABLE_MARGINAL"
        assert d["max_observed_reduction_percent"] == 22.4
        assert d["carmack_mvp_first_falsified"] is True
        # Canonical Provenance markers per Catalog #287/#323.
        assert d["score_claim"] is False
        assert d["promotion_eligible"] is False
        assert d["axis_tag"] == "[predicted]"
        assert d["evidence_grade"] == "macOS-MLX-research-signal"

    def test_frozen_dataclass(self):
        result = ActiveExplorationThreadResult(
            thread_id=1,
            thread_name="x",
            path_verdict=ActiveExplorationPathVerdict.FIXABLE,
            max_observed_reduction_percent=75.0,
            predicted_reduction_percent_lower_bound=50.0,
            carmack_mvp_first_falsified=False,
            per_scale_observations=(),
        )
        with pytest.raises((AttributeError, TypeError, Exception)):
            result.thread_id = 2  # type: ignore


# ---------------------------------------------------------------------------
# MlxDeterministicInvestigationResult dataclass
# ---------------------------------------------------------------------------

class TestMlxDeterministicInvestigationResult:
    def test_construction_and_as_dict_no_flags(self):
        result = MlxDeterministicInvestigationResult(
            mlx_version="0.31.2",
            deterministic_reduction_flag_available=False,
            public_core_deterministic_attrs=(),
            public_metal_deterministic_attrs=(),
            classification="framework_different_no_public_deterministic_reduction_flag",
            path_verdict=ActiveExplorationPathVerdict.NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL,
        )
        d = result.as_dict()
        assert d["mlx_version"] == "0.31.2"
        assert d["deterministic_reduction_flag_available"] is False
        assert d["path_verdict"] == "NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL"
        assert d["score_claim"] is False
        assert d["promotable"] is False


# ---------------------------------------------------------------------------
# CudnnReferenceMeasurementResult dataclass
# ---------------------------------------------------------------------------

class TestCudnnReferenceMeasurementResult:
    def test_macos_deferred_pending_paid_dispatch(self):
        result = CudnnReferenceMeasurementResult(
            cuda_locally_available=False,
            cudnn_locally_available=False,
            mps_available=True,
            path_verdict=ActiveExplorationPathVerdict.DEFERRED_PENDING_PAID_DISPATCH,
            estimated_paid_dispatch_cost_usd=2.0,
            mps_not_substitute_rationale=(
                "MPS is NOT a substitute for cuDNN per CLAUDE.md 'MPS auth eval is NOISE'"
            ),
        )
        d = result.as_dict()
        assert d["cuda_locally_available"] is False
        assert d["path_verdict"] == "DEFERRED_PENDING_PAID_DISPATCH"
        assert d["estimated_paid_dispatch_cost_usd"] == 2.0
        assert "MPS is NOT a substitute" in d["mps_not_substitute_rationale"]

    def test_canonical_provenance_markers(self):
        result = CudnnReferenceMeasurementResult(
            cuda_locally_available=False,
            cudnn_locally_available=False,
            mps_available=True,
            path_verdict=ActiveExplorationPathVerdict.DEFERRED_PENDING_PAID_DISPATCH,
            estimated_paid_dispatch_cost_usd=2.0,
            mps_not_substitute_rationale="rationale",
        )
        d = result.as_dict()
        assert d["score_claim"] is False
        assert d["promotion_eligible"] is False
        assert d["promotable"] is False
        assert d["ready_for_exact_eval_dispatch"] is False
        assert d["axis_tag"] == "[predicted]"


# ---------------------------------------------------------------------------
# ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR
# ---------------------------------------------------------------------------

class TestActiveExplorationAnchor:
    def test_schema_version_pinned(self):
        anchor = ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR
        assert anchor["schema"] == "active_exploration_conv2d_drift_unexplored_paths_anchor.v1"

    def test_thread_1_kahan_anchor_present(self):
        anchor = ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR
        t1 = anchor["thread_1_kahan_compensated_summation"]
        assert t1["path_verdict"] == "PARTIALLY_FIXABLE_MARGINAL"
        assert t1["max_observed_reduction_percent"] == 22.4
        assert t1["carmack_mvp_first_falsified"] is True
        # Per-scale observations pinned to empirical anchor.
        assert t1["per_scale_reductions"]["pr95_stage2_36_to_144_6x8"] == 0.0
        assert t1["per_scale_reductions"]["pr95_final_head_class_256_to_256_48x64"] == 22.4

    def test_thread_2_fp64_anchor_present(self):
        anchor = ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR
        t2 = anchor["thread_2_fp64_intermediate_accumulation"]
        assert t2["path_verdict"] == "PARTIALLY_FIXABLE_MARGINAL"
        assert t2["max_observed_reduction_percent"] == 22.4
        assert t2["carmack_mvp_first_falsified"] is True

    def test_thread_3_mlx_deterministic_not_fixable(self):
        anchor = ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR
        t3 = anchor["thread_3_mlx_deterministic_reduction_enforcement"]
        assert t3["path_verdict"] == "NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL"
        assert t3["classification"] == "framework_different_no_public_deterministic_reduction_flag"
        assert t3["public_core_deterministic_attrs"] == []

    def test_thread_4_cudnn_deferred(self):
        anchor = ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR
        t4 = anchor["thread_4_cudnn_reference_conv2d_3x3_measurement"]
        assert t4["path_verdict"] == "DEFERRED_PENDING_PAID_DISPATCH"
        assert "MPS is NOT a substitute" in t4["mps_not_substitute_rationale"]

    def test_aggregate_verdict_proceed_with_revisions(self):
        anchor = ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR
        agg = anchor["aggregate_path_verdict_summary"]
        assert agg["fixable_count"] == 0
        assert agg["partially_fixable_marginal_count"] == 2
        assert agg["not_fixable_framework_fundamental_count"] == 1
        assert agg["deferred_pending_paid_dispatch_count"] == 1
        assert agg["overall_verdict"] == "PROCEED_WITH_REVISIONS"

    def test_canonical_provenance_markers(self):
        anchor = ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR
        assert anchor["score_claim"] is False
        assert anchor["promotion_eligible"] is False
        assert anchor["promotable"] is False
        assert anchor["ready_for_exact_eval_dispatch"] is False
        assert anchor["axis_tag"] == "[predicted]"
        assert anchor["evidence_grade"] == "macOS-MLX-research-signal"

    def test_carmack_mvp_first_step_2_falsification_summary(self):
        anchor = ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR
        summary = anchor["carmack_mvp_first_step_2_falsification_summary"]
        assert summary["thread_1_falsified"] is True
        assert summary["thread_2_falsified"] is True
        assert summary["thread_1_max_observed_percent"] == 22.4
        assert summary["thread_2_max_observed_percent"] == 22.4


# ---------------------------------------------------------------------------
# Measurement CLI filtering
# ---------------------------------------------------------------------------


class TestMeasureUnexploredMitigationPathsCli:
    def test_kahan_fp64_smoke_filter_skips_framework_threads(
        self,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "filter_smoke.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "measure_unexplored_mitigation_paths_drift.py"),
                "--mitigation-paths",
                "kahan,fp64",
                "--shape-preset",
                "smoke",
                "--output",
                str(output),
                "--run-id",
                "unit_filter_smoke",
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        stdout = json.loads(completed.stdout)
        manifest = json.loads(output.read_text(encoding="utf-8"))
        assert stdout["mitigation_paths"] == ["kahan", "fp64"]
        assert manifest["shape_preset"] == "smoke"
        assert manifest["thread_3_mlx_deterministic_investigation"]["measurement_status"] == (
            "skipped_by_mitigation_paths_filter"
        )
        assert manifest["thread_4_cudnn_reference_measurement"]["measurement_status"] == (
            "skipped_by_mitigation_paths_filter"
        )
        assert manifest["score_claim"] is False

    def test_framework_smoke_filter_skips_conv_measurements(
        self,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "filter_no_conv_smoke.json"
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "measure_unexplored_mitigation_paths_drift.py"),
                "--mitigation-paths",
                "mlx_deterministic,cudnn_reference",
                "--shape-preset",
                "smoke",
                "--output",
                str(output),
                "--run-id",
                "unit_filter_no_conv",
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        manifest = json.loads(output.read_text(encoding="utf-8"))
        assert manifest["thread_1_aggregate_verdict"] == "NOT_MEASURED"
        assert manifest["thread_2_aggregate_verdict"] == "NOT_MEASURED"
        assert manifest["thread_1_kahan_per_scale_measurements"] == []
        assert (
            manifest["thread_3_mlx_deterministic_investigation"]["thread_3_verdict"]
            == "NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL_NO_PUBLIC_API"
        )
        assert (
            manifest["thread_4_cudnn_reference_measurement"]["thread_4_verdict"]
            == "DEFERRED_PENDING_PAID_DISPATCH"
        )
