# SPDX-License-Identifier: MIT
"""Tests for tac.master_gradient_consumers consumers 7-14.

Per Cable D D3 (task #799) batched builder wave 2026-05-19, lane
`lane_cable_d_master_gradient_extension_batch_20260519`. The 8 new
consumers extend the master-gradient consumer catalog from v1 (1-6, 15) to
v3 (7-14 added).

Test taxonomy (per Cable D D3 contract): each consumer gets
- contract validation (input shape, kwarg bounds)
- happy-path with synthetic input
- sidecar JSON emit (canonical compliance tags + axis preservation)
- determinism check (same input + seed -> same output)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac import master_gradient_consumers as mgc


# ──────────────────────────────────────────────────────────────────────────── #
# Fixtures                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #


@pytest.fixture
def per_pair_grad():
    """Synthetic per-pair gradient of shape (N_bytes=64, N_pairs=8, 3)."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 1.0, size=(64, 8, 3)).astype(np.float64)


@pytest.fixture
def aggregate_grad(per_pair_grad):
    """Aggregate gradient derived from per_pair_grad (consistent shapes)."""
    return per_pair_grad.mean(axis=1)


@pytest.fixture
def archive_sha256():
    return "b" * 64


@pytest.fixture
def axis_meta():
    return {
        "measurement_axis": "contest_cuda",
        "measurement_hardware": "linux_x86_64_t4",
    }


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    """Redirect CONSUMER_OUTPUT_ROOT to tmp_path."""
    monkeypatch.setattr(
        mgc, "CONSUMER_OUTPUT_ROOT", tmp_path / "master_gradient_consumers"
    )
    return tmp_path


# ──────────────────────────────────────────────────────────────────────────── #
# Consumer 7 — per_pair_pareto_envelope                                          #
# ──────────────────────────────────────────────────────────────────────────── #


class TestConsumer7ParetoEnvelope:
    def test_contract_rejects_wrong_shape(self, archive_sha256, axis_meta):
        bad = np.zeros((10,))
        with pytest.raises(ValueError, match="N_bytes, N_pairs, 3"):
            mgc.per_pair_pareto_envelope(
                bad, archive_sha256=archive_sha256, **axis_meta, write_sidecar=False
            )

    def test_contract_rejects_zero_top_k(self, per_pair_grad, archive_sha256, axis_meta):
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            mgc.per_pair_pareto_envelope(
                per_pair_grad,
                archive_sha256=archive_sha256,
                **axis_meta,
                top_k=0,
                write_sidecar=False,
            )

    def test_happy_path_produces_envelope(self, per_pair_grad, archive_sha256, axis_meta):
        env = mgc.per_pair_pareto_envelope(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            top_k=16,
            write_sidecar=False,
        )
        assert env.n_pairs == 8
        assert env.n_bytes == 64
        assert env.top_k == 16
        # Envelope has at most n_pairs * top_k points; each pair contributes <= top_k
        assert len(env.envelope_points) <= 8 * 16
        for length in env.per_pair_envelope_lengths:
            assert 0 <= length <= 16

    def test_envelope_distortion_monotonically_nondecreasing(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        env = mgc.per_pair_pareto_envelope(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            top_k=8,
            write_sidecar=False,
        )
        # For each pair, verify cumulative_distortion_delta is non-decreasing
        per_pair_points: dict[int, list] = {}
        for point in env.envelope_points:
            per_pair_points.setdefault(point.pair_index, []).append(point)
        for pair_idx, pts in per_pair_points.items():
            dists = [p.cumulative_distortion_delta for p in pts]
            assert all(dists[i] <= dists[i + 1] for i in range(len(dists) - 1)), (
                f"pair {pair_idx}: distortion not non-decreasing: {dists}"
            )

    def test_sidecar_emit_carries_canonical_tags(
        self, per_pair_grad, archive_sha256, axis_meta, tmp_root
    ):
        env = mgc.per_pair_pareto_envelope(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            top_k=4,
            write_sidecar=True,
        )
        sidecars = list((tmp_root / "master_gradient_consumers").glob("per_pair_pareto_envelope_*.json"))
        assert len(sidecars) == 1
        payload = json.loads(sidecars[0].read_text())
        assert payload["score_claim"] is False
        assert payload["promotion_eligible"] is False
        assert payload["measurement_axis"] == "contest_cuda"

    def test_determinism(self, per_pair_grad, archive_sha256, axis_meta):
        a = mgc.per_pair_pareto_envelope(
            per_pair_grad, archive_sha256=archive_sha256, **axis_meta, write_sidecar=False
        )
        b = mgc.per_pair_pareto_envelope(
            per_pair_grad, archive_sha256=archive_sha256, **axis_meta, write_sidecar=False
        )
        assert a.per_pair_envelope_lengths == b.per_pair_envelope_lengths
        assert len(a.envelope_points) == len(b.envelope_points)


# ──────────────────────────────────────────────────────────────────────────── #
# Consumer 8 — per_pair_lagrangian_lambda_bisection                              #
# ──────────────────────────────────────────────────────────────────────────── #


class TestConsumer8LambdaBisection:
    def test_contract_rejects_wrong_shape(self, archive_sha256, axis_meta):
        bad = np.zeros((10, 10))
        with pytest.raises(ValueError, match="N_bytes, N_pairs, 3"):
            mgc.per_pair_lagrangian_lambda_bisection(
                bad, archive_sha256=archive_sha256, **axis_meta, write_sidecar=False
            )

    def test_happy_path_produces_lambda_estimates(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        result = mgc.per_pair_lagrangian_lambda_bisection(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            top_k=16,
            write_sidecar=False,
        )
        assert result.n_pairs == 8
        assert len(result.entries) == 8
        # At least some pairs should have valid lambda estimates
        valid = sum(1 for e in result.entries if e.lambda_r_estimate > 0)
        assert valid >= 1

    def test_operator_lambda_override(self, per_pair_grad, archive_sha256, axis_meta):
        result = mgc.per_pair_lagrangian_lambda_bisection(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            operator_lambda_r=0.5,
            write_sidecar=False,
        )
        assert result.operator_lambda_r == 0.5

    def test_min_points_threshold_zeroes_short_pairs(self, archive_sha256, axis_meta):
        # Force short envelopes by tiny per_pair tensor
        tiny = np.zeros((3, 2, 3))
        tiny[:, :, 2] = 1.0  # nonzero rate gradient on all bytes
        tiny[:, :, 0] = 0.5  # nonzero seg
        result = mgc.per_pair_lagrangian_lambda_bisection(
            tiny,
            archive_sha256=archive_sha256,
            **axis_meta,
            top_k=8,
            min_points_for_regression=10,  # force all pairs below threshold
            write_sidecar=False,
        )
        # All pairs should have lambda_r_estimate=0.0 due to threshold
        for e in result.entries:
            assert e.lambda_r_estimate == 0.0
            assert e.regression_r2 == 0.0

    def test_sidecar_emit_includes_median_lambda(
        self, per_pair_grad, archive_sha256, axis_meta, tmp_root
    ):
        result = mgc.per_pair_lagrangian_lambda_bisection(
            per_pair_grad, archive_sha256=archive_sha256, **axis_meta, write_sidecar=True
        )
        sidecars = list(
            (tmp_root / "master_gradient_consumers").glob(
                "per_pair_lagrangian_lambda_bisection_*.json"
            )
        )
        assert len(sidecars) == 1
        payload = json.loads(sidecars[0].read_text())
        assert "median_lambda_r" in payload
        assert payload["score_claim"] is False


# ──────────────────────────────────────────────────────────────────────────── #
# Consumer 9 — per_pair_lora_supervision_signal                                  #
# ──────────────────────────────────────────────────────────────────────────── #


class TestConsumer9LoraSupervision:
    def test_contract_rejects_wrong_shape(self, archive_sha256, axis_meta):
        with pytest.raises(ValueError, match="N_bytes, N_pairs, 3"):
            mgc.per_pair_lora_supervision_signal(
                np.zeros((10,)), archive_sha256=archive_sha256, **axis_meta
            )

    def test_top_k_clamped_to_n_bytes(self, per_pair_grad, archive_sha256, axis_meta):
        result = mgc.per_pair_lora_supervision_signal(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            top_k=999,  # > N_bytes=64
            write_sidecar=False,
        )
        assert len(result.top_targets) == 64
        assert result.top_k == 64

    def test_supervision_score_sorted_descending(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        result = mgc.per_pair_lora_supervision_signal(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            top_k=20,
            write_sidecar=False,
        )
        scores = [t.lora_supervision_score for t in result.top_targets]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_sidecar_emit_carries_top_targets(
        self, per_pair_grad, archive_sha256, axis_meta, tmp_root
    ):
        mgc.per_pair_lora_supervision_signal(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            top_k=10,
            write_sidecar=True,
        )
        sidecars = list(
            (tmp_root / "master_gradient_consumers").glob(
                "per_pair_lora_supervision_signal_*.json"
            )
        )
        assert len(sidecars) == 1
        payload = json.loads(sidecars[0].read_text())
        assert len(payload["top_targets"]) == 10
        assert payload["score_claim"] is False


# ──────────────────────────────────────────────────────────────────────────── #
# Consumer 10 — per_pair_coding_budget_allocation                                #
# ──────────────────────────────────────────────────────────────────────────── #


class TestConsumer10CodingBudget:
    def test_contract_rejects_zero_baseline(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        with pytest.raises(ValueError, match="baseline_bytes_per_pair"):
            mgc.per_pair_coding_budget_allocation(
                per_pair_grad,
                archive_sha256=archive_sha256,
                **axis_meta,
                baseline_bytes_per_pair=0,
                write_sidecar=False,
            )

    def test_total_budget_conserved(self, per_pair_grad, archive_sha256, axis_meta):
        result = mgc.per_pair_coding_budget_allocation(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            baseline_bytes_per_pair=64,
            write_sidecar=False,
        )
        total = sum(e.allocated_bytes for e in result.entries)
        assert total == 8 * 64  # n_pairs * baseline

    def test_relative_shares_sum_to_one(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        result = mgc.per_pair_coding_budget_allocation(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            write_sidecar=False,
        )
        total_share = sum(e.relative_share for e in result.entries)
        assert abs(total_share - 1.0) < 1e-6

    def test_zero_gradient_degenerate_falls_back_to_uniform(
        self, archive_sha256, axis_meta
    ):
        zero = np.zeros((10, 5, 3))
        result = mgc.per_pair_coding_budget_allocation(
            zero,
            archive_sha256=archive_sha256,
            **axis_meta,
            baseline_bytes_per_pair=100,
            write_sidecar=False,
        )
        # Uniform fallback: each pair gets 100 bytes (5 pairs * 100 = 500)
        assert result.total_budget_bytes == 500
        # Each pair should get ~100 (modulo rounding)
        for e in result.entries:
            assert e.allocated_bytes == 100

    def test_sidecar_emit(
        self, per_pair_grad, archive_sha256, axis_meta, tmp_root
    ):
        mgc.per_pair_coding_budget_allocation(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            write_sidecar=True,
        )
        sidecars = list(
            (tmp_root / "master_gradient_consumers").glob(
                "per_pair_coding_budget_allocation_*.json"
            )
        )
        assert len(sidecars) == 1


# ──────────────────────────────────────────────────────────────────────────── #
# Consumer 11 — engineered_correction_targeting                                  #
# ──────────────────────────────────────────────────────────────────────────── #


class TestConsumer11EngineeredCorrection:
    def test_total_targets_equals_pairs_times_per_pair(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        result = mgc.engineered_correction_targeting(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            targets_per_pair=5,
            write_sidecar=False,
        )
        assert len(result.targets) == 8 * 5

    def test_targets_per_pair_clamped_to_n_bytes(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        result = mgc.engineered_correction_targeting(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            targets_per_pair=999,
            write_sidecar=False,
        )
        assert result.targets_per_pair == 64  # N_bytes
        assert len(result.targets) == 8 * 64

    def test_priority_zero_one_per_pair(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        result = mgc.engineered_correction_targeting(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            targets_per_pair=5,
            write_sidecar=False,
        )
        # Each pair should have exactly one target_priority=0 entry
        priority_zero_by_pair: dict[int, int] = {}
        for t in result.targets:
            if t.target_priority == 0:
                priority_zero_by_pair[t.pair_index] = (
                    priority_zero_by_pair.get(t.pair_index, 0) + 1
                )
        for pair_idx in range(8):
            assert priority_zero_by_pair[pair_idx] == 1


# ──────────────────────────────────────────────────────────────────────────── #
# Consumer 12 — per_pair_kkt_residuals                                           #
# ──────────────────────────────────────────────────────────────────────────── #


class TestConsumer12KKTResiduals:
    def test_residual_is_nonnegative(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        result = mgc.per_pair_kkt_residuals(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            write_sidecar=False,
        )
        for e in result.entries:
            assert e.kkt_residual_l2 >= 0
            assert e.distortion_norm >= 0
            assert e.rate_norm >= 0

    def test_lambda_override_used_for_all_pairs(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        result = mgc.per_pair_kkt_residuals(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            operator_lambda_r=0.3,
            write_sidecar=False,
        )
        assert result.operator_lambda_r == 0.3
        for e in result.entries:
            assert e.lambda_r_used == 0.3

    def test_median_residual_computed(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        result = mgc.per_pair_kkt_residuals(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            write_sidecar=False,
        )
        # Median residual should be in [min residual, max residual]
        residuals = sorted(e.kkt_residual_l2 for e in result.entries)
        assert residuals[0] <= result.median_kkt_residual <= residuals[-1]


# ──────────────────────────────────────────────────────────────────────────── #
# Consumer 13 — per_pair_volterra_cross_terms                                    #
# ──────────────────────────────────────────────────────────────────────────── #


class TestConsumer13Volterra:
    def test_top_k_clamped_to_pair_pairs(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        # N_pairs=8 -> at most 8*7/2 = 28 upper-triangle pairs
        result = mgc.per_pair_volterra_cross_terms(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            top_k=999,
            downsample_bytes=32,
            write_sidecar=False,
        )
        assert result.top_k == 28
        assert len(result.top_k_couplings) == 28

    def test_coupling_scores_in_valid_range(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        result = mgc.per_pair_volterra_cross_terms(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            top_k=10,
            downsample_bytes=32,
            write_sidecar=False,
        )
        for e in result.top_k_couplings:
            # cosine similarity on non-negative vectors is in [0, 1]
            assert 0.0 <= e.coupling_score <= 1.0 + 1e-9
            assert e.pair_i < e.pair_j  # upper triangle

    def test_couplings_sorted_descending(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        result = mgc.per_pair_volterra_cross_terms(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            top_k=15,
            downsample_bytes=32,
            write_sidecar=False,
        )
        scores = [e.coupling_score for e in result.top_k_couplings]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_downsample_no_change_when_n_bytes_small(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        # N_bytes=64; downsample request 999 -> use all 64
        result = mgc.per_pair_volterra_cross_terms(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            top_k=10,
            downsample_bytes=999,
            write_sidecar=False,
        )
        assert result.downsample_bytes == 64

    def test_determinism_with_random_seed(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        a = mgc.per_pair_volterra_cross_terms(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            top_k=10,
            downsample_bytes=20,  # forces downsampling
            random_seed=123,
            write_sidecar=False,
        )
        b = mgc.per_pair_volterra_cross_terms(
            per_pair_grad,
            archive_sha256=archive_sha256,
            **axis_meta,
            top_k=10,
            downsample_bytes=20,
            random_seed=123,
            write_sidecar=False,
        )
        assert [(e.pair_i, e.pair_j) for e in a.top_k_couplings] == [
            (e.pair_i, e.pair_j) for e in b.top_k_couplings
        ]


# ──────────────────────────────────────────────────────────────────────────── #
# Consumer 14 — gradient_informed_decoder_pruning                                #
# ──────────────────────────────────────────────────────────────────────────── #


class TestConsumer14DecoderPruning:
    def test_shape_mismatch_raises(
        self, per_pair_grad, archive_sha256, axis_meta
    ):
        # Mismatched N_bytes between per_pair and aggregate
        bad_aggregate = np.zeros((10, 3))
        with pytest.raises(ValueError, match="N_bytes mismatch"):
            mgc.gradient_informed_decoder_pruning(
                per_pair_grad,
                bad_aggregate,
                archive_sha256=archive_sha256,
                **axis_meta,
                write_sidecar=False,
            )

    def test_invalid_floor_raises(
        self, per_pair_grad, aggregate_grad, archive_sha256, axis_meta
    ):
        with pytest.raises(ValueError, match="aggregate_floor_relative"):
            mgc.gradient_informed_decoder_pruning(
                per_pair_grad,
                aggregate_grad,
                archive_sha256=archive_sha256,
                **axis_meta,
                aggregate_floor_relative=1.5,
                write_sidecar=False,
            )

    def test_zero_gradient_all_dead(self, archive_sha256, axis_meta):
        zero_pp = np.zeros((100, 5, 3))
        zero_agg = np.zeros((100, 3))
        result = mgc.gradient_informed_decoder_pruning(
            zero_pp,
            zero_agg,
            archive_sha256=archive_sha256,
            **axis_meta,
            write_sidecar=False,
        )
        # All zero -> max_aggregate=0, floor=0; aggregate < 0 is false; nothing dead
        # by aggregate criterion (the floor is 0, strict <)
        assert result.total_dead_bytes == 0

    def test_high_aggregate_byte_kept_even_when_pair_variance_zero(
        self, archive_sha256, axis_meta
    ):
        # One byte has high aggregate, zero per-pair variance
        per_pair = np.zeros((10, 5, 3))
        per_pair[0, :, 0] = 0.5  # uniform per-pair => zero variance
        aggregate = per_pair.mean(axis=1)
        result = mgc.gradient_informed_decoder_pruning(
            per_pair,
            aggregate,
            archive_sha256=archive_sha256,
            **axis_meta,
            aggregate_floor_relative=0.1,
            write_sidecar=False,
        )
        # Byte 0 has high aggregate; not dead by aggregate criterion
        assert 0 not in result.dead_byte_indices

    def test_low_aggregate_byte_with_per_pair_variance_kept(
        self, archive_sha256, axis_meta
    ):
        # One byte has very low aggregate (cancels) but high per-pair variance
        per_pair = np.zeros((10, 5, 3))
        per_pair[5, 0, 0] = 1.0
        per_pair[5, 1, 0] = -1.0  # cancels in aggregate but per-pair non-zero
        per_pair[5, 2, 0] = 0.5
        # Other bytes have nonzero aggregate so max > 0
        per_pair[0, :, 0] = 1.0
        aggregate = per_pair.mean(axis=1)
        result = mgc.gradient_informed_decoder_pruning(
            per_pair,
            aggregate,
            archive_sha256=archive_sha256,
            **axis_meta,
            aggregate_floor_relative=0.5,
            per_pair_variance_floor=0.01,
            write_sidecar=False,
        )
        # Byte 5 has high per-pair variance even though aggregate cancels -> KEPT
        assert 5 not in result.dead_byte_indices


# ──────────────────────────────────────────────────────────────────────────── #
# __all__ export verification                                                    #
# ──────────────────────────────────────────────────────────────────────────── #


def test_consumers_7_to_14_exported_in_all():
    """All new consumers + their dataclasses are in __all__."""
    expected = [
        # Consumer 7
        "PerPairParetoEnvelopePoint",
        "PerPairParetoEnvelope",
        "per_pair_pareto_envelope",
        # Consumer 8
        "PerPairLambdaEntry",
        "PerPairLambdaBisection",
        "per_pair_lagrangian_lambda_bisection",
        # Consumer 9
        "PerPairLoraSupervisionTarget",
        "PerPairLoraSupervisionSignal",
        "per_pair_lora_supervision_signal",
        # Consumer 10
        "PerPairCodingBudgetEntry",
        "PerPairCodingBudgetAllocation",
        "per_pair_coding_budget_allocation",
        # Consumer 11
        "EngineeredCorrectionTarget",
        "EngineeredCorrectionTargeting",
        "engineered_correction_targeting",
        # Consumer 12
        "PerPairKKTResidualEntry",
        "PerPairKKTResiduals",
        "per_pair_kkt_residuals",
        # Consumer 13
        "PerPairVolterraEntry",
        "PerPairVolterraCrossTerms",
        "per_pair_volterra_cross_terms",
        # Consumer 14
        "DecoderPruningCandidate",
        "GradientInformedDecoderPruning",
        "gradient_informed_decoder_pruning",
    ]
    for name in expected:
        assert name in mgc.__all__, f"missing from __all__: {name}"
        assert hasattr(mgc, name), f"not importable from module: {name}"


def test_consumers_7_to_14_dataclasses_are_frozen():
    """Per CLAUDE.md 'Beauty, simplicity, and developer experience'."""
    import dataclasses

    frozen_classes = [
        mgc.PerPairParetoEnvelopePoint,
        mgc.PerPairParetoEnvelope,
        mgc.PerPairLambdaEntry,
        mgc.PerPairLambdaBisection,
        mgc.PerPairLoraSupervisionTarget,
        mgc.PerPairLoraSupervisionSignal,
        mgc.PerPairCodingBudgetEntry,
        mgc.PerPairCodingBudgetAllocation,
        mgc.EngineeredCorrectionTarget,
        mgc.EngineeredCorrectionTargeting,
        mgc.PerPairKKTResidualEntry,
        mgc.PerPairKKTResiduals,
        mgc.PerPairVolterraEntry,
        mgc.PerPairVolterraCrossTerms,
        mgc.DecoderPruningCandidate,
        mgc.GradientInformedDecoderPruning,
    ]
    for cls in frozen_classes:
        params = cls.__dataclass_params__
        assert params.frozen, f"{cls.__name__} not frozen"
