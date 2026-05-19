# SPDX-License-Identifier: MIT
"""Tests for the 8 Wave 2C empirical-sweep canonical helpers.

Per CLAUDE.md "Beauty, simplicity, and developer experience" + Catalog #125
6-hook wire-in declaration: every helper has ≥4 dedicated tests covering
basic / error-path / regression / integration with the canonical Atom +
Provenance contract per Catalog #323.

Lane: ``lane_arbitrariness_extinction_wave_2c_path1_experimental_zero_batch_20260518``
"""

from __future__ import annotations

import pytest

from tac.atom import Atom
from tac.experimental_extinctions import (
    BrotliSweepInput,
    CodecSweepInput,
    ConvergenceSweepInput,
    CouncilCadenceInput,
    EmpiricalSweepResult,
    MemoryDecayInput,
    NegationWindowInput,
    ProbeDecayInput,
    SigmaCalibrationInput,
    brotli_quality_10_vs_11_payload_sweep,
    council_cadence_empirical_calibration,
    lzma_vs_zstd_vs_brotli_per_payload_sweep,
    memory_file_category_decay_calibration,
    negation_window_fp_fn_corpus_sweep,
    per_substrate_convergence_aware_early_stopping,
    probe_outcome_staleness_decay_calibration,
    segnet_boundary_curvature_sigma_calibration,
)


# ============================================================================
# Row #1: per-substrate convergence-aware early stopping
# ============================================================================


class TestPerSubstrateConvergenceAwareEarlyStopping:
    def test_basic_convergence_detected(self) -> None:
        """val-score series that flattens converges and emits epoch."""
        r = per_substrate_convergence_aware_early_stopping(
            ConvergenceSweepInput(
                substrate_id="substrate_a1",
                val_score_series=[0.5, 0.3, 0.2, 0.15, 0.149, 0.1489, 0.1488, 0.1487],
                epoch_step=10,
                slope_epsilon=0.01,
                K_consecutive_windows=2,
            )
        )
        assert isinstance(r, EmpiricalSweepResult)
        assert r.intermediate_values["converged"] is True
        assert isinstance(r.solved_value, int)
        assert r.solved_value >= 10

    def test_no_convergence_returns_final_epoch(self) -> None:
        """val-score never flattens -> falls back to final epoch + converged=False."""
        r = per_substrate_convergence_aware_early_stopping(
            ConvergenceSweepInput(
                substrate_id="substrate_b1",
                val_score_series=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                epoch_step=20,
                slope_epsilon=0.001,
                K_consecutive_windows=2,
            )
        )
        assert r.intermediate_values["converged"] is False
        assert r.solved_value == 6 * 20  # final epoch

    def test_rejects_too_short_series(self) -> None:
        """ValueError when val_score_series is too short."""
        with pytest.raises(ValueError, match="length"):
            ConvergenceSweepInput(
                substrate_id="x",
                val_score_series=[0.1, 0.2],
            )

    def test_rejects_empty_substrate_id(self) -> None:
        with pytest.raises(ValueError, match="substrate_id"):
            ConvergenceSweepInput(
                substrate_id="",
                val_score_series=[0.1, 0.2, 0.3, 0.4],
            )

    def test_emits_atom_when_requested(self) -> None:
        """emit_arbitrariness_atom=True attaches canonical Atom to coupled."""
        r = per_substrate_convergence_aware_early_stopping(
            ConvergenceSweepInput(
                substrate_id="substrate_a1",
                val_score_series=[0.5, 0.3, 0.2, 0.15, 0.149, 0.1489, 0.1488, 0.1487],
            ),
            emit_arbitrariness_atom=True,
        )
        assert "atom" in r.coupled_adjustments
        assert isinstance(r.coupled_adjustments["atom"], Atom)

    def test_compute_savings_factor_well_defined(self) -> None:
        r = per_substrate_convergence_aware_early_stopping(
            ConvergenceSweepInput(
                substrate_id="substrate_a1",
                val_score_series=[0.5, 0.3, 0.15, 0.1, 0.099, 0.0989, 0.0988, 0.0987],
                epoch_step=10,
                slope_epsilon=0.01,
                K_consecutive_windows=2,
            )
        )
        factor = r.coupled_adjustments["compute_savings_factor"]
        assert factor > 0
        assert factor >= 1.0  # savings = full_epochs / early_stop_epoch


# ============================================================================
# Row #2: brotli quality 10 vs 11 payload sweep
# ============================================================================


class TestBrotliQuality10vs11PayloadSweep:
    def test_basic_sweep_returns_winner(self) -> None:
        r = brotli_quality_10_vs_11_payload_sweep(
            BrotliSweepInput(
                payload_id="test_payload",
                payload_bytes=b"hello world " * 100,
            )
        )
        assert isinstance(r, EmpiricalSweepResult)
        assert r.solved_value in (10, 11)
        assert len(r.sweep_points) == 2

    def test_synthetic_estimator_fallback_when_no_brotli(self) -> None:
        """Synthetic estimator path always returns 11 as bytes-only winner."""
        # Real brotli may or may not be installed; both backends exit cleanly
        r = brotli_quality_10_vs_11_payload_sweep(
            BrotliSweepInput(
                payload_id="test",
                payload_bytes=b"a" * 1000,
                wall_clock_penalty_per_second=0.0,
            )
        )
        # When wall_clock_penalty=0, lower bytes wins
        bytes_per_q = {p["quality"]: p["compressed_bytes"] for p in r.sweep_points}
        assert r.solved_value == min(bytes_per_q, key=lambda q: bytes_per_q[q])

    def test_rejects_empty_payload(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            BrotliSweepInput(payload_id="x", payload_bytes=b"")

    def test_rejects_non_bytes(self) -> None:
        with pytest.raises(TypeError):
            BrotliSweepInput(payload_id="x", payload_bytes="abc")  # type: ignore[arg-type]

    def test_emits_atom(self) -> None:
        r = brotli_quality_10_vs_11_payload_sweep(
            BrotliSweepInput(payload_id="test", payload_bytes=b"abc" * 50),
            emit_arbitrariness_atom=True,
        )
        assert isinstance(r.coupled_adjustments["atom"], Atom)


# ============================================================================
# Row #3: lzma vs zstd vs brotli per-payload sweep
# ============================================================================


class TestLzmaVsZstdVsBrotliPerPayloadSweep:
    def test_basic_sweep_returns_codec_string(self) -> None:
        r = lzma_vs_zstd_vs_brotli_per_payload_sweep(
            CodecSweepInput(payload_id="test", payload_bytes=b"hello world " * 100)
        )
        assert isinstance(r, EmpiricalSweepResult)
        assert r.solved_value in ("lzma", "zstd", "brotli")
        assert len(r.sweep_points) == 3

    def test_all_three_codecs_attempted(self) -> None:
        r = lzma_vs_zstd_vs_brotli_per_payload_sweep(
            CodecSweepInput(payload_id="test", payload_bytes=b"x" * 1000)
        )
        codecs = {p["codec"] for p in r.sweep_points}
        assert codecs == {"lzma", "zstd", "brotli"}

    def test_rejects_invalid_lzma_preset(self) -> None:
        with pytest.raises(ValueError, match="lzma_preset"):
            CodecSweepInput(payload_id="x", payload_bytes=b"abc", lzma_preset=10)

    def test_rejects_invalid_zstd_level(self) -> None:
        with pytest.raises(ValueError, match="zstd_level"):
            CodecSweepInput(payload_id="x", payload_bytes=b"abc", zstd_level=0)

    def test_rejects_invalid_brotli_quality(self) -> None:
        with pytest.raises(ValueError, match="brotli_quality"):
            CodecSweepInput(payload_id="x", payload_bytes=b"abc", brotli_quality=12)

    def test_emits_atom(self) -> None:
        r = lzma_vs_zstd_vs_brotli_per_payload_sweep(
            CodecSweepInput(payload_id="test", payload_bytes=b"abc" * 100),
            emit_arbitrariness_atom=True,
        )
        assert isinstance(r.coupled_adjustments["atom"], Atom)


# ============================================================================
# Row #4: SegNet boundary curvature sigma calibration
# ============================================================================


class TestSegnetBoundaryCurvatureSigmaCalibration:
    def test_basic_calibration_returns_sigma(self) -> None:
        r = segnet_boundary_curvature_sigma_calibration(
            SigmaCalibrationInput(
                pixel_population_id="test_pop",
                boundary_curvature_samples=[10.0, 12.0, 15.0, 11.0, 14.0, 13.0],
            )
        )
        assert isinstance(r.solved_value, float)
        assert r.solved_value > 0
        assert "canonical_sigma_from_mad" in r.intermediate_values

    def test_rejects_nan_samples(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            SigmaCalibrationInput(
                pixel_population_id="x",
                boundary_curvature_samples=[1.0, float("nan"), 3.0],
            )

    def test_rejects_inf_samples(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            SigmaCalibrationInput(
                pixel_population_id="x",
                boundary_curvature_samples=[1.0, float("inf"), 3.0],
            )

    def test_rejects_empty_samples(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            SigmaCalibrationInput(
                pixel_population_id="x", boundary_curvature_samples=[]
            )

    def test_rejects_negative_sigma_grid(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SigmaCalibrationInput(
                pixel_population_id="x",
                boundary_curvature_samples=[1.0],
                sigma_grid=(0.0,),
            )

    def test_emits_atom(self) -> None:
        r = segnet_boundary_curvature_sigma_calibration(
            SigmaCalibrationInput(
                pixel_population_id="test",
                boundary_curvature_samples=[5.0, 7.0, 8.0],
            ),
            emit_arbitrariness_atom=True,
        )
        assert isinstance(r.coupled_adjustments["atom"], Atom)


# ============================================================================
# Row #5: council cadence empirical calibration
# ============================================================================


class TestCouncilCadenceEmpiricalCalibration:
    def test_basic_calibration(self) -> None:
        r = council_cadence_empirical_calibration(
            CouncilCadenceInput(
                deliberation_anchors=[
                    {"council_tier": "T2", "written_at_utc": "2026-05-15T00:00:00+00:00"},
                    {"council_tier": "T2", "written_at_utc": "2026-05-16T00:00:00+00:00"},
                    {"council_tier": "T3", "written_at_utc": "2026-05-17T00:00:00+00:00"},
                ],
                window_days=30,
                end_utc="2026-05-19T00:00:00+00:00",
            )
        )
        assert isinstance(r.solved_value, dict)
        assert "T2" in r.solved_value
        assert r.solved_value["T2"] > 0
        assert r.solved_value["T3"] > 0

    def test_handles_empty_anchors(self) -> None:
        r = council_cadence_empirical_calibration(
            CouncilCadenceInput(deliberation_anchors=[])
        )
        assert r.intermediate_values["n_anchors_in_window"] == 0

    def test_parses_deliberation_id_date(self) -> None:
        r = council_cadence_empirical_calibration(
            CouncilCadenceInput(
                deliberation_anchors=[
                    {"council_tier": "T2", "deliberation_id": "x_20260515"},
                ],
                end_utc="2026-05-19T00:00:00+00:00",
            )
        )
        assert r.intermediate_values["n_anchors_in_window"] == 1

    def test_rejects_invalid_window(self) -> None:
        with pytest.raises(ValueError, match="window_days"):
            CouncilCadenceInput(deliberation_anchors=[], window_days=0)

    def test_emits_atom(self) -> None:
        r = council_cadence_empirical_calibration(
            CouncilCadenceInput(
                deliberation_anchors=[
                    {"council_tier": "T2", "written_at_utc": "2026-05-15T00:00:00+00:00"},
                ],
                end_utc="2026-05-19T00:00:00+00:00",
            ),
            emit_arbitrariness_atom=True,
        )
        assert isinstance(r.coupled_adjustments["atom"], Atom)


# ============================================================================
# Row #6: probe-outcome staleness decay calibration
# ============================================================================


class TestProbeOutcomeStalenessDecayCalibration:
    def test_basic_calibration(self) -> None:
        r = probe_outcome_staleness_decay_calibration(
            ProbeDecayInput(
                probe_outcomes=[
                    {
                        "surface": "substrate_class_shift",
                        "adjudicated_at_utc": "2026-05-10T00:00:00+00:00",
                    },
                    {
                        "surface": "substrate_class_shift",
                        "adjudicated_at_utc": "2026-05-05T00:00:00+00:00",
                    },
                    {
                        "surface": "substrate_class_shift",
                        "adjudicated_at_utc": "2026-05-15T00:00:00+00:00",
                    },
                ],
                end_utc="2026-05-19T00:00:00+00:00",
            )
        )
        assert isinstance(r.solved_value, dict)
        assert "substrate_class_shift" in r.solved_value
        assert r.solved_value["substrate_class_shift"] > 0

    def test_fallback_when_too_few_outcomes(self) -> None:
        r = probe_outcome_staleness_decay_calibration(
            ProbeDecayInput(
                probe_outcomes=[
                    {
                        "surface": "rare_surface",
                        "adjudicated_at_utc": "2026-05-10T00:00:00+00:00",
                    },
                ],
                end_utc="2026-05-19T00:00:00+00:00",
            )
        )
        assert r.solved_value["rare_surface"] == 30.0  # hardcoded fallback

    def test_handles_empty_outcomes(self) -> None:
        r = probe_outcome_staleness_decay_calibration(
            ProbeDecayInput(probe_outcomes=[])
        )
        assert r.solved_value == {}

    def test_classifies_by_substrate_id_when_no_surface(self) -> None:
        r = probe_outcome_staleness_decay_calibration(
            ProbeDecayInput(
                probe_outcomes=[
                    {
                        "substrate_id": "substrate_pr101_x",
                        "adjudicated_at_utc": "2026-05-15T00:00:00+00:00",
                    },
                ],
                end_utc="2026-05-19T00:00:00+00:00",
            )
        )
        assert "surface_pr101" in r.solved_value

    def test_emits_atom(self) -> None:
        r = probe_outcome_staleness_decay_calibration(
            ProbeDecayInput(
                probe_outcomes=[
                    {
                        "surface": "x",
                        "adjudicated_at_utc": "2026-05-15T00:00:00+00:00",
                    }
                ]
            ),
            emit_arbitrariness_atom=True,
        )
        assert isinstance(r.coupled_adjustments["atom"], Atom)


# ============================================================================
# Row #7: negation window FP/FN corpus sweep
# ============================================================================


class TestNegationWindowFpFnCorpusSweep:
    def test_basic_sweep(self) -> None:
        r = negation_window_fp_fn_corpus_sweep(
            NegationWindowInput(
                labeled_corpus=[
                    {
                        "text": "auth-eval 100ep landed cleanly today",
                        "trigger_offset": 10,
                        "label": "affirmative",
                    },
                    {
                        "text": "previously 100ep auth-eval was attempted",
                        "trigger_offset": 11,
                        "label": "negation",
                    },
                    {
                        "text": "discussion of 100ep tradeoff today",
                        "trigger_offset": 14,
                        "label": "negation",
                    },
                ],
            )
        )
        assert isinstance(r.solved_value, int)
        assert r.solved_value in (40, 60, 80, 100, 120, 160)

    def test_rejects_empty_corpus(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            NegationWindowInput(labeled_corpus=[])

    def test_rejects_invalid_label(self) -> None:
        with pytest.raises(ValueError, match="label"):
            NegationWindowInput(
                labeled_corpus=[
                    {"text": "x", "trigger_offset": 0, "label": "invalid"}
                ]
            )

    def test_rejects_missing_field(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            NegationWindowInput(
                labeled_corpus=[{"text": "x", "trigger_offset": 0}]
            )

    def test_emits_atom(self) -> None:
        r = negation_window_fp_fn_corpus_sweep(
            NegationWindowInput(
                labeled_corpus=[
                    {
                        "text": "previously 100ep",
                        "trigger_offset": 11,
                        "label": "negation",
                    },
                ],
            ),
            emit_arbitrariness_atom=True,
        )
        assert isinstance(r.coupled_adjustments["atom"], Atom)


# ============================================================================
# Row #8: memory file category decay calibration
# ============================================================================


class TestMemoryFileCategoryDecayCalibration:
    def test_basic_calibration(self) -> None:
        r = memory_file_category_decay_calibration(
            MemoryDecayInput(
                memory_file_metadata=[
                    {"filename": "feedback_codex_a.md", "mtime_utc": "2026-05-15T00:00:00+00:00"},
                    {"filename": "feedback_codex_b.md", "mtime_utc": "2026-05-10T00:00:00+00:00"},
                    {"filename": "feedback_codex_c.md", "mtime_utc": "2026-05-05T00:00:00+00:00"},
                ],
                end_utc="2026-05-19T00:00:00+00:00",
            )
        )
        assert isinstance(r.solved_value, dict)
        assert "feedback_codex_" in r.solved_value
        assert r.solved_value["feedback_codex_"] > 0

    def test_fallback_when_too_few_files(self) -> None:
        r = memory_file_category_decay_calibration(
            MemoryDecayInput(
                memory_file_metadata=[
                    {"filename": "feedback_codex_a.md", "mtime_utc": "2026-05-15T00:00:00+00:00"},
                ],
                end_utc="2026-05-19T00:00:00+00:00",
            )
        )
        assert r.solved_value["feedback_codex_"] == 60.0  # hardcoded fallback

    def test_rejects_missing_filename(self) -> None:
        with pytest.raises(ValueError, match="filename"):
            MemoryDecayInput(
                memory_file_metadata=[{"mtime_utc": "2026-05-15T00:00:00+00:00"}]
            )

    def test_rejects_missing_mtime(self) -> None:
        with pytest.raises(ValueError, match="mtime_utc"):
            MemoryDecayInput(memory_file_metadata=[{"filename": "x.md"}])

    def test_other_category_classification(self) -> None:
        r = memory_file_category_decay_calibration(
            MemoryDecayInput(
                memory_file_metadata=[
                    {"filename": "feedback_random_a.md", "mtime_utc": "2026-05-15T00:00:00+00:00"},
                    {"filename": "feedback_random_b.md", "mtime_utc": "2026-05-10T00:00:00+00:00"},
                    {"filename": "feedback_random_c.md", "mtime_utc": "2026-05-05T00:00:00+00:00"},
                ],
                end_utc="2026-05-19T00:00:00+00:00",
            )
        )
        assert "feedback_other_" in r.solved_value

    def test_emits_atom(self) -> None:
        r = memory_file_category_decay_calibration(
            MemoryDecayInput(
                memory_file_metadata=[
                    {"filename": "feedback_codex_a.md", "mtime_utc": "2026-05-15T00:00:00+00:00"},
                ]
            ),
            emit_arbitrariness_atom=True,
        )
        assert isinstance(r.coupled_adjustments["atom"], Atom)


# ============================================================================
# Cross-cutting: API contract regression guards
# ============================================================================


class TestPackagePublicApiContract:
    def test_all_8_helpers_importable_from_package(self) -> None:
        """Regression guard against accidental removal from __init__.py __all__."""
        from tac.experimental_extinctions import __all__

        required = {
            "EmpiricalSweepResult",
            "ConvergenceSweepInput",
            "per_substrate_convergence_aware_early_stopping",
            "BrotliSweepInput",
            "brotli_quality_10_vs_11_payload_sweep",
            "CodecSweepInput",
            "lzma_vs_zstd_vs_brotli_per_payload_sweep",
            "SigmaCalibrationInput",
            "segnet_boundary_curvature_sigma_calibration",
            "CouncilCadenceInput",
            "council_cadence_empirical_calibration",
            "ProbeDecayInput",
            "probe_outcome_staleness_decay_calibration",
            "NegationWindowInput",
            "negation_window_fp_fn_corpus_sweep",
            "MemoryDecayInput",
            "memory_file_category_decay_calibration",
        }
        missing = required - set(__all__)
        assert not missing, f"Missing from __all__: {missing}"

    def test_all_8_helpers_return_empirical_sweep_result(self) -> None:
        """Every helper returns the shared EmpiricalSweepResult dataclass."""
        r1 = per_substrate_convergence_aware_early_stopping(
            ConvergenceSweepInput(
                substrate_id="x", val_score_series=[0.5, 0.3, 0.2, 0.15]
            )
        )
        r2 = brotli_quality_10_vs_11_payload_sweep(
            BrotliSweepInput(payload_id="x", payload_bytes=b"abc" * 10)
        )
        r3 = lzma_vs_zstd_vs_brotli_per_payload_sweep(
            CodecSweepInput(payload_id="x", payload_bytes=b"abc" * 10)
        )
        r4 = segnet_boundary_curvature_sigma_calibration(
            SigmaCalibrationInput(
                pixel_population_id="x", boundary_curvature_samples=[1.0, 2.0, 3.0]
            )
        )
        r5 = council_cadence_empirical_calibration(
            CouncilCadenceInput(deliberation_anchors=[])
        )
        r6 = probe_outcome_staleness_decay_calibration(
            ProbeDecayInput(probe_outcomes=[])
        )
        r7 = negation_window_fp_fn_corpus_sweep(
            NegationWindowInput(
                labeled_corpus=[
                    {"text": "x", "trigger_offset": 0, "label": "affirmative"}
                ]
            )
        )
        r8 = memory_file_category_decay_calibration(
            MemoryDecayInput(memory_file_metadata=[])
        )
        for r in (r1, r2, r3, r4, r5, r6, r7, r8):
            assert isinstance(r, EmpiricalSweepResult)
            assert r.canonical_helper_invocation.startswith(
                "tac.experimental_extinctions."
            )
            assert r.literature_citation  # non-empty
