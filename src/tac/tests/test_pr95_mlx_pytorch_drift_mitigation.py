# SPDX-License-Identifier: MIT
"""Canonical regression tests for PR95 MLX/PyTorch drift mitigation engineering.

PR95-MLX-PYTORCH-DRIFT-MITIGATION-ENGINEERING 2026-05-25 (task #1255).

Tests cover:
- ``DriftClassification`` enum semantics + threshold edge cases
- ``classify_operation_drift`` per-pair classifier
- ``validate_mlx_pytorch_parity_within_tolerance`` runtime helper
- ``portability_attestation_for_state_dict`` per-tensor attestation
- ``canonical_drift_bands_for_pr95_hnerv_decoder`` exposes Slot 1 anchor
- ``measure_per_op_drift`` CLI end-to-end empirical match against the
  Slot 1 trained-checkpoint anchor (3.05e-5)
- Non-promotable markers preserved per CLAUDE.md "MLX portable-local-substrate
  authority" non-negotiable + Catalog #287/#323 canonical Provenance
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from tac.local_acceleration.deterministic_primitives import (
    PR95_HNERV_DECODER_PER_OP_DRIFT_ANCHORS,
    DriftAttestation,
    DriftClassification,
    canonical_drift_bands_for_pr95_hnerv_decoder,
    classify_operation_drift,
    portability_attestation_for_state_dict,
    validate_mlx_pytorch_parity_within_tolerance,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# DriftClassification enum
# ---------------------------------------------------------------------------


class TestDriftClassificationEnum:
    def test_byte_stable_by_default_value(self) -> None:
        assert DriftClassification.BYTE_STABLE_BY_DEFAULT.value == "byte_stable_by_default"

    def test_numeric_tolerance_inherent_value(self) -> None:
        assert (
            DriftClassification.NUMERIC_TOLERANCE_INHERENT.value
            == "numeric_tolerance_inherent"
        )

    def test_framework_different_value(self) -> None:
        assert DriftClassification.FRAMEWORK_DIFFERENT.value == "framework_different"

    def test_enum_is_string(self) -> None:
        """DriftClassification members are str-comparable for JSON round-trip."""
        assert DriftClassification.BYTE_STABLE_BY_DEFAULT == "byte_stable_by_default"


# ---------------------------------------------------------------------------
# classify_operation_drift threshold semantics
# ---------------------------------------------------------------------------


class TestClassifyOperationDrift:
    def test_byte_stable_typical(self) -> None:
        cls = classify_operation_drift(1e-7, 1e-8)
        assert cls == DriftClassification.BYTE_STABLE_BY_DEFAULT

    def test_byte_stable_edge_case_at_max_threshold(self) -> None:
        # Both at the upper bound of BYTE_STABLE_BY_DEFAULT bands.
        cls = classify_operation_drift(1e-6, 1e-7)
        assert cls == DriftClassification.BYTE_STABLE_BY_DEFAULT

    def test_numeric_tolerance_typical(self) -> None:
        # Above BYTE_STABLE threshold but within NUMERIC_TOLERANCE.
        cls = classify_operation_drift(3e-5, 4e-6)
        assert cls == DriftClassification.NUMERIC_TOLERANCE_INHERENT

    def test_numeric_tolerance_edge_case_at_max_threshold(self) -> None:
        cls = classify_operation_drift(1e-4, 1e-5)
        assert cls == DriftClassification.NUMERIC_TOLERANCE_INHERENT

    def test_framework_different_above_max_abs(self) -> None:
        cls = classify_operation_drift(1e-3, 1e-6)
        assert cls == DriftClassification.FRAMEWORK_DIFFERENT

    def test_framework_different_above_mean_abs(self) -> None:
        cls = classify_operation_drift(1e-5, 1e-3)
        assert cls == DriftClassification.FRAMEWORK_DIFFERENT

    def test_zero_drift_byte_stable(self) -> None:
        cls = classify_operation_drift(0.0, 0.0)
        assert cls == DriftClassification.BYTE_STABLE_BY_DEFAULT

    def test_negative_max_abs_raises(self) -> None:
        with pytest.raises(ValueError, match="measured_max_abs"):
            classify_operation_drift(-1e-6, 0.0)

    def test_nan_max_abs_raises(self) -> None:
        with pytest.raises(ValueError, match="measured_max_abs"):
            classify_operation_drift(float("nan"), 0.0)

    def test_inf_mean_abs_raises(self) -> None:
        with pytest.raises(ValueError, match="measured_mean_abs"):
            classify_operation_drift(0.0, float("inf"))


# ---------------------------------------------------------------------------
# validate_mlx_pytorch_parity_within_tolerance runtime helper
# ---------------------------------------------------------------------------


class TestValidateMlxPytorchParityWithinTolerance:
    def test_byte_stable_pass(self) -> None:
        # Two byte-identical arrays.
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        att = validate_mlx_pytorch_parity_within_tolerance(
            operation_name="sin",
            mlx_output=a,
            pytorch_output=b,
        )
        assert att.attested_within_band is True
        assert att.actual_class == DriftClassification.BYTE_STABLE_BY_DEFAULT
        assert att.measured_max_abs == 0.0
        assert att.measured_mean_abs == 0.0

    def test_unknown_op_defaults_to_numeric_tolerance_band(self) -> None:
        # Unknown operation defaults to the safer (wider) NUMERIC_TOLERANCE band.
        # Construct b s.t. mean_abs <= 1e-5 AND max_abs <= 1e-4 (NUMERIC_TOLERANCE band).
        a = np.zeros(10, dtype=np.float32)
        b = np.zeros(10, dtype=np.float32)
        b[0] = 9e-5  # max_abs at 9e-5; mean_abs ~ 9e-6 < 1e-5
        att = validate_mlx_pytorch_parity_within_tolerance(
            operation_name="some_unknown_op",
            mlx_output=a,
            pytorch_output=b,
        )
        assert att.expected_class == DriftClassification.NUMERIC_TOLERANCE_INHERENT
        assert att.attested_within_band is True

    def test_drift_exceeds_expected_byte_stable_band(self) -> None:
        # sin is BYTE_STABLE_BY_DEFAULT; a 1e-3 drift exceeds the band.
        a = np.zeros(10, dtype=np.float32)
        b = np.full(10, 1e-3, dtype=np.float32)
        att = validate_mlx_pytorch_parity_within_tolerance(
            operation_name="sin",
            mlx_output=a,
            pytorch_output=b,
        )
        assert att.expected_class == DriftClassification.BYTE_STABLE_BY_DEFAULT
        assert att.actual_class == DriftClassification.FRAMEWORK_DIFFERENT
        assert att.attested_within_band is False

    def test_shape_mismatch_raises(self) -> None:
        a = np.zeros(10, dtype=np.float32)
        b = np.zeros(11, dtype=np.float32)
        with pytest.raises(ValueError, match="shape mismatch"):
            validate_mlx_pytorch_parity_within_tolerance(
                operation_name="sin", mlx_output=a, pytorch_output=b
            )

    def test_nan_in_mlx_output_raises(self) -> None:
        a = np.array([1.0, float("nan")], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)
        with pytest.raises(ValueError, match="mlx_output"):
            validate_mlx_pytorch_parity_within_tolerance(
                operation_name="sin", mlx_output=a, pytorch_output=b
            )

    def test_explicit_expected_class_override(self) -> None:
        # Force NUMERIC_TOLERANCE band even for a known BYTE_STABLE op.
        # Construct b s.t. mean_abs <= 1e-5 AND max_abs <= 1e-4 (NUMERIC_TOLERANCE band).
        a = np.zeros(10, dtype=np.float32)
        b = np.zeros(10, dtype=np.float32)
        b[0] = 9e-5  # max_abs at 9e-5; mean_abs ~ 9e-6 < 1e-5
        att = validate_mlx_pytorch_parity_within_tolerance(
            operation_name="sin",
            mlx_output=a,
            pytorch_output=b,
            expected_class=DriftClassification.NUMERIC_TOLERANCE_INHERENT,
        )
        assert att.expected_class == DriftClassification.NUMERIC_TOLERANCE_INHERENT
        assert att.attested_within_band is True

    def test_framework_different_explicit_refusal_never_attests(self) -> None:
        # FRAMEWORK_DIFFERENT is a refusal class, not an infinite tolerance band.
        a = np.zeros(10, dtype=np.float32)
        b = np.full(10, 1.0, dtype=np.float32)
        att = validate_mlx_pytorch_parity_within_tolerance(
            operation_name="sin",
            mlx_output=a,
            pytorch_output=b,
            expected_class=DriftClassification.FRAMEWORK_DIFFERENT,
        )
        assert att.expected_class == DriftClassification.FRAMEWORK_DIFFERENT
        assert att.attested_within_band is False


# ---------------------------------------------------------------------------
# DriftAttestation canonical Provenance markers per Catalog #287/#323
# ---------------------------------------------------------------------------


class TestDriftAttestationProvenance:
    def test_attestation_carries_canonical_non_promotable_markers(self) -> None:
        att = DriftAttestation(
            operation_name="sin",
            measured_max_abs=1e-7,
            measured_mean_abs=1e-8,
            expected_class=DriftClassification.BYTE_STABLE_BY_DEFAULT,
            actual_class=DriftClassification.BYTE_STABLE_BY_DEFAULT,
            attested_max_abs_band=1e-6,
            attested_mean_abs_band=1e-7,
            attested_within_band=True,
        )
        row = att.as_dict()
        assert row["score_claim"] is False
        assert row["promotion_eligible"] is False
        assert row["promotable"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert row["axis_tag"] == "[predicted]"
        assert row["evidence_grade"] == "macOS-MLX-research-signal"

    def test_attestation_is_frozen(self) -> None:
        att = DriftAttestation(
            operation_name="sin",
            measured_max_abs=0.0,
            measured_mean_abs=0.0,
            expected_class=DriftClassification.BYTE_STABLE_BY_DEFAULT,
            actual_class=DriftClassification.BYTE_STABLE_BY_DEFAULT,
            attested_max_abs_band=1e-6,
            attested_mean_abs_band=1e-7,
            attested_within_band=True,
        )
        with pytest.raises(FrozenInstanceError):
            att.operation_name = "sigmoid"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# portability_attestation_for_state_dict per-tensor surface
# ---------------------------------------------------------------------------


class TestPortabilityAttestationForStateDict:
    def test_byte_stable_state_dict_attestation(self) -> None:
        mlx_sd = {
            "a": np.array([1.0, 2.0], dtype=np.float32),
            "b": np.array([3.0, 4.0], dtype=np.float32),
        }
        pt_sd = {
            "a": np.array([1.0, 2.0], dtype=np.float32),
            "b": np.array([3.0, 4.0], dtype=np.float32),
        }
        att = portability_attestation_for_state_dict(
            mlx_state_dict_np=mlx_sd,
            pytorch_state_dict_np=pt_sd,
        )
        assert att["verdict"] == "BYTE_STABLE_PER_TENSOR_STATE_DICT_BRIDGE"
        assert att["all_tensors_byte_stable"] is True
        assert att["aggregate_max_abs"] == 0.0
        assert att["aggregate_mean_abs"] == 0.0
        assert att["tensor_count"] == 2

    def test_state_dict_with_drift_exceeds_byte_stable(self) -> None:
        mlx_sd = {"a": np.array([1.0, 2.0], dtype=np.float32)}
        pt_sd = {"a": np.array([1.001, 2.001], dtype=np.float32)}
        att = portability_attestation_for_state_dict(
            mlx_state_dict_np=mlx_sd,
            pytorch_state_dict_np=pt_sd,
        )
        assert att["verdict"] == "STATE_DICT_BRIDGE_DRIFT_EXCEEDS_BYTE_STABLE_THRESHOLD"
        assert att["all_tensors_byte_stable"] is False

    def test_state_dict_key_mismatch_raises(self) -> None:
        mlx_sd = {"a": np.zeros(3, dtype=np.float32)}
        pt_sd = {"b": np.zeros(3, dtype=np.float32)}
        with pytest.raises(KeyError, match="state_dict key mismatch"):
            portability_attestation_for_state_dict(
                mlx_state_dict_np=mlx_sd,
                pytorch_state_dict_np=pt_sd,
            )

    def test_state_dict_shape_mismatch_raises(self) -> None:
        mlx_sd = {"a": np.zeros(3, dtype=np.float32)}
        pt_sd = {"a": np.zeros(4, dtype=np.float32)}
        with pytest.raises(ValueError, match="shape mismatch"):
            portability_attestation_for_state_dict(
                mlx_state_dict_np=mlx_sd,
                pytorch_state_dict_np=pt_sd,
            )

    def test_state_dict_attestation_carries_non_promotable_markers(self) -> None:
        mlx_sd = {"a": np.zeros(3, dtype=np.float32)}
        pt_sd = {"a": np.zeros(3, dtype=np.float32)}
        att = portability_attestation_for_state_dict(
            mlx_state_dict_np=mlx_sd,
            pytorch_state_dict_np=pt_sd,
        )
        assert att["score_claim"] is False
        assert att["promotion_eligible"] is False
        assert att["promotable"] is False
        assert att["ready_for_exact_eval_dispatch"] is False
        assert att["axis_tag"] == "[predicted]"
        assert "local_mlx_pytorch_state_dict_attestation_is_not_contest_auth_eval" in att["blockers"]

    def test_empty_state_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one parameter"):
            portability_attestation_for_state_dict(
                mlx_state_dict_np={},
                pytorch_state_dict_np={},
            )


# ---------------------------------------------------------------------------
# Canonical drift bands match Slot 1 trained-checkpoint anchor
# ---------------------------------------------------------------------------


class TestCanonicalDriftBands:
    def test_canonical_bands_returns_dict_copy(self) -> None:
        d1 = canonical_drift_bands_for_pr95_hnerv_decoder()
        d2 = canonical_drift_bands_for_pr95_hnerv_decoder()
        # Returned dicts are independent (mutation does not poison module state).
        d1["mutated"] = "yes"
        assert "mutated" not in d2

    def test_canonical_bands_carry_slot1_anchor(self) -> None:
        bands = canonical_drift_bands_for_pr95_hnerv_decoder()
        # The Slot 1 trained-checkpoint anchor was 3.0517578125e-05 exact.
        # Random init produces the same 3.0518e-05 max_abs as the trained anchor.
        assert bands["anchor_max_abs"] == pytest.approx(3.0518e-05, rel=1e-3)
        assert bands["verdict_classification"] == "PORTABLE_WITH_ATTESTED_TOLERANCE"
        assert bands["canonical_class"] == "numeric_tolerance_inherent"

    def test_per_op_anchors_complete_for_pr95_decoder(self) -> None:
        """Every operation in the HNeRVDecoder forward pass must have a canonical anchor."""
        required_ops = {
            "bilinear_resize_2x_align_corners_false_nhwc",
            "sin",
            "sigmoid",
            "pixel_shuffle_2x_nhwc",
            "linear_stem",
            "conv2d_3x3_pad1",
            "hnerv_decoder_full",
        }
        missing = required_ops - set(PR95_HNERV_DECODER_PER_OP_DRIFT_ANCHORS.keys())
        assert not missing, f"missing per-op anchors: {missing}"


# ---------------------------------------------------------------------------
# End-to-end CLI smoke test (per-op drift measurement)
# ---------------------------------------------------------------------------


class TestMeasurePerOpDriftEndToEnd:
    """End-to-end test of the CLI helper. Requires MLX + PyTorch installed."""

    def test_measure_per_op_drift_cpu(self, tmp_path: Path) -> None:
        pytest.importorskip("mlx.core")
        pytest.importorskip("torch")

        from tools.measure_pr95_mlx_pytorch_per_op_drift import measure_per_op_drift

        report = measure_per_op_drift(mlx_device="cpu", seed=42)

        # All 7 ops measured.
        assert len(report["per_op"]) == 7

        # All 7 ops within their attested bands.
        assert report["aggregate"]["all_within_attested_bands"] is True

        # No FRAMEWORK_DIFFERENT classifications.
        assert report["aggregate"]["framework_different_op_count"] == 0

        # Full decoder drift matches the Slot 1 anchor (3.05e-5) within rtol=1e-1.
        full = report["per_op"]["hnerv_decoder_full"]
        assert full["measured_max_abs"] == pytest.approx(3.0518e-05, rel=1e-1)

        # Conv2d drift in the NUMERIC_TOLERANCE band.
        conv = report["per_op"]["conv2d_3x3_pad1"]
        assert conv["actual_class"] == "numeric_tolerance_inherent"

        # bilinear / sin / sigmoid / pixel_shuffle / linear all BYTE_STABLE.
        for op_name in (
            "bilinear_resize_2x_align_corners_false_nhwc",
            "sin",
            "sigmoid",
            "pixel_shuffle_2x_nhwc",
            "linear_stem",
        ):
            assert (
                report["per_op"][op_name]["actual_class"]
                == "byte_stable_by_default"
            ), f"{op_name} should be byte_stable_by_default"

        # Non-promotable markers preserved per CLAUDE.md MLX authority non-negotiable.
        assert report["score_claim"] is False
        assert report["promotion_eligible"] is False
        assert report["ready_for_exact_eval_dispatch"] is False
        assert report["evidence_grade"] == "macOS-MLX-research-signal"
        assert report["axis_tag"] == "[predicted]"

    def test_cli_smoke(self, tmp_path: Path) -> None:
        """End-to-end CLI smoke: --require-all-within-bands exits 0 when all PASS."""
        pytest.importorskip("mlx.core")
        pytest.importorskip("torch")

        import subprocess

        report_out = tmp_path / "drift_report.json"
        proc = subprocess.run(
            [
                ".venv/bin/python",
                "tools/measure_pr95_mlx_pytorch_per_op_drift.py",
                "--report-out", str(report_out),
                "--mlx-device", "cpu",
                "--seed", "42",
                "--require-all-within-bands",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, f"CLI failed: {proc.stderr}"
        assert report_out.exists()
        report = json.loads(report_out.read_text())
        assert report["aggregate"]["all_within_attested_bands"] is True
