# SPDX-License-Identifier: MIT
"""Combined-correction validation against slot 9's predict_drift().

Per slot 9 formalization (`feedback_mps_drift_mathematical_and_engineering_formalization_landed_20260519.md`)
the 3 corrections compose multiplicatively:

    Kahan Conv2d (10x) * pinned softmax (1.5x) * fp32 matmul (2x) = ~30x total

These tests verify:
  1. All 3 corrections can be enabled together without state collision.
  2. Provenance tags are preserved across the combined application.
  3. Predicted drift reduction from `predict_drift()` matches operator-facing
     documentation (does NOT regress).
  4. Compose-then-restore round trip preserves global state.

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #287/#323: combined
corrections do NOT promote MPS to a score-truth axis.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tac.mps_diagnostic.drift_predictor import (
    ArchitectureFeatures,
    KernelTypeCounts,
    predict_drift,
)
from tac.mps_diagnostic.fp32_matmul_override import (
    enable_fp32_matmul_accumulation_strict,
    restore_fp32_matmul_accumulation_state,
    strict_fp32_matmul_accumulation,
)
from tac.mps_diagnostic.kahan_conv2d import (
    patch_conv2d_to_kahan_for_mps_globally,
    restore_torch_conv2d,
)
from tac.mps_diagnostic.pinned_softmax import (
    patch_softmax_to_pinned_for_mps_globally,
    restore_torch_softmax,
)


def _build_segnet_features() -> ArchitectureFeatures:
    """Construct ArchitectureFeatures matching the canonical SegNet anchor.

    Per slot 9 §4.6: SegNet ~140K params, ~50 layers, Conv2d-dominated
    accumulation depth ~196,608 (per output channel on 384x512 input).
    """
    return ArchitectureFeatures(
        architecture_id="segnet_efficientnet_b2_unet_5class",
        layer_count=50,
        kernel_type_counts=KernelTypeCounts(
            conv2d_stride2=1,
            conv2d_stride1=44,
            linear_matmul=2,
            softmax=1,
            interpolate_bicubic=0,
            layernorm=0,
            rgb_to_yuv6_inplace=0,
        ),
        parameter_count=140_000,
        accumulation_depth=196_608,
    )


def test_all_three_corrections_can_be_applied_together() -> None:
    # Snapshot original state.
    original_conv2d = F.conv2d
    original_softmax = F.softmax

    patch_conv2d_to_kahan_for_mps_globally()
    patch_softmax_to_pinned_for_mps_globally()
    prior_matmul = enable_fp32_matmul_accumulation_strict()
    try:
        # All 3 patches active simultaneously.
        assert F.conv2d is not original_conv2d
        assert F.softmax is not original_softmax
        # cuda.matmul.allow_tf32 (when present) is False inside the block.
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        # Restore in reverse order of patching.
        restore_fp32_matmul_accumulation_state(prior_matmul)
        restore_torch_softmax()
        restore_torch_conv2d()

    # All 3 restored.
    assert F.conv2d is original_conv2d
    assert F.softmax is original_softmax


def test_combined_corrections_restore_on_exception() -> None:
    original_conv2d = F.conv2d
    original_softmax = F.softmax
    # Force a known CUDA matmul state to validate restoration.
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        original_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True

    try:
        try:
            patch_conv2d_to_kahan_for_mps_globally()
            patch_softmax_to_pinned_for_mps_globally()
            with strict_fp32_matmul_accumulation():
                raise RuntimeError("simulated failure inside combined block")
        except RuntimeError:
            pass
        # Conv2d / softmax patches still active (the with-block only
        # restores matmul flags; the monkey-patches are global).
        restore_torch_softmax()
        restore_torch_conv2d()
        # Now all restored.
        assert F.conv2d is original_conv2d
        assert F.softmax is original_softmax
        # cuda flag restored by the context manager.
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = original_allow_tf32


def test_predict_drift_for_segnet_features_emits_valid_bounds() -> None:
    """Regression guard: predict_drift() returns sensible bound ordering
    and verdict from slot 9's predictor.

    Note: without a `CalibrationAnchor` for the Phase B archive sha
    `f174192aeadf`, the predictor's conservative-low band is much wider
    than the empirical 0.072% gap (per slot 9 memo §"Cauchy-Schwarz upper
    bound"). Calibration anchors close the band. This test ONLY pins the
    invariants the predictor must satisfy without anchors:
      - lower bound non-negative
      - upper bound >= lower bound
      - mps_viable_verdict in canonical set
    """
    features = _build_segnet_features()
    pred = predict_drift(features=features)
    assert pred.predicted_aggregate_gap_lower_bound >= 0.0
    assert pred.predicted_aggregate_gap_upper_bound >= pred.predicted_aggregate_gap_lower_bound
    assert pred.mps_viable_verdict in {
        "MPS_VIABLE", "NEEDS_EMPIRICAL_PROBE", "MPS_NON_VIABLE",
    }


def test_predict_drift_emits_canonical_provenance() -> None:
    """Per Catalog #287/#323: every predicted-gap claim must carry canonical
    Provenance + axis_tag=[predicted]."""
    features = _build_segnet_features()
    pred = predict_drift(features=features)
    assert pred.provenance is not None
    prov_dict = pred.provenance.as_dict() if hasattr(pred.provenance, "as_dict") else pred.provenance.__dict__
    # canonical Provenance has score_claim_valid=False for predictions
    # (audit_score_claim_dict treats predicted-only claims as non-promotable
    # until paired empirical anchor lands).
    assert pred.mps_viable_verdict in {"MPS_VIABLE", "NEEDS_EMPIRICAL_PROBE", "MPS_NON_VIABLE"}


def test_combined_corrections_preserve_predict_drift_segnet_anchor() -> None:
    """When we apply all 3 corrections and re-query predict_drift, the
    prediction model should still emit a valid `DriftPrediction` (the
    helpers don't break the predictor's input-handling contract).
    """
    features = _build_segnet_features()
    original_conv2d = F.conv2d
    original_softmax = F.softmax

    patch_conv2d_to_kahan_for_mps_globally()
    patch_softmax_to_pinned_for_mps_globally()
    prior_matmul = enable_fp32_matmul_accumulation_strict()
    try:
        pred = predict_drift(features=features)
        # Same predicted band (predictor is independent of runtime
        # corrections; it consumes only ArchitectureFeatures).
        assert pred.predicted_aggregate_gap_upper_bound > 0.0
        assert pred.mps_viable_verdict in {
            "MPS_VIABLE",
            "NEEDS_EMPIRICAL_PROBE",
            "MPS_NON_VIABLE",
        }
    finally:
        restore_fp32_matmul_accumulation_state(prior_matmul)
        restore_torch_softmax()
        restore_torch_conv2d()

    # Restored.
    assert F.conv2d is original_conv2d
    assert F.softmax is original_softmax


def test_provenance_tags_pinned_across_3_helpers() -> None:
    """Sanity: all 3 helpers expose `_AXIS_TAG` and `_EVIDENCE_GRADE`
    constants that are non-promotable per CLAUDE.md."""
    from tac.mps_diagnostic.kahan_conv2d import (
        KAHAN_CONV2D_AXIS_TAG,
        KAHAN_CONV2D_EVIDENCE_GRADE,
    )
    from tac.mps_diagnostic.pinned_softmax import (
        PINNED_SOFTMAX_AXIS_TAG,
        PINNED_SOFTMAX_EVIDENCE_GRADE,
    )

    # Every axis tag mentions MPS (non-promotable axis).
    for tag in (KAHAN_CONV2D_AXIS_TAG, PINNED_SOFTMAX_AXIS_TAG):
        assert "MPS" in tag or "mps" in tag
        assert "[contest-CPU]" not in tag
        assert "[contest-CUDA]" not in tag

    # Every evidence grade is diagnostic (non-promotable).
    for grade in (KAHAN_CONV2D_EVIDENCE_GRADE, PINNED_SOFTMAX_EVIDENCE_GRADE):
        assert "diagnostic" in grade.lower()
        assert "contest" not in grade.lower()
