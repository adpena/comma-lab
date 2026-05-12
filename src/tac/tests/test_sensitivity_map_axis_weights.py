"""Tests for ``tac.sensitivity_map.axis_weights`` (COUNCIL-A1 landing 2026-05-12).

Covers
------
- Static anchors (PR106 r2 / OLD 1x / PR102 CUDA) have the expected fields.
- :func:`compute_axis_weights` reproduces the closed-form ratio at PR106 r2
  to within 1e-4 of the CLAUDE.md-cited ``2.7116`` value.
- :func:`compute_axis_weights` validates inputs (d_pose>0, d_seg>=0,
  seg_normalizer>=0, no nan/inf).
- :func:`axis_weights_for_named_operating_point` returns canonical anchors
  and raises on unknown names.
- :func:`validate_axis_weights_mapping` accepts valid dicts and rejects
  missing axes / negative values / nan / inf.
- :class:`AxisWeights` rejects nan/inf/negative on construction, accepts
  zero (free axis), exposes ``as_mapping()`` and ``evidence_tag()``.
- Package re-export surface (``from tac.sensitivity_map import AxisWeights``)
  matches the submodule surface.

Plus integration smoke tests proving the FIX-C bridge + the GGGG A-1 probe
consume the SAME canonical anchors.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tac.score_geometry import score_gradient
from tac.sensitivity_map import axis_weights as axis_weights_module
from tac.sensitivity_map.axis_weights import (
    AXIS_NAMES,
    OLD_1X_OPERATING_POINT_AXIS_WEIGHTS,
    OPERATING_POINT_ANCHORS,
    PR102_CUDA_AXIS_WEIGHTS,
    PR106_R2_FRONTIER_AXIS_WEIGHTS,
    PR106_R2_POSE_PER_SEG_MARGINAL_RATIO,
    AxisWeights,
    AxisWeightsError,
    axis_weights_for_named_operating_point,
    compute_axis_weights,
    default_axis_weights,
    validate_axis_weights_mapping,
)

# ──────────────────────────────────────────────────────────────────────────
# Anchor sanity
# ──────────────────────────────────────────────────────────────────────────


def test_axis_names_canonical_4tuple():
    assert AXIS_NAMES == ("pose", "seg", "rate", "mixed")


def test_pr106_r2_frontier_anchor_fields():
    w = PR106_R2_FRONTIER_AXIS_WEIGHTS
    assert w.pose == 2.71
    assert w.seg == 1.00
    assert w.rate == 1.00
    assert w.mixed == 1.50
    assert w.operating_point_tag == "pr106_r2_frontier"
    assert w.basis == "closed-form gradient per src/tac/score_geometry.py:253-257"


def test_old_1x_anchor_fields():
    w = OLD_1X_OPERATING_POINT_AXIS_WEIGHTS
    assert w.pose == 0.10
    assert w.seg == 1.00
    assert w.rate == 1.00
    assert w.mixed == 0.55
    assert w.operating_point_tag == "old_1x"
    assert "legacy 10/100 loss-weight default" in w.basis


def test_pr102_cuda_anchor_fields():
    w = PR102_CUDA_AXIS_WEIGHTS
    assert w.pose == 2.24
    assert w.seg == 1.00
    assert w.rate == 1.00
    assert w.mixed == 1.62
    assert w.operating_point_tag == "pr102_cuda"
    assert "closed-form gradient at PR102 third-prize CUDA" in w.basis


def test_pr106_r2_marginal_ratio_constant():
    assert pytest.approx(2.7116, abs=1e-4) == PR106_R2_POSE_PER_SEG_MARGINAL_RATIO


def test_operating_point_anchors_lookup_table():
    assert set(OPERATING_POINT_ANCHORS) == {"pr106_r2", "old_1x", "pr102_cuda"}
    assert OPERATING_POINT_ANCHORS["pr106_r2"] is PR106_R2_FRONTIER_AXIS_WEIGHTS
    assert OPERATING_POINT_ANCHORS["old_1x"] is OLD_1X_OPERATING_POINT_AXIS_WEIGHTS
    assert OPERATING_POINT_ANCHORS["pr102_cuda"] is PR102_CUDA_AXIS_WEIGHTS


def test_default_axis_weights_returns_pr106_r2():
    assert default_axis_weights() is PR106_R2_FRONTIER_AXIS_WEIGHTS


# ──────────────────────────────────────────────────────────────────────────
# Closed-form math: PR106 r2 2.7116 verification
# ──────────────────────────────────────────────────────────────────────────


def test_compute_axis_weights_pr106_r2_matches_271():
    # CLAUDE.md cites 2.71 (3-sig-fig rounding); the closed-form value is
    # 2.7116. Verify the closed-form computation matches that to <1e-4.
    w = compute_axis_weights(d_pose=3.4e-5, operating_point_tag="pr106_r2_check")
    assert w.pose == pytest.approx(2.7116, abs=1e-4)
    assert w.seg == 1.0
    assert w.rate == 1.0
    assert w.mixed == pytest.approx((w.pose + w.seg) / 2)


def test_compute_axis_weights_old_1x_closed_form():
    # At d_pose=0.18, dS/d(d_pose) = 0.5 * sqrt(10/0.18) ~= 3.727
    # ratio = 3.727 / 100 ~= 0.0373
    w = compute_axis_weights(d_pose=0.18, operating_point_tag="old_1x_closed_form")
    assert w.pose == pytest.approx(0.0373, abs=1e-3)


def test_compute_axis_weights_pr102_cuda_closed_form():
    # At d_pose=5e-5, dS/d(d_pose) = 0.5 * sqrt(10/5e-5) ~= 223.6
    # ratio = 223.6 / 100 ~= 2.236
    w = compute_axis_weights(d_pose=5e-5, operating_point_tag="pr102_cuda_check")
    assert w.pose == pytest.approx(2.236, abs=1e-3)


def test_compute_axis_weights_seg_normalizer_scales_all_axes():
    w = compute_axis_weights(d_pose=3.4e-5, seg_normalizer=2.0)
    assert w.seg == 2.0
    assert w.rate == 2.0
    # pose scales linearly with seg_normalizer
    assert w.pose == pytest.approx(2.0 * 2.7116, abs=1e-4)


def test_compute_axis_weights_basis_string_includes_d_pose():
    w = compute_axis_weights(d_pose=1.5e-4)
    assert "d_pose=0.00015" in w.basis or "d_pose=1.5e-04" in w.basis


def test_compute_axis_weights_uses_score_gradient_as_source_of_truth():
    # Independently call score_gradient with the same d_pose; verify the
    # computed pose weight equals the gradient ratio exactly.
    d_pose = 3.4e-5
    grad = score_gradient(d_seg=0.0, d_pose=d_pose)
    expected_ratio = grad.d_pose / grad.d_seg
    w = compute_axis_weights(d_pose=d_pose)
    assert w.pose == pytest.approx(expected_ratio, rel=1e-12)


# ──────────────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────────────


def test_compute_axis_weights_rejects_zero_d_pose():
    with pytest.raises(AxisWeightsError, match="d_pose must be > 0"):
        compute_axis_weights(d_pose=0.0)


def test_compute_axis_weights_rejects_negative_d_pose():
    with pytest.raises(AxisWeightsError, match="d_pose must be > 0"):
        compute_axis_weights(d_pose=-1e-5)


def test_compute_axis_weights_rejects_nan_d_pose():
    with pytest.raises(AxisWeightsError, match="d_pose must be"):
        compute_axis_weights(d_pose=float("nan"))


def test_compute_axis_weights_rejects_inf_d_pose():
    with pytest.raises(AxisWeightsError, match="d_pose must be"):
        compute_axis_weights(d_pose=float("inf"))


def test_compute_axis_weights_rejects_negative_d_seg():
    with pytest.raises(AxisWeightsError, match="d_seg must be non-negative"):
        compute_axis_weights(d_pose=3.4e-5, d_seg=-1.0)


def test_compute_axis_weights_accepts_zero_d_seg():
    w = compute_axis_weights(d_pose=3.4e-5, d_seg=0.0)
    assert w.pose > 0


def test_compute_axis_weights_rejects_negative_seg_normalizer():
    with pytest.raises(AxisWeightsError, match="seg_normalizer"):
        compute_axis_weights(d_pose=3.4e-5, seg_normalizer=-1.0)


def test_compute_axis_weights_rejects_nan_seg_normalizer():
    with pytest.raises(AxisWeightsError, match="seg_normalizer"):
        compute_axis_weights(d_pose=3.4e-5, seg_normalizer=float("nan"))


def test_compute_axis_weights_rejects_inf_d_seg():
    with pytest.raises(AxisWeightsError, match="d_seg must be"):
        compute_axis_weights(d_pose=3.4e-5, d_seg=float("inf"))


# ──────────────────────────────────────────────────────────────────────────
# AxisWeights dataclass invariants
# ──────────────────────────────────────────────────────────────────────────


def test_axis_weights_rejects_negative():
    with pytest.raises(AxisWeightsError, match="must be non-negative"):
        AxisWeights(
            pose=-0.1,
            seg=1.0,
            rate=1.0,
            mixed=1.0,
            operating_point_tag="x",
            basis="y",
        )


def test_axis_weights_rejects_nan():
    with pytest.raises(AxisWeightsError, match="must be finite"):
        AxisWeights(
            pose=float("nan"),
            seg=1.0,
            rate=1.0,
            mixed=1.0,
            operating_point_tag="x",
            basis="y",
        )


def test_axis_weights_rejects_inf():
    with pytest.raises(AxisWeightsError, match="must be finite"):
        AxisWeights(
            pose=float("inf"),
            seg=1.0,
            rate=1.0,
            mixed=1.0,
            operating_point_tag="x",
            basis="y",
        )


def test_axis_weights_accepts_zero():
    # Zero weight on an axis is valid: "I do not care about pose at all".
    w = AxisWeights(
        pose=0.0,
        seg=1.0,
        rate=1.0,
        mixed=0.5,
        operating_point_tag="custom",
        basis="custom",
    )
    assert w.pose == 0.0


def test_axis_weights_rejects_empty_operating_point_tag():
    with pytest.raises(AxisWeightsError, match="operating_point_tag"):
        AxisWeights(
            pose=1.0, seg=1.0, rate=1.0, mixed=1.0,
            operating_point_tag="", basis="y",
        )


def test_axis_weights_rejects_empty_basis():
    with pytest.raises(AxisWeightsError, match="basis"):
        AxisWeights(
            pose=1.0, seg=1.0, rate=1.0, mixed=1.0,
            operating_point_tag="x", basis="",
        )


def test_axis_weights_as_mapping_returns_4_axes():
    w = PR106_R2_FRONTIER_AXIS_WEIGHTS.as_mapping()
    assert set(w) == {"pose", "seg", "rate", "mixed"}
    assert w["pose"] == 2.71
    assert w["seg"] == 1.00


def test_axis_weights_as_mapping_does_not_leak_provenance():
    w = PR106_R2_FRONTIER_AXIS_WEIGHTS.as_mapping()
    assert "operating_point_tag" not in w
    assert "basis" not in w


def test_axis_weights_evidence_tag_includes_provenance():
    tag = PR106_R2_FRONTIER_AXIS_WEIGHTS.evidence_tag()
    assert "axis_weights v1" in tag
    assert "pr106_r2_frontier" in tag
    assert "closed-form" in tag


def test_axis_weights_is_frozen():
    with pytest.raises(FrozenInstanceError):
        PR106_R2_FRONTIER_AXIS_WEIGHTS.pose = 99.0  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────
# axis_weights_for_named_operating_point
# ──────────────────────────────────────────────────────────────────────────


def test_axis_weights_for_named_operating_point_pr106_r2():
    assert axis_weights_for_named_operating_point("pr106_r2") is PR106_R2_FRONTIER_AXIS_WEIGHTS


def test_axis_weights_for_named_operating_point_old_1x():
    assert axis_weights_for_named_operating_point("old_1x") is OLD_1X_OPERATING_POINT_AXIS_WEIGHTS


def test_axis_weights_for_named_operating_point_pr102_cuda():
    assert axis_weights_for_named_operating_point("pr102_cuda") is PR102_CUDA_AXIS_WEIGHTS


def test_axis_weights_for_named_operating_point_unknown_raises():
    with pytest.raises(AxisWeightsError, match="unknown operating-point"):
        axis_weights_for_named_operating_point("never_seen")


# ──────────────────────────────────────────────────────────────────────────
# validate_axis_weights_mapping
# ──────────────────────────────────────────────────────────────────────────


def test_validate_axis_weights_mapping_accepts_valid():
    validate_axis_weights_mapping({"pose": 2.71, "seg": 1.0, "rate": 1.0, "mixed": 1.5})


def test_validate_axis_weights_mapping_rejects_missing_axis():
    with pytest.raises(AxisWeightsError, match="missing axes"):
        validate_axis_weights_mapping({"pose": 2.71, "seg": 1.0, "rate": 1.0})


def test_validate_axis_weights_mapping_rejects_negative():
    with pytest.raises(AxisWeightsError, match="non-negative"):
        validate_axis_weights_mapping(
            {"pose": -1.0, "seg": 1.0, "rate": 1.0, "mixed": 1.0}
        )


def test_validate_axis_weights_mapping_rejects_nan():
    with pytest.raises(AxisWeightsError, match="finite"):
        validate_axis_weights_mapping(
            {"pose": float("nan"), "seg": 1.0, "rate": 1.0, "mixed": 1.0}
        )


def test_validate_axis_weights_mapping_rejects_inf():
    with pytest.raises(AxisWeightsError, match="finite"):
        validate_axis_weights_mapping(
            {"pose": float("inf"), "seg": 1.0, "rate": 1.0, "mixed": 1.0}
        )


def test_validate_axis_weights_mapping_rejects_uncoercible():
    with pytest.raises(AxisWeightsError, match="coercible to float"):
        validate_axis_weights_mapping(
            {"pose": "not a number", "seg": 1.0, "rate": 1.0, "mixed": 1.0}
        )


def test_validate_axis_weights_mapping_accepts_zero():
    # Zero is valid (axis disabled).
    validate_axis_weights_mapping(
        {"pose": 0.0, "seg": 0.0, "rate": 0.0, "mixed": 0.0}
    )


# ──────────────────────────────────────────────────────────────────────────
# Package re-export surface
# ──────────────────────────────────────────────────────────────────────────


def test_top_level_re_export_includes_axis_weights_surface():
    from tac import sensitivity_map as sm

    assert hasattr(sm, "AxisWeights")
    assert hasattr(sm, "AxisWeightsError")
    assert hasattr(sm, "PR106_R2_FRONTIER_AXIS_WEIGHTS")
    assert hasattr(sm, "OLD_1X_OPERATING_POINT_AXIS_WEIGHTS")
    assert hasattr(sm, "PR102_CUDA_AXIS_WEIGHTS")
    assert hasattr(sm, "axis_weights")  # submodule accessible
    assert hasattr(sm, "compute_axis_weights")
    assert hasattr(sm, "default_axis_weights")
    assert hasattr(sm, "axis_weights_for_named_operating_point")
    assert hasattr(sm, "validate_axis_weights_mapping")


def test_top_level_re_export_identity_with_submodule():
    from tac import sensitivity_map as sm

    assert sm.AxisWeights is AxisWeights
    assert sm.PR106_R2_FRONTIER_AXIS_WEIGHTS is PR106_R2_FRONTIER_AXIS_WEIGHTS
    assert sm.compute_axis_weights is compute_axis_weights
    assert sm.axis_weights is axis_weights_module


def test_legacy_sensitivity_map_surface_still_works():
    # The package conversion must not break existing imports.
    from tac.sensitivity_map import (
        SENSITIVITY_MAP_FORMAT,
        SensitivityMapError,
        SensitivityMapStats,
        load_sensitivity_map,
        save_sensitivity_map,
        validate_sensitivity_vector,
    )

    assert SENSITIVITY_MAP_FORMAT == "tac_score_sensitivity_map_v1"
    assert issubclass(SensitivityMapError, ValueError)
    # SensitivityMapStats / load / save / validate are callable/typed.
    assert callable(load_sensitivity_map)
    assert callable(save_sensitivity_map)
    assert callable(validate_sensitivity_vector)
    assert SensitivityMapStats is not None


# ──────────────────────────────────────────────────────────────────────────
# Downstream consumer coherence: bridge + probe
# ──────────────────────────────────────────────────────────────────────────


def test_bridge_default_axis_weights_sourced_from_canonical():
    """The FIX-C composition bridge must consume the canonical AxisWeights.

    Prior to COUNCIL-A1 (this landing), ``DEFAULT_AXIS_WEIGHTS`` was
    inline-defined inside the bridge file. After the landing it MUST
    derive from :data:`PR106_R2_FRONTIER_AXIS_WEIGHTS.as_mapping()` so
    both consumers stay coherent on the operating-point rule.
    """
    import sys

    sys.path.insert(0, "tools")
    try:
        import build_composition_ranking_json as bridge
    finally:
        sys.path.pop(0)

    assert PR106_R2_FRONTIER_AXIS_WEIGHTS.as_mapping() == bridge.DEFAULT_AXIS_WEIGHTS
    # ALSO verify the dict has the 4 canonical axes.
    assert set(bridge.DEFAULT_AXIS_WEIGHTS) == set(AXIS_NAMES)


def test_probe_re_exports_canonical_anchor_table():
    """The GGGG A-1 probe must surface the canonical anchor table.

    The probe's own ``OPERATING_POINT_ANCHORS`` stores raw
    ``(d_pose, d_seg)`` coordinates (those are what the probe's CLI
    consumes), but the canonical :data:`OPERATING_POINT_ANCHORS` of
    typed AxisWeights MUST be re-exported via the alias so callers can
    consume the precomputed multiplier without re-importing
    ``tac.sensitivity_map``.
    """
    import sys

    sys.path.insert(0, "tools")
    try:
        import probe_seg_pose_weight_at_operating_point as probe
    finally:
        sys.path.pop(0)

    # The probe keeps its own raw-coordinates table (this is on purpose;
    # the CLI consumes raw d_pose/d_seg, not the precomputed AxisWeights).
    assert "pr106_r2" in probe.OPERATING_POINT_ANCHORS
    # AND the canonical AxisWeights table is re-exported via alias.
    assert hasattr(probe, "_CANONICAL_AXIS_WEIGHT_ANCHORS")
    assert probe._CANONICAL_AXIS_WEIGHT_ANCHORS["pr106_r2"] is PR106_R2_FRONTIER_AXIS_WEIGHTS


def test_probe_pr106_r2_marginal_ratio_matches_canonical_anchor():
    """End-to-end: probe at PR106 r2 returns the same 2.71x the canonical anchor stores.

    This is the coherence proof: the probe's closed-form output
    (compute_optimal_weights) and the canonical anchor
    (PR106_R2_FRONTIER_AXIS_WEIGHTS.pose) agree at 3-sig-fig precision.
    """
    import sys

    sys.path.insert(0, "tools")
    try:
        import probe_seg_pose_weight_at_operating_point as probe
    finally:
        sys.path.pop(0)

    anchor = probe.OPERATING_POINT_ANCHORS["pr106_r2"]
    weights = probe.compute_optimal_weights(
        d_pose=anchor["d_pose"], d_seg=anchor["d_seg"]
    )
    probe_ratio = weights["ratio_pose_over_seg"]
    canonical_ratio = PR106_R2_FRONTIER_AXIS_WEIGHTS.pose / PR106_R2_FRONTIER_AXIS_WEIGHTS.seg
    # Allow a 3-sig-fig tolerance because the canonical anchor stores
    # the rounded 2.71 value while the probe returns 2.7116.
    assert probe_ratio == pytest.approx(canonical_ratio, abs=2e-3)


# ──────────────────────────────────────────────────────────────────────────
# Edge-case smoke tests
# ──────────────────────────────────────────────────────────────────────────


def test_compute_axis_weights_below_flip_threshold_pose_dominates():
    # d_pose well below 2.5e-4 -> pose marginal > seg marginal -> pose > 1.
    w = compute_axis_weights(d_pose=1e-6)
    assert w.pose > 1.0


def test_compute_axis_weights_above_flip_threshold_seg_dominates():
    # d_pose well above 2.5e-4 -> seg marginal > pose marginal -> pose < 1.
    w = compute_axis_weights(d_pose=1e-2)
    assert w.pose < 1.0


def test_compute_axis_weights_at_flip_threshold_pose_equals_seg():
    # At the flip threshold, ratio is exactly 1.
    from tac.score_geometry import importance_flip_threshold

    w = compute_axis_weights(d_pose=importance_flip_threshold())
    assert w.pose == pytest.approx(1.0, abs=1e-9)


def test_compute_axis_weights_mixed_is_midpoint():
    w = compute_axis_weights(d_pose=3.4e-5)
    assert w.mixed == pytest.approx((w.pose + w.seg) / 2)


def test_module_exports_score_geometry_constants():
    # axis_weights re-exports SEG_COEFFICIENT and POSE_COEFFICIENT_INSIDE_SQRT.
    assert axis_weights_module.SEG_COEFFICIENT == 100.0
    assert axis_weights_module.POSE_COEFFICIENT_INSIDE_SQRT == 10.0
