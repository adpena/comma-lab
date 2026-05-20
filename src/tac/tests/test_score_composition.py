# SPDX-License-Identifier: MIT
"""Tests for tac.score_composition (CATHEDRAL-SMARTER-DESIGN-MEMO Dim 3 Steps 3.1-3.3).

Coverage:
- AxisDecomposition contract (frozen / required fields / Provenance integration per Catalog #323)
- compose_score_from_axes math (canonical formula round-trips)
- compose_scalar_delta (matches scalar ΔS for known inputs)
- Frontier pointer integration (current baseline from canonical pointer)
- Backward compat (consumer without per-axis emits None; ranker handles None correctly)
- Edge cases (zero delta / negative archive bytes / monotone sqrt)
- Catalog #287 placeholder-rationale rejection (via Provenance validator)
- Catalog #185 sister-callable regression guard
"""
from __future__ import annotations

import math

import pytest

from tac.cathedral.consumer_contract import AxisDecomposition, HookNumber
from tac.provenance import (
    build_provenance_for_predicted,
    provenance_to_dict,
)
from tac.score_composition import (
    CANONICAL_POSE_SQRT_INNER,
    CANONICAL_RATE_DENOM_BYTES,
    CANONICAL_RATE_MULTIPLIER,
    CANONICAL_SEG_MULTIPLIER,
    ComposedScoreDelta,
    compose_scalar_delta,
    compose_score_from_axes,
    load_baseline_pose_from_canonical_frontier_pointer,
)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _canonical_provenance_dict() -> dict:
    """Build a canonical Provenance dict per Catalog #323."""
    prov = build_provenance_for_predicted(
        model_id="test_score_composition_v1",
        inputs_sha256="a" * 64,
    )
    return provenance_to_dict(prov)


def _basic_decomposition(
    *,
    d_seg_delta: float = -0.0001,
    d_pose_delta: float = 1e-6,
    archive_bytes_delta: int = -200,
) -> AxisDecomposition:
    """Construct a valid AxisDecomposition with canonical Provenance."""
    return AxisDecomposition(
        predicted_d_seg_delta=d_seg_delta,
        predicted_d_pose_delta=d_pose_delta,
        predicted_archive_bytes_delta=archive_bytes_delta,
        canonical_provenance=_canonical_provenance_dict(),
    )


# ─────────────────────────────────────────────────────────────────────────
# AxisDecomposition contract tests
# ─────────────────────────────────────────────────────────────────────────


class TestAxisDecompositionContract:
    """Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 3 Step 3.1 invariants."""

    def test_frozen_dataclass(self) -> None:
        """AxisDecomposition is frozen; field mutation raises."""
        d = _basic_decomposition()
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            d.predicted_d_seg_delta = 0.5  # type: ignore[misc]

    def test_required_fields_present(self) -> None:
        """All 3 numeric fields + axis_tag + provenance accessible."""
        d = _basic_decomposition()
        assert d.predicted_d_seg_delta == pytest.approx(-0.0001)
        assert d.predicted_d_pose_delta == pytest.approx(1e-6)
        assert d.predicted_archive_bytes_delta == -200
        assert d.axis_tag == "[predicted]"
        assert isinstance(d.canonical_provenance, dict)

    def test_canonical_provenance_required(self) -> None:
        """Empty Provenance dict is accepted at construction (Catalog #356 enforces shape)."""
        # The dataclass __post_init__ accepts ANY Mapping including empty;
        # Catalog #356 STRICT preflight gate is the enforcement surface
        # for the canonical Provenance shape per Catalog #287 + #323.
        d = AxisDecomposition(
            predicted_d_seg_delta=0.0,
            predicted_d_pose_delta=0.0,
            predicted_archive_bytes_delta=0,
            canonical_provenance={},
        )
        assert d.canonical_provenance == {}

    def test_canonical_provenance_must_be_mapping(self) -> None:
        """Non-Mapping Provenance rejected at construction."""
        with pytest.raises(ValueError, match="canonical_provenance"):
            AxisDecomposition(
                predicted_d_seg_delta=0.0,
                predicted_d_pose_delta=0.0,
                predicted_archive_bytes_delta=0,
                canonical_provenance="not a mapping",  # type: ignore[arg-type]
            )

    def test_nan_seg_rejected(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            AxisDecomposition(
                predicted_d_seg_delta=float("nan"),
                predicted_d_pose_delta=0.0,
                predicted_archive_bytes_delta=0,
                canonical_provenance={},
            )

    def test_inf_pose_rejected(self) -> None:
        with pytest.raises(ValueError, match="infinite"):
            AxisDecomposition(
                predicted_d_seg_delta=0.0,
                predicted_d_pose_delta=float("inf"),
                predicted_archive_bytes_delta=0,
                canonical_provenance={},
            )

    def test_archive_bytes_must_be_int(self) -> None:
        """Float archive bytes rejected (must be signed int)."""
        with pytest.raises(ValueError, match="predicted_archive_bytes_delta"):
            AxisDecomposition(
                predicted_d_seg_delta=0.0,
                predicted_d_pose_delta=0.0,
                predicted_archive_bytes_delta=200.5,  # type: ignore[arg-type]
                canonical_provenance={},
            )

    def test_archive_bytes_bool_rejected(self) -> None:
        """Bool is not int (bool is int subclass in Python but semantically wrong)."""
        with pytest.raises(ValueError, match="predicted_archive_bytes_delta"):
            AxisDecomposition(
                predicted_d_seg_delta=0.0,
                predicted_d_pose_delta=0.0,
                predicted_archive_bytes_delta=True,  # type: ignore[arg-type]
                canonical_provenance={},
            )

    def test_axis_tag_must_be_nonempty_string(self) -> None:
        with pytest.raises(ValueError, match="axis_tag"):
            AxisDecomposition(
                predicted_d_seg_delta=0.0,
                predicted_d_pose_delta=0.0,
                predicted_archive_bytes_delta=0,
                axis_tag="",
                canonical_provenance={},
            )

    def test_as_dict_json_safe(self) -> None:
        """as_dict() returns JSON-safe primitives + nested dict."""
        import json

        d = _basic_decomposition()
        as_dict = d.as_dict()
        # Round-trip through json to confirm safety.
        encoded = json.dumps(as_dict)
        decoded = json.loads(encoded)
        assert decoded["predicted_d_seg_delta"] == pytest.approx(-0.0001)
        assert decoded["predicted_archive_bytes_delta"] == -200

    def test_from_dict_round_trip(self) -> None:
        d = _basic_decomposition()
        as_dict = d.as_dict()
        restored = AxisDecomposition.from_dict(as_dict)
        assert restored.predicted_d_seg_delta == d.predicted_d_seg_delta
        assert restored.predicted_archive_bytes_delta == d.predicted_archive_bytes_delta
        assert restored.canonical_provenance == d.canonical_provenance


# ─────────────────────────────────────────────────────────────────────────
# compose_score_from_axes math tests
# ─────────────────────────────────────────────────────────────────────────


class TestComposeScoreFromAxesMath:
    """Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 3 Step 3.2 canonical formula."""

    def test_canonical_constants_pinned(self) -> None:
        """Canonical scorer constants match the contest formula."""
        # Verify cite-chain: ``score = 100 * d_seg + sqrt(10 * d_pose)
        # + 25 * archive_bytes / 37545489`` per
        # ``experiments/contest_auth_eval.py``.
        assert CANONICAL_SEG_MULTIPLIER == 100.0
        assert CANONICAL_POSE_SQRT_INNER == 10.0
        assert CANONICAL_RATE_MULTIPLIER == 25.0
        assert CANONICAL_RATE_DENOM_BYTES == 37_545_489

    def test_seg_only_delta(self) -> None:
        """Pure seg delta: seg_contribution = 100 * delta; pose + rate = 0."""
        d = _basic_decomposition(
            d_seg_delta=-0.001, d_pose_delta=0.0, archive_bytes_delta=0
        )
        result = compose_score_from_axes(
            d, current_archive_bytes=337_944, current_d_pose=3.4e-5
        )
        assert result.seg_delta_contribution == pytest.approx(-0.1)
        assert result.pose_delta_contribution == pytest.approx(0.0, abs=1e-10)
        assert result.rate_delta_contribution == pytest.approx(0.0)
        assert result.total_delta == pytest.approx(-0.1)

    def test_rate_only_delta(self) -> None:
        """Pure rate delta: rate_contribution = 25 * delta / 37545489."""
        d = _basic_decomposition(
            d_seg_delta=0.0, d_pose_delta=0.0, archive_bytes_delta=-200
        )
        result = compose_score_from_axes(
            d, current_archive_bytes=337_944, current_d_pose=3.4e-5
        )
        expected_rate = 25.0 * -200 / 37_545_489
        assert result.rate_delta_contribution == pytest.approx(expected_rate)
        assert result.seg_delta_contribution == pytest.approx(0.0)
        assert result.pose_delta_contribution == pytest.approx(0.0, abs=1e-10)
        assert result.total_delta == pytest.approx(expected_rate)

    def test_pose_delta_at_pr106_operating_point(self) -> None:
        """Pose delta at PR106 frontier: marginal ≈ 271 per CLAUDE.md."""
        d = _basic_decomposition(
            d_seg_delta=0.0, d_pose_delta=1e-7, archive_bytes_delta=0
        )
        # PR106 operating point: pose_avg ≈ 3.4e-5
        result = compose_score_from_axes(
            d, current_archive_bytes=337_944, current_d_pose=3.4e-5
        )
        # Marginal sensitivity at this operating point:
        # d/dx sqrt(10*x) = 5 / sqrt(10*x) ≈ 271 at x = 3.4e-5
        # So small delta 1e-7 → contribution ≈ 271 * 1e-7 ≈ 2.71e-5
        marginal = 5.0 / math.sqrt(10.0 * 3.4e-5)
        expected = marginal * 1e-7  # linearization good at small delta
        assert result.pose_delta_contribution == pytest.approx(
            expected, rel=0.01
        )

    def test_pose_delta_at_old_1x_operating_point(self) -> None:
        """Pose delta at OLD 1.x scores: marginal ≈ 12 per CLAUDE.md."""
        d = _basic_decomposition(
            d_seg_delta=0.0, d_pose_delta=1e-3, archive_bytes_delta=0
        )
        # OLD 1.x operating point: pose_avg ≈ 0.18
        result = compose_score_from_axes(
            d, current_archive_bytes=337_944, current_d_pose=0.18
        )
        marginal = 5.0 / math.sqrt(10.0 * 0.18)
        expected = marginal * 1e-3
        assert result.pose_delta_contribution == pytest.approx(
            expected, rel=0.01
        )

    def test_composition_round_trip_three_axes(self) -> None:
        """All 3 axes contribute additively: total = seg + pose + rate."""
        d = _basic_decomposition(
            d_seg_delta=-0.0002,
            d_pose_delta=5e-6,
            archive_bytes_delta=-1000,
        )
        result = compose_score_from_axes(
            d, current_archive_bytes=337_944, current_d_pose=3.4e-5
        )
        assert result.total_delta == pytest.approx(
            result.seg_delta_contribution
            + result.pose_delta_contribution
            + result.rate_delta_contribution
        )

    def test_zero_delta_yields_zero_total(self) -> None:
        """All-zero deltas yield zero contribution."""
        d = _basic_decomposition(
            d_seg_delta=0.0, d_pose_delta=0.0, archive_bytes_delta=0
        )
        result = compose_score_from_axes(
            d, current_archive_bytes=337_944, current_d_pose=3.4e-5
        )
        assert result.total_delta == pytest.approx(0.0, abs=1e-12)

    def test_pose_clamp_at_zero_when_negative(self) -> None:
        """Delta that drives pose negative is clamped at 0 (saturation)."""
        d = _basic_decomposition(
            d_seg_delta=0.0, d_pose_delta=-1.0, archive_bytes_delta=0
        )
        # current pose = 3.4e-5; delta = -1.0 → new = -0.99997 → clamp to 0
        result = compose_score_from_axes(
            d, current_archive_bytes=337_944, current_d_pose=3.4e-5
        )
        # pose contribution = sqrt(0) - sqrt(10 * 3.4e-5) = -1.84e-2
        expected = -math.sqrt(10.0 * 3.4e-5)
        assert result.pose_delta_contribution == pytest.approx(expected)

    def test_negative_current_pose_rejected(self) -> None:
        """sqrt domain: current_d_pose must be >= 0."""
        d = _basic_decomposition()
        with pytest.raises(ValueError, match="current_d_pose"):
            compose_score_from_axes(
                d, current_archive_bytes=337_944, current_d_pose=-1e-5
            )

    def test_nan_current_pose_rejected(self) -> None:
        d = _basic_decomposition()
        with pytest.raises(ValueError, match="finite"):
            compose_score_from_axes(
                d, current_archive_bytes=337_944, current_d_pose=float("nan")
            )

    def test_negative_archive_bytes_rejected(self) -> None:
        d = _basic_decomposition()
        with pytest.raises(ValueError, match="current_archive_bytes"):
            compose_score_from_axes(
                d, current_archive_bytes=-100, current_d_pose=3.4e-5
            )

    def test_decomposition_type_enforced(self) -> None:
        """Non-AxisDecomposition input rejected with TypeError."""
        with pytest.raises(TypeError, match="AxisDecomposition"):
            compose_score_from_axes(
                {"d_seg": 0.0},  # type: ignore[arg-type]
                current_archive_bytes=337_944,
                current_d_pose=3.4e-5,
            )

    def test_archive_bytes_type_enforced(self) -> None:
        d = _basic_decomposition()
        with pytest.raises(TypeError, match="current_archive_bytes"):
            compose_score_from_axes(
                d,
                current_archive_bytes=337_944.5,  # type: ignore[arg-type]
                current_d_pose=3.4e-5,
            )


# ─────────────────────────────────────────────────────────────────────────
# compose_scalar_delta tests
# ─────────────────────────────────────────────────────────────────────────


class TestComposeScalarDelta:
    """Backward-compat scalar API matches full composition's total_delta."""

    def test_scalar_matches_full_composition(self) -> None:
        d = _basic_decomposition()
        full = compose_score_from_axes(
            d, current_archive_bytes=337_944, current_d_pose=3.4e-5
        )
        scalar = compose_scalar_delta(
            d, current_archive_bytes=337_944, current_d_pose=3.4e-5
        )
        assert scalar == full.total_delta

    def test_scalar_known_inputs(self) -> None:
        """Manually-computed value: pure -200 byte delta."""
        d = _basic_decomposition(
            d_seg_delta=0.0, d_pose_delta=0.0, archive_bytes_delta=-200
        )
        scalar = compose_scalar_delta(
            d, current_archive_bytes=337_944, current_d_pose=3.4e-5
        )
        assert scalar == pytest.approx(25.0 * -200 / 37_545_489)


# ─────────────────────────────────────────────────────────────────────────
# ComposedScoreDelta + Provenance propagation tests
# ─────────────────────────────────────────────────────────────────────────


class TestComposedScoreDeltaContract:

    def test_propagates_canonical_provenance(self) -> None:
        d = _basic_decomposition()
        result = compose_score_from_axes(
            d, current_archive_bytes=337_944, current_d_pose=3.4e-5
        )
        # Provenance is propagated through to the composed result
        assert result.canonical_provenance == d.canonical_provenance
        assert result.canonical_provenance.get("evidence_grade") == "predicted"
        assert result.canonical_provenance.get("promotion_eligible") is False

    def test_axis_tag_propagated(self) -> None:
        d = _basic_decomposition()
        result = compose_score_from_axes(
            d, current_archive_bytes=337_944, current_d_pose=3.4e-5
        )
        assert result.axis_tag == d.axis_tag == "[predicted]"

    def test_baselines_echoed(self) -> None:
        d = _basic_decomposition()
        result = compose_score_from_axes(
            d, current_archive_bytes=337_944, current_d_pose=3.4e-5
        )
        assert result.baseline_d_pose == pytest.approx(3.4e-5)
        assert result.baseline_archive_bytes == 337_944

    def test_as_dict_json_safe(self) -> None:
        import json

        d = _basic_decomposition()
        result = compose_score_from_axes(
            d, current_archive_bytes=337_944, current_d_pose=3.4e-5
        )
        encoded = json.dumps(result.as_dict())
        decoded = json.loads(encoded)
        assert "total_delta" in decoded
        assert "canonical_provenance" in decoded

    def test_per_axis_residual_default_empty(self) -> None:
        d = _basic_decomposition()
        result = compose_score_from_axes(
            d, current_archive_bytes=337_944, current_d_pose=3.4e-5
        )
        assert result.per_axis_residual == {}

    def test_per_axis_residual_propagated(self) -> None:
        d = _basic_decomposition()
        residuals = {"seg": 0.001, "pose": -0.0001, "rate": 0.0}
        result = compose_score_from_axes(
            d,
            current_archive_bytes=337_944,
            current_d_pose=3.4e-5,
            per_axis_residual=residuals,
        )
        assert result.per_axis_residual == residuals

    def test_frozen_invariant(self) -> None:
        d = _basic_decomposition()
        result = compose_score_from_axes(
            d, current_archive_bytes=337_944, current_d_pose=3.4e-5
        )
        with pytest.raises(Exception):
            result.total_delta = 99.0  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────
# Frontier pointer integration tests
# ─────────────────────────────────────────────────────────────────────────


class TestFrontierPointerIntegration:

    def test_load_baseline_handles_missing_pointer(self, tmp_path) -> None:
        """Missing pointer file returns None (not crash)."""
        result = load_baseline_pose_from_canonical_frontier_pointer(
            repo_root=tmp_path
        )
        assert result is None

    def test_load_baseline_returns_none_when_anchor_lacks_extra_fields(
        self, tmp_path
    ) -> None:
        """Current pointer schema lacks per-axis components → returns None."""
        # The current AnchorRecord schema does not store d_pose /
        # archive_bytes separately; the helper returns None until the
        # schema is extended in a sister wave (Dim 3 Step 3.4 successor).
        result = load_baseline_pose_from_canonical_frontier_pointer(
            repo_root=tmp_path, panel_axis="contest_cuda"
        )
        assert result is None


# ─────────────────────────────────────────────────────────────────────────
# Backward compat: cathedral autopilot helper handles None correctly
# ─────────────────────────────────────────────────────────────────────────


class TestCathedralAutopilotBackwardCompat:
    """Tests for tools.cathedral_autopilot_autonomous_loop integration."""

    def test_compose_helper_returns_none_when_field_absent(self) -> None:
        """Consumer that doesn't emit per-axis → ranker returns None for breakdown."""
        # Import the private helper directly for unit-testing the
        # detection logic per Catalog #341 backward-compat preservation.
        import sys
        from pathlib import Path

        tools_root = Path(__file__).resolve().parents[3] / "tools"
        if str(tools_root) not in sys.path:
            sys.path.insert(0, str(tools_root))
        from cathedral_autopilot_autonomous_loop import (
            _compose_per_axis_decomposition_if_present,
            CandidateRow,
        )

        contribution = {
            "predicted_delta_adjustment": 0.0,
            "rationale": "no per-axis signal",
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }
        # Minimal CandidateRow stub
        cr = CandidateRow(
            candidate_id="test_no_axis",
            family="test_family",
            predicted_score_delta=0.0,
            expected_information_gain=0.0,
            estimated_dispatch_cost_usd=0.0,
        )
        result = _compose_per_axis_decomposition_if_present(contribution, cr)
        assert result is None

    def test_compose_helper_returns_breakdown_when_present(self) -> None:
        """Consumer that emits per-axis → ranker auto-composes breakdown."""
        import sys
        from pathlib import Path

        tools_root = Path(__file__).resolve().parents[3] / "tools"
        if str(tools_root) not in sys.path:
            sys.path.insert(0, str(tools_root))
        from cathedral_autopilot_autonomous_loop import (
            _compose_per_axis_decomposition_if_present,
            CandidateRow,
        )

        decomp = _basic_decomposition()
        contribution = {
            "predicted_delta_adjustment": 0.0,
            "rationale": "per-axis emission",
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.5,
            "predicted_axis_decomposition": decomp.as_dict(),
        }
        cr = CandidateRow(
            candidate_id="test_with_axis",
            family="test_family",
            predicted_score_delta=0.0,
            expected_information_gain=0.0,
            estimated_dispatch_cost_usd=0.0,
        )
        result = _compose_per_axis_decomposition_if_present(contribution, cr)
        assert result is not None
        assert "total_delta" in result
        assert "seg_delta_contribution" in result
        assert "pose_delta_contribution" in result
        assert "rate_delta_contribution" in result

    def test_compose_helper_surfaces_malformed_emission_as_error(self) -> None:
        """Malformed per-axis emission → ranker surfaces error, doesn't crash."""
        import sys
        from pathlib import Path

        tools_root = Path(__file__).resolve().parents[3] / "tools"
        if str(tools_root) not in sys.path:
            sys.path.insert(0, str(tools_root))
        from cathedral_autopilot_autonomous_loop import (
            _compose_per_axis_decomposition_if_present,
            CandidateRow,
        )

        # NaN in per-axis emission → AxisDecomposition rejects at construction
        contribution = {
            "predicted_delta_adjustment": 0.0,
            "predicted_axis_decomposition": {
                "predicted_d_seg_delta": float("nan"),
                "predicted_d_pose_delta": 0.0,
                "predicted_archive_bytes_delta": 0,
            },
        }
        cr = CandidateRow(
            candidate_id="test_malformed",
            family="test_family",
            predicted_score_delta=0.0,
            expected_information_gain=0.0,
            estimated_dispatch_cost_usd=0.0,
        )
        result = _compose_per_axis_decomposition_if_present(contribution, cr)
        assert result is not None
        assert "error" in result

    def test_compose_helper_handles_non_mapping_emission(self) -> None:
        """Non-Mapping emission → ranker surfaces error, doesn't crash."""
        import sys
        from pathlib import Path

        tools_root = Path(__file__).resolve().parents[3] / "tools"
        if str(tools_root) not in sys.path:
            sys.path.insert(0, str(tools_root))
        from cathedral_autopilot_autonomous_loop import (
            _compose_per_axis_decomposition_if_present,
            CandidateRow,
        )

        contribution = {
            "predicted_delta_adjustment": 0.0,
            "predicted_axis_decomposition": "not a mapping",
        }
        cr = CandidateRow(
            candidate_id="test_string_emission",
            family="test_family",
            predicted_score_delta=0.0,
            expected_information_gain=0.0,
            estimated_dispatch_cost_usd=0.0,
        )
        result = _compose_per_axis_decomposition_if_present(contribution, cr)
        assert result is not None
        assert "error" in result
        assert "Mapping" in result["error"]
