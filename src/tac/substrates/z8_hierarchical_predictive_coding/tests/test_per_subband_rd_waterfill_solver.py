# SPDX-License-Identifier: MIT
"""Tests for the Z8 per-subband RD water-fill solver (#1591 / #1592).

NO-FAKE discipline: tests verify ACTUAL solver behavior (Pareto reduction,
Lagrangian monotonicity, budget/ceiling satisfaction, actuator-key round-trip)
on BOTH synthetic curves AND the REAL landed entropy-headroom report — not
constants. The headline integration guard asserts the emitted map is consumable
by the canonical actuator's own key parser, which is what makes this an extend
(not orphan code).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tac.substrates.z8_hierarchical_predictive_coding.joint_coefficient_waterfill import (
    Z8JointCoefficientWaterfillConfig,
    _parse_entropy_detail_step_key,
)
from tac.substrates.z8_hierarchical_predictive_coding.per_subband_rd_waterfill_solver import (
    PER_SUBBAND_RD_WATERFILL_SCHEMA,
    SubbandRDCurve,
    SubbandRDPoint,
    _pareto_frontier,
    _parse_subband_label,
    emit_actuator_quant_steps,
    load_subband_rd_curves_from_report,
    solve_per_subband_waterfill,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
REAL_REPORT = (
    REPO_ROOT / ".omx" / "research" / "z8_detail_entropy_headroom_20260531T185438Z.json"
)


# --------------------------------------------------------------------------- #
# dataclass invariants
# --------------------------------------------------------------------------- #
def test_rd_point_baseline_none_step_ok():
    p = SubbandRDPoint(quant_step=None, bytes_per_coeff=3.26, distortion_mse=0.0)
    assert p.quant_step is None


def test_rd_point_rejects_nonpositive_step():
    with pytest.raises(ValueError):
        SubbandRDPoint(quant_step=0.0, bytes_per_coeff=1.0, distortion_mse=1e-3)


def test_rd_point_rejects_negative_bytes():
    with pytest.raises(ValueError):
        SubbandRDPoint(quant_step=0.5, bytes_per_coeff=-1.0, distortion_mse=1e-3)


def test_rd_point_rejects_negative_mse():
    with pytest.raises(ValueError):
        SubbandRDPoint(quant_step=0.5, bytes_per_coeff=1.0, distortion_mse=-1e-3)


def test_rd_curve_rejects_bad_orientation():
    with pytest.raises(ValueError):
        SubbandRDCurve(
            name="L0_xx", level=0, orientation="xx", n_coeffs=10,
            points=(SubbandRDPoint(None, 3.0, 0.0),),
        )


def test_rd_curve_rejects_empty_points():
    with pytest.raises(ValueError):
        SubbandRDCurve(name="L0_hh", level=0, orientation="hh", n_coeffs=10, points=())


# --------------------------------------------------------------------------- #
# label parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "label,expected",
    [("L0_hh", (0, "hh")), ("L0_hl", (0, "hl")), ("L1_lh", (1, "lh")), ("L2_HH", (2, "hh"))],
)
def test_parse_subband_label(label, expected):
    assert _parse_subband_label(label) == expected


def test_parse_subband_label_rejects_garbage():
    with pytest.raises(ValueError):
        _parse_subband_label("garbage")


# --------------------------------------------------------------------------- #
# Pareto frontier (the correctness core for non-monotone RD)
# --------------------------------------------------------------------------- #
def test_pareto_drops_dominated_point():
    # B dominates C (fewer bytes AND lower mse) -> C must be dropped.
    a = SubbandRDPoint(None, 3.0, 0.0)
    b = SubbandRDPoint(0.1, 1.0, 1e-4)
    c = SubbandRDPoint(0.2, 1.5, 2e-4)  # dominated by b
    hull = _pareto_frontier([a, b, c])
    assert c not in hull
    assert a in hull and b in hull


def test_pareto_handles_nonmonotone_codec_switch():
    # Mirror the real L0 codec-switch: Δ=0.25 costs MORE than Δ=0.125.
    pts = [
        SubbandRDPoint(None, 3.262, 0.0),
        SubbandRDPoint(0.015625, 0.97525, 1.9786e-05),
        SubbandRDPoint(0.03125, 0.83496, 6.74e-05),
        SubbandRDPoint(0.0625, 0.62897, 2.3514e-04),
        SubbandRDPoint(0.125, 0.42798, 7.7642e-04),
        SubbandRDPoint(0.25, 0.55353, 2.3036e-03),  # MORE bytes than 0.125 -> dominated
    ]
    hull = _pareto_frontier(pts)
    # Δ=0.25 point is strictly dominated by Δ=0.125 (more bytes + more mse) -> dropped.
    assert all(p.quant_step != 0.25 for p in hull)
    # hull bytes strictly decreasing, mse strictly increasing
    for i in range(1, len(hull)):
        assert hull[i].bytes_per_coeff > hull[i - 1].bytes_per_coeff
        assert hull[i].distortion_mse < hull[i - 1].distortion_mse


def test_pareto_hull_is_convex():
    hull = _pareto_frontier(
        [
            SubbandRDPoint(None, 4.0, 0.0),
            SubbandRDPoint(0.1, 2.0, 1e-4),
            SubbandRDPoint(0.2, 1.0, 5e-4),
        ]
    )
    # slope magnitudes |ΔD/ΔR| must be decreasing as bytes increase (convex lower hull)
    slopes = []
    for i in range(1, len(hull)):
        dr = hull[i].bytes_per_coeff - hull[i - 1].bytes_per_coeff
        dd = hull[i].distortion_mse - hull[i - 1].distortion_mse
        slopes.append(abs(dd / dr))
    assert slopes == sorted(slopes, reverse=True)


# --------------------------------------------------------------------------- #
# solver — fixed lambda extremes
# --------------------------------------------------------------------------- #
def _toy_curves():
    return [
        SubbandRDCurve(
            name="L0_hh", level=0, orientation="hh", n_coeffs=100,
            points=(
                SubbandRDPoint(None, 3.0, 0.0),
                SubbandRDPoint(0.1, 1.0, 1e-4),
                SubbandRDPoint(0.2, 0.5, 5e-4),
            ),
        ),
        SubbandRDCurve(
            name="L1_lh", level=1, orientation="lh", n_coeffs=50,
            points=(
                SubbandRDPoint(None, 3.5, 0.0),
                SubbandRDPoint(0.1, 1.2, 2e-4),
                SubbandRDPoint(0.2, 0.6, 6e-4),
            ),
        ),
    ]


def test_solve_lambda_zero_keeps_raw():
    # λ=0 minimizes distortion only -> keep-raw (mse=0) everywhere.
    sol = solve_per_subband_waterfill(_toy_curves(), lambda_value=0.0)
    assert all(c.chosen_quant_step is None for c in sol.choices)
    assert sol.weighted_mean_mse == pytest.approx(0.0)
    assert sol.bytes_saved == pytest.approx(0.0)


def test_solve_large_lambda_picks_cheapest():
    # huge λ -> minimize bytes -> coarsest point per subband.
    sol = solve_per_subband_waterfill(_toy_curves(), lambda_value=1e9)
    assert all(c.bytes_per_coeff <= 0.6 + 1e-9 for c in sol.choices)
    assert sol.bytes_saved > 0.0


def test_solve_requires_exactly_one_mode():
    with pytest.raises(ValueError):
        solve_per_subband_waterfill(_toy_curves())
    with pytest.raises(ValueError):
        solve_per_subband_waterfill(
            _toy_curves(), target_total_bytes=100.0, lambda_value=1.0
        )


# --------------------------------------------------------------------------- #
# solver — byte budget bisection
# --------------------------------------------------------------------------- #
def test_solve_byte_budget_satisfied():
    curves = _toy_curves()
    baseline = sum(c.points[0].bytes_per_coeff * c.n_coeffs for c in curves)  # 3*100+3.5*50
    target = 0.6 * baseline
    sol = solve_per_subband_waterfill(curves, target_total_bytes=target)
    assert sol.total_bytes <= target + 1e-6
    assert sol.bytes_saved > 0.0


def test_solve_byte_budget_monotone_in_target():
    curves = _toy_curves()
    tight = solve_per_subband_waterfill(curves, target_total_bytes=120.0)
    loose = solve_per_subband_waterfill(curves, target_total_bytes=300.0)
    # tighter budget -> fewer or equal bytes AND >= distortion
    assert tight.total_bytes <= loose.total_bytes + 1e-6
    assert tight.weighted_mean_mse >= loose.weighted_mean_mse - 1e-12


# --------------------------------------------------------------------------- #
# solver — distortion ceiling
# --------------------------------------------------------------------------- #
def test_solve_distortion_ceiling_satisfied():
    curves = _toy_curves()
    sol = solve_per_subband_waterfill(curves, max_weighted_mse=2e-4)
    assert sol.weighted_mean_mse <= 2e-4 + 1e-12
    # should have saved something vs keep-raw while staying under ceiling
    assert sol.bytes_saved > 0.0


# --------------------------------------------------------------------------- #
# actuator-key emission (the integration-correctness guard)
# --------------------------------------------------------------------------- #
def test_emit_actuator_keys_roundtrip_through_actuator_parser():
    sol = solve_per_subband_waterfill(_toy_curves(), lambda_value=1e9)
    steps = emit_actuator_quant_steps(sol)
    assert steps  # non-empty
    for key, step in steps.items():
        # MUST parse through the actuator's own key parser
        frame, level, orient = _parse_entropy_detail_step_key(key)
        assert frame in ("frame_0_details", "frame_1_details")
        assert step > 0.0
        assert isinstance(level, int)
        assert orient in ("lh", "hl", "hh")


def test_emit_actuator_map_consumable_by_actuator_config():
    sol = solve_per_subband_waterfill(_toy_curves(), lambda_value=1e9)
    steps = emit_actuator_quant_steps(sol)
    # The canonical actuator config must accept the emitted map without error.
    cfg = Z8JointCoefficientWaterfillConfig(
        entropy_code_quantized_details=True,
        entropy_detail_quantization_steps=steps,
    )
    assert cfg.entropy_detail_quantization_steps == steps


def test_emit_omits_keep_raw_subbands():
    # λ=0 keeps everything raw -> empty map.
    sol = solve_per_subband_waterfill(_toy_curves(), lambda_value=0.0)
    assert emit_actuator_quant_steps(sol) == {}


def test_emit_both_frames_share_same_step():
    sol = solve_per_subband_waterfill(_toy_curves(), lambda_value=1e9)
    steps = emit_actuator_quant_steps(sol)
    for choice in sol.choices:
        if choice.chosen_quant_step is None:
            continue
        k0 = f"frame_0_details:{choice.level}:{choice.orientation}"
        k1 = f"frame_1_details:{choice.level}:{choice.orientation}"
        assert steps[k0] == steps[k1] == choice.chosen_quant_step


# --------------------------------------------------------------------------- #
# REAL report end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not REAL_REPORT.exists(), reason="real entropy-headroom report absent")
def test_load_real_report_six_curves():
    curves = load_subband_rd_curves_from_report(REAL_REPORT)
    assert len(curves) == 6
    names = {c.name for c in curves}
    assert names == {"L0_hh", "L0_hl", "L0_lh", "L1_hh", "L1_hl", "L1_lh"}
    # every curve has the keep-raw baseline + 5 sweep points
    for c in curves:
        assert c.points[0].quant_step is None
        assert len(c.points) == 6


@pytest.mark.skipif(not REAL_REPORT.exists(), reason="real entropy-headroom report absent")
def test_real_report_waterfill_saves_bytes_and_emits_valid_map():
    curves = load_subband_rd_curves_from_report(REAL_REPORT)
    baseline = sum(c.points[0].bytes_per_coeff * c.n_coeffs for c in curves)
    # near-lossless distortion ceiling (the report's finest Δ MSE band is ~2e-5)
    sol = solve_per_subband_waterfill(curves, max_weighted_mse=5e-5)
    assert sol.schema == PER_SUBBAND_RD_WATERFILL_SCHEMA
    assert sol.baseline_total_bytes == pytest.approx(baseline)
    # near-lossless should still cut the raw-f32 cost dramatically (report: ~90%)
    assert sol.bytes_saved > 0.5 * baseline
    assert sol.advisory_markers["promotable"] is False
    steps = emit_actuator_quant_steps(sol)
    assert steps
    for key in steps:
        _parse_entropy_detail_step_key(key)  # must not raise


@pytest.mark.skipif(not REAL_REPORT.exists(), reason="real entropy-headroom report absent")
def test_real_report_tighter_budget_costs_more_distortion():
    curves = load_subband_rd_curves_from_report(REAL_REPORT)
    baseline = sum(c.points[0].bytes_per_coeff * c.n_coeffs for c in curves)
    aggressive = solve_per_subband_waterfill(curves, target_total_bytes=0.10 * baseline)
    gentle = solve_per_subband_waterfill(curves, target_total_bytes=0.30 * baseline)
    assert aggressive.total_bytes <= 0.10 * baseline + 1e-6
    assert gentle.total_bytes <= 0.30 * baseline + 1e-6
    assert aggressive.weighted_mean_mse >= gentle.weighted_mean_mse - 1e-12


@pytest.mark.skipif(not REAL_REPORT.exists(), reason="real entropy-headroom report absent")
def test_real_report_as_dict_serializable():
    curves = load_subband_rd_curves_from_report(REAL_REPORT)
    sol = solve_per_subband_waterfill(curves, max_weighted_mse=5e-5)
    d = sol.as_dict()
    json.dumps(d)  # must be JSON-serializable
    assert d["schema"] == PER_SUBBAND_RD_WATERFILL_SCHEMA
    assert len(d["choices"]) == 6
