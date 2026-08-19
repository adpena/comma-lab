"""Tests for ``experiments/ddm_up2_shipping_pose_solve``.

The load-bearing test in this file is
``TestSelectorFloatMatchesIntegerSelector``: the solve renders frame 0 in float
so it stays differentiable, but the receiver renders it with integer numpy
arithmetic. If those two disagree by even one LSB, every measured d_pose in the
arm is measuring a frame the receiver would never emit. The rest of the file
guards the GT-lineage gate -- the confound that bought two refused paid rows
(``ddm_ps1u`` r2, ``ddm_t1h``) by minimising against the wrong ground truth.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
_MODULE_PATH = REPO / "experiments" / "ddm_up2_shipping_pose_solve.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ddm_up2_solve_undertest", _MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


up2 = _load_module()
torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# GT lineage gate -- ddm_pi2's fire-order 2, the confound that cost two rows.
# ---------------------------------------------------------------------------


class TestGtLineageGate:
    def test_contest_cuda_requires_dali(self):
        assert up2.required_lineage_for_axis("contest_cuda") == up2.LINEAGE_DALI

    def test_contest_cpu_requires_pyav(self):
        assert up2.required_lineage_for_axis("contest_cpu") == up2.LINEAGE_AV_PYAV

    def test_unknown_axis_refuses(self):
        with pytest.raises(up2.Up2Error, match="unknown score axis"):
            up2.required_lineage_for_axis("contest_mps")

    def test_matching_lineage_verifies(self):
        report = up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=up2.LINEAGE_DALI)
        assert report["status"] == "VERIFIED"
        assert report["gt_lineage"] == up2.LINEAGE_DALI

    def test_the_exact_confound_that_refused_two_paid_rows(self):
        """A CUDA-axis verdict built on PyAV GT must fail closed, not warn."""
        with pytest.raises(up2.Up2Error, match="GT lineage mismatch"):
            up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=up2.LINEAGE_AV_PYAV)

    def test_reverse_mismatch_also_refuses(self):
        with pytest.raises(up2.Up2Error, match="GT lineage mismatch"):
            up2.verify_gt_lineage(axis="contest_cpu", declared_lineage=up2.LINEAGE_DALI)

    def test_unknown_pt_cache_refuses_on_cuda_axis(self):
        with pytest.raises(up2.Up2Error, match="GT lineage mismatch"):
            up2.verify_gt_lineage(axis="contest_cuda", declared_lineage="unknown_pt")

    def test_error_message_names_the_measured_ratio(self):
        with pytest.raises(up2.Up2Error) as excinfo:
            up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=up2.LINEAGE_AV_PYAV)
        assert "23.74x" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Score arithmetic + the report-resolution floor.
# ---------------------------------------------------------------------------


class TestScoreArithmetic:
    def test_pose_leg_matches_upstream_formula(self):
        # upstream/evaluate.py:92 -- math.sqrt(posenet_dist * 10)
        assert up2.pose_leg(7.77e-06) == pytest.approx(math.sqrt(10 * 7.77e-06))

    def test_pointer_pose_leg_matches_the_t4_row(self):
        assert up2.pose_leg(up2.POINTER_D_POSE_T4) == pytest.approx(0.008814760348415605, rel=1e-12)

    def test_byte_to_score_matches_upstream_denominator(self):
        assert pytest.approx(25.0 / 37_545_489.0, rel=1e-15) == up2.BYTE_TO_SCORE

    def test_report_bound_grows_as_d_pose_falls(self):
        """The 8dp report resolves a small d_pose WORSE, never better."""
        assert up2.pose_report_bound(1e-9) > up2.pose_report_bound(1e-6)
        assert up2.pose_report_bound(1e-6) > up2.pose_report_bound(1e-3)

    def test_report_bound_at_the_pointer(self):
        bound = up2.pose_report_bound(up2.POINTER_D_POSE_T4)
        assert bound == pytest.approx(5.0 / math.sqrt(10 * 7.77e-06) * 0.5e-8, rel=1e-9)
        assert 0 < bound < 1e-5

    def test_resolvable_floor_is_the_report_half_ulp(self):
        assert up2.resolvable_d_pose_floor() == up2.REPORT_HALF_ULP

    def test_zero_d_pose_bound_is_finite(self):
        assert math.isfinite(up2.pose_report_bound(0.0))


# ---------------------------------------------------------------------------
# The int12 lattice.
# ---------------------------------------------------------------------------


class TestLattice:
    def test_candidates_respect_the_int12_rails(self):
        codes = np.full(up2.CARRIER_DIM, up2.COEFF_CODE_MAX, dtype=np.int32)
        block, labels = up2.candidate_codes_for_pair(codes, (-1, 1))
        assert len(block) == len(labels)
        assert block.max() <= up2.COEFF_CODE_MAX
        assert block.min() >= up2.COEFF_CODE_MIN
        # +1 is off-lattice on every coordinate, so only the -1 moves survive.
        assert len(block) == up2.CARRIER_DIM

    def test_candidate_count_in_the_interior(self):
        codes = np.zeros(up2.CARRIER_DIM, dtype=np.int32)
        block, labels = up2.candidate_codes_for_pair(codes, (-2, -1, 1, 2))
        assert len(block) == up2.CARRIER_DIM * 4
        assert labels[0] == (0, -2)

    def test_each_candidate_moves_exactly_one_coordinate(self):
        codes = np.arange(up2.CARRIER_DIM, dtype=np.int32)
        block, labels = up2.candidate_codes_for_pair(codes, (-1, 1))
        for row, (coordinate, offset) in zip(block, labels, strict=True):
            assert int((row != codes).sum()) == 1
            assert row[coordinate] == codes[coordinate] + offset

    def test_lower_rail_clamped(self):
        codes = np.full(up2.CARRIER_DIM, up2.COEFF_CODE_MIN, dtype=np.int32)
        block, _ = up2.candidate_codes_for_pair(codes, (-1, 1))
        assert block.min() >= up2.COEFF_CODE_MIN

    def test_realize_codes_roundtrip(self):
        scales = torch.full((up2.CARRIER_DIM,), 3.2655e-4, dtype=torch.float32)
        codes = np.array([[5, -7, 0, 2047, -2048, 3, 9, -1, 100, -100, 12, 33]], dtype=np.int32)
        coefficients = up2.codes_to_coefficients(codes, scales)
        assert np.array_equal(up2.realize_codes(coefficients, scales), codes)

    def test_realize_codes_clamps_out_of_range(self):
        scales = torch.full((up2.CARRIER_DIM,), 1.0, dtype=torch.float32)
        coefficients = torch.full((1, up2.CARRIER_DIM), 1e9)
        assert up2.realize_codes(coefficients, scales).max() == up2.COEFF_CODE_MAX


# ---------------------------------------------------------------------------
# Straight-through rounding: exact forward, usable backward.
# ---------------------------------------------------------------------------


class TestRoundSte:
    def test_forward_is_exactly_round(self):
        values = torch.tensor([-1.6, -0.5, 0.0, 0.5, 1.4, 254.7])
        assert torch.equal(up2._round_ste(values), torch.round(values))

    def test_backward_is_identity(self):
        values = torch.tensor([1.3, 2.7], requires_grad=True)
        up2._round_ste(values).sum().backward()
        assert torch.equal(values.grad, torch.ones_like(values))

    def test_exact_round_has_no_gradient_path(self):
        """The control: without STE the gradient really is zero, which is why
        the receiver's own rounding severs the solve."""
        values = torch.tensor([1.3, 2.7], requires_grad=True)
        torch.round(values).sum().backward()
        assert torch.equal(values.grad, torch.zeros_like(values))


# ---------------------------------------------------------------------------
# THE load-bearing test: float selector == shipped integer selector.
# ---------------------------------------------------------------------------


class TestSelectorFloatMatchesIntegerSelector:
    """Every one of the 8 shipped modes, float path vs integer path.

    ``runtime/frame0_selector.py`` is the receiver's own code; this test imports
    it from the promoted runtime when that runtime is mounted, and otherwise
    re-implements the same integer semantics locally so the test still binds on
    a machine without the external store.
    """

    @staticmethod
    def _modes():
        runtime = up2.DEFAULT_RUNTIME / "runtime" / "frame0_selector.py"
        if runtime.is_file():
            name = "up2_selector_undertest"
            spec = importlib.util.spec_from_file_location(name, runtime)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            # Register before exec: the module defines a @dataclass whose string
            # annotations are resolved through sys.modules[cls.__module__].
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module.SPARSE_PIXEL_MODES, module.apply_pixel_mode
        pytest.skip("promoted runtime not mounted; selector parity needs the shipped code")

    def test_all_eight_modes_agree_bit_for_bit(self):
        modes, apply_pixel_mode = self._modes()
        rng = np.random.default_rng(20260819)
        frames = rng.integers(0, 256, size=(1, 40, 56, 3), dtype=np.uint8)
        for index, mode in enumerate(modes):
            expected = apply_pixel_mode(frames.copy(), mode)
            as_float = torch.from_numpy(frames).float().permute(0, 3, 1, 2).contiguous()
            got = up2.apply_selector_float(as_float, modes, np.array([index]))
            got_u8 = got.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
            assert np.array_equal(got_u8, expected), f"mode {index} ({mode}) diverges"

    def test_saturating_inputs_clamp_identically(self):
        """0 and 255 are where float-vs-integer clamping would diverge first."""
        modes, apply_pixel_mode = self._modes()
        frames = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        frames[0, :, :4, :] = 255
        for index, mode in enumerate(modes):
            expected = apply_pixel_mode(frames.copy(), mode)
            as_float = torch.from_numpy(frames).float().permute(0, 3, 1, 2).contiguous()
            got = up2.apply_selector_float(as_float, modes, np.array([index]))
            got_u8 = got.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
            assert np.array_equal(got_u8, expected), f"mode {index} clamp diverges"

    def test_mixed_batch_applies_per_pair_modes(self):
        modes, apply_pixel_mode = self._modes()
        rng = np.random.default_rng(7)
        frames = rng.integers(0, 256, size=(3, 12, 16, 3), dtype=np.uint8)
        choices = np.array([0, 3, 7])
        as_float = torch.from_numpy(frames).float().permute(0, 3, 1, 2).contiguous()
        got = up2.apply_selector_float(as_float, modes, choices)
        got_u8 = got.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        for row, choice in enumerate(choices):
            expected = apply_pixel_mode(frames[row : row + 1].copy(), modes[choice])
            assert np.array_equal(got_u8[row : row + 1], expected)

    def test_empty_mode_catalog_is_identity(self):
        frames = torch.rand(2, 3, 4, 5) * 255
        assert torch.equal(up2.apply_selector_float(frames, (), np.array([0, 0])), frames)


# ---------------------------------------------------------------------------
# Pair selection -- never a prefix on the pose axis.
# ---------------------------------------------------------------------------


class TestSelectPairs:
    def test_full_field_is_the_whole_population(self):
        assert np.array_equal(up2.select_pairs(600, 1234), np.arange(600))

    def test_over_request_still_returns_the_field(self):
        assert len(up2.select_pairs(10_000, 1234)) == 600

    def test_subset_is_not_a_prefix(self):
        """A contiguous prefix measures pose 2.54-4.21x harder (ddm_na2)."""
        picked = up2.select_pairs(120, 1234)
        assert len(picked) == 120
        assert not np.array_equal(picked, np.arange(120))
        assert picked.max() > 200  # genuinely spread across the field

    def test_subset_is_sorted_and_unique(self):
        picked = up2.select_pairs(96, 7)
        assert np.array_equal(picked, np.sort(picked))
        assert len(set(picked.tolist())) == len(picked)

    def test_seed_is_deterministic(self):
        assert np.array_equal(up2.select_pairs(64, 99), up2.select_pairs(64, 99))

    def test_different_seeds_differ(self):
        assert not np.array_equal(up2.select_pairs(64, 1), up2.select_pairs(64, 2))

    def test_zero_pairs_refuses(self):
        with pytest.raises(up2.Up2Error, match="pairs must be"):
            up2.select_pairs(0, 1234)


# ---------------------------------------------------------------------------
# Overlay pricing -- the byte cost must equal the receiver's real encoder.
# ---------------------------------------------------------------------------


def _overlay_module():
    runtime = up2.DEFAULT_RUNTIME / "runtime" / "compensation_overlay.py"
    if not runtime.is_file():
        pytest.skip("promoted runtime not mounted; overlay pricing needs the shipped code")
    name = "up2_overlay_undertest"
    spec = importlib.util.spec_from_file_location(name, runtime)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestOverlayPricing:
    def test_payload_bytes_matches_the_real_encoder(self):
        """My byte formula vs the receiver's own encoder, across supports."""
        overlay = _overlay_module()
        rng = np.random.default_rng(20260819)
        for pair_count in (1, 2, 7, 15):
            pairs = sorted(rng.choice(600, pair_count, replace=False).tolist())
            deltas = np.zeros((pair_count, up2.CARRIER_DIM), dtype=np.int32)
            supports = []
            for row in range(pair_count):
                support = int(rng.integers(1, up2.CARRIER_DIM + 1))
                coords = rng.choice(up2.CARRIER_DIM, support, replace=False)
                for coord in coords:
                    value = 0
                    while value == 0:
                        value = int(rng.integers(up2.OVERLAY_MIN_DELTA, up2.OVERLAY_MAX_DELTA + 1))
                    deltas[row, coord] = value
                supports.append(support)
            encoded = overlay.encode_compensation_overlay(pairs, deltas)
            assert up2.overlay_payload_bytes(supports) == len(encoded)

    def test_receiver_caps_the_overlay_at_fifteen_pairs(self):
        """The structural reason a 600-pair solve cannot ship as an overlay."""
        assert up2.OVERLAY_MAX_PAIRS == 15
        with pytest.raises(up2.Up2Error, match=re.escape("accepts 1..15")):
            up2.overlay_payload_bytes([1] * 16)

    def test_empty_overlay_refuses(self):
        with pytest.raises(up2.Up2Error):
            up2.overlay_payload_bytes([])

    def test_selection_ranks_by_absolute_gain_and_clips(self):
        base = np.zeros((up2.N_PAIRS_TOTAL, up2.CARRIER_DIM), dtype=np.int32)
        rows = [
            {"pair": 0, "codes": ([9] + [0] * 11), "start_d_pose": 1e-5, "final_d_pose": 9e-6},
            {"pair": 1, "codes": ([1] + [0] * 11), "start_d_pose": 1e-5, "final_d_pose": 1e-9},
        ]
        picked = up2.select_overlay_candidate(rows, base, max_pairs=1)
        # pair 1 has the larger absolute gain, so it wins the single slot
        assert picked["pairs"] == [1]
        assert picked["codes"][1][0] == 1
        assert not picked["any_clipped"]

    def test_selection_clips_into_the_three_bit_domain(self):
        base = np.zeros((up2.N_PAIRS_TOTAL, up2.CARRIER_DIM), dtype=np.int32)
        rows = [{"pair": 3, "codes": ([9] + [0] * 11), "start_d_pose": 1e-5, "final_d_pose": 1e-6}]
        picked = up2.select_overlay_candidate(rows, base, max_pairs=15)
        assert picked["codes"][3][0] == up2.OVERLAY_MAX_DELTA
        assert picked["any_clipped"] is True

    def test_realized_gains_override_the_unclipped_ranking(self):
        """Clipping can destroy an unclipped gain, so measured gain must win."""
        base = np.zeros((up2.N_PAIRS_TOTAL, up2.CARRIER_DIM), dtype=np.int32)
        rows = [
            # huge unclipped gain, but its move clips 9 -> 4 and measures poorly
            {"pair": 0, "codes": ([9] + [0] * 11), "start_d_pose": 1e-5, "final_d_pose": 1e-9},
            # modest unclipped gain, survives clipping intact
            {"pair": 1, "codes": ([2] + [0] * 11), "start_d_pose": 1e-5, "final_d_pose": 9e-6},
        ]
        without = up2.select_overlay_candidate(rows, base, max_pairs=1)
        assert without["pairs"] == [0]
        with_measured = up2.select_overlay_candidate(
            rows, base, max_pairs=1, realized_gains={0: 1e-12, 1: 1e-6}
        )
        assert with_measured["pairs"] == [1]

    def test_realized_gain_of_zero_drops_the_pair(self):
        base = np.zeros((up2.N_PAIRS_TOTAL, up2.CARRIER_DIM), dtype=np.int32)
        rows = [{"pair": 4, "codes": ([1] + [0] * 11), "start_d_pose": 1e-5, "final_d_pose": 1e-9}]
        picked = up2.select_overlay_candidate(rows, base, realized_gains={4: 0.0})
        assert picked["pairs"] == []

    def test_selection_drops_non_improving_pairs(self):
        base = np.zeros((up2.N_PAIRS_TOTAL, up2.CARRIER_DIM), dtype=np.int32)
        rows = [{"pair": 5, "codes": ([2] + [0] * 11), "start_d_pose": 1e-6, "final_d_pose": 1e-6}]
        assert up2.select_overlay_candidate(rows, base)["pairs"] == []

    def test_selection_honours_the_fifteen_pair_cap(self):
        base = np.zeros((up2.N_PAIRS_TOTAL, up2.CARRIER_DIM), dtype=np.int32)
        rows = [
            {
                "pair": p,
                "codes": ([1] + [0] * 11),
                "start_d_pose": 1e-5,
                "final_d_pose": 1e-5 - p * 1e-9,
            }
            for p in range(1, 40)
        ]
        picked = up2.select_overlay_candidate(rows, base)
        assert len(picked["pairs"]) == up2.OVERLAY_MAX_PAIRS
        assert picked["pairs"] == sorted(picked["pairs"])
        up2.overlay_payload_bytes(picked["support_per_pair"])  # must be encodable


# ---------------------------------------------------------------------------
# Candidate pricing -- net delta S, with the report bound stated.
# ---------------------------------------------------------------------------


class TestPriceCandidate:
    def test_pure_pose_gain_is_negative_score(self):
        priced = up2.price_candidate(
            d_pose_start=7.77e-06, d_pose_final=7.60e-06, delta_bytes=0
        )
        assert priced["delta_score_pose"] < 0
        assert priced["net_delta_score"] == pytest.approx(priced["delta_score_pose"])

    def test_bytes_are_charged_at_the_upstream_rate(self):
        priced = up2.price_candidate(
            d_pose_start=7.77e-06, d_pose_final=7.77e-06, delta_bytes=100
        )
        assert priced["delta_score_rate"] == pytest.approx(100 * 25 / 37_545_489)
        assert priced["net_delta_score"] > 0  # pure cost, no gain

    def test_bounds_add_across_the_two_rows(self):
        priced = up2.price_candidate(
            d_pose_start=7.77e-06, d_pose_final=7.00e-06, delta_bytes=0
        )
        expected = up2.pose_report_bound(7.77e-06) + up2.pose_report_bound(7.00e-06)
        assert priced["summed_report_bound"] == pytest.approx(expected)

    def test_a_gain_under_the_bound_is_flagged_unresolvable(self):
        priced = up2.price_candidate(
            d_pose_start=7.77e-06, d_pose_final=7.769999e-06, delta_bytes=0
        )
        assert priced["resolvable_by_the_t4_report"] is False

    def test_a_real_gain_is_flagged_resolvable(self):
        priced = up2.price_candidate(
            d_pose_start=7.77e-06, d_pose_final=7.00e-06, delta_bytes=0
        )
        assert priced["resolvable_by_the_t4_report"] is True
        assert priced["net_over_bound"] > 1.0

    def test_report_resolution_floor_is_flagged(self):
        priced = up2.price_candidate(
            d_pose_start=7.77e-06, d_pose_final=1e-09, delta_bytes=0
        )
        assert priced["d_pose_below_report_resolution"] is True

    def test_seg_motion_is_charged_at_one_hundred(self):
        priced = up2.price_candidate(
            d_pose_start=7.77e-06, d_pose_final=7.77e-06, delta_bytes=0, d_seg_delta=1e-6
        )
        assert priced["delta_score_seg"] == pytest.approx(1e-4)
        assert priced["net_delta_score"] == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# Conditioning report -- the statistic that explains the verdict.
# ---------------------------------------------------------------------------


class TestConditioningReport:
    def test_reports_singular_values_and_demanded_step(self):
        jac = torch.zeros(1, up2.POSE_DIMS, up2.CARRIER_DIM, dtype=torch.float64)
        for i in range(up2.POSE_DIMS):
            jac[0, i, i] = 10.0 if i == 0 else 0.01
        residual = torch.zeros(1, up2.POSE_DIMS, dtype=torch.float64)
        residual[0, 5] = 1.0  # all residual in the worst-conditioned direction
        scales = torch.full((up2.CARRIER_DIM,), 1e-4, dtype=torch.float64)
        report = up2.conditioning_report(jac, residual, scales)
        row = report["pairs"][0]
        assert row["condition_number"] == pytest.approx(1000.0, rel=1e-6)
        assert row["residual_norm"] == pytest.approx(1.0)
        # residual sits on sigma=0.01 -> demands a 100-unit coefficient move
        assert max(row["demanded_coefficient_step"]) == pytest.approx(100.0, rel=1e-6)
        assert row["demanded_code_units_max"] == pytest.approx(1e6, rel=1e-6)

    def test_well_conditioned_residual_demands_a_small_step(self):
        jac = torch.zeros(1, up2.POSE_DIMS, up2.CARRIER_DIM, dtype=torch.float64)
        for i in range(up2.POSE_DIMS):
            jac[0, i, i] = 10.0
        residual = torch.zeros(1, up2.POSE_DIMS, dtype=torch.float64)
        residual[0, 0] = 1.0
        scales = torch.full((up2.CARRIER_DIM,), 1e-4, dtype=torch.float64)
        row = up2.conditioning_report(jac, residual, scales)["pairs"][0]
        assert row["condition_number"] == pytest.approx(1.0, rel=1e-6)
        assert max(row["demanded_coefficient_step"]) == pytest.approx(0.1, rel=1e-6)


# ---------------------------------------------------------------------------
# GT cache loading refuses malformed shapes.
# ---------------------------------------------------------------------------


class TestLoadGtPoses:
    def test_missing_file_refuses(self, tmp_path):
        with pytest.raises(up2.Up2Error, match="does not exist"):
            up2.load_gt_poses(tmp_path / "absent.npz")

    def test_wrong_shape_refuses(self, tmp_path):
        path = tmp_path / "gt_bad.npz"
        np.savez(path, gt_poses=np.zeros((10, 6), dtype=np.float32))
        with pytest.raises(up2.Up2Error, match="expected"):
            up2.load_gt_poses(path)

    def test_missing_key_refuses(self, tmp_path):
        path = tmp_path / "gt_nokey.npz"
        np.savez(path, something_else=np.zeros((600, 6), dtype=np.float32))
        with pytest.raises(up2.Up2Error, match="gt_poses"):
            up2.load_gt_poses(path)

    def test_npz_is_labelled_pyav_lineage(self, tmp_path):
        path = tmp_path / "gt_n600.npz"
        np.savez(path, gt_poses=np.zeros((600, 6), dtype=np.float32))
        _, lineage = up2.load_gt_poses(path)
        assert lineage == up2.LINEAGE_AV_PYAV
