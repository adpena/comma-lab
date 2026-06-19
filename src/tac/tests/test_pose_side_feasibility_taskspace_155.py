"""NO-FAKE tests for the POSE-SIDE feasibility probe (#155 task-space rep).

Class-2 discipline: each test would FAIL if the function body were replaced by a constant/marker.
These verify the probe ACTUALLY measures the work it names:

  * the Chebyshev basis is the REAL recurrence T_k = 2 t T_{k-1} - T_{k-2}, bounded on [-1,1],
    and exactly reproduces a known low-degree polynomial (a real fit, not a constant);
  * fit_pose_code's coded trajectory ACTUALLY tracks the input (a smoother input -> lower d_pose;
    a jittery input -> high irreducible d_pose) AND the byte accounting is a real monotone
    function of degree + residual-keep (not a constant);
  * dpose_of_code is exactly the contest reduction mean_pairs( mean_dims( (a-b)^2 ) );
  * the degraded pose carrier ACTUALLY degrades (block-average kills local edges, fewer luma
    bits quantize) and the identity carrier (div=1, 8b) is a no-op (returns the frame unchanged);
  * _carrier_bytes is a real monotone function of retained spatial DOF;
  * the verdict is driven by the MEASURED cheap/holds flags, not hardcoded.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))

PROBE_PATH = REPO / "experiments/probe_pose_side_feasibility_taskspace_155.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("_pose_side_probe", PROBE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe():
    return _load_probe()


# --------------------------------------------------------------------------
# Chebyshev basis: real recurrence, bounded, exactly fits a known polynomial
# --------------------------------------------------------------------------
def test_cheby_basis_is_bounded_and_well_conditioned(probe):
    B, t = probe.cheby_basis(600, 8)
    assert B.shape == (600, 9)
    # Chebyshev-T on [-1,1] are bounded in [-1,1] (would be violated by a wrong recurrence)
    assert np.abs(B).max() <= 1.0 + 1e-9
    assert np.isfinite(B).all()
    # T_1(t) == t exactly
    np.testing.assert_allclose(B[:, 1], t, atol=1e-12)
    # well-conditioned (a wrong/degenerate basis would blow this up)
    assert np.linalg.cond(B) < 10.0


def test_cheby_fit_exactly_reproduces_a_known_low_degree_poly(probe):
    # A degree-3 polynomial in t must be fit EXACTLY by a degree>=3 Chebyshev code at d_pose ~ 0.
    n = 200
    t = np.linspace(-1, 1, n)
    signal = 2.0 + 0.5 * t - 1.5 * t**2 + 0.3 * t**3
    T = np.tile(signal[:, None], (1, 6))  # same smooth signal in all 6 dims
    T_code, b = probe.fit_pose_code(T, degree=4, coef_bits=16, resid_keep_frac=0.0)
    dp = probe.dpose_of_code(T_code, T)
    assert dp < 1e-10, f"smooth poly must be fit exactly, got d_pose={dp}"


def test_jittery_signal_has_high_irreducible_dpose_under_smooth_code(probe):
    # The CORE FINDING: a high-frequency (jittery) signal CANNOT be coded by a low-degree poly.
    # A smooth-code d_pose on pure white noise must stay near the noise variance (NOT ~0).
    rng = np.random.default_rng(0)
    n = 600
    jitter = rng.normal(0, 1.0, size=(n, 6))  # var ~ 1 per dim
    T_code, _ = probe.fit_pose_code(jitter, degree=8, coef_bits=16, resid_keep_frac=0.0)
    dp = probe.dpose_of_code(T_code, jitter)
    # a deg-8 poly removes negligible variance from white noise -> d_pose stays O(1)
    assert dp > 0.5, f"jitter must not be cheaply codeable, got d_pose={dp}"


def test_fit_pose_code_bytes_monotone_in_degree(probe):
    T = np.tile(np.linspace(-1, 1, 100)[:, None], (1, 6))
    b2 = probe.fit_pose_code(T, degree=2, coef_bits=16)[1]
    b8 = probe.fit_pose_code(T, degree=8, coef_bits=16)[1]
    # more degree -> more coeff bytes (a real monotone byte model, not a constant)
    assert b8["total_bytes"] > b2["total_bytes"]
    assert b2["n_coef"] == 6 * 3 and b8["n_coef"] == 6 * 9


def test_residual_keep_adds_bytes_and_lowers_dpose(probe):
    # smooth trend + sparse spikes: residual coding should both add bytes AND lower d_pose
    n = 300
    t = np.linspace(-1, 1, n)
    base = np.tile((1.0 + t)[:, None], (1, 6))
    spikes = np.zeros((n, 6))
    spikes[::20] = 5.0  # sparse large residuals the poly can't fit
    T = base + spikes
    code0, b0 = probe.fit_pose_code(T, degree=3, resid_keep_frac=0.0)
    code1, b1 = probe.fit_pose_code(T, degree=3, resid_keep_frac=0.2)
    dp0 = probe.dpose_of_code(code0, T)
    dp1 = probe.dpose_of_code(code1, T)
    assert b1["total_bytes"] > b0["total_bytes"], "residual coding must cost bytes"
    assert dp1 < dp0, "coding the sparse residuals must lower d_pose"


def test_dpose_of_code_is_exact_contest_reduction(probe):
    a = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]])
    b = np.array([[1.0, 2.0, 4.0, 0.0, 0.0, 0.0]])  # differs by 1 in dim 2
    # contest reduction: mean over pairs of mean over 6 dims of squared diff = (1^2)/6
    expected = (1.0**2) / 6.0
    assert abs(probe.dpose_of_code(a, b) - expected) < 1e-12


# --------------------------------------------------------------------------
# Pose carrier degradation: real degradation + identity no-op
# --------------------------------------------------------------------------
def test_degrade_carrier_identity_is_noop(probe):
    rng = np.random.default_rng(2)
    f = rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)
    out = probe._degrade_pose_carrier(f, spatial_div=1, luma_bits=8, mode="bicubic")
    # div=1, 8b must be a no-op (the d_pose~0 sanity row)
    np.testing.assert_array_equal(out, f)


def test_degrade_carrier_block_average_kills_local_variation(probe):
    # A checkerboard's local variance must drop under block-averaging (the edge-killing failure mode)
    f = np.zeros((64, 64, 3), dtype=np.uint8)
    f[::2, ::2] = 255
    f[1::2, 1::2] = 255
    deg = probe._degrade_pose_carrier(f, spatial_div=8, luma_bits=8, mode="block")
    # local std of an 8x8 block must collapse vs the original checkerboard
    assert deg[:8, :8].std() < f[:8, :8].std() * 0.5


def test_degrade_carrier_luma_bits_quantizes(probe):
    rng = np.random.default_rng(3)
    f = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    deg = probe._degrade_pose_carrier(f, spatial_div=1, luma_bits=2, mode="bicubic")
    # 2-bit luma -> at most 4 distinct levels per channel
    assert len(np.unique(deg)) <= 4


def test_carrier_bytes_monotone_in_retained_dof(probe):
    b_coarse = probe._carrier_bytes(spatial_div=16, luma_bits=8, n_pairs=600)
    b_fine = probe._carrier_bytes(spatial_div=2, luma_bits=8, n_pairs=600)
    # finer grid retains more DOF -> more bytes (real monotone model)
    assert b_fine["total_amort_bytes"] > b_coarse["total_amort_bytes"]
    # coarse grid dims are H/div, W/div
    assert b_coarse["coarse_grid"] == [874 // 16, 1164 // 16]


# --------------------------------------------------------------------------
# Verdict logic: driven by measured flags, not hardcoded
# --------------------------------------------------------------------------
def test_verdict_red_when_vector_floor_not_cheap(probe):
    l1 = {"n_byte_cheap_AND_holds_loose": 0, "n_byte_cheap_AND_holds_tight": 0}
    l2 = {"min_bytes_carrier_holding_loose": None, "min_bytes_carrier_holding_tight": None}
    assert probe.compute_verdict(l1, l2) == "RED_EVEN_VECTOR_FLOOR_NEEDS_MANY_BYTES"


def test_verdict_green_when_both_cheap(probe):
    l1 = {"n_byte_cheap_AND_holds_loose": 1, "n_byte_cheap_AND_holds_tight": 1}
    cheap = {"byte_cheap_amort": True}
    l2 = {"min_bytes_carrier_holding_loose": cheap, "min_bytes_carrier_holding_tight": cheap}
    assert probe.compute_verdict(l1, l2) == "GREEN_POSE_CHEAP_AND_REALIZABLE_TIGHT"


def test_verdict_amber_when_vector_cheap_but_frames_costly(probe):
    l1 = {"n_byte_cheap_AND_holds_loose": 1, "n_byte_cheap_AND_holds_tight": 0}
    l2 = {"min_bytes_carrier_holding_loose": None, "min_bytes_carrier_holding_tight": None}
    assert probe.compute_verdict(l1, l2) == "AMBER_VECTOR_CHEAP_BUT_FRAMES_COST_MORE"


# --------------------------------------------------------------------------
# comma2k19 GT prior was actually downloaded (verified shape, not fabricated)
# --------------------------------------------------------------------------
def test_comma2k19_gt_pose_raw_is_the_verified_segment(probe):
    raw = probe.GT_POSE_RAW
    if not raw.exists():
        pytest.skip("comma2k19 GT not downloaded in this environment")
    g = np.load(raw, allow_pickle=True)
    # exactly 1200 frames @ 20Hz = the contest's 600 pairs * 2
    assert g["frame_positions"].shape == (1200, 3)
    assert g["frame_velocities"].shape == (1200, 3)
    assert g["frame_orientations"].shape == (1200, 4)
    assert str(g["segment_id"]) == "b0c9d2329ad1606b|2018-07-27--06-03-57/10"
    # quaternions are unit-norm (real orientation data, not zeros)
    qn = np.linalg.norm(g["frame_orientations"], axis=1)
    assert abs(qn.mean() - 1.0) < 1e-3
    # highway segment: ECEF speed ~ 30 m/s, very smooth (the ~1-2 DOF prior)
    spd = np.linalg.norm(g["frame_velocities"], axis=1)
    assert 20.0 < spd.mean() < 40.0
    assert np.diff(spd).std() < 0.1  # physical speed is smooth (NOT the PoseNet's jitter)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
