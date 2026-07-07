# SPDX-License-Identifier: MIT
"""Tests for tac.boundary_math.xi_spline_residual_coder — the ξ spline-predictor residual coder.

Deliverable gates:
  * the inlined numpy spline fit/eval are BIT-PARITY with the canonical oracles
    (``ego_xi_trajectory.fit_se3_bspline_controls`` / ``se3_bspline.se3_bspline_eval_numpy``) —
    the decode half never imports the mlx-touching module, so parity is pinned here;
  * ALL THREE residual schemes ({varint, zlib9, rice}) are individually LOSSLESS;
  * the full container round-trip (``coder="spline_residual"``) is BIT-IDENTICAL to the source
    quantized table at multiple P / knot counts / q_levels (the NO-FAKE gate);
  * the DEFAULT ``xi_pose_coder`` path is byte-identical to the pre-spline coder (default is
    still delta_ar; the delta_ar/none code paths are untouched);
  * the coder module stays numpy+stdlib on the DECODE path (no mlx/torch import);
  * bad inputs fail closed.

All TINY + CPU-only. NO GPU, NO MPS, NO touch of any live run.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math import xi_pose_coder as X
from tac.boundary_math import xi_spline_residual_coder as SR
from tac.boundary_math import warp_real_luma_frame0 as W


def _smooth_xi(P: int, seed: int = 0) -> np.ndarray:
    """A smooth ego-twist trajectory (P,6) via the real calibration path (live-#205-style)."""
    rng = np.random.default_rng(seed)
    poses = np.zeros((P, 6))
    poses[:, 0] = 30.0 + 0.4 * np.arange(P) + 0.2 * rng.standard_normal(P)
    poses[:, 1] = 0.05 * np.sin(np.arange(P) / 7.0)
    poses[:, 2] = -0.03 + 0.01 * rng.standard_normal(P)
    poses[:, 3:] = 0.002 * rng.standard_normal((P, 3))
    return np.stack([W.xi_from_pose_calibration(poses[p], 0.16, 1.0, 0.02) for p in range(P)])


# --------------------------------------------------------------------------- #
# 1. inlined predictor parity vs the canonical oracles (exact)
# --------------------------------------------------------------------------- #
def test_inlined_fit_bit_parity_with_ego_xi_trajectory():
    from tac.boundary_math.ego_xi_trajectory import fit_se3_bspline_controls

    xi = _smooth_xi(40, 1)
    xi_lift = np.concatenate([np.zeros((1, 6)), xi], axis=0)
    for M in (4, 8, 16, 41):
        ours = SR._fit_spline_controls(xi_lift, M)
        ref = fit_se3_bspline_controls(xi_lift, M)
        assert np.array_equal(ours, ref), f"M={M}: inlined fit != canonical fit"


def test_inlined_eval_bit_parity_with_se3_bspline_numpy_oracle():
    from tac.lie.se3_bspline import se3_bspline_eval_numpy

    xi = _smooth_xi(30, 2)
    xi_lift = np.concatenate([np.zeros((1, 6)), xi], axis=0)
    ctrl = SR._fit_spline_controls(xi_lift, 8)
    tt = np.linspace(0.0, 5.0, 31)
    ours = SR._spline_eval(ctrl, tt)
    ref = se3_bspline_eval_numpy(ctrl, tt)
    assert np.array_equal(ours, ref), "inlined spline eval != se3_bspline_eval_numpy"


# --------------------------------------------------------------------------- #
# 2. each residual scheme is individually LOSSLESS (incl. adversarial residuals)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scheme", SR.RESIDUAL_SCHEMES)
def test_each_residual_scheme_lossless(scheme):
    rng = np.random.default_rng(3)
    res = np.concatenate([
        rng.integers(-4, 5, (200, 6)),                 # typical small residuals
        rng.integers(-3000, 3001, (8, 6)),             # rare large outliers
        np.zeros((16, 6), dtype=np.int64),             # zero runs
    ]).astype(np.int64)
    blob = SR._encode_residual(res, scheme)
    scheme_id = SR._SCHEME_IDS[scheme]
    out = SR._decode_residual(blob, scheme_id, res.shape[0], 6)
    assert np.array_equal(out, res), f"{scheme}: residual round-trip lost data"


def test_measure_residual_schemes_reports_all_three_real_sizes():
    res = np.random.default_rng(4).integers(-6, 7, (120, 6)).astype(np.int64)
    sizes = SR.measure_residual_schemes(res)
    assert set(sizes) == set(SR.RESIDUAL_SCHEMES)
    for s, b in sizes.items():
        assert b == len(SR._encode_residual(res, s)), f"{s}: reported size != real encoded bytes"


# --------------------------------------------------------------------------- #
# 3. full container round-trip is BIT-IDENTICAL (the NO-FAKE gate)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("P,M", [(20, 4), (60, 8), (60, 16), (128, 32)])
@pytest.mark.parametrize("ql", [512, 4096])
def test_container_roundtrip_bit_identical(P, M, ql):
    xi = _smooth_xi(P, seed=P + M + ql)
    q, scales = X.quantize_xi(xi, q_levels=ql)
    blob = X.serialize_xi_payload(q, scales, coder="spline_residual", spline_knots=M)
    q2, s2 = X.parse_xi_payload(blob)
    assert np.array_equal(q2, q), "spline_residual container decode != source table"
    assert np.array_equal(s2, scales)
    assert X.decode_xi_payload(blob).shape == (P, 6)


def test_all_zero_channels_roundtrip():
    """live #205 has s_r=0 → 3 all-zero rotation channels; must survive predict+residual."""
    xi = _smooth_xi(50, 9)
    xi[:, 3:] = 0.0
    q, scales = X.quantize_xi(xi, q_levels=4096)
    blob = X.serialize_xi_payload(q, scales, coder="spline_residual", spline_knots=8)
    q2, _ = X.parse_xi_payload(blob)
    assert np.array_equal(q2, q)


def test_rate_report_measures_and_is_consistent():
    xi = _smooth_xi(200, 11)
    q, scales = X.quantize_xi(xi, q_levels=4096)
    rep = SR.spline_residual_rate_report(q, scales, knots=16)
    assert rep["bit_identical"] is True
    assert set(rep["residual_scheme_bytes"]) == set(SR.RESIDUAL_SCHEMES)
    picked = rep["picked_scheme"]
    assert rep["residual_scheme_bytes"][picked] == min(rep["residual_scheme_bytes"].values())
    # total container >= knots + picked residual (+ headers); rate accounting exact
    assert rep["total_payload_bytes"] > rep["knot_payload_bytes"]
    assert abs(rep["rate_term"] - 25.0 * rep["total_payload_bytes"] / 37_545_489.0) < 1e-15


# --------------------------------------------------------------------------- #
# 3b. delta_res (the MEASURED n600 winner): lossless + edge cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("P", [1, 2, 5, 60, 300])
def test_delta_res_container_roundtrip_bit_identical(P):
    xi = _smooth_xi(P, seed=P + 77)
    q, scales = X.quantize_xi(xi, q_levels=4096)
    blob = X.serialize_xi_payload(q, scales, coder="delta_res")
    q2, s2 = X.parse_xi_payload(blob)
    assert np.array_equal(q2, q), "delta_res container decode != source table"
    assert np.array_equal(s2, scales)


def test_delta_res_pinned_schemes_all_lossless():
    q, _ = X.quantize_xi(_smooth_xi(80, 78), q_levels=4096)
    for scheme in SR.RESIDUAL_SCHEMES:
        body = SR.encode_delta_res_body(q, scheme=scheme)
        q2, off = SR.decode_delta_res_body(body, 0, q.shape[0], q.shape[1])
        assert off == len(body) and np.array_equal(q2, q), f"delta_res[{scheme}] lost data"


# --------------------------------------------------------------------------- #
# 4. the DEFAULT xi_pose_coder path is byte-identical to the pre-spline coder
# --------------------------------------------------------------------------- #
def test_default_coder_unchanged_and_byte_identical_to_delta_ar():
    import inspect

    # the default kwarg is still delta_ar
    sig = inspect.signature(X.serialize_xi_payload)
    assert sig.parameters["coder"].default == "delta_ar"
    xi = _smooth_xi(80, 12)
    q, scales = X.quantize_xi(xi)
    default_bytes = X.serialize_xi_payload(q, scales)
    delta_ar_bytes = X.serialize_xi_payload(q, scales, coder="delta_ar")
    assert default_bytes == delta_ar_bytes, "DEFAULT payload must be the delta_ar path, byte-identical"
    # coder ids are stable (the shipped inflate's verbatim copies depend on them)
    assert (X._CODER_RAW, X._CODER_DELTA_AR, X._CODER_SPLINE_RESIDUAL, X._CODER_DELTA_RES) \
        == (0, 1, 2, 3)
    # and the delta_ar body layout is untouched: header byte says cid=1
    assert default_bytes[len(X._XI_MAGIC)] == X._CODER_DELTA_AR


def test_strict_parity_across_all_four_coders():
    xi = _smooth_xi(90, 13)
    q, scales = X.quantize_xi(xi)
    decoded = {}
    for coder in ("none", "delta_ar", "spline_residual", "delta_res"):
        blob = X.serialize_xi_payload(q, scales, coder=coder, spline_knots=8)
        decoded[coder], _ = X.parse_xi_payload(blob)
    for coder, qd in decoded.items():
        assert np.array_equal(qd, q), f"{coder}: all coders must decode to the IDENTICAL q"


# --------------------------------------------------------------------------- #
# 5. decode path stays numpy+stdlib (inflate-portable): no mlx / torch imports
# --------------------------------------------------------------------------- #
def test_module_decode_path_pure_numpy_no_mlx_no_torch():
    import inspect

    src = inspect.getsource(SR)
    assert "import torch" not in src, "coder must never import torch"
    assert "import mlx" not in src and "from mlx" not in src, \
        "coder must never import mlx (decode is inflate-portable numpy+stdlib)"
    assert "se3_bspline" not in [ln.split()[1] for ln in src.splitlines()
                                 if ln.startswith(("import ", "from ")) and len(ln.split()) > 1], \
        "coder must not import tac.lie.se3_bspline (it imports mlx at module top)"


# --------------------------------------------------------------------------- #
# 6. bad inputs fail closed (NO-FAKE)
# --------------------------------------------------------------------------- #
def test_bad_inputs_raise():
    xi = _smooth_xi(20, 14)
    q, scales = X.quantize_xi(xi)
    with pytest.raises(SR.XiSplineResidualError):
        SR.encode_spline_residual_body(q, scales, knots=3)      # cubic needs >= 4
    with pytest.raises(SR.XiSplineResidualError):
        SR.encode_spline_residual_body(q, scales, knots=22)     # > P+1 path poses
    with pytest.raises(SR.XiSplineResidualError):
        SR.encode_spline_residual_body(np.zeros((5, 4), np.int16), np.ones(4, np.float32))
    with pytest.raises(SR.XiSplineResidualError):
        SR._encode_residual(np.zeros((4, 6), np.int64), "bogus")
    # XiSplineResidualError is a XiPoseCoderError (one except-clause catches the family)
    assert issubclass(SR.XiSplineResidualError, X.XiPoseCoderError)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
