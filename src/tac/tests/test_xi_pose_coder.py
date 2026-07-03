# SPDX-License-Identifier: MIT
"""Tests for tac.boundary_math.xi_pose_coder — the #257 store-nothing derive-H + ξ entropy coder.

Covers the deliverable gates:
  * quantize/dequantize round-trip is bit-exact on the quantized grid;
  * BOTH coders (delta_ar / none) decode to the IDENTICAL quantized ξ (strict parity → the raw
    fallback is a correct reference for the arithmetic coder);
  * the arithmetic coder is LOSSLESS (decode(encode(q)) == q) at every precision + tricky inputs
    (all-zero channels, size-1 alphabets, P=1, negative deltas crossing zero);
  * derive-H (homographies_from_xi) == the per-pair module authority homography_from_xi_numpy
    BIT-FOR-BIT (deriving H introduces ZERO error → the uint8 warp is unchanged);
  * the coded variant is <= the raw variant on a smooth trajectory (the coder helps);
  * rate accounting is correct;
  * NO-MPS by construction (pure numpy + stdlib; the module never scores / never touches torch).

All TINY + CPU-only. NO GPU, NO MPS, NO touch of any live run.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math import xi_pose_coder as X
from tac.boundary_math import warp_real_luma_frame0 as W


def _smooth_xi(P: int, seed: int = 0) -> np.ndarray:
    """A smooth ego-twist trajectory (P,6) via the real calibration path (live-#205-style: fwd + jitter)."""
    rng = np.random.default_rng(seed)
    poses = np.zeros((P, 6))
    poses[:, 0] = 30.0 + 0.4 * np.arange(P) + 0.2 * rng.standard_normal(P)   # forward
    poses[:, 1] = 0.05 * np.sin(np.arange(P) / 7.0)                          # small lateral
    poses[:, 2] = -0.03 + 0.01 * rng.standard_normal(P)                      # small vertical
    poses[:, 3:] = 0.002 * rng.standard_normal((P, 3))                       # small rotation
    return np.stack([W.xi_from_pose_calibration(poses[p], 0.16, 1.0, 0.02) for p in range(P)])


# --------------------------------------------------------------------------- #
# 1. quantize / dequantize round-trip is exact on the grid
# --------------------------------------------------------------------------- #
def test_quantize_roundtrip_exact_on_grid():
    xi = _smooth_xi(40, 1)
    for ql in (512, 2048, 4096, 32767):
        q, scales = X.quantize_xi(xi, q_levels=ql)
        assert q.dtype == np.int16 and scales.dtype == np.float32
        assert np.abs(q).max() <= ql
        xi_dq = X.dequantize_xi(q, scales)
        q2, _ = X.quantize_xi(xi_dq, q_levels=ql)  # re-quantizing a grid point is a fixed point
        assert np.array_equal(q, q2), f"q_levels={ql}: dequant->quant is not a fixed point"


def test_quantize_all_zero_channel_no_div0():
    xi = _smooth_xi(20, 2)
    xi[:, 4] = 0.0  # an all-zero rotation channel (live #205 has s_r=0 -> 3 zero channels)
    q, scales = X.quantize_xi(xi)
    assert np.all(np.isfinite(scales)) and np.all(q[:, 4] == 0)


# --------------------------------------------------------------------------- #
# 2. the arithmetic coder is LOSSLESS + both variants strict-parity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("P", [1, 2, 5, 48, 128])
@pytest.mark.parametrize("ql", [512, 4096, 32767])
def test_coder_lossless_and_strict_parity(P, ql):
    xi = _smooth_xi(P, seed=P + ql)
    q, scales = X.quantize_xi(xi, q_levels=ql)
    raw = X.serialize_xi_payload(q, scales, coder="none")
    coded = X.serialize_xi_payload(q, scales, coder="delta_ar")
    q_raw, s_raw = X.parse_xi_payload(raw)
    q_cod, s_cod = X.parse_xi_payload(coded)
    assert np.array_equal(q_raw, q), "raw variant lost the quantized ξ"
    assert np.array_equal(q_cod, q), "delta_ar coder is NOT lossless"
    assert np.array_equal(q_raw, q_cod), "raw vs coded decode to DIFFERENT q (strict-parity broken)"
    assert np.array_equal(s_raw, scales) and np.array_equal(s_cod, scales)


def test_coder_handles_zero_crossing_and_size1_alphabet():
    # a channel oscillating through zero (negative deltas) + a constant channel (size-1 delta alphabet).
    P = 60
    xi = np.zeros((P, 6))
    xi[:, 0] = 0.1 * np.sin(np.arange(P) / 3.0)   # crosses zero -> negative deltas
    xi[:, 1] = 0.5                                 # constant -> all deltas 0 (size-1 alphabet)
    q, scales = X.quantize_xi(xi, q_levels=4096)
    coded = X.serialize_xi_payload(q, scales, coder="delta_ar")
    q2, _ = X.parse_xi_payload(coded)
    assert np.array_equal(q2, q)


# --------------------------------------------------------------------------- #
# 3. derive-H == the per-pair module authority BIT-FOR-BIT (deriving H introduces ZERO error)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pitch", [0.0, 0.02, -0.05])
def test_derive_H_bit_matches_module_authority(pitch):
    xi = _smooth_xi(50, 5)
    q, scales = X.quantize_xi(xi)
    xi_dq = X.dequantize_xi(q, scales)
    H_batch = X.homographies_from_xi(xi_dq, pitch)
    geom = W.GroundHomographyGeom.eon(pitch=pitch)
    for p in range(xi_dq.shape[0]):
        Hp = W.homography_from_xi_numpy(xi_dq[p], geom)
        assert np.array_equal(H_batch[p], Hp), f"pair {p}: batched derive-H != per-pair module authority"


def test_derive_H_render_unchanged_vs_stored_fp64_H():
    """DELIVERABLE: the derived H reproduces the fp64 H one would STORE from the same ξ, so dropping
    the stored H leaves the realized uint8 render BIT-IDENTICAL (d_pose invariant by construction)."""
    from tac.boundary_math import warp_real_luma_frame0 as WW

    pitch = 0.02
    xi = _smooth_xi(8, 6)
    q, scales = X.quantize_xi(xi)
    xi_dq = X.dequantize_xi(q, scales)
    geom = WW.GroundHomographyGeom.eon(pitch=pitch)
    src = np.random.default_rng(9).integers(0, 256, (WW.NATIVE_H, WW.NATIVE_W, 3)).astype(np.float64)
    H_derived = X.homographies_from_xi(xi_dq, pitch)
    for p in range(xi_dq.shape[0]):
        H_stored = WW.homography_from_xi_numpy(xi_dq[p], geom)   # what a store-H path would ship
        f_stored = WW.warp_frame0_uint8_numpy(src, xi_dq[p], geom)          # via internal H(ξ)==H_stored
        # warp with the DERIVED H directly (inverse-warp bilinear, uint8) — must be bit-identical
        Hinv = np.linalg.inv(H_derived[p])
        us, vs = np.meshgrid(np.arange(WW.NATIVE_W), np.arange(WW.NATIVE_H))
        grid = np.stack([us.ravel(), vs.ravel(), np.ones(WW.NATIVE_H * WW.NATIVE_W)], 0).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sh = Hinv @ grid; z = sh[2]; su = sh[0] / z; sv = sh[1] / z
        valid = (np.isfinite(su) & np.isfinite(sv) & (z > 0) & (su >= 0) & (su <= WW.NATIVE_W - 1)
                 & (sv >= 0) & (sv <= WW.NATIVE_H - 1))
        suc = np.clip(su, 0, WW.NATIVE_W - 1); svc = np.clip(sv, 0, WW.NATIVE_H - 1)
        x0 = np.floor(suc).astype(np.int64); y0 = np.floor(svc).astype(np.int64)
        x1 = np.minimum(x0 + 1, WW.NATIVE_W - 1); y1 = np.minimum(y0 + 1, WW.NATIVE_H - 1)
        wx = (suc - x0)[:, None]; wy = (svc - y0)[:, None]
        flat = src.reshape(-1, 3)
        Ia = flat[y0 * WW.NATIVE_W + x0]; Ib = flat[y0 * WW.NATIVE_W + x1]
        Ic = flat[y1 * WW.NATIVE_W + x0]; Id = flat[y1 * WW.NATIVE_W + x1]
        top = Ia * (1 - wx) + Ib * wx; bot = Ic * (1 - wx) + Id * wx
        samp = top * (1 - wy) + bot * wy
        out = np.where(valid[:, None], samp, flat).reshape(WW.NATIVE_H, WW.NATIVE_W, 3)
        f_derived = np.clip(np.round(out), 0, 255).astype(np.uint8)
        assert np.array_equal(H_derived[p], H_stored), f"pair {p}: derived H != stored fp64 H"
        assert np.array_equal(f_derived, f_stored), f"pair {p}: derived-H render != stored-H render"


# --------------------------------------------------------------------------- #
# 4. the coder helps on a smooth trajectory + rate accounting is correct
# --------------------------------------------------------------------------- #
def test_coder_beats_raw_on_smooth_trajectory_and_rate_report():
    xi = _smooth_xi(600, 7)
    rep = X.xi_payload_rate_report(xi, pitch=0.02, q_levels=4096)
    assert rep["round_trip_bit_exact"] and rep["raw_coded_strict_parity"]
    assert rep["raw_bytes"] == 600 * 6 * 2 + len(X._XI_MAGIC) + 4 + 6 * 4  # magic+hdr(4)+scales(24)+q
    assert rep["coded_bytes"] < rep["raw_bytes"], "the coder must beat raw on a smooth trajectory"
    denom = 37_545_489.0
    assert abs(rep["coded_rate"] - 25.0 * rep["coded_bytes"] / denom) < 1e-12
    assert rep["coded_vs_raw_ratio"] == rep["coded_bytes"] / rep["raw_bytes"]


# --------------------------------------------------------------------------- #
# 5. NO-MPS by construction: pure numpy + stdlib, never scores / never imports torch
# --------------------------------------------------------------------------- #
def test_module_is_pure_numpy_no_torch_no_mps():
    import inspect
    src = inspect.getsource(X)
    assert "import torch" not in src and "mps" not in src.lower(), \
        "xi_pose_coder must be pure numpy/stdlib (NO torch/MPS) — it produces bytes, never a score"


# --------------------------------------------------------------------------- #
# 6. bad inputs fail closed (NO-FAKE)
# --------------------------------------------------------------------------- #
def test_bad_inputs_raise():
    with pytest.raises(X.XiPoseCoderError):
        X.quantize_xi(np.zeros((5, 4)))          # not 6-dim
    with pytest.raises(X.XiPoseCoderError):
        X.quantize_xi(_smooth_xi(4), q_levels=0)  # invalid precision
    with pytest.raises(X.XiPoseCoderError):
        X.parse_xi_payload(b"NOTXIP2" + b"\x00" * 8)  # bad magic
    with pytest.raises(X.XiPoseCoderError):
        q, s = X.quantize_xi(_smooth_xi(3))
        X.serialize_xi_payload(q, s, coder="bogus")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
