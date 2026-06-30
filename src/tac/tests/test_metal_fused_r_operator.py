# SPDX-License-Identifier: MIT
"""CPU parity tests for the fused contest-faithful R-operator (metal_fused_r_operator).

These are CPU/no-GPU tests: the numpy oracle is validated for self-consistency and
(when MLX imports) bit-faithfulness to the MLX production R on the **CPU device**
(``mx.set_default_device(mx.cpu)`` — non-contending with the live GPU training arm).
The Metal kernel itself needs a GPU and is validated by the GPU-gated harness
``assert_metal_matches_cpu_oracle`` (NOT exercised here).
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.local_acceleration.metal_fused_r_operator import (
    CUBIC_A,
    fused_r_forward_numpy,
    fused_r_vjp_numpy,
    resize_indices_weights_numpy,
)

# Small, fast shapes for CPU.
_SMALL = dict(in_hw=(24, 32), camera_hw=(55, 73), output_hw=(24, 32))


def _mlx_cpu():
    """Return mlx.core pinned to the CPU device, or skip if unavailable."""

    try:
        import mlx.core as mx
    except Exception:  # pragma: no cover - environment guard
        pytest.skip("mlx not importable")
    mx.set_default_device(mx.cpu)
    return mx


# --------------------------------------------------------------------------- #
# numpy oracle self-consistency
# --------------------------------------------------------------------------- #


def test_forward_numpy_shape_dtype_and_leading_dims():
    rng = np.random.default_rng(0)
    x = (rng.random((2, 3, 24, 32, 3)) * 255.0).astype(np.float32)
    y = fused_r_forward_numpy(x, camera_hw=(55, 73), output_hw=(24, 32))
    assert y.shape == (2, 3, 24, 32, 3)
    assert y.dtype == np.float32
    # Output is bilinear of a uint8 camera frame => within [0, 255].
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 255.0


def test_forward_numpy_rejects_bad_shape():
    with pytest.raises(ValueError):
        fused_r_forward_numpy(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        fused_r_forward_numpy(np.zeros((2, 4, 4, 4), dtype=np.float32))  # last dim != 3


def test_indices_weights_bilinear_partition_of_unity():
    idx, w = resize_indices_weights_numpy(in_size=24, out_size=55, mode="bilinear")
    assert idx.shape == (55, 2)
    assert w.shape == (55, 2)
    # Bilinear weights always sum to 1.
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-6)
    assert idx.min() >= 0 and idx.max() <= 23


def test_indices_weights_bicubic_taps_and_clip():
    idx, w = resize_indices_weights_numpy(in_size=24, out_size=55, mode="bicubic")
    assert idx.shape == (55, 4)
    assert w.shape == (55, 4)
    # Interior bicubic coefficients sum to 1 (cubic convolution, a=-0.75).
    interior = w[5:50].sum(axis=1)
    np.testing.assert_allclose(interior, 1.0, atol=1e-5)
    assert idx.min() >= 0 and idx.max() <= 23


def test_cubic_a_is_pytorch_default():
    assert CUBIC_A == -0.75  # matches the REAL MLX production R, not the loose memo a=-0.5


def test_identity_resize_is_noop_then_round():
    # in==out for both axes => up is identity then clip+round @ camera... but here
    # camera==in==out so the only change is uint8 quantization round-trip.
    rng = np.random.default_rng(1)
    x = (rng.random((1, 16, 20, 3)) * 255.0).astype(np.float32)
    y = fused_r_forward_numpy(x, camera_hw=(16, 20), output_hw=(16, 20))
    # camera == in (identity bicubic) then round; down camera==out identity bilinear.
    np.testing.assert_array_equal(y, np.rint(np.clip(x, 0, 255)).astype(np.float32))


# --------------------------------------------------------------------------- #
# numpy oracle == MLX production R (CPU device, bit-faithful or sub-LSB)
# --------------------------------------------------------------------------- #


def test_numpy_oracle_matches_mlx_indices_weights():
    mx = _mlx_cpu()
    from tac.local_acceleration import pr95_hnerv_mlx_training as P

    for mode in ("bilinear", "bicubic"):
        for in_size, out_size in [(24, 55), (55, 24), (32, 73), (73, 32)]:
            idx_np, w_np = resize_indices_weights_numpy(
                in_size=in_size, out_size=out_size, mode=mode
            )
            idx_mx, w_mx = P._resize_indices_weights(
                in_size=in_size, out_size=out_size, mode=mode
            )
            np.testing.assert_array_equal(idx_np, np.asarray(idx_mx))
            np.testing.assert_allclose(w_np, np.asarray(w_mx), atol=0.0, rtol=0.0)


def test_numpy_oracle_forward_matches_mlx_production_R():
    mx = _mlx_cpu()
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc,
    )

    rng = np.random.default_rng(7)
    x = (rng.random((2, _SMALL["in_hw"][0], _SMALL["in_hw"][1], 3)) * 255.0).astype(np.float32)
    y_np = fused_r_forward_numpy(
        x, camera_hw=_SMALL["camera_hw"], output_hw=_SMALL["output_hw"], ste_round=True
    )
    y_mx = np.asarray(
        apply_contest_faithful_roundtrip_nhwc(
            mx.array(x),
            camera_hw=_SMALL["camera_hw"],
            output_hw=_SMALL["output_hw"],
            ste_round=True,
        )
    )
    max_abs = float(np.max(np.abs(y_np - y_mx)))
    # The round @ camera can amplify a sub-LSB float-reduction diff to <= 1 LSB at a
    # handful of half-boundary pixels. Assert the bulk is bit-identical and the worst
    # case is <= 1 LSB (the documented sub-LSB note).
    assert max_abs <= 1.0, f"numpy oracle vs MLX R max|Δ|={max_abs} exceeds 1 LSB"
    frac_exact = float(np.mean(y_np == y_mx))
    assert frac_exact >= 0.999, f"only {frac_exact:.5f} pixels bit-identical"


def test_numpy_vjp_matches_mlx_autodiff():
    mx = _mlx_cpu()
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc,
    )

    rng = np.random.default_rng(11)
    # Use values away from the [0,255] clip boundary so the clip mask is all-ones
    # (the analytic and autodiff grads then agree to float tol everywhere).
    x = (rng.random((1, _SMALL["in_hw"][0], _SMALL["in_hw"][1], 3)) * 200.0 + 28.0).astype(
        np.float32
    )
    y_shape = fused_r_forward_numpy(
        x, camera_hw=_SMALL["camera_hw"], output_hw=_SMALL["output_hw"]
    ).shape
    cot = rng.standard_normal(y_shape).astype(np.float32)

    def ref(z):
        return apply_contest_faithful_roundtrip_nhwc(
            z, camera_hw=_SMALL["camera_hw"], output_hw=_SMALL["output_hw"], ste_round=True
        )

    _, (gx_mx,) = mx.vjp(ref, (mx.array(x),), (mx.array(cot),))
    gx_mx = np.asarray(gx_mx)
    gx_np = fused_r_vjp_numpy(
        x, cot, camera_hw=_SMALL["camera_hw"], output_hw=_SMALL["output_hw"], ste_round=True
    )
    np.testing.assert_allclose(gx_np, gx_mx, rtol=2e-3, atol=2e-4)


def test_vjp_clip_kills_gradient_fully_saturated():
    # Input above 255 => EVERY camera pixel (interior 400*1=400; edges 400*wsum with
    # wsum in ~[0.9,1.1] => >=360) is clipped to 255 => clip mask 0 everywhere =>
    # the gradient is EXACTLY zero (STE round passes through, clip kills it).
    x_over = np.full((1, 12, 16, 3), 400.0, dtype=np.float32)
    y_shape = fused_r_forward_numpy(x_over, camera_hw=(27, 35), output_hw=(12, 16)).shape
    cot = np.ones(y_shape, dtype=np.float32)
    gx = fused_r_vjp_numpy(x_over, cot, camera_hw=(27, 35), output_hw=(12, 16))
    assert float(np.max(np.abs(gx))) == 0.0


def test_vjp_ste_passthrough_interior():
    # Mid-range input stays comfortably inside (0,255) after bicubic => clip mask is
    # all-ones => the STE passes the round straight through => the grad equals the
    # no-round linear transpose grad EXACTLY (STE passthrough property).
    rng = np.random.default_rng(3)
    x_mid = (rng.random((1, 12, 16, 3)) * 60.0 + 100.0).astype(np.float32)  # ~[100, 160]
    y_shape = fused_r_forward_numpy(x_mid, camera_hw=(27, 35), output_hw=(12, 16)).shape
    cot = rng.standard_normal(y_shape).astype(np.float32)
    g_round = fused_r_vjp_numpy(
        x_mid, cot, camera_hw=(27, 35), output_hw=(12, 16), ste_round=True
    )
    g_noround = fused_r_vjp_numpy(
        x_mid, cot, camera_hw=(27, 35), output_hw=(12, 16), ste_round=False
    )
    assert float(np.max(np.abs(g_round))) > 0.0
    np.testing.assert_allclose(g_round, g_noround, atol=1e-6)


def test_ste_round_false_keeps_float_camera():
    rng = np.random.default_rng(5)
    x = (rng.random((1, 16, 20, 3)) * 255.0).astype(np.float32)
    y_round = fused_r_forward_numpy(x, camera_hw=(16, 20), output_hw=(16, 20), ste_round=True)
    y_float = fused_r_forward_numpy(x, camera_hw=(16, 20), output_hw=(16, 20), ste_round=False)
    # Identity resizes => round version is rint(x), float version is x (clipped).
    np.testing.assert_array_equal(y_round, np.rint(np.clip(x, 0, 255)))
    np.testing.assert_allclose(y_float, np.clip(x, 0, 255), atol=0.0)
