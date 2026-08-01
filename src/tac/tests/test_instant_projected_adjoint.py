# SPDX-License-Identifier: MIT
"""Boundary tests for the OSS-faithful INSTANT pointwise adjoint."""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.instant_projected_adjoint import (
    ProjectionProof,
    calibrate_adaptive_projector_numpy,
    instant_pointwise_conv2d,
    load_calibration,
    pointwise_input_adjoint_numpy,
    project_cotangent_mlx,
    project_cotangent_numpy,
    project_cotangent_torch,
    save_calibration,
)


def _rng_values(*, samples: int, channels: int, height: int, width: int) -> np.ndarray:
    return np.random.default_rng(20260712).normal(size=(samples, 1, channels, height, width))


def test_calibration_selects_smaller_axis_and_official_rank_law() -> None:
    channel_values = _rng_values(samples=3, channels=4, height=3, width=3)
    channel = calibrate_adaptive_projector_numpy(channel_values, energy_target=0.80, oversampling=2)
    assert channel.axis == "channels"
    assert channel.rank == min(channel.channels, channel.base_rank + 2)
    assert channel.retained_energy >= 0.80

    spatial_values = _rng_values(samples=3, channels=12, height=2, width=2)
    spatial = calibrate_adaptive_projector_numpy(spatial_values, energy_target=0.80, oversampling=2)
    assert spatial.axis == "spatial"
    assert spatial.rank == min(4, spatial.base_rank + 2)
    assert spatial.retained_energy >= 0.80


@pytest.mark.parametrize(
    ("channels", "height", "width", "axis"),
    [(3, 3, 3, "channels"), (8, 2, 2, "spatial")],
)
def test_full_rank_projector_is_exact_numpy_and_torch(
    channels: int, height: int, width: int, axis: str
) -> None:
    torch = pytest.importorskip("torch")
    values = _rng_values(samples=3, channels=channels, height=height, width=width)
    calibration = calibrate_adaptive_projector_numpy(values, energy_target=1.0, oversampling=5)
    assert calibration.axis == axis
    observed_numpy = project_cotangent_numpy(values[0], calibration)
    np.testing.assert_allclose(observed_numpy, values[0], rtol=1e-12, atol=1e-12)
    observed_torch = project_cotangent_torch(torch.tensor(values[0]), calibration).numpy()
    np.testing.assert_allclose(observed_torch, observed_numpy, rtol=1e-12, atol=1e-12)


def test_pointwise_torch_backward_matches_numpy_reference_and_forward_is_exact() -> None:
    torch = pytest.importorskip("torch")
    generator = np.random.default_rng(7)
    x = torch.tensor(generator.normal(size=(1, 3, 3, 3)), dtype=torch.float64, requires_grad=True)
    weight = torch.tensor(generator.normal(size=(4, 3, 1, 1)), dtype=torch.float64)
    bias = torch.tensor(generator.normal(size=(4,)), dtype=torch.float64)
    bank = generator.normal(size=(3, 1, 4, 3, 3))
    calibration = calibrate_adaptive_projector_numpy(bank, energy_target=0.75, oversampling=0)
    proof = ProjectionProof()

    exact = torch.nn.functional.conv2d(x, weight, bias)
    observed = instant_pointwise_conv2d(x, weight, bias, calibration, proof=proof)
    assert torch.equal(observed, exact)
    cotangent = torch.tensor(generator.normal(size=tuple(observed.shape)), dtype=torch.float64)
    observed.backward(cotangent)
    expected = pointwise_input_adjoint_numpy(cotangent.numpy(), weight.numpy(), calibration)
    np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-12, atol=1e-12)
    assert proof.backward_calls == 1
    assert proof.channel_axis_calls + proof.spatial_axis_calls == 1
    assert proof.dense_conv2d_input_calls == 0


def test_calibration_persistence_is_content_authenticated(tmp_path) -> None:
    calibration = calibrate_adaptive_projector_numpy(
        _rng_values(samples=2, channels=3, height=2, width=2), energy_target=0.95, oversampling=5
    )
    path = tmp_path / "projector.npz"
    save_calibration(path, calibration)
    loaded = load_calibration(path)
    assert loaded.metadata() == calibration.metadata()
    np.testing.assert_array_equal(loaded.basis, calibration.basis)

    with np.load(path, allow_pickle=False) as payload:
        basis = np.asarray(payload["basis"]).copy()
        singular = np.asarray(payload["singular_values"]).copy()
        metadata = np.asarray(payload["metadata"]).copy()
    basis.flat[0] += 0.25
    np.savez(path, basis=basis, singular_values=singular, metadata=metadata)
    with pytest.raises(ValueError, match="fingerprint"):
        load_calibration(path)


def test_calibration_save_atomically_replaces_complete_stage(tmp_path) -> None:
    values = _rng_values(samples=3, channels=4, height=3, width=3)
    first = calibrate_adaptive_projector_numpy(values, energy_target=0.80, oversampling=0)
    second = calibrate_adaptive_projector_numpy(values, energy_target=0.99, oversampling=0)
    path = tmp_path / "stage.npz"
    save_calibration(path, first)
    save_calibration(path, second)
    loaded = load_calibration(path)
    assert loaded.energy_target == second.energy_target
    assert loaded.calibration_fingerprint == second.calibration_fingerprint
    assert not list(tmp_path.glob(".*.tmp"))


def test_ineligible_convolutions_and_bad_calibration_fail_closed() -> None:
    torch = pytest.importorskip("torch")
    calibration = calibrate_adaptive_projector_numpy(
        _rng_values(samples=2, channels=4, height=3, width=3), energy_target=0.95, oversampling=5
    )
    x = torch.ones((1, 3, 3, 3), requires_grad=True)
    with pytest.raises(ValueError, match="1x1"):
        instant_pointwise_conv2d(x, torch.ones((4, 3, 3, 3)), None, calibration)
    trainable = torch.ones((4, 3, 1, 1), requires_grad=True)
    with pytest.raises(ValueError, match="frozen"):
        instant_pointwise_conv2d(x, trainable, None, calibration)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        calibrate_adaptive_projector_numpy(
            np.full((1, 1, 4, 3, 3), np.nan), energy_target=0.95, oversampling=5
        )


def test_mlx_projection_matches_numpy_when_metal_is_available() -> None:
    """MLX projector vs the NumPy reference, split by MLX stream.

    This assertion used to be a single ``rtol=1e-5`` comparison against the default (GPU) stream,
    and it was RED for as long as anyone has run it on Metal (CI cannot execute MLX). ddm_tr6
    localized it on 2026-08-01 and it is NOT a fp32-vs-fp64 artifact and NOT our arithmetic:

      * MEASURED -- the numpy reference computed in float32 vs float64 agrees to 1.4e-07, so the
        reference is precision-insensitive here.
      * MEASURED -- the MLX intermediates ``matrix`` and ``matrix @ q.T`` are BIT-IDENTICAL to
        numpy. Only the final ``@ q`` diverges, by 1.0e-03 relative.
      * MEASURED -- running the identical MLX graph on the CPU stream gives 6.0e-08.
      * MEASURED (standalone ``mx.matmul``, MLX 0.31.2) -- normalized error ``|err| / (|a| @ |b|)``
        is 1.7e-03 .. 1.7e-04 on the GPU stream across shapes from (4,1,3) to (1024,1024,256),
        versus 3.4e-08 .. 2.7e-07 on the CPU stream. A 1x1x1 matmul is exact on both.
      * INFERRED -- MLX dispatches to a reduced-precision tiled/simdgroup GEMM on Metal above a
        very small size threshold. Mechanism not confirmed upstream; the numbers above are.

    So the honest contract is per-stream. CPU is held to fp32 (that is the real test of OUR math).
    The GPU deviation is PINNED with a measured bound rather than papered over with a loose global
    tolerance: a bound that is tight enough to catch an actual algorithmic break, and that will
    fail loudly if MLX's GPU precision gets worse. Note this projector shapes ADJOINTS, so the
    GPU-side error is a gradient-quality fact worth keeping visible.
    """
    mx = pytest.importorskip("mlx.core")
    values = _rng_values(samples=2, channels=3, height=2, width=2)
    calibration = calibrate_adaptive_projector_numpy(values, energy_target=0.7, oversampling=0)
    expected = project_cotangent_numpy(values[0], calibration)

    try:
        with mx.stream(mx.cpu):
            observed_cpu = project_cotangent_mlx(mx.array(values[0]), calibration)
            mx.eval(observed_cpu)
    except RuntimeError as exc:
        pytest.skip(f"MLX runtime unavailable: {exc}")

    # fp32-exact on the CPU stream: this is the assertion that actually tests our arithmetic.
    np.testing.assert_allclose(np.asarray(observed_cpu), expected, rtol=1e-5, atol=1e-5)

    observed_gpu = project_cotangent_mlx(mx.array(values[0]), calibration)
    mx.eval(observed_gpu)
    gpu_rel = float(
        np.abs(np.asarray(observed_gpu) - expected).max() / (np.abs(expected).max() + 1e-30)
    )
    # Measured 1.02e-03 on MLX 0.31.2 / M-series. 1e-02 leaves ~10x headroom for MLX build drift
    # while still refusing an algorithmic break (wrong axis, wrong basis => O(1) relative).
    assert gpu_rel < 1.0e-2, (
        f"MLX GPU-stream projection deviates from the NumPy reference by {gpu_rel:.3g} relative, "
        "beyond the measured reduced-precision GEMM bound — suspect an algorithmic break, "
        "not rounding"
    )
