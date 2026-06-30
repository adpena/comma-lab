# SPDX-License-Identifier: MIT
"""CPU tests for the mx.compile d_seg-step wrapper (mlx_compile_step).

``mx.compile`` is device-agnostic, so bit-identity + determinism are validated on
the CPU device (``mx.set_default_device(mx.cpu)`` — non-contending with the live
GPU training arm). Pose is excluded by construction (the representative loss is
seg-only).
"""

from __future__ import annotations

import numpy as np
import pytest


def _mlx_cpu():
    try:
        import mlx.core as mx
        import mlx.nn  # noqa: F401  (ensure nn importable)
    except Exception:  # pragma: no cover - environment guard
        pytest.skip("mlx not importable")
    mx.set_default_device(mx.cpu)
    return mx


def test_compile_loss_and_grad_disabled_is_identity():
    from tac.local_acceleration.mlx_compile_step import compile_loss_and_grad

    sentinel = lambda *a: ("loss", a)  # noqa: E731
    assert compile_loss_and_grad(sentinel, enabled=False) is sentinel


def test_representative_trunk_compile_equivalent_and_deterministic():
    _mlx_cpu()
    from tac.local_acceleration.mlx_compile_step import (
        assert_compile_bit_identical,
        build_representative_dseg_trunk,
        representative_dseg_loss_and_grad,
    )

    model = build_representative_dseg_trunk(seed=0)
    lg, args = representative_dseg_loss_and_grad(model, seed=0)
    # Equivalence within the fp-fusion bound (spec: <1e-6); determinism EXACT.
    report = assert_compile_bit_identical(lg, args, atol=1e-6, deterministic_runs=3)
    assert report["equivalent_within_tol"] is True
    assert report["deterministic"] is True
    assert report["loss_max_abs_delta"] < 1e-6
    assert report["grad_max_abs_delta"] < 1e-6
    assert report["determinism_max_abs_delta"] == 0.0  # compiled graph is byte-stable
    assert report["n_grad_leaves"] > 0


def test_compiled_loss_value_matches_uncompiled_within_tol():
    mx = _mlx_cpu()
    from tac.local_acceleration.mlx_compile_step import (
        build_representative_dseg_trunk,
        compile_loss_and_grad,
        representative_dseg_loss_and_grad,
    )

    model = build_representative_dseg_trunk(seed=1)
    lg, args = representative_dseg_loss_and_grad(model, seed=1)
    loss_u, _ = lg(*args)
    compiled = compile_loss_and_grad(lg, enabled=True)
    loss_c, _ = compiled(*args)
    mx.eval(loss_u, loss_c)
    lu, lc = float(np.asarray(loss_u)), float(np.asarray(loss_c))
    assert np.isfinite(lc)
    assert abs(lu - lc) < 1e-6  # graph-fusion ULP-scale, same math


def test_harness_detects_real_delta_with_strict_atol():
    # Non-no-op guard: with atol below the fp-fusion noise floor, the gate MUST fire
    # on the genuine compiled-vs-uncompiled ULP-scale delta (proves the gate compares).
    _mlx_cpu()
    from tac.local_acceleration.mlx_compile_step import (
        assert_compile_bit_identical,
        build_representative_dseg_trunk,
        representative_dseg_loss_and_grad,
    )

    model = build_representative_dseg_trunk(seed=2)
    lg, args = representative_dseg_loss_and_grad(model, seed=2)
    with pytest.raises(AssertionError):
        assert_compile_bit_identical(lg, args, atol=-1.0)
