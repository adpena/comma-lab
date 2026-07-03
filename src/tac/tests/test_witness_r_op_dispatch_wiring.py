# SPDX-License-Identifier: MIT
"""Wiring tests for the witness trainer R-op dispatch (compute-facet #252).

Guards the ``--fused-r-kernel`` / ``--mx-compile`` / ``--profile-timing`` wiring in
``experiments/train_witness_realized_through_R_mlx.py``:

* DEFAULT (no flags) => ``_apply_R`` is BYTE-IDENTICAL to the pre-existing
  ``apply_contest_faithful_roundtrip_nhwc`` call (the running #205 --resume-from is
  unaffected).
* ``--fused-r-kernel`` => the fused Metal kernel, FORWARD bit-identical to the
  reference (max|Δ|=0) and VJP within the shared ~1 ULP GPU-reduction floor.
* ``--mx-compile`` => a FAIL-CLOSED bit-identity gate (mx.compile reintroduces
  fp-contraction that flips the uint8-STE d_seg argmax -> it must RAISE, never
  silently drift the score, unless the compile is genuinely bit-identical).
* ``r_isolated_microbench`` returns the timing keys the profile emit consumes.

NO-FAKE: these guard that the compute swap buys SPEED without changing the number
the loss/verdict sees. MLX/MPS is never a score authority.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO, REPO / "src", REPO / "experiments", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_base_trainer():
    name = "train_witness_realized_through_R_mlx"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _metal_available() -> bool:
    try:
        from tac.local_acceleration.metal_fused_r_operator import metal_fused_r_available

        return bool(metal_fused_r_available())
    except Exception:  # pragma: no cover - env guard
        return False


_needs_metal = pytest.mark.skipif(not _metal_available(), reason="requires a Metal (GPU) default device")


def _rgb(seed: int, rh: int = 24, rw: int = 32, n: int = 2):
    import mlx.core as mx

    rng = np.random.default_rng(seed)
    return mx.array((rng.random((n, rh, rw, 3)) * 255.0).astype(np.float32))


def test_default_off_apply_r_is_byte_identical_to_reference() -> None:
    """No flags => ``_apply_R`` == ``apply_contest_faithful_roundtrip_nhwc`` (byte-identical)."""

    import mlx.core as mx

    mx.set_default_device(mx.cpu)  # deterministic, CI-safe (no Metal needed)
    T = _load_base_trainer()
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc as REF,
    )

    T.set_fused_r_kernel(False)
    T.set_mx_compile_r(None)
    assert T.fused_r_kernel_enabled() is False
    x = _rgb(1)
    got = np.asarray(T._apply_R(x))
    ref = np.asarray(REF(x, output_hw=(T.SEG_H, T.SEG_W), ste_round=True))
    assert np.array_equal(got, ref), f"OFF path not byte-identical: max|Δ|={float(np.abs(got - ref).max())}"


def test_set_fused_r_kernel_toggles_global() -> None:
    T = _load_base_trainer()
    T.set_fused_r_kernel(True)
    assert T.fused_r_kernel_enabled() is True
    T.set_fused_r_kernel(False)
    assert T.fused_r_kernel_enabled() is False


def test_set_mx_compile_r_clear_restores_reference() -> None:
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    T = _load_base_trainer()
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc as REF,
    )

    # install a deliberately-wrong compiled fn, then clear it -> reference restored.
    T.set_fused_r_kernel(False)
    T.set_mx_compile_r(lambda z: z[..., : T.SEG_W, :][:, : T.SEG_H])  # nonsense shape-ish sentinel
    T.set_mx_compile_r(None)
    x = _rgb(2)
    got = np.asarray(T._apply_R(x))
    ref = np.asarray(REF(x, output_hw=(T.SEG_H, T.SEG_W), ste_round=True))
    assert np.array_equal(got, ref)


@_needs_metal
def test_fused_r_kernel_forward_bit_identical_to_reference() -> None:
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    T = _load_base_trainer()
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc as REF,
    )

    x = _rgb(3, rh=48, rw=64)
    ref = np.asarray(REF(x, output_hw=(T.SEG_H, T.SEG_W), ste_round=True))
    T.set_fused_r_kernel(True)
    try:
        got = np.asarray(T._apply_R(x))
    finally:
        T.set_fused_r_kernel(False)
    assert np.array_equal(got, ref), f"fused fwd not bit-identical: max|Δ|={float(np.abs(got - ref).max())}"


@_needs_metal
def test_fused_r_kernel_vjp_matches_reference_within_ulp_floor() -> None:
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    T = _load_base_trainer()
    from tac.local_acceleration.metal_fused_r_operator import CAMERA_HW, make_fused_r_roundtrip
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc as REF,
    )

    x = _rgb(4, rh=48, rw=64)
    y = np.asarray(REF(x, output_hw=(T.SEG_H, T.SEG_W), ste_round=True))
    cot = mx.array(np.random.default_rng(5).standard_normal(y.shape).astype(np.float32))
    _, (g_ref,) = mx.vjp(
        lambda z: REF(z, output_hw=(T.SEG_H, T.SEG_W), ste_round=True), (x,), (cot,)
    )
    fn = make_fused_r_roundtrip(camera_hw=CAMERA_HW, output_hw=(T.SEG_H, T.SEG_W), ste_round=True)
    _, (g_fused,) = mx.vjp(fn, (x,), (cot,))
    # both paths share the ~1 ULP GPU-reduction non-determinism floor.
    np.testing.assert_allclose(np.asarray(g_fused), np.asarray(g_ref), rtol=1e-4, atol=1e-4)


@_needs_metal
def test_mx_compile_gate_is_fail_closed_or_bit_identical() -> None:
    """The ``--mx-compile`` gate must EITHER raise (non-bit-identical) OR install a
    bit-identical compiled R -- never silently drift d_seg."""

    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    T = _load_base_trainer()
    try:
        res = T.maybe_enable_mx_compile_r(True, render_hw=(24, 32))
    except AssertionError as e:
        assert "fail-closed" in str(e).lower() or "bit-identical" in str(e).lower()
        return
    finally:
        T.set_mx_compile_r(None)
    # If it did NOT raise, it must have proven forward bit-identity.
    assert res.get("fwd_bit_identical") is True
    assert res.get("fwd_max_abs_delta") == 0.0


def test_maybe_enable_mx_compile_r_off_is_noop() -> None:
    T = _load_base_trainer()
    T.set_mx_compile_r(None)
    assert T.maybe_enable_mx_compile_r(False) == {}
    assert T._MX_COMPILE_R_FN is None


@_needs_metal
def test_r_isolated_microbench_returns_expected_keys() -> None:
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    T = _load_base_trainer()
    mb = T.r_isolated_microbench(render_h=24, render_w=32, n_frames=2, reps=3)
    for k in ("ref_fwd_ms_per_frame", "ref_fwdbwd_ms_per_frame", "render_hw", "n_frames"):
        assert k in mb, k
    assert mb["ref_fwd_ms_per_frame"] > 0.0
    # fused keys present when a Metal device is active.
    assert "fused_fwdbwd_speedup" in mb
    assert mb["fused_fwdbwd_speedup"] > 1.0
