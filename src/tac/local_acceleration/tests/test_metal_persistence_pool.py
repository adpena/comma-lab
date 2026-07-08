# SPDX-License-Identifier: MIT
"""Tests for the fused 3x3 morphological pool Metal kernel (#212 compute suite).

Bit-identity to the numpy authority (``_pool3x3_np``), determinism, env-flag
gating, dispatch conditions, and CPU/pure-MLX fallback.
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tac.boundary_math import persistence_topology_loss as P  # noqa: E402
from tac.local_acceleration import metal_persistence_pool as K  # noqa: E402

_GPU = K.metal_persistence_pool_available()
_REQUIRES_GPU = pytest.mark.skipif(not _GPU, reason="requires MLX GPU (Metal)")

_REAL_SHAPES = [(4, 384, 512), (2, 384, 512), (1, 384, 512), (8, 384, 512)]


def _rand(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


# --- bit-identity to the numpy authority ---------------------------------
@_REQUIRES_GPU
@pytest.mark.parametrize("shape", _REAL_SHAPES)
@pytest.mark.parametrize("kind,red", [("min", np.min), ("max", np.max), ("mean", np.mean)])
def test_bit_identical_to_numpy_reference(shape, kind, red):
    x = _rand(shape, seed=hash((shape, kind)) & 0xFFFF)
    ref = P._pool3x3_np(x, red)
    ker = np.array(K.pool3x3_metal(mx.array(x), kind))
    assert np.max(np.abs(ker - ref)) == 0.0, f"{shape} {kind} not bit-identical"


@_REQUIRES_GPU
def test_bit_identical_2d_input():
    x = _rand((384, 512), seed=7)
    for kind, red in [("min", np.min), ("max", np.max), ("mean", np.mean)]:
        ker = np.array(K.pool3x3_metal(mx.array(x), kind))
        assert np.max(np.abs(ker - P._pool3x3_np(x, red))) == 0.0


@_REQUIRES_GPU
def test_bit_identical_extra_leading_dims():
    # (N, K, H, W) leading dims flatten to M internally.
    x = _rand((2, 3, 32, 40), seed=11)
    ker = np.array(K.pool3x3_metal(mx.array(x), "max"))
    assert ker.shape == x.shape
    assert np.max(np.abs(ker - P._pool3x3_np(x, np.max))) == 0.0


@_REQUIRES_GPU
def test_edge_clamp_matches_reference_at_borders():
    # A ramp makes border (edge-clamp) handling load-bearing.
    x = np.arange(1 * 5 * 5, dtype=np.float32).reshape(1, 5, 5)
    for kind, red in [("min", np.min), ("max", np.max), ("mean", np.mean)]:
        ker = np.array(K.pool3x3_metal(mx.array(x), kind))
        assert np.array_equal(ker, P._pool3x3_np(x, red))


# --- determinism (in-process; cross-process covered by the probe tool) ---
@_REQUIRES_GPU
def test_in_process_repeatable():
    x = mx.array(_rand((4, 128, 128), seed=3))
    for kind in ("min", "max", "mean"):
        a = np.array(K.pool3x3_metal(x, kind))
        b = np.array(K.pool3x3_metal(x, kind))
        assert np.array_equal(a, b)


# --- env-flag gating / enabled predicate ---------------------------------
def test_enabled_false_when_flag_unset(monkeypatch):
    monkeypatch.delenv(K.PERSISTENCE_POOL_METAL_KERNEL_FLAG, raising=False)
    assert K.persistence_pool_metal_enabled() is False


def test_enabled_false_for_falsey_values(monkeypatch):
    for v in ("0", "false", "no", "off", "", "  "):
        monkeypatch.setenv(K.PERSISTENCE_POOL_METAL_KERNEL_FLAG, v)
        assert K.persistence_pool_metal_enabled() is False


@_REQUIRES_GPU
def test_enabled_true_when_flag_set_and_gpu(monkeypatch):
    for v in ("1", "true", "YES", "On"):
        monkeypatch.setenv(K.PERSISTENCE_POOL_METAL_KERNEL_FLAG, v)
        assert K.persistence_pool_metal_enabled() is True


# --- dispatch: density smoothing routes through the kernel when enabled ---
@_REQUIRES_GPU
def test_smooth_density_metal_dispatch_bit_identical(monkeypatch):
    x = mx.array(np.abs(_rand((3, 96, 96), seed=21)).astype(np.float32))
    monkeypatch.delenv(K.PERSISTENCE_POOL_METAL_KERNEL_FLAG, raising=False)
    base = np.array(P._smooth_density_mlx(x, 4))  # pure-MLX path
    monkeypatch.setenv(K.PERSISTENCE_POOL_METAL_KERNEL_FLAG, "1")
    metal = np.array(P._smooth_density_mlx(x, 4))  # kernel path
    # both must equal the numpy authority (bit-identical mean pool composition)
    xn = np.array(x)
    ref = xn
    for _ in range(4):
        ref = P._pool3x3_np(ref, np.mean)
    assert np.max(np.abs(metal - ref)) == 0.0
    assert np.max(np.abs(base - ref)) <= 1e-6  # pure-MLX may drift; kernel is exact


def test_smooth_density_fallback_when_flag_off(monkeypatch):
    # Flag off ⇒ byte-identical to the historical pure-MLX path (no kernel import needed).
    monkeypatch.delenv(K.PERSISTENCE_POOL_METAL_KERNEL_FLAG, raising=False)
    x = mx.array(np.abs(_rand((2, 48, 48), seed=31)).astype(np.float32))
    out = np.array(P._smooth_density_mlx(x, 3))
    ref = np.array(x)
    for _ in range(3):
        ref = P._pool3x3_np(ref, np.mean)
    assert np.max(np.abs(out - ref)) <= 1e-6


# --- input validation -----------------------------------------------------
@_REQUIRES_GPU
def test_bad_kind_raises():
    with pytest.raises(ValueError):
        K.pool3x3_metal(mx.array(_rand((1, 8, 8))), "median")


@_REQUIRES_GPU
def test_bad_ndim_raises():
    with pytest.raises(ValueError):
        K.pool3x3_metal(mx.array(_rand((8,))), "max")


# --- signature reflects the built kernel ----------------------------------
def test_signature_marks_built():
    sig = P.metal_pool_kernel_signature()
    assert sig["built"] is True
    assert sig["module"] == "tac.local_acceleration.metal_persistence_pool"
    assert sig["env_flag"] == K.PERSISTENCE_POOL_METAL_KERNEL_FLAG
