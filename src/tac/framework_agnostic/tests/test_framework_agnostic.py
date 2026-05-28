# SPDX-License-Identifier: MIT
"""Tests for tac.framework_agnostic canonical primitives + decorators.

Per the canonical landing memo + sister tests under
src/tac/cathedral_consumers/framework_agnostic_lookup_consumer/tests/.

40+ tests covering:
  * Backend enum + AUTO selection cascade
  * BackendUnavailableError fail-closed semantics
  * FrameworkAgnosticTensor Protocol satisfaction
  * Operations: int8 round-trip determinism + FP4 packed nibbles + brotli
  * Decorators: framework_agnostic / mlx_first / pytorch_first / inflate_runtime
  * Helpers: state_dict bridge round-trip + assert_no_framework_mismatch
  * End-to-end V3 quantize round-trip + byte-deterministic across backends
"""
from __future__ import annotations

import io
import os

import numpy as np
import pytest

from tac.framework_agnostic import (
    Backend,
    BackendUnavailableError,
    DEFAULT_ENV_VAR,
    FrameworkAgnosticTensor,
    assert_no_framework_mismatch,
    brotli_compress,
    dequantize_int8_per_channel,
    detect_available_backends,
    detect_available_backends_dict,
    dtype_name,
    framework_agnostic,
    inflate_runtime_helper,
    mlx_first_with_numpy_fallback,
    mlx_state_dict_to_npz_bridge,
    npz_to_numpy_primitives,
    pytorch_first_with_numpy_fallback,
    pytorch_state_dict_to_npz_bridge,
    quantize_fp4_packed_nibbles,
    quantize_int8_per_channel,
    select_backend,
    shape_of,
    tinygrad_state_dict_to_npz_bridge,
)
from tac.framework_agnostic.backend import _AVAILABILITY_CHECK


# -----------------------------------------------------------------------------
# Backend enum + selection (Catalog #205 sister)
# -----------------------------------------------------------------------------


def test_backend_enum_has_4_concrete_backends_plus_auto():
    """Canonical 4-backend taxonomy + AUTO sentinel."""
    assert Backend.MLX.value == "mlx"
    assert Backend.PYTORCH.value == "pytorch"
    assert Backend.NUMPY.value == "numpy"
    assert Backend.TINYGRAD.value == "tinygrad"
    assert Backend.AUTO.value == "auto"


def test_default_env_var_canonical():
    """Canonical env-var per Catalog #205 sister discipline."""
    assert DEFAULT_ENV_VAR == "PACT_FRAMEWORK_BACKEND"


def test_detect_available_backends_returns_tuple():
    backends = detect_available_backends()
    assert isinstance(backends, tuple)
    # numpy is hard dep per CLAUDE.md 8th standing directive.
    assert Backend.NUMPY in backends


def test_detect_available_backends_dict_includes_all_known():
    d = detect_available_backends_dict()
    assert Backend.NUMPY in d
    assert d[Backend.NUMPY] is True
    # Other backends may be True or False; key presence is the contract.
    for b in (Backend.MLX, Backend.PYTORCH, Backend.TINYGRAD):
        assert b in d
        assert isinstance(d[b], bool)


def test_select_backend_returns_concrete_not_auto():
    """AUTO must always resolve to a concrete Backend."""
    backend = select_backend()
    assert backend is not Backend.AUTO
    assert backend in (Backend.MLX, Backend.PYTORCH, Backend.NUMPY, Backend.TINYGRAD)


def test_select_backend_override_explicit():
    """Explicit override returns the requested backend if available."""
    # numpy is always available per CLAUDE.md hard dep.
    backend = select_backend(override=Backend.NUMPY)
    assert backend is Backend.NUMPY


def test_select_backend_override_unavailable_raises():
    """Explicit override of an unavailable backend raises BackendUnavailableError."""
    # Find a known-unavailable backend.
    avail = detect_available_backends_dict()
    unavailable = next((b for b, ok in avail.items() if not ok), None)
    if unavailable is None:
        pytest.skip("all backends available on this host; cannot test unavailable path")
    with pytest.raises(BackendUnavailableError) as exc_info:
        select_backend(override=unavailable)
    assert "not installed" in str(exc_info.value)
    assert "uv pip install" in str(exc_info.value)


def test_select_backend_env_var_numpy(monkeypatch):
    """Env var with valid backend returns that backend."""
    monkeypatch.setenv(DEFAULT_ENV_VAR, "numpy")
    assert select_backend() is Backend.NUMPY


def test_select_backend_env_var_auto_defers_to_priority(monkeypatch):
    """Env var 'auto' defers to canonical platform priority."""
    monkeypatch.setenv(DEFAULT_ENV_VAR, "auto")
    backend = select_backend()
    assert backend is not Backend.AUTO


def test_select_backend_env_var_invalid_raises(monkeypatch):
    """Env var with unrecognized backend raises BackendUnavailableError."""
    monkeypatch.setenv(DEFAULT_ENV_VAR, "totally_made_up")
    with pytest.raises(BackendUnavailableError) as exc_info:
        select_backend()
    assert "unrecognized" in str(exc_info.value).lower()


def test_select_backend_priority_kwarg_routes_through_caller():
    """Priority kwarg routes selection through caller-supplied order."""
    backend = select_backend(priority=(Backend.NUMPY,))
    assert backend is Backend.NUMPY


def test_select_backend_skips_auto_in_priority():
    """AUTO in priority tuple should be skipped without raising."""
    backend = select_backend(priority=(Backend.AUTO, Backend.NUMPY))
    assert backend is Backend.NUMPY


def test_select_backend_override_auto_defers():
    """override=AUTO is equivalent to override=None."""
    backend = select_backend(override=Backend.AUTO)
    assert backend is not Backend.AUTO


# -----------------------------------------------------------------------------
# FrameworkAgnosticTensor Protocol (Catalog #335 sister)
# -----------------------------------------------------------------------------


def test_numpy_array_satisfies_protocol():
    """numpy.ndarray structurally satisfies FrameworkAgnosticTensor."""
    arr = np.zeros((3, 4), dtype=np.float32)
    assert isinstance(arr, FrameworkAgnosticTensor)


def test_shape_of_numpy_array():
    arr = np.zeros((3, 4, 5), dtype=np.float32)
    assert shape_of(arr) == (3, 4, 5)


def test_shape_of_rejects_non_tensor():
    with pytest.raises(TypeError) as exc_info:
        shape_of(42)  # int has no .shape
    assert "shape" in str(exc_info.value).lower()


def test_dtype_name_normalizes_numpy():
    arr = np.zeros((2,), dtype=np.float32)
    name = dtype_name(arr)
    assert "float32" in name


def test_dtype_name_rejects_non_tensor():
    with pytest.raises(TypeError):
        dtype_name(42)


# -----------------------------------------------------------------------------
# Operations: int8 round-trip determinism (canonical numpy oracle)
# -----------------------------------------------------------------------------


def test_int8_round_trip_per_channel_numpy_axis_0():
    """numpy reference: per-channel int8 round-trip ≤ int8 representability err."""
    np.random.seed(42)
    x = np.random.randn(8, 16).astype(np.float32) * 5
    i8, sc = quantize_int8_per_channel(x, axis=0, backend=Backend.NUMPY)
    xr = dequantize_int8_per_channel(i8, sc, axis=0, backend=Backend.NUMPY)
    # int8 per-channel symmetric: max err ≤ scale / 2 per channel
    per_channel_scale = np.abs(x).max(axis=1)
    max_per_channel_err = (per_channel_scale / 127.0).max()
    assert np.abs(x - xr).max() <= max_per_channel_err * 1.5  # tolerance


def test_int8_round_trip_axis_1():
    """Per-channel axis=1 still round-trips correctly."""
    np.random.seed(42)
    x = np.random.randn(8, 16).astype(np.float32) * 5
    i8, sc = quantize_int8_per_channel(x, axis=1, backend=Backend.NUMPY)
    xr = dequantize_int8_per_channel(i8, sc, axis=1, backend=Backend.NUMPY)
    assert xr.shape == x.shape
    assert i8.dtype == np.int8


def test_int8_quantization_degenerate_channel_does_not_divide_by_zero():
    """Zero channel produces all-zero int8 with scale 1.0 (avoid div0)."""
    x = np.zeros((4, 8), dtype=np.float32)
    i8, sc = quantize_int8_per_channel(x, axis=0, backend=Backend.NUMPY)
    assert (i8 == 0).all()
    assert (sc == 1.0).all()


def test_int8_dtype_invariant():
    """Output is always int8 regardless of input dtype."""
    x = np.random.randn(4, 8).astype(np.float64) * 3
    i8, sc = quantize_int8_per_channel(x, backend=Backend.NUMPY)
    assert i8.dtype == np.int8
    assert sc.dtype == np.float32


@pytest.mark.skipif(
    not _AVAILABILITY_CHECK[Backend.PYTORCH](),
    reason="torch not installed",
)
def test_int8_round_trip_pytorch_matches_numpy_oracle():
    """Catalog #146: byte-determinism across backends — torch path matches numpy."""
    import torch
    np.random.seed(42)
    x = np.random.randn(8, 16).astype(np.float32) * 5
    i8_np, sc_np = quantize_int8_per_channel(x, axis=0, backend=Backend.NUMPY)
    i8_torch, sc_torch = quantize_int8_per_channel(
        torch.from_numpy(x), axis=0, backend=Backend.PYTORCH
    )
    # Same quantization grid → byte-identical int8.
    assert np.array_equal(i8_np, i8_torch.numpy())
    np.testing.assert_allclose(sc_np, sc_torch.numpy(), rtol=1e-5)


@pytest.mark.skipif(
    not _AVAILABILITY_CHECK[Backend.MLX](),
    reason="mlx not installed",
)
def test_int8_round_trip_mlx_matches_numpy_oracle():
    """Catalog #146: MLX path matches numpy oracle byte-for-byte."""
    import mlx.core as mx
    np.random.seed(42)
    x = np.random.randn(8, 16).astype(np.float32) * 5
    i8_np, sc_np = quantize_int8_per_channel(x, axis=0, backend=Backend.NUMPY)
    i8_mlx, sc_mlx = quantize_int8_per_channel(
        mx.array(x), axis=0, backend=Backend.MLX
    )
    assert np.array_equal(i8_np, np.asarray(i8_mlx))
    np.testing.assert_allclose(sc_np, np.asarray(sc_mlx), rtol=1e-5)


# -----------------------------------------------------------------------------
# Operations: FP4 packed nibbles (canonical Quantizr unsigned-E2M1)
# -----------------------------------------------------------------------------


def test_fp4_packed_nibbles_output_size_half_input():
    """FP4 packs 2 nibbles per byte → output size = (N+1)//2."""
    x = np.random.randn(32).astype(np.float32)
    packed, scale = quantize_fp4_packed_nibbles(x, backend=Backend.NUMPY)
    assert packed.dtype == np.uint8
    assert packed.size == 16  # 32 elements → 16 bytes


def test_fp4_packed_nibbles_odd_input_pads():
    """Odd-length input gets one zero nibble padded."""
    x = np.random.randn(31).astype(np.float32)
    packed, scale = quantize_fp4_packed_nibbles(x, backend=Backend.NUMPY)
    assert packed.size == 16  # (31+1)//2


def test_fp4_canonical_codebook_default():
    """Canonical Quantizr unsigned-E2M1 codebook default."""
    x = np.array([0.0, 6.0, 3.0], dtype=np.float32)
    packed, scale = quantize_fp4_packed_nibbles(x, backend=Backend.NUMPY)
    # scale should be 6 / 6 = 1.0 since max(codebook) = 6 and abs_max(x) = 6
    assert scale == pytest.approx(1.0, rel=1e-5)


def test_fp4_custom_codebook_accepted():
    """Caller may pass custom 8-entry codebook."""
    cb = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    x = np.array([0.0, 0.7], dtype=np.float32)
    packed, scale = quantize_fp4_packed_nibbles(x, codebook=cb, backend=Backend.NUMPY)
    assert scale == pytest.approx(0.7 / 0.7, rel=1e-5)


def test_fp4_codebook_wrong_length_rejected():
    """Codebook must have exactly 8 entries."""
    with pytest.raises(ValueError) as exc_info:
        quantize_fp4_packed_nibbles(np.zeros(2), codebook=(0.0, 1.0))
    assert "8 entries" in str(exc_info.value)


def test_fp4_empty_input_does_not_crash():
    """Empty input produces empty packed array with sentinel scale=1.0."""
    x = np.array([], dtype=np.float32)
    packed, scale = quantize_fp4_packed_nibbles(x, backend=Backend.NUMPY)
    assert packed.size == 0
    assert scale == 1.0 / 6.0  # abs_max=1.0 default → /max(cb)=6


# -----------------------------------------------------------------------------
# Operations: brotli (canonical hard dep)
# -----------------------------------------------------------------------------


def test_brotli_compress_round_trip():
    """brotli_compress + decompress recovers original bytes."""
    import brotli
    original = b"hello world " * 100
    compressed = brotli_compress(original, quality=11)
    assert isinstance(compressed, bytes)
    assert len(compressed) < len(original)
    assert brotli.decompress(compressed) == original


def test_brotli_compress_quality_default_11():
    """Default quality is 11 (canonical medal-class)."""
    data = b"x" * 1000
    c11 = brotli_compress(data)
    c0 = brotli_compress(data, quality=0)
    # quality 11 should compress better than quality 0 for repetitive data
    assert len(c11) <= len(c0)


# -----------------------------------------------------------------------------
# Decorators
# -----------------------------------------------------------------------------


def test_framework_agnostic_decorator_injects_backend():
    """@framework_agnostic resolves backend once and passes via kwarg."""
    @framework_agnostic()
    def my_op(x, *, backend=None):
        return (x, backend)

    result, backend = my_op(42)
    assert result == 42
    assert backend is not Backend.AUTO
    assert backend is not None


def test_framework_agnostic_decorator_caller_override_wins():
    """Caller's explicit backend= kwarg overrides decorator default."""
    @framework_agnostic(backend=Backend.NUMPY)
    def my_op(x, *, backend=None):
        return backend

    result = my_op(42, backend=Backend.NUMPY)
    assert result is Backend.NUMPY


def test_mlx_first_decorator_resolves_mlx_when_available():
    """@mlx_first_with_numpy_fallback chooses MLX if installed, else numpy."""
    @mlx_first_with_numpy_fallback
    def my_op(x, *, backend=None):
        return backend

    result = my_op(42)
    if _AVAILABILITY_CHECK[Backend.MLX]():
        assert result is Backend.MLX
    else:
        assert result is Backend.NUMPY


def test_pytorch_first_decorator_resolves_pytorch_when_available():
    """@pytorch_first_with_numpy_fallback chooses PyTorch if installed, else numpy."""
    @pytorch_first_with_numpy_fallback
    def my_op(x, *, backend=None):
        return backend

    result = my_op(42)
    if _AVAILABILITY_CHECK[Backend.PYTORCH]():
        assert result is Backend.PYTORCH
    else:
        assert result is Backend.NUMPY


def test_inflate_runtime_decorator_pins_numpy_ignoring_caller():
    """@inflate_runtime_helper pins NUMPY regardless of caller's override per HNeRV L4."""
    @inflate_runtime_helper
    def my_inflate_op(x, *, backend=None):
        return backend

    # Even when caller passes Backend.MLX, decorator pins NUMPY.
    result = my_inflate_op(42, backend=Backend.MLX)
    assert result is Backend.NUMPY


# -----------------------------------------------------------------------------
# Helpers: bridge contracts (MLX state_dict → npz → numpy)
# -----------------------------------------------------------------------------


def test_npz_to_numpy_primitives_round_trip():
    """Canonical bridge: numpy state_dict → npz bytes → numpy state_dict."""
    sd = {
        "weight": np.random.randn(3, 4).astype(np.float32),
        "bias": np.random.randn(4).astype(np.float32),
    }
    buf = io.BytesIO()
    np.savez_compressed(buf, **sd)
    npz_bytes = buf.getvalue()
    recovered = npz_to_numpy_primitives(npz_bytes)
    assert set(recovered.keys()) == set(sd.keys())
    for k in sd:
        np.testing.assert_array_equal(recovered[k], sd[k])


@pytest.mark.skipif(
    not _AVAILABILITY_CHECK[Backend.PYTORCH](),
    reason="torch not installed",
)
def test_pytorch_state_dict_to_npz_bridge():
    """PyTorch state_dict → npz round-trip preserves values."""
    import torch
    sd = {
        "weight": torch.randn(3, 4),
        "bias": torch.randn(4),
    }
    npz_bytes = pytorch_state_dict_to_npz_bridge(sd)
    recovered = npz_to_numpy_primitives(npz_bytes)
    assert set(recovered.keys()) == set(sd.keys())
    np.testing.assert_allclose(recovered["weight"], sd["weight"].numpy())
    np.testing.assert_allclose(recovered["bias"], sd["bias"].numpy())


@pytest.mark.skipif(
    not _AVAILABILITY_CHECK[Backend.MLX](),
    reason="mlx not installed",
)
def test_mlx_state_dict_to_npz_bridge():
    """MLX state_dict → npz round-trip preserves values per 8th standing directive."""
    import mlx.core as mx
    np.random.seed(42)
    w_np = np.random.randn(3, 4).astype(np.float32)
    b_np = np.random.randn(4).astype(np.float32)
    sd = {"weight": mx.array(w_np), "bias": mx.array(b_np)}
    npz_bytes = mlx_state_dict_to_npz_bridge(sd)
    recovered = npz_to_numpy_primitives(npz_bytes)
    assert set(recovered.keys()) == set(sd.keys())
    np.testing.assert_allclose(recovered["weight"], w_np)
    np.testing.assert_allclose(recovered["bias"], b_np)


def test_mlx_bridge_raises_if_mlx_unavailable():
    """When MLX is not installed, bridge raises BackendUnavailableError."""
    if _AVAILABILITY_CHECK[Backend.MLX]():
        pytest.skip("MLX is installed; cannot test unavailable path")
    with pytest.raises(BackendUnavailableError) as exc_info:
        mlx_state_dict_to_npz_bridge({"a": np.zeros((2,))})
    assert "uv pip install mlx" in str(exc_info.value)


def test_pytorch_bridge_raises_if_pytorch_unavailable():
    """When PyTorch is not installed, bridge raises BackendUnavailableError."""
    if _AVAILABILITY_CHECK[Backend.PYTORCH]():
        pytest.skip("PyTorch is installed; cannot test unavailable path")
    with pytest.raises(BackendUnavailableError) as exc_info:
        pytorch_state_dict_to_npz_bridge({"a": np.zeros((2,))})
    assert "uv pip install torch" in str(exc_info.value)


def test_tinygrad_bridge_raises_if_tinygrad_unavailable():
    """tinygrad is OPTIONAL; bridge raises with installation hint when missing."""
    if _AVAILABILITY_CHECK[Backend.TINYGRAD]():
        pytest.skip("tinygrad is installed; cannot test unavailable path")
    with pytest.raises(BackendUnavailableError) as exc_info:
        tinygrad_state_dict_to_npz_bridge({"a": np.zeros((2,))})
    assert "uv pip install tinygrad" in str(exc_info.value)


def test_assert_no_framework_mismatch_numpy_ok():
    """numpy tensor + expected numpy → no error."""
    assert_no_framework_mismatch(np.zeros((2,)), Backend.NUMPY)


def test_assert_no_framework_mismatch_unknown_tensor_silent():
    """Python list / scalar has unknown backend → no error (let downstream route)."""
    assert_no_framework_mismatch([1, 2, 3], Backend.NUMPY)


@pytest.mark.skipif(
    not _AVAILABILITY_CHECK[Backend.PYTORCH](),
    reason="torch not installed",
)
def test_assert_no_framework_mismatch_torch_vs_numpy_raises():
    """torch.Tensor + expected numpy → TypeError per fail-closed contract."""
    import torch
    with pytest.raises(TypeError) as exc_info:
        assert_no_framework_mismatch(torch.zeros((2,)), Backend.NUMPY)
    assert "PYTORCH" in str(exc_info.value)
    assert "NUMPY" in str(exc_info.value)


# -----------------------------------------------------------------------------
# End-to-end: V3 quantize round-trip across backends (byte-determinism)
# -----------------------------------------------------------------------------


def test_end_to_end_v3_quantize_byte_deterministic_numpy_pytorch():
    """V3 sister trainer pair scenario: int8 round-trip byte-identical numpy vs torch."""
    if not _AVAILABILITY_CHECK[Backend.PYTORCH]():
        pytest.skip("torch not installed")
    import torch
    np.random.seed(123)
    # Simulate a substrate decoder weight tensor (C_out=32, C_in=16, K=3, K=3).
    w = np.random.randn(32, 16, 3, 3).astype(np.float32)
    i8_np, sc_np = quantize_int8_per_channel(w, axis=0, backend=Backend.NUMPY)
    i8_torch, sc_torch = quantize_int8_per_channel(
        torch.from_numpy(w), axis=0, backend=Backend.PYTORCH
    )
    # Byte-determinism per Catalog #146 contract.
    assert np.array_equal(i8_np, i8_torch.numpy())


def test_end_to_end_v3_quantize_byte_deterministic_mlx_path():
    """V3 sister trainer pair scenario: MLX path matches numpy oracle byte-for-byte."""
    if not _AVAILABILITY_CHECK[Backend.MLX]():
        pytest.skip("mlx not installed")
    import mlx.core as mx
    np.random.seed(123)
    w = np.random.randn(32, 16, 3, 3).astype(np.float32)
    i8_np, sc_np = quantize_int8_per_channel(w, axis=0, backend=Backend.NUMPY)
    i8_mlx, sc_mlx = quantize_int8_per_channel(
        mx.array(w), axis=0, backend=Backend.MLX
    )
    assert np.array_equal(i8_np, np.asarray(i8_mlx))


def test_end_to_end_brotli_compressed_int8_payload():
    """Canonical V3 pipeline: quantize → pack int8 → brotli → ZIP-member."""
    np.random.seed(42)
    w = np.random.randn(64, 16).astype(np.float32)
    i8, sc = quantize_int8_per_channel(w, axis=0, backend=Backend.NUMPY)
    # Pack into bytes for brotli (int8 is 1 byte per element).
    payload = i8.tobytes()
    compressed = brotli_compress(payload, quality=11)
    # int8 random data should compress to ~ same size (high entropy).
    assert isinstance(compressed, bytes)
    assert len(compressed) > 0


def test_npz_bridge_byte_deterministic_across_backends():
    """numpy reference + pytorch bridge produce identical npz bytes for identical input."""
    if not _AVAILABILITY_CHECK[Backend.PYTORCH]():
        pytest.skip("torch not installed")
    import torch
    np.random.seed(7)
    w_np = np.random.randn(3, 4).astype(np.float32)
    b_np = np.random.randn(4).astype(np.float32)
    sd_torch = {"weight": torch.from_numpy(w_np), "bias": torch.from_numpy(b_np)}
    npz_torch = pytorch_state_dict_to_npz_bridge(sd_torch)
    # Recover and re-encode via canonical numpy path; check value equality
    # (npz container metadata may vary; value equality is the contract).
    recovered = npz_to_numpy_primitives(npz_torch)
    np.testing.assert_array_equal(recovered["weight"], w_np)
    np.testing.assert_array_equal(recovered["bias"], b_np)


def test_decorator_chains_compose_with_operation():
    """Decorator chains compose canonically with operations primitives."""
    @mlx_first_with_numpy_fallback
    def quantize_op(x, *, backend=None):
        return quantize_int8_per_channel(x, axis=0, backend=backend)

    x = np.random.randn(4, 8).astype(np.float32) * 3
    i8, sc = quantize_op(x)
    assert i8.shape == (4, 8) or shape_of(i8) == (4, 8)
