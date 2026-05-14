# SPDX-License-Identifier: MIT
"""Unit tests for the Mallat wavelet-coefficient importance proxy.

Sister of ``test_pr101_sensitivity_aware_quantization_unit.py``. Verifies that
the new proxy:
  1. respects the existing :class:`TensorImportance` schema,
  2. returns 0 for DC/zero tensors (no detail content) and positive for
     high-frequency content,
  3. handles 1D/2D/3D/4D reshapes,
  4. is sign-flip invariant (squared magnitudes),
  5. scales linearly with amplitude at fixed pattern,
  6. raises on empty blobs,
  7. handles single-element / smaller-than-filter-length blobs gracefully,
  8. is NOT identical to Xavier-L2 (diagnostic — proves we built a different
     proxy, not a renaming).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from pr101_sensitivity_aware_mallat_wavelet import (  # noqa: E402
    SENSITIVITY_PROXY,
    WAVELET_LEVEL,
    WAVELET_NAME,
    _mallat_detail_energy,
    _reshape_to_2d,
    compute_mallat_wavelet_importance,
)
from pr101_sensitivity_aware_quantization import (  # noqa: E402
    TensorImportance,
    compute_xavier_l2_importance,
)


def _make_blob(name: str, values: np.ndarray):
    class _B:
        pass

    b = _B()
    b.name = name
    b.raw = values.astype(np.int32)
    return b


def test_mallat_zero_tensor_yields_zero_importance() -> None:
    blob = _make_blob("zero", np.zeros(64, dtype=np.int32))
    out = compute_mallat_wavelet_importance([blob])
    assert out[0].importance == 0.0
    assert out[0].rms == 0.0
    assert out[0].numel == 64


def test_mallat_pure_dc_constant_tensor_has_near_zero_detail_energy() -> None:
    # All-ones tensor → only approximation coefficients carry energy; details ~0.
    blob = _make_blob("dc", np.ones(64, dtype=np.int32) * 5)
    out = compute_mallat_wavelet_importance([blob])
    # Periodization mode + db4 on a constant vector has tiny boundary leakage but
    # detail energy should be many orders of magnitude below any non-trivial
    # high-frequency tensor. We assert the absolute value is small relative to
    # the total signal energy.
    sum_sq = float((blob.raw.astype(np.float64) ** 2).sum())
    assert out[0].importance * out[0].numel < 1e-6 * sum_sq


def test_mallat_high_frequency_pattern_higher_than_dc() -> None:
    n = 64
    dc_blob = _make_blob("dc", np.ones(n, dtype=np.int32) * 5)
    # Checkerboard sign pattern — alternating ±5 → maximum high-frequency content.
    sign = np.array([1 if (i % 2 == 0) else -1 for i in range(n)], dtype=np.int32)
    hf_blob = _make_blob("hf", sign * 5)
    out = compute_mallat_wavelet_importance([dc_blob, hf_blob])
    assert out[1].importance > out[0].importance
    # On checkerboard, detail energy dominates → importance should be a
    # meaningful fraction of mean-square magnitude (which is 25.0).
    assert out[1].importance > 1.0


def test_mallat_handles_1d_2d_3d_4d_shapes() -> None:
    blobs = [
        _make_blob("1d", np.arange(64, dtype=np.int32)),
        _make_blob("2d", np.arange(64, dtype=np.int32).reshape(8, 8)),
        _make_blob("3d", np.arange(64, dtype=np.int32).reshape(4, 4, 4)),
        _make_blob("4d", np.arange(64, dtype=np.int32).reshape(2, 4, 4, 2)),
    ]
    out = compute_mallat_wavelet_importance(blobs)
    assert len(out) == 4
    for ti in out:
        assert ti.numel == 64
        assert isinstance(ti.importance, float)
        assert ti.importance >= 0.0


def test_mallat_is_sign_flip_invariant() -> None:
    rng = np.random.default_rng(seed=42)
    pattern = rng.integers(-50, 50, size=128, dtype=np.int32)
    pos = _make_blob("pos", pattern)
    neg = _make_blob("neg", -pattern)
    out = compute_mallat_wavelet_importance([pos, neg])
    assert out[0].importance == pytest.approx(out[1].importance, rel=1e-9, abs=1e-9)


def test_mallat_scales_linearly_with_amplitude() -> None:
    # Same pattern × 2 → detail energy × 4 → per-element importance × 4 (since
    # numel is identical). Use a checkerboard pattern so detail energy is
    # non-trivial.
    n = 128
    pat = np.array([1 if (i % 2 == 0) else -1 for i in range(n)], dtype=np.int32)
    a = _make_blob("a", pat * 5)
    b = _make_blob("b", pat * 10)
    out = compute_mallat_wavelet_importance([a, b])
    assert out[0].importance > 0.0
    ratio = out[1].importance / out[0].importance
    # 10/5 = 2× amplitude → 4× detail energy → 4× per-element importance.
    assert ratio == pytest.approx(4.0, rel=1e-9, abs=1e-9)


def test_mallat_raises_on_empty_blob() -> None:
    blob = _make_blob("empty", np.zeros(0, dtype=np.int32))
    with pytest.raises(ValueError):
        compute_mallat_wavelet_importance([blob])


def test_mallat_handles_smaller_than_filter_length_gracefully() -> None:
    # db4 has filter length 8; a 3-element tensor (rgb_*.bias in PR101) cannot
    # produce a meaningful 2-level decomposition. The proxy should return 0.0
    # rather than raise.
    blob_3 = _make_blob("rgb_bias", np.array([1, -1, 1], dtype=np.int32))
    out = compute_mallat_wavelet_importance([blob_3])
    assert out[0].importance == 0.0
    assert out[0].numel == 3
    # Single-element edge case
    blob_1 = _make_blob("singleton", np.array([7], dtype=np.int32))
    out_1 = compute_mallat_wavelet_importance([blob_1])
    assert out_1[0].importance == 0.0
    assert out_1[0].numel == 1


def test_mallat_differs_from_xavier_l2_on_known_pattern() -> None:
    # Construct two tensors with EQUAL RMS but very different frequency content.
    # Tensor A: smooth ramp (all-positive, low-frequency dominant).
    # Tensor B: sign-flipped ramp (high-frequency between adjacent samples).
    n = 64
    smooth = np.arange(1, n + 1, dtype=np.int32)
    flipped = smooth * np.where(np.arange(n) % 2 == 0, 1, -1)
    a = _make_blob("smooth", smooth)
    b = _make_blob("flipped", flipped.astype(np.int32))

    xavier = compute_xavier_l2_importance([a, b])
    mallat = compute_mallat_wavelet_importance([a, b])

    # Xavier-L2 measures magnitude only — both tensors have IDENTICAL RMS, so
    # Xavier importances should match.
    assert xavier[0].importance == pytest.approx(xavier[1].importance, rel=1e-9, abs=1e-9)
    # Mallat must distinguish them: the flipped one carries more detail energy.
    assert mallat[1].importance > mallat[0].importance


def test_mallat_returns_full_schema_match() -> None:
    blob = _make_blob("schema", np.arange(64, dtype=np.int32))
    out = compute_mallat_wavelet_importance([blob])
    ti = out[0]
    assert isinstance(ti, TensorImportance)
    assert ti.name == "schema"
    assert isinstance(ti.importance, float)
    assert isinstance(ti.numel, int)
    assert isinstance(ti.rms, float)
    assert ti.numel == 64
    assert ti.importance >= 0.0


def test_module_constants_are_documented() -> None:
    assert SENSITIVITY_PROXY == "mallat_wavelet"
    assert WAVELET_NAME == "db4"
    assert WAVELET_LEVEL == 2


def test_reshape_to_2d_handles_all_dims() -> None:
    # 0-d
    out = _reshape_to_2d(np.asarray(5, dtype=np.int32))
    assert out.shape == (1, 1)
    # 1-d → near-square
    out = _reshape_to_2d(np.arange(10, dtype=np.int32))
    assert out.shape[0] * out.shape[1] >= 10
    # 2-d unchanged
    out = _reshape_to_2d(np.zeros((3, 5), dtype=np.int32))
    assert out.shape == (3, 5)
    # 4-d → flatten leading dims
    out = _reshape_to_2d(np.zeros((2, 3, 4, 5), dtype=np.int32))
    assert out.shape == (2 * 3 * 4, 5)


def test_mallat_detail_energy_zero_matrix_is_zero() -> None:
    assert _mallat_detail_energy(np.zeros((16, 16), dtype=np.float64)) == 0.0
