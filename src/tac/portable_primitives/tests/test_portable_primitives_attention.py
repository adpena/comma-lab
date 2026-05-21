# SPDX-License-Identifier: MIT
"""Cross-backend numerical-equivalence tests for MLX-ARCH-2 attention primitives.

Sister of ``test_portable_primitives_extended.py`` for the 4 attention
primitives required by FastViT-T12 PoseNet backbone:

- :class:`PortableLayerScale`
- :class:`PortableMHSA`
- :class:`PortableTokenMixer`
- :class:`PortableRepMixer`

Per OVERNIGHT-WW + Phase 1 PV finding + ARCH-1 empirical confirmation:
every primitive constructed with ``backend="mlx"`` must produce
numerically-equivalent results to its ``backend="pytorch"`` sister
within a documented ε band.

**Empirical ε band per Phase 1 PV / ARCH-1** (M5 Max Metal GPU vs PyTorch
CPU fp32):
- ATOL_FP32 = 5e-3 (Metal MPS uses different FMA ordering than CPU fp32)
- RTOL_FP32 = 5e-3 (relative differences scale with magnitude per FMA
  re-ordering)

Per CLAUDE.md "Apples-to-apples evidence discipline" extended to the
portable-primitives surface: a future refactor that silently breaks
backend equivalence beyond ε would corrupt the train-anywhere
eval-anywhere pipeline (the MLX-train -> PyTorch-eval canonical
contest-axis path).

Tests are skipped if MLX is not available on the host (e.g. Linux CI).
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.portable_primitives import (
    is_mlx_available,
    is_pytorch_available,
)
from tac.portable_primitives.nn_attention import (
    PortableLayerScale,
    PortableMHSA,
    PortableRepMixer,
    PortableTokenMixer,
)
from tac.portable_primitives.tensor import from_numpy, to_numpy

# Empirical ε band per Phase 1 PV (sister test convention).
ATOL_FP32 = 5e-3
RTOL_FP32 = 5e-3


pytestmark = pytest.mark.skipif(
    not (is_mlx_available() and is_pytorch_available()),
    reason="Both MLX + PyTorch must be installed for cross-backend equivalence tests",
)


def _seeded_input(shape: tuple[int, ...], seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.standard_normal(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# PortableLayerScale
# ---------------------------------------------------------------------------


def test_layerscale_token_form_equivalence() -> None:
    """PortableLayerScale on (B, N, C) token form MLX vs PyTorch within ε."""
    rng = np.random.RandomState(0)
    num_channels = 32
    gamma_np = rng.standard_normal((num_channels,)).astype(np.float32) * 0.01
    x_np = _seeded_input((2, 16, num_channels))

    ls_mlx = PortableLayerScale(num_channels, backend="mlx")
    ls_pt = PortableLayerScale(num_channels, backend="pytorch")
    ls_mlx.load_weights(gamma_np)
    ls_pt.load_weights(gamma_np)

    y_mlx = to_numpy(ls_mlx(from_numpy(x_np, "mlx")), "mlx")
    y_pt = to_numpy(ls_pt(from_numpy(x_np, "pytorch")), "pytorch")

    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_layerscale_nchw_form_equivalence() -> None:
    """PortableLayerScale on (B, C, H, W) NCHW form MLX vs PyTorch within ε."""
    rng = np.random.RandomState(1)
    num_channels = 16
    gamma_np = rng.standard_normal((num_channels,)).astype(np.float32) * 0.1
    x_np = _seeded_input((2, num_channels, 8, 8))

    ls_mlx = PortableLayerScale(num_channels, backend="mlx")
    ls_pt = PortableLayerScale(num_channels, backend="pytorch")
    ls_mlx.load_weights(gamma_np)
    ls_pt.load_weights(gamma_np)

    y_mlx = to_numpy(ls_mlx(from_numpy(x_np, "mlx")), "mlx")
    y_pt = to_numpy(ls_pt(from_numpy(x_np, "pytorch")), "pytorch")

    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_layerscale_default_init_value() -> None:
    """PortableLayerScale default init value is 1e-5 (FastViT paper)."""
    ls = PortableLayerScale(16, backend="pytorch")
    gamma = ls.export_weights()
    assert gamma.shape == (16,)
    np.testing.assert_allclose(gamma, np.full((16,), 1e-5, dtype=np.float32))


def test_layerscale_export_round_trip() -> None:
    """LayerScale.export_weights round-trips through load_weights."""
    gamma_np = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    ls = PortableLayerScale(4, backend="pytorch")
    ls.load_weights(gamma_np)
    exported = ls.export_weights()
    np.testing.assert_array_equal(exported, gamma_np)


# ---------------------------------------------------------------------------
# PortableMHSA
# ---------------------------------------------------------------------------


def _make_canonical_mhsa_weights(
    dim: int, num_heads: int, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate canonical seeded MHSA weight set (qkv_w, qkv_b, proj_w, proj_b)."""
    rng = np.random.RandomState(seed)
    qkv_w = rng.standard_normal((3 * dim, dim)).astype(np.float32) * 0.02
    qkv_b = rng.standard_normal((3 * dim,)).astype(np.float32) * 0.01
    proj_w = rng.standard_normal((dim, dim)).astype(np.float32) * 0.02
    proj_b = rng.standard_normal((dim,)).astype(np.float32) * 0.01
    return qkv_w, qkv_b, proj_w, proj_b


def test_mhsa_forward_equivalence_8_heads() -> None:
    """PortableMHSA forward MLX vs PyTorch within ε (8-head canonical FastViT)."""
    dim = 64
    num_heads = 8
    qkv_w, qkv_b, proj_w, proj_b = _make_canonical_mhsa_weights(dim, num_heads)
    x_np = _seeded_input((2, 16, dim), seed=11)

    mhsa_mlx = PortableMHSA(dim, num_heads, backend="mlx", qkv_bias=True)
    mhsa_pt = PortableMHSA(dim, num_heads, backend="pytorch", qkv_bias=True)
    mhsa_mlx.load_weights(qkv_w, qkv_b, proj_w, proj_b)
    mhsa_pt.load_weights(qkv_w, qkv_b, proj_w, proj_b)

    y_mlx = to_numpy(mhsa_mlx(from_numpy(x_np, "mlx")), "mlx")
    y_pt = to_numpy(mhsa_pt(from_numpy(x_np, "pytorch")), "pytorch")

    assert y_mlx.shape == y_pt.shape == (2, 16, dim)
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_mhsa_forward_shape_preservation() -> None:
    """PortableMHSA preserves input shape (B, N, dim)."""
    dim = 32
    mhsa = PortableMHSA(dim, num_heads=4, backend="pytorch")
    x_np = _seeded_input((1, 8, dim))
    y = mhsa(from_numpy(x_np, "pytorch"))
    assert tuple(y.shape) == (1, 8, dim)


def test_mhsa_rejects_dim_not_divisible_by_heads() -> None:
    """PortableMHSA constructor rejects dim not divisible by num_heads."""
    with pytest.raises(ValueError, match="must be divisible"):
        PortableMHSA(dim=10, num_heads=3, backend="pytorch")


def test_mhsa_qkv_no_bias_equivalence() -> None:
    """PortableMHSA with qkv_bias=False produces equivalent results across backends."""
    dim = 32
    num_heads = 4
    qkv_w, _, proj_w, proj_b = _make_canonical_mhsa_weights(dim, num_heads, seed=13)
    x_np = _seeded_input((1, 8, dim), seed=17)

    mhsa_mlx = PortableMHSA(dim, num_heads, backend="mlx", qkv_bias=False)
    mhsa_pt = PortableMHSA(dim, num_heads, backend="pytorch", qkv_bias=False)
    # qkv_bias=False => pass None for qkv_b; primitive auto-substitutes zeros.
    mhsa_mlx.load_weights(qkv_w, None, proj_w, proj_b)
    mhsa_pt.load_weights(qkv_w, None, proj_w, proj_b)

    y_mlx = to_numpy(mhsa_mlx(from_numpy(x_np, "mlx")), "mlx")
    y_pt = to_numpy(mhsa_pt(from_numpy(x_np, "pytorch")), "pytorch")

    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_mhsa_export_round_trip() -> None:
    """MHSA.export_weights returns the 4-field canonical dict."""
    dim = 16
    mhsa = PortableMHSA(dim, num_heads=2, backend="pytorch", seed=42)
    weights = mhsa.export_weights()
    assert set(weights.keys()) == {"qkv_weight", "qkv_bias", "proj_weight", "proj_bias"}
    assert weights["qkv_weight"].shape == (3 * dim, dim)
    assert weights["qkv_bias"].shape == (3 * dim,)
    assert weights["proj_weight"].shape == (dim, dim)
    assert weights["proj_bias"].shape == (dim,)


# ---------------------------------------------------------------------------
# PortableTokenMixer
# ---------------------------------------------------------------------------


def test_tokenmixer_token_form_equivalence() -> None:
    """PortableTokenMixer on (B, N, dim) token form MLX vs PyTorch within ε."""
    dim = 32
    hidden_dim = 128
    rng = np.random.RandomState(20)
    fc1_w = rng.standard_normal((hidden_dim, dim)).astype(np.float32) * 0.02
    fc1_b = rng.standard_normal((hidden_dim,)).astype(np.float32) * 0.01
    fc2_w = rng.standard_normal((dim, hidden_dim)).astype(np.float32) * 0.02
    fc2_b = rng.standard_normal((dim,)).astype(np.float32) * 0.01
    x_np = _seeded_input((2, 16, dim), seed=21)

    tm_mlx = PortableTokenMixer(dim, hidden_dim, backend="mlx")
    tm_pt = PortableTokenMixer(dim, hidden_dim, backend="pytorch")
    tm_mlx.load_weights(fc1_w, fc1_b, fc2_w, fc2_b)
    tm_pt.load_weights(fc1_w, fc1_b, fc2_w, fc2_b)

    y_mlx = to_numpy(tm_mlx(from_numpy(x_np, "mlx")), "mlx")
    y_pt = to_numpy(tm_pt(from_numpy(x_np, "pytorch")), "pytorch")

    assert y_mlx.shape == y_pt.shape == (2, 16, dim)
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_tokenmixer_nchw_form_equivalence() -> None:
    """PortableTokenMixer on (B, dim, H, W) NCHW MLX vs PyTorch within ε."""
    dim = 16
    hidden_dim = 64
    rng = np.random.RandomState(22)
    fc1_w = rng.standard_normal((hidden_dim, dim)).astype(np.float32) * 0.02
    fc1_b = rng.standard_normal((hidden_dim,)).astype(np.float32) * 0.01
    fc2_w = rng.standard_normal((dim, hidden_dim)).astype(np.float32) * 0.02
    fc2_b = rng.standard_normal((dim,)).astype(np.float32) * 0.01
    x_np = _seeded_input((1, dim, 4, 4), seed=23)

    tm_mlx = PortableTokenMixer(dim, hidden_dim, backend="mlx")
    tm_pt = PortableTokenMixer(dim, hidden_dim, backend="pytorch")
    tm_mlx.load_weights(fc1_w, fc1_b, fc2_w, fc2_b)
    tm_pt.load_weights(fc1_w, fc1_b, fc2_w, fc2_b)

    y_mlx = to_numpy(tm_mlx(from_numpy(x_np, "mlx")), "mlx")
    y_pt = to_numpy(tm_pt(from_numpy(x_np, "pytorch")), "pytorch")

    assert y_mlx.shape == y_pt.shape == (1, dim, 4, 4)
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_tokenmixer_default_hidden_dim_is_4x() -> None:
    """PortableTokenMixer default hidden_dim is 4*dim (transformer convention)."""
    tm = PortableTokenMixer(32, backend="pytorch")
    assert tm.hidden_dim == 128


# ---------------------------------------------------------------------------
# PortableRepMixer
# ---------------------------------------------------------------------------


def _make_canonical_repmixer_weights(
    dim: int, kernel_size: int = 3, seed: int = 30
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate canonical seeded RepMixer weight set (dw3_w, dw3_b, dw1_w, dw1_b)."""
    rng = np.random.RandomState(seed)
    dw3_w = rng.standard_normal((dim, 1, kernel_size, kernel_size)).astype(np.float32) * 0.1
    dw3_b = rng.standard_normal((dim,)).astype(np.float32) * 0.01
    dw1_w = rng.standard_normal((dim, 1, 1, 1)).astype(np.float32) * 0.1
    dw1_b = rng.standard_normal((dim,)).astype(np.float32) * 0.01
    return dw3_w, dw3_b, dw1_w, dw1_b


def test_repmixer_training_mode_equivalence() -> None:
    """PortableRepMixer training mode (3 branches) MLX vs PyTorch within ε."""
    dim = 16
    dw3_w, dw3_b, dw1_w, dw1_b = _make_canonical_repmixer_weights(dim)
    x_np = _seeded_input((2, dim, 8, 8), seed=31)

    rm_mlx = PortableRepMixer(dim, backend="mlx")
    rm_pt = PortableRepMixer(dim, backend="pytorch")
    rm_mlx.load_weights(dw3_w, dw3_b, dw1_w, dw1_b)
    rm_pt.load_weights(dw3_w, dw3_b, dw1_w, dw1_b)
    # Both default to train mode (3 branches).

    y_mlx = to_numpy(rm_mlx(from_numpy(x_np, "mlx")), "mlx")
    y_pt = to_numpy(rm_pt(from_numpy(x_np, "pytorch")), "pytorch")

    assert y_mlx.shape == y_pt.shape == (2, dim, 8, 8)
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_repmixer_train_inference_equivalence_post_reparameterize() -> None:
    """RepMixer training-mode output equals inference-mode output after reparameterize().

    This is the canonical correctness contract from FastViT paper: the 3-branch
    sum DW3x3(x) + DW1x1(x) + x must equal DW3x3_fused(x) where the fused
    kernel = W_3x3 + pad(W_1x1) + identity. This test pins the math.
    """
    dim = 16
    dw3_w, dw3_b, dw1_w, dw1_b = _make_canonical_repmixer_weights(dim, seed=32)
    x_np = _seeded_input((1, dim, 6, 6), seed=33)

    rm = PortableRepMixer(dim, backend="pytorch")
    rm.load_weights(dw3_w, dw3_b, dw1_w, dw1_b)

    # Training-mode output (3 branches).
    rm.train()
    y_train = to_numpy(rm(from_numpy(x_np, "pytorch")), "pytorch")

    # Re-parameterize and switch to inference mode.
    rm.reparameterize()
    rm.eval()
    y_eval = to_numpy(rm(from_numpy(x_np, "pytorch")), "pytorch")

    # Equivalent within fp32 floating-point noise; bias is summed both ways so
    # difference should be tiny.
    np.testing.assert_allclose(y_train, y_eval, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_repmixer_reparameterize_equivalence_mlx_vs_pytorch() -> None:
    """RepMixer reparameterized (inference path) MLX vs PyTorch within ε."""
    dim = 8
    dw3_w, dw3_b, dw1_w, dw1_b = _make_canonical_repmixer_weights(dim, seed=34)
    x_np = _seeded_input((1, dim, 4, 4), seed=35)

    rm_mlx = PortableRepMixer(dim, backend="mlx")
    rm_pt = PortableRepMixer(dim, backend="pytorch")
    rm_mlx.load_weights(dw3_w, dw3_b, dw1_w, dw1_b)
    rm_pt.load_weights(dw3_w, dw3_b, dw1_w, dw1_b)
    rm_mlx.reparameterize()
    rm_pt.reparameterize()

    y_mlx = to_numpy(rm_mlx(from_numpy(x_np, "mlx")), "mlx")
    y_pt = to_numpy(rm_pt(from_numpy(x_np, "pytorch")), "pytorch")

    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_repmixer_export_after_reparameterize() -> None:
    """RepMixer.export_weights returns fused dict after reparameterize()."""
    dim = 8
    dw3_w, dw3_b, dw1_w, dw1_b = _make_canonical_repmixer_weights(dim, seed=36)
    rm = PortableRepMixer(dim, backend="pytorch")
    rm.load_weights(dw3_w, dw3_b, dw1_w, dw1_b)

    # Before reparam: 4-field training dict.
    weights_train = rm.export_weights()
    assert set(weights_train.keys()) == {"dw3_weight", "dw3_bias", "dw1_weight", "dw1_bias"}

    # After reparam: 2-field fused dict.
    rm.reparameterize()
    weights_fused = rm.export_weights()
    assert set(weights_fused.keys()) == {"fused_weight", "fused_bias"}
    assert weights_fused["fused_weight"].shape == (dim, 1, 3, 3)
    assert weights_fused["fused_bias"].shape == (dim,)
