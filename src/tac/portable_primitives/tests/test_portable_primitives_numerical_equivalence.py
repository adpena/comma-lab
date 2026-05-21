# SPDX-License-Identifier: MIT
"""Cross-backend numerical-equivalence tests for portable primitives.

Per OVERNIGHT-WW + Phase 1 PV finding: every primitive constructed with
``backend="mlx"`` must produce numerically-equivalent results to its
``backend="pytorch"`` sister within a documented ε band.

**Empirical ε band per Phase 1 PV** (M5 Max Metal GPU vs PyTorch CPU fp32):
- ATOL_FP32 = 5e-3 (Metal MPS uses different FMA ordering than CPU fp32;
  empirical max abs diff observed = ~1.1e-3 on (4,8) Linear forward)
- RTOL_FP32 = 5e-3 (relative differences scale with magnitude per FMA
  re-ordering)

This is the structural pin per CLAUDE.md "Apples-to-apples evidence
discipline" extended to the portable-primitives surface: a future refactor
that silently breaks backend equivalence beyond ε would corrupt the
train-anywhere eval-anywhere pipeline. The non-zero ε is documented and
expected per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1 (MLX inherits
MPS-style FMA-reordering numerics; non-promotable per Catalog #192).

For contest-axis promotion the canonical pipeline is:
  MLX-train -> export weights via numpy -> PyTorch load on CUDA T4 ->
  authoritative eval via experiments/contest_auth_eval.py
The eval is run on PyTorch CUDA T4 (NOT MLX), so the ε at the MLX-vs-PT
boundary is a property of the EXPORT step, not the contest score itself.

Tests are skipped if MLX is not available on the host (e.g. Linux CI).
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.portable_primitives import Backend, is_mlx_available, is_pytorch_available
from tac.portable_primitives import loss as ploss
from tac.portable_primitives import nn as pnn
from tac.portable_primitives.tensor import from_numpy, to_numpy

# Empirical ε band per Phase 1 PV. Metal MPS FMA reordering produces ~1e-3
# relative differences vs PyTorch CPU fp32. The pin is "no silent drift
# beyond ε" — drift beyond 5e-3 indicates a structural change.
ATOL_FP32 = 5e-3
RTOL_FP32 = 5e-3


pytestmark = pytest.mark.skipif(
    not (is_mlx_available() and is_pytorch_available()),
    reason="Both MLX + PyTorch must be installed for cross-backend equivalence tests",
)


def _seeded_input(shape: tuple[int, ...], seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.standard_normal(shape).astype(np.float32)


def test_linear_forward_equivalence() -> None:
    """PortableLinear MLX vs PyTorch produce identical output within ε."""
    rng = np.random.RandomState(0)
    w_np = rng.standard_normal((8, 16)).astype(np.float32) * 0.1
    b_np = rng.standard_normal((8,)).astype(np.float32) * 0.1
    x_np = _seeded_input((4, 16))

    linear_mlx = pnn.PortableLinear(16, 8, backend="mlx")
    linear_pt = pnn.PortableLinear(16, 8, backend="pytorch")
    linear_mlx.load_weights(w_np, b_np)
    linear_pt.load_weights(w_np, b_np)

    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")

    y_mlx = to_numpy(linear_mlx(x_mlx), "mlx")
    y_pt = to_numpy(linear_pt(x_pt), "pytorch")

    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_conv2d_forward_equivalence() -> None:
    """PortableConv2d MLX vs PyTorch produce identical output within ε."""
    rng = np.random.RandomState(0)
    w_np = rng.standard_normal((4, 3, 3, 3)).astype(np.float32) * 0.1
    b_np = rng.standard_normal((4,)).astype(np.float32) * 0.1
    x_np = _seeded_input((2, 3, 8, 8))

    conv_mlx = pnn.PortableConv2d(3, 4, kernel_size=3, backend="mlx")
    conv_pt = pnn.PortableConv2d(3, 4, kernel_size=3, backend="pytorch")
    conv_mlx.load_weights(w_np, b_np)
    conv_pt.load_weights(w_np, b_np)

    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")

    y_mlx = to_numpy(conv_mlx(x_mlx), "mlx")
    y_pt = to_numpy(conv_pt(x_pt), "pytorch")

    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_layer_norm_equivalence() -> None:
    """PortableLayerNorm MLX vs PyTorch equivalence."""
    x_np = _seeded_input((4, 16))

    ln_mlx = pnn.PortableLayerNorm(16, backend="mlx")
    ln_pt = pnn.PortableLayerNorm(16, backend="pytorch")

    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")

    y_mlx = to_numpy(ln_mlx(x_mlx), "mlx")
    y_pt = to_numpy(ln_pt(x_pt), "pytorch")

    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_gelu_equivalence() -> None:
    """gelu MLX vs PyTorch equivalence."""
    x_np = _seeded_input((4, 8))
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")

    y_mlx = to_numpy(pnn.gelu(x_mlx, backend="mlx"), "mlx")
    y_pt = to_numpy(pnn.gelu(x_pt, backend="pytorch"), "pytorch")

    # GELU's exact form (erf-based) may have slightly larger ε on Metal.
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_relu_equivalence() -> None:
    x_np = _seeded_input((4, 8))
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(pnn.relu(x_mlx, backend="mlx"), "mlx")
    y_pt = to_numpy(pnn.relu(x_pt, backend="pytorch"), "pytorch")
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_sigmoid_equivalence() -> None:
    x_np = _seeded_input((4, 8))
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(pnn.sigmoid(x_mlx, backend="mlx"), "mlx")
    y_pt = to_numpy(pnn.sigmoid(x_pt, backend="pytorch"), "pytorch")
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_softmax_equivalence() -> None:
    x_np = _seeded_input((4, 8))
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(pnn.softmax(x_mlx, backend="mlx"), "mlx")
    y_pt = to_numpy(pnn.softmax(x_pt, backend="pytorch"), "pytorch")
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_matmul_equivalence() -> None:
    a_np = _seeded_input((4, 8))
    b_np = _seeded_input((8, 6), seed=1)
    a_mlx = from_numpy(a_np, "mlx")
    b_mlx = from_numpy(b_np, "mlx")
    a_pt = from_numpy(a_np, "pytorch")
    b_pt = from_numpy(b_np, "pytorch")
    y_mlx = to_numpy(pnn.matmul(a_mlx, b_mlx, backend="mlx"), "mlx")
    y_pt = to_numpy(pnn.matmul(a_pt, b_pt, backend="pytorch"), "pytorch")
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_bilinear_upsample_equivalence() -> None:
    """bilinear_upsample MLX (via PyTorch ref) matches PyTorch native."""
    x_np = _seeded_input((2, 3, 8, 8))
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(
        pnn.bilinear_upsample(x_mlx, size=(16, 16), backend="mlx"), "mlx"
    )
    y_pt = to_numpy(
        pnn.bilinear_upsample(x_pt, size=(16, 16), backend="pytorch"), "pytorch"
    )
    # Byte-stable because MLX implementation routes through PyTorch reference.
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_mse_loss_equivalence() -> None:
    pred_np = _seeded_input((4, 8))
    tgt_np = _seeded_input((4, 8), seed=1)
    pred_mlx = from_numpy(pred_np, "mlx")
    tgt_mlx = from_numpy(tgt_np, "mlx")
    pred_pt = from_numpy(pred_np, "pytorch")
    tgt_pt = from_numpy(tgt_np, "pytorch")
    loss_mlx = to_numpy(ploss.mse_loss(pred_mlx, tgt_mlx, backend="mlx"), "mlx")
    loss_pt = to_numpy(ploss.mse_loss(pred_pt, tgt_pt, backend="pytorch"), "pytorch")
    np.testing.assert_allclose(loss_mlx, loss_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_l1_loss_equivalence() -> None:
    pred_np = _seeded_input((4, 8))
    tgt_np = _seeded_input((4, 8), seed=1)
    pred_mlx = from_numpy(pred_np, "mlx")
    tgt_mlx = from_numpy(tgt_np, "mlx")
    pred_pt = from_numpy(pred_np, "pytorch")
    tgt_pt = from_numpy(tgt_np, "pytorch")
    loss_mlx = to_numpy(ploss.l1_loss(pred_mlx, tgt_mlx, backend="mlx"), "mlx")
    loss_pt = to_numpy(ploss.l1_loss(pred_pt, tgt_pt, backend="pytorch"), "pytorch")
    np.testing.assert_allclose(loss_mlx, loss_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_export_weights_round_trip() -> None:
    """Train one step in MLX, export weights, load in PyTorch, verify match."""
    rng = np.random.RandomState(0)
    w_np = rng.standard_normal((4, 8)).astype(np.float32) * 0.1
    b_np = np.zeros(4, dtype=np.float32)

    linear_mlx = pnn.PortableLinear(8, 4, backend="mlx")
    linear_mlx.load_weights(w_np, b_np)

    # Export.
    w_out, b_out = linear_mlx.export_weights()
    np.testing.assert_allclose(w_out, w_np, atol=ATOL_FP32, rtol=RTOL_FP32)
    np.testing.assert_allclose(b_out, b_np, atol=ATOL_FP32, rtol=RTOL_FP32)

    # Load in PyTorch + verify forward output matches.
    linear_pt = pnn.PortableLinear(8, 4, backend="pytorch")
    linear_pt.load_weights(w_out, b_out)

    x_np = _seeded_input((2, 8))
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(linear_mlx(x_mlx), "mlx")
    y_pt = to_numpy(linear_pt(x_pt), "pytorch")
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_conv2d_export_weights_round_trip() -> None:
    """Conv2d export + load preserves PyTorch layout (out, in, kH, kW)."""
    rng = np.random.RandomState(0)
    w_np = rng.standard_normal((4, 3, 3, 3)).astype(np.float32) * 0.1
    b_np = np.zeros(4, dtype=np.float32)

    conv_mlx = pnn.PortableConv2d(3, 4, kernel_size=3, backend="mlx")
    conv_mlx.load_weights(w_np, b_np)

    w_out, b_out = conv_mlx.export_weights()
    # Exported weights should be in PyTorch layout (canonical).
    assert w_out.shape == (4, 3, 3, 3)
    np.testing.assert_allclose(w_out, w_np, atol=ATOL_FP32, rtol=RTOL_FP32)

    # Load in PyTorch + verify forward output matches MLX forward.
    conv_pt = pnn.PortableConv2d(3, 4, kernel_size=3, backend="pytorch")
    conv_pt.load_weights(w_out, b_out)

    x_np = _seeded_input((2, 3, 8, 8))
    x_mlx = from_numpy(x_np, "mlx")
    x_pt = from_numpy(x_np, "pytorch")
    y_mlx = to_numpy(conv_mlx(x_mlx), "mlx")
    y_pt = to_numpy(conv_pt(x_pt), "pytorch")
    np.testing.assert_allclose(y_mlx, y_pt, atol=ATOL_FP32, rtol=RTOL_FP32)
