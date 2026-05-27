# SPDX-License-Identifier: MIT
"""MLX-SCORE-AWARE-HARNESS-WAVE unlock tests for coin_pp_implicit_neural_representation.

Verifies the trainable ``CoinPPRendererMLX`` nn.Module (the prior blocker was
that the substrate shipped only config + estimators with NO renderer forward).
MLX-bound; skip cleanly off Apple Silicon.
"""
from __future__ import annotations

import pytest

try:
    import mlx.core as _mx  # noqa: F401

    _MLX = True
except ImportError:
    _MLX = False

mlx_only = pytest.mark.skipif(not _MLX, reason="MLX required (Apple Silicon)")


def _tiny_cfg():
    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPImplicitNeuralRepresentationConfig,
    )

    return CoinPPImplicitNeuralRepresentationConfig(
        mod_dim=16, pos_dim=8, hidden_dim=32, num_hidden_layers=2, num_pairs=4
    )


@mlx_only
def test_renderer_is_nn_module() -> None:
    import mlx.nn as nn

    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPRendererMLX,
    )

    model = CoinPPRendererMLX(_tiny_cfg())
    assert isinstance(model, nn.Module)
    assert len(model.parameters()) > 0


@mlx_only
def test_reconstruct_pair_shape_and_range() -> None:
    import mlx.core as mx

    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPRendererMLX,
    )

    cfg = _tiny_cfg()
    model = CoinPPRendererMLX(cfg)
    idx = mx.array([0, 1], dtype=mx.int32)
    rgb_0, rgb_1 = model.reconstruct_pair(idx)
    assert rgb_0.shape == (2, 3, cfg.eval_h, cfg.eval_w)
    assert rgb_1.shape == (2, 3, cfg.eval_h, cfg.eval_w)
    mx.eval(rgb_0, rgb_1)
    assert float(rgb_0.min()) >= 0.0 and float(rgb_0.max()) <= 1.0


@mlx_only
def test_value_and_grad_trains() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPRendererMLX,
    )

    model = CoinPPRendererMLX(_tiny_cfg())
    idx = mx.array([0, 1, 2, 3], dtype=mx.int32)

    def loss(m):
        a, b = m.reconstruct_pair(idx)
        return mx.mean(a * a) + mx.mean(b * b)

    lg = nn.value_and_grad(model, loss)
    v, g = lg(model)
    mx.eval(v)
    assert float(v.item()) == float(v.item())
    assert len(g) > 0


@mlx_only
def test_two_frames_differ_by_coord_time() -> None:
    """frame_0 (t=-1) and frame_1 (t=+1) must not be identical (t conditioning)."""
    import mlx.core as mx

    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPRendererMLX,
    )

    model = CoinPPRendererMLX(_tiny_cfg())
    rgb_0, rgb_1 = model.reconstruct_pair(mx.array([0], dtype=mx.int32))
    mx.eval(rgb_0, rgb_1)
    assert float(mx.mean(mx.abs(rgb_0 - rgb_1)).item()) > 0.0
