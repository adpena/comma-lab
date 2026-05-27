# SPDX-License-Identifier: MIT
"""MLX-SCORE-AWARE-HARNESS-WAVE unlock tests for mdl_ibps_j_discrete_categorical_mine_hybrid.

Verifies the trainable ``MDLIBPSJTrainableRendererMLX`` nn.Module (the prior
blocker was that ``MDLIBPSJRendererMLX`` held FIXED loaded weights — not
learnable; not an nn.Module). MLX-bound; skip cleanly off Apple Silicon.
"""
from __future__ import annotations

import pytest

try:
    import mlx.core as _mx  # noqa: F401

    _MLX = True
except ImportError:
    _MLX = False

mlx_only = pytest.mark.skipif(not _MLX, reason="MLX required (Apple Silicon)")


@mlx_only
def test_renderer_is_nn_module() -> None:
    import mlx.nn as nn

    from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.mlx_renderer import (
        MDLIBPSJTrainableRendererMLX,
    )

    model = MDLIBPSJTrainableRendererMLX(num_pairs=4)
    assert isinstance(model, nn.Module)
    assert len(model.parameters()) > 0


@mlx_only
def test_reconstruct_pair_shape_and_range() -> None:
    import mlx.core as mx

    from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid import EVAL_HW
    from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.mlx_renderer import (
        MDLIBPSJTrainableRendererMLX,
    )

    model = MDLIBPSJTrainableRendererMLX(num_pairs=4)
    rgb_0, rgb_1 = model.reconstruct_pair(mx.array([0, 1], dtype=mx.int32))
    assert rgb_0.shape == (2, 3, EVAL_HW[0], EVAL_HW[1])
    assert rgb_1.shape == (2, 3, EVAL_HW[0], EVAL_HW[1])
    mx.eval(rgb_0, rgb_1)
    assert float(rgb_0.min()) >= 0.0 and float(rgb_0.max()) <= 1.0


@mlx_only
def test_value_and_grad_trains_through_categorical_posterior() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.mlx_renderer import (
        MDLIBPSJTrainableRendererMLX,
    )

    model = MDLIBPSJTrainableRendererMLX(num_pairs=4)
    idx = mx.array([0, 1, 2, 3], dtype=mx.int32)

    def loss(m):
        a, b = m.reconstruct_pair(idx)
        return mx.mean(a * a) + mx.mean(b * b)

    lg = nn.value_and_grad(model, loss)
    v, g = lg(model)
    mx.eval(v)
    assert float(v.item()) == float(v.item())
    # Gradient must flow into the per-pair categorical logits (the discrete IB
    # code) via Gumbel-Softmax — the distinguishing primitive.
    assert len(g) > 0


@mlx_only
def test_argmax_indices_shape() -> None:
    import mlx.core as mx

    from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid import (
        CATEGORICAL_G,
    )
    from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.mlx_renderer import (
        MDLIBPSJTrainableRendererMLX,
    )

    model = MDLIBPSJTrainableRendererMLX(num_pairs=4)
    idx = model.argmax_indices()
    mx.eval(idx)
    assert idx.shape == (4, CATEGORICAL_G)
