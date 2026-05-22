# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import numpy as np

from tac.local_acceleration.mlx_scorer_adapters import (
    MLXEfficientNetSqueezeExciteAdapter,
    _mlx_array_to_numpy,
    mlx_sigmoid,
    mlx_silu,
    nchw_to_nhwc,
    nhwc_to_nchw,
    temporary_mlx_device,
)
from tac.local_acceleration.mlx_scorer_response import _load_upstream_distortion_net

REPO = Path(__file__).resolve().parents[3]


def test_efficientnet_squeeze_excite_uses_sequential_width_then_height_pool() -> None:
    se = _load_upstream_distortion_net(REPO).segnet.encoder.model.blocks[0][0].se
    x_nchw = np.linspace(-3.0, 5.0, num=2 * 32 * 64 * 80, dtype=np.float32).reshape(
        2,
        32,
        64,
        80,
    )

    with temporary_mlx_device("cpu"):
        import mlx.core as mx

        adapter = MLXEfficientNetSqueezeExciteAdapter(se)
        x = mx.array(nchw_to_nhwc(x_nchw))
        actual = nhwc_to_nchw(_mlx_array_to_numpy(adapter(x)))

        sequential_pool = mx.mean(mx.mean(x, axis=2, keepdims=True), axis=1, keepdims=True)
        sequential = x * mlx_sigmoid(adapter.conv_expand(mlx_silu(adapter.conv_reduce(sequential_pool))))
        sequential = nhwc_to_nchw(_mlx_array_to_numpy(sequential))

        tuple_pool = mx.mean(x, axis=(1, 2), keepdims=True)
        tuple_path = x * mlx_sigmoid(adapter.conv_expand(mlx_silu(adapter.conv_reduce(tuple_pool))))
        tuple_path = nhwc_to_nchw(_mlx_array_to_numpy(tuple_path))

    assert np.max(np.abs(actual - sequential)) == 0.0
    assert np.max(np.abs(actual - tuple_path)) > 0.0
