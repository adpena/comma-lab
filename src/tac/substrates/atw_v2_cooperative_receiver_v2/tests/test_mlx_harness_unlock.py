# SPDX-License-Identifier: MIT
"""MLX-SCORE-AWARE-HARNESS-WAVE unlock tests for atw_v2_cooperative_receiver_v2.

Verifies the trainable ``ATWv2CooperativeReceiverV2TrainableMLX`` nn.Module
renderer that unblocked ``_full_main`` (the prior blocker was a non-nn.Module
renderer + a call-time ``pose_delta`` arg). MLX-bound; skip cleanly off Apple
Silicon.
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
    from tac.substrates.atw_v2_cooperative_receiver_v2.numpy_reference import (
        CooperativeReceiverConfig,
    )

    return CooperativeReceiverConfig(
        num_pairs=4,
        latent_dim=8,
        cond_embed_dim=4,
        decoder_embed_dim=8,
        decoder_initial_grid_h=3,
        decoder_initial_grid_w=4,
        decoder_channels=(8, 6),
        decoder_num_upsample_blocks=2,
        output_height=12,
        output_width=16,
    )


@mlx_only
def test_trainable_renderer_is_nn_module() -> None:
    import mlx.nn as nn

    from tac.substrates.atw_v2_cooperative_receiver_v2.mlx_renderer import (
        ATWv2CooperativeReceiverV2TrainableMLX,
    )

    model = ATWv2CooperativeReceiverV2TrainableMLX(_tiny_cfg())
    assert isinstance(model, nn.Module)
    assert len(model.parameters()) > 0


@mlx_only
def test_reconstruct_pair_shape_and_range() -> None:
    import mlx.core as mx

    from tac.substrates.atw_v2_cooperative_receiver_v2.mlx_renderer import (
        ATWv2CooperativeReceiverV2TrainableMLX,
    )

    cfg = _tiny_cfg()
    model = ATWv2CooperativeReceiverV2TrainableMLX(cfg)
    idx = mx.array([0, 1], dtype=mx.int32)
    rgb_0, rgb_1 = model.reconstruct_pair(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (2, 3, cfg.output_height, cfg.output_width)
    mx.eval(rgb_0, rgb_1)
    assert float(rgb_0.min()) >= 0.0 and float(rgb_0.max()) <= 1.0


@mlx_only
def test_value_and_grad_trains() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    from tac.substrates.atw_v2_cooperative_receiver_v2.mlx_renderer import (
        ATWv2CooperativeReceiverV2TrainableMLX,
    )

    model = ATWv2CooperativeReceiverV2TrainableMLX(_tiny_cfg())
    idx = mx.array([0, 1, 2, 3], dtype=mx.int32)

    def loss(m):
        a, b = m.reconstruct_pair(idx)
        return mx.mean(a * a) + mx.mean(b * b)

    lg = nn.value_and_grad(model, loss)
    v, g = lg(model)
    mx.eval(v)
    assert float(v.item()) == float(v.item())  # finite
    assert len(g) > 0


@mlx_only
def test_full_main_via_harness_e2e(tmp_path) -> None:
    """End-to-end: trainable renderer through the canonical harness."""
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware import (
        RendererBundle,
        run_mlx_score_aware_full_main,
    )
    from tac.substrates.atw_v2_cooperative_receiver_v2.mlx_renderer import (
        ATWv2CooperativeReceiverV2TrainableMLX,
    )

    cfg = _tiny_cfg()
    model = ATWv2CooperativeReceiverV2TrainableMLX(cfg)
    t0 = mx.zeros((cfg.num_pairs, cfg.output_height, cfg.output_width, 3))
    t1 = mx.zeros((cfg.num_pairs, cfg.output_height, cfg.output_width, 3))
    bundle = RendererBundle(
        model=model,
        target_rgb_0=t0,
        target_rgb_1=t1,
        num_pairs=cfg.num_pairs,
        forward_convention="reconstruct_pair_nchw01",
        distillation_weight=0.0,
    )
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="atw_v2_cooperative_receiver_v2",
        lane_id="lane_path_3_h_atw_v2_cooperative_receiver_cargo_cult_first_20260526",
        output_dir=tmp_path / "run",
        epochs=2,
        batch_pair_indices_per_step=2,
        notes="atw_v2 harness-unlock e2e test: trainable renderer + zero targets",
    )
    assert artifact.total_epochs_completed == 2
    assert artifact.promotable is False
