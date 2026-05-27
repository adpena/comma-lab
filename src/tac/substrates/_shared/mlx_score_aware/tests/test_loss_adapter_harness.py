# SPDX-License-Identifier: MIT
"""MLX-bound tests: score-aware loss + adapter + end-to-end harness run.

MLX-bound tests skip cleanly on non-Apple-Silicon CI.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.substrates._shared.mlx_score_aware import (
    MlxScoreAwareAdapter,
    RendererBundle,
    decode_frames_nhwc01,
    run_mlx_score_aware_full_main,
    score_aware_loss,
)

try:
    import mlx.core as _mx  # noqa: F401

    _MLX = True
except ImportError:
    _MLX = False

mlx_only = pytest.mark.skipif(not _MLX, reason="MLX required (Apple Silicon)")

_LANE = "lane_mlx_score_aware_harness_refactor_plus_4_unlock_20260527"


def _tiny_dreamer_bundle(num_pairs: int = 4, distill: float = 0.5) -> RendererBundle:
    import mlx.core as mx

    from tac.substrates.dreamer_v3_rssm.module import (
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
    )

    cfg = DreamerV3RSSMConfig(
        num_pairs=num_pairs,
        num_groups=2,
        num_categories=4,
        decoder_latent_dim=8,
        base_channels=4,
        eval_size=(384, 512),
    )
    model = DreamerV3RSSMSubstrateMLX(cfg)
    t0 = mx.zeros((num_pairs, 384, 512, 3))
    t1 = mx.zeros((num_pairs, 384, 512, 3))
    return RendererBundle(
        model=model,
        target_rgb_0=t0,
        target_rgb_1=t1,
        num_pairs=num_pairs,
        forward_convention="call_b2chw_255",
        distillation_weight=distill,
    )


# --------------------------------------------------------------------------- #
# loss module
# --------------------------------------------------------------------------- #


@mlx_only
def test_decode_frames_nhwc01_shapes() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle()
    idx = mx.array([0, 1], dtype=mx.int32)
    rgb_0, rgb_1 = decode_frames_nhwc01(bundle, idx)
    assert rgb_0.shape == (2, 384, 512, 3)
    assert rgb_1.shape == (2, 384, 512, 3)


@mlx_only
def test_score_aware_loss_is_finite_and_decomposed() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(distill=0.5)
    idx = mx.array([0, 1], dtype=mx.int32)
    total, parts = score_aware_loss(bundle, idx)
    mx.eval(total)
    assert float(total.item()) == float(total.item())  # not NaN
    assert {"recon", "distill", "total"} <= set(parts)


@mlx_only
def test_score_aware_loss_no_distill_when_weight_zero() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(distill=0.0)
    idx = mx.array([0, 1], dtype=mx.int32)
    _total, parts = score_aware_loss(bundle, idx)
    assert "distill" not in parts
    assert "recon" in parts


@mlx_only
def test_score_aware_loss_extra_term_weighted() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(distill=0.0)
    bundle.extra_loss_terms = lambda _m, _i: {"commit": mx.array(2.0)}
    bundle.extra_loss_weights = {"commit": 0.5}
    idx = mx.array([0], dtype=mx.int32)
    _total, parts = score_aware_loss(bundle, idx)
    assert "commit" in parts
    mx.eval(parts["commit"])
    assert abs(float(parts["commit"].item()) - 2.0) < 1e-5


# --------------------------------------------------------------------------- #
# adapter module
# --------------------------------------------------------------------------- #


@mlx_only
def test_adapter_train_step_reduces_loss_over_steps() -> None:
    import mlx.core as mx

    bundle = _tiny_dreamer_bundle(distill=0.0)
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    batch = mx.array([0, 1, 2, 3], dtype=mx.int32)
    losses = [
        adapter.train_step(batch, learning_rate=1e-2, loss_weights={})["total"]
        for _ in range(20)
    ]
    assert losses[-1] < losses[0]


@mlx_only
def test_adapter_satisfies_protocol() -> None:
    from tac.training.long_training_canonical import validate_substrate_adapter

    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(), substrate_id="dreamer_v3_rssm"
    )
    validate_substrate_adapter(adapter)


@mlx_only
def test_adapter_optimizer_step_raises_style_a_stub() -> None:
    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(), substrate_id="dreamer_v3_rssm"
    )
    with pytest.raises(NotImplementedError, match="Style B train_step"):
        adapter.optimizer_step(adapter.model, None, 1e-3)


@mlx_only
def test_adapter_export_state_dict_writes_portable_npsd(tmp_path: Path) -> None:
    from tac.substrates._shared.numpy_portable_inflate import (
        unpack_state_dict_numpy,
    )

    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(), substrate_id="dreamer_v3_rssm"
    )
    target = tmp_path / "ckpt.state"
    adapter.export_state_dict(adapter.model, target)
    blob = target.with_suffix(target.suffix + ".npsd")
    assert blob.is_file()
    restored = unpack_state_dict_numpy(blob.read_bytes())
    assert len(restored) > 0


@mlx_only
def test_adapter_score_aware_components_defers() -> None:
    import mlx.core as mx

    adapter = MlxScoreAwareAdapter(
        _tiny_dreamer_bundle(), substrate_id="dreamer_v3_rssm"
    )
    assert adapter.score_aware_components(adapter.model, mx.array([0])) is None


# --------------------------------------------------------------------------- #
# harness orchestrator (end-to-end through canonical run_long_training)
# --------------------------------------------------------------------------- #


@mlx_only
def test_run_mlx_score_aware_full_main_end_to_end(tmp_path: Path) -> None:
    bundle = _tiny_dreamer_bundle(num_pairs=4, distill=0.5)
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="dreamer_v3_rssm",
        lane_id=_LANE,
        output_dir=tmp_path / "run",
        epochs=3,
        batch_pair_indices_per_step=2,
        learning_rate=1e-3,
        seed=0,
        notes="harness refactor end-to-end: dreamer renderer + zero targets",
    )
    assert artifact.total_epochs_completed == 3
    assert artifact.promotable is False
    d = artifact.as_dict()
    assert d.get("score_claim") is False
    assert d.get("promotion_eligible") is False


@mlx_only
def test_run_verifies_inflate_portability_fails_closed(tmp_path: Path) -> None:
    from tac.substrates._shared.mlx_score_aware import MlxScoreAwareHarnessError

    bad_inflate = tmp_path / "inflate.py"
    bad_inflate.write_text("import torch\n", encoding="utf-8")
    bundle = _tiny_dreamer_bundle(num_pairs=4, distill=0.0)
    with pytest.raises(MlxScoreAwareHarnessError, match="forbidden non-portable"):
        run_mlx_score_aware_full_main(
            bundle=bundle,
            substrate_id="dreamer_v3_rssm",
            lane_id=_LANE,
            output_dir=tmp_path / "run",
            epochs=1,
            batch_pair_indices_per_step=2,
            inflate_py_path=bad_inflate,
            notes="inflate portability fail-closed before training",
        )
