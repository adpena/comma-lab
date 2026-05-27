# SPDX-License-Identifier: MIT
"""Tests for the canonical MLX-first score-aware training harness.

MLX-SCORE-AWARE-HARNESS-WAVE 2026-05-27. Covers the substrate-AGNOSTIC harness:
numpy-portable inflate contract verification (ast-based; no MLX/torch import at
decode), the RendererBundle contract, the score-aware loss (MSE + gradient-
reachable Hinton-KL surrogate), the Style-B adapter Protocol conformance, and a
real end-to-end MLX training run through the canonical ``run_long_training``.

MLX-bound tests skip cleanly on non-Apple-Silicon CI; the numpy-portable
inflate contract test runs everywhere (ast-only, no MLX).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.substrates._shared.mlx_score_aware_full_main import (
    FORBIDDEN_INFLATE_IMPORT_ROOTS,
    MLX_EVIDENCE_GRADE,
    MlxScoreAwareHarnessError,
    RendererBundle,
    assert_numpy_portable_inflate,
)

try:
    import mlx.core as _mx  # noqa: F401

    _MLX = True
except ImportError:
    _MLX = False

mlx_only = pytest.mark.skipif(not _MLX, reason="MLX required (Apple Silicon)")


# ---------------------------------------------------------------------------
# Numpy-portable inflate contract (ast-only; runs everywhere)
# ---------------------------------------------------------------------------


def test_assert_numpy_portable_inflate_accepts_numpy_pil(tmp_path: Path) -> None:
    inflate = tmp_path / "inflate.py"
    inflate.write_text(
        "import numpy as np\n"
        "from PIL import Image\n"
        "import struct\n"
        "def inflate_one_video(b, out, device='cpu'):\n"
        "    pass\n",
        encoding="utf-8",
    )
    result = assert_numpy_portable_inflate(inflate)
    assert result["numpy_portable"] is True
    assert "numpy" in result["import_roots"]


def test_assert_numpy_portable_inflate_rejects_mlx_import(tmp_path: Path) -> None:
    inflate = tmp_path / "inflate.py"
    inflate.write_text("import mlx.core as mx\n", encoding="utf-8")
    with pytest.raises(MlxScoreAwareHarnessError, match="forbidden non-portable"):
        assert_numpy_portable_inflate(inflate)


def test_assert_numpy_portable_inflate_rejects_torch_import(tmp_path: Path) -> None:
    inflate = tmp_path / "inflate.py"
    inflate.write_text("import torch\n", encoding="utf-8")
    with pytest.raises(MlxScoreAwareHarnessError, match="forbidden non-portable"):
        assert_numpy_portable_inflate(inflate)


def test_assert_numpy_portable_inflate_rejects_from_torch_import(tmp_path: Path) -> None:
    inflate = tmp_path / "inflate.py"
    inflate.write_text("from torch.nn import functional as F\n", encoding="utf-8")
    with pytest.raises(MlxScoreAwareHarnessError, match="forbidden non-portable"):
        assert_numpy_portable_inflate(inflate)


def test_assert_numpy_portable_inflate_rejects_dotted_mlx_submodule(tmp_path: Path) -> None:
    inflate = tmp_path / "inflate.py"
    inflate.write_text("from mlx.utils import tree_flatten\n", encoding="utf-8")
    with pytest.raises(MlxScoreAwareHarnessError, match="forbidden non-portable"):
        assert_numpy_portable_inflate(inflate)


def test_assert_numpy_portable_inflate_allows_mlx_in_comment(tmp_path: Path) -> None:
    inflate = tmp_path / "inflate.py"
    inflate.write_text(
        "import numpy as np\n"
        "# this decodes MLX-trained weights but imports no mlx / torch\n"
        '"""docstring mentions torch and mlx but imports neither."""\n',
        encoding="utf-8",
    )
    assert assert_numpy_portable_inflate(inflate)["numpy_portable"] is True


def test_assert_numpy_portable_inflate_allows_relative_import(tmp_path: Path) -> None:
    inflate = tmp_path / "inflate.py"
    inflate.write_text(
        "import numpy as np\nfrom . import archive\n", encoding="utf-8"
    )
    assert assert_numpy_portable_inflate(inflate)["numpy_portable"] is True


def test_assert_numpy_portable_inflate_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="not found"):
        assert_numpy_portable_inflate(tmp_path / "does_not_exist.py")


def test_forbidden_roots_are_mlx_and_torch() -> None:
    assert set(FORBIDDEN_INFLATE_IMPORT_ROOTS) == {"mlx", "torch"}


def test_mlx_evidence_grade_is_macos_mlx_research_signal() -> None:
    assert MLX_EVIDENCE_GRADE == "[macOS-MLX research-signal]"


# ---------------------------------------------------------------------------
# RendererBundle contract validation (no MLX arrays needed for these)
# ---------------------------------------------------------------------------


def test_renderer_bundle_rejects_bad_convention() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="forward_convention"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            forward_convention="not_a_real_convention",
        )


def test_renderer_bundle_rejects_zero_pairs() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="num_pairs"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=0,
        )


def test_renderer_bundle_rejects_negative_distillation_weight() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="distillation_weight"):
        RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            distillation_weight=-0.1,
        )


def test_renderer_bundle_accepts_canonical_conventions() -> None:
    for conv in ("reconstruct_pair_nchw01", "call_b2chw_255"):
        b = RendererBundle(
            model=object(),
            target_rgb_0=None,
            target_rgb_1=None,
            num_pairs=4,
            forward_convention=conv,
        )
        assert b.forward_convention == conv


# ---------------------------------------------------------------------------
# MLX-bound: score-aware loss + adapter + end-to-end training
# ---------------------------------------------------------------------------


def _tiny_dreamer_bundle(num_pairs: int = 4, distill: float = 0.5):
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
    # The dreamer/HNeRV renderer hardcodes 384x512 output.
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


@mlx_only
def test_score_aware_loss_is_finite_and_decomposed() -> None:
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware_full_main import score_aware_loss

    bundle = _tiny_dreamer_bundle(distill=0.5)
    idx = mx.array([0, 1], dtype=mx.int32)
    total, parts = score_aware_loss(bundle, idx)
    mx.eval(total)
    assert float(total.item()) == float(total.item())  # not NaN
    assert "recon" in parts
    assert "distill" in parts  # distillation_weight > 0
    assert "total" in parts


@mlx_only
def test_score_aware_loss_no_distill_when_weight_zero() -> None:
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware_full_main import score_aware_loss

    bundle = _tiny_dreamer_bundle(distill=0.0)
    idx = mx.array([0, 1], dtype=mx.int32)
    _total, parts = score_aware_loss(bundle, idx)
    assert "distill" not in parts
    assert "recon" in parts


@mlx_only
def test_adapter_train_step_reduces_loss_over_steps() -> None:
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware_full_main import (
        MlxScoreAwareAdapter,
    )

    bundle = _tiny_dreamer_bundle(distill=0.0)
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    batch = mx.array([0, 1, 2, 3], dtype=mx.int32)
    losses = []
    for _ in range(20):
        out = adapter.train_step(batch, learning_rate=1e-2, loss_weights={})
        losses.append(out["total"])
    # Training should reduce the loss (gradient flows into renderer params).
    assert losses[-1] < losses[0]


@mlx_only
def test_adapter_satisfies_protocol() -> None:
    from tac.substrates._shared.mlx_score_aware_full_main import (
        MlxScoreAwareAdapter,
    )
    from tac.training.long_training_canonical import validate_substrate_adapter

    bundle = _tiny_dreamer_bundle()
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    # Should not raise (Protocol conformance: has all required methods).
    validate_substrate_adapter(adapter)


@mlx_only
def test_adapter_optimizer_step_raises_style_a_stub() -> None:
    from tac.substrates._shared.mlx_score_aware_full_main import (
        MlxScoreAwareAdapter,
    )

    bundle = _tiny_dreamer_bundle()
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    with pytest.raises(NotImplementedError, match="Style B train_step"):
        adapter.optimizer_step(adapter.model, None, 1e-3)


@mlx_only
def test_adapter_export_state_dict_writes_portable_npz(tmp_path: Path) -> None:
    from tac.substrates._shared.mlx_score_aware_full_main import (
        MlxScoreAwareAdapter,
    )

    bundle = _tiny_dreamer_bundle()
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    target = tmp_path / "ckpt.state"
    adapter.export_state_dict(adapter.model, target)
    npz = target.with_suffix(target.suffix + ".mlx.npz")
    assert npz.is_file()


@mlx_only
def test_adapter_score_aware_components_defers_to_pytorch_sister() -> None:
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware_full_main import (
        MlxScoreAwareAdapter,
    )

    bundle = _tiny_dreamer_bundle()
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")
    assert adapter.score_aware_components(adapter.model, mx.array([0])) is None


@mlx_only
def test_run_mlx_score_aware_full_main_end_to_end(tmp_path: Path) -> None:
    from tac.substrates._shared.mlx_score_aware_full_main import (
        run_mlx_score_aware_full_main,
    )

    bundle = _tiny_dreamer_bundle(num_pairs=4, distill=0.5)
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="dreamer_v3_rssm",
        lane_id="lane_mlx_score_aware_harness_plus_6_substrate_unlock_20260527",
        output_dir=tmp_path / "run",
        epochs=3,
        batch_pair_indices_per_step=2,
        learning_rate=1e-3,
        seed=0,
        notes=(
            "harness end-to-end test: dreamer renderer + synthetic-zero "
            "targets + Hinton-KL scorer surrogate via canonical "
            "run_long_training; non-promotable research signal"
        ),
    )
    assert artifact.total_epochs_completed == 3
    # Non-promotable by construction (Catalog #192/#317/#341).
    assert artifact.promotable is False
    d = artifact.as_dict()
    assert d.get("score_claim") is False
    assert d.get("promotion_eligible") is False


@mlx_only
def test_run_mlx_score_aware_full_main_verifies_inflate_portability(
    tmp_path: Path,
) -> None:
    from tac.substrates._shared.mlx_score_aware_full_main import (
        run_mlx_score_aware_full_main,
    )

    # A non-portable inflate.py must fail closed BEFORE training.
    bad_inflate = tmp_path / "inflate.py"
    bad_inflate.write_text("import torch\n", encoding="utf-8")
    bundle = _tiny_dreamer_bundle(num_pairs=4, distill=0.0)
    with pytest.raises(MlxScoreAwareHarnessError, match="forbidden non-portable"):
        run_mlx_score_aware_full_main(
            bundle=bundle,
            substrate_id="dreamer_v3_rssm",
            lane_id="lane_mlx_score_aware_harness_plus_6_substrate_unlock_20260527",
            output_dir=tmp_path / "run",
            epochs=1,
            batch_pair_indices_per_step=2,
            inflate_py_path=bad_inflate,
            notes="inflate portability fail-closed test before training",
        )
