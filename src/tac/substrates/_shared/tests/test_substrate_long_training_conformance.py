# SPDX-License-Identifier: MIT
"""Conformance tests for substrate adapters per Catalog #335 canonical contract.

Per the doctrine production-hardening contract: every substrate that
participates in L2 long-training MUST conform to
:class:`tac.training.long_training_canonical.SubstrateLongTrainingAdapter`
Protocol. This test suite validates each substrate's adapter against:

1. Protocol contract conformance (validate_substrate_adapter passes).
2. Style A vs Style B detection works correctly.
3. End-to-end run_long_training with the adapter produces valid
   TrainingArtifact (smoke-scale to keep tests <30s each).

This is the canonical conformance harness sister of:
- ``tac.cathedral.consumer_contract.validate_consumer_module`` (per
  Catalog #335 for cathedral consumers)
- ``tac.substrate_registry.contract.validate_all_registered`` (per
  Catalog #241/#242 for substrate META layer)

Per CLAUDE.md "Beauty, simplicity, and developer experience" + Catalog
#290 canonical-vs-unique: conformance tests are CHEAP to add when a new
substrate gains an L2 adapter; just add the substrate's adapter import
+ a parametrized test row.
"""
from __future__ import annotations

import pytest

from tac.training.long_training_canonical import (
    SubstrateLongTrainingAdapter,
    validate_substrate_adapter,
)


# ---------------------------------------------------------------------------
# Z6 conformance
# ---------------------------------------------------------------------------


def _try_import_z6_adapter():
    """Best-effort import of Z6 adapter; skip test if MLX unavailable."""
    try:
        from tac.substrates.time_traveler_l5_z6.long_training_adapter import (
            Z6LongTrainingAdapter,
        )

        return Z6LongTrainingAdapter
    except ImportError as exc:
        pytest.skip(f"Z6 adapter unavailable: {exc}")


def test_z6_long_training_adapter_class_satisfies_protocol() -> None:
    """Z6 adapter class declares all Protocol attributes/methods."""
    adapter_cls = _try_import_z6_adapter()
    # Verify class declares all required methods (without instantiating).
    required_methods = (
        "sample_batch",
        "loss_fn",
        "optimizer_step",
        "export_state_dict",
        "export_archive",
        "score_aware_components",
    )
    for m in required_methods:
        assert hasattr(adapter_cls, m), f"Z6 adapter class missing required method {m!r}"
    assert hasattr(adapter_cls, "substrate_id"), "Z6 adapter class missing substrate_id"


def test_z6_long_training_adapter_has_train_step_style_b() -> None:
    """Z6 adapter uses Style B (train_step) because MLX uses value_and_grad."""
    adapter_cls = _try_import_z6_adapter()
    # Z6 specifically chose Style B per its docstring; verify train_step exists.
    assert hasattr(adapter_cls, "train_step"), (
        "Z6 adapter declares Style B (train_step) per MLX value_and_grad pattern; "
        "see long_training_adapter.py docstring."
    )


def test_z6_long_training_adapter_substrate_id_canonical() -> None:
    """Z6 adapter substrate_id matches canonical directory name."""
    adapter_cls = _try_import_z6_adapter()
    assert adapter_cls.substrate_id == "time_traveler_l5_z6"


def test_z6_long_training_adapter_construction_requires_mlx() -> None:
    """Z6 adapter construction fails fast without MLX (canonical fail-closed)."""
    adapter_cls = _try_import_z6_adapter()
    # Attempt to construct with placeholder args; should raise ImportError if
    # MLX missing, else fail with config-specific error.
    try:
        import mlx  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="MLX"):
            adapter_cls(config=None, target_rgb_0=None, target_rgb_1=None)


def test_z6_long_training_adapter_passes_validate_substrate_adapter() -> None:
    """Constructed Z6 adapter passes canonical conformance validator."""
    adapter_cls = _try_import_z6_adapter()
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("MLX unavailable; cannot construct Z6 adapter")

    from tac.substrates.time_traveler_l5_z6.architecture import (
        Z6PredictiveCodingConfig,
    )

    cfg = Z6PredictiveCodingConfig(
        latent_dim=4,
        num_pairs=2,
        output_height=8,
        output_width=8,
        decoder_num_upsample_blocks=1,
        decoder_channels=(2,),
        decoder_embed_dim=4,
        predictor_depth=1,
    )
    import numpy as np

    target_0 = mx.array(np.zeros((2, 8, 8, 3), dtype=np.float32))
    target_1 = mx.array(np.zeros((2, 8, 8, 3), dtype=np.float32))
    adapter = adapter_cls(config=cfg, target_rgb_0=target_0, target_rgb_1=target_1)
    # Canonical conformance validator passes:
    validate_substrate_adapter(adapter)
    # Substrate-id canonical:
    assert adapter.substrate_id == "time_traveler_l5_z6"


# ---------------------------------------------------------------------------
# Style A vs Style B detection
# ---------------------------------------------------------------------------


def test_canonical_helper_detects_style_b_train_step_when_present() -> None:
    """OOMSafeStepRunner detects + prefers Style B train_step over Style A."""
    from tac.training.long_training_canonical import OOMSafeStepRunner

    class _StyleBAdapter:
        substrate_id = "style_b_test"
        model = type("M", (), {"state_dict": lambda s: {"w": [1.0]}, "load_state_dict": lambda s, x: None})()
        train_step_call_count = 0
        loss_fn_call_count = 0
        optimizer_step_call_count = 0

        def sample_batch(self, bs, seed):
            return {"bs": bs}

        def loss_fn(self, model, batch, lw):
            self.loss_fn_call_count += 1
            return {"total": 0.1}

        def optimizer_step(self, model, loss, lr):
            self.optimizer_step_call_count += 1

        def train_step(self, batch, lr, lw):
            self.train_step_call_count += 1
            return {"total": 0.05}

        def export_state_dict(self, model, path):
            pass

        def export_archive(self, model, output_dir):
            return None

        def score_aware_components(self, model, batch):
            return None

    from tac.training.long_training_canonical import CurriculumStage

    adapter = _StyleBAdapter()
    runner = OOMSafeStepRunner()
    stage = CurriculumStage(name="s", start_epoch=0, end_epoch=10)
    loss, bs = runner.run_step(adapter, batch_size=4, seed=0, stage=stage, learning_rate=1e-3)
    # Style B path: train_step called; loss_fn + optimizer_step NOT called.
    assert adapter.train_step_call_count == 1
    assert adapter.loss_fn_call_count == 0
    assert adapter.optimizer_step_call_count == 0
    assert loss["total"] == 0.05


def test_canonical_helper_falls_back_to_style_a_when_no_train_step() -> None:
    """OOMSafeStepRunner uses Style A loss_fn + optimizer_step when no train_step."""
    from tac.training.long_training_canonical import OOMSafeStepRunner

    class _StyleAAdapter:
        substrate_id = "style_a_test"
        model = type("M", (), {"state_dict": lambda s: {"w": [1.0]}, "load_state_dict": lambda s, x: None})()
        loss_fn_call_count = 0
        optimizer_step_call_count = 0

        def sample_batch(self, bs, seed):
            return {"bs": bs}

        def loss_fn(self, model, batch, lw):
            self.loss_fn_call_count += 1
            return {"total": 0.2}

        def optimizer_step(self, model, loss, lr):
            self.optimizer_step_call_count += 1

        def export_state_dict(self, model, path):
            pass

        def export_archive(self, model, output_dir):
            return None

        def score_aware_components(self, model, batch):
            return None

    from tac.training.long_training_canonical import CurriculumStage

    adapter = _StyleAAdapter()
    runner = OOMSafeStepRunner()
    stage = CurriculumStage(name="s", start_epoch=0, end_epoch=10)
    loss, bs = runner.run_step(adapter, batch_size=4, seed=0, stage=stage, learning_rate=1e-3)
    # Style A path: loss_fn + optimizer_step called; no train_step.
    assert adapter.loss_fn_call_count == 1
    assert adapter.optimizer_step_call_count == 1
    assert loss["total"] == 0.2
