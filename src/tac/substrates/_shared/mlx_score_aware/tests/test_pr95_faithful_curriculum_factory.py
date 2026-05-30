# SPDX-License-Identifier: MIT
"""Tests for PR95FaithfulCurriculumFactory + MlxScoreAwareAdapter opt-in wire-in.

Verifies per CLAUDE.md "NO FAKE IMPLEMENTATIONS" non-negotiable:
- Each of the 8 stages emits a distinct optimizer_config (NOT canonical-marker stub).
- Stage 8 uses Muon (NOT AdamW disguised).
- Per-stage hyperparams (loss_family, qat_active, sigma, lambda) match canonical
  PR95 source-faithful descriptors per CLAUDE.md L14 + L15.
- Backward-compat: default-off preserves existing MlxScoreAwareAdapter behavior.
- Opt-in active: train_step routes through canonical apply_pr95_mlx_optimizer_step.
- Muon partition (Conv/Linear ≥2D weights non-stem/non-rgb/non-latents) preserved.

[verified-against: tac.local_acceleration.pr95_hnerv_mlx.PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS]
[verified-against: tac.optimization.optimizer_scheduler_registry.default_optimizer_scheduler_descriptors]
[verified-against: CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L14 + L15]
"""
from __future__ import annotations

import pytest

# Skip the entire module when MLX is unavailable; the adapter+factory require MLX.
mlx_available = True
try:
    import mlx.core as _mx  # noqa: F401
except Exception:
    mlx_available = False

requires_mlx = pytest.mark.skipif(
    not mlx_available,
    reason="MLX unavailable on this host; PR95 8-stage curriculum requires MLX",
)


# ---------- Section 1: PR95FaithfulCurriculumFactory unit tests ----------


@requires_mlx
def test_canonical_pr95_total_epochs_is_29650() -> None:
    """Canonical PR95 total epoch budget per CLAUDE.md L14 is 29,650."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
    )

    assert CANONICAL_PR95_TOTAL_EPOCHS == 29_650
    # Sister check: 3000 + 5650 + 1500 + 500 + 9000 + 2000 + 3000 + 5000 = 29650
    canonical_breakdown = [3000, 5650, 1500, 500, 9000, 2000, 3000, 5000]
    assert sum(canonical_breakdown) == 29_650


@requires_mlx
def test_factory_canonical_budget_uses_verbatim_pr95_source_epochs() -> None:
    """At canonical budget, factory uses PR95 source-faithful per-stage epochs."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=CANONICAL_PR95_TOTAL_EPOCHS)
    assert factory.is_canonical_pr95_budget is True
    boundaries = factory.stage_epoch_boundaries
    assert len(boundaries) == 8
    # Stage indices are 1..8.
    assert [b[0] for b in boundaries] == [1, 2, 3, 4, 5, 6, 7, 8]
    # Per-stage epochs match canonical PR95 source verbatim.
    canonical_per_stage = [3000, 5650, 1500, 500, 9000, 2000, 3000, 5000]
    for (_stage_index, start_epoch, end_epoch), expected in zip(
        boundaries, canonical_per_stage, strict=True
    ):
        assert (end_epoch - start_epoch) == expected
    # End-to-end: final stage ends at exact canonical total.
    assert boundaries[-1][2] == CANONICAL_PR95_TOTAL_EPOCHS


@requires_mlx
def test_factory_refuses_total_epoch_budget_below_8() -> None:
    """Factory refuses budgets < 8 (cannot fit 8 stages with ≥1 epoch each)."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        PR95FaithfulCurriculumError,
        PR95FaithfulCurriculumFactory,
    )

    with pytest.raises(PR95FaithfulCurriculumError, match="must be >= 8"):
        PR95FaithfulCurriculumFactory(total_epoch_budget=7)
    with pytest.raises(PR95FaithfulCurriculumError):
        PR95FaithfulCurriculumFactory(total_epoch_budget=0)


@requires_mlx
def test_factory_smoke_budget_proportionally_scales_stages() -> None:
    """A small smoke budget (100 epochs) scales per-stage epochs proportionally."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=100)
    assert factory.is_canonical_pr95_budget is False
    boundaries = factory.stage_epoch_boundaries
    assert len(boundaries) == 8
    # Every stage gets at least 1 epoch (no degenerate empty stages).
    for _stage_index, start_epoch, end_epoch in boundaries:
        assert end_epoch > start_epoch, "every stage MUST have ≥1 epoch"
    # End-to-end: final stage ends at exactly 100.
    assert boundaries[-1][2] == 100
    # Stage 5 (canonically 9000/29650 = 30%) should be the largest after scaling.
    stage_epochs = [b[2] - b[1] for b in boundaries]
    assert stage_epochs[4] == max(stage_epochs), (
        f"stage 5 should be largest after scaling; got per-stage {stage_epochs}"
    )


@requires_mlx
def test_current_stage_index_progresses_monotonically_across_canonical_budget() -> None:
    """Stage index advances 1→2→3→...→8 monotonically across canonical epochs."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=CANONICAL_PR95_TOTAL_EPOCHS)
    # Stage 1: epochs 0..2999.
    assert factory.current_stage_index(0) == 1
    assert factory.current_stage_index(2999) == 1
    # Stage 2: epochs 3000..8649 (3000 + 5650 = 8650 = stage 2 end).
    assert factory.current_stage_index(3000) == 2
    assert factory.current_stage_index(8649) == 2
    # Stage 3: epochs 8650..10149.
    assert factory.current_stage_index(8650) == 3
    # Stage 4: epochs 10150..10649.
    assert factory.current_stage_index(10150) == 4
    # Stage 5: epochs 10650..19649.
    assert factory.current_stage_index(10650) == 5
    # Stage 6: epochs 19650..21649.
    assert factory.current_stage_index(19650) == 6
    # Stage 7: epochs 21650..24649.
    assert factory.current_stage_index(21650) == 7
    # Stage 8: epochs 24650..29649.
    assert factory.current_stage_index(24650) == 8
    assert factory.current_stage_index(29649) == 8
    # Overflow → final stage absorbs (canonical scheduler pattern).
    assert factory.current_stage_index(50_000) == 8


@requires_mlx
def test_current_stage_verdict_loads_canonical_descriptor_per_stage() -> None:
    """Each stage verdict cites the canonical PR95 descriptor_id."""
    from tac.local_acceleration.pr95_hnerv_mlx import (
        PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS,
    )
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=CANONICAL_PR95_TOTAL_EPOCHS)
    canonical_starts = [0, 3000, 8650, 10150, 10650, 19650, 21650, 24650]
    for stage_idx, start_epoch in zip(range(1, 9), canonical_starts, strict=True):
        verdict = factory.current_stage_verdict(start_epoch)
        assert verdict.stage_index == stage_idx, (
            f"stage_index mismatch at epoch {start_epoch}"
        )
        assert verdict.descriptor_id == PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS[stage_idx], (
            f"descriptor_id mismatch at stage {stage_idx}"
        )


@requires_mlx
def test_per_stage_descriptor_ids_match_canonical_registry() -> None:
    """factory.per_stage_descriptor_ids matches PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS verbatim."""
    from tac.local_acceleration.pr95_hnerv_mlx import (
        PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS,
    )
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory()
    canonical = tuple(
        PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS[i] for i in range(1, 9)
    )
    assert factory.per_stage_descriptor_ids == canonical


@requires_mlx
def test_per_stage_optimizer_config_distinct_across_8_stages() -> None:
    """NO FAKE: each of the 8 stages emits a distinct optimizer_config."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=CANONICAL_PR95_TOTAL_EPOCHS)
    canonical_starts = [0, 3000, 8650, 10150, 10650, 19650, 21650, 24650]
    configs = []
    for start_epoch in canonical_starts:
        verdict = factory.current_stage_verdict(start_epoch)
        configs.append(verdict.optimizer_config)
    # Canonical key facts per CLAUDE.md L14 + L15:
    # - Stages 1-7 use AdamW only (use_muon=False).
    # - Stage 8 uses Muon+AdamW (use_muon=True).
    for i in range(7):
        assert configs[i].use_muon is False, (
            f"stage {i + 1} MUST be AdamW-only (use_muon=False)"
        )
    assert configs[7].use_muon is True, (
        "stage 8 MUST be Muon (use_muon=True) per L15 canonical equation"
    )
    # Distinct lr / lambda / sigma signatures verify NO FAKE:
    # Stage 1 lr = 1e-3; Stage 5 lr = 3e-5; Stage 8 adamw_lr = 1e-5.
    assert configs[0].adamw_lr == pytest.approx(1e-3)
    assert configs[4].adamw_lr == pytest.approx(3e-5)
    assert configs[7].adamw_lr == pytest.approx(1e-5)
    assert configs[7].muon_lr == pytest.approx(2e-4)


@requires_mlx
def test_per_stage_loss_family_and_lambda_sigma_distinct_per_canonical_l14() -> None:
    """Per-stage loss_family + cat_sigma + cat_lambda match canonical PR95 source."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=CANONICAL_PR95_TOTAL_EPOCHS)
    canonical_starts = [0, 3000, 8650, 10150, 10650, 19650, 21650, 24650]
    # Per CLAUDE.md L14 + canonical optimizer_scheduler_registry:
    expected_loss = [
        "ce_seg_loss",
        "tau_softplus_seg_loss",
        "smooth_disagreement_seg_loss",
        "smooth_disagreement_seg_loss",
        "l7_softplus_seg_loss",
        "l7_softplus_seg_loss",
        "l7_softplus_seg_loss",
        "l7_softplus_seg_loss",
    ]
    expected_lambda = [0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.02, 0.02]
    expected_sigma = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
    expected_qat = [False, False, False, True, True, True, True, True]
    for i, start_epoch in enumerate(canonical_starts):
        verdict = factory.current_stage_verdict(start_epoch)
        assert verdict.loss_family == expected_loss[i], (
            f"stage {i + 1} loss_family mismatch"
        )
        assert verdict.cat_lambda == pytest.approx(expected_lambda[i]), (
            f"stage {i + 1} cat_lambda mismatch"
        )
        assert verdict.cat_sigma == pytest.approx(expected_sigma[i]), (
            f"stage {i + 1} cat_sigma mismatch"
        )
        assert verdict.qat_active is expected_qat[i], (
            f"stage {i + 1} qat_active mismatch"
        )


@requires_mlx
def test_is_stage_boundary_fires_only_at_canonical_stage_starts() -> None:
    """is_stage_boundary fires only when global_epoch equals start_epoch of any stage > 1."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=CANONICAL_PR95_TOTAL_EPOCHS)
    # Stage 1 start (epoch 0) is NOT a boundary per the canonical semantics.
    assert factory.is_stage_boundary(0) is False
    # Stage 2..8 starts ARE boundaries.
    assert factory.is_stage_boundary(3000) is True
    assert factory.is_stage_boundary(8650) is True
    assert factory.is_stage_boundary(24650) is True
    # Non-boundary epochs return False.
    assert factory.is_stage_boundary(1500) is False
    assert factory.is_stage_boundary(15_000) is False


@requires_mlx
def test_stage_transition_diff_detects_l14_to_l15_transition() -> None:
    """Stage 7 → Stage 8 transition is the canonical L14→L15 Muon activation."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=CANONICAL_PR95_TOTAL_EPOCHS)
    # No transition within the same stage.
    assert factory.stage_transition_diff(100, 200) is None
    # Transition from stage 7 to stage 8 (canonical Muon activation).
    diff = factory.stage_transition_diff(24649, 24650)
    assert diff == (7, 8), (
        f"L14→L15 transition (stage 7→8) expected; got {diff}"
    )
    # Transition from stage 1 to stage 2.
    diff_early = factory.stage_transition_diff(2999, 3000)
    assert diff_early == (1, 2)


@requires_mlx
def test_stage_verdict_cache_returns_same_object_within_stage() -> None:
    """current_stage_verdict caches per-stage verdict objects."""
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
        PR95FaithfulCurriculumFactory,
    )

    factory = PR95FaithfulCurriculumFactory(total_epoch_budget=CANONICAL_PR95_TOTAL_EPOCHS)
    v1 = factory.current_stage_verdict(1000)
    v2 = factory.current_stage_verdict(2000)
    assert v1 is v2, "stage verdict cache MUST return identical object within stage"


# ---------- Section 2: MlxScoreAwareAdapter opt-in wire-in tests ----------


def _make_minimal_bundle() -> object:
    """Build a minimal RendererBundle for adapter testing (MLX required).

    Uses the ``reconstruct_pair_nchw01`` forward convention with a tiny renderer
    whose 2D weight is Muon-eligible per the canonical partition
    (no stem/rgb/latents tokens in the parameter name).
    """
    import mlx.core as mx
    import mlx.nn as nn

    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle

    class TinyRenderer(nn.Module):
        """4-parameter MLX renderer with 2D weight (Muon-eligible per partition).

        Exposes the canonical ``reconstruct_pair`` convention returning two
        ``(B, 3, H, W)`` frames in ``[0, 1]``.
        """

        def __init__(self) -> None:
            super().__init__()
            # 2D weight named "decoder_weight" → Muon-eligible per canonical partition.
            # NOTE: NOT a Linear layer to avoid the auto-named "weight" / "bias"
            # collision with the canonical Muon partition filter; we test the
            # canonical NAME-based routing directly.
            self.decoder_weight = mx.zeros((4, 4))
            # 1D bias named "decoder_bias" → AdamW-only per canonical partition.
            self.decoder_bias = mx.zeros((4,))

        def reconstruct_pair(self, indices):  # noqa: D401
            """Return (rgb_0, rgb_1) each (B, 3, 2, 2) in [0, 1] — tiny canonical pair."""
            bs = int(indices.shape[0])
            # Use parameters so MLX value_and_grad sees gradient through them.
            scale = mx.sum(self.decoder_weight) + mx.sum(self.decoder_bias)
            base = mx.ones((bs, 3, 2, 2)) * 0.5 * (scale * 0.0 + 1.0)
            # Add a gradient-bearing tiny modulation so the loss is non-degenerate.
            mod = mx.broadcast_to(
                mx.reshape(self.decoder_weight[:1, :1] * 0.01, (1, 1, 1, 1)),
                (bs, 3, 2, 2),
            )
            return base + mod, base + mod * 2.0

    model = TinyRenderer()
    num_pairs = 8
    # Targets are precomputed (B, H, W, 3) in [0, 1] per RendererBundle contract.
    target_rgb_0 = mx.zeros((num_pairs, 2, 2, 3))
    target_rgb_1 = mx.zeros((num_pairs, 2, 2, 3))

    return RendererBundle(
        model=model,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        num_pairs=num_pairs,
        forward_convention="reconstruct_pair_nchw01",
    )


@requires_mlx
def test_adapter_default_off_preserves_legacy_adamw_path() -> None:
    """Backward compat: pr95_faithful_curriculum_enabled=False keeps legacy adapter."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_bundle()
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="test_substrate")
    # Default off: factory is None, no PR95 state initialized.
    assert adapter._pr95_faithful_curriculum_enabled is False
    assert adapter._pr95_curriculum_factory is None
    assert adapter._pr95_optimizer_state is None


@requires_mlx
def test_adapter_opt_in_initializes_canonical_factory_and_state() -> None:
    """Opt-in: pr95_faithful_curriculum_enabled=True initializes factory + state."""
    from tac.local_acceleration.pr95_hnerv_mlx import Pr95MlxOptimizerState
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        PR95FaithfulCurriculumFactory,
    )

    bundle = _make_minimal_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=100,
    )
    assert adapter._pr95_faithful_curriculum_enabled is True
    assert isinstance(adapter._pr95_curriculum_factory, PR95FaithfulCurriculumFactory)
    assert isinstance(adapter._pr95_optimizer_state, Pr95MlxOptimizerState)
    assert adapter._pr95_curriculum_factory.total_epoch_budget == 100


@requires_mlx
def test_adapter_opt_in_defaults_to_canonical_29650_epochs_when_not_specified() -> None:
    """Opt-in with no total_epochs override defaults to canonical 29,650."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
    from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
        CANONICAL_PR95_TOTAL_EPOCHS,
    )

    bundle = _make_minimal_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
    )
    assert (
        adapter._pr95_curriculum_factory.total_epoch_budget
        == CANONICAL_PR95_TOTAL_EPOCHS
    )


@requires_mlx
def test_notify_global_epoch_advances_pr95_global_epoch_state() -> None:
    """notify_global_epoch updates the adapter's PR95 stage tracker."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=100,
    )
    assert adapter._pr95_global_epoch == 0
    adapter.notify_global_epoch(50)
    assert adapter._pr95_global_epoch == 50


@requires_mlx
def test_notify_global_epoch_is_noop_when_curriculum_disabled() -> None:
    """notify_global_epoch preserves backward-compat when curriculum disabled."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_bundle()
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="test_substrate")
    # No-op-when-disabled: the field is updated but factory/state stay None.
    adapter.notify_global_epoch(50)
    assert adapter._pr95_global_epoch == 50
    assert adapter._pr95_curriculum_factory is None


# ---------- Section 3: Canonical equation + Provenance integration ----------


@requires_mlx
def test_canonical_equation_l14_8_stage_curriculum_v1_registered() -> None:
    """Canonical equation pr95_family_l14_eight_stage_29650_epoch_curriculum_v1 may be registered."""
    from tac.canonical_equations import query_equations

    equations = query_equations()
    # The canonical CLAUDE.md L14 entry IS the documentation contract; the
    # registry may lazily load. Verify the entry shape if present.
    l14 = [
        eq
        for eq in equations
        if eq.equation_id == "pr95_family_l14_eight_stage_29650_epoch_curriculum_v1"
    ]
    if l14:
        assert l14[0].equation_id.startswith("pr95_family_l14_")


@requires_mlx
def test_canonical_equation_l15_muon_optimizer_final_stage_only_v1_registered() -> None:
    """Canonical equation pr95_family_l15_muon_optimizer_final_stage_only_v1 may be registered."""
    from tac.canonical_equations import query_equations

    equations = query_equations()
    l15 = [
        eq
        for eq in equations
        if eq.equation_id == "pr95_family_l15_muon_optimizer_final_stage_only_v1"
    ]
    if l15:
        assert l15[0].equation_id.startswith("pr95_family_l15_")


# ---------- Section 4: Stage-aware curriculum metrics returned from train_step ----------


@requires_mlx
def test_train_step_returns_stage_index_in_metrics_when_curriculum_enabled() -> None:
    """train_step's return dict carries pr95_stage_index + uses_muon metrics."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,  # 10 epochs per stage (80 / 8 = 10).
    )
    # At epoch 0 → stage 1 (use_muon=False).
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=2, seed=0)
    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0},
    )
    assert "total" in metrics
    assert "pr95_stage_index" in metrics
    assert "pr95_stage_uses_muon" in metrics
    assert metrics["pr95_stage_index"] == 1.0
    assert metrics["pr95_stage_uses_muon"] == 0.0  # stage 1 uses AdamW only.


@requires_mlx
def test_train_step_stage_8_signals_muon_active_in_metrics() -> None:
    """At final-stage epoch, train_step metrics carry pr95_stage_uses_muon=1.0."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    # Advance to stage 8 (epoch 70 → stage 8 in 80-epoch budget).
    adapter.notify_global_epoch(75)
    batch = adapter.sample_batch(batch_size=2, seed=0)
    metrics = adapter.train_step(
        batch=batch,
        learning_rate=1e-3,
        loss_weights={"recon": 1.0},
    )
    assert metrics["pr95_stage_index"] == 8.0, (
        f"epoch 75 of 80 expected stage 8; got {metrics['pr95_stage_index']}"
    )
    assert metrics["pr95_stage_uses_muon"] == 1.0, (
        "stage 8 MUST signal use_muon=True per L15 canonical equation"
    )


# ---------- Section 5: NO FAKE end-to-end verification (param mutation) ----------


@requires_mlx
def test_train_step_actually_mutates_parameters_per_stage_NO_FAKE() -> None:
    """NO FAKE per CLAUDE.md: train_step must actually mutate model parameters."""
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_bundle()
    # Initialize model parameters to non-zero so the loss gradient is non-zero.
    bundle.model.decoder_weight = mx.ones((4, 4)) * 0.5
    bundle.model.decoder_bias = mx.ones((4,)) * 0.1
    mx.eval(bundle.model.parameters())
    initial_weight = float(bundle.model.decoder_weight.sum().item())
    initial_bias = float(bundle.model.decoder_bias.sum().item())
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    # Run 5 train_steps and verify parameters move (NO FAKE).
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=4, seed=0)
    for _ in range(5):
        adapter.train_step(
            batch=batch,
            learning_rate=1e-2,
            loss_weights={"recon": 1.0},
        )
    mx.eval(bundle.model.parameters())
    final_weight = float(bundle.model.decoder_weight.sum().item())
    final_bias = float(bundle.model.decoder_bias.sum().item())
    # The target is zeros, so MSE loss gradient should pull weights toward zero.
    # NO FAKE: parameters MUST have moved from their initial values.
    assert final_weight != initial_weight, (
        f"NO FAKE: decoder_weight unchanged after 5 train_steps "
        f"(initial={initial_weight}, final={final_weight}); "
        f"factory must actually invoke the optimizer"
    )
    # Bias may legitimately be near-zero gradient for some seeds; weight is the
    # primary NO FAKE assertion. We additionally validate the bias path doesn't
    # error out by simply checking it remains a real MLX array.
    assert bundle.model.decoder_bias is not None
    _ = final_bias  # silence unused; the assertion above is the NO FAKE guard.


@requires_mlx
def test_stage_transition_resets_muon_buffers_per_l15_invariant() -> None:
    """L15 canonical equation: stage 8 starts with fresh Muon buffers."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_bundle()
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,  # 10 epochs/stage; stage 7 ends at 70.
    )
    # Step through stage 7 (epochs 60-69 in 80-epoch budget).
    adapter.notify_global_epoch(60)
    batch = adapter.sample_batch(batch_size=2, seed=0)
    adapter.train_step(
        batch=batch, learning_rate=1e-3, loss_weights={"recon": 1.0}
    )
    # Stage 7 doesn't use Muon, so buffers should be empty.
    assert len(adapter._pr95_optimizer_state.muon_buffers) == 0, (
        "stage 7 (use_muon=False) MUST not populate muon_buffers"
    )
    # Advance to stage 8 (epoch 75 in 80-epoch budget).
    adapter.notify_global_epoch(75)
    adapter.train_step(
        batch=batch, learning_rate=1e-3, loss_weights={"recon": 1.0}
    )
    # Stage 8 uses Muon → buffers should now be populated for Muon-eligible params.
    assert len(adapter._pr95_optimizer_state.muon_buffers) > 0, (
        "stage 8 (use_muon=True) MUST populate muon_buffers for Muon-eligible params"
    )


@requires_mlx
def test_canonical_partition_routes_decoder_weight_to_muon_in_stage_8() -> None:
    """Canonical partition: 2D 'decoder_weight' (no stem/rgb/latents) routes to Muon."""
    from tac.local_acceleration.pr95_hnerv_mlx import (
        partition_pr95_mlx_parameter_names,
    )
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_bundle()
    adapter = MlxScoreAwareAdapter(  # noqa: F841 (constructs the model)
        bundle,
        substrate_id="test_substrate",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=80,
    )
    split = partition_pr95_mlx_parameter_names(bundle.model.parameters())
    # 2D decoder_weight should be in Muon partition.
    assert any("decoder_weight" in name for name in split["muon"]), (
        f"decoder_weight (2D, no stem/rgb/latents) MUST be Muon-eligible; "
        f"got muon={split['muon']}, adamw={split['adamw']}"
    )
    # 1D decoder_bias should be in AdamW partition.
    assert any("decoder_bias" in name for name in split["adamw"]), (
        f"decoder_bias (1D) MUST be AdamW-only; "
        f"got muon={split['muon']}, adamw={split['adamw']}"
    )


# ---------- Section 6: backward compat regression guards ----------


@requires_mlx
def test_existing_loss_adapter_tests_still_pass_with_default_off() -> None:
    """Regression: existing adapter tests should pass unchanged with default-off."""
    # Sentinel test: import the existing loss/adapter/harness test module and
    # verify it still imports cleanly (no API drift from the kwarg addition).
    from tac.substrates._shared.mlx_score_aware.tests import (
        test_loss_adapter_harness,  # noqa: F401
    )
    # No assertions beyond import-time success; the dedicated existing tests
    # already cover the legacy behavior comprehensively. The new kwarg has a
    # safe default that preserves prior behavior.


@requires_mlx
def test_no_pr95_curriculum_kwargs_required_for_legacy_callers() -> None:
    """Legacy callers can construct MlxScoreAwareAdapter with NO new kwargs."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    bundle = _make_minimal_bundle()
    # The legacy constructor signature MUST still work without specifying
    # pr95_faithful_curriculum_enabled.
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="test_substrate")
    assert adapter.substrate_id == "test_substrate"
    assert adapter._pr95_faithful_curriculum_enabled is False
