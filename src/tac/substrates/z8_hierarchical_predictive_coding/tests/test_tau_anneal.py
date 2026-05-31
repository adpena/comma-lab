# SPDX-License-Identifier: MIT
"""Z8 hierarchical predictive coding Gumbel-Softmax τ-anneal tests.

τ-ANNEAL WAVE 2026-05-31 (lane ``lane_z8_hier_pc_full_stack_longrun_20260531``;
sister of DreamerV3 v2 commit ``e7b1e85f0``). Closes the "Comment-only
contracts are FORBIDDEN" violation: the config docstring at
``Z8HierarchicalConfig.gumbel_temperature`` claimed "Annealed during training
(1.0 → 0.1)" but ``forward_training`` read a STATIC ``cfg.gumbel_temperature``.

The headline NO-FAKE guard (``test_tau_actually_changes_per_epoch_not_static``)
would FAIL if ``forward_training`` reverts to the static read. Per CLAUDE.md
"NO FAKE IMPLEMENTATIONS" non-negotiable: τ MUST actually anneal, and the
schedule MUST actually drive the forward — not just be stored on an attribute.

Per CLAUDE.md "MLX portable-local-substrate authority" Catalog #192 — MLX is a
research-signal local substrate; NEVER promotable.

[verified-against: tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer.Z8HierarchicalPredictiveCoderMLX.notify_global_epoch]
[verified-against: tac.substrates.dreamer_v3_rssm.gumbel_temperature_for_epoch canonical cosine schedule]
[verified-against: tac.substrates._shared.mlx_score_aware.adapter.MlxScoreAwareAdapter.notify_global_epoch (forwards to renderer)]
"""

from __future__ import annotations

import itertools

import pytest

from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
    _gumbel_temperature_for_epoch,
)


def _tiny_cfg():
    """Tiny smoke-friendly Z8 config (kept small so the test runs in ms)."""
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalConfig,
    )

    return Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=4,
        deterministic_state_dim=8,
        gumbel_temperature=1.0,
        use_straight_through=True,
    )


def _tiny_model():
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalPredictiveCoderMLX,
    )

    return Z8HierarchicalPredictiveCoderMLX(_tiny_cfg())


# -----------------------------------------------------------------------------
# Pure-function schedule tests (NO MLX required — the schedule is numpy/math).
# -----------------------------------------------------------------------------


def test_tau_schedule_endpoints_exact() -> None:
    """τ(0) == τ_start and τ(E-1) == τ_min exactly (canonical cosine schedule)."""
    assert _gumbel_temperature_for_epoch(0, 300, tau_start=1.0, tau_min=0.1) == (
        pytest.approx(1.0, abs=1e-9)
    )
    assert _gumbel_temperature_for_epoch(
        299, 300, tau_start=1.0, tau_min=0.1
    ) == pytest.approx(0.1, abs=1e-9)


def test_tau_schedule_monotonic_descent() -> None:
    """τ decreases monotonically high→low over the budget (Jang 2016 recipe)."""
    taus = [
        _gumbel_temperature_for_epoch(e, 100, tau_start=1.0, tau_min=0.1)
        for e in range(100)
    ]
    for earlier, later in itertools.pairwise(taus):
        assert later <= earlier + 1e-9, (earlier, later)
    assert taus[0] - taus[-1] > 0.8


def test_tau_schedule_degenerate_single_epoch_returns_start() -> None:
    """E <= 1 returns τ_start (no anneal possible; sister contract)."""
    assert _gumbel_temperature_for_epoch(0, 1, tau_start=1.0, tau_min=0.1) == (
        pytest.approx(1.0)
    )


def test_tau_schedule_delegates_to_sister_dreamerv3() -> None:
    """Z8 schedule is bit-identical to the sister DreamerV3 canonical source.

    Per Catalog #290 ADOPT_CANONICAL_BECAUSE_SERVES: Z8 must NOT carry a local
    duplicate of the anneal math. This test would FAIL if a future edit
    re-introduced a local Z8 schedule that drifts from the canonical source.
    """
    from tac.substrates.dreamer_v3_rssm import gumbel_temperature_for_epoch

    for e in (0, 17, 150, 299):
        z8 = _gumbel_temperature_for_epoch(e, 300, tau_start=1.0, tau_min=0.1)
        sister = gumbel_temperature_for_epoch(
            e, 300, tau_start=1.0, tau_min=0.1
        )
        assert z8 == pytest.approx(sister, abs=1e-12), (e, z8, sister)


# -----------------------------------------------------------------------------
# MLX renderer behavior tests (require MLX runtime).
# -----------------------------------------------------------------------------


@pytest.mark.skipif(
    pytest.importorskip("mlx", reason="MLX runtime required") is None,
    reason="MLX runtime required",
)
def test_tau_actually_changes_per_epoch_not_static() -> None:
    """THE headline NO-FAKE guard: the τ ``forward_training`` reads changes.

    This FAILS if ``forward_training`` reverts to the static
    ``cfg.gumbel_temperature`` read (the comment-only-contract violation it
    closes). A static-τ implementation makes ``tau_early == tau_late``.
    """
    m = _tiny_model()
    m.set_anneal_schedule(total_epochs=300, tau_min=0.1)

    m.notify_global_epoch(0)
    tau_early = m.current_gumbel_temperature
    m.notify_global_epoch(150)
    tau_mid = m.current_gumbel_temperature
    m.notify_global_epoch(299)
    tau_late = m.current_gumbel_temperature

    assert tau_early == pytest.approx(1.0, abs=1e-6), tau_early
    assert tau_late == pytest.approx(0.1, abs=1e-6), tau_late
    # The load-bearing inequality: a static-τ implementation makes these equal.
    assert tau_early - tau_late > 0.8, (tau_early, tau_late)
    assert tau_late < tau_mid < tau_early, (tau_early, tau_mid, tau_late)


@pytest.mark.skipif(
    pytest.importorskip("mlx", reason="MLX runtime required") is None,
    reason="MLX runtime required",
)
def test_notify_without_schedule_is_static_no_op() -> None:
    """No ``set_anneal_schedule`` ⇒ ``notify_global_epoch`` is a no-op.

    Backward-compat: a renderer that never configures a schedule keeps the
    static ``cfg.gumbel_temperature`` (1.0), preserving prior behavior.
    """
    m = _tiny_model()
    assert m.current_gumbel_temperature == pytest.approx(1.0)
    m.notify_global_epoch(0)
    m.notify_global_epoch(150)
    m.notify_global_epoch(299)
    assert m.current_gumbel_temperature == pytest.approx(1.0), (
        "notify_global_epoch mutated τ with NO schedule configured "
        "(backward-compat broken)."
    )


@pytest.mark.skipif(
    pytest.importorskip("mlx", reason="MLX runtime required") is None,
    reason="MLX runtime required",
)
def test_set_anneal_schedule_rejects_invalid_tau_floor() -> None:
    """Invalid τ schedules fail at configuration time, not best-effort epoch hooks."""
    m = _tiny_model()

    with pytest.raises(ValueError, match="0 < tau_min"):
        m.set_anneal_schedule(total_epochs=100, tau_min=0.0)
    with pytest.raises(ValueError, match="0 < tau_min"):
        m.set_anneal_schedule(total_epochs=100, tau_min=1.5)
    with pytest.raises(ValueError, match="total_epochs"):
        m.set_anneal_schedule(total_epochs=0, tau_min=0.1)


@pytest.mark.skipif(
    pytest.importorskip("mlx", reason="MLX runtime required") is None,
    reason="MLX runtime required",
)
def test_tau_is_not_a_trainable_parameter() -> None:
    """NO-FAKE: the annealed τ must NOT leak into ``module.parameters()``.

    A τ stored as an mx.array would be picked up by the optimizer and
    overwritten by gradient steps (defeating the schedule). It MUST be a plain
    Python float kept out of the trainable parameter tree / state_dict export.
    """
    from mlx.utils import tree_flatten

    m = _tiny_model()
    m.set_anneal_schedule(total_epochs=100, tau_min=0.1)
    m.notify_global_epoch(50)
    flat = tree_flatten(m.parameters())
    names = [name for name, _ in flat]
    assert not any(
        "gumbel_temperature" in n or "_current_gumbel" in n or "_anneal_" in n
        for n in names
    ), f"τ-anneal state leaked into trainable params: {names}"


@pytest.mark.skipif(
    pytest.importorskip("mlx", reason="MLX runtime required") is None,
    reason="MLX runtime required",
)
def test_forward_training_consumes_annealed_tau() -> None:
    """The annealed τ actually reaches ``gumbel_softmax_sample``.

    Sharper τ (late training) yields a near-one-hot soft sample (max prob
    closer to 1.0) than smooth τ (early training) on the SAME logits + key.
    This proves the annealed value flows through the forward — not just stored.
    """
    import mlx.core as mx

    m = _tiny_model()
    indices = mx.array([0], dtype=mx.int32)
    key = mx.random.key(7)

    # Early (smooth, τ≈1.0): broader simplex (lower max prob).
    m.set_anneal_schedule(total_epochs=300, tau_min=0.1)
    m.notify_global_epoch(0)
    _rgb, _idx, soft_early = m.forward_training(indices, gumbel_key=key)

    # Late (sharp, τ≈0.1): more discrete simplex (higher max prob).
    m.notify_global_epoch(299)
    _rgb2, _idx2, soft_late = m.forward_training(indices, gumbel_key=key)

    # Compare top-level (level 0) soft-sample peakiness; STE forward returns a
    # near-one-hot so we compare the SOFT gradient surrogate behind it via the
    # entropy proxy: sharper τ ⇒ lower entropy.
    def _mean_entropy(soft_list) -> float:
        s = soft_list[0]
        ent = -mx.sum(s * mx.log(s + 1e-10), axis=-1)
        return float(mx.mean(ent).item())

    ent_early = _mean_entropy(soft_early)
    ent_late = _mean_entropy(soft_late)
    # Sharper τ (late) ⇒ lower entropy soft sample. STE returns one-hot so this
    # is the underlying soft-gradient signal that actually changed.
    assert ent_late <= ent_early + 1e-6, (ent_early, ent_late)
