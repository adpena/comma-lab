# SPDX-License-Identifier: MIT
"""Tests for the diagnostics-cadence throughput fix + vectorized flood-fill.

Lane ``lane_throughput_fix_mlx_score_aware_20260609``. Per CLAUDE.md "NO FAKE
IMPLEMENTATIONS" (the speedup must PRESERVE the training math, proven by
measurement) + "Bugs must be permanently fixed AND self-protected against" +
Slot EEE 5 forbidden classes (Class 2: tests verify behavior, not constants):

These tests prove BEHAVIOR, not metadata:

1. **Math parity (the load-bearing invariant)**: the per-step ``total`` loss
   trajectory is byte-IDENTICAL between ``diagnostics_every_n_steps=1`` (the
   byte-stable default == pre-fix adapter) and a higher cadence. Gating the
   per-step diagnostics is observability-only on the default (guard-off) path,
   so it CANNOT change the loss/gradient/optimizer trajectory. If the cadence
   gate ever started feeding the gradient, this test would FAIL.

2. **Default byte-stability for sisters**: at the default cadence the returned
   metrics key set is unchanged (no new cadence keys leak), so sister
   substrates (z7/z8/dreamer/etc.) that assert exact key sets are unaffected.

3. **The sampler actually samples**: at cadence N the per-step diagnostics
   (which DO appear in the metrics dict, e.g. ``dynamics_pre_update_*``) are
   present on sampled steps and ABSENT on skipped steps — proving the gate
   genuinely skips the work rather than running it anyway.

4. **Flood-fill vectorization bit-identity**: the vectorized
   ``scipy.ndimage.label`` worst-connected-region-margin path returns results
   IDENTICAL to the preserved pure-Python reference oracle across realistic
   (large contiguous regions), fragmented (many tiny components), tie, and
   even-count-median cases.

[verified-against: scipy.ndimage.{label,median} 4-connectivity == von Neumann]
[verified-against: np.percentile(.,50) linear-interpolation median]
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def tiny_recon_bundle():
    """Minimal recon-only MLX bundle whose forward depends on trainable weights.

    No scorer teacher -> the score-aware loss runs in reconstruction-only mode,
    which is enough to exercise train_step end-to-end deterministically and
    cheaply (no SegNet/PoseNet construction). The renderer output DEPENDS on the
    trainable weight so gradients are nonzero and the loss actually moves.
    """
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx.nn")
    import mlx.core as mx
    import mlx.nn as mlx_nn

    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle

    class TinyRenderer(mlx_nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = mx.random.normal((3, 4 * 4 * 3))

        def reconstruct_pair(self, batch):
            n = batch.shape[0]
            base = mx.broadcast_to(self.w.sum(), (n, 3, 4, 4))
            return base, base * 1.0

    return RendererBundle(
        model=TinyRenderer(),
        target_rgb_0=mx.ones((4, 4, 4, 3)) * 0.5,
        target_rgb_1=mx.ones((4, 4, 4, 3)) * 0.5,
        num_pairs=4,
        forward_convention="reconstruct_pair_nchw01",
    )


def _build(bundle, cadence):
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    mx.random.seed(0)
    # Rebuild bundle weights deterministically by re-seeding then re-creating the
    # renderer weight so both cadences start from IDENTICAL parameters.
    bundle.model.w = mx.random.normal((3, 4 * 4 * 3))
    return MlxScoreAwareAdapter(
        bundle, substrate_id="cadence_test", diagnostics_every_n_steps=cadence
    )


def _run_trajectory(bundle, cadence, steps=12):
    import mlx.core as mx

    adapter = _build(bundle, cadence)
    losses = []
    metrics_by_step = []
    for _step in range(steps):
        idx = mx.array([0, 1, 2, 3])
        out = adapter.train_step(idx, 1e-2, {})
        losses.append(float(out["total"]))
        metrics_by_step.append(dict(out))
    return losses, metrics_by_step, adapter


# -----------------------------------------------------------------------------
# CONSTRUCTOR + VALIDATION
# -----------------------------------------------------------------------------


def test_default_cadence_is_one(tiny_recon_bundle):
    """Default diagnostics_every_n_steps == 1 (byte-stable; every step)."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    a = MlxScoreAwareAdapter(tiny_recon_bundle, substrate_id="t")
    assert a._diagnostics_every_n_steps == 1


def test_cadence_coerced_to_minimum_one(tiny_recon_bundle):
    """A cadence of 0 or negative coerces to 1 (never divide-by-zero / skip-all)."""
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    for bad in (0, -5):
        a = MlxScoreAwareAdapter(
            tiny_recon_bundle, substrate_id="t", diagnostics_every_n_steps=bad
        )
        assert a._diagnostics_every_n_steps == 1


def test_explicit_cadence_respected(tiny_recon_bundle):
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    a = MlxScoreAwareAdapter(
        tiny_recon_bundle, substrate_id="t", diagnostics_every_n_steps=50
    )
    assert a._diagnostics_every_n_steps == 50


# -----------------------------------------------------------------------------
# THE LOAD-BEARING INVARIANT: gating does NOT change training math
# -----------------------------------------------------------------------------


def test_loss_trajectory_identical_cadence_1_vs_50(tiny_recon_bundle):
    """Cadence gating is observability-only: loss trajectory MUST be identical.

    This is the NO-FAKE proof that the speedup preserves the training math. If
    the gate ever fed the gradient or skipped a load-bearing step, the two
    trajectories would diverge and this test would fail.
    """
    losses_1, _m1, _a1 = _run_trajectory(tiny_recon_bundle, 1)
    losses_50, _m50, _a50 = _run_trajectory(tiny_recon_bundle, 50)
    assert len(losses_1) == len(losses_50)
    max_abs_diff = max(abs(a - b) for a, b in zip(losses_1, losses_50, strict=True))
    assert max_abs_diff == 0.0, (
        f"cadence gating changed the loss trajectory (max abs diff "
        f"{max_abs_diff}); the gate must be observability-only"
    )


def test_loss_trajectory_identical_for_several_cadences(tiny_recon_bundle):
    """Several cadences all reproduce the exact cadence-1 loss trajectory."""
    base, _m, _a = _run_trajectory(tiny_recon_bundle, 1)
    for cadence in (2, 3, 7, 100):
        other, _m2, _a2 = _run_trajectory(tiny_recon_bundle, cadence)
        max_abs_diff = max(abs(a - b) for a, b in zip(base, other, strict=True))
        assert max_abs_diff == 0.0, f"cadence={cadence} diverged ({max_abs_diff})"


def test_loss_actually_moves(tiny_recon_bundle):
    """Sanity: the trajectory is a REAL training signal (loss changes), so the
    parity test above is non-vacuous (it is not comparing two constant loss
    sequences)."""
    losses, _m, _a = _run_trajectory(tiny_recon_bundle, 1)
    assert len({round(x, 8) for x in losses}) > 1, "loss never moved"


def _run_trajectory_guard_on(bundle, cadence, steps=12):
    """Run a trajectory with the scorer-space step guard ENABLED.

    When the guard is on, the guard-FEEDING diagnostics (pre/post loss-part +
    receiver snapshot + the pre-update param trace the guard restores from) are
    load-bearing and must ALWAYS run regardless of cadence. This helper proves
    the cadence gate does not break the guard's reject/restore decision.
    """
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    mx.random.seed(0)
    bundle.model.w = mx.random.normal((3, 4 * 4 * 3))
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="cadence_guard_test",
        diagnostics_every_n_steps=cadence,
        scorer_space_step_guard_enabled=True,
    )
    losses = []
    guard_flags = []
    for _step in range(steps):
        idx = mx.array([0, 1, 2, 3])
        out = adapter.train_step(idx, 1e-2, {})
        losses.append(float(out["total"]))
        guard_flags.append(
            float(out.get("scorer_space_step_guard_rejected", 0.0))
        )
    return losses, guard_flags


def test_guard_on_loss_trajectory_identical_across_cadences(tiny_recon_bundle):
    """SAFETY-CRITICAL: with the scorer-space step guard ENABLED, the loss
    trajectory MUST be identical across cadences.

    The guard can reject/restore an optimizer step based on the pre/post
    loss-part metrics + receiver snapshot. The cadence gate keeps those
    guard-feeding diagnostics running EVERY step when the guard is on, so the
    guard's decision (and thus the training trajectory) is unchanged. If a
    skipped step ever starved the guard of its inputs, the trajectory would
    diverge and this test would fail.
    """
    base, base_flags = _run_trajectory_guard_on(tiny_recon_bundle, 1)
    for cadence in (5, 50):
        other, other_flags = _run_trajectory_guard_on(tiny_recon_bundle, cadence)
        assert max(abs(a - b) for a, b in zip(base, other, strict=True)) == 0.0, (
            f"guard-on trajectory diverged at cadence={cadence}"
        )
        # The guard reject/accept decision sequence must also be identical.
        assert base_flags == other_flags, (
            f"guard reject/accept sequence diverged at cadence={cadence}"
        )


def test_guard_on_always_runs_guard_feeding_diagnostics(tiny_recon_bundle):
    """With the guard ON, even SKIPPED telemetry steps still emit the
    guard-feeding pre-update loss-part metrics (the guard consumed them)."""
    losses, _flags = _run_trajectory_guard_on(tiny_recon_bundle, 50, steps=4)
    # Re-run capturing full metrics to inspect guard-feeding keys per step.
    import mlx.core as mx

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter

    mx.random.seed(0)
    tiny_recon_bundle.model.w = mx.random.normal((3, 4 * 4 * 3))
    adapter = MlxScoreAwareAdapter(
        tiny_recon_bundle,
        substrate_id="g",
        diagnostics_every_n_steps=50,
        scorer_space_step_guard_enabled=True,
    )
    idx = mx.array([0, 1, 2, 3])
    # Step 1 is a SKIPPED telemetry step, but guard is on, so the guard-feeding
    # pre-update loss-part metrics must still be present.
    adapter.train_step(idx, 1e-2, {})  # step 0
    out1 = adapter.train_step(idx, 1e-2, {})  # step 1 (telemetry-skipped)
    assert any(k.startswith("dynamics_pre_update_") for k in out1), (
        "guard-on skipped step must still run guard-feeding pre-update metrics"
    )


# -----------------------------------------------------------------------------
# DEFAULT BYTE-STABILITY: no new keys leak at the default cadence
# -----------------------------------------------------------------------------


def test_default_cadence_returns_no_cadence_keys(tiny_recon_bundle):
    """At default cadence the returned metric key set carries NO cadence keys.

    This keeps the default returned dict byte-identical for sister substrates
    that assert exact key sets.
    """
    _losses, metrics_by_step, _a = _run_trajectory(tiny_recon_bundle, 1)
    cadence_keys = {
        "diagnostics_every_n_steps",
        "diagnostics_sampled_step_count",
        "diagnostics_skipped_step_count",
    }
    for m in metrics_by_step:
        assert cadence_keys.isdisjoint(m.keys())


def test_higher_cadence_adds_only_cadence_keys(tiny_recon_bundle):
    """cadence>1 adds ONLY the 3 cadence observability keys vs the default keys.

    On a SAMPLED step the cadence>1 metric set equals the cadence-1 set plus the
    3 cadence keys; gating must not silently drop any non-diagnostic key.
    """
    _l1, m1_by_step, _a1 = _run_trajectory(tiny_recon_bundle, 1)
    _l50, m50_by_step, _a50 = _run_trajectory(tiny_recon_bundle, 50)
    cadence_keys = {
        "diagnostics_every_n_steps",
        "diagnostics_sampled_step_count",
        "diagnostics_skipped_step_count",
    }
    # Step 0 is always a SAMPLED step under either cadence.
    keys_1_step0 = set(m1_by_step[0].keys())
    keys_50_step0 = set(m50_by_step[0].keys())
    assert keys_50_step0 - keys_1_step0 == cadence_keys
    assert keys_1_step0 - keys_50_step0 == set()


# -----------------------------------------------------------------------------
# THE SAMPLER ACTUALLY SAMPLES (skips the work, doesn't run it anyway)
# -----------------------------------------------------------------------------


def test_sampler_counts_sampled_and_skipped(tiny_recon_bundle):
    """Over 12 steps at cadence 5, exactly steps 0,5,10 sample (3 sampled / 9 skipped)."""
    _losses, _m, adapter = _run_trajectory(tiny_recon_bundle, 5, steps=12)
    cm = adapter.diagnostics_cadence_metrics()
    assert cm["diagnostics_every_n_steps"] == 5.0
    assert cm["diagnostics_sampled_step_count"] == 3.0
    assert cm["diagnostics_skipped_step_count"] == 9.0


def test_skipped_steps_omit_sampled_diagnostics(tiny_recon_bundle):
    """A SKIPPED step omits the sampled per-step diagnostic keys; a SAMPLED step
    includes them. This proves the gate genuinely skips the diagnostic work.

    ``dynamics_pre_update_*`` keys come from the pre-update loss-part RECOMPUTE
    which is gated; they must be present on the sampled step 0 and absent on a
    skipped step (e.g. step 1 at cadence 50).
    """
    _losses, metrics_by_step, _a = _run_trajectory(
        tiny_recon_bundle, 50, steps=4
    )

    def _has_dynamics_pre_update(m):
        return any(k.startswith("dynamics_pre_update_") for k in m)

    # step 0 == sampled
    assert _has_dynamics_pre_update(metrics_by_step[0])
    # step 1,2,3 == skipped at cadence 50
    for skipped in (1, 2, 3):
        assert not _has_dynamics_pre_update(metrics_by_step[skipped]), (
            f"step {skipped} should be skipped but ran the gated pre-update "
            "loss-part recompute"
        )


def test_request_diagnostics_flush_forces_next_step(tiny_recon_bundle):
    """request_diagnostics_flush() forces the NEXT step to sample even mid-cadence."""
    import mlx.core as mx

    adapter = _build(tiny_recon_bundle, 100)
    idx = mx.array([0, 1, 2, 3])
    out0 = adapter.train_step(idx, 1e-2, {})  # step 0 always sampled
    assert any(k.startswith("dynamics_pre_update_") for k in out0)
    out1 = adapter.train_step(idx, 1e-2, {})  # step 1 normally skipped
    assert not any(k.startswith("dynamics_pre_update_") for k in out1)
    adapter.request_diagnostics_flush()
    out2 = adapter.train_step(idx, 1e-2, {})  # forced sample
    assert any(k.startswith("dynamics_pre_update_") for k in out2)


def test_flush_does_not_change_loss(tiny_recon_bundle):
    """Forcing a diagnostics flush is observability-only: loss is unchanged vs
    the non-flushed cadence-1 trajectory at the same step."""
    import mlx.core as mx

    base, _m, _a = _run_trajectory(tiny_recon_bundle, 1, steps=4)
    adapter = _build(tiny_recon_bundle, 100)
    losses = []
    idx = mx.array([0, 1, 2, 3])
    for step in range(4):
        if step == 2:
            adapter.request_diagnostics_flush()
        out = adapter.train_step(idx, 1e-2, {})
        losses.append(float(out["total"]))
    assert max(abs(a - b) for a, b in zip(base, losses, strict=True)) == 0.0


# -----------------------------------------------------------------------------
# FLOOD-FILL VECTORIZATION BIT-IDENTITY (vs preserved reference oracle)
# -----------------------------------------------------------------------------


def _vec():
    from tac.substrates._shared.mlx_score_aware.adapter import (
        _worst_connected_region_margin_p50_from_numpy,
    )

    return _worst_connected_region_margin_p50_from_numpy


def _ref():
    from tac.substrates._shared.mlx_score_aware.adapter import (
        _worst_connected_region_margin_p50_reference_from_numpy,
    )

    return _worst_connected_region_margin_p50_reference_from_numpy


def _assert_match(ref_result, vec_result):
    if ref_result is None or vec_result is None:
        assert ref_result is None and vec_result is None
        return
    assert abs(ref_result[0] - vec_result[0]) < 1e-9, (ref_result, vec_result)
    assert ref_result[1] == vec_result[1], (ref_result, vec_result)
    assert ref_result[2] == vec_result[2], (ref_result, vec_result)


def test_floodfill_vectorized_matches_reference_realistic():
    """Realistic SegNet-like argmax (few large contiguous regions)."""
    pytest.importorskip("scipy")
    from scipy import ndimage as ndi

    H, W, K = 96, 128, 5
    for seed in range(5):
        r = np.random.default_rng(seed)
        field = r.standard_normal((1, H, W, K)).astype(np.float32)
        field = ndi.gaussian_filter(field, (0, 6, 6, 0))
        target = np.argmax(field, axis=-1).astype(np.int32)
        logits = r.standard_normal((1, H, W, K)).astype(np.float32)
        _assert_match(_ref()(logits, target), _vec()(logits, target))


def test_floodfill_vectorized_matches_reference_fragmented():
    """Fragmented argmax (many tiny components) — the worst case for the
    pure-Python flood-fill and the case the vectorized path most accelerates."""
    pytest.importorskip("scipy")
    H, W, K = 48, 64, 5
    for seed in range(5):
        r = np.random.default_rng(100 + seed)
        logits = r.standard_normal((1, H, W, K)).astype(np.float32)
        target = np.argmax(
            logits + r.standard_normal((1, H, W, K)) * 0.3, axis=-1
        ).astype(np.int32)
        _assert_match(_ref()(logits, target), _vec()(logits, target))


def test_floodfill_vectorized_matches_reference_multibatch():
    """Multi-batch input — best across batch items, identical to reference."""
    pytest.importorskip("scipy")
    H, W, K = 32, 40, 4
    r = np.random.default_rng(7)
    logits = r.standard_normal((3, H, W, K)).astype(np.float32)
    target = np.argmax(
        logits + r.standard_normal((3, H, W, K)) * 0.4, axis=-1
    ).astype(np.int32)
    _assert_match(_ref()(logits, target), _vec()(logits, target))


def test_floodfill_vectorized_tie_break_matches_reference():
    """On EQUAL p50 margins the first-encountered (row-major) component wins —
    the vectorized path replicates this raster-order tie-break exactly."""
    pytest.importorskip("scipy")
    logits = np.zeros((1, 16, 16, 3), dtype=np.float32)
    logits[..., 1] = 0.5
    target = np.ones((1, 16, 16), dtype=np.int32)
    target[0, 2:5, 2:5] = 0
    target[0, 10:13, 10:13] = 0
    _assert_match(_ref()(logits, target), _vec()(logits, target))


def test_floodfill_vectorized_even_count_median_matches_reference():
    """A 2-pixel component exercises the linear-interpolation (even-count)
    median; scipy.ndimage.median must match np.percentile(.,50) exactly."""
    pytest.importorskip("scipy")
    logits = np.zeros((1, 4, 4, 3), dtype=np.float32)
    target = np.ones((1, 4, 4), dtype=np.int32)
    target[0, 0, 0] = 0
    target[0, 0, 1] = 0  # 2-px class-0 component, margins 3 and 1 -> median 2
    logits[0, 0, 0, 0] = 3.0
    logits[0, 0, 1, 0] = 1.0
    _assert_match(_ref()(logits, target), _vec()(logits, target))


def test_floodfill_method_returns_none_on_bad_shapes():
    """The METHOD wrapper returns None on malformed shapes (the helper assumes
    method-validated 4D logits / 3D target). The defensive ndim/shape guard
    lives at the method surface, preserved from the pre-fix implementation."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import mlx.nn as mlx_nn

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle

    class TinyRenderer(mlx_nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = mx.zeros((2, 3))

        def reconstruct_pair(self, batch):
            n = batch.shape[0]
            z = mx.zeros((n, 3, 4, 4))
            return z, z

    bundle = RendererBundle(
        model=TinyRenderer(),
        target_rgb_0=mx.zeros((4, 4, 4, 3)),
        target_rgb_1=mx.zeros((4, 4, 4, 3)),
        num_pairs=4,
        forward_convention="reconstruct_pair_nchw01",
    )
    adapter = MlxScoreAwareAdapter(bundle, substrate_id="t")
    # 3D logits (missing class axis) + 3D target -> None (shape mismatch guard).
    bad_logits = mx.zeros((1, 4, 4))
    bad_target = mx.zeros((1, 4, 4))
    assert (
        adapter._receiver_surface_worst_connected_region_margin_p50(
            logits=bad_logits, target_argmax=bad_target
        )
        is None
    )
    assert (
        adapter._receiver_surface_worst_connected_region_margin_p50_reference(
            logits=bad_logits, target_argmax=bad_target
        )
        is None
    )
    # None inputs -> None.
    assert (
        adapter._receiver_surface_worst_connected_region_margin_p50(
            logits=None, target_argmax=None
        )
        is None
    )


# -----------------------------------------------------------------------------
# HARNESS -> ADAPTER FORWARDING (the long-training path reachability fix)
# -----------------------------------------------------------------------------
# Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
# the throughput-fix cadence lived on the adapter but was NOT threaded through
# `run_mlx_score_aware_full_main`, so the ~1.65-1.78x speedup was unreachable
# from the canonical long-training path that every MLX-first substrate trainer
# (incl. the B1 229K HiNeRV pilot) routes through. These tests prove the
# harness FORWARDS the kwarg to the adapter (behavior, not a constant), so the
# regression cannot silently return.


class _AdapterConstructionCaptured(Exception):
    """Sentinel raised by the spy to short-circuit harness execution."""

    def __init__(self, captured_kwargs: dict) -> None:
        super().__init__("adapter construction captured")
        self.captured_kwargs = captured_kwargs


def _run_harness_capturing_adapter_kwargs(monkeypatch, bundle, **harness_overrides):
    """Call the harness with a spy adapter that captures ctor kwargs then stops."""
    import tac.substrates._shared.mlx_score_aware.harness as harness_mod

    captured: dict = {}

    def _spy_adapter(_bundle, **kwargs):
        captured.update(kwargs)
        raise _AdapterConstructionCaptured(dict(kwargs))

    monkeypatch.setattr(harness_mod, "MlxScoreAwareAdapter", _spy_adapter)

    base_kwargs = {
        "bundle": bundle,
        "substrate_id": "test_diag_cadence_harness",
        "lane_id": "lane_test_diag_cadence_harness",
        "output_dir": "/Volumes/__never_written__/diag_cadence_test",
        "epochs": 1,
        "batch_pair_indices_per_step": 1,
    }
    base_kwargs.update(harness_overrides)
    try:
        harness_mod.run_mlx_score_aware_full_main(**base_kwargs)
    except _AdapterConstructionCaptured:
        pass
    return captured


def test_harness_forwards_diagnostics_every_n_steps_to_adapter(
    tiny_recon_bundle, monkeypatch
):
    """The harness MUST pass an explicit cadence through to the adapter ctor."""
    pytest.importorskip("mlx.core")
    captured = _run_harness_capturing_adapter_kwargs(
        monkeypatch, tiny_recon_bundle, diagnostics_every_n_steps=50
    )
    assert "diagnostics_every_n_steps" in captured, (
        "run_mlx_score_aware_full_main did not forward diagnostics_every_n_steps "
        "to MlxScoreAwareAdapter; the long-training-path throughput fix is "
        "unreachable (regression)."
    )
    assert int(captured["diagnostics_every_n_steps"]) == 50


def test_harness_diagnostics_cadence_defaults_to_1(tiny_recon_bundle, monkeypatch):
    """Default (omitted) cadence forwards as 1 == byte-stable every-step path."""
    pytest.importorskip("mlx.core")
    captured = _run_harness_capturing_adapter_kwargs(monkeypatch, tiny_recon_bundle)
    assert int(captured.get("diagnostics_every_n_steps", -1)) == 1
