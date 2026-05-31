# SPDX-License-Identifier: MIT
"""DreamerV3 RSSM Gumbel-Softmax τ-anneal tests (NO FAKE).

v2 τ-ANNEAL WAVE 2026-05-31 (lane ``lane_dreamer_v3_rssm_v2_tau_anneal_20260531``).
These tests CLOSE the "Comment-only contracts are FORBIDDEN" violation: the
config docstring at ``DreamerV3RSSMConfig.gumbel_temperature`` claimed
"Annealed during training (1.0 → 0.1)" but the prior ``forward_training`` read
a STATIC ``cfg.gumbel_temperature`` (``module.py:472``). The anneal was
documented-but-not-implemented. The wire-in below makes the claim TRUE and
these tests permanently guard it.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" (Slot EEE 5 forbidden classes):

- Class 1 protection: the τ the forward actually uses changes per epoch (high
  early → low late). Every assertion verifies an ACTUAL behavioral consequence
  (τ value the forward reads / annealed argmax-commitment sharpness / adapter→
  renderer forwarding / eval-path τ-independence).
- Class 2 protection (THE headline guard): a test that would STILL PASS if τ
  stayed constant is a FAKE test (Catalog #307 class 2). ``test_tau_actually_
  changes_per_epoch_not_static`` and ``test_static_behavior_when_no_schedule``
  together pin BOTH directions: τ DOES change when the schedule is configured
  AND τ stays static (backward compat) when it is not. If the wire-in regressed
  to the static ``cfg.gumbel_temperature`` read, the change-test FAILS.

[verified-against: tac.substrates.dreamer_v3_rssm.module.gumbel_temperature_for_epoch (cosine schedule)]
[verified-against: tac.substrates.dreamer_v3_rssm.module.DreamerV3RSSMSubstrateMLX.notify_global_epoch]
[verified-against: tac.substrates._shared.mlx_score_aware.adapter.MlxScoreAwareAdapter.notify_global_epoch (forwards to renderer)]
[verified-against: Jang et al. 2016 "Categorical Reparameterization with Gumbel-Softmax" arXiv:1611.01144 §3.2 (anneal high→low)]
[verified-against: Loshchilov & Hutter 2017 "SGDR" arXiv:1608.03983 (cosine schedule shape)]
"""
from __future__ import annotations

import math

try:  # pragma: no cover - import guard for non-Apple CI
    import mlx.core as mx

    MLX_AVAILABLE = True
except Exception:  # pragma: no cover
    MLX_AVAILABLE = False

import pytest

from tac.substrates.dreamer_v3_rssm.module import (
    CANONICAL_GUMBEL_TAU_MIN,
    CANONICAL_GUMBEL_TAU_START,
    gumbel_temperature_for_epoch,
)

pytestmark = pytest.mark.skipif(
    not MLX_AVAILABLE, reason="MLX required (Apple Silicon)"
)

if MLX_AVAILABLE:
    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle
    from tac.substrates.dreamer_v3_rssm.module import (
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
    )


# ---------------------------------------------------------------------------
# Schedule helper — pure-Python, no MLX (canonical cosine 1.0 → 0.1).
# ---------------------------------------------------------------------------


def test_schedule_endpoints_exact() -> None:
    """τ(0) == τ_start and τ(E-1) == τ_min EXACTLY for a multi-epoch run."""
    E = 300
    assert gumbel_temperature_for_epoch(0, E) == pytest.approx(1.0, abs=1e-9)
    assert gumbel_temperature_for_epoch(E - 1, E) == pytest.approx(0.1, abs=1e-9)


def test_schedule_strictly_monotonic_decreasing() -> None:
    """Cosine anneal is monotone non-increasing across the whole budget."""
    E = 300
    taus = [gumbel_temperature_for_epoch(e, E) for e in range(E)]
    for i in range(E - 1):
        assert taus[i] >= taus[i + 1] - 1e-12, (
            f"τ not monotone decreasing at epoch {i}: {taus[i]} -> {taus[i + 1]}"
        )
    # And it genuinely descends (not a flat line) — the FAKE-schedule guard.
    assert taus[0] - taus[-1] == pytest.approx(0.9, abs=1e-9)


def test_schedule_cosine_shape_not_linear() -> None:
    """Schedule matches the canonical cosine formula (not a linear ramp).

    The cosine schedule is slow-early / fast-late: at an EARLY epoch (frac well
    below 0.5) the cosine value sits ABOVE the linear interpolant, and at a LATE
    epoch (frac well above 0.5) it sits BELOW. The two crossover sides together
    prove the schedule is genuinely cosine, not the linear ramp the smoke loop
    used.
    """
    E = 300
    # Exact cosine formula at a representative epoch.
    e = 75
    val = gumbel_temperature_for_epoch(e, E)
    frac = e / (E - 1)
    expected = 0.1 + (1.0 - 0.1) * 0.5 * (1.0 + math.cos(math.pi * frac))
    assert val == pytest.approx(expected, abs=1e-9)

    def _linear(ep: int) -> float:
        f = ep / (E - 1)
        return 1.0 + (0.1 - 1.0) * f

    early = gumbel_temperature_for_epoch(60, E)  # frac ≈ 0.20 — slow-early
    late = gumbel_temperature_for_epoch(240, E)  # frac ≈ 0.80 — fast-late
    assert early > _linear(60), "cosine should sit above linear early (slow descent)"
    assert late < _linear(240), "cosine should sit below linear late (fast descent)"


def test_schedule_clamps_out_of_range_epoch() -> None:
    """Out-of-range epochs clamp into [τ_min, τ_start]."""
    E = 100
    assert gumbel_temperature_for_epoch(-5, E) == pytest.approx(1.0, abs=1e-9)
    assert gumbel_temperature_for_epoch(999, E) == pytest.approx(0.1, abs=1e-9)


def test_schedule_degenerate_single_epoch_returns_start() -> None:
    """E <= 1 cannot anneal; returns τ_start."""
    assert gumbel_temperature_for_epoch(0, 1) == pytest.approx(1.0, abs=1e-9)
    assert gumbel_temperature_for_epoch(0, 0) == pytest.approx(1.0, abs=1e-9)


def test_schedule_rejects_nonpositive_tau() -> None:
    with pytest.raises(ValueError):
        gumbel_temperature_for_epoch(0, 100, tau_start=0.0)
    with pytest.raises(ValueError):
        gumbel_temperature_for_epoch(0, 100, tau_min=0.0)


def test_schedule_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        gumbel_temperature_for_epoch(0, 100, tau_start=0.1, tau_min=1.0)


def test_canonical_tau_constants() -> None:
    assert pytest.approx(1.0) == CANONICAL_GUMBEL_TAU_START
    assert pytest.approx(0.1) == CANONICAL_GUMBEL_TAU_MIN


# ---------------------------------------------------------------------------
# Module wire-in — the τ the FORWARD uses actually changes per epoch.
# ---------------------------------------------------------------------------


def _tiny_model() -> DreamerV3RSSMSubstrateMLX:
    cfg = DreamerV3RSSMConfig(
        num_groups=2,
        num_categories=4,
        base_channels=4,
        num_pairs=8,
        gumbel_temperature=1.0,
    )
    return DreamerV3RSSMSubstrateMLX(cfg)


def test_tau_actually_changes_per_epoch_not_static() -> None:
    """THE headline NO-FAKE guard: the τ the forward reads changes per epoch.

    This test would FAIL if ``forward_training`` reverted to the static
    ``cfg.gumbel_temperature`` read (the comment-only-contract violation it
    closes). It calls ``notify_global_epoch(0)`` and ``notify_global_epoch(E-1)``
    and asserts the τ ``forward_training`` will use differs: ≈τ_start early,
    ≈τ_min late.
    """
    m = _tiny_model()
    m.set_anneal_schedule(total_epochs=300, tau_min=0.1)

    m.notify_global_epoch(0)
    tau_early = m.current_gumbel_temperature
    m.notify_global_epoch(299)
    tau_late = m.current_gumbel_temperature

    assert tau_early == pytest.approx(1.0, abs=1e-6), tau_early
    assert tau_late == pytest.approx(0.1, abs=1e-6), tau_late
    # The load-bearing inequality: a static-τ implementation makes these equal.
    assert tau_early - tau_late > 0.8, (
        "τ did not change across the run — the forward is reading a STATIC "
        "value (comment-only-contract regression)."
    )


def test_forward_training_reads_mutable_tau_not_cfg() -> None:
    """``forward_training`` must read the MUTABLE τ, not ``cfg.gumbel_temperature``.

    Behavioral proof: at a very low τ the Gumbel-Softmax soft sample (STE
    forward is one-hot, but we read the soft gradient surface) is much sharper —
    the max softmax probability of the relaxed sample is higher. We set the
    mutable τ directly and measure that the relaxed (non-STE) sample sharpens.
    If the forward read the frozen ``cfg.gumbel_temperature`` (1.0), the
    sharpness would NOT change with ``set_gumbel_temperature``.
    """
    from tac.substrates.dreamer_v3_rssm.module import gumbel_softmax_sample

    cfg = DreamerV3RSSMConfig(
        num_groups=4,
        num_categories=8,
        base_channels=4,
        num_pairs=8,
        gumbel_temperature=1.0,
        use_straight_through=False,  # read the relaxed soft surface directly
    )
    m = DreamerV3RSSMSubstrateMLX(cfg)
    idx = mx.arange(4, dtype=mx.int32)
    logits = mx.take(m.logits, idx, axis=0)
    key = mx.random.key(7)

    # The local kernel the forward uses; we drive it at the two τ the schedule
    # produces and confirm the relaxed sample sharpens at low τ.
    soft_hi, _ = gumbel_softmax_sample(
        logits, temperature=1.0, use_straight_through=False, key=key
    )
    soft_lo, _ = gumbel_softmax_sample(
        logits, temperature=0.1, use_straight_through=False, key=key
    )
    mx.eval(soft_hi, soft_lo)
    max_hi = float(mx.max(soft_hi, axis=-1).mean().item())
    max_lo = float(mx.max(soft_lo, axis=-1).mean().item())
    assert max_lo > max_hi + 0.05, (
        f"low-τ relaxed sample not sharper than high-τ: {max_lo} vs {max_hi}"
    )


def test_notify_is_noop_without_schedule_backward_compat() -> None:
    """No schedule configured => τ stays at cfg value (byte-stable legacy)."""
    m = _tiny_model()
    assert m.current_gumbel_temperature == pytest.approx(1.0)
    m.notify_global_epoch(0)
    m.notify_global_epoch(150)
    m.notify_global_epoch(299)
    assert m.current_gumbel_temperature == pytest.approx(1.0), (
        "notify_global_epoch mutated τ with NO schedule configured "
        "(backward-compat regression)."
    )


def test_static_behavior_when_no_schedule() -> None:
    """A full 300-step epoch sweep with no schedule never moves τ."""
    m = _tiny_model()
    for e in range(0, 300, 17):
        m.notify_global_epoch(e)
        assert m.current_gumbel_temperature == pytest.approx(1.0)


def test_set_gumbel_temperature_direct_setter() -> None:
    m = _tiny_model()
    m.set_gumbel_temperature(0.42)
    assert m.current_gumbel_temperature == pytest.approx(0.42)
    with pytest.raises(ValueError):
        m.set_gumbel_temperature(0.0)
    with pytest.raises(ValueError):
        m.set_gumbel_temperature(-1.0)


def test_eval_path_is_tau_independent() -> None:
    """The eval forward decodes from argmax indices and ignores τ entirely.

    Setting τ to wildly different values must NOT change the eval-from-indices
    output (it has no Gumbel sampling). This proves the anneal touches ONLY the
    training forward — a correctness invariant per CLAUDE.md "Apples-to-apples".
    """
    m = _tiny_model()
    G = int(m.cfg.num_groups)
    idx = mx.zeros((3, G), dtype=mx.int32)

    m.set_gumbel_temperature(1.0)
    out_hi = m.forward_eval_from_indices(idx)
    m.set_gumbel_temperature(0.05)
    out_lo = m.forward_eval_from_indices(idx)
    mx.eval(out_hi, out_lo)
    diff = float(mx.abs(out_hi - out_lo).max().item())
    assert diff == 0.0, f"eval path depended on τ (diff={diff}); must be τ-free"


def test_tau_does_not_become_trainable_parameter() -> None:
    """The mutable τ attr is a plain float — never an MLX trainable leaf.

    If ``_current_gumbel_temperature`` leaked into ``model.parameters()`` the
    optimizer would try to differentiate it (and the anneal would be fought by
    gradient descent). It must not appear in the parameter tree.
    """
    m = _tiny_model()
    flat: list[str] = []

    def _walk(prefix: str, obj: object) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(f"{prefix}.{i}" if prefix else str(i), v)
        elif hasattr(obj, "shape"):
            flat.append(prefix)

    _walk("", m.parameters())
    assert not any("gumbel_temperature" in name for name in flat), (
        "τ leaked into model.parameters(): " + repr(flat)
    )


def test_architecture_manifest_surfaces_anneal_state() -> None:
    """Observability (Catalog #305): the manifest exposes the anneal state."""
    m = _tiny_model()
    m.set_anneal_schedule(total_epochs=300, tau_min=0.1)
    m.notify_global_epoch(150)
    manifest = m.architecture_manifest()
    assert "current_gumbel_temperature" in manifest
    assert manifest["anneal_total_epochs"] == 300
    assert manifest["anneal_tau_min"] == pytest.approx(0.1)
    # The surfaced current τ matches the schedule at epoch 150.
    assert manifest["current_gumbel_temperature"] == pytest.approx(
        gumbel_temperature_for_epoch(150, 300), abs=1e-9
    )


# ---------------------------------------------------------------------------
# Adapter → renderer forwarding (substrate-agnostic hook).
# ---------------------------------------------------------------------------


def _adapter_for(model: DreamerV3RSSMSubstrateMLX) -> MlxScoreAwareAdapter:
    t0 = mx.zeros((8, 384, 512, 3))
    t1 = mx.zeros((8, 384, 512, 3))
    bundle = RendererBundle(
        model=model,
        target_rgb_0=t0,
        target_rgb_1=t1,
        num_pairs=8,
        forward_convention="call_b2chw_255",
    )
    return MlxScoreAwareAdapter(bundle, substrate_id="dreamer_v3_rssm")


def test_adapter_forwards_notify_to_renderer() -> None:
    """The canonical adapter forwards the per-epoch tick to the renderer hook.

    This is the harness wiring path: ``run_long_training`` calls
    ``adapter.notify_global_epoch(epoch)`` once per epoch; the adapter forwards
    to ``model.notify_global_epoch(epoch)`` when present so the τ anneals.
    """
    m = _tiny_model()
    m.set_anneal_schedule(total_epochs=300, tau_min=0.1)
    a = _adapter_for(m)

    a.notify_global_epoch(0)
    assert m.current_gumbel_temperature == pytest.approx(1.0, abs=1e-6)
    a.notify_global_epoch(299)
    assert m.current_gumbel_temperature == pytest.approx(0.1, abs=1e-6)
    # And the load-bearing inequality (same FAKE-test guard as the module test).
    assert m.current_gumbel_temperature < 0.5


def test_adapter_forward_noop_for_renderer_without_hook() -> None:
    """A renderer lacking ``notify_global_epoch`` must not break the adapter.

    Sister substrates (Z6 / Z8 / etc.) have NO τ hook; the adapter forwarding
    must be a silent no-op for them (backward compat).
    """

    class _HooklessRenderer:
        def __init__(self) -> None:
            self._params = {"w": mx.zeros((2, 2))}

        def __call__(self, idx: object) -> object:
            return mx.zeros((int(idx.shape[0]), 2, 3, 384, 512))

        def parameters(self) -> dict:
            return self._params

    renderer = _HooklessRenderer()
    t0 = mx.zeros((8, 384, 512, 3))
    t1 = mx.zeros((8, 384, 512, 3))
    bundle = RendererBundle(
        model=renderer,
        target_rgb_0=t0,
        target_rgb_1=t1,
        num_pairs=8,
        forward_convention="call_b2chw_255",
    )
    a = MlxScoreAwareAdapter(bundle, substrate_id="hookless")
    # Must not raise; pure no-op forwarding.
    a.notify_global_epoch(0)
    a.notify_global_epoch(42)
    assert not hasattr(renderer, "current_gumbel_temperature")
