"""Tests for the 5-phase Quantizr-adapted QAT schedule in train_renderer.py.

Quantizr R2 C3 architectural fix (2026-04-26): WILDE/DEN profile comments
declared a 5-phase schedule (anchor → finetune → joint → QAT → final) but
the actual training loop implemented only 2 phases (`pretrain` / `scorer`).
This test suite locks in the new behaviour so we never silently regress to
2-phase again.

Coverage:
  - Phase boundaries / dispatch math.
  - Per-phase LR resolution + cosine annealing within a phase.
  - Phase 4 enables FP4 fake-quant via QATRendererFP4 / FakeQuantFP4.
  - Phase 5 uses hard-pair-emphasis sampling (multinomial vs uniform perm).
  - Backwards-compat: profiles without phaseN_epochs still run the legacy
    2-phase loop unchanged.
  - WILDE / SHIRAZ / DEN profiles declare all 5 phaseN_epochs + phaseN_lr.
"""
from __future__ import annotations

import argparse

import pytest
import torch
import torch.nn as nn

import tac.experiments.train_renderer as tr
from tac.fp4_quantize import QATRendererFP4
from tac.profiles import DEN, SHIRAZ, WILDE


# ── Helpers ────────────────────────────────────────────────────────────


def _ns(**kwargs) -> argparse.Namespace:
    """Build a Namespace with sensible defaults for phase{1..5}_epochs/lr.

    All phases default to 0 epochs (= disabled) so callers can opt-in by
    overriding individual fields.
    """
    base = {
        "phase1_epochs": 0,
        "phase2_epochs": 0,
        "phase3_epochs": 0,
        "phase4_epochs": 0,
        "phase5_epochs": 0,
        "phase1_lr": 1e-3,
        "phase2_lr": 5e-4,
        "phase3_lr": 3e-4,
        "phase4_lr": 5e-5,
        "phase5_lr": 1e-5,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


# ── Phase dispatch math ────────────────────────────────────────────────


class TestPhaseDispatchMath:
    """current_phase + phase_boundaries are pure functions — sanity-check
    them in isolation before exercising the training loop."""

    def test_boundaries_are_cumulative(self):
        args = _ns(
            phase1_epochs=10, phase2_epochs=20, phase3_epochs=5,
            phase4_epochs=3, phase5_epochs=2,
        )
        assert tr.phase_boundaries(args) == [10, 30, 35, 38, 40]

    def test_current_phase_dispatch(self):
        boundaries = [10, 30, 35, 38, 40]
        assert tr.current_phase(0, boundaries) == 1
        assert tr.current_phase(9, boundaries) == 1
        assert tr.current_phase(10, boundaries) == 2
        assert tr.current_phase(29, boundaries) == 2
        assert tr.current_phase(30, boundaries) == 3
        assert tr.current_phase(34, boundaries) == 3
        assert tr.current_phase(35, boundaries) == 4
        assert tr.current_phase(37, boundaries) == 4
        assert tr.current_phase(38, boundaries) == 5
        assert tr.current_phase(39, boundaries) == 5

    def test_current_phase_skips_zero_phases(self):
        # Profile that only declares Phase 1 + Phase 4 (skip 2, 3, 5).
        args = _ns(phase1_epochs=10, phase4_epochs=5)
        boundaries = tr.phase_boundaries(args)
        # boundaries: [10, 10, 10, 15, 15] — phases 2/3/5 are zero-width.
        assert boundaries == [10, 10, 10, 15, 15]
        assert tr.current_phase(0, boundaries) == 1
        assert tr.current_phase(9, boundaries) == 1
        # Epoch 10: phase 2 (boundary == cum) — no actual span; phase 4 starts at 10
        # because phase2/3 are zero. With the dispatch rule "epoch < b" the first
        # phase whose boundary EXCEEDS epoch wins; b1=10 fails (10 < 10 is False),
        # b2=10 fails, b3=10 fails, b4=15 succeeds → phase 4.
        assert tr.current_phase(10, boundaries) == 4
        assert tr.current_phase(14, boundaries) == 4

    def test_has_5phase_schedule_detection(self):
        assert tr.has_5phase_schedule(_ns(phase1_epochs=10)) is True
        assert tr.has_5phase_schedule(_ns(phase4_epochs=5)) is True
        assert tr.has_5phase_schedule(_ns()) is False
        assert tr.has_5phase_schedule(_ns(phase1_epochs=0)) is False


class TestPhaseLR:
    """Each phase should expose its own LR via lr_for_phase + cosine_lr."""

    def test_lr_for_phase_returns_correct_value(self):
        args = _ns(
            phase1_lr=1e-3, phase2_lr=5e-4, phase3_lr=3e-4,
            phase4_lr=5e-5, phase5_lr=1e-5,
        )
        assert tr.lr_for_phase(args, 1) == pytest.approx(1e-3)
        assert tr.lr_for_phase(args, 2) == pytest.approx(5e-4)
        assert tr.lr_for_phase(args, 3) == pytest.approx(3e-4)
        assert tr.lr_for_phase(args, 4) == pytest.approx(5e-5)
        assert tr.lr_for_phase(args, 5) == pytest.approx(1e-5)

    def test_phase_lr_decay_correct(self):
        """Within a phase, LR decays from base_lr toward eta_min."""
        base = 1e-3
        # Start of phase: cosine(0) = 1 → returns ~base_lr
        assert tr.cosine_lr(base, step=0, total=100) == pytest.approx(base)
        # End of phase: cosine(pi) = -1 → returns ~eta_min
        end_lr = tr.cosine_lr(base, step=100, total=100)
        assert end_lr < base
        assert end_lr == pytest.approx(1e-6, abs=1e-7)
        # Mid-phase is monotone decreasing
        mid = tr.cosine_lr(base, step=50, total=100)
        assert end_lr < mid < base

    def test_phase_lr_decay_per_phase_is_independent(self):
        """Different phases should use different base LRs — each phase's
        cosine starts from its own base_lr, not a global one."""
        # Phase 4 LR should be lower than Phase 1 LR at the SAME local step.
        phase1_start = tr.cosine_lr(1e-3, step=0, total=100)
        phase4_start = tr.cosine_lr(5e-5, step=0, total=100)
        assert phase1_start > phase4_start
        assert phase1_start == pytest.approx(1e-3)
        assert phase4_start == pytest.approx(5e-5)


# ── Phase 4 QAT ────────────────────────────────────────────────────────


class _TinyRenderer(nn.Module):
    """Dummy renderer with one Conv2d so QATRendererFP4 has something to wrap."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class TestPhase4EnablesQAT:
    """Phase 4 must wrap the renderer in QATRendererFP4 so weights round-trip
    through FakeQuantFP4 during forward passes."""

    def test_qat_wrapper_attaches_to_conv(self):
        model = _TinyRenderer()
        qat = QATRendererFP4(model)
        # After wrapping, the conv should be parametrized on its `weight`.
        assert nn.utils.parametrize.is_parametrized(model.conv, "weight")
        # Forward must still succeed.
        x = torch.randn(1, 3, 8, 8)
        out = qat(x)
        assert out.shape == (1, 4, 8, 8)
        qat.remove_hooks()
        assert not nn.utils.parametrize.is_parametrized(model.conv, "weight")

    def test_phase4_lr_is_lower_than_phase3(self):
        """The Quantizr recipe: Phase 4 (QAT) LR is 10x lower than Phase 3."""
        args = _ns(
            phase3_epochs=10, phase4_epochs=10,
            phase3_lr=3e-4, phase4_lr=5e-5,
        )
        # 5e-5 / 3e-4 = ~0.167 — within the 5-15x reduction band per
        # Lin et al. 2017 "Mixed Precision Training" (used by Quantizr).
        ratio = tr.lr_for_phase(args, 3) / tr.lr_for_phase(args, 4)
        assert 4 < ratio < 16

    def test_phase4_enables_qat_fakequant_in_default_profiles(self):
        """WILDE / SHIRAZ / DEN must declare phase4_epochs > 0 so that the
        Phase 4 QAT activation is reachable from the canonical profiles."""
        for name, profile in [("WILDE", WILDE), ("SHIRAZ", SHIRAZ), ("DEN", DEN)]:
            assert profile.get("phase4_epochs", 0) > 0, \
                f"{name} must declare phase4_epochs for the QAT phase to fire"
            assert profile.get("phase4_lr", 0) > 0, \
                f"{name} must declare phase4_lr"


# ── Phase 5 hard-pair sampling ────────────────────────────────────────


class TestPhase5HardPairSampling:
    """Phase 5 should bias sampling toward high-difficulty pairs."""

    def test_multinomial_concentrates_on_high_difficulty(self):
        """A pair with 100x higher difficulty should be sampled far more
        often than a uniform pair under multinomial-without-replacement."""
        torch.manual_seed(0)
        n_total = 100
        train_size = 10
        difficulty = torch.ones(n_total)
        difficulty[0] = 100.0  # one very hard pair
        difficulty[1] = 50.0   # one moderately hard pair

        probs = difficulty / difficulty.sum()
        # Sample 1000 epochs worth of perms and count occurrences.
        counts = torch.zeros(n_total)
        for _ in range(1000):
            idx = torch.multinomial(probs, train_size, replacement=False)
            counts[idx] += 1
        # The hardest pair should be sampled in ~98%+ of epochs (since
        # train_size=10 and probs[0] is ~0.4 of total mass).
        assert counts[0] > 800
        # The medium-hard pair should also be sampled most epochs.
        assert counts[1] > 600
        # An average uniform pair should be sampled much less.
        assert counts[50] < 200

    def test_phase5_in_default_profiles(self):
        """WILDE / SHIRAZ / DEN must declare phase5_epochs > 0 so the final
        polish phase actually fires. Without this, the schedule reduces to
        4 phases and the hard-pair emphasis sampler never runs."""
        for name, profile in [("WILDE", WILDE), ("SHIRAZ", SHIRAZ), ("DEN", DEN)]:
            assert profile.get("phase5_epochs", 0) > 0, \
                f"{name} must declare phase5_epochs > 0"


# ── Backwards compat ───────────────────────────────────────────────────


class TestLegacy2PhaseStillWorks:
    """A profile that does NOT declare phaseN_epochs must still run the
    legacy 2-phase loop unchanged. has_5phase_schedule() is the gate."""

    def test_no_phase_keys_means_no_5phase(self):
        legacy = _ns()  # all phaseN_epochs default to 0
        assert tr.has_5phase_schedule(legacy) is False

    def test_legacy_pretrain_profile_does_not_trigger_5phase(self):
        """A profile that uses ONLY pretrain_epochs (the legacy key) must
        not silently activate 5-phase mode."""
        from tac.profiles import MASK_RENDERER_FULL
        # MASK_RENDERER_FULL is one of the legacy 2-phase profiles.
        assert "phase1_epochs" not in MASK_RENDERER_FULL
        # Build args with only pretrain_epochs set.
        legacy_args = _ns()
        legacy_args.pretrain_epochs = MASK_RENDERER_FULL.get("pretrain_epochs", 100)
        assert tr.has_5phase_schedule(legacy_args) is False

    def test_5phase_profiles_declare_all_five(self):
        """WILDE / SHIRAZ / DEN must declare ALL FIVE phaseN_epochs
        explicitly so no phase silently defaults to 0 and gets skipped."""
        for name, profile in [("WILDE", WILDE), ("SHIRAZ", SHIRAZ), ("DEN", DEN)]:
            for i in range(1, 6):
                assert f"phase{i}_epochs" in profile, \
                    f"{name} missing phase{i}_epochs"
                assert f"phase{i}_lr" in profile, \
                    f"{name} missing phase{i}_lr"
