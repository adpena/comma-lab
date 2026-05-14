# SPDX-License-Identifier: MIT
"""Round 7 Defect #3 regression — qat_finetune Phase A→B EMA shadow staleness.

Council Round 7 §6.3 identified that ``experiments/qat_finetune.py`` constructs
``ema = EMA(model, decay=cfg.ema_decay)`` at L1084 BEFORE Phase A. During
Phase A the model gains INT8 parametrize keys
(``*.parametrizations.weight.original``, ``*.parametrizations.weight.0.scale_int8``
etc.); EMA's late-bound guard (training.py:385-387) adds them to the shadow
on first ``ema.update()`` call. Between Phase A and Phase B,
``remove_parametrizations(int8_wrapped)`` strips the INT8 keys from the live
model — but the EMA shadow STILL has them. Phase B then adds DIFFERENT
parametrize keys (FP4 / mixed-precision). The next ``ema.apply(model)`` call
calls ``model.load_state_dict(self.shadow)`` — and the canonical EMA at
``training.py:397`` uses default ``strict=True`` so the key mismatch raises
RuntimeError.

The fix (Option A): rebuild ``ema = EMA(model, decay=cfg.ema_decay)`` at
the start of Phase B (immediately after ``remove_parametrizations``). The
Phase A averaging is sacrificed — but Phase A is a warm-up whose value is
in the live model state, not the shadow average.

This test:
  1. Asserts the OLD pre-fix behaviour DID raise on Phase A → Phase B
     transition (regression-proves the bug class).
  2. Asserts the NEW Option A path (rebuild EMA between phases) does
     NOT raise.

Memory:
  - .omx/research/council_round7_adversarial_20260429.md (Defect #3)
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from tac.training import EMA


# ── Tiny INT8/FP4 fake-quant parametrizations (mimic real wrap shape) ──
#
# We intentionally use parametrize.register_parametrization directly so the
# test does not depend on the full apply_int8_fake_quant /
# apply_fp4_fake_quant pipeline (which has its own dependencies on the
# tac.quantization module). The shape we need is "register_parametrization
# adds extra keys to state_dict; remove_parametrizations strips them."


class _IntScaleParam(nn.Module):
    """Tiny parametrize-style int-scale module — adds a single buffer."""

    def __init__(self, weight_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.register_buffer("scale_int8", torch.ones(1))

    def forward(self, w: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return w * self.scale_int8


class _Fp4ScaleParam(nn.Module):
    """Tiny parametrize-style fp4-scale module — adds a different buffer."""

    def __init__(self, weight_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.register_buffer("scale_fp4", torch.ones(1))

    def forward(self, w: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return w * self.scale_fp4


def _make_tiny_model() -> nn.Module:
    """Two-conv tiny model — enough to demonstrate the parametrize lifecycle."""
    m = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.Conv2d(4, 3, kernel_size=3, padding=1),
    )
    return m


def _wrap_with_int8(model: nn.Module) -> list[nn.Module]:
    """Mimic apply_int8_fake_quant: register a parametrization on each conv."""
    wrapped = []
    for layer in model:
        if isinstance(layer, nn.Conv2d):
            parametrize.register_parametrization(
                layer, "weight", _IntScaleParam(layer.weight.shape),
            )
            wrapped.append(layer)
    return wrapped


def _wrap_with_fp4(model: nn.Module) -> list[nn.Module]:
    """Mimic apply_fp4_fake_quant: register a DIFFERENT parametrization."""
    wrapped = []
    for layer in model:
        if isinstance(layer, nn.Conv2d):
            parametrize.register_parametrization(
                layer, "weight", _Fp4ScaleParam(layer.weight.shape),
            )
            wrapped.append(layer)
    return wrapped


def _remove_parametrizations(wrapped: list[nn.Module]) -> None:
    """Mimic remove_parametrizations: undo the parametrize registration."""
    for layer in wrapped:
        if parametrize.is_parametrized(layer, "weight"):
            parametrize.remove_parametrizations(
                layer, "weight", leave_parametrized=True,
            )


# ── Regression tests ───────────────────────────────────────────────────


def test_pre_fix_phase_a_to_b_ema_apply_raises_runtimerror() -> None:
    """Pre-fix behaviour proof: WITHOUT the EMA rebuild, ema.apply() after
    Phase A → Phase B transition crashes with RuntimeError because the
    shadow has Phase A keys and the live model has Phase B keys.

    This test PROVES the bug class exists (regression evidence). The fix
    test below shows Option A eliminates it.
    """
    torch.manual_seed(0)
    model = _make_tiny_model()
    ema = EMA(model, decay=0.997)

    # Phase A: wrap + step + EMA-update so the shadow gains Phase A keys.
    int8_wrapped = _wrap_with_int8(model)
    ema.update(model)  # shadow now has scale_int8 keys
    # Sanity: shadow has Phase A keys.
    has_int8_keys = any("scale_int8" in k for k in ema.shadow)
    assert has_int8_keys, "Phase A wrap should expose scale_int8 buffer"

    # Phase A → Phase B transition: strip int8, add fp4.
    _remove_parametrizations(int8_wrapped)
    _wrap_with_fp4(model)

    # ema.apply now calls model.load_state_dict(self.shadow) with strict=True.
    # The shadow has scale_int8 (no longer in model) AND lacks scale_fp4
    # (which the model now has) → load_state_dict raises.
    with pytest.raises(RuntimeError):
        ema.apply(model)


def test_option_a_rebuild_ema_between_phases_does_not_raise() -> None:
    """The Option A fix: rebuild ema = EMA(model, decay=...) AFTER the
    Phase B wrap. The new shadow is seeded from the FP4-wrapped live
    model whose state_dict already includes the .scale_fp4 keys —
    subsequent ema.update() and ema.apply() calls inside the Phase B
    training loop see exact-key-match.

    The fix in qat_finetune.py inserts (after the FP4-wrap selector):
        ...
        fp4_wrapped = apply_fp4_fake_quant(model, ...)
        ema = EMA(model, decay=cfg.ema_decay)   # ← Option A: AFTER wrap
        optimizer = torch.optim.Adam(...)
        for epoch in range(cfg.fp4_epochs):
            ...
            ema.update(model)
            ...
            ema.apply(model)   # no key mismatch

    Why AFTER wrap, not before: rebuild-before-wrap leaves shadow with
    plain '0.weight' keys, then FP4-wrap mutates the live model's keys
    to '0.parametrizations.weight.original' + '.scale_fp4' — the same
    key-mismatch failure mode just shifted.
    """
    torch.manual_seed(0)
    model = _make_tiny_model()
    ema = EMA(model, decay=0.997)

    # Phase A
    int8_wrapped = _wrap_with_int8(model)
    ema.update(model)

    # Phase A → B transition.
    _remove_parametrizations(int8_wrapped)

    # Phase B wrap FIRST, then rebuild EMA on the now-FP4-wrapped model.
    _wrap_with_fp4(model)
    ema = EMA(model, decay=0.997)  # ← Option A: fresh shadow from FP4 model

    ema.update(model)
    # ema.apply MUST NOT raise — the shadow now matches the live model.
    ema.apply(model)


def test_option_a_phase_b_has_no_phase_a_keys_in_shadow() -> None:
    """Defense in depth: after the AFTER-FP4-wrap rebuild, the shadow
    contains scale_fp4 but NO scale_int8 (Phase A is gone)."""
    torch.manual_seed(0)
    model = _make_tiny_model()
    ema = EMA(model, decay=0.997)

    int8_wrapped = _wrap_with_int8(model)
    ema.update(model)

    _remove_parametrizations(int8_wrapped)
    _wrap_with_fp4(model)
    ema = EMA(model, decay=0.997)  # ← rebuild AFTER FP4 wrap
    ema.update(model)

    has_int8 = any("scale_int8" in k for k in ema.shadow)
    has_fp4 = any("scale_fp4" in k for k in ema.shadow)
    assert not has_int8, (
        f"Option A rebuild should leave NO scale_int8 keys in shadow, "
        f"got: {[k for k in ema.shadow if 'scale_int8' in k]}"
    )
    assert has_fp4, (
        f"Option A rebuild on FP4-wrapped model should seed scale_fp4 "
        f"keys into the shadow, got shadow keys: {list(ema.shadow)}"
    )
