"""Synthetic tests for paradigm ε — Self-Compressing NN joint width × precision.

All tests are CPU-only; no scorer load; no GPU. Math/integration only.

References
----------
- Module: src/tac/self_compressing_nn.py
- Paper: Wang et al. 2023 arXiv:2301.13142
- Council: .omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md §4.2 ε
"""
from __future__ import annotations

import math

import pytest
import torch

from tac.self_compressing_nn import (
    SCNN_FRAMEWORK_VERSION,
    JointBitsAndWidthAccountant,
    LearnableChannelGate,
    SCNNRateScheduler,
    expected_active_channels,
    expected_total_bits,
    hard_threshold_gates,
    joint_compress_loss,
)


# ── LearnableChannelGate ───────────────────────────────────────────────


def test_gate_init_validates_args():
    with pytest.raises(ValueError, match="n_channels"):
        LearnableChannelGate(n_channels=0)
    with pytest.raises(ValueError, match="temperature"):
        LearnableChannelGate(n_channels=4, temperature=0.0)


def test_gate_init_log_alpha_default_active():
    """Default init_log_alpha=2.0 → most channels active in expectation."""
    gate = LearnableChannelGate(n_channels=64)
    p_active = gate.expected_active()
    assert p_active.shape == (64,)
    # Default 2.0 with hard concrete defaults → should be > 0.85 expected active
    assert p_active.mean().item() > 0.85


def test_gate_protected_channels_force_on():
    protected = torch.zeros(8, dtype=torch.bool)
    protected[0:3] = True
    gate = LearnableChannelGate(
        n_channels=8, init_log_alpha=-10.0, protected=protected
    )
    # First 3 channels are protected — expected active = 1 exactly
    p_active = gate.expected_active()
    assert torch.allclose(p_active[0:3], torch.ones(3))
    # Other channels are deeply suppressed
    assert p_active[3:].max().item() < 0.1


def test_gate_eval_is_deterministic():
    gate = LearnableChannelGate(n_channels=16, init_log_alpha=2.0)
    gate.eval()
    out1 = gate.forward()
    out2 = gate.forward()
    assert torch.equal(out1, out2)
    # All ones because init_log_alpha=2.0 > 0
    assert torch.equal(out1, torch.ones(16))


def test_gate_train_is_stochastic():
    torch.manual_seed(0)
    gate = LearnableChannelGate(n_channels=64, init_log_alpha=0.0)
    gate.train()
    s1 = gate.forward()
    s2 = gate.forward()
    # Two different stochastic samples
    assert not torch.equal(s1, s2)
    # Values are in [0, 1]
    assert s1.min().item() >= 0.0 and s1.max().item() <= 1.0


def test_gate_l0_penalty_is_differentiable():
    gate = LearnableChannelGate(n_channels=8, init_log_alpha=0.5)
    pen = gate.l0_penalty()
    pen.backward()
    assert gate.log_alpha.grad is not None
    # Gradient should be non-zero (driven by sigmoid derivative)
    assert gate.log_alpha.grad.abs().sum().item() > 0.0


def test_gate_protected_mask_shape_validation():
    with pytest.raises(ValueError, match="protected mask shape"):
        LearnableChannelGate(
            n_channels=4, protected=torch.zeros(5, dtype=torch.bool)
        )


def test_gate_hard_count_matches_threshold():
    log_alpha = torch.tensor([1.0, -1.0, 2.0, -3.0, 0.5])
    gate = LearnableChannelGate(n_channels=5, init_log_alpha=0.0)
    with torch.no_grad():
        gate.log_alpha.copy_(log_alpha)
    # Hard threshold: log_alpha > 0 → 3 channels (1.0, 2.0, 0.5)
    assert gate.hard_count() == 3


# ── JointBitsAndWidthAccountant ─────────────────────────────────────────


def test_accountant_register_validates():
    acc = JointBitsAndWidthAccountant()
    with pytest.raises(ValueError, match="name must be"):
        acc.register(name="", n_out_channels=4, n_weights_per_out_channel=9)
    with pytest.raises(ValueError, match="n_out_channels"):
        acc.register(name="L1", n_out_channels=0, n_weights_per_out_channel=9)
    with pytest.raises(ValueError, match="n_weights_per_out_channel"):
        acc.register(name="L1", n_out_channels=4, n_weights_per_out_channel=0)
    # Duplicate name
    acc.register(name="L1", n_out_channels=4, n_weights_per_out_channel=9)
    with pytest.raises(ValueError, match="duplicate"):
        acc.register(name="L1", n_out_channels=4, n_weights_per_out_channel=9)


def test_accountant_no_layers_returns_zero():
    acc = JointBitsAndWidthAccountant()
    assert acc.expected_active_channels().item() == 0.0
    assert acc.expected_total_bits().item() == 0.0


def test_accountant_no_gate_no_bits_uses_full_8bit():
    """Layer with neither gate nor bits → all channels active, 8 bits each."""
    acc = JointBitsAndWidthAccountant()
    acc.register(name="L1", n_out_channels=4, n_weights_per_out_channel=9)
    # 4 channels × 8 bits × 9 weights/ch = 288 bits
    assert acc.expected_total_bits().item() == pytest.approx(288.0)
    assert acc.expected_active_channels().item() == 4.0


def test_accountant_gate_only_reduces_bits_proportionally():
    """Gate that suppresses 50% of channels reduces bits by ≈50%."""
    torch.manual_seed(0)
    gate = LearnableChannelGate(n_channels=8, init_log_alpha=0.0)
    acc = JointBitsAndWidthAccountant()
    acc.register(
        name="L1",
        n_out_channels=8,
        n_weights_per_out_channel=27,
        gate_module=gate,
    )
    # init_log_alpha=0.0 → P(active) = sigmoid(0 - β·log(-γ/ζ))
    # With γ=-0.1, ζ=1.1, β=2/3: log_ratio = log(0.1/1.1) = -2.3979
    # P(active) = sigmoid(0 - (2/3)*(-2.3979)) = sigmoid(1.5986) ≈ 0.832
    expected = 8.0 * 0.832 * 8.0 * 27.0  # 8 ch · prob · 8 bits · 27 weights
    actual = acc.expected_total_bits().item()
    assert actual == pytest.approx(expected, rel=0.05)


def test_accountant_bits_module_per_channel_used():
    """Bits module returning per-CHANNEL tensor is consumed correctly."""
    import torch.nn as nn

    class _PerChannelBits(nn.Module):
        def __init__(self, bits_vec):
            super().__init__()
            self._bits = nn.Parameter(bits_vec.clone())

        def bits_used(self):
            return self._bits

    bits_vec = torch.tensor([2.0, 4.0, 6.0, 8.0])
    bits_mod = _PerChannelBits(bits_vec)
    acc = JointBitsAndWidthAccountant()
    acc.register(
        name="L1",
        n_out_channels=4,
        n_weights_per_out_channel=10,
        bits_module=bits_mod,  # type: ignore[arg-type]
    )
    # 4 channels active × bits × 10 weights/ch = 10 · (2+4+6+8) = 200
    assert acc.expected_total_bits().item() == pytest.approx(200.0)


def test_accountant_layer_names_preserve_registration_order():
    acc = JointBitsAndWidthAccountant()
    acc.register(name="L1", n_out_channels=4, n_weights_per_out_channel=9)
    acc.register(name="L2", n_out_channels=8, n_weights_per_out_channel=18)
    acc.register(name="L3", n_out_channels=16, n_weights_per_out_channel=27)
    assert acc.layer_names() == ["L1", "L2", "L3"]


def test_accountant_hard_summary_structure():
    gate = LearnableChannelGate(n_channels=8, init_log_alpha=2.0)
    acc = JointBitsAndWidthAccountant()
    acc.register(
        name="L1",
        n_out_channels=8,
        n_weights_per_out_channel=27,
        gate_module=gate,
    )
    summary = acc.hard_summary()
    assert "L1" in summary
    assert summary["L1"]["n_total_channels"] == 8
    # All channels active under threshold (init_log_alpha=2.0 > 0)
    assert summary["L1"]["n_active_channels"] == 8
    assert summary["L1"]["weights_per_out_channel"] == 27


# ── joint_compress_loss ─────────────────────────────────────────────────


def test_joint_compress_loss_validates():
    acc = JointBitsAndWidthAccountant()
    acc.register(name="L1", n_out_channels=4, n_weights_per_out_channel=9)
    with pytest.raises(ValueError, match="lambda_bits"):
        joint_compress_loss(
            task_loss=torch.tensor(1.0),
            accountant=acc,
            lambda_bits=-1.0,
        )
    with pytest.raises(TypeError, match="task_loss must be"):
        joint_compress_loss(
            task_loss=1.0,  # type: ignore[arg-type]
            accountant=acc,
            lambda_bits=0.001,
        )
    with pytest.raises(ValueError, match="task_loss must be scalar"):
        joint_compress_loss(
            task_loss=torch.tensor([1.0, 2.0]),
            accountant=acc,
            lambda_bits=0.001,
        )


def test_joint_compress_loss_linear_rate_term():
    """Linear λ·bits adds correctly to task loss."""
    acc = JointBitsAndWidthAccountant()
    acc.register(name="L1", n_out_channels=4, n_weights_per_out_channel=9)
    # No gate, no bits → 288 bits
    task = torch.tensor(0.5)
    total, parts = joint_compress_loss(
        task_loss=task, accountant=acc, lambda_bits=0.001
    )
    # 0.5 + 0.001 · 288 = 0.788
    assert total.item() == pytest.approx(0.788, abs=1e-6)
    assert parts["task"].item() == 0.5
    assert parts["bits"].item() == 288.0
    assert parts["rate_term"].item() == pytest.approx(0.288, abs=1e-6)


def test_joint_compress_loss_target_hinge_zero_below_target():
    """target_bits hinge: bits below target → rate term zero."""
    acc = JointBitsAndWidthAccountant()
    acc.register(name="L1", n_out_channels=4, n_weights_per_out_channel=9)
    # 288 bits actual; target 500 → hinge max(0, 288-500)^2 = 0
    total, parts = joint_compress_loss(
        task_loss=torch.tensor(0.5),
        accountant=acc,
        lambda_bits=0.001,
        target_bits=500.0,
    )
    assert parts["rate_term"].item() == 0.0
    assert total.item() == pytest.approx(0.5, abs=1e-6)


def test_joint_compress_loss_target_hinge_squared_above_target():
    """target_bits hinge: bits above target → rate term = λ·(excess)²."""
    acc = JointBitsAndWidthAccountant()
    acc.register(name="L1", n_out_channels=4, n_weights_per_out_channel=9)
    # 288 actual; target 100 → excess 188; hinge = 0.001 · 188² = 35.344
    total, parts = joint_compress_loss(
        task_loss=torch.tensor(0.5),
        accountant=acc,
        lambda_bits=0.001,
        target_bits=100.0,
    )
    assert parts["rate_term"].item() == pytest.approx(0.001 * 188.0 ** 2, abs=1e-3)


def test_joint_compress_loss_target_bits_validates():
    acc = JointBitsAndWidthAccountant()
    acc.register(name="L1", n_out_channels=4, n_weights_per_out_channel=9)
    with pytest.raises(ValueError, match="target_bits"):
        joint_compress_loss(
            task_loss=torch.tensor(0.5),
            accountant=acc,
            lambda_bits=0.001,
            target_bits=-1.0,
        )


def test_joint_compress_loss_gradient_flows_to_gate():
    """Backprop through joint loss reaches LearnableChannelGate.log_alpha."""
    gate = LearnableChannelGate(n_channels=4, init_log_alpha=0.0)
    acc = JointBitsAndWidthAccountant()
    acc.register(
        name="L1",
        n_out_channels=4,
        n_weights_per_out_channel=9,
        gate_module=gate,
    )
    # Use bits=8 (no bits module) and a fake task loss that depends on no params
    task = torch.tensor(1.0, requires_grad=False)
    total, _ = joint_compress_loss(
        task_loss=task, accountant=acc, lambda_bits=0.001
    )
    total.backward()
    assert gate.log_alpha.grad is not None
    assert gate.log_alpha.grad.abs().sum().item() > 0.0


# ── Functional aliases ────────────────────────────────────────────────


def test_functional_aliases_match_methods():
    gate = LearnableChannelGate(n_channels=4, init_log_alpha=0.5)
    acc = JointBitsAndWidthAccountant()
    acc.register(
        name="L1",
        n_out_channels=4,
        n_weights_per_out_channel=9,
        gate_module=gate,
    )
    assert torch.equal(
        expected_active_channels(accountant=acc), acc.expected_active_channels()
    )
    assert torch.equal(
        expected_total_bits(accountant=acc), acc.expected_total_bits()
    )
    h = hard_threshold_gates(accountant=acc)
    assert "L1" in h
    assert h["L1"].shape == (4,)


# ── SCNNRateScheduler ──────────────────────────────────────────────────


def test_scheduler_validates():
    with pytest.raises(ValueError, match="peak"):
        SCNNRateScheduler(peak=-0.1, warmup_epochs=10, hold_epochs=10, cool_epochs=10)
    with pytest.raises(ValueError, match="warmup_epochs"):
        SCNNRateScheduler(peak=0.001, warmup_epochs=-1, hold_epochs=10, cool_epochs=10)
    with pytest.raises(ValueError, match="at least one"):
        SCNNRateScheduler(peak=0.001, warmup_epochs=0, hold_epochs=0, cool_epochs=0)


def test_scheduler_phases_correct():
    s = SCNNRateScheduler(
        peak=1.0, warmup_epochs=10, hold_epochs=10, cool_epochs=10
    )
    # Warmup: linear ramp 0→1 over 10 epochs
    assert s.lambda_at(0) == pytest.approx(0.0)
    assert s.lambda_at(5) == pytest.approx(0.5)
    # Hold: at peak
    assert s.lambda_at(10) == pytest.approx(1.0)
    assert s.lambda_at(15) == pytest.approx(1.0)
    # Cool: cosine 1→0 over 10 epochs
    assert s.lambda_at(20) == pytest.approx(1.0)  # cool_pos=0 → cos(0)=1 → 0.5*1*(1+1)=1
    assert s.lambda_at(25) == pytest.approx(0.5, abs=1e-6)  # cool_pos=5 → cos(π/2)=0 → 0.5
    assert s.lambda_at(30) == pytest.approx(0.0, abs=1e-6)  # post-cycle
    # Post-cycle non-cyclic: 0
    assert s.lambda_at(50) == pytest.approx(0.0)


def test_scheduler_cyclic_repeats():
    s = SCNNRateScheduler(
        peak=1.0,
        warmup_epochs=10,
        hold_epochs=10,
        cool_epochs=10,
        cyclic=True,
    )
    # epoch 30 wraps to 0 → 0.0
    assert s.lambda_at(30) == pytest.approx(0.0)
    # epoch 35 wraps to 5 → 0.5
    assert s.lambda_at(35) == pytest.approx(0.5)


def test_scheduler_negative_epoch_rejected():
    s = SCNNRateScheduler(peak=1.0, warmup_epochs=10, hold_epochs=10, cool_epochs=10)
    with pytest.raises(ValueError, match="epoch"):
        s.lambda_at(-1)


# ── Smoke / integration ────────────────────────────────────────────────


def test_framework_version_is_int():
    assert isinstance(SCNN_FRAMEWORK_VERSION, int)
    assert SCNN_FRAMEWORK_VERSION >= 1


def test_full_train_step_integration():
    """One full training step: bits + gate jointly minimized."""
    torch.manual_seed(0)
    gate = LearnableChannelGate(n_channels=8, init_log_alpha=2.0)
    acc = JointBitsAndWidthAccountant()
    acc.register(
        name="L1",
        n_out_channels=8,
        n_weights_per_out_channel=9,
        gate_module=gate,
    )

    # Synthetic task loss = constant (proxy for renderer eval)
    optimizer = torch.optim.Adam(gate.parameters(), lr=0.1)
    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        # Deliberately constant task loss — only rate gradient flows
        task = torch.tensor(0.0)
        total, parts = joint_compress_loss(
            task_loss=task, accountant=acc, lambda_bits=0.01
        )
        total.backward()
        optimizer.step()
        losses.append(parts["bits"].item())
    # Expected bits should monotonically decrease (gate suppression)
    assert losses[-1] < losses[0] * 0.95
