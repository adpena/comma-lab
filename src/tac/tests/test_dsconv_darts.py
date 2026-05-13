"""Tests for Lane K-DARTS — DSConv channel-dim search.

Verifies:
  * DSConvVariant forward shape matches the candidate (B, output_ch, H/4, W/4).
  * 16-candidate cell forward + α gradient flow.
  * Per-candidate param count is monotonically increasing in (base, mid).
  * Budget penalty is a 0-D tensor that gradients into α.
  * Budget penalty pushes α MASS away from the over-budget candidate
    after a few steps of synthetic search.
  * Discrete-arch-spec returns a (base_ch, mid_ch) tuple from the candidates.
"""

from __future__ import annotations

import pytest
import torch

from tac.contrib.dsconv_darts import (
    DSCONV_BASE_CHANNELS,
    DSCONV_MID_CHANNELS,
    DSConvChannelDARTSCell,
    DSConvVariant,
    PARAM_BUDGET,
    build_dsconv_arch_optimizer,
    build_dsconv_channel_supernet,
    make_trajectory,
    param_budget_penalty,
)


# ── DSConvVariant ───────────────────────────────────────────────────────


def test_dsconv_variant_forward_shape():
    op = DSConvVariant(c_in=6, base_ch=24, mid_ch=32, output_ch=32)
    x = torch.randn(2, 6, 32, 32)
    y = op(x)
    # Stem stride 1 → 32; down1 stride 2 → 16; down2 stride 2 → 8.
    assert y.shape == (2, 32, 8, 8)


def test_dsconv_variant_param_count_increases_with_dims():
    """Bigger base + mid → more params, monotonically."""
    p_small = DSConvVariant(c_in=6, base_ch=16, mid_ch=24, output_ch=32).param_count()
    p_med = DSConvVariant(c_in=6, base_ch=24, mid_ch=32, output_ch=32).param_count()
    p_large = DSConvVariant(c_in=6, base_ch=48, mid_ch=64, output_ch=32).param_count()
    assert p_small < p_med < p_large


# ── DSConvChannelDARTSCell ──────────────────────────────────────────────


def test_cell_has_16_candidates():
    cell = DSConvChannelDARTSCell(c_in=6, output_ch=32)
    assert len(cell.ops) == 16
    assert cell.alpha.numel() == 16


def test_cell_forward_shape_matches_a_single_candidate():
    cell = DSConvChannelDARTSCell(c_in=6, output_ch=32)
    x = torch.randn(2, 6, 32, 32)
    y = cell(x)
    assert y.shape == (2, 32, 8, 8)


def test_cell_alpha_gradient_flows():
    cell = DSConvChannelDARTSCell(c_in=6, output_ch=32)
    x = torch.randn(2, 6, 16, 16)
    cell(x).sum().backward()
    assert cell.alpha.grad is not None
    assert cell.alpha.grad.abs().sum() > 0


def test_cell_default_candidate_specs():
    cell = DSConvChannelDARTSCell(c_in=6, output_ch=32)
    expected = {(b, m) for b in DSCONV_BASE_CHANNELS for m in DSCONV_MID_CHANNELS}
    assert set(cell.candidate_specs) == expected


def test_cell_rejects_singleton_axis():
    with pytest.raises(ValueError):
        DSConvChannelDARTSCell(c_in=6, output_ch=32, base_channels=(24,))
    with pytest.raises(ValueError):
        DSConvChannelDARTSCell(c_in=6, output_ch=32, mid_channels=(32,))


def test_cell_discrete_arch_spec_returns_tuple():
    cell = DSConvChannelDARTSCell(c_in=6, output_ch=32)
    # Bias α toward the largest-dim candidate (base=48, mid=64 → last index).
    with torch.no_grad():
        cell.alpha[-1] = 100.0
    spec = cell.discrete_arch_spec()
    assert spec == (DSCONV_BASE_CHANNELS[-1], DSCONV_MID_CHANNELS[-1])


def test_expected_param_count_starts_at_mean():
    cell = DSConvChannelDARTSCell(c_in=6, output_ch=32)
    # α=0 → uniform mixture → expected = mean of candidate counts.
    counts = [op.param_count() for op in cell.ops]
    expected = cell.expected_param_count().item()
    assert expected == pytest.approx(sum(counts) / len(counts), rel=1e-5)


def test_expected_param_count_is_differentiable_wrt_alpha():
    cell = DSConvChannelDARTSCell(c_in=6, output_ch=32)
    cell.alpha.grad = None
    cell.expected_param_count().backward()
    assert cell.alpha.grad is not None
    assert cell.alpha.grad.abs().sum() > 0


# ── Budget penalty ──────────────────────────────────────────────────────


def test_budget_penalty_zero_when_under_budget():
    """If every candidate is under-budget (small base/mid), penalty=0."""
    cell = DSConvChannelDARTSCell(
        c_in=6, output_ch=32, base_channels=(16, 24), mid_channels=(24, 32),
    )
    pen = param_budget_penalty(cell, budget=PARAM_BUDGET)
    assert pen.item() == pytest.approx(0.0, abs=1e-9)


def test_budget_penalty_positive_when_over_budget():
    cell = DSConvChannelDARTSCell(c_in=6, output_ch=32)
    # Bias α toward biggest candidate → expected count exceeds budget for a
    # tight budget like 5K.
    with torch.no_grad():
        cell.alpha[-1] = 100.0
    pen = param_budget_penalty(cell, budget=5_000, weight=1.0)
    assert pen.item() > 0.0


def test_budget_penalty_pushes_alpha_off_overbudget_candidate():
    """Run a few synthetic SGD steps with ONLY the budget penalty as
    loss. α-mass on the biggest-spec candidate must decrease.

    The hinge applies to the *expected* (softmax-mixture) param count,
    so the budget must be tighter than the uniform-mixture mean for the
    gradient to be non-zero at α=0. Per-candidate counts span
    ~2448-8864; uniform mean ≈ 4976. Setting budget=3000 makes the
    penalty active from step 0, with the gradient pushing α-mass away
    from the biggest candidates (which contribute most to the expected
    count) and toward the smallest."""
    torch.manual_seed(0)
    cell = DSConvChannelDARTSCell(c_in=6, output_ch=32)
    biggest_idx = len(cell.ops) - 1
    arch_opt = torch.optim.Adam([cell.alpha], lr=0.5)

    initial_mass = cell.alpha_softmax_distribution(temperature=1.0)[biggest_idx].item()
    for step in range(20):
        cell.temperature_anneal(epoch=step, total_epochs=20)
        arch_opt.zero_grad()
        # Budget tight enough that the EXPECTED (mixture) count is well over.
        loss = param_budget_penalty(cell, budget=3_000, weight=1e-3)
        loss.backward()
        arch_opt.step()
    final_mass = cell.alpha_softmax_distribution(temperature=1.0)[biggest_idx].item()
    assert final_mass < initial_mass, (
        f"budget penalty failed to reduce α-mass on biggest candidate; "
        f"initial={initial_mass:.4f}, final={final_mass:.4f}, "
        f"alpha={cell.alpha.detach().tolist()}"
    )


# ── Supernet + arch optimizer ───────────────────────────────────────────


def test_supernet_forward_shape():
    net = build_dsconv_channel_supernet(c_in=6, output_ch=32)
    x = torch.randn(2, 6, 32, 32)
    y = net(x)
    assert y.shape == (2, 3, 8, 8)


def test_supernet_temperature_anneal():
    net = build_dsconv_channel_supernet(c_in=6, output_ch=32)
    net.temperature_anneal(epoch=0, total_epochs=10)
    assert net.cell.current_temperature == pytest.approx(5.0)
    net.temperature_anneal(epoch=9, total_epochs=10)
    assert net.cell.current_temperature == pytest.approx(0.1)


def test_arch_optimizer_only_steps_alpha():
    net = build_dsconv_channel_supernet(c_in=6, output_ch=32)
    arch_opt = build_dsconv_arch_optimizer(net, lr=1e-1)
    weight_snaps = {
        id(p): p.detach().clone()
        for p in net.parameters() if id(p) != id(net.cell.alpha)
    }
    alpha_snap = net.cell.alpha.detach().clone()

    x = torch.randn(2, 6, 16, 16)
    net(x).sum().backward()
    arch_opt.step()

    assert not torch.allclose(net.cell.alpha.detach(), alpha_snap)
    for p in net.parameters():
        if id(p) in weight_snaps:
            assert torch.allclose(p.detach(), weight_snaps[id(p)])


def test_make_trajectory_includes_all_specs():
    cell = DSConvChannelDARTSCell(c_in=6, output_ch=32)
    traj = make_trajectory(cell)
    assert len(traj.op_names) == 16
    assert all("base_" in name and "_mid_" in name for name in traj.op_names)
