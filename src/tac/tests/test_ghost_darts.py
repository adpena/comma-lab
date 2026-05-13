"""Tests for Lane GH-DARTS — Ghost convolution ratio search.

Verifies:
  * GhostConvVariant accepts fractional ratios (1.5) — relaxes the
    integer guard in the original GhostConv2d.
  * Forward output shape matches a plain Conv2d with the same kwargs.
  * GhostRatioDARTSCell forward + backward populates α grad.
  * Param count of small-ratio variants > param count of big-ratio
    variants (sanity check on the geometry).
  * Synthetic-data convergence: with a regression target tuned to favor
    one specific ratio, α concentrates on that ratio after enough steps
    (KL-to-uniform > 0.5 nats — we don't assert the strict 2.0 nats
    threshold here because the synthetic data is short and we want the
    test to be deterministic + fast; the 2.0 nats check is for the
    actual full-budget search and is asserted in the operator-facing
    convergence diagnostic).
"""

from __future__ import annotations


import pytest
import torch

from tac.contrib.ghost_darts import (
    GHOST_RATIO_CANDIDATES,
    GhostConvVariant,
    GhostRatioDARTSCell,
    build_ghost_arch_optimizer,
    build_ghost_ratio_supernet,
    make_trajectory,
)


# ── GhostConvVariant ────────────────────────────────────────────────────


@pytest.mark.parametrize("ratio", [1.5, 2.0, 2.5, 3.0, 4.0])
def test_ghost_conv_variant_forward_shape(ratio: float):
    op = GhostConvVariant(c_in=6, c_out=36, kernel=3, ratio=ratio, padding=1)
    x = torch.randn(2, 6, 16, 16)
    y = op(x)
    assert y.shape == (2, 36, 16, 16)


def test_ghost_conv_variant_accepts_fractional_ratio():
    """Original GhostConv2d rejects ratio<2; the variant accepts ≥1."""
    op = GhostConvVariant(c_in=6, c_out=36, kernel=3, ratio=1.5, padding=1)
    assert op.ratio == 1.5


def test_ghost_conv_variant_rejects_ratio_below_one():
    with pytest.raises(ValueError):
        GhostConvVariant(c_in=6, c_out=36, kernel=3, ratio=0.5)


def test_ghost_conv_variant_param_count_decreases_with_ratio():
    """Higher ratio → fewer intrinsic channels → fewer primary-conv
    params. The ghost-branch param count is bounded by c_intrinsic so
    the total must be monotone non-increasing in ratio (within the
    integer-ceil discretization)."""
    counts = []
    for r in (1.5, 2.0, 3.0, 4.0):
        op = GhostConvVariant(c_in=6, c_out=36, kernel=3, ratio=r, padding=1)
        counts.append(op.param_count())
    # Strict monotonic for these (well-separated) ratios.
    assert counts[0] > counts[1]
    assert counts[1] > counts[2]
    assert counts[2] >= counts[3]


def test_ghost_conv_variant_stride_handled():
    op = GhostConvVariant(c_in=6, c_out=36, kernel=3, ratio=2.0, stride=2, padding=1)
    x = torch.randn(2, 6, 16, 16)
    y = op(x)
    assert y.shape == (2, 36, 8, 8)


# ── GhostRatioDARTSCell ─────────────────────────────────────────────────


def test_ratio_cell_forward_shape():
    cell = GhostRatioDARTSCell(c_in=6, c_out=36, kernel=3, padding=1)
    x = torch.randn(2, 6, 16, 16)
    y = cell(x)
    assert y.shape == (2, 36, 16, 16)


def test_ratio_cell_alpha_gradient_flows():
    cell = GhostRatioDARTSCell(c_in=6, c_out=36, kernel=3, padding=1)
    x = torch.randn(2, 6, 8, 8)
    cell(x).sum().backward()
    assert cell.alpha.grad is not None
    assert cell.alpha.grad.abs().sum() > 0


def test_ratio_cell_default_candidates_match_module_constant():
    cell = GhostRatioDARTSCell(c_in=6, c_out=36, kernel=3, padding=1)
    assert cell.candidate_ratios == GHOST_RATIO_CANDIDATES


def test_ratio_cell_candidate_param_counts_have_all_names():
    cell = GhostRatioDARTSCell(c_in=6, c_out=36, kernel=3, padding=1)
    counts = cell.candidate_param_counts()
    assert set(counts.keys()) == set(cell.names)
    assert all(v > 0 for v in counts.values())


def test_ratio_cell_rejects_singleton_candidate():
    with pytest.raises(ValueError):
        GhostRatioDARTSCell(c_in=6, c_out=36, kernel=3, candidate_ratios=(2.0,))


def test_ratio_cell_discrete_arch_ratio_is_argmax():
    cell = GhostRatioDARTSCell(c_in=6, c_out=36, kernel=3, padding=1)
    # Bias α toward index 2 (ratio=2.5).
    with torch.no_grad():
        cell.alpha[2] = 5.0
    assert cell.discrete_arch_ratio() == GHOST_RATIO_CANDIDATES[2]


# ── Supernet ────────────────────────────────────────────────────────────


def test_ghost_supernet_forward_shape():
    net = build_ghost_ratio_supernet(c_in=6, widths=(36, 60, 60))
    x = torch.randn(2, 6, 32, 32)
    y = net(x)
    # 32 → 32 (stem) → 16 (down s=2) → 8 (down2 s=2). Output ch = 3.
    assert y.shape == (2, 3, 8, 8)


def test_ghost_supernet_temperature_anneal_synced():
    net = build_ghost_ratio_supernet(c_in=6, widths=(36, 60, 60))
    net.temperature_anneal(epoch=0, total_epochs=10)
    assert net.stem.current_temperature == pytest.approx(5.0)
    assert net.down.current_temperature == pytest.approx(5.0)
    assert net.down2.current_temperature == pytest.approx(5.0)


def test_ghost_arch_optimizer_only_steps_alpha():
    net = build_ghost_ratio_supernet(c_in=6, widths=(36, 60, 60))
    arch_opt = build_ghost_arch_optimizer(net, lr=1e-1)
    # Snapshot weight params.
    weight_snaps = {
        id(p): p.detach().clone()
        for p in net.parameters()
        if id(p) not in {id(net.stem.alpha), id(net.down.alpha), id(net.down2.alpha)}
    }
    alpha_snaps = {
        id(p): p.detach().clone()
        for p in (net.stem.alpha, net.down.alpha, net.down2.alpha)
    }
    x = torch.randn(2, 6, 16, 16)
    net(x).sum().backward()
    arch_opt.step()
    # α moved.
    for p in (net.stem.alpha, net.down.alpha, net.down2.alpha):
        assert not torch.allclose(p.detach(), alpha_snaps[id(p)])
    # No weight param moved.
    for p in net.parameters():
        if id(p) in weight_snaps:
            assert torch.allclose(p.detach(), weight_snaps[id(p)])


# ── Synthetic convergence ───────────────────────────────────────────────


def test_alpha_converges_on_synthetic_target():
    """Construct a synthetic target that variant 1 reproduces EXACTLY
    (its own forward output on a fixed input), and zero out every other
    variant's output. The DARTS gradient must then drive α toward
    variant 1 because it is the only candidate whose output reduces
    the MSE — every other candidate is locked at 0, so any α-mass on
    them increases the residual.

    Math: with y_1 = target and y_{i≠1} = 0,
        loss = ||(Σ w_i · y_i) - t||² = ||w_1 · y_1 - y_1||² = (w_1 - 1)² · ||y_1||²
    so the optimal w_1 → 1, which through softmax means α_1 grows
    without bound (in practice clamped by Adam + the schedule). After
    50 steps with lr=0.1 the search should commit to index 1 with
    KL-to-uniform > 0.5 nats."""
    torch.manual_seed(0)
    cell = GhostRatioDARTSCell(c_in=4, c_out=8, kernel=3, padding=1)
    # Zero out every variant EXCEPT index 1 (set primary weights AND
    # bias to zero so that y_i = 0 for i ≠ 1).
    for i, op in enumerate(cell.ops):
        if i == 1:
            continue
        with torch.no_grad():
            op.primary.weight.zero_()
            if op.primary.bias is not None:
                op.primary.bias.zero_()
    # Compute variant-1's output on a fixed input — that becomes the
    # target the search must commit to.
    x = torch.randn(8, 4, 8, 8)
    with torch.no_grad():
        target = cell.ops[1](x).detach().clone()
    # Train α only (don't touch op weights).
    arch_opt = torch.optim.Adam([cell.alpha], lr=0.1)
    for step in range(80):
        cell.temperature_anneal(epoch=step, total_epochs=80)
        arch_opt.zero_grad()
        loss = ((cell(x) - target) ** 2).mean()
        loss.backward()
        arch_opt.step()
    assert cell.discrete_arch() == 1, (
        f"DARTS should commit to index 1 (the only non-zero variant); "
        f"got argmax={cell.discrete_arch()}, alpha={cell.alpha.detach().tolist()}"
    )
    assert cell.alpha_kl_nats(temperature=1.0) > 0.5


# ── Trajectory ──────────────────────────────────────────────────────────


def test_make_trajectory_uses_ratio_names():
    cell = GhostRatioDARTSCell(c_in=4, c_out=8, kernel=3, padding=1)
    traj = make_trajectory(cell)
    assert tuple(traj.op_names) == cell.names
    assert all(name.startswith("ratio_") for name in traj.op_names)
