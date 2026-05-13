"""Tests for Lane I-DARTS — Cool-Chic dim search.

Verifies:
  * CoolChicVariant forward shape matches CoolChicLatentRenderer's
    ``(B, 3, H, W)`` contract.
  * 16-candidate cell forward + α gradient flow.
  * Discrete-arch-spec returns a (hidden_dim, latent_grid) pair.
  * Param count of small-hidden / small-grid candidates is strictly less
    than big-hidden / big-grid candidates.
"""

from __future__ import annotations

import pytest
import torch

from tac.contrib.coolchic_darts import (
    COOLCHIC_HIDDEN_CANDIDATES,
    COOLCHIC_LATENT_GRID_CANDIDATES,
    CoolChicDARTSCell,
    CoolChicVariant,
    build_coolchic_arch_optimizer,
    build_coolchic_supernet,
    make_trajectory,
)


# ── CoolChicVariant ─────────────────────────────────────────────────────


def test_coolchic_variant_forward_shape():
    op = CoolChicVariant(hidden=16, latent_shapes=((6, 8), (3, 4), (2, 3)))
    masks = torch.randint(low=0, high=5, size=(2, 24, 32))
    y = op(masks)
    assert y.shape == (2, 3, 24, 32)


def test_coolchic_variant_param_count_increases_with_hidden():
    p_small = CoolChicVariant(hidden=8, latent_shapes=((6, 8), (3, 4), (2, 3))).param_count()
    p_large = CoolChicVariant(hidden=32, latent_shapes=((6, 8), (3, 4), (2, 3))).param_count()
    assert p_small < p_large


def test_coolchic_variant_param_count_increases_with_grid():
    p_small = CoolChicVariant(hidden=16, latent_shapes=((6, 8), (3, 4), (2, 3))).param_count()
    p_large = CoolChicVariant(hidden=16, latent_shapes=((18, 24), (9, 12), (6, 8))).param_count()
    assert p_small < p_large


# ── CoolChicDARTSCell ───────────────────────────────────────────────────


def test_cell_has_16_candidates():
    cell = CoolChicDARTSCell()
    assert len(cell.ops) == 16
    assert cell.alpha.numel() == 16


def test_cell_forward_shape_matches_input_spatial():
    cell = CoolChicDARTSCell()
    masks = torch.randint(low=0, high=5, size=(2, 16, 24))
    y = cell(masks)
    assert y.shape == (2, 3, 16, 24)


def test_cell_alpha_gradient_flows():
    """CoolChicLatentRenderer._init_weights() zeros the decoder's last
    layer (so the renderer starts as a constant image — by design). For
    DARTS to receive a non-trivial gradient through α, candidates must
    produce DIFFERENT outputs. We perturb each variant's last-layer
    weights with distinct random init so the mixture is α-sensitive."""
    torch.manual_seed(0)
    cell = CoolChicDARTSCell()
    for i, op in enumerate(cell.ops):
        with torch.no_grad():
            # CoolChicLatentRenderer.decoder is nn.Sequential; index -1 is
            # the final Conv2d that's zero-init. Re-init non-zero per-op so
            # the variants produce distinct outputs.
            last = op.renderer.decoder[-1]
            last.weight.normal_(mean=0.0, std=0.1)
            last.bias.normal_(mean=float(i) * 0.1, std=0.1)
    masks = torch.randint(low=0, high=5, size=(1, 12, 16))
    cell(masks).sum().backward()
    assert cell.alpha.grad is not None
    assert cell.alpha.grad.abs().sum() > 0


def test_cell_default_specs_match_module_constants():
    cell = CoolChicDARTSCell()
    expected = {
        (h, lg)
        for h in COOLCHIC_HIDDEN_CANDIDATES
        for lg in COOLCHIC_LATENT_GRID_CANDIDATES
    }
    assert set(cell.candidate_specs) == expected


def test_cell_discrete_arch_spec_returns_pair():
    cell = CoolChicDARTSCell()
    with torch.no_grad():
        cell.alpha[0] = 100.0
    h, lg = cell.discrete_arch_spec()
    assert h == COOLCHIC_HIDDEN_CANDIDATES[0]
    assert lg == COOLCHIC_LATENT_GRID_CANDIDATES[0]


def test_cell_rejects_singleton_axis():
    with pytest.raises(ValueError):
        CoolChicDARTSCell(hidden_candidates=(16,))
    with pytest.raises(ValueError):
        CoolChicDARTSCell(latent_grid_candidates=(((6, 8), (3, 4), (2, 3)),))


def test_cell_candidate_param_counts_includes_all():
    cell = CoolChicDARTSCell()
    counts = cell.candidate_param_counts()
    assert len(counts) == 16
    assert all(v > 0 for v in counts.values())


# ── Supernet ────────────────────────────────────────────────────────────


def test_supernet_forward_shape():
    net = build_coolchic_supernet()
    masks = torch.randint(low=0, high=5, size=(2, 16, 24))
    y = net(masks)
    assert y.shape == (2, 3, 16, 24)


def test_supernet_temperature_anneal():
    net = build_coolchic_supernet()
    net.temperature_anneal(epoch=0, total_epochs=10)
    assert net.cell.current_temperature == pytest.approx(5.0)
    net.temperature_anneal(epoch=9, total_epochs=10)
    assert net.cell.current_temperature == pytest.approx(0.1)


def test_arch_optimizer_only_steps_alpha():
    """Perturb each variant's last-layer weights so the mixture is α-
    sensitive (otherwise zero-init last layer makes every output
    constant ⇒ zero α gradient ⇒ vacuous test)."""
    torch.manual_seed(0)
    net = build_coolchic_supernet()
    for i, op in enumerate(net.cell.ops):
        with torch.no_grad():
            last = op.renderer.decoder[-1]
            last.weight.normal_(mean=0.0, std=0.1)
            last.bias.normal_(mean=float(i) * 0.1, std=0.1)
    arch_opt = build_coolchic_arch_optimizer(net, lr=1e-1)
    weight_snaps = {
        id(p): p.detach().clone()
        for p in net.parameters() if id(p) != id(net.cell.alpha)
    }
    alpha_snap = net.cell.alpha.detach().clone()
    masks = torch.randint(low=0, high=5, size=(1, 12, 16))
    net(masks).sum().backward()
    arch_opt.step()
    assert not torch.allclose(net.cell.alpha.detach(), alpha_snap)
    for p in net.parameters():
        if id(p) in weight_snaps:
            assert torch.allclose(p.detach(), weight_snaps[id(p)])


def test_make_trajectory_uses_descriptive_names():
    cell = CoolChicDARTSCell()
    traj = make_trajectory(cell)
    assert len(traj.op_names) == 16
    for name in traj.op_names:
        assert name.startswith("hidden_")
        assert "_grid_" in name
