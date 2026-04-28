"""Unit tests for the generic DARTS framework (:mod:`tac.darts`).

Verifies:
  * DARTSCell forward returns the softmax-weighted mixture of op outputs.
  * DARTSCell backward populates gradient on `alpha` AND on candidate
    op weights.
  * Temperature anneal is monotonic and clamped to [T_end, T_start].
  * KL-to-uniform diagnostic returns 0 for uniform α and ≈ log(N) for a
    one-hot α.
  * DARTSOptimizer ONLY steps on the α parameter (weight params untouched).
  * split_arch_weight_params correctly partitions a multi-cell supernet.
  * DARTSAlphaTrajectory records produce the expected JSON shape.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from tac.darts import (
    DARTSAnnealSchedule,
    DARTSAlphaTrajectory,
    DARTSCell,
    DARTSOptimizer,
    alpha_kl_to_uniform,
    alpha_softmax,
    discrete_arch_index,
    split_arch_weight_params,
    darts_search_step,
)


# ── Tiny ops for the tests ──────────────────────────────────────────────


class _ScaleOp(nn.Module):
    """Multiplies its input by a learnable scalar."""

    def __init__(self, init: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


# ── DARTSAnnealSchedule ─────────────────────────────────────────────────


def test_anneal_schedule_monotonic_and_clamped():
    sched = DARTSAnnealSchedule(T_start=5.0, T_end=0.1)
    Ts = [sched.temperature(e, 100) for e in range(101)]
    # Monotonically non-increasing.
    assert all(Ts[i] >= Ts[i + 1] - 1e-9 for i in range(100))
    # Endpoints exact.
    assert Ts[0] == pytest.approx(5.0)
    # Past-the-end epoch is clamped to last.
    assert sched.temperature(999, 100) == pytest.approx(0.1)
    # Negative epoch is clamped to first.
    assert sched.temperature(-1, 100) == pytest.approx(5.0)


def test_anneal_schedule_rejects_non_positive_temps():
    with pytest.raises(ValueError):
        DARTSAnnealSchedule(T_start=0.0, T_end=0.1)
    with pytest.raises(ValueError):
        DARTSAnnealSchedule(T_start=5.0, T_end=-0.1)
    with pytest.raises(ValueError):
        DARTSAnnealSchedule(T_start=0.5, T_end=1.0)  # not annealing DOWN


def test_anneal_total_epochs_one_returns_t_end():
    sched = DARTSAnnealSchedule(T_start=5.0, T_end=0.1)
    assert sched.temperature(0, 1) == pytest.approx(0.1)


# ── alpha_softmax + KL diagnostic ───────────────────────────────────────


def test_alpha_softmax_uniform_when_all_zero():
    a = torch.zeros(5)
    p = alpha_softmax(a, temperature=1.0)
    assert torch.allclose(p, torch.full((5,), 1 / 5))


def test_alpha_softmax_high_temp_more_uniform_than_low():
    a = torch.tensor([5.0, 0.0, 0.0])
    p_hot = alpha_softmax(a, temperature=5.0)
    p_cold = alpha_softmax(a, temperature=0.1)
    # Cold softmax is sharper → max prob is larger.
    assert p_cold.max() > p_hot.max()


def test_alpha_softmax_rejects_non_positive_temp():
    a = torch.zeros(3)
    with pytest.raises(ValueError):
        alpha_softmax(a, temperature=0.0)


def test_alpha_kl_to_uniform_zero_when_uniform():
    a = torch.zeros(7)
    kl = alpha_kl_to_uniform(a, temperature=1.0)
    assert kl == pytest.approx(0.0, abs=1e-6)


def test_alpha_kl_to_uniform_max_when_onehot():
    """KL → log(N) as α concentrates on one candidate."""
    n = 5
    a = torch.zeros(n)
    a[2] = 100.0  # huge logit → near-Dirac softmax
    kl = alpha_kl_to_uniform(a, temperature=1.0)
    assert kl == pytest.approx(math.log(n), abs=1e-3)


def test_alpha_kl_nonnegative():
    a = torch.randn(8)
    assert alpha_kl_to_uniform(a, temperature=1.0) >= 0.0


def test_discrete_arch_index_returns_argmax():
    a = torch.tensor([1.0, 3.0, 2.0])
    assert discrete_arch_index(a) == 1


# ── DARTSCell forward + backward ────────────────────────────────────────


def _make_simple_cell() -> DARTSCell:
    ops = [_ScaleOp(1.0), _ScaleOp(2.0), _ScaleOp(3.0)]
    return DARTSCell(ops=ops, names=("a", "b", "c"))


def test_darts_cell_forward_is_softmax_mixture():
    cell = _make_simple_cell()
    cell._current_T = 1.0  # uniform softmax (α=0)
    x = torch.ones(2, 3)
    out = cell(x)
    # Uniform mixture of ScaleOps with scales [1, 2, 3] = mean(1,2,3) = 2.0
    assert torch.allclose(out, torch.full_like(out, 2.0))


def test_darts_cell_backward_populates_alpha_grad():
    cell = _make_simple_cell()
    x = torch.ones(2, 3)
    loss = cell(x).sum()
    loss.backward()
    assert cell.alpha.grad is not None
    assert cell.alpha.grad.abs().sum() > 0


def test_darts_cell_backward_populates_op_weight_grads():
    cell = _make_simple_cell()
    x = torch.ones(2, 3)
    loss = cell(x).sum()
    loss.backward()
    for op in cell.ops:
        assert op.scale.grad is not None
        assert op.scale.grad.abs().sum() > 0


def test_darts_cell_temperature_anneal_updates_current_T():
    cell = _make_simple_cell()
    cell.temperature_anneal(epoch=0, total_epochs=10)
    assert cell.current_temperature == pytest.approx(5.0)
    cell.temperature_anneal(epoch=9, total_epochs=10)
    assert cell.current_temperature == pytest.approx(0.1)


def test_darts_cell_rejects_singleton_op_list():
    with pytest.raises(ValueError):
        DARTSCell(ops=[_ScaleOp(1.0)])


def test_darts_cell_rejects_mismatched_names():
    with pytest.raises(ValueError):
        DARTSCell(ops=[_ScaleOp(1.0), _ScaleOp(2.0)], names=("only_one",))


def test_darts_cell_default_alpha_is_zero():
    cell = _make_simple_cell()
    assert torch.allclose(cell.alpha.detach(), torch.zeros_like(cell.alpha))


# ── DARTSOptimizer ──────────────────────────────────────────────────────


def test_darts_optimizer_only_steps_on_alpha():
    cell = _make_simple_cell()
    arch_opt = DARTSOptimizer(cell.arch_parameters(), lr=1e-1)
    weight_snapshots = {id(op): op.scale.detach().clone() for op in cell.ops}
    alpha_before = cell.alpha.detach().clone()

    x = torch.ones(2, 3)
    loss = cell(x).sum()
    loss.backward()
    arch_opt.step()

    # α moved.
    assert not torch.allclose(cell.alpha.detach(), alpha_before)
    # Op weights did NOT move (we never stepped weight_opt).
    for op in cell.ops:
        assert torch.allclose(op.scale.detach(), weight_snapshots[id(op)])


def test_darts_optimizer_rejects_empty_param_list():
    with pytest.raises(ValueError):
        DARTSOptimizer(arch_params=[])


def test_darts_optimizer_rejects_non_1d_params():
    with pytest.raises(ValueError):
        DARTSOptimizer(arch_params=[nn.Parameter(torch.zeros(3, 4))])


# ── split_arch_weight_params on a multi-cell supernet ───────────────────


def test_split_arch_weight_params_multi_cell():
    class _Twin(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell_a = _make_simple_cell()
            self.cell_b = _make_simple_cell()
            self.head = nn.Linear(3, 1)

        def forward(self, x):
            return self.head(self.cell_b(self.cell_a(x)))

    net = _Twin()
    arch_params, weight_params = split_arch_weight_params(net)
    assert len(arch_params) == 2  # cell_a.alpha + cell_b.alpha
    arch_ids = {id(p) for p in arch_params}
    for p in weight_params:
        assert id(p) not in arch_ids
    # Sanity: every model parameter is in exactly one of the two groups.
    all_ids = {id(p) for p in net.parameters()}
    split_ids = arch_ids | {id(p) for p in weight_params}
    assert all_ids == split_ids


# ── DARTSAlphaTrajectory ────────────────────────────────────────────────


def test_trajectory_records_match_cell_state():
    cell = _make_simple_cell()
    cell.temperature_anneal(epoch=0, total_epochs=2)
    traj = DARTSAlphaTrajectory(op_names=cell.names)
    traj.record(epoch=0, cell=cell, train_loss=1.0, val_loss=2.0)
    cell.temperature_anneal(epoch=1, total_epochs=2)
    # Bias α toward op index 1.
    with torch.no_grad():
        cell.alpha[1] = 5.0
    traj.record(epoch=1, cell=cell, train_loss=0.5, val_loss=1.0)

    d = traj.to_dict()
    assert len(d["records"]) == 2
    assert d["op_names"] == list(cell.names)
    assert d["discovered"]["argmax_index"] == 1
    assert d["discovered"]["argmax_name"] == "b"
    # KL > 0 because α is non-uniform.
    assert d["discovered"]["kl_nats_final"] > 0.0


def test_trajectory_convergence_verdict_string():
    cell = _make_simple_cell()
    traj = DARTSAlphaTrajectory(op_names=cell.names)
    # Uniform α → KL=0 → "inconclusive".
    traj.record(epoch=0, cell=cell)
    assert traj.to_dict()["discovered"]["convergence_verdict"] == "inconclusive"
    # One-hot α → KL≈log(3) ≈ 1.10 → "moderate".
    with torch.no_grad():
        cell.alpha[0] = 100.0
    traj2 = DARTSAlphaTrajectory(op_names=cell.names)
    traj2.record(epoch=0, cell=cell)
    verdict = traj2.to_dict()["discovered"]["convergence_verdict"]
    assert verdict in {"moderate", "decisive"}


# ── Alternating-SGD step ────────────────────────────────────────────────


def test_darts_search_step_runs_without_error():
    cell = _make_simple_cell()
    arch_opt = DARTSOptimizer(cell.arch_parameters(), lr=1e-2)
    weight_opt = torch.optim.SGD(cell.weight_parameters(), lr=1e-2)
    x_train = torch.randn(4, 3)
    x_val = torch.randn(4, 3)
    target = torch.zeros(4, 3)

    val_loss, train_loss = darts_search_step(
        supernet=cell,
        val_loss_fn=lambda: ((cell(x_val) - target) ** 2).mean(),
        train_loss_fn=lambda: ((cell(x_train) - target) ** 2).mean(),
        arch_opt=arch_opt,
        weight_opt=weight_opt,
    )
    assert isinstance(val_loss, float)
    assert isinstance(train_loss, float)
    assert math.isfinite(val_loss)
    assert math.isfinite(train_loss)
