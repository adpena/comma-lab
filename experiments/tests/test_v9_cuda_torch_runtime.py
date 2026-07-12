# ruff: noqa: E402
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.train_levelset_witness_realized_through_R_torch import (
    _accumulated_pair_step,
    _generated_pose_pair_dispatch,
)
from tac.boundary_math.warp_real_luma_frame0 import GroundHomographyGeom
from tac.cuda_levelset_training import (
    CudaLevelSetConfig,
    TorchLevelSetWitness,
    TorchPoseCarrier,
)


def test_generated_pose_dispatch_changes_only_even_frame_with_dxi():
    cfg = CudaLevelSetConfig(
        n_pairs=2, in_feat=5, hidden_dim=8, n_hidden=1, mod_dim=4,
        render_h=6, render_w=8, camera_h=10, camera_w=12,
    )
    model = TorchLevelSetWitness.build(cfg, seed=13)
    geom = GroundHomographyGeom.eon(native_hw=(cfg.camera_h, cfg.camera_w), pitch=0.0)
    carrier = TorchPoseCarrier.build(np.zeros((cfg.n_pairs, 6), np.float32), geom)
    model.pose_carrier = carrier  # child attach before EMA/optimizer in production
    feats = torch.randn(cfg.render_h * cfg.render_w, cfg.in_feat)
    calls = {"out_sdf": 0}
    hook = model.out_sdf.register_forward_hook(
        lambda *_args: calls.__setitem__("out_sdf", calls["out_sdf"] + 1)
    )
    before, _ = _generated_pose_pair_dispatch(model, feats, [0], carrier, cfg)
    assert calls["out_sdf"] == 2  # exactly one plain f0 + one f1 witness forward
    hook.remove()
    with torch.no_grad():
        carrier.dxi[0, 0] = 0.02
        carrier.dxi[0, 5] = 0.01
    after, _ = _generated_pose_pair_dispatch(model, feats, [0], carrier, cfg)
    assert not torch.allclose(before[:, 0], after[:, 0])
    assert torch.equal(before[:, 1], after[:, 1])
    after[:, 0].square().mean().backward()
    assert carrier.dxi.grad is not None
    assert float(carrier.dxi.grad[0].abs().sum()) > 0.0


def test_accumulated_pair_step_matches_one_step_on_mean_accepted_loss():
    model = torch.nn.Linear(2, 1, bias=False)
    expected = copy.deepcopy(model)
    xs = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, -1.0]])]
    ys = [torch.tensor([[0.5]]), torch.tensor([[-0.2]])]
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt_expected = torch.optim.SGD(expected.parameters(), lr=0.1)

    row = _accumulated_pair_step(
        model, opt, [0, 1],
        lambda i: (model(xs[i]) - ys[i]).square().mean(),
        grad_clip=1e9,
    )
    opt_expected.zero_grad(set_to_none=True)
    mean_loss = torch.stack([
        (expected(xs[i]) - ys[i]).square().mean() for i in (0, 1)
    ]).mean()
    mean_loss.backward()
    opt_expected.step()
    assert row["accepted"] == 1 and row["attempted"] == 1
    assert row["pair_count"] == 2
    assert row["accepted_frac"] == 1.0 and row["weights_stepped"]
    assert torch.allclose(model.weight, expected.weight, atol=1e-7)


def test_accumulated_pair_step_rejects_entire_chunk_without_partial_update():
    model = torch.nn.Linear(1, 1, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    before = model.weight.detach().clone()
    row = _accumulated_pair_step(
        model, opt, [0, 1, 2],
        lambda i: None if i == 1 else model(torch.tensor([[2.0]])).square().mean(),
        grad_clip=1e9,
    )
    assert row["accepted"] == 0 and row["attempted"] == 1
    assert row["accepted_frac"] == 0.0 and not row["weights_stepped"]
    assert row["pair_count"] == 3
    assert torch.equal(model.weight, before)


def test_accumulated_pair_step_visits_each_pair_once():
    model = torch.nn.Linear(1, 1, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    visited = []

    def loss_builder(i):
        visited.append(i)
        return model(torch.tensor([[float(i + 1)]])).square().mean()

    row = _accumulated_pair_step(
        model, opt, [3, 1, 2, 0], loss_builder, grad_clip=1e9,
    )
    assert visited == [3, 1, 2, 0]
    assert row["pair_count"] == 4 and row["weights_stepped"]
