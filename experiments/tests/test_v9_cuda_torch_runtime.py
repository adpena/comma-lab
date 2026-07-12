# ruff: noqa: E402
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.train_levelset_witness_realized_through_R_torch import (
    _accumulated_pair_step,
    _attach_generated_pose_carrier,
    _checkpoint_blob,
    _generated_pose_pair_dispatch,
    _hosc_beta_at_epoch,
    _restore,
    _run_structured_prefit,
    _softmax_temp_at_epoch,
)
from tac.boundary_math.warp_real_luma_frame0 import GroundHomographyGeom
from tac.cuda_levelset_training import (
    CudaLevelSetConfig,
    DeterministicPairCursor,
    TorchLevelSetWitness,
    TorchPoseCarrier,
)
from tac.cuda_v9_controller_runtime import TorchProtectedIslandSeed
from tac.witness_control.tail_cycles import TailController, TailCycleConfig


def _pose_flags():
    return {
        "--pose-carrier": True,
        "--pose-carrier-source": "generated",
        "--pose-carrier-residual-mode": "table",
        "--pose-carrier-residual-scale": 1.0,
        "--pose-carrier-s-t": 0.044,
        "--pose-carrier-s-r": 0.0,
        "--pose-carrier-pitch": 0.0,
    }


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
    before, _ = _generated_pose_pair_dispatch(model, feats, [0, 1], carrier, cfg)
    assert calls["out_sdf"] == 2  # one B=2 f0 + one B=2 f1 witness forward
    hook.remove()
    with torch.no_grad():
        carrier.dxi[0, 0] = 0.02
        carrier.dxi[0, 5] = 0.01
    after, _ = _generated_pose_pair_dispatch(model, feats, [0, 1], carrier, cfg)
    assert not torch.allclose(before[0, 0], after[0, 0])
    assert torch.equal(before[1, 0], after[1, 0])
    assert torch.equal(before[:, 1], after[:, 1])
    after[:, 0].square().mean().backward()
    assert carrier.dxi.grad is not None
    assert float(carrier.dxi.grad[0].abs().sum()) > 0.0


def test_typed_pose_attach_registers_child_before_optimizer_and_ema():
    cfg = CudaLevelSetConfig(
        n_pairs=3, in_feat=5, hidden_dim=8, n_hidden=1, mod_dim=4,
        render_h=6, render_w=8, camera_h=10, camera_w=12,
    )
    model = TorchLevelSetWitness.build(cfg, seed=2)
    carrier, row = _attach_generated_pose_carrier(
        model, _pose_flags(), np.zeros((3, 6), np.float32), (10, 12), torch.device("cpu")
    )
    assert carrier is model.pose_carrier and row["s_t"] == pytest.approx(0.044)
    names = dict(model.named_parameters())
    assert "pose_carrier.dxi" in names
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert any(names["pose_carrier.dxi"] is p for g in opt.param_groups for p in g["params"])


def test_structured_prefit_uses_typed_values_and_skips_resume():
    cfg = CudaLevelSetConfig(
        n_pairs=2, in_feat=5, hidden_dim=12, n_hidden=1, mod_dim=4,
        render_h=12, render_w=16,
    )
    model = TorchLevelSetWitness.build(cfg, seed=4)
    feats = torch.randn(cfg.render_h * cfg.render_w, cfg.in_feat)
    lstars = np.zeros((4, cfg.render_h, cfg.render_w), np.int64)
    lstars[:, :3] = 2
    lstars[:, 9:] = 4
    lstars[:, 4:9, 7:8] = 1
    lstars[:, 5:7, 2:5] = 3
    flags = {
        "--structured-init": True,
        "--structured-init-include-lane": True,
        "--structured-init-thresh": 0.5,
        "--structured-init-steps": 10,
        "--structured-init-lr": 5e-3,
        "--structured-init-subsample": 192,
        "--structured-init-sdf-clip": 20.0,
    }
    code_before = model.code.detach().clone()
    row = _run_structured_prefit(model, flags, lstars, feats, seed=4, is_resume=False)
    assert row["active"] and row["applied"] and row["steps"] == 10
    assert torch.equal(model.code, code_before)
    frozen = copy.deepcopy(model.state_dict())
    skipped = _run_structured_prefit(model, flags, lstars, feats, seed=4, is_resume=True)
    assert skipped["reason"] == "resume_preserves_checkpoint"
    assert all(torch.equal(v, frozen[k]) for k, v in model.state_dict().items())


def test_checkpoint_roundtrip_restores_pair_cursor_and_controller_state():
    model = torch.nn.Linear(2, 1)
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cursor = DeterministicPairCursor(7, seed=11)
    cursor.begin_epoch(3)
    first = cursor.next_epoch_indices(3)
    controllers = {"event": {"engaged": True}, "ladder": {"rung": 2}}
    seed = TorchProtectedIslandSeed(
        torch.ones(7, 2, 3, 3),
        torch.ones(7, 2, 3, dtype=torch.bool),
        mode="shield",
        damp=0.1,
    )
    seed_opt = torch.optim.AdamW(seed.parameters(), lr=0.02, weight_decay=0.0)
    tail = TailController(
        TailCycleConfig(k_max=2, cycle_floor_epochs=20, dwell_min=5),
        tau_ref=0.8,
        lr_ref=1e-3,
        tau0=0.8,
    )
    tail.step(3, [(2, 0.1)])
    blob = copy.deepcopy(_checkpoint_blob(
        model, ema, opt, 3, "cfg", ("python", "trainer"),
        pair_cursor=cursor, controller_state=controllers,
        protected_seed=seed, seed_optimizer=seed_opt,
        tail_controller=tail,
    ))
    with torch.no_grad():
        seed.residual.zero_()
    restored_cursor = DeterministicPairCursor(7, seed=0)
    restored_controllers = {"stale": True}
    restored_tail = TailController(
        TailCycleConfig(k_max=2, cycle_floor_epochs=20, dwell_min=5),
        tau_ref=0.8,
        lr_ref=1e-3,
        tau0=0.8,
    )
    epoch = _restore(
        blob, model, ema, opt, "cfg",
        pair_cursor=restored_cursor, controller_state=restored_controllers,
        protected_seed=seed, seed_optimizer=seed_opt,
        tail_controller=restored_tail,
    )
    assert epoch == 3 and restored_controllers == controllers
    assert torch.all(seed.residual == 1.0)
    assert restored_tail.state_dict() == tail.state_dict()
    rest = []
    while not restored_cursor.epoch_complete():
        rest.extend(restored_cursor.next_epoch_indices(3))
    assert sorted(first + rest) == list(range(7))


def test_torch_schedule_helpers_match_mlx_authority_endpoints_and_shapes():
    flags = {
        "--softmax-temp-start": 1.0,
        "--softmax-temp-end": 0.25,
        "--tau-anneal-shape": "geometric",
        "--anneal-epochs": 9,
        "--hosc-beta": 1.0,
        "--hosc-beta-end": 5.0,
        "--hosc-beta-anneal": "cosine",
    }
    assert _softmax_temp_at_epoch(1, 9, flags) == 1.0
    assert _softmax_temp_at_epoch(9, 9, flags) == 0.25
    assert _softmax_temp_at_epoch(5, 9, flags) == pytest.approx(0.5)
    assert _hosc_beta_at_epoch(1, 9, flags) == 1.0
    assert _hosc_beta_at_epoch(9, 9, flags) == 5.0
    assert _hosc_beta_at_epoch(5, 9, flags) == pytest.approx(3.0)


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
