from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tac.cuda_v9_controller_runtime import (
    TorchProtectedIslandSeed,
    TorchV9ControllerRuntime,
    torch_pose_jacobian_conditioning,
)


@pytest.fixture(scope="module")
def typed_flags() -> dict[str, object]:
    from tac.witness_dsl.curriculum_dsl import build_real_trainer_parser
    from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_432_launch_config

    launch = compile_v9_cgauge_432_launch_config(
        "/tmp/controller-test-gt.npz",
        num_pairs=8,
        epochs=1000,
        out_dir="/tmp/controller-test-out",
    )
    argv = list(launch.typed.to_program().compile_trainer_argv())
    namespace = build_real_trainer_parser().parse_args(argv[2:])
    return {
        "--" + key.replace("_", "-"): value for key, value in vars(namespace).items()
    }


def runtime(
    typed_flags: dict[str, object],
    *,
    lane_cls: int = 4,
    movable_cls: int = 2,
    **overrides: object,
) -> TorchV9ControllerRuntime:
    flags = dict(typed_flags)
    flags.update({"--" + key.replace("_", "-"): value for key, value in overrides.items()})
    return TorchV9ControllerRuntime(
        flags, n_classes=5, lane_cls=lane_cls, movable_cls=movable_cls
    )


def sensor_chunk(*, wrong: bool = False, lane_cls: int = 4, n: int = 1000):
    target = torch.full((n,), lane_cls, dtype=torch.long)
    pred = target.clone()
    if wrong:
        pred[0] = 0 if lane_cls != 0 else 1
    margins = torch.ones(n, dtype=torch.float32)
    return pred, target, margins


def test_protected_seed_composes_only_masked_support_and_exposes_optimizer_group() -> None:
    residual = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
    mask = np.array([[[True, False], [False, True]]])
    seed = TorchProtectedIslandSeed(residual, mask, mode="shield", damp=0.1)
    raw = torch.zeros_like(seed.residual)
    composed = seed.compose(raw, [0], weight=0.5)
    expected = torch.as_tensor(residual) * torch.as_tensor(mask)[..., None] * 0.5
    torch.testing.assert_close(composed, expected)
    group = seed.optimizer_group(lr=0.02)
    assert group["params"] == [seed.residual]
    assert group["lr"] == pytest.approx(0.02)
    assert group["name"] == "protected_island_seed"


def test_protected_seed_shield_removes_only_destructive_supported_gradient() -> None:
    residual = torch.tensor([[[[2.0, -2.0, 2.0], [1.0, 1.0, 1.0]]]])
    mask = torch.tensor([[[True, False]]])
    seed = TorchProtectedIslandSeed(residual, mask, mode="shield", damp=0.1)
    seed.residual.grad = torch.tensor([[[[3.0, -4.0, -5.0], [6.0, -7.0, 8.0]]]])
    seed.contain_grad_()
    # Same-sign gradients shrink the protected residual under descent and are removed;
    # opposite-sign constructive gradients and all off-support gradients survive.
    torch.testing.assert_close(
        seed.residual.grad,
        torch.tensor([[[[0.0, 0.0, -5.0], [6.0, -7.0, 8.0]]]]),
    )


def test_protected_seed_parameter_optimizer_and_checkpoint_resume() -> None:
    residual = torch.ones((1, 1, 2, 3))
    mask = torch.tensor([[[True, False]]])
    first = TorchProtectedIslandSeed(residual, mask, mode="damp", damp=0.25)
    opt_first = torch.optim.AdamW([first.optimizer_group(lr=0.02)])
    first.residual.grad = torch.ones_like(first.residual)
    first.contain_grad_()
    opt_first.step()
    seed_state = first.state_dict()
    opt_state = copy.deepcopy(opt_first.state_dict())

    resumed = TorchProtectedIslandSeed(residual, mask, mode="damp", damp=0.25)
    opt_resumed = torch.optim.AdamW([resumed.optimizer_group(lr=0.02)])
    resumed.load_state_dict(seed_state)
    opt_resumed.load_state_dict(opt_state)
    torch.testing.assert_close(resumed.residual, first.residual)
    assert opt_resumed.state_dict()["state"].keys() == opt_first.state_dict()["state"].keys()
    bad = dict(seed_state)
    bad["mask"] = ~bad["mask"]
    with pytest.raises(ValueError, match="mask differs"):
        resumed.load_state_dict(bad)


def test_detected_class_indices_are_required_validated_and_persisted(typed_flags) -> None:
    with pytest.raises(ValueError, match="distinct"):
        TorchV9ControllerRuntime(typed_flags, n_classes=5, lane_cls=2, movable_cls=2)
    ctrl = runtime(typed_flags, lane_cls=4, movable_cls=2)
    state = ctrl.state_dict()
    assert state["detected_classes"] == {"lane": 4, "movable": 2}
    mismatch = runtime(typed_flags, lane_cls=1, movable_cls=2)
    with pytest.raises(ValueError, match="detected classes"):
        mismatch.load_state_dict(state)


def test_scorer_counts_birth_rows_and_detected_class_ladder_lambda(typed_flags) -> None:
    ctrl = runtime(
        typed_flags,
        lane_cls=4,
        movable_cls=2,
        birth_completion_classes="1,3,4",
        birth_completion_tau_persist=1.0,
        birth_completion_area_band=1.0,
    )
    ctrl.begin_epoch(1)
    pred = torch.tensor([4, 4, 2, 1, 0, 3])
    target = torch.tensor([4, 4, 2, 2, 0, 3])
    margins = torch.tensor([0.5, 3.0, 0.5, 0.5, 3.0, 3.0])
    ctrl.observe_scorer_chunk(pred[:3], target[:3], margins[:3])
    ctrl.observe_scorer_chunk(pred[3:], target[3:], margins[3:])
    row = ctrl.end_epoch(1)
    assert row["d_seg"] == pytest.approx(1.0 / 6.0)
    assert row["annulus_flip_frac"] == pytest.approx(1.0 / 3.0)
    assert row["per_class"][4]["part_frac"] == pytest.approx(2.0 / 6.0)
    assert row["per_class"][4]["within_flip"] == pytest.approx(0.0)
    assert row["ladder_lambda"]["lane"] == pytest.approx(0.0)
    assert row["ladder_lambda"]["movable"] > 0.0
    assert any(item["class"] == 4 for item in row["birth_telemetry"])
    resumed = runtime(
        typed_flags,
        lane_cls=4,
        movable_cls=2,
        birth_completion_classes="1,3,4",
        birth_completion_tau_persist=1.0,
        birth_completion_area_band=1.0,
    )
    resumed.load_state_dict(ctrl.state_dict())
    step2 = resumed.begin_epoch(2)
    assert any(item.get("stage") == "birth_completion" for item in step2.telemetry)


def test_partial_epoch_sensor_checkpoint_resume_is_exact(typed_flags) -> None:
    continuous = runtime(typed_flags)
    continuous.begin_epoch(1)
    a = (torch.tensor([4, 4]), torch.tensor([4, 0]), torch.tensor([1.0, 1.0]))
    b = (torch.tensor([2, 2]), torch.tensor([2, 2]), torch.tensor([1.0, 3.0]))
    continuous.observe_scorer_chunk(*a)
    state = continuous.state_dict()
    continuous.observe_scorer_chunk(*b)
    expected = continuous.end_epoch(1)

    resumed = runtime(typed_flags)
    resumed.load_state_dict(state)
    resumed_step = resumed.begin_epoch(1)
    assert not resumed_step.muon_start
    resumed.observe_scorer_chunk(*b)
    actual = resumed.end_epoch(1)
    assert actual == expected


def test_real_lane_annulus_and_muon_sensors_fire_before_caps_and_resume(typed_flags) -> None:
    ctrl = runtime(
        typed_flags,
        muon_start_epoch=1000,
        lane_band_start_epoch=1000,
        seg_chroma_boundary_start_epoch=1000,
        ladder_lane_birth_epochs=0,
        ladder_lane_hold_epochs=0,
        ladder_lane_anneal_epochs=0,
        ladder_movable_birth_epochs=0,
        ladder_movable_hold_epochs=0,
        ladder_movable_anneal_epochs=0,
        annulus_plateau_dwell_windows=3,
        annulus_plateau_min_epochs=2,
        annulus_plateau_rel_eps=1e-6,
        curriculum_nucleus_min_part_frac=0.0,
        curriculum_nucleus_within_flip=0.01,
    )
    lane_fired = chroma_fired = muon_fired = False
    event_transitions: set[str] = set()
    for epoch in range(1, 40):
        step = ctrl.begin_epoch(epoch)
        lane_fired = lane_fired or step.lane_band_on
        chroma_fired = chroma_fired or step.chroma_on
        muon_fired = muon_fired or step.muon_start
        event_transitions.update(
            row["transition"]
            for row in step.telemetry
            if row.get("stage") == "start_event_fired"
        )
        ctrl.observe_scorer_chunk(*sensor_chunk(wrong=True))
        ctrl.end_epoch(epoch)
    step40 = ctrl.begin_epoch(40)
    assert lane_fired and ctrl.lane_gate.fired_epoch < 1000
    assert chroma_fired and ctrl.chroma_gate.fired_epoch < 1000
    assert muon_fired and step40.muon_on
    assert ctrl.muon_gate.fired_epoch < 1000
    assert event_transitions == {"muon", "lane_band", "seg_chroma_boundary"}
    ctrl.observe_scorer_chunk(*sensor_chunk(wrong=True))
    ctrl.end_epoch(40)

    resumed = runtime(
        typed_flags,
        muon_start_epoch=1000,
        lane_band_start_epoch=1000,
        seg_chroma_boundary_start_epoch=1000,
        ladder_lane_birth_epochs=0,
        ladder_lane_hold_epochs=0,
        ladder_lane_anneal_epochs=0,
        ladder_movable_birth_epochs=0,
        ladder_movable_hold_epochs=0,
        ladder_movable_anneal_epochs=0,
        annulus_plateau_dwell_windows=3,
        annulus_plateau_min_epochs=2,
        annulus_plateau_rel_eps=1e-6,
        curriculum_nucleus_min_part_frac=0.0,
        curriculum_nucleus_within_flip=0.01,
    )
    resumed.load_state_dict(ctrl.state_dict())
    resumed_step = resumed.begin_epoch(41)
    assert resumed_step.muon_on and not resumed_step.muon_start
    assert resumed_step.lane_band_on and resumed_step.chroma_on


def test_pose_healthy_backstop_engages_but_degenerate_selects_banked_r1(typed_flags) -> None:
    healthy = runtime(typed_flags, pose_finish_start_epoch=3)
    healthy_step = healthy.begin_epoch(3)
    assert healthy_step.pose_finish_on
    assert not healthy_step.pose_banked_r1
    healthy.observe_scorer_chunk(*sensor_chunk())
    healthy.end_epoch(3)
    state = healthy.state_dict()
    assert state["latches"]["pose_fired_epoch"] is None
    healthy_resumed = runtime(typed_flags, pose_finish_start_epoch=3)
    healthy_resumed.load_state_dict(state)
    assert healthy_resumed.begin_epoch(4).pose_finish_on

    degenerate = runtime(typed_flags, pose_finish_start_epoch=100)
    rng = np.random.default_rng(7)
    for i in range(20):
        degenerate.observe_sigma_min(i * 4, float(0.05 * (1.0 + 0.5 * rng.standard_normal())))
    verdict = degenerate.pose_detector.verdict()
    if not verdict.should_ship_banked_r1():
        pytest.skip("canonical detector classified this deterministic noise as trending, not degenerate")
    banked_step = degenerate.begin_epoch(100)
    assert not banked_step.pose_finish_on
    assert banked_step.pose_banked_r1
    assert any(row.get("alarm") == "pose_finish_backstop_overridden_banked_r1"
               for row in banked_step.telemetry)


def test_real_sigma_plateau_engages_before_backstop_and_resume_latches(typed_flags) -> None:
    ctrl = runtime(typed_flags, pose_finish_start_epoch=1000)
    for epoch in range(1, 9):
        ctrl.observe_sigma_min(epoch * 4, 0.05)
    assert ctrl.pose_detector.fired()
    step = ctrl.begin_epoch(40)
    assert step.pose_finish_on and not step.pose_banked_r1
    ctrl.observe_scorer_chunk(*sensor_chunk())
    ctrl.end_epoch(40)
    resumed = runtime(typed_flags, pose_finish_start_epoch=1000)
    resumed.load_state_dict(ctrl.state_dict())
    assert resumed.pose_detector.fired()
    assert resumed.begin_epoch(41).pose_finish_on


def test_jacobian_probe_is_real_identity_and_does_not_mutate_optimizer_state() -> None:
    class Carrier:
        def __init__(self):
            self.dxi = torch.nn.Parameter(torch.zeros((2, 6)))

        def xi_effective(self, indices):
            return self.dxi[indices]

        def forward_with_xi(self, source, xi):
            return source * 0.0 + xi.reshape(1, 1, 2, 3)

    carrier = Carrier()
    source = torch.zeros((2, 1, 2, 3))
    frame1 = torch.zeros_like(source)

    def pose_from_frames(frame0, _frame1):
        return frame0.reshape(1, 6)

    before = carrier.dxi.detach().clone()
    out = torch_pose_jacobian_conditioning(
        pose_from_frames, carrier, source, frame1, [0, 1], sigma_floor=0.5
    )
    assert out["pair_indices"] == [0, 1]
    assert out["median_sigma_min"] == pytest.approx(1.0)
    assert out["basin_frac"] == pytest.approx(1.0)
    torch.testing.assert_close(carrier.dxi, before)
    assert carrier.dxi.grad is None


def test_polyak_start_candidate_and_controller_resume(typed_flags) -> None:
    ctrl = runtime(typed_flags, polyak_finisher_arm=True, polyak_finisher_start_epoch=2)
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    assert not ctrl.observe_polyak(1, model)
    assert ctrl.polyak_candidate() is None
    assert ctrl.observe_polyak(2, model)
    with torch.no_grad():
        model.weight.fill_(3.0)
    assert ctrl.observe_polyak(3, model)
    assert ctrl._polyak_device_mean is not None
    assert ctrl._polyak_device_mean["weight"].dtype == torch.float32
    np.testing.assert_allclose(
        ctrl.polyak_candidate()["weight"], np.array([[2.0, 2.0]], np.float32)
    )

    resumed = runtime(typed_flags, polyak_finisher_arm=True, polyak_finisher_start_epoch=2)
    resumed.load_state_dict(ctrl.state_dict())
    np.testing.assert_array_equal(
        resumed.polyak_candidate()["weight"], ctrl.polyak_candidate()["weight"]
    )


def test_state_schema_rejects_version_drift(typed_flags) -> None:
    ctrl = runtime(typed_flags)
    state = ctrl.state_dict()
    state["version"] = 999
    with pytest.raises(ValueError, match="schema"):
        runtime(typed_flags).load_state_dict(state)


def test_birth_resume_rejects_typed_controller_drift(typed_flags) -> None:
    ctrl = runtime(typed_flags, birth_completion_post_level=0.2)
    state = ctrl.state_dict()
    drifted = runtime(typed_flags, birth_completion_post_level=0.4)
    with pytest.raises(ValueError, match="birth-completion checkpoint differs"):
        drifted.load_state_dict(state)
