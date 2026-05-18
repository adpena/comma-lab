# SPDX-License-Identifier: MIT
"""Tests for tac.training_curriculum package (PAUSING-EXPLOITS-WAVE).

Lane: lane_pausing_exploits_wave_20260517
Memory: feedback_pausing_exploits_wave_landed_20260517.md
Design memo: .omx/research/pausing_exploits_design_and_implementation_landed_20260517.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from tac.training_curriculum import (
    A1PatternBiasCorrectionPlan,
    CurriculumStage,
    CurriculumStageBudgetError,
    DiagnosticCheckpoint,
    DiagnosticMetric,
    DistillationConfig,
    DistillationError,
    EarlyStoppingTracker,
    GeneralizedInflateBiasCorrector,
    GreedyModelSoup,
    IMPLEMENTED_MODULES,
    InflateBiasCorrectionError,
    LossSwap,
    LossSwapError,
    MasterGradientPairWeights,
    ModelSoupError,
    PauseAndDiagnoseError,
    PauseQuantizeFinetuneError,
    PauseQuantizePlan,
    PolyakAverager,
    ResumeFromBestCheckpoint,
    StageScheduler,
    SWAScheduler,
    SWASchedulerError,
    UniformModelSoup,
    apply_pause_quantize_finetune_plan,
    derive_master_gradient_pair_weights,
    kl_on_logits_distillation,
    pause_and_capture,
    swap_loss_at_pause,
    teacher_student_pair,
)
from tac.training_curriculum.demo_nscs01_wiring import (
    NSCS01CurriculumRecipe,
    build_nscs01_canonical_curriculum,
    nscs01_diagnostic_metric_fns,
)


# ----- package surface -----


def test_package_exposes_all_11_modules() -> None:
    assert len(IMPLEMENTED_MODULES) == 11
    expected = {
        "a1_pattern_inflate_time_bias_correction",
        "early_stopping_with_resume",
        "master_gradient_pair_weights",
        "model_soup_averaging",
        "multi_stage_curriculum",
        "pause_and_diagnose",
        "pause_distill_resume",
        "pause_quantize_finetune",
        "pause_to_swap_loss",
        "quantizr_5_stage_staircase",
        "swa_polyak_averaging",
    }
    assert set(IMPLEMENTED_MODULES) == expected


def test_master_gradient_pair_weights_are_bounded_mean_normalized() -> None:
    per_pair = np.zeros((3, 4, 3), dtype=np.float64)
    per_pair[:, 0, :] = 0.1
    per_pair[:, 1, :] = 1.0
    per_pair[:, 2, :] = 3.0
    weights = derive_master_gradient_pair_weights(
        per_pair,
        archive_sha256="deadbeef1234567890abcdef",
        measurement_axis="[macOS-CPU advisory]",
        measurement_hardware="macos_arm64",
        min_weight=0.25,
        max_weight=4.0,
        top_k=2,
        bottom_k=2,
    )
    assert isinstance(weights, MasterGradientPairWeights)
    assert len(weights.pair_weights) == 4
    assert min(weights.pair_weights) >= 0.25
    assert max(weights.pair_weights) <= 4.0
    assert weights.top_k_hardest_pair_indices[0] == 2
    assert weights.bottom_k_easiest_pair_indices[0] == 3
    assert weights.score_claim is False
    assert weights.promotion_eligible is False
    policy = weights.as_policy()
    assert policy["measurement_axis"] == "[macOS-CPU advisory]"
    assert policy["ready_for_exact_eval_dispatch"] is False


def test_nscs01_demo_wiring_has_canonical_three_stage_budget() -> None:
    scheduler = build_nscs01_canonical_curriculum()

    assert scheduler.total_epochs == 1000
    assert [stage.name for stage in scheduler.stages] == [
        "anchor",
        "joint",
        "distill",
    ]
    assert scheduler.stage_for_epoch(99).name == "anchor"
    assert scheduler.stage_for_epoch(100).name == "joint"
    assert scheduler.stage_for_epoch(700).name == "distill"
    assert NSCS01CurriculumRecipe().ema_shadow_as_teacher_for_distill is True


def test_nscs01_demo_diagnostics_use_optional_head_attributes() -> None:
    class ToyNSCS01(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frame_0_head = nn.Linear(2, 2)
            self.frame_1_head = nn.Linear(2, 2)

    metrics = nscs01_diagnostic_metric_fns()
    model = ToyNSCS01()

    assert set(metrics) == {"head0_l2_norm", "head1_l2_norm"}
    assert metrics["head0_l2_norm"][0](model) > 0
    assert metrics["head1_l2_norm"][0](model) > 0
    assert metrics["head0_l2_norm"][0](nn.Linear(1, 1)) == 0.0


# ----- pause_and_diagnose -----


def _toy_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_diagnostic_metric_validates_canonical_axis() -> None:
    DiagnosticMetric(
        name="grad_norm", value=0.5, axis="diagnostic", rationale="train sanity"
    )
    with pytest.raises(PauseAndDiagnoseError):
        DiagnosticMetric(name="x", value=0, axis="MPS-PSEUDO", rationale="r")
    with pytest.raises(PauseAndDiagnoseError):
        DiagnosticMetric(name="", value=0, axis="diagnostic", rationale="r")
    with pytest.raises(PauseAndDiagnoseError):
        DiagnosticMetric(name="x", value=0, axis="diagnostic", rationale="")


def test_diagnostic_checkpoint_validates() -> None:
    with pytest.raises(PauseAndDiagnoseError):
        DiagnosticCheckpoint(
            epoch=-1,
            state_dict_path="/x",
            metrics=(),
            substrate_id="s",
            utc_iso="t",
            notes="n",
        )
    with pytest.raises(PauseAndDiagnoseError):
        DiagnosticCheckpoint(
            epoch=0,
            state_dict_path="/x",
            metrics=(),
            substrate_id="s",
            utc_iso="t",
            notes="",
        )


def test_pause_and_capture_round_trip(tmp_path: Path) -> None:
    model = _toy_model()
    checkpoint = pause_and_capture(
        model,
        epoch=42,
        output_dir=tmp_path,
        substrate_id="toy",
        metric_fns={
            "param_l2": (
                lambda m: float(
                    sum((p.detach() ** 2).sum() for p in m.parameters()).sqrt()
                ),
                "diagnostic",
                "L2 norm of all params",
            ),
        },
        notes="pause test",
        utc_iso="2026-05-17T12:00:00Z",
    )
    assert checkpoint.epoch == 42
    assert Path(checkpoint.state_dict_path).exists()
    assert len(checkpoint.metrics) == 1
    assert checkpoint.metrics[0].name == "param_l2"
    assert checkpoint.metrics[0].value > 0
    # Manifest written
    manifest_path = tmp_path / "pause_epoch_0042.manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["epoch"] == 42
    assert manifest["substrate_id"] == "toy"


def test_pause_and_capture_uses_ema_shadow_when_provided(tmp_path: Path) -> None:
    model = _toy_model()
    ema_shadow = {
        k: torch.zeros_like(v) for k, v in model.state_dict().items()
    }
    checkpoint = pause_and_capture(
        model,
        epoch=0,
        output_dir=tmp_path,
        substrate_id="toy",
        notes="ema shadow snapshot",
        utc_iso="2026-05-17T12:00:00Z",
        ema_shadow=ema_shadow,
    )
    state = torch.load(checkpoint.state_dict_path, weights_only=True)
    for k in state:
        if state[k].is_floating_point():
            assert torch.allclose(state[k], torch.zeros_like(state[k]))


# ----- multi_stage_curriculum -----


def test_curriculum_stage_validates() -> None:
    with pytest.raises(CurriculumStageBudgetError):
        CurriculumStage(name="", epochs=10, loss_key="k", notes="n")
    with pytest.raises(CurriculumStageBudgetError):
        CurriculumStage(name="s", epochs=0, loss_key="k", notes="n")
    with pytest.raises(CurriculumStageBudgetError):
        CurriculumStage(name="s", epochs=1, loss_key="", notes="n")
    with pytest.raises(CurriculumStageBudgetError):
        CurriculumStage(
            name="s", epochs=1, loss_key="k", lr_multiplier=0.0, notes="n"
        )
    with pytest.raises(CurriculumStageBudgetError):
        CurriculumStage(
            name="s",
            epochs=1,
            loss_key="k",
            optimizer_state_policy="bogus",  # type: ignore[arg-type]
            notes="n",
        )
    with pytest.raises(CurriculumStageBudgetError):
        CurriculumStage(name="s", epochs=1, loss_key="k", notes="")


def test_stage_scheduler_assigns_epochs_correctly() -> None:
    stages = (
        CurriculumStage(
            name="a", epochs=3, loss_key="L1", notes="warmup"
        ),
        CurriculumStage(
            name="b", epochs=2, loss_key="L2", notes="finetune",
            optimizer_state_policy="reset",
        ),
        CurriculumStage(
            name="c", epochs=4, loss_key="L3", notes="distill",
            optimizer_state_policy="inherit_lr_reset",
        ),
    )
    sched = StageScheduler(stages)
    assert sched.total_epochs == 9
    assert sched.stage_for_epoch(0).name == "a"
    assert sched.stage_for_epoch(2).name == "a"
    assert sched.stage_for_epoch(3).name == "b"
    assert sched.stage_for_epoch(4).name == "b"
    assert sched.stage_for_epoch(5).name == "c"
    assert sched.stage_for_epoch(8).name == "c"
    with pytest.raises(CurriculumStageBudgetError):
        sched.stage_for_epoch(9)
    with pytest.raises(CurriculumStageBudgetError):
        sched.stage_for_epoch(-1)


def test_stage_scheduler_transitions() -> None:
    stages = (
        CurriculumStage(name="a", epochs=3, loss_key="L1", notes="n"),
        CurriculumStage(
            name="b", epochs=2, loss_key="L2", notes="n",
            optimizer_state_policy="reset",
        ),
        CurriculumStage(
            name="c", epochs=4, loss_key="L3", notes="n",
            optimizer_state_policy="inherit_lr_reset",
        ),
    )
    sched = StageScheduler(stages)
    assert sched.is_transition_epoch(0) is False
    assert sched.is_transition_epoch(3) is True
    assert sched.is_transition_epoch(5) is True
    assert sched.is_transition_epoch(4) is False

    t = sched.transition_at_epoch(3)
    assert t.from_stage_name == "a"
    assert t.to_stage_name == "b"
    assert "optimizer_reset" in t.action_keys
    assert "scheduler_reset" in t.action_keys

    t2 = sched.transition_at_epoch(5)
    assert t2.from_stage_name == "b"
    assert t2.to_stage_name == "c"
    assert "loss_swapped" in t2.action_keys
    assert "scheduler_reset" in t2.action_keys
    assert "optimizer_reset" not in t2.action_keys


def test_stage_scheduler_refuses_duplicate_names() -> None:
    with pytest.raises(CurriculumStageBudgetError):
        StageScheduler(
            (
                CurriculumStage(name="a", epochs=1, loss_key="L", notes="n"),
                CurriculumStage(name="a", epochs=1, loss_key="L", notes="n"),
            )
        )


# ----- model_soup_averaging -----


def _model_pair_with_known_avg() -> tuple[
    dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]
]:
    a = {"w": torch.tensor([1.0, 2.0])}
    b = {"w": torch.tensor([3.0, 4.0])}
    expected_avg = {"w": torch.tensor([2.0, 3.0])}
    return {"A": a, "B": b}, expected_avg


def test_uniform_soup_simple_average() -> None:
    checkpoints, expected = _model_pair_with_known_avg()
    soup = UniformModelSoup()
    result = soup(checkpoints)
    assert result.num_checkpoints_in_soup == 2
    assert set(result.checkpoint_keys_kept) == {"A", "B"}
    assert torch.allclose(result.soup_state_dict["w"], expected["w"])


def test_uniform_soup_weighted() -> None:
    checkpoints, _ = _model_pair_with_known_avg()
    soup = UniformModelSoup()
    result = soup(checkpoints, weights={"A": 3.0, "B": 1.0})
    # 0.75 * [1,2] + 0.25 * [3,4] = [1.5, 2.5]
    assert torch.allclose(
        result.soup_state_dict["w"], torch.tensor([1.5, 2.5])
    )


def test_uniform_soup_refuses_shape_mismatch() -> None:
    a = {"w": torch.tensor([1.0])}
    b = {"w": torch.tensor([1.0, 2.0])}
    with pytest.raises(ModelSoupError):
        UniformModelSoup()({"A": a, "B": b})


def test_uniform_soup_refuses_key_mismatch() -> None:
    a = {"w": torch.tensor([1.0])}
    b = {"v": torch.tensor([1.0])}
    with pytest.raises(ModelSoupError):
        UniformModelSoup()({"A": a, "B": b})


def test_uniform_soup_negative_weight_rejected() -> None:
    checkpoints, _ = _model_pair_with_known_avg()
    with pytest.raises(ModelSoupError):
        UniformModelSoup()(checkpoints, weights={"A": -1.0, "B": 1.0})


def test_greedy_soup_picks_best_first_then_filters() -> None:
    a = {"w": torch.tensor([1.0])}
    b = {"w": torch.tensor([2.0])}
    c = {"w": torch.tensor([100.0])}  # poison; should be rejected
    # Metric: |w - 1.5|; lower is better. A=0.5, B=0.5, C=98.5.
    # Greedy: start with A (alphabetic tiebreak). Soup(A,B) = 1.5 → metric 0;
    # better, keep. Soup(A,B,C) = ~34.3 → metric ~32.8; reject.
    def metric(sd: dict[str, torch.Tensor]) -> float:
        return abs(float(sd["w"].item()) - 1.5)

    result = GreedyModelSoup()(
        {"A": a, "B": b, "C": c},
        held_out_metric_fn=metric,
        minimize=True,
    )
    assert "C" not in result.checkpoint_keys_kept
    assert "A" in result.checkpoint_keys_kept
    assert "B" in result.checkpoint_keys_kept
    assert result.num_checkpoints_in_soup == 2
    assert result.held_out_metric_after is not None
    assert result.held_out_metric_after < result.held_out_metric_before


def test_greedy_soup_single_checkpoint() -> None:
    only = {"w": torch.tensor([5.0])}
    result = GreedyModelSoup()(
        {"only": only},
        held_out_metric_fn=lambda sd: float(sd["w"].item()),
        minimize=True,
    )
    assert result.num_checkpoints_in_soup == 1
    assert result.checkpoint_keys_kept == ("only",)


# ----- swa_polyak_averaging -----


def test_polyak_averager_uniform_average() -> None:
    averager = PolyakAverager()
    snapshots = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 4.0]),
        torch.tensor([5.0, 6.0]),
    ]
    for snap in snapshots:
        m = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            m.weight.copy_(snap.unsqueeze(0))
        averager.update(m)
    assert averager.count == 3
    state = averager.state_dict()
    # Online mean equals true mean for ints/floats: [3.0, 4.0]
    assert torch.allclose(state["weight"], torch.tensor([[3.0, 4.0]]))


def test_polyak_averager_apply_loads_into_model() -> None:
    averager = PolyakAverager()
    m1 = nn.Linear(2, 1, bias=False)
    m2 = nn.Linear(2, 1, bias=False)
    averager.update(m1)
    averager.update(m2)
    target = nn.Linear(2, 1, bias=False)
    averager.apply(target)
    assert torch.allclose(target.weight, averager.state_dict()["weight"])


def test_polyak_averager_apply_before_update_raises() -> None:
    averager = PolyakAverager()
    with pytest.raises(SWASchedulerError):
        averager.apply(nn.Linear(2, 1))


def test_swa_scheduler_validates() -> None:
    with pytest.raises(SWASchedulerError):
        SWAScheduler(total_epochs=0)
    with pytest.raises(SWASchedulerError):
        SWAScheduler(total_epochs=10, swa_start_fraction=1.0)
    with pytest.raises(SWASchedulerError):
        SWAScheduler(total_epochs=10, swa_start_fraction=-0.1)
    with pytest.raises(SWASchedulerError):
        SWAScheduler(total_epochs=10, update_every=0)


def test_swa_scheduler_should_update_window() -> None:
    sched = SWAScheduler(
        total_epochs=100, swa_start_fraction=0.8, update_every=5
    )
    # SWA window starts at epoch 80.
    assert sched.should_update(79) is False
    assert sched.should_update(80) is True
    assert sched.should_update(81) is False
    assert sched.should_update(85) is True
    assert sched.should_update(90) is True
    assert sched.should_update(95) is True
    assert sched.should_update(99) is False
    assert sched.should_update(100) is False


# ----- pause_to_swap_loss -----


def test_loss_swap_validates() -> None:
    with pytest.raises(LossSwapError):
        LossSwap(
            epoch=-1, old_loss_key="a", new_loss_key="b",
            optimizer_state_after="inherit", rationale="r",
        )
    with pytest.raises(LossSwapError):
        LossSwap(
            epoch=0, old_loss_key="", new_loss_key="b",
            optimizer_state_after="inherit", rationale="r",
        )
    with pytest.raises(LossSwapError):
        LossSwap(
            epoch=0, old_loss_key="a", new_loss_key="b",
            optimizer_state_after="bogus",  # type: ignore[arg-type]
            rationale="r",
        )
    with pytest.raises(LossSwapError):
        LossSwap(
            epoch=0, old_loss_key="a", new_loss_key="b",
            optimizer_state_after="inherit", rationale="",
        )


def test_swap_loss_at_pause_inherit_keeps_optimizer() -> None:
    m = nn.Linear(2, 1)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    new_loss = lambda x: x.sum()
    new_loss_fn, new_opt, new_sched = swap_loss_at_pause(
        optimizer=opt, new_loss_fn=new_loss, optimizer_state_after="inherit"
    )
    assert new_opt is opt
    assert new_loss_fn is new_loss
    assert new_sched is None


def test_swap_loss_at_pause_reset_requires_factory() -> None:
    m = nn.Linear(2, 1)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    with pytest.raises(LossSwapError):
        swap_loss_at_pause(
            optimizer=opt,
            new_loss_fn=lambda x: x.sum(),
            optimizer_state_after="reset",
        )


def test_swap_loss_at_pause_reset_calls_factory() -> None:
    m = nn.Linear(2, 1)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    factory_called = []

    def factory() -> torch.optim.Optimizer:
        factory_called.append(True)
        return torch.optim.SGD(m.parameters(), lr=0.01)

    _, new_opt, _ = swap_loss_at_pause(
        optimizer=opt,
        new_loss_fn=lambda x: x.sum(),
        optimizer_state_after="reset",
        optimizer_factory=factory,
    )
    assert factory_called == [True]
    assert isinstance(new_opt, torch.optim.SGD)


# ----- a1_pattern_inflate_time_bias_correction -----


_VALID_SHA = "a" * 64


def test_a1_plan_validates() -> None:
    A1PatternBiasCorrectionPlan(
        substrate_id="s",
        baseline_archive_path="/tmp/x",
        baseline_archive_sha256=_VALID_SHA,
        sweep_offsets=(0, 1),
        sweep_deltas=(1, -1),
        held_out_metric_axis="contest-CPU",
        rationale="probe",
    )
    with pytest.raises(InflateBiasCorrectionError):
        A1PatternBiasCorrectionPlan(
            substrate_id="s",
            baseline_archive_path="/tmp/x",
            baseline_archive_sha256="not_hex_64",
            sweep_offsets=(0,),
            sweep_deltas=(1,),
            held_out_metric_axis="contest-CPU",
            rationale="r",
        )
    with pytest.raises(InflateBiasCorrectionError):
        A1PatternBiasCorrectionPlan(
            substrate_id="s",
            baseline_archive_path="/tmp/x",
            baseline_archive_sha256=_VALID_SHA,
            sweep_offsets=(0,),
            sweep_deltas=(0,),  # delta=0 forbidden
            held_out_metric_axis="contest-CPU",
            rationale="r",
        )
    with pytest.raises(InflateBiasCorrectionError):
        A1PatternBiasCorrectionPlan(
            substrate_id="s",
            baseline_archive_path="/tmp/x",
            baseline_archive_sha256=_VALID_SHA,
            sweep_offsets=(0,),
            sweep_deltas=(1,),
            held_out_metric_axis="macOS-CPU advisory",  # type: ignore[arg-type]
            rationale="r",
        )


def test_a1_corrector_materialize_and_classify(tmp_path: Path) -> None:
    baseline = tmp_path / "base.zip"
    baseline.write_bytes(b"abcdefghij")  # 10 bytes
    plan = A1PatternBiasCorrectionPlan(
        substrate_id="toy",
        baseline_archive_path=str(baseline),
        baseline_archive_sha256=_VALID_SHA,
        sweep_offsets=(0, 5),
        sweep_deltas=(1, -1),
        held_out_metric_axis="contest-CPU",
        rationale="test",
    )
    corrector = GeneralizedInflateBiasCorrector(plan)
    assert len(corrector.grid()) == 4

    out_dir = tmp_path / "out"
    candidates = corrector.materialize_candidates(output_dir=out_dir)
    assert len(candidates) == 4
    for path in candidates.values():
        assert path.exists()
        bytes_out = path.read_bytes()
        assert len(bytes_out) == 10

    # Classify with synthetic scores
    results = {
        (0, 1): ("b" * 64, 0.193),  # +0.001 from baseline 0.192 → WITHIN_NOISE
        (0, -1): ("c" * 64, 0.200),  # +0.008 → REGRESSION
        (5, 1): ("d" * 64, 0.180),  # -0.012 → IMPROVEMENT
        (5, -1): ("e" * 64, None),  # NOT_EVALUATED
    }
    verdicts = corrector.classify_results(
        baseline_score=0.192,
        results_per_candidate=results,
        noise_threshold=0.005,
    )
    assert verdicts[(0, 1)].verdict == "WITHIN_NOISE"
    assert verdicts[(0, -1)].verdict == "REGRESSION"
    assert verdicts[(5, 1)].verdict == "IMPROVEMENT"
    assert verdicts[(5, -1)].verdict == "NOT_EVALUATED"


def test_a1_corrector_classify_refuses_grid_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "b.zip"
    baseline.write_bytes(b"x" * 10)
    plan = A1PatternBiasCorrectionPlan(
        substrate_id="s",
        baseline_archive_path=str(baseline),
        baseline_archive_sha256=_VALID_SHA,
        sweep_offsets=(0,),
        sweep_deltas=(1, 2),
        held_out_metric_axis="contest-CPU",
        rationale="t",
    )
    corrector = GeneralizedInflateBiasCorrector(plan)
    with pytest.raises(InflateBiasCorrectionError):
        corrector.classify_results(
            baseline_score=0.0,
            results_per_candidate={(0, 1): ("a" * 64, 0.0)},  # missing (0,2)
        )


def test_a1_corrector_refuses_offset_beyond_archive(tmp_path: Path) -> None:
    baseline = tmp_path / "b.zip"
    baseline.write_bytes(b"x" * 5)
    plan = A1PatternBiasCorrectionPlan(
        substrate_id="s",
        baseline_archive_path=str(baseline),
        baseline_archive_sha256=_VALID_SHA,
        sweep_offsets=(10,),  # beyond 5-byte archive
        sweep_deltas=(1,),
        held_out_metric_axis="contest-CPU",
        rationale="t",
    )
    corrector = GeneralizedInflateBiasCorrector(plan)
    with pytest.raises(InflateBiasCorrectionError):
        corrector.materialize_candidates(output_dir=tmp_path / "out")


# ----- pause_distill_resume -----


def test_distillation_config_validates() -> None:
    DistillationConfig(rationale="default Hinton T=2.0")
    with pytest.raises(DistillationError):
        DistillationConfig(temperature=0.0, rationale="r")
    with pytest.raises(DistillationError):
        DistillationConfig(kl_weight=-0.1, rationale="r")
    with pytest.raises(DistillationError):
        DistillationConfig(rationale="")


def test_teacher_student_pair_freezes_teacher() -> None:
    teacher = nn.Linear(4, 2)
    student = nn.Linear(4, 2)
    cfg = DistillationConfig(rationale="t")
    t, s = teacher_student_pair(
        teacher_module=teacher, student_module=student, config=cfg
    )
    assert t is teacher
    assert s is student
    for p in teacher.parameters():
        assert p.requires_grad is False
    for p in student.parameters():
        assert p.requires_grad is True
    assert teacher.training is False


def test_teacher_student_pair_refuses_same_module() -> None:
    m = nn.Linear(4, 2)
    with pytest.raises(DistillationError):
        teacher_student_pair(
            teacher_module=m,
            student_module=m,
            config=DistillationConfig(rationale="r"),
        )


def test_kl_distillation_loss_zero_when_teacher_equals_student() -> None:
    logits = torch.randn(4, 10)
    cfg = DistillationConfig(rationale="r")
    loss = kl_on_logits_distillation(
        student_logits=logits.clone(),
        teacher_logits=logits.clone(),
        config=cfg,
    )
    assert float(loss) < 1e-5


def test_kl_distillation_loss_positive_when_teacher_differs() -> None:
    student = torch.zeros(4, 10)
    teacher = torch.zeros(4, 10)
    teacher[:, 0] = 5.0  # teacher strongly prefers class 0
    cfg = DistillationConfig(rationale="r")
    loss = kl_on_logits_distillation(
        student_logits=student,
        teacher_logits=teacher,
        config=cfg,
    )
    assert float(loss) > 0


def test_kl_distillation_refuses_shape_mismatch() -> None:
    s = torch.randn(4, 10)
    t = torch.randn(4, 5)
    with pytest.raises(DistillationError):
        kl_on_logits_distillation(
            student_logits=s, teacher_logits=t,
            config=DistillationConfig(rationale="r"),
        )


def test_kl_distillation_does_not_backprop_into_teacher() -> None:
    student = torch.zeros(4, 10, requires_grad=True)
    teacher = torch.randn(4, 10, requires_grad=True)
    cfg = DistillationConfig(rationale="r")
    loss = kl_on_logits_distillation(
        student_logits=student, teacher_logits=teacher, config=cfg
    )
    loss.backward()
    assert student.grad is not None
    # teacher_logits.detach() means teacher grad is None
    assert teacher.grad is None


# ----- pause_quantize_finetune -----


def test_pause_quantize_plan_validates() -> None:
    PauseQuantizePlan(
        quantization_kind="int8_per_channel", mode="ptq", rationale="r"
    )
    PauseQuantizePlan(
        quantization_kind="int4_per_channel",
        mode="qat_ft",
        finetune_epochs=5,
        rationale="r",
    )
    with pytest.raises(PauseQuantizeFinetuneError):
        PauseQuantizePlan(
            quantization_kind="bogus",  # type: ignore[arg-type]
            mode="ptq", rationale="r",
        )
    with pytest.raises(PauseQuantizeFinetuneError):
        PauseQuantizePlan(
            quantization_kind="int4_per_channel",
            mode="qat_ft",
            finetune_epochs=0,
            rationale="r",
        )
    with pytest.raises(PauseQuantizeFinetuneError):
        PauseQuantizePlan(
            quantization_kind="int4_per_channel",
            mode="ptq",
            finetune_epochs=5,
            rationale="r",
        )
    with pytest.raises(PauseQuantizeFinetuneError):
        PauseQuantizePlan(
            quantization_kind="int8_per_channel", mode="ptq", rationale=""
        )


def test_apply_plan_ptq_no_rollback_when_within_threshold() -> None:
    m = nn.Linear(4, 2)
    plan = PauseQuantizePlan(
        quantization_kind="int8_per_channel",
        mode="ptq",
        accuracy_recovery_threshold=0.01,
        rationale="t",
    )

    def quantize(model: nn.Module, kind: str) -> None:
        # No-op (perfect quantization).
        pass

    result = apply_pause_quantize_finetune_plan(
        model=m,
        plan=plan,
        pre_quantize_validate_fn=lambda _: 0.20,
        post_quantize_validate_fn=lambda _: 0.205,  # +0.005 < 0.01
        quantize_fn=quantize,
    )
    assert result["rollback_invoked"] is False
    assert result["final_state_kind"] == "post_quantize"
    assert result["final_metric"] == 0.205


def test_apply_plan_ptq_rollback_when_regression_exceeds_threshold() -> None:
    m = nn.Linear(4, 2)
    original_w = m.weight.detach().clone()
    plan = PauseQuantizePlan(
        quantization_kind="int8_per_channel",
        mode="ptq",
        accuracy_recovery_threshold=0.005,
        rationale="t",
    )

    def quantize(model: nn.Module, kind: str) -> None:
        with torch.no_grad():
            model.weight.add_(1.0)  # destroy weights

    result = apply_pause_quantize_finetune_plan(
        model=m,
        plan=plan,
        pre_quantize_validate_fn=lambda _: 0.20,
        post_quantize_validate_fn=lambda _: 0.30,  # +0.10 >> 0.005
        quantize_fn=quantize,
    )
    assert result["rollback_invoked"] is True
    assert result["final_state_kind"] == "pre_quantize_rollback"
    assert result["final_metric"] == 0.20
    # Verify rollback restored weights
    assert torch.allclose(m.weight, original_w)


def test_apply_plan_qat_ft_requires_finetune_fn() -> None:
    m = nn.Linear(4, 2)
    plan = PauseQuantizePlan(
        quantization_kind="int4_per_channel",
        mode="qat_ft",
        finetune_epochs=2,
        rationale="t",
    )
    with pytest.raises(PauseQuantizeFinetuneError):
        apply_pause_quantize_finetune_plan(
            model=m,
            plan=plan,
            pre_quantize_validate_fn=lambda _: 0.0,
            post_quantize_validate_fn=lambda _: 0.0,
            quantize_fn=lambda m, k: None,
            finetune_fn=None,
        )


def test_apply_plan_qat_ft_runs_finetune_then_evaluates() -> None:
    m = nn.Linear(4, 2)
    finetune_calls: list[tuple[int, float]] = []
    plan = PauseQuantizePlan(
        quantization_kind="int4_per_channel",
        mode="qat_ft",
        finetune_epochs=3,
        finetune_lr_multiplier=0.1,
        accuracy_recovery_threshold=0.1,
        rationale="t",
    )

    def finetune(model: nn.Module, epochs: int, lr: float) -> float:
        finetune_calls.append((epochs, lr))
        return 0.21  # close to pre-quantize 0.20

    result = apply_pause_quantize_finetune_plan(
        model=m, plan=plan,
        pre_quantize_validate_fn=lambda _: 0.20,
        post_quantize_validate_fn=lambda _: 0.25,
        quantize_fn=lambda m, k: None,
        finetune_fn=finetune,
        base_lr=1e-3,
    )
    assert finetune_calls == [(3, 1e-4)]
    assert result["post_finetune_metric"] == 0.21
    assert result["rollback_invoked"] is False
    assert result["final_state_kind"] == "post_finetune"


# ----- early_stopping_with_resume -----


def test_early_stopping_tracker_validates() -> None:
    with pytest.raises(ValueError):
        EarlyStoppingTracker(patience=0, minimize=True, checkpoint_path=Path("/x"))
    with pytest.raises(ValueError):
        EarlyStoppingTracker(
            patience=1, minimize=True, checkpoint_path=Path("/x"),
            improvement_threshold=-0.1,
        )


def test_early_stopping_tracker_improves_then_stops(tmp_path: Path) -> None:
    tracker = EarlyStoppingTracker(
        patience=3, minimize=True, checkpoint_path=tmp_path / "best.pt"
    )
    m = nn.Linear(2, 1)
    # Epoch 0: new best
    is_new_best = tracker.update(m, 0.5, 0)
    assert is_new_best
    assert tracker.state.best_metric == 0.5
    assert (tmp_path / "best.pt").exists()
    # Epoch 1: better
    assert tracker.update(m, 0.3, 1)
    # Epochs 2-4: no improvement
    for ep in (2, 3, 4):
        assert not tracker.update(m, 0.4, ep)
    assert tracker.state.stopped is True
    assert tracker.state.best_metric == 0.3
    assert tracker.state.best_epoch == 1


def test_early_stopping_tracker_maximize_mode(tmp_path: Path) -> None:
    tracker = EarlyStoppingTracker(
        patience=2, minimize=False, checkpoint_path=tmp_path / "best.pt"
    )
    m = nn.Linear(2, 1)
    assert tracker.update(m, 0.5, 0)  # new best (started at -inf)
    assert tracker.update(m, 0.8, 1)  # better
    assert not tracker.update(m, 0.6, 2)
    assert not tracker.update(m, 0.7, 3)
    assert tracker.state.stopped is True
    assert tracker.state.best_metric == 0.8


def test_resume_from_best_loads_checkpoint(tmp_path: Path) -> None:
    tracker = EarlyStoppingTracker(
        patience=10, minimize=True, checkpoint_path=tmp_path / "best.pt"
    )
    m_train = nn.Linear(2, 1)
    with torch.no_grad():
        m_train.weight.fill_(42.0)
    tracker.update(m_train, 0.5, 0)
    # Mutate after best saved
    with torch.no_grad():
        m_train.weight.fill_(99.0)
    # Resume into fresh model
    m_resume = nn.Linear(2, 1)
    ResumeFromBestCheckpoint()(m_resume, tracker)
    assert torch.allclose(m_resume.weight, torch.full_like(m_resume.weight, 42.0))


def test_resume_from_best_raises_if_no_checkpoint(tmp_path: Path) -> None:
    tracker = EarlyStoppingTracker(
        patience=3, minimize=True, checkpoint_path=tmp_path / "best.pt"
    )
    with pytest.raises(FileNotFoundError):
        ResumeFromBestCheckpoint()(nn.Linear(2, 1), tracker)
