# SPDX-License-Identifier: MIT
"""Comprehensive tests for tac.training_curriculum package.

Covers all 9 helpers landed in the PAUSING-EXPLOITS-WAVE 2026-05-17:

* pause_and_diagnose
* multi_stage_curriculum
* model_soup_averaging
* swa_polyak_averaging
* pause_to_swap_loss
* a1_pattern_inflate_time_bias_correction
* pause_distill_resume
* pause_quantize_finetune
* early_stopping_with_resume

Per CLAUDE.md "Apples-to-apples evidence discipline": tests use tiny
synthetic torch.nn.Module fixtures so the test surface is fully deterministic
and runs in <1s on CPU. No GPU required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.training_curriculum import (
    DEFERRED_MODULES,
    DEFERRED_RATIONALE,
    IMPLEMENTED_MODULES,
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
    InflateBiasCorrectionError,
    LossSwap,
    LossSwapError,
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
    kl_on_logits_distillation,
    pause_and_capture,
    swap_loss_at_pause,
    teacher_student_pair,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class _TinyHead(nn.Module):
    """4-param head with deterministic output for testing pause / averaging."""

    def __init__(self, n_in: int = 2, n_out: int = 2, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.lin = nn.Linear(n_in, n_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


@pytest.fixture
def tiny_head() -> _TinyHead:
    return _TinyHead()


@pytest.fixture
def tiny_head_alt() -> _TinyHead:
    return _TinyHead(seed=1)


# ---------------------------------------------------------------------------
# Package-level sanity
# ---------------------------------------------------------------------------


class TestPackageState:
    def test_implemented_modules_canonical(self) -> None:
        assert "a1_pattern_inflate_time_bias_correction" in IMPLEMENTED_MODULES
        assert "pause_distill_resume" in IMPLEMENTED_MODULES
        assert "quantizr_5_stage_staircase" in IMPLEMENTED_MODULES
        assert len(IMPLEMENTED_MODULES) == 10

    def test_deferred_modules_empty(self) -> None:
        assert DEFERRED_MODULES == ()

    def test_deferred_rationale_non_empty(self) -> None:
        assert DEFERRED_RATIONALE.strip()


# ---------------------------------------------------------------------------
# pause_and_diagnose
# ---------------------------------------------------------------------------


class TestPauseAndDiagnose:
    def test_diagnostic_metric_valid(self) -> None:
        m = DiagnosticMetric(
            name="grad_norm", value=1.23, axis="diagnostic", rationale="rn"
        )
        assert m.value == pytest.approx(1.23)

    def test_diagnostic_metric_rejects_empty_name(self) -> None:
        with pytest.raises(PauseAndDiagnoseError, match="name must be non-empty"):
            DiagnosticMetric(name="", value=0.0, axis="diagnostic", rationale="r")

    def test_diagnostic_metric_rejects_unknown_axis(self) -> None:
        with pytest.raises(PauseAndDiagnoseError, match="not in canonical set"):
            DiagnosticMetric(
                name="x", value=0.0, axis="bogus", rationale="r"
            )

    def test_diagnostic_metric_rejects_empty_rationale(self) -> None:
        with pytest.raises(PauseAndDiagnoseError, match="rationale must be non-empty"):
            DiagnosticMetric(name="x", value=0.0, axis="diagnostic", rationale="")

    def test_diagnostic_checkpoint_valid(self) -> None:
        ckpt = DiagnosticCheckpoint(
            epoch=10,
            state_dict_path="/tmp/x.pt",
            metrics=(),
            substrate_id="test_sub",
            utc_iso="2026-05-17T00:00:00Z",
            notes="test",
        )
        assert ckpt.epoch == 10

    def test_diagnostic_checkpoint_rejects_negative_epoch(self) -> None:
        with pytest.raises(PauseAndDiagnoseError, match="must be >= 0"):
            DiagnosticCheckpoint(
                epoch=-1,
                state_dict_path="/tmp/x.pt",
                metrics=(),
                substrate_id="t",
                utc_iso="2026-05-17T00:00:00Z",
                notes="n",
            )

    def test_diagnostic_checkpoint_rejects_empty_notes(self) -> None:
        with pytest.raises(PauseAndDiagnoseError, match="notes must be non-empty"):
            DiagnosticCheckpoint(
                epoch=0,
                state_dict_path="/tmp/x.pt",
                metrics=(),
                substrate_id="t",
                utc_iso="2026-05-17T00:00:00Z",
                notes="",
            )

    def test_pause_and_capture_persists(
        self, tmp_path: Path, tiny_head: _TinyHead
    ) -> None:
        ckpt = pause_and_capture(
            tiny_head,
            epoch=5,
            output_dir=tmp_path,
            substrate_id="tiny",
            metric_fns={
                "weight_norm": (
                    lambda m: float(m.lin.weight.norm().item()),
                    "diagnostic",
                    "weight L2 norm",
                ),
            },
            notes="smoke test pause",
            utc_iso="2026-05-17T12:00:00Z",
        )
        assert (tmp_path / "pause_epoch_0005.pt").exists()
        assert (tmp_path / "pause_epoch_0005.manifest.json").exists()
        assert len(ckpt.metrics) == 1
        assert ckpt.metrics[0].name == "weight_norm"

        manifest = json.loads(
            (tmp_path / "pause_epoch_0005.manifest.json").read_text()
        )
        assert manifest["epoch"] == 5
        assert manifest["substrate_id"] == "tiny"

    def test_pause_and_capture_uses_ema_shadow_when_provided(
        self, tmp_path: Path, tiny_head: _TinyHead
    ) -> None:
        ema_shadow = {
            k: torch.zeros_like(v) for k, v in tiny_head.state_dict().items()
        }
        ckpt = pause_and_capture(
            tiny_head,
            epoch=1,
            output_dir=tmp_path,
            substrate_id="t",
            metric_fns=None,
            notes="ema test",
            utc_iso="2026-05-17T12:00:00Z",
            ema_shadow=ema_shadow,
        )
        loaded = torch.load(ckpt.state_dict_path, weights_only=True)
        # EMA shadow is all-zeros; live model is random — loaded should be zeros.
        for v in loaded.values():
            assert torch.all(v == 0.0)

    def test_pause_and_capture_propagates_metric_failure(
        self, tmp_path: Path, tiny_head: _TinyHead
    ) -> None:
        def broken(_m: nn.Module) -> float:
            raise RuntimeError("boom")

        with pytest.raises(PauseAndDiagnoseError, match="raised"):
            pause_and_capture(
                tiny_head,
                epoch=0,
                output_dir=tmp_path,
                substrate_id="t",
                metric_fns={
                    "broken": (broken, "diagnostic", "intentionally broken"),
                },
                notes="probe failure",
                utc_iso="2026-05-17T12:00:00Z",
            )


# ---------------------------------------------------------------------------
# multi_stage_curriculum
# ---------------------------------------------------------------------------


class TestMultiStageCurriculum:
    def _quantizr_stages(self) -> tuple[CurriculumStage, ...]:
        return (
            CurriculumStage(
                name="anchor",
                epochs=10,
                loss_key="pixel_only",
                lr_multiplier=1.0,
                optimizer_state_policy="reset",
                notes="warmup",
            ),
            CurriculumStage(
                name="joint",
                epochs=20,
                loss_key="pixel_plus_scorer",
                lr_multiplier=0.5,
                optimizer_state_policy="inherit_lr_reset",
                notes="add scorer",
            ),
            CurriculumStage(
                name="distill",
                epochs=10,
                loss_key="kl_T2",
                lr_multiplier=0.1,
                optimizer_state_policy="inherit",
                notes="KL distill T=2",
            ),
        )

    def test_stage_validation_rejects_zero_epochs(self) -> None:
        with pytest.raises(CurriculumStageBudgetError, match="must be >= 1"):
            CurriculumStage(
                name="x", epochs=0, loss_key="l", notes="n",
            )

    def test_stage_validation_rejects_negative_lr(self) -> None:
        with pytest.raises(CurriculumStageBudgetError, match="must be > 0"):
            CurriculumStage(
                name="x", epochs=1, loss_key="l", lr_multiplier=-0.1, notes="n",
            )

    def test_stage_validation_rejects_unknown_optimizer_policy(self) -> None:
        with pytest.raises(CurriculumStageBudgetError, match="not in canonical set"):
            CurriculumStage(
                name="x",
                epochs=1,
                loss_key="l",
                optimizer_state_policy="bogus",  # type: ignore[arg-type]
                notes="n",
            )

    def test_scheduler_total_epochs(self) -> None:
        sched = StageScheduler(self._quantizr_stages())
        assert sched.total_epochs == 40

    def test_scheduler_stage_for_epoch(self) -> None:
        sched = StageScheduler(self._quantizr_stages())
        assert sched.stage_for_epoch(0).name == "anchor"
        assert sched.stage_for_epoch(9).name == "anchor"
        assert sched.stage_for_epoch(10).name == "joint"
        assert sched.stage_for_epoch(29).name == "joint"
        assert sched.stage_for_epoch(30).name == "distill"
        assert sched.stage_for_epoch(39).name == "distill"

    def test_scheduler_out_of_range_raises(self) -> None:
        sched = StageScheduler(self._quantizr_stages())
        with pytest.raises(CurriculumStageBudgetError, match="outside total budget"):
            sched.stage_for_epoch(40)
        with pytest.raises(CurriculumStageBudgetError, match="outside total budget"):
            sched.stage_for_epoch(-1)

    def test_scheduler_is_transition_epoch(self) -> None:
        sched = StageScheduler(self._quantizr_stages())
        assert not sched.is_transition_epoch(0)
        assert not sched.is_transition_epoch(5)
        assert sched.is_transition_epoch(10)
        assert sched.is_transition_epoch(30)

    def test_scheduler_transition_actions(self) -> None:
        sched = StageScheduler(self._quantizr_stages())
        t1 = sched.transition_at_epoch(10)  # anchor → joint (inherit_lr_reset)
        assert t1.from_stage_name == "anchor"
        assert t1.to_stage_name == "joint"
        assert "loss_swapped" in t1.action_keys
        assert "scheduler_reset" in t1.action_keys
        assert "optimizer_reset" not in t1.action_keys

        t2 = sched.transition_at_epoch(30)  # joint → distill (inherit)
        assert t2.action_keys == ("loss_swapped",)

    def test_scheduler_rejects_duplicate_names(self) -> None:
        dup = (
            CurriculumStage(name="x", epochs=1, loss_key="a", notes="n"),
            CurriculumStage(name="x", epochs=1, loss_key="b", notes="n"),
        )
        with pytest.raises(CurriculumStageBudgetError, match="unique"):
            StageScheduler(dup)


# ---------------------------------------------------------------------------
# model_soup_averaging
# ---------------------------------------------------------------------------


class TestModelSoupAveraging:
    def _three_checkpoints(self) -> dict[str, dict[str, torch.Tensor]]:
        # 3 distinct state_dicts; same architecture.
        return {
            "a": {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.1])},
            "b": {"w": torch.tensor([3.0, 4.0]), "b": torch.tensor([0.3])},
            "c": {"w": torch.tensor([5.0, 6.0]), "b": torch.tensor([0.5])},
        }

    def test_uniform_soup_averages_correctly(self) -> None:
        result = UniformModelSoup()(self._three_checkpoints())
        assert torch.allclose(
            result.soup_state_dict["w"], torch.tensor([3.0, 4.0])
        )
        assert torch.allclose(
            result.soup_state_dict["b"], torch.tensor([0.3])
        )
        assert result.num_checkpoints_in_soup == 3

    def test_uniform_soup_custom_weights(self) -> None:
        ckpts = self._three_checkpoints()
        weights = {"a": 0.5, "b": 0.5, "c": 0.0}
        result = UniformModelSoup()(ckpts, weights=weights)
        # (1+3)/2=2, (2+4)/2=3
        assert torch.allclose(
            result.soup_state_dict["w"], torch.tensor([2.0, 3.0])
        )

    def test_uniform_soup_rejects_negative_weight(self) -> None:
        with pytest.raises(ModelSoupError, match="must be non-negative"):
            UniformModelSoup()(
                self._three_checkpoints(),
                weights={"a": -1.0, "b": 0.5, "c": 0.5},
            )

    def test_uniform_soup_rejects_shape_mismatch(self) -> None:
        bad = {
            "a": {"w": torch.tensor([1.0, 2.0])},
            "b": {"w": torch.tensor([3.0, 4.0, 5.0])},
        }
        with pytest.raises(ModelSoupError, match="shape"):
            UniformModelSoup()(bad)

    def test_greedy_soup_keeps_only_improving(self) -> None:
        ckpts = self._three_checkpoints()

        # Metric: prefer state with w[0] near 3.0 (so "a" alone is bad,
        # adding "b" to "a" helps, adding "c" hurts).
        def metric(sd: dict[str, torch.Tensor]) -> float:
            return float(abs(sd["w"][0].item() - 3.0))

        result = GreedyModelSoup()(ckpts, held_out_metric_fn=metric, minimize=True)
        # "b" alone scores 0 (best individual). "a" + "b" averages to 2.0 (score 1.0).
        # "b" + "c" averages to 4.0 (score 1.0). Soup starts with "b" (best);
        # then "a" candidate (score 1.0 > 0) REJECTED; then "c" candidate REJECTED.
        assert result.checkpoint_keys_kept == ("b",)
        assert result.held_out_metric_after == pytest.approx(0.0)

    def test_greedy_soup_rejects_shape_mismatch(self) -> None:
        bad = {
            "a": {"w": torch.tensor([1.0])},
            "b": {"w": torch.tensor([1.0, 2.0])},
        }
        with pytest.raises(ModelSoupError, match="shape"):
            GreedyModelSoup()(bad, held_out_metric_fn=lambda _: 0.0)


# ---------------------------------------------------------------------------
# swa_polyak_averaging
# ---------------------------------------------------------------------------


class TestSWAPolyakAveraging:
    def test_polyak_averager_running_mean(
        self, tiny_head: _TinyHead, tiny_head_alt: _TinyHead
    ) -> None:
        avg = PolyakAverager()
        avg.update(tiny_head)
        avg.update(tiny_head_alt)
        assert avg.count == 2
        sd = avg.state_dict()
        # Average of two state_dicts; verify dtype + shape preserved.
        for k, v in tiny_head.state_dict().items():
            assert sd[k].shape == v.shape

    def test_polyak_apply_before_update_raises(self, tiny_head: _TinyHead) -> None:
        avg = PolyakAverager()
        with pytest.raises(SWASchedulerError, match="before any update"):
            avg.apply(tiny_head)

    def test_polyak_state_dict_before_update_raises(self) -> None:
        with pytest.raises(SWASchedulerError, match="before any update"):
            PolyakAverager().state_dict()

    def test_swa_scheduler_validation(self) -> None:
        with pytest.raises(SWASchedulerError, match="must be >= 1"):
            SWAScheduler(total_epochs=0)
        with pytest.raises(SWASchedulerError, match="must be in \\[0, 1\\)"):
            SWAScheduler(total_epochs=100, swa_start_fraction=1.0)
        with pytest.raises(SWASchedulerError, match="must be >= 1"):
            SWAScheduler(total_epochs=100, update_every=0)

    def test_swa_scheduler_should_update(self) -> None:
        sched = SWAScheduler(
            total_epochs=100, swa_start_fraction=0.8, update_every=5
        )
        assert sched.swa_start_epoch == 80
        assert not sched.should_update(79)
        assert sched.should_update(80)
        assert not sched.should_update(81)
        assert sched.should_update(85)
        assert sched.should_update(95)
        assert not sched.should_update(100)


# ---------------------------------------------------------------------------
# pause_to_swap_loss
# ---------------------------------------------------------------------------


class TestPauseToSwapLoss:
    def test_loss_swap_record_validates(self) -> None:
        rec = LossSwap(
            epoch=50,
            old_loss_key="pixel",
            new_loss_key="kl_distill",
            optimizer_state_after="inherit",
            rationale="ok",
        )
        assert rec.epoch == 50

    def test_loss_swap_rejects_empty_rationale(self) -> None:
        with pytest.raises(LossSwapError, match="rationale must be non-empty"):
            LossSwap(
                epoch=0,
                old_loss_key="a",
                new_loss_key="b",
                optimizer_state_after="inherit",
                rationale="",
            )

    def test_swap_loss_inherit_returns_same_optimizer(
        self, tiny_head: _TinyHead
    ) -> None:
        opt = torch.optim.SGD(tiny_head.parameters(), lr=1e-3)
        new_loss = lambda p, t: F.mse_loss(p, t)  # noqa: E731
        new_loss_fn, new_opt, new_lr = swap_loss_at_pause(
            optimizer=opt,
            new_loss_fn=new_loss,
            optimizer_state_after="inherit",
        )
        assert new_opt is opt
        assert new_loss_fn is new_loss
        assert new_lr is None

    def test_swap_loss_reset_requires_factory(
        self, tiny_head: _TinyHead
    ) -> None:
        opt = torch.optim.SGD(tiny_head.parameters(), lr=1e-3)
        with pytest.raises(LossSwapError, match="requires optimizer_factory"):
            swap_loss_at_pause(
                optimizer=opt,
                new_loss_fn=lambda p, t: torch.zeros(1),
                optimizer_state_after="reset",
            )

    def test_swap_loss_reset_with_factory(
        self, tiny_head: _TinyHead
    ) -> None:
        opt = torch.optim.SGD(tiny_head.parameters(), lr=1e-3)

        def factory() -> torch.optim.Optimizer:
            return torch.optim.AdamW(tiny_head.parameters(), lr=1e-4)

        _, new_opt, _ = swap_loss_at_pause(
            optimizer=opt,
            new_loss_fn=lambda p, t: torch.zeros(1),
            optimizer_state_after="reset",
            optimizer_factory=factory,
        )
        assert isinstance(new_opt, torch.optim.AdamW)
        assert new_opt is not opt


# ---------------------------------------------------------------------------
# a1_pattern_inflate_time_bias_correction
# ---------------------------------------------------------------------------


class TestA1PatternBiasCorrection:
    def _plan(self, tmp_path: Path) -> A1PatternBiasCorrectionPlan:
        archive_path = tmp_path / "baseline.zip"
        archive_path.write_bytes(b"\x00" * 1024)
        import hashlib

        sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        return A1PatternBiasCorrectionPlan(
            substrate_id="test_sub",
            baseline_archive_path=str(archive_path),
            baseline_archive_sha256=sha,
            sweep_offsets=(100, 200),
            sweep_deltas=(+1, -1),
            held_out_metric_axis="contest-CPU",
            rationale="test",
        )

    def test_plan_validates(self, tmp_path: Path) -> None:
        plan = self._plan(tmp_path)
        assert plan.substrate_id == "test_sub"

    def test_plan_rejects_zero_delta(self, tmp_path: Path) -> None:
        archive_path = tmp_path / "baseline.zip"
        archive_path.write_bytes(b"\x00" * 1024)
        import hashlib

        sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        with pytest.raises(InflateBiasCorrectionError, match="cannot be 0"):
            A1PatternBiasCorrectionPlan(
                substrate_id="t",
                baseline_archive_path=str(archive_path),
                baseline_archive_sha256=sha,
                sweep_offsets=(0,),
                sweep_deltas=(0,),
                held_out_metric_axis="contest-CPU",
                rationale="r",
            )

    def test_plan_rejects_invalid_sha(self, tmp_path: Path) -> None:
        archive_path = tmp_path / "baseline.zip"
        archive_path.write_bytes(b"\x00" * 1024)
        with pytest.raises(InflateBiasCorrectionError, match="64-char hex"):
            A1PatternBiasCorrectionPlan(
                substrate_id="t",
                baseline_archive_path=str(archive_path),
                baseline_archive_sha256="too_short",
                sweep_offsets=(0,),
                sweep_deltas=(+1,),
                held_out_metric_axis="contest-CPU",
                rationale="r",
            )

    def test_plan_rejects_macos_cpu_advisory_axis(
        self, tmp_path: Path
    ) -> None:
        archive_path = tmp_path / "baseline.zip"
        archive_path.write_bytes(b"\x00" * 1024)
        import hashlib

        sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        with pytest.raises(InflateBiasCorrectionError, match="REFUSED"):
            A1PatternBiasCorrectionPlan(
                substrate_id="t",
                baseline_archive_path=str(archive_path),
                baseline_archive_sha256=sha,
                sweep_offsets=(0,),
                sweep_deltas=(+1,),
                held_out_metric_axis="macOS-CPU advisory",  # type: ignore[arg-type]
                rationale="r",
            )

    def test_corrector_grid_cartesian(self, tmp_path: Path) -> None:
        plan = self._plan(tmp_path)
        corr = GeneralizedInflateBiasCorrector(plan)
        grid = corr.grid()
        assert len(grid) == 4  # 2 offsets * 2 deltas
        assert (100, +1) in grid
        assert (200, -1) in grid

    def test_materialize_candidates(self, tmp_path: Path) -> None:
        plan = self._plan(tmp_path)
        corr = GeneralizedInflateBiasCorrector(plan)
        out_dir = tmp_path / "candidates"
        paths = corr.materialize_candidates(output_dir=out_dir)
        assert len(paths) == 4
        for (off, delta), p in paths.items():
            assert p.exists()
            data = p.read_bytes()
            # Verify the targeted byte was perturbed.
            assert data[off] == (0 + delta) % 256

    def test_classify_results(self, tmp_path: Path) -> None:
        plan = self._plan(tmp_path)
        corr = GeneralizedInflateBiasCorrector(plan)
        baseline = 0.1928
        # Synthetic results: one improvement, one regression, one within-noise.
        results = {
            (100, +1): ("a" * 64, 0.1920),  # improvement
            (100, -1): ("b" * 64, 0.1928),  # within-noise
            (200, +1): ("c" * 64, 0.1950),  # regression
            (200, -1): ("d" * 64, None),  # not evaluated
        }
        verdicts = corr.classify_results(
            baseline_score=baseline,
            results_per_candidate=results,
        )
        assert verdicts[(100, +1)].verdict == "IMPROVEMENT"
        assert verdicts[(100, -1)].verdict == "WITHIN_NOISE"
        assert verdicts[(200, +1)].verdict == "REGRESSION"
        assert verdicts[(200, -1)].verdict == "NOT_EVALUATED"

    def test_classify_results_rejects_grid_mismatch(
        self, tmp_path: Path
    ) -> None:
        plan = self._plan(tmp_path)
        corr = GeneralizedInflateBiasCorrector(plan)
        with pytest.raises(InflateBiasCorrectionError, match="grid mismatch"):
            corr.classify_results(
                baseline_score=0.1,
                results_per_candidate={(999, 999): ("0" * 64, 0.1)},
            )


# ---------------------------------------------------------------------------
# pause_distill_resume
# ---------------------------------------------------------------------------


class TestPauseDistillResume:
    def test_config_validates(self) -> None:
        cfg = DistillationConfig(temperature=2.0, rationale="ok")
        assert cfg.temperature == 2.0

    def test_config_rejects_non_positive_temperature(self) -> None:
        with pytest.raises(DistillationError, match="must be > 0"):
            DistillationConfig(temperature=0.0, rationale="r")

    def test_config_rejects_negative_kl_weight(self) -> None:
        with pytest.raises(DistillationError, match="must be >= 0"):
            DistillationConfig(kl_weight=-0.1, rationale="r")

    def test_config_rejects_empty_rationale(self) -> None:
        with pytest.raises(DistillationError, match="non-empty"):
            DistillationConfig(rationale="")

    def test_teacher_student_pair_freezes_teacher(
        self, tiny_head: _TinyHead, tiny_head_alt: _TinyHead
    ) -> None:
        cfg = DistillationConfig(rationale="freeze test")
        teacher, student = teacher_student_pair(
            teacher_module=tiny_head,
            student_module=tiny_head_alt,
            config=cfg,
        )
        for p in teacher.parameters():
            assert not p.requires_grad
        # Student is untouched.
        for p in student.parameters():
            assert p.requires_grad

    def test_teacher_student_rejects_same_module(
        self, tiny_head: _TinyHead
    ) -> None:
        cfg = DistillationConfig(rationale="ok")
        with pytest.raises(DistillationError, match="distinct module"):
            teacher_student_pair(
                teacher_module=tiny_head,
                student_module=tiny_head,
                config=cfg,
            )

    def test_kl_on_logits_computes_finite_loss(self) -> None:
        cfg = DistillationConfig(temperature=2.0, kl_weight=1.0, rationale="ok")
        student_logits = torch.randn(4, 5, requires_grad=True)
        teacher_logits = torch.randn(4, 5)
        loss = kl_on_logits_distillation(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            config=cfg,
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert student_logits.grad is not None
        # Teacher gradient flow MUST be blocked.
        assert not teacher_logits.requires_grad

    def test_kl_on_logits_rejects_shape_mismatch(self) -> None:
        cfg = DistillationConfig(rationale="ok")
        s = torch.randn(4, 5)
        t = torch.randn(4, 6)
        with pytest.raises(DistillationError, match="shape"):
            kl_on_logits_distillation(
                student_logits=s, teacher_logits=t, config=cfg
            )

    def test_kl_on_logits_temperature_scaling(self) -> None:
        # Equal logits → KL=0 regardless of temperature (sanity).
        cfg = DistillationConfig(temperature=2.0, rationale="ok")
        logits = torch.zeros(4, 5)
        loss = kl_on_logits_distillation(
            student_logits=logits, teacher_logits=logits, config=cfg
        )
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# pause_quantize_finetune
# ---------------------------------------------------------------------------


class TestPauseQuantizeFinetune:
    def test_plan_validates(self) -> None:
        p = PauseQuantizePlan(
            quantization_kind="int8_per_channel",
            mode="ptq",
            finetune_epochs=0,
            rationale="ok",
        )
        assert p.mode == "ptq"

    def test_plan_ptq_rejects_nonzero_finetune_epochs(self) -> None:
        with pytest.raises(PauseQuantizeFinetuneError, match=r"ptq.*== 0"):
            PauseQuantizePlan(
                quantization_kind="int8_per_channel",
                mode="ptq",
                finetune_epochs=5,
                rationale="r",
            )

    def test_plan_qat_ft_requires_finetune_epochs(self) -> None:
        with pytest.raises(PauseQuantizeFinetuneError, match=">= 1"):
            PauseQuantizePlan(
                quantization_kind="int8_per_channel",
                mode="qat_ft",
                finetune_epochs=0,
                rationale="r",
            )

    def test_apply_plan_ptq_path(self, tiny_head: _TinyHead) -> None:
        plan = PauseQuantizePlan(
            quantization_kind="int8_per_channel",
            mode="ptq",
            finetune_epochs=0,
            accuracy_recovery_threshold=0.1,
            rationale="ok",
        )

        def pre_val(_m: nn.Module) -> float:
            return 0.5

        def post_val(_m: nn.Module) -> float:
            return 0.52  # minor regression

        def quantize_fn(_m: nn.Module, _k: str) -> None:
            pass  # no-op stub

        result = apply_pause_quantize_finetune_plan(
            model=tiny_head,
            plan=plan,
            pre_quantize_validate_fn=pre_val,
            post_quantize_validate_fn=post_val,
            quantize_fn=quantize_fn,
        )
        assert result["mode"] == "ptq"
        assert result["rollback_invoked"] is False
        assert result["final_state_kind"] == "post_quantize"
        assert result["final_metric"] == pytest.approx(0.52)

    def test_apply_plan_rollback_on_large_regression(
        self, tiny_head: _TinyHead
    ) -> None:
        plan = PauseQuantizePlan(
            quantization_kind="int8_per_channel",
            mode="ptq",
            finetune_epochs=0,
            accuracy_recovery_threshold=0.01,
            rationale="ok",
        )

        result = apply_pause_quantize_finetune_plan(
            model=tiny_head,
            plan=plan,
            pre_quantize_validate_fn=lambda _: 0.5,
            post_quantize_validate_fn=lambda _: 0.6,  # +0.1 regression
            quantize_fn=lambda _m, _k: None,
        )
        assert result["rollback_invoked"] is True
        assert result["final_state_kind"] == "pre_quantize_rollback"

    def test_apply_plan_qat_ft_path(self, tiny_head: _TinyHead) -> None:
        plan = PauseQuantizePlan(
            quantization_kind="int4_per_channel",
            mode="qat_ft",
            finetune_epochs=2,
            accuracy_recovery_threshold=0.5,
            rationale="qat ft test",
        )

        def finetune_fn(_m: nn.Module, _e: int, _lr: float) -> float:
            return 0.48  # better than post-quantize

        result = apply_pause_quantize_finetune_plan(
            model=tiny_head,
            plan=plan,
            pre_quantize_validate_fn=lambda _: 0.5,
            post_quantize_validate_fn=lambda _: 0.55,
            quantize_fn=lambda _m, _k: None,
            finetune_fn=finetune_fn,
        )
        assert result["post_finetune_metric"] == pytest.approx(0.48)
        assert result["rollback_invoked"] is False
        assert result["final_state_kind"] == "post_finetune"

    def test_apply_plan_qat_ft_requires_finetune_fn(
        self, tiny_head: _TinyHead
    ) -> None:
        plan = PauseQuantizePlan(
            quantization_kind="int4_per_channel",
            mode="qat_ft",
            finetune_epochs=2,
            rationale="ok",
        )
        with pytest.raises(PauseQuantizeFinetuneError, match="requires finetune_fn"):
            apply_pause_quantize_finetune_plan(
                model=tiny_head,
                plan=plan,
                pre_quantize_validate_fn=lambda _: 0.5,
                post_quantize_validate_fn=lambda _: 0.5,
                quantize_fn=lambda _m, _k: None,
                finetune_fn=None,
            )


# ---------------------------------------------------------------------------
# early_stopping_with_resume
# ---------------------------------------------------------------------------


class TestEarlyStoppingWithResume:
    def test_tracker_validates_patience(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            EarlyStoppingTracker(
                patience=0,
                minimize=True,
                checkpoint_path=tmp_path / "x.pt",
            )

    def test_tracker_minimize_path(
        self, tmp_path: Path, tiny_head: _TinyHead
    ) -> None:
        ckpt_path = tmp_path / "best.pt"
        tracker = EarlyStoppingTracker(
            patience=3, minimize=True, checkpoint_path=ckpt_path
        )

        # Epoch 0: improvement 1.0 → save.
        assert tracker.update(tiny_head, 1.0, 0) is True
        assert tracker.state.best_metric == 1.0
        assert ckpt_path.exists()

        # Epoch 1-3: no improvement (0.9 saved actually counts, but here
        # 1.5, 1.1, 1.05 — all WORSE).
        assert tracker.update(tiny_head, 1.5, 1) is False
        assert tracker.update(tiny_head, 1.1, 2) is False
        assert tracker.update(tiny_head, 1.05, 3) is False
        assert tracker.state.stopped is True

    def test_tracker_maximize_path(
        self, tmp_path: Path, tiny_head: _TinyHead
    ) -> None:
        tracker = EarlyStoppingTracker(
            patience=2, minimize=False, checkpoint_path=tmp_path / "best.pt"
        )
        assert tracker.update(tiny_head, 0.5, 0) is True
        assert tracker.update(tiny_head, 0.6, 1) is True  # improvement
        assert tracker.update(tiny_head, 0.55, 2) is False
        assert tracker.update(tiny_head, 0.5, 3) is False
        assert tracker.state.stopped is True

    def test_resume_loads_best(
        self, tmp_path: Path, tiny_head: _TinyHead
    ) -> None:
        ckpt_path = tmp_path / "best.pt"
        tracker = EarlyStoppingTracker(
            patience=2, minimize=True, checkpoint_path=ckpt_path
        )
        tracker.update(tiny_head, 1.0, 0)
        # Mutate live state.
        with torch.no_grad():
            tiny_head.lin.weight.zero_()
        # Resume to restore.
        ResumeFromBestCheckpoint()(tiny_head, tracker)
        # Loaded state should match what was saved (non-zero random init).
        assert not torch.all(tiny_head.lin.weight == 0.0)

    def test_resume_raises_without_save(
        self, tmp_path: Path, tiny_head: _TinyHead
    ) -> None:
        tracker = EarlyStoppingTracker(
            patience=2, minimize=True, checkpoint_path=tmp_path / "ne.pt"
        )
        with pytest.raises(FileNotFoundError, match="best checkpoint not found"):
            ResumeFromBestCheckpoint()(tiny_head, tracker)
