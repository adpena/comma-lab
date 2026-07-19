# SPDX-License-Identifier: MIT
"""p0_resume_warmup_geometry_20260717 (#518) — resume/boundary-geometry build tests.

Covers the four surfaces of the P0 set:
  * the beta2-derived warmup-length law (adam_v_variance_warmup_length_v1) + its LawRef
    evaluator + sister-bound (c=1) consistency with rewarmup_beta2_memory_window_v1;
  * the new DSL Lever factories (ResumeLRWarmup / PoseEngageWPoseRamp / ForkHeadSolve /
    MarginStepCap / ForkEmaClearance / WarmStartRestoreBoundaryState) — values, validation,
    LawRef resolve == emitted override;
  * the trainer argparse holds every new flag with the byte-identity-preserving defaults
    (never-invent-flags: the DSL emits ONLY flags the parser owns);
  * the trainer's pure ramp helper `_stage_rewarmup_factor` (identity off / ramp on);
  * the launcher's `parse_dry_start_run_metrics` fork-verdict extraction (item 5b).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"
_LAUNCHER = _REPO / "tools" / "launch_witness_run.py"


def _load(path: pathlib.Path, name: str):
    if not path.exists():
        pytest.skip(f"module not found at {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"could not import {path.name}: {type(exc).__name__}: {exc}")
    return mod


# ---- item 2: the beta2-derived warmup-length law ----------------------------------------------
class TestAdamVVarianceWarmupLaw:
    def test_c2_defaults_derive_27_epochs(self):
        from tac.canonical_equations.adam_v_variance_warmup_20260717 import (
            adam_v_variance_warmup_epochs,
        )
        assert adam_v_variance_warmup_epochs(0.999, 75, c=2.0) == 27

    def test_c1_reproduces_sister_memory_bound(self):
        from tac.canonical_equations.adam_v_variance_warmup_20260717 import (
            adam_v_variance_warmup_epochs,
        )
        from tac.canonical_equations.curriculum_derivation_laws_20260705 import (
            min_rewarmup_epochs,
        )
        for beta2 in (0.99, 0.999, 0.9999):
            for spe in (1, 25, 75, 600):
                assert adam_v_variance_warmup_epochs(beta2, spe, c=1.0) == min_rewarmup_epochs(
                    beta2, spe
                ), f"c=1 must equal the sister bound at beta2={beta2}, spe={spe}"

    def test_monotone_in_beta2_and_c(self):
        from tac.canonical_equations.adam_v_variance_warmup_20260717 import (
            adam_v_variance_warmup_epochs,
        )
        assert adam_v_variance_warmup_epochs(0.9999, 75) > adam_v_variance_warmup_epochs(0.999, 75)
        assert adam_v_variance_warmup_epochs(0.999, 75, c=4.0) > adam_v_variance_warmup_epochs(
            0.999, 75, c=2.0
        )

    @pytest.mark.parametrize(
        "beta2,spe,c", [(0.0, 75, 2.0), (1.0, 75, 2.0), (0.999, 0, 2.0), (0.999, 75, 0.0)]
    )
    def test_rejects_out_of_domain(self, beta2, spe, c):
        from tac.canonical_equations.adam_v_variance_warmup_20260717 import (
            adam_v_variance_warmup_epochs,
        )
        with pytest.raises(ValueError):
            adam_v_variance_warmup_epochs(beta2, spe, c=c)

    def test_lawref_evaluator_registered_and_matches_callable(self):
        from tac.canonical_equations.adam_v_variance_warmup_20260717 import (
            adam_v_variance_warmup_epochs,
        )
        from tac.canonical_equations.evaluators import (
            populate_lawref_evaluators,
            resolve_equation_value,
        )
        assert "adam_v_variance_warmup_length_v1" in populate_lawref_evaluators()
        got = resolve_equation_value(
            "adam_v_variance_warmup_length_v1",
            {"beta2": 0.999, "steps_per_epoch": 75, "c": 2.0},
        )
        assert got == adam_v_variance_warmup_epochs(0.999, 75, c=2.0) == 27

    def test_evaluator_default_c(self):
        from tac.canonical_equations.evaluators import resolve_equation_value

        assert resolve_equation_value(
            "adam_v_variance_warmup_length_v1", {"beta2": 0.999, "steps_per_epoch": 75}
        ) == 27

    def test_equation_builds_with_provisional_domain(self):
        from tac.canonical_equations.adam_v_variance_warmup_20260717 import (
            build_adam_v_variance_warmup_length_v1,
        )
        eq = build_adam_v_variance_warmup_length_v1()
        assert eq.equation_id == "adam_v_variance_warmup_length_v1"
        dov = dict(eq.domain_of_validity)
        assert dov.get("score_claim") is False
        assert dov.get("promotion_eligible") is False
        assert "PROVISIONAL" in str(dov.get("note", ""))


# ---- item 2/3/4/6b/7/8: the DSL Lever factories -----------------------------------------------
class TestDSLLeverFactories:
    def test_resume_lr_warmup_derives_and_lawref_matches(self):
        from tac.witness_dsl.curriculum_dsl import ResumeLRWarmup
        from tac.witness_dsl.lawref import resolve

        lv = ResumeLRWarmup()
        assert lv.overrides["--stage-transition-rewarmup-epochs"] == 27
        assert lv.overrides["--stage-transition-rewarmup-floor"] == 0.1
        assert lv.overrides["--stage-transition-rewarmup-shape"] == "linear"
        ref = lv.lawrefs["--stage-transition-rewarmup-epochs"]
        assert ref.equation_id == "adam_v_variance_warmup_length_v1"
        resolved = resolve(ref)
        assert resolved.value == 27
        assert resolved.fallback_used is False
        # the config-of-record 8 is the recorded fallback (DERIVED > CONFIG ladder).
        assert ref.fallback == 8

    def test_resume_lr_warmup_validation(self):
        from tac.witness_dsl.curriculum_dsl import ResumeLRWarmup

        with pytest.raises(ValueError):
            ResumeLRWarmup(beta2=1.0)
        with pytest.raises(ValueError):
            ResumeLRWarmup(steps_per_epoch=0)
        with pytest.raises(ValueError):
            ResumeLRWarmup(c=0.0)
        with pytest.raises(ValueError):
            ResumeLRWarmup(floor=1.5)
        with pytest.raises(ValueError):
            ResumeLRWarmup(shape="step")

    def test_pose_engage_wpose_ramp(self):
        from tac.witness_dsl.curriculum_dsl import PoseEngageWPoseRamp

        assert PoseEngageWPoseRamp().overrides == {"--pose-engage-wpose-ramp": True}

    def test_fork_head_solve_modes_and_validation(self):
        from tac.witness_dsl.curriculum_dsl import ForkHeadSolve

        lv = ForkHeadSolve(mode="flip_median", tau=1.5, freeze_epochs=8)
        assert lv.overrides == {"--fork-head-solve": "flip_median",
                                "--fork-head-solve-tau": 1.5,
                                "--fork-head-freeze-epochs": 8}
        with pytest.raises(ValueError):
            ForkHeadSolve(mode="off")  # the DSL lever is the ON state; off = don't compose it
        with pytest.raises(ValueError):
            ForkHeadSolve(tau=0.0)
        with pytest.raises(ValueError):
            ForkHeadSolve(freeze_epochs=-1)

    def test_margin_step_cap(self):
        from tac.witness_dsl.curriculum_dsl import MarginStepCap

        lv = MarginStepCap(0.05)
        assert lv.overrides == {"--margin-step-cap": 0.05, "--margin-step-cap-window": -1}
        with pytest.raises(ValueError):
            MarginStepCap(0.0)

    def test_fork_ema_clearance_and_warm_start_restore(self):
        from tac.witness_dsl.curriculum_dsl import (
            ForkEmaClearance,
            WarmStartRestoreBoundaryState,
        )

        assert ForkEmaClearance().overrides == {"--fork-ema-clearance": True}
        assert WarmStartRestoreBoundaryState().overrides == {
            "--warm-start-restore-boundary-state": True}

    def test_all_new_flags_held_by_lever_registry(self):
        from tac.witness_dsl.lever_registry import completeness

        unmapped = set(completeness().unmapped or [])
        for flag in ("--pose-engage-wpose-ramp", "--fork-head-solve", "--fork-head-solve-tau",
                     "--fork-head-freeze-epochs", "--margin-step-cap", "--margin-step-cap-window",
                     "--fork-ema-clearance", "--warm-start-restore-boundary-state"):
            assert flag not in unmapped, f"{flag} is DSL-orphaned (config-orphan confound)"


# ---- trainer surfaces (argparse + pure ramp helper) -------------------------------------------
@pytest.fixture(scope="module")
def trainer_mod():
    pytest.importorskip("mlx", reason="level-set witness trainer requires mlx")
    return _load(_TRAINER, "_p0_518_trainer_under_test")


class TestTrainerFlagsAndRamp:
    def test_stage_rewarmup_factor_identity_and_ramp(self, trainer_mod):
        f = trainer_mod._stage_rewarmup_factor
        # OFF paths are EXACTLY 1.0 (byte-identity contract).
        assert f(100, None, 8, 0.1, "linear") == 1.0
        assert f(100, 90, 0, 0.1, "linear") == 1.0
        assert f(100, 90, 8, 0.1, "linear") == 1.0  # past the window
        # ramp: floor at the boundary epoch, monotone to 1.0.
        vals = [f(90 + d, 90, 8, 0.1, "linear") for d in range(8)]
        assert vals[0] == pytest.approx(0.1)
        assert all(b > a for a, b in zip(vals, vals[1:]))
        assert f(98, 90, 8, 0.1, "linear") == 1.0

    def test_argparse_flag_defaults_via_source(self):
        """Never-invent-flags: every DSL-emitted flag exists in the trainer argparse with the
        byte-identity default (off/0/off-mode). Source-level check (no mlx import needed)."""
        src = _TRAINER.read_text()
        for flag, default_frag in (
            ('"--pose-engage-wpose-ramp"', "default=False"),
            ('"--fork-head-solve"', 'default="off"'),
            ('"--fork-head-solve-tau"', "default=1.0"),
            ('"--fork-head-freeze-epochs"', "default=0"),
            ('"--margin-step-cap"', "default=0.0"),
            ('"--margin-step-cap-window"', "default=-1"),
            ('"--fork-ema-clearance"', "default=False"),
            ('"--warm-start-restore-boundary-state"', ""),
        ):
            i = src.find(flag)
            assert i >= 0, f"trainer argparse missing {flag}"
            if default_frag:
                window = src[i:i + 600]
                assert default_frag in window, (
                    f"{flag} default drifted from the byte-identity contract "
                    f"(expected {default_frag} nearby)")

    def test_widened_resume_trigger_is_outside_stiff_block(self):
        """Item 1 structural check: the boundary registration keys off `_retreatment` at the
        resume-block level (8-space indent), no longer nested under `if _stiff_added:`."""
        src = _TRAINER.read_text()
        i = src.find('"stage": "resume_lr_rewarmup"')
        assert i >= 0, "resume_lr_rewarmup telemetry row missing (item 1)"
        j = src.rfind("if _retreatment and int(getattr(args, "
                      '"stage_transition_rewarmup_epochs", 0) or 0) > 0:', 0, i)
        assert j >= 0
        line_start = src.rfind("\n", 0, j) + 1
        indent = j - line_start
        assert indent == 8, (
            f"the widened trigger must sit at resume-block level (8-space indent), got {indent}")

    def test_pose_engage_registers_boundary(self):
        """Item 6a structural check: pose_finish_engage registers last_boundary_epoch under the
        `not muon_switched` guard."""
        src = _TRAINER.read_text()
        i = src.find('"stage": "pose_finish_engage"')
        assert i >= 0
        window = src[max(0, i - 2500):i]
        assert "if not muon_switched:" in window and "last_boundary_epoch = ep" in window, (
            "pose_finish_engage no longer registers the LR-rewarmup boundary (item 6a)")


# ---- item 5b: the launcher fork-verdict extraction --------------------------------------------
@pytest.fixture(scope="module")
def launcher():
    return _load(_LAUNCHER, "_p0_518_launcher_under_test")


class TestLauncherForkVerdict:
    def test_extracts_baseline_v0_verdict(self, launcher, tmp_path):
        log = tmp_path / "run.log"
        rows = [
            {"stage": "gt", "secs": 120.5},
            {"stage": "resume_model_source", "resume_model_from": "ema"},
            {"stage": "resume_start_epoch", "resume_start_epoch": 651, "resume_ckpt_epoch": 650},
            {"stage": "verdict", "epoch": 650, "phase": "baseline_v0",
             "d_seg": 0.003366, "d_pose": 0.00161, "implied_S": 0.4747},
            {"stage": "checkpoint", "resume_latest": "levelset_resume_state.npz", "epoch": 651},
            {"ep": 651},
        ]
        log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        m = launcher.parse_dry_start_run_metrics(log)
        assert m["baseline_v0_d_seg"] == pytest.approx(0.003366)
        assert m["baseline_v0_d_pose"] == pytest.approx(0.00161)
        assert m["baseline_v0_implied_S"] == pytest.approx(0.4747)
        assert m["baseline_v0_skipped_reason"] is None
        # incumbent fields unaffected
        assert m["resume_start_epoch"] == 651 and m["resume_ckpt_epoch"] == 650
        assert m["checkpoint_written"] is True and m["epochs_completed"] == 651

    def test_skip_row_reason_recorded(self, launcher, tmp_path):
        log = tmp_path / "run.log"
        log.write_text(json.dumps({
            "stage": "baseline_verdict_skipped", "epoch": 650,
            "reason": "delta_bench_inherited_from prior_receipt"}) + "\n")
        m = launcher.parse_dry_start_run_metrics(log)
        assert m["baseline_v0_d_seg"] is None
        assert m["baseline_v0_skipped_reason"] == "delta_bench_inherited_from prior_receipt"

    def test_missing_file_all_absent(self, launcher, tmp_path):
        m = launcher.parse_dry_start_run_metrics(tmp_path / "nope.log")
        assert m["baseline_v0_d_seg"] is None
        assert m["baseline_v0_d_pose"] is None
        assert m["baseline_v0_implied_S"] is None
        assert m["baseline_v0_skipped_reason"] is None

    def test_in_loop_verdicts_do_not_pollute(self, launcher, tmp_path):
        log = tmp_path / "run.log"
        rows = [
            {"stage": "verdict", "epoch": 650, "phase": "baseline_v0", "d_seg": 0.0034,
             "d_pose": 0.002, "implied_S": 0.475},
            {"stage": "verdict", "epoch": 675, "d_seg": 0.0030, "d_pose": 0.0015,
             "implied_S": 0.470},
        ]
        log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        m = launcher.parse_dry_start_run_metrics(log)
        assert m["baseline_v0_d_seg"] == pytest.approx(0.0034), (
            "in-loop verdict rows (no phase=baseline_v0) must not overwrite the fork verdict")
