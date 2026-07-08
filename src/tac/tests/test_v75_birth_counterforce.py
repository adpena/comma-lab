"""Tests for the crucible v7.5 birth-counter-force stack (road_anomaly_probe_20260708.md):

  * Lever-1 CHAN-VESE AREA CONSTRAINT — the balance law + numpy reference + DSL factory + trainer wiring.
  * Lever-2 MORSE-SMALE BIRTH-COMPLETION EVENT — the predicate + ramp + controller + resume roundtrip.
  * Lever-3 LOGIT-ADJUST REGIME COHERENCE — the companion law + offset mask.
  * v7.5 composition — the three levers land in the emitted argv; byte-identity when off.

means != ends: advisory apparatus; pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.canonical_equations.chan_vese_area_constraint_birth_balance_20260708 import (
    DEFAULT_AREA_TOLERANCE,
    MEASURED_GT_AREA_N600,
    MEASURED_PART_FRAC_EP125,
    area_constraint_lambda,
    area_penalty,
    build_chan_vese_area_constraint_birth_balance_v1,
    dominance_at_runaway,
    equilibrium_overshoot,
)
from tac.witness_control.birth_completion import (
    BirthCompletionController,
    birth_completion_restore_from_cfg,
    birth_completion_state_arrays,
    birth_complete,
    birth_persistence,
    birth_ramp_multiplier,
)
from tac.witness_dsl.curriculum_dsl import (
    AreaConstraintBirth,
    BirthCompletionEvent,
    logit_adjust_classes_for_basis_regime,
)


# ============================ Lever-1: Chan-Vese area-constraint balance =========================

class TestAreaConstraintBalanceLaw:
    def test_lambda_matches_derivation(self):
        # lambda_c = birth_force / (tolerance * A_GT); measured lane/movable from the module docstring.
        assert area_constraint_lambda(0.00585) == pytest.approx(683.8, rel=1e-3)
        assert area_constraint_lambda(0.0124) == pytest.approx(322.6, rel=1e-3)

    def test_equilibrium_overshoot_is_tolerance_times_gt(self):
        # A* - A_GT = birth_force / lambda = tolerance * A_GT (the balance law).
        for gt in (0.00585, 0.0124, 0.05):
            lam = area_constraint_lambda(gt, birth_force=1.0, tolerance=DEFAULT_AREA_TOLERANCE)
            over = equilibrium_overshoot(lam, birth_force=1.0)
            assert over == pytest.approx(DEFAULT_AREA_TOLERANCE * gt, rel=1e-9)

    def test_dominance_at_measured_ep125_runaway(self):
        # the operator's requirement: at the 13.8x/4.6x runaway the retraction DOMINATES the birth force.
        dl = dominance_at_runaway(MEASURED_PART_FRAC_EP125[1], MEASURED_GT_AREA_N600[1])
        dm = dominance_at_runaway(MEASURED_PART_FRAC_EP125[3], MEASURED_GT_AREA_N600[3])
        assert dl == pytest.approx(51.0, abs=1.0) and dl > 1.0
        assert dm == pytest.approx(14.3, abs=1.0) and dm > 1.0

    def test_dominance_independent_of_birth_force(self):
        # the ratio cancels F_birth (it is a property of the overshoot ratio + tolerance).
        assert dominance_at_runaway(0.08, 0.00585, tolerance=0.25) == pytest.approx(
            (0.08 / 0.00585 - 1.0) / 0.25, rel=1e-9)

    def test_penalty_one_sided_zero_below_gt(self):
        # a class UNDER its GT area pays nothing (nucleation unopposed).
        m = np.array([0.20, 0.001, 0.49, 0.005, 0.25])   # lane/movable UNDER gt
        assert area_penalty(m, np.array(MEASURED_GT_AREA_N600), (1, 3)) == 0.0

    def test_penalty_positive_on_overshoot_and_quadratic(self):
        gt = np.array(MEASURED_GT_AREA_N600)
        m1 = gt.copy(); m1[1] += 0.01
        m2 = gt.copy(); m2[1] += 0.02
        p1 = area_penalty(m1, gt, (1,))
        p2 = area_penalty(m2, gt, (1,))
        assert p1 > 0.0
        assert p2 == pytest.approx(4.0 * p1, rel=1e-6)  # quadratic: 2x overshoot => 4x penalty

    def test_lambda_fail_closed(self):
        with pytest.raises(ValueError):
            area_constraint_lambda(0.01, tolerance=0.0)
        with pytest.raises(ValueError):
            area_constraint_lambda(-0.1)

    def test_area_floor_prevents_inf_for_absent_class(self):
        lam = area_constraint_lambda(0.0)     # absent class => floored, finite
        assert np.isfinite(lam) and lam > 0.0

    def test_equation_builds_and_registers_shape(self):
        eq = build_chan_vese_area_constraint_birth_balance_v1()
        assert eq.equation_id == "chan_vese_area_constraint_birth_balance_v1"
        assert "tac.witness_dsl.curriculum_dsl" in eq.canonical_consumers
        assert len(eq.empirical_anchors) == 2


class TestAreaConstraintDSL:
    def test_factory_emits_real_flags(self):
        lev = AreaConstraintBirth()
        ov = lev.overrides
        assert ov["--area-constraint-birth"] is True
        assert ov["--area-constraint-birth-force"] == 1.0
        assert ov["--area-constraint-tolerance"] == 0.25
        assert ov["--area-constraint-classes"] == "1,3"
        assert lev.epochs_delta == 0

    def test_factory_fail_closed(self):
        with pytest.raises(ValueError):
            AreaConstraintBirth(tolerance=0.0)
        with pytest.raises(ValueError):
            AreaConstraintBirth(birth_force=-1.0)

    def test_factory_custom_params(self):
        ov = AreaConstraintBirth(birth_force=2.0, tolerance=0.1, classes="3").overrides
        assert ov["--area-constraint-birth-force"] == 2.0
        assert ov["--area-constraint-tolerance"] == 0.1
        assert ov["--area-constraint-classes"] == "3"


# ============================ Lever-2: Morse-Smale birth completion ==============================

class TestBirthCompletionPredicate:
    def test_persistence_is_one_minus_within_flip(self):
        assert birth_persistence(0.1) == pytest.approx(0.9)
        assert birth_persistence(0.0) == 1.0
        assert birth_persistence(1.0) == 0.0
        assert birth_persistence(-0.5) == 1.0     # clamped
        assert birth_persistence(1.5) == 0.0      # clamped

    def test_complete_requires_both_persistence_and_area(self):
        gt = 0.00585
        # both satisfied
        assert birth_complete(gt, gt, 0.1)  # persist 0.9, area == gt
        # persistence too low
        assert not birth_complete(gt, gt, 0.5)
        # area too high (over-paint)
        assert not birth_complete(0.08, gt, 0.1)
        # area too low (under-born)
        assert not birth_complete(gt * 0.5, gt, 0.1)

    def test_area_band_edges(self):
        gt = 0.01
        assert birth_complete((1 - 0.25) * gt, gt, 0.0, area_band=0.25)    # lo edge in-band
        assert birth_complete((1 + 0.25) * gt, gt, 0.0, area_band=0.25)    # hi edge in-band
        assert not birth_complete((1 + 0.26) * gt, gt, 0.0, area_band=0.25)

    def test_unscored_class_never_complete(self):
        assert not birth_complete(0.0, 0.0, 0.0)

    def test_predicate_fail_closed(self):
        with pytest.raises(ValueError):
            birth_complete(0.01, 0.01, 0.1, tau_persist=1.5)
        with pytest.raises(ValueError):
            birth_complete(0.01, 0.01, 0.1, area_band=-0.1)


class TestBirthRampMultiplier:
    def test_not_fired_full_pressure(self):
        assert birth_ramp_multiplier(None, 500) == 1.0

    def test_ramp_linear_to_post_level(self):
        assert birth_ramp_multiplier(100, 100, ramp_epochs=50, post_level=0.0) == 1.0
        assert birth_ramp_multiplier(100, 125, ramp_epochs=50, post_level=0.0) == pytest.approx(0.5)
        assert birth_ramp_multiplier(100, 150, ramp_epochs=50, post_level=0.0) == pytest.approx(0.0)
        assert birth_ramp_multiplier(100, 999, ramp_epochs=50, post_level=0.0) == pytest.approx(0.0)

    def test_ramp_to_nonzero_post_level(self):
        assert birth_ramp_multiplier(100, 150, ramp_epochs=50, post_level=0.3) == pytest.approx(0.3)
        assert birth_ramp_multiplier(100, 125, ramp_epochs=50, post_level=0.3) == pytest.approx(0.65)

    def test_ramp_monotone_non_increasing_after_fire(self):
        vals = [birth_ramp_multiplier(100, e, ramp_epochs=50, post_level=0.0) for e in range(100, 200)]
        assert all(a >= b for a, b in zip(vals, vals[1:]))


class TestBirthCompletionController:
    def _stats(self, part_frac, gt_area, within_flip, cls=1, total=1000):
        return {cls: {"part_frac": part_frac, "within_flip": within_flip,
                      "gt_px": int(gt_area * total), "pred_px": int(part_frac * total),
                      "total_px": total, "gt_area": gt_area}}

    def test_observe_latches_completion(self):
        c = BirthCompletionController(classes=(1,))
        # not complete: over-paint
        assert c.observe(50, self._stats(0.08, 0.00585, 0.1)) == []
        assert 1 not in c.fired
        # complete: in-band + persistent
        rows = c.observe(120, self._stats(0.006, 0.00585, 0.1))
        assert len(rows) == 1 and rows[0]["event"] == "fired" and rows[0]["class"] == 1
        assert c.fired[1] == 120

    def test_latch_is_monotone(self):
        c = BirthCompletionController(classes=(1,))
        c.observe(120, self._stats(0.006, 0.00585, 0.1))
        # a later dip out of band does NOT un-latch (birth completion is monotone)
        c.observe(200, self._stats(0.08, 0.00585, 0.9))
        assert c.fired[1] == 120

    def test_multiplier_reflects_fire(self):
        c = BirthCompletionController(classes=(1,), ramp_epochs=50, post_level=0.0)
        assert c.multiplier(1, 500) == 1.0             # not fired
        c.observe(120, self._stats(0.006, 0.00585, 0.1))
        assert c.multiplier(1, 120) == 1.0
        assert c.multiplier(1, 145) == pytest.approx(0.5)
        assert c.multiplier(1, 170) == pytest.approx(0.0)

    def test_resume_roundtrip_bit_faithful(self):
        c = BirthCompletionController(classes=(1, 3), tau_persist=0.8, area_band=0.25,
                                      ramp_epochs=50, post_level=0.0)
        c.observe(120, self._stats(0.006, 0.00585, 0.1, cls=1))
        arrays = birth_completion_state_arrays(c)
        # emulate a checkpoint sidecar dict
        cfg = {k: np.asarray(v) for k, v in arrays.items()}
        r = birth_completion_restore_from_cfg(cfg)
        assert r is not None
        assert r.classes == (1, 3)
        assert r.fired == {1: 120}
        # the restored controller reproduces the identical multiplier trajectory
        assert r.multiplier(1, 145) == pytest.approx(0.5)
        assert r.multiplier(3, 145) == 1.0             # class 3 never fired

    def test_state_arrays_empty_when_none(self):
        assert birth_completion_state_arrays(None) == {}

    def test_restore_none_when_absent(self):
        assert birth_completion_restore_from_cfg({}) is None
        assert birth_completion_restore_from_cfg(None) is None


class TestBirthCompletionDSL:
    def test_factory_emits_real_flags(self):
        ov = BirthCompletionEvent().overrides
        assert ov["--birth-completion-event"] is True
        assert ov["--birth-completion-tau-persist"] == 0.8
        assert ov["--birth-completion-area-band"] == 0.25
        assert ov["--birth-completion-ramp-epochs"] == 50
        assert ov["--birth-completion-post-level"] == 0.0
        assert ov["--birth-completion-classes"] == "1,3"

    def test_factory_fail_closed(self):
        with pytest.raises(ValueError):
            BirthCompletionEvent(tau_persist=1.5)
        with pytest.raises(ValueError):
            BirthCompletionEvent(ramp_epochs=0)


# ============================ Lever-3: logit-adjust regime coherence =============================

class TestLogitAdjustRegimeCoherence:
    def test_lane_offloaded_drops_lane(self):
        # under lane_offloaded, lane rides the analytic band => drop it from the boost (movable only).
        assert logit_adjust_classes_for_basis_regime("lane_offloaded") == "3"

    def test_lane_carried_keeps_all(self):
        assert logit_adjust_classes_for_basis_regime("lane_carried") == "all"

    def test_unknown_regime_fail_closed(self):
        with pytest.raises(ValueError):
            logit_adjust_classes_for_basis_regime("bogus")

    def test_custom_class_indices(self):
        assert logit_adjust_classes_for_basis_regime("lane_offloaded", movable_class=4) == "4"


# ============================ Lever-3: trainer-side offset mask ==================================

def _load_trainer():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_lv_trainer_for_test", "experiments/train_levelset_witness_realized_through_R_mlx.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestLogitAdjustClassMaskTrainer:
    def test_mask_bitmask_stable(self):
        t = _load_trainer()
        assert t._logit_adjust_classes_mask("all") == -1
        assert t._logit_adjust_classes_mask("3") == 8          # 2^3
        assert t._logit_adjust_classes_mask("1,3") == 10       # 2^1 + 2^3
        assert t._logit_adjust_classes_mask("") == 0

    def test_offset_masking_zeros_unlisted(self):
        t = _load_trainer()
        ls = [np.array([[0, 1, 2, 3, 4]])]                     # one pixel per class
        off_all, _ = t._logit_adjust_offsets_np(ls, 1.0)
        off_mov, _ = t._logit_adjust_offsets_np(ls, 1.0, allowed_classes=(3,))
        # every class BUT 3 zeroed; class 3 unchanged.
        for c in range(5):
            if c == 3:
                assert off_mov[c] == pytest.approx(off_all[c])
            else:
                assert off_mov[c] == 0.0

    def test_offset_masking_none_is_incumbent(self):
        t = _load_trainer()
        ls = [np.array([[0, 1, 2, 3, 4]])]
        off_none, _ = t._logit_adjust_offsets_np(ls, 1.0, allowed_classes=None)
        off_default, _ = t._logit_adjust_offsets_np(ls, 1.0)
        assert np.allclose(off_none, off_default)


# ============================ v7.5 composition ==================================================

class TestCrucibleV75Compose:
    def _emit(self):
        from tac.witness_autoconfig import compile_crucible_v7_config
        comp = compile_crucible_v7_config(
            "experiments/results/mlx_fleet_gt_cache/gt_n600.npz", num_pairs=8, epochs=3000)
        argv = comp.typed.to_program().compile_trainer_argv()
        pairs = {}
        i = 0
        while i < len(argv):
            t = argv[i]
            if t.startswith("--"):
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    pairs[t] = argv[i + 1]; i += 2
                else:
                    pairs[t] = True; i += 1
            else:
                i += 1
        return pairs

    def test_all_three_levers_composed(self):
        p = self._emit()
        # Lever-1 area constraint
        assert p.get("--area-constraint-birth") is True
        assert p.get("--area-constraint-classes") == "1,3"
        assert float(p["--area-constraint-tolerance"]) == 0.25
        # Lever-2 completion event
        assert p.get("--birth-completion-event") is True
        assert p.get("--birth-completion-classes") == "1,3"
        # Lever-3 regime coherence: lane dropped from the logit-adjust boost under lane_offloaded
        assert p.get("--logit-adjust-classes") == "3"
        # and the logit-adjust boost itself is still on (v6 base carries it)
        assert float(p["--logit-adjust-loss-tau"]) == 1.0

    def test_regime_coherence_persistence_and_logit_agree(self):
        # both regime-derived class subsets exclude lane under lane_offloaded (coherent).
        p = self._emit()
        assert p.get("--persistence-classes") == "3"
        assert p.get("--logit-adjust-classes") == "3"

    def test_every_emitted_flag_is_declared(self):
        # never-invent-flags: every emitted flag must exist in the trainer argparse (else argparse crash).
        import re
        src = open("experiments/train_levelset_witness_realized_through_R_mlx.py").read()
        declared = set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', src))
        for f in ("--area-constraint-birth", "--area-constraint-birth-force",
                  "--area-constraint-tolerance", "--area-constraint-classes",
                  "--birth-completion-event", "--birth-completion-tau-persist",
                  "--birth-completion-area-band", "--birth-completion-ramp-epochs",
                  "--birth-completion-post-level", "--birth-completion-classes",
                  "--logit-adjust-classes"):
            assert f in declared, f"emitted flag {f} not declared in trainer argparse"
