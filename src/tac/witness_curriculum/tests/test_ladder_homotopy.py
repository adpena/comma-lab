# SPDX-License-Identifier: MIT
"""Tests for the #323 FULL LADDER island-birth homotopy (per-class-λ-gated continuation).

Covers: the per-class-λ gate (below gate ⇒ ZERO support — the anti-uniform-amplification
guarantee), per-arm forms (movable dilation-GO release ceiling vs lane curve-prior), the LADDER
continuation schedule (eased-first r0 → anneal → 0), the 1-Lipschitz stepper (no hard switch),
LawRef consumption for r*, the measured λ proxy, the eased-mask per-class byte-identity, and the
DSL factory spelling + registry mapping (never-invent-flags).
"""
from __future__ import annotations

import inspect
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from tac.witness_curriculum.ladder_homotopy import (
    ARM_LANE,
    ARM_MOVABLE,
    LadderArmSpec,
    LadderHomotopy,
    homotopy_from_flags,
    perclass_lambda_proxy,
)


# --------------------------------------------------------------------------------------------
# Per-class-λ gate — the load-bearing anti-uniform-amplification guarantee
# --------------------------------------------------------------------------------------------
def test_gate_closed_below_floor_gives_zero_support():
    """λ_c strictly below the (hard) gate ⇒ EXACTLY zero support (no amplification of a won class)."""
    spec = LadderArmSpec(r0=2.0, lambda_gate=0.01, gate_softness=1e-9)
    assert spec.gate_multiplier(0.0) == 0.0
    assert spec.gate_multiplier(0.009) == 0.0
    h = homotopy_from_flags(lane_r0=2.0, lane_lambda_gate=0.01, gate_softness=1e-9)
    # gate closed ⇒ 0 regardless of the schedule being in its held-r0 window
    assert h.support_radius(ARM_LANE, 5, lambda_c=0.0) == 0.0


def test_gate_open_above_floor_gives_full_support():
    spec = LadderArmSpec(r0=2.0, lambda_gate=0.01, gate_softness=0.5)
    assert spec.gate_multiplier(0.01) == pytest.approx(1.0)
    assert spec.gate_multiplier(1.0) == pytest.approx(1.0)


def test_gate_is_fractional_and_continuous_across_the_soft_band():
    """The soft gate fades support continuously (no hard switch) across [gate·(1−soft), gate]."""
    spec = LadderArmSpec(r0=2.0, lambda_gate=0.10, gate_softness=0.5)  # band [0.05, 0.10]
    assert spec.gate_multiplier(0.05) == pytest.approx(0.0)
    mid = spec.gate_multiplier(0.075)
    assert 0.0 < mid < 1.0
    assert spec.gate_multiplier(0.10) == pytest.approx(1.0)
    # monotone non-decreasing across the band
    xs = np.linspace(0.05, 0.10, 21)
    ys = [spec.gate_multiplier(float(x)) for x in xs]
    assert all(b >= a - 1e-12 for a, b in pairwise(ys))


def test_lambda_gate_zero_is_ungated():
    """lambda_gate==0 ⇒ always-open (the movable dilation-GO default; sound independent of share)."""
    spec = LadderArmSpec(r0=2.0, lambda_gate=0.0)
    assert spec.gate_multiplier(0.0) == 1.0
    assert spec.gate_multiplier(-5.0) == 1.0


# --------------------------------------------------------------------------------------------
# LADDER continuation schedule — eased FIRST, anneal to the true target
# --------------------------------------------------------------------------------------------
def test_schedule_starts_eased_holds_then_anneals_to_zero():
    spec = LadderArmSpec(r0=2.0, birth_epochs=10, hold_epochs=0, anneal_epochs=20)
    assert spec.scheduled_radius(1) == pytest.approx(2.0)      # eased-first (winnable)
    assert spec.scheduled_radius(10) == pytest.approx(2.0)     # held through birth window
    assert 0.0 < spec.scheduled_radius(20) < 2.0               # mid-anneal
    assert spec.scheduled_radius(30) == pytest.approx(0.0)     # transfer complete (true target)
    assert spec.scheduled_radius(0) == 0.0                     # unborn before ep1


def test_schedule_is_1_lipschitz_across_the_anneal():
    """Per-epoch scheduled change never exceeds ~ r0·(max smoothstep slope 1.5)/anneal — smooth."""
    spec = LadderArmSpec(r0=2.0, birth_epochs=5, hold_epochs=0, anneal_epochs=200)
    prev = spec.scheduled_radius(6)
    for ep in range(7, 220):
        cur = spec.scheduled_radius(ep)
        assert abs(cur - prev) <= 2.0 * 1.5 / 200 + 1e-9
        prev = cur


def test_r0_zero_is_inert():
    spec = LadderArmSpec(r0=0.0, birth_epochs=10, anneal_epochs=10)
    assert all(spec.scheduled_radius(e) == 0.0 for e in range(0, 40))


# --------------------------------------------------------------------------------------------
# Per-arm forms — movable release ceiling (LawRef) vs lane curve-prior (no isotropic ceiling)
# --------------------------------------------------------------------------------------------
def test_movable_release_ceiling_consumes_lawref():
    """r* = coeff·σ_eff (LawRef critical_nucleus_release_v1). Default 0.95·1.5 = 1.425."""
    h = homotopy_from_flags(release_coeff=0.95, sigma_eff=1.5)
    assert h.release_ceiling(ARM_MOVABLE) == pytest.approx(1.425)
    # shrinks as σ_eff shrinks (τ anneals) — the on-schedule release
    assert h.release_ceiling(ARM_MOVABLE, sigma_eff=1.0) == pytest.approx(0.95)
    assert h.release_ceiling(ARM_MOVABLE, sigma_eff=0.5) == pytest.approx(0.475)


def test_movable_support_is_ceilinged_by_r_star():
    """Even with r0=3 held, movable support cannot exceed the nucleus-release ceiling r*."""
    h = homotopy_from_flags(movable_r0=3.0, movable_birth_epochs=50, release_coeff=0.95, sigma_eff=1.5)
    assert h.support_radius(ARM_MOVABLE, 5, lambda_c=float("inf")) == pytest.approx(1.425)


def test_lane_has_no_isotropic_nucleus_ceiling():
    """Lane's barrier is area/margin (not nucleus-radius) ⇒ no isotropic ceiling (+inf)."""
    h = homotopy_from_flags(lane_r0=2.0)
    assert h.release_ceiling(ARM_LANE) == float("inf")
    assert h.support_radius(ARM_LANE, 5, lambda_c=float("inf")) == pytest.approx(2.0)


def test_lane_dash_gate_zeroes_support_outside_the_dash_phase_window():
    h = homotopy_from_flags(lane_r0=2.0, lane_birth_epochs=10, lane_anneal_epochs=20,
                            lane_dash_gate=True)
    end = 10 + 0 + 20
    assert h.support_radius(ARM_LANE, 5, lambda_c=float("inf")) > 0.0
    assert h.support_radius(ARM_LANE, end + 5, lambda_c=float("inf")) == 0.0
    # OFF ⇒ the schedule alone governs (past-window schedule is already 0 here, so equal)
    h2 = homotopy_from_flags(lane_r0=2.0, lane_birth_epochs=10, lane_anneal_epochs=20,
                             lane_dash_gate=False)
    assert h2.support_radius(ARM_LANE, end + 5, lambda_c=float("inf")) == 0.0


# --------------------------------------------------------------------------------------------
# 1-Lipschitz stepper — the continuation adiabatic guard (no hard switch)
# --------------------------------------------------------------------------------------------
def test_step_radius_caps_the_per_epoch_change():
    h = homotopy_from_flags(movable_r0=2.0, max_step_px=1.0)
    # target 1.425 from prev 0.0 ⇒ step capped to 1.0
    assert h.step_radius(ARM_MOVABLE, 5, float("inf"), prev_radius=0.0) == pytest.approx(1.0)
    # next step reaches the target (Δ 0.425 < cap)
    assert h.step_radius(ARM_MOVABLE, 5, float("inf"), prev_radius=1.0) == pytest.approx(1.425)


def test_step_radius_downward_is_also_capped():
    h = homotopy_from_flags(movable_r0=2.0, max_step_px=0.5)
    # gate slams shut (target 0) but the step only releases 0.5 px — a controlled continuation
    assert h.step_radius(ARM_MOVABLE, 500, 0.0, prev_radius=1.425) == pytest.approx(1.425 - 0.5)


def test_rung_rounds_the_stepped_radius():
    h = homotopy_from_flags(movable_r0=2.0, max_step_px=1.0, release_coeff=0.95, sigma_eff=1.5)
    assert h.rung(ARM_MOVABLE, 5, float("inf"), prev_radius=1.0) == 1   # round(1.425)
    assert h.rung(ARM_MOVABLE, 500, float("inf"), prev_radius=0.0) == 0  # past anneal
    assert h.rung(ARM_MOVABLE, 5, float("inf"), prev_radius=None) >= 0


# --------------------------------------------------------------------------------------------
# Measured λ proxy
# --------------------------------------------------------------------------------------------
def test_perclass_lambda_proxy_is_share_times_rate():
    assert perclass_lambda_proxy(0.02, 0.19) == pytest.approx(0.02 * 0.19)
    assert perclass_lambda_proxy(0.0, 0.5) == 0.0      # won class ⇒ 0 marginal value
    assert perclass_lambda_proxy(-1.0, 0.5) == 0.0     # clamped non-negative
    assert perclass_lambda_proxy(0.5, -1.0) == 0.0


# --------------------------------------------------------------------------------------------
# Validation / guards
# --------------------------------------------------------------------------------------------
def test_arm_spec_rejects_bad_params():
    with pytest.raises(ValueError):
        LadderArmSpec(r0=-1.0)
    with pytest.raises(ValueError):
        LadderArmSpec(gate_softness=0.0)
    with pytest.raises(ValueError):
        LadderArmSpec(gate_softness=1.5)
    with pytest.raises(ValueError):
        LadderArmSpec(lambda_gate=-0.1)


def test_homotopy_rejects_bad_params_and_unknown_arm():
    with pytest.raises(ValueError):
        LadderHomotopy(release_coeff=-1.0)
    with pytest.raises(ValueError):
        LadderHomotopy(sigma_eff_default=0.0)
    with pytest.raises(ValueError):
        LadderHomotopy(max_step_px=0.0)
    with pytest.raises(ValueError):
        homotopy_from_flags().support_radius("bogus", 1, 1.0)


# --------------------------------------------------------------------------------------------
# Composition — eased_island_masks per-class radii default to byte-identity
# --------------------------------------------------------------------------------------------
def _toy_lstar(h: int = 24, w: int = 24) -> np.ndarray:
    a = np.full((h, w), 2, dtype=np.int64)          # Undrivable background
    a[10:14, 4:20] = 1                               # a lane stripe (curve class)
    a[16:20, 8:14] = 3                               # a movable blob
    return a


def test_eased_masks_per_class_radius_defaults_to_shared_dilate_px():
    from tac.boundary_math.island_protection import eased_island_masks

    a = _toy_lstar()
    base = eased_island_masks(a, lane_cls=1, movable_cls=3, dilate_px=2)
    # lane_px/movable_px None ⇒ both use dilate_px=2 ⇒ identical to the single-radius call
    same = eased_island_masks(a, lane_cls=1, movable_cls=3, dilate_px=2,
                              lane_px=None, movable_px=None)
    assert np.array_equal(base.any_mask, same.any_mask)
    assert np.array_equal(base.lane_mask, same.lane_mask)
    assert np.array_equal(base.movable_mask, same.movable_mask)


def test_eased_masks_per_class_radius_grows_classes_independently():
    from tac.boundary_math.island_protection import eased_island_masks

    a = _toy_lstar()
    # movable radius 0 ⇒ movable at TRUE target (gate-closed); lane still grown at 2
    m = eased_island_masks(a, lane_cls=1, movable_cls=3, dilate_px=2, lane_px=2, movable_px=0)
    assert int(m.movable_mask.sum()) == int((a == 3).sum())    # movable = true target (no easing)
    assert int(m.lane_mask.sum()) >= int((a == 1).sum())        # lane grown along tangent
    # lane radius 0 ⇒ lane at true target
    m2 = eased_island_masks(a, lane_cls=1, movable_cls=3, dilate_px=2, lane_px=0, movable_px=2)
    assert int(m2.lane_mask.sum()) == int((a == 1).sum())


# --------------------------------------------------------------------------------------------
# DSL factory — spelling + registry mapping (never-invent-flags)
# --------------------------------------------------------------------------------------------
def test_dsl_factory_returns_lever_with_expected_flags():
    from tac.witness_dsl.curriculum_dsl import LadderIslandHomotopy

    lv = LadderIslandHomotopy()
    assert lv.name == "n323_ladder_island_homotopy"
    assert lv.overrides["--ladder-island-homotopy"] is True
    assert lv.overrides["--amplify-weight"] == 1.0
    assert lv.overrides["--ladder-lane-r0"] == 2.0
    assert lv.overrides["--ladder-movable-r0"] == pytest.approx(
        2.0 / 8.881199197033954, rel=1e-15
    )
    assert (
        lv.overrides["--ladder-lane-r0"] / lv.overrides["--ladder-movable-r0"]
        == pytest.approx(8.881199197033954, rel=1e-15)
    )
    # a store-bool flag must be a bool (else compile emits a bad value)
    assert isinstance(lv.overrides["--ladder-lane-dash-gate"], bool)
    for f in ("--ladder-movable-r0", "--ladder-lane-lambda-gate", "--ladder-release-coeff",
              "--ladder-sigma-eff", "--ladder-max-step-px"):
        assert f in lv.overrides


def test_dsl_factory_has_one_absolute_scale_knob_and_derives_both_radii():
    from tac.witness_dsl.curriculum_dsl import LadderIslandHomotopy

    parameters = inspect.signature(LadderIslandHomotopy).parameters
    assert "absolute_scale" in parameters
    assert parameters["absolute_scale"].default == 2.0
    assert "lane_r0" not in parameters
    assert "movable_r0" not in parameters

    lv = LadderIslandHomotopy(absolute_scale=4.0, amplify_weight=0.75)
    assert lv.overrides["--ladder-lane-r0"] == 4.0
    assert lv.overrides["--ladder-movable-r0"] == pytest.approx(
        4.0 / 8.881199197033954, rel=1e-15
    )
    assert lv.overrides["--amplify-weight"] == 0.75
    for bad in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="absolute_scale must be finite and > 0"):
            LadderIslandHomotopy(absolute_scale=bad)


def test_dsl_factory_resolves_receipt_backed_lawrefs_and_exposes_manifest():
    from tac.canonical_equations.chan_vese_area_constraint_birth_balance_20260708 import (
        ISLAND_BIRTH_RATIO_RECEIPT_PATH,
        ISLAND_BIRTH_RATIO_RECEIPT_SHA256,
    )
    from tac.witness_dsl.curriculum_dsl import LadderIslandHomotopy
    from tac.witness_dsl.lawref import LawRef

    lv = LadderIslandHomotopy()
    assert set(lv.lawrefs) == {"--ladder-lane-r0", "--ladder-movable-r0"}
    assert set(lv.constant_manifest) == set(lv.lawrefs)
    for flag, lawref in lv.lawrefs.items():
        assert isinstance(lawref, LawRef)
        assert lawref.inputs["class_p_over_a"].kind == "anchor"
        assert lawref.inputs["reference_p_over_a"].kind == "anchor"
        assert lawref.inputs["class_p_over_a"].artifact_path == ISLAND_BIRTH_RATIO_RECEIPT_PATH
        assert lawref.inputs["class_p_over_a"].expected_sha256 == ISLAND_BIRTH_RATIO_RECEIPT_SHA256
        manifest = lv.constant_manifest[flag]
        assert manifest["equation_id"] == "isoperimetric_birth_weight_scaling_v1"
        assert manifest["fallback_used"] is False
        assert manifest["value"] == lv.overrides[flag]
        assert all(row["sha256"] == ISLAND_BIRTH_RATIO_RECEIPT_SHA256
                   for row in manifest["inputs"] if row["kind"] == "anchor")
    assert all(not isinstance(value, LawRef) for value in lv.overrides.values())


def test_dsl_factory_fails_closed_if_receipt_is_tampered(tmp_path, monkeypatch):
    from tac.canonical_equations import chan_vese_area_constraint_birth_balance_20260708 as law
    from tac.witness_dsl.curriculum_dsl import LadderIslandHomotopy
    from tac.witness_dsl.lawref import LawResolveError

    receipt = json.loads(Path(law.ISLAND_BIRTH_RATIO_RECEIPT_PATH).read_text())
    receipt["classes"]["movable"]["p_over_a"] = 1.0
    tampered = tmp_path / "tampered_receipt.json"
    tampered.write_text(json.dumps(receipt))
    monkeypatch.setattr(law, "ISLAND_BIRTH_RATIO_RECEIPT_PATH", str(tampered))
    with pytest.raises(LawResolveError, match="sha256 mismatch"):
        LadderIslandHomotopy()


def test_ladder_default_off_preserves_program_argv_identity():
    from tac.witness_dsl.curriculum_dsl import BASELINE, LadderIslandHomotopy

    before = BASELINE.compile_trainer_argv()
    lever = LadderIslandHomotopy()
    after = BASELINE.compile_trainer_argv()
    assert after == before
    for flag in ("--ladder-island-homotopy", "--ladder-lane-r0", "--ladder-movable-r0"):
        assert flag not in before

    composed = BASELINE.with_lever(lever).compile_trainer_argv()
    assert "--ladder-island-homotopy" in composed
    assert str(lever.overrides["--ladder-lane-r0"]) in composed
    assert str(lever.overrides["--ladder-movable-r0"]) in composed


def test_dsl_factory_flags_all_exist_in_trainer_argparse():
    """Never-invent-flags: every flag the factory emits must exist in the levelset trainer."""
    import re
    from pathlib import Path

    from tac.witness_dsl.curriculum_dsl import LadderIslandHomotopy

    src = Path("experiments/train_levelset_witness_realized_through_R_mlx.py").read_text()
    trainer_flags = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src))
    for f in LadderIslandHomotopy().overrides:
        assert f in trainer_flags, f"invented flag not in trainer argparse: {f}"


def test_registry_maps_all_ladder_flags():
    from tac.witness_dsl import lever_registry as R

    facs = R.lever_factories()
    assert "LadderIslandHomotopy" in facs
    emitted = facs["LadderIslandHomotopy"]
    assert "--ladder-island-homotopy" in emitted
    # none of the ladder flags remain UNMAPPED by the DSL
    comp = R.completeness()
    assert not [f for f in comp.unmapped if f.startswith("--ladder-")]


def test_ladder_is_never_fired_and_duty_to_measure_on_empty_ledger(tmp_path):
    from tac.witness_dsl.activation_ledger import duty_to_measure, known_levers, never_fired

    empty = tmp_path / "empty_activation_ledger.jsonl"
    assert "LadderIslandHomotopy" in known_levers()
    assert "LadderIslandHomotopy" in never_fired(path=empty)
    assert "LadderIslandHomotopy" in duty_to_measure(path=empty)
