# SPDX-License-Identifier: MIT
"""Tests for the increment-1a decoupling-partition-screen harness.

Coverage: assembler determinism + no_offset==argmax; meter vs a KNOWN fixture + per-class
sum identity; kill-criterion refusal paths + the three verdicts; reconciliation
monotonicity; control-arm matching rule; the analytic smoke end-to-end on a tiny subset.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.bitmask_dseg import d_seg_reference
from tac.boundary_math.laguerre_logit_offset import power_diagram_argmax
from tac.inc1a_harness.composite_assembler import (
    BC_MODES,
    CarrierField,
    Inc1aAssemblerError,
    assemble_fields,
    compose_partition,
    reconcile_partition,
)
from tac.inc1a_harness.decoupling_screen import (
    DELTA_MASK_FRAME_SAMPLING_FLOOR,
    VERDICT_CONFIRMED,
    VERDICT_INCONCLUSIVE,
    VERDICT_KILLED,
    VERDICT_REFUSED,
    ArmResult,
    DecouplingScreenError,
    evaluate_kill,
    matched_control_spec,
    operative_delta_mask,
)
from tac.inc1a_harness.mask_dseg_meter import (
    MaskDsegMeterError,
    measure_mask_dseg,
)

H, W, K = 12, 16, 5


def _random_phi(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((H, W, K)).astype(np.float32)


# --------------------------------------------------------------------------- assembler
def test_no_offset_equals_argmax_of_stack():
    phi = _random_phi(1)
    res = compose_partition(phi_hwk=phi, bc_mode="no_offset")
    assert np.array_equal(res.partition, np.argmax(phi, axis=-1))
    assert np.array_equal(res.offsets, np.zeros(K))
    assert res.reconcile_iters == 0


def test_assembler_is_deterministic():
    phi = _random_phi(2)
    a = compose_partition(phi_hwk=phi, bc_mode="no_offset").partition
    b = compose_partition(phi_hwk=phi, bc_mode="no_offset").partition
    assert np.array_equal(a, b)


def test_assemble_fields_places_carriers_and_complement():
    # lane carrier (class 1) with a +field patch wins over complement road (class 0, level 0).
    lane = np.full((H, W), -5.0, np.float32)
    lane[3:5, 3:5] = 2.0  # positive patch
    carriers = [CarrierField(class_id=1, phi_hw=lane, name="lane", geometric_home="G2")]
    phi = assemble_fields(
        carriers, shape=(H, W), complement_class=0, complement_level=0.0,
    )
    part = power_diagram_argmax(phi, np.zeros(K))
    assert (part[3:5, 3:5] == 1).all()  # lane wins where its field > 0
    assert part[0, 0] == 0  # complement road elsewhere


def test_duplicate_geometric_home_raises():
    f = np.zeros((H, W), np.float32)
    carriers = [
        CarrierField(class_id=1, phi_hw=f, name="a", geometric_home="G2"),
        CarrierField(class_id=1, phi_hw=f, name="b", geometric_home="G2"),
    ]
    with pytest.raises(Inc1aAssemblerError):
        assemble_fields(carriers, shape=(H, W), complement_class=0)


def test_carrier_rejects_bad_repr_mode():
    with pytest.raises(Inc1aAssemblerError):
        CarrierField(class_id=1, phi_hw=np.zeros((H, W), np.float32),
                     name="x", geometric_home="G2", representation_mode="WHATEVER")


def test_bc_modes_include_386_dispatcher():
    for m in ("no_offset", "menon", "ot_newton", "flip_weighted", "flip_median"):
        assert m in BC_MODES


def test_flip_median_bc_changes_nothing_when_perfect():
    # gt == argmax(phi): no flips -> flip_median offsets are zero -> partition unchanged.
    phi = _random_phi(3)
    gt = np.argmax(phi, axis=-1)
    res = compose_partition(phi_hwk=phi, bc_mode="flip_median", gt=gt)
    assert np.array_equal(res.partition, gt)


def test_unknown_bc_mode_raises():
    with pytest.raises(Inc1aAssemblerError):
        compose_partition(phi_hwk=_random_phi(4), bc_mode="not_a_mode")


# --------------------------------------------------------------------------- reconciliation
def test_reconcile_max_iter0_is_single_pass():
    phi = _random_phi(5)
    part, off, info = reconcile_partition(phi, bc_mode="no_offset", max_iter=0)
    assert info["reconcile_iters"] == 0
    assert np.array_equal(part, np.argmax(phi, axis=-1))


def test_reconcile_never_raises_dseg_vs_its_single_pass():
    # monotonicity REJECT guarantee: reconciled d_seg <= the SAME mode's single-pass d_seg
    # (reconcile never WORSENS the chosen bc_mode; it is a bounded local refinement of it).
    phi = _random_phi(6)
    gt = np.argmax(phi, axis=-1).copy()
    gt[0, :] = (gt[0, :] + 1) % K  # inject some flips so there is work to do
    single_pass, _o0, _i0 = reconcile_partition(phi, bc_mode="flip_median", gt=gt, max_iter=0)
    base = d_seg_reference(single_pass, gt)
    part, _off, _info = reconcile_partition(
        phi, bc_mode="flip_median", gt=gt, max_iter=3,
    )
    assert d_seg_reference(part, gt) <= base + 1e-12


def test_reconcile_requires_gt_when_iterating():
    with pytest.raises(Inc1aAssemblerError):
        reconcile_partition(_random_phi(7), bc_mode="no_offset", max_iter=2)


# --------------------------------------------------------------------------- meter
def test_meter_known_fixture():
    # 3 frames, each 4x4; flip a known number of pixels -> known d_seg.
    gt = np.zeros((3, 4, 4), dtype=np.int64)
    gt[:, :2, :] = 2  # top half Undrivable(2), bottom half Road(0)
    parts = [g.copy() for g in gt]
    parts[0][0, 0] = 1  # 1 flip in frame 0
    parts[1][0, 0] = 1
    parts[1][0, 1] = 1  # 2 flips in frame 1
    # frame 2 perfect. per-frame d_seg = 1/16, 2/16, 0 -> mean = (1/16+2/16)/3
    res = measure_mask_dseg(parts, gt, require_n600=False)
    assert res.agg_dseg == pytest.approx((1 / 16 + 2 / 16 + 0) / 3)
    # per-class sum identity: sum(flips)/sum(pixels) == d_seg over the SAME pairs.
    recomputed = res.total_flips / res.total_pixels
    micro_dseg = 3 / (3 * 16)  # 3 total flips over 3*16 pixels
    assert recomputed == pytest.approx(micro_dseg)


def test_meter_refuses_non_n600_by_default():
    gt = np.zeros((5, 4, 4), dtype=np.int64)
    parts = [g.copy() for g in gt]
    with pytest.raises(MaskDsegMeterError):
        measure_mask_dseg(parts, gt, require_n600=True)


def test_meter_shape_mismatch_raises():
    gt = [np.zeros((4, 4), np.int64)]
    parts = [np.zeros((4, 5), np.int64)]
    with pytest.raises(MaskDsegMeterError):
        measure_mask_dseg(parts, gt, require_n600=False)


# --------------------------------------------------------------------------- control arm
def test_matched_control_spec_window_and_match():
    spec = matched_control_spec(100_000, param_tolerance=0.05)
    lo, hi = spec.target_param_window()
    assert lo == 95_000 and hi == 105_000
    assert spec.params_match(102_000)
    assert not spec.params_match(120_000)
    cfg = spec.to_config_dict()
    assert cfg["arm"] == "control_shared_head" and cfg["paint_free"] is True
    assert cfg["param_window"] == [lo, hi]


def test_matched_control_spec_rejects_bad_count():
    with pytest.raises(DecouplingScreenError):
        matched_control_spec(0)


# --------------------------------------------------------------------------- kill criterion
def _arm(name: str, dseg: float, n: int = 600, toy: bool = False) -> ArmResult:
    return ArmResult(name=name, n_frames=n, agg_dseg=dseg, is_toy=toy)


def test_kill_confirmed_at_realistic_delta():
    # F-P5-P9-1: the floor is now the R7 MEASURED 3.46e-6, so a REALISTIC decoupling
    # improvement (0.005) clears it and returns a REAL CONFIRMED (the old 0.0196 proxy
    # made the screen decision-inert — 0.005 would have been INCONCLUSIVE).
    v = evaluate_kill(_arm("decoupled", 0.010), _arm("control", 0.015))
    assert v.verdict == VERDICT_CONFIRMED and v.passes_preregistered_gate
    assert v.delta_mask == pytest.approx(DELTA_MASK_FRAME_SAMPLING_FLOOR)
    assert "R7 MEASURED-ANCHOR" in v.delta_mask_provenance


def test_kill_killed_at_realistic_delta():
    v = evaluate_kill(_arm("decoupled", 0.015), _arm("control", 0.010))
    assert v.verdict == VERDICT_KILLED and not v.passes_preregistered_gate


def test_kill_inconclusive_within_floor():
    # a difference BELOW the 3.46e-6 frame-sampling floor is indistinguishable.
    v = evaluate_kill(
        _arm("decoupled", 0.010), _arm("control", 0.010 + DELTA_MASK_FRAME_SAMPLING_FLOOR / 2)
    )
    assert v.verdict == VERDICT_INCONCLUSIVE and not v.passes_preregistered_gate


def test_kill_uses_seed_spread_when_it_dominates():
    # F-P5-2: operative floor = max(frame-sampling, seed spread). With a measured seed spread
    # of 0.004 (>> 3.46e-6), a 0.002 improvement is now BELOW the floor -> INCONCLUSIVE.
    v = evaluate_kill(
        _arm("decoupled", 0.010), _arm("control", 0.012),
        seed_spread=0.004, n_seed_replicates=3,
    )
    assert v.delta_mask == pytest.approx(0.004)
    assert v.verdict == VERDICT_INCONCLUSIVE
    assert "seed spread" in v.delta_mask_provenance
    # a LARGER improvement (0.01) clears even the seed-dominated floor.
    v2 = evaluate_kill(
        _arm("decoupled", 0.010), _arm("control", 0.020),
        seed_spread=0.004, n_seed_replicates=3,
    )
    assert v2.verdict == VERDICT_CONFIRMED


def test_kill_refuses_when_seed_replicates_exist_but_spread_unprovided():
    # F-P5-2 / P2 seed honesty: >=2 replicates make the seed component MEASURABLE; dropping it
    # would fire a kill on within-seed noise. The evaluator REFUSES rather than silently using
    # the frame-sampling floor alone.
    v = evaluate_kill(
        _arm("decoupled", 0.010), _arm("control", 0.015),
        seed_spread=None, n_seed_replicates=3,
    )
    assert v.verdict == VERDICT_REFUSED and not v.passes_preregistered_gate
    assert "seed" in v.reason.lower()


def test_operative_delta_mask_floor_and_refusal():
    # single seed -> frame-sampling lower bound only.
    assert operative_delta_mask() == pytest.approx(DELTA_MASK_FRAME_SAMPLING_FLOOR)
    # seed spread dominates when larger.
    assert operative_delta_mask(seed_spread=0.01, n_seed_replicates=3) == pytest.approx(0.01)
    # frame-sampling floor dominates when seed spread is smaller.
    assert operative_delta_mask(seed_spread=1e-8, n_seed_replicates=3) == pytest.approx(
        DELTA_MASK_FRAME_SAMPLING_FLOOR
    )
    # replicates without a spread is under-specified -> REFUSE.
    with pytest.raises(DecouplingScreenError):
        operative_delta_mask(seed_spread=None, n_seed_replicates=3)
    with pytest.raises(DecouplingScreenError):
        operative_delta_mask(seed_spread=-1.0, n_seed_replicates=3)


def test_kill_explicit_delta_mask_override():
    # an explicit delta_mask bypasses the operative computation (e.g. a sensitivity sweep).
    v = evaluate_kill(_arm("decoupled", 0.010), _arm("control", 0.030), delta_mask=0.05)
    assert v.delta_mask == pytest.approx(0.05)
    assert v.verdict == VERDICT_INCONCLUSIVE  # 0.02 improvement < 0.05 floor
    assert "override" in v.delta_mask_provenance


def test_kill_refuses_toy_arm():
    v = evaluate_kill(_arm("decoupled", 0.010, toy=True), _arm("control", 0.030))
    assert v.verdict == VERDICT_REFUSED and not v.passes_preregistered_gate


def test_kill_refuses_non_n600_arm():
    v = evaluate_kill(_arm("decoupled", 0.010, n=96), _arm("control", 0.030))
    assert v.verdict == VERDICT_REFUSED


def test_kill_refuses_missing_arm():
    v = evaluate_kill(None, _arm("control", 0.030))  # type: ignore[arg-type]
    assert v.verdict == VERDICT_REFUSED


# --------------------------------------------------------------------------- analytic smoke (tiny subset)
def test_analytic_smoke_end_to_end_subset():
    import os

    from tac.inc1a_harness.analytic_smoke import DEFAULT_NPZ, run_analytic_smoke

    if not os.path.exists(DEFAULT_NPZ):
        pytest.skip("gt_n600 cache not present")
    res = run_analytic_smoke(DEFAULT_NPZ, n_frames=3, require_n600=False)
    assert 0.0 <= res.agg_dseg <= 1.0
    assert res.n_frames == 3 and not res.is_n600
    assert res.label == "[analytic-generators, no-trained-fields]"
    assert set(res.per_class_dseg) == {"Road", "Lane", "Undrivable", "Movable", "MyCar"}
