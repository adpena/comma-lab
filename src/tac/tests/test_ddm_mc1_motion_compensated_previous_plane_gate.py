"""Re-derivation guards for the ddm_mc1 alignment-gate equation.

The law was MEASURED by the ddm_mc1 arm; its memo is the primary artifact.  These tests do
not restate the memo -- they pin the arithmetic that the headline constants are built from
and the three decisions the equation exists to make:

  * the GATE: every decoder-derivable motion-compensated plane fails it on this field, and the
    oracle block plane (which reads the target) passes it -- so the gate separates the two;
  * the REFUSAL: the best decoder-derivable coder ceiling is below the pre-registered bar by
    more than two orders of magnitude, and the residual is that gap;
  * the CARRIED-MOTION arithmetic: carriage bytes are the persisted entropy over eight bits.
"""

from __future__ import annotations

import math

from tac.canonical_equations.motion_compensated_previous_plane_gate_20260904 import (
    BAND_AGREEMENT_COLOCATED,
    BAND_AGREEMENT_DERIVABLE,
    BAND_AGREEMENT_ORACLE,
    BARE_CATEGORICAL_BASELINE_BYTES,
    BARE_CATEGORICAL_CTX_MC_ALONE_BYTES,
    BLOCK_INDICATOR_MC_X_ARG_3SEED_MIN_BYTES,
    BLOCK_ROW_CORR_DY,
    CARRIED_ORACLE_BLOCK_MOTION_BITS,
    CARRIED_ORACLE_BLOCK_MOTION_BYTES,
    CARRIED_ORACLE_BLOCK_MOTION_BYTES_PER_PAIR,
    CEILING_BEST_HELD_OUT_BYTES,
    EQUATION_ID,
    INSTRUMENT_NOISE_FLOOR_BYTES,
    IOU_COLOCATED,
    IOU_LANE_DERIVABLE,
    IOU_LANE_ORACLE,
    ORACLE_BLOCK_CEILING_HELD_OUT_BYTES,
    ORACLE_BLOCK_CEILING_INDICATOR_3SEED_MIN_BYTES,
    PRIOR_LAW_PREDICTED_SAVING_BYTES,
    REFUSE_BELOW_BYTES,
    STREAM_BYTES,
    alignment_gain,
    build_motion_compensated_previous_plane_alignment_gate_v1,
    carried_motion_breakeven_open,
    ceiling_refused,
    plane_passes_alignment_gate,
    temporal_predictability_supports_extrapolation,
)


def test_every_derivable_plane_fails_the_alignment_gate() -> None:
    for family, iou_lane in IOU_LANE_DERIVABLE.items():
        assert not plane_passes_alignment_gate(
            iou_candidate_lane=iou_lane,
            iou_colocated_lane=IOU_COLOCATED["Lane"],
            band_candidate=BAND_AGREEMENT_DERIVABLE[family],
            band_colocated=BAND_AGREEMENT_COLOCATED,
        ), family
        assert alignment_gain(iou_lane, IOU_COLOCATED["Lane"]) < 0.0, family


def test_oracle_planes_pass_the_gate_so_the_gate_discriminates() -> None:
    for family, iou_lane in IOU_LANE_ORACLE.items():
        assert plane_passes_alignment_gate(
            iou_candidate_lane=iou_lane,
            iou_colocated_lane=IOU_COLOCATED["Lane"],
            band_candidate=BAND_AGREEMENT_ORACLE[family],
            band_colocated=BAND_AGREEMENT_COLOCATED,
        ), family


def test_ceiling_is_refused_by_more_than_thirty_fold() -> None:
    best = max(CEILING_BEST_HELD_OUT_BYTES.values())
    assert len(CEILING_BEST_HELD_OUT_BYTES) == 6  # all six decoder-derivable families priced
    assert CEILING_BEST_HELD_OUT_BYTES["block"] == best  # the block plane is the best derivable one
    assert ceiling_refused(best)
    assert ceiling_refused(best, REFUSE_BELOW_BYTES)
    assert REFUSE_BELOW_BYTES / best > 30.0
    assert best < 2e-3 * STREAM_BYTES
    assert best > INSTRUMENT_NOISE_FLOOR_BYTES  # a real, tiny signal -- not a null
    assert BLOCK_INDICATOR_MC_X_ARG_3SEED_MIN_BYTES < best  # the 3-seed minimum is the conservative row


def test_oracle_plane_is_refused_and_cannot_pay_its_carriage() -> None:
    # The most ANY member of the family can deliver through this instrument, with perfect motion.
    assert ceiling_refused(ORACLE_BLOCK_CEILING_HELD_OUT_BYTES)
    assert ORACLE_BLOCK_CEILING_INDICATOR_3SEED_MIN_BYTES < ORACLE_BLOCK_CEILING_HELD_OUT_BYTES
    assert ORACLE_BLOCK_CEILING_HELD_OUT_BYTES > max(CEILING_BEST_HELD_OUT_BYTES.values())
    assert not carried_motion_breakeven_open(
        carriage_bytes=CARRIED_ORACLE_BLOCK_MOTION_BYTES,
        oracle_plane_ceiling_bytes=ORACLE_BLOCK_CEILING_HELD_OUT_BYTES,
    )
    # Second instrument agrees on direction: mc alone beats coloc alone ONLY for the oracle plane.
    for family, ctx_mc in BARE_CATEGORICAL_CTX_MC_ALONE_BYTES.items():
        if family == "oracle_block":
            assert ctx_mc < BARE_CATEGORICAL_BASELINE_BYTES
        else:
            assert ctx_mc > BARE_CATEGORICAL_BASELINE_BYTES, family


def test_ceiling_residual_is_the_prior_law_gap() -> None:
    equation = build_motion_compensated_previous_plane_alignment_gate_v1()
    assert equation.equation_id == EQUATION_ID
    ceiling_anchor = next(a for a in equation.empirical_anchors if "ceiling" in a.anchor_id)
    best = max(CEILING_BEST_HELD_OUT_BYTES.values())
    assert math.isclose(ceiling_anchor.residual, PRIOR_LAW_PREDICTED_SAVING_BYTES - best)
    assert equation.predicted_vs_empirical_residual[ceiling_anchor.anchor_id] == ceiling_anchor.residual
    assert ceiling_anchor.empirical_output["typed_verdict"] == "CEILING-REFUSED"


def test_block_motion_is_not_extrapolable_on_any_road_row() -> None:
    for row, corr in BLOCK_ROW_CORR_DY.items():
        assert not temporal_predictability_supports_extrapolation(corr), row
    assert temporal_predictability_supports_extrapolation(0.9)


def test_carried_motion_arithmetic_and_breakeven() -> None:
    assert math.isclose(CARRIED_ORACLE_BLOCK_MOTION_BYTES, CARRIED_ORACLE_BLOCK_MOTION_BITS / 8.0)
    assert math.isclose(CARRIED_ORACLE_BLOCK_MOTION_BYTES_PER_PAIR, CARRIED_ORACLE_BLOCK_MOTION_BYTES / 599)
    assert CARRIED_ORACLE_BLOCK_MOTION_BYTES > PRIOR_LAW_PREDICTED_SAVING_BYTES
    # Even a carried plane must clear its own carriage: with the derivable ceilings it cannot.
    for best in CEILING_BEST_HELD_OUT_BYTES.values():
        assert not carried_motion_breakeven_open(
            carriage_bytes=CARRIED_ORACLE_BLOCK_MOTION_BYTES, oracle_plane_ceiling_bytes=best
        )
    assert carried_motion_breakeven_open(carriage_bytes=100.0, oracle_plane_ceiling_bytes=200.0)


def test_equation_builds_with_three_verified_anchors() -> None:
    equation = build_motion_compensated_previous_plane_alignment_gate_v1()
    assert len(equation.empirical_anchors) == 3
    assert all(a.empirical_verification_status == "VERIFIED_VIA_EMPIRICAL_ANCHOR" for a in equation.empirical_anchors)
    assert equation.canonical_producers and equation.canonical_consumers
