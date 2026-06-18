# SPDX-License-Identifier: MIT
"""Tests for the capacity-RD score-aware-QAT desk model (tac.capacity_rd_qat).

These verify the desk-model ARITHMETIC and the MEASURED-anchor consistency — NOT a
score claim (the module is [advisory] NON-PROMOTABLE by construction). Positive +
negative + consistency-with-measured-anchors + boundary cases.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from tac import capacity_rd_qat as crq


def test_score_matches_contest_formula():
    # S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/B0
    d_seg, d_pose, b = 0.0026, 0.00034, 89_136
    expected = 100 * d_seg + math.sqrt(10 * d_pose) + 25 * b / crq.RATE_DENOM
    assert crq.score(d_seg, d_pose, b) == pytest.approx(expected, rel=1e-12)


def test_bc20_anchor_reproduces_measured_score():
    # The bc20 anchor's components must reproduce the measured fp32 row score 0.37790.
    a = crq.ANCHOR_BC20
    s = crq.score(a.d_seg, a.d_pose, a.archive_bytes)
    assert s == pytest.approx(0.377897965875575, abs=1e-9)


def test_frontier_anchor_reproduces_pointer_score():
    # The frontier anchor must reproduce the canonical pointer S=0.19110 (d_pose backed out).
    a = crq.ANCHOR_FRONTIER
    s = crq.score(a.d_seg, a.d_pose, a.archive_bytes)
    assert s == pytest.approx(0.19109982419209975, abs=1e-9)
    assert a.d_seg == 0.00056  # symposium component value
    assert a.d_pose >= 0.0  # backed out, must be physical


def test_frontier_section_split_sums_to_archive():
    total = (
        crq.FRONTIER_DECODER_SECTION_BYTES
        + crq.FRONTIER_LATENT_SECTION_BYTES
        + crq.FRONTIER_SIDECAR_BYTES
        + crq.FRONTIER_OTHER_BYTES
    )
    assert total == crq.FRONTIER_ARCHIVE_BYTES
    # decoder dominates (~91%) — the QAT-attackable share.
    assert crq.FRONTIER_DECODER_SECTION_BYTES / crq.FRONTIER_ARCHIVE_BYTES > 0.88


def test_measured_byte_shrink_ratios_monotone():
    # Fewer bits -> fewer bytes (measured bc20 ratios), strictly decreasing.
    fracs = [crq.qat_byte_fraction(n) for n in (8, 7, 6, 5, 4)]
    assert fracs[0] == 1.0
    for a, b in pairwise(fracs):
        assert b < a
    # int4 ~ 0.52 of int8 (measured 46590/89136).
    assert crq.qat_byte_fraction(4) == pytest.approx(46590 / 89136, rel=1e-9)


def test_qat_byte_fraction_unknown_nbits_raises():
    with pytest.raises(ValueError):
        crq.qat_byte_fraction(3)


def test_qat_byte_fraction_mixed_bounds():
    # frac_low=0 -> int8 (1.0); frac_low=1 -> uniform low fraction.
    assert crq.qat_byte_fraction_mixed(0.0, 4) == pytest.approx(1.0)
    assert crq.qat_byte_fraction_mixed(1.0, 4) == pytest.approx(crq.qat_byte_fraction(4))
    mid = crq.qat_byte_fraction_mixed(0.5, 4)
    assert crq.qat_byte_fraction(4) < mid < 1.0


def test_qat_byte_fraction_mixed_invalid_frac_raises():
    with pytest.raises(ValueError):
        crq.qat_byte_fraction_mixed(1.5, 4)
    with pytest.raises(ValueError):
        crq.qat_byte_fraction_mixed(-0.1, 4)


def test_native_bytes_calibrated_to_bc20_anchor():
    # The native byte model must reproduce the bc20 measured archive bytes at bc20.
    nb = crq.native_archive_bytes(20)
    assert nb == pytest.approx(crq.ANCHOR_BC20.archive_bytes, abs=2)
    # Bytes grow with capacity.
    assert crq.native_archive_bytes(36) > crq.native_archive_bytes(20)


def test_dseg_at_capacity_measured_at_bc20_modelled_above():
    d20, ev20 = crq.dseg_at_capacity(20)
    assert d20 == crq.ANCHOR_BC20.d_seg
    assert ev20.startswith("MEAS")
    d28, ev28 = crq.dseg_at_capacity(28)
    assert ev28.startswith("MODEL")
    # d_seg decreases with capacity (power law).
    assert d28 < d20
    d36, _ = crq.dseg_at_capacity(36)
    assert d36 < d28


def test_run_desk_calc_native_dominated_by_frontier():
    # The native higher-capacity path is DOMINATED by the existing frontier — the
    # central honest finding. The best native/QAT S must be ABOVE the 0.19110 frontier.
    res = crq.run_desk_calc()
    assert res.frontier_S == pytest.approx(0.19109982419209975, abs=1e-9)
    assert res.argmin_native.native_S > res.frontier_S
    assert res.argmin_qat.qat_S > res.frontier_S
    # ... yet the QAT path still beats bc20 native (the pivot's literal win).
    assert res.argmin_qat.qat_S < res.bc20_native_S


def test_run_desk_calc_proceed_gate_fires():
    res = crq.run_desk_calc(proceed_threshold=0.30)
    assert res.proceed is True
    assert res.chosen is not None
    # STOP if threshold is below the best QAT S.
    res_stop = crq.run_desk_calc(proceed_threshold=0.20)
    assert res_stop.proceed is False
    assert res_stop.chosen is None


def test_frontier_qat_int4_perfect_hold_is_sub_015():
    # The decisive finding: QAT-shrinking the EXISTING frontier decoder to int4 with a
    # perfect distortion hold crosses sub-0.15 in the model.
    rows = crq.frontier_qat_rows()
    best = min(rows, key=lambda r: r.qat_S_perfect_hold)
    assert best.qat_S_perfect_hold < 0.15
    assert best.qat_nbits == 4
    assert best.frac_low_precision == 1.0
    # The +spill row beats the 0.191 frontier but misses sub-0.15 (the hold-quality crux).
    assert best.qat_S_with_spill < 0.19109982419209975
    assert best.qat_S_with_spill > 0.15


def test_frontier_qat_only_shrinks_decoder_section():
    # The non-decoder bytes (latents+sidecar+other) are held verbatim in every row.
    fixed = crq.FRONTIER_LATENT_SECTION_BYTES + crq.FRONTIER_SIDECAR_BYTES + crq.FRONTIER_OTHER_BYTES
    for r in crq.frontier_qat_rows():
        assert r.qat_archive_bytes == r.decoder_section_bytes + fixed
        assert r.decoder_section_bytes <= crq.FRONTIER_DECODER_SECTION_BYTES


def test_frontier_qat_rows_monotone_in_bits():
    # At fixed frac_low, fewer bits -> fewer bytes -> lower S(perfect hold).
    rows = [r for r in crq.frontier_qat_rows() if r.frac_low_precision == 1.0]
    rows.sort(key=lambda r: r.qat_nbits)
    for a, b in pairwise(rows):
        assert b.qat_archive_bytes > a.qat_archive_bytes  # higher bits = more bytes
        assert b.qat_S_perfect_hold > a.qat_S_perfect_hold
