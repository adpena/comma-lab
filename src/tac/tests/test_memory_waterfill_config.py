# SPDX-License-Identifier: MIT
"""Tests for tools/memory_waterfill_config.py (BUILD #294 piece A).

Guards: the solver imports the REAL preflight projection (never forks it), excludes the
UNMEASURED micro-batch knob instead of inventing a curve, labels every ∝1/batch throughput
number [modeled], and picks the operator-policy answer (verdict-batch 64 at safe-frac 0.85 on the
fresh bank-4 config) by measured+modeled arithmetic, not assertion."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import memory_waterfill_config as mwc  # noqa: E402
import witness_memory_preflight as wmp  # noqa: E402

RAM = 128.0


# ── knob assessment: UNMEASURED => excluded, never invented ─────────────────────────────────────
def test_default_261_points_are_unmeasured_for_n600():
    st = mwc.assess_micro_batch(mwc.DEFAULT_MICRO_BATCH_POINTS, target_n_pairs=600)
    assert not st.measured
    assert st.grid == (1,)  # B pinned to the byte-identical serial path
    assert "UNMEASURED" in st.reason
    assert "contention" in st.reason or "n=" in st.reason


def test_uncontended_points_at_target_scale_unlock_the_knob():
    pts = (
        mwc.CurvePoint(knob="micro_batch", value=1, step_s=80.0, rss_mib=60000, n_pairs=600,
                       contended=False, source="hypothetical dedicated-GPU measure"),
        mwc.CurvePoint(knob="micro_batch", value=4, step_s=30.0, rss_mib=64000, n_pairs=600,
                       contended=False, source="hypothetical dedicated-GPU measure"),
    )
    st = mwc.assess_micro_batch(pts, target_n_pairs=600)
    assert st.measured
    assert st.grid == (1, 4)


def test_contended_points_at_target_scale_stay_unmeasured():
    pts = tuple(
        mwc.CurvePoint(knob="micro_batch", value=v, step_s=2.0, rss_mib=6000, n_pairs=600,
                       contended=True, source="contended") for v in (1, 4))
    st = mwc.assess_micro_batch(pts, target_n_pairs=600)
    assert not st.measured


def test_no_points_reports_missing_data():
    st = mwc.assess_micro_batch((), target_n_pairs=600)
    assert not st.measured and "no curve points" in st.reason


# ── verdict-wall model: labeled [modeled], anchored at the measured vb=32 wall ──────────────────
def test_verdict_wall_anchor_reproduces_measured():
    assert mwc.verdict_wall_s(32) == mwc.MEASURED_VERDICT_WALL_REF_S


def test_verdict_wall_is_inverse_in_batch():
    assert mwc.verdict_wall_s(64) == mwc.MEASURED_VERDICT_WALL_REF_S / 2.0
    assert mwc.verdict_wall_s(8) == mwc.MEASURED_VERDICT_WALL_REF_S * 4.0


def test_verdict_wall_refuses_unchunked_zero():
    import pytest

    with pytest.raises(ValueError):
        mwc.verdict_wall_s(0)


def test_async_hidden_exposes_only_overhang():
    # wall(64) = 1219.5 < train window 2062 => fully hidden
    assert mwc.exposed_verdict_wait_s(64) == 0.0
    # wall(32) = 2439 > 2062 => 377 s exposed (worst-wall vs fastest-window conservatism)
    assert abs(mwc.exposed_verdict_wait_s(32) - 377.0) < 1.0


def test_sync_mode_exposes_full_wall():
    assert mwc.exposed_verdict_wait_s(32, async_hidden=False) == mwc.MEASURED_VERDICT_WALL_REF_S


def test_throughput_multiplier_64_beats_32_async():
    m64 = mwc.throughput_multiplier(64)
    m32 = mwc.throughput_multiplier(32)
    assert m32 == 1.0
    assert m64 > 1.0  # the operator's 32->64 memory-leverage ask emerges from the arithmetic


def test_throughput_multiplier_small_batch_penalized():
    assert mwc.throughput_multiplier(8) < mwc.throughput_multiplier(32)


# ── the solver: real preflight projection, envelope, tie-breaks ─────────────────────────────────
def _solve(safe_frac=0.85, **kw):
    return mwc.solve_waterfill(num_pairs=600, total_ram_gib=RAM, safe_frac=safe_frac, **kw)


def test_solver_uses_real_preflight_numbers():
    res = _solve()
    c32 = next(c for c in res.candidates if c.verdict_batch == 32 and c.micro_batch == 1)
    ref = wmp.project_peak_rss_gib(num_pairs=600, verdict_batch=32, total_ram_gib=RAM,
                                   safe_frac=0.85)
    assert c32.projected_peak_gib == ref.projected_peak_gib == 67.61


def test_solver_best_is_vb64_at_085_envelope():
    """At the 0.85 single-workload envelope on the fresh bank-4 config, vb=64 is the argmax:
    equal-throughput ties (64/128/256 all fully hidden) break to LOWEST (adjusted) memory."""
    res = _solve()
    assert res.best is not None
    assert res.best.micro_batch == 1          # micro-batch UNMEASURED-excluded
    assert res.best.verdict_batch == 64
    assert res.best.projected_peak_gib == 68.65  # the FEED-04n dry-run number
    assert res.best.adjusted_peak_gib == 77.43   # net - 7.04 + (12.3 + 0.11*32) [modeled]
    assert res.best.throughput_label == mwc.MODELED_TAG


# ── realized-spike recalibration (mine 2026-07-04: +12.3 GiB, never scale from the +5.6) ────────
def test_realized_spike_anchor_is_measured_12_3():
    assert mwc.realized_verdict_spike_gib(32) == 12.3


def test_realized_spike_floors_below_anchor():
    # the ~+7 GiB pool climb is not assumed to shrink at smaller batches
    assert mwc.realized_verdict_spike_gib(8) == 12.3
    assert mwc.realized_verdict_spike_gib(16) == 12.3


def test_realized_spike_extrapolates_upward_with_measured_marginal():
    assert abs(mwc.realized_verdict_spike_gib(64) - (12.3 + wmp.VERDICT_PER_PAIR_GIB * 32)) < 1e-9


def test_adjusted_peak_rebases_on_realized_spike_not_component():
    res = _solve()
    c32 = next(c for c in res.candidates if c.verdict_batch == 32)
    # 67.61 - 6.0 (preflight component) + 12.3 (realized) = 73.91
    assert c32.adjusted_peak_gib == 73.91
    assert c32.adjusted_peak_gib > c32.projected_peak_gib  # conservative, fail-closed


def test_safety_is_the_conjunction_of_net_and_adjusted():
    # Envelope tight enough that net (68.65) fits but adjusted (77.43) does not => REFUSE.
    res = mwc.solve_waterfill(num_pairs=600, total_ram_gib=RAM, safe_frac=0.58,
                              verdict_batch_grid=(64,))
    c = res.candidates[0]
    assert c.projected_peak_gib <= res.envelope_gib < c.adjusted_peak_gib
    assert not c.safe


def test_solver_excludes_unsafe_candidates():
    res = _solve(verdict_batch_grid=(600,))
    # vb=600 => verdict 66 GiB => peak 127.61 > 108.8 envelope => REFUSE
    assert res.best is None
    assert all(not c.safe for c in res.candidates)


def test_solver_grid_boundary_vb256_safe_vb600_not():
    res = _solve(verdict_batch_grid=(256, 600))
    verdicts = {c.verdict_batch: c.safe for c in res.candidates}
    assert verdicts[256] is True and verdicts[600] is False


def test_envelope_070_still_admits_vb64():
    res = _solve(safe_frac=0.70)
    assert res.best is not None and res.best.verdict_batch == 64  # 68.65 <= 89.6


def test_floor_violation_refuses_everything():
    res = _solve(safe_frac=0.95)  # 128 - 121.6 = 6.4 < 10 GiB floor
    assert not res.floor_ok
    assert res.best is None
    assert any("control-plane floor" in n for n in res.notes)


def test_unmeasured_knob_is_reported_in_notes():
    res = _solve()
    assert any("UNMEASURED" in n for n in res.notes)


def test_measured_micro_batch_points_expand_the_grid():
    pts = (
        mwc.CurvePoint(knob="micro_batch", value=1, step_s=80.0, rss_mib=60000, n_pairs=600,
                       contended=False, source="hypothetical"),
        mwc.CurvePoint(knob="micro_batch", value=4, step_s=30.0, rss_mib=64000, n_pairs=600,
                       contended=False, source="hypothetical"),
    )
    res = _solve(micro_batch_points=pts)
    assert {c.micro_batch for c in res.candidates} == {1, 4}


def test_frontier_table_renders_every_candidate_and_labels():
    res = _solve()
    table = mwc.format_frontier_table(res)
    for c in res.candidates:
        assert f"{c.verdict_batch:>6}" in table
    assert mwc.MODELED_TAG in table
    assert "BEST" in table


def test_sync_verdict_mode_prefers_largest_batch():
    """Without async hiding (pre-M2 model), throughput strictly improves with vb => the solver
    spends memory up to the envelope (waterfill): best = largest SAFE vb."""
    res = _solve(async_hidden=False)
    safe_vbs = [c.verdict_batch for c in res.candidates if c.safe]
    assert res.best is not None and res.best.verdict_batch == max(safe_vbs)


def test_cli_json_smoke(capsys):
    rc = mwc.main(["--num-pairs", "600", "--total-ram-gib", "128", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"best"' in out and '"knob_status"' in out


def test_cli_table_smoke(capsys):
    rc = mwc.main(["--num-pairs", "600", "--total-ram-gib", "128"])
    assert rc == 0
    assert "WATERFILL frontier" in capsys.readouterr().out
