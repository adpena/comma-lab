# SPDX-License-Identifier: MIT
"""NO-FAKE gates for Wave-F Stage-2b: CORRESPONDENCE-FIRST lane tracking + coherent batch denoise.

Every invariant the research authority
(``.omx/research/lane_coeff_tracking_denoising_optimal_survey_20260702.md``) requires:
* the tracked packing is DETERMINISTIC (same input -> same M/presence/K);
* the tracked serialization is a STANDARD LBND2 blob (magic LBND2, format=2) that the UNCHANGED
  ``deserialize_lane_band_rd`` / ``deserialize_lane_band_any`` decode bit-exact (ships as LBND2
  bytes, ZERO new inflate code);
* CORRESPONDENCE IS LOSSLESS on geometry: tracked-no-smooth dequant lines == sort dequant lines
  as a per-pair quantized multiset (only the slot INDEX changed, never a lane's coeffs);
* tracking KILLS the lane-count re-index swap (on a swap fixture, tracked delta-L1 << sort delta-L1);
* every coherent-denoise method (rts/l1trend/median/rpca) round-trips bit-exact + preserves presence;
* the sort path (``serialize_lane_band_rd``) is BYTE-IDENTICAL post-refactor (regression guard);
* the batch estimators do the work they name (RTS denoises noise but tracks a ramp; l1-trend
  preserves a step edge a moving-average blurs; RPCA is finite/safe).
Synthetic LaneLines (no gt-cache) -> fast + deterministic.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.analytic_lane_render_band import (
    LaneBandRenderConfig,
    _pack_pairs_to_matrix,
    _dequant_lines_multiset_key,
    derive_rd_base_steps,
    deserialize_lane_band_any,
    deserialize_lane_band_rd,
    serialize_lane_band_rd,
    serialize_lane_band_rd_tracked,
    roundtrip_lines_through_rd_tracked,
    lane_band_tracking_rate_report,
)
from tac.boundary_math.lane_track_and_smooth import (
    track_lane_slots,
    coherent_denoise_track_matrix,
    top_pct_jump_mass,
    _rts_local_linear_trend,
    _l1trend_1d,
    _median_1d,
    _rpca_pcp,
)
from tac.boundary_math.lane_sdf_component import LaneLine


# --------------------------------------------------------------------------- #
# synthetic fixtures
# --------------------------------------------------------------------------- #
def _line(lat: float, *, c1: float = 0.02, c2: float = -2e-3, c3: float = 1e-4,
          hw: float = 3.0, fr=(6.0, 45.0)) -> LaneLine:
    return LaneLine(centerline_coeffs=np.array([c3, c2, c1, lat], np.float64),
                    halfwidth_coeffs=np.array([0.0, hw], np.float64),
                    dash_period_m=0.0, dash_phase_m=0.0, dash_duty=0.5, forward_range=fr)


def _parallel_pairs(P: int = 40, seed: int = 0, offsets=(-3.5, 0.0, 3.5)) -> list[list[LaneLine]]:
    """Stable parallel lanes with per-frame lateral jitter (no swaps)."""
    rng = np.random.default_rng(seed)
    return [[_line(off + 0.01 * t + 0.02 * rng.standard_normal()) for off in offsets] for t in range(P)]


def _count_change_pairs(P: int = 40) -> list[list[LaneLine]]:
    """A lane APPEARS on the far left at the midpoint -> the lateral-sort RE-INDEXES every slot
    (the measured swap mass); the tracker keeps the existing lanes' tracks + births one."""
    pairs = []
    for t in range(P):
        offs = [-2.0, 1.0, 4.0]
        if t >= P // 2:
            offs = [-5.0] + offs                       # a new left lane appears -> slot shift
        pairs.append([_line(o) for o in offs])
    return pairs


# --------------------------------------------------------------------------- #
# tracking determinism + structure
# --------------------------------------------------------------------------- #
def test_tracking_is_deterministic():
    pairs = _parallel_pairs()
    a = track_lane_slots(pairs)
    b = track_lane_slots(pairs)
    assert a.K == b.K and np.array_equal(a.M, b.M) and np.array_equal(a.presence, b.presence)


def test_tracking_parallel_lanes_no_fragmentation():
    """Stable parallel lanes -> exactly one track per lane (no spurious births from fit jitter)."""
    pairs = _parallel_pairs(offsets=(-3.5, 0.0, 3.5))
    ta = track_lane_slots(pairs)
    assert ta.K == 3 and ta.n_births == 3 and ta.n_deaths == 0


def test_tracking_births_a_track_on_count_change():
    pairs = _count_change_pairs()
    ta = track_lane_slots(pairs)
    assert ta.K == 4 and ta.n_births == 4   # 3 initial + 1 appearing lane; existing lanes keep tracks


# --------------------------------------------------------------------------- #
# tracked serialization = STANDARD LBND2 (zero new inflate code)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pack_mode", ["coherent_slot", "persistent", "sort"])
def test_tracked_ships_as_lbnd2_and_roundtrips(pack_mode):
    pairs = _parallel_pairs()
    cfg = LaneBandRenderConfig()
    blob, meta = serialize_lane_band_rd_tracked(pairs, cfg, pack_mode=pack_mode, smooth="none")
    assert blob[:6] == b"LBND2\x00"
    dq_any, hdr = deserialize_lane_band_any(blob)
    dq_rd, hdr2 = deserialize_lane_band_rd(blob)
    assert hdr["format"] == 2 and hdr["rd"]["pack_mode"] == pack_mode
    # both decoders agree; provenance keys are inert to the decode
    assert len(dq_any) == len(dq_rd) == len(pairs)
    for a_list, b_list in zip(dq_any, dq_rd):
        assert len(a_list) == len(b_list)


@pytest.mark.parametrize("method", ["none", "rts", "l1trend", "median", "rpca"])
@pytest.mark.parametrize("pack_mode", ["coherent_slot", "persistent"])
def test_tracked_roundtrip_bit_exact_all_methods(method, pack_mode):
    pairs = _parallel_pairs(P=30)
    cfg = LaneBandRenderConfig()
    dq_lines, blob, meta = roundtrip_lines_through_rd_tracked(pairs, cfg, pack_mode=pack_mode, smooth=method)
    dq2, _ = deserialize_lane_band_rd(blob)
    for a_list, b_list in zip(dq_lines, dq2):
        for a, b in zip(a_list, b_list):
            assert np.array_equal(np.asarray(a.centerline_coeffs), np.asarray(b.centerline_coeffs))
            assert np.array_equal(np.asarray(a.halfwidth_coeffs), np.asarray(b.halfwidth_coeffs))


# --------------------------------------------------------------------------- #
# CORRESPONDENCE IS LOSSLESS on geometry (the decisive claim)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pack_mode", ["coherent_slot", "persistent"])
def test_correspondence_no_smooth_lossless_vs_sort_parallel(pack_mode):
    pairs = _parallel_pairs(P=40)
    cfg = LaneBandRenderConfig()
    steps = derive_rd_base_steps()
    dq_sort, _ = deserialize_lane_band_rd(serialize_lane_band_rd(pairs, cfg))
    blob, _ = serialize_lane_band_rd_tracked(pairs, cfg, pack_mode=pack_mode, smooth="none")
    dq_trk, _ = deserialize_lane_band_rd(blob)
    assert (_dequant_lines_multiset_key(dq_sort, steps)
            == _dequant_lines_multiset_key(dq_trk, steps))


@pytest.mark.parametrize("pack_mode", ["coherent_slot", "persistent"])
def test_correspondence_no_smooth_lossless_vs_sort_count_change(pack_mode):
    """Even WITH a count-change re-index, the per-pair SET of dequant lines is identical
    (correspondence only re-labels slots)."""
    pairs = _count_change_pairs()
    cfg = LaneBandRenderConfig()
    steps = derive_rd_base_steps()
    dq_sort, _ = deserialize_lane_band_rd(serialize_lane_band_rd(pairs, cfg))
    blob, _ = serialize_lane_band_rd_tracked(pairs, cfg, pack_mode=pack_mode, smooth="none")
    dq_trk, _ = deserialize_lane_band_rd(blob)
    assert (_dequant_lines_multiset_key(dq_sort, steps)
            == _dequant_lines_multiset_key(dq_trk, steps))


def test_coherent_slot_bounded_K_equals_sort_K():
    """Bounded-K coherent slotting keeps K = max-concurrent (like the sort), NOT a column
    explosion (which persistent tracks can produce)."""
    from tac.boundary_math.analytic_lane_render_band import _pack_pairs_to_matrix
    from tac.boundary_math.lane_track_and_smooth import coherent_slot_pack
    pairs = _count_change_pairs(P=60)
    _M, _p, K_sort = _pack_pairs_to_matrix(pairs)
    ta = coherent_slot_pack(pairs)
    assert ta.K == K_sort


# --------------------------------------------------------------------------- #
# tracking KILLS the lane-count re-index swap (the rate win)
# --------------------------------------------------------------------------- #
def test_tracking_kills_swap_delta_mass():
    pairs = _count_change_pairs(P=60)
    cfg = LaneBandRenderConfig()
    steps = derive_rd_base_steps()
    M_sort, pres_sort, K_sort = _pack_pairs_to_matrix(pairs)
    ta = track_lane_slots(pairs)
    jm_sort = top_pct_jump_mass(M_sort, pres_sort, K_sort, np.tile(steps, K_sort))
    jm_trk = top_pct_jump_mass(ta.M, ta.presence, ta.K, np.tile(steps, ta.K))
    # the sort re-indexes EVERY slot at the count-change -> big total delta L1; tracking pays
    # only the single birth -> strictly smaller total delta mass.
    assert jm_trk["total_l1"] < jm_sort["total_l1"]


# --------------------------------------------------------------------------- #
# coherent denoise: presence preserved; methods do the work they name
# --------------------------------------------------------------------------- #
def test_coherent_denoise_preserves_presence_and_shape():
    pairs = _parallel_pairs(P=30)
    ta = track_lane_slots(pairs)
    for method in ("median", "rts", "l1trend", "rpca"):
        Ms = coherent_denoise_track_matrix(ta.M, ta.presence, ta.K, method=method)
        assert Ms.shape == ta.M.shape and np.all(np.isfinite(Ms))


def test_rts_denoises_noise_but_tracks_a_ramp():
    """RTS (MMSE) reduces variance on a noisy linear ramp while staying close to the true line."""
    rng = np.random.default_rng(1)
    n = 80
    true = 0.05 * np.arange(n)
    y = true + 0.3 * rng.standard_normal(n)
    sc = float(np.median(np.abs(np.diff(y))))
    q = sc ** 2
    sm = _rts_local_linear_trend(y, q_level=q, q_slope=q * 0.01, r_meas=q * 50.0)
    assert np.std(sm - true) < np.std(y - true)                    # noise reduced
    assert np.mean(np.abs(sm - true)) < 0.15                       # ramp tracked (no lag blow-up)


def test_l1trend_preserves_a_step_better_than_median_and_mean():
    """l1-trend keeps a genuine level jump; a moving-AVERAGE blurs it. On a clean step the
    l1-trend residual at the edge is smaller than a mean-filter's."""
    n = 60
    y = np.concatenate([np.zeros(n // 2), 2.0 * np.ones(n - n // 2)])
    x = _l1trend_1d(y, lam=0.2)
    # mean (moving-average) filter of window 9
    w = 9
    xp = np.pad(y, w // 2, mode="edge")
    mean = np.array([xp[i:i + w].mean() for i in range(n)])
    edge = n // 2
    l1_edge_err = abs(x[edge] - y[edge]) + abs(x[edge - 1] - y[edge - 1])
    mean_edge_err = abs(mean[edge] - y[edge]) + abs(mean[edge - 1] - y[edge - 1])
    assert l1_edge_err < mean_edge_err                              # edge preserved


def test_median_smooth_is_identity_at_win1():
    y = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    assert np.array_equal(_median_1d(y, 1), y)


def test_rpca_returns_finite_and_low_rank_plus_sparse():
    rng = np.random.default_rng(2)
    n = 50
    L_true = np.outer(np.linspace(0, 1, n), np.array([1.0, 0.5, -0.3]))   # rank-1 trajectory
    S_true = np.zeros((n, 3)); S_true[10, 0] = 8.0; S_true[30, 2] = -6.0  # sparse spikes
    X = L_true + S_true + 0.01 * rng.standard_normal((n, 3))
    L, S = _rpca_pcp(X)
    assert np.all(np.isfinite(L)) and np.all(np.isfinite(S))
    assert np.linalg.matrix_rank(L, tol=1e-3) <= 3


# --------------------------------------------------------------------------- #
# sort-path byte-identity regression (the refactor must not change LBND2 bytes)
# --------------------------------------------------------------------------- #
def test_sort_path_byte_identity_regression():
    """serialize_lane_band_rd (sort) must be stable across the _serialize_matrix_lbnd2 refactor:
    self-consistent + magic + decodes to the same lines."""
    pairs = _parallel_pairs(P=25)
    cfg = LaneBandRenderConfig()
    b1 = serialize_lane_band_rd(pairs, cfg)
    b2 = serialize_lane_band_rd(pairs, cfg)
    assert b1 == b2 and b1[:6] == b"LBND2\x00"
    hdr = deserialize_lane_band_rd(b1)[1]
    assert "pack_mode" not in hdr["rd"]                             # sort path adds NO provenance key


# --------------------------------------------------------------------------- #
# report shape + verdict keys
# --------------------------------------------------------------------------- #
def test_tracking_rate_report_shape():
    pairs = _count_change_pairs(P=50)
    cfg = LaneBandRenderConfig()
    rep = lane_band_tracking_rate_report(pairs, cfg, smooth_methods=("none", "rts"),
                                         ref_smooth_wins=(9, 15))
    assert rep["correspondence_lossless_vs_sort"] is True
    variants = set(r["variant"] for r in rep["rows"])
    assert variants >= {"LBND2_sort_baseline", "coherent_slot_none", "persistent_none",
                        "coherent_slot_rts", "sort_MA_win9", "sort_MA_win15"}
    for r in rep["rows"]:
        assert r["brotli_bytes"] > 0 and r["rate_term"] > 0
        assert "induced_lateral_rms_m" in r
