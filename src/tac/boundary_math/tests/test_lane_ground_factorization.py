"""Tests for the v8 lane ground-frame anisotropic factorization primitives."""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.lane_ground_factorization import (
    COEFF_DIM,
    FORWARD_KNOTS_M,
    GroundLaneObs,
    LaneTrack,
    encode_shift_stream,
    encode_tracks_occupancy,
    encode_tracks_spd,
    estimate_global_shifts,
    evaluate_tracks_raster,
    fit_frame_ground_lanes,
    lane_band_metrics,
    lane_line_from_vec,
    obs_from_lane_line,
    per_frame_quantized_ground,
    pooled_hilbert_distance,
    robust_dim_scales,
    shifts_to_bins,
    track_ground_lanes,
)
from tac.boundary_math.lane_sdf_component import LaneLine, rasterize_lane_band

H, W = 384, 512


def _mk_line(lat0: float, curv: float = 0.0, hw: float = 3.0) -> LaneLine:
    """A ground-frame lane line: lateral = lat0 + curv*forward^2 (deg-2 poly)."""
    return LaneLine(
        centerline_coeffs=np.array([curv, 0.0, lat0], np.float64),
        halfwidth_coeffs=np.array([0.0, hw], np.float64),
        forward_range=(6.0, 45.0),
        n_pixels=500,
    )


def _synthetic_frame(lines: list[LaneLine]) -> np.ndarray:
    """Rasterize lines into a 5-class label map (lane=1 on road=0)."""
    band = rasterize_lane_band(lines, h=H, w=W, dash_gate=False)
    lab = np.zeros((H, W), np.uint8)
    lab[band] = 1
    return lab


def test_knot_parametrization_roundtrip_exact():
    ln = _mk_line(-1.8, curv=2e-3)
    obs = obs_from_lane_line(0, ln)
    assert obs.vec.shape == (COEFF_DIM,)
    ln2 = lane_line_from_vec(obs.vec)
    fwd = np.linspace(6.0, 45.0, 50)
    np.testing.assert_allclose(
        ln2.lateral_of_forward(fwd), ln.lateral_of_forward(fwd), atol=1e-9
    )
    rows = np.linspace(200, 380, 20)
    np.testing.assert_allclose(
        ln2.halfwidth_of_v(rows), ln.halfwidth_of_v(rows), atol=1e-9
    )
    assert ln2.forward_range == pytest.approx(ln.forward_range)


def test_fit_frame_ground_lanes_recovers_synthetic():
    truth = [_mk_line(-1.8), _mk_line(1.8)]
    lab = _synthetic_frame(truth)
    obs = fit_frame_ground_lanes(lab, 0, fit_dash=False)
    assert len(obs) == 2
    lats = sorted(o.lat_ref for o in obs)
    assert lats[0] == pytest.approx(-1.8, abs=0.35)
    assert lats[1] == pytest.approx(1.8, abs=0.35)


def test_full_pipeline_roundtrip_high_recall_small_bytes():
    """Synthetic multi-frame sequence: fit -> track -> SPD encode -> raster.

    The factorized code must reproduce the lane band at high recall for a small
    byte cost (the whole point of the construction)."""
    n_frames = 24
    per_frame: list[list[GroundLaneObs]] = []
    labels = np.zeros((n_frames, H, W), np.uint8)
    for t in range(n_frames):
        drift = 0.02 * t  # slow shared ego drift
        lines = [_mk_line(-1.8 + drift), _mk_line(1.8 + drift)]
        labels[t] = _synthetic_frame(lines)
        per_frame.append(fit_frame_ground_lanes(labels[t], t, fit_dash=False))
    tracks = track_ground_lanes(per_frame)
    assert 1 <= len(tracks) <= 4
    scales = robust_dim_scales(tracks)
    enc = encode_tracks_spd(tracks, water_level=1e-4, scales=scales)
    assert enc["total_bytes"] > 0
    # far under a 97 B/frame image-space cost on this easy sequence
    assert enc["total_bytes"] / n_frames < 97
    m = evaluate_tracks_raster(tracks, labels, np.arange(n_frames), occ_gate=False)
    assert m["recall"] > 0.9
    assert m["iou"] > 0.6


def test_tracker_splits_distant_lines_and_keeps_continuity():
    per_frame = []
    for t in range(10):
        per_frame.append(
            [
                GroundLaneObs(t, np.full(COEFF_DIM, -2.0), lat_ref=-2.0, n_pixels=100),
                GroundLaneObs(t, np.full(COEFF_DIM, 2.0), lat_ref=2.0, n_pixels=100),
            ]
        )
    tracks = track_ground_lanes(per_frame)
    assert len(tracks) == 2
    assert all(t.n_obs == 10 for t in tracks)


def test_world_static_dash_transport_recovers_shift_and_beats_solid_precision():
    """Dashes are static world paint: a dashed lane observed under known ego
    travel must yield (a) a recovered per-frame travel close to truth, (b) a
    LOSSLESS occupancy round-trip, and (c) occupancy-gated reconstruction with
    better precision than the solid band at no recall collapse."""
    n_frames, travel = 20, 1.5  # m per frame
    per, duty, w0 = 9.0, 0.5, 3.0
    labels = np.zeros((n_frames, H, W), np.uint8)
    per_frame = []
    for t in range(n_frames):
        s_t = travel * t
        ln = LaneLine(
            centerline_coeffs=np.array([0.0, 0.0, -1.8]),
            halfwidth_coeffs=np.array([0.0, 3.0]),
            dash_period_m=per,
            dash_phase_m=w0 - s_t,  # static world pattern seen from a moving ego
            dash_duty=duty,
            forward_range=(4.0, 50.0),
        )
        solid = LaneLine(  # a second, solid line to anchor tracking
            centerline_coeffs=np.array([0.0, 0.0, 1.8]),
            halfwidth_coeffs=np.array([0.0, 3.0]),
            forward_range=(4.0, 50.0),
        )
        band = rasterize_lane_band([ln, solid], h=H, w=W, dash_gate=True)
        labels[t][band] = 1
        per_frame.append(fit_frame_ground_lanes(labels[t], t, fit_dash=False))
    tracks = track_ground_lanes(per_frame)
    assert len(tracks) >= 2
    shifts = estimate_global_shifts(tracks, n_frames)
    mean_travel = shifts[-1] / (n_frames - 1)
    assert mean_travel == pytest.approx(travel, abs=0.5)
    scales = robust_dim_scales(tracks)
    encode_tracks_spd(tracks, water_level=1e-4, scales=scales)
    sbins = shifts_to_bins(shifts)
    occ_enc = encode_tracks_occupancy(tracks, sbins)
    assert occ_enc["occ_bytes"] > 0
    for tr in tracks:  # LOSSLESS: decoded occupancy == original, bit for bit
        np.testing.assert_array_equal(tr.occ_decoded, tr.occ)
    m_solid = evaluate_tracks_raster(tracks, labels, np.arange(n_frames), occ_gate=False)
    m_occ = evaluate_tracks_raster(tracks, labels, np.arange(n_frames), occ_gate=True)
    assert m_occ["precision"] > m_solid["precision"] + 0.05
    assert m_occ["recall"] > 0.75
    # the world-aligned XOR must compress well below the raw bitmap
    assert occ_enc["occ_bytes"] < occ_enc["occ_bits_raw"] // 8


def test_occupancy_codec_lossless_even_with_wrong_shifts():
    """A wrong S(t) must only cost bytes, never fidelity (XOR stays exact)."""
    rng = np.random.default_rng(3)
    n = 12
    occ = rng.random((n, 105)) < 0.4
    tr = LaneTrack(
        frames=np.arange(n),
        coeffs=np.zeros((n, COEFF_DIM)),
        dash_obs=np.zeros((n, 3)),
        lat_ref=np.zeros(n),
        occ=occ,
    )
    bad_shifts = rng.integers(0, 30, size=n).astype(np.int64)
    encode_tracks_occupancy([tr], bad_shifts)
    np.testing.assert_array_equal(tr.occ_decoded, tr.occ)


def test_shift_stream_bytes_positive_and_small():
    sbins = shifts_to_bins(np.cumsum(np.full(600, 1.5)))
    b = encode_shift_stream(sbins)
    assert 0 < b < 600


def test_spd_anisotropic_control_beats_isotropic_at_matched_theta():
    """Synthetic control: an anisotropic coefficient matrix (smooth temporal
    trajectories, spread spectrum) must code to fewer bytes than an isotropic
    one at the same water level, and its d_H must be much larger."""
    rng = np.random.default_rng(0)
    n = 400
    t = np.linspace(0, 1, n)
    smooth = np.stack(
        [np.sin(2 * np.pi * (k + 1) * t) * (2.0 ** -k) for k in range(COEFF_DIM)],
        axis=1,
    )
    aniso = LaneTrack(
        frames=np.arange(n),
        coeffs=smooth + 1e-4 * rng.standard_normal((n, COEFF_DIM)),
        dash_obs=np.zeros((n, 3)),
        lat_ref=np.zeros(n),
    )
    iso = LaneTrack(
        frames=np.arange(n),
        coeffs=rng.standard_normal((n, COEFF_DIM)),
        dash_obs=np.zeros((n, 3)),
        lat_ref=np.zeros(n),
    )
    ones = np.ones(COEFF_DIM)
    d_h_aniso = pooled_hilbert_distance([aniso], ones)
    d_h_iso = pooled_hilbert_distance([iso], ones)
    assert d_h_aniso > d_h_iso + 2.0
    enc_a = encode_tracks_spd([aniso], water_level=1e-4, scales=ones)
    enc_i = encode_tracks_spd([iso], water_level=1e-4, scales=ones)
    assert enc_a["total_bytes"] < enc_i["total_bytes"]
    # round-trip fidelity honest on the anisotropic case
    err = float(np.abs(aniso.decoded - aniso.coeffs).max())
    assert err < 0.15


def test_per_frame_quantized_ground_roundtrip():
    obs = GroundLaneObs(0, np.linspace(-2, 40, COEFF_DIM), lat_ref=-2.0, n_pixels=50)
    obs2 = GroundLaneObs(1, np.linspace(-1.9, 40.5, COEFF_DIM), lat_ref=-1.9, n_pixels=50)
    quant, bpf = per_frame_quantized_ground([[obs], [obs2]], bits=12)
    assert bpf > 0
    err = np.abs(quant[0][0].vec - obs.vec).max()
    span = (obs2.vec - obs.vec).max() + 1e-9
    assert err <= abs(span)  # within one quantization span
    assert np.all(np.isfinite(quant[1][0].vec))


def test_lane_band_metrics_definition():
    band = np.zeros((4, 4), bool)
    lane = np.zeros((4, 4), bool)
    band[0, :2] = True
    lane[0, 1:3] = True
    m = lane_band_metrics(band, lane)
    assert m["recall"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["iou"] == pytest.approx(1 / 3)


def test_knots_are_generic_constants():
    # guard: the knots are algorithm constants (rule 118 free) — fixed, sorted
    assert FORWARD_KNOTS_M.tolist() == sorted(FORWARD_KNOTS_M.tolist())
    assert len(FORWARD_KNOTS_M) == 4
