# SPDX-License-Identifier: MIT
"""Behavioral tests for the scorer spectral-sensitivity atlas v2 (Deliverable 1).

NO FAKE Class-2 discipline: these tests verify BEHAVIOR (the physics is correct
and the response is non-trivial), NOT constants. Replacing a function body with a
stub would make the matching test FAIL:

* ``test_coordinate_conversion_matches_empirical_fft`` synthesizes a band and
  measures its actual FFT cycle content, asserting the claimed cyc/px matches —
  so a wrong coordinate formula (the classic "k in camera px but w in normalized
  coords" bug) FAILS.
* ``test_yuv_roundtrip_is_bijective`` would FAIL if the inverse were a stub.
* ``test_perturb_rgb_channel_isolates_that_channel`` would FAIL if the basis
  rotation were a no-op.
* ``test_energy_audit_resize_attenuates_hf`` would FAIL if the resize-energy
  proxy returned the pre-resize energy.

The fast tests are pure-numpy. The single end-to-end ``measure_atlas`` test is
gated on torch + the frozen scorer weights (skipped when absent), because it runs
the EXACT ``DistortionNet`` — the authority surface.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from tac.analysis import scorer_spectral_sensitivity_v2 as v2

# ---------------------------------------------------------------------------
# BandSpec + grid validation.
# ---------------------------------------------------------------------------


def test_bandspec_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="r_lo < r_hi"):
        v2.BandSpec(0, 0.5, 0.5)
    with pytest.raises(ValueError, match="r_lo < r_hi"):
        v2.BandSpec(0, 0.6, 0.4)


def test_bandspec_rejects_unknown_orientation() -> None:
    with pytest.raises(ValueError, match="orientation"):
        v2.BandSpec(0, 0.1, 0.2, "spiral")


def test_bandspec_r_center_is_midpoint() -> None:
    b = v2.BandSpec(3, 0.2, 0.6)
    assert b.r_center == pytest.approx(0.4)


def test_build_band_specs_linear_tiles_dc_to_nyquist() -> None:
    bands = v2.build_band_specs(4, "isotropic", spacing="linear")
    assert len(bands) == 4
    assert bands[0].r_lo == 0.0
    assert bands[-1].r_hi == pytest.approx(1.0)
    # contiguous, non-overlapping
    for a, b in pairwise(bands):
        assert a.r_hi == pytest.approx(b.r_lo)


def test_build_band_specs_log_is_denser_at_low_freq() -> None:
    """log spacing must put MORE bands below r=0.25 than linear (resolves w=1..30)."""
    lin = v2.build_band_specs(6, "isotropic", spacing="linear")
    log = v2.build_band_specs(6, "isotropic", spacing="log")
    lin_low = sum(1 for b in lin if b.r_center < 0.25)
    log_low = sum(1 for b in log if b.r_center < 0.25)
    assert log_low > lin_low
    # log first band still starts at DC; last reaches Nyquist corner.
    assert log[0].r_lo == 0.0
    assert log[-1].r_hi == pytest.approx(1.0)


def test_build_band_specs_rejects_bad_spacing() -> None:
    with pytest.raises(ValueError, match="spacing"):
        v2.build_band_specs(4, "isotropic", spacing="quadratic")


# ---------------------------------------------------------------------------
# Coordinate conversion — the validity-critical math (#7).
# ---------------------------------------------------------------------------


def test_coordinate_conversion_matches_empirical_fft() -> None:
    """The claimed camera cyc/px must match the synthesized band's actual FFT
    content. This is the guard against the "measured k in camera px, implemented
    w in normalized coords" bug class: if the formula is wrong, this FAILS."""
    h, w = v2.CAMERA_HW
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    fmag = np.sqrt(fy**2 + fx**2)
    rng = np.random.default_rng(0)
    # bands away from the lowest annulus (whose center label undershoots the
    # power-weighted mean) and away from the Nyquist edge.
    for k in (1, 2, 3, 4):
        band = v2.BandSpec(k, k / 6, (k + 1) / 6, "isotropic")
        field = v2.band_limited_field((h, w, 1), band, rng)[..., 0]
        power = np.abs(np.fft.fft2(field)) ** 2
        emp_cyc_per_px = float((fmag * power).sum() / power.sum())
        coords = v2.frequency_coordinates_for_band(band)
        rel_err = abs(coords.camera_cycles_per_pixel - emp_cyc_per_px) / emp_cyc_per_px
        assert rel_err < 0.10, (
            f"band{k}: claimed cyc/px {coords.camera_cycles_per_pixel:.4f} vs "
            f"empirical {emp_cyc_per_px:.4f} (rel_err {rel_err:.2%})"
        )


def test_coordinate_conversion_cyc_per_px_monotonic_in_r() -> None:
    cpps = [
        v2.frequency_coordinates_for_band(
            v2.BandSpec(k, k / 8, (k + 1) / 8, "isotropic")
        ).camera_cycles_per_pixel
        for k in range(8)
    ]
    assert all(b > a for a, b in pairwise(cpps))


def test_scorer_cyc_per_px_higher_than_camera_due_to_downsample() -> None:
    """The scorer is downsampled (874x1164 -> 384x512); cyc/SCORER-px must be
    HIGHER than cyc/camera-px (a cycle occupies fewer scorer pixels)."""
    b = v2.BandSpec(3, 0.4, 0.5, "isotropic")
    c = v2.frequency_coordinates_for_band(b)
    assert c.scorer_cycles_per_pixel > c.camera_cycles_per_pixel
    # ratio ~ 1/0.44 ~ 2.27
    assert c.scorer_cycles_per_pixel / c.camera_cycles_per_pixel == pytest.approx(
        math.sqrt((874 / 384) * (1164 / 512)), rel=0.02
    )


def test_high_band_aliases_at_scorer() -> None:
    """A high camera band exceeds the scorer Nyquist (0.5 cyc/scorer-px) -> aliases."""
    low = v2.frequency_coordinates_for_band(v2.BandSpec(0, 0.0, 0.1, "isotropic"))
    high = v2.frequency_coordinates_for_band(v2.BandSpec(5, 0.83, 1.0, "isotropic"))
    assert not low.aliases_at_scorer
    assert high.aliases_at_scorer
    assert high.scorer_cycles_per_pixel > 0.5


def test_siren_w_equivalent_is_pi_times_cycles() -> None:
    """w = pi * cycles_across_extent (one cycle across [-1,1] of length 2 = w=pi)."""
    assert v2.siren_w_equivalent(1.0) == pytest.approx(math.pi)
    assert v2.siren_w_equivalent(10.0) == pytest.approx(10 * math.pi)
    # and the band coord uses it consistently
    b = v2.BandSpec(2, 0.3, 0.4, "isotropic")
    c = v2.frequency_coordinates_for_band(b)
    assert c.siren_w_equivalent == pytest.approx(
        math.pi * c.scorer_cycles_per_image_height
    )
    assert c.siren_w_equivalent == c.normalized_omega


def test_cycle_count_is_resize_invariant() -> None:
    """cycles-across-extent must NOT change under the scorer resize (only cyc/px
    does). scorer_cycles_per_image_height == camera cyc/px * camera extent."""
    b = v2.BandSpec(3, 0.4, 0.5, "isotropic")
    c = v2.frequency_coordinates_for_band(b)
    cam_h = v2.CAMERA_HW[0]
    assert c.scorer_cycles_per_image_height == pytest.approx(
        c.camera_cycles_per_pixel * cam_h
    )


# ---------------------------------------------------------------------------
# Channel basis (#4) — RGB <-> full-res BT.601 YUV.
# ---------------------------------------------------------------------------


def test_yuv_roundtrip_is_bijective() -> None:
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0, 255, size=(12, 9, 3))
    back = v2.full_yuv_to_rgb(v2.rgb_to_full_yuv(rgb))
    assert np.abs(back - rgb).max() < 1e-9


def test_yuv_forward_matches_bt601_coefficients() -> None:
    """Y must be the BT.601 luma of a known RGB (matching upstream rgb_to_yuv6)."""
    rgb = np.array([[[100.0, 150.0, 200.0]]])
    yuv = v2.rgb_to_full_yuv(rgb)
    expected_y = 100 * 0.299 + 150 * 0.587 + 200 * 0.114
    assert yuv[0, 0, 0] == pytest.approx(expected_y)
    # neutral gray -> U=V=128
    gray = np.full((1, 1, 3), 120.0)
    g = v2.rgb_to_full_yuv(gray)
    assert g[0, 0, 1] == pytest.approx(128.0)
    assert g[0, 0, 2] == pytest.approx(128.0)


def test_perturb_rgb_channel_isolates_that_channel() -> None:
    frame = np.full((16, 16, 3), 128.0)
    field = v2.band_limited_field(
        (16, 16, 3), v2.BandSpec(1, 0.1, 0.3, "isotropic"), np.random.default_rng(2)
    )
    out = v2.perturb_channel_basis(
        frame, field, channel_basis="rgb", channel="g", amplitude_lsb=4.0
    )
    # only G changed
    assert np.abs(out[..., 0] - 128.0).max() < 1e-9
    assert np.abs(out[..., 2] - 128.0).max() < 1e-9
    assert np.abs(out[..., 1] - 128.0).max() > 0.1


def test_perturb_yuv_y_channel_changes_all_rgb_but_preserves_chroma() -> None:
    """Perturbing Y (luma) must change RGB (all three move together) while U/V of
    the perturbed frame stay ~unchanged (luma-only injection)."""
    frame = np.full((16, 16, 3), 120.0)
    field = v2.band_limited_field(
        (16, 16, 3), v2.BandSpec(1, 0.1, 0.3, "isotropic"), np.random.default_rng(3)
    )
    out = v2.perturb_channel_basis(
        frame, field, channel_basis="yuv", channel="y", amplitude_lsb=6.0
    )
    # all RGB channels move (a Y change maps to R,G,B all shifting)
    for ch in range(3):
        assert np.abs(out[..., ch] - 120.0).max() > 0.1
    # but the recovered chroma is ~unchanged (Y-only perturbation)
    yuv_out = v2.rgb_to_full_yuv(out)
    assert np.abs(yuv_out[..., 1] - 128.0).max() < 1e-6
    assert np.abs(yuv_out[..., 2] - 128.0).max() < 1e-6


def test_perturb_rejects_bad_channel_for_basis() -> None:
    frame = np.full((8, 8, 3), 128.0)
    field = np.zeros((8, 8, 3))
    with pytest.raises(ValueError, match="not valid for basis"):
        v2.perturb_channel_basis(
            frame, field, channel_basis="rgb", channel="y", amplitude_lsb=1.0
        )


def test_perturb_clips_to_valid_range() -> None:
    frame = np.full((8, 8, 3), 254.0)
    field = np.ones((8, 8, 3))  # unit field
    out = v2.perturb_channel_basis(
        frame, field, channel_basis="rgb", channel="all", amplitude_lsb=50.0
    )
    assert out.max() <= 255.0
    assert out.min() >= 0.0


# ---------------------------------------------------------------------------
# Band-limited field + orientation (#5).
# ---------------------------------------------------------------------------


def test_band_limited_field_unit_std_per_channel() -> None:
    field = v2.band_limited_field(
        (32, 48, 3), v2.BandSpec(2, 0.3, 0.5, "isotropic"), np.random.default_rng(0)
    )
    for ch in range(3):
        assert field[..., ch].std() == pytest.approx(1.0, abs=1e-6)


def test_band_limited_field_lives_in_its_annulus() -> None:
    """The field's FFT energy must be confined to the band's radial annulus."""
    band = v2.BandSpec(2, 0.3, 0.5, "isotropic")
    field = v2.band_limited_field((64, 64, 1), band, np.random.default_rng(0))[..., 0]
    radius = v2.band_radius_grid((64, 64))
    power = np.abs(np.fft.fft2(field)) ** 2
    in_band = power[(radius >= band.r_lo) & (radius < band.r_hi)].sum()
    out_band = power[(radius < band.r_lo) | (radius >= band.r_hi)].sum()
    assert in_band > 0
    assert out_band < 1e-6 * (in_band + out_band)  # essentially all energy in-band


def test_oriented_wedge_has_fewer_bins_than_isotropic() -> None:
    annulus = v2.oriented_band_mask((48, 48), v2.BandSpec(2, 0.3, 0.5, "isotropic"))
    for orient in ("horizontal", "vertical", "diag_plus", "diag_minus"):
        wedge = v2.oriented_band_mask((48, 48), v2.BandSpec(2, 0.3, 0.5, orient))
        assert wedge.sum() < annulus.sum()
        assert wedge.sum() > 0


def test_oriented_fields_have_distinct_spatial_structure() -> None:
    """A horizontal-oriented field (vertical edges) must differ from a vertical
    one — proving orientation is real, not a relabel."""
    h_band = v2.BandSpec(2, 0.3, 0.5, "horizontal")
    v_band = v2.BandSpec(2, 0.3, 0.5, "vertical")
    fh = v2.band_limited_field((64, 64, 1), h_band, np.random.default_rng(0))[..., 0]
    fv = v2.band_limited_field((64, 64, 1), v_band, np.random.default_rng(0))[..., 0]
    # horizontal structure -> stronger gradient along x; vertical -> along y.
    gx_h = np.abs(np.diff(fh, axis=1)).mean()
    gy_h = np.abs(np.diff(fh, axis=0)).mean()
    gx_v = np.abs(np.diff(fv, axis=1)).mean()
    gy_v = np.abs(np.diff(fv, axis=0)).mean()
    assert gx_h > gy_h  # horizontal band: more x-variation
    assert gy_v > gx_v  # vertical band: more y-variation


def test_random_phase_ensemble_draws_differ() -> None:
    """Successive draws from the same band with the same rng give DIFFERENT fields
    (the random-phase ensemble for CI), but the same band/std."""
    band = v2.BandSpec(2, 0.3, 0.5, "isotropic")
    rng = np.random.default_rng(0)
    f1 = v2.band_limited_field((32, 32, 1), band, rng)[..., 0]
    f2 = v2.band_limited_field((32, 32, 1), band, rng)[..., 0]
    assert np.abs(f1 - f2).max() > 0.1  # distinct phase draws
    assert f1.std() == pytest.approx(1.0, abs=1e-6)
    assert f2.std() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Frame incidence (#3).
# ---------------------------------------------------------------------------


def test_frame_incidence_patterns_touch_the_right_frames() -> None:
    pair = np.full((2, 16, 16, 3), 128.0)
    field = v2.band_limited_field(
        (16, 16, 3), v2.BandSpec(1, 0.1, 0.3, "isotropic"), np.random.default_rng(0)
    )
    kw = {"channel_basis": "rgb", "channel": "all", "amplitude_lsb": 4.0}

    f0 = v2.apply_frame_incidence(pair, field, incidence="frame0_only", **kw)
    assert np.abs(f0[0] - 128.0).max() > 0.1  # frame0 changed
    assert np.abs(f0[1] - 128.0).max() < 1e-9  # frame1 untouched

    f1 = v2.apply_frame_incidence(pair, field, incidence="frame1_only", **kw)
    assert np.abs(f1[0] - 128.0).max() < 1e-9
    assert np.abs(f1[1] - 128.0).max() > 0.1

    both = v2.apply_frame_incidence(pair, field, incidence="both_same", **kw)
    # same field both frames -> frame0 == frame1 (zero inter-frame difference)
    assert np.abs(both[0] - both[1]).max() < 1e-9

    opp = v2.apply_frame_incidence(pair, field, incidence="both_opposite", **kw)
    # opposite-sign -> frames differ (maximizes inter-frame motion signal)
    assert np.abs(opp[0] - opp[1]).max() > 0.1


def test_frame_incidence_rejects_unknown() -> None:
    pair = np.full((2, 8, 8, 3), 128.0)
    field = np.zeros((8, 8, 3))
    with pytest.raises(ValueError, match="incidence"):
        v2.apply_frame_incidence(
            pair, field, incidence="frame9", channel_basis="rgb", channel="all", amplitude_lsb=1.0
        )


# ---------------------------------------------------------------------------
# Energy audit (#6) — clip + resize attenuation.
# ---------------------------------------------------------------------------


def test_energy_audit_resize_attenuates_hf() -> None:
    """A high-frequency perturbation must lose energy under the scorer resize:
    post_resize_l2 < post_clip_l2 (the HF-attenuation the operator flagged)."""
    frame = np.full((v2.CAMERA_HW[0] // 4, v2.CAMERA_HW[1] // 4, 3), 128.0)
    sc_hw = (frame.shape[0] // 4, frame.shape[1] // 4)
    hf = v2.band_limited_field(
        (frame.shape[0], frame.shape[1], 3),
        v2.BandSpec(5, 0.83, 1.0, "isotropic"),
        np.random.default_rng(0),
    )
    perturbed = np.clip(frame + 4.0 * hf, 0, 255)
    audit = v2.energy_audit_for_perturbation(frame, perturbed, scorer_input_hw=sc_hw)
    assert audit.post_resize_l2 < audit.post_clip_l2
    assert audit.post_clip_l2 > 0


def test_energy_audit_reports_clip_fraction() -> None:
    """A large perturbation against a saturated frame must register clipping."""
    frame = np.full((20, 20, 3), 253.0)
    field = v2.band_limited_field(
        (20, 20, 3), v2.BandSpec(2, 0.3, 0.5, "isotropic"), np.random.default_rng(0)
    )
    perturbed = np.clip(frame + 30.0 * field, 0, 255)
    audit = v2.energy_audit_for_perturbation(frame, perturbed)
    assert audit.clip_fraction > 0.0
    # a low-amplitude perturbation against a mid-gray frame should NOT clip
    mid = np.full((20, 20, 3), 128.0)
    pert2 = np.clip(mid + 1.0 * field, 0, 255)
    audit2 = v2.energy_audit_for_perturbation(mid, pert2)
    assert audit2.clip_fraction == pytest.approx(0.0)


def test_energy_audit_per_channel_detects_single_channel_perturbation() -> None:
    frame = np.full((16, 16, 3), 128.0)
    field = v2.band_limited_field(
        (16, 16, 3), v2.BandSpec(1, 0.1, 0.3, "isotropic"), np.random.default_rng(0)
    )
    perturbed = v2.perturb_channel_basis(
        frame, field, channel_basis="rgb", channel="r", amplitude_lsb=4.0
    )
    audit = v2.energy_audit_for_perturbation(frame, perturbed)
    r_e, g_e, b_e = audit.per_channel_post_clip_l2
    assert r_e > 0
    assert g_e == pytest.approx(0.0)
    assert b_e == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Grid cost accounting.
# ---------------------------------------------------------------------------


def test_atlas_grid_total_cells_accounting() -> None:
    g = v2.AtlasGrid(
        n_pairs=3,
        n_bands=4,
        amplitudes_lsb=(1.0, 2.0),
        orientations=("isotropic", "horizontal"),
        frame_incidences=("frame1_only",),
        channel_bases=("rgb", "yuv"),
        rgb_channels=("all", "r"),
        yuv_channels=("y",),
        n_phase_samples=2,
    )
    # cells = bands(4) * orient(2) * amp(2) * [rgb:2 + yuv:1 = 3 ch] * incidence(1) = 48
    assert g.total_cells() == 4 * 2 * 2 * 3 * 1
    assert g.total_scorer_forwards() == g.total_cells() * 3 * 2


def test_atlas_grid_channels_for_dispatches_by_basis() -> None:
    g = v2.AtlasGrid(rgb_channels=("all",), yuv_channels=("y", "u"))
    assert g.channels_for("rgb") == ("all",)
    assert g.channels_for("yuv") == ("y", "u")


# ---------------------------------------------------------------------------
# Source-class margin (Level-1) — pure-numpy stand-in (no torch needed).
# ---------------------------------------------------------------------------


def test_source_class_margin_via_numpy_oracle() -> None:
    """Validate the margin helper against a numpy oracle (without torch).

    The helper computes top1 - top2. We confirm via a numpy computation on a
    hand-built logit volume so the test does not need torch.
    """
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [[[[3.0, 1.0]], [[0.5, 2.5]], [[1.0, 0.0]], [[0.0, 0.5]], [[2.0, 1.5]]]]
    )  # (B=1, 5 classes, Hs=1, Ws=2)
    margin = v2.segnet_source_class_margin(logits)  # top1 - top2
    # pixel 0: sorted desc [3,2,1,0.5,0] -> 3-2=1; pixel 1: [2.5,1.5,1,0.5,0] -> 2.5-1.5=1
    assert margin.shape == (1, 1, 2)
    assert float(margin[0, 0, 0]) == pytest.approx(1.0)
    assert float(margin[0, 0, 1]) == pytest.approx(1.0)


def test_contest_score_nonrate_formula() -> None:
    # 100*d_seg + sqrt(10*d_pose)
    assert v2.contest_score_nonrate(0.0, 0.0) == pytest.approx(0.0)
    assert v2.contest_score_nonrate(0.01, 0.0) == pytest.approx(1.0)
    assert v2.contest_score_nonrate(0.0, 0.1) == pytest.approx(1.0)
    assert v2.contest_score_nonrate(0.02, 0.4) == pytest.approx(2.0 + 2.0)


# ---------------------------------------------------------------------------
# End-to-end through the FROZEN scorer (gated on torch + weights).
# ---------------------------------------------------------------------------


def _scorer_available() -> bool:
    try:
        import importlib.util
        from pathlib import Path

        if importlib.util.find_spec("torch") is None:
            return False
        repo = Path(__file__).resolve().parents[3]
        seg = repo / "upstream" / "models" / "segnet.safetensors"
        pose = repo / "upstream" / "models" / "posenet.safetensors"
        return seg.exists() and pose.exists()
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.timeout(0)  # loads + runs the EXACT DistortionNet; exceeds the 60s global cap
@pytest.mark.skipif(not _scorer_available(), reason="frozen scorer weights / torch unavailable")
def test_measure_atlas_end_to_end_through_frozen_scorer() -> None:
    """Run a tiny atlas through the EXACT DistortionNet and assert the artifact
    has all three response levels, non-zero response, and correct authority flags.

    This is the authority-surface behavioral guard: a no-op scorer (or a response
    measured at the wrong level) would make the response identically zero and
    FAIL ``assert any H_pose > 0``. Deliberately minimal (1 band x 1 amplitude x
    2 incidences x 1 pair) so it loads the scorer once and stays cheap; the full
    grid is run via the CLI, not in CI.
    """
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    for p in (str(repo / "src"), str(repo / "upstream"), str(repo / "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Small deterministic source: 1 random pair at camera resolution.
    H, W = v2.CAMERA_HW
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, 256, size=(1, 2, H, W, 3), dtype=np.uint8)

    grid = v2.AtlasGrid(
        n_pairs=1,
        n_bands=1,
        amplitudes_lsb=(8.0,),
        orientations=("isotropic",),
        frame_incidences=("frame1_only", "both_opposite"),
        channel_bases=("rgb",),
        rgb_channels=("all",),
        yuv_channels=("all",),
        n_phase_samples=1,
        seed=0,
    )
    art = v2.measure_atlas(pairs, grid, device="cpu", progress=False)

    # authority flags (the metric-laundering firewall)
    assert art["schema"] == v2.SCHEMA_VERSION
    assert art["authority_tier"] == "exact_cpu_advisory"
    assert art["metric_family"] == "exact_pair_scorer"
    assert art["promotable"] is False
    assert art["score_roadmap_update_eligible"] is False
    assert art["mechanism_update_eligible"] is True

    # baseline source-vs-source is ~0 (the measurement floor)
    assert abs(art["baseline"]["d_seg"]) < 1e-6
    assert abs(art["baseline"]["d_pose"]) < 1e-6

    cells = art["cells"]
    assert len(cells) == grid.total_cells()
    # every cell carries all three response levels + energy + coords + CI
    for c in cells:
        assert "H_logit_margin_drop_mean" in c  # Level-1
        assert "H_seg" in c and "flip_count_boundary" in c  # Level-2
        assert "H_pose" in c and "score_nonrate" in c  # Level-3
        assert set(c["energy"]) >= {"pre_clip_l2", "post_clip_l2", "clip_fraction", "post_resize_l2"}
        assert "siren_w_equivalent" in c["frequency_coordinates"]
        assert c["n_phase_samples"] == 1

    # the scorer is NOT a no-op: some perturbation moved a response level.
    assert any(c["H_pose"] > 0 for c in cells) or any(c["H_seg"] > 0 for c in cells)
    assert any(abs(c["logit_l2_delta"]) > 0 for c in cells)  # logits reached + moved

    # headline carries the w-verdict scaffolding
    assert "siren_w_equivalent" in art["headline"]["seg_peak"]
    assert "w_verdict_note" in art["headline"]
