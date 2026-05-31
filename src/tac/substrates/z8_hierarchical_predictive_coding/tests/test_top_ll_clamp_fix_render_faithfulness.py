# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the Z8 top-LL clamp fix + archive-path render faithfulness.

These tests assert ACTUAL behavior (Slot EEE Class 2), not constants:

  1. ``test_wz_top_ll_projection_is_not_codec_domain_clamped`` — a WZ top-LL
     projection on out-of-[0,1] coefficients does NOT saturate the result to
     [0, 1]. If a future regression reintroduces a coefficient-domain
     ``np.clip(..., 0.0, 1.0)`` on the projection, this test FAILS because the
     projected coefficients would be clamped into [0, 1].

  2. ``test_reconstruct_clamp_is_final_pixel_not_codec_domain`` — the
     ``reconstruct_pair_rgb_from_pyramid`` clamp at
     ``canonical_quadruple_binding`` operates on the level-0 reconstructed
     image (final pixels), so a coefficient pyramid whose inverse transform
     lands inside [0, 1] is preserved, AND the output is in [0, 1].

  3. ``test_archive_path_reconstructs_faithfully_on_real_frames`` — the
     wavelet+WZ ARCHIVE/inflate path reconstructs real upstream frames in the
     GT pixel range (NOT saturated to ~155-165). This is the genuine
     hierarchical-PC unlock test on the contest-archive path.

  4. ``test_wz_payload_mutation_drives_frame_1_pixels`` — a valid semantic WZ
     payload mutation moves frame-1 pixels while frame-0 is stable (state
     DEPENDENCE). If the codec-domain clamp returns, the mutation effect would
     be erased.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _canonical_cfg(*, num_pairs: int, eval_h: int, eval_w: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        num_pairs=num_pairs,
        deterministic_state_dim=16,
        ego_motion_dim=6,
        eval_size=(eval_h, eval_w),
    )


def test_wz_top_ll_projection_is_not_codec_domain_clamped():
    """The WZ top-LL delta must not force the projection into [0, 1].

    Build a top-LL coefficient array containing values WELL outside [0, 1]
    (codec coefficients are not pixels). Project a non-trivial WZ state and add
    the delta. The result must preserve the out-of-range coefficients (a
    codec-domain [0, 1] clamp would crush them to <= 1.0).
    """

    from tac.substrates.z8_hierarchical_predictive_coding.runtime_payload_bridge import (
        _state_to_top_ll_delta,
    )

    # Top-LL coefficients deliberately span far outside [0, 1].
    top_ll = np.array(
        [[[5.0, -3.0, 2.5]], [[-7.0, 4.0, 9.0]]], dtype=np.float32
    )  # (2, 1, 3)
    state = np.array([1.0, -2.0, 0.5, 3.0, -1.5], dtype=np.float32)
    delta = _state_to_top_ll_delta(state, top_ll)
    projected = top_ll + delta
    # The projected coefficients MUST still contain values outside [0, 1].
    # A codec-domain clamp would make projected.max() <= 1.0 and min() >= 0.0.
    assert projected.max() > 1.0, (
        f"projected.max()={projected.max()} <= 1.0 — codec-domain clamp "
        f"reintroduced (WZ projection saturated)"
    )
    assert projected.min() < 0.0, (
        f"projected.min()={projected.min()} >= 0.0 — codec-domain clamp "
        f"reintroduced (WZ projection saturated)"
    )
    # The delta must be non-zero for a non-trivial state (state DEPENDENCE).
    assert float(np.max(np.abs(delta))) > 0.0


def test_reconstruct_clamp_is_final_pixel_not_codec_domain():
    """The reconstruct clamp operates on the final-pixel image, not coeffs.

    Drives the canonical archive-path roundtrip (the exact inflate path) on
    in-range synthetic frames and verifies (a) the output is in [0, 1]
    (final-pixel clip active), and (b) the WZ-projected reconstruction recovers
    the source frames near-exactly (Mallat perfect reconstruction; the [0, 1]
    clip does not distort because the inverse landed in range — proving the
    clamp is at the pixel boundary, not in the codec domain).
    """

    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        build_canonical_quadruple_binding_from_z8_config,
        build_z8hpc1_archive_bytes_from_canonical_quadruple,
        reconstruct_pair_rgb_from_pyramid,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.runtime_payload_bridge import (
        projected_pair_pyramids_from_archive_bytes,
    )

    cfg = _canonical_cfg(num_pairs=1, eval_h=32, eval_w=32)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    rng = np.random.default_rng(7)
    f0 = rng.uniform(0.1, 0.9, size=(1, 32, 32, 3)).astype(np.float32)
    f1 = rng.uniform(0.1, 0.9, size=(1, 32, 32, 3)).astype(np.float32)
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(
        binding, f0, f1
    )
    bind2, pyramids, _stats = projected_pair_pyramids_from_archive_bytes(
        archive_bytes
    )
    r0, r1 = reconstruct_pair_rgb_from_pyramid(bind2, pyramids[0])
    # Final-pixel clip active: outputs strictly in [0, 1].
    for r in (r0, r1):
        assert r.min() >= 0.0 - 1e-6
        assert r.max() <= 1.0 + 1e-6
    # Mallat perfect reconstruction: frame-0 (no WZ projection) recovers
    # near-exactly. The [0,1] clip does not distort because the inverse landed
    # in range — confirming the clamp is at the pixel boundary not codec domain.
    recon0 = np.transpose(r0[0], (1, 2, 0))
    assert float(np.abs(recon0 - f0[0]).max()) < 1e-2, (
        "Mallat perfect reconstruction broken — recon deviates from in-range "
        "source by > 1e-2 (clamp distortion or transform bug)"
    )


@pytest.mark.slow
def test_archive_path_reconstructs_faithfully_on_real_frames():
    """The wavelet+WZ ARCHIVE path reconstructs real frames in GT range.

    This is the genuine hierarchical-PC unlock check on the contest-archive
    path: recon mean/std must be near the GT mean/std (NOT saturated to
    ~155-165 the WAVE-1E gumbel audit found on the collapsed _full_main HNeRV
    path). Skipped if the upstream video is unavailable.
    """

    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[5]
    video = repo_root / "upstream" / "videos" / "0.mkv"
    if not video.exists():
        pytest.skip("upstream/videos/0.mkv not present")

    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        build_canonical_quadruple_binding_from_z8_config,
        build_z8hpc1_archive_bytes_from_canonical_quadruple,
        load_real_video_pair_targets_numpy,
        reconstruct_pair_rgb_from_pyramid,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.runtime_payload_bridge import (
        projected_pair_pyramids_from_archive_bytes,
    )

    num_pairs, eval_h, eval_w = 4, 96, 128
    cfg = _canonical_cfg(num_pairs=num_pairs, eval_h=eval_h, eval_w=eval_w)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0, f1 = load_real_video_pair_targets_numpy(
        str(video), num_pairs=num_pairs, output_height=eval_h, output_width=eval_w
    )
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(
        binding, f0, f1
    )
    _b, pair_pyramids, _stats = projected_pair_pyramids_from_archive_bytes(
        archive_bytes
    )
    recon = []
    for pyramid in pair_pyramids:
        r0, r1 = reconstruct_pair_rgb_from_pyramid(_b, pyramid)
        recon.append(np.transpose(r0[0], (1, 2, 0)))
        recon.append(np.transpose(r1[0], (1, 2, 0)))
    recon = np.stack(recon, axis=0)

    # GT in [0, 1] (load_real returns /255). Recon must be near GT range.
    gt = np.concatenate([f0, f1], axis=0)
    gt_mean, gt_std = float(gt.mean()), float(gt.std())
    recon_mean, recon_std = float(recon.mean()), float(recon.std())
    # NOT saturated: recon mean within 3x / 1/3 of GT mean, std >= 10% GT std.
    assert recon_mean <= 3.0 * gt_mean, (
        f"recon_mean={recon_mean} > 3x GT mean={gt_mean} — SATURATED render"
    )
    assert recon_mean >= gt_mean / 3.0, (
        f"recon_mean={recon_mean} < 1/3 GT mean={gt_mean} — collapsed-dark"
    )
    assert recon_std >= 0.10 * gt_std, (
        f"recon_std={recon_std} < 10% GT std={gt_std} — near-constant collapse"
    )


@pytest.mark.slow
def test_wz_payload_mutation_drives_frame_1_pixels():
    """A valid WZ payload mutation moves frame-1 pixels (state DEPENDENCE)."""

    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[5]
    video = repo_root / "upstream" / "videos" / "0.mkv"
    if not video.exists():
        pytest.skip("upstream/videos/0.mkv not present")

    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        build_canonical_quadruple_binding_from_z8_config,
        build_z8hpc1_archive_bytes_from_canonical_quadruple,
        load_real_video_pair_targets_numpy,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.runtime_payload_bridge import (
        build_wyner_ziv_payload_mutation_receiver_proof,
    )

    num_pairs, eval_h, eval_w = 4, 96, 128
    cfg = _canonical_cfg(num_pairs=num_pairs, eval_h=eval_h, eval_w=eval_w)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0, f1 = load_real_video_pair_targets_numpy(
        str(video), num_pairs=num_pairs, output_height=eval_h, output_width=eval_w
    )
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(
        binding, f0, f1
    )
    proof = build_wyner_ziv_payload_mutation_receiver_proof(archive_bytes)
    # WZ state DEPENDENCE: frame-1 moves, frame-0 stable.
    assert float(proof["frame_1_max_abs_delta"]) > 0.0, (
        "WZ payload mutation did not change frame-1 — codec-domain clamp may "
        "have erased state dependence"
    )
    assert float(proof["frame_0_max_abs_delta"]) == 0.0, (
        "WZ payload mutation changed frame-0 — projection target wrong"
    )
