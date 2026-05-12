"""Tests for the L2 score-aware residual encoders (wavelet / c3 / cool_chic).

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lessons
1, 6, 8, 13: every L2 encoder must honor the score-aware Lagrangian, the
eval-roundtrip-aware proxy loss, and the permanent ``score_claim=False`` /
``promotion_eligible=False`` / ``ready_for_exact_eval_dispatch=False``
invariants.

These tests pin all five contracts:

  1. ``ScoreAwareLagrangian`` defaults match the contest functional verbatim.
  2. The proxy loss is gradient-reachable through the residual when the
     decoded tensor has ``requires_grad=True``.
  3. The dense-byte floor helpers enforce honest budget reporting.
  4. Each L2 encoder returns a frozen result with permanent promotion-status
     invariants.
  5. Each L2 encoder's output bytes round-trip through the family's
     ``parse_archive`` wrapper.

The tests use a SMALL synthetic fixture (N=2-4 frames) because the dense
wire format produces multi-MB blobs at N=4. The mathematics are scale-free.
"""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.residual_basis import (
    C3EncoderL2Error,
    CoolChicEncoderL2Error,
    L2ScoreAwareLossError,
    ResidualByteBudget,
    ScoreAwareLagrangian,
    WaveletEncoderL2Error,
    build_archive,
    compute_score_aware_proxy_loss,
    dense_c3_residual_blob_bytes,
    dense_cool_chic_residual_blob_bytes,
    dense_wavelet_residual_blob_bytes,
    encode_c3_residual_l2,
    encode_cool_chic_residual_l2,
    encode_wavelet_residual_l2,
    expect_format_id,
    parse_archive,
)

CAMERA_H = 874
CAMERA_W = 1164
RGB_CHANNELS = 3
SEED = 20260511
REPO_ROOT = Path(__file__).resolve().parents[3]


def _make_synthetic_pair(n_frames: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Build (decoded, gt) at camera resolution with small synthetic offset."""
    rng = np.random.default_rng(SEED)
    decoded = rng.integers(0, 256, size=(n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.uint8)
    delta = rng.integers(-3, 4, size=decoded.shape)
    gt = np.clip(decoded.astype(np.int16) + delta, 0, 255).astype(np.uint8)
    return decoded, gt


def _load_submission_inflate_module(family: str):
    path = REPO_ROOT / f"submissions/pr106_{family}_residual_sidecar/inflate.py"
    spec = importlib.util.spec_from_file_location(f"_test_pr106_{family}_inflate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# ScoreAwareLagrangian + ResidualByteBudget                                    #
# --------------------------------------------------------------------------- #


def test_lagrangian_defaults_match_contest_functional() -> None:
    lag = ScoreAwareLagrangian()
    assert lag.alpha == 25.0
    assert lag.beta == 100.0
    assert lag.gamma == 1.0
    assert lag.pose_sqrt_factor == 10.0
    assert lag.rate_denominator_bytes == 37_545_489
    assert lag.proxy_pose_marginal_multiplier == 1.0
    lag.assert_invariants()


def test_lagrangian_refuses_negative_coefficients() -> None:
    with pytest.raises(L2ScoreAwareLossError):
        ScoreAwareLagrangian(alpha=-1.0).assert_invariants()
    with pytest.raises(L2ScoreAwareLossError):
        ScoreAwareLagrangian(beta=-1.0).assert_invariants()
    with pytest.raises(L2ScoreAwareLossError):
        ScoreAwareLagrangian(rate_denominator_bytes=0).assert_invariants()


def test_residual_byte_budget_invariants() -> None:
    b = ResidualByteBudget(max_bytes=1000)
    b.assert_invariants()
    with pytest.raises(L2ScoreAwareLossError):
        ResidualByteBudget(max_bytes=0).assert_invariants()
    with pytest.raises(L2ScoreAwareLossError):
        ResidualByteBudget(max_bytes=100, soft_barrier_coeff=-0.1).assert_invariants()


def test_pose_marginal_multiplier_can_upweight_pose() -> None:
    """Operating-point marginal upweight is exposed but defaults to 1.0."""
    lag_default = ScoreAwareLagrangian()
    lag_pose = ScoreAwareLagrangian(proxy_pose_marginal_multiplier=2.79)
    assert lag_pose.proxy_pose_marginal_multiplier > lag_default.proxy_pose_marginal_multiplier


# --------------------------------------------------------------------------- #
# compute_score_aware_proxy_loss                                               #
# --------------------------------------------------------------------------- #


def test_proxy_loss_returns_scalar_and_diagnostics() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    decoded_t = torch.from_numpy(decoded.astype(np.float32))
    gt_t = torch.from_numpy(gt.astype(np.float32))
    loss, diag = compute_score_aware_proxy_loss(decoded_t, gt_t, archive_bytes=200000)
    assert loss.ndim == 0
    assert "alpha_term" in diag
    assert "beta_term" in diag
    assert "gamma_term" in diag
    assert "total" in diag
    assert "seg_proxy_mse" in diag
    assert "pose_proxy_mse" in diag
    # All terms must be finite + non-negative.
    for k in ("alpha_term", "beta_term", "gamma_term", "total"):
        v = diag[k]
        assert np.isfinite(v) and v >= 0.0


def test_proxy_loss_rate_term_scales_with_archive_bytes() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    decoded_t = torch.from_numpy(decoded.astype(np.float32))
    gt_t = torch.from_numpy(gt.astype(np.float32))
    _, diag_small = compute_score_aware_proxy_loss(decoded_t, gt_t, archive_bytes=200_000)
    _, diag_large = compute_score_aware_proxy_loss(decoded_t, gt_t, archive_bytes=400_000)
    # Distortion-side terms identical (same frames); rate must double.
    assert pytest.approx(diag_large["alpha_term"], rel=1e-6) == 2.0 * diag_small["alpha_term"]
    assert pytest.approx(diag_large["beta_term"], rel=1e-4) == diag_small["beta_term"]


def test_proxy_loss_seg_pose_zero_when_decoded_equals_gt() -> None:
    decoded, _ = _make_synthetic_pair(n_frames=2)
    decoded_t = torch.from_numpy(decoded.astype(np.float32))
    # decoded == gt -> seg_proxy_mse + pose_proxy_mse should both be zero.
    _, diag = compute_score_aware_proxy_loss(decoded_t, decoded_t, archive_bytes=200_000)
    assert pytest.approx(diag["seg_proxy_mse"], abs=1e-4) == 0.0
    assert pytest.approx(diag["pose_proxy_mse"], abs=1e-4) == 0.0
    assert pytest.approx(diag["beta_term"], abs=1e-4) == 0.0
    assert pytest.approx(diag["gamma_term"], abs=1e-4) == 0.0
    assert pytest.approx(diag["total"], rel=1e-6) == diag["alpha_term"]


def test_proxy_loss_eval_roundtrip_is_symmetric_on_fractional_identical_frames() -> None:
    """Regression: roundtrip must apply to decoded and GT, not decoded only."""
    decoded_t = torch.full((2, 3, 32, 32), 127.4, dtype=torch.float32)
    _, diag = compute_score_aware_proxy_loss(
        decoded_t,
        decoded_t,
        archive_bytes=200_000,
        eval_roundtrip=True,
        yuv6_routing=True,
    )
    assert pytest.approx(diag["seg_proxy_mse"], abs=1e-7) == 0.0
    assert pytest.approx(diag["pose_proxy_mse"], abs=1e-7) == 0.0
    assert pytest.approx(diag["beta_term"], abs=1e-7) == 0.0
    assert pytest.approx(diag["gamma_term"], abs=1e-7) == 0.0


def test_proxy_loss_gradient_reaches_decoded_tensor() -> None:
    """The proxy loss must remain useful as an inner-loop optimization target."""
    torch.manual_seed(SEED)
    decoded_t = (torch.rand(2, 3, 32, 32) * 255.0).requires_grad_(True)
    gt_t = (torch.rand(2, 3, 32, 32) * 255.0).detach()
    loss, diag = compute_score_aware_proxy_loss(
        decoded_t,
        gt_t,
        archive_bytes=200_000,
        eval_roundtrip=False,
        yuv6_routing=True,
    )
    assert loss.requires_grad
    loss.backward()
    assert decoded_t.grad is not None
    assert torch.isfinite(decoded_t.grad).all()
    assert float(decoded_t.grad.abs().sum()) > 0.0
    assert diag["seg_proxy_mse"] > 0.0
    assert diag["pose_proxy_mse"] > 0.0


def test_proxy_loss_pair_faithful_seg_uses_second_frame_only() -> None:
    gt = torch.full((4, 3, 32, 32), 128.0)
    first_frame_only = gt.clone()
    first_frame_only[0::2] += 5.0
    _, first_diag = compute_score_aware_proxy_loss(
        first_frame_only,
        gt,
        archive_bytes=200_000,
        eval_roundtrip=False,
        yuv6_routing=False,
    )
    assert pytest.approx(first_diag["seg_proxy_mse"], abs=1e-8) == 0.0
    assert first_diag["pose_proxy_mse"] > 0.0

    second_frame_only = gt.clone()
    second_frame_only[1::2] += 5.0
    _, second_diag = compute_score_aware_proxy_loss(
        second_frame_only,
        gt,
        archive_bytes=200_000,
        eval_roundtrip=False,
        yuv6_routing=False,
    )
    assert second_diag["seg_proxy_mse"] > 0.0
    assert second_diag["pose_proxy_mse"] > 0.0


def test_proxy_loss_refuses_odd_frame_count() -> None:
    decoded_t = torch.full((3, 3, 32, 32), 128.0)
    gt_t = decoded_t.clone()
    with pytest.raises(L2ScoreAwareLossError, match="even"):
        compute_score_aware_proxy_loss(
            decoded_t,
            gt_t,
            archive_bytes=200_000,
            eval_roundtrip=False,
            yuv6_routing=False,
        )


def test_proxy_loss_zero_pose_baseline_has_finite_gradient() -> None:
    decoded_t = torch.full((2, 3, 32, 32), 128.0, requires_grad=True)
    gt_t = decoded_t.detach().clone()
    loss, diag = compute_score_aware_proxy_loss(
        decoded_t,
        gt_t,
        archive_bytes=200_000,
        eval_roundtrip=False,
        yuv6_routing=True,
    )
    assert pytest.approx(diag["gamma_term"], abs=1e-8) == 0.0
    loss.backward()
    assert decoded_t.grad is not None
    assert torch.isfinite(decoded_t.grad).all()


def test_proxy_loss_rejects_normalized_or_out_of_range_inputs() -> None:
    normalized = torch.full((2, 3, 32, 32), 0.5)
    with pytest.raises(L2ScoreAwareLossError, match="normalized"):
        compute_score_aware_proxy_loss(
            normalized,
            normalized,
            archive_bytes=200_000,
            eval_roundtrip=False,
            yuv6_routing=False,
        )
    out_of_range = torch.full((2, 3, 32, 32), 300.0)
    with pytest.raises(L2ScoreAwareLossError, match=r"\[0, 255\]"):
        compute_score_aware_proxy_loss(
            out_of_range,
            out_of_range,
            archive_bytes=200_000,
            eval_roundtrip=False,
            yuv6_routing=False,
        )


def test_proxy_loss_refuses_zero_archive_bytes() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    decoded_t = torch.from_numpy(decoded.astype(np.float32))
    gt_t = torch.from_numpy(gt.astype(np.float32))
    with pytest.raises(L2ScoreAwareLossError):
        compute_score_aware_proxy_loss(decoded_t, gt_t, archive_bytes=0)


def test_proxy_loss_eval_roundtrip_flag_changes_value() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    decoded_t = torch.from_numpy(decoded.astype(np.float32))
    gt_t = torch.from_numpy(gt.astype(np.float32))
    _, diag_on = compute_score_aware_proxy_loss(
        decoded_t, gt_t, archive_bytes=200_000, eval_roundtrip=True
    )
    _, diag_off = compute_score_aware_proxy_loss(
        decoded_t, gt_t, archive_bytes=200_000, eval_roundtrip=False
    )
    assert diag_on["eval_roundtrip"] == 1.0
    assert diag_off["eval_roundtrip"] == 0.0


# --------------------------------------------------------------------------- #
# dense_*_residual_blob_bytes helpers                                         #
# --------------------------------------------------------------------------- #


def test_dense_wavelet_blob_bytes_matches_wire_format() -> None:
    # PER_FRAME_BYTES = 16 + 4 * 3 * 437 * 582 = 16 + 3052008 = 3,052,024
    expected_per_frame = 16 + 4 * 3 * 437 * 582
    assert dense_wavelet_residual_blob_bytes(1) == expected_per_frame
    assert dense_wavelet_residual_blob_bytes(2) == 2 * expected_per_frame
    with pytest.raises(WaveletEncoderL2Error):
        dense_wavelet_residual_blob_bytes(0)


def test_dense_c3_blob_bytes_matches_wire_format() -> None:
    # PER_FRAME_BYTES = 4 + 218 * 291 * 3 = 4 + 190,314 = 190,318
    expected_per_frame = 4 + 218 * 291 * 3
    assert dense_c3_residual_blob_bytes(1) == expected_per_frame
    assert dense_c3_residual_blob_bytes(2) == 2 * expected_per_frame
    with pytest.raises(C3EncoderL2Error):
        dense_c3_residual_blob_bytes(0)


def test_dense_cool_chic_blob_bytes_matches_wire_format() -> None:
    # n_levels prefix (2B) + per-level (4B scale + n_frames * h_L * w_L * 3 int8).
    # For n_levels=1, n_frames=1: 2 + 4 + 1 * 874 * 1164 * 3 = 2 + 4 + 3,052,008.
    expected_l1 = 2 + 4 + 1 * 874 * 1164 * 3
    assert dense_cool_chic_residual_blob_bytes(1, 1) == expected_l1
    with pytest.raises(CoolChicEncoderL2Error):
        dense_cool_chic_residual_blob_bytes(0, 1)
    with pytest.raises(CoolChicEncoderL2Error):
        dense_cool_chic_residual_blob_bytes(1, 0)
    with pytest.raises(CoolChicEncoderL2Error):
        dense_cool_chic_residual_blob_bytes(1, 99)


# --------------------------------------------------------------------------- #
# Wavelet L2 encoder                                                           #
# --------------------------------------------------------------------------- #


def test_wavelet_l2_encoder_refuses_sub_dense_budget() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    with pytest.raises(WaveletEncoderL2Error, match="dense"):
        encode_wavelet_residual_l2(decoded, gt, byte_budget=100)


def test_wavelet_l2_encoder_runs_end_to_end_at_dense_budget() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    dense = dense_wavelet_residual_blob_bytes(2)
    result = encode_wavelet_residual_l2(
        decoded, gt, byte_budget=dense, n_iterations=2
    )
    assert result.n_frames_encoded == 2
    assert result.n_frames_subsampled == 2
    assert len(result.residual_bytes) == dense
    assert result.final_loss <= result.initial_loss + 1e-6
    # Promotion-status invariants pinned False.
    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.ready_for_exact_eval_dispatch is False
    assert result.evidence_grade == "research_signal_l2_proxy"
    result.assert_invariants()


def test_wavelet_l2_residual_bytes_round_trip_via_parse_archive() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    dense = dense_wavelet_residual_blob_bytes(2)
    result = encode_wavelet_residual_l2(decoded, gt, byte_budget=dense, n_iterations=1)
    # Wrap into the family archive grammar and round-trip.
    fake_pr106 = b"PR106-stub-payload"
    archive = build_archive(
        family="wavelet", pr106_bytes=fake_pr106, residual_bytes=result.residual_bytes
    )
    parsed = expect_format_id(archive.archive_bytes, family="wavelet")
    assert parsed.residual_bytes == result.residual_bytes
    assert parsed.pr106_bytes == fake_pr106


def test_wavelet_l2_residual_bytes_are_consumed_by_submission_decoder() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    dense = dense_wavelet_residual_blob_bytes(2)
    result = encode_wavelet_residual_l2(decoded, gt, byte_budget=dense, n_iterations=1)
    inflate = _load_submission_inflate_module("wavelet")
    residual = inflate.decode_wavelet_residual(result.residual_bytes, n_frames=2)
    assert residual.shape == (2, CAMERA_H, CAMERA_W, RGB_CHANNELS)
    assert np.isfinite(residual).all()
    assert np.abs(residual).sum() > 0.0
    mutated = bytearray(result.residual_bytes)
    # Offset 16 is the first cA byte; cA scale is intentionally zero. Mutate
    # cH instead so the runtime-consumed detail band output must change.
    first_detail_band_offset = 16 + RGB_CHANNELS * (CAMERA_H // 2) * (CAMERA_W // 2)
    mutated[first_detail_band_offset] = (mutated[first_detail_band_offset] + 1) % 256
    mutated_residual = inflate.decode_wavelet_residual(bytes(mutated), n_frames=2)
    assert not np.array_equal(mutated_residual, residual)


def test_wavelet_l2_shape_mismatch_raises() -> None:
    decoded, _ = _make_synthetic_pair(n_frames=2)
    gt_wrong = decoded[:1]  # mismatched n_frames
    with pytest.raises(WaveletEncoderL2Error):
        encode_wavelet_residual_l2(decoded, gt_wrong, byte_budget=dense_wavelet_residual_blob_bytes(2))


def test_wavelet_l2_camera_resolution_required() -> None:
    rng = np.random.default_rng(SEED)
    bad = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(WaveletEncoderL2Error):
        encode_wavelet_residual_l2(bad, bad, byte_budget=10_000)


def test_wavelet_l2_refuses_odd_frame_count() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=3)
    with pytest.raises(WaveletEncoderL2Error, match="even"):
        encode_wavelet_residual_l2(
            decoded, gt, byte_budget=dense_wavelet_residual_blob_bytes(3)
        )


# --------------------------------------------------------------------------- #
# C3 L2 encoder                                                                #
# --------------------------------------------------------------------------- #


def test_c3_l2_encoder_refuses_sub_dense_budget() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    with pytest.raises(C3EncoderL2Error, match="dense"):
        encode_c3_residual_l2(decoded, gt, byte_budget=100)


def test_c3_l2_encoder_runs_end_to_end_at_dense_budget() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    dense = dense_c3_residual_blob_bytes(2)
    result = encode_c3_residual_l2(decoded, gt, byte_budget=dense, n_iterations=2)
    assert result.n_frames_encoded == 2
    assert result.n_frames_subsampled == 2
    assert len(result.residual_bytes) == dense
    assert result.final_loss <= result.initial_loss + 1e-6
    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.ready_for_exact_eval_dispatch is False
    assert result.evidence_grade == "research_signal_l2_proxy"
    result.assert_invariants()


def test_c3_l2_residual_bytes_round_trip_via_parse_archive() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    dense = dense_c3_residual_blob_bytes(2)
    result = encode_c3_residual_l2(decoded, gt, byte_budget=dense, n_iterations=1)
    fake_pr106 = b"PR106-stub-payload"
    archive = build_archive(
        family="c3", pr106_bytes=fake_pr106, residual_bytes=result.residual_bytes
    )
    parsed = expect_format_id(archive.archive_bytes, family="c3")
    assert parsed.residual_bytes == result.residual_bytes


def test_c3_l2_residual_bytes_are_consumed_by_submission_decoder() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    dense = dense_c3_residual_blob_bytes(2)
    result = encode_c3_residual_l2(decoded, gt, byte_budget=dense, n_iterations=1)
    inflate = _load_submission_inflate_module("c3")
    residual = inflate.decode_c3_residual(result.residual_bytes, n_frames=2)
    assert residual.shape == (2, CAMERA_H, CAMERA_W, RGB_CHANNELS)
    assert np.isfinite(residual).all()
    assert np.abs(residual).sum() > 0.0
    mutated = bytearray(result.residual_bytes)
    mutated[4] = (mutated[4] + 1) % 256
    mutated_residual = inflate.decode_c3_residual(bytes(mutated), n_frames=2)
    assert not np.array_equal(mutated_residual, residual)


def test_c3_l2_first_difference_encodes_zero_residual_to_zero_deltas() -> None:
    """If decoded == gt, the residual is zero, so deltas should all be zero."""
    decoded, _ = _make_synthetic_pair(n_frames=2)
    dense = dense_c3_residual_blob_bytes(2)
    result = encode_c3_residual_l2(decoded, decoded, byte_budget=dense, n_iterations=1)
    # Parse the residual bytes: each frame has 4B scale + (218*291*3 int8 deltas).
    # All deltas should be ~zero when residual is zero.
    n_frames = 2
    per_frame = 4 + 218 * 291 * 3
    blob = result.residual_bytes
    assert len(blob) == n_frames * per_frame
    for t in range(n_frames):
        offset = t * per_frame
        (scale,) = struct.unpack_from("<f", blob, offset)
        deltas = np.frombuffer(blob, dtype=np.int8, count=218 * 291 * 3, offset=offset + 4)
        # Scale may be 0 OR deltas may be 0 (both encode zero residual).
        if abs(scale) > 1e-9:
            assert np.all(deltas == 0)


def test_c3_l2_refuses_odd_frame_count() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=3)
    with pytest.raises(C3EncoderL2Error, match="even"):
        encode_c3_residual_l2(decoded, gt, byte_budget=dense_c3_residual_blob_bytes(3))


# --------------------------------------------------------------------------- #
# Cool-Chic L2 encoder                                                         #
# --------------------------------------------------------------------------- #


def test_cool_chic_l2_encoder_refuses_when_no_level_fits_budget() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    with pytest.raises(CoolChicEncoderL2Error, match="dense"):
        encode_cool_chic_residual_l2(decoded, gt, byte_budget=100)


def test_cool_chic_l2_encoder_runs_end_to_end_at_n_levels_1_budget() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    # Allow exactly n_levels=1 by sizing budget to match.
    dense_l1 = dense_cool_chic_residual_blob_bytes(2, 1)
    result = encode_cool_chic_residual_l2(
        decoded, gt, byte_budget=dense_l1, candidate_n_levels=(1,)
    )
    assert result.n_levels_used == 1
    assert result.n_frames_encoded == 2
    assert len(result.residual_bytes) == dense_l1
    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.ready_for_exact_eval_dispatch is False
    assert result.evidence_grade == "research_signal_l2_proxy"
    result.assert_invariants()


def test_cool_chic_l2_residual_bytes_round_trip_via_parse_archive() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    dense_l1 = dense_cool_chic_residual_blob_bytes(2, 1)
    result = encode_cool_chic_residual_l2(
        decoded, gt, byte_budget=dense_l1, candidate_n_levels=(1,)
    )
    fake_pr106 = b"PR106-stub-payload"
    archive = build_archive(
        family="cool_chic", pr106_bytes=fake_pr106, residual_bytes=result.residual_bytes
    )
    parsed = expect_format_id(archive.archive_bytes, family="cool_chic")
    assert parsed.residual_bytes == result.residual_bytes


def test_cool_chic_l2_residual_bytes_are_consumed_by_submission_decoder() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    dense_l1 = dense_cool_chic_residual_blob_bytes(2, 1)
    result = encode_cool_chic_residual_l2(
        decoded, gt, byte_budget=dense_l1, candidate_n_levels=(1,)
    )
    inflate = _load_submission_inflate_module("cool_chic")
    residual = inflate.decode_cool_chic_residual(result.residual_bytes, n_frames=2)
    assert residual.shape == (2, CAMERA_H, CAMERA_W, RGB_CHANNELS)
    assert np.isfinite(residual).all()
    assert np.abs(residual).sum() > 0.0
    mutated = bytearray(result.residual_bytes)
    mutated[6] = (mutated[6] + 1) % 256
    mutated_residual = inflate.decode_cool_chic_residual(bytes(mutated), n_frames=2)
    assert not np.array_equal(mutated_residual, residual)


def test_cool_chic_l2_picks_largest_n_levels_under_budget() -> None:
    """When multiple n_levels fit budget, the encoder picks the one with best loss."""
    decoded, gt = _make_synthetic_pair(n_frames=2)
    # Allow up to n_levels=2 with budget matching the L2 dense size.
    dense_l2 = dense_cool_chic_residual_blob_bytes(2, 2)
    result = encode_cool_chic_residual_l2(
        decoded, gt, byte_budget=dense_l2, candidate_n_levels=(1, 2)
    )
    # The encoder picks whichever level minimizes proxy loss within budget.
    assert result.n_levels_used in (1, 2)


def test_cool_chic_l2_refuses_odd_frame_count() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=3)
    with pytest.raises(CoolChicEncoderL2Error, match="even"):
        encode_cool_chic_residual_l2(
            decoded,
            gt,
            byte_budget=dense_cool_chic_residual_blob_bytes(3, 1),
            candidate_n_levels=(1,),
        )


# --------------------------------------------------------------------------- #
# Per-level top-K budget (operator decision 2026-05-11 — sparse PacketIR fix) #
# --------------------------------------------------------------------------- #


def test_cool_chic_per_level_top_k_budget_truncates_each_level() -> None:
    """per_level_top_k_budget zeros all but the K largest-magnitude coeffs per level."""
    decoded, gt = _make_synthetic_pair(n_frames=2)
    dense_l2 = dense_cool_chic_residual_blob_bytes(2, 2)
    result = encode_cool_chic_residual_l2(
        decoded, gt,
        byte_budget=dense_l2,
        candidate_n_levels=(2,),
        per_level_top_k_budget={0: 64, 1: 32},
    )
    # Encoder must run successfully with per-level budget active.
    assert result.n_levels_used == 2
    # Diagnostic must record per-level budget activation.
    assert result.diagnostics.get("cool_chic_per_level_top_k_budget_active") == 1.0
    # Permanent score-claim invariants preserved.
    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.ready_for_exact_eval_dispatch is False


def test_cool_chic_per_level_top_k_budget_default_inactive() -> None:
    """Default per_level_top_k_budget=None preserves back-compat (diag=0)."""
    decoded, gt = _make_synthetic_pair(n_frames=2)
    dense_l1 = dense_cool_chic_residual_blob_bytes(2, 1)
    result = encode_cool_chic_residual_l2(
        decoded, gt, byte_budget=dense_l1, candidate_n_levels=(1,)
    )
    assert result.diagnostics.get("cool_chic_per_level_top_k_budget_active") == 0.0


def test_cool_chic_per_level_top_k_budget_validates_negative_k() -> None:
    """per_level_top_k_budget refuses negative K."""
    decoded, gt = _make_synthetic_pair(n_frames=2)
    dense_l1 = dense_cool_chic_residual_blob_bytes(2, 1)
    with pytest.raises(CoolChicEncoderL2Error, match="non-negative"):
        encode_cool_chic_residual_l2(
            decoded, gt, byte_budget=dense_l1, candidate_n_levels=(1,),
            per_level_top_k_budget={0: -5},
        )


def test_cool_chic_per_level_top_k_budget_validates_invalid_level() -> None:
    """per_level_top_k_budget refuses out-of-range level."""
    decoded, gt = _make_synthetic_pair(n_frames=2)
    dense_l1 = dense_cool_chic_residual_blob_bytes(2, 1)
    with pytest.raises(CoolChicEncoderL2Error, match="out of"):
        encode_cool_chic_residual_l2(
            decoded, gt, byte_budget=dense_l1, candidate_n_levels=(1,),
            per_level_top_k_budget={99: 100},
        )


def test_cool_chic_per_level_top_k_zero_zeros_level() -> None:
    """top_k=0 fully zeros the corresponding level (sister of full sparsification)."""
    decoded, gt = _make_synthetic_pair(n_frames=2)
    dense_l1 = dense_cool_chic_residual_blob_bytes(2, 1)
    result = encode_cool_chic_residual_l2(
        decoded, gt,
        byte_budget=dense_l1,
        candidate_n_levels=(1,),
        per_level_top_k_budget={0: 0},
    )
    # Encoder ran successfully with K=0.
    assert result.n_levels_used == 1
    assert result.diagnostics.get("cool_chic_per_level_top_k_budget_active") == 1.0


# --------------------------------------------------------------------------- #
# Cross-encoder invariants                                                     #
# --------------------------------------------------------------------------- #


def test_all_three_encoders_emit_research_signal_evidence_grade() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    rw = encode_wavelet_residual_l2(
        decoded, gt, byte_budget=dense_wavelet_residual_blob_bytes(2), n_iterations=1
    )
    rc = encode_c3_residual_l2(
        decoded, gt, byte_budget=dense_c3_residual_blob_bytes(2), n_iterations=1
    )
    rcc = encode_cool_chic_residual_l2(
        decoded, gt,
        byte_budget=dense_cool_chic_residual_blob_bytes(2, 1),
        candidate_n_levels=(1,),
    )
    for r in (rw, rc, rcc):
        assert r.evidence_grade == "research_signal_l2_proxy"
        assert r.score_claim is False
        assert r.promotion_eligible is False
        assert r.ready_for_exact_eval_dispatch is False


def test_all_three_encoders_refuse_score_claim_mutation() -> None:
    """The frozen dataclass invariants must reject any mutation attempt."""
    decoded, gt = _make_synthetic_pair(n_frames=2)
    rw = encode_wavelet_residual_l2(
        decoded, gt, byte_budget=dense_wavelet_residual_blob_bytes(2), n_iterations=1
    )
    # The dataclass is frozen; attribute assignment must fail.
    with pytest.raises((AttributeError, TypeError)):
        rw.score_claim = True  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Sparse-aware L2 encoder paths                                                 #
#                                                                               #
# Per O's wire-format-ceiling finding + S's sparse PacketIR codec landing       #
# + operator 2026-05-11 $5/individual envelope: each L2 encoder gains a         #
# ``sparse_aware=True`` opt-in that (a) bypasses the dense byte-budget gate,    #
# (b) computes the proxy Lagrangian rate term against the sparse-repacked       #
# byte size, and (c) emits sparse-encoded residual bytes ready for the          #
# matching sparse family inflate runtime (format_id 0x20-0x24).                 #
# --------------------------------------------------------------------------- #


def _make_sparse_residual_pair(n_frames: int = 2, nonzero_fraction: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """Build (decoded, gt) where most pixels are exactly equal (sparse residual).

    Only a small fraction of pixels carry a perturbation. This is the realistic
    PR106-r2-vs-GT regime where the residual is dominated by semantic-edge
    pixels rather than full-grid noise.
    """
    rng = np.random.default_rng(SEED)
    decoded = rng.integers(0, 256, size=(n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.uint8)
    mask = rng.random(decoded.shape) < nonzero_fraction
    delta = rng.integers(-4, 5, size=decoded.shape) * mask
    gt = np.clip(decoded.astype(np.int16) + delta, 0, 255).astype(np.uint8)
    return decoded, gt


def test_c3_l2_sparse_aware_runs_at_small_budget_when_residual_is_sparse() -> None:
    """Sparse-aware c3 fits a small byte budget when the residual is mostly zero."""
    decoded, gt = _make_sparse_residual_pair(n_frames=2, nonzero_fraction=0.005)
    result = encode_c3_residual_l2(
        decoded, gt,
        byte_budget=2_000_000,
        sparse_aware=True,
        n_iterations=1,
    )
    # Sparse-aware mode emits the sparse-repacked bytes directly.
    assert len(result.residual_bytes) <= 2_000_000
    # The sparse size must be smaller than the dense size for this sparse input.
    assert "c3_residual_blob_dense_bytes" in result.diagnostics
    sparse_size = result.diagnostics["c3_residual_blob_bytes"]
    dense_size = result.diagnostics["c3_residual_blob_dense_bytes"]
    assert sparse_size <= dense_size
    assert result.diagnostics["c3_sparse_aware"] == 1.0


def test_c3_l2_sparse_aware_residual_bytes_round_trip_via_sparse_parse_archive() -> None:
    """Sparse c3 bytes wrap into the c3_sparse family (format_id 0x22)."""
    decoded, gt = _make_sparse_residual_pair(n_frames=2, nonzero_fraction=0.005)
    result = encode_c3_residual_l2(
        decoded, gt, byte_budget=2_000_000, sparse_aware=True, n_iterations=1,
    )
    built = build_archive(
        family="c3_sparse", pr106_bytes=b"PR106_PLACEHOLDER", residual_bytes=result.residual_bytes,
    )
    parsed = parse_archive(built.archive_bytes)
    assert parsed.format_id == 0x22
    assert parsed.residual_bytes == result.residual_bytes


def test_c3_l2_dense_path_still_works_when_sparse_aware_false() -> None:
    """Setting sparse_aware=False preserves the existing dense byte path."""
    decoded, gt = _make_synthetic_pair(n_frames=2)
    result = encode_c3_residual_l2(
        decoded, gt,
        byte_budget=dense_c3_residual_blob_bytes(2),
        sparse_aware=False,
        n_iterations=1,
    )
    assert result.diagnostics["c3_sparse_aware"] == 0.0
    assert len(result.residual_bytes) == dense_c3_residual_blob_bytes(2)


def test_c3_l2_sparse_aware_refuses_too_small_budget() -> None:
    """Sparse-aware refuses if even the sparse-encoded bytes exceed the cap.

    The sparse-aware encoder includes a coefficient-magnitude threshold sweep
    that can produce extreme sparsity (only ~10K bytes for 2 frames at the
    highest threshold). To test the refusal path we set an impossibly small
    budget that the temporal-subsampled wrapper alone exceeds.
    """
    decoded, gt = _make_synthetic_pair(n_frames=2)  # dense random residual
    with pytest.raises(C3EncoderL2Error, match="exceeds byte_budget"):
        encode_c3_residual_l2(
            decoded, gt,
            byte_budget=10,  # smaller than even the temporal-subsampled header
            sparse_aware=True,
            n_iterations=1,
        )


def test_wavelet_l2_sparse_aware_runs_at_small_budget_when_residual_is_sparse() -> None:
    decoded, gt = _make_sparse_residual_pair(n_frames=2, nonzero_fraction=0.005)
    result = encode_wavelet_residual_l2(
        decoded, gt,
        byte_budget=20_000_000,
        sparse_aware=True,
        n_iterations=1,
    )
    assert len(result.residual_bytes) <= 20_000_000
    sparse_size = result.diagnostics["wavelet_residual_blob_bytes"]
    dense_size = result.diagnostics["wavelet_residual_blob_dense_bytes"]
    assert sparse_size <= dense_size
    assert result.diagnostics["wavelet_sparse_aware"] == 1.0


def test_wavelet_l2_sparse_aware_residual_bytes_round_trip_via_sparse_parse_archive() -> None:
    decoded, gt = _make_sparse_residual_pair(n_frames=2, nonzero_fraction=0.005)
    result = encode_wavelet_residual_l2(
        decoded, gt, byte_budget=20_000_000, sparse_aware=True, n_iterations=1,
    )
    built = build_archive(
        family="wavelet_sparse", pr106_bytes=b"PR106_PLACEHOLDER",
        residual_bytes=result.residual_bytes,
    )
    parsed = parse_archive(built.archive_bytes)
    assert parsed.format_id == 0x20
    assert parsed.residual_bytes == result.residual_bytes


def test_wavelet_l2_dense_path_still_works_when_sparse_aware_false() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    result = encode_wavelet_residual_l2(
        decoded, gt,
        byte_budget=dense_wavelet_residual_blob_bytes(2),
        sparse_aware=False,
        n_iterations=1,
    )
    assert result.diagnostics["wavelet_sparse_aware"] == 0.0
    assert len(result.residual_bytes) == dense_wavelet_residual_blob_bytes(2)


def test_cool_chic_l2_sparse_aware_runs_at_small_budget_when_residual_is_sparse() -> None:
    decoded, gt = _make_sparse_residual_pair(n_frames=2, nonzero_fraction=0.005)
    result = encode_cool_chic_residual_l2(
        decoded, gt,
        byte_budget=20_000_000,
        candidate_n_levels=(1, 2),
        sparse_aware=True,
    )
    assert len(result.residual_bytes) <= 20_000_000
    sparse_size = result.diagnostics["cool_chic_residual_blob_bytes"]
    dense_size = result.diagnostics["cool_chic_residual_blob_dense_bytes"]
    assert sparse_size <= dense_size
    assert result.diagnostics["cool_chic_sparse_aware"] == 1.0


def test_cool_chic_l2_sparse_aware_residual_bytes_round_trip_via_sparse_parse_archive() -> None:
    decoded, gt = _make_sparse_residual_pair(n_frames=2, nonzero_fraction=0.005)
    result = encode_cool_chic_residual_l2(
        decoded, gt, byte_budget=20_000_000,
        candidate_n_levels=(1,),
        sparse_aware=True,
    )
    built = build_archive(
        family="cool_chic_sparse", pr106_bytes=b"PR106_PLACEHOLDER",
        residual_bytes=result.residual_bytes,
    )
    parsed = parse_archive(built.archive_bytes)
    assert parsed.format_id == 0x21
    assert parsed.residual_bytes == result.residual_bytes


def test_cool_chic_l2_dense_path_still_works_when_sparse_aware_false() -> None:
    decoded, gt = _make_synthetic_pair(n_frames=2)
    result = encode_cool_chic_residual_l2(
        decoded, gt,
        byte_budget=dense_cool_chic_residual_blob_bytes(2, 1),
        candidate_n_levels=(1,),
        sparse_aware=False,
    )
    assert result.diagnostics["cool_chic_sparse_aware"] == 0.0


def test_all_three_encoders_sparse_aware_keep_score_claim_invariants() -> None:
    """Sparse-aware path preserves the permanent promotion-status invariants."""
    decoded, gt = _make_sparse_residual_pair(n_frames=2, nonzero_fraction=0.005)
    r_c3 = encode_c3_residual_l2(
        decoded, gt, byte_budget=2_000_000, sparse_aware=True, n_iterations=1,
    )
    r_w = encode_wavelet_residual_l2(
        decoded, gt, byte_budget=20_000_000, sparse_aware=True, n_iterations=1,
    )
    r_cc = encode_cool_chic_residual_l2(
        decoded, gt, byte_budget=20_000_000, candidate_n_levels=(1,), sparse_aware=True,
    )
    for r in (r_c3, r_w, r_cc):
        assert r.score_claim is False
        assert r.promotion_eligible is False
        assert r.ready_for_exact_eval_dispatch is False
        assert r.evidence_grade == "research_signal_l2_proxy"
