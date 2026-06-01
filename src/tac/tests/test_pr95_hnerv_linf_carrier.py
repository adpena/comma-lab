# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the PR95-HNeRV inverse-steganalysis carrier wiring.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" (Slot EEE 5-class) + "tests-verify-
constants-not-behavior" forbidden class: every test here would FAIL if the
function body were replaced by ``return canonical_markers``. The proofs run on
the REAL PR95-HNeRV carrier (``archive.pr95_repacked.zip``) + REAL
``upstream/videos/0.mkv`` frames — NOT synthetic-noise fixtures, NOT toy tensors.

Key behavioral proofs:
  * the carrier renders DISTINCT real content per pair (not a constant);
  * the rate term is EXACTLY ``25 * archive_bytes / N`` (computed from REAL bytes);
  * the Z8-falsification ratio is computed from the REAL carrier bytes;
  * the Fisher-pullback latent saliency VARIES with the pixel saliency (a
    different pixel-saliency surface -> a different latent saliency);
  * the L-inf latent allocation DIFFERS from L2 and is forced to spend >= L2 rate
    (the §7 anti-gaming guard), and a RANDOM saliency -> a DIFFERENT allocation
    than the oracle (genuine detector-aiming, not a placeholder);
  * latent quantization actually CHANGES the latents (the decode the rate model
    assumes is real);
  * the advisory d_seg/d_pose are NON-PROMOTABLE and finite;
  * the head-to-head row carries the full NON_PROMOTABLE markers.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from tac.analysis.pr95_hnerv_linf_carrier import (
    CONTEST_LAMBDA,
    CONTEST_RATE_DENOM_BYTES,
    CONTEST_RATE_MULTIPLIER,
    NON_PROMOTABLE_MARKERS,
    Z8_NEAR_LOSSLESS_ARCHIVE_BYTES,
    Pr95HnervCarrierError,
    allocate_latent_linf_vs_l2,
    build_head_to_head_row,
    carrier_rate_term,
    load_carrier_decoder,
    measure_carrier_distortion,
    push_pixel_saliency_to_latent,
    quantize_latent_with_steps,
    render_carrier_pair_bcthw,
    z8_falsification,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex"
    / "archive.pr95_repacked.zip"
)
REAL_VIDEO = REPO_ROOT / "upstream/videos/0.mkv"


def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401

        return True
    except Exception:
        return False


requires_real_carrier = pytest.mark.skipif(
    not REAL_ARCHIVE.is_file(), reason="real PR95-HNeRV carrier archive not present"
)
requires_mlx = pytest.mark.skipif(not _mlx_available(), reason="MLX not available")
requires_video = pytest.mark.skipif(
    not REAL_VIDEO.is_file(), reason="real upstream/videos/0.mkv not present"
)


# ---------------------------------------------------------------------------
# Rate term — computed from REAL carrier bytes, exactly 25*bytes/N.
# ---------------------------------------------------------------------------


@requires_real_carrier
def test_rate_term_is_exactly_lambda_times_real_bytes() -> None:
    rt = carrier_rate_term(REAL_ARCHIVE)
    # The bytes are the REAL on-disk archive size, not asserted constant.
    assert rt.archive_bytes == REAL_ARCHIVE.stat().st_size
    assert rt.archive_bytes > 100_000  # a real PR95 carrier, not empty
    expected = CONTEST_RATE_MULTIPLIER * rt.archive_bytes / CONTEST_RATE_DENOM_BYTES
    assert rt.rate_term == pytest.approx(expected, rel=1e-12)
    # If rate_term were a hardcoded constant, this would fail for the real bytes.
    assert rt.rate_term == pytest.approx(
        25.0 * rt.archive_bytes / 37_545_489, rel=1e-12
    )
    assert rt.n_pairs == 600
    assert rt.latent_dim == 28
    assert rt.cheap_by_construction is True


@requires_real_carrier
def test_rate_term_scales_with_archive_bytes() -> None:
    """A bigger archive => a strictly bigger rate term (the rate is byte-linear)."""
    rt = carrier_rate_term(REAL_ARCHIVE)
    bigger = rt.archive_bytes * 2
    bigger_rate = CONTEST_RATE_MULTIPLIER * bigger / CONTEST_RATE_DENOM_BYTES
    assert bigger_rate == pytest.approx(2.0 * rt.rate_term, rel=1e-12)


def test_lambda_constant_is_verified_contest_value() -> None:
    assert pytest.approx(25.0 / 37_545_489, rel=1e-15) == CONTEST_LAMBDA
    # 1502 bytes ~ 0.001 score (the verified byte price).
    assert pytest.approx(0.001, rel=2e-3) == 1502 * CONTEST_LAMBDA


def test_missing_archive_raises() -> None:
    with pytest.raises(Pr95HnervCarrierError):
        carrier_rate_term(REPO_ROOT / "experiments/results/__does_not_exist__/x.zip")


# ---------------------------------------------------------------------------
# Z8-falsification — ratio computed from REAL carrier bytes.
# ---------------------------------------------------------------------------


@requires_real_carrier
def test_z8_falsification_ratio_from_real_bytes() -> None:
    rt = carrier_rate_term(REAL_ARCHIVE)
    fz = z8_falsification(rt)
    assert fz.z8_archive_bytes == Z8_NEAR_LOSSLESS_ARCHIVE_BYTES
    assert fz.pr95_hnerv_archive_bytes == rt.archive_bytes
    # The ratio IS Z8_bytes / carrier_bytes (computed, not a constant).
    assert fz.z8_over_pr95_byte_ratio == pytest.approx(
        Z8_NEAR_LOSSLESS_ARCHIVE_BYTES / rt.archive_bytes, rel=1e-9
    )
    # Z8 is >> 100x heavier than the cheap-by-construction PR95-HNeRV carrier.
    assert fz.z8_over_pr95_byte_ratio > 100.0
    assert fz.z8_disease_confirmed is True
    # The carrier rate term alone is far below the Z8 rate term.
    assert fz.pr95_hnerv_rate_term < fz.z8_rate_term
    assert fz.z8_rate_term > 1.0  # Z8 rate alone exceeds the whole frontier score


def test_z8_falsification_ratio_inverts_with_carrier_size() -> None:
    """A heavier carrier => a SMALLER Z8/carrier ratio (the ratio is real, not fixed)."""
    from tac.analysis.pr95_hnerv_linf_carrier import CarrierRateTerm

    light = CarrierRateTerm(
        archive_bytes=178_321, archive_sha256="x", n_pairs=600, latent_dim=28,
        base_channels=36, rate_term=0.0, cheap_by_construction=True, archive_path="x",
    )
    heavy = CarrierRateTerm(
        archive_bytes=356_642, archive_sha256="x", n_pairs=600, latent_dim=28,
        base_channels=36, rate_term=0.0, cheap_by_construction=True, archive_path="x",
    )
    assert z8_falsification(heavy).z8_over_pr95_byte_ratio < z8_falsification(light).z8_over_pr95_byte_ratio
    assert z8_falsification(light).z8_over_pr95_byte_ratio == pytest.approx(
        2.0 * z8_falsification(heavy).z8_over_pr95_byte_ratio, rel=1e-9
    )


# ---------------------------------------------------------------------------
# Carrier render — DISTINCT real content per pair (the carrier is real).
# ---------------------------------------------------------------------------


@requires_real_carrier
@requires_mlx
def test_carrier_renders_distinct_real_content_per_pair() -> None:
    decoder, latents, rt = load_carrier_decoder(REAL_ARCHIVE)
    assert latents.shape == (rt.n_pairs, rt.latent_dim)
    p0 = render_carrier_pair_bcthw(decoder, latents[0])
    p5 = render_carrier_pair_bcthw(decoder, latents[5])
    # Layout: BTCHW (1,2,3,H,W) in [0,255].
    assert p0.shape[0] == 1 and p0.shape[1] == 2 and p0.shape[2] == 3
    assert float(p0.min()) >= 0.0 and float(p0.max()) <= 255.0001
    # Distinct latents render DISTINCT content (not a constant frame).
    diff = float((p0 - p5).abs().mean().item())
    assert diff > 1.0, f"carrier pairs 0 and 5 too similar ({diff:.3f}) — not real content"
    # The render itself is not uniform (real video has spatial structure).
    assert float(p0.std().item()) > 5.0


@requires_real_carrier
@requires_mlx
def test_carrier_render_responds_to_latent_change() -> None:
    """Perturbing the latent CHANGES the rendered frame (the decoder is real)."""
    decoder, latents, _ = load_carrier_decoder(REAL_ARCHIVE)
    base = render_carrier_pair_bcthw(decoder, latents[0])
    z = latents[0].copy()
    z[0] += 1.0
    perturbed = render_carrier_pair_bcthw(decoder, z)
    assert float((base - perturbed).abs().mean().item()) > 1e-3


# ---------------------------------------------------------------------------
# Fisher-pullback — latent saliency VARIES with pixel saliency.
# ---------------------------------------------------------------------------


@requires_real_carrier
@requires_mlx
def test_fisher_pullback_nonnegative_and_full_rank() -> None:
    decoder, latents, _ = load_carrier_decoder(REAL_ARCHIVE)
    h, w = 384, 512
    sp = np.ones((h, w), dtype=np.float64)  # uniform pixel saliency
    ls = push_pixel_saliency_to_latent(decoder, latents[0], sp, frame_slot=1, eps=1e-2)
    assert ls.s_latent.shape == (28,)
    assert np.all(np.isfinite(ls.s_latent))
    assert np.all(ls.s_latent >= 0.0)
    # The carrier decoder's Jacobian is full-rank w.r.t. the latent => every dim
    # has non-zero pullback energy under a uniform pixel weight.
    assert int((ls.s_latent > 0).sum()) >= 20
    assert ls.method == "central_finite_difference_jacobian_columns"


@requires_real_carrier
@requires_mlx
def test_fisher_pullback_responds_to_pixel_saliency_pattern() -> None:
    """A CONCENTRATED pixel saliency -> a DIFFERENT latent saliency than uniform.

    If push_pixel_saliency_to_latent ignored s_pixel and returned a constant, the
    concentrated and uniform pullbacks would be proportional; they are not.
    """
    decoder, latents, _ = load_carrier_decoder(REAL_ARCHIVE)
    h, w = 384, 512
    uniform = np.ones((h, w), dtype=np.float64)
    concentrated = np.zeros((h, w), dtype=np.float64)
    concentrated[: h // 4, : w // 4] = 1.0  # top-left quadrant only
    ls_u = push_pixel_saliency_to_latent(decoder, latents[0], uniform, frame_slot=1, eps=1e-2)
    ls_c = push_pixel_saliency_to_latent(decoder, latents[0], concentrated, frame_slot=1, eps=1e-2)
    # Normalize each to unit sum, then compare directions.
    u = ls_u.s_latent / (ls_u.s_latent.sum() + 1e-30)
    c = ls_c.s_latent / (ls_c.s_latent.sum() + 1e-30)
    # The saliency SHAPES differ — the pullback genuinely depends on s_pixel.
    assert float(np.abs(u - c).sum()) > 1e-3, "pullback ignores s_pixel pattern"


@requires_real_carrier
@requires_mlx
def test_fisher_pullback_rejects_bad_saliency_shape() -> None:
    decoder, latents, _ = load_carrier_decoder(REAL_ARCHIVE)
    with pytest.raises(Pr95HnervCarrierError):
        push_pixel_saliency_to_latent(
            decoder, latents[0], np.ones((3, 4, 5), dtype=np.float64), frame_slot=1
        )


# ---------------------------------------------------------------------------
# L-inf-vs-L2 latent allocation — DIFFERS from L2, >= L2 rate, oracle != random.
# ---------------------------------------------------------------------------


def test_linf_allocation_differs_from_l2_and_spends_at_least_l2_bits() -> None:
    rng = np.random.default_rng(0)
    latent_values = rng.normal(size=28).astype(np.float64)
    # A NON-uniform saliency (some dims very salient).
    s_latent = np.abs(rng.normal(size=28)) + np.array([100.0] + [0.01] * 27)
    alloc = allocate_latent_linf_vs_l2(s_latent, latent_values, target_bits=28 * 4.0)
    # L-inf forced to spend >= L2 bits (disadvantage_linf anti-gaming guard).
    assert alloc.linf_bits >= alloc.l2_bits - 1e-6
    # The allocations DIFFER (L-inf is not uniform).
    assert alloc.allocations_differ is True
    assert float(np.abs(alloc.linf_steps - alloc.l2_steps).max()) > 1e-9
    # The L-inf step is FINE for the very-salient dim (index 0) and COARSE elsewhere.
    assert alloc.linf_steps[0] < np.median(alloc.linf_steps)


def test_linf_allocation_oracle_differs_from_random_saliency() -> None:
    """Genuine detector-aiming: oracle saliency -> a DIFFERENT allocation than random."""
    rng = np.random.default_rng(1)
    latent_values = rng.normal(size=28).astype(np.float64)
    oracle = np.array([50.0, 40.0] + [0.01] * 26)  # concentrated (real oracle shape)
    random_sal = np.abs(rng.normal(size=28)) + 0.01  # flat-ish random
    a_oracle = allocate_latent_linf_vs_l2(oracle, latent_values, target_bits=28 * 4.0)
    a_random = allocate_latent_linf_vs_l2(random_sal, latent_values, target_bits=28 * 4.0)
    # Different saliency surfaces -> different step maps.
    assert float(np.abs(a_oracle.linf_steps - a_random.linf_steps).max()) > 1e-6


def test_linf_allocation_rejects_mismatched_or_negative() -> None:
    with pytest.raises(Pr95HnervCarrierError):
        allocate_latent_linf_vs_l2(np.ones(28), np.ones(10), target_bits=10.0)
    with pytest.raises(Pr95HnervCarrierError):
        allocate_latent_linf_vs_l2(np.array([-1.0] * 28), np.ones(28), target_bits=10.0)


# ---------------------------------------------------------------------------
# Quantization — actually CHANGES the latents (the decode is real).
# ---------------------------------------------------------------------------


def test_quantize_changes_latents_at_coarse_step() -> None:
    z = np.array([0.13, -0.27, 0.51, 1.02], dtype=np.float64)
    coarse = np.full(4, 0.25, dtype=np.float64)
    q = quantize_latent_with_steps(z, coarse)
    # Mid-rise round-to-step: q must be a multiple of the step and differ from z.
    assert np.allclose(q / coarse, np.round(z / coarse))
    assert float(np.abs(q - z).max()) > 0.0  # quantization actually moved values


def test_quantize_finer_step_is_closer_to_original() -> None:
    z = np.array([0.137, -0.271, 0.513], dtype=np.float64)
    coarse_err = float(np.abs(quantize_latent_with_steps(z, np.full(3, 0.5)) - z).sum())
    fine_err = float(np.abs(quantize_latent_with_steps(z, np.full(3, 0.05)) - z).sum())
    assert fine_err < coarse_err  # finer step => smaller quantization error


def test_quantize_rejects_size_mismatch() -> None:
    with pytest.raises(Pr95HnervCarrierError):
        quantize_latent_with_steps(np.ones(4), np.ones(3))


# ---------------------------------------------------------------------------
# Advisory carrier distortion — REAL carrier vs REAL gt, NON-PROMOTABLE.
# ---------------------------------------------------------------------------


@requires_real_carrier
@requires_mlx
@requires_video
def test_carrier_distortion_advisory_is_finite_and_nonpromotable() -> None:
    from tac.analysis.score_exact_saliency import (
        decode_real_pairs,
        load_score_exact_scorers,
    )

    decoder, latents, _ = load_carrier_decoder(REAL_ARCHIVE)
    gt = decode_real_pairs(str(REAL_VIDEO), 2, pair_stride=64, start_pair=0, device="cpu")
    posenet, segnet = load_score_exact_scorers("upstream", device="cpu")
    dist = measure_carrier_distortion(
        decoder, latents, gt, posenet, segnet, pair_indices=[0, 64]
    )
    assert np.isfinite(dist.d_seg) and np.isfinite(dist.d_pose)
    assert 0.0 <= dist.d_seg <= 1.0  # argmax-flip RATE
    assert dist.d_pose >= 0.0  # pose MSE
    assert dist.advisory_score == pytest.approx(
        100.0 * dist.d_seg + float(np.sqrt(10.0 * dist.d_pose)), rel=1e-9
    )
    assert dist.measure_axis_tag == "[macOS-CPU advisory]"
    assert dist.render_axis_tag == "[macOS-MLX research-signal]"
    assert dist.n_pairs_measured == 2


@requires_real_carrier
@requires_mlx
@requires_video
def test_carrier_distortion_responds_to_wrong_pairing() -> None:
    """Mapping the WRONG carrier latent to a gt pair raises distortion (real metric).

    If d_seg/d_pose were constants, swapping which carrier latent renders against a
    given gt pair could not change the measured distortion. It does.
    """
    from tac.analysis.score_exact_saliency import (
        decode_real_pairs,
        load_score_exact_scorers,
    )

    decoder, latents, _ = load_carrier_decoder(REAL_ARCHIVE)
    gt = decode_real_pairs(str(REAL_VIDEO), 1, pair_stride=1, start_pair=0, device="cpu")
    posenet, segnet = load_score_exact_scorers("upstream", device="cpu")
    correct = measure_carrier_distortion(
        decoder, latents, gt, posenet, segnet, pair_indices=[0]
    )
    wrong = measure_carrier_distortion(
        decoder, latents, gt, posenet, segnet, pair_indices=[400]
    )
    # A mismatched carrier latent vs the same gt pair gives a DIFFERENT distortion.
    assert abs(correct.advisory_score - wrong.advisory_score) > 1e-4


def test_measure_carrier_distortion_rejects_bad_shapes() -> None:
    class _Dummy:
        eval_size = (384, 512)

    with pytest.raises(Pr95HnervCarrierError):
        measure_carrier_distortion(
            _Dummy(), np.zeros((1, 28)), torch.zeros(1, 3, 3, 8, 8), None, None,
            pair_indices=[0],
        )


# ---------------------------------------------------------------------------
# Head-to-head row + NON-PROMOTABLE markers.
# ---------------------------------------------------------------------------


@requires_real_carrier
def test_head_to_head_row_carries_nonpromotable_markers() -> None:
    rt = carrier_rate_term(REAL_ARCHIVE)
    fz = z8_falsification(rt)
    row = build_head_to_head_row(rt, None, fz)
    # The whole point: this row can NEVER be promoted to a score claim.
    assert row["score_claim"] is False
    assert row["promotable"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["axis_tag"] == "[macOS-CPU advisory]"
    assert row["rate_term"] == pytest.approx(rt.rate_term, rel=1e-12)
    assert row["carrier_archive_bytes"] == rt.archive_bytes
    assert row["z8_falsification"]["z8_disease_confirmed"] is True
    assert row["advisory_d_seg"] is None  # render skipped
    assert row.get("advisory_render_skipped") is True


def test_nonpromotable_markers_are_falsey_for_promotion() -> None:
    assert NON_PROMOTABLE_MARKERS["score_claim"] is False
    assert NON_PROMOTABLE_MARKERS["promotable"] is False
    assert NON_PROMOTABLE_MARKERS["axis_tag"] == "[macOS-CPU advisory]"
    assert NON_PROMOTABLE_MARKERS["render_axis_tag"] == "[macOS-MLX research-signal]"


@requires_real_carrier
def test_head_to_head_row_with_latent_allocation_records_objective() -> None:
    rt = carrier_rate_term(REAL_ARCHIVE)
    fz = z8_falsification(rt)
    rng = np.random.default_rng(2)
    latent_values = rng.normal(size=28).astype(np.float64)
    s_latent = np.abs(rng.normal(size=28)) + np.array([100.0] + [0.01] * 27)
    from tac.analysis.pr95_hnerv_linf_carrier import allocate_latent_linf_vs_l2

    alloc = allocate_latent_linf_vs_l2(s_latent, latent_values, target_bits=28 * 4.0)
    row = build_head_to_head_row(rt, None, fz, latent_allocation=alloc)
    assert "latent_linf_vs_l2" in row
    assert row["latent_linf_vs_l2"]["objective"] == "linf_margin_budget_section_7_proven"
    assert row["latent_linf_vs_l2"]["allocations_differ"] is True
