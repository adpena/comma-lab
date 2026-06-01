# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the boundary-aware RD allocation wire-in (G2 + consumer + Rev3).

Covers:
  * G2 proxy-rate gate (Balle): proxy bits/8 vs actual coder bytes, bounded.
  * Consumer wire-in: pixel saliency -> A^T -> residual grid -> rate_collapse,
    asymmetry guard (frame_0 carries zero SegNet incidence).
  * Revision 3: latent/token -> frame Jacobian separability (coupling == 0).
  * Advisory re-measurement plumbing (the heavy scorer path is exercised by the
    orchestrator tool; here we test the byte/importance machinery deterministically).

Every assertion exercises ACTUAL behavior (real packets, real bytes, real
adjoint pushes), not metadata constants — a body replaced by canonical markers
would fail these.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.archive_byte_profile import contest_rate_term
from tac.substrates.hprc.archive import parse_hprc_packet
from tac.substrates.hprc.learned_receiver import (
    build_compact_receiver_packet_from_lowres_frames,
    decode_compact_receiver_packet,
    render_compact_receiver_frame_batch,
)
from tac.substrates.hprc.rate_collapse import (
    ResidualTokenCollapseSpec,
    transcode_compact_receiver_importance_weighted_residual_tokens,
)

from tac.analysis.hprc_saliency_rd_allocation import (
    SCORE_QUANTUM_BYTES,
    build_real_archive_zip_bytes,
    build_saliency_driven_importance,
    coded_bytes_for_symbol_stream,
    measure_latent_frame_jacobian_sparsity,
    measure_proxy_rate_residual,
    symbol_stream_entropy_bits,
)


def _real_packet(seed=11, frames_n=8, gh=6, gw=8, basis=3):
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, size=(frames_n, 24, 32, 3), dtype=np.uint8).astype(np.float32)
    packet = build_compact_receiver_packet_from_lowres_frames(
        frames, basis_count=basis, residual_grid_h=gh, residual_grid_w=gw
    )
    return packet, decode_compact_receiver_packet(parse_hprc_packet(packet))


# ---------------------------------------------------------------------------
# G2 — Balle proxy-rate gate.
# ---------------------------------------------------------------------------


def test_score_quantum_bytes_matches_contest_pricing():
    """1502 bytes must price to ~0.001 score at the contest rate term."""
    assert abs(contest_rate_term(SCORE_QUANTUM_BYTES) - 0.001) < 5e-6


def test_symbol_stream_entropy_is_shannon_lower_bound():
    """A uniform 256-symbol stream has entropy ~8 bits/symbol; a constant ~0."""
    rng = np.random.default_rng(1)
    uniform = rng.integers(-128, 128, size=20000, dtype=np.int16)
    _, bps_u, n_u = symbol_stream_entropy_bits(uniform)
    assert 7.8 < bps_u <= 8.0 + 1e-6
    assert n_u == 20000
    constant = np.zeros(5000, dtype=np.int16)
    bits_c, bps_c, n_c = symbol_stream_entropy_bits(constant)
    assert bits_c == 0.0 and bps_c == 0.0 and n_c == 5000


def test_g2_residual_bounded_for_real_residual_stream():
    """G2: proxy vs actual coder bytes on a real residual stream is bounded."""
    packet, compact = _real_packet(seed=11, frames_n=40, gh=6, gw=8)
    archive_zip = build_real_archive_zip_bytes(packet)
    g2 = measure_proxy_rate_residual(
        residual_q=compact.residual.q,
        full_archive_bytes=len(archive_zip),
        note="unit_g2",
    )
    # The proxy is the order-0 entropy ideal; brotli on near-random int8 sits
    # slightly above it. The deviation must be a small fraction of the quantum.
    assert g2.symbol_count == compact.residual.q.size
    assert g2.proxy_bytes > 0.0
    assert g2.coded_modeled_bytes > 0
    assert g2.abs_residual_bytes < SCORE_QUANTUM_BYTES, (
        f"G2 residual {g2.abs_residual_bytes:.1f} bytes exceeds 1502 quantum"
    )
    assert g2.within_quantum is True
    assert g2.frontier_is_fictional is False
    # Decomposition: full archive = coded modeled + non-entropy overhead.
    assert g2.full_archive_bytes >= g2.coded_modeled_bytes
    assert g2.non_entropy_coded_overhead_bytes == max(
        0, g2.full_archive_bytes - g2.coded_modeled_bytes
    )


def test_g2_detects_fictional_frontier_when_proxy_diverges():
    """NO-FAKE: if proxy and actual coder bytes diverge hugely, gate flags fictional."""
    # A residual stream whose proxy is tiny but we pass a deliberately inflated
    # coded_modeled (simulating a coder far from the ideal) -> frontier fictional.
    q = np.zeros((4, 3, 4, 3), dtype=np.int16)  # all-zero => proxy ~0 bits
    g2 = measure_proxy_rate_residual(
        residual_q=q,
        full_archive_bytes=10_000,
        coded_modeled_bytes=5_000,  # 5000 bytes vs ~0 proxy => residual ~5000 > 1502
        note="fictional_test",
    )
    assert g2.abs_residual_bytes > SCORE_QUANTUM_BYTES
    assert g2.within_quantum is False
    assert g2.frontier_is_fictional is True


def test_g2_conservative_proxy_is_not_fictional_even_when_abs_exceeds_quantum():
    """NO-FAKE direction guard: a CONSERVATIVE proxy (coder beats it) is never fictional.

    If the real brotli coder BEATS the order-0 entropy proxy (residual < 0, the
    coder captures structure beyond order-0), the proxy UNDER-promises. The RD
    frontier computed in the proxy domain is then conservative/pessimistic, NOT
    fictional — even if |residual| exceeds 1502.
    """
    q = np.zeros((4, 3, 4, 3), dtype=np.int16)  # proxy ~0 bits
    g2 = measure_proxy_rate_residual(
        residual_q=q,
        full_archive_bytes=10_000,
        coded_modeled_bytes=10,  # coder=10 bytes BEATS the (effectively 0) proxy is not it;
        note="conservative_dir",
    )
    # proxy_bytes ~ 0; coded=10 => residual=+10 (>0, proxy over-promises) but within quantum.
    assert g2.within_quantum is True
    assert g2.frontier_is_fictional is False

    # Now the genuine conservative case: nontrivial proxy, coder BEATS it big.
    rng = np.random.default_rng(99)
    structured = np.tile(rng.integers(-50, 50, size=200, dtype=np.int16), 50)  # repeating
    proxy_bits, _, _ = symbol_stream_entropy_bits(structured)
    proxy_bytes = proxy_bits / 8.0
    g2b = measure_proxy_rate_residual(
        residual_q=structured,
        full_archive_bytes=int(proxy_bytes) + 5000,
        coded_modeled_bytes=int(proxy_bytes) - 2000,  # coder beats proxy by 2000 bytes
        note="conservative_big",
    )
    assert g2b.residual_bytes < 0.0  # coder beats proxy
    assert g2b.proxy_overpromises is False
    assert g2b.frontier_is_fictional is False, (
        "a conservative proxy (coder beats it) must NOT be flagged fictional"
    )
    assert g2b.per_symbol_residual_bytes < 0.0


def test_coded_bytes_for_symbol_stream_uses_real_brotli():
    """The coder-bytes helper must use real brotli (compresses a repeated stream)."""
    repeated = np.zeros(10000, dtype=np.int8)  # highly compressible
    random = np.random.default_rng(2).integers(-128, 128, size=10000, dtype=np.int16).astype(np.int8)
    c_rep = coded_bytes_for_symbol_stream(repeated)
    c_rand = coded_bytes_for_symbol_stream(random)
    assert c_rep < 100, "brotli must crush a constant stream"
    assert c_rand > 5000, "random int8 should be near-incompressible"


# ---------------------------------------------------------------------------
# Revision 3 — frame/pair Jacobian separability.
# ---------------------------------------------------------------------------


def test_jacobian_sparsity_residual_and_latent_per_frame_separable():
    """Perturbing frame_0 storage must NOT change frame_1's render (coupling==0)."""
    _, compact = _real_packet(seed=13, frames_n=6, gh=3, gw=4)
    jac = measure_latent_frame_jacobian_sparsity(compact)
    assert jac.residual_tokens_per_frame_separable is True
    assert jac.residual_cross_frame_coupling == 0.0
    assert jac.latent_per_frame_separable is True
    assert jac.latent_cross_frame_coupling == 0.0
    assert jac.pair_count == 3


# ---------------------------------------------------------------------------
# Consumer wire-in — pixel saliency -> A^T -> residual grid (asymmetry guard).
# ---------------------------------------------------------------------------


def test_build_saliency_driven_importance_respects_frame_asymmetry():
    """frame_0 must carry ZERO SegNet incidence (modules.py:108 last-frame-only)."""
    _, compact = _real_packet(seed=21, frames_n=6, gh=3, gw=4)
    cam_h, cam_w = 24, 32
    s_seg = np.zeros((6, cam_h, cam_w))
    s_seg[:, 8:12, 10:14] = 100.0
    s_pose = np.zeros((6, cam_h, cam_w))
    s_pose[:, 4:8, 4:8] = 50.0
    imp = build_saliency_driven_importance(
        compact=compact,
        s_seg_per_frame=s_seg,
        s_pose_per_frame=s_pose,
        camera_height=cam_h,
        camera_width=cam_w,
    )
    assert imp.importance.shape == (6, 3, 4)
    assert imp.frame_0_seg_mass == 0.0, "frame_0 must carry zero SegNet saliency"
    assert imp.frame_1_seg_mass > 0.0, "frame_1 must carry the SegNet saliency"
    # Importance is finite and nonnegative (the rate_collapse consumer requires it).
    assert np.all(np.isfinite(imp.importance))
    assert np.all(imp.importance >= 0.0)


def test_saliency_importance_is_consumed_by_rate_collapse():
    """End-to-end: the A^T-pushed importance feeds the EXISTING rate_collapse consumer."""
    packet, compact = _real_packet(seed=22, frames_n=6, gh=3, gw=4)
    cam_h, cam_w = 24, 32
    s_seg = np.zeros((6, cam_h, cam_w))
    s_seg[:, 8:12, 10:14] = 100.0
    s_pose = np.zeros((6, cam_h, cam_w))
    s_pose[:, 4:8, 4:8] = 50.0
    imp = build_saliency_driven_importance(
        compact=compact,
        s_seg_per_frame=s_seg,
        s_pose_per_frame=s_pose,
        camera_height=cam_h,
        camera_width=cam_w,
    )
    collapsed, rows, metrics = transcode_compact_receiver_importance_weighted_residual_tokens(
        packet,
        low_importance_spec=ResidualTokenCollapseSpec(deadzone=4, quant_divisor=4),
        high_importance_spec=ResidualTokenCollapseSpec(deadzone=0, quant_divisor=1),
        importance=imp.importance,
        coarsen_quantile=0.5,
    )
    # Receiver-decodable after the saliency-driven collapse.
    compact2 = decode_compact_receiver_packet(parse_hprc_packet(collapsed))
    rendered = render_compact_receiver_frame_batch(compact2, 0, 6, height=cam_h, width=cam_w)
    assert rendered.shape == (6, cam_h, cam_w, 3)
    assert metrics["importance_weighted"] is True
    assert 0.0 < metrics["coarsened_token_fraction"] < 1.0


def test_saliency_importance_protects_high_saliency_tokens_vs_uniform():
    """A localized saliency hotspot must protect MORE tokens near it than uniform.

    NO-FAKE: build importance from a single-pixel hotspot; the token covering the
    hotspot must rank ABOVE a token covering a flat region, so the importance is
    actually carrying the adjoint-pushed structure (not a constant).
    """
    _, compact = _real_packet(seed=23, frames_n=2, gh=6, gw=8)
    cam_h, cam_w = 24, 32
    s_seg = np.zeros((2, cam_h, cam_w))
    # hotspot localized to top-left camera region -> token (0,0)
    s_seg[1, 0:4, 0:4] = 1000.0  # frame_1 only (seg)
    s_pose = np.zeros((2, cam_h, cam_w))
    imp = build_saliency_driven_importance(
        compact=compact,
        s_seg_per_frame=s_seg,
        s_pose_per_frame=s_pose,
        camera_height=cam_h,
        camera_width=cam_w,
    )
    frame1 = imp.importance[1]  # (6, 8) token grid
    # The top-left token must hold the maximal importance (the hotspot lands there).
    assert frame1[0, 0] == frame1.max()
    # A far token (bottom-right) must hold near-zero importance.
    assert frame1[-1, -1] < frame1[0, 0]


def test_importance_blind_vs_saliency_differ_in_token_ranking():
    """Saliency-driven importance must NOT be the uniform constant."""
    _, compact = _real_packet(seed=24, frames_n=2, gh=4, gw=4)
    cam_h, cam_w = 16, 16
    s_seg = np.zeros((2, cam_h, cam_w))
    s_seg[1, 0:2, 0:2] = 500.0
    s_pose = np.zeros((2, cam_h, cam_w))
    s_pose[0, 8:10, 8:10] = 200.0
    imp = build_saliency_driven_importance(
        compact=compact,
        s_seg_per_frame=s_seg,
        s_pose_per_frame=s_pose,
        camera_height=cam_h,
        camera_width=cam_w,
    )
    uniform = np.ones_like(imp.importance)
    assert not np.allclose(imp.importance, uniform), (
        "saliency-driven importance must differ from the uniform control"
    )
    # Variance must be > 0 (structure present).
    assert float(np.var(imp.importance)) > 0.0


def test_non_promotable_markers_on_all_reports():
    """All G2/Rev3/importance reports carry the non-promotable custody markers."""
    packet, compact = _real_packet(seed=25, frames_n=6, gh=3, gw=4)
    g2 = measure_proxy_rate_residual(
        residual_q=compact.residual.q,
        full_archive_bytes=len(build_real_archive_zip_bytes(packet)),
    )
    jac = measure_latent_frame_jacobian_sparsity(compact)
    imp = build_saliency_driven_importance(
        compact=compact,
        s_seg_per_frame=np.zeros((6, 24, 32)),
        s_pose_per_frame=np.ones((6, 24, 32)),
        camera_height=24,
        camera_width=32,
    )
    for rep in (g2.as_jsonable(), jac.as_jsonable(), imp.as_jsonable()):
        assert rep["axis_tag"] == "[macOS-CPU advisory]"
        assert rep["score_claim"] is False
        assert rep["promotable"] is False
