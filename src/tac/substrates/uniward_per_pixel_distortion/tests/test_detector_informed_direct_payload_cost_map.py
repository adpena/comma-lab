# SPDX-License-Identifier: MIT
"""NO-FAKE behavioral tests for the detector-informed direct-payload cost-map.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" + Catalog #307 Class 2 (tests verify
BEHAVIOR not constants): every assertion checks that the composition actually
changes the ranking / the kept-index set, NOT that a metadata field equals a
literal. The headline guard (``test_detector_informed_changes_allocation_vs_*``)
FAILS if the cost-map degenerates to a no-op.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tac.substrates.uniward_per_pixel_distortion.detector_informed_direct_payload_cost_map import (
    SIDECAR_ROLE_ATTACK,
    SIDECAR_ROLE_CORRECTION,
    DetectorCostMapError,
    allocation_diff_proof,
    compose_detector_informed_cost_map,
    update_from_anchor,
)


def _synthetic_texture_and_boundary(seed: int = 7, B: int = 2, H: int = 24, W: int = 32):
    """Real-shaped synthetic texture + boundary maps (deterministic).

    texture: random non-negative (stand-in for S-UNIWARD on real frames; the
    smoke uses the REAL kernel — these tests exercise the COMPOSITION math).
    boundary: a thin band of high weight (exp(-margin/τ) ≈ 1) on a vertical seam,
    low elsewhere — i.e. a realistic SegNet decision-boundary band.
    """
    rng = np.random.default_rng(seed)
    texture = rng.random((B, H, W), dtype=np.float32) * 4.0  # S-UNIWARD is unnormalized
    boundary = np.full((B, H, W), 0.02, dtype=np.float32)  # confident interior
    seam = W // 2
    boundary[:, :, seam - 1 : seam + 2] = 0.95  # the decision-boundary band
    return texture, boundary


# ---------------------------------------------------------------------------
# Headline NO-FAKE guard: the detector-informed map CHANGES the allocation.
# ---------------------------------------------------------------------------


def test_detector_informed_changes_allocation_vs_uniform() -> None:
    texture, boundary = _synthetic_texture_and_boundary()
    correction = compose_detector_informed_cost_map(
        texture, boundary, role=SIDECAR_ROLE_CORRECTION
    )
    uniform = np.ones_like(texture)  # flat cost-map = rank by |δ| only
    n_kept = int(0.05 * texture.size)
    proof = allocation_diff_proof(correction.cost_bhw, uniform, n_kept=n_kept)
    # The whole point: detector-informed ranking keeps a DIFFERENT set than uniform.
    assert proof["allocation_changed"], proof
    assert proof["kept_set_symmetric_difference"] > 0


def test_detector_informed_changes_allocation_vs_texture_only() -> None:
    texture, boundary = _synthetic_texture_and_boundary()
    detector = compose_detector_informed_cost_map(
        texture, boundary, role=SIDECAR_ROLE_CORRECTION
    )
    # texture-only is the prior-negative form: rank by S-UNIWARD texture alone.
    n_kept = int(0.05 * texture.size)
    proof = allocation_diff_proof(detector.cost_bhw, texture, n_kept=n_kept)
    assert proof["allocation_changed"], proof
    assert proof["kept_set_symmetric_difference"] > 0


def test_correction_role_boosts_boundary_band() -> None:
    """correction role must rank boundary-band pixels HIGHER than texture-only."""
    texture, boundary = _synthetic_texture_and_boundary()
    correction = compose_detector_informed_cost_map(
        texture, boundary, role=SIDECAR_ROLE_CORRECTION
    )
    band = boundary > math.exp(-1.0)
    interior = ~band
    # In the correction map, the boundary band's mean cost (per unit texture) is
    # boosted relative to the texture-only baseline. Compare cost/texture ratio.
    tex_max = np.maximum(texture.reshape(texture.shape[0], -1).max(axis=1), 1e-8)
    tex_norm = texture / tex_max[:, None, None]
    band_ratio = (correction.cost_bhw[band] / np.maximum(tex_norm[band], 1e-8)).mean()
    interior_ratio = (
        correction.cost_bhw[interior] / np.maximum(tex_norm[interior], 1e-8)
    ).mean()
    assert band_ratio > interior_ratio, (band_ratio, interior_ratio)


def test_attack_role_inverts_boundary_preference() -> None:
    """attack role must rank textured NON-boundary interiors higher (opposite)."""
    texture, boundary = _synthetic_texture_and_boundary()
    attack = compose_detector_informed_cost_map(
        texture, boundary, role=SIDECAR_ROLE_ATTACK
    )
    band = boundary > math.exp(-1.0)
    interior = ~band
    tex_max = np.maximum(texture.reshape(texture.shape[0], -1).max(axis=1), 1e-8)
    tex_norm = texture / tex_max[:, None, None]
    band_ratio = (attack.cost_bhw[band] / np.maximum(tex_norm[band], 1e-8)).mean()
    interior_ratio = (
        attack.cost_bhw[interior] / np.maximum(tex_norm[interior], 1e-8)
    ).mean()
    # attack protects the boundary => interior ranks higher than band.
    assert interior_ratio > band_ratio, (band_ratio, interior_ratio)


def test_correction_and_attack_allocate_differently() -> None:
    texture, boundary = _synthetic_texture_and_boundary()
    correction = compose_detector_informed_cost_map(
        texture, boundary, role=SIDECAR_ROLE_CORRECTION
    )
    attack = compose_detector_informed_cost_map(
        texture, boundary, role=SIDECAR_ROLE_ATTACK
    )
    n_kept = int(0.05 * texture.size)
    proof = allocation_diff_proof(correction.cost_bhw, attack.cost_bhw, n_kept=n_kept)
    assert proof["allocation_changed"], proof


# ---------------------------------------------------------------------------
# Determinism + shape + provenance markers.
# ---------------------------------------------------------------------------


def test_deterministic_for_identical_inputs() -> None:
    texture, boundary = _synthetic_texture_and_boundary()
    a = compose_detector_informed_cost_map(texture, boundary)
    b = compose_detector_informed_cost_map(texture, boundary)
    np.testing.assert_array_equal(a.cost_bhw, b.cost_bhw)


def test_cost_map_shape_matches_inputs() -> None:
    texture, boundary = _synthetic_texture_and_boundary(B=3, H=16, W=20)
    cm = compose_detector_informed_cost_map(texture, boundary)
    assert cm.cost_bhw.shape == (3, 16, 20)
    assert cm.cost_bhw.dtype == np.float32


def test_2d_inputs_are_promoted_to_bhw() -> None:
    rng = np.random.default_rng(1)
    texture = rng.random((10, 12), dtype=np.float32) * 3.0
    boundary = np.full((10, 12), 0.5, dtype=np.float32)
    cm = compose_detector_informed_cost_map(texture, boundary)
    assert cm.cost_bhw.shape == (1, 10, 12)


def test_as_dict_carries_nonpromotable_markers() -> None:
    texture, boundary = _synthetic_texture_and_boundary()
    cm = compose_detector_informed_cost_map(texture, boundary)
    d = cm.as_dict()
    assert d["score_claim"] is False
    assert d["promotable"] is False
    assert d["axis_tag"] == "[macOS-CPU advisory]"
    assert 0.0 <= d["boundary_band_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# Input validation (fail-closed).
# ---------------------------------------------------------------------------


def test_bad_role_raises() -> None:
    texture, boundary = _synthetic_texture_and_boundary()
    with pytest.raises(DetectorCostMapError):
        compose_detector_informed_cost_map(texture, boundary, role="nonsense")


def test_shape_mismatch_raises() -> None:
    texture, _ = _synthetic_texture_and_boundary(H=10, W=10)
    _, boundary = _synthetic_texture_and_boundary(H=12, W=12)
    with pytest.raises(DetectorCostMapError):
        compose_detector_informed_cost_map(texture, boundary)


def test_negative_texture_raises() -> None:
    texture, boundary = _synthetic_texture_and_boundary()
    texture = texture.copy()
    texture[0, 0, 0] = -1.0
    with pytest.raises(DetectorCostMapError):
        compose_detector_informed_cost_map(texture, boundary)


def test_boundary_out_of_range_raises() -> None:
    texture, boundary = _synthetic_texture_and_boundary()
    boundary = boundary.copy()
    boundary[0, 0, 0] = 1.5  # exp(-margin/τ) can never exceed 1
    with pytest.raises(DetectorCostMapError):
        compose_detector_informed_cost_map(texture, boundary)


def test_nonfinite_input_raises() -> None:
    texture, boundary = _synthetic_texture_and_boundary()
    texture = texture.copy()
    texture[0, 1, 1] = np.inf
    with pytest.raises(DetectorCostMapError):
        compose_detector_informed_cost_map(texture, boundary)


def test_allocation_diff_n_kept_zero_is_empty() -> None:
    texture, boundary = _synthetic_texture_and_boundary()
    cm = compose_detector_informed_cost_map(texture, boundary)
    proof = allocation_diff_proof(cm.cost_bhw, np.ones_like(texture), n_kept=0)
    assert proof["kept_set_symmetric_difference"] == 0
    assert proof["allocation_changed"] is False


def test_allocation_diff_shape_mismatch_raises() -> None:
    texture, boundary = _synthetic_texture_and_boundary(H=10, W=10)
    cm = compose_detector_informed_cost_map(texture, boundary)
    other = np.ones((1, 12, 12), dtype=np.float32)
    with pytest.raises(DetectorCostMapError):
        allocation_diff_proof(cm.cost_bhw, other, n_kept=5)


# ---------------------------------------------------------------------------
# Direct-payload byte-closure integration: the cost-map feeds pack_sparse_delta,
# and a DIFFERENT cost-map yields a DIFFERENT kept set on a REAL byte budget.
# ---------------------------------------------------------------------------


def test_cost_map_changes_pack_sparse_delta_kept_set() -> None:
    """End-to-end byte-closure NO-FAKE: pack the SAME δ with detector-informed vs
    uniform cost-maps at the SAME target_bytes → the recovered δ kept positions
    differ. This proves the cost-map is consumed by the real wire format."""
    import torch

    from tac.uniward_delta import pack_sparse_delta, unpack_sparse_delta

    rng = np.random.default_rng(11)
    B, H, W = 2, 24, 32
    texture, boundary = _synthetic_texture_and_boundary(seed=11, B=B, H=H, W=W)
    # Real δ residual (random but fixed): (B, 3, H, W) in [0, 255] units.
    delta = torch.from_numpy(
        rng.standard_normal((B, 3, H, W)).astype(np.float32) * 6.0
    )
    correction = compose_detector_informed_cost_map(
        texture, boundary, role=SIDECAR_ROLE_CORRECTION
    )
    uniform_cost = torch.ones((B, H, W), dtype=torch.float32)
    detector_cost = torch.from_numpy(correction.cost_bhw)

    target_bytes = 600
    blob_det = pack_sparse_delta(
        delta, detector_cost, l_inf_budget=8.0, target_bytes=target_bytes
    )
    blob_uni = pack_sparse_delta(
        delta, uniform_cost, l_inf_budget=8.0, target_bytes=target_bytes
    )
    spec_det = unpack_sparse_delta(blob_det)
    spec_uni = unpack_sparse_delta(blob_uni)

    # Gather kept global positions from each spec's per-frame local indices.
    def _global_positions(spec) -> set[int]:
        pos: set[int] = set()
        for f in range(spec.n_frames):
            li = spec.per_frame_local_idx[f]
            if li is None:
                continue
            base = f * (spec.H * spec.W * 3)
            pos.update(int(base + int(v)) for v in li.tolist())
        return pos

    det_pos = _global_positions(spec_det)
    uni_pos = _global_positions(spec_uni)
    # Both fit roughly the same budget; the cost-map must move the kept set.
    assert det_pos != uni_pos, "cost-map had no effect on the real wire kept set"
    assert len(det_pos ^ uni_pos) > 0


def test_update_from_anchor_is_nonpromotable() -> None:
    out = update_from_anchor({"role": "correction", "byte_delta": -123})
    assert out["score_claim"] is False
    assert out["promotable"] is False
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["role"] == "correction"
