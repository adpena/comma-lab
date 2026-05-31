# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the full-grid SegNet-response cost-map (sister of #1585).

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" Slot EEE Class 2 (tests verify BEHAVIOR not
constants): the headline guard ``test_correction_vs_attack_allocate_differently``
FAILS if the role term ever stops actually inverting the ranking, and
``test_full_grid_response_denser_than_boundary_band`` FAILS if the full-grid response
is not structurally denser than the sparse boundary band (the codex Finding 2 premise).
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.uniward_per_pixel_distortion.full_grid_segnet_response_cost_map import (
    RESPONSE_ROLE_ATTACK,
    RESPONSE_ROLE_CORRECTION,
    VALID_RESPONSE_ROLES,
    FullGridResponseError,
    FullGridSegNetResponseCostMap,
    compose_full_grid_response_cost_map,
    full_grid_response_allocation_diff_proof,
    update_from_anchor,
)


def _rng_texture(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32) * 3.0  # S-UNIWARD-scale non-negative


def _rng_response(shape, seed=1):
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32) * 1e-6  # gradient-scale non-negative


# ── contract / validation ──────────────────────────────────────────────────


def test_valid_roles_frozen():
    assert frozenset({RESPONSE_ROLE_CORRECTION, RESPONSE_ROLE_ATTACK}) == VALID_RESPONSE_ROLES


def test_rejects_unknown_role():
    with pytest.raises(FullGridResponseError, match="role must be one of"):
        compose_full_grid_response_cost_map(
            _rng_texture((1, 4, 4)), _rng_response((1, 4, 4)), role="bogus"
        )


def test_rejects_shape_mismatch():
    with pytest.raises(FullGridResponseError, match="shape mismatch"):
        compose_full_grid_response_cost_map(_rng_texture((1, 4, 4)), _rng_response((1, 5, 5)))


def test_rejects_negative_texture():
    tex = _rng_texture((1, 4, 4))
    tex[0, 0, 0] = -1.0
    with pytest.raises(FullGridResponseError, match="texture_cost must be non-negative"):
        compose_full_grid_response_cost_map(tex, _rng_response((1, 4, 4)))


def test_rejects_negative_response():
    resp = _rng_response((1, 4, 4))
    resp[0, 0, 0] = -1.0
    with pytest.raises(FullGridResponseError, match="segnet_response must be non-negative"):
        compose_full_grid_response_cost_map(_rng_texture((1, 4, 4)), resp)


def test_rejects_nonfinite():
    resp = _rng_response((1, 4, 4))
    resp[0, 1, 1] = np.inf
    with pytest.raises(FullGridResponseError, match="must be finite"):
        compose_full_grid_response_cost_map(_rng_texture((1, 4, 4)), resp)


def test_accepts_2d_hw():
    out = compose_full_grid_response_cost_map(_rng_texture((8, 8)), _rng_response((8, 8)))
    assert out.cost_bhw.shape == (1, 8, 8)


# ── BEHAVIOR (NO-FAKE Class 2) ──────────────────────────────────────────────


def test_correction_boosts_high_response_pixels():
    """correction role: a pixel with high SegNet response should rank ABOVE a pixel
    with equal texture but low response (BEHAVIOR, not a constant)."""
    tex = np.ones((1, 1, 4), dtype=np.float32)  # equal texture
    resp = np.array([[[0.0, 0.0, 1e-6, 1e-6]]], dtype=np.float32)  # rightmost = high resp
    out = compose_full_grid_response_cost_map(tex, resp, role=RESPONSE_ROLE_CORRECTION)
    c = out.cost_bhw[0, 0]
    # high-response pixels (idx 2,3) must outrank low-response (idx 0,1)
    assert c[2] > c[0] and c[3] > c[1]


def test_attack_boosts_low_response_pixels():
    """attack role inverts: low-response pixels rank ABOVE high-response."""
    tex = np.ones((1, 1, 4), dtype=np.float32)
    resp = np.array([[[0.0, 0.0, 1e-6, 1e-6]]], dtype=np.float32)
    out = compose_full_grid_response_cost_map(tex, resp, role=RESPONSE_ROLE_ATTACK)
    c = out.cost_bhw[0, 0]
    assert c[0] > c[2] and c[1] > c[3]


def test_correction_vs_attack_allocate_differently():
    """HEADLINE no-op guard: correction and attack must keep DIFFERENT top-K sets.

    If the role term ever stops inverting the ranking, this FAILS — the gate cannot
    pass on constants alone.
    """
    tex = _rng_texture((1, 16, 16), seed=3)
    resp = _rng_response((1, 16, 16), seed=4)
    corr = compose_full_grid_response_cost_map(tex, resp, role=RESPONSE_ROLE_CORRECTION)
    att = compose_full_grid_response_cost_map(tex, resp, role=RESPONSE_ROLE_ATTACK)
    diff = full_grid_response_allocation_diff_proof(corr.cost_bhw, att.cost_bhw, n_kept=64)
    assert diff["allocation_changed"] is True
    assert diff["kept_set_symmetric_difference"] > 0


def test_detector_differs_from_texture_only():
    """The detector-informed cost-map must allocate differently than texture alone."""
    tex = _rng_texture((1, 16, 16), seed=5)
    resp = _rng_response((1, 16, 16), seed=6)
    det = compose_full_grid_response_cost_map(tex, resp, role=RESPONSE_ROLE_CORRECTION)
    diff = full_grid_response_allocation_diff_proof(det.cost_bhw, tex, n_kept=64)
    assert diff["allocation_changed"] is True


def test_full_grid_response_denser_than_boundary_band():
    """Codex Finding 2 premise: the FULL-GRID response is structurally DENSER than the
    sparse boundary band. A realistic gradient saliency is non-zero almost everywhere;
    a boundary band (exp(-margin/τ)>exp(-1)) is sparse. We assert the module reports a
    high non-zero fraction for a dense response."""
    rng = np.random.default_rng(7)
    # dense response: only a tiny fraction is exactly zero
    resp = rng.random((1, 32, 32), dtype=np.float32) * 1e-6
    resp[resp < 1e-9] = 0.0  # essentially none
    tex = _rng_texture((1, 32, 32), seed=8)
    out = compose_full_grid_response_cost_map(tex, resp, role=RESPONSE_ROLE_CORRECTION)
    assert out.response_nonzero_fraction > 0.5  # full-grid is dense (codex Finding 2)


def test_floor_keeps_reweight_not_mask():
    """Zero-response pixels must NOT be annihilated (floor keeps it a strict reweight)."""
    tex = np.ones((1, 1, 3), dtype=np.float32)
    resp = np.zeros((1, 1, 3), dtype=np.float32)
    out = compose_full_grid_response_cost_map(tex, resp, role=RESPONSE_ROLE_CORRECTION)
    assert np.all(out.cost_bhw > 0.0)  # floored, not masked


# ── stats / provenance ──────────────────────────────────────────────────────


def test_stats_dataclass_frozen_and_dict():
    out = compose_full_grid_response_cost_map(_rng_texture((1, 8, 8)), _rng_response((1, 8, 8)))
    assert isinstance(out, FullGridSegNetResponseCostMap)
    d = out.as_dict()
    assert d["axis_tag"] == "[macOS-CPU advisory]"
    assert d["score_claim"] is False
    assert d["promotable"] is False
    assert d["schema"] == "full_grid_segnet_response_cost_map_v1"
    assert 0.0 <= d["response_gini"] <= 1.0
    assert 0.0 <= d["cost_gini"] <= 1.0


def test_update_from_anchor_observability_only():
    out = update_from_anchor({"role": "correction"})
    assert out["consumed"] is True
    assert out["score_claim"] is False
    assert out["promotable"] is False
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["role"] == "correction"


def test_allocation_diff_proof_rejects_negative_n_kept():
    with pytest.raises(FullGridResponseError, match="n_kept must be non-negative"):
        full_grid_response_allocation_diff_proof(
            _rng_texture((1, 4, 4)), _rng_response((1, 4, 4)), n_kept=-1
        )


def test_allocation_diff_proof_shape_mismatch():
    with pytest.raises(FullGridResponseError, match="shape mismatch"):
        full_grid_response_allocation_diff_proof(
            _rng_texture((1, 4, 4)), _rng_response((1, 5, 5)), n_kept=4
        )


def test_identical_maps_no_allocation_change():
    """Sanity: identical cost maps allocate identically (symdiff == 0)."""
    tex = _rng_texture((1, 8, 8), seed=9)
    diff = full_grid_response_allocation_diff_proof(tex, tex, n_kept=16)
    assert diff["kept_set_symmetric_difference"] == 0
    assert diff["allocation_changed"] is False
