# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the #386 crucible-3 N-1 reformulation head-offset modes: the
flip-weighted OT arm (``flip_weighted``) and S1's Hamming-optimal per-edge median
(``flip_median`` / :func:`flip_median_offsets`).

These pin the REAL properties: the median solver actually computes per-edge flip-margin
medians and reconciles them zero-sum; both modes RAISE without gt (never silently
area-match — the N-1-falsified objective); ``flip_weighted`` really targets the flip
share (not GT area); and both fold byte-free into ``out_sdf.bias``. The through-R
d_seg VERDICT is the n600 gate's job — these tests pin MECHANISM, not score."""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.laguerre_logit_offset import (
    HEAD_OFFSET_SOLVERS,
    LaguerreLogitOffsetError,
    _flip_share_by_class,
    apply_offset_to_sdf_bias,
    flip_median_offsets,
    power_diagram_argmax,
    solve_head_offsets,
)


def _erasure_phi(seed: int = 1, n: int = 20000, k: int = 5, erase_class: int = 1, erase: float = 0.5):
    """A REALISTIC field: GT = argmax(clean); the witness systematically UNDER-predicts ``erase_class``
    by ``erase`` (near-boundary erasure, small flip margins — the Lane long-tail shape). Returns
    ``(phi, gt)``."""
    rng = np.random.default_rng(seed)
    clean = rng.standard_normal((n, k))
    clean[:, 0] += 0.8
    clean[:, 2] += 0.8  # Road / Undrivable dominate (imbalance)
    gt = np.argmax(clean, axis=1)
    phi = clean.copy()
    phi[:, erase_class] -= erase  # the witness erases erase_class at the margin
    return phi, gt


# ---- the new solver set --------------------------------------------------
def test_head_offset_solvers_includes_flip_modes():
    assert set(HEAD_OFFSET_SOLVERS) == {"menon", "ot_newton", "flip_weighted", "flip_median"}


# ---- flip_share helper (delegates to the canonical perclass_verdict sensor) ----
def test_flip_share_sums_to_one_and_targets_erased_class():
    phi, gt = _erasure_phi(erase_class=1, erase=0.6)
    pred = np.argmax(phi, axis=1)
    share = _flip_share_by_class(pred, gt, 5)
    assert share.shape == (5,)
    assert abs(float(share.sum()) - 1.0) < 1e-12
    # the erased class (1) must carry the largest flip share (it is the systematically dropped class)
    assert int(np.argmax(share)) == 1


def test_flip_share_raises_when_no_flips():
    rng = np.random.default_rng(0)
    phi = rng.standard_normal((100, 5))
    gt = np.argmax(phi, axis=1)  # witness == gt => zero flips
    with pytest.raises(LaguerreLogitOffsetError):
        _flip_share_by_class(np.argmax(phi, axis=1), gt, 5)


# ---- flip_median: the genuine per-edge median solve ----------------------
def test_flip_median_is_zero_sum():
    phi, gt = _erasure_phi()
    b, info = flip_median_offsets(phi, gt)
    assert abs(float(b.sum())) < 1e-9
    assert info["converged"] == 1.0
    assert info["n_edges_used"] >= 1.0
    assert info["total_flips"] > 0.0
    # mass is NOT the objective — the info flag says so with a NaN
    assert np.isnan(info["max_mass_err"])


def test_flip_median_boosts_the_erased_class():
    """The systematically-erased class (1) must receive a POSITIVE offset (its argmax cell enlarged)
    relative to the dominant Road class (0) — the erasure-correction direction."""
    phi, gt = _erasure_phi(erase_class=1, erase=0.5)
    b, _ = flip_median_offsets(phi, gt)
    assert b[1] > b[0]


def test_flip_median_offset_equals_negative_edge_median():
    """DIRECT NO-FAKE check that a per-edge offset difference IS the negative flip-margin median: a
    2-class field where only the {0,1} edge has flips => b0 - b1 == -median(phi0-phi1 over flips)."""
    rng = np.random.default_rng(3)
    n = 5000
    phi = rng.standard_normal((n, 2))
    gt = np.argmax(phi, axis=1)
    # erase class 1: flip a band of GT=1 pixels to predicted-0 by depressing phi1 on them
    phi[:, 1] -= 0.4
    pred0 = np.argmax(phi, axis=1)
    flip = (gt == 1) & (pred0 == 0)
    assert flip.sum() > 10
    m = phi[flip, 0] - phi[flip, 1]
    expected_delta = -float(np.median(m))  # b0 - b1
    b, _ = flip_median_offsets(phi, gt)
    assert abs((b[0] - b[1]) - expected_delta) < 1e-9


def test_flip_median_no_flips_returns_zero():
    rng = np.random.default_rng(4)
    phi = rng.standard_normal((200, 5))
    gt = np.argmax(phi, axis=1)  # perfect witness => no flips => no offset
    b, info = flip_median_offsets(phi, gt)
    assert np.allclose(b, 0.0)
    assert info["n_edges_used"] == 0.0
    assert info["total_flips"] == 0.0


def test_flip_median_helps_on_realistic_erasure():
    """On boundary-concentrated erasure (small flip margins), the median placement should REDUCE the
    phi-space d_seg (the through-R re-confirmation is the n600 gate's job; this pins the mechanism is
    directionally sane, not a proxy score claim)."""
    phi, gt = _erasure_phi(erase_class=1, erase=0.5)
    d0 = float(np.mean(np.argmax(phi, axis=1) != gt))
    b, _ = flip_median_offsets(phi, gt)
    d1 = float(np.mean(power_diagram_argmax(phi, b).reshape(-1) != gt))
    assert d1 < d0


def test_flip_median_rejects_bad_gt():
    phi, gt = _erasure_phi()
    with pytest.raises(LaguerreLogitOffsetError):
        flip_median_offsets(phi, gt[:-10])  # size mismatch
    with pytest.raises(LaguerreLogitOffsetError):
        flip_median_offsets(phi, np.full(gt.shape, 99))  # label out of range


# ---- dispatcher wiring ---------------------------------------------------
def test_dispatcher_flip_median_matches_direct():
    phi, gt = _erasure_phi()
    b_direct, _ = flip_median_offsets(phi, gt)
    b_disp, info = solve_head_offsets("flip_median", phi=phi, gt=gt)
    assert np.array_equal(b_disp, b_direct)
    assert info["solver"] == 3.0


def test_dispatcher_flip_weighted_targets_flip_share_not_area():
    """flip_weighted runs the OT solve against the FLIP SHARE, so its offsets DIFFER from ot_newton's
    (which targets GT AREA). Both are real OT solves on the same phi; the TARGET is what changes."""
    phi, gt = _erasure_phi(erase_class=1, erase=0.6)
    counts = np.bincount(gt, minlength=5).astype(np.float64)
    b_fw, info_fw = solve_head_offsets("flip_weighted", phi=phi, gt=gt, tau=1.0)
    b_area, _ = solve_head_offsets("ot_newton", phi=phi, target_masses=counts, tau=1.0)
    assert info_fw["solver"] == 2.0
    assert info_fw["target_is_flip_share"] == 1.0
    assert not np.allclose(b_fw, b_area, atol=1e-3)
    assert abs(float(b_fw.sum())) < 1e-8  # zero-sum


def test_dispatcher_flip_modes_require_gt_no_fake():
    """flip_weighted / flip_median with NO gt must RAISE — they never silently area-match (which would
    re-inherit N-1's falsified objective)."""
    phi, _ = _erasure_phi()
    with pytest.raises(LaguerreLogitOffsetError):
        solve_head_offsets("flip_weighted", phi=phi)
    with pytest.raises(LaguerreLogitOffsetError):
        solve_head_offsets("flip_median", phi=phi)


def test_flip_median_fold_is_byte_free_argmax_identity():
    """The byte-free fold changes out_sdf.bias VALUE, not size: ``phi(bias+b) == phi(bias)+b`` so
    ``argmax(phi + folded_bias) == power_diagram_argmax(phi, b)`` (exact in float64)."""
    phi, gt = _erasure_phi(n=2000)
    b, _ = flip_median_offsets(phi, gt)
    bias0 = np.full(5, 0.3, np.float32)
    params = {"out_sdf.bias": bias0.copy()}
    folded = apply_offset_to_sdf_bias(params, b)
    # the fold IS bias0 + b (byte-free value change; stored in the float32 deploy dtype)
    assert np.allclose(folded["out_sdf.bias"].astype(np.float64), bias0.astype(np.float64) + b, atol=1e-6)
    lab_offset = power_diagram_argmax(phi, b)  # argmax(phi + b), invariant to the global bias0
    lab_folded = np.argmax(phi + folded["out_sdf.bias"].astype(np.float64), axis=-1)
    # exact in real arithmetic; the float32 deploy-dtype storage flips only a negligible near-tie
    # fraction (< 0.1%), so agreement is ~1.0 (NOT a bug — the deploy bias IS float32)
    assert float(np.mean(lab_offset.reshape(-1) == lab_folded)) > 0.999
    assert np.array_equal(params["out_sdf.bias"], bias0)  # copy semantics — input not mutated
