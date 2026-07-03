# SPDX-License-Identifier: MIT
"""Tests for tac.boundary_math.persistence_topology_loss (soft-clDice + island recall).

Covers: numpy morphology correctness, soft-skeleton, clDice semantics (identity/erasure/birth),
persistence recall weight (interior→0, thin→high), presence-gating, external-margin compose,
numpy↔MLX bit/ratio parity, MLX gradient flow, mx.compile equivalence, class self-detection
(synthetic + real n600 guarded), anneal schedule, Metal-kernel flag, canonical-equation build.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.boundary_math import persistence_topology_loss as P

mx = pytest.importorskip("mlx.core")

GT_N96 = Path("experiments/results/mlx_fleet_gt_cache/gt_n96.npz")


def _oh(lab, C=5):
    return np.eye(C, dtype=np.float32)[lab]


# --------------------------------------------------------------------------- morphology
def test_soft_erode_is_3x3_min():
    x = np.zeros((5, 5), np.float32)
    x[2, 2] = 1.0
    er = P.soft_erode_np(x)
    assert er.max() == 0.0  # a single hot pixel is eroded away by 3x3 min


def test_soft_dilate_is_3x3_max():
    x = np.zeros((5, 5), np.float32)
    x[2, 2] = 1.0
    di = P.soft_dilate_np(x)
    assert di[1:4, 1:4].min() == 1.0  # dilated into the 3x3 neighborhood


def test_soft_open_removes_thin_keeps_thick():
    thin = np.zeros((7, 7), np.float32)
    thin[3, :] = 1.0  # 1px horizontal line
    opened = P.soft_open_np(thin)
    assert opened.sum() < thin.sum()  # opening erodes the 1px line


def test_soft_skeleton_nonzero_on_line_zero_on_uniform():
    line = np.zeros((9, 9), np.float32)
    line[4, 1:8] = 1.0
    sk = P.soft_skeleton_np(line, 5)
    assert sk.sum() > 0.0
    uni = np.ones((9, 9), np.float32)
    assert P.soft_skeleton_np(uni, 5).sum() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------- sg-cache (bit-identical speed, #260)
def _rand_seg(n=2, H=24, W=32, C=5, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((n, H, W, C)).astype(np.float32)
    oh = np.eye(C, dtype=np.float32)[rng.integers(0, C, size=(n, H, W))]
    return mx.array(logits), mx.array(oh)


def test_sg_precompute_is_bit_identical():
    """precompute_sg_mlx + sg_precomputed= gives the EXACT same loss (sg is a constant → free)."""
    logits, oh = _rand_seg(seed=7)
    cls = [1, 3]
    base = P.persistence_topology_loss_mlx(logits, oh, cls)
    sg = P.precompute_sg_mlx(oh, cls)
    cached = P.persistence_topology_loss_mlx(logits, oh, cls, sg_precomputed=sg)
    mx.eval(base, cached)
    assert float(base) == float(cached)  # BIT-IDENTICAL, not approx


def test_sg_precompute_bit_identical_gradient():
    """The gradient w.r.t. logits must ALSO be bit-identical (sg carries no gradient)."""
    logits, oh = _rand_seg(seed=11)
    cls = [1, 3]
    sg = P.precompute_sg_mlx(oh, cls)
    gb = mx.grad(lambda lg: P.persistence_topology_loss_mlx(lg, oh, cls))(logits)
    gc = mx.grad(lambda lg: P.persistence_topology_loss_mlx(lg, oh, cls, sg_precomputed=sg))(logits)
    mx.eval(gb, gc)
    assert bool(mx.all(gb == gc))


def test_sg_precompute_shape_guard():
    logits, oh = _rand_seg(seed=3)
    with pytest.raises(ValueError):
        P.persistence_topology_loss_mlx(logits, oh, [1, 3], sg_precomputed=mx.zeros((99, 4, 4)))


# --------------------------------------------------------------------------- clDice
def test_cldice_identity_is_zero():
    g = np.zeros((16, 16), np.float32)
    g[7, 2:14] = 1.0
    loss = P.soft_cldice_np(g, g, 5)
    assert loss < 1e-3  # identical fields -> clDice ~1 -> loss ~0


def test_cldice_erasure_raises_loss():
    g = np.zeros((16, 16), np.float32)
    g[7, 2:14] = 1.0
    erased = g.copy()
    erased[7, 5:10] = 0.0  # cut the line -> disconnected -> topology broken
    assert P.soft_cldice_np(erased, g, 5) > P.soft_cldice_np(g, g, 5) + 1e-3


def test_cldice_full_dropout_is_max_loss():
    g = np.zeros((16, 16), np.float32)
    g[7, 2:14] = 1.0
    pred = np.zeros_like(g)  # fully erased
    assert P.soft_cldice_np(pred, g, 5) > 0.9  # birth force: max penalty


# --------------------------------------------------------------------------- recall weight
def test_recall_weight_interior_zero_thin_high():
    big = np.zeros((32, 32), np.float32)
    big[8:24, 8:24] = 1.0  # 16x16 block
    w = P.persistence_recall_weight_np(big, density_iters=4)
    assert w[15, 15] < 0.05  # deep interior -> density~1 -> weight~0 (BULK untouched)
    thin = np.zeros((32, 32), np.float32)
    thin[16, 4:28] = 1.0  # 1px line
    wt = P.persistence_recall_weight_np(thin, density_iters=4)
    assert wt[16, 16] > 0.3  # thin structure -> low density -> high weight


def test_recall_weight_nonnegative_and_masked():
    g = (np.random.default_rng(0).random((20, 20)) > 0.5).astype(np.float32)
    w = P.persistence_recall_weight_np(g)
    assert (w >= 0).all()
    assert (w[g == 0] == 0).all()  # zero outside the class mask


def test_recall_bce_extra_weight_reweights():
    # A SPATIALLY-VARYING extra weight (the margin-map use case) re-weights the recall; a
    # constant does NOT (mean-normalization is scale-invariant, by design).
    lab = np.zeros((16, 16), np.int64)
    lab[7, 2:14] = 1
    g = _oh(lab)[..., 1]
    prob = g.copy() * 0.0
    prob[7, 2:8] = 0.9   # left half well predicted
    prob[7, 8:14] = 0.05  # right half erased
    base = P.persistence_recall_bce_np(prob, g)
    ew = np.ones((16, 16), np.float32)
    ew[7, 8:14] = 5.0  # upweight the erased (low-margin) half
    reweighted = P.persistence_recall_bce_np(prob, g, extra_weight=ew)
    assert reweighted > base  # focusing the erased half raises the recall penalty
    const = P.persistence_recall_bce_np(prob, g, extra_weight=np.full((16, 16), 3.0, np.float32))
    assert const == pytest.approx(base, rel=1e-5)  # constant weight is scale-invariant


# --------------------------------------------------------------------------- presence gate
def test_presence_gate_absent_class_zero_contribution():
    lab = np.zeros((24, 24), np.int64)
    lab[10, 2:22] = 1  # only class 1 present, class 3 absent
    logits = _oh(lab) * 6.0
    oh = _oh(lab)
    only1 = P.persistence_topology_loss_np(logits, oh, (1,))
    with3 = P.persistence_topology_loss_np(logits, oh, (1, 3))  # class 3 absent -> same
    assert only1 == pytest.approx(with3, abs=1e-6)


def test_empty_target_classes_is_zero_both_backends():
    lab = np.zeros((16, 16), np.int64)
    lab[7, 2:14] = 1
    logits = _oh(lab) * 6.0
    oh = _oh(lab)
    assert P.persistence_topology_loss_np(logits, oh, ()) == 0.0
    # MLX must not crash on mx.stack([]) — guarded return
    assert float(P.persistence_topology_loss_mlx(mx.array(logits), mx.array(oh), ()).item()) == 0.0


def test_no_present_class_is_zero():
    lab = np.zeros((16, 16), np.int64)  # all class 0
    logits = _oh(lab) * 6.0
    oh = _oh(lab)
    assert P.persistence_topology_loss_np(logits, oh, (1, 3)) == 0.0
    lm = P.persistence_topology_loss_mlx(mx.array(logits), mx.array(oh), (1, 3))
    assert float(lm.item()) == 0.0


# --------------------------------------------------------------------------- parity
def test_pool_bit_identical():
    x = np.random.default_rng(1).random((3, 20, 24)).astype(np.float32)
    en = P.soft_erode_np(x)
    em = np.asarray(P.soft_erode_mlx(mx.array(x)))
    assert np.abs(en - em).max() == 0.0  # selection ops -> bit-identical


def test_skeleton_bit_identical():
    x = np.random.default_rng(2).random((2, 24, 24)).astype(np.float32)
    sn = P.soft_skeleton_np(x, 5)
    sm = np.asarray(P.soft_skeleton_mlx(mx.array(x), 5))
    assert np.abs(sn - sm).max() < 1e-6


def test_toplevel_parity_ge_0_9997():
    rng = np.random.default_rng(3)
    logits = (rng.standard_normal((48, 64, 5)) * 3).astype(np.float32)
    oh = _oh(logits.argmax(-1))
    ln = P.persistence_topology_loss_np(logits, oh, (1, 3))
    lm = float(P.persistence_topology_loss_mlx(mx.array(logits), mx.array(oh), (1, 3)).item())
    ratio = min(ln, lm) / max(ln, lm)
    assert ratio >= 0.9997


def test_batched_matches_per_frame_mean():
    rng = np.random.default_rng(4)
    logits = (rng.standard_normal((3, 32, 40, 5)) * 3).astype(np.float32)
    oh = _oh(logits.argmax(-1))
    batched = float(P.persistence_topology_loss_mlx(mx.array(logits), mx.array(oh), (1, 3)).item())
    ref = P.persistence_topology_loss_np(logits, oh, (1, 3))
    assert min(batched, ref) / max(batched, ref) >= 0.9997


# --------------------------------------------------------------------------- gradient / compile
def test_mlx_gradient_finite_and_flows():
    rng = np.random.default_rng(5)
    logits = mx.array((rng.standard_normal((40, 48, 5)) * 3).astype(np.float32))
    oh = mx.array(_oh(np.asarray(logits).argmax(-1)))

    def loss(lg):
        return P.persistence_topology_loss_mlx(lg, oh, (1, 3))

    g = mx.grad(loss)(logits)
    assert bool(mx.all(mx.isfinite(g)).item())
    assert float(mx.abs(g).sum().item()) > 0.0


def test_compiled_matches_uncompiled():
    rng = np.random.default_rng(6)
    logits = mx.array((rng.standard_normal((32, 40, 5)) * 3).astype(np.float32))
    oh = mx.array(_oh(np.asarray(logits).argmax(-1)))
    un = float(P.persistence_topology_loss_mlx(logits, oh, (1, 3)).item())
    fn = P.make_persistence_topology_loss_mlx_compiled((1, 3))
    co = float(fn(logits, oh).item())
    assert un == pytest.approx(co, abs=1e-4)


# --------------------------------------------------------------------------- self-detection
def test_detect_synthetic_picks_thin_class():
    # class 1 = many thin lines; class 2 = one big block; rest background (0).
    lab = np.zeros((2, 64, 64), np.int64)
    for f in range(2):
        for r in range(4, 60, 8):
            lab[f, r, 4:60] = 1  # thin dashes
        lab[f, 20:50, 20:50] = 2  # big block
    targets, ev = P.detect_persistence_tail_classes(lab, top_k=1)
    assert targets == (1,)  # the thin/multi-component class
    risks = {e.cls: e.erasure_risk for e in ev}
    assert risks[1] > risks[2]


@pytest.mark.skipif(not GT_N96.exists(), reason="n96 GT cache absent")
def test_detect_real_n600_picks_lane_movable():
    lstars = np.load(GT_N96)["lstars"]
    targets, _ = P.detect_persistence_tail_classes(lstars, top_k=2, max_frames=48)
    assert sorted(targets) == [1, 3]  # Lane + Movable (self-detected, never hardcoded)


# --------------------------------------------------------------------------- misc
def test_anneal_ramp():
    assert P.persistence_anneal_weight(0, 1.0, 10) == 0.0
    assert P.persistence_anneal_weight(5, 1.0, 10) == pytest.approx(0.5)
    assert P.persistence_anneal_weight(20, 1.0, 10) == pytest.approx(1.0)  # clamps at base
    assert P.persistence_anneal_weight(3, 0.0, 10) == 0.0  # zero base stays zero
    assert P.persistence_anneal_weight(3, 2.0, 0) == 2.0  # no warmup -> full


def test_metal_kernel_signature():
    sig = P.metal_pool_kernel_signature()
    assert sig["env_flag"] == P.PERSISTENCE_POOL_METAL_KERNEL_FLAG
    assert "parity_reference" in sig and "persistence_topology_loss" in sig["parity_reference"]
    assert sig["input_names"] and sig["output_names"]


def test_build_canonical_equation_valid():
    eq = P.build_canonical_equation(
        island_recall_gain=0.443,
        bulk_dseg_delta=0.0,
        cldice_erasure_sensitivity=8.83,
        ce_erasure_sensitivity=0.08,
        verification_artifact="experiments/results/persistence_topology_verification/verification.json",
        measured_utc="2026-07-01T00:00:00Z",
    )
    assert eq.equation_id == P.CANONICAL_EQUATION_ID
    assert eq.empirical_anchors and eq.empirical_anchors[0].residual >= 0.0
    assert eq.canonical_consumers  # not orphan
