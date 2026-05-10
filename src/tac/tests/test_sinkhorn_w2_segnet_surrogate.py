"""Tests for the opt-in entropic Sinkhorn SegNet training surrogate (T8).

Council coverage:
- Sinkhorn convergence (residual decreases monotonically vs iteration count)
- Fisher-Rao consistency at low blur (W₂ at blur→0 is bounded by FR² × max-cost,
  not numerically equal — these are different geodesics — but both satisfy
  d(p, p) = 0 and have monotone scaling in distribution discrepancy)
- Gradient finite at simplex boundary (one-hot argmax is the failure case for
  L²; W₂ should remain differentiable there)
- Mask-distortion monotone in mask discrepancy (interpolating GT → uniform
  should produce a monotone distortion curve)
- Batched-vs-loop equivalence (the einsum-batched implementation must match a
  per-pixel loop reference)
- Validation (blur out of range, n_iters out of range, cost matrix shape /
  symmetry / sign / diagonal)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tac.losses import (
    DEFAULT_SEGNET_NUM_CLASSES,
    DEFAULT_SINKHORN_BLUR,
    DEFAULT_SINKHORN_ITERS,
    SINKHORN_MIN_BLUR,
    SEGMENTATION_SURROGATE_FISHER_RAO,
    SEGMENTATION_SURROGATE_SINKHORN,
    SEGMENTATION_SURROGATE_SOFT_COSINE,
    _default_categorical_cost_matrix,
    segnet_fisher_rao_per_pixel,
    segnet_surrogate_per_pixel,
    sinkhorn_w2_mask_distortion_per_pixel,
)


def _one_hot(labels: torch.Tensor, *, classes: int = 5) -> torch.Tensor:
    return F.one_hot(labels, num_classes=classes).permute(0, 3, 1, 2).float()


def _renorm(probs: torch.Tensor) -> torch.Tensor:
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)


# -------------------- Identity / symmetry / scale --------------------


def test_sinkhorn_identical_distributions_are_near_zero() -> None:
    probs = _renorm(torch.rand(2, 5, 3, 4) + 1e-3)

    out = sinkhorn_w2_mask_distortion_per_pixel(probs, probs)

    # Entropic regularization makes the diagonal slightly nonzero (~ε * H(p))
    # but it should be O(1e-2) at default blur=0.05 and O(1e-3) at blur=0.01.
    assert torch.all(out >= 0.0)
    assert out.max().item() < 0.3, (
        "Sinkhorn-W2(p, p) should be near zero at default blur"
    )


def test_sinkhorn_smaller_blur_makes_diagonal_tighter() -> None:
    """Numerical sanity: smaller ε → tighter (smaller) W2(p, p)."""
    probs = _renorm(torch.rand(1, 5, 4, 4) + 1e-3)

    big = sinkhorn_w2_mask_distortion_per_pixel(probs, probs, blur=0.5).mean().item()
    small = sinkhorn_w2_mask_distortion_per_pixel(probs, probs, blur=0.01).mean().item()

    assert small < big, (
        f"Sinkhorn-W2(p, p) should decrease with smaller blur; got "
        f"blur=0.5 → {big}, blur=0.01 → {small}"
    )


def test_sinkhorn_disjoint_one_hots_have_unit_scale() -> None:
    """With default cost = 1 - I, transport between disjoint one-hots
    should approach 1.0 (the entire mass moves a unit cost). The entropic
    regularization gives a small upward bias for small ε; we accept any
    value in [0.9, 1.1]."""
    labels_a = torch.zeros(1, 4, 4, dtype=torch.long)
    labels_b = torch.ones(1, 4, 4, dtype=torch.long)

    out = sinkhorn_w2_mask_distortion_per_pixel(
        _one_hot(labels_a), _one_hot(labels_b), blur=0.01, n_iters=50
    )

    assert torch.all(out > 0.9), out
    assert torch.all(out < 1.1), out


def test_sinkhorn_is_symmetric_at_high_iters() -> None:
    """The classic Sinkhorn algorithm alternates f-then-g updates, so at
    finite n_iters there is a small one-step asymmetry between W₂(p, q)
    and W₂(q, p). At the converged limit they are identical. This test
    pushes n_iters high enough (200) that the residual asymmetry is at
    the FP32 single-precision noise floor (a few × 1e-3 worst-case)."""
    torch.manual_seed(0)
    a = _renorm(torch.rand(1, 5, 3, 3) + 1e-3)
    b = _renorm(torch.rand(1, 5, 3, 3) + 1e-3)

    ab = sinkhorn_w2_mask_distortion_per_pixel(a, b, n_iters=200)
    ba = sinkhorn_w2_mask_distortion_per_pixel(b, a, n_iters=200)

    # FP32 Sinkhorn alternates f/g updates so even at 200 iters the
    # symmetry residual is bounded by the single-precision noise floor
    # (typically O(1e-3) worst-case on small unbalanced rows).
    diff = (ab - ba).abs().max().item()
    assert diff < 5e-3, f"Sinkhorn asymmetry at 200 iters: max diff = {diff}"


# -------------------- Convergence (residual ↓ with iters) --------------------


def test_sinkhorn_residual_decreases_monotonically_with_iters() -> None:
    """The Sinkhorn surrogate must converge: as n_iters grows, the
    transport-cost estimate should monotonically approach a fixed limit
    (residual_n - residual_∞ → 0 from above).

    We approximate "limit" by a high-iter run and check that
    intermediate iterates monotonically descend toward it.
    """
    p = _renorm(torch.tensor(
        [[[[0.7]], [[0.1]], [[0.1]], [[0.05]], [[0.05]]]]
    ))
    q = _renorm(torch.tensor(
        [[[[0.05]], [[0.7]], [[0.1]], [[0.1]], [[0.05]]]]
    ))

    limit = sinkhorn_w2_mask_distortion_per_pixel(
        p, q, blur=0.05, n_iters=200
    ).item()
    intermediates = [
        sinkhorn_w2_mask_distortion_per_pixel(p, q, blur=0.05, n_iters=k).item()
        for k in (1, 3, 5, 10, 20, 40)
    ]
    residuals = [abs(v - limit) for v in intermediates]

    # Residual should be (weakly) monotonically decreasing modulo FP32
    # noise floor (~1e-4 in single precision).
    for prev, cur in zip(residuals, residuals[1:]):
        assert cur <= prev + 1e-4, (
            f"residual not monotonically decreasing: {residuals}"
        )
    assert residuals[-1] < 1e-3, (
        f"residual at 40 iters should be ≤ 1e-3; got {residuals[-1]}"
    )


# -------------------- Fisher-Rao consistency --------------------


def test_sinkhorn_and_fisher_rao_agree_on_zero_for_identical() -> None:
    """Both surrogates should be ≈ 0 for identical distributions."""
    probs = _renorm(torch.rand(1, 5, 4, 4) + 1e-3)

    fr = segnet_fisher_rao_per_pixel(probs, probs).max().item()
    sk = sinkhorn_w2_mask_distortion_per_pixel(
        probs, probs, blur=0.01, n_iters=50
    ).max().item()

    assert fr < 1e-5
    # Sinkhorn at small ε should also be small but is NOT identically 0
    # (entropic regularization). We accept ≤ 0.05 at blur=0.01.
    assert sk < 0.05, sk


def test_sinkhorn_and_fisher_rao_both_increase_with_discrepancy() -> None:
    """Geometric consistency: as we move predicted distribution AWAY from
    GT, both surrogates must increase monotonically.

    This is the key composition rule both surrogates share — they are
    "tighter" than L² because they respect the simplex structure.
    """
    gt = _one_hot(torch.zeros(1, 1, 1, dtype=torch.long))  # one-hot at class 0
    fr_seq = []
    sk_seq = []
    for alpha in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        # Interpolate predicted between gt and uniform-on-non-class-0.
        non_zero_uniform = torch.tensor(
            [[[[0.0]], [[0.25]], [[0.25]], [[0.25]], [[0.25]]]]
        )
        pred = (1 - alpha) * gt + alpha * non_zero_uniform
        pred = _renorm(pred)
        fr_seq.append(segnet_fisher_rao_per_pixel(pred, gt).item())
        sk_seq.append(
            sinkhorn_w2_mask_distortion_per_pixel(
                pred, gt, blur=0.01, n_iters=50
            ).item()
        )
    # Both must be monotone non-decreasing.
    for prev, cur in zip(fr_seq, fr_seq[1:]):
        assert cur >= prev - 1e-6, f"FR not monotone: {fr_seq}"
    for prev, cur in zip(sk_seq, sk_seq[1:]):
        assert cur >= prev - 1e-6, f"Sinkhorn not monotone: {sk_seq}"


# -------------------- Gradient finite at simplex boundary --------------------


def test_sinkhorn_surrogate_has_finite_gradients_near_one_hot() -> None:
    """L² loss has VANISHING gradient at the simplex boundary. Sinkhorn-W2
    must remain differentiable there (the entropic regularization is the
    smoothness guarantee)."""
    pred_logits = torch.tensor(
        [[[[8.0]], [[-2.0]], [[-3.0]], [[-4.0]], [[-5.0]]]],
        requires_grad=True,
    )
    gt_probs = _one_hot(torch.zeros(1, 1, 1, dtype=torch.long))

    out = segnet_surrogate_per_pixel(
        pred_logits,
        gt_probs,
        surrogate=SEGMENTATION_SURROGATE_SINKHORN,
        gt_already_probs=True,
    ).mean()
    out.backward()

    assert pred_logits.grad is not None
    assert torch.isfinite(pred_logits.grad).all()
    # And the gradient must be NON-ZERO (the whole point of W2 over L²).
    assert pred_logits.grad.abs().max().item() > 0.0


def test_sinkhorn_gradient_finite_with_extreme_logits() -> None:
    """Stress: very large positive logit on wrong class → gradient must
    remain finite (no overflow / no NaN from log-domain underflow)."""
    pred_logits = torch.tensor(
        [[[[-20.0]], [[20.0]], [[-20.0]], [[-20.0]], [[-20.0]]]],
        requires_grad=True,
    )
    gt_probs = _one_hot(torch.zeros(1, 1, 1, dtype=torch.long))

    out = segnet_surrogate_per_pixel(
        pred_logits,
        gt_probs,
        surrogate=SEGMENTATION_SURROGATE_SINKHORN,
        sinkhorn_blur=0.05,
        sinkhorn_n_iters=20,
        gt_already_probs=True,
    ).mean()
    out.backward()

    assert pred_logits.grad is not None
    assert torch.isfinite(pred_logits.grad).all()


# -------------------- Batched-vs-loop equivalence --------------------


def test_batched_matches_per_pixel_loop_reference() -> None:
    """The row-batched implementation MUST agree with a hand-rolled
    per-pixel loop. This guards against accidental broadcasting bugs in
    the (N, C, C) tensor shapes."""
    torch.manual_seed(42)
    pred = _renorm(torch.rand(2, 5, 3, 4) + 1e-3)
    gt = _renorm(torch.rand(2, 5, 3, 4) + 1e-3)

    batched = sinkhorn_w2_mask_distortion_per_pixel(
        pred, gt, blur=0.05, n_iters=20
    )

    # Reference: loop pixel-by-pixel.
    b, c, h, w = pred.shape
    expected = torch.zeros(b, h, w)
    for bi in range(b):
        for hi in range(h):
            for wi in range(w):
                p_ij = pred[bi : bi + 1, :, hi : hi + 1, wi : wi + 1]
                q_ij = gt[bi : bi + 1, :, hi : hi + 1, wi : wi + 1]
                # Reuse the same function; with 1 spatial position the
                # batched path collapses to a single (1, C, C) computation.
                expected[bi, hi, wi] = sinkhorn_w2_mask_distortion_per_pixel(
                    p_ij, q_ij, blur=0.05, n_iters=20
                )[0, 0, 0]

    assert torch.allclose(batched, expected, atol=1e-5), (
        f"batched-vs-loop residual: {(batched - expected).abs().max().item()}"
    )


def test_chunked_sinkhorn_matches_unchunked_and_preserves_gradients() -> None:
    torch.manual_seed(20260510)
    pred = _renorm(torch.rand(2, 5, 4, 5) + 1e-3).requires_grad_(True)
    gt = _renorm(torch.rand(2, 5, 4, 5) + 1e-3)

    unchunked = sinkhorn_w2_mask_distortion_per_pixel(
        pred,
        gt,
        blur=0.05,
        n_iters=20,
        max_positions_per_chunk=None,
    )
    chunked = sinkhorn_w2_mask_distortion_per_pixel(
        pred,
        gt,
        blur=0.05,
        n_iters=20,
        max_positions_per_chunk=3,
    )

    assert torch.allclose(chunked, unchunked, atol=1e-6)
    chunked.mean().backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# -------------------- Surrogate dispatch --------------------


def test_segnet_surrogate_dispatches_sinkhorn_path() -> None:
    pred_logits = torch.randn(1, 5, 2, 2)
    gt_probs = _one_hot(torch.zeros(1, 2, 2, dtype=torch.long))

    via_dispatch = segnet_surrogate_per_pixel(
        pred_logits,
        gt_probs,
        surrogate=SEGMENTATION_SURROGATE_SINKHORN,
        sinkhorn_blur=0.05,
        sinkhorn_n_iters=20,
        gt_already_probs=True,
    )
    direct = sinkhorn_w2_mask_distortion_per_pixel(
        F.softmax(pred_logits, dim=1),
        gt_probs,
        blur=0.05,
        n_iters=20,
    )

    assert torch.allclose(via_dispatch, direct, atol=1e-6)


def test_soft_cosine_surrogate_unchanged_when_sinkhorn_added() -> None:
    """Backward compatibility: adding the sinkhorn dispatch path MUST NOT
    perturb the legacy soft_cosine numeric values."""
    pred_logits = torch.randn(2, 5, 3, 4)
    gt_logits = torch.randn(2, 5, 3, 4)

    expected = 1.0 - (
        F.softmax(pred_logits, dim=1) * F.softmax(gt_logits, dim=1)
    ).sum(dim=1)
    actual = segnet_surrogate_per_pixel(
        pred_logits,
        gt_logits,
        surrogate=SEGMENTATION_SURROGATE_SOFT_COSINE,
    )

    assert torch.equal(actual, expected)


def test_cached_segnet_probabilities_reject_non_unit_temperature_for_sinkhorn() -> None:
    pred_logits = torch.randn(1, 5, 2, 2)
    gt_probs = F.softmax(torch.randn(1, 5, 2, 2), dim=1)

    with pytest.raises(ValueError, match="cached SegNet probabilities"):
        segnet_surrogate_per_pixel(
            pred_logits,
            gt_probs,
            surrogate=SEGMENTATION_SURROGATE_SINKHORN,
            temperature=2.0,
            gt_already_probs=True,
        )


# -------------------- Validation --------------------


@pytest.mark.parametrize("bad_blur", [0.0, -1e-3, 1e-3, 1.5, float("inf"), float("nan")])
def test_sinkhorn_rejects_invalid_blur(bad_blur: float) -> None:
    probs = _renorm(torch.rand(1, 5, 2, 2))

    with pytest.raises(ValueError, match="sinkhorn_blur"):
        sinkhorn_w2_mask_distortion_per_pixel(probs, probs, blur=bad_blur)


@pytest.mark.parametrize("bad_iters", [0, -1, 1001])
def test_sinkhorn_rejects_invalid_n_iters(bad_iters: int) -> None:
    probs = _renorm(torch.rand(1, 5, 2, 2))

    with pytest.raises(ValueError, match="sinkhorn_n_iters"):
        sinkhorn_w2_mask_distortion_per_pixel(probs, probs, n_iters=bad_iters)


def test_sinkhorn_rejects_class_count_mismatch() -> None:
    probs = _renorm(torch.full((1, 4, 1, 1), 0.25))

    with pytest.raises(ValueError, match="expects 5 classes"):
        sinkhorn_w2_mask_distortion_per_pixel(probs, probs)


def test_sinkhorn_rejects_non_bchw_input() -> None:
    probs = torch.full((5, 1, 1), 0.2)

    with pytest.raises(ValueError, match="BCHW probability tensors"):
        sinkhorn_w2_mask_distortion_per_pixel(probs, probs)


def test_sinkhorn_rejects_non_symmetric_cost_matrix() -> None:
    probs = _renorm(torch.rand(1, 5, 2, 2))
    bad = torch.eye(5)
    bad[0, 1] = 1.0
    bad[1, 0] = 0.5  # asymmetric

    with pytest.raises(ValueError, match="cost_matrix must be symmetric"):
        sinkhorn_w2_mask_distortion_per_pixel(probs, probs, cost_matrix=bad)


def test_sinkhorn_rejects_negative_cost_matrix() -> None:
    probs = _renorm(torch.rand(1, 5, 2, 2))
    bad = (1.0 - torch.eye(5)).clone()
    bad[0, 1] = -1.0
    bad[1, 0] = -1.0

    with pytest.raises(ValueError, match="cost_matrix must be non-negative"):
        sinkhorn_w2_mask_distortion_per_pixel(probs, probs, cost_matrix=bad)


def test_sinkhorn_rejects_non_zero_diagonal_cost_matrix() -> None:
    probs = _renorm(torch.rand(1, 5, 2, 2))
    bad = (1.0 - torch.eye(5)).clone()
    bad[2, 2] = 0.5

    with pytest.raises(ValueError, match="cost_matrix must have zero diagonal"):
        sinkhorn_w2_mask_distortion_per_pixel(probs, probs, cost_matrix=bad)


def test_sinkhorn_rejects_wrong_cost_matrix_shape() -> None:
    probs = _renorm(torch.rand(1, 5, 2, 2))
    bad = torch.zeros(4, 4)

    with pytest.raises(ValueError, match="does not match"):
        sinkhorn_w2_mask_distortion_per_pixel(probs, probs, cost_matrix=bad)


def test_sinkhorn_rejects_shape_mismatch() -> None:
    pred = _renorm(torch.rand(1, 5, 2, 2))
    gt = _renorm(torch.rand(1, 5, 3, 3))

    with pytest.raises(ValueError, match="does not match"):
        sinkhorn_w2_mask_distortion_per_pixel(pred, gt)


def test_default_cost_matrix_shape_and_zero_diagonal() -> None:
    cost = _default_categorical_cost_matrix(
        DEFAULT_SEGNET_NUM_CLASSES,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert cost.shape == (DEFAULT_SEGNET_NUM_CLASSES, DEFAULT_SEGNET_NUM_CLASSES)
    assert torch.allclose(cost.diagonal(), torch.zeros(DEFAULT_SEGNET_NUM_CLASSES))
    assert torch.allclose(cost, cost.T)
    assert (cost >= 0).all()


def test_sinkhorn_at_min_blur_keeps_soft_mass_swap_visible() -> None:
    """At the minimum allowed blur, a real soft class-mass swap must stay
    visible. This guards the pre-fix low-blur failure where 20 iterations
    returned ~0 for an expected ~0.6 transport cost."""
    pred = _renorm(torch.tensor(
        [[[[0.7]], [[0.1]], [[0.1]], [[0.05]], [[0.05]]]]
    ))
    gt = _renorm(torch.tensor(
        [[[[0.1]], [[0.7]], [[0.1]], [[0.05]], [[0.05]]]]
    ))
    out = sinkhorn_w2_mask_distortion_per_pixel(
        pred, gt, blur=SINKHORN_MIN_BLUR, n_iters=20
    )
    assert torch.isfinite(out).all()
    assert 0.55 < out.item() < 0.65


def test_sinkhorn_at_min_blur_gradient_does_not_nan() -> None:
    """At the minimum allowed blur, the loss and gradient must remain usable."""
    pred = torch.tensor(
        [[[[0.5, 0.5]], [[0.5, 0.5]], [[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]]],
        requires_grad=True,
    )
    gt = _renorm(torch.tensor(
        [[[[0.0, 0.0]], [[0.0, 0.0]], [[0.5, 0.5]], [[0.5, 0.5]], [[0.0, 0.0]]]]
    ))
    out = sinkhorn_w2_mask_distortion_per_pixel(
        _renorm(pred), gt, blur=SINKHORN_MIN_BLUR, n_iters=20
    )
    assert torch.isfinite(out).all()
    out.mean().backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_sinkhorn_default_blur_and_iters_are_constants() -> None:
    """Sanity: the documented defaults match the module-level constants
    (so callers reading the docstring see the same value the function
    uses)."""
    assert DEFAULT_SINKHORN_BLUR == 0.05
    assert DEFAULT_SINKHORN_ITERS == 20
