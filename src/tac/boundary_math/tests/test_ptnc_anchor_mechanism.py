# SPDX-License-Identifier: MIT
"""Behavior tests for the PTNC weighted-anchor loss MECHANISM (task #61).

These test the load-bearing piece of the PTNC trainer (``tools/ptnc_train_pose_carrier.py``): the
saliency-WEIGHTED reconstruction anchor that distinguishes PTNC from the #57 dense anchor. The unit
under test is the anchor computation ``(w * (carrier - gt)**2).mean()`` and its three modes:

  * dense / identity : uniform weight => the anchor == plain mean-MSE (the #57 control).
  * ptnc             : Jacobian-saliency weight => the anchor concentrates penalty on pose-relevant
                       pixels and tolerates error in the pose-null.

NO-FAKE (class 2 + class 6): a CONSTANT carrier yields a HIGH anchor (does not satisfy the loss) — a
stub returning a constant fails. Replacing the saliency weight with IDENTITY recovers the dense anchor
exactly (the weight field is load-bearing, not cosmetic). The PTNC weight makes a pose-null error
CHEAPER and a pose-relevant error MORE EXPENSIVE than dense — the genuine, falsifiable distinction.

The on-scorer RD comparison (PTNC vs dense at the quantized operating point) is recorded empirically in
the verdict memo; these tests lock the mechanism that comparison rests on so a future edit cannot
silently turn PTNC into a rename of dense.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tac.boundary_math.posenet_jacobian_saliency import (  # noqa: E402
    PixelSaliencyField,
    identity_weight_map,
    saliency_to_weight_map,
)


def _weighted_anchor(carrier_chw, gt_chw, weight_hw):
    """The exact anchor the PTNC trainer minimises: mean over channels of w*(carrier-gt)^2."""
    w = torch.as_tensor(weight_hw, dtype=torch.float32)
    resid2 = (carrier_chw - gt_chw) ** 2  # (3,H,W)
    return (w.unsqueeze(0) * resid2).mean()


def _field(saliency: np.ndarray) -> PixelSaliencyField:
    s = np.asarray(saliency, dtype=np.float32)
    return PixelSaliencyField(
        saliency=s, h=s.shape[0], w=s.shape[1], frame_slot=0, compute_path="cpu_torch",
        nonzero_fraction=float(np.mean(s > 0)), max_value=float(s.max()),
    )


def test_identity_anchor_equals_plain_mse():
    """dense / identity weight => the weighted anchor is exactly plain mean-MSE."""
    g = torch.rand(3, 8, 10) * 255.0
    c = torch.rand(3, 8, 10) * 255.0
    w = identity_weight_map(8, 10)
    anchor = _weighted_anchor(c, g, w)
    plain = torch.nn.functional.mse_loss(c, g)
    assert torch.allclose(anchor, plain, atol=1e-4)


def test_constant_carrier_has_high_anchor():
    """A constant carrier (the stub) does NOT satisfy the anchor (high residual) vs a near-GT carrier."""
    g = (torch.rand(3, 8, 10) * 255.0)
    const = torch.full((3, 8, 10), 128.0)
    near = g + 1.0
    w = identity_weight_map(8, 10)
    assert _weighted_anchor(const, g, w) > 10.0 * _weighted_anchor(near, g, w)


def test_ptnc_makes_pose_null_error_cheaper_than_dense():
    """An error placed in a pose-NULL pixel costs LESS under PTNC than under dense (the free subspace)."""
    h, w = 6, 6
    g = torch.zeros(3, h, w)
    # one pose-null pixel (low saliency) carries an error.
    sal = np.full((h, w), 1.0, dtype=np.float32)
    sal[0, 0] = 1e-4  # pose-null at (0,0)
    wm = saliency_to_weight_map(_field(sal), floor=0.02, gamma=1.0, normalize=True)
    c = torch.zeros(3, h, w)
    c[:, 0, 0] = 50.0  # error in the pose-null pixel
    ptnc_cost = _weighted_anchor(c, g, torch.from_numpy(wm))
    dense_cost = _weighted_anchor(c, g, torch.from_numpy(identity_weight_map(h, w)))
    assert ptnc_cost < dense_cost  # PTNC tolerates the pose-null error


def test_ptnc_makes_pose_relevant_error_more_expensive_than_dense():
    """An error in a pose-RELEVANT (high-saliency) pixel costs MORE under PTNC than under dense."""
    h, w = 6, 6
    g = torch.zeros(3, h, w)
    sal = np.full((h, w), 1e-3, dtype=np.float32)
    sal[3, 3] = 1.0  # the pose tube at (3,3)
    wm = saliency_to_weight_map(_field(sal), floor=0.02, gamma=1.0, normalize=True)
    c = torch.zeros(3, h, w)
    c[:, 3, 3] = 50.0  # error in the pose-relevant pixel
    ptnc_cost = _weighted_anchor(c, g, torch.from_numpy(wm))
    dense_cost = _weighted_anchor(c, g, torch.from_numpy(identity_weight_map(h, w)))
    assert ptnc_cost > dense_cost  # PTNC penalises the pose-tube error harder


def test_zero_residual_zero_anchor_any_weight():
    """A perfect carrier => zero anchor regardless of weight map."""
    g = torch.rand(3, 5, 5) * 255.0
    sal = np.random.default_rng(0).random((5, 5)).astype(np.float32) + 1e-3
    wm = saliency_to_weight_map(_field(sal))
    assert _weighted_anchor(g.clone(), g, torch.from_numpy(wm)).item() == pytest.approx(0.0, abs=1e-6)


def test_anchor_is_differentiable_to_carrier():
    """The anchor must produce a gradient to the carrier (it is the training signal)."""
    g = torch.zeros(3, 4, 4)
    c = (torch.rand(3, 4, 4) * 255.0).requires_grad_(True)
    sal = np.full((4, 4), 1.0, dtype=np.float32)
    sal[0, 0] = 1.0
    wm = saliency_to_weight_map(_field(sal))
    loss = _weighted_anchor(c, g, torch.from_numpy(wm))
    loss.backward()
    assert c.grad is not None and torch.any(c.grad != 0)


def test_higher_floor_moves_ptnc_toward_dense():
    """floor->1 makes PTNC's pose-null tolerance shrink (toward the dense anchor) — the diagnosis fix."""
    h, w = 6, 6
    g = torch.zeros(3, h, w)
    sal = np.full((h, w), 1.0, dtype=np.float32)
    sal[0, 0] = 1e-4
    c = torch.zeros(3, h, w)
    c[:, 0, 0] = 50.0
    cost_lowfloor = _weighted_anchor(c, g, torch.from_numpy(saliency_to_weight_map(_field(sal), floor=0.02)))
    cost_hifloor = _weighted_anchor(c, g, torch.from_numpy(saliency_to_weight_map(_field(sal), floor=0.9)))
    dense = _weighted_anchor(c, g, torch.from_numpy(identity_weight_map(h, w)))
    # higher floor => pose-null error penalised more (closer to dense).
    assert cost_lowfloor < cost_hifloor <= dense + 1e-5
