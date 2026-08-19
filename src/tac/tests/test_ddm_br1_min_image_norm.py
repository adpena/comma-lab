"""Tests for ddm_br1's core geometry: the minimum-IMAGE-norm in-span step.

These cover the two claims the arm's verdict rests on:

1. ``min_image_norm_step`` returns the smallest step in the SPAN metric that
   cancels the residual to first order.  If it were merely *a* feasible step --
   which is what ``torch.linalg.lstsq`` returns on an underdetermined system --
   the resulting "basis penalty" would be basis-dependent and could not be read
   as a property of the span.

2. The IMAGE step it induces is invariant under re-mixing the basis.  This is the
   whole reason a rotation of the 12 carrier dimensions cannot buy pose, so it is
   tested rather than asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))


def _load_module():
    """Import the arm module without importing its heavy up2 dependency chain."""
    return pytest.importorskip("ddm_br1_pose_basis_reorientation")


def _random_problem(seed: int, dims: int = 12, pose: int = 6, field: int = 40):
    generator = torch.Generator().manual_seed(seed)
    basis = torch.randn(dims, field, generator=generator, dtype=torch.float64)
    jac = torch.randn(pose, dims, generator=generator, dtype=torch.float64)
    residual = torch.randn(pose, generator=generator, dtype=torch.float64)
    gram = basis @ basis.T
    return basis, jac, residual, gram


def test_step_is_feasible():
    """The step must cancel the residual exactly, to first order."""
    module = _load_module()
    _basis, jac, residual, gram = _random_problem(1)
    step = module.min_image_norm_step(jac, residual, gram)
    assert torch.allclose(jac @ step, -residual, atol=1e-9)


def test_step_is_minimal_in_the_image_metric():
    """No other feasible step has a smaller IMAGE norm."""
    module = _load_module()
    basis, jac, residual, gram = _random_problem(2)
    step = module.min_image_norm_step(jac, residual, gram)
    best = float(step @ gram @ step)

    # Feasible perturbations: anything in the null space of the Jacobian.
    null = torch.linalg.svd(jac, full_matrices=True)[2][jac.shape[0] :]
    generator = torch.Generator().manual_seed(3)
    for _ in range(64):
        mix = torch.randn(null.shape[0], generator=generator, dtype=torch.float64)
        other = step + null.T @ mix
        assert torch.allclose(jac @ other, -residual, atol=1e-9)
        assert float(other @ gram @ other) >= best - 1e-12


def test_image_step_is_invariant_under_basis_remixing():
    """Re-mixing the basis leaves the induced IMAGE step unchanged.

    This is the ddm_br1 refusal in miniature: a rotation is a different basis for
    the same subspace, so the reachable image step cannot move.
    """
    module = _load_module()
    basis, jac, residual, gram = _random_problem(4)
    step = module.min_image_norm_step(jac, residual, gram)
    image = step @ basis

    generator = torch.Generator().manual_seed(5)
    for _ in range(8):
        rot = torch.randn(basis.shape[0], basis.shape[0], generator=generator,
                          dtype=torch.float64)
        if abs(float(torch.linalg.det(rot))) < 1e-6:
            continue
        basis_r = rot @ basis
        # Chain rule for the re-mixed basis.  With basis_r = rot @ basis, a new
        # coefficient vector c_r produces the same image as c = rot.T @ c_r, so
        # d(pose)/d(c_r) = jac @ rot.T.  (Writing inv(rot) here instead is wrong
        # and makes this test fail -- it did, on the first run.)
        jac_r = jac @ rot.T
        step_r = module.min_image_norm_step(jac_r, residual, basis_r @ basis_r.T)
        image_r = step_r @ basis_r
        assert torch.allclose(image_r, image, atol=1e-8)


def test_lstsq_is_not_the_min_image_norm_solution():
    """Guards the instrument correction this arm had to make.

    ``lstsq`` on an underdetermined system returns a feasible step, not the
    minimum-image-norm one, so a penalty computed from it overstates the span's
    true cost.  If a future torch ever made these agree, this test should fail so
    the correction gets re-derived rather than silently inherited.
    """
    module = _load_module()
    basis, jac, residual, gram = _random_problem(6)
    step = module.min_image_norm_step(jac, residual, gram)
    lstsq = torch.linalg.lstsq(jac, -residual.unsqueeze(-1)).solution.squeeze(-1)
    assert torch.allclose(jac @ lstsq, -residual, atol=1e-9)
    assert float(lstsq @ gram @ lstsq) > float(step @ gram @ step)


def test_realize_clamps_to_the_shipped_int12_lattice():
    module = _load_module()
    values = np.array([-9e9, -2048.4, -0.4, 0.6, 2047.4, 9e9], dtype=np.float64)
    out = module.realize(values)
    assert out.dtype == np.int32
    assert out.min() >= module.CODE_MIN
    assert out.max() <= module.CODE_MAX
    assert out[2] == 0
    assert out[3] == 1


def test_sample_pairs_is_never_a_prefix():
    """Pose prefixes measure 2.54-4.21x harder than the population (ddm_na2)."""
    module = _load_module()
    picked = module.sample_pairs(24, seed=20260819)
    assert len(picked) == 24
    assert len(set(picked.tolist())) == 24
    assert not np.array_equal(picked, np.arange(24))
    assert picked.max() > 24
    full = module.sample_pairs(600, seed=1)
    assert np.array_equal(full, np.arange(600))
