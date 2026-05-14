# SPDX-License-Identifier: MIT
"""Bridge audit: tac.score_geometry (torch-free) vs tac.contest_rate_distortion_system (torch).

Both modules implement the same contest objective:

    S = 100*d_seg + sqrt(10*d_pose) + 25*B/N_REF

`score_geometry` is the torch-free planning analyzer (used by tools like
`dispatch_advisor.py`, `contest_score_pareto_3axis.py`).

`contest_rate_distortion_system` is the torch-differentiable companion
(used by Joint-ADMM coordinators that need autograd-enabled marginals).

These tests confirm they agree numerically at 6+ test points so a future
maintainer can swap one for the other without changing semantics. Any
divergence (other than torch dtype noise) means a regression in one of
them.

Audit boundary: scalar contest score + per-axis marginals + decomposition.
The two modules diverge in surface only (autograd vs pure float), not in
mathematical contract.
"""
from __future__ import annotations

import math

import pytest
import torch

from tac.contest_rate_distortion_system import (
    contest_score as rd_contest_score,
)
from tac.contest_rate_distortion_system import (
    contest_score_decomposition as rd_decomp,
)
from tac.contest_rate_distortion_system import (
    contest_score_marginals as rd_marginals,
)
from tac.score_geometry import (
    contest_score as sg_contest_score,
)
from tac.score_geometry import (
    score_decomposition as sg_decomp,
)
from tac.score_geometry import (
    score_gradient as sg_gradient,
)

# 6+ test points spanning the operating-regime spectrum:
#   - PR106 frontier (pose-dominated, real)
#   - importance-flip threshold (boundary)
#   - legacy 1.x (seg-dominated, real)
#   - tiny score (stress lower bound)
#   - large score (stress upper bound)
#   - all-zeros distortion (pure rate floor)
ANCHOR_POINTS = [
    ("pr106_frontier", 6.7e-4, 3.4e-5, 178258),
    ("flip_threshold", 5e-4, 2.5e-4, 178258),  # at the importance flip
    ("legacy_1x", 0.001, 0.18, 300_000),
    ("near_zero", 1e-6, 1e-7, 100_000),
    ("near_one", 0.005, 0.08, 400_000),
    ("rate_only_floor", 0.0, 0.0, 178258),
    ("smallest", 0.0, 1e-30, 1),  # singular pose; check graceful handling
]


@pytest.mark.parametrize("label,d_seg,d_pose,archive_bytes", ANCHOR_POINTS)
def test_contest_score_agrees(label: str, d_seg: float, d_pose: float, archive_bytes: int) -> None:
    """Both score functions must produce the same scalar at every anchor."""
    sg_score = sg_contest_score(d_seg=d_seg, d_pose=d_pose, archive_bytes=archive_bytes)
    rd_score = float(rd_contest_score(
        seg_distortion=d_seg,
        pose_distortion=d_pose,
        archive_bytes=archive_bytes,
    ))
    # Floats can drift by a few ULP through torch — use an absolute
    # tolerance scaled to the magnitude of the score
    assert math.isclose(sg_score, rd_score, rel_tol=1e-6, abs_tol=1e-9), (
        f"score divergence at {label}: score_geometry={sg_score} "
        f"vs contest_rate_distortion_system={rd_score}"
    )


@pytest.mark.parametrize("label,d_seg,d_pose,archive_bytes",
                         [p for p in ANCHOR_POINTS if p[2] > 0])
def test_pose_marginal_agrees(
    label: str, d_seg: float, d_pose: float, archive_bytes: int
) -> None:
    """Pose marginal must agree at every non-singular anchor.

    The torch-free formula is `0.5 * sqrt(POSE_INSIDE / d_pose)`.
    The torch formula is `sqrt(POSE_INSIDE) / (2 * sqrt(d_pose))`.
    Algebraically equivalent.
    """
    sg_grad = sg_gradient(d_seg=d_seg, d_pose=d_pose)
    rd_marg = rd_marginals(
        seg_distortion=d_seg,
        pose_distortion=d_pose,
        archive_bytes=archive_bytes,
    )
    assert math.isclose(sg_grad.d_pose, rd_marg["dS_dpose"], rel_tol=1e-9), (
        f"pose marginal divergence at {label}: "
        f"sg={sg_grad.d_pose} vs rd={rd_marg['dS_dpose']}"
    )


@pytest.mark.parametrize("label,d_seg,d_pose,archive_bytes", ANCHOR_POINTS)
def test_seg_and_byte_marginals_constant_and_match(
    label: str, d_seg: float, d_pose: float, archive_bytes: int
) -> None:
    """Seg and byte marginals are constants and must match exactly."""
    sg_grad = sg_gradient(d_seg=d_seg, d_pose=d_pose)
    rd_marg = rd_marginals(
        seg_distortion=d_seg,
        pose_distortion=d_pose,
        archive_bytes=archive_bytes,
    )
    assert sg_grad.d_seg == rd_marg["dS_dseg"] == 100.0
    assert math.isclose(sg_grad.d_bytes, rd_marg["dS_dbytes"], rel_tol=1e-12)


@pytest.mark.parametrize("label,d_seg,d_pose,archive_bytes", ANCHOR_POINTS)
def test_decomposition_agrees(
    label: str, d_seg: float, d_pose: float, archive_bytes: int
) -> None:
    """Per-term contributions (seg/pose/rate) must agree."""
    sg_d = sg_decomp(d_seg=d_seg, d_pose=d_pose, archive_bytes=archive_bytes)
    rd_d = rd_decomp(
        seg_distortion=d_seg,
        pose_distortion=d_pose,
        archive_bytes=archive_bytes,
    )
    # The torch RD module may use slightly different key names; tolerate
    # both shapes by mapping to the canonical 'seg_term', 'pose_term', 'rate_term'
    rd_seg = rd_d.get("seg_term", rd_d.get("seg"))
    rd_pose = rd_d.get("pose_term", rd_d.get("pose"))
    rd_rate = rd_d.get("rate_term", rd_d.get("rate"))
    assert math.isclose(sg_d.seg_term, rd_seg, rel_tol=1e-9, abs_tol=1e-9), label
    assert math.isclose(sg_d.pose_term, rd_pose, rel_tol=1e-6, abs_tol=1e-9), label
    assert math.isclose(sg_d.rate_term, rd_rate, rel_tol=1e-9, abs_tol=1e-9), label


def test_torch_autograd_marginal_matches_analytic() -> None:
    """The torch RD's autograd of contest_score must match my closed-form gradient.

    This is the strongest "they agree" test: differentiate the torch
    function via autograd and confirm it equals my analytic formula at
    a non-singular point.
    """
    d_seg = torch.tensor(6.7e-4, requires_grad=True)
    d_pose = torch.tensor(3.4e-5, requires_grad=True)
    archive_bytes = torch.tensor(178258.0, requires_grad=True)
    score = rd_contest_score(
        seg_distortion=d_seg,
        pose_distortion=d_pose,
        archive_bytes=archive_bytes,
    )
    score.backward()

    sg_grad = sg_gradient(d_seg=6.7e-4, d_pose=3.4e-5)
    assert math.isclose(d_seg.grad.item(), sg_grad.d_seg, rel_tol=1e-6)
    assert math.isclose(d_pose.grad.item(), sg_grad.d_pose, rel_tol=1e-5)
    # Bytes axis is constant 25/N_REF ~ 6.66e-7. Torch autograd uses
    # float32 scalars by default, rounding at ~1e-7 absolute. Use rel_tol
    # large enough to absorb that without hiding actual divergence.
    assert math.isclose(archive_bytes.grad.item(), sg_grad.d_bytes, rel_tol=1e-6)
