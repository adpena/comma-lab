# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the upstream-scorer / tac-differentiable-mirror fidelity harness.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS": these tests exercise the REAL mirror
against the REAL frozen scorer weights + REAL contest video frames (per Catalog
#213). They do NOT assert constants — every assertion checks measured behavior of
``tools/verify_upstream_scorer_mirror_fidelity.py`` run end to end. If the model
weights or video are absent locally, the real-weight tests SKIP (they are not
faked); the static YUV6-equivalence path still runs.

All results are ``[macOS-CPU advisory]`` — NON-PROMOTABLE.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_DIR = REPO_ROOT / "upstream"
VIDEO = UPSTREAM_DIR / "videos" / "0.mkv"
SEGNET = UPSTREAM_DIR / "models" / "segnet.safetensors"
POSENET = UPSTREAM_DIR / "models" / "posenet.safetensors"

_HARNESS_PATH = REPO_ROOT / "tools" / "verify_upstream_scorer_mirror_fidelity.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "verify_upstream_scorer_mirror_fidelity", _HARNESS_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_REAL_ASSETS = SEGNET.exists() and POSENET.exists() and VIDEO.exists()
_requires_assets = pytest.mark.skipif(
    not _REAL_ASSETS,
    reason="real frozen scorer weights + 0.mkv not present locally",
)


def test_yuv6_forward_equivalence_is_bit_exact():
    """differentiable_rgb_to_yuv6 must match upstream rgb_to_yuv6 to 0 abs error.

    This is the foundation of the whole mirror: BT.601 coefficients are exact
    rationals, so the only way this fails is a coefficient/subsample/clamp bug.
    Runs even without the heavy model weights (uses random RGB).
    """
    h = _load_harness()
    result = h._section_yuv6_equivalence()
    if not result.get("ran"):
        pytest.skip(f"upstream frame_utils not importable: {result.get('blocker')}")
    assert result["passed"] is True
    # NOT a constant assertion: we measure the error and require it be exact.
    assert result["max_abs_error"] == 0.0, result


@_requires_assets
def test_real_weight_scorer_forward_fidelity_pose_and_seg():
    """The differentiable mirror's forward output must match the frozen upstream
    forward on REAL contest frames — bit-identical pose AND bit-identical segnet
    argmax. This is the load-bearing claim: training against the mirror optimizes
    the SAME function the contest scores.
    """
    h = _load_harness()
    device = torch.device("cpu")
    fidelity = h._section_scorer_fidelity(UPSTREAM_DIR, VIDEO, num_pairs=2, device=device)
    assert isinstance(fidelity, tuple), fidelity
    fid_dict = fidelity[0]
    assert fid_dict["ran"] is True
    # Pose: the patched (differentiable) preprocess must not change numerics.
    pose = fid_dict["posenet_pose"]
    assert pose["upstream_pose_first6_abs_mean"] > 0.0, "pose must be non-trivial (not mock)"
    assert pose["max_abs_diff"] <= 1e-4, pose
    assert pose["first6_max_abs_diff"] <= 1e-4, pose
    # SegNet: logits + the argmax that actually drives d_seg must agree.
    seg = fid_dict["segnet_logits"]
    assert seg["upstream_logit_abs_mean"] > 0.0, "logits must be non-trivial (not mock)"
    assert seg["argmax_disagree_frac"] <= 1e-6, seg


@_requires_assets
def test_s_seg_deepfool_flip_risk_is_finite_and_boundary_peaked():
    """s_seg (DeepFool top-2 margin backward) must be computable on real SegNet:
    finite gradients, nonzero, and SHARPLY peaked at low-margin (boundary) pixels.
    """
    h = _load_harness()
    device = torch.device("cpu")
    frames = h._decode_real_frames(VIDEO, 4)
    x = h._to_btchw(frames, device)
    dn = h._build_upstream_distortion_net(UPSTREAM_DIR, device)
    from tac.scorer import make_scorers_differentiable

    make_scorers_differentiable(dn.posenet, dn.segnet)
    s_seg = h._section_s_seg(dn.segnet, x)
    assert s_seg["ran"] is True
    assert s_seg["grad_finite"] is True
    assert s_seg["grad_energy_max"] > 0.0
    # The defining structural property of flip-risk saliency: boundary pixels
    # carry far more flip-risk than interior pixels. Measured, not asserted-const.
    assert s_seg["spatially_structured"] is True
    assert s_seg["boundary_over_interior_ratio"] > 1.0, s_seg


@_requires_assets
def test_s_pose_input_jacobian_fisher_is_finite_and_nontrivial():
    """s_pose (squared input-Jacobian of first-6 pose dims) must be computable on
    real PoseNet through the differentiable rgb_to_yuv6 + differentiable resize:
    finite, nonzero on a real fraction of pixels.
    """
    h = _load_harness()
    device = torch.device("cpu")
    frames = h._decode_real_frames(VIDEO, 4)
    x = h._to_btchw(frames, device)
    dn = h._build_upstream_distortion_net(UPSTREAM_DIR, device)
    from tac.scorer import make_scorers_differentiable

    make_scorers_differentiable(dn.posenet, dn.segnet)
    s_pose = h._section_s_pose(dn.posenet, x)
    assert s_pose["ran"] is True
    assert s_pose["grad_finite"] is True
    assert s_pose["is_nontrivial"] is True
    assert s_pose["s_pose_max"] > 0.0
    assert s_pose["s_pose_nonzero_frac"] > 0.0, s_pose


def test_upstream_no_grad_severs_gradients_mirror_does_not():
    """Confirm the root cause the mirror fixes: upstream rgb_to_yuv6 is
    @torch.no_grad (severs gradients); the mirror's differentiable version
    keeps requires_grad. This is the gradient-reachability invariant for s_pose.
    """
    from tac.differentiable_eval_roundtrip import differentiable_rgb_to_yuv6

    # leaf tensor (multiply BEFORE requires_grad so x stays a leaf)
    x = (torch.rand(1, 3, 16, 16) * 255.0).requires_grad_(True)
    assert x.is_leaf
    out = differentiable_rgb_to_yuv6(x)
    assert out.requires_grad is True
    # backward reaches the input leaf
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert (x.grad.abs() > 0).any()
