# SPDX-License-Identifier: MIT
"""[C1] TorchScorerBridge fails closed when the pose path's yuv6 is not patched.

The real upstream ``PoseNet.preprocess_input`` (``upstream/modules.py:74``) calls
the module-level ``modules.rgb_to_yuv6``, which is ``@torch.no_grad()`` / in-place
upstream — so the pose pixel-gradient is SEVERED unless
``patch_upstream_yuv6_globally`` swapped in the differentiable version. The bridge
asserts the patch is in place (for the REAL upstream PoseNet only) so a caller with
an un-patched scorer gets a clear error, not a silently pose-inert loss.

These tests verify:
  * building the bridge over a REAL upstream PoseNet UN-PATCHED raises;
  * building it over the PATCHED production scorer passes;
  * a proto/stand-in PoseNet (no global dependency) is EXEMPT (no false positive).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


def _build_proto_bridge_with_pose():
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    class _Pose(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(12, 6)

        def forward(self, x):
            return {"pose": self.l(x.mean(dim=(2, 3)))}

    class _Seg(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 5, 1)

        def forward(self, x):
            return self.c(x)

    class _DN(nn.Module):
        def __init__(self):
            super().__init__()
            self.segnet = _Seg()
            self.posenet = _Pose()

        def preprocess_input(self, bhwc):
            last = bhwc[:, -1].permute(0, 3, 1, 2)
            first = bhwc[:, 0].permute(0, 3, 1, 2)
            pose_in = torch.cat([first.repeat(1, 2, 1, 1), last.repeat(1, 2, 1, 1)], dim=1)
            return pose_in, last

    dn = _DN().eval()
    for p in dn.parameters():
        p.requires_grad = False
    seg_t = torch.zeros(3, 48, 64, dtype=torch.long)
    pose_t = torch.zeros(3, 6, dtype=torch.float32)
    return TorchScorerBridge(dn, seg_t, pose_t, seg_loss_form="ce_seg_loss")


def test_c1_proto_posenet_is_exempt_from_yuv6_assertion():
    """A proto/stand-in PoseNet has no global yuv6 dependency -> bridge builds fine."""
    bridge = _build_proto_bridge_with_pose()  # must NOT raise
    assert bridge.pose_enabled is True


def test_c1_assertion_helper_detects_real_upstream_posenet_only():
    """``_posenet_routes_through_upstream_yuv6`` is True only for the real upstream PoseNet."""
    from tac.differentiable_eval_roundtrip import _resolve_upstream_modules
    from tac.mlx_pr95_port.score_bridge import (
        _posenet_routes_through_upstream_yuv6,
    )

    _fu, modules = _resolve_upstream_modules()
    if modules is None or not hasattr(modules, "PoseNet"):
        pytest.skip("upstream modules not importable in this environment")

    # a proto net is NOT the upstream PoseNet.
    class _ProtoDN:
        posenet = nn.Linear(2, 2)

    assert _posenet_routes_through_upstream_yuv6(_ProtoDN()) is False

    # the REAL upstream PoseNet IS detected.
    class _RealDN:
        posenet = modules.PoseNet()

    assert _posenet_routes_through_upstream_yuv6(_RealDN()) is True


def test_c1_real_upstream_posenet_unpatched_raises():
    """Building the bridge over the REAL upstream PoseNet UN-PATCHED fails closed."""
    from tac.differentiable_eval_roundtrip import (
        _resolve_upstream_modules,
        differentiable_rgb_to_yuv6,
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge, Yuv6NotPatchedError

    _fu, modules = _resolve_upstream_modules()
    if modules is None or not hasattr(modules, "PoseNet"):
        pytest.skip("upstream modules not importable in this environment")

    # Ensure UN-patched state for this test (restore afterwards).
    saved = modules.rgb_to_yuv6
    try:
        # force the upstream (non-differentiable) function back if currently patched.
        from tac.differentiable_eval_roundtrip import _resolve_upstream_modules as _rum

        _fu2, _m2 = _rum()
        # if frame_utils is importable, use its original rgb_to_yuv6 as the un-patched.
        import importlib

        try:
            frame_utils = importlib.import_module("frame_utils")
            upstream_fn = frame_utils.rgb_to_yuv6
            # if frame_utils itself is patched, we cannot get the true original; only
            # run the negative test when the un-patched function is NOT differentiable.
        except Exception:
            upstream_fn = None

        if upstream_fn is None or upstream_fn is differentiable_rgb_to_yuv6:
            pytest.skip("cannot establish an un-patched upstream yuv6 in this env")

        modules.rgb_to_yuv6 = upstream_fn  # un-patched (severs pose grad)
        assert modules.rgb_to_yuv6 is not differentiable_rgb_to_yuv6

        dn = modules.DistortionNet() if hasattr(modules, "DistortionNet") else None
        if dn is None:
            # construct a minimal real-PoseNet holder.
            class _RealDN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.posenet = modules.PoseNet()
                    self.segnet = nn.Conv2d(3, 5, 1)

                def preprocess_input(self, bhwc):
                    last = bhwc[:, -1].permute(0, 3, 1, 2)
                    return last, last

            dn = _RealDN()
        for p in dn.parameters():
            p.requires_grad = False

        seg_t = torch.zeros(2, 48, 64, dtype=torch.long)
        pose_t = torch.zeros(2, 6, dtype=torch.float32)
        with pytest.raises(Yuv6NotPatchedError):
            TorchScorerBridge(dn, seg_t, pose_t, seg_loss_form="ce_seg_loss")

        # and PATCHING fixes it: the bridge builds.
        token = patch_upstream_yuv6_globally()
        try:
            TorchScorerBridge(dn, seg_t, pose_t, seg_loss_form="ce_seg_loss")
        finally:
            unpatch_upstream_yuv6(token)
    finally:
        modules.rgb_to_yuv6 = saved


def test_c1_pose_disabled_bridge_never_asserts():
    """A pose-DISABLED bridge (pose_targets=None) skips the yuv6 assertion entirely."""
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    class _Seg(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 5, 1)

        def forward(self, x):
            return self.c(x)

    class _DN(nn.Module):
        def __init__(self):
            super().__init__()
            self.segnet = _Seg()
            self.posenet = None

        def preprocess_input(self, bhwc):
            last = bhwc[:, -1].permute(0, 3, 1, 2)
            return None, last

    dn = _DN().eval()
    for p in dn.parameters():
        p.requires_grad = False
    seg_t = torch.zeros(2, 48, 64, dtype=torch.long)
    bridge = TorchScorerBridge(dn, seg_t, None, seg_loss_form="ce_seg_loss")
    assert bridge.pose_enabled is False
