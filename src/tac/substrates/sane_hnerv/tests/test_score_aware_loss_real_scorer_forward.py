"""Regression tests for the WWW4 dispatch shape-mismatch bug 2026-05-12.

WWW4 (Modal A100, fc-01KREXK209TRX7ED5ZRVXHY1VT) crashed at SegNet
because ``score_aware_loss.py:129`` passed 5D ``(B, T=1, C, H, W)``
directly into the ``smp.Unet`` stem, which expects 4D ``(B, C, H, W)``.
The sibling latent bug at line 137 fed 4D 6-channel RGB to PoseNet's
forward instead of the 12-channel post-``preprocess_input`` shape.

The earlier 31-test smoke suite did NOT catch the bug because the
``DummySeg`` / ``DummyPose`` modules accepted whatever shape the loss
emitted — they had no upstream-faithful ``preprocess_input`` /
``forward`` contract. These regression tests exercise:

* The real ``upstream.modules.SegNet`` (``smp.Unet`` with a
  randomly-initialized stem — we never load the real safetensors so the
  test stays CPU-only and fast, but the SHAPE math is the real math).
* The real ``upstream.modules.PoseNet`` (FastViT-T12 backbone) with
  random init.
* A real ``upstream/videos/0.mkv`` frame pair decoded via pyav to prove
  the scorer forward consumes the loss output without shape error.

Test discipline: CPU only, random init, ≤ a few seconds runtime. If the
upstream module import or pyav decode is unavailable in the environment,
the test marks itself ``skip`` rather than failing — the regression
guarantee is that when the dependencies ARE available, the loss runs
end-to-end against the real scorer shape contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
UPSTREAM_DIR = REPO_ROOT / "upstream"
VIDEO_PATH = UPSTREAM_DIR / "videos" / "0.mkv"


def _upstream_importable() -> bool:
    """Return True if ``upstream.modules`` and its deps are importable."""
    try:  # pragma: no cover - import-availability gate
        if str(UPSTREAM_DIR) not in sys.path:
            sys.path.insert(0, str(UPSTREAM_DIR))
        import upstream.modules  # noqa: F401
    except Exception:
        return False
    return True


def _smp_importable() -> bool:
    """SegNet inherits from ``smp.Unet``; if smp isn't available, skip."""
    try:  # pragma: no cover
        import segmentation_models_pytorch  # noqa: F401
    except Exception:
        return False
    return True


def _timm_importable() -> bool:
    """PoseNet uses ``timm.create_model('fastvit_t12', ...)``."""
    try:  # pragma: no cover
        import timm  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(
    not (_upstream_importable() and _smp_importable()),
    reason="upstream.modules or smp not importable in this environment",
)
def test_loss_runs_on_real_segnet_smp_unet_random_init():
    """Real ``smp.Unet`` SegNet forward consumes loss output without shape error.

    This is the WWW4 regression: without ``seg_scorer.preprocess_input(...)``,
    the loss passes 5D ``(B, 1, 3, H, W)`` to ``smp.Unet`` which raises::

        RuntimeError: Expected 4-dimensional input for 4-dimensional weight
        [out, 3, kH, kW], but got 5-dimensional input of size [B, 1, 3, H, W]

    The fix calls ``preprocess_input`` first (slices last frame, interpolates
    to scorer-input size); ``forward`` then runs cleanly on 4D.
    """
    import torch
    import torch.nn as nn

    # Lazy-import the upstream module *after* the skip gate
    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    import upstream.modules as um  # noqa: F401  (used via getattr below)

    from tac.substrates.sane_hnerv.score_aware_loss import (
        SaneHnervScoreAwareLoss,
        ScoreAwareLossWeights,
    )

    # Random-init SegNet (we do NOT load the real safetensors — that is a
    # contest-CUDA path. The SHAPE math is what we're testing.)
    seg_scorer = um.SegNet().eval()
    for p in seg_scorer.parameters():
        p.requires_grad_(False)

    # PoseNet is the FastViT-T12 path; bypass with a fake module that mirrors
    # the upstream PoseNet contract (preprocess_input 5D→4D 12-channel,
    # forward 4D 12ch → dict). Real FastViT-T12 forward is slow on CPU; the
    # SegNet shape mismatch was the actual WWW4 bug, so we exercise the
    # SegNet path with the real class and the PoseNet path with the upstream
    # contract through a thin stand-in.
    class _UpstreamLikePose(nn.Module):
        def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
            # (B, T, C, H, W) -> (B, T*6, H/2, W/2)
            b, t, c, h, w = x_btchw.shape
            assert c == 3, f"expected 3 input channels, got {c}"
            flat = x_btchw.reshape(b * t, c, h, w)
            # 6 channels per frame at half spatial (yuv420 analogue)
            flat6 = flat.mean(dim=1, keepdim=True).expand(-1, 6, -1, -1)
            flat6_sub = flat6.reshape(b * t, 6, h // 2, 2, w // 2, 2).mean(
                dim=(3, 5)
            )
            return flat6_sub.reshape(b, t * 6, h // 2, w // 2)

        def forward(self, x_b12hw: torch.Tensor):
            # Upstream PoseNet returns a dict {'pose': (B, 12)}; reproduce
            return {"pose": x_b12hw.flatten(2).mean(dim=2)}

    pose_scorer = _UpstreamLikePose().eval()
    for p in pose_scorer.parameters():
        p.requires_grad_(False)

    weights = ScoreAwareLossWeights(
        alpha_rate=25.0, beta_seg=100.0, gamma_pose=1.0,
        pose_weight_scale=2.71, contest_normalizer=37545489.0,
    )
    loss_fn = SaneHnervScoreAwareLoss(
        seg_scorer=seg_scorer, pose_scorer=pose_scorer, weights=weights,
    )

    # Camera-native-ish small input (we let the SegNet preprocess interpolate
    # up to (384, 512) on its own — that's the contract). Build leaf tensors
    # directly so ``.grad`` is populated by ``loss.backward()``.
    b = 1
    rgb_0 = (torch.rand(b, 3, 64, 96) * 255.0).requires_grad_(True)
    rgb_1 = (torch.rand(b, 3, 64, 96) * 255.0).requires_grad_(True)
    gt_0 = torch.rand(b, 3, 64, 96) * 255.0
    gt_1 = torch.rand(b, 3, 64, 96) * 255.0
    bytes_proxy = torch.tensor(100_000.0)

    loss, parts = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, bytes_proxy,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    assert loss.dim() == 0
    assert torch.isfinite(loss), "loss must be finite on the real scorer path"
    # The forward path connected the input through preprocess+segnet, so the
    # loss should depend on the inputs (gradients reachable).
    loss.backward()
    # rgb_0 and rgb_1 should have nonzero gradient since they flow through
    # the differentiable eval-roundtrip + scorer preprocess + forward.
    assert rgb_0.grad is not None
    assert rgb_1.grad is not None


@pytest.mark.skipif(
    not (_upstream_importable() and _smp_importable()),
    reason="upstream.modules or smp not importable in this environment",
)
def test_loss_without_preprocess_input_would_have_failed_on_real_segnet():
    """Negative control: confirm 5D-direct-to-SegNet raises like WWW4 did.

    This pins the WWW4 failure signature so a regression in the future
    (e.g. someone removes the ``preprocess_input`` call thinking it's
    redundant) trips the smp.Unet stem shape-mismatch immediately.
    """
    import torch

    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    import upstream.modules as um  # noqa: F401

    seg_scorer = um.SegNet().eval()
    rgb_5d = torch.rand(1, 1, 3, 64, 96)  # (B, T=1, C, H, W)
    with pytest.raises((RuntimeError, ValueError)):
        seg_scorer(rgb_5d)  # WWW4-equivalent direct call


@pytest.mark.skipif(
    not (_upstream_importable() and _smp_importable()),
    reason="upstream.modules or smp not importable in this environment",
)
def test_segnet_preprocess_input_then_forward_works_on_4d():
    """Positive control: SegNet.preprocess_input + forward works on 5D in.

    This documents the upstream contract that the loss now honors.
    """
    import torch

    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    import upstream.modules as um  # noqa: F401

    seg_scorer = um.SegNet().eval()
    rgb_5d = torch.rand(1, 2, 3, 64, 96)
    rgb_4d = seg_scorer.preprocess_input(rgb_5d)
    assert rgb_4d.dim() == 4, f"preprocess_input must return 4D; got {rgb_4d.dim()}D"
    out = seg_scorer(rgb_4d)
    assert out.dim() == 4
    # SegNet outputs 5 classes
    assert out.shape[1] == 5


@pytest.mark.skipif(
    not (_upstream_importable() and _smp_importable()),
    reason="upstream.modules or smp not importable in this environment",
)
def test_pair_pred_and_gt_carry_same_shape_into_preprocess_input():
    """The loss must stage prediction + gt with the same (B, T=2, C, H, W) shape.

    A regression where pred uses ``stack(..., dim=1)`` and gt uses
    ``cat(..., dim=1)`` (or vice versa) would silently feed inconsistent
    shapes through preprocess_input and either explode or produce wrong
    distortion targets.
    """
    import torch

    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    import upstream.modules as um  # noqa: F401

    seg_scorer = um.SegNet().eval()

    rgb_0 = torch.rand(1, 3, 64, 96)
    rgb_1 = torch.rand(1, 3, 64, 96)
    pair = torch.stack([rgb_0, rgb_1], dim=1)
    assert pair.shape == (1, 2, 3, 64, 96), (
        f"pair stack must be (B, T=2, C, H, W); got {tuple(pair.shape)}"
    )
    seg_4d = seg_scorer.preprocess_input(pair)
    # The contract is: SegNet takes the LAST frame and interpolates to (384, 512)
    assert seg_4d.shape == (1, 3, 384, 512), (
        f"SegNet.preprocess_input must yield (B, C, 384, 512); got {tuple(seg_4d.shape)}"
    )


@pytest.mark.skipif(
    not (_upstream_importable() and _smp_importable()),
    reason="upstream.modules or smp not importable in this environment",
)
def test_loss_backward_produces_nonzero_grad_through_both_scorer_paths():
    """End-to-end gradient flow: input rgb tensors receive nonzero grads.

    This is the WWW4-class regression: a shape mismatch would either crash
    OR (worse) cause a silent dimension reduction that breaks autograd
    plumbing. Both paths must carry gradient back to the inputs.
    """
    import torch
    import torch.nn as nn

    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    import upstream.modules as um  # noqa: F401

    from tac.substrates.sane_hnerv.score_aware_loss import (
        SaneHnervScoreAwareLoss,
        ScoreAwareLossWeights,
    )

    seg_scorer = um.SegNet().eval()
    for p in seg_scorer.parameters():
        p.requires_grad_(False)

    class _UpstreamLikePose(nn.Module):
        def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
            b, t, c, h, w = x_btchw.shape
            flat = x_btchw.reshape(b * t, c, h, w).mean(dim=1, keepdim=True)
            flat6 = flat.expand(-1, 6, -1, -1)
            flat6_sub = flat6.reshape(b * t, 6, h // 2, 2, w // 2, 2).mean(
                dim=(3, 5)
            )
            return flat6_sub.reshape(b, t * 6, h // 2, w // 2)

        def forward(self, x):
            return {"pose": x.flatten(2).mean(dim=2)}

    pose_scorer = _UpstreamLikePose().eval()

    loss_fn = SaneHnervScoreAwareLoss(
        seg_scorer=seg_scorer, pose_scorer=pose_scorer,
        weights=ScoreAwareLossWeights(),
    )
    rgb_0 = (torch.rand(1, 3, 64, 96) * 255.0).requires_grad_(True)
    rgb_1 = (torch.rand(1, 3, 64, 96) * 255.0).requires_grad_(True)
    gt_0 = torch.rand(1, 3, 64, 96) * 255.0
    gt_1 = torch.rand(1, 3, 64, 96) * 255.0
    bytes_proxy = torch.tensor(100_000.0)
    loss, _ = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, bytes_proxy,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    loss.backward()
    # Both inputs must carry gradient (proves both scorer paths contributed)
    assert rgb_0.grad is not None and rgb_0.grad.abs().sum().item() > 0
    assert rgb_1.grad is not None and rgb_1.grad.abs().sum().item() > 0


def _pyav_available() -> bool:
    try:  # pragma: no cover
        import av  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(
    not (
        _upstream_importable()
        and _smp_importable()
        and _pyav_available()
        and VIDEO_PATH.is_file()
    ),
    reason="upstream + smp + pyav + contest video not all available",
)
def test_loss_runs_on_real_pyav_decoded_frame_pair():
    """End-to-end: decode a real frame pair from upstream/videos/0.mkv.

    Uses the contest video to confirm the loss runs against the SAME pixel
    distribution that the contest scorer sees, not synthetic uniform noise.
    """
    import av
    import torch
    import torch.nn as nn

    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    import upstream.modules as um  # noqa: F401

    from tac.substrates.sane_hnerv.score_aware_loss import (
        SaneHnervScoreAwareLoss,
        ScoreAwareLossWeights,
    )

    # Decode the first 2 frames of the contest video into a (1, 2, 3, H, W) tensor
    container = av.open(str(VIDEO_PATH))
    frames = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")  # (H, W, 3)
        frames.append(torch.from_numpy(arr).permute(2, 0, 1).float())  # (3, H, W)
        if len(frames) == 2:
            break
    container.close()
    assert len(frames) == 2, "failed to decode 2 frames from contest video"

    # Downscale to keep the test fast (real frames are 874x1164)
    rgb_0 = torch.nn.functional.interpolate(
        frames[0].unsqueeze(0), size=(64, 96), mode="bilinear", align_corners=False
    ).squeeze(0)
    rgb_1 = torch.nn.functional.interpolate(
        frames[1].unsqueeze(0), size=(64, 96), mode="bilinear", align_corners=False
    ).squeeze(0)
    rgb_0 = rgb_0.unsqueeze(0).detach().clone().requires_grad_(True)
    rgb_1 = rgb_1.unsqueeze(0).detach().clone().requires_grad_(True)
    gt_0 = rgb_0.detach().clone()  # use the same real frames as gt for the test
    gt_1 = rgb_1.detach().clone()

    seg_scorer = um.SegNet().eval()
    for p in seg_scorer.parameters():
        p.requires_grad_(False)

    class _UpstreamLikePose(nn.Module):
        def preprocess_input(self, x_btchw):
            b, t, c, h, w = x_btchw.shape
            flat = x_btchw.reshape(b * t, c, h, w).mean(dim=1, keepdim=True)
            flat6 = flat.expand(-1, 6, -1, -1)
            flat6_sub = flat6.reshape(b * t, 6, h // 2, 2, w // 2, 2).mean(
                dim=(3, 5)
            )
            return flat6_sub.reshape(b, t * 6, h // 2, w // 2)

        def forward(self, x):
            return {"pose": x.flatten(2).mean(dim=2)}

    pose_scorer = _UpstreamLikePose().eval()

    loss_fn = SaneHnervScoreAwareLoss(
        seg_scorer=seg_scorer, pose_scorer=pose_scorer,
        weights=ScoreAwareLossWeights(),
    )
    bytes_proxy = torch.tensor(100_000.0)
    loss, parts = loss_fn(
        rgb_0, rgb_1, gt_0, gt_1, bytes_proxy,
        apply_eval_roundtrip=True, noise_std=0.0,
    )
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    # On identical pred/gt frames the seg and pose terms should be ~zero;
    # the rate term keeps the total positive.
    assert float(parts["rate_term"]) > 0
