"""Shared real-scorer regression helpers for substrate score-aware losses.

These helpers defend the WWW4-class scorer-shape bug without paying the full
contest model-loading cost. They use the real upstream ``SegNet`` class with
random weights so the SMP/Unet 4D input contract is exercised, and pair it with
an upstream-contract PoseNet stand-in that keeps the test CPU-fast.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

# File lives at <repo>/src/tac/substrates/_shared/score_aware_loss_real_scorer_test_kit.py
# parents[0]=_shared, [1]=substrates, [2]=tac, [3]=src, [4]=<repo>.
REPO_ROOT = Path(__file__).resolve().parents[4]
UPSTREAM_DIR = REPO_ROOT / "upstream"
VIDEO_PATH = UPSTREAM_DIR / "videos" / "0.mkv"


def _upstream_importable() -> bool:
    try:  # pragma: no cover - environment availability gate
        if str(UPSTREAM_DIR) not in sys.path:
            sys.path.insert(0, str(UPSTREAM_DIR))
        import upstream.modules  # noqa: F401
    except Exception:
        return False
    return True


def _smp_importable() -> bool:
    try:  # pragma: no cover - environment availability gate
        import segmentation_models_pytorch  # noqa: F401
    except Exception:
        return False
    return True


def _pyav_importable() -> bool:
    try:  # pragma: no cover - environment availability gate
        import av  # noqa: F401
    except Exception:
        return False
    return True


def skip_unless_real_scorer_stack_present() -> None:
    """Skip when the optional upstream scorer dependency stack is absent."""

    if not (_upstream_importable() and _smp_importable()):
        pytest.skip("upstream.modules or segmentation_models_pytorch unavailable")


def _load_upstream_modules() -> Any:
    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    import upstream.modules as upstream_modules

    return upstream_modules


def _freeze(module: Any) -> Any:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def _build_seg_scorer() -> Any:
    upstream_modules = _load_upstream_modules()
    return _freeze(upstream_modules.SegNet())


def _build_pose_standin() -> Any:
    import torch
    import torch.nn as nn

    class UpstreamLikePose(nn.Module):
        def preprocess_input(self, x_btchw: torch.Tensor) -> torch.Tensor:
            b, t, c, h, w = x_btchw.shape
            assert c == 3, f"expected 3 RGB input channels, got {c}"
            flat = x_btchw.reshape(b * t, c, h, w).mean(dim=1, keepdim=True)
            flat6 = flat.expand(-1, 6, -1, -1)
            half_h = h // 2
            half_w = w // 2
            flat6_sub = flat6[..., : half_h * 2, : half_w * 2].reshape(
                b * t,
                6,
                half_h,
                2,
                half_w,
                2,
            ).mean(dim=(3, 5))
            return flat6_sub.reshape(b, t * 6, half_h, half_w)

        def forward(self, x_b12hw: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"pose": x_b12hw.flatten(2).mean(dim=2)}

    return _freeze(UpstreamLikePose())


def _random_context() -> dict[str, Any]:
    import torch

    torch.manual_seed(0)
    b = 1
    return {
        "rgb_0": (torch.rand(b, 3, 64, 96) * 255.0).requires_grad_(True),
        "rgb_1": (torch.rand(b, 3, 64, 96) * 255.0).requires_grad_(True),
        "gt_0": torch.rand(b, 3, 64, 96) * 255.0,
        "gt_1": torch.rand(b, 3, 64, 96) * 255.0,
        "bytes_proxy": torch.tensor(100_000.0),
    }


def _decoded_context() -> dict[str, Any]:
    import av
    import torch
    import torch.nn.functional as F

    if not VIDEO_PATH.is_file():
        pytest.skip(f"contest video unavailable: {VIDEO_PATH}")
    container = av.open(str(VIDEO_PATH))
    try:
        frames = []
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgb24")
            frames.append(torch.from_numpy(arr).permute(2, 0, 1).float())
            if len(frames) == 2:
                break
    finally:
        container.close()
    if len(frames) != 2:
        pytest.skip("could not decode two frames from upstream/videos/0.mkv")

    rgb_0 = F.interpolate(
        frames[0].unsqueeze(0),
        size=(64, 96),
        mode="bilinear",
        align_corners=False,
    ).detach().clone().requires_grad_(True)
    rgb_1 = F.interpolate(
        frames[1].unsqueeze(0),
        size=(64, 96),
        mode="bilinear",
        align_corners=False,
    ).detach().clone().requires_grad_(True)
    return {
        "rgb_0": rgb_0,
        "rgb_1": rgb_1,
        "gt_0": rgb_0.detach().clone(),
        "gt_1": rgb_1.detach().clone(),
        "bytes_proxy": torch.tensor(100_000.0),
    }


def assert_loss_runs_on_real_segnet(
    *,
    loss_factory: Callable[[Any, Any], Any],
    invoke_loss: Callable[[Any, dict[str, Any]], tuple[Any, dict[str, Any]]],
    extra_kwargs_factory: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    """Assert a substrate loss runs through real SegNet and backpropagates."""

    import torch

    skip_unless_real_scorer_stack_present()
    ctx = _random_context()
    ctx["extra"] = extra_kwargs_factory(ctx) if extra_kwargs_factory is not None else {}
    loss_fn = loss_factory(_build_seg_scorer(), _build_pose_standin())
    loss, parts = invoke_loss(loss_fn, ctx)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert isinstance(parts, dict)
    loss.backward()
    assert ctx["rgb_0"].grad is not None
    assert ctx["rgb_1"].grad is not None
    assert ctx["rgb_0"].grad.abs().sum().item() > 0
    assert ctx["rgb_1"].grad.abs().sum().item() > 0


def assert_loss_runs_on_pyav_decoded_pair(
    *,
    loss_factory: Callable[[Any, Any], Any],
    invoke_loss: Callable[[Any, dict[str, Any]], tuple[Any, dict[str, Any]]],
    extra_kwargs_factory: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    """Assert a substrate loss consumes a real decoded contest-video pair."""

    import torch

    skip_unless_real_scorer_stack_present()
    if not _pyav_importable():
        pytest.skip("pyav unavailable")
    ctx = _decoded_context()
    ctx["extra"] = extra_kwargs_factory(ctx) if extra_kwargs_factory is not None else {}
    loss_fn = loss_factory(_build_seg_scorer(), _build_pose_standin())
    loss, parts = invoke_loss(loss_fn, ctx)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert isinstance(parts, dict)
