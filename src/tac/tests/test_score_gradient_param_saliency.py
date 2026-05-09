"""Tests for ``tac.score_gradient_param_saliency``.

The full pipeline test (decoder + scorers + GT video) is GPU-bound and runs
~25 min on M5 Max CPU; we keep that under the integration-only path. Unit
tests here cover the contract: shape/key invariants, deterministic output
on a tiny stub model, and validation of input shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _make_tiny_decoder() -> torch.nn.Module:
    """A tiny decoder that maps (B, 4) -> (B, 2, 3, 8, 8) for the purpose
    of exercising the score-gradient-saliency forward+backward."""

    class TinyDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = torch.nn.Linear(4, 6 * 8 * 8)
            self.refine = torch.nn.Conv2d(6, 6, 3, padding=1)
            self.rgb0 = torch.nn.Conv2d(6, 3, 1)
            self.rgb1 = torch.nn.Conv2d(6, 3, 1)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            B = z.shape[0]
            x = self.stem(z).view(B, 6, 8, 8)
            x = torch.tanh(self.refine(x))
            f0 = torch.sigmoid(self.rgb0(x)) * 255.0
            f1 = torch.sigmoid(self.rgb1(x)) * 255.0
            return torch.stack([f0, f1], dim=1)

    return TinyDecoder()


def _make_stub_scorer(out_dim: int = 16) -> torch.nn.Module:
    """A stub scorer that mirrors PoseNet's API: preprocess + forward dict."""

    class StubScorer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = torch.nn.Linear(3 * 4 * 4, out_dim)

        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 2, 3, H, W) → flatten + downsample by mean pool
            B = x.shape[0]
            x_flat = x.reshape(B * 2, 3, x.shape[-2], x.shape[-1]).mean(dim=(-2, -1))  # (B*2, 3)
            # Tile to (B, 3, 4, 4) just to give the head a plausible input
            return x_flat.reshape(B, 2 * 3)[:, :3].unsqueeze(-1).unsqueeze(-1).expand(B, 3, 4, 4)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            B = x.shape[0]
            return {"pose": self.head(x.reshape(B, -1))}

    return StubScorer()


def _make_stub_segnet(n_classes: int = 5) -> torch.nn.Module:
    """A stub segnet: (B, 2, 3, H, W) -> logits (B, n_classes, H', W')."""

    class StubSegNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, n_classes, 3, padding=1)

        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            # Take the LAST frame, like upstream SegNet
            return x[:, -1, ...]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    return StubSegNet()


def test_compute_score_gradient_saliency_returns_one_scalar_per_param() -> None:
    from tac.score_gradient_param_saliency import compute_score_gradient_param_saliency

    torch.manual_seed(0)
    decoder = _make_tiny_decoder().eval()
    posenet = _make_stub_scorer().eval()
    segnet = _make_stub_segnet().eval()

    n_pairs = 4
    latents = torch.randn(n_pairs, 4)
    # GT pairs: small synthetic frames at the upstream camera shape (we
    # don't actually pass through real frame_utils here; the stub scorers
    # don't care about the camera-size assertion). Use the same shape as
    # decoded output to keep the surrogate-loss math consistent.
    gt_pairs = (torch.rand(n_pairs, 2, 8, 8, 3) * 255.0).to(torch.uint8)

    # NOTE: the helper assumes camera_h/camera_w = 874/1164 by default. Pass
    # tiny values so we don't blow up memory in the test.
    saliency = compute_score_gradient_param_saliency(
        decoder=decoder,
        latents=latents,
        gt_pairs_uint8=gt_pairs,
        posenet=posenet,
        segnet=segnet,
        device="cpu",
        eval_h=8,
        eval_w=8,
        camera_h=8,
        camera_w=8,
        forward_batch_size=2,
        saliency_batch_size=2,
        progress=False,
    )

    expected_keys = {name for name, _ in decoder.named_parameters()}
    assert set(saliency.keys()) == expected_keys
    # Every value is a positive finite float.
    for k, v in saliency.items():
        assert isinstance(v, float), k
        assert v >= 0.0, k
        assert v < float("inf"), k


def test_pose_distortion_score_derivative_matches_contest_formula() -> None:
    from tac.score_gradient_param_saliency import (
        DEFAULT_CPU_POSE_SCORE_WEIGHT,
        pose_distortion_score_derivative,
    )

    # d sqrt(10 * d) / dd = sqrt(10) / (2 * sqrt(d)).
    assert pose_distortion_score_derivative(3.0e-5) == pytest.approx(288.6751345948129)
    assert DEFAULT_CPU_POSE_SCORE_WEIGHT == pytest.approx(
        pose_distortion_score_derivative(3.0e-5)
    )


def test_compute_score_gradient_saliency_rejects_mismatched_shapes() -> None:
    from tac.score_gradient_param_saliency import compute_score_gradient_param_saliency

    decoder = _make_tiny_decoder().eval()
    posenet = _make_stub_scorer().eval()
    segnet = _make_stub_segnet().eval()
    # latents has 4 pairs but gt has 2 — must raise.
    latents = torch.randn(4, 4)
    gt_pairs = (torch.rand(2, 2, 8, 8, 3) * 255.0).to(torch.uint8)
    try:
        compute_score_gradient_param_saliency(
            decoder=decoder,
            latents=latents,
            gt_pairs_uint8=gt_pairs,
            posenet=posenet,
            segnet=segnet,
            eval_h=8, eval_w=8, camera_h=8, camera_w=8,
            forward_batch_size=2, saliency_batch_size=2,
        )
    except ValueError as exc:
        assert "shape[0]" in str(exc) or "latents" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("shape mismatch should raise")


def test_compute_score_gradient_saliency_reproducible_for_fixed_seed() -> None:
    """Two runs with the same seed produce identical saliency values."""
    from tac.score_gradient_param_saliency import compute_score_gradient_param_saliency

    def _run() -> dict:
        torch.manual_seed(42)
        decoder = _make_tiny_decoder().eval()
        posenet = _make_stub_scorer().eval()
        segnet = _make_stub_segnet().eval()
        latents = torch.randn(4, 4)
        gt = (torch.rand(4, 2, 8, 8, 3) * 255.0).to(torch.uint8)
        return compute_score_gradient_param_saliency(
            decoder=decoder,
            latents=latents,
            gt_pairs_uint8=gt,
            posenet=posenet,
            segnet=segnet,
            eval_h=8, eval_w=8, camera_h=8, camera_w=8,
            forward_batch_size=2, saliency_batch_size=2,
        )

    a = _run()
    b = _run()
    assert set(a) == set(b)
    for k in a:
        assert abs(a[k] - b[k]) < 1e-10, f"{k}: {a[k]} != {b[k]}"


def test_score_gradient_saliency_is_batch_size_invariant() -> None:
    """Fisher saliency must average per-sample squared gradients.

    A pre-fix implementation squared the batch-averaged gradient, so opposite
    signed sample gradients could cancel before squaring and the output changed
    with ``saliency_batch_size``. This is the Track 4 v1 bug class.
    """
    from tac.score_gradient_param_saliency import compute_score_gradient_param_saliency

    torch.manual_seed(123)
    decoder = _make_tiny_decoder().eval()
    posenet = _make_stub_scorer().eval()
    segnet = _make_stub_segnet().eval()
    latents = torch.randn(4, 4)
    gt = (torch.rand(4, 2, 8, 8, 3) * 255.0).to(torch.uint8)

    saliency_one = compute_score_gradient_param_saliency(
        decoder=decoder,
        latents=latents,
        gt_pairs_uint8=gt,
        posenet=posenet,
        segnet=segnet,
        eval_h=8,
        eval_w=8,
        camera_h=8,
        camera_w=8,
        forward_batch_size=2,
        saliency_batch_size=1,
    )
    saliency_four = compute_score_gradient_param_saliency(
        decoder=decoder,
        latents=latents,
        gt_pairs_uint8=gt,
        posenet=posenet,
        segnet=segnet,
        eval_h=8,
        eval_w=8,
        camera_h=8,
        camera_w=8,
        forward_batch_size=2,
        saliency_batch_size=4,
    )

    assert set(saliency_one) == set(saliency_four)
    for key in saliency_one:
        assert saliency_four[key] == pytest.approx(saliency_one[key], rel=0.0, abs=1e-10)
