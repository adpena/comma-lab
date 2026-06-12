# SPDX-License-Identifier: MIT
"""Scorer contexts for the torch-vehicle driver.

Two implementations of the :class:`tac.torch_vehicle.driver.ScorerContext`
protocol:

* :class:`RealScorerContext` — production. Binds the REAL frozen contest
  SegNet/PoseNet via the vendored ``data.precompute_targets`` (GT decoded ONLY
  via ``frame_utils.yuv420_to_rgb`` — the canonical GT path; PyAV rgb24 is
  FORBIDDEN per CLAUDE.md as it manufactures ~100× phantom pose) and the
  vendored ``score.evaluate_decoder`` / ``score.compute_score`` for the BEST-by-
  canonical-score exact eval. The seg/pose forward matches ``common.py`` 1:1.
* :class:`SyntheticScorerContext` — test. A tiny deterministic frozen scorer +
  random GT targets so the resume round-trip (architecture-AGNOSTIC) is fast.
  This mirrors the MLX ``test_capstone_vq_nerv._build_capstone_setup`` synthetic-
  scorer fixture; it is RESEARCH-ONLY (``research_only=True``) and NEVER a score
  claim — its only job is to exercise the driver's STATE serialization.

Authority: real context = torch-CPU TRUSTED (NO MPS). The in-loop d_seg/d_pose
are ``[contest-CPU advisory]`` NON-PROMOTABLE until the byte-closed archive is
run through ``upstream/evaluate.py``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class RealScorerContext:
    """Production scorer context: real frozen SegNet/PoseNet + canonical GT.

    Precomputes the per-pair ``seg_targets_hard`` + ``pose_targets`` once (the
    vendored ``precompute_targets``), holds the frozen ``distortion_net``, and
    routes the per-step forward + the BEST-tracking exact eval through the
    vendored primitives unchanged.
    """

    research_only = False

    def __init__(self, video_path: str | Path, device: str = "cpu"):
        if str(device).lower().startswith("mps"):
            raise ValueError("MPS is NEVER trusted (CLAUDE.md). Use 'cpu' (TRUSTED) or 'cuda'.")
        self.device = torch.device(device)
        self.video_path = Path(video_path)
        from tac.torch_vehicle.vendored_imports import import_vendored

        self._data = import_vendored("data")  # applies the differentiable-yuv6 patch
        self._score = import_vendored("score")
        (
            self.distortion_net,
            self.seg_targets_hard,
            self.pose_targets,
            _gt_half,
            self.n_pairs,
        ) = self._data.precompute_targets(self.video_path, self.device)

    def seg_pose_forward(self, decoded_bhwc: torch.Tensor):
        """Run the real frozen scorer on roundtripped decoder output, 1:1 with
        ``common.py:187-189``."""
        posenet_in, segnet_in = self.distortion_net.preprocess_input(decoded_bhwc)
        seg_out = self.distortion_net.segnet(segnet_in)
        pose_out = self.distortion_net.posenet(posenet_in)
        return seg_out, pose_out["pose"][:, :6]

    def exact_eval(self, ema_decoder: nn.Module, ema_latents: torch.Tensor, archive_bytes: int):
        """Canonical d_seg/d_pose/rate/score on the EMA shadow (BEST tracker).

        Routes through the vendored ``evaluate_decoder`` (streams GT via
        ``yuv420_to_rgb``) + ``compute_score`` (the official metric). The
        archive_bytes is the BYTE-CLOSED size from the build_archive the driver
        already produced (so the rate term is the real archive bytes)."""
        tvb = self._score.total_video_bytes(self.video_path)
        dist = self._score.evaluate_decoder(
            ema_decoder,
            ema_latents.to(self.device),
            self.distortion_net,
            self.video_path,
            batch_pairs=8,
            device=self.device,
        )
        result = self._score.compute_score(
            dist["seg_distortion"], dist["pose_distortion"], archive_bytes, tvb
        )
        return {
            "seg_distortion": result["seg_distortion"],
            "pose_distortion": result["pose_distortion"],
            "rate": result["rate"],
            "score": result["score"],
        }


class _TinyFrozenScorer(nn.Module):
    """A tiny deterministic 'frozen scorer' for the resume round-trip test.

    NOT the contest scorer — a fixed-weight conv that maps roundtripped frames to
    5-class seg logits + a 6-dim pose, just enough that the driver's loss has a
    real gradient to the decoder. Frozen (requires_grad=False). Deterministic
    given a seed.
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # seg head: 3->5 over the 384x512 last-frame (downsampled by the driver's
        # preprocess is skipped here; we operate on the raw 384x512x3).
        self.seg = nn.Conv2d(3, 5, 3, padding=1)
        self.pose = nn.Linear(3, 6)
        for p in self.parameters():
            p.data = torch.empty_like(p).uniform_(-0.05, 0.05, generator=g)
            p.requires_grad_(False)


class SyntheticScorerContext:
    """Test scorer context — fast, deterministic, RESEARCH-ONLY (no score claim).

    Provides the same interface the driver needs; the resume round-trip is
    architecture-agnostic so this exercises STATE serialization without the real
    EfficientNet scorer. Mirrors the MLX synthetic-scorer fixture.
    """

    research_only = True

    def __init__(self, n_pairs: int = 6, device: str = "cpu", seed: int = 0):
        self.device = torch.device(device)
        self.n_pairs = int(n_pairs)
        self._scorer = _TinyFrozenScorer(seed=seed).to(self.device).eval()
        g = torch.Generator().manual_seed(seed + 1)
        # GT targets: random-but-fixed seg labels (0..4) over 384x512, pose 6-d.
        self.seg_targets_hard = torch.randint(
            0, 5, (self.n_pairs, 384, 512), generator=g
        ).to(self.device)
        self.pose_targets = torch.empty(self.n_pairs, 6).uniform_(-1, 1, generator=g).to(self.device)

    def seg_pose_forward(self, decoded_bhwc: torch.Tensor):
        """decoded_bhwc: (B, 2, 384, 512, 3) float [0,255]. Use last frame for seg
        (matching the contest's last-frame seg), both-frame mean for pose."""
        last = decoded_bhwc[:, -1].permute(0, 3, 1, 2) / 255.0  # (B,3,384,512)
        seg_out = self._scorer.seg(last)  # (B,5,384,512)
        # pose from a cheap global pool of the pair mean.
        pooled = decoded_bhwc.mean(dim=(1, 2, 3)) / 255.0  # (B,3)
        pose_pred6 = self._scorer.pose(pooled)  # (B,6)
        return seg_out, pose_pred6

    def exact_eval(self, ema_decoder: nn.Module, ema_latents: torch.Tensor, archive_bytes: int):
        """A deterministic synthetic 'score' from the EMA shadow argmax-flip vs
        targets + a byte rate. RESEARCH-ONLY — not a contest score."""
        with torch.inference_mode():
            n = min(self.n_pairs, ema_latents.shape[0])
            z = ema_latents[:n].to(self.device)
            decoded = ema_decoder(z)  # (n,2,3,384,512)
            last = decoded[:, -1] / 255.0  # (n,3,384,512)
            seg_logits = self._scorer.seg(last)
            d_seg = (
                (seg_logits.argmax(dim=1) != self.seg_targets_hard[:n]).float().mean().item()
            )
            pooled = decoded.mean(dim=(1, 3, 4)) / 255.0  # (n,3)... careful dims
            # decoded is (n,2,3,384,512); mean over frames+spatial -> (n,3)
            pooled = decoded.mean(dim=(1, 3, 4))[:, :3] / 255.0
            pose_pred = self._scorer.pose(pooled)
            d_pose = torch.nn.functional.mse_loss(pose_pred, self.pose_targets[:n]).item()
        rate = archive_bytes / 37_545_489.0
        score = 100.0 * d_seg + (10.0 * d_pose + 1e-12) ** 0.5 + 25.0 * rate
        return {
            "seg_distortion": d_seg,
            "pose_distortion": d_pose,
            "rate": rate,
            "score": score,
        }


__all__ = ["RealScorerContext", "SyntheticScorerContext"]
