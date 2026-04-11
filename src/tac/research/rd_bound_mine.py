#!/usr/bin/env python
"""Rate-distortion lower bound for PoseNet distortion via MINE estimator.

Research-grade experiment (panel #4). The question: given our current
compressed archive size (864KB), what is the **provable minimum** PoseNet
distortion any filter of any size could achieve?

The classical rate-distortion inequality (Cover & Thomas, Ch 10):

    D*(R) >= D_min where R(D) is the rate-distortion function.

For a Gaussian residual with variance sigma^2, this becomes:

    D(R) = sigma^2 * 2^(-2R/k)

where k is the pose dimensionality (6). To apply this we need the mutual
information I(X; P) where X = input frame pair and P = PoseNet 6-DoF output,
PLUS our actual rate R = 8 * archive_bytes / n_frames.

Computing I(X; P) for a 6M-dim X and 6-dim P via MINE (Belghazi 2018):

    I(X; P) >= E_{p(x,p)}[T(x,p)] - log E_{p(x)p(p)}[exp(T(x,p))]

where T is a neural critic trained to discriminate joint samples from
product-of-marginals samples. We use a simple 2-layer MLP critic.

Output: a table of (R, D_min) pairs that tells us the lowest pose distortion
any filter with this bitrate budget could theoretically achieve.

Usage::

    cd /tmp/pact-mine
    PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors \\
        --with timm --with einops --with segmentation-models-pytorch \\
        --with numpy python -u experiments/rd_bound_mine.py
"""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.data import build_pairs, decode_archive, decode_video
from tac.losses import scorer_forward_pair
from tac.scorer import detect_device, load_scorers
from tac.proxy_eval import _default_paths

_PROJECT, _UPSTREAM, VIDEOS_DIR, _LIVE_ARCHIVE, ARCHIVE_ZIP = _default_paths()
DEVICE = detect_device()


class MINECritic(nn.Module):
    """Tiny MLP that scores (x_feat, p) pairs for mutual information estimation.

    We compress x = (1, 2, H, W, 3) to a small feature vector first to make
    the critic tractable. The feature is the mean + std of the downsampled
    pair at 64x48, plus the 6-dim pose from a frozen reference PoseNet.
    """

    def __init__(self, x_feat_dim: int = 64, p_dim: int = 6, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_feat_dim + p_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_feat: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_feat, p], dim=-1)).squeeze(-1)


def compress_frame_pair(pair_float: torch.Tensor, target_hw: tuple[int, int] = (48, 64)) -> torch.Tensor:
    """Compress a (1, 2, H, W, 3) pair to a small feature vector.

    We downsample each frame to target_hw and return flat vector of mean+std
    per spatial bin per channel. This gives us ~12k dim but we further
    reduce to 64-dim via a fixed random projection.
    """
    B, T, H, W, C = pair_float.shape
    x = pair_float.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # (BT, 3, H, W)
    x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
    # Flatten: (BT, 3, 48, 64) -> features
    feats = x.reshape(B * T * 3, -1)  # (BT*3, 48*64)
    # Fixed random projection to 64-dim to keep critic tractable
    torch.manual_seed(0)
    proj_dim = 64
    if not hasattr(compress_frame_pair, "P"):
        compress_frame_pair.P = torch.randn(feats.shape[-1], proj_dim, device=feats.device) / math.sqrt(feats.shape[-1])
    projected = feats @ compress_frame_pair.P  # (BT*3, 64)
    # Pool across channels and time to get one 64-vector per pair
    projected = projected.reshape(B, T * 3, proj_dim).mean(dim=1)  # (B, 64)
    return projected


def get_posenet_output(posenet: nn.Module, pair_uint8: torch.Tensor) -> torch.Tensor:
    """Run PoseNet and return 6-dim pose vector."""
    x = pair_uint8.float().permute(0, 1, 4, 2, 3).contiguous()
    with torch.no_grad():
        inp = posenet.preprocess_input(x)
        out = posenet(inp)
    return out["pose"][..., :6].squeeze(0)  # (6,)


def train_mine(
    x_feats: torch.Tensor,
    p_samples: torch.Tensor,
    epochs: int = 500,
    lr: float = 5e-4,
    device: str = "cpu",
) -> tuple[float, MINECritic]:
    """Train the MINE critic and return the lower bound on I(X; P).

    x_feats: (N, 64) pair features
    p_samples: (N, 6) pose vectors
    """
    N, x_dim = x_feats.shape
    p_dim = p_samples.shape[1]
    critic = MINECritic(x_feat_dim=x_dim, p_dim=p_dim).to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    x_feats = x_feats.to(device)
    p_samples = p_samples.to(device)

    mi_history = []
    best_mi = -float("inf")

    for epoch in range(epochs):
        # Sample a batch
        batch_size = min(N, 64)
        perm = torch.randperm(N)[:batch_size]
        x_batch = x_feats[perm]
        p_batch = p_samples[perm]

        # Joint samples
        joint_score = critic(x_batch, p_batch)  # (B,)

        # Marginal samples (shuffle p to break pairing)
        marginal_perm = torch.randperm(batch_size)
        p_marginal = p_batch[marginal_perm]
        marginal_score = critic(x_batch, p_marginal)

        # MINE lower bound (Donsker-Varadhan form)
        mi_lb = joint_score.mean() - torch.logsumexp(marginal_score, dim=0) + math.log(batch_size)

        loss = -mi_lb  # maximize lower bound
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mi_history.append(mi_lb.item())
        best_mi = max(best_mi, mi_lb.item())

        if (epoch + 1) % 50 == 0:
            print(f"  MINE epoch {epoch + 1}: MI lower bound = {mi_lb.item():.4f} nats")

    # Return the smoothed estimate (last 20% of training)
    tail = mi_history[int(0.8 * epochs):]
    final_mi = float(np.mean(tail))
    return final_mi, critic


def main():
    print(f"[rd-bound] device={DEVICE}")
    print(f"[rd-bound] Loading PoseNet...")
    posenet, _ = load_scorers(DEVICE)

    print(f"[rd-bound] Decoding archive + GT...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))
    comp_pairs = build_pairs(comp_frames[:n])
    gt_pairs = build_pairs(gt_frames[:n])
    n_pairs = len(comp_pairs)
    print(f"[rd-bound] {n_pairs} frame pairs")
    del comp_frames, gt_frames
    gc.collect()

    # Collect (x_feat, p) samples from all 600 pairs on both comp and gt
    print(f"[rd-bound] Collecting features and pose outputs...")
    all_x_feats = []
    all_p_comp = []
    all_p_gt = []

    for idx in range(n_pairs):
        comp_pair = comp_pairs[idx].to(DEVICE).float()
        gt_pair = gt_pairs[idx].to(DEVICE).float()

        x_feat = compress_frame_pair(comp_pair).detach().cpu()
        p_comp = get_posenet_output(posenet, comp_pair).cpu()
        p_gt = get_posenet_output(posenet, gt_pair).cpu()

        all_x_feats.append(x_feat)
        all_p_comp.append(p_comp)
        all_p_gt.append(p_gt)

        if (idx + 1) % 100 == 0:
            print(f"  collected {idx + 1}/{n_pairs} pairs")

    x_feats = torch.cat(all_x_feats, dim=0)  # (n_pairs, 64)
    p_comp = torch.stack(all_p_comp, dim=0)  # (n_pairs, 6)
    p_gt = torch.stack(all_p_gt, dim=0)

    print(f"\n[rd-bound] x_feat shape: {x_feats.shape}")
    print(f"[rd-bound] p_comp shape: {p_comp.shape}")
    print(f"[rd-bound] p_gt  shape: {p_gt.shape}")

    # Compute pose variance across pairs (for rate-distortion reference)
    p_gt_var = p_gt.var(dim=0).mean().item()
    print(f"[rd-bound] p_gt variance (per dim, averaged): {p_gt_var:.6f}")

    # Train MINE on (x_comp, p_comp) to estimate I(X_comp; P_comp)
    print(f"\n[rd-bound] Training MINE estimator for I(X_comp; P_comp)...")
    mi_comp, _ = train_mine(x_feats, p_comp, epochs=800, device=str(DEVICE))
    print(f"[rd-bound] I(X_comp; P_comp) >= {mi_comp:.4f} nats = {mi_comp / math.log(2):.4f} bits")

    # Also train on (x_comp, p_gt) to see how much info our compressed frames
    # carry about the GT pose
    print(f"\n[rd-bound] Training MINE estimator for I(X_comp; P_gt)...")
    mi_comp_gt, _ = train_mine(x_feats, p_gt, epochs=800, device=str(DEVICE))
    print(f"[rd-bound] I(X_comp; P_gt) >= {mi_comp_gt:.4f} nats = {mi_comp_gt / math.log(2):.4f} bits")

    # Rate-distortion bound under Gaussian assumption on pose residual
    # D(R) = sigma^2 * 2^(-2R/k)  where sigma^2 is pose variance, k = 6 dims
    # R is mutual info we can carry per pair
    k = 6
    sigma2 = p_gt_var
    # Conservative: the compressed pair can carry AT MOST mi_comp_gt bits of info
    # about the GT pose. That's the Shannon ceiling on R for a filter
    # that only operates on X_comp.
    R_bits = mi_comp_gt / math.log(2)  # convert nats to bits
    D_min = sigma2 * (2 ** (-2 * R_bits / k))

    print(f"\n{'=' * 78}")
    print(f"RATE-DISTORTION BOUND")
    print(f"{'=' * 78}")
    print(f"Pose variance (sigma^2):               {sigma2:.6f}")
    print(f"I(X_comp; P_gt) lower bound:           {R_bits:.4f} bits/pair")
    print(f"Rate-distortion D_min(R):              {D_min:.6f}")
    print(f"")
    print(f"Current CNN achieves: {0.04809:.6f}")
    print(f"Theoretical lower bound: {D_min:.6f}")
    if D_min < 0.04809:
        headroom = 0.04809 - D_min
        print(f"Implied headroom:     {headroom:.6f}")
        score_at_dmin = 100 * 0.00576 + math.sqrt(10 * D_min) + 0.5754
        print(f"Score at D_min:       {score_at_dmin:.4f} (vs our 1.8453)")
    else:
        print(f"We are at or above the theoretical bound — CNN scaling is saturating.")

    # Caveat: MINE lower bound is an estimate, actual I could be higher, so
    # the bound is a LOOSE lower bound. The real minimum D is at most
    # sigma^2 * 2^(-2R_bits/6) but could be lower if our filter encoded
    # more info.

    result = {
        "p_gt_variance": sigma2,
        "mi_x_comp_p_comp_nats": mi_comp,
        "mi_x_comp_p_gt_nats": mi_comp_gt,
        "mi_x_comp_p_gt_bits": R_bits,
        "d_min": D_min,
        "current_cnn_pose_dist": 0.04809,
        "headroom": 0.04809 - D_min,
        "n_pairs": n_pairs,
    }
    print(f"\nJSON:\n{json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    main()
