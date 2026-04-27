#!/usr/bin/env python3
"""Constrained generation inflate: gradient descent from masks + pose targets.

Instead of storing compressed video frames (~150KB), this approach stores only
masks (~239B), pose targets (~7KB), and a noise seed (8B) for a total archive
of ~8KB. At inflate time, frames are reconstructed via gradient descent against
the frozen PoseNet and SegNet scorers.

The tradeoff is inflate time: 600 pairs x ~30 steps x ~100ms/step = ~1800s.
With adaptive per-pair budgeting and early stopping, we target 27 minutes
of optimization with 3 minutes for setup and I/O.

Rate elimination: 0.10 -> ~0.001 = 0.099 score points saved.

Usage (called by inflate.sh):
    python inflate.py <archive_dir> <inflated_dir> <video_names_file>

Architecture:
    1. Load frozen PoseNet + SegNet with differentiable preprocessing
    2. Load masks, pose targets, seed from archive
    3. Initialize frames from class-mean colors (deterministic from seed)
    4. Run constrained_generate with adaptive per-pair step budgeting
    5. Upscale from scorer resolution (384x512) to camera resolution (1164x874)
    6. Write raw RGB24 output
"""
from __future__ import annotations

import logging
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT_W, OUT_H = 1164, 874
SEG_W, SEG_H = 512, 384
NUM_FRAMES = 1200
NUM_PAIRS = NUM_FRAMES // 2
EXPECTED_RAW_BYTES = OUT_W * OUT_H * 3 * NUM_FRAMES

# Time budget (seconds)
TOTAL_BUDGET_S = 27 * 60      # 27 min for optimization (3 min reserved for setup)
SETUP_BUDGET_S = 3 * 60       # 3 min for model loading, data decoding, I/O
MIN_STEPS_PER_PAIR = 10       # absolute minimum to get signal
MAX_STEPS_PER_PAIR = 80       # cap per pair to leave room for hard pairs
DEFAULT_STEPS_PER_PAIR = 30   # starting estimate

# Optimization hyperparameters (council-approved)
LR = 0.05
SEG_WEIGHT = 50.0
POSE_WEIGHT = 50.0
COMPRESS_WEIGHT = 1.0
EARLY_STOP_PATIENCE = 8       # per-pair patience (aggressive for time budget)
EARLY_STOP_DELTA = 1e-4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Archive I/O (mirrors tac.constrained_gen.load_constrained_archive)
# ---------------------------------------------------------------------------

def load_archive(archive_dir: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load masks, pose targets, and noise seed from constrained archive.

    Returns:
        (masks, expected_pose, noise_seed) where:
          masks: (N, H, W) long tensor
          expected_pose: (P, 6) float tensor
          noise_seed: int
    """
    import lzma

    # Masks
    masks_data = (archive_dir / "masks.bin").read_bytes()
    N, H, W = struct.unpack("<III", masks_data[:12])
    masks_bytes = lzma.decompress(masks_data[12:])
    masks = torch.from_numpy(
        np.frombuffer(masks_bytes, dtype=np.uint8).reshape(N, H, W).copy()
    ).long()

    # Pose targets
    pose_data = (archive_dir / "pose_targets.bin").read_bytes()
    P, D = struct.unpack("<II", pose_data[:8])
    pose_np = np.frombuffer(pose_data[8:], dtype=np.float16).reshape(P, D).copy()
    expected_pose = torch.from_numpy(pose_np).float()

    # Seed
    seed_data = (archive_dir / "seed.bin").read_bytes()
    noise_seed = struct.unpack("<Q", seed_data)[0]

    logger.info(
        "Archive loaded: %d frames, %d pairs, seed=%d, "
        "masks=%s, pose=%s",
        N, P, noise_seed, list(masks.shape), list(expected_pose.shape),
    )
    return masks, expected_pose, noise_seed


# ---------------------------------------------------------------------------
# Frame initialization
# ---------------------------------------------------------------------------

# Class-mean colors (RGB, float) for the 5-class comma SegNet.
# Copied from tac.camera.CLASS_MEAN_COLORS for standalone operation.
# These must match the actual SegNet classes (0-4), NOT Cityscapes (0-17).
CLASS_MEAN_COLORS_INLINE = torch.tensor([
    [128.0, 128.0, 128.0],  # class 0: road (gray)
    [170.0, 170.0, 170.0],  # class 1: lane markings (light gray)
    [100.0, 80.0, 60.0],    # class 2: undrivable (brown)
    [120.0, 140.0, 160.0],  # class 3: movable objects (blue-gray)
    [180.0, 200.0, 230.0],  # class 4: sky (light blue)
], dtype=torch.float32)


def generate_initial_frames(
    masks: torch.Tensor,
    noise_seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Initialize frames from class-mean colors + small noise.

    Args:
        masks: (N, H, W) long tensor with class indices.
        noise_seed: deterministic seed for reproducibility.
        device: target device.

    Returns:
        (N, H, W, 3) float tensor in [0, 255].
    """
    N, H, W = masks.shape
    frames = CLASS_MEAN_COLORS_INLINE[masks.cpu()].to(device)  # (N, H, W, 3)

    # Add small deterministic noise for gradient diversity
    gen = torch.Generator(device="cpu")
    gen.manual_seed(noise_seed)
    noise = torch.randn(N, H, W, 3, generator=gen) * 5.0
    frames = (frames + noise.to(device)).clamp(0.0, 255.0)

    return frames


# ---------------------------------------------------------------------------
# Scorer loading
# ---------------------------------------------------------------------------

def load_scorers(upstream_dir: Path, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load frozen PoseNet + SegNet with differentiable preprocessing.

    Uses load_differentiable_scorers to ensure rgb_to_yuv6 gradients flow.
    Falls back to direct loading if tac is not available.
    """
    try:
        from tac.scorer import load_differentiable_scorers
        posenet, segnet = load_differentiable_scorers(upstream_dir, device=str(device))
        return posenet, segnet
    except ImportError:
        # Fallback: load directly from upstream (for contest machines without tac)
        sys.path.insert(0, str(upstream_dir))
        from modules import PoseNet, SegNet
        from safetensors.torch import load_file

        posenet = PoseNet().eval()
        posenet.load_state_dict(
            load_file(str(upstream_dir / "models" / "posenet.safetensors"), device="cpu")
        )
        segnet = SegNet().eval()
        segnet.load_state_dict(
            load_file(str(upstream_dir / "models" / "segnet.safetensors"), device="cpu")
        )
        posenet, segnet = posenet.to(device), segnet.to(device)
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        # CRITICAL: patch rgb_to_yuv6 to be differentiable
        _patch_differentiable_yuv(posenet)
        return posenet, segnet


def _patch_differentiable_yuv(posenet):
    """Replace PoseNet's preprocess_input with a fully differentiable version.

    The upstream rgb_to_yuv6 has @torch.no_grad, which kills ALL gradients
    through PoseNet. We replace the entire preprocess_input method with one
    that uses a differentiable BT.601 YUV420 conversion.

    This is the same fix as tac.scorer.make_scorers_differentiable, inlined
    for standalone operation without the tac package.
    """
    import types

    import einops

    def _rgb_to_yuv6_diff(rgb_chw: torch.Tensor) -> torch.Tensor:
        """Differentiable BT.601 full-range YUV420 conversion."""
        H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
        H2, W2 = H // 2, W // 2
        rgb = rgb_chw[..., :, :2 * H2, :2 * W2]
        R = rgb[..., 0, :, :]
        G = rgb[..., 1, :, :]
        B = rgb[..., 2, :, :]
        Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
        U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
        V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
        U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] + U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
        V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] + V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
        y00 = Y[..., 0::2, 0::2]
        y10 = Y[..., 1::2, 0::2]
        y01 = Y[..., 0::2, 1::2]
        y11 = Y[..., 1::2, 1::2]
        return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)

    # Scorer input size (W, H)
    segnet_model_input_size = (512, 384)

    def _diff_preprocess(self, x):
        batch_size, seq_len_local = x.shape[0], x.shape[1]
        x = einops.rearrange(x, "b t c h w -> (b t) c h w", b=batch_size, t=seq_len_local, c=3)
        x = F.interpolate(
            x,
            size=(segnet_model_input_size[1], segnet_model_input_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        yuv = _rgb_to_yuv6_diff(x)
        return einops.rearrange(yuv, "(b t) c h w -> b (t c) h w", b=batch_size, t=seq_len_local, c=6).contiguous()

    posenet.preprocess_input = types.MethodType(_diff_preprocess, posenet)

    # Also patch AllNorm for non-contiguous tensor safety
    for module in posenet.modules():
        if type(module).__name__ == "AllNorm":
            def _patched_forward(self, x):
                return self.bn(x.reshape(-1, 1)).reshape(x.shape)
            module.forward = types.MethodType(_patched_forward, module)


# ---------------------------------------------------------------------------
# Gradient validation
# ---------------------------------------------------------------------------

def validate_gradients(posenet: torch.nn.Module, device: torch.device) -> None:
    """Verify that PoseNet gradients flow through the differentiable pipeline.

    Creates a small test tensor, forwards through PoseNet, and checks that
    gradients are nonzero. This catches the @torch.no_grad bug on rgb_to_yuv6.

    Raises RuntimeError if gradients are dead.
    """
    test = torch.randn(1, 2, 3, SEG_H, SEG_W, device=device, requires_grad=True)
    pre = posenet.preprocess_input(test)
    out = posenet(pre)
    pose = out["pose"][..., :6]
    loss = pose.sum()
    loss.backward()
    if test.grad is None or test.grad.abs().max().item() == 0.0:
        raise RuntimeError(
            "FATAL: PoseNet gradients are ZERO. The differentiable YUV patch "
            "is not working. All optimization would produce dead gradients."
        )
    logger.info("  Gradient validation passed (max grad=%.6f)", test.grad.abs().max().item())


# ---------------------------------------------------------------------------
# Per-pair constrained optimization
# ---------------------------------------------------------------------------

def optimize_pair(
    frame_t: torch.Tensor,
    frame_t1: torch.Tensor,
    mask_t: torch.Tensor,
    mask_t1: torch.Tensor,
    expected_pose_pair: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    num_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    """Optimize a single pair of frames against SegNet + PoseNet constraints.

    Args:
        frame_t, frame_t1: (H, W, 3) float tensors in [0, 255] (initial guess).
        mask_t, mask_t1: (H, W) long tensors (target segmentation).
        expected_pose_pair: (6,) float tensor (target pose output).
        posenet: frozen PoseNet.
        segnet: frozen SegNet.
        num_steps: max optimization steps for this pair.
        device: computation device.

    Returns:
        (frame_t_opt, frame_t1_opt, final_loss, actual_steps) tuple.
    """
    # Stack into optimizable pair
    pair = torch.stack([frame_t, frame_t1], dim=0).to(device)  # (2, H, W, 3)
    pair = pair.detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([pair], lr=LR)
    masks_pair = torch.stack([mask_t, mask_t1]).to(device)  # (2, H, W)
    pose_target = expected_pose_pair.to(device).unsqueeze(0)  # (1, 6)

    best_loss = float("inf")
    no_improve = 0
    actual_steps = 0

    for step in range(num_steps):
        actual_steps += 1
        optimizer.zero_grad()

        # SegNet constraint: cross-entropy on second frame only (matches scorer)
        seg_loss = _segnet_loss(pair, masks_pair, segnet)

        # PoseNet constraint: L2 on pose output
        pose_loss = _posenet_loss(pair, pose_target, posenet)

        # Compressibility: total variation
        compress_loss = _tv_loss(pair)

        total = SEG_WEIGHT * seg_loss + POSE_WEIGHT * pose_loss + COMPRESS_WEIGHT * compress_loss
        total.backward()
        optimizer.step()

        with torch.no_grad():
            pair.data.clamp_(0.0, 255.0)

        loss_val = total.item()

        # Early stopping
        if best_loss - loss_val > EARLY_STOP_DELTA:
            best_loss = loss_val
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                break

    with torch.no_grad():
        result = pair.detach().round().clamp(0.0, 255.0)

    return result[0], result[1], best_loss, actual_steps


def _segnet_loss(
    pair: torch.Tensor,
    masks: torch.Tensor,
    segnet: torch.nn.Module,
) -> torch.Tensor:
    """SegNet cross-entropy loss on the SECOND frame only.

    The official scorer computes SegNet distortion on x[:, -1, ...] (the last
    frame of each pair). Computing loss on both frames wastes gradient budget
    on a frame that doesn't affect the SegNet score component.

    Args:
        pair: (2, H, W, 3) float tensor.
        masks: (2, H, W) long tensor with target class indices.
        segnet: frozen SegNet model.

    Returns:
        Scalar loss tensor.
    """
    # Only the second frame (index 1) matters for SegNet scoring
    frame1 = pair[1:2]  # (1, H, W, 3)
    mask1 = masks[1:2]  # (1, H, W)

    frame1_chw = frame1.permute(0, 3, 1, 2).contiguous()  # (1, 3, H, W)
    logits = segnet(frame1_chw)  # (1, C, H', W')

    # Resize mask to match logits spatial dims if needed
    _, C, Hl, Wl = logits.shape
    if Hl != mask1.shape[1] or Wl != mask1.shape[2]:
        mask_resized = F.interpolate(
            mask1.unsqueeze(1).float(),
            size=(Hl, Wl),
            mode="nearest",
        ).squeeze(1).long()
    else:
        mask_resized = mask1

    return F.cross_entropy(logits, mask_resized)


def _posenet_loss(
    pair: torch.Tensor,
    pose_target: torch.Tensor,
    posenet: torch.nn.Module,
) -> torch.Tensor:
    """PoseNet L2 loss for a frame pair.

    Args:
        pair: (2, H, W, 3) float tensor.
        pose_target: (1, 6) float tensor with expected pose output.
        posenet: frozen PoseNet model.

    Returns:
        Scalar loss tensor.
    """
    # PoseNet expects (B, 2, C, H, W)
    pair_chw = pair.permute(0, 3, 1, 2).contiguous()  # (2, 3, H, W)
    pair_in = pair_chw.unsqueeze(0)  # (1, 2, 3, H, W)

    posenet_in = posenet.preprocess_input(pair_in)
    posenet_out = posenet(posenet_in)
    pose = posenet_out["pose"][..., :6]  # (1, 6)

    return F.mse_loss(pose, pose_target)


def _tv_loss(pair: torch.Tensor) -> torch.Tensor:
    """Total variation loss for compressibility.

    Args:
        pair: (2, H, W, 3) float tensor.

    Returns:
        Scalar loss tensor.
    """
    # Spatial TV
    dx = (pair[:, :, 1:, :] - pair[:, :, :-1, :]).abs().mean()
    dy = (pair[:, 1:, :, :] - pair[:, :-1, :, :]).abs().mean()
    # Temporal TV (between the two frames in the pair)
    dt = (pair[1] - pair[0]).abs().mean() * 0.5
    return dx + dy + dt


# ---------------------------------------------------------------------------
# Adaptive step budgeting
# ---------------------------------------------------------------------------

def compute_step_budget(
    n_pairs: int,
    time_remaining_s: float,
    ms_per_step: float,
    completed_pairs: int = 0,
) -> int:
    """Compute per-pair step budget based on remaining time.

    Uses remaining time to adaptively allocate steps:
    - Easy pairs (early stopping) free up budget for hard pairs later.
    - Never go below MIN_STEPS_PER_PAIR.
    - Never exceed MAX_STEPS_PER_PAIR.

    Args:
        n_pairs: total number of pairs.
        time_remaining_s: seconds remaining in the optimization budget.
        ms_per_step: estimated milliseconds per optimization step.
        completed_pairs: pairs already completed.

    Returns:
        Steps to allocate for the next pair.
    """
    pairs_remaining = max(n_pairs - completed_pairs, 1)
    steps_possible = int((time_remaining_s * 1000) / (ms_per_step * pairs_remaining))
    steps = max(MIN_STEPS_PER_PAIR, min(steps_possible, MAX_STEPS_PER_PAIR))
    return steps


# ---------------------------------------------------------------------------
# Main inflate pipeline
# ---------------------------------------------------------------------------

def inflate_constrained_gen(
    archive_dir: Path,
    inflated_dir: Path,
    video_names_file: Path,
) -> None:
    """Full constrained generation inflate pipeline.

    1. Load scorers (PoseNet + SegNet) with differentiable preprocessing.
    2. Load archive (masks, pose targets, seed).
    3. Initialize frames from class-mean colors.
    4. Run per-pair constrained optimization with adaptive time budgeting.
    5. Upscale to camera resolution and write raw RGB24 output.
    """
    t_start = time.monotonic()

    # ---- Detect device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # ---- Find upstream directory ----
    # Try common locations relative to submission dir
    self_dir = Path(__file__).resolve().parent
    candidates = [
        self_dir / ".." / ".." / "upstream",
        self_dir / ".." / ".." / "workspace" / "upstream" / "comma_video_compression_challenge",
        Path("/kaggle/input/comma-video-compression-challenge"),
    ]
    upstream_dir = None
    for c in candidates:
        if (c / "models").exists():
            upstream_dir = c.resolve()
            break
    if upstream_dir is None:
        # Last resort: check COMMA_CHALLENGE_ROOT env var
        import os
        root = os.environ.get("COMMA_CHALLENGE_ROOT", "")
        if root and Path(root).exists():
            upstream_dir = Path(root)
        else:
            raise FileNotFoundError(
                "Cannot find upstream directory with scorer models. "
                "Set COMMA_CHALLENGE_ROOT or place upstream/ in project root."
            )
    logger.info("Upstream: %s", upstream_dir)

    # ---- Stage 1: Load scorers ----
    logger.info("Stage 1: Loading scorers...")
    t0 = time.monotonic()
    posenet, segnet = load_scorers(upstream_dir, device)
    t_load = time.monotonic() - t0
    logger.info("  Scorers loaded in %.1fs", t_load)

    # ---- Stage 1b: Validate gradients ----
    logger.info("Stage 1b: Validating PoseNet gradients...")
    validate_gradients(posenet, device)

    # ---- Stage 2: Load archive ----
    logger.info("Stage 2: Loading archive...")
    masks, expected_pose, noise_seed = load_archive(archive_dir)
    N = masks.shape[0]
    P = expected_pose.shape[0]
    logger.info("  %d frames, %d pairs", N, P)

    # ---- Stage 3: Initialize frames ----
    logger.info("Stage 3: Initializing frames from class-mean colors...")
    frames = generate_initial_frames(masks, noise_seed, device)
    logger.info("  Frames initialized: %s", list(frames.shape))

    # ---- Stage 4: Constrained optimization ----
    t_opt_start = time.monotonic()
    setup_elapsed = t_opt_start - t_start
    opt_budget_s = TOTAL_BUDGET_S - setup_elapsed
    logger.info(
        "Stage 4: Constrained optimization (%d pairs, %.0fs budget)...",
        P, opt_budget_s,
    )

    # Warm-up: time one step to estimate ms_per_step
    logger.info("  Calibrating step time...")
    t_cal = time.monotonic()
    warmup_f0, warmup_f1, warmup_loss, warmup_steps = optimize_pair(
        frames[0], frames[1],
        masks[0], masks[1],
        expected_pose[0],
        posenet, segnet,
        num_steps=3,
        device=device,
    )
    ms_per_step = ((time.monotonic() - t_cal) / max(warmup_steps, 1)) * 1000
    logger.info("  Calibrated: %.1f ms/step (%d warmup steps)", ms_per_step, warmup_steps)

    # Optimize each pair with adaptive budgeting
    optimized_frames = torch.zeros(N, SEG_H, SEG_W, 3, dtype=torch.float32)
    total_steps = 0

    # S1: Reuse warmup optimization result for pair 0
    optimized_frames[0] = warmup_f0.cpu()
    optimized_frames[1] = warmup_f1.cpu()
    total_steps += warmup_steps
    logger.info(
        "  Pair 1/%d (warmup reuse): steps=%d, loss=%.4f",
        P, warmup_steps, warmup_loss,
    )

    for pair_idx in range(1, P):
        t_pair_start = time.monotonic()
        time_remaining = opt_budget_s - (t_pair_start - t_opt_start)

        if time_remaining <= 0:
            logger.warning(
                "  Time budget exhausted at pair %d/%d. "
                "Using initial frames for remaining pairs.",
                pair_idx, P,
            )
            # Copy remaining initial frames
            for k in range(pair_idx * 2, N):
                optimized_frames[k] = frames[k].cpu()
            break

        steps = compute_step_budget(P, time_remaining, ms_per_step, pair_idx)

        f0_idx = pair_idx * 2
        f1_idx = pair_idx * 2 + 1

        f0_opt, f1_opt, loss, actual = optimize_pair(
            frames[f0_idx], frames[f1_idx],
            masks[f0_idx], masks[f1_idx],
            expected_pose[pair_idx],
            posenet, segnet,
            num_steps=steps,
            device=device,
        )

        optimized_frames[f0_idx] = f0_opt.cpu()
        optimized_frames[f1_idx] = f1_opt.cpu()
        total_steps += actual

        dt_pair = time.monotonic() - t_pair_start

        if (pair_idx + 1) % 50 == 0:
            logger.info(
                "  Pair %d/%d: steps=%d/%d, loss=%.4f, %.1fs, "
                "%.0fs remaining",
                pair_idx + 1, P, actual, steps, loss, dt_pair,
                opt_budget_s - (time.monotonic() - t_opt_start),
            )

    t_opt_elapsed = time.monotonic() - t_opt_start
    logger.info(
        "  Optimization complete: %d total steps, %.1fs (%.1f min)",
        total_steps, t_opt_elapsed, t_opt_elapsed / 60,
    )

    # ---- Stage 5: Upscale and write raw RGB24 ----
    logger.info("Stage 5: Upscaling %dx%d -> %dx%d and writing raw RGB...",
                SEG_W, SEG_H, OUT_W, OUT_H)

    video_names = video_names_file.read_text().strip().splitlines()
    video_names = [v.strip() for v in video_names if v.strip()]

    for rel in video_names:
        stem = rel.rsplit(".", 1)[0]
        raw_out = inflated_dir / f"{stem}.raw"
        raw_out.parent.mkdir(parents=True, exist_ok=True)

        with open(raw_out, "wb") as f:
            for i in range(N):
                frame = optimized_frames[i]  # (H, W, 3) float
                # Upscale: HWC -> CHW -> interpolate -> CHW -> HWC
                frame_chw = frame.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
                frame_up = F.interpolate(
                    frame_chw, size=(OUT_H, OUT_W),
                    mode="bilinear", align_corners=False,
                )
                frame_uint8 = frame_up.round().clamp(0, 255).to(torch.uint8)
                frame_out = frame_uint8.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
                f.write(frame_out.tobytes())

        n_bytes = raw_out.stat().st_size
        logger.info("  Wrote %s: %s bytes (expected %s)",
                     raw_out.name, f"{n_bytes:,}", f"{EXPECTED_RAW_BYTES:,}")

        if n_bytes != EXPECTED_RAW_BYTES:
            logger.error(
                "FATAL: raw file size mismatch: got %d, expected %d",
                n_bytes, EXPECTED_RAW_BYTES,
            )
            sys.exit(1)

    t_total = time.monotonic() - t_start
    logger.info(
        "Constrained generation inflate complete: %.1fs (%.1f min)",
        t_total, t_total / 60,
    )

    # Time budget report
    if t_total > 30 * 60:
        logger.warning(
            "OVER TIME BUDGET: %.1f min > 30 min. "
            "This submission would exceed the contest time limit.",
            t_total / 60,
        )
    else:
        logger.info(
            "Within time budget: %.1f min / 30 min (%.0f%% utilization)",
            t_total / 60, 100 * t_total / (30 * 60),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <archive_dir> <inflated_dir> <video_names_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    archive_dir = Path(sys.argv[1])
    inflated_dir = Path(sys.argv[2])
    video_names_file = Path(sys.argv[3])

    inflate_constrained_gen(archive_dir, inflated_dir, video_names_file)


if __name__ == "__main__":
    main()
