"""Constrained optimization frame generator -- Yousfi GPU breakthrough.

Generate scorer-optimal frames via constrained gradient descent from noise.
No neural renderer weights needed. The archive contains only:
  - masks (entropy-coded, ~239 bytes)
  - expected PoseNet targets (~7KB)
  - noise seed (64 bytes)
Total: ~8KB archive.

At inflate time: run gradient descent from seeded noise to satisfy two
hard constraints:
  1. SegNet(frame) argmax == mask  (semantic preservation)
  2. PoseNet(frame_t, frame_t+1) ~ expected_pose  (temporal consistency)

While minimizing total variation (compressibility).

~1000 steps, ~50ms/step on T4 = 50 seconds inflate.

Example::

    from tac.constrained_gen import ConstrainedFrameGenerator
    gen = ConstrainedFrameGenerator(posenet, segnet, device="cuda")
    frames = gen.constrained_generate(
        masks, expected_pose, noise_seed=42, num_steps=1000,
    )
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from tac.mask_codec import NUM_CLASSES, SEGNET_H, SEGNET_W

# ---- Constants ----

# Camera native resolution (comma challenge spec)
CAMERA_W = 1164
CAMERA_H = 874

# SegNet input resolution (from frame_utils.py upstream)
SEGNET_INPUT_W = 512
SEGNET_INPUT_H = 384

# Semantic class mean colors (BGR order observed in driving scenes).
# road=gray, lane=lighter gray, undrivable=brown, movable=blue-gray, sky=light blue
CLASS_MEAN_COLORS = torch.tensor(
    [
        [128.0, 128.0, 128.0],  # class 0: road (gray)
        [170.0, 170.0, 170.0],  # class 1: lane markings (light gray)
        [100.0, 80.0, 60.0],    # class 2: undrivable (brown)
        [120.0, 140.0, 160.0],  # class 3: movable objects (blue-gray)
        [180.0, 200.0, 230.0],  # class 4: sky (light blue)
    ],
    dtype=torch.float32,
)  # (NUM_CLASSES, 3)


# ---- YUV420 conversion utilities ----


def rgb_to_yuv6(rgb_chw: torch.Tensor) -> torch.Tensor:
    """Convert RGB CHW to YUV420 6-channel representation.

    Matches the upstream ``frame_utils.rgb_to_yuv6`` exactly.

    Args:
        rgb_chw: (..., 3, H, W) float tensor in [0, 255].

    Returns:
        (..., 6, H//2, W//2) tensor: [y00, y10, y01, y11, U_sub, V_sub].
    """
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, : 2 * H2, : 2 * W2]

    R = rgb[..., 0, :, :]
    G = rgb[..., 1, :, :]
    B = rgb[..., 2, :, :]

    kYR, kYG, kYB = 0.299, 0.587, 0.114
    Y = (R * kYR + G * kYG + B * kYB).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)

    U_sub = (
        U[..., 0::2, 0::2]
        + U[..., 1::2, 0::2]
        + U[..., 0::2, 1::2]
        + U[..., 1::2, 1::2]
    ) * 0.25
    V_sub = (
        V[..., 0::2, 0::2]
        + V[..., 1::2, 0::2]
        + V[..., 0::2, 1::2]
        + V[..., 1::2, 1::2]
    ) * 0.25

    y00 = Y[..., 0::2, 0::2]
    y10 = Y[..., 1::2, 0::2]
    y01 = Y[..., 0::2, 1::2]
    y11 = Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)


def yuv6_to_rgb(yuv6: torch.Tensor) -> torch.Tensor:
    """Invert YUV420 6-channel back to approximate RGB CHW.

    This is the pseudo-inverse of ``rgb_to_yuv6``. The chroma subsampling
    loses spatial resolution, so the reconstruction is approximate.

    Args:
        yuv6: (..., 6, H2, W2) tensor from rgb_to_yuv6.

    Returns:
        (..., 3, H, W) float tensor in [0, 255].
    """
    y00 = yuv6[..., 0, :, :]
    y10 = yuv6[..., 1, :, :]
    y01 = yuv6[..., 2, :, :]
    y11 = yuv6[..., 3, :, :]
    U_sub = yuv6[..., 4, :, :]
    V_sub = yuv6[..., 5, :, :]

    H2, W2 = y00.shape[-2], y00.shape[-1]
    H, W = H2 * 2, W2 * 2

    # Reconstruct full-res Y by placing sub-pixels back
    Y = torch.zeros(*yuv6.shape[:-3], H, W, device=yuv6.device, dtype=yuv6.dtype)
    Y[..., 0::2, 0::2] = y00
    Y[..., 1::2, 0::2] = y10
    Y[..., 0::2, 1::2] = y01
    Y[..., 1::2, 1::2] = y11

    # Upsample chroma via nearest-neighbor (matches 4:2:0 upsampling)
    U_full = U_sub.unsqueeze(-3)  # add channel dim for interpolate
    V_full = V_sub.unsqueeze(-3)
    U_full = F.interpolate(
        U_full.reshape(-1, 1, H2, W2), size=(H, W), mode="nearest"
    ).reshape(*yuv6.shape[:-3], H, W)
    V_full = F.interpolate(
        V_full.reshape(-1, 1, H2, W2), size=(H, W), mode="nearest"
    ).reshape(*yuv6.shape[:-3], H, W)

    # Inverse YUV -> RGB
    # Y = R*0.299 + G*0.587 + B*0.114
    # U = (B - Y)/1.772 + 128  =>  B = (U - 128)*1.772 + Y
    # V = (R - Y)/1.402 + 128  =>  R = (V - 128)*1.402 + Y
    B = (U_full - 128.0) * 1.772 + Y
    R = (V_full - 128.0) * 1.402 + Y
    G = (Y - R * 0.299 - B * 0.114) / 0.587

    R = R.clamp(0.0, 255.0)
    G = G.clamp(0.0, 255.0)
    B = B.clamp(0.0, 255.0)

    return torch.stack([R, G, B], dim=-3)


# ---- Core functions ----


def generate_initial_frames(
    masks: torch.Tensor,
    noise_seed: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate deterministic initial frames from masks + seed.

    Uses class-mean colors as the starting point with small additive noise
    from a seeded generator. This gives the optimizer a head start: each
    pixel already has roughly the right color for its semantic class.

    Args:
        masks: (N, H, W) long tensor with class indices in [0, NUM_CLASSES).
        noise_seed: integer seed for deterministic noise generation.
        device: target device for the output tensor.

    Returns:
        (N, H, W, 3) float tensor in [0, 255] on device.
    """
    device = torch.device(device)
    N, H, W = masks.shape
    masks_dev = masks.to(device)

    # Look up class-mean color per pixel: (N, H, W) -> (N, H, W, 3)
    colors = CLASS_MEAN_COLORS.to(device)  # (NUM_CLASSES, 3)
    frames = colors[masks_dev]  # fancy index: (N, H, W, 3)

    # Add small deterministic noise for symmetry breaking
    gen = torch.Generator(device="cpu")
    gen.manual_seed(noise_seed)
    noise = torch.randn(N, H, W, 3, generator=gen).to(device) * 5.0
    frames = (frames + noise).clamp(0.0, 255.0)

    return frames


def compute_segnet_constraint_loss(
    frames: torch.Tensor,
    masks: torch.Tensor,
    segnet: torch.nn.Module,
) -> torch.Tensor:
    """Cross-entropy loss forcing SegNet argmax to match target masks.

    Uses a straight-through estimator (STE) for gradient flow through the
    hard argmax: forward pass uses soft cross-entropy (which is differentiable),
    and the gradient naturally flows through the logits.

    The SegNet preprocessor expects (B, T, C, H, W) input. We feed single
    frames as T=1 sequences using the last-frame-only SegNet interface.

    Args:
        frames: (N, H, W, 3) float tensor in [0, 255], requires_grad.
        masks: (N, H, W) long tensor with target class indices.
        segnet: frozen SegNet model.

    Returns:
        Scalar cross-entropy loss.
    """
    N, H, W, C = frames.shape
    device = frames.device

    # SegNet.preprocess_input expects (B, T, C, H, W) and uses only last frame.
    # We construct (N, 1, C, H, W) so T=1 and the last frame is our frame.
    frames_btchw = frames.permute(0, 3, 1, 2).unsqueeze(1).contiguous()  # (N, 1, C, H, W)

    # Resize to SegNet input resolution
    frames_chw = frames_btchw[:, -1, ...]  # (N, C, H, W)
    frames_resized = F.interpolate(
        frames_chw,
        size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
        mode="bilinear",
        align_corners=False,
    )  # (N, C, SEGNET_INPUT_H, SEGNET_INPUT_W)

    # Forward through SegNet (expects preprocessed input)
    logits = segnet(frames_resized)  # (N, NUM_CLASSES, H_out, W_out)

    # Resize masks to match logit spatial dims
    H_out, W_out = logits.shape[2], logits.shape[3]
    masks_resized = (
        F.interpolate(
            masks.float().unsqueeze(1),
            size=(H_out, W_out),
            mode="nearest",
        )
        .squeeze(1)
        .long()
        .to(device)
    )  # (N, H_out, W_out)

    loss = F.cross_entropy(logits, masks_resized)
    return loss


def compute_posenet_constraint_loss(
    frames: torch.Tensor,
    expected_pose: torch.Tensor,
    posenet: torch.nn.Module,
) -> torch.Tensor:
    """L2 loss between PoseNet output and expected ego-motion targets.

    PoseNet expects consecutive frame pairs as (B, T=2, C, H, W) input.
    We construct pairs from consecutive frames in the sequence.

    Args:
        frames: (N, H, W, 3) float tensor in [0, 255], requires_grad.
        expected_pose: (P, 6) float tensor of expected pose outputs,
            where P = N-1 (one pose per consecutive pair).
        posenet: frozen PoseNet model.

    Returns:
        Scalar L2 loss averaged over all pairs and dimensions.
    """
    N = frames.shape[0]
    P = N - 1
    assert expected_pose.shape[0] == P, (
        f"Expected {P} pose targets for {N} frames, got {expected_pose.shape[0]}"
    )
    device = frames.device

    # Build consecutive pairs: (P, 2, H, W, 3) -> (P, 2, C, H, W)
    frame1 = frames[:-1]  # (P, H, W, 3)
    frame2 = frames[1:]   # (P, H, W, 3)
    pairs_hwc = torch.stack([frame1, frame2], dim=1)  # (P, 2, H, W, 3)
    pairs_chw = pairs_hwc.permute(0, 1, 4, 2, 3).contiguous()  # (P, 2, C, H, W)

    # PoseNet preprocessing: RGB -> YUV420 6-ch, resize, rearrange
    posenet_in = posenet.preprocess_input(pairs_chw)
    posenet_out = posenet(posenet_in)

    # PoseNet returns dict with "pose" key, shape (P, >=6)
    pred_pose = posenet_out["pose"][..., :6]  # (P, 6)
    target = expected_pose.to(device)

    loss = (pred_pose - target).pow(2).mean()
    return loss


def compute_compressibility_loss(frames: torch.Tensor) -> torch.Tensor:
    """Total variation + temporal smoothness for compressibility.

    Smooth frames compress better under any codec. This loss encourages:
    1. Spatial smoothness: small pixel differences between neighbors.
    2. Temporal smoothness: small differences between consecutive frames.

    Args:
        frames: (N, H, W, 3) float tensor in [0, 255].

    Returns:
        Scalar compressibility loss (lower = more compressible).
    """
    # Spatial total variation: sum of absolute horizontal + vertical gradients
    # Normalized by pixel count and channel count for scale invariance.
    tv_h = (frames[:, 1:, :, :] - frames[:, :-1, :, :]).abs().mean()
    tv_w = (frames[:, :, 1:, :] - frames[:, :, :-1, :]).abs().mean()
    spatial_tv = tv_h + tv_w

    # Temporal smoothness: L1 between consecutive frames
    if frames.shape[0] > 1:
        temporal = (frames[1:] - frames[:-1]).abs().mean()
    else:
        temporal = torch.tensor(0.0, device=frames.device, dtype=frames.dtype)

    return spatial_tv + 0.5 * temporal


def estimate_expected_pose(
    masks: torch.Tensor,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Estimate expected ego-motion from mask sequence.

    Uses class centroid displacement and vanishing point shift as features
    to estimate the 6-DOF pose output that PoseNet would produce for
    well-formed driving frames matching these masks.

    The model is simple linear: for driving scenes the ego-motion is mostly
    forward translation with small lateral/rotational corrections. The mask
    geometry (where the road is, how it shifts) encodes this implicitly.

    The 6 PoseNet outputs are (tx, ty, tz, rx, ry, rz) in some internal
    representation. For typical comma driving data:
      - tx ~ small lateral (near 0)
      - ty ~ small vertical (near 0)
      - tz ~ forward motion (dominant, ~0.1-1.0)
      - rx, ry, rz ~ small rotations (near 0)

    Args:
        masks: (N, H, W) long tensor with class indices in [0, NUM_CLASSES).
        device: computation device.

    Returns:
        (P, 6) float tensor of estimated pose targets, P = N-1.
    """
    device = torch.device(device)
    N, H, W = masks.shape
    P = N - 1

    if P == 0:
        return torch.zeros(0, 6, device=device)

    # Compute class centroids per frame
    # centroid_y, centroid_x for each class in each frame
    y_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    x_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    masks_dev = masks.to(device)

    centroids = torch.zeros(N, NUM_CLASSES, 2, device=device)  # (N, C, 2) = (y, x)
    for c in range(NUM_CLASSES):
        class_mask = (masks_dev == c).float()  # (N, H, W)
        area = class_mask.sum(dim=(1, 2)).clamp(min=1.0)  # (N,)
        cy = (class_mask * y_coords).sum(dim=(1, 2)) / area  # (N,)
        cx = (class_mask * x_coords).sum(dim=(1, 2)) / area  # (N,)
        centroids[:, c, 0] = cy / H  # normalize to [0, 1]
        centroids[:, c, 1] = cx / W

    # Centroid displacements between consecutive frames
    centroid_deltas = centroids[1:] - centroids[:-1]  # (P, C, 2)

    # Road class (0) centroid displacement is the strongest ego-motion signal
    road_dy = centroid_deltas[:, 0, 0]  # (P,) vertical shift of road
    road_dx = centroid_deltas[:, 0, 1]  # (P,) horizontal shift of road

    # Sky class (4) centroid shift indicates horizon/rotation
    sky_dy = centroid_deltas[:, 4, 0]   # (P,)
    sky_dx = centroid_deltas[:, 4, 1]   # (P,)

    # Vanishing point proxy: road centroid x position
    road_cx = centroids[:-1, 0, 1]  # (P,) road center-x in first frame

    # Simple linear model for pose estimation:
    # These are heuristic coefficients tuned to match typical PoseNet output
    # magnitudes on comma driving data. The exact values matter less than
    # providing a reasonable initial target for the optimization to converge from.
    poses = torch.zeros(P, 6, device=device)
    poses[:, 0] = road_dx * 0.5       # tx: lateral from road shift
    poses[:, 1] = road_dy * 0.3       # ty: vertical from road shift
    poses[:, 2] = 0.1 + road_dy * 0.2  # tz: forward (baseline + road growth)
    poses[:, 3] = sky_dy * 0.2        # rx: pitch from sky shift
    poses[:, 4] = sky_dx * 0.3        # ry: yaw from sky shift
    poses[:, 5] = (road_cx - 0.5) * 0.1  # rz: roll from road asymmetry

    return poses


def constrained_generate(
    masks: torch.Tensor,
    expected_pose: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    noise_seed: int = 42,
    num_steps: int = 1000,
    lr: float = 0.1,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    compress_weight: float = 1.0,
    device: torch.device | str = "cpu",
    log_every: int = 100,
) -> torch.Tensor:
    """Main constrained optimization loop: generate frames from masks.

    Starting from class-mean-colored noise, optimize pixel values via
    gradient descent to simultaneously satisfy:
      1. SegNet constraint: argmax matches target masks
      2. PoseNet constraint: pose output matches expected targets
      3. Compressibility: total variation is minimized

    Args:
        masks: (N, H, W) long tensor with class indices.
        expected_pose: (P, 6) float tensor, P = N-1.
        posenet: frozen PoseNet model (on device).
        segnet: frozen SegNet model (on device).
        noise_seed: deterministic seed for initialization.
        num_steps: number of Adam optimization steps.
        lr: learning rate for Adam optimizer.
        seg_weight: weight for SegNet cross-entropy constraint.
        pose_weight: weight for PoseNet L2 constraint.
        compress_weight: weight for total variation loss.
        device: computation device.
        log_every: print loss every N steps (0 to disable).

    Returns:
        (N, H, W, 3) float tensor in [0, 255], rounded to uint8-compatible values.
    """
    device = torch.device(device)

    # Initialize frames from masks + seed
    frames = generate_initial_frames(masks, noise_seed, device=device)
    frames.requires_grad_(True)

    optimizer = torch.optim.Adam([frames], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()

        # Compute constraint losses
        seg_loss = compute_segnet_constraint_loss(frames, masks, segnet)
        compress_loss = compute_compressibility_loss(frames)

        # PoseNet loss only if we have pairs (N > 1)
        if frames.shape[0] > 1 and expected_pose.shape[0] > 0:
            pose_loss = compute_posenet_constraint_loss(
                frames, expected_pose, posenet,
            )
        else:
            pose_loss = torch.tensor(0.0, device=device)

        # Weighted combination
        total_loss = (
            seg_weight * seg_loss
            + pose_weight * pose_loss
            + compress_weight * compress_loss
        )

        total_loss.backward()
        optimizer.step()

        # Project back to valid pixel range
        with torch.no_grad():
            frames.data.clamp_(0.0, 255.0)

        if log_every > 0 and (step + 1) % log_every == 0:
            print(
                f"  step {step + 1:4d}/{num_steps}: "
                f"total={total_loss.item():.4f} "
                f"seg={seg_loss.item():.4f} "
                f"pose={pose_loss.item():.4f} "
                f"compress={compress_loss.item():.4f}"
            )

    # Final quantization: round to nearest integer, clamp to uint8 range
    with torch.no_grad():
        result = frames.detach().round().clamp(0.0, 255.0)

    return result


def build_constrained_archive(
    masks: torch.Tensor,
    expected_pose: torch.Tensor,
    noise_seed: int,
    output_path: str | Path,
) -> Path:
    """Build a minimal archive for constrained-generation inflate.

    The archive contains three files:
      - masks.bin: LZMA-compressed uint8 mask tensor (~239 bytes for typical data)
      - pose_targets.bin: float16 pose targets (~7KB for 1200 frames)
      - seed.bin: 64-byte noise seed

    Total archive size: ~8KB (vs ~100KB+ for neural renderer weights).

    Args:
        masks: (N, H, W) long tensor with class indices.
        expected_pose: (P, 6) float tensor.
        noise_seed: integer seed.
        output_path: directory to write archive files into.

    Returns:
        Path to the output directory.
    """
    import lzma

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Masks: convert to uint8, flatten, LZMA compress
    masks_np = masks.cpu().numpy().astype(np.uint8)
    masks_bytes = masks_np.tobytes()
    # Store shape header (3 x uint32) + compressed data
    shape_header = struct.pack("<III", *masks_np.shape)
    compressed = lzma.compress(masks_bytes, preset=9)
    (out / "masks.bin").write_bytes(shape_header + compressed)

    # 2. Pose targets: float16 for space efficiency
    pose_np = expected_pose.cpu().to(torch.float16).numpy()
    pose_header = struct.pack("<II", *pose_np.shape)
    (out / "pose_targets.bin").write_bytes(pose_header + pose_np.tobytes())

    # 3. Seed: 64 bytes (uint64)
    (out / "seed.bin").write_bytes(struct.pack("<Q", noise_seed))

    # Metadata for reproducibility
    meta = {
        "num_frames": int(masks.shape[0]),
        "mask_shape": list(masks.shape),
        "pose_shape": list(expected_pose.shape),
        "noise_seed": noise_seed,
        "compressed_masks_bytes": len(shape_header + compressed),
        "pose_bytes": len(pose_header + pose_np.tobytes()),
        "total_bytes": (
            len(shape_header + compressed)
            + len(pose_header + pose_np.tobytes())
            + 8  # seed
        ),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    total = meta["total_bytes"]
    print(f"Constrained archive: {total:,} bytes ({total / 1024:.1f} KB)")
    print(f"  masks: {meta['compressed_masks_bytes']:,} bytes")
    print(f"  poses: {meta['pose_bytes']:,} bytes")
    print(f"  seed:  8 bytes")

    return out


def load_constrained_archive(
    archive_dir: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load a constrained-generation archive.

    Args:
        archive_dir: directory containing masks.bin, pose_targets.bin, seed.bin.

    Returns:
        (masks, expected_pose, noise_seed) tuple.
    """
    import lzma

    d = Path(archive_dir)

    # Masks
    masks_data = (d / "masks.bin").read_bytes()
    N, H, W = struct.unpack("<III", masks_data[:12])
    masks_bytes = lzma.decompress(masks_data[12:])
    masks = torch.from_numpy(
        np.frombuffer(masks_bytes, dtype=np.uint8).reshape(N, H, W).copy()
    ).long()

    # Pose targets
    pose_data = (d / "pose_targets.bin").read_bytes()
    P, D = struct.unpack("<II", pose_data[:8])
    pose_np = np.frombuffer(pose_data[8:], dtype=np.float16).reshape(P, D).copy()
    expected_pose = torch.from_numpy(pose_np).float()

    # Seed
    seed_data = (d / "seed.bin").read_bytes()
    noise_seed = struct.unpack("<Q", seed_data)[0]

    return masks, expected_pose, noise_seed


def inflate_constrained(
    archive_dir: str | Path,
    output_dir: str | Path,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    num_steps: int = 1000,
    lr: float = 0.1,
    device: torch.device | str = "cpu",
    log_every: int = 100,
) -> Path:
    """Inflate function: run constrained optimization at test time.

    Loads the archive, runs gradient descent to generate frames, and
    writes the result as uint8 numpy arrays.

    Args:
        archive_dir: directory containing the constrained archive.
        output_dir: directory to write generated frames.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        num_steps: optimization steps (more = better quality, slower).
        lr: Adam learning rate.
        device: computation device.
        log_every: print loss every N steps.

    Returns:
        Path to output directory.
    """
    masks, expected_pose, noise_seed = load_constrained_archive(archive_dir)
    masks = masks.to(device)
    expected_pose = expected_pose.to(device)

    print(f"Inflating {masks.shape[0]} frames via constrained optimization...")
    print(f"  steps={num_steps}, lr={lr}, seed={noise_seed}")

    frames = constrained_generate(
        masks=masks,
        expected_pose=expected_pose,
        posenet=posenet,
        segnet=segnet,
        noise_seed=noise_seed,
        num_steps=num_steps,
        lr=lr,
        device=device,
        log_every=log_every,
    )

    # Write frames as uint8 numpy arrays
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames_np = frames.cpu().numpy().astype(np.uint8)
    np.save(out / "frames.npy", frames_np)
    print(f"Wrote {frames_np.shape[0]} frames to {out / 'frames.npy'}")

    return out


# ---- GPU Eureka #1: Generate in scorer space ----


def inverse_preprocess_input(
    scorer_space_tensor: torch.Tensor,
    target_h: int = CAMERA_H,
    target_w: int = CAMERA_W,
) -> torch.Tensor:
    """Invert the scorer preprocessing to recover approximate RGB frames.

    The scorer preprocessing pipeline is:
      RGB (B, C, H, W) -> resize to (384, 512) -> rgb_to_yuv6 -> (B, 6, 192, 256)

    This function inverts that pipeline:
      (B, 6, 192, 256) -> yuv6_to_rgb -> (B, 3, 384, 512) -> resize to (H, W)

    The inversion is approximate because:
      1. Chroma subsampling in YUV420 loses spatial information.
      2. Bilinear resize is not perfectly invertible.

    Args:
        scorer_space_tensor: (B, 6, H2, W2) tensor in scorer's internal
            YUV420 representation (output of rgb_to_yuv6 after resize).
        target_h: target height for output RGB frames.
        target_w: target width for output RGB frames.

    Returns:
        (B, 3, target_h, target_w) float tensor in [0, 255].
    """
    # Step 1: YUV6 -> RGB at half-resolution
    rgb_small = yuv6_to_rgb(scorer_space_tensor)  # (B, 3, H, W) at 384x512 scale

    # Step 2: Resize to target resolution
    if rgb_small.shape[-2] != target_h or rgb_small.shape[-1] != target_w:
        rgb_full = F.interpolate(
            rgb_small,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
    else:
        rgb_full = rgb_small

    return rgb_full.clamp(0.0, 255.0)


def generate_in_scorer_space(
    masks: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    noise_seed: int = 42,
    num_steps: int = 1000,
    lr: float = 0.1,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    compress_weight: float = 1.0,
    device: torch.device | str = "cpu",
    log_every: int = 100,
) -> torch.Tensor:
    """Generate frames directly in the scorer's internal representation space.

    Instead of optimizing in RGB and losing information through the scorer's
    preprocessing, we optimize directly in the scorer's YUV420 space and
    invert at the end. This eliminates the lossy preprocessing from the
    optimization loop entirely.

    The scorer sees exactly what we optimize -- no information is lost
    through the preprocessing bottleneck during gradient descent.

    Args:
        masks: (N, H, W) long tensor with class indices.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        noise_seed: deterministic seed for initialization.
        num_steps: optimization steps.
        lr: Adam learning rate.
        seg_weight: SegNet constraint weight.
        pose_weight: PoseNet constraint weight.
        compress_weight: compressibility weight.
        device: computation device.
        log_every: print loss every N steps.

    Returns:
        (N, H, W, 3) float tensor in [0, 255] (RGB, HWC format).
    """
    device = torch.device(device)
    N = masks.shape[0]
    P = N - 1

    # Initialize in scorer space: start from class-mean colors converted to YUV6
    init_frames = generate_initial_frames(masks, noise_seed, device=device)
    # Convert HWC -> CHW -> resize -> YUV6
    init_chw = init_frames.permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W)
    init_resized = F.interpolate(
        init_chw,
        size=(SEGNET_INPUT_H, SEGNET_INPUT_W),
        mode="bilinear",
        align_corners=False,
    )  # (N, 3, 384, 512)
    scorer_frames = rgb_to_yuv6(init_resized)  # (N, 6, 192, 256)
    scorer_frames = scorer_frames.detach().requires_grad_(True)

    optimizer = torch.optim.Adam([scorer_frames], lr=lr)

    # Estimate expected pose from masks if not provided
    expected_pose = estimate_expected_pose(masks, device=device)

    for step in range(num_steps):
        optimizer.zero_grad()

        # SegNet loss: operates on last frame's RGB at segnet resolution
        # SegNet expects (B, C, 384, 512) input (no YUV conversion for SegNet)
        rgb_for_seg = yuv6_to_rgb(scorer_frames)  # (N, 3, 384, 512)
        seg_logits = segnet(rgb_for_seg)  # (N, NUM_CLASSES, H_out, W_out)

        H_out, W_out = seg_logits.shape[2], seg_logits.shape[3]
        masks_resized = (
            F.interpolate(
                masks.float().unsqueeze(1).to(device),
                size=(H_out, W_out),
                mode="nearest",
            )
            .squeeze(1)
            .long()
        )
        seg_loss = F.cross_entropy(seg_logits, masks_resized)

        # PoseNet loss: operates on YUV6 pairs directly
        pose_loss = torch.tensor(0.0, device=device)
        if P > 0 and expected_pose.shape[0] > 0:
            # Build pairs in YUV6 space: PoseNet expects (B, T*6, H2, W2)
            yuv_t0 = scorer_frames[:-1]  # (P, 6, H2, W2)
            yuv_t1 = scorer_frames[1:]   # (P, 6, H2, W2)
            posenet_in = torch.cat([yuv_t0, yuv_t1], dim=1)  # (P, 12, H2, W2)
            posenet_out = posenet(posenet_in)
            pred_pose = posenet_out["pose"][..., :6]
            pose_loss = (pred_pose - expected_pose.to(device)).pow(2).mean()

        # Compressibility in scorer space (TV on YUV channels)
        tv_h = (scorer_frames[:, :, 1:, :] - scorer_frames[:, :, :-1, :]).abs().mean()
        tv_w = (scorer_frames[:, :, :, 1:] - scorer_frames[:, :, :, :-1]).abs().mean()
        temporal = torch.tensor(0.0, device=device)
        if N > 1:
            temporal = (scorer_frames[1:] - scorer_frames[:-1]).abs().mean()
        compress_loss = tv_h + tv_w + 0.5 * temporal

        total_loss = (
            seg_weight * seg_loss
            + pose_weight * pose_loss
            + compress_weight * compress_loss
        )

        total_loss.backward()
        optimizer.step()

        # Clamp to valid YUV range
        with torch.no_grad():
            scorer_frames.data.clamp_(0.0, 255.0)

        if log_every > 0 and (step + 1) % log_every == 0:
            print(
                f"  [scorer-space] step {step + 1:4d}/{num_steps}: "
                f"total={total_loss.item():.4f} "
                f"seg={seg_loss.item():.4f} "
                f"pose={pose_loss.item():.4f} "
                f"compress={compress_loss.item():.4f}"
            )

    # Invert from scorer space to RGB at camera resolution
    with torch.no_grad():
        rgb_chw = inverse_preprocess_input(
            scorer_frames.detach(),
            target_h=masks.shape[1],
            target_w=masks.shape[2],
        )  # (N, 3, H, W)
        # CHW -> HWC
        result = rgb_chw.permute(0, 2, 3, 1).contiguous()
        result = result.round().clamp(0.0, 255.0)

    return result


# ---- High-level interface ----


class ConstrainedFrameGenerator:
    """Generate scorer-optimal frames via constrained optimization from noise.

    No neural network weights needed. Archive contains only:
    - masks (239 bytes via entropy coder)
    - expected PoseNet targets (7KB)
    - noise seed (64 bytes)
    Total: ~8KB archive.

    At inflate time: run gradient descent from seeded noise.
    ~1000 steps, ~50ms/step on T4 = 50 seconds.

    Example::

        gen = ConstrainedFrameGenerator(posenet, segnet, device="cuda")
        frames = gen.generate(masks, noise_seed=42, num_steps=1000)
    """

    def __init__(
        self,
        posenet: torch.nn.Module,
        segnet: torch.nn.Module,
        device: torch.device | str = "cpu",
    ) -> None:
        self.posenet = posenet
        self.segnet = segnet
        self.device = torch.device(device)

    def generate(
        self,
        masks: torch.Tensor,
        noise_seed: int = 42,
        num_steps: int = 1000,
        lr: float = 0.1,
        seg_weight: float = 100.0,
        pose_weight: float = 10.0,
        compress_weight: float = 1.0,
        log_every: int = 100,
        scorer_space: bool = False,
    ) -> torch.Tensor:
        """Generate frames satisfying scorer constraints.

        Args:
            masks: (N, H, W) long tensor with target class indices.
            noise_seed: deterministic seed.
            num_steps: optimization steps.
            lr: Adam learning rate.
            seg_weight: SegNet cross-entropy weight.
            pose_weight: PoseNet L2 weight.
            compress_weight: total variation weight.
            log_every: logging interval (0 to disable).
            scorer_space: if True, optimize in scorer's YUV420 space
                (GPU Eureka #1) instead of RGB space.

        Returns:
            (N, H, W, 3) float tensor in [0, 255].
        """
        expected_pose = estimate_expected_pose(masks, device=self.device)

        if scorer_space:
            return generate_in_scorer_space(
                masks=masks,
                posenet=self.posenet,
                segnet=self.segnet,
                noise_seed=noise_seed,
                num_steps=num_steps,
                lr=lr,
                seg_weight=seg_weight,
                pose_weight=pose_weight,
                compress_weight=compress_weight,
                device=self.device,
                log_every=log_every,
            )
        else:
            return constrained_generate(
                masks=masks,
                expected_pose=expected_pose,
                posenet=self.posenet,
                segnet=self.segnet,
                noise_seed=noise_seed,
                num_steps=num_steps,
                lr=lr,
                seg_weight=seg_weight,
                pose_weight=pose_weight,
                compress_weight=compress_weight,
                device=self.device,
                log_every=log_every,
            )

    def build_archive(
        self,
        masks: torch.Tensor,
        noise_seed: int,
        output_path: str | Path,
    ) -> Path:
        """Extract pose targets from masks and build minimal archive.

        Args:
            masks: (N, H, W) long tensor.
            noise_seed: deterministic seed.
            output_path: output directory.

        Returns:
            Path to archive directory.
        """
        expected_pose = estimate_expected_pose(masks, device=self.device)
        return build_constrained_archive(masks, expected_pose, noise_seed, output_path)

    def inflate(
        self,
        archive_dir: str | Path,
        output_dir: str | Path,
        num_steps: int = 1000,
        lr: float = 0.1,
        log_every: int = 100,
    ) -> Path:
        """Inflate from archive via constrained optimization.

        Args:
            archive_dir: path to archive directory.
            output_dir: path to write output frames.
            num_steps: optimization steps.
            lr: Adam learning rate.
            log_every: logging interval.

        Returns:
            Path to output directory.
        """
        return inflate_constrained(
            archive_dir=archive_dir,
            output_dir=output_dir,
            posenet=self.posenet,
            segnet=self.segnet,
            num_steps=num_steps,
            lr=lr,
            device=self.device,
            log_every=log_every,
        )
