"""Scorer-gradient saliency computation (PoseNet/SegNet gradient maps).

Backpropagates through the scoring function to find which pixels matter most.
This produces the most scientifically grounded ROI map possible.

Output: per-frame importance maps based on scorer gradient magnitude.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

import av


def load_models(models_dir, device: str = "cpu"):
    """Load PoseNet and SegNet scorer models from safetensors files.

    Parameters
    ----------
    models_dir : Path-like
        Directory containing ``posenet.safetensors`` and ``segnet.safetensors``.
    device : str
        Torch device string.

    Returns
    -------
    tuple
        ``(posenet, segnet)`` in eval mode on *device*.
    """
    from pathlib import Path
    from safetensors.torch import load_file

    # Upstream modules must be importable (caller is responsible for sys.path)
    from modules import PoseNet, SegNet

    models_dir = Path(models_dir)

    posenet = PoseNet()
    posenet.load_state_dict(load_file(str(models_dir / "posenet.safetensors")))
    posenet.to(device).eval()

    segnet = SegNet()
    segnet.load_state_dict(load_file(str(models_dir / "segnet.safetensors")))
    segnet.to(device).eval()

    return posenet, segnet


def rgb_to_yuv6(frames: torch.Tensor) -> torch.Tensor:
    """Match frame_utils.py rgb_to_yuv6 exactly."""
    kYR, kYG, kYB = 0.299, 0.587, 0.114
    R, G, B = frames[..., 0], frames[..., 1], frames[..., 2]
    Y = (R * kYR + G * kYG + B * kYB).clamp_(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp_(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp_(0.0, 255.0)

    U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] +
             U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
    V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] +
             V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25

    y00 = Y[..., 0::2, 0::2]
    y10 = Y[..., 1::2, 0::2]
    y01 = Y[..., 0::2, 1::2]
    y11 = Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)


def compute_saliency(video_path: str, models_dir: str, output_path: str,
                     sample_step: int = 20, device: str = "cpu"):
    """Compute per-pixel saliency from PoseNet + SegNet gradients.

    Parameters
    ----------
    video_path : str
        Path to source video (``.mkv``).
    models_dir : str
        Directory with scorer safetensors.
    output_path : str
        Where to save the ``.npy`` saliency array.
    sample_step : int
        Only use every *sample_step*-th frame.
    device : str
        Torch device.

    Returns
    -------
    np.ndarray
        Saliency maps with shape ``(N, H, W)``.
    """
    from frame_utils import yuv420_to_rgb, camera_size, segnet_model_input_size

    posenet, segnet = load_models(models_dir, device)
    W, H = camera_size  # 1164, 874
    seg_W, seg_H = segnet_model_input_size  # 512, 384

    # Read frames from video
    container = av.open(video_path)
    stream = container.streams.video[0]
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i % sample_step == 0:
            t = yuv420_to_rgb(frame)  # (H, W, 3) uint8
            frames.append(t)
        if i % 300 == 0:
            print(f"  Read {i} frames ...", file=sys.stderr, flush=True)
    container.close()
    print(f"  Total keyframes: {len(frames)}", file=sys.stderr)

    saliency_maps = []

    for idx in range(0, len(frames) - 1, 2):
        if idx + 1 >= len(frames):
            break

        # Build a 2-frame sequence (matching evaluator's seq_len=2)
        f1 = frames[idx].float().to(device)   # even frame
        f2 = frames[idx + 1].float().to(device)  # odd frame (SegNet sees this)

        # Enable gradients on the input
        f1.requires_grad_(True)
        f2.requires_grad_(True)

        # Stack as batch: (1, 2, H, W, 3)
        seq = torch.stack([f1, f2], dim=0).unsqueeze(0)

        # Resize to scorer input: bilinear to 512x384
        seq_resized = seq.permute(0, 1, 4, 2, 3).float()  # (1, 2, 3, H, W)
        B, S, C, sH, sW = seq_resized.shape
        seq_flat = seq_resized.reshape(B * S, C, sH, sW)
        seq_small = F.interpolate(seq_flat, size=(seg_H, seg_W), mode='bilinear', align_corners=False)
        seq_small = seq_small.reshape(B, S, C, seg_H, seg_W)

        # Convert to YUV6
        seq_hwc = seq_small.permute(0, 1, 3, 4, 2)  # (B, S, H, W, C)
        yuv6 = rgb_to_yuv6(seq_hwc)  # (B, S, 6, H//2, W//2)
        yuv6_flat = yuv6.reshape(B, S * 6, seg_H // 2, seg_W // 2)

        # PoseNet forward — returns dict with 'pose' key
        pose_out = posenet(yuv6_flat)
        pose_tensor = pose_out['pose'] if isinstance(pose_out, dict) else pose_out
        pose_loss = pose_tensor[:, :pose_tensor.shape[1] // 2].pow(2).sum()

        # Backward to get gradient on input frames
        pose_loss.backward(retain_graph=False)

        # Extract gradient magnitude per pixel
        grad1 = f1.grad.abs().sum(dim=-1)  # (H, W) gradient magnitude
        grad2 = f2.grad.abs().sum(dim=-1)

        # Normalize
        combined = (grad1 + grad2) / 2
        gmax = combined.max()
        if gmax > 0:
            combined = combined / gmax

        saliency_maps.append(combined.detach().cpu().numpy())

        if len(saliency_maps) % 10 == 0:
            print(f"  Computed {len(saliency_maps)} saliency pairs ...", file=sys.stderr, flush=True)

        # Zero gradients
        f1.grad = None
        f2.grad = None

    saliency = np.stack(saliency_maps, axis=0)
    np.save(output_path, saliency.astype(np.float32))
    print(f"Saved saliency map: shape={saliency.shape}, mean={saliency.mean():.3f}", file=sys.stderr)
    return saliency
