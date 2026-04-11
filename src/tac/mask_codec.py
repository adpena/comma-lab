"""Mask extraction, encoding, and decoding for the GPU-lane submission.

The GPU-lane pipeline replaces traditional video compression with:
    GT frames → SegNet → 5-class masks → AV1 encode → (tiny file) → AV1 decode → neural render

Segmentation masks are ideal for AV1: discrete values (0-4), enormous spatial
coherence, near-zero temporal change within road segments. Typical AV1 encoding
of 1200 frames of masks at 512x384 fits in ~200KB at CRF 20.

Codec options (pass codec= to encode/decode functions):
    - "av1": AV1 via ffmpeg (default, lossy, ~33KB at CRF 20)
    - "entropy": custom delta+RLE+LZMA coder (lossless, ~8-15KB)

Functions:
    - extract_masks: run frozen SegNet on frames, take argmax
    - encode_masks: write masks as AV1 video via ffmpeg
    - decode_masks: read AV1 video back to mask tensors
    - encode_masks_auto: encode with chosen codec
    - decode_masks_auto: decode with chosen codec
    - masks_to_pairs: build consecutive mask pairs for PairGenerator
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# SegNet native resolution — all masks are produced and consumed at this size
SEGNET_H = 384
SEGNET_W = 512
NUM_CLASSES = 5


def extract_masks(
    frames: list[torch.Tensor],
    segnet,
    device: str | torch.device = "cpu",
    batch_size: int = 4,
) -> torch.Tensor:
    """Run frozen SegNet on frames and extract argmax class masks.

    Args:
        frames: list of (H, W, 3) uint8 tensors (any resolution)
        segnet: frozen SegNet model (on device)
        device: computation device
        batch_size: frames per forward pass (memory vs speed tradeoff)

    Returns:
        (N, SEGNET_H, SEGNET_W) long tensor with values in [0, NUM_CLASSES)
    """
    masks = []
    segnet.eval()

    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]

            # Convert HWC uint8 → BCHW float and resize to SegNet resolution
            batch = []
            for f in batch_frames:
                # (H, W, 3) → (3, H, W) float
                t = f.float().permute(2, 0, 1).unsqueeze(0)
                # Resize to SegNet native resolution if needed
                if t.shape[2] != SEGNET_H or t.shape[3] != SEGNET_W:
                    t = F.interpolate(
                        t, size=(SEGNET_H, SEGNET_W),
                        mode="bilinear", align_corners=False,
                    )
                batch.append(t)

            batch_tensor = torch.cat(batch, dim=0).to(device)

            # SegNet expects (B, T=1, C, H, W) via preprocess_input
            # but we can also feed (B, C, H, W) directly if the model supports it.
            # Use preprocess_input for compatibility with the scorer pipeline.
            # Reshape to (B, 1, C, H, W) for preprocess_input
            bt_input = batch_tensor.unsqueeze(1)  # (B, 1, C, H, W)
            # SegNet preprocess expects (B, T, C, H, W)
            seg_input = segnet.preprocess_input(bt_input)
            logits = segnet(seg_input)  # (B, NUM_CLASSES, H, W)

            # Argmax to get discrete class labels
            class_masks = logits.argmax(dim=1).cpu()  # (B, H, W) long
            masks.append(class_masks)

    return torch.cat(masks, dim=0)


def encode_masks(
    masks: torch.Tensor,
    output_path: str | Path,
    crf: int = 20,
    fps: int = 20,
) -> int:
    """Encode segmentation masks as AV1 video using ffmpeg.

    Masks are discrete class labels (0-4), so we scale them to spread
    across the 0-255 range for maximum AV1 encoding fidelity:
        pixel_value = class_label * (255 // (NUM_CLASSES - 1))

    The scaling is invertible at decode time.

    Args:
        masks: (N, H, W) long tensor with values in [0, NUM_CLASSES)
        output_path: path for output .mp4 file
        crf: AV1 quality (lower = better quality, larger file)
        fps: frame rate for the encoded video

    Returns:
        File size in bytes
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    N, H, W = masks.shape

    # Scale class labels to byte range: 0→0, 1→63, 2→127, 3→191, 4→255
    scale_factor = 255 // (NUM_CLASSES - 1)
    pixels = (masks * scale_factor).clamp(0, 255).to(torch.uint8).numpy()

    # Write raw frames to ffmpeg stdin, encode as AV1
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "gray",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libsvtav1",
        "-crf", str(crf),
        "-preset", "6",
        # Disable loop filter + CDEF for categorical mask data (Fraunhofer/NTT/Habr convergence)
        # AV1's restoration filters are designed for natural images and actively harm
        # discrete class boundaries. Disabling them preserves sharp mask edges at zero cost.
        "-svtav1-params", "enable-restoration=0:enable-cdef=0",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path),
    ]

    proc = subprocess.run(
        cmd,
        input=pixels.tobytes(),
        capture_output=True,
        timeout=300,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg mask encoding failed (rc={proc.returncode}):\n"
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        )

    size = output_path.stat().st_size
    print(f"[mask_codec] Encoded {N} masks ({H}x{W}) → {output_path} ({size:,} bytes, CRF={crf})")
    return size


def decode_masks(
    mask_video_path: str | Path,
    expected_frames: int | None = None,
) -> torch.Tensor:
    """Decode AV1 mask video back to class label tensors.

    Inverts the scaling applied during encoding:
        class_label = round(pixel_value / (255 // (NUM_CLASSES - 1)))

    Args:
        mask_video_path: path to .mp4 mask video
        expected_frames: if set, verify frame count matches

    Returns:
        (N, H, W) long tensor with values in [0, NUM_CLASSES)
    """
    mask_video_path = Path(mask_video_path)
    if not mask_video_path.exists():
        raise FileNotFoundError(f"Mask video not found: {mask_video_path}")

    # Probe video dimensions
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames",
        "-of", "csv=p=0",
        str(mask_video_path),
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe.stderr}")

    parts = probe.stdout.strip().split(",")
    W, H = int(parts[0]), int(parts[1])

    # Decode to raw gray frames
    cmd = [
        "ffmpeg",
        "-i", str(mask_video_path),
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-v", "error",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg mask decoding failed:\n{proc.stderr.decode('utf-8', errors='replace')}"
        )

    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_size = H * W
    N = len(raw) // frame_size
    if len(raw) % frame_size != 0:
        raise ValueError(
            f"Decoded data size {len(raw)} not divisible by frame size {H}x{W}={frame_size}"
        )

    pixels = raw.reshape(N, H, W)

    # Invert scaling: pixel → class label
    scale_factor = 255 // (NUM_CLASSES - 1)
    # Round to nearest class (handles AV1 lossy compression artifacts)
    masks = np.round(pixels.astype(np.float32) / scale_factor).astype(np.int64)
    masks = np.clip(masks, 0, NUM_CLASSES - 1)

    result = torch.from_numpy(masks)

    if expected_frames is not None and N != expected_frames:
        print(f"[mask_codec] WARNING: expected {expected_frames} frames, got {N}")

    print(f"[mask_codec] Decoded {N} masks ({H}x{W}) from {mask_video_path}")
    return result


def masks_to_pairs(
    masks: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build consecutive mask pairs for PairGenerator.

    PoseNet evaluates pairs of frames. This function produces corresponding
    pairs of masks that PairGenerator can render into frame pairs.

    Args:
        masks: (N, H, W) long tensor

    Returns:
        list of (mask_t, mask_t+1) tuples, each (1, H, W) long
    """
    pairs = []
    for i in range(0, len(masks) - 1, 2):
        if i + 1 >= len(masks):
            break
        m_t = masks[i].unsqueeze(0)    # (1, H, W)
        m_t1 = masks[i + 1].unsqueeze(0)  # (1, H, W)
        pairs.append((m_t, m_t1))
    return pairs


def mask_pair_from_index(
    masks: torch.Tensor,
    start_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a single mask pair on-the-fly (lazy, memory-efficient).

    Args:
        masks: (N, H, W) long tensor
        start_idx: index of first mask in pair

    Returns:
        (mask_t, mask_t+1) each (1, H, W) long
    """
    return masks[start_idx].unsqueeze(0), masks[start_idx + 1].unsqueeze(0)


def encode_masks_auto(
    masks: torch.Tensor,
    output_path: str | Path,
    codec: str = "av1",
    **kwargs,
) -> int:
    """Encode masks using the chosen codec.

    Args:
        masks: (N, H, W) long tensor with values in [0, NUM_CLASSES)
        output_path: output file path (.mp4 for av1, .msk for entropy)
        codec: "av1" or "entropy"
        **kwargs: passed to underlying encoder (crf/fps for av1, backend for entropy)

    Returns:
        File size in bytes
    """
    if codec == "av1":
        return encode_masks(masks, output_path, **kwargs)
    elif codec == "entropy":
        from .mask_entropy_coder import encode_masks_entropy
        return encode_masks_entropy(masks, output_path, **kwargs)
    else:
        raise ValueError(f"Unknown mask codec: {codec!r} (use 'av1' or 'entropy')")


def decode_masks_auto(
    mask_path: str | Path,
    codec: str = "av1",
    **kwargs,
) -> torch.Tensor:
    """Decode masks using the chosen codec.

    Args:
        mask_path: path to encoded mask file
        codec: "av1" or "entropy"
        **kwargs: passed to underlying decoder

    Returns:
        (N, H, W) long tensor with values in [0, NUM_CLASSES)
    """
    if codec == "av1":
        return decode_masks(mask_path, **kwargs)
    elif codec == "entropy":
        from .mask_entropy_coder import decode_masks_entropy
        return decode_masks_entropy(mask_path, **kwargs)
    else:
        raise ValueError(f"Unknown mask codec: {codec!r} (use 'av1' or 'entropy')")


def measure_mask_rate(
    mask_video_path: str | Path,
    num_frames: int,
    frame_h: int = 874,
    frame_w: int = 1164,
) -> float:
    """Compute the rate contribution of the mask video.

    Rate = archive_size / uncompressed_size, where uncompressed = N * H * W * 3.
    The archive will also contain the neural renderer weights, so this is just
    the mask video contribution.

    Args:
        mask_video_path: path to encoded mask .mp4
        num_frames: number of original video frames
        frame_h: original frame height
        frame_w: original frame width

    Returns:
        Rate as float (typically 0.001 - 0.01 for masks alone)
    """
    mask_size = Path(mask_video_path).stat().st_size
    uncompressed_size = num_frames * frame_h * frame_w * 3
    rate = mask_size / uncompressed_size
    print(f"[mask_codec] Mask rate: {mask_size:,} / {uncompressed_size:,} = {rate:.6f}")
    return rate
