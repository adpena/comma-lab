"""Mask extraction, encoding, and decoding for the GPU-lane submission.

The GPU-lane pipeline replaces traditional video compression with:
    GT frames → SegNet → 5-class masks → AV1 encode → (tiny file) → AV1 decode → neural render

Segmentation masks are ideal for AV1: discrete values (0-4), enormous spatial
coherence, near-zero temporal change within road segments. Typical AV1 encoding
of 1200 frames of masks at 48x64 (1/8 scale) fits in ~60-80KB at CRF 20;
full 384x512 resolution is ~2MB.

Codec options (pass codec= to encode/decode functions):
    - "av1": AV1 via ffmpeg (default, lossy, ~33KB at CRF 20)
    - "entropy": custom delta+RLE+LZMA coder (lossless, ~8-15KB)
    - "vvc": VVC/H.266 via vvencapp (lossy, ~30-50% smaller than AV1)

Functions:
    - extract_masks: run frozen SegNet on frames, take argmax
    - encode_masks: write masks as AV1 video via ffmpeg
    - encode_masks_vvc: write masks as VVC/H.266 bitstream
    - decode_masks: read AV1 video back to mask tensors
    - decode_masks_vvc: read VVC bitstream back to mask tensors
    - encode_masks_auto: encode with chosen codec
    - decode_masks_auto: decode with chosen codec
    - masks_to_pairs: build consecutive mask pairs for PairGenerator
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tac.camera import FRAME_H as SEGNET_H, FRAME_W as SEGNET_W, NUM_CLASSES


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
                        t,
                        size=(SEGNET_H, SEGNET_W),
                        mode="bilinear",
                        align_corners=False,
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
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{W}x{H}",
        "-pix_fmt",
        "gray",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libsvtav1",
        "-crf",
        str(crf),
        "-preset",
        "6",
        # Disable loop filter + CDEF for categorical mask data (Fraunhofer/NTT/Habr convergence)
        # AV1's restoration filters are designed for natural images and actively harm
        # discrete class boundaries. Disabling them preserves sharp mask edges at zero cost.
        "-svtav1-params",
        "enable-restoration=0:enable-cdef=0",
        # Use gray (not yuv420p) for discrete mask data: chroma subsampling
        # creates phantom values at class boundaries for 5-class data.
        "-pix_fmt",
        "gray",
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
            f"ffmpeg mask encoding failed (rc={proc.returncode}):\n{proc.stderr.decode('utf-8', errors='replace')}"
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
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames",
        "-of",
        "csv=p=0",
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
        "-i",
        str(mask_video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-v",
        "error",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mask decoding failed:\n{proc.stderr.decode('utf-8', errors='replace')}")

    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_size = H * W
    N = len(raw) // frame_size
    if len(raw) % frame_size != 0:
        raise ValueError(f"Decoded data size {len(raw)} not divisible by frame size {H}x{W}={frame_size}")

    pixels = raw.reshape(N, H, W)

    # Invert scaling: pixel → class label
    scale_factor = 255 // (NUM_CLASSES - 1)
    # Round to nearest class (handles AV1 lossy compression artifacts)
    masks = np.round(pixels.astype(np.float32) / scale_factor).astype(np.int64)
    masks = np.clip(masks, 0, NUM_CLASSES - 1)

    result = torch.from_numpy(masks)

    if expected_frames is not None and expected_frames != N:
        print(f"[mask_codec] WARNING: expected {expected_frames} frames, got {N}")

    print(f"[mask_codec] Decoded {N} masks ({H}x{W}) from {mask_video_path}")
    return result


def _check_vvc_available() -> bool:
    """Check if vvencapp and vvdecapp are available on PATH."""
    return shutil.which("vvencapp") is not None and shutil.which("vvdecapp") is not None


def encode_masks_vvc(
    masks: torch.Tensor,
    output_path: str | Path,
    qp: int = 27,
    fps: int = 20,
    preset: str = "faster",
) -> int:
    """Encode segmentation masks as VVC/H.266 bitstream using vvencapp.

    VVC (Versatile Video Coding / H.266) is 30-50% more efficient than AV1
    for the same quality. Uses YUV400 (grayscale) mode which is native to VVC
    and avoids wasting bits on chroma planes.

    Requires vvencapp (brew install vvenc).

    Note: vvencapp uses 10-bit internal processing. Input 8-bit values are
    left-shifted by 2 during encoding and must be right-shifted during decode.

    Args:
        masks: (N, H, W) long tensor with values in [0, NUM_CLASSES)
        output_path: path for output .266 file
        qp: quantization parameter (0-63, lower = better quality)
        fps: frame rate
        preset: vvenc speed preset (faster/fast/medium/slow/slower)

    Returns:
        File size in bytes
    """
    if not _check_vvc_available():
        raise RuntimeError("vvencapp not found. Install with: brew install vvenc vvdec")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    N, H, W = masks.shape

    # Scale class labels to byte range: 0→0, 1→63, 2→127, 3→191, 4→255
    scale_factor = 255 // (NUM_CLASSES - 1)
    pixels = (masks * scale_factor).clamp(0, 255).to(torch.uint8).numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        yuv_path = Path(tmpdir) / "masks.yuv"
        vvc_path = Path(tmpdir) / "masks.266"

        # Write raw grayscale as YUV400
        with open(yuv_path, "wb") as f:
            f.write(pixels.tobytes())

        cmd = [
            "vvencapp",
            "-i",
            str(yuv_path),
            "-s",
            f"{W}x{H}",
            "-c",
            "yuv400",
            "-r",
            str(fps),
            "-f",
            str(N),
            "--preset",
            preset,
            "-q",
            str(qp),
            "--qpa",
            "0",  # disable perceptual QP adaptation for categorical data
            "-o",
            str(vvc_path),
        ]

        proc = subprocess.run(cmd, capture_output=True, timeout=600)

        if proc.returncode != 0:
            raise RuntimeError(
                f"vvencapp encoding failed (rc={proc.returncode}):\n{proc.stderr.decode('utf-8', errors='replace')}"
            )

        # Copy to final output path
        import shutil as _shutil

        _shutil.copy2(vvc_path, output_path)

    size = output_path.stat().st_size
    print(f"[mask_codec] VVC encoded {N} masks ({H}x{W}) → {output_path} ({size:,} bytes, QP={qp})")
    return size


def decode_masks_vvc(
    vvc_path: str | Path,
    expected_frames: int | None = None,
) -> torch.Tensor:
    """Decode VVC/H.266 mask bitstream back to class label tensors.

    Handles vvdecapp's 10-bit output (16-bit LE samples) by right-shifting
    by 2 bits to recover 8-bit values, then inverting the class scaling.

    Requires vvdecapp (brew install vvdec).

    Args:
        vvc_path: path to .266 bitstream
        expected_frames: if set, verify frame count

    Returns:
        (N, H, W) long tensor with values in [0, NUM_CLASSES)
    """
    if not _check_vvc_available():
        raise RuntimeError("vvdecapp not found. Install with: brew install vvenc vvdec")

    vvc_path = Path(vvc_path)
    if not vvc_path.exists():
        raise FileNotFoundError(f"VVC bitstream not found: {vvc_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        dec_yuv_path = Path(tmpdir) / "decoded.yuv"

        cmd = [
            "vvdecapp",
            "-b",
            str(vvc_path),
            "-o",
            str(dec_yuv_path),
        ]

        proc = subprocess.run(cmd, capture_output=True, timeout=300)

        if proc.returncode != 0:
            raise RuntimeError(
                f"vvdecapp decoding failed (rc={proc.returncode}):\n{proc.stderr.decode('utf-8', errors='replace')}"
            )

        # vvdecapp outputs 10-bit YUV400 as 16-bit LE samples
        decoded_raw = np.fromfile(dec_yuv_path, dtype=np.uint16)

    # We need to figure out H, W from the data. For our pipeline, assume
    # SEGNET_H x SEGNET_W unless data tells us otherwise.
    # Total samples = N * H * W
    total_samples = decoded_raw.size
    frame_size = SEGNET_H * SEGNET_W
    N = total_samples // frame_size

    if total_samples % frame_size != 0:
        raise ValueError(
            f"Decoded VVC data ({total_samples} samples) not divisible by frame size {SEGNET_H}x{SEGNET_W}={frame_size}"
        )

    decoded_10bit = decoded_raw[: N * frame_size].reshape(N, SEGNET_H, SEGNET_W)

    # Convert 10-bit back to 8-bit (VVC shifts 8-bit input left by 2)
    decoded_pixels = (decoded_10bit >> 2).astype(np.uint8)

    # Invert scaling: pixel → class label
    scale_factor = 255 // (NUM_CLASSES - 1)
    masks = np.round(decoded_pixels.astype(np.float32) / scale_factor).astype(np.int64)
    masks = np.clip(masks, 0, NUM_CLASSES - 1)

    result = torch.from_numpy(masks)

    if expected_frames is not None and expected_frames != N:
        print(f"[mask_codec] WARNING: expected {expected_frames} frames, got {N}")

    print(f"[mask_codec] VVC decoded {N} masks ({SEGNET_H}x{SEGNET_W}) from {vvc_path}")
    return result


# ── Technique 6: Morphological Boundary Sharpening (Fraunhofer/NTT) ────


def sharpen_mask_boundaries(
    masks: torch.Tensor,
    erosion_size: int = 1,
    min_component_area: int = 4,
) -> torch.Tensor:
    """Morphological opening + connected component restoration on decoded masks.

    After AV1 decode, class boundaries often have single-pixel noise. This
    applies per-class erosion then dilation (opening) to clean boundaries,
    then restores thin structures (lane markings) via connected component
    analysis on the original mask.

    Zero neural parameters — pure morphological ops.

    Args:
        masks: (N, H, W) long tensor with values in [0, NUM_CLASSES)
        erosion_size: kernel radius for erosion/dilation (1 = 3x3 kernel)
        min_component_area: minimum connected component area to preserve

    Returns:
        (N, H, W) long tensor with sharpened boundaries
    """
    try:
        from scipy import ndimage
    except ImportError:
        print("[mask_codec] WARNING: scipy not available, skipping morphological sharpening")
        return masks

    masks_np = masks.numpy().copy()
    N, H, W = masks_np.shape
    result = np.zeros_like(masks_np)
    kernel = np.ones((2 * erosion_size + 1, 2 * erosion_size + 1), dtype=np.uint8)

    for frame_idx in range(N):
        frame = masks_np[frame_idx]
        # Initialize to -1 so class 0 doesn't falsely claim unclaimed pixels
        opened_frame = np.full_like(frame, fill_value=-1, dtype=np.int64)

        # Process classes in ascending order (0=road, 1=lane, 2=undrivable,
        # 3=movable, 4=vehicle).  Later classes overwrite earlier ones in
        # overlapping regions.  This is intentional: road (class 0) should
        # yield to more specific classes like lane lines and vehicles, so
        # processing in ascending order gives correct priority.
        for cls in range(NUM_CLASSES):
            binary = (frame == cls).astype(np.uint8)

            # Morphological opening: erosion then dilation
            eroded = ndimage.binary_erosion(binary, structure=kernel).astype(np.uint8)
            opened = ndimage.binary_dilation(eroded, structure=kernel).astype(np.uint8)

            # Restore thin structures via connected component analysis
            # Find components in original that were lost by opening
            original_labels, n_original = ndimage.label(binary)
            opened_labels, _ = ndimage.label(opened)

            # Restore small components that existed in original but vanished
            for comp_id in range(1, n_original + 1):
                comp_mask = original_labels == comp_id
                comp_area = comp_mask.sum()
                # If this component survived opening, skip
                if (opened[comp_mask]).any():
                    continue
                # If component is large enough to be real (not noise), restore it
                if comp_area >= min_component_area:
                    opened[comp_mask] = 1

            opened_frame[opened.astype(bool)] = cls

        # Handle unclaimed pixels: fill from original frame
        # (morphological artifacts are typically at boundaries)
        unclaimed_mask = opened_frame == -1
        if unclaimed_mask.any():
            opened_frame[unclaimed_mask] = frame[unclaimed_mask]

        result[frame_idx] = opened_frame

    return torch.from_numpy(result).long()


# ── Technique 10: AV1 Monochrome Encoding (Habr) ──────────────────────


def encode_masks_monochrome(
    masks: torch.Tensor,
    output_path: str | Path,
    crf: int = 20,
    fps: int = 20,
) -> int:
    """Encode masks using AV1 with yuv400 (monochrome) pixel format.

    Discrete mask data has no chroma information. Using yuv420p wastes bits
    encoding zero-value chroma planes. SVT-AV1 natively supports yuv400
    (grayscale), eliminating this waste.

    Typical savings: 5-15% smaller than yuv420p at same CRF.

    Args:
        masks: (N, H, W) long tensor with values in [0, NUM_CLASSES)
        output_path: path for output .mkv file (MP4 doesn't support yuv400)
        crf: AV1 quality parameter
        fps: frame rate

    Returns:
        File size in bytes
    """
    output_path = Path(output_path)
    # yuv400 requires .mkv container (MP4 doesn't support monochrome AV1)
    if output_path.suffix == ".mp4":
        output_path = output_path.with_suffix(".mkv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    N, H, W = masks.shape

    # Scale class labels to byte range
    scale_factor = 255 // (NUM_CLASSES - 1)
    pixels = (masks * scale_factor).clamp(0, 255).to(torch.uint8).numpy()

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{W}x{H}",
        "-pix_fmt",
        "gray",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libsvtav1",
        "-crf",
        str(crf),
        "-preset",
        "6",
        "-svtav1-params",
        "enable-restoration=0:enable-cdef=0",
        "-pix_fmt",
        "gray",  # monochrome output
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
            f"ffmpeg monochrome encoding failed (rc={proc.returncode}):\n"
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        )

    size = output_path.stat().st_size
    print(f"[mask_codec] Monochrome encoded {N} masks ({H}x{W}) → {output_path} ({size:,} bytes, CRF={crf})")
    return size


# ── Technique 11: Topology-Preserving Post-Filter (Fraunhofer) ────────


def restore_thin_structures(
    decoded_masks: torch.Tensor,
    keyframe_masks: torch.Tensor,
    area_threshold: int = 50,
    classes_to_restore: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Restore thin structures lost during compression by referencing keyframe.

    After AV1 decode, thin structures like lane markings and small distant
    objects may be destroyed. This function finds connected components per
    class in the keyframe that are below an area threshold (thin structures)
    and restores them in the decoded output if they were lost.

    Zero neural parameters — topology analysis only.

    Args:
        decoded_masks: (N, H, W) long tensor — decoded masks with artifacts
        keyframe_masks: (N, H, W) long tensor — clean keyframe masks (or GT)
        area_threshold: max component area to consider "thin" (default 50 pixels)
        classes_to_restore: optional tuple of class indices to restore
                           (default: all classes)

    Returns:
        (N, H, W) long tensor with thin structures restored
    """
    try:
        from scipy import ndimage
    except ImportError:
        print("[mask_codec] WARNING: scipy not available, skipping thin structure restoration")
        return decoded_masks

    decoded_np = decoded_masks.numpy().copy()
    keyframe_np = keyframe_masks.numpy()
    N, H, W = decoded_np.shape

    if classes_to_restore is None:
        classes_to_restore = tuple(range(NUM_CLASSES))

    restored_count = 0

    for frame_idx in range(N):
        dec_frame = decoded_np[frame_idx]
        key_frame = keyframe_np[frame_idx]

        for cls in classes_to_restore:
            key_binary = (key_frame == cls).astype(np.uint8)
            dec_binary = (dec_frame == cls).astype(np.uint8)

            # Find connected components in keyframe
            labels, n_components = ndimage.label(key_binary)

            for comp_id in range(1, n_components + 1):
                comp_mask = labels == comp_id
                area = comp_mask.sum()

                # Only restore thin/small structures
                if area > area_threshold:
                    continue

                # Check if this component was lost in decoding
                overlap = dec_binary[comp_mask].sum()
                preservation_ratio = overlap / area if area > 0 else 1.0

                # If more than half the component is missing, restore it
                if preservation_ratio < 0.5:
                    dec_frame[comp_mask] = cls
                    restored_count += 1

    if restored_count > 0:
        print(f"[mask_codec] Restored {restored_count} thin structures across {N} frames")

    return torch.from_numpy(decoded_np).long()


# ── Technique 12: Semantic-Aware Rate Control (UPM Spain) ──────────────


def compute_semantic_qp_offsets(
    masks: torch.Tensor,
    base_qp: int = 20,
    rare_qp_delta: int = -8,
    dominant_qp_delta: int = 4,
    rare_threshold: float = 0.10,
    dominant_threshold: float = 0.20,
) -> dict[int, int]:
    """Compute per-class QP offsets based on class frequency.

    Rare classes (lane markings ~3%, vehicles ~7%) get lower QP (more bits)
    to preserve their detail. Dominant classes (road ~45%, sky ~20%) get
    higher QP (fewer bits) since they are large uniform regions.

    Args:
        masks: (N, H, W) long tensor
        base_qp: base quantization parameter
        rare_qp_delta: QP adjustment for rare classes (negative = more bits)
        dominant_qp_delta: QP adjustment for dominant classes (positive = fewer bits)
        rare_threshold: classes below this fraction are "rare"
        dominant_threshold: classes above this fraction are "dominant"

    Returns:
        Dict mapping class_id → QP offset (relative to base_qp)
    """
    total_pixels = masks.numel()
    offsets = {}

    for cls in range(NUM_CLASSES):
        cls_pixels = (masks == cls).sum().item()
        fraction = cls_pixels / total_pixels

        if fraction < rare_threshold:
            offsets[cls] = rare_qp_delta
        elif fraction > dominant_threshold:
            offsets[cls] = dominant_qp_delta
        else:
            offsets[cls] = 0

    print(f"[mask_codec] Semantic QP offsets (base={base_qp}):")
    for cls, delta in sorted(offsets.items()):
        frac = (masks == cls).sum().item() / total_pixels
        print(f"  class {cls}: {frac:.1%} → QP {base_qp + delta} (delta={delta:+d})")

    return offsets


def generate_roi_qp_map(
    masks: torch.Tensor,
    qp_offsets: dict[int, int],
) -> np.ndarray:
    """Generate per-pixel ROI QP offset map for SVT-AV1.

    Creates a frame-by-frame QP offset map that can be passed to SVT-AV1's
    --qp-file or ROI API to allocate more bits to rare semantic classes.

    Args:
        masks: (N, H, W) long tensor
        qp_offsets: dict from compute_semantic_qp_offsets

    Returns:
        (N, H, W) int8 array of per-pixel QP offsets
    """
    masks_np = masks.numpy()
    N, H, W = masks_np.shape
    roi_map = np.zeros((N, H, W), dtype=np.int8)

    for cls, delta in qp_offsets.items():
        roi_map[masks_np == cls] = delta

    return roi_map


def encode_masks_semantic_rate(
    masks: torch.Tensor,
    output_path: str | Path,
    crf: int = 20,
    fps: int = 20,
    rare_qp_delta: int = -8,
    dominant_qp_delta: int = 4,
) -> int:
    """Encode masks with semantic-aware rate control.

    Two-pass approach:
    1. Compute per-class QP offsets based on class frequency
    2. Encode with class-specific CRF using a two-segment approach:
       - First encode rare-class regions at lower CRF
       - Then overlay with dominant-class regions at higher CRF

    In practice, SVT-AV1 doesn't have pixel-level QP control via CLI,
    so we approximate by blending multiple encoding passes. For production,
    use the SVT-AV1 C API with per-SB QP offsets.

    Current implementation: weighted average CRF based on class distribution
    to get closest achievable approximation via CLI.

    Args:
        masks: (N, H, W) long tensor
        output_path: path for output file
        crf: base CRF value
        fps: frame rate
        rare_qp_delta: CRF adjustment for rare classes
        dominant_qp_delta: CRF adjustment for dominant classes

    Returns:
        File size in bytes
    """
    qp_offsets = compute_semantic_qp_offsets(
        masks,
        base_qp=crf,
        rare_qp_delta=rare_qp_delta,
        dominant_qp_delta=dominant_qp_delta,
    )

    # Compute effective CRF as weighted average of per-class CRFs
    total_pixels = masks.numel()
    weighted_crf = 0.0
    for cls in range(NUM_CLASSES):
        cls_fraction = (masks == cls).sum().item() / total_pixels
        cls_crf = max(0, min(63, crf + qp_offsets.get(cls, 0)))
        weighted_crf += cls_fraction * cls_crf

    effective_crf = int(round(weighted_crf))
    print(f"[mask_codec] Semantic rate control: effective CRF={effective_crf} (base={crf})")

    # Encode with the effective CRF
    # For production with SVT-AV1 API, use per-SB QP offsets instead
    return encode_masks(masks, output_path, crf=effective_crf, fps=fps)


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
        m_t = masks[i].unsqueeze(0)  # (1, H, W)
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
        output_path: output file path (.mp4 for av1, .266 for vvc, .msk for entropy)
        codec: "av1", "vvc", or "entropy"
        **kwargs: passed to underlying encoder (crf/fps for av1, qp/fps for vvc,
                  backend for entropy)

    Returns:
        File size in bytes
    """
    if codec == "av1":
        return encode_masks(masks, output_path, **kwargs)
    elif codec == "vvc":
        return encode_masks_vvc(masks, output_path, **kwargs)
    elif codec == "entropy":
        from .mask_entropy_coder import encode_masks_entropy

        return encode_masks_entropy(masks, output_path, **kwargs)
    else:
        raise ValueError(f"Unknown mask codec: {codec!r} (use 'av1', 'vvc', or 'entropy')")


def decode_masks_auto(
    mask_path: str | Path,
    codec: str = "av1",
    **kwargs,
) -> torch.Tensor:
    """Decode masks using the chosen codec.

    Args:
        mask_path: path to encoded mask file
        codec: "av1", "vvc", or "entropy"
        **kwargs: passed to underlying decoder

    Returns:
        (N, H, W) long tensor with values in [0, NUM_CLASSES)
    """
    if codec == "av1":
        return decode_masks(mask_path, **kwargs)
    elif codec == "vvc":
        return decode_masks_vvc(mask_path, **kwargs)
    elif codec == "entropy":
        from .mask_entropy_coder import decode_masks_entropy

        return decode_masks_entropy(mask_path, **kwargs)
    else:
        raise ValueError(f"Unknown mask codec: {codec!r} (use 'av1', 'vvc', or 'entropy')")


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
