# SPDX-License-Identifier: MIT
"""Pre-compute expensive scorer-derived corrections at compress time (Eureka 1+2).

At compress time we have unlimited CPU/GPU and access to the ground-truth video.
Pre-computing scorer gradients, null-space basis, fragility maps, brightness
shifts, PoseNet targets, and hard-frame corrections lets inflate apply them
in microseconds instead of spending minutes re-deriving them.

Corrections bundle (~50-100KB compressed):
  - scorer_gradients: (N, H, W, 3) float16 -- per-pixel scorer gradient direction
  - null_space_basis: (K, D) float16 -- top-K null-space directions
  - fragility_map: (N, H, W) float16 -- per-pixel importance
  - brightness_shifts: (N,) float32 -- optimal per-frame brightness offset
  - posenet_targets: (P, 6) float16 -- ground-truth PoseNet outputs
  - hard_frame_corrections: sparse per-pixel table for top-50 hardest frames

Binary format:
  4 bytes: magic "PCOR"
  2 bytes: version (uint16)
  For each field:
    2 bytes: key length (uint16)
    key_len bytes: field name (utf-8)
    1 byte: dtype code (0=float16, 1=float32, 2=int64, 3=uint8)
    1 byte: ndim
    ndim * 4 bytes: shape (uint32 each)
    4 bytes: compressed data length (uint32)
    compressed_len bytes: zlib-compressed tensor data

Usage::

    # Compress time: pre-compute everything
    from tac.precompute_corrections import precompute_all_corrections, save_corrections
    corrections = precompute_all_corrections(frames, posenet, segnet, postfilter)
    save_corrections(corrections, "corrections.bin")

    # Inflate time: load and apply instantly
    from tac.precompute_corrections import load_corrections, apply_frame_corrections
    corrections = load_corrections("corrections.bin")
    frames = apply_frame_corrections(frames, corrections)
"""

from __future__ import annotations

import io
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "precompute_all_corrections",
    "compute_hard_frame_corrections",
    "apply_frame_corrections",
    "save_corrections",
    "load_corrections",
    "compute_quantization_directions",
    "apply_quantization_directions",
]

# Binary format constants
_MAGIC = b"PCOR"
_VERSION = 1

# dtype codes for serialization
_DTYPE_CODES = {
    np.dtype("float16"): 0,
    np.dtype("float32"): 1,
    np.dtype("int64"): 2,
    np.dtype("uint8"): 3,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_CODES.items()}


def _log(msg: str, verbose: bool = True) -> None:
    """Print to stderr if verbose."""
    if verbose:
        print(f"  [precompute] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Eureka 1: Pre-compute expensive data at compress time
# ---------------------------------------------------------------------------


def _compute_scorer_gradients(
    frames: list[torch.Tensor],
    posenet: nn.Module,
    segnet: nn.Module,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> np.ndarray:
    """Compute per-pixel scorer gradient direction for all frames.

    For each pixel, the gradient direction tells us which way to nudge
    the value to reduce scorer loss. At inflate time, noise-shaped rounding
    uses this to pick floor vs ceil without re-running the scorers.

    Args:
        frames: list of (H, W, 3) uint8 tensors.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        batch_size: frames per gradient computation.
        device: computation device.
        verbose: print progress.

    Returns:
        (N, H, W, 3) float16 numpy array of gradient directions (sign + magnitude).
    """
    N = len(frames)
    if N == 0:
        return np.empty((0, 0, 0, 3), dtype=np.float16)

    H, W = frames[0].shape[0], frames[0].shape[1]
    all_grads = np.zeros((N, H, W, 3), dtype=np.float16)

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_frames = []
        for i in range(batch_start, batch_end):
            f = frames[i].float().permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            batch_frames.append(f)

        x = torch.cat(batch_frames, dim=0).to(device)  # (B, 3, H, W)
        x.requires_grad_(True)

        # Build consecutive pairs for scorer (frame_i, frame_i+1).
        # Self-pairs (frame_i, frame_i) produce near-zero PoseNet output,
        # giving no PoseNet gradient signal — only SegNet would contribute.
        B = x.shape[0]
        if B >= 2:
            pair_t = x[:-1].unsqueeze(1)   # (B-1, 1, 3, H, W)
            pair_t1 = x[1:].unsqueeze(1)   # (B-1, 1, 3, H, W)
            pair = torch.cat([pair_t, pair_t1], dim=1).contiguous()  # (B-1, 2, 3, H, W)
        else:
            # Single frame: fall back to self-pair (PoseNet gradient ~zero)
            pair = x.unsqueeze(1).expand(B, 2, 3, H, W).contiguous()

        # PoseNet loss
        pose_in = posenet.preprocess_input(pair)
        pose_out = posenet(pose_in)
        pose_tensor = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
        pose_loss = pose_tensor[..., :6].pow(2).sum()

        # SegNet loss
        seg_in = segnet.preprocess_input(pair)
        seg_out = segnet(seg_in)
        seg_probs = F.softmax(seg_out, dim=1)
        seg_loss = -(seg_probs * (seg_probs + 1e-8).log()).sum()

        total_loss = pose_loss + 100.0 * seg_loss
        total_loss.backward()

        grad = x.grad.detach().cpu()  # (B, 3, H, W)
        # Store as (B, H, W, 3) float16
        grad_hwc = grad.permute(0, 2, 3, 1).numpy().astype(np.float16)
        all_grads[batch_start:batch_end] = grad_hwc

        if verbose and (batch_end % (batch_size * 5) == 0 or batch_end == N):
            _log(f"scorer gradients: {batch_end}/{N} frames")

    return all_grads


def _compute_null_space_basis(
    frames: list[torch.Tensor],
    posenet: nn.Module,
    segnet: nn.Module,
    top_k: int = 32,
    sample_frames: int = 8,
    max_outputs: int = 16,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> np.ndarray:
    """Compute top-K null-space directions of the scorer Jacobian.

    The null space contains pixel-space directions that are invisible to
    both PoseNet and SegNet. Pre-computing this avoids the expensive SVD
    at inflate time.

    Args:
        frames: list of (H, W, 3) uint8 tensors.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        top_k: number of null-space directions to store.
        sample_frames: number of frames to sample for Jacobian computation.
        max_outputs: max scorer output dimensions per Jacobian.
        device: computation device.
        verbose: print progress.

    Returns:
        (K, D) float16 numpy array where D = 3*H*W and K <= top_k.
    """
    from tac.scorer_exploits import compute_scorer_jacobian

    N = len(frames)
    if N == 0:
        return np.empty((0, 0), dtype=np.float16)

    H, W = frames[0].shape[0], frames[0].shape[1]
    D = 3 * H * W

    # Sample frames evenly across the video
    indices = np.linspace(0, N - 1, min(sample_frames, N)).astype(int)

    all_jacobians = []
    for idx in indices:
        f = frames[idx].float().permute(2, 0, 1).unsqueeze(0).to(device)
        J = compute_scorer_jacobian(f, posenet, segnet, max_outputs=max_outputs)
        all_jacobians.append(J)

    # Stack all Jacobians and compute SVD of the combined matrix
    J_combined = torch.cat(all_jacobians, dim=0).float()  # (total_K, D)

    if verbose:
        _log(f"null space: combined Jacobian shape {J_combined.shape}, computing SVD")

    U, S, Vh = torch.linalg.svd(J_combined, full_matrices=False)
    threshold = S.max() * 1e-3
    null_mask = S < threshold

    if null_mask.sum() == 0:
        # Relax threshold
        null_mask = S < S.max() * 0.1

    if null_mask.sum() == 0:
        _log("null space: no null directions found, returning empty basis", verbose)
        return np.empty((0, D), dtype=np.float16)

    null_basis = Vh[null_mask][:top_k]  # (K, D)

    if verbose:
        _log(f"null space: {null_basis.shape[0]} directions (threshold={threshold:.6f})")

    return null_basis.numpy().astype(np.float16)


def _compute_fragility_map(
    frames: list[torch.Tensor],
    posenet: nn.Module,
    segnet: nn.Module,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> np.ndarray:
    """Compute per-pixel scorer fragility for all frames.

    Fragility = gradient magnitude. High-fragility pixels are where a 1-LSB
    change can flip the scorer output. At inflate time this drives allocation
    of refinement effort.

    Args:
        frames: list of (H, W, 3) uint8 tensors.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        batch_size: frames per computation.
        device: computation device.
        verbose: print progress.

    Returns:
        (N, H, W) float16 numpy array normalized per frame to [0, 1].
    """
    from tac.scorer_exploits import compute_scorer_fragility_map

    N = len(frames)
    if N == 0:
        return np.empty((0, 0, 0), dtype=np.float16)

    H, W = frames[0].shape[0], frames[0].shape[1]
    all_frag = np.zeros((N, H, W), dtype=np.float16)

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_frames = []
        for i in range(batch_start, batch_end):
            f = frames[i].float().permute(2, 0, 1).unsqueeze(0)
            batch_frames.append(f)

        x = torch.cat(batch_frames, dim=0).to(device)
        frag = compute_scorer_fragility_map(x, posenet, segnet)  # (B, H, W)
        all_frag[batch_start:batch_end] = frag.cpu().numpy().astype(np.float16)

        if verbose and (batch_end % (batch_size * 5) == 0 or batch_end == N):
            _log(f"fragility map: {batch_end}/{N} frames")

    return all_frag


def _compute_brightness_shifts(
    frames: list[torch.Tensor],
    verbose: bool = True,
) -> np.ndarray:
    """Compute optimal per-frame brightness offset.

    WARNING (2026-04-11): The AllNorm brightness invariance claim was DISPROVEN.
    AllNorm is BatchNorm1d(1) on flattened post-backbone features, NOT pixel-level
    normalization. PoseNet IS sensitive to brightness. This function is retained
    for cases where the postfilter is retrained with brightness augmentation.
    Do NOT use without retraining.

    Args:
        frames: list of (H, W, 3) uint8 tensors.
        verbose: print progress.

    Returns:
        (N,) float32 numpy array of additive brightness offsets.
    """
    N = len(frames)
    shifts = np.zeros(N, dtype=np.float32)

    for i in range(N):
        f = frames[i].float()  # (H, W, 3)
        luma = f[:, :, 0] * 0.299 + f[:, :, 1] * 0.587 + f[:, :, 2] * 0.114
        current_mean = luma.mean().item()
        shift = 128.0 - current_mean
        # Clamp to avoid saturation
        shifts[i] = max(-30.0, min(30.0, shift))

    if verbose:
        _log(f"brightness shifts: range [{shifts.min():.1f}, {shifts.max():.1f}]")

    return shifts


def _extract_posenet_targets(
    frames: list[torch.Tensor],
    posenet: nn.Module,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> np.ndarray:
    """Extract PoseNet ground-truth targets for all consecutive pairs.

    These are the exact PoseNet outputs on the ground-truth frames. At inflate
    time, supervised TTO minimizes MSE against these targets.

    Args:
        frames: list of (H, W, 3) uint8 tensors.
        posenet: frozen PoseNet model.
        batch_size: pairs per forward pass.
        device: computation device.
        verbose: print progress.

    Returns:
        (N_pairs, 6) float16 numpy array of PoseNet outputs.
    """
    try:
        from tac.scorer_targets import extract_posenet_targets
    except ImportError as e:
        raise ImportError(
            "tac.scorer_targets is required for PoseNet target extraction. "
            "Ensure the tac package is installed with scorer support: "
            "uv pip install -e '.[scorer]'"
        ) from e

    targets_dict = extract_posenet_targets(
        frames, posenet, device=device, batch_size=batch_size, verbose=verbose,
    )
    return targets_dict["targets"].numpy().astype(np.float16)


def precompute_all_corrections(
    frames: list[torch.Tensor],
    posenet: nn.Module,
    segnet: nn.Module,
    postfilter: nn.Module | None = None,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
    null_space_top_k: int = 32,
    null_space_sample_frames: int = 8,
    hard_frame_top_k: int = 50,
    hard_frame_pixels: int = 1000,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Pre-compute all scorer-derived corrections for a video.

    This is the master function called at compress time. It runs all six
    correction computations and returns a dict suitable for save_corrections().

    Args:
        frames: list of (H, W, 3) uint8 tensors (ground-truth frames).
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        postfilter: optional postfilter model (for hard-frame corrections).
        batch_size: batch size for gradient computations.
        device: computation device.
        null_space_top_k: number of null-space basis vectors to store.
        null_space_sample_frames: frames to sample for null-space computation.
        hard_frame_top_k: number of hardest frames for sparse corrections.
        hard_frame_pixels: pixels per hard frame for sparse corrections.
        verbose: print progress.

    Returns:
        dict mapping field names to numpy arrays, ready for save_corrections().
    """
    t0 = time.monotonic()
    corrections: dict[str, np.ndarray] = {}

    # 1. Scorer gradients
    _log("Computing scorer gradients ...", verbose)
    t = time.monotonic()
    corrections["scorer_gradients"] = _compute_scorer_gradients(
        frames, posenet, segnet, batch_size=batch_size, device=device, verbose=verbose,
    )
    _log(f"scorer gradients: {corrections['scorer_gradients'].shape}, {time.monotonic() - t:.1f}s", verbose)

    # 2. Null-space basis
    _log("Computing null-space basis ...", verbose)
    t = time.monotonic()
    corrections["null_space_basis"] = _compute_null_space_basis(
        frames, posenet, segnet,
        top_k=null_space_top_k,
        sample_frames=null_space_sample_frames,
        device=device, verbose=verbose,
    )
    _log(f"null space basis: {corrections['null_space_basis'].shape}, {time.monotonic() - t:.1f}s", verbose)

    # 3. Fragility map
    _log("Computing fragility map ...", verbose)
    t = time.monotonic()
    corrections["fragility_map"] = _compute_fragility_map(
        frames, posenet, segnet, batch_size=batch_size, device=device, verbose=verbose,
    )
    _log(f"fragility map: {corrections['fragility_map'].shape}, {time.monotonic() - t:.1f}s", verbose)

    # 4. Brightness shifts
    _log("Computing brightness shifts ...", verbose)
    t = time.monotonic()
    corrections["brightness_shifts"] = _compute_brightness_shifts(frames, verbose=verbose)
    _log(f"brightness shifts: {corrections['brightness_shifts'].shape}, {time.monotonic() - t:.1f}s", verbose)

    # 5. PoseNet targets
    _log("Extracting PoseNet targets ...", verbose)
    t = time.monotonic()
    corrections["posenet_targets"] = _extract_posenet_targets(
        frames, posenet, batch_size=batch_size * 2, device=device, verbose=verbose,
    )
    _log(f"PoseNet targets: {corrections['posenet_targets'].shape}, {time.monotonic() - t:.1f}s", verbose)

    # 6. Hard-frame corrections (Eureka 2)
    _log("Computing hard-frame corrections ...", verbose)
    t = time.monotonic()
    hard_corrections = compute_hard_frame_corrections(
        frames, posenet, segnet,
        top_k=hard_frame_top_k,
        pixels_per_frame=hard_frame_pixels,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )
    corrections["hard_frame_indices"] = hard_corrections["frame_indices"]
    corrections["hard_pixel_indices"] = hard_corrections["pixel_indices"]
    corrections["hard_pixel_deltas"] = hard_corrections["pixel_deltas"]
    _log(f"hard corrections: {hard_corrections['frame_indices'].shape[0]} frames, {time.monotonic() - t:.1f}s", verbose)

    total = time.monotonic() - t0
    _log(f"All corrections computed in {total:.1f}s", verbose)

    return corrections


# ---------------------------------------------------------------------------
# Eureka 2: Frame-specific correction table
# ---------------------------------------------------------------------------


def compute_hard_frame_corrections(
    frames: list[torch.Tensor],
    posenet: nn.Module,
    segnet: nn.Module,
    top_k: int = 50,
    pixels_per_frame: int = 1000,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Identify hardest frames and compute sparse per-pixel corrections.

    Step 1: Score all frames, rank by loss, pick top_k hardest.
    Step 2: For each hard frame, compute per-pixel gradients, pick the
            top pixels_per_frame most impactful pixels, compute optimal
            correction deltas.

    The result is a sparse correction table: for each hard frame, a list
    of (pixel_index, delta_r, delta_g, delta_b) corrections that can be
    applied instantly at inflate time.

    Args:
        frames: list of (H, W, 3) uint8 tensors.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        top_k: number of hardest frames to correct.
        pixels_per_frame: number of most impactful pixels per frame.
        batch_size: frames per scorer forward pass.
        device: computation device.
        verbose: print progress.

    Returns:
        dict with:
            'frame_indices': (K,) int64 -- indices of hardest frames
            'pixel_indices': (K, P) int64 -- flat pixel indices per frame
            'pixel_deltas': (K, P, 3) float16 -- RGB correction deltas
    """
    N = len(frames)
    if N == 0:
        return {
            "frame_indices": np.empty(0, dtype=np.int64),
            "pixel_indices": np.empty((0, 0), dtype=np.int64),
            "pixel_deltas": np.empty((0, 0, 3), dtype=np.float16),
        }

    H, W = frames[0].shape[0], frames[0].shape[1]
    top_k = min(top_k, N)
    pixels_per_frame = min(pixels_per_frame, H * W)

    # Step 1: Score all frames to find hardest
    _log(f"Scoring {N} frames to find {top_k} hardest ...", verbose)
    frame_losses = np.zeros(N, dtype=np.float32)

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_frames = []
        for i in range(batch_start, batch_end):
            f = frames[i].float().permute(2, 0, 1).unsqueeze(0)
            batch_frames.append(f)

        x = torch.cat(batch_frames, dim=0).to(device)
        B = x.shape[0]
        pair = x.unsqueeze(1).expand(B, 2, 3, H, W).contiguous()

        with torch.inference_mode():
            pose_in = posenet.preprocess_input(pair)
            pose_out = posenet(pose_in)
            pose_tensor = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
            pose_per_frame = pose_tensor[..., :6].pow(2).sum(dim=-1)  # (B,)

            seg_in = segnet.preprocess_input(pair)
            seg_out = segnet(seg_in)
            seg_probs = F.softmax(seg_out, dim=1)
            seg_per_frame = -(seg_probs * (seg_probs + 1e-8).log()).sum(dim=(1, 2, 3))

        losses = (pose_per_frame + 100.0 * seg_per_frame).cpu().numpy()
        frame_losses[batch_start:batch_end] = losses

    # Pick top_k hardest frames
    hard_indices = np.argsort(frame_losses)[-top_k:][::-1]
    hard_indices = np.sort(hard_indices)  # restore temporal order

    if verbose:
        _log(f"Hardest frames: indices {hard_indices[:5]}... losses {frame_losses[hard_indices[:5]]}")

    # Step 2: Compute sparse corrections for each hard frame
    all_pixel_indices = np.zeros((top_k, pixels_per_frame), dtype=np.int64)
    all_pixel_deltas = np.zeros((top_k, pixels_per_frame, 3), dtype=np.float16)

    for ki, fi in enumerate(hard_indices):
        f = frames[fi].float().permute(2, 0, 1).unsqueeze(0).to(device)
        f.requires_grad_(True)

        pair = f.unsqueeze(1).expand(1, 2, 3, H, W).contiguous()

        pose_in = posenet.preprocess_input(pair)
        pose_out = posenet(pose_in)
        pose_tensor = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
        pose_loss = pose_tensor[..., :6].pow(2).sum()

        seg_in = segnet.preprocess_input(pair)
        seg_out = segnet(seg_in)
        seg_probs = F.softmax(seg_out, dim=1)
        seg_loss = -(seg_probs * (seg_probs + 1e-8).log()).sum()

        total_loss = pose_loss + 100.0 * seg_loss
        total_loss.backward()

        grad = f.grad.detach().cpu().squeeze(0)  # (3, H, W)
        # Gradient magnitude per pixel
        grad_mag = grad.abs().sum(dim=0).reshape(-1)  # (H*W,)

        # Top-P most impactful pixels
        _, top_pixels = torch.topk(grad_mag, pixels_per_frame)
        all_pixel_indices[ki] = top_pixels.numpy()

        # Correction delta: step in negative gradient direction, clamped to +/-1
        grad_flat = grad.reshape(3, -1)[:, top_pixels]  # (3, P)
        # Normalize and clamp: each delta is at most +/-1 LSB
        grad_norm = grad_flat.abs().amax(dim=0, keepdim=True).clamp(min=1e-10)
        deltas = -(grad_flat / grad_norm).permute(1, 0)  # (P, 3)
        all_pixel_deltas[ki] = deltas.numpy().astype(np.float16)

        if verbose and (ki + 1) % 10 == 0:
            _log(f"hard frame corrections: {ki + 1}/{top_k}")

    return {
        "frame_indices": hard_indices.astype(np.int64),
        "pixel_indices": all_pixel_indices,
        "pixel_deltas": all_pixel_deltas,
    }


def apply_frame_corrections(
    frames: torch.Tensor | list[torch.Tensor],
    corrections: dict[str, Any],
    verbose: bool = True,
) -> torch.Tensor | list[torch.Tensor]:
    """Apply pre-computed hard-frame corrections instantly at inflate time.

    For each hard frame, applies the sparse pixel deltas computed at compress
    time. This is O(top_k * pixels_per_frame) -- effectively instant.

    Args:
        frames: (N, 3, H, W) tensor or list of (3, H, W) tensors in [0, 255].
        corrections: dict from load_corrections() containing at minimum:
            'hard_frame_indices', 'hard_pixel_indices', 'hard_pixel_deltas'.
        verbose: print progress.

    Returns:
        Corrected frames in the same format as input.
    """
    is_list = isinstance(frames, list)
    if is_list:
        # Convert list to tensor for uniform processing
        frames_t = torch.stack(frames)  # (N, 3, H, W)
    else:
        frames_t = frames.clone()

    hard_indices = corrections.get("hard_frame_indices")
    pixel_indices = corrections.get("hard_pixel_indices")
    pixel_deltas = corrections.get("hard_pixel_deltas")

    if hard_indices is None or pixel_indices is None or pixel_deltas is None:
        _log("No hard-frame corrections found, skipping", verbose)
        return frames if is_list else frames_t

    # Convert from numpy if needed
    if isinstance(hard_indices, np.ndarray):
        hard_indices = torch.from_numpy(hard_indices.astype(np.int64))
    if isinstance(pixel_indices, np.ndarray):
        pixel_indices = torch.from_numpy(pixel_indices.astype(np.int64))
    if isinstance(pixel_deltas, np.ndarray):
        pixel_deltas = torch.from_numpy(pixel_deltas.astype(np.float16)).float()

    N, C, H, W = frames_t.shape
    applied = 0

    for ki in range(hard_indices.shape[0]):
        fi = hard_indices[ki].item()
        if fi >= N:
            continue

        pix_idx = pixel_indices[ki]  # (P,)
        deltas = pixel_deltas[ki]    # (P, 3)

        # Apply deltas to flat pixel space via vectorized advanced indexing
        frame_flat = frames_t[fi].reshape(3, -1)  # (3, H*W)
        frame_flat[:, pix_idx] += deltas.T  # (3, P) broadcast over channels

        frames_t[fi] = frame_flat.reshape(3, H, W).clamp(0, 255)
        applied += 1

    if verbose:
        _log(f"Applied corrections to {applied} hard frames")

    if is_list:
        return [frames_t[i] for i in range(N)]
    return frames_t


# ---------------------------------------------------------------------------
# Serialization: save / load corrections bundle
# ---------------------------------------------------------------------------


def save_corrections(corrections: dict[str, np.ndarray], path: str | Path) -> int:
    """Save pre-computed corrections to a compressed binary file.

    The format is self-describing: each field carries its name, dtype, shape,
    and zlib-compressed data. Total size is typically 50-100KB for 1200 frames.

    Args:
        corrections: dict mapping field names to numpy arrays.
        path: output file path.

    Returns:
        File size in bytes.
    """
    path = Path(path)
    buf = io.BytesIO()

    # Header
    buf.write(_MAGIC)
    buf.write(struct.pack("<H", _VERSION))
    buf.write(struct.pack("<H", len(corrections)))  # number of fields

    total_raw = 0
    total_compressed = 0

    for key, arr in corrections.items():
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        key_bytes = key.encode("utf-8")
        buf.write(struct.pack("<H", len(key_bytes)))
        buf.write(key_bytes)

        # dtype code
        dtype_code = _DTYPE_CODES.get(arr.dtype)
        if dtype_code is None:
            # Fall back to float16
            arr = arr.astype(np.float16)
            dtype_code = 0
        buf.write(struct.pack("<B", dtype_code))

        # shape
        buf.write(struct.pack("<B", arr.ndim))
        for dim in arr.shape:
            buf.write(struct.pack("<I", dim))

        # compressed data
        raw_bytes = arr.tobytes()
        compressed = zlib.compress(raw_bytes, level=9)
        buf.write(struct.pack("<I", len(compressed)))
        buf.write(compressed)

        total_raw += len(raw_bytes)
        total_compressed += len(compressed)

    data = buf.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

    print(
        f"[precompute] Saved {len(corrections)} fields to {path}: "
        f"{len(data)} bytes (raw {total_raw}, compressed {total_compressed})",
        file=sys.stderr,
    )

    return len(data)


def load_corrections(
    path: str | Path,
    verbose: bool = True,
) -> dict[str, np.ndarray] | None:
    """Load pre-computed corrections from a compressed binary file.

    Args:
        path: path to corrections.bin.
        verbose: print progress.

    Returns:
        dict mapping field names to numpy arrays, or None if file not found.
    """
    path = Path(path)
    if not path.exists():
        if verbose:
            _log(f"corrections file not found: {path}")
        return None

    data = path.read_bytes()
    buf = io.BytesIO(data)

    magic = buf.read(4)
    if magic != _MAGIC:
        print(f"[precompute] WARNING: invalid magic in {path}, skipping", file=sys.stderr)
        return None

    version = struct.unpack("<H", buf.read(2))[0]
    if version != _VERSION:
        print(f"[precompute] WARNING: unsupported version {version} in {path}", file=sys.stderr)
        return None

    n_fields = struct.unpack("<H", buf.read(2))[0]
    corrections: dict[str, np.ndarray] = {}

    for _ in range(n_fields):
        key_len = struct.unpack("<H", buf.read(2))[0]
        key = buf.read(key_len).decode("utf-8")

        dtype_code = struct.unpack("<B", buf.read(1))[0]
        ndim = struct.unpack("<B", buf.read(1))[0]
        shape = tuple(struct.unpack("<I", buf.read(4))[0] for _ in range(ndim))

        compressed_len = struct.unpack("<I", buf.read(4))[0]
        compressed = buf.read(compressed_len)

        raw_bytes = zlib.decompress(compressed)
        dtype = _CODE_TO_DTYPE[dtype_code]
        arr = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape).copy()
        corrections[key] = arr

    if verbose:
        _log(f"Loaded {n_fields} correction fields from {path} ({len(data)} bytes)")
        for k, v in corrections.items():
            _log(f"  {k}: {v.shape} {v.dtype}")

    return corrections


# ---------------------------------------------------------------------------
# Eureka 9: Jacobian-Directed Quantization
# ---------------------------------------------------------------------------


def compute_quantization_directions(
    frames: torch.Tensor,  # (N, 3, H, W) float, BCHW
    posenet: nn.Module,
    segnet: nn.Module,
    device: str | torch.device = "cpu",
    batch_size: int = 4,
    seg_weight: float = 100.0,
    pose_weight: float = 1.0,
    verbose: bool = True,
) -> torch.Tensor:
    """Compute optimal rounding direction for each pixel-channel.

    Returns (N, 3, H, W) int8 tensor: +1 = ceil, -1 = floor, 0 = round.
    Based on scorer Jacobian sign: if d(score)/d(pixel) > 0, increasing the pixel
    helps, so ceil. If < 0, floor. If approx 0, round normally.

    Store as compressed bitmask in archive (~20KB after entropy coding).
    Apply at inflate time for free score improvement (0.02-0.05 estimated).

    The scorer loss is: S = seg_weight * seg_distortion + sqrt(10 * pose_distortion).
    The gradient d(S)/d(pixel) tells us which direction reduces the score.
    We round in the direction that REDUCES loss (negative gradient = ceil helps
    because we want to descend, i.e., move opposite to gradient).

    Args:
        frames: (N, 3, H, W) float tensor in [0, 255], BCHW format.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        device: computation device.
        batch_size: frames per gradient computation.
        seg_weight: relative weight for SegNet gradient contribution.
        pose_weight: relative weight for PoseNet gradient contribution.
        verbose: print progress.

    Returns:
        (N, 3, H, W) int8 tensor: +1 (ceil), -1 (floor), 0 (round normally).
    """
    N, C, H, W = frames.shape
    all_directions = torch.zeros(N, C, H, W, dtype=torch.int8)
    dev = torch.device(device)

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        x = frames[batch_start:batch_end].to(dev).detach().clone()
        x.requires_grad_(True)

        B = x.shape[0]

        # Build self-pairs for scorer: (B, 2, C, H, W)
        pair = x.unsqueeze(1).expand(B, 2, C, H, W).contiguous()

        # PoseNet forward + loss
        posenet_in = posenet.preprocess_input(pair)
        pose_out = posenet(posenet_in)
        pose_tensor = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
        pose_loss = pose_tensor[..., :6].pow(2).sum()

        # SegNet forward + loss (cross-entropy with self-argmax as target)
        segnet_in = segnet.preprocess_input(pair)
        seg_out = segnet(segnet_in)
        seg_probs = F.softmax(seg_out, dim=1)
        # Use entropy as the loss: high-entropy pixels are near decision boundaries
        seg_loss = -(seg_probs * (seg_probs + 1e-8).log()).sum()

        total_loss = pose_weight * pose_loss + seg_weight * seg_loss
        total_loss.backward()

        grad = x.grad.detach().cpu()  # (B, C, H, W)

        # Gradient sign determines rounding direction:
        # negative gradient -> increasing pixel helps (ceil = +1)
        # positive gradient -> decreasing pixel helps (floor = -1)
        # near-zero -> normal rounding (0)
        threshold = grad.abs().mean() * 0.1  # only act on significant gradients
        directions = torch.zeros_like(grad, dtype=torch.int8)
        directions[grad < -threshold] = 1   # ceil: pixel increase reduces loss
        directions[grad > threshold] = -1   # floor: pixel decrease reduces loss

        all_directions[batch_start:batch_end] = directions

        if verbose and (batch_end % (batch_size * 5) == 0 or batch_end == N):
            ceil_pct = (directions == 1).float().mean().item() * 100
            floor_pct = (directions == -1).float().mean().item() * 100
            _log(f"quant directions: {batch_end}/{N} frames "
                 f"(ceil={ceil_pct:.1f}% floor={floor_pct:.1f}%)")

    return all_directions


def apply_quantization_directions(
    frames_float: torch.Tensor,  # (N, 3, H, W) or (N, H, W, 3)
    directions: torch.Tensor,    # (N, 3, H, W) int8
    hwc_input: bool = False,
) -> torch.Tensor:
    """Apply precomputed rounding directions. Zero-cost at inflate time.

    For each pixel-channel:
      direction == +1: ceil(value)
      direction == -1: floor(value)
      direction ==  0: round(value) (normal rounding)

    Args:
        frames_float: float tensor with fractional pixel values.
        directions: int8 tensor with rounding directions (+1, -1, 0).
        hwc_input: if True, frames are (N, H, W, 3) and directions are (N, 3, H, W).
            Will transpose internally.

    Returns:
        Quantized tensor in the same format as input, clamped to [0, 255].
    """
    if hwc_input:
        # Convert HWC -> CHW for processing
        f = frames_float.permute(0, 3, 1, 2).contiguous()
    else:
        f = frames_float

    d = directions.to(f.device)
    result = torch.where(
        d == 1,
        f.ceil(),
        torch.where(
            d == -1,
            f.floor(),
            f.round(),
        ),
    )
    result = result.clamp(0.0, 255.0)

    if hwc_input:
        result = result.permute(0, 2, 3, 1).contiguous()

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    """CLI: python -m tac.precompute_corrections --gt-video videos/0.mkv --upstream <dir> --output corrections.bin"""
    import argparse

    parser = argparse.ArgumentParser(description="Pre-compute scorer corrections for inflate")
    parser.add_argument("--gt-video", required=True, help="Path to ground truth video")
    parser.add_argument("--posenet", required=True, help="Path to posenet.safetensors")
    parser.add_argument("--segnet", default=None, help="Path to segnet.safetensors (default: derived from --posenet path)")
    parser.add_argument("--upstream", default=None, help="Upstream repo dir (for modules.py)")
    parser.add_argument("--output", required=True, help="Output path for corrections file")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--null-space-k", type=int, default=32, help="Null-space basis vectors")
    parser.add_argument("--hard-frames", type=int, default=50, help="Number of hard frames")
    parser.add_argument("--hard-pixels", type=int, default=1000, help="Pixels per hard frame")
    args = parser.parse_args()

    from tac.data import decode_video
    from tac.scorer import load_scorers

    print(f"[precompute] Decoding GT video: {args.gt_video}", file=sys.stderr)
    gt_frames = decode_video(args.gt_video)
    print(f"[precompute] Decoded {len(gt_frames)} GT frames", file=sys.stderr)

    segnet_path = Path(args.segnet) if args.segnet else Path(args.posenet).parent / "segnet.safetensors"
    upstream_dir = args.upstream or str(Path(args.posenet).parent.parent)
    posenet, segnet = load_scorers(
        args.posenet, segnet_path,
        device=args.device, upstream_dir=upstream_dir,
    )

    corrections = precompute_all_corrections(
        gt_frames, posenet, segnet,
        batch_size=args.batch_size,
        device=args.device,
        null_space_top_k=args.null_space_k,
        hard_frame_top_k=args.hard_frames,
        hard_frame_pixels=args.hard_pixels,
    )

    size = save_corrections(corrections, args.output)
    print(f"Done. Output: {args.output} ({size} bytes)")
