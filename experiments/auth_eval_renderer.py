#!/usr/bin/env python
"""Authoritative evaluation for renderer checkpoints.

Standalone script that runs the full inflate -> score pipeline for any
renderer checkpoint (.pt training checkpoint or .bin ASYM/DPSM export).

Works on ANY platform with CUDA + upstream repo. This is the reusable
core that both Modal and Lightning auth eval scripts invoke.

Pipeline:
    1. Load checkpoint (detect format + pair_mode from config)
    2. Load upstream SegNet for mask extraction
    3. Decode GT video (upstream/videos/0.mkv via PyAV)
    4. Extract masks via SegNet
    5. Generate frames via renderer (pair-wise for asymmetric)
    6. Upscale to 874x1164, write as .raw
    7. Score via upstream DistortionNet (PoseNet + SegNet)
    8. Compute rate from archive size
    9. Print score breakdown + save results JSON

Usage:
    PYTHONPATH=src:upstream python experiments/auth_eval_renderer.py \\
        --checkpoint experiments/results/fridrich_renderer/renderer_best.pt \\
        --upstream-dir upstream \\
        --device cuda

    PYTHONPATH=src:upstream python experiments/auth_eval_renderer.py \\
        --checkpoint submissions/robust_current/renderer.bin \\
        --upstream-dir /home/zeus/content/upstream \\
        --device cuda --batch-size 16

    # Specify explicit archive size for rate calculation:
    PYTHONPATH=src:upstream python experiments/auth_eval_renderer.py \\
        --checkpoint renderer_best.pt \\
        --upstream-dir upstream \\
        --archive-size-bytes 180000 \\
        --device cuda
"""
from __future__ import annotations

# DX-fix 2026-04-25: line-buffer stdout/stderr so progress logs flush
# immediately when piped to log files (Python buffers ~8KB by default,
# making long-running scripts appear silent for hours per the optimize_poses
# incident on the A100 today).
import sys as _dx_sys
try:
    _dx_sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    _dx_sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass


import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Constants
# ============================================================
OUT_W, OUT_H = 1164, 874
SEG_W, SEG_H = 512, 384
NUM_FRAMES = 1200
EXPECTED_RAW_BYTES = OUT_W * OUT_H * 3 * NUM_FRAMES


# ============================================================
# Canonical YUV->RGB (BT.601 limited range, matches frame_utils.py)
# ============================================================
def _yuv420_to_rgb(frame) -> torch.Tensor:
    H, W = frame.height, frame.width
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(H, frame.planes[0].line_size)[:, :W]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(H // 2, frame.planes[1].line_size)[:, :W // 2]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(H // 2, frame.planes[2].line_size)[:, :W // 2]

    y_t = torch.from_numpy(y.copy()).float()
    u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)

    u_up = F.interpolate(u_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()
    v_up = F.interpolate(v_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()

    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)

    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)


# ============================================================
# Upstream path setup
# ============================================================
def _ensure_upstream(upstream_dir: str) -> Path:
    """Validate upstream dir and add to sys.path."""
    upstream = Path(upstream_dir).resolve()
    if not (upstream / "modules.py").exists():
        raise FileNotFoundError(
            f"Cannot find modules.py in upstream dir: {upstream}\n"
            f"Set --upstream-dir to the comma_video_compression_challenge repo root."
        )
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    return upstream


# ============================================================
# Model loading — delegates to inflate_renderer.py patterns
# ============================================================
def _load_renderer_checkpoint(ckpt_path: str, device: str) -> tuple[nn.Module, dict, int]:
    """Load a renderer from .pt or .bin checkpoint.

    Returns:
        (model, config_dict, archive_size_bytes)

    archive_size_bytes is the on-disk size used for rate calculation.
    For .bin files this is the file size. For .pt files we try to find
    a companion .bin or fall back to exporting one.
    """
    ckpt_path = Path(ckpt_path).resolve()
    raw_bytes = ckpt_path.read_bytes()
    magic = raw_bytes[:4]
    file_size = len(raw_bytes)

    # ── Binary formats (ASYM, DPSM, FP4A, I4LZ) ──
    # Use the canonical loader which handles ALL formats
    # R38 fix: MXLZ added to dispatch. Mixed-precision LZMA2 was missing
    # → fell through to .pt path → cryptic torch.load error on a binary file.
    if magic in (b"ASYM", b"DPSM", b"FP4A", b"I4LZ", b"MXLZ"):
        try:
            from tac.renderer_export import load_any_renderer_checkpoint
            model = load_any_renderer_checkpoint(str(ckpt_path), device=device)
        except ImportError:
            # Fall back to inflate_renderer.py inline loaders
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submissions" / "robust_current"))
            from inflate_renderer import _load_renderer
            model = _load_renderer(str(ckpt_path), device)

        # Extract config from header (if present)
        config = {}
        if magic in (b"ASYM", b"DPSM", b"FP4A", b"MXLZ"):
            header_len = struct.unpack("<I", raw_bytes[4:8])[0]
            config = json.loads(raw_bytes[8:8 + header_len].decode("utf-8"))
        elif magic == b"I4LZ" and hasattr(model, 'arch_dict'):
            config = model.arch_dict()

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        fmt_name = magic.decode("ascii")
        print(f"  Loaded {fmt_name} binary: {file_size:,} bytes")
        return model, config, file_size

    # ── .pt training checkpoint ──
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            f"Unsupported .pt format. Expected dict with 'model_state_dict'. "
            f"Got: {type(ckpt)} with keys {list(ckpt.keys()) if isinstance(ckpt, dict) else 'N/A'}"
        )

    config = ckpt.get("config", {})
    epoch = ckpt.get("epoch", "?")
    best_score = ckpt.get("best_score")
    print(f"  Training checkpoint: epoch={epoch}" +
          (f", proxy best_score={best_score:.4f}" if best_score else ""))

    # Reconstruct model from config
    from tac.renderer import AsymmetricPairGenerator, MaskRenderer
    pair_mode = config.get("pair_mode", "asymmetric")

    if pair_mode == "asymmetric" or any(
        k.startswith("motion.") for k in ckpt["model_state_dict"]
    ):
        model = AsymmetricPairGenerator(
            num_classes=config.get("num_classes", 5),
            embed_dim=config.get("embed_dim", 6),
            base_ch=config.get("base_ch", 36),
            mid_ch=config.get("mid_ch", 60),
            motion_hidden=config.get("motion_hidden", 32),
            depth=config.get("renderer_depth", config.get("depth", 1)),
            max_flow_px=config.get("max_flow_px", 20.0),
            max_residual=config.get("max_residual", 20.0),
            flow_only=config.get("flow_only", False),
            pose_dim=config.get("pose_dim", 0),
            use_dsconv=config.get("use_dsconv", False),
            padding_mode=config.get("padding_mode", "zeros"),
            use_dilation=config.get("use_dilation", False),
            use_zoom_flow=config.get("use_zoom_flow", False),
        )
    else:
        model = MaskRenderer(
            num_classes=config.get("num_classes", 5),
            embed_dim=config.get("embed_dim", 6),
            base_ch=config.get("base_ch", 36),
            mid_ch=config.get("mid_ch", 60),
            depth=config.get("renderer_depth", config.get("depth", 1)),
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # Determine archive size for rate calculation
    # Try companion .bin first, then export
    ckpt_dir = ckpt_path.parent
    bin_candidates = [
        ckpt_dir / "renderer.bin",
        ckpt_dir / "renderer_best.bin",
        ckpt_dir / (ckpt_path.stem.replace(".pt", "") + ".bin"),
    ]
    archive_size = None
    for bc in bin_candidates:
        if bc.exists():
            archive_size = bc.stat().st_size
            print(f"  Rate from companion .bin: {bc.name} ({archive_size:,} bytes)")
            break

    if archive_size is None:
        try:
            from tac.renderer_export import export_asymmetric_checkpoint
            export_bytes = export_asymmetric_checkpoint(model, bits=4)
            archive_size = len(export_bytes)
            print(f"  Exported for rate: {archive_size:,} bytes (4-bit quantized)")
            del export_bytes
        except Exception as e:
            raise RuntimeError(
                f"Cannot determine accurate archive size for rate calculation. "
                f"No companion .bin found and export failed: {e}. "
                f"Using .pt file size would give 5-10x wrong rate. "
                f"Export a .bin first or pass --archive-size-bytes explicitly."
            )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {type(model).__name__}, {n_params:,} params")
    del ckpt
    return model, config, archive_size


def _is_asymmetric(model: nn.Module) -> bool:
    """Detect whether a loaded model is an AsymmetricPairGenerator."""
    return (
        type(model).__name__ == "AsymmetricPairGenerator"
        or (hasattr(model, "renderer") and hasattr(model, "motion"))
    )


# ============================================================
# Core pipeline stages
# ============================================================
def decode_gt_video(mkv_path: str) -> list[np.ndarray]:
    """Decode GT video via PyAV -> list of (H, W, 3) uint8 ndarrays."""
    import av

    t0 = time.monotonic()
    container = av.open(mkv_path)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        rgb = _yuv420_to_rgb(frame)
        frames.append(rgb.numpy())
    container.close()
    elapsed = time.monotonic() - t0
    print(f"  Decoded {len(frames)} GT frames ({elapsed:.1f}s)")
    return frames


def extract_masks(
    frames: list[np.ndarray],
    segnet: nn.Module,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """Extract SegNet masks from GT frames -> (N, 384, 512) int8 tensor."""
    t0 = time.monotonic()
    masks_list = []
    N = len(frames)

    with torch.inference_mode():
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch_np = np.stack(frames[i:end], axis=0)
            batch_t = torch.from_numpy(batch_np).float().permute(0, 3, 1, 2).to(device)
            inp = batch_t.unsqueeze(1)  # (B, 1, 3, H, W) for preprocess_input
            seg_in = segnet.preprocess_input(inp)
            logits = segnet(seg_in)
            mask = logits.argmax(dim=1)
            masks_list.append(mask.to(torch.int8).cpu())

            if end % (batch_size * 10) == 0 or end == N:
                print(f"    Masks: {end}/{N} frames", flush=True)

    masks = torch.cat(masks_list, dim=0)
    elapsed = time.monotonic() - t0
    print(f"  Extracted {masks.shape[0]} masks ({elapsed:.1f}s)")
    return masks


def generate_raw(
    masks: torch.Tensor,
    model: nn.Module,
    raw_path: str,
    device: str,
    batch_size: int,
    poses: torch.Tensor | None = None,
    zoom_warp: nn.Module | None = None,
) -> int:
    """Generate frames from masks, upscale to 874x1164, write .raw.

    Args:
        masks: (N, H, W) segmentation masks
        model: renderer model (AsymmetricPairGenerator or DPSIMSRenderer)
        raw_path: output path for raw frames
        device: torch device
        batch_size: number of pairs per batch
        poses: (N//2, 6) optional FiLM conditioning vectors
        zoom_warp: RadialZoomWarp for ego_flow (use_zoom_flow models)

    Returns number of frames written.
    """
    t0 = time.monotonic()
    N = masks.shape[0]
    n_written = 0
    is_asym = _is_asymmetric(model)
    has_film = hasattr(model, 'pose_dim') and model.pose_dim > 0
    has_zoom = hasattr(model, 'use_zoom_flow') and model.use_zoom_flow
    torch.manual_seed(42)

    with open(raw_path, "wb") as f:
        with torch.inference_mode():
            if is_asym:
                print(f"  Mode: asymmetric pair generation ({N} masks)")
                if has_film and poses is not None:
                    print(f"  FiLM conditioning: {poses.shape[0]} pose vectors")
                pair_idx = 0
                while pair_idx < N - 1:
                    batch_t_list = []
                    batch_t1_list = []
                    batch_end = min(pair_idx + batch_size * 2, N - 1)
                    for j in range(pair_idx, batch_end, 2):
                        if j + 1 < N:
                            batch_t_list.append(masks[j])
                            batch_t1_list.append(masks[j + 1])
                    if not batch_t_list:
                        break

                    masks_t = torch.stack(batch_t_list).to(device=device, dtype=torch.long)
                    masks_t1 = torch.stack(batch_t1_list).to(device=device, dtype=torch.long)

                    # FiLM pose conditioning
                    batch_pose = None
                    if has_film and poses is not None:
                        pose_start = pair_idx // 2
                        pose_end = pose_start + masks_t.shape[0]
                        if pose_end <= poses.shape[0]:
                            batch_pose = poses[pose_start:pose_end].to(device=device)
                        else:
                            # Clamp to available poses, pad remainder with zeros
                            avail = max(0, poses.shape[0] - pose_start)
                            if avail > 0:
                                batch_pose = torch.zeros(masks_t.shape[0], poses.shape[1], device=device)
                                batch_pose[:avail] = poses[pose_start:pose_start + avail].to(device=device)
                            print(f"    WARNING: pose index {pose_start}:{pose_end} exceeds "
                                  f"poses.shape[0]={poses.shape[0]}", file=sys.stderr)

                    # Zoom flow (ego_flow from RadialZoomWarp)
                    ego_flow = None
                    if has_zoom and zoom_warp is not None:
                        pair_indices = torch.arange(
                            pair_idx // 2,
                            pair_idx // 2 + masks_t.shape[0],
                            device=device,
                        )
                        ego_flow = zoom_warp(pair_indices, masks_t.shape[1], masks_t.shape[2])

                    kwargs = {}
                    if batch_pose is not None:
                        kwargs["pose"] = batch_pose
                    if ego_flow is not None:
                        kwargs["ego_flow"] = ego_flow
                    pairs = model(masks_t, masks_t1, **kwargs)  # (B, 2, H, W, 3)

                    B_pairs = pairs.shape[0]
                    for p_idx in range(B_pairs):
                        for frame_idx in range(2):
                            frame_hwc = pairs[p_idx, frame_idx]
                            frame_chw = frame_hwc.permute(2, 0, 1).unsqueeze(0)
                            frame_up = F.interpolate(
                                frame_chw, size=(OUT_H, OUT_W),
                                mode="bilinear", align_corners=False,
                            )
                            frame_uint8 = frame_up.round().clamp(0, 255).to(torch.uint8)
                            frame_out = frame_uint8.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
                            f.write(frame_out.tobytes())
                            n_written += 1

                    pair_idx += len(batch_t_list) * 2
                    if n_written % 200 == 0 or pair_idx >= N - 1:
                        print(f"    Generated: {n_written}/{N} frames", flush=True)

                # Handle odd trailing mask
                if N % 2 != 0:
                    last_mask = masks[N - 1:N].to(device=device, dtype=torch.long)
                    frame = model.renderer(last_mask)
                    frame_up = F.interpolate(frame, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)
                    frame_uint8 = frame_up.round().clamp(0, 255).to(torch.uint8)
                    frame_out = frame_uint8.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
                    f.write(frame_out.tobytes())
                    n_written += 1
            else:
                print(f"  Mode: independent frame generation ({N} masks)")
                for i in range(0, N, batch_size):
                    end = min(i + batch_size, N)
                    batch_masks = masks[i:end].to(device=device, dtype=torch.long)
                    frames = model(batch_masks)
                    frames_up = F.interpolate(frames, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)
                    frames_uint8 = frames_up.round().clamp(0, 255).to(torch.uint8)
                    frames_hwc = frames_uint8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                    f.write(frames_hwc.tobytes())
                    n_written += batch_masks.shape[0]
                    if end % 200 == 0 or end == N:
                        print(f"    Generated: {end}/{N} frames", flush=True)

    raw_size = os.path.getsize(raw_path)
    expected_size = OUT_W * OUT_H * 3 * n_written
    assert raw_size == expected_size, f"Raw size mismatch: {raw_size} vs {expected_size}"
    elapsed = time.monotonic() - t0
    print(f"  Generated {n_written} frames ({elapsed:.1f}s)")
    return n_written


def score_with_upstream(
    raw_path: str,
    upstream_dir: Path,
    device: str,
    batch_size: int,
) -> tuple[float, float]:
    """Score generated .raw against GT using upstream DistortionNet.

    Returns (avg_posenet_dist, avg_segnet_dist).
    """
    from modules import DistortionNet
    from modules import segnet_sd_path, posenet_sd_path
    from frame_utils import TensorVideoDataset, AVVideoDataset

    t0 = time.monotonic()

    distortion_net = DistortionNet().eval().to(device)
    distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    video_names = ["0.mkv"]

    # GT dataset — AVVideoDataset (PyAV decode, non-cuda device required)
    ds_gt = AVVideoDataset(
        video_names,
        data_dir=upstream_dir / "videos",
        batch_size=batch_size,
        device=torch.device("cpu"),
    )
    ds_gt.prepare_data()
    dl_gt = torch.utils.data.DataLoader(ds_gt, batch_size=None, num_workers=0)

    # Compressed dataset — TensorVideoDataset (reads .raw)
    # Caller must ensure raw_path is named 0.raw (run_auth_eval does this)
    raw_dir = Path(raw_path).parent

    ds_comp = TensorVideoDataset(
        video_names,
        data_dir=raw_dir,
        batch_size=batch_size,
        device=torch.device("cpu"),
    )
    ds_comp.prepare_data()
    dl_comp = torch.utils.data.DataLoader(ds_comp, batch_size=None, num_workers=0)

    posenet_dists = torch.zeros([], device=device)
    segnet_dists = torch.zeros([], device=device)
    batch_count = torch.zeros([], device=device)

    with torch.inference_mode():
        for (_, _, batch_gt), (_, _, batch_comp) in zip(dl_gt, dl_comp):
            batch_gt = batch_gt.to(device)
            batch_comp = batch_comp.to(device)
            posenet_dist, segnet_dist = distortion_net.compute_distortion(batch_gt, batch_comp)
            posenet_dists += posenet_dist.sum()
            segnet_dists += segnet_dist.sum()
            batch_count += batch_gt.shape[0]

    avg_posenet = (posenet_dists / batch_count).item()
    avg_segnet = (segnet_dists / batch_count).item()
    n_samples = int(batch_count.item())

    elapsed = time.monotonic() - t0
    print(f"  Scored {n_samples} samples ({elapsed:.1f}s)")

    # Clean up the .raw file (3.6 GB)
    if os.path.exists(raw_path):
        os.remove(raw_path)
        print(f"  Cleaned up: {raw_path}")

    return avg_posenet, avg_segnet


# ============================================================
# Main entry point
# ============================================================
def run_auth_eval(
    checkpoint: str,
    upstream_dir: str,
    device: str = "cuda",
    batch_size: int | None = None,
    archive_size_override: int | None = None,
    output_dir: str | None = None,
    poses_path: str | None = None,
) -> dict:
    """Run the full authoritative evaluation pipeline.

    Args:
        checkpoint: path to .pt or .bin checkpoint
        upstream_dir: path to upstream repo root (has modules.py, videos/, models/)
        device: torch device
        batch_size: inference batch size (auto-detected if None)
        archive_size_override: explicit archive size for rate calculation
        output_dir: directory for output files (default: alongside checkpoint)
        poses_path: path to optimized poses (.pt or .bin) for FiLM conditioning

    Returns:
        dict with all score components
    """
    t_start = time.monotonic()

    if batch_size is None:
        batch_size = 16 if device == "cuda" else 4

    upstream = _ensure_upstream(upstream_dir)

    print(f"\n{'=' * 60}")
    print(f"  Authoritative Renderer Evaluation")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"{'=' * 60}\n")

    # Stage 1: Load renderer
    print("Stage 1: Loading renderer ...")
    model, config, archive_size = _load_renderer_checkpoint(checkpoint, device)
    if archive_size_override is not None:
        archive_size = archive_size_override
        print(f"  Rate override: {archive_size:,} bytes")

    # Stage 2: Load SegNet
    print("\nStage 2: Loading SegNet ...")
    from modules import SegNet
    from safetensors.torch import load_file

    segnet = SegNet()
    segnet_path = upstream / "models" / "segnet.safetensors"
    sd = load_file(segnet_path, device=device)
    segnet.load_state_dict(sd)
    segnet.to(device).eval()
    for p in segnet.parameters():
        p.requires_grad = False
    print(f"  SegNet loaded from {segnet_path}")

    # Stage 3: Decode GT video
    print("\nStage 3: Decoding GT video ...")
    gt_video_path = str(upstream / "videos" / "0.mkv")
    gt_frames = decode_gt_video(gt_video_path)
    assert len(gt_frames) == NUM_FRAMES, f"Expected {NUM_FRAMES} frames, got {len(gt_frames)}"

    # Stage 4: Extract masks
    print("\nStage 4: Extracting SegNet masks ...")
    masks = extract_masks(gt_frames, segnet, device, batch_size)
    del gt_frames, segnet  # free memory

    # Stage 4b: Load optimized poses (for FiLM conditioning)
    poses = None
    if poses_path is not None:
        poses_p = Path(poses_path)
        if poses_p.suffix == ".bin":
            pose_dim = getattr(model, 'pose_dim', 6)
            raw = poses_p.read_bytes()
            poses = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(-1, pose_dim).float()
        else:
            poses = torch.load(str(poses_p), map_location="cpu", weights_only=True).float()
        print(f"  Loaded poses: {poses.shape} from {poses_p.name}")
    elif hasattr(model, 'pose_dim') and model.pose_dim > 0:
        print("  WARNING: Model has FiLM (pose_dim>0) but no --poses provided. "
              "Using zero poses — PoseNet score will be degraded.")

    # Stage 4c: Load zoom warp (for use_zoom_flow models)
    zoom_warp = None
    if hasattr(model, 'use_zoom_flow') and model.use_zoom_flow:
        from tac.radial_zoom import RadialZoomWarp
        n_pairs = masks.shape[0] // 2
        zoom_warp = RadialZoomWarp(n_pairs=n_pairs).to(device)
        # Try loading saved zoom scalars from checkpoint
        ckpt_path = Path(checkpoint)
        if ckpt_path.suffix == ".pt":
            ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            if isinstance(ckpt_data, dict) and "zoom_warp_state_dict" in ckpt_data:
                zoom_warp.load_state_dict(ckpt_data["zoom_warp_state_dict"])
                print(f"  Loaded zoom scalars from checkpoint")
            else:
                print(f"  WARNING: use_zoom_flow model but no zoom_warp_state_dict in checkpoint. "
                      f"Using identity zoom — PoseNet may be degraded.")
        else:
            # For .bin format, check for companion zoom_scalars.bin
            zoom_path = ckpt_path.parent / "zoom_scalars.bin"
            if zoom_path.exists():
                from tac.radial_zoom import load_zoom_scalars
                zoom_warp = load_zoom_scalars(zoom_path, device=device)
                print(f"  Loaded zoom scalars from {zoom_path.name}")
            else:
                print(f"  WARNING: use_zoom_flow but no zoom_scalars.bin found. Using identity zoom.")

    # Stage 5: Generate frames
    print("\nStage 5: Generating frames ...")
    if output_dir is None:
        output_dir = str(Path(checkpoint).resolve().parent)
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "auth_eval_inflated.raw")
    n_written = generate_raw(masks, model, raw_path, device, batch_size, poses=poses, zoom_warp=zoom_warp)
    del model, masks, poses  # free VRAM

    # Rename raw for scorer (expects 0.raw)
    expected_raw = os.path.join(output_dir, "0.raw")
    if raw_path != expected_raw:
        if os.path.exists(expected_raw):
            os.remove(expected_raw)
        os.rename(raw_path, expected_raw)

    # Stage 6: Score
    print("\nStage 6: Scoring via upstream DistortionNet ...")
    avg_posenet, avg_segnet = score_with_upstream(
        expected_raw, upstream, device, batch_size,
    )

    # Stage 7: Compute rate and final score
    gt_size = os.path.getsize(gt_video_path)
    rate = archive_size / gt_size

    score_seg = 100 * avg_segnet
    score_pose = math.sqrt(10 * avg_posenet)
    score_rate = 25 * rate
    final_score = score_seg + score_pose + score_rate

    t_total = time.monotonic() - t_start

    print(f"\n{'=' * 60}")
    print(f"  AUTHORITATIVE RESULTS")
    print(f"{'=' * 60}")
    print(f"  PoseNet dist:     {avg_posenet:.8f}")
    print(f"  SegNet dist:      {avg_segnet:.8f}")
    print(f"  Archive size:     {archive_size:,} bytes")
    print(f"  GT size:          {gt_size:,} bytes")
    print(f"  Rate:             {rate:.8f}")
    print(f"")
    print(f"  Score breakdown:")
    print(f"    100*seg       = {score_seg:.4f}")
    print(f"    sqrt(10*pose) = {score_pose:.4f}")
    print(f"    25*rate       = {score_rate:.4f}")
    print(f"  ──────────────────")
    print(f"  FINAL SCORE:      {final_score:.4f}")
    print(f"  Total time:       {t_total:.1f}s")
    print(f"{'=' * 60}")

    # Schema-first sentinel line for downstream tooling. Parsing the human-
    # readable output above with regex is fragile (multiple R28+R29 review
    # rounds caught regex bugs); the sentinel + JSON payload is the canonical
    # contract. Parser: re.search(r"^RESULT_JSON: (.+)$", line, re.M) +
    # AuthEvalResult.model_validate_json(payload).
    import json as _json
    _result_payload = {
        "schema_version": 1,
        "final_score": float(final_score),
        "score_seg": float(score_seg),
        "score_pose": float(score_pose),
        "score_rate": float(score_rate),
        "avg_segnet_dist": float(avg_segnet),
        "avg_posenet_dist": float(avg_posenet),
        "rate": float(rate),
        "archive_size_bytes": int(archive_size),
        "gt_size_bytes": int(gt_size),
    }
    print(f"RESULT_JSON: {_json.dumps(_result_payload, separators=(',', ':'))}")

    # Save results JSON
    result = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "avg_posenet_dist": avg_posenet,
        "avg_segnet_dist": avg_segnet,
        "archive_size_bytes": archive_size,
        "gt_size_bytes": gt_size,
        "rate": rate,
        "score_seg": score_seg,
        "score_pose": score_pose,
        "score_rate": score_rate,
        "final_score": final_score,
        "n_frames": n_written,
        "device": device,
        "config": config,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(t_total, 1),
    }

    ckpt_stem = Path(checkpoint).stem
    result_path = os.path.join(output_dir, f"auth_eval_{ckpt_stem}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {result_path}")

    return result


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Authoritative evaluation for renderer checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to .pt training checkpoint or .bin export (ASYM/DPSM)",
    )
    parser.add_argument(
        "--upstream-dir", required=True,
        help="Path to upstream repo root (has modules.py, videos/, models/)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Torch device (default: cuda)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Inference batch size (default: 16 for cuda, 4 for cpu)",
    )
    parser.add_argument(
        "--archive-size-bytes", type=int, default=None,
        help="Override archive size for rate calculation (bytes)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for output files (default: alongside checkpoint)",
    )
    parser.add_argument(
        "--poses", default=None,
        help="Path to optimized poses (.pt or .bin) for FiLM conditioning",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    result = run_auth_eval(
        checkpoint=args.checkpoint,
        upstream_dir=args.upstream_dir,
        device=args.device,
        batch_size=args.batch_size,
        archive_size_override=args.archive_size_bytes,
        output_dir=args.output_dir,
        poses_path=args.poses,
    )

    # Exit with non-zero if score is suspiciously high (sanity check)
    if result["final_score"] > 50.0:
        print(f"\nWARNING: Score {result['final_score']:.4f} is very high — possible pipeline error")
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
