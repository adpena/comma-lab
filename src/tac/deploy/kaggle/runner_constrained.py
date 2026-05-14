# SPDX-License-Identifier: MIT
"""Constrained generation runner — provider-agnostic.

Works on Kaggle, Modal, Lightning, or locally. Platform is auto-detected via
tac.deploy.cloud_paths.CloudPaths.detect(); override with CLOUD_PLATFORM,
CLOUD_INPUT_ROOT, CLOUD_WORKING_DIR env vars.

Full inflate pipeline (no model training):
  GT video → SegNet masks → PoseNet pose targets → constrained_generate() → inflated/

Archive size: 64 bytes (noise seed only). Masks + targets extracted at inflate time
from the GT video using upstream SegNet/PoseNet.

Scorer alignment:
  - Decode: av + frame_utils.yuv420_to_rgb — matches NVDEC output pixel-for-pixel.
  - PoseNet pairs: non-overlapping (f_{2k}, f_{2k+1}), seq_len=2, matching scorer.
  - Frame resolution: generate at SegNet input size (384×512, ~884 MB), upsample to
    camera size (874×1164) for frames.raw — scorer resizes internally anyway.
  - Frame count: truncated to (N//2)*2 matching DALI/AVVideoDataset pair semantics.
  - Output: working_dir/inflated/{video_name}.raw + working_dir/archive.zip

Memory on T4: frames at (1200, 384, 512, 3) float32 ≈ 884 MB — fits with 15 GB headroom.

Pre-registered hypothesis: proxy < 0.80 after 1000 steps on T4.
Kill: proxy > 1.50 after 500 steps.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # R41 fix: torch only imported lazily inside functions to keep the script
    # importable on cold Kaggle workers without torch in path. TYPE_CHECKING
    # gives the type checker (ty/ruff) what it needs without runtime cost.
    import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASSET_DATASET_SLUG: str = "comma-lab-private-assets"
UPSTREAM_REPO: str = "https://github.com/commaai/comma_video_compression_challenge.git"

PIP_DEPS: tuple[str, ...] = (
    "av", "safetensors", "timm", "einops", "segmentation-models-pytorch",
    "pydantic", "click",
)

NUM_STEPS: int = 1000
LR: float = 0.1
SEG_WEIGHT: float = 50.0
POSE_WEIGHT: float = 50.0
NOISE_SEED: int = 42
LOG_EVERY: int = 100
EXTRACT_BATCH: int = 16  # batch size for mask + pose extraction

# Pre-registered kill criterion: proxy > KILL_THRESHOLD after KILL_STEP steps.
# Per experiment protocol (CLAUDE.md): kill experiments showing no signal early.
KILL_STEP: int = 500
KILL_THRESHOLD: float = 1.50

# Upstream scorer camera + segnet dimensions (from frame_utils.py)
CAMERA_H: int = 874
CAMERA_W: int = 1164
SEGNET_H: int = 384   # segnet_model_input_size[1]
SEGNET_W: int = 512   # segnet_model_input_size[0]


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _add_upstream_to_path(upstream_root: Path) -> None:
    """Add upstream repo root to sys.path so frame_utils + modules are importable."""
    upstream_str = str(upstream_root)
    if upstream_str not in sys.path:
        sys.path.insert(0, upstream_str)


def ensure_upstream(working_dir: Path) -> Path:
    """Clone upstream repo + git-lfs assets into working_dir/upstream."""
    upstream = working_dir / "upstream"
    if not (upstream / "models").exists():
        subprocess.check_call(
            ["git", "clone", "--depth", "1", UPSTREAM_REPO, str(upstream)]
        )
        for attempt in range(1, 4):
            try:
                subprocess.check_call(["git", "lfs", "pull"], cwd=upstream)
                break
            except subprocess.CalledProcessError:
                if attempt == 3:
                    raise RuntimeError("git lfs pull failed after 3 attempts.")
                print(f"  git lfs pull attempt {attempt} failed — retrying...")
                time.sleep(5 * attempt)
    return upstream


# ---------------------------------------------------------------------------
# Core: decode GT video + extract masks/targets from upstream models
# ---------------------------------------------------------------------------

def _decode_frames(video_path: Path) -> torch.Tensor:
    """Decode all frames via av + yuv420_to_rgb.

    frame_utils.yuv420_to_rgb uses bilinear chroma upsampling + BT.601 limited-range
    conversion, matching NVDEC (DALI) output pixel-for-pixel.

    Returns:
        (N, H, W, 3) uint8 torch tensor.
    """
    import av
    import torch
    from frame_utils import yuv420_to_rgb  # matches NVDEC output

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        frames_list: list[torch.Tensor] = []
        for frame in container.decode(stream):
            frames_list.append(yuv420_to_rgb(frame))  # (H, W, 3) uint8
    if not frames_list:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return torch.stack(frames_list, dim=0)  # (N, H, W, 3) uint8


def _load_models(
    upstream_root: Path, device: str
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load SegNet + PoseNet with weights from upstream safetensors files.

    Returns:
        (segnet, posenet) both on device, in eval mode with weights loaded.
    """
    from modules import SegNet, PoseNet
    from safetensors.torch import load_file

    segnet = SegNet().to(device).eval()
    segnet.load_state_dict(
        load_file(str(upstream_root / "models" / "segnet.safetensors"), device=str(device))
    )
    posenet = PoseNet().to(device).eval()
    posenet.load_state_dict(
        load_file(str(upstream_root / "models" / "posenet.safetensors"), device=str(device))
    )
    return segnet, posenet


def extract_masks_and_targets(
    video_path: Path,
    upstream_root: Path,
    device: str,
    segnet: torch.nn.Module,
    posenet: torch.nn.Module,
    batch_size: int = EXTRACT_BATCH,
) -> tuple:
    """Extract SegNet masks + PoseNet targets from a GT video.

    Models are accepted as parameters — load them once with :func:`_load_models`
    and reuse across extraction, optimization, and proxy scoring to avoid
    redundant weight I/O.

    Frame count is truncated to (N//2)*2 to match DALI/AVVideoDataset pair semantics
    (both discard an odd trailing frame).

    PoseNet targets use NON-OVERLAPPING pairs (f_{2k}, f_{2k+1}), seq_len=2, matching
    the upstream scorer's DaliVideoDataset.

    Returns:
        (masks, pose_targets, gt_frames) where:
          masks:        (M, SEGNET_H, SEGNET_W) long tensor — SegNet argmax per frame
          pose_targets: (M//2, 6) float tensor — PoseNet output per non-overlapping pair
          gt_frames:    (M, CAMERA_H, CAMERA_W, 3) float tensor in [0, 255]
          (M = (N//2)*2, even frame count)
    """
    import torch

    print(f"  Decoding {video_path} ...")
    gt_frames_uint8 = _decode_frames(video_path)  # (N, H, W, 3) uint8 at camera resolution
    N_raw = gt_frames_uint8.shape[0]

    # Truncate to even frame count — matches DALI/AVVideoDataset which discard odd trailing frame
    N = (N_raw // 2) * 2
    if N != N_raw:
        print(f"  Truncated {N_raw} → {N} frames (odd trailing frame discarded, matching DALI)")
    gt_frames_uint8 = gt_frames_uint8[:N]
    gt_frames = gt_frames_uint8.float()  # (N, H, W, 3) float in [0, 255]
    del gt_frames_uint8  # free 3.66 GB uint8 before extraction loops
    N, H, W, C = gt_frames.shape
    print(f"  {N} frames at {H}×{W}×{C}")

    # ── SegNet masks ──────────────────────────────────────────────────────────
    # SegNet.preprocess_input(x) expects (B, T, C, H, W) and takes x[:, -1, ...]
    # Feed each frame as T=1 so last-frame logic picks it up.
    # Masks come out at (SEGNET_H=384, SEGNET_W=512).
    print("  Extracting SegNet masks ...")
    masks_list: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = gt_frames[start:end].to(device)              # (B, H, W, 3)
            batch_tchw = batch.permute(0, 3, 1, 2).unsqueeze(1)   # (B, 1, C, H, W)
            seg_in = segnet.preprocess_input(batch_tchw)
            logits = segnet(seg_in)                               # (B, 5, SEGNET_H, SEGNET_W)
            masks_list.append(logits.argmax(dim=1).cpu())         # (B, SEGNET_H, SEGNET_W)
    masks = torch.cat(masks_list, dim=0)  # (N, SEGNET_H, SEGNET_W)
    print(f"  Masks: {masks.shape}, classes: {masks.unique().tolist()}")

    # ── PoseNet targets — non-overlapping pairs ───────────────────────────────
    # Upstream scorer uses DaliVideoDataset(seq_len=2): non-overlapping pairs
    # (f0,f1), (f2,f3), ..., (f_{N-2}, f_{N-1}).  N//2 pairs total.
    # PoseNet.preprocess_input(x) expects (B, 2, C, H, W).
    print("  Extracting PoseNet targets (non-overlapping pairs, seq_len=2) ...")
    num_pairs = N // 2
    targets_list: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, num_pairs, batch_size):
            end = min(start + batch_size, num_pairs)
            # Pair k = (frames[2k], frames[2k+1]).
            # Stride-2 slice extracts first/second frame of each pair.
            f1 = gt_frames[2 * start:2 * end:2].to(device)          # (B, H, W, 3)
            f2 = gt_frames[2 * start + 1:2 * end + 1:2].to(device)  # (B, H, W, 3)
            pairs_hwc = torch.stack([f1, f2], dim=1)                 # (B, 2, H, W, 3)
            pairs_chw = pairs_hwc.permute(0, 1, 4, 2, 3).contiguous()   # (B, 2, C, H, W)
            pose_in = posenet.preprocess_input(pairs_chw)
            pose_out = posenet(pose_in)                              # {"pose": (B, 12)}
            targets_list.append(pose_out["pose"][..., :6].cpu())     # (B, 6)
    pose_targets = torch.cat(targets_list, dim=0)  # (N//2, 6)
    print(f"  Pose targets: {pose_targets.shape}")

    return masks, pose_targets, gt_frames


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def compute_proxy_score(
    generated_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    segnet: torch.nn.Module,
    posenet: torch.nn.Module,
    device: str,
    archive_size_bytes: int = 0,
) -> dict:
    """Approximate proxy score: 100*seg_dist + sqrt(10*pose_dist) + 25*rate.

    Both *generated_frames* and *gt_frames* must be at the SAME spatial resolution.
    PoseNet.preprocess_input performs channel operations (RGB → YUV6) at H/2 × W/2 —
    if the two tensors have different spatial sizes the YUV6 shapes will mismatch and
    distortion will be wrong.

    Input tensors may be on CPU; per-batch .to(device) is applied inside the function.
    This avoids pre-allocating full camera-res tensors on GPU (14.65 GB each → OOM on T4).

    Accepts pre-loaded models to avoid redundant weight loading.
    SegNet and PoseNet distortions are computed with proper weighted averaging
    (matches upstream evaluate.py accumulation pattern).

    Args:
        generated_frames:    (N, H, W, 3) float tensor — same H×W as gt_frames; may be CPU
        gt_frames:           (N, H, W, 3) float tensor — same H×W as generated_frames; may be CPU
        segnet:              loaded SegNet model on device
        posenet:             loaded PoseNet model on device
        device:              target device string
        archive_size_bytes:  size of archive.zip in bytes for rate computation.
                             0 → rate = 0 (use when archive not yet written).

    NOTE — Rate formula deviation from upstream evaluate.py:
        Upstream uses:  rate = archive_bytes / sum_of_gt_video_file_sizes_on_disk
        This proxy uses: rate = archive_bytes / (N * CAMERA_H * CAMERA_W * 3)
        The denominators differ by ~20–70x (MKV vs raw). For seed-only archives
        (< 100 bytes) this is negligible. For archives > 1 MB, the proxy rate
        will be ~20–70x smaller than the true rate contribution.
    """
    import math
    import torch

    N = generated_frames.shape[0]
    # Do NOT pre-load all frames on GPU — at camera resolution each tensor is
    # N×874×1164×3 float32 = 14.65 GB; two pre-loads = 29.3 GB → OOM on T4 (16 GB).
    # Load one batch at a time via per-slice .to(device) inside each loop.

    # ── SegNet distortion ─────────────────────────────────────────────────────
    # Weighted accumulation (matches evaluate.py: sum then divide by total count).
    seg_total: float = 0.0
    seg_count: int = 0
    with torch.no_grad():
        for start in range(0, N, 16):
            end = min(start + 16, N)
            gen_b = generated_frames[start:end].to(device).permute(0, 3, 1, 2).unsqueeze(1)  # (B, 1, C, H, W)
            gt_b = gt_frames[start:end].to(device).permute(0, 3, 1, 2).unsqueeze(1)
            gen_mask = segnet(segnet.preprocess_input(gen_b)).argmax(dim=1)
            gt_mask = segnet(segnet.preprocess_input(gt_b)).argmax(dim=1)
            seg_total += (gen_mask != gt_mask).float().sum().item()
            seg_count += gen_mask.numel()  # B * SEGNET_H * SEGNET_W
    seg_dist = seg_total / max(seg_count, 1)

    # ── PoseNet distortion — non-overlapping pairs ────────────────────────────
    num_pairs = N // 2
    pose_total: float = 0.0
    pose_count: int = 0
    with torch.no_grad():
        for start in range(0, num_pairs, 16):
            end = min(start + 16, num_pairs)
            gen_f1 = generated_frames[2 * start:2 * end:2].to(device)
            gen_f2 = generated_frames[2 * start + 1:2 * end + 1:2].to(device)
            gt_f1 = gt_frames[2 * start:2 * end:2].to(device)
            gt_f2 = gt_frames[2 * start + 1:2 * end + 1:2].to(device)

            gen_pairs = torch.stack([gen_f1, gen_f2], dim=1).permute(0, 1, 4, 2, 3).contiguous()
            gt_pairs = torch.stack([gt_f1, gt_f2], dim=1).permute(0, 1, 4, 2, 3).contiguous()

            gen_pose = posenet(posenet.preprocess_input(gen_pairs))["pose"][..., :6]
            gt_pose = posenet(posenet.preprocess_input(gt_pairs))["pose"][..., :6]
            pose_total += (gen_pose - gt_pose).pow(2).sum().item()
            pose_count += gen_pose.numel()  # B * 6
    pose_dist = pose_total / max(pose_count, 1)

    # Rate: archive_bytes / raw_video_bytes  (upstream evaluate.py formula)
    n_raw_bytes = N * CAMERA_H * CAMERA_W * 3
    rate = archive_size_bytes / max(n_raw_bytes, 1)
    score = 100.0 * seg_dist + math.sqrt(10.0 * pose_dist) + 25.0 * rate
    return {"score": score, "seg_dist": seg_dist, "pose_dist": pose_dist, "rate": rate}


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main(
    input_root: Path | None = None,
    working_dir: Path | None = None,
    launcher_path: Path | None = None,
) -> int:
    """Full constrained gen evaluation — provider-agnostic.

    Paths default to platform-detected values (Kaggle/Modal/Lightning/local).
    Override via CLOUD_INPUT_ROOT, CLOUD_WORKING_DIR, CLOUD_PLATFORM env vars.

    Environment variables:
        CONSTRAINED_NUM_STEPS  — optimization steps (default 1000)
        CONSTRAINED_LR         — Adam learning rate (default 0.1)
        CONSTRAINED_VIDEO      — relative path inside asset slug to test video
    """
    from tac.deploy.cloud_paths import CloudPaths, try_ensure_uv, ensure_packages

    # Resolve platform paths
    if input_root is None or working_dir is None:
        cloud = CloudPaths.detect()
        if input_root is None:
            input_root = cloud.input_root
        if working_dir is None:
            working_dir = cloud.working_dir
        print(f"  Platform: {cloud.platform}")
    working_dir.mkdir(parents=True, exist_ok=True)

    num_steps = int(os.environ.get("CONSTRAINED_NUM_STEPS", NUM_STEPS))
    lr = float(os.environ.get("CONSTRAINED_LR", LR))
    video_rel = os.environ.get("CONSTRAINED_VIDEO", "decode_base_archive/0.mkv")
    # Disable early stopping for first run — loss landscape unknown, patience=50 too aggressive.
    # Set CONSTRAINED_EARLY_STOP=1 to re-enable after we have empirical loss curves.
    early_stop_patience = 0 if not int(os.environ.get("CONSTRAINED_EARLY_STOP", 0)) else 50

    print("=== Constrained generation experiment ===")
    print(f"  num_steps={num_steps}, lr={lr}, seed={NOISE_SEED}")
    print(f"  seg_weight={SEG_WEIGHT}, pose_weight={POSE_WEIGHT}")
    print(f"  kill_threshold={KILL_THRESHOLD} after {KILL_STEP} steps")

    import torch
    if torch.cuda.is_available():
        # PyTorch >= 2.5 dropped sm_60 (P100) support. Minimum is now sm_70 (V100+).
        # Capability check alone is insufficient: Kaggle can reassign GPUs between the
        # check and actual compute. Warmup with a real CUDA op to catch execution failure.
        _major, _minor = torch.cuda.get_device_capability(0)
        _dev_name = torch.cuda.get_device_name(0)
        if _major < 7:
            print(f"  WARNING: {_dev_name} (sm_{_major}{_minor}) < sm_70 — capability check failed, using CPU")
            device = "cpu"
        else:
            # Warmup: force a real CUDA kernel to confirm the device is actually usable.
            # This catches GPU reassignment races where capability() returns stale info.
            try:
                _ = torch.zeros(1, device="cuda") + 1
                device = "cuda"
            except Exception as _e:
                print(f"  WARNING: {_dev_name} CUDA warmup failed ({_e}) — falling back to CPU")
                device = "cpu"
    else:
        device = "cpu"
    print(f"  Device: {device}")

    # Install deps — uv preferred, pip fallback for envs where uv is unavailable
    uv = try_ensure_uv()
    ensure_packages(uv, *PIP_DEPS)

    upstream = ensure_upstream(working_dir)
    _add_upstream_to_path(upstream)
    existing_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        f"{upstream}:{existing_path}" if existing_path else str(upstream)
    )

    # Locate test video.  Four search paths in priority order:
    #   1. Explicit CONSTRAINED_VIDEO env var override
    #   2. Dataset mount: /kaggle/input/<slug>/<video_rel>
    #   3. Working dir fallback: /kaggle/working/<video_rel>
    #   4. Upstream clone: ensure_upstream() puts videos/ at upstream/videos/
    #   5. Dataset root flat: /kaggle/input/<slug>/<stem>.mkv (video uploaded without subdir)
    video_path = input_root / ASSET_DATASET_SLUG / video_rel
    if not video_path.exists():
        video_path = working_dir / video_rel
    if not video_path.exists():
        video_path = upstream / "videos" / Path(video_rel).name
    if not video_path.exists():
        video_path = input_root / ASSET_DATASET_SLUG / Path(video_rel).name
    if not video_path.exists():
        raise FileNotFoundError(
            f"Test video not found.\n"
            f"  Checked: {input_root / ASSET_DATASET_SLUG / video_rel}\n"
            f"  Checked: {working_dir / video_rel}\n"
            f"  Checked: {upstream / 'videos' / Path(video_rel).name}\n"
            f"  Checked: {input_root / ASSET_DATASET_SLUG / Path(video_rel).name}\n"
            f"  Set CONSTRAINED_VIDEO env var to the correct relative path."
        )
    print(f"  Video: {video_path}")

    # Load models ONCE — shared across extraction, optimization, and proxy scoring.
    # Avoids redundant safetensors I/O (each load reads ~600 MB from disk).
    segnet, posenet = _load_models(upstream, device)
    print(f"  Loaded SegNet + PoseNet from {upstream / 'models'}")

    # Extract masks + pose targets from GT video
    t0 = time.time()
    masks, pose_targets, gt_frames = extract_masks_and_targets(
        video_path, upstream, device, segnet, posenet, batch_size=EXTRACT_BATCH,
    )
    print(f"  Asset extraction: {time.time() - t0:.1f}s")

    # Run constrained generation
    # Frames are generated at SegNet input resolution (384×512, ~884 MB on T4).
    # Both SegNet and PoseNet resize their inputs internally, so generating at
    # this resolution is semantically equivalent to camera resolution (874×1164).
    from tac.constrained_gen import constrained_generate

    print(f"\n  Running constrained_generate ({num_steps} steps) ...")
    t1 = time.time()

    # Phase 1: run first KILL_STEP steps, evaluate, enforce kill criterion.
    generated = constrained_generate(
        masks=masks.to(device),
        expected_pose=pose_targets.to(device),
        posenet=posenet,
        segnet=segnet,
        noise_seed=NOISE_SEED,
        num_steps=KILL_STEP,
        lr=lr,
        seg_weight=SEG_WEIGHT,
        pose_weight=POSE_WEIGHT,
        device=device,
        log_every=LOG_EVERY,
        early_stop_patience=early_stop_patience,
        segnet_batch_size=EXTRACT_BATCH,
        posenet_batch_size=EXTRACT_BATCH,
    )

    # Kill check: pre-registered hypothesis says kill if proxy > KILL_THRESHOLD
    # after KILL_STEP steps (experiment protocol per CLAUDE.md).
    # Memory-safe: use SegNet resolution (384×512, ~884 MB) for both gen and gt.
    # Camera-res GPU upsample = 14.65 GB → OOM on T4.
    # CPU-side full-tensor .contiguous() on camera-res gt = ANOTHER 14.65 GB → peak 29.3 GB
    # → also OOM on Kaggle (26–29 GB CPU RAM).
    # Fix: batch downsample gt in 64-frame chunks; peak per chunk = 784 MB, total safe.
    import torch.nn.functional as _F
    _DSAMP_BATCH = 64
    _gt_small_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for _ds in range(0, gt_frames.shape[0], _DSAMP_BATCH):
            _de = min(_ds + _DSAMP_BATCH, gt_frames.shape[0])
            _chunk = gt_frames[_ds:_de].permute(0, 3, 1, 2).contiguous()  # (B, 3, CAMERA_H, CAMERA_W)
            _small = _F.interpolate(_chunk, size=(SEGNET_H, SEGNET_W), mode="bilinear", align_corners=False)
            _gt_small_chunks.append(_small.permute(0, 2, 3, 1))  # (B, SEGNET_H, SEGNET_W, 3)
            del _chunk, _small
    gt_small = torch.cat(_gt_small_chunks, dim=0)  # (N, SEGNET_H, SEGNET_W, 3) ~884 MB
    del _gt_small_chunks
    kill_scores = compute_proxy_score(
        generated.cpu(), gt_small, segnet, posenet, device,
        archive_size_bytes=0,  # seed not written yet
    )
    del gt_small  # free ~884 MB CPU before phase 2
    print(f"\n  === KILL CHECK (step {KILL_STEP}) ===")
    print(f"  proxy: {kill_scores['score']:.4f}  (kill if > {KILL_THRESHOLD})")
    if kill_scores["score"] > KILL_THRESHOLD:
        print(f"  KILLED — proxy {kill_scores['score']:.4f} exceeds threshold {KILL_THRESHOLD}.")
        print("  Hypothesis: constrained gen from noise does not converge fast enough.")
        return 1

    print(f"  Passed kill check — continuing to {num_steps} steps ...")

    # Phase 2: continue remaining steps warm-started from phase-1 result.
    remaining_steps = num_steps - KILL_STEP
    if remaining_steps > 0:
        generated = constrained_generate(
            masks=masks.to(device),
            expected_pose=pose_targets.to(device),
            posenet=posenet,
            segnet=segnet,
            noise_seed=NOISE_SEED,  # ignored — init_frames overrides initialization
            num_steps=remaining_steps,
            lr=lr,
            seg_weight=SEG_WEIGHT,
            pose_weight=POSE_WEIGHT,
            device=device,
            log_every=LOG_EVERY,
            early_stop_patience=early_stop_patience,
            segnet_batch_size=EXTRACT_BATCH,
            posenet_batch_size=EXTRACT_BATCH,
            init_frames=generated,  # warm-start from phase-1 result
        )

    gen_time = time.time() - t1
    print(f"  Generation time: {gen_time:.1f}s ({num_steps / gen_time:.1f} steps/s)")

    # Upsample from SegNet resolution (384×512) to camera resolution (874×1164)
    # for frames.raw compatibility with TensorVideoDataset.
    # Write in 64-frame chunks — avoids materializing a 14.65 GB camera-res float32
    # tensor and a separate 3.66 GB uint8 numpy array simultaneously.
    # (14.65 + 3.66 = 18.3 GB; adding gt_frames = 33 GB → OOM on Kaggle 26 GB CPU RAM)
    # Chunk approach: peak per batch = 64×874×1164×3×4 = 784 MB. Total ≈ 16.5 GB.
    import torch.nn.functional as F  # noqa: PLC0415 — deferred to avoid top-level torch dep
    import numpy as np

    video_name = Path(video_rel).stem  # e.g. "0"
    inflated_dir = working_dir / "inflated"
    inflated_dir.mkdir(parents=True, exist_ok=True)
    raw_path = inflated_dir / f"{video_name}.raw"

    _WRITE_BATCH = 64
    generated_bchw_cpu = generated.cpu().permute(0, 3, 1, 2).float()  # (N, 3, 384, 512) CPU ~884 MB
    n_written = 0
    with open(raw_path, "wb") as _raw_f:
        for _ws in range(0, generated_bchw_cpu.shape[0], _WRITE_BATCH):
            _we = min(_ws + _WRITE_BATCH, generated_bchw_cpu.shape[0])
            _chunk_cam = (
                F.interpolate(
                    generated_bchw_cpu[_ws:_we],
                    size=(CAMERA_H, CAMERA_W),
                    mode="bilinear",
                    align_corners=False,
                )
                .clamp(0.0, 255.0)
                .permute(0, 2, 3, 1)
                .contiguous()  # permute() returns non-contiguous view; .numpy() requires contiguous
            )  # (B, 874, 1164, 3) float — 784 MB peak per chunk
            _raw_f.write(_chunk_cam.numpy().astype(np.uint8))  # numpy arrays satisfy buffer protocol; no .tobytes() copy needed
            n_written += _we - _ws
            del _chunk_cam
    del generated_bchw_cpu  # free 884 MB CPU before proxy score
    print(f"  Wrote {n_written} frames to {raw_path}")

    # Write minimal archive.zip (seed only) so upstream evaluate.py stat() succeeds
    archive_path = working_dir / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("seed.bin", struct.pack("<Q", NOISE_SEED))
    print(f"  Archive: {archive_path} ({archive_path.stat().st_size} bytes)")

    # Compute proxy score at SegNet resolution — avoids materializing a second
    # 14.65 GB camera-res tensor alongside gt_frames (would be 29.3 GB peak → OOM).
    # Score at SegNet res (384×512) vs camera res produces slightly different absolute
    # values but the same relative signal; clearly labeled in output.
    # NOTE — rate denominator is hardcoded to CAMERA_H×CAMERA_W bytes (not input H×W).
    # For a 64-byte seed archive the rate ≈ 0 regardless, so this is inconsequential.
    print("\n  Computing proxy score (at SegNet resolution 384×512) ...")
    _gt_small_proxy_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for _ps in range(0, gt_frames.shape[0], _DSAMP_BATCH):
            _pe = min(_ps + _DSAMP_BATCH, gt_frames.shape[0])
            _pc = gt_frames[_ps:_pe].permute(0, 3, 1, 2).contiguous()
            _sm = _F.interpolate(_pc, size=(SEGNET_H, SEGNET_W), mode="bilinear", align_corners=False)
            _gt_small_proxy_chunks.append(_sm.permute(0, 2, 3, 1))
            del _pc, _sm
    gt_small_proxy = torch.cat(_gt_small_proxy_chunks, dim=0)
    del _gt_small_proxy_chunks
    scores = compute_proxy_score(
        generated.cpu(), gt_small_proxy, segnet, posenet, device,
        archive_size_bytes=archive_path.stat().st_size,
    )
    del gt_small_proxy
    print("\n  === RESULT ===")
    print(f"  score:     {scores['score']:.4f}")
    print(f"  seg_dist:  {scores['seg_dist']:.6f}")
    print(f"  pose_dist: {scores['pose_dist']:.6f}")
    print(f"  rate:      {scores['rate']:.6f} (seed-only archive)")

    # Save manifest
    manifest = {
        "experiment": "constrained_gen_smoke",
        "platform": os.environ.get("CLOUD_PLATFORM", "auto"),
        "num_steps": num_steps,
        "lr": lr,
        "seg_weight": SEG_WEIGHT,
        "pose_weight": POSE_WEIGHT,
        "noise_seed": NOISE_SEED,
        "video": str(video_path),
        "n_frames": n_written,
        "n_pairs": int(masks.shape[0]) // 2,
        "gen_time_s": gen_time,
        "device": device,
        "score": scores["score"],
        "seg_dist": scores["seg_dist"],
        "pose_dist": scores["pose_dist"],
        "rate": scores["rate"],
        "inflated_path": str(raw_path),
        "archive_path": str(archive_path),
    }
    manifest_path = working_dir / "constrained_gen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest: {manifest_path}")

    return 0
