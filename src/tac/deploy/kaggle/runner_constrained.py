"""Kaggle runner for constrained generation inflate experiment.

No model training. Full inflate pipeline:
  GT video -> SegNet masks -> PoseNet pose targets -> constrained_generate() -> .raw frames

Archive size: 64 bytes (noise seed only). Masks + targets extracted at inflate time
from the GT video using upstream SegNet/PoseNet (already on scorer machine).

Pre-registered hypothesis: proxy < 0.80 after 1000 steps on T4.
Kill: proxy > 1.50 after 500 steps.

Decode path: av + frame_utils.yuv420_to_rgb (matches NVDEC output pixel-for-pixel).
PoseNet pairs: non-overlapping (f_{2k}, f_{2k+1}) — seq_len=2, matching upstream scorer.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASSET_DATASET_SLUG: str = "comma-lab-private-assets"
UPSTREAM_REPO: str = "https://github.com/commaai/comma_video_compression_challenge.git"
PIP_DEPS: tuple[str, ...] = ("av", "safetensors", "timm", "einops",
                              "segmentation-models-pytorch", "pydantic", "click")

NUM_STEPS: int = 1000
LR: float = 0.1
SEG_WEIGHT: float = 50.0
POSE_WEIGHT: float = 50.0
NOISE_SEED: int = 42
LOG_EVERY: int = 100
EXTRACT_BATCH: int = 16  # batch size for mask + pose extraction


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def ensure_deps() -> None:
    missing = [d for d in PIP_DEPS if not _is_importable(d)]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)


def _is_importable(pkg: str) -> bool:
    try:
        __import__(pkg.replace("-", "_"))
        return True
    except ImportError:
        return False


def ensure_upstream(working_dir: Path) -> Path:
    upstream = working_dir / "upstream"
    if not (upstream / "models").exists():
        subprocess.check_call(["git", "clone", "--depth", "1", UPSTREAM_REPO, str(upstream)])
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


def _add_upstream_to_path(upstream_root: Path) -> None:
    """Add upstream repo root to sys.path so frame_utils + modules are importable."""
    upstream_str = str(upstream_root)
    if upstream_str not in sys.path:
        sys.path.insert(0, upstream_str)


# ---------------------------------------------------------------------------
# Core: decode GT video + extract masks/targets from upstream models
# ---------------------------------------------------------------------------

def _decode_frames(video_path: Path) -> "torch.Tensor":
    """Decode all frames via av + yuv420_to_rgb.

    yuv420_to_rgb (from frame_utils) uses bilinear chroma upsampling + BT.601
    limited-range conversion, which matches NVDEC (DALI) output pixel-for-pixel.

    Returns:
        (N, H, W, 3) uint8 torch tensor.
    """
    import av
    import torch
    from frame_utils import yuv420_to_rgb  # matches NVDEC output

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames_list: list["torch.Tensor"] = []
    for frame in container.decode(stream):
        frames_list.append(yuv420_to_rgb(frame))  # (H, W, 3) uint8
    container.close()
    if not frames_list:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return torch.stack(frames_list, dim=0)  # (N, H, W, 3) uint8


def _load_models(
    upstream_root: Path, device: str
) -> "tuple[torch.nn.Module, torch.nn.Module]":
    """Load SegNet + PoseNet with weights from upstream safetensors files."""
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
    batch_size: int = EXTRACT_BATCH,
) -> tuple:
    """Extract SegNet masks + PoseNet targets from a GT video.

    Decode path: av + frame_utils.yuv420_to_rgb (matches NVDEC/DALI output).
    PoseNet uses NON-OVERLAPPING pairs (f_{2k}, f_{2k+1}), seq_len=2, to match
    the upstream scorer's DaliVideoDataset semantics.

    Returns:
        (masks, pose_targets, gt_frames) where:
          masks:        (N, H_out, W_out) long tensor — SegNet argmax per frame
          pose_targets: (N//2, 6) float tensor — PoseNet output per non-overlapping pair
          gt_frames:    (N, H, W, 3) float tensor in [0, 255]
    """
    import torch

    _add_upstream_to_path(upstream_root)
    segnet, posenet = _load_models(upstream_root, device)
    print(f"  Loaded SegNet + PoseNet from {upstream_root / 'models'}")

    print(f"  Decoding {video_path} ...")
    gt_frames_uint8 = _decode_frames(video_path)          # (N, H, W, 3) uint8
    gt_frames = gt_frames_uint8.float()                    # (N, H, W, 3) float
    N, H, W, C = gt_frames.shape
    print(f"  {N} frames decoded at {H}x{W}x{C}")

    # ── SegNet masks ──────────────────────────────────────────────────────────
    # SegNet.preprocess_input(x) expects (B, T, C, H, W) and uses x[:, -1, ...]
    # Feed each frame as a T=1 sequence so the last-frame logic picks it up.
    print("  Extracting SegNet masks ...")
    masks_list: list["torch.Tensor"] = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = gt_frames[start:end].to(device)             # (B, H, W, 3)
            batch_tchw = batch.permute(0, 3, 1, 2).unsqueeze(1)  # (B, 1, C, H, W)
            seg_in = segnet.preprocess_input(batch_tchw)
            logits = segnet(seg_in)                              # (B, 5, H_out, W_out)
            masks_list.append(logits.argmax(dim=1).cpu())        # (B, H_out, W_out)
    masks = torch.cat(masks_list, dim=0)  # (N, H_out, W_out)
    print(f"  Masks: {masks.shape}, classes: {masks.unique().tolist()}")

    # ── PoseNet targets — non-overlapping pairs ───────────────────────────────
    # Upstream scorer uses DaliVideoDataset(seq_len=2) which yields non-overlapping
    # pairs: (f0,f1), (f2,f3), ..., (f_{N-2},f_{N-1}).
    # Number of pairs = N // 2.
    # PoseNet.preprocess_input(x) expects (B, 2, C, H, W).
    print("  Extracting PoseNet targets (non-overlapping pairs, seq_len=2) ...")
    num_pairs = N // 2
    targets_list: list["torch.Tensor"] = []
    with torch.no_grad():
        for start in range(0, num_pairs, batch_size):
            end = min(start + batch_size, num_pairs)
            # Pair k uses frames[2k] and frames[2k+1].
            # Stride-2 slice from 2*start to 2*end gives the first frame of each pair;
            # offset by 1 gives the second frame.
            f1 = gt_frames[2 * start:2 * end:2].to(device)          # (B, H, W, 3)
            f2 = gt_frames[2 * start + 1:2 * end + 1:2].to(device)  # (B, H, W, 3)
            pairs_hwc = torch.stack([f1, f2], dim=1)                 # (B, 2, H, W, 3)
            pairs_chw = pairs_hwc.permute(0, 1, 4, 2, 3).contiguous()  # (B, 2, C, H, W)
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
    generated_frames: "torch.Tensor",
    gt_frames: "torch.Tensor",
    upstream_root: Path,
    device: str,
) -> dict:
    """Approximate proxy score: 100*seg_dist + sqrt(10*pose_dist) + 25*rate.

    SegNet distortion: fraction of pixels where argmax disagrees.
    PoseNet distortion: MSE between pose outputs for non-overlapping pairs.
    Rate: 0 (seed-only archive, no neural weights in zip).

    Uses the same decode-path and pair semantics as extract_masks_and_targets.
    """
    import math
    import torch

    _add_upstream_to_path(upstream_root)
    segnet, posenet = _load_models(upstream_root, device)

    N = generated_frames.shape[0]
    gen = generated_frames.to(device)  # (N, H, W, 3)
    gt = gt_frames.to(device)          # (N, H, W, 3)

    # ── SegNet distortion ─────────────────────────────────────────────────────
    seg_dists: list[float] = []
    with torch.no_grad():
        for start in range(0, N, 16):
            end = min(start + 16, N)
            gen_b = gen[start:end].permute(0, 3, 1, 2).unsqueeze(1)  # (B, 1, C, H, W)
            gt_b = gt[start:end].permute(0, 3, 1, 2).unsqueeze(1)
            gen_mask = segnet(segnet.preprocess_input(gen_b)).argmax(dim=1)
            gt_mask = segnet(segnet.preprocess_input(gt_b)).argmax(dim=1)
            seg_dists.append((gen_mask != gt_mask).float().mean().item())
    seg_dist = sum(seg_dists) / len(seg_dists)

    # ── PoseNet distortion — non-overlapping pairs ────────────────────────────
    num_pairs = N // 2
    pose_dists: list[float] = []
    with torch.no_grad():
        for start in range(0, num_pairs, 16):
            end = min(start + 16, num_pairs)
            gen_f1 = gen[2 * start:2 * end:2]
            gen_f2 = gen[2 * start + 1:2 * end + 1:2]
            gt_f1 = gt[2 * start:2 * end:2]
            gt_f2 = gt[2 * start + 1:2 * end + 1:2]

            gen_pairs = torch.stack([gen_f1, gen_f2], dim=1).permute(0, 1, 4, 2, 3).contiguous()
            gt_pairs = torch.stack([gt_f1, gt_f2], dim=1).permute(0, 1, 4, 2, 3).contiguous()

            gen_pose = posenet(posenet.preprocess_input(gen_pairs))["pose"][..., :6]
            gt_pose = posenet(posenet.preprocess_input(gt_pairs))["pose"][..., :6]
            pose_dists.append((gen_pose - gt_pose).pow(2).mean().item())
    pose_dist = sum(pose_dists) / len(pose_dists)

    score = 100.0 * seg_dist + math.sqrt(10.0 * pose_dist) + 0.0  # rate=0 (seed-only)
    return {"score": score, "seg_dist": seg_dist, "pose_dist": pose_dist, "rate": 0.0}


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main(
    input_root: Path | None = None,
    working_dir: Path | None = None,
    launcher_path: Path | None = None,
) -> int:
    """Full constrained gen evaluation on Kaggle T4.

    Environment variables:
        CONSTRAINED_NUM_STEPS  — optimization steps (default 1000)
        CONSTRAINED_LR         — Adam learning rate (default 0.1)
        CONSTRAINED_VIDEO      — relative path inside dataset to test video
    """
    if input_root is None:
        input_root = Path("/kaggle/input")
    if working_dir is None:
        working_dir = Path("/kaggle/working")
    working_dir.mkdir(parents=True, exist_ok=True)

    num_steps = int(os.environ.get("CONSTRAINED_NUM_STEPS", NUM_STEPS))
    lr = float(os.environ.get("CONSTRAINED_LR", LR))
    video_rel = os.environ.get("CONSTRAINED_VIDEO", "decode_base_archive/0.mkv")

    print("=== Kaggle constrained generation experiment ===")
    print(f"  num_steps={num_steps}, lr={lr}, seed={NOISE_SEED}")
    print(f"  seg_weight={SEG_WEIGHT}, pose_weight={POSE_WEIGHT}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    ensure_deps()
    upstream = ensure_upstream(working_dir)

    # Ensure upstream is on PYTHONPATH so sub-imports inside modules.py work
    _add_upstream_to_path(upstream)
    existing_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        f"{upstream}:{existing_path}" if existing_path else str(upstream)
    )

    # Locate test video
    video_path = input_root / ASSET_DATASET_SLUG / video_rel
    if not video_path.exists():
        video_path = working_dir / video_rel
    if not video_path.exists():
        raise FileNotFoundError(
            f"Test video not found.\n"
            f"  Checked: {input_root / ASSET_DATASET_SLUG / video_rel}\n"
            f"  Checked: {working_dir / video_rel}\n"
            f"  Set CONSTRAINED_VIDEO env var to the correct relative path."
        )
    print(f"  Video: {video_path}")

    # Extract masks + pose targets from GT video
    t0 = time.time()
    masks, pose_targets, gt_frames = extract_masks_and_targets(
        video_path, upstream, device, batch_size=EXTRACT_BATCH,
    )
    print(f"  Asset extraction: {time.time() - t0:.1f}s")

    # Load models for optimization (same weights used in extraction)
    segnet_opt, posenet_opt = _load_models(upstream, device)

    # Run constrained generation
    from tac.constrained_gen import constrained_generate

    print(f"\n  Running constrained_generate ({num_steps} steps) ...")
    t1 = time.time()
    generated = constrained_generate(
        masks=masks.to(device),
        expected_pose=pose_targets.to(device),
        posenet=posenet_opt,
        segnet=segnet_opt,
        noise_seed=NOISE_SEED,
        num_steps=num_steps,
        lr=lr,
        seg_weight=SEG_WEIGHT,
        pose_weight=POSE_WEIGHT,
        device=device,
        log_every=LOG_EVERY,
        segnet_batch_size=EXTRACT_BATCH,
        posenet_batch_size=EXTRACT_BATCH,
    )
    gen_time = time.time() - t1
    print(f"  Generation time: {gen_time:.1f}s ({num_steps / gen_time:.1f} steps/s)")

    # Write output
    import numpy as np
    out_dir = working_dir / "constrained_gen_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_np = generated.cpu().numpy().astype(np.uint8)
    raw_path = out_dir / "frames.raw"
    generated_np.tofile(raw_path)
    print(f"  Wrote {generated_np.shape[0]} frames to {raw_path}")

    # Compute proxy score
    print("\n  Computing proxy score ...")
    scores = compute_proxy_score(generated, gt_frames, upstream, device)
    print(f"\n  === RESULT ===")
    print(f"  score:     {scores['score']:.4f}")
    print(f"  seg_dist:  {scores['seg_dist']:.6f}")
    print(f"  pose_dist: {scores['pose_dist']:.6f}")
    print(f"  rate:      {scores['rate']:.6f} (seed-only archive)")

    # Save manifest
    manifest = {
        "experiment": "constrained_gen_smoke",
        "num_steps": num_steps,
        "lr": lr,
        "seg_weight": SEG_WEIGHT,
        "pose_weight": POSE_WEIGHT,
        "noise_seed": NOISE_SEED,
        "video": str(video_path),
        "n_frames": int(generated_np.shape[0]),
        "n_pairs": int(masks.shape[0]) // 2,
        "gen_time_s": gen_time,
        "device": device,
        "score": scores["score"],
        "seg_dist": scores["seg_dist"],
        "pose_dist": scores["pose_dist"],
        "rate": scores["rate"],
    }
    manifest_path = working_dir / "constrained_gen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest: {manifest_path}")

    return 0
