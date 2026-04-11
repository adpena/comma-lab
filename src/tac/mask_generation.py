#!/usr/bin/env python3
"""Generate per-frame importance masks using ML segmentation.

Uses Falcon Perception for text-prompted segmentation and SAM 3 (or SAM 2
fallback) for temporal propagation across video frames.

Toolchain priority (conditioned on availability):
  1. SAM 3 MLX (native Apple Silicon — fastest local path)
  2. SAM 3 (requires CUDA/Triton — use on bat00 with RTX 2070 Super)
  3. SAM 2.1 (works on Apple Silicon via MPS — local fallback)
  4. Gradient + temporal variance masking (no ML — last resort)

Output: a numpy .npy file containing float32 masks (N, H, W) in [0, 1]
where 1 = important, 0 = unimportant.

Requirements:
  - Falcon Perception (MLX backend): installed at workspace/tools/falcon-perception
  - SAM 3 (CUDA): installed at workspace/tools/sam3 (preferred)
  - SAM 2.1 (MPS fallback): installed in same venv if SAM 3 unavailable
  - ffmpeg/ffprobe on PATH for video decoding

Usage:
  python -m tac.mask_generation \\
    --input <video_path> \\
    --output <masks.npy> \\
    --strategy [specific|general|hybrid] \\
    --model [falcon|sam3|both] \\
    --sample-step 10 \\
    --feather-radius 24
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TOOLS_DIR = _PROJECT_ROOT / "workspace" / "tools"
_FALCON_VENV = _TOOLS_DIR / "falcon-perception" / ".venv"
_SAM2_VENV = _TOOLS_DIR / "sam3" / ".venv"  # SAM 2 installed inside sam3 venv
_MLX_SAM3_VENV = _TOOLS_DIR / "mlx_sam3" / ".venv"  # SAM 3 MLX for Apple Silicon

# ---------------------------------------------------------------------------
# Prompting strategies
# ---------------------------------------------------------------------------
SPECIFIC_PROMPTS = [
    "road surface",
    "cars",
    "lane markings",
    "pedestrians",
    "traffic signs",
]

GENERAL_PROMPT = (
    "everything dynamic and salient to a self-driving car from dashcam footage"
)


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def ffprobe_meta(video: Path, ffprobe_bin: str = "ffprobe") -> dict:
    """Return width, height, fps, and total_frames for *video*."""
    cmd = [
        ffprobe_bin, "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video),
    ]
    cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(cp.stdout)
    stream = data["streams"][0]
    w, h = int(stream["width"]), int(stream["height"])
    num, den = (int(x) for x in stream.get("avg_frame_rate", "0/1").split("/"))
    fps = num / den if den else 30.0
    nf = stream.get("nb_read_frames") or stream.get("nb_frames")
    if nf in (None, "N/A"):
        dur = float(stream.get("duration") or data.get("format", {}).get("duration", 0))
        nf = int(round(dur * fps))
    else:
        nf = int(nf)
    return {"width": w, "height": h, "fps": fps, "total_frames": nf}


def extract_frames(
    video: Path,
    meta: dict,
    sample_step: int = 1,
    ffmpeg_bin: str = "ffmpeg",
) -> np.ndarray:
    """Extract frames from video as (N, H, W, 3) uint8 array.

    If sample_step > 1, only every Nth frame is extracted.
    """
    w, h = meta["width"], meta["height"]
    total = meta["total_frames"]
    frame_bytes = w * h * 3

    cmd = [
        ffmpeg_bin, "-v", "error",
        "-i", str(video),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Pre-allocate to avoid double-copy from list + np.stack
    expected_n = (total // sample_step) + 1
    sampled = np.empty((expected_n, h, w, 3), dtype=np.uint8)
    slot = 0
    idx = 0
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            if idx % sample_step == 0 and slot < expected_n:
                np.copyto(sampled[slot], np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)))
                slot += 1
            idx += 1
            if idx % 300 == 0:
                print(f"  Extracted {idx}/{total} frames ...", file=sys.stderr, flush=True)
    except Exception:
        proc.kill()
        raise
    finally:
        proc.stdout.close()

    proc.wait()
    if proc.returncode != 0:
        print(f"WARNING: ffmpeg exited with code {proc.returncode}", file=sys.stderr)
    trailing = idx - (slot - 1) * sample_step if slot > 0 else 0
    if trailing > 0 and idx % sample_step != 0:
        print(f"  WARNING: {trailing} trailing frames after last sampled frame were dropped", file=sys.stderr)
    print(f"  Extracted {slot} frames (step={sample_step})", file=sys.stderr)
    return sampled[:slot]


# ---------------------------------------------------------------------------
# Gaussian helpers (pure numpy, from roi_preprocess.py)
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = int(np.ceil(3.0 * sigma))
    if radius < 1:
        radius = 1
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def _convolve_1d_axis(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[axis] = (radius, radius)
    padded = np.pad(arr, pad_widths, mode="reflect")
    out = np.zeros_like(arr)
    for i in range(len(kernel)):
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(i, i + arr.shape[axis])
        out += padded[tuple(slices)] * kernel[i]
    return out


def gaussian_blur_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel_1d(sigma)
    tmp = _convolve_1d_axis(arr, kernel, axis=1)
    return _convolve_1d_axis(tmp, kernel, axis=0)


def feather_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    """Feather (smooth) a binary/soft mask with Gaussian blur."""
    if radius <= 0:
        return mask
    blurred = gaussian_blur_2d(mask, sigma=radius)
    mx = blurred.max()
    if mx > 0:
        blurred = np.clip(blurred / mx, 0.0, 1.0)
    return blurred


# ---------------------------------------------------------------------------
# Fallback: gradient + temporal masking (from roi_preprocess.py)
# ---------------------------------------------------------------------------

def _sobel_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    padded = np.pad(gray, 1, mode="reflect")
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    for dy in range(3):
        for dx in range(3):
            gx += padded[dy:dy + gray.shape[0], dx:dx + gray.shape[1]] * kx[dy, dx]
            gy += padded[dy:dy + gray.shape[0], dx:dx + gray.shape[1]] * ky[dy, dx]
    return np.sqrt(gx * gx + gy * gy)


def fallback_adaptive_mask(
    frame: np.ndarray,
    prev_frame: np.ndarray | None,
    feather_radius: float,
) -> np.ndarray:
    """Gradient + temporal difference mask (no ML required).

    This is the fallback used when neither Falcon Perception nor SAM 2 are
    available.  It reproduces the adaptive masking logic from roi_preprocess.py
    but without a fixed spatial prior -- instead it relies purely on gradient
    magnitude and inter-frame motion.
    """
    h, w = frame.shape[:2]
    gray = (0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1]
            + 0.114 * frame[:, :, 2]).astype(np.float32)

    # Gradient magnitude (normalised)
    grad = _sobel_gradient_magnitude(gray)
    gmax = grad.max()
    grad_norm = grad / gmax if gmax > 0 else grad

    # Temporal difference (normalised)
    if prev_frame is not None:
        prev_gray = (0.299 * prev_frame[:, :, 0] + 0.587 * prev_frame[:, :, 1]
                     + 0.114 * prev_frame[:, :, 2]).astype(np.float32)
        temporal = np.abs(gray - prev_gray)
        tmax = temporal.max()
        temporal_norm = temporal / tmax if tmax > 0 else temporal
    else:
        temporal_norm = np.zeros((h, w), dtype=np.float32)

    # Simple spatial prior: lower half of frame is more important (road area)
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    spatial_prior = np.clip(ys * 1.5 - 0.3, 0.0, 1.0) * np.ones((1, w), dtype=np.float32)

    raw_mask = np.maximum(
        spatial_prior * 0.5,
        0.35 * grad_norm + 0.45 * temporal_norm + 0.20 * spatial_prior,
    )
    raw_mask = np.clip(raw_mask * 1.5, 0.0, 1.0)

    if feather_radius > 0:
        raw_mask = feather_mask(raw_mask, radius=min(feather_radius, 8.0))

    return raw_mask


# ---------------------------------------------------------------------------
# Falcon Perception backend
# ---------------------------------------------------------------------------

def _try_import_falcon():
    """Try to import Falcon Perception.  Returns (module, available)."""
    falcon_site = _FALCON_VENV / "lib"
    # Find the python version directory inside the venv
    if falcon_site.exists():
        for d in falcon_site.iterdir():
            sp = d / "site-packages"
            if sp.exists() and str(sp) not in sys.path:
                sys.path.insert(0, str(sp))
    try:
        import falcon_perception
        return falcon_perception, True
    except ImportError:
        return None, False


def falcon_segment_frame(  # noqa: PLR0913 -- mirrors Falcon Perception's inference API
    model,
    tokenizer,
    model_args,
    engine,
    frame_rgb: np.ndarray,
    prompts: list[str],
    task: str = "segmentation",
    min_dim: int = 256,
    max_dim: int = 1024,
) -> np.ndarray:
    """Run Falcon Perception on a single frame, returning a float32 mask (H, W) in [0,1].

    Runs segmentation for each prompt and unions all masks.
    """
    from PIL import Image

    h, w = frame_rgb.shape[:2]
    pil_image = Image.fromarray(frame_rgb)

    # Lazy import of the MLX batch inference engine
    from falcon_perception.mlx.batch_inference import process_batch_and_generate
    from falcon_perception import build_prompt_for_task

    combined_mask = np.zeros((h, w), dtype=np.float32)

    for prompt_text in prompts:
        prompt = build_prompt_for_task(prompt_text, task)
        batch = process_batch_and_generate(
            tokenizer,
            [(pil_image, prompt)],
            max_length=model_args.max_seq_len,
            min_dimension=min_dim,
            max_dimension=max_dim,
        )

        output_tokens, aux_outputs = engine.generate(
            tokens=batch["tokens"],
            pos_t=batch["pos_t"],
            pos_hw=batch["pos_hw"],
            pixel_values=batch["pixel_values"],
            pixel_mask=batch["pixel_mask"],
            max_new_tokens=200,
            temperature=0.0,
            task=task,
        )

        aux = aux_outputs[0]
        # Decode RLE masks and union them
        if hasattr(aux, "masks_rle") and aux.masks_rle:
            try:
                from pycocotools import mask as mask_utils
                for rle in aux.masks_rle:
                    m = mask_utils.decode(rle).astype(np.float32)
                    if m.shape != (h, w):
                        from PIL import Image as PILImg
                        m = np.array(PILImg.fromarray((m * 255).astype(np.uint8)).resize(
                            (w, h), PILImg.NEAREST
                        )).astype(np.float32) / 255.0
                    combined_mask = np.maximum(combined_mask, m)
            except ImportError:
                print("  [warn] pycocotools not available, skipping mask decode",
                      file=sys.stderr)

    return combined_mask


def run_falcon_on_frames(
    frames: np.ndarray,
    prompts: list[str],
) -> np.ndarray:
    """Run Falcon Perception on sampled frames.  Returns (N, H, W) float32 masks."""
    fp_mod, available = _try_import_falcon()
    if not available:
        raise RuntimeError("Falcon Perception not available")

    from falcon_perception import (
        load_and_prepare_model,
        PERCEPTION_MODEL_ID,
    )
    from falcon_perception.mlx.batch_inference import BatchInferenceEngine

    print("Loading Falcon Perception model (MLX) ...", file=sys.stderr)
    t0 = time.perf_counter()
    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=PERCEPTION_MODEL_ID,
        dtype="float16",
        backend="mlx",
    )
    engine = BatchInferenceEngine(model, tokenizer)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    masks = np.zeros((n, h, w), dtype=np.float32)

    for i in range(n):
        t0 = time.perf_counter()
        masks[i] = falcon_segment_frame(
            model, tokenizer, model_args, engine,
            frames[i], prompts,
        )
        elapsed = time.perf_counter() - t0
        print(f"  Frame {i+1}/{n}: {elapsed:.2f}s", file=sys.stderr, flush=True)

    return masks


# ---------------------------------------------------------------------------
# SAM 3 / SAM 2 backend (conditioned on available toolchain)
# ---------------------------------------------------------------------------

# Module-level cache for SAM import results.  Intentionally never cleared:
# the import availability does not change within a single process lifetime.
_sam_import_cache: dict = {}


def _try_import_sam3_mlx():
    """Try to import SAM 3 MLX (Apple Silicon native).

    Checks if workspace/tools/mlx_sam3 venv exists, adds its site-packages
    to sys.path, and attempts ``import sam3``.

    Returns (module, available).
    """
    mlx_site = _MLX_SAM3_VENV / "lib"
    if not mlx_site.exists():
        return None, False
    # Inject the venv's site-packages so the import can resolve
    for d in mlx_site.iterdir():
        sp = d / "site-packages"
        if sp.exists() and str(sp) not in sys.path:
            sys.path.insert(0, str(sp))
    try:
        import sam3  # noqa: F811 – intentional re-import attempt
        return sam3, True
    except ImportError:
        return None, False


def _try_import_sam():
    """Try to import SAM 3, then SAM 2.  Returns (module, version, available).

    Priority:
      1. SAM 3 MLX (Apple Silicon native — fastest local path)
      2. SAM 3 (requires Triton/CUDA — works on bat00 with RTX 2070 Super)
      3. SAM 2.1 (works on Apple Silicon MPS — local fallback)
    """
    if _sam_import_cache:
        return _sam_import_cache["mod"], _sam_import_cache["ver"], _sam_import_cache["ok"]

    # ---- 1. SAM 3 MLX (Apple Silicon) ----
    mlx_mod, mlx_ok = _try_import_sam3_mlx()
    if mlx_ok:
        _sam_import_cache.update(mod=mlx_mod, ver="sam3-mlx", ok=True)
        return mlx_mod, "sam3-mlx", True

    # ---- 2. SAM 3 CUDA / 3. SAM 2 ----
    sam_site = _SAM2_VENV / "lib"
    if sam_site.exists():
        for d in sam_site.iterdir():
            sp = d / "site-packages"
            if sp.exists() and str(sp) not in sys.path:
                sys.path.insert(0, str(sp))

    # Try SAM 3 (CUDA)
    try:
        import sam3
        _sam_import_cache.update(mod=sam3, ver="sam3", ok=True)
        return sam3, "sam3", True
    except ImportError:
        pass

    # Fall back to SAM 2
    try:
        import sam2
        _sam_import_cache.update(mod=sam2, ver="sam2", ok=True)
        return sam2, "sam2", True
    except ImportError:
        pass

    _sam_import_cache.update(mod=None, ver=None, ok=False)
    return None, None, False


def _try_import_sam2():
    """Compat wrapper — returns (module, available)."""
    mod, ver, ok = _try_import_sam()
    return mod, ok


def run_sam2_propagation(
    video_path: Path,
    keyframe_masks: np.ndarray,
    keyframe_indices: list[int],
    total_frames: int,
    meta: dict,
    ffmpeg_bin: str = "ffmpeg",
) -> np.ndarray:
    """Use SAM 2 video predictor to propagate keyframe masks across all frames.

    Takes sparse keyframe masks and propagates them temporally to produce
    dense per-frame masks.

    Returns (total_frames, H, W) float32 masks.
    """
    sam2_mod, available = _try_import_sam2()
    if not available:
        raise RuntimeError("SAM 2 not available")

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    h, w = meta["height"], meta["width"]

    # SAM 2 video predictor expects a directory of JPEG frames
    import tempfile
    tmpdir_obj = tempfile.TemporaryDirectory(prefix="sam2_frames_")
    tmpdir = tmpdir_obj.name
    print(f"  Extracting frames to {tmpdir} for SAM 2 ...", file=sys.stderr)

    cmd = [
        ffmpeg_bin, "-v", "error",
        "-i", str(video_path),
        "-q:v", "2",
        f"{tmpdir}/%06d.jpg",
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # Build SAM 2 predictor
    # NOTE: SAM 2 checkpoint must be downloaded separately.
    # Check for local checkpoint first.
    sam2_ckpt_dir = _TOOLS_DIR / "sam3"
    ckpt_candidates = [
        sam2_ckpt_dir / "checkpoints" / "sam2.1_hiera_small.pt",
        sam2_ckpt_dir / "checkpoints" / "sam2_hiera_small.pt",
        Path.home() / ".cache" / "sam2" / "sam2.1_hiera_small.pt",
    ]
    config_candidates = [
        sam2_ckpt_dir / "configs" / "sam2.1" / "sam2.1_hiera_s.yaml",
        sam2_ckpt_dir / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_s.yaml",
    ]

    ckpt_path = None
    for c in ckpt_candidates:
        if c.exists():
            ckpt_path = c
            break

    config_path = None
    for c in config_candidates:
        if c.exists():
            config_path = str(c)
            break

    if ckpt_path is None:
        print("  [warn] SAM 2 checkpoint not found.  Download with:", file=sys.stderr)
        print("    cd workspace/tools/sam3 && bash scripts/download_ckpts.sh", file=sys.stderr)
        print("  Falling back to linear interpolation of keyframe masks.", file=sys.stderr)
        tmpdir_obj.cleanup()
        return _interpolate_masks(keyframe_masks, keyframe_indices, total_frames, h, w)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Loading SAM 2 on {device} ...", file=sys.stderr)

    predictor = build_sam2_video_predictor(
        config_path or "sam2.1_hiera_s",
        str(ckpt_path),
        device=device,
    )

    # Initialize video state
    state = predictor.init_state(video_path=tmpdir)

    # Add keyframe masks as prompts
    for i, kf_idx in enumerate(keyframe_indices):
        mask = keyframe_masks[i]
        # SAM 2 expects mask prompts as (1, H, W) torch tensor
        mask_tensor = torch.from_numpy(mask[None, :, :]).float().to(device)
        predictor.add_new_mask(
            inference_state=state,
            frame_idx=kf_idx,
            obj_id=1,
            mask=mask_tensor,
        )

    # Propagate forward through video
    all_masks = np.zeros((total_frames, h, w), dtype=np.float32)
    print("  Propagating masks with SAM 2 ...", file=sys.stderr)
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        if len(masks) > 0:
            m = masks[0].cpu().numpy().squeeze()
            if m.shape != (h, w):
                from PIL import Image
                m = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize(
                    (w, h), Image.NEAREST
                )).astype(np.float32) / 255.0
            all_masks[frame_idx] = m

    # Cleanup temp directory
    tmpdir_obj.cleanup()

    return all_masks


# ---------------------------------------------------------------------------
# Mask interpolation (fallback when SAM 2 checkpoint not available)
# ---------------------------------------------------------------------------

def _interpolate_masks(
    keyframe_masks: np.ndarray,
    keyframe_indices: list[int],
    total_frames: int,
    h: int,
    w: int,
) -> np.ndarray:
    """Linearly interpolate between keyframe masks to fill all frames."""
    all_masks = np.zeros((total_frames, h, w), dtype=np.float32)

    if len(keyframe_indices) == 0:
        return all_masks

    # Fill keyframes
    for i, kf_idx in enumerate(keyframe_indices):
        if kf_idx < total_frames:
            all_masks[kf_idx] = keyframe_masks[i]

    # Interpolate between keyframes
    for seg_i in range(len(keyframe_indices) - 1):
        start_idx = keyframe_indices[seg_i]
        end_idx = keyframe_indices[seg_i + 1]
        if end_idx <= start_idx:
            continue
        mask_start = keyframe_masks[seg_i]
        mask_end = keyframe_masks[seg_i + 1]
        span = end_idx - start_idx
        for f in range(start_idx + 1, end_idx):
            alpha = (f - start_idx) / span
            all_masks[f] = mask_start * (1.0 - alpha) + mask_end * alpha

    # Extend first keyframe backward
    first_kf = keyframe_indices[0]
    if first_kf > 0:
        all_masks[:first_kf] = keyframe_masks[0]

    # Extend last keyframe forward
    last_kf = keyframe_indices[-1]
    if last_kf < total_frames - 1:
        all_masks[last_kf + 1:] = keyframe_masks[-1]

    return all_masks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_prompts(strategy: str) -> list[str]:
    """Return the text prompts for the chosen strategy."""
    if strategy == "specific":
        return SPECIFIC_PROMPTS
    elif strategy == "general":
        return [GENERAL_PROMPT]
    elif strategy == "hybrid":
        return SPECIFIC_PROMPTS + [GENERAL_PROMPT]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def generate_masks(
    video_path: Path,
    output_path: Path,
    strategy: str = "specific",
    model_choice: str = "falcon",
    sample_step: int = 10,
    feather_radius: float = 24.0,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> Path:
    """Main entry point: generate per-frame importance masks for a video.

    Returns the path to the saved .npy file.
    """
    print(f"=== ML Mask Generator ===", file=sys.stderr)
    print(f"  Input:    {video_path}", file=sys.stderr)
    print(f"  Output:   {output_path}", file=sys.stderr)
    print(f"  Strategy: {strategy}", file=sys.stderr)
    print(f"  Model:    {model_choice}", file=sys.stderr)
    print(f"  Step:     {sample_step}", file=sys.stderr)
    print(f"  Feather:  {feather_radius}", file=sys.stderr)

    # Probe video
    meta = ffprobe_meta(video_path, ffprobe_bin)
    total_frames = meta["total_frames"]
    h, w = meta["height"], meta["width"]
    print(f"  Video:    {w}x{h}, {meta['fps']:.2f}fps, {total_frames} frames",
          file=sys.stderr)

    prompts = build_prompts(strategy)

    # ---- Check which backends are available ----
    _, falcon_ok = _try_import_falcon()
    sam_mod, sam_ver, sam_ok = _try_import_sam()

    use_falcon = model_choice in ("falcon", "both") and falcon_ok
    use_sam = model_choice in ("sam3", "sam2", "both") and sam_ok
    use_fallback = not use_falcon and not use_sam

    if model_choice in ("falcon", "both") and not falcon_ok:
        print("  [warn] Falcon Perception not available, will use fallback",
              file=sys.stderr)
    if model_choice in ("sam3", "sam2", "both") and not sam_ok:
        print("  [warn] SAM 3/2 not available, will use fallback", file=sys.stderr)
    elif sam_ok:
        print(f"  Using {sam_ver} for temporal propagation", file=sys.stderr)

    # ---- Extract sampled frames ----
    print(f"\nExtracting frames (step={sample_step}) ...", file=sys.stderr)
    sampled_frames = extract_frames(video_path, meta, sample_step, ffmpeg_bin)
    n_sampled = sampled_frames.shape[0]
    keyframe_indices = list(range(0, total_frames, sample_step))[:n_sampled]

    # ---- Generate keyframe masks ----
    keyframe_masks = np.zeros((n_sampled, h, w), dtype=np.float32)

    if use_falcon:
        print(f"\nRunning Falcon Perception ({len(prompts)} prompts) ...",
              file=sys.stderr)
        try:
            falcon_masks = run_falcon_on_frames(sampled_frames, prompts)
            keyframe_masks = np.maximum(keyframe_masks, falcon_masks)
        except Exception as e:
            print(f"  [error] Falcon Perception failed: {e}", file=sys.stderr)
            print("  Falling back to gradient+temporal masking", file=sys.stderr)
            use_fallback = True

    if use_fallback or (not use_falcon and not use_sam):
        print("\nUsing fallback gradient+temporal masking ...", file=sys.stderr)
        prev_frame = None
        for i in range(n_sampled):
            keyframe_masks[i] = fallback_adaptive_mask(
                sampled_frames[i], prev_frame, feather_radius,
            )
            prev_frame = sampled_frames[i]

    # ---- Feather keyframe masks ----
    if feather_radius > 0:
        print(f"\nFeathering masks (radius={feather_radius}) ...", file=sys.stderr)
        for i in range(n_sampled):
            keyframe_masks[i] = feather_mask(keyframe_masks[i], feather_radius)

    # ---- Propagate / interpolate to all frames ----
    if use_sam and not use_fallback:
        print(f"\nPropagating with {sam_ver} video predictor ...", file=sys.stderr)
        try:
            all_masks = run_sam2_propagation(
                video_path, keyframe_masks, keyframe_indices,
                total_frames, meta, ffmpeg_bin=ffmpeg_bin,
            )
        except Exception as e:
            print(f"  [error] {sam_ver} propagation failed: {e}", file=sys.stderr)
            print("  Falling back to linear interpolation", file=sys.stderr)
            all_masks = _interpolate_masks(
                keyframe_masks, keyframe_indices, total_frames, h, w,
            )
    else:
        print("\nInterpolating masks to all frames ...", file=sys.stderr)
        all_masks = _interpolate_masks(
            keyframe_masks, keyframe_indices, total_frames, h, w,
        )

    # ---- Ensure [0, 1] range ----
    all_masks = np.clip(all_masks, 0.0, 1.0)

    # ---- Save ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), all_masks)
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved masks: {output_path}", file=sys.stderr)
    print(f"  Shape: {all_masks.shape}  dtype: {all_masks.dtype}", file=sys.stderr)
    print(f"  Size:  {file_size:.1f} MB", file=sys.stderr)
    print(f"  Range: [{all_masks.min():.4f}, {all_masks.max():.4f}]", file=sys.stderr)
    nonzero_pct = 100.0 * (all_masks > 0.1).mean()
    print(f"  Coverage (>0.1): {nonzero_pct:.1f}%", file=sys.stderr)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate per-frame importance masks using ML segmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use Falcon Perception with specific driving prompts
  python ml_mask_generator.py --input video.mp4 --output masks.npy --strategy specific --model falcon

  # Use SAM 2 for temporal propagation on top of Falcon keyframes
  python ml_mask_generator.py --input video.mp4 --output masks.npy --model both --sample-step 15

  # Fallback mode (no ML, gradient+temporal only)
  python ml_mask_generator.py --input video.mp4 --output masks.npy --model falcon --sample-step 5
  # (automatically falls back if Falcon is not importable)
""",
    )
    p.add_argument("--input", required=True, type=Path,
                    help="Input video path")
    p.add_argument("--output", required=True, type=Path,
                    help="Output .npy mask file path")
    p.add_argument("--strategy", default="specific",
                    choices=["specific", "general", "hybrid"],
                    help="Prompting strategy (default: specific)")
    p.add_argument("--model", default="falcon",
                    choices=["falcon", "sam3", "sam2", "both"],
                    help="Which ML model(s) to use (default: falcon)")
    p.add_argument("--sample-step", type=int, default=10,
                    help="Process every Nth frame, interpolate between (default: 10)")
    p.add_argument("--feather-radius", type=float, default=24.0,
                    help="Gaussian feather radius for mask edges (default: 24)")
    p.add_argument("--ffmpeg-bin", default="ffmpeg")
    p.add_argument("--ffprobe-bin", default="ffprobe")
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        generate_masks(
            video_path=args.input,
            output_path=args.output,
            strategy=args.strategy,
            model_choice=args.model,
            sample_step=args.sample_step,
            feather_radius=args.feather_radius,
            ffmpeg_bin=args.ffmpeg_bin,
            ffprobe_bin=args.ffprobe_bin,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
