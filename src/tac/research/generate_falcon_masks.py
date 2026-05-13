#!/usr/bin/env python3
"""Generate Falcon Perception segmentation masks for dashcam video keyframes.

Extracts every 20th frame (60 keyframes from 1200 frames), runs Falcon Perception
with driving-relevant prompts, and combines masks into importance maps.

Output: reports/masks/falcon_keyframes.npy  shape (60, 874, 1164) float32
"""

import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
VIDEO = ROOT / "workspace/upstream/comma_video_compression_challenge/videos/0.mkv"
OUT_DIR = ROOT / "reports/masks"
FRAMES_DIR = OUT_DIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

N_FRAMES = 1200
STEP = 20
N_KEYFRAMES = N_FRAMES // STEP  # 60
H, W = 874, 1164

# Focused prompts: high-value driving content + sky (to mark as deprioritized)
# Based on single-frame profiling:
#   road:     22.9% cov, 17s   -- key foreground
#   car:       1.6% cov,  9.6s -- important objects
#   sky:      35.0% cov,  4.2s -- background to deprioritize
#   general:  23.4% cov,  2.2s -- catches misc important things
FOREGROUND_PROMPTS = ["road", "car", "everything important for driving a car"]
BACKGROUND_PROMPTS = ["sky"]


def extract_keyframes():
    """Extract every 20th frame from the video."""
    existing = list(FRAMES_DIR.glob("frame_*.png"))
    if len(existing) >= N_KEYFRAMES:
        print(f"  {len(existing)} keyframes already extracted, skipping.")
        return
    print(f"Extracting {N_KEYFRAMES} keyframes (every {STEP}th frame)...")
    for i in range(N_KEYFRAMES):
        frame_idx = i * STEP
        out_path = FRAMES_DIR / f"frame_{i:04d}.png"
        if out_path.exists():
            continue
        subprocess.run([
            "ffmpeg", "-v", "error",
            "-i", str(VIDEO),
            "-vf", f"select=eq(n\\,{frame_idx})",
            "-vframes", "1",
            "-f", "image2",
            str(out_path),
        ], check=True)
    print(f"  Keyframes extracted to {FRAMES_DIR}")


def decode_rle_mask(rle: dict, h: int, w: int) -> np.ndarray | None:
    """Decode RLE mask to binary array."""
    try:
        from pycocotools import mask as mask_utils
        m = mask_utils.decode(rle).astype(np.uint8)
        if m.shape != (h, w):
            m = np.array(Image.fromarray(m).resize((w, h), Image.NEAREST))
        return m
    except Exception as e:
        print(f"  Warning: failed to decode RLE mask: {e}")
        return None


def run_falcon_on_frame(engine, tokenizer, model_args, pil_image, prompt, task="segmentation"):
    """Run Falcon Perception on a single frame with a given prompt."""
    from falcon_perception import build_prompt_for_task
    from falcon_perception.mlx.batch_inference import process_batch_and_generate

    prompt_text = build_prompt_for_task(prompt, task)
    batch = process_batch_and_generate(
        tokenizer,
        [(pil_image, prompt_text)],
        max_length=model_args.max_seq_len,
        min_dimension=256,
        max_dimension=1024,
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
    w_img, h_img = pil_image.size
    masks = []
    for rle in aux.masks_rle:
        m = decode_rle_mask(rle, h_img, w_img)
        if m is not None:
            masks.append(m)
    return masks, len(aux.masks_rle)


def main():
    t_start = time.time()

    # Step 1: Extract keyframes
    extract_keyframes()

    # Step 2: Load Falcon Perception model
    print("\nLoading Falcon Perception (MLX)...")
    from falcon_perception import PERCEPTION_MODEL_ID, load_and_prepare_model
    from falcon_perception.mlx.batch_inference import BatchInferenceEngine

    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=PERCEPTION_MODEL_ID,
        dtype="float16",
        backend="mlx",
    )
    engine = BatchInferenceEngine(model, tokenizer)
    print(f"  Model loaded in {time.time() - t_start:.1f}s")

    # Step 3: Generate masks for each keyframe
    # foreground_maps: union of all foreground detections (protect these)
    # background_maps: sky etc (compress these more)
    foreground_maps = np.zeros((N_KEYFRAMES, H, W), dtype=np.float32)
    background_maps = np.zeros((N_KEYFRAMES, H, W), dtype=np.float32)

    # Per-prompt coverage tracking
    all_prompts = FOREGROUND_PROMPTS + BACKGROUND_PROMPTS
    prompt_coverage = {p: [] for p in all_prompts}
    frame_fg_coverage = []
    frame_bg_coverage = []

    for ki in range(N_KEYFRAMES):
        frame_path = FRAMES_DIR / f"frame_{ki:04d}.png"
        pil_image = Image.open(frame_path).convert("RGB")
        assert pil_image.size == (W, H), f"Unexpected size: {pil_image.size}"

        fg_mask = np.zeros((H, W), dtype=np.float32)
        bg_mask = np.zeros((H, W), dtype=np.float32)
        t_frame = time.time()

        # Foreground prompts
        for prompt in FOREGROUND_PROMPTS:
            masks, n = run_falcon_on_frame(engine, tokenizer, model_args, pil_image, prompt)
            prompt_mask = np.zeros((H, W), dtype=np.float32)
            for m in masks:
                prompt_mask = np.maximum(prompt_mask, m.astype(np.float32))
            fg_mask = np.maximum(fg_mask, prompt_mask)
            cov = prompt_mask.mean() * 100
            prompt_coverage[prompt].append(cov)

        # Background prompts
        for prompt in BACKGROUND_PROMPTS:
            masks, n = run_falcon_on_frame(engine, tokenizer, model_args, pil_image, prompt)
            prompt_mask = np.zeros((H, W), dtype=np.float32)
            for m in masks:
                prompt_mask = np.maximum(prompt_mask, m.astype(np.float32))
            bg_mask = np.maximum(bg_mask, prompt_mask)
            cov = prompt_mask.mean() * 100
            prompt_coverage[prompt].append(cov)

        foreground_maps[ki] = fg_mask
        background_maps[ki] = bg_mask
        fc = fg_mask.mean() * 100
        bc = bg_mask.mean() * 100
        frame_fg_coverage.append(fc)
        frame_bg_coverage.append(bc)

        dt = time.time() - t_frame
        # Report every frame for progress tracking
        print(f"  [{ki:2d}/{N_KEYFRAMES}] fg={fc:.1f}% bg={bc:.1f}% ({dt:.1f}s)", flush=True)

    # Step 4: Build final importance map
    # Importance = foreground union, with sky regions reduced
    # Where foreground and background overlap, foreground wins
    importance = foreground_maps.copy()
    # Where there's no foreground but there IS background (sky), mark as low importance (0.2)
    sky_only = (background_maps > 0.5) & (foreground_maps < 0.5)
    importance[sky_only] = 0.2
    # Where nothing is detected, use neutral importance (0.5)
    nothing = (foreground_maps < 0.5) & (background_maps < 0.5)
    importance[nothing] = 0.5

    # Save both the raw importance map and the foreground/background layers
    out_path = OUT_DIR / "falcon_keyframes.npy"
    np.save(str(out_path), importance)
    print(f"\nSaved importance maps to {out_path}")
    print(f"  Shape: {importance.shape}")
    print(f"  Dtype: {importance.dtype}")

    # Also save the raw layers for later use
    np.save(str(OUT_DIR / "falcon_foreground.npy"), foreground_maps)
    np.save(str(OUT_DIR / "falcon_background.npy"), background_maps)

    # Step 5: Report metrics
    print("\n" + "=" * 60)
    print("  MASK QUALITY REPORT")
    print("=" * 60)

    fg_arr = np.array(frame_fg_coverage)
    bg_arr = np.array(frame_bg_coverage)

    print("\n  Foreground coverage (protected regions):")
    print(f"    Mean:   {fg_arr.mean():.1f}%")
    print(f"    Median: {np.median(fg_arr):.1f}%")
    print(f"    Min:    {fg_arr.min():.1f}%")
    print(f"    Max:    {fg_arr.max():.1f}%")
    print(f"    Std:    {fg_arr.std():.1f}%")

    print("\n  Background coverage (deprioritized):")
    print(f"    Mean:   {bg_arr.mean():.1f}%")
    print(f"    Median: {np.median(bg_arr):.1f}%")
    print(f"    Min:    {bg_arr.min():.1f}%")
    print(f"    Max:    {bg_arr.max():.1f}%")

    print("\n  Static trapezoid comparison: 44% coverage")
    diff = fg_arr.mean() - 44.0
    print(f"    Falcon fg vs trapezoid: {diff:+.1f}% ({'more' if diff > 0 else 'less'} coverage)")

    # Importance map stats
    protected = (importance > 0.5).mean() * 100
    deprioritized = (importance < 0.3).mean() * 100
    neutral = 100 - protected - deprioritized
    print("\n  Importance map distribution:")
    print(f"    Protected (>0.5):     {protected:.1f}%")
    print(f"    Neutral (0.3-0.5):    {neutral:.1f}%")
    print(f"    Deprioritized (<0.3): {deprioritized:.1f}%")

    print("\n  Per-prompt mean coverage:")
    for prompt, covs in prompt_coverage.items():
        if covs:
            arr = np.array(covs)
            label = "FG" if prompt in FOREGROUND_PROMPTS else "BG"
            print(f"    [{label}] {prompt:40s}: {arr.mean():.1f}% (std {arr.std():.1f}%)")

    # Quality checks
    print("\n  Quality checks:")
    empty_fg = np.sum(fg_arr < 5.0)
    print(f"    Frames with <5% fg coverage: {empty_fg}/{N_KEYFRAMES}")
    saturated_fg = np.sum(fg_arr > 80.0)
    print(f"    Frames with >80% fg coverage: {saturated_fg}/{N_KEYFRAMES} (possible errors)")
    no_bg = np.sum(bg_arr < 1.0)
    print(f"    Frames with <1% bg (no sky): {no_bg}/{N_KEYFRAMES}")

    total_time = time.time() - t_start
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Per-frame: {total_time/N_KEYFRAMES:.1f}s")
    print("=" * 60)

    # Save visualization previews
    try:
        for ki in [0, 15, 30, 45, 59]:
            # Color-coded: green=protected, red=deprioritized, gray=neutral
            vis = np.zeros((H, W, 3), dtype=np.uint8)
            m = importance[ki]
            vis[m > 0.5] = [0, 200, 0]      # green = protected foreground
            vis[m < 0.3] = [200, 50, 50]     # red = deprioritized background
            vis[(m >= 0.3) & (m <= 0.5)] = [128, 128, 128]  # gray = neutral
            Image.fromarray(vis).save(str(OUT_DIR / f"mask_preview_{ki:04d}.png"))

            # Also save raw importance as grayscale
            gray = (m * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(gray).save(str(OUT_DIR / f"importance_{ki:04d}.png"))
        print(f"\n  Saved mask previews to {OUT_DIR}/mask_preview_*.png")
        print(f"  Saved importance maps to {OUT_DIR}/importance_*.png")
    except Exception as e:
        print(f"  Warning: could not save previews: {e}")


if __name__ == "__main__":
    main()
