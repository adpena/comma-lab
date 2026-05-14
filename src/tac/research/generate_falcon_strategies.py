#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate three strategy-specific Falcon Perception mask sets.

Strategies:
  - pose_focused: prompts targeted at PoseNet structural features
  - seg_focused : prompts targeted at SegNet semantic classes
  - combined    : union of pose_focused and seg_focused

Uses already-extracted keyframes in reports/masks/frames/.
Saves importance arrays of shape (60, 874, 1164) float32 in [0,1].
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "reports/masks"
FRAMES_DIR = OUT_DIR / "frames"

N_KEYFRAMES = 60
H, W = 874, 1164

POSE_FG = [
    "the road surface and lane markings",
    "the horizon line",
    "buildings and structures with sharp edges",
    "things needed for camera pose estimation",
    "perspective vanishing points",
]
POSE_BG = ["uniform sky", "uniform pavement"]

SEG_FG = [
    "vehicles",
    "pedestrians",
    "road signs",
    "traffic lights",
    "lane markings",
    "road boundaries",
]
SEG_BG = ["sky", "uniform background"]


def decode_rle_mask(rle: dict, h: int, w: int) -> np.ndarray | None:
    try:
        from pycocotools import mask as mask_utils

        m = mask_utils.decode(rle).astype(np.uint8)
        if m.shape != (h, w):
            m = np.array(Image.fromarray(m).resize((w, h), Image.NEAREST))
        return m
    except Exception as e:
        print(f"  warn: decode_rle failed: {e}")
        return None


def run_falcon(engine, tokenizer, model_args, pil_image, prompt, task="segmentation"):
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
    _tokens, aux_outputs = engine.generate(
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
    return masks


def union_masks(engine, tokenizer, model_args, pil_image, prompts, cache):
    """Return union mask over a list of prompts, using a per-frame prompt cache."""
    out = np.zeros((H, W), dtype=np.float32)
    for p in prompts:
        if p in cache:
            pm = cache[p]
        else:
            ms = run_falcon(engine, tokenizer, model_args, pil_image, p)
            pm = np.zeros((H, W), dtype=np.float32)
            for m in ms:
                pm = np.maximum(pm, m.astype(np.float32))
            cache[p] = pm
        out = np.maximum(out, pm)
    return out


def to_importance(fg: np.ndarray, bg: np.ndarray) -> np.ndarray:
    imp = fg.copy()
    sky_only = (bg > 0.5) & (fg < 0.5)
    imp[sky_only] = 0.2
    nothing = (fg < 0.5) & (bg < 0.5)
    imp[nothing] = 0.5
    return imp


def main():
    t0 = time.time()
    print("Loading Falcon Perception (MLX)...")
    from falcon_perception import PERCEPTION_MODEL_ID, load_and_prepare_model
    from falcon_perception.mlx.batch_inference import BatchInferenceEngine

    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=PERCEPTION_MODEL_ID,
        dtype="float16",
        backend="mlx",
    )
    engine = BatchInferenceEngine(model, tokenizer)
    print(f"  loaded in {time.time() - t0:.1f}s")

    pose_imp = np.zeros((N_KEYFRAMES, H, W), dtype=np.float32)
    seg_imp = np.zeros((N_KEYFRAMES, H, W), dtype=np.float32)
    comb_imp = np.zeros((N_KEYFRAMES, H, W), dtype=np.float32)

    cov = {
        "pose_fg": [], "pose_bg": [],
        "seg_fg":  [], "seg_bg":  [],
        "comb_fg": [], "comb_bg": [],
    }
    per_prompt = {p: [] for p in POSE_FG + POSE_BG + SEG_FG + SEG_BG}

    for ki in range(N_KEYFRAMES):
        frame_path = FRAMES_DIR / f"frame_{ki:04d}.png"
        pil = Image.open(frame_path).convert("RGB")
        assert pil.size == (W, H)
        cache: dict[str, np.ndarray] = {}
        tf = time.time()

        # Pose
        pose_fg_mask = union_masks(engine, tokenizer, model_args, pil, POSE_FG, cache)
        pose_bg_mask = union_masks(engine, tokenizer, model_args, pil, POSE_BG, cache)
        # Seg
        seg_fg_mask = union_masks(engine, tokenizer, model_args, pil, SEG_FG, cache)
        seg_bg_mask = union_masks(engine, tokenizer, model_args, pil, SEG_BG, cache)
        # Combined
        comb_fg_mask = np.maximum(pose_fg_mask, seg_fg_mask)
        # Combined background = only where BOTH say background, AND no foreground
        comb_bg_mask = np.minimum(pose_bg_mask, seg_bg_mask)

        pose_imp[ki] = to_importance(pose_fg_mask, pose_bg_mask)
        seg_imp[ki] = to_importance(seg_fg_mask, seg_bg_mask)
        comb_imp[ki] = to_importance(comb_fg_mask, comb_bg_mask)

        cov["pose_fg"].append(pose_fg_mask.mean() * 100)
        cov["pose_bg"].append(pose_bg_mask.mean() * 100)
        cov["seg_fg"].append(seg_fg_mask.mean() * 100)
        cov["seg_bg"].append(seg_bg_mask.mean() * 100)
        cov["comb_fg"].append(comb_fg_mask.mean() * 100)
        cov["comb_bg"].append(comb_bg_mask.mean() * 100)
        for p, m in cache.items():
            per_prompt[p].append(m.mean() * 100)

        dt = time.time() - tf
        print(
            f"  [{ki:2d}/{N_KEYFRAMES}] "
            f"pose fg/bg={cov['pose_fg'][-1]:.1f}/{cov['pose_bg'][-1]:.1f} "
            f"seg fg/bg={cov['seg_fg'][-1]:.1f}/{cov['seg_bg'][-1]:.1f} "
            f"comb fg/bg={cov['comb_fg'][-1]:.1f}/{cov['comb_bg'][-1]:.1f} "
            f"({dt:.1f}s)",
            flush=True,
        )

    np.save(str(OUT_DIR / "falcon_pose_focused.npy"), pose_imp)
    np.save(str(OUT_DIR / "falcon_seg_focused.npy"), seg_imp)
    np.save(str(OUT_DIR / "falcon_combined.npy"), comb_imp)

    print("\n" + "=" * 60)
    print("  STRATEGY MASK REPORT")
    print("=" * 60)
    for label in ("pose", "seg", "comb"):
        fg = np.array(cov[f"{label}_fg"])
        bg = np.array(cov[f"{label}_bg"])
        print(f"\n  {label}_focused:")
        print(f"    fg mean={fg.mean():.1f}% median={np.median(fg):.1f}% min={fg.min():.1f}% max={fg.max():.1f}%")
        print(f"    bg mean={bg.mean():.1f}% median={np.median(bg):.1f}% min={bg.min():.1f}% max={bg.max():.1f}%")

    for name, arr in (
        ("pose_focused", pose_imp),
        ("seg_focused", seg_imp),
        ("combined", comb_imp),
    ):
        protected = (arr > 0.5).mean() * 100
        deprio = (arr < 0.3).mean() * 100
        neutral = 100 - protected - deprio
        print(f"\n  {name} importance distribution:")
        print(f"    protected (>0.5):     {protected:.1f}%")
        print(f"    neutral   (0.3-0.5):  {neutral:.1f}%")
        print(f"    deprio    (<0.3):     {deprio:.1f}%")

    print("\n  Per-prompt mean coverage:")
    for p, covs in per_prompt.items():
        if covs:
            arr = np.array(covs)
            print(f"    {p:55s}: {arr.mean():5.1f}% (std {arr.std():.1f}%)")

    print(f"\n  Total time: {time.time() - t0:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
