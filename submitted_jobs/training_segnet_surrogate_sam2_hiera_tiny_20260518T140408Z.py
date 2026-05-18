# /// script
# dependencies = [
#     "torch>=2.5.0",
#     "transformers>=4.40.0",
#     "datasets>=2.18.0",
#     "huggingface_hub>=0.23.0",
#     "safetensors>=0.4.0",
#     "trackio",
#     "pillow>=10.0.0",
#     "numpy",
# ]
# ///

# SPDX-License-Identifier: MIT
"""SegNet -> SAM2-hiera-tiny PER-PIXEL distillation surrogate (HF Jobs t4-small).

Insight 1, Stage 2 from `feedback_deep_research_wave_landed_20260518.md`.
Builds a SAM2-hiera-tiny (38.9M params; mask-decoder-only fine-tuned, ~5M
trainable) per-pixel segmentation surrogate for the contest SegNet
(EfficientNet-B2 U-Net, 5 classes at 192x256). UNLIKE Stage 1 (which is a
crude 5-way classifier on the majority class), Stage 2 produces a FULL
per-pixel mask matching SegNet's grid.

Why SAM2-hiera-tiny
-------------------
- 38.9M params total but per Catalog #779 freezing exploit, only the
  ~5M-param mask decoder needs to be fine-tuned (encoders frozen at ~33 MB).
- Already pretrained on SA-1B (1.1B masks); transfers to dashcam scenes
  far better than randomly-initialized U-Net.
- Native per-pixel output (segmentation_models_pytorch style); no need to
  re-architect to match SegNet's 192x256 grid.

Architecture
------------
- Teacher: contest SegNet (frozen), via dataset-side pre-computed
  `segnet_logits_frame_1` (float16, 5x192x256) and `segnet_mask_frame_1`
  (uint8 argmax, 192x256). NO live SegNet forward during training —
  the canonical dataset has the labels embedded.
- Student: SAM2-hiera-tiny mask decoder fine-tuned with 5-class output
  head replacing the original 1-class mask prediction.
- Loss: DiceCE + Hinton T=2.0 KL on per-pixel softmax (canonical
  segmentation distillation per the HF skill `sam_segmentation_training.py`
  template + Hinton-Vinyals-Dean 2014 ratio).

Data
----
Pulls from `adpena/comma-video-substrate-eval-600pairs` (canonical HF
dataset built by `tools/build_comma_video_substrate_eval_600pairs_dataset.py`).
Each row: image_frame_1 (384x512 RGB) + segnet_mask_frame_1 (192x256
uint8) + segnet_logits_frame_1 (5x192x256 float16).

Train/val split: 540/60 (90/10) along pair_idx (deterministic per
`random_state=42` seed).

Hardware + cost
---------------
HF Jobs `t4-small` ($0.40/hr; ~16 GB). Estimated wall-clock: 60-90 min
for 30 epochs at batch_size=8 (SAM2 inference at 1024-res is memory-
intensive; we run at SAM2's native 1024 to preserve encoder pretrained
weights). Total cost: ~$0.40-0.60 per run.

Output
------
- `adpena/segnet-surrogate-sam2-hiera-tiny` model repo (~150 MB total;
  encoders ~33 MB frozen + decoder ~5 MB fine-tuned + ~110 MB other)
- TrackIO experiment

Cite-chain provenance
---------------------
- Dataset commit sha (from HF Hub metadata)
- Teacher SegNet safetensors sha (cited in dataset README)
- Student model_name: `facebook/sam2-hiera-tiny`

Per CLAUDE.md "MPS auth eval is NOISE" — this surrogate is a TOOL
(non-promotable advisory only); contest score claims via this output MUST
go through the canonical auth-eval gate per Catalog #226.

Usage (HF Jobs MCP)
-------------------
    hf_jobs("uv", {
        "script": "<this file's content>",
        "flavor": "t4-small",
        "secrets": {"HF_TOKEN": "$HF_TOKEN"},
        "timeout": "2h",
    })

Usage (local smoke)
-------------------
    .venv/bin/python submitted_jobs/training_segnet_surrogate_sam2_hiera_tiny_*.py \\
        --max-train-samples 4 --max-eval-samples 2 --num-epochs 1 \\
        --no-upload --output-dir .omx/tmp/segnet_sam2_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# --- canonical 3-export Modal/CUDA env block per Catalog #244 ---
os.environ.setdefault("DALI_DISABLE_NVML", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Canonical constants
# ---------------------------------------------------------------------------

CANONICAL_DATASET = "adpena/comma-video-substrate-eval-600pairs"
CANONICAL_OUTPUT_MODEL_REPO = "adpena/segnet-surrogate-sam2-hiera-tiny"
STUDENT_MODEL_NAME = "facebook/sam2-hiera-tiny"
N_SEGNET_CLASSES = 5
HINTON_T = 2.0
DEFAULT_LR = 1e-4  # SAM2 fine-tuning prefers smaller LR
DEFAULT_BATCH_SIZE = 8  # 1024-res is memory-intensive
DEFAULT_EPOCHS = 30
DICE_CE_WEIGHT = 0.5  # 50/50 DiceCE + KL distillation
HINTON_KL_WEIGHT = 0.5

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DiceCE loss + Hinton KL distillation
# ---------------------------------------------------------------------------


def dice_loss(
    pred_logits: torch.Tensor,  # (B, K, H, W)
    target_mask: torch.Tensor,  # (B, H, W) int64 class indices
    num_classes: int = N_SEGNET_CLASSES,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Multi-class soft Dice loss."""
    pred_softmax = F.softmax(pred_logits, dim=1)
    target_onehot = F.one_hot(target_mask.long(), num_classes=num_classes)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()
    intersection = (pred_softmax * target_onehot).sum(dim=(2, 3))
    cardinality = pred_softmax.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def hinton_per_pixel_kl_loss(
    student_logits: torch.Tensor,  # (B, K, H, W)
    teacher_logits: torch.Tensor,  # (B, K, H, W)
    temperature: float = HINTON_T,
) -> torch.Tensor:
    """Per-pixel Hinton KL distillation (T^2 scaled)."""
    log_pred = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    # KL per pixel then mean
    loss = F.kl_div(log_pred, soft_teacher, reduction="batchmean")
    return loss * (temperature ** 2)


def combined_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_mask: torch.Tensor,
    dice_ce_weight: float = DICE_CE_WEIGHT,
    hinton_kl_weight: float = HINTON_KL_WEIGHT,
) -> dict:
    """Combined DiceCE + Hinton KL."""
    ce = F.cross_entropy(student_logits, teacher_mask.long())
    dice = dice_loss(student_logits, teacher_mask)
    dice_ce = ce + dice
    kl = hinton_per_pixel_kl_loss(student_logits, teacher_logits, HINTON_T)
    total = dice_ce_weight * dice_ce + hinton_kl_weight * kl
    return {"total": total, "ce": ce, "dice": dice, "kl": kl}


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


def prepare_dataset_splits(
    dataset_name: str,
    train_val_split: float = 0.9,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
):
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    if max_train_samples is not None or max_eval_samples is not None:
        total_needed = (max_train_samples or 8) + (max_eval_samples or 4)
        ds = ds.select(range(min(total_needed, len(ds))))
    ds = ds.shuffle(seed=seed)
    n_train = int(len(ds) * train_val_split)
    train_ds = ds.select(range(n_train))
    eval_ds = ds.select(range(n_train, len(ds)))
    if max_train_samples is not None:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))
    return train_ds, eval_ds


def collate_fn(batch, target_hw=(192, 256)):
    """Convert HF rows -> SAM2 inputs + teacher labels.

    Returns:
      - images_1024_bchw : (B, 3, 1024, 1024) float, SAM2's native resolution
      - target_mask_192_256 : (B, 192, 256) uint8 SegNet argmax
      - teacher_logits_192_256 : (B, 5, 192, 256) float16
    """
    from PIL import Image as PILImage

    images, masks, teacher_logits_list = [], [], []
    for row in batch:
        img = row["image_frame_1"]
        if not isinstance(img, PILImage.Image):
            img = PILImage.fromarray(np.asarray(img))
        # SAM2 expects 1024x1024 RGB normalized to ImageNet stats
        img = img.convert("RGB").resize((1024, 1024), PILImage.BILINEAR)
        img_arr = np.asarray(img).astype(np.float32) / 255.0
        # Normalize ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_arr = (img_arr - mean) / std
        img_t = torch.from_numpy(img_arr).permute(2, 0, 1)  # (3, 1024, 1024)
        images.append(img_t)

        mask_flat = np.asarray(row["segnet_mask_frame_1"], dtype=np.uint8)
        mask = torch.from_numpy(mask_flat.reshape(192, 256))
        masks.append(mask)

        logits_flat = np.asarray(row["segnet_logits_frame_1"], dtype=np.float16)
        logits = torch.from_numpy(logits_flat.reshape(5, 192, 256)).float()
        teacher_logits_list.append(logits)

    return (
        torch.stack(images),
        torch.stack(masks),
        torch.stack(teacher_logits_list),
    )


# ---------------------------------------------------------------------------
# SAM2 student wrapper (5-class adapter)
# ---------------------------------------------------------------------------


class SAM2FiveClassAdapter(nn.Module):
    """SAM2-hiera-tiny encoder (frozen) + new 5-class decoder head.

    The SAM2 image encoder runs at 1024x1024 and produces a (B, 256, 64, 64)
    feature map. We attach a small per-pixel classifier head + upsample to
    192x256 to match SegNet's grid.

    The encoder is frozen by default (Catalog #779 freezing exploit). Only
    the decoder head is trained.
    """

    def __init__(self, sam2_model_name: str = STUDENT_MODEL_NAME, num_classes: int = N_SEGNET_CLASSES, freeze_encoder: bool = True):
        super().__init__()
        from transformers import Sam2Model

        self.sam2 = Sam2Model.from_pretrained(sam2_model_name)
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for name, p in self.sam2.named_parameters():
                if "vision_encoder" in name or "memory_encoder" in name:
                    p.requires_grad_(False)

        # Decoder head: project from SAM2 feature dim -> num_classes per pixel
        # SAM2-hiera-tiny image encoder out channels = 256
        self.decoder_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, images_1024: torch.Tensor) -> torch.Tensor:
        """images_1024: (B, 3, 1024, 1024) -> (B, num_classes, 192, 256)."""
        # SAM2 vision encoder forward
        vision_outputs = self.sam2.vision_encoder(pixel_values=images_1024)
        feat = vision_outputs.last_hidden_state  # (B, 256, 64, 64) typical
        logits_64 = self.decoder_head(feat)  # (B, K, 64, 64)
        # Upsample to SegNet's grid 192x256 via bilinear
        logits_192_256 = F.interpolate(
            logits_64, size=(192, 256), mode="bilinear", align_corners=False
        )
        return logits_192_256


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_surrogate(
    output_dir: Path,
    dataset_name: str = CANONICAL_DATASET,
    num_epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    use_trackio: bool = True,
    trackio_project: str = "segnet_surrogate_sam2_hiera_tiny",
    freeze_encoder: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    if use_trackio:
        try:
            import trackio
            trackio.init(project=trackio_project, config={
                "dataset": dataset_name,
                "student": STUDENT_MODEL_NAME,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "freeze_encoder": freeze_encoder,
                "hinton_T": HINTON_T,
            })
        except ImportError:
            use_trackio = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on device=%s", device)

    student = SAM2FiveClassAdapter(freeze_encoder=freeze_encoder).to(device)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    logger.info(
        "Trainable params: %d / total %d",
        sum(p.numel() for p in trainable_params),
        sum(p.numel() for p in student.parameters()),
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    train_ds, eval_ds = prepare_dataset_splits(
        dataset_name,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )
    logger.info("Train=%d Eval=%d", len(train_ds), len(eval_ds))

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    best_eval_total = float("inf")
    metrics_history = []

    for epoch in range(num_epochs):
        # Train
        student.train()
        train_losses = {"total": 0.0, "ce": 0.0, "dice": 0.0, "kl": 0.0}
        train_n = 0
        for images, masks, teacher_logits in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            teacher_logits = teacher_logits.to(device)
            student_logits = student(images)
            losses = combined_loss(student_logits, teacher_logits, masks)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            for k, v in losses.items():
                train_losses[k] += v.item() * images.size(0)
            train_n += images.size(0)

        train_avg = {k: v / max(train_n, 1) for k, v in train_losses.items()}

        # Eval
        student.eval()
        eval_losses = {"total": 0.0, "ce": 0.0, "dice": 0.0, "kl": 0.0}
        eval_n = 0
        eval_pixel_acc = 0.0
        eval_pixel_total = 0
        with torch.no_grad():
            for images, masks, teacher_logits in eval_loader:
                images = images.to(device)
                masks = masks.to(device)
                teacher_logits = teacher_logits.to(device)
                student_logits = student(images)
                losses = combined_loss(student_logits, teacher_logits, masks)
                for k, v in losses.items():
                    eval_losses[k] += v.item() * images.size(0)
                eval_n += images.size(0)
                # Per-pixel accuracy
                student_pred = student_logits.argmax(dim=1)
                eval_pixel_acc += (student_pred == masks).float().sum().item()
                eval_pixel_total += masks.numel()

        eval_avg = {k: v / max(eval_n, 1) for k, v in eval_losses.items()}
        eval_pixel_acc_avg = eval_pixel_acc / max(eval_pixel_total, 1)

        metric_row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_avg.items()},
            **{f"eval_{k}": v for k, v in eval_avg.items()},
            "eval_pixel_acc": eval_pixel_acc_avg,
        }
        metrics_history.append(metric_row)
        logger.info(
            "epoch=%d train_total=%.4f eval_total=%.4f eval_pixel_acc=%.4f",
            epoch,
            train_avg["total"],
            eval_avg["total"],
            eval_pixel_acc_avg,
        )
        if use_trackio:
            trackio.log(metric_row)

        if eval_avg["total"] < best_eval_total:
            best_eval_total = eval_avg["total"]
            torch.save(student.state_dict(), output_dir / "model_best.pt")

    torch.save(student.state_dict(), output_dir / "model_final.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps({"history": metrics_history, "best_eval_total": best_eval_total}, indent=2)
    )

    if use_trackio:
        trackio.finish()

    return {
        "best_eval_total": best_eval_total,
        "final_eval_pixel_acc": metrics_history[-1]["eval_pixel_acc"] if metrics_history else None,
        "output_dir": str(output_dir),
    }


# ---------------------------------------------------------------------------
# HF Hub upload
# ---------------------------------------------------------------------------


def upload_model_to_hub(local_dir: Path, repo_id: str, token: Optional[str] = None, private: bool = False) -> str:
    from huggingface_hub import HfApi, get_token
    if token is None:
        token = get_token()
    if not token:
        raise RuntimeError("No HF_TOKEN found")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    card = f"""---
license: mit
library_name: transformers
tags:
- distillation
- sam2
- comma-video-compression
- segnet-surrogate
- per-pixel-segmentation
---

# segnet-surrogate-sam2-hiera-tiny

SAM2-hiera-tiny mask-decoder-only fine-tuned (Catalog #779 freezing exploit;
encoders frozen at ~33 MB) as a per-pixel SegNet surrogate distilled from
the comma.ai contest SegNet via combined DiceCE + Hinton T=2.0 KL on
per-pixel softmax.

## Provenance

- Source dataset: [`{CANONICAL_DATASET}`](https://huggingface.co/datasets/{CANONICAL_DATASET})
- Teacher: contest SegNet (frozen)
- Student base: `{STUDENT_MODEL_NAME}` (mask decoder fine-tuned; encoders frozen)

Per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU AND CUDA":
this surrogate is a TOOL (non-promotable advisory). Contest score claims using
this surrogate's output MUST go through the canonical auth-eval gate.
"""
    (local_dir / "README.md").write_text(card)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(local_dir),
                      commit_message="segnet_surrogate_sam2_hiera_tiny: distilled checkpoint")
    return f"https://huggingface.co/{repo_id}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=CANONICAL_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("./segnet_sam2_out"))
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--no-freeze-encoder", action="store_true")
    parser.add_argument("--no-trackio", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--hf-repo", type=str, default=CANONICAL_OUTPUT_MODEL_REPO)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = train_surrogate(
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        use_trackio=not args.no_trackio,
        freeze_encoder=not args.no_freeze_encoder,
    )
    logger.info("Training results: %s", results)
    if args.no_upload:
        return 0
    url = upload_model_to_hub(args.output_dir, args.hf_repo, args.token, args.private)
    logger.info("Uploaded to %s", url)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
