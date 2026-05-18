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
"""PoseNet -> SAM2-hiera-tiny SISTER EXPLORATION (HF Jobs t4-small).

Insight 3 from `feedback_deep_research_wave_landed_20260518.md`. Research-
only exploration: build a SAM2-hiera-tiny based predictive surrogate for
the contest PoseNet (FastViT-T12 + Hydra, 6-dim pose distortion).

THIS IS A RESEARCH-SUBSTRATE-TRAP-ADJACENT PROBE per CLAUDE.md "Substrate
scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable. The
distillation surrogate is OPERATIONAL (produces an actual 6-dim pose
regression output); whether it's BETTER than the existing PoseNet for
substrate cooperative-receiver use is the research question.

Architecture
------------
- Teacher: contest PoseNet (frozen), via dataset-side pre-computed
  `posenet_pose` (12,) float32. The contest's `compute_distortion` uses
  the first 6 dims.
- Student: SAM2-hiera-tiny vision encoder (frozen, ~33 MB) + custom 6-dim
  regression head (~1M trainable).
- Loss: MSE on first 6 pose dims (matches `PoseNet.compute_distortion`
  formula).

Cross-pollination with Z6 4c (Atick-Redlich cooperative-receiver) and Z7
(predictive coding) per the deep-research wave: SAM2's hierarchical
features at multiple scales (Hiera tiny has 4 stages: 96, 192, 384, 768
channels) provide a rich predictor for "what will frame 2 look like given
frame 1 + ego-motion pose prior?" — which is the Z7 predictive-coding
substrate's core architectural question.

THIS IS RESEARCH-ONLY per the design memo. The SAM2 PoseNet sister is
NOT a substrate. It's a CANDIDATE COOPERATIVE-RECEIVER for ATW V2 + Z6 +
Z7 substrates.

Data
----
Pulls from `adpena/comma-video-substrate-eval-600pairs`. Uses both
image_frame_0 and image_frame_1 (the SAM2 vision encoder runs on the
concatenated or stacked pair).

Hardware + cost
---------------
HF Jobs `t4-small` ($0.40/hr; ~16 GB). Estimated 60-90 min for 30 epochs
at batch_size=8 (1024-res). Total: ~$0.40-0.60.

Output
------
- `adpena/posenet-surrogate-sam2-hiera-tiny` model repo
- TrackIO experiment

Cite-chain provenance
---------------------
- Dataset commit sha (HF Hub metadata)
- Teacher PoseNet safetensors sha (cited in dataset README)
- Student model_name + version

Per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council
symposium" — this PoseNet surrogate is a TOOL, not a substrate. Before any
substrate dispatches CONSUMING this surrogate's output, the substrate must
have its own per-substrate symposium per Catalog #325.

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
    .venv/bin/python submitted_jobs/training_posenet_surrogate_sam2_*.py \\
        --max-train-samples 4 --max-eval-samples 2 --num-epochs 1 \\
        --no-upload --output-dir .omx/tmp/posenet_sam2_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("DALI_DISABLE_NVML", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CANONICAL_DATASET = "adpena/comma-video-substrate-eval-600pairs"
CANONICAL_OUTPUT_MODEL_REPO = "adpena/posenet-surrogate-sam2-hiera-tiny"
STUDENT_MODEL_NAME = "facebook/sam2-hiera-tiny"
N_POSE_DIMS_DISTORTION = 6  # PoseNet.compute_distortion uses first 6
N_POSE_DIMS_TOTAL = 12  # PoseNet head output
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 30

logger = logging.getLogger(__name__)


class SAM2PoseRegressor(nn.Module):
    """SAM2-hiera-tiny vision encoder (frozen) + 6-dim pose regression head.

    Takes a (B, 6, 1024, 1024) input formed by concatenating frame_0 and
    frame_1 RGB-normalized at 1024-res (matches SAM2's native), runs the
    SAM2 image encoder via a wrapper that accepts 6-channel input (Conv2d
    expansion), then a small head outputs (B, N_POSE_DIMS_TOTAL).

    Distillation target: PoseNet's raw 12-dim "pose" head output.
    """

    def __init__(self, sam2_model_name: str = STUDENT_MODEL_NAME, freeze_encoder: bool = True):
        super().__init__()
        from transformers import Sam2Model

        self.sam2 = Sam2Model.from_pretrained(sam2_model_name)
        self.freeze_encoder = freeze_encoder

        # Adapter: 6-channel -> 3-channel (so we can reuse SAM2's pretrained
        # encoder weights without retraining the patch embedding).
        # We initialize the adapter so frame_0 and frame_1 each get half-weight.
        self.channel_adapter = nn.Conv2d(6, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.channel_adapter.weight.data.zero_()
            # Average frame_0 and frame_1 channels to match each RGB channel
            for c in range(3):
                self.channel_adapter.weight.data[c, c, 0, 0] = 0.5
                self.channel_adapter.weight.data[c, c + 3, 0, 0] = 0.5

        if freeze_encoder:
            for name, p in self.sam2.named_parameters():
                if "vision_encoder" in name or "memory_encoder" in name:
                    p.requires_grad_(False)

        # Pose head: 256-dim feature -> 12-dim pose (SAM2 image encoder out 256 channels)
        # We global-average-pool the 64x64 feature map then MLP to 12 dims.
        self.pose_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, N_POSE_DIMS_TOTAL),
        )

    def forward(self, pair_6ch_1024: torch.Tensor) -> torch.Tensor:
        # pair_6ch_1024: (B, 6, 1024, 1024)
        rgb = self.channel_adapter(pair_6ch_1024)  # (B, 3, 1024, 1024)
        vision_outputs = self.sam2.vision_encoder(pixel_values=rgb)
        feat = vision_outputs.last_hidden_state  # (B, 256, 64, 64)
        pose = self.pose_head(feat)  # (B, 12)
        return pose


def posenet_distortion_loss(
    student_pose: torch.Tensor,  # (B, 12)
    teacher_pose: torch.Tensor,  # (B, 12)
) -> torch.Tensor:
    """MSE on first 6 dims (matches PoseNet.compute_distortion)."""
    return F.mse_loss(student_pose[:, :N_POSE_DIMS_DISTORTION], teacher_pose[:, :N_POSE_DIMS_DISTORTION])


def collate_fn(batch):
    from PIL import Image as PILImage

    pair_imgs, target_poses = [], []
    for row in batch:
        f0 = row["image_frame_0"]
        f1 = row["image_frame_1"]
        if not isinstance(f0, PILImage.Image):
            f0 = PILImage.fromarray(np.asarray(f0))
        if not isinstance(f1, PILImage.Image):
            f1 = PILImage.fromarray(np.asarray(f1))
        f0 = f0.convert("RGB").resize((1024, 1024), PILImage.BILINEAR)
        f1 = f1.convert("RGB").resize((1024, 1024), PILImage.BILINEAR)
        f0_arr = (np.asarray(f0).astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        f1_arr = (np.asarray(f1).astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        pair_arr = np.concatenate([f0_arr, f1_arr], axis=-1)  # (1024, 1024, 6)
        pair_t = torch.from_numpy(pair_arr.astype(np.float32)).permute(2, 0, 1)  # (6, 1024, 1024)
        pair_imgs.append(pair_t)
        pose = np.asarray(row["posenet_pose"], dtype=np.float32)
        target_poses.append(torch.from_numpy(pose))
    return torch.stack(pair_imgs), torch.stack(target_poses)


def prepare_dataset_splits(dataset_name, train_val_split=0.9, seed=42, max_train_samples=None, max_eval_samples=None):
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


def train_surrogate(
    output_dir: Path,
    dataset_name: str = CANONICAL_DATASET,
    num_epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    use_trackio: bool = True,
    freeze_encoder: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    if use_trackio:
        try:
            import trackio
            trackio.init(project="posenet_surrogate_sam2_hiera_tiny", config={
                "dataset": dataset_name, "student": STUDENT_MODEL_NAME,
                "num_epochs": num_epochs, "batch_size": batch_size, "lr": lr,
                "freeze_encoder": freeze_encoder,
            })
        except ImportError:
            use_trackio = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student = SAM2PoseRegressor(freeze_encoder=freeze_encoder).to(device)
    trainable = [p for p in student.parameters() if p.requires_grad]
    logger.info("Trainable params: %d / total %d", sum(p.numel() for p in trainable),
                sum(p.numel() for p in student.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    train_ds, eval_ds = prepare_dataset_splits(
        dataset_name, max_train_samples=max_train_samples, max_eval_samples=max_eval_samples,
    )
    logger.info("Train=%d Eval=%d", len(train_ds), len(eval_ds))

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    best_eval = float("inf")
    history = []
    for epoch in range(num_epochs):
        student.train()
        train_loss = 0.0
        train_n = 0
        for pair_imgs, target_poses in train_loader:
            pair_imgs = pair_imgs.to(device)
            target_poses = target_poses.to(device)
            student_pose = student(pair_imgs)
            loss = posenet_distortion_loss(student_pose, target_poses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * pair_imgs.size(0)
            train_n += pair_imgs.size(0)
        train_avg = train_loss / max(train_n, 1)

        student.eval()
        eval_loss = 0.0
        eval_n = 0
        with torch.no_grad():
            for pair_imgs, target_poses in eval_loader:
                pair_imgs = pair_imgs.to(device)
                target_poses = target_poses.to(device)
                student_pose = student(pair_imgs)
                loss = posenet_distortion_loss(student_pose, target_poses)
                eval_loss += loss.item() * pair_imgs.size(0)
                eval_n += pair_imgs.size(0)
        eval_avg = eval_loss / max(eval_n, 1)

        row = {"epoch": epoch, "train_mse": train_avg, "eval_mse": eval_avg}
        history.append(row)
        logger.info("epoch=%d train_mse=%.6f eval_mse=%.6f", epoch, train_avg, eval_avg)
        if use_trackio:
            trackio.log(row)
        if eval_avg < best_eval:
            best_eval = eval_avg
            torch.save(student.state_dict(), output_dir / "model_best.pt")

    torch.save(student.state_dict(), output_dir / "model_final.pt")
    (output_dir / "metrics.json").write_text(json.dumps({"history": history, "best_eval_mse": best_eval}, indent=2))
    if use_trackio:
        trackio.finish()
    return {"best_eval_mse": best_eval, "output_dir": str(output_dir)}


def upload_model_to_hub(local_dir: Path, repo_id: str, token: Optional[str] = None, private: bool = False) -> str:
    from huggingface_hub import HfApi, get_token
    if token is None:
        token = get_token()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    card = f"""---
license: mit
library_name: transformers
tags:
- distillation
- sam2
- comma-video-compression
- posenet-surrogate
- research-only
---

# posenet-surrogate-sam2-hiera-tiny

SAM2-hiera-tiny vision encoder (frozen) + 6-dim pose regression head fine-
tuned (Catalog #779 freezing exploit) on the comma.ai contest PoseNet
output. RESEARCH-ONLY — non-promotable advisory only.

## Provenance

- Source dataset: [`{CANONICAL_DATASET}`](https://huggingface.co/datasets/{CANONICAL_DATASET})
- Teacher: contest PoseNet (frozen)
- Student base: `{STUDENT_MODEL_NAME}` (encoder frozen; pose head + channel adapter trained)

Per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM" non-negotiable + Catalog #325:
this surrogate is a CANDIDATE COOPERATIVE-RECEIVER for ATW V2 / Z6 / Z7
substrates, NOT a substrate itself. Substrates consuming this surrogate
must have their own per-substrate symposium before dispatch.
"""
    (local_dir / "README.md").write_text(card)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(local_dir),
                      commit_message="posenet_surrogate_sam2_hiera_tiny: research-only candidate")
    return f"https://huggingface.co/{repo_id}"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=CANONICAL_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("./posenet_sam2_out"))
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
        output_dir=args.output_dir, dataset_name=args.dataset,
        num_epochs=args.num_epochs, batch_size=args.batch_size, lr=args.lr,
        max_train_samples=args.max_train_samples, max_eval_samples=args.max_eval_samples,
        use_trackio=not args.no_trackio, freeze_encoder=not args.no_freeze_encoder,
    )
    logger.info("Results: %s", results)
    if args.no_upload:
        return 0
    url = upload_model_to_hub(args.output_dir, args.hf_repo, args.token, args.private)
    logger.info("Uploaded to %s", url)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
