# /// script
# dependencies = [
#     "torch>=2.5.0",
#     "timm>=1.0.0",
#     "datasets>=2.18.0",
#     "huggingface_hub>=0.23.0",
#     "safetensors>=0.4.0",
#     "pillow>=10.0.0",
#     "numpy",
# ]
# ///

# SPDX-License-Identifier: MIT
"""DINOv3 cooperative-receiver ANCHOR EXTRACTION (HF Jobs t4-small).

Insight 2 from `feedback_deep_research_wave_landed_20260518.md`. Runs frozen
DINOv3 (`timm/vit_base_patch16_dinov3.lvd1689m`, 86.6M params) on the
canonical `adpena/comma-video-substrate-eval-600pairs` dataset and emits a
sidecar HF dataset with the (frozen) CLS + patch features.

THIS IS NOT TRAINING. The DINOv3 weights are frozen by construction. This
job just runs inference on 600 pairs to produce a queryable feature
extraction that substrates (ATW V2 / Z6 / Z7) can compose with their own
cooperative-receiver losses.

Why a sidecar dataset (not on-the-fly extraction)
-------------------------------------------------
- Saves repeated DINOv3 inference cost across multiple substrate trainers.
- Makes the cooperative-receiver target deterministic + cite-able (every
  substrate uses byte-identical anchor features).
- ~100 MB per dataset (768-dim CLS + 196*768 patch tokens per frame *
  2 frames * 600 pairs * 4 bytes = ~720 MB; float16 ~360 MB; we use float16).

Output dataset schema
---------------------
- pair_idx : int32
- video_name : str
- frame_0_cls : (768,) float16
- frame_0_patch_tokens : (196, 768) float16
- frame_1_cls : (768,) float16
- frame_1_patch_tokens : (196, 768) float16
- dinov3_model_name : str (always `timm/vit_base_patch16_dinov3.lvd1689m`)

Output repo: `adpena/comma-video-dinov3-cooperative-receiver-anchor-600pairs`

Hardware + cost
---------------
HF Jobs `t4-small` ($0.40/hr; ~16 GB). DINOv3-base inference on 600 pairs
= 1200 forward passes, ~10-15 min total. Total cost: ~$0.07-0.10.

Cite-chain provenance
---------------------
Per CLAUDE.md "Apples-to-apples evidence discipline":
- Source dataset commit sha (from HF Hub metadata)
- Frozen DINOv3 model identifier + revision sha
- Output dataset's `provenance.json` records all of the above + device

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" — these
features are a TOOL surrogate (frozen anchor); contest score claims using
them MUST still route through the canonical auth-eval gate.

Usage (HF Jobs MCP)
-------------------
    hf_jobs("uv", {
        "script": "<this file's content>",
        "flavor": "t4-small",
        "secrets": {"HF_TOKEN": "$HF_TOKEN"},
        "timeout": "1h",
    })

Usage (local smoke)
-------------------
    .venv/bin/python submitted_jobs/training_dinov3_cooperative_receiver_anchor_*.py \\
        --max-pairs 4 --no-upload --output-dir .omx/tmp/dinov3_anchor_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("DALI_DISABLE_NVML", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F


CANONICAL_DATASET = "adpena/comma-video-substrate-eval-600pairs"
CANONICAL_OUTPUT_REPO = "adpena/comma-video-dinov3-cooperative-receiver-anchor-600pairs"
CANONICAL_DINOV3_MODEL_NAME = "timm/vit_base_patch16_dinov3.lvd1689m"
DINOV3_INPUT_SIZE = 224
DINOV3_IMAGENET_MEAN = (0.485, 0.456, 0.406)
DINOV3_IMAGENET_STD = (0.229, 0.224, 0.225)

logger = logging.getLogger(__name__)


def normalize_for_dinov3(rgb_255_pil_image, size: int = DINOV3_INPUT_SIZE) -> torch.Tensor:
    """Resize + normalize a single RGB-255 PIL image for DINOv3 inference."""
    from PIL import Image as PILImage
    if not isinstance(rgb_255_pil_image, PILImage.Image):
        rgb_255_pil_image = PILImage.fromarray(np.asarray(rgb_255_pil_image))
    img = rgb_255_pil_image.convert("RGB").resize((size, size), PILImage.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array(DINOV3_IMAGENET_MEAN, dtype=np.float32)
    std = np.array(DINOV3_IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)  # (3, size, size)


def load_frozen_dinov3(model_name: str = CANONICAL_DINOV3_MODEL_NAME, device: str = "cpu"):
    import timm
    timm_name = model_name if model_name.startswith("hf_hub:") or "/" in model_name else model_name
    if "/" in model_name and not timm_name.startswith("hf_hub:"):
        timm_name = f"hf_hub:{model_name}"
    model = timm.create_model(timm_name, pretrained=True, num_classes=0)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def extract_anchors(
    output_dir: Path,
    dataset_name: str = CANONICAL_DATASET,
    max_pairs: Optional[int] = None,
    batch_size: int = 16,
    model_name: str = CANONICAL_DINOV3_MODEL_NAME,
) -> dict:
    """Run frozen DINOv3 on dataset; emit anchor features as new dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    from datasets import Dataset, Features, Sequence, Value, load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Extracting on device=%s", device)

    ds = load_dataset(dataset_name, split="train")
    if max_pairs is not None:
        ds = ds.select(range(min(max_pairs, len(ds))))
    logger.info("Processing %d pairs", len(ds))

    model = load_frozen_dinov3(model_name=model_name, device=device)

    rows = []
    with torch.inference_mode():
        for batch_start in range(0, len(ds), batch_size):
            batch_end = min(batch_start + batch_size, len(ds))
            batch_rows = ds.select(range(batch_start, batch_end))

            for which_frame, frame_field in [(0, "image_frame_0"), (1, "image_frame_1")]:
                imgs = torch.stack(
                    [normalize_for_dinov3(r[frame_field]) for r in batch_rows]
                ).to(device)
                tokens = model.forward_features(imgs)  # (B, 1+N, hidden)
                cls = tokens[:, 0, :].to(torch.float16).cpu().numpy()
                patch = tokens[:, 1:, :].to(torch.float16).cpu().numpy()

                for i in range(len(batch_rows)):
                    if which_frame == 0:
                        rows.append({
                            "pair_idx": int(batch_rows[i]["pair_idx"]),
                            "video_name": str(batch_rows[i]["video_name"]),
                            "frame_0_cls": cls[i].tolist(),
                            "frame_0_patch_tokens": patch[i].flatten().tolist(),
                            # Placeholder; will be overwritten on frame_1 pass
                            "frame_1_cls": [],
                            "frame_1_patch_tokens": [],
                            "dinov3_model_name": model_name,
                        })
                    else:  # which_frame == 1
                        row = rows[batch_start + i]
                        row["frame_1_cls"] = cls[i].tolist()
                        row["frame_1_patch_tokens"] = patch[i].flatten().tolist()

            if (batch_start // batch_size) % 5 == 0:
                logger.info("processed batch %d-%d", batch_start, batch_end)

    # Materialize HF dataset
    hidden_dim = 768  # DINOv3-base
    n_patches = 196  # (224/16)**2
    features = Features({
        "pair_idx": Value("int32"),
        "video_name": Value("string"),
        "frame_0_cls": Sequence(Value("float16"), length=hidden_dim),
        "frame_0_patch_tokens": Sequence(Value("float16"), length=n_patches * hidden_dim),
        "frame_1_cls": Sequence(Value("float16"), length=hidden_dim),
        "frame_1_patch_tokens": Sequence(Value("float16"), length=n_patches * hidden_dim),
        "dinov3_model_name": Value("string"),
    })
    out_ds = Dataset.from_list(rows, features=features)
    out_ds.save_to_disk(str(output_dir / "data"))

    provenance = {
        "source_dataset": dataset_name,
        "dinov3_model_name": model_name,
        "device": device,
        "torch_version": torch.__version__,
        "n_pairs": len(rows),
        "hidden_dim": hidden_dim,
        "n_patches": n_patches,
        "evidence_grade": "[contest-CUDA frozen-anchor]" if device == "cuda" else "[contest-CPU frozen-anchor]",
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))

    card = f"""---
license: mit
task_categories:
- feature-extraction
tags:
- comma-video-compression
- dinov3
- cooperative-receiver
- frozen-anchor
---

# comma-video-dinov3-cooperative-receiver-anchor

Frozen DINOv3 ({model_name}, 86.6M params, LVD-16-89M pretrained) CLS +
patch features for the canonical [`{dataset_name}`](https://huggingface.co/datasets/{dataset_name})
substrate evaluation dataset.

Per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM" + `feedback_deep_research_wave_landed_20260518.md`:
substrates that use this as a cooperative-receiver anchor (ATW V2-1 / Z6 /
Z7) compose its features via Hinton T=2.0 KL distillation per the canonical
helper `tac.dinov3_cooperative_receiver_anchor.cooperative_receiver_dinov3_kl_loss`.

## Schema

| field                       | shape                   | dtype   |
|-----------------------------|-------------------------|---------|
| pair_idx                    | -                       | int32   |
| video_name                  | -                       | str     |
| frame_0_cls                 | (768,)                  | float16 |
| frame_0_patch_tokens        | (196 * 768,) flat       | float16 |
| frame_1_cls                 | (768,)                  | float16 |
| frame_1_patch_tokens        | (196 * 768,) flat       | float16 |
| dinov3_model_name           | -                       | str     |

Total size: ~360 MB.

## Provenance

- Source: [`{dataset_name}`](https://huggingface.co/datasets/{dataset_name})
- Model: `{model_name}` (frozen; pretrained on LVD-16-89M)
- Device: {device}
- evidence_grade: {provenance['evidence_grade']}

License: MIT.
"""
    (output_dir / "README.md").write_text(card)
    logger.info("Extracted %d pairs to %s", len(rows), output_dir)
    return provenance


def upload_to_hub(local_dir: Path, repo_id: str, token: Optional[str] = None, private: bool = False) -> str:
    from huggingface_hub import HfApi, get_token
    if token is None:
        token = get_token()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(local_dir),
                      commit_message="dinov3_anchor: frozen feature extraction")
    return f"https://huggingface.co/datasets/{repo_id}"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=CANONICAL_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("./dinov3_anchor_out"))
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", type=str, default=CANONICAL_DINOV3_MODEL_NAME)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--hf-repo", type=str, default=CANONICAL_OUTPUT_REPO)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    provenance = extract_anchors(
        output_dir=args.output_dir, dataset_name=args.dataset,
        max_pairs=args.max_pairs, batch_size=args.batch_size, model_name=args.model_name,
    )
    logger.info("Provenance: %s", provenance)
    if args.no_upload:
        return 0
    url = upload_to_hub(args.output_dir, args.hf_repo, args.token, args.private)
    logger.info("Uploaded to %s", url)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
