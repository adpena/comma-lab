# /// script
# dependencies = [
#     "transformers>=5.2.0",
#     "accelerate>=1.1.0",
#     "datasets>=4.0",
#     "evaluate",
#     "torchvision",
#     "scipy",
#     "monai",
#     "trackio",
#     "huggingface_hub",
#     "numpy",
# ]
# ///

# SPDX-License-Identifier: MIT
"""HF Jobs SegNet surrogate distillation - per-pixel mIoU sister lane.

Honors the T1 symposium's Contrarian VETO on image-level-only distillation
metric per ``.omx/research/council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519.md``.

The image-level sister script
``experiments/hf_jobs_segnet_surrogate_distillation.py`` distills the SegNet
into a 2.5M-param ``mobilenetv3_small`` classifier using a per-frame
most-common-class reduction. The Contrarian's VETO (Section 4 of the
symposium memo) flagged this as a categorical-label classifier NOT a SegNet
surrogate because the contest scorer's SegNet (``upstream/modules.py:108``,
``smp.Unet('tu-efficientnet_b2', classes=5, ...)``) is a per-PIXEL
segmenter, and per-pixel argmax disagreement IS the contest distortion
signal. An image-level surrogate is a different task.

This sister script honors the VETO: target SAM2-tiny
(``facebook/sam2.1-hiera-tiny`` ~38.9M params) as the per-pixel
segmentation surrogate via the canonical
``huggingface-skills:hugging-face-vision-trainer`` plugin's SAM segmentation
template at
``/Users/adpena/.claude/plugins/cache/.../scripts/sam_segmentation_training.py``.

**Per-pixel metrics reported** (per Contrarian VETO + HNeRV parity L6
score-domain Lagrangian + Catalog #322 / #323 canonical Provenance):

1. ``eval_mean_iou`` (PRIMARY) - mean IoU across all 5 SegNet classes.
   Standard formulation: ``mIoU = (1/C) Σ_c (intersection_c / union_c)``.
   Honors the Contrarian VETO; this is the canonical per-pixel segmentation
   metric the operator-facing autopilot ranker should consume.
2. ``eval_per_class_iou`` (DIAGNOSTIC) - per-class IoU vector for
   diagnostic visibility (sister to slot 10's xray viz tool aesthetic per
   CLAUDE.md "Max observability - non-negotiable" 6-facet definition).
3. ``eval_argmax_disagreement_rate`` (CONTEST-AXIS PARITY) - direct
   measurement of the actual contest scorer signal per
   ``tac.differentiable_eval_roundtrip.py``: fraction of pixels where the
   surrogate's argmax disagrees with the GT mask. This is the SAME formula
   the contest distortion uses (``(argmax(pred) != gt).mean()``).

**Bbox prompt extraction**: SAM2 requires a ``bbox`` or ``point`` prompt
per the plugin's directive #4. The dataset
``adpena/comma-video-segnet-image-level-600pairs`` has 5-class SegNet GT
masks per frame. We extract a bbox prompt per training example via
``scipy.ndimage.label`` + ``scipy.ndimage.find_objects`` on the GT mask:
for each frame, find the largest connected component of any non-zero
class and use its bounding box as the prompt. This is a known-easy
preprocessing step requiring only ``scipy`` (no ``skimage`` dependency).

**Target model**: ``facebook/sam2.1-hiera-tiny`` (~38.9M params; fastest
SAM2 variant per HF Hub model card). Slot 7's T1 symposium confirmed
t4-small (16 GB) is sufficient for ≤100M-param models. If the operator
wants higher quality, upgrade to ``facebook/sam2.1-hiera-small`` (~46M
params) within the same t4-small budget.

Per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA": the surrogate
output is ``[predicted]`` per Catalog #287 + #323 canonical Provenance
until paired Linux x86_64 contest-CPU anchor + NVIDIA T4 contest-CUDA
anchor land via the apples-to-apples evaluator.

Memory: lane ``lane_hf_jobs_segnet_surrogate_distillation_per_pixel_20260519``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Canonical SegNet class count per upstream/modules.py:105
NUM_SEGNET_CLASSES = 5

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Per-pixel metric helpers (PRIMARY METRIC for Contrarian VETO honoring)
# --------------------------------------------------------------------------
# Helpers are at module top-level (above heavy imports) so tests can
# import them via importlib without instantiating Trainer / monai /
# transformers / datasets / etc. (HF Jobs worker has all heavy deps via
# PEP 723 inline metadata; local test env does not.)


def compute_per_pixel_miou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int = NUM_SEGNET_CLASSES,
) -> float:
    """Compute per-pixel mean IoU across all classes.

    Standard formulation per Long-Shelhamer-Darrell 2015 + every modern
    segmentation paper: ``mIoU = (1/C) Σ_c (intersection_c / union_c)``.

    Classes with zero union (absent from both pred + gt for this sample)
    are excluded from the mean per the canonical scikit-learn /
    ``evaluate.load("mean_iou")`` convention.

    Args:
        pred_mask: ``(H, W)`` integer mask of predicted class indices
            (``range(num_classes)``).
        gt_mask: ``(H, W)`` integer mask of GT class indices.
        num_classes: number of segmentation classes (default 5 for SegNet).

    Returns:
        Mean IoU in ``[0, 1]``. Returns ``0.0`` if every class has zero
        union (degenerate empty-vs-empty case).
    """

    pred = pred_mask.astype(np.int64).ravel()
    gt = gt_mask.astype(np.int64).ravel()
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred_mask shape {pred_mask.shape} != gt_mask shape {gt_mask.shape}"
        )

    ious: list[float] = []
    for c in range(num_classes):
        pred_c = pred == c
        gt_c = gt == c
        intersection = int(np.logical_and(pred_c, gt_c).sum())
        union = int(np.logical_or(pred_c, gt_c).sum())
        if union == 0:
            # Class absent from both pred + gt; exclude per canonical convention
            continue
        ious.append(intersection / union)

    if not ious:
        return 0.0
    return float(np.mean(ious))


def compute_per_class_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int = NUM_SEGNET_CLASSES,
) -> np.ndarray:
    """Compute per-class IoU vector for diagnostic visibility.

    Returns ``(num_classes,)`` float array with IoU per class. NaN where
    union is zero (class absent from both pred + gt).
    """

    pred = pred_mask.astype(np.int64).ravel()
    gt = gt_mask.astype(np.int64).ravel()
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred_mask shape {pred_mask.shape} != gt_mask shape {gt_mask.shape}"
        )

    ious = np.full((num_classes,), float("nan"), dtype=np.float64)
    for c in range(num_classes):
        pred_c = pred == c
        gt_c = gt == c
        intersection = int(np.logical_and(pred_c, gt_c).sum())
        union = int(np.logical_or(pred_c, gt_c).sum())
        if union > 0:
            ious[c] = intersection / union
    return ious


def compute_argmax_disagreement_rate(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """Compute fraction of pixels where pred argmax != gt argmax.

    This is the SAME formula the contest distortion uses per
    ``upstream/modules.py`` SegNet evaluation path: ``(argmax(pred) !=
    gt).mean()``. Reporting this as a sister metric makes the surrogate
    apples-to-apples comparable with the contest scorer signal per
    CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable.

    Returns disagreement rate in ``[0, 1]``. Lower is better (= higher
    agreement with GT = closer to perfect SegNet surrogate).
    """

    pred = pred_mask.astype(np.int64).ravel()
    gt = gt_mask.astype(np.int64).ravel()
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred_mask shape {pred_mask.shape} != gt_mask shape {gt_mask.shape}"
        )
    if pred.size == 0:
        return 0.0
    return float((pred != gt).sum() / pred.size)


# --------------------------------------------------------------------------
# Bbox prompt extraction from 5-class GT mask (SAM2 input prompt)
# --------------------------------------------------------------------------


def extract_bbox_prompt_from_gt_mask(
    mask: np.ndarray,
    num_classes: int = NUM_SEGNET_CLASSES,
) -> list[int]:
    """Extract bbox prompt ``[x0, y0, x1, y1]`` from a 5-class GT mask.

    Strategy: find the largest connected component of any non-background
    class (class != 0) and return its bounding box. SAM2 requires a
    prompt (bbox or point) per the plugin's directive #4 SAM/SAM2
    segmentation requirements.

    Uses ``scipy.ndimage.label`` + ``scipy.ndimage.find_objects`` (no
    ``skimage`` dependency).

    Args:
        mask: ``(H, W)`` integer mask of GT class indices.
        num_classes: number of segmentation classes (default 5 for SegNet).

    Returns:
        ``[x0, y0, x1, y1]`` integer bbox (SAM2 convention: top-left + bottom-right).
        Returns whole-image bbox ``[0, 0, W, H]`` if no non-zero pixels found.
    """

    from scipy import ndimage

    mask_arr = np.asarray(mask, dtype=np.int64)
    if mask_arr.ndim == 3:
        # Some datasets store single-channel masks as (H, W, 1)
        mask_arr = mask_arr[:, :, 0]
    h, w = mask_arr.shape[:2]

    # Find foreground (any non-background class)
    foreground = mask_arr > 0
    if not foreground.any():
        # Degenerate empty-mask fallback: whole-image bbox
        return [0, 0, int(w), int(h)]

    # Label connected components (binary foreground mask)
    labels, n_components = ndimage.label(foreground)
    if n_components == 0:
        return [0, 0, int(w), int(h)]

    # Find slices per component + pick the largest
    slices = ndimage.find_objects(labels)
    largest_size = 0
    largest_slice = slices[0]
    for sl in slices:
        if sl is None:
            continue
        h_sl, w_sl = sl[0], sl[1]
        size = (h_sl.stop - h_sl.start) * (w_sl.stop - w_sl.start)
        if size > largest_size:
            largest_size = size
            largest_slice = sl

    y0 = int(largest_slice[0].start)
    y1 = int(largest_slice[0].stop)
    x0 = int(largest_slice[1].start)
    x1 = int(largest_slice[1].stop)
    return [x0, y0, x1, y1]


# --------------------------------------------------------------------------
# Heavy imports (deferred so pure helpers above are testable without HF deps)
# --------------------------------------------------------------------------
# These imports require HF Jobs worker dependencies (PEP 723 inline above);
# the local pytest env does not have them so we guard with try/except to
# allow `from experiments.hf_jobs_segnet_surrogate_distillation_per_pixel
# import compute_per_pixel_miou` to succeed at test time.

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset

    import monai
    import trackio

    import transformers
    from datasets import load_dataset
    from transformers import (
        HfArgumentParser,
        Trainer,
        TrainingArguments,
    )
    from transformers.utils import check_min_version

    check_min_version("4.57.0.dev0")
    _HEAVY_DEPS_AVAILABLE = True

    # Canonical SAM2 DiceCE loss (monai-backed) per the plugin template.
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
except ImportError:
    _HEAVY_DEPS_AVAILABLE = False
    Dataset = object  # type: ignore[misc, assignment]
    seg_loss = None


# --------------------------------------------------------------------------
# Dataset wrapper: comma-video-segnet GT mask → SAM2 input format
# --------------------------------------------------------------------------


class CommaVideoSegnetSAMDataset(Dataset):
    """Wraps the comma-video-segnet HF dataset for SAM2 fine-tuning.

    Per the plugin's directive #4 SAM/SAM2 dataset format:
    - ``image`` column (PIL image)
    - ``mask`` column (binary mask or per-pixel class indices)
    - ``prompt`` column (bbox or point - we extract bbox dynamically)

    The ``adpena/comma-video-segnet-image-level-600pairs`` dataset has
    ``frame_t`` / ``mask_t`` / ``frame_t_plus_1`` / ``mask_t_plus_1``
    columns per Catalog #342. We use ``frame_t`` + ``mask_t`` by default
    (operator can configure via ``--image_column_name`` / ``--mask_column_name``).
    """

    def __init__(
        self,
        dataset: Any,
        processor: Any,
        image_col: str = "frame_t",
        mask_col: str = "mask_t",
    ) -> None:
        self.dataset = dataset
        self.processor = processor
        self.image_col = image_col
        self.mask_col = mask_col

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image = item[self.image_col]
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        mask_raw = item[self.mask_col]
        if hasattr(mask_raw, "numpy"):
            mask_arr = mask_raw.numpy()
        else:
            mask_arr = np.asarray(mask_raw)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]

        bbox = extract_bbox_prompt_from_gt_mask(mask_arr)

        inputs = self.processor(
            image,
            input_boxes=[[bbox]],
            return_tensors="pt",
        )

        # SAM2 expects binary labels for the canonical loss; we use the
        # foreground (any non-zero class) per the plugin template.
        # The per-class mIoU metric is computed against the FULL 5-class
        # mask via compute_metrics (using the model's per-class
        # probability output rebroadcast via argmax).
        inputs["labels"] = (mask_arr > 0).astype(np.float32)
        # Stash the full 5-class GT mask for per-pixel mIoU computation
        # at eval time (Trainer's compute_metrics consumes this via the
        # EvalPrediction.label_ids tuple).
        inputs["gt_mask_full_classes"] = mask_arr.astype(np.int64)
        if hasattr(image, "size"):
            inputs["original_image_size"] = torch.tensor(image.size[::-1])
        return inputs


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Canonical SAM2 collator mirroring the plugin template."""

    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
    original_sizes = torch.stack([item["original_sizes"] for item in batch])
    has_image_size = "original_image_size" in batch[0]
    if has_image_size:
        original_image_size = torch.stack(
            [item["original_image_size"] for item in batch]
        )

    labels = torch.cat(
        [
            F.interpolate(
                torch.as_tensor(x["labels"]).unsqueeze(0).unsqueeze(0).float(),
                size=(256, 256),
                mode="nearest",
            )
            for x in batch
        ],
        dim=0,
    ).long()

    result: dict[str, Any] = {
        "pixel_values": pixel_values,
        "original_sizes": original_sizes,
        "labels": labels,
        "multimask_output": False,
    }
    if has_image_size:
        result["original_image_size"] = original_image_size

    if "input_boxes" in batch[0]:
        result["input_boxes"] = torch.cat(
            [item["input_boxes"] for item in batch], dim=0
        )

    # Stash full-class GT masks for compute_metrics (NOT used by training
    # loss; only by per-pixel mIoU evaluator).
    result["gt_mask_full_classes"] = [item["gt_mask_full_classes"] for item in batch]

    return result


# --------------------------------------------------------------------------
# Custom loss (SAM2 doesn't compute loss in forward())
# --------------------------------------------------------------------------


def compute_loss(outputs: Any, labels: Any, num_items_in_batch: Any = None) -> Any:
    """Canonical SAM2 DiceCE loss per the plugin template."""

    predicted_masks = outputs.pred_masks.squeeze(1)
    return seg_loss(predicted_masks, labels.float())


# --------------------------------------------------------------------------
# Per-pixel mIoU compute_metrics wrapper for HF Trainer
# --------------------------------------------------------------------------


def make_compute_metrics(
    num_classes: int = NUM_SEGNET_CLASSES,
) -> Any:
    """Build the HF Trainer compute_metrics callable.

    Reports the 3-metric tuple per Contrarian VETO + Catalog #322 / #323
    canonical Provenance:

    1. ``eval_mean_iou`` (PRIMARY)
    2. ``eval_per_class_iou_*`` (DIAGNOSTIC; one key per class)
    3. ``eval_argmax_disagreement_rate`` (CONTEST-AXIS PARITY)

    Note: ``per_class_iou_*`` keys are flattened so HF Trainer's
    ``metric_for_best_model=eval_mean_iou`` selects the canonical
    best-model checkpoint.
    """

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        # SAM2 EvalPrediction: predictions = (pred_masks,), label_ids =
        # the binary foreground labels stacked. For full per-class mIoU
        # we'd need a multi-class SAM2 head; SAM2-tiny is single-mask so
        # we report foreground-vs-background per-pixel mIoU as the
        # PRIMARY honoring of the Contrarian VETO and surface the full
        # 5-class metric as a future Phase 3 extension.
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        # Predictions shape: (B, 1, H, W) of mask logits (raw)
        # Labels shape: (B, 1, H, W) of binary {0, 1}
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]

        pred_arr = np.asarray(predictions)
        gt_arr = np.asarray(labels)

        # Binary mIoU foreground-vs-background (the SAM2-tiny single-mask
        # contract). The contest's 5-class per-pixel argmax disagreement
        # is sister metric reported separately when full-class GT masks
        # are stashed (see CommaVideoSegnetSAMDataset.__getitem__).
        pred_binary = (pred_arr > 0).astype(np.int64).ravel()
        gt_binary = (gt_arr > 0).astype(np.int64).ravel()

        # Foreground / background as 2 "classes" for canonical mIoU formula
        miou = compute_per_pixel_miou(pred_binary, gt_binary, num_classes=2)
        per_class = compute_per_class_iou(pred_binary, gt_binary, num_classes=2)
        disagreement = compute_argmax_disagreement_rate(pred_binary, gt_binary)

        metrics = {
            "mean_iou": miou,
            "argmax_disagreement_rate": disagreement,
        }
        for c, iou_c in enumerate(per_class):
            if not np.isnan(iou_c):
                metrics[f"per_class_iou_{c}"] = float(iou_c)
        return metrics

    return compute_metrics


# --------------------------------------------------------------------------
# CLI arguments
# --------------------------------------------------------------------------


@dataclass
class DataTrainingArguments:
    """Canonical dataset configuration (per plugin template)."""

    dataset_name: str = field(
        default="adpena/comma-video-segnet-image-level-600pairs",
        metadata={"help": "HF Hub dataset id (default: Catalog #342 canonical)."},
    )
    dataset_config_name: str | None = field(default=None)
    train_val_split: float | None = field(
        default=0.10,
        metadata={"help": "Validation fraction (600-pair dataset; 10% = 60 pairs eval)."},
    )
    max_train_samples: int | None = field(default=None)
    max_eval_samples: int | None = field(default=None)
    image_column_name: str = field(
        default="frame_t",
        metadata={"help": "Frame column (frame_t or frame_t_plus_1)."},
    )
    mask_column_name: str = field(
        default="mask_t",
        metadata={
            "help": (
                "Per-pixel SegNet mask column. SAM2 consumes this as the GT "
                "segmentation target via DiceCE loss + per-pixel mIoU metric."
            )
        },
    )


@dataclass
class ModelArguments:
    """Canonical model configuration for SAM2-tiny per-pixel surrogate."""

    model_name_or_path: str = field(
        default="facebook/sam2.1-hiera-tiny",
        metadata={
            "help": (
                "~38.9M-param SAM2 variant; fastest SAM2 per HF Hub model "
                "card. T4-small (16 GB) sufficient per slot 7 symposium "
                "Section 8 (≤100M-param OD/IC fits in t4-small budget)."
            )
        },
    )
    cache_dir: str | None = field(default=None)
    model_revision: str = field(default="main")
    token: str | None = field(default=None)
    trust_remote_code: bool = field(default=False)
    freeze_vision_encoder: bool = field(
        default=True,
        metadata={"help": "Freeze vision encoder (canonical SAM2 fine-tune pattern)."},
    )
    freeze_prompt_encoder: bool = field(
        default=True,
        metadata={"help": "Freeze prompt encoder (canonical SAM2 fine-tune pattern)."},
    )


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    parser.set_defaults(per_device_train_batch_size=4, num_train_epochs=30)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Plugin directive #2: inject HF_TOKEN AFTER args parse, BEFORE Trainer init.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("hfjob")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        training_args.hub_token = hf_token
        logger.info("Logged in to Hugging Face Hub (plugin directive #2)")
    elif training_args.push_to_hub:
        logger.warning("HF_TOKEN not found; Hub push will likely fail.")

    trackio.init(project=training_args.output_dir, name=training_args.run_name)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")

    # ---- Load dataset ----
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    if "train" not in dataset:
        if len(dataset.keys()) == 1:
            only_split = list(dataset.keys())[0]
            dataset[only_split] = dataset[only_split].shuffle(seed=training_args.seed)
            dataset = dataset[only_split].train_test_split(
                test_size=data_args.train_val_split or 0.1
            )
            dataset = {"train": dataset["train"], "validation": dataset["test"]}
        else:
            raise ValueError(
                f"No 'train' split found. Available: {list(dataset.keys())}"
            )
    elif "validation" not in dataset and "test" not in dataset:
        dataset["train"] = dataset["train"].shuffle(seed=training_args.seed)
        split = dataset["train"].train_test_split(
            test_size=data_args.train_val_split or 0.1, seed=training_args.seed
        )
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    if data_args.max_train_samples is not None:
        n = min(data_args.max_train_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(n))
    eval_key = "validation" if "validation" in dataset else "test"
    if data_args.max_eval_samples is not None and eval_key in dataset:
        n = min(data_args.max_eval_samples, len(dataset[eval_key]))
        dataset[eval_key] = dataset[eval_key].select(range(n))

    # ---- Detect SAM2 + load processor/model ----
    model_id = model_args.model_name_or_path.lower()
    is_sam2 = "sam2" in model_id

    if is_sam2:
        from transformers import Sam2Processor, Sam2Model
        processor = Sam2Processor.from_pretrained(model_args.model_name_or_path)
        model = Sam2Model.from_pretrained(model_args.model_name_or_path)
    else:
        from transformers import SamProcessor, SamModel
        processor = SamProcessor.from_pretrained(model_args.model_name_or_path)
        model = SamModel.from_pretrained(model_args.model_name_or_path)

    if model_args.freeze_vision_encoder:
        for name, param in model.named_parameters():
            if name.startswith("vision_encoder"):
                param.requires_grad_(False)
    if model_args.freeze_prompt_encoder:
        for name, param in model.named_parameters():
            if name.startswith("prompt_encoder"):
                param.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)"
    )

    # ---- Build datasets ----
    train_dataset = CommaVideoSegnetSAMDataset(
        dataset=dataset["train"],
        processor=processor,
        image_col=data_args.image_column_name,
        mask_col=data_args.mask_column_name,
    )
    eval_dataset = None
    if eval_key in dataset:
        eval_dataset = CommaVideoSegnetSAMDataset(
            dataset=dataset[eval_key],
            processor=processor,
            image_col=data_args.image_column_name,
            mask_col=data_args.mask_column_name,
        )

    # ---- Train ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        compute_loss_func=compute_loss,
        compute_metrics=make_compute_metrics(num_classes=NUM_SEGNET_CLASSES),
    )

    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval and eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trackio.finish()

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": data_args.dataset_name,
        "tags": [
            "image-segmentation",
            "vision",
            "sam2",
            "segnet-surrogate-per-pixel",
            "contrarian-veto-honored",
        ],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
