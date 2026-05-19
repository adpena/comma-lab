# /// script
# dependencies = [
#     "transformers>=5.2.0",
#     "accelerate>=1.1.0",
#     "timm",
#     "datasets>=4.0",
#     "evaluate",
#     "scikit-learn",
#     "torchvision",
#     "trackio",
#     "huggingface_hub",
# ]
# ///

# SPDX-License-Identifier: MIT
"""HF Jobs SegNet surrogate distillation script (Catalog #342 + #523).

Mirrors the canonical ``huggingface-skills:hugging-face-vision-trainer``
plugin's ``image_classification_training.py`` template per its directives:

- **Directive #1**: ALL config via ``script_args`` parsed by
  ``HfArgumentParser`` (NOT by editing Python variables).
- **Directive #2**: ``HF_TOKEN`` injected into ``training_args.hub_token``
  AFTER args parse, BEFORE Trainer init (so the Trainer's
  ``push_to_hub=True`` path has the credentials it needs).
- **Directive #3**: PEP 723 inline metadata above with all heavy
  dependencies (transformers + accelerate + timm + datasets + evaluate +
  torchvision + trackio + huggingface_hub) so HF Jobs `uv` resolves the
  full closure.
- **Directive #4**: invoked with
  ``--no_remove_unused_columns --push_to_hub --metric_for_best_model
  eval_accuracy --greater_is_better True`` per the canonical CLI
  contract.

**Task**: Hinton-style knowledge distillation from the upstream SegNet
(EfficientNet-B2 UNet, 5 classes) into a lightweight
``timm/mobilenetv3_small_100.lamb_in1k`` surrogate (~2.5M params) by
training on the per-frame SegNet argmax class indices in the
``adpena/comma-video-segnet-image-level-600pairs`` dataset.

**Important**: this is the IMAGE-LEVEL distillation surrogate (Catalog
#523 L2 scaffold). The full per-pixel mask distillation is a sister lane
(higher complexity; deferred to post-symposium Phase 2). The image-level
surrogate produces a single "most common class index" per frame,
empirically useful as a fast advisory signal for substrate-design ranking
+ as a soft-target Hinton distillation teacher in downstream substrate
training.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": the surrogate
output is ``[predicted]`` per Catalog #287 + #323 canonical Provenance
until paired Linux x86_64 contest-CPU anchor + NVIDIA T4 contest-CUDA
anchor land via the apples-to-apples evaluator.

Memory: lane ``lane_hf_jobs_segnet_surrogate_distillation_20260519``.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import trackio  # noqa: F401  (tracker init via env vars per plugin convention)

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer import EvalPrediction
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

check_min_version("4.57.0.dev0")
require_version("datasets>=2.0.0")


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
    label_column_name: str = field(
        default="mask_t",
        metadata={
            "help": (
                "SegNet argmax mask column (mask_t or mask_t_plus_1). For "
                "image-level distillation the dataset preprocessor reduces "
                "the per-pixel mask to a single most-common class index."
            )
        },
    )


@dataclass
class ModelArguments:
    """Canonical model configuration (per plugin template)."""

    model_name_or_path: str = field(
        default="timm/mobilenetv3_small_100.lamb_in1k",
        metadata={
            "help": (
                "~2.5M-param timm surrogate; canonical default per the "
                "plugin's directive #6 recommendation for OD/IC models "
                "under 100M params on t4-small."
            )
        },
    )
    config_name: str | None = field(default=None)
    image_processor_name: str | None = field(default=None)
    cache_dir: str | None = field(default=None)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    trust_remote_code: bool = field(default=False)
    ignore_mismatched_sizes: bool = field(default=True)


# --------------------------------------------------------------------------
# Per-frame mask → image-level label reduction (Hinton distillation step)
# --------------------------------------------------------------------------


def _reduce_mask_to_image_level_class(mask: Any) -> int:
    """Reduce per-pixel SegNet mask to a single image-level class index.

    Per Hinton 2014 knowledge-distillation discipline: the surrogate learns
    the soft per-frame class signature; the "most common class index" is
    a reasonable hard-label image-level reduction for the
    AutoModelForImageClassification template. Future iterations may
    upgrade to soft-label cross-entropy with the per-pixel SegNet logits
    as teacher signal.

    Accepts numpy ndarray, PIL Image, or torch Tensor; returns int in
    ``range(5)``.
    """

    if hasattr(mask, "numpy"):
        arr = mask.numpy()
    elif hasattr(mask, "size") and hasattr(mask, "getdata"):
        arr = np.array(mask)
    elif isinstance(mask, np.ndarray):
        arr = mask
    else:
        arr = np.asarray(mask)
    arr = arr.astype("int64").ravel()
    if arr.size == 0:
        return 0
    bincount = np.bincount(arr, minlength=5)
    return int(np.argmax(bincount))


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")

    # Plugin directive #2: inject HF_TOKEN AFTER args parse, BEFORE Trainer
    # init. The HF Jobs runtime sets HF_TOKEN env var via the dispatcher's
    # `secrets={"HF_TOKEN": <actual token>}` block.
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and training_args.push_to_hub:
        training_args.hub_token = hf_token
        logger.info("Injected HF_TOKEN into training_args.hub_token (plugin directive #2)")

    # Load dataset (Hub-backed; pulls from
    # adpena/comma-video-segnet-image-level-600pairs by default).
    raw = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
    )

    if "train" not in raw:
        raise RuntimeError(
            f"Dataset {data_args.dataset_name!r} has no train split; "
            "got splits: " + ", ".join(raw.keys())
        )

    # 600-pair dataset — split off validation fraction if no eval split.
    if "validation" in raw:
        train_ds = raw["train"]
        eval_ds = raw["validation"]
    else:
        split = raw["train"].train_test_split(
            test_size=data_args.train_val_split, seed=training_args.seed
        )
        train_ds = split["train"]
        eval_ds = split["test"]

    if data_args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(len(train_ds), data_args.max_train_samples)))
    if data_args.max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(len(eval_ds), data_args.max_eval_samples)))

    # Image processor + model
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=5,  # canonical SegNet class count per upstream/modules.py:105
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # Image preprocessing — per plugin template, use the image_processor's
    # canonical size + normalization to match the timm pretrained backbone's
    # expected input distribution.
    size = (
        (image_processor.size["height"], image_processor.size["width"])
        if "height" in image_processor.size
        else (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])
    )
    normalize = (
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        if hasattr(image_processor, "image_mean") and image_processor.image_mean is not None
        else Compose([])
    )
    train_transforms = Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])
    eval_transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])

    def _preprocess(batch: dict[str, Any], transforms: Compose) -> dict[str, Any]:
        images = [img.convert("RGB") for img in batch[data_args.image_column_name]]
        labels = [_reduce_mask_to_image_level_class(m) for m in batch[data_args.label_column_name]]
        out = {
            "pixel_values": [transforms(img) for img in images],
            "labels": labels,
        }
        return out

    train_ds = train_ds.with_transform(
        lambda b: _preprocess(b, train_transforms)
    )
    eval_ds = eval_ds.with_transform(
        lambda b: _preprocess(b, eval_transforms)
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(p: EvalPrediction) -> dict[str, float]:
        preds = np.argmax(p.predictions, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=eval_ds if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=image_processor,
        data_collator=DefaultDataCollator(),
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_ds)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
