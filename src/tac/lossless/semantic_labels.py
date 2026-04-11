from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import load_commavq_dataset


def _bucket(value: float, *, thresholds: tuple[float, ...]) -> int:
    for index, threshold in enumerate(thresholds):
        if value < threshold:
            return index
    return len(thresholds)


def pose_label_vector(pose) -> list[int]:
    arr = np.asarray(pose, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError("pose array must have shape (frames, 6)")

    finite = np.where(np.isfinite(arr), arr, np.nan)
    means = np.nanmean(np.abs(finite), axis=0)
    means = np.where(np.isfinite(means), means, 0.0)

    return [
        _bucket(float(means[0]), thresholds=(4.0, 8.0, 12.0)),
        _bucket(float(means[1]), thresholds=(0.25, 0.75, 1.5)),
        _bucket(float(means[2]), thresholds=(0.25, 0.75, 1.5)),
        _bucket(float(np.nanmean(means[3:])), thresholds=(0.05, 0.15, 0.35)),
    ]


def build_pose_label_map_sample(
    *,
    output_path: str | Path,
    split=None,
    max_records: int = 64,
    dataset_loader=None,
) -> dict[str, object]:
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    dataset = load_commavq_dataset(split=split, dataset_loader=dataset_loader, num_proc=1)
    train = dataset["train"]
    if hasattr(train, "__getitem__"):
        examples = [train[index] for index in range(max_records)]
    else:
        examples = []
        for example in train:
            examples.append(example)
            if len(examples) >= max_records:
                break

    label_map: dict[str, list[int]] = {}
    for example in examples:
        file_name = str(example["json"]["file_name"])
        label_map[file_name] = pose_label_vector(example["pose.npy"])

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(label_map, indent=2) + "\n")
    return {
        "command": "lossless_pose_labels_sample",
        "output_path": str(target),
        "record_count": len(label_map),
        "split": list(split) if isinstance(split, (list, tuple)) else (["challenge"] if split is None else [str(split)]),
    }
