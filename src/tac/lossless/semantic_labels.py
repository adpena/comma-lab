# SPDX-License-Identifier: MIT
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


def _finite_1d(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _mean_abs(values) -> float:
    finite = _finite_1d(values)
    if finite.size == 0:
        return 0.0
    return float(np.mean(np.abs(finite)))


def _std_abs(values) -> float:
    finite = _finite_1d(values)
    if finite.size == 0:
        return 0.0
    return float(np.std(np.abs(finite)))


def _mean_abs_delta(values) -> float:
    finite = _finite_1d(values)
    if finite.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(finite))))


def _stop_go_bucket(forward_values) -> int:
    finite = np.abs(_finite_1d(forward_values))
    if finite.size == 0:
        return 0
    stop_fraction = float(np.mean(finite < 0.5))
    go_fraction = float(np.mean(finite >= 8.0))
    peak = float(np.max(finite, initial=0.0))
    if peak < 2.0:
        return 0
    if stop_fraction >= 0.2 and go_fraction >= 0.2:
        return 3
    if go_fraction >= 0.5:
        return 2
    return 1


def _turn_regime_bucket(yaw_values) -> int:
    finite = _finite_1d(yaw_values)
    if finite.size == 0:
        return 0
    signed_mean = float(np.mean(finite))
    turn_strength = float(np.mean(np.abs(finite)))
    if turn_strength < 0.05:
        return 0
    has_left = bool(np.any(finite <= -0.05))
    has_right = bool(np.any(finite >= 0.05))
    if has_left and has_right:
        return 3
    if signed_mean < 0.0:
        return 1
    return 2


def _motion_variance_bucket(forward_values) -> int:
    return _bucket(_std_abs(forward_values), thresholds=(0.25, 2.0, 8.0))


def _jerk_bucket(forward_values, yaw_values) -> int:
    forward_delta = _mean_abs_delta(forward_values)
    yaw_delta = 20.0 * _mean_abs_delta(yaw_values)
    jerk_score = max(forward_delta, yaw_delta)
    return _bucket(jerk_score, thresholds=(0.5, 2.0, 6.0))


def pose_label_vector(pose) -> list[int]:
    arr = np.asarray(pose, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError("pose array must have shape (frames, 6)")

    finite = np.where(np.isfinite(arr), arr, np.nan)
    forward = finite[:, 0]
    lateral = finite[:, 1]
    vertical = finite[:, 2]
    yaw = finite[:, 5]

    return [
        _bucket(_mean_abs(forward), thresholds=(2.0, 8.0, 16.0)),
        _bucket(_mean_abs(lateral), thresholds=(0.10, 0.40, 0.80)),
        _bucket(_mean_abs(vertical), thresholds=(0.02, 0.08, 0.16)),
        _bucket(_mean_abs(yaw), thresholds=(0.04, 0.18, 0.35)),
        _stop_go_bucket(forward),
        _turn_regime_bucket(yaw),
        _motion_variance_bucket(forward),
        _jerk_bucket(forward, yaw),
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
    target.write_text(json.dumps(label_map, indent=2, sort_keys=True) + "\n")
    return {
        "command": "lossless_pose_labels_sample",
        "output_path": str(target),
        "record_count": len(label_map),
        "split": list(split) if isinstance(split, (list, tuple)) else (["challenge"] if split is None else [str(split)]),
    }
