#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only parsing and math for preserved v10 blocked evidence.

This module deliberately has no resume, scratch-write, certification, cleanup,
unlink, or recursive-removal surface. It parses immutable checkpoint state and
performs deterministic fitting, quotient convolution, and strict PDW1 byte
accounting only.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Final

import brotli
import numpy as np
import torch
import torch.nn.functional as torch_functional

from tac.boundary_math.power_diagram_witness import (
    PowerDiagramTarget,
    PowerDiagramWitnessError,
    decode_pdw1,
    encode_pdw1,
    make_power_diagram_target,
)

CHECKPOINT_SCHEMA: Final = "v10_power_diagram_generator_byteclose_progress.v1"
CUSTODY_DERIVATION: Final = "LIVE_CANONICAL_FILE_AND_CANONICAL_EQUATION_REDERIVATION_SUPERSEDES_STALE_PROMPT_LITERALS"
SCRATCH_MARKER_NAME: Final = ".v10_power_diagram_byteclose_scratch"
SCRATCH_MARKER_BYTES: Final = b"v10_power_diagram_byteclose_scratch_v1\n"
FEATURE_CACHE_NAME: Final = "quotient_features.f32.npy"
PROGRESS_CHECKPOINT_NAME: Final = "extraction_progress.json"
EXPECTED_PAIRS: Final = 600
EXPECTED_SEG_HW: Final = (384, 512)
EXPECTED_CAMERA_HWC: Final = (874, 1164, 3)
EXPECTED_CLASSES: Final = 5
EXPECTED_HEAD_RANK: Final = 4
DEFAULT_TORCH_THREADS: Final = 6
DEFAULT_TORCH_INTEROP_THREADS: Final = 18
PINNED_GT_CACHE_SHA256: Final = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
PINNED_SEGNET_SHA256: Final = "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
PINNED_MODULES_SHA256: Final = "065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa"
PINNED_FRAME_UTILS_SHA256: Final = "d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90"
SSD_ROOTS: Final = (
    "/Volumes/VertigoDataTier/pact",
    "/Volumes/APDataStore/pact",
)


@dataclass
class StreamingRidgeSufficientStatistics:
    feature_dim: int
    n_classes: int

    def __post_init__(self) -> None:
        if self.feature_dim < 1 or self.n_classes < 2:
            raise ValueError("feature_dim must be positive and n_classes at least two")
        augmented = self.feature_dim + 1
        self.gram = np.zeros((augmented, augmented), dtype=np.float64)
        self.rhs = np.zeros((augmented, self.n_classes), dtype=np.float64)
        self.label_counts = np.zeros(self.n_classes, dtype=np.int64)
        self.sample_count = 0

    def update(self, features: np.ndarray, labels: np.ndarray) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels)
        if x.ndim != 2 or x.shape[1] != self.feature_dim or x.shape[0] == 0:
            raise ValueError("features must be nonempty (N, feature_dim)")
        if y.shape != (x.shape[0],) or y.dtype.kind not in "iu":
            raise ValueError("labels must be an integer vector paired with features")
        if not np.isfinite(x).all() or np.any(y < 0) or np.any(y >= self.n_classes):
            raise ValueError("statistics input contains non-finite features or invalid labels")
        total_x = np.sum(x, axis=0, dtype=np.float64)
        count = int(x.shape[0])
        d = self.feature_dim
        self.gram[:d, :d] += x.T @ x
        self.gram[:d, d] += total_x
        self.gram[d, :d] += total_x
        self.gram[d, d] += count
        total_augmented = np.concatenate((total_x, np.array([float(count)])))
        self.rhs -= total_augmented[:, None] / self.n_classes
        for class_index in range(self.n_classes):
            selected = y == class_index
            class_count = int(np.count_nonzero(selected))
            if class_count:
                self.rhs[:d, class_index] += np.sum(x[selected], axis=0, dtype=np.float64)
                self.rhs[d, class_index] += class_count
            self.label_counts[class_index] += class_count
        self.sample_count += count

    def solve(self, regularization: float) -> tuple[np.ndarray, np.ndarray]:
        ridge = float(regularization)
        if self.sample_count < 1:
            raise ValueError("cannot solve empty sufficient statistics")
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("regularization must be finite and positive")
        normal = self.gram + ridge * np.eye(self.feature_dim + 1, dtype=np.float64)
        coefficients = np.linalg.solve(normal, self.rhs)
        return coefficients[:-1].T, coefficients[-1]


@dataclass
class ExtractionState:
    next_frame: int
    statistics: StreamingRidgeSufficientStatistics
    adjacency: set[tuple[int, int]]
    positive_power_mismatches: int = 0
    positive_forward_mismatches: int = 0
    status: str = "extracting"
    blocked_reason: str | None = None


def _strict_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"checkpoint {name} must be an integer >= {minimum}")
    return value


def _strict_float_matrix(value: Any, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError(f"checkpoint {name} must have shape {shape}")
    for row in value:
        if not isinstance(row, list) or len(row) != shape[1]:
            raise ValueError(f"checkpoint {name} must have shape {shape}")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in row):
            raise ValueError(f"checkpoint {name} entries must be JSON numbers")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"checkpoint {name} must have finite shape {shape}")
    return array


def validate_extraction_checkpoint(payload: dict[str, Any], *, expected_identity: dict[str, Any]) -> ExtractionState:
    expected_keys = {
        "schema",
        "status",
        "next_canonical_frame",
        "immutable_identity",
        "statistics",
        "adjacency",
        "positive_control",
        "blocked_reason",
        "updated_utc",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("checkpoint top-level keys are noncanonical")
    if payload["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    if payload["immutable_identity"] != expected_identity:
        raise ValueError("checkpoint immutable identity mismatch")
    geometry = expected_identity.get("geometry")
    config = expected_identity.get("config")
    if not isinstance(geometry, dict) or not isinstance(config, dict):
        raise ValueError("checkpoint identity lacks geometry/config")
    expected_pairs = _strict_integer(geometry.get("expected_pairs"), name="expected_pairs", minimum=1)
    n_classes = _strict_integer(geometry.get("n_classes"), name="n_classes", minimum=2)
    head_rank = _strict_integer(geometry.get("head_rank"), name="head_rank", minimum=1)
    seg_hw = geometry.get("seg_hw")
    if (
        not isinstance(seg_hw, list)
        or len(seg_hw) != 2
        or any(type(value) is not int or value <= 0 for value in seg_hw)
    ):
        raise ValueError("checkpoint identity seg_hw is invalid")
    if config.get("batch_size") != 1:
        raise ValueError("checkpoint batch size drift")
    status = payload["status"]
    if status not in {"extracting", "extraction_complete", "blocked"}:
        raise ValueError("checkpoint status is invalid")
    next_frame = _strict_integer(payload["next_canonical_frame"], name="next_canonical_frame")
    if next_frame > expected_pairs:
        raise ValueError("checkpoint next frame exceeds geometry")
    statistics_payload = payload["statistics"]
    if not isinstance(statistics_payload, dict) or set(statistics_payload) != {
        "gram",
        "rhs",
        "label_counts",
        "sample_count",
    }:
        raise ValueError("checkpoint statistics keys are noncanonical")
    augmented = head_rank + 1
    gram = _strict_float_matrix(statistics_payload["gram"], name="gram", shape=(augmented, augmented))
    rhs = _strict_float_matrix(statistics_payload["rhs"], name="rhs", shape=(augmented, n_classes))
    raw_counts = statistics_payload["label_counts"]
    if not isinstance(raw_counts, list) or len(raw_counts) != n_classes:
        raise ValueError("checkpoint label_counts shape is invalid")
    counts = [_strict_integer(value, name="label count") for value in raw_counts]
    sample_count = _strict_integer(statistics_payload["sample_count"], name="sample_count")
    if sample_count != next_frame * math.prod(seg_hw) or sum(counts) != sample_count:
        raise ValueError("checkpoint sample counts disagree with prefix geometry")
    raw_adjacency = payload["adjacency"]
    if not isinstance(raw_adjacency, list):
        raise ValueError("checkpoint adjacency must be a list")
    edges: list[tuple[int, int]] = []
    for edge in raw_adjacency:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("checkpoint adjacency edge is malformed")
        i = _strict_integer(edge[0], name="adjacency class")
        j = _strict_integer(edge[1], name="adjacency class")
        if not 0 <= i < j < n_classes:
            raise ValueError("checkpoint adjacency edge is noncanonical")
        edges.append((i, j))
    if edges != sorted(set(edges)):
        raise ValueError("checkpoint adjacency must be sorted and unique")
    positive = payload["positive_control"]
    if not isinstance(positive, dict) or set(positive) != {
        "power_target_mismatch_count",
        "cpu_torch_forward_mismatch_count",
    }:
        raise ValueError("checkpoint positive-control keys are noncanonical")
    power_count = _strict_integer(positive["power_target_mismatch_count"], name="power mismatch")
    forward_count = _strict_integer(positive["cpu_torch_forward_mismatch_count"], name="forward mismatch")
    reason = payload["blocked_reason"]
    if status == "blocked":
        if not isinstance(reason, str) or not reason or power_count + forward_count < 1:
            raise ValueError("blocked checkpoint lacks blocker custody")
    elif reason is not None or power_count or forward_count:
        raise ValueError("non-blocked checkpoint carries blocker custody")
    if not isinstance(payload["updated_utc"], str) or not payload["updated_utc"]:
        raise ValueError("checkpoint updated_utc is invalid")
    statistics = StreamingRidgeSufficientStatistics(head_rank, n_classes)
    statistics.gram[...] = gram
    statistics.rhs[...] = rhs
    statistics.label_counts[...] = np.asarray(counts, dtype=np.int64)
    statistics.sample_count = sample_count
    return ExtractionState(
        next_frame=next_frame,
        statistics=statistics,
        adjacency=set(edges),
        positive_power_mismatches=power_count,
        positive_forward_mismatches=forward_count,
        status=status,
        blocked_reason=reason,
    )


def affine_scores_to_power_target(
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    adjacency: tuple[tuple[int, int], ...],
) -> PowerDiagramTarget:
    rows = np.asarray(weight, dtype=np.float64)
    offsets = np.asarray(bias, dtype=np.float64)
    if rows.ndim != 2 or offsets.shape != (rows.shape[0],):
        raise ValueError("affine weight/bias shapes are inconsistent")
    if not np.isfinite(rows).all() or not np.isfinite(offsets).all():
        raise ValueError("affine scores contain non-finite values")
    centered_rows = rows - rows.mean(axis=0, keepdims=True)
    centered_bias = offsets - offsets.mean()
    sites = centered_rows / 2.0
    ungauged = centered_bias + np.sum(sites * sites, axis=1)
    weights = ungauged - ungauged.mean()
    return make_power_diagram_target(sites, weights, adjacency=adjacency)


def quotient_convolution(
    feature_map: torch.Tensor,
    quotient_basis: np.ndarray,
    *,
    head_weight_shape: tuple[int, int, int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> torch.Tensor:
    if feature_map.ndim != 4 or feature_map.shape[0] != 1:
        raise ValueError("feature_map must have batch-one geometry")
    _classes, channels_per_group, kernel_h, kernel_w = head_weight_shape
    basis = np.asarray(quotient_basis, dtype=np.float64)
    if basis.shape[0] != channels_per_group * kernel_h * kernel_w:
        raise ValueError("quotient basis rows do not match final head")
    filters = torch.from_numpy(basis.T.reshape(basis.shape[1], channels_per_group, kernel_h, kernel_w)).to(
        device=feature_map.device, dtype=feature_map.dtype
    )
    return torch_functional.conv2d(
        feature_map,
        filters,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def order0_ideal_entropy_estimate(payload: bytes) -> dict[str, Any]:
    if not payload:
        raise ValueError("order-0 estimate requires nonempty bytes")
    counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
    nonzero = counts[counts > 0].astype(np.float64)
    total = float(len(payload))
    entropy = float(-np.sum((nonzero / total) * np.log2(nonzero / total)))
    ideal_bits = entropy * total
    return {
        "label": "DERIVED_OPTIMISTIC_ROUNDED_UP_IDEAL_ENTROPY_BYTES",
        "assumptions": "empirical PMF free; no model/header/termination overhead",
        "entropy_bits_per_byte_symbol": entropy,
        "ideal_bits": ideal_bits,
        "rounded_up_ideal_entropy_bytes": math.ceil(ideal_bits / 8.0),
    }


def compression_accounting(target: PowerDiagramTarget) -> dict[str, Any]:
    raw = encode_pdw1(target)
    decoded = decode_pdw1(raw)
    if encode_pdw1(decoded) != raw:
        raise PowerDiagramWitnessError("strict PDW1 decode/re-encode identity failed")
    compressed = brotli.compress(raw, quality=11)
    return {
        "pdw1_hex": raw.hex(),
        "raw": {
            "label": "MEASURED_ACTUAL_PDW1_BYTES",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "brotli_quality11": {
            "label": "MEASURED_ACTUAL_BROTLI_QUALITY11_BYTES",
            "quality": 11,
            "bytes": len(compressed),
            "sha256": hashlib.sha256(compressed).hexdigest(),
            "payload_hex": compressed.hex(),
        },
        "order0_ideal_entropy_estimate": order0_ideal_entropy_estimate(raw),
        "strict_parseback_byte_identical": True,
    }


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CUSTODY_DERIVATION",
    "DEFAULT_TORCH_INTEROP_THREADS",
    "DEFAULT_TORCH_THREADS",
    "EXPECTED_CAMERA_HWC",
    "EXPECTED_CLASSES",
    "EXPECTED_HEAD_RANK",
    "EXPECTED_PAIRS",
    "EXPECTED_SEG_HW",
    "FEATURE_CACHE_NAME",
    "PINNED_FRAME_UTILS_SHA256",
    "PINNED_GT_CACHE_SHA256",
    "PINNED_MODULES_SHA256",
    "PINNED_SEGNET_SHA256",
    "PROGRESS_CHECKPOINT_NAME",
    "SCRATCH_MARKER_BYTES",
    "SCRATCH_MARKER_NAME",
    "SSD_ROOTS",
    "ExtractionState",
    "StreamingRidgeSufficientStatistics",
    "affine_scores_to_power_target",
    "compression_accounting",
    "order0_ideal_entropy_estimate",
    "quotient_convolution",
    "validate_extraction_checkpoint",
]
