#!/usr/bin/env python3
"""Measure a counted implicit carrier for HG1's exact categorical residual.

The carrier is a small coordinate INR conditioned on the already-decoded HG1
generator class.  It has deterministic Fourier coordinates, counted per-pair
modulation, and six per-class actions (keep the generator output, or emit one
of five classes).  The action argmax therefore implies evaluator cells; it
does not enumerate boundaries or cell addresses.  A unique-home residual then
returns every candidate to the exact retained DX2 categorical field.

Every trained checkpoint, quantized model, real-coder output, candidate field,
residual stream, receiver packet, archive, and parse-back field is retained.
The tool is CPU-only and scorer-free.  Exact categorical identity inherits the
current object's distortion but is not a new score measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import ddm_et1_edge_topology_container_gate as et1
import ddm_hg1_heterogeneous_analytic_generator_gate as hg1
import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.nn import functional as F

from tac.boundary_math.lever_b_generator import (
    build_coords,
    deterministic_fourier_B,
    residual_component_stats,
)

AXIS: Final = "[macOS-CPU advisory / scorer-free exact byte measurement]"
SCHEMA: Final = "ddm_hr3_residual_implicit_carrier.v1"
SEED: Final = 20260823
TOKEN_SHAPE: Final = hg1.TOKEN_SHAPE
N_PAIRS, HEIGHT, WIDTH = TOKEN_SHAPE
FRAME_POSITIONS: Final = HEIGHT * WIDTH
TOTAL_POSITIONS: Final = int(np.prod(TOKEN_SHAPE))

SOURCE_ARCHIVE_BYTES: Final = 180_368
SOURCE_ARCHIVE_SHA256: Final = (
    "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
)
SOURCE_TOKENS_BYTES: Final = 117_964_800
SOURCE_TOKENS_SHA256: Final = (
    "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
)
HG1_GENERATED_BYTES: Final = 117_964_800
HG1_GENERATED_SHA256: Final = (
    "2884c5701dc2b2059df0e9f8e4ee3ed81809457b127a48ad3fd3fb6f7a17152b"
)
HG1_EXACT_RESIDUAL_BYTES: Final = 359_280
HG1_EXACT_CORRECTIONS: Final = 1_334_939
BL1_COST_BYTES: Final = 943_718_400
BL1_COST_SHA256: Final = (
    "99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86"
)
RESIDUAL_TARGET_BYTES: Final = 36_858
CONTAINER_TARGET_BYTES: Final = 137_986
HG1_NON_RESIDUAL_BASE_BYTES: Final = 101_128
HG1_BASE_FRAMING_BYTES: Final = 499
RATE_EXCHANGE_S_PER_BYTE: Final = 6.658590e-07
RESUME_COMPATIBLE_RUNNER_SHA256: Final = {
    "b51284e210259167b4a864406cc8dd6b304b6a58f86e10e897b79740fb23559c": (
        "replace broken large float32 NumPy matmul with deterministic finite einsum "
        "and batch bias variants without changing trained checkpoints or payload semantics"
    ),
    "631227dd0d0ad71e96706bf9e0ee52a0f8f53c4abda9eee3bb80c78896760a62": (
        "add a declared fail-closed full-field width bound after APDataStore free space "
        "fell below the retained full-ladder requirement; completed rows are unchanged"
    ),
}

MODEL_MAGIC: Final = b"HRI1"
MODEL_VERSION: Final = 1
MODEL_HEADER: Final = struct.Struct("<4sBI")

PACKET_MAGIC: Final = b"HR3P"
PACKET_VERSION: Final = 1
PACKET_HEADER: Final = struct.Struct("<4sBBH")
PACKET_ROW: Final = struct.Struct("<BBII32s32s")
STREAMS: Final = (*hg1.GENERATOR_STREAMS, "implicit_model", "residual")
STREAM_IDS: Final = {name: index + 1 for index, name in enumerate(STREAMS)}
ID_STREAMS: Final = {value: key for key, value in STREAM_IDS.items()}

COMPLETE_MAGIC: Final = b"HR3C"
COMPLETE_VERSION: Final = 1
COMPLETE_HEADER: Final = struct.Struct("<4sBIIII")

BIAS_VALUES: Final = (3.5, 4.5, 5.5, 6.5)
MODEL_SPECS: Final = (
    (8, 4),
    (16, 4),
    (32, 8),
    (64, 8),
    (96, 8),
    (128, 8),
)
FULL_FIELD_MAX_WIDTH: Final = 96
N_FOURIER: Final = 16
N_HIDDEN: Final = 2
BASE_EMBED_DIM: Final = 4
FOURIER_SIGMA: Final = 8.0
TRAIN_EPOCHS: Final = 4
TRAIN_BATCH: Final = 16_384
CORRECT_SAMPLE_MULTIPLIER: Final = 2


class HR3Error(RuntimeError):
    """A provenance, training, payload, receiver, or accounting invariant failed."""


@dataclass(frozen=True)
class ImplicitConfig:
    """Complete counted model configuration."""

    width: int
    mod_dim: int
    n_fourier: int = N_FOURIER
    n_hidden: int = N_HIDDEN
    base_embed_dim: int = BASE_EMBED_DIM
    fourier_sigma: float = FOURIER_SIGMA
    n_actions: int = 6
    n_pairs: int = N_PAIRS


class ResidualActionINR(nn.Module):
    """V8-shaped coordinate INR whose argmax emits one of six class actions."""

    def __init__(self, cfg: ImplicitConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mod = nn.Embedding(cfg.n_pairs, cfg.mod_dim)
        self.base_embedding = nn.Embedding(5, cfg.base_embed_dim)
        self.in_proj = nn.Linear(2 * cfg.n_fourier + cfg.base_embed_dim, cfg.width)
        self.hidden = nn.ModuleList(
            [nn.Linear(cfg.width, cfg.width) for _ in range(cfg.n_hidden)]
        )
        self.film = nn.Linear(cfg.mod_dim, cfg.n_hidden * 2 * cfg.width)
        self.out = nn.Linear(cfg.width, cfg.n_actions)

    def forward(
        self,
        coords: torch.Tensor,
        base_class: torch.Tensor,
        pair_index: torch.Tensor,
        fourier_b: torch.Tensor,
    ) -> torch.Tensor:
        proj = coords @ fourier_b
        features = torch.cat(
            (torch.sin(proj), torch.cos(proj), self.base_embedding(base_class)), dim=-1
        )
        hidden = F.relu(self.in_proj(features))
        film = self.film(self.mod(pair_index)).reshape(
            -1, self.cfg.n_hidden, 2, self.cfg.width
        )
        for layer_index, layer in enumerate(self.hidden):
            scale = 1.0 + film[:, layer_index, 0]
            shift = film[:, layer_index, 1]
            hidden = F.relu(layer(hidden) * scale + shift)
        return self.out(hidden)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    return et1.sha256_path(path)


def current_git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def fact_matches(path: Path, fact: object) -> bool:
    return hg1.fact_matches(path, fact)


def retained_facts_valid(value: object) -> tuple[bool, int]:
    """Validate every nested retained-file fact and report how many were found."""

    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            try:
                path = Path(str(value["path"]))
            except (TypeError, ValueError):
                return False, 1
            return fact_matches(path, value), 1
        valid = True
        count = 0
        for child in value.values():
            child_valid, child_count = retained_facts_valid(child)
            valid = valid and child_valid
            count += child_count
        return valid, count
    if isinstance(value, list):
        valid = True
        count = 0
        for child in value:
            child_valid, child_count = retained_facts_valid(child)
            valid = valid and child_valid
            count += child_count
        return valid, count
    return True, 0


def candidate_row_is_resumable(row: object) -> bool:
    """Return whether a completed candidate row still has all retained payloads."""

    if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str):
        return False
    valid, count = retained_facts_valid(row)
    if not valid or count == 0:
        return False
    if row.get("receiver_closed"):
        required = (
            "implicit_field",
            "model",
            "residual_orders",
            "packet",
            "complete_archive",
            "complete_archive_repeat",
            "archive_parseback_tokens",
            "container_framing",
        )
        return all(key in row for key in required)
    if row.get("type") == "BUILT_MODEL_ONLY_BYTE_LOWER_BOUND":
        return "model" in row
    if row.get("type") == "BUILT_ARCHIVE_PACKET_PARSED_PENDING_RECEIVER_SELECTION":
        required = (
            "implicit_field",
            "model",
            "residual_orders",
            "packet",
            "complete_archive",
            "complete_archive_repeat",
            "container_framing",
        )
        return all(key in row for key in required)
    return False


def require_file(path: Path, *, byte_count: int, sha256: str, label: str) -> None:
    hg1.require_file(path, byte_count=byte_count, sha256=sha256, label=label)


def load_manifest(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or loaded.get("schema") != SCHEMA:
        raise HR3Error(f"resume manifest schema mismatch: {path}")
    return loaded


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    manifest["updated_at_unix"] = time.time()
    atomic_json(path, manifest)


def storage_preflight(output_root: Path, minimum_free_bytes: int) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root)
    if usage.free < minimum_free_bytes:
        raise HR3Error(
            f"storage preflight refused: {output_root} has {usage.free} free bytes; "
            f"requires {minimum_free_bytes}"
        )
    return {
        "path": str(output_root),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes_before": usage.free,
        "minimum_free_bytes": minimum_free_bytes,
    }


def copy_exact(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or sha256_path(destination) != sha256_path(source):
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    if sha256_path(destination) != sha256_path(source):
        raise HR3Error(f"copy identity failure: {source} -> {destination}")
    return et1.file_fact(destination)


def entropy_bits(counts: np.ndarray) -> float:
    values = np.asarray(counts, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values[values > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def gini(values: np.ndarray) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.size == 0 or float(array.sum()) == 0.0:
        return 0.0
    indices = np.arange(1, array.size + 1, dtype=np.float64)
    return float((2 * np.dot(indices, array) / (array.size * array.sum())) - (array.size + 1) / array.size)


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:-1] |= labels[:-1] != labels[1:]
    boundary[1:] |= labels[:-1] != labels[1:]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return boundary


def characterize_residual(
    source: np.ndarray,
    generated: np.ndarray,
    top1_path: Path,
    bl1_cost_path: Path,
    output_root: Path,
) -> dict[str, object]:
    retained = output_root / "retained" / "characterization"
    retained.mkdir(parents=True, exist_ok=True)
    mask_path = retained / "hg1_residual_mask.n600.packbits"
    top1_overlap_path = retained / "hg1_residual_top1_overlap.n600.packbits"
    component_path = retained / "component_rows.jsonl"
    frame_path = retained / "frame_rows.jsonl"

    mismatch_counts = np.zeros(N_PAIRS, dtype=np.int64)
    confusion = np.zeros((5, 5), dtype=np.int64)
    target_counts = np.zeros(5, dtype=np.int64)
    source_counts = np.zeros(5, dtype=np.int64)
    boundary_bins = np.zeros(8, dtype=np.int64)
    component_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    residual_mask = np.memmap(
        retained / ".residual_mask.bool.tmp", mode="w+", dtype=np.bool_, shape=TOKEN_SHAPE
    )
    residual_targets = np.zeros(5, dtype=np.int64)

    distance_edges = np.asarray((0, 1, 2, 4, 8, 16, 32), dtype=np.float64)
    for pair in range(N_PAIRS):
        source_frame = np.asarray(source[pair])
        generated_frame = np.asarray(generated[pair])
        mismatch = source_frame != generated_frame
        residual_mask[pair] = mismatch
        mismatch_count = int(np.count_nonzero(mismatch))
        mismatch_counts[pair] = mismatch_count
        source_counts += np.bincount(source_frame.reshape(-1), minlength=5)
        if mismatch_count:
            bases = generated_frame[mismatch].astype(np.int64)
            targets = source_frame[mismatch].astype(np.int64)
            np.add.at(confusion, (bases, targets), 1)
            bincount = np.bincount(targets, minlength=5)
            target_counts += bincount
            residual_targets += bincount
        stats = residual_component_stats(generated_frame, source_frame)
        component_rows.append({"pair": pair, **stats.to_dict()})
        distances = ndimage.distance_transform_edt(~boundary_mask(generated_frame))
        residual_distances = distances[mismatch]
        bins = np.searchsorted(distance_edges, residual_distances, side="right")
        boundary_bins += np.bincount(bins, minlength=8)[:8]
        frame_rows.append(
            {
                "pair": pair,
                "mismatch_positions": mismatch_count,
                "mismatch_fraction": mismatch_count / FRAME_POSITIONS,
                "target_class_counts": np.bincount(
                    source_frame[mismatch].astype(np.int64), minlength=5
                ).tolist(),
            }
        )
    residual_mask.flush()

    packed = np.packbits(np.asarray(residual_mask).reshape(-1), bitorder="little")
    et1.atomic_bytes(mask_path, packed.tobytes())
    top1 = hg1.unpack_mask(
        top1_path, expected_bytes=hg1.TOP1_BYTES, expected_sha256=hg1.TOP1_SHA256
    )
    overlap = np.asarray(residual_mask) & top1
    et1.atomic_bytes(
        top1_overlap_path,
        np.packbits(overlap.reshape(-1), bitorder="little").tobytes(),
    )
    overlap_positions = int(np.count_nonzero(overlap))

    costs = np.memmap(bl1_cost_path, mode="r", dtype="<f8", shape=(TOTAL_POSITIONS,))
    mask_flat = np.asarray(residual_mask).reshape(-1)
    residual_model_bits = float(costs[mask_flat].sum(dtype=np.float64))
    top1_residual_model_bits = float(costs[overlap.reshape(-1)].sum(dtype=np.float64))

    temporal_rows = []
    for lag in (1, 2, 4, 8, 16, 32):
        left = np.asarray(residual_mask[:-lag])
        right = np.asarray(residual_mask[lag:])
        intersection = int(np.count_nonzero(left & right))
        union = int(np.count_nonzero(left | right))
        left_count = int(np.count_nonzero(left))
        same_target = 0
        if intersection:
            common = left & right
            same_target = int(
                np.count_nonzero(
                    np.asarray(source[:-lag])[common] == np.asarray(source[lag:])[common]
                )
            )
        temporal_rows.append(
            {
                "lag": lag,
                "intersection": intersection,
                "union": union,
                "jaccard": intersection / union if union else 0.0,
                "persistence_given_left": intersection / left_count if left_count else 0.0,
                "same_target_given_overlap": same_target / intersection if intersection else 0.0,
            }
        )

    with component_path.open("w", encoding="utf-8") as handle:
        for row in component_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with frame_path.open("w", encoding="utf-8") as handle:
        for row in frame_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    total_mismatch = int(mismatch_counts.sum())
    event_rate = total_mismatch / TOTAL_POSITIONS
    event_entropy = 0.0
    if 0.0 < event_rate < 1.0:
        event_entropy = -event_rate * math.log2(event_rate) - (1.0 - event_rate) * math.log2(1.0 - event_rate)
    label_entropy = entropy_bits(residual_targets)
    component_aggregate = {
        "n_pairs": N_PAIRS,
        "total_components": int(sum(int(row["n_components"]) for row in component_rows)),
        "single_pixel_components": int(
            sum(int(row["single_pixel_components"]) for row in component_rows)
        ),
        "contiguous_flip_positions": int(
            sum(round(float(row["contiguous_fraction"]) * int(row["n_flips"])) for row in component_rows)
        ),
        "largest_component_pixels": int(
            max(int(row["largest_component_pixels"]) for row in component_rows)
        ),
    }
    component_aggregate["single_pixel_component_fraction"] = (
        component_aggregate["single_pixel_components"] / component_aggregate["total_components"]
        if component_aggregate["total_components"]
        else 0.0
    )
    component_aggregate["contiguous_flip_fraction"] = (
        component_aggregate["contiguous_flip_positions"] / total_mismatch
        if total_mismatch
        else 0.0
    )

    temporary_bool = Path(residual_mask.filename)
    del costs, mask_flat, overlap, top1, residual_mask
    temporary_bool.unlink()
    return {
        "axis": AXIS,
        "denominator_positions": TOTAL_POSITIONS,
        "mismatch_positions": total_mismatch,
        "mismatch_fraction": event_rate,
        "confusion_base_to_target": confusion.tolist(),
        "target_class_counts": target_counts.tolist(),
        "source_class_counts": source_counts.tolist(),
        "entropy": {
            "event_bernoulli_bits_per_position": event_entropy,
            "target_label_bits_per_event": label_entropy,
            "iid_event_plus_label_floor_bytes": (
                TOTAL_POSITIONS * event_entropy + total_mismatch * label_entropy
            )
            / 8.0,
            "hg1_real_coded_bytes": HG1_EXACT_RESIDUAL_BYTES,
        },
        "spatial": {
            "component_aggregate": component_aggregate,
            "distance_to_generated_boundary_bins": {
                "distance_0": int(boundary_bins[1]),
                "distance_1": int(boundary_bins[2]),
                "distance_2_to_lt4": int(boundary_bins[3]),
                "distance_4_to_lt8": int(boundary_bins[4]),
                "distance_8_to_lt16": int(boundary_bins[5]),
                "distance_16_to_lt32": int(boundary_bins[6]),
                "distance_ge_32": int(boundary_bins[7]),
                "distance_negative_impossible_control": int(boundary_bins[0]),
            },
            "frame_mismatch_gini": gini(mismatch_counts),
            "frame_min": int(mismatch_counts.min()),
            "frame_median": float(np.median(mismatch_counts)),
            "frame_max": int(mismatch_counts.max()),
        },
        "temporal": temporal_rows,
        "bl1_join": {
            "top1_positions": hg1.BL1_TOP1_POSITIONS,
            "residual_positions_in_top1": overlap_positions,
            "fraction_of_residual_positions_in_top1": overlap_positions / total_mismatch,
            "fraction_of_top1_positions_in_residual": overlap_positions / hg1.BL1_TOP1_POSITIONS,
            "residual_incumbent_model_bits": residual_model_bits,
            "residual_incumbent_model_bytes": residual_model_bits / 8.0,
            "top1_overlap_incumbent_model_bits": top1_residual_model_bits,
            "fraction_residual_model_bits_in_top1": top1_residual_model_bits / residual_model_bits,
            "bl1_total_model_bits": hg1.BL1_BITS_TOTAL,
            "fraction_bl1_total_model_bits_on_hg1_residual": residual_model_bits / hg1.BL1_BITS_TOTAL,
        },
        "retained": {
            "residual_mask": et1.file_fact(mask_path),
            "top1_overlap_mask": et1.file_fact(top1_overlap_path),
            "component_rows": et1.file_fact(component_path),
            "frame_rows": et1.file_fact(frame_path),
        },
    }


def build_training_sample(
    source: np.ndarray,
    generated: np.ndarray,
    output_path: Path,
) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    pairs: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for pair in range(N_PAIRS):
        source_flat = np.asarray(source[pair]).reshape(-1)
        generated_flat = np.asarray(generated[pair]).reshape(-1)
        mismatch = source_flat != generated_flat
        mismatch_positions = np.flatnonzero(mismatch).astype(np.uint32)
        correct_positions = np.flatnonzero(~mismatch)
        correct_count = min(
            correct_positions.size,
            CORRECT_SAMPLE_MULTIPLIER * mismatch_positions.size,
        )
        sampled_correct = np.sort(
            rng.choice(correct_positions, size=correct_count, replace=False)
        ).astype(np.uint32)
        pair_positions = np.concatenate((mismatch_positions, sampled_correct))
        pair_actions = np.zeros(pair_positions.size, dtype=np.uint8)
        pair_actions[: mismatch_positions.size] = (
            source_flat[mismatch_positions].astype(np.uint8) + 1
        )
        pairs.append(np.full(pair_positions.size, pair, dtype=np.uint16))
        positions.append(pair_positions)
        actions.append(pair_actions)
    pair_array = np.concatenate(pairs)
    position_array = np.concatenate(positions)
    action_array = np.concatenate(actions)
    atomic_npz(
        output_path,
        pair=pair_array,
        position=position_array,
        action=action_array,
    )
    counts = np.bincount(action_array.astype(np.int64), minlength=6)
    return {
        **et1.file_fact(output_path),
        "samples": int(action_array.size),
        "action_counts": counts.tolist(),
        "all_hg1_mismatches_included": int(counts[1:].sum()) == HG1_EXACT_CORRECTIONS,
        "seed": SEED,
    }


def checkpoint_config(cfg: ImplicitConfig) -> dict[str, object]:
    return {
        **asdict(cfg),
        "seed": SEED,
        "train_epochs": TRAIN_EPOCHS,
        "batch": TRAIN_BATCH,
        "correct_sample_multiplier": CORRECT_SAMPLE_MULTIPLIER,
    }


def train_model(
    cfg: ImplicitConfig,
    sample_path: Path,
    generated: np.ndarray,
    model_root: Path,
) -> dict[str, object]:
    torch.manual_seed(SEED + cfg.width)
    np.random.seed(SEED + cfg.width)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = ResidualActionINR(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    config = checkpoint_config(cfg)
    config_sha = sha256_bytes(json.dumps(config, sort_keys=True).encode("utf-8"))
    start_epoch = 0
    history: list[dict[str, object]] = []
    checkpoint_facts: list[dict[str, object]] = []

    for epoch in range(TRAIN_EPOCHS, 0, -1):
        candidate = model_root / f"stage_epoch_{epoch:02d}.pt"
        if candidate.is_file():
            loaded = torch.load(candidate, map_location="cpu", weights_only=False)
            if loaded.get("config_sha256") != config_sha:
                raise HR3Error(f"checkpoint config drift: {candidate}")
            model.load_state_dict(loaded["model"])
            optimizer.load_state_dict(loaded["optimizer"])
            torch.set_rng_state(loaded["torch_rng_state"])
            np.random.set_state(loaded["numpy_rng_state"])
            start_epoch = int(loaded["epoch"])
            history = list(loaded["history"])
            break
    if start_epoch == 0:
        init_path = model_root / "stage_epoch_00_init.pt"
        atomic_torch_save(
            init_path,
            {
                "config": config,
                "config_sha256": config_sha,
                "epoch": 0,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": [],
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
            },
        )
        checkpoint_facts.append(et1.file_fact(init_path))

    with np.load(sample_path) as sample:
        pair_array = np.asarray(sample["pair"], dtype=np.int64)
        position_array = np.asarray(sample["position"], dtype=np.int64)
        action_array = np.asarray(sample["action"], dtype=np.int64)
    counts = np.bincount(action_array, minlength=6).astype(np.float64)
    class_weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    class_weights /= class_weights.mean()
    class_weights = np.clip(class_weights, 0.25, 8.0)
    weight_tensor = torch.from_numpy(class_weights.astype(np.float32))
    fourier_b = torch.from_numpy(
        deterministic_fourier_B(cfg.n_fourier, cfg.fourier_sigma)
    )

    for epoch in range(start_epoch + 1, TRAIN_EPOCHS + 1):
        permutation_rng = np.random.default_rng(SEED + cfg.width * 100 + epoch)
        permutation = permutation_rng.permutation(action_array.size)
        total_loss = 0.0
        total_seen = 0
        model.train()
        for begin in range(0, permutation.size, TRAIN_BATCH):
            indices = permutation[begin : begin + TRAIN_BATCH]
            pair_batch = pair_array[indices]
            position_batch = position_array[indices]
            y = position_batch // WIDTH
            x = position_batch % WIDTH
            coords = np.stack(
                (
                    2.0 * x / (WIDTH - 1) - 1.0,
                    2.0 * y / (HEIGHT - 1) - 1.0,
                ),
                axis=-1,
            ).astype(np.float32)
            addresses = pair_batch * FRAME_POSITIONS + position_batch
            base = np.asarray(generated).reshape(-1)[addresses].astype(np.int64)
            target = action_array[indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                torch.from_numpy(coords),
                torch.from_numpy(base),
                torch.from_numpy(pair_batch),
                fourier_b,
            )
            loss = F.cross_entropy(logits, torch.from_numpy(target), weight=weight_tensor)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * target.size
            total_seen += target.size
        row = {
            "epoch": epoch,
            "mean_weighted_cross_entropy": total_loss / total_seen,
            "samples": total_seen,
        }
        history.append(row)
        checkpoint_path = model_root / f"stage_epoch_{epoch:02d}.pt"
        atomic_torch_save(
            checkpoint_path,
            {
                "config": config,
                "config_sha256": config_sha,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
            },
        )
        checkpoint_facts.append(et1.file_fact(checkpoint_path))
        print(
            f"[hr3] trained width={cfg.width} epoch={epoch}/{TRAIN_EPOCHS} "
            f"loss={row['mean_weighted_cross_entropy']:.6f}",
            flush=True,
        )
    checkpoint_facts = [
        et1.file_fact(path)
        for path in sorted(model_root.glob("stage_epoch_*.pt"))
    ]
    return {
        "config": config,
        "config_sha256": config_sha,
        "history": history,
        "checkpoint_facts": checkpoint_facts,
        "final_checkpoint": et1.file_fact(model_root / f"stage_epoch_{TRAIN_EPOCHS:02d}.pt"),
    }


def load_final_state(cfg: ImplicitConfig, checkpoint_path: Path) -> dict[str, np.ndarray]:
    model = ResidualActionINR(cfg)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(loaded["model"])
    return {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in model.state_dict().items()
    }


def quantize_state(state: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, float]]:
    quantized: dict[str, tuple[np.ndarray, float]] = {}
    for name in sorted(state):
        array = np.asarray(state[name], dtype=np.float32)
        maximum = float(np.max(np.abs(array))) if array.size else 0.0
        scale = maximum / 127.0 if maximum > 0 else 1.0
        q = np.rint(array / scale).clip(-127, 127).astype(np.int8)
        quantized[name] = (q, scale)
    return quantized


def serialize_model(
    cfg: ImplicitConfig,
    quantized: dict[str, tuple[np.ndarray, float]],
    no_change_bias: float,
) -> bytes:
    roster = []
    body = bytearray()
    for name in sorted(quantized):
        array, scale = quantized[name]
        payload = np.ascontiguousarray(array).tobytes()
        roster.append(
            {
                "name": name,
                "shape": list(array.shape),
                "scale_f32_hex": struct.pack("<f", float(scale)).hex(),
                "bytes": len(payload),
            }
        )
        body.extend(payload)
    header_value = {
        "config": asdict(cfg),
        "no_change_bias_f32_hex": struct.pack("<f", float(no_change_bias)).hex(),
        "roster": roster,
        "seed": SEED,
    }
    header = json.dumps(
        header_value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return MODEL_HEADER.pack(MODEL_MAGIC, MODEL_VERSION, len(header)) + header + body


def parse_model(payload: bytes) -> tuple[ImplicitConfig, dict[str, np.ndarray], float]:
    if len(payload) < MODEL_HEADER.size:
        raise HR3Error("implicit model payload truncated")
    magic, version, header_size = MODEL_HEADER.unpack_from(payload)
    if magic != MODEL_MAGIC or version != MODEL_VERSION:
        raise HR3Error("implicit model magic/version mismatch")
    header_end = MODEL_HEADER.size + header_size
    if header_end > len(payload):
        raise HR3Error("implicit model roster truncated")
    header = json.loads(payload[MODEL_HEADER.size:header_end].decode("utf-8"))
    cfg = ImplicitConfig(**header["config"])
    no_change_bias = struct.unpack(
        "<f", bytes.fromhex(header["no_change_bias_f32_hex"])
    )[0]
    if not math.isfinite(no_change_bias):
        raise HR3Error("implicit model no-change bias is invalid")
    cursor = header_end
    state: dict[str, np.ndarray] = {}
    for row in header["roster"]:
        name = str(row["name"])
        if name in state:
            raise HR3Error("implicit model contains a duplicate tensor")
        size = int(row["bytes"])
        raw = payload[cursor : cursor + size]
        cursor += size
        shape = tuple(int(value) for value in row["shape"])
        if any(value <= 0 for value in shape):
            raise HR3Error("implicit model tensor shape is non-positive")
        expected = int(np.prod(shape))
        if len(raw) != size or size != expected:
            raise HR3Error("implicit model tensor size mismatch")
        scale = struct.unpack("<f", bytes.fromhex(row["scale_f32_hex"]))[0]
        if not math.isfinite(scale) or scale <= 0:
            raise HR3Error("implicit model tensor scale is invalid")
        state[name] = (
            np.frombuffer(raw, dtype=np.int8).reshape(shape).astype(np.float32) * scale
        )
    if cursor != len(payload):
        raise HR3Error("implicit model has trailing bytes")
    return cfg, state, float(no_change_bias)


def numpy_logits(
    cfg: ImplicitConfig,
    state: dict[str, np.ndarray],
    coords: np.ndarray,
    base_class: np.ndarray,
    pair_index: int,
) -> np.ndarray:
    fourier_b = deterministic_fourier_B(cfg.n_fourier, cfg.fourier_sigma)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        projection = np.einsum(
            "ni,ij->nj", np.asarray(coords, dtype=np.float32), fourier_b, optimize=False
        )
        base_embedding = state["base_embedding.weight"][base_class]
        features = np.concatenate(
            (np.sin(projection), np.cos(projection), base_embedding), axis=-1
        ).astype(np.float32)
        hidden = np.maximum(
            np.einsum(
                "ni,oi->no", features, state["in_proj.weight"], optimize=False
            )
            + state["in_proj.bias"],
            0.0,
        )
        mod = state["mod.weight"][pair_index]
        film = (
            np.einsum("i,oi->o", mod, state["film.weight"], optimize=False)
            + state["film.bias"]
        )
        film = film.reshape(cfg.n_hidden, 2, cfg.width)
        for layer_index in range(cfg.n_hidden):
            hidden = np.maximum(
                (
                    np.einsum(
                        "ni,oi->no",
                        hidden,
                        state[f"hidden.{layer_index}.weight"],
                        optimize=False,
                    )
                    + state[f"hidden.{layer_index}.bias"]
                )
                * (1.0 + film[layer_index, 0])
                + film[layer_index, 1],
                0.0,
            )
        logits = (
            np.einsum("ni,oi->no", hidden, state["out.weight"], optimize=False)
            + state["out.bias"]
        )
    if not np.isfinite(logits).all():
        raise HR3Error("implicit model produced non-finite logits")
    return logits


def render_implicit_model(
    model_payload: bytes,
    base_path: Path,
    output_path: Path,
) -> dict[str, object]:
    cfg, state, no_change_bias = parse_model(model_payload)
    coords = build_coords(HEIGHT, WIDTH)
    base = np.memmap(base_path, mode="r", dtype=np.uint8, shape=TOKEN_SHAPE)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    output = np.memmap(temporary, mode="w+", dtype=np.uint8, shape=TOKEN_SHAPE)
    changed = 0
    for pair in range(N_PAIRS):
        base_frame = np.asarray(base[pair]).reshape(-1)
        logits = numpy_logits(cfg, state, coords, base_frame.astype(np.int64), pair)
        logits[:, 0] += no_change_bias
        actions = np.argmax(logits, axis=-1).astype(np.uint8)
        frame = base_frame.copy()
        selected = actions > 0
        frame[selected] = actions[selected] - 1
        changed += int(np.count_nonzero(frame != base_frame))
        output[pair] = frame.reshape(HEIGHT, WIDTH)
    output.flush()
    del base, output
    os.replace(temporary, output_path)
    return {**et1.file_fact(output_path), "model_changed_positions_vs_hg1": changed}


def render_implicit_variants(
    cfg: ImplicitConfig,
    state: dict[str, np.ndarray],
    bias_outputs: dict[float, Path],
    base_path: Path,
) -> dict[float, dict[str, object]]:
    """Render all no-change-bias variants in one retained full-field pass."""

    coords = build_coords(HEIGHT, WIDTH)
    base = np.memmap(base_path, mode="r", dtype=np.uint8, shape=TOKEN_SHAPE)
    temporary_paths = {
        bias: path.with_name(f".{path.name}.{os.getpid()}.tmp")
        for bias, path in bias_outputs.items()
    }
    outputs = {}
    changed = dict.fromkeys(bias_outputs, 0)
    for bias, path in temporary_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        outputs[bias] = np.memmap(path, mode="w+", dtype=np.uint8, shape=TOKEN_SHAPE)
    for pair in range(N_PAIRS):
        base_frame = np.asarray(base[pair]).reshape(-1)
        logits = numpy_logits(cfg, state, coords, base_frame.astype(np.int64), pair)
        for bias, output in outputs.items():
            biased = logits.copy()
            biased[:, 0] += bias
            actions = np.argmax(biased, axis=-1).astype(np.uint8)
            frame = base_frame.copy()
            selected = actions > 0
            frame[selected] = actions[selected] - 1
            changed[bias] += int(np.count_nonzero(frame != base_frame))
            output[pair] = frame.reshape(HEIGHT, WIDTH)
    for output in outputs.values():
        output.flush()
    del base, outputs
    facts = {}
    for bias, temporary in temporary_paths.items():
        os.replace(temporary, bias_outputs[bias])
        facts[bias] = {
            **et1.file_fact(bias_outputs[bias]),
            "model_changed_positions_vs_hg1": changed[bias],
            "shared_forward_across_bias_variants": True,
        }
    return facts


def make_external_race(
    name: str,
    raw_path: Path,
    coded_path: Path,
    coder: str,
) -> dict[str, object]:
    raw = raw_path.read_bytes()
    coded = coded_path.read_bytes()
    if et1.decompress_payload(coded, coder) != raw:
        raise HR3Error(f"external generator race does not parse: {name}")
    return {
        "name": name,
        "raw": et1.file_fact(raw_path),
        "winner": coder,
        "coders": {coder: {"coded": et1.file_fact(coded_path)}},
    }


def build_packet(races: Sequence[dict[str, object]], output_path: Path) -> dict[str, object]:
    rows = bytearray()
    bodies = bytearray()
    for race in races:
        name = str(race["name"])
        winner = str(race["winner"])
        raw_path = Path(str(race["raw"]["path"]))
        coded_path = Path(str(race["coders"][winner]["coded"]["path"]))
        raw = raw_path.read_bytes()
        coded = coded_path.read_bytes()
        rows.extend(
            PACKET_ROW.pack(
                STREAM_IDS[name],
                et1.CODER_IDS[winner],
                len(raw),
                len(coded),
                bytes.fromhex(sha256_bytes(raw)),
                bytes.fromhex(sha256_bytes(coded)),
            )
        )
        bodies.extend(coded)
    packet = PACKET_HEADER.pack(PACKET_MAGIC, PACKET_VERSION, len(races), 0) + rows + bodies
    et1.atomic_bytes(output_path, packet)
    return et1.file_fact(output_path)


def parse_packet(packet: bytes) -> dict[str, bytes]:
    if len(packet) < PACKET_HEADER.size:
        raise HR3Error("packet truncated")
    magic, version, count, reserved = PACKET_HEADER.unpack_from(packet)
    if magic != PACKET_MAGIC or version != PACKET_VERSION or count != len(STREAMS) or reserved:
        raise HR3Error("packet header mismatch")
    cursor = PACKET_HEADER.size
    roster = []
    for _ in range(count):
        if cursor + PACKET_ROW.size > len(packet):
            raise HR3Error("packet roster truncated")
        roster.append(PACKET_ROW.unpack_from(packet, cursor))
        cursor += PACKET_ROW.size
    streams: dict[str, bytes] = {}
    for stream_id, coder_id, raw_size, coded_size, raw_sha, coded_sha in roster:
        if stream_id not in ID_STREAMS or coder_id not in et1.CODER_NAMES:
            raise HR3Error("packet enum invalid")
        coded = packet[cursor : cursor + coded_size]
        cursor += coded_size
        if len(coded) != coded_size or sha256_bytes(coded) != coded_sha.hex():
            raise HR3Error("packet coded identity mismatch")
        raw = et1.decompress_payload(coded, et1.CODER_NAMES[coder_id])
        if len(raw) != raw_size or sha256_bytes(raw) != raw_sha.hex():
            raise HR3Error("packet raw identity mismatch")
        name = ID_STREAMS[stream_id]
        if name in streams:
            raise HR3Error("duplicate packet stream")
        streams[name] = raw
    if cursor != len(packet) or set(streams) != set(STREAMS):
        raise HR3Error("packet roster/trailing mismatch")
    return streams


def decode_packet_to_file(
    packet: bytes,
    output_path: Path,
    *,
    base_output_path: Path,
    implicit_output_path: Path,
) -> dict[str, object]:
    streams = parse_packet(packet)
    base_output_path.parent.mkdir(parents=True, exist_ok=True)
    hg1.render_generators(streams, base_output_path)
    render_implicit_model(streams["implicit_model"], base_output_path, implicit_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    shutil.copyfile(implicit_output_path, temporary)
    os.replace(temporary, output_path)
    output = np.memmap(output_path, mode="r+", dtype=np.uint8, shape=TOKEN_SHAPE)
    corrections = hg1.apply_residual(streams["residual"], output)
    output.flush()
    del output
    return {
        **et1.file_fact(output_path),
        "remaining_explicit_corrections": corrections,
        "receiver_base": et1.file_fact(base_output_path),
        "receiver_implicit": et1.file_fact(implicit_output_path),
    }


def build_complete_archive(
    output_path: Path,
    packet: bytes,
    semantic: bytes,
    carrier: bytes,
    compact_residual: bytes,
) -> None:
    header = COMPLETE_HEADER.pack(
        COMPLETE_MAGIC,
        COMPLETE_VERSION,
        len(semantic),
        len(carrier),
        len(compact_residual),
        len(packet),
    )
    member = header + semantic + carrier + compact_residual + packet
    info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, mode="w") as archive:
        archive.writestr(info, member)
    os.replace(temporary, output_path)


def parse_complete_archive(path: Path) -> tuple[dict[str, bytes], bytes]:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["p"]:
            raise HR3Error("archive member roster mismatch")
        member = archive.read("p")
    if len(member) < COMPLETE_HEADER.size:
        raise HR3Error("archive member truncated")
    magic, version, semantic_n, carrier_n, compact_n, packet_n = COMPLETE_HEADER.unpack_from(member)
    if magic != COMPLETE_MAGIC or version != COMPLETE_VERSION:
        raise HR3Error("archive magic/version mismatch")
    if len(member) != COMPLETE_HEADER.size + semantic_n + carrier_n + compact_n + packet_n:
        raise HR3Error("archive sections do not close")
    cursor = COMPLETE_HEADER.size
    sections = {}
    for name, size in (
        ("semantic_renderer", semantic_n),
        ("pose_carrier", carrier_n),
        ("compact_residual", compact_n),
    ):
        sections[name] = member[cursor : cursor + size]
        cursor += size
    return sections, member[cursor:]


def retain_framing(archive_path: Path, packet: bytes, output_path: Path) -> dict[str, object]:
    archive_bytes = archive_path.read_bytes()
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo("p")
        member = archive.read("p")
    local_offset = int(info.header_offset)
    if archive_bytes[local_offset : local_offset + 4] != b"PK\x03\x04":
        raise HR3Error("archive local ZIP header is malformed")
    filename_bytes = int.from_bytes(archive_bytes[local_offset + 26 : local_offset + 28], "little")
    extra_bytes = int.from_bytes(archive_bytes[local_offset + 28 : local_offset + 30], "little")
    member_start = local_offset + 30 + filename_bytes + extra_bytes
    member_end = member_start + int(info.compress_size)
    roster_bytes = PACKET_HEADER.size + len(STREAMS) * PACKET_ROW.size
    framing = (
        archive_bytes[:member_start]
        + member[: COMPLETE_HEADER.size]
        + packet[:roster_bytes]
        + archive_bytes[member_end:]
    )
    et1.atomic_bytes(output_path, framing)
    return et1.file_fact(output_path)


def best_order_and_coder(
    residual_races: dict[str, dict[str, object]],
) -> tuple[str, dict[str, object]]:
    order = min(
        residual_races,
        key=lambda name: (
            int(
                residual_races[name]["coders"][residual_races[name]["winner"]]["coded"]["bytes"]
            ),
            list(hg1.RESIDUAL_ORDER_IDS).index(name),
        ),
    )
    return order, residual_races[order]


def retained_inventory(output_root: Path, manifest_path: Path) -> list[dict[str, object]]:
    return hg1.retained_inventory(output_root, manifest_path)


def run(args: argparse.Namespace) -> dict[str, object]:
    output_root = args.output_root.resolve()
    manifest_path = args.resume_from.resolve()
    if manifest_path.parent != output_root:
        raise HR3Error("--resume-from must be OUTPUT_ROOT/manifest.json")
    preflight = storage_preflight(output_root, args.minimum_free_bytes)

    source_archive = args.source_archive.resolve()
    source_tokens_path = args.source_tokens.resolve()
    hg1_root = args.hg1_root.resolve()
    top1_path = args.top1_mask.resolve()
    bl1_cost_path = args.bl1_cost_field.resolve()
    generated_path = hg1_root / "retained" / "generators" / "generated_tokens.u8"
    hg1_manifest_path = hg1_root / "manifest.json"
    require_file(
        source_archive,
        byte_count=SOURCE_ARCHIVE_BYTES,
        sha256=SOURCE_ARCHIVE_SHA256,
        label="source DX2 archive",
    )
    require_file(
        source_tokens_path,
        byte_count=SOURCE_TOKENS_BYTES,
        sha256=SOURCE_TOKENS_SHA256,
        label="source DX2 tokens",
    )
    require_file(
        generated_path,
        byte_count=HG1_GENERATED_BYTES,
        sha256=HG1_GENERATED_SHA256,
        label="HG1 generated field",
    )
    require_file(
        top1_path,
        byte_count=hg1.TOP1_BYTES,
        sha256=hg1.TOP1_SHA256,
        label="BL1 top-1% mask",
    )
    require_file(
        bl1_cost_path,
        byte_count=BL1_COST_BYTES,
        sha256=BL1_COST_SHA256,
        label="BL1 per-position cost field",
    )
    hg1_manifest = json.loads(hg1_manifest_path.read_text(encoding="utf-8"))
    if int(hg1_manifest["final"]["exact"]["residual_coded_bytes"]) != HG1_EXACT_RESIDUAL_BYTES:
        raise HR3Error("HG1 residual byte pin drift")
    if int(hg1_manifest["final"]["exact"]["direct_decode"]["corrections"]) != HG1_EXACT_CORRECTIONS:
        raise HR3Error("HG1 correction count pin drift")

    manifest = load_manifest(manifest_path)
    resumed_manifest = manifest is not None
    runner_fact = et1.file_fact(Path(__file__).resolve())
    if manifest is not None:
        recorded_runner = manifest.get("provenance", {}).get("runner")
        if recorded_runner != runner_fact:
            recorded_sha = (
                str(recorded_runner.get("sha256"))
                if isinstance(recorded_runner, dict)
                else ""
            )
            reason = RESUME_COMPATIBLE_RUNNER_SHA256.get(recorded_sha)
            if reason is None:
                raise HR3Error(
                    "resume manifest runner identity differs from the current script; "
                    "use the original script bytes or a new output root"
                )
            migrations = manifest["provenance"].setdefault("runner_migrations", [])
            migrations.append(
                {
                    "from": recorded_runner,
                    "to": runner_fact,
                    "reason": reason,
                }
            )
            manifest["provenance"]["runner"] = runner_fact
            write_manifest(manifest_path, manifest)
    if manifest is None:
        manifest = {
            "schema": SCHEMA,
            "axis": AXIS,
            "seed": SEED,
            "stages": {},
            "provenance": {
                "argv": sys.argv,
                "cwd": str(Path.cwd()),
                "git_head_before_serializer": current_git_head(),
                "platform": platform.platform(),
                "python": sys.version,
                "torch": torch.__version__,
                "runner": runner_fact,
                "config": {
                    "model_specs": MODEL_SPECS,
                    "bias_values": BIAS_VALUES,
                    "train_epochs": TRAIN_EPOCHS,
                    "train_batch": TRAIN_BATCH,
                    "full_field_max_width": args.full_field_max_width,
                },
            },
            "source": {
                "archive": et1.file_fact(source_archive),
                "tokens": et1.file_fact(source_tokens_path),
                "hg1_manifest": et1.file_fact(hg1_manifest_path),
                "hg1_generated": et1.file_fact(generated_path),
                "bl1_top1_mask": et1.file_fact(top1_path),
                "bl1_cost_field": et1.file_fact(bl1_cost_path),
            },
            "preflight": preflight,
        }
        write_manifest(manifest_path, manifest)

    original_full_field_max_width = int(
        manifest["provenance"]["config"].get(
            "original_full_field_max_width",
            manifest["provenance"]["config"].get(
                "full_field_max_width", FULL_FIELD_MAX_WIDTH
            ),
        )
    )
    manifest["provenance"]["config"][
        "original_full_field_max_width"
    ] = original_full_field_max_width
    if args.full_field_max_width > original_full_field_max_width:
        raise HR3Error("resume may not expand the preregistered full-field width bound")
    if args.full_field_max_width < original_full_field_max_width:
        manifest["provenance"]["scope_reduction"] = {
            "type": "BOUNDING_NOT_SOLVING_CARRIER_FIT",
            "original_full_field_max_width": original_full_field_max_width,
            "active_full_field_max_width": args.full_field_max_width,
            "reason": (
                "APDataStore free space fell below the measured remaining retention "
                "requirement while an external writer was active"
            ),
        }
    manifest["provenance"]["config"][
        "active_full_field_max_width"
    ] = args.full_field_max_width
    manifest.setdefault("preflight_history", []).append(preflight)
    write_manifest(manifest_path, manifest)

    source = np.memmap(source_tokens_path, mode="r", dtype=np.uint8, shape=TOKEN_SHAPE)
    generated = np.memmap(generated_path, mode="r", dtype=np.uint8, shape=TOKEN_SHAPE)

    characterization_path = output_root / "characterization.json"
    prior_characterization = manifest["stages"].get("01_characterization", {})
    if (
        isinstance(prior_characterization, dict)
        and fact_matches(characterization_path, prior_characterization.get("result"))
    ):
        characterization = json.loads(characterization_path.read_text(encoding="utf-8"))
        characterization_resumed = True
    else:
        characterization = characterize_residual(
            source,
            generated,
            top1_path,
            bl1_cost_path,
            output_root,
        )
        atomic_json(characterization_path, characterization)
        characterization_resumed = False
    if int(characterization["mismatch_positions"]) != HG1_EXACT_CORRECTIONS:
        raise HR3Error("characterization mismatch count does not reproduce HG1")
    manifest["stages"]["01_characterization"] = {
        "result": et1.file_fact(characterization_path),
        "resumed_from_checkpoint": characterization_resumed,
    }
    write_manifest(manifest_path, manifest)
    print(
        f"[hr3] characterization residual={characterization['mismatch_positions']} "
        f"fraction={characterization['mismatch_fraction']:.9f}",
        flush=True,
    )

    sample_path = output_root / "retained" / "training" / "training_sample.n600.npz"
    prior_sample = manifest["stages"].get("02_training_sample", {})
    if isinstance(prior_sample, dict) and fact_matches(sample_path, prior_sample.get("sample")):
        sample_fact = prior_sample["sample"]
        sample_resumed = True
    else:
        sample_fact = build_training_sample(source, generated, sample_path)
        sample_resumed = False
    if not bool(sample_fact["all_hg1_mismatches_included"]):
        raise HR3Error("training sample omitted an HG1 mismatch")
    manifest["stages"]["02_training_sample"] = {
        "sample": sample_fact,
        "resumed_from_checkpoint": sample_resumed,
    }
    write_manifest(manifest_path, manifest)

    training_rows: dict[str, object] = {}
    for width, mod_dim in MODEL_SPECS:
        cfg = ImplicitConfig(width=width, mod_dim=mod_dim)
        model_id = f"w{width:03d}_m{mod_dim:02d}"
        model_root = output_root / "retained" / "models" / model_id / "checkpoints"
        training_rows[model_id] = train_model(cfg, sample_path, generated, model_root)
        manifest["stages"]["03_training"] = training_rows
        write_manifest(manifest_path, manifest)

    source_sections_root = output_root / "retained" / "source_sections"
    copied_sections = {}
    for name in hg1.GENERATOR_STREAMS:
        copied_sections[name] = copy_exact(
            hg1_root / "retained" / "generators" / f"{name}.raw",
            source_sections_root / f"{name}.raw",
        )
    for name in ("semantic_renderer", "pose_carrier", "compact_residual"):
        copied_sections[name] = copy_exact(
            hg1_root / "retained" / "source_sections" / f"source_{name}.bin",
            source_sections_root / f"{name}.bin",
        )
    manifest["stages"]["04_source_sections"] = copied_sections
    write_manifest(manifest_path, manifest)

    generator_races: dict[str, dict[str, object]] = {}
    for name in hg1.GENERATOR_STREAMS:
        generator_races[name] = hg1.coder_race(
            f"generator_{name}", source_sections_root / f"{name}.raw", output_root
        )
        generator_races[name]["name"] = name
    generator_bytes = sum(
        int(row["coders"][row["winner"]]["coded"]["bytes"])
        for row in generator_races.values()
    )
    if generator_bytes != 47_667:
        raise HR3Error(f"generator coder reproduction drift: {generator_bytes}")
    receiver_base_path = (
        output_root / "retained" / "receiver_common" / "generated_tokens.u8"
    )

    prior_candidate_stage = manifest["stages"].get("05_candidates", [])
    prior_candidate_rows = (
        {
            str(row["candidate_id"]): row
            for row in prior_candidate_stage
            if isinstance(row, dict) and isinstance(row.get("candidate_id"), str)
        }
        if isinstance(prior_candidate_stage, list)
        else {}
    )
    candidate_rows: list[dict[str, object]] = []
    for width, mod_dim in MODEL_SPECS:
        cfg = ImplicitConfig(width=width, mod_dim=mod_dim)
        model_id = f"w{width:03d}_m{mod_dim:02d}"
        final_checkpoint = Path(
            str(training_rows[model_id]["final_checkpoint"]["path"])
        )
        state = load_final_state(cfg, final_checkpoint)
        quantized = quantize_state(state)
        rendered_variants: dict[float, dict[str, object]] | None = None
        for bias in BIAS_VALUES:
            bias_id = f"b{round(bias * 10):03d}"
            candidate_id = f"{model_id}_{bias_id}"
            prior_row = prior_candidate_rows.get(candidate_id)
            if candidate_row_is_resumable(prior_row):
                candidate_rows.append(prior_row)
                print(f"[hr3] resumed retained candidate {candidate_id}", flush=True)
                continue
            candidate_root = output_root / "retained" / "candidates" / candidate_id
            model_raw_path = candidate_root / "implicit_model.hr1"
            model_payload = serialize_model(cfg, quantized, bias)
            et1.atomic_bytes(model_raw_path, model_payload)
            parsed_cfg, parsed_state, parsed_bias = parse_model(model_payload)
            if parsed_cfg != cfg or abs(parsed_bias - bias) > 1e-6 or set(parsed_state) != set(state):
                raise HR3Error(f"model parse-back mismatch: {candidate_id}")
            model_race = hg1.coder_race(f"{candidate_id}_model", model_raw_path, output_root)
            model_race["name"] = "implicit_model"
            model_coded_bytes = int(
                model_race["coders"][model_race["winner"]]["coded"]["bytes"]
            )

            if width > args.full_field_max_width or model_coded_bytes > RESIDUAL_TARGET_BYTES:
                candidate_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "type": "BUILT_MODEL_ONLY_BYTE_LOWER_BOUND",
                        "config": asdict(cfg),
                        "no_change_bias": bias,
                        "model": model_race,
                        "model_coded_bytes": model_coded_bytes,
                        "remaining_residual_coded_bytes": "UNMEASURED_MODEL_ALONE_AT_OR_ABOVE_GATE"
                        if model_coded_bytes > RESIDUAL_TARGET_BYTES
                        else "UNMEASURED_WIDTH_ABOVE_ACTIVE_FULL_FIELD_LADDER_BOUND",
                        "residual_total_target_bytes": RESIDUAL_TARGET_BYTES,
                        "receiver_closed": False,
                    }
                )
                manifest["stages"]["05_candidates"] = candidate_rows
                write_manifest(manifest_path, manifest)
                continue

            implicit_field_path = candidate_root / "implicit_generated_tokens.u8"
            if rendered_variants is None:
                bias_outputs = {
                    value: (
                        output_root
                        / "retained"
                        / "candidates"
                        / f"{model_id}_b{round(value * 10):03d}"
                        / "implicit_generated_tokens.u8"
                    )
                    for value in BIAS_VALUES
                }
                rendered_variants = render_implicit_variants(
                    parsed_cfg,
                    parsed_state,
                    bias_outputs,
                    generated_path,
                )
            implicit_fact = rendered_variants[bias]
            implicit_field = np.memmap(
                implicit_field_path, mode="r", dtype=np.uint8, shape=TOKEN_SHAPE
            )
            before_facts = hg1.mismatch_facts(source, implicit_field)
            residual_races = {}
            residual_raw_facts = {}
            for order in hg1.RESIDUAL_ORDER_IDS:
                residual_raw_path = candidate_root / "residuals" / f"{order}.raw"
                residual_raw_facts[order] = hg1.encode_residual(
                    source, implicit_field, residual_raw_path, None, order
                )
                residual_races[order] = hg1.coder_race(
                    f"{candidate_id}_residual_{order}", residual_raw_path, output_root
                )
                residual_races[order]["name"] = "residual"
            selected_order, selected_residual_race = best_order_and_coder(residual_races)
            remaining_residual_bytes = int(
                selected_residual_race["coders"][selected_residual_race["winner"]]["coded"]["bytes"]
            )
            selected_races = [
                generator_races[name] for name in hg1.GENERATOR_STREAMS
            ] + [model_race, selected_residual_race]
            packet_path = candidate_root / "candidate.hr3p"
            packet_fact = build_packet(selected_races, packet_path)
            packet = packet_path.read_bytes()
            semantic = (source_sections_root / "semantic_renderer.bin").read_bytes()
            carrier = (source_sections_root / "pose_carrier.bin").read_bytes()
            compact = (source_sections_root / "compact_residual.bin").read_bytes()
            archive_path = candidate_root / "candidate_hr3.zip"
            repeat_path = candidate_root / "candidate_hr3.repeat.zip"
            build_complete_archive(archive_path, packet, semantic, carrier, compact)
            build_complete_archive(repeat_path, packet, semantic, carrier, compact)
            if archive_path.read_bytes() != repeat_path.read_bytes():
                raise HR3Error(f"archive repeat mismatch: {candidate_id}")
            parsed_sections, parsed_packet = parse_complete_archive(archive_path)
            if parsed_packet != packet:
                raise HR3Error(f"archive packet parse-back mismatch: {candidate_id}")
            expected_sections = {
                "semantic_renderer": semantic,
                "pose_carrier": carrier,
                "compact_residual": compact,
            }
            if parsed_sections != expected_sections:
                raise HR3Error(f"archive inherited-section mismatch: {candidate_id}")
            parsed_streams = parse_packet(parsed_packet)
            if parsed_streams["implicit_model"] != model_payload:
                raise HR3Error(f"archive model stream mismatch: {candidate_id}")
            selected_residual_raw = Path(
                str(selected_residual_race["raw"]["path"])
            ).read_bytes()
            if parsed_streams["residual"] != selected_residual_raw:
                raise HR3Error(f"archive residual stream mismatch: {candidate_id}")
            framing_path = candidate_root / "container_framing.bin"
            framing_fact = retain_framing(archive_path, packet, framing_path)
            archive_fact = et1.file_fact(archive_path)
            archive_repeat_fact = et1.file_fact(repeat_path)
            if archive_fact["sha256"] != archive_repeat_fact["sha256"]:
                raise HR3Error(f"archive repeat sha mismatch: {candidate_id}")
            residual_equivalent = int(archive_fact["bytes"]) - HG1_NON_RESIDUAL_BASE_BYTES
            extra_framing_vs_hg1 = int(framing_fact["bytes"]) - HG1_BASE_FRAMING_BYTES
            typed_sum = (
                52_962
                + generator_bytes
                + model_coded_bytes
                + remaining_residual_bytes
                + int(framing_fact["bytes"])
            )
            if typed_sum != int(archive_fact["bytes"]):
                raise HR3Error(f"archive typed accounting gap: {candidate_id}")
            row = {
                "candidate_id": candidate_id,
                "type": "BUILT_ARCHIVE_PACKET_PARSED_PENDING_RECEIVER_SELECTION",
                "axis": AXIS,
                "config": asdict(cfg),
                "no_change_bias": bias,
                "implicit_field": implicit_fact,
                "mismatch_before_explicit_residual": before_facts,
                "model": model_race,
                "model_coded_bytes": model_coded_bytes,
                "residual_orders": residual_races,
                "selected_residual_order": selected_order,
                "selected_residual_coder": selected_residual_race["winner"],
                "remaining_residual_coded_bytes": remaining_residual_bytes,
                "packet": packet_fact,
                "complete_archive": archive_fact,
                "complete_archive_repeat": archive_repeat_fact,
                "archive_repeat_equal": True,
                "archive_packet_parseback_equal": True,
                "container_framing": framing_fact,
                "extra_framing_vs_hg1_bytes": extra_framing_vs_hg1,
                "residual_equivalent_bytes": residual_equivalent,
                "residual_target_bytes": RESIDUAL_TARGET_BYTES,
                "residual_bytes_over_target": residual_equivalent - RESIDUAL_TARGET_BYTES,
                "residual_reduction_factor_vs_hg1": HG1_EXACT_RESIDUAL_BYTES / residual_equivalent,
                "container_bytes": int(archive_fact["bytes"]),
                "container_target_bytes": CONTAINER_TARGET_BYTES,
                "container_bytes_over_target": int(archive_fact["bytes"]) - CONTAINER_TARGET_BYTES,
                "distortion": "PENDING_SELECTED_RECEIVER_CLOSE",
                "receiver_closed": False,
                "score_claim": False,
            }
            candidate_rows.append(row)
            del implicit_field
            print(
                f"[hr3] {candidate_id} model={model_coded_bytes} "
                f"residual={remaining_residual_bytes} equivalent={residual_equivalent} "
                f"container={archive_fact['bytes']}",
                flush=True,
            )
            manifest["stages"]["05_candidates"] = candidate_rows
            write_manifest(manifest_path, manifest)

    archive_rows = [
        row
        for row in candidate_rows
        if row.get("type")
        in {
            "BUILT_ARCHIVE_PACKET_PARSED_PENDING_RECEIVER_SELECTION",
            "BUILT_RECEIVER_CLOSED_EXACT",
        }
    ]
    if not archive_rows:
        raise HR3Error("no complete candidate archive was built")
    best = min(archive_rows, key=lambda row: int(row["residual_equivalent_bytes"]))
    best_archive = Path(str(best["complete_archive"]["path"]))
    parsed_sections, best_packet = parse_complete_archive(best_archive)
    expected_sections = {
        "semantic_renderer": (source_sections_root / "semantic_renderer.bin").read_bytes(),
        "pose_carrier": (source_sections_root / "pose_carrier.bin").read_bytes(),
        "compact_residual": (source_sections_root / "compact_residual.bin").read_bytes(),
    }
    if parsed_sections != expected_sections:
        raise HR3Error("selected archive inherited-section mismatch")
    best_root = best_archive.parent
    parseback_path = best_root / "archive_parseback_tokens.u8"
    parseback_fact = decode_packet_to_file(
        best_packet,
        parseback_path,
        base_output_path=receiver_base_path,
        implicit_output_path=Path(str(best["implicit_field"]["path"])),
    )
    if parseback_fact["sha256"] != SOURCE_TOKENS_SHA256:
        raise HR3Error("selected archive receiver failed exact token identity")
    if parseback_fact["receiver_implicit"]["sha256"] != best["implicit_field"]["sha256"]:
        raise HR3Error("selected receiver implicit field differs from batched direct field")
    direct_repeat_path = best_root / "direct_repeat_tokens.u8"
    direct_repeat_fact = decode_packet_to_file(
        best_packet,
        direct_repeat_path,
        base_output_path=receiver_base_path,
        implicit_output_path=Path(str(best["implicit_field"]["path"])),
    )
    if direct_repeat_fact["sha256"] != SOURCE_TOKENS_SHA256:
        raise HR3Error("best direct repeat receiver failed")
    if direct_repeat_fact["receiver_implicit"]["sha256"] != best["implicit_field"]["sha256"]:
        raise HR3Error("repeat receiver implicit field differs from batched direct field")
    best["type"] = "BUILT_RECEIVER_CLOSED_EXACT"
    best["distortion"] = "UNCHANGED_BY_EXACT_CATEGORICAL_FIELD_IDENTITY"
    best["receiver_closed"] = True
    best["archive_parseback_tokens"] = parseback_fact
    best["direct_repeat_tokens"] = direct_repeat_fact
    best["direct_vs_archive_parseback_equal"] = (
        direct_repeat_fact["sha256"] == parseback_fact["sha256"]
    )
    manifest["stages"]["05_candidates"] = candidate_rows
    write_manifest(manifest_path, manifest)

    if int(best["residual_equivalent_bytes"]) <= RESIDUAL_TARGET_BYTES:
        prediction = "REFUTED_BY_FALSIFIER"
    elif args.full_field_max_width < original_full_field_max_width:
        prediction = "UNADJUDICATED_SCOPE_REDUCED_AT_WIDTH_BOUND"
    else:
        prediction = "CONFIRMED_ON_MEASURED_FORMULATION"
    result = {
        "schema": SCHEMA,
        "axis": AXIS,
        "prediction": prediction,
        "prediction_number_residual_bytes": int(best["residual_equivalent_bytes"]),
        "prediction_number_container_bytes": int(best["container_bytes"]),
        "residual_target_bytes": RESIDUAL_TARGET_BYTES,
        "container_target_bytes": CONTAINER_TARGET_BYTES,
        "best_candidate": best,
        "candidate_rows": candidate_rows,
        "characterization": characterization,
        "distortion_measurement": "INHERITED_BY_EXACT_CATEGORICAL_AND_RECEIVER_SECTION_IDENTITY",
        "verdict_scope": (
            "FORMULATION — counted Fourier-coordinate residual-action INR with per-pair FiLM, "
            f"per-class heads, full-field widths through {args.full_field_max_width}, model-only "
            "bounds above that width, and exact unique-home correction"
        ),
        "scope_reduction": manifest["provenance"].get("scope_reduction"),
        "currencies": {
            "fixed_distortion_cap_bytes": CONTAINER_TARGET_BYTES,
            "bytes_over_fixed_distortion_cap": int(best["container_bytes"]) - CONTAINER_TARGET_BYTES,
            "fixed_distortion_excess_rate_s": (
                int(best["container_bytes"]) - CONTAINER_TARGET_BYTES
            )
            * RATE_EXCHANGE_S_PER_BYTE,
            "zero_distortion_required_shed_from_dx2_bytes": 150,
            "zero_distortion_equivalent_shed_achieved_bytes": SOURCE_ARCHIVE_BYTES
            - int(best["container_bytes"]),
            "zero_distortion_shortfall_bytes": max(0, int(best["container_bytes"]) - 180_218),
            "rate_exchange_s_per_byte": RATE_EXCHANGE_S_PER_BYTE,
        },
        "source": manifest["source"],
        "provenance": manifest["provenance"],
        "resumed_manifest": resumed_manifest,
        "upstream_mutated": False,
        "scorer_fired": False,
        "metal_used": False,
        "modal_fired": False,
    }
    result_path = output_root / "RESULT.json"
    atomic_json(result_path, result)
    manifest["stages"]["06_final"] = {
        "result": et1.file_fact(result_path),
        "best_candidate_id": best["candidate_id"],
        "direct_repeat_tokens": direct_repeat_fact,
    }
    inventory = retained_inventory(output_root, manifest_path)
    manifest["retained_inventory"] = {
        "files": len(inventory),
        "bytes": sum(int(row["bytes"]) for row in inventory),
        "rows": inventory,
    }
    write_manifest(manifest_path, manifest)
    return result


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-tokens", type=Path, required=True)
    parser.add_argument("--hg1-root", type=Path, required=True)
    parser.add_argument("--top1-mask", type=Path, required=True)
    parser.add_argument("--bl1-cost-field", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument(
        "--full-field-max-width",
        type=int,
        default=FULL_FIELD_MAX_WIDTH,
        choices=tuple(width for width, _ in MODEL_SPECS),
    )
    parser.add_argument("--minimum-free-bytes", type=int, default=6 * 1024**3)
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result = run(args)
    print(json.dumps({
        "best_candidate": result["best_candidate"]["candidate_id"],
        "residual_bytes": result["prediction_number_residual_bytes"],
        "container_bytes": result["prediction_number_container_bytes"],
        "prediction": result["prediction"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
