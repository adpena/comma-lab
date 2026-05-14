#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan PMG-HOTSPOT-v1 predictive mask grammar candidates.

This is a deterministic planning surface only. It screens row-span predictive
mask grammar policies against named hotspot pairs and protected confusion
directions, estimates charged bytes, and emits builder-readiness metadata. It
does not build a contest archive, does not load scorer networks, does not
dispatch jobs, and cannot support a score claim.
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import importlib.util
import itertools
import json
import lzma
import math
import platform
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIVE_PROBE_PATH = REPO_ROOT / "experiments" / "probe_predictive_mask_grammar.py"
SCHEMA = "predictive_mask_hotspot_plan_v1"
TOOL = "experiments/plan_predictive_mask_hotspot.py"
EVIDENCE_GRADE = "empirical_planning_only_non_score"
REPORT_NAME = "pmg_hotspot_plan.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "pmg_hotspot_v1_20260502_codex"
DEFAULT_HOTSPOT_PAIRS = (69, 67, 290, 285, 70, 289, 286, 294)
DEFAULT_PROTECTED_CONFUSIONS = ((2, 3), (0, 3))
DEFAULT_ROW_STRIDES = (1, 2, 4, 8)
DEFAULT_COMPRESSORS = ("lzma_xz", "zlib9", "bz2_9")
ROW_FILL_POLICIES = ("nearest", "forward", "linear")
ORIGINAL_VIDEO_BYTES = 37_545_489
DEFAULT_FRONTIER_ARCHIVE_BYTES = 276_214
DEFAULT_FRONTIER_ARCHIVE_SHA256 = "226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a"
DEFAULT_TARGET_SAVINGS_BYTES = 23_455
DEFAULT_BASELINE_MASK_STREAM_BYTES = 219_472
DEFAULT_ARCHIVE_WRAPPER_OVERHEAD_BYTES = 5_410
DEFAULT_HEADER_BYTES = 768
DEFAULT_RESIDUAL_TABLE_OVERHEAD_BYTES = 64
DEFAULT_MAX_SELECTED_ATOMS = 4096
DEFAULT_PROTECTED_CONFUSION_WEIGHT = 1_000_000
DEFAULT_HOTSPOT_DISAGREEMENT_WEIGHT = 10
DEFAULT_GLOBAL_PROTECTED_CONFUSION_WEIGHT = 1000
MAX_CLASS_COUNT = 8


class PlannerError(ValueError):
    """Raised for unsafe or unsupported PMG-HOTSPOT planning inputs."""


@dataclass(frozen=True)
class HotspotFrames:
    pair_index: int
    frame_indices: tuple[int, ...]
    requested_frame_indices: tuple[int, ...]
    in_bounds: bool


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(payload))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PlannerError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_masks(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 3:
        raise PlannerError(f"decoded mask array must have rank 3, got shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise PlannerError(f"decoded mask array must contain integer class ids, got dtype={arr.dtype}")
    if arr.size == 0:
        raise PlannerError("decoded mask array is empty")
    if int(arr.min()) < 0:
        raise PlannerError(f"decoded mask array contains negative class ids: min={int(arr.min())}")
    if int(arr.max()) >= MAX_CLASS_COUNT:
        raise PlannerError(
            f"decoded mask array has {int(arr.max()) + 1} classes; "
            f"PMG-HOTSPOT bitset planner supports at most {MAX_CLASS_COUNT}"
        )
    return np.ascontiguousarray(arr.astype(np.uint8, copy=False))


def _class_count(arr: np.ndarray) -> int:
    return int(arr.max()) + 1


def _varint_len(value: int) -> int:
    if value < 0:
        raise PlannerError(f"varint value must be nonnegative: {value}")
    count = 1
    while value >= 128:
        value >>= 7
        count += 1
    return count


def _rate_delta(byte_delta: int) -> float:
    return 25.0 * float(byte_delta) / float(ORIGINAL_VIDEO_BYTES)


def _compress(raw: bytes, compressor: str) -> bytes:
    if compressor == "none":
        return raw
    if compressor == "lzma_xz":
        return lzma.compress(raw, preset=6, format=lzma.FORMAT_XZ)
    if compressor == "zlib9":
        return zlib.compress(raw, level=9)
    if compressor == "bz2_9":
        return bz2.compress(raw, compresslevel=9)
    raise PlannerError(f"unsupported compressor: {compressor!r}")


def _csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def _csv_confusions(raw: str) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "->" in item:
            left, right = item.split("->", 1)
        elif ":" in item:
            left, right = item.split(":", 1)
        else:
            raise argparse.ArgumentTypeError(f"confusion must be src->dst or src:dst, got {item!r}")
        out.append((int(left.strip()), int(right.strip())))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one protected confusion")
    return tuple(out)


def discover_default_decoded_mask_array() -> Path | None:
    if not PREDICTIVE_PROBE_PATH.exists():
        return None
    probe = _load_module(PREDICTIVE_PROBE_PATH, "_pmg_hotspot_predictive_probe")
    discovery = probe.discover_decoded_mask_array(repo_root=REPO_ROOT)
    if discovery is None:
        return None
    return Path(discovery.path)


def discover_default_baseline_bytes() -> int | None:
    if not PREDICTIVE_PROBE_PATH.exists():
        return None
    probe = _load_module(PREDICTIVE_PROBE_PATH, "_pmg_hotspot_predictive_probe_baseline")
    baseline = probe.discover_baseline(repo_root=REPO_ROOT)
    if baseline is None:
        return None
    return int(baseline.bytes)


def row_spans(arr: np.ndarray, *, row_stride: int, class_count: int) -> np.ndarray:
    if row_stride <= 0 or row_stride > arr.shape[1]:
        raise PlannerError(f"row_stride must be in [1,{arr.shape[1]}], got {row_stride}")
    frames, height, width = arr.shape
    rows = np.arange(0, height, row_stride, dtype=np.int32)
    spans = np.full((frames, class_count, len(rows), 2), -1, dtype=np.int16)
    for cls in range(class_count):
        for row_index, y in enumerate(rows):
            present = arr[:, int(y), :] == np.uint8(cls)
            any_present = present.any(axis=1)
            if not bool(any_present.any()):
                continue
            first = present.argmax(axis=1)
            last = width - 1 - present[:, ::-1].argmax(axis=1)
            spans[any_present, cls, row_index, 0] = first[any_present].astype(np.int16)
            spans[any_present, cls, row_index, 1] = last[any_present].astype(np.int16)
    return spans


def expanded_row_spans(spans: np.ndarray, *, height: int, row_stride: int, row_fill: str) -> np.ndarray:
    if spans.dtype != np.int16 or spans.ndim != 4 or spans.shape[-1] != 2:
        raise PlannerError(f"spans must be int16 rank-4 with endpoint axis, got {spans.shape} {spans.dtype}")
    sampled_rows = spans.shape[2]
    rows = np.arange(height, dtype=np.int32)
    if row_fill == "nearest":
        row_indices = np.minimum((rows + row_stride // 2) // row_stride, sampled_rows - 1)
        return np.ascontiguousarray(spans[:, :, row_indices, :])
    if row_fill == "forward":
        row_indices = np.minimum(rows // row_stride, sampled_rows - 1)
        return np.ascontiguousarray(spans[:, :, row_indices, :])
    if row_fill == "linear":
        lower = np.minimum(rows // row_stride, sampled_rows - 1)
        upper = np.minimum(lower + 1, sampled_rows - 1)
        denom = np.maximum((upper - lower) * row_stride, 1).astype(np.float32)
        alpha = ((rows - lower * row_stride).astype(np.float32) / denom).reshape(1, 1, height, 1)
        lo = spans[:, :, lower, :].astype(np.float32, copy=False)
        hi = spans[:, :, upper, :].astype(np.float32, copy=False)
        lo_valid = (lo[..., 0] >= 0) & (lo[..., 1] >= lo[..., 0])
        hi_valid = (hi[..., 0] >= 0) & (hi[..., 1] >= hi[..., 0])
        interpolated = np.rint((1.0 - alpha) * lo + alpha * hi).astype(np.int16)
        out = np.full_like(interpolated, -1, dtype=np.int16)
        both = lo_valid & hi_valid
        only_lo = lo_valid & ~hi_valid
        only_hi = hi_valid & ~lo_valid
        out[both] = interpolated[both]
        out[only_lo] = lo.astype(np.int16)[only_lo]
        out[only_hi] = hi.astype(np.int16)[only_hi]
        inverted = out[..., 1] < out[..., 0]
        out[inverted] = -1
        return np.ascontiguousarray(out)
    raise PlannerError(f"unsupported row_fill policy: {row_fill}")


def coverage_bitsets(spans: np.ndarray, *, height: int, width: int, row_stride: int, row_fill: str) -> np.ndarray:
    expanded = expanded_row_spans(spans, height=height, row_stride=row_stride, row_fill=row_fill)
    frames, class_count, _height, _endpoints = expanded.shape
    coverage = np.zeros((frames, height, width), dtype=np.uint16)
    x = np.arange(width, dtype=np.int32).reshape(1, 1, width)
    for cls in range(class_count):
        starts = expanded[:, cls, :, 0].astype(np.int32, copy=False)
        ends = expanded[:, cls, :, 1].astype(np.int32, copy=False)
        valid = (starts >= 0) & (ends >= starts)
        covered = valid[:, :, None] & (x >= starts[:, :, None]) & (x <= ends[:, :, None])
        coverage[covered] |= np.uint16(1 << cls)
    return coverage


def _prediction_table(*, class_count: int, default_class: int, draw_order: tuple[int, ...]) -> np.ndarray:
    if sorted(draw_order) != list(range(class_count)):
        raise PlannerError(f"draw_order must be a class permutation, got {draw_order}")
    table = np.full(1 << class_count, int(default_class), dtype=np.uint8)
    for coverage in range(1 << class_count):
        pred = int(default_class)
        for cls in draw_order:
            if coverage & (1 << int(cls)):
                pred = int(cls)
        table[coverage] = np.uint8(pred)
    return table


def _coverage_source_counts(source: np.ndarray, coverage: np.ndarray, *, class_count: int) -> np.ndarray:
    joint = coverage.astype(np.int64, copy=False) * class_count + source.astype(np.int64, copy=False)
    return np.bincount(joint.reshape(-1), minlength=(1 << class_count) * class_count).reshape(
        1 << class_count, class_count
    )


def _confusion_from_counts(counts: np.ndarray, table: np.ndarray, *, class_count: int) -> np.ndarray:
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for coverage in range(1 << class_count):
        pred = int(table[coverage])
        confusion[:, pred] += counts[coverage]
    return confusion


def _confusion_count(confusion: np.ndarray, protected_confusions: tuple[tuple[int, int], ...]) -> int:
    total = 0
    class_count = confusion.shape[0]
    for source_class, predicted_class in protected_confusions:
        if 0 <= source_class < class_count and 0 <= predicted_class < class_count:
            total += int(confusion[source_class, predicted_class])
    return total


def _hotspot_frames(
    hotspot_pairs: tuple[int, ...],
    *,
    frame_count: int,
    pair_frame_mode: str,
) -> list[HotspotFrames]:
    frames: list[HotspotFrames] = []
    for pair in hotspot_pairs:
        if pair_frame_mode == "pair_index":
            requested = (int(pair),)
        elif pair_frame_mode == "video_pair_frames":
            requested = (int(pair) * 2, int(pair) * 2 + 1)
        else:
            raise PlannerError(f"unsupported pair frame mode: {pair_frame_mode!r}")
        in_bounds = tuple(frame for frame in requested if 0 <= frame < frame_count)
        frames.append(
            HotspotFrames(
                pair_index=int(pair),
                frame_indices=tuple(int(v) for v in in_bounds),
                requested_frame_indices=tuple(int(v) for v in requested),
                in_bounds=len(in_bounds) == len(requested),
            )
        )
    return frames


def _unique_hotspot_frame_indices(hotspot_frames: list[HotspotFrames]) -> tuple[int, ...]:
    return tuple(sorted({frame for record in hotspot_frames for frame in record.frame_indices}))


def _run_segments(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask_1d)
    if indices.size == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for raw in indices[1:].tolist():
        value = int(raw)
        if value == prev + 1:
            prev = value
            continue
        segments.append((start, prev + 1))
        start = value
        prev = value
    segments.append((start, prev + 1))
    return segments


def _atom_byte_estimate(
    *,
    pair_index: int,
    frame_index: int,
    source_class: int,
    predicted_class: int,
    y: int,
    x0: int,
    length: int,
) -> int:
    return (
        7
        + _varint_len(pair_index)
        + _varint_len(frame_index)
        + _varint_len(source_class)
        + _varint_len(predicted_class)
        + _varint_len(y)
        + _varint_len(x0)
        + _varint_len(length)
    )


def _residual_atoms(
    source: np.ndarray,
    coverage: np.ndarray,
    table: np.ndarray,
    *,
    hotspot_frames: list[HotspotFrames],
    protected_confusions: tuple[tuple[int, int], ...],
    max_selected_atoms: int,
) -> dict[str, Any]:
    protected_set = set(protected_confusions)
    atoms: list[dict[str, Any]] = []
    class_count = _class_count(source)
    for pair_order, record in enumerate(hotspot_frames):
        for frame in record.frame_indices:
            pred = table[coverage[frame]]
            src_hw = source[frame]
            for source_class in range(class_count):
                for predicted_class in range(class_count):
                    if source_class == predicted_class:
                        continue
                    mismatch = (src_hw == np.uint8(source_class)) & (pred == np.uint8(predicted_class))
                    if not bool(mismatch.any()):
                        continue
                    is_protected = (source_class, predicted_class) in protected_set
                    for y in range(mismatch.shape[0]):
                        for x0, x1 in _run_segments(mismatch[y]):
                            length = int(x1 - x0)
                            cost = _atom_byte_estimate(
                                pair_index=record.pair_index,
                                frame_index=frame,
                                source_class=source_class,
                                predicted_class=predicted_class,
                                y=y,
                                x0=x0,
                                length=length,
                            )
                            atoms.append(
                                {
                                    "atom_id": (
                                        f"pmg_hotspot_pair{record.pair_index:04d}_frame{frame:04d}_"
                                        f"c{source_class}_to_c{predicted_class}_y{y:04d}_x{x0:04d}_{x1:04d}"
                                    ),
                                    "atom_family": "hotspot_exact_row_residual",
                                    "pair_priority_order": int(pair_order),
                                    "identity": {
                                        "pair_index": int(record.pair_index),
                                        "frame_index": int(frame),
                                        "source_class": int(source_class),
                                        "predicted_class_before_residual": int(predicted_class),
                                        "y": int(y),
                                        "x0": int(x0),
                                        "x1_exclusive": int(x1),
                                        "length": length,
                                    },
                                    "reason": "protected_confusion" if is_protected else "hotspot_pair_disagreement",
                                    "protected_confusion": bool(is_protected),
                                    "pixels_corrected_estimate": length,
                                    "protected_confusion_pixels_corrected_estimate": length if is_protected else 0,
                                    "byte_cost_estimate": cost,
                                    "density_pixels_per_byte": float(length / max(cost, 1)),
                                    "payload_charged": True,
                                    "score_claim": False,
                                }
                            )

    atoms.sort(
        key=lambda atom: (
            not bool(atom["protected_confusion"]),
            int(atom["pair_priority_order"]),
            -float(atom["density_pixels_per_byte"]),
            int(atom["identity"]["frame_index"]),
            int(atom["identity"]["source_class"]),
            int(atom["identity"]["predicted_class_before_residual"]),
            int(atom["identity"]["y"]),
            int(atom["identity"]["x0"]),
            str(atom["atom_id"]),
        )
    )
    selected = atoms[:max_selected_atoms]
    return {
        "total_atom_count": len(atoms),
        "selected_atom_count": len(selected),
        "selected_atoms_truncated": len(selected) < len(atoms),
        "selected_atoms": selected,
        "selected_byte_cost_estimate": int(sum(int(atom["byte_cost_estimate"]) for atom in selected)),
        "selected_pixels_corrected_estimate": int(sum(int(atom["pixels_corrected_estimate"]) for atom in selected)),
        "selected_protected_confusion_pixels_corrected_estimate": int(
            sum(int(atom["protected_confusion_pixels_corrected_estimate"]) for atom in selected)
        ),
    }


def _policy_record(
    *,
    row_fill: str,
    default_class: int,
    draw_order: tuple[int, ...],
    counts_all: np.ndarray,
    counts_hotspot: np.ndarray,
    class_count: int,
    protected_confusions: tuple[tuple[int, int], ...],
    total_pixels: int,
    hotspot_pixels: int,
    protected_confusion_weight: int,
    hotspot_disagreement_weight: int,
    global_protected_confusion_weight: int,
) -> dict[str, Any]:
    table = _prediction_table(class_count=class_count, default_class=default_class, draw_order=draw_order)
    confusion = _confusion_from_counts(counts_all, table, class_count=class_count)
    hotspot_confusion = _confusion_from_counts(counts_hotspot, table, class_count=class_count)
    disagreement = int(total_pixels - int(np.trace(confusion)))
    hotspot_disagreement = int(hotspot_pixels - int(np.trace(hotspot_confusion)))
    protected_confusion_hotspot = _confusion_count(hotspot_confusion, protected_confusions)
    protected_confusion_global = _confusion_count(confusion, protected_confusions)
    objective = (
        disagreement
        + protected_confusion_weight * protected_confusion_hotspot
        + hotspot_disagreement_weight * hotspot_disagreement
        + global_protected_confusion_weight * protected_confusion_global
    )
    return {
        "row_fill": row_fill,
        "default_class": int(default_class),
        "draw_order": [int(v) for v in draw_order],
        "objective": int(objective),
        "pixel_disagreement_count": disagreement,
        "pixel_disagreement_fraction": float(disagreement / max(total_pixels, 1)),
        "hotspot_disagreement_count": hotspot_disagreement,
        "hotspot_disagreement_fraction": float(hotspot_disagreement / max(hotspot_pixels, 1)),
        "protected_confusion_hotspot_pixels": int(protected_confusion_hotspot),
        "protected_confusion_global_pixels": int(protected_confusion_global),
        "confusion_matrix": confusion.tolist(),
        "hotspot_confusion_matrix": hotspot_confusion.tolist(),
    }


def _screen_stride(
    arr: np.ndarray,
    *,
    row_stride: int,
    hotspot_frames: list[HotspotFrames],
    protected_confusions: tuple[tuple[int, int], ...],
    compressors: tuple[str, ...],
    baseline_mask_stream_bytes: int,
    frontier_archive_bytes: int,
    target_archive_bytes: int,
    archive_wrapper_overhead_bytes: int,
    max_selected_atoms: int,
    protected_confusion_weight: int,
    hotspot_disagreement_weight: int,
    global_protected_confusion_weight: int,
) -> dict[str, Any]:
    frames, height, width = (int(v) for v in arr.shape)
    class_count = _class_count(arr)
    hotspot_indices = _unique_hotspot_frame_indices(hotspot_frames)
    if not hotspot_indices:
        raise PlannerError("no hotspot pair maps to an in-bounds frame")

    spans = row_spans(arr, row_stride=row_stride, class_count=class_count)
    raw_spans = np.ascontiguousarray(spans.astype("<i2", copy=False)).tobytes(order="C")
    compression = []
    for compressor in compressors:
        body = _compress(raw_spans, compressor)
        compression.append(
            {
                "compressor": compressor,
                "compressed_size_bytes": len(body),
                "compressed_sha256": _sha256_bytes(body),
            }
        )
    compression.sort(key=lambda item: (int(item["compressed_size_bytes"]), str(item["compressor"])))

    policies: list[dict[str, Any]] = []
    draw_orders = tuple(itertools.permutations(range(class_count)))
    total_pixels = int(arr.size)
    hotspot_pixels = int(len(hotspot_indices) * height * width)
    for row_fill in ROW_FILL_POLICIES:
        coverage = coverage_bitsets(spans, height=height, width=width, row_stride=row_stride, row_fill=row_fill)
        counts_all = _coverage_source_counts(arr, coverage, class_count=class_count)
        counts_hotspot = _coverage_source_counts(arr[np.array(hotspot_indices)], coverage[np.array(hotspot_indices)], class_count=class_count)
        for default_class in range(class_count):
            for draw_order in draw_orders:
                policies.append(
                    _policy_record(
                        row_fill=row_fill,
                        default_class=default_class,
                        draw_order=tuple(int(v) for v in draw_order),
                        counts_all=counts_all,
                        counts_hotspot=counts_hotspot,
                        class_count=class_count,
                        protected_confusions=protected_confusions,
                        total_pixels=total_pixels,
                        hotspot_pixels=hotspot_pixels,
                        protected_confusion_weight=protected_confusion_weight,
                        hotspot_disagreement_weight=hotspot_disagreement_weight,
                        global_protected_confusion_weight=global_protected_confusion_weight,
                    )
                )

    policies.sort(
        key=lambda item: (
            int(item["objective"]),
            int(item["protected_confusion_hotspot_pixels"]),
            int(item["hotspot_disagreement_count"]),
            int(item["pixel_disagreement_count"]),
            str(item["row_fill"]),
            int(item["default_class"]),
            item["draw_order"],
        )
    )
    winner = policies[0]
    winner_table = _prediction_table(
        class_count=class_count,
        default_class=int(winner["default_class"]),
        draw_order=tuple(int(v) for v in winner["draw_order"]),
    )
    winner_coverage = coverage_bitsets(
        spans,
        height=height,
        width=width,
        row_stride=row_stride,
        row_fill=str(winner["row_fill"]),
    )
    residual = _residual_atoms(
        arr,
        winner_coverage,
        winner_table,
        hotspot_frames=hotspot_frames,
        protected_confusions=protected_confusions,
        max_selected_atoms=max_selected_atoms,
    )

    selected_byte_cost = int(residual["selected_byte_cost_estimate"])
    selected_atoms_complete = not bool(residual["selected_atoms_truncated"])
    hotspot_after = max(0, int(winner["hotspot_disagreement_count"]) - int(residual["selected_pixels_corrected_estimate"]))
    protected_after = max(
        0,
        int(winner["protected_confusion_hotspot_pixels"])
        - int(residual["selected_protected_confusion_pixels_corrected_estimate"]),
    )
    base_payload_bytes = int(compression[0]["compressed_size_bytes"]) + DEFAULT_HEADER_BYTES
    estimated_mask_payload_bytes = (
        base_payload_bytes
        + (DEFAULT_RESIDUAL_TABLE_OVERHEAD_BYTES if selected_byte_cost else 0)
        + selected_byte_cost
    )
    archive_bytes_estimate = (
        int(frontier_archive_bytes)
        - int(baseline_mask_stream_bytes)
        + estimated_mask_payload_bytes
        + int(archive_wrapper_overhead_bytes)
    )
    target_margin = int(target_archive_bytes) - int(archive_bytes_estimate)

    protected_confusion_complete = protected_after == 0 and selected_atoms_complete
    hotspot_pair_complete = hotspot_after == 0 and selected_atoms_complete
    builder_relevant = bool(target_margin >= 0 and protected_confusion_complete)
    if builder_relevant and hotspot_pair_complete:
        relevance_class = "builder_ready_full_hotspot_guard_planning_only"
        reason = "target byte estimate survives complete hotspot-pair residual protection"
    elif builder_relevant:
        relevance_class = "builder_ready_confusion_guard_only_planning_only"
        reason = "protected confusion residuals fit the target estimate, but other hotspot disagreements remain"
    elif protected_confusion_complete:
        relevance_class = "byte_no_go_after_confusion_guard"
        reason = "protected confusion residuals are complete but estimated archive bytes miss the target"
    else:
        relevance_class = "trust_no_go_unprotected_confusion"
        reason = "selected residual budget leaves protected hotspot confusion pixels uncorrected"
    global_disagreement_fraction = float(winner["pixel_disagreement_fraction"])
    if global_disagreement_fraction > 0.02:
        global_risk_class = "high_proxy_distortion_risk"
    elif global_disagreement_fraction > 0.005:
        global_risk_class = "elevated_proxy_distortion_risk"
    else:
        global_risk_class = "low_proxy_distortion_risk"

    return {
        "candidate_id": f"pmg_hotspot_rowspan_stride{row_stride}",
        "score_claim": False,
        "promotion_eligible": False,
        "exact_evaluable_now": False,
        "row_stride": int(row_stride),
        "row_fill": str(winner["row_fill"]),
        "default_class": int(winner["default_class"]),
        "draw_order": [int(v) for v in winner["draw_order"]],
        "span_shape": [int(v) for v in spans.shape],
        "span_tensor_bytes": len(raw_spans),
        "span_tensor_sha256": _sha256_bytes(raw_spans),
        "compression": compression,
        "best_compression": compression[0],
        "searched_policy_count": len(policies),
        "policy_objective": int(winner["objective"]),
        "policy_search": {
            "type": "complete_finite_rowspan_policy_space_with_hotspot_penalty",
            "row_fill_policies": list(ROW_FILL_POLICIES),
            "draw_order_permutations": len(draw_orders),
            "default_class_count": class_count,
            "protected_confusion_weight": int(protected_confusion_weight),
            "hotspot_disagreement_weight": int(hotspot_disagreement_weight),
            "global_protected_confusion_weight": int(global_protected_confusion_weight),
            "ranking": "planning objective only; no scorer network or exact CUDA eval",
        },
        "top_policy_records": policies[:16],
        "trust_region_metrics": {
            "pixel_disagreement_count": int(winner["pixel_disagreement_count"]),
            "pixel_disagreement_fraction": float(winner["pixel_disagreement_fraction"]),
            "hotspot_disagreement_count": int(winner["hotspot_disagreement_count"]),
            "hotspot_disagreement_fraction": float(winner["hotspot_disagreement_fraction"]),
            "protected_confusion_hotspot_pixels_before_residual": int(winner["protected_confusion_hotspot_pixels"]),
            "protected_confusion_global_pixels_before_residual": int(winner["protected_confusion_global_pixels"]),
            "hotspot_disagreement_pixels_after_selected_residual": int(hotspot_after),
            "protected_confusion_hotspot_pixels_after_selected_residual": int(protected_after),
            "protected_confusion_complete": bool(protected_confusion_complete),
            "hotspot_pair_protection_complete": bool(hotspot_pair_complete),
            "selected_atoms_complete": bool(selected_atoms_complete),
        },
        "byte_estimates": {
            "scope": (
                "planning estimate for charged mask payload bytes; excludes future decoder code review, "
                "validator coverage, and exact archive entropy interactions"
            ),
            "baseline_mask_stream_bytes": int(baseline_mask_stream_bytes),
            "frontier_archive_bytes": int(frontier_archive_bytes),
            "target_archive_bytes": int(target_archive_bytes),
            "target_savings_bytes_vs_frontier": int(frontier_archive_bytes - target_archive_bytes),
            "archive_wrapper_overhead_bytes": int(archive_wrapper_overhead_bytes),
            "base_payload_bytes_estimate": int(base_payload_bytes),
            "selected_residual_atoms_byte_estimate": int(selected_byte_cost),
            "residual_table_overhead_bytes": DEFAULT_RESIDUAL_TABLE_OVERHEAD_BYTES if selected_byte_cost else 0,
            "estimated_mask_payload_bytes": int(estimated_mask_payload_bytes),
            "archive_bytes_if_replaces_mask_estimate": int(archive_bytes_estimate),
            "target_margin_bytes": int(target_margin),
            "formula_only_rate_delta_vs_frontier": _rate_delta(int(archive_bytes_estimate) - int(frontier_archive_bytes)),
        },
        "protected_atoms": residual,
        "dispatch_relevance": {
            "dispatchable_now": False,
            "remote_or_gpu_job_allowed_by_this_manifest": False,
            "local_archive_builder_relevant": bool(builder_relevant),
            "relevance_class": relevance_class,
            "global_distortion_risk_class": global_risk_class,
            "reason": reason,
            "required_before_remote_dispatch": [
                "convert the selected policy and residual atoms into a deterministic archive member",
                "prove local unpack/inflate runtime consumes the new member",
                "record a dispatch claim before any future remote/GPU exact eval",
                "run exact CUDA auth eval on exact archive bytes before any score claim",
            ],
        },
    }


def build_plan(
    *,
    decoded_mask_array: Path,
    output_dir: Path,
    output_json: Path | None = None,
    hotspot_pairs: tuple[int, ...] = DEFAULT_HOTSPOT_PAIRS,
    protected_confusions: tuple[tuple[int, int], ...] = DEFAULT_PROTECTED_CONFUSIONS,
    pair_frame_mode: str = "pair_index",
    row_strides: tuple[int, ...] = DEFAULT_ROW_STRIDES,
    compressors: tuple[str, ...] = DEFAULT_COMPRESSORS,
    baseline_mask_stream_bytes: int = DEFAULT_BASELINE_MASK_STREAM_BYTES,
    frontier_archive_bytes: int = DEFAULT_FRONTIER_ARCHIVE_BYTES,
    frontier_archive_sha256: str = DEFAULT_FRONTIER_ARCHIVE_SHA256,
    target_savings_bytes: int = DEFAULT_TARGET_SAVINGS_BYTES,
    archive_wrapper_overhead_bytes: int = DEFAULT_ARCHIVE_WRAPPER_OVERHEAD_BYTES,
    max_selected_atoms: int = DEFAULT_MAX_SELECTED_ATOMS,
    protected_confusion_weight: int = DEFAULT_PROTECTED_CONFUSION_WEIGHT,
    hotspot_disagreement_weight: int = DEFAULT_HOTSPOT_DISAGREEMENT_WEIGHT,
    global_protected_confusion_weight: int = DEFAULT_GLOBAL_PROTECTED_CONFUSION_WEIGHT,
    command: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_json = (output_json or (output_dir / REPORT_NAME)).resolve()
    arr = _load_masks(decoded_mask_array.resolve())
    frames, height, width = (int(v) for v in arr.shape)
    class_count = _class_count(arr)
    for source_class, predicted_class in protected_confusions:
        if source_class >= class_count or predicted_class >= class_count:
            raise PlannerError(
                f"protected confusion {(source_class, predicted_class)} is outside class range [0,{class_count})"
            )
    target_archive_bytes = int(frontier_archive_bytes) - int(target_savings_bytes)
    hotspot_frame_records = _hotspot_frames(
        hotspot_pairs,
        frame_count=frames,
        pair_frame_mode=pair_frame_mode,
    )
    in_bounds_records = [record for record in hotspot_frame_records if record.frame_indices]
    if not in_bounds_records:
        raise PlannerError("no hotspot pairs map to in-bounds frames for this decoded mask tensor")
    candidates = [
        _screen_stride(
            arr,
            row_stride=int(row_stride),
            hotspot_frames=in_bounds_records,
            protected_confusions=protected_confusions,
            compressors=compressors,
            baseline_mask_stream_bytes=baseline_mask_stream_bytes,
            frontier_archive_bytes=frontier_archive_bytes,
            target_archive_bytes=target_archive_bytes,
            archive_wrapper_overhead_bytes=archive_wrapper_overhead_bytes,
            max_selected_atoms=max_selected_atoms,
            protected_confusion_weight=protected_confusion_weight,
            hotspot_disagreement_weight=hotspot_disagreement_weight,
            global_protected_confusion_weight=global_protected_confusion_weight,
        )
        for row_stride in row_strides
    ]
    candidates.sort(
        key=lambda item: (
            item["dispatch_relevance"]["relevance_class"] not in {
                "builder_ready_full_hotspot_guard_planning_only",
                "builder_ready_confusion_guard_only_planning_only",
            },
            item["trust_region_metrics"]["protected_confusion_hotspot_pixels_after_selected_residual"],
            item["trust_region_metrics"]["hotspot_disagreement_pixels_after_selected_residual"],
            item["trust_region_metrics"]["pixel_disagreement_count"],
            -item["byte_estimates"]["target_margin_bytes"],
            item["candidate_id"],
        )
    )
    best = candidates[0]
    confusion_guard_exists = any(
        item["byte_estimates"]["target_margin_bytes"] >= 0
        and item["trust_region_metrics"]["protected_confusion_complete"]
        for item in candidates
    )
    full_hotspot_guard_exists = any(
        item["byte_estimates"]["target_margin_bytes"] >= 0
        and item["trust_region_metrics"]["hotspot_pair_protection_complete"]
        for item in candidates
    )

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "gpu_required": False,
        "cuda_jobs_launched": False,
        "cloud_jobs_dispatched": False,
        "scorer_network_loaded": False,
        "exact_evaluable_now": False,
        "canonical_score_source_required": (
            "archive.zip -> inflate.sh -> upstream/evaluate.py via "
            "experiments/contest_auth_eval.py --device cuda"
        ),
        "command": list(command or []),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "source": {
            "decoded_mask_array_path": str(decoded_mask_array.resolve()),
            "decoded_mask_array_npy_sha256": _sha256_file(decoded_mask_array.resolve()),
            "decoded_mask_tensor_sha256": _sha256_bytes(arr.tobytes(order="C")),
            "shape": [frames, height, width],
            "dtype_loaded": str(arr.dtype),
            "class_count": class_count,
            "class_min": int(arr.min()),
            "class_max": int(arr.max()),
        },
        "frontier_context": {
            "frontier_archive_bytes": int(frontier_archive_bytes),
            "frontier_archive_sha256": frontier_archive_sha256,
            "target_savings_bytes": int(target_savings_bytes),
            "target_archive_bytes": int(target_archive_bytes),
            "baseline_mask_stream_bytes": int(baseline_mask_stream_bytes),
            "archive_wrapper_overhead_bytes": int(archive_wrapper_overhead_bytes),
            "score_claim": False,
        },
        "hotspot_contract": {
            "lane_id": "PMG-HOTSPOT-v1",
            "pair_frame_mode": pair_frame_mode,
            "hotspot_pairs": [int(v) for v in hotspot_pairs],
            "protected_confusions": [
                {"source_class": int(src), "predicted_class": int(dst)}
                for src, dst in protected_confusions
            ],
            "hotspot_frame_records": [
                {
                    "pair_index": record.pair_index,
                    "requested_frame_indices": list(record.requested_frame_indices),
                    "frame_indices": list(record.frame_indices),
                    "in_bounds": bool(record.in_bounds),
                }
                for record in hotspot_frame_records
            ],
            "in_bounds_hotspot_pairs": [record.pair_index for record in in_bounds_records],
            "score_claim": False,
        },
        "planning_config": {
            "row_strides": [int(v) for v in row_strides],
            "row_fill_policies": list(ROW_FILL_POLICIES),
            "compressors": list(compressors),
            "max_selected_atoms": int(max_selected_atoms),
            "protected_confusion_weight": int(protected_confusion_weight),
            "hotspot_disagreement_weight": int(hotspot_disagreement_weight),
            "global_protected_confusion_weight": int(global_protected_confusion_weight),
            "determinism": {
                "wall_clock_timestamps_recorded": False,
                "randomness_used": False,
                "stable_sort_keys": True,
            },
        },
        "source_patterns_reused": [
            "experiments/probe_predictive_mask_grammar.py row-span byte-screen shape and non-score contract",
            "experiments/build_cmg3_rowspan_candidate.py finite row-fill/default/draw-order policy search",
            "experiments/plan_charged_mask_grammar_atoms.py charged-byte planning and dispatch relevance fields",
        ],
        "candidate_table": candidates,
        "best_candidate": best,
        "recommendation": {
            "hard_no_go_for_full_hotspot_pair_preservation": not full_hotspot_guard_exists,
            "confusion_guard_candidate_exists": bool(confusion_guard_exists),
            "full_hotspot_guard_candidate_exists": bool(full_hotspot_guard_exists),
            "selected_candidate_id": best["candidate_id"],
            "local_next_action": (
                "build a local non-score archive candidate from the selected policy/residual atoms"
                if confusion_guard_exists
                else "hard no-go for PMG-HOTSPOT-v1 under current byte and atom budgets"
            ),
            "score_claim": False,
        },
        "required_next_steps_for_score_claim": [
            "implement a reviewed deterministic archive member and inflate decoder for the selected PMG policy",
            "prove local unpack/inflate parity and validator allowlist coverage",
            "claim the lane before any future remote/GPU dispatch",
            "run exact CUDA auth eval on exact archive bytes before ranking, promotion, or score claims",
        ],
        "artifacts": {
            "manifest": {
                "path": str(output_json),
            }
        },
    }
    if not math.isfinite(float(best["byte_estimates"]["formula_only_rate_delta_vs_frontier"])):
        raise PlannerError("non-finite byte estimate")
    _json_dump(output_json, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-mask-array", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--hotspot-pairs", type=_csv_ints, default=DEFAULT_HOTSPOT_PAIRS)
    parser.add_argument("--protected-confusions", type=_csv_confusions, default=DEFAULT_PROTECTED_CONFUSIONS)
    parser.add_argument("--pair-frame-mode", choices=("pair_index", "video_pair_frames"), default="pair_index")
    parser.add_argument("--row-strides", type=_csv_ints, default=DEFAULT_ROW_STRIDES)
    parser.add_argument("--compressors", type=lambda raw: tuple(part.strip() for part in raw.split(",") if part.strip()), default=DEFAULT_COMPRESSORS)
    parser.add_argument("--baseline-mask-stream-bytes", type=int)
    parser.add_argument("--frontier-archive-bytes", type=int, default=DEFAULT_FRONTIER_ARCHIVE_BYTES)
    parser.add_argument("--frontier-archive-sha256", default=DEFAULT_FRONTIER_ARCHIVE_SHA256)
    parser.add_argument("--target-savings-bytes", type=int, default=DEFAULT_TARGET_SAVINGS_BYTES)
    parser.add_argument("--archive-wrapper-overhead-bytes", type=int, default=DEFAULT_ARCHIVE_WRAPPER_OVERHEAD_BYTES)
    parser.add_argument("--max-selected-atoms", type=int, default=DEFAULT_MAX_SELECTED_ATOMS)
    parser.add_argument("--protected-confusion-weight", type=int, default=DEFAULT_PROTECTED_CONFUSION_WEIGHT)
    parser.add_argument("--hotspot-disagreement-weight", type=int, default=DEFAULT_HOTSPOT_DISAGREEMENT_WEIGHT)
    parser.add_argument("--global-protected-confusion-weight", type=int, default=DEFAULT_GLOBAL_PROTECTED_CONFUSION_WEIGHT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    decoded_mask_array = args.decoded_mask_array or discover_default_decoded_mask_array()
    if decoded_mask_array is None:
        raise SystemExit("no --decoded-mask-array supplied and no default decoded mask array was discovered")
    baseline_bytes = args.baseline_mask_stream_bytes
    if baseline_bytes is None:
        baseline_bytes = discover_default_baseline_bytes() or DEFAULT_BASELINE_MASK_STREAM_BYTES
    manifest = build_plan(
        decoded_mask_array=Path(decoded_mask_array),
        output_dir=args.output_dir,
        output_json=args.output_json,
        hotspot_pairs=tuple(args.hotspot_pairs),
        protected_confusions=tuple(args.protected_confusions),
        pair_frame_mode=args.pair_frame_mode,
        row_strides=tuple(args.row_strides),
        compressors=tuple(args.compressors),
        baseline_mask_stream_bytes=int(baseline_bytes),
        frontier_archive_bytes=int(args.frontier_archive_bytes),
        frontier_archive_sha256=str(args.frontier_archive_sha256),
        target_savings_bytes=int(args.target_savings_bytes),
        archive_wrapper_overhead_bytes=int(args.archive_wrapper_overhead_bytes),
        max_selected_atoms=int(args.max_selected_atoms),
        protected_confusion_weight=int(args.protected_confusion_weight),
        hotspot_disagreement_weight=int(args.hotspot_disagreement_weight),
        global_protected_confusion_weight=int(args.global_protected_confusion_weight),
        command=[str(Path(__file__).relative_to(REPO_ROOT)), *(argv if argv is not None else sys.argv[1:])],
    )
    print(_canonical_json({"manifest": manifest["artifacts"]["manifest"]["path"], "best_candidate": manifest["best_candidate"]["candidate_id"], "score_claim": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
