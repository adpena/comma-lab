#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Mine native representations of the solved n600 exact witness.

This is a read-only, local-CPU measurement.  Large inputs are always memory
mapped, work is split into restartable pair chunks, and every completed chunk
is an atomic JSON checkpoint on the SSD evidence tier.  It never calls a
scorer, mutates a run, emits a candidate, or moves a frontier pointer.

The camera-space and logit-space quotients are deliberately kept separate:

* camera: ``X = P_range(A) X + P_ker(A) X`` and exact resize numerator
  ``N = round(N / d) d + rho``;
* logits: ``z = mean_class(z) 1 + z_perp``.

Their energies are orthogonal algebraic measurements.  Compressed sizes are
declared-code-family upper bounds, not additive information lower bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
import zipfile
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import brotli
import cv2
import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.lie._se3_numpy import exp_se3, rotation_of, translation_of  # noqa: E402
from tac.optimization.resize_full_kernel import FullResizeKernel  # noqa: E402

SCHEMA = "rep_mine_solved_binary_measurement.v1"
AXIS = "[macOS-CPU advisory]"
POINTER = "0.1910828242 [contest-CPU] UNMOVED"
N_PAIRS, CAMERA_H, CAMERA_W, CHANNELS = 600, 874, 1164, 3
SEG_H, SEG_W, CLASSES = 384, 512, 5
RAW_SHAPE = (2 * N_PAIRS, CAMERA_H, CAMERA_W, CHANNELS)
LOGIT_SHAPE = (N_PAIRS, CLASSES, SEG_H, SEG_W)
MARGIN_EDGES = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, float("inf"))
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
EXPECTED_GT_SHA = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
EXPECTED_RAW_SHA = "a7192f9387856c849d406a322a08ff77080502751ac200cc63fe80a704989dd5"
EXPECTED_LOGIT_SHA = "41d3ef535f5b5855fe17aab678580114a50309dc48d04948af62c2f563ed3b52"
EXPECTED_FISHER_SHA = "765457d424eaf1de7e05ed8703853175ef415bd3f19fb00137a74a29de52ae00"
EXPECTED_FLIP_TREE_SHA = "9287ba63fdb3eaf8d0ca58189487ac02fe8995c131daef021bf220255dffe5fc"

# Same calibrated G1 ground-plane transport used by the canonical n600 receipt.
NATIVE_H, NATIVE_W = 874, 1164
NATIVE_INTRINSICS = {"fx": 910.0, "fy": 910.0, "cx": 582.0, "cy": 437.0}
CALIBRATION = {"s_t": -0.00143, "s_r": 0.0, "pitch_rad": -0.05}
CAMERA_HEIGHT_M = 1.22


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_named_files(paths: Iterable[Path]) -> str:
    """Hash ordered file names and content hashes using the repository tree convention."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def stored_npy_memmap(path: Path, key: str) -> np.memmap:
    """Memory-map one ZIP_STORED NPY member without inflating its siblings."""
    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED or info.file_size != info.compress_size:
            raise ValueError(f"{path}:{member} is not ZIP_STORED")
        local_header = int(info.header_offset)
    with path.open("rb") as handle:
        handle.seek(local_header)
        header = handle.read(30)
        fields = struct.unpack("<IHHHHHIIIHH", header)
        if fields[0] != 0x04034B50:
            raise ValueError(f"bad ZIP local header for {member}")
        handle.seek(local_header + 30 + int(fields[-2]) + int(fields[-1]))
        version = np.lib.format.read_magic(handle)
        shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
        offset = handle.tell()
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=offset,
        shape=shape,
        order="F" if fortran else "C",
    )


def entropy_bits(counts: np.ndarray) -> float:
    values = np.asarray(counts, dtype=np.float64).reshape(-1)
    values = values[values > 0]
    total = float(values.sum())
    if total == 0.0:
        return 0.0
    return float(total * math.log2(total) - np.dot(values, np.log2(values)))


def conditional_entropy_bits(counts: np.ndarray) -> float:
    """Return sum_c n_c H(target|context=c); target is the final axis."""
    x = np.asarray(counts, dtype=np.int64)
    return float(sum(entropy_bits(row) for row in x.reshape(-1, x.shape[-1])))


def log2_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("inf")
    if k in (0, n):
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    x = np.asarray(labels)
    out = np.zeros(x.shape, dtype=bool)
    diff = x[:, 1:] != x[:, :-1]
    out[:, 1:] |= diff
    out[:, :-1] |= diff
    diff = x[1:, :] != x[:-1, :]
    out[1:, :] |= diff
    out[:-1, :] |= diff
    return out


def run_length_summary(labels: np.ndarray) -> dict[str, Any]:
    """Count exact horizontal/vertical constant-label runs without materializing runs."""
    x = np.asarray(labels, dtype=np.uint8)
    horizontal_starts = np.ones(x.shape, dtype=bool)
    horizontal_starts[:, 1:] = x[:, 1:] != x[:, :-1]
    vertical_starts = np.ones(x.shape, dtype=bool)
    vertical_starts[1:, :] = x[1:, :] != x[:-1, :]
    horizontal = np.bincount(x[horizontal_starts], minlength=CLASSES)[:CLASSES]
    vertical = np.bincount(x[vertical_starts], minlength=CLASSES)[:CLASSES]
    pixels = np.bincount(x.reshape(-1), minlength=CLASSES)[:CLASSES]
    return {
        "horizontal_runs_by_class": horizontal,
        "vertical_runs_by_class": vertical,
        "pixels_by_class": pixels,
        "horizontal_equal_adjacencies": int((x[:, 1:] == x[:, :-1]).sum()),
        "horizontal_adjacencies": int(x.shape[0] * (x.shape[1] - 1)),
        "vertical_equal_adjacencies": int((x[1:, :] == x[:-1, :]).sum()),
        "vertical_adjacencies": int((x.shape[0] - 1) * x.shape[1]),
    }


def _zlib_bytes(value: np.ndarray, level: int = 6) -> int:
    raw = np.ascontiguousarray(value).tobytes(order="C")
    return len(zlib.compress(raw, level))


def _brotli_bytes(value: np.ndarray, quality: int = 5) -> int:
    raw = np.ascontiguousarray(value).tobytes(order="C")
    return len(brotli.compress(raw, quality=quality))


def _margin_band(margins: np.ndarray, index: int) -> np.ndarray:
    lo, hi = MARGIN_EDGES[index], MARGIN_EDGES[index + 1]
    return (np.abs(margins) >= lo) & (np.abs(margins) < hi)


def _group_code_lengths(residual: np.ndarray, labels: np.ndarray, margins: np.ndarray) -> dict[str, Any]:
    edge = boundary_mask(labels)

    def row(mask: np.ndarray) -> dict[str, int]:
        values = np.ascontiguousarray(residual[mask], dtype="<i4")
        return {
            "cells": int(np.count_nonzero(mask)),
            "values": int(values.size),
            "zlib6_bytes": _zlib_bytes(values),
        }

    by_class = {str(c): row(labels == c) for c in range(CLASSES)}
    by_margin = {}
    for i in range(len(MARGIN_EDGES) - 1):
        hi = MARGIN_EDGES[i + 1]
        name = f"[{MARGIN_EDGES[i]:.0e},{hi:.0e})" if math.isfinite(hi) else f"[{MARGIN_EDGES[i]:.0e},inf)"
        by_margin[name] = row(_margin_band(margins, i))
    return {
        "by_class": by_class,
        "by_margin": by_margin,
        "boundary": row(edge),
        "interior": row(~edge),
    }


def _sum_group_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    first = rows[0]
    out: dict[str, Any] = {}
    for key, value in first.items():
        if isinstance(value, dict):
            out[key] = _sum_group_rows(row[key] for row in rows)
        elif isinstance(value, (int, float)):
            out[key] = sum(row[key] for row in rows)
        else:
            out[key] = value
    return out


def _blind_owned_mask(kernel: FullResizeKernel) -> np.ndarray:
    owned = np.zeros((CAMERA_H, CAMERA_W), dtype=bool)
    rows = np.asarray([s.indices for s in kernel.operator.row_supports], dtype=np.intp)
    cols = np.asarray([s.indices for s in kernel.operator.col_supports], dtype=np.intp)
    owned[np.ix_(np.unique(rows), np.unique(cols))] = True
    return owned


def measure_camera_chunk(
    raw: np.memmap,
    labels: np.memmap,
    margins: np.memmap,
    kernel: FullResizeKernel,
    owned: np.ndarray,
    start: int,
    stop: int,
) -> dict[str, Any]:
    started = time.time()
    frames = np.asarray(raw[2 * start : 2 * stop])
    numerator_rows: list[np.ndarray] = []
    rounded_rows: list[np.ndarray] = []
    residual_rows: list[np.ndarray] = []
    canonical_delta_rows: list[np.ndarray] = []
    raw_energy = range_energy = kernel_energy = 0.0
    for frame in frames:
        x64 = frame.astype(np.float64)
        projection = kernel.project_range(frame, dtype=np.float32).astype(np.float64)
        null = x64 - projection
        raw_energy += float(np.square(x64).sum(dtype=np.float64))
        range_energy += float(np.square(projection).sum(dtype=np.float64))
        kernel_energy += float(np.square(null).sum(dtype=np.float64))
        numerator, denominator = kernel.operator.apply_numerators(frame)
        rounded = np.clip((numerator + denominator // 2) // denominator, 0, 255).astype(np.uint8)
        residual = (numerator - rounded.astype(np.int64) * denominator).astype(np.int32)
        canonical = kernel.operator.realize_factor2_uint8(rounded)
        numerator_rows.append(numerator.astype(np.uint32))
        rounded_rows.append(rounded)
        residual_rows.append(residual)
        canonical_delta_rows.append(frame.astype(np.int16) - canonical.astype(np.int16))
    numerators = np.stack(numerator_rows)
    rounded = np.stack(rounded_rows)
    residual = np.stack(residual_rows)
    canonical_delta = np.stack(canonical_delta_rows)
    # Class/margin authority exists only for the second frame of each pair.
    f1_residual = residual[1::2]
    group_lengths = _group_code_lengths(f1_residual, np.asarray(labels[start:stop]), np.asarray(margins[start:stop]))
    previous_residual = []
    for pair in range(start, stop):
        current = residual[2 * (pair - start) + 1]
        if pair:
            previous_frame = np.asarray(raw[2 * (pair - 1) + 1])
            previous_num, _ = kernel.operator.apply_numerators(previous_frame)
            previous_round = np.clip((previous_num + denominator // 2) // denominator, 0, 255).astype(np.uint8)
            previous = previous_num - previous_round.astype(np.int64) * denominator
            previous_residual.append((current.astype(np.int64) - previous).astype(np.int32))
    previous_residual_array = np.stack(previous_residual) if previous_residual else np.empty((0,), np.int32)
    return {
        "schema": "rep_mine_camera_chunk.v1",
        "pair_start": start,
        "pair_stop": stop,
        "frame_count": 2 * (stop - start),
        "raw_energy": raw_energy,
        "range_energy": range_energy,
        "kernel_energy": kernel_energy,
        "kernel_energy_fraction": kernel_energy / raw_energy,
        "orthogonal_energy_relative_error": abs(raw_energy - range_energy - kernel_energy) / raw_energy,
        "exact_resize_denominator": int(denominator),
        "raw_bytes": int(frames.nbytes),
        "owned_seed_values": int(frames[:, owned, :].size),
        "code_lengths": {
            "raw_camera_zlib6_bytes": _zlib_bytes(frames),
            "owned_camera_seed_zlib6_bytes": _zlib_bytes(frames[:, owned, :]),
            "owned_camera_seed_brotli5_bytes": _brotli_bytes(frames[:, owned, :]),
            "exact_numerator_u32_zlib6_bytes": _zlib_bytes(numerators),
            "rounded_rgb_plane_u8_zlib6_bytes": _zlib_bytes(rounded),
            "fractional_numerator_residual_i32_zlib6_bytes": _zlib_bytes(residual),
            "rounded_plus_fractional_sum_zlib6_bytes": _zlib_bytes(rounded) + _zlib_bytes(residual),
            "camera_delta_from_rounded_canonical_fill_i16_zlib6_bytes": _zlib_bytes(canonical_delta),
            "f1_fractional_residual_previous_frame_delta_i32_zlib6_bytes": _zlib_bytes(previous_residual_array),
        },
        "f1_fractional_residual_group_code_lengths": group_lengths,
        "elapsed_seconds": time.time() - started,
    }


def run_camera_stage(args: argparse.Namespace) -> dict[str, Any]:
    stage = args.work_dir / "camera_chunks"
    stage.mkdir(parents=True, exist_ok=True)
    raw = np.memmap(args.raw, dtype=np.uint8, mode="r", shape=RAW_SHAPE)
    labels = stored_npy_memmap(args.gt_cache, "lstars")
    margins = stored_npy_memmap(args.gt_cache, "margins")
    kernel = FullResizeKernel.build()
    owned = _blind_owned_mask(kernel)
    rows = []
    for start in range(0, N_PAIRS, args.chunk_pairs):
        stop = min(N_PAIRS, start + args.chunk_pairs)
        path = stage / f"pairs-{start:04d}-{stop:04d}.json"
        if args.resume and path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            row = measure_camera_chunk(raw, labels, margins, kernel, owned, start, stop)
            atomic_json(path, row)
        rows.append(row)
        print(f"camera {start}:{stop} complete", flush=True)
    code_keys = rows[0]["code_lengths"]
    code_lengths = {key: sum(int(row["code_lengths"][key]) for row in rows) for key in code_keys}
    raw_energy = sum(float(row["raw_energy"]) for row in rows)
    range_energy = sum(float(row["range_energy"]) for row in rows)
    kernel_energy = sum(float(row["kernel_energy"]) for row in rows)
    return {
        "stage": "camera",
        "chunks": len(rows),
        "frame_count": 2 * N_PAIRS,
        "raw_bytes": args.raw.stat().st_size,
        "full_kernel_dimension_fraction": kernel.coverage().full_nullity / kernel.coverage().domain_dimension,
        "implemented_blind_coordinate_fraction": float(np.mean(~owned)),
        "implemented_blind_coordinate_values": int((~owned).sum() * 2 * N_PAIRS * CHANNELS),
        "actual_raw_energy": raw_energy,
        "actual_range_energy": range_energy,
        "actual_kernel_energy": kernel_energy,
        "actual_kernel_energy_fraction": kernel_energy / raw_energy,
        "orthogonal_energy_relative_error": abs(raw_energy - range_energy - kernel_energy) / raw_energy,
        "code_lengths": code_lengths,
        "fractional_residual_groups": _sum_group_rows(row["f1_fractional_residual_group_code_lengths"] for row in rows),
        "interpretation": {
            "kernel": "energy decomposition; dimension/energy is not a byte saving",
            "blind_seed": "exact reconstruction of the measured M2 fill from owned bytes plus a 2-bit-per-chunk fill selector and generic fill code",
            "fractional_residual": "exact target numerator information beyond the nearest rounded RGB scorer plane",
            "canonical_fill_scope": "rounded RGB scorer plane, not argmax labels; argmax labels alone do not determine RGB/Pose values",
        },
    }


def _cell_stats(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    centered = logits.astype(np.float32) - logits.astype(np.float32).mean(axis=0, keepdims=True)
    cells = pixels = singletons = constant_cells = 0
    sse = energy = 0.0
    for target in range(CLASSES):
        components, count = ndimage.label(labels == target, structure=structure)
        if count == 0:
            continue
        sizes = np.bincount(components.reshape(-1), minlength=count + 1)[1:].astype(np.float64)
        cells += int(count)
        pixels += int(sizes.sum())
        singletons += int(np.count_nonzero(sizes == 1))
        cell_sse = np.zeros(count, dtype=np.float64)
        for plane in range(CLASSES):
            values = centered[plane].astype(np.float64, copy=False)
            sums = np.bincount(components.reshape(-1), weights=values.reshape(-1), minlength=count + 1)[1:]
            squares = np.bincount(components.reshape(-1), weights=np.square(values).reshape(-1), minlength=count + 1)[
                1:
            ]
            cell_sse += np.maximum(0.0, squares - np.square(sums) / sizes)
            energy += float(squares.sum())
        sse += float(cell_sse.sum())
        constant_cells += int(np.count_nonzero(cell_sse <= 1e-12))
    return {
        "cells": cells,
        "pixels": pixels,
        "singleton_cells": singletons,
        "constant_cells": constant_cells,
        "within_cell_sse": sse,
        "centered_energy": energy,
    }


def measure_logits_chunk(logits_mm: np.memmap, labels_mm: np.memmap, start: int, stop: int) -> dict[str, Any]:
    started = time.time()
    logits = np.asarray(logits_mm[start:stop], dtype=np.float16)
    labels = np.asarray(labels_mm[start:stop], dtype=np.uint8)
    z = logits.astype(np.float64)
    gauge = z.mean(axis=1, keepdims=True)
    centered = z - gauge
    gauge_energy = float(CLASSES * np.square(gauge).sum())
    quotient_energy = float(np.square(centered).sum())
    raw_energy = float(np.square(z).sum())
    samples = np.moveaxis(centered, 1, -1).reshape(-1, CLASSES)
    gram = samples.T @ samples
    difference_f16 = (z[:, :4] - z[:, 4:5]).astype(np.float16)
    quotient_argmax = np.concatenate(
        [difference_f16.astype(np.float32), np.zeros((*difference_f16.shape[:1], 1, SEG_H, SEG_W), np.float32)],
        axis=1,
    ).argmax(axis=1)
    raw_argmax = logits.argmax(axis=1)
    distinct = np.zeros((CLASSES, 1 << 16), dtype=np.uint8)
    codes = logits.view(np.uint16)
    for plane in range(CLASSES):
        distinct[plane, np.unique(codes[:, plane])] = 1
    cell = dict.fromkeys(
        ("cells", "pixels", "singleton_cells", "constant_cells", "within_cell_sse", "centered_energy"),
        0.0,
    )
    for i in range(stop - start):
        row = _cell_stats(logits[i], labels[i])
        for key, value in row.items():
            cell[key] += value
    return {
        "schema": "rep_mine_logits_chunk.v1",
        "pair_start": start,
        "pair_stop": stop,
        "raw_energy": raw_energy,
        "gauge_energy": gauge_energy,
        "quotient_energy": quotient_energy,
        "energy_closure_relative_error": abs(raw_energy - gauge_energy - quotient_energy) / raw_energy,
        "centered_gram": gram.tolist(),
        "code_lengths": {
            "raw_f16_zlib6_bytes": _zlib_bytes(logits),
            "gauge_mean_f32_zlib6_bytes": _zlib_bytes(gauge.astype(np.float32)),
            "four_difference_coordinates_f16_zlib6_bytes": _zlib_bytes(difference_f16),
        },
        "four_difference_argmax_mismatches": int(np.count_nonzero(quotient_argmax != raw_argmax)),
        "four_difference_argmax_values": int(raw_argmax.size),
        "cell_constancy": cell,
        "distinct_value_bitsets_hex": [np.packbits(row).tobytes().hex() for row in distinct],
        "elapsed_seconds": time.time() - started,
    }


def run_logits_stage(args: argparse.Namespace) -> dict[str, Any]:
    stage = args.work_dir / "logit_chunks"
    stage.mkdir(parents=True, exist_ok=True)
    logits = np.memmap(args.logits, dtype=np.float16, mode="r", shape=LOGIT_SHAPE)
    labels = stored_npy_memmap(args.gt_cache, "lstars")
    rows = []
    for start in range(0, N_PAIRS, args.chunk_pairs):
        stop = min(N_PAIRS, start + args.chunk_pairs)
        path = stage / f"pairs-{start:04d}-{stop:04d}.json"
        if args.resume and path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            row = measure_logits_chunk(logits, labels, start, stop)
            atomic_json(path, row)
        rows.append(row)
        print(f"logits {start}:{stop} complete", flush=True)
    raw_energy = sum(float(row["raw_energy"]) for row in rows)
    gauge_energy = sum(float(row["gauge_energy"]) for row in rows)
    quotient_energy = sum(float(row["quotient_energy"]) for row in rows)
    gram = sum((np.asarray(row["centered_gram"], np.float64) for row in rows), np.zeros((5, 5)))
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    distinct = np.zeros((CLASSES, 1 << 16), dtype=bool)
    for row in rows:
        for plane, encoded in enumerate(row["distinct_value_bitsets_hex"]):
            distinct[plane] |= np.unpackbits(np.frombuffer(bytes.fromhex(encoded), np.uint8)).astype(bool)
    code_keys = rows[0]["code_lengths"]
    cells = {key: sum(float(row["cell_constancy"][key]) for row in rows) for key in rows[0]["cell_constancy"]}
    return {
        "stage": "logits",
        "pairs": N_PAIRS,
        "source_dtype": "float16",
        "source_bytes": args.logits.stat().st_size,
        "raw_energy": raw_energy,
        "gauge_energy": gauge_energy,
        "gauge_energy_fraction": gauge_energy / raw_energy,
        "quotient_energy": quotient_energy,
        "quotient_energy_fraction": quotient_energy / raw_energy,
        "energy_closure_relative_error": abs(raw_energy - gauge_energy - quotient_energy) / raw_energy,
        "centered_class_gram_eigenvalues": eigenvalues.tolist(),
        "fifth_direction_energy_fraction": max(0.0, float(eigenvalues[-1])) / float(eigenvalues.sum()),
        "empirical_rank_relative_1e_10": int(np.count_nonzero(eigenvalues > eigenvalues[0] * 1e-10)),
        "distinct_f16_values_per_plane": [int(row.sum()) for row in distinct],
        "code_lengths": {key: sum(int(row["code_lengths"][key]) for row in rows) for key in code_keys},
        "four_difference_argmax_mismatches": sum(int(row["four_difference_argmax_mismatches"]) for row in rows),
        "four_difference_argmax_values": sum(int(row["four_difference_argmax_values"]) for row in rows),
        "cell_constancy": {
            **cells,
            "constant_cell_fraction": cells["constant_cells"] / cells["cells"],
            "singleton_cell_fraction": cells["singleton_cells"] / cells["cells"],
            "within_cell_energy_fraction": cells["within_cell_sse"] / cells["centered_energy"],
            "cell_semantics": "4-neighbor connected components of the digital argmax raster; not a classical Morse-Smale certificate",
        },
        "scope": "source-derived fp16 SegNet teacher logits; 24 ppm tie-rounding caveat in its custody manifest",
    }


def _intrinsics() -> np.ndarray:
    sx, sy = SEG_W / NATIVE_W, SEG_H / NATIVE_H
    return np.array(
        [
            [NATIVE_INTRINSICS["fx"] * sx, 0.0, NATIVE_INTRINSICS["cx"] * sx],
            [0.0, NATIVE_INTRINSICS["fy"] * sy, NATIVE_INTRINSICS["cy"] * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def pose_homography(pose6: np.ndarray) -> np.ndarray:
    xi = np.empty(6, dtype=np.float64)
    xi[:3] = CALIBRATION["s_t"] * np.array([pose6[2], pose6[1], pose6[0]])
    xi[3:] = CALIBRATION["s_r"] * np.asarray(pose6[3:6], dtype=np.float64)
    transform = exp_se3(xi)
    rotation = rotation_of(transform)
    translation = translation_of(transform)
    pitch = CALIBRATION["pitch_rad"]
    normal = np.array([0.0, -math.cos(pitch), -math.sin(pitch)], dtype=np.float64)
    plane = rotation - np.outer(translation, normal) / CAMERA_HEIGHT_M
    k = _intrinsics()
    return k @ plane @ np.linalg.inv(k)


def run_label_stage(args: argparse.Namespace) -> dict[str, Any]:
    labels = stored_npy_memmap(args.gt_cache, "lstars")
    poses = stored_npy_memmap(args.gt_cache, "gt_poses")
    counts = np.zeros(CLASSES, np.int64)
    previous = np.zeros((CLASSES, CLASSES), np.int64)
    xi_counts = np.zeros((CLASSES + 1, CLASSES), np.int64)
    # row-position bin x left/none x up/none x xi/invalid x target
    context = np.zeros((12, CLASSES + 1, CLASSES + 1, CLASSES + 1, CLASSES), np.int64)
    innovation_masks = []
    xi_innovation_masks = []
    horizontal_runs = np.zeros(CLASSES, np.int64)
    vertical_runs = np.zeros(CLASSES, np.int64)
    horizontal_equal = 0
    horizontal_total = 0
    vertical_equal = 0
    vertical_total = 0
    previous_frame = None
    for index in range(N_PAIRS):
        current = np.asarray(labels[index], dtype=np.uint8)
        counts += np.bincount(current.reshape(-1), minlength=CLASSES)[:CLASSES]
        runs = run_length_summary(current)
        horizontal_runs += runs["horizontal_runs_by_class"]
        vertical_runs += runs["vertical_runs_by_class"]
        horizontal_equal += runs["horizontal_equal_adjacencies"]
        horizontal_total += runs["horizontal_adjacencies"]
        vertical_equal += runs["vertical_equal_adjacencies"]
        vertical_total += runs["vertical_adjacencies"]
        if previous_frame is None:
            advected = np.full(current.shape, CLASSES, dtype=np.uint8)
        else:
            previous += np.bincount(
                (previous_frame.astype(np.int64) * CLASSES + current).reshape(-1),
                minlength=CLASSES * CLASSES,
            ).reshape(CLASSES, CLASSES)
            homography = pose_homography(np.asarray(poses[index]))
            advected = cv2.warpPerspective(
                previous_frame,
                homography,
                (SEG_W, SEG_H),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=CLASSES,
            )
            innovation_masks.append((current != previous_frame).astype(np.uint8))
            xi_innovation_masks.append((current != advected).astype(np.uint8))
        xi_counts += np.bincount(
            (advected.astype(np.int64) * CLASSES + current).reshape(-1),
            minlength=(CLASSES + 1) * CLASSES,
        ).reshape(CLASSES + 1, CLASSES)
        left = np.full(current.shape, CLASSES, np.uint8)
        up = np.full(current.shape, CLASSES, np.uint8)
        left[:, 1:] = current[:, :-1]
        up[1:, :] = current[:-1, :]
        row_bin = np.minimum(11, np.arange(SEG_H) * 12 // SEG_H)[:, None]
        key = (((row_bin * 6 + left) * 6 + up) * 6 + advected) * 5 + current
        context += np.bincount(key.reshape(-1), minlength=context.size).reshape(context.shape)
        previous_frame = current
    innovation = np.stack(innovation_masks)
    xi_innovation = np.stack(xi_innovation_masks)
    context_bits = conditional_entropy_bits(context)
    return {
        "stage": "labels",
        "pixels": int(counts.sum()),
        "class_counts": {str(i): int(v) for i, v in enumerate(counts)},
        "unconditional_ideal_bits": entropy_bits(counts),
        "unconditional_bits_per_cell": entropy_bits(counts) / counts.sum(),
        "previous_pair_f1_conditional_bits": conditional_entropy_bits(previous),
        "previous_pair_f1_conditional_bits_per_cell": conditional_entropy_bits(previous) / previous.sum(),
        "pose_proxy_advected_conditional_bits": conditional_entropy_bits(xi_counts),
        "pose_proxy_advected_conditional_bits_per_cell": conditional_entropy_bits(xi_counts) / xi_counts.sum(),
        "position_adjacency_pose_proxy_conditional_bits": context_bits,
        "position_adjacency_pose_proxy_conditional_bits_per_cell": context_bits / context.sum(),
        "innovation": {
            "previous_pair_f1_changed_fraction": float(innovation.mean()),
            "pose_proxy_advected_changed_fraction": float(xi_innovation.mean()),
            "previous_pair_f1_mask_zlib6_bytes": _zlib_bytes(innovation),
            "pose_proxy_advected_mask_zlib6_bytes": _zlib_bytes(xi_innovation),
        },
        "constant_label_runs": {
            "horizontal_runs_by_class": {str(i): int(v) for i, v in enumerate(horizontal_runs)},
            "vertical_runs_by_class": {str(i): int(v) for i, v in enumerate(vertical_runs)},
            "horizontal_mean_run_length_by_class": {
                str(i): float(counts[i] / horizontal_runs[i]) for i in range(CLASSES)
            },
            "vertical_mean_run_length_by_class": {str(i): float(counts[i] / vertical_runs[i]) for i in range(CLASSES)},
            "horizontal_equal_adjacency_fraction": horizontal_equal / horizontal_total,
            "vertical_equal_adjacency_fraction": vertical_equal / vertical_total,
            "scope": "exact constant-argmax runs on the 600 scorer rasters; no contour/header bytes inferred",
        },
        "context_definition": "12 row-position bins + raster-causal left/up + G1-calibrated ground-plane pose proxy; static empirical conditional entropy, model/header/pose-side-information excluded",
        "temporal_scope": "600 cached f1 planes only: previous means f1[p-1] (two video frames earlier), while gt_poses[p] is the exact f0[p]->f1[p] target and only a nearest-target proxy for f1[p-1]->f1[p]; use G1 for authoritative 1,199-transition boundary transport",
        "registered_equations_consumed": [
            "worldsheet_transport_residual_event_rate_v1",
            "partition_temporal_transport_amortization_v1",
        ],
    }


def _load_sites(stage_dir: Path) -> list[tuple[int, int, int]]:
    files = sorted(stage_dir.glob("batch-*.json"))
    if len(files) != 38:
        raise ValueError(f"expected 38 flip stages, found {len(files)}")
    sites = []
    for path in files:
        row = json.loads(path.read_text(encoding="utf-8"))
        sites.extend((int(x[0]), int(x[1]), int(x[2])) for x in row["flips"])
    if len(sites) != 17_926 or len(set(sites)) != len(sites):
        raise ValueError("flip inventory custody drift")
    return sites


def _load_fisher_sites(path: Path) -> list[tuple[int, int, int]]:
    lines = brotli.decompress(path.read_bytes()).splitlines()
    header = json.loads(lines[0])
    if int(header["candidate_count"]) != len(lines) - 1:
        raise ValueError("Fisher ordering count drift")
    return [(int(row[0]), int(row[1]), int(row[2])) for row in map(json.loads, lines[1:])]


def fisher_exception_plan(
    sites: set[tuple[int, int, int]], ranking: list[tuple[int, int, int]], universe: int
) -> dict[str, Any]:
    k = len(sites)
    ranked_set = set(ranking)
    hits = 0
    best = {"bits": log2_choose(universe, k), "prefix": 0, "hits": 0}
    at_k_hits = 0
    for m, site in enumerate(ranking, start=1):
        hits += int(site in sites)
        if m == k:
            at_k_hits = hits
        removed = m - hits
        additions = k - hits
        bits = log2_choose(m, removed) + log2_choose(universe - m, additions)
        if bits < best["bits"]:
            best = {"bits": bits, "prefix": m, "hits": hits}
    best.update(
        {
            "deletions": int(best["prefix"] - best["hits"]),
            "additions": int(k - best["hits"]),
            "agreement_at_k": at_k_hits / k,
            "actual_sites_in_ranking": len(sites & ranked_set),
            "ranking_sites": len(ranking),
        }
    )
    return best


def run_site_stage(args: argparse.Namespace) -> dict[str, Any]:
    sites = set(_load_sites(args.flip_stage_dir))
    ranking = _load_fisher_sites(args.fisher_ordering)
    universe = N_PAIRS * SEG_H * SEG_W
    colex_bits = log2_choose(universe, len(sites))
    fisher = fisher_exception_plan(sites, ranking, universe)
    return {
        "stage": "sites",
        "site_count": len(sites),
        "universe_sites": universe,
        "raw_coordinate_bits": 28 * len(sites),
        "raw_coordinate_bytes": math.ceil(28 * len(sites) / 8),
        "colex_enumerative_ideal_bits": colex_bits,
        "colex_enumerative_ideal_bytes": colex_bits / 8,
        "fisher_topk_exception_plan": fisher,
        "fisher_topk_exception_ideal_bytes": fisher["bits"] / 8,
        "fisher_ordering_payload_bytes_if_shipped": args.fisher_ordering.stat().st_size,
        "legality": "ranking algorithm is free only if recomputed from already-decoded seed; the current 0.mkv-derived ordering payload is counted and cannot be hidden in code",
    }


def _validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if args.raw.stat().st_size != int(np.prod(RAW_SHAPE)):
        raise ValueError("raw geometry/bytes drift")
    if args.logits.stat().st_size != int(np.prod(LOGIT_SHAPE)) * 2:
        raise ValueError("logit geometry/bytes drift")
    flip_files = sorted(args.flip_stage_dir.glob("batch-*.json"))
    if len(flip_files) != 38:
        raise ValueError(f"flip-stage closure drift: {len(flip_files)} != 38")
    hashes = {
        "gt_cache": sha256_file(args.gt_cache),
        "raw": sha256_file(args.raw),
        "logits": sha256_file(args.logits),
        "fisher_ordering": sha256_file(args.fisher_ordering),
        "flip_stage_tree": sha256_named_files(flip_files),
    }
    expected = {
        "gt_cache": EXPECTED_GT_SHA,
        "raw": EXPECTED_RAW_SHA,
        "logits": EXPECTED_LOGIT_SHA,
        "fisher_ordering": EXPECTED_FISHER_SHA,
        "flip_stage_tree": EXPECTED_FLIP_TREE_SHA,
    }
    if hashes != expected:
        raise ValueError(f"input custody mismatch: {hashes}")
    paths = {
        "gt_cache": args.gt_cache,
        "raw": args.raw,
        "logits": args.logits,
        "fisher_ordering": args.fisher_ordering,
        "flip_stage_tree": args.flip_stage_dir,
    }
    return {key: {"path": str(paths[key]), "sha256": value} for key, value in hashes.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--logits", type=Path, required=True)
    ap.add_argument("--flip-stage-dir", type=Path, required=True)
    ap.add_argument("--fisher-ordering", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk-pairs", type=int, default=12)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stage", choices=("camera", "logits", "labels", "sites", "all"), default="all")
    args = ap.parse_args()
    if args.chunk_pairs < 1:
        raise SystemExit("chunk-pairs must be positive")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if os.statvfs(args.work_dir).f_bavail * os.statvfs(args.work_dir).f_frsize < 8 * 1024**3:
        raise SystemExit("storage preflight refused: evidence tier needs >=8 GiB free")
    started = time.time()
    custody_path = args.work_dir / "input_custody.json"
    custody = _validate_inputs(args)
    if args.resume and custody_path.is_file():
        prior_custody = json.loads(custody_path.read_text(encoding="utf-8"))
        if custody != prior_custody:
            raise ValueError("resume input custody differs from the checkpoint binding")
    else:
        atomic_json(custody_path, custody)
    stage_functions = {
        "camera": run_camera_stage,
        "logits": run_logits_stage,
        "labels": run_label_stage,
        "sites": run_site_stage,
    }
    selected = tuple(stage_functions) if args.stage == "all" else (args.stage,)
    results: dict[str, Any] = {}
    for name in selected:
        stage_path = args.work_dir / f"stage_{name}.json"
        if args.resume and stage_path.is_file():
            result = json.loads(stage_path.read_text(encoding="utf-8"))
        else:
            result = stage_functions[name](args)
            atomic_json(stage_path, result)
        results[name] = result
    receipt = {
        "schema": SCHEMA,
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "lineage": "source-derived/our-solve only; no inherited candidate bytes",
        "input_custody": custody,
        "stages": results,
        "elapsed_seconds_this_invocation": time.time() - started,
        "checkpoint_policy": "atomic per-12-pair JSON chunks plus atomic stage summaries; --resume hash-binds inputs",
        "cleanup_policy": "preserve small measurement checkpoints on SSD; no source bytes copied; no cleanup needed",
        "verdict_scope": "representation measurements and declared-family code-length estimates only; no receiver/archive or score authority",
    }
    atomic_json(args.out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
