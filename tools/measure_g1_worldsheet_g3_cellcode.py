#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure G1 worldsheet transport and G3 argmax-cell coding on real n600 data.

This is a design measurement only: ``[macOS-CPU advisory]``, no contest score,
and the borrowed-bank 0.18804 pointer is explicitly unmoved.  The 4.7 GiB GT
cache is never inflated: every ZIP_STORED NPY member is opened as a read-only
memmap.  Frame-0 labels are generated in batch-32 scorer geometry into a
resumable SSD sidecar with one preserved receipt per batch.

G1 measures all 600 within-pair transitions ``(2k -> 2k+1)`` and all 599
cross-pair transitions ``(2k+1 -> 2k+2)``.  The scorer only banks poses for the
former.  Cross-pair rows therefore use ``pose[k+1]`` as a clearly labelled
nearest-target-pair proxy; they are not exact cross-pair pose measurements.

G3 consumes the exact live batch-16 hard-oracle flip-stage inventory.  The
receiver baseline class is excluded from the locally admissible alphabet,
leaving four possible target cells.  Context priors are causal and Laplace
smoothed.  Ideal arithmetic-code bytes exclude coder headers and, critically,
assume the flip sites are supplied by another mechanism; this is an identity-
stream floor, not a byte-closed carrier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.lie._se3_numpy import exp_se3, rotation_of, translation_of  # noqa: E402

SCHEMA = "g1_worldsheet_g3_cellcode_measurements.v1"
AXIS = "[macOS-CPU advisory]"
POINTER = "0.18804"
N_PAIRS = 600
SEG_H, SEG_W = 384, 512
NATIVE_H, NATIVE_W = 874, 1164
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
CLASS_PAIRS = tuple((a, b) for a in range(5) for b in range(a + 1, 5))
GT_CACHE_SHA256 = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
BREAK_EVEN_EQUATION_ID = "realization_breakeven_bytes_v1"
R1B4_FIXTURE_CARRIER_BYTES = 2114.0
R1B4_FIXTURE_SOURCE = (
    "commit 1e574f44e1:.omx/research/"
    "r1b4_receiver_carrier_gate_20260720T194049Z.json"
)

# Already-settled n200 within-pair label calibration, applied rather than
# re-opened.  Source: screw_warp_through_R_gap2_20260629T195829Z.md.
CALIBRATION = {"s_t": -0.00143, "s_r": 0.0, "pitch_rad": -0.05}
CAMERA_HEIGHT_M = 1.22
NATIVE_INTRINSICS = {
    "fx": 910.0,
    "fy": 910.0,
    "cx": 582.0,
    "cy": 437.0,
}

RESIDUAL_HISTOGRAM_EDGES = np.array(
    [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, np.inf],
    dtype=np.float64,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def npz_member_memmap(path: Path, key: str) -> np.memmap:
    """Open one uncompressed NPY member of an NPZ without inflating its siblings."""
    member_name = f"{key}.npy"
    with zipfile.ZipFile(path, "r") as archive:
        info = archive.getinfo(member_name)
        if info.compress_type != zipfile.ZIP_STORED or info.compress_size != info.file_size:
            raise ValueError(f"{member_name} must be ZIP_STORED for bounded memmap access")
        with path.open("rb") as handle:
            handle.seek(info.header_offset)
            local = handle.read(30)
            if len(local) != 30:
                raise ValueError(f"truncated ZIP local header for {member_name}")
            values = struct.unpack("<IHHHHHIIIHH", local)
            if values[0] != 0x04034B50:
                raise ValueError(f"invalid ZIP local header signature for {member_name}")
            name_len, extra_len = values[-2:]
            handle.seek(info.header_offset + 30 + name_len + extra_len)
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            else:
                shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)
            data_offset = handle.tell()
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=data_offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )


def _tree_hash(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _load_flip_inventory(stage_dir: Path) -> tuple[list[list[float | int]], dict[str, Any]]:
    files = sorted(stage_dir.glob("batch-*.json"))
    if len(files) != 38:
        raise ValueError(f"expected 38 n600 batch-stage files, found {len(files)} in {stage_dir}")
    flips: list[list[float | int]] = []
    pair_cursor = 0
    cache_mismatches = 0
    for path in files:
        row = json.loads(path.read_text(encoding="utf-8"))
        start, stop = int(row["pair_start"]), int(row["pair_stop"])
        if start != pair_cursor or not start < stop <= N_PAIRS:
            raise ValueError(f"non-contiguous inventory stage {path}: {start}:{stop}, cursor={pair_cursor}")
        batch_flips = row["flips"]
        if int(row["flip_count"]) != len(batch_flips):
            raise ValueError(f"flip_count drift in {path}")
        flips.extend(batch_flips)
        cache_mismatches += int(row["cache_label_mismatches"])
        pair_cursor = stop
    if pair_cursor != N_PAIRS or len(flips) != 17_926:
        raise ValueError(f"inventory custody drift: pairs={pair_cursor}, flips={len(flips)}")
    keys = [(int(r[0]), int(r[1]), int(r[2])) for r in flips]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise ValueError("flip inventory must be unique and lexicographically sorted")
    return flips, {
        "stage_dir": str(stage_dir),
        "stage_count": len(files),
        "stage_tree_sha256": _tree_hash(files, stage_dir),
        "cache_label_mismatches": cache_mismatches,
    }


def _edge_stratum(labels: np.ndarray, pair: int, row: int, col: int) -> str:
    center = int(labels[pair, row, col])
    neighbors: list[int] = []
    if row:
        neighbors.append(int(labels[pair, row - 1, col]))
    if row + 1 < SEG_H:
        neighbors.append(int(labels[pair, row + 1, col]))
    if col:
        neighbors.append(int(labels[pair, row, col - 1]))
    if col + 1 < SEG_W:
        neighbors.append(int(labels[pair, row, col + 1]))
    is_edge = any(value != center for value in neighbors)
    road_lane = center in (0, 1) and any({center, value} == {0, 1} for value in neighbors)
    return "road_lane_edge" if road_lane else ("other_edge" if is_edge else "nonedge")


def _bit_costs_for_flip(
    labels: np.ndarray,
    pair: int,
    row: int,
    col: int,
    target: int,
    baseline: int,
) -> dict[str, float]:
    """Ideal causal code lengths for the target class at one known flip site."""
    if target == baseline:
        raise ValueError("flip target must differ from baseline class")
    spatial: list[int] = []
    # Raster-causal neighbors only: left and up target cells are decoded first.
    if col:
        spatial.append(int(labels[pair, row, col - 1]))
    if row:
        spatial.append(int(labels[pair, row - 1, col]))
    spatial_hits = sum(value == target for value in spatial)
    spatial_admissible = sum(value != baseline for value in spatial)
    p_spatial = (1.0 + spatial_hits) / (4.0 + spatial_admissible)

    previous = int(labels[pair - 1, row, col]) if pair else None
    temporal_hit = int(previous == target)
    temporal_admissible = int(previous is not None and previous != baseline)
    p_temporal = (1.0 + temporal_hit) / (4.0 + temporal_admissible)

    joint = spatial + ([] if previous is None else [previous])
    joint_hits = sum(value == target for value in joint)
    joint_admissible = sum(value != baseline for value in joint)
    p_joint = (1.0 + joint_hits) / (4.0 + joint_admissible)
    return {
        "uniform_5ary": math.log2(5.0),
        "uniform_4ary_excluding_baseline": 2.0,
        "spatial_potts_laplace": -math.log2(p_spatial),
        "temporal_same_site_laplace": -math.log2(p_temporal),
        "spatial_temporal_laplace": -math.log2(p_joint),
    }


def _sum_code_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    priors = tuple(rows[0]["bits"]) if rows else ()
    bits = {name: float(sum(float(row["bits"][name]) for row in rows)) for name in priors}
    return {
        "flip_count": len(rows),
        "ideal_bits": bits,
        "ideal_bytes": {name: value / 8.0 for name, value in bits.items()},
        "raw_coordinate": {
            "bits_per_site": 28,
            "bit_fields": {"pair": 10, "row": 9, "col": 9},
            "ideal_bits": len(rows) * 28,
            "ideal_bytes": len(rows) * 28 / 8.0,
            "packed_ceiling_bytes": math.ceil(len(rows) * 28 / 8.0),
        },
    }


def _consume_breakeven_equation() -> dict[str, Any]:
    from tac.canonical_equations.registry import get_equation_by_id

    equation = get_equation_by_id(BREAK_EVEN_EQUATION_ID)
    if equation is None:
        raise ValueError(f"canonical equation {BREAK_EVEN_EQUATION_ID!r} is absent")
    payload = equation.to_dict()
    anchors = payload.get("empirical_anchors") or []
    matches = [a for a in anchors if a.get("anchor_id") == "r2b_sparse_stream_breakeven_n600_20260720"]
    if len(matches) != 1:
        raise ValueError("n600 realization break-even anchor is absent or ambiguous")
    anchor = matches[0]
    value = float(anchor["empirical_output"]["breakeven_bytes"])
    module_name, callable_name = str(payload["python_callable_module_path"]).split(":", 1)
    module = __import__(module_name, fromlist=[callable_name])
    evaluated = float(getattr(module, callable_name)(float(anchor["inputs"]["realized_recovery_s"])))
    return {
        "equation_id": BREAK_EVEN_EQUATION_ID,
        "anchor_id": anchor["anchor_id"],
        "empirical_bytes": value,
        "callable_evaluated_bytes": evaluated,
        "callable_abs_residual_bytes": abs(evaluated - value),
        "python_callable_module_path": payload["python_callable_module_path"],
    }


def measure_g3(labels: np.ndarray, flip_stage_dir: Path) -> dict[str, Any]:
    flips, custody = _load_flip_inventory(flip_stage_dir)
    rows: list[dict[str, Any]] = []
    cache_target_mismatches = 0
    for pair_, row_, col_, target_, baseline_, margin_ in flips:
        pair, row, col = int(pair_), int(row_), int(col_)
        target, baseline, margin = int(target_), int(baseline_), abs(float(margin_))
        cache_target_mismatches += int(int(labels[pair, row, col]) != target)
        rows.append(
            {
                "pair": pair,
                "row": row,
                "col": col,
                "target_class": target,
                "baseline_class": baseline,
                "margin": margin,
                "margin_stratum": "moderate_[1e-3,1)" if margin >= 1e-3 else "tight_<1e-3",
                "edge_stratum": _edge_stratum(labels, pair, row, col),
                "class_transition": f"{CLASS_NAMES[target]}->{CLASS_NAMES[baseline]}",
                "bits": _bit_costs_for_flip(labels, pair, row, col, target, baseline),
            }
        )
    if cache_target_mismatches > custody["cache_label_mismatches"]:
        raise ValueError("live inventory/cache target mismatch exceeds stage custody")

    all_summary = _sum_code_rows(rows)
    moderate_rows = [row for row in rows if row["margin_stratum"] == "moderate_[1e-3,1)"]
    moderate = _sum_code_rows(moderate_rows)
    if moderate["flip_count"] != 16_319:
        raise ValueError(f"moderate-band inventory drift: {moderate['flip_count']} != 16319")

    def grouped(field: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        return {key: _sum_code_rows(group) for key, group in sorted(groups.items())}

    break_even = _consume_breakeven_equation()
    comparisons: dict[str, Any] = {}
    raw_bytes = float(all_summary["raw_coordinate"]["ideal_bytes"])
    for prior, byte_count in all_summary["ideal_bytes"].items():
        value = float(byte_count)
        comparisons[prior] = {
            "cell_to_raw_coordinate_ratio": value / raw_bytes,
            "saves_vs_raw_coordinate_bytes": raw_bytes - value,
            "below_realization_breakeven_1852": value < break_even["empirical_bytes"],
            "below_r1b4_fixture_carrier_2114": value < R1B4_FIXTURE_CARRIER_BYTES,
        }
    moderate_comparisons: dict[str, Any] = {}
    moderate_raw = float(moderate["raw_coordinate"]["ideal_bytes"])
    for prior, byte_count in moderate["ideal_bytes"].items():
        value = float(byte_count)
        moderate_comparisons[prior] = {
            "cell_to_raw_coordinate_ratio": value / moderate_raw,
            "below_realization_breakeven_1852": value < break_even["empirical_bytes"],
            "below_r1b4_fixture_carrier_2114": value < R1B4_FIXTURE_CARRIER_BYTES,
        }
    best_prior = min(all_summary["ideal_bytes"], key=all_summary["ideal_bytes"].get)
    return {
        "schema": "g3_cellcode_floor.v1",
        "axis": AXIS,
        "score_claim": False,
        "pointer_unmoved": POINTER,
        "inventory_custody": {**custody, "cache_target_mismatches": cache_target_mismatches},
        "alphabet": {
            "legal_full": "5 target argmax classes",
            "local": "4 target classes after excluding the known receiver baseline class",
            "site_location_cost_included": False,
            "headers_and_finite_coder_overhead_included": False,
        },
        "prior_assumptions": {
            "uniform_5ary": "uniform over all five target cells",
            "uniform_4ary_excluding_baseline": "target differs from known baseline at every inventoried flip",
            "spatial_potts_laplace": "left/up decoded target cells only; add-one smoothing over four alternatives",
            "temporal_same_site_laplace": "previous pair-index target at same site; add-one smoothing over four alternatives",
            "spatial_temporal_laplace": "union of the two causal contexts; add-one smoothing",
        },
        "all_flips": all_summary,
        "moderate_band": moderate,
        "per_pair": grouped("pair"),
        "per_edge_stratum": grouped("edge_stratum"),
        "per_class_transition": grouped("class_transition"),
        "comparators": {
            "realization_breakeven": break_even,
            "r1b4_fixture_carrier_bytes": R1B4_FIXTURE_CARRIER_BYTES,
            "r1b4_fixture_carrier_source": R1B4_FIXTURE_SOURCE,
        },
        "all_flip_comparisons": comparisons,
        "moderate_band_comparisons": moderate_comparisons,
        "best_measured_prior": best_prior,
        "verdict": "CELL_ID_PAYS_VS_RAW_COORDINATES_BUT_CHEAP_PRIORS_MISS_LIVE_BYTE_GATES",
        "verdict_scope": (
            "known-site ideal cell-identity stream under the measured uniform, local Potts, "
            "and same-site temporal priors; excludes site-location, candidate-set, coder-header, "
            "receiver, and realized-flip costs"
        ),
        "consumer": ["#572", "r1b5 GAP-3"],
    }


def _load_segnet(upstream: Path):
    sys.path.insert(0, str(upstream))
    import torch
    from modules import SegNet, segnet_sd_path
    from safetensors.torch import load_file

    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    model = SegNet().eval().to("cpu")
    model.load_state_dict(load_file(str(segnet_sd_path), device="cpu"))
    return torch, model


def _score_seg_batch(torch: Any, model: Any, frames: np.ndarray) -> np.ndarray:
    batch = torch.from_numpy(np.array(frames, copy=True)).permute(0, 3, 1, 2).float()
    pairs = batch[:, None].expand(-1, 2, -1, -1, -1)
    with torch.inference_mode():
        logits = model(model.preprocess_input(pairs))
    return logits.argmax(dim=1).cpu().numpy().astype(np.uint8)


def build_f0_labels(
    gt_f0: np.ndarray,
    gt_f1: np.ndarray,
    f1_labels: np.ndarray,
    *,
    output: Path,
    stage_dir: Path,
    upstream: Path,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Build or resume the complete batch-32 f0 SegNet label sidecar."""
    if batch_size != 32:
        raise ValueError("f0 labels must use the cache's batch-32 scorer geometry")
    if shutil.disk_usage(output.parent).free < 2 * 1024**3:
        raise ValueError("storage preflight refused: f0 sidecar tier needs >=2 GiB free")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    if output.exists():
        out = np.lib.format.open_memmap(output, mode="r+")
        if out.shape != (N_PAIRS, SEG_H, SEG_W) or out.dtype != np.uint8:
            raise ValueError(f"f0 label sidecar geometry drifted: {out.shape} {out.dtype}")
    else:
        out = np.lib.format.open_memmap(
            output, mode="w+", dtype=np.uint8, shape=(N_PAIRS, SEG_H, SEG_W)
        )
    missing = [
        (start, min(N_PAIRS, start + batch_size))
        for start in range(0, N_PAIRS, batch_size)
        if not (stage_dir / f"batch-{start:04d}.json").is_file()
    ]
    torch = model = None
    if missing:
        torch, model = _load_segnet(upstream)
    for start, stop in missing:
        f0 = _score_seg_batch(torch, model, gt_f0[start:stop])
        f1_check = _score_seg_batch(torch, model, gt_f1[start:stop])
        mismatches = int(np.count_nonzero(f1_check != f1_labels[start:stop]))
        out[start:stop] = f0
        out.flush()
        chunk = np.asarray(out[start:stop])
        atomic_json(
            stage_dir / f"batch-{start:04d}.json",
            {
                "schema": "g1_f0_label_batch.v1",
                "pair_start": start,
                "pair_stop": stop,
                "batch_size": stop - start,
                "scorer_batch_geometry": 32,
                "f1_cache_label_mismatches": mismatches,
                "f0_label_bytes_sha256": hashlib.sha256(chunk.tobytes(order="C")).hexdigest(),
            },
        )
        print(f"f0-label-stage {start}:{stop} complete", flush=True)
    del out
    stages = sorted(stage_dir.glob("batch-*.json"))
    if len(stages) != math.ceil(N_PAIRS / batch_size):
        raise ValueError("f0 sidecar stage closure incomplete")
    stage_rows = [json.loads(path.read_text(encoding="utf-8")) for path in stages]
    total_f1_mismatches = sum(int(row["f1_cache_label_mismatches"]) for row in stage_rows)
    manifest = {
        "schema": "g1_f0_label_sidecar.v1",
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "shape": [N_PAIRS, SEG_H, SEG_W],
        "dtype": "uint8",
        "scorer_batch_geometry": 32,
        "f1_cache_label_mismatches": total_f1_mismatches,
        "f1_cache_binding_status": (
            "EXACT" if total_f1_mismatches == 0 else "MEASURED_KERNEL_GEOMETRY_DRIFT"
        ),
        "stage_count": len(stages),
        "stage_tree_sha256": _tree_hash(stages, stage_dir),
        "rebuildable": True,
        "cleanup_policy": "certify before cold-store or deletion; preserve this manifest and stage tree",
    }
    atomic_json(stage_dir / "manifest.json", manifest)
    return manifest


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
    """Ground-plane H from PoseNet output through tac.lie's translation-first exp."""
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


def extract_interclass_edges(labels: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Return both sides of every 4-neighbor interclass boundary, by class pair."""
    masks = {pair: np.zeros((SEG_H, SEG_W), dtype=bool) for pair in CLASS_PAIRS}
    left, right = labels[:, :-1], labels[:, 1:]
    ys, xs = np.nonzero(left != right)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        pair = tuple(sorted((int(left[y, x]), int(right[y, x]))))
        masks[pair][y, x] = True
        masks[pair][y, x + 1] = True
    top, bottom = labels[:-1, :], labels[1:, :]
    ys, xs = np.nonzero(top != bottom)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        pair = tuple(sorted((int(top[y, x]), int(bottom[y, x]))))
        masks[pair][y, x] = True
        masks[pair][y + 1, x] = True
    return {
        pair: np.column_stack(np.nonzero(mask)).astype(np.float64)
        for pair, mask in masks.items()
    }


def _warp_points(points_yx: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(points_yx):
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=bool)
    xy1 = np.column_stack((points_yx[:, 1], points_yx[:, 0], np.ones(len(points_yx))))
    warped = (homography @ xy1.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        x = warped[:, 0] / warped[:, 2]
        y = warped[:, 1] / warped[:, 2]
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (warped[:, 2] > 0)
        & (x >= 0)
        & (x <= SEG_W - 1)
        & (y >= 0)
        & (y <= SEG_H - 1)
    )
    return np.column_stack((y[valid], x[valid])), valid


def symmetric_chamfer_row(
    source_yx: np.ndarray,
    target_yx: np.ndarray,
    homography: np.ndarray,
) -> dict[str, Any]:
    """Symmetric nearest-neighbor residuals; invalid transported points are events."""
    from scipy.spatial import cKDTree

    source_n, target_n = len(source_yx), len(target_yx)
    warped, valid = _warp_points(source_yx, homography)
    invalid_n = source_n - len(warped)
    state = "both_present"
    if not source_n and not target_n:
        state = "both_empty"
    elif not source_n:
        state = "birth"
    elif not target_n:
        state = "death"
    finite = np.empty(0, dtype=np.float64)
    if len(warped) and target_n:
        d_source = cKDTree(target_yx).query(warped, k=1, workers=1)[0]
        d_target = cKDTree(warped).query(target_yx, k=1, workers=1)[0]
        finite = np.concatenate((d_source, d_target)).astype(np.float64)
    total_observations = source_n + target_n
    infinite_events = invalid_n
    if source_n and not target_n:
        # Every source observation is an unmatched death.  Do not double-count
        # the subset that also transported out of bounds.
        infinite_events = source_n
    elif target_n and not len(warped):
        infinite_events += target_n
    # When target exists and some transported points exist, target-to-source is
    # finite; only invalid source points remain infinite events.
    denom = max(total_observations, 1)
    event_rates = {
        str(radius): float((np.count_nonzero(finite > radius) + infinite_events) / denom)
        for radius in (1, 2, 4)
    }
    hist = np.histogram(finite, bins=RESIDUAL_HISTOGRAM_EDGES)[0].astype(int).tolist()
    return {
        "source_edge_pixels": source_n,
        "target_edge_pixels": target_n,
        "valid_transported_source_pixels": len(warped),
        "invalid_transported_source_pixels": invalid_n,
        "presence_state": state,
        "finite_residual_count": len(finite),
        "infinite_event_count": int(infinite_events),
        "symmetric_chamfer_px_finite_mean": None if not len(finite) else float(finite.mean()),
        "median_residual_px_finite": None if not len(finite) else float(np.median(finite)),
        "event_fraction_gt_px": event_rates,
        "residual_histogram_counts": hist,
    }


def _aggregate_g1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["transition_type"], row["stratum"])].append(row)

    def aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        finite_count = sum(int(r["finite_residual_count"]) for r in group)
        # Mean of means is weighted by its exact finite residual count.
        finite_sum = sum(
            float(r["symmetric_chamfer_px_finite_mean"]) * int(r["finite_residual_count"])
            for r in group
            if r["symmetric_chamfer_px_finite_mean"] is not None
        )
        observations = sum(int(r["source_edge_pixels"]) + int(r["target_edge_pixels"]) for r in group)
        hist = np.sum(
            np.asarray([r["residual_histogram_counts"] for r in group], dtype=np.int64), axis=0
        ).tolist()
        medians = [float(r["median_residual_px_finite"]) for r in group if r["median_residual_px_finite"] is not None]
        return {
            "transition_count": len(group),
            "presence_states": dict(Counter(r["presence_state"] for r in group)),
            "finite_residual_count": finite_count,
            "observation_count": observations,
            "weighted_finite_chamfer_mean_px": None if not finite_count else finite_sum / finite_count,
            "median_of_transition_medians_px": None if not medians else float(np.median(medians)),
            "event_fraction_gt_px": {
                str(radius): sum(
                    float(r["event_fraction_gt_px"][str(radius)])
                    * (int(r["source_edge_pixels"]) + int(r["target_edge_pixels"]))
                    for r in group
                )
                / max(observations, 1)
                for radius in (1, 2, 4)
            },
            "residual_histogram_counts": hist,
        }

    by_transition_stratum = {
        f"{kind}:{stratum}": aggregate(group)
        for (kind, stratum), group in sorted(groups.items())
    }
    by_transition = {
        kind: aggregate([r for r in rows if r["transition_type"] == kind])
        for kind in ("within_pair", "cross_pair")
    }
    by_stratum = {
        stratum: aggregate([r for r in rows if r["stratum"] == stratum])
        for stratum in sorted({r["stratum"] for r in rows})
    }
    return {
        "histogram_edges_px": [*RESIDUAL_HISTOGRAM_EDGES.tolist()[:-1], "inf"],
        "by_transition": by_transition,
        "by_stratum": by_stratum,
        "by_transition_and_stratum": by_transition_stratum,
    }


def measure_g1(
    f0_labels: np.ndarray,
    f1_labels: np.ndarray,
    poses: np.ndarray,
    *,
    stage_dir: Path,
) -> dict[str, Any]:
    if f0_labels.shape != (N_PAIRS, SEG_H, SEG_W):
        raise ValueError(f"f0 labels have wrong shape {f0_labels.shape}")
    if f1_labels.shape != (N_PAIRS, SEG_H, SEG_W) or poses.shape != (N_PAIRS, 6):
        raise ValueError("f1-label or pose geometry drifted")
    stage_dir.mkdir(parents=True, exist_ok=True)
    for pair in range(N_PAIRS):
        stage_path = stage_dir / f"pair-{pair:04d}.json"
        if stage_path.is_file():
            continue
        pair_rows: list[dict[str, Any]] = []
        f0_edges = extract_interclass_edges(f0_labels[pair])
        f1_edges = extract_interclass_edges(f1_labels[pair])
        h_within = pose_homography(poses[pair])
        for class_pair in CLASS_PAIRS:
            stratum = f"{CLASS_NAMES[class_pair[0]]}-{CLASS_NAMES[class_pair[1]]}"
            row = symmetric_chamfer_row(f0_edges[class_pair], f1_edges[class_pair], h_within)
            pair_rows.append(
                {
                    "transition_type": "within_pair",
                    "transition_index": pair,
                    "source_frame": 2 * pair,
                    "target_frame": 2 * pair + 1,
                    "pose_target_index": pair,
                    "pose_authority": "exact banked target for this non-overlapping pair",
                    "stratum": stratum,
                    **row,
                }
            )
        if pair + 1 < N_PAIRS:
            next_f0_edges = extract_interclass_edges(f0_labels[pair + 1])
            h_cross = pose_homography(poses[pair + 1])
            for class_pair in CLASS_PAIRS:
                stratum = f"{CLASS_NAMES[class_pair[0]]}-{CLASS_NAMES[class_pair[1]]}"
                row = symmetric_chamfer_row(f1_edges[class_pair], next_f0_edges[class_pair], h_cross)
                pair_rows.append(
                    {
                        "transition_type": "cross_pair",
                        "transition_index": pair,
                        "source_frame": 2 * pair + 1,
                        "target_frame": 2 * pair + 2,
                        "pose_target_index": pair + 1,
                        "pose_authority": "nearest target-pair proxy; no banked cross-pair PoseNet target exists",
                        "stratum": stratum,
                        **row,
                    }
                )
        expected_pair_rows = len(CLASS_PAIRS) * (2 if pair + 1 < N_PAIRS else 1)
        if len(pair_rows) != expected_pair_rows:
            raise AssertionError(f"G1 pair-stage row count drifted at {pair}")
        atomic_json(
            stage_path,
            {
                "schema": "g1_worldsheet_pair_stage.v1",
                "pair_index": pair,
                "rows": pair_rows,
            },
        )
        print(f"g1-pair {pair + 1}/{N_PAIRS}", flush=True)
    stage_paths = sorted(stage_dir.glob("pair-*.json"))
    if len(stage_paths) != N_PAIRS:
        raise ValueError(f"G1 stage closure incomplete: {len(stage_paths)} != {N_PAIRS}")
    rows: list[dict[str, Any]] = []
    for pair, path in enumerate(stage_paths):
        stage = json.loads(path.read_text(encoding="utf-8"))
        if int(stage["pair_index"]) != pair:
            raise ValueError(f"G1 pair-stage ordering drifted in {path}")
        rows.extend(stage["rows"])
    if len(rows) != (N_PAIRS + N_PAIRS - 1) * len(CLASS_PAIRS):
        raise AssertionError("G1 transition/stratum row count drifted")
    aggregate = _aggregate_g1(rows)
    worst = sorted(
        (r for r in rows if r["symmetric_chamfer_px_finite_mean"] is not None),
        key=lambda r: (float(r["event_fraction_gt_px"]["4"]), float(r["symmetric_chamfer_px_finite_mean"])),
        reverse=True,
    )[:30]
    global_rows = list(aggregate["by_transition"].values())
    broadband = any(
        row["median_of_transition_medians_px"] is not None
        and (
            float(row["median_of_transition_medians_px"]) > 1.0
            or float(row["event_fraction_gt_px"]["4"]) > 0.10
        )
        for row in global_rows
    )
    return {
        "schema": "g1_worldsheet_transport_fidelity.v1",
        "axis": AXIS,
        "score_claim": False,
        "pointer_unmoved": POINTER,
        "adjacency": {
            "pair_structure": "(2k,2k+1)",
            "within_pair_transitions": N_PAIRS,
            "cross_pair_transitions": N_PAIRS - 1,
            "total_sequential_transitions": 2 * N_PAIRS - 1,
            "rows_per_transition": len(CLASS_PAIRS),
        },
        "transport_assumptions": {
            "formulation": "single ground-plane homography applied to every interclass edge stratum",
            "pose_source": "gt_poses 600x6 frozen PoseNet targets from scorer cache",
            "se3_engine": "tac.lie._se3_numpy.exp_se3, translation-first (rho,omega)",
            "posenet_to_twist_map": "rho=s_t*[pose2,pose1,pose0]; omega=s_r*pose[3:6]",
            "calibration": CALIBRATION,
            "calibration_source": "settled n200 within-pair label fit; not re-opened in this pass",
            "camera_height_m": CAMERA_HEIGHT_M,
            "native_intrinsics": NATIVE_INTRINSICS,
            "cross_pair_pose_limitation": (
                "no exact banked cross-pair target; uses pose[k+1] nearest-target-pair proxy"
            ),
        },
        "metric_definition": {
            "edge_extraction": "both pixels of every unlike 4-neighbor adjacency, unordered class pair",
            "symmetric_chamfer": "mean of transported-source-to-target and target-to-transported-source nearest distances",
            "empty_strata": "birth/death tracked; no finite Chamfer fabricated",
            "invalid_transport": "counted as residual events at all thresholds",
            "event_thresholds_px": [1, 2, 4],
        },
        "verdict_operationalization": {
            "sparse_event_go": (
                "for each cadence aggregate, median_of_transition_medians_px <= 1 and "
                "event_fraction_gt4_px <= 0.10"
            ),
            "status": "analysis convention made explicit after the qualitative pre-registration; not a score gate",
        },
        "aggregate": aggregate,
        "stage_custody": {
            "stage_dir": str(stage_dir),
            "stage_count": len(stage_paths),
            "stage_tree_sha256": _tree_hash(stage_paths, stage_dir),
        },
        "per_transition_per_stratum": rows,
        "worst_30_rows_by_event_gt4_then_chamfer": worst,
        "verdict": (
            "GROUND_PLANE_WARP_REALIZATION_BROADBAND_NEGATIVE"
            if broadband
            else "GROUND_PLANE_WARP_REALIZATION_SPARSE_EVENT_GO"
        ),
        "verdict_scope": (
            "single global ground-plane-homography realization using exact within-pair poses and "
            "nearest-target-pair proxy cross poses; not the worldsheet object/family"
        ),
        "elevation_target": "post-row #574",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--flip-stage-dir", type=Path, required=True)
    parser.add_argument("--f0-label-cache", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("g3", "build-f0", "g1", "all"), default="all")
    parser.add_argument("--skip-cache-sha", action="store_true", help="tests only; never use for n600 verdict")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    if not args.cache.is_file():
        raise SystemExit(f"missing cache: {args.cache}")
    cache_sha = "SKIPPED_TEST_ONLY" if args.skip_cache_sha else sha256_file(args.cache)
    if not args.skip_cache_sha and cache_sha != GT_CACHE_SHA256:
        raise SystemExit(f"GT cache SHA-256 drifted: {cache_sha}")
    labels = npz_member_memmap(args.cache, "lstars")
    poses = npz_member_memmap(args.cache, "gt_poses")
    if labels.shape != (N_PAIRS, SEG_H, SEG_W) or poses.shape != (N_PAIRS, 6):
        raise SystemExit(f"n600 cache geometry drift: labels={labels.shape} poses={poses.shape}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    parts: dict[str, Any] = {}
    if args.stage in {"g3", "all"}:
        parts["g3"] = measure_g3(labels, args.flip_stage_dir)
        atomic_json(args.work_dir / "g3.json", parts["g3"])
    if args.stage in {"build-f0", "all"}:
        gt_f0 = npz_member_memmap(args.cache, "gt_f0")
        gt_f1 = npz_member_memmap(args.cache, "gt_f1")
        parts["f0_label_sidecar"] = build_f0_labels(
            gt_f0,
            gt_f1,
            labels,
            output=args.f0_label_cache,
            stage_dir=args.work_dir / "f0_label_stages",
            upstream=args.upstream,
        )
    if args.stage in {"g1", "all"}:
        if not args.f0_label_cache.is_file():
            raise SystemExit(f"missing f0 label sidecar: {args.f0_label_cache}")
        f0_labels = np.load(args.f0_label_cache, mmap_mode="r", allow_pickle=False)
        parts["g1"] = measure_g1(
            f0_labels,
            labels,
            poses,
            stage_dir=args.work_dir / "g1_transition_stages",
        )
        atomic_json(args.work_dir / "g1.json", parts["g1"])
    receipt = {
        "schema": SCHEMA,
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": f"{POINTER} UNMOVED",
        "cache": {
            "path": str(args.cache),
            "bytes": args.cache.stat().st_size,
            "sha256": cache_sha,
            "access": "ZIP_STORED member memmaps; never dense-loaded",
        },
        "command": " ".join(sys.argv),
        "tool_sha256": sha256_file(Path(__file__)),
        "elapsed_seconds": time.time() - started,
        **parts,
        "main_landing_review_required": True,
    }
    atomic_json(args.out, receipt)
    print(json.dumps({"out": str(args.out), "keys": sorted(parts), "elapsed_seconds": receipt["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
