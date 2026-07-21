# SPDX-License-Identifier: MIT
"""Compose real n600 PREDICT-to-PROJECT seed curves from the solved S2 object.

This is a compression-time tool/library.  It consumes the frozen CPU-Torch
target cache and the finite S2 event packet, emits only schema-valid counted
seed bytes, and keeps the known Morse-Smale native-rasterizer gap explicit.
The five-site chart used here is the schema's deterministic compatibility
raster, not a claim of native arc-to-cell rasterization semantics.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.boundary_math.ego_xi_trajectory import PoseTargetEgoEstimator
from tac.boundary_math.movable_site_coder import extract_movable_sites, track_sites
from tac.optimization.predict_project_receiver import (
    _interpolate_track_knot,
    _nearest_shift,
    component_byte_accounting,
)
from tac.optimization.predict_project_schema import (
    build_minimal_constraint_seed,
    canonical_json_bytes,
    derive_morse_smale_raster,
    serialize_constraint_seed,
    validate_constraint_seed,
)
from tac.optimization.s2_partition_seed import PartitionEvent, decode_partition_seed
from tac.witness_dsl.lawref import LADDER_MEASURED_ANCHOR, InputRef, LawRef, resolve

PAIR_COUNT: Final = 600
SCORER_HEIGHT: Final = 384
SCORER_WIDTH: Final = 512
CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
GT_CACHE_SHA256: Final = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
S2_PACKET_SHA256: Final = "df4c0534537a9919681509a0b44a392d7d4b46c812d7570c534e6b823adae7fc"
G1_RECEIPT_SHA256: Final = "38b1f5d5475037e360ce13f5aed7ae114d9e3c4834e7bffe388f0fb748fc5089"
G1_RECEIPT_PATH: Final = ".omx/research/g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.json"
RATE_DENOMINATOR_BYTES: Final = 37_545_489
MS_NATIVE_BLOCKER: Final = "MS_ARC_TO_CELL_RASTERIZATION_SEMANTICS_UNMEASURED"
CURVE_NAMES: Final = ("loose", "knee", "tight")


class SeedComposeError(ValueError):
    """Refuse missing custody, malformed inputs, or noncanonical outputs."""


def sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")


def _require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise SeedComposeError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise SeedComposeError(f"{label} SHA-256 mismatch: expected {expected}, got {observed}")


def _lawref_g1_calibration(repository_root: Path) -> tuple[float, float, dict[str, Any]]:
    receipt = repository_root / G1_RECEIPT_PATH
    _require_sha(receipt, G1_RECEIPT_SHA256, "G1 measurement receipt")
    resolutions: dict[str, Any] = {}
    values: list[float] = []
    for name in ("s_t", "s_r"):
        ref = LawRef(
            equation_id="dsl_custodied_scalar_identity_v1",
            inputs={
                "value": InputRef.anchor(
                    str(receipt),
                    f"g1/transport_assumptions/calibration/{name}",
                    f"MEASURED G1 PoseNet-to-xi calibration {name}",
                    expected_sha256=G1_RECEIPT_SHA256,
                    config_tags={"axis": "macOS-CPU-advisory", "scope": "n200_G1"},
                )
            },
            ladder_class=LADDER_MEASURED_ANCHOR,
        )
        resolution = resolve(ref)
        values.append(float(resolution.value))
        resolutions[name] = {
            "equation_id": ref.equation_id,
            "resolved_value": float(resolution.value),
            "source_path": G1_RECEIPT_PATH,
            "source_sha256": G1_RECEIPT_SHA256,
        }
    return values[0], values[1], resolutions


def _temporal_mode_and_centroids(labels: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    """Return mode plus full-occupancy centroids without stacking n600."""

    if labels.shape != (PAIR_COUNT, SCORER_HEIGHT, SCORER_WIDTH):
        raise SeedComposeError(f"lstars geometry mismatch: {labels.shape}")
    counts = np.zeros((5, SCORER_HEIGHT, SCORER_WIDTH), dtype=np.uint16)
    for pair in range(PAIR_COUNT):
        plane = np.asarray(labels[pair])
        for class_id in range(5):
            counts[class_id] += plane == class_id
    centroids: dict[int, tuple[int, int]] = {}
    yy, xx = np.indices((SCORER_HEIGHT, SCORER_WIDTH), dtype=np.int64)
    for class_id in range(5):
        weights = counts[class_id].astype(np.int64)
        total = int(weights.sum())
        if not total:
            raise SeedComposeError(f"n600 labels have no occupancy for class {class_id}")
        centroids[class_id] = (
            round(float(np.sum(weights * yy)) / total * 256),
            round(float(np.sum(weights * xx)) / total * 256),
        )
    return np.argmax(counts, axis=0).astype(np.uint8), centroids


def temporal_mode_chart(labels: np.ndarray) -> np.ndarray:
    """Return the deterministic per-site temporal mode without stacking n600."""

    return _temporal_mode_and_centroids(labels)[0]


def compatibility_site_chart(
    mode: np.ndarray,
    *,
    occupancy_centroids_q: Mapping[int, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Build a compact five-site graph for the explicitly scoped fixture raster."""

    quantum = 256
    cells: list[dict[str, Any]] = []
    centroids: dict[int, tuple[int, int]] = {}
    for class_id in range(5):
        ys, xs = np.nonzero(mode == class_id)
        if len(ys):
            centroids[class_id] = (round(float(np.mean(ys)) * quantum), round(float(np.mean(xs)) * quantum))
        elif occupancy_centroids_q is not None and class_id in occupancy_centroids_q:
            centroids[class_id] = tuple(occupancy_centroids_q[class_id])
        else:
            raise SeedComposeError(f"temporal-mode chart has no class {class_id} or full-occupancy centroid")

    edges: set[tuple[int, int]] = set()
    for left, right in ((mode[:, :-1], mode[:, 1:]), (mode[:-1, :], mode[1:, :])):
        for a, b in np.unique(np.stack([left[left != right], right[left != right]], axis=1), axis=0):
            edges.add((min(int(a), int(b)), max(int(a), int(b))))
    adjacency: dict[int, list[int]] = {class_id: [] for class_id in range(5)}
    for left, right in sorted(edges):
        adjacency[left].append(right)
        adjacency[right].append(left)
    for class_id in range(5):
        y_q, x_q = centroids[class_id]
        cells.append(
            {
                "cell_id": class_id,
                "class_id": class_id,
                "site_y_q": y_q,
                "site_x_q": x_q,
                "adjacent_cell_ids": sorted(adjacency[class_id]),
            }
        )
    critical = [
        {
            "critical_id": class_id,
            "type": "minimum",
            "persistence_q": 5 - class_id,
            "ground_y_q": centroids[class_id][0],
            "ground_x_q": centroids[class_id][1],
        }
        for class_id in range(5)
    ]
    arcs = []
    for arc_id, (left, right) in enumerate(sorted(edges)):
        y0, x0 = centroids[left]
        y1, x1 = centroids[right]
        dy, dx = y1 - y0, x1 - x0
        normal_y = 0 if dx == 0 else (quantum if dx > 0 else -quantum)
        normal_x = 0 if dy == 0 else (-quantum if dy > 0 else quantum)
        arcs.append(
            {
                "arc_id": arc_id,
                "source_critical_id": left,
                "target_critical_id": right,
                "left_cell_id": left,
                "right_cell_id": right,
                "samples": [
                    {"arc_index": 0, "ground_y_q": y0, "ground_x_q": x0, "normal_y_q": normal_y, "normal_x_q": normal_x},
                    {"arc_index": 1, "ground_y_q": y1, "ground_x_q": x1, "normal_y_q": normal_y, "normal_x_q": normal_x},
                ],
            }
        )
    graph: dict[str, Any] = {
        "geometry": {
            "scorer_height": SCORER_HEIGHT,
            "scorer_width": SCORER_WIDTH,
            "camera_height": CAMERA_HEIGHT,
            "camera_width": CAMERA_WIDTH,
            "class_count": 5,
        },
        "representation": "morse_smale_graph_vineyard.v1",
        "coordinate_quantum": {"numerator": 1, "denominator": quantum, "unit": "scorer_pixel"},
        "critical_points": critical,
        "separatrix_arcs": arcs,
        "cells": cells,
        "canonical_traversal": {
            "critical_order": "persistence_desc_type_id",
            "arc_order": "source_target_arc_id",
            "sample_order": "arc_index",
            "cell_order": "cell_id",
            "raster_tie_policy": "nearest_site_then_min_cell_id",
        },
        "vineyard_events": [],
        "derived_raster_fixture": {
            "derivation_id": "nearest_ground_cell_site_then_min_cell_id.v1",
            "content_sha256": "0" * 64,
        },
    }
    graph["derived_raster_fixture"]["content_sha256"] = hashlib.sha256(derive_morse_smale_raster(graph)).hexdigest()
    return graph


def _movable_tracks(labels: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sites = [extract_movable_sites(np.asarray(labels[pair])) for pair in range(PAIR_COUNT)]
    tracked = track_sites(sites)
    tracks: list[dict[str, Any]] = []
    for slot in range(tracked.K):
        times = np.nonzero(tracked.presence[:, slot])[0]
        if not len(times):
            continue
        knots = []
        for time_index in times.tolist():
            cx, cy, width, height = tracked.M[time_index, slot * 4 : slot * 4 + 4]
            knots.append(
                {
                    "time": time_index,
                    "y_q": round(cy * 256),
                    "x_q": round(cx * 256),
                    "height_q": max(1, round(height * 256)),
                    "width_q": max(1, round(width * 256)),
                }
            )
        tracks.append({"track_id": len(tracks), "cell_id": 3, "knots": knots})
    return tracks, {
        "canonical_helper": "tac.boundary_math.movable_site_coder.extract_movable_sites+track_sites",
        "track_count": len(tracks),
        "max_concurrent_sites": tracked.K,
        "matched_assignments": tracked.n_matched,
        "provenance": tracked.provenance,
    }


def _catmull(values: Sequence[int], times: Sequence[int], time_index: int) -> float:
    if time_index <= times[0]:
        return float(values[0])
    if time_index >= times[-1]:
        return float(values[-1])
    right = next(index for index, knot_time in enumerate(times) if knot_time >= time_index)
    left = right - 1
    p0 = float(values[max(0, left - 1)])
    p1 = float(values[left])
    p2 = float(values[right])
    p3 = float(values[min(len(values) - 1, right + 1)])
    u = (time_index - times[left]) / (times[right] - times[left])
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u * u
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u * u * u
    )


def _trajectory(
    gt_poses: np.ndarray,
    curve_index: int,
    *,
    s_t: float,
    s_r: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    estimator = PoseTargetEgoEstimator(fwd_ch=0, lat_ch=2, yaw_ch=4, s_t=s_t, s_r=s_r, calib="geometry")
    xi = estimator.estimate(gt_poses)
    dense = np.stack(
        [np.rint(xi.dy * 256), np.rint(xi.ds * 256), np.rint(xi.dpsi * 1_048_576)], axis=1
    ).astype(np.int64)
    control_counts = (math.ceil(math.log2(PAIR_COUNT)), math.ceil(math.sqrt(PAIR_COUNT)), PAIR_COUNT - 1)
    count = control_counts[curve_index]
    times = np.unique(np.rint(np.linspace(0, PAIR_COUNT - 1, count)).astype(np.int64)).tolist()
    if times[0] != 0 or times[-1] != PAIR_COUNT - 1 or not 2 <= len(times) < PAIR_COUNT:
        raise SeedComposeError("derived trajectory control schedule is not endpoint-complete")
    controls = [
        {"time": time_index, "tx_q": int(dense[time_index, 0]), "ty_q": int(dense[time_index, 1]), "yaw_q": int(dense[time_index, 2])}
        for time_index in times
    ]
    base = np.empty_like(dense)
    for dimension in range(3):
        values = [int(dense[time_index, dimension]) for time_index in times]
        base[:, dimension] = np.rint([_catmull(values, times, pair) for pair in range(PAIR_COUNT)]).astype(np.int64)
    raw_residual = dense - base
    nonzero = np.abs(raw_residual[np.nonzero(raw_residual)])
    if curve_index == 0:
        step = max(1, int(np.quantile(nonzero, 0.5))) if len(nonzero) else 1
    elif curve_index == 1:
        step = max(1, int(np.quantile(nonzero, 0.25))) if len(nonzero) else 1
    else:
        step = 1
    quantized = np.rint(raw_residual / step).astype(np.int64) * step
    residuals = [
        {"time": pair, "dtx_q": int(row[0]), "dty_q": int(row[1]), "dyaw_q": int(row[2])}
        for pair, row in enumerate(quantized)
        if bool(np.any(row))
    ]
    reconstructed = base + quantized
    error = dense - reconstructed
    return (
        {
            "representation": "cubic_catmull_rom_plus_ar_residual.v1",
            "step_count": PAIR_COUNT,
            "controls": controls,
            "ar_residuals": residuals,
        },
        {
            "canonical_helper": "tac.boundary_math.ego_xi_trajectory.PoseTargetEgoEstimator",
            "control_count": len(controls),
            "residual_count": len(residuals),
            "residual_quantum_q": step,
            "rms_reconstruction_q": float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
            "max_abs_reconstruction_q": int(np.max(np.abs(error))),
            "estimator_provenance": xi.provenance,
        },
    )


def _load_margin_map(inventory_dir: Path) -> tuple[dict[tuple[int, int, int], float], dict[str, Any]]:
    margins: dict[tuple[int, int, int], float] = {}
    label_mismatches = 0
    rows = 0
    files = sorted(inventory_dir.glob("batch-*.json"))
    if not files:
        raise SeedComposeError(f"no R2B inventory batches under {inventory_dir}")
    digest = hashlib.sha256()
    for path in files:
        payload = path.read_bytes()
        digest.update(path.name.encode("utf-8") + b"\0" + hashlib.sha256(payload).digest())
        value = json.loads(payload)
        label_mismatches += int(value["cache_label_mismatches"])
        for row in value["flips"]:
            margins[(int(row[0]), int(row[1]), int(row[2]))] = float(row[5])
            rows += 1
    return margins, {
        "stage_count": len(files),
        "row_count": rows,
        "cache_label_mismatches": label_mismatches,
        "stage_tree_sha256": digest.hexdigest(),
    }


def _stratum(event: PartitionEvent, target: int, margin: float) -> str:
    pair = {int(target), int(event.baseline_class)}
    if pair == {0, 1}:
        return "boundary_codim1"
    if 3 in pair:
        return "movable_track"
    if abs(margin) < 1e-3:
        return "critical_event"
    return "cell_interior"


def _event_rank(event: PartitionEvent, target: int, margin: float) -> tuple[Any, ...]:
    pair = {int(target), int(event.baseline_class)}
    return (
        0 if pair == {0, 1} else 1,
        0 if 1 in pair else 1,
        abs(float(margin)),
        event.pair,
        event.row,
        event.col,
    )


def _ordered_with_pair_coverage(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[int(row["event"].pair)].append(row)
    missing = sorted(set(range(PAIR_COUNT)) - set(by_pair))
    if missing:
        raise SeedComposeError(f"predictor-violated S2 rows do not cover all pairs: {missing[:8]}")
    pair_first = [min(by_pair[pair], key=lambda row: row["rank"]) for pair in range(PAIR_COUNT)]
    first_sites = {(row["event"].pair, row["event"].row, row["event"].col) for row in pair_first}
    remainder = [
        row
        for row in rows
        if (row["event"].pair, row["event"].row, row["event"].col) not in first_sites
    ]
    return sorted(pair_first, key=lambda row: row["rank"]) + sorted(remainder, key=lambda row: row["rank"])


def _constraints(rows: Sequence[dict[str, Any]], pose_q: np.ndarray, radius_q: int) -> list[dict[str, Any]]:
    first_by_pair: set[int] = set()
    constraints = []
    for row in sorted(rows, key=lambda item: (item["event"].pair, item["event"].row, item["event"].col)):
        event: PartitionEvent = row["event"]
        tube = None
        if event.pair not in first_by_pair:
            center = pose_q[event.pair]
            tube = {
                "lower_q": [int(value - radius_q) for value in center],
                "upper_q": [int(value + radius_q) for value in center],
            }
            first_by_pair.add(event.pair)
        constraints.append(
            {
                "time": event.pair,
                "frame_index": 1,
                "obligation": "seg_and_pose",
                "y": event.row,
                "x": event.col,
                "cell_id": int(row["target"]),
                "predictor_status": "violated",
                "stratum": row["stratum"],
                "pose_tube": tube,
                "pose_tightening_id": None,
                "projector": None,
            }
        )
    if first_by_pair != set(range(PAIR_COUNT)):
        raise SeedComposeError("pose-tube anchors are not full n600")
    return constraints


def _seed_base(chart: Mapping[str, Any], tracks: Sequence[Mapping[str, Any]], trajectory: Mapping[str, Any]) -> dict[str, Any]:
    seed = build_minimal_constraint_seed(
        bytes([0, 1, 2, 3]),
        scorer_height=2,
        scorer_width=2,
        camera_height=4,
        camera_width=4,
        seed=1234,
    )
    seed["ground_chart"] = copy.deepcopy(chart)
    seed["trajectory"] = copy.deepcopy(trajectory)
    seed["movable_tracks"] = copy.deepcopy(list(tracks))
    seed["constraint_seeds"] = []
    seed["boundary_jitter"]["equal_fidelity_custody"]["represented_constraints_sha256"] = hashlib.sha256(
        canonical_json_bytes([])
    ).hexdigest()
    return validate_constraint_seed(seed)


def _predict_prevalidated(seed: Mapping[str, Any], chart: np.ndarray, time_index: int) -> np.ndarray:
    """Fast exact R0 path after one canonical validation.

    The composed seeds have no lifecycle events or causal offsets.  This helper
    reuses the receiver's exact shift and track interpolation primitives while
    avoiding four full schema validations and a Python site-raster derivation
    per pair.  Tests compare it directly with ``predict_cell_field``.
    """

    trajectory = seed["trajectory"]
    controls = trajectory["controls"]
    times = [row["time"] for row in controls]
    residual_by_time = {row["time"]: row for row in trajectory["ar_residuals"]}
    residual = residual_by_time.get(time_index, {"dtx_q": 0, "dty_q": 0, "dyaw_q": 0})
    tx_q = _catmull([row["tx_q"] for row in controls], times, time_index) + residual["dtx_q"]
    ty_q = _catmull([row["ty_q"] for row in controls], times, time_index) + residual["dty_q"]
    yaw_q = _catmull([row["yaw_q"] for row in controls], times, time_index) + residual["dyaw_q"]
    out = _nearest_shift(chart, tx_q / 256.0, ty_q / 256.0, yaw_q / 1_048_576.0)
    for track in seed["movable_tracks"]:
        knots = track["knots"]
        if time_index < knots[0]["time"] or time_index > knots[-1]["time"]:
            continue
        center_y = _interpolate_track_knot(knots, time_index, "y_q") / 256.0
        center_x = _interpolate_track_knot(knots, time_index, "x_q") / 256.0
        height = max(1, round(_interpolate_track_knot(knots, time_index, "height_q") / 256.0))
        width = max(1, round(_interpolate_track_knot(knots, time_index, "width_q") / 256.0))
        y0 = max(0, round(center_y - height / 2))
        y1 = min(out.shape[0], y0 + height)
        x0 = max(0, round(center_x - width / 2))
        x1 = min(out.shape[1], x0 + width)
        out[y0:y1, x0:x1] = track["cell_id"]
    return out


def _component_compressed(seed: Mapping[str, Any]) -> dict[str, int]:
    raw = component_byte_accounting(seed)
    values = {
        "chart": seed["ground_chart"],
        "trajectory": seed["trajectory"],
        "tracks": seed["movable_tracks"],
        "events": seed["events"],
        "constraints": seed["constraint_seeds"],
        "jitter": seed["boundary_jitter"],
    }
    return {**{f"raw_{key}": int(value) for key, value in raw.items()}, **{f"zlib9_{key}": len(zlib.compress(canonical_json_bytes(value), 9)) for key, value in values.items()}}


def _factorization(constraints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def rows_for(key: str, value: Any) -> list[Mapping[str, Any]]:
        return [row for row in constraints if row[key] == value]

    per_class = []
    for class_id in range(5):
        rows = rows_for("cell_id", class_id)
        per_class.append(
            {"class_id": class_id, "constraint_count": len(rows), "standalone_zlib9_bytes": len(zlib.compress(canonical_json_bytes(rows), 9))}
        )
    per_stratum = []
    for stratum in ("cell_interior", "boundary_codim1", "movable_track", "critical_event"):
        rows = rows_for("stratum", stratum)
        per_stratum.append(
            {"stratum": stratum, "constraint_count": len(rows), "standalone_zlib9_bytes": len(zlib.compress(canonical_json_bytes(rows), 9))}
        )
    return {"per_class": per_class, "per_stratum": per_stratum}


def compose_seed_curve(
    *,
    repository_root: Path,
    gt_cache: Path,
    s2_packet: Path,
    inventory_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Compose three nested real n600 seed points and their advisory KKT row."""

    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    _require_sha(gt_cache, GT_CACHE_SHA256, "frozen GT cache")
    _require_sha(s2_packet, S2_PACKET_SHA256, "S2 event packet")
    packet = decode_partition_seed(s2_packet.read_bytes())
    if (packet.n_pairs, packet.height, packet.width) != (PAIR_COUNT, SCORER_HEIGHT, SCORER_WIDTH):
        raise SeedComposeError("S2 packet geometry is not exact n600 scorer geometry")
    margins, inventory_custody = _load_margin_map(inventory_dir)
    with np.load(gt_cache, allow_pickle=False) as cache:
        labels = cache["lstars"]
        gt_poses = np.asarray(cache["gt_poses"], dtype=np.float64)
        mode, occupancy_centroids_q = _temporal_mode_and_centroids(labels)
        chart = compatibility_site_chart(mode, occupancy_centroids_q=occupancy_centroids_q)
        tracks, track_custody = _movable_tracks(labels)
        pose_q = np.rint(gt_poses * 1_048_576).astype(np.int64)
        pose_steps = np.abs(np.diff(pose_q, axis=0)).reshape(-1)
        radii = (
            math.ceil(float(np.quantile(pose_steps, 0.9))),
            math.ceil(float(np.quantile(pose_steps, 0.5))),
            0,
        )
        s_t, s_r, lawref_custody = _lawref_g1_calibration(repository_root)
        chart_raster = np.frombuffer(derive_morse_smale_raster(chart), dtype=np.uint8).reshape(
            SCORER_HEIGHT, SCORER_WIDTH
        )
        events_by_pair: dict[int, list[PartitionEvent]] = defaultdict(list)
        for event in packet.events:
            events_by_pair[event.pair].append(event)
        contexts: list[dict[str, Any]] = []
        for curve_index, name in enumerate(CURVE_NAMES):
            trajectory, trajectory_custody = _trajectory(gt_poses, curve_index, s_t=s_t, s_r=s_r)
            base = _seed_base(chart, tracks, trajectory)
            event_rows: dict[tuple[int, int, int], dict[str, Any]] = {}
            whole_correct = Counter()
            whole_total = Counter()
            predictor_sha = hashlib.sha256()
            for pair in range(PAIR_COUNT):
                predicted = _predict_prevalidated(base, chart_raster, pair)
                target_plane = np.asarray(labels[pair], dtype=np.uint8)
                predictor_sha.update(predicted.tobytes())
                for class_id in range(5):
                    mask = target_plane == class_id
                    whole_total[class_id] += int(np.count_nonzero(mask))
                    whole_correct[class_id] += int(np.count_nonzero(predicted[mask] == class_id))
                for event in events_by_pair[pair]:
                    target = int(target_plane[event.row, event.col])
                    margin = margins.get((event.pair, event.row, event.col), float("inf"))
                    if int(predicted[event.row, event.col]) != target:
                        key = (event.pair, event.row, event.col)
                        event_rows[key] = {
                            "event": event,
                            "target": target,
                            "margin": margin,
                            "stratum": _stratum(event, target, margin),
                            "rank": _event_rank(event, target, margin),
                        }
            contexts.append(
                {
                    "name": name,
                    "base": base,
                    "trajectory_custody": trajectory_custody,
                    "event_rows": event_rows,
                    "whole_correct": whole_correct,
                    "whole_total": whole_total,
                    "predictor_sha256": predictor_sha.hexdigest(),
                }
            )
        common_sites = set(contexts[0]["event_rows"])
        for context in contexts[1:]:
            common_sites.intersection_update(context["event_rows"])
        ordered = _ordered_with_pair_coverage([contexts[0]["event_rows"][site] for site in common_sites])
        points: list[dict[str, Any]] = []
        seed_paths: dict[str, str] = {}
        previous_sites: set[tuple[int, int, int]] = set()
        for curve_index, context in enumerate(contexts):
            name = context["name"]
            base = context["base"]
            whole_correct = context["whole_correct"]
            whole_total = context["whole_total"]
            fractions = (0.25, 0.5, 1.0)
            requested = max(PAIR_COUNT, math.ceil(len(ordered) * fractions[curve_index]))
            selected = ordered[:requested]
            selected_sites = {(row["event"].pair, row["event"].row, row["event"].col) for row in selected}
            if curve_index and not previous_sites.issubset(selected_sites):
                raise SeedComposeError("constraint curve is not nested")
            previous_sites = selected_sites
            constraints = _constraints(selected, pose_q, radii[curve_index])
            seed = copy.deepcopy(base)
            seed["constraint_seeds"] = constraints
            seed["boundary_jitter"]["equal_fidelity_custody"]["represented_constraints_sha256"] = hashlib.sha256(
                canonical_json_bytes(constraints)
            ).hexdigest()
            seed = validate_constraint_seed(seed)
            seed_bytes = serialize_constraint_seed(seed)
            seed_path = output_root / "seeds" / f"seed_compose_b2_{name}.ppcs"
            _atomic_write(seed_path, seed_bytes)
            seed_paths[name] = str(seed_path)
            d_seg = (sum(whole_total.values()) - sum(whole_correct.values()) - len(constraints)) / sum(whole_total.values())
            d_seg = max(0.0, float(d_seg))
            rate_term = 25.0 * len(seed_bytes) / RATE_DENOMINATOR_BYTES
            proxy_score = 100.0 * d_seg + rate_term
            point = {
                "name": name,
                "seed_path": str(seed_path),
                "seed_sha256": hashlib.sha256(seed_bytes).hexdigest(),
                "seed_bytes": len(seed_bytes),
                "seed_zlib9_bytes": len(zlib.compress(seed_bytes, 9)),
                "selected_constraint_count": len(constraints),
                "eligible_violated_s2_event_count": len(context["event_rows"]),
                "common_nested_eligible_event_count": len(ordered),
                "pose_tube_radius_q": radii[curve_index],
                "trajectory": context["trajectory_custody"],
                "predictor_n600_sha256": context["predictor_sha256"],
                "description_metrics": {
                    "d_seg_represented_vs_frozen_cpu_torch_lstar": d_seg,
                    "d_pose_banked_target_outside_declared_tube_mse": 0.0,
                    "rate_term_using_seed_bytes_not_archive_bytes": rate_term,
                    "advisory_description_objective": proxy_score,
                    "score_claim": False,
                },
                "D3_generic_predictor": {
                    "selected_site_satisfaction_before_projection": 0.0,
                    "selected_site_satisfaction_after_projection": 1.0,
                    "whole_field_accuracy_by_target_class": [
                        {
                            "class_id": class_id,
                            "correct": whole_correct[class_id],
                            "total": whole_total[class_id],
                            "fraction": whole_correct[class_id] / whole_total[class_id],
                        }
                        for class_id in range(5)
                    ],
                    "low_satisfaction_causes": [
                        "five-site Voronoi compatibility raster is not the unresolved native Morse-Smale arc-to-cell rasterizer",
                        "PoseNet-derived xi is calibrated on n200 G1 and carries no per-pixel partition residual",
                        "movable boxes approximate connected components and cannot express non-box topology",
                    ],
                    "native_rasterizer_blocker": MS_NATIVE_BLOCKER,
                },
                "D4_factorization": _factorization(constraints),
                "component_bytes": _component_compressed(seed),
                "axis": "[macOS-CPU advisory]",
                "contest_authority": False,
                "promotion_eligible": False,
            }
            points.append(point)
            _atomic_json(output_root / "checkpoints" / f"stage_{curve_index + 1:02d}_{name}.json", point)

    objectives = [float(point["description_metrics"]["advisory_description_objective"]) for point in points]
    knee_index = int(np.argmin(objectives))
    receipt = {
        "schema": "seed_compose_b2_curve.v1",
        "verdict": "REAL_N600_CONSTRAINT_SEED_CURVE_COMPOSED_RECEIVER_RGB_CLOSURE_BLOCKED",
        "verdict_scope": (
            "real frozen CPU-Torch target-cache cell descriptions and PoseNet tubes, compatibility site raster, "
            "single schema object; not receiver-closed RGB, not contest score"
        ),
        "D1_curve": points,
        "D2_hard_oracle": {
            "status": "READY_FOR_N16_N64_N600_CACHE_REPLAY",
            "adapter": "tac.optimization.seed_compose_b2:hard_oracle_cache_replay",
            "selected_seed": seed_paths[points[knee_index]["name"]],
            "required_prefixes": [16, 64, 600],
            "uint8_factor2_exact_expected": False,
            "blocker": "single-object cell description has no receiver-closed camera-RGB realization",
        },
        "D3_predictor": [point["D3_generic_predictor"] for point in points],
        "D4_factorization": [
            {"name": point["name"], "factorization": point["D4_factorization"], "component_bytes": point["component_bytes"]}
            for point in points
        ],
        "D5_advisory": {
            "advisory_score": objectives[knee_index],
            "selected_point": points[knee_index]["name"],
            "lambda_or_kkt_knee": {
                "criterion": "minimum 100*d_seg_description + 25*seed_bytes/37545489 over preregistered nested points",
                "knee_index": knee_index,
                "looser_neighbor": points[knee_index - 1]["name"] if knee_index > 0 else None,
                "tighter_neighbor": points[knee_index + 1]["name"] if knee_index + 1 < len(points) else None,
            },
            "measured_full_pipeline_score": None,
            "receiver_realizability_gap": "no camera-RGB inverse-R realization bound to the single object",
            "archive_realizability_gap": "PPCS bytes are counted description bytes, not a compliant archive.zip",
            "pointer": "UNCHANGED",
            "score_claim": False,
        },
        "input_custody": {
            "gt_cache": {"path": str(gt_cache), "sha256": GT_CACHE_SHA256, "bytes": gt_cache.stat().st_size},
            "s2_packet": {"path": str(s2_packet), "sha256": S2_PACKET_SHA256, "bytes": s2_packet.stat().st_size},
            "inventory": inventory_custody,
            "g1_lawrefs": lawref_custody,
        },
        "reuse_manifest": {
            "schema_builder_validator_serializer_parser": "tac.optimization.predict_project_schema",
            "receiver_chart_pose_component_plane_contract": "tac.optimization.predict_project_receiver",
            "s2_finite_event_packet": "tac.optimization.s2_partition_seed",
            "pose_to_xi": "tac.boundary_math.ego_xi_trajectory.PoseTargetEgoEstimator",
            "movable_correspondence": "tac.boundary_math.movable_site_coder.extract_movable_sites+track_sites",
            "lawrefs": "tac.witness_dsl.lawref",
            "tracks": track_custody,
        },
        "blockers": [MS_NATIVE_BLOCKER, "SINGLE_OBJECT_TO_CAMERA_RGB_REALIZATION_UNMEASURED"],
        "automatic_disk_hygiene": {
            "scratch_policy": "in-memory per-rung predictor planes; no bulk local scratch created",
            "durable_output_tier": str(output_root),
        },
        "measured_seconds": time.perf_counter() - started,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
    }
    _atomic_json(output_root / "receipt.json", receipt)
    return receipt


_ORACLE_CACHE: dict[str, Any] = {}


def _oracle_paths() -> tuple[Path, Path, Path, Path]:
    repository_root = Path(__file__).resolve().parents[3]
    gt_cache = Path(
        os.environ.get(
            "SEED_COMPOSE_GT_CACHE",
            "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        )
    )
    upstream = repository_root / "upstream"
    if not upstream.is_dir():
        upstream = Path("/Users/adpena/Projects/pact/upstream")
    return gt_cache, upstream / "modules.py", upstream / "models/segnet.safetensors", upstream / "models/posenet.safetensors"


def _oracle_state() -> dict[str, Any]:
    if _ORACLE_CACHE:
        return _ORACLE_CACHE
    gt_cache, scorer_source, seg_weights, pose_weights = _oracle_paths()
    _require_sha(gt_cache, GT_CACHE_SHA256, "hard-oracle GT cache")
    for path in (scorer_source, seg_weights, pose_weights):
        if not path.is_file():
            raise SeedComposeError(f"hard-oracle custody file missing: {path}")
    archive = np.load(gt_cache, allow_pickle=False)
    _ORACLE_CACHE.update(
        {
            "archive": archive,
            "lstars": archive["lstars"],
            "gt_poses": archive["gt_poses"],
            "scorer_source_sha256": sha256_file(scorer_source),
            "segnet_weights_sha256": sha256_file(seg_weights),
            "posenet_weights_sha256": sha256_file(pose_weights),
        }
    )
    return _ORACLE_CACHE


def hard_oracle_cache_replay(**kwargs: Any) -> dict[str, Any]:
    """Replay the frozen CPU-Torch n600 target cache against represented cells.

    This measures description-space cell/tube obligations only.  It explicitly
    returns ``uint8_factor2_exact=False`` until a camera-RGB realization exists.
    """

    started = time.perf_counter()
    state = _oracle_state()
    pair = int(kwargs["pair_index"])
    seed = kwargs["seed"]
    predicted = np.asarray(kwargs["predicted"], dtype=np.uint8)
    represented = np.asarray(kwargs["represented"], dtype=np.uint8)
    desired = np.asarray(state["lstars"][pair], dtype=np.uint8)
    if predicted.shape != desired.shape or represented.shape != desired.shape:
        raise SeedComposeError("hard-oracle predictor/represented geometry drifted from frozen targets")
    projection_seconds = time.perf_counter() - started
    constraints = [row for row in seed["constraint_seeds"] if row["time"] == pair and row["frame_index"] == 1]
    cell_exact = all(int(represented[row["y"], row["x"]]) == int(desired[row["y"], row["x"]]) for row in constraints)
    pose_q = np.rint(np.asarray(state["gt_poses"][pair], dtype=np.float64) * 1_048_576).astype(np.int64)
    tubes = [row["pose_tube"] for row in constraints if row["pose_tube"] is not None]
    if not tubes:
        raise SeedComposeError(f"pair {pair} has no custodied pose tube")
    outside = []
    for tube in tubes:
        lower = np.asarray(tube["lower_q"], dtype=np.int64)
        upper = np.asarray(tube["upper_q"], dtype=np.int64)
        outside.append(np.maximum(lower - pose_q, 0) + np.maximum(pose_q - upper, 0))
    best_outside = min(outside, key=lambda value: float(np.sum(value.astype(np.float64) ** 2)))
    realization_seconds = time.perf_counter() - started - projection_seconds
    d_seg = float(np.mean(represented != desired))
    d_pose = float(np.mean((best_outside.astype(np.float64) / 1_048_576) ** 2))
    verification_seconds = time.perf_counter() - started - projection_seconds - realization_seconds
    seed_bytes = serialize_constraint_seed(seed)
    adapter_source = Path(inspect.getsourcefile(hard_oracle_cache_replay) or __file__).resolve()
    return {
        "schema": "predict_project_hard_oracle_pair.v0",
        "pair_index": pair,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "cell_exact": bool(cell_exact),
        "pose_within_tube": bool(np.all(best_outside == 0)),
        "uint8_factor2_exact": False,
        "stage_seconds": {
            "projection": projection_seconds,
            "realization": realization_seconds,
            "verification": verification_seconds,
        },
        "custody": {
            "schema": "predict_project_hard_oracle_custody.v0",
            "seed": int(kwargs["measurement_seed"]),
            "batch_size": int(kwargs["batch_size"]),
            "measurement_axis": "[macOS-CPU advisory]",
            "scorer": {
                "implementation_id": "upstream.modules.DistortionNet.frozen_cpu_torch_cache_replay",
                "version": "gt_n600_cf8d83605d21",
                "source_sha256": state["scorer_source_sha256"],
                "segnet_weights_sha256": state["segnet_weights_sha256"],
                "posenet_weights_sha256": state["posenet_weights_sha256"],
            },
            "inputs": {
                "source_sha256": hashlib.sha256(seed_bytes).hexdigest(),
                "cache_sha256": GT_CACHE_SHA256,
                "evaluated_input_sha256": GT_CACHE_SHA256,
            },
            "adapter": {
                "identity": f"{hard_oracle_cache_replay.__module__}:{hard_oracle_cache_replay.__qualname__}",
                "source_sha256": sha256_file(adapter_source),
            },
        },
        "desired_cells": desired,
    }


__all__ = [
    "CURVE_NAMES",
    "GT_CACHE_SHA256",
    "MS_NATIVE_BLOCKER",
    "S2_PACKET_SHA256",
    "SeedComposeError",
    "compatibility_site_chart",
    "compose_seed_curve",
    "hard_oracle_cache_replay",
    "sha256_file",
    "temporal_mode_chart",
]
