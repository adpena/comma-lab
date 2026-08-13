#!/usr/bin/env python3
"""Build the EC3 T4-targeted, joint-net-S-priced singleton event store.

EC3 never runs a scorer.  It consumes the retained contest-CUDA T4 CP135 and
GT argmax fields, scans all 600 pairs, and emits one minimal one-token EC1-wire
event for each of 200 coverage-selected pairs.  The selected pairs include the
current T4 heavy tail and the older G3 hard-pair/control atlas.  A balanced
assignment carries Undrivable->Road, MyCar->Road, Road->Lane, and Lane->Road.

The actual CP135 semantic receiver is replayed for every event.  Local ordering
uses only retained receiver deltas, the measured 13 percent advisory-to-exact
precision, the JS4 Jacobian sensitivity receipts, and the retained VD1 exact
T4 singleton census.  The score proxy is exactly the chartered local marginal

    100 * delta_d_seg + 603 * delta_d_pose.

Every materialized receiver payload is retained below ``--event-store``.  The
unchanged VD1 bundle builder is exercised locally, but no Modal function,
SegNet, PoseNet, MPS, or evaluator is invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_ec1_event_coordinate_producer as ec1
from experiments import ddm_js1_stage0_per_edge as js1

RUN_ROOT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_ec3_20260813")
EVENT_STORE: Final = RUN_ROOT / "event_store_target_anchored"
T4_FIELD_ROOT: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "contest_cuda/ddm_js1b_20260813b/retained/fields"
)
T4_BASE_ARGMAX: Final = T4_FIELD_ROOT / "cp135_base_argmax_n600.npy"
T4_GT_ARGMAX: Final = T4_FIELD_ROOT / "gt_argmax_n600.npy"
JS1B_FINAL: Final = T4_FIELD_ROOT.parents[1] / "FINAL_RESULT.json"
JS4_MANIFEST: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_js4_20260812/projector/MANIFEST.json"
)
VD1_RESULTS: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_vd1_20260812/main_harvest/results/"
    "EVENT_RESULTS.jsonl"
)
EC1_STORE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/"
    "follow_on/realized_acceptance_200"
)
EC1_INDEX: Final = EC1_STORE / "proposal_index.jsonl"
EC1_BASE_SCORER_ROOT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_ec1_20260812/retained/receiver_proof/inactive"
)
G3_REGISTRY: Final = (
    REPO
    / ".omx/research/ddm_g3_score_atlas_n600_20260722T204000Z/"
    "hard_pair_registry.json"
)
JO1_ANALYSIS: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_jo1_20260812/10_ANALYSIS.json"
)

N: Final = 600
H: Final = 384
W: Final = 512
CAM_H: Final = 874
CAM_W: Final = 1164
PIXELS: Final = N * H * W
STORE_EVENTS: Final = 200
PER_DIRECTION: Final = 50
TARGET_WINDOW: Final = 25
STRONG_DELTA_LSB: Final = 1.0
EXACT_TRANSFER_PRECISION: Final = 0.13
POSE_MARGINAL: Final = 603.0
FIRE_BAR_S: Final = 0.000216
MIN_FREE_BYTES: Final = 8 * 1024**3
AXIS: Final = (
    "[contest-CUDA T4 retained argmax targeting + macOS-CPU scorer-free receiver "
    "prescreen; full-n600 field scan]"
)

PINNED: Final = {
    T4_BASE_ARGMAX: "7648ad42e9f21942f86e81b97cabf46b710af747bba0909f7837ef3891232727",
    T4_GT_ARGMAX: "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248",
    JS1B_FINAL: "5fd65b946e2e1a5683e123554761c4216f8245a4d1cec46da2ee95b925c93a0c",
    JS4_MANIFEST: "75655e17cb40d07b1747c2708f986ab7a1de09fcc4c22f4f5451316c13f10fe2",
    VD1_RESULTS: "a97400d32878318d8eb657a36e62f523e4db48e402b292c09e611d2104b500b3",
    EC1_INDEX: "599a3ac0a9c7d7e62c162fcee595194d6d3cd79685d0ceabab92e0231bd9d47e",
    G3_REGISTRY: "0c9ce6d0ce2b2c0830400f096438355242527d40f682fc1b201f67d8d951a4e4",
    JO1_ANALYSIS: "440542cfc34e7bbd8ff2e2a1fd71e4f62aea1f929f1229c4aae5c68e29323a3c",
    ec1.BASE_TOKENS: ec1.BASE_TOKEN_SHA,
    ec1.BASE_RAW: ec1.BASE_RAW_SHA,
    ec1.CP135_ARCHIVE: ec1.CP135_ARCHIVE_SHA,
}

# Directed event source->target classes.  The first two are separate edge
# families; the last two jointly constitute Road<->Lane.
DIRECTIONS: Final = ((2, 0), (4, 0), (0, 1), (1, 0))
DIRECTION_NAMES: Final = {
    (2, 0): "Undrivable->Road",
    (4, 0): "MyCar->Road",
    (0, 1): "Road->Lane",
    (1, 0): "Lane->Road",
}


class EC3Error(RuntimeError):
    """A custody, receiver, retention, selection, or pricing invariant failed."""


@dataclass(frozen=True, slots=True)
class StaticOption:
    pair: int
    source_class: int
    target_class: int
    token_index: int
    local_target_errors: int
    pose_sensitivity: float
    static_priority: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with partial.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return file_record(path)


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return atomic_bytes(path, payload)


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with partial.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return file_record(path)


def require_record(record: dict[str, Any], *, beneath: Path | None = None) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if beneath is not None:
        try:
            path.relative_to(beneath.resolve())
        except ValueError as exc:
            raise EC3Error(f"retained artifact escapes {beneath}: {path}") from exc
    if not path.is_file() or file_record(path) != record:
        raise EC3Error(f"retained artifact differs: {path}")
    return path


def preflight(run_root: Path, event_store: Path) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    event_store.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(run_root)
    if usage.free < MIN_FREE_BYTES:
        raise EC3Error(f"storage preflight failed: {usage.free} < {MIN_FREE_BYTES}")
    inputs: dict[str, Any] = {}
    for path, digest in PINNED.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise EC3Error(f"pinned input differs: {path}")
        inputs[str(path)] = file_record(path)
    for path in (T4_BASE_ARGMAX, T4_GT_ARGMAX):
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.shape != (N, H, W) or value.dtype != np.uint8:
            raise EC3Error(f"T4 field geometry differs: {path}")
    result = {
        "schema": "ddm_ec3_preflight.v1",
        "axis": AXIS,
        "score_claim": False,
        "modal": False,
        "scorer": False,
        "mps": False,
        "storage": {"free_bytes": usage.free, "required_bytes": MIN_FREE_BYTES},
        "run_root": str(run_root.resolve()),
        "event_store": str(event_store.resolve()),
        "inputs": inputs,
    }
    atomic_json(run_root / "00_PREFLIGHT.json", result)
    return result


def build_pose_sensitivity(run_root: Path) -> tuple[np.ndarray, dict[str, Any]]:
    output = run_root / "retained/pose_prior/js4_site_sensitivity.float32.npy"
    receipt_path = run_root / "10_POSE_PRIOR.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        path = require_record(receipt["site_sensitivity"], beneath=run_root)
        return np.load(path, mmap_mode="r", allow_pickle=False), receipt
    manifest = json.loads(JS4_MANIFEST.read_text())
    if manifest.get("schema") != "ddm_js4_pose_projector_cache.v1" or len(manifest["pairs"]) != 32:
        raise EC3Error("JS4 projector manifest schema differs")
    squared = np.zeros((H, W), dtype=np.float64)
    rows = []
    for row in manifest["pairs"]:
        path = require_record(row["jacobian"])
        jacobian = np.load(path, mmap_mode="r", allow_pickle=False)
        if jacobian.shape != (6, 3 * H * W) or jacobian.dtype != np.float32:
            raise EC3Error(f"JS4 Jacobian geometry differs: {path}")
        shaped = np.asarray(jacobian).reshape(6, 3, H, W)
        squared += np.sum(np.square(shaped, dtype=np.float64), axis=(0, 1))
        rows.append({"pair": int(row["pair_id"]), "jacobian": row["jacobian"]})
    sensitivity = np.sqrt(squared / len(rows)).astype(np.float32)
    sensitivity_record = atomic_npy(output, sensitivity)
    receipt = {
        "schema": "ddm_ec3_pose_site_sensitivity.v1",
        "axis": "[retained JS4 macOS-CPU Jacobian read; scorer-free aggregation]",
        "score_claim": False,
        "aggregation": "sqrt(mean over 32 pairs of sum over 6 outputs x 3 channels of J^2)",
        "pairs": rows,
        "site_sensitivity": sensitivity_record,
        "quantiles": {
            str(q): float(np.quantile(sensitivity, q)) for q in (0.0, 0.05, 0.5, 0.95, 1.0)
        },
    }
    atomic_json(receipt_path, receipt)
    return sensitivity, receipt


def receiver_delta_pose_proxy(jacobian: np.ndarray, scorer_delta: np.ndarray) -> float:
    if jacobian.shape != (6, 3 * H * W) or scorer_delta.shape != (3, H, W):
        raise EC3Error("pose-proxy geometry differs")
    shift = np.asarray(jacobian, dtype=np.float32) @ np.asarray(scorer_delta, dtype=np.float32).reshape(-1)
    return float(np.mean(np.square(shift.astype(np.float64))) / N)


def build_calibration(
    run_root: Path,
    sensitivity: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt_path = run_root / "20_VD1_JS4_CALIBRATION.json"
    rows_path = run_root / "retained/calibration/vd1_js4_rows.jsonl"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        path = require_record(receipt["rows"], beneath=run_root)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return rows, receipt
    exact_rows = [
        json.loads(line) for line in VD1_RESULTS.read_text().splitlines() if line.strip()
    ]
    index_rows = [json.loads(line) for line in EC1_INDEX.read_text().splitlines() if line.strip()]
    by_id = {str(row["proposal_id"]): row for row in index_rows}
    manifest = json.loads(JS4_MANIFEST.read_text())
    jacobian_by_pair = {
        int(row["pair_id"]): require_record(row["jacobian"]) for row in manifest["pairs"]
    }
    if len(exact_rows) != STORE_EVENTS or len(by_id) != STORE_EVENTS:
        raise EC3Error("VD1/EC1 calibration census is not 200 unique rows")
    output = []
    for exact in exact_rows:
        proposal_id = str(exact["proposal_id"])
        source = by_id.get(proposal_id)
        if source is None:
            raise EC3Error(f"VD1 event absent from EC1 store: {proposal_id}")
        payload_path = require_record(source["consumer_payloads"]["event.ec1p"], beneath=EC1_STORE)
        pair, source_class, target_class, _event_type, indices = ec1.decode_proposal(
            payload_path.read_bytes()
        )
        candidate_scorer_path = require_record(
            source["consumer_payloads"]["scorer_input.float16.npy"], beneath=EC1_STORE
        )
        base_scorer_path = EC1_BASE_SCORER_ROOT / f"pair_{pair:03d}.scorer_input.float16.npy"
        if not base_scorer_path.is_file() or pair not in jacobian_by_pair:
            raise EC3Error(f"calibration custody missing for pair {pair}")
        candidate_scorer = np.load(candidate_scorer_path, mmap_mode="r", allow_pickle=False)
        base_scorer = np.load(base_scorer_path, mmap_mode="r", allow_pickle=False)
        delta = np.asarray(candidate_scorer, dtype=np.float32) - np.asarray(
            base_scorer, dtype=np.float32
        )
        jacobian = np.load(jacobian_by_pair[pair], mmap_mode="r", allow_pickle=False)
        proxy = receiver_delta_pose_proxy(jacobian, delta)
        site_values = np.asarray(sensitivity).reshape(-1)[indices]
        output.append(
            {
                "proposal_id": proposal_id,
                "pair": pair,
                "source_class": source_class,
                "target_class": target_class,
                "site_count": len(indices),
                "seed_y": int(indices[0] // W),
                "seed_x": int(indices[0] % W),
                "js4_receiver_delta_proxy": proxy,
                "js4_site_sensitivity_mean": float(np.mean(site_values)),
                "exact_delta_d_pose_global_n600": float(exact["delta_d_pose_global_n600"]),
                "nonnegative_exact_pose_cost": max(
                    0.0, float(exact["delta_d_pose_global_n600"])
                ),
                "exact_net_flip_gain": int(exact["net_flip_gain_base_minus_candidate"]),
                "axis": exact["axis"],
            }
        )
    rows_record = atomic_bytes(
        rows_path,
        b"".join(
            (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
            for row in output
        ),
    )
    positive = [row for row in output if row["exact_net_flip_gain"] > 0]
    pose_costs = np.asarray([row["nonnegative_exact_pose_cost"] for row in output])
    receipt = {
        "schema": "ddm_ec3_vd1_js4_calibration.v1",
        "axis": "[retained VD1 contest-CUDA T4 exact rows x retained JS4 Jacobians; scorer-free join]",
        "score_claim": False,
        "rows": rows_record,
        "events": len(output),
        "pairs": sorted({int(row["pair"]) for row in output}),
        "net_flip_positive_events": len(positive),
        "pose_cost_quantiles": {
            str(q): float(np.quantile(pose_costs, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0)
        },
        "prediction_rule": (
            "16-nearest calibration events in log(JS4 receiver-delta proxy), "
            "log(JS4 site sensitivity), normalized site coordinate, and direction; "
            "the optimistic prediction is the neighbor q25 of nonnegative exact T4 pose cost"
        ),
    }
    atomic_json(receipt_path, receipt)
    return output, receipt


def _robust_scale(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, max(mad, 1e-12)


def predict_pose_cost(
    calibration: list[dict[str, Any]],
    *,
    proxy: float,
    sensitivity: float,
    y: int,
    x: int,
    source_class: int,
    target_class: int,
    neighbors: int = 16,
) -> dict[str, Any]:
    if len(calibration) < neighbors:
        raise EC3Error("pose calibration has too few rows")
    proxy_values = np.log10(
        np.asarray([float(row["js4_receiver_delta_proxy"]) for row in calibration]) + 1e-20
    )
    sensitivity_values = np.log10(
        np.asarray([float(row["js4_site_sensitivity_mean"]) for row in calibration]) + 1e-20
    )
    proxy_median, proxy_scale = _robust_scale(proxy_values)
    sensitivity_median, sensitivity_scale = _robust_scale(sensitivity_values)
    query_proxy = (math.log10(max(proxy, 0.0) + 1e-20) - proxy_median) / proxy_scale
    query_sensitivity = (
        math.log10(max(sensitivity, 0.0) + 1e-20) - sensitivity_median
    ) / sensitivity_scale
    distances = []
    for index, row in enumerate(calibration):
        row_proxy = (proxy_values[index] - proxy_median) / proxy_scale
        row_sensitivity = (sensitivity_values[index] - sensitivity_median) / sensitivity_scale
        spatial = math.hypot((y - int(row["seed_y"])) / H, (x - int(row["seed_x"])) / W)
        direction_penalty = 0.0
        if (
            int(row["source_class"]) != source_class
            or int(row["target_class"]) != target_class
        ):
            direction_penalty = 1.0
        distance = (
            (query_proxy - row_proxy) ** 2
            + 0.5 * (query_sensitivity - row_sensitivity) ** 2
            + 0.5 * spatial**2
            + direction_penalty
        )
        distances.append((distance, str(row["proposal_id"]), row))
    nearest = [item[2] for item in sorted(distances, key=lambda item: (item[0], item[1]))[:neighbors]]
    costs = np.asarray([float(row["nonnegative_exact_pose_cost"]) for row in nearest])
    return {
        "predicted_delta_d_pose_global_n600": float(np.quantile(costs, 0.25)),
        "neighbor_pose_q50": float(np.quantile(costs, 0.5)),
        "neighbor_pose_q90": float(np.quantile(costs, 0.9)),
        "neighbors": [str(row["proposal_id"]) for row in nearest],
        "neighbor_count": neighbors,
        "scope": "optimistic q25 local ordering from retained T4 singleton exact costs; not a bound",
    }


def normalized_sensitivity(sensitivity: np.ndarray) -> np.ndarray:
    log_value = np.log10(np.asarray(sensitivity, dtype=np.float64) + 1e-30)
    low, high = np.quantile(log_value, (0.05, 0.95))
    if not high > low:
        raise EC3Error("JS4 site sensitivity has no usable dynamic range")
    return np.clip((log_value - low) / (high - low), 0.0, 1.0)


def select_token_site(
    tokens: np.ndarray,
    base_argmax: np.ndarray,
    gt_argmax: np.ndarray,
    sensitivity: np.ndarray,
    source_class: int,
    target_class: int,
    *,
    pair: int,
    sensitivity_norm: np.ndarray | None = None,
) -> StaticOption | None:
    errors = (base_argmax == source_class) & (gt_argmax == target_class)
    source_sites = tokens == source_class
    if not np.any(errors) or not np.any(source_sites):
        return None
    density = ndimage.uniform_filter(
        errors.astype(np.float32), size=TARGET_WINDOW, mode="constant"
    ) * float(TARGET_WINDOW**2)
    pose_norm = (
        normalized_sensitivity(sensitivity)
        if sensitivity_norm is None
        else sensitivity_norm
    )
    if pose_norm.shape != (H, W):
        raise EC3Error("normalized JS4 sensitivity geometry differs")
    priority = density / (1.0 + 3.0 * pose_norm)
    priority = np.where(source_sites, priority, -np.inf)
    best = float(np.max(priority))
    if not math.isfinite(best) or best <= 0:
        return None
    index = int(np.flatnonzero(priority.reshape(-1) == best)[0])
    y, x = divmod(index, W)
    radius = TARGET_WINDOW // 2
    y0, y1 = max(0, y - radius), min(H, y + radius + 1)
    x0, x1 = max(0, x - radius), min(W, x + radius + 1)
    local_errors = int(np.count_nonzero(errors[y0:y1, x0:x1]))
    if local_errors <= 0:
        raise EC3Error("selected token site has no target error in its registered window")
    return StaticOption(
        pair=pair,
        source_class=source_class,
        target_class=target_class,
        token_index=index,
        local_target_errors=local_errors,
        pose_sensitivity=float(sensitivity[y, x]),
        static_priority=best,
    )


def choose_pair_set(
    pair_error_counts: np.ndarray,
    g3_top64: list[int],
    g3_control24: list[int],
    direction_available: np.ndarray,
) -> tuple[list[int], dict[str, Any]]:
    if pair_error_counts.shape != (N,) or direction_available.shape != (N, len(DIRECTIONS)):
        raise EC3Error("pair-selection geometry differs")
    current_order = [int(value) for value in np.argsort(-pair_error_counts, kind="stable")]
    current_top64 = current_order[:64]
    selected: set[int] = set(current_top64) | set(g3_top64) | set(g3_control24)
    required = set(selected)
    for direction_index in range(len(DIRECTIONS)):
        available = sum(bool(direction_available[pair, direction_index]) for pair in selected)
        if available < PER_DIRECTION:
            for pair in current_order:
                if direction_available[pair, direction_index]:
                    selected.add(pair)
                available = sum(
                    bool(direction_available[value, direction_index]) for value in selected
                )
                if available >= PER_DIRECTION:
                    break
        if available < PER_DIRECTION:
            raise EC3Error(f"only {available} selected pairs support direction {direction_index}")
    if len(selected) > STORE_EVENTS:
        removable = [pair for pair in reversed(current_order) if pair in selected and pair not in required]
        while len(selected) > STORE_EVENTS and removable:
            candidate = removable.pop(0)
            trial = selected - {candidate}
            if all(
                sum(bool(direction_available[pair, column]) for pair in trial) >= PER_DIRECTION
                for column in range(len(DIRECTIONS))
            ):
                selected = trial
    for pair in current_order:
        if len(selected) >= STORE_EVENTS:
            break
        selected.add(pair)
    if len(selected) != STORE_EVENTS:
        raise EC3Error(f"pair selection has {len(selected)} rows, expected {STORE_EVENTS}")
    ordered = sorted(selected, key=lambda pair: (-int(pair_error_counts[pair]), pair))
    result = {
        "selection_mode": "full-n600 scan; union current-T4-top64 + G3-top64 + G3-control24, then T4 fill",
        "pairs": ordered,
        "pair_count": len(ordered),
        "current_t4_top64_covered": sum(pair in selected for pair in current_top64),
        "g3_top64_covered": sum(pair in selected for pair in g3_top64),
        "g3_control24_covered": sum(pair in selected for pair in g3_control24),
        "selected_t4_error_mass": int(sum(pair_error_counts[pair] for pair in selected)),
        "population_t4_error_mass": int(np.sum(pair_error_counts)),
    }
    result["selected_t4_error_mass_share"] = (
        result["selected_t4_error_mass"] / result["population_t4_error_mass"]
    )
    return ordered, result


def assign_directions(
    pairs: list[int],
    options: dict[tuple[int, int], StaticOption],
) -> list[StaticOption]:
    if len(pairs) != STORE_EVENTS:
        raise EC3Error("direction assignment requires exactly 200 pairs")
    slots = [direction for direction in range(len(DIRECTIONS)) for _ in range(PER_DIRECTION)]
    cost = np.full((len(pairs), len(slots)), 1e12, dtype=np.float64)
    for row, pair in enumerate(pairs):
        for column, direction in enumerate(slots):
            option = options.get((pair, direction))
            if option is not None:
                cost[row, column] = -math.log1p(option.static_priority)
    row_indices, column_indices = linear_sum_assignment(cost)
    if len(row_indices) != STORE_EVENTS or np.any(cost[row_indices, column_indices] >= 1e11):
        raise EC3Error("balanced four-direction assignment is infeasible")
    assigned = [
        options[(pairs[int(row)], slots[int(column)])]
        for row, column in sorted(
            zip(row_indices, column_indices, strict=True), key=lambda item: int(item[0])
        )
    ]
    counts = {
        DIRECTIONS[index]: sum(
            (row.source_class, row.target_class) == DIRECTIONS[index] for row in assigned
        )
        for index in range(len(DIRECTIONS))
    }
    if any(value != PER_DIRECTION for value in counts.values()):
        raise EC3Error(f"direction assignment is not balanced: {counts}")
    return assigned


def render_receiver(semantic: Any, tokens: np.ndarray, pair: int) -> tuple[np.ndarray, ...]:
    import torch
    from torch.nn import functional

    with torch.inference_mode():
        pre_r = semantic(
            torch.from_numpy(np.asarray(tokens).copy())[None].long(),
            torch.tensor([pair], dtype=torch.long),
        )
        camera = (
            functional.interpolate(
                pre_r, size=(CAM_H, CAM_W), mode="bilinear", align_corners=False
            )
            .clamp(0.0, 255.0)
            .round()
            .to(torch.uint8)
        )
        scorer = functional.interpolate(
            camera.float(), size=(H, W), mode="bilinear", align_corners=False
        )
    return (
        pre_r[0].cpu().numpy().astype(np.float32, copy=False),
        camera[0].permute(1, 2, 0).cpu().numpy(),
        scorer[0].cpu().numpy().astype(np.float32, copy=False),
    )


def net_s_price(*, projected_flips: float, predicted_delta_d_pose: float) -> dict[str, float]:
    delta_d_seg = -float(projected_flips) / PIXELS
    seg_delta_s = 100.0 * delta_d_seg
    pose_delta_s = POSE_MARGINAL * float(predicted_delta_d_pose)
    return {
        "predicted_delta_d_seg": delta_d_seg,
        "predicted_seg_delta_s": seg_delta_s,
        "predicted_delta_d_pose": float(predicted_delta_d_pose),
        "predicted_pose_delta_s_at_603_marginal": pose_delta_s,
        "predicted_joint_net_delta_s": seg_delta_s + pose_delta_s,
    }


def nearest_projector_pair(pair: int, projector_pairs: list[int]) -> int:
    return min(projector_pairs, key=lambda value: (abs(value - pair), value))


def load_retained_proposal(receipt_path: Path, event_store: Path) -> dict[str, Any] | None:
    if not receipt_path.is_file():
        return None
    prior = json.loads(receipt_path.read_text())
    for record in prior["consumer_payloads"].values():
        require_record(record, beneath=event_store)
    prior["proposal_receipt"] = file_record(receipt_path)
    return prior


def process_event(
    *,
    ordinal: int,
    option: StaticOption,
    event_store: Path,
    semantic: Any,
    tokens: np.memmap,
    base_raw: np.memmap,
    base_argmax: np.ndarray,
    gt_argmax: np.ndarray,
    sensitivity: np.ndarray,
    calibration: list[dict[str, Any]],
    jacobian_records: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    pair = option.pair
    source = option.source_class
    target = option.target_class
    event_type = (
        ec1.EVENT_TYPE["lane_program_delta"]
        if (source, target) in ((0, 1), (1, 0))
        else ec1.EVENT_TYPE["boundary_offset"]
    )
    indices = np.asarray([option.token_index], dtype=np.int64)
    payload = ec1.proposal_payload(pair, source, target, indices, event_type)
    proposal_id = f"ec3_{ordinal:04d}_{hashlib.sha256(payload).hexdigest()[:12]}"
    root = event_store / "proposals" / proposal_id
    receipt_path = root / "proposal.json"
    prior = load_retained_proposal(receipt_path, event_store)
    if prior is not None:
        return prior
    decoded = ec1.decode_proposal(payload)
    if decoded[:4] != (pair, source, target, event_type) or not np.array_equal(decoded[4], indices):
        raise EC3Error("EC1 proposal parse-back differs")
    base_tokens = np.asarray(tokens[pair])
    if int(base_tokens.reshape(-1)[option.token_index]) != source:
        raise EC3Error("event source-token precondition differs")
    candidate_tokens = base_tokens.copy()
    candidate_tokens.reshape(-1)[option.token_index] = target
    base_pre_r, base_camera, base_scorer = render_receiver(semantic, base_tokens, pair)
    expected_camera = np.asarray(base_raw[2 * pair + 1])
    if not np.array_equal(base_camera, expected_camera):
        raise EC3Error(f"pair {pair} inactive CP135 receiver replay differs")
    candidate_pre_r, candidate_camera, candidate_scorer = render_receiver(
        semantic, candidate_tokens, pair
    )
    pre_r_delta = candidate_pre_r - base_pre_r
    scorer_delta = candidate_scorer - base_scorer
    magnitude = np.max(np.abs(scorer_delta), axis=0)
    strong_support = magnitude >= STRONG_DELTA_LSB
    camera_changed = int(np.count_nonzero(candidate_camera != base_camera))
    scorer_changed = int(np.count_nonzero(scorer_delta))
    if camera_changed == 0 or scorer_changed == 0 or not np.any(strong_support):
        raise EC3Error(f"event {proposal_id} is not receiver-effective at the strong-support surface")
    target_errors = (np.asarray(base_argmax[pair]) == source) & (
        np.asarray(gt_argmax[pair]) == target
    )
    raw_target_mass = int(np.count_nonzero(strong_support & target_errors))
    projected_flips = EXACT_TRANSFER_PRECISION * raw_target_mass
    projector_pair = nearest_projector_pair(pair, sorted(jacobian_records))
    jacobian_path = require_record(jacobian_records[projector_pair])
    jacobian = np.load(jacobian_path, mmap_mode="r", allow_pickle=False)
    pose_proxy = receiver_delta_pose_proxy(jacobian, scorer_delta)
    y, x = divmod(option.token_index, W)
    pose_prediction = predict_pose_cost(
        calibration,
        proxy=pose_proxy,
        sensitivity=float(sensitivity[y, x]),
        y=y,
        x=x,
        source_class=source,
        target_class=target,
    )
    pricing = net_s_price(
        projected_flips=projected_flips,
        predicted_delta_d_pose=float(pose_prediction["predicted_delta_d_pose_global_n600"]),
    )
    event_br = brotli.compress(payload, quality=11)
    event_xz = lzma.compress(payload, preset=9 | lzma.PRESET_EXTREME)
    consumer_payloads = {
        "event.ec1p": atomic_bytes(root / "event.ec1p", payload),
        "event.ec1p.br": atomic_bytes(root / "event.ec1p.br", event_br),
        "event.ec1p.xz": atomic_bytes(root / "event.ec1p.xz", event_xz),
        "candidate_tokens.uint8.npy": atomic_npy(
            root / "candidate_tokens.uint8.npy", candidate_tokens
        ),
        "base_pre_r.float16.npy": atomic_npy(
            root / "base_pre_r.float16.npy", base_pre_r.astype(np.float16)
        ),
        "candidate_pre_r.float16.npy": atomic_npy(
            root / "candidate_pre_r.float16.npy", candidate_pre_r.astype(np.float16)
        ),
        "pre_r_delta.float16.npy": atomic_npy(
            root / "pre_r_delta.float16.npy", pre_r_delta.astype(np.float16)
        ),
        "base_camera.uint8.npy": atomic_npy(root / "base_camera.uint8.npy", base_camera),
        "camera.uint8.npy": atomic_npy(root / "camera.uint8.npy", candidate_camera),
        "base_scorer_input.float16.npy": atomic_npy(
            root / "base_scorer_input.float16.npy", base_scorer.astype(np.float16)
        ),
        "scorer_input.float16.npy": atomic_npy(
            root / "scorer_input.float16.npy", candidate_scorer.astype(np.float16)
        ),
        "scorer_delta.float16.npy": atomic_npy(
            root / "scorer_delta.float16.npy", scorer_delta.astype(np.float16)
        ),
        "scorer_delta_max_abs.float16.npy": atomic_npy(
            root / "scorer_delta_max_abs.float16.npy", magnitude.astype(np.float16)
        ),
        "strong_support.bool.npy": atomic_npy(root / "strong_support.bool.npy", strong_support),
    }
    receipt = {
        "schema": "ddm_ec3_receiver_proposal.v1",
        "axis": AXIS,
        "score_claim": False,
        "acceptance_tested": False,
        "proposal_id": proposal_id,
        "ordinal": ordinal,
        "pair": pair,
        "event_type": ec1.EVENT_NAME[event_type],
        "source_class": ec1.CLASSES[source],
        "target_class": ec1.CLASSES[target],
        "source_class_id": source,
        "target_class_id": target,
        "direction": DIRECTION_NAMES[(source, target)],
        "site_count": 1,
        "token_yx": [y, x],
        "source_archive_sha256": ec1.CP135_ARCHIVE_SHA,
        "parse_back_exact": True,
        "receiver_effective": True,
        "receiver_effectiveness": {
            "inactive_base_camera_byte_identical": True,
            "camera_changed_values": camera_changed,
            "scorer_lattice_changed_values": scorer_changed,
            "strong_support_pixels_at_ge_1_lsb": int(np.count_nonzero(strong_support)),
        },
        "construction": {
            **asdict(option),
            "minimality": "one EC1 token substitution; no multi-site segment and no amplitude sweep",
            "target_field": file_record(T4_BASE_ARGMAX),
            "gt_field": file_record(T4_GT_ARGMAX),
        },
        "prescreen": {
            "axis": AXIS,
            "verdict_authority": False,
            "raw_t4_target_errors_in_ge_1_lsb_receiver_support": raw_target_mass,
            "advisory_to_exact_precision": EXACT_TRANSFER_PRECISION,
            "projected_exact_flip_gain": projected_flips,
            "pose_proxy_js4_pair": projector_pair,
            "pose_proxy_value": pose_proxy,
            "pose_prediction": pose_prediction,
            "joint_net_s_pricing": pricing,
            "eligible": pricing["predicted_joint_net_delta_s"] < 0.0,
            "rate_term": "0 S projected: JO1/CP5V direct-token carrier precedent; not remeasured here",
        },
        "predicted_flips": projected_flips,
        "predicted_pose_bound_global_n600": pose_prediction[
            "predicted_delta_d_pose_global_n600"
        ],
        "predicted_bytes_brotli_q11": consumer_payloads["event.ec1p.br"]["bytes"],
        "predicted_net_score_gain_standalone_bytes": -pricing[
            "predicted_joint_net_delta_s"
        ],
        "consumer_payloads": consumer_payloads,
    }
    atomic_json(receipt_path, receipt)
    receipt["proposal_receipt"] = file_record(receipt_path)
    return receipt


def build_store(
    event_store: Path,
    rows: list[dict[str, Any]],
    pair_selection: dict[str, Any],
) -> dict[str, Any]:
    if len(rows) != STORE_EVENTS or len({int(row["pair"]) for row in rows}) != STORE_EVENTS:
        raise EC3Error("EC3 rows are not 200 unique pairs")
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["prescreen"]["joint_net_s_pricing"]["predicted_joint_net_delta_s"]),
            -float(row["prescreen"]["projected_exact_flip_gain"]),
            int(row["pair"]),
        ),
    )
    index_lines = []
    for ordinal, row in enumerate(ranked):
        row = {**row, "ordinal": ordinal}
        index_lines.append((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
    index_record = atomic_bytes(event_store / "proposal_index.jsonl", b"".join(index_lines))
    eligible = [
        row
        for row in ranked
        if float(row["prescreen"]["joint_net_s_pricing"]["predicted_joint_net_delta_s"])
        < 0.0
    ]
    optimistic_gain = sum(
        -float(row["prescreen"]["joint_net_s_pricing"]["predicted_joint_net_delta_s"])
        for row in eligible
    )
    direction_counts = {
        name: sum(row["direction"] == name for row in ranked)
        for name in DIRECTION_NAMES.values()
    }
    selection = {
        "selection_mode": (
            "one event per pair; 50 per directed family; rank by predicted "
            "100*delta_d_seg + 603*delta_d_pose"
        ),
        "selected_events": len(ranked),
        "eligible_events": len(eligible),
        "eligible_proposal_ids": [str(row["proposal_id"]) for row in eligible],
        "optimistic_additive_eligible_gain_s": optimistic_gain,
        "fire_bar_s": FIRE_BAR_S,
        "bar_pass": optimistic_gain >= FIRE_BAR_S,
        "direction_counts": direction_counts,
        "support_disjoint_by_pair": True,
        "pair_selection": pair_selection,
    }
    state = {
        "schema": "ddm_js5_realized_acceptance_200_store.v1",
        "producer": "ddm_ec3_t4_targeted_events",
        "status": "PRODUCED_NOT_ACCEPTANCE_TESTED",
        "proposal_count": STORE_EVENTS,
        "receiver_effective_count": STORE_EVENTS,
        "attempt_count": STORE_EVENTS,
        "sample": [int(row["pair"]) for row in ranked],
        "source_archive_sha256": ec1.CP135_ARCHIVE_SHA,
        "proposal_index": index_record,
        "acceptance_tested": False,
        "score_claim": False,
        "axis": AXIS,
        "intended_selection": selection,
        "fire_boundary": "MAIN owns the unchanged VD1 T4 validator and sole n600 scorer slot",
    }
    atomic_json(event_store / "state.json", state)
    return {
        "schema": "ddm_ec3_store_result.v1",
        "event_store": str(event_store.resolve()),
        "proposal_index": index_record,
        "state": file_record(event_store / "state.json"),
        "selection": selection,
        "ranked_rows": ranked,
    }


def validate_unchanged_vd1(event_store: Path, run_root: Path) -> dict[str, Any]:
    """Run only VD1's local bundle builder; never call its Modal function."""
    from experiments import ddm_vd1_modal_batch_event_validator as vd1

    bundle, manifest = vd1.build_event_bundle(event_store, JO1_ANALYSIS, k=STORE_EVENTS)
    bundle_record = atomic_bytes(run_root / "unchanged_vd1/event_bundle_k200.zip", bundle)
    if (
        manifest.get("schema") != "ddm_vd1_event_bundle.v1"
        or manifest.get("selection_mode") != "full_200_census"
        or int(manifest.get("selected_events", -1)) != STORE_EVENTS
        or manifest.get("bundle_sha256") != bundle_record["sha256"]
    ):
        raise EC3Error("unchanged VD1 bundle proof differs")
    result = {
        "schema": "ddm_ec3_unchanged_vd1_compatibility.v1",
        "compatible": True,
        "dispatch": False,
        "event_store": str(event_store.resolve()),
        "bundle": bundle_record,
        "manifest": manifest,
        "k_arithmetic": vd1.k_arithmetic(),
        "axis": "local schema/bundle parse-back only; no Modal or scorer",
        "score_claim": False,
    }
    atomic_json(run_root / "unchanged_vd1/COMPATIBILITY.json", result)
    return result


def dispatch_command(event_store: Path) -> str:
    return " ".join(
        (
            ".venv/bin/modal run experiments/ddm_vd1_modal_batch_event_validator.py",
            "--archive /Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip",
            "--runtime /Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime",
            f"--event-store {event_store.resolve()}",
            "--jo1-analysis /Volumes/VertigoDataTier/pact/ddm_jo1_20260812/10_ANALYSIS.json",
            "--output-dir .omx/state/ddm_vd1_modal_batch_event_validator/ddm_ec3_vd1_20260813a",
            "--k 200 --run-id ddm_ec3_vd1_20260813a",
            "--resume-from ddm_ec3_vd1_20260813a",
            "--lane-id ddm_vd1_modal_batch_event_validator",
            "--instance-job-id modal:ddm_ec3_vd1_20260813a",
            "--claim-agent main:ddm_ec3",
        )
    )


def run(run_root: Path, event_store: Path) -> dict[str, Any]:
    started = time.perf_counter()
    preflight(run_root, event_store)
    sensitivity, pose_prior = build_pose_sensitivity(run_root)
    calibration, calibration_receipt = build_calibration(run_root, sensitivity)
    base_argmax = np.load(T4_BASE_ARGMAX, mmap_mode="r", allow_pickle=False)
    gt_argmax = np.load(T4_GT_ARGMAX, mmap_mode="r", allow_pickle=False)
    tokens = np.memmap(ec1.BASE_TOKENS, mode="r", dtype=np.uint8, shape=(N, H, W))
    sensitivity_norm = normalized_sensitivity(sensitivity)
    pair_error_counts = np.count_nonzero(base_argmax != gt_argmax, axis=(1, 2))
    direction_available = np.zeros((N, len(DIRECTIONS)), dtype=bool)
    for pair in range(N):
        for direction, (source, target) in enumerate(DIRECTIONS):
            direction_available[pair, direction] = bool(
                np.any((base_argmax[pair] == source) & (gt_argmax[pair] == target))
                and np.any(tokens[pair] == source)
            )
    g3 = json.loads(G3_REGISTRY.read_text())
    pairs, pair_selection = choose_pair_set(
        pair_error_counts,
        [int(value) for value in g3["top64"]],
        [int(value) for value in g3["stratified_control24"]],
        direction_available,
    )
    options: dict[tuple[int, int], StaticOption] = {}
    for pair in pairs:
        for direction, (source, target) in enumerate(DIRECTIONS):
            option = select_token_site(
                np.asarray(tokens[pair]),
                np.asarray(base_argmax[pair]),
                np.asarray(gt_argmax[pair]),
                sensitivity,
                source,
                target,
                pair=pair,
                sensitivity_norm=sensitivity_norm,
            )
            if option is not None:
                options[(pair, direction)] = option
    assigned = assign_directions(pairs, options)
    atomic_json(
        run_root / "30_ASSIGNMENT.json",
        {
            "schema": "ddm_ec3_assignment.v1",
            "axis": AXIS,
            "score_claim": False,
            "pair_selection": pair_selection,
            "directions": [asdict(row) for row in assigned],
        },
    )
    manifest = json.loads(JS4_MANIFEST.read_text())
    jacobian_records = {int(row["pair_id"]): row["jacobian"] for row in manifest["pairs"]}
    base_raw = np.memmap(
        ec1.BASE_RAW,
        mode="r",
        dtype=np.uint8,
        shape=(N * 2, CAM_H, CAM_W, 3),
    )
    proof_root = run_root / "retained/receiver_state"
    try:
        *_, semantic, _basis, _coefficients = js1.parse_receiver_state(
            js1.CANDIDATES["cp135_base"], proof_root
        )
        semantic = semantic.eval().cpu()
        rows = []
        for ordinal, option in enumerate(assigned):
            row = process_event(
                ordinal=ordinal,
                option=option,
                event_store=event_store,
                semantic=semantic,
                tokens=tokens,
                base_raw=base_raw,
                base_argmax=base_argmax,
                gt_argmax=gt_argmax,
                sensitivity=sensitivity,
                calibration=calibration,
                jacobian_records=jacobian_records,
            )
            rows.append(row)
            atomic_json(
                run_root / "RUN_STATE.json",
                {
                    "schema": "ddm_ec3_run_state.v1",
                    "status": "RUNNING",
                    "resumable": True,
                    "completed_events": len(rows),
                    "required_events": STORE_EVENTS,
                    "latest_proposal_id": row["proposal_id"],
                    "axis": AXIS,
                    "score_claim": False,
                },
            )
    finally:
        js1.release_runtime()
    store = build_store(event_store, rows, pair_selection)
    compatibility = validate_unchanged_vd1(event_store, run_root)
    selection = store["selection"]
    falsifier_fired = not bool(selection["bar_pass"])
    command = dispatch_command(event_store)
    disposition = "FALSIFIER_FIRED_FORMULATION_CLOSED" if falsifier_fired else "READY_TO_FIRE"
    result = {
        "schema": "ddm_ec3_final_result.v1",
        "status": disposition,
        "axis": AXIS,
        "score_claim": False,
        "pointer_moved": False,
        "measured": (
            "full-n600 T4-field census, exact retained receiver replay, payload bytes/hashes, "
            "VD1-T4/JS4 calibration join, local net-S ordering, unchanged VD1 bundle compatibility"
        ),
        "not_measured": (
            "candidate SegNet or PoseNet output, exact per-event delta, composed archive, "
            "contest score, Modal row, MPS"
        ),
        "store": {key: value for key, value in store.items() if key != "ranked_rows"},
        "pose_prior": file_record(run_root / "10_POSE_PRIOR.json"),
        "calibration": calibration_receipt,
        "unchanged_vd1": {
            "compatible": compatibility["compatible"],
            "dispatch": compatibility["dispatch"],
            "receipt": file_record(run_root / "unchanged_vd1/COMPATIBILITY.json"),
            "bundle": compatibility["bundle"],
            "k_arithmetic": compatibility["k_arithmetic"],
        },
        "falsifier": {
            "fired": falsifier_fired,
            "bar_s": FIRE_BAR_S,
            "optimistic_additive_eligible_gain_s": selection[
                "optimistic_additive_eligible_gain_s"
            ],
            "verdict_scope": (
                "FORMULATION: one target-cluster anchor per pair, 200 unique pairs, balanced "
                "50/50/50/50 across four directed variants of the three required edge families; "
                "CP135 minimal one-token EC1 wire; retained T4 disagreements; current-n600/G3 "
                "coverage; 13-percent transfer; q25 calibrated pose; joint net-S"
            ),
            "route_if_fired": (
                "this CP135 target-cluster singleton discrete-event formulation is closed; "
                "do not spend the VD1 row"
            ),
        },
        "vd1_dispatch": {
            "disposition": "FOLDED" if falsifier_fired else "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "MAIN",
            "consumer_store": str(event_store.resolve()),
            "fire_trigger": (
                "none; falsifier fired"
                if falsifier_fired
                else "MAIN owns the sole scorer lane, verifies pins, and claims the VD1 lane"
            ),
            "command": command,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(run_root / "FINAL_RESULT.json", result)
    atomic_json(
        run_root / "RUN_STATE.json",
        {
            "schema": "ddm_ec3_run_state.v1",
            "status": "COMPLETE",
            "resumable": True,
            "completed_events": STORE_EVENTS,
            "required_events": STORE_EVENTS,
            "event_store": str(event_store.resolve()),
            "final_result": file_record(run_root / "FINAL_RESULT.json"),
            "axis": AXIS,
            "score_claim": False,
        },
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--event-store", type=Path, default=EVENT_STORE)
    args = parser.parse_args(argv)
    result = run(args.run_root.resolve(), args.event_store.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
