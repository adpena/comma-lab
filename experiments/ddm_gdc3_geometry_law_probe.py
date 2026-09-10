#!/usr/bin/env python3
"""Measure exact-residual rate by error geometry on two retained n600 renders.

This scorer-free probe partitions every categorical mismatch by horizontal run
length, target-boundary proximity, and target class. Each non-empty partition
is serialized in the existing HG1 residual wire format, raced through the same
three physical coders, and retained with a deterministic repeat. Standalone
partition byte sizes are deliberately reported as non-additive measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

import numpy as np
from scipy import ndimage

REPO: Final = Path(__file__).resolve().parents[1]
for _root in (REPO, REPO / "src", REPO / "experiments"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_et1_edge_topology_container_gate as et1
from experiments import ddm_hg1_heterogeneous_analytic_generator_gate as hg1

N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
PLANE: Final = HEIGHT * WIDTH
SHAPE: Final = (N_PAIRS, HEIGHT, WIDTH)
FIELD_BYTES: Final = N_PAIRS * PLANE
SEED: Final = 20_260_910
AXIS: Final = "[macOS-CPU advisory / scorer-free exact byte measurement]"
CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
RUN_NAMES: Final = ("isolated", "short", "long")
PROXIMITY_NAMES: Final = ("boundary_adjacent", "interior")
GEOMETRY_NAMES: Final = tuple(f"{run}_{proximity}" for run in RUN_NAMES for proximity in PROXIMITY_NAMES)
REQUIRED_ORDERS: Final = ("tile64_time", "frame_raster")
DEFAULT_OUTPUT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_gdc3_next_construction_against_the_geometry_law/geometry_probe_v1"
)
MIN_FREE_BYTES: Final = 1_000_000_000

TARGET: Final = (
    Path("/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/pass6.u8"),
    "78e57545439515eb29f806cc5a5f7d8b14acf658955cdd7561debb4edf3b7db6",
)
SOURCES: Final = (
    (
        "gdc1_k8",
        Path("/Volumes/VertigoDataTier/pact/ddm_gdc1_generator_door/scanline_v1/k08/receiver_render.u8"),
        "abd130921beec23255112e8b56148d55c49cf4fede03e99e0e99b55378b16ea2",
    ),
    (
        "gdc2_stageC_lam1_0.0003_final",
        Path(
            "/Volumes/VertigoDataTier/pact/"
            "ddm_gdc2_categorical_coolchic_k8_distill/governed_v1/exports/"
            "stageC_lam1_0.0003_final/render.u8"
        ),
        "62a25bc44658b3872df7e4913900e70441d4bc4a06251b00e50fa7db160890d7",
    ),
)


class GeometryProbeError(RuntimeError):
    """A pinned-source, geometry, coder, or custody invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verified_field(path: Path, expected_sha256: str) -> np.ndarray:
    if not path.is_file():
        raise GeometryProbeError(f"missing pinned field: {path}")
    if path.stat().st_size != FIELD_BYTES:
        raise GeometryProbeError(f"pinned field has {path.stat().st_size} B, expected {FIELD_BYTES}: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise GeometryProbeError(f"pinned field hash mismatch: {observed} != {expected_sha256}: {path}")
    field = np.memmap(path, dtype=np.uint8, mode="r", shape=SHAPE)
    if any(int(np.max(field[pair])) > 4 for pair in range(N_PAIRS)):
        raise GeometryProbeError(f"categorical field escaped classes 0..4: {path}")
    return field


def two_sided_boundary(labels: np.ndarray) -> np.ndarray:
    """Return both endpoints of each horizontal or vertical label transition."""

    values = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError("boundary input must be a 2-D label plane")
    boundary = np.zeros(values.shape, dtype=bool)
    horizontal = values[:, 1:] != values[:, :-1]
    vertical = values[1:, :] != values[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def horizontal_run_buckets(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exclusive run bucket, run id, and run length for true cells.

    Bucket codes are 0 for length 1, 1 for lengths 2..8, and 2 for lengths >=9.
    False cells receive -1 in every output. Runs never cross a row boundary.
    """

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("run input must be a 2-D mismatch plane")
    bucket = np.full(values.shape, -1, dtype=np.int8)
    run_id = np.full(values.shape, -1, dtype=np.int64)
    run_length = np.full(values.shape, -1, dtype=np.int32)
    next_id = 0
    for y, row in enumerate(values):
        padded = np.concatenate((np.array([False]), row, np.array([False])))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        for start, stop in zip(starts.tolist(), stops.tolist(), strict=True):
            length = stop - start
            code = 0 if length == 1 else 1 if length <= 8 else 2
            bucket[y, start:stop] = code
            run_id[y, start:stop] = next_id
            run_length[y, start:stop] = length
            next_id += 1
    return bucket, run_id, run_length


def classify_plane(target: np.ndarray, generated: np.ndarray, run_id_offset: int = 0) -> dict[str, np.ndarray]:
    """Classify one full label plane into the six exclusive geometry cells."""

    target_values = np.asarray(target, dtype=np.uint8)
    generated_values = np.asarray(generated, dtype=np.uint8)
    if target_values.shape != generated_values.shape or target_values.ndim != 2:
        raise ValueError("target and generated must be equal-shape 2-D planes")
    mismatch = target_values != generated_values
    run_bucket, run_id, run_length = horizontal_run_buckets(mismatch)
    boundary = two_sided_boundary(target_values)
    if boundary.any():
        distance = ndimage.distance_transform_edt(~boundary)
    else:
        distance = np.full(target_values.shape, np.inf, dtype=np.float64)
    proximity = np.where(distance <= 1.0, 0, 1).astype(np.int8)
    geometry = np.where(mismatch, run_bucket * 2 + proximity, -1).astype(np.int8)
    run_id = np.where(mismatch, run_id + run_id_offset, -1)
    return {
        "mismatch": mismatch,
        "geometry": geometry,
        "run_id": run_id,
        "run_length": run_length,
        "target_class": target_values,
    }


def collect_mismatches(target: np.ndarray, generated: np.ndarray) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {
        "address": [],
        "target_class": [],
        "geometry": [],
        "run_id": [],
        "run_length": [],
    }
    run_id_offset = 0
    for pair in range(N_PAIRS):
        classified = classify_plane(target[pair], generated[pair], run_id_offset)
        positions = np.flatnonzero(classified["mismatch"].reshape(-1))
        if positions.size:
            chunks["address"].append((pair * PLANE + positions).astype(np.int64))
            for name in ("target_class", "geometry", "run_id", "run_length"):
                chunks[name].append(classified[name].reshape(-1)[positions])
            run_id_offset = int(np.max(chunks["run_id"][-1])) + 1
    if not chunks["address"]:
        raise GeometryProbeError("retained render has zero mismatches; geometry premise is void")
    return {name: np.concatenate(values) for name, values in chunks.items()}


def residual_sort_order(addresses: np.ndarray, labels: np.ndarray, order: str) -> np.ndarray:
    if order == "frame_raster":
        return np.argsort(addresses, kind="stable")
    if order != "tile64_time":
        raise ValueError(f"unsupported required order: {order}")
    pair, position = np.divmod(addresses, PLANE)
    y, x = np.divmod(position, WIDTH)
    in_tile = (y % 64) * 64 + (x % 64)
    return np.lexsort((labels, in_tile, pair, x // 64, y // 64))


def serialize_residual(addresses: np.ndarray, labels: np.ndarray, order: str) -> bytes:
    permutation = residual_sort_order(addresses, labels, order)
    ordered_addresses = np.asarray(addresses, dtype=np.int64)[permutation]
    ordered_labels = np.asarray(labels, dtype=np.uint8)[permutation]
    output = bytearray(
        hg1.RESIDUAL_HEADER.pack(
            hg1.RESIDUAL_MAGIC,
            2,
            hg1.RESIDUAL_ORDER_IDS[order],
            N_PAIRS,
            HEIGHT,
            WIDTH,
            len(ordered_addresses),
        )
    )
    previous = -1
    for address, label in zip(ordered_addresses.tolist(), ordered_labels.tolist(), strict=True):
        hg1.put_uleb(output, hg1.zigzag(int(address) - previous))
        output.append(int(label))
        previous = int(address)
    return bytes(output)


def parse_residual(payload: bytes) -> tuple[np.ndarray, np.ndarray, str]:
    if len(payload) < hg1.RESIDUAL_HEADER.size:
        raise GeometryProbeError("residual payload is truncated")
    magic, version, order_id, pairs, height, width, count = hg1.RESIDUAL_HEADER.unpack_from(payload)
    if (
        magic != hg1.RESIDUAL_MAGIC
        or version != 2
        or order_id not in hg1.ID_RESIDUAL_ORDERS
        or (pairs, height, width) != SHAPE
    ):
        raise GeometryProbeError("residual header does not match the pinned HG1 format")
    offset = hg1.RESIDUAL_HEADER.size
    previous = -1
    addresses = np.empty(count, dtype=np.int64)
    labels = np.empty(count, dtype=np.uint8)
    for index in range(count):
        coded_delta, offset = hg1.get_uleb(payload, offset)
        address = previous + hg1.unzigzag(coded_delta)
        if not 0 <= address < FIELD_BYTES or offset >= len(payload):
            raise GeometryProbeError("residual record escaped the field")
        label = payload[offset]
        offset += 1
        if label > 4:
            raise GeometryProbeError("residual target class escaped 0..4")
        addresses[index] = address
        labels[index] = label
        previous = address
    if offset != len(payload):
        raise GeometryProbeError("residual payload has trailing bytes")
    return addresses, labels, hg1.ID_RESIDUAL_ORDERS[order_id]


def retain_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise GeometryProbeError(f"refusing to overwrite divergent retained payload: {path}")
    else:
        et1.atomic_bytes(path, payload)
    return file_fact(path)


def coder_race_strict(name: str, raw_path: Path, root: Path) -> dict[str, Any]:
    raw = raw_path.read_bytes()
    for coder in hg1.CODERS:
        directory = root / "retained" / "coder_races" / name / coder
        coded_path = directory / "payload.coded"
        repeat_path = directory / "payload.repeat.coded"
        present = (coded_path.exists(), repeat_path.exists())
        if any(present) and not all(present):
            raise GeometryProbeError(f"incomplete retained coder pair: {directory}")
        if all(present):
            coded = coded_path.read_bytes()
            repeated = repeat_path.read_bytes()
            if coded != repeated or et1.decompress_payload(coded, coder) != raw:
                raise GeometryProbeError(f"divergent retained coder pair: {directory}")
    return hg1.coder_race(name, raw_path, root)


def verify_subset_payload(
    payload: bytes, expected_addresses: np.ndarray, target_flat: np.ndarray, expected_order: str
) -> None:
    addresses, labels, observed_order = parse_residual(payload)
    if observed_order != expected_order:
        raise GeometryProbeError(f"residual order changed: {observed_order} != {expected_order}")
    if len(np.unique(addresses)) != len(addresses):
        raise GeometryProbeError("residual payload contains duplicate addresses")
    if not np.array_equal(np.sort(addresses), np.sort(expected_addresses)):
        raise GeometryProbeError("residual payload address set differs from selected geometry")
    if not np.array_equal(labels, np.asarray(target_flat)[addresses]):
        raise GeometryProbeError("residual payload labels differ from the exact target")


def selection_rows(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    total = len(data["address"])
    rows: list[dict[str, Any]] = [
        {"scope": "full", "geometry": None, "target_class": None, "mask": np.ones(total, bool)}
    ]
    for geometry, name in enumerate(GEOMETRY_NAMES):
        rows.append(
            {
                "scope": "geometry_marginal",
                "geometry": name,
                "target_class": None,
                "mask": data["geometry"] == geometry,
            }
        )
    for class_id, class_name in enumerate(CLASS_NAMES):
        rows.append(
            {
                "scope": "class_marginal",
                "geometry": None,
                "target_class": class_name,
                "target_class_id": class_id,
                "mask": data["target_class"] == class_id,
            }
        )
    for geometry, name in enumerate(GEOMETRY_NAMES):
        for class_id, class_name in enumerate(CLASS_NAMES):
            rows.append(
                {
                    "scope": "geometry_by_class",
                    "geometry": name,
                    "target_class": class_name,
                    "target_class_id": class_id,
                    "mask": (data["geometry"] == geometry) & (data["target_class"] == class_id),
                }
            )
    return rows


def run_statistics(data: dict[str, np.ndarray], selected: np.ndarray) -> dict[str, Any]:
    ids = data["run_id"][selected]
    lengths = data["run_length"][selected]
    if ids.size == 0:
        return {"runs": 0, "run_length_quantiles": None}
    _, first = np.unique(ids, return_index=True)
    unique_lengths = lengths[first].astype(np.float64)
    quantiles = np.quantile(unique_lengths, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "runs": len(first),
        "run_length_quantiles": {
            key: float(value) for key, value in zip(("min", "q25", "median", "q75", "max"), quantiles, strict=True)
        },
    }


def close_full_residual(
    source: np.ndarray,
    target_path: Path,
    raw_payload: bytes,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        fact = file_fact(output_path)
        target_fact = file_fact(target_path)
        if fact["bytes"] != target_fact["bytes"] or fact["sha256"] != target_fact["sha256"]:
            raise GeometryProbeError(f"retained exact-closure field diverged: {output_path}")
        return fact
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    closed = np.memmap(temporary, mode="w+", dtype=np.uint8, shape=SHAPE)
    closed[:] = source
    corrected = hg1.apply_residual(raw_payload, closed)
    closed.flush()
    del closed
    os.replace(temporary, output_path)
    if corrected <= 0:
        raise GeometryProbeError("full residual unexpectedly applied zero corrections")
    fact = file_fact(output_path)
    target_fact = file_fact(target_path)
    if fact["bytes"] != target_fact["bytes"] or fact["sha256"] != target_fact["sha256"]:
        raise GeometryProbeError(f"full residual did not close the target: {output_path}")
    return fact


def measure_source(
    source_name: str,
    source_path: Path,
    source: np.ndarray,
    target_path: Path,
    target: np.ndarray,
    root: Path,
) -> dict[str, Any]:
    data = collect_mismatches(target, source)
    target_flat = target.reshape(-1)
    total = len(data["address"])
    rows: list[dict[str, Any]] = []
    for selection_with_mask in selection_rows(data):
        selection = dict(selection_with_mask)
        selected = np.asarray(selection.pop("mask"), dtype=bool)
        count = int(np.count_nonzero(selected))
        row: dict[str, Any] = {
            **selection,
            "mismatches": count,
            "mismatch_share_of_source": count / total,
            **run_statistics(data, selected),
            "standalone_bytes_are_non_additive": True,
            "orders": [],
        }
        if count == 0:
            rows.append(row)
            continue
        addresses = data["address"][selected]
        labels = data["target_class"][selected]
        slug = "__".join(
            str(value)
            for value in (
                row["scope"],
                row.get("geometry") or "all_geometry",
                row.get("target_class") or "all_classes",
            )
        ).lower()
        for order in REQUIRED_ORDERS:
            payload = serialize_residual(addresses, labels, order)
            raw_path = root / "retained" / source_name / slug / order / "residual.raw"
            raw_fact = retain_bytes(raw_path, payload)
            verify_subset_payload(payload, addresses, target_flat, order)
            race_name = f"{source_name}__{slug}__{order}"
            race = coder_race_strict(race_name, raw_path, root)
            winner = str(race["winner"])
            coded = race["coders"][winner]["coded"]
            order_row: dict[str, Any] = {
                "order": order,
                "raw": raw_fact,
                "coders": {
                    coder: {
                        "coded": race["coders"][coder]["coded"],
                        "repeat": race["coders"][coder]["repeat"],
                        "deterministic_repeat_equal": race["coders"][coder]["deterministic_repeat_equal"],
                        "raw_parseback_equal": race["coders"][coder]["raw_parseback_equal"],
                    }
                    for coder in hg1.CODERS
                },
                "winner": winner,
                "winning_coded_bytes": int(coded["bytes"]),
                "winning_bytes_per_mismatch": int(coded["bytes"]) / count,
                "subset_parseback_exact": True,
            }
            if row["scope"] == "full":
                closure_path = root / "retained" / source_name / "exact_closure" / order / "field.u8"
                order_row["exact_closure"] = close_full_residual(source, target_path, payload, closure_path)
            row["orders"].append(order_row)
        rows.append(row)
    if sum(int(row["mismatches"]) for row in rows if row["scope"] == "geometry_by_class") != total:
        raise GeometryProbeError("six-by-five geometry table did not partition every mismatch")
    return {
        "source": source_name,
        "render": file_fact(source_path),
        "mismatches": total,
        "mismatch_fraction": total / FIELD_BYTES,
        "rows": rows,
    }


def manifest_rows(root: Path) -> list[dict[str, Any]]:
    manifest = root / "MANIFEST.json"
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != manifest):
        rows.append(file_fact(path))
    return rows


def git_commit() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    root: Path = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < MIN_FREE_BYTES:
        raise GeometryProbeError(f"storage preflight refused: {free} free B < {MIN_FREE_BYTES} required B")

    started = time.monotonic()
    target_path, target_sha = TARGET
    target = verified_field(target_path, target_sha)
    results = []
    for name, path, expected_sha in SOURCES:
        source = verified_field(path, expected_sha)
        results.append(measure_source(name, path, source, target_path, target, root))
        del source

    result = {
        "schema": "ddm_gdc3_geometry_law_probe.v1",
        "axis": AXIS,
        "score_claim": False,
        "research_only": True,
        "selection_mode": "full n600, no subset",
        "seed": SEED,
        "geometry_definition": {
            "run_buckets": {"isolated": "1", "short": "2..8", "long": ">=9"},
            "target_boundary": "two-sided endpoints of 4-neighbour label transitions",
            "boundary_adjacent": "Euclidean distance to target boundary <= 1.0 pixel",
            "interior": "Euclidean distance to target boundary > 1.0 pixel",
            "exclusive_cells": list(GEOMETRY_NAMES),
            "standalone_subset_bytes_are_non_additive": True,
        },
        "orders": list(REQUIRED_ORDERS),
        "coders": list(hg1.CODERS),
        "target": file_fact(target_path),
        "sources": results,
        "implementation": file_fact(Path(__file__).resolve()),
        "git_commit_at_measurement": git_commit(),
        "host": os.uname().nodename,
        "elapsed_seconds": time.monotonic() - started,
        "storage": {
            "root": str(root),
            "free_bytes_at_preflight": free,
            "minimum_required_bytes": MIN_FREE_BYTES,
            "deletion": "none",
            "atomic_writes": True,
        },
    }
    result_path = root / "RESULT.json"
    et1.atomic_json(result_path, result)
    et1.atomic_json(
        root / "LATEST.json",
        {"schema": "ddm_gdc3_geometry_law_probe.latest.v1", "result": file_fact(result_path)},
    )
    rows = manifest_rows(root)
    et1.atomic_json(
        root / "MANIFEST.json",
        {
            "schema": "ddm_gdc3_geometry_law_probe.manifest.v1",
            "root": str(root),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "files": rows,
        },
    )
    print(
        json.dumps(
            {
                "result": file_fact(result_path),
                "manifest": file_fact(root / "MANIFEST.json"),
                "source_mismatches": {row["source"]: row["mismatches"] for row in results},
                "selection_mode": result["selection_mode"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
