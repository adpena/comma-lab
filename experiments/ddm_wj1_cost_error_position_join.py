#!/usr/bin/env python3
"""Join DX2 per-position RC64 cost with retained render-manufactured errors.

This is a scorer-free, deterministic n600 field join.  It copies the exact
retained inputs into the WJ1 local receipt store, verifies their hashes and
published counts, emits four nested cost-threshold masks, and writes a direct
position-list payload for JF1.  It never mutates the shipped receiver or the
source custody trees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = ROOT / ".omx/tmp/arm_receipts_local/ddm_wj1_cost_error_position_join/measurement_v1"
BL1_STORE = Path("/Volumes/VertigoDataTier/pact/ddm_bl1_per_position_bit_allocation/measurement_v1")
BL1_RESULT = BL1_STORE / "RESULT.json"
MST1_STORE = ROOT / ".omx/tmp/arm_receipts_local/ddm_mst1_manufactured_stage_split/capture_r2_local"
MST1_RESULT = MST1_STORE / "MST1_RESULT.json"

N_FRAMES = 600
HEIGHT = 384
WIDTH = 512
PLANE = HEIGHT * WIDTH
POSITIONS = N_FRAMES * PLANE
PACKED_BYTES_PER_FRAME = PLANE // 8
SHAPE = (N_FRAMES, HEIGHT, WIDTH)
STREAM_BYTES = 113_777
STREAM_BITS = STREAM_BYTES * 8
DEMAND_BYTES = 42_382
ARCHIVE_BYTES = 180_368
MODEL_BITS = 910_209.2806090603
RATE_S_PER_BYTE = 25.0 / 37_545_489.0

EXPECTED = {
    "archive": {
        "bytes": ARCHIVE_BYTES,
        "sha256": "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674",
    },
    "stream": {
        "bytes": STREAM_BYTES,
        "sha256": "e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5",
    },
    "decoded": {
        "bytes": POSITIONS,
        "sha256": "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",
    },
    "cost": {
        "bytes": POSITIONS * 8,
        "sha256": "99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86",
    },
    "gt": {
        "bytes": POSITIONS + 128,
        "sha256": "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248",
    },
}

MASKS = {
    "representation_error_support": 9_182,
    "state_wrong_native_render_head": 31_503,
    "state_wrong_preuint8_roundtrip_head": 24_523,
    "state_wrong_uint8_roundtrip_head": 23_752,
    "final_error_support": 23_757,
    "final_manufactured_support": 21_493,
    "gross_manufactured_native_render_head": 28_602,
}

CLASSES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
THRESHOLDS = (
    ("top_0p1pct", 0.001),
    ("top_1pct", 0.01),
    ("top_5pct", 0.05),
    ("top_10pct", 0.10),
)


class Wj1Error(RuntimeError):
    """Fail-closed WJ1 input or accounting error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_file(path: Path, expected_sha: str, expected_bytes: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise Wj1Error(f"required payload missing: {path}")
    fact = file_fact(path)
    if expected_bytes is not None and fact["bytes"] != expected_bytes:
        raise Wj1Error(f"byte drift for {path}: {fact['bytes']} != {expected_bytes}")
    if fact["sha256"] != expected_sha:
        raise Wj1Error(f"sha256 drift for {path}: {fact['sha256']} != {expected_sha}")
    return fact


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with tmp.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def copy_verified(source: Path, destination: Path, expected_sha: str, expected_bytes: int) -> dict[str, Any]:
    source_fact = verify_file(source, expected_sha, expected_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination_fact = verify_file(destination, expected_sha, expected_bytes)
        return {"source": source_fact, "retained": destination_fact, "resumed": True}
    tmp = destination.with_name(destination.name + f".partial.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    with source.open("rb") as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(tmp, destination)
    destination_fact = verify_file(destination, expected_sha, expected_bytes)
    return {"source": source_fact, "retained": destination_fact, "resumed": False}


def packed_frame(path: Path, frame: int) -> np.ndarray:
    packed = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=frame * PACKED_BYTES_PER_FRAME,
        shape=(PACKED_BYTES_PER_FRAME,),
    )
    return np.unpackbits(packed, bitorder="little", count=PLANE).astype(bool, copy=False)


def checked_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def cell_payload(
    *,
    positions: int,
    bits: float,
    expected_positions: float,
    expected_bits: float,
    position_denominator: int,
) -> dict[str, Any]:
    byte_mass = bits / 8.0
    return {
        "positions": positions,
        "position_denominator": position_denominator,
        "position_fraction": positions / position_denominator,
        "bits": bits,
        "bytes_equivalent": byte_mass,
        "stream_share_denominator_bytes": STREAM_BYTES,
        "share_of_physical_stream": byte_mass / STREAM_BYTES,
        "demand_share_denominator_bytes": DEMAND_BYTES,
        "share_of_42382B_demand": byte_mass / DEMAND_BYTES,
        "expected_positions_under_independence": expected_positions,
        "observed_to_expected_positions": checked_ratio(positions, expected_positions),
        "expected_bits_under_membership_independence": expected_bits,
        "expected_bytes_under_membership_independence": expected_bits / 8.0,
        "observed_to_expected_bits": checked_ratio(bits, expected_bits),
    }


def table_from_counts(
    counts: np.ndarray,
    bits: np.ndarray,
    *,
    position_denominator: int,
) -> dict[str, Any]:
    # Code: expensive*2 + manufactured.
    bucket_counts = {
        False: int(counts[0] + counts[1]),
        True: int(counts[2] + counts[3]),
    }
    bucket_bits = {
        False: float(bits[0] + bits[1]),
        True: float(bits[2] + bits[3]),
    }
    membership_counts = {
        False: int(counts[0] + counts[2]),
        True: int(counts[1] + counts[3]),
    }
    cells: dict[str, Any] = {}
    for expensive in (False, True):
        for manufactured in (False, True):
            code = int(expensive) * 2 + int(manufactured)
            expected_positions = bucket_counts[expensive] * membership_counts[manufactured] / position_denominator
            expected_bits = bucket_bits[expensive] * membership_counts[manufactured] / position_denominator
            label = (
                ("expensive" if expensive else "cheap")
                + "__"
                + ("render_manufactured" if manufactured else "not_render_manufactured")
            )
            cells[label] = cell_payload(
                positions=int(counts[code]),
                bits=float(bits[code]),
                expected_positions=expected_positions,
                expected_bits=expected_bits,
                position_denominator=position_denominator,
            )

    expensive_rate = checked_ratio(counts[3], bucket_counts[True])
    cheap_rate = checked_ratio(counts[1], bucket_counts[False])
    body_rate = membership_counts[True] / position_denominator
    expensive_odds = checked_ratio(counts[3], counts[2])
    cheap_odds = checked_ratio(counts[1], counts[0])
    cells["association"] = {
        "render_manufactured_denominator": membership_counts[True],
        "population_denominator": position_denominator,
        "body_render_manufactured_rate": body_rate,
        "expensive_position_denominator": bucket_counts[True],
        "render_manufactured_rate_in_expensive": expensive_rate,
        "expensive_enrichment_vs_independence": checked_ratio(float(expensive_rate), body_rate),
        "cheap_position_denominator": bucket_counts[False],
        "render_manufactured_rate_in_cheap": cheap_rate,
        "risk_ratio_expensive_vs_cheap": checked_ratio(float(expensive_rate), float(cheap_rate)),
        "odds_ratio_expensive_vs_cheap": checked_ratio(float(expensive_odds), float(cheap_odds)),
    }
    return cells


def stage_cell(positions: int, bits: float, position_denominator: int) -> dict[str, Any]:
    byte_mass = bits / 8.0
    return {
        "positions": positions,
        "position_denominator": position_denominator,
        "bits": bits,
        "bytes_equivalent": byte_mass,
        "stream_share_denominator_bytes": STREAM_BYTES,
        "share_of_physical_stream": byte_mass / STREAM_BYTES,
        "demand_share_denominator_bytes": DEMAND_BYTES,
        "share_of_42382B_demand": byte_mass / DEMAND_BYTES,
    }


def threshold_facts(bl1: dict[str, Any]) -> dict[str, dict[str, Any]]:
    curve = bl1["distribution"]["concentration"]
    by_fraction = {float(row["top_position_fraction"]): row for row in curve}
    facts: dict[str, dict[str, Any]] = {}
    for name, fraction in THRESHOLDS:
        if fraction not in by_fraction:
            raise Wj1Error(f"BL1 result lacks required threshold {fraction}")
        row = by_fraction[fraction]
        expected_positions = math.ceil(POSITIONS * fraction)
        if int(row["positions"]) != expected_positions:
            raise Wj1Error(f"BL1 threshold count drift at {fraction}")
        facts[name] = {
            "fraction": fraction,
            "positions": expected_positions,
            "threshold_bits": float(row["threshold_bits"]),
            "bl1_bits": float(row["bits"]),
            "bl1_bit_fraction": float(row["bit_fraction"]),
        }
    return facts


def load_bindings() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bl1 = json.loads(BL1_RESULT.read_text())
    mst1 = json.loads(MST1_RESULT.read_text())
    if bl1.get("schema") != "ddm_bl1_per_position_bit_allocation.v1":
        raise Wj1Error("unexpected BL1 result schema")
    if mst1.get("schema") != "ddm_mst1_manufactured_stage_split.v1":
        raise Wj1Error("unexpected MST1 result schema")
    mask_manifest = {row["name"]: row for row in mst1["mask_manifest"]}
    for name, expected_count in MASKS.items():
        if name not in mask_manifest:
            raise Wj1Error(f"MST1 mask missing: {name}")
        row = mask_manifest[name]
        if row["count_true"] != expected_count:
            raise Wj1Error(f"MST1 mask count drift: {name}")
        if row["bytes"] != POSITIONS // 8:
            raise Wj1Error(f"MST1 mask byte drift: {name}")
    return bl1, mst1, mask_manifest


def input_sources(bl1: dict[str, Any], mst1: dict[str, Any], masks: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = bl1["source_binding"]["sources"]
    result = {
        "archive": sources["archive"],
        "stream": sources["to2_stream"],
        "decoded": bl1["fields"]["decoded"],
        "cost": bl1["fields"]["frequency_cost"],
        "gt": mst1["source_fields"]["gt"],
    }
    for name in MASKS:
        result[f"mask__{name}"] = masks[name]
    return result


def verify_pins(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for name in ("archive", "stream", "decoded", "cost", "gt"):
        expected = EXPECTED[name]
        source = sources[name]
        if source["sha256"] != expected["sha256"]:
            raise Wj1Error(f"receipt pin drift for {name}")
        verified[name] = verify_file(Path(source["path"]), expected["sha256"], expected["bytes"])
    for name, expected_count in MASKS.items():
        source = sources[f"mask__{name}"]
        fact = verify_file(Path(source["path"]), source["sha256"], POSITIONS // 8)
        actual_count = 0
        for frame in range(N_FRAMES):
            actual_count += int(packed_frame(Path(source["path"]), frame).sum())
        if actual_count != expected_count:
            raise Wj1Error(f"mask population drift for {name}: {actual_count} != {expected_count}")
        fact["count_true"] = actual_count
        verified[f"mask__{name}"] = fact
    return verified


def retain_inputs(store: Path, sources: dict[str, dict[str, Any]], verified: dict[str, Any]) -> dict[str, Any]:
    retained: dict[str, Any] = {}
    retained["cost"] = copy_verified(
        Path(sources["cost"]["path"]),
        store / "retained/inputs/position_rc64_frequency_cost_bits.f64le.bin",
        EXPECTED["cost"]["sha256"],
        EXPECTED["cost"]["bytes"],
    )
    retained["gt"] = copy_verified(
        Path(sources["gt"]["path"]),
        store / "retained/inputs/gt_argmax_n600.npy",
        EXPECTED["gt"]["sha256"],
        EXPECTED["gt"]["bytes"],
    )
    for name in MASKS:
        source = sources[f"mask__{name}"]
        retained[f"mask__{name}"] = copy_verified(
            Path(source["path"]),
            store / f"retained/inputs/{name}.n600.packbits",
            source["sha256"],
            POSITIONS // 8,
        )
        retained[f"mask__{name}"]["count_true"] = verified[f"mask__{name}"]["count_true"]
    return retained


def write_checkpoint(store: Path, stage: str, payload: dict[str, Any]) -> None:
    atomic_json(
        store / "CHECKPOINT.json",
        {
            "schema": "ddm_wj1_checkpoint.v1",
            "stage": stage,
            "store": str(store),
            **payload,
        },
    )


def run_join(
    store: Path,
    bl1: dict[str, Any],
    mst1: dict[str, Any],
    retained: dict[str, Any],
) -> dict[str, Any]:
    cost_path = Path(retained["cost"]["retained"]["path"])
    gt_path = Path(retained["gt"]["retained"]["path"])
    cost = np.memmap(cost_path, dtype="<f8", mode="r", shape=SHAPE)
    gt = np.load(gt_path, mmap_mode="r", allow_pickle=False)
    if gt.shape != SHAPE or gt.dtype != np.uint8:
        raise Wj1Error("retained GT is not uint8 n600 384x512")

    total_bits = float(cost.sum(dtype=np.float64))
    if not math.isclose(total_bits, MODEL_BITS, rel_tol=0.0, abs_tol=1e-7):
        raise Wj1Error(f"BL1 cost sum drift: {total_bits} != {MODEL_BITS}")
    if int(bl1["stream"]["bits"]) != STREAM_BITS:
        raise Wj1Error("BL1 physical stream bit total drift")
    if not (0.0 <= STREAM_BITS - total_bits < 9.0):
        raise Wj1Error("modeled cost no longer reconciles to physical stream")

    state_errors = {
        "labels": MASKS["representation_error_support"],
        "native_render_head": MASKS["state_wrong_native_render_head"],
        "preuint8_roundtrip_head": MASKS["state_wrong_preuint8_roundtrip_head"],
        "uint8_roundtrip_head": MASKS["state_wrong_uint8_roundtrip_head"],
        "cpu_to_cuda_terminal_unseparated_head": MASKS["final_error_support"],
    }
    published_state_errors = mst1["state_errors"]
    if state_errors != published_state_errors:
        raise Wj1Error("MST1 stage counts do not reproduce")
    stage_deltas = {
        "native_render_head": state_errors["native_render_head"] - state_errors["labels"],
        "preuint8_roundtrip_head": state_errors["preuint8_roundtrip_head"] - state_errors["native_render_head"],
        "uint8_roundtrip_head": state_errors["uint8_roundtrip_head"] - state_errors["preuint8_roundtrip_head"],
        "cpu_to_cuda_terminal_unseparated_head": state_errors["cpu_to_cuda_terminal_unseparated_head"]
        - state_errors["uint8_roundtrip_head"],
    }
    if stage_deltas != {
        "native_render_head": 22_321,
        "preuint8_roundtrip_head": -6_980,
        "uint8_roundtrip_head": -771,
        "cpu_to_cuda_terminal_unseparated_head": 5,
    }:
        raise Wj1Error("MST1 stage deltas do not reproduce")

    thresholds = threshold_facts(bl1)
    strict_counts = {name: 0 for name, _ in THRESHOLDS}
    for frame in range(N_FRAMES):
        values = np.asarray(cost[frame]).reshape(-1)
        for name, _ in THRESHOLDS:
            strict_counts[name] += int(np.count_nonzero(values > thresholds[name]["threshold_bits"]))
    ties_remaining = {name: thresholds[name]["positions"] - strict_counts[name] for name, _ in THRESHOLDS}
    if any(value < 0 for value in ties_remaining.values()):
        raise Wj1Error("BL1 threshold strict-count exceeds exact target count")

    mask_paths = {name: store / f"retained/threshold_masks/{name}.n600.packbits" for name, _ in THRESHOLDS}
    target_mask_paths = {
        name: store / f"retained/target_masks/{name}__render_manufactured.n600.packbits" for name, _ in THRESHOLDS
    }
    temp_paths = {
        path: path.with_name(path.name + f".partial.{os.getpid()}")
        for path in (*mask_paths.values(), *target_mask_paths.values())
    }
    for path in temp_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

    global_counts = {name: np.zeros(4, dtype=np.int64) for name, _ in THRESHOLDS}
    global_bits = {name: np.zeros(4, dtype=np.float64) for name, _ in THRESHOLDS}
    class_counts = {name: np.zeros((len(CLASSES), 4), dtype=np.int64) for name, _ in THRESHOLDS}
    class_bits = {name: np.zeros((len(CLASSES), 4), dtype=np.float64) for name, _ in THRESHOLDS}
    stage_counts = {name: np.zeros(4, dtype=np.int64) for name, _ in THRESHOLDS}
    stage_bits = {name: np.zeros(4, dtype=np.float64) for name, _ in THRESHOLDS}
    threshold_selected = {name: 0 for name, _ in THRESHOLDS}
    threshold_bits_observed = {name: 0.0 for name, _ in THRESHOLDS}
    class_population = np.zeros(len(CLASSES), dtype=np.int64)
    class_total_bits = np.zeros(len(CLASSES), dtype=np.float64)
    target_columns: dict[str, list[np.ndarray]] = {
        "flat_index": [],
        "cost_bits": [],
        "gt_class": [],
        "persistent_final": [],
        **{name: [] for name, _ in THRESHOLDS},
    }

    gross_path = Path(retained["mask__gross_manufactured_native_render_head"]["retained"]["path"])
    final_manufactured_path = Path(retained["mask__final_manufactured_support"]["retained"]["path"])
    mask_handles: dict[str, Any] = {}
    target_handles: dict[str, Any] = {}
    try:
        for name, _ in THRESHOLDS:
            mask_handles[name] = temp_paths[mask_paths[name]].open("wb")
            target_handles[name] = temp_paths[target_mask_paths[name]].open("wb")
        for frame in range(N_FRAMES):
            values = np.asarray(cost[frame]).reshape(-1)
            gt_frame = np.asarray(gt[frame]).reshape(-1)
            manufactured = packed_frame(gross_path, frame)
            persistent = manufactured & packed_frame(final_manufactured_path, frame)
            class_population += np.bincount(gt_frame, minlength=len(CLASSES))
            class_total_bits += np.bincount(gt_frame, weights=values, minlength=len(CLASSES))
            expensive_masks: dict[str, np.ndarray] = {}
            for name, _ in THRESHOLDS:
                threshold = thresholds[name]["threshold_bits"]
                expensive = values > threshold
                if ties_remaining[name]:
                    equal = np.flatnonzero(values == threshold)
                    take = min(ties_remaining[name], equal.size)
                    expensive[equal[:take]] = True
                    ties_remaining[name] -= take
                expensive_masks[name] = expensive
                mask_handles[name].write(np.packbits(expensive, bitorder="little").tobytes())
                target_mask = expensive & manufactured
                target_handles[name].write(np.packbits(target_mask, bitorder="little").tobytes())
                threshold_selected[name] += int(expensive.sum())
                threshold_bits_observed[name] += float(values[expensive].sum())

                combo = expensive.astype(np.uint8) * 2 + manufactured.astype(np.uint8)
                global_counts[name] += np.bincount(combo, minlength=4)
                global_bits[name] += np.bincount(combo, weights=values, minlength=4)
                class_combo = gt_frame.astype(np.int64) * 4 + combo
                class_counts[name] += np.bincount(class_combo, minlength=len(CLASSES) * 4).reshape(len(CLASSES), 4)
                class_bits[name] += np.bincount(
                    class_combo,
                    weights=values,
                    minlength=len(CLASSES) * 4,
                ).reshape(len(CLASSES), 4)

                manufactured_expensive = expensive[manufactured].astype(np.uint8)
                persistent_manufactured = persistent[manufactured].astype(np.uint8)
                stage_combo = manufactured_expensive * 2 + persistent_manufactured
                stage_counts[name] += np.bincount(stage_combo, minlength=4)
                stage_bits[name] += np.bincount(stage_combo, weights=values[manufactured], minlength=4)

            primary = expensive_masks["top_10pct"] & manufactured
            local_indices = np.flatnonzero(primary)
            if local_indices.size:
                target_columns["flat_index"].append(local_indices.astype(np.uint64) + np.uint64(frame * PLANE))
                target_columns["cost_bits"].append(values[local_indices].astype("<f8"))
                target_columns["gt_class"].append(gt_frame[local_indices].astype(np.uint8))
                target_columns["persistent_final"].append(persistent[local_indices].astype(np.uint8))
                for name, _ in THRESHOLDS:
                    target_columns[name].append(expensive_masks[name][local_indices].astype(np.uint8))
        for handle in (*mask_handles.values(), *target_handles.values()):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        mask_handles.clear()
        target_handles.clear()
        for final_path, temp_path in temp_paths.items():
            os.replace(temp_path, final_path)
    finally:
        for handle in (*mask_handles.values(), *target_handles.values()):
            handle.close()

    for name, _ in THRESHOLDS:
        if ties_remaining[name] != 0:
            raise Wj1Error(f"threshold tie budget did not close: {name}")
        if threshold_selected[name] != thresholds[name]["positions"]:
            raise Wj1Error(f"threshold population did not close: {name}")
        if not math.isclose(
            threshold_bits_observed[name],
            thresholds[name]["bl1_bits"],
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise Wj1Error(f"threshold bit mass did not reproduce BL1: {name}")
        if int(global_counts[name].sum()) != POSITIONS:
            raise Wj1Error(f"global contingency did not close: {name}")
        if not math.isclose(float(global_bits[name].sum()), total_bits, rel_tol=0.0, abs_tol=1e-6):
            raise Wj1Error(f"global contingency bits did not close: {name}")
        if not np.array_equal(class_counts[name].sum(axis=1), class_population):
            raise Wj1Error(f"class contingency did not close: {name}")
        if int(stage_counts[name].sum()) != MASKS["gross_manufactured_native_render_head"]:
            raise Wj1Error(f"stage split did not close: {name}")
        persistent_count_all = int(stage_counts[name][1] + stage_counts[name][3])
        repaired_count_all = int(stage_counts[name][0] + stage_counts[name][2])
        if persistent_count_all != 16_917 or repaired_count_all != 11_685:
            raise Wj1Error(f"render-manufactured terminal split drift: {name}")

    if int(class_population.sum()) != POSITIONS:
        raise Wj1Error("GT class population does not close")
    if not math.isclose(float(class_total_bits.sum()), total_bits, rel_tol=0.0, abs_tol=1e-6):
        raise Wj1Error("GT class bit mass does not close")

    target_data = {key: np.concatenate(value) if value else np.empty(0) for key, value in target_columns.items()}
    target_count = target_data["flat_index"].size
    target_dtype = np.dtype(
        [
            ("flat_index", "<u8"),
            ("frame", "<u2"),
            ("y", "<u2"),
            ("x", "<u2"),
            ("cost_bits", "<f8"),
            ("gt_class", "u1"),
            ("persistent_final", "u1"),
            ("top_0p1pct", "u1"),
            ("top_1pct", "u1"),
            ("top_5pct", "u1"),
            ("top_10pct", "u1"),
        ]
    )
    target_array = np.empty(target_count, dtype=target_dtype)
    flat = target_data["flat_index"].astype(np.uint64, copy=False)
    target_array["flat_index"] = flat
    target_array["frame"] = flat // PLANE
    within_frame = flat % PLANE
    target_array["y"] = within_frame // WIDTH
    target_array["x"] = within_frame % WIDTH
    for name in (
        "cost_bits",
        "gt_class",
        "persistent_final",
        "top_0p1pct",
        "top_1pct",
        "top_5pct",
        "top_10pct",
    ):
        target_array[name] = target_data[name]
    if target_count and not np.all(target_array["flat_index"][1:] > target_array["flat_index"][:-1]):
        raise Wj1Error("target position list is not strictly raster-sorted")
    target_list_path = store / "retained/targets/top_10pct_render_manufactured_positions.npy"
    atomic_npy(target_list_path, target_array)

    global_tables = {
        name: table_from_counts(global_counts[name], global_bits[name], position_denominator=POSITIONS)
        for name, _ in THRESHOLDS
    }
    class_tables: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(CLASSES):
        row: dict[str, Any] = {
            "class_id": class_id,
            "class_name": class_name,
            "gt_lineage": "contest-CUDA DALI GT from MST1/MS9/QS3",
            "positions": int(class_population[class_id]),
            "position_denominator": POSITIONS,
            "area_fraction": float(class_population[class_id] / POSITIONS),
            "bits": float(class_total_bits[class_id]),
            "bit_denominator": total_bits,
            "thresholds": {},
        }
        for name, _ in THRESHOLDS:
            row["thresholds"][name] = table_from_counts(
                class_counts[name][class_id],
                class_bits[name][class_id],
                position_denominator=int(class_population[class_id]),
            )
        class_tables.append(row)

    stage_tables: dict[str, Any] = {}
    for name, _ in THRESHOLDS:
        counts = stage_counts[name]
        bits = stage_bits[name]
        # Code: expensive*2 + persistent. Membership is already restricted to
        # gross render-manufactured positions.
        stage_tables[name] = {
            "cheap__later_repaired": stage_cell(
                int(counts[0]), float(bits[0]), MASKS["gross_manufactured_native_render_head"]
            ),
            "cheap__terminal_persistent": stage_cell(
                int(counts[1]), float(bits[1]), MASKS["gross_manufactured_native_render_head"]
            ),
            "expensive__later_repaired": stage_cell(
                int(counts[2]), float(bits[2]), MASKS["gross_manufactured_native_render_head"]
            ),
            "expensive__terminal_persistent": stage_cell(
                int(counts[3]), float(bits[3]), MASKS["gross_manufactured_native_render_head"]
            ),
            "all__later_repaired": stage_cell(
                int(counts[0] + counts[2]),
                float(bits[0] + bits[2]),
                MASKS["gross_manufactured_native_render_head"],
            ),
            "all__terminal_persistent": stage_cell(
                int(counts[1] + counts[3]),
                float(bits[1] + bits[3]),
                MASKS["gross_manufactured_native_render_head"],
            ),
        }

    persistent_count = int(target_array["persistent_final"].sum())
    all_repaired = stage_tables["top_1pct"]["all__later_repaired"]
    all_persistent = stage_tables["top_1pct"]["all__terminal_persistent"]
    all_manufactured_bits = all_repaired["bits"] + all_persistent["bits"]
    primary_joint = global_tables["top_1pct"]["expensive__render_manufactured"]
    enrichment = global_tables["top_1pct"]["association"]["expensive_enrichment_vs_independence"]
    all_bit_enrichments = [
        global_tables[name]["expensive__render_manufactured"]["observed_to_expected_bits"] for name, _ in THRESHOLDS
    ]
    falsifier = (
        all(value is not None and value <= 1.25 for value in all_bit_enrichments)
        or primary_joint["bytes_equivalent"] < 1_000.0
    )
    prior_confirmed = enrichment is not None and enrichment >= 2.0 and primary_joint["bytes_equivalent"] > 5_000.0

    threshold_manifest = []
    target_manifest = []
    for name, _ in THRESHOLDS:
        threshold_fact = file_fact(mask_paths[name])
        threshold_fact.update(
            {
                "name": name,
                "positions": threshold_selected[name],
                "threshold_bits": thresholds[name]["threshold_bits"],
                "modeled_bits": threshold_bits_observed[name],
            }
        )
        threshold_manifest.append(threshold_fact)
        target_fact = file_fact(target_mask_paths[name])
        target_fact.update(
            {
                "name": name,
                "positions": int(global_counts[name][3]),
                "definition": "cost-threshold AND gross render-manufactured support",
            }
        )
        target_manifest.append(target_fact)

    target_list_fact = file_fact(target_list_path)
    target_list_fact.update(
        {
            "positions": target_count,
            "persistent_final_positions": persistent_count,
            "later_repaired_positions": target_count - persistent_count,
            "logical_definition": "top-10%-by-modeled-bits AND L-correct-to-native-wrong; nested threshold flags included",
            "dtype": target_array.dtype.descr,
            "sorted_by": "flat_index raster order",
        }
    )
    result = {
        "schema": "ddm_wj1_cost_error_position_join.v1",
        "status": "MEASURED_N600_SCORER_FREE_POSITION_JOIN",
        "verdict_scope": "INSTANCE:DX2_archive_976f706d_n600_BL1_cost_x_MST1_intermediate_observations",
        "axis": "[macOS-CPU advisory / scorer-free retained-field join]",
        "score_claim": False,
        "pointer_moved": False,
        "shape": list(SHAPE),
        "position_denominator": POSITIONS,
        "physical_stream": {
            "bytes": STREAM_BYTES,
            "bits": STREAM_BITS,
            "modeled_bits": total_bits,
            "modeled_bytes": total_bits / 8.0,
            "stream_minus_modeled_bits": STREAM_BITS - total_bits,
        },
        "campaign_demand_bytes": DEMAND_BYTES,
        "rate_exchange_s_per_byte": RATE_S_PER_BYTE,
        "reproduced_mst1_state_errors": state_errors,
        "reproduced_mst1_stage_deltas": stage_deltas,
        "membership_definition": {
            "render_manufactured": "decoded label L equals DALI GT G AND native-render-plus-frozen-head argmax is wrong",
            "count": MASKS["gross_manufactured_native_render_head"],
            "why_not_22321": "+22,321 is the net native stage delta (28,602 gross breaks minus 6,281 gross repairs), so it has no position-membership mask",
            "terminal_persistent_count": 16_917,
            "later_repaired_count": MASKS["gross_manufactured_native_render_head"] - 16_917,
            "modeled_bits": all_manufactured_bits,
            "modeled_bytes": all_manufactured_bits / 8.0,
            "share_of_physical_stream": (all_manufactured_bits / 8.0) / STREAM_BYTES,
            "share_of_42382B_demand": (all_manufactured_bits / 8.0) / DEMAND_BYTES,
            "complement_label": "not_render_manufactured",
            "complement_warning": "The complement is not synonymous with render-correct: it includes carried transmitted-label errors as well as native-head-correct positions.",
        },
        "thresholds": thresholds,
        "global_contingency": global_tables,
        "per_class_contingency": class_tables,
        "render_manufactured_terminal_split": stage_tables,
        "target_payloads": {
            "threshold_masks": threshold_manifest,
            "joint_target_masks": target_manifest,
            "position_list": target_list_fact,
        },
        "prior_law": {
            "top_1pct_count_enrichment": enrichment,
            "top_1pct_joint_bytes": primary_joint["bytes_equivalent"],
            "prediction_confirmed": prior_confirmed,
            "registered_falsifier_fired": falsifier,
        },
        "boundaries": {
            "byte_win_measured": False,
            "shipping_candidate_built": False,
            "distortion_consequence_predicted": False,
            "fixed_model_coarsening_parent": "LD1 measured every rung larger; WJ1 emits membership only",
            "mechanism_owner": "ddm_jf1_joint_field_model_refit",
        },
    }
    return result


def artifact_manifest(store: Path) -> list[dict[str, Any]]:
    excluded = {"MANIFEST.json", "COMPLETED_VERIFICATION.json"}
    rows = []
    for path in sorted(store.rglob("*")):
        if path.is_file() and path.name not in excluded and ".partial." not in path.name:
            rows.append(file_fact(path))
    return rows


def verify_completed(store: Path, result: dict[str, Any]) -> dict[str, Any]:
    manifest = artifact_manifest(store)
    by_path = {row["path"]: row for row in manifest}
    target = result["target_payloads"]["position_list"]
    if target["path"] not in by_path:
        raise Wj1Error("target list absent from final manifest")
    if by_path[target["path"]]["sha256"] != target["sha256"]:
        raise Wj1Error("target list hash changed before completion")
    for row in result["target_payloads"]["joint_target_masks"]:
        if row["path"] not in by_path or by_path[row["path"]]["sha256"] != row["sha256"]:
            raise Wj1Error(f"target mask changed before completion: {row['name']}")
    return {
        "schema": "ddm_wj1_completed_verification.v1",
        "status": "COMPLETE",
        "artifact_count": len(manifest),
        "artifact_bytes": sum(int(row["bytes"]) for row in manifest),
        "join_result": file_fact(store / "JOIN_RESULT.json"),
        "target_position_list": target,
        "all_manifest_hashes_recomputed": True,
        "no_volume_writes": True,
        "retention_tier": "local_disk_explicit_opt_in",
    }


def git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def self_test() -> None:
    counts = np.asarray([70, 10, 15, 5], dtype=np.int64)
    bits = np.asarray([7.0, 2.0, 30.0, 11.0], dtype=np.float64)
    table = table_from_counts(counts, bits, position_denominator=100)
    joint = table["expensive__render_manufactured"]
    assert joint["positions"] == 5
    assert math.isclose(joint["expected_positions_under_independence"], 3.0)
    assert math.isclose(joint["expected_bits_under_membership_independence"], 6.15)
    assert math.isclose(table["association"]["expensive_enrichment_vs_independence"], 5 / 3)
    assert math.isclose(table["association"]["risk_ratio_expensive_vs_cheap"], 2.0)
    print("ddm_wj1 self-test: PASS")


def run(store: Path, argv: list[str]) -> None:
    allowed_root = (ROOT / ".omx/tmp/arm_receipts_local/ddm_wj1_cost_error_position_join").resolve()
    resolved_store = store.resolve()
    if allowed_root not in (resolved_store, *resolved_store.parents):
        raise Wj1Error(f"store is outside the chartered local receipt root: {store}")
    if str(resolved_store).startswith("/Volumes/"):
        raise Wj1Error("WJ1 charter forbids volume writes")
    store.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(store).free
    required_free_bytes = 3_000_000_000
    preflight = {
        "schema": "ddm_wj1_local_storage_preflight.v1",
        "store": str(store),
        "retention_tier": "local_disk_explicit_opt_in",
        "free_bytes_before": free_bytes,
        "required_free_bytes": required_free_bytes,
        "passed": free_bytes >= required_free_bytes,
        "volume_writes_allowed": False,
    }
    atomic_json(store / "LOCAL_STORAGE_PREFLIGHT.json", preflight)
    if not preflight["passed"]:
        raise Wj1Error("local disk storage preflight failed")

    bl1, mst1, masks = load_bindings()
    sources = input_sources(bl1, mst1, masks)
    verified = verify_pins(sources)
    source_receipts = {
        "schema": "ddm_wj1_source_receipts.v1",
        "bl1_result": verify_file(BL1_RESULT, sha256_file(BL1_RESULT), BL1_RESULT.stat().st_size),
        "mst1_result": verify_file(MST1_RESULT, sha256_file(MST1_RESULT), MST1_RESULT.stat().st_size),
        "verified_sources": verified,
    }
    atomic_json(store / "SOURCE_RECEIPTS.json", source_receipts)
    write_checkpoint(store, "PINS_VERIFIED", {"source_receipts": file_fact(store / "SOURCE_RECEIPTS.json")})

    retained = retain_inputs(store, sources, verified)
    atomic_json(
        store / "RETAINED_INPUTS.json",
        {"schema": "ddm_wj1_retained_inputs.v1", "inputs": retained},
    )
    write_checkpoint(
        store,
        "INPUTS_RETAINED",
        {"retained_inputs": file_fact(store / "RETAINED_INPUTS.json")},
    )

    result = run_join(store, bl1, mst1, retained)
    result["analysis_argv"] = argv
    result["git_head_at_measurement"] = git_head()
    result["implementation"] = file_fact(Path(__file__).resolve())
    atomic_json(store / "JOIN_RESULT.json", result)
    atomic_json(
        store / "JF1_HANDOFF.json",
        {
            "schema": "ddm_wj1_jf1_handoff.v1",
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "ddm_jf1_joint_field_model_refit",
            "consumer_store": str(
                ROOT / ".omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/wj1_target_consumer"
            ),
            "fire_trigger": "WJ1 COMPLETED_VERIFICATION.json status=COMPLETE and target position-list sha256 matches this receipt",
            "target_position_list": result["target_payloads"]["position_list"],
            "mechanism_boundary": "JF1 owns coarsening with model refit; WJ1 claims no byte win",
        },
    )
    write_checkpoint(
        store,
        "JOIN_WRITTEN",
        {"join_result": file_fact(store / "JOIN_RESULT.json")},
    )

    implementation_sha = sha256_file(Path(__file__).resolve())
    source_copy = store / (f"retained/provenance/ddm_wj1_cost_error_position_join__{implementation_sha[:16]}.py")
    copy_verified(
        Path(__file__).resolve(),
        source_copy,
        implementation_sha,
        Path(__file__).stat().st_size,
    )
    manifest = artifact_manifest(store)
    atomic_json(
        store / "MANIFEST.json",
        {
            "schema": "ddm_wj1_manifest.v1",
            "artifacts": manifest,
            "artifact_count": len(manifest),
            "artifact_bytes": sum(int(row["bytes"]) for row in manifest),
        },
    )
    completed = verify_completed(store, result)
    atomic_json(store / "COMPLETED_VERIFICATION.json", completed)
    print(json.dumps(completed, indent=2, sort_keys=True))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=DEFAULT_STORE,
        help="local WJ1 receipt directory; completed stages are hash-reused",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    args = parse_args(raw)
    if args.self_test:
        self_test()
        return 0
    run(args.resume_from, [str(Path(__file__).resolve()), *raw])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
