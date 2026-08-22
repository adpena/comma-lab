#!/usr/bin/env python3
"""Measure DX2 positions coded above the five-symbol uniform cost.

This scorer-free arm consumes BL1's retained primary per-position RC64 cost
field.  It does not replay or re-instrument the shipped decoder.  It retains:

* the complete excess-over-uniform field and exact masks;
* denominator-complete class, time, group, token, and predictor-context joins;
* real-coded explicit-flag payloads, deterministic repeats, and parse-backs;
* counted static uniform-overlay member descriptors and their parse-backs.

The learned-overlay rows are model-code-length diagnostics over BL1's selected
integer-frequency probabilities.  They are not finite-precision RC64 streams,
archive candidates, or score claims.  The explicit-flag row is likewise a net
ceiling: its signalling bytes are real, while its token credit is BL1's exact
gross model-cost ceiling rather than a newly encoded token stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import os
import shutil
import struct
import time
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
VERTIGO = Path("/Volumes/VertigoDataTier/pact")
DEFAULT_OUTPUT = VERTIGO / "ddm_ae1_anti_predicted_excess" / "measurement_v1"

BL1_ROOT = VERTIGO / "ddm_bl1_per_position_bit_allocation" / "measurement_v1"
BL1_FIELD = BL1_ROOT / "retained/fields/position_rc64_frequency_cost_bits.f64le.bin"
BL1_RESULT = BL1_ROOT / "RESULT.json"
BL1_MANIFEST = BL1_ROOT / "MANIFEST.json"
TO2_ROOT = VERTIGO / "ddm_to2_token_ordering_race/measurement_v1/retained/input"
DX2_ARCHIVE = TO2_ROOT / "archive.zip"
TO2_STREAM = TO2_ROOT / "dx2_token_stream_rc64.bin"
TO2_TOKENS = TO2_ROOT / "dx2_tokens_decoded.u8"
GT_FIELD = VERTIGO / "ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy"
PREDICTOR_ARGMAX = Path("/Volumes/APDataStore/pact/ddm_fs2/retained/token_rd/argmax_field.npy")
PREDICTOR_U_INDEX = Path("/Volumes/APDataStore/pact/ddm_fs2/retained/token_rd/u_index_field.npy")
RUNTIME_ROOT = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2")
CORRECTOR_SOURCES = (
    RUNTIME_ROOT / "runtime/free_corrector.py",
    RUNTIME_ROOT / "runtime/fx1_logistic_mixer_corrector.py",
    RUNTIME_ROOT / "runtime/fx2_model_axis_corrector.py",
)

EXPECTED = {
    "archive": (180_368, "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"),
    "stream": (113_777, "e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5"),
    "tokens": (117_964_800, "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"),
    "bl1_field": (943_718_400, "99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86"),
    "bl1_result": (318_937, "f8835acf27c3b46bf95f7cd1954e08d72d591854f8f78ac6c902889a064b6621"),
    "bl1_manifest": (56_421, "0b2ca8ec51738b6e7ee5940d262be7226457fcd5a4f8e56f4bfb5b98184a59ac"),
    "gt": (117_964_928, "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"),
    "argmax": (117_964_928, "93cdf71daedd39505c5031aca7cf8524a6358fc862ce838acfbcc1cc73dcae33"),
    "u_index": (235_929_728, "74470f44a5333b27b131fcd0cf5d17fd41d82cc219d5fbd1b0557feb8825295f"),
}

N, HEIGHT, WIDTH = 600, 384, 512
PLANE = HEIGHT * WIDTH
POSITIONS = N * PLANE
GROUPS = 190
CLASSES = 5
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
FIELD_DTYPE = np.dtype("<f8")
UNIFORM_BITS = math.log2(CLASSES)
BL1_TOTAL_BITS = 910_209.2806090603
BL1_GINI = 0.9951593787014772
BL1_CONCENTRATION = (
    (0.001, 117_965, 481_962.0735017459, 0.5295068769011498, 1.9200950217101145),
    (0.01, 1_179_648, 876_748.5484900061, 0.9632384190846042, 0.04536116152491587),
)
BL1_LANE = {
    "positions": 690_754,
    "bits": 305_463.96947306144,
    "bit_fraction": 0.33559751145216,
}
DEMAND_BYTES = 42_382
STAGE_FRAMES = 100
STORAGE_REQUIRED_BYTES = 12 * (1 << 30)
ALPHA_SCALE = (1 << 16) - 1
MIX_HEADER = struct.Struct("<4sBBBBI")
MIX_MAGIC = b"AE1M"
FLAG_HEADER = struct.Struct("<4sBBHQQ")
FLAG_MAGIC = b"AE1F"
LZMA_FILTERS = [
    {
        "id": lzma.FILTER_LZMA1,
        "dict_size": 1 << 20,
        "lc": 3,
        "lp": 0,
        "pb": 2,
        "mode": lzma.MODE_NORMAL,
        "nice_len": 128,
        "mf": lzma.MF_BT4,
        "depth": 0,
    }
]


class Ae1Error(RuntimeError):
    """Fail-closed custody, retention, or measurement error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_pin(path: Path, expected: tuple[int, str]) -> dict[str, Any]:
    if not path.is_file():
        raise Ae1Error(f"required pinned input is absent: {path}")
    fact = file_fact(path)
    if (fact["bytes"], fact["sha256"]) != expected:
        raise Ae1Error(f"pinned input drifted: {path}: {fact}")
    return fact


def fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        fact = file_fact(path)
        if (fact["bytes"], fact["sha256"]) != (len(payload), digest):
            raise Ae1Error(f"refusing to overwrite differing retained payload: {path}")
        return fact
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_parent(path)
    return {"path": str(path), "bytes": len(payload), "sha256": digest}


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    return atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = np.load(path, mmap_mode="r", allow_pickle=False)
        if current.shape != array.shape or current.dtype != array.dtype or not np.array_equal(current, array):
            raise Ae1Error(f"refusing to overwrite differing retained array: {path}")
        return file_fact(path)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_parent(path)
    return file_fact(path)


def source_binding() -> dict[str, Any]:
    sources = {
        "archive": verify_pin(DX2_ARCHIVE, EXPECTED["archive"]),
        "rc64_stream": verify_pin(TO2_STREAM, EXPECTED["stream"]),
        "decoded_tokens": verify_pin(TO2_TOKENS, EXPECTED["tokens"]),
        "bl1_primary_field": verify_pin(BL1_FIELD, EXPECTED["bl1_field"]),
        "bl1_result": verify_pin(BL1_RESULT, EXPECTED["bl1_result"]),
        "bl1_manifest": verify_pin(BL1_MANIFEST, EXPECTED["bl1_manifest"]),
        "gt_field": verify_pin(GT_FIELD, EXPECTED["gt"]),
        "predictor_argmax": verify_pin(PREDICTOR_ARGMAX, EXPECTED["argmax"]),
        "predictor_u_index": verify_pin(PREDICTOR_U_INDEX, EXPECTED["u_index"]),
        "implementation": file_fact(Path(__file__)),
    }
    sources["corrector_sources"] = [file_fact(path) for path in CORRECTOR_SOURCES]
    bl1 = json.loads(BL1_RESULT.read_text())
    if bl1["positions"] != POSITIONS or bl1["shape"] != [N, HEIGHT, WIDTH]:
        raise Ae1Error("BL1 result geometry drifted")
    return {
        "schema": "ddm_ae1_source_binding.v1",
        "axis": "[macOS-CPU advisory / scorer-free exact retained-field measurement]",
        "shape": [N, HEIGHT, WIDTH],
        "positions": POSITIONS,
        "sources": sources,
    }


def storage_preflight(output: Path) -> dict[str, Any]:
    resolved = output.resolve()
    if not resolved.is_relative_to(VERTIGO.resolve()):
        raise Ae1Error(f"output must remain on the Vertigo SSD tier: {resolved}")
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    receipt = {
        "schema": "ddm_ae1_storage_preflight.v1",
        "checked_at_unix_ns": time.time_ns(),
        "tier": "VertigoDataTier",
        "path": str(resolved),
        "free_bytes": usage.free,
        "required_free_bytes": STORAGE_REQUIRED_BYTES,
        "pass": usage.free >= STORAGE_REQUIRED_BYTES,
    }
    receipt_fact = atomic_json(
        output / "preflight" / f"receipt_{receipt['checked_at_unix_ns']}.json", receipt
    )
    receipt["receipt"] = receipt_fact
    if not receipt["pass"]:
        raise Ae1Error(f"insufficient Vertigo free space: {usage.free} < {STORAGE_REQUIRED_BYTES}")
    return receipt


def group_map_and_order() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((HEIGHT, WIDTH), dtype=np.int64)
    groups = ((xx % 64) + 2 * (yy % 64)).reshape(-1)
    if groups.min() != 0 or groups.max() != GROUPS - 1:
        raise Ae1Error("HPAC group formula no longer spans 0..189")
    return groups, np.argsort(groups, kind="stable")


def pack_mask(mask: np.ndarray) -> bytes:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if flat.size % 8:
        raise Ae1Error("packed-mask geometry is not byte aligned")
    return np.packbits(flat, bitorder="little").tobytes()


def unpack_mask(payload: bytes, count: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little", count=count).astype(bool)


def stage_bounds() -> Iterable[tuple[int, int]]:
    for start in range(0, N, STAGE_FRAMES):
        yield start, min(start + STAGE_FRAMES, N)


def stage_root(output: Path, start: int, end: int) -> Path:
    return output / "retained/stages" / f"frames_{start:04d}_{end - 1:04d}"


def validate_stage(output: Path, binding: dict[str, Any], start: int, end: int) -> dict[str, Any] | None:
    receipt_path = stage_root(output, start, end) / "RECEIPT.json"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("source_binding") != binding or receipt.get("frame_range") != [start, end]:
        raise Ae1Error(f"stage binding drifted: {receipt_path}")
    for fact in receipt["artifacts"]:
        if file_fact(Path(fact["path"])) != fact:
            raise Ae1Error(f"stage artifact drifted: {fact['path']}")
    return receipt


def measure_stages(output: Path, binding: dict[str, Any]) -> list[dict[str, Any]]:
    cost = np.memmap(BL1_FIELD, dtype=FIELD_DTYPE, mode="r", shape=(N, HEIGHT, WIDTH))
    tokens = np.memmap(TO2_TOKENS, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    gt = np.load(GT_FIELD, mmap_mode="r", allow_pickle=False)
    argmax = np.load(PREDICTOR_ARGMAX, mmap_mode="r", allow_pickle=False).reshape(N, HEIGHT, WIDTH)
    u_index = np.load(PREDICTOR_U_INDEX, mmap_mode="r", allow_pickle=False).reshape(N, HEIGHT, WIDTH)
    if gt.shape != (N, HEIGHT, WIDTH) or gt.dtype != np.uint8:
        raise Ae1Error("GT field geometry or dtype drifted")
    if argmax.dtype != np.uint8 or u_index.dtype != np.uint16:
        raise Ae1Error("predictor trace dtype drifted")
    groups, _ = group_map_and_order()

    receipts: list[dict[str, Any]] = []
    for start, end in stage_bounds():
        existing = validate_stage(output, binding, start, end)
        if existing is not None:
            receipts.append(existing)
            continue
        root = stage_root(output, start, end)
        root.mkdir(parents=True, exist_ok=True)
        chunk = np.asarray(cost[start:end], dtype=np.float64)
        mask = chunk > UNIFORM_BITS
        excess = np.where(mask, chunk - UNIFORM_BITS, 0.0).astype(FIELD_DTYPE)
        artifacts = [atomic_npy(root / "excess_bits.f64.npy", excess)]
        artifacts.append(atomic_bytes(root / "overshoot.raster.packbits", pack_mask(mask)))
        for class_id, class_name in enumerate(CLASS_NAMES):
            class_mask = mask & (np.asarray(gt[start:end]) == class_id)
            artifacts.append(
                atomic_bytes(root / f"overshoot.class_{class_id}_{class_name}.packbits", pack_mask(class_mask))
            )

        flat_cost = chunk.reshape(end - start, PLANE)
        flat_mask = mask.reshape(end - start, PLANE)
        flat_excess = excess.reshape(end - start, PLANE)
        flat_gt = np.asarray(gt[start:end]).reshape(end - start, PLANE)
        flat_tokens = np.asarray(tokens[start:end]).reshape(end - start, PLANE)
        flat_argmax = np.asarray(argmax[start:end]).reshape(end - start, PLANE)
        flat_u64 = np.minimum(
            np.asarray(u_index[start:end], dtype=np.int64).reshape(end - start, PLANE) // 4, 63
        )

        context = (
            groups[None, :] * (CLASSES * 64)
            + flat_argmax.astype(np.int64) * 64
            + flat_u64
        )
        context_domain = GROUPS * CLASSES * 64
        context_count = np.bincount(context.reshape(-1), minlength=context_domain)
        context_over = np.bincount(context[flat_mask], minlength=context_domain)
        context_bits = np.bincount(context[flat_mask], weights=flat_cost[flat_mask], minlength=context_domain)
        context_excess = np.bincount(context[flat_mask], weights=flat_excess[flat_mask], minlength=context_domain)
        context_arrays = np.stack((context_count, context_over), axis=1).astype(np.uint64)
        context_values = np.stack((context_bits, context_excess), axis=1).astype(np.float64)
        artifacts.append(atomic_npy(root / "predictor_context_counts.u64.npy", context_arrays))
        artifacts.append(atomic_npy(root / "predictor_context_bits.f64.npy", context_values))

        frame_rows = []
        for local, frame in enumerate(range(start, end)):
            active = flat_mask[local]
            frame_rows.append(
                {
                    "frame": frame,
                    "positions": PLANE,
                    "overshoot_positions": int(active.sum()),
                    "overshoot_bits": float(flat_cost[local][active].sum()),
                    "gross_excess_bits": float(flat_excess[local].sum()),
                }
            )
        stage = {
            "schema": "ddm_ae1_stage.v1",
            "source_binding": binding,
            "frame_range": [start, end],
            "positions": (end - start) * PLANE,
            "overshoot_positions": int(mask.sum()),
            "overshoot_bits": float(chunk[mask].sum()),
            "gross_excess_bits": float(excess.sum()),
            "frame_rows": frame_rows,
            "class_rows": [
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "positions": int((flat_gt == class_id).sum()),
                    "overshoot_positions": int((flat_mask & (flat_gt == class_id)).sum()),
                    "overshoot_bits": float(flat_cost[flat_mask & (flat_gt == class_id)].sum()),
                    "gross_excess_bits": float(flat_excess[flat_gt == class_id].sum()),
                }
                for class_id in range(CLASSES)
            ],
            "token_rows": [
                {
                    "token": symbol,
                    "positions": int((flat_tokens == symbol).sum()),
                    "overshoot_positions": int((flat_mask & (flat_tokens == symbol)).sum()),
                    "overshoot_bits": float(flat_cost[flat_mask & (flat_tokens == symbol)].sum()),
                    "gross_excess_bits": float(flat_excess[flat_tokens == symbol].sum()),
                }
                for symbol in range(CLASSES)
            ],
            "prediction_rows": [
                {
                    "relation": name,
                    "positions": int(select.sum()),
                    "overshoot_positions": int((flat_mask & select).sum()),
                    "overshoot_bits": float(flat_cost[flat_mask & select].sum()),
                    "gross_excess_bits": float(flat_excess[select].sum()),
                }
                for name, select in (
                    ("hit_token_equals_predictor_argmax", flat_tokens == flat_argmax),
                    ("miss_token_differs_from_predictor_argmax", flat_tokens != flat_argmax),
                )
            ],
            "artifacts": artifacts,
        }
        atomic_json(root / "RECEIPT.json", stage)
        receipts.append(stage)
    return receipts


def concatenate_stage_payloads(output: Path, name: str, destination: Path) -> dict[str, Any]:
    sources = [stage_root(output, start, end) / name for start, end in stage_bounds()]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    if destination.exists():
        expected_digest = hashlib.sha256()
        expected_bytes = 0
        for source in sources:
            with source.open("rb") as incoming:
                for chunk in iter(lambda: incoming.read(8 << 20), b""):
                    expected_digest.update(chunk)
                    expected_bytes += len(chunk)
        fact = file_fact(destination)
        if (fact["bytes"], fact["sha256"]) != (expected_bytes, expected_digest.hexdigest()):
            raise Ae1Error(f"assembled retained payload drifted: {destination}")
        return fact
    with temporary.open("wb") as out:
        for source in sources:
            with source.open("rb") as incoming:
                shutil.copyfileobj(incoming, out, length=8 << 20)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, destination)
    fsync_parent(destination)
    return file_fact(destination)


def assemble_fields(output: Path) -> dict[str, Any]:
    fields = output / "retained/fields"
    fields.mkdir(parents=True, exist_ok=True)
    excess_destination = fields / "excess_over_uniform_bits.f64le.bin"
    if not excess_destination.exists():
        temporary = excess_destination.with_name(f".{excess_destination.name}.partial.{os.getpid()}")
        with temporary.open("wb") as out:
            for start, end in stage_bounds():
                stage = np.load(stage_root(output, start, end) / "excess_bits.f64.npy", mmap_mode="r")
                out.write(np.ascontiguousarray(stage).tobytes())
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, excess_destination)
        fsync_parent(excess_destination)
    if excess_destination.stat().st_size != POSITIONS * FIELD_DTYPE.itemsize:
        raise Ae1Error("assembled excess field has wrong size")
    expected_digest = hashlib.sha256()
    for start, end in stage_bounds():
        stage_path = stage_root(output, start, end) / "excess_bits.f64.npy"
        stage = np.load(stage_path, mmap_mode="r", allow_pickle=False)
        for frame in range(stage.shape[0]):
            expected_digest.update(np.ascontiguousarray(stage[frame]).tobytes())
    if sha256_file(excess_destination) != expected_digest.hexdigest():
        raise Ae1Error("assembled excess field differs from retained stages")

    result: dict[str, Any] = {"excess_field": file_fact(excess_destination)}
    result["overshoot_mask"] = concatenate_stage_payloads(
        output, "overshoot.raster.packbits", fields / "overshoot_gt_log2_5.raster.packbits"
    )
    result["class_masks"] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        result["class_masks"].append(
            concatenate_stage_payloads(
                output,
                f"overshoot.class_{class_id}_{class_name}.packbits",
                fields / f"overshoot.class_{class_id}_{class_name}.n600.packbits",
            )
        )
    return result


def reproduce_bl1(output: Path) -> dict[str, Any]:
    sorted_path = output / "retained/fields/bl1_cost_sorted_ascending.f64le.bin"
    reused_from: dict[str, Any] | None = None
    if not sorted_path.exists():
        sorted_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = sorted_path.with_name(f".{sorted_path.name}.partial.{os.getpid()}")
        reusable = None
        for candidate in sorted(output.parent.glob("measurement_v*/BL1_REPRODUCTION_GATE.json")):
            candidate_result = json.loads(candidate.read_text())
            candidate_sorted = Path(candidate_result.get("sorted_field", {}).get("path", ""))
            if (
                candidate_result.get("status") == "PASS_BEFORE_NEW_MEASUREMENT"
                and candidate_sorted.is_file()
                and file_fact(candidate_sorted) == candidate_result["sorted_field"]
            ):
                reusable = candidate_sorted
                break
        if reusable is None:
            shutil.copyfile(BL1_FIELD, temporary)
            sorted_cost = np.memmap(
                temporary, dtype=FIELD_DTYPE, mode="r+", shape=(POSITIONS,)
            )
            sorted_cost.sort()
            sorted_cost.flush()
            del sorted_cost
        else:
            shutil.copyfile(reusable, temporary)
            reused_from = file_fact(reusable)
        os.replace(temporary, sorted_path)
        fsync_parent(sorted_path)
    sorted_cost = np.memmap(sorted_path, dtype=FIELD_DTYPE, mode="r", shape=(POSITIONS,))
    total = float(sorted_cost.sum(dtype=np.float64))
    if not math.isclose(total, BL1_TOTAL_BITS, rel_tol=0.0, abs_tol=1e-8):
        raise Ae1Error(f"BL1 total does not reproduce: {total}")
    rows = []
    for fraction, count, expected_bits, expected_fraction, expected_threshold in BL1_CONCENTRATION:
        bits = float(sorted_cost[-count:].sum(dtype=np.float64))
        threshold = float(sorted_cost[-count])
        bit_fraction = bits / total
        if (
            not math.isclose(bits, expected_bits, rel_tol=0.0, abs_tol=1e-8)
            or not math.isclose(bit_fraction, expected_fraction, rel_tol=0.0, abs_tol=1e-14)
            or threshold != expected_threshold
        ):
            raise Ae1Error(f"BL1 concentration row does not reproduce: top={fraction}")
        rows.append(
            {
                "top_fraction": fraction,
                "positions": count,
                "bits": bits,
                "bit_fraction": bit_fraction,
                "threshold_bits": threshold,
            }
        )
    weighted_sum = 0.0
    chunk_positions = 1 << 20
    for start in range(0, POSITIONS, chunk_positions):
        end = min(start + chunk_positions, POSITIONS)
        indexes = np.arange(start + 1, end + 1, dtype=np.float64)
        weighted_sum += float(np.dot(indexes, sorted_cost[start:end]))
    gini = float((2.0 * weighted_sum / (POSITIONS * total)) - (POSITIONS + 1) / POSITIONS)
    if not math.isclose(gini, BL1_GINI, rel_tol=0.0, abs_tol=1e-14):
        raise Ae1Error(f"BL1 Gini does not reproduce: {gini}")

    cost = np.memmap(BL1_FIELD, dtype=FIELD_DTYPE, mode="r", shape=(N, HEIGHT, WIDTH))
    gt = np.load(GT_FIELD, mmap_mode="r", allow_pickle=False)
    class_positions = np.zeros(CLASSES, dtype=np.int64)
    class_bits = np.zeros(CLASSES, dtype=np.float64)
    for start, end in stage_bounds():
        stage_gt = np.asarray(gt[start:end]).reshape(-1)
        stage_cost = np.asarray(cost[start:end]).reshape(-1)
        class_positions += np.bincount(stage_gt, minlength=CLASSES)
        class_bits += np.bincount(stage_gt, weights=stage_cost, minlength=CLASSES)
    lane = {
        "class_id": 1,
        "class_name": "Lane",
        "positions": int(class_positions[1]),
        "bits": float(class_bits[1]),
        "bit_fraction": float(class_bits[1] / total),
    }
    if (
        lane["positions"] != BL1_LANE["positions"]
        or not math.isclose(lane["bits"], BL1_LANE["bits"], rel_tol=0.0, abs_tol=1e-8)
        or not math.isclose(lane["bit_fraction"], BL1_LANE["bit_fraction"], rel_tol=0.0, abs_tol=1e-14)
    ):
        raise Ae1Error(f"BL1 Lane row does not reproduce: {lane}")
    return {
        "status": "PASS_BEFORE_NEW_MEASUREMENT",
        "total_bits": total,
        "gini": gini,
        "concentration": rows,
        "lane_row": lane,
        "sorted_field": file_fact(sorted_path),
        "sorted_field_reused_from": reused_from,
    }


def aggregate_stage_rows(stages: list[dict[str, Any]], key: str, identities: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label, identity in identities:
        selected = []
        for stage in stages:
            match = [row for row in stage[key] if row.get(label) == identity]
            if len(match) != 1:
                raise Ae1Error(f"stage {key} identity missing: {label}={identity}")
            selected.append(match[0])
        row = {label: identity}
        for field in ("positions", "overshoot_positions", "overshoot_bits", "gross_excess_bits"):
            row[field] = sum(item[field] for item in selected)
        rows.append(row)
    return rows


def group_rows(cost: np.memmap, mask: np.ndarray, excess: np.memmap) -> list[dict[str, Any]]:
    groups, _ = group_map_and_order()
    counts = np.bincount(groups, minlength=GROUPS).astype(np.int64) * N
    overshoot_count = np.zeros(GROUPS, dtype=np.int64)
    overshoot_bits = np.zeros(GROUPS, dtype=np.float64)
    excess_bits = np.zeros(GROUPS, dtype=np.float64)
    for frame in range(N):
        active = mask[frame].reshape(-1)
        frame_cost = np.asarray(cost[frame]).reshape(-1)
        frame_excess = np.asarray(excess[frame]).reshape(-1)
        overshoot_count += np.bincount(groups[active], minlength=GROUPS)
        overshoot_bits += np.bincount(groups[active], weights=frame_cost[active], minlength=GROUPS)
        excess_bits += np.bincount(groups, weights=frame_excess, minlength=GROUPS)
    return [
        {
            "group": group,
            "positions": int(counts[group]),
            "overshoot_positions": int(overshoot_count[group]),
            "overshoot_bits": float(overshoot_bits[group]),
            "gross_excess_bits": float(excess_bits[group]),
        }
        for group in range(GROUPS)
    ]


def encode_uleb(value: int) -> bytes:
    if value < 0:
        raise Ae1Error("ULEB cannot encode a negative value")
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(output)


def decode_uleb(payload: bytes, cursor: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if cursor >= len(payload) or shift > 63:
            raise Ae1Error("invalid ULEB flag stream")
        byte = payload[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, cursor
        shift += 7


def serialize_flags(mask: np.ndarray, representation: str, order: np.ndarray) -> bytes:
    raster = np.asarray(mask, dtype=bool).reshape(N, PLANE)
    event = raster[:, order].reshape(-1)
    if representation == "event_packbits":
        body = pack_mask(event)
        rep_id = 1
    elif representation == "event_delta_uleb":
        positions = np.flatnonzero(event)
        body_array = bytearray()
        previous = -1
        for position in positions.tolist():
            body_array.extend(encode_uleb(int(position) - previous - 1))
            previous = int(position)
        body = bytes(body_array)
        rep_id = 2
    else:
        raise Ae1Error(f"unknown flag representation: {representation}")
    return FLAG_HEADER.pack(FLAG_MAGIC, 1, rep_id, 0, POSITIONS, int(event.sum())) + body


def parse_flags(payload: bytes, order: np.ndarray) -> np.ndarray:
    if len(payload) < FLAG_HEADER.size:
        raise Ae1Error("truncated flag payload")
    magic, version, rep_id, reserved, positions, active_count = FLAG_HEADER.unpack_from(payload)
    if (magic, version, reserved, positions) != (FLAG_MAGIC, 1, 0, POSITIONS):
        raise Ae1Error("invalid flag header")
    body = payload[FLAG_HEADER.size :]
    if rep_id == 1:
        event = unpack_mask(body, POSITIONS)
    elif rep_id == 2:
        event = np.zeros(POSITIONS, dtype=bool)
        cursor = 0
        previous = -1
        for _ in range(active_count):
            gap, cursor = decode_uleb(body, cursor)
            position = previous + 1 + gap
            if position >= POSITIONS or event[position]:
                raise Ae1Error("invalid delta flag position")
            event[position] = True
            previous = position
        if cursor != len(body):
            raise Ae1Error("trailing bytes in delta flag stream")
    else:
        raise Ae1Error("unknown flag representation id")
    if int(event.sum()) != active_count:
        raise Ae1Error("flag active-count mismatch")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(PLANE)
    return event.reshape(N, PLANE)[:, inverse].reshape(N, HEIGHT, WIDTH)


def code_payload(raw: bytes, coder: str) -> bytes:
    if coder == "raw":
        return raw
    if coder == "brotli_q11":
        return bytes(brotli.compress(raw, quality=11))
    if coder == "zlib9":
        return zlib.compress(raw, level=9)
    if coder == "lzma1_1m":
        return lzma.compress(raw, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    raise Ae1Error(f"unknown coder: {coder}")


def decode_payload(payload: bytes, coder: str) -> bytes:
    if coder == "raw":
        return payload
    if coder == "brotli_q11":
        return bytes(brotli.decompress(payload))
    if coder == "zlib9":
        return zlib.decompress(payload)
    if coder == "lzma1_1m":
        return lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    raise Ae1Error(f"unknown coder: {coder}")


def price_flags(output: Path, mask: np.ndarray) -> dict[str, Any]:
    _, order = group_map_and_order()
    root = output / "retained/signalling"
    rows = []
    for representation in ("event_packbits", "event_delta_uleb"):
        raw = serialize_flags(mask, representation, order)
        if not np.array_equal(parse_flags(raw, order), mask):
            raise Ae1Error(f"raw flag parse-back failed: {representation}")
        raw_fact = atomic_bytes(root / representation / "flags.ae1f.raw", raw)
        variants = []
        for coder in ("raw", "brotli_q11", "zlib9", "lzma1_1m"):
            payload = code_payload(raw, coder)
            repeat = code_payload(raw, coder)
            if payload != repeat or decode_payload(payload, coder) != raw:
                raise Ae1Error(f"flag coder failed deterministic inverse: {representation}/{coder}")
            if not np.array_equal(parse_flags(decode_payload(payload, coder), order), mask):
                raise Ae1Error(f"coded flag parse-back failed: {representation}/{coder}")
            fact = atomic_bytes(root / representation / f"flags.{coder}.bin", payload)
            repeat_fact = atomic_bytes(root / representation / f"flags.{coder}.repeat.bin", repeat)
            variants.append({"coder": coder, "payload": fact, "repeat": repeat_fact})
        rows.append({"representation": representation, "raw": raw_fact, "variants": variants})
    candidates = [
        {"representation": row["representation"], **variant}
        for row in rows
        for variant in row["variants"]
    ]
    winner = min(candidates, key=lambda item: int(item["payload"]["bytes"]))
    return {"rows": rows, "winner": winner}


def alpha_derivatives(cost: np.memmap, group_alpha: np.ndarray, global_alpha: float, groups: np.ndarray) -> tuple[float, np.ndarray]:
    global_derivative = 0.0
    group_derivative = np.zeros(GROUPS, dtype=np.float64)
    delta_group = 0.2
    for start, end in stage_bounds():
        p = np.exp2(-np.asarray(cost[start:end], dtype=np.float64).reshape(-1))
        delta = delta_group - p
        q_global = p + global_alpha * delta
        global_derivative += float((-delta / q_global).sum(dtype=np.float64))
        repeated_groups = np.tile(groups, end - start)
        q_group = p + group_alpha[repeated_groups] * delta
        group_derivative += np.bincount(
            repeated_groups, weights=-delta / q_group, minlength=GROUPS
        )
    return global_derivative / math.log(2.0), group_derivative / math.log(2.0)


def optimize_alphas(output: Path, cost: np.memmap, groups: np.ndarray) -> tuple[float, np.ndarray]:
    checkpoint_root = output / "retained/member_descriptors/optimizer"
    checkpoints = sorted(checkpoint_root.glob("bisection_iter_*.json"))
    if checkpoints:
        state = json.loads(checkpoints[-1].read_text())
        completed_iterations = int(state["completed_iterations"])
        global_lo = float(state["global_lo"])
        global_hi = float(state["global_hi"])
        global_active = bool(state["global_active"])
        group_lo = np.asarray(state["group_lo"], dtype=np.float64)
        group_hi = np.asarray(state["group_hi"], dtype=np.float64)
        group_active = np.asarray(state["group_active"], dtype=bool)
        if group_lo.shape != (GROUPS,) or group_hi.shape != (GROUPS,) or group_active.shape != (GROUPS,):
            raise Ae1Error("uniform-member optimizer checkpoint geometry drifted")
    else:
        completed_iterations = 0
        global_lo, global_hi = 0.0, 1.0
        group_lo = np.zeros(GROUPS, dtype=np.float64)
        group_hi = np.ones(GROUPS, dtype=np.float64)
        global_derivative_at_zero, derivative_at_zero = alpha_derivatives(
            cost, group_lo, 0.0, groups
        )
        group_active = derivative_at_zero < 0.0
        global_active = global_derivative_at_zero < 0.0
        atomic_json(
            checkpoint_root / "bisection_iter_000.json",
            {
                "schema": "ddm_ae1_uniform_member_optimizer_checkpoint.v1",
                "completed_iterations": 0,
                "global_lo": global_lo,
                "global_hi": global_hi,
                "global_active": global_active,
                "group_lo": group_lo.tolist(),
                "group_hi": group_hi.tolist(),
                "group_active": group_active.tolist(),
            },
        )
    for iteration in range(completed_iterations, 24):
        global_mid = (global_lo + global_hi) / 2.0
        group_mid = (group_lo + group_hi) / 2.0
        global_derivative, group_derivative = alpha_derivatives(cost, group_mid, global_mid, groups)
        if global_active:
            if global_derivative < 0.0:
                global_lo = global_mid
            else:
                global_hi = global_mid
        group_lo = np.where(group_active & (group_derivative < 0.0), group_mid, group_lo)
        group_hi = np.where(group_active & (group_derivative >= 0.0), group_mid, group_hi)
        atomic_json(
            checkpoint_root / f"bisection_iter_{iteration + 1:03d}.json",
            {
                "schema": "ddm_ae1_uniform_member_optimizer_checkpoint.v1",
                "completed_iterations": iteration + 1,
                "global_lo": global_lo,
                "global_hi": global_hi,
                "global_active": global_active,
                "group_lo": group_lo.tolist(),
                "group_hi": group_hi.tolist(),
                "group_active": group_active.tolist(),
            },
        )
    global_alpha = (global_lo + global_hi) / 2.0 if global_active else 0.0
    group_alpha = np.where(group_active, (group_lo + group_hi) / 2.0, 0.0)
    return global_alpha, group_alpha


def mixture_costs(cost: np.memmap, global_codes: np.ndarray, group_codes: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    global_totals = np.zeros(global_codes.size, dtype=np.float64)
    group_totals = np.zeros((2, GROUPS), dtype=np.float64)
    global_alpha = global_codes.astype(np.float64) / ALPHA_SCALE
    group_alpha = group_codes.astype(np.float64) / ALPHA_SCALE
    for start, end in stage_bounds():
        p = np.exp2(-np.asarray(cost[start:end], dtype=np.float64).reshape(-1))
        delta = 0.2 - p
        for index, alpha in enumerate(global_alpha):
            global_totals[index] += float((-np.log2(p + alpha * delta)).sum(dtype=np.float64))
        repeated_groups = np.tile(groups, end - start)
        for choice in range(2):
            values = -np.log2(p + group_alpha[choice, repeated_groups] * delta)
            group_totals[choice] += np.bincount(repeated_groups, weights=values, minlength=GROUPS)
    return global_totals, group_totals


def serialize_member(kind: str, codes: np.ndarray) -> bytes:
    if kind == "global":
        kind_id, contexts = 1, 1
    elif kind == "group190":
        kind_id, contexts = 2, GROUPS
    else:
        raise Ae1Error(f"unknown member kind: {kind}")
    values = np.asarray(codes, dtype="<u2").reshape(-1)
    if values.size != contexts:
        raise Ae1Error("member descriptor context count mismatch")
    return MIX_HEADER.pack(MIX_MAGIC, 1, kind_id, 16, 0, contexts) + values.tobytes()


def parse_member(payload: bytes) -> tuple[str, np.ndarray]:
    if len(payload) < MIX_HEADER.size:
        raise Ae1Error("truncated member descriptor")
    magic, version, kind_id, bits, reserved, contexts = MIX_HEADER.unpack_from(payload)
    if (magic, version, bits, reserved) != (MIX_MAGIC, 1, 16, 0):
        raise Ae1Error("invalid member descriptor")
    kind = {1: "global", 2: "group190"}.get(kind_id)
    expected = 1 if kind == "global" else GROUPS if kind == "group190" else -1
    if contexts != expected or len(payload) != MIX_HEADER.size + 2 * contexts:
        raise Ae1Error("member descriptor length mismatch")
    return kind, np.frombuffer(payload[MIX_HEADER.size :], dtype="<u2").copy()


def price_members(output: Path, cost: np.memmap) -> dict[str, Any]:
    groups, _ = group_map_and_order()
    global_alpha, group_alpha = optimize_alphas(output, cost, groups)
    global_center = round(global_alpha * ALPHA_SCALE)
    global_codes = np.unique(
        np.clip(np.arange(global_center - 2, global_center + 3), 0, ALPHA_SCALE)
    ).astype(np.uint16)
    group_scaled = group_alpha * ALPHA_SCALE
    group_candidates = np.stack((np.floor(group_scaled), np.ceil(group_scaled))).astype(np.uint16)
    global_costs, group_costs = mixture_costs(cost, global_codes, group_candidates, groups)
    global_code = np.asarray([global_codes[int(np.argmin(global_costs))]], dtype=np.uint16)
    group_choice = np.argmin(group_costs, axis=0)
    group_code = group_candidates[group_choice, np.arange(GROUPS)].astype(np.uint16)
    group_total = float(group_costs[group_choice, np.arange(GROUPS)].sum())
    global_total = float(global_costs[int(np.argmin(global_costs))])

    root = output / "retained/member_descriptors"
    rows = []
    for kind, codes, model_bits in (
        ("global", global_code, global_total),
        ("group190", group_code, group_total),
    ):
        raw = serialize_member(kind, codes)
        parsed_kind, parsed_codes = parse_member(raw)
        if parsed_kind != kind or not np.array_equal(parsed_codes, codes):
            raise Ae1Error(f"member descriptor parse-back failed: {kind}")
        raw_fact = atomic_bytes(root / kind / "member.ae1m.raw", raw)
        variants = []
        for coder in ("raw", "brotli_q11", "zlib9", "lzma1_1m"):
            payload = code_payload(raw, coder)
            repeat = code_payload(raw, coder)
            if payload != repeat or decode_payload(payload, coder) != raw:
                raise Ae1Error(f"member coder failed deterministic inverse: {kind}/{coder}")
            parsed = parse_member(decode_payload(payload, coder))
            if parsed[0] != kind or not np.array_equal(parsed[1], codes):
                raise Ae1Error(f"coded member parse-back failed: {kind}/{coder}")
            fact = atomic_bytes(root / kind / f"member.{coder}.bin", payload)
            repeat_fact = atomic_bytes(root / kind / f"member.{coder}.repeat.bin", repeat)
            variants.append({"coder": coder, "payload": fact, "repeat": repeat_fact})
        winner = min(variants, key=lambda item: int(item["payload"]["bytes"]))
        gain_bits = BL1_TOTAL_BITS - model_bits
        rows.append(
            {
                "kind": kind,
                "contexts": int(codes.size),
                "alpha_code_scale": ALPHA_SCALE,
                "alpha_codes": codes.tolist(),
                "modelled_bits": model_bits,
                "modelled_gain_bits": gain_bits,
                "modelled_gain_bytes": gain_bits / 8.0,
                "description": {"raw": raw_fact, "variants": variants, "winner": winner},
                "modelled_net_bytes_after_description": gain_bits / 8.0 - int(winner["payload"]["bytes"]),
                "authority_boundary": (
                    "selected-probability model code length only; no finite-precision RC64 stream, "
                    "receiver integration, archive, or score claim"
                ),
            }
        )
    return {
        "rows": rows,
        "online_uniform_member_stored_growth_bytes": 0,
        "online_uniform_member_realized_gain": "UNKNOWN_NOT_BUILT_OR_REPLAYED",
        "online_member_reason": (
            "the incumbent mixer weights are fixed-initialized and updated causally from already-decoded symbols; "
            "generic code and regenerated state are rule-118-free, but this arm does not build the mechanism"
        ),
    }


def context_rows(output: Path) -> dict[str, Any]:
    counts = np.zeros((GROUPS * CLASSES * 64, 2), dtype=np.uint64)
    values = np.zeros((GROUPS * CLASSES * 64, 2), dtype=np.float64)
    for start, end in stage_bounds():
        root = stage_root(output, start, end)
        counts += np.load(root / "predictor_context_counts.u64.npy", allow_pickle=False)
        values += np.load(root / "predictor_context_bits.f64.npy", allow_pickle=False)
    active = np.flatnonzero(counts[:, 0])
    rows = []
    for context in active.tolist():
        group, remainder = divmod(context, CLASSES * 64)
        predicted, u64 = divmod(remainder, 64)
        rows.append(
            {
                "context": context,
                "group": group,
                "predicted_class": predicted,
                "surprise_bin_u64": u64,
                "positions": int(counts[context, 0]),
                "overshoot_positions": int(counts[context, 1]),
                "overshoot_bits": float(values[context, 0]),
                "gross_excess_bits": float(values[context, 1]),
            }
        )
    retained = output / "retained/aggregates"
    return {
        "active_contexts": len(rows),
        "domain_contexts": GROUPS * CLASSES * 64,
        "definition": "group190 x FS2 predictor argmax x 64 half-bit confidence/surprise bins",
        "counts": atomic_npy(retained / "predictor_context_counts.u64.npy", counts),
        "bits": atomic_npy(retained / "predictor_context_bits.f64.npy", values),
        "rows": rows,
    }


def write_manifest(output: Path, result: dict[str, Any]) -> dict[str, Any]:
    result_fact = file_fact(output / "RESULT.json")
    artifacts = []
    for path in sorted((output / "retained").rglob("*")):
        if path.is_file():
            artifacts.append(file_fact(path))
    manifest = {
        "schema": "ddm_ae1_retention_manifest.v1",
        "tier": "VertigoDataTier",
        "root": str(output),
        "source_binding": result["source_binding"],
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(int(item["bytes"]) for item in artifacts),
        "result": result_fact,
        "retention": "No listed artifact may be deleted or moved without a replacement custody manifest.",
    }
    atomic_json(output / "MANIFEST.json", manifest)
    return manifest


def seal_failed_output(output: Path, reason: str) -> dict[str, Any]:
    if not output.is_dir():
        raise Ae1Error(f"failed output root is absent: {output}")
    source_path = output / "SOURCE_BINDING.json"
    receipt = {
        "schema": "ddm_ae1_failed_run_receipt.v1",
        "status": "FAILED_RETAINED_NOT_A_MEASUREMENT_VERDICT",
        "reason": reason,
        "source_binding": json.loads(source_path.read_text()) if source_path.is_file() else None,
    }
    receipt_fact = atomic_json(output / "FAILURE_RECEIPT.json", receipt)
    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "FAILURE_MANIFEST.json":
            artifacts.append(file_fact(path))
    manifest = {
        "schema": "ddm_ae1_failed_run_manifest.v1",
        "status": receipt["status"],
        "root": str(output),
        "receipt": receipt_fact,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(int(item["bytes"]) for item in artifacts),
        "retention": "Retained failed-run bytes; no measurement verdict may cite this root.",
    }
    manifest_fact = atomic_json(output / "FAILURE_MANIFEST.json", manifest)
    return {"receipt": receipt_fact, "manifest": manifest_fact, **manifest}


def self_test() -> None:
    _, order = group_map_and_order()
    mask = np.zeros((N, HEIGHT, WIDTH), dtype=bool)
    mask[0, 0, 0] = True
    mask[0, 1, 1] = True
    mask[-1, -1, -1] = True
    for representation in ("event_packbits", "event_delta_uleb"):
        payload = serialize_flags(mask, representation, order)
        if not np.array_equal(parse_flags(payload, order), mask):
            raise Ae1Error(f"flag self-test failed: {representation}")
    for kind, codes in (
        ("global", np.asarray([17], dtype=np.uint16)),
        ("group190", np.arange(GROUPS, dtype=np.uint16)),
    ):
        parsed_kind, parsed_codes = parse_member(serialize_member(kind, codes))
        if parsed_kind != kind or not np.array_equal(parsed_codes, codes):
            raise Ae1Error(f"member self-test failed: {kind}")
    raw = b"AE1 deterministic coder test" * 100
    for coder in ("raw", "brotli_q11", "zlib9", "lzma1_1m"):
        encoded = code_payload(raw, coder)
        if encoded != code_payload(raw, coder) or decode_payload(encoded, coder) != raw:
            raise Ae1Error(f"coder self-test failed: {coder}")


def run(output: Path) -> dict[str, Any]:
    started = time.time()
    self_test()
    preflight = storage_preflight(output)
    binding = source_binding()
    atomic_json(output / "SOURCE_BINDING.json", binding)
    reproduction = reproduce_bl1(output)
    atomic_json(output / "BL1_REPRODUCTION_GATE.json", reproduction)
    stages = measure_stages(output, binding)
    fields = assemble_fields(output)

    cost = np.memmap(BL1_FIELD, dtype=FIELD_DTYPE, mode="r", shape=(N, HEIGHT, WIDTH))
    excess = np.memmap(
        fields["excess_field"]["path"], dtype=FIELD_DTYPE, mode="r", shape=(N, HEIGHT, WIDTH)
    )
    mask_payload = Path(fields["overshoot_mask"]["path"]).read_bytes()
    mask = unpack_mask(mask_payload, POSITIONS).reshape(N, HEIGHT, WIDTH)
    overshoot_positions = sum(int(stage["overshoot_positions"]) for stage in stages)
    overshoot_bits = sum(float(stage["overshoot_bits"]) for stage in stages)
    gross_excess_bits = sum(float(stage["gross_excess_bits"]) for stage in stages)
    if overshoot_positions != int(mask.sum()) or not math.isclose(
        gross_excess_bits, float(excess.sum(dtype=np.float64)), rel_tol=0.0, abs_tol=1e-8
    ):
        raise Ae1Error("stage aggregates do not reconcile with assembled fields")

    class_rows = aggregate_stage_rows(
        stages, "class_rows", [("class_id", class_id) for class_id in range(CLASSES)]
    )
    for row in class_rows:
        row["class_name"] = CLASS_NAMES[int(row["class_id"])]
    token_rows = aggregate_stage_rows(
        stages, "token_rows", [("token", token) for token in range(CLASSES)]
    )
    prediction_rows = aggregate_stage_rows(
        stages,
        "prediction_rows",
        [
            ("relation", "hit_token_equals_predictor_argmax"),
            ("relation", "miss_token_differs_from_predictor_argmax"),
        ],
    )
    frames = [row for stage in stages for row in stage["frame_rows"]]
    groups = group_rows(cost, mask, excess)
    contexts = context_rows(output)
    signalling = price_flags(output, mask)
    members = price_members(output, cost)
    signal_bytes = int(signalling["winner"]["payload"]["bytes"])

    result = {
        "schema": "ddm_ae1_anti_predicted_excess.v1",
        "status": "MEASURED_GROSS_AND_PRICED_NET_CEILINGS",
        "verdict_scope": "INSTANCE:DX2_archive_976f706d_BL1_primary_RC64_field",
        "axis": binding["axis"],
        "score_claim": False,
        "pointer_moved": False,
        "source_binding": binding,
        "preflight": preflight,
        "uniform_cost_bits": UNIFORM_BITS,
        "positions": POSITIONS,
        "overshoot_positions": overshoot_positions,
        "overshoot_position_fraction": overshoot_positions / POSITIONS,
        "overshoot_bits": overshoot_bits,
        "gross_excess_bits": gross_excess_bits,
        "gross_excess_bytes": gross_excess_bits / 8.0,
        "gross_share_of_demand": (gross_excess_bits / 8.0) / DEMAND_BYTES,
        "explicit_flag_signalling": signalling,
        "explicit_flag_net_ceiling_bytes": gross_excess_bits / 8.0 - signal_bytes,
        "explicit_flag_net_share_of_demand": (gross_excess_bits / 8.0 - signal_bytes) / DEMAND_BYTES,
        "uniform_member_pricing": members,
        "bl1_reproduction": reproduction,
        "class_rows": class_rows,
        "frame_rows": frames,
        "group_rows": groups,
        "token_rows": token_rows,
        "predictor_hit_miss_rows": prediction_rows,
        "predictor_context_rows": contexts,
        "fields": fields,
        "elapsed_seconds": time.time() - started,
        "authority_boundaries": [
            "gross excess is an exact selected-symbol model-cost ceiling from BL1's reconciled RC64 field",
            "signalling payload sizes are exact retained real-coder bytes with deterministic inverse",
            "member gains are modelled selected-probability code lengths, not finite-precision RC64 streams",
            "no token recode, archive candidate, receiver integration, scorer, or exact evaluator ran",
            "the FS2 predictor join is a same-token-field conditioning coordinate that predates DX2's final 70-byte corrector improvement",
        ],
    }
    atomic_json(output / "RESULT.json", result)
    manifest = write_manifest(output, result)
    result["manifest"] = {
        "path": str(output / "MANIFEST.json"),
        "bytes": (output / "MANIFEST.json").stat().st_size,
        "sha256": sha256_file(output / "MANIFEST.json"),
        "artifact_count": manifest["artifact_count"],
        "artifact_bytes": manifest["artifact_bytes"],
    }
    return result


def verify_completed(output: Path) -> dict[str, Any]:
    result_path = output / "RESULT.json"
    manifest_path = output / "MANIFEST.json"
    if not result_path.is_file() or not manifest_path.is_file():
        raise Ae1Error("completed verification requires RESULT.json and MANIFEST.json")
    result = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if result["source_binding"] != source_binding():
        raise Ae1Error("completed result source binding drifted")
    for fact in manifest["artifacts"]:
        if file_fact(Path(fact["path"])) != fact:
            raise Ae1Error(f"manifest artifact drifted: {fact['path']}")
    if file_fact(result_path) != manifest["result"]:
        raise Ae1Error("manifest result receipt drifted")
    return {
        "status": "VERIFIED_COMPLETE",
        "result": file_fact(result_path),
        "manifest": file_fact(manifest_path),
        "artifacts_verified": len(manifest["artifacts"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-completed", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--seal-failed-output", action="store_true")
    parser.add_argument("--failure-reason")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("ddm_ae1 self-test: PASS")
        return
    if args.verify_completed:
        print(json.dumps(verify_completed(args.output), indent=2, sort_keys=True))
        return
    if args.seal_failed_output:
        if not args.failure_reason:
            raise Ae1Error("--seal-failed-output requires --failure-reason")
        print(json.dumps(seal_failed_output(args.output, args.failure_reason), indent=2, sort_keys=True))
        return
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise Ae1Error("output exists; pass --resume only after inspecting retained receipts")
    result = run(args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "overshoot_positions": result["overshoot_positions"],
                "gross_excess_bytes": result["gross_excess_bytes"],
                "explicit_flag_net_ceiling_bytes": result["explicit_flag_net_ceiling_bytes"],
                "member_rows": [
                    {
                        "kind": row["kind"],
                        "modelled_gain_bytes": row["modelled_gain_bytes"],
                        "modelled_net_bytes_after_description": row[
                            "modelled_net_bytes_after_description"
                        ],
                    }
                    for row in result["uniform_member_pricing"]["rows"]
                ],
                "result": file_fact(args.output / "RESULT.json"),
                "manifest": result["manifest"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
