#!/usr/bin/env python3
"""Full-n600 scorer-free falsifier for the GDC1 ordered-scanline program.

The packet is the complete source object.  It stores every video-selected row
transition in counted bytes; the decoder gets no target field, free fitted
constant, previous token plane, or scorer output.  All materialized payloads
and deterministic coder repeats are retained under the requested SSD root.
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
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
for _root in (REPO, REPO / "src", REPO / "experiments"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_hg1_heterogeneous_analytic_generator_gate as hg1

N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
CLASSES: Final = 5
TOTAL: Final = N_PAIRS * HEIGHT * WIDTH
FIELD_SHA256: Final = "78e57545439515eb29f806cc5a5f7d8b14acf658955cdd7561debb4edf3b7db6"
HG1_SHA256: Final = "4026c4e2c805beb5b79be2879bb4a84311655d0d7d80dbc766654847522a5d19"
RATE_PER_BYTE: Final = 25.0 / 37_545_489
POINTER_BYTES: Final = 180_466
POINTER_D_SEG: Final = 0.00010345
POINTER_D_POSE: Final = 0.00000459
SUB012_REAL_CAP: Final = (0.12 - (100.0 * POINTER_D_SEG + math.sqrt(10.0 * POINTER_D_POSE))) / RATE_PER_BYTE
SUB012_INTEGER_CAP: Final = math.ceil(SUB012_REAL_CAP) - 1
TAIL_ENVELOPE_BYTES: Final = 119_969
FIXED_NONTAIL_BYTES: Final = POINTER_BYTES - TAIL_ENVELOPE_BYTES
REPLACEMENT_REAL_CAP: Final = SUB012_REAL_CAP - FIXED_NONTAIL_BYTES
REPLACEMENT_INTEGER_CAP: Final = SUB012_INTEGER_CAP - FIXED_NONTAIL_BYTES
GF1_BYTES_PER_MISMATCH: Final = 0.2909
MIN_FREE_BYTES: Final = 40 * 1024**3

RAW_MAGIC: Final = b"GDC1SR1!"
RAW_HEADER: Final = struct.Struct("<8sBHHHH")
PACKET_MAGIC: Final = b"GDC1SP1!"
PACKET_HEADER: Final = struct.Struct("<8sBBHHHHII32s32s")


class GDC1Error(RuntimeError):
    """A GDC1 source, packet, receiver, or custody invariant failed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != payload:
            raise GDC1Error(f"existing payload differs: {path}")
        return file_fact(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return file_fact(path)


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return atomic_bytes(path, payload)


def install_completed_file_once(temporary: Path, destination: Path) -> dict[str, Any]:
    temporary_fact = file_fact(temporary)
    if destination.is_file():
        destination_fact = file_fact(destination)
        if destination_fact == {**temporary_fact, "path": str(destination)}:
            temporary.unlink()
            return destination_fact
        conflict = destination.with_name(
            f"{destination.name}.conflict-{temporary_fact['sha256'][:12]}"
        )
        os.replace(temporary, conflict)
        raise GDC1Error(
            f"completed payload differs from existing destination; retained conflict at {conflict}"
        )
    os.replace(temporary, destination)
    return file_fact(destination)


def _rle(row: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundaries = np.flatnonzero(row[1:] != row[:-1]) + 1
    starts = np.concatenate((np.asarray([0]), boundaries))
    ends = np.concatenate((boundaries, np.asarray([row.size])))
    labels = row[starts].astype(np.uint8, copy=False)
    return starts.astype(np.int16), ends.astype(np.int16), labels


def optimal_segments(row: np.ndarray, maximum_runs: int) -> tuple[np.ndarray, np.ndarray]:
    """Return exact minimum-Hamming transition endpoints and labels.

    An optimum has a representative whose endpoints are source transitions, so
    the dynamic program operates on source runs rather than all 512 pixels.
    Ties choose the earlier predecessor and then the lower class.
    """

    starts, ends, source_labels = _rle(np.asarray(row, dtype=np.uint8))
    runs = len(source_labels)
    if runs <= maximum_runs:
        return ends[:-1].astype(np.uint16), source_labels.copy()
    lengths = (ends - starts).astype(np.int32)
    prefix = np.zeros((runs + 1, CLASSES), dtype=np.int32)
    for index, (label, length) in enumerate(zip(source_labels.tolist(), lengths.tolist(), strict=True)):
        prefix[index + 1] = prefix[index]
        prefix[index + 1, label] += length
    costs = np.zeros((runs, runs + 1), dtype=np.int32)
    choices = np.zeros((runs, runs + 1), dtype=np.uint8)
    for left in range(runs):
        for right in range(left + 1, runs + 1):
            counts = prefix[right] - prefix[left]
            label = int(np.argmax(counts))
            costs[left, right] = int(counts.sum() - counts[label])
            choices[left, right] = label
    infinity = np.iinfo(np.int32).max // 4
    dp = np.full((maximum_runs + 1, runs + 1), infinity, dtype=np.int32)
    previous = np.full((maximum_runs + 1, runs + 1), -1, dtype=np.int16)
    dp[0, 0] = 0
    for segment_count in range(1, maximum_runs + 1):
        for right in range(segment_count, runs + 1):
            best_cost = infinity
            best_left = -1
            for left in range(segment_count - 1, right):
                candidate = int(dp[segment_count - 1, left] + costs[left, right])
                if candidate < best_cost:
                    best_cost = candidate
                    best_left = left
            dp[segment_count, right] = best_cost
            previous[segment_count, right] = best_left
    groups: list[tuple[int, int]] = []
    right = runs
    for segment_count in range(maximum_runs, 0, -1):
        left = int(previous[segment_count, right])
        if left < 0:
            raise GDC1Error("scanline dynamic program has no predecessor")
        groups.append((left, right))
        right = left
    groups.reverse()
    raw_ends: list[int] = []
    raw_labels: list[int] = []
    for left, right in groups:
        label = int(choices[left, right])
        endpoint = int(ends[right - 1])
        if raw_labels and label == raw_labels[-1]:
            raw_ends[-1] = endpoint
        else:
            raw_labels.append(label)
            raw_ends.append(endpoint)
    return np.asarray(raw_ends[:-1], dtype=np.uint16), np.asarray(raw_labels, dtype=np.uint8)


def render_segments(endpoints: np.ndarray, labels: np.ndarray, width: int = WIDTH) -> np.ndarray:
    output = np.empty(width, dtype=np.uint8)
    left = 0
    for endpoint, label in zip((*endpoints.tolist(), width), labels.tolist(), strict=True):
        right = int(endpoint)
        if right <= left or right > width or label >= CLASSES:
            raise GDC1Error("invalid scanline segment")
        output[left:right] = label
        left = right
    if left != width:
        raise GDC1Error("scanline does not cover the row")
    return output


def build_raw_program(counts: np.ndarray, starts: np.ndarray, endpoints: np.ndarray, labels: np.ndarray,
                      *, pairs: int, height: int, width: int, maximum_runs: int) -> bytes:
    if counts.shape != (pairs * height,) or starts.shape != counts.shape:
        raise GDC1Error("count/start shape mismatch")
    slots = maximum_runs - 1
    if endpoints.shape != (pairs * height, slots) or labels.shape != endpoints.shape:
        raise GDC1Error("endpoint/label shape mismatch")
    endpoint_cube = endpoints.reshape(pairs, height, slots).astype(np.int32)
    differences = np.empty_like(endpoint_cube)
    differences[:, 0] = endpoint_cube[:, 0]
    differences[:, 1:] = endpoint_cube[:, 1:] - endpoint_cube[:, :-1]
    zigzag = np.where(differences >= 0, differences * 2, -differences * 2 - 1)
    if np.any(zigzag > np.iinfo(np.uint16).max):
        raise GDC1Error("endpoint delta escaped uint16")
    return b"".join((
        RAW_HEADER.pack(RAW_MAGIC, 1, maximum_runs, pairs, height, width),
        counts.astype(np.uint8, copy=False).tobytes(),
        starts.astype(np.uint8, copy=False).tobytes(),
        zigzag.astype("<u2", copy=False).tobytes(),
        labels.astype(np.uint8, copy=False).tobytes(),
    ))


def parse_raw_program(payload: bytes) -> dict[str, Any]:
    if len(payload) < RAW_HEADER.size:
        raise GDC1Error("raw program truncated")
    magic, version, maximum_runs, pairs, height, width = RAW_HEADER.unpack_from(payload)
    if magic != RAW_MAGIC or version != 1 or not (1 <= maximum_runs <= width):
        raise GDC1Error("raw program header invalid")
    rows = pairs * height
    slots = maximum_runs - 1
    expected = RAW_HEADER.size + rows * 2 + rows * slots * 3
    if len(payload) != expected:
        raise GDC1Error("raw program length mismatch")
    cursor = RAW_HEADER.size
    counts = np.frombuffer(payload, dtype=np.uint8, count=rows, offset=cursor).copy()
    cursor += rows
    starts = np.frombuffer(payload, dtype=np.uint8, count=rows, offset=cursor).copy()
    cursor += rows
    zigzag = np.frombuffer(payload, dtype="<u2", count=rows * slots, offset=cursor).reshape(pairs, height, slots)
    cursor += rows * slots * 2
    labels = np.frombuffer(payload, dtype=np.uint8, count=rows * slots, offset=cursor).reshape(rows, slots).copy()
    differences = np.where(zigzag % 2 == 0, zigzag // 2, -((zigzag.astype(np.int32) + 1) // 2))
    endpoints = np.cumsum(differences, axis=1, dtype=np.int32).reshape(rows, slots)
    if np.any((counts < 1) | (counts > maximum_runs)) or np.any(starts >= CLASSES):
        raise GDC1Error("raw program class/count invalid")
    for row_index, count in enumerate(counts.tolist()):
        active = count - 1
        if active and (np.any(endpoints[row_index, :active] <= 0) or
                       np.any(endpoints[row_index, :active] >= width) or
                       np.any(np.diff(endpoints[row_index, :active]) <= 0) or
                       np.any(labels[row_index, :active] >= CLASSES)):
            raise GDC1Error("raw program active transition invalid")
        if np.any(endpoints[row_index, active:] != 0) or np.any(labels[row_index, active:] != 255):
            raise GDC1Error("raw program inactive slot is noncanonical")
    return {
        "pairs": pairs,
        "height": height,
        "width": width,
        "maximum_runs": maximum_runs,
        "counts": counts,
        "starts": starts,
        "endpoints": endpoints,
        "labels": labels,
    }


def coder_race(raw: bytes, root: Path) -> tuple[dict[str, Any], str, bytes]:
    rows: dict[str, Any] = {}
    for coder in hg1.CODERS:
        directory = root / "coder_race" / coder
        coded = hg1.et1.compress_payload(raw, coder)
        repeated = hg1.et1.compress_payload(raw, coder)
        coded_fact = atomic_bytes(directory / "payload.coded", coded)
        repeat_fact = atomic_bytes(directory / "payload.repeat.coded", repeated)
        if coded != repeated or hg1.et1.decompress_payload(coded, coder) != raw:
            raise GDC1Error(f"{coder} repeat or parse-back failed")
        rows[coder] = {
            "coded": coded_fact,
            "repeat": repeat_fact,
            "deterministic_repeat_equal": True,
            "raw_parseback_equal": True,
        }
    winner = min(hg1.CODERS, key=lambda name: (rows[name]["coded"]["bytes"], hg1.CODERS.index(name)))
    return rows, winner, Path(rows[winner]["coded"]["path"]).read_bytes()


def build_packet(raw: bytes, coder: str, coded: bytes) -> bytes:
    return PACKET_HEADER.pack(
        PACKET_MAGIC, 1, hg1.et1.CODER_IDS[coder], 0, 0, 0, 0, len(raw), len(coded),
        bytes.fromhex(sha256_bytes(raw)), bytes.fromhex(sha256_bytes(coded)),
    ) + coded


def parse_packet(packet: bytes) -> dict[str, Any]:
    if len(packet) < PACKET_HEADER.size:
        raise GDC1Error("packet truncated")
    magic, version, coder_id, r0, r1, r2, r3, raw_size, coded_size, raw_sha, coded_sha = PACKET_HEADER.unpack_from(packet)
    coded = packet[PACKET_HEADER.size:]
    if (magic != PACKET_MAGIC or version != 1 or coder_id not in hg1.et1.CODER_NAMES or
            any((r0, r1, r2, r3)) or len(coded) != coded_size or sha256_bytes(coded) != coded_sha.hex()):
        raise GDC1Error("packet header or coded identity invalid")
    raw = hg1.et1.decompress_payload(coded, hg1.et1.CODER_NAMES[coder_id])
    if len(raw) != raw_size or sha256_bytes(raw) != raw_sha.hex():
        raise GDC1Error("packet raw identity invalid")
    return parse_raw_program(raw)


def render_packet(packet: bytes, output_path: Path) -> tuple[dict[str, Any], float]:
    program = parse_packet(packet)
    shape = (program["pairs"], program["height"], program["width"])
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    started = time.monotonic()
    output = np.memmap(temporary, dtype=np.uint8, mode="w+", shape=shape)
    for row_index, count in enumerate(program["counts"].tolist()):
        active = count - 1
        row_labels = np.concatenate((
            np.asarray([program["starts"][row_index]], dtype=np.uint8),
            program["labels"][row_index, :active],
        ))
        output.reshape(-1, shape[2])[row_index] = render_segments(
            program["endpoints"][row_index, :active], row_labels, shape[2]
        )
    output.flush()
    del output
    fact = install_completed_file_once(temporary, output_path)
    seconds = time.monotonic() - started
    return fact, seconds


def save_fit_checkpoint(
    root: Path,
    *,
    pair_count: int,
    counts: np.ndarray,
    starts: np.ndarray,
    endpoints: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    row_count = pair_count * HEIGHT
    path = root / "checkpoints" / f"pair_{pair_count:03d}.npz"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("xb") as handle:
            np.savez(
                handle,
                pair_count=np.asarray(pair_count, dtype=np.int64),
                counts=counts[:row_count],
                starts=starts[:row_count],
                endpoints=endpoints[:row_count],
                labels=labels[:row_count],
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    fact = file_fact(path)
    atomic_json(root / f"stage_fit_pair_{pair_count:03d}.json", {
        "stage": "fit", "pairs_complete": pair_count, "checkpoint": fact,
    })
    return fact


def load_latest_fit_checkpoint(
    root: Path,
    *,
    maximum_runs: int,
    counts: np.ndarray,
    starts: np.ndarray,
    endpoints: np.ndarray,
    labels: np.ndarray,
) -> int:
    candidates = sorted((root / "checkpoints").glob("pair_*.npz"))
    if not candidates:
        return 0
    path = candidates[-1]
    with np.load(path, allow_pickle=False) as checkpoint:
        pair_count = int(checkpoint["pair_count"])
        row_count = pair_count * HEIGHT
        expected_endpoint_shape = (row_count, maximum_runs - 1)
        if (
            pair_count <= 0
            or pair_count > N_PAIRS
            or checkpoint["counts"].shape != (row_count,)
            or checkpoint["starts"].shape != (row_count,)
            or checkpoint["endpoints"].shape != expected_endpoint_shape
            or checkpoint["labels"].shape != expected_endpoint_shape
        ):
            raise GDC1Error("fit checkpoint shape invalid")
        counts[:row_count] = checkpoint["counts"]
        starts[:row_count] = checkpoint["starts"]
        endpoints[:row_count] = checkpoint["endpoints"]
        labels[:row_count] = checkpoint["labels"]
    return pair_count


def fit_candidate(target: np.ndarray, maximum_runs: int, root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "RESULT.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        for key in ("raw_program", "packet", "packet_repeat", "fit_render", "receiver_render"):
            fact = receipt[key]
            path = Path(fact["path"])
            if not path.is_file() or file_fact(path) != fact:
                raise GDC1Error(f"resume fact failed: {key}")
        return receipt
    rows = N_PAIRS * HEIGHT
    counts = np.empty(rows, dtype=np.uint8)
    starts = np.empty(rows, dtype=np.uint8)
    endpoints = np.zeros((rows, maximum_runs - 1), dtype=np.uint16)
    labels = np.full((rows, maximum_runs - 1), 255, dtype=np.uint8)
    start_pair = load_latest_fit_checkpoint(
        root,
        maximum_runs=maximum_runs,
        counts=counts,
        starts=starts,
        endpoints=endpoints,
        labels=labels,
    )
    fit_path = root / "fit_render.u8"
    temporary = fit_path.with_name(f".{fit_path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    fitted = np.memmap(temporary, dtype=np.uint8, mode="w+", shape=(N_PAIRS, HEIGHT, WIDTH))
    started = time.monotonic()
    for row_index in range(start_pair * HEIGHT):
        active = int(counts[row_index]) - 1
        row_labels = np.concatenate((
            np.asarray([starts[row_index]], dtype=np.uint8), labels[row_index, :active]
        ))
        fitted.reshape(-1, WIDTH)[row_index] = render_segments(
            endpoints[row_index, :active], row_labels
        )
    for pair in range(start_pair, N_PAIRS):
        for y in range(HEIGHT):
            row_index = pair * HEIGHT + y
            row_endpoints, row_labels = optimal_segments(target[pair, y], maximum_runs)
            rendered = render_segments(row_endpoints, row_labels)
            fitted[pair, y] = rendered
            counts[row_index] = len(row_labels)
            starts[row_index] = row_labels[0]
            active = len(row_endpoints)
            endpoints[row_index, :active] = row_endpoints
            labels[row_index, :active] = row_labels[1:]
        if (pair + 1) % 100 == 0:
            fitted.flush()
            save_fit_checkpoint(
                root,
                pair_count=pair + 1,
                counts=counts,
                starts=starts,
                endpoints=endpoints,
                labels=labels,
            )
    fitted.flush()
    del fitted
    fit_fact = install_completed_file_once(temporary, fit_path)
    raw = build_raw_program(counts, starts, endpoints, labels, pairs=N_PAIRS, height=HEIGHT,
                            width=WIDTH, maximum_runs=maximum_runs)
    raw_fact = atomic_bytes(root / "program.raw", raw)
    race, winner, coded = coder_race(raw, root)
    packet = build_packet(raw, winner, coded)
    packet_fact = atomic_bytes(root / "program.packet", packet)
    packet_repeat = atomic_bytes(root / "program.repeat.packet", build_packet(raw, winner, coded))
    receiver_fact, decode_seconds = render_packet(packet, root / "receiver_render.u8")
    if fit_fact["sha256"] != receiver_fact["sha256"]:
        raise GDC1Error("fit and receiver render differ")
    receiver = np.memmap(receiver_fact["path"], dtype=np.uint8, mode="r", shape=target.shape)
    mismatch_mask = receiver != target
    mismatch = int(np.count_nonzero(mismatch_mask))
    by_target_class = [int(np.count_nonzero(mismatch_mask & (target == label))) for label in range(CLASSES)]
    del mismatch_mask, receiver
    projected_residual = GF1_BYTES_PER_MISMATCH * mismatch
    result = {
        "schema": "ddm_gdc1_scanline_candidate.v1",
        "axis": "[macOS-CPU scorer-free exact-field measurement, n600]",
        "selection_mode": "all 600 pairs; all 117,964,800 token cells",
        "score_claim": False,
        "maximum_runs": maximum_runs,
        "mismatches": mismatch,
        "mismatch_fraction": mismatch / TOTAL,
        "mismatches_by_target_class_0_to_4": by_target_class,
        "raw_program": raw_fact,
        "coder_race": race,
        "winning_coder": winner,
        "packet": packet_fact,
        "packet_repeat": packet_repeat,
        "fit_render": fit_fact,
        "receiver_render": receiver_fact,
        "receiver_identity": True,
        "decode_seconds": decode_seconds,
        "projected_residual_bytes_at_gf1_0_2909": projected_residual,
        "projected_replacement_bytes": packet_fact["bytes"] + projected_residual,
        "projected_mismatch_budget_at_this_packet": max(
            -1, math.floor((REPLACEMENT_REAL_CAP - packet_fact["bytes"]) / GF1_BYTES_PER_MISMATCH)
        ),
        "projected_gate_pass": packet_fact["bytes"] + projected_residual < REPLACEMENT_REAL_CAP,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(receipt_path, result)
    return result


def hg1_ceiling(target: np.ndarray, hg1_path: Path) -> dict[str, Any]:
    if hg1_path.stat().st_size != TOTAL or sha256_file(hg1_path) != HG1_SHA256:
        raise GDC1Error("HG1 retained output identity mismatch")
    generated = np.memmap(hg1_path, dtype=np.uint8, mode="r", shape=target.shape)
    mismatch = generated != target
    transition = np.zeros(target.shape, dtype=bool)
    transition[:, 1:, :] |= target[:, 1:, :] != target[:, :-1, :]
    transition[:, :-1, :] |= target[:, :-1, :] != target[:, 1:, :]
    transition[:, :, 1:] |= target[:, :, 1:] != target[:, :, :-1]
    transition[:, :, :-1] |= target[:, :, :-1] != target[:, :, 1:]
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1] = True
    edge_counts: dict[str, int] = {"distance_0": int(np.count_nonzero(mismatch & transition))}
    dilated = transition
    for distance in range(1, 5):
        dilated = ndimage.binary_dilation(dilated, structure=structure, iterations=1)
        if distance in (1, 2, 4):
            edge_counts[f"distance_le_{distance}"] = int(np.count_nonzero(mismatch & dilated))
    confusion = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    for target_class in range(CLASSES):
        for generated_class in range(CLASSES):
            if target_class != generated_class:
                confusion[target_class, generated_class] = np.count_nonzero(
                    (target == target_class) & (generated == generated_class)
                )
    row_bands = {
        "rows_0_127": int(np.count_nonzero(mismatch[:, :128])),
        "rows_128_255": int(np.count_nonzero(mismatch[:, 128:256])),
        "rows_256_383": int(np.count_nonzero(mismatch[:, 256:])),
    }
    result = {
        "path": str(hg1_path), "bytes": hg1_path.stat().st_size, "sha256": HG1_SHA256,
        "mismatches": int(np.count_nonzero(mismatch)),
        "mismatch_fraction": float(np.mean(mismatch)),
        "mismatches_by_target_class_0_to_4": confusion.sum(axis=1).tolist(),
        "target_to_generated_confusion_off_diagonal": confusion.tolist(),
        "target_edge_cells": int(np.count_nonzero(transition)),
        "mismatches_near_target_edge": edge_counts,
        "mismatches_by_row_band": row_bands,
    }
    del mismatch, transition, dilated, generated
    return result


def real_residual_race(target: np.ndarray, generated_path: Path, root: Path) -> dict[str, Any]:
    generated = np.memmap(generated_path, dtype=np.uint8, mode="r", shape=target.shape)
    rows: list[dict[str, Any]] = []
    for order in ("frame_raster", "class_frame_raster", "tile64_time"):
        raw_path = root / order / "residual.raw"
        fact = hg1.encode_residual(target, generated, raw_path, None, order)
        race = hg1.coder_race(f"winning_residual_{order}", raw_path, root)
        winner = race["winner"]
        coded = race["coders"][winner]["coded"]
        rows.append({"order": order, "raw": fact, "coder_race": race, "winner": winner,
                     "coded": coded, "coded_bytes": coded["bytes"]})
    best = min(rows, key=lambda row: (row["coded_bytes"], row["order"]))
    corrected = np.array(generated, copy=True)
    hg1.apply_residual(Path(best["raw"]["path"]).read_bytes(), corrected)
    if not np.array_equal(corrected, target):
        raise GDC1Error("winning residual did not close the target")
    return {"rows": rows, "best_order": best["order"], "best_coded_bytes": best["coded_bytes"],
            "exact_target_closure": True}


def build_manifest(root: Path) -> list[dict[str, Any]]:
    excluded = {root / "MANIFEST.json"}
    return [file_fact(path) for path in sorted(root.rglob("*")) if path.is_file() and path not in excluded]


def verify_terminal_manifest(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "MANIFEST.json").read_text())
    for expected in manifest["entries"]:
        path = Path(expected["path"])
        if not path.is_file() or file_fact(path) != expected:
            raise GDC1Error(f"terminal manifest fact failed: {path}")
    return manifest


def render_packet_array(packet: bytes) -> np.ndarray:
    program = parse_packet(packet)
    shape = (program["pairs"], program["height"], program["width"])
    output = np.empty(shape, dtype=np.uint8)
    flat = output.reshape(-1, shape[2])
    for row_index, count in enumerate(program["counts"].tolist()):
        active = count - 1
        row_labels = np.concatenate((
            np.asarray([program["starts"][row_index]], dtype=np.uint8),
            program["labels"][row_index, :active],
        ))
        flat[row_index] = render_segments(program["endpoints"][row_index, :active], row_labels, shape[2])
    return output


def verify_terminal_science(root: Path, target: np.ndarray, result: dict[str, Any]) -> dict[str, Any]:
    point_checks: list[dict[str, Any]] = []
    winning_render: np.ndarray | None = None
    winning_k = int(result["selection"]["winning_k"])
    for point in result["points"]:
        rendered = render_packet_array(Path(point["packet"]["path"]).read_bytes())
        render_sha = sha256_bytes(memoryview(rendered))
        mismatches = int(np.count_nonzero(rendered != target))
        if render_sha != point["receiver_render"]["sha256"] or mismatches != point["mismatches"]:
            raise GDC1Error(f"terminal scientific replay differs at K={point['maximum_runs']}")
        point_checks.append({
            "maximum_runs": point["maximum_runs"], "render_sha256": render_sha,
            "mismatches": mismatches, "receiver_equal": True,
        })
        if int(point["maximum_runs"]) == winning_k:
            winning_render = rendered
    if winning_render is None:
        raise GDC1Error("terminal scientific replay did not find winning render")
    best_order = result["real_residual"]["best_order"]
    residual_row = next(row for row in result["real_residual"]["rows"] if row["order"] == best_order)
    corrections = hg1.apply_residual(Path(residual_row["raw"]["path"]).read_bytes(), winning_render)
    if corrections != result["selection"]["winning_mismatches"] or not np.array_equal(winning_render, target):
        raise GDC1Error("terminal residual replay did not close the target")
    return {"points": point_checks, "winning_residual_exact_target_closure": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--hg1-render", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--k-values", type=int, nargs="+", default=(4, 6, 8, 12, 16, 24))
    args = parser.parse_args(argv)
    if args.output_dir.resolve() != args.resume_from.resolve():
        raise GDC1Error("--resume-from must name the complete output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.output_dir).free
    terminal_resume = (args.output_dir / "RESULT.json").is_file() and (args.output_dir / "MANIFEST.json").is_file()
    if free < MIN_FREE_BYTES and not terminal_resume:
        raise GDC1Error(f"storage preflight failed: {free} B free < {MIN_FREE_BYTES} B")
    if args.field.stat().st_size != TOTAL or sha256_file(args.field) != FIELD_SHA256:
        raise GDC1Error("move43 field identity mismatch")
    if args.seed != 20260910:
        raise GDC1Error("this deterministic no-RNG run requires the registered seed")
    launch = {
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "axis": "[macOS-CPU scorer-free exact-field measurement, n600]",
        "seed": args.seed, "rng_used": False, "k_values": args.k_values,
        "field": file_fact(args.field), "hg1_render": file_fact(args.hg1_render),
        "storage_free_bytes": free, "minimum_free_bytes": MIN_FREE_BYTES,
        "resume_from": str(args.resume_from), "score_claim": False,
    }
    launch_path = args.output_dir / "LAUNCH.json"
    if launch_path.is_file():
        prior_launch = json.loads(launch_path.read_text())
        for key in ("seed", "rng_used", "k_values", "field", "hg1_render", "resume_from", "score_claim"):
            if prior_launch[key] != launch[key]:
                raise GDC1Error(f"resume launch identity differs at {key}")
    else:
        atomic_json(launch_path, launch)
    if terminal_resume:
        manifest = verify_terminal_manifest(args.output_dir)
        result = json.loads((args.output_dir / "RESULT.json").read_text())
        target = np.memmap(args.field, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
        scientific_replay = verify_terminal_science(args.output_dir, target, result)
        print(json.dumps({
            "terminal_resume_verified": True,
            "scientific_replay_verified": True,
            "scientific_replay_points": len(scientific_replay["points"]),
            "winning_residual_exact_target_closure": scientific_replay["winning_residual_exact_target_closure"],
            "manifest_entries": manifest["entry_count"],
            "manifest_bytes": manifest["total_bytes"],
            "winning_k": result["selection"]["winning_k"],
            "actual_replacement_bytes": result["actual_replacement_bytes"],
            "verdict": result["verdict"],
        }, indent=2))
        return 0
    target = np.memmap(args.field, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
    hg1_result = hg1_ceiling(target, args.hg1_render)
    points = [fit_candidate(target, value, args.output_dir / f"k{value:02d}") for value in args.k_values]
    best = min(points, key=lambda row: (row["projected_replacement_bytes"], row["packet"]["bytes"]))
    residual = real_residual_race(
        target, Path(best["receiver_render"]["path"]), args.output_dir / "winning_residual"
    )
    actual_replacement = int(best["packet"]["bytes"]) + int(residual["best_coded_bytes"])
    result = {
        "schema": "ddm_gdc1_scanline_partition_program.v1",
        "axis": "[macOS-CPU scorer-free exact-field measurement, n600]",
        "score_claim": False,
        "target_field": file_fact(args.field),
        "hg1_reference": hg1_result,
        "bar": {
            "rate_per_byte": RATE_PER_BYTE,
            "pointer_bytes": POINTER_BYTES,
            "pointer_d_seg": POINTER_D_SEG,
            "pointer_d_pose": POINTER_D_POSE,
            "held_distortion": 100.0 * POINTER_D_SEG + math.sqrt(10.0 * POINTER_D_POSE),
            "sub012_real_archive_cap": SUB012_REAL_CAP,
            "sub012_strict_integer_archive_cap": SUB012_INTEGER_CAP,
            "current_tail_envelope_bytes": TAIL_ENVELOPE_BYTES,
            "fixed_nontail_bytes": FIXED_NONTAIL_BYTES,
            "replacement_real_cap": REPLACEMENT_REAL_CAP,
            "replacement_strict_integer_cap": REPLACEMENT_INTEGER_CAP,
            "demand_bytes": POINTER_BYTES - SUB012_INTEGER_CAP,
            "tail_savings_fraction_required": (POINTER_BYTES - SUB012_INTEGER_CAP) / TAIL_ENVELOPE_BYTES,
            "gf1_transferred_bytes_per_mismatch": GF1_BYTES_PER_MISMATCH,
        },
        "points": points,
        "selection": {
            "mode": "minimum packet + transferred GF1 0.2909 B/mismatch over complete physical roster",
            "winning_k": best["maximum_runs"],
            "winning_packet_bytes": best["packet"]["bytes"],
            "winning_mismatches": best["mismatches"],
            "projected_replacement_bytes": best["projected_replacement_bytes"],
        },
        "real_residual": residual,
        "actual_replacement_bytes": actual_replacement,
        "actual_over_gate_bytes": actual_replacement - REPLACEMENT_INTEGER_CAP,
        "actual_gate_pass": actual_replacement <= REPLACEMENT_INTEGER_CAP,
        "decode_ceiling_seconds": 1260,
        "max_measured_decode_seconds": max(float(row["decode_seconds"]) for row in points),
        "decoder_time_gate_pass": max(float(row["decode_seconds"]) for row in points) <= 1260,
        "verdict": "PASS" if actual_replacement <= REPLACEMENT_INTEGER_CAP else "FORMULATION-NO-GO",
        "verdict_scope": "FORMULATION: exact-DP ordered scanline transition program at K=4,6,8,12,16,24",
        "calls": {"scorer": 0, "modal": 0, "mps": 0, "training": 0},
    }
    atomic_json(args.output_dir / "RESULT.json", result)
    manifest = build_manifest(args.output_dir)
    atomic_json(args.output_dir / "MANIFEST.json", {
        "schema": "ddm_gdc1_retained_manifest.v1", "entries": manifest,
        "entry_count": len(manifest), "total_bytes": sum(row["bytes"] for row in manifest),
    })
    print(json.dumps({
        "winning_k": best["maximum_runs"], "packet_bytes": best["packet"]["bytes"],
        "mismatches": best["mismatches"], "actual_residual_bytes": residual["best_coded_bytes"],
        "actual_replacement_bytes": actual_replacement, "gate": REPLACEMENT_INTEGER_CAP,
        "verdict": result["verdict"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
