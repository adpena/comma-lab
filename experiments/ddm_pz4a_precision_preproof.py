#!/usr/bin/env python3
"""Scorer-free sensitivity-allocated CPR1 coefficient precision pre-proof.

The pass-03 selected coefficient lattice is recoded through the exact shipped
CPR1 Rice coder and the exact outer Brotli-q9 carrier cell.  Allocation uses
the explicitly stale pass-02-parent Jacobian only as a PLANNING-BAND
linearization.  Every materialized coefficient stream, carrier, coder output,
depth map, and deterministic repeat is retained before a scalar result is
published.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.util
import json
import lzma
import math
import os
import platform
import shutil
import struct
import sys
import time
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
AP_ROOT = Path("/Volumes/APDataStore/pact")
DEFAULT_OUTPUT = AP_ROOT / "ddm_pz4a/retained/preproof_v1"
REHEARSAL = Path("/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/t0_rehearsal_pass03")
COEFFICIENTS = REHEARSAL / "objects/terminal_coefficients.int16.npy"
CARRIER = REHEARSAL / "objects/terminal_carrier.cpr1"
ARCHIVE = REHEARSAL / "objects/terminal_archive.zip"
SENSITIVITY = REHEARSAL / "diagnostics/UNBOUND_pass03_input_parent_sensitivity_map.npz"
SENSITIVITY_RECEIPT = REHEARSAL / "diagnostics/UNBOUND_sensitivity_map_receipt.json"
CODEC_SOURCE = REPO / "src/tac/pr130_runtime/fx1_runtime_tree/carrier_codec.py"

EXPECTED = {
    COEFFICIENTS: (14_528, "2daec0ae99e86f2a6583a96561335186992a2a1235791af461083ada44d3503d"),
    CARRIER: (23_050, "a532057d6c786c5e367d83c0a686d7b0c313a7d5b2a2fa6bd2ed7fc47e837684"),
    ARCHIVE: (187_222, "93f8d7b4b668919d2357a02cde2a96fc0488ec7e2ac00a250f509d27dbef4c6e"),
    SENSITIVITY: (285_478, "1ac48d8323526729c4ed1d4d507a85e9d22a53a7c0ffeaaedc4e735daed020de"),
}
PRODUCER_PARENT_SHA256 = "b8c3b1187cff48eb8208973536c8f94874c78fb4ad68df84e92fbe9418c1b24a"
CONSUMER_PARENT_SHA256 = EXPECTED[ARCHIVE][1]
N = 600
D = 12
POSE_DIMS = 6
MIN_DEPTH = 2
MAX_DEPTH = 12
TOLERANCES = (0.0083, 0.008675, 0.00905, 0.009425, 0.0098)
DEPTH_HEADER = struct.Struct("<4sBBHBII")
DEPTH_MAGIC = b"PZ4D"
DEPTH_VERSION = 1
MIN_FREE_BYTES = 64 * 1024 * 1024
AXIS = "[scorer-free coder measurement + Jacobian-linearized planning derivation]"
PLANNING_PARENT = (
    "pass-02-parent Jacobian b8c3b118... applied to pass-03 selected "
    "coefficients 93f8d7b4...; stale for terminal binding"
)


class PreproofError(RuntimeError):
    """A custody, coder, allocation, or retention invariant failed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pending(path: Path) -> Path:
    return path.with_name(f".{path.name}.pending.{os.getpid()}")


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _pending(path)
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return file_record(path)


def atomic_json(path: Path, payload: Any) -> dict[str, Any]:
    return atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )


def atomic_npy(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _pending(path)
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)
    return file_record(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _pending(path)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    return file_record(path)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verified_record(record: dict[str, Any]) -> bool:
    path = Path(str(record.get("path", "")))
    return path.is_file() and path.stat().st_size == record.get("bytes") and sha256_file(path) == record.get("sha256")


def import_codec() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ddm_pz4a_carrier_codec", CODEC_SOURCE)
    if spec is None or spec.loader is None:
        raise PreproofError("cannot import the pinned CPR1 codec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_output_root(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(AP_ROOT.resolve()):
        raise PreproofError(f"output must remain below {AP_ROOT}")
    resolved.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(resolved).free
    if free < MIN_FREE_BYTES:
        raise PreproofError(f"APDataStore free-space preflight failed: {free}")
    return resolved


def verify_inputs() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for path, (expected_bytes, expected_sha) in EXPECTED.items():
        if not path.is_file():
            raise PreproofError(f"missing pinned input: {path}")
        observed = file_record(path)
        if observed["bytes"] != expected_bytes or observed["sha256"] != expected_sha:
            raise PreproofError(f"pinned input changed: {path}")
        records[path.name] = observed
    receipt = json.loads(SENSITIVITY_RECEIPT.read_text())
    if (
        receipt.get("producer_parent_sha256") != PRODUCER_PARENT_SHA256
        or receipt.get("consumer_parent_sha256") != CONSUMER_PARENT_SHA256
        or receipt.get("freshness_ok") is not False
    ):
        raise PreproofError("sensitivity parent-mismatch receipt changed")
    records["sensitivity_receipt"] = file_record(SENSITIVITY_RECEIPT)
    records["codec_source"] = file_record(CODEC_SOURCE)
    return records


def parse_archive_carrier_stream(archive_path: Path) -> tuple[bytes, bytes]:
    with zipfile.ZipFile(archive_path) as handle:
        entries = handle.infolist()
        if (
            len(entries) != 1
            or entries[0].filename != "p"
            or entries[0].compress_type != zipfile.ZIP_STORED
            or entries[0].flag_bits & 1
        ):
            raise PreproofError("selected archive is not the expected stored single-member ZIP")
        member = handle.read("p")
        if handle.testzip() is not None:
            raise PreproofError("selected archive failed CRC")
    if len(member) < 16:
        raise PreproofError("selected member is truncated")
    model_word = struct.unpack_from("<I", member)[0]
    model_length_mask = (1 << 29) - 1
    if model_word & (3 << 29) != 3 << 29 or model_word & (1 << 31) == 0:
        raise PreproofError("selected archive is not CX2 split-Brotli plus ANS")
    model_bytes = model_word & model_length_mask
    models = member[4 : 4 + model_bytes]
    if len(models) != model_bytes or 4 + model_bytes >= len(member):
        raise PreproofError("selected archive model framing is invalid")
    lengths = struct.unpack_from("<III", models)
    if 12 + sum(lengths) != len(models) or any(length == 0 for length in lengths):
        raise PreproofError("selected archive split-stream framing is invalid")
    offset = 12 + lengths[0]
    carrier_stream = models[offset : offset + lengths[1]]
    try:
        decoded = brotli.decompress(carrier_stream)
    except brotli.error as error:
        raise PreproofError("selected carrier Brotli stream is invalid") from error
    return carrier_stream, decoded


def signed_codes_from_encoded(encoded: np.ndarray) -> np.ndarray:
    encoded64 = np.asarray(encoded, dtype=np.int64)
    delta = (encoded64 >> 1) ^ -(encoded64 & 1)
    unsigned = np.cumsum(delta, axis=0, dtype=np.int64) & 0xFFF
    return np.where(unsigned >= 0x800, unsigned - 0x1000, unsigned).astype(np.int16)


def encoded_from_signed_codes(codes: np.ndarray) -> np.ndarray:
    codes64 = np.asarray(codes, dtype=np.int64)
    if codes64.shape != (N, D) or np.any(codes64 < -2048) or np.any(codes64 > 2047):
        raise PreproofError("coefficient lattice is outside signed int12")
    unsigned = codes64 & 0xFFF
    previous = np.zeros_like(unsigned)
    previous[1:] = unsigned[:-1]
    delta_unsigned = (unsigned - previous) & 0xFFF
    delta = np.where(delta_unsigned >= 0x800, delta_unsigned - 0x1000, delta_unsigned)
    return (((delta << 1) ^ (delta >> 63)) & 0xFFF).astype(np.int32)


def split_coefficient_component(blob: bytes, codec: ModuleType) -> bytes:
    magic, basis_bits, coefficient_bits = codec.HEADER.unpack_from(blob)
    if magic != codec.MAGIC:
        raise PreproofError("carrier is not CPR1")
    cursor = codec.HEADER.size
    scale_bytes = D * 4
    cursor += scale_bytes
    coefficient_scales = blob[cursor : cursor + scale_bytes]
    cursor += scale_bytes
    cursor += codec.ALPHABET_SIZE
    ks = blob[cursor : cursor + D]
    cursor += D
    cursor += (basis_bits + 7) // 8
    coefficient_payload = blob[cursor : cursor + (coefficient_bits + 7) // 8]
    if cursor + len(coefficient_payload) != len(blob):
        raise PreproofError("CPR1 coefficient boundary did not consume the carrier")
    return struct.pack("<I", coefficient_bits) + coefficient_scales + ks + coefficient_payload


def coefficient_coder_stats(blob: bytes, codec: ModuleType) -> dict[str, Any]:
    _, _, coefficient_bits = codec.HEADER.unpack_from(blob)
    cursor = codec.HEADER.size + D * 8 + codec.ALPHABET_SIZE
    ks = np.frombuffer(blob[cursor : cursor + D], dtype=np.uint8).copy()
    _, _, _, encoded = codec.decode_compact_carrier(blob, basis_count=D * 3 * 24 * 32, frames=N, dimensions=D)
    per_dim_bits = [int(codec._rice_bit_count(encoded[:, dim], int(ks[dim]))) for dim in range(D)]
    if sum(per_dim_bits) != coefficient_bits:
        raise PreproofError("per-dimension Rice accounting does not close")
    return {
        "rice_k_by_dimension": ks.astype(int).tolist(),
        "rice_bits_by_dimension": per_dim_bits,
        "rice_bits": int(coefficient_bits),
        "rice_payload_bytes": math.ceil(int(coefficient_bits) / 8),
    }


def quantize_signed(codes: np.ndarray, depth: int) -> np.ndarray:
    if depth < MIN_DEPTH or depth > MAX_DEPTH:
        raise ValueError("depth is outside the declared signed-int12 ladder")
    source = np.asarray(codes, dtype=np.int64)
    quantum = 1 << (MAX_DEPTH - depth)
    magnitude = np.abs(source)
    rounded = ((magnitude + quantum // 2) // quantum) * quantum
    quantized = np.where(source < 0, -rounded, rounded)
    maximum_positive = (2047 // quantum) * quantum
    return np.clip(quantized, -2048, maximum_positive).astype(np.int16)


def contribution(jacobian: np.ndarray, errors: np.ndarray) -> float:
    output_error = np.einsum("pjd,pd->pj", jacobian, errors.astype(np.float64), optimize=False)
    return math.sqrt(10.0 * float(np.mean(np.square(output_error))))


def reverse_waterfill(
    codes: np.ndarray,
    jacobian: np.ndarray,
    active_dimensions: np.ndarray,
    tolerances: tuple[float, ...] = TOLERANCES,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Coarsen the least full-J-costly available quantum below each tolerance."""

    if codes.shape != (N, D) or jacobian.shape != (N, POSE_DIMS, D):
        raise ValueError("reverse-waterfill inputs have the wrong shape")
    if active_dimensions.shape != (N, 3):
        raise ValueError("active-dimension map has the wrong shape")
    if tuple(sorted(tolerances)) != tolerances:
        raise ValueError("tolerances must be ordered from tight to loose")

    quantized_by_depth = np.stack([quantize_signed(codes, depth) for depth in range(MIN_DEPTH, MAX_DEPTH + 1)])
    depths = np.full((N, D), MAX_DEPTH, dtype=np.uint8)
    current = codes.copy()
    output_error = np.zeros((N, POSE_DIMS), dtype=np.float64)
    pair_sse = np.sum(np.square(output_error), axis=1)
    active_mask = np.zeros((N, D), dtype=bool)
    active_mask[np.arange(N)[:, None], active_dimensions.astype(np.int64)] = True
    versions = np.zeros(N, dtype=np.int64)
    heap: list[tuple[float, int, int, int, int, int]] = []

    def best_move(pair: int) -> tuple[float, int, int, int, int, int] | None:
        best: tuple[float, int, int] | None = None
        for dim in range(D):
            depth = int(depths[pair, dim])
            if depth <= MIN_DEPTH:
                continue
            new_value = int(quantized_by_depth[depth - 1 - MIN_DEPTH, pair, dim])
            old_value = int(current[pair, dim])
            delta_error = float(new_value - old_value)
            proposed = output_error[pair] + jacobian[pair, :, dim] * delta_error
            damage = float(np.dot(proposed, proposed) - pair_sse[pair])
            key = (damage, int(active_mask[pair, dim]), dim)
            if best is None or key < best:
                best = key
        if best is None:
            return None
        damage, active, dim = best
        return (
            damage,
            active,
            pair,
            dim,
            int(depths[pair, dim]),
            int(versions[pair]),
        )

    for pair in range(N):
        move = best_move(pair)
        if move is not None:
            heapq.heappush(heap, move)

    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    precision_bits_removed = 0
    for tolerance in tolerances:
        while heap:
            damage, active, pair, dim, expected_depth, version = heapq.heappop(heap)
            if version != versions[pair] or expected_depth != int(depths[pair, dim]):
                continue
            proposed_sse = float(pair_sse.sum()) + damage
            proposed_contribution = math.sqrt(10.0 * max(0.0, proposed_sse) / (N * POSE_DIMS))
            if proposed_contribution > tolerance:
                heapq.heappush(heap, (damage, active, pair, dim, expected_depth, version))
                break
            old_value = int(current[pair, dim])
            new_depth = expected_depth - 1
            new_value = int(quantized_by_depth[new_depth - MIN_DEPTH, pair, dim])
            delta_error = float(new_value - old_value)
            output_error[pair] += jacobian[pair, :, dim] * delta_error
            pair_sse[pair] = float(np.dot(output_error[pair], output_error[pair]))
            current[pair, dim] = new_value
            depths[pair, dim] = new_depth
            versions[pair] += 1
            precision_bits_removed += 1
            move = best_move(pair)
            if move is not None:
                heapq.heappush(heap, move)

        candidate_id = f"tol_{tolerance:.6f}".replace(".", "p")
        measured_contribution = math.sqrt(10.0 * float(pair_sse.sum()) / (N * POSE_DIMS))
        direct = contribution(jacobian, current.astype(np.int64) - codes.astype(np.int64))
        if not math.isclose(measured_contribution, direct, rel_tol=0.0, abs_tol=1e-14):
            raise PreproofError("incremental and direct Jacobian contributions differ")
        arrays[f"depth_map__{candidate_id}"] = depths.copy()
        arrays[f"quantized_codes__{candidate_id}"] = current.copy()
        rows.append(
            {
                "candidate_id": candidate_id,
                "tolerance_pose_contribution": tolerance,
                "predicted_induced_pose_contribution": measured_contribution,
                "predicted_induced_d_pose": measured_contribution**2 / 10.0,
                "precision_bits_removed_from_depth12": precision_bits_removed,
                "planning_parent": PLANNING_PARENT,
            }
        )
    return rows, arrays


def pack_depth_nibbles(depths: np.ndarray) -> bytes:
    flat = np.asarray(depths, dtype=np.uint8).reshape(-1)
    if flat.size != N * D or np.any(flat < MIN_DEPTH) or np.any(flat > MAX_DEPTH):
        raise ValueError("depth map is outside the declared ladder")
    if flat.size % 2:
        flat = np.pad(flat, (0, 1))
    return ((flat[0::2] << 4) | flat[1::2]).tobytes()


def unpack_depth_nibbles(payload: bytes) -> np.ndarray:
    packed = np.frombuffer(payload, dtype=np.uint8)
    flat = np.empty(packed.size * 2, dtype=np.uint8)
    flat[0::2] = packed >> 4
    flat[1::2] = packed & 0xF
    result = flat[: N * D].reshape(N, D)
    if np.any(result < MIN_DEPTH) or np.any(result > MAX_DEPTH):
        raise PreproofError("decoded depth map is outside the declared ladder")
    return result


def depth_coder_payloads(depths: np.ndarray) -> dict[str, bytes | str | int]:
    raw = pack_depth_nibbles(depths)
    brotli_payload = brotli.compress(raw, quality=11)
    lzma_payload = lzma.compress(raw, format=lzma.FORMAT_ALONE, preset=9 | lzma.PRESET_EXTREME)
    choices = {"raw": raw, "brotli_q11": brotli_payload, "lzma1": lzma_payload}
    codec_ids = {"raw": 0, "brotli_q11": 1, "lzma1": 2}
    selected_name, selected = min(choices.items(), key=lambda item: (len(item[1]), item[0]))
    header = DEPTH_HEADER.pack(
        DEPTH_MAGIC,
        DEPTH_VERSION,
        codec_ids[selected_name],
        N,
        D,
        len(raw),
        len(selected),
    )
    wire = header + selected
    magic, version, codec_id, frames, dimensions, raw_bytes, payload_bytes = DEPTH_HEADER.unpack_from(wire)
    if (
        magic != DEPTH_MAGIC
        or version != DEPTH_VERSION
        or frames != N
        or dimensions != D
        or raw_bytes != len(raw)
        or payload_bytes != len(selected)
    ):
        raise PreproofError("depth wire header failed parse-back")
    body = wire[DEPTH_HEADER.size :]
    if codec_id == 0:
        decoded = body
    elif codec_id == 1:
        decoded = brotli.decompress(body)
    elif codec_id == 2:
        decoded = lzma.decompress(body, format=lzma.FORMAT_ALONE)
    else:  # pragma: no cover - guarded by the locally constructed header.
        raise PreproofError("unknown depth-map codec")
    if not np.array_equal(unpack_depth_nibbles(decoded), depths):
        raise PreproofError("depth-map wire failed exact parse-back")
    return {
        "raw": raw,
        "brotli_q11": brotli_payload,
        "lzma1": lzma_payload,
        "selected_name": selected_name,
        "selected_codec_id": codec_ids[selected_name],
        "wire": wire,
    }


def retain_payload(root: Path, name: str, payload: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    primary = atomic_bytes(root / name, payload)
    path = Path(name)
    repeat_name = f"{path.stem}.repeat{path.suffix}"
    repeat = atomic_bytes(root / repeat_name, payload)
    if primary["sha256"] != repeat["sha256"] or primary["bytes"] != repeat["bytes"]:
        raise PreproofError(f"deterministic repeat differs for {name}")
    return primary, repeat


def sensitivity_summary(
    jacobian_norm: np.ndarray,
    active_dimensions: np.ndarray,
    codes: np.ndarray,
    baseline_coder: dict[str, Any],
) -> list[dict[str, Any]]:
    active_mask = np.zeros((N, D), dtype=bool)
    active_mask[np.arange(N)[:, None], active_dimensions.astype(np.int64)] = True
    rows = []
    for dim in range(D):
        values = jacobian_norm[:, dim]
        rows.append(
            {
                "dimension": dim,
                "planning_parent": PLANNING_PARENT,
                "j_norm_min": float(values.min()),
                "j_norm_p10": float(np.quantile(values, 0.10)),
                "j_norm_median": float(np.median(values)),
                "j_norm_p90": float(np.quantile(values, 0.90)),
                "j_norm_max": float(values.max()),
                "active_pair_count": int(active_mask[:, dim].sum()),
                "coefficient_abs_median_quanta": float(np.median(np.abs(codes[:, dim]))),
                "coefficient_std_quanta": float(np.std(codes[:, dim].astype(np.float64))),
                "baseline_rice_k": baseline_coder["rice_k_by_dimension"][dim],
                "baseline_rice_bits": baseline_coder["rice_bits_by_dimension"][dim],
            }
        )
    return rows


def error_summary(codes: np.ndarray, candidate: np.ndarray, depths: np.ndarray) -> list[dict[str, Any]]:
    error = candidate.astype(np.int64) - codes.astype(np.int64)
    rows = []
    for dim in range(D):
        absolute = np.abs(error[:, dim])
        rows.append(
            {
                "dimension": dim,
                "max_abs_error_quanta": int(absolute.max()),
                "p95_abs_error_quanta": float(np.quantile(absolute, 0.95)),
                "mean_abs_error_quanta": float(absolute.mean()),
                "changed_pairs": int(np.count_nonzero(absolute)),
                "mean_depth_bits": float(depths[:, dim].mean()),
                "min_depth_bits": int(depths[:, dim].min()),
                "max_depth_bits": int(depths[:, dim].max()),
            }
        )
    return rows


def load_completed_candidate(path: Path, config_sha256: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != "ddm_pz4a_candidate.v1"
        or payload.get("complete") is not True
        or payload.get("config_sha256") != config_sha256
    ):
        raise PreproofError(f"completed candidate receipt is incompatible: {path}")
    records = payload.get("artifacts", {})
    if not isinstance(records, dict) or not records or not all(verified_record(record) for record in records.values()):
        raise PreproofError(f"completed candidate payload custody failed: {path}")
    return payload


def materialize_candidate(
    output: Path,
    allocation_row: dict[str, Any],
    depths: np.ndarray,
    candidate_codes: np.ndarray,
    source_codes: np.ndarray,
    source_carrier: bytes,
    source_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    source_component: bytes,
    shipped_carrier_stream: bytes,
    codec: ModuleType,
    config_sha256: str,
) -> dict[str, Any]:
    candidate_id = allocation_row["candidate_id"]
    root = output / "candidates" / candidate_id
    prior = load_completed_candidate(root / "receipt.json", config_sha256)
    if prior is not None:
        return prior

    basis_scales, basis_codes, coefficient_scales, _ = source_state
    encoded = encoded_from_signed_codes(candidate_codes)
    carrier = codec.encode_compact_carrier(basis_scales, basis_codes, coefficient_scales, encoded)
    decoded = codec.decode_compact_carrier(carrier, basis_count=D * 3 * 24 * 32, frames=N, dimensions=D)[3]
    if not np.array_equal(signed_codes_from_encoded(decoded), candidate_codes):
        raise PreproofError(f"{candidate_id}: CPR1 parse-back changed coefficients")
    component = split_coefficient_component(carrier, codec)
    carrier_q9 = brotli.compress(carrier, quality=9)
    depth_payloads = depth_coder_payloads(depths)
    depth_wire = depth_payloads["wire"]
    assert isinstance(depth_wire, bytes)

    artifacts: dict[str, Any] = {}
    arrays = {
        "coefficients.int16.npy": candidate_codes.astype(np.int16),
        "depth_map.uint8.npy": depths.astype(np.uint8),
    }
    for name, array in arrays.items():
        artifacts[name] = atomic_npy(root / name, array)
        repeat_path = root / name.replace(".npy", ".repeat.npy")
        artifacts[repeat_path.name] = atomic_npy(repeat_path, array)
        if artifacts[name]["sha256"] != artifacts[repeat_path.name]["sha256"]:
            raise PreproofError(f"{candidate_id}: NPY repeat differs for {name}")

    byte_payloads = {
        "coefficient_component.rice": component,
        "carrier.cpr1": carrier,
        "depth_map.nibbles": depth_payloads["raw"],
        "depth_map.q11.br": depth_payloads["brotli_q11"],
        "depth_map.lzma1": depth_payloads["lzma1"],
        "depth_map.pz4d": depth_wire,
    }
    for name, value in byte_payloads.items():
        assert isinstance(value, bytes)
        primary, repeat = retain_payload(root, name, value)
        artifacts[name] = primary
        artifacts[Path(repeat["path"]).name] = repeat
    carrier_q9_primary, carrier_q9_repeat = retain_payload(root, "carrier.q9.br", carrier_q9)
    artifacts["carrier.q9.br"] = carrier_q9_primary
    artifacts[Path(carrier_q9_repeat["path"]).name] = carrier_q9_repeat

    coder = coefficient_coder_stats(carrier, codec)
    inner_gross = len(source_component) - len(component)
    joint_gross = len(shipped_carrier_stream) - len(carrier_q9)
    metadata_bytes = len(depth_wire)
    raw_int12_bits_saved = int(np.sum(MAX_DEPTH - depths.astype(np.int64)))
    raw_int16_bits_saved = int(np.sum(16 - depths.astype(np.int64)))
    receipt = {
        "schema": "ddm_pz4a_candidate.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": AXIS,
        "score_claim": False,
        "scorer_run": False,
        "planning_band": True,
        "planning_parent": PLANNING_PARENT,
        "config_sha256": config_sha256,
        **allocation_row,
        "coded_sizes": {
            "baseline_inner_cpr1_coefficient_component_bytes": len(source_component),
            "candidate_inner_cpr1_coefficient_component_bytes": len(component),
            "inner_gross_saving_bytes": inner_gross,
            "baseline_joint_carrier_brotli_q9_bytes": len(shipped_carrier_stream),
            "candidate_joint_carrier_brotli_q9_bytes": len(carrier_q9),
            "joint_gross_saving_bytes": joint_gross,
            "depth_metadata_wire_bytes": metadata_bytes,
            "inner_net_saving_after_metadata_bytes": inner_gross - metadata_bytes,
            "joint_net_saving_after_metadata_bytes": joint_gross - metadata_bytes,
        },
        "raw_domain_projection": {
            "logical_int12_bits_saved_before_metadata": raw_int12_bits_saved,
            "logical_int12_bytes_saved_before_metadata": raw_int12_bits_saved / 8.0,
            "physical_int16_bits_saved_before_metadata": raw_int16_bits_saved,
            "physical_int16_bytes_saved_before_metadata": raw_int16_bits_saved / 8.0,
            "warning": "projection only; the retained exact CPR1/Brotli payloads govern",
        },
        "depth_metadata": {
            "raw_nibble_bytes": len(depth_payloads["raw"]),
            "brotli_q11_bytes": len(depth_payloads["brotli_q11"]),
            "lzma1_bytes": len(depth_payloads["lzma1"]),
            "selected": depth_payloads["selected_name"],
            "wire_bytes_including_header": metadata_bytes,
        },
        "depth_histogram": {
            str(depth): int(np.count_nonzero(depths == depth)) for depth in range(MIN_DEPTH, MAX_DEPTH + 1)
        },
        "mean_depth_bits": float(depths.mean()),
        "active_dimension_mean_depth_bits": None,
        "coefficient_error_by_dimension": error_summary(source_codes, candidate_codes, depths),
        "coefficient_coder": coder,
        "artifacts": artifacts,
        "source_carrier_sha256": sha256_bytes(source_carrier),
    }
    atomic_json(root / "receipt.json", receipt)
    return receipt


def retain_baseline(
    output: Path,
    codes: np.ndarray,
    carrier: bytes,
    component: bytes,
    shipped_stream: bytes,
    reproduced_stream: bytes,
    coder: dict[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    root = output / "baseline"
    prior = load_completed_candidate(root / "receipt.json", config_sha256)
    if prior is not None:
        return prior
    raw = codes.astype("<i2").tobytes(order="C")
    raw_q11 = brotli.compress(raw, quality=11)
    raw_lzma1 = lzma.compress(raw, format=lzma.FORMAT_ALONE, preset=9 | lzma.PRESET_EXTREME)
    artifacts: dict[str, Any] = {}
    # Keep direct binding-to-persister calls visible to the repository's P0
    # static retention proof as well as to the runtime artifact audit below.
    raw_primary, raw_repeat = retain_payload(root, "source_coefficients.int16.raw", raw)
    raw_q11_primary, raw_q11_repeat = retain_payload(root, "source_coefficients.q11.br", raw_q11)
    raw_lzma1_primary, raw_lzma1_repeat = retain_payload(root, "source_coefficients.lzma1", raw_lzma1)
    for name, primary, repeat in (
        ("source_coefficients.int16.raw", raw_primary, raw_repeat),
        ("source_coefficients.q11.br", raw_q11_primary, raw_q11_repeat),
        ("source_coefficients.lzma1", raw_lzma1_primary, raw_lzma1_repeat),
    ):
        artifacts[name] = primary
        artifacts[Path(repeat["path"]).name] = repeat

    payloads = (
        ("source_coefficients.int16.npy", COEFFICIENTS.read_bytes()),
        ("coefficient_component.rice", component),
        ("carrier.cpr1", carrier),
        ("carrier.shipped.q9.br", shipped_stream),
        ("carrier.reproduced.q9.br", reproduced_stream),
    )
    for name, payload in payloads:
        primary, repeat = retain_payload(root, name, payload)
        artifacts[name] = primary
        artifacts[Path(repeat["path"]).name] = repeat
    receipt = {
        "schema": "ddm_pz4a_candidate.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "candidate_id": "baseline_shipped_pass03",
        "axis": AXIS,
        "score_claim": False,
        "scorer_run": False,
        "planning_band": True,
        "planning_parent": PLANNING_PARENT,
        "config_sha256": config_sha256,
        "coded_sizes": {
            "npy_file_bytes": COEFFICIENTS.stat().st_size,
            "logical_int16_raw_bytes": len(raw),
            "raw_brotli_q11_proxy_bytes": len(raw_q11),
            "raw_lzma1_proxy_bytes": len(raw_lzma1),
            "inner_cpr1_coefficient_component_bytes": len(component),
            "whole_carrier_raw_bytes": len(carrier),
            "joint_carrier_brotli_q9_bytes": len(shipped_stream),
        },
        "coefficient_coder": coder,
        "artifacts": artifacts,
    }
    atomic_json(root / "receipt.json", receipt)
    return receipt


def audit_result(result: dict[str, Any]) -> None:
    if not all(verified_record(record) for record in result["allocation_checkpoint"].values()):
        raise PreproofError("final audit failed for allocation checkpoint")
    for section in ("baseline",):
        records = result[section]["artifacts"]
        if not all(verified_record(record) for record in records.values()):
            raise PreproofError(f"final audit failed for {section}")
    for row in result["rows"]:
        if not all(verified_record(record) for record in row["artifacts"].values()):
            raise PreproofError(f"final audit failed for {row['candidate_id']}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = require_output_root(args.output)
    if args.resume_from.resolve() != (output / "state.json").resolve():
        raise PreproofError("--resume-from must name OUTPUT/state.json for this arm")
    inputs = verify_inputs()
    config = {
        "schema": "ddm_pz4a_config.v1",
        "runner_sha256": sha256_file(Path(__file__)),
        "input_sha256": {name: record["sha256"] for name, record in inputs.items()},
        "tolerances": list(TOLERANCES),
        "depth_range": [MIN_DEPTH, MAX_DEPTH],
        "allocation": "dynamic full-J reverse-waterfill by least current marginal MSE damage per coarsening bit",
        "coefficient_coder": "exact shipped CPR1 per-dimension Rice",
        "joint_cell_coder": "Brotli 1.2.0 quality 9 over the full held-basis carrier",
        "metadata_race": ["raw_nibbles", "Brotli 1.2.0 q11", "LZMA1 extreme"],
        "planning_parent": PLANNING_PARENT,
    }
    config_sha256 = canonical_hash(config)
    state_path = args.resume_from
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if state.get("config_sha256") != config_sha256:
            raise PreproofError("resume state config differs from the current run")
    else:
        state = {
            "schema": "ddm_pz4a_resume.v1",
            "config_sha256": config_sha256,
            "completed_stages": [],
            "created_at_utc": utc_now(),
        }
        atomic_json(state_path, state)

    preflight = {
        "schema": "ddm_pz4a_preflight.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": AXIS,
        "score_claim": False,
        "scorer_run": False,
        "output": str(output),
        "free_bytes": shutil.disk_usage(output).free,
        "config": config,
        "config_sha256": config_sha256,
        "invocation": {
            "argv": [str(item) for item in sys.argv],
            "python": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "brotli_version": getattr(brotli, "__version__", "unknown"),
        },
        "inputs": inputs,
        "parent_freshness": {
            "producer_parent_sha256": PRODUCER_PARENT_SHA256,
            "consumer_parent_sha256": CONSUMER_PARENT_SHA256,
            "freshness_ok": False,
            "admissibility": "PLANNING-BAND_ONLY",
        },
    }
    atomic_json(output / "00_PREFLIGHT.json", preflight)

    codec = import_codec()
    source_carrier = CARRIER.read_bytes()
    shipped_carrier_stream, parsed_carrier = parse_archive_carrier_stream(ARCHIVE)
    if parsed_carrier != source_carrier:
        raise PreproofError("selected archive carrier differs from rehearsal custody")
    reproduced_carrier_stream = brotli.compress(source_carrier, quality=9)
    if reproduced_carrier_stream != shipped_carrier_stream:
        raise PreproofError("Brotli 1.2.0 q9 did not reproduce the shipped carrier cell")

    source_state = codec.decode_compact_carrier(source_carrier, basis_count=D * 3 * 24 * 32, frames=N, dimensions=D)
    rebuilt = codec.encode_compact_carrier(*source_state)
    if rebuilt != source_carrier:
        raise PreproofError("shipped CPR1 carrier is not byte-stable under its codec")
    source_codes = signed_codes_from_encoded(source_state[3])
    npy_codes = np.load(COEFFICIENTS, allow_pickle=False)
    if npy_codes.shape != (N, D) or npy_codes.dtype != np.int16:
        raise PreproofError("coefficient NPY shape or dtype changed")
    if not np.array_equal(source_codes, npy_codes):
        raise PreproofError("coefficient NPY differs from the selected carrier parse")
    source_component = split_coefficient_component(source_carrier, codec)
    baseline_coder = coefficient_coder_stats(source_carrier, codec)

    with np.load(SENSITIVITY, allow_pickle=False) as payload:
        jacobian = payload["jacobian_6x12"].astype(np.float64)
        jacobian_norm = payload["jacobian_norm"].astype(np.float64)
        active_dimensions = payload["active_dimensions"].astype(np.int8)
    if (
        jacobian.shape != (N, POSE_DIMS, D)
        or jacobian_norm.shape != (N, D)
        or active_dimensions.shape != (N, 3)
        or not np.all(np.isfinite(jacobian))
    ):
        raise PreproofError("sensitivity map content is invalid")
    if not np.allclose(np.linalg.norm(jacobian, axis=1), jacobian_norm, rtol=0, atol=1e-15):
        raise PreproofError("sensitivity norm cache differs from its Jacobian")

    allocation_path = output / "stages/01_allocations.npz"
    allocation_receipt_path = output / "stages/01_allocations.json"
    if allocation_path.is_file() and allocation_receipt_path.is_file():
        allocation_receipt = json.loads(allocation_receipt_path.read_text())
        if allocation_receipt.get("config_sha256") != config_sha256 or allocation_receipt.get("payload", {}).get(
            "sha256"
        ) != sha256_file(allocation_path):
            raise PreproofError("allocation resume checkpoint changed")
        allocation_rows = allocation_receipt["rows"]
        with np.load(allocation_path, allow_pickle=False) as stored:
            allocation_arrays = {name: stored[name].copy() for name in stored.files}
    else:
        allocation_rows, allocation_arrays = reverse_waterfill(source_codes, jacobian, active_dimensions)
        allocation_record = atomic_npz(allocation_path, **allocation_arrays)
        allocation_receipt = {
            "schema": "ddm_pz4a_allocations.v1",
            "complete": True,
            "written_at_utc": utc_now(),
            "config_sha256": config_sha256,
            "planning_parent": PLANNING_PARENT,
            "payload": allocation_record,
            "rows": allocation_rows,
        }
        atomic_json(allocation_receipt_path, allocation_receipt)

    baseline = retain_baseline(
        output,
        source_codes,
        source_carrier,
        source_component,
        shipped_carrier_stream,
        reproduced_carrier_stream,
        baseline_coder,
        config_sha256,
    )
    active_mask = np.zeros((N, D), dtype=bool)
    active_mask[np.arange(N)[:, None], active_dimensions.astype(np.int64)] = True
    rows = []
    for allocation_row in allocation_rows:
        candidate_id = allocation_row["candidate_id"]
        depths = allocation_arrays[f"depth_map__{candidate_id}"]
        candidate_codes = allocation_arrays[f"quantized_codes__{candidate_id}"]
        row = materialize_candidate(
            output,
            allocation_row,
            depths,
            candidate_codes,
            source_codes,
            source_carrier,
            source_state,
            source_component,
            shipped_carrier_stream,
            codec,
            config_sha256,
        )
        row["active_dimension_mean_depth_bits"] = float(depths[active_mask].mean())
        row["inactive_dimension_mean_depth_bits"] = float(depths[~active_mask].mean())
        atomic_json(output / "candidates" / candidate_id / "receipt.json", row)
        rows.append(row)
        state["completed_stages"] = [
            "preflight",
            "allocations",
            "baseline",
            *[item["candidate_id"] for item in rows],
        ]
        state["updated_at_utc"] = utc_now()
        atomic_json(state_path, state)

    best_inner = max(rows, key=lambda row: row["coded_sizes"]["inner_net_saving_after_metadata_bytes"])
    best_joint = max(rows, key=lambda row: row["coded_sizes"]["joint_net_saving_after_metadata_bytes"])
    best_inner_bytes = best_inner["coded_sizes"]["inner_net_saving_after_metadata_bytes"]
    best_joint_bytes = best_joint["coded_sizes"]["joint_net_saving_after_metadata_bytes"]
    if best_joint_bytes >= 2_000:
        verdict = "CLEARS_2000B_GATE"
    elif (best_inner_bytes >= 2_000) != (best_joint_bytes >= 2_000):
        verdict = "NEAR_GATE_SIGN_UNCERTAIN"
    else:
        verdict = "REFUTED"

    result = {
        "schema": "ddm_pz4a_final.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": AXIS,
        "score_claim": False,
        "scorer_run": False,
        "modal_dispatch": False,
        "frontier_moved": False,
        "config": config,
        "config_sha256": config_sha256,
        "planning_parent": PLANNING_PARENT,
        "parent_freshness": preflight["parent_freshness"],
        "allocation_checkpoint": {
            "payload": file_record(allocation_path),
            "receipt": file_record(allocation_receipt_path),
        },
        "baseline": baseline,
        "sensitivity_by_dimension": sensitivity_summary(jacobian_norm, active_dimensions, source_codes, baseline_coder),
        "rows": rows,
        "gate": {
            "required_net_saving_bytes": 2_000,
            "governing_boundary": "exact held-basis whole-carrier Brotli-q9 delta minus exact depth-map wire",
            "verdict": verdict,
            "best_joint_candidate_id": best_joint["candidate_id"],
            "best_joint_net_saving_bytes": best_joint_bytes,
            "best_inner_candidate_id": best_inner["candidate_id"],
            "best_inner_net_saving_bytes": best_inner_bytes,
            "mechanism_test": (
                "exactly tests whether representation-changing heterogeneous precision escapes "
                "the shipped per-dimension Rice coder and whole-carrier Brotli cell after counted metadata"
            ),
        },
        "not_measured": [
            "PoseNet or SegNet response",
            "compensation-stage recovery",
            "terminal-parent sensitivity",
            "receiver-framed complete archive with depth metadata",
            "exact contest score",
        ],
    }
    audit_result(result)
    result_record = atomic_json(output / "FINAL_RESULT.json", result)
    audit = {
        "schema": "ddm_pz4a_retention_audit.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "result": result_record,
        "candidate_count": len(rows),
        "all_candidate_payloads_verified": True,
        "baseline_payloads_verified": True,
    }
    atomic_json(output / "RETENTION_AUDIT.json", audit)
    state["completed_stages"] = list(dict.fromkeys([*state["completed_stages"], "final_audit"]))
    state["complete"] = True
    state["updated_at_utc"] = utc_now()
    atomic_json(state_path, state)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--resume-from", type=Path, default=DEFAULT_OUTPUT / "state.json")
    return value


def main() -> int:
    result = run(parser().parse_args())
    print(json.dumps({"verdict": result["gate"]["verdict"], "gate": result["gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
