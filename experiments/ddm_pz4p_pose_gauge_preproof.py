#!/usr/bin/env python3
"""Scorer-free learned pose-gauge quantization pre-proof for pz4.

The measured object is deliberately narrower than a submission candidate.  It
compresses the banked 600x6 PoseNet outputs produced by the shipped LC2 CPR1
carrier, decodes them through a compact low-rank quantized gauge, and measures
output reconstruction MSE.  It also puts every gauge payload through LC2's
exact Brotli-q9 carrier slot and deterministic outer-container path to obtain a
real-coder rate envelope.  The resulting envelope is not receiver-closed: an
actual pz4 arm must teach a renderer to consume the decoded gauge before any
rendered d_pose or score claim is possible.

No scorer is imported or executed.  Every candidate payload, decoded output,
coder stream, rate envelope, training checkpoint, and repeat is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import shutil
import struct
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PS135_RUNNER = REPO / "experiments/ddm_ps135_pose_resolve.py"
TEST_PATH = REPO / "experiments/tests/test_ddm_pz4p_pose_gauge_preproof.py"
CHARTER = REPO / ".omx/research/charters/ddm_pz4p_pose_gauge_preproof_charter_20260811.md"
COMMON_CONTRACT = REPO / ".omx/tmp/codex_runs/_common_contract.md"

LC2_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_lc2_20260810/submission/archive.zip")
LC2_CARRIER = Path("/Volumes/VertigoDataTier/pact/ddm_lc2_20260810/retained/inputs/carrier.raw")
LC2_OUTPUTS = Path("/Volumes/APDataStore/pact/ddm_ps135_20260810/baseline_outputs/lc2_native/pose_outputs.float32.npy")
LC2_OUTPUT_RECEIPT = LC2_OUTPUTS.with_name("receipt.json")
DEFAULT_OUT = Path("/Volumes/VertigoDataTier/pact/ddm_pz4p_20260811/preproof_v3")

LC2_ARCHIVE_BYTES = 187_226
LC2_ARCHIVE_SHA256 = "f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45"
LC2_CARRIER_BYTES = 23_054
LC2_CARRIER_SHA256 = "a05d0985ca5a8d5110bd5bf5be39f238c6f89640b8a8bb888a3e1269bdf636e4"
LC2_OUTPUT_BYTES = 14_528
LC2_OUTPUT_SHA256 = "23319e2f0406040ee5d9e904daacc1017f8da44a02e7c259055e72c937515312"
LC2_OUTPUT_RECEIPT_SHA256 = "4deb10bd46296f893e85bda024e572128b0abb303286b73144432c05560920fa"
LC2_OUTPUT_SEMANTIC_SHA256 = "e80dcb0b4ce6afb7ac74db91dc29ce9cbbce09acbd0058f9450364a40f4ebfe2"
CHARTER_SHA256 = "07bad21719c11788601d77d41ced0e4de744b057159cbd6603a6f276164bc196"
COMMON_CONTRACT_SHA256 = "eeae9e0035582e6bdd65fd837e4aa35a65e064fd09900b9c212d41ac02086771"

N = 600
POSE_DIMS = 6
PK2_POSE_MARGINAL_BYTES = 23_384
MSE_GATE = 2.5e-6
SAVINGS_GATE_BYTES = 2_000
AXIS = "[macOS-CPU scorer-free banked-output MSE + exact real-coder rate envelope]"
SCORE_CLAIM = False
MAGIC = b"PGQ1"
VERSION = 1
QAT_ROUNDS = 8
SCALE_MULTIPLIERS = (0.875, 0.9375, 1.0, 1.0625, 1.125)
DEPTHS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15)
RANKS = (1, 2, 3, 4, 5, 6)
CELL_MODES: dict[str, tuple[int, bool, int]] = {
    "global": (N, False, 0),
    "per_rank": (N, True, 1),
    "block100_rank": (100, True, 2),
    "block50_rank": (50, True, 3),
    "block25_rank": (25, True, 4),
}
MIN_FREE_BYTES = 1_000_000_000
RUNNER_SOURCE_SHA256_AT_IMPORT = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class PreproofError(RuntimeError):
    """Raised when custody, determinism, retention, or schema closure fails."""


@dataclass(frozen=True)
class GaugeConfig:
    rank: int
    depth: int
    cell_mode: str

    def __post_init__(self) -> None:
        if self.rank < 1 or self.rank > POSE_DIMS:
            raise PreproofError(f"gauge rank must be in [1, {POSE_DIMS}]")
        if self.depth < 1 or self.depth >= 16:
            raise PreproofError("gauge depth must be sub-int16 and positive")
        if self.cell_mode not in CELL_MODES:
            raise PreproofError(f"unknown cell mode: {self.cell_mode}")

    @property
    def candidate_id(self) -> str:
        return f"r{self.rank}_b{self.depth}_{self.cell_mode}"


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def assert_runner_source_unchanged() -> None:
    actual = sha256_file(Path(__file__).resolve())
    if actual != RUNNER_SOURCE_SHA256_AT_IMPORT:
        raise PreproofError(f"runner source changed during execution: {actual} != {RUNNER_SOURCE_SHA256_AT_IMPORT}")


def bytes_record(path: Path, payload: bytes) -> dict[str, object]:
    atomic_bytes(path, payload)
    return file_record(path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise PreproofError(f"refusing symlinked atomic-write target: {path}")
    temporary.write_bytes(payload)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def atomic_numpy(path: Path, value: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise PreproofError(f"refusing symlinked atomic-write target: {path}")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_record(path)


def atomic_npz(path: Path, **values: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise PreproofError(f"refusing symlinked atomic-write target: {path}")
    with temporary.open("wb") as stream:
        np.savez(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_record(path)


def validate_output_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = Path("/Volumes/VertigoDataTier/pact/ddm_pz4p_20260811").resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise PreproofError(f"output must be a versioned child of {allowed}")
    if Path("/tmp") in resolved.parents or resolved == Path("/tmp"):
        raise PreproofError("durable evidence may not use /tmp")
    return resolved


def require_storage(root: Path) -> dict[str, int]:
    root.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root.parent)
    if usage.free < MIN_FREE_BYTES:
        raise PreproofError(f"insufficient retained-evidence space: {usage.free} < {MIN_FREE_BYTES}")
    return {"total": usage.total, "used": usage.used, "free": usage.free}


def verify_pinned(path: Path, *, size: int | None, digest: str, label: str) -> None:
    if not path.is_file():
        raise PreproofError(f"missing pinned {label}: {path}")
    if size is not None and path.stat().st_size != size:
        raise PreproofError(f"pinned {label} size changed: {path}")
    actual = sha256_file(path)
    if actual != digest:
        raise PreproofError(f"pinned {label} SHA-256 changed: {actual}")


def verify_custody() -> tuple[np.ndarray, dict[str, object]]:
    pins = (
        (LC2_ARCHIVE, LC2_ARCHIVE_BYTES, LC2_ARCHIVE_SHA256, "LC2 archive"),
        (LC2_CARRIER, LC2_CARRIER_BYTES, LC2_CARRIER_SHA256, "LC2 carrier"),
        (LC2_OUTPUTS, LC2_OUTPUT_BYTES, LC2_OUTPUT_SHA256, "LC2 PoseNet output bank"),
        (
            LC2_OUTPUT_RECEIPT,
            None,
            LC2_OUTPUT_RECEIPT_SHA256,
            "LC2 output receipt",
        ),
        (CHARTER, None, CHARTER_SHA256, "pz4p charter"),
        (COMMON_CONTRACT, None, COMMON_CONTRACT_SHA256, "common contract"),
    )
    for path, size, digest, label in pins:
        verify_pinned(path, size=size, digest=digest, label=label)
    receipt = json.loads(LC2_OUTPUT_RECEIPT.read_text(encoding="utf-8"))
    if (
        receipt.get("complete") is not True
        or receipt.get("pair_count") != N
        or receipt.get("pose_outputs") != file_record(LC2_OUTPUTS)
        or receipt.get("payloads_retained") is not True
        or receipt.get("score_claim") is not False
    ):
        raise PreproofError("LC2 output receipt does not bind the complete retained bank")
    outputs = np.load(LC2_OUTPUTS, allow_pickle=False)
    if outputs.dtype != np.float32 or outputs.shape != (N, POSE_DIMS):
        raise PreproofError("LC2 output bank has the wrong dtype or shape")
    if not np.all(np.isfinite(outputs)):
        raise PreproofError("LC2 output bank contains non-finite values")
    if sha256_array(outputs) != LC2_OUTPUT_SEMANTIC_SHA256:
        raise PreproofError("LC2 output bank semantic SHA-256 changed")
    return outputs, {
        "archive": file_record(LC2_ARCHIVE),
        "carrier": file_record(LC2_CARRIER),
        "pose_outputs": file_record(LC2_OUTPUTS),
        "pose_outputs_semantic_sha256": LC2_OUTPUT_SEMANTIC_SHA256,
        "pose_output_receipt": file_record(LC2_OUTPUT_RECEIPT),
        "charter": file_record(CHARTER),
        "common_contract": file_record(COMMON_CONTRACT),
    }


def load_ps135() -> Any:
    spec = importlib.util.spec_from_file_location("ddm_ps135_for_pz4p", PS135_RUNNER)
    if spec is None or spec.loader is None:
        raise PreproofError(f"cannot import exact LC2 coder path from {PS135_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bit_pack_unsigned(values: np.ndarray, depth: int) -> bytes:
    flat = np.asarray(values, dtype=np.uint32).reshape(-1)
    limit = 1 << depth
    if depth < 1 or depth > 31 or np.any(flat >= limit):
        raise PreproofError("unsigned values exceed the requested bit depth")
    output = bytearray((len(flat) * depth + 7) // 8)
    bit_index = 0
    for value in flat.tolist():
        remaining = depth
        integer = int(value)
        while remaining:
            byte_index = bit_index >> 3
            offset = bit_index & 7
            take = min(8 - offset, remaining)
            mask = (1 << take) - 1
            output[byte_index] |= (integer & mask) << offset
            integer >>= take
            bit_index += take
            remaining -= take
    return bytes(output)


def bit_unpack_unsigned(payload: bytes, count: int, depth: int) -> np.ndarray:
    expected = (count * depth + 7) // 8
    if len(payload) != expected:
        raise PreproofError(f"packed-code length differs: {len(payload)} != {expected}")
    output = np.empty(count, dtype=np.uint32)
    bit_index = 0
    for index in range(count):
        remaining = depth
        shift = 0
        value = 0
        while remaining:
            byte_index = bit_index >> 3
            offset = bit_index & 7
            take = min(8 - offset, remaining)
            mask = (1 << take) - 1
            value |= ((payload[byte_index] >> offset) & mask) << shift
            bit_index += take
            shift += take
            remaining -= take
        output[index] = value
    return output


def mode_geometry(config: GaugeConfig) -> tuple[int, bool, int, np.ndarray, int]:
    if config.cell_mode not in CELL_MODES:
        raise PreproofError(f"unknown cell mode: {config.cell_mode}")
    block_rows, per_rank, mode_id = CELL_MODES[config.cell_mode]
    groups = np.arange(N, dtype=np.int64) // block_rows
    group_count = int(groups.max()) + 1
    return block_rows, per_rank, mode_id, groups, group_count


def initial_scales(coefficients: np.ndarray, config: GaugeConfig) -> np.ndarray:
    block_rows, per_rank, _, groups, group_count = mode_geometry(config)
    del block_rows
    qmax = (1 << (config.depth - 1)) - 1
    columns = config.rank if per_rank else 1
    scales = np.empty((group_count, columns), dtype=np.float64)
    for group in range(group_count):
        cell = np.abs(coefficients[groups == group])
        maxima = np.max(cell, axis=0) if per_rank else np.asarray([np.max(cell)])
        scales[group] = np.maximum(maxima / qmax, np.finfo(np.float64).tiny)
    return scales


def expanded_scales(scales: np.ndarray, config: GaugeConfig) -> np.ndarray:
    _, per_rank, _, groups, _ = mode_geometry(config)
    selected = scales[groups]
    if not per_rank:
        selected = np.broadcast_to(selected, (N, config.rank))
    return np.asarray(selected, dtype=np.float64)


def nearest_codes(coefficients: np.ndarray, scales: np.ndarray, config: GaugeConfig) -> np.ndarray:
    qmax = (1 << (config.depth - 1)) - 1
    rows = expanded_scales(scales, config)
    return np.clip(np.rint(coefficients / rows), -qmax, qmax).astype(np.int32)


def compensate(
    codes: np.ndarray,
    scales: np.ndarray,
    config: GaugeConfig,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    dequantized = codes.astype(np.float64) * expanded_scales(scales, config)
    design = np.concatenate((np.ones((N, 1), dtype=np.float64), dequantized), axis=1)
    weights = np.linalg.lstsq(design, reference.astype(np.float64), rcond=1e-12)[0]
    reconstruction = np.einsum("ij,jk->ik", design, weights, optimize=False)
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(reconstruction)):
        raise PreproofError("compensation solve produced non-finite values")
    mse = float(
        np.mean(
            (reconstruction - reference.astype(np.float64)) ** 2,
            dtype=np.float64,
        )
    )
    return weights, reconstruction, mse


def optimize_rounding(
    coefficients: np.ndarray,
    scales: np.ndarray,
    config: GaugeConfig,
    reference: np.ndarray,
    checkpoint_root: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    list[dict[str, object]],
]:
    """Hard quantizer-in-loop fitting with exact PR133-style compensation.

    Each round uses only decoded quantized coefficients.  It (1) solves the
    compensated output map, (2) learns cell scales by a bounded discrete search,
    and (3) learns floor/ceil assignments by exact enumeration of all at-most-64
    rank combinations per pair.  The best hard state is retained, so an
    optimization round cannot silently replace a better initialization.
    """

    qmax = (1 << (config.depth - 1)) - 1
    _, per_rank, _, groups, group_count = mode_geometry(config)
    codes = nearest_codes(coefficients, scales, config)
    weights, reconstruction, mse = compensate(codes, scales, config, reference)
    best = (codes.copy(), scales.copy(), weights.copy(), reconstruction.copy(), mse)
    rows: list[dict[str, object]] = []
    combinations = np.asarray(list(itertools.product((0, 1), repeat=config.rank)), dtype=np.int32)

    for round_index in range(1, QAT_ROUNDS + 1):
        weights, _, _ = compensate(codes, scales, config, reference)
        for group in range(group_count):
            row_mask = groups == group
            columns: Iterable[int] = range(config.rank) if per_rank else (0,)
            for column in columns:
                old_scale = float(scales[group, column])
                best_local = (math.inf, old_scale, codes[row_mask].copy())
                for multiplier in SCALE_MULTIPLIERS:
                    proposal_scales = scales.copy()
                    proposal_scales[group, column] = old_scale * multiplier
                    local_scales = _expanded_local_scales(proposal_scales, config, group, int(row_mask.sum()))
                    proposal_codes = np.clip(
                        np.rint(coefficients[row_mask] / local_scales),
                        -qmax,
                        qmax,
                    ).astype(np.int32)
                    candidate_codes = codes[row_mask].copy()
                    if per_rank:
                        candidate_codes[:, column] = proposal_codes[:, column]
                    else:
                        candidate_codes = proposal_codes
                    design = np.concatenate(
                        (
                            np.ones((int(row_mask.sum()), 1), dtype=np.float64),
                            candidate_codes.astype(np.float64) * local_scales,
                        ),
                        axis=1,
                    )
                    prediction = np.einsum("ij,jk->ik", design, weights, optimize=False)
                    error = float(np.mean((prediction - reference[row_mask]) ** 2))
                    if error < best_local[0]:
                        best_local = (error, old_scale * multiplier, candidate_codes)
                scales[group, column] = best_local[1]
                codes[row_mask] = best_local[2]

        scale_rows = expanded_scales(scales, config)
        normalized = coefficients / scale_rows
        lower = np.clip(np.floor(normalized), -qmax, qmax).astype(np.int32)
        upper = np.clip(lower + 1, -qmax, qmax).astype(np.int32)
        for row in range(N):
            candidates = lower[row] + combinations * (upper[row] - lower[row])
            candidate_design = np.concatenate(
                (
                    np.ones((len(candidates), 1), dtype=np.float64),
                    candidates.astype(np.float64) * scale_rows[row],
                ),
                axis=1,
            )
            predictions = np.einsum("ij,jk->ik", candidate_design, weights, optimize=False)
            errors = np.mean((predictions - reference[row]) ** 2, axis=1)
            codes[row] = candidates[int(np.argmin(errors))]

        weights, reconstruction, mse = compensate(codes, scales, config, reference)
        checkpoint = atomic_npz(
            checkpoint_root / f"qat_round_{round_index:02d}.npz",
            codes=codes.astype(np.int16),
            scales=scales.astype(np.float32),
            compensation=weights.astype(np.float32),
            reconstruction=reconstruction.astype(np.float32),
        )
        rows.append(
            {
                "round": round_index,
                "hard_quantized_float64_mse_before_wire_rounding": mse,
                "checkpoint": checkpoint,
            }
        )
        if mse < best[4]:
            best = (
                codes.copy(),
                scales.copy(),
                weights.copy(),
                reconstruction.copy(),
                mse,
            )
    return best[0], best[1], best[2], best[3], best[4], rows


def _local_scales(scales: np.ndarray, config: GaugeConfig, group: int, rows: int) -> np.ndarray:
    del rows
    _, per_rank, _, _, _ = mode_geometry(config)
    value = scales[group : group + 1]
    if not per_rank:
        value = np.broadcast_to(value, (1, config.rank)).copy()
    return value


def _expanded_local_scales(scales: np.ndarray, config: GaugeConfig, group: int, rows: int) -> np.ndarray:
    value = _local_scales(scales, config, group, rows)
    return np.broadcast_to(value, (rows, config.rank))


def encode_gauge(
    config: GaugeConfig,
    codes: np.ndarray,
    scales: np.ndarray,
    compensation_weights: np.ndarray,
) -> bytes:
    block_rows, per_rank, mode_id, _, group_count = mode_geometry(config)
    scale_columns = config.rank if per_rank else 1
    if codes.shape != (N, config.rank) or codes.dtype.kind not in "iu":
        raise PreproofError("gauge codes have the wrong shape or dtype")
    if scales.shape != (group_count, scale_columns):
        raise PreproofError("gauge scale table has the wrong shape")
    if compensation_weights.shape != (config.rank + 1, POSE_DIMS):
        raise PreproofError("gauge compensation matrix has the wrong shape")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0) or not np.all(np.isfinite(compensation_weights)):
        raise PreproofError("gauge scales or compensation values are invalid")
    qmax = (1 << (config.depth - 1)) - 1
    if np.any(codes < -qmax) or np.any(codes > qmax):
        raise PreproofError("gauge signed codes exceed their bit depth")
    unsigned = codes.astype(np.int64) + qmax
    header = struct.pack(
        "<4sBBBBHHHH",
        MAGIC,
        VERSION,
        config.depth,
        config.rank,
        mode_id,
        N,
        POSE_DIMS,
        block_rows,
        scale_columns,
    )
    return b"".join(
        (
            header,
            np.asarray(scales, dtype="<f4").tobytes(),
            np.asarray(compensation_weights, dtype="<f4").tobytes(),
            bit_pack_unsigned(unsigned.astype(np.uint32), config.depth),
        )
    )


def decode_gauge(payload: bytes) -> tuple[GaugeConfig, np.ndarray, dict[str, np.ndarray]]:
    header_format = "<4sBBBBHHHH"
    header_bytes = struct.calcsize(header_format)
    if len(payload) < header_bytes:
        raise PreproofError("gauge payload is truncated before its header")
    magic, version, depth, rank, mode_id, n, dims, block_rows, scale_columns = struct.unpack_from(
        header_format, payload
    )
    if magic != MAGIC or version != VERSION or n != N or dims != POSE_DIMS:
        raise PreproofError("gauge header identity differs")
    matching = [
        name
        for name, (block, per_rank, identifier) in CELL_MODES.items()
        if identifier == mode_id and block == block_rows and scale_columns == (rank if per_rank else 1)
    ]
    if len(matching) != 1:
        raise PreproofError("gauge cell geometry is not canonical")
    config = GaugeConfig(int(rank), int(depth), matching[0])
    _, _, _, _, group_count = mode_geometry(config)
    scale_count = group_count * scale_columns
    weight_count = (rank + 1) * dims
    scale_bytes = scale_count * 4
    weight_bytes = weight_count * 4
    code_bytes = (n * rank * depth + 7) // 8
    expected = header_bytes + scale_bytes + weight_bytes + code_bytes
    if len(payload) != expected:
        raise PreproofError(f"gauge payload length differs: {len(payload)} != {expected}")
    cursor = header_bytes
    scales = np.frombuffer(payload[cursor : cursor + scale_bytes], dtype="<f4").copy()
    scales = scales.reshape(group_count, scale_columns)
    cursor += scale_bytes
    weights = np.frombuffer(payload[cursor : cursor + weight_bytes], dtype="<f4").copy()
    weights = weights.reshape(rank + 1, dims)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0) or not np.all(np.isfinite(weights)):
        raise PreproofError("decoded gauge scales or compensation values are invalid")
    cursor += weight_bytes
    qmax = (1 << (depth - 1)) - 1
    unsigned = bit_unpack_unsigned(payload[cursor:], n * rank, depth)
    codes = unsigned.astype(np.int64).reshape(n, rank) - qmax
    dequantized = codes.astype(np.float32) * expanded_scales(scales, config).astype(np.float32)
    design = np.concatenate((np.ones((N, 1), dtype=np.float32), dequantized), axis=1)
    reconstruction = np.einsum("ij,jk->ik", design, weights.astype(np.float32), optimize=False).astype(np.float32)
    return (
        config,
        reconstruction,
        {
            "codes": codes.astype(np.int16),
            "scales": scales.astype(np.float32),
            "compensation": weights.astype(np.float32),
        },
    )


def output_mse(decoded: np.ndarray, reference: np.ndarray) -> float:
    if decoded.dtype != np.float32 or decoded.shape != (N, POSE_DIMS):
        raise PreproofError("decoded gauge output has the wrong dtype or shape")
    difference = decoded.astype(np.float64) - reference.astype(np.float64)
    return float(np.mean(difference * difference, dtype=np.float64))


def rate_envelope(
    gauge: bytes,
    *,
    ps135: Any,
    source: Any,
    semantic_stream: bytes,
    hpac_stream: bytes,
    receiver: Any,
) -> tuple[bytes, bytes]:
    carrier_stream = ps135.brotli_compress(gauge, quality=9)
    model_pack = ps135.split_pack((semantic_stream, carrier_stream, hpac_stream))
    member = receiver.pack_payload(
        model_pack,
        source.tokens,
        token_codec="ans",
        model_codec="split_brotli_cx2",
    )
    return carrier_stream, ps135.deterministic_stored_zip(member)


def verify_lc2_rate_path(ps135: Any) -> tuple[Any, bytes, bytes, Any, dict[str, object]]:
    source = ps135.load_lc2_source()
    semantic_stream, hpac_stream, selected_receipt = ps135.selected_lc2_streams()
    _, receiver, _ = ps135.import_runtime_modules()
    original_stream, rebuilt = rate_envelope(
        source.carrier,
        ps135=ps135,
        source=source,
        semantic_stream=semantic_stream,
        hpac_stream=hpac_stream,
        receiver=receiver,
    )
    if rebuilt != LC2_ARCHIVE.read_bytes():
        raise PreproofError("exact LC2 coder path did not reproduce the pinned archive")
    return (
        source,
        semantic_stream,
        hpac_stream,
        receiver,
        {
            "ps135_runner": file_record(PS135_RUNNER),
            "selected_stream_receipt": file_record(
                Path(selected_receipt["receipt_path"]) if "receipt_path" in selected_receipt else ps135.LC2_SEARCH
            ),
            "semantic_stream": {
                "bytes": len(semantic_stream),
                "sha256": sha256_bytes(semantic_stream),
            },
            "original_carrier_q9": {
                "bytes": len(original_stream),
                "sha256": sha256_bytes(original_stream),
            },
            "hpac_stream": {
                "bytes": len(hpac_stream),
                "sha256": sha256_bytes(hpac_stream),
            },
            "rebuilt_archive": {
                "bytes": len(rebuilt),
                "sha256": sha256_bytes(rebuilt),
                "byte_identical_to_lc2": True,
            },
        },
    )


def candidate_configs() -> list[GaugeConfig]:
    return [GaugeConfig(rank, depth, cell_mode) for rank in RANKS for depth in DEPTHS for cell_mode in CELL_MODES]


def validate_record(record: object, *, allowed_root: Path | None = None) -> None:
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        raise PreproofError("retained file record has the wrong schema")
    path = Path(str(record["path"]))
    if path.is_symlink():
        raise PreproofError(f"retained payload may not be a symlink: {path}")
    if allowed_root is not None:
        resolved = path.resolve()
        root = allowed_root.resolve()
        if resolved == root or root not in resolved.parents:
            raise PreproofError(f"retained payload escaped its candidate root: {path}")
    if file_record(path) != record:
        raise PreproofError(f"retained file record changed: {path}")


def validate_completed_candidate(path: Path, expected_id: str) -> dict[str, object]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "ddm_pz4p_candidate.v1"
        or receipt.get("complete") is not True
        or receipt.get("candidate_id") != expected_id
        or receipt.get("payloads_retained") is not True
        or receipt.get("runner_source_sha256") != RUNNER_SOURCE_SHA256_AT_IMPORT
    ):
        raise PreproofError(f"completed candidate receipt is invalid: {path}")
    for record in receipt["records"].values():
        validate_record(record, allowed_root=path.parent)
    for row in receipt.get("qat_history", []):
        validate_record(row["checkpoint"], allowed_root=path.parent)
    return receipt


def pareto_frontier(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["rate_envelope_bytes"]),
            float(row["decoded_output_mse"]),
            str(row["candidate_id"]),
        ),
    )
    frontier = []
    best_mse = math.inf
    for row in ordered:
        mse = float(row["decoded_output_mse"])
        if mse < best_mse:
            frontier.append(row)
            best_mse = mse
    return frontier


def run_candidate(
    config: GaugeConfig,
    *,
    output_root: Path,
    reference: np.ndarray,
    svd_u: np.ndarray,
    singular_values: np.ndarray,
    ps135: Any,
    source: Any,
    semantic_stream: bytes,
    hpac_stream: bytes,
    receiver: Any,
) -> dict[str, object]:
    assert_runner_source_unchanged()
    root = output_root / "candidates" / config.candidate_id
    receipt_path = root / "receipt.json"
    if receipt_path.exists():
        return validate_completed_candidate(receipt_path, config.candidate_id)
    root.mkdir(parents=True, exist_ok=True)
    coefficients = svd_u[:, : config.rank] * singular_values[: config.rank]
    scales = initial_scales(coefficients, config)
    codes, learned_scales, weights, _, prewire_mse, qat_rows = optimize_rounding(
        coefficients,
        scales,
        config,
        reference,
        root / "checkpoints",
    )
    gauge = encode_gauge(config, codes, learned_scales, weights)
    decoded_config, decoded, decoded_parts = decode_gauge(gauge)
    if decoded_config != config:
        raise PreproofError("decoded gauge config differs from its candidate")
    mse = output_mse(decoded, reference)
    repeat_gauge = encode_gauge(
        config,
        decoded_parts["codes"].astype(np.int32),
        decoded_parts["scales"].astype(np.float64),
        decoded_parts["compensation"].astype(np.float64),
    )
    if repeat_gauge != gauge:
        raise PreproofError("gauge parse/re-encode is not byte-identical")
    carrier_stream, envelope = rate_envelope(
        gauge,
        ps135=ps135,
        source=source,
        semantic_stream=semantic_stream,
        hpac_stream=hpac_stream,
        receiver=receiver,
    )
    repeat_stream, repeat_envelope = rate_envelope(
        gauge,
        ps135=ps135,
        source=source,
        semantic_stream=semantic_stream,
        hpac_stream=hpac_stream,
        receiver=receiver,
    )
    if repeat_stream != carrier_stream or repeat_envelope != envelope:
        raise PreproofError("real-coder rate envelope is not deterministic")
    records = {
        "gauge_payload": bytes_record(root / "gauge.pgq1", gauge),
        "gauge_repeat": bytes_record(root / "gauge.repeat.pgq1", repeat_gauge),
        "decoded_outputs": atomic_numpy(root / "decoded_outputs.float32.npy", decoded),
        "codes": atomic_numpy(root / "codes.int16.npy", decoded_parts["codes"]),
        "scales": atomic_numpy(root / "scales.float32.npy", decoded_parts["scales"]),
        "compensation": atomic_numpy(root / "compensation.float32.npy", decoded_parts["compensation"]),
        "carrier_q9": bytes_record(root / "gauge.q9.br", carrier_stream),
        "carrier_q9_repeat": bytes_record(root / "gauge.repeat.q9.br", repeat_stream),
        "rate_envelope": bytes_record(root / "rate_envelope.not_receiver_candidate.zip", envelope),
        "rate_envelope_repeat": bytes_record(
            root / "rate_envelope.repeat.not_receiver_candidate.zip",
            repeat_envelope,
        ),
    }
    savings = LC2_ARCHIVE_BYTES - len(envelope)
    row = {
        "schema": "ddm_pz4p_candidate.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "candidate_id": config.candidate_id,
        "runner_source_sha256": RUNNER_SOURCE_SHA256_AT_IMPORT,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "receiver_closed": False,
        "rate_envelope_only": True,
        "reference": "banked exact LC2-native PoseNet first-six outputs for all 600 pairs",
        "rank": config.rank,
        "depth_bits": config.depth,
        "cell_mode": config.cell_mode,
        "qat_rounds": QAT_ROUNDS,
        "compensation": "hard-quantized least-squares solve after every round",
        "prewire_float64_mse": prewire_mse,
        "decoded_output_mse": mse,
        "gauge_raw_bytes": len(gauge),
        "gauge_q9_bytes": len(carrier_stream),
        "rate_envelope_bytes": len(envelope),
        "whole_container_bytes_saved_vs_lc2": savings,
        "bytes_saved_vs_pk2_pose_marginal": PK2_POSE_MARGINAL_BYTES - len(carrier_stream),
        "mse_gate_passes": mse < MSE_GATE,
        "savings_gate_passes": savings >= SAVINGS_GATE_BYTES,
        "joint_fire_gate_passes": mse < MSE_GATE and savings >= SAVINGS_GATE_BYTES,
        "payloads_retained": True,
        "records": records,
        "qat_history": qat_rows,
    }
    atomic_json(receipt_path, row)
    assert_runner_source_unchanged()
    return row


def run(output_root: Path) -> dict[str, object]:
    assert_runner_source_unchanged()
    output_root = validate_output_root(output_root)
    storage = require_storage(output_root)
    reference, custody = verify_custody()
    ps135 = load_ps135()
    source, semantic_stream, hpac_stream, receiver, rate_path = verify_lc2_rate_path(ps135)
    centered = reference.astype(np.float64) - np.mean(reference.astype(np.float64), axis=0)
    svd_u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    if svd_u.shape != (N, POSE_DIMS) or singular_values.shape != (POSE_DIMS,):
        raise PreproofError("reference SVD has unexpected geometry")
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "state.json"
    rows: list[dict[str, object]] = []
    configs = candidate_configs()
    for index, config in enumerate(configs, start=1):
        row = run_candidate(
            config,
            output_root=output_root,
            reference=reference,
            svd_u=svd_u,
            singular_values=singular_values,
            ps135=ps135,
            source=source,
            semantic_stream=semantic_stream,
            hpac_stream=hpac_stream,
            receiver=receiver,
        )
        rows.append(row)
        atomic_json(
            state_path,
            {
                "schema": "ddm_pz4p_state.v1",
                "written_at_utc": utc_now(),
                "complete": False,
                "completed_candidates": index,
                "candidate_denominator": len(configs),
                "last_candidate": config.candidate_id,
                "resume_from": str(state_path),
            },
        )
        print(
            json.dumps(
                {
                    "candidate": config.candidate_id,
                    "completed": index,
                    "denominator": len(configs),
                    "mse": row["decoded_output_mse"],
                    "rate_envelope_bytes": row["rate_envelope_bytes"],
                    "fire": row["joint_fire_gate_passes"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summaries = [
        {
            key: row[key]
            for key in (
                "candidate_id",
                "rank",
                "depth_bits",
                "cell_mode",
                "decoded_output_mse",
                "gauge_raw_bytes",
                "gauge_q9_bytes",
                "rate_envelope_bytes",
                "whole_container_bytes_saved_vs_lc2",
                "joint_fire_gate_passes",
            )
        }
        for row in rows
    ]
    frontier = pareto_frontier(summaries)
    passes = [row for row in summaries if row["joint_fire_gate_passes"]]
    winner = (
        min(
            passes,
            key=lambda row: (
                int(row["rate_envelope_bytes"]),
                float(row["decoded_output_mse"]),
                str(row["candidate_id"]),
            ),
        )
        if passes
        else None
    )
    rank_floors = {}
    for rank in RANKS:
        rank_rows = [row for row in summaries if row["rank"] == rank]
        rank_floors[str(rank)] = min(rank_rows, key=lambda row: float(row["decoded_output_mse"]))
    result = {
        "schema": "ddm_pz4p_final.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "pointer_moved": False,
        "scorer_forward_passes": 0,
        "pair_denominator": N,
        "candidate_denominator": len(rows),
        "reference": {
            "definition": (
                "banked exact PoseNet first-six outputs produced by the shipped LC2 "
                "carrier; decoded-gauge reconstruction is a surrogate pre-proof only"
            ),
            "custody": custody,
        },
        "rate_path": rate_path,
        "storage_preflight": storage,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "argv": sys.argv,
        },
        "source_identity": {
            "runner": file_record(Path(__file__)),
            "tests": file_record(TEST_PATH),
            "runner_source_sha256_at_import": RUNNER_SOURCE_SHA256_AT_IMPORT,
        },
        "grid": {
            "depths": list(DEPTHS),
            "ranks": list(RANKS),
            "cell_modes": list(CELL_MODES),
            "qat_rounds": QAT_ROUNDS,
            "candidate_denominator": len(rows),
        },
        "gate": {
            "mse_strictly_below": MSE_GATE,
            "whole_container_savings_at_least_bytes": SAVINGS_GATE_BYTES,
            "passes": bool(passes),
            "passing_candidate_count": len(passes),
            "winner": winner,
        },
        "rank_floors": rank_floors,
        "pareto_frontier": frontier,
        "all_candidates": summaries,
        "receiver_boundary": (
            "Every envelope replaces CPR1 with PGQ1 in the exact LC2 coder/container path "
            "but the unchanged receiver cannot parse PGQ1. No envelope is a submission "
            "candidate and no rendered d_pose is measured."
        ),
        "payloads_retained": True,
    }
    atomic_json(output_root / "FINAL_RESULT.json", result)
    atomic_json(output_root / "PARETO_FRONTIER.json", frontier)
    atomic_json(
        state_path,
        {
            "schema": "ddm_pz4p_state.v1",
            "written_at_utc": utc_now(),
            "complete": True,
            "completed_candidates": len(rows),
            "candidate_denominator": len(rows),
            "resume_from": str(state_path),
            "final_result": file_record(output_root / "FINAL_RESULT.json"),
        },
    )
    assert_runner_source_unchanged()
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--out", type=Path, default=DEFAULT_OUT)
    value.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="must name OUT/state.json; completed candidate receipts are verified and reused",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    output_root = validate_output_root(args.out)
    expected_state = output_root / "state.json"
    if args.resume_from is not None and args.resume_from.resolve() != expected_state.resolve():
        raise PreproofError(f"--resume-from must be {expected_state}")
    result = run(output_root)
    print(json.dumps(result["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
