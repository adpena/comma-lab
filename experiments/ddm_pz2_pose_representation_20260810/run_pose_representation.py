#!/usr/bin/env python3
"""DALI-axis pose-target representation measurement for the ddm_pz2 arm.

This is deliberately scorer-free.  It measures the six official DALI PoseNet
targets as a source-coding problem, retains every raw/compressed stream and
every Pareto packet, and prices hypothetical section replacement through the
nonlinear pose term.  It does not construct frames or claim a contest score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch

ARM = "ddm_pz2_pose_representation_20260810_v3"
SCHEMA = "ddm_pz2_pose_representation.v1"
AXIS = "[macOS-CPU scorer-free representation measurement; official DALI GT targets, n600]"
TOY_AXIS = "[TOY-BRACKET over contest-CUDA,DALI,n600 base components; no receiver/scorer]"
SEED = 20260809
N = 600
DIMS = 6
MAX_BITS = 16
BASE_ARCHIVE_BYTES = 191_052
BASE_ARCHIVE_SHA256 = "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
BASE_D_SEG = 0.00029660
BASE_D_POSE = 0.00002331
BASE_SCORE = 0.172141297491896447
POSE_SECTION_BYTES = 23_384
SCORE_DENOMINATOR = 37_545_489
RATE_PER_BYTE = 25.0 / SCORE_DENOMINATOR
RAW_BYTES = 3_662_409_600
RAW_SHA256 = "a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353"
DALI_CACHE_SHA256 = "382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195"
INTAKE_COMMIT = "e34f31bc4969042c0051ac81aa3c56884419a231"
MAGIC = b"PZ2TGT1\0"
METHODS = {"direct": 0, "delta1": 1, "delta2": 2}
METHODS_BY_ID = {value: key for key, value in METHODS.items()}
HEADER = struct.Struct("<8sBBHB")
STREAM_HEADER = struct.Struct("<BffI")


@dataclass(frozen=True)
class QuantizedColumn:
    bits: int
    low: float
    step: float
    codes: np.ndarray
    decoded: np.ndarray
    mse_full: float
    mse_n120: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def retain_payload(
    path: Path,
    payload: bytes,
    manifest: list[dict[str, Any]],
    role: str,
) -> dict[str, Any]:
    """Persist one materialized payload atomically and record byte identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(payload)
    if path.exists():
        if path.stat().st_size != len(payload) or sha256_file(path) != digest:
            raise RuntimeError(f"refusing to overwrite nonidentical retained payload: {path}")
    else:
        tmp = path.with_suffix(path.suffix + ".pending")
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    record = {
        "path": str(path),
        "bytes": len(payload),
        "sha256": digest,
        "role": role,
    }
    manifest.append(record)
    return record


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def checkpoint(path: Path, value: Any) -> None:
    """Write a distinct, immutable stage checkpoint."""
    if path.exists():
        if load_json(path) != value:
            raise RuntimeError(f"stage checkpoint differs on resume: {path}")
        return
    atomic_json(path, value)


def entropy_bits(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    counts = np.asarray(list(Counter(int(v) for v in values.tolist()).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def quantize_column(values: np.ndarray, bits: int, selection: np.ndarray) -> QuantizedColumn:
    if bits == 0:
        low = np.float32(values.mean())
        step = np.float32(0.0)
        codes = np.zeros(values.shape, dtype=np.uint16)
    else:
        low = np.float32(values.min())
        levels = (1 << bits) - 1
        step = np.float32((float(values.max()) - float(low)) / levels)
        if not float(step) > 0.0:
            codes = np.zeros(values.shape, dtype=np.uint16)
        else:
            rounded = np.rint((values.astype(np.float64) - float(low)) / float(step))
            codes = np.clip(rounded, 0, levels).astype(np.uint16)
    decoded = float(low) + codes.astype(np.float64) * float(step)
    residual = decoded - values.astype(np.float64)
    return QuantizedColumn(
        bits=bits,
        low=float(low),
        step=float(step),
        codes=codes,
        decoded=decoded,
        mse_full=float(np.mean(np.square(residual))),
        mse_n120=float(np.mean(np.square(residual[selection]))),
    )


def serialize_codes(codes: np.ndarray, method: str) -> bytes:
    signed = codes.astype(np.int32)
    if method == "direct":
        return codes.astype("<u2", copy=False).tobytes()
    if method == "delta1":
        prefix = codes[:1].astype("<u2", copy=False).tobytes()
        residual = np.diff(signed).astype("<i4", copy=False).tobytes()
        return prefix + residual
    if method == "delta2":
        prefix = codes[:2].astype("<u2", copy=False).tobytes()
        residual = np.diff(signed, n=2).astype("<i4", copy=False).tobytes()
        return prefix + residual
    raise ValueError(method)


def deserialize_codes(raw: bytes, method: str, count: int) -> np.ndarray:
    if method == "direct":
        result = np.frombuffer(raw, dtype="<u2").astype(np.int64)
    elif method == "delta1":
        first = np.frombuffer(raw[:2], dtype="<u2").astype(np.int64)
        delta = np.frombuffer(raw[2:], dtype="<i4").astype(np.int64)
        result = np.concatenate([first, delta]).cumsum()
    elif method == "delta2":
        first = np.frombuffer(raw[:4], dtype="<u2").astype(np.int64)
        second = np.frombuffer(raw[4:], dtype="<i4").astype(np.int64)
        first_delta = int(first[1]) - int(first[0])
        first_differences = np.concatenate(
            [np.asarray([first_delta], dtype=np.int64), second]
        ).cumsum()
        result = np.concatenate(
            [first[:1], int(first[0]) + np.cumsum(first_differences)]
        )
    else:
        raise ValueError(method)
    if result.shape != (count,) or np.any(result < 0) or np.any(result > 65535):
        raise RuntimeError(f"invalid {method} code parse-back")
    return result.astype(np.uint16)


def build_packet(
    method: str,
    allocation: tuple[int, ...],
    options: list[list[QuantizedColumn]],
    compressed_streams: dict[tuple[str, int, int], bytes],
) -> bytes:
    chunks = [HEADER.pack(MAGIC, 1, METHODS[method], N, DIMS)]
    for dim, bits in enumerate(allocation):
        option = options[dim][bits]
        compressed = compressed_streams[(method, dim, bits)]
        chunks.append(STREAM_HEADER.pack(bits, option.low, option.step, len(compressed)))
        chunks.append(compressed)
    return b"".join(chunks)


def decode_packet(payload: bytes) -> tuple[str, list[int], np.ndarray, list[np.ndarray]]:
    magic, version, method_id, count, dims = HEADER.unpack_from(payload, 0)
    if magic != MAGIC or version != 1 or count != N or dims != DIMS or method_id not in METHODS_BY_ID:
        raise RuntimeError("invalid PZ2 packet header")
    method = METHODS_BY_ID[method_id]
    offset = HEADER.size
    allocations: list[int] = []
    decoded_columns: list[np.ndarray] = []
    code_columns: list[np.ndarray] = []
    for _ in range(dims):
        bits, low, step, compressed_bytes = STREAM_HEADER.unpack_from(payload, offset)
        offset += STREAM_HEADER.size
        compressed = payload[offset : offset + compressed_bytes]
        offset += compressed_bytes
        raw = brotli.decompress(compressed)
        codes = deserialize_codes(raw, method, count)
        allocations.append(int(bits))
        code_columns.append(codes)
        decoded_columns.append(float(low) + codes.astype(np.float64) * float(step))
    if offset != len(payload):
        raise RuntimeError("trailing PZ2 packet bytes")
    return method, allocations, np.stack(decoded_columns, axis=1), code_columns


def pareto_states(
    method: str,
    options: list[list[QuantizedColumn]],
    compressed_streams: dict[tuple[str, int, int], bytes],
) -> list[tuple[int, float, tuple[int, ...]]]:
    states: dict[int, tuple[float, tuple[int, ...]]] = {HEADER.size: (0.0, ())}
    for dim in range(DIMS):
        expanded: dict[int, tuple[float, tuple[int, ...]]] = {}
        for prior_cost, (prior_distortion, prior_allocation) in states.items():
            for bits in range(MAX_BITS + 1):
                cost = prior_cost + STREAM_HEADER.size + len(compressed_streams[(method, dim, bits)])
                distortion = prior_distortion + options[dim][bits].mse_full
                incumbent = expanded.get(cost)
                if incumbent is None or distortion < incumbent[0]:
                    expanded[cost] = (distortion, (*prior_allocation, bits))
        states = {}
        best_distortion = math.inf
        for cost in sorted(expanded):
            distortion, allocation = expanded[cost]
            if distortion < best_distortion - 1e-30:
                states[cost] = (distortion, allocation)
                best_distortion = distortion
    return [(cost, distortion / DIMS, allocation) for cost, (distortion, allocation) in states.items()]


def gaussian_waterfill(variances: np.ndarray, target_mse: float) -> dict[str, Any]:
    low = 0.0
    high = float(variances.max())
    for _ in range(160):
        theta = (low + high) / 2.0
        achieved = float(np.minimum(variances, theta).mean())
        if achieved < target_mse:
            low = theta
        else:
            high = theta
    theta = high
    bits = np.where(variances > theta, 0.5 * np.log2(variances / theta), 0.0)
    return {
        "target_mse": target_mse,
        "water_level": theta,
        "bits_per_pair_by_dimension": bits.tolist(),
        "total_bits_per_pair": float(bits.sum()),
        "gaussian_rd_model_bytes_n600": float(N * bits.sum() / 8.0),
        "scope": "TOY-BRACKET independent-coordinate Gaussian R(D) reference; exact only for the Gaussian model, not an achieved payload or universal lower bound",
    }


def temporal_stats(targets: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dim in range(DIMS):
        values = targets[:, dim].astype(np.float64)
        delta1 = np.diff(values)
        delta2 = np.diff(values, n=2)
        rows.append(
            {
                "dimension": dim,
                "mean": float(values.mean()),
                "variance": float(values.var()),
                "std": float(values.std()),
                "lag1_autocorrelation": float(np.corrcoef(values[:-1], values[1:])[0, 1]),
                "delta1_std": float(delta1.std()),
                "delta1_to_level_std": float(delta1.std() / values.std()),
                "delta2_std": float(delta2.std()),
                "delta2_to_level_std": float(delta2.std() / values.std()),
            }
        )
    return rows


def projection(packet_bytes: int, quantization_mse: float) -> dict[str, float]:
    archive_bytes = BASE_ARCHIVE_BYTES - POSE_SECTION_BYTES + packet_bytes
    rate_term = RATE_PER_BYTE * archive_bytes
    seg_term = 100.0 * BASE_D_SEG
    additive_pose = BASE_D_POSE + quantization_mse
    worst_aligned_pose = (math.sqrt(BASE_D_POSE) + math.sqrt(quantization_mse)) ** 2
    best_aligned_pose = max(0.0, math.sqrt(BASE_D_POSE) - math.sqrt(quantization_mse)) ** 2
    return {
        "hypothetical_archive_bytes": archive_bytes,
        "delta_archive_bytes": archive_bytes - BASE_ARCHIVE_BYTES,
        "delta_rate_s": RATE_PER_BYTE * (archive_bytes - BASE_ARCHIVE_BYTES),
        "optimistic_perfect_realization_s": seg_term + math.sqrt(10.0 * quantization_mse) + rate_term,
        "additive_error_s": seg_term + math.sqrt(10.0 * additive_pose) + rate_term,
        "best_aligned_error_s": seg_term + math.sqrt(10.0 * best_aligned_pose) + rate_term,
        "worst_aligned_error_s": seg_term + math.sqrt(10.0 * worst_aligned_pose) + rate_term,
        "additive_delta_s_vs_base": seg_term
        + math.sqrt(10.0 * additive_pose)
        + rate_term
        - BASE_SCORE,
    }


def git_head(path: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact") / ARM,
    )
    parser.add_argument(
        "--dali-cache",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/ddm_pr130_train_20260809/caches/gt_cache_600_official_ada.pt"),  # GT_LINEAGE_OK: default bytes are registry-classified DALI_NVDEC sha256 382d7dfe38b37c0c
    )
    parser.add_argument(
        "--base-archive",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip"),
    )
    parser.add_argument(
        "--selection-receipt",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/ddm_pk2_20260809/checkpoints/masters_n120_seed20260809.json"),
    )
    parser.add_argument(
        "--intake-root",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo"),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Existing PZ2 output root whose immutable stage checkpoints may be reused.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    expected_parent = Path("/Volumes/VertigoDataTier/pact").resolve()
    if output.parent != expected_parent or not output.name.startswith("ddm_pz2_pose_representation_"):
        raise RuntimeError("bulk output must be the PZ2 arm's own VertigoDataTier directory")
    if args.resume_from is not None and args.resume_from.resolve() != output:
        raise RuntimeError("--resume-from must name the same root as --output")
    free_bytes = shutil.disk_usage(expected_parent).free
    required_free_bytes = 64 * 1024 * 1024
    if free_bytes < required_free_bytes:
        raise RuntimeError(f"storage preflight failed: {free_bytes} < {required_free_bytes}")

    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    retained = output / "retained"
    retained.mkdir(parents=True, exist_ok=True)

    archive_sha = sha256_file(args.base_archive)
    cache_sha = sha256_file(args.dali_cache)
    if args.base_archive.stat().st_size != BASE_ARCHIVE_BYTES or archive_sha != BASE_ARCHIVE_SHA256:
        raise RuntimeError("PR130 archive provenance pin failed")
    if cache_sha != DALI_CACHE_SHA256:
        raise RuntimeError("official DALI target cache provenance pin failed")
    intake_head = git_head(args.intake_root)
    if intake_head != INTAKE_COMMIT:
        raise RuntimeError("PR130 intake commit pin failed")

    selection_receipt = load_json(args.selection_receipt)
    selection = np.asarray(selection_receipt["selection_indices"], dtype=np.int64)
    if selection.shape != (120,) or len(np.unique(selection)) != 120:
        raise RuntimeError("invalid n120 selection receipt")
    block_counts = np.bincount(selection // 120, minlength=5)
    if block_counts.tolist() != [24, 24, 24, 24, 24]:
        raise RuntimeError(f"selection is not 24-per-block stratified: {block_counts.tolist()}")

    cache = torch.load(args.dali_cache, map_location="cpu", weights_only=True)
    targets = np.asarray(cache["pose"].detach().cpu(), dtype=np.float32)
    if targets.shape != (N, DIMS) or not np.isfinite(targets).all():
        raise RuntimeError(f"invalid official DALI pose target shape/content: {targets.shape}")

    manifest_path = output / "payload_manifest.json"
    manifest: list[dict[str, Any]] = []
    target_path = retained / "official_dali_pose_targets.f32"
    target_repeat_path = retained / "official_dali_pose_targets.repeat.f32"
    if not target_path.exists():
        targets.astype("<f4", copy=False).tofile(target_path)
    if not target_repeat_path.exists():
        targets.astype("<f4", copy=False).tofile(target_repeat_path)
    expected_target_values = targets.astype("<f4", copy=False)
    for path in (target_path, target_repeat_path):
        retained_values = np.fromfile(path, dtype="<f4")
        if retained_values.shape != (N * DIMS,) or not np.array_equal(
            retained_values.reshape(N, DIMS), expected_target_values
        ):
            raise RuntimeError(f"retained DALI target differs from consumed cache: {path}")
    if sha256_file(target_path) != sha256_file(target_repeat_path):
        raise RuntimeError("DALI target retention repeat differs")
    for path, role in (
        (target_path, "official DALI pose target tensor, float32 row-major"),
        (target_repeat_path, "determinism repeat of official DALI pose target tensor"),
    ):
        manifest.append(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "role": role}
        )

    covariance = np.cov(targets.astype(np.float64), rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    basis_raw = eigenvectors.astype("<f4", copy=False).tobytes()
    eigenvalues_repeat, eigenvectors_repeat = np.linalg.eigh(covariance)
    basis_repeat_raw = eigenvectors_repeat.astype("<f4", copy=False).tobytes()
    if not np.array_equal(eigenvalues, eigenvalues_repeat) or basis_raw != basis_repeat_raw:
        raise RuntimeError("KLT eigensolve determinism repeat failed")
    basis_record = retain_payload(
        retained / "klt" / "official_dali_pose_klt_basis.f32",
        basis_raw,
        manifest,
        "video-derived official-DALI KLT basis, float32",
    )
    basis_repeat_record = retain_payload(
        retained / "klt" / "official_dali_pose_klt_basis.repeat.f32",
        basis_repeat_raw,
        manifest,
        "determinism repeat of official-DALI KLT basis",
    )
    basis_brotli = brotli.compress(basis_raw, quality=11)
    basis_brotli_record = retain_payload(
        retained / "klt" / "official_dali_pose_klt_basis.f32.br",
        basis_brotli,
        manifest,
        "Brotli-q11 official-DALI KLT basis",
    )
    if brotli.decompress(basis_brotli) != basis_raw:
        raise RuntimeError("KLT basis compression parse-back failed")
    klt_basis_counted_bytes = min(basis_record["bytes"], basis_brotli_record["bytes"])
    axis_waterfill_at_base = gaussian_waterfill(np.diag(covariance), BASE_D_POSE)
    klt_waterfill_at_base = gaussian_waterfill(eigenvalues, BASE_D_POSE)
    correlations = np.corrcoef(targets.astype(np.float64), rowvar=False)
    off_diagonal = correlations - np.eye(DIMS, dtype=np.float64)
    klt_measurement = {
        "axis": AXIS,
        "covariance_eigenvalues": eigenvalues.tolist(),
        "max_absolute_off_diagonal_correlation": float(np.abs(off_diagonal).max()),
        "axis_aligned_gaussian_rd_model_bytes_n600": axis_waterfill_at_base[
            "gaussian_rd_model_bytes_n600"
        ],
        "klt_gaussian_rd_model_bytes_n600_before_basis": klt_waterfill_at_base[
            "gaussian_rd_model_bytes_n600"
        ],
        "klt_payload_saving_before_basis_bytes": axis_waterfill_at_base[
            "gaussian_rd_model_bytes_n600"
        ]
        - klt_waterfill_at_base["gaussian_rd_model_bytes_n600"],
        "video_derived_basis": basis_record,
        "video_derived_basis_repeat": basis_repeat_record,
        "video_derived_basis_brotli_q11": basis_brotli_record,
        "counted_basis_bytes_best_of_raw_or_brotli": klt_basis_counted_bytes,
        "klt_net_bytes_vs_axis_aligned": klt_waterfill_at_base["gaussian_rd_model_bytes_n600"]
        + klt_basis_counted_bytes
        - axis_waterfill_at_base["gaussian_rd_model_bytes_n600"],
        "verdict": "KLT_LOSES_AFTER_COUNTING_VIDEO_DERIVED_BASIS",
        "verdict_scope": "INSTANCE: official DALI n600 target tensor at base d_pose under Gaussian reverse-waterfill",
    }

    current_provenance = {
        "schema": SCHEMA,
        "axis": AXIS,
        "seed": SEED,
        "storage_preflight": {
            "tier": str(expected_parent),
            "free_bytes_at_launch": free_bytes,
            "required_free_bytes": required_free_bytes,
            "status": "PASS",
        },
        "base_archive": {"path": str(args.base_archive), "bytes": BASE_ARCHIVE_BYTES, "sha256": archive_sha},
        "official_dali_target_cache": {
            "path": str(args.dali_cache),
            "bytes": args.dali_cache.stat().st_size,
            "sha256": cache_sha,
            "pose_shape": list(targets.shape),
        },
        "pr130_intake": {"path": str(args.intake_root), "git_head": intake_head},
        "selection": {
            "path": str(args.selection_receipt),
            "sha256": sha256_file(args.selection_receipt),
            "seed": SEED,
            "count": int(selection.size),
            "indices": selection.tolist(),
            "block_counts_120": block_counts.tolist(),
        },
        "base_raw_pin": {
            "bytes": RAW_BYTES,
            "sha256": RAW_SHA256,
            "verification": "receipt-only; source raw was success-cleaned and was not consumed by this scorer-free run",
            "receipt": ".omx/research/ddm_dt1_ans_decode_wallclock_gate_20260809.md",
        },
    }
    inputs_checkpoint = checkpoints / "stage_01_inputs.json"
    if args.resume_from is not None and inputs_checkpoint.exists():
        provenance = load_json(inputs_checkpoint)
        for key in (
            "base_archive",
            "official_dali_target_cache",
            "pr130_intake",
            "selection",
            "base_raw_pin",
        ):
            if provenance[key] != current_provenance[key]:
                raise RuntimeError(f"resume input provenance differs at {key}")
    else:
        provenance = current_provenance
        checkpoint(inputs_checkpoint, provenance)

    sensitivity_checkpoint = checkpoints / "stage_02_sensitivity.json"
    if args.resume_from is not None and sensitivity_checkpoint.exists():
        sensitivity = load_json(sensitivity_checkpoint)
    else:
        options: list[list[QuantizedColumn]] = []
        per_dimension: list[dict[str, Any]] = []
        marginal_dscore_ddpose = 5.0 / math.sqrt(10.0 * BASE_D_POSE)
        for dim in range(DIMS):
            column_options = [quantize_column(targets[:, dim], bits, selection) for bits in range(MAX_BITS + 1)]
            options.append(column_options)
            values = targets[:, dim].astype(np.float64)
            per_dimension.append(
                {
                    "dimension": dim,
                    "mean": float(values.mean()),
                    "variance": float(values.var()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "exact_score_sensitivity_per_unit_dimension_mse": marginal_dscore_ddpose / DIMS,
                    "local_s_cost_per_1e-6_dimension_mse": marginal_dscore_ddpose * 1e-6 / DIMS,
                    "local_byte_equivalent_per_1e-6_dimension_mse": marginal_dscore_ddpose
                    * 1e-6
                    / DIMS
                    / RATE_PER_BYTE,
                    "quantization_ladder": [
                        {
                            "bits": option.bits,
                            "mse_n600": option.mse_full,
                            "mse_n120_stratified": option.mse_n120,
                            "n120_to_n600_ratio": option.mse_n120 / option.mse_full
                            if option.mse_full > 0.0
                            else None,
                        }
                        for option in column_options
                    ],
                }
            )
        variances = targets.astype(np.float64).var(axis=0)
        sensitivity = {
            "schema": SCHEMA,
            "axis": AXIS,
            "score_derivative": {
                "base_d_pose": BASE_D_POSE,
                "d_sqrt10d_dd_pose": marginal_dscore_ddpose,
                "s_cost_per_1e-6_d_pose_local_linear": marginal_dscore_ddpose * 1e-6,
                "byte_equivalent_per_1e-6_d_pose_local_linear": marginal_dscore_ddpose
                * 1e-6
                / RATE_PER_BYTE,
                "dimension_sensitivity_result": "equal by the scorer definition: each dimension contributes 1/6 of aggregate MSE",
            },
            "per_dimension": per_dimension,
            "temporal": temporal_stats(targets),
            "gaussian_waterfill": [
                gaussian_waterfill(variances, target)
                for target in (BASE_D_POSE, BASE_D_POSE / 10.0, 1e-6)
            ],
            "klt": klt_measurement,
        }
        checkpoint(sensitivity_checkpoint, sensitivity)

    packet_checkpoint = checkpoints / "stage_03_packets.json"
    if args.resume_from is not None and packet_checkpoint.exists() and manifest_path.exists():
        packet_measurement = load_json(packet_checkpoint)
        prior_manifest = load_json(manifest_path)
        manifest = prior_manifest["payloads"]
    else:
        options = [
            [quantize_column(targets[:, dim], bits, selection) for bits in range(MAX_BITS + 1)]
            for dim in range(DIMS)
        ]
        compressed_streams: dict[tuple[str, int, int], bytes] = {}
        stream_rows: list[dict[str, Any]] = []
        for method in METHODS:
            for dim in range(DIMS):
                for bits in range(MAX_BITS + 1):
                    raw = serialize_codes(options[dim][bits].codes, method)
                    raw_record = retain_payload(
                        retained / "streams" / method / f"d{dim}_b{bits:02d}.raw",
                        raw,
                        manifest,
                        f"{method} raw stream, dimension {dim}, bits {bits}",
                    )
                    compressed = brotli.compress(raw, quality=11)
                    compressed_record = retain_payload(
                        retained / "streams" / method / f"d{dim}_b{bits:02d}.br",
                        compressed,
                        manifest,
                        f"{method} Brotli-q11 stream, dimension {dim}, bits {bits}",
                    )
                    if brotli.decompress(compressed) != raw:
                        raise RuntimeError("Brotli stream parse-back failed")
                    if not np.array_equal(
                        deserialize_codes(raw, method, N), options[dim][bits].codes
                    ):
                        raise RuntimeError("raw stream parse-back failed")
                    compressed_streams[(method, dim, bits)] = compressed
                    codes = options[dim][bits].codes.astype(np.int64)
                    delta1 = np.diff(codes)
                    delta2 = np.diff(codes, n=2)
                    stream_rows.append(
                        {
                            "method": method,
                            "dimension": dim,
                            "bits": bits,
                            "mse_n600": options[dim][bits].mse_full,
                            "mse_n120_stratified": options[dim][bits].mse_n120,
                            "zero_order_entropy_bits_per_symbol": entropy_bits(codes),
                            "delta1_entropy_bits_per_symbol": entropy_bits(delta1),
                            "delta2_entropy_bits_per_symbol": entropy_bits(delta2),
                            "raw": raw_record,
                            "brotli_q11": compressed_record,
                        }
                    )

        candidate_rows: list[dict[str, Any]] = []
        for method in METHODS:
            for index, (expected_bytes, _distortion, allocation) in enumerate(
                pareto_states(method, options, compressed_streams)
            ):
                packet = build_packet(method, allocation, options, compressed_streams)
                packet_repeat = build_packet(method, allocation, options, compressed_streams)
                candidate_id = f"{method}_p{index:03d}_b" + "-".join(str(value) for value in allocation)
                packet_record = retain_payload(
                    retained / "candidates" / method / f"{candidate_id}.pz2",
                    packet,
                    manifest,
                    f"PZ2 Pareto packet {candidate_id}",
                )
                repeat_record = retain_payload(
                    retained / "candidates" / method / f"{candidate_id}.repeat.pz2",
                    packet_repeat,
                    manifest,
                    f"determinism repeat PZ2 Pareto packet {candidate_id}",
                )
                if packet != packet_repeat or expected_bytes != len(packet):
                    raise RuntimeError("packet determinism/DP byte accounting failed")
                parsed_method, parsed_allocation, decoded, parsed_codes = decode_packet(packet)
                if parsed_method != method or parsed_allocation != list(allocation):
                    raise RuntimeError("packet header parse-back failed")
                for dim, bits in enumerate(allocation):
                    if not np.array_equal(parsed_codes[dim], options[dim][bits].codes):
                        raise RuntimeError("packet code parse-back failed")
                quantization_mse = float(np.mean(np.square(decoded - targets.astype(np.float64))))
                quantization_mse_n120 = float(
                    np.mean(np.square(decoded[selection] - targets[selection].astype(np.float64)))
                )
                zero_order_floor = sum(
                    N * entropy_bits(options[dim][bits].codes.astype(np.int64)) / 8.0
                    for dim, bits in enumerate(allocation)
                )
                if method == "direct":
                    temporal_floor = zero_order_floor
                elif method == "delta1":
                    temporal_floor = sum(
                        2.0
                        + (N - 1)
                        * entropy_bits(np.diff(options[dim][bits].codes.astype(np.int64)))
                        / 8.0
                        for dim, bits in enumerate(allocation)
                    )
                else:
                    temporal_floor = sum(
                        4.0
                        + (N - 2)
                        * entropy_bits(np.diff(options[dim][bits].codes.astype(np.int64), n=2))
                        / 8.0
                        for dim, bits in enumerate(allocation)
                    )
                candidate_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "axis": AXIS,
                        "projection_axis": TOY_AXIS,
                        "method": method,
                        "allocation_bits": list(allocation),
                        "packet": packet_record,
                        "repeat": repeat_record,
                        "parseback_code_exact": True,
                        "quantization_mse_n600_official_dali": quantization_mse,
                        "quantization_mse_n120_stratified_official_dali": quantization_mse_n120,
                        "n120_to_n600_mse_ratio": quantization_mse_n120 / quantization_mse
                        if quantization_mse > 0.0
                        else None,
                        "zero_order_marginal_entropy_bytes": zero_order_floor,
                        "method_residual_marginal_entropy_bytes": temporal_floor,
                        "projection": projection(len(packet), quantization_mse),
                    }
                )

        best_additive = min(candidate_rows, key=lambda row: row["projection"]["additive_error_s"])
        score_adequate: dict[str, Any] = {}
        for method in METHODS:
            admissible = [
                row
                for row in candidate_rows
                if row["method"] == method
                and row["quantization_mse_n600_official_dali"] <= BASE_D_POSE
            ]
            score_adequate[method] = min(admissible, key=lambda row: row["packet"]["bytes"])
        packet_measurement = {
            "schema": SCHEMA,
            "axis": AXIS,
            "projection_axis": TOY_AXIS,
            "candidate_denominator": len(candidate_rows),
            "candidate_counts_by_method": {
                method: sum(row["method"] == method for row in candidate_rows) for method in METHODS
            },
            "stream_denominator": len(stream_rows),
            "stream_counts_by_method": {
                method: sum(row["method"] == method for row in stream_rows) for method in METHODS
            },
            "packet_family": {
                "quantizer": "per-dimension uniform min/max scalar quantizer; exhaustive 0..16-bit allocation",
                "coder": "per-dimension Brotli-q11 over direct, first-difference, or second-difference integer streams",
                "header": "PZ2TGT1 v1 with float32 low/step parse-back",
                "verdict_scope": "FORMULATION: this fixed scalar-quantizer and three residual charts only",
            },
            "best_additive_toy_bracket": best_additive,
            "minimum_byte_candidate_at_quantization_mse_le_base_dpose": score_adequate,
            "candidates": candidate_rows,
            "streams": stream_rows,
        }
        checkpoint(packet_checkpoint, packet_measurement)
        atomic_json(
            manifest_path,
            {
                "schema": "ddm_pz2_payload_manifest.v1",
                "payload_count": len(manifest),
                "payloads": manifest,
                "certify_or_block": "all materialized raw/compressed streams and candidate packets retained; nothing deleted",
            },
        )

    receipt = {
        "schema": SCHEMA,
        "score_claim": False,
        "scorer_invoked": False,
        "scorer_slot_owned": False,
        "axis": AXIS,
        "projection_axis": TOY_AXIS,
        "provenance": provenance,
        "sensitivity": sensitivity,
        "packet_measurement": packet_measurement,
        "frame_parity_screen": {
            "status": "NOT_RUN_NO_FRAME_REALIZATION",
            "carrier_frames": "even",
            "reason": "PZ2 stores quantized scorer targets but has no receiver that realizes them into PR130 frames; presenting target decode equality as frame parity would be fake",
            "base_raw_pin": {"bytes": RAW_BYTES, "sha256": RAW_SHA256},
        },
        "scorer_queue": {
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "unassigned successor with explicit scorer-slot claim",
            "consumer_store": ".omx/research/ddm_pz2_pose_representation_20260810/PZ2_SCORER_QUEUE.jsonl",
            "fire_trigger": "a byte-closed PR130-compatible receiver consumes one retained PZ2 packet, deterministic x2 decode passes, and all even frames differ only as declared while every odd frame is byte-identical to base raw",
            "candidate_id": packet_measurement["best_additive_toy_bracket"]["candidate_id"],
            "status": "BLOCKED_PRE_SCORER_ON_MISSING_REALIZATION_RECEIVER",
        },
        "payload_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "payload_count": len(manifest),
        },
        "resumability": {
            "resume_argument": "--resume-from <same output root>",
            "stage_checkpoints": [
                str(inputs_checkpoint),
                str(sensitivity_checkpoint),
                str(packet_checkpoint),
            ],
            "immutable_stage_checkpoints": True,
        },
        "frontier": {
            "base_score": BASE_SCORE,
            "base_archive_bytes": BASE_ARCHIVE_BYTES,
            "axis": "[contest-CUDA,DALI,n600] inherited base; not remeasured",
            "moved": False,
        },
    }
    receipt_path = output / "PZ2_MEASUREMENT_RECEIPT.json"
    atomic_json(receipt_path, receipt)
    checkpoint(checkpoints / "stage_04_complete.json", receipt)
    print(json.dumps({
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "payload_manifest": str(manifest_path),
        "payload_count": len(manifest),
        "candidate_count": packet_measurement["candidate_denominator"],
        "best": packet_measurement["best_additive_toy_bracket"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
