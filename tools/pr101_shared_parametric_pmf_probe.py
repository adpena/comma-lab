#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Shared-model PMF probe for PR101 quantized decoder weights.

This is a CPU-only planning probe for the shared-model entropy path. It tests
whether a compact shared PMF family can predict per-tensor PR101 weight
distributions without transmitting 28 explicit PMF tables.

The measured variants are deliberately fail-closed:

* parametric spike/Gaussian/Laplace canonical libraries over identity symbols;
* the same canonical libraries over a fixed modulo-255 delta transform;
* one shared empirical PMF plus per-tensor temperature/spike assignments;
* clustered canonical PMF tables with charged shared tables and assignments.

No range/ANS bitstream or runtime decoder is emitted. The output is a
deterministic byte-accounting manifest for deciding whether a runtime coder is
worth building.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_shared_parametric_pmf_probe.py"
SCHEMA_VERSION = "pr101_shared_parametric_pmf_probe.v1"
EVIDENCE_GRADE = "empirical"
EVIDENCE_MARKER = "[CPU-prep empirical]"
EVIDENCE_SEMANTICS = "cpu_shared_model_pmf_planning_probe"

N_CATEGORIES = 255
ARCHIVE_OVERHEAD_BYTES = 16_094
MODEL_HEADER_BYTES = 16
REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES = 178_144
REFERENCE_PER_TENSOR_AAC_ARCHIVE_BYTES = 178_181
REFERENCE_IID_PER_TENSOR_FLOOR_ARCHIVE_BYTES = 175_916

DEFAULT_CLUSTER_K_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
DEFAULT_SCALE_GRID_SIZE = 8
SPIKE_GRID = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70)
UNIFORM_GRID = (0.0, 0.001, 0.005, 0.02)
TEMPERATURE_GRID = tuple(float(v) for v in np.exp(np.linspace(math.log(0.20), math.log(6.0), 17)))
ZERO_BOOST_GRID = tuple(float(v) for v in np.exp(np.linspace(math.log(0.125), math.log(32.0), 13)))

AUTODISCOVER_STATE_DICT_GLOBS = (
    "experiments/results/optuna_pr101_real_substrate_*/pr101_decoder_state_dict.pt",
    "experiments/results/cma_pr101_real_substrate_cmaes_*/pr101_decoder_state_dict.pt",
    "experiments/results/cma_pr101_real_substrate_*/pr101_decoder_state_dict.pt",
    "experiments/results/*pr101*/*decoder_state_dict*.pt",
)


@dataclass(frozen=True)
class TensorSymbols:
    idx: int
    name: str
    shape: tuple[int, ...]
    symbols: np.ndarray

    @property
    def counts(self) -> np.ndarray:
        return np.bincount(self.symbols, minlength=N_CATEGORIES).astype(np.float64)

    @property
    def n_symbols(self) -> int:
        return int(self.symbols.size)


@dataclass(frozen=True)
class CandidatePMF:
    family: str
    descriptor: tuple[float, ...]
    pmf: np.ndarray


@dataclass(frozen=True)
class ModelAccounting:
    primary_model_parameter_bytes: int
    raw_parameter_bytes: int
    brotli_parameter_bytes: int
    assignment_bytes: int
    header_bytes: int
    parameter_encoding: str


def discover_default_state_dict(repo_root: Path = REPO_ROOT) -> Path | None:
    """Return the newest PR101 decoder state dict from known CMA/Optuna surfaces."""
    matches: list[Path] = []
    for pattern in AUTODISCOVER_STATE_DICT_GLOBS:
        matches.extend(repo_root.glob(pattern))
    files = sorted({path for path in matches if path.is_file()}, key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_tensor_symbols(state_dict_path: Path) -> tuple[str, list[TensorSymbols]]:
    input_sha256 = _sha256_file(state_dict_path)
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise SystemExit(f"loaded {state_dict_path} is not a dict")

    rows: list[TensorSymbols] = []
    for idx, (name, expected_shape) in enumerate(FIXED_STATE_SCHEMA):
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        shape = tuple(int(v) for v in getattr(qt, "shape", expected_shape))
        if shape != tuple(expected_shape):
            raise SystemExit(f"shape mismatch for {name!r}: schema {expected_shape}, quantized {shape}")
        q_i8 = np.asarray(qt.q_i8, dtype=np.int8)
        symbols = q_i8.astype(np.int32).reshape(-1) + 127
        if symbols.size and (int(symbols.min()) < 0 or int(symbols.max()) >= N_CATEGORIES):
            raise SystemExit(f"quantized symbols outside [0, {N_CATEGORIES - 1}] for {name!r}")
        rows.append(TensorSymbols(idx=idx, name=name, shape=shape, symbols=symbols.astype(np.int32)))
    return input_sha256, rows


def _delta_mod255(symbols: np.ndarray) -> np.ndarray:
    if symbols.size == 0:
        return symbols.astype(np.int32)
    deltas = np.empty_like(symbols, dtype=np.int32)
    deltas[0] = int(symbols[0])
    deltas[1:] = (symbols[1:] - symbols[:-1]) % N_CATEGORIES
    return deltas


def _transform_rows(rows: list[TensorSymbols], transform: str) -> list[TensorSymbols]:
    if transform == "identity":
        return rows
    if transform == "delta_mod255":
        return [
            TensorSymbols(idx=row.idx, name=row.name, shape=row.shape, symbols=_delta_mod255(row.symbols))
            for row in rows
        ]
    raise SystemExit(f"unknown transform: {transform}")


def _entropy_bits_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    nz = counts > 0
    probs = counts[nz] / total
    return float(-(counts[nz] * np.log2(probs)).sum())


def _counts_cost_bits(counts: np.ndarray, pmf: np.ndarray) -> float:
    observed = counts > 0
    if np.any(pmf[observed] <= 0):
        return math.inf
    return float(-(counts[observed] * np.log2(pmf[observed])).sum())


def _pmf_from_counts(counts: np.ndarray) -> np.ndarray:
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("cannot build PMF from empty counts")
    pmf = np.zeros_like(counts, dtype=np.float64)
    observed = counts > 0
    pmf[observed] = counts[observed] / total
    return pmf


def _fp16_roundtrip_pmfs(pmfs: np.ndarray) -> np.ndarray:
    rounded = pmfs.astype(np.float16).astype(np.float64)
    totals = rounded.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("fp16 PMF table rounded to all zeros")
    return rounded / totals


def _assignment_bytes(n_tensors: int, n_models: int) -> int:
    if n_models <= 1:
        return 0
    return math.ceil(n_tensors * math.ceil(math.log2(n_models)) / 8)


def _descriptor_blob(candidates: list[CandidatePMF]) -> bytes:
    parts: list[bytes] = []
    family_ids = {
        "laplace": 1,
        "gaussian": 2,
        "temperature_spike": 3,
    }
    for candidate in candidates:
        parts.append(struct.pack("<B", family_ids[candidate.family]))
        parts.append(np.asarray(candidate.descriptor, dtype=np.float16).tobytes())
    return b"".join(parts)


def _table_accounting(table_blob: bytes, assignment_bytes: int, *, encoding_name: str) -> ModelAccounting:
    raw_parameter_bytes = len(table_blob)
    brotli_parameter_bytes = len(brotli.compress(table_blob, quality=11))
    primary = brotli_parameter_bytes + assignment_bytes + MODEL_HEADER_BYTES
    return ModelAccounting(
        primary_model_parameter_bytes=primary,
        raw_parameter_bytes=raw_parameter_bytes,
        brotli_parameter_bytes=brotli_parameter_bytes,
        assignment_bytes=assignment_bytes,
        header_bytes=MODEL_HEADER_BYTES,
        parameter_encoding=encoding_name,
    )


def _distance(values: np.ndarray, center: int, *, circular: bool) -> np.ndarray:
    values_i = values.astype(np.int32)
    if not circular:
        return np.abs(values_i - center).astype(np.float64)
    return np.minimum((values_i - center) % N_CATEGORIES, (center - values_i) % N_CATEGORIES).astype(np.float64)


def _discrete_laplace_pmf(center: int, scale: float, *, circular: bool, spike: float, uniform: float) -> np.ndarray:
    symbols = np.arange(N_CATEGORIES, dtype=np.int32)
    dist = _distance(symbols, center, circular=circular)
    safe_scale = max(float(scale), 1e-3)
    base = np.exp(-dist / safe_scale)
    base /= base.sum()
    pmf = (1.0 - uniform) * ((1.0 - spike) * base)
    pmf += uniform / N_CATEGORIES
    pmf[center] += (1.0 - uniform) * spike
    pmf = np.clip(pmf, 1e-12, None)
    return pmf / pmf.sum()


def _discrete_gaussian_pmf(center: int, std: float, *, circular: bool, spike: float, uniform: float) -> np.ndarray:
    symbols = np.arange(N_CATEGORIES, dtype=np.int32)
    dist = _distance(symbols, center, circular=circular)
    safe_std = max(float(std), 1e-3)
    base = np.exp(-0.5 * (dist / safe_std) ** 2)
    base /= base.sum()
    pmf = (1.0 - uniform) * ((1.0 - spike) * base)
    pmf += uniform / N_CATEGORIES
    pmf[center] += (1.0 - uniform) * spike
    pmf = np.clip(pmf, 1e-12, None)
    return pmf / pmf.sum()


def _quantile_grid(values: list[float], grid_size: int) -> list[float]:
    if grid_size <= 0:
        raise SystemExit("--scale-grid-size must be > 0")
    if grid_size == 1:
        return [float(np.percentile(values, 50))]
    quantiles = np.linspace(0.0, 100.0, grid_size + 2)[1:-1]
    return [max(float(value), 1e-3) for value in np.percentile(values, quantiles)]


def _build_parametric_candidates(rows: list[TensorSymbols], *, transform: str, scale_grid_size: int) -> list[CandidatePMF]:
    center = 127 if transform == "identity" else 0
    circular = transform == "delta_mod255"
    laplace_scales: list[float] = []
    gaussian_stds: list[float] = []
    for row in rows:
        dist = _distance(row.symbols, center, circular=circular)
        laplace_scales.append(max(float(np.mean(dist)), 1e-3))
        gaussian_stds.append(max(float(np.sqrt(np.mean(dist * dist))), 1e-3))

    candidates: list[CandidatePMF] = []
    for scale in _quantile_grid(laplace_scales, scale_grid_size):
        for spike in SPIKE_GRID:
            for uniform in UNIFORM_GRID:
                candidates.append(
                    CandidatePMF(
                        family="laplace",
                        descriptor=(float(center), float(circular), scale, spike, uniform),
                        pmf=_discrete_laplace_pmf(
                            center,
                            scale,
                            circular=circular,
                            spike=spike,
                            uniform=uniform,
                        ),
                    )
                )
    for std in _quantile_grid(gaussian_stds, scale_grid_size):
        for spike in SPIKE_GRID:
            for uniform in UNIFORM_GRID:
                candidates.append(
                    CandidatePMF(
                        family="gaussian",
                        descriptor=(float(center), float(circular), std, spike, uniform),
                        pmf=_discrete_gaussian_pmf(
                            center,
                            std,
                            circular=circular,
                            spike=spike,
                            uniform=uniform,
                        ),
                    )
                )
    return candidates


def _build_temperature_candidates(rows: list[TensorSymbols]) -> list[CandidatePMF]:
    pooled_counts = np.sum([row.counts for row in rows], axis=0)
    pooled_pmf = _pmf_from_counts(pooled_counts)
    candidates: list[CandidatePMF] = []
    for alpha in TEMPERATURE_GRID:
        for zero_boost in ZERO_BOOST_GRID:
            pmf = np.zeros_like(pooled_pmf)
            observed = pooled_pmf > 0
            pmf[observed] = pooled_pmf[observed] ** alpha
            pmf[127] *= zero_boost
            pmf /= pmf.sum()
            candidates.append(
                CandidatePMF(
                    family="temperature_spike",
                    descriptor=(alpha, zero_boost),
                    pmf=pmf,
                )
            )
    return candidates


def _summarize_assigned_model(
    *,
    name: str,
    transform: str,
    model_family: str,
    rows: list[TensorSymbols],
    pmfs: np.ndarray,
    accounting: ModelAccounting,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counts = np.array([row.counts for row in rows])
    cost = np.stack([[_counts_cost_bits(row_counts, pmf) for pmf in pmfs] for row_counts in counts], axis=0)
    assignments = cost.argmin(axis=1)
    chosen_bits = cost[np.arange(len(rows)), assignments]
    if not np.all(np.isfinite(chosen_bits)):
        raise ValueError(f"{name} assigned an impossible PMF")

    payload_bits = float(chosen_bits.sum())
    payload_bytes = math.ceil(payload_bits / 8.0)
    archive_estimate = payload_bytes + accounting.primary_model_parameter_bytes + ARCHIVE_OVERHEAD_BYTES
    row: dict[str, Any] = {
        "name": name,
        "transform": transform,
        "model_family": model_family,
        "n_models": int(pmfs.shape[0]),
        "used_models": len({int(v) for v in assignments}),
        "n_tensors": len(rows),
        "n_symbols": int(sum(tensor.n_symbols for tensor in rows)),
        "payload_estimate_bits": payload_bits,
        "payload_estimate_bytes": payload_bytes,
        "model_parameter_bytes": accounting.primary_model_parameter_bytes,
        "model_parameter_byte_estimate": {
            "primary_model_parameter_bytes": accounting.primary_model_parameter_bytes,
            "raw_parameter_bytes": accounting.raw_parameter_bytes,
            "brotli_parameter_bytes": accounting.brotli_parameter_bytes,
            "assignment_bytes": accounting.assignment_bytes,
            "header_bytes": accounting.header_bytes,
            "parameter_encoding": accounting.parameter_encoding,
        },
        "archive_estimate_bytes": archive_estimate,
        "delta_vs_178144": archive_estimate - REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES,
        "delta_vs_brotli_optuna_archive_bytes": archive_estimate - REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES,
        "delta_vs_178181": archive_estimate - REFERENCE_PER_TENSOR_AAC_ARCHIVE_BYTES,
        "delta_vs_per_tensor_aac_archive_bytes": archive_estimate - REFERENCE_PER_TENSOR_AAC_ARCHIVE_BYTES,
        "delta_vs_175916": archive_estimate - REFERENCE_IID_PER_TENSOR_FLOOR_ARCHIVE_BYTES,
        "delta_vs_iid_per_tensor_floor_archive_bytes": archive_estimate - REFERENCE_IID_PER_TENSOR_FLOOR_ARCHIVE_BYTES,
        "verdict": (
            "positive_estimate_not_dispatchable_without_runtime_coder"
            if archive_estimate < REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES
            else "negative_loses_after_charging_model_bytes"
        ),
        "fit_on_same_symbols": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "per_tensor_assignments": [
            {
                "idx": tensor.idx,
                "name": tensor.name,
                "n_symbols": tensor.n_symbols,
                "n_unique_symbols": int(np.count_nonzero(tensor.counts)),
                "assigned_model": int(assignments[i]),
                "model_bits": float(chosen_bits[i]),
                "model_bytes": math.ceil(float(chosen_bits[i]) / 8.0),
            }
            for i, tensor in enumerate(rows)
        ],
    }
    if extra:
        row.update(extra)
    return row


def _summarize_parametric_library(
    rows: list[TensorSymbols],
    *,
    transform: str,
    scale_grid_size: int,
) -> dict[str, Any]:
    candidates = _build_parametric_candidates(rows, transform=transform, scale_grid_size=scale_grid_size)
    pmfs = np.array([candidate.pmf for candidate in candidates], dtype=np.float64)
    assignment_bytes = _assignment_bytes(len(rows), len(candidates))
    accounting = _table_accounting(
        _descriptor_blob(candidates),
        assignment_bytes,
        encoding_name="brotli_fp16_param_descriptors_plus_assignments",
    )
    return _summarize_assigned_model(
        name=f"shared_parametric_spike_laplace_gaussian_{transform}",
        transform=transform,
        model_family="shared_spike_laplace_gaussian_grid",
        rows=rows,
        pmfs=pmfs,
        accounting=accounting,
        extra={
            "scale_grid_size": scale_grid_size,
            "spike_grid": list(SPIKE_GRID),
            "uniform_grid": list(UNIFORM_GRID),
        },
    )


def _summarize_temperature_model(rows: list[TensorSymbols]) -> dict[str, Any]:
    candidates = _build_temperature_candidates(rows)
    pmfs = np.array([candidate.pmf for candidate in candidates], dtype=np.float64)
    pooled_counts = np.sum([row.counts for row in rows], axis=0)
    pooled_pmf_blob = _pmf_from_counts(pooled_counts).astype(np.float16).tobytes()
    descriptor_blob = _descriptor_blob(candidates)
    assignment_bytes = _assignment_bytes(len(rows), len(candidates))
    accounting = _table_accounting(
        pooled_pmf_blob + descriptor_blob,
        assignment_bytes,
        encoding_name="brotli_global_fp16_pmf_and_temperature_descriptors_plus_assignments",
    )
    return _summarize_assigned_model(
        name="shared_empirical_temperature_spike_identity",
        transform="identity",
        model_family="shared_empirical_pmf_temperature_spike_residual",
        rows=rows,
        pmfs=pmfs,
        accounting=accounting,
        extra={
            "temperature_grid": list(TEMPERATURE_GRID),
            "zero_boost_grid": list(ZERO_BOOST_GRID),
        },
    )


def _cluster_initial_orders(counts: np.ndarray) -> dict[str, list[int]]:
    totals = counts.sum(axis=1)
    orders: dict[str, list[int]] = {
        "schema_order": list(range(counts.shape[0])),
        "largest_tensors_first": [int(v) for v in np.argsort(-totals)],
    }
    starts = sorted({
        0,
        int(np.argmax(totals)),
        *list(range(0, counts.shape[0], 2)),
    })
    for start in starts:
        selected = [start]
        while len(selected) < counts.shape[0]:
            pmfs = [_pmf_from_counts(counts[idx]) for idx in selected]
            cost = np.stack([_cluster_cost_vector(counts, pmf) for pmf in pmfs], axis=1).min(axis=1)
            cost[selected] = -1.0
            selected.append(int(np.argmax(cost)))
        orders[f"farthest_from_{start}"] = selected
    return orders


def _cluster_cost_vector(counts: np.ndarray, pmf: np.ndarray) -> np.ndarray:
    costs = np.empty(counts.shape[0], dtype=np.float64)
    for idx, row_counts in enumerate(counts):
        costs[idx] = _counts_cost_bits(row_counts, pmf)
    return costs


def _fit_cluster_pmfs(counts: np.ndarray, *, k: int, init_order: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0 or k > counts.shape[0]:
        raise SystemExit(f"cluster K must be in [1, {counts.shape[0]}], got {k}")
    pmfs: list[np.ndarray] = [_pmf_from_counts(counts.sum(axis=0))]
    for idx in init_order:
        if len(pmfs) >= k:
            break
        pmfs.append(_pmf_from_counts(counts[idx]))
    pmf_matrix = np.array(pmfs[:k], dtype=np.float64)
    assignments: np.ndarray | None = None
    for _iteration in range(100):
        cost = np.stack([_cluster_cost_vector(counts, pmf) for pmf in pmf_matrix], axis=1)
        new_assignments = cost.argmin(axis=1)
        if assignments is not None and np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
        for cluster_idx in range(k):
            mask = assignments == cluster_idx
            if np.any(mask):
                pmf_matrix[cluster_idx] = _pmf_from_counts(counts[mask].sum(axis=0))
    final_cost = np.stack([_cluster_cost_vector(counts, pmf) for pmf in pmf_matrix], axis=1)
    return pmf_matrix, final_cost.argmin(axis=1)


def _summarize_cluster_model(rows: list[TensorSymbols], *, k: int) -> dict[str, Any]:
    counts = np.array([row.counts for row in rows])
    best: tuple[float, str, np.ndarray, np.ndarray] | None = None
    for init_name, order in _cluster_initial_orders(counts).items():
        pmfs, assignments = _fit_cluster_pmfs(counts, k=k, init_order=order)
        fp16_pmfs = _fp16_roundtrip_pmfs(pmfs)
        bits = 0.0
        possible = True
        for row_idx, assigned_idx in enumerate(assignments):
            row_bits = _counts_cost_bits(counts[row_idx], fp16_pmfs[assigned_idx])
            if not math.isfinite(row_bits):
                possible = False
                break
            bits += row_bits
        if possible and (best is None or bits < best[0]):
            best = (bits, init_name, fp16_pmfs, assignments)
    if best is None:
        raise ValueError(f"no feasible cluster PMF assignment for K={k}")
    _bits, init_name, pmfs, assignments = best
    assignment_bytes = _assignment_bytes(len(rows), k)
    accounting = _table_accounting(
        pmfs.astype(np.float16).tobytes(),
        assignment_bytes,
        encoding_name="brotli_fp16_shared_pmf_tables_plus_assignments",
    )
    summary = _summarize_assigned_model(
        name=f"shared_canonical_pmf_clusters_identity_k{k}",
        transform="identity",
        model_family="shared_canonical_pmf_clusters",
        rows=rows,
        pmfs=pmfs,
        accounting=accounting,
        extra={
            "cluster_k": k,
            "cluster_init": init_name,
            "cluster_sizes": [int(np.count_nonzero(assignments == cluster_idx)) for cluster_idx in range(k)],
        },
    )
    return summary


def _baseline_per_tensor_empirical(rows: list[TensorSymbols]) -> dict[str, Any]:
    counts = np.array([row.counts for row in rows])
    iid_bits = float(sum(_entropy_bits_from_counts(row_counts) for row_counts in counts))
    empirical_pmfs = np.array([_pmf_from_counts(row_counts) for row_counts in counts])
    fp16_pmfs = _fp16_roundtrip_pmfs(empirical_pmfs)
    fp16_bits = float(
        sum(_counts_cost_bits(row_counts, fp16_pmfs[idx]) for idx, row_counts in enumerate(counts))
    )
    raw_table_bytes = len(fp16_pmfs.astype(np.float16).tobytes())
    brotli_table_bytes = len(brotli.compress(fp16_pmfs.astype(np.float16).tobytes(), quality=11))
    payload_iid_bytes = math.ceil(iid_bits / 8.0)
    payload_fp16_bytes = math.ceil(fp16_bits / 8.0)
    return {
        "name": "per_tensor_empirical_pmf_reference",
        "n_tensors": len(rows),
        "n_symbols": int(sum(row.n_symbols for row in rows)),
        "iid_floor_payload_bytes": payload_iid_bytes,
        "iid_floor_archive_bytes_without_model_tables": payload_iid_bytes + ARCHIVE_OVERHEAD_BYTES,
        "fp16_table_payload_bytes": payload_fp16_bytes,
        "raw_fp16_table_bytes": raw_table_bytes,
        "brotli_fp16_table_bytes": brotli_table_bytes,
        "archive_estimate_bytes_raw_fp16_table": payload_fp16_bytes + raw_table_bytes + ARCHIVE_OVERHEAD_BYTES,
        "archive_estimate_bytes_brotli_fp16_table": (
            payload_fp16_bytes + brotli_table_bytes + ARCHIVE_OVERHEAD_BYTES
        ),
        "delta_iid_floor_archive_vs_175916": (
            payload_iid_bytes + ARCHIVE_OVERHEAD_BYTES - REFERENCE_IID_PER_TENSOR_FLOOR_ARCHIVE_BYTES
        ),
    }


def build_shared_model_report(
    state_dict_path: Path,
    *,
    cluster_k_values: list[int] | None = None,
    scale_grid_size: int = DEFAULT_SCALE_GRID_SIZE,
) -> dict[str, Any]:
    input_sha256, base_rows = _load_tensor_symbols(state_dict_path)
    cluster_values = cluster_k_values if cluster_k_values is not None else list(DEFAULT_CLUSTER_K_VALUES)
    valid_cluster_values = [k for k in cluster_values if 1 <= k <= len(base_rows)]
    if not valid_cluster_values:
        raise SystemExit(f"no valid cluster K values for n_tensors={len(base_rows)}")

    identity_rows = _transform_rows(base_rows, "identity")
    delta_rows = _transform_rows(base_rows, "delta_mod255")
    baseline = _baseline_per_tensor_empirical(identity_rows)
    results = [
        _summarize_parametric_library(
            identity_rows,
            transform="identity",
            scale_grid_size=scale_grid_size,
        ),
        _summarize_parametric_library(
            delta_rows,
            transform="delta_mod255",
            scale_grid_size=scale_grid_size,
        ),
        _summarize_temperature_model(identity_rows),
    ]
    results.extend(_summarize_cluster_model(identity_rows, k=k) for k in valid_cluster_values)
    for row in results:
        row["sharing_gain_vs_per_tensor_raw_fp16_table_bytes"] = (
            baseline["archive_estimate_bytes_raw_fp16_table"] - row["archive_estimate_bytes"]
        )
        row["sharing_gain_vs_per_tensor_brotli_fp16_table_bytes"] = (
            baseline["archive_estimate_bytes_brotli_fp16_table"] - row["archive_estimate_bytes"]
        )

    best = min(results, key=lambda row: row["archive_estimate_bytes"])
    disposition = (
        "positive_estimate_not_dispatchable_without_runtime_coder"
        if best["archive_estimate_bytes"] < REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES
        else "planning_negative_shared_model_probe_loses_after_charged_model_bytes"
    )
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "input_state_dict": _repo_relative(state_dict_path),
        "input_state_dict_sha256": input_sha256,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_marker": EVIDENCE_MARKER,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "dispatch_blockers": [
            "planning_probe_only",
            "no_actual_range_or_ans_bitstream",
            "no_runtime_model_serializer_or_decoder",
            "model_parameter_serializations_are_estimates_not_packet_verified",
            "no_archive_substitution_performed",
            "missing_exact_cuda_auth_eval",
        ],
        "disposition": disposition,
        "cpu_only": True,
        "n_tensors": len(base_rows),
        "n_symbols": int(sum(row.n_symbols for row in base_rows)),
        "n_symbols_total": int(sum(row.n_symbols for row in base_rows)),
        "n_categories": N_CATEGORIES,
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "comparison_brotli_optuna_archive_bytes": REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES,
        "comparison_per_tensor_aac_archive_bytes": REFERENCE_PER_TENSOR_AAC_ARCHIVE_BYTES,
        "comparison_iid_per_tensor_floor_archive_bytes": REFERENCE_IID_PER_TENSOR_FLOOR_ARCHIVE_BYTES,
        "baseline_per_tensor_empirical_pmf": baseline,
        "best_model_by_archive_estimate": {
            "name": best["name"],
            "archive_estimate_bytes": best["archive_estimate_bytes"],
            "payload_estimate_bytes": best["payload_estimate_bytes"],
            "model_parameter_bytes": best["model_parameter_bytes"],
            "delta_vs_178144": best["delta_vs_178144"],
            "delta_vs_178181": best["delta_vs_178181"],
            "delta_vs_175916": best["delta_vs_175916"],
            "verdict": best["verdict"],
        },
        "model_results": results,
        "per_tensor_input_summary": [
            {
                "idx": row.idx,
                "name": row.name,
                "shape": [int(v) for v in row.shape],
                "n_symbols": row.n_symbols,
                "n_unique_symbols": int(np.count_nonzero(row.counts)),
            }
            for row in base_rows
        ],
    }


def _parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit("expected at least one integer")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict-path", "--state-dict", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cluster-k-values",
        type=str,
        default=",".join(str(k) for k in DEFAULT_CLUSTER_K_VALUES),
        help="Comma-separated shared canonical PMF cluster counts to sweep.",
    )
    parser.add_argument("--scale-grid-size", type=int, default=DEFAULT_SCALE_GRID_SIZE)
    parser.add_argument(
        "--output-evidence",
        type=Path,
        default=None,
        help="Optional cathedral_autopilot JSONL evidence row for the best shared PMF model.",
    )
    args = parser.parse_args(argv)

    state_dict_path = args.state_dict_path
    autodiscovered = False
    if state_dict_path is None:
        state_dict_path = discover_default_state_dict()
        autodiscovered = True
    if state_dict_path is None or not state_dict_path.is_file():
        raise SystemExit(
            "state_dict not found; pass --state-dict-path or place pr101_decoder_state_dict.pt "
            "under an existing PR101 CMA/Optuna result directory"
        )

    manifest = build_shared_model_report(
        state_dict_path,
        cluster_k_values=_parse_int_list(args.cluster_k_values),
        scale_grid_size=args.scale_grid_size,
    )
    manifest["autodiscovered_state_dict"] = autodiscovered
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if args.output_evidence is not None:
        best = manifest["best_model_by_archive_estimate"]
        evidence_row = {
            "technique": "shared_canonical_pmf_clusters",
            "empirical_archive_bytes": best["archive_estimate_bytes"],
            "empirical_d_seg": None,
            "empirical_d_pose": None,
            "evidence_grade": EVIDENCE_GRADE,
            "evidence_marker": EVIDENCE_MARKER,
            "evidence_semantics": EVIDENCE_SEMANTICS,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "score_affecting_payload_changed": False,
            "charged_bits_changed": False,
            "dispatch_blockers": manifest["dispatch_blockers"],
            "source": f"{EVIDENCE_MARKER} {_repo_relative(args.output)} ({best['name']})",
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row, sort_keys=True) + "\n")

    best = manifest["best_model_by_archive_estimate"]
    print(f"Wrote {args.output}")
    print(f"State dict:                    {state_dict_path}")
    print(f"Input SHA-256:                 {manifest['input_state_dict_sha256']}")
    print(f"Tensors / symbols:             {manifest['n_tensors']} / {manifest['n_symbols']:,}")
    print(f"Best shared model:             {best['name']}")
    print(f"Payload estimate:              {best['payload_estimate_bytes']:>10,} bytes")
    print(f"Model parameter bytes:         {best['model_parameter_bytes']:>10,} bytes")
    print(f"Archive estimate:              {best['archive_estimate_bytes']:>10,} bytes")
    print(f"Delta vs 178,144:              {best['delta_vs_178144']:>+10,} bytes")
    print(f"Delta vs 178,181:              {best['delta_vs_178181']:>+10,} bytes")
    print(f"Delta vs 175,916:              {best['delta_vs_175916']:>+10,} bytes")
    print(f"Disposition:                   {manifest['disposition']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
