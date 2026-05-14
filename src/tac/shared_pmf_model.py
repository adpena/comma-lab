# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import math
import struct
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

import brotli
import numpy as np

from tac.lossless.range_coder import RangeDecoder, RangeEncoder, cumulative_frequencies, normalize_probability_rows

MODEL_MAGIC = b"SPMFMOD1"
PAYLOAD_MAGIC = b"SPMFPAY1"
WIRE_VERSION = 1
DEFAULT_N_CATEGORIES = 255
DEFAULT_TOTAL_FREQUENCY = 1 << 15
DEFAULT_ALPHA = 1.0
DEFAULT_SEED = 20_260_507

_MODEL_HEADER = struct.Struct("<8sHHHHIQd")
_PAYLOAD_HEADER = struct.Struct("<8sHHIQ")


@dataclass(frozen=True)
class TensorSymbolStream:
    """One named symbol stream to code against a shared PMF model."""

    name: str
    symbols: np.ndarray
    shape: tuple[int, ...] = ()

    @property
    def n_symbols(self) -> int:
        return int(self.symbols.size)


@dataclass(frozen=True)
class SharedPMFConfig:
    """Deterministic shared-PMF fitting and wire-accounting config."""

    n_models: int = 4
    n_categories: int = DEFAULT_N_CATEGORIES
    total_frequency: int = DEFAULT_TOTAL_FREQUENCY
    alpha: float = DEFAULT_ALPHA
    seed: int = DEFAULT_SEED
    max_iterations: int = 100


@dataclass(frozen=True)
class SharedPMFModel:
    """Static shared PMF tables plus per-stream assignments."""

    config: SharedPMFConfig
    frequencies: np.ndarray
    assignments: tuple[int, ...]
    tensor_lengths: tuple[int, ...]
    init_strategy: str
    iterations: int
    estimated_payload_bits: float

    @property
    def n_models(self) -> int:
        return int(self.frequencies.shape[0])

    @property
    def n_categories(self) -> int:
        return int(self.frequencies.shape[1])

    @property
    def table_bytes_raw(self) -> int:
        return int(self.frequencies.astype("<u2", copy=False).nbytes)

    @property
    def assignment_bytes_raw(self) -> int:
        return len(self.assignments)

    @property
    def tensor_length_bytes_raw(self) -> int:
        return len(self.tensor_lengths) * 4

    @property
    def model_header_bytes(self) -> int:
        return _MODEL_HEADER.size

    @property
    def cluster_sizes(self) -> tuple[int, ...]:
        return tuple(self.assignments.count(idx) for idx in range(self.n_models))


@dataclass(frozen=True)
class SharedPMFEncodedPayload:
    payload_bytes: bytes
    range_stream_bytes: bytes

    @property
    def payload_size_bytes(self) -> int:
        return len(self.payload_bytes)

    @property
    def range_stream_size_bytes(self) -> int:
        return len(self.range_stream_bytes)

    @property
    def header_size_bytes(self) -> int:
        return _PAYLOAD_HEADER.size


@dataclass(frozen=True)
class SharedPMFProbeResult:
    """Exact byte accounting for one fitted shared-PMF model."""

    model: SharedPMFModel
    estimated_payload_bytes: int
    encoded_payload_bytes: int
    range_stream_bytes: int
    payload_header_bytes: int
    model_raw_bytes: int
    model_brotli_bytes: int
    archive_overhead_bytes: int
    archive_estimate_bytes: int
    source_symbol_sha256: str
    reconstructed_symbol_sha256: str
    model_raw_sha256: str
    model_brotli_sha256: str
    payload_sha256: str
    model_roundtrip_ok: bool
    compressed_model_roundtrip_ok: bool
    payload_roundtrip_ok: bool

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "n_models": self.model.n_models,
            "n_categories": self.model.n_categories,
            "total_frequency": self.model.config.total_frequency,
            "alpha": self.model.config.alpha,
            "deterministic_seed": self.model.config.seed,
            "init_strategy": self.model.init_strategy,
            "iterations": self.model.iterations,
            "cluster_sizes": list(self.model.cluster_sizes),
            "assignments": list(self.model.assignments),
            "payload_estimate_bits": self.model.estimated_payload_bits,
            "payload_estimate_bytes": self.estimated_payload_bytes,
            "encoded_payload_bytes": self.encoded_payload_bytes,
            "range_stream_bytes": self.range_stream_bytes,
            "payload_header_bytes": self.payload_header_bytes,
            "model_bytes": self.model_brotli_bytes,
            "model_raw_bytes": self.model_raw_bytes,
            "table_model_overhead_bytes": self.model_brotli_bytes,
            "table_model_overhead_breakdown": {
                "model_header_bytes": self.model.model_header_bytes,
                "raw_frequency_table_bytes": self.model.table_bytes_raw,
                "raw_assignment_bytes": self.model.assignment_bytes_raw,
                "raw_tensor_length_bytes": self.model.tensor_length_bytes_raw,
                "raw_model_bytes": self.model_raw_bytes,
                "brotli_model_bytes": self.model_brotli_bytes,
                "model_storage": "brotli(serialized_shared_pmf_model)",
            },
            "archive_overhead_bytes": self.archive_overhead_bytes,
            "archive_estimate_bytes": self.archive_estimate_bytes,
            "source_symbol_sha256": self.source_symbol_sha256,
            "reconstructed_symbol_sha256": self.reconstructed_symbol_sha256,
            "model_raw_sha256": self.model_raw_sha256,
            "model_brotli_sha256": self.model_brotli_sha256,
            "payload_sha256": self.payload_sha256,
            "roundtrip": {
                "model_serialization_roundtrip_ok": self.model_roundtrip_ok,
                "compressed_model_roundtrip_ok": self.compressed_model_roundtrip_ok,
                "range_payload_roundtrip_ok": self.payload_roundtrip_ok,
                "exact_reconstruction_ok": (
                    self.model_roundtrip_ok
                    and self.compressed_model_roundtrip_ok
                    and self.payload_roundtrip_ok
                    and self.source_symbol_sha256 == self.reconstructed_symbol_sha256
                ),
            },
        }


def coerce_symbol_stream(
    name: str,
    symbols: Any,
    *,
    shape: tuple[int, ...] | None = None,
    n_categories: int = DEFAULT_N_CATEGORIES,
) -> TensorSymbolStream:
    arr = np.asarray(symbols, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"symbol stream {name!r} is empty")
    if int(arr.min()) < 0 or int(arr.max()) >= n_categories:
        raise ValueError(f"symbol stream {name!r} contains symbols outside [0, {n_categories - 1}]")
    return TensorSymbolStream(name=name, symbols=arr.astype(np.int64, copy=False), shape=tuple(shape or (arr.size,)))


def symbol_stream_sha256(rows: list[TensorSymbolStream]) -> str:
    hasher = hashlib.sha256()
    for row in rows:
        arr = np.asarray(row.symbols, dtype=np.uint8).reshape(-1)
        hasher.update(len(arr).to_bytes(8, "little"))
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def _validate_config(config: SharedPMFConfig) -> None:
    if config.n_models <= 0:
        raise ValueError("n_models must be positive")
    if config.n_categories <= 1:
        raise ValueError("n_categories must be greater than 1")
    if config.total_frequency <= config.n_categories:
        raise ValueError("total_frequency must exceed n_categories so every symbol can have positive mass")
    if config.total_frequency > 65_535:
        raise ValueError("total_frequency must fit uint16 model tables")
    if config.alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")


def _validate_rows(rows: list[TensorSymbolStream], config: SharedPMFConfig) -> None:
    if not rows:
        raise ValueError("at least one symbol stream is required")
    if config.n_models > len(rows):
        raise ValueError("n_models cannot exceed the number of symbol streams")
    for row in rows:
        coerce_symbol_stream(row.name, row.symbols, shape=row.shape, n_categories=config.n_categories)


def _counts_matrix(rows: list[TensorSymbolStream], n_categories: int) -> np.ndarray:
    return np.stack(
        [np.bincount(row.symbols.astype(np.int64), minlength=n_categories).astype(np.float64) for row in rows],
        axis=0,
    )


def _frequencies_from_counts(counts: np.ndarray, config: SharedPMFConfig) -> np.ndarray:
    probs = np.asarray(counts, dtype=np.float64) + config.alpha
    return np.asarray(normalize_probability_rows(probs, total=config.total_frequency), dtype=np.int64)


def _cost_bits_matrix(counts: np.ndarray, frequencies: np.ndarray, total_frequency: int) -> np.ndarray:
    log_probs = -np.log2(frequencies.astype(np.float64) / float(total_frequency))
    return counts @ log_probs.T


def _initial_orders(counts: np.ndarray, seed: int) -> dict[str, list[int]]:
    totals = counts.sum(axis=1)
    rng = np.random.default_rng(seed)
    seeded_order = [int(idx) for idx in rng.permutation(counts.shape[0]).tolist()]
    orders = {
        "schema_order": list(range(counts.shape[0])),
        "largest_streams_first": [int(idx) for idx in np.argsort(-totals, kind="stable").tolist()],
        "seeded_permutation": seeded_order,
    }
    return orders


def _extend_order_farthest(counts: np.ndarray, order: list[int], k: int) -> list[int]:
    selected = list(dict.fromkeys(int(idx) for idx in order[:k]))
    if not selected:
        selected.append(0)
    row_totals = counts.sum(axis=1, keepdims=True)
    row_pmfs = (counts + 1.0) / (row_totals + counts.shape[1])
    while len(selected) < k:
        selected_pmfs = row_pmfs[selected]
        distances = np.abs(row_pmfs[:, None, :] - selected_pmfs[None, :, :]).sum(axis=2).min(axis=1)
        distances[selected] = -1.0
        selected.append(int(np.argmax(distances)))
    return selected[:k]


def _fit_from_order(
    counts: np.ndarray,
    tensor_lengths: tuple[int, ...],
    config: SharedPMFConfig,
    *,
    init_strategy: str,
    order: list[int],
) -> SharedPMFModel:
    selected = _extend_order_farthest(counts, order, config.n_models)
    cluster_counts = counts[selected].copy()
    frequencies = _frequencies_from_counts(cluster_counts, config)
    assignments: np.ndarray | None = None
    iterations = 0

    for iteration in range(1, config.max_iterations + 1):
        cost = _cost_bits_matrix(counts, frequencies, config.total_frequency)
        new_assignments = cost.argmin(axis=1).astype(np.int64)
        for cluster_idx in range(config.n_models):
            if np.any(new_assignments == cluster_idx):
                continue
            row_cost = cost[np.arange(counts.shape[0]), new_assignments]
            for row_idx in np.argsort(-row_cost, kind="stable").tolist():
                if np.count_nonzero(new_assignments == new_assignments[row_idx]) > 1:
                    new_assignments[row_idx] = cluster_idx
                    break
        cluster_counts = np.zeros((config.n_models, counts.shape[1]), dtype=np.float64)
        for cluster_idx in range(config.n_models):
            mask = new_assignments == cluster_idx
            if not np.any(mask):
                raise ValueError("failed to repair empty shared-PMF cluster")
            cluster_counts[cluster_idx] = counts[mask].sum(axis=0)
        new_frequencies = _frequencies_from_counts(cluster_counts, config)
        iterations = iteration
        if assignments is not None and np.array_equal(new_assignments, assignments):
            frequencies = new_frequencies
            break
        assignments = new_assignments
        frequencies = new_frequencies

    if assignments is None:
        raise ValueError("shared-PMF fitting did not assign any streams")
    payload_bits = payload_bits_for_assignments(counts, frequencies, assignments, config.total_frequency)
    return SharedPMFModel(
        config=config,
        frequencies=frequencies,
        assignments=tuple(int(value) for value in assignments.tolist()),
        tensor_lengths=tensor_lengths,
        init_strategy=init_strategy,
        iterations=iterations,
        estimated_payload_bits=payload_bits,
    )


def payload_bits_for_assignments(
    counts: np.ndarray,
    frequencies: np.ndarray,
    assignments: np.ndarray,
    total_frequency: int,
) -> float:
    cost = _cost_bits_matrix(counts, frequencies, total_frequency)
    chosen = cost[np.arange(counts.shape[0]), assignments.astype(np.int64)]
    if not np.all(np.isfinite(chosen)):
        raise ValueError("shared PMF assigned an impossible model")
    return float(chosen.sum())


def fit_shared_pmf_model(rows: list[TensorSymbolStream], config: SharedPMFConfig) -> SharedPMFModel:
    _validate_config(config)
    _validate_rows(rows, config)
    counts = _counts_matrix(rows, config.n_categories)
    tensor_lengths = tuple(row.n_symbols for row in rows)
    best: SharedPMFModel | None = None
    for init_strategy, order in _initial_orders(counts, config.seed).items():
        candidate = _fit_from_order(counts, tensor_lengths, config, init_strategy=init_strategy, order=order)
        if best is None or candidate.estimated_payload_bits < best.estimated_payload_bits:
            best = candidate
    if best is None:
        raise ValueError("no shared-PMF model candidate was fitted")
    return best


def serialize_model(model: SharedPMFModel) -> bytes:
    if model.frequencies.shape != (model.config.n_models, model.config.n_categories):
        raise ValueError("frequency table shape does not match model config")
    if len(model.assignments) != len(model.tensor_lengths):
        raise ValueError("assignment count must match tensor length count")
    if any(value < 0 or value >= model.config.n_models for value in model.assignments):
        raise ValueError("assignment outside model table range")
    if np.any(model.frequencies <= 0):
        raise ValueError("frequency tables must be positive")
    if np.any(model.frequencies > 65_535):
        raise ValueError("frequency tables must fit uint16")
    if not np.all(model.frequencies.sum(axis=1) == model.config.total_frequency):
        raise ValueError("frequency table rows must sum to total_frequency")

    header = _MODEL_HEADER.pack(
        MODEL_MAGIC,
        WIRE_VERSION,
        len(model.tensor_lengths),
        model.config.n_models,
        model.config.n_categories,
        model.config.total_frequency,
        model.config.seed,
        model.config.alpha,
    )
    lengths = np.asarray(model.tensor_lengths, dtype="<u4").tobytes()
    assignments = np.asarray(model.assignments, dtype=np.uint8).tobytes()
    frequencies = model.frequencies.astype("<u2", copy=False).tobytes()
    return header + lengths + assignments + frequencies


def deserialize_model(data: bytes) -> SharedPMFModel:
    raw = bytes(data)
    if len(raw) < _MODEL_HEADER.size:
        raise ValueError("shared-PMF model is shorter than the header")
    magic, version, n_tensors, n_models, n_categories, total_frequency, seed, alpha = _MODEL_HEADER.unpack_from(raw)
    if magic != MODEL_MAGIC or version != WIRE_VERSION:
        raise ValueError("invalid shared-PMF model header")
    if n_models <= 0 or n_categories <= 1 or n_tensors <= 0:
        raise ValueError("invalid shared-PMF model dimensions")
    offset = _MODEL_HEADER.size
    lengths_end = offset + n_tensors * 4
    assignments_end = lengths_end + n_tensors
    frequencies_end = assignments_end + n_models * n_categories * 2
    if len(raw) != frequencies_end:
        raise ValueError("shared-PMF model has trailing or truncated bytes")
    tensor_lengths = tuple(int(v) for v in np.frombuffer(raw[offset:lengths_end], dtype="<u4").tolist())
    assignments = tuple(int(v) for v in np.frombuffer(raw[lengths_end:assignments_end], dtype=np.uint8).tolist())
    frequencies = np.frombuffer(raw[assignments_end:frequencies_end], dtype="<u2").astype(np.int64)
    frequencies = frequencies.reshape(n_models, n_categories)
    config = SharedPMFConfig(
        n_models=n_models,
        n_categories=n_categories,
        total_frequency=total_frequency,
        alpha=alpha,
        seed=seed,
    )
    model = SharedPMFModel(
        config=config,
        frequencies=frequencies,
        assignments=assignments,
        tensor_lengths=tensor_lengths,
        init_strategy="deserialized",
        iterations=0,
        estimated_payload_bits=0.0,
    )
    serialize_model(model)
    return model


def compress_model(model: SharedPMFModel) -> bytes:
    return brotli.compress(serialize_model(model), quality=11)


def decompress_model(data: bytes) -> SharedPMFModel:
    return deserialize_model(brotli.decompress(bytes(data)))


def encode_rows_with_model(rows: list[TensorSymbolStream], model: SharedPMFModel) -> SharedPMFEncodedPayload:
    if len(rows) != len(model.assignments):
        raise ValueError("row count must match model assignments")
    if tuple(row.n_symbols for row in rows) != model.tensor_lengths:
        raise ValueError("row lengths must match serialized model lengths")
    cumulative_tables = [cumulative_frequencies(row.tolist()) for row in model.frequencies]
    encoder = RangeEncoder()
    for row, assignment in zip(rows, model.assignments, strict=True):
        cumulative, total = cumulative_tables[assignment]
        for symbol in row.symbols.astype(np.int64).tolist():
            encoder.encode(symbol=int(symbol), cumulative=cumulative, total=total)
    range_stream = encoder.finish()
    total_symbols = sum(model.tensor_lengths)
    header = _PAYLOAD_HEADER.pack(
        PAYLOAD_MAGIC,
        WIRE_VERSION,
        len(model.tensor_lengths),
        model.config.total_frequency,
        total_symbols,
    )
    return SharedPMFEncodedPayload(payload_bytes=header + range_stream, range_stream_bytes=range_stream)


def decode_rows_with_model(payload: bytes, model: SharedPMFModel) -> list[np.ndarray]:
    raw = bytes(payload)
    if len(raw) < _PAYLOAD_HEADER.size:
        raise ValueError("shared-PMF payload is shorter than the header")
    magic, version, n_tensors, total_frequency, total_symbols = _PAYLOAD_HEADER.unpack_from(raw)
    if magic != PAYLOAD_MAGIC or version != WIRE_VERSION:
        raise ValueError("invalid shared-PMF payload header")
    if n_tensors != len(model.tensor_lengths):
        raise ValueError("payload tensor count does not match model")
    if total_frequency != model.config.total_frequency:
        raise ValueError("payload total_frequency does not match model")
    if total_symbols != sum(model.tensor_lengths):
        raise ValueError("payload symbol count does not match model")

    cumulative_tables = [cumulative_frequencies(row.tolist()) for row in model.frequencies]
    decoder = RangeDecoder(raw[_PAYLOAD_HEADER.size :])
    restored: list[np.ndarray] = []
    for length, assignment in zip(model.tensor_lengths, model.assignments, strict=True):
        cumulative, total = cumulative_tables[assignment]
        symbols: list[int] = []
        for _ in range(length):
            target = decoder.target(total)
            symbol = bisect_right(cumulative, target) - 1
            if symbol < 0 or symbol + 1 >= len(cumulative):
                raise ValueError("range payload decoded a symbol outside the cumulative table")
            decoder.update(low_count=cumulative[symbol], high_count=cumulative[symbol + 1], total=total)
            symbols.append(symbol)
        restored.append(np.asarray(symbols, dtype=np.int64))
    return restored


def build_shared_pmf_probe_result(
    rows: list[TensorSymbolStream],
    config: SharedPMFConfig,
    *,
    archive_overhead_bytes: int,
    require_roundtrip: bool = True,
) -> SharedPMFProbeResult:
    if archive_overhead_bytes < 0:
        raise ValueError("archive_overhead_bytes must be non-negative")
    model = fit_shared_pmf_model(rows, config)
    raw_model = serialize_model(model)
    compressed_model = brotli.compress(raw_model, quality=11)
    raw_model_sha = hashlib.sha256(raw_model).hexdigest()
    compressed_model_sha = hashlib.sha256(compressed_model).hexdigest()
    model_roundtrip_ok = serialize_model(deserialize_model(raw_model)) == raw_model
    compressed_model_roundtrip_ok = serialize_model(decompress_model(compressed_model)) == raw_model

    encoded = encode_rows_with_model(rows, model)
    restored = decode_rows_with_model(encoded.payload_bytes, decompress_model(compressed_model))
    restored_rows = [
        TensorSymbolStream(name=row.name, symbols=restored_symbols, shape=row.shape)
        for row, restored_symbols in zip(rows, restored, strict=True)
    ]
    source_sha = symbol_stream_sha256(rows)
    restored_sha = symbol_stream_sha256(restored_rows)
    payload_roundtrip_ok = all(
        np.array_equal(row.symbols, restored_symbols) for row, restored_symbols in zip(rows, restored, strict=True)
    )
    if require_roundtrip and not (
        model_roundtrip_ok and compressed_model_roundtrip_ok and payload_roundtrip_ok and source_sha == restored_sha
    ):
        raise ValueError("shared-PMF artifact failed exact roundtrip")

    estimated_payload_bytes = math.ceil(model.estimated_payload_bits / 8.0)
    archive_estimate = encoded.payload_size_bytes + len(compressed_model) + archive_overhead_bytes
    return SharedPMFProbeResult(
        model=model,
        estimated_payload_bytes=estimated_payload_bytes,
        encoded_payload_bytes=encoded.payload_size_bytes,
        range_stream_bytes=encoded.range_stream_size_bytes,
        payload_header_bytes=encoded.header_size_bytes,
        model_raw_bytes=len(raw_model),
        model_brotli_bytes=len(compressed_model),
        archive_overhead_bytes=archive_overhead_bytes,
        archive_estimate_bytes=archive_estimate,
        source_symbol_sha256=source_sha,
        reconstructed_symbol_sha256=restored_sha,
        model_raw_sha256=raw_model_sha,
        model_brotli_sha256=compressed_model_sha,
        payload_sha256=hashlib.sha256(encoded.payload_bytes).hexdigest(),
        model_roundtrip_ok=model_roundtrip_ok,
        compressed_model_roundtrip_ok=compressed_model_roundtrip_ok,
        payload_roundtrip_ok=payload_roundtrip_ok,
    )

