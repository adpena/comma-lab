#!/usr/bin/env python3
"""DDM TM1: same-object post-logit priors for PR130's exact token field.

This scorer-free harness consumes DT1's retained *causal* int16 logit lattice
and target symbols.  It is deliberately narrower than an architecture retrain:
only decoder-computable corrections to the shipped IntegerHPAC logits are
admissible.  Learned correction bytes are stored in the candidate archive and
counted.  Retained logits are measurement cache only and are never shipped.

The run has durable stages and is resumable at stage/candidate boundaries.  A
fresh run creates ``run_state.json``.  A continuation must name that exact file
with ``--resume-from``; silently reusing a partial output tree is forbidden.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
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
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import constriction
import numpy as np

AXIS = "[macOS-CPU advisory, scorer-free]"
SCORE_CLAIM = False
SCHEMA = "ddm_tm1_token_model_lever.v1"

N = 600
H = 384
W = 512
K = 5
TOKENS_PER_FRAME = H * W
TOKEN_COUNT = N * TOKENS_PER_FRAME
PATCH = 64
DELTA = 2
GROUP_COUNT = (1 + DELTA) * PATCH - DELTA
LOGIT_PRECISION = 8

DEFAULT_MANIFEST = Path("/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/chunk_manifest.json")
DEFAULT_BASE_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip")
DEFAULT_BASE_ANS = Path("/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/ans_n600.bin")
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_tm1_20260809")

EXPECTED_MANIFEST_SHA256 = "23089d6f627e1da56a3f947900727e94ee4a99d1a2ce30fd582aeeac3130caea"
EXPECTED_ARCHIVE_SHA256 = "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
EXPECTED_ARCHIVE_BYTES = 191_052
EXPECTED_RANGE_SHA256 = "948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb"
EXPECTED_RANGE_BYTES = 116_980
EXPECTED_ANS_SHA256 = "a0b18dc0803ef541d3eb265bba5380f7aa067593f6af584b0891ded5bdd74488"
EXPECTED_ANS_BYTES = 114_860
EXPECTED_RAW_TOKEN_SHA256 = "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"
EXPECTED_MODELS_XZ_BYTES = 73_968
EXPECTED_MODELS_RAW_BYTES = 83_493
EXPECTED_MODELS_RAW_SHA256 = "62dd72dfa0858a25ca32bdee1e536627a17883b6fc7efd7cd5b2de7b13b84517"
EXPECTED_IDEAL_BYTES = 114_851.8102562373

# The charter's leave-one-out convention.  Standalone diagnostics use 15,164 B.
BASELINE_HPAC_MARGINAL_BYTES = 15_092
BASELINE_HPAC_STANDALONE_BYTES = 15_164
STRICT_JOINT_TARGET_BYTES = EXPECTED_IDEAL_BYTES + BASELINE_HPAC_MARGINAL_BYTES
RATE_DENOMINATOR = 37_545_489
BASE_SCORE = 0.172141297491896447

SEED = 20260809
SAMPLE_SIZE = 120
FOLD_COUNT = 5
ADDITIVE_SHRINK = 0.5
SMOOTHING = 0.5
MARGIN_EDGES = np.asarray((1, 3, 7, 15, 31, 63, 127), dtype=np.int32)

LZMA_FILTERS = [
    {
        "id": lzma.FILTER_LZMA2,
        "dict_size": 1 << 16,
        "lc": 0,
        "lp": 1,
        "pb": 0,
        "mode": lzma.MODE_NORMAL,
        "nice_len": 273,
        "mf": lzma.MF_BT4,
        "depth": 0,
    }
]

SIDE_ENVELOPE_MAGIC = b"TM1P"
SIDE_TRAILER_MAGIC = b"T1E1"
CANDIDATE_IDS = {
    "temperature": 1,
    "class_bias": 2,
    "confidence_lut": 3,
    "frame_block_bias": 4,
    "global_tile_bias": 5,
    "temporal_reversion": 6,
}
CANDIDATES = tuple(CANDIDATE_IDS)
ADDITIVE_CONTEXTS = {
    "class_bias": 1,
    "confidence_lut": K * (len(MARGIN_EDGES) + 1),
    "frame_block_bias": N // 30,
    "global_tile_bias": 4 * 4,
}
HYPERPARAMETER_GRIDS: dict[str, tuple[dict[str, float], ...]] = {
    "temperature": tuple({"shrink": value} for value in (0.25, 0.5, 0.75, 1.0)),
    "class_bias": tuple(
        {"shrink": shrink, "smoothing": smoothing} for shrink in (0.25, 0.5, 0.75, 1.0) for smoothing in (0.1, 0.5)
    ),
    "confidence_lut": tuple(
        {"shrink": shrink, "smoothing": smoothing} for shrink in (0.25, 0.5, 0.75, 1.0) for smoothing in (0.5, 2.0)
    ),
    "frame_block_bias": tuple(
        {"shrink": shrink, "smoothing": smoothing} for shrink in (0.25, 0.5, 0.75, 1.0) for smoothing in (0.5, 2.0)
    ),
    "global_tile_bias": tuple(
        {"shrink": shrink, "smoothing": smoothing} for shrink in (0.25, 0.5, 0.75, 1.0) for smoothing in (0.5, 2.0)
    ),
    "temporal_reversion": tuple({"shrink": value} for value in (0.25, 0.5, 0.75, 1.0)),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    atomic_bytes(path, encoded)


def atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def host_facts() -> dict[str, Any]:
    return {
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "constriction": getattr(constriction, "__version__", "0.5.0"),
    }


def codec_environment() -> dict[str, Any]:
    try:
        import brotli

        brotli_version: str | None = getattr(brotli, "__version__", "unknown")
    except ImportError:
        brotli_version = None
    return {
        "numpy": np.__version__,
        "constriction": getattr(constriction, "__version__", "0.5.0"),
        "brotli": brotli_version,
        "sidecar_codec_order": [
            "raw",
            "zlib9",
            "lzma2",
            *(["brotli11"] if brotli_version is not None else []),
        ],
    }


def stratified_frames(seed: int = SEED) -> np.ndarray:
    """One seeded random frame from each of 120 contiguous five-frame strata."""

    rng = np.random.default_rng(seed)
    frames = np.asarray(
        [rng.integers(start, start + 5) for start in range(0, N, 5)],
        dtype=np.int16,
    )
    if len(frames) != SAMPLE_SIZE or len(np.unique(frames)) != SAMPLE_SIZE:
        raise RuntimeError("stratified frame selection is not one-per-stratum")
    if np.array_equal(frames, np.arange(SAMPLE_SIZE, dtype=np.int16)):
        raise RuntimeError("development selection unexpectedly became a prefix")
    return frames


def fold_by_frame(frames: np.ndarray) -> dict[int, int]:
    return {int(frame): index % FOLD_COUNT for index, frame in enumerate(frames.tolist())}


def scan_permutation() -> np.ndarray:
    """Map HPAC group-major symbol order to row-major raster offsets."""

    yy, xx = np.indices((H, W), dtype=np.int32)
    group = (xx % PATCH) + DELTA * (yy % PATCH)
    permutation = np.concatenate([np.flatnonzero(group.ravel() == index) for index in range(GROUP_COUNT)]).astype(
        np.int32
    )
    if len(permutation) != TOKENS_PER_FRAME:
        raise RuntimeError("scan permutation has the wrong length")
    if not np.array_equal(np.sort(permutation), np.arange(TOKENS_PER_FRAME)):
        raise RuntimeError("scan permutation is not a bijection")
    return permutation


def global_tile_contexts(permutation: np.ndarray) -> np.ndarray:
    yy, xx = np.indices((H, W), dtype=np.int16)
    tile = (yy // (H // 4)) * 4 + (xx // (W // 4))
    contexts = tile.ravel()[permutation].astype(np.int16)
    if contexts.min() != 0 or contexts.max() != 15:
        raise RuntimeError("global tile contexts do not cover 4x4 tiles")
    return contexts


@dataclass(frozen=True)
class ManifestRow:
    start: int
    end: int
    symbols: Path
    codes: Path
    symbols_sha256: str
    codes_sha256: str


class RetainedCorpus:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path.resolve()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is not True:
            raise RuntimeError("DT1 retained manifest is not complete")
        self.rows = tuple(
            ManifestRow(
                start=int(row["start_frame"]),
                end=int(row["end_frame"]),
                symbols=Path(row["symbols_path"]).resolve(),
                codes=Path(row["codes_path"]).resolve(),
                symbols_sha256=str(row["symbols_sha256"]),
                codes_sha256=str(row["codes_sha256"]),
            )
            for row in manifest["chunks"]
        )
        self._arrays: dict[Path, np.ndarray] = {}
        self._frame_to_row: list[tuple[int, int] | None] = [None] * N
        expected_start = 0
        for row_index, row in enumerate(self.rows):
            if row.start != expected_start or row.end <= row.start:
                raise RuntimeError("DT1 chunk frames are not contiguous")
            for frame in range(row.start, row.end):
                self._frame_to_row[frame] = (row_index, frame - row.start)
            expected_start = row.end
        if expected_start != N or any(item is None for item in self._frame_to_row):
            raise RuntimeError("DT1 chunk manifest does not cover n600")

    def validate(self, *, deep_hash: bool) -> dict[str, Any]:
        if sha256_file(self.manifest_path) != EXPECTED_MANIFEST_SHA256:
            raise RuntimeError("DT1 manifest SHA-256 pin failed")
        token_total = 0
        checked = 0
        for row in self.rows:
            symbols = self._load(row.symbols)
            codes = self._load(row.codes)
            expected = (row.end - row.start) * TOKENS_PER_FRAME
            if symbols.dtype != np.uint8 or symbols.shape != (expected,):
                raise RuntimeError(f"invalid symbol array: {row.symbols}")
            if codes.dtype != np.int16 or codes.shape != (expected, K):
                raise RuntimeError(f"invalid code array: {row.codes}")
            if deep_hash:
                if sha256_file(row.symbols) != row.symbols_sha256:
                    raise RuntimeError(f"symbol SHA-256 mismatch: {row.symbols}")
                if sha256_file(row.codes) != row.codes_sha256:
                    raise RuntimeError(f"code SHA-256 mismatch: {row.codes}")
                checked += 2
            token_total += expected
        if token_total != TOKEN_COUNT:
            raise RuntimeError("retained corpus token denominator changed")
        return {
            "manifest": str(self.manifest_path),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "chunks": len(self.rows),
            "tokens": token_total,
            "deep_hash_files_checked": checked,
        }

    def _load(self, path: Path) -> np.ndarray:
        if path not in self._arrays:
            self._arrays[path] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._arrays[path]

    def frame(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        located = self._frame_to_row[frame]
        if located is None:
            raise IndexError(frame)
        row_index, local = located
        row = self.rows[row_index]
        start = local * TOKENS_PER_FRAME
        end = start + TOKENS_PER_FRAME
        symbols = self._load(row.symbols)[start:end]
        codes = self._load(row.codes)[start:end]
        return symbols, codes


def probability_tables(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes)
    if codes.dtype != np.int16 or codes.ndim != 2 or codes.shape[1] != K:
        raise ValueError("codes must be int16 [tokens,5]")
    logits = codes.astype(np.float64) / LOGIT_PRECISION
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def ideal_bits(tables: np.ndarray, symbols: np.ndarray) -> float:
    symbols_i64 = np.asarray(symbols, dtype=np.int64)
    probabilities = tables[np.arange(len(symbols_i64)), symbols_i64].astype(np.float64)
    return float(-np.log2(probabilities).sum())


def context_ids(
    name: str,
    codes: np.ndarray,
    frame: int,
    tile_context: np.ndarray,
) -> np.ndarray:
    if name == "class_bias":
        return np.zeros(len(codes), dtype=np.int16)
    if name == "confidence_lut":
        top = codes.argmax(axis=1).astype(np.int16)
        second = np.partition(codes, kth=K - 2, axis=1)[:, -2]
        margin = codes[np.arange(len(codes)), top].astype(np.int32) - second.astype(np.int32)
        margin_bin = np.digitize(margin, MARGIN_EDGES, right=False).astype(np.int16)
        return top * (len(MARGIN_EDGES) + 1) + margin_bin
    if name == "frame_block_bias":
        return np.full(len(codes), frame // 30, dtype=np.int16)
    if name == "global_tile_bias":
        return tile_context
    raise ValueError(f"candidate has no additive contexts: {name}")


def _round_div_half_away(numerator: np.ndarray, denominator: int) -> np.ndarray:
    magnitude = (np.abs(numerator) + denominator // 2) // denominator
    return np.where(numerator < 0, -magnitude, magnitude)


@dataclass(frozen=True)
class CandidateModel:
    name: str
    scale_q8: int | None = None
    corrections: np.ndarray | None = None

    def validate(self) -> None:
        if self.name == "temperature":
            if self.scale_q8 is None or not 128 <= self.scale_q8 <= 384:
                raise ValueError("temperature scale must be Q8 in [0.5,1.5]")
            if self.corrections is not None:
                raise ValueError("temperature cannot carry additive corrections")
            return
        if self.name == "temporal_reversion":
            if self.scale_q8 is not None or self.corrections is None:
                raise ValueError("temporal reversion payload is incomplete")
            if self.corrections.dtype != np.int8 or self.corrections.shape != (K, K):
                raise ValueError("temporal reversion table has the wrong geometry")
            if np.any(np.diag(self.corrections) != 0):
                raise ValueError("temporal reversion diagonal must be zero")
            return
        expected_rows = ADDITIVE_CONTEXTS[self.name]
        if self.scale_q8 is not None or self.corrections is None:
            raise ValueError("additive candidate payload is incomplete")
        if self.corrections.dtype != np.int8 or self.corrections.shape != (expected_rows, K):
            raise ValueError("additive correction table has the wrong geometry")
        if not np.all(self.corrections[:, 0] == 0):
            raise ValueError("additive correction gauge must pin class 0 to zero")


def apply_candidate(
    model: CandidateModel,
    codes: np.ndarray,
    frame: int,
    tile_context: np.ndarray,
    *,
    previous_one: np.ndarray | None = None,
    previous_two: np.ndarray | None = None,
) -> np.ndarray:
    model.validate()
    base = np.asarray(codes, dtype=np.int32)
    if model.name == "temperature":
        centered = base - base.max(axis=1, keepdims=True)
        scaled = _round_div_half_away(centered * int(model.scale_q8), 256)
        return np.clip(scaled, -32768, 32767).astype(np.int16)
    if model.name == "temporal_reversion":
        if previous_one is None or previous_two is None:
            raise ValueError("temporal reversion requires two causal prior frames")
        previous_one = np.asarray(previous_one, dtype=np.uint8)
        previous_two = np.asarray(previous_two, dtype=np.uint8)
        if previous_one.shape != (len(codes),) or previous_two.shape != (len(codes),):
            raise ValueError("temporal reversion prior-frame geometry changed")
        corrected = base.copy()
        changed = previous_one != previous_two
        indices = np.flatnonzero(changed)
        prior_one = previous_one[indices].astype(np.int64)
        prior_two = previous_two[indices].astype(np.int64)
        corrected[indices, prior_two] += model.corrections[
            prior_one,
            prior_two,
        ].astype(np.int32)
        return np.clip(corrected, -32768, 32767).astype(np.int16)
    contexts = context_ids(model.name, codes, frame, tile_context)
    corrected = base + model.corrections[contexts].astype(np.int32)
    return np.clip(corrected, -32768, 32767).astype(np.int16)


def _add_context_stats(
    counts: np.ndarray,
    predicted: np.ndarray,
    contexts: np.ndarray,
    symbols: np.ndarray,
    tables: np.ndarray,
) -> None:
    rows = counts.shape[0]
    counts += np.bincount(
        contexts.astype(np.int64) * K + symbols.astype(np.int64),
        minlength=rows * K,
    ).reshape(rows, K)
    flat_index = contexts.astype(np.int64)[:, None] * K + np.arange(K, dtype=np.int64)
    predicted += np.bincount(
        flat_index.ravel(), weights=tables.astype(np.float64, copy=False).ravel(), minlength=rows * K
    ).reshape(rows, K)


def _temperature_stats(codes: np.ndarray, symbols: np.ndarray, tables: np.ndarray) -> tuple[float, float]:
    logits = codes.astype(np.float64) / LOGIT_PRECISION
    logits -= logits.max(axis=1, keepdims=True)
    mean = (tables.astype(np.float64) * logits).sum(axis=1)
    selected = logits[np.arange(len(symbols)), symbols.astype(np.int64)]
    gradient = float((mean - selected).sum())
    second = (tables.astype(np.float64) * np.square(logits)).sum(axis=1)
    hessian = float(np.maximum(second - np.square(mean), 0.0).sum())
    return gradient, hessian


def compute_stats(
    corpus: RetainedCorpus,
    sample_frames: np.ndarray,
    tile_context: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {
        "temperature_full": np.zeros(2, dtype=np.float64),
        "temperature_fold": np.zeros((FOLD_COUNT, 2), dtype=np.float64),
        "baseline_full_bits": np.zeros(1, dtype=np.float64),
        "baseline_fold_bits": np.zeros(FOLD_COUNT, dtype=np.float64),
        "temporal_reversion_full": np.zeros((K * K, 3), dtype=np.float64),
        "temporal_reversion_fold": np.zeros((FOLD_COUNT, K * K, 3), dtype=np.float64),
    }
    for name, rows in ADDITIVE_CONTEXTS.items():
        arrays[f"{name}_full_counts"] = np.zeros((rows, K), dtype=np.float64)
        arrays[f"{name}_full_predicted"] = np.zeros((rows, K), dtype=np.float64)
        arrays[f"{name}_fold_counts"] = np.zeros((FOLD_COUNT, rows, K), dtype=np.float64)
        arrays[f"{name}_fold_predicted"] = np.zeros((FOLD_COUNT, rows, K), dtype=np.float64)

    folds = fold_by_frame(sample_frames)
    prior_one = np.zeros(TOKENS_PER_FRAME, dtype=np.uint8)
    prior_two = np.zeros(TOKENS_PER_FRAME, dtype=np.uint8)
    started = time.perf_counter()
    for frame in range(N):
        symbols_raw, codes_raw = corpus.frame(frame)
        symbols = np.asarray(symbols_raw, dtype=np.uint8)
        codes = np.asarray(codes_raw, dtype=np.int16)
        tables = probability_tables(codes)
        bits = ideal_bits(tables, symbols)
        arrays["baseline_full_bits"][0] += bits
        temperature = _temperature_stats(codes, symbols, tables)
        arrays["temperature_full"] += temperature
        fold = folds.get(frame)
        if fold is not None:
            arrays["baseline_fold_bits"][fold] += bits
            arrays["temperature_fold"][fold] += temperature
        transition = prior_one.astype(np.int16) * K + prior_two.astype(np.int16)
        changed_prior = prior_one != prior_two
        changed_indices = np.flatnonzero(changed_prior)
        if len(changed_indices):
            transition_changed = transition[changed_indices]
            prior_two_changed = prior_two[changed_indices].astype(np.int64)
            selected_probability = tables[
                changed_indices,
                prior_two_changed,
            ].astype(np.float64)
            temporal = np.zeros((K * K, 3), dtype=np.float64)
            temporal[:, 0] = np.bincount(
                transition_changed,
                weights=(symbols[changed_indices].astype(np.int64) == prior_two_changed),
                minlength=K * K,
            )
            temporal[:, 1] = np.bincount(
                transition_changed,
                weights=selected_probability,
                minlength=K * K,
            )
            temporal[:, 2] = np.bincount(
                transition_changed,
                weights=selected_probability * (1.0 - selected_probability),
                minlength=K * K,
            )
            arrays["temporal_reversion_full"] += temporal
            if fold is not None:
                arrays["temporal_reversion_fold"][fold] += temporal
        for name in ADDITIVE_CONTEXTS:
            contexts = context_ids(name, codes, frame, tile_context)
            _add_context_stats(
                arrays[f"{name}_full_counts"],
                arrays[f"{name}_full_predicted"],
                contexts,
                symbols,
                tables,
            )
            if fold is not None:
                _add_context_stats(
                    arrays[f"{name}_fold_counts"][fold],
                    arrays[f"{name}_fold_predicted"][fold],
                    contexts,
                    symbols,
                    tables,
                )
        prior_two = prior_one
        prior_one = symbols.copy()
    elapsed = time.perf_counter() - started
    measured = float(arrays["baseline_full_bits"][0] / 8.0)
    if not math.isclose(measured, EXPECTED_IDEAL_BYTES, rel_tol=0.0, abs_tol=2e-8):
        raise RuntimeError(f"baseline ideal bytes did not reproduce: {measured} != {EXPECTED_IDEAL_BYTES}")
    return arrays, {
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "tokens": TOKEN_COUNT,
        "ideal_bits": float(arrays["baseline_full_bits"][0]),
        "ideal_bytes": measured,
        "displayed_ideal_bytes": round(measured, 1),
        "elapsed_s": elapsed,
    }


def fit_temperature(
    stats: np.ndarray,
    *,
    shrink: float = ADDITIVE_SHRINK,
) -> CandidateModel:
    gradient, hessian = map(float, stats)
    if not math.isfinite(gradient) or not math.isfinite(hessian) or hessian <= 0:
        raise RuntimeError("temperature fit has invalid curvature")
    optimum = 1.0 - gradient / hessian
    if not 0.0 < shrink <= 1.0:
        raise ValueError("temperature shrink must be in (0,1]")
    shrunk = 1.0 + shrink * (optimum - 1.0)
    scale_q8 = int(np.clip(np.rint(shrunk * 256.0), 128, 384))
    model = CandidateModel("temperature", scale_q8=scale_q8)
    model.validate()
    return model


def fit_additive(
    name: str,
    counts: np.ndarray,
    predicted: np.ndarray,
    *,
    shrink: float = ADDITIVE_SHRINK,
    smoothing: float = SMOOTHING,
) -> CandidateModel:
    if counts.shape != predicted.shape or counts.shape != (ADDITIVE_CONTEXTS[name], K):
        raise ValueError("fit statistics have the wrong geometry")
    if not 0.0 < shrink <= 1.0 or smoothing <= 0.0:
        raise ValueError("additive shrink/smoothing are outside the fitted grid")
    ratio = np.log((counts + smoothing) / (predicted + smoothing))
    codes = np.rint(LOGIT_PRECISION * shrink * ratio).astype(np.int32)
    codes -= codes[:, :1]
    corrections = np.clip(codes, -127, 127).astype(np.int8)
    model = CandidateModel(name, corrections=corrections)
    model.validate()
    return model


def fit_temporal_reversion(
    stats: np.ndarray,
    *,
    shrink: float,
) -> CandidateModel:
    if stats.shape != (K * K, 3) or not 0.0 < shrink <= 1.0:
        raise ValueError("temporal reversion statistics/configuration are invalid")
    count_true = stats[:, 0]
    predicted = stats[:, 1]
    curvature = stats[:, 2]
    delta = np.divide(
        count_true - predicted,
        curvature,
        out=np.zeros_like(curvature),
        where=curvature > 0.0,
    )
    correction = np.rint(LOGIT_PRECISION * shrink * delta).astype(np.int32)
    correction = np.clip(correction, -16, 16).astype(np.int8).reshape(K, K)
    np.fill_diagonal(correction, 0)
    model = CandidateModel("temporal_reversion", corrections=correction)
    model.validate()
    return model


def fit_model(
    name: str,
    arrays: dict[str, np.ndarray],
    *,
    excluded_fold: int | None,
    hyperparameters: dict[str, float],
) -> CandidateModel:
    if name == "temperature":
        stats = arrays["temperature_full"]
        if excluded_fold is not None:
            stats = arrays["temperature_fold"].sum(axis=0) - arrays["temperature_fold"][excluded_fold]
        return fit_temperature(stats, shrink=hyperparameters["shrink"])
    if name == "temporal_reversion":
        stats = arrays["temporal_reversion_full"]
        if excluded_fold is not None:
            stats = arrays["temporal_reversion_fold"].sum(axis=0) - arrays["temporal_reversion_fold"][excluded_fold]
        return fit_temporal_reversion(stats, shrink=hyperparameters["shrink"])
    counts = arrays[f"{name}_full_counts"]
    predicted = arrays[f"{name}_full_predicted"]
    if excluded_fold is not None:
        counts = arrays[f"{name}_fold_counts"].sum(axis=0) - arrays[f"{name}_fold_counts"][excluded_fold]
        predicted = arrays[f"{name}_fold_predicted"].sum(axis=0) - arrays[f"{name}_fold_predicted"][excluded_fold]
    return fit_additive(
        name,
        counts,
        predicted,
        shrink=hyperparameters["shrink"],
        smoothing=hyperparameters["smoothing"],
    )


def model_to_raw(model: CandidateModel) -> bytes:
    model.validate()
    candidate_id = CANDIDATE_IDS[model.name]
    if model.name == "temperature":
        body = struct.pack("<h", int(model.scale_q8))
        rows = 0
    elif model.name == "temporal_reversion":
        body = model.corrections[~np.eye(K, dtype=bool)].tobytes(order="C")
        rows = K
    else:
        body = model.corrections.tobytes(order="C")
        rows = model.corrections.shape[0]
    return b"TM1C" + struct.pack("<BBHB", 1, candidate_id, rows, K) + body


def model_from_raw(raw: bytes) -> CandidateModel:
    if len(raw) < 9 or raw[:4] != b"TM1C":
        raise ValueError("invalid TM1 correction payload")
    version, candidate_id, rows, classes = struct.unpack_from("<BBHB", raw, 4)
    if version != 1 or classes != K:
        raise ValueError("unsupported TM1 correction version or class count")
    inverse = {value: key for key, value in CANDIDATE_IDS.items()}
    if candidate_id not in inverse:
        raise ValueError("unknown TM1 candidate id")
    name = inverse[candidate_id]
    body = raw[9:]
    if name == "temperature":
        if rows != 0 or len(body) != 2:
            raise ValueError("invalid temperature payload")
        model = CandidateModel(name, scale_q8=struct.unpack("<h", body)[0])
    elif name == "temporal_reversion":
        if rows != K or len(body) != K * (K - 1):
            raise ValueError("invalid temporal reversion payload")
        corrections = np.zeros((K, K), dtype=np.int8)
        corrections[~np.eye(K, dtype=bool)] = np.frombuffer(
            body,
            dtype=np.int8,
        )
        model = CandidateModel(name, corrections=corrections)
    else:
        if rows != ADDITIVE_CONTEXTS[name] or len(body) != rows * K:
            raise ValueError("invalid additive payload")
        corrections = np.frombuffer(body, dtype=np.int8).copy().reshape(rows, K)
        model = CandidateModel(name, corrections=corrections)
    model.validate()
    return model


def pack_model(model: CandidateModel) -> tuple[bytes, dict[str, Any]]:
    raw = model_to_raw(model)
    choices: dict[str, bytes] = {
        "raw": raw,
        "zlib9": zlib.compress(raw, level=9),
        "lzma2": lzma.compress(raw, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS),
    }
    try:
        import brotli

        choices["brotli11"] = brotli.compress(raw, quality=11)
    except ImportError:
        pass
    codec_order = tuple(choices)
    codec_name, payload = min(choices.items(), key=lambda item: (len(item[1]), codec_order.index(item[0])))
    codec_id = codec_order.index(codec_name)
    envelope = SIDE_ENVELOPE_MAGIC + struct.pack("<BBI", 1, codec_id, len(raw)) + payload
    decoded = unpack_model(envelope, codec_order)
    if model_to_raw(decoded) != raw:
        raise RuntimeError("packed correction model did not round-trip")
    return envelope, {
        "raw_bytes": len(raw),
        "packed_bytes": len(envelope),
        "codec": codec_name,
        "codec_order": list(codec_order),
        "candidate_payload_bytes": {name: len(value) for name, value in choices.items()},
        "sha256": sha256_bytes(envelope),
        "verified_exact": True,
    }


def unpack_model(envelope: bytes, codec_order: Iterable[str] | None = None) -> CandidateModel:
    if len(envelope) < 10 or envelope[:4] != SIDE_ENVELOPE_MAGIC:
        raise ValueError("invalid TM1 packed envelope")
    version, codec_id, raw_length = struct.unpack_from("<BBI", envelope, 4)
    if version != 1:
        raise ValueError("unsupported TM1 packed version")
    order = tuple(codec_order or ("raw", "zlib9", "lzma2", "brotli11"))
    if codec_id >= len(order):
        raise ValueError("invalid TM1 packed codec id")
    codec = order[codec_id]
    payload = envelope[10:]
    if codec == "raw":
        raw = payload
    elif codec == "zlib9":
        raw = zlib.decompress(payload)
    elif codec == "lzma2":
        raw = lzma.decompress(payload)
    elif codec == "brotli11":
        import brotli

        raw = brotli.decompress(payload)
    else:
        raise ValueError(f"unknown TM1 packed codec: {codec}")
    if len(raw) != raw_length:
        raise ValueError("TM1 packed raw length mismatch")
    return model_from_raw(raw)


def append_sidecar(models_raw: bytes, sidecar: bytes) -> bytes:
    return models_raw + sidecar + struct.pack("<I", len(sidecar)) + SIDE_TRAILER_MAGIC


def split_sidecar(candidate_raw: bytes) -> tuple[bytes, bytes]:
    if len(candidate_raw) < 8 or candidate_raw[-4:] != SIDE_TRAILER_MAGIC:
        raise ValueError("candidate model bundle lacks the TM1 trailer")
    length = struct.unpack_from("<I", candidate_raw, len(candidate_raw) - 8)[0]
    start = len(candidate_raw) - 8 - length
    if start < 0:
        raise ValueError("candidate sidecar length is out of bounds")
    return candidate_raw[:start], candidate_raw[start : start + length]


def load_base_archive(path: Path) -> tuple[bytes, bytes, dict[str, Any]]:
    if path.stat().st_size != EXPECTED_ARCHIVE_BYTES or sha256_file(path) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("PR130 base archive pin failed")
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["p"]:
            raise RuntimeError("PR130 archive member grammar changed")
        payload = archive.read("p")
    models_bytes = struct.unpack_from("<I", payload)[0]
    models_xz = payload[4 : 4 + models_bytes]
    tokens = payload[4 + models_bytes :]
    models_raw = lzma.decompress(models_xz)
    if len(models_xz) != EXPECTED_MODELS_XZ_BYTES:
        raise RuntimeError("PR130 model bundle size pin failed")
    if len(models_raw) != EXPECTED_MODELS_RAW_BYTES or sha256_bytes(models_raw) != EXPECTED_MODELS_RAW_SHA256:
        raise RuntimeError("PR130 raw model bundle pin failed")
    if len(tokens) != EXPECTED_RANGE_BYTES or sha256_bytes(tokens) != EXPECTED_RANGE_SHA256:
        raise RuntimeError("PR130 Range token field pin failed")
    rebuilt = lzma.compress(models_raw, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS)
    if rebuilt != models_xz:
        raise RuntimeError("PR130 model bundle does not reproduce under pinned filters")
    return (
        models_raw,
        tokens,
        {
            "archive": str(path.resolve()),
            "archive_bytes": path.stat().st_size,
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "models_xz_bytes": len(models_xz),
            "models_raw_bytes": len(models_raw),
            "models_raw_sha256": EXPECTED_MODELS_RAW_SHA256,
            "range_bytes": len(tokens),
            "range_sha256": EXPECTED_RANGE_SHA256,
        },
    )


def write_deterministic_zip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with zipfile.ZipFile(temporary, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, payload)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def encode_candidate(
    corpus: RetainedCorpus,
    model: CandidateModel | None,
    tile_context: np.ndarray,
) -> tuple[bytes, float, float]:
    coder = constriction.stream.stack.AnsCoder()
    family = constriction.stream.model.Categorical(perfect=False)
    total_bits = 0.0
    started = time.perf_counter()
    for frame in range(N - 1, -1, -1):
        symbols_raw, codes_raw = corpus.frame(frame)
        symbols = np.asarray(symbols_raw, dtype=np.int32)
        codes = np.asarray(codes_raw, dtype=np.int16)
        previous_one = (
            np.zeros(TOKENS_PER_FRAME, dtype=np.uint8)
            if frame < 1
            else np.asarray(corpus.frame(frame - 1)[0], dtype=np.uint8)
        )
        previous_two = (
            np.zeros(TOKENS_PER_FRAME, dtype=np.uint8)
            if frame < 2
            else np.asarray(corpus.frame(frame - 2)[0], dtype=np.uint8)
        )
        corrected = (
            codes
            if model is None
            else apply_candidate(
                model,
                codes,
                frame,
                tile_context,
                previous_one=previous_one,
                previous_two=previous_two,
            )
        )
        tables = probability_tables(corrected)
        total_bits += ideal_bits(tables, symbols)
        coder.encode_reverse(symbols, family, tables)
    blob = coder.get_compressed().astype("<u4", copy=False).tobytes(order="C")
    return blob, total_bits, time.perf_counter() - started


def decode_candidate(
    corpus: RetainedCorpus,
    model: CandidateModel | None,
    blob: bytes,
    tile_context: np.ndarray,
    permutation: np.ndarray,
) -> dict[str, Any]:
    if not blob or len(blob) % 4:
        raise RuntimeError("candidate ANS stream is not uint32-aligned")
    words = np.frombuffer(blob, dtype="<u4").astype(np.uint32, copy=False)
    coder = constriction.stream.stack.AnsCoder(words)
    family = constriction.stream.model.Categorical(perfect=False)
    raw_digest = hashlib.sha256()
    exact = True
    previous_one = np.zeros(TOKENS_PER_FRAME, dtype=np.uint8)
    previous_two = np.zeros(TOKENS_PER_FRAME, dtype=np.uint8)
    started = time.perf_counter()
    for frame in range(N):
        expected_raw, codes_raw = corpus.frame(frame)
        codes = np.asarray(codes_raw, dtype=np.int16)
        corrected = (
            codes
            if model is None
            else apply_candidate(
                model,
                codes,
                frame,
                tile_context,
                previous_one=previous_one,
                previous_two=previous_two,
            )
        )
        tables = probability_tables(corrected)
        decoded = coder.decode(family, tables).astype(np.uint8, copy=False)
        expected = np.asarray(expected_raw, dtype=np.uint8)
        exact = exact and np.array_equal(decoded, expected)
        raster = np.empty(TOKENS_PER_FRAME, dtype=np.uint8)
        raster[permutation] = decoded
        raw_digest.update(raster.tobytes(order="C"))
        previous_two = previous_one
        previous_one = decoded.copy()
    if not coder.is_empty():
        raise RuntimeError("candidate ANS decoder retained terminal state")
    digest = raw_digest.hexdigest()
    if not exact or digest != EXPECTED_RAW_TOKEN_SHA256:
        raise RuntimeError("candidate did not reconstruct the canonical n600 tokens")
    return {
        "frames": N,
        "tokens": TOKEN_COUNT,
        "exact_target_equality": True,
        "raw_token_sha256": digest,
        "ans_terminal_state_empty": True,
        "wall_s": time.perf_counter() - started,
    }


def evaluate_development(
    corpus: RetainedCorpus,
    arrays: dict[str, np.ndarray],
    sample_frames: np.ndarray,
    tile_context: np.ndarray,
) -> dict[str, Any]:
    fold_models = {
        (fold, name, config_index): fit_model(
            name,
            arrays,
            excluded_fold=fold,
            hyperparameters=config,
        )
        for fold in range(FOLD_COUNT)
        for name, configs in HYPERPARAMETER_GRIDS.items()
        for config_index, config in enumerate(configs)
    }
    frame_fold = fold_by_frame(sample_frames)
    candidate_bits = {name: np.zeros(len(HYPERPARAMETER_GRIDS[name]), dtype=np.float64) for name in CANDIDATES}
    started = time.perf_counter()
    for frame in sample_frames.tolist():
        frame = int(frame)
        symbols_raw, codes_raw = corpus.frame(frame)
        symbols = np.asarray(symbols_raw, dtype=np.uint8)
        codes = np.asarray(codes_raw, dtype=np.int16)
        previous_one = (
            np.zeros(TOKENS_PER_FRAME, dtype=np.uint8)
            if frame < 1
            else np.asarray(corpus.frame(frame - 1)[0], dtype=np.uint8)
        )
        previous_two = (
            np.zeros(TOKENS_PER_FRAME, dtype=np.uint8)
            if frame < 2
            else np.asarray(corpus.frame(frame - 2)[0], dtype=np.uint8)
        )
        fold = frame_fold[frame]
        for name, configs in HYPERPARAMETER_GRIDS.items():
            for config_index, _ in enumerate(configs):
                model = fold_models[(fold, name, config_index)]
                corrected = apply_candidate(
                    model,
                    codes,
                    frame,
                    tile_context,
                    previous_one=previous_one,
                    previous_two=previous_two,
                )
                candidate_bits[name][config_index] += ideal_bits(
                    probability_tables(corrected),
                    symbols,
                )
    baseline_bits = float(arrays["baseline_fold_bits"].sum())
    rows = []
    for name in CANDIDATES:
        configurations = []
        for config_index, config in enumerate(HYPERPARAMETER_GRIDS[name]):
            bits = float(candidate_bits[name][config_index])
            configurations.append(
                {
                    "hyperparameters": config,
                    "candidate_ideal_bytes": bits / 8.0,
                    "heldout_saved_bytes": (baseline_bits - bits) / 8.0,
                }
            )
        selected_index, selected = min(
            enumerate(configurations),
            key=lambda item: (item[1]["candidate_ideal_bytes"], item[0]),
        )
        saved_bytes = float(selected["heldout_saved_bytes"])
        rows.append(
            {
                "candidate": name,
                "selected_config_index": selected_index,
                "selected_hyperparameters": selected["hyperparameters"],
                "configuration_rows": configurations,
                "out_of_fold_frames": SAMPLE_SIZE,
                "out_of_fold_tokens": SAMPLE_SIZE * TOKENS_PER_FRAME,
                "baseline_ideal_bytes": baseline_bits / 8.0,
                "candidate_ideal_bytes": selected["candidate_ideal_bytes"],
                "heldout_saved_bytes": saved_bytes,
                "heldout_improved": saved_bytes > 0,
                "linear_n600_saved_bytes_projection": saved_bytes * (N / SAMPLE_SIZE),
            }
        )
    return {
        "schema": "ddm_tm1_development.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "selection": {
            "seed": SEED,
            "mode": "120 equal-width five-frame strata; one seeded random draw per stratum",
            "frames": [int(value) for value in sample_frames],
            "folds": FOLD_COUNT,
            "fold_assignment": "stratum_index modulo 5",
            "prefix": False,
        },
        "model_fit": {
            "method": (
                "decoder-computable residual correction; candidate-specific "
                "integer-lattice grids selected by five-fold out-of-fold ideal bytes"
            ),
            "hyperparameter_grids": HYPERPARAMETER_GRIDS,
            "no_frame_scored_by_its_own_fit": True,
        },
        "rows": rows,
        "elapsed_s": time.perf_counter() - started,
    }


def initialize_state(
    output: Path,
    resume_from: Path | None,
    identity: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    state_path = output / "run_state.json"
    if state_path.exists():
        if resume_from is None or resume_from.resolve() != state_path.resolve():
            raise RuntimeError(f"partial/existing output requires --resume-from {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema") != "ddm_tm1_run_state.v1" or state.get("input_identity") != identity:
            raise RuntimeError("resume state input identity changed")
        return state_path, state
    if resume_from is not None:
        raise RuntimeError("--resume-from was supplied but no run state exists")
    existing = sorted(str(path) for path in output.iterdir() if path.name != ".run.lock")
    if existing:
        raise RuntimeError("fresh TM1 output root must be empty; found: " + ", ".join(existing))
    state = {
        "schema": "ddm_tm1_run_state.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "input_identity": identity,
        "stages": {},
    }
    atomic_json(state_path, state)
    return state_path, state


def mark_stage(
    state_path: Path,
    state: dict[str, Any],
    name: str,
    payload: dict[str, Any],
) -> None:
    state["stages"][name] = payload
    atomic_json(state_path, state)


def _load_stats(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _candidate_archive_receipt(
    archive_path: Path,
    model: CandidateModel,
    sidecar: bytes,
    expected_ans_sha: str,
    expected_ans_bytes: int,
) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["p"]:
            raise RuntimeError("candidate archive member grammar changed")
        payload = archive.read("p")
    models_bytes = struct.unpack_from("<I", payload)[0]
    models_raw_candidate = lzma.decompress(payload[4 : 4 + models_bytes])
    tokens = payload[4 + models_bytes :]
    models_raw, parsed_sidecar = split_sidecar(models_raw_candidate)
    if sha256_bytes(models_raw) != EXPECTED_MODELS_RAW_SHA256:
        raise RuntimeError("candidate archive changed the base model bundle")
    if parsed_sidecar != sidecar or model_to_raw(unpack_model(parsed_sidecar)) != model_to_raw(model):
        raise RuntimeError("candidate archive sidecar parse-back failed")
    if len(tokens) != expected_ans_bytes or sha256_bytes(tokens) != expected_ans_sha:
        raise RuntimeError("candidate archive ANS field changed during assembly")
    return {
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "models_xz_bytes": models_bytes,
        "model_bundle_delta_bytes": models_bytes - EXPECTED_MODELS_XZ_BYTES,
        "token_bytes": len(tokens),
        "base_models_preserved_exact": True,
        "sidecar_parseback_exact": True,
        "token_field_preserved_exact": True,
        "evaluator_runnable": False,
        "evaluator_blocker": "TM1 research adapter is not yet wired into inflate.py",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_handle = (output / ".run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"TM1 output root is already active: {output}") from error
    usage = shutil.disk_usage(output)
    if usage.free < args.required_free_bytes:
        raise RuntimeError(f"storage preflight needs {args.required_free_bytes} free bytes, found {usage.free}")

    corpus = RetainedCorpus(args.manifest)
    validation = corpus.validate(deep_hash=True)
    models_raw, _, archive_identity = load_base_archive(args.base_archive)
    if args.base_ans.stat().st_size != EXPECTED_ANS_BYTES or sha256_file(args.base_ans) != EXPECTED_ANS_SHA256:
        raise RuntimeError("retained n600 ANS pin failed")
    identity = {
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "base_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "base_ans_sha256": EXPECTED_ANS_SHA256,
        "seed": SEED,
        "candidates": list(CANDIDATES),
        "hyperparameter_grids": {name: list(configs) for name, configs in HYPERPARAMETER_GRIDS.items()},
        "codec_environment": codec_environment(),
        "script_sha256": sha256_file(Path(__file__)),
    }
    state_path, state = initialize_state(output, args.resume_from, identity)
    permutation = scan_permutation()
    tile_context = global_tile_contexts(permutation)
    sample_frames = stratified_frames()

    stats_path = output / "checkpoints" / "stage_01_stats.npz"
    baseline_path = output / "checkpoints" / "stage_01_baseline.json"
    if "stats" not in state["stages"]:
        arrays, baseline = compute_stats(corpus, sample_frames, tile_context)
        atomic_savez(stats_path, arrays)
        baseline.update(
            {
                "schema": "ddm_tm1_baseline.v1",
                "manifest_validation": validation,
                "archive_identity": archive_identity,
                "retained_ans": {
                    "path": str(args.base_ans.resolve()),
                    "bytes": EXPECTED_ANS_BYTES,
                    "sha256": EXPECTED_ANS_SHA256,
                },
                "source_definition": (
                    "/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/"
                    "repro_repo/code/codec_hpac_integer.py@e34f31bc4969042c0051ac81aa3c56884419a231"
                ),
            }
        )
        atomic_json(baseline_path, baseline)
        mark_stage(
            state_path,
            state,
            "stats",
            {
                "status": "complete",
                "stats_path": str(stats_path),
                "stats_sha256": sha256_file(stats_path),
                "baseline_path": str(baseline_path),
                "baseline_sha256": sha256_file(baseline_path),
            },
        )
    else:
        stage = state["stages"]["stats"]
        if sha256_file(Path(stage["stats_path"])) != stage["stats_sha256"]:
            raise RuntimeError("resume stats checkpoint hash changed")
        if sha256_file(Path(stage["baseline_path"])) != stage["baseline_sha256"]:
            raise RuntimeError("resume baseline checkpoint hash changed")
    arrays = _load_stats(stats_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    development_path = output / "checkpoints" / "stage_02_development.json"
    if "development" not in state["stages"]:
        development = evaluate_development(corpus, arrays, sample_frames, tile_context)
        atomic_json(development_path, development)
        mark_stage(
            state_path,
            state,
            "development",
            {
                "status": "complete",
                "path": str(development_path),
                "sha256": sha256_file(development_path),
            },
        )
    else:
        stage = state["stages"]["development"]
        if sha256_file(Path(stage["path"])) != stage["sha256"]:
            raise RuntimeError("resume development checkpoint hash changed")
    development = json.loads(development_path.read_text(encoding="utf-8"))

    ans_control_dir = output / "baselines" / "ans_control"
    ans_control_path = ans_control_dir / "tokens.ans"
    ans_control_archive_path = ans_control_dir / "archive.zip"
    ans_control_receipt_path = ans_control_dir / "result.json"
    if "ans_control" not in state["stages"]:
        ans_control_blob, ans_control_ideal_bits, ans_control_encode_wall = encode_candidate(corpus, None, tile_context)
        if len(ans_control_blob) != EXPECTED_ANS_BYTES or sha256_bytes(ans_control_blob) != EXPECTED_ANS_SHA256:
            raise RuntimeError("baseline ANS coder grammar did not reproduce exactly")
        if not math.isclose(
            ans_control_ideal_bits / 8.0,
            EXPECTED_IDEAL_BYTES,
            rel_tol=0.0,
            abs_tol=2e-8,
        ):
            raise RuntimeError("baseline ANS control changed the ideal byte total")
        atomic_bytes(ans_control_path, ans_control_blob)
        ans_control_decode = decode_candidate(
            corpus,
            None,
            ans_control_blob,
            tile_context,
            permutation,
        )
        base_models_xz = lzma.compress(models_raw, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS)
        ans_control_payload = struct.pack("<I", len(base_models_xz)) + base_models_xz + ans_control_blob
        write_deterministic_zip(ans_control_archive_path, ans_control_payload)
        ans_control_receipt = {
            "schema": "ddm_tm1_ans_control.v1",
            "axis": AXIS,
            "score_claim": SCORE_CLAIM,
            "ideal_bytes": ans_control_ideal_bits / 8.0,
            "token_bytes": len(ans_control_blob),
            "token_sha256": sha256_bytes(ans_control_blob),
            "token_byte_identical_to_retained_dt1": True,
            "encode_wall_s": ans_control_encode_wall,
            "decode": ans_control_decode,
            "archive_bytes": ans_control_archive_path.stat().st_size,
            "archive_sha256": sha256_file(ans_control_archive_path),
            "archive_delta_vs_shipped_range_archive_bytes": (
                ans_control_archive_path.stat().st_size - EXPECTED_ARCHIVE_BYTES
            ),
            "base_models_preserved_exact": True,
            "evaluator_runnable": False,
            "evaluator_blocker": ("research ANS control is not wired into the shipped inflate.py"),
        }
        atomic_json(ans_control_receipt_path, ans_control_receipt)
        mark_stage(
            state_path,
            state,
            "ans_control",
            {
                "status": "complete",
                "path": str(ans_control_receipt_path),
                "sha256": sha256_file(ans_control_receipt_path),
                "archive_path": str(ans_control_archive_path),
                "archive_sha256": sha256_file(ans_control_archive_path),
                "tokens_path": str(ans_control_path),
                "tokens_sha256": sha256_file(ans_control_path),
            },
        )
    else:
        stage = state["stages"]["ans_control"]
        for path_key, hash_key in (
            ("path", "sha256"),
            ("archive_path", "archive_sha256"),
            ("tokens_path", "tokens_sha256"),
        ):
            if sha256_file(Path(stage[path_key])) != stage[hash_key]:
                raise RuntimeError(f"resume ANS control artifact changed: {stage[path_key]}")
    ans_control_receipt = json.loads(ans_control_receipt_path.read_text(encoding="utf-8"))

    development_by_name = {row["candidate"]: row for row in development["rows"]}
    full_models = {
        name: fit_model(
            name,
            arrays,
            excluded_fold=None,
            hyperparameters=development_by_name[name]["selected_hyperparameters"],
        )
        for name in CANDIDATES
    }
    candidate_rows: list[dict[str, Any]] = []
    for name in CANDIDATES:
        stage_name = f"candidate_{name}"
        candidate_dir = output / "candidates" / name
        candidate_result_path = candidate_dir / "result.json"
        if stage_name in state["stages"]:
            stage = state["stages"][stage_name]
            for path_key, hash_key in (
                ("path", "sha256"),
                ("archive_path", "archive_sha256"),
                ("archive_repeat_path", "archive_repeat_sha256"),
                ("tokens_path", "tokens_sha256"),
                ("sidecar_path", "sidecar_sha256"),
            ):
                if sha256_file(Path(stage[path_key])) != stage[hash_key]:
                    raise RuntimeError(f"resume candidate artifact changed ({path_key}): {name}")
            candidate_rows.append(json.loads(candidate_result_path.read_text(encoding="utf-8")))
            continue

        model = full_models[name]
        sidecar, pack_report = pack_model(model)
        sidecar_path = candidate_dir / "model.tm1p"
        atomic_bytes(sidecar_path, sidecar)
        blob, candidate_ideal_bits, encode_wall = encode_candidate(corpus, model, tile_context)
        token_path = candidate_dir / "tokens.ans"
        atomic_bytes(token_path, blob)
        decode = decode_candidate(corpus, model, blob, tile_context, permutation)

        candidate_models_raw = append_sidecar(models_raw, sidecar)
        candidate_models_xz = lzma.compress(candidate_models_raw, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS)
        payload = struct.pack("<I", len(candidate_models_xz)) + candidate_models_xz + blob
        archive_path = candidate_dir / "archive.zip"
        write_deterministic_zip(archive_path, payload)
        archive_repeat_path = candidate_dir / "archive.repeat.zip"
        write_deterministic_zip(archive_repeat_path, payload)
        if archive_path.read_bytes() != archive_repeat_path.read_bytes():
            raise RuntimeError("candidate archive repeat build was not byte-identical")
        archive = _candidate_archive_receipt(
            archive_path,
            model,
            sidecar,
            sha256_bytes(blob),
            len(blob),
        )
        archive["repeat_build_byte_identical"] = True
        archive["repeat_archive_path"] = str(archive_repeat_path)
        archive["repeat_archive_sha256"] = sha256_file(archive_repeat_path)
        token_delta = len(blob) - EXPECTED_ANS_BYTES
        model_delta = archive["model_bundle_delta_bytes"]
        archive_delta = archive["archive_bytes"] - EXPECTED_ARCHIVE_BYTES
        mechanism_archive_delta = archive["archive_bytes"] - ans_control_receipt["archive_bytes"]
        if mechanism_archive_delta != token_delta + model_delta:
            raise RuntimeError("candidate-vs-ANS-control archive delta does not equal token plus model deltas")
        if archive_delta != len(blob) - EXPECTED_RANGE_BYTES + model_delta:
            raise RuntimeError("candidate-vs-shipped archive delta does not equal token plus model deltas")
        charter_joint = len(blob) + BASELINE_HPAC_MARGINAL_BYTES + model_delta
        row = {
            "schema": "ddm_tm1_candidate_result.v1",
            "candidate": name,
            "axis": AXIS,
            "score_claim": SCORE_CLAIM,
            "model_direction": {
                "temperature": "global output-lattice scale",
                "class_bias": "global class-prior residual",
                "confidence_lut": "top-class and confidence-bin residual expert",
                "frame_block_bias": "coarse temporal conditioning residual",
                "global_tile_bias": "global 4x4 spatial-position prior absent from patch-local coordinates",
                "temporal_reversion": "second-order causal reversion to the t-2 decoded class",
            }[name],
            "direction_group": {
                "temperature": "global_calibration",
                "class_bias": "global_calibration",
                "confidence_lut": "confidence_conditioning",
                "frame_block_bias": "coarse_frame_conditioning",
                "global_tile_bias": "global_spatial_conditioning",
                "temporal_reversion": "second_order_temporal_context",
            }[name],
            "development": development_by_name[name],
            "selected_hyperparameters": development_by_name[name]["selected_hyperparameters"],
            "predictive_support_out_of_fold": development_by_name[name]["heldout_improved"],
            "ideal_token_bytes": candidate_ideal_bits / 8.0,
            "ideal_token_delta_bytes": candidate_ideal_bits / 8.0 - EXPECTED_IDEAL_BYTES,
            "real_token_bytes": len(blob),
            "real_token_sha256": sha256_bytes(blob),
            "delta_token_stream_bytes_vs_ans": token_delta,
            "packed_sidecar": pack_report,
            "sidecar_extension_bytes_before_joint_model_compression": len(sidecar) + 8,
            "delta_joint_model_bundle_bytes": model_delta,
            "delta_hpac_weights_bytes_full_bundle": model_delta,
            "charter_leave_one_out_joint_bytes": charter_joint,
            "charter_strict_target_bytes": STRICT_JOINT_TARGET_BYTES,
            "delta_vs_charter_strict_target_bytes": charter_joint - STRICT_JOINT_TARGET_BYTES,
            "beats_charter_strict_target": charter_joint < STRICT_JOINT_TARGET_BYTES,
            "rate_mechanism_supported": (
                charter_joint < STRICT_JOINT_TARGET_BYTES and development_by_name[name]["heldout_improved"]
            ),
            "standalone_diagnostic_joint_bytes": (len(blob) + BASELINE_HPAC_STANDALONE_BYTES + len(sidecar) + 8),
            "archive": archive,
            "artifacts": {
                "archive": str(archive_path),
                "archive_repeat": str(archive_repeat_path),
                "tokens": str(token_path),
                "sidecar": str(sidecar_path),
                "result": str(candidate_result_path),
            },
            "archive_delta_vs_shipped_range_archive_bytes": archive_delta,
            "archive_delta_vs_ans_control_bytes": mechanism_archive_delta,
            "projected_rate_score_delta": 25.0 * archive_delta / RATE_DENOMINATOR,
            "projected_score_if_all_other_components_hold": (BASE_SCORE + 25.0 * archive_delta / RATE_DENOMINATOR),
            "encode_wall_s": encode_wall,
            "decode": decode,
            "held_dseg_derivation": {
                "decoded_tokens_exact": True,
                "raw_token_sha256": EXPECTED_RAW_TOKEN_SHA256,
                "base_semantic_carrier_hpac_models_preserved_exact": True,
                "conclusion": (
                    "d_seg is held conditional on the measured research parse-back contract: "
                    "the semantic payload, renderer payload, HPAC model, and decoded token tensor "
                    "are byte-identical"
                ),
                "receiver_closed": False,
                "receiver_blocker": "candidate adapter is not wired into inflate.py",
                "score_claim": False,
            },
            "runtime": {
                "neural_forward_growth": "none",
                "measured_candidate_table_plus_ans_encode_s": encode_wall,
                "measured_candidate_table_plus_ans_decode_s": decode["wall_s"],
                "measured_retained_table_decode_delta_vs_ans_control_s": (
                    decode["wall_s"] - ans_control_receipt["decode"]["wall_s"]
                ),
                "reference_full_receiver_decode_s": 804.876493541,
                "reference_full_receiver_decode_source": (
                    "/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/"
                    "retained_n600_result.json#ans_decode.wall_s"
                ),
                "reference_full_receiver_axis": AXIS,
                "projection_boundary": (
                    "candidate adds an integer correction lookup before the existing float32 table; "
                    "contest CPU/CUDA full-job runtime remains unmeasured"
                ),
            },
        }
        atomic_json(candidate_result_path, row)
        mark_stage(
            state_path,
            state,
            stage_name,
            {
                "status": "complete",
                "path": str(candidate_result_path),
                "sha256": sha256_file(candidate_result_path),
                "archive_path": str(archive_path),
                "archive_sha256": sha256_file(archive_path),
                "archive_repeat_path": str(archive_repeat_path),
                "archive_repeat_sha256": sha256_file(archive_repeat_path),
                "tokens_path": str(token_path),
                "tokens_sha256": sha256_file(token_path),
                "sidecar_path": str(sidecar_path),
                "sidecar_sha256": sha256_file(sidecar_path),
            },
        )
        candidate_rows.append(row)

    candidate_rows.sort(
        key=lambda row: (
            not row["rate_mechanism_supported"],
            row["archive_delta_vs_ans_control_bytes"],
        )
    )
    winner = candidate_rows[0]
    distinct_directions = len({row["direction_group"] for row in candidate_rows})
    if distinct_directions < 3:
        raise RuntimeError("TM1 falsifier requires at least three distinct directions")
    any_strict_win = any(row["rate_mechanism_supported"] for row in candidate_rows)
    if "final" in state["stages"]:
        final_stage = state["stages"]["final"]
        final_path = Path(final_stage["path"])
        if sha256_file(final_path) != final_stage["sha256"]:
            raise RuntimeError("resume final receipt changed")
        return json.loads(final_path.read_text(encoding="utf-8"))
    receipt = {
        "schema": SCHEMA,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "measurement_complete": True,
        "receiver_complete": False,
        "invocation": sys.argv,
        "host": host_facts(),
        "storage_preflight": {
            "path": str(output),
            "free_bytes": usage.free,
            "required_free_bytes": args.required_free_bytes,
        },
        "provenance": identity | archive_identity | {"manifest_validation": validation},
        "baseline": baseline,
        "ans_control": ans_control_receipt,
        "development": development,
        "accounting_conventions": {
            "charter_leave_one_out_hpac_bytes": BASELINE_HPAC_MARGINAL_BYTES,
            "standalone_hpac_bytes": BASELINE_HPAC_STANDALONE_BYTES,
            "strict_charter_joint_target_bytes": STRICT_JOINT_TARGET_BYTES,
            "rate_accounting_authority": (
                "literal full candidate research-archive delta against the byte-identical ANS control"
            ),
            "admission_boundary": ("rate mechanism only; receiver admission awaits inflate.py integration"),
            "candidate_archive_runtime_status": "research parser only; not evaluator runnable",
        },
        "candidate_rows": candidate_rows,
        "winner": {
            "candidate": winner["candidate"],
            "archive_bytes": winner["archive"]["archive_bytes"],
            "archive_delta_vs_shipped_range_archive_bytes": winner["archive_delta_vs_shipped_range_archive_bytes"],
            "archive_delta_vs_ans_control_bytes": winner["archive_delta_vs_ans_control_bytes"],
            "real_token_bytes": winner["real_token_bytes"],
            "delta_hpac_weights_bytes_full_bundle": winner["delta_hpac_weights_bytes_full_bundle"],
            "delta_joint_model_bundle_bytes": winner["delta_joint_model_bundle_bytes"],
            "beats_charter_strict_target": winner["beats_charter_strict_target"],
            "predictive_support_out_of_fold": winner["predictive_support_out_of_fold"],
            "rate_mechanism_supported": winner["rate_mechanism_supported"],
        },
        "verdict": {
            "candidates_measured": len(candidate_rows),
            "distinct_direction_groups_measured": distinct_directions,
            "any_predictively_supported_rate_mechanism_below_charter_joint": any_strict_win,
            "scope": (
                "MECHANISM / RECEIVER-BLOCKED: grid-tuned decoder-computable residual priors "
                "on shipped PR130 IntegerHPAC causal logits"
            ),
            "falsifier_fired": False,
            "falsifier_reason": (
                "a candidate crossed the rate threshold"
                if any_strict_win
                else "research candidate archives are not receiver-closed"
            ),
            "formulation_closed": False,
            "token_model_axis_closed": False,
            "family_boundary": (
                "No result here closes architecture, mask/receptive-field, inherited-feature ablation, "
                "CL1 rate-lambda, or OP1R edge-context directions."
            ),
        },
        "not_measured": [
            "contest-CPU or contest-CUDA score",
            "contest-host full-job runtime",
            "a shipping inflate.py integration",
            "CL1 fixed-topology rate-lambda spending",
            "OP1R causal edge context",
            "changed-architecture logits",
        ],
    }
    receipt_path = output / "tm1_result.json"
    atomic_json(receipt_path, receipt)
    mark_stage(
        state_path,
        state,
        "final",
        {
            "status": "complete",
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
        },
    )
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-archive", type=Path, default=DEFAULT_BASE_ARCHIVE)
    parser.add_argument("--base-ans", type=Path, default=DEFAULT_BASE_ANS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--required-free-bytes", type=int, default=2_000_000_000)
    return parser.parse_args(argv)


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result["winner"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
