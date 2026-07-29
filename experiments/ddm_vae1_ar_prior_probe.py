# SPDX-License-Identifier: MIT
"""Counted learned autoregressive prior probe for the DDM int4 token lattice.

This is a deliberately narrow VQ-VAE-2 lineage test.  It fits a static
categorical prior

``p(delta_t | channel, temporal_mode, delta_(t-1))``

to the exact token payload, stores every fitted frequency as a counted model
section, and range-codes the residual symbols with that model.  The decoder
uses only the counted model, the counted temporal-mode field, and its already
decoded prefix.  No fitted value is hidden in free receiver code.

The command-line runner is stage-resumable.  It preserves the fitted model,
the complete byte-close frame, and a final receipt as distinct atomic
artifacts.  It never invokes a scorer or evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import math
import os
import platform
import struct
import subprocess
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from experiments.ddm_r7_token_coder import (
    decode_token_codes,
    encode_token_codes,
    factor_mode_delta,
    pack_nibbles,
    reconstruct_mode_delta,
    unpack_nibbles,
)
from tac.optimization import ddm_tr1_runtime as tr1


class DDMVAE1PriorError(ValueError):
    """The learned-prior model, frame, or resume contract failed closed."""


MAGIC: Final = b"DVA1"
VERSION: Final = 1
MODEL_KIND: Final = 1
FREQUENCY_TOTAL: Final = 1 << 15
MAX_VALUES: Final = 16_000_000
MAX_CHANNELS: Final = 256
MAX_MODEL_BYTES: Final = 16 * 1024 * 1024
HEADER: Final = struct.Struct("<4sBBBB4HIII32s")
POINTER_LOCAL: Final = "0.1910828242 [contest-CPU] UNMOVED"
SMEVR_ENDPOINT_BYTES: Final = 557_238
SCHEMA: Final = "ddm_vae1_counted_ar_prior_probe.v1"
FORMULATION_SCOPE: Final = "STATIC_POOLED_MODE_DELTA_PREV1_COUNTED_CONFIG"
COLD_STORE_ROOTS: Final = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)

_STATE_BITS: Final = 32
_FULL_RANGE: Final = 1 << _STATE_BITS
_HALF: Final = _FULL_RANGE >> 1
_QUARTER: Final = _HALF >> 1
_THREE_QUARTERS: Final = 3 * _QUARTER


@dataclass(frozen=True, slots=True)
class LearnedPriorAccounting:
    framed_bytes: int
    header_bytes: int
    base_bytes: int
    model_bytes: int
    residual_bytes: int
    raw_model_bytes: int
    raw_token_bytes: int
    sha256: str


class _BitWriter:
    def __init__(self) -> None:
        self.output = bytearray()
        self.current = 0
        self.used = 0

    def write(self, bit: int) -> None:
        self.current = (self.current << 1) | int(bit)
        self.used += 1
        if self.used == 8:
            self.output.append(self.current)
            self.current = 0
            self.used = 0

    def finish(self) -> bytes:
        if self.used:
            self.output.append(self.current << (8 - self.used))
        return bytes(self.output)


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.byte_index = 0
        self.bit_index = 0

    def read(self) -> int:
        if self.byte_index >= len(self.payload):
            return 0
        value = (self.payload[self.byte_index] >> (7 - self.bit_index)) & 1
        self.bit_index += 1
        if self.bit_index == 8:
            self.bit_index = 0
            self.byte_index += 1
        return value


class _StaticRangeEncoder:
    def __init__(self) -> None:
        self.writer = _BitWriter()
        self.low = 0
        self.high = _FULL_RANGE - 1
        self.pending = 0

    def _emit(self, bit: int) -> None:
        self.writer.write(bit)
        while self.pending:
            self.writer.write(1 - bit)
            self.pending -= 1

    def encode(self, symbol: int, cumulative: np.ndarray) -> None:
        lower = int(cumulative[symbol])
        upper = int(cumulative[symbol + 1])
        width = self.high - self.low + 1
        self.high = self.low + (width * upper // FREQUENCY_TOTAL) - 1
        self.low += width * lower // FREQUENCY_TOTAL
        while True:
            if self.high < _HALF:
                self._emit(0)
            elif self.low >= _HALF:
                self._emit(1)
                self.low -= _HALF
                self.high -= _HALF
            elif self.low >= _QUARTER and self.high < _THREE_QUARTERS:
                self.pending += 1
                self.low -= _QUARTER
                self.high -= _QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> bytes:
        self.pending += 1
        self._emit(0 if self.low < _QUARTER else 1)
        return self.writer.finish()


class _StaticRangeDecoder:
    def __init__(self, payload: bytes) -> None:
        if not payload:
            raise DDMVAE1PriorError("range stream is empty")
        self.reader = _BitReader(payload)
        self.low = 0
        self.high = _FULL_RANGE - 1
        self.code = 0
        for _ in range(_STATE_BITS):
            self.code = (self.code << 1) | self.reader.read()

    def decode(self, cumulative: np.ndarray) -> int:
        width = self.high - self.low + 1
        target = ((self.code - self.low + 1) * FREQUENCY_TOTAL - 1) // width
        symbol = int(np.searchsorted(cumulative, target, side="right") - 1)
        if symbol < 0 or symbol + 1 >= cumulative.size:
            raise DDMVAE1PriorError("range target escaped learned model")
        lower = int(cumulative[symbol])
        upper = int(cumulative[symbol + 1])
        self.high = self.low + (width * upper // FREQUENCY_TOTAL) - 1
        self.low += width * lower // FREQUENCY_TOTAL
        while True:
            if self.high < _HALF:
                pass
            elif self.low >= _HALF:
                self.low -= _HALF
                self.high -= _HALF
                self.code -= _HALF
            elif self.low >= _QUARTER and self.high < _THREE_QUARTERS:
                self.low -= _QUARTER
                self.high -= _QUARTER
                self.code -= _QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = (self.code << 1) | self.reader.read()
        return symbol


def _codes(value: np.ndarray, levels: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 4:
        raise DDMVAE1PriorError("token codes must be uint8 [P,H,W,C]")
    if not (2 <= levels <= 16):
        raise DDMVAE1PriorError("levels must be in [2,16]")
    if array.size == 0 or array.size > MAX_VALUES:
        raise DDMVAE1PriorError("token element count is outside the production bound")
    if any(int(dimension) <= 0 or int(dimension) > 65535 for dimension in array.shape):
        raise DDMVAE1PriorError("token dimensions are outside uint16 framing")
    if int(array.shape[-1]) > MAX_CHANNELS:
        raise DDMVAE1PriorError("token channel count exceeds the model bound")
    if np.any(array >= levels):
        raise DDMVAE1PriorError("token code exceeds levels")
    return np.ascontiguousarray(array)


def _raw_lzma_encode(payload: bytes) -> bytes:
    return lzma.compress(
        payload,
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}],
    )


def _raw_lzma_decode(payload: bytes, *, expected_length: int) -> bytes:
    try:
        decoder = lzma.LZMADecompressor(
            format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}],
        )
        restored = decoder.decompress(payload, max_length=expected_length + 1)
    except lzma.LZMAError as exc:
        raise DDMVAE1PriorError("invalid LZMA1 mode stream") from exc
    if len(restored) != expected_length or not decoder.eof or decoder.unused_data:
        raise DDMVAE1PriorError("LZMA1 mode stream length or termination differs")
    return restored


def _normalize_count_rows(counts: np.ndarray) -> np.ndarray:
    if counts.ndim != 4 or counts.shape[-1] < 2:
        raise DDMVAE1PriorError("learned count tensor shape differs")
    raw = counts.astype(np.int64) * 2 + 1
    row_totals = raw.sum(axis=-1, keepdims=True)
    scaled = np.maximum(1, raw * FREQUENCY_TOTAL // row_totals)
    remainders = raw * FREQUENCY_TOTAL % row_totals
    deltas = FREQUENCY_TOTAL - scaled.sum(axis=-1)
    flat_scaled = scaled.reshape(-1, scaled.shape[-1])
    flat_remainders = remainders.reshape(-1, remainders.shape[-1])
    for row_index, delta in enumerate(deltas.reshape(-1).tolist()):
        if delta > 0:
            order = np.argsort(-flat_remainders[row_index], kind="stable")
            flat_scaled[row_index, order[:delta]] += 1
        elif delta < 0:
            remaining = -delta
            order = np.argsort(flat_remainders[row_index], kind="stable")
            for symbol in order.tolist():
                reducible = int(flat_scaled[row_index, symbol] - 1)
                take = min(reducible, remaining)
                flat_scaled[row_index, symbol] -= take
                remaining -= take
                if remaining == 0:
                    break
            if remaining:
                raise DDMVAE1PriorError("learned frequency normalization underflowed")
    if np.any(scaled <= 0) or np.any(scaled > np.iinfo(np.uint16).max):
        raise DDMVAE1PriorError("learned frequency escaped uint16")
    if np.any(scaled.sum(axis=-1) != FREQUENCY_TOTAL):
        raise DDMVAE1PriorError("learned frequency row does not normalize")
    return np.ascontiguousarray(scaled.astype("<u2"))


def fit_counted_ar_prior(codes: np.ndarray, *, levels: int = 16) -> np.ndarray:
    """Fit the canonical counted first-order temporal categorical prior."""

    value = _codes(codes, levels)
    base, delta = factor_mode_delta(value, levels)
    pair_count, height, width, channels = delta.shape
    frame_values = height * width
    counts = np.zeros((channels, levels, levels, levels), dtype=np.uint32)
    for channel in range(channels):
        symbols = np.ascontiguousarray(delta[..., channel]).reshape(pair_count, frame_values)
        previous = np.zeros_like(symbols)
        previous[1:] = symbols[:-1]
        bases = np.broadcast_to(base[..., channel].reshape(1, frame_values), symbols.shape)
        np.add.at(
            counts[channel],
            (bases.reshape(-1), previous.reshape(-1), symbols.reshape(-1)),
            1,
        )
    return _normalize_count_rows(counts)


def _validate_frequencies(
    frequencies: np.ndarray,
    *,
    channels: int,
    levels: int,
) -> np.ndarray:
    value = np.asarray(frequencies)
    expected = (channels, levels, levels, levels)
    if value.dtype != np.dtype("<u2") or value.shape != expected:
        raise DDMVAE1PriorError("learned frequency tensor dtype/shape differs")
    if np.any(value == 0) or np.any(value.astype(np.uint32).sum(axis=-1) != FREQUENCY_TOTAL):
        raise DDMVAE1PriorError("learned frequency tensor is not normalized")
    return np.ascontiguousarray(value)


def _cumulative_rows(frequencies: np.ndarray) -> np.ndarray:
    zeros = np.zeros((*frequencies.shape[:-1], 1), dtype=np.uint32)
    return np.concatenate(
        [zeros, np.cumsum(frequencies, axis=-1, dtype=np.uint32)],
        axis=-1,
    )


def _encode_residual(
    base: np.ndarray,
    delta: np.ndarray,
    frequencies: np.ndarray,
) -> bytes:
    pair_count, height, width, channels = delta.shape
    frame_values = height * width
    cumulative = _cumulative_rows(frequencies)
    encoder = _StaticRangeEncoder()
    for channel in range(channels):
        source = np.ascontiguousarray(delta[..., channel]).reshape(pair_count, frame_values)
        base_channel = base[..., channel].reshape(frame_values)
        previous = np.zeros(frame_values, dtype=np.uint8)
        for pair_index in range(pair_count):
            current = source[pair_index]
            for cell, raw_symbol in enumerate(current):
                encoder.encode(
                    int(raw_symbol),
                    cumulative[channel, int(base_channel[cell]), int(previous[cell])],
                )
            previous = current
    return encoder.finish()


def _decode_residual(
    payload: bytes,
    base: np.ndarray,
    frequencies: np.ndarray,
    shape: tuple[int, int, int, int],
) -> np.ndarray:
    pair_count, height, width, channels = shape
    frame_values = height * width
    cumulative = _cumulative_rows(frequencies)
    decoder = _StaticRangeDecoder(payload)
    output = np.zeros(shape, dtype=np.uint8)
    for channel in range(channels):
        base_channel = base[..., channel].reshape(frame_values)
        previous = np.zeros(frame_values, dtype=np.uint8)
        for pair_index in range(pair_count):
            current = output[pair_index, ..., channel].reshape(frame_values)
            for cell in range(frame_values):
                current[cell] = decoder.decode(
                    cumulative[channel, int(base_channel[cell]), int(previous[cell])]
                )
            previous = current
    return np.ascontiguousarray(output)


def _semantic_digest(levels: int, shape: tuple[int, ...], raw: bytes) -> bytes:
    metadata = struct.pack("<BB4H", VERSION, levels, *shape)
    return hashlib.sha256(metadata + raw).digest()


def encode_counted_ar_prior(
    codes: np.ndarray,
    *,
    levels: int = 16,
    frequencies: np.ndarray | None = None,
) -> bytes:
    """Encode one canonical frame with all learned prior weights counted."""

    value = _codes(codes, levels)
    base, delta = factor_mode_delta(value, levels)
    fitted = (
        fit_counted_ar_prior(value, levels=levels)
        if frequencies is None
        else _validate_frequencies(frequencies, channels=value.shape[-1], levels=levels)
    )
    base_stream = _raw_lzma_encode(pack_nibbles(base))
    model_raw = fitted.tobytes(order="C")
    model_stream = zlib.compress(model_raw, level=9)
    residual_stream = _encode_residual(base, delta, fitted)
    header = HEADER.pack(
        MAGIC,
        VERSION,
        MODEL_KIND,
        levels,
        value.ndim,
        *value.shape,
        len(base_stream),
        len(model_stream),
        len(residual_stream),
        _semantic_digest(levels, tuple(value.shape), value.tobytes()),
    )
    return header + base_stream + model_stream + residual_stream


def _decode_counted_ar_prior(frame: bytes, *, canonical: bool) -> np.ndarray:
    if len(frame) < HEADER.size:
        raise DDMVAE1PriorError("learned-prior frame is truncated")
    (
        magic,
        version,
        model_kind,
        levels,
        rank,
        pair_count,
        height,
        width,
        channels,
        base_length,
        model_length,
        residual_length,
        digest,
    ) = HEADER.unpack_from(frame)
    if magic != MAGIC or version != VERSION or model_kind != MODEL_KIND or rank != 4:
        raise DDMVAE1PriorError("learned-prior frame magic/version/model/rank differs")
    shape = (pair_count, height, width, channels)
    count = math.prod(shape)
    if (
        not (2 <= levels <= 16)
        or count <= 0
        or count > MAX_VALUES
        or channels > MAX_CHANNELS
    ):
        raise DDMVAE1PriorError("learned-prior frame levels/shape differs")
    expected = HEADER.size + base_length + model_length + residual_length
    if min(base_length, model_length, residual_length) <= 0 or len(frame) != expected:
        raise DDMVAE1PriorError("learned-prior frame lengths do not close")
    base_end = HEADER.size + base_length
    model_end = base_end + model_length
    base_count = height * width * channels
    base = unpack_nibbles(
        _raw_lzma_decode(
            frame[HEADER.size:base_end],
            expected_length=(base_count + 1) // 2,
        ),
        base_count,
    ).reshape(height, width, channels)
    if np.any(base >= levels):
        raise DDMVAE1PriorError("decoded mode base exceeds levels")
    raw_model_bytes = channels * levels * levels * levels * np.dtype("<u2").itemsize
    if raw_model_bytes > MAX_MODEL_BYTES:
        raise DDMVAE1PriorError("counted prior model exceeds the allocation bound")
    try:
        decompressor = zlib.decompressobj()
        model_raw = decompressor.decompress(
            frame[base_end:model_end],
            raw_model_bytes + 1,
        )
    except zlib.error as exc:
        raise DDMVAE1PriorError("invalid counted prior model stream") from exc
    if (
        len(model_raw) != raw_model_bytes
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise DDMVAE1PriorError("counted prior model length or termination differs")
    frequencies = np.frombuffer(model_raw, dtype="<u2").reshape(
        channels,
        levels,
        levels,
        levels,
    )
    frequencies = _validate_frequencies(
        frequencies,
        channels=channels,
        levels=levels,
    )
    delta = _decode_residual(frame[model_end:], base, frequencies, shape)
    restored = reconstruct_mode_delta(base, delta, levels)
    if _semantic_digest(levels, shape, restored.tobytes()) != digest:
        raise DDMVAE1PriorError("decoded token semantic SHA-256 differs")
    if canonical and encode_counted_ar_prior(restored, levels=levels) != frame:
        raise DDMVAE1PriorError("learned-prior frame is noncanonical or has inert bytes")
    return restored


def decode_counted_ar_prior(frame: bytes) -> np.ndarray:
    """Decode and canonically re-encode one learned-prior frame."""

    try:
        return _decode_counted_ar_prior(bytes(frame), canonical=True)
    except (OverflowError, struct.error) as exc:
        raise DDMVAE1PriorError("learned-prior frame structure is invalid") from exc


def learned_prior_accounting(frame: bytes) -> LearnedPriorAccounting:
    if len(frame) < HEADER.size:
        raise DDMVAE1PriorError("learned-prior frame is truncated")
    unpacked = HEADER.unpack_from(frame)
    if unpacked[0] != MAGIC or unpacked[1] != VERSION or unpacked[2] != MODEL_KIND:
        raise DDMVAE1PriorError("learned-prior frame identity differs")
    levels = int(unpacked[3])
    shape = tuple(int(item) for item in unpacked[5:9])
    channels = shape[-1]
    return LearnedPriorAccounting(
        framed_bytes=len(frame),
        header_bytes=HEADER.size,
        base_bytes=int(unpacked[9]),
        model_bytes=int(unpacked[10]),
        residual_bytes=int(unpacked[11]),
        raw_model_bytes=channels * levels * levels * levels * np.dtype("<u2").itemsize,
        raw_token_bytes=math.prod(shape),
        sha256=hashlib.sha256(frame).hexdigest(),
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _formulation_config() -> dict[str, Any]:
    return {
        "base": {
            "codec": "raw_lzma1",
            "extreme": True,
            "packed_nibbles": "high_then_low",
            "preset": 9,
            "residual": "(token-stored_global_mode) mod levels",
            "stored_global_mode": "lowest-symbol temporal mode at each h,w,c",
        },
        "context": {
            "conditioning": [
                "channel",
                "stored_global_mode[h,w,c]",
                "delta[t-1,h,w,c]",
            ],
            "pooling": "all h,w cells sharing channel,mode,previous_delta",
            "t0_previous_delta": 0,
            "target": "delta[t,h,w,c]",
        },
        "frame": {
            "header": "<4sBBBB4HIII32s",
            "max_channels": MAX_CHANNELS,
            "max_model_bytes": MAX_MODEL_BYTES,
            "semantic_hash": "sha256(version,levels,shape,uint8_token_bytes)",
        },
        "model": {
            "codec": "zlib",
            "compression_level": 9,
            "frequency_dtype": "little-endian uint16",
            "frequency_total": FREQUENCY_TOTAL,
            "normalization": (
                "raw=2*count+1; proportional floor; min-one; "
                "stable largest-remainder add or stable reducible-mass remove"
            ),
            "order": ["channel", "mode", "previous_delta", "current_delta"],
            "smoothing": "2*count+1",
        },
        "range_coder": {
            "state_bits": _STATE_BITS,
            "traversal": ["channel", "pair", "row-major cell"],
        },
        "schema": f"{SCHEMA}.formulation",
    }


def _formulation_config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return _sha256(canonical)


def _file_custody(path: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            size += len(block)
            hasher.update(block)
    return {"bytes": size, "sha256": hasher.hexdigest()}


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    _atomic_bytes(path, payload)


def _repo_evidence_reference(
    path: Path,
    *,
    repo: Path,
    evidence_root: Path,
) -> str:
    resolved_repo = repo.resolve()
    resolved_evidence = evidence_root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_evidence):
        raise DDMVAE1PriorError("stage artifact escaped .omx/research")
    return resolved.relative_to(resolved_repo).as_posix()


def _resolve_repo_evidence_reference(
    value: str,
    *,
    repo: Path,
    evidence_root: Path,
) -> Path:
    reference = Path(value)
    if reference.is_absolute():
        raise DDMVAE1PriorError("stage artifact reference must be repo-relative")
    resolved = (repo.resolve() / reference).resolve()
    if not resolved.is_relative_to(evidence_root.resolve()):
        raise DDMVAE1PriorError("stage artifact reference escaped .omx/research")
    return resolved


def _resolve_preserved_stage(
    stage: dict[str, Any],
    *,
    repo: Path,
    evidence_root: Path,
    cold_store_roots: tuple[Path, ...] = COLD_STORE_ROOTS,
) -> Path:
    """Resolve a preserved stage locally or from a hash-checked SSD fallback."""
    primary = _resolve_repo_evidence_reference(
        str(stage.get("path", "")),
        repo=repo,
        evidence_root=evidence_root,
    )
    if primary.is_file():
        return primary
    cold_value = stage.get("cold_store_path")
    if not isinstance(cold_value, str) or not cold_value:
        raise DDMVAE1PriorError("preserved stage is missing and has no cold-store fallback")
    cold_reference = Path(cold_value)
    if not cold_reference.is_absolute():
        raise DDMVAE1PriorError("cold-store stage reference must be absolute")
    cold_resolved = cold_reference.resolve()
    allowed_roots = tuple(root.resolve() for root in cold_store_roots)
    if not any(cold_resolved.is_relative_to(root) for root in allowed_roots):
        raise DDMVAE1PriorError("cold-store stage reference escaped approved SSD roots")
    if not cold_resolved.is_file():
        raise DDMVAE1PriorError("cold-store stage artifact is missing")
    return cold_resolved


def _model_checkpoint_bytes(
    frequencies: np.ndarray,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    token_sha256: str,
    shape: tuple[int, ...],
    levels: int,
) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(
        output,
        frequencies=frequencies,
        checkpoint_path=np.array(str(checkpoint_path)),
        checkpoint_sha256=np.array(checkpoint_sha256),
        token_sha256=np.array(token_sha256),
        shape=np.asarray(shape, dtype=np.int64),
        levels=np.array(levels, dtype=np.int64),
    )
    return output.getvalue()


def _load_model_checkpoint(
    path: Path,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    token_sha256: str,
    shape: tuple[int, ...],
    levels: int,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as bundle:
        if (
            str(bundle["checkpoint_path"].item()) != str(checkpoint_path)
            or str(bundle["checkpoint_sha256"].item()) != checkpoint_sha256
            or str(bundle["token_sha256"].item()) != token_sha256
            or tuple(int(item) for item in bundle["shape"].tolist()) != shape
            or int(bundle["levels"].item()) != levels
        ):
            raise DDMVAE1PriorError("model checkpoint source contract differs")
        return _validate_frequencies(
            bundle["frequencies"],
            channels=shape[-1],
            levels=levels,
        )


def _source_hashes() -> dict[str, dict[str, Any]]:
    repo = Path(__file__).resolve().parents[1]
    return {
        name: _file_custody(path)
        for name, path in {
            "probe": Path(__file__).resolve(),
            "r7_coder": repo / "experiments/ddm_r7_token_coder.py",
            "tr1_runtime": Path(tr1.__file__).resolve(),
        }.items()
    }


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if result.returncode != 0 or len(value) != 40:
        raise DDMVAE1PriorError("git HEAD provenance is unavailable")
    return value


def _load_or_initialize_progress(
    path: Path,
    *,
    checkpoint_path: Path,
    checkpoint_custody: dict[str, Any],
    source_hashes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = {
        "checkpoint": {
            "path": str(checkpoint_path),
            **checkpoint_custody,
        },
        "schema": f"{SCHEMA}.progress",
        "source_files": source_hashes,
    }
    if path.exists():
        value = json.loads(path.read_text())
        if any(value.get(key) != item for key, item in expected.items()):
            raise DDMVAE1PriorError("resume progress contract differs")
        return value
    value = {
        **expected,
        "created_at_unix": time.time(),
        "stages": {},
    }
    _atomic_json(path, value)
    return value


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_cli()
    started = time.time()
    repo = Path(__file__).resolve().parents[1]
    evidence_root = (repo / ".omx/research").resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    resume_path = args.resume_from.resolve()
    for target in (output_path, resume_path):
        if not target.is_relative_to(evidence_root):
            raise DDMVAE1PriorError("output and resume paths must stay under .omx/research")
    if output_path == resume_path or checkpoint_path in {output_path, resume_path}:
        raise DDMVAE1PriorError("evidence paths alias")
    checkpoint_custody = _file_custody(checkpoint_path)
    source_hashes = _source_hashes()
    progress = _load_or_initialize_progress(
        resume_path,
        checkpoint_path=checkpoint_path,
        checkpoint_custody=checkpoint_custody,
        source_hashes=source_hashes,
    )

    compiled = tr1.compile_archive_from_checkpoint(checkpoint_path)
    parsed = tr1.parse_archive(compiled.archive_bytes).packet
    codes = np.ascontiguousarray(parsed.token_codes)
    levels = int(parsed.selector["token_quant_levels"])
    codes = _codes(codes, levels)
    token_sha256 = _sha256(codes.tobytes())
    shape = tuple(int(item) for item in codes.shape)

    prefix = resume_path.with_suffix("")
    model_path = prefix.with_name(
        f"{prefix.name}.stage-001-{checkpoint_custody['sha256'][:12]}.npz"
    )
    frame_path = prefix.with_name(
        f"{prefix.name}.stage-002-{checkpoint_custody['sha256'][:12]}.bin"
    )
    stages = progress.setdefault("stages", {})
    if not isinstance(stages, dict):
        raise DDMVAE1PriorError("resume stage ledger differs")

    if "fit_model" not in stages:
        frequencies = fit_counted_ar_prior(codes, levels=levels)
        _atomic_bytes(
            model_path,
            _model_checkpoint_bytes(
                frequencies,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=str(checkpoint_custody["sha256"]),
                token_sha256=token_sha256,
                shape=shape,
                levels=levels,
            ),
        )
        stages["fit_model"] = {
            "path": _repo_evidence_reference(
                model_path,
                repo=repo,
                evidence_root=evidence_root,
            ),
            **_file_custody(model_path),
        }
        _atomic_json(resume_path, progress)
    model_stage = stages["fit_model"]
    model_stage_path = _resolve_preserved_stage(
        model_stage,
        repo=repo,
        evidence_root=evidence_root,
    )
    if _file_custody(model_stage_path) != {
        "bytes": model_stage["bytes"],
        "sha256": model_stage["sha256"],
    }:
        raise DDMVAE1PriorError("preserved model checkpoint custody differs")
    frequencies = _load_model_checkpoint(
        model_stage_path,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=str(checkpoint_custody["sha256"]),
        token_sha256=token_sha256,
        shape=shape,
        levels=levels,
    )

    if "encode_frame" not in stages:
        frame = encode_counted_ar_prior(
            codes,
            levels=levels,
            frequencies=frequencies,
        )
        _atomic_bytes(frame_path, frame)
        stages["encode_frame"] = {
            "path": _repo_evidence_reference(
                frame_path,
                repo=repo,
                evidence_root=evidence_root,
            ),
            **_file_custody(frame_path),
        }
        _atomic_json(resume_path, progress)
    frame_stage = stages["encode_frame"]
    frame_stage_path = _resolve_preserved_stage(
        frame_stage,
        repo=repo,
        evidence_root=evidence_root,
    )
    if _file_custody(frame_stage_path) != {
        "bytes": frame_stage["bytes"],
        "sha256": frame_stage["sha256"],
    }:
        raise DDMVAE1PriorError("preserved frame checkpoint custody differs")
    frame = frame_stage_path.read_bytes()

    decode_started = time.monotonic()
    restored = decode_counted_ar_prior(frame)
    decode_seconds = time.monotonic() - decode_started
    if not np.array_equal(restored, codes):
        raise DDMVAE1PriorError("learned-prior parse-back differs")
    incumbent_started = time.monotonic()
    incumbent = encode_token_codes(codes, levels=levels, codec="smevr")
    incumbent_seconds = time.monotonic() - incumbent_started
    if not np.array_equal(decode_token_codes(incumbent), codes):
        raise DDMVAE1PriorError("SMEVR control parse-back differs")
    accounting = learned_prior_accounting(frame)
    measured_falsifier = accounting.framed_bytes >= len(incumbent)
    formulation_config = _formulation_config()
    formulation_config_sha256 = _formulation_config_sha256(formulation_config)
    resumability = {
        "model_checkpoint": stages["fit_model"],
        "progress": {
            "path": _repo_evidence_reference(
                resume_path,
                repo=repo,
                evidence_root=evidence_root,
            ),
            **_file_custody(resume_path),
        },
        "token_frame_checkpoint": stages["encode_frame"],
    }
    custody_reference = progress.get("custody_manifest_path")
    if custody_reference is not None:
        custody_path = _resolve_repo_evidence_reference(
            str(custody_reference),
            repo=repo,
            evidence_root=evidence_root,
        )
        resumability["artifact_custody"] = {
            "path": str(custody_reference),
            **_file_custody(custody_path),
        }
    receipt = {
        "authority": {
            "axis": "[macOS-CPU advisory, rate-only]",
            "pointer": POINTER_LOCAL,
            "promotion_eligible": False,
            "score_claim": False,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            **checkpoint_custody,
        },
        "input": {
            "levels": levels,
            "shape": list(shape),
            "token_sha256": token_sha256,
        },
        "provenance": {
            "command_argv": [
                "uv",
                "run",
                "--no-project",
                "--with",
                "numpy",
                "--with",
                "brotli",
                "python",
                "-m",
                "experiments.ddm_vae1_ar_prior_probe",
                "--checkpoint",
                str(checkpoint_path),
                "--output",
                _repo_evidence_reference(
                    output_path,
                    repo=repo,
                    evidence_root=evidence_root,
                ),
                "--resume-from",
                _repo_evidence_reference(
                    resume_path,
                    repo=repo,
                    evidence_root=evidence_root,
                ),
            ],
            "cwd": ".",
            "environment": {
                "PYTHONPATH": os.environ.get("PYTHONPATH"),
            },
            "git_head_at_measurement": _git_head(repo),
            "rng": "none",
            "seed": None,
        },
        "learned_ar_prior": {
            **asdict(accounting),
            "canonical_reencode_exact": True,
            "formulation_config": formulation_config,
            "formulation_config_sha256": formulation_config_sha256,
            "formulation_scope": FORMULATION_SCOPE,
            "model": (
                "static pooled p(delta[t,h,w,c] | channel c, "
                "stored_global_mode[h,w,c], delta[t-1,h,w,c]), with t0 previous=0; "
                "(2*count+1) smoothing, uint16 CDF total 2^15, "
                "channel-major 32-bit range coder, zlib9 model, "
                "LZMA1-preset9-extreme mode-base side stream"
            ),
            "model_weights_counted": True,
            "parseback_exact": True,
            "rule_118": (
                "all fitted uint16 frequencies are carried in the counted model section; "
                "free code contains only generic fit/encode/decode algorithms"
            ),
        },
        "race": {
            "delta_bytes_ar_minus_smevr": accounting.framed_bytes - len(incumbent),
            "disposition": (
                f"FALSIFIED_AT_{FORMULATION_SCOPE}_FORMULATION_SCOPE"
                if measured_falsifier
                else "ADOPT_AS_R7_SUCCESSOR_RACE_ROW"
            ),
            "falsifier": "coded_bytes + prior_bytes >= SMEVR same-object bytes",
            "falsifier_fired": measured_falsifier,
            "verdict_scope": (
                f"FORMULATION x {FORMULATION_SCOPE} x "
                f"config_sha256={formulation_config_sha256}"
            ),
            "prompt_incumbent_bytes": SMEVR_ENDPOINT_BYTES,
            "same_object_smevr_bytes": len(incumbent),
            "same_object_smevr_sha256": _sha256(incumbent),
        },
        "resumability": resumability,
        "runtime": {
            "decode_seconds_including_canonical_reencode": round(decode_seconds, 6),
            "incumbent_encode_seconds": round(incumbent_seconds, 6),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": sys.version,
            "wall_seconds": round(time.time() - started, 6),
            "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        },
        "schema": SCHEMA,
        "source_files": source_hashes,
    }
    _atomic_json(output_path, receipt)
    print(json.dumps(receipt["race"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
