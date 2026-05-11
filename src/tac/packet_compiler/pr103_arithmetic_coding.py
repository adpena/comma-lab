"""PR103 arithmetic-coding — reusable byte-level entropy-coder primitives.

This module extracts the REUSABLE arithmetic-coding pieces from the PR103
public submission (``hnerv_lc_ac/inflate.py``) into typed primitives:

1. **Merged range stream over multiple weight tensors**
   (:func:`encode_merged_range_stream` / :func:`decode_merged_range_stream`)

   PR103 encodes the 8 largest weight tensors of its HNeRV decoder into a
   *single* range-coded byte string. Storing one stream instead of eight
   eliminates the per-stream tail/start overhead (≈ 8–16 bytes per stream
   under constriction) and allows the coder to amortise small-symbol regions
   across tensor boundaries. The per-tensor categorical distributions are
   provided alongside the stream; the merge order is part of the contract.

2. **Latent-hi arithmetic coding**
   (:func:`encode_latent_hi_arithmetic` /
   :func:`decode_latent_hi_arithmetic`)

   PR103's latent quantisation uses a 16-bit zigzag delta split into ``lo``
   (uint8, Brotli) and ``hi`` (uint8, arithmetic-coded). The ``hi`` byte
   distribution has a sharp peak at 0 so arithmetic coding beats LZMA by
   ≈ 8 KB on a 600×28 latent stream. We surface that split as a typed
   primitive so any future per-pair latent stream (HNeRV/MNeRV/NeRV/SIREN,
   etc.) can opt into the same trick.

3. **Adaptive Brotli parameter search**
   (:func:`adaptive_brotli_param_search`)

   PR103 sweeps Brotli ``lgwin`` ∈ {10..24} × ``quality`` ∈ {0..11} on a
   per-archive basis and keeps the smallest output. We surface that sweep
   with a budget (time + count) so other packet compilers can re-use it.
   The result includes the chosen ``(lgwin, quality)`` so downstream tools
   can record it in the archive's build manifest.

Constriction is the underlying range-coder dependency (already a project
dep; see ``pyproject.toml``). The wire format is constriction's uint32
range-coded stream; we emit/parse it byte-faithfully so future native ports
can target the same byte layout.

Source: ``experiments/results/public_pr_archive_kaggle_mirror/public_pr103_intake_20260505_auto/source/submissions/hnerv_lc_ac/inflate.py``.
Schema analysis: ``experiments/results/hnerv_pr103_lc_ac_schema_refresh_20260510_codex/manifest.md``.

CLAUDE.md compliance:

* No scorer load — pure numpy / brotli / constriction / stdlib.
* No MPS / torch import.
* No ``/tmp`` paths.
* Frozen dataclasses; ``encode → decode`` is covered by focused Python
  conformance tests. Native ports must add golden vectors before promotion.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import brotli
import constriction
import numpy as np

# ── Public dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class WeightTensorACSpec:
    """Per-tensor specification for the merged range stream.

    Parameters
    ----------
    name:
        Short identifier (e.g. ``"blocks.0.weight"``). Used for diagnostics
        and golden-vector indexing only — never encoded into the bytes.
    shape:
        Expected tensor shape. The encoder flattens to a 1D symbol stream;
        the decoder reshapes back.
    histogram:
        Categorical distribution of length ``alphabet_size`` (typically 256
        for int8 tensors offset to ``[0, 256)``). Float-valued; the coder
        normalises internally. PR103 hardcodes ``alphabet_size=256`` for the
        int8 → uint8 offset-128 representation.
    alphabet_size:
        Number of symbols. Default 256 (PR103 convention: int8 weights
        offset by +128 so symbols ∈ ``[0, 256)``).

    Notes
    -----
    The histogram is part of the *out-of-band* schema: PR103 ships the eight
    256-byte histograms in a separate Brotli stream because the count is
    fixed in the inflate runtime. The encoder/decoder pair here assumes the
    caller has agreed on order and histograms; this primitive only handles
    the range-coded payload.
    """

    name: str
    shape: tuple[int, ...]
    histogram: np.ndarray
    alphabet_size: int = 256

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.shape or any(d <= 0 for d in self.shape):
            raise ValueError(f"shape must have all-positive dims; got {self.shape}")
        if self.alphabet_size <= 1:
            raise ValueError(f"alphabet_size must be >= 2; got {self.alphabet_size}")
        if self.histogram.shape != (self.alphabet_size,):
            raise ValueError(
                f"histogram must have shape ({self.alphabet_size},); "
                f"got {self.histogram.shape}"
            )


@dataclass(frozen=True)
class MergedRangeStream:
    """Range-coded byte payload over a sequence of weight tensors.

    Attributes
    ----------
    payload:
        Range-coded bytes (constriction's uint32 stream serialised as
        big-endian bytes).
    tensor_symbol_counts:
        Per-tensor symbol counts in encode order. Required by the decoder
        because the range-coded stream itself does not carry boundaries.
    word_count:
        Length of the underlying uint32 array. ``len(payload) == word_count*4``.
    """

    payload: bytes
    tensor_symbol_counts: tuple[int, ...]
    word_count: int


@dataclass(frozen=True)
class AdaptiveBrotliResult:
    """Best-found Brotli parameter set + payload.

    Attributes
    ----------
    payload:
        Brotli-compressed bytes corresponding to ``(lgwin, quality)``.
    lgwin, quality:
        Chosen parameters.
    tested_count:
        Number of (lgwin, quality) tuples actually evaluated. Useful for
        adaptive budgeting.
    elapsed_seconds:
        Wall-clock spent in the search. Capped by ``time_budget_s``.
    """

    payload: bytes
    lgwin: int
    quality: int
    tested_count: int
    elapsed_seconds: float
    explored: tuple[tuple[int, int, int], ...] = field(default_factory=tuple)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_categorical(weights: np.ndarray) -> constriction.stream.model.Categorical:
    """Build a constriction Categorical distribution from raw weights.

    Mirrors PR103's ``make_categorical``: floor at 1e-10, re-normalise to
    sum to 1, use ``perfect=False`` (lossy normalisation; standard for
    integer arithmetic coders).
    """
    p = np.asarray(weights, dtype=np.float64)
    p = np.maximum(p, 1e-10)
    p /= p.sum()
    return constriction.stream.model.Categorical(p, perfect=False)


def _uint32_bytes_to_words(payload: bytes) -> np.ndarray:
    """Reconstruct the constriction uint32 word array from bytes.

    The wire format is big-endian (constriction's serialisation is
    target-arch native; we normalise to big-endian here so the wire
    format is portable across architectures and matches the documented
    "uint32 stream" in the schema).
    """
    if len(payload) % 4 != 0:
        raise ValueError(f"payload size {len(payload)} is not a multiple of 4")
    return np.frombuffer(payload, dtype=">u4").astype(np.uint32)


def _words_to_uint32_bytes(words: np.ndarray) -> bytes:
    """Serialise a uint32 word array to big-endian bytes."""
    return np.asarray(words, dtype=">u4").tobytes()


# ── Merged range stream over weight tensors ─────────────────────────────────


def encode_merged_range_stream(
    tensors: Sequence[np.ndarray],
    specs: Sequence[WeightTensorACSpec],
) -> MergedRangeStream:
    """Encode multiple weight tensors into a single range-coded byte string.

    The tensors are encoded in order; each symbol is decoded against the
    corresponding ``WeightTensorACSpec.histogram`` Categorical. Storing one
    merged stream saves the per-stream tail overhead.

    Parameters
    ----------
    tensors:
        One ``np.ndarray`` per tensor, each already quantised to ``[0,
        alphabet_size)``. The encoder flattens to 1D in C-order.
    specs:
        One :class:`WeightTensorACSpec` per tensor (parallel to ``tensors``).

    Returns
    -------
    MergedRangeStream
        Container with the range-coded bytes plus per-tensor symbol counts.
    """
    if len(tensors) != len(specs):
        raise ValueError(
            f"tensor count {len(tensors)} != spec count {len(specs)}"
        )
    if not tensors:
        raise ValueError("must encode at least one tensor")

    encoder = constriction.stream.queue.RangeEncoder()
    counts: list[int] = []
    for t, s in zip(tensors, specs, strict=False):
        if t.shape != s.shape:
            raise ValueError(
                f"tensor {s.name!r} shape {t.shape} != spec shape {s.shape}"
            )
        flat = np.asarray(t, dtype=np.int32).reshape(-1)
        if flat.size == 0:
            raise ValueError(f"tensor {s.name!r} is empty")
        if flat.min() < 0 or flat.max() >= s.alphabet_size:
            raise ValueError(
                f"tensor {s.name!r} symbols out of range [0, {s.alphabet_size}); "
                f"min={int(flat.min())} max={int(flat.max())}"
            )
        cat = _make_categorical(s.histogram)
        # Encode one tensor at a time. constriction's ``encode`` accepts a
        # numpy array directly.
        encoder.encode(flat, cat)
        counts.append(int(flat.size))

    words = encoder.get_compressed()
    word_count = int(words.size)
    payload = _words_to_uint32_bytes(words)
    return MergedRangeStream(
        payload=payload,
        tensor_symbol_counts=tuple(counts),
        word_count=word_count,
    )


def decode_merged_range_stream(
    stream: MergedRangeStream,
    specs: Sequence[WeightTensorACSpec],
) -> list[np.ndarray]:
    """Inverse of :func:`encode_merged_range_stream`.

    Returns a list of ``np.ndarray`` in the same order as ``specs``, each
    reshaped to the corresponding ``spec.shape`` and dtype ``int32`` (the
    caller decides what offset / cast to apply).
    """
    if len(stream.tensor_symbol_counts) != len(specs):
        raise ValueError(
            f"stream has {len(stream.tensor_symbol_counts)} tensor entries; "
            f"specs has {len(specs)}"
        )
    words = _uint32_bytes_to_words(stream.payload)
    if words.size != stream.word_count:
        raise ValueError(
            f"payload word count mismatch: bytes imply {words.size} words; "
            f"stream declared {stream.word_count}"
        )
    decoder = constriction.stream.queue.RangeDecoder(words)
    out: list[np.ndarray] = []
    for s, count in zip(specs, stream.tensor_symbol_counts, strict=False):
        cat = _make_categorical(s.histogram)
        symbols = np.zeros(count, dtype=np.int32)
        for i in range(count):
            symbols[i] = decoder.decode(cat)
        out.append(symbols.reshape(s.shape))
    return out


# ── Latent-hi arithmetic coding ─────────────────────────────────────────────


def encode_latent_hi_arithmetic(
    latents: np.ndarray,
    *,
    histogram: np.ndarray,
) -> bytes:
    """Encode the high-byte of a uint16 latent stream via arithmetic coding.

    The expected input is the **uint16 zigzag-encoded delta** of a per-pair
    quantised latent (PR103 layout). The high byte typically has a sharp
    peak at 0; arithmetic coding beats Brotli/LZMA by 8–10 KB on the
    contest's 600×28 latent stream.

    Parameters
    ----------
    latents:
        Flat ``uint16`` array (already zigzag-encoded deltas). The high
        byte is ``(latents >> 8) & 0xFF``.
    histogram:
        256-element categorical distribution over the high byte. Caller
        produces this from the empirical hi-byte histogram.

    Returns
    -------
    bytes
        Range-coded payload (constriction uint32 stream, big-endian bytes).
    """
    latents = np.asarray(latents, dtype=np.uint16)
    if latents.ndim != 1 or latents.size == 0:
        raise ValueError(
            f"latents must be a non-empty 1D uint16 array; got shape {latents.shape}"
        )
    hi = ((latents.astype(np.int32) >> 8) & 0xFF).astype(np.int32)
    cat = _make_categorical(histogram)
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(hi, cat)
    return _words_to_uint32_bytes(encoder.get_compressed())


def decode_latent_hi_arithmetic(
    payload: bytes,
    *,
    histogram: np.ndarray,
    n_symbols: int,
) -> np.ndarray:
    """Inverse of :func:`encode_latent_hi_arithmetic`.

    Returns
    -------
    np.ndarray
        ``uint8`` array of length ``n_symbols`` containing the decoded high
        bytes. The caller is responsible for combining with the low bytes
        to reconstruct the original ``uint16`` zigzag deltas.
    """
    if n_symbols <= 0:
        raise ValueError(f"n_symbols must be > 0; got {n_symbols}")
    words = _uint32_bytes_to_words(payload)
    decoder = constriction.stream.queue.RangeDecoder(words)
    cat = _make_categorical(histogram)
    hi = np.zeros(n_symbols, dtype=np.int32)
    for i in range(n_symbols):
        hi[i] = decoder.decode(cat)
    return hi.astype(np.uint8)


# ── Adaptive Brotli parameter search ────────────────────────────────────────


def adaptive_brotli_param_search(
    payload: bytes,
    *,
    time_budget_s: float = 30.0,
    lgwin_range: tuple[int, int] = (10, 24),
    quality_range: tuple[int, int] = (0, 11),
    max_evaluations: int | None = None,
) -> AdaptiveBrotliResult:
    """Sweep Brotli ``(lgwin, quality)`` and return the smallest output.

    Parameters
    ----------
    payload:
        Source bytes to compress.
    time_budget_s:
        Soft wall-clock budget. Once exceeded, the search returns the best
        result found so far. The first parameter set is always evaluated so
        the result is always non-None.
    lgwin_range:
        Inclusive ``(min_lgwin, max_lgwin)``. Brotli accepts 10..24.
    quality_range:
        Inclusive ``(min_q, max_q)``. Brotli accepts 0..11.
    max_evaluations:
        Optional hard cap on combinations. ``None`` means "let the time
        budget decide".

    Returns
    -------
    AdaptiveBrotliResult
        Best payload + parameters + diagnostics. ``elapsed_seconds`` may
        slightly exceed ``time_budget_s`` because the budget is checked
        between evaluations.
    """
    if not payload:
        raise ValueError("payload must be non-empty")
    if time_budget_s <= 0:
        raise ValueError(f"time_budget_s must be > 0; got {time_budget_s}")
    lo_w, hi_w = lgwin_range
    lo_q, hi_q = quality_range
    if not (10 <= lo_w <= hi_w <= 24):
        raise ValueError(f"lgwin_range {lgwin_range} out of [10, 24]")
    if not (0 <= lo_q <= hi_q <= 11):
        raise ValueError(f"quality_range {quality_range} out of [0, 11]")

    # Enumerate from highest quality down so we tend to converge fast.
    candidates = [
        (w, q)
        for q in range(hi_q, lo_q - 1, -1)
        for w in range(hi_w, lo_w - 1, -1)
    ]
    start = time.monotonic()
    best_payload: bytes | None = None
    best_size = -1
    best_w = lo_w
    best_q = lo_q
    tested = 0
    explored: list[tuple[int, int, int]] = []
    for w, q in candidates:
        if max_evaluations is not None and tested >= max_evaluations:
            break
        # Permit at least one evaluation regardless of budget.
        elapsed = time.monotonic() - start
        if tested > 0 and elapsed >= time_budget_s:
            break
        comp = brotli.compress(
            payload, mode=brotli.MODE_GENERIC, quality=q, lgwin=w
        )
        explored.append((w, q, len(comp)))
        tested += 1
        if best_payload is None or len(comp) < best_size:
            best_payload = comp
            best_size = len(comp)
            best_w = w
            best_q = q
    elapsed_final = time.monotonic() - start
    assert best_payload is not None  # invariant: candidates is non-empty
    return AdaptiveBrotliResult(
        payload=best_payload,
        lgwin=best_w,
        quality=best_q,
        tested_count=tested,
        elapsed_seconds=elapsed_final,
        explored=tuple(explored),
    )


__all__ = [
    "AdaptiveBrotliResult",
    "MergedRangeStream",
    "WeightTensorACSpec",
    "adaptive_brotli_param_search",
    "decode_latent_hi_arithmetic",
    "decode_merged_range_stream",
    "encode_latent_hi_arithmetic",
    "encode_merged_range_stream",
]
