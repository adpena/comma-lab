from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from .range_coder import RangeDecoder, RangeEncoder, cumulative_frequencies, normalize_probabilities


def _normalize_tokens(tokens) -> np.ndarray:
    arr = np.asarray(tokens, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise ValueError("tokens must be non-empty")
    if np.any(arr < 0):
        raise ValueError("tokens must be non-negative")
    return arr


def _frequencies_from_logits(logits, *, vocab_size: int, total: int) -> list[int]:
    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    if arr.size < vocab_size:
        raise ValueError("logits_fn returned fewer logits than vocab_size")
    clipped = arr[:vocab_size]
    shifted = clipped - np.max(clipped)
    probs = np.exp(shifted)
    return normalize_probabilities(probs.tolist(), total=total)


def encode_tokens_with_logits_fn(
    tokens,
    *,
    logits_fn: Callable[[np.ndarray], np.ndarray],
    context_tokens: int,
    vocab_size: int,
    total_frequency: int = 1 << 15,
) -> dict[str, object]:
    arr = _normalize_tokens(tokens)
    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than 1")

    encoder = RangeEncoder()
    total_nll_nats = 0.0
    for index in range(1, arr.size):
        start = max(0, index - context_tokens)
        context = arr[start:index]
        target = int(arr[index])
        if target >= vocab_size:
            raise ValueError("token is outside vocab_size")
        logits = logits_fn(context)
        frequencies = _frequencies_from_logits(logits, vocab_size=vocab_size, total=total_frequency)
        cumulative, total = cumulative_frequencies(frequencies)
        encoder.encode(symbol=target, cumulative=cumulative, total=total)
        total_nll_nats += -math.log(frequencies[target] / total_frequency)

    encoded_bytes = encoder.finish()
    scored_tokens = int(arr.size - 1)
    return {
        "encoded_bytes": encoded_bytes,
        "scored_tokens": scored_tokens,
        "bits_per_token": (total_nll_nats / scored_tokens) / math.log(2.0) if scored_tokens else 0.0,
    }


def decode_tokens_with_logits_fn(
    encoded: bytes,
    *,
    token_count: int,
    first_token: int,
    logits_fn: Callable[[np.ndarray], np.ndarray],
    context_tokens: int,
    vocab_size: int,
    total_frequency: int = 1 << 15,
) -> list[int]:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if token_count > 1 and len(encoded) < 2:
        raise ValueError("truncated stream: need at least 2 bytes to decode multiple tokens")
    restored = [int(first_token)]
    decoder = RangeDecoder(encoded)
    for _ in range(1, token_count):
        start = max(0, len(restored) - context_tokens)
        context = np.asarray(restored[start:], dtype=np.int64)
        frequencies = _frequencies_from_logits(logits_fn(context), vocab_size=vocab_size, total=total_frequency)
        cumulative, total = cumulative_frequencies(frequencies)
        target = decoder.target(total)
        symbol = max(index for index in range(len(frequencies)) if cumulative[index] <= target)
        decoder.update(low_count=cumulative[symbol], high_count=cumulative[symbol + 1], total=total)
        restored.append(symbol)
    return restored
