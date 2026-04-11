from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
import time

import numpy as np

from .gpt_score import load_official_commavq_gpt_model
from .range_coder import RangeDecoder, RangeEncoder, cumulative_frequencies, normalize_probabilities

STREAM_MAGIC = b"GTA1"


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


def _read_bounded_uint16_token_prefix(
    token_path: str | Path,
    *,
    max_tokens: int,
) -> tuple[int, np.ndarray]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    source = Path(token_path)
    size_bytes = source.stat().st_size
    if size_bytes % 2 != 0:
        raise ValueError(f"token stream must contain an even number of bytes: {source}")
    raw_token_count = size_bytes // 2
    if raw_token_count <= 0:
        raise ValueError(f"token stream is empty: {source}")
    sample = np.fromfile(source, dtype=np.uint16, count=max_tokens)
    if sample.size < 2:
        raise ValueError("sample must contain at least two tokens")
    return raw_token_count, np.asarray(sample, dtype=np.uint16)


def _encode_tokens_from_logits_rows(
    tokens: np.ndarray,
    *,
    logits_rows,
    vocab_size: int,
    total_frequency: int,
) -> dict[str, object]:
    arr = _normalize_tokens(tokens)
    rows = np.asarray(logits_rows, dtype=np.float64)
    if rows.shape[0] < arr.size - 1:
        raise ValueError("logits_rows must contain one row per predicted token")
    encoder = RangeEncoder()
    total_nll_nats = 0.0
    for index in range(1, arr.size):
        target = int(arr[index])
        if target >= vocab_size:
            raise ValueError("token is outside vocab_size")
        frequencies = _frequencies_from_logits(rows[index - 1], vocab_size=vocab_size, total=total_frequency)
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


def encode_token_stream_with_logits_fn(
    tokens,
    *,
    logits_fn: Callable[[np.ndarray], np.ndarray],
    context_tokens: int,
    vocab_size: int,
    total_frequency: int = 1 << 15,
) -> dict[str, object]:
    arr = _normalize_tokens(tokens)
    encoded = encode_tokens_with_logits_fn(
        arr,
        logits_fn=logits_fn,
        context_tokens=context_tokens,
        vocab_size=vocab_size,
        total_frequency=total_frequency,
    )
    header = bytearray(STREAM_MAGIC)
    header.extend(len(arr).to_bytes(4, "big"))
    header.extend(int(arr[0]).to_bytes(2, "big"))
    payload = bytes(header) + encoded["encoded_bytes"]
    return {
        "encoded_bytes": payload,
        "token_count": int(arr.size),
        "first_token": int(arr[0]),
        "scored_tokens": encoded["scored_tokens"],
        "bits_per_token": encoded["bits_per_token"],
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


def decode_token_stream_with_logits_fn(
    encoded: bytes,
    *,
    logits_fn: Callable[[np.ndarray], np.ndarray],
    context_tokens: int,
    vocab_size: int,
    total_frequency: int = 1 << 15,
) -> list[int]:
    data = bytes(encoded)
    if not data.startswith(STREAM_MAGIC):
        raise ValueError("invalid GPT arithmetic stream header")
    token_count = int.from_bytes(data[4:8], "big")
    first_token = int.from_bytes(data[8:10], "big")
    payload = data[10:]
    return decode_tokens_with_logits_fn(
        payload,
        token_count=token_count,
        first_token=first_token,
        logits_fn=logits_fn,
        context_tokens=context_tokens,
        vocab_size=vocab_size,
        total_frequency=total_frequency,
    )


def encode_commavq_gpt_sample(
    *,
    token_path: str | Path,
    encoded_path: str | Path,
    profile: str = "gpt_arithmetic_small",
    max_tokens: int = 256,
    context_tokens: int | None = None,
    vocab_size: int = 1025,
    device: str = "mps",
    dtype: str = "auto",
    verify_decode: bool = False,
    cache_dir: str | Path | None = None,
    model_url: str | None = None,
    gpt_module_path: str | Path | None = None,
    model_loader: Callable[..., object] | None = None,
) -> dict[str, object]:
    from .gpt_score import _load_frame_major_segments_from_path

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    try:
        raw_token_count, segments = _load_frame_major_segments_from_path(token_path, max_scored_tokens=max_tokens)
    except ValueError as exc:
        if "GPT-scoreable segments" in str(exc):
            raise ValueError("no non-empty frame-major segments") from exc
        raise
    if not segments:
        raise ValueError("no non-empty frame-major segments")
    segment = segments[0]
    token_count = min(int(segment.size), max_tokens)
    sample = np.asarray(segment[:token_count], dtype=np.uint16)
    if sample.size < 2:
        raise ValueError("sample must contain at least two tokens")

    loader = model_loader or load_official_commavq_gpt_model
    model = loader(
        device=device,
        dtype=dtype,
        cache_dir=cache_dir,
        model_url=model_url,
        gpt_module_path=gpt_module_path,
    )
    effective_context = 2580 if context_tokens is None else context_tokens
    if hasattr(model, "token_logits"):
        logits_rows = model.token_logits(sample, context_tokens=effective_context)
        encoded = _encode_tokens_from_logits_rows(
            sample,
            logits_rows=logits_rows,
            vocab_size=vocab_size,
            total_frequency=1 << 15,
        )
        header = bytearray(STREAM_MAGIC)
        header.extend(len(sample).to_bytes(4, "big"))
        header.extend(int(sample[0]).to_bytes(2, "big"))
        encoded = {
            "encoded_bytes": bytes(header) + encoded["encoded_bytes"],
            "token_count": int(sample.size),
            "first_token": int(sample[0]),
            "scored_tokens": encoded["scored_tokens"],
            "bits_per_token": encoded["bits_per_token"],
        }
    else:
        encoded = encode_token_stream_with_logits_fn(
            sample,
            logits_fn=model.next_token_logits,
            context_tokens=effective_context,
            vocab_size=vocab_size,
        )
    target = Path(encoded_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded["encoded_bytes"])
    encoded_bytes = target.stat().st_size
    original_bytes = int(sample.size) * 2
    exact_match = None
    if verify_decode:
        restored = decode_token_stream_with_logits_fn(
            encoded["encoded_bytes"],
            logits_fn=model.next_token_logits,
            context_tokens=effective_context,
            vocab_size=vocab_size,
        )
        exact_match = restored == sample.tolist()
    return {
        "command": "lossless_gpt_arithmetic_sample",
        "token_path": str(Path(token_path)),
        "encoded_path": str(target),
        "profile": profile,
        "device": device,
        "dtype": dtype if dtype != "auto" else "float32",
        "context_tokens": effective_context,
        "token_count": int(sample.size),
        "raw_token_count": int(raw_token_count),
        "encoded_bytes": encoded_bytes,
        "original_bytes": original_bytes,
        "compression_ratio": original_bytes / encoded_bytes if encoded_bytes else 0.0,
        "bits_per_token": encoded["bits_per_token"],
        "exact_match": exact_match,
        "local_only": True,
        "measured": False,
    }


def encode_commavq_gpt_global_sample(
    *,
    token_path: str | Path,
    encoded_path: str | Path,
    profile: str = "gpt_arithmetic_small",
    max_tokens: int = 256,
    context_tokens: int | None = None,
    vocab_size: int = 1025,
    device: str = "mps",
    dtype: str = "auto",
    verify_decode: bool = False,
    cache_dir: str | Path | None = None,
    model_url: str | None = None,
    gpt_module_path: str | Path | None = None,
    model_loader: Callable[..., object] | None = None,
) -> dict[str, object]:
    raw_token_count, sample = _read_bounded_uint16_token_prefix(token_path, max_tokens=max_tokens)

    loader = model_loader or load_official_commavq_gpt_model
    model = loader(
        device=device,
        dtype=dtype,
        cache_dir=cache_dir,
        model_url=model_url,
        gpt_module_path=gpt_module_path,
    )
    effective_context = 2580 if context_tokens is None else context_tokens
    encoded = encode_token_stream_with_logits_fn(
        sample,
        logits_fn=model.next_token_logits,
        context_tokens=effective_context,
        vocab_size=vocab_size,
    )
    target = Path(encoded_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded["encoded_bytes"])
    encoded_bytes = target.stat().st_size
    original_bytes = int(sample.size) * 2
    exact_match = None
    if verify_decode:
        restored = decode_token_stream_with_logits_fn(
            encoded["encoded_bytes"],
            logits_fn=model.next_token_logits,
            context_tokens=effective_context,
            vocab_size=vocab_size,
        )
        exact_match = restored == sample.tolist()
    return {
        "command": "lossless_gpt_arithmetic_global_sample",
        "token_path": str(Path(token_path)),
        "encoded_path": str(target),
        "profile": profile,
        "device": device,
        "dtype": dtype if dtype != "auto" else "float32",
        "context_tokens": effective_context,
        "token_count": int(sample.size),
        "raw_token_count": int(raw_token_count),
        "encoded_bytes": encoded_bytes,
        "original_bytes": original_bytes,
        "compression_ratio": original_bytes / encoded_bytes if encoded_bytes else 0.0,
        "bits_per_token": encoded["bits_per_token"],
        "exact_match": exact_match,
        "local_only": True,
        "measured": False,
    }


def probe_commavq_gpt_arithmetic_devices(
    *,
    token_path: str | Path,
    output_path: str | Path | None = None,
    profile: str = "gpt_arithmetic_small",
    max_tokens: int = 256,
    context_tokens: int | None = None,
    devices: tuple[str, ...] = ("cpu", "mps"),
    dtype: str = "auto",
    verify_decode: bool = False,
    cache_dir: str | Path | None = None,
    model_url: str | None = None,
    gpt_module_path: str | Path | None = None,
    encode_fn: Callable[..., dict[str, object]] = encode_commavq_gpt_sample,
) -> dict[str, object]:
    if not devices:
        raise ValueError("devices must contain at least one backend")

    results: list[dict[str, object]] = []
    for device in devices:
        encoded_path = None if output_path is None else f"{output_path}.{device}"
        started = time.perf_counter()
        result = encode_fn(
            token_path=token_path,
            encoded_path=encoded_path,
            profile=profile,
            max_tokens=max_tokens,
            context_tokens=context_tokens,
            device=device,
            dtype=dtype,
            verify_decode=verify_decode,
            cache_dir=cache_dir,
            model_url=model_url,
            gpt_module_path=gpt_module_path,
        )
        elapsed = time.perf_counter() - started
        results.append(
            {
                "device": device,
                "seconds": elapsed,
                "encoded_bytes": int(result["encoded_bytes"]),
                "compression_ratio": float(result["compression_ratio"]),
                "bits_per_token": float(result["bits_per_token"]),
            }
        )

    fastest = max(results, key=lambda item: (1.0 / item["seconds"]) if item["seconds"] > 0 else float("inf"))
    best_ratio = max(results, key=lambda item: item["compression_ratio"])
    payload = {
        "command": "lossless_gpt_arithmetic_probe",
        "token_path": str(Path(token_path)),
        "output_path": str(Path(output_path)) if output_path is not None else None,
        "profile": profile,
        "max_tokens": max_tokens,
        "context_tokens": 2580 if context_tokens is None else context_tokens,
        "fastest_device": fastest["device"],
        "best_ratio_device": best_ratio["device"],
        "results": results,
        "local_only": True,
        "measured": False,
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(__import__("json").dumps(payload, indent=2) + "\n")
    return payload
