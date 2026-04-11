from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .gpt_score import _load_frame_major_segments_from_path, load_official_commavq_gpt_model
from .profiles import load_gpt_next_frame_profile
from .range_coder import RangeDecoder, RangeEncoder, cumulative_frequencies, normalize_probabilities

STREAM_MAGIC = b"NFG1"


def _normalize_frames(frames) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.uint16)
    if arr.ndim != 3 or arr.shape[1:] != (8, 16):
        raise ValueError("frames must have shape (N, 8, 16)")
    if arr.shape[0] < 2:
        raise ValueError("frames must contain at least two frames")
    return arr


def _freqs_from_frame_logits(logits, *, vocab_size: int, total: int) -> list[list[int]]:
    rows = np.asarray(logits, dtype=np.float64)
    if rows.shape[0] != 128 or rows.shape[1] < vocab_size:
        raise ValueError("next_frame_logits must return shape (128, vocab_size)")
    freqs = []
    for row in rows:
        clipped = row[:vocab_size]
        shifted = clipped - np.max(clipped)
        probs = np.exp(shifted)
        freqs.append(normalize_probabilities(probs.tolist(), total=total))
    return freqs


def encode_next_frame_stream_with_logits_fn(
    frames,
    *,
    logits_fn: Callable[[np.ndarray], np.ndarray],
    vocab_size: int,
    context_frames: int = 1,
    total_frequency: int = 1 << 15,
) -> dict[str, object]:
    arr = _normalize_frames(frames)
    encoder = RangeEncoder()
    for index in range(1, arr.shape[0]):
        prefix = arr[max(0, index - context_frames) : index]
        freqs_rows = _freqs_from_frame_logits(logits_fn(prefix), vocab_size=vocab_size, total=total_frequency)
        target = arr[index].reshape(-1)
        for symbol, freqs in zip(target.tolist(), freqs_rows):
            cumulative, total = cumulative_frequencies(freqs)
            encoder.encode(symbol=int(symbol), cumulative=cumulative, total=total)
    header = bytearray(STREAM_MAGIC)
    header.extend(arr.shape[0].to_bytes(4, "big"))
    header.extend(arr[0].tobytes())
    return {
        "encoded_bytes": bytes(header) + encoder.finish(),
        "frame_count": int(arr.shape[0]),
    }


def decode_next_frame_stream_with_logits_fn(
    encoded: bytes,
    *,
    logits_fn: Callable[[np.ndarray], np.ndarray],
    vocab_size: int,
    context_frames: int = 1,
    total_frequency: int = 1 << 15,
) -> np.ndarray:
    data = bytes(encoded)
    if not data.startswith(STREAM_MAGIC):
        raise ValueError("invalid next-frame stream header")
    frame_count = int.from_bytes(data[4:8], "big")
    first_frame = np.frombuffer(data[8 : 8 + 256], dtype=np.uint16).reshape(8, 16)
    decoder = RangeDecoder(data[8 + 256 :])
    restored = [first_frame]
    for _ in range(1, frame_count):
        prefix = np.asarray(restored[max(0, len(restored) - context_frames) :], dtype=np.uint16)
        freqs_rows = _freqs_from_frame_logits(logits_fn(prefix), vocab_size=vocab_size, total=total_frequency)
        frame = np.empty((128,), dtype=np.uint16)
        for idx, freqs in enumerate(freqs_rows):
            cumulative, total = cumulative_frequencies(freqs)
            target = decoder.target(total)
            symbol = max(i for i in range(len(freqs)) if cumulative[i] <= target)
            decoder.update(low_count=cumulative[symbol], high_count=cumulative[symbol + 1], total=total)
            frame[idx] = symbol
        restored.append(frame.reshape(8, 16))
    return np.asarray(restored, dtype=np.uint16)


def encode_commavq_next_frame_sample(
    *,
    token_path: str | Path,
    encoded_path: str | Path,
    profile: str = "gpt_next_frame_small",
    max_frames: int = 32,
    context_frames: int | None = None,
    vocab_size: int = 1024,
    device: str = "mps",
    dtype: str = "auto",
    verify_decode: bool = False,
    cache_dir: str | Path | None = None,
    model_url: str | None = None,
    gpt_module_path: str | Path | None = None,
    model_loader: Callable[..., object] | None = None,
) -> dict[str, object]:
    config = load_gpt_next_frame_profile(profile)
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    effective_context_frames = config.context_frames if context_frames is None else int(context_frames)
    if effective_context_frames <= 0:
        raise ValueError("context_frames must be positive")
    raw_token_count, segments = _load_frame_major_segments_from_path(token_path, max_scored_tokens=max_frames * 129)
    if not segments:
        raise ValueError("no non-empty frame-major segments")
    segment = segments[0]
    body = segment[:-1] if int(segment[-1]) == 1025 else segment
    if body.size % 129 != 0:
        raise ValueError("frame-major stream body must be divisible by 129")
    frames = body.reshape(-1, 129)[:max_frames, 1:].reshape(-1, 8, 16).astype(np.uint16)
    if frames.shape[0] < 2:
        raise ValueError("sample must contain at least two frames")

    loader = model_loader or load_official_commavq_gpt_model
    model = loader(
        device=device,
        dtype=dtype,
        cache_dir=cache_dir,
        model_url=model_url,
        gpt_module_path=gpt_module_path,
    )
    encoded = encode_next_frame_stream_with_logits_fn(
        frames,
        logits_fn=lambda prefix: model.next_frame_logits(prefix, context_frames=effective_context_frames),
        vocab_size=vocab_size,
        context_frames=effective_context_frames,
    )
    target = Path(encoded_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded["encoded_bytes"])
    exact_match = None
    if verify_decode:
        restored = decode_next_frame_stream_with_logits_fn(
            encoded["encoded_bytes"],
            logits_fn=lambda prefix: model.next_frame_logits(prefix, context_frames=effective_context_frames),
            vocab_size=vocab_size,
            context_frames=effective_context_frames,
        )
        exact_match = np.array_equal(restored, frames)
    original_bytes = int(frames.size) * 2
    return {
        "command": "lossless_next_frame_sample",
        "token_path": str(Path(token_path)),
        "encoded_path": str(target),
        "profile": profile,
        "device": device,
        "dtype": dtype if dtype != "auto" else "float32",
        "context_frames": effective_context_frames,
        "frame_count": int(frames.shape[0]),
        "raw_token_count": int(raw_token_count),
        "encoded_bytes": target.stat().st_size,
        "original_bytes": original_bytes,
        "compression_ratio": original_bytes / target.stat().st_size if target.stat().st_size else 0.0,
        "exact_match": exact_match,
        "local_only": True,
        "measured": False,
    }
