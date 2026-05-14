# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .gpt_score import (
    _load_frame_major_segments_from_path,
    gpt_model_runtime_metadata,
    load_official_commavq_gpt_model,
)
from .profiles import load_next_frame_predictor_profile
from .range_coder import RangeDecoder, RangeEncoder, cumulative_frequencies, normalize_probabilities
from .tiny_frame_predictor import load_tiny_frame_predictor_runtime

STREAM_MAGIC = b"NFG1"


class PositionTransitionModel:
    def __init__(self, logits_table: np.ndarray) -> None:
        self._logits_table = np.asarray(logits_table, dtype=np.float64)

    def next_frame_logits(self, prefix_frames: np.ndarray, *, context_frames: int) -> np.ndarray:
        last_frame = np.asarray(prefix_frames[-1], dtype=np.uint16).reshape(-1)
        return self._logits_table[np.arange(128), last_frame]


class PositionPairTransitionModel:
    def __init__(
        self,
        *,
        pair_rows: dict[tuple[int, int, int], np.ndarray],
        fallback_rows: dict[tuple[int, int], np.ndarray],
        global_pair_rows: dict[tuple[int, int], np.ndarray],
        global_fallback_rows: dict[int, np.ndarray],
        vocab_size: int,
    ) -> None:
        self._pair_rows = {key: np.asarray(row, dtype=np.float64) for key, row in pair_rows.items()}
        self._fallback_rows = {key: np.asarray(row, dtype=np.float64) for key, row in fallback_rows.items()}
        self._global_pair_rows = {
            key: np.asarray(row, dtype=np.float64) for key, row in global_pair_rows.items()
        }
        self._global_fallback_rows = {
            key: np.asarray(row, dtype=np.float64) for key, row in global_fallback_rows.items()
        }
        self._uniform_row = np.ones((int(vocab_size),), dtype=np.float64)

    def _lookup(self, table: dict[object, np.ndarray], key: object) -> np.ndarray:
        return table.get(key, self._uniform_row)

    def _mix_rows(self, *components: tuple[float, np.ndarray]) -> np.ndarray:
        mixed = np.zeros_like(self._uniform_row)
        total_weight = 0.0
        for weight, row in components:
            if weight <= 0.0:
                continue
            mixed += weight * (row / row.sum())
            total_weight += weight
        if total_weight <= 0.0:
            return np.log(self._uniform_row / self._uniform_row.sum())
        return np.log(mixed / total_weight)

    def next_frame_logits(self, prefix_frames: np.ndarray, *, context_frames: int) -> np.ndarray:
        prefix = np.asarray(prefix_frames, dtype=np.uint16)
        prev1 = prefix[-1].reshape(-1)
        rows: list[np.ndarray] = []
        if prefix.shape[0] < 2:
            for position, previous in enumerate(prev1.tolist()):
                rows.append(
                    self._mix_rows(
                        (1.0, self._lookup(self._fallback_rows, (position, previous))),
                        (0.5, self._lookup(self._global_fallback_rows, previous)),
                    )
                )
            return np.asarray(rows, dtype=np.float64)
        prev2 = prefix[-2].reshape(-1)
        for position, (older, previous) in enumerate(zip(prev2.tolist(), prev1.tolist())):
            rows.append(
                self._mix_rows(
                    (1.0, self._lookup(self._pair_rows, (position, older, previous))),
                    (0.5, self._lookup(self._fallback_rows, (position, previous))),
                    (1.0, self._lookup(self._global_pair_rows, (older, previous))),
                    (0.5, self._lookup(self._global_fallback_rows, previous)),
                )
            )
        return np.asarray(rows, dtype=np.float64)


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


def build_position_transition_model(frames, *, vocab_size: int = 1024) -> PositionTransitionModel:
    arr = _normalize_frames(frames)
    counts = np.ones((128, vocab_size, vocab_size), dtype=np.float64)
    for previous, current in zip(arr[:-1], arr[1:]):
        prev_flat = previous.reshape(-1)
        curr_flat = current.reshape(-1)
        for position, (p, c) in enumerate(zip(prev_flat.tolist(), curr_flat.tolist())):
            if p < vocab_size and c < vocab_size:
                counts[position, p, c] += 1.0
    logits = np.log(counts)
    return PositionTransitionModel(logits)


def build_position_pair_transition_model(frames, *, vocab_size: int = 1024) -> PositionPairTransitionModel:
    arr = _normalize_frames(frames)
    pair_rows: dict[tuple[int, int, int], np.ndarray] = {}
    fallback_rows: dict[tuple[int, int], np.ndarray] = {}
    global_pair_rows: dict[tuple[int, int], np.ndarray] = {}
    global_fallback_rows: dict[int, np.ndarray] = {}

    def row_for(table: dict[object, np.ndarray], key: object) -> np.ndarray:
        row = table.get(key)
        if row is None:
            row = np.ones((vocab_size,), dtype=np.float64)
            table[key] = row
        return row

    for previous, current in zip(arr[:-1], arr[1:]):
        prev_flat = previous.reshape(-1)
        curr_flat = current.reshape(-1)
        for position, (b, c) in enumerate(zip(prev_flat.tolist(), curr_flat.tolist())):
            if b < vocab_size and c < vocab_size:
                row_for(fallback_rows, (position, b))[c] += 1.0
                row_for(global_fallback_rows, b)[c] += 1.0

    for prev2, prev1, current in zip(arr[:-2], arr[1:-1], arr[2:]):
        prev2_flat = prev2.reshape(-1)
        prev1_flat = prev1.reshape(-1)
        curr_flat = current.reshape(-1)
        for position, (a, b, c) in enumerate(zip(prev2_flat.tolist(), prev1_flat.tolist(), curr_flat.tolist())):
            if a < vocab_size and b < vocab_size and c < vocab_size:
                row_for(pair_rows, (position, a, b))[c] += 1.0
                row_for(global_pair_rows, (a, b))[c] += 1.0
    return PositionPairTransitionModel(
        pair_rows=pair_rows,
        fallback_rows=fallback_rows,
        global_pair_rows=global_pair_rows,
        global_fallback_rows=global_fallback_rows,
        vocab_size=vocab_size,
    )


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


def _load_next_frame_runtime(
    *,
    profile: str,
    method: str,
    context_frames: int,
    vocab_size: int,
    device: str,
    dtype: str,
    cache_dir: str | Path | None,
    model_url: str | None,
    gpt_module_path: str | Path | None,
    model_loader: Callable[..., object] | None,
    checkpoint_path: str | Path | None,
) -> object:
    if model_loader is not None:
        return model_loader(
            device=device,
            dtype=dtype,
            cache_dir=cache_dir,
            model_url=model_url,
            gpt_module_path=gpt_module_path,
        )
    if method == "tiny_frame_predictor":
        return load_tiny_frame_predictor_runtime(
            profile,
            context_frames=context_frames,
            vocab_size=vocab_size,
            device=device,
            dtype=dtype,
            checkpoint_path=checkpoint_path,
        )
    return load_official_commavq_gpt_model(
        device=device,
        dtype=dtype,
        cache_dir=cache_dir,
        model_url=model_url,
        gpt_module_path=gpt_module_path,
    )


def _next_frame_runtime_metadata(model: object) -> dict[str, object]:
    payload = gpt_model_runtime_metadata(model)
    if hasattr(model, "_tac_model_profile"):
        payload["model_profile"] = getattr(model, "_tac_model_profile")
    return payload


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
    checkpoint_path: str | Path | None = None,
) -> dict[str, object]:
    config = load_next_frame_predictor_profile(profile)
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

    model = _load_next_frame_runtime(
        profile=profile,
        method=config.method,
        context_frames=effective_context_frames,
        vocab_size=vocab_size,
        device=device,
        dtype=dtype,
        cache_dir=cache_dir,
        model_url=model_url,
        gpt_module_path=gpt_module_path,
        model_loader=model_loader,
        checkpoint_path=checkpoint_path,
    )
    runtime_metadata = _next_frame_runtime_metadata(model)
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
    result = {
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
    result.update({key: value for key, value in runtime_metadata.items() if key != "model_url"})
    if "model_url" in runtime_metadata:
        result["model_url"] = runtime_metadata["model_url"]
    return result
