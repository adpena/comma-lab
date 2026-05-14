# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import (
    load_commavq_dataset,
    load_local_commavq_record_sample,
    resolve_commavq_data_files,
    resolve_local_commavq_cached_data_files,
)
from .token_rgb_bridge import decode_commavq_tokens_to_rgb_frames


def _bucket(value: float, *, thresholds: tuple[float, ...]) -> int:
    for index, threshold in enumerate(thresholds):
        if value < threshold:
            return index
    return len(thresholds)


def _quantize_nonnegative(value: float, *, step: float) -> int:
    if step <= 0.0:
        raise ValueError("step must be positive")
    return max(0, int(round(max(0.0, float(value)) / step)))


def _coerce_rgb_frames(frames) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError("frames must have shape (frames, height, width, 3) or (height, width, 3)")
    if arr.shape[0] == 0:
        raise ValueError("frames must contain at least one frame")
    normalized = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=255.0, neginf=0.0)
    if float(np.max(normalized)) <= 1.5:
        normalized = normalized * 255.0
    return np.clip(normalized, 0.0, 255.0)


def _coerce_nchw_frames(frames) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 4 or arr.shape[1] != 3:
        raise ValueError("frames must have shape (frames, 3, height, width) or (3, height, width)")
    if arr.shape[0] == 0:
        raise ValueError("frames must contain at least one frame")
    normalized = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=255.0, neginf=0.0)
    if float(np.max(normalized)) <= 1.5:
        normalized = normalized * 255.0
    return np.clip(normalized, 0.0, 255.0)


def _select_keyframe_indices(frame_count: int, *, max_keyframes: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if max_keyframes <= 0:
        raise ValueError("max_keyframes must be positive")
    if frame_count <= max_keyframes:
        return np.arange(frame_count, dtype=np.int64)
    if max_keyframes == 1:
        return np.array([frame_count // 2], dtype=np.int64)
    return np.array(
        [(index * (frame_count - 1)) // (max_keyframes - 1) for index in range(max_keyframes)],
        dtype=np.int64,
    )


def _sample_token_keyframes(tokens, *, max_keyframes: int) -> np.ndarray:
    token_arr = np.asarray(tokens)
    return token_arr[_select_keyframe_indices(token_arr.shape[0], max_keyframes=max_keyframes)]


def _luma(frames: np.ndarray) -> np.ndarray:
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.tensordot(frames, weights, axes=([-1], [0]))


def _mean_abs_gradient(values: np.ndarray) -> float:
    parts: list[np.ndarray] = []
    if values.shape[1] > 1:
        parts.append(np.abs(np.diff(values, axis=1)))
    if values.shape[2] > 1:
        parts.append(np.abs(np.diff(values, axis=2)))
    if not parts:
        return 0.0
    return float(np.mean([part.mean(dtype=np.float64) for part in parts]))


def _unpack_bridge_loader_result(result):
    if not isinstance(result, tuple):
        raise TypeError("bridge_loader must return a tuple")
    if len(result) == 2:
        decoder, transpose_and_clip_fn = result
        return decoder, transpose_and_clip_fn, {}
    if len(result) == 3:
        decoder, transpose_and_clip_fn, metadata = result
        return decoder, transpose_and_clip_fn, dict(metadata or {})
    raise ValueError("bridge_loader must return (decoder, transpose_and_clip_fn[, metadata])")


def _resolve_decode_device(device: str) -> str:
    return "cpu" if device == "auto" else device


def _load_rgb_bridge(
    *,
    bridge_loader,
    device: str,
    dtype: str,
    commavq_root: str | Path | None,
    decoder_url: str | None,
):
    if bridge_loader is None:
        raise ValueError("bridge_loader is required for local-only RGB semantic labeling")
    loader_kwargs = {
        "device": device,
        "dtype": dtype,
        "commavq_root": commavq_root,
    }
    if decoder_url is not None:
        loader_kwargs["decoder_url"] = decoder_url
    decoder, transpose_and_clip_fn, _bridge_metadata = _unpack_bridge_loader_result(bridge_loader(**loader_kwargs))
    return decoder, transpose_and_clip_fn, _resolve_decode_device(device)


def _extract_rgb_semantic_label_from_loaded_bridge(
    tokens,
    *,
    decoder,
    transpose_and_clip_fn,
    batch_size: int,
    max_keyframes: int,
    device: str,
) -> tuple[int, ...]:
    sampled_tokens = _sample_token_keyframes(tokens, max_keyframes=max_keyframes)
    frames = decode_commavq_tokens_to_rgb_frames(
        sampled_tokens,
        decoder=decoder,
        transpose_and_clip_fn=transpose_and_clip_fn,
        batch_size=batch_size,
        device=device,
    )
    return _rgb_semantic_label_tuple_from_sampled_frames(_coerce_rgb_frames(frames))


def _rgb_semantic_label_tuple_from_sampled_frames(sampled: np.ndarray) -> tuple[int, ...]:
    luma = _luma(sampled)
    chroma_span = sampled.max(axis=-1) - sampled.min(axis=-1)
    frame_luma_means = luma.mean(axis=(1, 2), dtype=np.float64)

    split = max(1, sampled.shape[1] // 2)
    top = sampled[:, :split, :, :]
    bottom = sampled[:, split:, :, :]
    if bottom.shape[1] == 0:
        bottom = sampled[:, -1:, :, :]

    top_luma = _luma(top)
    bottom_luma = _luma(bottom)
    bottom_chroma = bottom.max(axis=-1) - bottom.min(axis=-1)

    sky_ratio = float(
        np.mean(
            (top[..., 2] >= top[..., 1] + 12.0)
            & (top[..., 2] >= top[..., 0] + 20.0)
            & (top_luma >= 96.0)
        )
    )
    road_ratio = float(
        np.mean(
            (bottom_luma >= 32.0)
            & (bottom_luma <= 144.0)
            & (bottom_chroma <= 36.0)
        )
    )
    temporal_delta = 0.0
    if frame_luma_means.size >= 2:
        temporal_delta = float(np.mean(np.abs(np.diff(frame_luma_means))))

    return (
        _quantize_nonnegative(float(luma.mean(dtype=np.float64)), step=8.0),
        _quantize_nonnegative(float(luma.std(dtype=np.float64)), step=4.0),
        _quantize_nonnegative(float(chroma_span.mean(dtype=np.float64)), step=4.0),
        _bucket(float((sampled[..., 0] - sampled[..., 2]).mean(dtype=np.float64)), thresholds=(-24.0, -8.0, 8.0, 24.0)),
        _bucket(sky_ratio, thresholds=(0.02, 0.12, 0.30)),
        _bucket(road_ratio, thresholds=(0.05, 0.20, 0.45)),
        _quantize_nonnegative(_mean_abs_gradient(luma), step=2.0),
        _bucket(temporal_delta, thresholds=(2.0, 8.0, 20.0)),
    )


def _rgb_semantic_label_tuple_from_sampled_nchw(sampled: np.ndarray) -> tuple[int, ...]:
    arr = _coerce_nchw_frames(sampled)
    red = arr[:, 0, :, :]
    green = arr[:, 1, :, :]
    blue = arr[:, 2, :, :]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    chroma_span = np.max(arr, axis=1) - np.min(arr, axis=1)
    frame_luma_means = luma.mean(axis=(1, 2), dtype=np.float64)

    split = max(1, arr.shape[2] // 2)
    top = arr[:, :, :split, :]
    bottom = arr[:, :, split:, :]
    if bottom.shape[2] == 0:
        bottom = arr[:, :, -1:, :]

    top_luma = 0.299 * top[:, 0] + 0.587 * top[:, 1] + 0.114 * top[:, 2]
    bottom_luma = 0.299 * bottom[:, 0] + 0.587 * bottom[:, 1] + 0.114 * bottom[:, 2]
    bottom_chroma = np.max(bottom, axis=1) - np.min(bottom, axis=1)

    sky_ratio = float(
        np.mean(
            (top[:, 2] >= top[:, 1] + 12.0)
            & (top[:, 2] >= top[:, 0] + 20.0)
            & (top_luma >= 96.0)
        )
    )
    road_ratio = float(
        np.mean(
            (bottom_luma >= 32.0)
            & (bottom_luma <= 144.0)
            & (bottom_chroma <= 36.0)
        )
    )
    temporal_delta = 0.0
    if frame_luma_means.size >= 2:
        temporal_delta = float(np.mean(np.abs(np.diff(frame_luma_means))))

    def mean_abs_gradient_nchw(values: np.ndarray) -> float:
        parts: list[np.ndarray] = []
        if values.shape[1] > 1:
            parts.append(np.abs(np.diff(values, axis=1)))
        if values.shape[2] > 1:
            parts.append(np.abs(np.diff(values, axis=2)))
        if not parts:
            return 0.0
        return float(np.mean([part.mean(dtype=np.float64) for part in parts]))

    return (
        _quantize_nonnegative(float(luma.mean(dtype=np.float64)), step=8.0),
        _quantize_nonnegative(float(luma.std(dtype=np.float64)), step=4.0),
        _quantize_nonnegative(float(chroma_span.mean(dtype=np.float64)), step=4.0),
        _bucket(float((red - blue).mean(dtype=np.float64)), thresholds=(-24.0, -8.0, 8.0, 24.0)),
        _bucket(sky_ratio, thresholds=(0.02, 0.12, 0.30)),
        _bucket(road_ratio, thresholds=(0.05, 0.20, 0.45)),
        _quantize_nonnegative(mean_abs_gradient_nchw(luma), step=2.0),
        _bucket(temporal_delta, thresholds=(2.0, 8.0, 20.0)),
    )


def _decode_sampled_tokens_to_nchw_frames(
    tokens,
    *,
    decoder,
    batch_size: int,
    device: str,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    arr = np.asarray(tokens)
    if getattr(decoder, "_tac_input_kind", "torch") == "numpy":
        outputs = [np.asarray(decoder(np.array(arr[start : start + batch_size], copy=False))) for start in range(0, arr.shape[0], batch_size)]
        return _coerce_nchw_frames(np.concatenate(outputs, axis=0))

    import torch

    outputs: list[np.ndarray] = []
    for start in range(0, arr.shape[0], batch_size):
        cube_batch = arr[start : start + batch_size]
        batch_input = torch.from_numpy(np.array(cube_batch.reshape(cube_batch.shape[0], -1), copy=True)).to(
            device=device,
            dtype=torch.long,
        )
        with torch.inference_mode():
            decoded = decoder(batch_input)
        decoded_np = decoded.detach().cpu().numpy() if hasattr(decoded, "detach") else np.asarray(decoded)
        outputs.append(decoded_np)
    return _coerce_nchw_frames(np.concatenate(outputs, axis=0))


def _write_label_map_json(path: Path, label_map: dict[str, list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(label_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _chunk_example_samples(
    file_names: list[str],
    sampled_tokens_per_example: list[np.ndarray],
    *,
    max_frames_per_chunk: int,
):
    if max_frames_per_chunk <= 0:
        raise ValueError("max_frames_per_chunk must be positive")
    chunk_file_names: list[str] = []
    chunk_tokens: list[np.ndarray] = []
    chunk_frames = 0
    for file_name, sampled_tokens in zip(file_names, sampled_tokens_per_example):
        frame_count = int(sampled_tokens.shape[0])
        if chunk_tokens and chunk_frames + frame_count > max_frames_per_chunk:
            yield chunk_file_names, chunk_tokens
            chunk_file_names = []
            chunk_tokens = []
            chunk_frames = 0
        chunk_file_names.append(file_name)
        chunk_tokens.append(sampled_tokens)
        chunk_frames += frame_count
    if chunk_tokens:
        yield chunk_file_names, chunk_tokens


def rgb_semantic_label_tuple(frames, *, max_keyframes: int = 6) -> tuple[int, ...]:
    arr = _coerce_rgb_frames(frames)
    sampled = arr[_select_keyframe_indices(arr.shape[0], max_keyframes=max_keyframes)]
    return _rgb_semantic_label_tuple_from_sampled_frames(sampled)


def extract_rgb_semantic_label_from_tokens(
    tokens,
    *,
    bridge_loader,
    batch_size: int = 64,
    max_keyframes: int = 6,
    device: str = "cpu",
    dtype: str = "auto",
    commavq_root: str | Path | None = None,
    decoder_url: str | None = None,
) -> tuple[int, ...]:
    decoder, transpose_and_clip_fn, decode_device = _load_rgb_bridge(
        bridge_loader=bridge_loader,
        device=device,
        dtype=dtype,
        commavq_root=commavq_root,
        decoder_url=decoder_url,
    )
    return _extract_rgb_semantic_label_from_loaded_bridge(
        tokens,
        decoder=decoder,
        transpose_and_clip_fn=transpose_and_clip_fn,
        batch_size=batch_size,
        max_keyframes=max_keyframes,
        device=decode_device,
    )


def build_rgb_label_map_sample(
    *,
    output_path: str | Path,
    split=None,
    max_records: int = 64,
    dataset_loader=None,
    bridge_loader=None,
    batch_size: int = 64,
    max_keyframes: int = 6,
    device: str = "cpu",
    dtype: str = "auto",
    commavq_root: str | Path | None = None,
    decoder_url: str | None = None,
) -> dict[str, object]:
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    if bridge_loader is None:
        raise ValueError("bridge_loader is required for local-only RGB semantic labeling")

    target = Path(output_path)
    existing_label_map: dict[str, list[int]] = {}
    if target.exists():
        existing_payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(existing_payload, dict):
            raise ValueError("existing RGB label output must contain a top-level JSON object")
        existing_label_map = {
            str(key): [int(item) for item in value]
            for key, value in existing_payload.items()
            if isinstance(value, list)
        }

    resolved_data_files = resolve_commavq_data_files(split)
    train_files = resolved_data_files.get("train", [])
    cached_train_files = resolve_local_commavq_cached_data_files(train_files)
    if cached_train_files is not None:
        examples = [
            example
            for example in load_local_commavq_record_sample(cached_train_files, max_records=max_records + len(existing_label_map))
            if str(example["json"]["file_name"]) not in existing_label_map
        ][:max_records]
    else:
        dataset = load_commavq_dataset(split=split, dataset_loader=dataset_loader, streaming=True)
        train = dataset["train"]
        examples = []
        for example in train:
            file_name = str(example["json"]["file_name"])
            if file_name in existing_label_map:
                continue
            examples.append(example)
            if len(examples) >= max_records:
                break

    label_map: dict[str, list[int]] = dict(existing_label_map)
    decoder, transpose_and_clip_fn, decode_device = _load_rgb_bridge(
        bridge_loader=bridge_loader,
        device=device,
        dtype=dtype,
        commavq_root=commavq_root,
        decoder_url=decoder_url,
    )
    sampled_tokens_per_example: list[np.ndarray] = []
    file_names: list[str] = []
    for example in examples:
        file_names.append(str(example["json"]["file_name"]))
        sampled_tokens_per_example.append(_sample_token_keyframes(example["token.npy"], max_keyframes=max_keyframes))

    if sampled_tokens_per_example:
        for chunk_file_names, chunk_tokens in _chunk_example_samples(
            file_names,
            sampled_tokens_per_example,
            max_frames_per_chunk=batch_size,
        ):
            merged_tokens = np.concatenate(chunk_tokens, axis=0)
            merged_frames = _decode_sampled_tokens_to_nchw_frames(
                merged_tokens,
                decoder=decoder,
                batch_size=batch_size,
                device=decode_device,
            )
            start = 0
            for file_name, sampled_tokens in zip(chunk_file_names, chunk_tokens):
                count = int(sampled_tokens.shape[0])
                label_map[file_name] = list(_rgb_semantic_label_tuple_from_sampled_nchw(merged_frames[start:start + count]))
                _write_label_map_json(target, label_map)
                start += count

    _write_label_map_json(target, label_map)
    return {
        "command": "lossless_rgb_labels_sample",
        "output_path": str(target),
        "record_count": len(label_map),
        "split": list(split) if isinstance(split, (list, tuple)) else (["challenge"] if split is None else [str(split)]),
        "max_keyframes": max_keyframes,
        "local_only": True,
    }
