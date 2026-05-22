# SPDX-License-Identifier: MIT
"""NumPy scorer-input cache for local MLX training and surrogate calibration.

The authoritative evaluator consumes inflated ``.raw`` RGB files through
non-overlapping frame pairs, then feeds:

- SegNet: last frame only, resized RGB, shape ``(B, 3, 384, 512)``.
- PoseNet: both frames, resized RGB -> YUV6, shape ``(B, 12, 192, 256)``.

This module builds those exact input tensors as NumPy arrays.  It uses PyTorch
CPU for interpolation/YUV math to match the upstream preprocessing semantics,
then writes NumPy artifacts that MLX and PyTorch can both consume.  The cache
is a training/surrogate artifact only; it is not a score.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

__all__ = [
    "CAMERA_HW",
    "CAMERA_SIZE",
    "SEGNET_INPUT_HW",
    "SEQ_LEN",
    "ScorerInputBatch",
    "count_video_pairs",
    "load_raw_video_memmap",
    "non_overlapping_pair_indices",
    "preprocess_scorer_inputs_from_pairs",
    "write_scorer_input_cache",
    "write_scorer_input_cache_from_raw_file",
    "write_scorer_input_cache_from_video_file",
    "write_scorer_input_cache_hash_manifest_from_raw_file",
]

SCHEMA_VERSION = "mlx_scorer_input_cache.v1"
HASH_MANIFEST_SCHEMA_VERSION = "mlx_scorer_input_cache_hashes.v1"
ARRAY_HASH_DOMAIN = "_array_sha256(dtype_string + json_shape + contiguous_bytes)"
SEQ_LEN = 2
CAMERA_SIZE = (1164, 874)  # upstream frame_utils.py: (W, H)
CAMERA_HW = (874, 1164)
SEGNET_INPUT_SIZE = (512, 384)  # (W, H)
SEGNET_INPUT_HW = (384, 512)
YUV6_INPUT_HW = (192, 256)


@dataclass(frozen=True)
class ScorerInputBatch:
    """Portable NumPy scorer-input batch."""

    segnet_last_rgb: np.ndarray
    posenet_yuv6_pair: np.ndarray
    pair_indices: np.ndarray
    metadata: dict[str, Any]


def load_raw_video_memmap(
    raw_path: str | Path,
    *,
    frame_hw: tuple[int, int] = CAMERA_HW,
) -> np.memmap:
    """Open an inflated contest ``.raw`` file as ``(N, H, W, 3)`` uint8."""

    path = Path(raw_path)
    h, w = int(frame_hw[0]), int(frame_hw[1])
    frame_bytes = h * w * 3
    size = path.stat().st_size
    if size % frame_bytes != 0:
        raise ValueError(
            f"raw byte count is not a multiple of frame size: "
            f"path={path} size={size} frame_bytes={frame_bytes}"
        )
    frame_count = size // frame_bytes
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(frame_count, h, w, 3))


def non_overlapping_pair_indices(frame_count: int, *, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Return upstream-compatible non-overlapping frame-pair indices."""

    if seq_len != 2:
        raise ValueError(f"only seq_len=2 is supported, got {seq_len}")
    usable = int(frame_count) // seq_len * seq_len
    starts = np.arange(0, usable, seq_len, dtype=np.int64)
    return np.stack([starts, starts + 1], axis=1)


def count_video_pairs(video_path: str | Path) -> int:
    """Return the number of non-overlapping frame pairs in an upstream video."""

    return _video_frame_count(Path(video_path)) // SEQ_LEN


def preprocess_scorer_inputs_from_pairs(
    pairs_rgb: np.ndarray,
    *,
    pair_indices: np.ndarray | None = None,
    source: str | None = None,
) -> ScorerInputBatch:
    """Build SegNet/PoseNet NumPy inputs from ``(B, 2, H, W, 3)`` uint8 pairs."""

    pairs = np.asarray(pairs_rgb)
    if pairs.ndim != 5 or pairs.shape[1] != 2 or pairs.shape[-1] != 3:
        raise ValueError(f"expected pairs shape (B, 2, H, W, 3), got {pairs.shape}")
    if pairs.dtype != np.uint8:
        raise TypeError(f"pairs must be uint8 RGB, got {pairs.dtype}")
    b, t, h, w, c = pairs.shape
    if t != SEQ_LEN or c != 3:
        raise ValueError(f"expected seq_len=2 and channels=3, got shape={pairs.shape}")
    if pair_indices is None:
        pair_indices = non_overlapping_pair_indices(b * 2)
    pair_indices = np.asarray(pair_indices, dtype=np.int64)
    if pair_indices.shape != (b, 2):
        raise ValueError(f"pair_indices must have shape {(b, 2)}, got {pair_indices.shape}")

    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.ascontiguousarray(pairs)).float()
    x = x.permute(0, 1, 4, 2, 3).reshape(b * t, 3, h, w).contiguous()
    resized = F.interpolate(x, size=SEGNET_INPUT_HW, mode="bilinear")
    resized_pair = resized.reshape(b, t, 3, *SEGNET_INPUT_HW).contiguous()

    segnet_last_rgb = resized_pair[:, -1, ...].detach().cpu().numpy().astype(np.float32)
    yuv6 = _rgb_to_yuv6_torch(resized)
    posenet_yuv6_pair = (
        yuv6.reshape(b, t, 6, *YUV6_INPUT_HW)
        .reshape(b, t * 6, *YUV6_INPUT_HW)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "frame_shape_hwc": [int(h), int(w), int(c)],
        "seq_len": SEQ_LEN,
        "pair_count": int(b),
        "segnet_last_rgb_shape": list(segnet_last_rgb.shape),
        "posenet_yuv6_pair_shape": list(posenet_yuv6_pair.shape),
        "pair_indices_shape": list(pair_indices.shape),
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return ScorerInputBatch(
        segnet_last_rgb=np.ascontiguousarray(segnet_last_rgb),
        posenet_yuv6_pair=np.ascontiguousarray(posenet_yuv6_pair),
        pair_indices=np.ascontiguousarray(pair_indices),
        metadata=metadata,
    )


def write_scorer_input_cache(
    batch: ScorerInputBatch,
    output_dir: str | Path,
    *,
    archive_sha256: str | None = None,
    inflated_outputs_aggregate_sha256: str | None = None,
    raw_sha256: str | None = None,
) -> dict[str, Any]:
    """Write NumPy scorer-input arrays and return a manifest."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seg_path = out / "segnet_last_rgb.npy"
    pose_path = out / "posenet_yuv6_pair.npy"
    pair_path = out / "pair_indices.npy"
    np.save(seg_path, batch.segnet_last_rgb)
    np.save(pose_path, batch.posenet_yuv6_pair)
    np.save(pair_path, batch.pair_indices)

    manifest = {
        **batch.metadata,
        "archive_sha256": archive_sha256,
        "inflated_outputs_aggregate_sha256": inflated_outputs_aggregate_sha256,
        "raw_sha256": raw_sha256,
        "hash_domain": ARRAY_HASH_DOMAIN,
        "producer_environment": _producer_environment(),
        "artifacts": {
            "segnet_last_rgb": _artifact_record(seg_path),
            "posenet_yuv6_pair": _artifact_record(pose_path),
            "pair_indices": _artifact_record(pair_path),
        },
        "array_sha256": {
            "segnet_last_rgb": _array_sha256(batch.segnet_last_rgb),
            "posenet_yuv6_pair": _array_sha256(batch.posenet_yuv6_pair),
            "pair_indices": _array_sha256(batch.pair_indices),
        },
        "device_contract": {
            "allowed_uses": [
                "local_mlx_training",
                "scorer_surrogate_calibration",
                "prepaid_dispatch_spend_filter_after_score_calibration",
                "cross_backend_tensor_parity",
            ],
            "forbidden_uses": [
                "auth_eval",
                "score_claim",
                "promotion",
                "rank_or_kill",
                "leaderboard_claim",
            ],
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_scorer_input_cache_hash_manifest_from_raw_file(
    raw_path: str | Path,
    output_path: str | Path,
    *,
    archive_sha256: str | None = None,
    inflated_outputs_aggregate_sha256: str | None = None,
    batch_pairs: int = 8,
) -> dict[str, Any]:
    """Stream scorer-input tensors from raw frames and write hash-only manifest.

    This is the contest-Linux bridge for MLX transfer calibration: Modal can
    emit a compact JSON identity artifact without returning multi-GB NumPy
    arrays through the function result payload. The array hashes use the same
    domain as :func:`_array_sha256`, so they are directly comparable to a full
    cache manifest.
    """

    if batch_pairs <= 0:
        raise ValueError(f"batch_pairs must be positive, got {batch_pairs}")
    raw_path = Path(raw_path)
    raw = load_raw_video_memmap(raw_path)
    pair_indices = non_overlapping_pair_indices(raw.shape[0])
    pair_count = len(pair_indices)

    seg_shape = (pair_count, 3, *SEGNET_INPUT_HW)
    pose_shape = (pair_count, 12, *YUV6_INPUT_HW)
    seg_hash = _StreamingArraySha256(seg_shape, np.dtype("float32"))
    pose_hash = _StreamingArraySha256(pose_shape, np.dtype("float32"))

    for start in range(0, pair_count, int(batch_pairs)):
        chunk_indices = pair_indices[start : start + int(batch_pairs)]
        frame_indices = chunk_indices.reshape(-1)
        pairs = np.asarray(raw[frame_indices]).reshape(
            len(chunk_indices), SEQ_LEN, *raw.shape[1:]
        )
        batch = preprocess_scorer_inputs_from_pairs(
            pairs,
            pair_indices=chunk_indices,
            source=str(raw_path),
        )
        seg_hash.update(batch.segnet_last_rgb)
        pose_hash.update(batch.posenet_yuv6_pair)

    manifest = {
        "schema_version": HASH_MANIFEST_SCHEMA_VERSION,
        "source": str(raw_path),
        "hash_only": True,
        "hash_domain": ARRAY_HASH_DOMAIN,
        "streaming_batch_pairs": int(batch_pairs),
        "frame_shape_hwc": [int(raw.shape[1]), int(raw.shape[2]), int(raw.shape[3])],
        "seq_len": SEQ_LEN,
        "pair_count": pair_count,
        "segnet_last_rgb_shape": list(seg_shape),
        "posenet_yuv6_pair_shape": list(pose_shape),
        "pair_indices_shape": list(pair_indices.shape),
        "archive_sha256": archive_sha256,
        "inflated_outputs_aggregate_sha256": inflated_outputs_aggregate_sha256,
        "raw_sha256": _file_sha256(raw_path),
        "producer_environment": _producer_environment(),
        "artifacts": {},
        "omitted_artifacts_reason": "hash_only_manifest_no_tensor_payloads_written",
        "array_sha256": {
            "segnet_last_rgb": seg_hash.hexdigest(),
            "posenet_yuv6_pair": pose_hash.hexdigest(),
            "pair_indices": _array_sha256(pair_indices),
        },
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "device_contract": {
            "allowed_uses": [
                "contest_linux_scorer_input_identity",
                "local_mlx_training_transfer_calibration",
                "surrogate_error_measurement_against_matching_auth_axis",
            ],
            "forbidden_uses": [
                "auth_eval",
                "score_claim",
                "promotion",
                "rank_or_kill",
                "leaderboard_claim",
            ],
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_scorer_input_cache_from_raw_file(
    raw_path: str | Path,
    output_dir: str | Path,
    *,
    archive_sha256: str | None = None,
    inflated_outputs_aggregate_sha256: str | None = None,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    """Build a scorer-input cache from one inflated contest ``.raw`` file."""

    raw = load_raw_video_memmap(raw_path)
    pair_indices = non_overlapping_pair_indices(raw.shape[0])
    if max_pairs is not None:
        max_pairs_int = int(max_pairs)
        if max_pairs_int < 1:
            raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")
        pair_indices = pair_indices[:max_pairs_int]
    frame_indices = pair_indices.reshape(-1)
    pairs = np.asarray(raw[frame_indices]).reshape(len(pair_indices), 2, *raw.shape[1:])
    batch = preprocess_scorer_inputs_from_pairs(
        pairs,
        pair_indices=pair_indices,
        source=str(raw_path),
    )
    return write_scorer_input_cache(
        batch,
        output_dir,
        archive_sha256=archive_sha256,
        inflated_outputs_aggregate_sha256=inflated_outputs_aggregate_sha256,
        raw_sha256=_file_sha256(raw_path),
    )


def write_scorer_input_cache_from_video_file(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    archive_sha256: str | None = None,
    inflated_outputs_aggregate_sha256: str | None = None,
    max_pairs: int | None = None,
    batch_pairs: int = 8,
) -> dict[str, Any]:
    """Build a scorer-input cache from an upstream-format video file.

    The evaluator's ground-truth side uses ``AVVideoDataset`` on CPU.  This
    helper mirrors that PyAV YUV420 -> RGB path, then reuses the same scorer
    preprocessing as inflated raw candidate caches.
    """

    if batch_pairs <= 0:
        raise ValueError(f"batch_pairs must be positive, got {batch_pairs}")
    video_path = Path(video_path)
    if max_pairs is not None:
        max_pairs_int = int(max_pairs)
        if max_pairs_int < 1:
            raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")
        pair_count = max_pairs_int
    else:
        pair_count = count_video_pairs(video_path)
    if pair_count < 1:
        raise ValueError(f"video has no complete frame pairs: {video_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seg_path = out / "segnet_last_rgb.npy"
    pose_path = out / "posenet_yuv6_pair.npy"
    pair_path = out / "pair_indices.npy"

    seg_mm = np.lib.format.open_memmap(
        seg_path,
        mode="w+",
        dtype=np.float32,
        shape=(pair_count, 3, *SEGNET_INPUT_HW),
    )
    pose_mm = np.lib.format.open_memmap(
        pose_path,
        mode="w+",
        dtype=np.float32,
        shape=(pair_count, 12, *YUV6_INPUT_HW),
    )
    pair_indices = non_overlapping_pair_indices(pair_count * SEQ_LEN)
    np.save(pair_path, pair_indices)

    written = 0
    for pairs in _iter_video_pair_chunks(video_path, max_pairs=pair_count, batch_pairs=batch_pairs):
        chunk_count = int(pairs.shape[0])
        chunk_indices = pair_indices[written : written + chunk_count]
        batch = preprocess_scorer_inputs_from_pairs(
            pairs,
            pair_indices=chunk_indices,
            source=str(video_path),
        )
        seg_mm[written : written + chunk_count] = batch.segnet_last_rgb
        pose_mm[written : written + chunk_count] = batch.posenet_yuv6_pair
        written += chunk_count

    if written != pair_count:
        raise ValueError(f"decoded {written} frame pairs from {video_path}, expected {pair_count}")
    seg_mm.flush()
    pose_mm.flush()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": str(video_path),
        "source_kind": "video",
        "source_video_sha256": _file_sha256(video_path),
        "frame_shape_hwc": [CAMERA_HW[0], CAMERA_HW[1], 3],
        "seq_len": SEQ_LEN,
        "pair_count": int(pair_count),
        "segnet_last_rgb_shape": list(seg_mm.shape),
        "posenet_yuv6_pair_shape": list(pose_mm.shape),
        "pair_indices_shape": list(pair_indices.shape),
        "archive_sha256": archive_sha256,
        "inflated_outputs_aggregate_sha256": inflated_outputs_aggregate_sha256,
        "raw_sha256": None,
        "hash_domain": ARRAY_HASH_DOMAIN,
        "producer_environment": _producer_environment(),
        "artifacts": {
            "segnet_last_rgb": _artifact_record(seg_path),
            "posenet_yuv6_pair": _artifact_record(pose_path),
            "pair_indices": _artifact_record(pair_path),
        },
        "array_sha256": {
            "segnet_last_rgb": _array_sha256(seg_mm),
            "posenet_yuv6_pair": _array_sha256(pose_mm),
            "pair_indices": _array_sha256(pair_indices),
        },
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "device_contract": {
            "allowed_uses": [
                "local_mlx_training",
                "scorer_surrogate_calibration",
                "prepaid_dispatch_spend_filter_after_score_calibration",
                "cross_backend_tensor_parity",
            ],
            "forbidden_uses": [
                "auth_eval",
                "score_claim",
                "promotion",
                "rank_or_kill",
                "leaderboard_claim",
            ],
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _rgb_to_yuv6_torch(rgb_chw: Any) -> Any:
    """Torch implementation copied structurally from upstream frame_utils.py."""

    import torch

    h, w = rgb_chw.shape[-2], rgb_chw.shape[-1]
    h2, w2 = h // 2, w // 2
    rgb = rgb_chw[..., :, : 2 * h2, : 2 * w2]

    r = rgb[..., 0, :, :]
    g = rgb[..., 1, :, :]
    b = rgb[..., 2, :, :]

    y = (r * 0.299 + g * 0.587 + b * 0.114).clamp(0.0, 255.0)
    u = ((b - y) / 1.772 + 128.0).clamp(0.0, 255.0)
    v = ((r - y) / 1.402 + 128.0).clamp(0.0, 255.0)

    u_sub = (u[..., 0::2, 0::2] + u[..., 1::2, 0::2] + u[..., 0::2, 1::2] + u[..., 1::2, 1::2]) * 0.25
    v_sub = (v[..., 0::2, 0::2] + v[..., 1::2, 0::2] + v[..., 0::2, 1::2] + v[..., 1::2, 1::2]) * 0.25

    y00 = y[..., 0::2, 0::2]
    y10 = y[..., 1::2, 0::2]
    y01 = y[..., 0::2, 1::2]
    y11 = y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, u_sub, v_sub], dim=-3)


def _video_frame_count(video_path: Path) -> int:
    import av

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        if int(stream.frames or 0) > 0:
            return int(stream.frames)
        return sum(1 for _ in container.decode(stream))
    finally:
        container.close()


def _iter_video_pair_chunks(
    video_path: Path,
    *,
    max_pairs: int,
    batch_pairs: int,
) -> Any:
    import av

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        seq_buf: list[np.ndarray] = []
        pair_buf: list[np.ndarray] = []
        yielded = 0
        for frame in container.decode(stream):
            seq_buf.append(_av_frame_to_rgb_uint8(frame))
            if len(seq_buf) != SEQ_LEN:
                continue
            pair_buf.append(np.stack(seq_buf, axis=0))
            seq_buf = []
            if len(pair_buf) == batch_pairs:
                chunk = np.stack(pair_buf, axis=0)
                yielded += int(chunk.shape[0])
                yield chunk
                pair_buf = []
                if yielded >= max_pairs:
                    return
            if yielded + len(pair_buf) >= max_pairs:
                break
        if pair_buf:
            chunk = np.stack(pair_buf[: max_pairs - yielded], axis=0)
            if chunk.size:
                yield chunk
    finally:
        container.close()


def _av_frame_to_rgb_uint8(frame: Any) -> np.ndarray:
    """Convert a PyAV YUV420 frame like upstream ``AVVideoDataset``."""

    import torch
    import torch.nn.functional as F

    h, w = frame.height, frame.width
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(h, frame.planes[0].line_size)[:, :w]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(h // 2, frame.planes[1].line_size)[:, : w // 2]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(h // 2, frame.planes[2].line_size)[:, : w // 2]

    y_t = torch.from_numpy(y.copy()).float()
    u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)

    u_up = F.interpolate(u_t, size=(h, w), mode="bilinear", align_corners=False).squeeze()
    v_up = F.interpolate(v_t, size=(h, w), mode="bilinear", align_corners=False).squeeze()

    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)

    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.stack([r, g, b], dim=-1).round().to(torch.uint8).numpy()


def _array_sha256(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(contiguous.dtype).encode("utf-8"))
    h.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("utf-8"))
    h.update(contiguous.tobytes())
    return h.hexdigest()


def _producer_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "numpy_version": np.__version__,
    }
    try:
        import torch

        env["torch_version"] = str(torch.__version__)
    except Exception as exc:  # pragma: no cover - only exercised in torch-free envs.
        env["torch_version"] = None
        env["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    return env


class _StreamingArraySha256:
    """Streaming equivalent of ``_array_sha256`` for first-axis chunks."""

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self._shape = tuple(int(x) for x in shape)
        self._dtype = np.dtype(dtype)
        self._seen = 0
        self._h = hashlib.sha256()
        self._h.update(str(self._dtype).encode("utf-8"))
        self._h.update(
            json.dumps(list(self._shape), separators=(",", ":")).encode("utf-8")
        )

    def update(self, chunk: np.ndarray) -> None:
        arr = np.ascontiguousarray(chunk)
        if arr.dtype != self._dtype:
            raise TypeError(f"chunk dtype {arr.dtype} != expected {self._dtype}")
        if arr.ndim != len(self._shape) or arr.shape[1:] != self._shape[1:]:
            raise ValueError(
                f"chunk shape {arr.shape} is incompatible with stream shape {self._shape}"
            )
        self._seen += int(arr.shape[0])
        if self._seen > self._shape[0]:
            raise ValueError(f"stream received too many rows: {self._seen}>{self._shape[0]}")
        self._h.update(arr.tobytes())

    def hexdigest(self) -> str:
        if self._seen != self._shape[0]:
            raise ValueError(f"stream incomplete: {self._seen}!={self._shape[0]}")
        return self._h.hexdigest()


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return value
