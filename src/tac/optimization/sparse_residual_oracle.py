# SPDX-License-Identifier: MIT
"""Charged sparse-residual oracle probes for raw contest outputs.

This module is an advisory search primitive.  It uses compress-time access to a
target raw video to choose a small sparse residual overlay, packs that overlay
with the existing engineered-corrections binary grammar, and applies it to an
already inflated raw output.  The resulting raw is useful for measuring whether
pixel-level residual capacity can move the scorer enough to justify building a
real ``inflate.py`` consumer.

Authority boundary: every payload emitted here is non-promotable until the
packed residual bytes are placed in a real archive grammar and consumed by the
stock inflate runtime.  The oracle may use the target video at compress time;
the inflate path remains scorer-free and target-free.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tac.engineered_corrections import pack_sparse_corrections
from tac.optimization.inflate_postprocess_surface import RawVideoShape, validate_raw_size

FrameSelector = Literal["all", "even", "odd"]
GainMetric = Literal["l1", "linf"]


@dataclass(frozen=True)
class SparseResidualOracleConfig:
    top_k_pixels: int
    max_abs_delta: int = 1
    frame_selector: FrameSelector = "all"
    gain_metric: GainMetric = "l1"
    chunk_frames: int = 8
    quantize_bits: int = 8
    compression: str = "zlib"
    rate_cap_bytes: int | None = None

    def validate(self) -> None:
        if self.top_k_pixels <= 0:
            raise ValueError("top_k_pixels must be positive")
        if self.max_abs_delta <= 0:
            raise ValueError("max_abs_delta must be positive")
        if self.frame_selector not in {"all", "even", "odd"}:
            raise ValueError(f"unsupported frame_selector={self.frame_selector!r}")
        if self.gain_metric not in {"l1", "linf"}:
            raise ValueError(f"unsupported gain_metric={self.gain_metric!r}")
        if self.chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        if self.quantize_bits not in {4, 8, 16}:
            raise ValueError("quantize_bits must be one of 4, 8, 16")
        if self.compression not in {"zlib", "none"}:
            raise ValueError("compression must be 'zlib' or 'none'")
        if self.rate_cap_bytes is not None and self.rate_cap_bytes <= 0:
            raise ValueError("rate_cap_bytes must be positive when supplied")

    def as_dict(self) -> dict[str, Any]:
        return {
            "top_k_pixels": self.top_k_pixels,
            "max_abs_delta": self.max_abs_delta,
            "frame_selector": self.frame_selector,
            "gain_metric": self.gain_metric,
            "chunk_frames": self.chunk_frames,
            "quantize_bits": self.quantize_bits,
            "compression": self.compression,
            "rate_cap_bytes": self.rate_cap_bytes,
        }


@dataclass(frozen=True)
class SparseResidualPlan:
    sparse: dict[str, Any]
    packed: bytes
    selected_gain_sum: float
    selected_max_gain: float
    selected_mean_gain: float
    dropped_for_rate_cap: int

    @property
    def packed_bytes(self) -> int:
        return len(self.packed)

    @property
    def packed_sha256(self) -> str:
        return hashlib.sha256(self.packed).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_kept": int(self.sparse["n_kept"]),
            "n_total": int(self.sparse["n_total"]),
            "shape": list(self.sparse["shape"]),
            "quantize_bits": int(self.sparse["quantize_bits"]),
            "scale": float(self.sparse["scale"]),
            "top_k_pct": float(self.sparse["top_k_pct"]),
            "packed_bytes": self.packed_bytes,
            "packed_sha256": self.packed_sha256,
            "selected_gain_sum": self.selected_gain_sum,
            "selected_max_gain": self.selected_max_gain,
            "selected_mean_gain": self.selected_mean_gain,
            "dropped_for_rate_cap": self.dropped_for_rate_cap,
        }


@dataclass(frozen=True)
class SparseResidualApplyResult:
    input_raw: Path
    output_raw: Path
    correction_bin: Path
    shape: RawVideoShape
    input_raw_sha256: str
    output_raw_sha256: str
    correction_bin_sha256: str
    packed_bytes: int
    changed_pixel_count: int
    changed_byte_count: int
    changed_frame_count: int
    max_abs_delta_applied: int

    @property
    def passed_visible_change(self) -> bool:
        return self.changed_byte_count > 0 and self.input_raw_sha256 != self.output_raw_sha256

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_raw": str(self.input_raw.resolve()),
            "output_raw": str(self.output_raw.resolve()),
            "correction_bin": str(self.correction_bin.resolve()),
            "shape": self.shape.as_dict(),
            "input_raw_sha256": self.input_raw_sha256,
            "output_raw_sha256": self.output_raw_sha256,
            "correction_bin_sha256": self.correction_bin_sha256,
            "packed_bytes": self.packed_bytes,
            "changed_pixel_count": self.changed_pixel_count,
            "changed_byte_count": self.changed_byte_count,
            "changed_frame_count": self.changed_frame_count,
            "max_abs_delta_applied": self.max_abs_delta_applied,
            "passed_visible_change": self.passed_visible_change,
            "authority": authority_payload(),
        }


def authority_payload() -> dict[str, Any]:
    return {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_blockers": [
            "oracle_raw_residual_advisory_only",
            "target_video_used_at_compress_time_for_selection",
            "not_stock_inflate_runtime_custody",
            "correction_bytes_not_yet_consumed_by_live_inflate_py",
            "exact_contest_cuda_eval_missing",
        ],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_sparse_residual_plan(
    *,
    baseline_raw: Path,
    target_raw: Path,
    shape: RawVideoShape,
    config: SparseResidualOracleConfig,
) -> SparseResidualPlan:
    """Select and pack top-k sparse residual pixels from target minus baseline."""

    config.validate()
    if shape.channels != 3:
        raise ValueError(
            "sparse residual oracle currently requires RGB raw shape with 3 channels; "
            "the canonical engineered-corrections packer stores (n, 3) values"
        )
    validate_raw_size(baseline_raw, shape)
    validate_raw_size(target_raw, shape)

    baseline = np.memmap(
        baseline_raw,
        dtype=np.uint8,
        mode="r",
        shape=(shape.frames, shape.height, shape.width, shape.channels),
    )
    target = np.memmap(
        target_raw,
        dtype=np.uint8,
        mode="r",
        shape=(shape.frames, shape.height, shape.width, shape.channels),
    )

    pixels_per_frame = shape.height * shape.width
    indices_parts: list[np.ndarray] = []
    values_parts: list[np.ndarray] = []
    gain_parts: list[np.ndarray] = []

    for start in range(0, shape.frames, config.chunk_frames):
        stop = min(shape.frames, start + config.chunk_frames)
        frame_ids = np.asarray(
            [idx for idx in range(start, stop) if _frame_selected(idx, config.frame_selector)],
            dtype=np.int64,
        )
        if frame_ids.size == 0:
            continue
        base_chunk = baseline[frame_ids].astype(np.int16)
        target_chunk = target[frame_ids].astype(np.int16)
        clipped = np.clip(
            target_chunk - base_chunk,
            -config.max_abs_delta,
            config.max_abs_delta,
        ).astype(np.int16)
        if config.gain_metric == "l1":
            gain = np.abs(clipped).sum(axis=-1)
        else:
            gain = np.abs(clipped).max(axis=-1)
        flat_gain = gain.reshape(-1)
        nonzero = int(np.count_nonzero(flat_gain))
        if nonzero == 0:
            continue
        local_k = min(config.top_k_pixels, nonzero)
        local_idx = np.argpartition(flat_gain, -local_k)[-local_k:]
        local_idx = local_idx[flat_gain[local_idx] > 0]
        if local_idx.size == 0:
            continue
        local_values = clipped.reshape(-1, shape.channels)[local_idx]
        local_frame_slot = local_idx // pixels_per_frame
        local_pixel_offset = local_idx % pixels_per_frame
        global_indices = frame_ids[local_frame_slot] * pixels_per_frame + local_pixel_offset
        indices_parts.append(global_indices.astype(np.uint32))
        values_parts.append(local_values.astype(np.int16))
        gain_parts.append(flat_gain[local_idx].astype(np.float32))

    del baseline
    del target

    if not indices_parts:
        sparse = _empty_sparse(shape, config)
        packed = pack_sparse_corrections(sparse, compression=config.compression)
        return SparseResidualPlan(
            sparse=sparse,
            packed=packed,
            selected_gain_sum=0.0,
            selected_max_gain=0.0,
            selected_mean_gain=0.0,
            dropped_for_rate_cap=0,
        )

    indices = np.concatenate(indices_parts)
    values = np.concatenate(values_parts)
    gains = np.concatenate(gain_parts)
    order = np.lexsort((indices.astype(np.uint64), -gains.astype(np.float64)))
    keep = order[: config.top_k_pixels]
    indices = indices[keep]
    values = values[keep]
    gains = gains[keep]

    sparse = _build_sparse(shape=shape, config=config, indices=indices, values=values)
    before_cap_n = int(sparse["n_kept"])
    packed = _pack_with_optional_cap(sparse, config)
    dropped = before_cap_n - int(sparse["n_kept"])

    kept_gains = gains[: int(sparse["n_kept"])]
    return SparseResidualPlan(
        sparse=sparse,
        packed=packed,
        selected_gain_sum=float(kept_gains.sum()) if kept_gains.size else 0.0,
        selected_max_gain=float(kept_gains.max()) if kept_gains.size else 0.0,
        selected_mean_gain=float(kept_gains.mean()) if kept_gains.size else 0.0,
        dropped_for_rate_cap=dropped,
    )


def write_sparse_residual_candidate(
    *,
    baseline_raw: Path,
    output_raw: Path,
    correction_bin: Path,
    plan: SparseResidualPlan,
    shape: RawVideoShape,
) -> SparseResidualApplyResult:
    """Copy ``baseline_raw`` and apply ``plan`` in place to ``output_raw``."""

    validate_raw_size(baseline_raw, shape)
    output_raw.parent.mkdir(parents=True, exist_ok=True)
    correction_bin.parent.mkdir(parents=True, exist_ok=True)
    if output_raw.exists():
        output_raw.unlink()
    shutil.copyfile(baseline_raw, output_raw)
    correction_bin.write_bytes(plan.packed)

    raw = np.memmap(output_raw, dtype=np.uint8, mode="r+", shape=(shape.frames * shape.height * shape.width, shape.channels))
    indices = np.asarray(plan.sparse["indices"], dtype=np.uint32)
    dequant = _dequantize_values(plan.sparse)
    before = raw[indices].astype(np.int16)
    after = np.clip(np.rint(before.astype(np.float32) + dequant), 0, 255).astype(np.uint8)
    raw[indices] = after
    raw.flush()
    del raw

    diff = after.astype(np.int16) - before
    changed_pixel_mask = np.any(diff != 0, axis=-1)
    changed_pixel_count = int(np.count_nonzero(changed_pixel_mask))
    changed_byte_count = int(np.count_nonzero(diff))
    if changed_pixel_count:
        changed_frames = np.unique((indices[changed_pixel_mask].astype(np.uint64) // (shape.height * shape.width))).size
        max_abs_delta = int(np.abs(diff).max())
    else:
        changed_frames = 0
        max_abs_delta = 0

    return SparseResidualApplyResult(
        input_raw=baseline_raw,
        output_raw=output_raw,
        correction_bin=correction_bin,
        shape=shape,
        input_raw_sha256=sha256_file(baseline_raw),
        output_raw_sha256=sha256_file(output_raw),
        correction_bin_sha256=sha256_file(correction_bin),
        packed_bytes=plan.packed_bytes,
        changed_pixel_count=changed_pixel_count,
        changed_byte_count=changed_byte_count,
        changed_frame_count=int(changed_frames),
        max_abs_delta_applied=max_abs_delta,
    )


def write_charge_proxy_archive(
    *,
    baseline_archive: Path,
    correction_payload: bytes,
    output_archive: Path,
) -> dict[str, Any]:
    """Write a size-accounting archive proxy: baseline archive plus payload."""

    if not baseline_archive.is_file():
        raise FileNotFoundError(baseline_archive)
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    if output_archive.exists():
        output_archive.unlink()
    with baseline_archive.open("rb") as src, output_archive.open("wb") as dst:
        shutil.copyfileobj(src, dst)
        dst.write(correction_payload)
    return {
        "baseline_archive": str(baseline_archive.resolve()),
        "baseline_archive_bytes": baseline_archive.stat().st_size,
        "baseline_archive_sha256": sha256_file(baseline_archive),
        "charge_proxy_archive": str(output_archive.resolve()),
        "charge_proxy_archive_bytes": output_archive.stat().st_size,
        "charge_proxy_archive_sha256": sha256_file(output_archive),
        "correction_payload_bytes": len(correction_payload),
        "extra_bytes_exact": output_archive.stat().st_size - baseline_archive.stat().st_size,
        "is_valid_submission_archive_claim": False,
        "notes": "Proxy file is for upstream/evaluate.py size charging only; not a promotion archive.",
    }


def build_sparse_residual_plan_from_global_values(
    *,
    shape: RawVideoShape,
    indices: np.ndarray,
    values: np.ndarray,
    quantize_bits: int = 8,
    compression: str = "zlib",
    rate_cap_bytes: int | None = None,
    gains: np.ndarray | None = None,
) -> SparseResidualPlan:
    """Pack explicit global pixel residuals with the canonical sparse grammar."""

    if shape.channels != 3:
        raise ValueError(
            "sparse residual oracle currently requires RGB raw shape with 3 channels; "
            "the canonical engineered-corrections packer stores (n, 3) values"
        )
    cfg = SparseResidualOracleConfig(
        top_k_pixels=max(1, int(np.asarray(indices).size)),
        quantize_bits=quantize_bits,
        compression=compression,
        rate_cap_bytes=rate_cap_bytes,
    )
    cfg.validate()
    sparse = _build_sparse(
        shape=shape,
        config=cfg,
        indices=np.asarray(indices, dtype=np.uint32),
        values=np.asarray(values, dtype=np.int16),
    )
    before_cap_n = int(sparse["n_kept"])
    packed = _pack_with_optional_cap(sparse, cfg)
    kept_n = int(sparse["n_kept"])
    kept_gains = np.asarray(gains if gains is not None else [], dtype=np.float32)[:kept_n]
    return SparseResidualPlan(
        sparse=sparse,
        packed=packed,
        selected_gain_sum=float(kept_gains.sum()) if kept_gains.size else 0.0,
        selected_max_gain=float(kept_gains.max()) if kept_gains.size else 0.0,
        selected_mean_gain=float(kept_gains.mean()) if kept_gains.size else 0.0,
        dropped_for_rate_cap=before_cap_n - kept_n,
    )


def plan_payload(config: SparseResidualOracleConfig, plan: SparseResidualPlan) -> dict[str, Any]:
    return {
        "schema": "sparse_residual_oracle_plan.v1",
        "config": config.as_dict(),
        "plan": plan.as_dict(),
        "authority": authority_payload(),
    }


def _frame_selected(frame_index: int, selector: FrameSelector) -> bool:
    if selector == "all":
        return True
    if selector == "even":
        return frame_index % 2 == 0
    if selector == "odd":
        return frame_index % 2 == 1
    raise ValueError(f"unsupported frame selector: {selector}")


def _empty_sparse(shape: RawVideoShape, config: SparseResidualOracleConfig) -> dict[str, Any]:
    dtype = np.float16 if config.quantize_bits == 16 else np.int8
    return {
        "indices": np.asarray([], dtype=np.uint32),
        "values": np.zeros((0, shape.channels), dtype=dtype),
        "scale": 1.0,
        "shape": [shape.frames, shape.height, shape.width, shape.channels],
        "top_k_pct": 0.0,
        "quantize_bits": config.quantize_bits,
        "n_kept": 0,
        "n_total": shape.frames * shape.height * shape.width,
    }


def _build_sparse(
    *,
    shape: RawVideoShape,
    config: SparseResidualOracleConfig,
    indices: np.ndarray,
    values: np.ndarray,
) -> dict[str, Any]:
    if values.size == 0:
        return _empty_sparse(shape, config)
    scale = float(np.abs(values).max())
    if scale < 1e-12:
        scale = 1.0
    if config.quantize_bits == 16:
        encoded = values.astype(np.float16)
    elif config.quantize_bits == 8:
        encoded = np.clip(np.rint(values.astype(np.float32) / scale * 127.0), -127, 127).astype(np.int8)
    elif config.quantize_bits == 4:
        encoded = np.clip(np.rint(values.astype(np.float32) / scale * 7.0), -7, 7).astype(np.int8)
    else:  # pragma: no cover - validated earlier
        raise ValueError(f"unsupported quantize_bits={config.quantize_bits}")
    n_total = shape.frames * shape.height * shape.width
    return {
        "indices": indices.astype(np.uint32),
        "values": encoded,
        "scale": scale,
        "shape": [shape.frames, shape.height, shape.width, shape.channels],
        "top_k_pct": float(indices.size) / float(n_total) * 100.0,
        "quantize_bits": config.quantize_bits,
        "n_kept": int(indices.size),
        "n_total": int(n_total),
    }


def _pack_with_optional_cap(sparse: dict[str, Any], config: SparseResidualOracleConfig) -> bytes:
    packed = pack_sparse_corrections(sparse, compression=config.compression)
    if config.rate_cap_bytes is None or len(packed) <= config.rate_cap_bytes:
        return packed

    indices = np.asarray(sparse["indices"])
    values = np.asarray(sparse["values"])
    n = int(indices.size)
    while n > 0 and len(packed) > config.rate_cap_bytes:
        drop = max(1, n // 10)
        n = max(0, n - drop)
        sparse["indices"] = indices[:n]
        sparse["values"] = values[:n]
        sparse["n_kept"] = n
        sparse["top_k_pct"] = float(n) / float(sparse["n_total"]) * 100.0
        packed = pack_sparse_corrections(sparse, compression=config.compression)
    return packed


def _dequantize_values(sparse: dict[str, Any]) -> np.ndarray:
    values = np.asarray(sparse["values"])
    scale = float(sparse["scale"])
    qbits = int(sparse["quantize_bits"])
    if qbits == 16:
        return values.astype(np.float32)
    if qbits == 8:
        return values.astype(np.float32) / 127.0 * scale
    if qbits == 4:
        return values.astype(np.float32) / 7.0 * scale
    raise ValueError(f"unsupported quantize_bits={qbits}")
