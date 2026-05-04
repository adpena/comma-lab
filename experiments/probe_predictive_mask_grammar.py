#!/usr/bin/env python3
"""Predictive mask-grammar empirical byte probe.

This is a planning and byte-screen tool only. It consumes an already-decoded
mask tensor and measures deterministic predictive/lossy payload sketches with
stdlib compression. It does not build a contest archive, does not load scorer
networks, does not dispatch cloud jobs, and cannot support a score claim.
"""
from __future__ import annotations

import argparse
import bz2
import dataclasses
import hashlib
import json
import lzma
import platform
import sys
import zlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "predictive_mask_grammar_byte_probe_v1"
PAYLOAD_SCHEMA = "predictive_mask_grammar_probe_payload_v1"
REPORT_NAME = "predictive_mask_grammar_probe_manifest.json"
EVIDENCE_GRADE = "empirical_byte_probe_only"
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
DEFAULT_PLAN_DIR = REPO_ROOT / "experiments" / "results" / "c063_trace_weighted_mask_grammar_plan_20260502_codex"
DEFAULT_OUTPUT_DIR = DEFAULT_PLAN_DIR / "predictive_mask_grammar_probe"
DEFAULT_COMPRESSORS = ("zlib9", "lzma6", "bz2_9")
PAYLOAD_MAGIC = b"PMGPROBE1"


@dataclass(frozen=True)
class DiscoveryResult:
    path: Path
    reason: str


@dataclass(frozen=True)
class BaselineRecord:
    role: str
    bytes: int
    source: str
    path: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class ProbeConfig:
    decoded_mask_array: Path
    output_dir: Path
    baseline_mask_stream: Path | None = None
    baseline_bytes: int | None = None
    compressors: tuple[str, ...] = DEFAULT_COMPRESSORS
    force: bool = False


@dataclass(frozen=True)
class CandidatePayload:
    candidate_id: str
    family: str
    description: str
    geometry_preservation: str
    lossless_under_probe_model: bool
    exact_evaluable_now: bool
    exact_evaluable_reason: str
    assumptions: tuple[str, ...]
    payload: bytes
    component_raw_bytes: dict[str, int]
    transform_stats: dict[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _json_bytes(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _load_masks(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"decoded mask array must have rank 3, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"decoded mask array must be uint8, got {arr.dtype}")
    if arr.size == 0:
        raise ValueError("decoded mask array is empty")
    return np.ascontiguousarray(arr)


def _class_count(arr: np.ndarray) -> int:
    return int(arr.max()) + 1


def _symbol_bits(arr: np.ndarray) -> int:
    return max(1, int(arr.max()).bit_length())


def _pack_symbols(arr: np.ndarray, *, bits: int | None = None) -> bytes:
    symbol_bits = _symbol_bits(arr) if bits is None else int(bits)
    flat = np.ascontiguousarray(arr).reshape(-1)
    planes = ((flat[:, None] >> np.arange(symbol_bits, dtype=np.uint8)) & 1).astype(np.uint8, copy=False)
    return np.packbits(planes.reshape(-1), bitorder="little").tobytes()


def _pack_bool(mask: np.ndarray) -> bytes:
    return np.packbits(np.ascontiguousarray(mask, dtype=np.uint8).reshape(-1), bitorder="little").tobytes()


def _payload_container(candidate_id: str, metadata: dict[str, Any], parts: list[tuple[str, bytes]]) -> bytes:
    header = {
        "schema": PAYLOAD_SCHEMA,
        "candidate_id": candidate_id,
        "metadata": metadata,
        "parts": [
            {
                "name": name,
                "bytes": len(data),
                "sha256": _sha256_bytes(data),
            }
            for name, data in parts
        ],
    }
    header_bytes = _json_bytes(header)
    return PAYLOAD_MAGIC + len(header_bytes).to_bytes(4, "little") + header_bytes + b"".join(
        data for _name, data in parts
    )


def _candidate(
    *,
    candidate_id: str,
    family: str,
    description: str,
    geometry_preservation: str,
    lossless_under_probe_model: bool,
    assumptions: tuple[str, ...],
    metadata: dict[str, Any],
    parts: list[tuple[str, bytes]],
    transform_stats: dict[str, Any],
) -> CandidatePayload:
    exact_reason = (
        "not exact-evaluable from this byte probe: no contest archive member, "
        "reviewed inflate decoder, archive validator coverage, or exact CUDA auth eval artifact exists"
    )
    payload = _payload_container(candidate_id, metadata, parts)
    return CandidatePayload(
        candidate_id=candidate_id,
        family=family,
        description=description,
        geometry_preservation=geometry_preservation,
        lossless_under_probe_model=lossless_under_probe_model,
        exact_evaluable_now=False,
        exact_evaluable_reason=exact_reason,
        assumptions=assumptions,
        payload=payload,
        component_raw_bytes={name: len(data) for name, data in parts},
        transform_stats=transform_stats,
    )


def _compressor(name: str) -> Callable[[bytes], bytes]:
    if name == "none":
        return bytes
    if name == "zlib6":
        return lambda data: zlib.compress(data, level=6)
    if name == "zlib9":
        return lambda data: zlib.compress(data, level=9)
    if name == "bz2_9":
        return lambda data: bz2.compress(data, compresslevel=9)
    if name == "lzma6":
        return lambda data: lzma.compress(data, preset=6 | lzma.PRESET_EXTREME)
    if name == "lzma9":
        return lambda data: lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
    raise ValueError(f"unknown compressor: {name}")


def _available_compressors(names: Iterable[str]) -> tuple[str, ...]:
    available: list[str] = []
    for name in names:
        _compressor(name)
        available.append(name)
    if not available:
        raise ValueError("at least one compressor is required")
    return tuple(available)


def _boundary_map(arr: np.ndarray) -> np.ndarray:
    boundary = np.zeros(arr.shape, dtype=bool)
    boundary[:, :, 1:] |= arr[:, :, 1:] != arr[:, :, :-1]
    boundary[:, 1:, :] |= arr[:, 1:, :] != arr[:, :-1, :]
    return boundary


def _block_mode_downsample(arr: np.ndarray, *, scale_y: int, scale_x: int) -> tuple[np.ndarray, np.ndarray, float]:
    if scale_y <= 0 or scale_x <= 0:
        raise ValueError("scale factors must be positive")
    frames, height, width = arr.shape
    if height % scale_y or width % scale_x:
        raise ValueError(f"shape {arr.shape} is not divisible by scale {(scale_y, scale_x)}")
    blocks = arr.reshape(frames, height // scale_y, scale_y, width // scale_x, scale_x)
    classes = _class_count(arr)
    counts = np.stack([(blocks == np.uint8(cls)).sum(axis=(2, 4)) for cls in range(classes)], axis=0)
    low = counts.argmax(axis=0).astype(np.uint8)
    recon = np.repeat(np.repeat(low, scale_y, axis=1), scale_x, axis=2).astype(np.uint8, copy=False)
    disagreement = float((recon != arr).mean())
    return low, recon, disagreement


def _row_spans(arr: np.ndarray, *, row_stride: int) -> np.ndarray:
    frames, height, width = arr.shape
    rows = np.arange(0, height, row_stride, dtype=np.int32)
    spans = np.full((frames, _class_count(arr), len(rows), 2), -1, dtype=np.int16)
    for cls in range(_class_count(arr)):
        for row_index, y in enumerate(rows):
            present = arr[:, int(y), :] == np.uint8(cls)
            any_present = present.any(axis=1)
            first = present.argmax(axis=1)
            last = width - 1 - present[:, ::-1].argmax(axis=1)
            spans[any_present, cls, row_index, 0] = first[any_present].astype(np.int16)
            spans[any_present, cls, row_index, 1] = last[any_present].astype(np.int16)
    return spans


def _column_spans(arr: np.ndarray, *, column_stride: int) -> np.ndarray:
    frames, height, width = arr.shape
    columns = np.arange(0, width, column_stride, dtype=np.int32)
    spans = np.full((frames, _class_count(arr), len(columns), 2), -1, dtype=np.int16)
    for cls in range(_class_count(arr)):
        for column_index, x in enumerate(columns):
            present = arr[:, :, int(x)] == np.uint8(cls)
            any_present = present.any(axis=1)
            first = present.argmax(axis=1)
            last = height - 1 - present[:, ::-1].argmax(axis=1)
            spans[any_present, cls, column_index, 0] = first[any_present].astype(np.int16)
            spans[any_present, cls, column_index, 1] = last[any_present].astype(np.int16)
    return spans


def _span_stats(spans: np.ndarray) -> dict[str, Any]:
    valid = spans[..., 0] >= 0
    return {
        "span_shape": [int(v) for v in spans.shape],
        "valid_span_records": int(valid.sum()),
        "missing_span_records": int((~valid).sum()),
    }


def _foveal_band_candidate(arr: np.ndarray, *, scale_y: int, scale_x: int) -> tuple[bytes, bytes, float, dict[str, Any]]:
    frames, height, width = arr.shape
    y0 = int(round(height * 0.35))
    y1 = int(round(height * 0.78))
    y0 = max(0, min(height, y0))
    y1 = max(y0 + 1, min(height, y1))
    low, recon, _base_disagreement = _block_mode_downsample(arr, scale_y=scale_y, scale_x=scale_x)
    recon[:, y0:y1, :] = arr[:, y0:y1, :]
    disagreement = float((recon != arr).mean())
    roi = np.ascontiguousarray(arr[:, y0:y1, :])
    stats = {
        "scale": [scale_y, scale_x],
        "exact_band_y0_y1": [int(y0), int(y1)],
        "exact_band_pixel_fraction": float((y1 - y0) / height),
        "pixel_disagreement_vs_source": disagreement,
        "reconstructed_tensor_sha256": _sha256_bytes(recon.tobytes(order="C")),
    }
    return _pack_symbols(low), _pack_symbols(roi), disagreement, stats


def build_candidate_payloads(arr: np.ndarray) -> list[CandidatePayload]:
    arr = np.ascontiguousarray(arr)
    frames, height, width = arr.shape
    bits = _symbol_bits(arr)
    classes = _class_count(arr)
    common_metadata = {
        "source_shape": [int(frames), int(height), int(width)],
        "class_min": int(arr.min()),
        "class_max": int(arr.max()),
        "symbol_bits": int(bits),
    }
    candidates: list[CandidatePayload] = []

    anchors = arr[0::2]
    pair_residual = np.bitwise_xor(arr[1::2], arr[0::2][: arr[1::2].shape[0]])
    candidates.append(
        _candidate(
            candidate_id="temporal_pair_anchor_xor_lossless",
            family="temporal_pair_delta",
            description="Store even frame-pair anchors and XOR class residuals for odd frames.",
            geometry_preservation="lossless frame-pair geometry under the probe decoder model",
            lossless_under_probe_model=True,
            assumptions=(
                "decoder reconstructs odd frame i+1 as anchor_i XOR residual_i",
                "XOR operates on class-id symbols, not scorer features",
            ),
            metadata={**common_metadata, "pair_anchor_frames": int(anchors.shape[0])},
            parts=[("pair_anchor_symbols", _pack_symbols(anchors, bits=bits)), ("odd_frame_xor_symbols", _pack_symbols(pair_residual, bits=bits))],
            transform_stats={
                "anchor_frames": int(anchors.shape[0]),
                "residual_frames": int(pair_residual.shape[0]),
                "nonzero_residual_fraction": float((pair_residual != 0).mean()) if pair_residual.size else 0.0,
            },
        )
    )

    prev_residual = np.bitwise_xor(arr[1:], arr[:-1]) if frames > 1 else arr[:0]
    candidates.append(
        _candidate(
            candidate_id="temporal_prev_frame_xor_lossless",
            family="temporal_delta",
            description="Store frame 0 and previous-frame XOR residuals for every later frame.",
            geometry_preservation="lossless temporal geometry under the probe decoder model",
            lossless_under_probe_model=True,
            assumptions=(
                "decoder maintains exact previous decoded frame state",
                "payload excludes future decoder code bytes",
            ),
            metadata=common_metadata,
            parts=[("frame0_symbols", _pack_symbols(arr[:1], bits=bits)), ("prev_frame_xor_symbols", _pack_symbols(prev_residual, bits=bits))],
            transform_stats={
                "residual_frames": int(prev_residual.shape[0]),
                "nonzero_residual_fraction": float((prev_residual != 0).mean()) if prev_residual.size else 0.0,
            },
        )
    )

    for interval in (8, 16):
        key_indices = np.arange(0, frames, interval, dtype=np.int32)
        keyframes = arr[key_indices]
        residual_indices = np.array([idx for idx in range(frames) if idx % interval != 0], dtype=np.int32)
        residuals = np.bitwise_xor(arr[residual_indices], arr[residual_indices - 1]) if len(residual_indices) else arr[:0]
        candidates.append(
            _candidate(
                candidate_id=f"keyframe{interval}_prev_xor_lossless",
                family="keyframe_residual_schedule",
                description=f"Store every {interval}th frame as a keyframe plus previous-frame XOR residuals.",
                geometry_preservation="lossless scheduled temporal geometry under the probe decoder model",
                lossless_under_probe_model=True,
                assumptions=(
                    "keyframe cadence is fixed and deterministic",
                    "non-key frames decode from the immediately previous reconstructed frame",
                ),
                metadata={
                    **common_metadata,
                    "keyframe_interval": int(interval),
                    "keyframe_count": int(len(key_indices)),
                },
                parts=[("keyframe_symbols", _pack_symbols(keyframes, bits=bits)), ("nonkey_prev_xor_symbols", _pack_symbols(residuals, bits=bits))],
                transform_stats={
                    "keyframe_count": int(len(key_indices)),
                    "residual_frame_count": int(len(residual_indices)),
                    "nonzero_residual_fraction": float((residuals != 0).mean()) if residuals.size else 0.0,
                },
            )
        )

    interval = 16
    key_indices = np.arange(0, frames, interval, dtype=np.int32)
    residual_indices = np.array([idx for idx in range(frames) if idx % interval != 0], dtype=np.int32)
    change_mask = arr[residual_indices] != arr[residual_indices - 1] if len(residual_indices) else np.zeros((0, height, width), dtype=bool)
    candidates.append(
        _candidate(
            candidate_id="keyframe16_change_map_lossy",
            family="keyframe_residual_schedule",
            description="Store 16-frame keyframes plus changed-pixel masks without changed class values.",
            geometry_preservation="preserves temporal change geometry but not all class identities",
            lossless_under_probe_model=False,
            assumptions=(
                "future decoder would need a deterministic fill or class predictor for changed pixels",
                "screen estimates payload bytes only and excludes decoder/model bytes",
            ),
            metadata={
                **common_metadata,
                "keyframe_interval": interval,
                "keyframe_count": int(len(key_indices)),
            },
            parts=[("keyframe_symbols", _pack_symbols(arr[key_indices], bits=bits)), ("nonkey_change_bits", _pack_bool(change_mask))],
            transform_stats={
                "residual_frame_count": int(len(residual_indices)),
                "changed_pixel_fraction": float(change_mask.mean()) if change_mask.size else 0.0,
            },
        )
    )

    boundary = _boundary_map(arr)
    class_boundary_parts = [
        (f"class{cls}_boundary_bits", _pack_bool(boundary & (arr == np.uint8(cls)))) for cls in range(classes)
    ]
    candidates.append(
        _candidate(
            candidate_id="class_boundary_planes_all_frames",
            family="class_boundary_maps",
            description="Store per-class boundary planes for every frame.",
            geometry_preservation="preserves class-labeled boundary geometry, not interior fill",
            lossless_under_probe_model=False,
            assumptions=(
                "a future grammar decoder would need interior fill seeds/rules",
                "class-boundary maps are a byte screen for geometry-carrying side information",
            ),
            metadata={**common_metadata, "class_count": int(classes)},
            parts=class_boundary_parts,
            transform_stats={
                "boundary_pixel_fraction": float(boundary.mean()),
                "boundary_pixels": int(boundary.sum()),
            },
        )
    )

    boundary_delta = boundary[1::2] ^ boundary[0::2][: boundary[1::2].shape[0]]
    candidates.append(
        _candidate(
            candidate_id="class_boundary_pair_delta",
            family="class_boundary_maps",
            description="Store even-frame boundaries plus pairwise XOR deltas for odd-frame boundaries.",
            geometry_preservation="preserves temporal boundary movement, not full masks",
            lossless_under_probe_model=False,
            assumptions=(
                "boundary deltas are measured at frame-pair level",
                "future decoder would need class/interior reconstruction rules",
            ),
            metadata={**common_metadata, "class_count": int(classes)},
            parts=[("even_boundary_bits", _pack_bool(boundary[0::2])), ("odd_boundary_xor_bits", _pack_bool(boundary_delta))],
            transform_stats={
                "even_boundary_frames": int(boundary[0::2].shape[0]),
                "delta_boundary_frames": int(boundary_delta.shape[0]),
                "delta_boundary_fraction": float(boundary_delta.mean()) if boundary_delta.size else 0.0,
            },
        )
    )

    for row_stride in (1, 4):
        spans = _row_spans(arr, row_stride=row_stride)
        candidates.append(
            _candidate(
                candidate_id=f"row_span_stride{row_stride}_class_predictor",
                family="low_rank_row_column_spans",
                description=f"Store per-frame per-class horizontal spans every {row_stride} row(s).",
                geometry_preservation="preserves row-wise class extents and lane/horizon geometry at the sampled rows",
                lossless_under_probe_model=False,
                assumptions=(
                    "overlap conflicts and unsampled rows require a deterministic future fill rule",
                    "int16 spans assume mask width and height fit signed 16-bit coordinates",
                ),
                metadata={**common_metadata, "row_stride": int(row_stride), "span_dtype": "int16"},
                parts=[("row_spans_int16", spans.tobytes(order="C"))],
                transform_stats=_span_stats(spans),
            )
        )

    column_stride = 4
    spans = _column_spans(arr, column_stride=column_stride)
    candidates.append(
        _candidate(
            candidate_id="column_span_stride4_class_predictor",
            family="low_rank_row_column_spans",
            description="Store per-frame per-class vertical spans every 4 columns.",
            geometry_preservation="preserves column-wise vertical extents at sampled columns",
            lossless_under_probe_model=False,
            assumptions=(
                "overlap conflicts and unsampled columns require a deterministic future fill rule",
                "int16 spans assume mask width and height fit signed 16-bit coordinates",
            ),
            metadata={**common_metadata, "column_stride": int(column_stride), "span_dtype": "int16"},
            parts=[("column_spans_int16", spans.tobytes(order="C"))],
            transform_stats=_span_stats(spans),
        )
    )

    for scale_y, scale_x in ((1, 2), (2, 1)):
        low, recon, disagreement = _block_mode_downsample(arr, scale_y=scale_y, scale_x=scale_x)
        candidates.append(
            _candidate(
                candidate_id=f"anisotropic_downsample_{scale_y}x{scale_x}_mode",
                family="anisotropic_foveal",
                description=f"Block-mode downsample with scale_y={scale_y}, scale_x={scale_x}.",
                geometry_preservation="keeps one spatial axis at full resolution while lossy-compressing the other",
                lossless_under_probe_model=False,
                assumptions=(
                    "nearest/block-mode reconstruction is only a geometry proxy until exact eval",
                    "candidate excludes archive wrapper and runtime decoder bytes",
                ),
                metadata={**common_metadata, "scale": [int(scale_y), int(scale_x)]},
                parts=[("low_mode_symbols", _pack_symbols(low, bits=bits))],
                transform_stats={
                    "low_shape": [int(v) for v in low.shape],
                    "pixel_disagreement_vs_source": disagreement,
                    "reconstructed_tensor_sha256": _sha256_bytes(recon.tobytes(order="C")),
                },
            )
        )

    low_bits, roi_bits, disagreement, stats = _foveal_band_candidate(arr, scale_y=2, scale_x=2)
    candidates.append(
        _candidate(
            candidate_id="foveal_band_exact_2x2_periphery_mode",
            family="anisotropic_foveal",
            description="Store a full-resolution road/horizon band plus 2x2 block-mode periphery.",
            geometry_preservation="keeps the central driving band exact and only downsamples periphery",
            lossless_under_probe_model=False,
            assumptions=(
                "band ratios are deterministic but not scorer-optimized",
                "future decoder must prove the full-resolution band is consumed at inflate time",
            ),
            metadata=common_metadata,
            parts=[("periphery_low_mode_symbols", low_bits), ("exact_foveal_band_symbols", roi_bits)],
            transform_stats=stats,
        )
    )

    return candidates


def _candidate_record(candidate: CandidatePayload, compressors: tuple[str, ...], baseline_bytes: int | None) -> dict[str, Any]:
    raw = candidate.payload
    compression: list[dict[str, Any]] = []
    for compressor_name in compressors:
        compressed = _compressor(compressor_name)(raw)
        entry: dict[str, Any] = {
            "compressor": compressor_name,
            "compressed_size_bytes": len(compressed),
            "compressed_sha256": _sha256_bytes(compressed),
        }
        if baseline_bytes is not None:
            entry["delta_bytes_vs_baseline"] = len(compressed) - int(baseline_bytes)
            entry["beats_baseline_bytescreen"] = len(compressed) < int(baseline_bytes)
        compression.append(entry)
    compression.sort(key=lambda item: (int(item["compressed_size_bytes"]), str(item["compressor"])))
    record: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "description": candidate.description,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "geometry_preservation": candidate.geometry_preservation,
        "lossless_under_probe_model": candidate.lossless_under_probe_model,
        "exact_evaluable_now": candidate.exact_evaluable_now,
        "exact_evaluable_reason": candidate.exact_evaluable_reason,
        "assumptions": list(candidate.assumptions),
        "raw_payload_size_bytes": len(raw),
        "raw_payload_sha256": _sha256_bytes(raw),
        "payload_component_raw_bytes": dict(sorted(candidate.component_raw_bytes.items())),
        "charged_payload_estimate_scope": (
            "compressed probe payload bytes only; excludes future inflate decoder, "
            "validator, archive wrapper, and packed-submission overhead"
        ),
        "compression": compression,
        "best_compression": compression[0],
        "transform_stats": candidate.transform_stats,
    }
    return record


def _iter_json_candidate_files(repo_root: Path) -> Iterator[Path]:
    roots = [
        repo_root / "experiments" / "results" / "c063_trace_weighted_mask_grammar_plan_20260502_codex",
        repo_root / "experiments" / "results",
    ]
    names = {
        "atom_plan_manifest.json",
        "build_manifest.json",
        "cmg2_foveated_repair_plan.json",
        "cmg2_mask_codec_probe_manifest.json",
    }
    yielded: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.name == "results":
            iterator = (
                path
                for pattern in ("c067_cmg2*/**/*.json", "c063_trace_weighted_mask_grammar_plan_20260502_codex/**/*.json")
                for path in root.glob(pattern)
            )
        else:
            iterator = root.rglob("*.json")
        for path in iterator:
            if path.name not in names:
                continue
            resolved = path.resolve()
            if resolved not in yielded:
                yielded.add(resolved)
                yield resolved


def _find_decoded_paths(obj: Any) -> Iterator[str]:
    if isinstance(obj, dict):
        decoded = obj.get("decoded_mask_array")
        if isinstance(decoded, dict) and isinstance(decoded.get("path"), str):
            yield decoded["path"]
        mask_inputs = obj.get("mask_inputs")
        if isinstance(mask_inputs, dict):
            nested = mask_inputs.get("decoded_mask_array")
            if isinstance(nested, dict) and isinstance(nested.get("path"), str):
                yield nested["path"]
        for key, value in obj.items():
            if key in {"decoded_mask_array_path", "decoded_mask_path"} and isinstance(value, str):
                yield value
            else:
                yield from _find_decoded_paths(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _find_decoded_paths(value)


def discover_decoded_mask_array(repo_root: Path = REPO_ROOT) -> DiscoveryResult | None:
    direct = repo_root / "experiments" / "results" / "c063_trace_weighted_mask_grammar_plan_20260502_codex" / "decoded_mask_array.npy"
    if direct.exists():
        return DiscoveryResult(path=direct, reason="canonical C-063/C-067 decoded_mask_array.npy")

    candidates: list[DiscoveryResult] = []
    for json_path in _iter_json_candidate_files(repo_root):
        try:
            payload = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for raw_path in _find_decoded_paths(payload):
            path = Path(raw_path)
            if not path.is_absolute():
                path = (repo_root / path).resolve()
            if path.exists() and path.suffix == ".npy":
                candidates.append(DiscoveryResult(path=path, reason=f"referenced by {json_path.relative_to(repo_root)}"))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (str(item.path), item.reason))
    return candidates[0]


def _baseline_from_stream(path: Path) -> BaselineRecord:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"baseline mask stream does not exist: {resolved}")
    return BaselineRecord(
        role="charged_current_mask_stream",
        bytes=resolved.stat().st_size,
        source="--baseline-mask-stream",
        path=str(resolved),
        sha256=_sha256_file(resolved),
    )


def discover_baseline(repo_root: Path = REPO_ROOT) -> BaselineRecord | None:
    probe_manifest = (
        repo_root
        / "experiments"
        / "results"
        / "c063_trace_weighted_mask_grammar_plan_20260502_codex"
        / "cmg2_lossless_probe_charged_pr67_20260502T0950Z"
        / "cmg2_mask_codec_probe_manifest.json"
    )
    if probe_manifest.exists():
        try:
            payload = json.loads(probe_manifest.read_text())
            baseline = payload.get("baseline")
            if isinstance(baseline, dict) and isinstance(baseline.get("bytes"), int):
                return BaselineRecord(
                    role=str(baseline.get("role", "charged_current_mask_stream")),
                    bytes=int(baseline["bytes"]),
                    source=str(probe_manifest.relative_to(repo_root)),
                    path=baseline.get("path") if isinstance(baseline.get("path"), str) else None,
                    sha256=baseline.get("sha256") if isinstance(baseline.get("sha256"), str) else None,
                )
        except (OSError, json.JSONDecodeError):
            pass
    extracted = (
        repo_root
        / "experiments"
        / "results"
        / "c063_trace_weighted_mask_grammar_plan_20260502_codex"
        / "extracted_mask_stream.bin"
    )
    if extracted.exists():
        return BaselineRecord(
            role="decoded_av1_mask_stream_probe_reference",
            bytes=extracted.stat().st_size,
            source=str(extracted.relative_to(repo_root)),
            path=str(extracted.resolve()),
            sha256=_sha256_file(extracted),
        )
    return None


def _baseline_record(config: ProbeConfig) -> dict[str, Any] | None:
    if config.baseline_mask_stream is not None:
        record = _baseline_from_stream(config.baseline_mask_stream)
        if config.baseline_bytes is not None and int(config.baseline_bytes) != record.bytes:
            raise ValueError(
                f"baseline byte mismatch: --baseline-bytes={config.baseline_bytes} but "
                f"{config.baseline_mask_stream} is {record.bytes} bytes"
            )
        return dataclasses.asdict(record)
    if config.baseline_bytes is not None:
        return {
            "role": "charged_current_mask_stream",
            "bytes": int(config.baseline_bytes),
            "source": "--baseline-bytes",
        }
    discovered = discover_baseline()
    return None if discovered is None else dataclasses.asdict(discovered)


def run_probe(config: ProbeConfig, *, command: list[str]) -> dict[str, Any]:
    output_dir = config.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not config.force:
        raise FileExistsError(f"output directory is non-empty; pass --force to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    compressors = _available_compressors(config.compressors)
    mask_path = config.decoded_mask_array.resolve()
    arr = _load_masks(mask_path)
    baseline = _baseline_record(config)
    baseline_bytes = int(baseline["bytes"]) if baseline is not None and baseline.get("bytes") is not None else None
    candidates = build_candidate_payloads(arr)
    candidate_records = [_candidate_record(candidate, compressors, baseline_bytes) for candidate in candidates]
    candidate_records.sort(
        key=lambda item: (
            int(item["best_compression"]["compressed_size_bytes"]),
            str(item["candidate_id"]),
        )
    )

    report = {
        "schema": SCHEMA,
        "tool": "experiments/probe_predictive_mask_grammar.py",
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "local_probe_only": True,
        "cloud_jobs_dispatched": False,
        "scorer_network_loaded": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "score_claim_warning": (
            "No score claim is made by this probe. Predictive/lossy byte screens "
            "are planning evidence only; exact CUDA auth eval on exact archive "
            "bytes is the only score truth."
        ),
        "command": list(command),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "source": {
            "decoded_mask_array_path": str(mask_path),
            "decoded_mask_array_npy_sha256": _sha256_file(mask_path),
            "decoded_mask_tensor_sha256": _sha256_bytes(arr.tobytes(order="C")),
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
            "class_min": int(arr.min()),
            "class_max": int(arr.max()),
            "symbol_bits": _symbol_bits(arr),
            "raw_u8_bytes": int(arr.nbytes),
            "pixel_count": int(arr.size),
        },
        "baseline": baseline,
        "probe_config": {
            "compressors": list(compressors),
            "candidate_count": len(candidate_records),
        },
        "candidate_table": candidate_records,
        "best_candidate_by_compressed_size": candidate_records[0] if candidate_records else None,
        "required_next_steps_for_score_claim": [
            "choose a candidate whose byte screen leaves room for decoder and archive wrapper bytes",
            "define a reviewed deterministic archive member envelope and inflate decoder",
            "prove runtime consumption with local unpack/inflate smoke and validator allowlist parity",
            "run exact CUDA auth eval on the exact archive bytes before ranking or promotion",
        ],
        "artifacts": {
            "manifest": {
                "path": str(output_dir / REPORT_NAME),
            }
        },
    }
    (output_dir / REPORT_NAME).write_text(_canonical_json(report))
    return report


def _csv(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one comma-separated value")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-mask-array", type=Path)
    parser.add_argument("--baseline-mask-stream", type=Path)
    parser.add_argument("--baseline-bytes", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compressors", type=_csv, default=DEFAULT_COMPRESSORS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    discovery: DiscoveryResult | None = None
    decoded_mask_array = args.decoded_mask_array
    if decoded_mask_array is None:
        discovery = discover_decoded_mask_array()
        if discovery is None:
            raise SystemExit(
                "could not discover decoded mask array; pass --decoded-mask-array explicitly"
            )
        decoded_mask_array = discovery.path
    config = ProbeConfig(
        decoded_mask_array=decoded_mask_array,
        baseline_mask_stream=args.baseline_mask_stream,
        baseline_bytes=args.baseline_bytes,
        output_dir=args.output_dir,
        compressors=tuple(args.compressors),
        force=bool(args.force),
    )
    report = run_probe(
        config,
        command=[Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])],
    )
    best = report["best_candidate_by_compressed_size"]
    print(
        json.dumps(
            {
                "manifest": report["artifacts"]["manifest"]["path"],
                "decoded_mask_array": str(decoded_mask_array.resolve()),
                "discovery": None if discovery is None else discovery.reason,
                "best_candidate_id": None if best is None else best["candidate_id"],
                "best_compressor": None if best is None else best["best_compression"]["compressor"],
                "best_bytes": None if best is None else best["best_compression"]["compressed_size_bytes"],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
