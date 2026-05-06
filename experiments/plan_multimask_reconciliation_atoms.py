#!/usr/bin/env python3
"""Plan non-promotable multimask reconciliation atoms.

This deterministic planner compares a source decoded-mask tensor with one or
more candidate decoded-mask tensors, then ranks fusion hypotheses by a charged
byte proxy plus reconstruction disagreement.  It does not build a contest
archive, load scorer networks, run CUDA, or make score evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "multimask_reconciliation_atom_plan_v1"
TOOL = "experiments/plan_multimask_reconciliation_atoms.py"
EVIDENCE_GRADE = "empirical"
NO_OP_RANK_PENALTY = 1_000_000_000_000
ORIGINAL_VIDEO_BYTES = 37_545_489
LAMBDA_RATE = 25.0 / ORIGINAL_VIDEO_BYTES
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
NON_PROMOTABLE_WARNING = (
    "No score claim is made by this multimask reconciliation planner. It emits "
    "byte/disagreement atom hypotheses only; a deterministic archive builder "
    "and exact CUDA auth eval of the exact archive bytes are required before "
    "promotion, ranking, or score claims."
)
MANIFEST_ARRAY_KEYS = (
    "decoded_mask_array",
    "decoded_masks",
    "mask_array",
    "mask_array_path",
    "source_mask_array",
    "candidate_mask_array",
    "array_path",
    "path",
    "file",
)
MANIFEST_ARRAY_ROLES = {
    "decoded_mask_array",
    "decoded_masks",
    "mask_array",
    "source_mask_array",
    "candidate_mask_array",
}


class PlannerError(ValueError):
    """Raised for unsupported multimask planning inputs."""


@dataclass(frozen=True)
class MaskInput:
    family_name: str
    path: Path
    array: np.ndarray
    source_manifest: Path | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlannerError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise PlannerError(f"{path} must contain a JSON object")
    return payload


def _candidate_paths(raw: str, *, base_dir: Path) -> tuple[Path, ...]:
    path = Path(raw)
    if path.is_absolute():
        return (path,)
    return (base_dir / path, REPO_ROOT / path, Path.cwd() / path)


def _resolve_existing_path(raw: Any, *, base_dir: Path) -> Path | None:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        return None
    for candidate in _candidate_paths(raw, base_dir=base_dir):
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


def _iter_manifest_path_values(payload: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(payload, dict):
        role = payload.get("role")
        if isinstance(role, str) and role in MANIFEST_ARRAY_ROLES:
            for key in MANIFEST_ARRAY_KEYS:
                if key in payload:
                    yield key, payload[key]
        for key, value in payload.items():
            if key in MANIFEST_ARRAY_KEYS:
                yield key, value
            yield from _iter_manifest_path_values(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_manifest_path_values(item)


def _array_path_from_manifest(path: Path) -> Path:
    manifest = _read_json(path)
    base_dir = path.resolve().parent
    matches: list[Path] = []
    for _, raw in _iter_manifest_path_values(manifest):
        resolved = _resolve_existing_path(raw, base_dir=base_dir)
        if resolved is not None and resolved.suffix.lower() in {".npy", ".npz"}:
            matches.append(resolved)
    deduped = sorted({str(item): item for item in matches}.values(), key=lambda item: str(item))
    if not deduped:
        raise PlannerError(f"{path} does not reference an existing .npy/.npz decoded mask array")
    if len(deduped) > 1:
        raise PlannerError(
            f"{path} references multiple decoded mask arrays; pass --source-mask-array/"
            "--candidate-mask-array explicitly"
        )
    return deduped[0]


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        try:
            keys = list(archive.files)
            preferred = [key for key in ("masks", "mask", "decoded_masks", "array") if key in keys]
            if len(preferred) == 1:
                array = archive[preferred[0]]
            elif len(keys) == 1:
                array = archive[keys[0]]
            else:
                raise PlannerError(f"{path} must contain one array or a masks/decoded_masks array")
        finally:
            archive.close()
    else:
        raise PlannerError(f"{path} must be a .npy or .npz decoded mask array")
    return _normalize_mask_array(np.asarray(array), path=path)


def _normalize_mask_array(array: np.ndarray, *, path: Path) -> np.ndarray:
    if array.ndim < 2:
        raise PlannerError(f"{path} must have at least 2 dimensions; got shape {array.shape}")
    if array.size == 0:
        raise PlannerError(f"{path} must not be empty")
    if np.issubdtype(array.dtype, np.bool_):
        return np.ascontiguousarray(array.astype(np.uint8, copy=False))
    if np.issubdtype(array.dtype, np.integer):
        return np.ascontiguousarray(array)
    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise PlannerError(f"{path} contains non-finite values")
        rounded = np.rint(array)
        if not np.array_equal(array, rounded):
            raise PlannerError(f"{path} must contain integer-valued mask labels")
        return np.ascontiguousarray(rounded.astype(np.int64))
    raise PlannerError(f"{path} must contain bool, integer, or integer-valued float masks")


def _safe_family_name(raw: str, *, fallback: str) -> str:
    name = (raw or fallback).strip()
    if not name:
        name = fallback
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in name)
    return safe[:96] or fallback


def _load_input(
    *,
    family_name: str,
    array_path: Path | None,
    manifest_path: Path | None = None,
) -> MaskInput:
    if array_path is None and manifest_path is None:
        raise PlannerError("mask input requires an array path or manifest path")
    selected_path = array_path.resolve() if array_path is not None else _array_path_from_manifest(manifest_path.resolve())  # type: ignore[union-attr]
    if not selected_path.exists():
        raise FileNotFoundError(f"mask array does not exist: {selected_path}")
    return MaskInput(
        family_name=_safe_family_name(family_name, fallback=selected_path.stem),
        path=selected_path,
        array=_load_array(selected_path),
        source_manifest=manifest_path.resolve() if manifest_path is not None else None,
    )


def _input_record(item: MaskInput) -> dict[str, Any]:
    return {
        "family_name": item.family_name,
        "path": str(item.path),
        "size_bytes": int(item.path.stat().st_size),
        "sha256": _sha256_file(item.path),
        "array_shape": [int(v) for v in item.array.shape],
        "array_dtype": str(item.array.dtype),
        "array_sha256": _array_sha256(item.array),
        "source_manifest": str(item.source_manifest) if item.source_manifest else None,
    }


def _validate_inputs(source: MaskInput, candidates: list[MaskInput]) -> None:
    if not candidates:
        raise PlannerError("at least one candidate mask array is required")
    names = [source.family_name] + [item.family_name for item in candidates]
    if len(names) != len(set(names)):
        raise PlannerError(f"family names must be unique; got {names!r}")
    shape = source.array.shape
    for candidate in candidates:
        if candidate.array.shape != shape:
            raise PlannerError(
                f"shape mismatch for {candidate.family_name}: source shape {shape}, "
                f"candidate shape {candidate.array.shape}"
            )


def _value_byte_width(arrays: list[np.ndarray]) -> int:
    min_value = min(int(np.min(array)) for array in arrays)
    max_value = max(int(np.max(array)) for array in arrays)
    if min_value >= 0 and max_value <= 255:
        return 1
    if min_value >= -32768 and max_value <= 32767:
        return 2
    if min_value >= -2147483648 and max_value <= 2147483647:
        return 4
    return 8


def _index_byte_width(element_count: int) -> int:
    if element_count <= 1:
        return 1
    return max(1, math.ceil(math.ceil(math.log2(element_count)) / 8))


def _class_histogram(array: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(array, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def _diff_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    diff = a != b
    count = int(np.count_nonzero(diff))
    total = int(diff.size)
    return {
        "disagreement_count": count,
        "disagreement_fraction": round(count / total, 12),
        "agreement_fraction": round(1.0 - (count / total), 12),
    }


def _mean_pairwise_disagreement(arrays: list[np.ndarray]) -> dict[str, Any]:
    if len(arrays) < 2:
        return {
            "pair_count": 0,
            "mean_disagreement_fraction": 0.0,
            "max_disagreement_fraction": 0.0,
        }
    fractions: list[float] = []
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            fractions.append(float(np.count_nonzero(arrays[i] != arrays[j]) / arrays[i].size))
    return {
        "pair_count": len(fractions),
        "mean_disagreement_fraction": round(sum(fractions) / len(fractions), 12),
        "max_disagreement_fraction": round(max(fractions), 12),
    }


def _label_values(arrays: list[np.ndarray]) -> np.ndarray:
    values = np.unique(np.concatenate([np.unique(array).reshape(-1) for array in arrays]))
    if values.size == 0:
        raise PlannerError("mask arrays contain no labels")
    if values.size > 4096:
        raise PlannerError(
            f"multimask vectorized voting supports at most 4096 distinct labels; got {values.size}"
        )
    return values


def _chunk_slices(element_count: int, *, chunk_elements: int = 4_194_304) -> Iterable[slice]:
    for start in range(0, element_count, chunk_elements):
        yield slice(start, min(start + chunk_elements, element_count))


def _majority_vote(arrays: list[np.ndarray], *, source: np.ndarray) -> np.ndarray:
    values = _label_values(arrays)
    flat_arrays = [np.ravel(array) for array in arrays]
    source_flat = source.reshape(-1)
    fused = np.empty(source_flat.shape[0], dtype=np.result_type(*[array.dtype for array in arrays]))
    column_indices: np.ndarray | None = None
    value_count = int(values.size)
    for chunk in _chunk_slices(source_flat.shape[0]):
        chunk_len = int(chunk.stop - chunk.start)
        if column_indices is None or column_indices.shape[0] < chunk_len:
            column_indices = np.arange(chunk_len, dtype=np.int64)
        cols = column_indices[:chunk_len]
        counts = np.zeros((value_count, chunk_len), dtype=np.uint16)
        for flat in flat_arrays:
            positions = np.searchsorted(values, flat[chunk])
            counts[positions, cols] += 1
        winner_positions = np.argmax(counts, axis=0)
        max_counts = counts[winner_positions, cols]
        source_positions = np.searchsorted(values, source_flat[chunk])
        preserve_source = counts[source_positions, cols] == max_counts
        fused_chunk = values[winner_positions].astype(fused.dtype, copy=False)
        fused_chunk = np.where(preserve_source, source_flat[chunk], fused_chunk)
        fused[chunk] = fused_chunk
    return fused.reshape(source.shape)


def _candidate_consensus(candidates: list[np.ndarray], *, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    values = _label_values(candidates)
    flat_arrays = [np.ravel(array) for array in candidates]
    element_count = int(flat_arrays[0].shape[0])
    fused = np.empty(element_count, dtype=np.result_type(*[array.dtype for array in candidates]))
    consensus = np.zeros(element_count, dtype=bool)
    column_indices: np.ndarray | None = None
    value_count = int(values.size)
    for chunk in _chunk_slices(element_count):
        chunk_len = int(chunk.stop - chunk.start)
        if column_indices is None or column_indices.shape[0] < chunk_len:
            column_indices = np.arange(chunk_len, dtype=np.int64)
        cols = column_indices[:chunk_len]
        counts = np.zeros((value_count, chunk_len), dtype=np.uint16)
        for flat in flat_arrays:
            positions = np.searchsorted(values, flat[chunk])
            counts[positions, cols] += 1
        winner_positions = np.argmax(counts, axis=0)
        best_counts = counts[winner_positions, cols]
        fused[chunk] = values[winner_positions].astype(fused.dtype, copy=False)
        consensus[chunk] = (best_counts.astype(np.float64) / float(len(candidates))) >= threshold
    return fused.reshape(candidates[0].shape), consensus.reshape(candidates[0].shape)


def _estimated_charged_bytes(
    fused: np.ndarray,
    *,
    source: np.ndarray,
    policy_overhead_bytes: int,
    index_bytes: int,
    value_bytes: int,
) -> int:
    changed = int(np.count_nonzero(fused != source))
    return int(policy_overhead_bytes + changed * (index_bytes + value_bytes))


def _row_run_segments(
    *,
    changed_row: np.ndarray,
    fused_row: np.ndarray,
) -> list[tuple[int, int, int]]:
    changed_x = np.flatnonzero(changed_row)
    if changed_x.size == 0:
        return []
    values = fused_row[changed_x].astype(np.int64, copy=False)
    split = np.flatnonzero((np.diff(changed_x) != 1) | (np.diff(values) != 0)) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), split))
    stops = np.concatenate((split, np.array([changed_x.size], dtype=np.int64)))
    return [
        (int(changed_x[start]), int(changed_x[stop - 1]) + 1, int(values[start]))
        for start, stop in zip(starts, stops, strict=True)
    ]


def _entropy_stream_summary(name: str, tokens: list[int]) -> dict[str, Any]:
    if not tokens:
        return {
            "name": name,
            "token_count": 0,
            "unique_count": 0,
            "entropy_bits_per_token": 0.0,
            "entropy_bits": 0.0,
            "min_symbol": None,
            "max_symbol": None,
        }
    values, counts = np.unique(np.asarray(tokens, dtype=np.int64), return_counts=True)
    probabilities = counts.astype(np.float64) / float(len(tokens))
    entropy_per_token = float(-np.sum(probabilities * np.log2(probabilities)))
    entropy_bits = entropy_per_token * float(len(tokens))
    return {
        "name": name,
        "token_count": int(len(tokens)),
        "unique_count": int(values.size),
        "entropy_bits_per_token": round(entropy_per_token, 12),
        "entropy_bits": round(entropy_bits, 6),
        "min_symbol": int(values[0]),
        "max_symbol": int(values[-1]),
    }


def _arithmetic_lower_bound_proxy(
    *,
    policy_overhead_bytes: int,
    changed_elements: int,
    touched_rows: int,
    run_count: int,
    row_delta_tokens: list[int],
    runs_per_row_tokens: list[int],
    x_gap_tokens: list[int],
    length_tokens: list[int],
    value_tokens: list[int],
) -> dict[str, Any]:
    streams = [
        _entropy_stream_summary("row_delta", row_delta_tokens),
        _entropy_stream_summary("runs_per_row", runs_per_row_tokens),
        _entropy_stream_summary("x_gap", x_gap_tokens),
        _entropy_stream_summary("length", length_tokens),
        _entropy_stream_summary("value", value_tokens),
    ]
    entropy_bits = float(sum(float(stream["entropy_bits"]) for stream in streams))
    model_overhead_bytes = 64 + 2 * sum(int(stream["unique_count"]) for stream in streams)
    entropy_floor_bytes = int(policy_overhead_bytes + math.ceil(entropy_bits / 8.0))
    estimated_bytes = int(entropy_floor_bytes + model_overhead_bytes)
    return {
        "kind": "ideal_adaptive_arithmetic_row_run_lower_bound_proxy",
        "planning_lower_bound_only": True,
        "dispatchable_without_coder": False,
        "estimated_charged_bytes": estimated_bytes,
        "entropy_floor_bytes_no_model_overhead": entropy_floor_bytes,
        "changed_elements_vs_source": int(changed_elements),
        "touched_row_count": int(touched_rows),
        "run_count": int(run_count),
        "entropy_bits": round(entropy_bits, 6),
        "model_overhead_bytes": int(model_overhead_bytes),
        "stream_summaries": streams,
        "formula": (
            "policy_overhead + adaptive_model_overhead + ceil(sum(stream_zero_order_entropy_bits)/8); "
            "lower bound only until a deterministic archive-local arithmetic coder exists"
        ),
    }


def _row_run_residual_proxy(
    fused: np.ndarray,
    *,
    source: np.ndarray,
    policy_overhead_bytes: int,
    value_bytes: int,
) -> dict[str, Any]:
    if fused.shape != source.shape:
        raise PlannerError(f"source/fused shape mismatch: {source.shape} != {fused.shape}")
    if source.ndim < 2:
        raise PlannerError("row-run proxy requires at least 2D masks")
    frame_shape = source.shape[:-2]
    height = int(source.shape[-2])
    width = int(source.shape[-1])
    frame_count = int(np.prod(frame_shape, dtype=np.int64)) if frame_shape else 1
    source_3d = source.reshape(frame_count, height, width)
    fused_3d = fused.reshape(frame_count, height, width)
    changed = fused_3d != source_3d
    row_index_bytes = _index_byte_width(max(1, frame_count * height))
    x_coord_bytes = _index_byte_width(max(1, width))
    run_record_bytes = 2 * x_coord_bytes + value_bytes
    row_run_count_bytes = _index_byte_width(max(1, width))
    touched_rows = 0
    run_count = 0
    changed_elements = int(np.count_nonzero(changed))
    previous_touched_row = 0
    row_delta_tokens: list[int] = []
    runs_per_row_tokens: list[int] = []
    x_gap_tokens: list[int] = []
    length_tokens: list[int] = []
    value_tokens: list[int] = []
    for frame_index in range(frame_count):
        for y in range(height):
            segments = _row_run_segments(
                changed_row=changed[frame_index, y],
                fused_row=fused_3d[frame_index, y],
            )
            if not segments:
                continue
            flat_row = frame_index * height + y
            row_delta_tokens.append(int(flat_row - previous_touched_row))
            previous_touched_row = int(flat_row)
            runs_per_row_tokens.append(len(segments))
            touched_rows += 1
            run_count += len(segments)
            prev_end = 0
            for x0, x1, value in segments:
                x_gap_tokens.append(int(x0 - prev_end))
                length_tokens.append(int(x1 - x0))
                value_tokens.append(int(value))
                prev_end = int(x1)
    estimated = (
        int(policy_overhead_bytes)
        + touched_rows * (row_index_bytes + row_run_count_bytes)
        + run_count * run_record_bytes
    )
    arithmetic_proxy = _arithmetic_lower_bound_proxy(
        policy_overhead_bytes=policy_overhead_bytes,
        changed_elements=changed_elements,
        touched_rows=touched_rows,
        run_count=run_count,
        row_delta_tokens=row_delta_tokens,
        runs_per_row_tokens=runs_per_row_tokens,
        x_gap_tokens=x_gap_tokens,
        length_tokens=length_tokens,
        value_tokens=value_tokens,
    )
    return {
        "kind": "compact_row_run_residual_over_source_proxy",
        "estimated_charged_bytes": int(estimated),
        "changed_elements_vs_source": changed_elements,
        "touched_row_count": int(touched_rows),
        "run_count": int(run_count),
        "row_index_bytes": int(row_index_bytes),
        "row_run_count_bytes": int(row_run_count_bytes),
        "x_coord_bytes": int(x_coord_bytes),
        "value_bytes": int(value_bytes),
        "run_record_bytes": int(run_record_bytes),
        "arithmetic_lower_bound_proxy": arithmetic_proxy,
        "formula": (
            "policy_overhead + touched_rows*(row_index_bytes+row_run_count_bytes) "
            "+ run_count*(2*x_coord_bytes+value_bytes)"
        ),
    }


def _policy_record(
    *,
    policy_id: str,
    policy_family: str,
    policy: dict[str, Any],
    fused: np.ndarray,
    source: MaskInput,
    candidates: list[MaskInput],
    policy_overhead_bytes: int,
    index_bytes: int,
    value_bytes: int,
    disagreement_byte_equivalent: float,
) -> dict[str, Any]:
    sparse_charged_bytes = _estimated_charged_bytes(
        fused,
        source=source.array,
        policy_overhead_bytes=policy_overhead_bytes,
        index_bytes=index_bytes,
        value_bytes=value_bytes,
    )
    row_run_proxy = _row_run_residual_proxy(
        fused,
        source=source.array,
        policy_overhead_bytes=policy_overhead_bytes,
        value_bytes=value_bytes,
    )
    row_run_charged_bytes = int(row_run_proxy["estimated_charged_bytes"])
    arithmetic_proxy = row_run_proxy["arithmetic_lower_bound_proxy"]
    arithmetic_charged_bytes = int(arithmetic_proxy["estimated_charged_bytes"])
    if row_run_charged_bytes < sparse_charged_bytes:
        charged_bytes = row_run_charged_bytes
        selected_cost_model = dict(row_run_proxy)
        selected_cost_model["selected"] = True
    else:
        charged_bytes = sparse_charged_bytes
        selected_cost_model = {
            "kind": "sparse_residual_over_source_proxy",
            "policy_overhead_bytes": int(policy_overhead_bytes),
            "index_bytes_per_changed_element": int(index_bytes),
            "value_bytes_per_changed_element": int(value_bytes),
            "changed_elements_vs_source": int(np.count_nonzero(fused != source.array)),
            "estimated_charged_bytes": int(sparse_charged_bytes),
            "selected": True,
        }
    arithmetic_estimated_rate_cost = float(LAMBDA_RATE) * float(arithmetic_charged_bytes)
    arithmetic_entropy_floor_rate_cost = float(LAMBDA_RATE) * float(
        arithmetic_proxy["entropy_floor_bytes_no_model_overhead"]
    )
    source_metrics = _diff_metrics(fused, source.array)
    candidate_metrics = {
        candidate.family_name: _diff_metrics(fused, candidate.array)
        for candidate in candidates
    }
    disagreement_pixels = int(source_metrics["disagreement_count"])
    no_op_vs_source = disagreement_pixels == 0
    ranking_score = charged_bytes + disagreement_pixels * disagreement_byte_equivalent
    if no_op_vs_source:
        ranking_score += NO_OP_RANK_PENALTY
    run_count = selected_cost_model.get("run_count")
    touched_rows = selected_cost_model.get("touched_row_count")
    source_height = int(source.array.shape[-2]) if source.array.ndim >= 2 else 1
    source_frame_count = int(source.array.size // max(1, source_height * int(source.array.shape[-1]))) if source.array.ndim >= 2 else 1
    total_rows = max(1, source_frame_count * source_height)
    density_metrics = {
        "changed_element_fraction": source_metrics["disagreement_fraction"],
        "estimated_bytes_per_changed_element": (
            None if disagreement_pixels == 0 else round(float(charged_bytes) / float(disagreement_pixels), 12)
        ),
        "estimated_bytes_per_run": (
            None
            if not isinstance(run_count, int) or run_count <= 0
            else round(float(charged_bytes) / float(run_count), 12)
        ),
        "changed_elements_per_run": (
            None
            if not isinstance(run_count, int) or run_count <= 0
            else round(float(disagreement_pixels) / float(run_count), 12)
        ),
        "touched_row_fraction": (
            None
            if not isinstance(touched_rows, int)
            else round(float(touched_rows) / float(total_rows), 12)
        ),
        "rate_score_cost": round(float(LAMBDA_RATE) * float(charged_bytes), 12),
        "arithmetic_estimated_rate_score_cost_with_model_overhead": round(
            arithmetic_estimated_rate_cost,
            12,
        ),
        "arithmetic_entropy_floor_rate_score_cost_no_model_overhead": round(
            arithmetic_entropy_floor_rate_cost,
            12,
        ),
        "arithmetic_estimated_bytes_with_model_overhead": int(arithmetic_charged_bytes),
        "arithmetic_entropy_floor_bytes_no_model_overhead": int(
            arithmetic_proxy["entropy_floor_bytes_no_model_overhead"]
        ),
        "break_even_total_component_score_improvement_required": round(
            float(LAMBDA_RATE) * float(charged_bytes),
            12,
        ),
        "break_even_component_score_improvement_per_changed_element": (
            None
            if disagreement_pixels == 0
            else round((float(LAMBDA_RATE) * float(charged_bytes)) / float(disagreement_pixels), 15)
        ),
        "lambda_rate": LAMBDA_RATE,
        "total_source_rows": int(total_rows),
        "learnable_feedback_role": (
            "negative_no_op_guard"
            if no_op_vs_source
            else "byte_distortion_density_observation"
        ),
    }
    return {
        "policy_id": policy_id,
        "policy_family": policy_family,
        "candidate_family_names": [candidate.family_name for candidate in candidates],
        "estimated_charged_bytes": int(charged_bytes),
        "estimated_byte_cost_model": {
            "selected": selected_cost_model,
            "alternatives": {
                "sparse_residual_over_source_proxy": {
                    "kind": "sparse_residual_over_source_proxy",
                    "policy_overhead_bytes": int(policy_overhead_bytes),
                    "index_bytes_per_changed_element": int(index_bytes),
                    "value_bytes_per_changed_element": int(value_bytes),
                    "changed_elements_vs_source": disagreement_pixels,
                    "estimated_charged_bytes": int(sparse_charged_bytes),
                },
                "compact_row_run_residual_over_source_proxy": row_run_proxy,
                "ideal_adaptive_arithmetic_row_run_lower_bound_proxy": arithmetic_proxy,
            },
        },
        "disagreement_metrics": {
            "fused_vs_source": source_metrics,
            "fused_vs_candidate": candidate_metrics,
            "candidate_pairwise": _mean_pairwise_disagreement([candidate.array for candidate in candidates]),
        },
        "fusion_reconciliation_policy": policy,
        "fused_mask_sha256": _array_sha256(fused),
        "rank_cost_proxy": round(float(ranking_score), 12),
        "density_metrics": density_metrics,
        "dispatch_relevance": {
            "no_op_vs_source": bool(no_op_vs_source),
            "dispatchable_byte_model": bool(not no_op_vs_source and charged_bytes > 0),
            "rank_penalty_applied": int(NO_OP_RANK_PENALTY if no_op_vs_source else 0),
            "reason": (
                "no source change; cannot reduce archive bytes or score by itself"
                if no_op_vs_source
                else "changes source under empirical byte proxy; still requires deterministic archive builder"
            ),
        },
        "differentiable_feedback_contract": {
            "reward_terms": {
                "negative_rate_cost": "-lambda_rate * estimated_charged_bytes",
                "negative_disagreement_proxy": "-disagreement_byte_equivalent * changed_elements_vs_source",
                "no_op_penalty": f"-{NO_OP_RANK_PENALTY} when no_op_vs_source=true",
            },
            "smooth_surrogate_targets": {
                "selection_probability": "relax hard policy choice with softmax/Gumbel over rank_cost_proxy",
                "atom_density": "maximize expected component benefit per charged byte",
                "trust_region": "penalize source-change mass outside measured hotspot/confusion priors",
            },
            "gradient_truth_boundary": (
                "Differentiable rewards are proposal signals only; exact CUDA auth eval "
                "of a deterministic byte-closed archive is the score truth."
            ),
        },
        "score_claim": False,
        "evidence_grade": EVIDENCE_GRADE,
        "non_promotable_warning": NON_PROMOTABLE_WARNING,
    }


def _build_policy_records(
    source: MaskInput,
    candidates: list[MaskInput],
    *,
    veto_min_agreement: float,
    disagreement_byte_equivalent: float,
) -> list[dict[str, Any]]:
    arrays = [source.array] + [candidate.array for candidate in candidates]
    index_bytes = _index_byte_width(int(source.array.size))
    value_bytes = _value_byte_width(arrays)
    records: list[dict[str, Any]] = []

    majority = _majority_vote(arrays, source=source.array)
    records.append(
        _policy_record(
            policy_id="majority_vote_all",
            policy_family="majority_vote",
            policy={
                "name": "majority_vote",
                "inputs": [source.family_name] + [candidate.family_name for candidate in candidates],
                "tie_break": "preserve_source_if_tied_else_lowest_label",
            },
            fused=majority,
            source=source,
            candidates=candidates,
            policy_overhead_bytes=24 + 4 * len(arrays),
            index_bytes=index_bytes,
            value_bytes=value_bytes,
            disagreement_byte_equivalent=disagreement_byte_equivalent,
        )
    )

    priority_fused = candidates[0].array
    records.append(
        _policy_record(
            policy_id="priority_order_" + "_then_".join(candidate.family_name for candidate in candidates),
            policy_family="priority_order",
            policy={
                "name": "priority_order",
                "priority": [candidate.family_name for candidate in candidates] + [source.family_name],
                "description": "Use the first candidate family as the proposed mask, with source as fallback metadata.",
            },
            fused=priority_fused,
            source=source,
            candidates=candidates,
            policy_overhead_bytes=20 + 4 * len(candidates),
            index_bytes=index_bytes,
            value_bytes=value_bytes,
            disagreement_byte_equivalent=disagreement_byte_equivalent,
        )
    )

    consensus_values, consensus_mask = _candidate_consensus(
        [candidate.array for candidate in candidates],
        threshold=veto_min_agreement,
    )
    veto = np.where(consensus_mask, consensus_values, source.array)
    records.append(
        _policy_record(
            policy_id=f"disagreement_gated_veto_{veto_min_agreement:.3f}".replace(".", "p"),
            policy_family="disagreement_gated_veto",
            policy={
                "name": "disagreement_gated_veto",
                "candidate_consensus_threshold": round(float(veto_min_agreement), 12),
                "veto_rule": "Preserve source values where candidate consensus is below threshold.",
            },
            fused=veto,
            source=source,
            candidates=candidates,
            policy_overhead_bytes=32 + 4 * len(candidates),
            index_bytes=index_bytes,
            value_bytes=value_bytes,
            disagreement_byte_equivalent=disagreement_byte_equivalent,
        )
    )

    for candidate in candidates:
        records.append(
            _policy_record(
                policy_id=f"cheap_residual_over_base_{source.family_name}_to_{candidate.family_name}",
                policy_family="cheap_residual_over_base",
                policy={
                    "name": "cheap_residual_over_base",
                    "base_family": source.family_name,
                    "residual_family": candidate.family_name,
                    "residual_atom": "changed element indices plus replacement labels",
                },
                fused=candidate.array,
                source=source,
                candidates=[candidate],
                policy_overhead_bytes=16,
                index_bytes=index_bytes,
                value_bytes=value_bytes,
                disagreement_byte_equivalent=disagreement_byte_equivalent,
            )
        )

    return sorted(
        records,
        key=lambda item: (
            float(item["rank_cost_proxy"]),
            int(item["estimated_charged_bytes"]),
            str(item["policy_id"]),
        ),
    )


def build_plan(
    *,
    source_mask_array: Path | None = None,
    candidate_mask_arrays: list[Path] | None = None,
    output_json: Path | None = None,
    source_manifest: Path | None = None,
    candidate_manifests: list[Path] | None = None,
    source_family: str = "source",
    candidate_families: list[str] | None = None,
    veto_min_agreement: float = 1.0,
    disagreement_byte_equivalent: float = 1.0,
    max_ranked_policies: int | None = None,
) -> dict[str, Any]:
    if not (0.0 < veto_min_agreement <= 1.0):
        raise PlannerError(f"veto_min_agreement must be in (0,1], got {veto_min_agreement}")
    if not math.isfinite(disagreement_byte_equivalent) or disagreement_byte_equivalent < 0.0:
        raise PlannerError("disagreement_byte_equivalent must be finite and nonnegative")
    if max_ranked_policies is not None and max_ranked_policies <= 0:
        raise PlannerError("max_ranked_policies must be positive when provided")

    candidate_mask_arrays = list(candidate_mask_arrays or [])
    candidate_manifests = list(candidate_manifests or [])
    candidate_families = list(candidate_families or [])
    candidate_count = len(candidate_mask_arrays) + len(candidate_manifests)
    if candidate_families and len(candidate_families) != candidate_count:
        raise PlannerError(
            f"--candidate-family count must match candidate inputs: "
            f"{len(candidate_families)} names for {candidate_count} candidates"
        )

    source = _load_input(
        family_name=source_family,
        array_path=source_mask_array,
        manifest_path=source_manifest,
    )
    candidates: list[MaskInput] = []
    family_index = 0
    for index, path in enumerate(candidate_mask_arrays):
        family = candidate_families[family_index] if candidate_families else path.stem
        candidates.append(_load_input(family_name=family, array_path=path, manifest_path=None))
        family_index += 1
    for index, path in enumerate(candidate_manifests):
        family = candidate_families[family_index] if candidate_families else f"candidate_manifest_{index}"
        candidates.append(_load_input(family_name=family, array_path=None, manifest_path=path))
        family_index += 1

    _validate_inputs(source, candidates)
    policies = _build_policy_records(
        source,
        candidates,
        veto_min_agreement=veto_min_agreement,
        disagreement_byte_equivalent=disagreement_byte_equivalent,
    )
    if max_ranked_policies is not None:
        policies = policies[:max_ranked_policies]
    for rank, policy in enumerate(policies, start=1):
        policy["rank"] = rank

    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "non_promotable_warning": NON_PROMOTABLE_WARNING,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "inputs": {
            "source": _input_record(source),
            "candidates": [_input_record(candidate) for candidate in candidates],
            "array_contract": {
                "shape": [int(v) for v in source.array.shape],
                "element_count": int(source.array.size),
                "dtype_rule": "bool/integer/integer-valued-float labels; exact elementwise reconciliation",
            },
        },
        "candidate_family_names": [candidate.family_name for candidate in candidates],
        "source_family_name": source.family_name,
        "source_class_histogram": _class_histogram(source.array),
        "candidate_vs_source": {
            candidate.family_name: _diff_metrics(candidate.array, source.array)
            for candidate in candidates
        },
        "ranking_model": {
            "rank_cost_proxy": "estimated_charged_bytes + fused_vs_source_disagreement_count * disagreement_byte_equivalent",
            "disagreement_byte_equivalent": round(float(disagreement_byte_equivalent), 12),
            "lower_is_better": True,
        },
        "candidate_policies": policies,
        "archive_builder_hooks": {
            "policy_json_contract": "candidate_policies[*].fusion_reconciliation_policy",
            "charged_residual_contract": (
                "Use estimated_byte_cost_model as a proxy only; a builder must "
                "emit deterministic charged bytes and record exact payload hashes."
            ),
            "required_next_validation": CUDA_AUTH_EVAL_PATH,
        },
    }
    if output_json is not None:
        _write_json(output_json.resolve(), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mask-array", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--candidate-mask-array", type=Path, action="append", default=[])
    parser.add_argument("--candidate-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source-family", default="source")
    parser.add_argument("--candidate-family", action="append", default=[])
    parser.add_argument("--veto-min-agreement", type=float, default=1.0)
    parser.add_argument("--disagreement-byte-equivalent", type=float, default=1.0)
    parser.add_argument("--max-ranked-policies", type=int)
    args = parser.parse_args(argv)

    if args.source_mask_array is not None and args.source_manifest is not None:
        parser.error("pass only one of --source-mask-array or --source-manifest")
    if args.source_mask_array is None and args.source_manifest is None:
        parser.error("one of --source-mask-array or --source-manifest is required")
    if not args.candidate_mask_array and not args.candidate_manifest:
        parser.error("at least one --candidate-mask-array or --candidate-manifest is required")

    try:
        build_plan(
            source_mask_array=args.source_mask_array,
            source_manifest=args.source_manifest,
            candidate_mask_arrays=args.candidate_mask_array,
            candidate_manifests=args.candidate_manifest,
            output_json=args.output_json,
            source_family=args.source_family,
            candidate_families=args.candidate_family,
            veto_min_agreement=args.veto_min_agreement,
            disagreement_byte_equivalent=args.disagreement_byte_equivalent,
            max_ranked_policies=args.max_ranked_policies,
        )
    except (FileNotFoundError, PlannerError) as exc:
        parser.exit(2, f"{TOOL}: error: {exc}\n")
    print(json.dumps({"output_json": str(args.output_json.resolve()), "score_claim": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
