#!/usr/bin/env python3
"""Plan CMG3/postfilter pixel-repair atoms from exact mask residuals.

This is a deterministic, planning-only ledger generator. It compares a source
decoded mask tensor against a reconstructed/candidate decoded mask tensor,
partitions exact residual pixels into scorer-aligned repair atoms, and ranks
those atoms by a Lagrangian byte/proxy-benefit density. It does not build an
archive, run scorers, dispatch jobs, or make score claims.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CMG3A_BUILDER_PATH = REPO_ROOT / "experiments" / "build_cmg3_adaptive_runs_candidate.py"
SCHEMA = "cmg3_pixel_lagrangian_atom_ledger_v1"
TOOL = "experiments/plan_cmg3_pixel_lagrangian_atoms.py"
EVIDENCE_GRADE = "planning_only"
ORIGINAL_VIDEO_BYTES = 37_545_489
LAMBDA_RATE = 25.0 / ORIGINAL_VIDEO_BYTES
BREAK_EVEN_BYTES_PER_SCORE = 1.0 / LAMBDA_RATE
CMG3_ROW_COUNT_BYTES = 1
CMG3_RUN_RECORD_BYTES = 5
DEFAULT_CLASS_COUNT = 5
DEFAULT_ATOM_FAMILIES = (
    "pair",
    "frame",
    "class",
    "pair_class",
    "row_run",
    "connected_row_run",
)


class PlannerError(ValueError):
    """Raised when planning inputs are invalid or unsupported."""


@dataclass(frozen=True)
class FoveationConfig:
    center_x: float
    center_y: float
    sigma: float
    strength: float
    frame_centers: tuple[tuple[float, float], ...] | None = None
    frame_center_indexing: str = "one_to_one"

    @property
    def mode(self) -> str:
        return "dynamic_per_frame" if self.frame_centers is not None else "static"

    def center_for_frame(self, frame_index: int) -> tuple[float, float]:
        if self.frame_centers is None:
            return self.center_x, self.center_y
        if self.frame_center_indexing == "pair_average_from_full_frames":
            first = 2 * frame_index
            second = first + 1
            if second >= len(self.frame_centers):
                raise PlannerError(
                    f"dynamic foveation has {len(self.frame_centers)} full-frame centers, "
                    f"but pair frame_index={frame_index} needs indices {first},{second}"
                )
            x0, y0 = self.frame_centers[first]
            x1, y1 = self.frame_centers[second]
            return (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if self.frame_center_indexing != "one_to_one":
            raise PlannerError(f"unknown dynamic foveation frame_center_indexing={self.frame_center_indexing!r}")
        if frame_index < 0 or frame_index >= len(self.frame_centers):
            raise PlannerError(
                f"dynamic foveation has {len(self.frame_centers)} frame centers, "
                f"but frame_index={frame_index} was requested"
            )
        return self.frame_centers[frame_index]

    def as_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "center_x": round(self.center_x, 12),
            "center_y": round(self.center_y, 12),
            "sigma": round(self.sigma, 12),
            "strength": round(self.strength, 12),
        }
        if self.frame_centers is not None:
            centers_bytes = _json_bytes(
                {
                    "frame_centers": [
                        [round(x, 12), round(y, 12)]
                        for x, y in self.frame_centers
                    ]
                }
            )
            payload["frame_center_count"] = len(self.frame_centers)
            payload["frame_center_indexing"] = self.frame_center_indexing
            payload["frame_centers_sha256"] = _sha256_bytes(centers_bytes)
            payload["first_frame_center"] = [
                round(self.frame_centers[0][0], 12),
                round(self.frame_centers[0][1], 12),
            ]
            payload["last_frame_center"] = [
                round(self.frame_centers[-1][0], 12),
                round(self.frame_centers[-1][1], 12),
            ]
        return payload


@dataclass(frozen=True)
class ResidualRun:
    run_index: int
    frame_index: int
    y: int
    x0: int
    x1: int
    source_class: int
    candidate_class_hist: tuple[tuple[int, int], ...]
    boundary_pixels: int
    foveal_weight_sum: float

    @property
    def pair_index(self) -> int:
        return self.frame_index // 2

    @property
    def length(self) -> int:
        return self.x1 - self.x0


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise PlannerError(f"{path} is not valid JSON") from exc


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PlannerError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlannerError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise PlannerError(f"{field} must be finite")
    return out


def _load_mask_array(path: Path, *, label: str) -> np.ndarray:
    path = path.resolve()
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "masks" in payload:
                arr = payload["masks"]
            elif len(payload.files) == 1:
                arr = payload[payload.files[0]]
            else:
                raise PlannerError(f"{label} {path} must contain a 'masks' array or one array")
    else:
        raise PlannerError(f"{label} must be .npy or .npz, got {path}")
    if arr.ndim != 3:
        raise PlannerError(f"{label} must have shape (frames,height,width), got {arr.shape}")
    if arr.size == 0:
        raise PlannerError(f"{label} is empty")
    if not np.issubdtype(arr.dtype, np.integer):
        raise PlannerError(f"{label} must contain integer class ids, got {arr.dtype}")
    if int(arr.min()) < 0:
        raise PlannerError(f"{label} contains negative class ids: min={int(arr.min())}")
    return np.ascontiguousarray(arr.astype(np.int16, copy=False))


def _hist(values: np.ndarray) -> tuple[tuple[int, int], ...]:
    counts = Counter(int(v) for v in values.tolist())
    return tuple(sorted((cls, count) for cls, count in counts.items()))


def _dict_hist(counter: Counter[int]) -> dict[str, int]:
    return {str(k): int(counter[k]) for k in sorted(counter)}


def _atom_id(atom_family: str, identity: dict[str, Any]) -> str:
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"cmg3pix_{atom_family}_{hashlib.sha256(raw).hexdigest()[:16]}"


def _parse_family_list(value: str | Iterable[str]) -> tuple[str, ...]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    families = tuple(item.strip() for item in raw if item and item.strip())
    allowed = set(DEFAULT_ATOM_FAMILIES) | {"frame_class"}
    unknown = [item for item in families if item not in allowed]
    if unknown:
        raise PlannerError(f"unsupported atom families: {unknown}; allowed={sorted(allowed)}")
    if not families:
        raise PlannerError("at least one atom family is required")
    return families


def _candidate_path_from_manifest(payload: dict[str, Any], *, manifest_dir: Path) -> Path | None:
    candidate_keys = (
        ("candidate_decoded_mask_array", "path"),
        ("candidate_mask_array", "path"),
        ("reconstructed_mask_array", "path"),
        ("reconstructed_decoded_mask_array", "path"),
        ("cmg3", "reconstructed_mask_array", "path"),
        ("cmg3", "candidate_mask_array", "path"),
    )
    for key_path in candidate_keys:
        current: Any = payload
        for key in key_path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, str):
            path = Path(current)
            return path if path.is_absolute() else (manifest_dir / path)
    return None


def _row_nonzero_runs(row: np.ndarray) -> list[tuple[int, int, int, int]]:
    runs: list[tuple[int, int, int, int]] = []
    if row.size == 0:
        return runs
    current = int(row[0])
    start = 0
    for x, raw in enumerate(row[1:], start=1):
        value = int(raw)
        if value == current:
            continue
        if current != 0:
            runs.append((start, x, current, x - start))
        current = value
        start = x
    if current != 0:
        runs.append((start, int(row.shape[0]), current, int(row.shape[0]) - start))
    return runs


def _reconstruct_nonzero_topk(source: np.ndarray, *, max_runs_per_row: int) -> np.ndarray:
    if not (0 <= max_runs_per_row <= 255):
        raise PlannerError(f"CMG3 max_runs_per_row must be in [0,255], got {max_runs_per_row}")
    recon = np.zeros_like(source)
    for frame_index, frame in enumerate(source):
        for y, row in enumerate(frame):
            selected = sorted(
                sorted(_row_nonzero_runs(row), key=lambda item: (-item[3], item[0]))[
                    :max_runs_per_row
                ],
                key=lambda item: item[0],
            )
            for x0, x1, class_id, _length in selected:
                recon[frame_index, y, x0:x1] = np.int16(class_id)
    return recon


def _row_spans(source: np.ndarray, *, row_stride: int, class_count: int) -> np.ndarray:
    if row_stride <= 0 or row_stride > source.shape[1]:
        raise PlannerError(f"row_stride must be in [1,{source.shape[1]}], got {row_stride}")
    frames, height, _width = source.shape
    rows = np.arange(0, height, row_stride, dtype=np.int32)
    spans = np.full((frames, class_count, len(rows), 2), -1, dtype=np.int32)
    for cls in range(class_count):
        for row_index, y in enumerate(rows.tolist()):
            present = source[:, y, :] == np.int16(cls)
            any_present = present.any(axis=1)
            first = present.argmax(axis=1)
            last = width - 1 - present[:, ::-1].argmax(axis=1)
            spans[any_present, cls, row_index, 0] = first[any_present].astype(np.int32)
            spans[any_present, cls, row_index, 1] = last[any_present].astype(np.int32)
    return spans


def _reconstruct_row_spans(
    source: np.ndarray,
    *,
    row_stride: int,
    default_class: int,
    row_fill: str,
    draw_order: tuple[int, ...],
    class_count: int,
) -> np.ndarray:
    spans = _row_spans(source, row_stride=row_stride, class_count=class_count)
    frames, _classes, sampled_rows, _endpoints = spans.shape
    height, width = source.shape[1:]
    sampled = np.full((frames, sampled_rows, width), default_class, dtype=np.int16)
    for cls in draw_order:
        class_spans = spans[:, int(cls), :, :]
        valid = (class_spans[..., 0] >= 0) & (class_spans[..., 1] >= class_spans[..., 0])
        for row_index in range(sampled_rows):
            for frame_index in np.flatnonzero(valid[:, row_index]).tolist():
                x0 = int(class_spans[frame_index, row_index, 0])
                x1 = int(class_spans[frame_index, row_index, 1]) + 1
                sampled[frame_index, row_index, x0:x1] = np.int16(cls)
    rows = np.arange(height, dtype=np.int32)
    if row_fill == "nearest":
        row_indices = np.minimum((rows + row_stride // 2) // row_stride, sampled_rows - 1)
    elif row_fill == "forward":
        row_indices = np.minimum(rows // row_stride, sampled_rows - 1)
    else:
        raise PlannerError(f"unsupported CMG3 row_fill policy in manifest: {row_fill!r}")
    return np.ascontiguousarray(sampled[:, row_indices, :])


def _reconstruct_cmg3a_adaptive_from_manifest(
    source: np.ndarray,
    payload: dict[str, Any],
    *,
    manifest_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    if source.dtype != np.int16:
        source_u8 = np.ascontiguousarray(source.astype(np.uint8))
    else:
        source_u8 = np.ascontiguousarray(source.astype(np.uint8))
    if not np.array_equal(source, source_u8.astype(source.dtype, copy=False)):
        raise PlannerError(f"{manifest_path}: source masks cannot be losslessly cast to uint8")
    policy = payload.get("policy")
    cmg3 = payload.get("cmg3")
    if not isinstance(policy, dict) or not isinstance(cmg3, dict):
        raise PlannerError(f"{manifest_path}: CMG3A manifest requires policy and cmg3 sections")
    weights = policy.get("weights")
    if not isinstance(weights, dict):
        raise PlannerError(f"{manifest_path}: CMG3A policy.weights must be an object")
    class_weights_raw = weights.get("class_weights", {})
    if not isinstance(class_weights_raw, dict):
        raise PlannerError(f"{manifest_path}: CMG3A policy.weights.class_weights must be an object")
    class_weights = {int(key): float(value) for key, value in class_weights_raw.items()}
    hard_frames = set()
    hard_frame_meta = policy.get("hard_frame_indices")
    if isinstance(hard_frame_meta, dict) and isinstance(hard_frame_meta.get("frames"), list):
        hard_frames = {int(value) for value in hard_frame_meta["frames"]}

    selected_extra = policy.get("selected_extra_runs")
    if not isinstance(selected_extra, int):
        raise PlannerError(f"{manifest_path}: CMG3A policy.selected_extra_runs must be int")
    builder = _load_module(CMG3A_BUILDER_PATH, "_cmg3_pixel_planner_cmg3a_builder")
    _run_stream, candidate_u8, _stats, _policy = builder.encode_adaptive_run_stream(
        source_u8,
        base_runs_per_row=int(policy.get("base_runs_per_row", 1)),
        target_extra_runs=selected_extra,
        target_body_bytes=None,
        adaptive_max_runs_per_row=int(policy.get("adaptive_max_runs_per_row", 8)),
        compressor="raw",
        class_weights=class_weights,
        hard_frame_indices=hard_frames,
        hard_frame_multiplier=float(weights.get("hard_frame_multiplier", 1.35)),
        foveal_row_weight=float(weights.get("foveal_row_weight", 0.20)),
        foveal_col_weight=float(weights.get("foveal_col_weight", 0.20)),
        boundary_detail_weight=float(weights.get("boundary_detail_weight", 0.08)),
        rank_decay=float(weights.get("rank_decay", 0.92)),
    )
    candidate = np.ascontiguousarray(candidate_u8.astype(np.int16, copy=False))
    actual_sha = _sha256_bytes(candidate_u8.tobytes(order="C"))
    expected_sha = cmg3.get("reconstructed_mask_u8_sha256")
    if expected_sha is None and isinstance(cmg3.get("run_stats"), dict):
        expected_sha = cmg3["run_stats"].get("reconstructed_tensor_sha256")
    if expected_sha is not None and actual_sha != expected_sha:
        raise PlannerError(
            f"{manifest_path}: reconstructed CMG3A mask SHA mismatch: "
            f"expected={expected_sha} actual={actual_sha}"
        )
    return candidate, {
        "mode": "reconstructed_from_cmg3a_adaptive_manifest",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "base_runs_per_row": int(policy.get("base_runs_per_row", 1)),
        "selected_extra_runs": int(selected_extra),
        "adaptive_max_runs_per_row": int(policy.get("adaptive_max_runs_per_row", 8)),
        "reconstructed_mask_u8_sha256": actual_sha,
        "compressed_payload_not_used": True,
    }


def _load_candidate_from_manifest(
    manifest_path: Path,
    *,
    source: np.ndarray,
    class_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict):
        raise PlannerError(f"{manifest_path} must contain a JSON object")
    manifest_dir = manifest_path.parent

    explicit_candidate = _candidate_path_from_manifest(payload, manifest_dir=manifest_dir)
    if explicit_candidate is not None:
        candidate = _load_mask_array(explicit_candidate, label="candidate mask array from manifest")
        return candidate, {
            "mode": "explicit_candidate_mask_array_path",
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "candidate_mask_array": {
                "path": str(explicit_candidate.resolve()),
                "npy_sha256": _sha256_file(explicit_candidate.resolve()),
            },
        }

    schema = payload.get("schema")
    cmg3 = payload.get("cmg3")
    if not isinstance(cmg3, dict):
        raise PlannerError(
            f"{manifest_path} does not expose a candidate mask path or a cmg3 section"
        )

    if schema == "cmg3a_adaptive_nonzero_row_runs_candidate_v1":
        return _reconstruct_cmg3a_adaptive_from_manifest(
            source,
            payload,
            manifest_path=manifest_path,
        )

    if schema == "cmg3_nonzero_row_runs_candidate_v1" or cmg3.get("mode") == "nonzero_row_runs_topk_v1":
        max_runs = int(cmg3.get("max_runs_per_row"))
        candidate = _reconstruct_nonzero_topk(source, max_runs_per_row=max_runs)
        return candidate, {
            "mode": "reconstructed_from_cmg3_nonzero_row_runs_manifest",
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "max_runs_per_row": max_runs,
            "compressed_payload_not_used": True,
        }

    if schema == "cmg3_rowspan_candidate_v1" or cmg3.get("mode") == "row_span_stride_class_predictor_v1":
        row_stride = int(cmg3.get("row_stride"))
        default_class = int(cmg3.get("default_class", 0))
        row_fill = str(cmg3.get("row_fill"))
        draw_order_raw = cmg3.get("draw_order")
        if not isinstance(draw_order_raw, list):
            raise PlannerError(f"{manifest_path}: cmg3.draw_order must be a list")
        draw_order = tuple(int(v) for v in draw_order_raw)
        candidate = _reconstruct_row_spans(
            source,
            row_stride=row_stride,
            default_class=default_class,
            row_fill=row_fill,
            draw_order=draw_order,
            class_count=max(class_count, len(draw_order)),
        )
        return candidate, {
            "mode": "reconstructed_from_cmg3_rowspan_manifest",
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "row_stride": row_stride,
            "row_fill": row_fill,
            "draw_order": list(draw_order),
            "compressed_payload_not_used": True,
        }

    raise PlannerError(
        f"{manifest_path}: unsupported CMG3 manifest schema={schema!r} mode={cmg3.get('mode')!r}; "
        "provide --candidate-mask-array instead"
    )


def _parse_hard_pair_payload(payload: Any) -> list[int]:
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        for key in (
            "hardest_pair_indices",
            "hard_pair_indices",
            "pair_indices",
            "top_pair_indices",
        ):
            if key in payload:
                values = payload[key]
                break
        else:
            raise PlannerError("hard-pair JSON object needs hardest_pair_indices or pair_indices")
    else:
        raise PlannerError("hard-pair payload must be a list or object")
    if not isinstance(values, list) or not all(isinstance(v, int) for v in values):
        raise PlannerError("hard-pair indices must be an integer list")
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value < 0:
            raise PlannerError(f"hard-pair index must be nonnegative: {value}")
        if value not in seen:
            seen.add(value)
            out.append(int(value))
    return out


def _load_hard_pairs(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = path.resolve()
    if path.suffix.lower() == ".json":
        indices = _parse_hard_pair_payload(_read_json(path))
    else:
        indices = []
        for line in path.read_text().splitlines():
            stripped = line.split("#", 1)[0].strip()
            if stripped:
                indices.append(int(stripped))
        indices = _parse_hard_pair_payload(indices)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "pair_indices": indices,
    }


def _load_component_trace(path: Path | None, *, pair_count: int) -> dict[str, Any] | None:
    if path is None:
        return None
    path = path.resolve()
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise PlannerError(f"{path} must contain a JSON object")
    if payload.get("score_claim") is True:
        raise PlannerError(f"{path} has score_claim=true; planning priors must not be score claims")

    signals = [0.0] * pair_count
    source = "unknown"
    if isinstance(payload.get("samples"), list):
        source = "samples.score_combined_contribution_first_order"
        for idx, sample in enumerate(payload["samples"]):
            if not isinstance(sample, dict):
                raise PlannerError(f"{path}: samples[{idx}] must be an object")
            pair = sample.get("pair_index")
            if not isinstance(pair, int):
                raise PlannerError(f"{path}: samples[{idx}].pair_index must be int")
            if pair < 0 or pair >= pair_count:
                continue
            signal = sample.get("score_combined_contribution_first_order")
            if signal is None:
                seg = _finite_float(sample.get("segnet_dist", 0.0), field=f"samples[{idx}].segnet_dist")
                pose = _finite_float(sample.get("posenet_dist", 0.0), field=f"samples[{idx}].posenet_dist")
                signal = 100.0 * seg + math.sqrt(max(0.0, 10.0 * pose))
                source = "samples.derived_from_component_distances"
            signals[pair] = max(0.0, _finite_float(signal, field=f"samples[{idx}].signal"))
    elif isinstance(payload.get("per_pair_weights"), list):
        source = "per_pair_weights"
        raw = payload["per_pair_weights"]
        for pair, value in enumerate(raw[:pair_count]):
            signals[pair] = max(0.0, _finite_float(value, field=f"per_pair_weights[{pair}]"))
    elif isinstance(payload.get("per_pair_combined_score_signal"), list):
        source = "per_pair_combined_score_signal"
        raw = payload["per_pair_combined_score_signal"]
        for pair, value in enumerate(raw[:pair_count]):
            signals[pair] = max(0.0, _finite_float(value, field=f"per_pair_combined_score_signal[{pair}]"))
    else:
        raise PlannerError(
            f"{path} must contain samples, per_pair_weights, or per_pair_combined_score_signal"
        )

    ranked = sorted(range(pair_count), key=lambda pair: (signals[pair], -pair), reverse=True)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "score_claim": payload.get("score_claim"),
        "evidence_grade": payload.get("evidence_grade"),
        "signal_source": source,
        "signals": signals,
        "max_signal": max(signals) if signals else 0.0,
        "top_pair_indices": [pair for pair in ranked if signals[pair] > 0.0][:32],
    }


def _load_class_weights(path: Path | None, *, class_count: int) -> tuple[list[float], dict[str, Any] | None]:
    weights = [1.0] * class_count
    if path is None:
        return weights, None
    path = path.resolve()
    payload = _read_json(path)
    if isinstance(payload, list):
        for idx, value in enumerate(payload[:class_count]):
            weights[idx] = _finite_float(value, field=f"class_weights[{idx}]")
    elif isinstance(payload, dict):
        raw = payload.get("class_weights", payload)
        if not isinstance(raw, dict):
            raise PlannerError("class-weights JSON object must map class ids to weights")
        for key, value in raw.items():
            idx = int(key)
            if 0 <= idx < class_count:
                weights[idx] = _finite_float(value, field=f"class_weights[{key}]")
    else:
        raise PlannerError("class-weights JSON must be a list or object")
    if any(weight <= 0 for weight in weights):
        raise PlannerError("class weights must be positive")
    return weights, {
        "path": str(path),
        "sha256": _sha256_file(path),
        "weights": {str(idx): round(weight, 12) for idx, weight in enumerate(weights)},
    }


def _foveation_from_payload(payload: dict[str, Any], *, width: int, height: int) -> FoveationConfig:
    frame_centers: tuple[tuple[float, float], ...] | None = None
    raw_frame_centers = payload.get("frame_centers", payload.get("centers_by_frame"))
    if raw_frame_centers is not None:
        if isinstance(raw_frame_centers, dict):
            parsed_by_index: dict[int, tuple[float, float]] = {}
            for key, value in raw_frame_centers.items():
                try:
                    frame_index = int(key)
                except (TypeError, ValueError) as exc:
                    raise PlannerError(f"dynamic foveation frame key must be an integer, got {key!r}") from exc
                if not isinstance(value, list) or len(value) != 2:
                    raise PlannerError(f"dynamic foveation center for frame {frame_index} must be [x,y]")
                parsed_by_index[frame_index] = (
                    _finite_float(value[0], field=f"foveation.frame_centers[{frame_index}].x"),
                    _finite_float(value[1], field=f"foveation.frame_centers[{frame_index}].y"),
                )
            if not parsed_by_index:
                raise PlannerError("dynamic foveation frame_centers cannot be empty")
            expected = list(range(max(parsed_by_index) + 1))
            missing = [idx for idx in expected if idx not in parsed_by_index]
            if missing:
                raise PlannerError(f"dynamic foveation centers_by_frame must be contiguous from 0; missing {missing[:8]}")
            frame_centers = tuple(parsed_by_index[idx] for idx in expected)
        elif isinstance(raw_frame_centers, list):
            if not raw_frame_centers:
                raise PlannerError("dynamic foveation frame_centers cannot be empty")
            parsed = []
            for idx, value in enumerate(raw_frame_centers):
                if not isinstance(value, list) or len(value) != 2:
                    raise PlannerError(f"dynamic foveation center for frame {idx} must be [x,y]")
                parsed.append(
                    (
                        _finite_float(value[0], field=f"foveation.frame_centers[{idx}].x"),
                        _finite_float(value[1], field=f"foveation.frame_centers[{idx}].y"),
                    )
                )
            frame_centers = tuple(parsed)
        else:
            raise PlannerError("dynamic foveation frame_centers must be a list or object")

    if "center" in payload:
        center = payload["center"]
    elif "center_xy" in payload:
        center = payload["center_xy"]
    elif "vanishing_point" in payload:
        center = payload["vanishing_point"]
    elif frame_centers is not None:
        xs = [xy[0] for xy in frame_centers]
        ys = [xy[1] for xy in frame_centers]
        center = [sum(xs) / len(xs), sum(ys) / len(ys)]
    else:
        center = [payload.get("center_x"), payload.get("center_y")]
    if not isinstance(center, list) or len(center) != 2:
        raise PlannerError("foveation center must be [x,y]")
    center_x = _finite_float(center[0], field="foveation.center_x")
    center_y = _finite_float(center[1], field="foveation.center_y")
    sigma = _finite_float(payload.get("sigma", payload.get("radius", max(width, height) / 4.0)), field="foveation.sigma")
    strength = _finite_float(payload.get("strength", 1.5), field="foveation.strength")
    if sigma <= 0.0:
        raise PlannerError("foveation sigma must be positive")
    if strength < 0.0:
        raise PlannerError("foveation strength must be nonnegative")
    return FoveationConfig(
        center_x=center_x,
        center_y=center_y,
        sigma=sigma,
        strength=strength,
        frame_centers=frame_centers,
    )


def _load_foveation_config(
    path: Path | None,
    *,
    center: tuple[float, float] | None,
    sigma: float | None,
    strength: float,
    width: int,
    height: int,
) -> tuple[FoveationConfig | None, dict[str, Any] | None]:
    if path is not None:
        path = path.resolve()
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise PlannerError("foveation JSON must contain an object")
        cfg = _foveation_from_payload(payload, width=width, height=height)
        if cfg.frame_centers is not None and len(cfg.frame_centers) != payload.get("frame_count", len(cfg.frame_centers)):
            raise PlannerError("foveation JSON frame_count does not match frame_centers length")
        return cfg, {"path": str(path), "sha256": _sha256_file(path), "config": cfg.as_json()}
    if center is None:
        return None, None
    effective_sigma = float(sigma) if sigma is not None else max(width, height) / 4.0
    cfg = FoveationConfig(
        center_x=float(center[0]),
        center_y=float(center[1]),
        sigma=effective_sigma,
        strength=float(strength),
    )
    if cfg.sigma <= 0.0:
        raise PlannerError("foveal sigma must be positive")
    if cfg.strength < 0.0:
        raise PlannerError("foveal strength must be nonnegative")
    return cfg, {"path": None, "config": cfg.as_json()}


def _boundary_pixels_for_span(
    source: np.ndarray,
    *,
    frame_index: int,
    y: int,
    x0: int,
    x1: int,
    class_id: int,
) -> int:
    width = source.shape[2]
    height = source.shape[1]
    span_len = x1 - x0
    boundary = np.zeros(span_len, dtype=bool)
    if x0 > 0:
        boundary[0] |= int(source[frame_index, y, x0 - 1]) != class_id
    if x1 < width:
        boundary[-1] |= int(source[frame_index, y, x1]) != class_id
    if x1 - x0 > 1:
        left = source[frame_index, y, x0 : x1 - 1] != class_id
        right = source[frame_index, y, x0 + 1 : x1] != class_id
        boundary[1:] |= left
        boundary[:-1] |= right
    if y > 0:
        boundary |= source[frame_index, y - 1, x0:x1] != class_id
    if y + 1 < height:
        boundary |= source[frame_index, y + 1, x0:x1] != class_id
    return int(boundary.sum())


def _foveal_weight_sum_for_span(
    *,
    frame_index: int,
    y: int,
    x0: int,
    x1: int,
    foveation: FoveationConfig | None,
) -> float:
    length = x1 - x0
    if foveation is None:
        return float(length)
    center_x, center_y = foveation.center_for_frame(frame_index)
    xs = np.arange(x0, x1, dtype=np.float64)
    d2 = (xs - center_x) ** 2 + (float(y) - center_y) ** 2
    weights = 1.0 + foveation.strength * np.exp(-d2 / (2.0 * foveation.sigma * foveation.sigma))
    return float(weights.sum())


def _row_residual_run_segments(
    *,
    diff_row: np.ndarray,
    source_row: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Return contiguous changed x-runs split by source class."""
    changed_x = np.flatnonzero(diff_row)
    if changed_x.size == 0:
        return []
    classes = source_row[changed_x].astype(np.int16, copy=False)
    split = np.flatnonzero((np.diff(changed_x) != 1) | (np.diff(classes) != 0)) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), split))
    stops = np.concatenate((split, np.array([changed_x.size], dtype=np.int64)))
    return [
        (int(changed_x[start]), int(changed_x[stop - 1]) + 1, int(classes[start]))
        for start, stop in zip(starts, stops, strict=True)
    ]


def _extract_residual_runs(
    source: np.ndarray,
    candidate: np.ndarray,
    *,
    foveation: FoveationConfig | None,
) -> list[ResidualRun]:
    if source.shape != candidate.shape:
        raise PlannerError(f"source/candidate shape mismatch: {source.shape} != {candidate.shape}")
    runs: list[ResidualRun] = []
    run_index = 0
    frames, height, width = source.shape
    for frame_index in range(frames):
        for y in range(height):
            src_row = source[frame_index, y]
            cand_row = candidate[frame_index, y]
            diff = src_row != cand_row
            for x0, x1, source_class in _row_residual_run_segments(
                diff_row=diff,
                source_row=src_row,
            ):
                boundary_pixels = _boundary_pixels_for_span(
                    source,
                    frame_index=frame_index,
                    y=y,
                    x0=x0,
                    x1=x1,
                    class_id=source_class,
                )
                runs.append(
                    ResidualRun(
                        run_index=run_index,
                        frame_index=frame_index,
                        y=y,
                        x0=x0,
                        x1=x1,
                        source_class=source_class,
                        candidate_class_hist=_hist(cand_row[x0:x1]),
                        boundary_pixels=boundary_pixels,
                        foveal_weight_sum=_foveal_weight_sum_for_span(
                            frame_index=frame_index,
                            y=y,
                            x0=x0,
                            x1=x1,
                            foveation=foveation,
                        ),
                    )
                )
                run_index += 1
    return runs


def _frame_indices_for_pair(pair: int, *, frame_count: int) -> list[int]:
    return [idx for idx in (pair * 2, pair * 2 + 1) if idx < frame_count]


def _connected_row_run_groups(runs: list[ResidualRun]) -> list[list[ResidualRun]]:
    by_frame_class: dict[tuple[int, int], list[ResidualRun]] = defaultdict(list)
    for run in runs:
        by_frame_class[(run.frame_index, run.source_class)].append(run)

    groups: list[list[ResidualRun]] = []
    for (_frame, _class_id), items in sorted(by_frame_class.items()):
        parent = {run.run_index: run.run_index for run in items}

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        by_y: dict[int, list[ResidualRun]] = defaultdict(list)
        for run in items:
            by_y[run.y].append(run)
        for row_runs in by_y.values():
            row_runs.sort(key=lambda run: (run.x0, run.x1, run.run_index))

        for y in sorted(by_y):
            previous = by_y.get(y - 1, [])
            if not previous:
                continue
            previous_sorted = previous
            current_sorted = by_y[y]
            start = 0
            for run in current_sorted:
                while start < len(previous_sorted) and previous_sorted[start].x1 < run.x0 - 1:
                    start += 1
                idx = start
                while idx < len(previous_sorted) and previous_sorted[idx].x0 <= run.x1 + 1:
                    prev = previous_sorted[idx]
                    if prev.x1 >= run.x0 - 1:
                        union(prev.run_index, run.run_index)
                    idx += 1

        buckets: dict[int, list[ResidualRun]] = defaultdict(list)
        for run in items:
            buckets[find(run.run_index)].append(run)
        for bucket in buckets.values():
            groups.append(sorted(bucket, key=lambda run: run.run_index))

    groups.sort(
        key=lambda group: (
            group[0].frame_index,
            group[0].source_class,
            min(run.y for run in group),
            min(run.x0 for run in group),
            min(run.run_index for run in group),
        )
    )
    return groups


def _groups_for_family(
    runs: list[ResidualRun],
    atom_family: str,
    *,
    frame_count: int,
) -> list[tuple[dict[str, Any], list[ResidualRun]]]:
    if atom_family == "row_run":
        return [
            (
                {
                    "frame_index": run.frame_index,
                    "pair_index": run.pair_index,
                    "class_id": run.source_class,
                    "y": run.y,
                    "x0": run.x0,
                    "x1_exclusive": run.x1,
                },
                [run],
            )
            for run in runs
        ]

    if atom_family == "connected_row_run":
        out = []
        for index, group in enumerate(_connected_row_run_groups(runs)):
            pair_indices = sorted({run.pair_index for run in group})
            frames = sorted({run.frame_index for run in group})
            xs = [run.x0 for run in group] + [run.x1 for run in group]
            ys = [run.y for run in group]
            classes = sorted({run.source_class for run in group})
            out.append(
                (
                    {
                        "component_index": index,
                        "pair_indices": pair_indices,
                        "frame_indices": frames,
                        "class_ids": classes,
                        "bbox_xyxy": [min(xs), min(ys), max(xs), max(ys) + 1],
                    },
                    group,
                )
            )
        return out

    grouped: dict[tuple[Any, ...], list[ResidualRun]] = defaultdict(list)
    for run in runs:
        if atom_family == "pair":
            key = (run.pair_index,)
        elif atom_family == "frame":
            key = (run.frame_index,)
        elif atom_family == "class":
            key = (run.source_class,)
        elif atom_family == "pair_class":
            key = (run.pair_index, run.source_class)
        elif atom_family == "frame_class":
            key = (run.frame_index, run.source_class)
        else:  # pragma: no cover - guarded by _parse_family_list
            raise AssertionError(atom_family)
        grouped[key].append(run)

    out = []
    for key in sorted(grouped):
        if atom_family == "pair":
            pair = int(key[0])
            identity = {
                "pair_index": pair,
                "frame_indices": _frame_indices_for_pair(pair, frame_count=frame_count),
            }
        elif atom_family == "frame":
            frame = int(key[0])
            identity = {"frame_index": frame, "pair_index": frame // 2}
        elif atom_family == "class":
            identity = {"class_id": int(key[0])}
        elif atom_family == "pair_class":
            pair = int(key[0])
            identity = {
                "pair_index": pair,
                "frame_indices": _frame_indices_for_pair(pair, frame_count=frame_count),
                "class_id": int(key[1]),
            }
        else:
            frame = int(key[0])
            identity = {"frame_index": frame, "pair_index": frame // 2, "class_id": int(key[1])}
        out.append((identity, sorted(grouped[key], key=lambda run: run.run_index)))
    return out


def _pair_component_weight(pair: int, pair_prior: dict[str, Any] | None, *, strength: float) -> float:
    if pair_prior is None:
        return 1.0
    max_signal = float(pair_prior.get("max_signal") or 0.0)
    if max_signal <= 0.0:
        return 1.0
    signals = pair_prior["signals"]
    signal = float(signals[pair]) if pair < len(signals) else 0.0
    return 1.0 + strength * max(0.0, signal) / max_signal


def _hard_pair_weight(pair: int, hard_pair_ranks: dict[int, int], *, bonus: float) -> float:
    rank = hard_pair_ranks.get(pair)
    if rank is None:
        return 1.0
    return 1.0 + bonus / math.sqrt(float(rank + 1))


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 12)


def _atom_from_runs(
    *,
    atom_family: str,
    identity: dict[str, Any],
    runs: list[ResidualRun],
    total_pixels: int,
    class_weights: list[float],
    pair_prior: dict[str, Any] | None,
    hard_pair_ranks: dict[int, int],
    component_trace_strength: float,
    hard_pair_bonus: float,
    boundary_bonus: float,
    long_run_bonus: float,
    long_run_reference: float,
) -> dict[str, Any]:
    residual_pixels = sum(run.length for run in runs)
    touched_rows = {(run.frame_index, run.y) for run in runs}
    pair_indices = sorted({run.pair_index for run in runs})
    frame_indices = sorted({run.frame_index for run in runs})
    class_ids = sorted({run.source_class for run in runs})
    xs = [run.x0 for run in runs] + [run.x1 for run in runs]
    ys = [run.y for run in runs]

    source_hist: Counter[int] = Counter()
    candidate_hist: Counter[int] = Counter()
    weighted_pixels = 0.0
    component_weight_sum = 0.0
    hard_weight_sum = 0.0
    class_weight_sum = 0.0
    foveal_weight_sum = 0.0
    boundary_weight_sum = 0.0
    long_run_weight_sum = 0.0
    boundary_pixels = 0

    for run in runs:
        length = run.length
        source_hist[run.source_class] += length
        for cls, count in run.candidate_class_hist:
            candidate_hist[int(cls)] += int(count)
        component_weight = _pair_component_weight(
            run.pair_index,
            pair_prior,
            strength=component_trace_strength,
        )
        hard_weight = _hard_pair_weight(run.pair_index, hard_pair_ranks, bonus=hard_pair_bonus)
        class_weight = class_weights[run.source_class] if run.source_class < len(class_weights) else 1.0
        foveal_weight = run.foveal_weight_sum / max(length, 1)
        boundary_fraction = run.boundary_pixels / max(length, 1)
        boundary_weight = 1.0 + boundary_bonus * boundary_fraction
        long_run_weight = 1.0 + long_run_bonus * min(1.0, length / long_run_reference)
        combined_weight = (
            component_weight
            * hard_weight
            * class_weight
            * foveal_weight
            * boundary_weight
            * long_run_weight
        )
        weighted_pixels += length * combined_weight
        component_weight_sum += length * component_weight
        hard_weight_sum += length * hard_weight
        class_weight_sum += length * class_weight
        foveal_weight_sum += length * foveal_weight
        boundary_weight_sum += length * boundary_weight
        long_run_weight_sum += length * long_run_weight
        boundary_pixels += run.boundary_pixels

    uncompressed_row_count_bytes = CMG3_ROW_COUNT_BYTES * len(touched_rows)
    uncompressed_run_record_bytes = CMG3_RUN_RECORD_BYTES * len(runs)
    estimated_charged_bytes = uncompressed_row_count_bytes + uncompressed_run_record_bytes
    estimated_rate_cost = LAMBDA_RATE * estimated_charged_bytes
    estimated_score_saved_proxy = weighted_pixels / max(total_pixels, 1)
    density = estimated_score_saved_proxy / estimated_charged_bytes if estimated_charged_bytes else None
    lagrangian_net = estimated_score_saved_proxy - estimated_rate_cost
    atom_id = _atom_id(atom_family, identity)

    return {
        "atom_id": atom_id,
        "atom_family": atom_family,
        "identity": identity,
        "pair_indices": pair_indices,
        "frame_indices": frame_indices,
        "class_ids": class_ids,
        "bbox_xyxy": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys) + 1)],
        "residual_pixels": int(residual_pixels),
        "residual_pixel_fraction_of_tensor": _round(residual_pixels / max(total_pixels, 1)),
        "run_count": int(len(runs)),
        "touched_row_count": int(len(touched_rows)),
        "source_class_histogram_pixels": _dict_hist(source_hist),
        "candidate_class_histogram_pixels": _dict_hist(candidate_hist),
        "boundary_pixels": int(boundary_pixels),
        "boundary_pixel_fraction": _round(boundary_pixels / max(residual_pixels, 1)),
        "cost_model": {
            "model": "cmg3_row_count_byte_plus_run_records_uncompressed_proxy",
            "row_count_byte_per_touched_frame_row": CMG3_ROW_COUNT_BYTES,
            "run_record_bytes": CMG3_RUN_RECORD_BYTES,
            "record_struct": "u8_count_then_u8_class_u16_start_u16_end_le",
            "uncompressed_row_count_bytes": int(uncompressed_row_count_bytes),
            "uncompressed_run_record_bytes": int(uncompressed_run_record_bytes),
            "estimated_charged_bytes": int(estimated_charged_bytes),
            "formula": "estimated_charged_bytes = touched_row_count*1 + run_count*5",
            "compressed_cost_caveat": (
                "CMG3 archive compression is non-additive. This atom ledger uses "
                "the uncompressed run-record proxy for deterministic planning; "
                "selected atoms must be rebuilt and byte-measured as one archive."
            ),
        },
        "weights": {
            "component_pair_weight_pixel_mean": _round(component_weight_sum / max(residual_pixels, 1)),
            "hard_pair_weight_pixel_mean": _round(hard_weight_sum / max(residual_pixels, 1)),
            "class_weight_pixel_mean": _round(class_weight_sum / max(residual_pixels, 1)),
            "foveal_weight_pixel_mean": _round(foveal_weight_sum / max(residual_pixels, 1)),
            "boundary_weight_pixel_mean": _round(boundary_weight_sum / max(residual_pixels, 1)),
            "long_run_weight_pixel_mean": _round(long_run_weight_sum / max(residual_pixels, 1)),
            "weighted_residual_pixel_proxy": _round(weighted_pixels),
        },
        "lagrangian": {
            "lambda_rate": _round(LAMBDA_RATE),
            "estimated_marginal_score_saved_proxy": _round(estimated_score_saved_proxy),
            "estimated_rate_score_cost": _round(estimated_rate_cost),
            "estimated_lagrangian_net_proxy": _round(lagrangian_net),
            "estimated_score_saved_per_charged_byte": _round(density),
            "break_even_bytes_per_score": _round(BREAK_EVEN_BYTES_PER_SCORE),
            "score_saved_needed_to_pay_rate": _round(estimated_rate_cost),
        },
        "score_claim": False,
        "no_score_claim": True,
        "evidence_grade": EVIDENCE_GRADE,
    }


def _sort_atom(atom: dict[str, Any]) -> tuple[float, float, float, int, int, str]:
    lagrangian = atom["lagrangian"]
    density = float(lagrangian["estimated_score_saved_per_charged_byte"] or 0.0)
    net = float(lagrangian["estimated_lagrangian_net_proxy"] or 0.0)
    saved = float(lagrangian["estimated_marginal_score_saved_proxy"] or 0.0)
    pixels = int(atom["residual_pixels"])
    charged = int(atom["cost_model"]["estimated_charged_bytes"])
    return (-density, -net, -saved, -pixels, charged, str(atom["atom_id"]))


def _build_atom_tables(
    runs: list[ResidualRun],
    *,
    atom_families: tuple[str, ...],
    frame_count: int,
    total_pixels: int,
    class_weights: list[float],
    pair_prior: dict[str, Any] | None,
    hard_pair_ranks: dict[int, int],
    component_trace_strength: float,
    hard_pair_bonus: float,
    boundary_bonus: float,
    long_run_bonus: float,
    long_run_reference: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    atoms: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for family in atom_families:
        family_groups = _groups_for_family(runs, family, frame_count=frame_count)
        counts[family] = len(family_groups)
        for identity, group in family_groups:
            atoms.append(
                _atom_from_runs(
                    atom_family=family,
                    identity=identity,
                    runs=group,
                    total_pixels=total_pixels,
                    class_weights=class_weights,
                    pair_prior=pair_prior,
                    hard_pair_ranks=hard_pair_ranks,
                    component_trace_strength=component_trace_strength,
                    hard_pair_bonus=hard_pair_bonus,
                    boundary_bonus=boundary_bonus,
                    long_run_bonus=long_run_bonus,
                    long_run_reference=long_run_reference,
                )
            )
    atoms.sort(key=_sort_atom)
    return atoms, counts


def build_ledger(
    *,
    source_mask_array: Path,
    output_json: Path,
    candidate_mask_array: Path | None = None,
    candidate_manifest: Path | None = None,
    component_trace_json: Path | None = None,
    hard_pair_json: Path | None = None,
    foveation_json: Path | None = None,
    foveal_center: tuple[float, float] | None = None,
    foveal_sigma: float | None = None,
    foveal_strength: float = 1.5,
    class_weights_json: Path | None = None,
    atom_families: tuple[str, ...] = DEFAULT_ATOM_FAMILIES,
    max_atoms: int = 256,
    component_trace_strength: float = 2.0,
    hard_pair_bonus: float = 1.5,
    boundary_bonus: float = 0.5,
    long_run_bonus: float = 0.25,
    long_run_reference: float = 64.0,
) -> dict[str, Any]:
    atom_families = _parse_family_list(atom_families)
    if (candidate_mask_array is None) == (candidate_manifest is None):
        raise PlannerError("provide exactly one of candidate_mask_array or candidate_manifest")
    if max_atoms <= 0:
        raise PlannerError("max_atoms must be positive")
    if component_trace_strength < 0 or hard_pair_bonus < 0 or boundary_bonus < 0 or long_run_bonus < 0:
        raise PlannerError("prior bonuses/strengths must be nonnegative")
    if long_run_reference <= 0:
        raise PlannerError("long_run_reference must be positive")

    source_path = source_mask_array.resolve()
    source = _load_mask_array(source_path, label="source mask array")
    frame_count, height, width = (int(v) for v in source.shape)
    class_count = max(DEFAULT_CLASS_COUNT, int(source.max()) + 1)

    candidate_source: dict[str, Any]
    if candidate_mask_array is not None:
        candidate_path = candidate_mask_array.resolve()
        candidate = _load_mask_array(candidate_path, label="candidate mask array")
        candidate_source = {
            "mode": "candidate_mask_array",
            "candidate_mask_array": {
                "path": str(candidate_path),
                "npy_sha256": _sha256_file(candidate_path),
            },
        }
    else:
        assert candidate_manifest is not None
        candidate, candidate_source = _load_candidate_from_manifest(
            candidate_manifest,
            source=source,
            class_count=class_count,
        )

    if candidate.shape != source.shape:
        raise PlannerError(f"source/candidate shape mismatch: {source.shape} != {candidate.shape}")
    class_count = max(class_count, int(candidate.max()) + 1)
    pair_count = (frame_count + 1) // 2

    foveation, foveation_meta = _load_foveation_config(
        foveation_json,
        center=foveal_center,
        sigma=foveal_sigma,
        strength=foveal_strength,
        width=width,
        height=height,
    )
    if foveation is not None and foveation.frame_centers is not None:
        center_count = len(foveation.frame_centers)
        if center_count == frame_count:
            pass
        elif center_count == frame_count * 2:
            foveation = replace(
                foveation,
                frame_center_indexing="pair_average_from_full_frames",
            )
            if foveation_meta is not None:
                foveation_meta = dict(foveation_meta)
                foveation_meta["config"] = foveation.as_json()
                foveation_meta["source_frame_count"] = frame_count
                foveation_meta["frame_count_mapping"] = (
                    "full-frame dynamic centers averaged into one center per "
                    "half-frame/pair mask row"
                )
        else:
            raise PlannerError(
                f"dynamic foveation frame center count must match source frame count "
                f"or exactly 2x for full-frame-to-pair mapping: {center_count} vs {frame_count}"
            )
    class_weights, class_weights_meta = _load_class_weights(class_weights_json, class_count=class_count)
    pair_prior = _load_component_trace(component_trace_json, pair_count=pair_count)
    hard_pair_meta = _load_hard_pairs(hard_pair_json)
    hard_pair_indices = [] if hard_pair_meta is None else hard_pair_meta["pair_indices"]
    hard_pair_ranks = {pair: rank for rank, pair in enumerate(hard_pair_indices) if pair < pair_count}
    ignored_hard_pairs = [pair for pair in hard_pair_indices if pair >= pair_count]

    runs = _extract_residual_runs(source, candidate, foveation=foveation)
    total_pixels = int(source.size)
    residual_pixels = sum(run.length for run in runs)
    atoms, family_counts = _build_atom_tables(
        runs,
        atom_families=atom_families,
        frame_count=frame_count,
        total_pixels=total_pixels,
        class_weights=class_weights,
        pair_prior=pair_prior,
        hard_pair_ranks=hard_pair_ranks,
        component_trace_strength=component_trace_strength,
        hard_pair_bonus=hard_pair_bonus,
        boundary_bonus=boundary_bonus,
        long_run_bonus=long_run_bonus,
        long_run_reference=long_run_reference,
    )
    top_atoms = atoms[:max_atoms]

    source_hist = Counter(int(v) for v in source.reshape(-1).tolist())
    candidate_hist = Counter(int(v) for v in candidate.reshape(-1).tolist())

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "no_score_claim": True,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "gpu_required": False,
        "cuda_jobs_launched": False,
        "remote_jobs_dispatched": False,
        "canonical_score_source_required": (
            "archive.zip -> inflate.sh -> upstream/evaluate.py via "
            "experiments/contest_auth_eval.py --device cuda"
        ),
        "planning_warning": (
            "This ledger ranks exact mask residual atoms with byte/proxy priors only. "
            "It is not score evidence and cannot promote, rank, retire, or kill a method."
        ),
        "inputs": {
            "source_mask_array": {
                "path": str(source_path),
                "npy_sha256": _sha256_file(source_path),
                "tensor_sha256": _sha256_bytes(source.tobytes(order="C")),
            },
            "candidate": candidate_source,
            "component_trace_json": None
            if pair_prior is None
            else {
                "path": pair_prior["path"],
                "sha256": pair_prior["sha256"],
                "score_claim": pair_prior.get("score_claim"),
                "evidence_grade": pair_prior.get("evidence_grade"),
                "signal_source": pair_prior["signal_source"],
                "top_pair_indices": pair_prior["top_pair_indices"],
            },
            "hard_pair_json": None
            if hard_pair_meta is None
            else {
                "path": hard_pair_meta["path"],
                "sha256": hard_pair_meta["sha256"],
                "pair_indices": hard_pair_indices,
                "applied_pair_indices": sorted(hard_pair_ranks),
                "ignored_out_of_range_pair_indices": ignored_hard_pairs,
            },
            "class_weights_json": class_weights_meta,
            "foveation": foveation_meta,
        },
        "tensor": {
            "shape": {"frames": frame_count, "height": height, "width": width},
            "class_count": class_count,
            "pair_count": pair_count,
            "total_pixels": total_pixels,
            "source_class_histogram_pixels": _dict_hist(source_hist),
            "candidate_class_histogram_pixels": _dict_hist(candidate_hist),
            "residual_pixels": int(residual_pixels),
            "residual_pixel_fraction": _round(residual_pixels / max(total_pixels, 1)),
            "residual_run_count": len(runs),
        },
        "formulas": {
            "contest_score_formula": (
                "score = 100*seg_dist + sqrt(10*pose_dist) + "
                "25*archive_bytes/37545489"
            ),
            "lambda_rate": LAMBDA_RATE,
            "lambda_rate_formula": "25 / 37545489",
            "break_even_bytes_per_score": BREAK_EVEN_BYTES_PER_SCORE,
            "break_even_bytes_per_score_formula": "1 / lambda_rate = 37545489 / 25",
            "cmg3_uncompressed_run_cost_formula": (
                "estimated_charged_bytes = touched_row_count*1 + run_count*5"
            ),
            "pixel_proxy_formula": (
                "estimated_marginal_score_saved_proxy = "
                "sum_residual_pixels(product(prior_weights)) / total_tensor_pixels"
            ),
            "lagrangian_net_formula": (
                "estimated_lagrangian_net_proxy = "
                "estimated_marginal_score_saved_proxy - lambda_rate*estimated_charged_bytes"
            ),
            "ranking_key": (
                "estimated_score_saved_per_charged_byte desc, "
                "estimated_lagrangian_net_proxy desc, residual_pixels desc, bytes asc"
            ),
            "compressed_cost_caveat": (
                "Per-atom byte costs are deterministic uncompressed CMG3 run-record "
                "proxies. Compression and header overhead are non-additive and must "
                "be measured on a concrete rebuilt archive before exact eval."
            ),
        },
        "priors": {
            "component_trace_strength": component_trace_strength,
            "hard_pair_bonus": hard_pair_bonus,
            "boundary_bonus": boundary_bonus,
            "long_run_bonus": long_run_bonus,
            "long_run_reference": long_run_reference,
            "class_weights": {str(idx): round(weight, 12) for idx, weight in enumerate(class_weights)},
            "component_pair_weight_formula": (
                "1 + component_trace_strength * pair_signal / max_pair_signal"
            )
            if pair_prior is not None
            else None,
            "hard_pair_weight_formula": "1 + hard_pair_bonus / sqrt(rank + 1)"
            if hard_pair_meta is not None
            else None,
            "foveal_weight_formula": "1 + strength*exp(-distance_squared/(2*sigma^2))"
            if foveation is not None
            else None,
            "boundary_weight_formula": "1 + boundary_bonus * boundary_pixel_fraction",
            "long_run_weight_formula": "1 + long_run_bonus * min(1, run_length/long_run_reference)",
        },
        "atom_families": list(atom_families),
        "atom_family_counts": family_counts,
        "atom_count": len(atoms),
        "top_atoms": top_atoms,
    }
    _write_json(output_json, payload)
    return payload


def _parse_center(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected X,Y")
    return float(parts[0]), float(parts[1])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mask-array", type=Path, required=True)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--candidate-mask-array", type=Path)
    candidate.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--component-trace-json", type=Path)
    parser.add_argument("--hard-pair-json", type=Path)
    parser.add_argument("--class-weights-json", type=Path)
    parser.add_argument("--foveation-json", type=Path)
    parser.add_argument("--foveal-center", type=_parse_center)
    parser.add_argument("--foveal-sigma", type=float)
    parser.add_argument("--foveal-strength", type=float, default=1.5)
    parser.add_argument("--atom-families", default=",".join(DEFAULT_ATOM_FAMILIES))
    parser.add_argument("--max-atoms", type=int, default=256)
    parser.add_argument("--component-trace-strength", type=float, default=2.0)
    parser.add_argument("--hard-pair-bonus", type=float, default=1.5)
    parser.add_argument("--boundary-bonus", type=float, default=0.5)
    parser.add_argument("--long-run-bonus", type=float, default=0.25)
    parser.add_argument("--long-run-reference", type=float, default=64.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_ledger(
        source_mask_array=args.source_mask_array,
        candidate_mask_array=args.candidate_mask_array,
        candidate_manifest=args.candidate_manifest,
        output_json=args.output_json,
        component_trace_json=args.component_trace_json,
        hard_pair_json=args.hard_pair_json,
        foveation_json=args.foveation_json,
        foveal_center=args.foveal_center,
        foveal_sigma=args.foveal_sigma,
        foveal_strength=args.foveal_strength,
        class_weights_json=args.class_weights_json,
        atom_families=_parse_family_list(args.atom_families),
        max_atoms=args.max_atoms,
        component_trace_strength=args.component_trace_strength,
        hard_pair_bonus=args.hard_pair_bonus,
        boundary_bonus=args.boundary_bonus,
        long_run_bonus=args.long_run_bonus,
        long_run_reference=args.long_run_reference,
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "schema": payload["schema"],
                "atom_count": payload["atom_count"],
                "residual_pixels": payload["tensor"]["residual_pixels"],
                "score_claim": False,
                "evidence_grade": EVIDENCE_GRADE,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
