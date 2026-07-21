#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable B1-B5 measurement runner for Task #597.

The default path measures scorer-free engineering surfaces only and reports B2
as literally incomplete.  A real hard-oracle adapter is an explicit
compress-time callback (``module:function``); scorer code and weights never
enter schema or receiver bytes.  Every chunk is atomically preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import platform
import resource
import sys
import tempfile
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.optimization.predict_project_receiver import (
    CANONICAL_LAW_RESOLUTION_CUSTODY,
    CANONICAL_LAW_RESOLUTION_SHA256,
    M1_ANCHORS,
    TEMPORAL_JITTER_AMORTIZATION_RATIO,
    TEMPORAL_JITTER_EQUATION_ID,
    PredictProjectReceiverError,
    component_byte_accounting,
    double_decode_hash,
    extract_constraint_violations,
    global_joint_waterfill,
    hard_oracle_custody_sha256,
    plane_cache_key,
    predict_cell_field,
    receiver_composition_metadata,
    stratify_predictor_quality,
    validate_global_joint_waterfill_evidence,
    validate_hard_oracle_custody,
)
from tac.optimization.predict_project_schema import (
    canonical_json_bytes,
    parse_constraint_seed,
    serialize_constraint_seed,
)

RECEIPT_SCHEMA: Final = "predict_project_b1_b5_measurement.v0"
HARD_ORACLE_SCHEMA: Final = "predict_project_hard_oracle_pair.v0"
SSD_ROOTS: Final = (Path("/Volumes/VertigoDataTier/pact"), Path("/Volumes/APDataStore/pact"))
IMPLEMENTATION_SOURCE_PATHS: Final = (
    "src/tac/optimization/predict_project_receiver.py",
    "src/tac/optimization/predict_project_schema.py",
    "tools/measure_predict_project_receiver.py",
    "src/tac/canonical_equations/predict_project_receiver_20260721.py",
)


class MeasurementError(ValueError):
    """Invalid CLI custody, callback output, or resumability state."""


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _implementation_source_custody() -> dict[str, Any]:
    """Hash every Task #597 implementation source that affects evidence use."""

    repository_root = Path(__file__).resolve().parents[1]
    files: dict[str, str] = {}
    for relative_path in IMPLEMENTATION_SOURCE_PATHS:
        path = repository_root / relative_path
        try:
            files[relative_path] = _sha256(path.read_bytes())
        except OSError as exc:
            raise MeasurementError(f"implementation source cannot be hashed: {relative_path}") from exc
    return {
        "schema": "predict_project_measurement_implementation_sources.v0",
        "files": files,
        "aggregate_sha256": _sha256(canonical_json_bytes(files)),
        "resume_policy": "EXACT_SOURCE_AND_CONFIG_ONLY_OLD_CODE_STAGES_REFUSED",
    }


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _load_adapter(path: str | None) -> Callable[..., Mapping[str, Any]] | None:
    if path is None:
        return None
    if ":" not in path:
        raise MeasurementError("hard-oracle adapter must be module:function")
    module_name, function_name = path.split(":", 1)
    callback = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(callback):
        raise MeasurementError("hard-oracle adapter is not callable")
    return callback


def _load_json_mapping(path: Path | None, label: str) -> Mapping[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"{label} cannot be read as JSON") from exc
    if not isinstance(value, dict):
        raise MeasurementError(f"{label} must contain one JSON object")
    return value


def _adapter_identity(callback: Callable[..., Mapping[str, Any]]) -> dict[str, str]:
    source_path = inspect.getsourcefile(callback)
    if source_path is None:
        raise MeasurementError("hard-oracle adapter must have a hashable source file")
    try:
        source_sha256 = _sha256(Path(source_path).resolve().read_bytes())
    except OSError as exc:
        raise MeasurementError("hard-oracle adapter source cannot be read") from exc
    return {
        "identity": f"{callback.__module__}:{callback.__qualname__}",
        "source_sha256": source_sha256,
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    envelope = f"{array.dtype.str}:{','.join(map(str, array.shape))}:".encode("ascii") + array.tobytes()
    return _sha256(envelope)


def _validate_hard_oracle_row(
    value: Mapping[str, Any],
    pair_index: int,
    *,
    seed_sha256: str,
    predicted: np.ndarray,
    represented: np.ndarray,
    expected_adapter: Mapping[str, str],
) -> dict[str, Any]:
    required = {
        "schema",
        "pair_index",
        "d_seg",
        "d_pose",
        "cell_exact",
        "pose_within_tube",
        "uint8_factor2_exact",
        "stage_seconds",
        "custody",
        "desired_cells",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise MeasurementError("hard-oracle adapter fields mismatch")
    if value["schema"] != HARD_ORACLE_SCHEMA or value["pair_index"] != pair_index:
        raise MeasurementError("hard-oracle pair custody mismatch")
    for key in ("d_seg", "d_pose"):
        metric = value[key]
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not np.isfinite(float(metric))
            or metric < 0
        ):
            raise MeasurementError(f"hard-oracle {key} is invalid")
    for key in ("cell_exact", "pose_within_tube", "uint8_factor2_exact"):
        if not isinstance(value[key], bool):
            raise MeasurementError(f"hard-oracle {key} must be boolean")
    try:
        custody = validate_hard_oracle_custody(value["custody"])
    except PredictProjectReceiverError as exc:
        raise MeasurementError(str(exc)) from exc
    if custody["adapter"] != expected_adapter:
        raise MeasurementError("hard-oracle adapter identity/source custody mismatch")
    if custody["inputs"]["source_sha256"] != seed_sha256:
        raise MeasurementError("hard-oracle source SHA-256 does not match exact seed bytes")
    stages = value["stage_seconds"]
    if not isinstance(stages, Mapping) or set(stages) != {"projection", "realization", "verification"}:
        raise MeasurementError("hard-oracle stage_seconds fields mismatch")
    for key, number in stages.items():
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not np.isfinite(float(number))
            or number < 0
        ):
            raise MeasurementError(f"hard-oracle stage_seconds.{key} is invalid")
    desired_cells = value["desired_cells"]
    desired_quality: dict[str, Any] | None = None
    desired_cells_sha256: str | None = None
    if desired_cells is not None:
        desired = np.asarray(desired_cells)
        if desired.shape != predicted.shape or desired.dtype != np.uint8:
            raise MeasurementError("hard-oracle desired_cells must equal predicted shape with uint8 dtype")
        desired_cells_sha256 = _array_sha256(desired)
        desired_quality = stratify_predictor_quality(
            predicted,
            desired,
            evidence_source="hard_oracle_real_desired_cells",
            desired_cells_sha256=desired_cells_sha256,
        )
    result = {key: value[key] for key in required - {"desired_cells", "custody"}}
    result["custody"] = custody
    result["custody_sha256"] = hard_oracle_custody_sha256(custody)
    result["pair_input_sha256"] = _array_sha256(predicted)
    result["represented_input_sha256"] = _array_sha256(represented)
    result["desired_cells_sha256"] = desired_cells_sha256
    result["desired_quality"] = desired_quality
    return result


def _desired_from_declared_violations(seed: Mapping[str, Any], pair_index: int, predicted: np.ndarray) -> np.ndarray:
    desired = predicted.copy()
    for row in seed["constraint_seeds"]:
        if row["time"] == pair_index and row["frame_index"] == 1:
            desired[row["y"], row["x"]] = row["cell_id"]
    return desired


def _component_compressed_bytes(seed: Mapping[str, Any]) -> dict[str, int]:
    chart = seed["ground_chart"]
    jitter = seed["boundary_jitter"]
    components = {
        "chart": {key: value for key, value in chart.items() if key != "cells"},
        "sites": chart["cells"],
        "trajectory": seed["trajectory"],
        "bulk": {
            "schema": seed["schema"],
            "container": seed["container"],
            "units": seed["units"],
            "receiver": seed["receiver"],
            "authority": seed["authority"],
        },
        "jitter": jitter["r0"],
        "response": {"selected_rung": jitter["selected_rung"], "r1": jitter["r1"], "r2": jitter["r2"]},
        "tracks": seed["movable_tracks"],
        "events": seed["events"],
        "pose_tightening": seed["pose_tightening"],
        "eat_flip": [],
        "constraints": seed["constraint_seeds"],
    }
    return {name: len(zlib.compress(canonical_json_bytes(value), level=9)) for name, value in components.items()}


def _choose_output_root(explicit: Path | None, *, allow_local: bool) -> Path:
    if explicit is not None:
        return explicit.resolve()
    for root in SSD_ROOTS:
        if root.is_dir() and os.access(root, os.W_OK):
            return root / "task597_predict_project"
    if allow_local:
        return Path(".omx/research/task597_predict_project_measurements").resolve()
    raise MeasurementError("no writable SSD tier; pass --allow-local-output for explicit local opt-in")


def _load_existing_chunks(
    stage_dir: Path,
    config_sha256: str,
    implementation_sources_sha256: str,
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not stage_dir.exists():
        return rows
    for path in sorted(stage_dir.glob("pair_*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MeasurementError(f"corrupt stage receipt {path}") from exc
        if value.get("implementation_sources_sha256") != implementation_sources_sha256:
            raise MeasurementError(f"stage implementation-source drift at {path}")
        if value.get("config_sha256") != config_sha256:
            raise MeasurementError(f"stage config drift at {path}")
        pair_index = value.get("pair_index")
        if isinstance(pair_index, bool) or not isinstance(pair_index, int) or pair_index in rows:
            raise MeasurementError(f"invalid/duplicate pair stage {path}")
        rows[pair_index] = value
    return rows


def run_measurement(
    seed_path: Path,
    output_root: Path,
    *,
    pair_start: int = 0,
    pair_end: int = 600,
    chunk_size: int = 16,
    workers: int = 1,
    hard_oracle: Callable[..., Mapping[str, Any]] | None = None,
    global_waterfill_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run/resume the requested pair range and write atomic B1-B5 receipts."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in (pair_start, pair_end, chunk_size, workers)
    ):
        raise MeasurementError("pair/chunk/worker values must be exact integers")
    if not 0 <= pair_start < pair_end <= 600 or chunk_size < 1 or workers < 1:
        raise MeasurementError("invalid pair range, chunk size, or worker count")
    seed_bytes = seed_path.read_bytes()
    seed = parse_constraint_seed(seed_bytes)
    if serialize_constraint_seed(seed) != seed_bytes:
        raise MeasurementError("input seed is not canonical")
    if hard_oracle is not None and seed["receiver"]["seed"] != 1234:
        raise MeasurementError("hard-oracle measurement requires receiver seed 1234")
    output_root.mkdir(parents=True, exist_ok=True)
    adapter_identity = _adapter_identity(hard_oracle) if hard_oracle is not None else None
    implementation_sources = _implementation_source_custody()
    config = {
        "schema": RECEIPT_SCHEMA,
        "implementation_status": "BUILT_INTERFACE_MEASUREMENT_BLOCKED",
        "m1_anchors": dict(M1_ANCHORS),
        "implementation_sources": implementation_sources,
        "seed_sha256": _sha256(seed_bytes),
        "pair_start": pair_start,
        "pair_end": pair_end,
        "chunk_size": chunk_size,
        "workers": workers,
        "hard_oracle_enabled": hard_oracle is not None,
        "hard_oracle_adapter": adapter_identity,
        "global_waterfill_evidence_sha256": (
            _sha256(canonical_json_bytes(global_waterfill_evidence)) if global_waterfill_evidence is not None else None
        ),
        "seed": seed["receiver"]["seed"],
        "batch": 16,
    }
    config_sha256 = _sha256(canonical_json_bytes(config))
    stage_dir = output_root / "stages"
    rows = _load_existing_chunks(stage_dir, config_sha256, implementation_sources["aggregate_sha256"])
    resumed_stage_rows = len([pair for pair in rows if pair_start <= pair < pair_end])
    plane_cache: dict[str, np.ndarray] = {}
    invocation_cache_hits = 0
    invocation_cache_misses = 0

    for chunk_begin in range(pair_start, pair_end, chunk_size):
        chunk_end = min(pair_end, chunk_begin + chunk_size)
        for pair_index in range(chunk_begin, chunk_end):
            if pair_index in rows:
                continue
            timings: dict[str, float] = {}
            started = time.perf_counter()
            cache_key = plane_cache_key(seed_bytes, pair_index, 1, seed["receiver"]["projection_policy"]["algorithm"])
            clock = time.perf_counter()
            if cache_key in plane_cache:
                predicted = plane_cache[cache_key].copy()
                cache_hit = True
                invocation_cache_hits += 1
            else:
                predicted = predict_cell_field(seed, pair_index)
                plane_cache[cache_key] = predicted.copy()
                cache_hit = False
                invocation_cache_misses += 1
            timings["predict"] = time.perf_counter() - clock

            clock = time.perf_counter()
            desired = _desired_from_declared_violations(seed, pair_index, predicted)
            violations = extract_constraint_violations(predicted, desired, time=pair_index)
            timings["violation_scan"] = time.perf_counter() - clock

            # Projection/realization require real scorer cell/tube geometry from
            # the optional compress-time adapter. Never substitute a proxy.
            hard_row: dict[str, Any] | None = None
            clock = time.perf_counter()
            if hard_oracle is not None:
                hard_row = _validate_hard_oracle_row(
                    hard_oracle(
                        seed=seed,
                        pair_index=pair_index,
                        predicted=predicted.copy(),
                        represented=desired.copy(),
                        workers=workers,
                        measurement_seed=1234,
                        batch_size=16,
                    ),
                    pair_index,
                    seed_sha256=config["seed_sha256"],
                    predicted=predicted,
                    represented=desired,
                    expected_adapter=adapter_identity,
                )
                timings.update({key: float(value) for key, value in hard_row["stage_seconds"].items()})
                callback_total = time.perf_counter() - clock
            else:
                timings.update({"projection": 0.0, "realization": 0.0, "verification": 0.0})
                callback_total = 0.0

            clock = time.perf_counter()
            quality = (
                hard_row["desired_quality"]
                if hard_row is not None and hard_row["desired_quality"] is not None
                else stratify_predictor_quality(predicted, desired)
            )
            decode = double_decode_hash(lambda pair_index=pair_index: predict_cell_field(seed, pair_index))
            timings["receiver_verification"] = time.perf_counter() - clock

            clock = time.perf_counter()
            serialized_violations = canonical_json_bytes(violations)
            timings["serialization"] = time.perf_counter() - clock
            timings["total"] = time.perf_counter() - started
            row = {
                "schema": "predict_project_pair_stage.v0",
                "config_sha256": config_sha256,
                "implementation_sources_sha256": implementation_sources["aggregate_sha256"],
                "pair_index": pair_index,
                "timings_seconds": timings,
                "plane_cache_key": cache_key,
                "plane_cache_hit": cache_hit,
                "output_sha256": decode.first_sha256,
                "double_decode_equal": decode.byte_identical,
                "b3": quality,
                "violation_records": len(violations),
                "seed_raw_bytes": len(serialized_violations),
                "seed_zlib9_bytes": len(zlib.compress(serialized_violations, level=9)),
                "hard_oracle": hard_row,
                "hard_oracle_stage_status": "MEASURED" if hard_row is not None else "NOT_RUN_NO_HARD_ORACLE",
                "hard_oracle_callback_total_seconds": callback_total,
                "output_bytes": int(predicted.nbytes),
                "peak_rss_bytes": _peak_rss_bytes(),
                "authority": "MEASURED [macOS-CPU advisory]" if hard_row is not None else "MEASURED engineering-only",
                "contest_authority": False,
                "score_claim": False,
                "promotion_eligible": False,
            }
            path = stage_dir / f"pair_{pair_index:04d}.json"
            _atomic_json(path, row)
            row["stage_sha256"] = _sha256(path.read_bytes())
            rows[pair_index] = row
        checkpoint = {
            "schema": "predict_project_chunk_checkpoint.v0",
            "config_sha256": config_sha256,
            "implementation_sources_sha256": implementation_sources["aggregate_sha256"],
            "completed_through_exclusive": chunk_end,
            "completed_pairs": len([pair for pair in rows if pair_start <= pair < pair_end]),
            "all_stage_files_preserved": True,
        }
        _atomic_json(output_root / f"checkpoint_{chunk_begin:04d}_{chunk_end:04d}.json", checkpoint)

    ordered = [rows[index] for index in range(pair_start, pair_end)]
    timing_keys = sorted({key for row in ordered for key in row["timings_seconds"]})
    timings = {key: sum(row["timings_seconds"].get(key, 0.0) for row in ordered) for key in timing_keys}
    hard_rows = [row["hard_oracle"] for row in ordered if row["hard_oracle"] is not None]
    full_n600 = pair_start == 0 and pair_end == 600 and len(hard_rows) == 600
    custody_payloads = {canonical_json_bytes(row["custody"]) for row in hard_rows}
    if len(custody_payloads) > 1:
        raise MeasurementError("mixed hard-oracle custody across measured rows")
    aggregate_custody = hard_rows[0]["custody"] if hard_rows else None
    aggregate_custody_sha256 = hard_rows[0]["custody_sha256"] if hard_rows else None
    b2 = {
        "schema": "predict_project_b2_hard_oracle.v0",
        "measurement_status": (
            "MEASURED" if full_n600 else ("MEASURED_PREFIX" if hard_rows else "INCOMPLETE_NO_HARD_ORACLE")
        ),
        "scope": "n600" if full_n600 else (f"prefix_or_slice_n{len(hard_rows)}" if hard_rows else "not_run"),
        "pair_count": len(hard_rows),
        "custody": aggregate_custody,
        "custody_sha256": aggregate_custody_sha256,
        "measurement_axis": aggregate_custody["measurement_axis"] if aggregate_custody else None,
        "custody_byte_identical_across_rows": bool(hard_rows) and len(custody_payloads) == 1,
        "cell_exact": bool(hard_rows) and all(row["cell_exact"] for row in hard_rows),
        "pose_within_tube": bool(hard_rows) and all(row["pose_within_tube"] for row in hard_rows),
        "uint8_factor2_exact": bool(hard_rows) and all(row["uint8_factor2_exact"] for row in hard_rows),
        "double_decode_equal": all(row["double_decode_equal"] for row in ordered),
        "d_seg": sum(float(row["d_seg"]) for row in hard_rows) / len(hard_rows) if hard_rows else None,
        "d_pose": sum(float(row["d_pose"]) for row in hard_rows) / len(hard_rows) if hard_rows else None,
        "score_claim": False,
        "promotion_eligible": False,
        "contest_authority": False,
        "n600_claim": full_n600,
    }
    if global_waterfill_evidence is not None:
        if not full_n600 or aggregate_custody is None:
            raise MeasurementError("global joint-waterfill evidence requires the real full-600 hard-oracle campaign")
        validated_global = validate_global_joint_waterfill_evidence(dict(global_waterfill_evidence))
        if canonical_json_bytes(validated_global["custody"]) != canonical_json_bytes(aggregate_custody):
            raise MeasurementError("global joint-waterfill custody must be byte-identical to aggregate B2 custody")
    b4 = global_joint_waterfill(global_waterfill_evidence)
    raw_components = component_byte_accounting(seed)
    compressed_components = _component_compressed_bytes(seed)
    per_frame_rows = []
    applied_constraint_count = 0
    for index in range(pair_start, pair_end):
        predicted = predict_cell_field(seed, index)
        represented = _desired_from_declared_violations(seed, index, predicted)
        applied_constraint_count += int(np.count_nonzero(predicted != represented))
        per_frame_rows.append({"pair_index": index, "cell_field_hex": represented.tobytes().hex()})
    per_frame_payload = canonical_json_bytes(per_frame_rows)
    per_frame_raw = len(per_frame_payload)
    per_frame_compressed = len(zlib.compress(per_frame_payload, level=9))
    single_object_raw = len(seed_bytes)
    single_object_compressed = len(zlib.compress(seed_bytes, level=9))
    b5 = {
        "schema": "predict_project_b5_single_object_vs_per_frame.v0",
        "equal_represented_pair_range": [pair_start, pair_end],
        "single_object": {
            "raw_bytes": single_object_raw,
            "zlib9_bytes": single_object_compressed,
            "raw_components": raw_components,
            "zlib9_components": compressed_components,
        },
        "per_frame": {
            "representation": "desired_from_declared_constraints",
            "raw_bytes": per_frame_raw,
            "zlib9_bytes": per_frame_compressed,
            "rows_sha256": _sha256(per_frame_payload),
            "applied_constraint_cells": applied_constraint_count,
        },
        "measured_boundary_normal_residual_error": None,
        "measured_event_count": len(seed["events"]),
        "causal_jitter_ladder": {
            "selected_rung": seed["boundary_jitter"]["selected_rung"],
            "equal_fidelity_evidence_status": seed["boundary_jitter"]["equal_fidelity_custody"]["evidence_status"],
            "R0": "BUILT_INTERFACE_MEASUREMENT_BLOCKED",
            "R1": "BUILT_INTERFACE_MEASUREMENT_BLOCKED",
            "R2": "BUILT_INTERFACE_MEASUREMENT_BLOCKED",
            "exceptions_term": "causal_sparse_exceptions",
        },
        "xi_custody": {
            "within_pair": "exact_seed_spline_ar",
            "cross_pair_proxy": "not_claimed",
        },
        "temporal_jitter_law": {
            "equation_id": TEMPORAL_JITTER_EQUATION_ID,
            "lawref_resolved_ratio": TEMPORAL_JITTER_AMORTIZATION_RATIO,
            "resolution_custody_sha256": CANONICAL_LAW_RESOLUTION_SHA256,
            "measured_ratio": single_object_compressed / per_frame_compressed if per_frame_compressed else None,
            "verdict_scope": "this chart realization and represented pair range only",
            "worldsheet_family_kill": False,
        },
        "archive_bytes_claim": False,
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "implementation_status": "BUILT_INTERFACE_MEASUREMENT_BLOCKED",
        "canonical_law_resolution": CANONICAL_LAW_RESOLUTION_CUSTODY,
        "canonical_law_resolution_sha256": CANONICAL_LAW_RESOLUTION_SHA256,
        "m1_anchors": dict(M1_ANCHORS),
        "config": config,
        "config_sha256": config_sha256,
        "b1": {
            "measurement_status": "MEASURED_PREFIX_OR_SLICE"
            if pair_end - pair_start < 600
            else "MEASURED_FULL_600_ENGINEERING",
            "pair_range": [pair_start, pair_end],
            "stage_seconds": timings,
            "worker_count": workers,
            "worker_env": os.environ.get("INFLATE_WORKERS"),
            "current_invocation_plane_cache_hits": invocation_cache_hits,
            "current_invocation_plane_cache_misses": invocation_cache_misses,
            "preserved_stage_plane_cache_hits": sum(bool(row["plane_cache_hit"]) for row in ordered),
            "preserved_stage_plane_cache_misses": sum(not bool(row["plane_cache_hit"]) for row in ordered),
            "resumed_stage_rows": resumed_stage_rows,
            "peak_rss_bytes": max(row["peak_rss_bytes"] for row in ordered),
            "output_bytes": sum(row["output_bytes"] for row in ordered),
            "output_hashes": [row["output_sha256"] for row in ordered],
            "double_decode_equal": all(row["double_decode_equal"] for row in ordered),
            "hardware_runtime": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
            },
            "optimization_ladder": ["algorithmic cut", "vectorize", "parallel workers", "Rust port"],
            "rust_hook_status": "interface_note_only",
            "one_shot_vs_pose_tolerance_sweep": "NOT_MEASURED_WITHOUT_HARD_ORACLE_ADAPTER",
            "runtime_rejection_authority": False,
        },
        "b2": b2,
        "b3": {
            "rows": [row["b3"] for row in ordered],
            "evidence_sources": sorted({row["b3"]["evidence_source"] for row in ordered}),
            "source_ground_truth_quality_claim": False,
            "authority": (
                "MEASURED_REAL_DESIRED_CELLS_NON_SOURCE_GROUND_TRUTH"
                if all(row["b3"]["evidence_source"] == "hard_oracle_real_desired_cells" for row in ordered)
                else "NON_AUTHORITATIVE_DECLARED_CONSTRAINT_FIXTURE"
            ),
            "canonical_seed_raw_bytes": len(seed_bytes),
            "canonical_seed_zlib9_bytes": len(zlib.compress(seed_bytes, level=9)),
            "archive_bytes_claim": False,
        },
        "b4": b4,
        "b5": b5,
        "composition": receiver_composition_metadata(),
        "gates": {
            "MS_vineyard_native_rasterizer": {
                "status": "MS_SCHEMA_BUILT_NATIVE_RASTERIZER_BLOCKED",
                "blocker": "MS_ARC_TO_CELL_RASTERIZATION_SEMANTICS_UNMEASURED",
            },
            "causal_jitter_r0_r1_r2": {
                "status": "BUILT_INTERFACE_MEASUREMENT_BLOCKED",
                "blocker": "equal-fidelity R0/R1/R2 canonical-byte measurements and causal exception counts by stratum are absent",
            },
            "G1_pose_blind_constraint_tightening": {
                "status": "BUILT_INTERFACE_MEASUREMENT_BLOCKED",
                "blocker": "real universal-within-box/tube Pose-tightening proof for the pose-blind decoder is absent",
            },
            "G2_camera_resolution_inverse_r": {
                "status": "BUILT_INTERFACE_MEASUREMENT_BLOCKED",
                "blocker": "full-n600 camera-resolution inverse-R uint8 parse-back is unmeasured",
            },
            "G3_frame_asymmetry": {
                "status": "BUILT_INTERFACE_MEASUREMENT_BLOCKED",
                "blocker": "full-n600 frame0 pose-only and frame1 seg-plus-pose behavior is unmeasured",
            },
            "G4_cross_host_byte_identity": {
                "status": "BUILT_INTERFACE_MEASUREMENT_BLOCKED",
                "blocker": "full-n600 CPU-CUDA decoded-byte identity is unmeasured",
            },
            "G5_named_section_container": {
                "status": "BUILT_INTERFACE_MEASUREMENT_BLOCKED",
                "blocker": "real full-600 named-section payload parse-back and receiver-consumption proof are absent",
            },
            "global_joint_waterfill": {
                "status": b4["status"],
                "blocker": (
                    None
                    if b4["status"] == "MEASURED_GLOBAL_JOINT_SWEEP"
                    else "same-joint-decode full-600 sweep with exact score/byte deltas, per-flip fixed point, action attribution/edit telemetry, learned-tail races, Pose knee, overlap-deduped flips, ordered compositions, and symmetric interactions is absent"
                ),
            },
            "per_flip_sellback": {
                "status": b4["per_flip_sellback_status"],
                "blocker": (
                    None
                    if b4["per_flip_sellback_status"] == "MEASURED_ITERATIVE_RECODE_FIXED_POINT"
                    else "M1-bound all-17926-flip iterative #557 recode fixed point is absent"
                ),
            },
            "action_level_ladder": {
                "status": b4["action_level_ladder_status"],
                "blocker": (
                    None
                    if b4["action_level_ladder_status"] == "MEASURED_SAME_JOINT_DECODE_ACTION_LEVEL_LADDER"
                    else "same-joint-decode L1-L5 pricing, ERF collateral, deterministic family selection, and all-17926 distribution are absent"
                ),
                "boundary_inverse_policy": b4["boundary_inverse_policy"],
            },
            "attribution_edit_telemetry": {
                "status": b4["attribution_edit_telemetry_status"],
                "blocker": (
                    None
                    if b4["attribution_edit_telemetry_status"] == "MEASURED_EXACT_ATTRIBUTION_AND_LADDER_EDITS"
                    else "all-flip #350 causal attribution and every L1-L5 #404/#420 deterministic through-R edit receipt are absent"
                ),
            },
            "learned_tail_race": {
                "status": b4["learned_tail_race_status"],
                "blocker": (
                    None
                    if b4["learned_tail_race_status"] == "MEASURED_EQUAL_FIDELITY_THREE_WAY_RACE"
                    else "per-stream equal-fidelity literal-versus-counted-S3-generator-versus-eaten races are absent"
                ),
            },
            "pose_tube_knee": {
                "status": b4["pose_tube_knee_status"],
                "blocker": (
                    None
                    if b4["pose_tube_knee_status"] == "MEASURED_SAME_JOINT_DECODE_POSE_TUBE_SWEEP"
                    else "same-joint-decode nonlinear Pose relaxation sweep and measured lambda crossing are absent"
                ),
            },
        },
        "stages_preserved": len(ordered),
        "authority": {
            "research_only": True,
            "measurement_axis": "[macOS-CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
        },
    }
    _atomic_json(output_root / "receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True, help="canonical predict-project seed bytes")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-local-output", action="store_true")
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-end", type=int, default=600)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("INFLATE_WORKERS", "1")))
    parser.add_argument("--hard-oracle", help="compress-time module:function adapter")
    parser.add_argument("--global-waterfill-evidence", type=Path, help="measured global joint-sweep JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = _choose_output_root(args.output_dir, allow_local=args.allow_local_output)
    receipt = run_measurement(
        args.seed,
        output_root,
        pair_start=args.pair_start,
        pair_end=args.pair_end,
        chunk_size=args.chunk_size,
        workers=args.workers,
        hard_oracle=_load_adapter(args.hard_oracle),
        global_waterfill_evidence=_load_json_mapping(
            args.global_waterfill_evidence,
            "global-waterfill evidence",
        ),
    )
    print(json.dumps({"receipt": str(output_root / "receipt.json"), "b2": receipt["b2"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
