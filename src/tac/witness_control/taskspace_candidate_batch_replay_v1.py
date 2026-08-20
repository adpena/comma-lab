# SPDX-License-Identifier: MIT
"""Resumable exact-batch replay of a realized task-space candidate.

This is the small, reusable closure between a byte-closed archive/raw pair and
the frozen upstream distortion functional.  It intentionally performs no
decode and no optimization: the CLI supplies chronological source/candidate
batches and the exact frozen-scorer callback.  Each scorer batch is an atomic
stage because the frozen networks have rare batch-geometry-dependent outputs.

The production use is the n600, batch-16 replay of the historical C1 full-
lattice witness.  That replay settles whether its very low batch-32 distortion
survives the actual upstream default geometry before the number is used as a
frontier byte budget.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.contest_score import UNCOMPRESSED_SIZE_BYTES, compute_contest_score
from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (
    CAMERA_HW_PUBLIC,
    PAIR_COUNT_PUBLIC,
    SSD_ROOTS,
    UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE,
    atomic_write_json,
    canonical_json_bytes,
    file_identity,
    load_json_mapping,
    payload_sha256,
    require_ssd_output_root,
    sha256_bytes,
    storage_preflight,
    verify_sealed_payload,
)

LEGACY_PREFLIGHT_SCHEMA: Final = "tac.taskspace_candidate_batch_replay_preflight.v1"
PREFLIGHT_SCHEMA: Final = "tac.taskspace_candidate_batch_replay_preflight.v2"
BATCH_SCHEMA: Final = "tac.taskspace_candidate_batch_replay_stage.v1"
LEGACY_RECEIPT_SCHEMA: Final = "tac.taskspace_candidate_batch_replay_receipt.v1"
RECEIPT_SCHEMA: Final = "tac.taskspace_candidate_batch_replay_receipt.v2"
EVIDENCE_AXIS: Final = "[macOS-CPU deterministic batch16 scorer advisory]"
UPSTREAM_CLOSURE_MEMBERS: Final = (
    "evaluate.py",
    "frame_utils.py",
    "modules.py",
    "public_test_video_names.txt",
    "models/segnet.safetensors",
    "models/posenet.safetensors",
)


class CandidateBatchReplayError(ValueError):
    """A custody, geometry, resume, scorer, or arithmetic invariant failed."""


@dataclass(frozen=True)
class BatchDistancesV1:
    """Per-pair distances plus the scorer's own float32 batch reductions."""

    d_pose: np.ndarray
    d_seg: np.ndarray
    d_pose_batch_sum_f32: float
    d_seg_batch_sum_f32: float


ScoreBatch = Callable[[np.ndarray, np.ndarray], BatchDistancesV1]


def _seal(payload: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if field in payload:
        raise CandidateBatchReplayError(f"payload already contains {field}")
    result = dict(payload)
    result[field] = payload_sha256(payload)
    return result


def _closure_identity(upstream_root: Path) -> dict[str, Any]:
    root = upstream_root.resolve()
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for relative in UPSTREAM_CLOSURE_MEMBERS:
        row = {"relative_path": relative, **file_identity(root / relative)}
        rows.append(row)
        digest.update(canonical_json_bytes(row))
        digest.update(b"\n")
    return {"root": str(root), "members": rows, "closure_sha256": digest.hexdigest()}


def _implementation_identity(paths: Sequence[Path]) -> dict[str, Any]:
    resolved = sorted({path.resolve() for path in paths}, key=str)
    if not resolved:
        raise CandidateBatchReplayError("implementation_files must be nonempty")
    rows = [file_identity(path) for path in resolved]
    return {"members": rows, "closure_sha256": payload_sha256(rows)}


def _raw_expected_bytes(pair_count: int, camera_hw: Sequence[int]) -> int:
    if not isinstance(pair_count, int) or isinstance(pair_count, bool) or not 1 <= pair_count <= PAIR_COUNT_PUBLIC:
        raise CandidateBatchReplayError("pair_count must be an exact integer in [1,600]")
    if len(camera_hw) != 2 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in camera_hw
    ):
        raise CandidateBatchReplayError("camera_hw must contain two positive exact integers")
    return pair_count * 2 * int(camera_hw[0]) * int(camera_hw[1]) * 3


def build_candidate_replay_preflight(
    *,
    source_video: Path,
    candidate_archive: Path,
    candidate_raw: Path,
    upstream_root: Path,
    output_root: Path,
    pair_count: int,
    batch_size: int,
    num_threads: int,
    seed: int,
    package_versions: Mapping[str, str],
    implementation_files: Sequence[Path],
    run_argv: Sequence[str],
    expected_candidate_raw_sha256: str | None = None,
    camera_hw: Sequence[int] = CAMERA_HW_PUBLIC,
    safety_reserve_bytes: int = 8 * (1 << 30),
    allowed_ssd_roots: Sequence[Path] = SSD_ROOTS,
) -> dict[str, Any]:
    """Hash-close the exact archive/raw/source/scorer object before replay."""

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= 64:
        raise CandidateBatchReplayError("batch_size must be an exact integer in [1,64]")
    if not isinstance(num_threads, int) or isinstance(num_threads, bool) or not 1 <= num_threads <= 256:
        raise CandidateBatchReplayError("num_threads must be an exact integer in [1,256]")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < (1 << 63):
        raise CandidateBatchReplayError("seed must be a nonnegative exact int64")
    expected_raw_bytes = _raw_expected_bytes(pair_count, camera_hw)
    raw_identity = file_identity(candidate_raw)
    if raw_identity["bytes"] != expected_raw_bytes:
        raise CandidateBatchReplayError(
            f"candidate raw has {raw_identity['bytes']} bytes, expected {expected_raw_bytes}"
        )
    if expected_candidate_raw_sha256 is not None and raw_identity["sha256"] != expected_candidate_raw_sha256:
        raise CandidateBatchReplayError("candidate raw SHA-256 differs from sealed expected identity")
    source_identity = file_identity(source_video)
    if pair_count == PAIR_COUNT_PUBLIC and source_identity["bytes"] != UNCOMPRESSED_SIZE_BYTES:
        raise CandidateBatchReplayError("full-public source byte denominator differs from canonical upstream")
    output = require_ssd_output_root(output_root, allowed_roots=allowed_ssd_roots)
    reserve = int(safety_reserve_bytes)
    if reserve < 0:
        raise CandidateBatchReplayError("safety_reserve_bytes must be nonnegative")
    storage = storage_preflight(
        output,
        required_free_bytes=max(64 << 20, reserve),
        allowed_roots=allowed_ssd_roots,
    )
    versions = {str(key): str(value) for key, value in sorted(package_versions.items())}
    argv = [str(token) for token in run_argv]
    if not versions or not argv or any(not token for token in argv):
        raise CandidateBatchReplayError("package_versions and run_argv must be nonempty")
    body = {
        "schema": PREFLIGHT_SCHEMA,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_mutation_allowed": False,
        "source_video": source_identity,
        "candidate_archive": file_identity(candidate_archive),
        "candidate_raw": raw_identity,
        "upstream_closure": _closure_identity(upstream_root),
        "implementation_closure": _implementation_identity(implementation_files),
        "output_root": str(output),
        "pair_count": pair_count,
        "frame_count": pair_count * 2,
        "camera_hw": [int(camera_hw[0]), int(camera_hw[1])],
        "batch_size": batch_size,
        "upstream_evaluate_default_batch_size": UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE,
        "batch_geometry_matches_upstream_default": batch_size == UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE,
        "last_batch_policy": "PARTIAL_NO_PADDING",
        "num_threads": num_threads,
        "seed": seed,
        "device": "cpu",
        "package_versions": versions,
        "run_argv": argv,
        "storage_preflight": storage,
        "stage_contract": [
            "00_custody_storage_preflight",
            "10_exact_scorer_batch_checkpoints",
            "11_batch16_replay_receipt",
        ],
        "resume_contract": {
            "batch_is_atomic_stage": True,
            "whole_original_batch_reforwarded_if_stage_missing": True,
            "source_and_candidate_batch_hashes_reverified": True,
            "scorer_batch_order_gap_overlap_refused": True,
        },
        "authority_contract": {
            "frozen_recursive_code_and_weights_hash_bound": True,
            "upstream_batch16_input_and_aggregation_geometry_reproduced": True,
            "unmodified_upstream_host_kernel_dispatch_reproduced": False,
            "exact_numeric_upstream_mirror_claim": False,
            "macos_cpu_is_contest_authority": False,
            "archive_and_realized_raw_are_exact_input_bytes": True,
            "targets_or_scorer_weights_enter_candidate": False,
        },
        "determinism_contract": {
            "torch_deterministic_algorithms": True,
            "mkldnn_disabled": True,
            "torch_num_threads": num_threads,
            "torch_num_interop_threads": 1,
        },
    }
    return _seal(body, field="preflight_sha256")


def reverify_candidate_replay_preflight(
    preflight: Mapping[str, Any],
    *,
    allowed_ssd_roots: Sequence[Path] = SSD_ROOTS,
) -> None:
    schema = preflight.get("schema")
    if schema not in (LEGACY_PREFLIGHT_SCHEMA, PREFLIGHT_SCHEMA):
        raise CandidateBatchReplayError("unexpected candidate replay preflight schema")
    verify_sealed_payload(preflight, hash_field="preflight_sha256")
    if any(
        (
            preflight.get("score_claim") is not False,
            preflight.get("promotion_eligible") is not False,
            preflight.get("pointer_mutation_allowed") is not False,
            preflight.get("device") != "cpu",
        )
    ):
        raise CandidateBatchReplayError("preflight weakened advisory authority")
    pair_count = int(preflight.get("pair_count", 0))
    camera_hw = preflight.get("camera_hw")
    if not isinstance(camera_hw, list):
        raise CandidateBatchReplayError("preflight camera_hw is absent")
    expected_raw_bytes = _raw_expected_bytes(pair_count, camera_hw)
    for field in ("source_video", "candidate_archive", "candidate_raw"):
        expected = preflight.get(field)
        if not isinstance(expected, Mapping) or file_identity(Path(str(expected.get("path")))) != dict(expected):
            raise CandidateBatchReplayError(f"preflight {field} identity drifted")
    if int(preflight["candidate_raw"]["bytes"]) != expected_raw_bytes:
        raise CandidateBatchReplayError("preflight candidate raw geometry drifted")
    closure = preflight.get("upstream_closure")
    if not isinstance(closure, Mapping) or _closure_identity(Path(str(closure.get("root")))) != dict(closure):
        raise CandidateBatchReplayError("preflight upstream recursive closure drifted")
    implementation = preflight.get("implementation_closure")
    if schema == PREFLIGHT_SCHEMA:
        if not isinstance(implementation, Mapping):
            raise CandidateBatchReplayError("preflight implementation closure is absent")
        members = implementation.get("members")
        if not isinstance(members, list) or not members:
            raise CandidateBatchReplayError("preflight implementation closure is empty")
        paths = [Path(str(member.get("path"))) for member in members if isinstance(member, Mapping)]
        if len(paths) != len(members) or _implementation_identity(paths) != dict(implementation):
            raise CandidateBatchReplayError("preflight implementation closure drifted")
    batch_size = int(preflight.get("batch_size", 0))
    expected_match = batch_size == UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE
    if (
        preflight.get("upstream_evaluate_default_batch_size") != UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE
        or preflight.get("batch_geometry_matches_upstream_default") is not expected_match
        or preflight.get("last_batch_policy") != "PARTIAL_NO_PADDING"
    ):
        raise CandidateBatchReplayError("preflight scorer batch geometry drifted")
    output = require_ssd_output_root(Path(str(preflight.get("output_root"))), allowed_roots=allowed_ssd_roots)
    storage = preflight.get("storage_preflight")
    if not isinstance(storage, Mapping) or storage.get("status") != "PASS":
        raise CandidateBatchReplayError("preflight storage admission is absent")
    storage_preflight(
        output,
        required_free_bytes=int(storage.get("required_free_bytes", 0)),
        allowed_roots=allowed_ssd_roots,
    )


def _array_sha256(value: np.ndarray) -> str:
    return sha256_bytes(memoryview(np.ascontiguousarray(value)).cast("B"))


def _stage_path(output_root: Path, batch_index: int) -> Path:
    return output_root / "10_batch_checkpoints" / f"batch_{batch_index:04d}.json"


def _finite_vector(value: Any, *, count: int, label: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (count,) or bool(np.any(~np.isfinite(array))) or bool(np.any(array < 0)):
        raise CandidateBatchReplayError(f"{label} must be a finite nonnegative vector of length {count}")
    return [float(item) for item in array]


def _validate_stage(
    stage: Mapping[str, Any],
    *,
    path: Path,
    preflight_sha256: str,
    batch_index: int,
    pair_start: int,
    pair_stop: int,
    source_sha256: str,
    candidate_sha256: str,
) -> dict[str, Any]:
    if stage.get("schema") != BATCH_SCHEMA:
        raise CandidateBatchReplayError(f"batch stage schema drifted: {path}")
    verify_sealed_payload(stage, hash_field="stage_sha256")
    expected = {
        "preflight_sha256": preflight_sha256,
        "batch_index": batch_index,
        "pair_start": pair_start,
        "pair_stop_exclusive": pair_stop,
        "source_batch_sha256": source_sha256,
        "candidate_batch_sha256": candidate_sha256,
    }
    for field, value in expected.items():
        if stage.get(field) != value:
            raise CandidateBatchReplayError(f"batch {batch_index} {field} drifted")
    count = pair_stop - pair_start
    _finite_vector(stage.get("d_pose_per_pair"), count=count, label="d_pose_per_pair")
    _finite_vector(stage.get("d_seg_per_pair"), count=count, label="d_seg_per_pair")
    for field in ("d_pose_batch_sum_f32", "d_seg_batch_sum_f32"):
        value = stage.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise CandidateBatchReplayError(f"batch {batch_index} has invalid {field}")
    return dict(stage)


def _score_and_write_stage(
    *,
    output_root: Path,
    preflight_sha256: str,
    batch_index: int,
    pair_start: int,
    source_batch: np.ndarray,
    candidate_batch: np.ndarray,
    source_sha256: str,
    candidate_sha256: str,
    score_batch: ScoreBatch,
) -> dict[str, Any]:
    result = score_batch(source_batch, candidate_batch)
    count = int(source_batch.shape[0])
    pose = _finite_vector(result.d_pose, count=count, label="d_pose")
    seg = _finite_vector(result.d_seg, count=count, label="d_seg")
    pose_sum = float(np.float32(result.d_pose_batch_sum_f32))
    seg_sum = float(np.float32(result.d_seg_batch_sum_f32))
    if not math.isfinite(pose_sum) or not math.isfinite(seg_sum) or pose_sum < 0 or seg_sum < 0:
        raise CandidateBatchReplayError("scorer returned invalid float32 batch sums")
    body = {
        "schema": BATCH_SCHEMA,
        "preflight_sha256": preflight_sha256,
        "batch_index": batch_index,
        "pair_start": pair_start,
        "pair_stop_exclusive": pair_start + count,
        "batch_size": count,
        "source_batch_sha256": source_sha256,
        "candidate_batch_sha256": candidate_sha256,
        "d_pose_per_pair": pose,
        "d_seg_per_pair": seg,
        "d_pose_batch_sum_f32": pose_sum,
        "d_seg_batch_sum_f32": seg_sum,
        "scorer_forward_geometry": "WHOLE_ORIGINAL_BATCH",
        "committed_atomically": True,
    }
    stage = _seal(body, field="stage_sha256")
    path = _stage_path(output_root, batch_index)
    atomic_write_json(path, stage)
    return _validate_stage(
        load_json_mapping(path),
        path=path,
        preflight_sha256=preflight_sha256,
        batch_index=batch_index,
        pair_start=pair_start,
        pair_stop=pair_start + count,
        source_sha256=source_sha256,
        candidate_sha256=candidate_sha256,
    )


def _f32_upstream_mean(stages: Sequence[Mapping[str, Any]], field: str, pair_count: int) -> float:
    total = np.float32(0.0)
    for stage in stages:
        total = np.float32(total + np.float32(stage[field]))
    return float(np.float32(total / np.float32(pair_count)))


def replay_candidate_batches(
    *,
    preflight: Mapping[str, Any],
    paired_batches: Iterable[tuple[np.ndarray, np.ndarray]],
    score_batch: ScoreBatch,
    allowed_ssd_roots: Sequence[Path] = SSD_ROOTS,
) -> dict[str, Any]:
    """Run or resume every exact scorer batch and seal the aggregate receipt."""

    reverify_candidate_replay_preflight(preflight, allowed_ssd_roots=allowed_ssd_roots)
    output_root = Path(str(preflight["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    stage_zero = output_root / "00_custody_storage_preflight.json"
    if stage_zero.exists():
        if load_json_mapping(stage_zero) != dict(preflight):
            raise CandidateBatchReplayError("stage-00 preflight names another run")
    else:
        atomic_write_json(stage_zero, dict(preflight))
    pair_count = int(preflight["pair_count"])
    batch_size = int(preflight["batch_size"])
    camera_hw = tuple(int(value) for value in preflight["camera_hw"])
    preflight_sha = str(preflight["preflight_sha256"])
    stages: list[dict[str, Any]] = []
    pair_start = 0
    for batch_index, (source_value, candidate_value) in enumerate(paired_batches):
        # Own writable buffers before handing them to torch.  A contiguous
        # read-only memmap view is otherwise preserved by ascontiguousarray.
        source = np.array(source_value, copy=True, order="C")
        candidate = np.array(candidate_value, copy=True, order="C")
        expected_count = min(batch_size, pair_count - pair_start)
        expected_shape = (expected_count, 2, camera_hw[0], camera_hw[1], 3)
        if source.dtype != np.uint8 or candidate.dtype != np.uint8:
            raise CandidateBatchReplayError("source and candidate batches must be uint8")
        if source.shape != expected_shape or candidate.shape != expected_shape:
            raise CandidateBatchReplayError(
                f"batch {batch_index} changed exact scorer geometry: {source.shape}, {candidate.shape}, expected {expected_shape}"
            )
        source_sha = _array_sha256(source)
        candidate_sha = _array_sha256(candidate)
        path = _stage_path(output_root, batch_index)
        if path.exists():
            stage = _validate_stage(
                load_json_mapping(path),
                path=path,
                preflight_sha256=preflight_sha,
                batch_index=batch_index,
                pair_start=pair_start,
                pair_stop=pair_start + expected_count,
                source_sha256=source_sha,
                candidate_sha256=candidate_sha,
            )
        else:
            stage = _score_and_write_stage(
                output_root=output_root,
                preflight_sha256=preflight_sha,
                batch_index=batch_index,
                pair_start=pair_start,
                source_batch=source,
                candidate_batch=candidate,
                source_sha256=source_sha,
                candidate_sha256=candidate_sha,
                score_batch=score_batch,
            )
        stages.append(stage)
        pair_start += expected_count
    if pair_start != pair_count:
        raise CandidateBatchReplayError(f"paired stream ended at {pair_start}, expected {pair_count}")
    expected_batches = (pair_count + batch_size - 1) // batch_size
    if len(stages) != expected_batches:
        raise CandidateBatchReplayError("paired stream has a batch gap, overlap, or trailing batch")
    d_pose = _f32_upstream_mean(stages, "d_pose_batch_sum_f32", pair_count)
    d_seg = _f32_upstream_mean(stages, "d_seg_batch_sum_f32", pair_count)
    archive_bytes = int(preflight["candidate_archive"]["bytes"])
    uncompressed_bytes = int(preflight["source_video"]["bytes"])
    score = compute_contest_score(
        d_seg,
        d_pose,
        archive_bytes,
        uncompressed_size=uncompressed_bytes,
    )
    stage_root = payload_sha256(
        [
            {
                "batch_index": stage["batch_index"],
                "pair_start": stage["pair_start"],
                "pair_stop_exclusive": stage["pair_stop_exclusive"],
                "stage_sha256": stage["stage_sha256"],
            }
            for stage in stages
        ]
    )
    body = {
        "schema": RECEIPT_SCHEMA,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_mutation_allowed": False,
        "preflight_sha256": preflight_sha,
        "source_video": preflight["source_video"],
        "candidate_archive": preflight["candidate_archive"],
        "candidate_raw": preflight["candidate_raw"],
        "upstream_closure": preflight["upstream_closure"],
        "implementation_closure": preflight.get("implementation_closure"),
        "pair_count": pair_count,
        "frame_count": pair_count * 2,
        "batch_size": batch_size,
        "batch_geometry_matches_upstream_default": preflight["batch_geometry_matches_upstream_default"],
        "last_batch_policy": preflight["last_batch_policy"],
        "batch_stage_count": len(stages),
        "batch_stage_root_sha256": stage_root,
        "batch_stages": stages,
        "distortion": {
            "d_pose": d_pose,
            "d_seg": d_seg,
            "aggregation_dag": "torch_float32_batch_sum_then_sequential_float32_sum_divide_pair_count",
            "realized_uint8_raw": True,
        },
        "rate": {
            "archive_bytes": archive_bytes,
            "uncompressed_bytes": uncompressed_bytes,
            "rate": float(archive_bytes) / uncompressed_bytes,
        },
        "advisory_score": score,
        "authority": {
            "exact_archive_and_raw_bytes": True,
            "frozen_upstream_recursive_closure_hash_bound": True,
            "implementation_closure_hash_bound": preflight.get("implementation_closure") is not None,
            "deterministic_local_kernel_controls_applied": True,
            "exact_numeric_upstream_host_mirror": False,
            "contest_linux_cpu_or_cuda_replay": False,
            "frontier_or_promotion_claim": False,
        },
        "cleanup_certificate": {
            "status": "PRESERVED_NO_DELETE",
            "output_root": str(output_root.resolve()),
            "rebuild_argv": preflight["run_argv"],
            "candidate_inputs_mutated": False,
        },
        "next_consumer": (
            "use this batch-16 low-distortion anchor for coupled quotient byte allocation only if "
            "batch_geometry_matches_upstream_default is true; candidate promotion still requires exact contest CPU/CUDA"
        ),
    }
    receipt = _seal(body, field="receipt_sha256")
    final_path = output_root / "11_batch_replay_receipt.json"
    atomic_write_json(final_path, receipt)
    reopened = load_json_mapping(final_path)
    verify_sealed_payload(reopened, hash_field="receipt_sha256")
    if reopened != receipt:
        raise CandidateBatchReplayError("final replay receipt changed across parse-back")
    return receipt


def load_and_reverify_candidate_replay_receipt(
    path: Path,
    *,
    allowed_ssd_roots: Sequence[Path] = SSD_ROOTS,
) -> dict[str, Any]:
    receipt = load_json_mapping(path)
    if receipt.get("schema") not in (LEGACY_RECEIPT_SCHEMA, RECEIPT_SCHEMA):
        raise CandidateBatchReplayError("unexpected replay receipt schema")
    verify_sealed_payload(receipt, hash_field="receipt_sha256")
    output_root = require_ssd_output_root(path.resolve().parent, allowed_roots=allowed_ssd_roots)
    preflight = load_json_mapping(output_root / "00_custody_storage_preflight.json")
    reverify_candidate_replay_preflight(preflight, allowed_ssd_roots=allowed_ssd_roots)
    if preflight.get("preflight_sha256") != receipt.get("preflight_sha256"):
        raise CandidateBatchReplayError("receipt names another preflight")
    stages = receipt.get("batch_stages")
    if not isinstance(stages, list):
        raise CandidateBatchReplayError("receipt lacks batch stages")
    stage_root = payload_sha256(
        [
            {
                "batch_index": stage["batch_index"],
                "pair_start": stage["pair_start"],
                "pair_stop_exclusive": stage["pair_stop_exclusive"],
                "stage_sha256": stage["stage_sha256"],
            }
            for stage in stages
        ]
    )
    if stage_root != receipt.get("batch_stage_root_sha256"):
        raise CandidateBatchReplayError("receipt batch-stage root drifted")
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise CandidateBatchReplayError("receipt contains a non-mapping batch stage")
        verify_sealed_payload(stage, hash_field="stage_sha256")
        path_on_disk = _stage_path(output_root, int(stage["batch_index"]))
        if load_json_mapping(path_on_disk) != stage:
            raise CandidateBatchReplayError(f"batch stage changed on disk: {path_on_disk}")
    pair_count = int(receipt.get("pair_count", 0))
    d_pose = _f32_upstream_mean(stages, "d_pose_batch_sum_f32", pair_count)
    d_seg = _f32_upstream_mean(stages, "d_seg_batch_sum_f32", pair_count)
    distortion = receipt.get("distortion")
    if not isinstance(distortion, Mapping) or distortion.get("d_pose") != d_pose or distortion.get("d_seg") != d_seg:
        raise CandidateBatchReplayError("receipt distortion does not recompose from batch stages")
    expected_score = compute_contest_score(
        d_seg,
        d_pose,
        int(receipt["candidate_archive"]["bytes"]),
        uncompressed_size=int(receipt["source_video"]["bytes"]),
    )
    if receipt.get("advisory_score") != expected_score:
        raise CandidateBatchReplayError("receipt score does not recompose from distortion and exact bytes")
    return receipt


__all__ = [
    "BATCH_SCHEMA",
    "LEGACY_PREFLIGHT_SCHEMA",
    "LEGACY_RECEIPT_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "RECEIPT_SCHEMA",
    "BatchDistancesV1",
    "CandidateBatchReplayError",
    "build_candidate_replay_preflight",
    "load_and_reverify_candidate_replay_receipt",
    "replay_candidate_batches",
    "reverify_candidate_replay_preflight",
]
