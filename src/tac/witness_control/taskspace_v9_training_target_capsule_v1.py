# SPDX-License-Identifier: MIT
"""Strict batch-16 source targets for fresh V9 witness training.

G46 owns the fresh upstream-default SegNet labels.  This companion materializer
replays the same chronological source population in the same 16-pair geometry,
derives the SegNet top1-minus-top2 margin and the first six PoseNet outputs from
one joint batch callback, and refuses the batch before commit if its SegNet
argmax differs from G46.

Everything produced here is encoder-only training evidence.  Dense labels,
margins, pose targets, and scorer weights are explicitly forbidden from
candidate payloads.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

CONFIG_SCHEMA: Final = "tac.taskspace_v9_training_target_capsule_config.v1"
PREFLIGHT_SCHEMA: Final = "tac.taskspace_v9_training_target_capsule_preflight.v1"
BATCH_SCHEMA: Final = "tac.taskspace_v9_training_target_capsule_batch.v1"
AGGREGATE_SCHEMA: Final = "tac.taskspace_v9_training_target_capsule_aggregate.v1"
CLEANUP_SCHEMA: Final = "tac.taskspace_v9_training_target_capsule_cleanup.v1"

EVIDENCE_AXIS: Final = "[macOS-CPU encoder-only upstream-batch16 frozen-scorer evidence]"
PRODUCTION_PAIR_COUNT: Final = 600
PRODUCTION_BATCH_PAIRS: Final = 16
PRODUCTION_CAMERA_HW: Final = (874, 1164)
PRODUCTION_SEG_HW: Final = (384, 512)
PRODUCTION_CLASS_COUNT: Final = 5
POSE_DIM: Final = 6
DEFAULT_SAFETY_RESERVE_BYTES: Final = 8 * (1 << 30)
SSD_ROOTS: Final = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)

U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]


class V9TrainingTargetCapsuleError(RuntimeError):
    """A source, scorer, G46, storage, resume, or capsule check failed."""


@dataclass(frozen=True, slots=True)
class ScoredSourceBatchV1:
    """Outputs from one joint source-batch scorer callback."""

    seg_logits_f32: F32
    source_pose6_f32: F32
    segnet_input_sha256: str
    posenet_input_sha256: str


@dataclass(frozen=True, slots=True)
class V9TrainingTargetsV1:
    """Read-only, path-backed tensors reopened by the strict loader."""

    seg_labels_u8: np.memmap
    seg_top1_minus_top2_margin_f32: np.memmap
    source_pose6_f32: np.memmap


ScoreSourceBatch = Callable[[np.ndarray], ScoredSourceBatchV1]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array_bytes(value: np.ndarray) -> str:
    return sha256_bytes(memoryview(np.ascontiguousarray(value)).cast("B"))


def file_identity(path: str | os.PathLike[str]) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise V9TrainingTargetCapsuleError(f"bound file is a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise V9TrainingTargetCapsuleError(f"bound file is absent, non-regular, or a symlink: {resolved}")
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V9TrainingTargetCapsuleError(f"{label} must be a lowercase SHA-256")
    return value


def _require_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise V9TrainingTargetCapsuleError(f"{label} must be an exact integer in [{minimum},{maximum}]")
    return value


def _seal(body: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if field in body:
        raise V9TrainingTargetCapsuleError(f"payload already contains {field}")
    return {**body, field: payload_sha256(body)}


def _verify_seal(value: Mapping[str, Any], *, field: str) -> None:
    expected = _require_sha(value.get(field), field)
    body = {key: item for key, item in value.items() if key != field}
    if payload_sha256(body) != expected:
        raise V9TrainingTargetCapsuleError(f"{field} canonical hash differs")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V9TrainingTargetCapsuleError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise V9TrainingTargetCapsuleError(f"{label} is not a JSON object")
    return value


def require_ssd_output_root(
    output_root: Path,
    *,
    allowed_roots: Sequence[Path] = SSD_ROOTS,
) -> Path:
    resolved = output_root.expanduser().resolve()
    roots = tuple(root.expanduser().resolve() for root in allowed_roots)
    if not roots or not any(resolved != root and resolved.is_relative_to(root) for root in roots):
        expected = ", ".join(str(root) for root in roots)
        raise V9TrainingTargetCapsuleError(
            f"capsule output must be a child of an SSD evidence root ({expected}), got {resolved}"
        )
    return resolved


def projected_materialization_bytes(
    *,
    pair_count: int,
    seg_hw: Sequence[int],
    safety_reserve_bytes: int = DEFAULT_SAFETY_RESERVE_BYTES,
) -> int:
    pairs = _require_int(pair_count, "pair_count", 1, PRODUCTION_PAIR_COUNT)
    if len(seg_hw) != 2 or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in seg_hw):
        raise V9TrainingTargetCapsuleError("seg_hw must be exact positive H,W")
    reserve = _require_int(
        safety_reserve_bytes,
        "safety_reserve_bytes",
        0,
        1 << 60,
    )
    pixels = pairs * int(seg_hw[0]) * int(seg_hw[1])
    labels = pixels
    margins = 4 * pixels
    poses = 4 * pairs * POSE_DIM
    # Batch shards + aggregate raw + deterministic NPZ + one in-flight NPZ.
    materialized = 2 * (labels + margins + poses) + 2 * (labels + margins + poses)
    metadata = max(256 << 20, pairs * 64_000)
    return materialized + metadata + reserve


def storage_preflight(
    output_root: Path,
    *,
    required_free_bytes: int,
    test_only_small_fixture: bool = False,
    allowed_roots: Sequence[Path] = SSD_ROOTS,
) -> dict[str, Any]:
    output = (
        output_root.expanduser().resolve()
        if test_only_small_fixture
        else require_ssd_output_root(output_root, allowed_roots=allowed_roots)
    )
    required = _require_int(
        required_free_bytes,
        "required_free_bytes",
        1,
        1 << 60,
    )
    anchor = output
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    free = int(shutil.disk_usage(anchor).free)
    if free < required:
        raise V9TrainingTargetCapsuleError(
            f"storage preflight refused: need {required} bytes, only {free} free at {anchor}"
        )
    return {
        "schema": "tac.taskspace_v9_training_target_capsule_storage_preflight.v1",
        "status": "PASS",
        "output_root": str(output),
        "filesystem_anchor": str(anchor),
        "required_free_bytes": required,
        "observed_free_bytes": free,
        "test_only_small_fixture": test_only_small_fixture,
        "cleanup_policy": (
            "success_scratch_deleted_automatically; crash_scratch_sha256_certified_before_cleanup; "
            "durable batches and aggregate preserved until cold-store certificate"
        ),
    }


def _scratch_root(output_root: Path) -> Path:
    return output_root / ".scratch"


def _output_root_for_path(path: Path) -> Path:
    stage_directories = {
        "01_cleanup_receipts",
        "10_batch_shards",
        "10_batch_checkpoints",
        "20_aggregate",
    }
    return path.parent.parent if path.parent.name in stage_directories else path.parent


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = _scratch_root(_output_root_for_path(path))
    scratch.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f"{path.name}.tmp.", dir=scratch)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if file_identity(path)["sha256"] != sha256_bytes(payload):
                raise V9TrainingTargetCapsuleError(f"immutable path already contains different bytes: {path}") from None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_immutable_bytes(path, canonical_json_bytes(value) + b"\n")


def certify_and_cleanup_scratch(
    output_root: Path,
    *,
    preflight_sha256: str,
) -> dict[str, Any] | None:
    """Certify and remove only uncommitted files in this materializer's scratch dir."""

    scratch = _scratch_root(output_root)
    if not scratch.exists():
        return None
    if scratch.is_symlink() or not scratch.is_dir():
        raise V9TrainingTargetCapsuleError("capsule scratch path is not a real directory")
    files = sorted(path for path in scratch.iterdir() if path.is_file() and not path.is_symlink())
    unexpected = [path for path in scratch.iterdir() if path not in files]
    if unexpected:
        raise V9TrainingTargetCapsuleError("capsule scratch contains a symlink or nested directory")
    if not files:
        return None
    rows = [file_identity(path) for path in files]
    body = {
        "schema": CLEANUP_SCHEMA,
        "preflight_sha256": _require_sha(preflight_sha256, "preflight_sha256"),
        "output_root": str(output_root.resolve()),
        "scratch_files": rows,
        "reason": "uncommitted atomic scratch; deterministically rebuildable from sealed inputs",
        "durable_batch_or_aggregate_deleted": False,
    }
    receipt = _seal(body, field="cleanup_receipt_sha256")
    receipt_path = output_root / "01_cleanup_receipts" / f"cleanup_{receipt['cleanup_receipt_sha256'][:16]}.json"
    write_immutable_json(receipt_path, receipt)
    for path in files:
        path.unlink()
    return {**file_identity(receipt_path), "cleanup_receipt_sha256": receipt["cleanup_receipt_sha256"]}


def seal_preflight(body: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "run_id",
        "evidence_axis",
        "research_only",
        "encoder_only",
        "score_claim",
        "candidate_claim",
        "promotion_eligible",
        "pointer_mutation_allowed",
        "dense_targets_candidate_payload_allowed",
        "scorer_weights_candidate_payload_allowed",
        "output_root",
        "pair_count",
        "batch_pairs",
        "camera_hw",
        "seg_hw",
        "class_count",
        "pose_dim",
        "seed",
        "num_threads",
        "test_only_small_fixture",
        "storage_preflight",
        "config",
        "g46_custody",
        "source_custody",
        "scorer_custody",
        "runtime_custody",
        "sealed_input_files",
        "run_argv",
        "resume_contract",
        "cleanup_contract",
    }
    if set(body) != required or body.get("schema") != PREFLIGHT_SCHEMA:
        raise V9TrainingTargetCapsuleError("preflight body keys/schema differ")
    return _seal(body, field="preflight_sha256")


def reverify_preflight(
    preflight: Mapping[str, Any],
    *,
    allowed_roots: Sequence[Path] = SSD_ROOTS,
) -> None:
    if preflight.get("schema") != PREFLIGHT_SCHEMA:
        raise V9TrainingTargetCapsuleError("preflight schema differs")
    _verify_seal(preflight, field="preflight_sha256")
    false_fields = (
        "score_claim",
        "candidate_claim",
        "promotion_eligible",
        "pointer_mutation_allowed",
        "dense_targets_candidate_payload_allowed",
        "scorer_weights_candidate_payload_allowed",
    )
    if (
        preflight.get("research_only") is not True
        or preflight.get("encoder_only") is not True
        or any(preflight.get(field) is not False for field in false_fields)
    ):
        raise V9TrainingTargetCapsuleError("preflight weakened false-authority fences")
    pair_count = _require_int(
        preflight.get("pair_count"),
        "pair_count",
        1,
        PRODUCTION_PAIR_COUNT,
    )
    batch_pairs = _require_int(
        preflight.get("batch_pairs"),
        "batch_pairs",
        1,
        PRODUCTION_BATCH_PAIRS,
    )
    camera_hw = preflight.get("camera_hw")
    seg_hw = preflight.get("seg_hw")
    if (
        not isinstance(camera_hw, list)
        or len(camera_hw) != 2
        or not isinstance(seg_hw, list)
        or len(seg_hw) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in [*camera_hw, *seg_hw])
    ):
        raise V9TrainingTargetCapsuleError("preflight camera/scorer geometry differs")
    test_only = preflight.get("test_only_small_fixture") is True
    if not test_only and (
        pair_count != PRODUCTION_PAIR_COUNT
        or batch_pairs != PRODUCTION_BATCH_PAIRS
        or tuple(camera_hw) != PRODUCTION_CAMERA_HW
        or tuple(seg_hw) != PRODUCTION_SEG_HW
        or preflight.get("class_count") != PRODUCTION_CLASS_COUNT
        or preflight.get("pose_dim") != POSE_DIM
        or preflight.get("evidence_axis") != EVIDENCE_AXIS
    ):
        raise V9TrainingTargetCapsuleError("production geometry is not exact source n600/upstream batch16")
    output = Path(str(preflight.get("output_root"))).expanduser().resolve()
    if not test_only:
        require_ssd_output_root(output, allowed_roots=allowed_roots)
    storage = preflight.get("storage_preflight")
    if not isinstance(storage, Mapping) or storage.get("status") != "PASS" or storage.get("output_root") != str(output):
        raise V9TrainingTargetCapsuleError("preflight storage receipt differs")
    storage_preflight(
        output,
        required_free_bytes=int(storage["required_free_bytes"]),
        test_only_small_fixture=test_only,
        allowed_roots=allowed_roots,
    )
    rows = preflight.get("sealed_input_files")
    if not isinstance(rows, list) or not rows:
        raise V9TrainingTargetCapsuleError("preflight sealed input closure is absent")
    seen: set[str] = set()
    for index, expected in enumerate(rows):
        if (
            not isinstance(expected, Mapping)
            or set(expected) != {"role", "path", "bytes", "sha256"}
            or str(expected.get("path")) in seen
        ):
            raise V9TrainingTargetCapsuleError(f"sealed input row {index} differs")
        seen.add(str(expected["path"]))
        if file_identity(str(expected["path"])) != {key: expected[key] for key in ("path", "bytes", "sha256")}:
            raise V9TrainingTargetCapsuleError(f"sealed input changed: {expected.get('role')}")
    g46 = preflight.get("g46_custody")
    labels = None if not isinstance(g46, Mapping) else g46.get("target_labels")
    if (
        not isinstance(g46, Mapping)
        or not isinstance(labels, Mapping)
        or labels.get("shape") != [pair_count, *seg_hw]
        or labels.get("dtype") != "uint8"
        or labels.get("encoder_only") is not True
        or labels.get("candidate_payload_allowed") is not False
        or file_identity(str(labels.get("path"))) != {key: labels[key] for key in ("path", "bytes", "sha256")}
    ):
        raise V9TrainingTargetCapsuleError("G46 target-label custody differs")
    scorer = preflight.get("scorer_custody")
    if not isinstance(scorer, Mapping):
        raise V9TrainingTargetCapsuleError("scorer custody is absent")
    for name in ("segnet_weights", "posenet_weights"):
        binding = scorer.get(name)
        if not isinstance(binding, Mapping) or file_identity(str(binding.get("path"))) != {
            key: binding[key] for key in ("path", "bytes", "sha256")
        }:
            raise V9TrainingTargetCapsuleError(f"{name} custody differs")
    runtime = preflight.get("runtime_custody")
    runtime_files = None if not isinstance(runtime, Mapping) else runtime.get("files")
    if not isinstance(runtime_files, list) or not runtime_files:
        raise V9TrainingTargetCapsuleError("runtime custody is absent")
    for index, binding in enumerate(runtime_files):
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"role", "path", "bytes", "sha256"}
            or file_identity(str(binding.get("path"))) != {key: binding[key] for key in ("path", "bytes", "sha256")}
            or str(binding.get("path")) not in seen
        ):
            raise V9TrainingTargetCapsuleError(f"runtime custody row {index} differs")
    if not test_only:
        from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (
            load_compile_ready_materialization_receipt,
        )

        receipt_binding = g46.get("receipt")
        if not isinstance(receipt_binding, Mapping):
            raise V9TrainingTargetCapsuleError("G46 receipt binding is absent")
        g46_receipt = load_compile_ready_materialization_receipt(
            Path(str(receipt_binding["path"])),
            allowed_ssd_roots=allowed_roots,
        )
        if (
            file_identity(str(receipt_binding["path"]))
            != {key: receipt_binding[key] for key in ("path", "bytes", "sha256")}
            or g46_receipt.get("receipt_sha256") != g46.get("receipt_sha256")
            or g46_receipt.get("target_labels", {}).get("sha256") != labels.get("sha256")
            or g46_receipt.get("scorer_pair_batch_size") != PRODUCTION_BATCH_PAIRS
        ):
            raise V9TrainingTargetCapsuleError("G46 recursive compile-ready custody differs")
        source = preflight.get("source_custody")
        source_video = None if not isinstance(source, Mapping) else source.get("source_video")
        if (
            not isinstance(source_video, Mapping)
            or source.get("g46_source_identity_equal") is not True
            or {key: source_video.get(key) for key in ("path", "bytes", "sha256")} != g46_receipt.get("source_video")
            or {key: scorer["segnet_weights"].get(key) for key in ("path", "bytes", "sha256")}
            != g46_receipt.get("segnet_weights")
            or scorer.get("upstream_closure") != g46_receipt.get("upstream_closure")
            or scorer.get("model") != "upstream.modules.DistortionNet"
            or scorer.get("segnet_model") != "upstream.modules.SegNet"
            or scorer.get("posenet_model") != "upstream.modules.PoseNet"
            or scorer.get("batch_pairs") != PRODUCTION_BATCH_PAIRS
        ):
            raise V9TrainingTargetCapsuleError("source/scorer custody is not the exact reopened G46 coordinate")
        upstream_root = Path(str(scorer["upstream_closure"]["root"])).resolve()
        if (
            Path(str(scorer["posenet_weights"]["path"])).resolve()
            != (upstream_root / "models/posenet.safetensors").resolve()
        ):
            raise V9TrainingTargetCapsuleError("PoseNet weights are outside the reopened upstream root")
        package_versions = scorer.get("package_versions")
        if (
            not isinstance(package_versions, Mapping)
            or not package_versions
            or any(
                importlib.metadata.version(str(distribution)) != expected
                for distribution, expected in package_versions.items()
            )
            or runtime.get("python") != sys.version.split()[0]
            or runtime.get("upstream_closure_sha256") != scorer["upstream_closure"]["closure_sha256"]
        ):
            raise V9TrainingTargetCapsuleError("runtime/package custody differs")


def _batch_paths(root: Path, start: int, stop: int) -> dict[str, Path]:
    stem = f"pairs_{start:04d}_{stop - 1:04d}"
    return {
        "labels": root / "10_batch_shards" / f"{stem}.seg_labels.u8",
        "margins": root / "10_batch_shards" / f"{stem}.seg_top1_minus_top2_margin.f32",
        "poses": root / "10_batch_shards" / f"{stem}.source_pose6.f32",
        "checkpoint": root / "10_batch_checkpoints" / f"{stem}.json",
    }


def _array_binding(
    path: Path,
    *,
    shape: Sequence[int],
    dtype: str,
) -> dict[str, Any]:
    return {**file_identity(path), "shape": list(shape), "dtype": dtype}


def _open_array(binding: Mapping[str, Any], *, mode: str = "r") -> np.memmap:
    if set(binding) != {"path", "bytes", "sha256", "shape", "dtype"}:
        raise V9TrainingTargetCapsuleError("array binding keys differ")
    dtype_names = {
        "uint8": np.dtype(np.uint8),
        "float32_le": np.dtype("<f4"),
    }
    if binding.get("dtype") not in dtype_names:
        raise V9TrainingTargetCapsuleError("array binding dtype differs")
    shape = binding.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
    ):
        raise V9TrainingTargetCapsuleError("array binding shape differs")
    if file_identity(str(binding.get("path"))) != {key: binding[key] for key in ("path", "bytes", "sha256")}:
        raise V9TrainingTargetCapsuleError("array binding file identity differs")
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype_names[str(binding["dtype"])].itemsize
    if binding.get("bytes") != expected_bytes:
        raise V9TrainingTargetCapsuleError("array binding byte length differs from shape/dtype")
    return np.memmap(
        str(binding["path"]),
        mode=mode,
        dtype=dtype_names[str(binding["dtype"])],
        shape=tuple(shape),
    )


def _derive_batch_targets(
    scored: ScoredSourceBatchV1,
    *,
    batch_size: int,
    class_count: int,
    seg_hw: tuple[int, int],
) -> tuple[U8, F32, F32]:
    logits = np.asarray(scored.seg_logits_f32)
    poses = np.asarray(scored.source_pose6_f32)
    expected_logits = (batch_size, class_count, *seg_hw)
    if logits.dtype != np.float32 or logits.shape != expected_logits:
        raise V9TrainingTargetCapsuleError(
            f"SegNet logits must be float32 {expected_logits}, got {logits.dtype} {logits.shape}"
        )
    if poses.dtype != np.float32 or poses.shape != (batch_size, POSE_DIM):
        raise V9TrainingTargetCapsuleError(
            f"PoseNet targets must be float32 {(batch_size, POSE_DIM)}, got {poses.dtype} {poses.shape}"
        )
    if not bool(np.isfinite(logits).all()) or not bool(np.isfinite(poses).all()):
        raise V9TrainingTargetCapsuleError("scorer outputs contain NaN or infinity")
    _require_sha(scored.segnet_input_sha256, "segnet_input_sha256")
    _require_sha(scored.posenet_input_sha256, "posenet_input_sha256")
    labels = np.argmax(logits, axis=1).astype(np.uint8, copy=False)
    top_two = np.partition(logits, kth=class_count - 2, axis=1)[:, -2:]
    margins = np.subtract(
        top_two[:, 1],
        top_two[:, 0],
        dtype=np.float32,
    )
    if bool(np.any(margins < 0)) or not bool(np.isfinite(margins).all()):
        raise V9TrainingTargetCapsuleError("derived SegNet margins are invalid")
    return (
        np.ascontiguousarray(labels, dtype=np.uint8),
        np.ascontiguousarray(margins, dtype="<f4"),
        np.ascontiguousarray(poses, dtype="<f4"),
    )


def _validate_batch_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    start: int,
    stop: int,
    expected_source_batch_sha256: str | None,
    g46_labels: np.memmap,
) -> dict[str, Any]:
    if checkpoint.get("schema") != BATCH_SCHEMA:
        raise V9TrainingTargetCapsuleError("batch checkpoint schema differs")
    _verify_seal(checkpoint, field="batch_receipt_sha256")
    exact = {
        "preflight_sha256": preflight["preflight_sha256"],
        "pair_range": [start, stop],
        "pair_ids": list(range(start, stop)),
        "scorer_pair_batch_size": stop - start,
        "upstream_default_batch_size": PRODUCTION_BATCH_PAIRS,
        "final_partial_batch": stop - start != PRODUCTION_BATCH_PAIRS,
        "argmax_equal_owned_g46": True,
        "joint_source_batch_products": True,
        "dense_targets_candidate_payload_allowed": False,
        "scorer_weights_candidate_payload_allowed": False,
        "score_claim": False,
        "candidate_claim": False,
        "promotion_eligible": False,
        "committed_atomically": True,
        "immutable_on_resume": True,
        "encoder_only": True,
    }
    if any(checkpoint.get(key) != value for key, value in exact.items()):
        raise V9TrainingTargetCapsuleError("batch checkpoint geometry/authority differs")
    source_sha = _require_sha(
        checkpoint.get("source_pair_batch_sha256"),
        "source_pair_batch_sha256",
    )
    if expected_source_batch_sha256 is not None and source_sha != _require_sha(
        expected_source_batch_sha256,
        "expected_source_batch_sha256",
    ):
        raise V9TrainingTargetCapsuleError("resumed batch source bytes differ")
    _require_sha(checkpoint.get("segnet_input_sha256"), "segnet_input_sha256")
    _require_sha(checkpoint.get("posenet_input_sha256"), "posenet_input_sha256")
    files = checkpoint.get("files")
    if not isinstance(files, Mapping) or set(files) != {"labels", "margins", "poses"}:
        raise V9TrainingTargetCapsuleError("batch files binding differs")
    batch_size = stop - start
    seg_hw = tuple(int(item) for item in preflight["seg_hw"])
    expected_shapes = {
        "labels": ([batch_size, *seg_hw], "uint8"),
        "margins": ([batch_size, *seg_hw], "float32_le"),
        "poses": ([batch_size, POSE_DIM], "float32_le"),
    }
    arrays: dict[str, np.memmap] = {}
    for name, (shape, dtype) in expected_shapes.items():
        binding = files.get(name)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "bytes", "sha256", "shape", "dtype"}
            or binding.get("shape") != shape
            or binding.get("dtype") != dtype
        ):
            raise V9TrainingTargetCapsuleError(f"batch {name} binding differs")
        arrays[name] = _open_array(binding)
    expected_labels = np.asarray(g46_labels[start:stop])
    if (
        checkpoint.get("g46_target_slice_sha256") != sha256_array_bytes(expected_labels)
        or checkpoint.get("observed_argmax_sha256") != sha256_array_bytes(arrays["labels"])
        or not bool(np.array_equal(arrays["labels"], expected_labels))
    ):
        raise V9TrainingTargetCapsuleError("batch argmax no longer equals G46 labels")
    if (
        not bool(np.isfinite(arrays["margins"]).all())
        or bool(np.any(arrays["margins"] < 0))
        or not bool(np.isfinite(arrays["poses"]).all())
    ):
        raise V9TrainingTargetCapsuleError("batch margin/pose bytes are invalid")
    return dict(checkpoint)


def _write_batch(
    *,
    root: Path,
    preflight: Mapping[str, Any],
    start: int,
    stop: int,
    source_batch_sha256: str,
    scored: ScoredSourceBatchV1,
    g46_labels: np.memmap,
) -> dict[str, Any]:
    seg_hw = tuple(int(item) for item in preflight["seg_hw"])
    labels, margins, poses = _derive_batch_targets(
        scored,
        batch_size=stop - start,
        class_count=int(preflight["class_count"]),
        seg_hw=seg_hw,
    )
    expected = np.asarray(g46_labels[start:stop])
    if not bool(np.array_equal(labels, expected)):
        mismatch = np.argwhere(labels != expected)[0].tolist()
        raise V9TrainingTargetCapsuleError(
            f"SegNet argmax differs from G46 at batch-global coordinate "
            f"[{start + mismatch[0]},{mismatch[1]},{mismatch[2]}]"
        )
    paths = _batch_paths(root, start, stop)
    _write_immutable_bytes(paths["labels"], labels.tobytes(order="C"))
    _write_immutable_bytes(paths["margins"], margins.tobytes(order="C"))
    _write_immutable_bytes(paths["poses"], poses.tobytes(order="C"))
    body = {
        "schema": BATCH_SCHEMA,
        "preflight_sha256": preflight["preflight_sha256"],
        "pair_range": [start, stop],
        "pair_ids": list(range(start, stop)),
        "scorer_pair_batch_size": stop - start,
        "upstream_default_batch_size": PRODUCTION_BATCH_PAIRS,
        "final_partial_batch": stop - start != PRODUCTION_BATCH_PAIRS,
        "source_pair_batch_sha256": _require_sha(
            source_batch_sha256,
            "source_pair_batch_sha256",
        ),
        "segnet_input_sha256": _require_sha(
            scored.segnet_input_sha256,
            "segnet_input_sha256",
        ),
        "posenet_input_sha256": _require_sha(
            scored.posenet_input_sha256,
            "posenet_input_sha256",
        ),
        "g46_target_slice_sha256": sha256_array_bytes(expected),
        "observed_argmax_sha256": sha256_array_bytes(labels),
        "argmax_equal_owned_g46": True,
        "joint_source_batch_products": True,
        "files": {
            "labels": _array_binding(
                paths["labels"],
                shape=[stop - start, *seg_hw],
                dtype="uint8",
            ),
            "margins": _array_binding(
                paths["margins"],
                shape=[stop - start, *seg_hw],
                dtype="float32_le",
            ),
            "poses": _array_binding(
                paths["poses"],
                shape=[stop - start, POSE_DIM],
                dtype="float32_le",
            ),
        },
        "committed_atomically": True,
        "immutable_on_resume": True,
        "encoder_only": True,
        "dense_targets_candidate_payload_allowed": False,
        "scorer_weights_candidate_payload_allowed": False,
        "score_claim": False,
        "candidate_claim": False,
        "promotion_eligible": False,
    }
    checkpoint = _seal(body, field="batch_receipt_sha256")
    write_immutable_json(paths["checkpoint"], checkpoint)
    return _validate_batch_checkpoint(
        _load_json(paths["checkpoint"], "batch checkpoint"),
        preflight=preflight,
        start=start,
        stop=stop,
        expected_source_batch_sha256=source_batch_sha256,
        g46_labels=g46_labels,
    )


def _aggregate_paths(root: Path) -> dict[str, Path]:
    aggregate = root / "20_aggregate"
    return {
        "labels": aggregate / "seg_labels_n600.u8",
        "margins": aggregate / "seg_top1_minus_top2_margin_n600.f32",
        "poses": aggregate / "source_pose6_n600.f32",
        "npz": aggregate / "v9_training_target_capsule.npz",
        "receipt": root / "21_v9_training_target_capsule_receipt.json",
    }


def _concatenate_batch_file(
    *,
    root: Path,
    checkpoints: Sequence[Mapping[str, Any]],
    name: str,
    destination: Path,
) -> None:
    scratch = _scratch_root(root)
    scratch.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{destination.name}.tmp.",
        dir=scratch,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            for checkpoint in checkpoints:
                binding = checkpoint["files"][name]
                with Path(str(binding["path"])).open("rb") as source:
                    shutil.copyfileobj(source, output, length=8 << 20)
            output.flush()
            os.fsync(output.fileno())
        payload_sha = sha256_file(temporary)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if file_identity(destination)["sha256"] != payload_sha:
                raise V9TrainingTargetCapsuleError(
                    f"immutable aggregate already contains different bytes: {destination}"
                ) from None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_deterministic_npz(
    *,
    root: Path,
    destination: Path,
    arrays: Sequence[tuple[str, np.ndarray]],
) -> None:
    scratch = _scratch_root(root)
    scratch.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{destination.name}.tmp.",
        dir=scratch,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for name, array in arrays:
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                with archive.open(info, mode="w", force_zip64=True) as member:
                    np.lib.format.write_array(
                        member,
                        np.asanyarray(array),
                        allow_pickle=False,
                    )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        payload_sha = sha256_file(temporary)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if file_identity(destination)["sha256"] != payload_sha:
                raise V9TrainingTargetCapsuleError(
                    f"immutable NPZ already contains different bytes: {destination}"
                ) from None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _assemble_aggregate(
    *,
    root: Path,
    preflight: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    paths = _aggregate_paths(root)
    for name in ("labels", "margins", "poses"):
        _concatenate_batch_file(
            root=root,
            checkpoints=batches,
            name=name,
            destination=paths[name],
        )
    pair_count = int(preflight["pair_count"])
    seg_hw = tuple(int(item) for item in preflight["seg_hw"])
    raw = {
        "labels": _array_binding(
            paths["labels"],
            shape=[pair_count, *seg_hw],
            dtype="uint8",
        ),
        "margins": _array_binding(
            paths["margins"],
            shape=[pair_count, *seg_hw],
            dtype="float32_le",
        ),
        "poses": _array_binding(
            paths["poses"],
            shape=[pair_count, POSE_DIM],
            dtype="float32_le",
        ),
    }
    g46 = preflight["g46_custody"]["target_labels"]
    if raw["labels"]["sha256"] != g46["sha256"]:
        raise V9TrainingTargetCapsuleError("aggregate observed labels do not equal the G46 bank")
    arrays = {name: _open_array(binding) for name, binding in raw.items()}
    npz_members = (
        ("seg_labels_u8", arrays["labels"]),
        ("seg_top1_minus_top2_margin_f32", arrays["margins"]),
        ("source_pose6_f32", arrays["poses"]),
    )
    _write_deterministic_npz(root=root, destination=paths["npz"], arrays=npz_members)
    batch_rows: list[dict[str, Any]] = []
    chain = hashlib.sha256()
    for checkpoint in batches:
        start, stop = checkpoint["pair_range"]
        path = _batch_paths(root, start, stop)["checkpoint"]
        chain.update(bytes.fromhex(checkpoint["batch_receipt_sha256"]))
        batch_rows.append(
            {
                **file_identity(path),
                "pair_range": [start, stop],
                "batch_receipt_sha256": checkpoint["batch_receipt_sha256"],
                "digest_chain_sha256": chain.hexdigest(),
            }
        )
    preflight_path = root / "00_preflight_receipt.json"
    body = {
        "schema": AGGREGATE_SCHEMA,
        "run_id": preflight["run_id"],
        "evidence_axis": preflight["evidence_axis"],
        "preflight": file_identity(preflight_path),
        "preflight_sha256": preflight["preflight_sha256"],
        "g46_custody": preflight["g46_custody"],
        "source_custody": preflight["source_custody"],
        "scorer_custody": preflight["scorer_custody"],
        "pair_count": pair_count,
        "batch_pairs": int(preflight["batch_pairs"]),
        "batch_count": len(batches),
        "camera_hw": preflight["camera_hw"],
        "seg_hw": preflight["seg_hw"],
        "class_count": preflight["class_count"],
        "pose_dim": preflight["pose_dim"],
        "batches": batch_rows,
        "batch_digest_chain_sha256": chain.hexdigest(),
        "raw_arrays": raw,
        "npz": {
            **file_identity(paths["npz"]),
            "member_order": [name for name, _array in npz_members],
            "member_array_sha256": {name: sha256_array_bytes(array) for name, array in npz_members},
        },
        "coverage": {
            "pair_range": [0, pair_count],
            "chronological_contiguous": True,
            "full_upstream_batch16_geometry": int(preflight["batch_pairs"]) == PRODUCTION_BATCH_PAIRS,
            "final_partial_batch_size": pair_count % int(preflight["batch_pairs"]) or int(preflight["batch_pairs"]),
            "all_argmax_equal_owned_g46": True,
            "seg_margin_definition": "top1_logit_minus_top2_logit",
            "pose_target_definition": "upstream.modules.PoseNet pose head first 6 outputs",
            "same_source_batch_callback_for_seg_and_pose": True,
        },
        "research_only": True,
        "encoder_only": True,
        "dense_targets_candidate_payload_allowed": False,
        "scorer_weights_candidate_payload_allowed": False,
        "score_claim": False,
        "candidate_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "cleanup_certificate": {
            "status": "PRESERVED_NO_DELETE",
            "output_root": str(root.resolve()),
            "rebuild_argv": preflight["run_argv"],
            "preflight_sha256": preflight["preflight_sha256"],
            "policy": "certify_or_block",
            "success_scratch_auto_cleaned": True,
        },
    }
    receipt = _seal(body, field="aggregate_receipt_sha256")
    write_immutable_json(paths["receipt"], receipt)
    return paths["receipt"], receipt


def materialize_v9_training_target_capsule(
    *,
    preflight: Mapping[str, Any],
    source_batches: Iterable[np.ndarray],
    score_source_batch: ScoreSourceBatch,
    allowed_roots: Sequence[Path] = SSD_ROOTS,
) -> tuple[Path, dict[str, Any]]:
    """Resume batch products and assemble a strict V9 training target capsule."""

    reverify_preflight(preflight, allowed_roots=allowed_roots)
    root = Path(str(preflight["output_root"])).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    certify_and_cleanup_scratch(
        root,
        preflight_sha256=str(preflight["preflight_sha256"]),
    )
    preflight_path = root / "00_preflight_receipt.json"
    write_immutable_json(preflight_path, preflight)
    g46_binding = preflight["g46_custody"]["target_labels"]
    g46_labels = np.memmap(
        str(g46_binding["path"]),
        mode="r",
        dtype=np.uint8,
        shape=tuple(g46_binding["shape"]),
    )
    pair_count = int(preflight["pair_count"])
    batch_pairs = int(preflight["batch_pairs"])
    camera_hw = tuple(int(item) for item in preflight["camera_hw"])
    observed = 0
    batches: list[dict[str, Any]] = []
    for source_batch_value in source_batches:
        source_batch = np.ascontiguousarray(source_batch_value)
        if (
            source_batch.dtype != np.uint8
            or source_batch.ndim != 5
            or tuple(source_batch.shape[1:]) != (2, *camera_hw, 3)
        ):
            raise V9TrainingTargetCapsuleError(f"source batch must be uint8 (B,2,{camera_hw[0]},{camera_hw[1]},3)")
        if observed >= pair_count:
            raise V9TrainingTargetCapsuleError("source population contains extra chronological pairs")
        expected_size = min(batch_pairs, pair_count - observed)
        if int(source_batch.shape[0]) != expected_size:
            raise V9TrainingTargetCapsuleError(
                f"source batch geometry differs at {observed}: {source_batch.shape[0]} != {expected_size}"
            )
        start, stop = observed, observed + expected_size
        source_sha = sha256_array_bytes(source_batch)
        paths = _batch_paths(root, start, stop)
        if paths["checkpoint"].exists():
            checkpoint = _validate_batch_checkpoint(
                _load_json(paths["checkpoint"], "batch checkpoint"),
                preflight=preflight,
                start=start,
                stop=stop,
                expected_source_batch_sha256=source_sha,
                g46_labels=g46_labels,
            )
        else:
            checkpoint = _write_batch(
                root=root,
                preflight=preflight,
                start=start,
                stop=stop,
                source_batch_sha256=source_sha,
                scored=score_source_batch(source_batch),
                g46_labels=g46_labels,
            )
        batches.append(checkpoint)
        observed = stop
    if observed != pair_count:
        raise V9TrainingTargetCapsuleError(f"source population ended at {observed}, expected {pair_count} pairs")
    receipt_path, receipt = _assemble_aggregate(
        root=root,
        preflight=preflight,
        batches=batches,
    )
    loader = V9TrainingTargetCapsuleLoaderV1.open(
        receipt_path,
        expected_sha256=sha256_file(receipt_path),
        allowed_roots=allowed_roots,
    )
    if loader.pair_count != pair_count:
        raise V9TrainingTargetCapsuleError("strict reopen lost target pairs")
    cleanup = certify_and_cleanup_scratch(
        root,
        preflight_sha256=str(preflight["preflight_sha256"]),
    )
    if cleanup is not None:
        raise V9TrainingTargetCapsuleError("successful materialization unexpectedly left scratch requiring cleanup")
    return receipt_path, receipt


class V9TrainingTargetCapsuleLoaderV1:
    """Strict receipt, custody, batch, raw-array, and NPZ loader."""

    def __init__(
        self,
        receipt_path: Path,
        receipt: Mapping[str, Any],
        *,
        allowed_roots: Sequence[Path],
    ) -> None:
        self.receipt_path = receipt_path
        self.receipt = dict(receipt)
        self.allowed_roots = tuple(allowed_roots)
        self._validate()

    @classmethod
    def open(
        cls,
        aggregate_receipt_path: str | os.PathLike[str],
        *,
        expected_sha256: str,
        allowed_roots: Sequence[Path] = SSD_ROOTS,
    ) -> V9TrainingTargetCapsuleLoaderV1:
        candidate = Path(aggregate_receipt_path).expanduser()
        if candidate.is_symlink():
            raise V9TrainingTargetCapsuleError("aggregate receipt path is a symlink")
        path = candidate.resolve()
        if sha256_file(path) != _require_sha(expected_sha256, "expected_sha256"):
            raise V9TrainingTargetCapsuleError("aggregate receipt file SHA-256 differs")
        return cls(
            path,
            _load_json(path, "aggregate receipt"),
            allowed_roots=allowed_roots,
        )

    def _validate(self) -> None:
        if self.receipt.get("schema") != AGGREGATE_SCHEMA:
            raise V9TrainingTargetCapsuleError("aggregate schema differs")
        _verify_seal(self.receipt, field="aggregate_receipt_sha256")
        if (
            self.receipt.get("research_only") is not True
            or self.receipt.get("encoder_only") is not True
            or self.receipt.get("dense_targets_candidate_payload_allowed") is not False
            or self.receipt.get("scorer_weights_candidate_payload_allowed") is not False
            or self.receipt.get("score_claim") is not False
            or self.receipt.get("candidate_claim") is not False
            or self.receipt.get("promotion_eligible") is not False
            or self.receipt.get("pointer_moved") is not False
        ):
            raise V9TrainingTargetCapsuleError("aggregate false-authority fences differ")
        preflight_binding = self.receipt.get("preflight")
        if not isinstance(preflight_binding, Mapping):
            raise V9TrainingTargetCapsuleError("aggregate preflight binding is absent")
        preflight_path = Path(str(preflight_binding.get("path")))
        if file_identity(preflight_path) != {key: preflight_binding[key] for key in ("path", "bytes", "sha256")}:
            raise V9TrainingTargetCapsuleError("aggregate preflight file differs")
        self.preflight = _load_json(preflight_path, "aggregate preflight")
        reverify_preflight(self.preflight, allowed_roots=self.allowed_roots)
        exact_top = {
            "run_id": self.preflight["run_id"],
            "evidence_axis": self.preflight["evidence_axis"],
            "preflight_sha256": self.preflight["preflight_sha256"],
            "g46_custody": self.preflight["g46_custody"],
            "source_custody": self.preflight["source_custody"],
            "scorer_custody": self.preflight["scorer_custody"],
            "pair_count": self.preflight["pair_count"],
            "batch_pairs": self.preflight["batch_pairs"],
            "camera_hw": self.preflight["camera_hw"],
            "seg_hw": self.preflight["seg_hw"],
            "class_count": self.preflight["class_count"],
            "pose_dim": self.preflight["pose_dim"],
        }
        if any(self.receipt.get(key) != value for key, value in exact_top.items()):
            raise V9TrainingTargetCapsuleError("aggregate top-level custody differs")
        pair_count = int(self.preflight["pair_count"])
        batch_pairs = int(self.preflight["batch_pairs"])
        expected_batch_count = (pair_count + batch_pairs - 1) // batch_pairs
        if self.receipt.get("batch_count") != expected_batch_count:
            raise V9TrainingTargetCapsuleError("aggregate batch count differs")
        g46_binding = self.preflight["g46_custody"]["target_labels"]
        g46_labels = np.memmap(
            str(g46_binding["path"]),
            mode="r",
            dtype=np.uint8,
            shape=tuple(g46_binding["shape"]),
        )
        rows = self.receipt.get("batches")
        if not isinstance(rows, list) or len(rows) != expected_batch_count:
            raise V9TrainingTargetCapsuleError("aggregate batch bindings are absent")
        chain = hashlib.sha256()
        expected_start = 0
        root = Path(str(self.preflight["output_root"]))
        for index, binding in enumerate(rows):
            if not isinstance(binding, Mapping) or set(binding) != {
                "path",
                "bytes",
                "sha256",
                "pair_range",
                "batch_receipt_sha256",
                "digest_chain_sha256",
            }:
                raise V9TrainingTargetCapsuleError(f"aggregate batch binding {index} differs")
            path = Path(str(binding["path"]))
            if file_identity(path) != {key: binding[key] for key in ("path", "bytes", "sha256")}:
                raise V9TrainingTargetCapsuleError(f"aggregate batch file {index} differs")
            start, stop = binding["pair_range"]
            if start != expected_start or stop != min(start + batch_pairs, pair_count):
                raise V9TrainingTargetCapsuleError("aggregate batch chronology differs")
            expected_path = _batch_paths(root, start, stop)["checkpoint"]
            if path != expected_path:
                raise V9TrainingTargetCapsuleError("aggregate batch path differs")
            checkpoint = _validate_batch_checkpoint(
                _load_json(path, f"aggregate batch {index}"),
                preflight=self.preflight,
                start=start,
                stop=stop,
                expected_source_batch_sha256=None,
                g46_labels=g46_labels,
            )
            if checkpoint["batch_receipt_sha256"] != binding["batch_receipt_sha256"]:
                raise V9TrainingTargetCapsuleError("aggregate batch self-hash differs")
            chain.update(bytes.fromhex(checkpoint["batch_receipt_sha256"]))
            if chain.hexdigest() != binding["digest_chain_sha256"]:
                raise V9TrainingTargetCapsuleError("aggregate batch digest chain differs")
            expected_start = stop
        if expected_start != pair_count or chain.hexdigest() != self.receipt.get("batch_digest_chain_sha256"):
            raise V9TrainingTargetCapsuleError("aggregate batch coverage/root differs")
        raw = self.receipt.get("raw_arrays")
        if not isinstance(raw, Mapping) or set(raw) != {"labels", "margins", "poses"}:
            raise V9TrainingTargetCapsuleError("aggregate raw arrays are absent")
        expected_raw = {
            "labels": ([pair_count, *self.preflight["seg_hw"]], "uint8"),
            "margins": ([pair_count, *self.preflight["seg_hw"]], "float32_le"),
            "poses": ([pair_count, POSE_DIM], "float32_le"),
        }
        for name, (shape, dtype) in expected_raw.items():
            binding = raw.get(name)
            if (
                not isinstance(binding, Mapping)
                or set(binding) != {"path", "bytes", "sha256", "shape", "dtype"}
                or binding.get("shape") != shape
                or binding.get("dtype") != dtype
            ):
                raise V9TrainingTargetCapsuleError(f"aggregate raw {name} binding differs")
        self._raw = {name: _open_array(binding) for name, binding in raw.items()}
        if raw["labels"]["sha256"] != g46_binding["sha256"] or not bool(
            np.array_equal(self._raw["labels"], g46_labels)
        ):
            raise V9TrainingTargetCapsuleError("aggregate labels differ from G46")
        npz_binding = self.receipt.get("npz")
        if not isinstance(npz_binding, Mapping) or file_identity(str(npz_binding.get("path"))) != {
            key: npz_binding[key] for key in ("path", "bytes", "sha256")
        }:
            raise V9TrainingTargetCapsuleError("aggregate NPZ identity differs")
        expected_members = [
            "seg_labels_u8",
            "seg_top1_minus_top2_margin_f32",
            "source_pose6_f32",
        ]
        if npz_binding.get("member_order") != expected_members:
            raise V9TrainingTargetCapsuleError("aggregate NPZ member order differs")
        raw_by_member = {
            "seg_labels_u8": self._raw["labels"],
            "seg_top1_minus_top2_margin_f32": self._raw["margins"],
            "source_pose6_f32": self._raw["poses"],
        }
        with np.load(str(npz_binding["path"]), allow_pickle=False) as archive:
            if list(archive.files) != expected_members:
                raise V9TrainingTargetCapsuleError("aggregate NPZ members differ")
            for name in expected_members:
                value = archive[name]
                expected = raw_by_member[name]
                if (
                    value.shape != expected.shape
                    or value.dtype != expected.dtype
                    or not bool(np.array_equal(value, expected))
                    or npz_binding.get("member_array_sha256", {}).get(name) != sha256_array_bytes(expected)
                ):
                    raise V9TrainingTargetCapsuleError(f"aggregate NPZ member {name} differs from raw custody")
        expected_coverage = {
            "pair_range": [0, pair_count],
            "chronological_contiguous": True,
            "full_upstream_batch16_geometry": batch_pairs == PRODUCTION_BATCH_PAIRS,
            "final_partial_batch_size": pair_count % batch_pairs or batch_pairs,
            "all_argmax_equal_owned_g46": True,
            "seg_margin_definition": "top1_logit_minus_top2_logit",
            "pose_target_definition": "upstream.modules.PoseNet pose head first 6 outputs",
            "same_source_batch_callback_for_seg_and_pose": True,
        }
        if self.receipt.get("coverage") != expected_coverage:
            raise V9TrainingTargetCapsuleError("aggregate coverage contract differs")
        expected_cleanup = {
            "status": "PRESERVED_NO_DELETE",
            "output_root": str(Path(str(self.preflight["output_root"])).resolve()),
            "rebuild_argv": self.preflight["run_argv"],
            "preflight_sha256": self.preflight["preflight_sha256"],
            "policy": "certify_or_block",
            "success_scratch_auto_cleaned": True,
        }
        if self.receipt.get("cleanup_certificate") != expected_cleanup:
            raise V9TrainingTargetCapsuleError("aggregate cleanup certificate differs")
        self.pair_count = pair_count
        self.batch_pairs = batch_pairs
        self.seg_hw = tuple(int(item) for item in self.preflight["seg_hw"])
        self.targets = V9TrainingTargetsV1(
            seg_labels_u8=self._raw["labels"],
            seg_top1_minus_top2_margin_f32=self._raw["margins"],
            source_pose6_f32=self._raw["poses"],
        )


__all__ = [
    "AGGREGATE_SCHEMA",
    "BATCH_SCHEMA",
    "CONFIG_SCHEMA",
    "DEFAULT_SAFETY_RESERVE_BYTES",
    "EVIDENCE_AXIS",
    "POSE_DIM",
    "PREFLIGHT_SCHEMA",
    "PRODUCTION_BATCH_PAIRS",
    "PRODUCTION_CAMERA_HW",
    "PRODUCTION_CLASS_COUNT",
    "PRODUCTION_PAIR_COUNT",
    "PRODUCTION_SEG_HW",
    "SSD_ROOTS",
    "ScoredSourceBatchV1",
    "V9TrainingTargetCapsuleError",
    "V9TrainingTargetCapsuleLoaderV1",
    "V9TrainingTargetsV1",
    "canonical_json_bytes",
    "certify_and_cleanup_scratch",
    "file_identity",
    "materialize_v9_training_target_capsule",
    "payload_sha256",
    "projected_materialization_bytes",
    "require_ssd_output_root",
    "reverify_preflight",
    "seal_preflight",
    "sha256_array_bytes",
    "sha256_bytes",
    "sha256_file",
    "storage_preflight",
    "write_immutable_json",
]
