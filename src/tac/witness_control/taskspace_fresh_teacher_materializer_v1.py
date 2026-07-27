# SPDX-License-Identifier: MIT
"""Fresh, encoder-only SegNet target materialization for the task-space codec.

This module owns the *source-side* target of the first G17 semantic actuator
row.  It deliberately does not know how to build a candidate archive and it
never serializes target labels into a receiver.  Its only job is to turn the
exact contest source video plus the frozen upstream SegNet into a path-backed,
resumable and hash-closed bank of 600 uint8 argmax maps.

The implementation is split in two:

* :func:`build_fresh_teacher_preflight` freezes the source, scorer, upstream
  code closure, package versions, geometry, storage and future run argv.
* :func:`materialize_fresh_teacher_from_batches` is a deterministic streaming
  engine.  It writes one atomic pair shard and checkpoint at a time, reopens
  every committed shard on resume, and only then assembles the aggregate bank.

The public CLI supplies the exact AVVideoDataset/SegNet callback.  Keeping that
callback outside this module makes the resumability and custody mechanics
unit-testable without loading a neural network while preserving one production
scorer implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

SCHEMA: Final = "tac.taskspace_fresh_teacher_materialization.v1"
PREFLIGHT_SCHEMA: Final = "tac.taskspace_fresh_teacher_preflight.v1"
PAIR_CHECKPOINT_SCHEMA: Final = "tac.taskspace_fresh_teacher_pair_checkpoint.v1"
STAGE_RECEIPT_SCHEMA: Final = "tac.taskspace_fresh_teacher_stage_receipt.v1"
EVIDENCE_AXIS: Final = "[macOS-CPU encoder-only frozen-scorer evidence]"
PAIR_COUNT_PUBLIC: Final = 600
CAMERA_HW_PUBLIC: Final = (874, 1164)
SEG_HW_PUBLIC: Final = (384, 512)
CLASS_COUNT: Final = 5
UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE: Final = 16
DEFAULT_SAFETY_RESERVE_BYTES: Final = 8 * (1 << 30)
SSD_ROOTS: Final = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)
UPSTREAM_CLOSURE_MEMBERS: Final = (
    "evaluate.py",
    "frame_utils.py",
    "modules.py",
    "public_test_video_names.txt",
)


class FreshTeacherMaterializationError(ValueError):
    """A source, scorer, storage, resume, geometry or custody check failed."""


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FreshTeacherMaterializationError(f"required file is absent: {resolved}")
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _require_exact_int(value: int, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise FreshTeacherMaterializationError(f"{field} must be an exact integer in [{minimum}, {maximum}]")
    return value


def _require_hw(value: Sequence[int], *, field: str) -> tuple[int, int]:
    if len(value) != 2:
        raise FreshTeacherMaterializationError(f"{field} must contain exactly H,W")
    height = _require_exact_int(value[0], field=f"{field}[0]", minimum=1, maximum=100_000)
    width = _require_exact_int(value[1], field=f"{field}[1]", minimum=1, maximum=100_000)
    return height, width


def _nearest_existing_parent(path: Path) -> Path:
    anchor = path.resolve()
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    return anchor


def require_ssd_output_root(
    output_root: Path,
    *,
    allowed_roots: Sequence[Path] = SSD_ROOTS,
) -> Path:
    resolved = output_root.resolve()
    roots = tuple(root.resolve() for root in allowed_roots)
    if not roots or not any(resolved != root and resolved.is_relative_to(root) for root in roots):
        expected = ", ".join(str(root) for root in roots)
        raise FreshTeacherMaterializationError(
            f"teacher output must be a child of an SSD evidence root ({expected}), got {resolved}"
        )
    return resolved


def projected_materialization_bytes(
    *,
    pair_count: int,
    seg_hw: Sequence[int],
    safety_reserve_bytes: int = DEFAULT_SAFETY_RESERVE_BYTES,
) -> int:
    pairs = _require_exact_int(pair_count, field="pair_count", minimum=1, maximum=PAIR_COUNT_PUBLIC)
    height, width = _require_hw(seg_hw, field="seg_hw")
    reserve = _require_exact_int(
        safety_reserve_bytes,
        field="safety_reserve_bytes",
        minimum=0,
        maximum=1 << 60,
    )
    # One bank in pair shards plus one atomic aggregate bank, checkpoint JSON,
    # and one aggregate temporary copy.  The factor 3 is deliberately explicit.
    label_bank = pairs * height * width
    metadata_margin = max(32 << 20, pairs * 16_384)
    return 3 * label_bank + metadata_margin + reserve


def storage_preflight(
    output_root: Path,
    *,
    required_free_bytes: int,
    allowed_roots: Sequence[Path] = SSD_ROOTS,
) -> dict[str, Any]:
    resolved = require_ssd_output_root(output_root, allowed_roots=allowed_roots)
    required = _require_exact_int(
        required_free_bytes,
        field="required_free_bytes",
        minimum=1,
        maximum=1 << 60,
    )
    anchor = _nearest_existing_parent(resolved)
    usage = shutil.disk_usage(anchor)
    if usage.free < required:
        raise FreshTeacherMaterializationError(
            f"storage preflight refused: need {required} bytes, only {usage.free} free at {anchor}"
        )
    return {
        "schema": "tac.taskspace_fresh_teacher_storage_preflight.v1",
        "status": "PASS",
        "output_root": str(resolved),
        "filesystem_anchor": str(anchor),
        "required_free_bytes": required,
        "observed_free_bytes": int(usage.free),
        "allowed_ssd_roots": [str(Path(root).resolve()) for root in allowed_roots],
        "cleanup_policy": "certify_or_block; preserve_shards_and_aggregate_until_cold_store_certificate",
    }


def _closure_identity(upstream_root: Path) -> dict[str, Any]:
    root = upstream_root.resolve()
    rows = []
    digest = hashlib.sha256()
    for relative in UPSTREAM_CLOSURE_MEMBERS:
        identity = file_identity(root / relative)
        row = {"relative_path": relative, **identity}
        rows.append(row)
        digest.update(canonical_json_bytes(row))
        digest.update(b"\n")
    return {
        "root": str(root),
        "members": rows,
        "closure_sha256": digest.hexdigest(),
    }


def _seal_payload(payload: Mapping[str, Any], *, hash_field: str) -> dict[str, Any]:
    if hash_field in payload:
        raise FreshTeacherMaterializationError(f"payload already contains {hash_field}")
    sealed = dict(payload)
    sealed[hash_field] = payload_sha256(payload)
    return sealed


def verify_sealed_payload(payload: Mapping[str, Any], *, hash_field: str) -> None:
    expected = payload.get(hash_field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise FreshTeacherMaterializationError(f"missing or malformed {hash_field}")
    body = {key: value for key, value in payload.items() if key != hash_field}
    if payload_sha256(body) != expected:
        raise FreshTeacherMaterializationError(f"{hash_field} does not match canonical payload")


def build_fresh_teacher_preflight(
    *,
    source_video: Path,
    segnet_weights: Path,
    upstream_root: Path,
    output_root: Path,
    pair_count: int,
    batch_size: int,
    num_threads: int,
    seed: int,
    package_versions: Mapping[str, str],
    run_argv: Sequence[str],
    camera_hw: Sequence[int] = CAMERA_HW_PUBLIC,
    seg_hw: Sequence[int] = SEG_HW_PUBLIC,
    safety_reserve_bytes: int = DEFAULT_SAFETY_RESERVE_BYTES,
    allowed_ssd_roots: Sequence[Path] = SSD_ROOTS,
) -> dict[str, Any]:
    """Freeze every input needed by a future materialization run.

    The result is self-hashed but contains no target data.  Production callers
    write it as stage ``00`` and the run path must reopen it byte-for-byte.
    """

    pairs = _require_exact_int(pair_count, field="pair_count", minimum=1, maximum=PAIR_COUNT_PUBLIC)
    batch = _require_exact_int(batch_size, field="batch_size", minimum=1, maximum=64)
    threads = _require_exact_int(num_threads, field="num_threads", minimum=1, maximum=256)
    exact_seed = _require_exact_int(seed, field="seed", minimum=0, maximum=(1 << 63) - 1)
    camera = _require_hw(camera_hw, field="camera_hw")
    seg = _require_hw(seg_hw, field="seg_hw")
    output = require_ssd_output_root(output_root, allowed_roots=allowed_ssd_roots)
    projected = projected_materialization_bytes(
        pair_count=pairs,
        seg_hw=seg,
        safety_reserve_bytes=safety_reserve_bytes,
    )
    preflight = storage_preflight(
        output,
        required_free_bytes=projected,
        allowed_roots=allowed_ssd_roots,
    )
    versions = {str(key): str(value) for key, value in sorted(package_versions.items())}
    if not versions or any(not key or not value for key, value in versions.items()):
        raise FreshTeacherMaterializationError("package_versions must be a nonempty exact mapping")
    argv = [str(token) for token in run_argv]
    if not argv or any(not token for token in argv):
        raise FreshTeacherMaterializationError("run_argv must be a nonempty exact argv")

    body = {
        "schema": PREFLIGHT_SCHEMA,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
        "pointer_mutation_allowed": False,
        "encoder_only": True,
        "candidate_payload_allowed": False,
        "source_video": file_identity(source_video),
        "segnet_weights": file_identity(segnet_weights),
        "upstream_closure": _closure_identity(upstream_root),
        "output_root": str(output),
        "pair_count": pairs,
        "full_public_population_requested": pairs == PAIR_COUNT_PUBLIC,
        "batch_size": batch,
        "scorer_pair_batch_size": batch,
        "upstream_evaluate_default_pair_batch_size": UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE,
        "batch_geometry_matches_upstream_default": batch == UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE,
        "batch_geometry_authority": (
            "UPSTREAM_DEFAULT_MATCH_MACOS_CPU_ADVISORY"
            if batch == UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE
            else "NONAUTHORITATIVE_DIAGNOSTIC_GEOMETRY"
        ),
        "contest_axis_authority": False,
        "source_sequence_length": 2,
        "segnet_frame_selector": "last_frame_index_1_of_nonoverlapping_pair",
        "num_threads": threads,
        "seed": exact_seed,
        "device": "cpu",
        "camera_hw": list(camera),
        "seg_hw": list(seg),
        "class_count": CLASS_COUNT,
        "package_versions": versions,
        "run_argv": argv,
        "storage_preflight": preflight,
        "stage_contract": [
            "00_custody_storage_preflight",
            "10_source_target_pair_shards",
            "11_target_label_aggregate",
            "12_encoder_only_receipt",
        ],
        "resume_contract": {
            "pair_checkpoint_atomic": True,
            "all_pair_shards_reopened_and_rehashed": True,
            "source_pair_rgb_rehashed_on_resume": True,
            "committed_range_gaps_or_overlaps_refused": True,
            "aggregate_rebuilt_from_verified_pair_shards": True,
        },
        "lineage_contract": {
            "teacher_evidence_may_feed_encoder": True,
            "teacher_evidence_may_ship_in_archive": False,
            "target_labels_may_ship_in_inflate_code": False,
            "historical_target_payload_contribution": False,
            "consumer": "fresh target versus exact current-P labels -> G17 label-local semantic debt",
        },
        "unified_solver_hooks": {
            "sensitivity_map": "per-pair SegNet target/current-P disagreement support",
            "pareto_constraint": "encoder-only evidence contributes zero candidate bytes",
            "bit_allocator": "marginal corrected target cells per counted G operand byte",
            "cathedral_autopilot": "blocks G17 semantic compile until receipt reopens",
            "continual_learning": "first changed-state n600 score row updates actuator economics",
            "probe_disambiguator": "not_applicable; frozen upstream SegNet argmax is the unique target",
        },
    }
    return _seal_payload(body, hash_field="preflight_sha256")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload) + b"\n")


def load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshTeacherMaterializationError(f"cannot read JSON mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FreshTeacherMaterializationError(f"JSON payload at {path} is not a mapping")
    return value


def reverify_preflight(
    preflight: Mapping[str, Any],
    *,
    allowed_ssd_roots: Sequence[Path] = SSD_ROOTS,
) -> None:
    if preflight.get("schema") != PREFLIGHT_SCHEMA:
        raise FreshTeacherMaterializationError("unexpected teacher preflight schema")
    verify_sealed_payload(preflight, hash_field="preflight_sha256")
    if (
        preflight.get("encoder_only") is not True
        or preflight.get("candidate_payload_allowed") is not False
        or preflight.get("pointer_mutation_allowed") is not False
        or preflight.get("device") != "cpu"
    ):
        raise FreshTeacherMaterializationError("teacher preflight weakened the encoder-only contract")
    # These fields were added after the first batch-4 diagnostic receipt.  Old
    # sealed receipts remain reopenable, but every new receipt is checked
    # against its explicit scorer geometry instead of silently inheriting an
    # authority claim from the word "frozen".
    if "scorer_pair_batch_size" in preflight:
        batch = _require_exact_int(
            preflight.get("batch_size"),
            field="batch_size",
            minimum=1,
            maximum=64,
        )
        expected_geometry = {
            "scorer_pair_batch_size": batch,
            "upstream_evaluate_default_pair_batch_size": UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE,
            "batch_geometry_matches_upstream_default": batch == UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE,
            "batch_geometry_authority": (
                "UPSTREAM_DEFAULT_MATCH_MACOS_CPU_ADVISORY"
                if batch == UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE
                else "NONAUTHORITATIVE_DIAGNOSTIC_GEOMETRY"
            ),
            "contest_axis_authority": False,
            "source_sequence_length": 2,
            "segnet_frame_selector": "last_frame_index_1_of_nonoverlapping_pair",
        }
        for field, expected in expected_geometry.items():
            if preflight.get(field) != expected:
                raise FreshTeacherMaterializationError(
                    f"teacher preflight {field} does not match its exact scorer geometry"
                )
    output_root = require_ssd_output_root(
        Path(str(preflight.get("output_root"))),
        allowed_roots=allowed_ssd_roots,
    )
    if str(output_root) != preflight.get("output_root"):
        raise FreshTeacherMaterializationError("teacher output_root is not canonical")
    for field in ("source_video", "segnet_weights"):
        expected = preflight.get(field)
        if not isinstance(expected, Mapping):
            raise FreshTeacherMaterializationError(f"preflight {field} is missing")
        actual = file_identity(Path(str(expected.get("path"))))
        if actual != dict(expected):
            raise FreshTeacherMaterializationError(f"preflight {field} identity drifted")
    upstream = preflight.get("upstream_closure")
    if not isinstance(upstream, Mapping):
        raise FreshTeacherMaterializationError("preflight upstream closure is missing")
    actual_closure = _closure_identity(Path(str(upstream.get("root"))))
    if actual_closure != dict(upstream):
        raise FreshTeacherMaterializationError("preflight upstream closure drifted")
    storage = preflight.get("storage_preflight")
    if not isinstance(storage, Mapping) or storage.get("status") != "PASS":
        raise FreshTeacherMaterializationError("teacher storage preflight is not PASS")
    storage_preflight(
        output_root,
        required_free_bytes=int(storage.get("required_free_bytes", 0)),
        allowed_roots=allowed_ssd_roots,
    )


@dataclass(frozen=True)
class PreparedTeacherBatchV1:
    """Exact preprocessed scorer inputs plus a selective forward callback.

    ``scorer_input_sha256`` covers every pair in ``source_batch``.  The
    callback receives local indices that do not already have a valid checkpoint
    and returns only those label maps, avoiding needless resumed forwards.
    """

    scorer_input_sha256: tuple[str, ...]
    infer_missing: Callable[[tuple[int, ...]], Mapping[int, np.ndarray]]


BatchPreparer = Callable[[np.ndarray], PreparedTeacherBatchV1]


def _raw_array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return sha256_bytes(memoryview(contiguous).cast("B"))


def _pair_paths(output_root: Path, pair_index: int) -> tuple[Path, Path]:
    stem = f"pair_{pair_index:04d}"
    return output_root / "10_pair_shards" / f"{stem}.u8", output_root / "10_pair_checkpoints" / f"{stem}.json"


def _validate_checkpoint(
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    shard_path: Path,
    pair_index: int,
    preflight_sha256: str,
    source_pair_rgb_sha256: str,
    scorer_input_sha256: str,
    seg_hw: tuple[int, int],
) -> dict[str, Any]:
    if checkpoint.get("schema") != PAIR_CHECKPOINT_SCHEMA:
        raise FreshTeacherMaterializationError(f"pair checkpoint schema drift: {checkpoint_path}")
    verify_sealed_payload(checkpoint, hash_field="checkpoint_sha256")
    exact = {
        "pair_index": pair_index,
        "preflight_sha256": preflight_sha256,
        "source_pair_rgb_sha256": source_pair_rgb_sha256,
        "scorer_input_sha256": scorer_input_sha256,
        "target_shape": list(seg_hw),
        "target_dtype": "uint8",
    }
    for key, expected in exact.items():
        if checkpoint.get(key) != expected:
            raise FreshTeacherMaterializationError(
                f"pair checkpoint {pair_index} {key} drift: {checkpoint.get(key)!r} != {expected!r}"
            )
    if Path(str(checkpoint.get("target_shard_path"))).resolve() != shard_path.resolve():
        raise FreshTeacherMaterializationError(f"pair checkpoint {pair_index} names another shard")
    identity = file_identity(shard_path)
    if int(identity["bytes"]) != seg_hw[0] * seg_hw[1]:
        raise FreshTeacherMaterializationError(f"pair shard {pair_index} byte size drifted")
    if (
        checkpoint.get("target_shard_bytes") != identity["bytes"]
        or checkpoint.get("target_labels_sha256") != identity["sha256"]
    ):
        raise FreshTeacherMaterializationError(f"pair shard {pair_index} identity drifted")
    labels = np.fromfile(shard_path, dtype=np.uint8)
    if labels.size != seg_hw[0] * seg_hw[1] or bool(np.any(labels >= CLASS_COUNT)):
        raise FreshTeacherMaterializationError(f"pair shard {pair_index} has invalid label values")
    return dict(checkpoint)


def _write_pair_checkpoint(
    *,
    output_root: Path,
    pair_index: int,
    labels: np.ndarray,
    preflight_sha256: str,
    source_pair_rgb_sha256: str,
    scorer_input_sha256: str,
    seg_hw: tuple[int, int],
) -> dict[str, Any]:
    target = np.ascontiguousarray(labels)
    if target.dtype != np.uint8 or target.shape != seg_hw:
        raise FreshTeacherMaterializationError(
            f"inferred pair {pair_index} labels must be uint8 {seg_hw}, got {target.dtype} {target.shape}"
        )
    if bool(np.any(target >= CLASS_COUNT)):
        raise FreshTeacherMaterializationError(f"inferred pair {pair_index} contains a class outside 0..4")
    shard_path, checkpoint_path = _pair_paths(output_root, pair_index)
    atomic_write_bytes(shard_path, memoryview(target).cast("B").tobytes())
    shard_identity = file_identity(shard_path)
    body = {
        "schema": PAIR_CHECKPOINT_SCHEMA,
        "pair_index": pair_index,
        "preflight_sha256": preflight_sha256,
        "source_pair_rgb_sha256": source_pair_rgb_sha256,
        "scorer_input_sha256": scorer_input_sha256,
        "target_shape": list(seg_hw),
        "target_dtype": "uint8",
        "target_shard_path": str(shard_path.resolve()),
        "target_shard_bytes": shard_identity["bytes"],
        "target_labels_sha256": shard_identity["sha256"],
        "committed_atomically": True,
        "encoder_only": True,
        "candidate_payload_allowed": False,
    }
    checkpoint = _seal_payload(body, hash_field="checkpoint_sha256")
    atomic_write_json(checkpoint_path, checkpoint)
    return _validate_checkpoint(
        checkpoint=load_json_mapping(checkpoint_path),
        checkpoint_path=checkpoint_path,
        shard_path=shard_path,
        pair_index=pair_index,
        preflight_sha256=preflight_sha256,
        source_pair_rgb_sha256=source_pair_rgb_sha256,
        scorer_input_sha256=scorer_input_sha256,
        seg_hw=seg_hw,
    )


def _aggregate_verified_shards(
    *,
    output_root: Path,
    checkpoints: Sequence[Mapping[str, Any]],
    seg_hw: tuple[int, int],
) -> dict[str, Any]:
    aggregate_path = output_root / "11_target_labels" / "target_labels_n600_or_bounded.u8"
    temporary = aggregate_path.with_name(aggregate_path.name + ".tmp")
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for expected_pair, checkpoint in enumerate(checkpoints):
            if checkpoint.get("pair_index") != expected_pair:
                raise FreshTeacherMaterializationError("pair checkpoints are not a gap-free chronological population")
            shard_path = Path(str(checkpoint["target_shard_path"]))
            payload = shard_path.read_bytes()
            if len(payload) != seg_hw[0] * seg_hw[1]:
                raise FreshTeacherMaterializationError(f"pair {expected_pair} shard size changed before aggregate")
            if sha256_bytes(payload) != checkpoint.get("target_labels_sha256"):
                raise FreshTeacherMaterializationError(f"pair {expected_pair} shard hash changed before aggregate")
            output.write(payload)
            digest.update(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, aggregate_path)
    identity = file_identity(aggregate_path)
    expected_bytes = len(checkpoints) * seg_hw[0] * seg_hw[1]
    if identity["bytes"] != expected_bytes or identity["sha256"] != digest.hexdigest():
        raise FreshTeacherMaterializationError("aggregate target-label bank failed close-after-write verification")
    return {
        **identity,
        "shape": [len(checkpoints), *seg_hw],
        "dtype": "uint8",
        "chronological_pair_order": list(range(len(checkpoints))),
    }


def materialize_fresh_teacher_from_batches(
    *,
    preflight: Mapping[str, Any],
    source_batches: Iterable[np.ndarray],
    prepare_batch: BatchPreparer,
    allowed_ssd_roots: Sequence[Path] = SSD_ROOTS,
) -> dict[str, Any]:
    """Materialize or resume exact teacher labels from chronological RGB batches."""

    reverify_preflight(preflight, allowed_ssd_roots=allowed_ssd_roots)
    output_root = Path(str(preflight["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    pair_count = int(preflight["pair_count"])
    scorer_batch_size = _require_exact_int(
        preflight["batch_size"],
        field="batch_size",
        minimum=1,
        maximum=64,
    )
    camera_hw = _require_hw(preflight["camera_hw"], field="camera_hw")
    seg_hw = _require_hw(preflight["seg_hw"], field="seg_hw")
    preflight_sha = str(preflight["preflight_sha256"])
    stage_zero = output_root / "00_custody_storage_preflight.json"
    if stage_zero.exists():
        on_disk = load_json_mapping(stage_zero)
        if on_disk != dict(preflight):
            raise FreshTeacherMaterializationError("stage-00 preflight bytes name another run")
    else:
        atomic_write_json(stage_zero, dict(preflight))

    checkpoints: list[dict[str, Any]] = []
    observed_pairs = 0
    for source_batch in source_batches:
        batch = np.ascontiguousarray(source_batch)
        if (
            batch.dtype != np.uint8
            or batch.ndim != 5
            or tuple(batch.shape[1:])
            != (
                2,
                camera_hw[0],
                camera_hw[1],
                3,
            )
        ):
            raise FreshTeacherMaterializationError(
                "source batch must be uint8 (B,2,H,W,3) under the frozen AVVideoDataset contract"
            )
        if observed_pairs + int(batch.shape[0]) > pair_count:
            raise FreshTeacherMaterializationError(
                f"source contains more than the exact requested {pair_count} chronological pairs"
            )
        expected_batch_pairs = min(scorer_batch_size, pair_count - observed_pairs)
        if int(batch.shape[0]) != expected_batch_pairs:
            raise FreshTeacherMaterializationError(
                "source batch cardinality changed scorer geometry: "
                f"observed {int(batch.shape[0])}, expected exactly {expected_batch_pairs} "
                f"at pair offset {observed_pairs}"
            )
        prepared = prepare_batch(batch)
        if len(prepared.scorer_input_sha256) != int(batch.shape[0]):
            raise FreshTeacherMaterializationError("prepare_batch returned the wrong scorer-input hash count")
        missing_local: list[int] = []
        batch_rows: list[dict[str, Any] | None] = [None] * int(batch.shape[0])
        for local_index in range(int(batch.shape[0])):
            pair_index = observed_pairs + local_index
            source_sha = _raw_array_sha256(batch[local_index])
            scorer_sha = prepared.scorer_input_sha256[local_index]
            if not isinstance(scorer_sha, str) or len(scorer_sha) != 64:
                raise FreshTeacherMaterializationError("prepare_batch returned a malformed scorer-input hash")
            shard_path, checkpoint_path = _pair_paths(output_root, pair_index)
            if checkpoint_path.exists() or shard_path.exists():
                if not checkpoint_path.is_file() or not shard_path.is_file():
                    raise FreshTeacherMaterializationError(
                        f"pair {pair_index} has an orphaned shard/checkpoint; certify or repair before resume"
                    )
                batch_rows[local_index] = _validate_checkpoint(
                    checkpoint=load_json_mapping(checkpoint_path),
                    checkpoint_path=checkpoint_path,
                    shard_path=shard_path,
                    pair_index=pair_index,
                    preflight_sha256=preflight_sha,
                    source_pair_rgb_sha256=source_sha,
                    scorer_input_sha256=scorer_sha,
                    seg_hw=seg_hw,
                )
            else:
                missing_local.append(local_index)
        inferred = prepared.infer_missing(tuple(missing_local)) if missing_local else {}
        if set(inferred) != set(missing_local):
            raise FreshTeacherMaterializationError("infer_missing did not return exactly the requested local indices")
        for local_index in missing_local:
            pair_index = observed_pairs + local_index
            batch_rows[local_index] = _write_pair_checkpoint(
                output_root=output_root,
                pair_index=pair_index,
                labels=np.asarray(inferred[local_index]),
                preflight_sha256=preflight_sha,
                source_pair_rgb_sha256=_raw_array_sha256(batch[local_index]),
                scorer_input_sha256=prepared.scorer_input_sha256[local_index],
                seg_hw=seg_hw,
            )
        if any(row is None for row in batch_rows):
            raise FreshTeacherMaterializationError("internal pair checkpoint assembly gap")
        checkpoints.extend(row for row in batch_rows if row is not None)
        observed_pairs += int(batch.shape[0])

    if observed_pairs != pair_count:
        raise FreshTeacherMaterializationError(
            f"source population mismatch: observed {observed_pairs}, expected exactly {pair_count} pairs"
        )
    if [int(row["pair_index"]) for row in checkpoints] != list(range(pair_count)):
        raise FreshTeacherMaterializationError("materialized pair population has a gap, overlap or reorder")

    aggregate = _aggregate_verified_shards(
        output_root=output_root,
        checkpoints=checkpoints,
        seg_hw=seg_hw,
    )
    checkpoint_root_sha = payload_sha256(
        [
            {
                "pair_index": row["pair_index"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "target_labels_sha256": row["target_labels_sha256"],
            }
            for row in checkpoints
        ]
    )
    body = {
        "schema": SCHEMA,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
        "pointer_mutation_allowed": False,
        "encoder_only": True,
        "candidate_payload_allowed": False,
        "preflight_sha256": preflight_sha,
        "source_video": preflight["source_video"],
        "segnet_weights": preflight["segnet_weights"],
        "upstream_closure": preflight["upstream_closure"],
        "pair_count": pair_count,
        "frame_count": pair_count * 2,
        "batch_size": preflight["batch_size"],
        "scorer_pair_batch_size": preflight.get("scorer_pair_batch_size", preflight["batch_size"]),
        "upstream_evaluate_default_pair_batch_size": preflight.get(
            "upstream_evaluate_default_pair_batch_size",
            UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE,
        ),
        "batch_geometry_matches_upstream_default": preflight.get(
            "batch_geometry_matches_upstream_default",
            preflight["batch_size"] == UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE,
        ),
        "batch_geometry_authority": preflight.get(
            "batch_geometry_authority",
            "LEGACY_RECEIPT_GEOMETRY_UNDECLARED",
        ),
        "contest_axis_authority": False,
        "source_sequence_length": preflight.get("source_sequence_length", 2),
        "segnet_frame_selector": preflight.get(
            "segnet_frame_selector",
            "last_frame_index_1_of_nonoverlapping_pair",
        ),
        "full_public_population_proven": pair_count == PAIR_COUNT_PUBLIC,
        "chronological_pair_order": list(range(pair_count)),
        "pair_checkpoint_root_sha256": checkpoint_root_sha,
        "pair_checkpoints": checkpoints,
        "target_labels": aggregate,
        "target_labels_encoder_only": True,
        "target_labels_serialized_in_candidate": False,
        "scorer_weights_serialized_in_candidate": False,
        "cleanup_certificate": {
            "status": "PRESERVED_NO_DELETE",
            "output_root": str(output_root.resolve()),
            "rebuild_argv": preflight["run_argv"],
            "preflight_sha256": preflight_sha,
            "aggregate_sha256": aggregate["sha256"],
            "policy": "cold-store or delete only after a successor machine-readable certificate",
        },
        "next_consumer_contract": {
            "current_predictor_labels_required": True,
            "exact_current_P_identity_required": True,
            "fresh_G_operands_only": True,
            "teacher_bytes_forbidden_from_archive": True,
            "semantic_compile_geometry_ready": preflight.get(
                "batch_geometry_matches_upstream_default",
                False,
            ),
            "first_authoritative_use": (
                "exact changed n600 archive replay through upstream/evaluate.py on contest CPU/CUDA; "
                "this materialization remains encoder-only macOS-CPU evidence"
            ),
        },
    }
    receipt = _seal_payload(body, hash_field="receipt_sha256")
    final_path = output_root / "12_encoder_only_receipt.json"
    atomic_write_json(final_path, receipt)
    reopened = load_json_mapping(final_path)
    verify_sealed_payload(reopened, hash_field="receipt_sha256")
    if reopened != receipt:
        raise FreshTeacherMaterializationError("final teacher receipt changed across atomic parse-back")
    return receipt


def load_and_reverify_materialization_receipt(
    path: Path,
    *,
    allowed_ssd_roots: Sequence[Path] = SSD_ROOTS,
) -> dict[str, Any]:
    """Reopen a completed receipt and every pair/aggregate byte it names."""

    receipt = load_json_mapping(path)
    if receipt.get("schema") != SCHEMA:
        raise FreshTeacherMaterializationError("unexpected teacher materialization receipt schema")
    verify_sealed_payload(receipt, hash_field="receipt_sha256")
    if (
        receipt.get("encoder_only") is not True
        or receipt.get("candidate_payload_allowed") is not False
        or receipt.get("target_labels_serialized_in_candidate") is not False
        or receipt.get("scorer_weights_serialized_in_candidate") is not False
    ):
        raise FreshTeacherMaterializationError("teacher receipt weakened the no-payload contract")
    if "scorer_pair_batch_size" in receipt:
        batch = _require_exact_int(
            receipt.get("batch_size"),
            field="receipt.batch_size",
            minimum=1,
            maximum=64,
        )
        expected_match = batch == UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE
        if (
            receipt.get("scorer_pair_batch_size") != batch
            or receipt.get("upstream_evaluate_default_pair_batch_size")
            != UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE
            or receipt.get("batch_geometry_matches_upstream_default") is not expected_match
            or receipt.get("contest_axis_authority") is not False
        ):
            raise FreshTeacherMaterializationError("teacher receipt scorer batch geometry drifted")
    aggregate = receipt.get("target_labels")
    rows = receipt.get("pair_checkpoints")
    if not isinstance(aggregate, Mapping) or not isinstance(rows, list):
        raise FreshTeacherMaterializationError("teacher receipt lacks path-backed evidence")
    aggregate_path = Path(str(aggregate.get("path")))
    output_root = require_ssd_output_root(
        aggregate_path.parent.parent,
        allowed_roots=allowed_ssd_roots,
    )
    preflight_path = output_root / "00_custody_storage_preflight.json"
    preflight = load_json_mapping(preflight_path)
    reverify_preflight(preflight, allowed_ssd_roots=allowed_ssd_roots)
    if preflight.get("preflight_sha256") != receipt.get("preflight_sha256"):
        raise FreshTeacherMaterializationError("teacher receipt names another stage-00 preflight")
    if "scorer_pair_batch_size" in receipt and (
        receipt.get("scorer_pair_batch_size")
        != preflight.get("scorer_pair_batch_size", preflight.get("batch_size"))
        or receipt.get("batch_geometry_matches_upstream_default")
        != preflight.get(
            "batch_geometry_matches_upstream_default",
            preflight.get("batch_size") == UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE,
        )
    ):
        raise FreshTeacherMaterializationError("teacher receipt scorer geometry differs from stage-00")
    aggregate_identity = file_identity(aggregate_path)
    if aggregate_identity != {key: aggregate[key] for key in ("path", "bytes", "sha256")}:
        raise FreshTeacherMaterializationError("aggregate target labels drifted")
    pair_count = int(receipt.get("pair_count", 0))
    seg_shape = aggregate.get("shape")
    if not isinstance(seg_shape, list) or len(seg_shape) != 3 or seg_shape[0] != pair_count:
        raise FreshTeacherMaterializationError("aggregate target-label geometry drifted")
    seg_hw = _require_hw(seg_shape[1:], field="target_labels.shape[1:]")
    ordered_root_rows = []
    for expected_pair, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise FreshTeacherMaterializationError("pair checkpoint row is not a mapping")
        shard_path = Path(str(row.get("target_shard_path")))
        checkpoint_path = shard_path.parent.parent / "10_pair_checkpoints" / f"pair_{expected_pair:04d}.json"
        parsed = _validate_checkpoint(
            checkpoint=load_json_mapping(checkpoint_path),
            checkpoint_path=checkpoint_path,
            shard_path=shard_path,
            pair_index=expected_pair,
            preflight_sha256=str(receipt["preflight_sha256"]),
            source_pair_rgb_sha256=str(row["source_pair_rgb_sha256"]),
            scorer_input_sha256=str(row["scorer_input_sha256"]),
            seg_hw=seg_hw,
        )
        if parsed != dict(row):
            raise FreshTeacherMaterializationError(f"pair {expected_pair} checkpoint receipt drifted")
        ordered_root_rows.append(
            {
                "pair_index": expected_pair,
                "checkpoint_sha256": parsed["checkpoint_sha256"],
                "target_labels_sha256": parsed["target_labels_sha256"],
            }
        )
    if payload_sha256(ordered_root_rows) != receipt.get("pair_checkpoint_root_sha256"):
        raise FreshTeacherMaterializationError("pair checkpoint root drifted")
    return receipt


def load_compile_ready_materialization_receipt(
    path: Path,
    *,
    allowed_ssd_roots: Sequence[Path] = SSD_ROOTS,
) -> dict[str, Any]:
    """Reopen a teacher bank and require upstream-default scorer geometry.

    This is a *compiler-input* gate, not a score-authority promotion.  macOS CPU
    labels remain encoder-only advisory evidence until a changed archive is
    replayed by the exact upstream evaluator on a contest axis.
    """

    receipt = load_and_reverify_materialization_receipt(
        path,
        allowed_ssd_roots=allowed_ssd_roots,
    )
    consumer = receipt.get("next_consumer_contract")
    if (
        receipt.get("scorer_pair_batch_size") != UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE
        or receipt.get("upstream_evaluate_default_pair_batch_size")
        != UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE
        or receipt.get("batch_geometry_matches_upstream_default") is not True
        or receipt.get("batch_geometry_authority") != "UPSTREAM_DEFAULT_MATCH_MACOS_CPU_ADVISORY"
        or receipt.get("contest_axis_authority") is not False
        or not isinstance(consumer, Mapping)
        or consumer.get("semantic_compile_geometry_ready") is not True
    ):
        raise FreshTeacherMaterializationError(
            "teacher receipt is not compile-ready under exact upstream-default batch geometry"
        )
    return receipt


__all__ = [
    "CAMERA_HW_PUBLIC",
    "CLASS_COUNT",
    "DEFAULT_SAFETY_RESERVE_BYTES",
    "EVIDENCE_AXIS",
    "PAIR_COUNT_PUBLIC",
    "PREFLIGHT_SCHEMA",
    "SCHEMA",
    "SEG_HW_PUBLIC",
    "SSD_ROOTS",
    "UPSTREAM_EVALUATE_DEFAULT_BATCH_SIZE",
    "BatchPreparer",
    "FreshTeacherMaterializationError",
    "PreparedTeacherBatchV1",
    "atomic_write_json",
    "build_fresh_teacher_preflight",
    "canonical_json_bytes",
    "file_identity",
    "load_and_reverify_materialization_receipt",
    "load_compile_ready_materialization_receipt",
    "load_json_mapping",
    "materialize_fresh_teacher_from_batches",
    "payload_sha256",
    "projected_materialization_bytes",
    "reverify_preflight",
    "sha256_file",
    "storage_preflight",
]
