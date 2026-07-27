# SPDX-License-Identifier: MIT
"""Encoder-only custody for G72 target margins and V15 base scorer fields.

The scorer's global batch-16 geometry and G72's five 120-pair stages are not
aligned.  This module therefore preserves two compatible checkpoint layers:

* 38 immutable global scorer-batch shards (the final shard contains 8 pairs);
* five immutable 120-pair stage caches assembled from verified batch shards.

The G46 target-label bank is reopened, never regenerated or copied.  The G51
Y0/Y1 operands are recursively reopened and hash-bound, never regenerated.
Only three new dense encoder-side fields are materialized:

* target winner margin;
* V15 camera-through-live-R described cell;
* V15 camera-through-live-R described winner margin.

All dense fields and frozen scorer weights are forbidden from candidate bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

CONFIG_SCHEMA: Final = "tac.taskspace_batch16_margin_base_scorer_cache_config.v1"
PREFLIGHT_SCHEMA: Final = "tac.taskspace_batch16_margin_base_scorer_cache_preflight.v1"
BATCH_CHECKPOINT_SCHEMA: Final = "tac.taskspace_batch16_margin_base_scorer_batch.v1"
STAGE_CHECKPOINT_SCHEMA: Final = "tac.taskspace_batch16_margin_base_scorer_stage.v1"
AGGREGATE_SCHEMA: Final = "tac.taskspace_batch16_margin_base_scorer_aggregate.v1"

PRODUCTION_PAIR_COUNT: Final = 600
PRODUCTION_STAGE_PAIRS: Final = 120
PRODUCTION_STAGE_COUNT: Final = 5
PRODUCTION_BATCH_PAIRS: Final = 16
PRODUCTION_SCORER_HW: Final = (384, 512)
PRODUCTION_CLASS_COUNT: Final = 5
SSD_ROOTS: Final = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)
DEFAULT_REQUIRED_FREE_BYTES: Final = 12 * (1 << 30)
EVIDENCE_AXIS: Final = "[macOS-CPU encoder-only batch16 frozen-scorer evidence]"
REMAINING_G72_BLOCKERS: Final = (
    "G72_G49_ROLE_PRESERVING_ANALYTIC_WIRE_ABI_OWED",
    "G72_V15_ROLE_AWARE_PREPAINT_ANALYTIC_DECODER_PROOF_OWED",
    "G72_FRESH_POSE_TARGET_AUTHORITY_OR_EXACT_UPSTREAM_FINAL_REPLAY_OWED",
    "G72_FIVE_STAGE_EXACT_WHOLE_OBJECT_JOINT_ADMISSION_OWED",
)

U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]


class Batch16MarginBaseScorerCacheError(RuntimeError):
    """A config, custody, scorer, resume, or immutable-cache check failed."""


@dataclass(frozen=True, slots=True)
class BatchProductsV1:
    """New fields derived by one exact global scorer-batch forward."""

    target_cells_u8: U8
    target_margins_f32: F32
    described_cells_u8: U8
    described_margins_f32: F32


@dataclass(frozen=True, slots=True)
class PreparedBatchV1:
    """Source/V15 custody plus a lazy exact scorer forward.

    ``infer`` is not called for an already committed batch.  Source RGB, target
    scorer input, fresh V15 camera bytes, and fresh V15 live-R scorer input are
    nevertheless regenerated and checked before a completed checkpoint may be
    reused.
    """

    source_pair_batch_sha256: str
    target_scorer_input_sha256: str
    v15_camera_sha256: str
    v15_scorer_input_sha256: str
    infer: Callable[[], BatchProductsV1]


@dataclass(frozen=True, slots=True)
class MarginBaseScorerStageV1:
    """One read-only 120-pair G72 operand stage."""

    stage_index: int
    pair_range: tuple[int, int]
    target_cells_u8: U8
    target_margins_f32: F32
    described_cells_u8: U8
    described_margins_f32: F32


BatchPreparer = Callable[[tuple[int, ...]], PreparedBatchV1]


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
    resolved = Path(path).expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise Batch16MarginBaseScorerCacheError(f"bound file is absent, non-regular, or a symlink: {resolved}")
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _seal(body: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if field in body:
        raise Batch16MarginBaseScorerCacheError(f"payload already contains {field}")
    return {**body, field: payload_sha256(body)}


def _verify_seal(value: Mapping[str, Any], *, field: str) -> None:
    expected = _require_sha(value.get(field), field)
    body = {key: item for key, item in value.items() if key != field}
    if payload_sha256(body) != expected:
        raise Batch16MarginBaseScorerCacheError(f"{field} canonical hash differs")


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Batch16MarginBaseScorerCacheError(f"{label} must be lowercase SHA-256")
    return value


def _require_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise Batch16MarginBaseScorerCacheError(f"{label} must be an exact integer in [{minimum},{maximum}]")
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Batch16MarginBaseScorerCacheError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Batch16MarginBaseScorerCacheError(f"{label} is not a JSON object")
    return value


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise Batch16MarginBaseScorerCacheError(
                    f"immutable path already contains different bytes: {path}"
                ) from None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_atomic(path, canonical_json_bytes(value) + b"\n")


def write_immutable_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    payload = memoryview(np.ascontiguousarray(value)).cast("B").tobytes()
    _write_atomic(path, payload)
    return file_identity(path)


def require_ssd_output_root(
    output_root: Path,
    *,
    allowed_roots: Sequence[Path] = SSD_ROOTS,
) -> Path:
    resolved = output_root.expanduser().resolve()
    roots = tuple(root.resolve() for root in allowed_roots)
    if not any(resolved != root and resolved.is_relative_to(root) for root in roots):
        raise Batch16MarginBaseScorerCacheError(
            "production output must be a child of /Volumes/VertigoDataTier/pact or /Volumes/APDataStore/pact"
        )
    return resolved


def storage_preflight(
    output_root: Path,
    *,
    required_free_bytes: int,
    test_only_small_fixture: bool = False,
    allowed_roots: Sequence[Path] = SSD_ROOTS,
) -> dict[str, Any]:
    required = _require_int(required_free_bytes, "required_free_bytes", 1, 1 << 60)
    resolved = output_root.expanduser().resolve()
    if not test_only_small_fixture:
        resolved = require_ssd_output_root(resolved, allowed_roots=allowed_roots)
    anchor = resolved
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    free = int(shutil.disk_usage(anchor).free)
    if free < required:
        raise Batch16MarginBaseScorerCacheError(
            f"storage preflight refused: need {required} bytes, only {free} free at {anchor}"
        )
    return {
        "schema": "tac.taskspace_batch16_margin_base_storage_preflight.v1",
        "status": "PASS",
        "output_root": str(resolved),
        "filesystem_anchor": str(anchor),
        "required_free_bytes": required,
        "observed_free_bytes": free,
        "test_only_small_fixture": test_only_small_fixture,
        "cleanup_policy": (
            "PRESERVE; cold-store or delete only after a machine-readable "
            "rebuild certificate binds inputs, argv, hashes, and destination"
        ),
    }


def seal_preflight(body: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a fully resolved preflight body.

    Production callers must use the public CLI's strict custody resolver.
    This small boundary also permits synthetic unit fixtures without importing
    the 5 GB source cache or frozen scorer.
    """

    required = {
        "schema",
        "run_id",
        "evidence_axis",
        "research_only",
        "score_claim",
        "candidate_claim",
        "promotion_eligible",
        "pointer_mutation_allowed",
        "encoder_only",
        "dense_fields_candidate_payload_allowed",
        "scorer_weights_candidate_payload_allowed",
        "output_root",
        "pair_count",
        "stage_pairs",
        "stage_count",
        "scorer_batch_pairs",
        "scorer_hw",
        "class_count",
        "seed",
        "num_threads",
        "test_only_small_fixture",
        "storage_preflight",
        "config",
        "source_custody",
        "scorer_custody",
        "target_custody",
        "g51_y0_y1_custody",
        "semantic_custody",
        "runtime_custody",
        "sealed_input_files",
        "run_argv",
        "resume_contract",
        "blockers_closed_by_successful_aggregate",
    }
    if set(body) != required or body.get("schema") != PREFLIGHT_SCHEMA:
        raise Batch16MarginBaseScorerCacheError("preflight body keys/schema differ")
    return _seal(body, field="preflight_sha256")


def reverify_preflight(
    preflight: Mapping[str, Any],
    *,
    allowed_roots: Sequence[Path] = SSD_ROOTS,
) -> None:
    if preflight.get("schema") != PREFLIGHT_SCHEMA:
        raise Batch16MarginBaseScorerCacheError("preflight schema differs")
    _verify_seal(preflight, field="preflight_sha256")
    false_fields = (
        "score_claim",
        "candidate_claim",
        "promotion_eligible",
        "pointer_mutation_allowed",
        "dense_fields_candidate_payload_allowed",
        "scorer_weights_candidate_payload_allowed",
    )
    if (
        preflight.get("research_only") is not True
        or preflight.get("encoder_only") is not True
        or any(preflight.get(field) is not False for field in false_fields)
    ):
        raise Batch16MarginBaseScorerCacheError("preflight weakened false-authority fences")
    pair_count = _require_int(preflight.get("pair_count"), "pair_count", 1, PRODUCTION_PAIR_COUNT)
    stage_pairs = _require_int(preflight.get("stage_pairs"), "stage_pairs", 1, pair_count)
    stage_count = _require_int(preflight.get("stage_count"), "stage_count", 1, PRODUCTION_STAGE_COUNT)
    batch_pairs = _require_int(
        preflight.get("scorer_batch_pairs"),
        "scorer_batch_pairs",
        1,
        PRODUCTION_BATCH_PAIRS,
    )
    scorer_hw = preflight.get("scorer_hw")
    if (
        not isinstance(scorer_hw, list)
        or len(scorer_hw) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in scorer_hw)
        or pair_count != stage_pairs * stage_count
    ):
        raise Batch16MarginBaseScorerCacheError("preflight population/stage geometry differs")
    test_only = preflight.get("test_only_small_fixture") is True
    if not test_only and (
        pair_count != PRODUCTION_PAIR_COUNT
        or stage_pairs != PRODUCTION_STAGE_PAIRS
        or stage_count != PRODUCTION_STAGE_COUNT
        or batch_pairs != PRODUCTION_BATCH_PAIRS
        or tuple(scorer_hw) != PRODUCTION_SCORER_HW
        or preflight.get("class_count") != PRODUCTION_CLASS_COUNT
        or preflight.get("evidence_axis") != EVIDENCE_AXIS
    ):
        raise Batch16MarginBaseScorerCacheError("production geometry is not exact n600/batch16/five-stage")
    output = Path(str(preflight.get("output_root"))).expanduser().resolve()
    if not test_only:
        require_ssd_output_root(output, allowed_roots=allowed_roots)
    storage = preflight.get("storage_preflight")
    if not isinstance(storage, Mapping) or storage.get("status") != "PASS" or storage.get("output_root") != str(output):
        raise Batch16MarginBaseScorerCacheError("preflight storage receipt differs")
    storage_preflight(
        output,
        required_free_bytes=int(storage["required_free_bytes"]),
        test_only_small_fixture=test_only,
        allowed_roots=allowed_roots,
    )
    rows = preflight.get("sealed_input_files")
    if not isinstance(rows, list) or not rows:
        raise Batch16MarginBaseScorerCacheError("preflight sealed input closure is absent")
    seen: set[str] = set()
    for index, expected in enumerate(rows):
        if not isinstance(expected, Mapping) or set(expected) != {"role", "path", "bytes", "sha256"}:
            raise Batch16MarginBaseScorerCacheError(f"sealed input row {index} keys differ")
        if expected["path"] in seen:
            raise Batch16MarginBaseScorerCacheError("sealed input closure repeats a path")
        seen.add(str(expected["path"]))
        actual = file_identity(str(expected["path"]))
        if actual != {key: expected[key] for key in ("path", "bytes", "sha256")}:
            raise Batch16MarginBaseScorerCacheError(
                f"sealed input changed: {expected.get('role')} {expected.get('path')}"
            )
    target = preflight.get("target_custody")
    if not isinstance(target, Mapping):
        raise Batch16MarginBaseScorerCacheError("target custody is absent")
    labels = target.get("target_labels")
    if (
        not isinstance(labels, Mapping)
        or labels.get("shape") != [pair_count, *scorer_hw]
        or labels.get("dtype") != "uint8"
        or labels.get("encoder_only") is not True
        or labels.get("candidate_payload_allowed") is not False
        or file_identity(str(labels.get("path"))) != {key: labels[key] for key in ("path", "bytes", "sha256")}
    ):
        raise Batch16MarginBaseScorerCacheError("target-label bank custody differs")
    g51 = preflight.get("g51_y0_y1_custody")
    g51_stages = None if not isinstance(g51, Mapping) else g51.get("stages")
    if (
        not isinstance(g51, Mapping)
        or g51.get("y0_y1_reused_not_rederived") is not True
        or not isinstance(g51_stages, list)
        or len(g51_stages) != stage_count
    ):
        raise Batch16MarginBaseScorerCacheError("G51 stage custody is absent")
    for index, stage in enumerate(g51_stages):
        start = index * stage_pairs
        stop = start + stage_pairs
        if (
            not isinstance(stage, Mapping)
            or stage.get("stage_index") != index
            or stage.get("pair_range") != [start, stop]
            or stage.get("y0_y1_rederive_performed_by_g78") is not False
        ):
            raise Batch16MarginBaseScorerCacheError(f"G51 stage {index} geometry/custody differs")
        for name in ("manifest", "y0_u8", "y1_u8", "gt_poses_f32"):
            binding = stage.get(name)
            if not isinstance(binding, Mapping) or file_identity(str(binding.get("path"))) != {
                key: binding[key] for key in ("path", "bytes", "sha256")
            }:
                raise Batch16MarginBaseScorerCacheError(f"G51 stage {index} {name} custody differs")
            if str(binding["path"]) not in seen:
                raise Batch16MarginBaseScorerCacheError(f"G51 stage {index} {name} is outside sealed inputs")
    semantic = preflight.get("semantic_custody")
    camera_identity = None if not isinstance(semantic, Mapping) else semantic.get("full_p_camera_identity")
    checkpoints = None if not isinstance(camera_identity, Mapping) else camera_identity.get("checkpoints")
    expected_batch_count = (pair_count + batch_pairs - 1) // batch_pairs
    if (
        not isinstance(camera_identity, Mapping)
        or camera_identity.get("pair_count") != pair_count
        or camera_identity.get("batch_size") != batch_pairs
        or camera_identity.get("batch_count") != expected_batch_count
        or camera_identity.get("all_camera_bytes_identical") is not True
        or not isinstance(checkpoints, list)
        or len(checkpoints) != expected_batch_count
    ):
        raise Batch16MarginBaseScorerCacheError("V15 camera-identity custody differs")
    camera_chain_material: list[str] = []
    for index, checkpoint in enumerate(checkpoints):
        start = index * batch_pairs
        stop = min(start + batch_pairs, pair_count)
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("pair_range") != [start, stop]
            or checkpoint.get("byte_identical") is not True
        ):
            raise Batch16MarginBaseScorerCacheError(f"V15 camera identity {index} geometry differs")
        _require_sha(checkpoint.get("camera_sha256"), f"V15 camera identity {index} camera_sha256")
        identity_path = Path(str(checkpoint.get("path")))
        if (
            file_identity(identity_path) != {key: checkpoint[key] for key in ("path", "bytes", "sha256")}
            or str(checkpoint["path"]) not in seen
        ):
            raise Batch16MarginBaseScorerCacheError(f"V15 camera identity {index} file custody differs")
        identity_payload = _load_json(identity_path, f"V15 camera identity {index}")
        if (
            identity_payload.get("schema") != "ddm_v15_full_p_camera_identity_batch.v1"
            or identity_payload.get("typed_config_sha256") != camera_identity.get("typed_config_sha256")
            or identity_payload.get("local_pair_range") != [start, stop]
            or identity_payload.get("base_camera_sha256") != checkpoint["camera_sha256"]
            or identity_payload.get("final_camera_sha256") != checkpoint["camera_sha256"]
            or identity_payload.get("byte_identical") is not True
            or identity_payload.get("camera_bytes_released_after_compare") is not True
            or identity_payload.get("score_claim") is not False
        ):
            raise Batch16MarginBaseScorerCacheError(f"V15 camera identity {index} payload differs")
        camera_chain_material.append(checkpoint["camera_sha256"] * 2)
    camera_chain = hashlib.sha256("".join(camera_chain_material).encode("ascii")).hexdigest()
    if camera_identity.get("digest_chain_sha256") != camera_chain:
        raise Batch16MarginBaseScorerCacheError("V15 camera identity digest chain differs")
    runtime = preflight.get("runtime_custody")
    runtime_files = None if not isinstance(runtime, Mapping) else runtime.get("files")
    if not isinstance(runtime_files, list) or not runtime_files:
        raise Batch16MarginBaseScorerCacheError("runtime source closure is absent")
    runtime_paths: set[str] = set()
    for index, binding in enumerate(runtime_files):
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"role", "path", "bytes", "sha256"}
            or file_identity(str(binding.get("path"))) != {key: binding[key] for key in ("path", "bytes", "sha256")}
            or str(binding["path"]) not in seen
            or str(binding["path"]) in runtime_paths
        ):
            raise Batch16MarginBaseScorerCacheError(f"runtime source closure row {index} differs")
        runtime_paths.add(str(binding["path"]))
    if not test_only:
        required_runtime_paths = {
            str(Path(__file__).resolve()),
            str((Path(__file__).resolve().parents[1] / "optimization/direct_description_carrier_compose.py").resolve()),
            str((Path(__file__).resolve().parents[1] / "through_r/resolution_chain.py").resolve()),
            str((Path(__file__).resolve().parents[1] / "contest_eval_contract.py").resolve()),
        }
        if not required_runtime_paths <= runtime_paths:
            raise Batch16MarginBaseScorerCacheError("runtime source closure omits a required V15 render dependency")


def _expected_v15_camera_sha256(
    preflight: Mapping[str, Any],
    *,
    start: int,
    stop: int,
) -> str:
    checkpoints = preflight["semantic_custody"]["full_p_camera_identity"]["checkpoints"]
    matches = [row for row in checkpoints if row.get("pair_range") == [start, stop]]
    if len(matches) != 1:
        raise Batch16MarginBaseScorerCacheError(f"V15 camera identity for batch {start}:{stop} is absent or ambiguous")
    return _require_sha(matches[0].get("camera_sha256"), f"V15 camera {start}:{stop} SHA-256")


def _validate_prepared_batch(
    prepared: PreparedBatchV1,
    *,
    start: int,
    stop: int,
    preflight: Mapping[str, Any],
) -> None:
    _require_sha(prepared.source_pair_batch_sha256, "source_pair_batch_sha256")
    _require_sha(prepared.target_scorer_input_sha256, "target_scorer_input_sha256")
    camera_sha256 = _require_sha(prepared.v15_camera_sha256, "v15_camera_sha256")
    _require_sha(prepared.v15_scorer_input_sha256, "v15_scorer_input_sha256")
    expected_camera = _expected_v15_camera_sha256(preflight, start=start, stop=stop)
    if camera_sha256 != expected_camera:
        raise Batch16MarginBaseScorerCacheError(f"fresh V15 camera bytes differ from owned identity at {start}:{stop}")


def _array_paths(root: Path, start: int, stop: int) -> dict[str, Path]:
    stem = f"batch_{start:04d}_{stop:04d}"
    shard_root = root / "10_batch_shards"
    return {
        "target_margins_f32": shard_root / f"{stem}.target_margins.f32",
        "described_cells_u8": shard_root / f"{stem}.described_cells.u8",
        "described_margins_f32": shard_root / f"{stem}.described_margins.f32",
        "checkpoint": root / "10_batch_checkpoints" / f"{stem}.json",
    }


def _validate_scored_arrays(
    *,
    target_cells: np.ndarray,
    target_margins: np.ndarray,
    described_cells: np.ndarray,
    described_margins: np.ndarray,
    expected_target: np.ndarray,
    scorer_hw: tuple[int, int],
) -> tuple[U8, F32, U8, F32]:
    count = int(expected_target.shape[0])
    expected_shape = (count, *scorer_hw)
    target = np.ascontiguousarray(target_cells, dtype=np.uint8)
    target_margin = np.ascontiguousarray(target_margins, dtype="<f4")
    described = np.ascontiguousarray(described_cells, dtype=np.uint8)
    described_margin = np.ascontiguousarray(described_margins, dtype="<f4")
    if (
        target.shape != expected_shape
        or target_margin.shape != expected_shape
        or described.shape != expected_shape
        or described_margin.shape != expected_shape
    ):
        raise Batch16MarginBaseScorerCacheError("scorer batch field geometry differs")
    if not np.array_equal(target, expected_target):
        mismatch = int(np.count_nonzero(target != expected_target))
        raise Batch16MarginBaseScorerCacheError(
            f"target scorer argmax differs from owned G46 labels at {mismatch} cells"
        )
    if bool(np.any(described >= PRODUCTION_CLASS_COUNT)):
        raise Batch16MarginBaseScorerCacheError("described cells contain a class outside 0..4")
    for name, margin in (
        ("target_margins", target_margin),
        ("described_margins", described_margin),
    ):
        if not bool(np.isfinite(margin).all()) or bool(np.any(margin < 0.0)):
            raise Batch16MarginBaseScorerCacheError(f"{name} must be finite and nonnegative")
    return target, target_margin, described, described_margin


def _checkpoint_file_row(path: Path, *, shape: list[int], dtype: str) -> dict[str, Any]:
    return {**file_identity(path), "shape": shape, "dtype": dtype}


def _validate_batch_checkpoint(
    *,
    root: Path,
    checkpoint: Mapping[str, Any],
    start: int,
    stop: int,
    preflight: Mapping[str, Any],
    target_labels: np.memmap,
    expected_source_sha256: str | None,
    expected_target_input_sha256: str | None,
    expected_v15_camera_sha256: str | None,
    expected_v15_input_sha256: str | None,
) -> dict[str, Any]:
    if checkpoint.get("schema") != BATCH_CHECKPOINT_SCHEMA:
        raise Batch16MarginBaseScorerCacheError("batch checkpoint schema differs")
    _verify_seal(checkpoint, field="batch_receipt_sha256")
    if (
        checkpoint.get("pair_range") != [start, stop]
        or checkpoint.get("pair_ids") != list(range(start, stop))
        or checkpoint.get("preflight_sha256") != preflight["preflight_sha256"]
        or checkpoint.get("scorer_pair_batch_size") != stop - start
        or checkpoint.get("upstream_default_batch_size") != PRODUCTION_BATCH_PAIRS
        or checkpoint.get("target_argmax_equal_owned_g46") is not True
        or checkpoint.get("dense_fields_candidate_payload_allowed") is not False
        or checkpoint.get("score_claim") is not False
    ):
        raise Batch16MarginBaseScorerCacheError("batch checkpoint range/authority differs")
    _require_sha(checkpoint.get("source_pair_batch_sha256"), "checkpoint source_pair_batch_sha256")
    _require_sha(checkpoint.get("target_scorer_input_sha256"), "checkpoint target_scorer_input_sha256")
    if expected_source_sha256 is not None and checkpoint.get("source_pair_batch_sha256") != _require_sha(
        expected_source_sha256, "source_pair_batch_sha256"
    ):
        raise Batch16MarginBaseScorerCacheError("resumed batch source bytes differ")
    if expected_target_input_sha256 is not None and checkpoint.get("target_scorer_input_sha256") != _require_sha(
        expected_target_input_sha256, "target_scorer_input_sha256"
    ):
        raise Batch16MarginBaseScorerCacheError("resumed target scorer input differs")
    owned_v15_camera_sha256 = _expected_v15_camera_sha256(
        preflight,
        start=start,
        stop=stop,
    )
    if checkpoint.get("v15_camera_sha256") != owned_v15_camera_sha256:
        raise Batch16MarginBaseScorerCacheError("batch V15 camera differs from owned identity")
    if expected_v15_camera_sha256 is not None and checkpoint.get("v15_camera_sha256") != _require_sha(
        expected_v15_camera_sha256, "v15_camera_sha256"
    ):
        raise Batch16MarginBaseScorerCacheError("resumed V15 camera bytes differ")
    _require_sha(checkpoint.get("v15_scorer_input_sha256"), "checkpoint v15_scorer_input_sha256")
    if expected_v15_input_sha256 is not None and checkpoint.get("v15_scorer_input_sha256") != _require_sha(
        expected_v15_input_sha256, "v15_scorer_input_sha256"
    ):
        raise Batch16MarginBaseScorerCacheError("resumed V15 scorer input differs")
    expected_target = np.ascontiguousarray(target_labels[start:stop])
    if checkpoint.get("g46_target_slice_sha256") != sha256_array_bytes(expected_target):
        raise Batch16MarginBaseScorerCacheError("batch G46 target slice differs")
    paths = _array_paths(root, start, stop)
    files = checkpoint.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "target_margins_f32",
        "described_cells_u8",
        "described_margins_f32",
    }:
        raise Batch16MarginBaseScorerCacheError("batch file bindings differ")
    shape = [stop - start, *preflight["scorer_hw"]]
    expected_dtypes = {
        "target_margins_f32": "float32_le",
        "described_cells_u8": "uint8",
        "described_margins_f32": "float32_le",
    }
    for name, dtype in expected_dtypes.items():
        actual = file_identity(paths[name])
        row = files[name]
        if (
            not isinstance(row, Mapping)
            or row.get("shape") != shape
            or row.get("dtype") != dtype
            or actual != {key: row[key] for key in ("path", "bytes", "sha256")}
        ):
            raise Batch16MarginBaseScorerCacheError(f"batch {start}:{stop} {name} differs")
    return dict(checkpoint)


def _write_batch(
    *,
    root: Path,
    start: int,
    stop: int,
    preflight: Mapping[str, Any],
    prepared: PreparedBatchV1,
    products: BatchProductsV1,
    target_labels: np.memmap,
) -> dict[str, Any]:
    scorer_hw = tuple(int(item) for item in preflight["scorer_hw"])
    expected_target = np.ascontiguousarray(target_labels[start:stop])
    target, target_margin, described, described_margin = _validate_scored_arrays(
        target_cells=products.target_cells_u8,
        target_margins=products.target_margins_f32,
        described_cells=products.described_cells_u8,
        described_margins=products.described_margins_f32,
        expected_target=expected_target,
        scorer_hw=scorer_hw,
    )
    paths = _array_paths(root, start, stop)
    write_immutable_array(paths["target_margins_f32"], target_margin)
    write_immutable_array(paths["described_cells_u8"], described)
    write_immutable_array(paths["described_margins_f32"], described_margin)
    shape = [stop - start, *scorer_hw]
    body = {
        "schema": BATCH_CHECKPOINT_SCHEMA,
        "preflight_sha256": preflight["preflight_sha256"],
        "pair_range": [start, stop],
        "pair_ids": list(range(start, stop)),
        "scorer_pair_batch_size": stop - start,
        "upstream_default_batch_size": PRODUCTION_BATCH_PAIRS,
        "final_partial_batch": stop - start != PRODUCTION_BATCH_PAIRS,
        "source_pair_batch_sha256": _require_sha(
            prepared.source_pair_batch_sha256,
            "source_pair_batch_sha256",
        ),
        "target_scorer_input_sha256": _require_sha(
            prepared.target_scorer_input_sha256,
            "target_scorer_input_sha256",
        ),
        "v15_camera_sha256": _require_sha(prepared.v15_camera_sha256, "v15_camera_sha256"),
        "v15_scorer_input_sha256": _require_sha(
            prepared.v15_scorer_input_sha256,
            "v15_scorer_input_sha256",
        ),
        "g46_target_slice_sha256": sha256_array_bytes(expected_target),
        "fresh_target_argmax_sha256": sha256_array_bytes(target),
        "target_argmax_equal_owned_g46": True,
        "files": {
            "target_margins_f32": _checkpoint_file_row(
                paths["target_margins_f32"],
                shape=shape,
                dtype="float32_le",
            ),
            "described_cells_u8": _checkpoint_file_row(
                paths["described_cells_u8"],
                shape=shape,
                dtype="uint8",
            ),
            "described_margins_f32": _checkpoint_file_row(
                paths["described_margins_f32"],
                shape=shape,
                dtype="float32_le",
            ),
        },
        "live_r": {
            "implementation": "upstream.modules.SegNet.preprocess_input",
            "frame_selector": "last_frame_index_1",
            "camera_input_dtype": "uint8",
            "scorer_input_dtype": "float32",
            "interpolation": "torch.nn.functional.interpolate(mode='bilinear')",
            "intermediate_uint8_roundtrip": False,
        },
        "committed_atomically": True,
        "encoder_only": True,
        "dense_fields_candidate_payload_allowed": False,
        "scorer_weights_candidate_payload_allowed": False,
        "candidate_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    checkpoint = _seal(body, field="batch_receipt_sha256")
    write_immutable_json(paths["checkpoint"], checkpoint)
    return _validate_batch_checkpoint(
        root=root,
        checkpoint=_load_json(paths["checkpoint"], "batch checkpoint"),
        start=start,
        stop=stop,
        preflight=preflight,
        target_labels=target_labels,
        expected_source_sha256=prepared.source_pair_batch_sha256,
        expected_target_input_sha256=prepared.target_scorer_input_sha256,
        expected_v15_camera_sha256=prepared.v15_camera_sha256,
        expected_v15_input_sha256=prepared.v15_scorer_input_sha256,
    )


def _open_batch_array(
    *,
    checkpoint: Mapping[str, Any],
    name: str,
    scorer_hw: tuple[int, int],
) -> np.memmap:
    start, stop = checkpoint["pair_range"]
    dtype = np.uint8 if name == "described_cells_u8" else np.dtype("<f4")
    return np.memmap(
        checkpoint["files"][name]["path"],
        mode="r",
        dtype=dtype,
        shape=(stop - start, *scorer_hw),
    )


def _stage_paths(root: Path, index: int, start: int, stop: int) -> dict[str, Path]:
    stage = root / f"20_stage_{index:02d}_{start:04d}_{stop:04d}"
    return {
        "target_margins_f32": stage / "target_margins.f32",
        "described_cells_u8": stage / "described_cells.u8",
        "described_margins_f32": stage / "described_margins.f32",
        "checkpoint": stage / "stage_receipt.json",
    }


def _batch_fragments(
    batches: Sequence[Mapping[str, Any]],
    *,
    stage_start: int,
    stage_stop: int,
) -> list[tuple[Mapping[str, Any], int, int]]:
    rows: list[tuple[Mapping[str, Any], int, int]] = []
    covered = stage_start
    for batch in batches:
        batch_start, batch_stop = batch["pair_range"]
        start = max(stage_start, batch_start)
        stop = min(stage_stop, batch_stop)
        if start >= stop:
            continue
        if start != covered:
            raise Batch16MarginBaseScorerCacheError("stage batch fragments contain a gap")
        rows.append((batch, start - batch_start, stop - batch_start))
        covered = stop
    if covered != stage_stop:
        raise Batch16MarginBaseScorerCacheError("stage batch fragments do not cover the stage")
    return rows


def _fragment_receipt_rows(
    fragments: Sequence[tuple[Mapping[str, Any], int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    for batch, local_start, local_stop in fragments:
        count = local_stop - local_start
        rows.append(
            {
                "batch_pair_range": list(batch["pair_range"]),
                "batch_receipt_sha256": batch["batch_receipt_sha256"],
                "local_slice": [local_start, local_stop],
                "stage_slice": [offset, offset + count],
            }
        )
        offset += count
    return rows


def _fragment_dense_sha256(
    fragments: Sequence[tuple[Mapping[str, Any], int, int]],
    *,
    name: str,
    scorer_hw: tuple[int, int],
) -> str:
    digest = hashlib.sha256()
    for batch, local_start, local_stop in fragments:
        source = _open_batch_array(checkpoint=batch, name=name, scorer_hw=scorer_hw)
        chunk = np.ascontiguousarray(source[local_start:local_stop])
        digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()


def _validate_g51_stage_binding(
    preflight: Mapping[str, Any],
    *,
    index: int,
    start: int,
    stop: int,
) -> Mapping[str, Any]:
    expected = preflight["g51_y0_y1_custody"]["stages"][index]
    if (
        expected.get("stage_index") != index
        or expected.get("pair_range") != [start, stop]
        or expected.get("y0_y1_rederive_performed_by_g78") is not False
    ):
        raise Batch16MarginBaseScorerCacheError(f"G51 stage {index} binding differs")
    for name in ("manifest", "y0_u8", "y1_u8", "gt_poses_f32"):
        binding = expected.get(name)
        if not isinstance(binding, Mapping) or file_identity(str(binding.get("path"))) != {
            key: binding[key] for key in ("path", "bytes", "sha256")
        }:
            raise Batch16MarginBaseScorerCacheError(f"G51 stage {index} {name} changed")
    return expected


def _validate_stage_checkpoint(
    *,
    root: Path,
    checkpoint: Mapping[str, Any],
    index: int,
    start: int,
    stop: int,
    preflight: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
    target_labels: np.memmap,
) -> dict[str, Any]:
    if checkpoint.get("schema") != STAGE_CHECKPOINT_SCHEMA:
        raise Batch16MarginBaseScorerCacheError("stage checkpoint schema differs")
    _verify_seal(checkpoint, field="stage_receipt_sha256")
    if (
        checkpoint.get("stage_index") != index
        or checkpoint.get("pair_range") != [start, stop]
        or checkpoint.get("pair_ids") != list(range(start, stop))
        or checkpoint.get("preflight_sha256") != preflight["preflight_sha256"]
        or checkpoint.get("target_cells_reused_not_copied") is not True
        or checkpoint.get("g51_y0_y1_reused_not_rederived") is not True
        or checkpoint.get("dense_fields_candidate_payload_allowed") is not False
        or checkpoint.get("score_claim") is not False
    ):
        raise Batch16MarginBaseScorerCacheError("stage range/reuse/authority differs")
    fragments = _batch_fragments(batches, stage_start=start, stage_stop=stop)
    if checkpoint.get("batch_fragments") != _fragment_receipt_rows(fragments):
        raise Batch16MarginBaseScorerCacheError(f"stage {index} batch fragments differ")
    expected_g51 = _validate_g51_stage_binding(
        preflight,
        index=index,
        start=start,
        stop=stop,
    )
    if checkpoint.get("g51_y0_y1_stage") != expected_g51:
        raise Batch16MarginBaseScorerCacheError(f"stage {index} G51 custody differs")
    target = np.ascontiguousarray(target_labels[start:stop])
    target_binding = checkpoint.get("target_cells")
    expected_target_binding = {
        "source_path": preflight["target_custody"]["target_labels"]["path"],
        "global_bank_sha256": preflight["target_custody"]["target_labels"]["sha256"],
        "shape": [stop - start, *preflight["scorer_hw"]],
        "dtype": "uint8",
        "slice_sha256": sha256_array_bytes(target),
    }
    if not isinstance(target_binding, Mapping) or target_binding != expected_target_binding:
        raise Batch16MarginBaseScorerCacheError("stage target-cell binding differs")
    paths = _stage_paths(root, index, start, stop)
    files = checkpoint.get("files")
    expected_dtypes = {
        "target_margins_f32": "float32_le",
        "described_cells_u8": "uint8",
        "described_margins_f32": "float32_le",
    }
    shape = [stop - start, *preflight["scorer_hw"]]
    if not isinstance(files, Mapping) or set(files) != set(expected_dtypes):
        raise Batch16MarginBaseScorerCacheError("stage files differ")
    for name, dtype in expected_dtypes.items():
        row = files[name]
        actual = file_identity(paths[name])
        if (
            not isinstance(row, Mapping)
            or row.get("shape") != shape
            or row.get("dtype") != dtype
            or actual != {key: row[key] for key in ("path", "bytes", "sha256")}
        ):
            raise Batch16MarginBaseScorerCacheError(f"stage {index} {name} differs")
        if row["sha256"] != _fragment_dense_sha256(
            fragments,
            name=name,
            scorer_hw=tuple(int(item) for item in preflight["scorer_hw"]),
        ):
            raise Batch16MarginBaseScorerCacheError(f"stage {index} {name} differs from validated batch fragments")
    return dict(checkpoint)


def _assemble_stage(
    *,
    root: Path,
    index: int,
    preflight: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
    target_labels: np.memmap,
) -> dict[str, Any]:
    stage_pairs = int(preflight["stage_pairs"])
    start = index * stage_pairs
    stop = start + stage_pairs
    paths = _stage_paths(root, index, start, stop)
    if paths["checkpoint"].exists():
        return _validate_stage_checkpoint(
            root=root,
            checkpoint=_load_json(paths["checkpoint"], "stage checkpoint"),
            index=index,
            start=start,
            stop=stop,
            preflight=preflight,
            batches=batches,
            target_labels=target_labels,
        )
    scorer_hw = tuple(int(item) for item in preflight["scorer_hw"])
    fragments = _batch_fragments(batches, stage_start=start, stage_stop=stop)
    arrays: dict[str, np.ndarray] = {
        "target_margins_f32": np.empty((stage_pairs, *scorer_hw), dtype="<f4"),
        "described_cells_u8": np.empty((stage_pairs, *scorer_hw), dtype=np.uint8),
        "described_margins_f32": np.empty((stage_pairs, *scorer_hw), dtype="<f4"),
    }
    offset = 0
    for batch, local_start, local_stop in fragments:
        count = local_stop - local_start
        for name in arrays:
            source = _open_batch_array(checkpoint=batch, name=name, scorer_hw=scorer_hw)
            arrays[name][offset : offset + count] = source[local_start:local_stop]
        offset += count
    if offset != stage_pairs:
        raise Batch16MarginBaseScorerCacheError("stage assembly pair count differs")
    for name, array in arrays.items():
        write_immutable_array(paths[name], array)
    target = np.ascontiguousarray(target_labels[start:stop])
    g51_rows = preflight["g51_y0_y1_custody"]["stages"]
    g51 = g51_rows[index]
    body = {
        "schema": STAGE_CHECKPOINT_SCHEMA,
        "preflight_sha256": preflight["preflight_sha256"],
        "stage_index": index,
        "pair_range": [start, stop],
        "pair_ids": list(range(start, stop)),
        "batch_fragments": _fragment_receipt_rows(fragments),
        "target_cells": {
            "source_path": preflight["target_custody"]["target_labels"]["path"],
            "global_bank_sha256": preflight["target_custody"]["target_labels"]["sha256"],
            "shape": [stage_pairs, *scorer_hw],
            "dtype": "uint8",
            "slice_sha256": sha256_array_bytes(target),
        },
        "target_cells_reused_not_copied": True,
        "g51_y0_y1_stage": g51,
        "g51_y0_y1_reused_not_rederived": True,
        "files": {
            "target_margins_f32": _checkpoint_file_row(
                paths["target_margins_f32"],
                shape=[stage_pairs, *scorer_hw],
                dtype="float32_le",
            ),
            "described_cells_u8": _checkpoint_file_row(
                paths["described_cells_u8"],
                shape=[stage_pairs, *scorer_hw],
                dtype="uint8",
            ),
            "described_margins_f32": _checkpoint_file_row(
                paths["described_margins_f32"],
                shape=[stage_pairs, *scorer_hw],
                dtype="float32_le",
            ),
        },
        "committed_atomically": True,
        "immutable_on_resume": True,
        "encoder_only": True,
        "dense_fields_candidate_payload_allowed": False,
        "candidate_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    checkpoint = _seal(body, field="stage_receipt_sha256")
    write_immutable_json(paths["checkpoint"], checkpoint)
    return _validate_stage_checkpoint(
        root=root,
        checkpoint=_load_json(paths["checkpoint"], "stage checkpoint"),
        index=index,
        start=start,
        stop=stop,
        preflight=preflight,
        batches=batches,
        target_labels=target_labels,
    )


def materialize_margin_base_scorer_cache(
    *,
    preflight: Mapping[str, Any],
    prepare_batch: BatchPreparer,
    allowed_roots: Sequence[Path] = SSD_ROOTS,
) -> tuple[Path, dict[str, Any]]:
    """Resume global batch shards, assemble five stages, and seal aggregate."""

    reverify_preflight(preflight, allowed_roots=allowed_roots)
    root = Path(str(preflight["output_root"]))
    root.mkdir(parents=True, exist_ok=True)
    preflight_path = root / "00_preflight_receipt.json"
    write_immutable_json(preflight_path, preflight)
    target_binding = preflight["target_custody"]["target_labels"]
    target_labels = np.memmap(
        target_binding["path"],
        mode="r",
        dtype=np.uint8,
        shape=tuple(target_binding["shape"]),
    )
    pair_count = int(preflight["pair_count"])
    batch_pairs = int(preflight["scorer_batch_pairs"])
    batches: list[dict[str, Any]] = []
    for start in range(0, pair_count, batch_pairs):
        stop = min(start + batch_pairs, pair_count)
        pair_ids = tuple(range(start, stop))
        prepared = prepare_batch(pair_ids)
        _validate_prepared_batch(
            prepared,
            start=start,
            stop=stop,
            preflight=preflight,
        )
        paths = _array_paths(root, start, stop)
        if paths["checkpoint"].exists():
            row = _validate_batch_checkpoint(
                root=root,
                checkpoint=_load_json(paths["checkpoint"], "batch checkpoint"),
                start=start,
                stop=stop,
                preflight=preflight,
                target_labels=target_labels,
                expected_source_sha256=prepared.source_pair_batch_sha256,
                expected_target_input_sha256=prepared.target_scorer_input_sha256,
                expected_v15_camera_sha256=prepared.v15_camera_sha256,
                expected_v15_input_sha256=prepared.v15_scorer_input_sha256,
            )
        else:
            row = _write_batch(
                root=root,
                start=start,
                stop=stop,
                preflight=preflight,
                prepared=prepared,
                products=prepared.infer(),
                target_labels=target_labels,
            )
        batches.append(row)
    stages = [
        _assemble_stage(
            root=root,
            index=index,
            preflight=preflight,
            batches=batches,
            target_labels=target_labels,
        )
        for index in range(int(preflight["stage_count"]))
    ]
    chain = hashlib.sha256()
    stage_rows: list[dict[str, Any]] = []
    for index, stage in enumerate(stages):
        chain.update(bytes.fromhex(stage["stage_receipt_sha256"]))
        start, stop = stage["pair_range"]
        checkpoint_path = _stage_paths(root, index, start, stop)["checkpoint"]
        stage_rows.append(
            {
                **file_identity(checkpoint_path),
                "stage_index": index,
                "pair_range": [start, stop],
                "stage_receipt_sha256": stage["stage_receipt_sha256"],
                "digest_chain_sha256": chain.hexdigest(),
            }
        )
    batch_rows = []
    for batch in batches:
        start, stop = batch["pair_range"]
        path = _array_paths(root, start, stop)["checkpoint"]
        batch_rows.append(
            {
                **file_identity(path),
                "pair_range": [start, stop],
                "batch_receipt_sha256": batch["batch_receipt_sha256"],
            }
        )
    body = {
        "schema": AGGREGATE_SCHEMA,
        "run_id": preflight["run_id"],
        "evidence_axis": preflight["evidence_axis"],
        "preflight": file_identity(preflight_path),
        "preflight_sha256": preflight["preflight_sha256"],
        "pair_count": pair_count,
        "stage_pairs": preflight["stage_pairs"],
        "stage_count": preflight["stage_count"],
        "scorer_batch_pairs": batch_pairs,
        "batch_count": len(batches),
        "scorer_hw": preflight["scorer_hw"],
        "class_count": preflight["class_count"],
        "target_custody": preflight["target_custody"],
        "g51_y0_y1_custody": preflight["g51_y0_y1_custody"],
        "semantic_custody": preflight["semantic_custody"],
        "batches": batch_rows,
        "stages": stage_rows,
        "stage_digest_chain_sha256": chain.hexdigest(),
        "coverage": {
            "pair_range": [0, pair_count],
            "chronological_contiguous": True,
            "global_batch_geometry_preserved_across_120_pair_boundaries": True,
            "target_argmax_equal_owned_g46_all_batches": True,
            "five_stage_cache_complete": len(stages) == PRODUCTION_STAGE_COUNT,
        },
        "closed_blockers": list(preflight["blockers_closed_by_successful_aggregate"]),
        "remaining_g72_blockers_unmodified": list(REMAINING_G72_BLOCKERS),
        "research_only": True,
        "encoder_only": True,
        "dense_fields_candidate_payload_allowed": False,
        "scorer_weights_candidate_payload_allowed": False,
        "candidate_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "cleanup_certificate": {
            "status": "PRESERVED_NO_DELETE",
            "output_root": str(root),
            "rebuild_argv": preflight["run_argv"],
            "preflight_sha256": preflight["preflight_sha256"],
            "policy": "certify_or_block",
        },
    }
    aggregate = _seal(body, field="aggregate_receipt_sha256")
    aggregate_path = root / "aggregate_receipt.json"
    write_immutable_json(aggregate_path, aggregate)
    loader = MarginBaseScorerCacheLoaderV1.open(
        aggregate_path,
        expected_sha256=sha256_file(aggregate_path),
        allowed_roots=allowed_roots,
    )
    if len(list(loader.iter_stages())) != int(preflight["stage_count"]):
        raise Batch16MarginBaseScorerCacheError("strict aggregate reopen lost a stage")
    return aggregate_path, aggregate


class MarginBaseScorerCacheLoaderV1:
    """Strict aggregate/stage loader for G72 and later encoder consumers."""

    def __init__(
        self,
        path: Path,
        receipt: Mapping[str, Any],
        *,
        allowed_roots: Sequence[Path],
    ) -> None:
        self.receipt_path = path
        self.receipt = dict(receipt)
        self.allowed_roots = tuple(allowed_roots)
        self._stages: list[dict[str, Any]] = []
        self._validate()

    @classmethod
    def open(
        cls,
        aggregate_receipt_path: str | os.PathLike[str],
        *,
        expected_sha256: str,
        allowed_roots: Sequence[Path] = SSD_ROOTS,
    ) -> MarginBaseScorerCacheLoaderV1:
        path = Path(aggregate_receipt_path).expanduser().resolve()
        if sha256_file(path) != _require_sha(
            expected_sha256,
            "expected_sha256",
        ):
            raise Batch16MarginBaseScorerCacheError("aggregate file SHA-256 differs")
        return cls(
            path,
            _load_json(path, "aggregate receipt"),
            allowed_roots=allowed_roots,
        )

    def _validate(self) -> None:
        if self.receipt.get("schema") != AGGREGATE_SCHEMA:
            raise Batch16MarginBaseScorerCacheError("aggregate schema differs")
        _verify_seal(self.receipt, field="aggregate_receipt_sha256")
        if (
            self.receipt.get("research_only") is not True
            or self.receipt.get("encoder_only") is not True
            or self.receipt.get("dense_fields_candidate_payload_allowed") is not False
            or self.receipt.get("scorer_weights_candidate_payload_allowed") is not False
            or self.receipt.get("candidate_claim") is not False
            or self.receipt.get("score_claim") is not False
            or self.receipt.get("promotion_eligible") is not False
            or self.receipt.get("pointer_moved") is not False
        ):
            raise Batch16MarginBaseScorerCacheError("aggregate false-authority fences differ")
        preflight_binding = self.receipt.get("preflight")
        if not isinstance(preflight_binding, Mapping):
            raise Batch16MarginBaseScorerCacheError("aggregate preflight binding is absent")
        preflight_path = Path(str(preflight_binding.get("path")))
        if file_identity(preflight_path) != {key: preflight_binding[key] for key in ("path", "bytes", "sha256")}:
            raise Batch16MarginBaseScorerCacheError("aggregate preflight file differs")
        self.preflight = _load_json(preflight_path, "aggregate preflight")
        reverify_preflight(self.preflight, allowed_roots=self.allowed_roots)
        if self.preflight["preflight_sha256"] != self.receipt.get("preflight_sha256"):
            raise Batch16MarginBaseScorerCacheError("aggregate names another preflight")
        pair_count = int(self.preflight["pair_count"])
        scorer_hw = tuple(int(item) for item in self.preflight["scorer_hw"])
        batch_pairs = int(self.preflight["scorer_batch_pairs"])
        expected_batch_count = (pair_count + batch_pairs - 1) // batch_pairs
        expected_coverage = {
            "pair_range": [0, pair_count],
            "chronological_contiguous": True,
            "global_batch_geometry_preserved_across_120_pair_boundaries": True,
            "target_argmax_equal_owned_g46_all_batches": True,
            "five_stage_cache_complete": int(self.preflight["stage_count"]) == PRODUCTION_STAGE_COUNT,
        }
        expected_cleanup = {
            "status": "PRESERVED_NO_DELETE",
            "output_root": str(self.preflight["output_root"]),
            "rebuild_argv": self.preflight["run_argv"],
            "preflight_sha256": self.preflight["preflight_sha256"],
            "policy": "certify_or_block",
        }
        exact_top_level = {
            "run_id": self.preflight["run_id"],
            "evidence_axis": self.preflight["evidence_axis"],
            "pair_count": pair_count,
            "stage_pairs": self.preflight["stage_pairs"],
            "stage_count": self.preflight["stage_count"],
            "scorer_batch_pairs": batch_pairs,
            "batch_count": expected_batch_count,
            "scorer_hw": self.preflight["scorer_hw"],
            "class_count": self.preflight["class_count"],
            "target_custody": self.preflight["target_custody"],
            "g51_y0_y1_custody": self.preflight["g51_y0_y1_custody"],
            "semantic_custody": self.preflight["semantic_custody"],
            "coverage": expected_coverage,
            "closed_blockers": list(self.preflight["blockers_closed_by_successful_aggregate"]),
            "remaining_g72_blockers_unmodified": list(REMAINING_G72_BLOCKERS),
            "cleanup_certificate": expected_cleanup,
        }
        if any(self.receipt.get(key) != value for key, value in exact_top_level.items()):
            raise Batch16MarginBaseScorerCacheError("aggregate top-level custody/closure differs")
        target = self.preflight["target_custody"]["target_labels"]
        self._target = np.memmap(
            target["path"],
            mode="r",
            dtype=np.uint8,
            shape=tuple(target["shape"]),
        )
        batch_rows = self.receipt.get("batches")
        if not isinstance(batch_rows, list) or len(batch_rows) != expected_batch_count:
            raise Batch16MarginBaseScorerCacheError("aggregate batch rows are absent")
        expected_start = 0
        parsed_batches: list[dict[str, Any]] = []
        for index, binding in enumerate(batch_rows):
            if not isinstance(binding, Mapping) or set(binding) != {
                "path",
                "bytes",
                "sha256",
                "pair_range",
                "batch_receipt_sha256",
            }:
                raise Batch16MarginBaseScorerCacheError("aggregate batch binding is invalid")
            path = Path(str(binding.get("path")))
            if file_identity(path) != {key: binding[key] for key in ("path", "bytes", "sha256")}:
                raise Batch16MarginBaseScorerCacheError(f"aggregate batch {index} file differs")
            start, stop = binding["pair_range"]
            if start != expected_start or stop != min(
                start + int(self.preflight["scorer_batch_pairs"]),
                pair_count,
            ):
                raise Batch16MarginBaseScorerCacheError("aggregate batch chronology differs")
            parsed = _validate_batch_checkpoint(
                root=Path(str(self.preflight["output_root"])),
                checkpoint=_load_json(path, f"aggregate batch {index}"),
                start=start,
                stop=stop,
                preflight=self.preflight,
                target_labels=self._target,
                expected_source_sha256=None,
                expected_target_input_sha256=None,
                expected_v15_camera_sha256=None,
                expected_v15_input_sha256=None,
            )
            if parsed["batch_receipt_sha256"] != binding.get("batch_receipt_sha256"):
                raise Batch16MarginBaseScorerCacheError("aggregate batch self-hash differs")
            parsed_batches.append(parsed)
            expected_start = stop
        if expected_start != pair_count:
            raise Batch16MarginBaseScorerCacheError("aggregate batch coverage differs")
        stage_rows = self.receipt.get("stages")
        if not isinstance(stage_rows, list) or len(stage_rows) != int(self.preflight["stage_count"]):
            raise Batch16MarginBaseScorerCacheError("aggregate stage count differs")
        chain = hashlib.sha256()
        root = Path(str(self.preflight["output_root"]))
        for index, binding in enumerate(stage_rows):
            if not isinstance(binding, Mapping) or set(binding) != {
                "path",
                "bytes",
                "sha256",
                "stage_index",
                "pair_range",
                "stage_receipt_sha256",
                "digest_chain_sha256",
            }:
                raise Batch16MarginBaseScorerCacheError("aggregate stage binding is invalid")
            path = Path(str(binding.get("path")))
            if file_identity(path) != {key: binding[key] for key in ("path", "bytes", "sha256")}:
                raise Batch16MarginBaseScorerCacheError(f"aggregate stage {index} file differs")
            start = index * int(self.preflight["stage_pairs"])
            stop = start + int(self.preflight["stage_pairs"])
            if binding["stage_index"] != index or binding["pair_range"] != [start, stop]:
                raise Batch16MarginBaseScorerCacheError("aggregate stage chronology differs")
            parsed = _validate_stage_checkpoint(
                root=root,
                checkpoint=_load_json(path, f"aggregate stage {index}"),
                index=index,
                start=start,
                stop=stop,
                preflight=self.preflight,
                batches=parsed_batches,
                target_labels=self._target,
            )
            if parsed["stage_receipt_sha256"] != binding.get("stage_receipt_sha256"):
                raise Batch16MarginBaseScorerCacheError("aggregate stage self-hash differs")
            chain.update(bytes.fromhex(parsed["stage_receipt_sha256"]))
            if chain.hexdigest() != binding.get("digest_chain_sha256"):
                raise Batch16MarginBaseScorerCacheError("aggregate stage digest chain differs")
            self._stages.append(parsed)
        if chain.hexdigest() != self.receipt.get("stage_digest_chain_sha256"):
            raise Batch16MarginBaseScorerCacheError("aggregate final stage chain differs")
        self.pair_count = pair_count
        self.stage_pairs = int(self.preflight["stage_pairs"])
        self.scorer_batch_pairs = int(self.preflight["scorer_batch_pairs"])
        self.scorer_hw = scorer_hw

    def iter_stages(self) -> Sequence[MarginBaseScorerStageV1]:
        rows: list[MarginBaseScorerStageV1] = []
        root = Path(str(self.preflight["output_root"]))
        for index, stage in enumerate(self._stages):
            start, stop = stage["pair_range"]
            paths = _stage_paths(root, index, start, stop)
            shape = (stop - start, *self.scorer_hw)
            rows.append(
                MarginBaseScorerStageV1(
                    stage_index=index,
                    pair_range=(start, stop),
                    target_cells_u8=self._target[start:stop],
                    target_margins_f32=np.memmap(
                        paths["target_margins_f32"],
                        mode="r",
                        dtype="<f4",
                        shape=shape,
                    ),
                    described_cells_u8=np.memmap(
                        paths["described_cells_u8"],
                        mode="r",
                        dtype=np.uint8,
                        shape=shape,
                    ),
                    described_margins_f32=np.memmap(
                        paths["described_margins_f32"],
                        mode="r",
                        dtype="<f4",
                        shape=shape,
                    ),
                )
            )
        return tuple(rows)


__all__ = [
    "AGGREGATE_SCHEMA",
    "BATCH_CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "DEFAULT_REQUIRED_FREE_BYTES",
    "EVIDENCE_AXIS",
    "PREFLIGHT_SCHEMA",
    "PRODUCTION_BATCH_PAIRS",
    "PRODUCTION_PAIR_COUNT",
    "PRODUCTION_SCORER_HW",
    "PRODUCTION_STAGE_COUNT",
    "PRODUCTION_STAGE_PAIRS",
    "SSD_ROOTS",
    "STAGE_CHECKPOINT_SCHEMA",
    "Batch16MarginBaseScorerCacheError",
    "BatchProductsV1",
    "MarginBaseScorerCacheLoaderV1",
    "MarginBaseScorerStageV1",
    "PreparedBatchV1",
    "canonical_json_bytes",
    "file_identity",
    "materialize_margin_base_scorer_cache",
    "payload_sha256",
    "reverify_preflight",
    "seal_preflight",
    "sha256_array_bytes",
    "sha256_file",
    "storage_preflight",
    "write_immutable_json",
]
