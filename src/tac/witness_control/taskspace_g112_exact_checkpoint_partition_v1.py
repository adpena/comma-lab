# SPDX-License-Identifier: MIT
"""Strict source-side partition of a fresh G111 V9 deploy checkpoint.

The physical G111 deploy checkpoint contains two different ownership domains:

* the exact G105 V9 semantic program (shared tensors plus odd Y1 code rows);
* a jointly trained generated-Y1 pose initializer (``xi_stored`` and ``dxi``).

This module first requires the deploy checkpoint's immutable, recursively
reopened zero-parent-to-current physical checkpoint ancestry and its exact
full-state companion.  It then separates the two ownership domains without
silently dropping any learned tensor.  The semantic child stores only
``code_y1[600, D]``.  Its strict loader uses G105's
``compile_from_y1_state`` API directly and never reconstructs even rows.  The
odd-only compile must produce the same packet as the source checkpoint's
legacy interleaved projection.  The folded pose table is explicitly an
initializer which requires a real post-G105 refit; it is never represented as
a final candidate payload.

No scorer is run here and no score or archive claim is made.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.witness_control.fresh_producer_lineage_v1 import (
    FRESH_PHYSICAL_CHECKPOINT_NODE_SCHEMA,
    FRESH_PRODUCER_LINEAGE_SCHEMA,
    FreshProducerLineageV1Error,
    FreshProducerPhysicalCheckpointChainV1,
    open_fresh_physical_checkpoint_chain_v1,
)
from tac.witness_control.taskspace_v9_training_target_binding_v1 import (
    CHECKPOINT_PROJECTION_KEY,
    CHECKPOINT_PROJECTION_SHA_KEY,
    V9TrainingTargetBindingError,
    checkpoint_target_arrays_from_projection,
    reopen_v9_training_target_projection,
)
from tac.witness_control.taskspace_v9_training_target_capsule_v1 import (
    PRODUCTION_BATCH_PAIRS,
    PRODUCTION_PAIR_COUNT,
)
from tac.witness_dsl import (
    taskspace_g105_exact_v9_semantic_root_adapter_v1 as g105_adapter,
)

SCHEMA: Final = "tac.g112_exact_checkpoint_partition.v2"
SEMANTIC_CHILD_SCHEMA: Final = "tac.g112_g105_semantic_odd_checkpoint.v1"
INITIALIZER_SCHEMA: Final = "tac.g112_generated_y1_pose_initializer.v1"
POSE_CHECKPOINT_CONTRACT_SCHEMA: Final = "tac.v9_pose_carrier_checkpoint_contract.v2"
Y1_SELECTED_PREIMAGE_SCHEMA: Final = "tac.v10_factor2_selected_preimage.v1"
SEMANTIC_CHILD_NAME: Final = "10_g105_semantic_child.npz"
INITIALIZER_NAME: Final = "20_generated_y1_pose_initializer.npz"
RECEIPT_NAME: Final = "30_g112_partition_receipt.json"
POSE_TENSOR_KEYS: Final = frozenset({"pose_carrier.xi_stored", "pose_carrier.dxi"})
POSE_CONFIG_KEYS: Final = frozenset(
    {
        "__cfg_pose_carrier_contract_schema",
        "__cfg_pose_carrier",
        "__cfg_pose_carrier_source",
        "__cfg_pose_carrier_residual_mode",
        "__cfg_pose_carrier_residual_scale",
        "__cfg_pose_carrier_s_t",
        "__cfg_pose_carrier_s_r",
        "__cfg_pose_carrier_pitch",
        "__cfg_pose_carrier_native_hw",
        "__cfg_pose_carrier_xi_formula",
        "__cfg_pose_carrier_y1_selected_preimage_schema",
    }
)
INITIALIZER_ARRAY_KEYS: Final = frozenset(
    {
        "xi_init",
        "__schema",
        "__status",
        "__pair_count",
        "__pose_dim",
        "__pose_carrier_source",
        "__pose_carrier_residual_mode",
        "__pose_carrier_residual_scale",
        "__pose_carrier_s_t",
        "__pose_carrier_s_r",
        "__pose_carrier_pitch",
        "__pose_carrier_native_hw",
        "__pose_carrier_xi_formula",
        "__pose_carrier_y1_selected_preimage_schema",
        "__semantic_packet_sha256",
        "__g109_target_projection_json",
        "__g109_target_projection_sha256",
        "__requires_post_g105_refit",
        "__candidate_payload_eligible",
        "__score_claim",
        "__pointer_mutation_allowed",
    }
)
SSD_ROOTS: Final = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)
_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
SEMANTIC_ODD_CODE_KEY: Final = "code_y1"
SEMANTIC_CHILD_META_KEYS: Final = frozenset(
    {
        "__g112_semantic_child_schema",
        "__g112_code_projection",
        "__g112_g105_adapter",
        "__g112_pair_count",
        "__g112_modulation_dim",
        "__g112_semantic_packet_sha256",
        "__g112_candidate_owned_code_rows",
    }
)


class G112CheckpointPartitionError(ValueError):
    """The physical checkpoint, partition, or immutable output failed closed."""


@dataclass(frozen=True, slots=True)
class G112CheckpointPartitionResultV1:
    """Published identities for one complete G112 materialization."""

    output_root: Path
    semantic_child_path: Path
    semantic_child_sha256: str
    initializer_path: Path
    initializer_sha256: str
    receipt_path: Path
    receipt_sha256: str
    semantic_packet_sha256: str
    source_checkpoint_sha256: str
    source_resume_checkpoint_sha256: str
    fresh_lineage_checkpoint_id_sha256: str


@dataclass(frozen=True, slots=True)
class G112PartitionReceiptV2:
    """Strict receipt reopen binding both children to one physical source pair."""

    receipt_path: Path
    receipt_sha256: str
    receipt_bytes: int
    semantic_child: G112SemanticChildV1
    initializer: G112PoseInitializerV1
    source_chain: FreshProducerPhysicalCheckpointChainV1
    semantic_packet_sha256: str


@dataclass(frozen=True, slots=True)
class G112SemanticChildV1:
    """Strictly reopened odd-only child and its exact G105 inputs."""

    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_bytes: int
    semantic_packet: bytes
    semantic_packet_sha256: str
    shared_params: dict[str, np.ndarray]
    code_y1: np.ndarray
    g105_scalars: dict[str, object]


@dataclass(frozen=True, slots=True)
class G112PoseInitializerV1:
    """Strictly reopened non-candidate initializer for real post-G105 refit."""

    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_bytes: int
    xi_init: np.ndarray
    semantic_packet_sha256: str
    target_projection_sha256: str
    residual_scale: float
    s_t: float
    s_r: float
    pitch: float
    native_hw: tuple[int, int]
    selected_preimage_schema: str = Y1_SELECTED_PREIMAGE_SCHEMA
    requires_post_g105_refit: bool = True
    candidate_payload_eligible: bool = False
    score_claim: bool = False


def _sha256(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise G112CheckpointPartitionError(f"{name} must be a canonical lowercase SHA-256")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise G112CheckpointPartitionError("receipt value is not finite canonical ASCII JSON") from exc


def _scalar(arrays: Mapping[str, np.ndarray], key: str) -> object:
    try:
        value = np.asarray(arrays[key])
    except KeyError as exc:
        raise G112CheckpointPartitionError(f"checkpoint is missing required config {key}") from exc
    if value.size != 1:
        raise G112CheckpointPartitionError(f"checkpoint config {key} must contain exactly one scalar")
    return value.item()


def _array_identity(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": array.dtype.str,
        "shape": [int(item) for item in array.shape],
        "nbytes": int(array.nbytes),
        "sha256": _sha256(array.tobytes(order="C")),
    }


def _strict_npz_members(payload: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if not members or len(names) != len(set(names)):
                raise G112CheckpointPartitionError("checkpoint NPZ has no members or has duplicate ZIP members")
            for member in members:
                name = member.filename
                if (
                    member.is_dir()
                    or "/" in name
                    or "\\" in name
                    or not name.endswith(".npy")
                    or name in {".npy", "..npy"}
                ):
                    raise G112CheckpointPartitionError("checkpoint NPZ has a noncanonical member name")
    except (OSError, zipfile.BadZipFile) as exc:
        raise G112CheckpointPartitionError("checkpoint is not a strict NPZ ZIP") from exc


def _open_physical_checkpoint(
    checkpoint: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        raise G112CheckpointPartitionError("checkpoint path must be absolute physical custody")
    expected = _require_sha256(
        expected_sha256,
        name="source checkpoint",
    )
    try:
        before = path.lstat()
    except OSError as exc:
        raise G112CheckpointPartitionError("source checkpoint is not readable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise G112CheckpointPartitionError("source checkpoint must be a regular non-symlink file")
    payload = path.read_bytes()
    observed_sha = _sha256(payload)
    if observed_sha != expected:
        raise G112CheckpointPartitionError("source checkpoint physical SHA-256 differs")
    _strict_npz_members(payload)
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]).copy(order="C") for key in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise G112CheckpointPartitionError("source checkpoint contains an invalid NPY member") from exc
    after = path.lstat()
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or _sha256(path.read_bytes()) != expected
    ):
        raise G112CheckpointPartitionError("source checkpoint changed during recursive reopen")
    for key, value in arrays.items():
        if value.dtype.hasobject:
            raise G112CheckpointPartitionError(f"checkpoint array {key} has forbidden object dtype")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise G112CheckpointPartitionError(f"checkpoint array {key} contains a non-finite value")
    return arrays, {
        "path": str(path),
        "bytes": len(payload),
        "sha256": observed_sha,
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "mtime_ns": int(after.st_mtime_ns),
        "regular_file": True,
        "symlink": False,
        "reopened_unchanged": True,
    }


def _checkpoint_views(
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    params = {key: np.asarray(value) for key, value in arrays.items() if not key.startswith("__")}
    scalars: dict[str, object] = {}
    for key, value in arrays.items():
        if not key.startswith("__"):
            continue
        raw = np.asarray(value)
        scalars[key] = raw.item() if raw.size == 1 else raw.copy(order="C")
    return params, scalars


def _validate_pose_contract(
    params: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, float], np.ndarray]:
    observed_pose_tensors = {key for key in params if key.startswith("pose_carrier.")}
    if observed_pose_tensors != POSE_TENSOR_KEYS:
        raise G112CheckpointPartitionError(
            f"pose learned tensor set is not exact; observed={sorted(observed_pose_tensors)}"
        )
    observed_pose_config = {key for key in arrays if key.startswith("__cfg_pose_carrier")}
    if observed_pose_config != POSE_CONFIG_KEYS:
        missing = sorted(POSE_CONFIG_KEYS - observed_pose_config)
        extra = sorted(observed_pose_config - POSE_CONFIG_KEYS)
        raise G112CheckpointPartitionError(f"pose config set is not exact; missing={missing}, extra={extra}")
    exact = {
        "__cfg_pose_carrier_contract_schema": POSE_CHECKPOINT_CONTRACT_SCHEMA,
        "__cfg_pose_carrier": 1,
        "__cfg_pose_carrier_source": "generated_y1",
        "__cfg_pose_carrier_residual_mode": "table",
        "__cfg_pose_carrier_xi_formula": "xi_stored+residual_scale*dxi",
        "__cfg_pose_carrier_y1_selected_preimage_schema": Y1_SELECTED_PREIMAGE_SCHEMA,
    }
    for key, expected in exact.items():
        if _scalar(arrays, key) != expected:
            raise G112CheckpointPartitionError(f"checkpoint pose config differs at {key}")
    native_hw = np.asarray(arrays["__cfg_pose_carrier_native_hw"])
    if (
        native_hw.dtype != np.dtype(np.int64)
        or native_hw.shape != (2,)
        or tuple(int(item) for item in native_hw) != (874, 1164)
    ):
        raise G112CheckpointPartitionError("pose native HW must be exact int64 [874,1164]")
    values: dict[str, float] = {}
    for key in (
        "__cfg_pose_carrier_residual_scale",
        "__cfg_pose_carrier_s_t",
        "__cfg_pose_carrier_s_r",
        "__cfg_pose_carrier_pitch",
    ):
        try:
            value = float(_scalar(arrays, key))
        except (TypeError, ValueError) as exc:
            raise G112CheckpointPartitionError(f"checkpoint pose config {key} must be numeric") from exc
        if not math.isfinite(value):
            raise G112CheckpointPartitionError(f"checkpoint pose config {key} must be finite")
        values[key] = value
    if values["__cfg_pose_carrier_residual_scale"] <= 0.0:
        raise G112CheckpointPartitionError("pose residual scale must be finite and positive")
    if values["__cfg_pose_carrier_s_t"] <= 0.0:
        raise G112CheckpointPartitionError("pose translation calibration must be finite and positive")
    if values["__cfg_pose_carrier_s_r"] < 0.0:
        raise G112CheckpointPartitionError("pose rotation calibration must be finite and nonnegative")
    for key in sorted(POSE_TENSOR_KEYS):
        value = np.asarray(params[key])
        if (
            value.dtype != np.dtype(np.float32)
            or value.shape != (PRODUCTION_PAIR_COUNT, 6)
            or not value.flags.c_contiguous
            or not np.isfinite(value).all()
        ):
            raise G112CheckpointPartitionError(f"{key} must be finite C-contiguous float32[600,6]")
    xi_init = np.ascontiguousarray(
        np.asarray(params["pose_carrier.xi_stored"], dtype=np.float64)
        + values["__cfg_pose_carrier_residual_scale"] * np.asarray(params["pose_carrier.dxi"], dtype=np.float64),
        dtype=np.float64,
    )
    if not np.isfinite(xi_init).all():
        raise G112CheckpointPartitionError("folded xi initializer contains a non-finite value")
    return values, xi_init


def _validate_target_custody(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    projection_json = _scalar(arrays, CHECKPOINT_PROJECTION_KEY)
    projection_sha = _scalar(arrays, CHECKPOINT_PROJECTION_SHA_KEY)
    authority_sha = _scalar(arrays, "__cfg_target_authority_sha256")
    if not isinstance(projection_json, str) or not isinstance(projection_sha, str):
        raise G112CheckpointPartitionError("checkpoint G109 projection config has the wrong scalar type")
    _require_sha256(authority_sha, name="active target authority")
    try:
        projection = reopen_v9_training_target_projection(
            projection_json=projection_json,
            expected_projection_sha256=projection_sha,
        )
        expected = checkpoint_target_arrays_from_projection(
            projection,
            active_target_authority_sha256=authority_sha,
            verdict_batch=PRODUCTION_BATCH_PAIRS,
        )
    except (OSError, KeyError, TypeError, ValueError, V9TrainingTargetBindingError) as exc:
        raise G112CheckpointPartitionError("checkpoint physical G109 projection failed recursive reopen") from exc
    if (
        projection.get("pair_count") != PRODUCTION_PAIR_COUNT
        or projection.get("scorer_pair_batch_size") != PRODUCTION_BATCH_PAIRS
        or projection.get("same_forward_seg_margin_pose") is not True
        or projection.get("encoder_only") is not True
        or projection.get("candidate_payload_allowed") is not False
    ):
        raise G112CheckpointPartitionError(
            "checkpoint target projection is not real n600 batch-16 encoder-only custody"
        )
    target_namespace = {
        key
        for key in arrays
        if key.startswith("__cfg_g109_")
        or key.startswith("__cfg_g46_")
        or key in {"__cfg_target_authority_sha256", "__cfg_verdict_batch"}
    }
    if target_namespace != set(expected):
        missing = sorted(set(expected) - target_namespace)
        extra = sorted(target_namespace - set(expected))
        raise G112CheckpointPartitionError(
            f"checkpoint target config set is not exact; missing={missing}, extra={extra}"
        )
    for key, expected_value in expected.items():
        observed = np.asarray(arrays[key])
        wanted = np.asarray(expected_value)
        if observed.dtype != wanted.dtype or not np.array_equal(observed, wanted):
            raise G112CheckpointPartitionError(f"checkpoint physical G109 target config differs at {key}")
    return projection


def _validate_semantic_and_compile(
    params: Mapping[str, np.ndarray],
    scalars: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], bytes]:
    semantic = {key: np.asarray(value) for key, value in params.items() if key not in POSE_TENSOR_KEYS}
    for key, value in semantic.items():
        if value.dtype != np.dtype(np.float32) or not value.flags.c_contiguous or not np.isfinite(value).all():
            raise G112CheckpointPartitionError(f"semantic learned tensor {key} must be finite C-contiguous float32")
    try:
        config = g105_adapter._checkpoint_config(
            dict(semantic),
            dict(scalars),
        )
        program = g105_adapter.compile_from_state(
            config=config,
            params={key: value for key, value in semantic.items() if key != "code"},
            interleaved_code=semantic["code"],
        )
        packet = g105_adapter.encode_packet(program)
        if g105_adapter.encode_packet(g105_adapter.parse_packet(packet)) != packet:
            raise AssertionError("G105 parse-back identity differs")
    except (
        AssertionError,
        KeyError,
        TypeError,
        ValueError,
        g105_adapter.ExactV9SemanticRootError,
    ) as exc:
        raise G112CheckpointPartitionError("checkpoint semantic tensor set/shapes/config are not exact G105") from exc
    return semantic, packet


def _compile_odd_semantic(
    *,
    shared_params: Mapping[str, np.ndarray],
    code_y1: np.ndarray,
    scalars: Mapping[str, object],
) -> bytes:
    """Compile the odd-only G112 child through G105's no-even-row surface."""

    y1 = np.asarray(code_y1)
    if (
        y1.dtype != np.dtype(np.float32)
        or y1.ndim != 2
        or y1.shape[0] != PRODUCTION_PAIR_COUNT
        or not y1.flags.c_contiguous
        or not np.isfinite(y1).all()
    ):
        raise G112CheckpointPartitionError("semantic child code_y1 must be finite C-contiguous float32[600,mod]")
    shared = {key: np.asarray(value) for key, value in shared_params.items()}
    if "code" in shared or SEMANTIC_ODD_CODE_KEY in shared:
        raise G112CheckpointPartitionError("semantic shared tensor map contains a code tensor")
    # _checkpoint_config derives the exact V9 dimensions and validates the
    # learned key set.  Its local ``code`` alias is this same [600,D] Y1 array;
    # no even rows or interleaved [1200,D] allocation are created.
    config_params = {**shared, "code": y1}
    try:
        config = g105_adapter._checkpoint_config(
            config_params,
            dict(scalars),
        )
        program = g105_adapter.compile_from_y1_state(
            config=config,
            params=shared,
            y1_code=y1,
        )
        packet = g105_adapter.encode_packet(program)
        if g105_adapter.encode_packet(g105_adapter.parse_packet(packet)) != packet:
            raise AssertionError("G105 odd-only parse-back identity differs")
    except (
        AssertionError,
        KeyError,
        TypeError,
        ValueError,
        g105_adapter.ExactV9SemanticRootError,
    ) as exc:
        raise G112CheckpointPartitionError("semantic child does not compile through exact G105 odd-only ABI") from exc
    return packet


def _parse_semantic_child_arrays(
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object], bytes]:
    observed_meta = {key for key in arrays if key.startswith("__g112_")}
    if observed_meta != SEMANTIC_CHILD_META_KEYS:
        missing = sorted(SEMANTIC_CHILD_META_KEYS - observed_meta)
        extra = sorted(observed_meta - SEMANTIC_CHILD_META_KEYS)
        raise G112CheckpointPartitionError(
            f"G112 semantic child metadata set differs; missing={missing}, extra={extra}"
        )
    exact = {
        "__g112_semantic_child_schema": SEMANTIC_CHILD_SCHEMA,
        "__g112_code_projection": "source.code[2*p+1]",
        "__g112_g105_adapter": "compile_from_y1_state.v1",
        "__g112_pair_count": PRODUCTION_PAIR_COUNT,
        "__g112_candidate_owned_code_rows": PRODUCTION_PAIR_COUNT,
    }
    for key, expected in exact.items():
        if _scalar(arrays, key) != expected:
            raise G112CheckpointPartitionError(f"G112 semantic child metadata differs at {key}")
    params, scalars = _checkpoint_views(arrays)
    if "code" in params or SEMANTIC_ODD_CODE_KEY not in params:
        raise G112CheckpointPartitionError("G112 semantic child must contain code_y1 and no interleaved code")
    y1 = np.asarray(params.pop(SEMANTIC_ODD_CODE_KEY))
    if _scalar(arrays, "__g112_modulation_dim") != (int(y1.shape[1]) if y1.ndim == 2 else -1):
        raise G112CheckpointPartitionError("G112 semantic child modulation dimension differs from code_y1")
    for key, value in params.items():
        if (
            key.startswith("pose_carrier.")
            or value.dtype != np.dtype(np.float32)
            or not value.flags.c_contiguous
            or not np.isfinite(value).all()
        ):
            raise G112CheckpointPartitionError(f"G112 semantic child shared tensor {key} is not exact finite float32")
    if any(key.startswith("__cfg_pose_carrier") for key in arrays):
        raise G112CheckpointPartitionError("G112 semantic child contains forbidden pose configuration")
    packet = _compile_odd_semantic(
        shared_params=params,
        code_y1=y1,
        scalars=scalars,
    )
    packet_sha = _sha256(packet)
    if _scalar(arrays, "__g112_semantic_packet_sha256") != packet_sha:
        raise G112CheckpointPartitionError("G112 semantic child packet SHA-256 marker differs")
    return params, y1, scalars, packet


def _npy_bytes(value: np.ndarray) -> bytes:
    handle = io.BytesIO()
    array = np.asarray(value)
    if array.ndim:
        array = np.ascontiguousarray(array)
    np.lib.format.write_array(
        handle,
        array,
        version=(1, 0),
        allow_pickle=False,
    )
    return handle.getvalue()


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    if not arrays:
        raise G112CheckpointPartitionError("deterministic NPZ cannot be empty")
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for key in sorted(arrays):
            if type(key) is not str or not key or not key.isascii() or "/" in key or "\\" in key:
                raise G112CheckpointPartitionError("deterministic NPZ key is not bounded flat ASCII")
            value = np.asarray(arrays[key])
            if value.dtype.hasobject:
                raise G112CheckpointPartitionError(f"deterministic NPZ array {key} has object dtype")
            info = zipfile.ZipInfo(f"{key}.npy", _ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                _npy_bytes(value),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    result = output.getvalue()
    _strict_npz_members(result)
    return result


def _reopen_npz_bytes(
    payload: bytes,
    *,
    expected: Mapping[str, np.ndarray],
) -> None:
    _strict_npz_members(payload)
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise G112CheckpointPartitionError("generated NPZ parse-back key set differs")
            for key, wanted in expected.items():
                observed = np.asarray(archive[key])
                target = np.asarray(wanted)
                if observed.dtype != target.dtype or not np.array_equal(observed, target):
                    raise G112CheckpointPartitionError(f"generated NPZ parse-back differs at {key}")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise G112CheckpointPartitionError("generated NPZ failed strict parse-back") from exc


def open_g112_semantic_child(
    checkpoint: Path,
    *,
    expected_sha256: str,
) -> G112SemanticChildV1:
    """Physically reopen an odd-only child and compile its exact G105 packet."""

    arrays, identity = _open_physical_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
    )
    _validate_target_custody(arrays)
    shared, code_y1, scalars, packet = _parse_semantic_child_arrays(arrays)
    return G112SemanticChildV1(
        checkpoint_path=Path(str(identity["path"])),
        checkpoint_sha256=str(identity["sha256"]),
        checkpoint_bytes=int(identity["bytes"]),
        semantic_packet=packet,
        semantic_packet_sha256=_sha256(packet),
        shared_params={key: np.asarray(value).copy(order="C") for key, value in shared.items()},
        code_y1=np.asarray(code_y1).copy(order="C"),
        g105_scalars=dict(scalars),
    )


def open_g112_pose_initializer(
    checkpoint: Path,
    *,
    expected_sha256: str,
) -> G112PoseInitializerV1:
    """Physically reopen the folded initializer and its physical G109 custody."""

    arrays, identity = _open_physical_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
    )
    if set(arrays) != INITIALIZER_ARRAY_KEYS:
        missing = sorted(INITIALIZER_ARRAY_KEYS - set(arrays))
        extra = sorted(set(arrays) - INITIALIZER_ARRAY_KEYS)
        raise G112CheckpointPartitionError(f"G112 initializer key set differs; missing={missing}, extra={extra}")
    exact = {
        "__schema": INITIALIZER_SCHEMA,
        "__status": "REQUIRES_REAL_POST_G105_REFIT",
        "__pair_count": PRODUCTION_PAIR_COUNT,
        "__pose_dim": 6,
        "__pose_carrier_source": "generated_y1",
        "__pose_carrier_residual_mode": "table",
        "__pose_carrier_xi_formula": "xi_stored+residual_scale*dxi",
        "__pose_carrier_y1_selected_preimage_schema": Y1_SELECTED_PREIMAGE_SCHEMA,
        "__requires_post_g105_refit": 1,
        "__candidate_payload_eligible": 0,
        "__score_claim": 0,
        "__pointer_mutation_allowed": 0,
    }
    for key, expected in exact.items():
        if _scalar(arrays, key) != expected:
            raise G112CheckpointPartitionError(f"G112 initializer metadata differs at {key}")
    semantic_packet_sha = _require_sha256(
        _scalar(arrays, "__semantic_packet_sha256"),
        name="initializer semantic packet",
    )
    projection_json = _scalar(arrays, "__g109_target_projection_json")
    projection_sha = _require_sha256(
        _scalar(arrays, "__g109_target_projection_sha256"),
        name="initializer G109 projection",
    )
    if not isinstance(projection_json, str):
        raise G112CheckpointPartitionError("initializer G109 projection must be scalar text")
    try:
        projection = reopen_v9_training_target_projection(
            projection_json=projection_json,
            expected_projection_sha256=projection_sha,
        )
    except (OSError, TypeError, ValueError, V9TrainingTargetBindingError) as exc:
        raise G112CheckpointPartitionError("initializer physical G109 projection failed recursive reopen") from exc
    if (
        projection.get("pair_count") != PRODUCTION_PAIR_COUNT
        or projection.get("scorer_pair_batch_size") != PRODUCTION_BATCH_PAIRS
        or projection.get("same_forward_seg_margin_pose") is not True
        or projection.get("encoder_only") is not True
        or projection.get("candidate_payload_allowed") is not False
    ):
        raise G112CheckpointPartitionError("initializer G109 custody is not exact n600 batch-16 encoder-only")
    xi_init = np.asarray(arrays["xi_init"])
    if (
        xi_init.dtype != np.dtype(np.float64)
        or xi_init.shape != (PRODUCTION_PAIR_COUNT, 6)
        or not xi_init.flags.c_contiguous
        or not np.isfinite(xi_init).all()
    ):
        raise G112CheckpointPartitionError("initializer xi_init must be finite C-contiguous float64[600,6]")
    native_hw = np.asarray(arrays["__pose_carrier_native_hw"])
    if (
        native_hw.dtype != np.dtype(np.int64)
        or native_hw.shape != (2,)
        or tuple(int(item) for item in native_hw) != (874, 1164)
    ):
        raise G112CheckpointPartitionError("initializer native HW must be exact int64 [874,1164]")
    values: dict[str, float] = {}
    for key in (
        "__pose_carrier_residual_scale",
        "__pose_carrier_s_t",
        "__pose_carrier_s_r",
        "__pose_carrier_pitch",
    ):
        try:
            value = float(_scalar(arrays, key))
        except (TypeError, ValueError) as exc:
            raise G112CheckpointPartitionError(f"initializer scalar {key} must be numeric") from exc
        if not math.isfinite(value):
            raise G112CheckpointPartitionError(f"initializer scalar {key} must be finite")
        values[key] = value
    if (
        values["__pose_carrier_residual_scale"] <= 0.0
        or values["__pose_carrier_s_t"] <= 0.0
        or values["__pose_carrier_s_r"] < 0.0
    ):
        raise G112CheckpointPartitionError("initializer calibration values are outside the exact contract")
    xi_copy = xi_init.copy(order="C")
    xi_copy.setflags(write=False)
    return G112PoseInitializerV1(
        checkpoint_path=Path(str(identity["path"])),
        checkpoint_sha256=str(identity["sha256"]),
        checkpoint_bytes=int(identity["bytes"]),
        xi_init=xi_copy,
        semantic_packet_sha256=semantic_packet_sha,
        target_projection_sha256=projection_sha,
        residual_scale=values["__pose_carrier_residual_scale"],
        s_t=values["__pose_carrier_s_t"],
        s_r=values["__pose_carrier_s_r"],
        pitch=values["__pose_carrier_pitch"],
        native_hw=(874, 1164),
        selected_preimage_schema=Y1_SELECTED_PREIMAGE_SCHEMA,
    )


def _require_output_root(
    output_root: Path,
    *,
    allowed_roots: Sequence[Path],
    required_free_bytes: int,
) -> tuple[Path, dict[str, object]]:
    output = Path(output_root).expanduser()
    if not output.is_absolute():
        raise G112CheckpointPartitionError("G112 output root must be absolute")
    output = output.resolve()
    allowed = tuple(Path(root).expanduser().resolve() for root in allowed_roots)
    if allowed and not any(output == root or root in output.parents for root in allowed):
        raise G112CheckpointPartitionError("G112 output root is outside the allowed SSD waterfall")
    parent = output.parent
    if not parent.exists() or parent.is_symlink() or not parent.is_dir():
        raise G112CheckpointPartitionError("G112 output parent must be an existing real directory")
    if output.exists() or output.is_symlink():
        raise G112CheckpointPartitionError("G112 output root is immutable and already exists")
    free = int(shutil.disk_usage(parent).free)
    if free < required_free_bytes:
        raise G112CheckpointPartitionError(f"G112 storage preflight needs {required_free_bytes} bytes, observed {free}")
    return output, {
        "status": "PASS",
        "output_parent": str(parent),
        "allowed_roots": [str(root) for root in allowed],
        "required_free_bytes": int(required_free_bytes),
        "observed_free_bytes": free,
        "explicit_nondefault_root_authorized": bool(allowed and not any(root in SSD_ROOTS for root in allowed)),
    }


def _write_immutable(path: Path, payload: bytes) -> None:
    scratch = path.parent / ".scratch"
    scratch.mkdir(mode=0o700, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".partial",
        dir=scratch,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise G112CheckpointPartitionError(f"immutable G112 output already exists: {path.name}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _seal_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in body:
        raise G112CheckpointPartitionError("unsealed receipt body contains receipt_sha256")
    sealed = dict(body)
    sealed["receipt_sha256"] = _sha256(_canonical_json(body))
    return sealed


def _open_physical_receipt_json(
    receipt_path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    path = Path(receipt_path).expanduser()
    if not path.is_absolute():
        raise G112CheckpointPartitionError(
            "G112 receipt path must be absolute physical custody"
        )
    expected = _require_sha256(expected_sha256, name="G112 receipt")
    try:
        before = path.lstat()
    except OSError as exc:
        raise G112CheckpointPartitionError(
            "G112 receipt is not readable"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise G112CheckpointPartitionError(
            "G112 receipt must be a regular non-symlink file"
        )
    payload = path.read_bytes()
    if _sha256(payload) != expected:
        raise G112CheckpointPartitionError(
            "G112 receipt physical SHA-256 differs"
        )
    try:
        parsed = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise G112CheckpointPartitionError(
            "G112 receipt is not ASCII JSON"
        ) from exc
    if not isinstance(parsed, dict) or _canonical_json(parsed) + b"\n" != payload:
        raise G112CheckpointPartitionError(
            "G112 receipt is not canonical newline-terminated JSON"
        )
    after = path.lstat()
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or _sha256(path.read_bytes()) != expected
    ):
        raise G112CheckpointPartitionError(
            "G112 receipt changed during recursive reopen"
        )
    return parsed, {
        "path": str(path),
        "bytes": len(payload),
        "sha256": expected,
    }


def open_g112_partition_receipt(
    receipt_path: Path,
    *,
    expected_sha256: str,
) -> G112PartitionReceiptV2:
    """Reopen the complete G112 source pair and both published children."""

    receipt, identity = _open_physical_receipt_json(
        receipt_path,
        expected_sha256=expected_sha256,
    )
    sealed_sha = receipt.pop("receipt_sha256", None)
    if sealed_sha != _sha256(_canonical_json(receipt)):
        raise G112CheckpointPartitionError(
            "G112 receipt self-hash differs"
        )
    exact = {
        "schema": SCHEMA,
        "status": "PARTITIONED_RESEARCH_ONLY_REQUIRES_POST_G105_REFIT",
        "research_only": True,
        "candidate": False,
        "score_claim": False,
        "pointer_mutation_allowed": False,
        "archive_claim": False,
        "scope": "compile_infrastructure_only",
    }
    for key, expected in exact.items():
        if receipt.get(key) != expected:
            raise G112CheckpointPartitionError(
                f"G112 receipt differs at {key}"
            )
    try:
        deploy_info = receipt["source_checkpoint"]
        resume_info = receipt["source_resume_checkpoint"]
        lineage_receipt_info = receipt["source_lineage_receipt"]
        lineage = receipt["fresh_producer_lineage"]
        semantic_info = receipt["semantic_child"]
        initializer_info = receipt["conditional_initializer"]
    except KeyError as exc:
        raise G112CheckpointPartitionError(
            "G112 receipt is missing a required custody section"
        ) from exc
    if not all(
        isinstance(value, dict)
        for value in (
            deploy_info,
            resume_info,
            lineage_receipt_info,
            lineage,
            semantic_info,
            initializer_info,
        )
    ):
        raise G112CheckpointPartitionError(
            "G112 receipt custody section has the wrong type"
        )
    if (
        semantic_info.get("filename") != SEMANTIC_CHILD_NAME
        or initializer_info.get("filename") != INITIALIZER_NAME
    ):
        raise G112CheckpointPartitionError(
            "G112 receipt child filename contract differs"
        )
    try:
        source_chain = open_fresh_physical_checkpoint_chain_v1(
            Path(str(lineage_receipt_info["path"])),
            expected_receipt_sha256=str(
                lineage_receipt_info["sha256"]
            ),
            expected_current_launch_dsl_compile_hash=str(
                lineage["current_launch_dsl_compile_hash"]
            ),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        FreshProducerLineageV1Error,
    ) as exc:
        raise G112CheckpointPartitionError(
            "G112 receipt source physical ancestry failed recursive reopen"
        ) from exc
    source_pair = source_chain.current.pair
    if (
        source_chain.complete_trajectory_proven is not True
        or str(source_pair.deploy.path) != deploy_info.get("path")
        or source_pair.deploy.sha256 != deploy_info.get("sha256")
        or source_pair.deploy.bytes != deploy_info.get("bytes")
        or str(source_pair.resume.path) != resume_info.get("path")
        or source_pair.resume.sha256 != resume_info.get("sha256")
        or source_pair.resume.bytes != resume_info.get("bytes")
        or str(source_chain.current.receipt_path)
        != lineage_receipt_info.get("path")
        or source_chain.current.receipt_sha256
        != lineage_receipt_info.get("sha256")
        or source_chain.current.receipt_bytes
        != lineage_receipt_info.get("bytes")
    ):
        raise G112CheckpointPartitionError(
            "G112 receipt source pair is not the recursively proven current node"
        )
    expected_lineage = {
        "schema": FRESH_PRODUCER_LINEAGE_SCHEMA,
        "physical_node_schema": FRESH_PHYSICAL_CHECKPOINT_NODE_SCHEMA,
        "seed": source_pair.seed,
        "root_sha256": source_pair.root_sha256,
        "initial_state_sha256": source_pair.initial_state_sha256,
        "root_dsl_compile_hash": source_pair.root_dsl_compile_hash,
        "current_launch_dsl_compile_hash": (
            source_pair.current_launch_dsl_compile_hash
        ),
        "target_projection_sha256": source_pair.target_projection_sha256,
        "parent_checkpoint_id_sha256": (
            source_pair.parent_checkpoint_id_sha256
        ),
        "state_sha256": source_pair.state_sha256,
        "checkpoint_id_sha256": source_pair.checkpoint_id_sha256,
        "epoch": source_pair.epoch,
        "stage": source_pair.stage,
        "current_sequence_index": source_chain.current.sequence_index,
        "physical_chain_node_count": len(source_chain.nodes),
        "live_tensor_count": source_pair.live_tensor_count,
        "ema_tensor_count": source_pair.ema_tensor_count,
        "optimizer_tensor_count": source_pair.optimizer_tensor_count,
        "polyak_tensor_count": source_pair.polyak_tensor_count,
        "config_array_count": source_pair.config_array_count,
        "rng_complete": True,
        "event_ledger_complete": True,
        "deploy_equals_companion_ema": True,
        "film_stiefel": False,
        "cold_root_recomputed": True,
        "full_semantic_state_recomputed": True,
        "checkpoint_id_recomputed": True,
        "current_launch_hash_external_custody_matched": True,
        "zero_parent_root_recursively_reopened": True,
        "every_parent_receipt_sha256_reopened": True,
        "complete_trajectory_proven": True,
    }
    if lineage != expected_lineage:
        raise G112CheckpointPartitionError(
            "G112 receipt fresh-lineage identity differs from physical source pair"
        )
    root = Path(str(identity["path"])).parent
    semantic = open_g112_semantic_child(
        root / SEMANTIC_CHILD_NAME,
        expected_sha256=str(semantic_info.get("sha256", "")),
    )
    initializer = open_g112_pose_initializer(
        root / INITIALIZER_NAME,
        expected_sha256=str(initializer_info.get("sha256", "")),
    )
    semantic_packet_sha = str(
        semantic_info.get("semantic_packet_sha256", "")
    )
    if (
        semantic.semantic_packet_sha256 != semantic_packet_sha
        or initializer.semantic_packet_sha256 != semantic_packet_sha
        or int(semantic_info.get("bytes", -1)) != semantic.checkpoint_bytes
        or int(initializer_info.get("bytes", -1))
        != initializer.checkpoint_bytes
    ):
        raise G112CheckpointPartitionError(
            "G112 receipt child identity or semantic-packet binding differs"
        )
    return G112PartitionReceiptV2(
        receipt_path=Path(str(identity["path"])),
        receipt_sha256=str(identity["sha256"]),
        receipt_bytes=int(identity["bytes"]),
        semantic_child=semantic,
        initializer=initializer,
        source_chain=source_chain,
        semantic_packet_sha256=semantic_packet_sha,
    )


def _lineage_physical_identity(
    physical: Any,
) -> dict[str, object]:
    return {
        "path": str(physical.path),
        "bytes": int(physical.bytes),
        "sha256": str(physical.sha256),
        "device": int(physical.device),
        "inode": int(physical.inode),
        "mtime_ns": int(physical.mtime_ns),
        "regular_file": True,
        "symlink": False,
        "reopened_unchanged": True,
    }


def materialize_g112_checkpoint_partition(
    *,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    resume_checkpoint: Path,
    expected_resume_checkpoint_sha256: str,
    lineage_receipt: Path,
    expected_lineage_receipt_sha256: str,
    expected_current_launch_dsl_compile_hash: str,
    output_root: Path,
    allowed_output_roots: Sequence[Path] = SSD_ROOTS,
) -> G112CheckpointPartitionResultV1:
    """Partition one physical G111 deploy/full-state pair into immutable artifacts."""

    try:
        source_chain = open_fresh_physical_checkpoint_chain_v1(
            lineage_receipt,
            expected_receipt_sha256=expected_lineage_receipt_sha256,
            expected_current_launch_dsl_compile_hash=(
                expected_current_launch_dsl_compile_hash
            ),
        )
    except FreshProducerLineageV1Error as exc:
        raise G112CheckpointPartitionError(
            "G111 physical fresh-lineage ancestry custody failed: "
            f"{exc}"
        ) from exc
    source_pair = source_chain.current.pair
    requested_deploy = Path(checkpoint).expanduser()
    requested_resume = Path(resume_checkpoint).expanduser()
    if (
        source_chain.complete_trajectory_proven is not True
        or source_pair.deploy.path != requested_deploy
        or source_pair.resume.path != requested_resume
        or source_pair.deploy.sha256
        != _require_sha256(
            expected_checkpoint_sha256,
            name="source checkpoint",
        )
        or source_pair.resume.sha256
        != _require_sha256(
            expected_resume_checkpoint_sha256,
            name="source resume checkpoint",
        )
    ):
        raise G112CheckpointPartitionError(
            "requested G111 deploy/resume pair is not the recursively proven current node"
        )
    arrays = source_pair.deploy.arrays
    checkpoint_identity = _lineage_physical_identity(source_pair.deploy)
    resume_checkpoint_identity = _lineage_physical_identity(source_pair.resume)
    params, scalars = _checkpoint_views(arrays)
    pose_values, xi_init = _validate_pose_contract(params, arrays)
    projection = _validate_target_custody(arrays)
    semantic, source_packet = _validate_semantic_and_compile(params, scalars)

    semantic_child_arrays = {
        key: np.asarray(value).copy(order="C")
        for key, value in arrays.items()
        if key not in POSE_TENSOR_KEYS and key not in POSE_CONFIG_KEYS and key != "code"
    }
    source_code = np.asarray(semantic["code"])
    code_y1 = np.ascontiguousarray(source_code[1::2], dtype=np.float32)
    shared_semantic = {key: value for key, value in semantic.items() if key != "code"}
    odd_packet = _compile_odd_semantic(
        shared_params=shared_semantic,
        code_y1=code_y1,
        scalars=scalars,
    )
    if odd_packet != source_packet:
        raise G112CheckpointPartitionError("G105 packet changed through the exact odd-only compiler")

    semantic_packet_sha = _sha256(source_packet)
    semantic_child_arrays.update(
        {
            SEMANTIC_ODD_CODE_KEY: code_y1,
            "__g112_semantic_child_schema": np.asarray(SEMANTIC_CHILD_SCHEMA),
            "__g112_code_projection": np.asarray("source.code[2*p+1]"),
            "__g112_g105_adapter": np.asarray("compile_from_y1_state.v1"),
            "__g112_pair_count": np.asarray(
                PRODUCTION_PAIR_COUNT,
                dtype=np.int64,
            ),
            "__g112_modulation_dim": np.asarray(
                int(code_y1.shape[1]),
                dtype=np.int64,
            ),
            "__g112_semantic_packet_sha256": np.asarray(semantic_packet_sha),
            "__g112_candidate_owned_code_rows": np.asarray(
                PRODUCTION_PAIR_COUNT,
                dtype=np.int64,
            ),
        }
    )
    parsed_shared, parsed_y1, _parsed_scalars, child_packet = _parse_semantic_child_arrays(semantic_child_arrays)
    if (
        child_packet != source_packet
        or set(parsed_shared) != set(shared_semantic)
        or not np.array_equal(parsed_y1, code_y1)
    ):
        raise AssertionError("internal odd-only semantic child differs")

    initializer_arrays = {
        "xi_init": xi_init,
        "__schema": np.asarray(INITIALIZER_SCHEMA),
        "__status": np.asarray("REQUIRES_REAL_POST_G105_REFIT"),
        "__pair_count": np.asarray(PRODUCTION_PAIR_COUNT, dtype=np.int64),
        "__pose_dim": np.asarray(6, dtype=np.int64),
        "__pose_carrier_source": np.asarray("generated_y1"),
        "__pose_carrier_residual_mode": np.asarray("table"),
        "__pose_carrier_residual_scale": np.asarray(
            pose_values["__cfg_pose_carrier_residual_scale"],
            dtype=np.float64,
        ),
        "__pose_carrier_s_t": np.asarray(
            pose_values["__cfg_pose_carrier_s_t"],
            dtype=np.float64,
        ),
        "__pose_carrier_s_r": np.asarray(
            pose_values["__cfg_pose_carrier_s_r"],
            dtype=np.float64,
        ),
        "__pose_carrier_pitch": np.asarray(
            pose_values["__cfg_pose_carrier_pitch"],
            dtype=np.float64,
        ),
        "__pose_carrier_native_hw": np.asarray([874, 1164], dtype=np.int64),
        "__pose_carrier_xi_formula": np.asarray("xi_stored+residual_scale*dxi"),
        "__pose_carrier_y1_selected_preimage_schema": np.asarray(Y1_SELECTED_PREIMAGE_SCHEMA),
        "__semantic_packet_sha256": np.asarray(semantic_packet_sha),
        "__g109_target_projection_json": np.asarray(_scalar(arrays, CHECKPOINT_PROJECTION_KEY)),
        "__g109_target_projection_sha256": np.asarray(_scalar(arrays, CHECKPOINT_PROJECTION_SHA_KEY)),
        "__requires_post_g105_refit": np.asarray(1, dtype=np.int8),
        "__candidate_payload_eligible": np.asarray(0, dtype=np.int8),
        "__score_claim": np.asarray(0, dtype=np.int8),
        "__pointer_mutation_allowed": np.asarray(0, dtype=np.int8),
    }
    if set(initializer_arrays) != INITIALIZER_ARRAY_KEYS:
        raise AssertionError("internal initializer array key set differs")

    semantic_payload = _deterministic_npz_bytes(semantic_child_arrays)
    initializer_payload = _deterministic_npz_bytes(initializer_arrays)
    _reopen_npz_bytes(
        semantic_payload,
        expected=semantic_child_arrays,
    )
    _reopen_npz_bytes(
        initializer_payload,
        expected=initializer_arrays,
    )
    semantic_sha = _sha256(semantic_payload)
    initializer_sha = _sha256(initializer_payload)
    required_free = len(semantic_payload) + len(initializer_payload) + len(semantic_payload) + (16 << 20)
    output, storage = _require_output_root(
        output_root,
        allowed_roots=allowed_output_roots,
        required_free_bytes=required_free,
    )

    semantic_learned_keys = sorted(semantic)
    semantic_shared_keys = sorted(key for key in semantic if key != "code")
    all_learned_keys = set(params)
    if set(semantic).intersection(POSE_TENSOR_KEYS) or set(semantic).union(POSE_TENSOR_KEYS) != all_learned_keys:
        raise AssertionError("internal learned tensor partition is not total/disjoint")
    even_source = np.ascontiguousarray(source_code[0::2], dtype=np.float32)
    odd_source = np.ascontiguousarray(source_code[1::2], dtype=np.float32)
    partition = {
        "all_source_learned_tensor_keys": sorted(all_learned_keys),
        "semantic_source_tensor_keys": semantic_learned_keys,
        "semantic_shared_tensor_keys": semantic_shared_keys,
        "conditional_initializer_source_tensor_keys": sorted(POSE_TENSOR_KEYS),
        "source_atom_ownership": {
            "semantic_shared_tensors": {
                "count": len(semantic_shared_keys),
                "owner": "G105 semantic child",
            },
            "odd_code_rows": {
                "rows": PRODUCTION_PAIR_COUNT,
                "owner": "G105 semantic child",
                "source_projection": "code[2*p+1]",
                "sha256": _sha256(odd_source.tobytes(order="C")),
            },
            "even_code_rows": {
                "rows": PRODUCTION_PAIR_COUNT,
                "owner": "encoder-only discarded source state",
                "candidate_owned": False,
                "source_projection": "code[2*p]",
                "sha256": _sha256(even_source.tobytes(order="C")),
                "semantic_child_storage": "absent",
                "source_values_copied_to_child": False,
                "odd_only_compile_packet_identical": True,
            },
            "pose_tables": {
                "owner": "conditional initializer",
                "fold": "xi_stored+residual_scale*dxi",
                "requires_post_g105_refit": True,
            },
        },
        "source_tensor_union_complete": (set(semantic).union(POSE_TENSOR_KEYS) == all_learned_keys),
        "source_tensor_owners_disjoint": (not set(semantic).intersection(POSE_TENSOR_KEYS)),
        "source_atoms_orphaned": 0,
        "semantic_packet_invariant_to_even_source_rows": True,
        "semantic_child_schema": SEMANTIC_CHILD_SCHEMA,
        "semantic_child_odd_code_key": SEMANTIC_ODD_CODE_KEY,
        "semantic_child_g105_compiler": "compile_from_y1_state",
    }
    receipt_body = {
        "schema": SCHEMA,
        "status": "PARTITIONED_RESEARCH_ONLY_REQUIRES_POST_G105_REFIT",
        "research_only": True,
        "candidate": False,
        "score_claim": False,
        "pointer_mutation_allowed": False,
        "archive_claim": False,
        "scope": "compile_infrastructure_only",
        "remaining_blockers": [
            "G114 self-orient versus shipped-quantizer contract remains open; this receipt does not make G111 launch-ready",
            "train/public V10 camera-realization parity is not yet closed",
            "conditional archive-domain raw-versus-Rice arbitration is not yet measured",
            "xi initializer requires a real post-G105 refit",
        ],
        "source_checkpoint": checkpoint_identity,
        "source_resume_checkpoint": resume_checkpoint_identity,
        "source_lineage_receipt": {
            "path": str(source_chain.current.receipt_path),
            "bytes": source_chain.current.receipt_bytes,
            "sha256": source_chain.current.receipt_sha256,
        },
        "fresh_producer_lineage": {
            "schema": FRESH_PRODUCER_LINEAGE_SCHEMA,
            "physical_node_schema": FRESH_PHYSICAL_CHECKPOINT_NODE_SCHEMA,
            "seed": source_pair.seed,
            "root_sha256": source_pair.root_sha256,
            "initial_state_sha256": source_pair.initial_state_sha256,
            "root_dsl_compile_hash": source_pair.root_dsl_compile_hash,
            "current_launch_dsl_compile_hash": (
                source_pair.current_launch_dsl_compile_hash
            ),
            "target_projection_sha256": source_pair.target_projection_sha256,
            "parent_checkpoint_id_sha256": (
                source_pair.parent_checkpoint_id_sha256
            ),
            "state_sha256": source_pair.state_sha256,
            "checkpoint_id_sha256": source_pair.checkpoint_id_sha256,
            "epoch": source_pair.epoch,
            "stage": source_pair.stage,
            "current_sequence_index": (
                source_chain.current.sequence_index
            ),
            "physical_chain_node_count": len(source_chain.nodes),
            "live_tensor_count": source_pair.live_tensor_count,
            "ema_tensor_count": source_pair.ema_tensor_count,
            "optimizer_tensor_count": source_pair.optimizer_tensor_count,
            "polyak_tensor_count": source_pair.polyak_tensor_count,
            "config_array_count": source_pair.config_array_count,
            "rng_complete": source_pair.rng_complete,
            "event_ledger_complete": source_pair.event_ledger_complete,
            "deploy_equals_companion_ema": source_pair.deploy_equals_ema,
            "film_stiefel": source_pair.film_stiefel,
            "cold_root_recomputed": True,
            "full_semantic_state_recomputed": True,
            "checkpoint_id_recomputed": True,
            "current_launch_hash_external_custody_matched": True,
            "zero_parent_root_recursively_reopened": True,
            "every_parent_receipt_sha256_reopened": True,
            "complete_trajectory_proven": True,
        },
        "physical_g109_target_custody": {
            "projection_sha256": _scalar(
                arrays,
                CHECKPOINT_PROJECTION_SHA_KEY,
            ),
            "aggregate_receipt": projection["aggregate_receipt"],
            "aggregate_receipt_sha256": projection["aggregate_receipt_sha256"],
            "batch_digest_chain_sha256": projection["batch_digest_chain_sha256"],
            "pair_count": projection["pair_count"],
            "scorer_pair_batch_size": projection["scorer_pair_batch_size"],
            "same_forward_seg_margin_pose": True,
            "encoder_only": True,
            "candidate_payload_allowed": False,
        },
        "learned_tensor_identities": {key: _array_identity(params[key]) for key in sorted(params)},
        "partition": partition,
        "semantic_child": {
            "filename": SEMANTIC_CHILD_NAME,
            "bytes": len(semantic_payload),
            "sha256": semantic_sha,
            "deterministic_npz": True,
            "strict_parse_back": True,
            "schema": SEMANTIC_CHILD_SCHEMA,
            "odd_code_key": SEMANTIC_ODD_CODE_KEY,
            "odd_code_shape": [
                PRODUCTION_PAIR_COUNT,
                int(code_y1.shape[1]),
            ],
            "interleaved_code_key_present": False,
            "g105_compiler_api": "compile_from_y1_state",
            "semantic_packet_sha256": semantic_packet_sha,
            "pose_tensors_present": False,
            "pose_config_present": False,
        },
        "conditional_initializer": {
            "filename": INITIALIZER_NAME,
            "bytes": len(initializer_payload),
            "sha256": initializer_sha,
            "deterministic_npz": True,
            "strict_parse_back": True,
            "xi_init": _array_identity(xi_init),
            "source": "generated_y1",
            "selected_preimage_schema": Y1_SELECTED_PREIMAGE_SCHEMA,
            "final_payload": False,
            "candidate_payload_eligible": False,
            "requires_real_post_g105_refit": True,
            "reason": (
                "G105 tensor/code quantization changes the final generated-Y1 "
                "source consumed by the conditional Y0 realization"
            ),
        },
        "storage_preflight": storage,
        "atomic_write_contract": {
            "fresh_output_directory": True,
            "same_filesystem_temp_then_hardlink": True,
            "receipt_written_last_as_completion_marker": True,
            "success_scratch_removed": True,
            "material_outputs_never_overwritten": True,
        },
    }
    receipt = _seal_receipt(receipt_body)
    receipt_payload = _canonical_json(receipt) + b"\n"

    output.mkdir(mode=0o700)
    semantic_path = output / SEMANTIC_CHILD_NAME
    initializer_path = output / INITIALIZER_NAME
    receipt_path = output / RECEIPT_NAME
    _write_immutable(semantic_path, semantic_payload)
    _write_immutable(initializer_path, initializer_payload)
    reopened_semantic = open_g112_semantic_child(
        semantic_path,
        expected_sha256=semantic_sha,
    )
    reopened_initializer = open_g112_pose_initializer(
        initializer_path,
        expected_sha256=initializer_sha,
    )
    if (
        reopened_semantic.semantic_packet != source_packet
        or reopened_initializer.semantic_packet_sha256 != semantic_packet_sha
        or not np.array_equal(reopened_initializer.xi_init, xi_init)
    ):
        raise G112CheckpointPartitionError("published G112 child/initializer recursive reopen differs")
    _write_immutable(receipt_path, receipt_payload)
    scratch = output / ".scratch"
    try:
        scratch.rmdir()
    except OSError as exc:
        raise G112CheckpointPartitionError("G112 scratch was not empty after successful publication") from exc
    if (
        _sha256(semantic_path.read_bytes()) != semantic_sha
        or _sha256(initializer_path.read_bytes()) != initializer_sha
        or _sha256(receipt_path.read_bytes()) != _sha256(receipt_payload)
    ):
        raise G112CheckpointPartitionError("published G112 artifact differs after durable reopen")
    reopened_receipt = open_g112_partition_receipt(
        receipt_path,
        expected_sha256=_sha256(receipt_payload),
    )
    if (
        reopened_receipt.semantic_packet_sha256 != semantic_packet_sha
        or reopened_receipt.source_chain.current.pair.checkpoint_id_sha256
        != source_pair.checkpoint_id_sha256
    ):
        raise G112CheckpointPartitionError(
            "published G112 receipt recursive closure differs"
        )

    return G112CheckpointPartitionResultV1(
        output_root=output,
        semantic_child_path=semantic_path,
        semantic_child_sha256=semantic_sha,
        initializer_path=initializer_path,
        initializer_sha256=initializer_sha,
        receipt_path=receipt_path,
        receipt_sha256=_sha256(receipt_payload),
        semantic_packet_sha256=semantic_packet_sha,
        source_checkpoint_sha256=str(checkpoint_identity["sha256"]),
        source_resume_checkpoint_sha256=str(
            resume_checkpoint_identity["sha256"]
        ),
        fresh_lineage_checkpoint_id_sha256=source_pair.checkpoint_id_sha256,
    )


__all__ = [
    "INITIALIZER_ARRAY_KEYS",
    "INITIALIZER_NAME",
    "INITIALIZER_SCHEMA",
    "POSE_CONFIG_KEYS",
    "POSE_TENSOR_KEYS",
    "RECEIPT_NAME",
    "SCHEMA",
    "SEMANTIC_CHILD_META_KEYS",
    "SEMANTIC_CHILD_NAME",
    "SEMANTIC_CHILD_SCHEMA",
    "SEMANTIC_ODD_CODE_KEY",
    "SSD_ROOTS",
    "Y1_SELECTED_PREIMAGE_SCHEMA",
    "G112CheckpointPartitionError",
    "G112CheckpointPartitionResultV1",
    "G112PartitionReceiptV2",
    "G112PoseInitializerV1",
    "G112SemanticChildV1",
    "materialize_g112_checkpoint_partition",
    "open_g112_partition_receipt",
    "open_g112_pose_initializer",
    "open_g112_semantic_child",
]
