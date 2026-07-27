# SPDX-License-Identifier: MIT
"""G110 generated-Y1 conditional pose product with post-G105 refit custody.

The disjoint G112 children supply the exact odd-only G105 semantic program and
a useful joint-descent pose initializer.  The initializer is not the final
conditional operand: G105 quantizes the semantic trunk and odd code, and the
public V10 camera realization changes the Y1 source seen by the pose carrier.
The only admitted final trajectory is therefore a real, resumable post-G105
refit performed through this exact public composition::

    parsed G105 scorer Y1 -> public V10 camera Y1
        -> native SE(3) homography warp -> stored-video uint8 camera Y0

The packet stores the exact G105 semantic program and a canonical XIP2 encoding
of the refit ``xi_eff``.  It stores no even code rows, no homography table, no
scorer, and no source frames.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

from tac.boundary_math.warp_real_luma_frame0 import (
    GroundHomographyGeom,
    warp_frame0_uint8_numpy,
)
from tac.boundary_math.xi_pose_coder import (
    XiPoseCoderError,
    dequantize_xi,
    parse_xi_payload,
    quantize_xi,
    serialize_xi_payload,
)
from tac.witness_control.taskspace_g112_exact_checkpoint_partition_v1 import (
    G112CheckpointPartitionError,
    G112PartitionReceiptV2,
    G112PoseInitializerV1,
    G112SemanticChildV1,
    open_g112_partition_receipt,
)
from tac.witness_control.taskspace_v9_training_target_capsule_v1 import (
    AGGREGATE_SCHEMA as TARGET_CAPSULE_SCHEMA,
)
from tac.witness_control.taskspace_v9_training_target_capsule_v1 import (
    PRODUCTION_BATCH_PAIRS,
    PRODUCTION_PAIR_COUNT,
    V9TrainingTargetCapsuleError,
    V9TrainingTargetCapsuleLoaderV1,
    sha256_file,
)
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    ExactV9SemanticRootError,
    ExactV9SemanticRootY1ProgramV1,
    Y1WireCodecV1,
    encode_packet_y1_variants,
)
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    _checkpoint_config as _g105_checkpoint_config,
)
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    _checkpoint_runtime_state as _g105_checkpoint_runtime_state,
)
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    compile_from_state as compile_v9_from_state,
)
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    encode_packet as encode_v9_packet,
)
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    parse_packet as parse_v9_packet,
)
from tac.witness_dsl.taskspace_g110_generic_two_layer_public_product_v1 import (
    FINAL_Y1_DOMAIN,
    G109_PROJECTION_KEY,
    G109_PROJECTION_SHA_KEY,
    G111_POSE_PARAM_KEYS,
    V9_VARIANT_ID,
    G110OuterZipMethodV1,
    G110TwoLayerError,
    _partition_g111_checkpoint_params,
    _population_digest,
    final_y1_binding_sha256,
    open_final_y1_provider,
)
from tac.witness_dsl.v10_factor2_selected_preimage_v1 import (
    SCHEMA as V10_FACTOR2_SELECTED_PREIMAGE_SCHEMA,
)

MAGIC: Final = b"G110PC01"
VERSION: Final = 1
VARIANT_ID: Final = "tac.semantic_root_y0.generated_y1_pose_xip2.v1"
PACKET_MEMBER: Final = "taskspace_two_layer_v1.bin"
PAIR_COUNT: Final = 600
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
CHANNELS: Final = 3
MAX_PACKET_BYTES: Final = 2_100_000
MAX_SEMANTIC_PACKET_BYTES: Final = 2_000_000
MAX_XIP2_BYTES: Final = 100_000
MAX_ARCHIVE_BYTES: Final = 2_100_000
POST_G105_REFIT_CHECKPOINT_SCHEMA: Final = (
    "tac.g110_post_g105_generated_y1_pose_refit_checkpoint.v1"
)
POST_G105_REFIT_RUN_SCHEMA: Final = (
    "tac.g110_post_g105_generated_y1_pose_refit_run.v1"
)
SOURCE_DOMAIN: Final = "parsed_g105_y1_v10_camera"
RENDER_ORDER: Final = (
    "parsed_g105_scorer_y1->v10_camera_y1->native_homography_warp"
    "->camera_uint8_y0"
)
XI_INITIALIZER_DOMAIN: Final = b"G110_XI_INITIALIZER_F64BE_N600_V1\x00"

_HEADER = struct.Struct(">8sBBHIId32sI")
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_F64_BE = np.dtype(">f8")


class G110GeneratedY1PoseError(ValueError):
    """The generated-Y1 pose product or its physical custody failed closed."""


_OUTER_ZIP_METHODS: Final = tuple(G110OuterZipMethodV1)


def _sha256(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


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
        raise G110GeneratedY1PoseError(
            "value is not finite canonical ASCII JSON"
        ) from exc


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise G110GeneratedY1PoseError(f"{name} must be canonical lowercase SHA-256")
    return value


def _regular_file(path: Path, *, name: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise G110GeneratedY1PoseError(f"{name} must not be a symlink")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise G110GeneratedY1PoseError(f"{name} must be a regular file")
    return resolved


def _file_binding(path: Path) -> dict[str, object]:
    resolved = _regular_file(path, name="bound artifact")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _reopen_binding(value: object, *, name: str) -> Path:
    if type(value) is not dict or set(value) != {"path", "bytes", "sha256"}:
        raise G110GeneratedY1PoseError(f"{name} binding key set differs")
    path = _regular_file(Path(str(value["path"])), name=name)
    if (
        value["path"] != str(path)
        or type(value["bytes"]) is not int
        or value["bytes"] != path.stat().st_size
        or value["sha256"] != sha256_file(path)
    ):
        raise G110GeneratedY1PoseError(f"{name} physical identity differs")
    _require_sha256(value["sha256"], name=name)
    return path


def _xi_digest(xi: np.ndarray) -> str:
    raw = np.asarray(xi)
    if raw.shape != (PAIR_COUNT, 6) or not np.all(np.isfinite(raw)):
        raise G110GeneratedY1PoseError("xi population must be finite [600,6]")
    return _sha256(
        XI_INITIALIZER_DOMAIN
        + np.asarray(raw, dtype=_F64_BE).tobytes(order="C")
    )


@dataclass(frozen=True, slots=True, init=False)
class G111GeneratedY1PoseInitializerCustodyV1:
    """Content-read ownership proof for the G111 checkpoint pose subtree."""

    checkpoint_sha256: str
    semantic_packet_sha256: str
    xi_initializer_sha256: str
    even_code_exclusion_sha256: str
    pitch: float
    residual_scale: float
    tensor_partition_sha256: str
    xi_initializer: np.ndarray = field(repr=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "G111 pose initializer custody requires from_physical_checkpoint(); "
            "hash-only construction is forbidden"
        )

    @classmethod
    def from_physical_checkpoint(
        cls,
        checkpoint: Path,
        *,
        expected_checkpoint_sha256: str,
        semantic_packet: bytes,
    ) -> G111GeneratedY1PoseInitializerCustodyV1:
        checkpoint_path = _regular_file(checkpoint, name="fresh G111 checkpoint")
        expected_sha = _require_sha256(
            expected_checkpoint_sha256,
            name="fresh G111 checkpoint",
        )
        if sha256_file(checkpoint_path) != expected_sha:
            raise G110GeneratedY1PoseError(
                "fresh G111 checkpoint physical SHA-256 differs"
            )
        try:
            params, scalars, checkpoint_sha, _checkpoint_bytes = (
                _g105_checkpoint_runtime_state(checkpoint_path)
            )
            semantic_params, pose_params = _partition_g111_checkpoint_params(
                params,
                scalars,
            )
            if set(pose_params) != G111_POSE_PARAM_KEYS:
                raise G110GeneratedY1PoseError(
                    "checkpoint has no complete generated_y1/table pose subtree"
                )
            config = _g105_checkpoint_config(semantic_params, scalars)
            program = compile_v9_from_state(
                config=config,
                params={
                    key: value
                    for key, value in semantic_params.items()
                    if key != "code"
                },
                interleaved_code=semantic_params["code"],
            )
            rederived_semantic = encode_v9_packet(program)
        except (
            OSError,
            ValueError,
            ExactV9SemanticRootError,
            G110TwoLayerError,
        ) as exc:
            raise G110GeneratedY1PoseError(
                "fresh G111 checkpoint did not partition/recompile exactly"
            ) from exc
        if rederived_semantic != semantic_packet:
            raise G110GeneratedY1PoseError(
                "semantic packet is not the exact G105 projection of G111"
            )
        code = np.asarray(semantic_params["code"], dtype=np.float32)
        if code.ndim != 2 or code.shape[0] != 2 * PAIR_COUNT:
            raise G110GeneratedY1PoseError(
                "G111 interleaved code is not exact n600 pair geometry"
            )
        mutated_code = np.array(code, dtype=np.float32, copy=True, order="C")
        mutated_code[0::2] = np.nextafter(
            mutated_code[0::2],
            np.float32(np.inf),
        )
        if not np.all(np.isfinite(mutated_code)):
            raise G110GeneratedY1PoseError(
                "even-code exclusion witness cannot be perturbed finitely"
            )
        mutated = compile_v9_from_state(
            config=config,
            params={
                key: value
                for key, value in semantic_params.items()
                if key != "code"
            },
            interleaved_code=mutated_code,
        )
        if encode_v9_packet(mutated) != semantic_packet:
            raise G110GeneratedY1PoseError(
                "G105 semantic packet unexpectedly consumes even code rows"
            )
        xi_stored = np.asarray(
            pose_params["pose_carrier.xi_stored"],
            dtype=np.float64,
        )
        dxi = np.asarray(
            pose_params["pose_carrier.dxi"],
            dtype=np.float64,
        )
        if (
            xi_stored.shape != (PAIR_COUNT, 6)
            or dxi.shape != (PAIR_COUNT, 6)
            or not np.all(np.isfinite(xi_stored))
            or not np.all(np.isfinite(dxi))
        ):
            raise G110GeneratedY1PoseError(
                "G111 pose tensors are not finite [600,6] tables"
            )
        residual_scale = float(scalars["__cfg_pose_carrier_residual_scale"])
        pitch = float(scalars["__cfg_pose_carrier_pitch"])
        xi_initializer = np.ascontiguousarray(
            xi_stored + residual_scale * dxi,
            dtype=np.float64,
        )
        xi_initializer.setflags(write=False)
        even_digest = _sha256(
            np.asarray(code[0::2], dtype=">f4").tobytes(order="C")
        )
        partition = {
            "schema": "tac.g110_g111_tensor_partition.v1",
            "all_checkpoint_tensor_keys": sorted(params),
            "semantic_owned_tensor_keys": sorted(semantic_params),
            "conditional_initializer_tensor_keys": sorted(pose_params),
            "even_code_rows": {
                "owner": "excluded",
                "reason": "generated_y1 source has no even-code runtime consumer",
                "sha256": even_digest,
                "perturbation_preserved_semantic_packet": True,
            },
            "total": set(params) == set(semantic_params).union(pose_params),
            "disjoint": not set(semantic_params).intersection(pose_params),
        }
        instance = object.__new__(cls)
        values = {
            "checkpoint_sha256": checkpoint_sha,
            "semantic_packet_sha256": _sha256(semantic_packet),
            "xi_initializer_sha256": _xi_digest(xi_initializer),
            "even_code_exclusion_sha256": even_digest,
            "pitch": pitch,
            "residual_scale": residual_scale,
            "tensor_partition_sha256": _sha256(_canonical_json(partition)),
            "xi_initializer": xi_initializer,
        }
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        return instance


@dataclass(frozen=True, slots=True, init=False)
class G110G112CompileCustodyV1:
    """One recursively reopened G112 partition plus its exact G109 aggregate."""

    partition_receipt_sha256: str
    semantic_child_sha256: str
    pose_initializer_sha256: str
    semantic_packet_sha256: str
    source_deploy_checkpoint_sha256: str
    source_resume_checkpoint_sha256: str
    source_lineage_receipt_sha256: str
    source_checkpoint_id_sha256: str
    source_root_sha256: str
    target_projection_sha256: str
    target_capsule_receipt_sha256: str
    pose_targets_sha256: str
    partition_receipt: G112PartitionReceiptV2 = field(repr=False)
    semantic_child: G112SemanticChildV1 = field(repr=False)
    pose_initializer: G112PoseInitializerV1 = field(repr=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "G110/G112 custody requires from_physical_partition_receipt(); "
            "hash-only construction is forbidden"
        )

    @classmethod
    def from_physical_partition_receipt(
        cls,
        *,
        partition_receipt_path: Path,
        expected_partition_receipt_sha256: str,
        target_capsule_receipt: Path,
        expected_target_capsule_receipt_sha256: str,
    ) -> G110G112CompileCustodyV1:
        """Reopen one recursively proven G112 partition and its target aggregate."""

        try:
            partition_receipt = open_g112_partition_receipt(
                partition_receipt_path,
                expected_sha256=_require_sha256(
                    expected_partition_receipt_sha256,
                    name="G112 partition receipt",
                ),
            )
        except (OSError, ValueError, G112CheckpointPartitionError) as exc:
            raise G110GeneratedY1PoseError(
                "G112 partition and recursive source ancestry did not reopen"
            ) from exc
        source_chain = partition_receipt.source_chain
        if (
            source_chain.complete_trajectory_proven is not True
            or not source_chain.nodes
            or source_chain.current is not source_chain.nodes[-1]
        ):
            raise G110GeneratedY1PoseError(
                "G112 source chain is not a complete physical trajectory"
            )
        source_pair = source_chain.current.pair
        semantic_child = partition_receipt.semantic_child
        pose_initializer = partition_receipt.initializer
        if (
            partition_receipt.semantic_packet_sha256
            != semantic_child.semantic_packet_sha256
            or
            semantic_child.semantic_packet_sha256
            != pose_initializer.semantic_packet_sha256
            or pose_initializer.selected_preimage_schema
            != V10_FACTOR2_SELECTED_PREIMAGE_SCHEMA
        ):
            raise G110GeneratedY1PoseError(
                "G112 children disagree or lack exact V10 selected-preimage custody"
            )
        scalars = semantic_child.g105_scalars
        projection_text = scalars.get(G109_PROJECTION_KEY)
        projection_sha = _require_sha256(
            scalars.get(G109_PROJECTION_SHA_KEY),
            name="G112 semantic target projection",
        )
        if (
            type(projection_text) is not str
            or projection_sha != pose_initializer.target_projection_sha256
        ):
            raise G110GeneratedY1PoseError(
                "G112 semantic and pose children bind different G109 projections"
            )
        try:
            projection = json.loads(projection_text)
        except json.JSONDecodeError as exc:
            raise G110GeneratedY1PoseError(
                "G112 semantic child target projection is not JSON"
            ) from exc
        if (
            type(projection) is not dict
            or _sha256(_canonical_json(projection)) != projection_sha
            or projection.get("pair_count") != PRODUCTION_PAIR_COUNT
            or projection.get("scorer_pair_batch_size")
            != PRODUCTION_BATCH_PAIRS
            or projection.get("same_forward_seg_margin_pose") is not True
            or projection.get("encoder_only") is not True
            or projection.get("candidate_payload_allowed") is not False
        ):
            raise G110GeneratedY1PoseError(
                "G112 target projection is not exact n600 batch-16 encoder custody"
            )
        receipt_sha = _require_sha256(
            expected_target_capsule_receipt_sha256,
            name="G109 target capsule receipt",
        )
        try:
            loader = V9TrainingTargetCapsuleLoaderV1.open(
                target_capsule_receipt,
                expected_sha256=receipt_sha,
            )
        except (OSError, ValueError, V9TrainingTargetCapsuleError) as exc:
            raise G110GeneratedY1PoseError(
                "G109 target capsule did not strictly reopen for G112"
            ) from exc
        raw = loader.receipt.get("raw_arrays")
        aggregate_binding = projection.get("aggregate_receipt")
        if (
            loader.receipt.get("schema") != TARGET_CAPSULE_SCHEMA
            or loader.pair_count != PRODUCTION_PAIR_COUNT
            or loader.batch_pairs != PRODUCTION_BATCH_PAIRS
            or loader.preflight.get("test_only_small_fixture") is not False
            or type(raw) is not dict
            or set(raw) != {"labels", "margins", "poses"}
            or type(aggregate_binding) is not dict
            or set(aggregate_binding) != {"path", "bytes", "sha256"}
            or aggregate_binding["path"] != str(loader.receipt_path)
            or aggregate_binding["bytes"] != loader.receipt_path.stat().st_size
            or aggregate_binding["sha256"] != receipt_sha
            or projection.get("aggregate_receipt_sha256")
            != loader.receipt.get("aggregate_receipt_sha256")
        ):
            raise G110GeneratedY1PoseError(
                "G112 projection and physical G109 aggregate disagree"
            )
        pose_targets_sha = _require_sha256(
            raw["poses"].get("sha256"),
            name="G109 PoseNet targets",
        )
        instance = object.__new__(cls)
        for name, value in {
            "partition_receipt_sha256": partition_receipt.receipt_sha256,
            "semantic_child_sha256": semantic_child.checkpoint_sha256,
            "pose_initializer_sha256": pose_initializer.checkpoint_sha256,
            "semantic_packet_sha256": semantic_child.semantic_packet_sha256,
            "source_deploy_checkpoint_sha256": source_pair.deploy.sha256,
            "source_resume_checkpoint_sha256": source_pair.resume.sha256,
            "source_lineage_receipt_sha256": (
                source_chain.current.receipt_sha256
            ),
            "source_checkpoint_id_sha256": (
                source_pair.checkpoint_id_sha256
            ),
            "source_root_sha256": source_pair.root_sha256,
            "target_projection_sha256": projection_sha,
            "target_capsule_receipt_sha256": receipt_sha,
            "pose_targets_sha256": pose_targets_sha,
            "partition_receipt": partition_receipt,
            "semantic_child": semantic_child,
            "pose_initializer": pose_initializer,
        }.items():
            object.__setattr__(instance, name, value)
        return instance


@dataclass(frozen=True, slots=True)
class PostG105PoseRefitV1:
    xi_eff: np.ndarray = field(repr=False)
    pitch: float
    q_levels: int
    selected_xip2_coder: str
    checkpoint_sha256: str
    run_receipt_sha256: str
    xi_eff_sha256: str


def _npz_scalar(value: np.ndarray, *, name: str) -> object:
    raw = np.asarray(value)
    if raw.size != 1:
        raise G110GeneratedY1PoseError(f"{name} must be a scalar NPZ member")
    return raw.reshape(()).item()


def _verify_post_g105_refit(
    *,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    run_receipt: Path,
    expected_run_receipt_sha256: str,
    base_custody: G110G112CompileCustodyV1,
    initializer: G112PoseInitializerV1,
    semantic_packet: bytes,
    final_y1_binding: str,
) -> PostG105PoseRefitV1:
    expected_checkpoint_sha = _require_sha256(
        expected_checkpoint_sha256,
        name="expected post-G105 pose refit checkpoint",
    )
    expected_run_sha = _require_sha256(
        expected_run_receipt_sha256,
        name="expected post-G105 pose refit run receipt",
    )
    checkpoint_path = _regular_file(
        checkpoint,
        name="post-G105 pose refit checkpoint",
    )
    run_path = _regular_file(
        run_receipt,
        name="post-G105 pose refit run receipt",
    )
    observed_checkpoint_sha = sha256_file(checkpoint_path)
    observed_run_sha = sha256_file(run_path)
    if (
        observed_checkpoint_sha != expected_checkpoint_sha
        or observed_run_sha != expected_run_sha
    ):
        raise G110GeneratedY1PoseError(
            "post-G105 refit differs from its externally expected SHA-256"
        )
    try:
        run = json.loads(run_path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G110GeneratedY1PoseError(
            "post-G105 pose refit run receipt is not JSON"
        ) from exc
    run_keys = {
        "schema",
        "run_id",
        "seed",
        "source_git_sha",
        "command",
        "fresh_own_lineage",
        "source_contract",
        "render_order",
        "y1_selected_preimage_schema",
        "source_g112_partition_receipt_sha256",
        "source_g112_semantic_child_sha256",
        "source_g112_pose_initializer_sha256",
        "source_g111_deploy_checkpoint_sha256",
        "source_g111_resume_checkpoint_sha256",
        "source_g111_lineage_receipt_sha256",
        "source_g111_checkpoint_id_sha256",
        "source_g111_root_sha256",
        "semantic_packet_sha256",
        "final_y1_binding_sha256",
        "xi_initializer_sha256",
        "target_projection_sha256",
        "target_capsule_receipt_sha256",
        "pose_targets_sha256",
        "selected_xip2_coder",
        "g110_selected_xip2_coder_abi_closed",
        "exact_public_receiver_in_loop",
        "resumable_from_disk",
        "stage_checkpoints_preserved",
        "stage_checkpoints",
        "final_checkpoint",
        "research_only",
        "candidate_claim",
        "score_claim",
        "pointer_moved",
        "receipt_sha256",
    }
    if type(run) is not dict or set(run) != run_keys:
        raise G110GeneratedY1PoseError(
            "post-G105 pose refit run receipt key set differs"
        )
    receipt_sha = _require_sha256(
        run["receipt_sha256"],
        name="post-G105 pose refit run receipt",
    )
    if _sha256(
        _canonical_json(
            {key: value for key, value in run.items() if key != "receipt_sha256"}
        )
    ) != receipt_sha:
        raise G110GeneratedY1PoseError(
            "post-G105 pose refit run receipt self-hash differs"
        )
    stages = run["stage_checkpoints"]
    final_binding = run["final_checkpoint"]
    if (
        run["schema"] != POST_G105_REFIT_RUN_SCHEMA
        or type(run["run_id"]) is not str
        or not run["run_id"]
        or type(run["seed"]) is not int
        or type(run["source_git_sha"]) is not str
        or len(run["source_git_sha"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in run["source_git_sha"]
        )
        or type(run["command"]) is not list
        or not run["command"]
        or any(type(token) is not str or not token for token in run["command"])
        or "--resume-from" not in run["command"]
        or run["fresh_own_lineage"] is not True
        or run["source_contract"] != SOURCE_DOMAIN
        or run["render_order"] != RENDER_ORDER
        or run["y1_selected_preimage_schema"]
        != V10_FACTOR2_SELECTED_PREIMAGE_SCHEMA
        or run["source_g112_partition_receipt_sha256"]
        != base_custody.partition_receipt_sha256
        or run["source_g112_semantic_child_sha256"]
        != base_custody.semantic_child_sha256
        or run["source_g112_pose_initializer_sha256"]
        != base_custody.pose_initializer_sha256
        or run["source_g111_deploy_checkpoint_sha256"]
        != base_custody.source_deploy_checkpoint_sha256
        or run["source_g111_resume_checkpoint_sha256"]
        != base_custody.source_resume_checkpoint_sha256
        or run["source_g111_lineage_receipt_sha256"]
        != base_custody.source_lineage_receipt_sha256
        or run["source_g111_checkpoint_id_sha256"]
        != base_custody.source_checkpoint_id_sha256
        or run["source_g111_root_sha256"]
        != base_custody.source_root_sha256
        or run["semantic_packet_sha256"] != _sha256(semantic_packet)
        or run["final_y1_binding_sha256"] != final_y1_binding
        or run["xi_initializer_sha256"] != _xi_digest(initializer.xi_init)
        or run["target_projection_sha256"]
        != base_custody.target_projection_sha256
        or run["target_capsule_receipt_sha256"]
        != base_custody.target_capsule_receipt_sha256
        or run["pose_targets_sha256"] != base_custody.pose_targets_sha256
        or run["selected_xip2_coder"] not in {"none", "delta_ar"}
        or run["g110_selected_xip2_coder_abi_closed"] is not True
        or run["exact_public_receiver_in_loop"] is not True
        or run["resumable_from_disk"] is not True
        or run["stage_checkpoints_preserved"] is not True
        or type(stages) is not list
        or not stages
        or stages[-1] != final_binding
        or run["research_only"] is not True
        or run["candidate_claim"] is not False
        or run["score_claim"] is not False
        or run["pointer_moved"] is not False
    ):
        raise G110GeneratedY1PoseError(
            "post-G105 refit is not exact-source resumable research lineage"
        )
    reopened = [
        _reopen_binding(value, name=f"post-G105 refit stage {index}")
        for index, value in enumerate(stages)
    ]
    final_path = _reopen_binding(
        final_binding,
        name="post-G105 refit final checkpoint",
    )
    if reopened[-1] != final_path or final_path != checkpoint_path:
        raise G110GeneratedY1PoseError(
            "post-G105 final checkpoint is not the preserved final stage"
        )
    try:
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            expected_members = {
                "schema",
                "run_id",
                "seed",
                "source_contract",
                "render_order",
                "y1_selected_preimage_schema",
                "source_g112_partition_receipt_sha256",
                "source_g112_semantic_child_sha256",
                "source_g112_pose_initializer_sha256",
                "source_g111_deploy_checkpoint_sha256",
                "source_g111_resume_checkpoint_sha256",
                "source_g111_lineage_receipt_sha256",
                "source_g111_checkpoint_id_sha256",
                "source_g111_root_sha256",
                "semantic_packet_sha256",
                "final_y1_binding_sha256",
                "xi_initializer_sha256",
                "target_projection_sha256",
                "target_capsule_receipt_sha256",
                "pose_targets_sha256",
                "exact_public_receiver_in_loop",
                "pitch",
                "q_levels",
                "selected_xip2_coder",
                "xi_eff",
            }
            if set(archive.files) != expected_members:
                raise G110GeneratedY1PoseError(
                    "post-G105 refit checkpoint member set differs"
                )
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise G110GeneratedY1PoseError(
            "post-G105 refit checkpoint is not a strict NPZ"
        ) from exc
    xi_raw = arrays["xi_eff"]
    pitch = float(_npz_scalar(arrays["pitch"], name="refit pitch"))
    q_levels = int(_npz_scalar(arrays["q_levels"], name="refit q_levels"))
    if (
        xi_raw.dtype != np.float64
        or xi_raw.shape != (PAIR_COUNT, 6)
        or not xi_raw.flags.c_contiguous
        or not np.all(np.isfinite(xi_raw))
        or not math.isfinite(pitch)
        or pitch != initializer.pitch
        or not 1 <= q_levels <= 32_767
    ):
        raise G110GeneratedY1PoseError(
            "post-G105 refit trajectory/pitch/quantizer is not canonical"
        )
    scalar_expectations = {
        "schema": POST_G105_REFIT_CHECKPOINT_SCHEMA,
        "run_id": run["run_id"],
        "seed": run["seed"],
        "source_contract": SOURCE_DOMAIN,
        "render_order": RENDER_ORDER,
        "y1_selected_preimage_schema": V10_FACTOR2_SELECTED_PREIMAGE_SCHEMA,
        "source_g112_partition_receipt_sha256": (
            base_custody.partition_receipt_sha256
        ),
        "source_g112_semantic_child_sha256": base_custody.semantic_child_sha256,
        "source_g112_pose_initializer_sha256": (
            base_custody.pose_initializer_sha256
        ),
        "source_g111_deploy_checkpoint_sha256": (
            base_custody.source_deploy_checkpoint_sha256
        ),
        "source_g111_resume_checkpoint_sha256": (
            base_custody.source_resume_checkpoint_sha256
        ),
        "source_g111_lineage_receipt_sha256": (
            base_custody.source_lineage_receipt_sha256
        ),
        "source_g111_checkpoint_id_sha256": (
            base_custody.source_checkpoint_id_sha256
        ),
        "source_g111_root_sha256": base_custody.source_root_sha256,
        "semantic_packet_sha256": _sha256(semantic_packet),
        "final_y1_binding_sha256": final_y1_binding,
        "xi_initializer_sha256": _xi_digest(initializer.xi_init),
        "target_projection_sha256": base_custody.target_projection_sha256,
        "target_capsule_receipt_sha256": base_custody.target_capsule_receipt_sha256,
        "pose_targets_sha256": base_custody.pose_targets_sha256,
        "exact_public_receiver_in_loop": 1,
        "selected_xip2_coder": run["selected_xip2_coder"],
    }
    for name, expected in scalar_expectations.items():
        if _npz_scalar(arrays[name], name=name) != expected:
            raise G110GeneratedY1PoseError(
                f"post-G105 refit checkpoint binding differs: {name}"
            )
    xi = np.ascontiguousarray(xi_raw, dtype=np.float64)
    xi.setflags(write=False)
    return PostG105PoseRefitV1(
        xi_eff=xi,
        pitch=pitch,
        q_levels=q_levels,
        selected_xip2_coder=run["selected_xip2_coder"],
        checkpoint_sha256=observed_checkpoint_sha,
        run_receipt_sha256=observed_run_sha,
        xi_eff_sha256=_xi_digest(xi),
    )


@dataclass(frozen=True, slots=True, eq=False)
class ParsedG110GeneratedY1PoseV1:
    semantic_packet: bytes = field(repr=False)
    final_y1_binding_sha256: str
    xip2_payload: bytes = field(repr=False)
    q: np.ndarray = field(repr=False)
    scales: np.ndarray = field(repr=False)
    xi_eff: np.ndarray = field(repr=False)
    pitch: float
    packet: bytes = field(repr=False)

    def render_camera_y0(
        self,
        pair_id: int,
        camera_y1: np.ndarray,
    ) -> np.ndarray:
        return render_camera_y0(
            camera_y1,
            pair_id=pair_id,
            xi_eff=self.xi_eff,
            pitch=self.pitch,
        )


def _encode_packet(
    *,
    semantic_packet: bytes,
    final_y1_binding: str,
    xip2_payload: bytes,
    pitch: float,
) -> bytes:
    if (
        type(semantic_packet) is not bytes
        or not 0 < len(semantic_packet) <= MAX_SEMANTIC_PACKET_BYTES
        or type(xip2_payload) is not bytes
        or not 0 < len(xip2_payload) <= MAX_XIP2_BYTES
        or type(pitch) is not float
        or not math.isfinite(pitch)
        or abs(pitch) > math.pi / 2.0
    ):
        raise G110GeneratedY1PoseError("pose packet operands are outside bounds")
    binding = bytes.fromhex(_require_sha256(final_y1_binding, name="final Y1 binding"))
    body = semantic_packet + xip2_payload
    packet = _HEADER.pack(
        MAGIC,
        VERSION,
        0,
        PAIR_COUNT,
        len(semantic_packet),
        len(xip2_payload),
        pitch,
        binding,
        zlib.crc32(body) & 0xFFFFFFFF,
    ) + body
    if len(packet) > MAX_PACKET_BYTES:
        raise G110GeneratedY1PoseError("pose packet exceeds bounded counted ABI")
    return packet


def parse_g110_generated_y1_pose_v1(
    payload: bytes,
) -> ParsedG110GeneratedY1PoseV1:
    if (
        type(payload) is not bytes
        or not _HEADER.size < len(payload) <= MAX_PACKET_BYTES
    ):
        raise G110GeneratedY1PoseError("pose packet must be bounded exact bytes")
    try:
        (
            magic,
            version,
            flags,
            pairs,
            semantic_length,
            xip2_length,
            pitch,
            binding_raw,
            expected_crc,
        ) = _HEADER.unpack_from(payload)
    except struct.error as exc:
        raise G110GeneratedY1PoseError("pose header is truncated") from exc
    if (
        magic != MAGIC
        or version != VERSION
        or flags != 0
        or pairs != PAIR_COUNT
        or not 0 < semantic_length <= MAX_SEMANTIC_PACKET_BYTES
        or not 0 < xip2_length <= MAX_XIP2_BYTES
        or not math.isfinite(pitch)
        or abs(pitch) > math.pi / 2.0
        or _HEADER.size + semantic_length + xip2_length != len(payload)
    ):
        raise G110GeneratedY1PoseError("pose header changes the closed n600 ABI")
    body = payload[_HEADER.size :]
    if zlib.crc32(body) & 0xFFFFFFFF != expected_crc:
        raise G110GeneratedY1PoseError("pose packet body CRC32 mismatch")
    semantic = body[:semantic_length]
    xip2 = body[semantic_length:]
    provider = open_final_y1_provider(semantic)
    if provider.variant_id != V9_VARIANT_ID:
        raise G110GeneratedY1PoseError(
            "generated-Y1 pose packet requires exact G105 semantic Y1"
        )
    try:
        q, scales = parse_xi_payload(xip2)
    except (ValueError, XiPoseCoderError, struct.error) as exc:
        raise G110GeneratedY1PoseError("pose XIP2 payload did not parse") from exc
    if (
        q.dtype != np.int16
        or q.shape != (PAIR_COUNT, 6)
        or scales.dtype != np.float32
        or scales.shape != (6,)
        or not np.all(np.isfinite(scales))
        or np.any(scales <= 0.0)
        or len(xip2) < 5
        or xip2[4] not in {0, 1}
    ):
        raise G110GeneratedY1PoseError("pose XIP2 geometry/scales/coder differ")
    coder = "none" if xip2[4] == 0 else "delta_ar"
    canonical_xip2 = serialize_xi_payload(q, scales, coder=coder)
    if canonical_xip2 != xip2:
        raise G110GeneratedY1PoseError(
            "pose XIP2 has trailing/noncanonical bytes"
        )
    xi = np.ascontiguousarray(dequantize_xi(q, scales), dtype=np.float64)
    if not np.all(np.isfinite(xi)):
        raise G110GeneratedY1PoseError("decoded pose trajectory is non-finite")
    binding = binding_raw.hex()
    canonical = _encode_packet(
        semantic_packet=semantic,
        final_y1_binding=binding,
        xip2_payload=xip2,
        pitch=float(pitch),
    )
    if canonical != payload:
        raise G110GeneratedY1PoseError(
            "pose packet changed under canonical re-emission"
        )
    q = np.ascontiguousarray(q)
    scales = np.ascontiguousarray(scales)
    for array in (q, scales, xi):
        array.setflags(write=False)
    return ParsedG110GeneratedY1PoseV1(
        semantic_packet=semantic,
        final_y1_binding_sha256=binding,
        xip2_payload=xip2,
        q=q,
        scales=scales,
        xi_eff=xi,
        pitch=float(pitch),
        packet=payload,
    )


def render_camera_y0(
    camera_y1: np.ndarray,
    *,
    pair_id: int,
    xi_eff: np.ndarray,
    pitch: float,
) -> np.ndarray:
    if type(pair_id) is not int or not 0 <= pair_id < PAIR_COUNT:
        raise G110GeneratedY1PoseError("pair_id is outside exact n600")
    source = np.asarray(camera_y1)
    if (
        source.dtype != np.uint8
        or source.shape != (CAMERA_H, CAMERA_W, CHANNELS)
    ):
        raise G110GeneratedY1PoseError(
            "pose source must be actual uint8 camera Y1"
        )
    xi = np.asarray(xi_eff)
    if xi.shape != (PAIR_COUNT, 6) or not np.all(np.isfinite(xi)):
        raise G110GeneratedY1PoseError("pose trajectory is not finite [600,6]")
    geom = GroundHomographyGeom.eon(
        native_hw=(CAMERA_H, CAMERA_W),
        pitch=float(pitch),
    )
    try:
        result = warp_frame0_uint8_numpy(
            np.ascontiguousarray(source),
            xi[pair_id],
            geom,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise G110GeneratedY1PoseError("native pose warp failed") from exc
    return np.ascontiguousarray(result)


@dataclass(frozen=True, slots=True)
class G110CompleteArchiveWireCandidateV1:
    """One complete outer-ZIP measurement in the joint wire/method matrix."""

    y1_wire_codec: Y1WireCodecV1
    outer_zip_method: G110OuterZipMethodV1
    semantic_packet_bytes: int
    semantic_packet_sha256: str
    product_packet_bytes: int
    product_packet_sha256: str
    archive_bytes: int
    archive_sha256: str


def _final_y1_binding_from_population(
    semantic_packet: bytes,
    population_digest: bytes,
) -> str:
    if (
        type(semantic_packet) is not bytes
        or type(population_digest) is not bytes
        or len(population_digest) != 32
    ):
        raise G110GeneratedY1PoseError(
            "final-Y1 binding operands are not exact bytes"
        )
    return _sha256(
        FINAL_Y1_DOMAIN
        + hashlib.sha256(semantic_packet).digest()
        + population_digest
    )


def _select_complete_archive_y1_wire(
    *,
    semantic_program: ExactV9SemanticRootY1ProgramV1,
    expected_population_digest: bytes,
    xip2_payload: bytes,
    pitch: float,
) -> tuple[
    bytes,
    bytes,
    G110CompleteArchiveWireCandidateV1,
    tuple[G110CompleteArchiveWireCandidateV1, ...],
]:
    """Measure both complete archives and select by exact final ZIP bytes."""

    try:
        variants = encode_packet_y1_variants(semantic_program)
    except (TypeError, ValueError, ExactV9SemanticRootError) as exc:
        raise G110GeneratedY1PoseError(
            "G105 did not enumerate both typed semantic wire families"
        ) from exc
    if tuple(codec for codec, _packet in variants) != tuple(Y1WireCodecV1):
        raise G110GeneratedY1PoseError(
            "G105 semantic wire enumeration is incomplete or reordered"
        )
    artifacts: list[tuple[bytes, bytes]] = []
    records: list[G110CompleteArchiveWireCandidateV1] = []
    for codec, semantic_packet in variants:
        parsed = parse_v9_packet(semantic_packet)
        if parsed.y1_wire_codec is not codec:
            raise G110GeneratedY1PoseError(
                "G105 parse-back lost the selected semantic wire family"
            )
        provider = open_final_y1_provider(semantic_packet)
        population_digest = _population_digest(provider)
        if population_digest != expected_population_digest:
            raise G110GeneratedY1PoseError(
                "raw/Rice semantic alternatives render different Y1 populations"
            )
        binding = _final_y1_binding_from_population(
            semantic_packet,
            population_digest,
        )
        packet = _encode_packet(
            semantic_packet=semantic_packet,
            final_y1_binding=binding,
            xip2_payload=xip2_payload,
            pitch=pitch,
        )
        parsed_product = parse_g110_generated_y1_pose_v1(packet)
        if (
            parsed_product.semantic_packet != semantic_packet
            or parsed_product.final_y1_binding_sha256 != binding
        ):
            raise AssertionError(
                "complete semantic-wire packet changed under parse-back"
            )
        for outer_zip_method in _OUTER_ZIP_METHODS:
            archive = _build_g110_archive_for_method(
                packet,
                outer_zip_method,
            )
            reopened_packet, reopened_method = _read_g110_archive_member(
                archive,
            )
            if reopened_packet != packet or reopened_method is not outer_zip_method:
                raise AssertionError(
                    "complete outer-ZIP alternative changed under parse-back"
                )
            records.append(
                G110CompleteArchiveWireCandidateV1(
                    y1_wire_codec=codec,
                    outer_zip_method=outer_zip_method,
                    semantic_packet_bytes=len(semantic_packet),
                    semantic_packet_sha256=_sha256(semantic_packet),
                    product_packet_bytes=len(packet),
                    product_packet_sha256=_sha256(packet),
                    archive_bytes=len(archive),
                    archive_sha256=_sha256(archive),
                )
            )
            artifacts.append((packet, archive))
    if len(records) != len(Y1WireCodecV1) * len(_OUTER_ZIP_METHODS):
        raise G110GeneratedY1PoseError(
            "complete-archive arbitration did not measure the full matrix"
        )
    selected_index = min(
        range(len(records)),
        key=lambda index: (
            records[index].archive_bytes,
            int(records[index].y1_wire_codec),
            int(records[index].outer_zip_method),
            records[index].archive_sha256,
        ),
    )
    if (
        parse_g110_generated_y1_pose_archive(
            artifacts[selected_index][1]
        )
        != artifacts[selected_index][0]
    ):
        raise AssertionError("selected complete archive changed under parse-back")
    return (
        artifacts[selected_index][0],
        artifacts[selected_index][1],
        records[selected_index],
        tuple(records),
    )


@dataclass(frozen=True, slots=True)
class CompiledG110GeneratedY1PoseV1:
    packet: bytes = field(repr=False)
    archive: bytes = field(repr=False)
    packet_sha256: str
    archive_sha256: str
    archive_bytes: int
    semantic_packet_sha256: str
    refit_source_semantic_packet_sha256: str
    final_y1_binding_sha256: str
    selected_y1_wire_codec: Y1WireCodecV1
    selected_xip2_coder: str
    selected_outer_zip_method: G110OuterZipMethodV1
    complete_archive_wire_candidates: tuple[
        G110CompleteArchiveWireCandidateV1,
        ...,
    ]
    xip2_bytes: int
    initializer_sha256: str
    g112_partition_receipt_sha256: str
    g112_semantic_child_sha256: str
    g112_pose_initializer_sha256: str
    g111_source_checkpoint_id_sha256: str
    g111_source_root_sha256: str
    refit_xi_sha256: str
    refit_checkpoint_sha256: str
    refit_run_receipt_sha256: str
    candidate_or_score_claim: bool = False


def compile_g110_generated_y1_pose_v1(
    *,
    target_capsule_receipt: Path,
    expected_target_capsule_receipt_sha256: str,
    g112_partition_receipt: Path,
    expected_g112_partition_receipt_sha256: str,
    post_g105_refit_checkpoint: Path,
    expected_post_g105_refit_checkpoint_sha256: str,
    post_g105_refit_run_receipt: Path,
    expected_post_g105_refit_run_receipt_sha256: str,
) -> CompiledG110GeneratedY1PoseV1:
    """Compile only from one recursive G112 partition plus the post-G105 refit."""

    base_custody = (
        G110G112CompileCustodyV1.from_physical_partition_receipt(
            partition_receipt_path=g112_partition_receipt,
            expected_partition_receipt_sha256=(
                expected_g112_partition_receipt_sha256
            ),
            target_capsule_receipt=target_capsule_receipt,
            expected_target_capsule_receipt_sha256=(
                expected_target_capsule_receipt_sha256
            ),
        )
    )
    semantic_packet = base_custody.semantic_child.semantic_packet
    initializer = base_custody.pose_initializer
    provider = open_final_y1_provider(semantic_packet)
    if provider.variant_id != V9_VARIANT_ID:
        raise G110GeneratedY1PoseError("pose compile admits only exact G105 Y1")
    if (
        base_custody.semantic_packet_sha256 != _sha256(semantic_packet)
        or initializer.semantic_packet_sha256 != _sha256(semantic_packet)
    ):
        raise G110GeneratedY1PoseError(
            "G112 children changed semantic packet identity after custody"
        )
    population_digest = _population_digest(provider)
    binding = _final_y1_binding_from_population(
        semantic_packet,
        population_digest,
    )
    if binding != final_y1_binding_sha256(provider):
        raise AssertionError("final-Y1 binding helper disagrees with generic ABI")
    refit = _verify_post_g105_refit(
        checkpoint=post_g105_refit_checkpoint,
        expected_checkpoint_sha256=(
            expected_post_g105_refit_checkpoint_sha256
        ),
        run_receipt=post_g105_refit_run_receipt,
        expected_run_receipt_sha256=(
            expected_post_g105_refit_run_receipt_sha256
        ),
        base_custody=base_custody,
        initializer=initializer,
        semantic_packet=semantic_packet,
        final_y1_binding=binding,
    )
    q, scales = quantize_xi(refit.xi_eff, q_levels=refit.q_levels)
    xip2 = serialize_xi_payload(
        q,
        scales,
        coder=refit.selected_xip2_coder,
    )
    semantic_program = parse_v9_packet(semantic_packet)
    packet, archive, selected_wire, wire_candidates = (
        _select_complete_archive_y1_wire(
            semantic_program=semantic_program,
            expected_population_digest=population_digest,
            xip2_payload=xip2,
            pitch=float(refit.pitch),
        )
    )
    parsed = parse_g110_generated_y1_pose_v1(packet)
    if (
        not np.array_equal(parsed.q, q)
        or not np.array_equal(parsed.scales, scales)
        or parsed.final_y1_binding_sha256
        != _final_y1_binding_from_population(
            parsed.semantic_packet,
            population_digest,
        )
    ):
        raise AssertionError("internal pose packet parse-back changed operands")
    return CompiledG110GeneratedY1PoseV1(
        packet=packet,
        archive=archive,
        packet_sha256=_sha256(packet),
        archive_sha256=_sha256(archive),
        archive_bytes=len(archive),
        semantic_packet_sha256=_sha256(parsed.semantic_packet),
        refit_source_semantic_packet_sha256=_sha256(semantic_packet),
        final_y1_binding_sha256=parsed.final_y1_binding_sha256,
        selected_y1_wire_codec=selected_wire.y1_wire_codec,
        selected_xip2_coder=refit.selected_xip2_coder,
        selected_outer_zip_method=selected_wire.outer_zip_method,
        complete_archive_wire_candidates=wire_candidates,
        xip2_bytes=len(xip2),
        initializer_sha256=_xi_digest(initializer.xi_init),
        g112_partition_receipt_sha256=(
            base_custody.partition_receipt_sha256
        ),
        g112_semantic_child_sha256=base_custody.semantic_child_sha256,
        g112_pose_initializer_sha256=base_custody.pose_initializer_sha256,
        g111_source_checkpoint_id_sha256=(
            base_custody.source_checkpoint_id_sha256
        ),
        g111_source_root_sha256=base_custody.source_root_sha256,
        refit_xi_sha256=refit.xi_eff_sha256,
        refit_checkpoint_sha256=refit.checkpoint_sha256,
        refit_run_receipt_sha256=refit.run_receipt_sha256,
    )


def _zip_member(method: G110OuterZipMethodV1) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(PACKET_MEMBER, date_time=_ZIP_TIMESTAMP)
    info.compress_type = int(method)
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits = 0
    return info


def _build_g110_archive_for_method(
    packet: bytes,
    method: G110OuterZipMethodV1,
) -> bytes:
    parse_g110_generated_y1_pose_v1(packet)
    if type(method) is not G110OuterZipMethodV1:
        raise G110GeneratedY1PoseError(
            "outer ZIP method must be a typed G110OuterZipMethodV1"
        )
    stream = io.BytesIO()
    compression_kwargs: dict[str, int] = {"compression": int(method)}
    if method is G110OuterZipMethodV1.DEFLATE:
        compression_kwargs["compresslevel"] = 9
    with zipfile.ZipFile(
        stream,
        "w",
        allowZip64=False,
        **compression_kwargs,
    ) as archive:
        write_kwargs: dict[str, int] = {"compress_type": int(method)}
        if method is G110OuterZipMethodV1.DEFLATE:
            write_kwargs["compresslevel"] = 9
        archive.writestr(_zip_member(method), packet, **write_kwargs)
    result = stream.getvalue()
    if not result or len(result) > MAX_ARCHIVE_BYTES:
        raise G110GeneratedY1PoseError("pose archive exceeds bounded envelope")
    return result


def _select_outer_zip_method(
    packet: bytes,
) -> tuple[G110OuterZipMethodV1, bytes]:
    alternatives = tuple(
        (method, _build_g110_archive_for_method(packet, method))
        for method in _OUTER_ZIP_METHODS
    )
    return min(
        alternatives,
        key=lambda item: (
            len(item[1]),
            int(item[0]),
            _sha256(item[1]),
        ),
    )


def build_g110_generated_y1_pose_archive(packet: bytes) -> bytes:
    _method, result = _select_outer_zip_method(packet)
    if parse_g110_generated_y1_pose_archive(result) != packet:
        raise AssertionError("internal pose archive parse-back changed packet")
    return result


def _read_g110_archive_member(
    archive_bytes: bytes,
) -> tuple[bytes, G110OuterZipMethodV1]:
    if (
        type(archive_bytes) is not bytes
        or not 0 < len(archive_bytes) <= MAX_ARCHIVE_BYTES
    ):
        raise G110GeneratedY1PoseError("pose archive must be bounded exact bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            infos = archive.infolist()
            if [info.filename for info in infos] != [PACKET_MEMBER]:
                raise G110GeneratedY1PoseError(
                    "pose archive member set/order differs"
                )
            info = infos[0]
            mode = (info.external_attr >> 16) & 0o170000
            try:
                method = G110OuterZipMethodV1(info.compress_type)
            except ValueError as exc:
                raise G110GeneratedY1PoseError(
                    "pose archive compression method is unsupported"
                ) from exc
            if (
                info.is_dir()
                or info.flag_bits & 0x1
                or mode == 0o120000
                or not _HEADER.size < info.file_size <= MAX_PACKET_BYTES
                or info.compress_size > len(archive_bytes)
            ):
                raise G110GeneratedY1PoseError(
                    "pose archive member is unsafe/noncanonical"
                )
            packet = archive.read(info)
            if len(packet) != info.file_size:
                raise G110GeneratedY1PoseError(
                    "pose archive member length differs"
                )
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise G110GeneratedY1PoseError("pose archive cannot be decoded") from exc
    parse_g110_generated_y1_pose_v1(packet)
    return packet, method


def parse_g110_generated_y1_pose_archive(archive_bytes: bytes) -> bytes:
    packet, method = _read_g110_archive_member(archive_bytes)
    selected_method, selected_archive = _select_outer_zip_method(packet)
    if method is not selected_method or archive_bytes != selected_archive:
        raise G110GeneratedY1PoseError(
            "pose archive is not the deterministic canonical method/layout"
        )
    return packet


__all__ = [
    "MAGIC",
    "POST_G105_REFIT_CHECKPOINT_SCHEMA",
    "POST_G105_REFIT_RUN_SCHEMA",
    "RENDER_ORDER",
    "SOURCE_DOMAIN",
    "V10_FACTOR2_SELECTED_PREIMAGE_SCHEMA",
    "CompiledG110GeneratedY1PoseV1",
    "G110CompleteArchiveWireCandidateV1",
    "G110G112CompileCustodyV1",
    "G110GeneratedY1PoseError",
    "G110OuterZipMethodV1",
    "G111GeneratedY1PoseInitializerCustodyV1",
    "ParsedG110GeneratedY1PoseV1",
    "PostG105PoseRefitV1",
    "build_g110_generated_y1_pose_archive",
    "compile_g110_generated_y1_pose_v1",
    "parse_g110_generated_y1_pose_archive",
    "parse_g110_generated_y1_pose_v1",
    "render_camera_y0",
]
