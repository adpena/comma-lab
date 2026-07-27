# SPDX-License-Identifier: MIT
"""Public-runtime two-layer packet over a generic final-Y1 provider.

The counted object is one semantic Y1 program plus one conditional Y0 stream::

    Y1[p] = semantic_provider.render_scorer_y1(packet, p)
    Y0[p] = round_clip(Y1[p] + upsample(sum_r c[p,r] * s[r] * B[r]))

The semantic provider is selected from the packet itself and currently admits
the committed G103 integer Coordinate-INR and the exact G105 V9 HOSC dual-head
program.  Conditional coefficients are first differences over chronological
pair order and Rice coded.  Only the decoder-effective per-rank scale ``s`` is
stored; the redundant basis-scale/coefficient-scale gauge does not exist.

This module performs real n600 rendering when it binds final Y1.  It does not
fit either stream, claim source closure, claim an evaluator score, or move the
frontier pointer.  Fresh source, same-forward batch-16 target/margin custody,
batch-16 PoseNet custody, and fresh checkpoint lineage are mandatory external
compile inputs and remain outside candidate bytes.

The deterministic one-member ``archive.zip`` is only the counted payload
container.  It is not receiver closure without the separately sealed public
runtime tree, clean-extract double decode, exact output-video proof, and
recursive ``upstream/evaluate.py`` evidence.
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
from enum import IntEnum
from pathlib import Path
from typing import Any, Final

import numpy as np

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
from tac.witness_dsl import taskspace_g105_exact_v9_semantic_root_adapter_v1 as g105_adapter
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    MAGIC as V9_MAGIC,
)
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    VARIANT_ID as V9_VARIANT_ID,
)
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    ExactV9SemanticRootError,
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
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    render_scorer_y1 as render_v9_y1,
)
from tac.witness_dsl.taskspace_pfree_semantic_root_v1 import (
    PAIR_COUNT_N600,
    SCORER_CHANNELS,
    SCORER_H,
    SCORER_W,
    encode_semantic_root_y1_v1,
    parse_semantic_root_y1_v1,
    render_semantic_root_y1_scorer,
)

MAGIC: Final = b"G110TL01"
VERSION: Final = 1
PACKET_MEMBER: Final = "taskspace_two_layer_v1.bin"
PUBLIC_RUNTIME_RELATIVE_ROOT: Final = "submissions/robust_current/g110_two_layer_receiver"
G103_VARIANT_ID: Final = "tac.semantic_root_y1.original_coordinr_film_mlp.v1"
G103_MAGIC: Final = b"SRY1V1\x00\x00"
CONDITIONAL_VARIANT_ID: Final = "tac.semantic_root_y0.conditional_lowrank_rice.v1"
PAIR_BATCH_SIZE: Final = 16
MAX_RANK: Final = 64
MAX_GRID_SIDE: Final = 64
MAX_SEMANTIC_PACKET_BYTES: Final = 2_000_000
MAX_PACKET_BYTES: Final = 2_100_000
MAX_ARCHIVE_BYTES: Final = 2_100_000
COEFFICIENT_CODEC_RICE_DELTA: Final = 0
FINAL_Y1_DOMAIN: Final = b"G110_FINAL_Y1_N600_V1\x00"
UPSTREAM_SOURCE_CLOSURE_SHA256: Final = (
    "e93f6c744fe0025ecc30d1f1cef00617a3f1397b68cadb856817766cfec63279"
)
G46_TARGET_LABELS_SHA256: Final = (
    "6d2ca48ac07323c7fc3a5299023bc291363192e10130eb3bc63d446bb8e65b85"
)
G46_SOURCE_PAIR_CHAIN_SHA256: Final = (
    "5b391fa4a5f651452fdf9a861af3f52abdc58017dcd8bfc0566ebcf86cab3559"
)
POSE_TARGET_CONTRACT_ID: Final = "UPSTREAM_POSENET_SOURCE_TARGET_ORDERED_N600_BATCH16_V1"
POSE_CANDIDATE_CONTRACT_ID: Final = "UPSTREAM_POSENET_FINAL_Y0_Y1_ORDERED_N600_BATCH16_V1"
CONDITIONAL_OPERAND_RECEIPT_SCHEMA: Final = "tac.g110_fresh_conditional_y0_operand_receipt.v1"
CONDITIONAL_PRODUCER_RUN_SCHEMA: Final = "tac.g110_conditional_y0_producer_run.v1"
CONDITIONAL_PRODUCER_CHECKPOINT_SCHEMA: Final = (
    "tac.g110_conditional_y0_producer_checkpoint.v1"
)
G109_PROJECTION_SCHEMA: Final = "tac.taskspace_v9_training_target_binding.v1"
G109_CONSUMER_SCHEMA: Final = "tac.taskspace_v9_training_target_consumer.v1"
G109_PROJECTION_KEY: Final = "__cfg_g109_target_projection_json"
G109_PROJECTION_SHA_KEY: Final = "__cfg_g109_target_projection_sha256"
G105_ADAPTER_SOURCE_SHA256: Final = (
    "549f2cd67d0e2961560faa810166332970a8270cdee681c9d15c748d68ed2aad"
)
G105_ADAPTER_GIT_BLOB_SHA1: Final = "1a354d51e10dd7d38097d74c96bbc2206f93e994"
G111_POSE_PARAM_KEYS: Final = frozenset(
    {"pose_carrier.xi_stored", "pose_carrier.dxi"}
)
G111_POSE_CHECKPOINT_CONTRACT_SCHEMA: Final = (
    "tac.v9_pose_carrier_checkpoint_contract.v2"
)
G111_POSE_Y1_SELECTED_PREIMAGE_SCHEMA: Final = (
    "tac.v10_factor2_selected_preimage.v1"
)

_HEADER: Final = struct.Struct(">8sBBHHHBBHHBBIIII32sI")
_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
_F32_BE: Final = np.dtype(">f4")


class G110TwoLayerError(ValueError):
    """The generic provider, conditional stream, custody, or packet failed."""


class G110OuterZipMethodV1(IntEnum):
    """Legal deterministic counted-archive methods, in tie-break order."""

    STORE = zipfile.ZIP_STORED
    DEFLATE = zipfile.ZIP_DEFLATED


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
        raise G110TwoLayerError("value is not finite canonical ASCII JSON") from exc


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise G110TwoLayerError(f"{name} must be canonical lowercase SHA-256")
    return value


def _resolve_regular_nonsymlink(path: Path, *, name: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise G110TwoLayerError(f"{name} must not be a symlink")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise G110TwoLayerError(f"{name} must be a regular file")
    return resolved


def _immutable_array(
    value: object,
    *,
    dtype: np.dtype[Any] | type[np.generic],
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.dtype(dtype) or raw.shape != shape or not raw.flags.c_contiguous:
        raise G110TwoLayerError(
            f"{name} must be C-contiguous {np.dtype(dtype)} with shape {shape}"
        )
    result = np.array(raw, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _partition_g111_checkpoint_params(
    params: dict[str, np.ndarray],
    scalars: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Separate the G105 semantic tensors from the exact generated-Y1 pose child.

    G105 consumes only the shared trunk plus odd code rows.  A G111 checkpoint
    may additionally contain the two table-mode pose tensors; those are owned
    exclusively by the conditional compiler.  Any partial/extra pose subtree
    fails closed instead of being silently dropped.
    """

    observed_pose = {key for key in params if key.startswith("pose_carrier.")}
    if not observed_pose:
        return dict(params), {}
    if observed_pose != G111_POSE_PARAM_KEYS:
        raise G110TwoLayerError(
            "G111 pose tensor set is partial or contains an unconsumed member"
        )
    if (
        scalars.get("__cfg_pose_carrier_contract_schema")
        != G111_POSE_CHECKPOINT_CONTRACT_SCHEMA
        or int(scalars.get("__cfg_pose_carrier", 0)) != 1
        or scalars.get("__cfg_pose_carrier_source") != "generated_y1"
        or scalars.get("__cfg_pose_carrier_residual_mode") != "table"
        or scalars.get("__cfg_pose_carrier_xi_formula")
        != "xi_stored+residual_scale*dxi"
        or scalars.get("__cfg_pose_carrier_y1_selected_preimage_schema")
        != G111_POSE_Y1_SELECTED_PREIMAGE_SCHEMA
        or tuple(
            int(value)
            for value in np.asarray(
                scalars.get("__cfg_pose_carrier_native_hw", ()),
                dtype=np.int64,
            ).reshape(-1)
        )
        != (874, 1164)
    ):
        raise G110TwoLayerError(
            "G111 pose tensors lack exact generated_y1/table decode custody"
        )
    for key in (
        "__cfg_pose_carrier_residual_scale",
        "__cfg_pose_carrier_s_t",
        "__cfg_pose_carrier_s_r",
        "__cfg_pose_carrier_pitch",
    ):
        try:
            value = float(scalars[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise G110TwoLayerError(
                f"G111 pose checkpoint lacks finite scalar {key}"
            ) from exc
        if not math.isfinite(value):
            raise G110TwoLayerError(
                f"G111 pose checkpoint scalar {key} is non-finite"
            )
    semantic = {
        key: value
        for key, value in params.items()
        if key not in G111_POSE_PARAM_KEYS
    }
    pose = {key: params[key] for key in sorted(G111_POSE_PARAM_KEYS)}
    if set(semantic).intersection(pose) or set(semantic).union(pose) != set(params):
        raise AssertionError("G111 tensor ownership partition is not total/disjoint")
    return semantic, pose


@dataclass(frozen=True, slots=True, init=False)
class G110Batch16SourcePoseCustodyV1:
    """Point-of-use result of reopening G109 and rederiving physical G105."""

    target_margins_sha256: str
    pose_targets_sha256: str
    target_capsule_receipt_sha256: str
    fresh_checkpoint_sha256: str
    semantic_packet_sha256: str
    g105_adapter_source_sha256: str = G105_ADAPTER_SOURCE_SHA256
    upstream_source_closure_sha256: str = UPSTREAM_SOURCE_CLOSURE_SHA256
    target_labels_sha256: str = G46_TARGET_LABELS_SHA256
    source_pair_chain_sha256: str = G46_SOURCE_PAIR_CHAIN_SHA256
    target_capsule_schema: str = TARGET_CAPSULE_SCHEMA
    pose_target_contract_id: str = POSE_TARGET_CONTRACT_ID
    pose_candidate_contract_id: str = POSE_CANDIDATE_CONTRACT_ID
    scorer_batch_size: int = PAIR_BATCH_SIZE
    pose_batch_size: int = PAIR_BATCH_SIZE
    live_verdict_batch_size: int = PAIR_BATCH_SIZE
    margins_from_same_batch16_forward: bool = True
    fresh_own_lineage: bool = True

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "G110 custody requires content-reading from_verified_v9_producer(); "
            "hash-only assertions are forbidden"
        )

    @classmethod
    def from_physical_v9_producer(
        cls,
        *,
        target_capsule_receipt: Path,
        expected_target_capsule_receipt_sha256: str,
        fresh_g105_checkpoint: Path,
        semantic_packet: bytes,
    ) -> G110Batch16SourcePoseCustodyV1:
        """Reopen G109 and recompile the exact packet from the physical NPZ."""

        if type(semantic_packet) is not bytes:
            raise G110TwoLayerError("custody requires exact semantic packet bytes")
        adapter_path = _resolve_regular_nonsymlink(
            Path(g105_adapter.__file__),
            name="committed G105 adapter source",
        )
        if sha256_file(adapter_path) != G105_ADAPTER_SOURCE_SHA256:
            raise G110TwoLayerError(
                "physical G105 adapter source differs from the committed ABI dependency"
            )
        expected_receipt_sha = _require_sha256(
            expected_target_capsule_receipt_sha256,
            name="target capsule receipt file",
        )
        try:
            loader = V9TrainingTargetCapsuleLoaderV1.open(
                target_capsule_receipt,
                expected_sha256=expected_receipt_sha,
            )
        except (OSError, ValueError, V9TrainingTargetCapsuleError) as exc:
            raise G110TwoLayerError("G109 target capsule did not strictly reopen") from exc
        if (
            loader.receipt.get("schema") != TARGET_CAPSULE_SCHEMA
            or loader.pair_count != PRODUCTION_PAIR_COUNT
            or loader.batch_pairs != PRODUCTION_BATCH_PAIRS
            or loader.preflight.get("test_only_small_fixture") is not False
        ):
            raise G110TwoLayerError("G109 target capsule is not exact n600 batch-16")
        raw = loader.receipt.get("raw_arrays")
        runtime = loader.preflight.get("runtime_custody")
        if (
            not isinstance(raw, dict)
            or not isinstance(runtime, dict)
            or set(raw) != {"labels", "margins", "poses"}
        ):
            raise G110TwoLayerError("G105/G109 custody sections are incomplete")
        checkpoint_path = _resolve_regular_nonsymlink(
            fresh_g105_checkpoint,
            name="fresh G105 checkpoint",
        )
        try:
            params, scalars, exact_checkpoint_sha, _checkpoint_bytes = (
                _g105_checkpoint_runtime_state(checkpoint_path)
            )
            semantic_params, _pose_params = _partition_g111_checkpoint_params(
                params,
                scalars,
            )
            config = _g105_checkpoint_config(semantic_params, scalars)
            rederived_program = compile_v9_from_state(
                config=config,
                params={
                    key: value
                    for key, value in semantic_params.items()
                    if key != "code"
                },
                interleaved_code=semantic_params["code"],
            )
            rederived_packet = encode_v9_packet(rederived_program)
        except (OSError, ValueError, ExactV9SemanticRootError) as exc:
            raise G110TwoLayerError(
                "physical G105 checkpoint did not recompile through the exact adapter"
            ) from exc
        if rederived_packet != semantic_packet:
            raise G110TwoLayerError(
                "semantic packet is not the exact recompilation of the physical G105 checkpoint"
            )
        exact_margin_sha = _require_sha256(
            raw["margins"].get("sha256"),
            name="G109 margins",
        )
        exact_pose_sha = _require_sha256(
            raw["poses"].get("sha256"),
            name="G109 poses",
        )
        expected_projection = {
            "schema": G109_PROJECTION_SCHEMA,
            "aggregate_schema": TARGET_CAPSULE_SCHEMA,
            "aggregate_receipt": {
                "path": str(loader.receipt_path),
                "bytes": loader.receipt_path.stat().st_size,
                "sha256": expected_receipt_sha,
            },
            "aggregate_receipt_sha256": loader.receipt["aggregate_receipt_sha256"],
            "preflight_sha256": loader.receipt["preflight_sha256"],
            "batch_digest_chain_sha256": loader.receipt["batch_digest_chain_sha256"],
            "g46_receipt_sha256": loader.receipt["g46_custody"]["receipt_sha256"],
            "source_pair_chain_sha256": G46_SOURCE_PAIR_CHAIN_SHA256,
            "source_video_sha256": loader.receipt["source_custody"]["source_video"]["sha256"],
            "segnet_weights_sha256": loader.receipt["scorer_custody"]["segnet_weights"]["sha256"],
            "posenet_weights_sha256": loader.receipt["scorer_custody"]["posenet_weights"]["sha256"],
            "arrays": {
                "seg_labels_u8": dict(raw["labels"]),
                "seg_top1_minus_top2_margin_f32": dict(raw["margins"]),
                "source_pose6_f32": dict(raw["poses"]),
            },
            "pair_count": PRODUCTION_PAIR_COUNT,
            "scorer_pair_batch_size": PRODUCTION_BATCH_PAIRS,
            "same_forward_seg_margin_pose": True,
            "encoder_only": True,
            "candidate_payload_allowed": False,
        }
        projection_sha = _sha256(_canonical_json(expected_projection))
        projection_text = scalars.get(G109_PROJECTION_KEY)
        if type(projection_text) is not str:
            raise G110TwoLayerError("G105 checkpoint lacks the canonical G109 projection")
        try:
            observed_projection = json.loads(projection_text)
        except json.JSONDecodeError as exc:
            raise G110TwoLayerError("G105 checkpoint G109 projection is not JSON") from exc
        active_target_sha = _require_sha256(
            scalars.get("__cfg_target_authority_sha256"),
            name="active target authority",
        )
        consumer_binding_sha = _sha256(
            _canonical_json(
                {
                    "schema": G109_CONSUMER_SCHEMA,
                    "target_projection_sha256": projection_sha,
                    "active_target_authority_sha256": active_target_sha,
                    "live_verdict_batch_size": PAIR_BATCH_SIZE,
                }
            )
        )
        target_evidence_sha = _sha256(
            _canonical_json(
                {
                    "schema": "tac.taskspace_v9_training_target_evidence.v1",
                    "target_projection_sha256": projection_sha,
                    "batch_digest_chain_sha256": loader.receipt[
                        "batch_digest_chain_sha256"
                    ],
                    "same_forward_seg_margin_pose": True,
                }
            )
        )
        checkpoint_target_binding = {
            G109_PROJECTION_SHA_KEY: projection_sha,
            "__cfg_g46_target_labels_sha256": raw["labels"].get("sha256"),
            "__cfg_g46_target_margins_sha256": exact_margin_sha,
            "__cfg_g46_source_pair_chain_sha256": G46_SOURCE_PAIR_CHAIN_SHA256,
            "__cfg_g46_margin_aggregate_schema": TARGET_CAPSULE_SCHEMA,
            "__cfg_g46_margin_aggregate_sha256": loader.receipt[
                "aggregate_receipt_sha256"
            ],
            "__cfg_g46_target_consumer_binding_sha256": consumer_binding_sha,
            "__cfg_g46_target_evidence_sha256": target_evidence_sha,
        }
        if (
            observed_projection != expected_projection
            or scalars.get(G109_PROJECTION_SHA_KEY) != projection_sha
            or any(scalars.get(key) != expected for key, expected in checkpoint_target_binding.items())
            or int(scalars.get("__cfg_g46_target_scorer_batch_size", 0))
            != PAIR_BATCH_SIZE
            or int(scalars.get("__cfg_g46_margin_same_forward", 0)) != 1
            or int(scalars.get("__cfg_verdict_batch", 0)) != PAIR_BATCH_SIZE
            or scalars.get("__cfg_upstream_snapshot_sha256")
            != UPSTREAM_SOURCE_CLOSURE_SHA256
            or runtime.get("upstream_closure_sha256")
            != UPSTREAM_SOURCE_CLOSURE_SHA256
            or sha256_file(target_capsule_receipt) != expected_receipt_sha
            or encode_v9_packet(parse_v9_packet(semantic_packet)) != semantic_packet
        ):
            raise G110TwoLayerError(
                "physical G105 checkpoint and G109 batch-16 target capsule disagree"
            )
        instance = object.__new__(cls)
        values = {
            "target_margins_sha256": exact_margin_sha,
            "pose_targets_sha256": exact_pose_sha,
            "target_capsule_receipt_sha256": expected_receipt_sha,
            "fresh_checkpoint_sha256": exact_checkpoint_sha,
            "semantic_packet_sha256": _sha256(semantic_packet),
            "g105_adapter_source_sha256": G105_ADAPTER_SOURCE_SHA256,
            "upstream_source_closure_sha256": UPSTREAM_SOURCE_CLOSURE_SHA256,
            "target_labels_sha256": G46_TARGET_LABELS_SHA256,
            "source_pair_chain_sha256": G46_SOURCE_PAIR_CHAIN_SHA256,
            "target_capsule_schema": TARGET_CAPSULE_SCHEMA,
            "pose_target_contract_id": POSE_TARGET_CONTRACT_ID,
            "pose_candidate_contract_id": POSE_CANDIDATE_CONTRACT_ID,
            "scorer_batch_size": PAIR_BATCH_SIZE,
            "pose_batch_size": PAIR_BATCH_SIZE,
            "live_verdict_batch_size": PAIR_BATCH_SIZE,
            "margins_from_same_batch16_forward": True,
            "fresh_own_lineage": True,
        }
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        for name, value in (
            ("target margins", self.target_margins_sha256),
            ("pose targets", self.pose_targets_sha256),
            ("target capsule receipt", self.target_capsule_receipt_sha256),
            ("fresh checkpoint", self.fresh_checkpoint_sha256),
            ("semantic packet", self.semantic_packet_sha256),
            ("G105 adapter source", self.g105_adapter_source_sha256),
        ):
            _require_sha256(value, name=name)
        if (
            self.upstream_source_closure_sha256 != UPSTREAM_SOURCE_CLOSURE_SHA256
            or self.target_labels_sha256 != G46_TARGET_LABELS_SHA256
            or self.source_pair_chain_sha256 != G46_SOURCE_PAIR_CHAIN_SHA256
            or self.target_capsule_schema != TARGET_CAPSULE_SCHEMA
            or self.pose_target_contract_id != POSE_TARGET_CONTRACT_ID
            or self.pose_candidate_contract_id != POSE_CANDIDATE_CONTRACT_ID
            or self.scorer_batch_size != PAIR_BATCH_SIZE
            or self.pose_batch_size != PAIR_BATCH_SIZE
            or self.live_verdict_batch_size != PAIR_BATCH_SIZE
            or self.margins_from_same_batch16_forward is not True
            or self.fresh_own_lineage is not True
            or self.g105_adapter_source_sha256 != G105_ADAPTER_SOURCE_SHA256
        ):
            raise G110TwoLayerError(
                "compile custody is not fresh own-lineage batch-16 source/margin/pose authority"
            )

    @property
    def identity_sha256(self) -> str:
        values = (
            self.upstream_source_closure_sha256,
            self.target_labels_sha256,
            self.target_margins_sha256,
            self.source_pair_chain_sha256,
            self.pose_targets_sha256,
            self.target_capsule_receipt_sha256,
            self.fresh_checkpoint_sha256,
            self.semantic_packet_sha256,
            self.g105_adapter_source_sha256,
            self.target_capsule_schema,
            self.pose_target_contract_id,
            self.pose_candidate_contract_id,
            str(self.scorer_batch_size),
            str(self.pose_batch_size),
            str(self.live_verdict_batch_size),
        )
        return _sha256("\x00".join(values).encode("ascii"))


@dataclass(frozen=True, slots=True)
class OpenedFinalY1ProviderV1:
    """One typed semantic provider behind the common receiver ABI."""

    variant_id: str
    packet: bytes = field(repr=False)
    parsed: object = field(repr=False)

    def render_scorer_y1(self, pair_id: int) -> np.ndarray:
        if type(pair_id) is not int or not 0 <= pair_id < PAIR_COUNT_N600:
            raise G110TwoLayerError("pair_id must be an exact integer in [0,599]")
        if self.variant_id == G103_VARIANT_ID:
            frame = render_semantic_root_y1_scorer(self.parsed, pair_id)  # type: ignore[arg-type]
        elif self.variant_id == V9_VARIANT_ID:
            frame = render_v9_y1(self.parsed, pair_id)  # type: ignore[arg-type]
        else:
            raise AssertionError("opened provider variant escaped the closed dispatch")
        raw = np.asarray(frame)
        if raw.dtype != np.uint8 or raw.shape != (
            SCORER_H,
            SCORER_W,
            SCORER_CHANNELS,
        ):
            raise G110TwoLayerError("semantic provider violated uint8 scorer-Y1 ABI")
        return np.ascontiguousarray(raw)


def open_final_y1_provider(packet: bytes) -> OpenedFinalY1ProviderV1:
    """Dispatch one exact semantic packet without cross-casting its model."""

    if type(packet) is not bytes or not 0 < len(packet) <= MAX_SEMANTIC_PACKET_BYTES:
        raise G110TwoLayerError("semantic packet must be bounded exact bytes")
    if packet.startswith(G103_MAGIC):
        parsed = parse_semantic_root_y1_v1(packet)
        if encode_semantic_root_y1_v1(parsed) != packet:
            raise G110TwoLayerError("G103 packet changed under exact re-emission")
        return OpenedFinalY1ProviderV1(G103_VARIANT_ID, packet, parsed)
    if packet.startswith(V9_MAGIC):
        parsed = parse_v9_packet(packet)
        if encode_v9_packet(parsed) != packet:
            raise G110TwoLayerError("V9 packet changed under exact re-emission")
        return OpenedFinalY1ProviderV1(V9_VARIANT_ID, packet, parsed)
    raise G110TwoLayerError("semantic packet matches no admitted final-Y1 provider")


def _population_digest(provider: OpenedFinalY1ProviderV1) -> bytes:
    digest = hashlib.sha256()
    for pair_id in range(PAIR_COUNT_N600):
        digest.update(struct.pack(">H", pair_id))
        digest.update(memoryview(provider.render_scorer_y1(pair_id)).cast("B"))
    return digest.digest()


def final_y1_binding_sha256(provider: OpenedFinalY1ProviderV1) -> str:
    """Bind packet identity to the actual ordered rendered n600 population."""

    if type(provider) is not OpenedFinalY1ProviderV1:
        raise G110TwoLayerError("final-Y1 binding requires an opened provider")
    return _sha256(
        FINAL_Y1_DOMAIN
        + hashlib.sha256(provider.packet).digest()
        + _population_digest(provider)
    )


class _BitWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.byte = 0
        self.used = 0

    def bit(self, value: int) -> None:
        self.byte = (self.byte << 1) | (value & 1)
        self.used += 1
        if self.used == 8:
            self.data.append(self.byte)
            self.byte = 0
            self.used = 0

    def finish(self) -> bytes:
        if self.used:
            self.data.append(self.byte << (8 - self.used))
        return bytes(self.data)


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def bit(self) -> int:
        if self.offset >= len(self.payload) * 8:
            raise G110TwoLayerError("conditional Rice stream is truncated")
        value = (self.payload[self.offset // 8] >> (7 - self.offset % 8)) & 1
        self.offset += 1
        return value

    def require_zero_padding(self) -> None:
        while self.offset < len(self.payload) * 8:
            if self.bit():
                raise G110TwoLayerError("conditional Rice stream has nonzero padding")


def _coefficient_unsigned_deltas(coefficients: np.ndarray) -> tuple[int, ...]:
    previous = np.zeros(coefficients.shape[1], dtype=np.int64)
    values: list[int] = []
    for row in coefficients.astype(np.int64):
        delta = row - previous
        previous = row
        values.extend(
            int(2 * value if value >= 0 else -2 * value - 1)
            for value in delta
        )
    return tuple(values)


def _optimal_rice_k(unsigned: tuple[int, ...]) -> int:
    return min(
        range(16),
        key=lambda k: (sum((value >> k) + 1 + k for value in unsigned), k),
    )


def _rice_encode(coefficients: np.ndarray) -> tuple[int, bytes]:
    if coefficients.shape[1] == 0:
        return 0, b""
    unsigned = _coefficient_unsigned_deltas(coefficients)
    rice_k = _optimal_rice_k(unsigned)
    writer = _BitWriter()
    mask = (1 << rice_k) - 1
    for value in unsigned:
        quotient = value >> rice_k
        for _ in range(quotient):
            writer.bit(1)
        writer.bit(0)
        remainder = value & mask
        for shift in range(rice_k - 1, -1, -1):
            writer.bit((remainder >> shift) & 1)
    return rice_k, writer.finish()


def _rice_decode(payload: bytes, *, rice_k: int, rank: int) -> np.ndarray:
    if rank == 0:
        if payload or rice_k != 0:
            raise G110TwoLayerError("rank-zero conditional stream must be empty Rice-0")
        return np.empty((PAIR_COUNT_N600, 0), dtype=np.int16)
    if not 0 <= rice_k <= 15 or not payload:
        raise G110TwoLayerError("conditional Rice header is invalid")
    reader = _BitReader(payload)
    result = np.empty((PAIR_COUNT_N600, rank), dtype=np.int16)
    previous = np.zeros(rank, dtype=np.int64)
    for pair_id in range(PAIR_COUNT_N600):
        for column in range(rank):
            quotient = 0
            while reader.bit():
                quotient += 1
                if quotient > 262_143:
                    raise G110TwoLayerError("conditional Rice quotient exceeds decoder bound")
            remainder = 0
            for _ in range(rice_k):
                remainder = (remainder << 1) | reader.bit()
            unsigned = (quotient << rice_k) | remainder
            delta = unsigned // 2 if not unsigned & 1 else -(unsigned // 2) - 1
            value = int(previous[column]) + delta
            if not -32_768 <= value <= 32_767:
                raise G110TwoLayerError("conditional temporal delta leaves int16 range")
            result[pair_id, column] = value
            previous[column] = value
    reader.require_zero_padding()
    canonical_k, canonical = _rice_encode(result)
    if canonical_k != rice_k or canonical != payload:
        raise G110TwoLayerError("conditional Rice stream is not canonical/minimal-k")
    return np.ascontiguousarray(result)


def _validate_conditional(
    basis_q: object,
    combined_scales: object,
    coefficients_q: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis_raw = np.asarray(basis_q)
    if basis_raw.dtype != np.int8 or basis_raw.ndim != 4 or basis_raw.shape[-1] != SCORER_CHANNELS:
        raise G110TwoLayerError("basis_q must be int8[rank,grid_h,grid_w,3]")
    rank, grid_h, grid_w, _ = basis_raw.shape
    if not 0 <= rank <= MAX_RANK:
        raise G110TwoLayerError("conditional rank is outside [0,64]")
    if rank == 0:
        if (grid_h, grid_w) != (0, 0):
            raise G110TwoLayerError("rank-zero conditional basis must have a 0x0 grid")
    elif not 1 <= grid_h <= MAX_GRID_SIDE or not 1 <= grid_w <= MAX_GRID_SIDE:
        raise G110TwoLayerError("conditional grid side is outside [1,64]")
    basis = _immutable_array(
        basis_raw,
        dtype=np.int8,
        shape=(rank, grid_h, grid_w, SCORER_CHANNELS),
        name="basis_q",
    )
    scales = _immutable_array(
        combined_scales,
        dtype=np.float32,
        shape=(rank,),
        name="combined_scales",
    )
    coefficients = _immutable_array(
        coefficients_q,
        dtype=np.int16,
        shape=(PAIR_COUNT_N600, rank),
        name="coefficients_q",
    )
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise G110TwoLayerError("combined scales must be finite positive float32")
    if rank:
        dead_rank = np.all(basis == 0, axis=(1, 2, 3)) | np.all(coefficients == 0, axis=0)
        if np.any(dead_rank):
            raise G110TwoLayerError("unused conditional ranks must be removed before encoding")
        amplitude_bounds: list[float] = []
        for rank_id in range(rank):
            flat_basis = basis[rank_id].reshape(-1)
            first_nonzero = flat_basis[np.flatnonzero(flat_basis)[0]]
            if int(first_nonzero) < 0:
                raise G110TwoLayerError(
                    "conditional rank sign gauge requires first nonzero basis value positive"
                )
            coefficient_gcd = 0
            for value in coefficients[:, rank_id]:
                coefficient_gcd = math.gcd(coefficient_gcd, abs(int(value)))
            if coefficient_gcd != 1:
                raise G110TwoLayerError(
                    "conditional coefficient/scale gauge requires per-rank coefficient gcd 1"
                )
            amplitude_bounds.append(
                float(np.max(np.abs(flat_basis.astype(np.int16))))
                * float(np.max(np.abs(coefficients[:, rank_id].astype(np.int32))))
                * float(scales[rank_id])
            )
        bounds = np.asarray(amplitude_bounds, dtype=np.float64)
        if (
            not np.all(np.isfinite(bounds))
            or np.any(bounds < 0.5)
            or float(np.sum(bounds, dtype=np.float64))
            > float(np.finfo(np.float32).max) / 4.0
        ):
            raise G110TwoLayerError(
                "conditional rank is decoder-dead or can overflow float32 intermediates"
            )
    return basis, scales, coefficients


def _reopen_bound_file(binding: object, *, name: str) -> Path:
    if type(binding) is not dict or set(binding) != {"path", "bytes", "sha256"}:
        raise G110TwoLayerError(f"{name} file binding differs")
    path = _resolve_regular_nonsymlink(Path(str(binding["path"])), name=name)
    if (
        binding["path"] != str(path)
        or type(binding["bytes"]) is not int
        or binding["bytes"] != path.stat().st_size
        or binding["sha256"] != sha256_file(path)
    ):
        raise G110TwoLayerError(f"{name} physical file identity differs")
    _require_sha256(binding["sha256"], name=name)
    return path


def _npz_scalar(value: np.ndarray, *, name: str) -> object:
    raw = np.asarray(value)
    if raw.size != 1:
        raise G110TwoLayerError(f"{name} must be a scalar checkpoint member")
    return raw.reshape(()).item()


def _verify_conditional_producer(
    *,
    checkpoint_binding: object,
    run_binding: object,
    custody: G110Batch16SourcePoseCustodyV1,
    semantic_packet: bytes,
    basis: np.ndarray,
    scales: np.ndarray,
    coefficients: np.ndarray,
) -> None:
    """Reopen the real operand NPZ and its sealed resumable producer run."""

    checkpoint_path = _reopen_bound_file(
        checkpoint_binding,
        name="conditional producer checkpoint",
    )
    run_path = _reopen_bound_file(
        run_binding,
        name="conditional producer run receipt",
    )
    try:
        run = json.loads(run_path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G110TwoLayerError("conditional producer run receipt is not JSON") from exc
    run_keys = {
        "schema",
        "run_id",
        "seed",
        "source_git_sha",
        "command",
        "fresh_own_lineage",
        "joint_pose_conditioned",
        "resumable_from_disk",
        "stage_checkpoints_preserved",
        "research_only",
        "candidate_claim",
        "score_claim",
        "pointer_moved",
        "semantic_packet_sha256",
        "target_capsule_receipt_sha256",
        "pose_targets_sha256",
        "producer_checkpoint_sha256",
        "producer_checkpoint_bytes",
        "stage_checkpoints",
        "receipt_sha256",
    }
    if type(run) is not dict or set(run) != run_keys:
        raise G110TwoLayerError("conditional producer run receipt key set differs")
    run_sha = _require_sha256(run["receipt_sha256"], name="conditional producer run")
    if _sha256(_canonical_json({key: value for key, value in run.items() if key != "receipt_sha256"})) != run_sha:
        raise G110TwoLayerError("conditional producer run receipt self-hash differs")
    source_git_sha = run["source_git_sha"]
    command = run["command"]
    stages = run["stage_checkpoints"]
    if (
        run["schema"] != CONDITIONAL_PRODUCER_RUN_SCHEMA
        or type(run["run_id"]) is not str
        or not run["run_id"]
        or type(run["seed"]) is not int
        or type(source_git_sha) is not str
        or len(source_git_sha) != 40
        or any(character not in "0123456789abcdef" for character in source_git_sha)
        or type(command) is not list
        or not command
        or any(type(token) is not str or not token for token in command)
        or run["fresh_own_lineage"] is not True
        or run["joint_pose_conditioned"] is not True
        or run["resumable_from_disk"] is not True
        or run["stage_checkpoints_preserved"] is not True
        or run["research_only"] is not True
        or run["candidate_claim"] is not False
        or run["score_claim"] is not False
        or run["pointer_moved"] is not False
        or run["semantic_packet_sha256"] != _sha256(semantic_packet)
        or run["target_capsule_receipt_sha256"]
        != custody.target_capsule_receipt_sha256
        or run["pose_targets_sha256"] != custody.pose_targets_sha256
        or run["producer_checkpoint_sha256"] != checkpoint_binding["sha256"]
        or run["producer_checkpoint_bytes"] != checkpoint_binding["bytes"]
        or type(stages) is not list
        or not stages
    ):
        raise G110TwoLayerError(
            "conditional producer run is not physical fresh joint-pose resumable lineage"
        )
    reopened_stages = [
        _reopen_bound_file(binding, name=f"conditional producer stage {index}")
        for index, binding in enumerate(stages)
    ]
    if (
        stages[-1] != checkpoint_binding
        or reopened_stages[-1] != checkpoint_path
    ):
        raise G110TwoLayerError("conditional producer final stage is not the operand checkpoint")
    try:
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "schema",
                "run_id",
                "seed",
                "fresh_own_lineage",
                "joint_pose_conditioned",
                "semantic_packet_sha256",
                "target_capsule_receipt_sha256",
                "pose_targets_sha256",
                "basis_q",
                "combined_scales",
                "coefficients_q",
            }:
                raise G110TwoLayerError(
                    "conditional producer checkpoint member set differs"
                )
            checkpoint_values = {
                name: np.asarray(archive[name]).copy() for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise G110TwoLayerError("conditional producer checkpoint is not a strict NPZ") from exc
    if (
        _npz_scalar(checkpoint_values["schema"], name="checkpoint schema")
        != CONDITIONAL_PRODUCER_CHECKPOINT_SCHEMA
        or _npz_scalar(checkpoint_values["run_id"], name="checkpoint run_id")
        != run["run_id"]
        or _npz_scalar(checkpoint_values["seed"], name="checkpoint seed")
        != run["seed"]
        or _npz_scalar(
            checkpoint_values["fresh_own_lineage"],
            name="checkpoint fresh_own_lineage",
        )
        != 1
        or _npz_scalar(
            checkpoint_values["joint_pose_conditioned"],
            name="checkpoint joint_pose_conditioned",
        )
        != 1
        or _npz_scalar(
            checkpoint_values["semantic_packet_sha256"],
            name="checkpoint semantic packet",
        )
        != _sha256(semantic_packet)
        or _npz_scalar(
            checkpoint_values["target_capsule_receipt_sha256"],
            name="checkpoint target capsule",
        )
        != custody.target_capsule_receipt_sha256
        or _npz_scalar(
            checkpoint_values["pose_targets_sha256"],
            name="checkpoint pose targets",
        )
        != custody.pose_targets_sha256
        or not np.array_equal(checkpoint_values["basis_q"], basis)
        or not np.array_equal(checkpoint_values["combined_scales"], scales)
        or not np.array_equal(checkpoint_values["coefficients_q"], coefficients)
    ):
        raise G110TwoLayerError(
            "conditional operands are not rederived from the physical producer checkpoint"
        )


def _verify_conditional_operand_receipt(
    *,
    receipt_path: Path,
    expected_receipt_file_sha256: str,
    custody: G110Batch16SourcePoseCustodyV1,
    semantic_packet: bytes,
    basis: np.ndarray,
    scales: np.ndarray,
    coefficients: np.ndarray,
) -> str:
    """Content-read the missing producer boundary instead of trusting labels."""

    expected_file_sha = _require_sha256(
        expected_receipt_file_sha256,
        name="conditional operand receipt file",
    )
    path = _resolve_regular_nonsymlink(
        receipt_path,
        name="conditional operand receipt",
    )
    if sha256_file(path) != expected_file_sha:
        raise G110TwoLayerError("conditional operand receipt file identity differs")
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G110TwoLayerError("conditional operand receipt is not readable JSON") from exc
    expected_keys = {
        "schema",
        "fresh_own_lineage",
        "research_only",
        "candidate_claim",
        "score_claim",
        "conditional_owner",
        "joint_pose_conditioned",
        "pair_count",
        "batch_pairs",
        "semantic_packet_sha256",
        "target_capsule_receipt_sha256",
        "pose_targets_sha256",
        "basis_q",
        "combined_scales",
        "coefficients_q",
        "producer_checkpoint",
        "producer_run_receipt",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise G110TwoLayerError("conditional operand receipt key set differs")
    receipt_sha = _require_sha256(value["receipt_sha256"], name="conditional operand receipt")
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if _sha256(_canonical_json(body)) != receipt_sha:
        raise G110TwoLayerError("conditional operand receipt self-hash differs")
    expected_arrays = {
        "basis_q": {
            "dtype": "int8",
            "shape": list(basis.shape),
            "sha256": _sha256(memoryview(np.ascontiguousarray(basis)).cast("B")),
        },
        "combined_scales": {
            "dtype": "float32_be",
            "shape": list(scales.shape),
            "sha256": _sha256(np.asarray(scales, dtype=_F32_BE).tobytes(order="C")),
        },
        "coefficients_q": {
            "dtype": "int16_be",
            "shape": list(coefficients.shape),
            "sha256": _sha256(np.asarray(coefficients, dtype=">i2").tobytes(order="C")),
        },
    }
    if (
        value["schema"] != CONDITIONAL_OPERAND_RECEIPT_SCHEMA
        or value["fresh_own_lineage"] is not True
        or value["research_only"] is not True
        or value["candidate_claim"] is not False
        or value["score_claim"] is not False
        or value["conditional_owner"] != CONDITIONAL_VARIANT_ID
        or value["joint_pose_conditioned"] is not True
        or value["pair_count"] != PAIR_COUNT_N600
        or value["batch_pairs"] != PAIR_BATCH_SIZE
        or value["semantic_packet_sha256"] != _sha256(semantic_packet)
        or value["target_capsule_receipt_sha256"]
        != custody.target_capsule_receipt_sha256
        or value["pose_targets_sha256"] != custody.pose_targets_sha256
        or any(value[name] != expected for name, expected in expected_arrays.items())
    ):
        raise G110TwoLayerError(
            "conditional operands are not content-bound fresh joint-pose batch-16 lineage"
        )
    _verify_conditional_producer(
        checkpoint_binding=value["producer_checkpoint"],
        run_binding=value["producer_run_receipt"],
        custody=custody,
        semantic_packet=semantic_packet,
        basis=basis,
        scales=scales,
        coefficients=coefficients,
    )
    return receipt_sha


@dataclass(frozen=True, slots=True, eq=False)
class ParsedG110TwoLayerV1:
    semantic_packet: bytes = field(repr=False)
    semantic_variant_id: str
    final_y1_binding_sha256: str
    basis_q: np.ndarray = field(repr=False)
    combined_scales: np.ndarray = field(repr=False)
    coefficients_q: np.ndarray = field(repr=False)
    rice_k: int
    packet: bytes = field(repr=False)

    def render_scorer_pair(
        self,
        provider: OpenedFinalY1ProviderV1,
        pair_id: int,
    ) -> np.ndarray:
        if provider.packet != self.semantic_packet:
            raise G110TwoLayerError("provider packet differs from parsed packet custody")
        y1 = provider.render_scorer_y1(pair_id)
        y0 = render_conditional_y0(
            y1,
            pair_id=pair_id,
            basis_q=self.basis_q,
            combined_scales=self.combined_scales,
            coefficients_q=self.coefficients_q,
        )
        return np.ascontiguousarray(np.stack((y0, y1), axis=0))


def _bilinear_resize(
    image: np.ndarray,
    *,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    input_height, input_width, _ = image.shape
    ys = (np.arange(output_height, dtype=np.float32) + np.float32(0.5)) * np.float32(
        input_height / output_height
    ) - np.float32(0.5)
    xs = (np.arange(output_width, dtype=np.float32) + np.float32(0.5)) * np.float32(
        input_width / output_width
    ) - np.float32(0.5)
    ys = np.clip(ys, np.float32(0.0), np.float32(input_height - 1))
    xs = np.clip(xs, np.float32(0.0), np.float32(input_width - 1))
    y0 = np.floor(ys).astype(np.intp)
    x0 = np.floor(xs).astype(np.intp)
    y1 = np.minimum(y0 + 1, input_height - 1)
    x1 = np.minimum(x0 + 1, input_width - 1)
    wy = ys - y0.astype(np.float32)
    wx = xs - x0.astype(np.float32)
    vertical = image[y0, :, :] * (np.float32(1.0) - wy[:, None, None])
    vertical += image[y1, :, :] * wy[:, None, None]
    return (
        vertical[:, x0, :] * (np.float32(1.0) - wx[None, :, None])
        + vertical[:, x1, :] * wx[None, :, None]
    )


def render_conditional_y0(
    scorer_y1: np.ndarray,
    *,
    pair_id: int,
    basis_q: np.ndarray,
    combined_scales: np.ndarray,
    coefficients_q: np.ndarray,
) -> np.ndarray:
    """Render Y0 from final Y1; legal zero-residual rows preserve Y1 exactly."""

    if type(pair_id) is not int or not 0 <= pair_id < PAIR_COUNT_N600:
        raise G110TwoLayerError("pair_id must be an exact integer in [0,599]")
    y1 = np.asarray(scorer_y1)
    if y1.dtype != np.uint8 or y1.shape != (SCORER_H, SCORER_W, SCORER_CHANNELS):
        raise G110TwoLayerError("conditional Y0 requires final uint8 scorer Y1")
    if basis_q.shape[0] == 0 or not np.any(coefficients_q[pair_id]):
        return np.ascontiguousarray(y1)
    with np.errstate(over="ignore", invalid="ignore"):
        weights = coefficients_q[pair_id].astype(np.float32) * combined_scales
        grid = np.einsum(
            "r,rhwc->hwc",
            weights,
            basis_q.astype(np.float32),
            optimize=True,
            dtype=np.float32,
        )
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(grid)):
        raise G110TwoLayerError("conditional low-rank synthesis overflowed")
    residual = _bilinear_resize(
        np.ascontiguousarray(grid, dtype=np.float32),
        output_height=SCORER_H,
        output_width=SCORER_W,
    )
    if not np.all(np.isfinite(residual)):
        raise G110TwoLayerError("conditional bilinear synthesis is non-finite")
    summed = y1.astype(np.float32) + residual
    if not np.all(np.isfinite(summed)):
        raise G110TwoLayerError("conditional Y0 accumulation is non-finite")
    return np.ascontiguousarray(
        np.clip(np.rint(summed), 0, 255).astype(np.uint8)
    )


def _encode_packet(
    *,
    semantic_packet: bytes,
    final_y1_binding: str,
    basis: np.ndarray,
    scales: np.ndarray,
    coefficients: np.ndarray,
) -> bytes:
    rank, grid_h, grid_w, _ = basis.shape
    rice_k, coefficient_stream = _rice_encode(coefficients)
    basis_bytes = basis.tobytes(order="C")
    scale_bytes = np.asarray(scales, dtype=_F32_BE).tobytes(order="C")
    body = semantic_packet + basis_bytes + scale_bytes + coefficient_stream
    packet = _HEADER.pack(
        MAGIC,
        VERSION,
        0,
        PAIR_COUNT_N600,
        SCORER_H,
        SCORER_W,
        SCORER_CHANNELS,
        rank,
        grid_h,
        grid_w,
        COEFFICIENT_CODEC_RICE_DELTA,
        rice_k,
        len(semantic_packet),
        len(basis_bytes),
        len(scale_bytes),
        len(coefficient_stream),
        bytes.fromhex(final_y1_binding),
        zlib.crc32(body) & 0xFFFFFFFF,
    ) + body
    if len(packet) > MAX_PACKET_BYTES:
        raise G110TwoLayerError("two-layer packet exceeds the bounded counted ABI")
    return packet


def build_g110_rank_zero_semantic_floor_packet(
    semantic_packet: bytes,
) -> bytes:
    """Wrap one exact semantic provider in the receiver-valid ``Y0 == Y1`` floor."""

    provider = open_final_y1_provider(semantic_packet)
    packet = _encode_packet(
        semantic_packet=provider.packet,
        final_y1_binding=final_y1_binding_sha256(provider),
        basis=np.empty((0, 0, 0, SCORER_CHANNELS), dtype=np.int8),
        scales=np.empty((0,), dtype=np.float32),
        coefficients=np.empty((PAIR_COUNT_N600, 0), dtype=np.int16),
    )
    parsed = parse_g110_two_layer_v1(packet)
    if (
        parsed.semantic_packet != provider.packet
        or parsed.basis_q.shape != (0, 0, 0, SCORER_CHANNELS)
        or parsed.combined_scales.shape != (0,)
        or parsed.coefficients_q.shape != (PAIR_COUNT_N600, 0)
    ):
        raise AssertionError("rank-zero semantic floor changed under parse-back")
    return packet


def render_g110_rank_zero_scorer_pair(
    packet: bytes,
    pair_id: int,
) -> np.ndarray:
    """Decode one rank-zero pair and prove its receiver equation is ``Y0 == Y1``."""

    parsed = parse_g110_two_layer_v1(packet)
    if parsed.basis_q.shape[0] != 0:
        raise G110TwoLayerError("semantic-floor render requires rank-zero G110")
    provider = open_final_y1_provider(parsed.semantic_packet)
    pair = parsed.render_scorer_pair(provider, pair_id)
    if not np.array_equal(pair[0], pair[1]):
        raise AssertionError("rank-zero receiver violated Y0 == Y1")
    return pair


@dataclass(frozen=True, slots=True)
class CompiledG110TwoLayerV1:
    packet: bytes = field(repr=False)
    archive: bytes = field(repr=False)
    semantic_variant_id: str
    final_y1_binding_sha256: str
    custody_identity_sha256: str
    conditional_operand_receipt_sha256: str
    packet_sha256: str
    archive_sha256: str
    archive_bytes: int
    rice_k: int
    zero_residual_rows: int
    candidate_or_score_claim: bool = False


def compile_g110_two_layer_v1(
    semantic_packet: bytes,
    *,
    basis_q: object,
    combined_scales: object,
    coefficients_q: object,
    target_capsule_receipt: Path,
    expected_target_capsule_receipt_sha256: str,
    fresh_g105_checkpoint: Path,
    conditional_operand_receipt: Path,
    expected_conditional_operand_receipt_sha256: str,
) -> CompiledG110TwoLayerV1:
    """Reopen all producer evidence at point of use, then compile."""

    provider = open_final_y1_provider(semantic_packet)
    if provider.variant_id != V9_VARIANT_ID:
        raise G110TwoLayerError(
            "current compile authority admits only a physical fresh G105 packet"
        )
    custody = G110Batch16SourcePoseCustodyV1.from_physical_v9_producer(
        target_capsule_receipt=target_capsule_receipt,
        expected_target_capsule_receipt_sha256=expected_target_capsule_receipt_sha256,
        fresh_g105_checkpoint=fresh_g105_checkpoint,
        semantic_packet=provider.packet,
    )
    basis, scales, coefficients = _validate_conditional(
        basis_q,
        combined_scales,
        coefficients_q,
    )
    conditional_receipt_sha = _verify_conditional_operand_receipt(
        receipt_path=conditional_operand_receipt,
        expected_receipt_file_sha256=expected_conditional_operand_receipt_sha256,
        custody=custody,
        semantic_packet=provider.packet,
        basis=basis,
        scales=scales,
        coefficients=coefficients,
    )
    binding = final_y1_binding_sha256(provider)
    packet = _encode_packet(
        semantic_packet=provider.packet,
        final_y1_binding=binding,
        basis=basis,
        scales=scales,
        coefficients=coefficients,
    )
    parsed = parse_g110_two_layer_v1(packet)
    if parsed.semantic_packet != semantic_packet:
        raise AssertionError("internal two-layer parse-back changed semantic bytes")
    archive = build_g110_public_archive(packet)
    return CompiledG110TwoLayerV1(
        packet=packet,
        archive=archive,
        semantic_variant_id=provider.variant_id,
        final_y1_binding_sha256=binding,
        custody_identity_sha256=custody.identity_sha256,
        conditional_operand_receipt_sha256=conditional_receipt_sha,
        packet_sha256=_sha256(packet),
        archive_sha256=_sha256(archive),
        archive_bytes=len(archive),
        rice_k=parsed.rice_k,
        zero_residual_rows=int(np.count_nonzero(np.all(coefficients == 0, axis=1))),
    )


def parse_g110_two_layer_v1(payload: bytes) -> ParsedG110TwoLayerV1:
    """Strict EOF/CRC/typed parse with canonical Rice and packet re-emission."""

    if type(payload) is not bytes or not _HEADER.size <= len(payload) <= MAX_PACKET_BYTES:
        raise G110TwoLayerError("two-layer packet must be bounded exact bytes")
    values = _HEADER.unpack_from(payload)
    (
        magic,
        version,
        flags,
        pairs,
        scorer_h,
        scorer_w,
        channels,
        rank,
        grid_h,
        grid_w,
        codec,
        rice_k,
        semantic_length,
        basis_length,
        scale_length,
        coefficient_length,
        binding_raw,
        expected_crc,
    ) = values
    if (
        magic != MAGIC
        or version != VERSION
        or flags != 0
        or (pairs, scorer_h, scorer_w, channels)
        != (PAIR_COUNT_N600, SCORER_H, SCORER_W, SCORER_CHANNELS)
        or codec != COEFFICIENT_CODEC_RICE_DELTA
        or not 0 <= rank <= MAX_RANK
        or semantic_length <= 0
        or semantic_length > MAX_SEMANTIC_PACKET_BYTES
    ):
        raise G110TwoLayerError("two-layer header changes the closed n600 ABI")
    if rank == 0:
        expected_lengths = (semantic_length, 0, 0, 0)
        if (grid_h, grid_w, rice_k) != (0, 0, 0):
            raise G110TwoLayerError("rank-zero conditional header is noncanonical")
    else:
        if not 1 <= grid_h <= MAX_GRID_SIDE or not 1 <= grid_w <= MAX_GRID_SIDE:
            raise G110TwoLayerError("conditional grid side is outside [1,64]")
        expected_lengths = (
            semantic_length,
            rank * grid_h * grid_w * SCORER_CHANNELS,
            rank * 4,
            coefficient_length,
        )
    observed_lengths = (semantic_length, basis_length, scale_length, coefficient_length)
    if observed_lengths != expected_lengths or _HEADER.size + sum(observed_lengths) != len(payload):
        raise G110TwoLayerError("typed section lengths or exact EOF disagree")
    body = payload[_HEADER.size :]
    if zlib.crc32(body) & 0xFFFFFFFF != expected_crc:
        raise G110TwoLayerError("two-layer body CRC32 mismatch")
    cursor = 0
    sections: list[bytes] = []
    for length in observed_lengths:
        sections.append(body[cursor : cursor + length])
        cursor += length
    provider = open_final_y1_provider(sections[0])
    basis = np.frombuffer(sections[1], dtype=np.int8).reshape(
        rank,
        grid_h,
        grid_w,
        SCORER_CHANNELS,
    )
    scales = np.frombuffer(sections[2], dtype=_F32_BE).astype(np.float32)
    coefficients = _rice_decode(sections[3], rice_k=rice_k, rank=rank)
    basis, scales, coefficients = _validate_conditional(
        np.ascontiguousarray(basis),
        np.ascontiguousarray(scales),
        coefficients,
    )
    binding = binding_raw.hex()
    _require_sha256(binding, name="final Y1 binding")
    canonical = _encode_packet(
        semantic_packet=sections[0],
        final_y1_binding=binding,
        basis=basis,
        scales=scales,
        coefficients=coefficients,
    )
    if canonical != payload:
        raise G110TwoLayerError("two-layer packet changed under canonical re-emission")
    return ParsedG110TwoLayerV1(
        semantic_packet=sections[0],
        semantic_variant_id=provider.variant_id,
        final_y1_binding_sha256=binding,
        basis_q=basis,
        combined_scales=scales,
        coefficients_q=coefficients,
        rice_k=rice_k,
        packet=payload,
    )


def _zip_member(method: G110OuterZipMethodV1) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(PACKET_MEMBER, date_time=_ZIP_TIMESTAMP)
    info.compress_type = int(method)
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits = 0
    return info


def _build_g110_public_archive_for_method(
    packet: bytes,
    method: G110OuterZipMethodV1,
) -> bytes:
    parsed = parse_g110_two_layer_v1(packet)
    if parsed.packet != packet:
        raise AssertionError("internal packet custody drifted")
    if type(method) is not G110OuterZipMethodV1:
        raise G110TwoLayerError(
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
        raise G110TwoLayerError("public archive exceeds the bounded envelope")
    return result


def build_g110_counted_archive_variant(
    packet: bytes,
    method: G110OuterZipMethodV1,
) -> bytes:
    """Build one exact typed counted-archive alternative without selecting it."""

    return _build_g110_public_archive_for_method(packet, method)


def _select_g110_public_archive(
    packet: bytes,
) -> tuple[G110OuterZipMethodV1, bytes]:
    alternatives = tuple(
        (method, _build_g110_public_archive_for_method(packet, method))
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


def build_g110_public_archive(packet: bytes) -> bytes:
    _method, result = _select_g110_public_archive(packet)
    if parse_g110_public_archive(result) != packet:
        raise AssertionError("internal archive parse-back changed packet bytes")
    return result


def _read_g110_public_archive(
    archive_bytes: bytes,
) -> tuple[bytes, G110OuterZipMethodV1]:
    if type(archive_bytes) is not bytes or not 0 < len(archive_bytes) <= MAX_ARCHIVE_BYTES:
        raise G110TwoLayerError("public archive must be bounded exact bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            infos = archive.infolist()
            if [info.filename for info in infos] != [PACKET_MEMBER]:
                raise G110TwoLayerError("public archive member set/order differs")
            info = infos[0]
            mode = (info.external_attr >> 16) & 0o170000
            try:
                method = G110OuterZipMethodV1(info.compress_type)
            except ValueError as exc:
                raise G110TwoLayerError(
                    "public archive compression method is unsupported"
                ) from exc
            if (
                info.is_dir()
                or info.flag_bits & 0x1
                or mode == 0o120000
                or not _HEADER.size <= info.file_size <= MAX_PACKET_BYTES
                or info.compress_size > len(archive_bytes)
            ):
                raise G110TwoLayerError("public archive member is unsafe/noncanonical")
            packet = archive.read(info)
            if len(packet) != info.file_size:
                raise G110TwoLayerError("public archive member length differs")
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise G110TwoLayerError("public archive cannot be decoded") from exc
    parse_g110_two_layer_v1(packet)
    return packet, method


def parse_g110_counted_archive_variant(
    archive_bytes: bytes,
    expected_method: G110OuterZipMethodV1,
) -> bytes:
    """Parse one exact typed alternative, including nonselected matrix entries."""

    if type(expected_method) is not G110OuterZipMethodV1:
        raise G110TwoLayerError(
            "expected archive method must be G110OuterZipMethodV1"
        )
    packet, method = _read_g110_public_archive(archive_bytes)
    if (
        method is not expected_method
        or archive_bytes
        != _build_g110_public_archive_for_method(packet, expected_method)
    ):
        raise G110TwoLayerError(
            "counted archive differs from its typed deterministic variant"
        )
    return packet


def parse_g110_public_archive(archive_bytes: bytes) -> bytes:
    packet, method = _read_g110_public_archive(archive_bytes)
    selected_method, selected_archive = _select_g110_public_archive(packet)
    if method is not selected_method or archive_bytes != selected_archive:
        raise G110TwoLayerError(
            "public archive is not the deterministic canonical method/layout"
        )
    return packet


__all__ = [
    "CONDITIONAL_VARIANT_ID",
    "G103_VARIANT_ID",
    "MAGIC",
    "PACKET_MEMBER",
    "PUBLIC_RUNTIME_RELATIVE_ROOT",
    "V9_VARIANT_ID",
    "CompiledG110TwoLayerV1",
    "G110Batch16SourcePoseCustodyV1",
    "G110OuterZipMethodV1",
    "G110TwoLayerError",
    "OpenedFinalY1ProviderV1",
    "ParsedG110TwoLayerV1",
    "build_g110_counted_archive_variant",
    "build_g110_public_archive",
    "build_g110_rank_zero_semantic_floor_packet",
    "compile_g110_two_layer_v1",
    "final_y1_binding_sha256",
    "open_final_y1_provider",
    "parse_g110_counted_archive_variant",
    "parse_g110_public_archive",
    "parse_g110_two_layer_v1",
    "render_conditional_y0",
    "render_g110_rank_zero_scorer_pair",
]
