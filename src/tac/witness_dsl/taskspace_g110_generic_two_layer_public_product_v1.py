# SPDX-License-Identifier: MIT
"""Receiver-closed two-layer packet over a generic final-Y1 provider.

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
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
import zlib
from dataclasses import dataclass, field
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
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    MAGIC as V9_MAGIC,
)
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    VARIANT_ID as V9_VARIANT_ID,
)
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    CompiledFreshV9SemanticRootV1,
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

_HEADER: Final = struct.Struct(">8sBBHHHBBHHBBIIII32sI")
_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
_F32_BE: Final = np.dtype(">f4")


class G110TwoLayerError(ValueError):
    """The generic provider, conditional stream, custody, or packet failed."""


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


@dataclass(frozen=True, slots=True, init=False)
class G110Batch16SourcePoseCustodyV1:
    """Content-read G109 + fresh-G105 evidence required before compilation."""

    target_margins_sha256: str
    pose_targets_sha256: str
    target_capsule_receipt_sha256: str
    fresh_checkpoint_sha256: str
    semantic_packet_sha256: str
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
    def from_verified_v9_producer(
        cls,
        *,
        target_capsule_receipt: Path,
        expected_target_capsule_receipt_sha256: str,
        compiled_v9: CompiledFreshV9SemanticRootV1,
    ) -> G110Batch16SourcePoseCustodyV1:
        """Reopen every G109 input and bind it to the fresh G105 packet."""

        if type(compiled_v9) is not CompiledFreshV9SemanticRootV1:
            raise G110TwoLayerError("custody requires exact fresh G105 compile output")
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
        external = compiled_v9.external_receipt
        if (
            type(external) is not dict
            or external.get("schema") != "tac.g105_fresh_v9_semantic_root_compile_receipt.v1"
            or external.get("candidate") is not False
            or external.get("score_claim") is not False
        ):
            raise G110TwoLayerError("G105 fresh compile receipt is absent or weakens authority")
        checkpoint = external.get("checkpoint")
        target = external.get("g46_batch16_training_target_evidence")
        wire = external.get("wire")
        raw = loader.receipt.get("raw_arrays")
        runtime = loader.preflight.get("runtime_custody")
        if (
            not isinstance(checkpoint, dict)
            or not isinstance(target, dict)
            or not isinstance(wire, dict)
            or not isinstance(raw, dict)
            or not isinstance(runtime, dict)
            or set(raw) != {"labels", "margins", "poses"}
        ):
            raise G110TwoLayerError("G105/G109 custody sections are incomplete")
        exact_checkpoint_sha = _require_sha256(
            checkpoint.get("sha256"),
            name="fresh checkpoint",
        )
        checkpoint_path = _resolve_regular_nonsymlink(
            Path(str(checkpoint.get("path"))),
            name="fresh G105 checkpoint",
        )
        if (
            checkpoint_path.stat().st_size != checkpoint.get("bytes")
            or sha256_file(checkpoint_path) != exact_checkpoint_sha
        ):
            raise G110TwoLayerError("fresh G105 checkpoint file no longer matches compile receipt")
        exact_margin_sha = _require_sha256(
            raw["margins"].get("sha256"),
            name="G109 margins",
        )
        exact_pose_sha = _require_sha256(
            raw["poses"].get("sha256"),
            name="G109 poses",
        )
        if (
            target.get("target_labels_sha256") != raw["labels"].get("sha256")
            or target.get("target_margins_sha256") != exact_margin_sha
            or target.get("scorer_pair_batch_size") != PAIR_BATCH_SIZE
            or target.get("live_verdict_batch_size") != PAIR_BATCH_SIZE
            or target.get("margins_from_same_batch16_forward") is not True
            or target.get("source_pair_chain_sha256") != G46_SOURCE_PAIR_CHAIN_SHA256
            or checkpoint.get("fresh_init") is not True
            or checkpoint.get("portable_upstream_closure_sha256")
            != UPSTREAM_SOURCE_CLOSURE_SHA256
            or runtime.get("upstream_closure_sha256")
            != UPSTREAM_SOURCE_CLOSURE_SHA256
            or sha256_file(target_capsule_receipt) != expected_receipt_sha
            or wire.get("packet_sha256") != _sha256(compiled_v9.packet)
            or wire.get("packet_bytes") != len(compiled_v9.packet)
            or wire.get("counted_y1_rows") != PAIR_COUNT_N600
            or wire.get("excluded_y0_rows") != PAIR_COUNT_N600
            or compiled_v9.wire_accounting.packet_sha256 != _sha256(compiled_v9.packet)
            or encode_v9_packet(parse_v9_packet(compiled_v9.packet)) != compiled_v9.packet
        ):
            raise G110TwoLayerError("G105 packet and G109 batch-16 target capsule disagree")
        instance = object.__new__(cls)
        values = {
            "target_margins_sha256": exact_margin_sha,
            "pose_targets_sha256": exact_pose_sha,
            "target_capsule_receipt_sha256": expected_receipt_sha,
            "fresh_checkpoint_sha256": exact_checkpoint_sha,
            "semantic_packet_sha256": _sha256(compiled_v9.packet),
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
    return basis, scales, coefficients


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
    ys = (np.arange(output_height, dtype=np.float64) + 0.5) * input_height / output_height - 0.5
    xs = (np.arange(output_width, dtype=np.float64) + 0.5) * input_width / output_width - 0.5
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    wy = (ys - y0).astype(np.float32)
    wx = (xs - x0).astype(np.float32)
    y0 = np.clip(y0, 0, input_height - 1)
    x0 = np.clip(x0, 0, input_width - 1)
    y1 = np.clip(y0 + 1, 0, input_height - 1)
    x1 = np.clip(x0 + 1, 0, input_width - 1)
    top = image[y0[:, None], x0[None, :]] * (1.0 - wx[None, :, None])
    top += image[y0[:, None], x1[None, :]] * wx[None, :, None]
    bottom = image[y1[:, None], x0[None, :]] * (1.0 - wx[None, :, None])
    bottom += image[y1[:, None], x1[None, :]] * wx[None, :, None]
    return top * (1.0 - wy[:, None, None]) + bottom * wy[:, None, None]


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
    weights = coefficients_q[pair_id].astype(np.float32) * combined_scales
    grid = np.einsum(
        "r,rhwc->hwc",
        weights,
        basis_q.astype(np.float32),
        optimize=True,
        dtype=np.float32,
    )
    residual = _bilinear_resize(
        np.ascontiguousarray(grid, dtype=np.float32),
        output_height=SCORER_H,
        output_width=SCORER_W,
    )
    return np.ascontiguousarray(
        np.clip(np.rint(y1.astype(np.float32) + residual), 0, 255).astype(np.uint8)
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
    custody: G110Batch16SourcePoseCustodyV1,
    conditional_operand_receipt: Path,
    expected_conditional_operand_receipt_sha256: str,
) -> CompiledG110TwoLayerV1:
    """Compile only from fresh batch-16/source/pose-bound external custody."""

    if type(custody) is not G110Batch16SourcePoseCustodyV1:
        raise G110TwoLayerError("compile requires exact G110 batch-16 custody")
    provider = open_final_y1_provider(semantic_packet)
    if provider.variant_id != V9_VARIANT_ID or _sha256(provider.packet) != custody.semantic_packet_sha256:
        raise G110TwoLayerError(
            "current compile authority admits only the exact content-verified fresh G105 packet"
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


def _zip_member(payload: bytes) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(PACKET_MEMBER, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits = 0
    return info


def build_g110_public_archive(packet: bytes) -> bytes:
    parsed = parse_g110_two_layer_v1(packet)
    if parsed.packet != packet:
        raise AssertionError("internal packet custody drifted")
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=False,
    ) as archive:
        archive.writestr(
            _zip_member(packet),
            packet,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )
    result = stream.getvalue()
    if not result or len(result) > MAX_ARCHIVE_BYTES:
        raise G110TwoLayerError("public archive exceeds the bounded envelope")
    if parse_g110_public_archive(result) != packet:
        raise AssertionError("internal archive parse-back changed packet bytes")
    return result


def parse_g110_public_archive(archive_bytes: bytes) -> bytes:
    if type(archive_bytes) is not bytes or not 0 < len(archive_bytes) <= MAX_ARCHIVE_BYTES:
        raise G110TwoLayerError("public archive must be bounded exact bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            infos = archive.infolist()
            if [info.filename for info in infos] != [PACKET_MEMBER]:
                raise G110TwoLayerError("public archive member set/order differs")
            info = infos[0]
            mode = (info.external_attr >> 16) & 0o170000
            if (
                info.is_dir()
                or info.flag_bits & 0x1
                or mode == 0o120000
                or info.compress_type != zipfile.ZIP_DEFLATED
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
    "G110TwoLayerError",
    "OpenedFinalY1ProviderV1",
    "ParsedG110TwoLayerV1",
    "build_g110_public_archive",
    "compile_g110_two_layer_v1",
    "final_y1_binding_sha256",
    "open_final_y1_provider",
    "parse_g110_public_archive",
    "parse_g110_two_layer_v1",
    "render_conditional_y0",
]
