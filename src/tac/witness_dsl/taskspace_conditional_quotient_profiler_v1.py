# SPDX-License-Identifier: MIT
"""Resumable full-population profiler for the V15-to-selected-preimage debt.

The profiler is encoder-side research instrumentation.  It does not build a
candidate and it never serializes a teacher plane.  Its input seam is the
existing C0B ``PlaneChunk`` contract: one strict V15-derived base chunk, one
custody-bound selected-preimage target chunk, and one fresh batch-16 Seg target
label chunk used only for class conditioning.

All measured bases are exactly reversible.  The result can therefore falsify
specific lossless layouts under a conditional byte budget, but cannot claim a
score or prove that a learned quotient is necessary.  The canonical existing
batch-16 debt receipt is the primary planning coordinate.  A later independent
batch-16 replay is bound only as corroboration, and the older batch-32 MS1 row
is retained only as historical comparison.  None is promoted into an archive
score by this encoder-only profiler.
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.witness_dsl import c0b_semantic_quotient as _c0b
from tac.witness_dsl.c0b_semantic_quotient import (
    PlaneChunk,
    SemanticQuotientError,
    canonical_json,
    storage_preflight,
    write_once_or_equal,
)

CONFIG_SCHEMA: Final = "tac.taskspace_conditional_quotient_profile_config.v1"
INPUT_BINDING_SCHEMA: Final = "tac.taskspace_conditional_quotient_profile_inputs.v1"
CHUNK_SCHEMA: Final = "tac.taskspace_conditional_quotient_chunk_profile.v1"
AGGREGATE_SCHEMA: Final = "tac.taskspace_conditional_quotient_aggregate.v1"
RUN_SCHEMA: Final = "tac.taskspace_conditional_quotient_run.v1"
EVIDENCE_AXIS: Final = "[encoder-only exact-byte quotient diagnostic]"
UPSTREAM_DEFAULT_BATCH_SIZE: Final = 16
PUBLIC_PAIR_COUNT: Final = 600
PUBLIC_SCORER_HW: Final = (384, 512)
PUBLIC_CHANNELS: Final = 3
REPRESENTATION_IDS: Final = (
    "c0b_xor_separate",
    "c0b_xor_interleaved",
    "signed_residual_separate",
    "common_differential_signed",
    "pair_temporal_signed_chunk_reset",
    "seg_y1_plus_pose_y0_xor_y1",
)
COMPRESSOR_IDS: Final = ("zlib9_block", "c0b_lzma_block")
INPUT_BINDING_KEYS: Final = frozenset(
    {
        "schema",
        "evidence_axis",
        "research_only",
        "score_claim",
        "promotion_eligible",
        "pointer_mutation_allowed",
        "candidate_payload_allowed",
        "teacher_payload_serialized",
        "scorer_weights_present",
        "pair_count",
        "scorer_hw",
        "channels",
        "v15_archive_path",
        "v15_archive_bytes",
        "v15_archive_sha256",
        "v15_strict_parse",
        "v15_current_receiver_source_sha256",
        "fresh_v15_derivation_custody",
        "base_coordinate_transform",
        "selected_plane_teacher_id",
        "selected_plane_y0_sha256",
        "selected_plane_y1_sha256",
        "selected_plane_origin_scorer_batch_size",
        "selected_plane_geometry_custody",
        "fresh_teacher_scorer_batch_size",
        "fresh_teacher_target_labels_path",
        "fresh_teacher_target_labels_sha256",
        "fresh_teacher_receipt",
        "upstream_default_scorer_batch_size",
        "current_planning_scorer_batch_size",
        "current_planning_matches_upstream_batch_geometry",
        "canonical_batch16_debt_receipt",
        "independent_batch16_replay_corroboration",
        "planning_coordinate_premise",
        "frontier_pointer",
        "current_batch16_planning_coordinate",
        "historical_ms1_batch32_counterfactual",
        "implementation_sources",
    }
)
FRESH_V15_DERIVATION_SCHEMA: Final = "tac.taskspace_fresh_v15_derivation_custody.v1"
FRESH_V15_RECEIPT_SCHEMA: Final = "ddm_v15_scorer_solved_template_receipt.v1"
FRESH_V15_RECEIVER_CHECKPOINT_SCHEMA: Final = "ddm_v15_receiver_closed_archive.v1"
FRESH_V15_ARCHIVE_BYTES: Final = 133_941
FRESH_V15_IDENTITY_CHECKPOINT_COUNT: Final = 38


class ConditionalQuotientProfilerError(ValueError):
    """A profile input, exact transform, checkpoint, or aggregate failed."""


def _sha256(payload: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_exact_int(
    value: Any,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ConditionalQuotientProfilerError(f"{field} must be an exact integer in [{minimum}, {maximum}]")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ConditionalQuotientProfilerError(f"{field} must be a lowercase SHA-256")
    return value


def _canonical_mapping(value: Mapping[str, Any], *, label: str) -> bytes:
    try:
        return canonical_json(dict(value))
    except SemanticQuotientError as exc:
        raise ConditionalQuotientProfilerError(f"{label} is not canonical JSON") from exc


def _read_canonical_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConditionalQuotientProfilerError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict) or _canonical_mapping(value, label=label) != payload:
        raise ConditionalQuotientProfilerError(f"{label} is not a canonical JSON object")
    return value


@dataclass(frozen=True, slots=True)
class ConditionalQuotientProfileConfigV1:
    """Closed scientific configuration; paths and custody live in the binding."""

    pair_count: int
    chunk_pairs: int
    scorer_hw: tuple[int, int] = PUBLIC_SCORER_HW
    channels: int = PUBLIC_CHANNELS
    resume: bool = True
    test_only_small_fixture: bool = False
    allow_local_storage: bool = False

    def __post_init__(self) -> None:
        pairs = _require_exact_int(
            self.pair_count,
            field="pair_count",
            minimum=1,
            maximum=PUBLIC_PAIR_COUNT,
        )
        chunk = _require_exact_int(
            self.chunk_pairs,
            field="chunk_pairs",
            minimum=1,
            maximum=pairs,
        )
        if (
            not isinstance(self.scorer_hw, tuple)
            or len(self.scorer_hw) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.scorer_hw)
        ):
            raise ConditionalQuotientProfilerError("scorer_hw must be an exact positive (H,W) tuple")
        _require_exact_int(self.channels, field="channels", minimum=1, maximum=16)
        if not all(
            isinstance(value, bool) for value in (self.resume, self.test_only_small_fixture, self.allow_local_storage)
        ):
            raise ConditionalQuotientProfilerError("resume/test/storage switches must be booleans")
        if not self.test_only_small_fixture and (
            pairs != PUBLIC_PAIR_COUNT or self.scorer_hw != PUBLIC_SCORER_HW or self.channels != PUBLIC_CHANNELS
        ):
            raise ConditionalQuotientProfilerError("scientific profile requires the exact n600 public scorer geometry")
        object.__setattr__(self, "pair_count", pairs)
        object.__setattr__(self, "chunk_pairs", chunk)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ConditionalQuotientProfileConfigV1:
        allowed = {
            "schema",
            "pair_count",
            "chunk_pairs",
            "scorer_hw",
            "channels",
            "resume",
            "test_only_small_fixture",
            "allow_local_storage",
        }
        if set(value) != allowed or value.get("schema") != CONFIG_SCHEMA:
            raise ConditionalQuotientProfilerError("profile config keys/schema differ")
        scorer_hw = value["scorer_hw"]
        if not isinstance(scorer_hw, list) or len(scorer_hw) != 2:
            raise ConditionalQuotientProfilerError("profile config scorer_hw must be [H,W]")
        return cls(
            pair_count=value["pair_count"],
            chunk_pairs=value["chunk_pairs"],
            scorer_hw=(scorer_hw[0], scorer_hw[1]),
            channels=value["channels"],
            resume=value["resume"],
            test_only_small_fixture=value["test_only_small_fixture"],
            allow_local_storage=value["allow_local_storage"],
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema": CONFIG_SCHEMA,
            "pair_count": self.pair_count,
            "chunk_pairs": self.chunk_pairs,
            "scorer_hw": list(self.scorer_hw),
            "channels": self.channels,
            "resume": self.resume,
            "test_only_small_fixture": self.test_only_small_fixture,
            "allow_local_storage": self.allow_local_storage,
        }


ChunkLoader = Callable[[int, tuple[int, ...]], tuple[PlaneChunk, PlaneChunk, np.ndarray]]


def _validate_fresh_v15_derivation_custody(
    value: Any,
    *,
    archive_path: Any,
    archive_bytes: Any,
    archive_sha256: Any,
) -> None:
    expected_keys = {
        "schema",
        "run_id",
        "derivation_proof_separate_from_archive_content_identity",
        "historical_path_fallback_allowed",
        "compile_receipt",
        "source_config",
        "adjacent_archive",
        "producer_sources",
        "receiver_checkpoint",
        "full_p_camera_identity",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("schema") != FRESH_V15_DERIVATION_SCHEMA
        or value.get("derivation_proof_separate_from_archive_content_identity") is not True
        or value.get("historical_path_fallback_allowed") is not False
    ):
        raise ConditionalQuotientProfilerError("fresh V15 derivation custody boundary differs")
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or len(run_id) < 8:
        raise ConditionalQuotientProfilerError("fresh V15 derivation run_id is invalid")
    compile_receipt = value.get("compile_receipt")
    if (
        not isinstance(compile_receipt, dict)
        or set(compile_receipt) != {"path", "bytes", "sha256", "schema", "run_id"}
        or compile_receipt.get("schema") != FRESH_V15_RECEIPT_SCHEMA
        or compile_receipt.get("run_id") != run_id
    ):
        raise ConditionalQuotientProfilerError("fresh V15 compile receipt custody differs")
    _require_exact_int(
        compile_receipt.get("bytes"),
        field="fresh_v15.compile_receipt.bytes",
        minimum=1,
        maximum=1 << 40,
    )
    _require_sha256(
        compile_receipt.get("sha256"),
        field="fresh_v15.compile_receipt.sha256",
    )
    source_config = value.get("source_config")
    if not isinstance(source_config, dict) or set(source_config) != {
        "path",
        "bytes",
        "sha256",
        "rfc8785_sha256",
    }:
        raise ConditionalQuotientProfilerError("fresh V15 source config custody differs")
    _require_exact_int(
        source_config.get("bytes"),
        field="fresh_v15.source_config.bytes",
        minimum=1,
        maximum=1 << 40,
    )
    _require_sha256(source_config.get("sha256"), field="fresh_v15.source_config.sha256")
    typed_config_sha = _require_sha256(
        source_config.get("rfc8785_sha256"),
        field="fresh_v15.source_config.rfc8785_sha256",
    )
    adjacent_archive = value.get("adjacent_archive")
    if (
        not isinstance(adjacent_archive, dict)
        or set(adjacent_archive) != {"path", "bytes", "sha256", "content_identity_only"}
        or adjacent_archive.get("path") != archive_path
        or adjacent_archive.get("bytes") != archive_bytes
        or adjacent_archive.get("bytes") != FRESH_V15_ARCHIVE_BYTES
        or adjacent_archive.get("sha256") != archive_sha256
        or adjacent_archive.get("content_identity_only") is not True
    ):
        raise ConditionalQuotientProfilerError("fresh V15 adjacent archive content identity differs")
    _require_sha256(adjacent_archive.get("sha256"), field="fresh_v15.adjacent_archive.sha256")
    producers = value.get("producer_sources")
    if not isinstance(producers, list) or not producers:
        raise ConditionalQuotientProfilerError("fresh V15 live producer source custody is absent")
    producer_paths: set[str] = set()
    for index, producer in enumerate(producers):
        if (
            not isinstance(producer, dict)
            or set(producer) != {
                "path",
                "resolved_path",
                "bytes",
                "sha256",
                "live_rehashed",
            }
            or producer.get("live_rehashed") is not True
        ):
            raise ConditionalQuotientProfilerError(
                f"fresh V15 live producer source {index} custody differs"
            )
        path = producer.get("path")
        if not isinstance(path, str) or not path or path in producer_paths:
            raise ConditionalQuotientProfilerError("fresh V15 producer source paths are invalid")
        producer_paths.add(path)
        _require_exact_int(
            producer.get("bytes"),
            field=f"fresh_v15.producer_sources[{index}].bytes",
            minimum=1,
            maximum=1 << 40,
        )
        _require_sha256(
            producer.get("sha256"),
            field=f"fresh_v15.producer_sources[{index}].sha256",
        )
    receiver_checkpoint = value.get("receiver_checkpoint")
    if (
        not isinstance(receiver_checkpoint, dict)
        or set(receiver_checkpoint)
        != {
            "path",
            "bytes",
            "sha256",
            "schema",
            "typed_config_sha256",
            "archive_sha256",
            "score_claim",
        }
        or receiver_checkpoint.get("schema") != FRESH_V15_RECEIVER_CHECKPOINT_SCHEMA
        or receiver_checkpoint.get("typed_config_sha256") != typed_config_sha
        or receiver_checkpoint.get("archive_sha256") != archive_sha256
        or receiver_checkpoint.get("score_claim") is not False
    ):
        raise ConditionalQuotientProfilerError("fresh V15 receiver checkpoint custody differs")
    _require_exact_int(
        receiver_checkpoint.get("bytes"),
        field="fresh_v15.receiver_checkpoint.bytes",
        minimum=1,
        maximum=1 << 40,
    )
    _require_sha256(
        receiver_checkpoint.get("sha256"),
        field="fresh_v15.receiver_checkpoint.sha256",
    )
    identity = value.get("full_p_camera_identity")
    expected_identity_keys = {
        "pair_count",
        "batch_count",
        "batch_size",
        "typed_config_sha256",
        "ordered_checkpoints",
        "receipt_digest_chain_sha256",
        "recomputed_digest_chain_sha256",
        "digest_chain_matches_receipt",
        "all_camera_bytes_identical",
        "score_claim",
    }
    if (
        not isinstance(identity, dict)
        or set(identity) != expected_identity_keys
        or identity.get("pair_count") != PUBLIC_PAIR_COUNT
        or identity.get("batch_count") != FRESH_V15_IDENTITY_CHECKPOINT_COUNT
        or identity.get("batch_size") != UPSTREAM_DEFAULT_BATCH_SIZE
        or identity.get("typed_config_sha256") != typed_config_sha
        or identity.get("digest_chain_matches_receipt") is not True
        or identity.get("all_camera_bytes_identical") is not True
        or identity.get("score_claim") is not False
    ):
        raise ConditionalQuotientProfilerError("fresh V15 full-P identity summary differs")
    checkpoints = identity.get("ordered_checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != FRESH_V15_IDENTITY_CHECKPOINT_COUNT:
        raise ConditionalQuotientProfilerError("fresh V15 full-P identity checkpoint count differs")
    digest_material: list[str] = []
    for index, checkpoint in enumerate(checkpoints):
        start = index * UPSTREAM_DEFAULT_BATCH_SIZE
        stop = min(start + UPSTREAM_DEFAULT_BATCH_SIZE, PUBLIC_PAIR_COUNT)
        if (
            not isinstance(checkpoint, dict)
            or set(checkpoint)
            != {
                "path",
                "bytes",
                "sha256",
                "local_pair_range",
                "typed_config_sha256",
                "base_camera_sha256",
                "final_camera_sha256",
                "byte_identical",
                "score_claim",
            }
            or checkpoint.get("local_pair_range") != [start, stop]
            or checkpoint.get("typed_config_sha256") != typed_config_sha
            or checkpoint.get("byte_identical") is not True
            or checkpoint.get("score_claim") is not False
        ):
            raise ConditionalQuotientProfilerError(
                f"fresh V15 full-P identity checkpoint {index} differs"
            )
        _require_exact_int(
            checkpoint.get("bytes"),
            field=f"fresh_v15.full_p_camera_identity.ordered_checkpoints[{index}].bytes",
            minimum=1,
            maximum=1 << 40,
        )
        _require_sha256(
            checkpoint.get("sha256"),
            field=f"fresh_v15.full_p_camera_identity.ordered_checkpoints[{index}].sha256",
        )
        base_digest = _require_sha256(
            checkpoint.get("base_camera_sha256"),
            field=f"fresh_v15.full_p_camera_identity.ordered_checkpoints[{index}].base_camera_sha256",
        )
        final_digest = _require_sha256(
            checkpoint.get("final_camera_sha256"),
            field=f"fresh_v15.full_p_camera_identity.ordered_checkpoints[{index}].final_camera_sha256",
        )
        if base_digest != final_digest:
            raise ConditionalQuotientProfilerError(
                f"fresh V15 full-P identity checkpoint {index} camera digests differ"
            )
        digest_material.append(base_digest + final_digest)
    recomputed = _sha256("".join(digest_material).encode("ascii"))
    receipt_chain = _require_sha256(
        identity.get("receipt_digest_chain_sha256"),
        field="fresh_v15.full_p_camera_identity.receipt_digest_chain_sha256",
    )
    if (
        identity.get("recomputed_digest_chain_sha256") != recomputed
        or receipt_chain != recomputed
    ):
        raise ConditionalQuotientProfilerError("fresh V15 full-P identity digest chain differs")


def _validate_input_binding(
    value: Mapping[str, Any],
    *,
    config: ConditionalQuotientProfileConfigV1,
) -> dict[str, Any]:
    binding = dict(value)
    if set(binding) != INPUT_BINDING_KEYS or binding.get("schema") != INPUT_BINDING_SCHEMA:
        raise ConditionalQuotientProfilerError("input binding keys/schema differ")
    required_false = (
        "score_claim",
        "promotion_eligible",
        "pointer_mutation_allowed",
        "candidate_payload_allowed",
        "teacher_payload_serialized",
        "scorer_weights_present",
    )
    if any(binding.get(field) is not False for field in required_false):
        raise ConditionalQuotientProfilerError("input binding weakened research/candidate boundaries")
    if binding.get("evidence_axis") != EVIDENCE_AXIS or binding.get("research_only") is not True:
        raise ConditionalQuotientProfilerError("input binding evidence axis differs")
    if binding.get("pair_count") != config.pair_count:
        raise ConditionalQuotientProfilerError("input binding pair count differs")
    if binding.get("scorer_hw") != list(config.scorer_hw) or binding.get("channels") != config.channels:
        raise ConditionalQuotientProfilerError("input binding scorer geometry differs")
    if (
        binding.get("v15_strict_parse") is not True
        or isinstance(binding.get("v15_archive_bytes"), bool)
        or not isinstance(binding.get("v15_archive_bytes"), int)
        or binding["v15_archive_bytes"] <= 0
    ):
        raise ConditionalQuotientProfilerError("input binding V15 strict parse/size differs")
    _validate_fresh_v15_derivation_custody(
        binding.get("fresh_v15_derivation_custody"),
        archive_path=binding.get("v15_archive_path"),
        archive_bytes=binding.get("v15_archive_bytes"),
        archive_sha256=binding.get("v15_archive_sha256"),
    )
    transform = binding.get("base_coordinate_transform")
    if transform != {
        "camera_hw": [874, 1164],
        "scorer_hw": list(config.scorer_hw),
        "method": "c0b_disjoint_factor2_exact_integer_resize_round_u8",
        "operator_source": "src/tac/optimization/uint8_lattice_feasibility.py",
    }:
        raise ConditionalQuotientProfilerError("input binding exact coordinate transform differs")
    origin_batch = _require_exact_int(
        binding.get("selected_plane_origin_scorer_batch_size"),
        field="selected_plane_origin_scorer_batch_size",
        minimum=1,
        maximum=64,
    )
    fresh_batch = _require_exact_int(
        binding.get("fresh_teacher_scorer_batch_size"),
        field="fresh_teacher_scorer_batch_size",
        minimum=1,
        maximum=64,
    )
    upstream_batch = _require_exact_int(
        binding.get("upstream_default_scorer_batch_size"),
        field="upstream_default_scorer_batch_size",
        minimum=1,
        maximum=64,
    )
    if fresh_batch != UPSTREAM_DEFAULT_BATCH_SIZE or upstream_batch != UPSTREAM_DEFAULT_BATCH_SIZE:
        raise ConditionalQuotientProfilerError("fresh teacher is not bound to upstream batch-16 geometry")
    planning_batch = _require_exact_int(
        binding.get("current_planning_scorer_batch_size"),
        field="current_planning_scorer_batch_size",
        minimum=1,
        maximum=64,
    )
    if origin_batch != 32:
        raise ConditionalQuotientProfilerError("historical selected-plane origin batch must remain recorded as 32")
    for field in (
        "v15_archive_sha256",
        "selected_plane_y0_sha256",
        "selected_plane_y1_sha256",
        "fresh_teacher_target_labels_sha256",
        "v15_current_receiver_source_sha256",
    ):
        _require_sha256(binding.get(field), field=field)
    expected_alignment = planning_batch == fresh_batch == upstream_batch
    if binding.get("current_planning_matches_upstream_batch_geometry") is not expected_alignment:
        raise ConditionalQuotientProfilerError("input binding current planning batch-alignment truth value differs")
    if not expected_alignment:
        raise ConditionalQuotientProfilerError("current planning coordinate must use upstream batch 16")
    if (
        binding.get("planning_coordinate_premise")
        != "PREEXISTING_CANONICAL_BATCH16_PRIMARY_G54_INDEPENDENT_CORROBORATION_NO_NOVELTY"
    ):
        raise ConditionalQuotientProfilerError("planning-coordinate prior-art premise differs")
    for field, expected_authority in (
        (
            "canonical_batch16_debt_receipt",
            "PRIMARY_EXISTING_BATCH16_PLANNING_COORDINATE_NOT_SCORE_AUTHORITY",
        ),
        (
            "independent_batch16_replay_corroboration",
            "INDEPENDENT_CORROBORATION_ONLY_NOT_PRIMARY_OR_SCORE_AUTHORITY",
        ),
    ):
        receipt = binding.get(field)
        if (
            not isinstance(receipt, dict)
            or receipt.get("authority") != expected_authority
            or receipt.get("score_claim") is not False
            or receipt.get("batch_size") != UPSTREAM_DEFAULT_BATCH_SIZE
        ):
            raise ConditionalQuotientProfilerError(f"{field} authority/batch contract differs")
        _require_sha256(receipt.get("sha256"), field=f"{field}.sha256")
        _require_sha256(receipt.get("receipt_sha256"), field=f"{field}.receipt_sha256")
    planning = binding.get("current_batch16_planning_coordinate")
    historical = binding.get("historical_ms1_batch32_counterfactual")
    if not isinstance(planning, dict) or not isinstance(historical, dict):
        raise ConditionalQuotientProfilerError("input binding lacks current/historical planning coordinates")
    for field in (
        "headroom_bytes_to_effective_frontier",
        "headroom_bytes_to_sub_0_15",
        "base_archive_bytes",
    ):
        _require_exact_int(planning.get(field), field=f"current.{field}", minimum=0, maximum=1 << 40)
        _require_exact_int(historical.get(field), field=f"historical.{field}", minimum=0, maximum=1 << 40)
    frontier_score = planning.get("effective_frontier_score")
    if (
        isinstance(frontier_score, bool)
        or not isinstance(frontier_score, (int, float))
        or not math.isfinite(float(frontier_score))
        or float(frontier_score) <= 0.0
    ):
        raise ConditionalQuotientProfilerError("counterfactual effective frontier score is invalid")
    if (
        planning.get("authority") != "canonical_batch16_planning_arithmetic_only_not_new_eval_or_frontier_authority"
        or planning.get("score_claim") is not False
    ):
        raise ConditionalQuotientProfilerError("current planning headroom authority label differs")
    if (
        historical.get("authority") != "historical_batch32_coupled_score_arithmetic_only_not_eval_or_frontier_authority"
        or historical.get("score_claim") is not False
    ):
        raise ConditionalQuotientProfilerError("historical headroom authority label differs")
    _canonical_mapping(binding, label="input binding")
    return binding


def _int16_le(array: np.ndarray) -> bytes:
    return np.asarray(array, dtype="<i2", order="C").tobytes(order="C")


def _zero_run_stats(array: np.ndarray) -> dict[str, Any]:
    zero = np.ravel(np.asarray(array) == 0)
    zero_count = int(np.count_nonzero(zero))
    if zero_count == 0:
        return {
            "values": int(zero.size),
            "zero_values": 0,
            "zero_fraction": 0.0,
            "zero_run_count": 0,
            "zero_run_mean": 0.0,
            "zero_run_max": 0,
        }
    padded = np.empty(zero.size + 2, dtype=np.int8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = zero
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    lengths = stops - starts
    return {
        "values": int(zero.size),
        "zero_values": zero_count,
        "zero_fraction": zero_count / int(zero.size),
        "zero_run_count": int(lengths.size),
        "zero_run_mean": float(lengths.mean()),
        "zero_run_max": int(lengths.max()),
    }


def _symbol_stats(array: np.ndarray, *, minimum: int, maximum: int) -> dict[str, Any]:
    values = np.asarray(array)
    if values.size == 0:
        raise ConditionalQuotientProfilerError("symbol statistics require at least one value")
    observed_min = int(values.min())
    observed_max = int(values.max())
    if observed_min < minimum or observed_max > maximum:
        raise ConditionalQuotientProfilerError("symbol values escaped the declared exact range")
    counts = np.bincount(
        np.asarray(values, dtype=np.int64).ravel() - minimum,
        minlength=maximum - minimum + 1,
    )
    nonzero = counts[counts > 0]
    probabilities = nonzero.astype(np.float64) / int(values.size)
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    result = {
        "minimum": minimum,
        "maximum": maximum,
        "observed_minimum": observed_min,
        "observed_maximum": observed_max,
        "value_count": int(values.size),
        "nonzero_value_count": int(np.count_nonzero(values)),
        "sum_abs": int(np.abs(values.astype(np.int64)).sum()),
        "sum_squares": int(np.square(values.astype(np.int64)).sum()),
        "entropy_bits_per_value": entropy,
        "entropy_lower_bound_bytes": math.ceil(entropy * int(values.size) / 8.0),
        "histogram_counts": counts.astype(np.int64).tolist(),
        "runs": _zero_run_stats(values),
    }
    return result


def _codec_sizes(parts: Sequence[bytes]) -> dict[str, Any]:
    if not parts or any(not isinstance(part, bytes) or not part for part in parts):
        raise ConditionalQuotientProfilerError("exact compressor parts must be non-empty bytes")
    raw = sum(len(part) for part in parts)
    try:
        zlib_bytes = sum(len(zlib.compress(part, level=9)) for part in parts)
        lzma_bytes = sum(len(_c0b._compress_quotient(part)) for part in parts)
    except (zlib.error, SemanticQuotientError) as exc:
        raise ConditionalQuotientProfilerError("exact block compression failed") from exc
    return {
        "part_count": len(parts),
        "raw_bytes": raw,
        "zlib9_block_bytes": zlib_bytes,
        "c0b_lzma_block_bytes": lzma_bytes,
    }


def _representation(
    component_arrays: Mapping[str, tuple[np.ndarray, int, int]],
    parts: Sequence[bytes],
) -> dict[str, Any]:
    return {
        "codec_sizes": _codec_sizes(parts),
        "components": {
            name: _symbol_stats(array, minimum=minimum, maximum=maximum)
            for name, (array, minimum, maximum) in component_arrays.items()
        },
    }


def _pair_zlib_bytes(parts: Sequence[bytes]) -> int:
    try:
        return sum(len(zlib.compress(part, level=9)) for part in parts)
    except zlib.error as exc:
        raise ConditionalQuotientProfilerError("per-pair zlib compression failed") from exc


def _class_conditioned_stats(
    residual: np.ndarray,
    labels: np.ndarray,
    *,
    plane: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    channels = residual.shape[-1]
    for class_id in range(5):
        class_mask = labels == class_id
        for channel in range(channels):
            values = residual[..., channel][class_mask]
            if values.size == 0:
                rows.append(
                    {
                        "plane": plane,
                        "class_id": class_id,
                        "channel": channel,
                        "support_pixels": 0,
                        "stats": None,
                    }
                )
                continue
            rows.append(
                {
                    "plane": plane,
                    "class_id": class_id,
                    "channel": channel,
                    "support_pixels": int(values.size),
                    "stats": _symbol_stats(values, minimum=-255, maximum=255),
                }
            )
    return rows


def profile_conditional_quotient_chunk(
    base: PlaneChunk,
    target: PlaneChunk,
    fresh_batch16_labels: np.ndarray,
    *,
    run_binding_sha256: str,
) -> dict[str, Any]:
    """Measure one exact chunk and return sufficient resumable statistics."""

    _require_sha256(run_binding_sha256, field="run_binding_sha256")
    if not isinstance(base, PlaneChunk) or not isinstance(target, PlaneChunk):
        raise ConditionalQuotientProfilerError("base and target must use the C0B PlaneChunk seam")
    if (
        base.chunk_index != target.chunk_index
        or base.pair_ids != target.pair_ids
        or base.y0.shape != target.y0.shape
        or base.y1.shape != target.y1.shape
    ):
        raise ConditionalQuotientProfilerError("base and target chunk coordinates differ")
    labels = np.asarray(fresh_batch16_labels)
    if labels.dtype != np.uint8 or labels.shape != base.y0.shape[:-1] or int(labels.min()) < 0 or int(labels.max()) > 4:
        raise ConditionalQuotientProfilerError("fresh labels must be uint8 [0,4] at the exact scorer grid")

    base0 = base.y0
    base1 = base.y1
    target0 = target.y0
    target1 = target.y1
    residual0 = target0.astype(np.int16) - base0.astype(np.int16)
    residual1 = target1.astype(np.int16) - base1.astype(np.int16)
    if not (
        np.array_equal((base0.astype(np.int16) + residual0).astype(np.uint8), target0)
        and np.array_equal((base1.astype(np.int16) + residual1).astype(np.uint8), target1)
    ):
        raise ConditionalQuotientProfilerError("signed residual did not reconstruct the selected planes")

    common = np.floor_divide(residual0.astype(np.int32) + residual1.astype(np.int32), 2).astype(np.int16)
    differential = (residual1.astype(np.int32) - residual0.astype(np.int32)).astype(np.int16)
    reconstructed0 = common.astype(np.int32) - np.floor_divide(differential.astype(np.int32), 2)
    reconstructed1 = reconstructed0 + differential.astype(np.int32)
    if not (
        np.array_equal(reconstructed0.astype(np.int16), residual0)
        and np.array_equal(reconstructed1.astype(np.int16), residual1)
    ):
        raise ConditionalQuotientProfilerError("common/differential factorization is not exact")

    temporal0 = residual0.copy()
    temporal1 = residual1.copy()
    if len(base.pair_ids) > 1:
        temporal0[1:] = residual0[1:] - residual0[:-1]
        temporal1[1:] = residual1[1:] - residual1[:-1]
        check0 = np.cumsum(temporal0.astype(np.int32), axis=0)
        check1 = np.cumsum(temporal1.astype(np.int32), axis=0)
    else:
        check0 = temporal0.astype(np.int32)
        check1 = temporal1.astype(np.int32)
    if not (np.array_equal(check0.astype(np.int16), residual0) and np.array_equal(check1.astype(np.int16), residual1)):
        raise ConditionalQuotientProfilerError("chunk-reset pair-temporal factorization is not exact")

    xor0 = np.bitwise_xor(base0, target0)
    xor1 = np.bitwise_xor(base1, target1)
    if not (
        np.array_equal(np.bitwise_xor(base0, xor0), target0) and np.array_equal(np.bitwise_xor(base1, xor1), target1)
    ):
        raise ConditionalQuotientProfilerError("C0B XOR coordinate is not exact")
    interleaved_xor = np.stack((xor0, xor1), axis=-1)
    seg_primary = xor1
    pose_enhancement = np.bitwise_xor(target1, target0)
    decoded1 = np.bitwise_xor(base1, seg_primary)
    decoded0 = np.bitwise_xor(decoded1, pose_enhancement)
    if not (np.array_equal(decoded0, target0) and np.array_equal(decoded1, target1)):
        raise ConditionalQuotientProfilerError("Seg-primary/Pose-enhancement layering is not exact")

    representations = {
        "c0b_xor_separate": _representation(
            {"y0_xor": (xor0, 0, 255), "y1_xor": (xor1, 0, 255)},
            (xor0.tobytes(order="C"), xor1.tobytes(order="C")),
        ),
        "c0b_xor_interleaved": _representation(
            {"y0_y1_interleaved_xor": (interleaved_xor, 0, 255)},
            (interleaved_xor.tobytes(order="C"),),
        ),
        "signed_residual_separate": _representation(
            {
                "y0_signed": (residual0, -255, 255),
                "y1_signed": (residual1, -255, 255),
            },
            (_int16_le(residual0), _int16_le(residual1)),
        ),
        "common_differential_signed": _representation(
            {
                "common_signed": (common, -255, 255),
                "differential_signed": (differential, -510, 510),
            },
            (_int16_le(common), _int16_le(differential)),
        ),
        "pair_temporal_signed_chunk_reset": _representation(
            {
                "y0_pair_delta": (temporal0, -510, 510),
                "y1_pair_delta": (temporal1, -510, 510),
            },
            (_int16_le(temporal0), _int16_le(temporal1)),
        ),
        "seg_y1_plus_pose_y0_xor_y1": _representation(
            {
                "seg_primary_y1_xor": (seg_primary, 0, 255),
                "pose_enhancement_y0_xor_y1": (pose_enhancement, 0, 255),
            },
            (seg_primary.tobytes(order="C"), pose_enhancement.tobytes(order="C")),
        ),
    }
    if tuple(representations) != REPRESENTATION_IDS:
        raise ConditionalQuotientProfilerError("representation registry order drifted")

    pair_rows: list[dict[str, Any]] = []
    for local_index, pair_id in enumerate(base.pair_ids):
        pair_xor0 = xor0[local_index]
        pair_xor1 = xor1[local_index]
        pair_interleaved = interleaved_xor[local_index]
        pair_residual0 = residual0[local_index]
        pair_residual1 = residual1[local_index]
        pair_common = common[local_index]
        pair_differential = differential[local_index]
        pair_temporal0 = temporal0[local_index]
        pair_temporal1 = temporal1[local_index]
        pair_seg = seg_primary[local_index]
        pair_pose = pose_enhancement[local_index]
        seg_effect_energy = int(np.square(pair_residual1.astype(np.int64)).sum())
        pose_effect_energy = int(np.square(pair_residual0.astype(np.int64)).sum())
        pair_rows.append(
            {
                "pair_id": pair_id,
                "zlib9_marginal_bytes": {
                    "c0b_xor_separate": _pair_zlib_bytes((pair_xor0.tobytes(order="C"), pair_xor1.tobytes(order="C"))),
                    "c0b_xor_interleaved": _pair_zlib_bytes((pair_interleaved.tobytes(order="C"),)),
                    "signed_residual_separate": _pair_zlib_bytes(
                        (_int16_le(pair_residual0), _int16_le(pair_residual1))
                    ),
                    "common_differential_signed": _pair_zlib_bytes(
                        (_int16_le(pair_common), _int16_le(pair_differential))
                    ),
                    "pair_temporal_signed_chunk_reset": _pair_zlib_bytes(
                        (_int16_le(pair_temporal0), _int16_le(pair_temporal1))
                    ),
                    "seg_y1_plus_pose_y0_xor_y1": _pair_zlib_bytes(
                        (pair_seg.tobytes(order="C"), pair_pose.tobytes(order="C"))
                    ),
                },
                "changed_values": {
                    "y0": int(np.count_nonzero(pair_residual0)),
                    "y1": int(np.count_nonzero(pair_residual1)),
                    "seg_primary_y1": int(np.count_nonzero(pair_seg)),
                    "pose_enhancement_y0_xor_y1": int(np.count_nonzero(pair_pose)),
                },
                "functional_operator_groups": {
                    "group_ids": [
                        "seg_primary_output_delta_y1",
                        "pose_enhancement_output_delta_y0_given_y1",
                    ],
                    "ambient_unweighted_gram": [
                        [seg_effect_energy, 0],
                        [0, pose_effect_energy],
                    ],
                    "ambient_cross_term_is_zero_by_disjoint_output_coordinates": True,
                    "task_weighted_cross_term_available": False,
                    "block_zlib9_bytes": {
                        "seg_primary_output_delta_y1": _pair_zlib_bytes((pair_seg.tobytes(order="C"),)),
                        "pose_enhancement_output_delta_y0_given_y1": _pair_zlib_bytes((pair_pose.tobytes(order="C"),)),
                    },
                    "whole_archive_zip_marginal_available": False,
                },
            }
        )

    class_rows = [
        *_class_conditioned_stats(residual0, labels, plane="y0"),
        *_class_conditioned_stats(residual1, labels, plane="y1"),
    ]
    body = {
        "schema": CHUNK_SCHEMA,
        "run_binding_sha256": run_binding_sha256,
        "chunk_index": base.chunk_index,
        "pair_ids": list(base.pair_ids),
        "base": {
            "y0_sha256": _sha256(base0.tobytes(order="C")),
            "y1_sha256": _sha256(base1.tobytes(order="C")),
        },
        "target": {
            "y0_sha256": _sha256(target0.tobytes(order="C")),
            "y1_sha256": _sha256(target1.tobytes(order="C")),
        },
        "fresh_batch16_labels": {
            "sha256": _sha256(labels.tobytes(order="C")),
            "conditioning_only": True,
            "serialized_in_candidate": False,
        },
        "representations": representations,
        "pair_marginals": pair_rows,
        "class_conditioned_signed_residual": class_rows,
        "all_representations_exact_roundtrip": True,
        "temporal_reset_policy": "first_pair_absolute_then_pair_delta_within_each_chunk",
        "score_claim": False,
        "promotion_eligible": False,
        "candidate_payload": False,
    }
    body["chunk_receipt_sha256"] = _sha256(_canonical_mapping(body, label="chunk profile"))
    return body


def _validate_chunk_receipt(
    value: Mapping[str, Any],
    *,
    run_binding_sha256: str,
    expected_chunk_index: int,
    expected_pair_ids: tuple[int, ...],
) -> dict[str, Any]:
    row = dict(value)
    if (
        row.get("schema") != CHUNK_SCHEMA
        or row.get("run_binding_sha256") != run_binding_sha256
        or row.get("chunk_index") != expected_chunk_index
        or row.get("pair_ids") != list(expected_pair_ids)
        or row.get("all_representations_exact_roundtrip") is not True
        or row.get("score_claim") is not False
        or row.get("promotion_eligible") is not False
        or row.get("candidate_payload") is not False
    ):
        raise ConditionalQuotientProfilerError("chunk checkpoint contract differs")
    expected_sha = row.get("chunk_receipt_sha256")
    _require_sha256(expected_sha, field="chunk_receipt_sha256")
    body = {key: item for key, item in row.items() if key != "chunk_receipt_sha256"}
    if _sha256(_canonical_mapping(body, label="chunk profile body")) != expected_sha:
        raise ConditionalQuotientProfilerError("chunk checkpoint self-hash differs")
    representations = row.get("representations")
    if not isinstance(representations, dict) or set(representations) != set(REPRESENTATION_IDS):
        raise ConditionalQuotientProfilerError("chunk representation registry differs")
    return row


def _combine_symbol_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ConditionalQuotientProfilerError("cannot aggregate empty symbol statistics")
    minimum = rows[0]["minimum"]
    maximum = rows[0]["maximum"]
    counts = np.zeros(maximum - minimum + 1, dtype=np.int64)
    total = 0
    nonzero_values = 0
    sum_abs = 0
    sum_squares = 0
    zero_runs = 0
    zero_run_max = 0
    zero_values = 0
    for row in rows:
        if row["minimum"] != minimum or row["maximum"] != maximum:
            raise ConditionalQuotientProfilerError("symbol histogram ranges differ across chunks")
        histogram = np.asarray(row["histogram_counts"], dtype=np.int64)
        if histogram.shape != counts.shape or np.any(histogram < 0):
            raise ConditionalQuotientProfilerError("symbol histogram shape/count differs")
        counts += histogram
        total += row["value_count"]
        nonzero_values += row["nonzero_value_count"]
        sum_abs += row["sum_abs"]
        sum_squares += row["sum_squares"]
        runs = row["runs"]
        zero_runs += runs["zero_run_count"]
        zero_run_max = max(zero_run_max, runs["zero_run_max"])
        zero_values += runs["zero_values"]
    if int(counts.sum()) != total:
        raise ConditionalQuotientProfilerError("aggregate symbol histogram mass differs")
    supported = counts[counts > 0]
    probabilities = supported.astype(np.float64) / total
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    occupied = np.flatnonzero(counts)
    return {
        "minimum": minimum,
        "maximum": maximum,
        "observed_minimum": int(occupied[0] + minimum),
        "observed_maximum": int(occupied[-1] + minimum),
        "value_count": total,
        "nonzero_value_count": nonzero_values,
        "sum_abs": sum_abs,
        "sum_squares": sum_squares,
        "entropy_bits_per_value": entropy,
        "entropy_lower_bound_bytes": math.ceil(entropy * total / 8.0),
        "histogram_counts": counts.tolist(),
        "runs": {
            "values": total,
            "zero_values": zero_values,
            "zero_fraction": zero_values / total,
            "zero_run_count": zero_runs,
            "zero_run_mean": zero_values / zero_runs if zero_runs else 0.0,
            "zero_run_max": zero_run_max,
            "aggregation_note": "chunk boundaries are explicit run resets",
        },
    }


def _quantile_summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        raise ConditionalQuotientProfilerError("cannot summarize empty marginal bytes")
    array = np.asarray(values, dtype=np.int64)
    return {
        "count": int(array.size),
        "sum": int(array.sum()),
        "minimum": int(array.min()),
        "mean": float(array.mean()),
        "p50": int(np.percentile(array, 50, method="higher")),
        "p90": int(np.percentile(array, 90, method="higher")),
        "p99": int(np.percentile(array, 99, method="higher")),
        "maximum": int(array.max()),
    }


def _functional_operator_surface(
    all_pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    group_ids = (
        "seg_primary_output_delta_y1",
        "pose_enhancement_output_delta_y0_given_y1",
    )
    total_seg_energy = 0
    total_pose_energy = 0
    seg_block_bytes: list[int] = []
    pose_block_bytes: list[int] = []
    for row in all_pair_rows:
        groups = row.get("functional_operator_groups")
        if not isinstance(groups, Mapping) or groups.get("group_ids") != list(group_ids):
            raise ConditionalQuotientProfilerError("functional operator group registry differs")
        gram = groups.get("ambient_unweighted_gram")
        if (
            not isinstance(gram, list)
            or len(gram) != 2
            or any(not isinstance(component, list) or len(component) != 2 for component in gram)
            or gram[0][1] != 0
            or gram[1][0] != 0
        ):
            raise ConditionalQuotientProfilerError("ambient functional group Gram shape differs")
        total_seg_energy += gram[0][0]
        total_pose_energy += gram[1][1]
        block_bytes = groups.get("block_zlib9_bytes")
        if not isinstance(block_bytes, Mapping):
            raise ConditionalQuotientProfilerError("functional group byte marginals are absent")
        seg_block_bytes.append(block_bytes[group_ids[0]])
        pose_block_bytes.append(block_bytes[group_ids[1]])
    trace = total_seg_energy + total_pose_energy
    return {
        "proposal_generation_only": True,
        "source_compatible": True,
        "function_space_groups": {
            group_ids[0]: {
                "output_support": "Y1",
                "scorer_visibility": ["SegNet", "PoseNet"],
                "ambient_squared_energy": total_seg_energy,
                "per_pair_block_zlib9_bytes": _quantile_summary(seg_block_bytes),
            },
            group_ids[1]: {
                "output_support": "Y0 conditioned on decoded Y1",
                "scorer_visibility": ["PoseNet"],
                "ambient_squared_energy": total_pose_energy,
                "per_pair_block_zlib9_bytes": _quantile_summary(pose_block_bytes),
            },
        },
        "ambient_unweighted_group_gram": [
            [total_seg_energy, 0],
            [0, total_pose_energy],
        ],
        "ambient_unweighted_largest_energy_fraction": (
            max(total_seg_energy, total_pose_energy) / trace if trace else 0.0
        ),
        "ambient_gram_authority": (
            "EXACT_OUTPUT_SPACE_DIAGNOSTIC_ONLY; disjoint plane coordinates force a zero "
            "cross-term and therefore cannot reveal evaluator coupling"
        ),
        "task_weighted_operator": {
            "status": "BLOCKED_MISSING_SCORER_COSTATE_EFFECTS",
            "required_inputs": [
                "receiver-R-frozen-SegNet effect vectors or JVP/VJP costates for Y1 atoms",
                "receiver-R-frozen-PoseNet effect vectors or JVP/VJP costates for Y0|Y1 atoms",
                "same-object exact archive ZIP byte marginals for proposed bundles",
            ],
            "missing_outputs": [
                "task_weighted_quotient_atom_gram",
                "low_rank_merge_candidates",
                "prune_candidates",
                "macro_segment_eviction_candidates",
            ],
            "reason": (
                "planes and class labels identify output coordinates but not scorer-induced "
                "function distance, cross-channel coupling, or whole-object score effect"
            ),
        },
        "hope_compatibility_fences": {
            "ph1_batchnorm_closed_forms_allowed": False,
            "reason": "V9/V15 source uses a FiLM-conditioned tanh(sin) trunk, not the paper's PH-1/BN surface",
            "static_parameter_count_as_rate_allowed": False,
            "rate_admission": "exact coded block bytes for proposals; exact whole-archive ZIP bytes for verdict",
            "whole_object_scorer_verification_required": True,
        },
    }


def aggregate_conditional_quotient_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    config: ConditionalQuotientProfileConfigV1,
    input_binding: Mapping[str, Any],
    run_binding_sha256: str,
) -> dict[str, Any]:
    """Build the exact full-population aggregate from sufficient chunk stages."""

    binding = _validate_input_binding(input_binding, config=config)
    expected_chunk_count = (config.pair_count + config.chunk_pairs - 1) // config.chunk_pairs
    if len(chunks) != expected_chunk_count:
        raise ConditionalQuotientProfilerError("chunk stage count differs from config")
    validated: list[dict[str, Any]] = []
    expected_pair = 0
    for chunk_index, chunk in enumerate(chunks):
        pair_ids = tuple(
            range(
                expected_pair,
                min(expected_pair + config.chunk_pairs, config.pair_count),
            )
        )
        row = _validate_chunk_receipt(
            chunk,
            run_binding_sha256=run_binding_sha256,
            expected_chunk_index=chunk_index,
            expected_pair_ids=pair_ids,
        )
        validated.append(row)
        expected_pair += len(pair_ids)
    if expected_pair != config.pair_count:
        raise ConditionalQuotientProfilerError("chunk stages do not cover the full configured population")

    representation_rows: dict[str, Any] = {}
    all_pair_rows: list[dict[str, Any]] = []
    for row in validated:
        all_pair_rows.extend(row["pair_marginals"])
    if [row["pair_id"] for row in all_pair_rows] != list(range(config.pair_count)):
        raise ConditionalQuotientProfilerError("per-pair marginal chronology differs")

    for representation_id in REPRESENTATION_IDS:
        source = [row["representations"][representation_id] for row in validated]
        component_names = tuple(source[0]["components"])
        if any(tuple(row["components"]) != component_names for row in source):
            raise ConditionalQuotientProfilerError("representation component registry differs")
        codec_totals = {
            field: sum(row["codec_sizes"][field] for row in source)
            for field in (
                "part_count",
                "raw_bytes",
                "zlib9_block_bytes",
                "c0b_lzma_block_bytes",
            )
        }
        components = {
            component: _combine_symbol_stats([row["components"][component] for row in source])
            for component in component_names
        }
        marginals = [row["zlib9_marginal_bytes"][representation_id] for row in all_pair_rows]
        representation_rows[representation_id] = {
            "codec_sizes_chunk_block_sum": codec_totals,
            "components": components,
            "per_pair_zlib9_marginal_summary": _quantile_summary(marginals),
        }

    class_keys: list[tuple[str, int, int]] = []
    first_class_rows = validated[0]["class_conditioned_signed_residual"]
    for row in first_class_rows:
        class_keys.append((row["plane"], row["class_id"], row["channel"]))
    aggregate_class: list[dict[str, Any]] = []
    for key in class_keys:
        matches = []
        support = 0
        for chunk in validated:
            candidates = [
                row
                for row in chunk["class_conditioned_signed_residual"]
                if (row["plane"], row["class_id"], row["channel"]) == key
            ]
            if len(candidates) != 1:
                raise ConditionalQuotientProfilerError("class-conditioned key coverage differs")
            candidate = candidates[0]
            support += candidate["support_pixels"]
            if candidate["stats"] is not None:
                matches.append(candidate["stats"])
        aggregate_class.append(
            {
                "plane": key[0],
                "class_id": key[1],
                "channel": key[2],
                "support_pixels": support,
                "stats": _combine_symbol_stats(matches) if matches else None,
            }
        )

    planning = binding["current_batch16_planning_coordinate"]
    historical = binding["historical_ms1_batch32_counterfactual"]
    headroom_frontier = planning["headroom_bytes_to_effective_frontier"]
    headroom_015 = planning["headroom_bytes_to_sub_0_15"]
    exact_size_rows = []
    for representation_id, row in representation_rows.items():
        sizes = row["codec_sizes_chunk_block_sum"]
        for compressor_id, field in (
            ("zlib9_block", "zlib9_block_bytes"),
            ("c0b_lzma_block", "c0b_lzma_block_bytes"),
        ):
            payload_bytes = sizes[field]
            exact_size_rows.append(
                {
                    "representation_id": representation_id,
                    "compressor_id": compressor_id,
                    "payload_bytes": payload_bytes,
                    "fits_current_batch16_headroom_to_effective_frontier": payload_bytes <= headroom_frontier,
                    "fits_current_batch16_headroom_to_0_15": payload_bytes <= headroom_015,
                    "fits_historical_batch32_headroom_to_effective_frontier": (
                        payload_bytes <= historical["headroom_bytes_to_effective_frontier"]
                    ),
                    "fits_historical_batch32_headroom_to_0_15": (
                        payload_bytes <= historical["headroom_bytes_to_sub_0_15"]
                    ),
                }
            )
    exact_size_rows.sort(
        key=lambda row: (
            row["payload_bytes"],
            row["representation_id"],
            row["compressor_id"],
        )
    )
    best = exact_size_rows[0]
    geometry_aligned = binding["current_planning_matches_upstream_batch_geometry"]
    if best["fits_current_batch16_headroom_to_effective_frontier"]:
        conclusion = (
            "At least one tested exact block-coded quotient fits the canonical batch-16 "
            "planning headroom; build a receiver-closed archive and exact-evaluate the "
            "same object before any score inference."
        )
    else:
        conclusion = (
            "No tested exact local/block quotient fits the canonical batch-16 planning "
            "payload headroom. A nonlocal analytic/generative factorization or a learned "
            "irreducible quotient is the next representation class; learned necessity is not proven."
        )
    hook_coverage = {
        "1": {
            "hook": "SENSITIVITY_MAP",
            "status": "TYPED_INCOMPATIBLE_NO_GRADIENT_SURFACE",
            "payload": "class_conditioned_signed_residual",
        },
        "2": {
            "hook": "PARETO_CONSTRAINT",
            "status": "CONDITIONAL_EXACT_BYTE_ROWS_ONLY",
            "payload": "conditional_budget_arbitration.exact_tested_basis_rows",
        },
        "3": {
            "hook": "BIT_ALLOCATOR",
            "status": "READY_ENCODER_DIAGNOSTIC_ONLY",
            "payload": "per_pair_marginals",
        },
        "4": {
            "hook": "CATHEDRAL_AUTOPILOT_DISPATCH",
            "status": "TIER_A_OBSERVABILITY_NO_DISPATCH",
            "payload": "conditional_budget_arbitration.best_tested_exact_basis",
        },
        "5": {
            "hook": "CONTINUAL_LEARNING_POSTERIOR",
            "status": "BLOCKED_NONAUTHORITY_LOCAL_BATCH16",
            "payload": None,
        },
        "6": {
            "hook": "PROBE_DISAMBIGUATOR",
            "status": "READY_FOR_OWNER_REVIEW",
            "payload": "conditional_budget_arbitration.conclusion",
        },
    }
    aggregate_body = {
        "schema": AGGREGATE_SCHEMA,
        "run_binding_sha256": run_binding_sha256,
        "config": config.as_mapping(),
        "input_binding": binding,
        "chunk_count": len(validated),
        "chunk_receipt_sha256": [row["chunk_receipt_sha256"] for row in validated],
        "chunk_receipt_root_sha256": _sha256(
            _canonical_mapping(
                {"chunk_receipt_sha256": [row["chunk_receipt_sha256"] for row in validated]},
                label="chunk receipt root",
            )
        ),
        "representations": representation_rows,
        "per_pair_marginals": all_pair_rows,
        "functional_operator_proposal_surface": _functional_operator_surface(all_pair_rows),
        "class_conditioned_signed_residual": aggregate_class,
        "conditional_budget_arbitration": {
            "authority": planning["authority"],
            "selected_plane_origin_scorer_batch_size": binding["selected_plane_origin_scorer_batch_size"],
            "current_planning_scorer_batch_size": binding["current_planning_scorer_batch_size"],
            "upstream_default_scorer_batch_size": binding["upstream_default_scorer_batch_size"],
            "current_planning_matches_upstream_batch_geometry": geometry_aligned,
            "frontier_feasibility_inference_allowed": False,
            "reason_frontier_inference_forbidden": (
                "Profiler output is encoder-only exact-byte diagnostics, not a receiver-closed "
                "archive/eval row; canonical batch-16 debt is a planning coordinate, not a new score."
            ),
            "effective_frontier_score_at_profile_start": planning["effective_frontier_score"],
            "current_batch16_headroom_bytes_to_effective_frontier": headroom_frontier,
            "current_batch16_headroom_bytes_to_sub_0_15": headroom_015,
            "historical_batch32_headroom_bytes_to_effective_frontier": historical[
                "headroom_bytes_to_effective_frontier"
            ],
            "historical_batch32_headroom_bytes_to_sub_0_15": historical["headroom_bytes_to_sub_0_15"],
            "planning_coordinate_premise": binding["planning_coordinate_premise"],
            "canonical_batch16_debt_receipt": binding["canonical_batch16_debt_receipt"],
            "independent_batch16_replay_corroboration": binding["independent_batch16_replay_corroboration"],
            "exact_tested_basis_rows": exact_size_rows,
            "best_tested_exact_basis": best,
            "learned_irreducible_quotient_mandatory": False,
            "learned_necessity_status": "NOT_PROVEN_BY_FINITE_EXACT_BASIS_RACE",
            "conclusion": conclusion,
        },
        "downstream_hook_coverage": hook_coverage,
        "resumable": True,
        "per_stage_checkpoints": True,
        "full_population_profiled": config.pair_count == PUBLIC_PAIR_COUNT,
        "score_claim": False,
        "promotion_eligible": False,
        "candidate_payload": False,
        "pointer_moved": False,
    }
    aggregate_body["aggregate_receipt_sha256"] = _sha256(_canonical_mapping(aggregate_body, label="aggregate profile"))
    return aggregate_body


def run_conditional_quotient_profile(
    *,
    config: ConditionalQuotientProfileConfigV1,
    input_binding: Mapping[str, Any],
    work_root: Path,
    chunk_loader: ChunkLoader,
) -> dict[str, Any]:
    """Run or resume every immutable chunk stage, then write the aggregate."""

    binding = _validate_input_binding(input_binding, config=config)
    root = Path(work_root).resolve(strict=False)
    if root.exists() and not config.resume and any(root.iterdir()):
        raise ConditionalQuotientProfilerError("fresh profile refuses a non-empty work root")
    try:
        preflight = storage_preflight(
            root,
            required_bytes=1 << 30,
            test_only_small_fixture=config.test_only_small_fixture,
            allow_local_storage=config.allow_local_storage,
        )
    except SemanticQuotientError as exc:
        raise ConditionalQuotientProfilerError("profile storage preflight refused") from exc
    root.mkdir(parents=True, exist_ok=True)
    stable_preflight = {
        key: preflight[key]
        for key in (
            "schema",
            "selected_tier",
            "required_bytes",
            "passed",
            "test_only_small_fixture",
            "allow_local_storage",
        )
    }
    run_identity_body = {
        "schema": RUN_SCHEMA,
        "config": config.as_mapping(),
        "input_binding": binding,
        "storage_preflight": stable_preflight,
        "representation_ids": list(REPRESENTATION_IDS),
        "compressor_ids": list(COMPRESSOR_IDS),
        "score_claim": False,
        "promotion_eligible": False,
        "candidate_payload": False,
    }
    run_binding_sha256 = _sha256(_canonical_mapping(run_identity_body, label="profile run identity"))
    stage0 = {**run_identity_body, "run_binding_sha256": run_binding_sha256}
    try:
        write_once_or_equal(
            root / "stage_checkpoints" / "00_inputs.json",
            _canonical_mapping(stage0, label="profile stage 00"),
        )
    except SemanticQuotientError as exc:
        raise ConditionalQuotientProfilerError("profile stage 00 write/resume failed") from exc

    chunks: list[dict[str, Any]] = []
    expected_pair = 0
    chunk_count = (config.pair_count + config.chunk_pairs - 1) // config.chunk_pairs
    for chunk_index in range(chunk_count):
        pair_ids = tuple(
            range(
                expected_pair,
                min(expected_pair + config.chunk_pairs, config.pair_count),
            )
        )
        stage_path = root / "stage_checkpoints" / f"10_chunk_{chunk_index:04d}.json"
        if stage_path.exists():
            if not config.resume:
                raise ConditionalQuotientProfilerError("fresh profile encountered a chunk checkpoint")
            row = _validate_chunk_receipt(
                _read_canonical_mapping(stage_path, label=f"chunk {chunk_index} checkpoint"),
                run_binding_sha256=run_binding_sha256,
                expected_chunk_index=chunk_index,
                expected_pair_ids=pair_ids,
            )
        else:
            base, target, labels = chunk_loader(chunk_index, pair_ids)
            row = profile_conditional_quotient_chunk(
                base,
                target,
                labels,
                run_binding_sha256=run_binding_sha256,
            )
            row = _validate_chunk_receipt(
                row,
                run_binding_sha256=run_binding_sha256,
                expected_chunk_index=chunk_index,
                expected_pair_ids=pair_ids,
            )
            try:
                write_once_or_equal(
                    stage_path,
                    _canonical_mapping(row, label=f"chunk {chunk_index} checkpoint"),
                )
            except SemanticQuotientError as exc:
                raise ConditionalQuotientProfilerError("chunk checkpoint write failed") from exc
        chunks.append(row)
        expected_pair += len(pair_ids)
    aggregate = aggregate_conditional_quotient_chunks(
        chunks,
        config=config,
        input_binding=binding,
        run_binding_sha256=run_binding_sha256,
    )
    try:
        write_once_or_equal(
            root / "stage_checkpoints" / "20_aggregate.json",
            _canonical_mapping(aggregate, label="aggregate profile"),
        )
        write_once_or_equal(
            root / "aggregate_receipt.json",
            _canonical_mapping(aggregate, label="aggregate receipt"),
        )
    except SemanticQuotientError as exc:
        raise ConditionalQuotientProfilerError("aggregate checkpoint write/resume failed") from exc
    return aggregate


__all__ = [
    "AGGREGATE_SCHEMA",
    "CHUNK_SCHEMA",
    "COMPRESSOR_IDS",
    "CONFIG_SCHEMA",
    "EVIDENCE_AXIS",
    "INPUT_BINDING_SCHEMA",
    "PUBLIC_CHANNELS",
    "PUBLIC_PAIR_COUNT",
    "PUBLIC_SCORER_HW",
    "REPRESENTATION_IDS",
    "UPSTREAM_DEFAULT_BATCH_SIZE",
    "ConditionalQuotientProfileConfigV1",
    "ConditionalQuotientProfilerError",
    "aggregate_conditional_quotient_chunks",
    "profile_conditional_quotient_chunk",
    "run_conditional_quotient_profile",
]
