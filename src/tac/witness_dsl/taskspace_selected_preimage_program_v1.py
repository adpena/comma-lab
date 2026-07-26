# SPDX-License-Identifier: MIT
"""Strict counted program for factorized, coupled task-space preimages.

This module is one strict inner operand packet between the compact original
``CarrierComposeReceiverV1.render_pairs`` V9--V13 scorer-grid grammar and the
exact V10 factor-2 realizer.  It is not the complete selected-solution product
type or an outer archive.  In particular, it does not yet carry the full
V9/PBR/IR/V10 population map, obligation-level COMPLETE/SPARSE ownership,
exclusive Pose/frame-0 ownership, physical outer-archive coding groups,
general interaction hyperedges, or receiver-executable R10 constraints.

It deliberately does *not* embed dense target planes, teacher labels, scorer
artifacts, explicit preimage arrays, or ancestor archive bytes.  The compile
entry point may reopen declared encoder-only dependencies and exact output
bytes, but the V1 packet carries only:

* a verified fresh-compile proof-dependency identity for the V15 carrier
  program (outer archive embedding of the semantic bytes remains an explicitly
  named runner blocker);
* exact batch-16 encoder-target custody metadata without target payload;
* compact analytic residual factors; and
* compact learned irreducible quotient factors with exact sparse pair support.

The decoder is an explicit protocol object, never a name imported from packet
bytes.  A concrete bound adapter composes the existing carrier receiver and
``DisjointResizeOperator``.  The built-in analytic factor uses the existing
parabolic boundary-shearlet primitive and has a real, behavior-changing
receiver implementation; it is not a dense residual wrapper.

No function in this module invokes SegNet, PoseNet, or an evaluator, and no
score or candidate claim is made.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import struct
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

import tac.optimization.direct_description_carrier_compose as _carrier_module
from tac.optimization.direct_description_carrier_compose import (
    BoundaryShearletAtomV1,
    CarrierComposeReceiverV1,
)
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Factor2ExactVerification,
    Uint8LatticeError,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)

PROGRAM_SCHEMA: Final = "tac.taskspace_selected_preimage_program.v1"
ANALYTIC_SCHEMA: Final = "tac.taskspace_analytic_shearlet_residual.v1"
LEARNED_SCHEMA: Final = "tac.taskspace_learned_irreducible_quotient.v1"
MAGIC: Final = b"TSPPV1\x00\x00"
LEARNED_MAGIC: Final = b"LIQPV1\x00\x00"
GENERIC_V10_FACTOR2_DECODER_ID: Final = "tac.v10.disjoint_half_pixel_factor2_uint8_pair_decoder.v1"
V15_SEMANTIC_RECEIVER_ID: Final = (
    "tac.optimization.direct_description_carrier_compose.CarrierComposeReceiverV1.render_pairs.v9_v13"
)
V15_SEMANTIC_COMPILE_RECEIPT_SCHEMA: Final = "ddm_v15_scorer_solved_template_receipt.v1"
V15_SEMANTIC_COMPILE_DERIVATION: Final = "FRESH_CURRENT_SOURCE_DERIVATION_FROM_SEALED_DECLARED_DEPENDENCIES"
SCORER_HEIGHT: Final = 384
SCORER_WIDTH: Final = 512
CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
CHANNELS: Final = 3
PAIR_COUNT_N600: Final = 600

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE: Final = re.compile(r"[A-Za-z0-9_.:+-]{1,192}\Z")
_PROGRAM_HEADER = struct.Struct("<8sI")
_LEARNED_HEADER = struct.Struct("<8sI")
_U32_MAX: Final = (1 << 32) - 1
_COMPILE_INPUT_CLASSES: Final = (
    "CURRENT_SOURCE_ENCODER_IMPLEMENTATION",
    "TYPED_COMPILE_CONFIG",
    "DECLARED_ANCESTOR_ARCHIVE_IDENTITY",
    "DECLARED_TEMPLATE_BANK_IDENTITY",
    "BATCH16_ENCODER_TARGET_CUSTODY_IDENTITY",
)
_FORBIDDEN_ATTESTATIONS: Final = (
    "NO_DENSE_TARGET_PAYLOAD",
    "NO_TEACHER_OR_LABEL_TABLE_PAYLOAD",
    "NO_SCORER_ARTIFACT_PAYLOAD",
    "NO_EXPLICIT_PREIMAGE_PLANE_PAYLOAD",
    "NO_HISTORICAL_ARCHIVE_BYTES",
)
OPEN_SELECTED_SOLUTION_PRODUCT_BLOCKERS: Final = (
    "FRESH_SEMANTIC_ARCHIVE_OUTER_EMBEDDING_AND_RECEIPT_CUSTODY_OWED",
    "G46_COMPILE_READY_TARGET_CUSTODY_REOPEN_BY_COMPILER_OWED",
    "FULL_V9_PBR_IR_V10_PAIR_POPULATION_MAP_OWED",
    "OBLIGATION_LEVEL_COMPLETE_OR_SPARSE_OWNED_RECEIPT_OWED",
    "EXCLUSIVE_V9_POSE6_FRAME0_RESIDUAL_REVERSE_CAUSAL_OWNERSHIP_OWED",
    "ACTUAL_EXPLICIT_PREIMAGE_PACKET_REOPEN_OR_CLAIM_REMOVAL_OWED",
    "OUTER_ARCHIVE_PHYSICAL_CODING_GROUPS_AND_INCIDENCE_OWED",
    "FUNCTIONAL_SPELLING_PHYSICAL_IDENTITY_SEPARATION_OWED",
    "SEALED_AUTHORITY_AND_PROOF_DEPENDENCY_VARIANTS_OWED",
    "RECEIVER_EXECUTABLE_R10_TYPED_CONSUMER_OWED",
    "GENERAL_INTERACTION_HYPEREDGES_OWED",
    "LAYERED_LOSSY_Y1_BASE_AND_Y0_GIVEN_Y1_ENHANCEMENT_ABI_OWED",
)


class TaskspaceSelectedPreimageProgramError(ValueError):
    """A counted packet, factor, identity, or receiver failed closed."""


class SelectedPreimageFactorRoleV1(StrEnum):
    """The only two video-derived factor homes admitted by V1."""

    ANALYTIC_RESIDUAL = "ANALYTIC_RESIDUAL"
    LEARNED_IRREDUCIBLE_QUOTIENT = "LEARNED_IRREDUCIBLE_QUOTIENT"


class SelectedPreimageFactorModeV1(StrEnum):
    """Closed factor semantics; packet bytes never name importable code."""

    SHEARLET_BOUNDARY_TRANSPORT_Q4 = "SHEARLET_BOUNDARY_TRANSPORT_Q4"
    COMPACT_LATENT_QUOTIENT_PLUGIN = "COMPACT_LATENT_QUOTIENT_PLUGIN"


_FACTOR_STAGE_ORDER: Final = {
    SelectedPreimageFactorRoleV1.ANALYTIC_RESIDUAL: 0,
    SelectedPreimageFactorRoleV1.LEARNED_IRREDUCIBLE_QUOTIENT: 1,
}


class SelectedPreimageByteHomeV1(StrEnum):
    """Exact byte home in the submission accounting partition."""

    COUNTED_PACKET_FRAMING = "COUNTED_PACKET_FRAMING"
    COUNTED_ANALYTIC_OPERAND = "COUNTED_ANALYTIC_OPERAND"
    COUNTED_LEARNED_OPERAND = "COUNTED_LEARNED_OPERAND"
    GENERIC_DECODER_CODE_FREE = "GENERIC_DECODER_CODE_FREE"
    ENCODER_ONLY_IDENTITY_NO_PAYLOAD = "ENCODER_ONLY_IDENTITY_NO_PAYLOAD"


class SelectedPreimageLineageClassV1(StrEnum):
    """Lineage is orthogonal to physical byte home."""

    FRESH_ORIGINAL_SEMANTIC_COMPILE_IDENTITY = "FRESH_ORIGINAL_SEMANTIC_COMPILE_IDENTITY"
    GENERIC_NON_VIDEO_DECODER = "GENERIC_NON_VIDEO_DECODER"
    VIDEO_DERIVED_ANALYTIC_FACTOR = "VIDEO_DERIVED_ANALYTIC_FACTOR"
    VIDEO_DERIVED_LEARNED_IRREDUCIBLE_FACTOR = "VIDEO_DERIVED_LEARNED_IRREDUCIBLE_FACTOR"
    ENCODER_TARGET_CUSTODY_IDENTITY_ONLY = "ENCODER_TARGET_CUSTODY_IDENTITY_ONLY"


class SelectedPreimageFrameSelectorV1(StrEnum):
    """Chronological member(s) affected by an analytic factor."""

    Y0 = "Y0"
    Y1 = "Y1"
    BOTH = "BOTH"


class ForbiddenSelectedPreimagePayloadClassV1(StrEnum):
    """Payload classes that must never enter this packet family."""

    DENSE_TARGET = "DENSE_TARGET"
    TEACHER_OR_LABEL_TABLE = "TEACHER_OR_LABEL_TABLE"
    SCORER_ARTIFACT = "SCORER_ARTIFACT"
    EXPLICIT_PREIMAGE_PLANE = "EXPLICIT_PREIMAGE_PLANE"
    HISTORICAL_ARCHIVE_COPY = "HISTORICAL_ARCHIVE_COPY"


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _source_sha256(value: object) -> str:
    source = Path(str(inspect.getsourcefile(value) or "")).resolve()
    if source.suffix != ".py" or not source.is_file():
        raise TaskspaceSelectedPreimageProgramError("receiver identity is not backed by exact Python source bytes")
    return _sha256(source.read_bytes())


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TaskspaceSelectedPreimageProgramError("value is not finite canonical ASCII JSON") from exc


def _decode_canonical_object(payload: bytes, *, label: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload:
        raise TaskspaceSelectedPreimageProgramError(f"{label} must be nonempty exact bytes")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TaskspaceSelectedPreimageProgramError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except TaskspaceSelectedPreimageProgramError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TaskspaceSelectedPreimageProgramError(f"{label} is not strict ASCII JSON") from exc
    if type(value) is not dict or _canonical_json(value) != payload:
        raise TaskspaceSelectedPreimageProgramError(f"{label} is not a canonical JSON object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if type(value) is not dict or set(value) != expected:
        raise TaskspaceSelectedPreimageProgramError(f"{label} fields differ from the closed schema")


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise TaskspaceSelectedPreimageProgramError(f"{label} must be canonical lowercase SHA-256")
    return value


def _require_id(value: object, *, label: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise TaskspaceSelectedPreimageProgramError(f"{label} must be a closed printable ASCII identifier")
    return value


def _require_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        suffix = "" if maximum is None else f"..{maximum}"
        raise TaskspaceSelectedPreimageProgramError(f"{label} must be an exact integer in {minimum}{suffix}")
    return value


def _require_active_pair_ranges(
    value: object,
    *,
    factor_start: int,
    factor_stop: int,
) -> tuple[tuple[int, int], ...]:
    """Return canonical, disjoint support for one learned factor.

    The factor interval is a custody envelope, not a claim that every pair in
    the envelope consumes the factor.  Exact active ranges keep a sparse
    learned section from being falsely rejected as locally inert outside its
    support.
    """

    if type(value) not in {list, tuple} or not value:
        raise TaskspaceSelectedPreimageProgramError("learned quotient requires nonempty active pair ranges")
    ranges: list[tuple[int, int]] = []
    for row in value:
        if type(row) not in {list, tuple} or len(row) != 2:
            raise TaskspaceSelectedPreimageProgramError("learned active pair range must be one exact [start,stop) pair")
        start = _require_int(
            row[0],
            label="learned.active_pair_range.start",
            minimum=factor_start,
            maximum=factor_stop - 1,
        )
        stop = _require_int(
            row[1],
            label="learned.active_pair_range.stop",
            minimum=factor_start + 1,
            maximum=factor_stop,
        )
        if start >= stop:
            raise TaskspaceSelectedPreimageProgramError("learned active pair range is empty or reversed")
        if ranges and start <= ranges[-1][1]:
            raise TaskspaceSelectedPreimageProgramError(
                "learned active pair ranges must be ordered, disjoint, and adjacency-merged"
            )
        ranges.append((start, stop))
    return tuple(ranges)


def _require_u8_rgb(value: object, *, label: str) -> tuple[int, int, int]:
    if (
        type(value) not in {list, tuple}
        or len(value) != 3
        or any(type(channel) is not int or not 0 <= channel <= 255 for channel in value)
    ):
        raise TaskspaceSelectedPreimageProgramError(f"{label} must be one uint8 RGB triple")
    return (int(value[0]), int(value[1]), int(value[2]))


def _require_plane(value: object, *, label: str) -> np.ndarray:
    plane = np.asarray(value)
    expected = (SCORER_HEIGHT, SCORER_WIDTH, CHANNELS)
    if plane.dtype != np.uint8 or plane.shape != expected:
        raise TaskspaceSelectedPreimageProgramError(f"{label} must be exact uint8 scorer plane {expected}")
    result = np.ascontiguousarray(plane).copy()
    result.setflags(write=False)
    return result


def _require_camera_frame(value: object, *, label: str) -> np.ndarray:
    frame = np.asarray(value)
    expected = (CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
    if frame.dtype != np.uint8 or frame.shape != expected:
        raise TaskspaceSelectedPreimageProgramError(f"{label} must be exact uint8 camera frame {expected}")
    result = np.ascontiguousarray(frame).copy()
    result.setflags(write=False)
    return result


def refuse_forbidden_selected_preimage_payload(
    payload_class: ForbiddenSelectedPreimagePayloadClassV1 | str,
) -> None:
    """Always refuse a forbidden data class through one typed entry point."""

    try:
        forbidden = ForbiddenSelectedPreimagePayloadClassV1(payload_class)
    except ValueError as exc:
        raise TaskspaceSelectedPreimageProgramError("unknown payload class cannot be admitted") from exc
    raise TaskspaceSelectedPreimageProgramError(f"{forbidden.value} is forbidden in TaskspaceSelectedPreimageProgramV1")


@dataclass(frozen=True, slots=True)
class ScorerTargetCustodyIdentityV1:
    """Batch geometry and hash custody only; no target cells cross the wire."""

    target_custody_receipt_sha256: str
    target_bank_sha256: str
    scorer_batch_size: int = 16
    pair_sequence_length: int = 2
    segnet_frame_index: int = 1
    pair_count: int = PAIR_COUNT_N600
    pairing_policy: str = "NONOVERLAPPING_CONTIGUOUS_PAIRS"
    geometry_authority: str = "UPSTREAM_EVALUATE_DEFAULT_BATCH16"
    target_payload_embedded: bool = False
    historical_batch32_targets_consumed: bool = False

    def __post_init__(self) -> None:
        _require_sha256(
            self.target_custody_receipt_sha256,
            label="target_custody_receipt_sha256",
        )
        _require_sha256(self.target_bank_sha256, label="target_bank_sha256")
        if (
            self.scorer_batch_size != 16
            or self.pair_sequence_length != 2
            or self.segnet_frame_index != 1
            or self.pair_count != PAIR_COUNT_N600
            or self.pairing_policy != "NONOVERLAPPING_CONTIGUOUS_PAIRS"
            or self.geometry_authority != "UPSTREAM_EVALUATE_DEFAULT_BATCH16"
            or self.target_payload_embedded is not False
            or self.historical_batch32_targets_consumed is not False
        ):
            raise TaskspaceSelectedPreimageProgramError(
                "target custody must bind exact upstream batch16 nonoverlapping "
                "last-frame geometry without embedded targets"
            )


@dataclass(frozen=True, slots=True)
class V15SemanticProgramIdentityV1:
    """Sealed proof-dependency identity for one fresh V15 derivation.

    Content identity is deliberately *not* an originality test.  A
    deterministic current-source compile may reproduce an older byte string.
    Freshness is established by reopening the compile receipt, its declared
    dependencies, current producer sources, checkpoints, receiver custody, and
    exact output bytes in :func:`verify_v15_semantic_compile_lineage`.
    """

    fresh_compile_receipt_schema: str
    fresh_compile_receipt_sha256: str
    compile_proof_dependency_sha256: str
    typed_compile_config_sha256: str
    compiler_source_sha256: str
    receiver_source_sha256: str
    compiled_semantic_archive_sha256: str
    compiled_semantic_archive_bytes: int
    source_pair_start: int
    pair_count: int
    declared_compile_dependency_sha256s: tuple[str, ...]
    compile_input_classes: tuple[str, ...] = _COMPILE_INPUT_CLASSES
    receiver_contract_id: str = V15_SEMANTIC_RECEIVER_ID
    compile_derivation: str = V15_SEMANTIC_COMPILE_DERIVATION
    output_archive_was_declared_compile_input: bool = False
    semantic_archive_embedded: bool = False

    def __post_init__(self) -> None:
        _require_id(
            self.fresh_compile_receipt_schema,
            label="fresh_compile_receipt_schema",
        )
        for field_name in (
            "fresh_compile_receipt_sha256",
            "compile_proof_dependency_sha256",
            "typed_compile_config_sha256",
            "compiler_source_sha256",
            "receiver_source_sha256",
            "compiled_semantic_archive_sha256",
        ):
            _require_sha256(getattr(self, field_name), label=field_name)
        _require_int(
            self.compiled_semantic_archive_bytes,
            label="compiled_semantic_archive_bytes",
            minimum=1,
        )
        _require_int(
            self.source_pair_start,
            label="source_pair_start",
            minimum=0,
            maximum=PAIR_COUNT_N600 - 1,
        )
        _require_int(
            self.pair_count,
            label="pair_count",
            minimum=1,
            maximum=PAIR_COUNT_N600,
        )
        if self.source_pair_start + self.pair_count > PAIR_COUNT_N600:
            raise TaskspaceSelectedPreimageProgramError("semantic pair window escapes n600")
        if (
            type(self.declared_compile_dependency_sha256s) is not tuple
            or not self.declared_compile_dependency_sha256s
            or tuple(sorted(set(self.declared_compile_dependency_sha256s))) != self.declared_compile_dependency_sha256s
        ):
            raise TaskspaceSelectedPreimageProgramError(
                "declared compile dependencies must be a nonempty canonical SHA set"
            )
        for index, digest in enumerate(self.declared_compile_dependency_sha256s):
            _require_sha256(digest, label=f"declared_compile_dependency_sha256s[{index}]")
        if (
            self.fresh_compile_receipt_schema != V15_SEMANTIC_COMPILE_RECEIPT_SCHEMA
            or self.receiver_contract_id != V15_SEMANTIC_RECEIVER_ID
            or self.compile_derivation != V15_SEMANTIC_COMPILE_DERIVATION
            or self.compile_input_classes != _COMPILE_INPUT_CLASSES
            or self.output_archive_was_declared_compile_input is not False
            or self.semantic_archive_embedded is not False
        ):
            raise TaskspaceSelectedPreimageProgramError(
                "V15 identity must bind a sealed current-source derivation, "
                "declared dependencies, and no embedded or input output archive"
            )
        if self.compiled_semantic_archive_sha256 in self.declared_compile_dependency_sha256s:
            raise TaskspaceSelectedPreimageProgramError("semantic output archive is also declared as a compile input")


def verify_v15_semantic_compile_lineage(
    *,
    compile_receipt_bytes: bytes,
    compiled_semantic_archive: bytes,
    producer_root: str | Path,
) -> V15SemanticProgramIdentityV1:
    """Reopen a sealed V15 receipt and derive its packet identity.

    The verifier binds proof lineage rather than using output SHA as a proxy
    for originality.  It accepts deterministic byte identity with an older
    output, but refuses a receipt whose output was itself a declared compile
    dependency, whose current producer sources drifted, or whose n600
    checkpoint/receiver custody cannot be reopened.
    """

    receipt = _decode_canonical_object(compile_receipt_bytes, label="V15 compile receipt")
    if receipt.get("schema") != V15_SEMANTIC_COMPILE_RECEIPT_SCHEMA:
        raise TaskspaceSelectedPreimageProgramError("V15 compile receipt schema is not supported")

    def required_mapping(parent: Mapping[str, Any], key: str, *, label: str) -> dict[str, Any]:
        value = parent.get(key)
        if type(value) is not dict:
            raise TaskspaceSelectedPreimageProgramError(f"{label} must be an exact object")
        return value

    def required_true(parent: Mapping[str, Any], key: str, *, label: str) -> None:
        if parent.get(key) is not True:
            raise TaskspaceSelectedPreimageProgramError(f"{label} must be true")

    config = required_mapping(receipt, "typed_config", label="typed_config")
    typed_config_sha256 = _require_sha256(
        receipt.get("typed_config_sha256"),
        label="typed_config_sha256",
    )
    if _sha256(_canonical_json(config)) != typed_config_sha256:
        raise TaskspaceSelectedPreimageProgramError("typed compile config does not match its sealed SHA")
    run_id = _require_id(receipt.get("run_id"), label="run_id")
    if config.get("run_id") != run_id:
        raise TaskspaceSelectedPreimageProgramError("receipt and typed config run IDs differ")
    source_pair_start = _require_int(
        config.get("pair_start"),
        label="typed_config.pair_start",
        minimum=0,
        maximum=PAIR_COUNT_N600 - 1,
    )
    pair_count = _require_int(
        config.get("pair_count"),
        label="typed_config.pair_count",
        minimum=1,
        maximum=PAIR_COUNT_N600,
    )
    if (
        source_pair_start != 0
        or pair_count != PAIR_COUNT_N600
        or config.get("scorer_batch_size") != 16
        or config.get("research_only") is not True
        or config.get("score_claim") is not False
        or receipt.get("research_only") is not True
        or receipt.get("score_claim") is not False
        or receipt.get("execution_allowed") is not False
    ):
        raise TaskspaceSelectedPreimageProgramError(
            "semantic compile receipt is not the research-only exact n600 batch16 derivation"
        )

    producer_rows = receipt.get("producer_custody")
    if type(producer_rows) is not list:
        raise TaskspaceSelectedPreimageProgramError("producer custody must be a list")
    producer_by_path: dict[str, tuple[int, str]] = {}
    for index, row in enumerate(producer_rows):
        if type(row) is not dict or set(row) != {"bytes", "path", "sha256"}:
            raise TaskspaceSelectedPreimageProgramError(f"producer custody row {index} is malformed")
        relative = row["path"]
        if type(relative) is not str or not relative:
            raise TaskspaceSelectedPreimageProgramError(f"producer custody row {index} path is invalid")
        if relative in producer_by_path:
            raise TaskspaceSelectedPreimageProgramError("producer custody repeats a source path")
        producer_by_path[relative] = (
            _require_int(row["bytes"], label=f"producer_custody[{index}].bytes", minimum=1),
            _require_sha256(row["sha256"], label=f"producer_custody[{index}].sha256"),
        )
    required_producers = (
        "src/tac/optimization/direct_description_carrier_compose.py",
        "tools/measure_ddm_v15_scorer_solved_templates.py",
    )
    if set(producer_by_path) != set(required_producers):
        raise TaskspaceSelectedPreimageProgramError("producer custody differs from the closed V15 producer set")
    root = Path(producer_root).resolve()
    verified_producers: dict[str, str] = {}
    for relative in required_producers:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise TaskspaceSelectedPreimageProgramError("producer custody path escapes its root")
        source = (root / relative_path).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise TaskspaceSelectedPreimageProgramError("producer custody path escapes its root") from exc
        if not source.is_file():
            raise TaskspaceSelectedPreimageProgramError(f"producer source is missing: {relative}")
        source_bytes = source.read_bytes()
        expected_bytes, expected_sha256 = producer_by_path[relative]
        if len(source_bytes) != expected_bytes or _sha256(source_bytes) != expected_sha256:
            raise TaskspaceSelectedPreimageProgramError(f"current producer source drifted: {relative}")
        verified_producers[relative] = expected_sha256

    if type(compiled_semantic_archive) is not bytes or not compiled_semantic_archive:
        raise TaskspaceSelectedPreimageProgramError("compiled semantic archive must be nonempty exact bytes")
    archive_sha256 = _sha256(compiled_semantic_archive)
    archive_bytes = len(compiled_semantic_archive)
    ladder = receipt.get("solved_template_ladder")
    if type(ladder) is not list or len(ladder) != 1 or type(ladder[0]) is not dict:
        raise TaskspaceSelectedPreimageProgramError("V15 receipt must select one solved-template row")
    solved = ladder[0]
    if (
        receipt.get("selected_candidate") != "v15_solved_templates"
        or solved.get("candidate") != "v15_solved_templates"
        or solved.get("archive_sha256") != archive_sha256
        or solved.get("archive_bytes") != archive_bytes
        or solved.get("batch_size") != 16
        or solved.get("batch_count") != 38
        or solved.get("all_batches_checkpointed_and_preserved") is not True
        or solved.get("score_claim") is not False
    ):
        raise TaskspaceSelectedPreimageProgramError(
            "compiled semantic bytes or n600 checkpoint row do not match the sealed receipt"
        )
    batch_digest_chain_sha256 = _require_sha256(
        solved.get("batch_digest_chain_sha256"),
        label="batch_digest_chain_sha256",
    )
    full_p = required_mapping(solved, "full_p_camera_identity", label="full_p_camera_identity")
    if (
        full_p.get("all_camera_bytes_identical") is not True
        or full_p.get("pair_count") != PAIR_COUNT_N600
        or full_p.get("batch_size") != 16
        or full_p.get("batch_count") != 38
    ):
        raise TaskspaceSelectedPreimageProgramError("full-P camera identity proof is incomplete")
    _require_sha256(full_p.get("digest_chain_sha256"), label="full_p_camera_identity.digest_chain_sha256")

    receiver = required_mapping(solved, "receiver_custody", label="receiver_custody")
    if (
        receiver.get("schema") != "direct_description_v15_scorer_solved_template_receiver.v1"
        or receiver.get("archive_sha256") != archive_sha256
        or receiver.get("archive_bytes") != archive_bytes
    ):
        raise TaskspaceSelectedPreimageProgramError("receiver custody does not reopen the semantic archive")
    for key in (
        "all_archive_bytes_have_one_home",
        "all_five_roles_consumed",
        "deterministic_probe_replay",
        "scorer_solved_template_parse_reencode_identical",
        "worldsheet_g1_semantic_parseback",
    ):
        required_true(receiver, key, label=f"receiver_custody.{key}")
    if receiver.get("worldsheet_g1_pair_count") != PAIR_COUNT_N600:
        raise TaskspaceSelectedPreimageProgramError("receiver custody is not full n600")

    resume = required_mapping(receipt, "resume", label="resume")
    for key in (
        "all_preserved",
        "per_scorer_batch_checkpoints",
        "receiver_checkpoint_preserved",
        "solver_checkpoint_preserved",
    ):
        required_true(resume, key, label=f"resume.{key}")
    if resume.get("batch_size") != 16:
        raise TaskspaceSelectedPreimageProgramError("resume custody is not batch16")
    mutation = required_mapping(receipt, "fail_closed_mutation_proof", label="fail_closed_mutation_proof")
    if (
        mutation.get("all_samples_refused_or_changed_decode") is not True
        or mutation.get("sampled_member_payload_homes") != 5
        or mutation.get("unique_home_coverage_bytes") != archive_bytes
    ):
        raise TaskspaceSelectedPreimageProgramError("semantic archive mutation/consumption proof is incomplete")

    dependency_fields = (
        "solve_archive_sha256",
        "v14_archive_sha256",
        "v14_receipt_sha256",
        "template_source_sha256",
        "target_cache_sha256",
    )
    dependency_sha256s = tuple(
        sorted(
            {
                _require_sha256(config.get(field_name), label=f"typed_config.{field_name}")
                for field_name in dependency_fields
            }
        )
    )
    if archive_sha256 in dependency_sha256s:
        raise TaskspaceSelectedPreimageProgramError(
            "semantic output archive was supplied as a declared compile dependency"
        )
    target_custody = required_mapping(receipt, "target_custody", label="target_custody")
    if (
        target_custody.get("sha256") != config.get("target_cache_sha256")
        or target_custody.get("bytes") != config.get("target_cache_bytes")
        or target_custody.get("mutated") is not False
    ):
        raise TaskspaceSelectedPreimageProgramError("encoder target custody does not match typed config")

    compile_receipt_sha256 = _sha256(compile_receipt_bytes)
    proof_dependency_sha256 = _sha256(
        _canonical_json(
            {
                "batch_digest_chain_sha256": batch_digest_chain_sha256,
                "compiled_semantic_archive_bytes": archive_bytes,
                "compiled_semantic_archive_sha256": archive_sha256,
                "declared_compile_dependency_sha256s": list(dependency_sha256s),
                "fresh_compile_receipt_sha256": compile_receipt_sha256,
                "producer_source_sha256s": verified_producers,
                "run_id": run_id,
                "schema": "tac.v15.semantic_compile_proof_dependency.v1",
                "typed_compile_config_sha256": typed_config_sha256,
            }
        )
    )
    return V15SemanticProgramIdentityV1(
        fresh_compile_receipt_schema=V15_SEMANTIC_COMPILE_RECEIPT_SCHEMA,
        fresh_compile_receipt_sha256=compile_receipt_sha256,
        compile_proof_dependency_sha256=proof_dependency_sha256,
        typed_compile_config_sha256=typed_config_sha256,
        compiler_source_sha256=verified_producers["tools/measure_ddm_v15_scorer_solved_templates.py"],
        receiver_source_sha256=verified_producers["src/tac/optimization/direct_description_carrier_compose.py"],
        compiled_semantic_archive_sha256=archive_sha256,
        compiled_semantic_archive_bytes=archive_bytes,
        source_pair_start=source_pair_start,
        pair_count=pair_count,
        declared_compile_dependency_sha256s=dependency_sha256s,
    )


@dataclass(frozen=True, slots=True)
class GenericV10Factor2DecoderIdentityV1:
    """Generic decoder identity; code is free, video-derived operands are not."""

    implementation_source_sha256: str
    decoder_id: str = GENERIC_V10_FACTOR2_DECODER_ID
    scorer_height: int = SCORER_HEIGHT
    scorer_width: int = SCORER_WIDTH
    camera_height: int = CAMERA_HEIGHT
    camera_width: int = CAMERA_WIDTH
    channels: int = CHANNELS
    resize_geometry: str = "DISJOINT_HALF_PIXEL_FACTOR2_INTEGER_NUMERATOR"
    scorer_dependency: bool = False
    video_derived_constants_in_decoder: bool = False

    def __post_init__(self) -> None:
        _require_sha256(
            self.implementation_source_sha256,
            label="implementation_source_sha256",
        )
        if (
            self.decoder_id != GENERIC_V10_FACTOR2_DECODER_ID
            or (
                self.scorer_height,
                self.scorer_width,
                self.camera_height,
                self.camera_width,
                self.channels,
            )
            != (
                SCORER_HEIGHT,
                SCORER_WIDTH,
                CAMERA_HEIGHT,
                CAMERA_WIDTH,
                CHANNELS,
            )
            or self.resize_geometry != "DISJOINT_HALF_PIXEL_FACTOR2_INTEGER_NUMERATOR"
            or self.scorer_dependency is not False
            or self.video_derived_constants_in_decoder is not False
        ):
            raise TaskspaceSelectedPreimageProgramError("generic V10 decoder identity changed semantics or custody")

    @classmethod
    def current(cls) -> GenericV10Factor2DecoderIdentityV1:
        return cls(implementation_source_sha256=_source_sha256(realize_factor2_uint8_scorer_plane))


@dataclass(frozen=True, slots=True)
class SelectedPreimageCompileConfigV1:
    """Encoder control and exact population identity for one program.

    The budget receipt, budget rule, and parser ceiling are compile/proof
    controls.  They do not affect receiver function and are intentionally
    excluded from packet bytes and dataclass equality so a frontier-pointer
    refresh cannot mutate an otherwise identical counted representation.
    """

    source_pair_start: int
    pair_count: int
    maximum_packet_bytes: int = field(compare=False, repr=False)
    score_budget_receipt_sha256: str = field(compare=False, repr=False)
    budget_rule_id: str = field(compare=False, repr=False)
    research_only: bool = True
    score_claim: bool = False

    def __post_init__(self) -> None:
        _require_int(
            self.source_pair_start,
            label="source_pair_start",
            minimum=0,
            maximum=PAIR_COUNT_N600 - 1,
        )
        _require_int(
            self.pair_count,
            label="pair_count",
            minimum=1,
            maximum=PAIR_COUNT_N600,
        )
        if self.source_pair_start + self.pair_count > PAIR_COUNT_N600:
            raise TaskspaceSelectedPreimageProgramError("compile pair window escapes n600")
        _require_int(
            self.maximum_packet_bytes,
            label="maximum_packet_bytes",
            minimum=1,
        )
        _require_sha256(
            self.score_budget_receipt_sha256,
            label="score_budget_receipt_sha256",
        )
        _require_id(self.budget_rule_id, label="budget_rule_id")
        if self.research_only is not True or self.score_claim is not False:
            raise TaskspaceSelectedPreimageProgramError("V1 is research-only and cannot carry a score claim")


@dataclass(frozen=True, slots=True)
class TaskspaceSelectedPreimageFactorV1:
    """One strictly typed, counted, video-derived factor section."""

    section_id: str
    role: SelectedPreimageFactorRoleV1
    mode: SelectedPreimageFactorModeV1
    source_pair_start: int
    source_pair_stop_exclusive: int
    payload: bytes
    source_receipt_sha256: str
    byte_home: SelectedPreimageByteHomeV1
    lineage_class: SelectedPreimageLineageClassV1

    def __post_init__(self) -> None:
        _require_id(self.section_id, label="section_id")
        if type(self.role) is not SelectedPreimageFactorRoleV1:
            raise TaskspaceSelectedPreimageProgramError("factor role must be the exact closed enum")
        if type(self.mode) is not SelectedPreimageFactorModeV1:
            raise TaskspaceSelectedPreimageProgramError("factor mode must be the exact closed enum")
        _require_int(
            self.source_pair_start,
            label="factor.source_pair_start",
            minimum=0,
            maximum=PAIR_COUNT_N600 - 1,
        )
        _require_int(
            self.source_pair_stop_exclusive,
            label="factor.source_pair_stop_exclusive",
            minimum=1,
            maximum=PAIR_COUNT_N600,
        )
        if self.source_pair_start >= self.source_pair_stop_exclusive:
            raise TaskspaceSelectedPreimageProgramError("factor pair interval is empty or reversed")
        if type(self.payload) is not bytes or not self.payload:
            raise TaskspaceSelectedPreimageProgramError("factor payload must be nonempty exact bytes")
        _require_sha256(
            self.source_receipt_sha256,
            label="factor.source_receipt_sha256",
        )
        expected = {
            SelectedPreimageFactorRoleV1.ANALYTIC_RESIDUAL: (
                SelectedPreimageFactorModeV1.SHEARLET_BOUNDARY_TRANSPORT_Q4,
                SelectedPreimageByteHomeV1.COUNTED_ANALYTIC_OPERAND,
                SelectedPreimageLineageClassV1.VIDEO_DERIVED_ANALYTIC_FACTOR,
            ),
            SelectedPreimageFactorRoleV1.LEARNED_IRREDUCIBLE_QUOTIENT: (
                SelectedPreimageFactorModeV1.COMPACT_LATENT_QUOTIENT_PLUGIN,
                SelectedPreimageByteHomeV1.COUNTED_LEARNED_OPERAND,
                SelectedPreimageLineageClassV1.VIDEO_DERIVED_LEARNED_IRREDUCIBLE_FACTOR,
            ),
        }[self.role]
        if (self.mode, self.byte_home, self.lineage_class) != expected:
            raise TaskspaceSelectedPreimageProgramError("factor role/mode/byte-home/lineage combination is invalid")
        if self.role is SelectedPreimageFactorRoleV1.ANALYTIC_RESIDUAL:
            _parse_analytic_payload(self.payload, factor=self)
        else:
            _parse_learned_payload(self.payload, factor=self)

    @property
    def payload_sha256(self) -> str:
        return _sha256(self.payload)

    def addresses(self, source_pair_id: int) -> bool:
        if not self.source_pair_start <= source_pair_id < self.source_pair_stop_exclusive:
            return False
        if self.role is SelectedPreimageFactorRoleV1.ANALYTIC_RESIDUAL:
            analytic = _parse_analytic_payload(self.payload, factor=self)
            return any(atom.pair_index == source_pair_id for atom in analytic.atoms)
        learned = _parse_learned_payload(self.payload, factor=self)
        return any(start <= source_pair_id < stop for start, stop in learned.active_pair_ranges)


@dataclass(frozen=True, slots=True)
class SelectedPreimageByteHomeRecordV1:
    section_id: str
    offset: int
    byte_length: int
    payload_sha256: str
    byte_home: SelectedPreimageByteHomeV1
    lineage_class: SelectedPreimageLineageClassV1


@dataclass(frozen=True, slots=True)
class _AnalyticShearletPayloadV1:
    frame_selector: SelectedPreimageFrameSelectorV1
    source_rgb_u8: tuple[int, int, int]
    added_rgb_u8: tuple[int, int, int]
    removed_rgb_u8: tuple[int, int, int]
    atoms: tuple[BoundaryShearletAtomV1, ...]


@dataclass(frozen=True, slots=True)
class _LearnedQuotientPayloadV1:
    active_pair_ranges: tuple[tuple[int, int], ...]
    decoder_contract_id: str
    decoder_implementation_source_sha256: str
    model_family_id: str
    latent_codec_id: str
    parameter_codec_id: str
    latent_dtype: str
    parameter_dtype: str
    latent_count: int
    parameter_count: int
    latent_payload: bytes
    parameter_payload: bytes


def _atom_row(atom: BoundaryShearletAtomV1) -> dict[str, Any]:
    # BoundaryShearletAtomV1.role is a routing field in the donor carrier:
    # CarrierComposeReceiverV1 filters atoms by role before calling the
    # geometry primitive.  This selected-preimage factor already routes by an
    # exact source RGB operand, and the geometry primitive does not read role.
    # Do not charge a repeated, receiver-inert role token on this wire.
    return {
        "amplitude_q4": atom.amplitude_q4,
        "center_x": atom.center_x,
        "center_y": atom.center_y,
        "pair_index": atom.pair_index,
        "scale_x": atom.scale_x,
        "scale_y": atom.scale_y,
        "shear_q4": atom.shear_q4,
    }


def _parse_analytic_payload(
    payload: bytes,
    *,
    factor: TaskspaceSelectedPreimageFactorV1 | None = None,
) -> _AnalyticShearletPayloadV1:
    value = _decode_canonical_object(payload, label="analytic residual payload")
    _exact_keys(
        value,
        {
            "added_rgb_u8",
            "atoms",
            "frame_selector",
            "removed_rgb_u8",
            "schema",
            "source_rgb_u8",
        },
        label="analytic residual payload",
    )
    if value["schema"] != ANALYTIC_SCHEMA:
        raise TaskspaceSelectedPreimageProgramError("analytic residual payload schema mismatch")
    try:
        frame_selector = SelectedPreimageFrameSelectorV1(value["frame_selector"])
    except ValueError as exc:
        raise TaskspaceSelectedPreimageProgramError("analytic frame selector is unknown") from exc
    rows = value["atoms"]
    if type(rows) is not list or not rows:
        raise TaskspaceSelectedPreimageProgramError("analytic factor requires a nonempty atom list")
    atoms: list[BoundaryShearletAtomV1] = []
    for row in rows:
        _exact_keys(
            row,
            {
                "amplitude_q4",
                "center_x",
                "center_y",
                "pair_index",
                "scale_x",
                "scale_y",
                "shear_q4",
            },
            label="analytic atom",
        )
        try:
            atom = BoundaryShearletAtomV1(role="Road", **row)
        except (TypeError, ValueError) as exc:
            raise TaskspaceSelectedPreimageProgramError("analytic atom is invalid") from exc
        atoms.append(atom)
    ordered = tuple(sorted(atoms))
    if tuple(atoms) != ordered or len(set(ordered)) != len(ordered):
        raise TaskspaceSelectedPreimageProgramError("analytic atoms must be unique canonical order")
    if factor is not None and any(
        not factor.source_pair_start <= atom.pair_index < factor.source_pair_stop_exclusive for atom in ordered
    ):
        raise TaskspaceSelectedPreimageProgramError("analytic atom escapes its factor pair interval")
    return _AnalyticShearletPayloadV1(
        frame_selector=frame_selector,
        source_rgb_u8=_require_u8_rgb(
            value["source_rgb_u8"],
            label="source_rgb_u8",
        ),
        added_rgb_u8=_require_u8_rgb(
            value["added_rgb_u8"],
            label="added_rgb_u8",
        ),
        removed_rgb_u8=_require_u8_rgb(
            value["removed_rgb_u8"],
            label="removed_rgb_u8",
        ),
        atoms=ordered,
    )


def _dtype_width(dtype: str, *, label: str) -> int:
    widths = {"int8": 1, "uint8": 1, "int16le": 2}
    if dtype not in widths:
        raise TaskspaceSelectedPreimageProgramError(f"{label} must be one compact integer dtype")
    return widths[dtype]


def _parse_learned_payload(
    payload: bytes,
    *,
    factor: TaskspaceSelectedPreimageFactorV1 | None = None,
) -> _LearnedQuotientPayloadV1:
    if len(payload) < _LEARNED_HEADER.size:
        raise TaskspaceSelectedPreimageProgramError("learned quotient payload is truncated")
    magic, header_bytes = _LEARNED_HEADER.unpack_from(payload)
    if magic != LEARNED_MAGIC or header_bytes < 1:
        raise TaskspaceSelectedPreimageProgramError("learned quotient magic or header length is invalid")
    header_stop = _LEARNED_HEADER.size + header_bytes
    if header_stop > len(payload):
        raise TaskspaceSelectedPreimageProgramError("learned quotient header escapes payload")
    header = _decode_canonical_object(
        payload[_LEARNED_HEADER.size : header_stop],
        label="learned quotient header",
    )
    _exact_keys(
        header,
        {
            "active_pair_ranges",
            "decoder_contract_id",
            "decoder_implementation_source_sha256",
            "latent_byte_length",
            "latent_codec_id",
            "latent_count",
            "latent_dtype",
            "latent_sha256",
            "model_family_id",
            "parameter_byte_length",
            "parameter_codec_id",
            "parameter_count",
            "parameter_dtype",
            "parameter_sha256",
            "schema",
        },
        label="learned quotient header",
    )
    if header["schema"] != LEARNED_SCHEMA:
        raise TaskspaceSelectedPreimageProgramError("learned quotient schema mismatch")
    factor_start = 0 if factor is None else factor.source_pair_start
    factor_stop = PAIR_COUNT_N600 if factor is None else factor.source_pair_stop_exclusive
    active_pair_ranges = _require_active_pair_ranges(
        header["active_pair_ranges"],
        factor_start=factor_start,
        factor_stop=factor_stop,
    )
    latent_bytes = _require_int(
        header["latent_byte_length"],
        label="latent_byte_length",
    )
    parameter_bytes = _require_int(
        header["parameter_byte_length"],
        label="parameter_byte_length",
    )
    latent_count = _require_int(
        header["latent_count"],
        label="latent_count",
        minimum=1,
    )
    parameter_count = _require_int(
        header["parameter_count"],
        label="parameter_count",
        minimum=1,
    )
    latent_width = _dtype_width(header["latent_dtype"], label="latent_dtype")
    parameter_width = _dtype_width(
        header["parameter_dtype"],
        label="parameter_dtype",
    )
    if (
        latent_count * latent_width != latent_bytes
        or parameter_count * parameter_width != parameter_bytes
        or header_stop + latent_bytes + parameter_bytes != len(payload)
    ):
        raise TaskspaceSelectedPreimageProgramError("learned quotient scalar/byte accounting is inconsistent")
    latent_payload = payload[header_stop : header_stop + latent_bytes]
    parameter_payload = payload[header_stop + latent_bytes :]
    if _sha256(latent_payload) != _require_sha256(header["latent_sha256"], label="latent_sha256") or _sha256(
        parameter_payload
    ) != _require_sha256(
        header["parameter_sha256"],
        label="parameter_sha256",
    ):
        raise TaskspaceSelectedPreimageProgramError("learned quotient payload hash mismatch")
    active_pair_count = sum(stop - start for start, stop in active_pair_ranges)
    explicit_scalar_count = active_pair_count * 2 * SCORER_HEIGHT * SCORER_WIDTH * CHANNELS
    if latent_count >= explicit_scalar_count or len(payload) >= explicit_scalar_count:
        raise TaskspaceSelectedPreimageProgramError(
            "learned quotient is not structurally smaller than explicit coupled preimage planes"
        )
    return _LearnedQuotientPayloadV1(
        active_pair_ranges=active_pair_ranges,
        decoder_contract_id=_require_id(
            header["decoder_contract_id"],
            label="decoder_contract_id",
        ),
        decoder_implementation_source_sha256=_require_sha256(
            header["decoder_implementation_source_sha256"],
            label="decoder_implementation_source_sha256",
        ),
        model_family_id=_require_id(
            header["model_family_id"],
            label="model_family_id",
        ),
        latent_codec_id=_require_id(
            header["latent_codec_id"],
            label="latent_codec_id",
        ),
        parameter_codec_id=_require_id(
            header["parameter_codec_id"],
            label="parameter_codec_id",
        ),
        latent_dtype=header["latent_dtype"],
        parameter_dtype=header["parameter_dtype"],
        latent_count=latent_count,
        parameter_count=parameter_count,
        latent_payload=latent_payload,
        parameter_payload=parameter_payload,
    )


def build_analytic_shearlet_residual_factor(
    *,
    section_id: str,
    source_pair_start: int,
    source_pair_stop_exclusive: int,
    frame_selector: SelectedPreimageFrameSelectorV1,
    source_rgb_u8: tuple[int, int, int],
    added_rgb_u8: tuple[int, int, int],
    removed_rgb_u8: tuple[int, int, int],
    atoms: Sequence[BoundaryShearletAtomV1],
    source_receipt_sha256: str,
) -> TaskspaceSelectedPreimageFactorV1:
    """Build a compact parametric shearlet residual, never a pixel patch."""

    if type(frame_selector) is not SelectedPreimageFrameSelectorV1:
        raise TaskspaceSelectedPreimageProgramError("frame_selector must be the exact closed enum")
    canonical_atoms = tuple(atoms)
    payload = _canonical_json(
        {
            "added_rgb_u8": list(_require_u8_rgb(added_rgb_u8, label="added_rgb_u8")),
            "atoms": [_atom_row(atom) for atom in canonical_atoms],
            "frame_selector": frame_selector.value,
            "removed_rgb_u8": list(_require_u8_rgb(removed_rgb_u8, label="removed_rgb_u8")),
            "schema": ANALYTIC_SCHEMA,
            "source_rgb_u8": list(_require_u8_rgb(source_rgb_u8, label="source_rgb_u8")),
        }
    )
    return TaskspaceSelectedPreimageFactorV1(
        section_id=section_id,
        role=SelectedPreimageFactorRoleV1.ANALYTIC_RESIDUAL,
        mode=SelectedPreimageFactorModeV1.SHEARLET_BOUNDARY_TRANSPORT_Q4,
        source_pair_start=source_pair_start,
        source_pair_stop_exclusive=source_pair_stop_exclusive,
        payload=payload,
        source_receipt_sha256=source_receipt_sha256,
        byte_home=SelectedPreimageByteHomeV1.COUNTED_ANALYTIC_OPERAND,
        lineage_class=SelectedPreimageLineageClassV1.VIDEO_DERIVED_ANALYTIC_FACTOR,
    )


def build_learned_irreducible_quotient_factor(
    *,
    section_id: str,
    source_pair_start: int,
    source_pair_stop_exclusive: int,
    decoder_contract_id: str,
    decoder_implementation_source_sha256: str,
    model_family_id: str,
    latent_codec_id: str,
    parameter_codec_id: str,
    latent_dtype: str,
    parameter_dtype: str,
    latent_payload: bytes,
    parameter_payload: bytes,
    source_receipt_sha256: str,
    active_pair_ranges: Sequence[tuple[int, int]] | None = None,
) -> TaskspaceSelectedPreimageFactorV1:
    """Build typed compact latents/parameters; raw dense arrays are impossible."""

    if type(latent_payload) is not bytes or not latent_payload:
        raise TaskspaceSelectedPreimageProgramError("learned latent payload must be nonempty exact bytes")
    if type(parameter_payload) is not bytes or not parameter_payload:
        raise TaskspaceSelectedPreimageProgramError("learned parameter payload must be nonempty exact bytes")
    latent_width = _dtype_width(latent_dtype, label="latent_dtype")
    parameter_width = _dtype_width(parameter_dtype, label="parameter_dtype")
    if len(latent_payload) % latent_width or len(parameter_payload) % parameter_width:
        raise TaskspaceSelectedPreimageProgramError("learned payload byte lengths do not align to declared dtypes")
    active_ranges = _require_active_pair_ranges(
        (
            ((source_pair_start, source_pair_stop_exclusive),)
            if active_pair_ranges is None
            else tuple(active_pair_ranges)
        ),
        factor_start=source_pair_start,
        factor_stop=source_pair_stop_exclusive,
    )
    header = _canonical_json(
        {
            "active_pair_ranges": [list(row) for row in active_ranges],
            "decoder_contract_id": _require_id(
                decoder_contract_id,
                label="decoder_contract_id",
            ),
            "decoder_implementation_source_sha256": _require_sha256(
                decoder_implementation_source_sha256,
                label="decoder_implementation_source_sha256",
            ),
            "latent_byte_length": len(latent_payload),
            "latent_codec_id": _require_id(
                latent_codec_id,
                label="latent_codec_id",
            ),
            "latent_count": len(latent_payload) // latent_width,
            "latent_dtype": latent_dtype,
            "latent_sha256": _sha256(latent_payload),
            "model_family_id": _require_id(
                model_family_id,
                label="model_family_id",
            ),
            "parameter_byte_length": len(parameter_payload),
            "parameter_codec_id": _require_id(
                parameter_codec_id,
                label="parameter_codec_id",
            ),
            "parameter_count": len(parameter_payload) // parameter_width,
            "parameter_dtype": parameter_dtype,
            "parameter_sha256": _sha256(parameter_payload),
            "schema": LEARNED_SCHEMA,
        }
    )
    if len(header) > _U32_MAX:
        raise TaskspaceSelectedPreimageProgramError("learned quotient header exceeds its uint32 wire length")
    payload = _LEARNED_HEADER.pack(LEARNED_MAGIC, len(header)) + header + latent_payload + parameter_payload
    return TaskspaceSelectedPreimageFactorV1(
        section_id=section_id,
        role=SelectedPreimageFactorRoleV1.LEARNED_IRREDUCIBLE_QUOTIENT,
        mode=SelectedPreimageFactorModeV1.COMPACT_LATENT_QUOTIENT_PLUGIN,
        source_pair_start=source_pair_start,
        source_pair_stop_exclusive=source_pair_stop_exclusive,
        payload=payload,
        source_receipt_sha256=source_receipt_sha256,
        byte_home=SelectedPreimageByteHomeV1.COUNTED_LEARNED_OPERAND,
        lineage_class=SelectedPreimageLineageClassV1.VIDEO_DERIVED_LEARNED_IRREDUCIBLE_FACTOR,
    )


@dataclass(frozen=True, slots=True)
class TaskspaceSelectedPreimageProgramV1:
    """Logical counted program with strict deterministic binary encoding."""

    semantic_program_identity: V15SemanticProgramIdentityV1
    target_custody_identity: ScorerTargetCustodyIdentityV1
    decoder_identity: GenericV10Factor2DecoderIdentityV1
    compile_config: SelectedPreimageCompileConfigV1
    factors: tuple[TaskspaceSelectedPreimageFactorV1, ...]
    forbidden_payload_attestations: tuple[str, ...] = _FORBIDDEN_ATTESTATIONS
    semantic_program_embedding_state: str = "FRESH_COMPILE_RECEIPT_IDENTITY_ONLY_RUNNER_EMBEDDING_OWED"
    standalone_decode_closed: bool = False
    research_only: bool = True
    score_claim: bool = False

    def __post_init__(self) -> None:
        if type(self.semantic_program_identity) is not V15SemanticProgramIdentityV1:
            raise TaskspaceSelectedPreimageProgramError("semantic identity must be exact V15SemanticProgramIdentityV1")
        if type(self.target_custody_identity) is not ScorerTargetCustodyIdentityV1:
            raise TaskspaceSelectedPreimageProgramError("target custody must be exact ScorerTargetCustodyIdentityV1")
        if type(self.decoder_identity) is not GenericV10Factor2DecoderIdentityV1:
            raise TaskspaceSelectedPreimageProgramError(
                "decoder identity must be exact GenericV10Factor2DecoderIdentityV1"
            )
        if type(self.compile_config) is not SelectedPreimageCompileConfigV1:
            raise TaskspaceSelectedPreimageProgramError("compile config must be exact SelectedPreimageCompileConfigV1")
        if (
            self.semantic_program_identity.source_pair_start != self.compile_config.source_pair_start
            or self.semantic_program_identity.pair_count != self.compile_config.pair_count
        ):
            raise TaskspaceSelectedPreimageProgramError("semantic and compile pair populations differ")
        if not self.factors or any(type(factor) is not TaskspaceSelectedPreimageFactorV1 for factor in self.factors):
            raise TaskspaceSelectedPreimageProgramError("program requires nonempty exact factor records")
        order = tuple(
            sorted(
                self.factors,
                key=lambda factor: (
                    _FACTOR_STAGE_ORDER[factor.role],
                    factor.source_pair_start,
                    factor.source_pair_stop_exclusive,
                    factor.section_id,
                ),
            )
        )
        if self.factors != order or len({row.section_id for row in order}) != len(order):
            raise TaskspaceSelectedPreimageProgramError("factor sections must be unique canonical order")
        start = self.compile_config.source_pair_start
        stop = start + self.compile_config.pair_count
        if any(factor.source_pair_start < start or factor.source_pair_stop_exclusive > stop for factor in self.factors):
            raise TaskspaceSelectedPreimageProgramError("factor interval escapes the program pair population")
        if (
            self.forbidden_payload_attestations != _FORBIDDEN_ATTESTATIONS
            or self.semantic_program_embedding_state != "FRESH_COMPILE_RECEIPT_IDENTITY_ONLY_RUNNER_EMBEDDING_OWED"
            or self.standalone_decode_closed is not False
            or self.research_only is not True
            or self.score_claim is not False
        ):
            raise TaskspaceSelectedPreimageProgramError(
                "V1 truth labels cannot claim standalone decode, candidate, or score"
            )

    @property
    def packet_bytes(self) -> bytes:
        return encode_selected_preimage_program(self)

    @property
    def packet_sha256(self) -> str:
        return _sha256(self.packet_bytes)

    @property
    def open_product_blockers(self) -> tuple[str, ...]:
        """Missing outer-product joins that prevent candidate closure."""

        return OPEN_SELECTED_SOLUTION_PRODUCT_BLOCKERS

    @property
    def required_counted_input_source_bytes(self) -> int:
        """Packet plus external semantic bytes before outer-container coding.

        This is not an archive-size or rate claim: joint/outer coding and
        container overhead are deliberately absent until an outer archive is
        built and strictly reopened.
        """

        return len(self.packet_bytes) + self.semantic_program_identity.compiled_semantic_archive_bytes

    def byte_homes(self) -> tuple[SelectedPreimageByteHomeRecordV1, ...]:
        """Partition only this inner packet, never the missing outer archive."""

        packet, header_bytes = _encode_program_components(self)
        prefix_bytes = _PROGRAM_HEADER.size + len(header_bytes)
        records = [
            SelectedPreimageByteHomeRecordV1(
                section_id="PROGRAM_FRAMING_AND_MANIFEST",
                offset=0,
                byte_length=prefix_bytes,
                payload_sha256=_sha256(packet[:prefix_bytes]),
                byte_home=SelectedPreimageByteHomeV1.COUNTED_PACKET_FRAMING,
                lineage_class=SelectedPreimageLineageClassV1.FRESH_ORIGINAL_SEMANTIC_COMPILE_IDENTITY,
            )
        ]
        offset = prefix_bytes
        for factor in self.factors:
            records.append(
                SelectedPreimageByteHomeRecordV1(
                    section_id=factor.section_id,
                    offset=offset,
                    byte_length=len(factor.payload),
                    payload_sha256=factor.payload_sha256,
                    byte_home=factor.byte_home,
                    lineage_class=factor.lineage_class,
                )
            )
            offset += len(factor.payload)
        if offset != len(packet):
            raise TaskspaceSelectedPreimageProgramError("byte-home records do not partition packet exactly")
        return tuple(records)


def _semantic_identity_row(value: V15SemanticProgramIdentityV1) -> dict[str, Any]:
    return {
        "compile_derivation": value.compile_derivation,
        "compile_input_classes": list(value.compile_input_classes),
        "compile_proof_dependency_sha256": value.compile_proof_dependency_sha256,
        "compiled_semantic_archive_bytes": value.compiled_semantic_archive_bytes,
        "compiled_semantic_archive_sha256": value.compiled_semantic_archive_sha256,
        "compiler_source_sha256": value.compiler_source_sha256,
        "declared_compile_dependency_sha256s": list(value.declared_compile_dependency_sha256s),
        "fresh_compile_receipt_schema": value.fresh_compile_receipt_schema,
        "fresh_compile_receipt_sha256": value.fresh_compile_receipt_sha256,
        "output_archive_was_declared_compile_input": value.output_archive_was_declared_compile_input,
        "pair_count": value.pair_count,
        "receiver_contract_id": value.receiver_contract_id,
        "receiver_source_sha256": value.receiver_source_sha256,
        "semantic_archive_embedded": value.semantic_archive_embedded,
        "source_pair_start": value.source_pair_start,
        "typed_compile_config_sha256": value.typed_compile_config_sha256,
    }


def _target_identity_row(value: ScorerTargetCustodyIdentityV1) -> dict[str, Any]:
    return {
        "geometry_authority": value.geometry_authority,
        "historical_batch32_targets_consumed": value.historical_batch32_targets_consumed,
        "pair_count": value.pair_count,
        "pair_sequence_length": value.pair_sequence_length,
        "pairing_policy": value.pairing_policy,
        "scorer_batch_size": value.scorer_batch_size,
        "segnet_frame_index": value.segnet_frame_index,
        "target_bank_sha256": value.target_bank_sha256,
        "target_custody_receipt_sha256": value.target_custody_receipt_sha256,
        "target_payload_embedded": value.target_payload_embedded,
    }


def _decoder_identity_row(
    value: GenericV10Factor2DecoderIdentityV1,
) -> dict[str, Any]:
    return {
        "camera_height": value.camera_height,
        "camera_width": value.camera_width,
        "channels": value.channels,
        "decoder_id": value.decoder_id,
        "implementation_source_sha256": value.implementation_source_sha256,
        "resize_geometry": value.resize_geometry,
        "scorer_dependency": value.scorer_dependency,
        "scorer_height": value.scorer_height,
        "scorer_width": value.scorer_width,
        "video_derived_constants_in_decoder": value.video_derived_constants_in_decoder,
    }


def _compile_config_row(value: SelectedPreimageCompileConfigV1) -> dict[str, Any]:
    return {
        "pair_count": value.pair_count,
        "research_only": value.research_only,
        "score_claim": value.score_claim,
        "source_pair_start": value.source_pair_start,
    }


def _factor_row(
    factor: TaskspaceSelectedPreimageFactorV1,
    *,
    payload_offset: int,
) -> dict[str, Any]:
    return {
        "byte_home": factor.byte_home.value,
        "byte_length": len(factor.payload),
        "lineage_class": factor.lineage_class.value,
        "mode": factor.mode.value,
        "payload_offset": payload_offset,
        "payload_sha256": factor.payload_sha256,
        "role": factor.role.value,
        "section_id": factor.section_id,
        "source_pair_start": factor.source_pair_start,
        "source_pair_stop_exclusive": factor.source_pair_stop_exclusive,
        "source_receipt_sha256": factor.source_receipt_sha256,
    }


def _encode_program_components(
    program: TaskspaceSelectedPreimageProgramV1,
) -> tuple[bytes, bytes]:
    offset = 0
    factor_rows: list[dict[str, Any]] = []
    payloads: list[bytes] = []
    for factor in program.factors:
        factor_rows.append(_factor_row(factor, payload_offset=offset))
        payloads.append(factor.payload)
        offset += len(factor.payload)
    header = _canonical_json(
        {
            "compile_config": _compile_config_row(program.compile_config),
            "decoder_identity": _decoder_identity_row(program.decoder_identity),
            "factor_sections": factor_rows,
            "forbidden_payload_attestations": list(program.forbidden_payload_attestations),
            "research_only": program.research_only,
            "schema": PROGRAM_SCHEMA,
            "score_claim": program.score_claim,
            "semantic_program_embedding_state": program.semantic_program_embedding_state,
            "semantic_program_identity": _semantic_identity_row(program.semantic_program_identity),
            "standalone_decode_closed": program.standalone_decode_closed,
            "target_custody_identity": _target_identity_row(program.target_custody_identity),
        }
    )
    if len(header) > _U32_MAX:
        raise TaskspaceSelectedPreimageProgramError("program header exceeds its uint32 wire length")
    packet = _PROGRAM_HEADER.pack(MAGIC, len(header)) + header + b"".join(payloads)
    return packet, header


def encode_selected_preimage_program(
    program: TaskspaceSelectedPreimageProgramV1,
) -> bytes:
    if type(program) is not TaskspaceSelectedPreimageProgramV1:
        raise TaskspaceSelectedPreimageProgramError("encode requires exact TaskspaceSelectedPreimageProgramV1")
    first, _ = _encode_program_components(program)
    second, _ = _encode_program_components(program)
    if first != second:
        raise TaskspaceSelectedPreimageProgramError("selected-preimage program encoding is nondeterministic")
    if len(first) > program.compile_config.maximum_packet_bytes:
        raise TaskspaceSelectedPreimageProgramError("counted program exceeds its score-derived packet ceiling")
    return first


def _parse_semantic_identity(value: object) -> V15SemanticProgramIdentityV1:
    _exact_keys(
        value,
        {
            "compile_derivation",
            "compile_input_classes",
            "compile_proof_dependency_sha256",
            "compiled_semantic_archive_bytes",
            "compiled_semantic_archive_sha256",
            "compiler_source_sha256",
            "declared_compile_dependency_sha256s",
            "fresh_compile_receipt_schema",
            "fresh_compile_receipt_sha256",
            "output_archive_was_declared_compile_input",
            "pair_count",
            "receiver_contract_id",
            "receiver_source_sha256",
            "semantic_archive_embedded",
            "source_pair_start",
            "typed_compile_config_sha256",
        },
        label="semantic program identity",
    )
    return V15SemanticProgramIdentityV1(
        **{
            **value,
            "compile_input_classes": tuple(value["compile_input_classes"]),
            "declared_compile_dependency_sha256s": tuple(value["declared_compile_dependency_sha256s"]),
        }
    )


def _parse_target_identity(value: object) -> ScorerTargetCustodyIdentityV1:
    _exact_keys(
        value,
        {
            "geometry_authority",
            "historical_batch32_targets_consumed",
            "pair_count",
            "pair_sequence_length",
            "pairing_policy",
            "scorer_batch_size",
            "segnet_frame_index",
            "target_bank_sha256",
            "target_custody_receipt_sha256",
            "target_payload_embedded",
        },
        label="target custody identity",
    )
    return ScorerTargetCustodyIdentityV1(**value)


def _parse_decoder_identity(
    value: object,
) -> GenericV10Factor2DecoderIdentityV1:
    _exact_keys(
        value,
        {
            "camera_height",
            "camera_width",
            "channels",
            "decoder_id",
            "implementation_source_sha256",
            "resize_geometry",
            "scorer_dependency",
            "scorer_height",
            "scorer_width",
            "video_derived_constants_in_decoder",
        },
        label="decoder identity",
    )
    return GenericV10Factor2DecoderIdentityV1(**value)


def _parse_compile_config(
    value: object,
    *,
    maximum_packet_bytes: int,
) -> SelectedPreimageCompileConfigV1:
    _exact_keys(
        value,
        {
            "pair_count",
            "research_only",
            "score_claim",
            "source_pair_start",
        },
        label="compile config",
    )
    return SelectedPreimageCompileConfigV1(
        **value,
        maximum_packet_bytes=maximum_packet_bytes,
        score_budget_receipt_sha256=_sha256(b"external compile proof control intentionally absent from counted packet"),
        budget_rule_id="external_compile_control_not_counted_v1",
    )


def parse_selected_preimage_program(
    packet: bytes,
    *,
    maximum_packet_bytes: int,
) -> TaskspaceSelectedPreimageProgramV1:
    """Strict parse-back with canonical re-emit and complete byte accounting."""

    if type(packet) is not bytes or not packet:
        raise TaskspaceSelectedPreimageProgramError("packet must be nonempty exact bytes")
    _require_int(
        maximum_packet_bytes,
        label="maximum_packet_bytes",
        minimum=1,
    )
    if len(packet) > maximum_packet_bytes or len(packet) < _PROGRAM_HEADER.size:
        raise TaskspaceSelectedPreimageProgramError("packet exceeds caller bound or is truncated")
    magic, header_bytes = _PROGRAM_HEADER.unpack_from(packet)
    if magic != MAGIC or header_bytes < 1:
        raise TaskspaceSelectedPreimageProgramError("packet magic or header length is invalid")
    header_stop = _PROGRAM_HEADER.size + header_bytes
    if header_stop > len(packet):
        raise TaskspaceSelectedPreimageProgramError("packet header escapes packet bytes")
    header = _decode_canonical_object(
        packet[_PROGRAM_HEADER.size : header_stop],
        label="selected-preimage program header",
    )
    _exact_keys(
        header,
        {
            "compile_config",
            "decoder_identity",
            "factor_sections",
            "forbidden_payload_attestations",
            "research_only",
            "schema",
            "score_claim",
            "semantic_program_embedding_state",
            "semantic_program_identity",
            "standalone_decode_closed",
            "target_custody_identity",
        },
        label="selected-preimage program header",
    )
    if header["schema"] != PROGRAM_SCHEMA:
        raise TaskspaceSelectedPreimageProgramError("selected-preimage program schema mismatch")
    factor_rows = header["factor_sections"]
    if type(factor_rows) is not list or not factor_rows:
        raise TaskspaceSelectedPreimageProgramError("factor descriptor list is invalid")
    payload_region = packet[header_stop:]
    factors: list[TaskspaceSelectedPreimageFactorV1] = []
    expected_offset = 0
    descriptor_fields = {
        "byte_home",
        "byte_length",
        "lineage_class",
        "mode",
        "payload_offset",
        "payload_sha256",
        "role",
        "section_id",
        "source_pair_start",
        "source_pair_stop_exclusive",
        "source_receipt_sha256",
    }
    for row in factor_rows:
        _exact_keys(row, descriptor_fields, label="factor descriptor")
        offset = _require_int(row["payload_offset"], label="payload_offset")
        length = _require_int(row["byte_length"], label="byte_length", minimum=1)
        if offset != expected_offset or offset + length > len(payload_region):
            raise TaskspaceSelectedPreimageProgramError(
                "factor descriptor leaves a gap, overlap, or out-of-bounds span"
            )
        payload = payload_region[offset : offset + length]
        if _sha256(payload) != _require_sha256(
            row["payload_sha256"],
            label="factor.payload_sha256",
        ):
            raise TaskspaceSelectedPreimageProgramError("factor payload SHA-256 mismatch")
        try:
            factor = TaskspaceSelectedPreimageFactorV1(
                section_id=row["section_id"],
                role=SelectedPreimageFactorRoleV1(row["role"]),
                mode=SelectedPreimageFactorModeV1(row["mode"]),
                source_pair_start=row["source_pair_start"],
                source_pair_stop_exclusive=row["source_pair_stop_exclusive"],
                payload=payload,
                source_receipt_sha256=row["source_receipt_sha256"],
                byte_home=SelectedPreimageByteHomeV1(row["byte_home"]),
                lineage_class=SelectedPreimageLineageClassV1(row["lineage_class"]),
            )
        except ValueError as exc:
            raise TaskspaceSelectedPreimageProgramError("factor descriptor enum or payload is invalid") from exc
        factors.append(factor)
        expected_offset += length
    if expected_offset != len(payload_region):
        raise TaskspaceSelectedPreimageProgramError("unowned trailing bytes remain after factor parse")
    program = TaskspaceSelectedPreimageProgramV1(
        semantic_program_identity=_parse_semantic_identity(header["semantic_program_identity"]),
        target_custody_identity=_parse_target_identity(header["target_custody_identity"]),
        decoder_identity=_parse_decoder_identity(header["decoder_identity"]),
        compile_config=_parse_compile_config(
            header["compile_config"],
            maximum_packet_bytes=maximum_packet_bytes,
        ),
        factors=tuple(factors),
        forbidden_payload_attestations=tuple(header["forbidden_payload_attestations"]),
        semantic_program_embedding_state=header["semantic_program_embedding_state"],
        standalone_decode_closed=header["standalone_decode_closed"],
        research_only=header["research_only"],
        score_claim=header["score_claim"],
    )
    if encode_selected_preimage_program(program) != packet:
        raise TaskspaceSelectedPreimageProgramError("packet parse/re-encode identity failed")
    homes = program.byte_homes()
    if sum(row.byte_length for row in homes) != len(packet):
        raise TaskspaceSelectedPreimageProgramError("packet bytes lack a unique complete byte home")
    return program


def compile_v9_v10_selected_preimage_program(
    *,
    semantic_compile_receipt_bytes: bytes,
    compiled_semantic_archive: bytes,
    semantic_producer_root: str | Path,
    target_custody_identity: ScorerTargetCustodyIdentityV1,
    decoder_identity: GenericV10Factor2DecoderIdentityV1,
    factors: Sequence[TaskspaceSelectedPreimageFactorV1],
    config: SelectedPreimageCompileConfigV1,
) -> TaskspaceSelectedPreimageProgramV1:
    """Compile factor operands only after reopening semantic proof lineage."""

    semantic_program_identity = verify_v15_semantic_compile_lineage(
        compile_receipt_bytes=semantic_compile_receipt_bytes,
        compiled_semantic_archive=compiled_semantic_archive,
        producer_root=semantic_producer_root,
    )
    canonical_factors = tuple(
        sorted(
            factors,
            key=lambda factor: (
                _FACTOR_STAGE_ORDER[factor.role],
                factor.source_pair_start,
                factor.source_pair_stop_exclusive,
                factor.section_id,
            ),
        )
    )
    program = TaskspaceSelectedPreimageProgramV1(
        semantic_program_identity=semantic_program_identity,
        target_custody_identity=target_custody_identity,
        decoder_identity=decoder_identity,
        compile_config=config,
        factors=canonical_factors,
    )
    packet = encode_selected_preimage_program(program)
    parsed = parse_selected_preimage_program(
        packet,
        maximum_packet_bytes=config.maximum_packet_bytes,
    )
    if parsed != program:
        raise TaskspaceSelectedPreimageProgramError("compiled program changed under strict parse-back")
    return program


@dataclass(frozen=True, slots=True)
class SelectedPreimageFactor2PairV1:
    """Coupled scorer planes plus exact generic factor-2 realization proofs."""

    scorer_y0: np.ndarray
    scorer_y1: np.ndarray
    camera_y0: np.ndarray
    camera_y1: np.ndarray
    proofs: tuple[Factor2ExactVerification, Factor2ExactVerification]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scorer_y0",
            _require_plane(self.scorer_y0, label="scorer_y0"),
        )
        object.__setattr__(
            self,
            "scorer_y1",
            _require_plane(self.scorer_y1, label="scorer_y1"),
        )
        object.__setattr__(
            self,
            "camera_y0",
            _require_camera_frame(self.camera_y0, label="camera_y0"),
        )
        object.__setattr__(
            self,
            "camera_y1",
            _require_camera_frame(self.camera_y1, label="camera_y1"),
        )
        if (
            type(self.proofs) is not tuple
            or len(self.proofs) != 2
            or any(
                type(proof) is not Factor2ExactVerification or not proof.certified_exact or not proof.numerator_exact
                for proof in self.proofs
            )
        ):
            raise TaskspaceSelectedPreimageProgramError("factor2 pair requires two exact integer realization proofs")


@dataclass(frozen=True, slots=True)
class SelectedPreimageDecodedPairV1:
    """One streamable coupled pair with its exact program/target custody."""

    pair_index: int
    source_pair_id: int
    segment_index: int
    segment_count: int
    scorer_y0: np.ndarray
    scorer_y1: np.ndarray
    program_packet_sha256: str
    target_custody_receipt_sha256: str
    target_bank_sha256: str

    def __post_init__(self) -> None:
        _require_int(self.pair_index, label="pair_index", maximum=PAIR_COUNT_N600 - 1)
        _require_int(self.source_pair_id, label="source_pair_id", maximum=PAIR_COUNT_N600 - 1)
        _require_int(self.segment_index, label="segment_index")
        _require_int(self.segment_count, label="segment_count", minimum=1)
        if self.segment_index >= self.segment_count:
            raise TaskspaceSelectedPreimageProgramError("segment index escapes segment population")
        object.__setattr__(
            self,
            "scorer_y0",
            _require_plane(self.scorer_y0, label="stream scorer_y0"),
        )
        object.__setattr__(
            self,
            "scorer_y1",
            _require_plane(self.scorer_y1, label="stream scorer_y1"),
        )
        for field_name in (
            "program_packet_sha256",
            "target_custody_receipt_sha256",
            "target_bank_sha256",
        ):
            _require_sha256(getattr(self, field_name), label=field_name)


@runtime_checkable
class TaskspaceSelectedPreimageDecoderV1(Protocol):
    """Pluggable audited receiver; packet bytes never select implementation."""

    @property
    def decoder_id(self) -> str: ...

    @property
    def implementation_source_sha256(self) -> str: ...

    def verify_semantic_program_identity(
        self,
        identity: V15SemanticProgramIdentityV1,
    ) -> bool: ...

    def verify_target_custody_identity(
        self,
        identity: ScorerTargetCustodyIdentityV1,
    ) -> bool: ...

    def decode_semantic_base_pair(
        self,
        source_pair_id: int,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def learned_quotient_decoder_contract_id(
        self,
        factor: TaskspaceSelectedPreimageFactorV1,
    ) -> str: ...

    def learned_quotient_decoder_implementation_source_sha256(
        self,
        factor: TaskspaceSelectedPreimageFactorV1,
    ) -> str: ...

    def apply_learned_irreducible_quotient(
        self,
        *,
        factor: TaskspaceSelectedPreimageFactorV1,
        source_pair_id: int,
        scorer_y0: np.ndarray,
        scorer_y1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def realize_factor2_pair(
        self,
        scorer_y0: np.ndarray,
        scorer_y1: np.ndarray,
    ) -> SelectedPreimageFactor2PairV1: ...


LearnedQuotientDecoderV1 = Callable[
    [TaskspaceSelectedPreimageFactorV1, int, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]


@dataclass(frozen=True, slots=True)
class BoundV10Factor2SelectedPreimageDecoderV1:
    """Concrete receiver binding a fresh carrier program and factor-2 operator."""

    semantic_identity: V15SemanticProgramIdentityV1
    target_custody_identity: ScorerTargetCustodyIdentityV1
    carrier_receiver: CarrierComposeReceiverV1
    factor2_operator: DisjointResizeOperator
    learned_quotient_decoder: LearnedQuotientDecoderV1 | None = None
    learned_quotient_decoder_contract_id_value: str | None = None

    def __post_init__(self) -> None:
        if type(self.semantic_identity) is not V15SemanticProgramIdentityV1:
            raise TaskspaceSelectedPreimageProgramError("bound decoder requires exact semantic identity")
        if type(self.target_custody_identity) is not ScorerTargetCustodyIdentityV1:
            raise TaskspaceSelectedPreimageProgramError("bound decoder requires exact target custody identity")
        if type(self.carrier_receiver) is not CarrierComposeReceiverV1:
            raise TaskspaceSelectedPreimageProgramError("bound decoder requires exact CarrierComposeReceiverV1")
        if type(self.factor2_operator) is not DisjointResizeOperator:
            raise TaskspaceSelectedPreimageProgramError("bound decoder requires exact DisjointResizeOperator")
        if _sha256(self.carrier_receiver.archive) != (self.semantic_identity.compiled_semantic_archive_sha256):
            raise TaskspaceSelectedPreimageProgramError(
                "fresh semantic archive bytes do not match the compile identity"
            )
        if len(self.carrier_receiver.archive) != self.semantic_identity.compiled_semantic_archive_bytes:
            raise TaskspaceSelectedPreimageProgramError(
                "fresh semantic archive byte count does not match the compile identity"
            )
        if self.semantic_identity.receiver_source_sha256 != _source_sha256(CarrierComposeReceiverV1):
            raise TaskspaceSelectedPreimageProgramError("carrier receiver source differs from the fresh identity")
        if (
            self.carrier_receiver.predictor.source_pair_start != self.semantic_identity.source_pair_start
            or self.carrier_receiver.z.n_pairs != self.semantic_identity.pair_count
        ):
            raise TaskspaceSelectedPreimageProgramError("carrier receiver pair population differs from identity")
        if (
            self.factor2_operator.camera_h,
            self.factor2_operator.camera_w,
            self.factor2_operator.scorer_h,
            self.factor2_operator.scorer_w,
        ) != (
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            SCORER_HEIGHT,
            SCORER_WIDTH,
        ):
            raise TaskspaceSelectedPreimageProgramError("factor2 operator geometry differs from the generic decoder ID")
        if (self.learned_quotient_decoder is None) != (self.learned_quotient_decoder_contract_id_value is None):
            raise TaskspaceSelectedPreimageProgramError(
                "learned quotient decoder and its contract ID must be bound together"
            )
        if self.learned_quotient_decoder is not None:
            if not callable(self.learned_quotient_decoder):
                raise TaskspaceSelectedPreimageProgramError("learned quotient decoder must be callable or absent")
            _require_id(
                self.learned_quotient_decoder_contract_id_value,
                label="learned_quotient_decoder_contract_id_value",
            )
            _source_sha256(self.learned_quotient_decoder)

    @property
    def decoder_id(self) -> str:
        return GENERIC_V10_FACTOR2_DECODER_ID

    @property
    def implementation_source_sha256(self) -> str:
        return _source_sha256(realize_factor2_uint8_scorer_plane)

    def verify_semantic_program_identity(
        self,
        identity: V15SemanticProgramIdentityV1,
    ) -> bool:
        return type(identity) is V15SemanticProgramIdentityV1 and identity == (self.semantic_identity)

    def verify_target_custody_identity(
        self,
        identity: ScorerTargetCustodyIdentityV1,
    ) -> bool:
        return type(identity) is ScorerTargetCustodyIdentityV1 and identity == (self.target_custody_identity)

    def decode_semantic_base_pair(
        self,
        source_pair_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        _require_int(
            source_pair_id,
            label="source_pair_id",
            minimum=self.semantic_identity.source_pair_start,
            maximum=(self.semantic_identity.source_pair_start + self.semantic_identity.pair_count - 1),
        )
        local_pair_id = source_pair_id - self.semantic_identity.source_pair_start
        rendered = self.carrier_receiver.render_pairs((local_pair_id,))
        expected = (1, 2, SCORER_HEIGHT, SCORER_WIDTH, CHANNELS)
        if rendered.dtype != np.uint8 or rendered.shape != expected:
            raise TaskspaceSelectedPreimageProgramError("carrier receiver did not return one coupled scorer-plane pair")
        return (
            _require_plane(rendered[0, 0], label="semantic scorer_y0"),
            _require_plane(rendered[0, 1], label="semantic scorer_y1"),
        )

    def learned_quotient_decoder_contract_id(
        self,
        factor: TaskspaceSelectedPreimageFactorV1,
    ) -> str:
        del factor
        if self.learned_quotient_decoder_contract_id_value is None:
            raise TaskspaceSelectedPreimageProgramError(
                "learned irreducible quotient factor has no bound decoder contract"
            )
        return self.learned_quotient_decoder_contract_id_value

    def learned_quotient_decoder_implementation_source_sha256(
        self,
        factor: TaskspaceSelectedPreimageFactorV1,
    ) -> str:
        del factor
        if self.learned_quotient_decoder is None:
            raise TaskspaceSelectedPreimageProgramError(
                "learned irreducible quotient factor has no bound decoder source"
            )
        return _source_sha256(self.learned_quotient_decoder)

    def apply_learned_irreducible_quotient(
        self,
        *,
        factor: TaskspaceSelectedPreimageFactorV1,
        source_pair_id: int,
        scorer_y0: np.ndarray,
        scorer_y1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.learned_quotient_decoder is None:
            raise TaskspaceSelectedPreimageProgramError("learned irreducible quotient factor has no bound decoder")
        before0 = _require_plane(scorer_y0, label="learned input y0")
        before1 = _require_plane(scorer_y1, label="learned input y1")
        hash0, hash1 = _sha256(before0), _sha256(before1)
        output = self.learned_quotient_decoder(
            factor,
            source_pair_id,
            before0,
            before1,
        )
        if _sha256(before0) != hash0 or _sha256(before1) != hash1:
            raise TaskspaceSelectedPreimageProgramError("learned quotient decoder mutated its immutable inputs")
        if type(output) is not tuple or len(output) != 2:
            raise TaskspaceSelectedPreimageProgramError("learned quotient decoder did not return coupled planes")
        return (
            _require_plane(output[0], label="learned output y0"),
            _require_plane(output[1], label="learned output y1"),
        )

    def realize_factor2_pair(
        self,
        scorer_y0: np.ndarray,
        scorer_y1: np.ndarray,
    ) -> SelectedPreimageFactor2PairV1:
        y0 = _require_plane(scorer_y0, label="factor2 scorer_y0")
        y1 = _require_plane(scorer_y1, label="factor2 scorer_y1")
        try:
            camera0 = realize_factor2_uint8_scorer_plane(
                self.factor2_operator,
                y0,
            )
            camera1 = realize_factor2_uint8_scorer_plane(
                self.factor2_operator,
                y1,
            )
            proof0 = verify_factor2_uint8_scorer_plane(
                self.factor2_operator,
                camera0,
                y0,
            )
            proof1 = verify_factor2_uint8_scorer_plane(
                self.factor2_operator,
                camera1,
                y1,
            )
        except Uint8LatticeError as exc:
            raise TaskspaceSelectedPreimageProgramError("generic factor2 realization failed") from exc
        return SelectedPreimageFactor2PairV1(
            scorer_y0=y0,
            scorer_y1=y1,
            camera_y0=camera0,
            camera_y1=camera1,
            proofs=(proof0, proof1),
        )


def _apply_analytic_factor(
    factor: TaskspaceSelectedPreimageFactorV1,
    *,
    source_pair_id: int,
    scorer_y0: np.ndarray,
    scorer_y1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    analytic = _parse_analytic_payload(factor.payload, factor=factor)
    atoms = tuple(atom for atom in analytic.atoms if atom.pair_index == source_pair_id)
    if not atoms:
        return scorer_y0, scorer_y1
    outputs = [
        np.ascontiguousarray(scorer_y0).copy(),
        np.ascontiguousarray(scorer_y1).copy(),
    ]
    selected_frames = {
        SelectedPreimageFrameSelectorV1.Y0: (0,),
        SelectedPreimageFrameSelectorV1.Y1: (1,),
        SelectedPreimageFrameSelectorV1.BOTH: (0, 1),
    }[analytic.frame_selector]
    source_rgb = np.asarray(analytic.source_rgb_u8, dtype=np.uint8)
    for frame_index in selected_frames:
        frame = outputs[frame_index]
        source_mask = np.all(frame == source_rgb, axis=-1)
        warped_mask = _carrier_module._apply_boundary_shearlet_atoms(
            source_mask,
            atoms,
        )
        removed = source_mask & ~warped_mask
        added = warped_mask & ~source_mask
        frame[removed] = analytic.removed_rgb_u8
        frame[added] = analytic.added_rgb_u8
    return (
        _require_plane(outputs[0], label="analytic output y0"),
        _require_plane(outputs[1], label="analytic output y1"),
    )


def _decode_selected_preimage_pair_once(
    program: TaskspaceSelectedPreimageProgramV1,
    pair_index: int,
    decoder: TaskspaceSelectedPreimageDecoderV1,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(decoder, TaskspaceSelectedPreimageDecoderV1):
        raise TaskspaceSelectedPreimageProgramError("decoder does not implement the selected-preimage protocol")
    if decoder.decoder_id != program.decoder_identity.decoder_id:
        raise TaskspaceSelectedPreimageProgramError("decoder ID differs from the counted generic V10 identity")
    if decoder.implementation_source_sha256 != program.decoder_identity.implementation_source_sha256:
        raise TaskspaceSelectedPreimageProgramError(
            "decoder implementation source differs from the counted generic V10 identity"
        )
    if not decoder.verify_semantic_program_identity(program.semantic_program_identity):
        raise TaskspaceSelectedPreimageProgramError("decoder semantic program identity mismatch")
    if not decoder.verify_target_custody_identity(program.target_custody_identity):
        raise TaskspaceSelectedPreimageProgramError("decoder batch16 target-custody identity mismatch")
    _require_int(
        pair_index,
        label="pair_index",
        minimum=0,
        maximum=program.compile_config.pair_count - 1,
    )
    source_pair_id = program.compile_config.source_pair_start + pair_index
    base = decoder.decode_semantic_base_pair(source_pair_id)
    if type(base) is not tuple or len(base) != 2:
        raise TaskspaceSelectedPreimageProgramError("semantic receiver did not return coupled planes")
    y0 = _require_plane(base[0], label="base scorer_y0")
    y1 = _require_plane(base[1], label="base scorer_y1")
    for factor in program.factors:
        if not factor.addresses(source_pair_id):
            continue
        before = (_sha256(y0), _sha256(y1))
        if factor.role is SelectedPreimageFactorRoleV1.ANALYTIC_RESIDUAL:
            y0, y1 = _apply_analytic_factor(
                factor,
                source_pair_id=source_pair_id,
                scorer_y0=y0,
                scorer_y1=y1,
            )
        else:
            learned = _parse_learned_payload(factor.payload, factor=factor)
            if decoder.learned_quotient_decoder_contract_id(factor) != learned.decoder_contract_id:
                raise TaskspaceSelectedPreimageProgramError(
                    "learned quotient decoder contract differs from the counted factor"
                )
            if (
                decoder.learned_quotient_decoder_implementation_source_sha256(factor)
                != learned.decoder_implementation_source_sha256
            ):
                raise TaskspaceSelectedPreimageProgramError(
                    "learned quotient decoder source differs from the counted factor"
                )
            y0, y1 = decoder.apply_learned_irreducible_quotient(
                factor=factor,
                source_pair_id=source_pair_id,
                scorer_y0=y0,
                scorer_y1=y1,
            )
            y0 = _require_plane(y0, label="learned factor scorer_y0")
            y1 = _require_plane(y1, label="learned factor scorer_y1")
        if (_sha256(y0), _sha256(y1)) == before:
            raise TaskspaceSelectedPreimageProgramError(
                f"counted factor {factor.section_id!r} is receiver-inert for source pair {source_pair_id}"
            )
    return y0, y1


def decode_selected_preimage_pair(
    program: TaskspaceSelectedPreimageProgramV1,
    pair_index: int,
    decoder: TaskspaceSelectedPreimageDecoderV1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic exact uint8 scorer-resolution ``(Y0, Y1)``."""

    if type(program) is not TaskspaceSelectedPreimageProgramV1:
        raise TaskspaceSelectedPreimageProgramError("decode requires exact TaskspaceSelectedPreimageProgramV1")
    first = _decode_selected_preimage_pair_once(program, pair_index, decoder)
    second = _decode_selected_preimage_pair_once(program, pair_index, decoder)
    if not np.array_equal(first[0], second[0]) or not np.array_equal(
        first[1],
        second[1],
    ):
        raise TaskspaceSelectedPreimageProgramError("selected-preimage pair decoder is nondeterministic")
    return first


def iter_selected_preimage_segment(
    program: TaskspaceSelectedPreimageProgramV1,
    decoder: TaskspaceSelectedPreimageDecoderV1,
    *,
    segment_index: int,
    pairs_per_segment: int = 120,
) -> Iterator[SelectedPreimageDecodedPairV1]:
    """Stream one deterministic resume segment without materializing an n600 bank.

    At the n600 default this exposes five independent 120-pair elementary
    streams.  A launcher can checkpoint after each exhausted iterator and
    resume at the next ``segment_index`` without decoding prior segments.
    """

    if type(program) is not TaskspaceSelectedPreimageProgramV1:
        raise TaskspaceSelectedPreimageProgramError("stream requires exact TaskspaceSelectedPreimageProgramV1")
    _require_int(
        pairs_per_segment,
        label="pairs_per_segment",
        minimum=1,
        maximum=PAIR_COUNT_N600,
    )
    segment_count = (program.compile_config.pair_count + pairs_per_segment - 1) // pairs_per_segment
    _require_int(
        segment_index,
        label="segment_index",
        maximum=segment_count - 1,
    )
    start = segment_index * pairs_per_segment
    stop = min(start + pairs_per_segment, program.compile_config.pair_count)
    packet_sha256 = program.packet_sha256
    custody = program.target_custody_identity
    for pair_index in range(start, stop):
        y0, y1 = decode_selected_preimage_pair(program, pair_index, decoder)
        yield SelectedPreimageDecodedPairV1(
            pair_index=pair_index,
            source_pair_id=program.compile_config.source_pair_start + pair_index,
            segment_index=segment_index,
            segment_count=segment_count,
            scorer_y0=y0,
            scorer_y1=y1,
            program_packet_sha256=packet_sha256,
            target_custody_receipt_sha256=custody.target_custody_receipt_sha256,
            target_bank_sha256=custody.target_bank_sha256,
        )


def realize_selected_preimage_pair_factor2(
    program: TaskspaceSelectedPreimageProgramV1,
    pair_index: int,
    decoder: TaskspaceSelectedPreimageDecoderV1,
) -> SelectedPreimageFactor2PairV1:
    """Decode coupled scorer planes and realize both through generic factor-2."""

    y0, y1 = decode_selected_preimage_pair(program, pair_index, decoder)
    realized = decoder.realize_factor2_pair(y0, y1)
    if type(realized) is not SelectedPreimageFactor2PairV1:
        raise TaskspaceSelectedPreimageProgramError("decoder did not return exact factor2 pair receipt")
    return realized


__all__ = [
    "ANALYTIC_SCHEMA",
    "GENERIC_V10_FACTOR2_DECODER_ID",
    "LEARNED_SCHEMA",
    "OPEN_SELECTED_SOLUTION_PRODUCT_BLOCKERS",
    "PROGRAM_SCHEMA",
    "V15_SEMANTIC_COMPILE_DERIVATION",
    "V15_SEMANTIC_COMPILE_RECEIPT_SCHEMA",
    "BoundV10Factor2SelectedPreimageDecoderV1",
    "ForbiddenSelectedPreimagePayloadClassV1",
    "GenericV10Factor2DecoderIdentityV1",
    "ScorerTargetCustodyIdentityV1",
    "SelectedPreimageByteHomeRecordV1",
    "SelectedPreimageByteHomeV1",
    "SelectedPreimageCompileConfigV1",
    "SelectedPreimageDecodedPairV1",
    "SelectedPreimageFactor2PairV1",
    "SelectedPreimageFactorModeV1",
    "SelectedPreimageFactorRoleV1",
    "SelectedPreimageFrameSelectorV1",
    "SelectedPreimageLineageClassV1",
    "TaskspaceSelectedPreimageDecoderV1",
    "TaskspaceSelectedPreimageFactorV1",
    "TaskspaceSelectedPreimageProgramError",
    "TaskspaceSelectedPreimageProgramV1",
    "V15SemanticProgramIdentityV1",
    "build_analytic_shearlet_residual_factor",
    "build_learned_irreducible_quotient_factor",
    "compile_v9_v10_selected_preimage_program",
    "decode_selected_preimage_pair",
    "encode_selected_preimage_program",
    "iter_selected_preimage_segment",
    "parse_selected_preimage_program",
    "realize_selected_preimage_pair_factor2",
    "refuse_forbidden_selected_preimage_payload",
    "verify_v15_semantic_compile_lineage",
]
