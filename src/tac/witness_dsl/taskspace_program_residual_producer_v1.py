# SPDX-License-Identifier: MIT
"""Typed, fail-closed producer seam for a G49 selected-preimage residual program.

This module deliberately does not promote G49 into the primary task-space
codec.  G49 V1 is a terminal residual transport with only two closed operand
roles.  Its generic analytic/learned decoder algorithms are free code, while
every video-derived analytic atom, latent, and parameter remains counted inside
the G49 packet.  Shared topology/worldsheet state, temporal transitions and
island birth/death, conditional Y0 given decoded Y1 pose enhancement, entropy
contexts, and joint physical coding groups already exist in the canonical G17
selected-solution ontology. The owed work is the executable G17 archive
producer and public receiver operation registry, not another schema or factor
vocabulary.

The executable part here is real and narrow:

* reopen an exact G49 packet and all source-bound production custody;
* publish a G58-compatible pre-encode identity;
* decode each of the five 120-pair stages directly from its G49 segment;
* preserve immutable, no-redecode stage receipts;
* build and reopen the exact two-member counted outer ZIP;
* publish the G58 terminal/outer proof artifacts; and
* feed those exact artifacts to G59.

G59 must still refuse candidate admission with the G17-owned
``G17_PRODUCTION_TERMINAL_ENVELOPE_RECEIVER_OWED``. That refusal is the
truthful output of the current V1 contract, not G63-owned primary-codec state
or a missing implementation hidden behind a success flag.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np

from tac.optimization.direct_description_carrier_compose import (
    receive_carrier_compose_archive,
)
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tac.witness_control.taskspace_fresh_scorer_plane_materializer_v1 import (
    FreshScorerPlaneOperandLoaderV1,
)
from tac.witness_dsl.taskspace_selected_preimage_operand_adapter_v1 import (
    ADMISSION_SCHEMA,
    BLOCKED_REPRESENTATION_STATUS,
    NEXT_PRECLOSURE_GATE,
    PRODUCTION_PRE_ENCODE_EVIDENCE_SCHEMA,
    PROGRAM_RESIDUAL_MODE,
    TERMINAL_STAGE_CHAIN_SCHEMA,
    AuxiliaryOperandCustodyV1,
    LearnedDecoderSourceCustodyV1,
    ProgramResidualOuterArchiveProofV1,
    ReopenableRegularFileIdentityV1,
    SelectedPreimageFreshOperandStageV1,
    SelectedPreimagePreEncodeAdmissionV1,
    SelectedPreimageProductionCustodyV1,
    TaskspaceSelectedPreimageFreshOperandAdapterV1,
    publish_program_residual_outer_archive_proof,
    reopen_program_residual_outer_archive,
    reopen_program_residual_production_pre_encode_evidence,
    validate_pre_encode_stage,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CHANNELS,
    PAIR_COUNT_N600,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    BoundV10Factor2SelectedPreimageDecoderV1,
    SelectedPreimageByteHomeV1,
    SelectedPreimageFactorRoleV1,
    SelectedPreimageLineageClassV1,
    TaskspaceSelectedPreimageProgramError,
    TaskspaceSelectedPreimageProgramV1,
    iter_selected_preimage_segment,
    parse_selected_preimage_program,
)

CONFIG_SCHEMA: Final = "tac.taskspace_program_residual_producer_config.v1"
EXAMPLE_CONFIG_SCHEMA: Final = "tac.taskspace_program_residual_producer_config_example.v1"
STAGE_RECEIPT_SCHEMA: Final = "tac.taskspace_program_residual_stage_checkpoint.v1"
RUN_RECEIPT_SCHEMA: Final = "tac.taskspace_program_residual_producer_run.v1"
PRODUCER_ROLE: Final = "G49_TERMINAL_RESIDUAL_TRANSPORT_V1"
PROGRAM_RESIDUAL_LAYERED: Final = "PROGRAM_RESIDUAL_LAYERED"

PRODUCTION_PAIR_COUNT: Final = 600
PRODUCTION_PAIRS_PER_STAGE: Final = 120
PRODUCTION_STAGE_COUNT: Final = 5
SCORER_BATCH_SIZE: Final = 16

CLOSED_G49_FACTOR_VOCABULARY: Final = (
    "SHEARLET_BOUNDARY_TRANSPORT_Q4",
    "COMPACT_LATENT_QUOTIENT_PLUGIN",
)
SSD_ROOTS: Final = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE: Final = re.compile(r"[A-Za-z0-9_.:+-]{1,192}\Z")
_MAX_CONFIG_BYTES: Final = 4 * 1024 * 1024
_MAX_SOURCE_BYTES: Final = 128 * 1024 * 1024
_FORBIDDEN_MEMBER_TOKENS: Final = frozenset(
    {
        "argmax",
        "gt",
        "label",
        "labels",
        "pose",
        "poses",
        "raw",
        "scorer",
        "source",
        "target",
        "teacher",
        "y0",
        "y1",
    }
)

_TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "schema",
        "run_id",
        "campaign_id",
        "requested_representation",
        "producer_role",
        "execution_ready",
        "pair_count",
        "pairs_per_stage",
        "stage_count",
        "scorer_batch_size",
        "program_packet",
        "semantic_archive",
        "semantic_compile_receipt",
        "target_custody_receipt",
        "auxiliary_aggregate_receipt",
        "compiler_source",
        "generic_v10_source",
        "learned_decoder_sources",
        "campaign_seal_receipt",
        "factor_operands",
        "outer_archive_members",
        "output_paths",
        "required_free_bytes",
        "resume_required",
        "test_only_small_fixture",
        "truth",
    }
)
_OUTPUT_FIELDS: Final = frozenset(
    {
        "output_root",
        "stage_checkpoint_dir",
        "g58_identity_receipt",
        "g58_terminal_stage_chain_receipt",
        "outer_archive",
        "g58_outer_proof_receipt",
        "g59_pre_encode_receipt",
        "run_receipt",
    }
)
_TRUTH_FIELDS: Final = frozenset(
    {
        "research_only",
        "score_claim",
        "candidate_claim",
        "evaluation_claim",
        "promotion_eligible",
        "historical_payload_reused",
        "direct_source_plane_fallback_allowed",
        "raw_labels_embedded",
        "source_planes_embedded",
        "generic_algorithm_code_free",
        "all_video_derived_operands_counted",
    }
)
_FACTOR_FIELDS: Final = frozenset(
    {
        "section_id",
        "role",
        "mode",
        "source_pair_start",
        "source_pair_stop_exclusive",
        "source_receipt_sha256",
        "payload_bytes",
        "payload_sha256",
        "operand_byte_home",
        "operand_lineage_class",
        "generic_algorithm_byte_home",
        "operand_payload_counted",
    }
)


class ProgramResidualProducerError(RuntimeError):
    """A typed config, source identity, resume, or G58/G59 join failed."""


def canonical_json(value: Any) -> bytes:
    return canonical_json_body(value) + b"\n"


def canonical_json_body(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def sha256_bytes(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ProgramResidualProducerError(f"{label} must be lowercase SHA-256")
    return value


def _require_id(value: object, label: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise ProgramResidualProducerError(f"{label} must be a path-free typed ID")
    return value


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProgramResidualProducerError(f"{label} must be an integer >= {minimum}")
    return value


def _require_exact_mapping(
    value: object,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != fields:
        actual = frozenset(value) if isinstance(value, Mapping) else frozenset()
        raise ProgramResidualProducerError(
            f"{label} fields differ: missing={sorted(fields - actual)}, extra={sorted(actual - fields)}"
        )
    return value


@dataclass(frozen=True, slots=True)
class StableFileIdentityV1:
    path: str
    bytes: int
    sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def _lexical_absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def stable_read_regular_file(
    path: str | os.PathLike[str],
    *,
    label: str,
    maximum_bytes: int = _MAX_SOURCE_BYTES,
) -> tuple[StableFileIdentityV1, bytes]:
    """Read one no-follow descriptor and reject path replacement or drift."""

    target = _lexical_absolute(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise ProgramResidualProducerError(f"{label} cannot open as a no-follow regular file: {target}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ProgramResidualProducerError(f"{label} is not a regular file")
        if before.st_size < 1 or before.st_size > maximum_bytes:
            raise ProgramResidualProducerError(f"{label} bytes {before.st_size} escape 1..{maximum_bytes}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ProgramResidualProducerError(f"{label} truncated during descriptor read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ProgramResidualProducerError(f"{label} grew during descriptor read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        named = os.stat(target, follow_symlinks=False)
    except OSError as exc:
        raise ProgramResidualProducerError(f"{label} disappeared after descriptor read") from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    named_identity = (
        named.st_dev,
        named.st_ino,
        named.st_mode,
        named.st_size,
        named.st_mtime_ns,
    )
    if not stat.S_ISREG(named.st_mode) or before_identity != after_identity or after_identity != named_identity:
        raise ProgramResidualProducerError(f"{label} identity changed during descriptor read")
    payload = b"".join(chunks)
    return (
        StableFileIdentityV1(
            path=str(target),
            bytes=len(payload),
            sha256=sha256_bytes(payload),
        ),
        payload,
    )


def stable_file_identity(
    path: str | os.PathLike[str],
    *,
    label: str,
    maximum_bytes: int = _MAX_SOURCE_BYTES,
) -> StableFileIdentityV1:
    return stable_read_regular_file(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
    )[0]


def _binding_from_mapping(value: object, label: str) -> StableFileIdentityV1:
    row = _require_exact_mapping(value, frozenset({"path", "bytes", "sha256"}), label)
    if type(row["path"]) is not str or not Path(row["path"]).is_absolute():
        raise ProgramResidualProducerError(f"{label}.path must be absolute")
    expected = StableFileIdentityV1(
        path=str(_lexical_absolute(row["path"])),
        bytes=_require_int(row["bytes"], f"{label}.bytes", minimum=1),
        sha256=_require_sha256(row["sha256"], f"{label}.sha256"),
    )
    observed = stable_file_identity(expected.path, label=label)
    if observed != expected:
        raise ProgramResidualProducerError(f"{label} descriptor-stable identity differs")
    return expected


def factor_operand_inventory(
    program: TaskspaceSelectedPreimageProgramV1,
) -> tuple[dict[str, Any], ...]:
    """Return the exact counted operand inventory, with free code separated."""

    if type(program) is not TaskspaceSelectedPreimageProgramV1:
        raise ProgramResidualProducerError("factor inventory requires exact G49 program")
    rows: list[dict[str, Any]] = []
    for factor in program.factors:
        rows.append(
            {
                "section_id": factor.section_id,
                "role": factor.role.value,
                "mode": factor.mode.value,
                "source_pair_start": factor.source_pair_start,
                "source_pair_stop_exclusive": factor.source_pair_stop_exclusive,
                "source_receipt_sha256": factor.source_receipt_sha256,
                "payload_bytes": len(factor.payload),
                "payload_sha256": factor.payload_sha256,
                "operand_byte_home": factor.byte_home.value,
                "operand_lineage_class": factor.lineage_class.value,
                "generic_algorithm_byte_home": (SelectedPreimageByteHomeV1.GENERIC_DECODER_CODE_FREE.value),
                "operand_payload_counted": True,
            }
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class ProgramResidualProducerConfigV1:
    path: Path
    file_identity: StableFileIdentityV1
    raw: Mapping[str, Any]
    run_id: str
    campaign_id: str
    pair_count: int
    pairs_per_stage: int
    stage_count: int
    program: TaskspaceSelectedPreimageProgramV1
    program_packet: StableFileIdentityV1
    program_packet_bytes: bytes
    semantic_archive: StableFileIdentityV1
    semantic_archive_bytes: bytes
    semantic_compile_receipt: StableFileIdentityV1
    target_custody_receipt: StableFileIdentityV1
    auxiliary_aggregate_receipt: StableFileIdentityV1
    compiler_source: StableFileIdentityV1
    generic_v10_source: StableFileIdentityV1
    learned_decoder_sources: tuple[Mapping[str, Any], ...]
    campaign_seal_receipt: StableFileIdentityV1
    factor_operands: tuple[Mapping[str, Any], ...]
    output_paths: Mapping[str, Path]
    outer_archive_members: tuple[str, str]
    required_free_bytes: int
    test_only_small_fixture: bool


def _validate_truth(value: object) -> None:
    row = _require_exact_mapping(value, _TRUTH_FIELDS, "truth")
    if row != {
        "research_only": True,
        "score_claim": False,
        "candidate_claim": False,
        "evaluation_claim": False,
        "promotion_eligible": False,
        "historical_payload_reused": False,
        "direct_source_plane_fallback_allowed": False,
        "raw_labels_embedded": False,
        "source_planes_embedded": False,
        "generic_algorithm_code_free": True,
        "all_video_derived_operands_counted": True,
    }:
        raise ProgramResidualProducerError("producer truth boundary was weakened")


def _safe_member_name(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ProgramResidualProducerError(f"{label} must be a nonempty member name")
    name = PurePosixPath(value)
    tokens = set(re.findall(r"[a-z0-9]+", value.lower()))
    if (
        name.is_absolute()
        or ".." in name.parts
        or value.endswith("/")
        or any(token in tokens for token in _FORBIDDEN_MEMBER_TOKENS)
    ):
        raise ProgramResidualProducerError(f"{label} is unsafe or names forbidden payload")
    return value


def _validate_output_paths(
    value: object,
    *,
    test_only_small_fixture: bool,
) -> Mapping[str, Path]:
    row = _require_exact_mapping(value, _OUTPUT_FIELDS, "output_paths")
    paths: dict[str, Path] = {}
    for key, raw in row.items():
        if type(raw) is not str or not Path(raw).is_absolute():
            raise ProgramResidualProducerError(f"output_paths.{key} must be absolute")
        paths[key] = _lexical_absolute(raw)
    root = paths["output_root"]
    if any(path != root and not path.is_relative_to(root) for key, path in paths.items() if key != "output_root"):
        raise ProgramResidualProducerError("every output path must be rooted under output_root")
    if len(set(paths.values())) != len(paths):
        raise ProgramResidualProducerError("output paths must be distinct")
    if not test_only_small_fixture and not any(
        root != ssd.resolve() and root.is_relative_to(ssd.resolve()) for ssd in SSD_ROOTS
    ):
        raise ProgramResidualProducerError("production output_root must be a child of the SSD waterfall")
    return paths


def _parse_execution_config(
    *,
    config_path: Path,
    config_identity: StableFileIdentityV1,
    raw: Mapping[str, Any],
) -> ProgramResidualProducerConfigV1:
    if raw["schema"] != CONFIG_SCHEMA:
        raise ProgramResidualProducerError("producer config schema drift")
    _require_id(raw["run_id"], "run_id")
    _require_id(raw["campaign_id"], "campaign_id")
    if (
        raw["requested_representation"] != PROGRAM_RESIDUAL_LAYERED
        or raw["producer_role"] != PRODUCER_ROLE
        or raw["execution_ready"] is not True
        or raw["resume_required"] is not True
    ):
        raise ProgramResidualProducerError("producer role/readiness/resume contract differs")
    test_only = raw["test_only_small_fixture"]
    if type(test_only) is not bool:
        raise ProgramResidualProducerError("test_only_small_fixture must be boolean")
    pair_count = _require_int(raw["pair_count"], "pair_count", minimum=1)
    pairs_per_stage = _require_int(raw["pairs_per_stage"], "pairs_per_stage", minimum=1)
    stage_count = _require_int(raw["stage_count"], "stage_count", minimum=1)
    if pair_count != pairs_per_stage * stage_count:
        raise ProgramResidualProducerError("stage lattice does not exactly cover pair_count")
    if not test_only and (
        pair_count,
        pairs_per_stage,
        stage_count,
        raw["scorer_batch_size"],
    ) != (
        PRODUCTION_PAIR_COUNT,
        PRODUCTION_PAIRS_PER_STAGE,
        PRODUCTION_STAGE_COUNT,
        SCORER_BATCH_SIZE,
    ):
        raise ProgramResidualProducerError("production must be exact n600/five-by-120/batch16")
    if raw["scorer_batch_size"] != SCORER_BATCH_SIZE:
        raise ProgramResidualProducerError("selected-preimage target geometry must remain batch16")
    _validate_truth(raw["truth"])

    program_packet, packet_payload = (
        _binding_from_mapping(raw["program_packet"], "program_packet"),
        None,
    )
    packet_identity, packet_bytes = stable_read_regular_file(
        program_packet.path,
        label="program_packet",
    )
    if packet_identity != program_packet:
        raise ProgramResidualProducerError("program packet changed between descriptor-stable reads")
    try:
        program = parse_selected_preimage_program(
            packet_bytes,
            maximum_packet_bytes=len(packet_bytes),
        )
    except TaskspaceSelectedPreimageProgramError as exc:
        raise ProgramResidualProducerError("program packet is not exact G49 V1") from exc
    if (
        program.compile_config.source_pair_start != 0
        or program.compile_config.pair_count != pair_count
        or program.target_custody_identity.scorer_batch_size != SCORER_BATCH_SIZE
        or program.target_custody_identity.pair_count != PAIR_COUNT_N600
    ):
        raise ProgramResidualProducerError("program population or target coordinate differs")
    del packet_payload

    semantic = _binding_from_mapping(raw["semantic_archive"], "semantic_archive")
    semantic_identity, semantic_bytes = stable_read_regular_file(
        semantic.path,
        label="semantic_archive",
    )
    if semantic_identity != semantic:
        raise ProgramResidualProducerError("semantic archive changed between descriptor-stable reads")
    semantic_program = program.semantic_program_identity
    if (
        semantic.sha256 != semantic_program.compiled_semantic_archive_sha256
        or semantic.bytes != semantic_program.compiled_semantic_archive_bytes
    ):
        raise ProgramResidualProducerError("semantic archive differs from G49 fresh identity")
    semantic_receipt = _binding_from_mapping(
        raw["semantic_compile_receipt"],
        "semantic_compile_receipt",
    )
    target_receipt = _binding_from_mapping(
        raw["target_custody_receipt"],
        "target_custody_receipt",
    )
    auxiliary = _binding_from_mapping(
        raw["auxiliary_aggregate_receipt"],
        "auxiliary_aggregate_receipt",
    )
    compiler = _binding_from_mapping(raw["compiler_source"], "compiler_source")
    generic = _binding_from_mapping(raw["generic_v10_source"], "generic_v10_source")
    campaign = _binding_from_mapping(raw["campaign_seal_receipt"], "campaign_seal_receipt")
    if (
        semantic_receipt.sha256 != semantic_program.fresh_compile_receipt_sha256
        or compiler.sha256 != semantic_program.compiler_source_sha256
        or generic.sha256 != program.decoder_identity.implementation_source_sha256
    ):
        raise ProgramResidualProducerError("semantic/compiler/generic source lineage differs from G49")

    learned_rows = raw["learned_decoder_sources"]
    if type(learned_rows) is not list:
        raise ProgramResidualProducerError("learned_decoder_sources must be a canonical list")
    canonical_learned: list[Mapping[str, Any]] = []
    seen_learned: set[str] = set()
    for index, value in enumerate(learned_rows):
        row = _require_exact_mapping(
            value,
            frozenset({"section_id", "decoder_contract_id", "implementation_source"}),
            f"learned_decoder_sources[{index}]",
        )
        section_id = _require_id(row["section_id"], f"learned_decoder_sources[{index}].section_id")
        contract_id = _require_id(
            row["decoder_contract_id"],
            f"learned_decoder_sources[{index}].decoder_contract_id",
        )
        if section_id in seen_learned:
            raise ProgramResidualProducerError("learned decoder source repeats section_id")
        seen_learned.add(section_id)
        source = _binding_from_mapping(
            row["implementation_source"],
            f"learned_decoder_sources[{index}].implementation_source",
        )
        canonical_learned.append(
            {
                "section_id": section_id,
                "decoder_contract_id": contract_id,
                "implementation_source": source.to_mapping(),
            }
        )
    learned_factor_ids = {
        factor.section_id
        for factor in program.factors
        if factor.role is SelectedPreimageFactorRoleV1.LEARNED_IRREDUCIBLE_QUOTIENT
    }
    if seen_learned != learned_factor_ids:
        raise ProgramResidualProducerError("learned decoder sources do not cover exact learned factors")
    if learned_factor_ids:
        raise ProgramResidualProducerError(
            "LEARNED_QUOTIENT_G17_PLACEMENT_OWED: learned decoder source/runtime "
            "has no canonical generic-versus-video-derived placement proof"
        )

    configured_factors = raw["factor_operands"]
    if type(configured_factors) is not list:
        raise ProgramResidualProducerError("factor_operands must be a canonical list")
    for index, row in enumerate(configured_factors):
        _require_exact_mapping(row, _FACTOR_FIELDS, f"factor_operands[{index}]")
    expected_factors = factor_operand_inventory(program)
    if configured_factors != list(expected_factors):
        raise ProgramResidualProducerError("factor operand inventory differs from parsed G49 packet")
    if any(
        row["generic_algorithm_byte_home"] != SelectedPreimageByteHomeV1.GENERIC_DECODER_CODE_FREE.value
        or row["operand_payload_counted"] is not True
        or row["operand_lineage_class"]
        not in {
            SelectedPreimageLineageClassV1.VIDEO_DERIVED_ANALYTIC_FACTOR.value,
            SelectedPreimageLineageClassV1.VIDEO_DERIVED_LEARNED_IRREDUCIBLE_FACTOR.value,
        }
        for row in expected_factors
    ):
        raise ProgramResidualProducerError("video-derived operands escaped counted byte homes")

    outer = _require_exact_mapping(
        raw["outer_archive_members"],
        frozenset({"semantic_member_name", "program_member_name"}),
        "outer_archive_members",
    )
    member_names = (
        _safe_member_name(outer["semantic_member_name"], "semantic_member_name"),
        _safe_member_name(outer["program_member_name"], "program_member_name"),
    )
    if member_names[0] == member_names[1]:
        raise ProgramResidualProducerError("outer archive members must be distinct")
    outputs = _validate_output_paths(
        raw["output_paths"],
        test_only_small_fixture=test_only,
    )
    required_free = _require_int(
        raw["required_free_bytes"],
        "required_free_bytes",
        minimum=1,
    )
    return ProgramResidualProducerConfigV1(
        path=config_path,
        file_identity=config_identity,
        raw=raw,
        run_id=raw["run_id"],
        campaign_id=raw["campaign_id"],
        pair_count=pair_count,
        pairs_per_stage=pairs_per_stage,
        stage_count=stage_count,
        program=program,
        program_packet=program_packet,
        program_packet_bytes=packet_bytes,
        semantic_archive=semantic,
        semantic_archive_bytes=semantic_bytes,
        semantic_compile_receipt=semantic_receipt,
        target_custody_receipt=target_receipt,
        auxiliary_aggregate_receipt=auxiliary,
        compiler_source=compiler,
        generic_v10_source=generic,
        learned_decoder_sources=tuple(canonical_learned),
        campaign_seal_receipt=campaign,
        factor_operands=tuple(expected_factors),
        output_paths=outputs,
        outer_archive_members=member_names,
        required_free_bytes=required_free,
        test_only_small_fixture=test_only,
    )


def load_config(
    path: str | os.PathLike[str],
    *,
    require_execution_ready: bool = True,
) -> ProgramResidualProducerConfigV1:
    identity, payload = stable_read_regular_file(
        path,
        label="program residual producer config",
        maximum_bytes=_MAX_CONFIG_BYTES,
    )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProgramResidualProducerError("producer config is not JSON") from exc
    raw = _require_exact_mapping(value, _TOP_LEVEL_FIELDS, "producer config")
    if raw["schema"] == EXAMPLE_CONFIG_SCHEMA and raw["execution_ready"] is False:
        if require_execution_ready:
            raise ProgramResidualProducerError(
                "typed example is deliberately non-executable; fresh G49 custody is required"
            )
        raise ProgramResidualProducerError("non-executable examples do not instantiate ProgramResidualProducerConfigV1")
    return _parse_execution_config(
        config_path=Path(identity.path),
        config_identity=identity,
        raw=raw,
    )


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return sha256_bytes(memoryview(contiguous).cast("B"))


def _ensure_output_root(config: ProgramResidualProducerConfigV1) -> Mapping[str, Any]:
    root = config.output_paths["output_root"]
    nearest = root
    while not nearest.exists() and nearest != nearest.parent:
        nearest = nearest.parent
    usage = shutil.disk_usage(nearest)
    if usage.free < config.required_free_bytes:
        raise ProgramResidualProducerError(f"storage preflight needs {config.required_free_bytes}, has {usage.free}")
    root.mkdir(parents=True, exist_ok=True)
    current = root
    while current != current.parent:
        if current.exists() and current.is_symlink():
            raise ProgramResidualProducerError(f"output path traverses symlink: {current}")
        if current == nearest:
            break
        current = current.parent
    return {
        "output_root": str(root),
        "nearest_existing_parent": str(nearest.resolve()),
        "free_bytes": int(usage.free),
        "required_free_bytes": config.required_free_bytes,
        "admitted": True,
        "completed_stage_policy": "immutable_receipt_no_redecode",
        "scratch_policy": "success_only_atomic_temp",
    }


def publish_write_once(
    path: str | os.PathLike[str],
    payload: bytes,
    *,
    label: str,
) -> StableFileIdentityV1:
    if type(payload) is not bytes or not payload:
        raise ProgramResidualProducerError(f"{label} payload must be nonempty bytes")
    target = _lexical_absolute(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        identity, current = stable_read_regular_file(target, label=f"preserved {label}")
        if current != payload:
            raise ProgramResidualProducerError(f"preserved write-once {label} drifted")
        return identity
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".partial",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            identity, current = stable_read_regular_file(target, label=f"raced {label}")
            if current != payload:
                raise ProgramResidualProducerError(f"write-once {label} raced with different bytes") from None
            return identity
        return stable_file_identity(target, label=label)
    finally:
        temporary.unlink(missing_ok=True)


def _deterministic_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_counted_outer_archive_bytes(
    config: ProgramResidualProducerConfigV1,
) -> bytes:
    """Build exactly the two typed counted members, deterministically."""

    buffer = io.BytesIO()
    semantic_name, program_name = config.outer_archive_members
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        archive.writestr(
            _deterministic_zip_info(semantic_name),
            config.semantic_archive_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )
        archive.writestr(
            _deterministic_zip_info(program_name),
            config.program_packet_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )
    first = buffer.getvalue()
    second_buffer = io.BytesIO()
    with zipfile.ZipFile(
        second_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        archive.writestr(
            _deterministic_zip_info(semantic_name),
            config.semantic_archive_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )
        archive.writestr(
            _deterministic_zip_info(program_name),
            config.program_packet_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )
    if first != second_buffer.getvalue():
        raise ProgramResidualProducerError("outer archive construction is nondeterministic")
    return first


def build_production_adapter(
    config: ProgramResidualProducerConfigV1,
    *,
    learned_decoder: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
    learned_decoder_contract_id: str | None = None,
) -> TaskspaceSelectedPreimageFreshOperandAdapterV1:
    """Construct the real G49/G58 runtime; learned factors require exact code."""

    has_learned = any(
        factor.role is SelectedPreimageFactorRoleV1.LEARNED_IRREDUCIBLE_QUOTIENT for factor in config.program.factors
    )
    if has_learned and (learned_decoder is None or learned_decoder_contract_id is None):
        raise ProgramResidualProducerError(
            "LEARNED_QUOTIENT_RUNTIME_REGISTRY_OWED: counted learned factor has no exact callable"
        )
    if not has_learned and (learned_decoder is not None or learned_decoder_contract_id is not None):
        raise ProgramResidualProducerError("analytic-only program cannot bind an uncounted learned runtime")
    try:
        receiver = receive_carrier_compose_archive(
            config.semantic_archive_bytes,
            verify_member_effects=True,
        )
        operator = DisjointResizeOperator.build(
            camera_h=CAMERA_HEIGHT,
            camera_w=CAMERA_WIDTH,
            scorer_h=SCORER_HEIGHT,
            scorer_w=SCORER_WIDTH,
        )
        decoder = BoundV10Factor2SelectedPreimageDecoderV1(
            semantic_identity=config.program.semantic_program_identity,
            target_custody_identity=config.program.target_custody_identity,
            carrier_receiver=receiver,
            factor2_operator=operator,
            learned_quotient_decoder=learned_decoder,
            learned_quotient_decoder_contract_id_value=learned_decoder_contract_id,
        )
        auxiliary_provider = FreshScorerPlaneOperandLoaderV1.open(
            config.auxiliary_aggregate_receipt.path,
            expected_sha256=config.auxiliary_aggregate_receipt.sha256,
        )
        auxiliary_custody = AuxiliaryOperandCustodyV1(
            aggregate_receipt_path=config.auxiliary_aggregate_receipt.path,
            aggregate_receipt_sha256=config.auxiliary_aggregate_receipt.sha256,
        )
        learned_sources = tuple(
            LearnedDecoderSourceCustodyV1(
                section_id=row["section_id"],
                decoder_contract_id=row["decoder_contract_id"],
                implementation_source=ReopenableRegularFileIdentityV1(**row["implementation_source"]),
            )
            for row in config.learned_decoder_sources
        )
        production_custody = SelectedPreimageProductionCustodyV1(
            semantic_archive=ReopenableRegularFileIdentityV1(**config.semantic_archive.to_mapping()),
            semantic_compile_receipt=ReopenableRegularFileIdentityV1(**config.semantic_compile_receipt.to_mapping()),
            target_custody_receipt=ReopenableRegularFileIdentityV1(**config.target_custody_receipt.to_mapping()),
            target_custody_receipt_seal_sha256=(config.program.target_custody_identity.target_custody_receipt_sha256),
            compiler_source=ReopenableRegularFileIdentityV1(**config.compiler_source.to_mapping()),
            generic_v10_source=ReopenableRegularFileIdentityV1(**config.generic_v10_source.to_mapping()),
            decoder_callable_source_sha256=(config.program.decoder_identity.implementation_source_sha256),
            learned_decoder_sources=learned_sources,
        )
        return TaskspaceSelectedPreimageFreshOperandAdapterV1(
            program=config.program,
            decoder=decoder,
            auxiliary_provider=auxiliary_provider,
            auxiliary_custody=auxiliary_custody,
            production_custody=production_custody,
            pairs_per_stage=config.pairs_per_stage,
            test_only_small_fixture=config.test_only_small_fixture,
        )
    except Exception as exc:
        if isinstance(exc, ProgramResidualProducerError):
            raise
        raise ProgramResidualProducerError("production G49/G58 adapter construction failed") from exc


def _auxiliary_stage_by_index(
    adapter: TaskspaceSelectedPreimageFreshOperandAdapterV1,
    stage_index: int,
) -> object:
    for index, stage in enumerate(adapter.auxiliary_provider.iter_stages(max_pairs=adapter.pairs_per_stage)):
        if index == stage_index:
            return stage
    raise ProgramResidualProducerError(f"auxiliary provider lacks stage {stage_index}")


def _stage_field(stage: object, name: str) -> Any:
    if isinstance(stage, Mapping):
        if name not in stage:
            raise ProgramResidualProducerError(f"auxiliary stage lacks {name}")
        return stage[name]
    try:
        return getattr(stage, name)
    except AttributeError as exc:
        raise ProgramResidualProducerError(f"auxiliary stage lacks {name}") from exc


def materialize_stage_admission(
    adapter: TaskspaceSelectedPreimageFreshOperandAdapterV1,
    *,
    stage_index: int,
    prior_stage_chain_sha256: str,
) -> tuple[SelectedPreimagePreEncodeAdmissionV1, SelectedPreimageFreshOperandStageV1]:
    """Decode one G49 segment directly; prior segments are never decoded."""

    if type(adapter) is not TaskspaceSelectedPreimageFreshOperandAdapterV1:
        raise ProgramResidualProducerError("stage materializer requires exact G58 adapter")
    if type(stage_index) is not int or not 0 <= stage_index < adapter.stage_count:
        raise ProgramResidualProducerError("stage index escapes immutable lattice")
    _require_sha256(prior_stage_chain_sha256, "prior_stage_chain_sha256")
    start = stage_index * adapter.pairs_per_stage
    stop = start + adapter.pairs_per_stage
    auxiliary = _auxiliary_stage_by_index(adapter, stage_index)
    pair_ids = np.asarray(_stage_field(auxiliary, "pair_ids"))
    direct_y0 = np.asarray(_stage_field(auxiliary, "y0_u8"))
    direct_y1 = np.asarray(_stage_field(auxiliary, "y1_u8"))
    labels = np.asarray(_stage_field(auxiliary, "target_labels_u8"))
    poses = np.asarray(_stage_field(auxiliary, "gt_poses_f32"))
    if (
        _stage_field(auxiliary, "pair_range") != (start, stop)
        or pair_ids.shape != (adapter.pairs_per_stage,)
        or not np.array_equal(pair_ids, np.arange(start, stop))
        or direct_y0.shape != (adapter.pairs_per_stage, SCORER_HEIGHT, SCORER_WIDTH, CHANNELS)
        or direct_y1.shape != direct_y0.shape
        or labels.shape != (adapter.pairs_per_stage, SCORER_HEIGHT, SCORER_WIDTH)
        or poses.shape != (adapter.pairs_per_stage, 6)
        or direct_y0.dtype != np.uint8
        or direct_y1.dtype != np.uint8
        or labels.dtype != np.uint8
        or poses.dtype != np.float32
    ):
        raise ProgramResidualProducerError("auxiliary stage geometry or chronology differs")

    y0 = np.empty_like(direct_y0)
    y1 = np.empty_like(direct_y1)
    decoded_count = 0
    for local_offset, pair in enumerate(
        iter_selected_preimage_segment(
            adapter.program,
            adapter.decoder,
            segment_index=stage_index,
            pairs_per_segment=adapter.pairs_per_stage,
        )
    ):
        if local_offset >= adapter.pairs_per_stage:
            raise ProgramResidualProducerError("G49 segment yielded extra pairs")
        expected_pair = start + local_offset
        if (
            pair.pair_index != expected_pair
            or pair.source_pair_id != expected_pair
            or pair.segment_index != stage_index
            or pair.segment_count != adapter.stage_count
        ):
            raise ProgramResidualProducerError("G49 decoded pair identity differs")
        y0[local_offset] = pair.scorer_y0
        y1[local_offset] = pair.scorer_y1
        decoded_count += 1
    if decoded_count != adapter.pairs_per_stage:
        raise ProgramResidualProducerError("G49 segment did not cover exact stage")
    if any(np.shares_memory(decoded, direct) for decoded in (y0, y1) for direct in (direct_y0, direct_y1)):
        raise ProgramResidualProducerError("decoded G49 planes alias direct source planes")

    program = adapter.program
    semantic = program.semantic_program_identity
    target = program.target_custody_identity
    factors = tuple(factor.section_id for factor in program.factors)
    behavior = tuple(
        factor.section_id
        for factor in program.factors
        if any(factor.addresses(pair_id) for pair_id in range(start, stop))
    )
    homes = program.byte_homes()
    values: dict[str, Any] = {
        "schema": ADMISSION_SCHEMA,
        "status": "PRE_ENCODE_SELECTED_PREIMAGE_STAGE_ADMITTED",
        "stage_index": stage_index,
        "stage_count": adapter.stage_count,
        "pair_range": (start, stop),
        "representation_source": "G49_TASKSPACE_SELECTED_PREIMAGE_PROGRAM_DECODE_ONLY",
        "program_packet_sha256": program.packet_sha256,
        "program_packet_bytes": len(program.packet_bytes),
        "semantic_archive_sha256": semantic.compiled_semantic_archive_sha256,
        "semantic_archive_bytes": semantic.compiled_semantic_archive_bytes,
        "fresh_semantic_compile_receipt_sha256": semantic.fresh_compile_receipt_sha256,
        "target_custody_receipt_sha256": target.target_custody_receipt_sha256,
        "target_bank_sha256": target.target_bank_sha256,
        "decoder_id": adapter.decoder.decoder_id,
        "decoder_implementation_source_sha256": (adapter.decoder.implementation_source_sha256),
        "factor_section_ids": factors,
        "behavior_changing_factor_section_ids": behavior,
        "factor_payload_bytes_inside_packet": sum(home.byte_length for home in homes[1:]),
        "scorer_y0_sha256": _array_sha256(y0),
        "scorer_y1_sha256": _array_sha256(y1),
        "discarded_direct_source_y0_sha256": _array_sha256(direct_y0),
        "discarded_direct_source_y1_sha256": _array_sha256(direct_y1),
        "target_labels_sha256": _array_sha256(labels),
        "gt_poses_f32_sha256": _array_sha256(poses),
        "auxiliary_planes_forwarded": False,
        "g49_decode_custody_verified": True,
        "prior_stage_chain_sha256": prior_stage_chain_sha256,
        "representation_status": BLOCKED_REPRESENTATION_STATUS,
        "next_preclosure_gate": NEXT_PRECLOSURE_GATE,
    }
    stage_chain = sha256_bytes(bytes.fromhex(prior_stage_chain_sha256) + canonical_json_body(values))
    admission = SelectedPreimagePreEncodeAdmissionV1(
        **values,
        stage_chain_sha256=stage_chain,
    )
    stage = SelectedPreimageFreshOperandStageV1(
        pair_range=(start, stop),
        pair_ids=np.ascontiguousarray(pair_ids),
        y0_u8=np.ascontiguousarray(y0),
        y1_u8=np.ascontiguousarray(y1),
        target_labels_u8=np.ascontiguousarray(labels),
        gt_poses_f32=np.ascontiguousarray(poses),
        pre_encode_admission=admission,
    )
    if (
        validate_pre_encode_stage(
            stage,
            expected_prior_stage_chain_sha256=prior_stage_chain_sha256,
        )
        != stage_chain
    ):
        raise ProgramResidualProducerError("G58 stage validation returned a different chain")
    return admission, stage


def _stage_receipt_body(
    *,
    config: ProgramResidualProducerConfigV1,
    stage_index: int,
    admission: SelectedPreimagePreEncodeAdmissionV1,
) -> dict[str, Any]:
    return {
        "schema": STAGE_RECEIPT_SCHEMA,
        "status": "IMMUTABLE_STAGE_DECODE_VERIFIED",
        "run_id": config.run_id,
        "campaign_id": config.campaign_id,
        "config": config.file_identity.to_mapping(),
        "program_packet": config.program_packet.to_mapping(),
        "stage_index": stage_index,
        "stage_count": config.stage_count,
        "pair_range": list(admission.pair_range),
        "admission": asdict(admission),
        "decoded_planes_persisted": False,
        "direct_source_planes_persisted": False,
        "raw_labels_persisted": False,
        "resume_source": "IMMUTABLE_G49_PACKET_AND_STAGE_RECEIPT",
        "no_prior_stage_redecode_required": True,
    }


def _stage_receipt_path(
    config: ProgramResidualProducerConfigV1,
    stage_index: int,
) -> Path:
    return config.output_paths["stage_checkpoint_dir"] / f"stage_{stage_index:02d}.json"


def _load_stage_checkpoint(
    config: ProgramResidualProducerConfigV1,
    *,
    stage_index: int,
    expected_prior_chain: str,
) -> SelectedPreimagePreEncodeAdmissionV1 | None:
    path = _stage_receipt_path(config, stage_index)
    if not path.exists() and not path.is_symlink():
        return None
    _, payload = stable_read_regular_file(path, label=f"stage {stage_index} checkpoint")
    try:
        receipt = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProgramResidualProducerError("stage checkpoint is not JSON") from exc
    if type(receipt) is not dict:
        raise ProgramResidualProducerError("stage checkpoint must be an object")
    seal = receipt.get("receipt_sha256")
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    if seal != sha256_bytes(canonical_json(body)) or payload != canonical_json(receipt):
        raise ProgramResidualProducerError("stage checkpoint seal or canonical bytes differ")
    if (
        body.get("schema") != STAGE_RECEIPT_SCHEMA
        or body.get("status") != "IMMUTABLE_STAGE_DECODE_VERIFIED"
        or body.get("run_id") != config.run_id
        or body.get("campaign_id") != config.campaign_id
        or body.get("config") != config.file_identity.to_mapping()
        or body.get("program_packet") != config.program_packet.to_mapping()
        or body.get("stage_index") != stage_index
        or body.get("stage_count") != config.stage_count
        or body.get("decoded_planes_persisted") is not False
        or body.get("direct_source_planes_persisted") is not False
        or body.get("raw_labels_persisted") is not False
        or body.get("no_prior_stage_redecode_required") is not True
    ):
        raise ProgramResidualProducerError("stage checkpoint identity/truth fields differ")
    try:
        admission = SelectedPreimagePreEncodeAdmissionV1(**body["admission"])
    except (KeyError, TypeError) as exc:
        raise ProgramResidualProducerError("stage checkpoint admission fields differ") from exc
    if (
        admission.prior_stage_chain_sha256 != expected_prior_chain
        or admission.stage_index != stage_index
        or admission.stage_count != config.stage_count
        or admission.program_packet_sha256 != config.program.packet_sha256
        or tuple(admission.factor_section_ids) != tuple(factor.section_id for factor in config.program.factors)
    ):
        raise ProgramResidualProducerError("resumed stage chain/program identity differs")
    chain_body = asdict(admission)
    chain_body.pop("stage_chain_sha256")
    expected_chain = sha256_bytes(bytes.fromhex(expected_prior_chain) + canonical_json_body(chain_body))
    if admission.stage_chain_sha256 != expected_chain:
        raise ProgramResidualProducerError("resumed stage chain hash differs")
    return admission


def run_stage_lattice(
    config: ProgramResidualProducerConfigV1,
    adapter: TaskspaceSelectedPreimageFreshOperandAdapterV1,
    *,
    stop_after_stage: int | None = None,
) -> tuple[SelectedPreimagePreEncodeAdmissionV1, ...]:
    """Resume from immutable receipts without decoding any completed segment."""

    prior_chain = "0" * 64
    admissions: list[SelectedPreimagePreEncodeAdmissionV1] = []
    observed_behavior: set[str] = set()
    for stage_index in range(config.stage_count):
        admission = _load_stage_checkpoint(
            config,
            stage_index=stage_index,
            expected_prior_chain=prior_chain,
        )
        if admission is None:
            admission, _stage = materialize_stage_admission(
                adapter,
                stage_index=stage_index,
                prior_stage_chain_sha256=prior_chain,
            )
            body = _stage_receipt_body(
                config=config,
                stage_index=stage_index,
                admission=admission,
            )
            receipt = {
                **body,
                "receipt_sha256": sha256_bytes(canonical_json(body)),
            }
            publish_write_once(
                _stage_receipt_path(config, stage_index),
                canonical_json(receipt),
                label=f"stage {stage_index} checkpoint",
            )
        admissions.append(admission)
        observed_behavior.update(admission.behavior_changing_factor_section_ids)
        prior_chain = admission.stage_chain_sha256
        if stop_after_stage is not None and stage_index >= stop_after_stage:
            break
    if len(admissions) == config.stage_count and observed_behavior != {
        factor.section_id for factor in config.program.factors
    }:
        raise ProgramResidualProducerError("stage lattice did not exercise every counted factor")
    return tuple(admissions)


def _publish_terminal_receipt(
    config: ProgramResidualProducerConfigV1,
    adapter: TaskspaceSelectedPreimageFreshOperandAdapterV1,
    *,
    identity_receipt: StableFileIdentityV1,
    admissions: Sequence[SelectedPreimagePreEncodeAdmissionV1],
) -> StableFileIdentityV1:
    if len(admissions) != config.stage_count:
        raise ProgramResidualProducerError("terminal receipt requires complete stage lattice")
    body = {
        "schema": TERMINAL_STAGE_CHAIN_SCHEMA,
        "status": "FULL_STAGE_LATTICE_DECODED_AND_CHAIN_CLOSED",
        "campaign_receipt": config.campaign_seal_receipt.to_mapping(),
        "pre_encode_identity_receipt": identity_receipt.to_mapping(),
        "auxiliary_aggregate_receipt": {
            **config.auxiliary_aggregate_receipt.to_mapping(),
            "self_seal_sha256": (adapter.auxiliary_custody.aggregate_receipt_self_seal_sha256),
        },
        "pair_count": config.pair_count,
        "pairs_per_stage": config.pairs_per_stage,
        "stage_count": config.stage_count,
        "scorer_batch_size": SCORER_BATCH_SIZE,
        "program_packet_sha256": config.program.packet_sha256,
        "stages": [asdict(admission) for admission in admissions],
        "terminal_stage_chain_sha256": admissions[-1].stage_chain_sha256,
        "representation_status": BLOCKED_REPRESENTATION_STATUS,
    }
    receipt = {
        **body,
        "receipt_sha256": sha256_bytes(canonical_json_body(body)),
    }
    return publish_write_once(
        config.output_paths["g58_terminal_stage_chain_receipt"],
        canonical_json_body(receipt),
        label="G58 terminal stage-chain receipt",
    )


def run_structural_producer(
    config: ProgramResidualProducerConfigV1,
    *,
    adapter: TaskspaceSelectedPreimageFreshOperandAdapterV1 | None = None,
) -> Mapping[str, Any]:
    """Close G58 residual transport and obtain the required G59 refusal."""

    preflight = _ensure_output_root(config)
    runtime = build_production_adapter(config) if adapter is None else adapter
    if type(runtime) is not TaskspaceSelectedPreimageFreshOperandAdapterV1:
        raise ProgramResidualProducerError("runner requires exact G58 adapter runtime")
    identity_row = runtime.publish_pre_encode_identity_receipt(config.output_paths["g58_identity_receipt"])
    identity = StableFileIdentityV1(
        path=identity_row["path"],
        bytes=identity_row["bytes"],
        sha256=identity_row["sha256"],
    )
    admissions = run_stage_lattice(config, runtime)
    terminal = _publish_terminal_receipt(
        config,
        runtime,
        identity_receipt=identity,
        admissions=admissions,
    )

    archive_payload = build_counted_outer_archive_bytes(config)
    archive = publish_write_once(
        config.output_paths["outer_archive"],
        archive_payload,
        label="counted two-member outer archive",
    )
    proof = reopen_program_residual_outer_archive(
        archive_path=archive.path,
        semantic_archive_bytes=config.semantic_archive_bytes,
        semantic_member_name=config.outer_archive_members[0],
        program_member_name=config.outer_archive_members[1],
        program=config.program,
        decoder=runtime.decoder,
    )
    if type(proof) is not ProgramResidualOuterArchiveProofV1:
        raise ProgramResidualProducerError("G58 outer reopen did not return exact proof")
    proof_row = publish_program_residual_outer_archive_proof(
        proof,
        config.output_paths["g58_outer_proof_receipt"],
    )
    proof_identity = StableFileIdentityV1(
        path=proof_row["path"],
        bytes=proof_row["bytes"],
        sha256=proof_row["sha256"],
    )
    production = reopen_program_residual_production_pre_encode_evidence(
        identity_path=identity.path,
        terminal_stage_chain_path=terminal.path,
        outer_proof_path=proof_identity.path,
    )
    if production.get("schema") != PRODUCTION_PRE_ENCODE_EVIDENCE_SCHEMA or production.get("status") != "ADMIT":
        raise ProgramResidualProducerError("G58 production evidence did not close")

    # Local import avoids a module cycle: G59 imports this module's verifier.
    from tac.witness_control.taskspace_codec_adversarial_gate_v2 import (
        G17_PRODUCTION_TERMINAL_ENVELOPE_RECEIVER_OWED,
        admit_pre_encode,
    )

    g59 = admit_pre_encode(
        campaign_seal_path=config.campaign_seal_receipt.path,
        output_path=config.output_paths["g59_pre_encode_receipt"],
        producer_config_path=config.path,
        g58_identity_receipt_path=identity.path,
        g58_terminal_stage_chain_path=terminal.path,
        g58_outer_proof_path=proof_identity.path,
        asserted_representation=PROGRAM_RESIDUAL_LAYERED,
    )
    if g59.get("status") != "REFUSE" or g59.get("refusals") != [
        G17_PRODUCTION_TERMINAL_ENVELOPE_RECEIVER_OWED
    ]:
        raise ProgramResidualProducerError("G49-only producer must end at the canonical G17 terminal-link blocker")
    body = {
        "schema": RUN_RECEIPT_SCHEMA,
        "status": "G58_RESIDUAL_TRANSPORT_CLOSED_G59_G17_TERMINAL_LINK_REFUSED",
        "run_id": config.run_id,
        "campaign_id": config.campaign_id,
        "config": config.file_identity.to_mapping(),
        "program_packet": config.program_packet.to_mapping(),
        "semantic_archive": config.semantic_archive.to_mapping(),
        "g58_identity_receipt": identity.to_mapping(),
        "g58_terminal_stage_chain_receipt": terminal.to_mapping(),
        "outer_archive": archive.to_mapping(),
        "g58_outer_proof_receipt": proof_identity.to_mapping(),
        "g59_pre_encode_receipt": stable_file_identity(
            config.output_paths["g59_pre_encode_receipt"],
            label="G59 PRE_ENCODE receipt",
        ).to_mapping(),
        "stage_count": len(admissions),
        "storage_preflight": preflight,
        "candidate_admission": False,
        "score_claim": False,
        "evaluation_claim": False,
        "promotion_eligible": False,
        "blocking_conditions": [G17_PRODUCTION_TERMINAL_ENVELOPE_RECEIVER_OWED],
    }
    receipt = {
        **body,
        "receipt_sha256": sha256_bytes(canonical_json(body)),
    }
    publish_write_once(
        config.output_paths["run_receipt"],
        canonical_json(receipt),
        label="program residual run receipt",
    )
    return receipt


def _strict_evidence(value: Mapping[str, Any]) -> Mapping[str, Any]:
    production = value.get("strict_production_evidence")
    if not isinstance(production, Mapping):
        raise ProgramResidualProducerError("G58 strict production evidence is absent")
    if (
        production.get("schema") != PRODUCTION_PRE_ENCODE_EVIDENCE_SCHEMA
        or production.get("status") != "ADMIT"
        or production.get("representation_mode") != PROGRAM_RESIDUAL_MODE
        or production.get("pair_count") != PRODUCTION_PAIR_COUNT
        or production.get("pairs_per_stage") != PRODUCTION_PAIRS_PER_STAGE
        or production.get("stage_count") != PRODUCTION_STAGE_COUNT
        or production.get("scorer_batch_size") != SCORER_BATCH_SIZE
        or production.get("target_payload_embedded") is not False
        or production.get("historical_payload_reused") is not False
    ):
        raise ProgramResidualProducerError("G58 strict production evidence fields differ")
    return production


def validate_program_residual_producer_config_for_g59(
    config_identity: Mapping[str, Any] | None,
    g58_evidence: Mapping[str, Any],
) -> tuple[str, ...]:
    """Reopen the exact config and return only honest candidate blockers.

    G59 calls this only after its own strict G58 verifier has reopened the
    identity, five-stage terminal chain, and physical two-member outer proof.
    """

    if not isinstance(config_identity, Mapping):
        raise ProgramResidualProducerError("G59 producer config identity is absent")
    expected_config = StableFileIdentityV1(
        path=str(config_identity.get("path", "")),
        bytes=_require_int(config_identity.get("bytes"), "producer config bytes", minimum=1),
        sha256=_require_sha256(
            config_identity.get("sha256"),
            "producer config sha256",
        ),
    )
    observed_config = stable_file_identity(
        expected_config.path,
        label="G59 producer config",
        maximum_bytes=_MAX_CONFIG_BYTES,
    )
    if observed_config != expected_config:
        raise ProgramResidualProducerError("G59 producer config identity drifted")
    config = load_config(expected_config.path)
    if config.test_only_small_fixture:
        raise ProgramResidualProducerError("test-only producer config cannot satisfy live G59")
    production = _strict_evidence(g58_evidence)
    comparisons = (
        ("program_packet_sha256", config.program.packet_sha256),
        ("pair_count", config.pair_count),
        ("pairs_per_stage", config.pairs_per_stage),
        ("stage_count", config.stage_count),
        ("scorer_batch_size", SCORER_BATCH_SIZE),
    )
    if any(production.get(key) != expected for key, expected in comparisons):
        raise ProgramResidualProducerError("G58 evidence differs from producer program/lattice")
    artifact_comparisons = (
        ("semantic_archive", config.semantic_archive),
        ("semantic_compile_receipt", config.semantic_compile_receipt),
        ("target_custody_receipt", config.target_custody_receipt),
        ("generic_v10_source", config.generic_v10_source),
        ("campaign_receipt", config.campaign_seal_receipt),
    )
    for key, expected in artifact_comparisons:
        if production.get(key) != expected.to_mapping():
            raise ProgramResidualProducerError(f"G58 evidence {key} differs from producer config")
    if production.get("factor_section_ids") != [factor.section_id for factor in config.program.factors]:
        raise ProgramResidualProducerError("G58 factor sections differ from producer config")
    learned_ids = sorted(
        factor.section_id
        for factor in config.program.factors
        if factor.role is SelectedPreimageFactorRoleV1.LEARNED_IRREDUCIBLE_QUOTIENT
    )
    if production.get("learned_factor_section_ids") != learned_ids:
        raise ProgramResidualProducerError("G58 learned factor roles differ from producer config")
    path_comparisons = (
        ("identity_receipt", "g58_identity_receipt"),
        ("terminal_stage_chain_receipt", "g58_terminal_stage_chain_receipt"),
        ("outer_proof_receipt", "g58_outer_proof_receipt"),
        ("outer_archive", "outer_archive"),
    )
    for evidence_key, output_key in path_comparisons:
        row = production.get(evidence_key)
        if (
            not isinstance(row, Mapping)
            or _lexical_absolute(str(row.get("path", ""))) != config.output_paths[output_key]
        ):
            raise ProgramResidualProducerError(f"G58 {evidence_key} path differs from bound producer output")
    # G63 owns only residual-transport schema and custody. G17 owns all
    # selected-solution placement, terminal-envelope, receiver, and authority
    # admission state.
    return ()


__all__ = [
    "CLOSED_G49_FACTOR_VOCABULARY",
    "CONFIG_SCHEMA",
    "EXAMPLE_CONFIG_SCHEMA",
    "PRODUCER_ROLE",
    "ProgramResidualProducerConfigV1",
    "ProgramResidualProducerError",
    "StableFileIdentityV1",
    "build_counted_outer_archive_bytes",
    "build_production_adapter",
    "canonical_json",
    "factor_operand_inventory",
    "load_config",
    "materialize_stage_admission",
    "publish_write_once",
    "run_stage_lattice",
    "run_structural_producer",
    "sha256_bytes",
    "stable_file_identity",
    "stable_read_regular_file",
    "validate_program_residual_producer_config_for_g59",
]
