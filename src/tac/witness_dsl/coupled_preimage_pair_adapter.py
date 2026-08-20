# SPDX-License-Identifier: MIT
"""Counted Pair adapter for the reverse-causal coupled-preimage program.

The counted ``A`` payload is exactly one canonical
``coupled_preimage_program`` packet.  This module does not wrap that packet in
an unpriced sidecar.  It places the exact packet bytes in the Pair manifest as
``FRAME0_FROM_EXACT_Y1`` because ``A`` materializes ``Y0 | exact Y1``.  Mapping
it to ``FRAME1_PREIMAGE`` or ``FRAME0_POSE_RESIDUAL`` would be a causal
misstatement: an A-only counterfactual must change Y0 while preserving Y1 and
does not consume or serialize Pose6.

Consumption also slices and strictly reapplies the exact counted ``G`` packet;
the caller's decoded G is only an expected value and never an execution input.

This remains a Python-reference, research-only bridge.  Direct source bytes are
reopened and bound, but defaults, globals, transitive imports, a standalone
archive runtime, candidate lineage, score authority, and originality are not
claimed.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

import tac.witness_dsl.coupled_preimage_program as coupled_preimage_module
from tac.witness_dsl.coupled_preimage_program import (
    CAUSAL_MATERIALIZATION_ORDER,
    DECODER_BINDING_SCOPE,
    DECODER_CONTRACT_ID,
    DECODER_DIRECT_SOURCE_SHA256,
    CoupledPreimageMode,
    CoupledPreimageProgramError,
    CoupledPreimageProgramV1,
    DecodedCoupledPreimageV1,
    decode_coupled_preimage_program,
    parse_coupled_preimage_program,
)
from tac.witness_dsl.coupled_witness_state import canonical_json_bytes, canonical_sha256
from tac.witness_dsl.generative_taskspace_correction import (
    DecodedGenerativeCorrectionV1,
    GenerativeTaskspaceCorrectionError,
    PredictorSemanticStateV1,
    apply_generative_taskspace_correction,
)
from tac.witness_dsl.pair_population_envelope import (
    CountedArtifactClass,
    CountedPayloadLineage,
    CountedPayloadProvenance,
    CountedProgramSection,
    CountedSectionRole,
    PairDomain,
    PairPopulation,
    PairPopulationEnvelopeError,
    RoleCounterfactualProgram,
    SectionCausalityReceipt,
    validate_reverse_causal_counted_program_sections,
)

ADAPTER_SCHEMA: Final = "tac.coupled_preimage_pair_adapter.v1"
ADAPTER_RECEIPT_ENVELOPE_SCHEMA: Final = "tac.coupled_preimage_pair_adapter.envelope.v1"
SOURCE_CUSTODY_SCHEMA: Final = "tac.coupled_preimage_source_custody.v1"
RUNTIME_BINDING_SCHEMA: Final = "tac.coupled_preimage_direct_runtime_binding.v1"
COUNTERFACTUAL_SCHEMA: Final = "tac.coupled_preimage_pair_counterfactual.v1"
A_SECTION_ROLE: Final = CountedSectionRole.FRAME0_FROM_EXACT_Y1
A_ARTIFACT_CLASS: Final = CountedArtifactClass.COUPLED_PREIMAGE_PARAMETERS
CAUSAL_ROLE: Final = "Y0_GIVEN_EXACT_REALIZED_UINT8_Y1"
PAIR_ORDER_POLICY: Final = "exact_A_source_ids_equal_PairPopulation_V10_local_order.v1"
RUNTIME_BLOCKER: Final = (
    "direct Python module bytes are bound, but defaults, globals, transitive imports, native dependencies, "
    "and a standalone archive runtime are not closed"
)
LINEAGE_BLOCKER: Final = (
    "producer-declared lineage does not prove the complete typed derivation of every A control byte"
)
PAIR_COMPACT_GENERALIZATION_BLOCKER: Final = (
    "the legacy CompactGeneratorDecode/PairPopulationEnvelope seal intentionally remains V9 Pose6-owned and "
    "rejects FRAME0_FROM_EXACT_Y1; the pose-independent reverse-causal path still needs its own complete envelope "
    "seal and ownership proof"
)


class CoupledPreimagePairAdapterError(ValueError):
    """Malformed custody, Pair placement, or reverse-causal A behavior."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CoupledPreimagePairAdapterError(f"{label} must be lowercase SHA-256 hex")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CoupledPreimagePairAdapterError(f"{label} must be nonempty trimmed text")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise CoupledPreimagePairAdapterError(f"{label} must be a nonnegative exact integer")
    return value


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _stable_regular_file(path: Path, *, label: str) -> tuple[Path, bytes]:
    candidate = Path(path)
    try:
        leaf_before = candidate.lstat()
    except OSError as exc:
        raise CoupledPreimagePairAdapterError(f"{label} cannot be reopened") from exc
    if stat.S_ISLNK(leaf_before.st_mode) or not stat.S_ISREG(leaf_before.st_mode):
        raise CoupledPreimagePairAdapterError(f"{label} must be a regular non-symlink file")

    # O_NOFOLLOW closes the leaf-follow race where the host provides it.  The
    # exact pre-open/fd/post-open identity checks remain the fail-closed
    # portable fallback; they do not claim that parent path components are
    # symlink-free or immutable.
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(candidate, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise CoupledPreimagePairAdapterError(f"{label} must be a regular non-symlink file") from exc
        raise CoupledPreimagePairAdapterError(f"{label} cannot be reopened") from exc

    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise CoupledPreimagePairAdapterError(f"{label} must be a regular non-symlink file")
        before_identity = _file_identity(before)
        if _file_identity(leaf_before) != before_identity:
            raise CoupledPreimagePairAdapterError(f"{label} mutated during reopen")

        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)

        after = os.fstat(fd)
        if not stat.S_ISREG(after.st_mode):
            raise CoupledPreimagePairAdapterError(f"{label} must be a regular non-symlink file")
        after_identity = _file_identity(after)
        if before_identity != after_identity or len(payload) != after.st_size:
            raise CoupledPreimagePairAdapterError(f"{label} mutated during reopen")

        try:
            leaf_after = candidate.lstat()
        except OSError as exc:
            raise CoupledPreimagePairAdapterError(f"{label} mutated during reopen") from exc
        leaf_after_identity = _file_identity(leaf_after)
        if stat.S_ISLNK(leaf_after.st_mode) or not stat.S_ISREG(leaf_after.st_mode):
            raise CoupledPreimagePairAdapterError(f"{label} mutated during reopen")
        if leaf_after_identity != after_identity:
            raise CoupledPreimagePairAdapterError(f"{label} mutated during reopen")
    except OSError as exc:
        raise CoupledPreimagePairAdapterError(f"{label} cannot be reopened") from exc
    finally:
        os.close(fd)

    # Keep the custody path lexical: resolving here would reopen and follow a
    # leaf that may have been replaced after the descriptor-bound snapshot.
    absolute_path = Path(os.path.abspath(os.fspath(candidate)))
    if not absolute_path.is_absolute():
        raise CoupledPreimagePairAdapterError(f"{label} mutated during reopen")
    return absolute_path, payload


@dataclass(frozen=True, slots=True)
class ReopenedFileCustodyV1:
    """Exact stable-file identity derived by reopening bytes, never attested."""

    role: str
    path: str
    byte_length: int
    sha256: str

    def __post_init__(self) -> None:
        _text(self.role, "file custody role")
        _text(self.path, "file custody path")
        if not Path(self.path).is_absolute():
            raise CoupledPreimagePairAdapterError("file custody path must be absolute")
        if type(self.byte_length) is not int or self.byte_length <= 0:
            raise CoupledPreimagePairAdapterError("file custody byte_length must be positive")
        _require_sha256(self.sha256, "file custody SHA-256")

    @classmethod
    def reopen(cls, path: Path, *, role: str) -> tuple[ReopenedFileCustodyV1, bytes]:
        resolved, payload = _stable_regular_file(path, label=role)
        if not payload:
            raise CoupledPreimagePairAdapterError(f"{role} must not be empty")
        return cls(role=role, path=str(resolved), byte_length=len(payload), sha256=_sha256(payload)), payload

    def reopen_bytes(self) -> bytes:
        resolved, payload = _stable_regular_file(Path(self.path), label=self.role)
        if str(resolved) != self.path or len(payload) != self.byte_length or _sha256(payload) != self.sha256:
            raise CoupledPreimagePairAdapterError(f"{self.role} differs from its reopened custody identity")
        return payload

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": self.path,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class CoupledPreimageSourceCustodyV1:
    """Source-side custody for one exact A artifact and its known producer.

    ``producer_argv`` and ``producer_config`` are optional because historical
    artifacts do not always carry them.  Their absence is explicit and cannot
    be promoted into complete derivation lineage.
    """

    artifact: ReopenedFileCustodyV1
    producer_source: ReopenedFileCustodyV1
    producer_id: str
    payload_lineage: CountedPayloadLineage
    video_derived: bool
    producer_argv: tuple[str, ...] | None = None
    producer_config: ReopenedFileCustodyV1 | None = None
    complete_typed_derivation_lineage: Literal[False] = False
    originality_claimed: Literal[False] = False

    def __post_init__(self) -> None:
        if self.artifact.role != "counted_coupled_preimage_A_packet":
            raise CoupledPreimagePairAdapterError("A artifact custody role differs")
        if self.producer_source.role != "coupled_preimage_A_producer_source":
            raise CoupledPreimagePairAdapterError("A producer-source custody role differs")
        _text(self.producer_id, "A producer_id")
        if not isinstance(self.payload_lineage, CountedPayloadLineage):
            raise CoupledPreimagePairAdapterError("A payload lineage must use CountedPayloadLineage")
        if type(self.video_derived) is not bool:
            raise CoupledPreimagePairAdapterError("A video_derived must be an exact boolean")
        if self.producer_argv is not None:
            if type(self.producer_argv) is not tuple or not self.producer_argv:
                raise CoupledPreimagePairAdapterError("producer_argv must be a nonempty exact tuple when present")
            for index, argument in enumerate(self.producer_argv):
                _text(argument, f"producer_argv[{index}]")
        if self.producer_config is not None and self.producer_config.role != "coupled_preimage_A_producer_config":
            raise CoupledPreimagePairAdapterError("A producer-config custody role differs")
        if self.complete_typed_derivation_lineage is not False or self.originality_claimed is not False:
            raise CoupledPreimagePairAdapterError("A source custody cannot claim complete lineage or originality")

    @classmethod
    def reopen(
        cls,
        artifact_path: Path,
        *,
        producer_source_path: Path,
        producer_id: str,
        payload_lineage: CountedPayloadLineage,
        video_derived: bool,
        producer_argv: tuple[str, ...] | None = None,
        producer_config_path: Path | None = None,
    ) -> CoupledPreimageSourceCustodyV1:
        artifact, packet = ReopenedFileCustodyV1.reopen(
            artifact_path,
            role="counted_coupled_preimage_A_packet",
        )
        try:
            parsed = parse_coupled_preimage_program(packet)
        except CoupledPreimageProgramError as exc:
            raise CoupledPreimagePairAdapterError("source A artifact failed strict typed parse-back") from exc
        if parsed.to_packet() != packet:
            raise CoupledPreimagePairAdapterError("source A artifact is not byte-canonical on typed re-emission")
        producer_source, _ = ReopenedFileCustodyV1.reopen(
            producer_source_path,
            role="coupled_preimage_A_producer_source",
        )
        producer_config = None
        if producer_config_path is not None:
            producer_config, _ = ReopenedFileCustodyV1.reopen(
                producer_config_path,
                role="coupled_preimage_A_producer_config",
            )
        return cls(
            artifact=artifact,
            producer_source=producer_source,
            producer_id=producer_id,
            payload_lineage=payload_lineage,
            video_derived=video_derived,
            producer_argv=producer_argv,
            producer_config=producer_config,
        )

    def reopen_packet(self) -> bytes:
        packet = self.artifact.reopen_bytes()
        try:
            parsed = parse_coupled_preimage_program(packet)
        except CoupledPreimageProgramError as exc:
            raise CoupledPreimagePairAdapterError("reopened A artifact failed strict typed parse-back") from exc
        if parsed.to_packet() != packet:
            raise CoupledPreimagePairAdapterError("reopened A artifact changed canonical bytes")
        self.producer_source.reopen_bytes()
        if self.producer_config is not None:
            self.producer_config.reopen_bytes()
        return packet

    def counted_provenance(self, program: CoupledPreimageProgramV1) -> CountedPayloadProvenance:
        source = program.source_binding
        derivation_input_sha256 = canonical_sha256(
            {
                "schema": "tac.coupled_preimage_A_derivation_inputs.v1",
                "predictor_semantic_binding_sha256": source.predictor_semantic_binding_sha256,
                "correction_binding_sha256": source.correction_binding_sha256,
                "source_pair_ids": list(source.source_pair_ids),
                "producer_id": self.producer_id,
                "producer_source_sha256": self.producer_source.sha256,
                "producer_argv": list(self.producer_argv) if self.producer_argv is not None else None,
                "producer_config_sha256": (self.producer_config.sha256 if self.producer_config is not None else None),
            }
        )
        return CountedPayloadProvenance(
            artifact_class=A_ARTIFACT_CLASS,
            lineage=self.payload_lineage,
            producer_id=self.producer_id,
            producer_source_sha256=self.producer_source.sha256,
            derivation_input_sha256=derivation_input_sha256,
            video_derived=self.video_derived,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_CUSTODY_SCHEMA,
            "artifact": self.artifact.as_dict(),
            "producer_source": self.producer_source.as_dict(),
            "producer_id": self.producer_id,
            "producer_argv": list(self.producer_argv) if self.producer_argv is not None else None,
            "producer_argv_available": self.producer_argv is not None,
            "producer_config": self.producer_config.as_dict() if self.producer_config is not None else None,
            "producer_config_available": self.producer_config is not None,
            "payload_lineage": self.payload_lineage.value,
            "video_derived": self.video_derived,
            "complete_typed_derivation_lineage": self.complete_typed_derivation_lineage,
            "originality_claimed": self.originality_claimed,
            "lineage_blocker": LINEAGE_BLOCKER,
        }


@dataclass(frozen=True, slots=True)
class CoupledPreimageDirectRuntimeBindingV1:
    """Direct decoder module identity, explicitly not transitive closure."""

    module_path: str
    module_bytes: int
    module_sha256: str
    decoder_contract_id: str = DECODER_CONTRACT_ID
    binding_scope: str = DECODER_BINDING_SCOPE
    direct_module_source_closed: Literal[True] = True
    defaults_globals_transitive_imports_bound: Literal[False] = False
    standalone_archive_runtime_closed: Literal[False] = False

    def __post_init__(self) -> None:
        _text(self.module_path, "A decoder module path")
        if not Path(self.module_path).is_absolute():
            raise CoupledPreimagePairAdapterError("A decoder module path must be absolute")
        if type(self.module_bytes) is not int or self.module_bytes <= 0:
            raise CoupledPreimagePairAdapterError("A decoder module byte count must be positive")
        _require_sha256(self.module_sha256, "A decoder module SHA-256")
        if self.decoder_contract_id != DECODER_CONTRACT_ID or self.binding_scope != DECODER_BINDING_SCOPE:
            raise CoupledPreimagePairAdapterError("A decoder contract or binding scope differs")
        if (
            self.direct_module_source_closed is not True
            or self.defaults_globals_transitive_imports_bound is not False
            or self.standalone_archive_runtime_closed is not False
        ):
            raise CoupledPreimagePairAdapterError("A runtime binding overclaims executable closure")

    @classmethod
    def reopen(cls) -> CoupledPreimageDirectRuntimeBindingV1:
        module_name = coupled_preimage_module.__file__
        if not isinstance(module_name, str) or not module_name:
            raise CoupledPreimagePairAdapterError("A decoder module has no inspectable source path")
        path, payload = _stable_regular_file(Path(module_name), label="coupled preimage decoder module")
        digest = _sha256(payload)
        if digest != DECODER_DIRECT_SOURCE_SHA256:
            raise CoupledPreimagePairAdapterError("live A decoder module differs from its imported source binding")
        return cls(module_path=str(path), module_bytes=len(payload), module_sha256=digest)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_BINDING_SCHEMA,
            "module_path": self.module_path,
            "module_bytes": self.module_bytes,
            "module_sha256": self.module_sha256,
            "decoder_contract_id": self.decoder_contract_id,
            "binding_scope": self.binding_scope,
            "direct_module_source_closed": self.direct_module_source_closed,
            "defaults_globals_transitive_imports_bound": self.defaults_globals_transitive_imports_bound,
            "standalone_archive_runtime_closed": self.standalone_archive_runtime_closed,
            "runtime_blocker": RUNTIME_BLOCKER,
        }


@dataclass(frozen=True, slots=True)
class CountedCoupledPreimageDecodeV1:
    """One strict Pair-sliced A decode over exact counted bytes."""

    consumed_program_sha256: str
    pair_population_sha256: str
    counted_g_packet_bytes: int
    counted_g_packet_sha256: str
    section: CountedProgramSection
    packet: bytes = field(repr=False)
    program: CoupledPreimageProgramV1 = field(repr=False)
    decoded: DecodedCoupledPreimageV1 = field(repr=False, compare=False)
    runtime_binding: CoupledPreimageDirectRuntimeBindingV1
    pair_order_policy: str = PAIR_ORDER_POLICY
    causal_role: str = CAUSAL_ROLE
    counted_g_strict_reapplied: Literal[True] = True
    caller_decoded_g_expected_value_matched: Literal[True] = True
    absolute_pose6_values_present: Literal[False] = False
    pose6_consumed_by_materializer: Literal[False] = False
    research_only: Literal[True] = True
    candidate_score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        _require_sha256(self.consumed_program_sha256, "consumed Pair program SHA-256")
        _require_sha256(self.pair_population_sha256, "PairPopulation SHA-256")
        if type(self.counted_g_packet_bytes) is not int or self.counted_g_packet_bytes <= 0:
            raise CoupledPreimagePairAdapterError("counted G packet byte count must be positive")
        _require_sha256(self.counted_g_packet_sha256, "counted G packet SHA-256")
        if self.section.role is not A_SECTION_ROLE:
            raise CoupledPreimagePairAdapterError("counted A section uses the wrong Pair causal role")
        if self.section.provenance.artifact_class is not A_ARTIFACT_CLASS:
            raise CoupledPreimagePairAdapterError("counted A section provenance class differs")
        if not isinstance(self.packet, bytes) or not self.packet:
            raise CoupledPreimagePairAdapterError("counted A decode did not retain exact packet bytes")
        if self.section.byte_length != len(self.packet) or self.section.payload_sha256 != _sha256(self.packet):
            raise CoupledPreimagePairAdapterError("counted A section accounting differs from retained packet")
        if type(self.program) is not CoupledPreimageProgramV1 or type(self.decoded) is not DecodedCoupledPreimageV1:
            raise CoupledPreimagePairAdapterError("counted A typed parse/decode result differs")
        if self.program.to_packet() != self.packet:
            raise CoupledPreimagePairAdapterError("counted A typed parse did not re-emit exact bytes")
        if self.decoded.materialization_order != CAUSAL_MATERIALIZATION_ORDER:
            raise CoupledPreimagePairAdapterError("counted A materialization order is not Y1-before-Y0")
        if self.decoded.exact_y1_content_sha256 != self.program.source_binding.exact_y1_content_sha256:
            raise CoupledPreimagePairAdapterError("counted A exact Y1 binding differs after decode")
        if not isinstance(self.runtime_binding, CoupledPreimageDirectRuntimeBindingV1):
            raise CoupledPreimagePairAdapterError("counted A lacks direct runtime-source custody")
        if self.pair_order_policy != PAIR_ORDER_POLICY or self.causal_role != CAUSAL_ROLE:
            raise CoupledPreimagePairAdapterError("counted A causal/order policy differs")
        if (
            self.counted_g_strict_reapplied is not True
            or self.caller_decoded_g_expected_value_matched is not True
            or self.absolute_pose6_values_present is not False
            or self.pose6_consumed_by_materializer is not False
            or self.research_only is not True
            or self.candidate_score_claim is not False
            or self.promotion_eligible is not False
        ):
            raise CoupledPreimagePairAdapterError("counted A overclaims Pose6, score, or promotion authority")

    @property
    def mode(self) -> CoupledPreimageMode:
        return self.program.mode

    @property
    def y0(self) -> np.ndarray:
        return self.decoded.y0

    @property
    def y1(self) -> np.ndarray:
        return self.decoded.y1

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "consumed_program_sha256": self.consumed_program_sha256,
            "pair_population_sha256": self.pair_population_sha256,
            "counted_g_packet_bytes": self.counted_g_packet_bytes,
            "counted_g_packet_sha256": self.counted_g_packet_sha256,
            "counted_g_artifact_class": CountedArtifactClass.GENERATOR_PARAMETERS.value,
            "counted_g_strict_reapplied": self.counted_g_strict_reapplied,
            "caller_decoded_g_expected_value_matched": self.caller_decoded_g_expected_value_matched,
            "section": self.section.as_dict(),
            "mode": self.mode.value,
            "source_pair_ids": list(self.program.source_binding.source_pair_ids),
            "exact_y1_content_sha256": self.decoded.exact_y1_content_sha256,
            "expected_y0_content_sha256": self.program.source_binding.expected_y0_content_sha256,
            "chronological_content_sha256": self.decoded.chronological_content_sha256,
            "materialization_order": list(self.decoded.materialization_order),
            "chronological_output_order": list(self.decoded.chronological_output_order),
            "pair_order_policy": self.pair_order_policy,
            "causal_role": self.causal_role,
            "absolute_pose6_values_present": self.absolute_pose6_values_present,
            "pose6_consumed_by_materializer": self.pose6_consumed_by_materializer,
            "runtime_binding": self.runtime_binding.as_dict(),
            "strict_typed_parse_back": True,
            "exact_section_byte_accounting": True,
            "rate_accounting": {
                "counted_G_packet_bytes": self.counted_g_packet_bytes,
                "counted_G_packet_sha256": self.counted_g_packet_sha256,
                "counted_A_packet_bytes": self.section.byte_length,
                "counted_A_packet_sha256": self.section.payload_sha256,
                "generic_decoder_module_bytes": self.runtime_binding.module_bytes,
                "generic_decoder_module_bytes_counted_as_payload": False,
                "rule_118_boundary": "generic deterministic receiver code is free; all source/video/target-derived constants must be counted",
                "source_or_target_derived_constants_in_decoder_independently_audited": False,
                "data_in_code_candidate_closure": False,
            },
            "source_runtime_closure": "DIRECT_MODULE_ONLY_NONTRANSITIVE",
            "legacy_pair_compact_pose_independence_closed": False,
            "candidate_payload_eligible": False,
            "candidate_payload_blockers": [
                LINEAGE_BLOCKER,
                RUNTIME_BLOCKER,
                PAIR_COMPACT_GENERALIZATION_BLOCKER,
                "exact n600 archive replay absent",
            ],
            "research_only": self.research_only,
            "candidate_score_claim": self.candidate_score_claim,
            "promotion_eligible": self.promotion_eligible,
        }

    def to_receipt_bytes(self) -> bytes:
        body = self.as_dict()
        return canonical_json_bytes(
            {
                "schema": ADAPTER_RECEIPT_ENVELOPE_SCHEMA,
                "body": body,
                "body_sha256": canonical_sha256(body),
            }
        )


@dataclass(frozen=True, slots=True)
class AdaptedCoupledPreimagePairProgramV1:
    """A source-custodied exact A section appended to a counted Pair prefix."""

    program_bytes: bytes = field(repr=False)
    section_manifest: tuple[CountedProgramSection, ...]
    source_custody: CoupledPreimageSourceCustodyV1
    consumed: CountedCoupledPreimageDecodeV1

    def __post_init__(self) -> None:
        if not isinstance(self.program_bytes, bytes) or not self.program_bytes:
            raise CoupledPreimagePairAdapterError("adapted Pair program must retain exact bytes")
        if type(self.section_manifest) is not tuple:
            raise CoupledPreimagePairAdapterError("adapted Pair section manifest must be an exact tuple")
        try:
            validate_reverse_causal_counted_program_sections(self.program_bytes, self.section_manifest)
        except PairPopulationEnvelopeError as exc:
            raise CoupledPreimagePairAdapterError("adapted Pair program failed counted manifest validation") from exc
        if self.section_manifest[-1] != self.consumed.section:
            raise CoupledPreimagePairAdapterError("adapted A is not in canonical terminal A position")
        if self.source_custody.artifact.sha256 != self.consumed.section.payload_sha256:
            raise CoupledPreimagePairAdapterError("source-custodied A artifact differs from counted section")

    @property
    def program_sha256(self) -> str:
        return _sha256(self.program_bytes)


@dataclass(frozen=True, slots=True)
class CoupledPreimageCounterfactualV1:
    """Behaviorally verified A-only whole-program mutation."""

    counterfactual: RoleCounterfactualProgram
    section_manifest: tuple[CountedProgramSection, ...]
    causality: SectionCausalityReceipt
    baseline_mode: CoupledPreimageMode
    counterfactual_mode: CoupledPreimageMode
    exact_y1_content_sha256: str
    source_custody: CoupledPreimageSourceCustodyV1
    target_only_bytes_changed: Literal[True] = True
    y0_changed: Literal[True] = True
    y1_changed: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.counterfactual.role is not A_SECTION_ROLE or self.causality.role is not A_SECTION_ROLE:
            raise CoupledPreimagePairAdapterError("A counterfactual uses the wrong Pair role")
        if self.baseline_mode is not self.counterfactual_mode:
            raise CoupledPreimagePairAdapterError("A counterfactual must preserve the selected mode")
        _require_sha256(self.exact_y1_content_sha256, "A counterfactual exact Y1 SHA-256")
        if (
            self.target_only_bytes_changed is not True
            or self.y0_changed is not True
            or self.y1_changed is not False
            or self.research_only is not True
        ):
            raise CoupledPreimagePairAdapterError("A counterfactual causal truth differs")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": COUNTERFACTUAL_SCHEMA,
            "role": self.counterfactual.role.value,
            "counterfactual_program_sha256": self.counterfactual.program_sha256,
            "section_manifest": [section.as_dict() for section in self.section_manifest],
            "causality": self.causality.as_dict(),
            "baseline_mode": self.baseline_mode.value,
            "counterfactual_mode": self.counterfactual_mode.value,
            "exact_y1_content_sha256": self.exact_y1_content_sha256,
            "source_custody": self.source_custody.as_dict(),
            "target_only_bytes_changed": self.target_only_bytes_changed,
            "y0_changed": self.y0_changed,
            "y1_changed": self.y1_changed,
            "research_only": self.research_only,
        }


def _a_section(sections: tuple[CountedProgramSection, ...]) -> CountedProgramSection:
    matches = tuple(section for section in sections if section.role is A_SECTION_ROLE)
    if len(matches) != 1:
        raise CoupledPreimagePairAdapterError("counted Pair program must contain exactly one A section")
    return matches[0]


def _g_section(sections: tuple[CountedProgramSection, ...]) -> CountedProgramSection:
    matches = tuple(section for section in sections if section.role is CountedSectionRole.GENERATIVE_CORRECTION)
    if len(matches) != 1:
        raise CoupledPreimagePairAdapterError("reverse-causal counted Pair program must contain exactly one G section")
    return matches[0]


def _require_pair_order(program: CoupledPreimageProgramV1, pair_population: PairPopulation) -> None:
    expected_ids = pair_population.domain_index(PairDomain.V10).local_to_source_pair_ids
    if program.source_binding.source_pair_ids != expected_ids:
        raise CoupledPreimagePairAdapterError("A source-pair IDs/order differ from the PairPopulation V10 local index")


def consume_counted_coupled_preimage(
    program_bytes: bytes,
    section_manifest: tuple[CountedProgramSection, ...],
    *,
    pair_population: PairPopulation,
    predictor_state: PredictorSemanticStateV1,
    decoded_g: DecodedGenerativeCorrectionV1,
) -> CountedCoupledPreimageDecodeV1:
    """Slice, strict-parse, and execute A from the exact counted Pair bytes."""

    if type(pair_population) is not PairPopulation:
        raise CoupledPreimagePairAdapterError("A consumer requires exact PairPopulation")
    try:
        validate_reverse_causal_counted_program_sections(program_bytes, section_manifest)
    except PairPopulationEnvelopeError as exc:
        raise CoupledPreimagePairAdapterError("counted Pair manifest failed before A consumption") from exc
    generative_section = _g_section(section_manifest)
    if generative_section.provenance.artifact_class is not CountedArtifactClass.GENERATOR_PARAMETERS:
        raise CoupledPreimagePairAdapterError("counted G section is not typed as generator parameters")
    generative_packet = program_bytes[generative_section.offset : generative_section.stop]
    try:
        strict_decoded_g = apply_generative_taskspace_correction(
            generative_packet,
            predictor_state=predictor_state,
        )
    except GenerativeTaskspaceCorrectionError as exc:
        raise CoupledPreimagePairAdapterError("counted G packet failed strict reapply before A consumption") from exc
    if strict_decoded_g.correction_packet_sha256 != generative_section.payload_sha256:
        raise CoupledPreimagePairAdapterError("strict counted G reapply differs from its retained section hash")
    if (
        type(decoded_g) is not DecodedGenerativeCorrectionV1
        or decoded_g.correction_packet_sha256 != strict_decoded_g.correction_packet_sha256
        or decoded_g.realization_profile != strict_decoded_g.realization_profile
        or not np.array_equal(decoded_g.labels, strict_decoded_g.labels)
    ):
        raise CoupledPreimagePairAdapterError("caller decoded_g differs from strict counted G reapply")
    section = _a_section(section_manifest)
    if section.provenance.artifact_class is not A_ARTIFACT_CLASS:
        raise CoupledPreimagePairAdapterError("FRAME0 section is not typed as coupled preimage A parameters")
    packet = program_bytes[section.offset : section.stop]
    try:
        parsed = parse_coupled_preimage_program(packet)
        _require_pair_order(parsed, pair_population)
        decoded = decode_coupled_preimage_program(
            packet,
            predictor_state=predictor_state,
            decoded_g=strict_decoded_g,
        )
    except CoupledPreimageProgramError as exc:
        raise CoupledPreimagePairAdapterError("counted A packet failed strict reverse-causal consumption") from exc
    runtime_binding = CoupledPreimageDirectRuntimeBindingV1.reopen()
    return CountedCoupledPreimageDecodeV1(
        consumed_program_sha256=_sha256(program_bytes),
        pair_population_sha256=pair_population.population_sha256,
        counted_g_packet_bytes=generative_section.byte_length,
        counted_g_packet_sha256=generative_section.payload_sha256,
        section=section,
        packet=packet,
        program=parsed,
        decoded=decoded,
        runtime_binding=runtime_binding,
    )


def append_source_custodied_coupled_preimage(
    prefix_program_bytes: bytes,
    prefix_sections: tuple[CountedProgramSection, ...],
    *,
    source_custody: CoupledPreimageSourceCustodyV1,
    pair_population: PairPopulation,
    predictor_state: PredictorSemanticStateV1,
    decoded_g: DecodedGenerativeCorrectionV1,
) -> AdaptedCoupledPreimagePairProgramV1:
    """Append exact source-custodied A bytes after the sole counted G section."""

    if not isinstance(prefix_program_bytes, bytes) or not prefix_program_bytes:
        raise CoupledPreimagePairAdapterError("counted A adapter requires a nonempty Pair program prefix")
    if type(prefix_sections) is not tuple:
        raise CoupledPreimagePairAdapterError("counted A prefix sections must be an exact tuple")
    prefix_roles = tuple(section.role for section in prefix_sections)
    if prefix_roles != (CountedSectionRole.GENERATIVE_CORRECTION,):
        raise CoupledPreimagePairAdapterError(
            "reverse-causal A adapter requires exactly counted G as its complete prefix"
        )
    cursor = 0
    for section in prefix_sections:
        if section.offset != cursor or section.stop > len(prefix_program_bytes):
            raise CoupledPreimagePairAdapterError("counted Pair prefix has a gap, overlap, or invalid span")
        payload = prefix_program_bytes[section.offset : section.stop]
        if _sha256(payload) != section.payload_sha256:
            raise CoupledPreimagePairAdapterError("counted Pair prefix section hash differs")
        cursor = section.stop
    if cursor != len(prefix_program_bytes):
        raise CoupledPreimagePairAdapterError("counted Pair prefix has unconsumed bytes")

    packet = source_custody.reopen_packet()
    try:
        parsed = parse_coupled_preimage_program(packet)
    except CoupledPreimageProgramError as exc:
        raise CoupledPreimagePairAdapterError("source-custodied A packet failed parse-back") from exc
    provenance = source_custody.counted_provenance(parsed)
    section = CountedProgramSection(
        role=A_SECTION_ROLE,
        offset=len(prefix_program_bytes),
        byte_length=len(packet),
        payload_sha256=_sha256(packet),
        provenance=provenance,
    )
    program_bytes = prefix_program_bytes + packet
    sections = (*prefix_sections, section)
    consumed = consume_counted_coupled_preimage(
        program_bytes,
        sections,
        pair_population=pair_population,
        predictor_state=predictor_state,
        decoded_g=decoded_g,
    )
    return AdaptedCoupledPreimagePairProgramV1(
        program_bytes=program_bytes,
        section_manifest=sections,
        source_custody=source_custody,
        consumed=consumed,
    )


def _conditioning_identity(program: CoupledPreimageProgramV1) -> tuple[Any, ...]:
    source = program.source_binding
    return (
        source.predictor_program_sha256,
        source.predictor_renderer_sha256,
        source.source_pair_ids,
        source.predictor_labels_sha256,
        source.correction_packet_sha256,
        source.correction_labels_sha256,
        source.realization_profile_sha256,
        source.exact_y1_content_sha256,
    )


def _replace_section(
    program_bytes: bytes,
    sections: tuple[CountedProgramSection, ...],
    *,
    role: CountedSectionRole,
    replacement: bytes,
) -> tuple[bytes, tuple[CountedProgramSection, ...]]:
    payloads: list[bytes] = []
    rebuilt: list[CountedProgramSection] = []
    cursor = 0
    for section in sections:
        payload = replacement if section.role is role else program_bytes[section.offset : section.stop]
        payloads.append(payload)
        rebuilt.append(
            CountedProgramSection(
                role=section.role,
                offset=cursor,
                byte_length=len(payload),
                payload_sha256=_sha256(payload),
                provenance=section.provenance,
            )
        )
        cursor += len(payload)
    result = b"".join(payloads)
    manifest = tuple(rebuilt)
    try:
        validate_reverse_causal_counted_program_sections(result, manifest)
    except PairPopulationEnvelopeError as exc:
        raise CoupledPreimagePairAdapterError("A counterfactual rebuilt an invalid counted Pair program") from exc
    return result, manifest


def build_target_only_a_counterfactual(
    baseline_program_bytes: bytes,
    baseline_sections: tuple[CountedProgramSection, ...],
    *,
    counterfactual_custody: CoupledPreimageSourceCustodyV1,
    pair_population: PairPopulation,
    predictor_state: PredictorSemanticStateV1,
    decoded_g: DecodedGenerativeCorrectionV1,
) -> CoupledPreimageCounterfactualV1:
    """Replace only counted A and prove Y0 changes while exact Y1 does not."""

    baseline = consume_counted_coupled_preimage(
        baseline_program_bytes,
        baseline_sections,
        pair_population=pair_population,
        predictor_state=predictor_state,
        decoded_g=decoded_g,
    )
    counterfactual_packet = counterfactual_custody.reopen_packet()
    try:
        counterfactual_program = parse_coupled_preimage_program(counterfactual_packet)
    except CoupledPreimageProgramError as exc:
        raise CoupledPreimagePairAdapterError("counterfactual A failed strict typed parse-back") from exc
    if counterfactual_program.mode is not baseline.mode:
        raise CoupledPreimagePairAdapterError("A-only counterfactual cannot change the selected A mode")
    if _conditioning_identity(counterfactual_program) != _conditioning_identity(baseline.program):
        raise CoupledPreimagePairAdapterError("A-only counterfactual changed P/G, pair order, or exact Y1 conditioning")
    if counterfactual_custody.counted_provenance(counterfactual_program) != baseline.section.provenance:
        raise CoupledPreimagePairAdapterError("A-only counterfactual changed typed section provenance")

    counterfactual_bytes, counterfactual_sections = _replace_section(
        baseline_program_bytes,
        baseline_sections,
        role=A_SECTION_ROLE,
        replacement=counterfactual_packet,
    )
    counterfactual = consume_counted_coupled_preimage(
        counterfactual_bytes,
        counterfactual_sections,
        pair_population=pair_population,
        predictor_state=predictor_state,
        decoded_g=decoded_g,
    )
    if not np.array_equal(baseline.y1, counterfactual.y1):
        raise CoupledPreimagePairAdapterError("A-only counterfactual changed exact Y1")
    if np.array_equal(baseline.y0, counterfactual.y0):
        raise CoupledPreimagePairAdapterError("A-only counterfactual did not change realized Y0")

    baseline_payloads = {
        section.role: baseline_program_bytes[section.offset : section.stop] for section in baseline_sections
    }
    counterfactual_payloads = {
        section.role: counterfactual_bytes[section.offset : section.stop] for section in counterfactual_sections
    }
    for role, payload in baseline_payloads.items():
        if role is not A_SECTION_ROLE and counterfactual_payloads[role] != payload:
            raise CoupledPreimagePairAdapterError(f"A-only counterfactual changed non-target {role.value} bytes")
    unchanged_non_target_sha256 = canonical_sha256(
        [
            {
                "role": role.value,
                "payload_sha256": _sha256(payload),
                "byte_length": len(payload),
            }
            for role, payload in baseline_payloads.items()
            if role is not A_SECTION_ROLE
        ]
    )
    causality = SectionCausalityReceipt(
        role=A_SECTION_ROLE,
        baseline_program_sha256=_sha256(baseline_program_bytes),
        counterfactual_program_sha256=_sha256(counterfactual_bytes),
        baseline_section_sha256=baseline.section.payload_sha256,
        counterfactual_section_sha256=counterfactual.section.payload_sha256,
        unchanged_non_target_sections_sha256=unchanged_non_target_sha256,
        y0_changed=True,
        y1_changed=False,
    )
    return CoupledPreimageCounterfactualV1(
        counterfactual=RoleCounterfactualProgram(A_SECTION_ROLE, counterfactual_bytes),
        section_manifest=counterfactual_sections,
        causality=causality,
        baseline_mode=baseline.mode,
        counterfactual_mode=counterfactual.mode,
        exact_y1_content_sha256=baseline.decoded.exact_y1_content_sha256,
        source_custody=counterfactual_custody,
    )


__all__ = [
    "ADAPTER_SCHEMA",
    "A_ARTIFACT_CLASS",
    "A_SECTION_ROLE",
    "AdaptedCoupledPreimagePairProgramV1",
    "CountedCoupledPreimageDecodeV1",
    "CoupledPreimageCounterfactualV1",
    "CoupledPreimageDirectRuntimeBindingV1",
    "CoupledPreimagePairAdapterError",
    "CoupledPreimageSourceCustodyV1",
    "ReopenedFileCustodyV1",
    "append_source_custodied_coupled_preimage",
    "build_target_only_a_counterfactual",
    "consume_counted_coupled_preimage",
]
