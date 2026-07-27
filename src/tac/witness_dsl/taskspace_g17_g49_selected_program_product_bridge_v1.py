# SPDX-License-Identifier: MIT
"""Exact G17/G49 preplacement product and current production blockers.

The canonical G17 active-A ABI now owns the population-global ``TSPPV1``
family/mode and strict parser.  This module binds exact fresh semantic P bytes,
an exact G49 packet, CarrierCompose, BoundV10, and factor-2 streaming without a
dense n600 bank.

The current monolithic G17 container deliberately rejects a ZIP-valued P
section, while the exact fresh semantic program is itself a CarrierCompose ZIP.
The canonical placement ontology also lacks exact ``SEMANTIC_PROGRAM`` and
``SELECTED_PREIMAGE_PROGRAM`` logical value types.  This module therefore
builds and twice reopens exact G/A/E sections, executes the real container
refusal, and emits those typed blockers.  It never recasts P as topology, G49
as VM bytecode, or receiver-incidence names as execution evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np

from tac.witness_dsl.taskspace_g17_production_envelope import (
    G17ADescriptorV1,
    G17AFamily,
    G17AMode,
    G17G49SelectedPreimageStrictParserV1,
    G17GStrictNestedParser,
    G17PopulationLayout,
    G17ProductionEnvelopeError,
    build_g17_a_packet,
    build_g17_g_packet,
    build_g17_production_archive,
    build_g17_terminal_envelope,
    parse_g17_a_packet,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    BoundV10Factor2SelectedPreimageDecoderV1,
    SelectedPreimageByteHomeV1,
    SelectedPreimageFactor2PairV1,
    SelectedPreimageFactorModeV1,
    SelectedPreimageFactorRoleV1,
    TaskspaceSelectedPreimageProgramV1,
    encode_selected_preimage_program,
    parse_selected_preimage_program,
    realize_selected_preimage_pair_factor2,
)
from tac.witness_dsl.taskspace_selected_solution_compiler import (
    G17CompilerPlacementManifestV1,
    G17ScientificRoleV1,
    G17SemanticStreamRoleV1,
)

SCHEMA: Final = "tac.g17_g49_selected_program_product_blocker.v1"
OWNER_INCIDENCE_SCHEMA: Final = "tac.g17_owner_specific_receiver_incidence.v1"

SEMANTIC_PROGRAM_TYPE_BLOCKER: Final = "G17_SEMANTIC_PROGRAM_LOGICAL_VALUE_TYPE_OWED"
SELECTED_PREIMAGE_PROGRAM_TYPE_BLOCKER: Final = "G17_SELECTED_PREIMAGE_PROGRAM_LOGICAL_VALUE_TYPE_OWED"
NESTED_P_CONTAINER_BLOCKER: Final = "G17_SEMANTIC_PROGRAM_P_NESTED_ARCHIVE_CONTAINER_ABI_OWED"
INCIDENCE_EXECUTION_BLOCKER: Final = "G17_OWNER_SPECIFIC_RECEIVER_INCIDENCE_EXECUTION_RECEIPT_OWED"
OPEN_PRODUCT_BLOCKERS: Final = (
    SEMANTIC_PROGRAM_TYPE_BLOCKER,
    SELECTED_PREIMAGE_PROGRAM_TYPE_BLOCKER,
    NESTED_P_CONTAINER_BLOCKER,
    INCIDENCE_EXECUTION_BLOCKER,
)

SHARED_ARCHIVE_RECEIVER_CONSUMER: Final = "G17_PRODUCTION_ARCHIVE_RECEIVER_V1"
SHARED_ARCHIVE_UNPACK_OPERATION: Final = "UNPACK_EXACT_G17_ARCHIVE_MEMBER_V1"
P_RECEIVER_CONSUMER: Final = "CARRIER_COMPOSE_RECEIVER_V1"
P_RECEIVER_OPERATION: Final = "REOPEN_EXACT_SEMANTIC_P_V1"
A_RECEIVER_CONSUMER: Final = "BOUND_V10_FACTOR2_SELECTED_PREIMAGE_DECODER_V1"
A_RECEIVER_OPERATION: Final = "PARSE_TSPPV1_AND_STREAM_FACTOR2_V1"


class G17G49SelectedProgramProductError(ValueError):
    """The exact selected-program product or blocker proof is invalid."""


def _sha256(payload: bytes | memoryview | np.ndarray) -> str:
    digest = hashlib.sha256()
    if type(payload) is np.ndarray:
        digest.update(memoryview(np.ascontiguousarray(payload)).cast("B"))
    else:
        digest.update(payload)
    return digest.hexdigest()


def _ascii(value: object, *, label: str) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise G17G49SelectedProgramProductError(f"{label} must be nonempty ASCII")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True, slots=True, order=True)
class G17OwnerSpecificReceiverIncidenceV1:
    """One logical owner's downstream operation after a physical-group unpack."""

    physical_group_id: str
    logical_owner_id: str
    receiver_consumer: str
    receiver_operation: str

    def __post_init__(self) -> None:
        for label in (
            "physical_group_id",
            "logical_owner_id",
            "receiver_consumer",
            "receiver_operation",
        ):
            _ascii(getattr(self, label), label=label)


@dataclass(frozen=True, slots=True)
class G17OwnerSpecificReceiverIncidenceManifestV1:
    """Generic owner-to-group-to-receiver relation, not execution evidence."""

    placement_manifest: G17CompilerPlacementManifestV1
    incidences: tuple[G17OwnerSpecificReceiverIncidenceV1, ...]

    def __post_init__(self) -> None:
        if type(self.placement_manifest) is not G17CompilerPlacementManifestV1:
            raise G17G49SelectedProgramProductError("incidence manifest requires exact G17 placement manifest")
        if (
            type(self.incidences) is not tuple
            or not self.incidences
            or any(type(row) is not G17OwnerSpecificReceiverIncidenceV1 for row in self.incidences)
        ):
            raise G17G49SelectedProgramProductError("owner-specific incidences require a nonempty exact tuple")
        if self.incidences != tuple(sorted(self.incidences)):
            raise G17G49SelectedProgramProductError("owner-specific incidences are not in canonical order")
        if len(set(self.incidences)) != len(self.incidences):
            raise G17G49SelectedProgramProductError("owner-specific receiver incidence is duplicated")

        groups = {group.group_id: group for group in self.placement_manifest.coding_groups}
        required_pairs = {
            (group.group_id, owner_id)
            for group in self.placement_manifest.coding_groups
            for owner_id in group.logical_owner_ids
        }
        observed_pairs: set[tuple[str, str]] = set()
        for row in self.incidences:
            group = groups.get(row.physical_group_id)
            if group is None or row.logical_owner_id not in group.logical_owner_ids:
                raise G17G49SelectedProgramProductError(
                    "owner-specific incidence references an absent physical-group ownership edge"
                )
            pair = (row.physical_group_id, row.logical_owner_id)
            if pair in observed_pairs:
                raise G17G49SelectedProgramProductError(
                    "one physical-group owner has multiple competing receiver operations"
                )
            observed_pairs.add(pair)
        if observed_pairs != required_pairs:
            raise G17G49SelectedProgramProductError(
                "owner-specific receiver incidences leave a physical-group owner dead or add an owner"
            )
        for group in self.placement_manifest.coding_groups:
            if len(group.logical_owner_ids) > 1 and (
                group.receiver_consumer != SHARED_ARCHIVE_RECEIVER_CONSUMER
                or group.receiver_operation != SHARED_ARCHIVE_UNPACK_OPERATION
            ):
                raise G17G49SelectedProgramProductError(
                    "a shared physical group must name only exact archive unpack; owner operations belong "
                    "on additive incidences"
                )

    @property
    def manifest_sha256(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "execution_evidence": False,
                    "incidences": [
                        {
                            "logical_owner_id": row.logical_owner_id,
                            "physical_group_id": row.physical_group_id,
                            "receiver_consumer": row.receiver_consumer,
                            "receiver_operation": row.receiver_operation,
                        }
                        for row in self.incidences
                    ],
                    "placement_manifest_sha256": self.placement_manifest.manifest_sha256,
                    "schema": OWNER_INCIDENCE_SCHEMA,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class G17G49SelectedProgramPreplacementProductV1:
    """Exact P x G49 packet with the real CarrierCompose/BoundV10 receiver."""

    semantic_p_archive: bytes = field(repr=False)
    selected_program_packet: bytes = field(repr=False)
    decoder: BoundV10Factor2SelectedPreimageDecoderV1 = field(repr=False)
    maximum_packet_bytes: int
    program: TaskspaceSelectedPreimageProgramV1 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.semantic_p_archive) is not bytes or not self.semantic_p_archive:
            raise G17G49SelectedProgramProductError("semantic P must be exact nonempty bytes")
        if type(self.selected_program_packet) is not bytes or not self.selected_program_packet:
            raise G17G49SelectedProgramProductError("selected-program A operand must be exact nonempty bytes")
        if type(self.decoder) is not BoundV10Factor2SelectedPreimageDecoderV1:
            raise G17G49SelectedProgramProductError("product requires the exact concrete BoundV10 decoder")
        if (
            type(self.maximum_packet_bytes) is not int
            or isinstance(self.maximum_packet_bytes, bool)
            or self.maximum_packet_bytes < len(self.selected_program_packet)
        ):
            raise G17G49SelectedProgramProductError("maximum packet bound is invalid or below exact A bytes")
        try:
            program = parse_selected_preimage_program(
                self.selected_program_packet,
                maximum_packet_bytes=self.maximum_packet_bytes,
            )
        except Exception as exc:
            raise G17G49SelectedProgramProductError("exact selected-program A operand failed strict parse") from exc
        if encode_selected_preimage_program(program) != self.selected_program_packet:
            raise G17G49SelectedProgramProductError("selected-program A operand changed on parse/re-encode")
        if self.decoder.carrier_receiver.archive != self.semantic_p_archive:
            raise G17G49SelectedProgramProductError("BoundV10 consumes different bytes than exact semantic P")
        if (
            program.semantic_program_identity != self.decoder.semantic_identity
            or program.target_custody_identity != self.decoder.target_custody_identity
        ):
            raise G17G49SelectedProgramProductError("G49 program identities differ from the concrete decoder")
        if (
            program.decoder_identity.decoder_id != self.decoder.decoder_id
            or program.decoder_identity.implementation_source_sha256 != self.decoder.implementation_source_sha256
        ):
            raise G17G49SelectedProgramProductError("G49 decoder identity/source differs from BoundV10")
        if program.semantic_program_identity.compiled_semantic_archive_sha256 != _sha256(
            self.semantic_p_archive
        ) or program.semantic_program_identity.compiled_semantic_archive_bytes != len(self.semantic_p_archive):
            raise G17G49SelectedProgramProductError("fresh semantic identity differs from exact P custody")
        object.__setattr__(self, "program", program)

    @property
    def semantic_p_sha256(self) -> str:
        return _sha256(self.semantic_p_archive)

    @property
    def selected_program_packet_sha256(self) -> str:
        return _sha256(self.selected_program_packet)

    def iter_factor2_pairs(
        self,
        *,
        pair_start: int,
        pair_count: int,
    ) -> Iterator[SelectedPreimageFactor2PairV1]:
        """Stream exact pairs and double-realize each pair without an n600 bank."""

        if (
            type(pair_start) is not int
            or isinstance(pair_start, bool)
            or type(pair_count) is not int
            or isinstance(pair_count, bool)
            or pair_start < 0
            or pair_count < 1
            or pair_start + pair_count > self.program.compile_config.pair_count
        ):
            raise G17G49SelectedProgramProductError("factor2 stream window escapes the exact program population")
        for pair_index in range(pair_start, pair_start + pair_count):
            first = realize_selected_preimage_pair_factor2(self.program, pair_index, self.decoder)
            second = realize_selected_preimage_pair_factor2(self.program, pair_index, self.decoder)
            if _factor2_pair_sha256(first) != _factor2_pair_sha256(second):
                raise G17G49SelectedProgramProductError("P x A factor2 receiver is nondeterministic")
            yield first


def _factor2_pair_sha256(pair: SelectedPreimageFactor2PairV1) -> str:
    if type(pair) is not SelectedPreimageFactor2PairV1:
        raise G17G49SelectedProgramProductError("factor2 stream returned a noncanonical pair type")
    digest = hashlib.sha256(b"G17-G49-FACTOR2-PAIR-V1\0")
    for value in (pair.scorer_y0, pair.scorer_y1, pair.camera_y0, pair.camera_y1):
        digest.update(bytes.fromhex(_sha256(value)))
    return digest.hexdigest()


def selected_program_role_incidences(
    program: TaskspaceSelectedPreimageProgramV1,
) -> tuple[tuple[G17ScientificRoleV1, G17SemanticStreamRoleV1], ...]:
    """Derive exact scientific placement roles from parsed factor roles/modes."""

    if type(program) is not TaskspaceSelectedPreimageProgramV1:
        raise G17G49SelectedProgramProductError("role derivation requires exact selected-preimage program")
    roles: set[tuple[G17ScientificRoleV1, G17SemanticStreamRoleV1]] = set()
    for factor in program.factors:
        if (
            factor.role is SelectedPreimageFactorRoleV1.ANALYTIC_RESIDUAL
            and factor.mode is SelectedPreimageFactorModeV1.SHEARLET_BOUNDARY_TRANSPORT_Q4
        ):
            roles.add(
                (
                    G17ScientificRoleV1.BULK_BOUNDARY,
                    G17SemanticStreamRoleV1.RESIDUAL,
                )
            )
        elif (
            factor.role is SelectedPreimageFactorRoleV1.LEARNED_IRREDUCIBLE_QUOTIENT
            and factor.mode is SelectedPreimageFactorModeV1.COMPACT_LATENT_QUOTIENT_PLUGIN
        ):
            roles.add(
                (
                    G17ScientificRoleV1.IRREDUCIBLE_QUOTIENT,
                    G17SemanticStreamRoleV1.RESIDUAL,
                )
            )
        else:
            raise G17G49SelectedProgramProductError(
                "selected-preimage factor role/mode lacks an exact scientific placement mapping"
            )
    if not roles:
        raise G17G49SelectedProgramProductError("selected-preimage program has no derived scientific roles")
    return tuple(sorted(roles, key=lambda row: (row[0].value, row[1].value)))


@dataclass(frozen=True, slots=True)
class G17SelectedProgramComponentIncidenceV1:
    """One exact inner-packet component; never a claimed ZIP byte range."""

    section_id: str
    packet_offset: int
    byte_length: int
    payload_sha256: str
    byte_home: str
    lineage_class: str
    factor_role: str | None
    factor_mode: str | None
    scientific_role: str | None
    semantic_role: str | None

    def as_dict(self) -> dict[str, object]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}


def selected_program_component_incidences(
    program: TaskspaceSelectedPreimageProgramV1,
) -> tuple[G17SelectedProgramComponentIncidenceV1, ...]:
    """Preserve exact packet offsets, multiplicity, byte homes, and factor roles."""

    if type(program) is not TaskspaceSelectedPreimageProgramV1:
        raise G17G49SelectedProgramProductError("component incidence requires exact selected-preimage program")
    factors = {factor.section_id: factor for factor in program.factors}
    rows: list[G17SelectedProgramComponentIncidenceV1] = []
    for home in program.byte_homes():
        factor = factors.get(home.section_id)
        if factor is None:
            if home.byte_home is not SelectedPreimageByteHomeV1.COUNTED_PACKET_FRAMING:
                raise G17G49SelectedProgramProductError("unmatched selected-program byte-home component")
            role = mode = scientific = semantic = None
        else:
            derived_program = TaskspaceSelectedPreimageProgramV1(
                semantic_program_identity=program.semantic_program_identity,
                target_custody_identity=program.target_custody_identity,
                decoder_identity=program.decoder_identity,
                compile_config=program.compile_config,
                factors=(factor,),
            )
            derived = selected_program_role_incidences(derived_program)
            if len(derived) != 1:
                raise G17G49SelectedProgramProductError("one factor derived an ambiguous scientific incidence")
            role = factor.role.value
            mode = factor.mode.value
            scientific = derived[0][0].value
            semantic = derived[0][1].value
        rows.append(
            G17SelectedProgramComponentIncidenceV1(
                section_id=home.section_id,
                packet_offset=home.offset,
                byte_length=home.byte_length,
                payload_sha256=home.payload_sha256,
                byte_home=home.byte_home.value,
                lineage_class=home.lineage_class.value,
                factor_role=role,
                factor_mode=mode,
                scientific_role=scientific,
                semantic_role=semantic,
            )
        )
    if sum(row.byte_length for row in rows) != len(program.packet_bytes):
        raise G17G49SelectedProgramProductError("component incidences do not partition exact TSPPV1 bytes")
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class G17G49SelectedProgramProductBlockerReceiptV1:
    schema: Literal["tac.g17_g49_selected_program_product_blocker.v1"]
    semantic_p_sha256: str
    semantic_p_bytes: int
    g_packet_sha256: str
    g_packet_bytes: int
    selected_program_packet_sha256: str
    selected_program_packet_bytes: int
    a_packet_sha256: str
    a_packet_bytes: int
    terminal_packet_sha256: str
    terminal_packet_bytes: int
    selected_program_component_incidences: tuple[G17SelectedProgramComponentIncidenceV1, ...]
    exact_g49_global_a_parse_reencode_twice: Literal[True]
    exact_p_is_boundv10_carrier_archive: Literal[True]
    boundv10_decoder_source_matched: Literal[True]
    preplacement_factor2_stream_callable: Literal[True]
    semantic_program_logical_value_type_available: Literal[False]
    selected_preimage_program_logical_value_type_available: Literal[False]
    owner_specific_receiver_incidence_type_landed: Literal[True]
    product_structural_routing_closed: Literal[False]
    owner_specific_receiver_execution_evidence: Literal[False]
    g17_product_archive_built: Literal[False]
    g17_product_archive_reopened: Literal[False]
    open_product_blockers: tuple[str, ...]
    dense_n600_materialized: Literal[False]
    additional_unowned_or_unembedded_target_payload_inputs: Literal[0]
    research_only: Literal[True]
    candidate_claim: Literal[False]
    score_claim: Literal[False]
    pointer_moved: Literal[False]

    def as_dict(self) -> dict[str, object]:
        result = {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}
        result["selected_program_component_incidences"] = [
            row.as_dict() for row in self.selected_program_component_incidences
        ]
        return result

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


@dataclass(frozen=True, slots=True)
class G17G49SelectedProgramProductBlockerProofV1:
    product: G17G49SelectedProgramPreplacementProductV1 = field(repr=False)
    g_section: bytes = field(repr=False)
    a_section: bytes = field(repr=False)
    terminal_section: bytes = field(repr=False)
    a_parser: G17G49SelectedPreimageStrictParserV1 = field(repr=False)
    receipt: G17G49SelectedProgramProductBlockerReceiptV1


def _exception_chain_contains(exc: BaseException, text: str) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if text in str(current):
            return True
        current = current.__cause__
    return False


def audit_g17_g49_selected_program_product_blockers(
    *,
    product: G17G49SelectedProgramPreplacementProductV1,
    g_section: bytes | None = None,
    g_active_parser: G17GStrictNestedParser | None = None,
    max_member_bytes: int = 64 * 1024 * 1024,
) -> G17G49SelectedProgramProductBlockerProofV1:
    """Build exact G/A/E, then execute and classify the nested-P refusal."""

    if type(product) is not G17G49SelectedProgramPreplacementProductV1:
        raise G17G49SelectedProgramProductError("blocker audit requires the exact preplacement product")
    start = product.program.compile_config.source_pair_start
    count = product.program.compile_config.pair_count
    if g_section is None:
        g_packet = build_g17_g_packet(
            p_section=product.semantic_p_archive,
            pair_start=start,
            pair_count=count,
        )
    elif type(g_section) is bytes and g_section:
        g_packet = g_section
    else:
        raise G17G49SelectedProgramProductError("G section must be exact nonempty bytes or omitted for PASS")
    a_parser = G17G49SelectedPreimageStrictParserV1(
        expected_semantic_program_identity=product.program.semantic_program_identity,
        expected_target_custody_identity=product.program.target_custody_identity,
        expected_decoder_identity=product.program.decoder_identity,
        maximum_packet_bytes=product.maximum_packet_bytes,
    )
    active = a_parser(
        product.selected_program_packet,
        start,
        count,
        G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
        G17AMode.SELECTED_PREIMAGE_PROGRAM,
    )
    a_section = build_g17_a_packet(
        p_section=product.semantic_p_archive,
        g_section=g_packet,
        pair_start=start,
        pair_count=count,
        layout=G17PopulationLayout.GLOBAL,
        descriptors=(
            G17ADescriptorV1(
                pair_start=start,
                pair_count=count,
                family=G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
                mode=G17AMode.SELECTED_PREIMAGE_PROGRAM,
                active=active,
            ),
        ),
    )
    first_a = parse_g17_a_packet(
        a_section,
        expected_p_section=product.semantic_p_archive,
        expected_g_section=g_packet,
        active_parser=a_parser,
    )
    second_a = parse_g17_a_packet(
        a_section,
        expected_p_section=product.semantic_p_archive,
        expected_g_section=g_packet,
        active_parser=a_parser,
    )
    if first_a != second_a or first_a.descriptors[0].payload != product.selected_program_packet:
        raise G17G49SelectedProgramProductError("exact global G49 A changed on deterministic double reopen")
    terminal = build_g17_terminal_envelope(
        p_section=product.semantic_p_archive,
        g_section=g_packet,
        a_section=a_section,
        g_active_parser=g_active_parser,
        a_active_parser=a_parser,
    )
    try:
        build_g17_production_archive(
            p_section=product.semantic_p_archive,
            g_section=g_packet,
            a_section=a_section,
            terminal_section=terminal,
            g_active_parser=g_active_parser,
            a_active_parser=a_parser,
            max_member_bytes=max_member_bytes,
        )
    except G17ProductionEnvelopeError as exc:
        if not _exception_chain_contains(exc, "nested ZIP"):
            raise G17G49SelectedProgramProductError(
                "G17 product failed for a reason other than the registered nested-P blocker"
            ) from exc
    else:
        raise G17G49SelectedProgramProductError(
            "nested semantic P was admitted; this blocker audit is obsolete and must be replaced by success closure"
        )

    components = selected_program_component_incidences(product.program)
    receipt = G17G49SelectedProgramProductBlockerReceiptV1(
        schema=SCHEMA,
        semantic_p_sha256=product.semantic_p_sha256,
        semantic_p_bytes=len(product.semantic_p_archive),
        g_packet_sha256=_sha256(g_packet),
        g_packet_bytes=len(g_packet),
        selected_program_packet_sha256=product.selected_program_packet_sha256,
        selected_program_packet_bytes=len(product.selected_program_packet),
        a_packet_sha256=_sha256(a_section),
        a_packet_bytes=len(a_section),
        terminal_packet_sha256=_sha256(terminal),
        terminal_packet_bytes=len(terminal),
        selected_program_component_incidences=components,
        exact_g49_global_a_parse_reencode_twice=True,
        exact_p_is_boundv10_carrier_archive=True,
        boundv10_decoder_source_matched=True,
        preplacement_factor2_stream_callable=True,
        semantic_program_logical_value_type_available=False,
        selected_preimage_program_logical_value_type_available=False,
        owner_specific_receiver_incidence_type_landed=True,
        product_structural_routing_closed=False,
        owner_specific_receiver_execution_evidence=False,
        g17_product_archive_built=False,
        g17_product_archive_reopened=False,
        open_product_blockers=OPEN_PRODUCT_BLOCKERS,
        dense_n600_materialized=False,
        additional_unowned_or_unembedded_target_payload_inputs=0,
        research_only=True,
        candidate_claim=False,
        score_claim=False,
        pointer_moved=False,
    )
    return G17G49SelectedProgramProductBlockerProofV1(
        product=product,
        g_section=g_packet,
        a_section=a_section,
        terminal_section=terminal,
        a_parser=a_parser,
        receipt=receipt,
    )


def refuse_g17_g49_product_archive(
    proof: G17G49SelectedProgramProductBlockerProofV1,
) -> None:
    """Refuse an archive/candidate claim while exact product blockers remain."""

    if type(proof) is not G17G49SelectedProgramProductBlockerProofV1:
        raise G17G49SelectedProgramProductError("product refusal requires exact blocker proof")
    raise G17G49SelectedProgramProductError(",".join(proof.receipt.open_product_blockers))


__all__ = [
    "A_RECEIVER_CONSUMER",
    "A_RECEIVER_OPERATION",
    "INCIDENCE_EXECUTION_BLOCKER",
    "NESTED_P_CONTAINER_BLOCKER",
    "OPEN_PRODUCT_BLOCKERS",
    "OWNER_INCIDENCE_SCHEMA",
    "P_RECEIVER_CONSUMER",
    "P_RECEIVER_OPERATION",
    "SCHEMA",
    "SELECTED_PREIMAGE_PROGRAM_TYPE_BLOCKER",
    "SEMANTIC_PROGRAM_TYPE_BLOCKER",
    "SHARED_ARCHIVE_RECEIVER_CONSUMER",
    "SHARED_ARCHIVE_UNPACK_OPERATION",
    "G17G49SelectedProgramPreplacementProductV1",
    "G17G49SelectedProgramProductBlockerProofV1",
    "G17G49SelectedProgramProductBlockerReceiptV1",
    "G17G49SelectedProgramProductError",
    "G17OwnerSpecificReceiverIncidenceManifestV1",
    "G17OwnerSpecificReceiverIncidenceV1",
    "G17SelectedProgramComponentIncidenceV1",
    "audit_g17_g49_selected_program_product_blockers",
    "refuse_g17_g49_product_archive",
    "selected_program_component_incidences",
    "selected_program_role_incidences",
]
