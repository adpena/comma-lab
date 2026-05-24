# SPDX-License-Identifier: MIT
"""Materializer registry for byte-shaving campaign queue compilation.

The byte-shaving planner is intentionally broader than the current executable
surface. This registry is the fail-closed boundary between planning rows and
queueable local work: every selected operation resolves to either a concrete
adapter or an auditable missing-materializer blocker.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tac.optimization.byte_shaving_campaign import (
    DEFAULT_OPERATION_FAMILIES,
    INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER,
    INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY,
    INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND,
)
from tac.optimization.proxy_candidate_contract import ordered_unique
from tac.packet_compiler.cooperative_receiver_grammars import compiler_hook_rows

REGISTRY_SCHEMA = "byte_shaving_materializer_registry.v1"
BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER = "byte_range_entropy_recode_adapter"
BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_ID = "byte_range_entropy_recode_receiver.v1"
BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_KIND = "archive_charged_byte_range_entropy_recode"
BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND = "byte_range_entropy_recode_v1"
DQS1_DROP_PAIR_MATERIALIZER = "dqs1_pairset_drop_pair_adapter"
DQS1_PAIRSET_TARGET_KIND = "dqs1_pairset_drop_pair"
DQS1_RECEIVER_CONTRACT_ID = "dqs1_pairset_decoderq_receiver.v1"
DQS1_RECEIVER_CONTRACT_KIND = "archive_charged_pairset_runtime_selector"
INVERSE_SCORER_CELL_MATERIALIZER = "inverse_scorer_cell_candidate_adapter"
INVERSE_SCORER_CELL_TARGET_KIND = "inverse_scorer_cell_candidate_v1"
INVERSE_SCORER_CELL_RECEIVER_CONTRACT_ID = "inverse_scorer_cell_receiver.v1"
INVERSE_SCORER_CELL_RECEIVER_CONTRACT_KIND = "inverse_scorer_coordinate_candidate"
INVERSE_SCORER_ACTION_FUNCTIONAL_MATERIALIZER = (
    "inverse_scorer_action_functional_adapter"
)
INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND = (
    "inverse_scorer_action_functional_v1"
)
INVERSE_SCORER_ACTION_FUNCTIONAL_RECEIVER_CONTRACT_ID = (
    "inverse_scorer_action_functional_receiver.v1"
)
INVERSE_SCORER_ACTION_FUNCTIONAL_RECEIVER_CONTRACT_KIND = (
    "planning_only_inverse_scorer_action_functional"
)
INVERSE_ACTION_HIGH_LEVEL_RECEIVER_CONTRACT_ID = (
    "inverse_steganalysis_high_level_operation_set.receiver.v1"
)
INVERSE_ACTION_HIGH_LEVEL_RECEIVER_CONTRACT_KIND = (
    "portfolio_level_inverse_steganalysis_operation_set"
)
ARCHIVE_SECTION_ENTROPY_RECODE_MATERIALIZER = "archive_section_entropy_recode_adapter"
ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND = "archive_section_entropy_recode_v1"
ARCHIVE_SECTION_HEADER_ELIDE_MATERIALIZER = "archive_section_header_elide_adapter"
ARCHIVE_SECTION_HEADER_ELIDE_TARGET_KIND = "archive_section_header_elide_v1"
ARCHIVE_SECTION_REORDER_MATERIALIZER = "archive_section_reorder_adapter"
ARCHIVE_SECTION_REORDER_TARGET_KIND = "archive_section_reorder_v1"
ARCHIVE_SECTION_PROCEDURALIZE_MATERIALIZER = "archive_section_proceduralize_adapter"
ARCHIVE_SECTION_PROCEDURALIZE_TARGET_KIND = "archive_section_proceduralize_v1"
TENSOR_QUANTIZE_MATERIALIZER = "tensor_quantize_adapter"
TENSOR_QUANTIZE_TARGET_KIND = "tensor_quantize_v1"
TENSOR_PRUNE_MATERIALIZER = "tensor_prune_adapter"
TENSOR_PRUNE_TARGET_KIND = "tensor_prune_v1"
TENSOR_FACTORIZE_MATERIALIZER = "tensor_factorize_adapter"
TENSOR_FACTORIZE_TARGET_KIND = "tensor_factorize_v1"
TENSOR_SHARED_CODEBOOK_MATERIALIZER = "tensor_shared_codebook_adapter"
TENSOR_SHARED_CODEBOOK_TARGET_KIND = "tensor_shared_codebook_v1"
PACKET_MEMBER_ZIP_HEADER_ELIDE_MATERIALIZER = "packet_member_zip_header_elide_adapter"
PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND = "packet_member_zip_header_elide_v1"
PACKET_MEMBER_RECOMPRESS_MATERIALIZER = "packet_member_recompress_adapter"
PACKET_MEMBER_RECOMPRESS_TARGET_KIND = "packet_member_recompress_v1"
PACKET_MEMBER_REORDER_MATERIALIZER = "packet_member_reorder_adapter"
PACKET_MEMBER_REORDER_TARGET_KIND = "packet_member_reorder_v1"
PACKET_MEMBER_MERGE_MATERIALIZER = "packet_member_merge_adapter"
PACKET_MEMBER_MERGE_TARGET_KIND = "packet_member_merge_v1"


@dataclass(frozen=True)
class MaterializerAdapter:
    """A concrete operation materializer known to the queue compiler."""

    materializer_id: str
    unit_kind: str
    operation_family: str
    target_kind: str
    executable: bool
    description: str
    receiver_contract_id: str
    receiver_contract_kind: str
    cooperative_receiver_required: bool
    materialization_resource_kind: str
    required_context_fields: tuple[str, ...] = ()
    implementation_module: str = ""
    plan_function: str = ""
    materialize_function: str = ""
    receiver_proof_function: str = ""
    receiver_verify_function: str = ""
    emits_candidate_archive: bool = True
    planning_only: bool = False


@dataclass(frozen=True)
class MaterializerResolution:
    """Resolution of one selected byte-shaving operation against the registry."""

    unit_id: str
    unit_kind: str
    operation_id: str
    operation_family: str
    explicit_materializer: str | None
    materializer_id: str | None
    target_kind: str | None
    receiver_contract_id: str | None
    receiver_contract_kind: str | None
    cooperative_receiver_required: bool
    materialization_resource_kind: str | None
    executable: bool
    blockers: tuple[str, ...]
    adapter: MaterializerAdapter | None = None


def _non_executable_family_adapter(
    *,
    materializer_id: str,
    unit_kind: str,
    operation_family: str,
    target_kind: str,
    description: str,
    receiver_contract_kind: str,
    required_context_fields: tuple[str, ...],
) -> MaterializerAdapter:
    return MaterializerAdapter(
        materializer_id=materializer_id,
        unit_kind=unit_kind,
        operation_family=operation_family,
        target_kind=target_kind,
        executable=False,
        description=description,
        receiver_contract_id=f"{target_kind}.receiver.v1",
        receiver_contract_kind=receiver_contract_kind,
        cooperative_receiver_required=True,
        materialization_resource_kind="local_cpu",
        required_context_fields=required_context_fields,
    )


_ADAPTERS: tuple[MaterializerAdapter, ...] = (
    MaterializerAdapter(
        materializer_id=BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER,
        unit_kind="byte_range",
        operation_family="entropy_recode",
        target_kind=BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND,
        executable=False,
        description=(
            "Fail-closed contract for byte-range entropy recode work; requires "
            "archive-member mapping and runtime-consumption proof before queue execution."
        ),
        receiver_contract_id=BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_ID,
        receiver_contract_kind=BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_KIND,
        cooperative_receiver_required=True,
        materialization_resource_kind="local_cpu",
        required_context_fields=(
            "archive_member_name",
            "archive_byte_range",
            "runtime_consumption_proof",
        ),
        implementation_module=(
            "tac.optimization.byte_range_entropy_recode_materializer"
        ),
        plan_function="build_byte_range_entropy_recode_plan",
        materialize_function="materialize_byte_range_entropy_recode_candidate",
        receiver_proof_function="build_byte_range_entropy_recode_receiver_proof",
        receiver_verify_function=(
            "verify_byte_range_entropy_recode_receiver_contract"
        ),
    ),
    MaterializerAdapter(
        materializer_id=DQS1_DROP_PAIR_MATERIALIZER,
        unit_kind="pair",
        operation_family="drop_pair",
        target_kind=DQS1_PAIRSET_TARGET_KIND,
        executable=True,
        description="Compile pair-unit drop operations into DQS1 pairset local-first queue rows.",
        receiver_contract_id=DQS1_RECEIVER_CONTRACT_ID,
        receiver_contract_kind=DQS1_RECEIVER_CONTRACT_KIND,
        cooperative_receiver_required=True,
        materialization_resource_kind="local_cpu",
        required_context_fields=("dqs1_base_pair_indices",),
        implementation_module="comma_lab.scheduler.byte_shaving_campaign_queue",
    ),
    MaterializerAdapter(
        materializer_id=INVERSE_SCORER_CELL_MATERIALIZER,
        unit_kind="scorer_inverse_surface_cell",
        operation_family="materialize_inverse_scorer_cell_candidate",
        target_kind=INVERSE_SCORER_CELL_TARGET_KIND,
        executable=True,
        description=(
            "Executable inverse-scorer coordinate-cell proof chain; exact-mode "
            "queue execution requires inflate parity context before it can emit "
            "a harvestable candidate chain."
        ),
        receiver_contract_id=INVERSE_SCORER_CELL_RECEIVER_CONTRACT_ID,
        receiver_contract_kind=INVERSE_SCORER_CELL_RECEIVER_CONTRACT_KIND,
        cooperative_receiver_required=True,
        materialization_resource_kind="local_mlx",
        required_context_fields=(
            "raw_contest_video_digest",
            "candidate_archive_template",
            "inverse_action_functional",
            "output_dir",
            "inflate_runtime_dir_or_source_and_candidate_inflate_output_dirs",
        ),
        implementation_module="tac.optimization.inverse_scorer_cell_materializer",
        plan_function="build_inverse_scorer_cell_candidate_plan",
        materialize_function="materialize_inverse_scorer_cell_candidate",
        receiver_proof_function="build_inverse_scorer_cell_receiver_proof",
        receiver_verify_function="verify_inverse_scorer_cell_receiver_contract",
    ),
    MaterializerAdapter(
        materializer_id=INVERSE_SCORER_ACTION_FUNCTIONAL_MATERIALIZER,
        unit_kind="scorer_inverse_surface_cell",
        operation_family="probe_inverse_scorer_surface_cell",
        target_kind=INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND,
        executable=True,
        description=(
            "Compile inverse-scorer cells into a local planning-only discrete "
            "action functional artifact. This proof-chain probe is not a "
            "candidate archive materializer."
        ),
        receiver_contract_id=INVERSE_SCORER_ACTION_FUNCTIONAL_RECEIVER_CONTRACT_ID,
        receiver_contract_kind=(
            INVERSE_SCORER_ACTION_FUNCTIONAL_RECEIVER_CONTRACT_KIND
        ),
        cooperative_receiver_required=False,
        materialization_resource_kind="local_cpu",
        required_context_fields=(
            "output",
            "inverse_action_source_surface",
        ),
        implementation_module="comma_lab.scheduler.byte_shaving_campaign_queue",
        plan_function="build_inverse_steganalysis_action_functional",
        emits_candidate_archive=False,
        planning_only=True,
    ),
    MaterializerAdapter(
        materializer_id=INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER,
        unit_kind="scorer_inverse_surface_cell",
        operation_family=INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY,
        target_kind=INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND,
        executable=False,
        description=(
            "Fail-closed contract for promoting bare inverse-action cells into "
            "portfolio-level archive/runtime operation sets. Bare cells are not "
            "candidate archives until this compiler maps them to a concrete "
            "family materializer with runtime-consumption proof."
        ),
        receiver_contract_id=INVERSE_ACTION_HIGH_LEVEL_RECEIVER_CONTRACT_ID,
        receiver_contract_kind=INVERSE_ACTION_HIGH_LEVEL_RECEIVER_CONTRACT_KIND,
        cooperative_receiver_required=True,
        materialization_resource_kind="local_mlx",
        required_context_fields=(
            "candidate_family",
            "archive_grammar",
            "receiver_contract_kind",
            "operation_set_compiler",
            "runtime_consumption_proof",
        ),
        implementation_module="",
        emits_candidate_archive=False,
        planning_only=True,
    ),
    MaterializerAdapter(
        materializer_id=ARCHIVE_SECTION_ENTROPY_RECODE_MATERIALIZER,
        unit_kind="archive_section",
        operation_family="section_entropy_recode",
        target_kind=ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
        executable=True,
        description=(
            "Family-agnostic HNeRV/BoostNeRV/non-NeRV archive-section entropy "
            "recode materializer. Runtime-consumption proof is required before "
            "exact-readiness, not before local candidate emission."
        ),
        receiver_contract_id=f"{ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND}.receiver.v1",
        receiver_contract_kind="family_agnostic_archive_section_entropy_recode",
        cooperative_receiver_required=True,
        materialization_resource_kind="local_cpu",
        required_context_fields=(
            "archive_path",
            "section_manifest",
            "output_archive",
            "output_manifest",
        ),
        implementation_module="tac.optimization.family_agnostic_materializers",
        materialize_function="materialize_archive_section_entropy_recode_candidate",
    ),
    _non_executable_family_adapter(
        materializer_id=ARCHIVE_SECTION_HEADER_ELIDE_MATERIALIZER,
        unit_kind="archive_section",
        operation_family="section_header_elide",
        target_kind=ARCHIVE_SECTION_HEADER_ELIDE_TARGET_KIND,
        description=(
            "Fail-closed contract for archive-section header elision across "
            "HNeRV/BoostNeRV/NeRV and non-NeRV payload grammars."
        ),
        receiver_contract_kind="family_agnostic_archive_section_header_elide",
        required_context_fields=(
            "archive_path",
            "section_manifest",
            "header_elision_contract",
            "runtime_consumption_proof",
        ),
    ),
    _non_executable_family_adapter(
        materializer_id=ARCHIVE_SECTION_REORDER_MATERIALIZER,
        unit_kind="archive_section",
        operation_family="section_reorder",
        target_kind=ARCHIVE_SECTION_REORDER_TARGET_KIND,
        description=(
            "Fail-closed contract for deterministic archive-section reordering "
            "when the receiver grammar proves order independence or remapping."
        ),
        receiver_contract_kind="family_agnostic_archive_section_reorder",
        required_context_fields=(
            "archive_path",
            "section_manifest",
            "section_order_contract",
            "runtime_consumption_proof",
        ),
    ),
    _non_executable_family_adapter(
        materializer_id=ARCHIVE_SECTION_PROCEDURALIZE_MATERIALIZER,
        unit_kind="archive_section",
        operation_family="section_proceduralize",
        target_kind=ARCHIVE_SECTION_PROCEDURALIZE_TARGET_KIND,
        description=(
            "Fail-closed contract for replacing reusable archive sections with "
            "deterministic receiver-side procedures across NeRV-family and "
            "non-NeRV payload grammars."
        ),
        receiver_contract_kind="family_agnostic_archive_section_proceduralize",
        required_context_fields=(
            "archive_path",
            "section_manifest",
            "procedural_receiver_spec",
            "runtime_consumption_proof",
        ),
    ),
    _non_executable_family_adapter(
        materializer_id=TENSOR_QUANTIZE_MATERIALIZER,
        unit_kind="tensor",
        operation_family="quantize_tensor",
        target_kind=TENSOR_QUANTIZE_TARGET_KIND,
        description=(
            "Fail-closed contract for representation tensor quantization in "
            "HNeRV, BoostNeRV, broader NeRV-family, and non-NeRV learned codecs."
        ),
        receiver_contract_kind="family_agnostic_tensor_quantize",
        required_context_fields=(
            "archive_path",
            "tensor_manifest",
            "quantization_contract",
            "runtime_consumption_proof",
        ),
    ),
    _non_executable_family_adapter(
        materializer_id=TENSOR_PRUNE_MATERIALIZER,
        unit_kind="tensor",
        operation_family="prune_tensor",
        target_kind=TENSOR_PRUNE_TARGET_KIND,
        description=(
            "Fail-closed contract for pruning inactive representation tensors "
            "across NeRV-family, BoostNeRV bolt-ons, and non-NeRV learned codecs."
        ),
        receiver_contract_kind="family_agnostic_tensor_prune",
        required_context_fields=(
            "archive_path",
            "tensor_manifest",
            "pruning_contract",
            "runtime_consumption_proof",
        ),
    ),
    MaterializerAdapter(
        materializer_id=TENSOR_FACTORIZE_MATERIALIZER,
        unit_kind="tensor",
        operation_family="factorize_tensor",
        target_kind=TENSOR_FACTORIZE_TARGET_KIND,
        executable=True,
        description=(
            "Family-agnostic tensor factorization candidate materializer for "
            "BoostNeRV bolt-on weights and generic representation tensors. "
            "Cooperative receiver proof is required before exact-readiness."
        ),
        receiver_contract_id=f"{TENSOR_FACTORIZE_TARGET_KIND}.receiver.v1",
        receiver_contract_kind="family_agnostic_tensor_factorize",
        cooperative_receiver_required=True,
        materialization_resource_kind="local_cpu",
        required_context_fields=(
            "archive_path",
            "tensor_manifest",
            "factorization_contract",
            "output_archive",
            "output_manifest",
        ),
        implementation_module="tac.optimization.family_agnostic_materializers",
        materialize_function="materialize_tensor_factorize_candidate",
    ),
    _non_executable_family_adapter(
        materializer_id=TENSOR_SHARED_CODEBOOK_MATERIALIZER,
        unit_kind="tensor",
        operation_family="shared_codebook_tensor",
        target_kind=TENSOR_SHARED_CODEBOOK_TARGET_KIND,
        description=(
            "Fail-closed contract for shared-codebook tensor payload rewrites "
            "across representation families."
        ),
        receiver_contract_kind="family_agnostic_tensor_shared_codebook",
        required_context_fields=(
            "archive_path",
            "tensor_manifest",
            "codebook_contract",
            "runtime_consumption_proof",
        ),
    ),
    _non_executable_family_adapter(
        materializer_id=PACKET_MEMBER_ZIP_HEADER_ELIDE_MATERIALIZER,
        unit_kind="packet_member",
        operation_family="zip_header_elide",
        target_kind=PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
        description=(
            "Fail-closed contract for ZIP/member header elision when the "
            "contest runtime can reconstruct deterministic header state."
        ),
        receiver_contract_kind="family_agnostic_packet_member_zip_header_elide",
        required_context_fields=(
            "archive_path",
            "packet_member_manifest",
            "zip_header_contract",
            "runtime_consumption_proof",
        ),
    ),
    MaterializerAdapter(
        materializer_id=PACKET_MEMBER_RECOMPRESS_MATERIALIZER,
        unit_kind="packet_member",
        operation_family="member_recompress",
        target_kind=PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
        executable=True,
        description=(
            "Family-agnostic packet-member recompression materializer for "
            "monolithic HNeRV packets, bolted-on side members, and non-NeRV "
            "archives. It preserves member payload bytes and still requires "
            "runtime-consumption proof before exact-readiness."
        ),
        receiver_contract_id=f"{PACKET_MEMBER_RECOMPRESS_TARGET_KIND}.receiver.v1",
        receiver_contract_kind="family_agnostic_packet_member_recompress",
        cooperative_receiver_required=True,
        materialization_resource_kind="local_cpu",
        required_context_fields=(
            "archive_path",
            "output_archive",
            "output_manifest",
        ),
        implementation_module="tac.optimization.family_agnostic_materializers",
        materialize_function="materialize_packet_member_recompress_candidate",
    ),
    _non_executable_family_adapter(
        materializer_id=PACKET_MEMBER_REORDER_MATERIALIZER,
        unit_kind="packet_member",
        operation_family="member_reorder",
        target_kind=PACKET_MEMBER_REORDER_TARGET_KIND,
        description=(
            "Fail-closed contract for packet-member ordering changes guarded by "
            "receiver-side member lookup proof."
        ),
        receiver_contract_kind="family_agnostic_packet_member_reorder",
        required_context_fields=(
            "archive_path",
            "packet_member_manifest",
            "member_order_contract",
            "runtime_consumption_proof",
        ),
    ),
    _non_executable_family_adapter(
        materializer_id=PACKET_MEMBER_MERGE_MATERIALIZER,
        unit_kind="packet_member",
        operation_family="member_merge",
        target_kind=PACKET_MEMBER_MERGE_TARGET_KIND,
        description=(
            "Fail-closed contract for merging side members or bolted-on payload "
            "members into a deterministic receiver-visible packet member."
        ),
        receiver_contract_kind="family_agnostic_packet_member_merge",
        required_context_fields=(
            "archive_path",
            "packet_member_manifest",
            "member_merge_contract",
            "runtime_consumption_proof",
        ),
    ),
)

_ADAPTERS_BY_TARGET_KEY: dict[tuple[str, str, str], MaterializerAdapter] = {
    (adapter.target_kind, adapter.unit_kind, adapter.operation_family): adapter
    for adapter in _ADAPTERS
}
_ADAPTERS_BY_ID: dict[str, MaterializerAdapter] = {
    adapter.materializer_id: adapter
    for adapter in _ADAPTERS
}
_ADAPTERS_BY_UNIT_FAMILY: dict[tuple[str, str], tuple[MaterializerAdapter, ...]] = {
    (unit_kind, operation_family): tuple(
        adapter
        for adapter in _ADAPTERS
        if adapter.unit_kind == unit_kind
        and adapter.operation_family == operation_family
    )
    for unit_kind, operation_family in {
        (adapter.unit_kind, adapter.operation_family)
        for adapter in _ADAPTERS
    }
}
_KNOWN_TARGET_KINDS: frozenset[str] = frozenset(
    adapter.target_kind for adapter in _ADAPTERS
)
KNOWN_OPERATION_FAMILIES: frozenset[str] = frozenset(
    family
    for families in DEFAULT_OPERATION_FAMILIES.values()
    for family in families
)


def _nonempty_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _operation_materializer(operation: Mapping[str, Any]) -> str | None:
    value = _nonempty_str(operation.get("materializer"))
    return value or None


def _operation_target_kind(operation: Mapping[str, Any]) -> str | None:
    value = _nonempty_str(operation.get("target_kind"))
    if value:
        return value
    params = operation.get("params")
    if isinstance(params, Mapping):
        value = _nonempty_str(
            params.get("target_kind") or params.get("materializer_target_kind")
        )
        if value:
            return value
    return None


def resolve_materializer(
    *,
    operation: Mapping[str, Any],
    unit: Mapping[str, Any] | None,
) -> MaterializerResolution:
    """Resolve a selected operation into a concrete adapter or blockers."""

    unit_id = _nonempty_str(operation.get("unit_id"))
    operation_id = _nonempty_str(operation.get("operation_id"))
    operation_family = _nonempty_str(operation.get("operation_family"))
    explicit_materializer = _operation_materializer(operation)
    explicit_target_kind = _operation_target_kind(operation)
    blockers: list[str] = []

    if unit is None:
        unit_kind = ""
        blockers.append(f"selected_unit_missing_from_ranked_units:{unit_id or '<missing>'}")
    else:
        unit_kind = _nonempty_str(unit.get("unit_kind") or unit.get("kind"))
        if not unit_kind:
            blockers.append(f"unit_kind_missing:{unit_id or '<missing>'}")

    adapter: MaterializerAdapter | None = None
    if explicit_materializer is not None:
        adapter = _ADAPTERS_BY_ID.get(explicit_materializer)
        if adapter is None:
            blockers.append(f"materializer_not_registered:{explicit_materializer}")
    elif explicit_target_kind and unit_kind and operation_family:
        adapter = _ADAPTERS_BY_TARGET_KEY.get(
            (explicit_target_kind, unit_kind, operation_family)
        )
        if adapter is None:
            blockers.append(
                f"materializer_not_registered:{explicit_target_kind}:"
                f"{unit_kind}:{operation_family}"
            )
    elif unit_kind and operation_family:
        blockers.append(
            f"materializer_target_kind_required:{unit_kind}:{operation_family}"
        )

    if not operation_family:
        blockers.append(f"operation_family_missing:{unit_id or operation_id or '<missing>'}")
    elif operation_family not in KNOWN_OPERATION_FAMILIES:
        blockers.append(f"unknown_operation_family:{operation_family}")
    elif adapter is None and explicit_materializer is not None:
        blockers.append(
            f"materializer_not_registered:{unit_kind or '<missing>'}:{operation_family}"
        )

    if adapter is not None:
        if explicit_target_kind and explicit_target_kind != adapter.target_kind:
            blockers.append(
                f"materializer_target_kind_mismatch:{adapter.materializer_id}:"
                f"{explicit_target_kind}:expected_{adapter.target_kind}"
            )
        if unit_kind and unit_kind != adapter.unit_kind:
            blockers.append(
                f"materializer_unit_kind_mismatch:{adapter.materializer_id}:"
                f"{unit_id or '<missing>'}:{unit_kind}:expected_{adapter.unit_kind}"
            )
        if operation_family and operation_family != adapter.operation_family:
            blockers.append(
                f"materializer_operation_family_mismatch:{adapter.materializer_id}:"
                f"{operation_family}:expected_{adapter.operation_family}"
            )
        if not adapter.executable:
            blockers.append(f"materializer_not_executable:{adapter.materializer_id}")
        if adapter.planning_only or not adapter.emits_candidate_archive:
            blockers.append(
                f"planning_only_materializer_not_candidate_archive:{adapter.materializer_id}"
            )

    blockers = ordered_unique(blockers)
    return MaterializerResolution(
        unit_id=unit_id,
        unit_kind=unit_kind,
        operation_id=operation_id,
        operation_family=operation_family,
        explicit_materializer=explicit_materializer,
        materializer_id=adapter.materializer_id if adapter is not None else explicit_materializer,
        target_kind=adapter.target_kind if adapter is not None else explicit_target_kind,
        receiver_contract_id=(
            adapter.receiver_contract_id if adapter is not None else None
        ),
        receiver_contract_kind=(
            adapter.receiver_contract_kind if adapter is not None else None
        ),
        cooperative_receiver_required=(
            bool(adapter.cooperative_receiver_required) if adapter is not None else False
        ),
        materialization_resource_kind=(
            adapter.materialization_resource_kind if adapter is not None else None
        ),
        executable=(
            adapter is not None
            and adapter.executable
            and adapter.emits_candidate_archive
            and not adapter.planning_only
            and not blockers
        ),
        blockers=tuple(blockers),
        adapter=adapter,
    )


def suggest_materializer_adapters(
    *,
    unit_kind: str,
    operation_family: str,
) -> tuple[MaterializerAdapter, ...]:
    """Return registered adapters matching a unit/family pair.

    Suggestions are not resolution authority. They exist so backlog artifacts
    can say which receiver/materializer contract to build next while still
    requiring an explicit target/materializer before queue execution.
    """

    return _ADAPTERS_BY_UNIT_FAMILY.get(
        (_nonempty_str(unit_kind), _nonempty_str(operation_family)),
        (),
    )


def known_materializer_target_kinds() -> frozenset[str]:
    """Return every target kind registered with the materializer boundary."""

    return _KNOWN_TARGET_KINDS


def registry_manifest() -> dict[str, Any]:
    """Return a machine-readable registry view for tests and runbooks."""

    cooperative_receiver_hooks = compiler_hook_rows()
    return {
        "schema": REGISTRY_SCHEMA,
        "adapters": [
            {
                "materializer_id": adapter.materializer_id,
                "unit_kind": adapter.unit_kind,
                "operation_family": adapter.operation_family,
                "target_kind": adapter.target_kind,
                "executable": adapter.executable,
                "emits_candidate_archive": adapter.emits_candidate_archive,
                "planning_only": adapter.planning_only,
                "receiver_contract_id": adapter.receiver_contract_id,
                "receiver_contract_kind": adapter.receiver_contract_kind,
                "cooperative_receiver_required": adapter.cooperative_receiver_required,
                "materialization_resource_kind": adapter.materialization_resource_kind,
                "required_context_fields": list(adapter.required_context_fields),
                "implementation_module": adapter.implementation_module,
                "plan_function": adapter.plan_function,
                "materialize_function": adapter.materialize_function,
                "receiver_proof_function": adapter.receiver_proof_function,
                "receiver_verify_function": adapter.receiver_verify_function,
                "description": adapter.description,
            }
            for adapter in _ADAPTERS
        ],
        "known_target_kinds": sorted(_KNOWN_TARGET_KINDS),
        "cooperative_receiver_grammar_registry": {
            "schema": "cooperative_receiver_packet_grammar_registry_hook.v1",
            "known_grammar_count": len(cooperative_receiver_hooks),
            "compiler_hook_rows": cooperative_receiver_hooks,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "known_operation_families": sorted(KNOWN_OPERATION_FAMILIES),
    }


__all__ = [
    "ARCHIVE_SECTION_ENTROPY_RECODE_MATERIALIZER",
    "ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND",
    "ARCHIVE_SECTION_HEADER_ELIDE_MATERIALIZER",
    "ARCHIVE_SECTION_HEADER_ELIDE_TARGET_KIND",
    "ARCHIVE_SECTION_PROCEDURALIZE_MATERIALIZER",
    "ARCHIVE_SECTION_PROCEDURALIZE_TARGET_KIND",
    "ARCHIVE_SECTION_REORDER_MATERIALIZER",
    "ARCHIVE_SECTION_REORDER_TARGET_KIND",
    "BYTE_RANGE_ENTROPY_RECODE_MATERIALIZER",
    "BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_ID",
    "BYTE_RANGE_ENTROPY_RECODE_RECEIVER_CONTRACT_KIND",
    "BYTE_RANGE_ENTROPY_RECODE_TARGET_KIND",
    "DQS1_DROP_PAIR_MATERIALIZER",
    "DQS1_PAIRSET_TARGET_KIND",
    "DQS1_RECEIVER_CONTRACT_ID",
    "DQS1_RECEIVER_CONTRACT_KIND",
    "INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER",
    "INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY",
    "INVERSE_ACTION_HIGH_LEVEL_RECEIVER_CONTRACT_ID",
    "INVERSE_ACTION_HIGH_LEVEL_RECEIVER_CONTRACT_KIND",
    "INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND",
    "INVERSE_SCORER_ACTION_FUNCTIONAL_MATERIALIZER",
    "INVERSE_SCORER_ACTION_FUNCTIONAL_RECEIVER_CONTRACT_ID",
    "INVERSE_SCORER_ACTION_FUNCTIONAL_RECEIVER_CONTRACT_KIND",
    "INVERSE_SCORER_ACTION_FUNCTIONAL_TARGET_KIND",
    "INVERSE_SCORER_CELL_MATERIALIZER",
    "INVERSE_SCORER_CELL_RECEIVER_CONTRACT_ID",
    "INVERSE_SCORER_CELL_RECEIVER_CONTRACT_KIND",
    "INVERSE_SCORER_CELL_TARGET_KIND",
    "KNOWN_OPERATION_FAMILIES",
    "PACKET_MEMBER_MERGE_MATERIALIZER",
    "PACKET_MEMBER_MERGE_TARGET_KIND",
    "PACKET_MEMBER_RECOMPRESS_MATERIALIZER",
    "PACKET_MEMBER_RECOMPRESS_TARGET_KIND",
    "PACKET_MEMBER_REORDER_MATERIALIZER",
    "PACKET_MEMBER_REORDER_TARGET_KIND",
    "PACKET_MEMBER_ZIP_HEADER_ELIDE_MATERIALIZER",
    "PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND",
    "REGISTRY_SCHEMA",
    "TENSOR_FACTORIZE_MATERIALIZER",
    "TENSOR_FACTORIZE_TARGET_KIND",
    "TENSOR_PRUNE_MATERIALIZER",
    "TENSOR_PRUNE_TARGET_KIND",
    "TENSOR_QUANTIZE_MATERIALIZER",
    "TENSOR_QUANTIZE_TARGET_KIND",
    "TENSOR_SHARED_CODEBOOK_MATERIALIZER",
    "TENSOR_SHARED_CODEBOOK_TARGET_KIND",
    "MaterializerAdapter",
    "MaterializerResolution",
    "known_materializer_target_kinds",
    "registry_manifest",
    "resolve_materializer",
    "suggest_materializer_adapters",
]
