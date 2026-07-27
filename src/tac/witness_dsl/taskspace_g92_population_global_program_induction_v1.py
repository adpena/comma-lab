# SPDX-License-Identifier: MIT
"""Population-global induction over sealed G90 V1 and V2 observation atlases.

The V1 path remains a partial atlas: its screening/Pareto policy leaves
discarded projections unresolved.  The V2 path strictly reopens every
deterministic physical group in the exact-all coarse atlas, including batches
whose physical group count exceeds the common eight-coordinate case.  Both
paths preserve the exact proposed atoms and derive only genuinely shared
physical-family identities plus collision-free storage branches.

Neither path composes isolated interventions, proves cumulative optimality, or
infers archive rate from operand member bytes.  G90 observations are anchored
at one incumbent operating point, so their component deltas are not
transferable to a cumulative G89/G94 state.  This module emits and prices no
archive; exact composed-state replay and full-n600 rows remain downstream.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.optimization.direct_description_carrier_compose import (
    BoundaryShearletAtomV1,
)
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    RoleAwareBoundaryShearletOperandV1,
)
from tac.witness_dsl.taskspace_g89_class_complete_semantic_compiler_v1 import (
    G83_DECODER_TRANSITION_ID,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    SelectedPreimageFrameSelectorV1,
)

PAIR_COUNT: Final = 600
SOURCE_VIDEO_BYTES: Final = 37_545_489
G90_AGGREGATE_SCHEMA: Final = "tac.taskspace_projected_population_costate_aggregate.v1"
G90_STAGE_SCHEMA: Final = "tac.taskspace_projected_population_costate_stage.v1"
G90_BATCH_SCHEMA: Final = "tac.taskspace_projected_population_costate_batch.v1"
G90_V2_AGGREGATE_SCHEMA: Final = "tac.taskspace_exact_coarse_costate_aggregate.v2"
G90_V2_STAGE_SCHEMA: Final = "tac.taskspace_exact_coarse_costate_stage.v2"
G90_V2_BATCH_SCHEMA: Final = "tac.taskspace_exact_coarse_costate_batch.v2"
G90_V2_EXACT_REPLAY_POLICY: Final = "ALL_DETERMINISTIC_PHYSICAL_GROUPS"
G92_FAMILY_SCHEMA: Final = "tac.taskspace_g92_population_global_prefix_family.v1"
G92_MEASUREMENT_SCHEMA: Final = "tac.taskspace_g92_exact_prefix_measurement.v1"
G92_SELECTION_CONTRACT: Final = "NO_PROXY_RANKING_REQUIRE_SAME_ARCHIVE_FULL_N600_REALIZED_THROUGH_R_PUBLIC_CLOSURE"
G51_PAYLOAD_POLICY: Final = "OPAQUE_EXACT_RECEIPT_PROVENANCE_NOT_PARSED_NO_PAYLOAD_AUTHORITY"
EXACT_LOWERING_CHAIN_BLOCKER: Final = (
    "G90_V2_EXACT_ALL_COARSE_ATLAS_PLUS_G92_TO_G94_LOWERING_PLUS_SAME_STATE_FULL_N600_ROWS_OWED"
)
V2_COMPOSED_LOWERING_BLOCKER: Final = "G92_TO_G94_EXACT_COMPOSED_STATE_REPLAY_PLUS_SAME_STATE_FULL_N600_ROWS_OWED"
# Compatibility name for callers of V1.  Its value is the current blocker,
# not the now-implemented G94 receiver seam.
SEQUENTIAL_LOWERING_BLOCKER: Final = EXACT_LOWERING_CHAIN_BLOCKER

_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_GROUP_RE: Final = re.compile(
    r"g72:(?P<start>[0-9]{4})_(?P<stop>[0-9]{4}):"
    r"(?P<role>Road|UndrivableBoundary):d(?P<direction>[01]):"
    r"a(?P<amplitude>0\.5|1):p(?P<partition>[0-9]+)\Z"
)
_ROLE_WIRE: Final = {"UndrivableBoundary": 0, "Road": 1}
_AGGREGATE_KEYS: Final = {
    "aggregate_receipt_sha256",
    "base_row",
    "candidate_claim",
    "dense_costates_persisted",
    "encoder_only",
    "pair_range",
    "pointer_moved",
    "projection_coordinate_count",
    "rate_axis",
    "research_only",
    "schema",
    "score_claim",
    "selection_consumer",
    "stages",
    "tangent_families",
}
_STAGE_KEYS: Final = {
    "base_pose_squared_error_sum_f32",
    "base_segmentation_error_count",
    "batch_count",
    "batches",
    "candidate_claim",
    "checkpoint_policy",
    "dense_costates_persisted",
    "encoder_only",
    "g78_stage_receipt_sha256",
    "g87_stage_checkpoint_sha256",
    "pair_range",
    "pointer_moved",
    "projection_coordinate_count",
    "research_only",
    "schema",
    "score_claim",
    "stage_index",
    "stage_receipt_sha256",
}
_BATCH_KEYS: Final = {
    "actual_zip_delta_measured",
    "actuator_basis_groups",
    "actuator_basis_reconstructs_exact_measured_proposed_atoms",
    "base_components",
    "batch_checkpoint_sha256",
    "candidate_claim",
    "dense_costates_persisted",
    "encoder_only",
    "incumbent_and_proposed_atom_custody_separate",
    "local_admission_performed",
    "member_bytes_used_as_rate",
    "pair_range",
    "pareto_nondominated_operand_ids",
    "population_pose_pair_mse_vjp_scale",
    "projection_coordinate_count",
    "projection_rows",
    "research_only",
    "schema",
    "score_claim",
    "selection_consumer",
    "source_custody",
}
_ROW_KEYS: Final = {
    "atom_count",
    "changed_camera_values",
    "exact_pose_mean_delta",
    "exact_pose_score_delta",
    "exact_seg_mismatch_delta",
    "exact_seg_score_delta",
    "exact_zip_delta_bytes",
    "family_id",
    "operand_id",
    "operand_member_bytes",
    "operand_sha256",
    "pair_ids",
    "pose_linearized_score_delta",
    "proposed_atoms_sha256",
    "incumbent_atoms_sha256",
    "rate_status",
    "seg_gap_directional_delta",
}
_BASIS_GROUP_KEYS: Final = {
    "amplitude_scale",
    "direction_rank",
    "group_id",
    "incumbent_atoms_sha256",
    "proposal_fingerprints",
    "proposals",
    "proposed_atoms_sha256",
    "role",
}
_V2_AGGREGATE_KEYS: Final = {
    "aggregate_receipt_sha256",
    "authority_drift",
    "base_row",
    "candidate_claim",
    "encoder_only",
    "exact_replay_count",
    "exact_replay_policy",
    "hierarchical_refinement_required_before_atom_selection",
    "pair_range",
    "pareto_pruning_performed",
    "pointer_moved",
    "projection_coordinate_count",
    "rate_axis",
    "research_only",
    "schema",
    "score_claim",
    "stages",
}
_V2_STAGE_KEYS: Final = {
    "base_pose_squared_error_sum_f32",
    "base_segmentation_error_count",
    "batch_count",
    "batches",
    "candidate_claim",
    "checkpoint_policy",
    "dense_costates_persisted",
    "differentiable_current_argmax_drift_cells",
    "differentiable_target_argmax_drift_cells",
    "encoder_only",
    "exact_replay_count",
    "exact_replay_policy",
    "g78_stage_receipt_sha256",
    "g87_stage_checkpoint_sha256",
    "pair_range",
    "pareto_pruning_performed",
    "projection_coordinate_count",
    "research_only",
    "schema",
    "score_claim",
    "stage_index",
    "stage_receipt_sha256",
}
_V2_BATCH_KEYS: Final = {
    "actual_zip_delta_measured",
    "actuator_basis_groups",
    "all_deterministic_physical_groups_exact_replayed",
    "authority_drift",
    "base_components",
    "batch_checkpoint_sha256",
    "candidate_claim",
    "dense_costates_persisted",
    "encoder_only",
    "exact_replay_policy",
    "exact_replay_state_custody",
    "expected_physical_group_count",
    "expected_physical_group_ids",
    "local_admission_performed",
    "member_bytes_used_as_rate",
    "pair_range",
    "pareto_pruning_performed",
    "population_pose_pair_mse_vjp_scale",
    "projection_coordinate_count",
    "projection_rows",
    "research_only",
    "schema",
    "score_claim",
    "source_custody",
}
_V2_SOURCE_CUSTODY_KEYS: Final = {
    "candidate_camera_sha256",
    "current_cells_sha256",
    "target_camera_sha256",
    "target_cells_sha256",
}
_V2_BASE_COMPONENT_KEYS: Final = {
    "pair_pose_mse_f32",
    "seg_mismatch_count",
    "target_minus_current_gap_sum",
}
_V2_REPLAY_STATE_KEYS: Final = {
    "candidate_y0_preserved",
    "operand_id",
    "pose_conditioning_y0_sha256",
    "pose_conditioning_y1_sha256",
    "seg_base_y1_sha256",
    "seg_candidate_y1_sha256",
}
_V2_AUTHORITY_DRIFT_KEYS: Final = {
    "authority_cells_drive_exact_replay",
    "authority_pose_targets_and_base_mse_drive_exact_replay",
    "current",
    "differentiable_argmax_has_no_authority",
    "pose",
    "target",
}
_V2_SEG_DRIFT_KEYS: Final = {
    "authority_cells_sha256",
    "axis",
    "differentiable_cells_sha256",
    "expected_cells_sha256",
    "minimum_top_two_margin_at_drift",
    "mismatch_cell_count",
    "mismatch_pair_ids",
}
_V2_POSE_DRIFT_KEYS: Final = {
    "authority_current_pose6_sha256",
    "authority_target_pose6_sha256",
    "differentiable_current_pose6_sha256",
    "differentiable_target_pose6_sha256",
    "maximum_abs_current_delta",
    "maximum_abs_target_delta",
}


class PopulationProgramInductionError(ValueError):
    """A sealed-input, physical program, archive, or authority invariant failed."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(value)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise PopulationProgramInductionError(f"{label} is not canonical SHA-256")
    return value


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise PopulationProgramInductionError(f"{label} cannot be read") from exc
    if type(value) is not dict:
        raise PopulationProgramInductionError(f"{label} is not one JSON object")
    return value


def _verify_seal(value: Mapping[str, Any], *, field: str, label: str) -> str:
    actual = _require_sha(value.get(field), label=f"{label}.{field}")
    body = {key: item for key, item in value.items() if key != field}
    if sha256_bytes(canonical_json_bytes(body)) != actual:
        raise PopulationProgramInductionError(f"{label} self seal differs")
    return actual


@dataclass(frozen=True, slots=True)
class ExactFileIdentityV1:
    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path:
            raise PopulationProgramInductionError("file identity path is empty")
        if type(self.bytes) is not int or self.bytes <= 0:
            raise PopulationProgramInductionError("file identity bytes must be positive")
        _require_sha(self.sha256, label="file identity SHA")

    @classmethod
    def from_mapping(cls, value: object, *, label: str) -> ExactFileIdentityV1:
        if type(value) is not dict or set(value) != {"bytes", "path", "sha256"}:
            raise PopulationProgramInductionError(f"{label} file identity key set differs")
        return cls(path=value["path"], bytes=value["bytes"], sha256=value["sha256"])

    def verify(self, *, label: str) -> Path:
        path = Path(self.path).expanduser().resolve()
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != self.bytes
            or sha256_file(path) != self.sha256
        ):
            raise PopulationProgramInductionError(f"{label} exact bytes/SHA differ")
        return path

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}


def _atom_to_dict(atom: BoundaryShearletAtomV1) -> dict[str, object]:
    return {
        "pair_index": atom.pair_index,
        "role": atom.role,
        "center_y": atom.center_y,
        "center_x": atom.center_x,
        "scale_y": atom.scale_y,
        "scale_x": atom.scale_x,
        "shear_q4": atom.shear_q4,
        "amplitude_q4": atom.amplitude_q4,
    }


def _atom_from_dict(value: object, *, label: str) -> BoundaryShearletAtomV1:
    fields = {
        "amplitude_q4",
        "center_x",
        "center_y",
        "pair_index",
        "role",
        "scale_x",
        "scale_y",
        "shear_q4",
    }
    if type(value) is not dict or set(value) != fields:
        raise PopulationProgramInductionError(f"{label} atom key set differs")
    try:
        return BoundaryShearletAtomV1(**value)
    except (TypeError, ValueError) as exc:
        raise PopulationProgramInductionError(f"{label} atom is invalid") from exc


def _atom_address(atom: BoundaryShearletAtomV1) -> tuple[int, int, int, int]:
    return atom.pair_index, _ROLE_WIRE[atom.role], atom.center_y, atom.center_x


@dataclass(frozen=True, slots=True)
class G90ExactInterventionV1:
    """One isolated exact-replayed G90 intervention with proposal custody."""

    operand_id: str
    pair_ids: tuple[int, ...]
    role: str
    direction_rank: int
    amplitude_scale: str
    partition_index: int
    atoms: tuple[BoundaryShearletAtomV1, ...]
    proposal_atom_fingerprints: tuple[str, ...]
    proposed_atoms_sha256: str
    operand_member_bytes: int
    operand_sha256: str
    changed_camera_values: int
    pose_linearized_score_delta: float
    seg_gap_directional_delta: float
    exact_seg_mismatch_delta: int
    exact_seg_score_delta: float
    exact_pose_mean_delta: float
    exact_pose_score_delta: float
    pareto_nondominated: bool = False

    def __post_init__(self) -> None:
        match = _GROUP_RE.fullmatch(self.operand_id)
        if match is None:
            raise PopulationProgramInductionError("G90 operand ID lost physical group identity")
        expected_ids = tuple(range(int(match["start"]), int(match["stop"])))
        if (
            self.pair_ids != expected_ids
            or self.role != match["role"]
            or self.direction_rank != int(match["direction"])
            or self.amplitude_scale != match["amplitude"]
            or self.partition_index != int(match["partition"])
        ):
            raise PopulationProgramInductionError("G90 group identity fields disagree")
        if not self.atoms or len(self.atoms) != len(self.proposal_atom_fingerprints):
            raise PopulationProgramInductionError("G90 proposal atom custody is empty/incomplete")
        keys = tuple(_atom_address(atom) for atom in self.atoms)
        if keys != tuple(sorted(set(keys))):
            raise PopulationProgramInductionError("G90 intervention atoms collide or are noncanonical")
        if any(atom.pair_index not in set(self.pair_ids) or atom.role != self.role for atom in self.atoms):
            raise PopulationProgramInductionError("G90 intervention atom escaped group address")
        for index, fingerprint in enumerate(self.proposal_atom_fingerprints):
            _require_sha(fingerprint, label=f"proposal fingerprint {index}")
        _require_sha(self.proposed_atoms_sha256, label="proposed atoms SHA")
        proposed_operand = RoleAwareBoundaryShearletOperandV1(
            frame_selector=SelectedPreimageFrameSelectorV1.Y1,
            atoms=self.atoms,
        )
        if proposed_operand.sha256 != self.proposed_atoms_sha256:
            raise PopulationProgramInductionError("G90 proposed operand atoms/SHA differ")
        _require_sha(self.operand_sha256, label="G90 operand SHA")
        if self.operand_sha256 != self.proposed_atoms_sha256 or self.operand_member_bytes != len(
            proposed_operand.to_bytes()
        ):
            raise PopulationProgramInductionError("G90 proposed operand row/basis custody differs")
        if (
            type(self.operand_member_bytes) is not int
            or self.operand_member_bytes <= 0
            or type(self.changed_camera_values) is not int
            or self.changed_camera_values < 0
            or type(self.exact_seg_mismatch_delta) is not int
        ):
            raise PopulationProgramInductionError("G90 exact row integer fields differ")
        for label, value in (
            ("pose linearized delta", self.pose_linearized_score_delta),
            ("Seg directional delta", self.seg_gap_directional_delta),
            ("exact Seg score delta", self.exact_seg_score_delta),
            ("exact pose mean delta", self.exact_pose_mean_delta),
            ("exact pose score delta", self.exact_pose_score_delta),
        ):
            if not math.isfinite(float(value)):
                raise PopulationProgramInductionError(f"{label} is not finite")

    @property
    def physical_family_key(self) -> tuple[str, int, str]:
        return self.role, self.direction_rank, self.amplitude_scale

    @property
    def addresses(self) -> frozenset[tuple[int, int, int, int]]:
        return frozenset(_atom_address(atom) for atom in self.atoms)


@dataclass(frozen=True, slots=True)
class SealedG90PopulationV1:
    aggregate: ExactFileIdentityV1
    aggregate_self_sha256: str
    base_d_seg: float
    base_d_pose: float
    base_archive_bytes: int
    base_archive_sha256: str
    interventions: tuple[G90ExactInterventionV1, ...]
    unresolved_projection_ids: tuple[str, ...]
    source_schema: str = G90_AGGREGATE_SCHEMA
    exact_replay_atlas_complete: bool = False

    def __post_init__(self) -> None:
        if not self.interventions:
            raise PopulationProgramInductionError("sealed G90 has no exact replay interventions")
        ids = tuple(row.operand_id for row in self.interventions)
        if ids != tuple(sorted(set(ids))):
            raise PopulationProgramInductionError("G90 interventions are not unique canonical order")
        if set(ids).intersection(self.unresolved_projection_ids):
            raise PopulationProgramInductionError("G90 resolved/unresolved projections overlap")
        if self.source_schema not in {
            G90_AGGREGATE_SCHEMA,
            G90_V2_AGGREGATE_SCHEMA,
        }:
            raise PopulationProgramInductionError("sealed G90 source schema is unsupported")
        if self.source_schema == G90_AGGREGATE_SCHEMA and self.exact_replay_atlas_complete:
            raise PopulationProgramInductionError("G90 V1 cannot claim an exact-all coarse atlas")
        if self.source_schema == G90_V2_AGGREGATE_SCHEMA and (
            not self.exact_replay_atlas_complete or self.unresolved_projection_ids
        ):
            raise PopulationProgramInductionError("G90 V2 lost exact-all coarse-atlas coverage")


def _parse_basis_group(
    value: object,
) -> tuple[
    str,
    tuple[BoundaryShearletAtomV1, ...],
    tuple[str, ...],
    str,
    str,
]:
    if type(value) is not dict or set(value) != _BASIS_GROUP_KEYS:
        raise PopulationProgramInductionError("G90 actuator basis group key set differs")
    group_id = value.get("group_id")
    match = _GROUP_RE.fullmatch(group_id) if type(group_id) is str else None
    if (
        match is None
        or value.get("role") != match["role"]
        or value.get("direction_rank") != int(match["direction"])
        or value.get("amplitude_scale") != match["amplitude"]
    ):
        raise PopulationProgramInductionError("G90 actuator basis physical identity differs")
    proposals = value.get("proposals")
    fingerprints = value.get("proposal_fingerprints")
    if type(proposals) is not list or type(fingerprints) is not list or len(proposals) != len(fingerprints):
        raise PopulationProgramInductionError("G90 actuator basis proposal custody is incomplete")
    atoms: list[BoundaryShearletAtomV1] = []
    for index, (proposal, fingerprint) in enumerate(zip(proposals, fingerprints, strict=True)):
        if (
            type(proposal) is not dict
            or set(proposal)
            != {
                "atom",
                "candidate_id",
                "fisher_priority",
                "schema",
            }
            or (
                proposal.get("schema") != "tac.g72.boundary_shearlet_proposal.v1"
                or type(proposal.get("candidate_id")) is not str
                or not proposal["candidate_id"]
                or type(proposal.get("fisher_priority")) is not str
                or not proposal["fisher_priority"]
            )
        ):
            raise PopulationProgramInductionError("G90 proposal key set differs")
        expected_fingerprint = _require_sha(
            fingerprint,
            label=f"{group_id}.proposal_fingerprints[{index}]",
        )
        if sha256_bytes(canonical_json_bytes(proposal)) != expected_fingerprint:
            raise PopulationProgramInductionError("G90 proposal fingerprint differs")
        atoms.append(
            _atom_from_dict(
                proposal["atom"],
                label=f"{group_id}.proposals[{index}]",
            )
        )
    proposed_sha = _require_sha(
        value.get("proposed_atoms_sha256"),
        label=f"{group_id}.proposed_atoms_sha256",
    )
    incumbent_sha = _require_sha(
        value.get("incumbent_atoms_sha256"),
        label=f"{group_id}.incumbent_atoms_sha256",
    )
    return group_id, tuple(atoms), tuple(fingerprints), proposed_sha, incumbent_sha


def _parse_exact_intervention(
    row: Mapping[str, Any],
    *,
    pareto_ids: set[str],
    basis_by_id: Mapping[
        str,
        tuple[
            tuple[BoundaryShearletAtomV1, ...],
            tuple[str, ...],
            str,
            str,
        ],
    ],
) -> G90ExactInterventionV1 | None:
    if set(row) != _ROW_KEYS:
        raise PopulationProgramInductionError("G90 projection row key set differs from G92 ABI")
    operand_id = row.get("operand_id")
    if type(operand_id) is not str:
        raise PopulationProgramInductionError("G90 operand ID is not a string")
    exact_fields = (
        "exact_seg_mismatch_delta",
        "exact_seg_score_delta",
        "exact_pose_mean_delta",
        "exact_pose_score_delta",
    )
    exact_present = tuple(row.get(field) is not None for field in exact_fields)
    if not any(exact_present):
        return None
    if not all(exact_present):
        raise PopulationProgramInductionError("G90 row carries a partial exact replay")
    if row.get("exact_zip_delta_bytes") is not None or row.get("rate_status") != (
        "BLOCKED_MEMBER_BYTES_ARE_NOT_A_ZIP_DELTA"
    ):
        raise PopulationProgramInductionError("G90 row falsely promoted member bytes to ZIP rate")
    if row.get("family_id") != "G72_CURRENT_BASE_COMPOSED_ROLE_AWARE_SHEARLET_BATCH_GROUP":
        raise PopulationProgramInductionError("G90 row family identity differs")
    if (
        type(row.get("atom_count")) is not int
        or type(row.get("pose_linearized_score_delta")) not in {int, float}
        or type(row.get("seg_gap_directional_delta")) not in {int, float}
        or type(row.get("exact_seg_score_delta")) not in {int, float}
        or type(row.get("exact_pose_mean_delta")) not in {int, float}
        or type(row.get("exact_pose_score_delta")) not in {int, float}
    ):
        raise PopulationProgramInductionError("G90 exact row numeric ABI differs")
    match = _GROUP_RE.fullmatch(operand_id)
    if match is None:
        raise PopulationProgramInductionError("G90 operand ID is not a physical G72 group")
    basis = basis_by_id.get(operand_id)
    if basis is None:
        raise PopulationProgramInductionError("G90 row lacks exact proposed-atom custody")
    atoms, fingerprints, proposed_sha, incumbent_sha = basis
    if row.get("atom_count") != len(atoms):
        raise PopulationProgramInductionError("G90 atom count differs from persisted proposal atoms")
    if row.get("proposed_atoms_sha256") != proposed_sha or row.get("incumbent_atoms_sha256") != incumbent_sha:
        raise PopulationProgramInductionError("G90 row/basis atom custody differs")
    pair_ids_value = row.get("pair_ids")
    if type(pair_ids_value) is not list or any(type(value) is not int for value in pair_ids_value):
        raise PopulationProgramInductionError("G90 pair IDs changed exact integer ABI")
    return G90ExactInterventionV1(
        operand_id=operand_id,
        pair_ids=tuple(pair_ids_value),
        role=match["role"],
        direction_rank=int(match["direction"]),
        amplitude_scale=match["amplitude"],
        partition_index=int(match["partition"]),
        atoms=atoms,
        proposal_atom_fingerprints=fingerprints,
        proposed_atoms_sha256=proposed_sha,
        operand_member_bytes=row["operand_member_bytes"],
        operand_sha256=row["operand_sha256"],
        changed_camera_values=row["changed_camera_values"],
        pose_linearized_score_delta=float(row["pose_linearized_score_delta"]),
        seg_gap_directional_delta=float(row["seg_gap_directional_delta"]),
        exact_seg_mismatch_delta=row["exact_seg_mismatch_delta"],
        exact_seg_score_delta=float(row["exact_seg_score_delta"]),
        exact_pose_mean_delta=float(row["exact_pose_mean_delta"]),
        exact_pose_score_delta=float(row["exact_pose_score_delta"]),
        pareto_nondominated=operand_id in pareto_ids,
    )


def _validate_v2_authority_drift(
    value: object,
    *,
    pair_ids: tuple[int, ...],
    source_custody: Mapping[str, Any],
) -> tuple[int, int]:
    if type(value) is not dict or set(value) != _V2_AUTHORITY_DRIFT_KEYS:
        raise PopulationProgramInductionError("G90 V2 authority-drift key set differs")
    if (
        value.get("authority_cells_drive_exact_replay") is not True
        or value.get("authority_pose_targets_and_base_mse_drive_exact_replay") is not True
        or value.get("differentiable_argmax_has_no_authority") is not True
    ):
        raise PopulationProgramInductionError("G90 V2 differentiable surface gained authority")
    drift_counts: list[int] = []
    for axis, custody_key in (
        ("current", "current_cells_sha256"),
        ("target", "target_cells_sha256"),
    ):
        row = value.get(axis)
        if type(row) is not dict or set(row) != _V2_SEG_DRIFT_KEYS:
            raise PopulationProgramInductionError(f"G90 V2 {axis} drift key set differs")
        mismatch_count = row.get("mismatch_cell_count")
        mismatch_pair_ids = row.get("mismatch_pair_ids")
        if (
            row.get("axis") != axis
            or type(mismatch_count) is not int
            or mismatch_count < 0
            or type(mismatch_pair_ids) is not list
            or any(type(pair_id) is not int or pair_id not in pair_ids for pair_id in mismatch_pair_ids)
            or mismatch_pair_ids != sorted(set(mismatch_pair_ids))
            or (mismatch_count == 0) != (not mismatch_pair_ids)
            or mismatch_count < len(mismatch_pair_ids)
        ):
            raise PopulationProgramInductionError(f"G90 V2 {axis} drift identity differs")
        expected_sha = _require_sha(
            row.get("expected_cells_sha256"),
            label=f"G90 V2 {axis} expected cells SHA",
        )
        authority_sha = _require_sha(
            row.get("authority_cells_sha256"),
            label=f"G90 V2 {axis} authority cells SHA",
        )
        _require_sha(
            row.get("differentiable_cells_sha256"),
            label=f"G90 V2 {axis} differentiable cells SHA",
        )
        if expected_sha != authority_sha or authority_sha != source_custody.get(custody_key):
            raise PopulationProgramInductionError(f"G90 V2 {axis} authority custody differs")
        margin = row.get("minimum_top_two_margin_at_drift")
        if (mismatch_count == 0 and margin is not None) or (
            mismatch_count > 0
            and (type(margin) not in {int, float} or not math.isfinite(float(margin)) or float(margin) < 0.0)
        ):
            raise PopulationProgramInductionError(f"G90 V2 {axis} tie-drift margin differs")
        drift_counts.append(mismatch_count)
    pose = value.get("pose")
    if type(pose) is not dict or set(pose) != _V2_POSE_DRIFT_KEYS:
        raise PopulationProgramInductionError("G90 V2 pose drift key set differs")
    for key in (
        "authority_current_pose6_sha256",
        "authority_target_pose6_sha256",
        "differentiable_current_pose6_sha256",
        "differentiable_target_pose6_sha256",
    ):
        _require_sha(pose.get(key), label=f"G90 V2 pose drift {key}")
    for key in ("maximum_abs_current_delta", "maximum_abs_target_delta"):
        number = pose.get(key)
        if type(number) not in {int, float} or not math.isfinite(float(number)) or float(number) < 0.0:
            raise PopulationProgramInductionError(f"G90 V2 pose drift {key} differs")
    return drift_counts[0], drift_counts[1]


def _load_sealed_g90_population_v2(
    aggregate_identity: ExactFileIdentityV1,
    *,
    expected_aggregate_self_sha256: str,
) -> SealedG90PopulationV1:
    """Reopen a complete exact-all V2 coarse atlas without composing its rows."""

    aggregate_path = aggregate_identity.verify(label="G90 V2 aggregate")
    aggregate = _load_mapping(aggregate_path, label="G90 V2 aggregate")
    if set(aggregate) != _V2_AGGREGATE_KEYS or aggregate.get("schema") != G90_V2_AGGREGATE_SCHEMA:
        raise PopulationProgramInductionError("G90 V2 aggregate schema/key set differs")
    aggregate_self = _verify_seal(
        aggregate,
        field="aggregate_receipt_sha256",
        label="G90 V2 aggregate",
    )
    if aggregate_self != _require_sha(
        expected_aggregate_self_sha256,
        label="expected G90 V2 aggregate self SHA",
    ):
        raise PopulationProgramInductionError("G90 V2 aggregate self SHA differs from caller custody")
    aggregate_drift = aggregate.get("authority_drift")
    if (
        aggregate.get("pair_range") != [0, PAIR_COUNT]
        or aggregate.get("exact_replay_policy") != G90_V2_EXACT_REPLAY_POLICY
        or aggregate.get("pareto_pruning_performed") is not False
        or aggregate.get("hierarchical_refinement_required_before_atom_selection") is not True
        or aggregate.get("rate_axis") != "UNMEASURED_UNTIL_G94_COMPOSES_ACTUAL_ZIP"
        or aggregate.get("candidate_claim") is not False
        or aggregate.get("score_claim") is not False
        or aggregate.get("pointer_moved") is not False
        or aggregate.get("research_only") is not True
        or aggregate.get("encoder_only") is not True
        or type(aggregate_drift) is not dict
        or set(aggregate_drift)
        != {
            "differentiable_current_argmax_drift_cells",
            "differentiable_target_argmax_drift_cells",
            "inference_cells_remain_authoritative",
        }
        or aggregate_drift.get("inference_cells_remain_authoritative") is not True
    ):
        raise PopulationProgramInductionError("G90 V2 aggregate weakened authority/rate boundary")
    stages = aggregate.get("stages")
    if type(stages) is not list or len(stages) != 5:
        raise PopulationProgramInductionError("G90 V2 aggregate is not sealed full n600")

    interventions: list[G90ExactInterventionV1] = []
    expected_stage_start = 0
    total_projection_count = 0
    total_exact_replay_count = 0
    total_base_pose_sum = 0.0
    total_base_seg_errors = 0
    total_current_drift = 0
    total_target_drift = 0
    for stage_index, stage_binding in enumerate(stages):
        if type(stage_binding) is not dict or set(stage_binding) != {
            "bytes",
            "pair_range",
            "path",
            "sha256",
            "stage_index",
            "stage_receipt_sha256",
        }:
            raise PopulationProgramInductionError("G90 V2 stage binding key set differs")
        stage_identity = ExactFileIdentityV1(
            stage_binding["path"],
            stage_binding["bytes"],
            stage_binding["sha256"],
        )
        stage_path = stage_identity.verify(label=f"G90 V2 stage {stage_index}")
        stage = _load_mapping(stage_path, label=f"G90 V2 stage {stage_index}")
        if set(stage) != _V2_STAGE_KEYS or stage.get("schema") != G90_V2_STAGE_SCHEMA:
            raise PopulationProgramInductionError("G90 V2 stage schema/key set differs")
        stage_self = _verify_seal(
            stage,
            field="stage_receipt_sha256",
            label=f"G90 V2 stage {stage_index}",
        )
        expected_stage_range = [expected_stage_start, expected_stage_start + 120]
        if (
            stage_binding.get("stage_index") != stage_index
            or stage.get("stage_index") != stage_index
            or stage_binding.get("stage_receipt_sha256") != stage_self
            or stage_binding.get("pair_range") != stage.get("pair_range")
            or stage.get("pair_range") != expected_stage_range
            or stage.get("exact_replay_policy") != G90_V2_EXACT_REPLAY_POLICY
            or stage.get("pareto_pruning_performed") is not False
            or stage.get("checkpoint_policy") != "immutable_atomic_preserve_every_120_pair_stage"
            or stage.get("dense_costates_persisted") is not False
            or stage.get("candidate_claim") is not False
            or stage.get("score_claim") is not False
            or stage.get("research_only") is not True
            or stage.get("encoder_only") is not True
        ):
            raise PopulationProgramInductionError("G90 V2 stage continuity/authority differs")
        _require_sha(
            stage.get("g78_stage_receipt_sha256"),
            label=f"G90 V2 stage {stage_index} G78 SHA",
        )
        _require_sha(
            stage.get("g87_stage_checkpoint_sha256"),
            label=f"G90 V2 stage {stage_index} G87 SHA",
        )
        expected_stage_start += 120
        batches = stage.get("batches")
        if (
            type(batches) is not list
            or not batches
            or type(stage.get("batch_count")) is not int
            or stage.get("batch_count") != len(batches)
        ):
            raise PopulationProgramInductionError("G90 V2 stage batch count differs")
        expected_batch_start = stage["pair_range"][0]
        stage_projection_count = 0
        stage_exact_replay_count = 0
        stage_base_pose_sum = 0.0
        stage_base_seg_errors = 0
        stage_current_drift = 0
        stage_target_drift = 0
        for batch_binding in batches:
            if type(batch_binding) is not dict or set(batch_binding) != {
                "batch_checkpoint_sha256",
                "bytes",
                "pair_range",
                "path",
                "sha256",
            }:
                raise PopulationProgramInductionError("G90 V2 batch binding key set differs")
            batch_identity = ExactFileIdentityV1(
                batch_binding["path"],
                batch_binding["bytes"],
                batch_binding["sha256"],
            )
            batch_path = batch_identity.verify(label="G90 V2 batch")
            batch = _load_mapping(batch_path, label="G90 V2 batch")
            if set(batch) != _V2_BATCH_KEYS or batch.get("schema") != G90_V2_BATCH_SCHEMA:
                raise PopulationProgramInductionError("G90 V2 batch schema/key set differs")
            batch_self = _verify_seal(
                batch,
                field="batch_checkpoint_sha256",
                label="G90 V2 batch",
            )
            pair_range = batch.get("pair_range")
            if (
                batch_binding.get("batch_checkpoint_sha256") != batch_self
                or batch_binding.get("pair_range") != pair_range
                or type(pair_range) is not list
                or len(pair_range) != 2
                or any(type(item) is not int for item in pair_range)
                or pair_range[0] != expected_batch_start
                or not pair_range[0] < pair_range[1] <= stage["pair_range"][1]
                or pair_range[1] - pair_range[0] > 16
            ):
                raise PopulationProgramInductionError("G90 V2 batch continuity/custody differs")
            expected_batch_start = pair_range[1]
            pair_ids = tuple(range(pair_range[0], pair_range[1]))
            expected_ids_value = batch.get("expected_physical_group_ids")
            expected_group_count = batch.get("expected_physical_group_count")
            if (
                type(expected_ids_value) is not list
                or any(type(group_id) is not str for group_id in expected_ids_value)
                or len(expected_ids_value) < 8
                or expected_ids_value != list(dict.fromkeys(expected_ids_value))
                or type(expected_group_count) is not int
                or expected_group_count != len(expected_ids_value)
                or batch.get("projection_coordinate_count") != expected_group_count
                or batch.get("exact_replay_policy") != G90_V2_EXACT_REPLAY_POLICY
                or batch.get("all_deterministic_physical_groups_exact_replayed") is not True
                or batch.get("pareto_pruning_performed") is not False
                or batch.get("local_admission_performed") is not False
                or batch.get("dense_costates_persisted") is not False
                or batch.get("actual_zip_delta_measured") is not False
                or batch.get("member_bytes_used_as_rate") is not False
                or batch.get("candidate_claim") is not False
                or batch.get("score_claim") is not False
                or batch.get("research_only") is not True
                or batch.get("encoder_only") is not True
            ):
                raise PopulationProgramInductionError("G90 V2 batch lost exact-all/no-admission authority")
            expected_group_ids = tuple(expected_ids_value)
            for group_id in expected_group_ids:
                match = _GROUP_RE.fullmatch(group_id)
                if match is None or int(match["start"]) != pair_range[0] or int(match["stop"]) != pair_range[1]:
                    raise PopulationProgramInductionError("G90 V2 expected physical group ID/range differs")
            source_custody = batch.get("source_custody")
            if type(source_custody) is not dict or set(source_custody) != _V2_SOURCE_CUSTODY_KEYS:
                raise PopulationProgramInductionError("G90 V2 source custody key set differs")
            for key in sorted(_V2_SOURCE_CUSTODY_KEYS):
                _require_sha(source_custody.get(key), label=f"G90 V2 source custody {key}")
            current_drift, target_drift = _validate_v2_authority_drift(
                batch.get("authority_drift"),
                pair_ids=pair_ids,
                source_custody=source_custody,
            )
            base_components = batch.get("base_components")
            if (
                type(base_components) is not dict
                or set(base_components) != _V2_BASE_COMPONENT_KEYS
                or type(base_components.get("pair_pose_mse_f32")) is not list
                or len(base_components["pair_pose_mse_f32"]) != len(pair_ids)
                or any(
                    type(number) not in {int, float} or not math.isfinite(float(number)) or float(number) < 0.0
                    for number in base_components["pair_pose_mse_f32"]
                )
                or type(base_components.get("seg_mismatch_count")) is not int
                or base_components["seg_mismatch_count"] < 0
                or type(base_components.get("target_minus_current_gap_sum")) not in {int, float}
                or not math.isfinite(float(base_components["target_minus_current_gap_sum"]))
            ):
                raise PopulationProgramInductionError("G90 V2 base components differ")
            pose_vjp_scale = batch.get("population_pose_pair_mse_vjp_scale")
            if (
                type(pose_vjp_scale) not in {int, float}
                or not math.isfinite(float(pose_vjp_scale))
                or float(pose_vjp_scale) <= 0.0
            ):
                raise PopulationProgramInductionError("G90 V2 pose VJP scale differs")

            rows = batch.get("projection_rows")
            basis_rows = batch.get("actuator_basis_groups")
            replay_rows = batch.get("exact_replay_state_custody")
            if (
                type(rows) is not list
                or type(basis_rows) is not list
                or type(replay_rows) is not list
                or len(rows) != expected_group_count
                or len(basis_rows) != expected_group_count
                or len(replay_rows) != expected_group_count
                or tuple(row.get("operand_id") if type(row) is dict else None for row in rows) != expected_group_ids
                or tuple(row.get("group_id") if type(row) is dict else None for row in basis_rows) != expected_group_ids
                or tuple(row.get("operand_id") if type(row) is dict else None for row in replay_rows)
                != expected_group_ids
            ):
                raise PopulationProgramInductionError("G90 V2 ordered row/basis/replay coverage differs")
            basis_by_id: dict[
                str,
                tuple[
                    tuple[BoundaryShearletAtomV1, ...],
                    tuple[str, ...],
                    str,
                    str,
                ],
            ] = {}
            for basis_value in basis_rows:
                group_id, atoms, fingerprints, proposed_sha, incumbent_sha = _parse_basis_group(basis_value)
                if group_id in basis_by_id:
                    raise PopulationProgramInductionError("G90 V2 basis IDs are duplicate")
                basis_by_id[group_id] = (
                    atoms,
                    fingerprints,
                    proposed_sha,
                    incumbent_sha,
                )
            if tuple(basis_by_id) != expected_group_ids:
                raise PopulationProgramInductionError("G90 V2 basis order differs")
            for replay_value, expected_group_id in zip(
                replay_rows,
                expected_group_ids,
                strict=True,
            ):
                if type(replay_value) is not dict or set(replay_value) != _V2_REPLAY_STATE_KEYS:
                    raise PopulationProgramInductionError("G90 V2 replay-state custody key set differs")
                if (
                    replay_value.get("operand_id") != expected_group_id
                    or replay_value.get("candidate_y0_preserved") is not True
                ):
                    raise PopulationProgramInductionError("G90 V2 replay-state identity/Y0 preservation differs")
                for key in (
                    "pose_conditioning_y0_sha256",
                    "pose_conditioning_y1_sha256",
                    "seg_base_y1_sha256",
                    "seg_candidate_y1_sha256",
                ):
                    _require_sha(
                        replay_value.get(key),
                        label=f"G90 V2 {expected_group_id} {key}",
                    )
                if replay_value["pose_conditioning_y1_sha256"] != replay_value["seg_base_y1_sha256"]:
                    raise PopulationProgramInductionError("G90 V2 pose/Seg base Y1 custody differs")
            for row_value in rows:
                if type(row_value) is not dict:
                    raise PopulationProgramInductionError("G90 V2 projection row is not one object")
                exact = _parse_exact_intervention(
                    row_value,
                    pareto_ids=set(),
                    basis_by_id=basis_by_id,
                )
                if exact is None:
                    raise PopulationProgramInductionError("G90 V2 exact-all row lacks exact replay")
                interventions.append(exact)

            batch_pose_sum = float(
                np.asarray(
                    base_components["pair_pose_mse_f32"],
                    dtype=np.float32,
                ).sum(dtype=np.float32)
            )
            stage_base_pose_sum += batch_pose_sum
            stage_base_seg_errors += base_components["seg_mismatch_count"]
            stage_current_drift += current_drift
            stage_target_drift += target_drift
            stage_projection_count += expected_group_count
            stage_exact_replay_count += expected_group_count
        if (
            expected_batch_start != stage["pair_range"][1]
            or type(stage.get("projection_coordinate_count")) is not int
            or stage["projection_coordinate_count"] < 0
            or stage.get("projection_coordinate_count") != stage_projection_count
            or type(stage.get("exact_replay_count")) is not int
            or stage["exact_replay_count"] < 0
            or stage.get("exact_replay_count") != stage_exact_replay_count
            or type(stage.get("base_pose_squared_error_sum_f32")) not in {int, float}
            or not math.isfinite(float(stage["base_pose_squared_error_sum_f32"]))
            or float(stage["base_pose_squared_error_sum_f32"]) < 0.0
            or float(stage["base_pose_squared_error_sum_f32"]) != stage_base_pose_sum
            or type(stage.get("base_segmentation_error_count")) is not int
            or stage["base_segmentation_error_count"] < 0
            or stage.get("base_segmentation_error_count") != stage_base_seg_errors
            or type(stage.get("differentiable_current_argmax_drift_cells")) is not int
            or stage["differentiable_current_argmax_drift_cells"] < 0
            or stage.get("differentiable_current_argmax_drift_cells") != stage_current_drift
            or type(stage.get("differentiable_target_argmax_drift_cells")) is not int
            or stage["differentiable_target_argmax_drift_cells"] < 0
            or stage.get("differentiable_target_argmax_drift_cells") != stage_target_drift
        ):
            raise PopulationProgramInductionError("G90 V2 stage child totals differ")
        total_projection_count += stage_projection_count
        total_exact_replay_count += stage_exact_replay_count
        total_base_pose_sum += stage_base_pose_sum
        total_base_seg_errors += stage_base_seg_errors
        total_current_drift += stage_current_drift
        total_target_drift += stage_target_drift

    base = aggregate.get("base_row")
    if type(base) is not dict or set(base) != {
        "archive_bytes",
        "archive_sha256",
        "d_pose",
        "d_seg",
        "exact_g85_components_reproduced_to_reported_precision",
    }:
        raise PopulationProgramInductionError("G90 V2 base row key set differs")
    derived_d_pose = total_base_pose_sum / PAIR_COUNT
    derived_d_seg = total_base_seg_errors / (PAIR_COUNT * 48 * 64)
    if (
        type(base.get("archive_bytes")) is not int
        or base["archive_bytes"] <= 0
        or type(base.get("d_pose")) not in {int, float}
        or type(base.get("d_seg")) not in {int, float}
        or not math.isfinite(float(base["d_pose"]))
        or not math.isfinite(float(base["d_seg"]))
        or float(base["d_pose"]) < 0.0
        or float(base["d_seg"]) < 0.0
        or expected_stage_start != PAIR_COUNT
        or type(aggregate.get("projection_coordinate_count")) is not int
        or aggregate["projection_coordinate_count"] < 0
        or aggregate.get("projection_coordinate_count") != total_projection_count
        or type(aggregate.get("exact_replay_count")) is not int
        or aggregate["exact_replay_count"] < 0
        or aggregate.get("exact_replay_count") != total_exact_replay_count
        or total_projection_count != total_exact_replay_count
        or type(aggregate_drift.get("differentiable_current_argmax_drift_cells")) is not int
        or aggregate_drift["differentiable_current_argmax_drift_cells"] < 0
        or aggregate_drift.get("differentiable_current_argmax_drift_cells") != total_current_drift
        or type(aggregate_drift.get("differentiable_target_argmax_drift_cells")) is not int
        or aggregate_drift["differentiable_target_argmax_drift_cells"] < 0
        or aggregate_drift.get("differentiable_target_argmax_drift_cells") != total_target_drift
        or base.get("exact_g85_components_reproduced_to_reported_precision") is not True
        or float(base.get("d_pose")) != derived_d_pose
        or float(base.get("d_seg")) != derived_d_seg
    ):
        raise PopulationProgramInductionError("G90 V2 aggregate child/base totals differ")
    ordered = tuple(sorted(interventions, key=lambda row: row.operand_id))
    if len(ordered) != total_exact_replay_count:
        raise PopulationProgramInductionError("G90 V2 intervention count differs")
    return SealedG90PopulationV1(
        aggregate=aggregate_identity,
        aggregate_self_sha256=aggregate_self,
        base_d_seg=derived_d_seg,
        base_d_pose=derived_d_pose,
        base_archive_bytes=base["archive_bytes"],
        base_archive_sha256=_require_sha(
            base["archive_sha256"],
            label="G90 V2 base archive SHA",
        ),
        interventions=ordered,
        unresolved_projection_ids=(),
        source_schema=G90_V2_AGGREGATE_SCHEMA,
        exact_replay_atlas_complete=True,
    )


def load_sealed_g90_population(
    aggregate_identity: ExactFileIdentityV1,
    *,
    expected_aggregate_self_sha256: str,
) -> SealedG90PopulationV1:
    """Dispatch only between the two closed G90 aggregate schema contracts."""

    aggregate_path = aggregate_identity.verify(label="G90 aggregate")
    aggregate = _load_mapping(aggregate_path, label="G90 aggregate")
    schema = aggregate.get("schema")
    if schema == G90_AGGREGATE_SCHEMA:
        return _load_sealed_g90_population_v1(
            aggregate_identity,
            expected_aggregate_self_sha256=expected_aggregate_self_sha256,
        )
    if schema == G90_V2_AGGREGATE_SCHEMA:
        return _load_sealed_g90_population_v2(
            aggregate_identity,
            expected_aggregate_self_sha256=expected_aggregate_self_sha256,
        )
    raise PopulationProgramInductionError("G90 aggregate schema is unsupported")


def _load_sealed_g90_population_v1(
    aggregate_identity: ExactFileIdentityV1,
    *,
    expected_aggregate_self_sha256: str,
) -> SealedG90PopulationV1:
    """Reopen the V1 aggregate→stage→batch seal graph.

    The input ABI intentionally requires proposal atoms in each row.  A G90
    aggregate that records only an operand hash cannot be lowered into G89 and
    is rejected rather than reconstructed from a possibly stale G87 teacher.
    """

    aggregate_path = aggregate_identity.verify(label="G90 aggregate")
    aggregate = _load_mapping(aggregate_path, label="G90 aggregate")
    if set(aggregate) != _AGGREGATE_KEYS or aggregate.get("schema") != G90_AGGREGATE_SCHEMA:
        raise PopulationProgramInductionError("G90 aggregate schema/key set differs")
    aggregate_self = _verify_seal(
        aggregate,
        field="aggregate_receipt_sha256",
        label="G90 aggregate",
    )
    if aggregate_self != _require_sha(
        expected_aggregate_self_sha256,
        label="expected G90 aggregate self SHA",
    ):
        raise PopulationProgramInductionError("G90 aggregate self SHA differs from caller custody")
    if (
        aggregate.get("pair_range") != [0, PAIR_COUNT]
        or aggregate.get("dense_costates_persisted") is not False
        or aggregate.get("rate_axis") != "UNMEASURED_UNTIL_G83_COMPOSES_ACTUAL_ZIP"
        or aggregate.get("selection_consumer") != "G83_WHOLE_STATE_ALLOCATOR_ONLY"
        or aggregate.get("candidate_claim") is not False
        or aggregate.get("score_claim") is not False
        or aggregate.get("research_only") is not True
        or aggregate.get("encoder_only") is not True
    ):
        raise PopulationProgramInductionError("G90 aggregate weakened authority/rate boundary")
    stages = aggregate.get("stages")
    if type(stages) is not list or len(stages) != 5:
        raise PopulationProgramInductionError("G90 aggregate is not sealed full n600")

    interventions: list[G90ExactInterventionV1] = []
    unresolved: list[str] = []
    expected_start = 0
    projection_count = 0
    for stage_index, stage_binding in enumerate(stages):
        if type(stage_binding) is not dict or set(stage_binding) != {
            "bytes",
            "pair_range",
            "path",
            "sha256",
            "stage_index",
            "stage_receipt_sha256",
        }:
            raise PopulationProgramInductionError("G90 stage binding key set differs")
        identity = ExactFileIdentityV1(stage_binding["path"], stage_binding["bytes"], stage_binding["sha256"])
        stage_path = identity.verify(label=f"G90 stage {stage_index}")
        stage = _load_mapping(stage_path, label=f"G90 stage {stage_index}")
        if set(stage) != _STAGE_KEYS or stage.get("schema") != G90_STAGE_SCHEMA:
            raise PopulationProgramInductionError("G90 stage schema/key set differs")
        self_sha = _verify_seal(
            stage,
            field="stage_receipt_sha256",
            label=f"G90 stage {stage_index}",
        )
        if (
            stage_binding["stage_index"] != stage_index
            or stage.get("stage_index") != stage_index
            or stage_binding["stage_receipt_sha256"] != self_sha
            or stage_binding["pair_range"] != stage.get("pair_range")
            or stage.get("pair_range") != [expected_start, expected_start + 120]
        ):
            raise PopulationProgramInductionError("G90 stage continuity/custody differs")
        expected_start += 120
        batches = stage.get("batches")
        if type(batches) is not list or stage.get("batch_count") != len(batches):
            raise PopulationProgramInductionError("G90 stage batch count differs")
        expected_batch_start = stage["pair_range"][0]
        stage_projection_count = 0
        for binding in batches:
            if type(binding) is not dict or set(binding) != {
                "batch_checkpoint_sha256",
                "bytes",
                "pair_range",
                "path",
                "sha256",
            }:
                raise PopulationProgramInductionError("G90 batch binding key set differs")
            batch_identity = ExactFileIdentityV1(binding["path"], binding["bytes"], binding["sha256"])
            batch_path = batch_identity.verify(label="G90 batch")
            batch = _load_mapping(batch_path, label="G90 batch")
            if set(batch) != _BATCH_KEYS or batch.get("schema") != G90_BATCH_SCHEMA:
                raise PopulationProgramInductionError("G90 batch schema/key set differs")
            batch_self = _verify_seal(
                batch,
                field="batch_checkpoint_sha256",
                label="G90 batch",
            )
            pair_range = batch.get("pair_range")
            if (
                binding["batch_checkpoint_sha256"] != batch_self
                or binding["pair_range"] != pair_range
                or type(pair_range) is not list
                or len(pair_range) != 2
                or pair_range[0] != expected_batch_start
                or not pair_range[0] < pair_range[1] <= stage["pair_range"][1]
                or pair_range[1] - pair_range[0] > 16
            ):
                raise PopulationProgramInductionError("G90 batch continuity/custody differs")
            expected_batch_start = pair_range[1]
            if (
                batch.get("actual_zip_delta_measured") is not False
                or batch.get("member_bytes_used_as_rate") is not False
                or batch.get("local_admission_performed") is not False
                or batch.get("selection_consumer") != "G83_WHOLE_STATE_ALLOCATOR_ONLY"
                or batch.get("actuator_basis_reconstructs_exact_measured_proposed_atoms") is not True
                or batch.get("incumbent_and_proposed_atom_custody_separate") is not True
            ):
                raise PopulationProgramInductionError("G90 batch weakened selection/rate boundary")
            rows = batch.get("projection_rows")
            pareto_value = batch.get("pareto_nondominated_operand_ids")
            basis_value = batch.get("actuator_basis_groups")
            if type(rows) is not list or type(pareto_value) is not list or type(basis_value) is not list:
                raise PopulationProgramInductionError("G90 batch rows/Pareto set changed ABI")
            basis_by_id: dict[
                str,
                tuple[
                    tuple[BoundaryShearletAtomV1, ...],
                    tuple[str, ...],
                    str,
                    str,
                ],
            ] = {}
            for basis_row in basis_value:
                group_id, atoms, fingerprints, proposed_sha, incumbent_sha = _parse_basis_group(basis_row)
                if group_id in basis_by_id:
                    raise PopulationProgramInductionError("G90 actuator basis group IDs are duplicate")
                basis_by_id[group_id] = (
                    atoms,
                    fingerprints,
                    proposed_sha,
                    incumbent_sha,
                )
            pareto_ids = set(pareto_value)
            row_ids: set[str] = set()
            for row_value in rows:
                if type(row_value) is not dict:
                    raise PopulationProgramInductionError("G90 projection row is not one object")
                operand_id = row_value.get("operand_id")
                if type(operand_id) is not str or operand_id in row_ids:
                    raise PopulationProgramInductionError("G90 batch operand IDs are invalid/duplicate")
                row_ids.add(operand_id)
                exact = _parse_exact_intervention(
                    row_value,
                    pareto_ids=pareto_ids,
                    basis_by_id=basis_by_id,
                )
                if exact is None:
                    unresolved.append(operand_id)
                else:
                    interventions.append(exact)
            if (
                set(basis_by_id) != row_ids
                or not pareto_ids.issubset(row_ids)
                or batch.get("projection_coordinate_count") != len(rows)
            ):
                raise PopulationProgramInductionError("G90 Pareto/projection count differs")
            stage_projection_count += len(rows)
        if expected_batch_start != stage["pair_range"][1]:
            raise PopulationProgramInductionError("G90 stage batch coverage has a gap")
        if stage.get("projection_coordinate_count") != stage_projection_count:
            raise PopulationProgramInductionError("G90 stage projection total differs")
        projection_count += stage_projection_count
    if expected_start != PAIR_COUNT or aggregate.get("projection_coordinate_count") != projection_count:
        raise PopulationProgramInductionError("G90 aggregate projection/population total differs")
    base = aggregate.get("base_row")
    if type(base) is not dict or set(base) != {
        "archive_bytes",
        "archive_sha256",
        "d_pose",
        "d_seg",
        "exact_g85_components_reproduced_to_reported_precision",
    }:
        raise PopulationProgramInductionError("G90 base row key set differs")
    if base["exact_g85_components_reproduced_to_reported_precision"] is not True:
        raise PopulationProgramInductionError("G90 base row was not exactly reproduced")
    ordered = tuple(sorted(interventions, key=lambda row: row.operand_id))
    return SealedG90PopulationV1(
        aggregate=aggregate_identity,
        aggregate_self_sha256=aggregate_self,
        base_d_seg=float(base["d_seg"]),
        base_d_pose=float(base["d_pose"]),
        base_archive_bytes=base["archive_bytes"],
        base_archive_sha256=_require_sha(base["archive_sha256"], label="G90 base archive SHA"),
        interventions=ordered,
        unresolved_projection_ids=tuple(sorted(unresolved)),
    )


@dataclass(frozen=True, slots=True)
class SharedPhysicalFamilyV1:
    role: str
    direction_rank: int
    amplitude_scale: str
    intervention_ids: tuple[str, ...]

    @property
    def family_id(self) -> str:
        return f"g92:{self.role}:d{self.direction_rank}:a{self.amplitude_scale}"


def induce_shared_physical_families(
    interventions: Sequence[G90ExactInterventionV1],
) -> tuple[SharedPhysicalFamilyV1, ...]:
    """Group only parameters actually shared by the G72 source variants."""

    groups: dict[tuple[str, int, str], list[str]] = {}
    for row in interventions:
        groups.setdefault(row.physical_family_key, []).append(row.operand_id)
    return tuple(
        SharedPhysicalFamilyV1(key[0], key[1], key[2], tuple(sorted(ids))) for key, ids in sorted(groups.items())
    )


@dataclass(frozen=True, slots=True)
class PopulationProgramPlanV1:
    """Partial collision-free atlas, not receiver-lowered or optimal states."""

    g90_aggregate_sha256: str
    g90_aggregate_self_sha256: str
    g90_source_schema: str
    exact_replay_atlas_complete: bool
    g51_receipt_sha256: str
    current_base_archive_bytes: int
    current_base_archive_sha256: str
    shared_families: tuple[SharedPhysicalFamilyV1, ...]
    branches: tuple[tuple[str, ...], ...]
    screening_only_projection_ids: tuple[str, ...]
    lowering_blocker: str = SEQUENTIAL_LOWERING_BLOCKER

    @property
    def partial_enumerated_branch_state_count(self) -> int:
        """Count exact replay rows once; the shared base is not duplicated."""

        return sum(len(branch) for branch in self.branches)

    @property
    def g83_ready(self) -> bool:
        return False

    @property
    def archive_pricing_allowed(self) -> bool:
        return False


def _branch_interventions(
    interventions: Sequence[G90ExactInterventionV1],
) -> tuple[tuple[G90ExactInterventionV1, ...], ...]:
    """Graph-colour collisions without admission or prefix-order optimality."""

    branches: list[list[G90ExactInterventionV1]] = []
    occupied: list[set[tuple[int, int, int, int]]] = []
    for row in sorted(interventions, key=lambda item: item.operand_id):
        addresses = set(row.addresses)
        for branch_index, used in enumerate(occupied):
            if not addresses.intersection(used):
                branches[branch_index].append(row)
                used.update(addresses)
                break
        else:
            branches.append([row])
            occupied.append(addresses)
    return tuple(tuple(branch) for branch in branches)


def compile_population_program_plan(
    *,
    g90: SealedG90PopulationV1,
    g51_receipt_identity: ExactFileIdentityV1,
) -> PopulationProgramPlanV1:
    """Compile a partial exact atlas while refusing false rate/optimality."""

    g51_receipt_identity.verify(label="G51 opaque provenance receipt")
    branch_rows = _branch_interventions(g90.interventions)
    return PopulationProgramPlanV1(
        g90_aggregate_sha256=g90.aggregate.sha256,
        g90_aggregate_self_sha256=g90.aggregate_self_sha256,
        g90_source_schema=g90.source_schema,
        exact_replay_atlas_complete=g90.exact_replay_atlas_complete,
        g51_receipt_sha256=g51_receipt_identity.sha256,
        current_base_archive_bytes=g90.base_archive_bytes,
        current_base_archive_sha256=g90.base_archive_sha256,
        shared_families=induce_shared_physical_families(g90.interventions),
        branches=tuple(tuple(row.operand_id for row in branch) for branch in branch_rows),
        screening_only_projection_ids=g90.unresolved_projection_ids,
        lowering_blocker=(
            V2_COMPOSED_LOWERING_BLOCKER if g90.exact_replay_atlas_complete else SEQUENTIAL_LOWERING_BLOCKER
        ),
    )


def materialize_prefix_family(*_args: object, **_kwargs: object) -> None:
    """Fail until V2 exact-all lowering through G94 has same-state n600 rows."""

    raise PopulationProgramInductionError(SEQUENTIAL_LOWERING_BLOCKER)


__all__ = [
    "EXACT_LOWERING_CHAIN_BLOCKER",
    "G51_PAYLOAD_POLICY",
    "G83_DECODER_TRANSITION_ID",
    "G90_V2_AGGREGATE_SCHEMA",
    "G90_V2_BATCH_SCHEMA",
    "G90_V2_EXACT_REPLAY_POLICY",
    "G90_V2_STAGE_SCHEMA",
    "G92_FAMILY_SCHEMA",
    "G92_MEASUREMENT_SCHEMA",
    "G92_SELECTION_CONTRACT",
    "SEQUENTIAL_LOWERING_BLOCKER",
    "V2_COMPOSED_LOWERING_BLOCKER",
    "ExactFileIdentityV1",
    "G90ExactInterventionV1",
    "PopulationProgramInductionError",
    "PopulationProgramPlanV1",
    "SealedG90PopulationV1",
    "SharedPhysicalFamilyV1",
    "canonical_json_bytes",
    "compile_population_program_plan",
    "induce_shared_physical_families",
    "load_sealed_g90_population",
    "materialize_prefix_family",
    "sha256_bytes",
    "sha256_file",
]
