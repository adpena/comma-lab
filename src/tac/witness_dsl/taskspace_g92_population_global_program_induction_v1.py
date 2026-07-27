# SPDX-License-Identifier: MIT
"""Population-global induction of a partial typed-actuator program atlas.

G90 exact-replayed interventions are observations at one incumbent operating
point.  V1's linear screening and Pareto filter are not sound completeness or
optimality certificates; discarded rows remain unresolved.  The exact rows
are useful as a partial atlas, but their isolated component deltas are not
transferable to a cumulative G89/G94 state.  This module therefore emits:

* the sealed G90 partial observation atlas and exact proposed atoms;
* genuinely shared physical-family identities; and
* collision-free intervention branches for later same-state lowering.

It deliberately emits and prices no archive.  The exact G85 incumbent selects
``BOTH`` while the new intervention selects ``Y1``.  G94 supplies that
sequential receiver type, but G92 still owes integration against a G90 V2
exact-all-coarse atlas and full-n600 same-state rows.  No threshold,
scalarized proxy, completeness claim, or historical dense/full-residual
payload is admitted here.
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
G92_FAMILY_SCHEMA: Final = "tac.taskspace_g92_population_global_prefix_family.v1"
G92_MEASUREMENT_SCHEMA: Final = "tac.taskspace_g92_exact_prefix_measurement.v1"
G92_SELECTION_CONTRACT: Final = "NO_PROXY_RANKING_REQUIRE_SAME_ARCHIVE_FULL_N600_REALIZED_THROUGH_R_PUBLIC_CLOSURE"
G51_PAYLOAD_POLICY: Final = "OPAQUE_EXACT_RECEIPT_PROVENANCE_NOT_PARSED_NO_PAYLOAD_AUTHORITY"
EXACT_LOWERING_CHAIN_BLOCKER: Final = (
    "G90_V2_EXACT_ALL_COARSE_ATLAS_PLUS_G92_TO_G94_LOWERING_PLUS_SAME_STATE_FULL_N600_ROWS_OWED"
)
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

    def __post_init__(self) -> None:
        if not self.interventions:
            raise PopulationProgramInductionError("sealed G90 has no exact replay interventions")
        ids = tuple(row.operand_id for row in self.interventions)
        if ids != tuple(sorted(set(ids))):
            raise PopulationProgramInductionError("G90 interventions are not unique canonical order")
        if set(ids).intersection(self.unresolved_projection_ids):
            raise PopulationProgramInductionError("G90 resolved/unresolved projections overlap")


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
        if type(proposal) is not dict or set(proposal) != {
            "atom",
            "candidate_id",
            "fisher_priority",
            "schema",
        }:
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


def load_sealed_g90_population(
    aggregate_identity: ExactFileIdentityV1,
    *,
    expected_aggregate_self_sha256: str,
) -> SealedG90PopulationV1:
    """Reopen the complete aggregate→stage→batch seal graph.

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
        g51_receipt_sha256=g51_receipt_identity.sha256,
        current_base_archive_bytes=g90.base_archive_bytes,
        current_base_archive_sha256=g90.base_archive_sha256,
        shared_families=induce_shared_physical_families(g90.interventions),
        branches=tuple(tuple(row.operand_id for row in branch) for branch in branch_rows),
        screening_only_projection_ids=g90.unresolved_projection_ids,
    )


def materialize_prefix_family(*_args: object, **_kwargs: object) -> None:
    """Fail until V2 exact-all lowering through G94 has same-state n600 rows."""

    raise PopulationProgramInductionError(SEQUENTIAL_LOWERING_BLOCKER)


__all__ = [
    "EXACT_LOWERING_CHAIN_BLOCKER",
    "G51_PAYLOAD_POLICY",
    "G83_DECODER_TRANSITION_ID",
    "G92_FAMILY_SCHEMA",
    "G92_MEASUREMENT_SCHEMA",
    "G92_SELECTION_CONTRACT",
    "SEQUENTIAL_LOWERING_BLOCKER",
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
