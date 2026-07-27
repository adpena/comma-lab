from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import tac.witness_dsl.taskspace_g92_population_global_program_induction_v1 as g92
from tac.optimization.direct_description_carrier_compose import (
    BoundaryShearletAtomV1,
)
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    RoleAwareBoundaryShearletOperandV1,
)
from tac.witness_dsl.taskspace_g92_population_global_program_induction_v1 import (
    G51_PAYLOAD_POLICY,
    G90_V2_AGGREGATE_SCHEMA,
    G90_V2_BATCH_SCHEMA,
    G90_V2_EXACT_REPLAY_POLICY,
    G90_V2_STAGE_SCHEMA,
    SEQUENTIAL_LOWERING_BLOCKER,
    V2_COMPOSED_LOWERING_BLOCKER,
    ExactFileIdentityV1,
    G90ExactInterventionV1,
    PopulationProgramInductionError,
    SealedG90PopulationV1,
    canonical_json_bytes,
    compile_population_program_plan,
    induce_shared_physical_families,
    load_sealed_g90_population,
    materialize_prefix_family,
    sha256_bytes,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    SelectedPreimageFrameSelectorV1,
)


def _atom_dict(atom: BoundaryShearletAtomV1) -> dict[str, object]:
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


def _intervention(
    *,
    direction: int,
    center_x: int = 40,
    amplitude: str = "1",
) -> G90ExactInterventionV1:
    atom = BoundaryShearletAtomV1(
        0,
        "Road",
        200,
        center_x,
        4,
        8,
        direction * 4,
        64 if amplitude == "1" else 32,
    )
    proposed = RoleAwareBoundaryShearletOperandV1(
        SelectedPreimageFrameSelectorV1.Y1,
        (atom,),
    )
    return G90ExactInterventionV1(
        operand_id=f"g72:0000_0016:Road:d{direction}:a{amplitude}:p0",
        pair_ids=tuple(range(16)),
        role="Road",
        direction_rank=direction,
        amplitude_scale=amplitude,
        partition_index=0,
        atoms=(atom,),
        proposal_atom_fingerprints=("1" * 64,),
        proposed_atoms_sha256=proposed.sha256,
        operand_member_bytes=len(proposed.to_bytes()),
        operand_sha256=proposed.sha256,
        changed_camera_values=12,
        pose_linearized_score_delta=-0.01,
        seg_gap_directional_delta=2.0,
        exact_seg_mismatch_delta=-3,
        exact_seg_score_delta=-1e-6,
        exact_pose_mean_delta=-2e-6,
        exact_pose_score_delta=-3e-6,
    )


def _identity(path: Path) -> ExactFileIdentityV1:
    return ExactFileIdentityV1(
        str(path),
        path.stat().st_size,
        sha256_bytes(path.read_bytes()),
    )


def _g90(tmp_path: Path, rows: tuple[G90ExactInterventionV1, ...]) -> SealedG90PopulationV1:
    aggregate_path = tmp_path / "g90.json"
    aggregate_path.write_bytes(b"sealed-g90-fixture")
    return SealedG90PopulationV1(
        aggregate=_identity(aggregate_path),
        aggregate_self_sha256="3" * 64,
        base_d_seg=0.02,
        base_d_pose=1.0,
        base_archive_bytes=100,
        base_archive_sha256="4" * 64,
        interventions=rows,
        unresolved_projection_ids=("screening-only",),
    )


def test_shared_family_groups_only_true_source_parameters() -> None:
    rows = (
        _intervention(direction=0, center_x=40),
        _intervention(direction=1, center_x=41),
    )
    families = induce_shared_physical_families(rows)
    assert [(row.role, row.direction_rank, row.amplitude_scale) for row in families] == [
        ("Road", 0, "1"),
        ("Road", 1, "1"),
    ]


def test_program_plan_collision_colors_every_exact_intervention(
    tmp_path: Path,
) -> None:
    g51_path = tmp_path / "g51.json"
    g51_path.write_bytes(b"teacher-only-receipt")
    rows = (
        _intervention(direction=0),
        _intervention(direction=1),  # same donor address: distinct branch
    )
    plan = compile_population_program_plan(
        g90=_g90(tmp_path, rows),
        g51_receipt_identity=_identity(g51_path),
    )
    assert len(plan.branches) == 2
    assert sorted(item for branch in plan.branches for item in branch) == sorted(row.operand_id for row in rows)
    assert plan.partial_enumerated_branch_state_count == 2
    assert plan.archive_pricing_allowed is False
    assert plan.g83_ready is False
    assert plan.lowering_blocker == SEQUENTIAL_LOWERING_BLOCKER


def test_archive_materialization_fails_closed_without_sequential_receiver() -> None:
    with pytest.raises(
        PopulationProgramInductionError,
        match=SEQUENTIAL_LOWERING_BLOCKER,
    ):
        materialize_prefix_family()


def test_g90_basis_group_reconstructs_exact_proposed_operand() -> None:
    row = _intervention(direction=0)
    atom = row.atoms[0]
    proposal = {
        "schema": "tac.g72.boundary_shearlet_proposal.v1",
        "candidate_id": "candidate_sh_d0_a1",
        "fisher_priority": "1",
        "atom": _atom_dict(atom),
    }
    fingerprint = sha256_bytes(canonical_json_bytes(proposal))
    basis = {
        "group_id": row.operand_id,
        "role": row.role,
        "direction_rank": row.direction_rank,
        "amplitude_scale": row.amplitude_scale,
        "proposed_atoms_sha256": row.proposed_atoms_sha256,
        "incumbent_atoms_sha256": "5" * 64,
        "proposals": [proposal],
        "proposal_fingerprints": [fingerprint],
    }
    group_id, atoms, fingerprints, proposed_sha, incumbent_sha = g92._parse_basis_group(basis)
    assert group_id == row.operand_id
    assert atoms == row.atoms
    assert fingerprints == (fingerprint,)
    assert proposed_sha == row.proposed_atoms_sha256
    assert incumbent_sha == "5" * 64


def test_g90_row_without_basis_custody_fails_closed() -> None:
    intervention = _intervention(direction=0)
    row = {
        "operand_id": intervention.operand_id,
        "family_id": "G72_CURRENT_BASE_COMPOSED_ROLE_AWARE_SHEARLET_BATCH_GROUP",
        "pair_ids": list(range(16)),
        "operand_member_bytes": intervention.operand_member_bytes,
        "operand_sha256": intervention.operand_sha256,
        "atom_count": 1,
        "changed_camera_values": 12,
        "pose_linearized_score_delta": -0.01,
        "seg_gap_directional_delta": 2.0,
        "exact_zip_delta_bytes": None,
        "rate_status": "BLOCKED_MEMBER_BYTES_ARE_NOT_A_ZIP_DELTA",
        "exact_seg_mismatch_delta": -3,
        "exact_seg_score_delta": -1e-6,
        "exact_pose_mean_delta": -2e-6,
        "exact_pose_score_delta": -3e-6,
        "proposed_atoms_sha256": intervention.proposed_atoms_sha256,
        "incumbent_atoms_sha256": "5" * 64,
    }
    with pytest.raises(
        PopulationProgramInductionError,
        match="lacks exact proposed-atom custody",
    ):
        g92._parse_exact_intervention(
            row,
            pareto_ids={intervention.operand_id},
            basis_by_id={},
        )


def test_exact_nonpareto_row_is_retained_but_not_promoted_to_optimum() -> None:
    intervention = _intervention(direction=0)
    row = {
        "operand_id": intervention.operand_id,
        "family_id": "G72_CURRENT_BASE_COMPOSED_ROLE_AWARE_SHEARLET_BATCH_GROUP",
        "pair_ids": list(range(16)),
        "operand_member_bytes": intervention.operand_member_bytes,
        "operand_sha256": intervention.operand_sha256,
        "atom_count": 1,
        "changed_camera_values": 12,
        "pose_linearized_score_delta": -0.01,
        "seg_gap_directional_delta": 2.0,
        "exact_zip_delta_bytes": None,
        "rate_status": "BLOCKED_MEMBER_BYTES_ARE_NOT_A_ZIP_DELTA",
        "exact_seg_mismatch_delta": -3,
        "exact_seg_score_delta": -1e-6,
        "exact_pose_mean_delta": -2e-6,
        "exact_pose_score_delta": -3e-6,
        "proposed_atoms_sha256": intervention.proposed_atoms_sha256,
        "incumbent_atoms_sha256": "5" * 64,
    }
    basis = {
        intervention.operand_id: (
            intervention.atoms,
            intervention.proposal_atom_fingerprints,
            intervention.proposed_atoms_sha256,
            "5" * 64,
        )
    }
    parsed = g92._parse_exact_intervention(row, pareto_ids=set(), basis_by_id=basis)
    assert parsed is not None
    assert parsed.pareto_nondominated is False


def test_g51_is_opaque_provenance_not_teacher_authority() -> None:
    assert G51_PAYLOAD_POLICY == ("OPAQUE_EXACT_RECEIPT_PROVENANCE_NOT_PARSED_NO_PAYLOAD_AUTHORITY")


def _sealed(value: dict[str, Any], *, field: str) -> dict[str, Any]:
    assert field not in value
    return {**value, field: sha256_bytes(canonical_json_bytes(value))}


def _write_json(path: Path, value: dict[str, Any]) -> ExactFileIdentityV1:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))
    return _identity(path)


def _v2_group(
    *,
    pair_start: int,
    pair_stop: int,
    role: str,
    direction: int,
    amplitude: str,
    partition: int,
    ordinal: int,
) -> tuple[str, dict[str, object], dict[str, object], dict[str, object]]:
    group_id = f"g72:{pair_start:04d}_{pair_stop:04d}:{role}:d{direction}:a{amplitude}:p{partition}"
    atom = BoundaryShearletAtomV1(
        pair_start,
        role,
        200,
        40 + ordinal,
        4,
        8,
        direction * 4,
        64 if amplitude == "1" else 32,
    )
    operand = RoleAwareBoundaryShearletOperandV1(
        SelectedPreimageFrameSelectorV1.Y1,
        (atom,),
    )
    proposal = {
        "schema": "tac.g72.boundary_shearlet_proposal.v1",
        "candidate_id": f"candidate-{ordinal}",
        "fisher_priority": str(ordinal),
        "atom": _atom_dict(atom),
    }
    fingerprint = sha256_bytes(canonical_json_bytes(proposal))
    incumbent_sha = "5" * 64
    row = {
        "operand_id": group_id,
        "family_id": "G72_CURRENT_BASE_COMPOSED_ROLE_AWARE_SHEARLET_BATCH_GROUP",
        "pair_ids": list(range(pair_start, pair_stop)),
        "operand_member_bytes": len(operand.to_bytes()),
        "operand_sha256": operand.sha256,
        "atom_count": 1,
        "changed_camera_values": 1,
        "pose_linearized_score_delta": 0.0,
        "seg_gap_directional_delta": 0.0,
        "exact_zip_delta_bytes": None,
        "rate_status": "BLOCKED_MEMBER_BYTES_ARE_NOT_A_ZIP_DELTA",
        "exact_seg_mismatch_delta": 0,
        "exact_seg_score_delta": 0.0,
        "exact_pose_mean_delta": 0.0,
        "exact_pose_score_delta": 0.0,
        "proposed_atoms_sha256": operand.sha256,
        "incumbent_atoms_sha256": incumbent_sha,
    }
    basis = {
        "group_id": group_id,
        "role": role,
        "direction_rank": direction,
        "amplitude_scale": amplitude,
        "proposed_atoms_sha256": operand.sha256,
        "incumbent_atoms_sha256": incumbent_sha,
        "proposals": [proposal],
        "proposal_fingerprints": [fingerprint],
    }
    replay = {
        "operand_id": group_id,
        "pose_conditioning_y0_sha256": "6" * 64,
        "pose_conditioning_y1_sha256": "7" * 64,
        "seg_base_y1_sha256": "7" * 64,
        "seg_candidate_y1_sha256": "8" * 64,
        "candidate_y0_preserved": True,
    }
    return group_id, row, basis, replay


def _v2_batch(
    pair_start: int,
    pair_stop: int,
    *,
    twelve_groups: bool,
    mutate: str | None,
) -> dict[str, Any]:
    coordinates = [
        (role, direction, amplitude, 0)
        for role in ("Road", "UndrivableBoundary")
        for direction in (0, 1)
        for amplitude in ("0.5", "1")
    ]
    if twelve_groups:
        coordinates.extend([("Road", direction, amplitude, 1) for direction in (0, 1) for amplitude in ("0.5", "1")])
    groups = [
        _v2_group(
            pair_start=pair_start,
            pair_stop=pair_stop,
            role=role,
            direction=direction,
            amplitude=amplitude,
            partition=partition,
            ordinal=ordinal,
        )
        for ordinal, (role, direction, amplitude, partition) in enumerate(coordinates)
    ]
    group_ids = [group[0] for group in groups]
    current_sha = "a" * 64
    target_sha = "b" * 64

    def drift_axis(axis: str, sha: str) -> dict[str, object]:
        return {
            "axis": axis,
            "expected_cells_sha256": sha,
            "authority_cells_sha256": sha,
            "differentiable_cells_sha256": sha,
            "mismatch_cell_count": 0,
            "mismatch_pair_ids": [],
            "minimum_top_two_margin_at_drift": None,
        }

    body: dict[str, Any] = {
        "schema": G90_V2_BATCH_SCHEMA,
        "pair_range": [pair_start, pair_stop],
        "source_custody": {
            "candidate_camera_sha256": "c" * 64,
            "target_camera_sha256": "d" * 64,
            "target_cells_sha256": target_sha,
            "current_cells_sha256": current_sha,
        },
        "authority_drift": {
            "current": drift_axis("current", current_sha),
            "target": drift_axis("target", target_sha),
            "pose": {
                "authority_current_pose6_sha256": "e" * 64,
                "differentiable_current_pose6_sha256": "e" * 64,
                "authority_target_pose6_sha256": "f" * 64,
                "differentiable_target_pose6_sha256": "f" * 64,
                "maximum_abs_current_delta": 0.0,
                "maximum_abs_target_delta": 0.0,
            },
            "authority_cells_drive_exact_replay": True,
            "authority_pose_targets_and_base_mse_drive_exact_replay": True,
            "differentiable_argmax_has_no_authority": True,
        },
        "base_components": {
            "pair_pose_mse_f32": [1.0] * (pair_stop - pair_start),
            "seg_mismatch_count": 0,
            "target_minus_current_gap_sum": 0.0,
        },
        "population_pose_pair_mse_vjp_scale": 1.0,
        "projection_coordinate_count": len(groups),
        "expected_physical_group_count": len(groups),
        "expected_physical_group_ids": group_ids,
        "projection_rows": [group[1] for group in groups],
        "actuator_basis_groups": [group[2] for group in groups],
        "exact_replay_state_custody": [group[3] for group in groups],
        "exact_replay_policy": G90_V2_EXACT_REPLAY_POLICY,
        "all_deterministic_physical_groups_exact_replayed": True,
        "pareto_pruning_performed": False,
        "local_admission_performed": False,
        "dense_costates_persisted": False,
        "actual_zip_delta_measured": False,
        "member_bytes_used_as_rate": False,
        "candidate_claim": False,
        "score_claim": False,
        "research_only": True,
        "encoder_only": True,
    }
    if mutate == "count":
        body["expected_physical_group_count"] -= 1
    elif mutate == "ids":
        body["expected_physical_group_ids"][0], body["expected_physical_group_ids"][1] = (
            body["expected_physical_group_ids"][1],
            body["expected_physical_group_ids"][0],
        )
    elif mutate == "coverage":
        body["projection_rows"][0]["exact_pose_score_delta"] = None
    return _sealed(body, field="batch_checkpoint_sha256")


def _v2_aggregate(
    tmp_path: Path,
    *,
    mutate: str | None = None,
) -> tuple[ExactFileIdentityV1, str]:
    stage_bindings = []
    total_groups = 0
    for stage_index in range(5):
        stage_start = stage_index * 120
        batch_bindings = []
        stage_groups = 0
        for pair_start in range(stage_start, stage_start + 120, 16):
            pair_stop = min(pair_start + 16, stage_start + 120)
            is_variable_batch = pair_start == 288
            batch = _v2_batch(
                pair_start,
                pair_stop,
                twelve_groups=is_variable_batch,
                mutate=mutate if is_variable_batch else None,
            )
            batch_path = tmp_path / f"stage-{stage_index}" / f"batch-{pair_start:04d}-{pair_stop:04d}.json"
            identity = _write_json(batch_path, batch)
            batch_bindings.append(
                {
                    **identity.to_dict(),
                    "pair_range": [pair_start, pair_stop],
                    "batch_checkpoint_sha256": batch["batch_checkpoint_sha256"],
                }
            )
            stage_groups += int(batch["expected_physical_group_count"])
        stage_body = {
            "schema": G90_V2_STAGE_SCHEMA,
            "stage_index": stage_index,
            "pair_range": [stage_start, stage_start + 120],
            "batches": batch_bindings,
            "batch_count": len(batch_bindings),
            "projection_coordinate_count": stage_groups,
            "exact_replay_count": stage_groups,
            "base_pose_squared_error_sum_f32": 120.0,
            "base_segmentation_error_count": 0,
            "differentiable_current_argmax_drift_cells": 0,
            "differentiable_target_argmax_drift_cells": 0,
            "exact_replay_policy": G90_V2_EXACT_REPLAY_POLICY,
            "pareto_pruning_performed": False,
            "checkpoint_policy": "immutable_atomic_preserve_every_120_pair_stage",
            "dense_costates_persisted": False,
            "candidate_claim": False,
            "score_claim": False,
            "research_only": True,
            "encoder_only": True,
            "g78_stage_receipt_sha256": "1" * 64,
            "g87_stage_checkpoint_sha256": "2" * 64,
        }
        stage = _sealed(stage_body, field="stage_receipt_sha256")
        stage_path = tmp_path / f"stage-{stage_index}" / "stage.json"
        stage_identity = _write_json(stage_path, stage)
        stage_bindings.append(
            {
                **stage_identity.to_dict(),
                "stage_index": stage_index,
                "pair_range": [stage_start, stage_start + 120],
                "stage_receipt_sha256": stage["stage_receipt_sha256"],
            }
        )
        total_groups += stage_groups
    aggregate_body = {
        "schema": G90_V2_AGGREGATE_SCHEMA,
        "pair_range": [0, 600],
        "stages": stage_bindings,
        "projection_coordinate_count": total_groups,
        "exact_replay_count": total_groups,
        "base_row": {
            "d_pose": 1.0,
            "d_seg": 0.0,
            "archive_bytes": 100,
            "archive_sha256": "3" * 64,
            "exact_g85_components_reproduced_to_reported_precision": True,
        },
        "authority_drift": {
            "differentiable_current_argmax_drift_cells": 0,
            "differentiable_target_argmax_drift_cells": 0,
            "inference_cells_remain_authoritative": True,
        },
        "exact_replay_policy": G90_V2_EXACT_REPLAY_POLICY,
        "pareto_pruning_performed": False,
        "hierarchical_refinement_required_before_atom_selection": True,
        "rate_axis": "UNMEASURED_UNTIL_G94_COMPOSES_ACTUAL_ZIP",
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
        "research_only": True,
        "encoder_only": True,
    }
    aggregate = _sealed(aggregate_body, field="aggregate_receipt_sha256")
    identity = _write_json(tmp_path / "aggregate.json", aggregate)
    return identity, aggregate["aggregate_receipt_sha256"]


def test_v2_variable_physical_groups_load_as_complete_isolated_atlas(
    tmp_path: Path,
) -> None:
    identity, self_sha = _v2_aggregate(tmp_path)
    atlas = load_sealed_g90_population(
        identity,
        expected_aggregate_self_sha256=self_sha,
    )
    assert atlas.source_schema == G90_V2_AGGREGATE_SCHEMA
    assert atlas.exact_replay_atlas_complete is True
    assert atlas.unresolved_projection_ids == ()
    assert len(atlas.interventions) == 324
    assert sum(":0288_0304:" in row.operand_id for row in atlas.interventions) == 12
    assert all(row.pareto_nondominated is False for row in atlas.interventions)
    g51_path = tmp_path / "g51.json"
    g51_path.write_bytes(b"opaque")
    plan = compile_population_program_plan(
        g90=atlas,
        g51_receipt_identity=_identity(g51_path),
    )
    assert plan.exact_replay_atlas_complete is True
    assert plan.lowering_blocker == V2_COMPOSED_LOWERING_BLOCKER
    assert plan.archive_pricing_allowed is False
    assert plan.g83_ready is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("count", "exact-all/no-admission authority"),
        ("ids", "ordered row/basis/replay coverage"),
        ("coverage", "partial exact replay"),
    ],
)
def test_v2_forged_group_count_ids_or_exact_coverage_refuses(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    identity, self_sha = _v2_aggregate(tmp_path, mutate=mutation)
    with pytest.raises(PopulationProgramInductionError, match=message):
        load_sealed_g90_population(
            identity,
            expected_aggregate_self_sha256=self_sha,
        )
