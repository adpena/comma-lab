from __future__ import annotations

from pathlib import Path

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
    SEQUENTIAL_LOWERING_BLOCKER,
    ExactFileIdentityV1,
    G90ExactInterventionV1,
    PopulationProgramInductionError,
    SealedG90PopulationV1,
    canonical_json_bytes,
    compile_population_program_plan,
    induce_shared_physical_families,
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
