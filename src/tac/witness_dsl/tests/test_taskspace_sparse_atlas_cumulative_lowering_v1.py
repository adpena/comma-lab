from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import (
    BoundaryShearletAtomV1,
)
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    RoleAwareBoundaryShearletOperandV1,
)
from tac.witness_dsl.taskspace_g92_population_global_program_induction_v1 import (
    G90_V2_AGGREGATE_SCHEMA,
    ExactFileIdentityV1,
    G90ExactInterventionV1,
    PopulationProgramPlanV1,
    SealedG90PopulationV1,
    induce_shared_physical_families,
)
from tac.witness_dsl.taskspace_outer_archive_codec import (
    parse_taskspace_outer_archive,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    SelectedPreimageFrameSelectorV1,
)
from tac.witness_dsl.taskspace_sparse_atlas_cumulative_lowering_v1 import (
    CHECKPOINT_SCHEMA,
    G94_PRECONDITIONAL_ABI_ID,
    LoweredSparseAtlasY1V1,
    SparseAtlasCumulativeLoweringError,
    SparseAtlasCumulativeReceiverV1,
    SparseAtlasPrefixBatchV1,
    SparseAtlasY1StepV1,
    build_sparse_atlas_outer_archive,
    lower_selected_sparse_atlas,
    materialize_next_prefix_checkpoint,
    parse_sparse_atlas_y1_operand,
)

CURRENT_BASE_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/g85_pvsa_public_receiver_20260727_r1/archive.zip")
CURRENT_BASE_ARCHIVE_BYTES = 129_392
CURRENT_BASE_ARCHIVE_SHA256 = "b9c8ab2af8886c5b26bba63e02b7c5fe9951bb42a871c5e8472483977788d9fd"
CURRENT_BASE_MEMBER_SHA256 = "d50aac6eab8114c2c15156354147d1cbfe007b474a0633d5cdec26e66751de31"
SEMANTIC_P_SHA256 = "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"

ROAD_ID = "g72:0000_0001:Road:d0:a1:p0"
UNDRIVABLE_ID = "g72:0000_0001:UndrivableBoundary:d0:a1:p0"
SELECTED_IDS = (ROAD_ID, UNDRIVABLE_ID)

ROAD_ATOM = BoundaryShearletAtomV1(
    0,
    "Road",
    240,
    494,
    4,
    8,
    0,
    64,
)
UNDRIVABLE_ATOM = BoundaryShearletAtomV1(
    0,
    "UndrivableBoundary",
    178,
    437,
    4,
    8,
    0,
    64,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> ExactFileIdentityV1:
    return ExactFileIdentityV1(
        path=str(path),
        bytes=path.stat().st_size,
        sha256=_sha256_file(path),
    )


def _intervention(
    *,
    operand_id: str,
    role: str,
    atom: BoundaryShearletAtomV1,
    direction_rank: int = 0,
) -> G90ExactInterventionV1:
    operand = RoleAwareBoundaryShearletOperandV1(
        frame_selector=SelectedPreimageFrameSelectorV1.Y1,
        atoms=(atom,),
    )
    return G90ExactInterventionV1(
        operand_id=operand_id,
        pair_ids=(0,),
        role=role,
        direction_rank=direction_rank,
        amplitude_scale="1",
        partition_index=0,
        atoms=(atom,),
        proposal_atom_fingerprints=(hashlib.sha256(operand_id.encode("ascii")).hexdigest(),),
        proposed_atoms_sha256=operand.sha256,
        operand_member_bytes=len(operand.to_bytes()),
        operand_sha256=operand.sha256,
        changed_camera_values=1,
        pose_linearized_score_delta=0.0,
        seg_gap_directional_delta=0.0,
        exact_seg_mismatch_delta=0,
        exact_seg_score_delta=0.0,
        exact_pose_mean_delta=0.0,
        exact_pose_score_delta=0.0,
        pareto_nondominated=False,
    )


def _atlas_and_plan(
    tmp_path: Path,
    *,
    rows: tuple[G90ExactInterventionV1, ...] | None = None,
) -> tuple[SealedG90PopulationV1, PopulationProgramPlanV1]:
    aggregate_path = tmp_path / "g90_v2_aggregate.json"
    aggregate_path.write_bytes(b'{"fixture":"typed exact-atom custody only"}')
    aggregate = _identity(aggregate_path)
    interventions = rows or (
        _intervention(
            operand_id=ROAD_ID,
            role="Road",
            atom=ROAD_ATOM,
        ),
        _intervention(
            operand_id=UNDRIVABLE_ID,
            role="UndrivableBoundary",
            atom=UNDRIVABLE_ATOM,
        ),
    )
    interventions = tuple(sorted(interventions, key=lambda row: row.operand_id))
    g90 = SealedG90PopulationV1(
        aggregate=aggregate,
        aggregate_self_sha256="a" * 64,
        base_d_seg=0.1,
        base_d_pose=1.0,
        base_archive_bytes=CURRENT_BASE_ARCHIVE_BYTES,
        base_archive_sha256=CURRENT_BASE_ARCHIVE_SHA256,
        interventions=interventions,
        unresolved_projection_ids=(),
        source_schema=G90_V2_AGGREGATE_SCHEMA,
        exact_replay_atlas_complete=True,
    )
    ids = tuple(row.operand_id for row in interventions)
    plan = PopulationProgramPlanV1(
        g90_aggregate_sha256=aggregate.sha256,
        g90_aggregate_self_sha256=g90.aggregate_self_sha256,
        g90_source_schema=G90_V2_AGGREGATE_SCHEMA,
        exact_replay_atlas_complete=True,
        g51_receipt_sha256="b" * 64,
        current_base_archive_bytes=CURRENT_BASE_ARCHIVE_BYTES,
        current_base_archive_sha256=CURRENT_BASE_ARCHIVE_SHA256,
        shared_families=induce_shared_physical_families(interventions),
        branches=(ids,),
        screening_only_projection_ids=(),
    )
    return g90, plan


@pytest.fixture(scope="module")
def current_archive_bytes() -> bytes:
    payload = CURRENT_BASE_ARCHIVE.read_bytes()
    assert len(payload) == CURRENT_BASE_ARCHIVE_BYTES
    assert hashlib.sha256(payload).hexdigest() == CURRENT_BASE_ARCHIVE_SHA256
    return payload


@pytest.fixture(scope="module")
def exact_sparse_fixture(
    tmp_path_factory: pytest.TempPathFactory,
    current_archive_bytes: bytes,
) -> tuple[
    LoweredSparseAtlasY1V1,
    SparseAtlasCumulativeReceiverV1,
    tuple[SparseAtlasPrefixBatchV1, ...],
]:
    g90, plan = _atlas_and_plan(tmp_path_factory.mktemp("g98-atlas"))
    lowered = lower_selected_sparse_atlas(
        g90=g90,
        plan=plan,
        selected_operand_ids=SELECTED_IDS,
        base_outer_archive_bytes=current_archive_bytes,
        verify_member_effects=False,
    )
    receiver = SparseAtlasCumulativeReceiverV1.open(
        base_pvsa_member_bytes=lowered.base_pvsa_member_bytes,
        sparse_operand_bytes=lowered.operand.to_bytes(),
        expected_sparse_operand_sha256=lowered.operand.sha256,
        verify_member_effects=False,
    )
    prefixes = receiver.render_cumulative_prefixes((0,))
    return lowered, receiver, prefixes


def test_sparse_wire_counts_every_selected_id_and_atom_without_g89_filler(
    exact_sparse_fixture: tuple[
        LoweredSparseAtlasY1V1,
        SparseAtlasCumulativeReceiverV1,
        tuple[SparseAtlasPrefixBatchV1, ...],
    ],
) -> None:
    lowered, _receiver, _prefixes = exact_sparse_fixture
    assert lowered.operand.selected_operand_ids == SELECTED_IDS
    assert lowered.operand.atom_count == 2
    assert tuple(step.atoms for step in lowered.operand.steps) == (
        (ROAD_ATOM,),
        (UNDRIVABLE_ATOM,),
    )
    assert lowered.counted_sparse_operand_bytes == len(lowered.operand.to_bytes())
    assert lowered.selected_source_operand_bytes == sum(len(step.operand.to_bytes()) for step in lowered.operand.steps)
    # Road-only is a lawful sparse program.  No all-five-role topology, Lane,
    # or Movable filler is required or present.
    road_only = replace(lowered.operand, steps=lowered.operand.steps[:1])
    assert road_only.atom_count == 1
    assert road_only.steps[0].atoms[0].role == "Road"


def test_exact_cumulative_y1_reproduces_direct_current_state_and_never_mutates_y0(
    exact_sparse_fixture: tuple[
        LoweredSparseAtlasY1V1,
        SparseAtlasCumulativeReceiverV1,
        tuple[SparseAtlasPrefixBatchV1, ...],
    ],
) -> None:
    _lowered, receiver, prefixes = exact_sparse_fixture
    assert len(prefixes) == 2
    direct = receiver.receiver_for_prefix(2).render_camera_pairs((0,))
    assert np.array_equal(prefixes[-1].preconditional_camera_pairs[:, 1], direct[:, 1])
    assert np.array_equal(
        prefixes[0].preconditional_camera_pairs[:, 0],
        prefixes[0].base_incumbent_camera_pairs[:, 0],
    )
    assert np.array_equal(
        prefixes[1].preconditional_camera_pairs[:, 0],
        prefixes[1].base_incumbent_camera_pairs[:, 0],
    )
    assert prefixes[-1].g94_preconditional_abi_id == G94_PRECONDITIONAL_ABI_ID
    assert prefixes[-1].conditioning_state_sha256 == receiver.conditioning_state_sha256


def test_second_prefix_is_conditioned_on_actual_first_prefix_not_isolated_delta(
    exact_sparse_fixture: tuple[
        LoweredSparseAtlasY1V1,
        SparseAtlasCumulativeReceiverV1,
        tuple[SparseAtlasPrefixBatchV1, ...],
    ],
) -> None:
    _lowered, receiver, prefixes = exact_sparse_fixture
    isolated_second_receiver = replace(
        receiver.semantic_receiver,
        boundary_shearlets=tuple(
            sorted(
                (
                    *receiver.semantic_receiver.boundary_shearlets,
                    *receiver.incumbent_atoms,
                    *receiver.operand.steps[1].atoms,
                ),
                key=lambda atom: (
                    atom.pair_index,
                    0 if atom.role == "UndrivableBoundary" else 1,
                    atom.center_y,
                    atom.center_x,
                ),
            )
        ),
    )
    cumulative_receiver = receiver.receiver_for_prefix(2)
    assert prefixes[1].previous_state_sha256 == prefixes[0].current_state_sha256
    assert prefixes[1].previous_combined_y1_sha256 == prefixes[0].combined_y1_sha256
    assert ROAD_ATOM in cumulative_receiver.boundary_shearlets
    assert ROAD_ATOM not in isolated_second_receiver.boundary_shearlets
    assert UNDRIVABLE_ATOM in cumulative_receiver.boundary_shearlets
    assert cumulative_receiver.boundary_shearlets != isolated_second_receiver.boundary_shearlets
    # This source-backed fixture can paint-shadow the first atom at the pixel
    # surface.  The exact current state and its state hash must still carry that
    # atom forward; pixel non-equality would incorrectly infer additivity.
    assert prefixes[1].changed_y1_values_from_previous_prefix > 0


def test_declared_order_is_counted_while_unknown_id_wrong_base_and_collision_fail_closed(
    tmp_path: Path,
    current_archive_bytes: bytes,
    exact_sparse_fixture: tuple[
        LoweredSparseAtlasY1V1,
        SparseAtlasCumulativeReceiverV1,
        tuple[SparseAtlasPrefixBatchV1, ...],
    ],
) -> None:
    g90, plan = _atlas_and_plan(tmp_path)
    reversed_lowering = lower_selected_sparse_atlas(
        g90=g90,
        plan=plan,
        selected_operand_ids=tuple(reversed(SELECTED_IDS)),
        base_outer_archive_bytes=current_archive_bytes,
        verify_member_effects=False,
    )
    assert reversed_lowering.operand.selected_operand_ids == tuple(reversed(SELECTED_IDS))
    assert reversed_lowering.operand.sha256 != exact_sparse_fixture[0].operand.sha256
    reversed_receiver = SparseAtlasCumulativeReceiverV1.open(
        base_pvsa_member_bytes=reversed_lowering.base_pvsa_member_bytes,
        sparse_operand_bytes=reversed_lowering.operand.to_bytes(),
        expected_sparse_operand_sha256=reversed_lowering.operand.sha256,
        verify_member_effects=False,
    )
    reversed_prefixes = reversed_receiver.render_cumulative_prefixes((0,))
    assert reversed_prefixes[0].selected_operand_ids != exact_sparse_fixture[2][0].selected_operand_ids
    assert reversed_prefixes[0].current_state_sha256 != exact_sparse_fixture[2][0].current_state_sha256
    assert np.array_equal(
        reversed_prefixes[-1].preconditional_camera_pairs,
        exact_sparse_fixture[2][-1].preconditional_camera_pairs,
    )
    with pytest.raises(SparseAtlasCumulativeLoweringError, match="unknown"):
        lower_selected_sparse_atlas(
            g90=g90,
            plan=plan,
            selected_operand_ids=("g72:0000_0001:Road:d1:a1:p9",),
            base_outer_archive_bytes=current_archive_bytes,
            verify_member_effects=False,
        )

    lowered, _receiver, _prefixes = exact_sparse_fixture
    wrong_state = replace(
        lowered.operand,
        base_pvsa_member_sha256="0" * 64,
    )
    with pytest.raises(SparseAtlasCumulativeLoweringError, match="different P/base"):
        SparseAtlasCumulativeReceiverV1.open(
            base_pvsa_member_bytes=lowered.base_pvsa_member_bytes,
            sparse_operand_bytes=wrong_state.to_bytes(),
            verify_member_effects=False,
        )

    duplicate_atom = SparseAtlasY1StepV1(
        operand_id="g72:0000_0001:Road:d1:a1:p0",
        operand=RoleAwareBoundaryShearletOperandV1(
            frame_selector=SelectedPreimageFrameSelectorV1.Y1,
            atoms=(ROAD_ATOM,),
        ),
    )
    with pytest.raises(SparseAtlasCumulativeLoweringError, match="collide"):
        replace(
            lowered.operand,
            steps=(lowered.operand.steps[0], duplicate_atom),
        )


def test_outer_archive_parseback_is_exact_and_operand_decode_has_no_receipt_dependency(
    exact_sparse_fixture: tuple[
        LoweredSparseAtlasY1V1,
        SparseAtlasCumulativeReceiverV1,
        tuple[SparseAtlasPrefixBatchV1, ...],
    ],
) -> None:
    lowered, _receiver, _prefixes = exact_sparse_fixture
    build = build_sparse_atlas_outer_archive(lowered.operand)
    assert build.stored == build.deflated == build.selected == lowered.operand
    selected = build.outer_build.selected
    reopened = parse_taskspace_outer_archive(
        selected.archive_bytes,
        expected_archive_sha256=selected.archive_sha256,
        expected_member_sha256=lowered.operand.sha256,
    )
    assert (
        parse_sparse_atlas_y1_operand(
            reopened.member_bytes,
            expected_sha256=lowered.operand.sha256,
        )
        == lowered.operand
    )
    # Donor IDs/hashes are embedded provenance; no aggregate/plan file path is
    # serialized into the decode-time operand.
    assert str(CURRENT_BASE_ARCHIVE).encode("utf-8") not in lowered.operand.to_bytes()


def test_encoder_only_materializer_checkpoints_one_exact_prefix_batch_and_refuses_forgery(
    tmp_path: Path,
    exact_sparse_fixture: tuple[
        LoweredSparseAtlasY1V1,
        SparseAtlasCumulativeReceiverV1,
        tuple[SparseAtlasPrefixBatchV1, ...],
    ],
) -> None:
    _lowered, receiver, _prefixes = exact_sparse_fixture
    status = materialize_next_prefix_checkpoint(
        receiver=receiver,
        checkpoint_root=tmp_path,
        population_pair_ids=(0,),
        batch_pairs=1,
    )
    assert status["status"] == "batch_complete"
    checkpoint_path = Path(status["checkpoint_path"])
    checkpoint = json.loads(checkpoint_path.read_bytes())
    assert checkpoint["schema"] == CHECKPOINT_SCHEMA
    assert checkpoint["encoder_only"] is True
    assert checkpoint["candidate_claim"] is False
    assert checkpoint["score_claim"] is False
    assert checkpoint["population_complete_n600"] is False

    checkpoint["pair_ids"] = [1]
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(
        SparseAtlasCumulativeLoweringError,
        match="self seal",
    ):
        materialize_next_prefix_checkpoint(
            receiver=receiver,
            checkpoint_root=tmp_path,
            population_pair_ids=(0,),
            batch_pairs=1,
        )


def test_base_outer_archive_bytes_are_bound_not_just_member_shape(
    tmp_path: Path,
    current_archive_bytes: bytes,
) -> None:
    g90, plan = _atlas_and_plan(tmp_path)
    tampered = current_archive_bytes[:-1] + bytes((current_archive_bytes[-1] ^ 1,))
    with pytest.raises(SparseAtlasCumulativeLoweringError, match="base outer archive differs"):
        lower_selected_sparse_atlas(
            g90=g90,
            plan=plan,
            selected_operand_ids=(ROAD_ID,),
            base_outer_archive_bytes=tampered,
            verify_member_effects=False,
        )


def test_current_base_member_identity_is_the_g94_product_state(
    current_archive_bytes: bytes,
) -> None:
    outer = parse_taskspace_outer_archive(
        current_archive_bytes,
        expected_archive_sha256=CURRENT_BASE_ARCHIVE_SHA256,
    )
    assert outer.member_sha256 == CURRENT_BASE_MEMBER_SHA256
    assert hashlib.sha256(outer.member_bytes).hexdigest() == CURRENT_BASE_MEMBER_SHA256
    assert SEMANTIC_P_SHA256 != CURRENT_BASE_MEMBER_SHA256
