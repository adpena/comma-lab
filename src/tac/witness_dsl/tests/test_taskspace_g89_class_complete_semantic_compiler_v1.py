from __future__ import annotations

import hashlib
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import (
    BoundaryShearletAtomV1,
    LanePeriodicProgramV1,
    MovableWorldsheetTrackV1,
)
from tac.witness_dsl.taskspace_g89_class_complete_semantic_compiler_v1 import (
    IRREDUCIBLE_QUOTIENT_ID,
    OPEN_PRODUCT_BLOCKERS,
    SELECTION_CONTRACT_ID,
    ClassCompleteSemanticError,
    ClassCompleteSemanticProgramV1,
    ClassCompleteSemanticReceiverV1,
    SharedTopologyApplicationV1,
    SharedTopologyTemplateV1,
    build_class_complete_archive,
    derive_irreducible_learned_quotient,
    parse_class_complete_semantic_program,
    receive_class_complete_archive,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
SEMANTIC_P_PATH = (
    REPO_ROOT
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "fresh_v15_semantic_base_n600_20260726"
    / "ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes"
)
SEMANTIC_P_BYTES = 133_941
SEMANTIC_P_SHA256 = "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"


def _semantic_p() -> bytes:
    payload = SEMANTIC_P_PATH.read_bytes()
    assert len(payload) == SEMANTIC_P_BYTES
    assert hashlib.sha256(payload).hexdigest() == SEMANTIC_P_SHA256
    return payload


def _program() -> ClassCompleteSemanticProgramV1:
    # The five placements are disjoint background boxes in source pair 0.
    # Application order follows the donor wire order, which differs from paint
    # order only for MyCar/Movable.
    applications = (
        SharedTopologyApplicationV1(0, "UndrivableBoundary", 0, 0, 0),
        SharedTopologyApplicationV1(0, "Road", 0, 0, 5),
        SharedTopologyApplicationV1(0, "Lane", 0, 0, 10),
        SharedTopologyApplicationV1(0, "MyCar", 0, 0, 20),
        SharedTopologyApplicationV1(0, "Movable", 0, 0, 15),
    )
    return ClassCompleteSemanticProgramV1(
        semantic_archive_sha256=SEMANTIC_P_SHA256,
        topology_templates=(
            SharedTopologyTemplateV1(
                template_id=0,
                action="birth",
                shape="box",
                lifetime=1,
                height=3,
                width=3,
            ),
        ),
        topology_applications=applications,
        boundary_shearlets=(
            BoundaryShearletAtomV1(
                0,
                "UndrivableBoundary",
                178,
                437,
                4,
                8,
                0,
                64,
            ),
            BoundaryShearletAtomV1(0, "Road", 240, 494, 4, 8, 0, 64),
        ),
        island_shapes=(),
        worldsheet_tracks=(
            MovableWorldsheetTrackV1(
                object_id=0,
                birth_pair=0,
                death_pair_exclusive=1,
                center_y=30,
                center_x=30,
                radius_y=2,
                radius_x=2,
                angle_u8=0,
                skew_q6=0,
                taper_q6=0,
                curvelet_q6=0,
            ),
        ),
        worldsheet_knots=(),
        lane_programs=(
            LanePeriodicProgramV1(
                line_index=0,
                birth_pair=0,
                death_pair_exclusive=600,
                dash_phase_origin_delta_q8=0,
                dash_phase_xi_gain_q8=0,
                width_bias_q8=64,
                width_slope_q12=0,
            ),
        ),
        lane_knots=(),
    )


def test_real_exact_p_class_complete_archive_double_decode() -> None:
    semantic_p = _semantic_p()
    program = _program()
    operand = program.to_bytes()

    assert len(operand) == 247
    assert program.factored_topology_bytes == 80
    assert program.unfactored_topology_bytes == 92
    assert program.factored_topology_bytes < program.unfactored_topology_bytes
    assert (
        parse_class_complete_semantic_program(
            operand,
            expected_sha256=program.sha256,
            maximum_operand_bytes=len(operand),
        )
        == program
    )

    build = build_class_complete_archive(semantic_p, operand)
    assert build.outer.selected.archive_nbytes == 129_812
    assert build.outer.selected.archive_sha256 == "ba761b8ec0e738ad458d589f377e98856e70a864380e9b6ff477d7d4f0725f52"
    first = receive_class_complete_archive(
        build.outer.selected.archive_bytes,
        expected_archive_sha256=build.outer.selected.archive_sha256,
        expected_semantic_archive_sha256=SEMANTIC_P_SHA256,
        expected_program_sha256=program.sha256,
        verify_member_effects=False,
    ).decode((0,))
    second = receive_class_complete_archive(
        build.outer.selected.archive_bytes,
        expected_archive_sha256=build.outer.selected.archive_sha256,
        expected_semantic_archive_sha256=SEMANTIC_P_SHA256,
        expected_program_sha256=program.sha256,
        verify_member_effects=False,
    ).decode((0,))

    assert np.array_equal(first.camera_pairs, second.camera_pairs)
    assert np.array_equal(first.exact_r_pairs, second.exact_r_pairs)
    assert np.array_equal(first.semantic_cells, second.semantic_cells)
    assert np.array_equal(first.camera_pairs[:, 0], first.base_camera_pairs[:, 0])
    assert np.array_equal(first.exact_r_pairs[:, 0], first.base_exact_r_pairs[:, 0])
    assert first.changed_camera_values == 11_760
    assert first.changed_exact_r_values == 2_616
    assert first.changed_semantic_cells == 745
    assert dict(first.changed_role_mask_cells) == {
        "UndrivableBoundary": 20,
        "Road": 9,
        "Lane": 830,
        "Movable": 13,
        "MyCar": 9,
    }
    assert all(value > 0 for _, value in first.changed_role_mask_cells)


def test_program_has_no_fixed_threshold_or_dense_quotient_field() -> None:
    names = {field.name for field in fields(ClassCompleteSemanticProgramV1)}
    assert not any("threshold" in name for name in names)
    assert not any("costate" in name for name in names)
    assert not any("target" in name for name in names)
    assert not any("residual" in name for name in names)
    assert "NO_FIXED_THRESHOLDS" in SELECTION_CONTRACT_ID
    assert "ANALYTIC_SPAN" in IRREDUCIBLE_QUOTIENT_ID
    assert "G89_FULL_N600_CLASS_COMPLETE_OPERAND_NOT_MATERIALIZED" in OPEN_PRODUCT_BLOCKERS


def test_irreducible_quotient_is_encoder_only_exact_remainder() -> None:
    target = np.zeros((1, 384, 512), dtype=np.uint8)
    analytic = np.zeros((1, 384, 512), dtype=np.int16)
    target[0, 7, 11] = 4
    residual, count, sha = derive_irreducible_learned_quotient(target, analytic)
    assert residual.dtype == np.bool_
    assert not residual.flags.writeable
    assert count == 1
    assert len(sha) == 64


def test_missing_one_semantic_role_fails_closed() -> None:
    program = _program()
    with pytest.raises(ClassCompleteSemanticError, match="all five roles"):
        ClassCompleteSemanticProgramV1(
            semantic_archive_sha256=program.semantic_archive_sha256,
            topology_templates=program.topology_templates,
            topology_applications=program.topology_applications[:-1],
            boundary_shearlets=program.boundary_shearlets,
            island_shapes=program.island_shapes,
            worldsheet_tracks=program.worldsheet_tracks,
            worldsheet_knots=program.worldsheet_knots,
            lane_programs=program.lane_programs,
            lane_knots=program.lane_knots,
        )


def test_y1_receiver_rejects_more_than_batch16() -> None:
    semantic_p = _semantic_p()
    program = _program()
    receiver = ClassCompleteSemanticReceiverV1.open(
        semantic_p,
        program.to_bytes(),
        expected_semantic_archive_sha256=SEMANTIC_P_SHA256,
        expected_program_sha256=program.sha256,
        verify_member_effects=False,
    )
    with pytest.raises(ClassCompleteSemanticError, match="batch <=16"):
        receiver.decode(tuple(range(17)))
